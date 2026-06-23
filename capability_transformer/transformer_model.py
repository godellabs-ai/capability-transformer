"""A minimal, analytically-weighted transformer-style evaluator (NumPy).

This module defines the execution machinery — a residual stream, attention heads with
explicit Q/K projection matrices, feed-forward Boolean gates, attention max-pool
aggregation, and an output projection — together with a deterministic forward pass. The
weights are *constructed analytically* by ``compiler.py`` from the policy IR; nothing here
is trained and there is no stochastic behavior.

Design contract (security-critical):

* Evidence is computed **per token**. Per-capability predicates live in each capability
  token's residual slots; the per-capability conjunction (``valid_capability``) is computed
  before any aggregation. The existential "∃ a valid capability" is a hard attention
  max-pool over capability tokens — it can never combine ``subject_match`` from one
  capability with ``object_match`` from another.
* Attention is deterministic: each query attends to a structurally fixed key (the request
  token, the policy token, itself, or — for aggregation — a token set), selected by
  hardmax. No softmax is used for any security decision.
* A match head emits the **attention score** (the Q·K compatibility) as its evidence bit:
  the inner product of two one-hot/mask fields is exactly the match predicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------------------
# Residual stream layout
# --------------------------------------------------------------------------------------
class ResidualLayout:
    """Named slots in the residual stream. Vector slots have width > 1; evidence bits 1."""

    def __init__(self) -> None:
        self._slots: dict[str, tuple[int, int]] = {}
        self._width = 0

    def add(self, name: str, width: int = 1) -> None:
        if name in self._slots:
            return
        self._slots[name] = (self._width, width)
        self._width += width

    def slice(self, name: str) -> slice:
        off, w = self._slots[name]
        return slice(off, off + w)

    def index(self, name: str) -> int:
        off, w = self._slots[name]
        assert w == 1, f"{name} is not a scalar slot"
        return off

    def selector(self, name: str) -> np.ndarray:
        """A (width × D) matrix that extracts slot ``name`` from a residual vector."""
        off, w = self._slots[name]
        M = np.zeros((w, self.width), dtype=np.float64)
        for i in range(w):
            M[i, off + i] = 1.0
        return M

    @property
    def width(self) -> int:
        return self._width

    @property
    def names(self) -> list[str]:
        return list(self._slots)

    def shape_of(self, name: str) -> tuple[int, int]:
        return self._slots[name]


# --------------------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------------------
@dataclass
class MatchHead:
    """An exact-match attention head: evidence = threshold( (Wq·r_query) · (Wk·r_key) )."""

    name: str
    Wq: np.ndarray              # (d × D) query projection
    Wk: np.ndarray              # (d × D) key projection
    query_set: str              # "request" | "capabilities" | "confirmations"
    key_token: str              # "request" | "policy" | "self"
    out_slot: str
    threshold: float = 0.5

    def apply(self, R: np.ndarray, idx: dict) -> None:
        queries = _token_indices(idx, self.query_set)
        out = idx["layout"].index(self.out_slot)
        Sq = R @ self.Wq.T                          # (N, d)
        Sk = R @ self.Wk.T                          # (N, d)
        for q in queries:
            k = q if self.key_token == "self" else idx[self.key_token]
            score = float(Sq[q] @ Sk[k])            # the Q·K compatibility
            R[q, out] = 1.0 if score >= self.threshold else 0.0


@dataclass
class PoolHead:
    """A hard-attention max-pool: out = max over a token set of an input slot (∃ for bits)."""

    name: str
    over: str                   # "capabilities" | "confirmations" | "request"
    in_slot: str
    out_slot: str
    Wk: np.ndarray              # (1 × D) value selector (reads in_slot); for inspection

    def apply(self, R: np.ndarray, idx: dict) -> None:
        members = _token_indices(idx, self.over)
        out = idx["layout"].index(self.out_slot)
        o = idx["output"]
        if not members:
            R[o, out] = 0.0
            return
        vals = (R[members] @ self.Wk.T).ravel()     # value per member token
        # hardmax: pick the max value; deterministic tie-break = lowest index (argmax does this)
        R[o, out] = float(vals[int(np.argmax(vals))])


@dataclass
class BoolGate:
    """A feed-forward Boolean gate: y = W2 · ReLU(W1·r + b1) + b2, written to out_slot.

    AND/OR/NOT over {0,1} inputs are realised analytically:
      AND(x1..xk) = ReLU(Σx − (k−1))
      OR (x1..xk) = 1 − ReLU(1 − Σx)
      NOT(x)      = 1 − x
    """

    name: str
    op: str                     # "and" | "or" | "not"
    inputs: tuple[str, ...]
    out_slot: str
    token_set: str              # "capabilities" | "confirmations" | "output" | "request"
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    def apply(self, R: np.ndarray, idx: dict) -> None:
        tokens = _token_indices(idx, self.token_set)
        out = idx["layout"].index(self.out_slot)
        for t in tokens:
            h = np.maximum(0.0, self.W1 @ R[t] + self.b1)   # ReLU
            y = float(np.ravel(self.W2 @ h + self.b2)[0])
            R[t, out] = y


@dataclass
class OutputProjection:
    """Projects the output token's [allow, deny, escalate] evidence to class logits."""

    W_out: np.ndarray           # (3 × D)
    classes: tuple[str, str, str] = ("ALLOW", "DENY", "ESCALATE")
    margin: float = 10.0

    def logits(self, R: np.ndarray, idx: dict) -> np.ndarray:
        return self.margin * (self.W_out @ R[idx["output"]])

    def decide(self, logits: np.ndarray) -> str:
        return self.classes[int(np.argmax(logits))]


@dataclass
class CompiledModel:
    """The fully compiled transformer-style evaluator (analytic weights)."""

    layout: ResidualLayout
    match_heads: list[MatchHead] = field(default_factory=list)
    cap_gates: list[BoolGate] = field(default_factory=list)       # per-capability / per-conf
    pool_heads: list[PoolHead] = field(default_factory=list)      # ∃ aggregation
    decision_gates: list[BoolGate] = field(default_factory=list)  # on the output token
    output_projection: OutputProjection | None = None
    config: dict = field(default_factory=dict)


@dataclass
class ForwardResult:
    decision: str
    logits: np.ndarray
    residual: np.ndarray
    idx: dict
    evidence: dict


def _token_indices(idx: dict, which: str) -> list[int]:
    if which == "request":
        return [idx["request"]]
    if which == "output":
        return [idx["output"]]
    if which == "policy":
        return [idx["policy"]]
    if which == "capabilities":
        return list(idx["capabilities"])
    if which == "confirmations":
        return list(idx["confirmations"])
    raise ValueError(which)


# --------------------------------------------------------------------------------------
# Forward pass — the layer schedule
# --------------------------------------------------------------------------------------
def forward(model: CompiledModel, R: np.ndarray, idx: dict) -> ForwardResult:
    """Run the compiled transformer-style evaluator. Mutates a copy of ``R``."""
    R = R.copy()
    idx = {**idx, "layout": model.layout}

    # Layer 1 — attention: per-token match evidence (capability/request/confirmation).
    for head in model.match_heads:
        head.apply(R, idx)

    # Layer 2 — feed-forward: per-capability / per-confirmation conjunctions.
    for gate in model.cap_gates:
        gate.apply(R, idx)

    # Layer 3 — attention max-pool: existential aggregation onto the output token.
    for head in model.pool_heads:
        head.apply(R, idx)

    # Layer 4 — feed-forward: decision gates on the output token.
    for gate in model.decision_gates:
        gate.apply(R, idx)

    # Layer 5 — output projection: [allow, deny, escalate] evidence -> class logits.
    logits = model.output_projection.logits(R, idx)
    decision = model.output_projection.decide(logits)

    evidence = _collect_evidence(model, R, idx)
    return ForwardResult(decision=decision, logits=logits, residual=R, idx=idx,
                         evidence=evidence)


def _collect_evidence(model: CompiledModel, R: np.ndarray, idx: dict) -> dict:
    """Snapshot the output-token decision evidence and per-capability validity."""
    L = model.layout
    o = idx["output"]
    out_bits = [
        "has_match", "prov_ok", "scope_ok", "delegation_ok", "required_ok",
        "high_risk_action", "confirmed", "is_delegation",
        "allow_evidence", "deny_evidence", "escalate_evidence",
    ]
    ev = {name: float(R[o, L.index(name)]) for name in out_bits if name in L.names}
    if idx["capabilities"]:
        ev["valid_capability_per_cap"] = [
            float(R[c, L.index("valid_capability")]) for c in idx["capabilities"]
        ]
    return ev
