"""TorchCapabilityHead — the compiled capability evaluator as a frozen torch module.

This is the deterministic capability transformer (``compiler.compile_policy``) ported to
``torch`` with the SAME analytically-constructed weights. Every tensor is a frozen buffer
(``requires_grad=False``); there is no training. It exists so the capability decision can
run as a real torch sub-module fused into a larger model's forward pass (see
``guarded_qwen.py``), while remaining bit-for-bit equivalent to the NumPy reference
(checked in ``tests/test_torch_head.py``).

The head reads ONLY the structured request (subject, action, object, provenance) and the
possessed capability tokens — never any language-model hidden state. That isolation is the
security boundary: an attacker cannot phrase their way past a frozen, exact-match head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import compiler, tokenizer
from .transformer_model import _token_indices


@dataclass
class HeadDecision:
    decision: str
    reasons: list
    evidence: dict


_REASON_BY_FAIL = {
    "subject_match": "subject_mismatch", "object_match": "object_mismatch",
    "right_match": "right_not_granted", "issuer_trusted": "issuer_not_trusted",
}


class TorchCapabilityHead(nn.Module):
    """The compiled object-capability evaluator as a frozen torch module."""

    def __init__(self, *, require_signatures: bool = False,
                 require_bound_confirmations: bool = False):
        super().__init__()
        self.require_signatures = require_signatures
        self.require_bound_confirmations = require_bound_confirmations
        self._np = compiler.compile_policy({
            "require_signatures": require_signatures,
            "require_bound_confirmations": require_bound_confirmations})
        self.layout = self._np.layout

        # Register every analytic matrix as a frozen buffer (shows up in state_dict, no grad).
        self._match, self._capg, self._decg, self._pool = [], [], [], []
        for i, h in enumerate(self._np.match_heads):
            self._buf(f"mh{i}_Wq", h.Wq); self._buf(f"mh{i}_Wk", h.Wk)
            self._match.append((f"mh{i}", h))
        for i, g in enumerate(self._np.cap_gates):
            self._gate_buf(f"cg{i}", g); self._capg.append((f"cg{i}", g))
        for i, g in enumerate(self._np.decision_gates):
            self._gate_buf(f"dg{i}", g); self._decg.append((f"dg{i}", g))
        for i, p in enumerate(self._np.pool_heads):
            self._pool.append((p,))
        self._buf("Wout", self._np.output_projection.W_out)
        self.margin = self._np.output_projection.margin
        for b in self.buffers():
            b.requires_grad_(False)

    def _buf(self, name, arr):
        self.register_buffer(name, torch.tensor(np.asarray(arr), dtype=torch.float32))

    def _gate_buf(self, pre, g):
        self._buf(f"{pre}_W1", g.W1); self._buf(f"{pre}_b1", g.b1)
        self._buf(f"{pre}_W2", g.W2); self._buf(f"{pre}_b2", g.b2)

    # ----------------------------------------------------------------------------------
    def forward(self, R: torch.Tensor, idx: dict) -> dict:
        """Run the compiled forward pass in torch. Mirrors transformer_model.forward."""
        R = R.clone()
        L = self.layout
        col = L.index

        # Layer 1 — exact-match attention heads (evidence = thresholded Q·K).
        for pre, h in self._match:
            Wq = getattr(self, f"{pre}_Wq"); Wk = getattr(self, f"{pre}_Wk")
            Sq, Sk = R @ Wq.T, R @ Wk.T
            out = col(h.out_slot)
            for q in _token_indices(idx, h.query_set):
                k = q if h.key_token == "self" else idx[h.key_token]
                R[q, out] = 1.0 if float(Sq[q] @ Sk[k]) >= h.threshold else 0.0

        # Layer 2 — per-capability / per-confirmation Boolean gates.
        self._run_gates(R, idx, self._capg)
        # Layer 3 — existential max-pool onto the output token.
        for (p,) in self._pool:
            members = _token_indices(idx, p.over)
            o, out = idx["output"], col(p.out_slot)
            R[o, out] = (R[torch.tensor(members), col(p.in_slot)].max() if members
                         else torch.tensor(0.0))
        # Layer 4 — decision gates on the output token.
        self._run_gates(R, idx, self._decg)

        # Layer 5 — output projection → argmax (hardmax, large margin).
        logits = self.margin * (self.Wout @ R[idx["output"]])
        classes = ["ALLOW", "DENY", "ESCALATE"]
        decision = classes[int(torch.argmax(logits))]
        return {"decision": decision, "logits": logits, "residual": R}

    def _run_gates(self, R, idx, gates):
        L = self.layout
        for pre, g in gates:
            W1 = getattr(self, f"{pre}_W1"); b1 = getattr(self, f"{pre}_b1")
            W2 = getattr(self, f"{pre}_W2"); b2 = getattr(self, f"{pre}_b2")
            out = L.index(g.out_slot)
            for t in _token_indices(idx, g.token_set):
                h = torch.relu(W1 @ R[t] + b1)
                R[t, out] = (W2 @ h + b2).reshape(-1)[0]

    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def decide(self, bundle) -> HeadDecision:
        enc = tokenizer.encode(bundle, require_signatures=self.require_signatures,
                               require_bound_confirmations=self.require_bound_confirmations)
        R_np, idx = compiler.build_residual(self._np, enc, bundle)
        idx = {**idx, "output": idx["output"]}
        out = self.forward(torch.tensor(R_np, dtype=torch.float32), idx)
        R, L = out["residual"], self.layout
        o = idx["output"]
        ev = {n: float(R[o, L.index(n)]) for n in
              ["has_match", "prov_ok", "scope_ok", "delegation_ok", "required_ok",
               "o_high_risk", "confirmed", "allow_evidence", "deny_evidence", "escalate_evidence"]}
        reasons = self._reasons(out["decision"], ev, R, idx)
        return HeadDecision(decision=out["decision"], reasons=reasons, evidence=ev)

    def _reasons(self, decision, ev, R, idx):
        if decision == "ALLOW":
            return ["allowed"]
        if decision == "ESCALATE":
            return ["confirmation_required"]
        L, caps = self.layout, idx["capabilities"]
        reasons = []
        if ev["has_match"] <= 0.5:
            if not caps:
                reasons.append("missing_capability")
            else:
                for slot, code in _REASON_BY_FAIL.items():
                    if not any(R[c, L.index(slot)] > 0.5 for c in caps):
                        reasons.append(code)
                if not reasons:
                    reasons.append("missing_capability")
        if ev["prov_ok"] <= 0.5:
            reasons.append("data_has_no_authority")
        return reasons or ["missing_capability"]
