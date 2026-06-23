"""Compile the policy IR into deterministic transformer-style weights.

Every matrix here is constructed analytically from the bounded vocabularies and the fixed
policy masks in ``compiled_weights``. There is no training and no data dependence in the
weights — only in the per-request residual produced by :func:`build_residual`.

The layout, attention heads, feed-forward gates, and output projection together implement
exactly the reference decision function (see ``ir.py`` and ``transformer_model.py``).
"""

from __future__ import annotations

import numpy as np

from . import compiled_weights as W
from .transformer_model import (
    BoolGate,
    CompiledModel,
    MatchHead,
    OutputProjection,
    PoolHead,
    ResidualLayout,
)

# Feature region = a verbatim copy of the 48-dim token vector at residual offset 0.
FEATURE_WIDTH = W.D

# Scalar evidence slots, in the order they are written.
_CAP_EVIDENCE = [
    "subject_match", "object_match", "right_match", "issuer_trusted",
    "has_delegate", "has_target_right",
    "not_revoked", "not_delegated", "chain_ok", "atten_ok",
    "valid_capability", "delegator_capability", "scope_violation",
]
_REQUEST_EVIDENCE = [
    "prov_trusted", "action_is_read", "high_risk_action", "action_is_delegate",
]
_CONF_EVIDENCE = ["conf_subject", "conf_object", "conf_action", "conf_issuer", "conf_valid"]
_OUTPUT_EVIDENCE = [
    "has_match", "delegation_pass", "scope_violated", "confirmed",
    "o_prov_trusted", "o_action_is_read", "o_high_risk", "o_action_is_delegate",
    "o_delegate_present",
    "prov_ok", "scope_ok", "is_delegation", "not_is_delegation", "delegation_ok",
    "required_ok", "not_confirmed", "not_high_risk", "low_or_confirmed",
    "allow_evidence", "deny_evidence", "escalate_evidence",
]


def build_layout() -> ResidualLayout:
    L = ResidualLayout()
    L.add("token_features", FEATURE_WIDTH)        # [0:48] verbatim token vector
    # Policy token vectors (fixed masks).
    L.add("pol_trusted_issuer", W.N_ISSUER)
    L.add("pol_trusted_prov", W.N_PROV)
    L.add("pol_read", W.N_RIGHT)
    L.add("pol_delegate", W.N_RIGHT)
    # Request extras.
    L.add("req_delegate_right", W.N_RIGHT)
    L.add("req_delegate_present", 1)
    # Capability extras.
    L.add("cap_delegated", 1)
    L.add("cap_scope_nonempty", 1)
    L.add("cap_scope_mismatch", 1)
    # Evidence slots.
    for name in _CAP_EVIDENCE + _REQUEST_EVIDENCE + _CONF_EVIDENCE + _OUTPUT_EVIDENCE:
        L.add(name, 1)
    return L


# ---- selector helpers ----------------------------------------------------------------
def _feature_selector(L: ResidualLayout, slot: str) -> np.ndarray:
    """A (w × D) matrix extracting one of the token's one-hot fields from [0:48]."""
    start, stop = W.SLOT[slot]
    M = np.zeros((stop - start, L.width), dtype=np.float64)
    for i in range(stop - start):
        M[i, start + i] = 1.0
    return M


def _coord(L: ResidualLayout, name: str) -> int:
    """Residual coordinate for a scalar slot or a named feature bit (``feat:<OFF>``)."""
    if name.startswith("feat:"):
        return getattr(W, name.split(":", 1)[1])
    return L.index(name)


# ---- Boolean feed-forward gate constructors ------------------------------------------
def _and_gate(L, name, inputs, out_slot, token_set) -> BoolGate:
    k = len(inputs)
    W1 = np.zeros((1, L.width));
    for s in inputs:
        W1[0, _coord(L, s)] = 1.0
    b1 = np.array([-(k - 1)], dtype=np.float64)
    W2 = np.array([[1.0]]); b2 = np.array([0.0])
    return BoolGate(name, "and", tuple(inputs), out_slot, token_set, W1, b1, W2, b2)


def _or_gate(L, name, inputs, out_slot, token_set) -> BoolGate:
    W1 = np.zeros((1, L.width))
    for s in inputs:
        W1[0, _coord(L, s)] = -1.0
    b1 = np.array([1.0]); W2 = np.array([[-1.0]]); b2 = np.array([1.0])
    return BoolGate(name, "or", tuple(inputs), out_slot, token_set, W1, b1, W2, b2)


def _not_gate(L, name, src, out_slot, token_set) -> BoolGate:
    W1 = np.zeros((1, L.width)); W1[0, _coord(L, src)] = 1.0
    b1 = np.array([0.0]); W2 = np.array([[-1.0]]); b2 = np.array([1.0])
    return BoolGate(name, "not", (src,), out_slot, token_set, W1, b1, W2, b2)


def _value_selector(L: ResidualLayout, slot: str) -> np.ndarray:
    M = np.zeros((1, L.width)); M[0, L.index(slot)] = 1.0
    return M


def compile_policy(config: dict | None = None) -> CompiledModel:
    """Construct the compiled transformer-style evaluator (analytic weights)."""
    config = dict(config or {})
    L = build_layout()
    sub = _feature_selector(L, "subject")
    obj = _feature_selector(L, "object")
    rights = _feature_selector(L, "rights")
    issuer = _feature_selector(L, "issuer")
    prov = _feature_selector(L, "provenance")

    def pol(slot):
        M = np.zeros((L.shape_of(slot)[1], L.width))
        s = L.slice(slot)
        for i in range(L.shape_of(slot)[1]):
            M[i, s.start + i] = 1.0
        return M

    # --- Layer 1: match heads ----------------------------------------------------------
    match_heads = [
        MatchHead("subject_match", sub, sub, "capabilities", "request", "subject_match"),
        MatchHead("object_match", obj, obj, "capabilities", "request", "object_match"),
        MatchHead("right_match", rights, rights, "capabilities", "request", "right_match"),
        MatchHead("issuer_trusted", issuer, pol("pol_trusted_issuer"),
                  "capabilities", "policy", "issuer_trusted"),
        MatchHead("has_delegate", rights, pol("pol_delegate"),
                  "capabilities", "policy", "has_delegate"),
        MatchHead("has_target_right", rights, pol("req_delegate_right"),
                  "capabilities", "request", "has_target_right"),
        # request-level
        MatchHead("prov_trusted", prov, pol("pol_trusted_prov"),
                  "request", "policy", "prov_trusted"),
        MatchHead("action_is_read", rights, pol("pol_read"),
                  "request", "policy", "action_is_read"),
        MatchHead("high_risk_action", W.HIGH_RISK.T @ obj, rights,
                  "request", "self", "high_risk_action"),
        MatchHead("action_is_delegate", rights, pol("pol_delegate"),
                  "request", "policy", "action_is_delegate"),
        # confirmation-level
        MatchHead("conf_subject", sub, sub, "confirmations", "request", "conf_subject"),
        MatchHead("conf_object", obj, obj, "confirmations", "request", "conf_object"),
        MatchHead("conf_action", rights, rights, "confirmations", "request", "conf_action"),
        MatchHead("conf_issuer", issuer, pol("pol_trusted_issuer"),
                  "confirmations", "policy", "conf_issuer"),
    ]

    # --- Layer 2: per-capability / per-confirmation feed-forward gates ------------------
    cap_gates = [
        _not_gate(L, "not_revoked", "feat:REVOKED_OFF", "not_revoked", "capabilities"),
        _not_gate(L, "not_delegated", "cap_delegated", "not_delegated", "capabilities"),
        _or_gate(L, "chain_ok", ["feat:CHAIN_OFF", "not_delegated"], "chain_ok", "capabilities"),
        _or_gate(L, "atten_ok", ["feat:ATTEN_OFF", "not_delegated"], "atten_ok", "capabilities"),
        _and_gate(L, "valid_capability",
                  ["subject_match", "object_match", "right_match", "issuer_trusted",
                   "feat:EXPIRY_OFF", "not_revoked", "feat:SIG_OFF", "chain_ok", "atten_ok"],
                  "valid_capability", "capabilities"),
        _and_gate(L, "delegator_capability",
                  ["subject_match", "object_match", "issuer_trusted", "feat:EXPIRY_OFF",
                   "not_revoked", "has_delegate", "has_target_right"],
                  "delegator_capability", "capabilities"),
        _and_gate(L, "scope_violation",
                  ["valid_capability", "cap_scope_nonempty", "cap_scope_mismatch"],
                  "scope_violation", "capabilities"),
        _and_gate(L, "conf_valid",
                  ["conf_subject", "conf_object", "conf_action", "conf_issuer", "feat:CBIND_OFF"],
                  "conf_valid", "confirmations"),
    ]

    # --- Layer 3: existential aggregation (hard attention max-pool to output) -----------
    def pool(name, over, in_slot, out_slot):
        return PoolHead(name, over, in_slot, out_slot, _value_selector(L, in_slot))

    pool_heads = [
        pool("has_match", "capabilities", "valid_capability", "has_match"),
        pool("delegation_pass", "capabilities", "delegator_capability", "delegation_pass"),
        pool("scope_violated", "capabilities", "scope_violation", "scope_violated"),
        pool("confirmed", "confirmations", "conf_valid", "confirmed"),
        pool("o_prov_trusted", "request", "prov_trusted", "o_prov_trusted"),
        pool("o_action_is_read", "request", "action_is_read", "o_action_is_read"),
        pool("o_high_risk", "request", "high_risk_action", "o_high_risk"),
        pool("o_action_is_delegate", "request", "action_is_delegate", "o_action_is_delegate"),
        pool("o_delegate_present", "request", "req_delegate_present", "o_delegate_present"),
    ]

    # --- Layer 4: decision gates on the output token -----------------------------------
    decision_gates = [
        _or_gate(L, "prov_ok", ["o_prov_trusted", "o_action_is_read"], "prov_ok", "output"),
        _not_gate(L, "scope_ok", "scope_violated", "scope_ok", "output"),
        _and_gate(L, "is_delegation", ["o_action_is_delegate", "o_delegate_present"],
                  "is_delegation", "output"),
        _not_gate(L, "not_is_delegation", "is_delegation", "not_is_delegation", "output"),
        _or_gate(L, "delegation_ok", ["not_is_delegation", "delegation_pass"],
                 "delegation_ok", "output"),
        _and_gate(L, "required_ok", ["has_match", "prov_ok", "scope_ok", "delegation_ok"],
                  "required_ok", "output"),
        _not_gate(L, "deny_evidence", "required_ok", "deny_evidence", "output"),
        _not_gate(L, "not_confirmed", "confirmed", "not_confirmed", "output"),
        _not_gate(L, "not_high_risk", "o_high_risk", "not_high_risk", "output"),
        _and_gate(L, "escalate_evidence", ["required_ok", "o_high_risk", "not_confirmed"],
                  "escalate_evidence", "output"),
        _or_gate(L, "low_or_confirmed", ["not_high_risk", "confirmed"], "low_or_confirmed", "output"),
        _and_gate(L, "allow_evidence", ["required_ok", "low_or_confirmed"],
                  "allow_evidence", "output"),
    ]

    # --- Layer 5: output projection ----------------------------------------------------
    W_out = np.zeros((3, L.width))
    W_out[0, L.index("allow_evidence")] = 1.0
    W_out[1, L.index("deny_evidence")] = 1.0
    W_out[2, L.index("escalate_evidence")] = 1.0
    output_projection = OutputProjection(W_out, ("ALLOW", "DENY", "ESCALATE"), margin=10.0)

    return CompiledModel(
        layout=L,
        match_heads=match_heads,
        cap_gates=cap_gates,
        pool_heads=pool_heads,
        decision_gates=decision_gates,
        output_projection=output_projection,
        config=config,
    )


# --------------------------------------------------------------------------------------
# Embedding: an EncodedBundle (shared front-end) -> the transformer residual stream.
# --------------------------------------------------------------------------------------
def _scope_bits(cap_scope: dict, req_scope: dict) -> tuple[float, float]:
    nonempty = 1.0 if cap_scope else 0.0
    mismatch = 1.0 if (cap_scope and not all(req_scope.get(k) == v for k, v in cap_scope.items())) else 0.0
    return nonempty, mismatch


def build_residual(model: CompiledModel, enc, bundle) -> tuple[np.ndarray, dict]:
    """Lay out [request, policy, caps..., confs..., output] into the residual stream."""
    L = model.layout
    n_caps = len(enc.cap_indices)
    n_conf = len(enc.conf_indices)
    n_tokens = 2 + n_caps + n_conf + 1
    R = np.zeros((n_tokens, L.width), dtype=np.float64)

    request_i, policy_i = 0, 1
    cap_is = list(range(2, 2 + n_caps))
    conf_is = list(range(2 + n_caps, 2 + n_caps + n_conf))
    output_i = n_tokens - 1
    feat = L.slice("token_features")

    # request token
    R[request_i, feat] = enc.X[enc.request_index]
    dr = bundle.delegate_right
    if dr is not None:
        R[request_i, L.slice("req_delegate_right")] = W.one_hot(W.RIGHT_IDX, dr, W.N_RIGHT)
        R[request_i, L.index("req_delegate_present")] = 1.0

    # policy token (fixed masks)
    R[policy_i, L.slice("pol_trusted_issuer")] = W.TRUSTED_ISSUER_MASK
    R[policy_i, L.slice("pol_trusted_prov")] = W.TRUSTED_PROV_MASK
    R[policy_i, L.slice("pol_read")] = W.NON_SIDE_EFFECT_MASK
    R[policy_i, L.slice("pol_delegate")] = W.one_hot(W.RIGHT_IDX, "delegate", W.N_RIGHT)

    # capability tokens
    req_scope = bundle.scope or {}
    for slot, ci in enumerate(cap_is):
        R[ci, feat] = enc.X[enc.cap_indices[slot]]
        R[ci, L.index("cap_delegated")] = 1.0 if enc.cap_delegated[slot] else 0.0
        nonempty, mismatch = _scope_bits(enc.cap_scopes[slot], req_scope)
        R[ci, L.index("cap_scope_nonempty")] = nonempty
        R[ci, L.index("cap_scope_mismatch")] = mismatch

    # confirmation tokens
    for slot, cj in enumerate(conf_is):
        R[cj, feat] = enc.X[enc.conf_indices[slot]]

    idx = {
        "request": request_i, "policy": policy_i, "output": output_i,
        "capabilities": cap_is, "confirmations": conf_is,
    }
    return R, idx
