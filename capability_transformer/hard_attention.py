"""Hard-attention heads — the enforcement boundary.

Every head is a pure tensor expression over the token matrix ``X``. There is **no
softmax**: each head produces a Boolean mask via exact-match attention
``mask = (Keys @ query) >= 1`` or an explicit bit/equality test. The conjunction of the
six matching heads across capabilities is the object-capability security boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import compiled_weights as W
from .tokenizer import EncodedBundle


@dataclass
class HeadResult:
    """The result of one hard-attention head."""

    name: str
    passed: bool
    per_cap_mask: np.ndarray            # Boolean mask over capability tokens
    matched_indices: list[int] = field(default_factory=list)
    matched_cap_ids: list[str] = field(default_factory=list)
    reason: str | None = None
    relevant: bool = True               # whether this head gates the current decision


@dataclass
class AttentionResult:
    """Everything the reducer needs, all derived from tensor operations."""

    heads: dict[str, HeadResult]
    matched_mask: np.ndarray            # caps passing the AND of the six matching heads
    matched_cap_ids: list[str]
    high_risk: bool
    has_capabilities: bool


def _bool(arr) -> np.ndarray:
    """Hard threshold of a (one-hot dot-product) score vector into Booleans."""
    return np.asarray(arr, dtype=np.float64) > 0.5


def _slice(C: np.ndarray, slot: str) -> np.ndarray:
    start, stop = W.SLOT[slot]
    return C[:, start:stop]


def compute(enc: EncodedBundle) -> AttentionResult:
    """Run all hard-attention heads over the encoded bundle."""
    X = enc.X
    q = X[enc.request_index]                       # request (query) token vector

    # Query field directions (one-hot) sliced out of the request token.
    q_subj = q[slice(*W.SLOT["subject"])]
    q_obj = q[slice(*W.SLOT["object"])]
    q_action = q[slice(*W.SLOT["rights"])]
    q_prov = q[slice(*W.SLOT["provenance"])]

    # Capability key/value matrix (rows = possessed capabilities).
    if enc.cap_indices:
        C = X[enc.cap_indices]
    else:
        C = np.zeros((0, W.D), dtype=np.float64)
    n_caps = C.shape[0]
    cap_ids = enc.cap_ids

    # ---- Heads 1-6: exact-match attention over capabilities --------------------------
    # head_subject_match: attend to caps whose subject equals the request subject.
    subj_mask = _bool(_slice(C, "subject") @ q_subj) if n_caps else np.zeros(0, bool)
    # head_object_match: attend to caps for the requested object.
    obj_mask = _bool(_slice(C, "object") @ q_obj) if n_caps else np.zeros(0, bool)
    # head_right_match: attend to caps whose rights multi-hot contains the action.
    right_mask = _bool(_slice(C, "rights") @ q_action) if n_caps else np.zeros(0, bool)
    # head_trusted_issuer: attend to caps minted by a trusted issuer.
    issuer_mask = _bool(_slice(C, "issuer") @ W.TRUSTED_ISSUER_MASK) if n_caps else np.zeros(0, bool)
    # head_not_expired: the compiled expiry bit must be set.
    expiry_mask = _bool(C[:, W.EXPIRY_OFF]) if n_caps else np.zeros(0, bool)
    # head_not_revoked: the compiled revoked bit must be clear.
    revoked_bit = _bool(C[:, W.REVOKED_OFF]) if n_caps else np.zeros(0, bool)
    not_revoked_mask = ~revoked_bit if n_caps else np.zeros(0, bool)

    matching = {
        "head_subject_match": subj_mask,
        "head_object_match": obj_mask,
        "head_right_match": right_mask,
        "head_trusted_issuer": issuer_mask,
        "head_not_expired": expiry_mask,
        "head_not_revoked": not_revoked_mask,
    }

    heads: dict[str, HeadResult] = {}
    for name, mask in matching.items():
        ids = [cap_ids[i] for i in np.nonzero(mask)[0]] if n_caps else []
        passed = bool(mask.any())
        heads[name] = HeadResult(
            name=name,
            passed=passed,
            per_cap_mask=mask,
            matched_indices=[enc.cap_indices[i] for i in np.nonzero(mask)[0]] if n_caps else [],
            matched_cap_ids=ids,
            reason=None if passed else W.HEAD_REASON[name],
        )

    # Security boundary: element-wise AND across the six heads, OR across capabilities.
    require = enc.require_signatures
    if n_caps:
        core_mask = subj_mask & obj_mask & right_mask & issuer_mask & expiry_mask & not_revoked_mask
        sig_mask = _bool(C[:, W.SIG_OFF])           # Phase 8a: signature valid
        chain_mask = _bool(C[:, W.CHAIN_OFF])       # Phase 8b: delegation chain valid
        atten_mask = _bool(C[:, W.ATTEN_OFF])       # Phase 8b: attenuation valid
        del_mask = np.asarray(enc.cap_delegated, dtype=bool)
        # Chain/attenuation only constrain delegated children; roots pass trivially.
        chain_ok = chain_mask | ~del_mask
        atten_ok = atten_mask | ~del_mask
    else:
        core_mask = sig_mask = chain_mask = atten_mask = np.zeros(0, bool)
        del_mask = chain_ok = atten_ok = np.zeros(0, bool)

    # When signatures are enforced, a capability is a valid match only if its crypto bits
    # also hold: valid signature, valid chain (if delegated), valid attenuation.
    if require and n_caps:
        matched_mask = core_mask & sig_mask & chain_ok & atten_ok
    else:
        matched_mask = core_mask
    matched_cap_ids = [cap_ids[i] for i in np.nonzero(matched_mask)[0]] if n_caps else []
    has_match = bool(matched_mask.any()) if n_caps else False

    # ---- Crypto heads (Phase 8a/8b) --------------------------------------------------
    # A crypto head fails when NO capability fully matched AND some capability that does
    # match the six core fields is blocked by this bit. Several crypto checks can fail at
    # once (e.g. a tampered child breaks both signature and attenuation) — we report all.
    def _crypto_head(name, own_mask, applies_mask, relevant):
        if n_caps:
            blocked = core_mask & applies_mask & ~own_mask
            passed = (not relevant) or has_match or (not bool(blocked.any()))
            ids = [cap_ids[i] for i in np.nonzero(matched_mask & own_mask)[0]]
        else:
            passed, ids = True, []
        return HeadResult(
            name=name,
            passed=passed,
            per_cap_mask=own_mask if n_caps else np.zeros(0, bool),
            matched_cap_ids=ids,
            reason=None if passed else W.HEAD_REASON[name],
            relevant=relevant,
        )

    all_true = np.ones(n_caps, bool) if n_caps else np.zeros(0, bool)
    has_delegated = bool(del_mask.any()) if n_caps else False
    heads["head_signature_valid"] = _crypto_head(
        "head_signature_valid", sig_mask, all_true, require)
    heads["head_chain_valid"] = _crypto_head(
        "head_chain_valid", chain_mask, del_mask, require and has_delegated)
    heads["head_attenuation_valid"] = _crypto_head(
        "head_attenuation_valid", atten_mask, del_mask, require and has_delegated)

    # ---- Head 7: provenance-safe -----------------------------------------------------
    # Untrusted data may drive a passive read but never a side effect.
    prov_trusted = bool(_bool(q_prov @ W.TRUSTED_PROV_MASK))
    action_is_passive = bool(_bool(q_action @ W.NON_SIDE_EFFECT_MASK))
    prov_safe = prov_trusted or action_is_passive
    heads["head_provenance_safe"] = HeadResult(
        name="head_provenance_safe",
        passed=prov_safe,
        per_cap_mask=matched_mask,
        matched_cap_ids=matched_cap_ids,
        reason=None if prov_safe else W.HEAD_REASON["head_provenance_safe"],
    )

    # ---- Head 8: confirmation (only relevant for high-risk actions) ------------------
    high_risk = bool(_bool(q_obj @ (W.HIGH_RISK @ q_action)))
    confirmed = False
    if enc.conf_indices:
        Cf = X[enc.conf_indices]
        cf_subj = _bool(_slice(Cf, "subject") @ q_subj)
        cf_obj = _bool(_slice(Cf, "object") @ q_obj)
        cf_action = _bool(_slice(Cf, "rights") @ q_action)
        cf_issuer = _bool(_slice(Cf, "issuer") @ W.TRUSTED_ISSUER_MASK)
        confirmed = bool((cf_subj & cf_obj & cf_action & cf_issuer).any())
    conf_passed = (not high_risk) or confirmed
    heads["head_confirmation"] = HeadResult(
        name="head_confirmation",
        passed=conf_passed,
        per_cap_mask=matched_mask,
        reason=None if conf_passed else W.HEAD_REASON["head_confirmation"],
        relevant=high_risk,
    )

    # ---- Head 9: scope (only relevant when a matched cap carries a scope) -------------
    scope_relevant = False
    scope_passed = True
    req_scope = enc.bundle.scope or {}
    for i in np.nonzero(matched_mask)[0] if n_caps else []:
        cap_scope = enc.cap_scopes[i]
        if cap_scope:
            scope_relevant = True
            if not all(req_scope.get(k) == v for k, v in cap_scope.items()):
                scope_passed = False
    heads["head_scope"] = HeadResult(
        name="head_scope",
        passed=scope_passed,
        per_cap_mask=matched_mask,
        reason=None if scope_passed else W.HEAD_REASON["head_scope"],
        relevant=scope_relevant,
    )

    # ---- Head 10: delegation (only relevant for a delegate request with a target) ----
    # A bare action == "delegate" with no `delegate_right` is just "does the subject hold
    # the delegate right?" and is governed by the normal right-match head. The richer
    # attenuation check only engages once a specific right is being granted.
    is_delegation_action = bool(_bool(q_action @ W.one_hot(W.RIGHT_IDX, "delegate", W.N_RIGHT)))
    dr = enc.bundle.delegate_right
    is_delegation = is_delegation_action and dr is not None
    delegation_passed = True
    if is_delegation:
        if n_caps:
            dr_vec = W.one_hot(W.RIGHT_IDX, dr, W.N_RIGHT)
            delegate_vec = W.one_hot(W.RIGHT_IDX, "delegate", W.N_RIGHT)
            has_delegate = _bool(_slice(C, "rights") @ delegate_vec)
            has_target = _bool(_slice(C, "rights") @ dr_vec)
            # A valid delegator: subject/object match, trusted, fresh, unrevoked,
            # and holds BOTH the delegate right and the target right (attenuation only).
            valid = subj_mask & obj_mask & issuer_mask & expiry_mask & not_revoked_mask
            delegation_passed = bool((valid & has_delegate & has_target).any())
        else:
            delegation_passed = False
    heads["head_delegation"] = HeadResult(
        name="head_delegation",
        passed=delegation_passed,
        per_cap_mask=matched_mask,
        reason=None if delegation_passed else W.HEAD_REASON["head_delegation"],
        relevant=is_delegation,
    )

    return AttentionResult(
        heads=heads,
        matched_mask=matched_mask,
        matched_cap_ids=matched_cap_ids,
        high_risk=high_risk,
        has_capabilities=n_caps > 0,
    )
