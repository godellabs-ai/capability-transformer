"""Tokenizer: structured bundle -> token matrix X (N x D).

Every fact (the request, each capability, each confirmation) becomes one fixed-width
token vector. The expiry and revoked Boolean bits are computed here so that the
hard-attention heads operate purely on tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from . import compiled_weights as W
from . import crypto, delegated_capability
from .schema import CapabilityBundle
from .util import aware as _aware
from .util import is_revoked as _is_revoked


@dataclass
class EncodedBundle:
    """The tokenized request: a token matrix plus row-type bookkeeping."""

    X: np.ndarray                       # (N, D) token matrix
    row_types: list[str]                # token type per row
    request_index: int                  # row index of the request (query) token
    cap_indices: list[int]              # row indices of capability tokens
    conf_indices: list[int]             # row indices of confirmation tokens
    cap_ids: list[str]                  # capability id per capability token
    cap_scopes: list[dict]              # scope dict per capability token
    bundle: CapabilityBundle            # original bundle (for scope/delegation helpers)
    require_signatures: bool = False    # whether the signature head gates the decision
    cap_delegated: list[bool] = field(default_factory=list)  # is each cap a child?
    cap_meta: list[dict] = field(default_factory=list)       # per-cap audit metadata


def _blank() -> np.ndarray:
    return np.zeros(W.D, dtype=np.float64)


def _set_slot(vec: np.ndarray, slot: str, one_hot: np.ndarray) -> None:
    start, stop = W.SLOT[slot]
    vec[start:stop] = one_hot


def encode(
    bundle: CapabilityBundle,
    *,
    keyring=None,
    require_signatures: bool = False,
    require_bound_confirmations: bool = False,
) -> EncodedBundle:
    """Convert a bundle into the (N x D) token matrix and index metadata.

    When ``require_signatures`` is set, each capability's HMAC signature is verified
    against ``keyring`` and reduced to the per-token signature-valid bit.
    """
    now = _aware(bundle.now) if bundle.now is not None else datetime.now(timezone.utc)

    rows: list[np.ndarray] = []
    row_types: list[str] = []

    # ---- request (query) token -------------------------------------------------------
    req = _blank()
    _set_slot(req, "type", W.one_hot(W.TYPE_IDX, "request", W.N_TYPE))
    _set_slot(req, "subject", W.one_hot(W.SUBJ_IDX, bundle.subject, W.N_SUBJ))
    _set_slot(req, "object", W.one_hot(W.OBJ_IDX, bundle.object, W.N_OBJ))
    # The request action lives in the RIGHTS slot — this is the right-match query.
    _set_slot(req, "rights", W.one_hot(W.RIGHT_IDX, bundle.action, W.N_RIGHT))
    _set_slot(req, "provenance", W.one_hot(W.PROV_IDX, bundle.source_provenance, W.N_PROV))
    request_index = 0
    rows.append(req)
    row_types.append("request")

    # ---- delegation / signature verification (Phase 8a/8b) ---------------------------
    # The whole chain is verified here (helper code), then collapsed to per-token bits so
    # the attention core only ever sees a deterministic tensor. In label-trust mode
    # (require_signatures=False) signatures and parent links are ignored.
    kr = keyring if keyring is not None else crypto.DEFAULT_KEYRING
    verdicts = (
        delegated_capability.verify_bundle(bundle, keyring=kr, now=now)
        if require_signatures
        else {}
    )

    # ---- capability (key/value) tokens -----------------------------------------------
    cap_indices: list[int] = []
    cap_ids: list[str] = []
    cap_scopes: list[dict] = []
    cap_delegated: list[bool] = []
    cap_meta: list[dict] = []
    for cap in bundle.capabilities:
        vec = _blank()
        _set_slot(vec, "type", W.one_hot(W.TYPE_IDX, "capability", W.N_TYPE))
        _set_slot(vec, "subject", W.one_hot(W.SUBJ_IDX, cap.subject, W.N_SUBJ))
        _set_slot(vec, "object", W.one_hot(W.OBJ_IDX, cap.object, W.N_OBJ))
        _set_slot(vec, "rights", W.multi_hot(W.RIGHT_IDX, cap.rights, W.N_RIGHT))
        _set_slot(vec, "issuer", W.one_hot(W.ISSUER_IDX, cap.issuer, W.N_ISSUER))
        vec[W.EXPIRY_OFF] = 1.0 if _aware(cap.expires_at) > now else 0.0
        vec[W.REVOKED_OFF] = 1.0 if _is_revoked(cap, bundle.revocations) else 0.0
        vec[W.DELEG_OFF] = 1.0 if cap.delegatable else 0.0

        is_delegated = cap.parent_id is not None
        if require_signatures:
            v = verdicts[cap.id]
            is_delegated = v.is_delegated
            vec[W.SIG_OFF] = 1.0 if v.sig_valid else 0.0
            vec[W.CHAIN_OFF] = 1.0 if v.chain_valid else 0.0
            vec[W.ATTEN_OFF] = 1.0 if v.atten_valid else 0.0
            meta = {
                "id": cap.id, "issuer": cap.issuer, "kid": cap.kid,
                "payload_sha256": crypto.capability_hash(cap),
                "signature_valid": v.sig_valid, "delegated": v.is_delegated,
                "parent_capability_id": v.parent_id, "parent_hash": v.parent_hash,
                "chain_valid": v.chain_valid, "attenuation_valid": v.atten_valid,
                "failed_restrictions": v.failed_restrictions, "chain_error": v.chain_error,
            }
        else:
            # Not enforcing: all crypto bits set; delegation links ignored.
            vec[W.SIG_OFF] = 1.0
            vec[W.CHAIN_OFF] = 1.0
            vec[W.ATTEN_OFF] = 1.0
            is_delegated = False
            meta = {"id": cap.id, "issuer": cap.issuer, "kid": cap.kid,
                    "payload_sha256": crypto.capability_hash(cap)}

        cap_indices.append(len(rows))
        cap_ids.append(cap.id)
        cap_scopes.append(cap.scope or {})
        cap_delegated.append(is_delegated)
        cap_meta.append(meta)
        rows.append(vec)
        row_types.append("capability")

    # ---- confirmation tokens ---------------------------------------------------------
    # Phase 8d: each confirmation carries an action-binding bit. A bound confirmation
    # (with action_hash) is only valid for the request whose action_hash it matches; an
    # unbound confirmation is valid only when bound confirmations are not required.
    req_hash = bundle.action_hash
    conf_indices: list[int] = []
    for conf in bundle.confirmations:
        vec = _blank()
        _set_slot(vec, "type", W.one_hot(W.TYPE_IDX, "confirmation", W.N_TYPE))
        _set_slot(vec, "subject", W.one_hot(W.SUBJ_IDX, conf.subject, W.N_SUBJ))
        _set_slot(vec, "object", W.one_hot(W.OBJ_IDX, conf.object, W.N_OBJ))
        _set_slot(vec, "rights", W.one_hot(W.RIGHT_IDX, conf.action, W.N_RIGHT))
        _set_slot(vec, "issuer", W.one_hot(W.ISSUER_IDX, conf.issuer, W.N_ISSUER))
        vec[W.CONFIRM_OFF] = 1.0
        if conf.action_hash is not None:
            bind_ok = req_hash is not None and conf.action_hash == req_hash
        else:
            bind_ok = not require_bound_confirmations
        vec[W.CBIND_OFF] = 1.0 if bind_ok else 0.0
        conf_indices.append(len(rows))
        rows.append(vec)
        row_types.append("confirmation")

    X = np.vstack(rows) if rows else np.zeros((0, W.D), dtype=np.float64)
    return EncodedBundle(
        X=X,
        row_types=row_types,
        request_index=request_index,
        cap_indices=cap_indices,
        conf_indices=conf_indices,
        cap_ids=cap_ids,
        cap_scopes=cap_scopes,
        bundle=bundle,
        require_signatures=require_signatures,
        cap_delegated=cap_delegated,
        cap_meta=cap_meta,
    )
