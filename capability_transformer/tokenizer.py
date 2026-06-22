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
from . import crypto
from .schema import CapabilityBundle


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


def _aware(dt: datetime) -> datetime:
    """Normalize to a timezone-aware UTC datetime for safe comparison."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_revoked(cap, revocations) -> bool:
    """A capability is revoked if any revocation matches it by id or by fields."""
    for rev in revocations:
        if rev.capability_id is not None:
            if rev.capability_id == cap.id:
                return True
            continue
        # Field-based revocation: revoke all caps matching the given subject/object.
        subj_ok = rev.subject is None or rev.subject == cap.subject
        obj_ok = rev.object is None or rev.object == cap.object
        if subj_ok and obj_ok and (rev.subject is not None or rev.object is not None):
            return True
    return False


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

    # ---- capability (key/value) tokens -----------------------------------------------
    cap_indices: list[int] = []
    cap_ids: list[str] = []
    cap_scopes: list[dict] = []
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
        # Signature-valid bit. When not enforcing, the bit is set (1) and the signature
        # head stays inactive; when enforcing, it reflects HMAC verification.
        if require_signatures:
            kr = keyring if keyring is not None else crypto.DEFAULT_KEYRING
            vec[W.SIG_OFF] = 1.0 if crypto.verify(cap, keyring=kr) else 0.0
        else:
            vec[W.SIG_OFF] = 1.0
        cap_indices.append(len(rows))
        cap_ids.append(cap.id)
        cap_scopes.append(cap.scope or {})
        rows.append(vec)
        row_types.append("capability")

    # ---- confirmation tokens ---------------------------------------------------------
    conf_indices: list[int] = []
    for conf in bundle.confirmations:
        vec = _blank()
        _set_slot(vec, "type", W.one_hot(W.TYPE_IDX, "confirmation", W.N_TYPE))
        _set_slot(vec, "subject", W.one_hot(W.SUBJ_IDX, conf.subject, W.N_SUBJ))
        _set_slot(vec, "object", W.one_hot(W.OBJ_IDX, conf.object, W.N_OBJ))
        _set_slot(vec, "rights", W.one_hot(W.RIGHT_IDX, conf.action, W.N_RIGHT))
        _set_slot(vec, "issuer", W.one_hot(W.ISSUER_IDX, conf.issuer, W.N_ISSUER))
        vec[W.CONFIRM_OFF] = 1.0
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
    )
