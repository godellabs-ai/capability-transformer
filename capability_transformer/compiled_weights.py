"""Compiled, fixed tensor weights for the capability transformer.

Nothing here is trained. These are the analog of compiled attention/FFN weights:
fixed one-hot vocabularies, slot offsets, Boolean relation masks and a reason-code
projection map. The enforcement pipeline (``hard_attention.py``) reads exclusively from
this module so that the "weights" of the machine live in exactly one place.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------------------
# Bounded universe (v1 vocabularies). Order matters: it defines one-hot index positions.
# --------------------------------------------------------------------------------------
TOKEN_TYPES = [
    "request",
    "capability",
    "confirmation",
    "revocation",
    "subject",
    "object",
    "provenance",
    "policy",
]
SUBJECTS = ["user", "agent", "document", "tool_result", "system"]
OBJECTS = ["gmail", "calendar", "file", "browser", "slack", "secrets_db"]
RIGHTS = ["read", "write", "draft", "send", "invoke", "delegate", "delete", "post"]
ISSUERS = ["trusted_user", "system", "document", "web_page", "tool_output", "model_generated"]
PROVENANCE = [
    "trusted_user",
    "system_policy",
    "retrieved_doc",
    "email_body",
    "web_page",
    "tool_output",
    "model_generated",
]

DECISIONS = ["ALLOW", "DENY", "ESCALATE"]
REASON_CODES = [
    "missing_capability",
    "right_not_granted",
    "object_mismatch",
    "subject_mismatch",
    "expired_capability",
    "revoked_capability",
    "untrusted_source",
    "data_has_no_authority",
    "delegation_not_allowed",
    "confirmation_required",
    "issuer_not_trusted",
    "scope_violation",
    "invalid_signature",
    "allowed",
]

# Index lookups -------------------------------------------------------------------------
TYPE_IDX = {v: i for i, v in enumerate(TOKEN_TYPES)}
SUBJ_IDX = {v: i for i, v in enumerate(SUBJECTS)}
OBJ_IDX = {v: i for i, v in enumerate(OBJECTS)}
RIGHT_IDX = {v: i for i, v in enumerate(RIGHTS)}
ISSUER_IDX = {v: i for i, v in enumerate(ISSUERS)}
PROV_IDX = {v: i for i, v in enumerate(PROVENANCE)}

# --------------------------------------------------------------------------------------
# Token vector layout (D = 44). Each token is a fixed-width vector of one-hot field slots
# plus four Boolean bits. ``request`` tokens store their action in the RIGHTS slot — that
# direction is the attention query for the right-match head.
# --------------------------------------------------------------------------------------
N_TYPE = len(TOKEN_TYPES)      # 8
N_SUBJ = len(SUBJECTS)         # 5
N_OBJ = len(OBJECTS)           # 6
N_RIGHT = len(RIGHTS)          # 8
N_ISSUER = len(ISSUERS)        # 6
N_PROV = len(PROVENANCE)       # 7

TYPE_OFF = 0
SUBJ_OFF = TYPE_OFF + N_TYPE          # 8
OBJ_OFF = SUBJ_OFF + N_SUBJ           # 13
RIGHTS_OFF = OBJ_OFF + N_OBJ          # 19
ISSUER_OFF = RIGHTS_OFF + N_RIGHT     # 27
PROV_OFF = ISSUER_OFF + N_ISSUER      # 33
EXPIRY_OFF = PROV_OFF + N_PROV        # 40
REVOKED_OFF = EXPIRY_OFF + 1          # 41
DELEG_OFF = REVOKED_OFF + 1           # 42
CONFIRM_OFF = DELEG_OFF + 1           # 43
SIG_OFF = CONFIRM_OFF + 1             # 44  (Phase 8a: signature-valid bit)
D = SIG_OFF + 1                       # 45

# Convenience slot slices (start, stop) for slicing the token matrix X.
SLOT = {
    "type": (TYPE_OFF, SUBJ_OFF),
    "subject": (SUBJ_OFF, OBJ_OFF),
    "object": (OBJ_OFF, RIGHTS_OFF),
    "rights": (RIGHTS_OFF, ISSUER_OFF),
    "issuer": (ISSUER_OFF, PROV_OFF),
    "provenance": (PROV_OFF, EXPIRY_OFF),
}


def one_hot(index_map: dict[str, int], value: str, width: int) -> np.ndarray:
    """Return a fixed one-hot embedding for ``value`` (no learned table)."""
    vec = np.zeros(width, dtype=np.float64)
    vec[index_map[value]] = 1.0
    return vec


def multi_hot(index_map: dict[str, int], values, width: int) -> np.ndarray:
    """Return a fixed multi-hot embedding for a set of ``values``."""
    vec = np.zeros(width, dtype=np.float64)
    for v in values:
        vec[index_map[v]] = 1.0
    return vec


# --------------------------------------------------------------------------------------
# Compiled Boolean relation masks (fixed "weights").
# --------------------------------------------------------------------------------------
# Trusted issuers: only trusted_user and system may mint authority.
TRUSTED_ISSUER_MASK = multi_hot(ISSUER_IDX, ["trusted_user", "system"], N_ISSUER)

# Trusted provenance: only a trusted_user or system_policy may *drive* a side effect.
TRUSTED_PROV_MASK = multi_hot(PROV_IDX, ["trusted_user", "system_policy"], N_PROV)

# Passive (non-side-effecting) rights that untrusted data is still allowed to drive.
NON_SIDE_EFFECT_MASK = multi_hot(RIGHT_IDX, ["read"], N_RIGHT)

# High-risk (object x right) relation matrix. A 1 means the action requires human
# confirmation before it can be ALLOWed.
HIGH_RISK = np.zeros((N_OBJ, N_RIGHT), dtype=np.float64)
for _obj, _right in [
    ("gmail", "send"),
    ("slack", "post"),
    ("file", "delete"),
    ("secrets_db", "read"),
    ("browser", "invoke"),
]:
    HIGH_RISK[OBJ_IDX[_obj], RIGHT_IDX[_right]] = 1.0

# Output projection: each attention head maps to exactly one reason code on failure.
HEAD_REASON = {
    "head_subject_match": "subject_mismatch",
    "head_object_match": "object_mismatch",
    "head_right_match": "right_not_granted",
    "head_trusted_issuer": "issuer_not_trusted",
    "head_not_expired": "expired_capability",
    "head_not_revoked": "revoked_capability",
    "head_provenance_safe": "data_has_no_authority",
    "head_confirmation": "confirmation_required",
    "head_scope": "scope_violation",
    "head_delegation": "delegation_not_allowed",
    "head_signature_valid": "invalid_signature",
}

# The six heads whose conjunction forms the capability-matching security boundary.
MATCHING_HEADS = [
    "head_subject_match",
    "head_object_match",
    "head_right_match",
    "head_trusted_issuer",
    "head_not_expired",
    "head_not_revoked",
]

ENGINE_NAME = "hard-attention-v1"
