"""Phase 8a/8b — cryptographically authenticated capabilities.

This implements **cryptographically authenticated capabilities under a trusted
symmetric-key issuer model** (HMAC-SHA256), plus **macaroon-style chained-HMAC
attenuation** for delegated child capabilities. It is NOT a full macaroon library
(no third-party/discharge caveats) and the symmetric key model is a single-verifier
mock — production should use asymmetric signatures (Ed25519) or real macaroons. See
implementation.md §21/§22.

Two signing modes:

* **Root issuance** (`issue`): a trusted issuer signs a capability with its secret key,
  selected by ``kid`` (key id) to support rotation.
* **Delegated attenuation** (`sign_child` / verified by `verify_child`): the *holder* of a
  parent capability derives a child by HMAC'ing the child's canonical payload (which
  embeds ``parent_hash`` and the attenuated fields) under the *parent's signature* as the
  key. No issuer key is needed to delegate; the gateway re-derives the chain.

Verification is always reduced to Boolean bits on the capability token — the attention
core never touches key material.
"""

from __future__ import annotations

import hmac
import json
from datetime import datetime, timezone
from hashlib import sha256
from typing import Mapping, Optional

# A keyring maps an issuer to a set of versioned keys plus the currently active kid.
# Only trusted issuers hold keys; untrusted issuers cannot produce a valid signature.
# Replace with a real secret store (and asymmetric keys) in production.
Keyring = Mapping[str, dict]
DEFAULT_KEYRING: dict[str, dict] = {
    "trusted_user": {
        "keys": {"trusted_user-key-1": "demo-secret::trusted_user::v1::do-not-use"},
        "active": "trusted_user-key-1",
    },
    "system": {
        "keys": {"system-key-1": "demo-secret::system::v1::do-not-use"},
        "active": "system-key-1",
    },
}


def active_kid(issuer: str, keyring: Keyring = DEFAULT_KEYRING) -> str:
    """Return the issuer's currently active key id (raises KeyError if no key)."""
    return keyring[issuer]["active"]


def _secret(issuer: str, kid: Optional[str], keyring: Keyring) -> str:
    """Look up the secret for (issuer, kid); raises KeyError if unknown."""
    return keyring[issuer]["keys"][kid]


def _canonical_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _payload_dict(cap) -> dict:
    """All signed fields of a capability, in a deterministic, attenuation-covering form.

    The signature/hash binds every authority-relevant field, including the delegation
    lineage (``parent_id``, ``parent_hash``) and limits (``delegation_depth``,
    ``max_delegation_depth``). The ``signature`` itself is excluded so hashes are stable.
    """
    return {
        "id": cap.id,
        "subject": cap.subject,
        "object": cap.object,
        "rights": sorted(cap.rights),
        "issuer": cap.issuer,
        "expires_at": _canonical_iso(cap.expires_at),
        "scope": cap.scope or {},
        "delegatable": bool(cap.delegatable),
        "kid": getattr(cap, "kid", None),
        "parent_id": getattr(cap, "parent_id", None),
        "parent_hash": getattr(cap, "parent_hash", None),
        "delegation_depth": getattr(cap, "delegation_depth", 0) or 0,
        "max_delegation_depth": getattr(cap, "max_delegation_depth", None),
    }


def canonical_payload(cap) -> str:
    """Deterministic JSON serialization of the signed fields."""
    return json.dumps(_payload_dict(cap), sort_keys=True, separators=(",", ":"))


def capability_hash(cap) -> str:
    """Stable SHA-256 over the canonical payload (safe to log; reveals no secret)."""
    return sha256(canonical_payload(cap).encode()).hexdigest()


def _hmac_hex(key: bytes, msg: str) -> str:
    return hmac.new(key, msg.encode(), sha256).hexdigest()


# ---- Root issuance ------------------------------------------------------------------
def issue(cap, *, keyring: Keyring = DEFAULT_KEYRING):
    """Return a signed copy of a root capability (populates ``kid`` and ``signature``)."""
    kid = getattr(cap, "kid", None) or active_kid(cap.issuer, keyring)  # KeyError if untrusted
    cap = cap.model_copy(update={"kid": kid})
    secret = _secret(cap.issuer, kid, keyring)
    signature = _hmac_hex(secret.encode(), canonical_payload(cap))
    return cap.model_copy(update={"signature": signature})


# Compatibility alias used by earlier examples/tests.
def mint(cap, *, keyring: Keyring = DEFAULT_KEYRING) -> str:
    """Sign a root capability and return its signature (issuer must be trusted)."""
    return issue(cap, keyring=keyring).signature


def verify(cap, *, keyring: Keyring = DEFAULT_KEYRING) -> bool:
    """Verify a *root* capability's issuer HMAC signature.

    False for missing signature/kid, unknown issuer or kid (no key), and any tampering.
    """
    signature = getattr(cap, "signature", None)
    kid = getattr(cap, "kid", None)
    if not signature or not kid:
        return False
    try:
        secret = _secret(cap.issuer, kid, keyring)
    except KeyError:
        return False
    expected = _hmac_hex(secret.encode(), canonical_payload(cap))
    return hmac.compare_digest(expected, signature)


# ---- Delegated (chained-HMAC) attenuation -------------------------------------------
def sign_child(parent_signature: str, child) -> str:
    """Derive a child signature: HMAC over the child payload, keyed by the parent sig."""
    return _hmac_hex(parent_signature.encode(), canonical_payload(child))


def verify_child(child, parent) -> bool:
    """Verify the chained-HMAC link from ``parent`` to ``child``."""
    parent_sig = getattr(parent, "signature", None)
    child_sig = getattr(child, "signature", None)
    if not parent_sig or not child_sig:
        return False
    expected = _hmac_hex(parent_sig.encode(), canonical_payload(child))
    return hmac.compare_digest(expected, child_sig)
