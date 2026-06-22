"""Phase 8a — unforgeable capabilities (mock issuer with real signatures).

v1 trusted a capability's ``issuer`` *label*. That is forgeable: untrusted text could
simply claim ``issuer="trusted_user"``. This module binds a capability's fields to an
issuer's secret key with an HMAC-SHA256 signature, so a capability cannot be minted or
mutated without the issuer key.

The signature is verified at tokenization time and reduced to a single Boolean bit on
the capability token, exactly like the expiry and revoked bits — keeping the enforcement
path a pure tensor pipeline. A dedicated hard-attention head (``head_signature_valid``)
consumes that bit when signature enforcement is enabled.

Note: HMAC with a shared per-issuer secret is a *symmetric* mock suitable for a single
trusted verifier (the gateway). A production system would use asymmetric signatures
(e.g. Ed25519) or macaroons so that verifiers need no secret. See implementation.md
Phase 8 / Future work.
"""

from __future__ import annotations

import hmac
import json
from datetime import datetime, timezone
from hashlib import sha256
from typing import Mapping, Optional

Keyring = Mapping[str, str]

# A demo keyring. ONLY trusted issuers hold keys; untrusted issuers (document, web_page,
# tool_output, model_generated) have none and therefore cannot produce a valid signature.
# Replace with a real secret store in production.
DEFAULT_KEYRING: dict[str, str] = {
    "trusted_user": "demo-key::trusted_user::do-not-use-in-production",
    "system": "demo-key::system::do-not-use-in-production",
}


def _canonical_iso(dt: datetime) -> str:
    """Normalize a datetime to a canonical UTC ISO-8601 string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def canonical_payload(
    *,
    id: str,
    subject: str,
    object: str,
    rights,
    issuer: str,
    expires_at: datetime,
    scope: Optional[dict] = None,
    delegatable: bool = False,
) -> str:
    """Deterministic, order-independent serialization of the signed fields.

    Rights are sorted and the scope dict is serialized with sorted keys so that two
    semantically identical capabilities always produce the same payload.
    """
    payload = {
        "id": id,
        "subject": subject,
        "object": object,
        "rights": sorted(rights),
        "issuer": issuer,
        "expires_at": _canonical_iso(expires_at),
        "scope": scope or {},
        "delegatable": bool(delegatable),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sign(payload: str, *, issuer: str, keyring: Keyring = DEFAULT_KEYRING) -> str:
    """Return the hex HMAC-SHA256 signature for ``payload`` under the issuer key."""
    key = keyring.get(issuer)
    if key is None:
        raise KeyError(f"no signing key for issuer {issuer!r}")
    return hmac.new(key.encode(), payload.encode(), sha256).hexdigest()


def mint(cap, *, keyring: Keyring = DEFAULT_KEYRING) -> str:
    """Sign a Capability (or capability-like object) and return its signature."""
    payload = canonical_payload(
        id=cap.id,
        subject=cap.subject,
        object=cap.object,
        rights=cap.rights,
        issuer=cap.issuer,
        expires_at=cap.expires_at,
        scope=cap.scope,
        delegatable=cap.delegatable,
    )
    return sign(payload, issuer=cap.issuer, keyring=keyring)


def verify(cap, *, keyring: Keyring = DEFAULT_KEYRING) -> bool:
    """Return True iff ``cap.signature`` is a valid HMAC for ``cap`` under its issuer key.

    Returns False for missing signatures, unknown issuers (no key), and any tampering
    with the signed fields.
    """
    signature = getattr(cap, "signature", None)
    if not signature:
        return False
    key = keyring.get(cap.issuer)
    if key is None:
        return False
    payload = canonical_payload(
        id=cap.id,
        subject=cap.subject,
        object=cap.object,
        rights=cap.rights,
        issuer=cap.issuer,
        expires_at=cap.expires_at,
        scope=cap.scope,
        delegatable=cap.delegatable,
    )
    expected = hmac.new(key.encode(), payload.encode(), sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
