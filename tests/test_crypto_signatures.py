"""Phase 8a — unforgeable, signature-gated capabilities."""

from datetime import datetime, timezone

import pytest

from capability_transformer import Capability, CapabilityBundle, CapabilityTransformer, crypto
from conftest import FUTURE


def _signed_cap(rights=("read",), object="file", subject="agent", issuer="trusted_user", **over):
    cap = Capability(id="cap1", subject=subject, object=object, rights=list(rights),
                     issuer=issuer, expires_at=FUTURE, scope={}, delegatable=False, **over)
    return cap.model_copy(update={"signature": crypto.mint(cap)})


def _bundle(cap, action="read", object="file", subject="agent", provenance="trusted_user"):
    return CapabilityBundle(subject=subject, action=action, object=object,
                            source_provenance=provenance, capabilities=[cap])


@pytest.fixture
def signed_engine():
    return CapabilityTransformer(require_signatures=True)


def test_valid_signature_allows(signed_engine):
    d = signed_engine.evaluate(_bundle(_signed_cap(rights=["read"])))
    assert d.decision == "ALLOW"
    assert d.reasons == ["allowed"]


def test_missing_signature_denied(signed_engine):
    cap = Capability(id="cap1", subject="agent", object="file", rights=["read"],
                     issuer="trusted_user", expires_at=FUTURE)  # no signature
    d = signed_engine.evaluate(_bundle(cap))
    assert d.decision == "DENY"
    assert d.reasons == ["invalid_signature"]


def test_forged_issuer_label_denied(signed_engine):
    # An attacker copies a real signature but flips a field: HMAC no longer matches.
    cap = _signed_cap(rights=["read"])
    tampered = cap.model_copy(update={"rights": ["read", "write"]})  # signature now stale
    d = signed_engine.evaluate(_bundle(tampered, action="write"))
    assert d.decision == "DENY"
    assert "invalid_signature" in d.reasons


def test_untrusted_issuer_cannot_sign(signed_engine):
    # Untrusted issuers hold no key, so no valid signature can exist for them.
    with pytest.raises(KeyError):
        crypto.mint(Capability(id="x", subject="agent", object="file", rights=["read"],
                               issuer="web_page", expires_at=FUTURE))


def test_signature_head_in_trace(signed_engine):
    d = signed_engine.evaluate(_bundle(_signed_cap(rights=["read"])))
    names = {h.name for h in d.trace.heads}
    assert "head_signature_valid" in names


def test_unsigned_mode_ignores_signatures():
    # Default engine (v1 behavior) does not require signatures.
    engine = CapabilityTransformer()
    cap = Capability(id="cap1", subject="agent", object="file", rights=["read"],
                     issuer="trusted_user", expires_at=FUTURE)
    d = engine.evaluate(_bundle(cap))
    assert d.decision == "ALLOW"
    names = {h.name for h in d.trace.heads}
    assert "head_signature_valid" not in names  # head inactive when not enforced


def test_tampered_expiry_detected(signed_engine):
    cap = _signed_cap(rights=["read"])
    tampered = cap.model_copy(update={"expires_at": datetime(2100, 6, 6, tzinfo=timezone.utc)})
    d = signed_engine.evaluate(_bundle(tampered))
    assert d.decision == "DENY"
    assert "invalid_signature" in d.reasons


def test_verify_roundtrip():
    cap = _signed_cap(rights=["read", "write"])
    assert crypto.verify(cap) is True
    assert crypto.verify(cap.model_copy(update={"object": "gmail"})) is False
