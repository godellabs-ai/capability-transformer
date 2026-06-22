"""Expiry and revocation always win over an otherwise-valid capability."""

from conftest import PAST, bundle, cap


def test_expired_capability_denied(engine):
    d = engine.evaluate(bundle(action="read", object="file",
                               capabilities=[cap(object="file", rights=["read"], expires_at=PAST)]))
    assert d.decision == "DENY"
    assert "expired_capability" in d.reasons


def test_non_expired_capability_allowed(engine):
    d = engine.evaluate(bundle(action="read", object="file",
                               capabilities=[cap(object="file", rights=["read"])]))
    assert d.decision == "ALLOW"


def test_revoked_capability_denied(engine):
    d = engine.evaluate(bundle(action="read", object="file",
                               capabilities=[cap(object="file", rights=["read"])],
                               revocations=[{"capability_id": "cap1"}]))
    assert d.decision == "DENY"
    assert "revoked_capability" in d.reasons


def test_non_revoked_capability_allowed(engine):
    d = engine.evaluate(bundle(action="read", object="file",
                               capabilities=[cap(object="file", rights=["read"])],
                               revocations=[{"capability_id": "other"}]))
    assert d.decision == "ALLOW"


def test_revocation_wins_over_valid_capability(engine):
    # A fully valid, fresh, trusted capability is still denied once revoked.
    d = engine.evaluate(bundle(action="read", object="file",
                               capabilities=[cap(object="file", rights=["read"])],
                               revocations=[{"capability_id": "cap1"}]))
    assert d.decision == "DENY"
    assert d.trace.matched_capabilities == []


def test_field_based_revocation(engine):
    d = engine.evaluate(bundle(action="read", object="file",
                               capabilities=[cap(object="file", rights=["read"])],
                               revocations=[{"subject": "agent", "object": "file"}]))
    assert d.decision == "DENY"
    assert "revoked_capability" in d.reasons
