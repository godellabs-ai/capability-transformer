"""Delegation requires `delegate` + the target right; attenuation only."""

from conftest import bundle, cap


def _delegate(rights, delegate_right):
    return bundle(
        subject="user",
        action="delegate",
        object="file",
        source_provenance="trusted_user",
        delegate_right=delegate_right,
        delegate_to="agent",
        capabilities=[cap(subject="user", object="file", rights=rights, delegatable=True)],
    )


def test_read_and_delegate_can_grant_read(engine):
    d = engine.evaluate(_delegate(["read", "delegate"], "read"))
    assert d.decision == "ALLOW"


def test_only_read_cannot_grant_read(engine):
    # No `delegate` right -> the right-match head for "delegate" fails.
    d = engine.evaluate(_delegate(["read"], "read"))
    assert d.decision == "DENY"


def test_read_and_delegate_cannot_grant_write(engine):
    # Holds `delegate` but not `write` -> cannot amplify beyond what is held.
    d = engine.evaluate(_delegate(["read", "delegate"], "write"))
    assert d.decision == "DENY"
    assert "delegation_not_allowed" in d.reasons


def test_read_write_delegate_can_grant_read_or_write(engine):
    assert engine.evaluate(_delegate(["read", "write", "delegate"], "read")).decision == "ALLOW"
    assert engine.evaluate(_delegate(["read", "write", "delegate"], "write")).decision == "ALLOW"


def test_delegation_must_be_attenuated_not_amplified(engine):
    # Holding read+delegate, attempting to grant `delete` (not held) must be denied.
    d = engine.evaluate(_delegate(["read", "delegate"], "delete"))
    assert d.decision == "DENY"
    assert "delegation_not_allowed" in d.reasons
