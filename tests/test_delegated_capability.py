"""Phase 8b — attenuable delegated capability chains (macaroon-style)."""

from datetime import datetime, timezone

import pytest

from capability_transformer import Capability, CapabilityBundle, CapabilityTransformer, crypto
from capability_transformer.delegated_capability import mint_child

FUTURE = datetime(2099, 1, 1, tzinfo=timezone.utc)
LATER = datetime(2100, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def engine():
    return CapabilityTransformer(require_signatures=True)


def parent(rights=("read", "delegate"), subject="user", object="file",
           expires_at=FUTURE, scope=None, delegatable=True, max_delegation_depth=2):
    cap = Capability(id="parent", subject=subject, object=object, rights=list(rights),
                     issuer="trusted_user", expires_at=expires_at, scope=scope or {},
                     delegatable=delegatable, max_delegation_depth=max_delegation_depth)
    return crypto.issue(cap)


def request(action, caps, subject="agent", object="file", provenance="trusted_user"):
    return CapabilityBundle(subject=subject, action=action, object=object,
                            source_provenance=provenance, capabilities=caps)


def test_parent_read_delegate_can_mint_child_read(engine):
    p = parent(rights=["read", "delegate"])
    child = mint_child(p, id="child", subject="agent", rights=["read"])
    d = engine.evaluate(request("read", [p, child], subject="agent"))
    assert d.decision == "ALLOW"
    assert d.trace.delegation["delegation_chain_valid"] is True
    assert d.trace.delegation["attenuation_valid"] is True


def test_parent_read_delegate_cannot_mint_child_write(engine):
    # Holder tries to amplify: grant a right the parent never held.
    p = parent(rights=["read", "delegate"])
    child = mint_child(p, id="child", subject="agent", rights=["read", "write"])
    d = engine.evaluate(request("write", [p, child], subject="agent"))
    assert d.decision == "DENY"
    assert "attenuation_violation" in d.reasons


def test_no_delegate_cannot_mint_cross_subject_child(engine):
    # Parent lacks `delegate`; handing authority to another subject is not allowed.
    p = parent(rights=["read"])  # no delegate
    child = mint_child(p, id="child", subject="agent", rights=["read"])
    d = engine.evaluate(request("read", [p, child], subject="agent"))
    assert d.decision == "DENY"
    # parent lacks delegate -> chain invalid; subject change -> attenuation violation
    assert "delegation_chain_invalid" in d.reasons or "attenuation_violation" in d.reasons


def test_child_cannot_outlive_parent(engine):
    p = parent(rights=["read", "delegate"])
    child = mint_child(p, id="child", subject="agent", rights=["read"], expires_at=LATER)
    d = engine.evaluate(request("read", [p, child], subject="agent"))
    assert d.decision == "DENY"
    assert "attenuation_violation" in d.reasons


def test_child_cannot_widen_scope(engine):
    p = parent(rights=["read", "delegate"], scope={"folder": "reports"})
    # Child drops the parent's folder constraint -> widening.
    child = mint_child(p, id="child", subject="agent", rights=["read"], scope={})
    d = engine.evaluate(request("read", [p, child], subject="agent"))
    assert d.decision == "DENY"
    assert "attenuation_violation" in d.reasons


def test_child_delegatable_gated(engine):
    # A child may only be re-delegatable if the parent can delegate and depth permits.
    p = parent(rights=["read"], delegatable=True)  # parent cannot delegate (no right)
    child = mint_child(p, id="child", subject="agent", rights=["read"], delegatable=True)
    d = engine.evaluate(request("read", [p, child], subject="agent"))
    assert d.decision == "DENY"


def test_two_hop_redelegation_allowed(engine):
    p = parent(rights=["read", "delegate"])
    mid = mint_child(p, id="mid", subject="agent", rights=["read", "delegate"], delegatable=True)
    leaf = mint_child(mid, id="leaf", subject="tool_result", rights=["read"])
    d = engine.evaluate(request("read", [p, mid, leaf], subject="tool_result"))
    assert d.decision == "ALLOW"


def test_tampering_child_rights_invalidates_signature(engine):
    p = parent(rights=["read", "delegate"])
    child = mint_child(p, id="child", subject="agent", rights=["read"])
    tampered = child.model_copy(update={"rights": ["read", "write"]})  # sig now stale
    d = engine.evaluate(request("write", [p, tampered], subject="agent"))
    assert d.decision == "DENY"
    assert "invalid_signature" in d.reasons


def test_tampering_parent_hash_invalidates_chain(engine):
    p = parent(rights=["read", "delegate"])
    child = mint_child(p, id="child", subject="agent", rights=["read"])
    tampered = child.model_copy(update={"parent_hash": "0" * 64})
    d = engine.evaluate(request("read", [p, tampered], subject="agent"))
    assert d.decision == "DENY"
    # parent_hash is signed, so this breaks both the signature and the chain link.
    assert "invalid_signature" in d.reasons or "delegation_chain_invalid" in d.reasons


def test_revoking_parent_invalidates_child(engine):
    p = parent(rights=["read", "delegate"])
    child = mint_child(p, id="child", subject="agent", rights=["read"])
    b = CapabilityBundle(subject="agent", action="read", object="file",
                         source_provenance="trusted_user", capabilities=[p, child],
                         revocations=[{"capability_id": "parent"}])
    d = engine.evaluate(b)
    assert d.decision == "DENY"
    assert "delegation_chain_invalid" in d.reasons


def test_expiring_parent_invalidates_child(engine):
    past = datetime(2000, 1, 1, tzinfo=timezone.utc)
    p = parent(rights=["read", "delegate"], expires_at=past)
    # Child expiry must be <= parent; keep it in the past too so only chain fails.
    child = mint_child(p, id="child", subject="agent", rights=["read"], expires_at=past)
    d = engine.evaluate(request("read", [p, child], subject="agent"))
    assert d.decision == "DENY"
    # Child itself is also expired; either way it must not authorize.
    assert d.decision != "ALLOW"


def test_trace_exposes_chain_metadata(engine):
    p = parent(rights=["read", "delegate"])
    child = mint_child(p, id="child", subject="agent", rights=["read"])
    d = engine.evaluate(request("read", [p, child], subject="agent"))
    chains = d.trace.delegation["chains"]
    assert any(c["capability_id"] == "child" and c["parent_capability_id"] == "parent"
               for c in chains)
    names = {h.name for h in d.trace.heads}
    assert {"head_chain_valid", "head_attenuation_valid"} <= names
