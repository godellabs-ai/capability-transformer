"""Phase 8d — action-hash-bound confirmations (no replay across actions)."""

from datetime import datetime, timezone

import pytest

from capability_transformer import Capability, CapabilityBundle, CapabilityTransformer, ToolCall
from capability_transformer.runtime import ToolGateway, compute_action_hash

FUTURE = datetime(2099, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)


def cap(rights=("send",), object="gmail"):
    return Capability(id="cap1", subject="agent", object=object, rights=list(rights),
                      issuer="trusted_user", expires_at=FUTURE)


def call(args, action="send", object="gmail"):
    return ToolCall(subject="agent", action=action, object=object, args=args)


def bundle(action_hash=None, confirmations=None, object="gmail", action="send"):
    return CapabilityBundle(
        subject="agent", action=action, object=object, source_provenance="trusted_user",
        capabilities=[cap([action], object)],
        confirmations=confirmations or [], action_hash=action_hash,
    )


def conf(action_hash=None, object="gmail", action="send"):
    c = {"subject": "agent", "object": object, "action": action, "issuer": "trusted_user"}
    if action_hash is not None:
        c["action_hash"] = action_hash
    return c


@pytest.fixture
def strict():
    # Strict engine: only action-bound confirmations are accepted.
    return CapabilityTransformer(require_bound_confirmations=True)


@pytest.fixture
def lax():
    return CapabilityTransformer()


def test_bound_confirmation_matching_hash_allows(strict):
    h = compute_action_hash(call({"to": "bob"}))
    d = strict.evaluate(bundle(action_hash=h, confirmations=[conf(action_hash=h)]))
    assert d.decision == "ALLOW"


def test_bound_confirmation_wrong_hash_escalates(strict):
    approved = compute_action_hash(call({"to": "bob"}))
    requested = compute_action_hash(call({"to": "attacker"}))
    d = strict.evaluate(bundle(action_hash=requested, confirmations=[conf(action_hash=approved)]))
    assert d.decision == "ESCALATE"
    assert d.reasons == ["confirmation_required"]


def test_strict_mode_rejects_unbound_confirmation(strict):
    h = compute_action_hash(call({"to": "bob"}))
    d = strict.evaluate(bundle(action_hash=h, confirmations=[conf()]))  # no action_hash
    assert d.decision == "ESCALATE"


def test_lax_mode_accepts_unbound_confirmation(lax):
    # Backward compatible: without binding required, an unbound confirmation works.
    d = lax.evaluate(bundle(confirmations=[conf()]))
    assert d.decision == "ALLOW"


def test_confirmation_cannot_replay_across_actions(strict):
    # A confirmation approving "send to bob" must not approve "send to attacker".
    approved = compute_action_hash(call({"to": "bob", "body": "hi"}))
    c = conf(action_hash=approved)
    ok = strict.evaluate(bundle(action_hash=approved, confirmations=[c]))
    assert ok.decision == "ALLOW"
    replay_hash = compute_action_hash(call({"to": "attacker", "body": "hi"}))
    replay = strict.evaluate(bundle(action_hash=replay_hash, confirmations=[c]))
    assert replay.decision == "ESCALATE"


def test_end_to_end_gateway_binds_confirmation():
    engine = CapabilityTransformer(require_bound_confirmations=True)
    gateway = ToolGateway(engine=engine)

    approved_call = call({"to": "bob", "body": "hi"})
    c = conf(action_hash=compute_action_hash(approved_call))

    # Authorizing the exact approved call -> ALLOW + grant.
    b_ok = CapabilityBundle(subject="agent", action="send", object="gmail",
                            source_provenance="trusted_user", capabilities=[cap(["send"])],
                            confirmations=[c])
    decision, grant = gateway.authorize(b_ok, approved_call, now=NOW, nonce="n1")
    assert decision.decision == "ALLOW"
    assert grant is not None

    # Same confirmation, different args -> the gateway sets a different action_hash -> no match.
    evil_call = call({"to": "attacker", "body": "hi"})
    b_evil = CapabilityBundle(subject="agent", action="send", object="gmail",
                              source_provenance="trusted_user", capabilities=[cap(["send"])],
                              confirmations=[c])
    decision2, grant2 = gateway.authorize(b_evil, evil_call, now=NOW, nonce="n2")
    assert decision2.decision == "ESCALATE"
    assert grant2 is None


def test_trace_exposes_action_hash(strict):
    h = compute_action_hash(call({"to": "bob"}))
    d = strict.evaluate(bundle(action_hash=h, confirmations=[conf(action_hash=h)]))
    assert d.trace.request["action_hash"] == h
