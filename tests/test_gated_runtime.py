"""Phase 8c — the gated tool runtime enforces fresh, bound, single-use grants."""

from datetime import datetime, timedelta, timezone

import pytest

from capability_transformer import Capability, CapabilityBundle, ToolCall
from capability_transformer.runtime import GatedToolRuntime, GrantIssuer, ToolGateway

FUTURE = datetime(2099, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)


def cap(rights, object="gmail", subject="agent", issuer="trusted_user"):
    return Capability(id="cap1", subject=subject, object=object, rights=list(rights),
                      issuer=issuer, expires_at=FUTURE)


def bundle(action, object, provenance="trusted_user", rights=None, confirmations=None):
    return CapabilityBundle(subject="agent", action=action, object=object,
                            source_provenance=provenance,
                            capabilities=[cap(rights or [action], object)],
                            confirmations=confirmations or [])


def call(action, object, args=None):
    return ToolCall(subject="agent", action=action, object=object, args=args or {})


@pytest.fixture
def gateway():
    return ToolGateway()


@pytest.fixture
def runtime():
    return GatedToolRuntime()


def test_allow_yields_grant_and_executes(gateway, runtime):
    b, c = bundle("draft", "gmail"), call("draft", "gmail", {"to": "x", "body": "hi"})
    decision, grant = gateway.authorize(b, c, now=NOW, nonce="n1")
    assert decision.decision == "ALLOW"
    assert grant is not None
    ex = runtime.execute(grant, c, now=NOW)
    assert ex.executed is True
    assert ex.tool == "gmail.draft"
    assert ex.result["to"] == "x"


def test_deny_yields_no_grant_and_refuses(gateway, runtime):
    # Untrusted document tries to send mail.
    b = bundle("send", "gmail", provenance="retrieved_doc", rights=["send"])
    c = call("send", "gmail")
    decision, grant = gateway.authorize(b, c, now=NOW, nonce="n2")
    assert decision.decision == "DENY"
    assert grant is None
    ex = runtime.execute(grant, c, now=NOW)
    assert ex.executed is False
    assert ex.refused_reason == "no_grant"


def test_escalate_yields_no_grant(gateway, runtime):
    # High-risk send with capability but no confirmation -> ESCALATE -> no grant.
    b = bundle("send", "gmail", rights=["send"])
    c = call("send", "gmail")
    decision, grant = gateway.authorize(b, c, now=NOW, nonce="n3")
    assert decision.decision == "ESCALATE"
    assert grant is None
    assert runtime.execute(grant, c, now=NOW).executed is False


def test_confirmed_high_risk_executes(gateway, runtime):
    b = bundle("send", "gmail", rights=["send"],
               confirmations=[{"subject": "agent", "object": "gmail", "action": "send", "issuer": "trusted_user"}])
    c = call("send", "gmail", {"to": "x"})
    decision, grant = gateway.authorize(b, c, now=NOW, nonce="n4")
    assert decision.decision == "ALLOW"
    assert runtime.execute(grant, c, now=NOW).executed is True


def test_grant_is_single_use(gateway, runtime):
    b, c = bundle("draft", "gmail"), call("draft", "gmail")
    _, grant = gateway.authorize(b, c, now=NOW, nonce="n5")
    assert runtime.execute(grant, c, now=NOW).executed is True
    replay = runtime.execute(grant, c, now=NOW)
    assert replay.executed is False
    assert replay.refused_reason == "grant_replayed"


def test_grant_expires(gateway, runtime):
    b, c = bundle("draft", "gmail"), call("draft", "gmail")
    _, grant = gateway.authorize(b, c, now=NOW, nonce="n6", ttl_seconds=30)
    ex = runtime.execute(grant, c, now=NOW + timedelta(seconds=31))
    assert ex.executed is False
    assert ex.refused_reason == "grant_expired"


def test_tampered_grant_action_refused(gateway, runtime):
    # Attacker upgrades a draft grant to a send -> signature no longer verifies.
    b, c = bundle("draft", "gmail"), call("draft", "gmail", {"to": "x", "body": "hi"})
    _, grant = gateway.authorize(b, c, now=NOW, nonce="n7")
    forged = grant.model_copy(update={"action": "send"})
    send_call = call("send", "gmail", {"to": "x", "body": "hi"})
    ex = runtime.execute(forged, send_call, now=NOW)
    assert ex.executed is False
    assert ex.refused_reason == "grant_signature_invalid"


def test_grant_bound_to_args(gateway, runtime):
    # A grant minted for one set of args cannot execute a different payload.
    b = bundle("draft", "gmail")
    c = call("draft", "gmail", {"to": "alice", "body": "hi"})
    _, grant = gateway.authorize(b, c, now=NOW, nonce="n8")
    other = call("draft", "gmail", {"to": "attacker", "body": "wire funds"})
    ex = runtime.execute(grant, other, now=NOW)
    assert ex.executed is False
    assert ex.refused_reason == "action_binding_mismatch"


def test_grant_from_other_secret_rejected(runtime):
    # A grant signed with a secret the runtime does not share is refused.
    rogue = GrantIssuer(secret="rogue-secret")
    c = call("draft", "gmail")
    grant = rogue.issue(c, now=NOW, nonce="n9", decision_id="x")
    ex = runtime.execute(grant, c, now=NOW)
    assert ex.executed is False
    assert ex.refused_reason == "grant_signature_invalid"


def test_grant_does_not_bind_mismatched_call(gateway):
    # If the tool call doesn't match the evaluated bundle, no grant is issued.
    b = bundle("draft", "gmail")
    mismatched = call("send", "gmail")
    decision, grant = gateway.authorize(b, mismatched, now=NOW, nonce="n10")
    assert grant is None


def test_unknown_tool_refused(runtime, gateway):
    # secrets_db has no mock tool registered; even a valid grant cannot execute it.
    b = bundle("read", "secrets_db", rights=["read"],
               confirmations=[{"subject": "agent", "object": "secrets_db", "action": "read", "issuer": "trusted_user"}])
    c = call("read", "secrets_db")
    decision, grant = gateway.authorize(b, c, now=NOW, nonce="n11")
    assert decision.decision == "ALLOW"
    ex = runtime.execute(grant, c, now=NOW)
    assert ex.executed is False
    assert ex.refused_reason == "unknown_tool"
