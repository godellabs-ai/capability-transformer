"""Phase 8e — hash-chained, tamper-evident audit log."""

from datetime import datetime, timezone

import pytest

from capability_transformer import AuditLog, Capability, CapabilityBundle, ToolCall
from capability_transformer.runtime import GatedToolRuntime, ToolGateway

FUTURE = datetime(2099, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)


def cap(rights, object="gmail"):
    return Capability(id="cap1", subject="agent", object=object, rights=list(rights),
                      issuer="trusted_user", expires_at=FUTURE)


def call(action="draft", object="gmail", args=None):
    return ToolCall(subject="agent", action=action, object=object, args=args or {"to": "bob"})


def bundle(action="draft", object="gmail", provenance="trusted_user", rights=None, confirmations=None):
    return CapabilityBundle(subject="agent", action=action, object=object,
                            source_provenance=provenance, capabilities=[cap(rights or [action], object)],
                            confirmations=confirmations or [])


@pytest.fixture
def wired():
    log = AuditLog()
    return log, ToolGateway(audit_log=log), GatedToolRuntime(audit_log=log)


def _populate(wired):
    log, gateway, runtime = wired
    # denied authorization
    gateway.authorize(bundle("send", provenance="retrieved_doc", rights=["send"]),
                      call("send"), now=NOW, nonce="n1")
    # allowed authorization + grant + execution
    _, grant = gateway.authorize(bundle("draft"), call("draft"), now=NOW, nonce="n2")
    runtime.execute(grant, call("draft"), now=NOW)
    return log


def test_valid_log_verifies(wired):
    log = _populate(wired)
    assert log.verify().ok is True
    assert len(log) == 4  # deny, allow, grant_minted, execute_allow


def test_denied_authorization_is_logged(wired):
    log = _populate(wired)
    assert log.events()[0].event_type == "authorize_deny"
    assert "data_has_no_authority" in log.events()[0].reasons


def test_successful_execution_is_logged(wired):
    log = _populate(wired)
    types = [e.event_type for e in log.events()]
    assert "grant_minted" in types
    assert "execute_allow" in types


def test_failed_execution_is_logged(wired):
    log, gateway, runtime = wired
    # No grant -> runtime refuses -> grant_rejected event.
    runtime.execute(None, call("draft"), now=NOW)
    assert log.events()[-1].event_type == "grant_rejected"
    assert log.events()[-1].reasons == ["no_grant"]


def test_tampered_decision_fails_verification(wired):
    log = _populate(wired)
    log._events[0].decision = "ALLOW"  # was DENY
    result = log.verify()
    assert result.ok is False
    assert result.broken_at == 0
    assert result.reason == "current_hash_mismatch"


def test_tampered_action_hash_fails_verification(wired):
    log = _populate(wired)
    log._events[1].action_hash = "0" * 64
    result = log.verify()
    assert result.ok is False
    assert result.reason == "current_hash_mismatch"


def test_tampered_reasons_fails_verification(wired):
    log = _populate(wired)
    log._events[0].reasons = ["allowed"]
    assert log.verify().ok is False


def test_removed_middle_event_fails_verification(wired):
    log = _populate(wired)
    events = log.events()
    del events[1]  # drop the allow event; grant_minted.previous_hash now dangles
    result = log.verify(events)
    assert result.ok is False
    assert result.reason == "previous_hash_mismatch"


def test_reordered_events_fail_verification(wired):
    log = _populate(wired)
    events = log.events()
    events[1], events[2] = events[2], events[1]
    result = log.verify(events)
    assert result.ok is False


def test_changed_previous_hash_fails_verification(wired):
    log = _populate(wired)
    log._events[2].previous_hash = "0" * 64
    result = log.verify()
    assert result.ok is False
    assert result.broken_at == 2
    assert result.reason == "previous_hash_mismatch"


def test_events_carry_versions_and_no_raw_args(wired):
    log = _populate(wired)
    e = log.events()[1]
    assert e.policy_version and e.compiled_matrix_version
    # args are stored only as a hash; the raw recipient/body never appears.
    dumped = e.model_dump()
    assert "args" not in dumped
    assert e.args_hash and len(e.args_hash) == 64


def test_get_by_event_id(wired):
    log = _populate(wired)
    eid = log.events()[2].event_id
    assert log.get(eid).event_type == "grant_minted"
    assert log.get("nope") is None
