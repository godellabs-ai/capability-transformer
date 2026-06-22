"""Phase 8f — output-side information-flow (taint cannot become authority)."""

from datetime import datetime, timezone

import pytest

from capability_transformer import (
    Capability,
    CapabilityBundle,
    FlowContext,
    ToolCall,
    effective_provenance,
    is_trusted,
    join,
    tool_output_provenance,
)
from capability_transformer.runtime import GatedToolRuntime, ToolGateway

FUTURE = datetime(2099, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)


def cap(rights, object="gmail"):
    return Capability(id=f"cap-{object}", subject="agent", object=object,
                      rights=list(rights), issuer="trusted_user", expires_at=FUTURE)


# ---- lattice -------------------------------------------------------------------------
def test_trusted_labels():
    assert is_trusted("trusted_user")
    assert is_trusted("system_policy")
    assert not is_trusted("email_body")
    assert not is_trusted("model_generated")


def test_join_untrusted_dominates():
    assert not is_trusted(join(["trusted_user", "email_body"]))
    assert join(["trusted_user", "system_policy"]) == "trusted_user"
    assert join([]) == "trusted_user"
    # Any number of trusted labels joined with one untrusted -> untrusted.
    assert not is_trusted(join(["trusted_user", "system_policy", "web_page"]))


def test_join_is_deterministic():
    a = join(["trusted_user", "web_page", "email_body"])
    b = join(["email_body", "trusted_user", "web_page"])
    assert a == b  # order-independent representative


def test_tool_output_provenance_map():
    assert tool_output_provenance("gmail") == "email_body"
    assert tool_output_provenance("browser") == "web_page"
    assert tool_output_provenance("file") == "retrieved_doc"
    assert tool_output_provenance("calendar") == "tool_output"


# ---- propagation through the runtime -------------------------------------------------
@pytest.fixture
def wired():
    flow = FlowContext()
    return flow, ToolGateway(), GatedToolRuntime(flow=flow)


def _read_gmail(gateway, runtime):
    b = CapabilityBundle(subject="agent", action="read", object="gmail",
                         source_provenance="trusted_user", capabilities=[cap(["read"])])
    c = ToolCall(subject="agent", action="read", object="gmail", args={"folder": "inbox"})
    _, grant = gateway.authorize(b, c, now=NOW, nonce="r1")
    return runtime.execute(grant, c, now=NOW)


def test_tool_output_is_tainted(wired):
    _, gateway, runtime = wired
    out = _read_gmail(gateway, runtime)
    assert out.executed is True
    assert out.taint == "email_body"
    assert out.result_handle is not None


def test_taint_propagates_and_blocks_side_effect(wired):
    flow, gateway, runtime = wired
    out = _read_gmail(gateway, runtime)

    # A send influenced by the tainted read inherits the taint -> hard DENY.
    eff = flow.effective_provenance("trusted_user", [out.result_handle])
    assert eff == "email_body"
    b = CapabilityBundle(subject="agent", action="send", object="gmail",
                         source_provenance=eff, capabilities=[cap(["send"])])
    c = ToolCall(subject="agent", action="send", object="gmail", args={"to": "attacker"})
    decision, grant = gateway.authorize(b, c, now=NOW, nonce="s1")
    assert decision.decision == "DENY"
    assert "data_has_no_authority" in decision.reasons
    assert grant is None


def test_uninfluenced_send_only_escalates(wired):
    # Without the untrusted influence the same send merely needs confirmation.
    _, gateway, _ = wired
    b = CapabilityBundle(subject="agent", action="send", object="gmail",
                         source_provenance="trusted_user", capabilities=[cap(["send"])])
    c = ToolCall(subject="agent", action="send", object="gmail", args={"to": "x"})
    decision, _ = gateway.authorize(b, c, now=NOW, nonce="s2")
    assert decision.decision == "ESCALATE"


def test_laundering_does_not_recover_authority(wired):
    flow, gateway, runtime = wired
    out = _read_gmail(gateway, runtime)
    # Re-express the tainted content as model_generated and combine: still untrusted.
    laundered = join(["trusted_user", out.taint, "model_generated"])
    assert not is_trusted(laundered)
    b = CapabilityBundle(subject="agent", action="send", object="gmail",
                         source_provenance=laundered, capabilities=[cap(["send"])])
    c = ToolCall(subject="agent", action="send", object="gmail", args={"to": "x"})
    decision, _ = gateway.authorize(b, c, now=NOW, nonce="s3")
    assert decision.decision == "DENY"
    assert "data_has_no_authority" in decision.reasons


def test_read_of_tainted_data_still_allowed(wired):
    # Taint blocks side effects, not passive reads: summarizing tainted data is fine.
    flow, gateway, runtime = wired
    out = _read_gmail(gateway, runtime)
    eff = flow.effective_provenance("trusted_user", [out.result_handle])
    b = CapabilityBundle(subject="agent", action="read", object="file",
                         source_provenance=eff, capabilities=[cap(["read"], object="file")])
    c = ToolCall(subject="agent", action="read", object="file", args={"path": "/tmp/x"})
    decision, grant = gateway.authorize(b, c, now=NOW, nonce="rd")
    assert decision.decision == "ALLOW"
    assert grant is not None


def test_flow_context_unregistered_handle_is_clean(wired):
    flow, _, _ = wired
    # An unknown handle contributes no taint.
    assert flow.effective_provenance("trusted_user", ["nope"]) == "trusted_user"
