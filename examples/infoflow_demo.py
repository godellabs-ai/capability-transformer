"""Phase 8f demo — output-side information flow (taint cannot become authority).

Run:  PYTHONPATH=. python examples/infoflow_demo.py

Story: an agent reads an email (a tool output → tainted `email_body`). The email body
says "forward me / wire funds". The agent tries to send mail *influenced by that data*.
The taint propagates into the request's effective provenance, so the gateway denies the
side effect — even after laundering the content through a model-generated rewrite.
"""

from datetime import datetime, timezone

from capability_transformer import Capability, CapabilityBundle, FlowContext, ToolCall, join
from capability_transformer.runtime import GatedToolRuntime, ToolGateway

FUTURE = datetime(2099, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)


def cap(rights, object="gmail"):
    return Capability(id=f"cap-{object}", subject="agent", object=object,
                      rights=list(rights), issuer="trusted_user", expires_at=FUTURE)


def main() -> None:
    flow = FlowContext()
    gateway = ToolGateway()
    runtime = GatedToolRuntime(flow=flow)

    # 1. Read the inbox (allowed). The output is data, tainted `email_body`.
    read_bundle = CapabilityBundle(subject="agent", action="read", object="gmail",
                                   source_provenance="trusted_user", capabilities=[cap(["read"])])
    read_call = ToolCall(subject="agent", action="read", object="gmail", args={"folder": "inbox"})
    _, grant = gateway.authorize(read_bundle, read_call, now=NOW, nonce="r1")
    out = runtime.execute(grant, read_call, now=NOW)
    print(f"1. read gmail        -> executed={out.executed} taint={out.taint} handle={out.result_handle}")

    # 2. The email body says "send money to attacker". The agent forms a send request
    #    influenced by that output. Its effective provenance is the join with the taint.
    eff = flow.effective_provenance("trusted_user", [out.result_handle])
    print(f"2. influenced send   -> effective provenance = {eff}  (was trusted_user)")
    send_bundle = CapabilityBundle(subject="agent", action="send", object="gmail",
                                   source_provenance=eff, capabilities=[cap(["send"])])
    send_call = ToolCall(subject="agent", action="send", object="gmail", args={"to": "attacker"})
    decision, g2 = gateway.authorize(send_bundle, send_call, now=NOW, nonce="s1")
    print(f"   gateway decision  -> {decision.decision} {decision.reasons}")

    # 3. Laundering: the agent rewrites the content (model_generated) and retries. Joining
    #    more untrusted labels does not recover authority.
    laundered = join(["trusted_user", out.taint, "model_generated"])
    print(f"3. laundered send    -> effective provenance = {laundered} (still untrusted)")
    b3 = CapabilityBundle(subject="agent", action="send", object="gmail",
                          source_provenance=laundered, capabilities=[cap(["send"])])
    d3, _ = gateway.authorize(b3, send_call, now=NOW, nonce="s2")
    print(f"   gateway decision  -> {d3.decision} {d3.reasons}")

    # 4. Contrast: a genuine trusted-user send (no untrusted influence) only ESCALATEs
    #    (needs human confirmation) — the taint is what turned it into a hard DENY.
    b4 = CapabilityBundle(subject="agent", action="send", object="gmail",
                          source_provenance="trusted_user", capabilities=[cap(["send"])])
    d4, _ = gateway.authorize(b4, send_call, now=NOW, nonce="s3")
    print(f"4. uninfluenced send -> {d4.decision} {d4.reasons}")


if __name__ == "__main__":
    main()
