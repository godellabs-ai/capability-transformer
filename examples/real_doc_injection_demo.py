"""End-to-end test with a REAL document on disk.

Run:  PYTHONPATH=. python examples/real_doc_injection_demo.py [path/to/doc]

Loads an actual file, treats its text as UNTRUSTED DATA (provenance `retrieved_doc`),
naively extracts the action the document is trying to induce, and shows that the gateway
refuses to let that document drive a side effect — while still allowing the agent to read
/ summarize it. Then it contrasts with the *same* action requested by a trusted user, runs
the full authorize -> execute path, and verifies the tamper-evident audit log.

The mock tools return canned results; the security property under test is provenance/
capability-based, so the document's real content is used to choose the attempted action.
"""

import sys
from datetime import datetime, timezone

from capability_transformer import (
    AuditLog,
    Capability,
    CapabilityBundle,
    FlowContext,
    ToolCall,
)
from capability_transformer.runtime import GatedToolRuntime, ToolGateway

FUTURE = datetime(2099, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)

# Naive "instruction extraction": map document keywords to a typed (object, action).
KEYWORDS = [
    ("wire", "gmail", "send"), ("forward", "gmail", "send"), ("send an email", "gmail", "send"),
    ("send email", "gmail", "send"), ("delete", "file", "delete"), ("post", "slack", "post"),
]


def extract_action(text: str):
    low = text.lower()
    for needle, obj, action in KEYWORDS:
        if needle in low:
            return obj, action, needle
    return "gmail", "send", "(default)"


def cap(rights, obj):
    return Capability(id=f"cap-{obj}", subject="agent", object=obj, rights=list(rights),
                      issuer="trusted_user", expires_at=FUTURE)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "examples/sample_malicious_doc.txt"
    text = open(path, encoding="utf-8").read()
    obj, action, hit = extract_action(text)

    log, flow = AuditLog(), FlowContext()
    gateway = ToolGateway(audit_log=log)
    runtime = GatedToolRuntime(audit_log=log, flow=flow)

    print(f"=== Loaded real document: {path} ({len(text)} bytes) ===")
    print(text.strip()[:280] + ("..." if len(text) > 280 else ""))
    print(f"\nDocument is trying to induce: {obj}.{action}  (matched on {hit!r})\n")

    # 1. The document (untrusted data) tries to drive its induced side effect. The agent
    #    even holds a real capability for it — but the AUTHORITY comes from a document.
    doc_bundle = CapabilityBundle(subject="agent", action=action, object=obj,
                                  source_provenance="retrieved_doc", capabilities=[cap([action], obj)])
    doc_call = ToolCall(subject="agent", action=action, object=obj, args={"from": "document"})
    decision, grant = gateway.authorize(doc_bundle, doc_call, now=NOW, nonce="doc1")
    print(f"1. document-driven {obj}.{action:6} -> {decision.decision} {decision.reasons}")
    ex = runtime.execute(grant, doc_call, now=NOW)
    print(f"   tool execution               -> executed={ex.executed} refused={ex.refused_reason}")

    # 2. Reading / summarizing the same document IS allowed (data may drive a read).
    read_bundle = CapabilityBundle(subject="agent", action="read", object="file",
                                   source_provenance="retrieved_doc", capabilities=[cap(["read"], "file")])
    read_call = ToolCall(subject="agent", action="read", object="file", args={"path": path})
    d2, g2 = gateway.authorize(read_bundle, read_call, now=NOW, nonce="doc2")
    ex2 = runtime.execute(g2, read_call, now=NOW)
    print(f"2. summarize the document       -> {d2.decision}; executed={ex2.executed} taint={ex2.taint}")

    # 3. The SAME action, but genuinely requested by a trusted user, is held for human
    #    confirmation (ESCALATE) rather than silently denied — and ALLOWs once confirmed.
    user_bundle = CapabilityBundle(subject="agent", action=action, object=obj,
                                   source_provenance="trusted_user", capabilities=[cap([action], obj)])
    user_call = ToolCall(subject="agent", action=action, object=obj, args={"from": "user"})
    d3, _ = gateway.authorize(user_bundle, user_call, now=NOW, nonce="doc3")
    print(f"3. trusted-user {obj}.{action:6}    -> {d3.decision} {d3.reasons}")

    confirmed = CapabilityBundle(subject="agent", action=action, object=obj,
                                 source_provenance="trusted_user", capabilities=[cap([action], obj)],
                                 confirmations=[{"subject": "agent", "object": obj,
                                                 "action": action, "issuer": "trusted_user"}])
    d4, g4 = gateway.authorize(confirmed, user_call, now=NOW, nonce="doc4")
    ex4 = runtime.execute(g4, user_call, now=NOW)
    print(f"4. trusted + confirmation       -> {d4.decision}; executed={ex4.executed} result={ex4.result}")

    # 5. The whole episode is in a tamper-evident audit log.
    print(f"\n5. audit log: {len(log)} events, verify -> {log.verify().model_dump()}")
    for e in log.events():
        print(f"   {e.event_id} {e.event_type:18} {e.object}.{e.action} dec={e.decision}")


if __name__ == "__main__":
    main()
