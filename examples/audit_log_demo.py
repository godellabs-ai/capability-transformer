"""Phase 8e demo — tamper-evident, hash-chained audit log.

Run:  PYTHONPATH=. python examples/audit_log_demo.py

Shows: a denied injected action, a valid authorization, a grant mint, a tool execution,
verification passing — then a simulated tamper that makes verification fail.
"""

from datetime import datetime, timezone

from capability_transformer import (
    AuditLog,
    Capability,
    CapabilityBundle,
    ToolCall,
)
from capability_transformer.runtime import GatedToolRuntime, ToolGateway

FUTURE = datetime(2099, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)


def cap(rights, object="gmail"):
    return Capability(id="cap1", subject="agent", object=object, rights=list(rights),
                      issuer="trusted_user", expires_at=FUTURE)


def main() -> None:
    log = AuditLog()
    gateway = ToolGateway(audit_log=log)
    runtime = GatedToolRuntime(audit_log=log)

    # 1. Prompt-injected document tries to send mail -> DENY (logged).
    inj = CapabilityBundle(subject="agent", action="send", object="gmail",
                           source_provenance="retrieved_doc", capabilities=[cap(["send"])])
    send_call = ToolCall(subject="agent", action="send", object="gmail", args={"to": "x"})
    gateway.authorize(inj, send_call, now=NOW, nonce="n1")

    # 2 + 3. Valid draft authorization -> ALLOW + grant minted (both logged).
    ok = CapabilityBundle(subject="agent", action="draft", object="gmail",
                          source_provenance="trusted_user", capabilities=[cap(["draft"])])
    draft_call = ToolCall(subject="agent", action="draft", object="gmail",
                          args={"to": "bob@example.com", "body": "hi"})
    _, grant = gateway.authorize(ok, draft_call, now=NOW, nonce="n2")

    # 4. Execute the tool with the grant (logged).
    runtime.execute(grant, draft_call, now=NOW)

    print("audit events:")
    for e in log.events():
        print(f"  {e.event_id}  {e.event_type:18} {e.action:6} dec={e.decision} "
              f"hash={e.current_hash[:12]}…")

    # 5. Verification passes on the intact chain.
    print("\nverify (intact):", log.verify().model_dump())

    # 6. Simulate tampering: flip a recorded decision in the middle of the chain.
    log.events()[0].decision = "ALLOW"          # the denied event now claims ALLOW
    log._events[0].decision = "ALLOW"            # mutate the actual stored event
    result = log.verify()
    print("verify (tampered):", result.model_dump())


if __name__ == "__main__":
    main()
