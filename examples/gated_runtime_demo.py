"""Phase 8c demo — the gated tool runtime is a real enforcement boundary.

Run:  PYTHONPATH=. python examples/gated_runtime_demo.py

Shows that the mock tool only runs for a fresh, valid, action-bound, single-use grant:
  * ALLOW            -> grant issued -> tool executes
  * replay the grant -> refused (single-use)
  * DENY (untrusted) -> no grant     -> tool refuses to run
  * tampered grant   -> refused (signature)
  * expired grant    -> refused (freshness)
"""

from datetime import datetime, timedelta, timezone

from capability_transformer import Capability, CapabilityBundle, ToolCall, crypto
from capability_transformer.runtime import GatedToolRuntime, ToolGateway

FUTURE = datetime(2099, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)


def cap(rights, object="gmail", subject="agent"):
    return Capability(id="cap1", subject=subject, object=object, rights=list(rights),
                      issuer="trusted_user", expires_at=FUTURE)


def bundle(action, object, provenance, rights):
    return CapabilityBundle(subject="agent", action=action, object=object,
                            source_provenance=provenance, capabilities=[cap(rights, object)])


def main() -> None:
    gateway = ToolGateway()
    runtime = GatedToolRuntime()

    # 1. ALLOW -> grant -> execute succeeds.
    b = bundle("draft", "gmail", "trusted_user", ["draft"])
    call = ToolCall(subject="agent", action="draft", object="gmail",
                    args={"to": "bob@example.com", "body": "hi"})
    decision, grant = gateway.authorize(b, call, now=NOW, nonce="nonce-1")
    print("1. allow + grant   -> decision:", decision.decision,
          "| executed:", runtime.execute(grant, call, now=NOW).executed)

    # 2. Replay the same grant -> refused (single-use nonce already consumed).
    rep = runtime.execute(grant, call, now=NOW)
    print("2. replay grant    -> executed:", rep.executed, "| reason:", rep.refused_reason)

    # 3. DENY (untrusted document tries to send) -> no grant -> refuse.
    b_deny = bundle("send", "gmail", "retrieved_doc", ["send"])
    send_call = ToolCall(subject="agent", action="send", object="gmail", args={"to": "x"})
    d2, g2 = gateway.authorize(b_deny, send_call, now=NOW, nonce="nonce-2")
    ex = runtime.execute(g2, send_call, now=NOW)
    print("3. deny -> no grant-> decision:", d2.decision, "| executed:", ex.executed,
          "| reason:", ex.refused_reason)

    # 4. Tampered grant (attacker swaps action draft->send) -> refused (signature).
    forged = grant.model_copy(update={"action": "send", "nonce": "nonce-x"})
    send_as_draft = ToolCall(subject="agent", action="send", object="gmail",
                             args={"to": "bob@example.com", "body": "hi"})
    ex4 = runtime.execute(forged, send_as_draft, now=NOW)
    print("4. tampered grant  -> executed:", ex4.executed, "| reason:", ex4.refused_reason)

    # 5. Expired grant -> refused (freshness).
    decision5, grant5 = gateway.authorize(b, call, now=NOW, nonce="nonce-5", ttl_seconds=30)
    later = NOW + timedelta(seconds=31)
    ex5 = runtime.execute(grant5, call, now=later)
    print("5. expired grant   -> executed:", ex5.executed, "| reason:", ex5.refused_reason)


if __name__ == "__main__":
    main()
