"""Phase 8c — gated mock tool runtime (the enforcement boundary).

Up to Phase 8b the service was an *evaluator*: it returned ALLOW/DENY/ESCALATE but never
gated anything. Phase 8c adds the component that actually holds the (mock) tools and that
**refuses to execute unless presented with a fresh, action-bound, single-use grant**
signed by the gateway. This is what turns a decision into enforcement.

Trust model: the tool runtime does NOT trust the LLM, the caller, or even a bare
``Decision`` object — any of those could be fabricated. It trusts only an
``ExecutionGrant`` whose HMAC it can verify with the shared gateway↔runtime key. The
grant is:

* **action-bound** — it carries a hash of the exact (subject, action, object, args), so a
  grant for "draft email X" cannot execute "send email Y";
* **time-bound** — it expires (default 30s TTL), so a leaked grant is not durable;
* **single-use** — its nonce is consumed on execution, so it cannot be replayed.

Everything fails closed: no grant, a bad signature, an expired/replayed grant, an
action-binding mismatch, or an unknown tool all refuse execution.

This is a mock: tools return fake results and the gateway↔runtime auth is a shared
symmetric secret. No real Gmail/Slack/file/browser side effects occur.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Callable, Optional

from pydantic import BaseModel, Field

from . import crypto
from .core import CapabilityTransformer
from .schema import CapabilityBundle, Decision, Object, Right, Subject
from .util import aware

# Shared symmetric secret authenticating the gateway to the tool runtime. In production
# this is a credential the tool service uses to trust the gateway (or an asymmetric key).
RUNTIME_SECRET = "demo-runtime-secret::gateway<->tool::do-not-use-in-production"
DEFAULT_GRANT_TTL_SECONDS = 30


class ToolCall(BaseModel):
    """A concrete tool invocation: the typed action plus its arguments."""

    subject: Subject
    action: Right
    object: Object
    args: dict = Field(default_factory=dict)


class ExecutionGrant(BaseModel):
    """A signed, fresh authorization to execute exactly one tool call."""

    subject: Subject
    action: Right
    object: Object
    action_hash: str
    nonce: str
    issued_at: datetime
    expires_at: datetime
    decision_id: str
    signature: str


class ToolExecution(BaseModel):
    """The result of attempting to run a tool through the gated runtime."""

    executed: bool
    tool: Optional[str] = None
    result: Optional[dict] = None
    refused_reason: Optional[str] = None


def compute_action_hash(call: ToolCall) -> str:
    """Stable hash binding a grant to the exact tool call (incl. args)."""
    blob = json.dumps(
        {"subject": call.subject, "action": call.action, "object": call.object, "args": call.args},
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(blob.encode()).hexdigest()


def _grant_payload(g: ExecutionGrant) -> str:
    """Canonical signed payload of a grant (excludes the signature itself)."""
    return json.dumps(
        {
            "subject": g.subject,
            "action": g.action,
            "object": g.object,
            "action_hash": g.action_hash,
            "nonce": g.nonce,
            "issued_at": aware(g.issued_at).isoformat(),
            "expires_at": aware(g.expires_at).isoformat(),
            "decision_id": g.decision_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


# --------------------------------------------------------------------------------------
# Mock tools. Each returns a fake result; NONE performs a real side effect.
# --------------------------------------------------------------------------------------
def _mock_gmail_draft(args: dict) -> dict:
    return {"draft_id": "draft-mock-1", "to": args.get("to"), "body": args.get("body")}


def _mock_gmail_send(args: dict) -> dict:
    return {"sent": True, "to": args.get("to"), "message_id": "msg-mock-1"}


def _mock_file_read(args: dict) -> dict:
    return {"path": args.get("path"), "content": "<mock file contents>"}


def _mock_file_delete(args: dict) -> dict:
    return {"deleted": True, "path": args.get("path")}


def _mock_calendar_read(args: dict) -> dict:
    return {"events": [{"title": "Mock standup", "at": "2026-06-22T09:00:00Z"}]}


def _mock_slack_post(args: dict) -> dict:
    return {"posted": True, "channel": args.get("channel"), "ts": "ts-mock-1"}


DEFAULT_TOOLS: dict[tuple[str, str], Callable[[dict], dict]] = {
    ("gmail", "draft"): _mock_gmail_draft,
    ("gmail", "send"): _mock_gmail_send,
    ("file", "read"): _mock_file_read,
    ("file", "delete"): _mock_file_delete,
    ("calendar", "read"): _mock_calendar_read,
    ("slack", "post"): _mock_slack_post,
}


class GrantIssuer:
    """Issues execution grants. Only the gateway holds the runtime secret."""

    def __init__(self, secret: str = RUNTIME_SECRET):
        self.secret = secret

    def issue(
        self,
        call: ToolCall,
        *,
        now: datetime,
        nonce: str,
        decision_id: str,
        ttl_seconds: int = DEFAULT_GRANT_TTL_SECONDS,
    ) -> ExecutionGrant:
        now = aware(now)
        action_hash = compute_action_hash(call)
        grant = ExecutionGrant(
            subject=call.subject,
            action=call.action,
            object=call.object,
            action_hash=action_hash,
            nonce=nonce,
            issued_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
            decision_id=decision_id,
            signature="",
        )
        signature = crypto.hmac_sign(self.secret, _grant_payload(grant))
        return grant.model_copy(update={"signature": signature})


class GatedToolRuntime:
    """Holds the (mock) tools and executes a call ONLY for a fresh, valid grant.

    Fail-closed: every verification failure refuses execution with a reason code and runs
    no tool. The nonce store enforces single-use grants (replay protection).
    """

    def __init__(self, secret: str = RUNTIME_SECRET, tools: dict | None = None):
        self.secret = secret
        self.tools = dict(DEFAULT_TOOLS if tools is None else tools)
        self._used_nonces: set[str] = set()

    def _refuse(self, reason: str) -> ToolExecution:
        return ToolExecution(executed=False, refused_reason=reason)

    def execute(
        self,
        grant: Optional[ExecutionGrant],
        call: ToolCall,
        *,
        now: datetime,
    ) -> ToolExecution:
        now = aware(now)

        # 1. There must be a grant at all (a DENY/ESCALATE yields no grant).
        if grant is None:
            return self._refuse("no_grant")

        # 2. The grant signature must verify under the shared runtime key.
        if not crypto.hmac_verify(self.secret, _grant_payload(grant), grant.signature):
            return self._refuse("grant_signature_invalid")

        # 3. The grant must be fresh (issued in the past, not yet expired).
        if now >= aware(grant.expires_at) or now < aware(grant.issued_at):
            return self._refuse("grant_expired")

        # 4. The grant must be bound to exactly this call (typed fields + arg hash).
        if (grant.subject, grant.action, grant.object) != (call.subject, call.action, call.object):
            return self._refuse("action_binding_mismatch")
        if grant.action_hash != compute_action_hash(call):
            return self._refuse("action_binding_mismatch")

        # 5. Single-use: a consumed nonce cannot be replayed.
        if grant.nonce in self._used_nonces:
            return self._refuse("grant_replayed")

        # 6. The tool must exist.
        tool = self.tools.get((call.object, call.action))
        if tool is None:
            return self._refuse("unknown_tool")

        # All checks passed: consume the grant and run the (mock) tool.
        self._used_nonces.add(grant.nonce)
        result = tool(call.args)
        return ToolExecution(executed=True, tool=f"{call.object}.{call.action}", result=result)


class ToolGateway:
    """Convenience facade: evaluate a bundle and, on ALLOW, issue a bound grant.

    Mirrors a real deployment where the policy gateway and the tool runtime are separate
    trust domains; here they share a secret so the demo runs in one process.
    """

    def __init__(self, engine: CapabilityTransformer | None = None, issuer: GrantIssuer | None = None):
        self.engine = engine or CapabilityTransformer()
        self.issuer = issuer or GrantIssuer()

    def authorize(
        self,
        bundle: CapabilityBundle,
        call: ToolCall,
        *,
        now: datetime | None = None,
        nonce: str | None = None,
        ttl_seconds: int = DEFAULT_GRANT_TTL_SECONDS,
    ) -> tuple[Decision, Optional[ExecutionGrant]]:
        now = aware(now) if now is not None else datetime.now(timezone.utc)
        nonce = nonce if nonce is not None else uuid.uuid4().hex

        # Phase 8d: bind the evaluation to the concrete action so action-bound confirmations
        # are checked against this exact call. Set action_hash unless the caller already did.
        if bundle.action_hash is None:
            bundle = bundle.model_copy(update={"action_hash": compute_action_hash(call)})

        decision = self.engine.evaluate(bundle)

        # Only ALLOW yields a grant, and only if the call matches the evaluated action.
        bound = (bundle.subject, bundle.action, bundle.object) == (call.subject, call.action, call.object)
        if decision.decision != "ALLOW" or not bound:
            return decision, None

        decision_id = sha256(f"{compute_action_hash(call)}:{nonce}".encode()).hexdigest()[:16]
        grant = self.issuer.issue(call, now=now, nonce=nonce, decision_id=decision_id, ttl_seconds=ttl_seconds)
        return decision, grant
