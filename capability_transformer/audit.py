"""Phase 8e — tamper-evident, hash-chained audit log.

Every authorization decision, grant mint, grant rejection, and tool execution is recorded
as an `AuditEvent` linked to the previous event by hash:

    current_hash = SHA256(canonical_json(event_without_current_hash))

where `event_without_current_hash` includes `previous_hash`. Any modification, deletion,
reordering, or previous-hash edit breaks the chain and is caught by `verify()`.

Privacy: the log stores **hashes** of sensitive material (args/body) and the trace, never
the raw payloads or any secret/key material.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from typing import Literal, Optional

from pydantic import BaseModel, Field

from . import compiled_weights as W

GENESIS_HASH = "0" * 64

EventType = Literal[
    "authorize_allow",
    "authorize_deny",
    "authorize_escalate",
    "grant_minted",
    "execute_allow",
    "execute_deny",
    "grant_rejected",
]

# Refusal reasons that indicate the grant itself was invalid (vs. an unknown tool).
_GRANT_REJECTION_REASONS = {
    "no_grant",
    "grant_signature_invalid",
    "grant_expired",
    "grant_replayed",
    "action_binding_mismatch",
}


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return sha256(text.encode()).hexdigest()


def hash_payload(obj) -> str:
    """SHA-256 over the canonical JSON of an object (args, trace, etc.)."""
    return _sha(canonical_json(obj))


class AuditEvent(BaseModel):
    event_id: str
    event_type: EventType
    timestamp: datetime
    subject: Optional[str] = None
    object: Optional[str] = None
    action: Optional[str] = None
    args_hash: Optional[str] = None
    action_hash: Optional[str] = None
    decision: Optional[str] = None
    reasons: list[str] = Field(default_factory=list)
    trace_hash: Optional[str] = None
    nonce: Optional[str] = None
    grant_decision_id: Optional[str] = None
    policy_version: str = W.POLICY_VERSION
    compiled_matrix_version: str = W.MATRIX_VERSION
    previous_hash: str = GENESIS_HASH
    current_hash: str = ""

    def compute_hash(self) -> str:
        """Hash of every field except `current_hash` (includes `previous_hash`)."""
        payload = self.model_dump(exclude={"current_hash"}, mode="json")
        return _sha(canonical_json(payload))


class VerificationResult(BaseModel):
    ok: bool
    length: int
    broken_at: Optional[int] = None
    event_id: Optional[str] = None
    reason: Optional[str] = None


class AuditLog:
    """An append-only, hash-chained log with an in-memory sink (+ optional JSONL file)."""

    def __init__(self, jsonl_path: Optional[str] = None):
        self._events: list[AuditEvent] = []
        self.jsonl_path = jsonl_path

    # -- writing -----------------------------------------------------------------------
    def record(
        self,
        event_type: EventType,
        *,
        timestamp: Optional[datetime] = None,
        subject: Optional[str] = None,
        object: Optional[str] = None,
        action: Optional[str] = None,
        args_hash: Optional[str] = None,
        action_hash: Optional[str] = None,
        decision: Optional[str] = None,
        reasons: Optional[list[str]] = None,
        trace_hash: Optional[str] = None,
        nonce: Optional[str] = None,
        grant_decision_id: Optional[str] = None,
    ) -> AuditEvent:
        ts = timestamp if timestamp is not None else datetime.now(timezone.utc)
        prev = self._events[-1].current_hash if self._events else GENESIS_HASH
        event = AuditEvent(
            event_id=f"evt-{len(self._events):06d}",
            event_type=event_type,
            timestamp=ts,
            subject=subject,
            object=object,
            action=action,
            args_hash=args_hash,
            action_hash=action_hash,
            decision=decision,
            reasons=reasons or [],
            trace_hash=trace_hash,
            nonce=nonce,
            grant_decision_id=grant_decision_id,
            previous_hash=prev,
        )
        event.current_hash = event.compute_hash()
        self._events.append(event)
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(canonical_json(event.model_dump(mode="json")) + "\n")
        return event

    # -- reading -----------------------------------------------------------------------
    def events(self) -> list[AuditEvent]:
        return list(self._events)

    def get(self, event_id: str) -> Optional[AuditEvent]:
        for e in self._events:
            if e.event_id == event_id:
                return e
        return None

    def __len__(self) -> int:
        return len(self._events)

    # -- verification ------------------------------------------------------------------
    def verify(self, events: Optional[list[AuditEvent]] = None) -> VerificationResult:
        """Recompute the chain. Catches modification, deletion, reorder, hash edits."""
        chain = self._events if events is None else events
        prev = GENESIS_HASH
        for i, e in enumerate(chain):
            if e.previous_hash != prev:
                return VerificationResult(ok=False, length=len(chain), broken_at=i,
                                          event_id=e.event_id, reason="previous_hash_mismatch")
            if e.compute_hash() != e.current_hash:
                return VerificationResult(ok=False, length=len(chain), broken_at=i,
                                          event_id=e.event_id, reason="current_hash_mismatch")
            prev = e.current_hash
        return VerificationResult(ok=True, length=len(chain))


# Helpers mapping a runtime/gateway outcome to an event type --------------------------
def authorize_event_type(decision: str) -> EventType:
    return {
        "ALLOW": "authorize_allow",
        "DENY": "authorize_deny",
        "ESCALATE": "authorize_escalate",
    }[decision]


def execution_event_type(executed: bool, refused_reason: Optional[str]) -> EventType:
    if executed:
        return "execute_allow"
    if refused_reason in _GRANT_REJECTION_REASONS:
        return "grant_rejected"
    return "execute_deny"
