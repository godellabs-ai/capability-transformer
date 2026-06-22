"""Typed input/output schemas (Pydantic v2).

Enum fields use ``Literal`` types so that unknown subjects/objects/actions/issuers/
provenance values fail validation (HTTP 422) rather than being silently allowed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Subject = Literal["user", "agent", "document", "tool_result", "system"]
Object = Literal["gmail", "calendar", "file", "browser", "slack", "secrets_db"]
Right = Literal["read", "write", "draft", "send", "invoke", "delegate", "delete", "post"]
Issuer = Literal["trusted_user", "system", "document", "web_page", "tool_output", "model_generated"]
Provenance = Literal[
    "trusted_user",
    "system_policy",
    "retrieved_doc",
    "email_body",
    "web_page",
    "tool_output",
    "model_generated",
]
DecisionValue = Literal["ALLOW", "DENY", "ESCALATE"]


class Capability(BaseModel):
    """An explicit, (mock-)unforgeable authority token."""

    id: str
    subject: Subject
    object: Object
    rights: list[Right]
    issuer: Issuer
    expires_at: datetime
    scope: dict[str, Any] = Field(default_factory=dict)
    delegatable: bool = False


class Revocation(BaseModel):
    """Revokes a capability by id, or by (subject, object) field match."""

    capability_id: Optional[str] = None
    subject: Optional[Subject] = None
    object: Optional[Object] = None


class Confirmation(BaseModel):
    """A trusted human-in-the-loop confirmation for a high-risk action."""

    subject: Subject
    object: Object
    action: Right
    issuer: Issuer


class CapabilityBundle(BaseModel):
    """The full evaluation request: the action plus the authority context."""

    subject: Subject
    action: Right
    object: Object
    source_provenance: Provenance

    capabilities: list[Capability] = Field(default_factory=list)
    revocations: list[Revocation] = Field(default_factory=list)
    confirmations: list[Confirmation] = Field(default_factory=list)

    # For action == "delegate": the right being granted and the grantee.
    delegate_right: Optional[Right] = None
    delegate_to: Optional[Subject] = None

    scope: dict[str, Any] = Field(default_factory=dict)
    # Optional deterministic time override for expiry evaluation.
    now: Optional[datetime] = None


class HeadTrace(BaseModel):
    name: str
    passed: bool
    matched_capability_ids: list[str] = Field(default_factory=list)
    reason: Optional[str] = None


class Trace(BaseModel):
    matched_capabilities: list[str] = Field(default_factory=list)
    passed_heads: list[str] = Field(default_factory=list)
    failed_heads: list[str] = Field(default_factory=list)
    heads: list[HeadTrace] = Field(default_factory=list)
    request: dict[str, Any] = Field(default_factory=dict)
    engine: str = "hard-attention-v1"
    softmax_used: bool = False
    trained: bool = False


class Decision(BaseModel):
    decision: DecisionValue
    reasons: list[str]
    trace: Trace
