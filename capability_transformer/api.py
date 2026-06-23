"""FastAPI gateway.

The gateway returns a *decision only* — it never performs real tool side effects.
Tool execution must happen downstream and only on ``ALLOW``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import compiled_weights as W
from . import crypto, infoflow
from .audit import AuditEvent, AuditLog, VerificationResult
from .core import DemoUnsignedCapabilityTransformer, SecureCapabilityTransformer
from .schema import Provenance
from .runtime import (
    ExecutionGrant,
    GatedToolRuntime,
    ToolCall,
    ToolExecution,
    ToolGateway,
)
from .schema import Capability, CapabilityBundle, Decision

app = FastAPI(
    title="capability-transformer",
    version="0.1.0",
    description="Attention as Capability Machine — transformer-native capability gateway.",
)

# Secure by default: the public API requires signed capabilities and action-bound
# confirmations. Unsigned (label-trust) mode is NOT production security and must be opted
# into explicitly by setting CAPABILITY_TRANSFORMER_DEMO_UNSIGNED=1.
DEMO_UNSIGNED = os.environ.get("CAPABILITY_TRANSFORMER_DEMO_UNSIGNED") == "1"
_engine = DemoUnsignedCapabilityTransformer() if DEMO_UNSIGNED else SecureCapabilityTransformer()

# The policy gateway and the gated tool runtime share a secret so the demo runs in one
# process; in production they are separate trust domains. Both write to one hash-chained
# audit log.
_audit_log = AuditLog()
_tool_gateway = ToolGateway(engine=_engine, audit_log=_audit_log)
_tool_runtime = GatedToolRuntime(audit_log=_audit_log)
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


class AuthorizeRequest(BaseModel):
    bundle: CapabilityBundle
    args: dict = Field(default_factory=dict)


class AuthorizeResponse(BaseModel):
    decision: Decision
    grant: Optional[ExecutionGrant] = None


class ExecuteRequest(BaseModel):
    grant: Optional[ExecutionGrant] = None
    call: ToolCall


class FlowRequest(BaseModel):
    base: Provenance = "trusted_user"
    influences: list[Provenance] = Field(default_factory=list)


class FlowResponse(BaseModel):
    effective_provenance: Provenance
    is_trusted: bool
    authorizes_side_effects: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": W.ENGINE_NAME, "trained": False, "softmax_used": False,
            "policy_version": W.POLICY_VERSION, "compiled_matrix_version": W.MATRIX_VERSION}


@app.post("/evaluate", response_model=Decision)
def evaluate(bundle: CapabilityBundle) -> Decision:
    """Evaluate a request bundle and return a decision + audit trace."""
    return _engine.evaluate(bundle)


@app.post("/authorize", response_model=AuthorizeResponse)
def authorize(req: AuthorizeRequest) -> AuthorizeResponse:
    """Phase 8c: evaluate a bundle and, on ALLOW, issue a fresh action-bound grant.

    The grant (if any) must be presented to ``/execute`` to actually run the tool.
    """
    call = ToolCall(
        subject=req.bundle.subject,
        action=req.bundle.action,
        object=req.bundle.object,
        args=req.args,
    )
    decision, grant = _tool_gateway.authorize(req.bundle, call)
    return AuthorizeResponse(decision=decision, grant=grant)


@app.post("/execute", response_model=ToolExecution)
def execute(req: ExecuteRequest) -> ToolExecution:
    """Phase 8c: run a tool ONLY for a fresh, valid, single-use grant. Fails closed."""
    from datetime import datetime, timezone

    return _tool_runtime.execute(req.grant, req.call, now=datetime.now(timezone.utc))


@app.post("/flow/provenance", response_model=FlowResponse)
def flow_provenance(req: FlowRequest) -> FlowResponse:
    """Phase 8f: join a base provenance with influencing tool-output taints.

    Untrusted taint dominates, so data laundered through tool outputs cannot regain the
    authority to drive side effects.
    """
    eff = infoflow.effective_provenance(req.base, req.influences)
    trusted = infoflow.is_trusted(eff)
    return FlowResponse(effective_provenance=eff, is_trusted=trusted,
                        authorizes_side_effects=trusted)


@app.get("/audit", response_model=list[AuditEvent])
def audit_events() -> list[AuditEvent]:
    """Phase 8e: the full hash-chained audit log (hashes only, no raw payloads)."""
    return _audit_log.events()


@app.get("/audit/verify", response_model=VerificationResult)
def audit_verify() -> VerificationResult:
    """Phase 8e: recompute the chain and report whether it is intact."""
    return _audit_log.verify()


@app.get("/audit/{event_id}", response_model=AuditEvent)
def audit_event(event_id: str) -> AuditEvent:
    event = _audit_log.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="unknown event_id")
    return event


@app.post("/mint", response_model=Capability)
def mint(cap: Capability) -> Capability:
    """Phase 8a: sign a capability with the demo issuer keyring.

    Returns the capability with its ``kid`` and ``signature`` populated. Only trusted
    issuers (``trusted_user``, ``system``) hold keys; minting for any other issuer fails
    with 422.
    """
    try:
        return crypto.issue(cap)
    except KeyError as exc:  # unknown / untrusted issuer has no signing key
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/schema")
def schema() -> dict:
    """Return the bounded vocabularies and the request JSON schema."""
    return {
        "subjects": W.SUBJECTS,
        "objects": W.OBJECTS,
        "rights": W.RIGHTS,
        "issuers": W.ISSUERS,
        "trusted_issuers": ["trusted_user", "system"],
        "provenance": W.PROVENANCE,
        "decisions": W.DECISIONS,
        "reason_codes": W.REASON_CODES,
        "high_risk_actions": [
            "gmail.send",
            "slack.post",
            "file.delete",
            "secrets_db.read",
            "browser.invoke",
        ],
        "grant_refusal_reasons": [
            "no_grant",
            "grant_signature_invalid",
            "grant_expired",
            "action_binding_mismatch",
            "grant_replayed",
            "unknown_tool",
        ],
        "request_schema": CapabilityBundle.model_json_schema(),
    }


@app.get("/examples")
def examples() -> dict:
    """Return the bundled example request bodies."""
    out = {}
    if _EXAMPLES_DIR.is_dir():
        for path in sorted(_EXAMPLES_DIR.glob("*.json")):
            out[path.stem] = json.loads(path.read_text())
    return out


def main() -> None:  # pragma: no cover - manual entry point
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
