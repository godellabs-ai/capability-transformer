"""LangChain integration — gate a real agent's tools through the capability gateway.

`CapabilityGuard.wrap(tool)` takes a real LangChain ``BaseTool`` and returns a drop-in
``StructuredTool`` that, before running, asks the capability gateway whether the call is
authorized. On ``DENY`` / ``ESCALATE`` the tool is NOT executed; the agent instead
receives a short refusal string (which it sees as the tool observation). On ``ALLOW`` the
real tool runs through the gated runtime (grant + audit + taint).

The guard tracks **session provenance**: a data-ingesting tool (retriever, file/email read,
web fetch) taints the session, and subsequent side-effecting calls are evaluated with that
taint. This is content-agnostic — the gateway never inspects the tool text for "malicious"
content; it denies untrusted *data* the *authority* to drive side effects.

Requires ``langchain-core`` (``pip install '.[langchain]'``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

from langchain_core.tools import BaseTool, StructuredTool

from ..infoflow import FlowContext, join
from ..runtime import GatedToolRuntime, ToolCall, ToolGateway
from ..schema import Capability, CapabilityBundle


class CapabilityGuard:
    """Wraps LangChain tools so every call is checked by the capability gateway."""

    def __init__(
        self,
        *,
        capabilities: Iterable[Capability],
        tool_map: dict[str, tuple[str, str]],   # tool name -> (object, action)
        ingest_tools: Iterable[str] = (),        # tools whose output is untrusted data
        subject: str = "agent",
        base_provenance: str = "trusted_user",
        confirmations: Optional[list[dict]] = None,
        gateway: Optional[ToolGateway] = None,
        runtime: Optional[GatedToolRuntime] = None,
        flow: Optional[FlowContext] = None,
        audit_log=None,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self.flow = flow or FlowContext()
        self.gateway = gateway or ToolGateway(audit_log=audit_log)
        self.runtime = runtime or GatedToolRuntime(audit_log=audit_log, flow=self.flow)
        self.audit_log = self.gateway.audit_log
        self.capabilities = list(capabilities)
        self.tool_map = tool_map
        self.ingest_tools = set(ingest_tools)
        self.subject = subject
        self.base_provenance = base_provenance
        self.confirmations = confirmations or []
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.session_provenance = base_provenance
        self._nonce = 0
        # The provenance that gated the most recent call (handy for assertions/printing).
        self.last_decision = None

    def reset(self) -> None:
        """Reset session taint to the base provenance (start of a new conversation)."""
        self.session_provenance = self.base_provenance
        self.last_decision = None

    def _next_nonce(self) -> str:
        self._nonce += 1
        return f"lc-{self._nonce}"

    def wrap(self, tool: BaseTool) -> StructuredTool:
        if tool.name not in self.tool_map:
            raise KeyError(f"no (object, action) mapping for tool {tool.name!r}")
        obj, action = self.tool_map[tool.name]
        # Register the REAL tool as the runtime's executor for this (object, action), so a
        # granted call runs the genuine capability (not a mock) and is audited.
        self.runtime.tools[(obj, action)] = lambda args, _t=tool: {"output": _t.invoke(args)}
        guard = self

        def _guarded(**kwargs) -> str:
            now = guard.clock()
            call = ToolCall(subject=guard.subject, action=action, object=obj, args=kwargs)
            bundle = CapabilityBundle(
                subject=guard.subject, action=action, object=obj,
                source_provenance=guard.session_provenance,
                capabilities=guard.capabilities, confirmations=guard.confirmations,
            )
            decision, grant = guard.gateway.authorize(bundle, call, now=now, nonce=guard._next_nonce())
            guard.last_decision = decision
            if decision.decision != "ALLOW":
                return (f"[capability-gateway {decision.decision}: "
                        f"{', '.join(decision.reasons)} — tool '{tool.name}' was NOT executed]")
            execution = guard.runtime.execute(grant, call, now=now)
            if not execution.executed:
                return f"[capability-gateway refused execution: {execution.refused_reason}]"
            # Data-ingesting tools taint the session: later side effects inherit it.
            if tool.name in guard.ingest_tools and execution.taint:
                guard.session_provenance = join([guard.session_provenance, execution.taint])
            return str(execution.result.get("output"))

        return StructuredTool.from_function(
            func=_guarded,
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
        )

    def wrap_all(self, tools: Iterable[BaseTool]) -> list[StructuredTool]:
        return [self.wrap(t) for t in tools]
