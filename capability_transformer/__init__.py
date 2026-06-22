"""capability-transformer — Attention as Capability Machine.

A deterministic, transformer-native object-capability enforcement gateway.
"""

from .audit import AuditEvent, AuditLog
from .core import CapabilityTransformer
from .infoflow import FlowContext, effective_provenance, is_trusted, join, tool_output_provenance
from .runtime import (
    ExecutionGrant,
    GatedToolRuntime,
    GrantIssuer,
    ToolCall,
    ToolExecution,
    ToolGateway,
)
from .schema import (
    Capability,
    CapabilityBundle,
    Confirmation,
    Decision,
    Revocation,
    Trace,
)

__all__ = [
    "CapabilityTransformer",
    "CapabilityBundle",
    "Capability",
    "Confirmation",
    "Revocation",
    "Decision",
    "Trace",
    "ToolGateway",
    "GatedToolRuntime",
    "GrantIssuer",
    "ExecutionGrant",
    "ToolCall",
    "ToolExecution",
    "AuditLog",
    "AuditEvent",
    "FlowContext",
    "effective_provenance",
    "is_trusted",
    "join",
    "tool_output_provenance",
]

__version__ = "0.1.0"
