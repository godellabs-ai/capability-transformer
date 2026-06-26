"""GuardedQwen — an open-weight LLM fused with the frozen capability head.

`GuardedQwen` is a single ``nn.Module`` containing an open-weight causal LM (Qwen2.5) and
the frozen, analytic ``TorchCapabilityHead``. The LM *proposes* a tool call; the head
*decides* ``ALLOW / DENY / ESCALATE`` on the structured action. No training, CPU-only.

The security boundary is the head, and it is **isolated**: it reads only the structured
``(subject, action, object)`` the LM committed to plus an externally-supplied provenance —
never the LM's hidden state or wording. So the model's (in)ability to resist a prompt
injection does not affect the decision; an attacker cannot phrase their way past it.

``GuardKernel`` holds the guard logic with no LM dependency, so it is unit-testable without
loading any weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .infoflow import FlowContext, join, tool_output_provenance
from .schema import Capability, CapabilityBundle
from .torch_head import TorchCapabilityHead

FUTURE = datetime(2099, 1, 1, tzinfo=timezone.utc)

# Map the agent's tools onto the bounded capability universe, and mark which tools ingest
# untrusted data (their outputs taint the session).
DEFAULT_TOOL_MAP = {
    "read_inbox": ("gmail", "read"), "read_email": ("gmail", "read"),
    "search_emails": ("gmail", "read"), "send_email": ("gmail", "send"),
    "draft_email": ("gmail", "draft"), "delete_email": ("gmail", "delete"),
    "read_file": ("file", "read"), "delete_file": ("file", "delete"),
}
INGEST_TOOLS = {"read_inbox", "read_email", "search_emails", "read_file"}


@dataclass
class GuardResult:
    tool: str
    object: str
    action: str
    provenance: str
    decision: str
    reasons: list
    evidence: dict = field(default_factory=dict)


def agent_email_capabilities() -> list[Capability]:
    """A legitimate assistant: it CAN read, draft and send the user's mail."""
    return [Capability(id="cap-gmail", subject="agent", object="gmail",
                       rights=["read", "draft", "send", "delete"], issuer="trusted_user",
                       expires_at=FUTURE)]


def parse_tool_call(text: str) -> Optional[dict]:
    """Extract the first balanced JSON object from the LM output (fail-closed)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
            else:
                if ch == '"': in_str = True
                elif ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start:i + 1])
                            if isinstance(obj, dict) and "tool" in obj:
                                obj.setdefault("args", {})
                                return obj
                        except json.JSONDecodeError:
                            break
        start = text.find("{", start + 1)
    return None


class GuardKernel:
    """The capability guard: maps a tool call to an action and decides on it.

    Stateful only in its *session provenance*: a data-ingesting tool taints the session, so
    later side effects inherit that taint. Provenance is supplied here by the harness — it
    is never read from the language model.
    """

    def __init__(self, *, capabilities=None, tool_map=None, ingest_tools=None,
                 require_signatures=False, require_bound_confirmations=False,
                 base_provenance="trusted_user"):
        self.head = TorchCapabilityHead(require_signatures=require_signatures,
                                        require_bound_confirmations=require_bound_confirmations)
        self.caps = capabilities or agent_email_capabilities()
        self.tool_map = dict(tool_map or DEFAULT_TOOL_MAP)
        self.ingest_tools = set(ingest_tools if ingest_tools is not None else INGEST_TOOLS)
        self.base_provenance = base_provenance
        self.session_provenance = base_provenance
        self.flow = FlowContext()

    def reset(self):
        self.session_provenance = self.base_provenance

    def ingest(self, tool: str):
        """Record that the agent read this tool's (untrusted) output into its context."""
        obj, _ = self.tool_map[tool]
        if tool in self.ingest_tools:
            self.session_provenance = join([self.session_provenance, tool_output_provenance(obj)])
        return self.session_provenance

    def evaluate(self, tool_call: dict, *, confirmations=None, action_hash=None) -> GuardResult:
        tool = tool_call.get("tool")
        if tool not in self.tool_map:
            return GuardResult(tool or "?", "?", "?", self.session_provenance, "DENY",
                               ["unknown_tool"])
        obj, action = self.tool_map[tool]
        bundle = CapabilityBundle(
            subject="agent", action=action, object=obj,
            source_provenance=self.session_provenance, capabilities=self.caps,
            confirmations=confirmations or [], action_hash=action_hash)
        d = self.head.decide(bundle)
        return GuardResult(tool, obj, action, self.session_provenance, d.decision,
                           d.reasons, d.evidence)


# Lazy import of torch.nn so importing this module is cheap when only GuardKernel is used.
def _GuardedQwen_base():
    import torch.nn as nn
    return nn.Module


class GuardedQwen(_GuardedQwen_base()):
    """One module: an open-weight LM that proposes, fused with the head that decides."""

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", *, capabilities=None,
                 require_signatures=False):
        super().__init__()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.lm.eval()
        self.kernel = GuardKernel(capabilities=capabilities, require_signatures=require_signatures)
        # The frozen capability head is a real sub-module of this model.
        self.head = self.kernel.head

    @property
    def session_provenance(self):
        return self.kernel.session_provenance

    def propose(self, messages, max_new_tokens=72) -> dict:
        """Run the LM to propose a tool call. Returns {raw, tool_call}."""
        import torch
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        with torch.no_grad():
            out = self.lm.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=self.tok.eos_token_id)
        raw = self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        return {"raw": raw, "tool_call": parse_tool_call(raw)}
