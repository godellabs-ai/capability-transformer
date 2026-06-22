"""Phase 8f — output-side information-flow control (taint tracking).

The decision core already enforces *input-side* provenance: a request whose
``source_provenance`` is untrusted data cannot drive a side effect. Phase 8f closes the
loop on the *output* side. Every tool output is **data, never authority**, so it is
labeled with a provenance taint. When a later request is influenced by tool outputs, its
effective provenance is the **join** (least-trusted) of all influences. Untrusted taint
therefore cannot be "laundered" — re-reading, summarizing, or chaining it through more
tools keeps the taint, and the gateway still denies the side effect.

This is a small lattice, not a full IFC type system: two trust levels (trusted control
plane vs. untrusted data) with provenance labels as representatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .schema import Object, Provenance

# Trusted (authority-bearing) provenance vs. untrusted data.
TRUSTED: tuple[Provenance, ...] = ("trusted_user", "system_policy")

# Deterministic priority for choosing a representative label when joining untrusted taints.
# (The choice never changes the *decision* — all untrusted labels are denied authority —
# only which untrusted label is reported.)
_UNTRUSTED_PRIORITY: tuple[Provenance, ...] = (
    "model_generated",
    "web_page",
    "email_body",
    "retrieved_doc",
    "tool_output",
)

# A tool output's taint reflects where its data comes from. Every tool result is data; the
# object gives a more specific untrusted label than the generic ``tool_output``.
TOOL_OUTPUT_PROVENANCE: dict[str, Provenance] = {
    "gmail": "email_body",
    "browser": "web_page",
    "file": "retrieved_doc",
    "slack": "tool_output",
    "calendar": "tool_output",
    "secrets_db": "tool_output",
}


def is_trusted(label: Provenance) -> bool:
    return label in TRUSTED


def tool_output_provenance(obj: Object) -> Provenance:
    """The provenance taint applied to the output of a tool acting on ``obj``."""
    return TOOL_OUTPUT_PROVENANCE.get(obj, "tool_output")


def join(labels: Iterable[Provenance]) -> Provenance:
    """Least-trusted combination of provenance labels (the taint lattice join).

    Any untrusted label dominates (data taint is sticky). With only trusted labels, the
    result is trusted (prefer ``trusted_user``). Empty -> ``trusted_user`` (no influence).
    """
    labels = list(labels)
    if not labels:
        return "trusted_user"
    untrusted = [lbl for lbl in labels if not is_trusted(lbl)]
    if untrusted:
        for candidate in _UNTRUSTED_PRIORITY:
            if candidate in untrusted:
                return candidate
        return untrusted[0]
    return "trusted_user" if "trusted_user" in labels else labels[0]


def effective_provenance(base: Provenance, influences: Iterable[Provenance]) -> Provenance:
    """Provenance the gateway should see, given a base label and influencing taints."""
    return join([base, *influences])


@dataclass
class FlowContext:
    """Tracks the taint of tool outputs so it can propagate into later requests."""

    _taints: dict[str, Provenance] = field(default_factory=dict)

    def register_output(self, handle: str, label: Provenance) -> None:
        self._taints[handle] = label

    def taint_of(self, handle: str) -> Optional[Provenance]:
        return self._taints.get(handle)

    def effective_provenance(self, base: Provenance, influences: Iterable[str]) -> Provenance:
        """Join ``base`` with the taints of the referenced output handles."""
        labels = [self._taints[h] for h in influences if h in self._taints]
        return effective_provenance_from(base, labels)


# Module-level alias so the dataclass method and the free function don't shadow.
effective_provenance_from = effective_provenance
