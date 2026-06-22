"""Audit trace rendering.

Heads 1-7 are always reported (they always gate matching/provenance). Heads 8-10
(confirmation, scope, delegation) are reported only when they actually participate in
the decision, so the common case mirrors the canonical example trace exactly.
"""

from __future__ import annotations

from .hard_attention import AttentionResult
from .schema import CapabilityBundle, HeadTrace, Trace

# Heads that always appear in the trace, in canonical order.
ALWAYS_HEADS = [
    "head_subject_match",
    "head_object_match",
    "head_right_match",
    "head_trusted_issuer",
    "head_not_expired",
    "head_not_revoked",
    "head_provenance_safe",
]
# Heads that appear only when relevant.
CONDITIONAL_HEADS = ["head_confirmation", "head_scope", "head_delegation"]


def _included_head_names(att: AttentionResult, required_ok: bool) -> list[str]:
    names = list(ALWAYS_HEADS)
    # Signature enforcement (Phase 8a) is a matching-style gate; show it next to the
    # other matching heads whenever it is active.
    if att.heads["head_signature_valid"].relevant:
        names.append("head_signature_valid")
    # Confirmation only gates once the required hard checks have passed.
    if att.heads["head_confirmation"].relevant and required_ok:
        names.append("head_confirmation")
    if att.heads["head_scope"].relevant:
        names.append("head_scope")
    if att.heads["head_delegation"].relevant:
        names.append("head_delegation")
    return names


def build_trace(
    bundle: CapabilityBundle,
    att: AttentionResult,
    decision: str,
    reasons: list[str],
) -> Trace:
    has_match = bool(att.matched_mask.any()) if att.matched_mask.size else False
    prov_ok = att.heads["head_provenance_safe"].passed
    scope_ok = att.heads["head_scope"].passed
    deleg = att.heads["head_delegation"]
    delegation_ok = deleg.passed or not deleg.relevant
    required_ok = has_match and prov_ok and scope_ok and delegation_ok

    names = _included_head_names(att, required_ok)

    head_traces: list[HeadTrace] = []
    passed_heads: list[str] = []
    failed_heads: list[str] = []
    for name in names:
        hr = att.heads[name]
        head_traces.append(
            HeadTrace(
                name=name,
                passed=hr.passed,
                matched_capability_ids=hr.matched_cap_ids,
                reason=hr.reason,
            )
        )
        (passed_heads if hr.passed else failed_heads).append(name)

    return Trace(
        matched_capabilities=att.matched_cap_ids,
        passed_heads=passed_heads,
        failed_heads=failed_heads,
        heads=head_traces,
        request={
            "subject": bundle.subject,
            "action": bundle.action,
            "object": bundle.object,
            "source_provenance": bundle.source_provenance,
            "high_risk": att.high_risk,
        },
        engine="hard-attention-v1",
        softmax_used=False,
        trained=False,
    )
