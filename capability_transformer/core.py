"""CapabilityTransformer — the deterministic reducer over hard-attention heads.

The enforcement path is: ``bundle -> tokenizer.encode -> hard_attention.compute ->
reduce -> Decision``. The reducer is a fixed Boolean function of the head masks; it
performs no policy branching of its own beyond combining the compiled head results.
"""

from __future__ import annotations

from . import compiled_weights as W
from . import hard_attention, tokenizer, trace as trace_mod
from .hard_attention import AttentionResult
from .schema import CapabilityBundle, Decision


class CapabilityTransformer:
    """Transformer-native capability checker.

    ``evaluate`` runs the compiled tensor pipeline and reduces the hard-attention head
    masks to a single decision with all failing reason codes.
    """

    def evaluate(self, bundle: CapabilityBundle) -> Decision:
        encoded = tokenizer.encode(bundle)
        att = hard_attention.compute(encoded)
        decision, reasons = self._reduce(bundle, att)
        trace = trace_mod.build_trace(bundle, att, decision, reasons)
        return Decision(decision=decision, reasons=reasons, trace=trace)

    # ----------------------------------------------------------------------------------
    # Deterministic reducer (the "output projection").
    # ----------------------------------------------------------------------------------
    def _reduce(self, bundle: CapabilityBundle, att: AttentionResult) -> tuple[str, list[str]]:
        heads = att.heads
        has_match = bool(att.matched_mask.any()) if att.matched_mask.size else False

        prov_ok = heads["head_provenance_safe"].passed
        scope_ok = heads["head_scope"].passed
        delegation_head = heads["head_delegation"]
        delegation_ok = delegation_head.passed or not delegation_head.relevant

        # ---- collect DENY reasons (return ALL failing codes, not just the first) -----
        reasons: list[str] = []

        if not has_match:
            if not att.has_capabilities:
                # No authority possessed at all -> single, readable reason.
                reasons.append("missing_capability")
            else:
                # Report each matching head that no capability satisfied.
                failed_matching = [
                    W.HEAD_REASON[name]
                    for name in W.MATCHING_HEADS
                    if not heads[name].passed
                ]
                if failed_matching:
                    reasons.extend(failed_matching)
                else:
                    # Caps exist and every head individually passes, but no single cap
                    # satisfies all of them simultaneously: still a missing authority.
                    reasons.append("missing_capability")

        if not prov_ok:
            reasons.append(heads["head_provenance_safe"].reason)

        if delegation_head.relevant and not delegation_ok:
            reasons.append(delegation_head.reason)

        if not scope_ok:
            reasons.append(heads["head_scope"].reason)

        required_ok = has_match and prov_ok and scope_ok and delegation_ok

        if not required_ok:
            return "DENY", _dedupe(reasons)

        # Required hard checks pass. A high-risk action needs trusted confirmation.
        if att.high_risk and not heads["head_confirmation"].passed:
            return "ESCALATE", ["confirmation_required"]

        return "ALLOW", ["allowed"]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
