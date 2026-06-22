"""Audit trace rendering.

Heads 1-7 are always reported (they always gate matching/provenance). The remaining
heads (signature, chain, attenuation, confirmation, scope, delegation) are reported only
when they actually participate in the decision, so the common case mirrors the canonical
example trace exactly. Crypto metadata (signature/delegation blocks) contains hashes and
booleans only — never secret material.
"""

from __future__ import annotations

from .hard_attention import AttentionResult
from .schema import CapabilityBundle, HeadTrace, Trace
from .tokenizer import EncodedBundle

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


def _included_head_names(att: AttentionResult, required_ok: bool) -> list[str]:
    names = list(ALWAYS_HEADS)
    # Crypto heads (Phase 8a/8b) are matching-style gates; show them whenever active.
    if att.heads["head_signature_valid"].relevant:
        names.append("head_signature_valid")
    if att.heads["head_chain_valid"].relevant:
        names.append("head_chain_valid")
    if att.heads["head_attenuation_valid"].relevant:
        names.append("head_attenuation_valid")
    # Confirmation only gates once the required hard checks have passed.
    if att.heads["head_confirmation"].relevant and required_ok:
        names.append("head_confirmation")
    if att.heads["head_scope"].relevant:
        names.append("head_scope")
    if att.heads["head_delegation"].relevant:
        names.append("head_delegation")
    return names


def _signature_block(enc: EncodedBundle) -> dict:
    if not enc.require_signatures:
        return {"required": False}
    caps = []
    for meta in enc.cap_meta:
        caps.append({
            "id": meta.get("id"),
            "issuer": meta.get("issuer"),
            "kid": meta.get("kid"),
            "valid": meta.get("signature_valid"),
            "payload_sha256": meta.get("payload_sha256"),
        })
    return {"required": True, "capabilities": caps}


def _delegation_block(enc: EncodedBundle, att: AttentionResult) -> dict:
    chains = []
    for meta in enc.cap_meta:
        if meta.get("delegated"):
            chains.append({
                "capability_id": meta.get("id"),
                "parent_capability_id": meta.get("parent_capability_id"),
                "parent_hash": meta.get("parent_hash"),
                "chain_valid": meta.get("chain_valid"),
                "attenuation_valid": meta.get("attenuation_valid"),
                "failed_restrictions": meta.get("failed_restrictions", []),
                "chain_error": meta.get("chain_error"),
            })
    if not chains:
        return {}
    return {
        "delegation_chain_valid": att.heads["head_chain_valid"].passed,
        "attenuation_valid": att.heads["head_attenuation_valid"].passed,
        "chains": chains,
    }


def build_trace(
    bundle: CapabilityBundle,
    enc: EncodedBundle,
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
        signature=_signature_block(enc),
        delegation=_delegation_block(enc, att),
        engine="hard-attention-v1",
        softmax_used=False,
        trained=False,
    )
