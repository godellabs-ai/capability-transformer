"""CompiledCapabilityTransformer — the analytically compiled transformer-style evaluator.

This evaluator shares the tokenization front-end with the reference
``CapabilityTransformer`` (the same crypto/scope/delegation verification produces the same
token features), but it computes the authorization **decision** entirely inside the
compiled transformer-style forward pass (``transformer_model.forward``). It does **not**
call the reference reducer.

Reasons are synthesized from the model's residual evidence (per-capability predicate
aggregates and the output-token decision bits); they are *comparable* to the reference
reasons, while the decision is *equivalent* (proved by ``tests/test_compiled_equivalence``).
"""

from __future__ import annotations

import numpy as np

from . import compiled_weights as W
from . import compiler, tokenizer, transformer_model
from .schema import CapabilityBundle, Decision, Trace


class CompiledCapabilityTransformer:
    """Run authorization as a deterministic transformer-style forward pass.

    Accepts the same configuration as the reference engine. Weights are analytically
    constructed once at construction time and reused for every request.
    """

    def __init__(self, *, keyring=None, require_signatures: bool = False,
                 require_bound_confirmations: bool = False):
        self.keyring = keyring
        self.require_signatures = require_signatures
        self.require_bound_confirmations = require_bound_confirmations
        self.model = compiler.compile_policy({
            "require_signatures": require_signatures,
            "require_bound_confirmations": require_bound_confirmations,
        })

    # ----------------------------------------------------------------------------------
    def forward(self, bundle: CapabilityBundle) -> transformer_model.ForwardResult:
        enc = tokenizer.encode(
            bundle, keyring=self.keyring,
            require_signatures=self.require_signatures,
            require_bound_confirmations=self.require_bound_confirmations,
        )
        R, idx = compiler.build_residual(self.model, enc, bundle)
        result = transformer_model.forward(self.model, R, idx)
        result.evidence["_enc"] = enc
        return result

    def evaluate(self, bundle: CapabilityBundle) -> Decision:
        result = self.forward(bundle)
        decision = result.decision
        reasons = self._reasons(result)
        trace = self._trace(bundle, result, decision, reasons)
        return Decision(decision=decision, reasons=reasons, trace=trace)

    # ----------------------------------------------------------------------------------
    def _cap_or(self, R: np.ndarray, idx: dict, slot: str) -> bool:
        caps = idx["capabilities"]
        if not caps:
            return False
        col = self.model.layout.index(slot)
        return bool(np.any(R[caps, col] > 0.5))

    def _feat_or(self, R: np.ndarray, idx: dict, off: int) -> bool:
        caps = idx["capabilities"]
        if not caps:
            return False
        return bool(np.any(R[caps, off] > 0.5))

    def _reasons(self, result: transformer_model.ForwardResult) -> list[str]:
        ev = result.evidence
        R, idx = result.residual, result.idx
        if result.decision == "ALLOW":
            return ["allowed"]
        if result.decision == "ESCALATE":
            return ["confirmation_required"]

        # DENY: synthesize comparable reason codes from the residual evidence.
        reasons: list[str] = []
        caps = idx["capabilities"]
        if ev.get("has_match", 0.0) <= 0.5:
            if not caps:
                reasons.append("missing_capability")
            else:
                checks = [
                    ("subject_match", "subject_mismatch", False),
                    ("object_match", "object_mismatch", False),
                    ("right_match", "right_not_granted", False),
                    ("issuer_trusted", "issuer_not_trusted", False),
                ]
                for slot, code, _ in checks:
                    if not self._cap_or(R, idx, slot):
                        reasons.append(code)
                if not self._feat_or(R, idx, W.EXPIRY_OFF):
                    reasons.append("expired_capability")
                if not self._cap_or(R, idx, "not_revoked"):
                    reasons.append("revoked_capability")
                if self.require_signatures and not self._feat_or(R, idx, W.SIG_OFF):
                    reasons.append("invalid_signature")
                if not reasons:
                    reasons.append("missing_capability")
        if ev.get("prov_ok", 1.0) <= 0.5:
            reasons.append("data_has_no_authority")
        if ev.get("is_delegation", 0.0) > 0.5 and ev.get("delegation_pass", 0.0) <= 0.5:
            reasons.append("delegation_not_allowed")
        if ev.get("scope_ok", 1.0) <= 0.5:
            reasons.append("scope_violation")
        # dedupe, preserve order
        seen, out = set(), []
        for r in reasons:
            if r not in seen:
                seen.add(r); out.append(r)
        return out or ["missing_capability"]

    def _trace(self, bundle, result, decision, reasons) -> Trace:
        ev = {k: v for k, v in result.evidence.items()
              if not k.startswith("_") and isinstance(v, float)}
        return Trace(
            matched_capabilities=[],
            passed_heads=[k for k, v in ev.items() if v > 0.5],
            failed_heads=[k for k, v in ev.items() if v <= 0.5],
            heads=[],
            request={
                "subject": bundle.subject, "action": bundle.action,
                "object": bundle.object, "source_provenance": bundle.source_provenance,
                "high_risk": ev.get("o_high_risk", 0.0) > 0.5 or ev.get("high_risk_action", 0.0) > 0.5,
            },
            signature={"required": self.require_signatures},
            delegation={},
            engine="compiled-transformer-v1",
            softmax_used=False,
            trained=False,
        )
