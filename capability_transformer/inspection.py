"""Reviewer-facing inspection utilities for the compiled transformer-style evaluator.

These functions let a reviewer point at the actual Q/K/V projection matrices, the residual
stream layout, the feed-forward Boolean gates, and the output projection, and walk a single
authorization decision from tokens -> attention heads -> residual evidence -> output
projection. Nothing here affects the decision; it is read-only introspection.
"""

from __future__ import annotations

import numpy as np

from . import compiler
from .compiled_core import CompiledCapabilityTransformer
from .transformer_model import CompiledModel


def describe_layout(model: CompiledModel) -> list[dict]:
    """Every named residual slot with its offset and width."""
    L = model.layout
    rows = []
    for name in L.names:
        off, width = L.shape_of(name)
        rows.append({"slot": name, "offset": off, "width": width})
    return rows


def describe_heads(model: CompiledModel) -> list[dict]:
    """Each attention head with its Q/K projection shapes and routing."""
    rows = []
    for h in model.match_heads:
        rows.append({
            "head": h.name, "kind": "match",
            "Wq_shape": tuple(h.Wq.shape), "Wk_shape": tuple(h.Wk.shape),
            "query_set": h.query_set, "key_token": h.key_token,
            "out_slot": h.out_slot, "threshold": h.threshold,
        })
    for p in model.pool_heads:
        rows.append({
            "head": p.name, "kind": "max_pool(exists)",
            "Wk_shape": tuple(p.Wk.shape), "over": p.over,
            "in_slot": p.in_slot, "out_slot": p.out_slot,
        })
    return rows


def describe_gates(model: CompiledModel) -> list[dict]:
    """Each feed-forward Boolean gate with its op, inputs, and weight shapes."""
    rows = []
    for g in model.cap_gates + model.decision_gates:
        rows.append({
            "gate": g.name, "op": g.op, "inputs": g.inputs, "out_slot": g.out_slot,
            "token_set": g.token_set,
            "W1_shape": tuple(g.W1.shape), "W2_shape": tuple(g.W2.shape),
        })
    return rows


def head_matrices(model: CompiledModel, head_name: str) -> dict:
    """Return the actual Q/K projection matrices for a named match head."""
    for h in model.match_heads:
        if h.name == head_name:
            return {"name": h.name, "Wq": h.Wq, "Wk": h.Wk,
                    "query_set": h.query_set, "key_token": h.key_token, "out_slot": h.out_slot}
    raise KeyError(head_name)


def output_projection_matrix(model: CompiledModel) -> np.ndarray:
    return model.output_projection.W_out


def inspect_decision(bundle, *, require_signatures=False, require_bound_confirmations=False) -> dict:
    """Walk one decision end to end: tokens -> heads -> residual evidence -> output."""
    engine = CompiledCapabilityTransformer(
        require_signatures=require_signatures,
        require_bound_confirmations=require_bound_confirmations)
    result = engine.forward(bundle)
    L = engine.model.layout
    R, idx = result.residual, result.idx

    def slot(token, name):
        return float(R[token, L.index(name)])

    per_cap = []
    for c in idx["capabilities"]:
        per_cap.append({
            "subject_match": slot(c, "subject_match"),
            "object_match": slot(c, "object_match"),
            "right_match": slot(c, "right_match"),
            "issuer_trusted": slot(c, "issuer_trusted"),
            "not_revoked": slot(c, "not_revoked"),
            "chain_ok": slot(c, "chain_ok"),
            "atten_ok": slot(c, "atten_ok"),
            "valid_capability": slot(c, "valid_capability"),
        })

    out = idx["output"]
    decision_bits = {
        name: float(R[out, L.index(name)])
        for name in ["has_match", "prov_ok", "scope_ok", "delegation_ok", "confirmed",
                     "o_high_risk", "required_ok",
                     "allow_evidence", "deny_evidence", "escalate_evidence"]
    }
    return {
        "token_order": ["request", "policy", *(f"cap[{i}]" for i in range(len(idx["capabilities"]))),
                        *(f"conf[{i}]" for i in range(len(idx["confirmations"]))), "output"],
        "per_capability_evidence": per_cap,
        "output_decision_evidence": decision_bits,
        "logits": {cls: float(v) for cls, v in
                   zip(engine.model.output_projection.classes, result.logits)},
        "decision": result.decision,
    }


def summary(model: CompiledModel | None = None) -> dict:
    """A compact reviewer summary of the compiled architecture."""
    model = model or compiler.compile_policy()
    L = model.layout
    return {
        "residual_width": L.width,
        "num_match_heads": len(model.match_heads),
        "num_pool_heads": len(model.pool_heads),
        "num_cap_gates": len(model.cap_gates),
        "num_decision_gates": len(model.decision_gates),
        "output_projection_shape": tuple(model.output_projection.W_out.shape),
        "softmax_used": False,
        "trained": False,
    }
