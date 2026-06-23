"""Step-by-step recorder for the compiled transformer-style forward pass.

Re-runs the same deterministic forward pass as ``transformer_model.forward`` but captures,
after **every** operation (embedding, each attention head, each feed-forward gate, each
max-pool aggregation, the output projection), a full snapshot of the residual stream plus
the operation's inputs, scores, and outputs. The result is a JSON-able trace the UI replays
with a scrubber (rewind / step / inspect). The recorded final decision is identical to the
non-recording forward pass and to the reference evaluator.
"""

from __future__ import annotations

import numpy as np

from . import compiled_weights as W
from . import compiler, tokenizer
from .compiled_core import CompiledCapabilityTransformer
from .core import CapabilityTransformer
from .transformer_model import _token_indices

# Friendly labels for raw token feature bits referenced by the gates.
FEAT_LABELS = {
    "feat:EXPIRY_OFF": "not_expired·bit",
    "feat:REVOKED_OFF": "revoked·bit",
    "feat:SIG_OFF": "signature·bit",
    "feat:CHAIN_OFF": "chain·bit",
    "feat:ATTEN_OFF": "attenuation·bit",
    "feat:CBIND_OFF": "conf_bind·bit",
}

LAYER_SCHEDULE = [
    ("embedding", "Embedding", "Tokens are laid into the residual stream."),
    ("attention", "Attention heads", "Exact-match selectors write per-token evidence."),
    ("ffn_per_cap", "Feed-forward (per token)", "Boolean gates compute per-capability validity."),
    ("pool", "Max-pool (∃)", "Existential aggregation onto the output token."),
    ("ffn_decision", "Feed-forward (decision)", "Decision gates on the output token."),
    ("output", "Output projection", "Argmax over [ALLOW, DENY, ESCALATE]."),
]


def _coord(L, name: str) -> int:
    if name.startswith("feat:"):
        return getattr(W, name.split(":", 1)[1])
    return L.index(name)


def _decode_token(R_row, role: str) -> dict:
    """Human-readable decode of a token's feature region (offsets in [0:48])."""
    def onehot(slot, vocab):
        seg = R_row[slice(*W.SLOT[slot])]
        i = int(np.argmax(seg)) if seg.sum() > 0 else None
        return vocab[i] if i is not None and seg[i] > 0.5 else None

    def multihot(slot, vocab):
        seg = R_row[slice(*W.SLOT[slot])]
        return [vocab[i] for i in range(len(vocab)) if seg[i] > 0.5]

    ttype = onehot("type", W.TOKEN_TYPES)
    fields = {}
    if role in ("request", "capability", "confirmation"):
        fields = {
            "subject": onehot("subject", W.SUBJECTS),
            "object": onehot("object", W.OBJECTS),
            "rights/action": multihot("rights", W.RIGHTS),
            "issuer": onehot("issuer", W.ISSUERS),
            "provenance": onehot("provenance", W.PROVENANCE),
        }
        bits = {b: int(R_row[getattr(W, off)] > 0.5) for b, off in
                [("expiry_ok", "EXPIRY_OFF"), ("revoked", "REVOKED_OFF"),
                 ("signature", "SIG_OFF"), ("chain", "CHAIN_OFF"),
                 ("attenuation", "ATTEN_OFF"), ("conf_bind", "CBIND_OFF")]}
        fields["bits"] = {k: v for k, v in bits.items() if v}
        fields = {k: v for k, v in fields.items() if v}
    return {"type": ttype, "role": role, "fields": fields}


def _token_roles(idx) -> list[str]:
    n = 2 + len(idx["capabilities"]) + len(idx["confirmations"]) + 1
    roles = [""] * n
    roles[idx["request"]] = "request"
    roles[idx["policy"]] = "policy"
    roles[idx["output"]] = "output"
    for c in idx["capabilities"]:
        roles[c] = "capability"
    for c in idx["confirmations"]:
        roles[c] = "confirmation"
    return roles


def _changed(before, after):
    diff = np.argwhere(np.abs(after - before) > 1e-9)
    return [[int(t), int(c)] for t, c in diff]


def record(bundle, *, require_signatures=False, require_bound_confirmations=False) -> dict:
    """Produce the full step-by-step trace for one bundle."""
    engine = CompiledCapabilityTransformer(
        require_signatures=require_signatures,
        require_bound_confirmations=require_bound_confirmations)
    model = engine.model
    L = model.layout
    enc = tokenizer.encode(bundle, keyring=engine.keyring,
                           require_signatures=require_signatures,
                           require_bound_confirmations=require_bound_confirmations)
    R, idx = compiler.build_residual(model, enc, bundle)
    idx = {**idx, "layout": L}
    roles = _token_roles(idx)

    def snap():
        return [[round(float(v), 4) for v in row] for row in R]

    steps = []

    def push(**kw):
        kw["index"] = len(steps)
        steps.append(kw)

    # 0 — embedding
    push(layer="embedding", op="embed", kind="embedding",
         title="Embed tokens into the residual stream",
         desc="Request, policy, capability, confirmation and output tokens are placed into "
              "the residual stream; intrinsic bits (expiry, revoked, signature…) are copied in.",
         changed=[], snapshot=snap())

    # 1 — attention heads
    for h in model.match_heads:
        before = R.copy()
        queries = _token_indices(idx, h.query_set)
        Sq, Sk = R @ h.Wq.T, R @ h.Wk.T
        matches = []
        for q in queries:
            k = q if h.key_token == "self" else idx[h.key_token]
            score = float(Sq[q] @ Sk[k])
            matches.append({"query": int(q), "key": int(k), "score": round(score, 4),
                            "value": 1.0 if score >= h.threshold else 0.0})
        h.apply(R, idx)
        push(layer="attention", op=f"head:{h.name}", kind="match", head=h.name,
             title=f"head · {h.name}",
             desc=f"Query {h.query_set} attend to the {h.key_token} token; evidence = "
                  f"(Wq·r)·(Wk·r) ≥ {h.threshold} written to slot '{h.out_slot}'.",
             out_slot=h.out_slot, out_col=L.index(h.out_slot),
             query_set=h.query_set, key_token=h.key_token,
             detail={"matches": matches}, changed=_changed(before, R), snapshot=snap())

    # 2 + 4 — feed-forward gates (per-capability/per-confirmation, then decision)
    def run_gates(gates, layer):
        for g in gates:
            before = R.copy()
            tokens = _token_indices(idx, g.token_set)
            rows = []
            for t in tokens:
                ins = [{"name": FEAT_LABELS.get(n, n), "value": round(float(R[t, _coord(L, n)]), 4)}
                       for n in g.inputs]
                rows.append({"token": int(t), "inputs": ins})
            g.apply(R, idx)
            for row in rows:
                row["output"] = round(float(R[row["token"], L.index(g.out_slot)]), 4)
            push(layer=layer, op=f"gate:{g.name}", kind="gate", gate_op=g.op, gate=g.name,
                 title=f"gate · {g.name}  ({g.op.upper()})",
                 desc=f"{g.op.upper()} of {list(g.inputs)} over {g.token_set} tokens → '{g.out_slot}'.",
                 out_slot=g.out_slot, out_col=L.index(g.out_slot), token_set=g.token_set,
                 detail={"tokens": rows}, changed=_changed(before, R), snapshot=snap())

    run_gates(model.cap_gates, "ffn_per_cap")

    # 3 — max-pool existential aggregation
    for p in model.pool_heads:
        before = R.copy()
        members = _token_indices(idx, p.over)
        vals = [{"token": int(m), "value": round(float(R[m, L.index(p.in_slot)]), 4)} for m in members]
        p.apply(R, idx)
        out_t = int(idx["output"])
        amax = int(np.argmax([v["value"] for v in vals])) if vals else None
        push(layer="pool", op=f"pool:{p.name}", kind="pool", pool=p.name,
             title=f"∃ · {p.name}  (max-pool)",
             desc=f"max over {p.over} of '{p.in_slot}' → output token slot '{p.out_slot}' "
                  f"(max of bits = OR = ∃).",
             over=p.over, in_slot=p.in_slot, out_slot=p.out_slot, out_col=L.index(p.out_slot),
             out_token=out_t,
             detail={"members": vals, "argmax": amax,
                     "max_value": round(float(R[out_t, L.index(p.out_slot)]), 4)},
             changed=_changed(before, R), snapshot=snap())

    run_gates(model.decision_gates, "ffn_decision")

    # 5 — output projection
    logits = model.output_projection.logits(R, idx)
    decision = model.output_projection.decide(logits)
    classes = list(model.output_projection.classes)
    out_t = int(idx["output"])
    ev = {c: round(float(R[out_t, L.index(s)]), 4) for c, s in
          zip(classes, ["allow_evidence", "deny_evidence", "escalate_evidence"])}
    push(layer="output", op="output_projection", kind="output",
         title="Output projection → argmax",
         desc="A (3×D) projection reads [allow, deny, escalate] evidence into class logits; "
              "argmax (hardmax) selects the decision.",
         detail={"logits": {c: round(float(v), 4) for c, v in zip(classes, logits)},
                 "evidence": ev, "classes": classes,
                 "margin": model.output_projection.margin, "decision": decision},
         changed=[], snapshot=snap())

    # Layout: every residual column with its slot name and group.
    columns = _columns(L)
    decision_obj = engine.evaluate(bundle)
    ref = CapabilityTransformer(require_signatures=require_signatures,
                                require_bound_confirmations=require_bound_confirmations).evaluate(bundle)

    return {
        "decision": decision_obj.decision,
        "reasons": decision_obj.reasons,
        "reference_decision": ref.decision,
        "matches_reference": decision_obj.decision == ref.decision,
        "tokens": [{"index": i, **_decode_token(np.array(R[i]), roles[i])} for i in range(len(roles))],
        "token_roles": roles,
        "layout": {"width": L.width, "columns": columns},
        "layer_schedule": [{"id": a, "name": b, "desc": c} for a, b, c in LAYER_SCHEDULE],
        "steps": steps,
        "config": {"require_signatures": require_signatures,
                   "require_bound_confirmations": require_bound_confirmations},
    }


def _columns(L) -> list[dict]:
    """Per-residual-column metadata: name and group, for the heatmap."""
    cols = []
    for name in L.names:
        off, width = L.shape_of(name)
        for i in range(width):
            if name == "token_features":
                label, group = _feature_col_label(off + i), "features"
            else:
                label = name if width == 1 else f"{name}[{i}]"
                group = ("policy" if name.startswith(("pol_", "req_", "cap_"))
                         else "evidence")
            cols.append({"col": off + i, "name": name, "label": label, "group": group})
    return cols


def _feature_col_label(col: int) -> str:
    for slot in ("type", "subject", "object", "rights", "issuer", "provenance"):
        start, stop = W.SLOT[slot]
        if start <= col < stop:
            vocab = {"type": W.TOKEN_TYPES, "subject": W.SUBJECTS, "object": W.OBJECTS,
                     "rights": W.RIGHTS, "issuer": W.ISSUERS, "provenance": W.PROVENANCE}[slot]
            return f"{slot}:{vocab[col - start]}"
    for bit in ("EXPIRY_OFF", "REVOKED_OFF", "DELEG_OFF", "CONFIRM_OFF",
                "SIG_OFF", "CHAIN_OFF", "ATTEN_OFF", "CBIND_OFF"):
        if col == getattr(W, bit):
            return bit.replace("_OFF", "").lower()
    return f"f{col}"
