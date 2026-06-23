"""Unit tests for the feed-forward Boolean gates and the decision composition."""

import itertools

import numpy as np

from capability_transformer import compiler
from capability_transformer.transformer_model import ResidualLayout


def _layout(*slots):
    L = ResidualLayout()
    for s in slots:
        L.add(s, 1)
    return L


def _apply(gate, L, values: dict):
    R = np.zeros((1, L.width))
    for k, v in values.items():
        R[0, L.index(k)] = float(v)
    idx = {"layout": L, "output": 0, "request": 0, "policy": 0,
           "capabilities": [], "confirmations": []}
    gate.token_set = "output"
    gate.apply(R, idx)
    return R[0, L.index(gate.out_slot)]


def test_and_gate_truth_table():
    L = _layout("a", "b", "c", "out")
    g = compiler._and_gate(L, "and3", ["a", "b", "c"], "out", "output")
    for a, b, c in itertools.product([0, 1], repeat=3):
        assert _apply(g, L, {"a": a, "b": b, "c": c}) == float(a and b and c)


def test_or_gate_truth_table():
    L = _layout("a", "b", "c", "out")
    g = compiler._or_gate(L, "or3", ["a", "b", "c"], "out", "output")
    for a, b, c in itertools.product([0, 1], repeat=3):
        assert _apply(g, L, {"a": a, "b": b, "c": c}) == float(a or b or c)


def test_not_gate_truth_table():
    L = _layout("a", "out")
    g = compiler._not_gate(L, "not", "a", "out", "output")
    assert _apply(g, L, {"a": 0}) == 1.0
    assert _apply(g, L, {"a": 1}) == 0.0


def test_gates_are_real_relu_ffn():
    # The gate must compute via W2·ReLU(W1·r + b1) + b2, not a Python branch.
    L = _layout("a", "b", "out")
    g = compiler._and_gate(L, "and2", ["a", "b"], "out", "output")
    assert g.W1.shape == (1, L.width)
    assert g.W2.shape == (1, 1)
    r = np.zeros(L.width); r[L.index("a")] = 1.0; r[L.index("b")] = 1.0
    h = np.maximum(0.0, g.W1 @ r + g.b1)
    y = float(np.ravel(g.W2 @ h + g.b2)[0])
    assert y == 1.0


# ---- full decision composition (output-token gates + projection) ---------------------
def _decide(model, **precursors):
    L = model.layout
    R = np.zeros((1, L.width))
    for k, v in precursors.items():
        R[0, L.index(k)] = float(v)
    idx = {"layout": L, "output": 0, "request": 0, "policy": 0,
           "capabilities": [], "confirmations": []}
    for g in model.decision_gates:
        g.apply(R, idx)
    logits = model.output_projection.logits(R, idx)
    return model.output_projection.decide(logits)


def test_decision_composition():
    model = compiler.compile_policy()
    base = dict(has_match=1, o_prov_trusted=1, o_action_is_read=0, scope_violated=0,
                o_action_is_delegate=0, o_delegate_present=0, delegation_pass=0,
                o_high_risk=0, confirmed=0)

    # all good, low-risk -> ALLOW
    assert _decide(model, **base) == "ALLOW"
    # high-risk, no confirmation -> ESCALATE
    assert _decide(model, **{**base, "o_high_risk": 1}) == "ESCALATE"
    # high-risk, confirmed -> ALLOW
    assert _decide(model, **{**base, "o_high_risk": 1, "confirmed": 1}) == "ALLOW"
    # no matching capability -> DENY
    assert _decide(model, **{**base, "has_match": 0}) == "DENY"
    # unsafe provenance (not trusted, not read) -> DENY
    assert _decide(model, **{**base, "o_prov_trusted": 0, "o_action_is_read": 0}) == "DENY"
    # provenance ok via read -> ALLOW
    assert _decide(model, **{**base, "o_prov_trusted": 0, "o_action_is_read": 1}) == "ALLOW"
    # scope violated -> DENY
    assert _decide(model, **{**base, "scope_violated": 1}) == "DENY"
    # delegation requested but no valid delegator -> DENY
    assert _decide(model, **{**base, "o_action_is_delegate": 1, "o_delegate_present": 1,
                             "delegation_pass": 0}) == "DENY"
    # delegation requested with a valid delegator -> ALLOW
    assert _decide(model, **{**base, "o_action_is_delegate": 1, "o_delegate_present": 1,
                             "delegation_pass": 1}) == "ALLOW"
