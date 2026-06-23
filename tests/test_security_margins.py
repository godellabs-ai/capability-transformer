"""Output projection uses hardmax with a large, unambiguous margin (no soft scoring).

The compiled evaluator does not use softmax for any decision. The output projection emits
exactly one evidence bit; the winning class logit must dominate by the full margin, and the
hard selection (argmax) must be deterministic, including in (hypothetical) ties.
"""

import random

import numpy as np

from capability_transformer import compiler
from capability_transformer.compiled_core import CompiledCapabilityTransformer
from capability_transformer.equivalence import random_bundle


def test_no_softmax_on_compiled_path():
    import inspect

    from capability_transformer import compiled_core, compiler as comp, transformer_model

    for mod in (transformer_model, comp, compiled_core):
        src = inspect.getsource(mod).lower()
        for forbidden in ("softmax(", "np.exp", ".exp(", "logsumexp"):
            assert forbidden not in src


def test_exactly_one_evidence_bit_and_full_margin():
    engine = CompiledCapabilityTransformer()
    rng = random.Random(1)
    margin = engine.model.output_projection.margin
    for _ in range(2000):
        b = random_bundle(rng)
        res = engine.forward(b)
        L = engine.model.layout
        o = res.idx["output"]
        bits = [res.residual[o, L.index(s)]
                for s in ("allow_evidence", "deny_evidence", "escalate_evidence")]
        assert sum(round(x) for x in bits) == 1, bits          # exactly one class fires
        logits = res.logits
        order = np.argsort(logits)
        winner, runner_up = logits[order[-1]], logits[order[-2]]
        assert winner - runner_up >= margin - 1e-9             # large-margin hard decision


def test_argmax_tie_break_is_deterministic():
    # If two logits were ever equal, numpy argmax must pick the lowest index deterministically.
    model = compiler.compile_policy()
    assert model.output_projection.decide(np.array([1.0, 1.0, 0.0])) == "ALLOW"
    assert model.output_projection.decide(np.array([0.0, 1.0, 1.0])) == "DENY"
    assert model.output_projection.decide(np.array([0.0, 0.0, 0.0])) == "ALLOW"


def test_pool_head_max_is_deterministic():
    # The existential max-pool selects the max value with a deterministic tie-break.
    from capability_transformer.transformer_model import PoolHead, ResidualLayout

    L = ResidualLayout(); L.add("v", 1); L.add("out", 1)
    Wk = np.zeros((1, L.width)); Wk[0, L.index("v")] = 1.0
    head = PoolHead("p", "capabilities", "v", "out", Wk)
    R = np.zeros((4, L.width))
    R[1, L.index("v")] = 1.0; R[2, L.index("v")] = 1.0
    idx = {"layout": L, "output": 3, "capabilities": [0, 1, 2]}
    head.apply(R, idx)
    assert R[3, L.index("out")] == 1.0   # OR over caps == max bit
    R2 = np.zeros((2, L.width))
    idx2 = {"layout": L, "output": 1, "capabilities": [0]}
    head.apply(R2, idx2)
    assert R2[1, L.index("out")] == 0.0  # no valid cap -> 0
