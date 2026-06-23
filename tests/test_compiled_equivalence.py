"""The compiled transformer-style evaluator must match the reference evaluator.

The reference ``CapabilityTransformer`` is the specification. These randomized tests assert
``reference.evaluate(bundle).decision == compiled.evaluate(bundle).decision`` across the
full feature surface (issuers, expiry, revocation, signatures, delegation chains,
attenuation, provenance, confirmations, scope, the delegate action).
"""

import random

import pytest

from capability_transformer.equivalence import run_equivalence


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_equivalence_unsigned(seed):
    report = run_equivalence(2500, seed=seed)
    assert report.ok, report.decision_mismatches[:3]
    assert report.decision_matches == report.total


@pytest.mark.parametrize("seed", [11, 22])
def test_equivalence_signed(seed):
    report = run_equivalence(1500, seed=seed, require_signatures=True, sign=True)
    assert report.ok, report.decision_mismatches[:3]


def test_equivalence_signed_and_bound_confirmations():
    report = run_equivalence(1500, seed=7, require_signatures=True,
                             require_bound_confirmations=True, sign=True)
    assert report.ok, report.decision_mismatches[:3]


def test_equivalence_bound_confirmations_only():
    report = run_equivalence(1500, seed=8, require_bound_confirmations=True)
    assert report.ok, report.decision_mismatches[:3]


def test_reason_families_overlap_on_deny():
    # Decisions are identical; DENY reason families should overlap (comparable explanations).
    report = run_equivalence(2000, seed=5)
    assert report.reason_family_matches == report.total


def test_compiled_is_deterministic():
    from capability_transformer.compiled_core import CompiledCapabilityTransformer
    from capability_transformer.equivalence import random_bundle

    comp = CompiledCapabilityTransformer()
    rng = random.Random(123)
    for _ in range(200):
        b = random_bundle(rng)
        first = comp.evaluate(b).decision
        for _ in range(3):
            assert comp.evaluate(b).decision == first


def test_compiled_does_not_call_reference_reducer():
    # The compiled engine must not import or call the reference reducer to decide.
    import inspect

    from capability_transformer import compiled_core

    src = inspect.getsource(compiled_core)
    assert "_reduce" not in src
    assert "hard_attention" not in src
