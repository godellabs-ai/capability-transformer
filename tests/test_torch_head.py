"""The frozen torch capability head must equal the NumPy reference, with no trainable weights."""

import random

import pytest

pytest.importorskip("torch")

from capability_transformer import CapabilityTransformer  # noqa: E402
from capability_transformer.equivalence import random_bundle  # noqa: E402
from capability_transformer.torch_head import TorchCapabilityHead  # noqa: E402


def test_head_has_no_trainable_weights():
    head = TorchCapabilityHead()
    assert list(head.parameters()) == []          # no nn.Parameters at all
    assert all(not b.requires_grad for b in head.buffers())
    assert sum(1 for _ in head.buffers()) > 50     # the analytic weights are registered buffers


@pytest.mark.parametrize("seed", [1, 2])
def test_torch_equals_reference_unsigned(seed):
    ref, head = CapabilityTransformer(), TorchCapabilityHead()
    rng = random.Random(seed)
    for _ in range(300):
        b = random_bundle(rng)
        assert ref.evaluate(b).decision == head.decide(b).decision


def test_torch_equals_reference_signed():
    ref = CapabilityTransformer(require_signatures=True)
    head = TorchCapabilityHead(require_signatures=True)
    rng = random.Random(9)
    for _ in range(250):
        b = random_bundle(rng, sign=True)
        assert ref.evaluate(b).decision == head.decide(b).decision


def test_decision_is_deterministic():
    head = TorchCapabilityHead()
    rng = random.Random(5)
    for _ in range(40):
        b = random_bundle(rng)
        first = head.decide(b).decision
        assert all(head.decide(b).decision == first for _ in range(3))
