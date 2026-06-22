"""Tensor-native properties of the enforcement core."""

import inspect

import numpy as np

from capability_transformer import compiled_weights as W
from capability_transformer import core, hard_attention, tokenizer
from conftest import bundle, cap


def test_tokenizer_produces_token_matrix():
    enc = tokenizer.encode(bundle(action="draft", object="gmail", capabilities=[cap(rights=["draft"])]))
    assert isinstance(enc.X, np.ndarray)
    assert enc.X.ndim == 2
    assert enc.X.shape[1] == W.D  # fixed-width token vectors
    # request token + 1 capability token
    assert enc.X.shape[0] == 2


def test_enforcement_uses_hard_attention_path(engine):
    enc = tokenizer.encode(bundle(action="draft", object="gmail", capabilities=[cap(rights=["draft"])]))
    att = hard_attention.compute(enc)
    # Heads return Boolean masks (hard attention), not probability distributions.
    for head in att.heads.values():
        assert head.per_cap_mask.dtype == bool
    assert att.matched_mask.dtype == bool


def test_deterministic_repeated_output(engine):
    b = bundle(action="send", object="gmail", source_provenance="trusted_user",
               capabilities=[cap(rights=["send"])])
    first = engine.evaluate(b).model_dump()
    for _ in range(25):
        assert engine.evaluate(b).model_dump() == first


def test_no_softmax_used_for_enforcement():
    # No softmax / exponential normalization is *called* anywhere on the enforcement
    # path. (Docstrings may mention the word to explain why it is avoided.)
    for mod in (hard_attention, core, tokenizer):
        src = inspect.getsource(mod).lower()
        for forbidden in ("softmax(", "np.exp", ".exp(", "logsumexp", "nn.functional"):
            assert forbidden not in src


def test_no_training_code():
    for mod in (hard_attention, core, tokenizer, W):
        src = inspect.getsource(mod).lower()
        for forbidden in ("backward", "loss.backward", "optimizer", "requires_grad", "torch.nn"):
            assert forbidden not in src


def test_trace_reports_head_pass_fail(engine):
    d = engine.evaluate(bundle(action="send", object="gmail", source_provenance="retrieved_doc",
                               capabilities=[cap(rights=["draft"])]))
    assert d.trace.failed_heads
    assert d.trace.passed_heads
    names = {h.name for h in d.trace.heads}
    assert "head_subject_match" in names
    assert d.trace.softmax_used is False
    assert d.trace.trained is False


def test_masks_are_exact_match_not_soft():
    # Subject head mask must be exactly the equality mask, computed by dot product.
    enc = tokenizer.encode(bundle(
        action="read", object="file",
        capabilities=[cap(id="a", subject="agent", object="file", rights=["read"]),
                      cap(id="b", subject="user", object="file", rights=["read"])],
    ))
    att = hard_attention.compute(enc)
    mask = att.heads["head_subject_match"].per_cap_mask
    assert list(mask) == [True, False]  # only the agent capability matches
