"""Each compiled attention head must match the reference hard-mask-head semantics."""

import random

import numpy as np

from capability_transformer import compiled_weights as W
from capability_transformer import hard_attention, tokenizer
from capability_transformer.compiled_core import CompiledCapabilityTransformer
from capability_transformer.equivalence import random_bundle


def _residual(engine, bundle):
    res = engine.forward(bundle)
    return res.residual, res.idx, engine.model.layout


def _cap_slot(R, idx, L, slot):
    return np.array([R[c, L.index(slot)] for c in idx["capabilities"]], dtype=bool)


def test_match_heads_equal_reference_masks():
    engine = CompiledCapabilityTransformer()
    rng = random.Random(42)
    head_pairs = [
        ("subject_match", "head_subject_match"),
        ("object_match", "head_object_match"),
        ("right_match", "head_right_match"),
        ("issuer_trusted", "head_trusted_issuer"),
    ]
    for _ in range(800):
        b = random_bundle(rng)
        if not b.capabilities:
            continue
        R, idx, L = _residual(engine, b)
        enc = tokenizer.encode(b)
        att = hard_attention.compute(enc)
        for comp_slot, ref_head in head_pairs:
            got = _cap_slot(R, idx, L, comp_slot)
            ref = att.heads[ref_head].per_cap_mask.astype(bool)
            assert np.array_equal(got, ref), (comp_slot, got, ref)


def test_not_revoked_and_not_expired_match_reference():
    engine = CompiledCapabilityTransformer()
    rng = random.Random(7)
    for _ in range(500):
        b = random_bundle(rng)
        if not b.capabilities:
            continue
        R, idx, L = _residual(engine, b)
        enc = tokenizer.encode(b)
        att = hard_attention.compute(enc)
        not_revoked = _cap_slot(R, idx, L, "not_revoked")
        assert np.array_equal(not_revoked, att.heads["head_not_revoked"].per_cap_mask.astype(bool))
        # not_expired is read straight from the token feature bit
        expiry = np.array([R[c, W.EXPIRY_OFF] for c in idx["capabilities"]], dtype=bool)
        assert np.array_equal(expiry, att.heads["head_not_expired"].per_cap_mask.astype(bool))


def test_request_heads_match_reference_predicates():
    engine = CompiledCapabilityTransformer()
    rng = random.Random(3)
    for _ in range(600):
        b = random_bundle(rng)
        R, idx, L = _residual(engine, b)
        req = idx["request"]
        q = R[req, L.slice("token_features")]
        q_prov = q[slice(*W.SLOT["provenance"])]
        q_action = q[slice(*W.SLOT["rights"])]
        q_obj = q[slice(*W.SLOT["object"])]
        prov_trusted = float(q_prov @ W.TRUSTED_PROV_MASK >= 0.5)
        action_is_read = float(q_action @ W.NON_SIDE_EFFECT_MASK >= 0.5)
        high_risk = float(q_obj @ (W.HIGH_RISK @ q_action) >= 0.5)
        assert R[req, L.index("prov_trusted")] == prov_trusted
        assert R[req, L.index("action_is_read")] == action_is_read
        assert R[req, L.index("high_risk_action")] == high_risk


def test_valid_capability_is_per_capability_conjunction():
    """Soundness: valid_capability must be the AND of one capability's own predicates."""
    engine = CompiledCapabilityTransformer()
    rng = random.Random(99)
    for _ in range(800):
        b = random_bundle(rng)
        if not b.capabilities:
            continue
        R, idx, L = _residual(engine, b)
        for c in idx["capabilities"]:
            bits = [
                R[c, L.index("subject_match")], R[c, L.index("object_match")],
                R[c, L.index("right_match")], R[c, L.index("issuer_trusted")],
                R[c, W.EXPIRY_OFF], R[c, L.index("not_revoked")], R[c, W.SIG_OFF],
                R[c, L.index("chain_ok")], R[c, L.index("atten_ok")],
            ]
            expected = 1.0 if all(x > 0.5 for x in bits) else 0.0
            assert R[c, L.index("valid_capability")] == expected


def test_has_match_is_existential_over_capabilities():
    """has_match must be OR over per-capability validity (∃), never a cross-cap mix."""
    engine = CompiledCapabilityTransformer()
    rng = random.Random(2024)
    for _ in range(800):
        b = random_bundle(rng)
        R, idx, L = _residual(engine, b)
        per_cap = [R[c, L.index("valid_capability")] > 0.5 for c in idx["capabilities"]]
        has_match = R[idx["output"], L.index("has_match")] > 0.5
        assert has_match == any(per_cap)


def test_cross_capability_evidence_does_not_leak():
    """subject_match from one cap + right_match from another must NOT authorize."""
    from capability_transformer.schema import Capability, CapabilityBundle

    FUT = "2099-01-01T00:00:00Z"
    # cap A: right subject/object but only 'read'; cap B: has 'send' but wrong object.
    capA = Capability(id="A", subject="agent", object="gmail", rights=["read"],
                      issuer="trusted_user", expires_at=FUT)
    capB = Capability(id="B", subject="agent", object="file", rights=["send"],
                      issuer="trusted_user", expires_at=FUT)
    b = CapabilityBundle(subject="agent", action="send", object="gmail",
                         source_provenance="trusted_user", capabilities=[capA, capB])
    engine = CompiledCapabilityTransformer()
    R, idx, L = _residual(engine, b)
    # No single capability satisfies subject∧object∧right -> has_match must be 0.
    assert R[idx["output"], L.index("has_match")] == 0.0
    assert engine.evaluate(b).decision == "DENY"
