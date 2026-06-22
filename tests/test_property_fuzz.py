"""Phase 7 — property-based fuzzing.

We generate thousands of random bundles from the bounded universe and cross-check the
engine against an *independent* reference oracle, plus a set of always-true security
invariants. The RNG is seeded so failures are reproducible.
"""

import random

import pytest

from capability_transformer import CapabilityBundle, CapabilityTransformer
from capability_transformer import compiled_weights as W

FUTURE = "2099-01-01T00:00:00Z"
PAST = "2000-01-01T00:00:00Z"
TRUSTED_ISSUERS = {"trusted_user", "system"}
TRUSTED_PROV = {"trusted_user", "system_policy"}
HIGH_RISK = {("gmail", "send"), ("slack", "post"), ("file", "delete"),
             ("secrets_db", "read"), ("browser", "invoke")}
# Fuzz over non-delegate actions (delegation has its own dedicated suite).
ACTIONS = [r for r in W.RIGHTS if r != "delegate"]


def _rand_cap(rng, idx):
    return {
        "id": f"cap{idx}",
        "subject": rng.choice(W.SUBJECTS),
        "object": rng.choice(W.OBJECTS),
        "rights": rng.sample(W.RIGHTS, rng.randint(1, 3)),
        "issuer": rng.choice(W.ISSUERS),
        "expires_at": rng.choice([FUTURE, FUTURE, PAST]),  # bias toward valid
        "scope": {},
        "delegatable": rng.choice([True, False]),
    }


def _rand_bundle(rng):
    caps = [_rand_cap(rng, i) for i in range(rng.randint(0, 4))]
    confs = []
    for _ in range(rng.randint(0, 2)):
        confs.append({
            "subject": rng.choice(W.SUBJECTS),
            "object": rng.choice(W.OBJECTS),
            "action": rng.choice(ACTIONS),
            "issuer": rng.choice(W.ISSUERS),
        })
    revs = []
    if caps and rng.random() < 0.3:
        revs.append({"capability_id": rng.choice(caps)["id"]})
    return CapabilityBundle(
        subject=rng.choice(W.SUBJECTS),
        action=rng.choice(ACTIONS),
        object=rng.choice(W.OBJECTS),
        source_provenance=rng.choice(W.PROVENANCE),
        capabilities=caps,
        revocations=revs,
        confirmations=confs,
    )


def _oracle(b: CapabilityBundle) -> str:
    """Independent reference implementation of the decision semantics."""
    revoked = {r.capability_id for r in b.revocations if r.capability_id}

    def valid(cap):
        return (
            cap.subject == b.subject
            and cap.object == b.object
            and b.action in cap.rights
            and cap.issuer in TRUSTED_ISSUERS
            and cap.expires_at.year > 2026  # FUTURE vs PAST
            and cap.id not in revoked
        )

    has_match = any(valid(c) for c in b.capabilities)
    prov_ok = b.source_provenance in TRUSTED_PROV or b.action == "read"
    if not (has_match and prov_ok):
        return "DENY"

    if (b.object, b.action) in HIGH_RISK:
        confirmed = any(
            c.subject == b.subject and c.object == b.object
            and c.action == b.action and c.issuer in TRUSTED_ISSUERS
            for c in b.confirmations
        )
        if not confirmed:
            return "ESCALATE"
    return "ALLOW"


@pytest.fixture
def engine():
    return CapabilityTransformer()


def test_fuzz_matches_oracle(engine):
    rng = random.Random(20260622)
    for _ in range(4000):
        b = _rand_bundle(rng)
        assert engine.evaluate(b).decision == _oracle(b), b.model_dump()


def test_fuzz_security_invariants(engine):
    rng = random.Random(1337)
    for _ in range(4000):
        b = _rand_bundle(rng)
        d = engine.evaluate(b)

        # 1. Determinism.
        assert engine.evaluate(b).decision == d.decision

        # 2. Reasons are never empty; ALLOW is exactly ["allowed"].
        assert d.reasons
        if d.decision == "ALLOW":
            assert d.reasons == ["allowed"]
        else:
            assert "allowed" not in d.reasons

        # 3. Untrusted data can never drive a side effect to ALLOW/ESCALATE.
        if b.source_provenance not in TRUSTED_PROV and b.action != "read":
            assert d.decision == "DENY"
            assert "data_has_no_authority" in d.reasons

        # 4. A high-risk action is never ALLOWed without a trusted confirmation.
        if (b.object, b.action) in HIGH_RISK and d.decision == "ALLOW":
            assert any(
                c.subject == b.subject and c.object == b.object
                and c.action == b.action and c.issuer in TRUSTED_ISSUERS
                for c in b.confirmations
            )

        # 5. No capabilities at all -> never authorized.
        if not b.capabilities:
            assert d.decision == "DENY"
