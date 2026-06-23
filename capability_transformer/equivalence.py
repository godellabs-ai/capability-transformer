"""Equivalence harness: reference evaluator vs. compiled transformer-style evaluator.

The reference ``CapabilityTransformer`` is the specification. This module runs both
evaluators over randomized request bundles and checks that their **decisions** are
identical and their **explanations** are comparable (the reference's reason set is a
subset/superset relationship is not required; we check decision identity and that the
DENY reason families overlap). Used by ``tests/test_compiled_equivalence.py``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone

from . import compiled_weights as W
from . import crypto
from .compiled_core import CompiledCapabilityTransformer
from .core import CapabilityTransformer
from .delegated_capability import mint_child
from .schema import Capability, CapabilityBundle

FUTURE = datetime(2099, 1, 1, tzinfo=timezone.utc)
PAST = datetime(2000, 1, 1, tzinfo=timezone.utc)


def make_engines(*, require_signatures=False, require_bound_confirmations=False):
    ref = CapabilityTransformer(require_signatures=require_signatures,
                                require_bound_confirmations=require_bound_confirmations)
    comp = CompiledCapabilityTransformer(require_signatures=require_signatures,
                                         require_bound_confirmations=require_bound_confirmations)
    return ref, comp


def random_bundle(rng: random.Random, *, sign: bool = False) -> CapabilityBundle:
    """A random bundle exercising every security feature."""
    def make_cap(i):
        issuer = rng.choice(["trusted_user", "system", "web_page", "document"])
        cap = Capability(
            id=f"c{i}", subject=rng.choice(W.SUBJECTS), object=rng.choice(W.OBJECTS),
            rights=rng.sample(W.RIGHTS, rng.randint(1, 4)), issuer=issuer,
            expires_at=rng.choice([FUTURE, FUTURE, PAST]),
            scope=rng.choice([{}, {}, {"region": "eu"}]),
            delegatable=rng.choice([True, False]),
        )
        if sign and issuer in ("trusted_user", "system") and rng.random() < 0.85:
            try:
                cap = crypto.issue(cap)
            except Exception:
                pass
            if rng.random() < 0.15 and cap.signature:        # tamper sometimes
                cap = cap.model_copy(update={"rights": cap.rights + ["delete"]})
        return cap

    caps = [make_cap(i) for i in range(rng.randint(0, 3))]
    # Occasionally add a delegated child of a signed, delegate-bearing parent.
    parents = [c for c in caps if c.signature and "delegate" in c.rights]
    if sign and parents and rng.random() < 0.5:
        p = rng.choice(parents)
        try:
            child = mint_child(
                p, id=f"ch{rng.randint(0, 9999)}", subject=rng.choice(W.SUBJECTS),
                rights=rng.sample(p.rights, rng.randint(1, len(p.rights))),
                expires_at=p.expires_at, delegatable=rng.choice([True, False]),
            )
            if rng.random() < 0.2:
                child = child.model_copy(update={"parent_hash": "0" * 64})  # break chain
            caps.append(child)
        except Exception:
            pass

    confs = []
    for _ in range(rng.randint(0, 2)):
        cd = {"subject": rng.choice(W.SUBJECTS), "object": rng.choice(W.OBJECTS),
              "action": rng.choice(W.RIGHTS), "issuer": rng.choice(W.ISSUERS)}
        if rng.random() < 0.5:
            cd["action_hash"] = "hash-" + str(rng.randint(0, 3))
        confs.append(cd)

    revs = [{"capability_id": rng.choice(caps).id}] if caps and rng.random() < 0.25 else []
    action = rng.choice(W.RIGHTS)
    dr = rng.choice([None, None, "read", "write"]) if action == "delegate" else None
    action_hash = "hash-" + str(rng.randint(0, 3)) if rng.random() < 0.6 else None

    return CapabilityBundle(
        subject=rng.choice(W.SUBJECTS), action=action, object=rng.choice(W.OBJECTS),
        source_provenance=rng.choice(W.PROVENANCE), capabilities=caps,
        revocations=revs, confirmations=confs, scope=rng.choice([{}, {"region": "eu"}]),
        delegate_right=dr, action_hash=action_hash,
    )


@dataclass
class EquivalenceReport:
    total: int
    decision_matches: int
    decision_mismatches: list[tuple]
    reason_family_matches: int

    @property
    def ok(self) -> bool:
        return not self.decision_mismatches


def _deny_families(reasons):
    return set(reasons)


def run_equivalence(n: int, *, seed: int, require_signatures=False,
                    require_bound_confirmations=False, sign=False) -> EquivalenceReport:
    ref, comp = make_engines(require_signatures=require_signatures,
                             require_bound_confirmations=require_bound_confirmations)
    rng = random.Random(seed)
    matches = reason_matches = 0
    mismatches: list[tuple] = []
    for _ in range(n):
        b = random_bundle(rng, sign=sign or require_signatures)
        d_ref = ref.evaluate(b)
        d_comp = comp.evaluate(b)
        if d_ref.decision == d_comp.decision:
            matches += 1
            if d_ref.decision != "DENY" or (_deny_families(d_ref.reasons) & _deny_families(d_comp.reasons)):
                reason_matches += 1
        else:
            mismatches.append((d_ref.decision, d_comp.decision, b.model_dump()))
    return EquivalenceReport(n, matches, mismatches, reason_matches)
