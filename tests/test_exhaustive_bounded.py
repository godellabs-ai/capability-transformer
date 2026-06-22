"""Exhaustive enumeration of the bounded universe.

For every (subject, object, right) we grant exactly one matching, valid capability and
assert the request on that triple is authorized (ALLOW or, for high-risk actions,
ESCALATE), while a mismatched object/right is denied. This proves exact-match access.
"""

import itertools

from capability_transformer import compiled_weights as W
from conftest import bundle, cap

HIGH_RISK = {("gmail", "send"), ("slack", "post"), ("file", "delete"),
             ("secrets_db", "read"), ("browser", "invoke")}


def _expected_grant(obj, right):
    return "ESCALATE" if (obj, right) in HIGH_RISK else "ALLOW"


def test_exhaustive_exact_match_grants(engine):
    """Every exact (subject, object, right) match is authorized."""
    total = 0
    for subj, obj, right in itertools.product(W.SUBJECTS, W.OBJECTS, W.RIGHTS):
        d = engine.evaluate(bundle(subject=subj, action=right, object=obj,
                                   source_provenance="trusted_user",
                                   capabilities=[cap(subject=subj, object=obj, rights=[right])]))
        assert d.decision == _expected_grant(obj, right), (subj, obj, right, d.decision)
        total += 1
    assert total == len(W.SUBJECTS) * len(W.OBJECTS) * len(W.RIGHTS)


def test_exhaustive_object_mismatch_denied(engine):
    """A capability for the wrong object never authorizes."""
    denied = 0
    for subj, obj, right in itertools.product(W.SUBJECTS, W.OBJECTS, W.RIGHTS):
        wrong_obj = next(o for o in W.OBJECTS if o != obj)
        d = engine.evaluate(bundle(subject=subj, action=right, object=obj,
                                   source_provenance="trusted_user",
                                   capabilities=[cap(subject=subj, object=wrong_obj, rights=[right])]))
        assert d.decision == "DENY"
        assert "object_mismatch" in d.reasons
        denied += 1
    assert denied == len(W.SUBJECTS) * len(W.OBJECTS) * len(W.RIGHTS)


def test_exhaustive_right_mismatch_denied(engine):
    """A capability granting a different right never authorizes."""
    for subj, obj, right in itertools.product(W.SUBJECTS, W.OBJECTS, W.RIGHTS):
        other_right = next(r for r in W.RIGHTS if r != right)
        d = engine.evaluate(bundle(subject=subj, action=right, object=obj,
                                   source_provenance="trusted_user",
                                   capabilities=[cap(subject=subj, object=obj, rights=[other_right])]))
        assert d.decision == "DENY"
        assert "right_not_granted" in d.reasons


def test_coverage_summary(engine, capsys):
    """Emit a coverage summary over the bounded universe."""
    allow = escalate = deny = 0
    for subj, obj, right in itertools.product(W.SUBJECTS, W.OBJECTS, W.RIGHTS):
        d = engine.evaluate(bundle(subject=subj, action=right, object=obj,
                                   source_provenance="trusted_user",
                                   capabilities=[cap(subject=subj, object=obj, rights=[right])]))
        allow += d.decision == "ALLOW"
        escalate += d.decision == "ESCALATE"
        deny += d.decision == "DENY"
    combos = len(W.SUBJECTS) * len(W.OBJECTS) * len(W.RIGHTS)
    with capsys.disabled():
        print(f"\n[coverage] {combos} (subject x object x right) combos with exact caps: "
              f"ALLOW={allow} ESCALATE={escalate} DENY={deny}")
    assert allow + escalate == combos  # every exact match is authorized
    assert deny == 0
