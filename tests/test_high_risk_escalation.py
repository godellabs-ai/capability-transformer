"""High-risk actions escalate unless a trusted confirmation is present."""

import pytest

from conftest import bundle, cap

HIGH_RISK = [
    ("gmail", "send"),
    ("slack", "post"),
    ("file", "delete"),
    ("secrets_db", "read"),
    ("browser", "invoke"),
]


@pytest.mark.parametrize("obj,action", HIGH_RISK)
def test_high_risk_without_confirmation_escalates(engine, obj, action):
    d = engine.evaluate(
        bundle(action=action, object=obj, source_provenance="trusted_user",
               capabilities=[cap(object=obj, rights=[action])])
    )
    assert d.decision == "ESCALATE"
    assert d.reasons == ["confirmation_required"]


@pytest.mark.parametrize("obj,action", HIGH_RISK)
def test_high_risk_with_trusted_confirmation_allows(engine, obj, action):
    d = engine.evaluate(
        bundle(action=action, object=obj, source_provenance="trusted_user",
               capabilities=[cap(object=obj, rights=[action])],
               confirmations=[{"subject": "agent", "object": obj, "action": action, "issuer": "trusted_user"}])
    )
    assert d.decision == "ALLOW"
    assert d.reasons == ["allowed"]


def test_untrusted_confirmation_does_not_satisfy(engine):
    d = engine.evaluate(
        bundle(action="send", object="gmail", source_provenance="trusted_user",
               capabilities=[cap(rights=["send"])],
               confirmations=[{"subject": "agent", "object": "gmail", "action": "send", "issuer": "web_page"}])
    )
    assert d.decision == "ESCALATE"


def test_low_risk_action_does_not_escalate(engine):
    d = engine.evaluate(bundle(action="draft", object="gmail", capabilities=[cap(rights=["draft"])]))
    assert d.decision == "ALLOW"
