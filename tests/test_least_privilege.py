"""Least privilege: rights do not imply one another; objects are specific."""

from conftest import bundle, cap


def test_read_does_not_allow_write(engine):
    d = engine.evaluate(bundle(action="write", object="file", capabilities=[cap(object="file", rights=["read"])]))
    assert d.decision == "DENY"
    assert "right_not_granted" in d.reasons


def test_draft_does_not_allow_send(engine):
    # send is high-risk, but without the right it must DENY (deny precedes escalate).
    d = engine.evaluate(bundle(action="send", object="gmail", capabilities=[cap(rights=["draft"])]))
    assert d.decision == "DENY"
    assert "right_not_granted" in d.reasons


def test_invoke_calendar_does_not_allow_invoke_gmail(engine):
    d = engine.evaluate(
        bundle(action="invoke", object="gmail", capabilities=[cap(object="calendar", rights=["invoke"])])
    )
    assert d.decision == "DENY"
    assert "object_mismatch" in d.reasons


def test_write_does_not_imply_delete(engine):
    d = engine.evaluate(bundle(action="delete", object="file", capabilities=[cap(object="file", rights=["write"])]))
    assert d.decision == "DENY"
    assert "right_not_granted" in d.reasons
