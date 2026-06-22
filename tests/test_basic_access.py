"""Basic allow/deny behavior."""

from conftest import bundle, cap


def test_allow_exact_match(engine):
    d = engine.evaluate(bundle(action="draft", object="gmail", capabilities=[cap(rights=["draft"])]))
    assert d.decision == "ALLOW"
    assert d.reasons == ["allowed"]


def test_deny_missing_capability(engine):
    d = engine.evaluate(bundle(action="draft", object="gmail", capabilities=[]))
    assert d.decision == "DENY"
    assert "missing_capability" in d.reasons


def test_deny_wrong_object(engine):
    d = engine.evaluate(
        bundle(action="invoke", object="gmail", capabilities=[cap(object="calendar", rights=["invoke"])])
    )
    assert d.decision == "DENY"
    assert "object_mismatch" in d.reasons


def test_deny_wrong_subject(engine):
    d = engine.evaluate(
        bundle(subject="agent", action="read", object="file",
               capabilities=[cap(subject="user", object="file", rights=["read"])])
    )
    assert d.decision == "DENY"
    assert "subject_mismatch" in d.reasons


def test_deny_wrong_right(engine):
    d = engine.evaluate(
        bundle(action="write", object="file", capabilities=[cap(object="file", rights=["read"])])
    )
    assert d.decision == "DENY"
    assert "right_not_granted" in d.reasons


def test_canonical_deny_trace(engine):
    """The canonical prompt-injection example reproduces the spec trace exactly."""
    d = engine.evaluate(
        bundle(action="send", object="gmail", source_provenance="retrieved_doc",
               capabilities=[cap(rights=["draft"])])
    )
    assert d.decision == "DENY"
    assert d.reasons == ["right_not_granted", "data_has_no_authority"]
    assert d.trace.failed_heads == ["head_right_match", "head_provenance_safe"]
    assert d.trace.passed_heads == [
        "head_subject_match", "head_object_match", "head_trusted_issuer",
        "head_not_expired", "head_not_revoked",
    ]
