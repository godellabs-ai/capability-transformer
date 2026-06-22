"""Provenance: untrusted data has no authority over side effects."""

import pytest

from conftest import bundle, cap


def test_trusted_user_may_use_capability(engine):
    d = engine.evaluate(bundle(action="draft", object="gmail", source_provenance="trusted_user",
                               capabilities=[cap(rights=["draft"])]))
    assert d.decision == "ALLOW"


def test_retrieved_doc_cannot_authorize_tool_call(engine):
    d = engine.evaluate(
        bundle(action="invoke", object="browser", source_provenance="retrieved_doc",
               capabilities=[cap(object="browser", rights=["invoke"])])
    )
    assert d.decision == "DENY"
    assert "data_has_no_authority" in d.reasons


def test_email_body_cannot_authorize_send(engine):
    d = engine.evaluate(
        bundle(action="send", object="gmail", source_provenance="email_body",
               capabilities=[cap(rights=["send"])])
    )
    assert d.decision == "DENY"
    assert "data_has_no_authority" in d.reasons


def test_web_page_cannot_invoke_browser(engine):
    d = engine.evaluate(
        bundle(action="invoke", object="browser", source_provenance="web_page",
               capabilities=[cap(object="browser", rights=["invoke"])])
    )
    assert d.decision == "DENY"
    assert "data_has_no_authority" in d.reasons


def test_model_generated_cannot_mint_capability_via_issuer(engine):
    # "minting": a capability whose issuer is model_generated is never trusted.
    d = engine.evaluate(
        bundle(action="draft", object="gmail", source_provenance="model_generated",
               capabilities=[cap(issuer="model_generated", rights=["draft"])])
    )
    assert d.decision == "DENY"
    assert "issuer_not_trusted" in d.reasons


def test_retrieved_doc_can_be_read_summarized(engine):
    # Passive read of a document for which a read capability exists is allowed.
    d = engine.evaluate(
        bundle(action="read", object="file", source_provenance="retrieved_doc",
               capabilities=[cap(object="file", rights=["read"])])
    )
    assert d.decision == "ALLOW"


@pytest.mark.parametrize("prov", ["retrieved_doc", "email_body", "web_page", "tool_output"])
def test_injection_data_cannot_drive_side_effects(engine, prov):
    d = engine.evaluate(
        bundle(action="post", object="slack", source_provenance=prov,
               capabilities=[cap(object="slack", rights=["post"])])
    )
    assert d.decision == "DENY"
    assert "data_has_no_authority" in d.reasons
