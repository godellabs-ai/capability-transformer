"""Only trusted_user and system may mint authority."""

import pytest

from conftest import bundle, cap


@pytest.mark.parametrize("issuer", ["trusted_user", "system"])
def test_trusted_issuer_allowed(engine, issuer):
    d = engine.evaluate(bundle(action="read", object="file",
                               capabilities=[cap(object="file", rights=["read"], issuer=issuer)]))
    assert d.decision == "ALLOW"


@pytest.mark.parametrize("issuer", ["document", "web_page", "tool_output", "model_generated"])
def test_untrusted_issuer_denied(engine, issuer):
    d = engine.evaluate(bundle(action="read", object="file",
                               capabilities=[cap(object="file", rights=["read"], issuer=issuer)]))
    assert d.decision == "DENY"
    assert "issuer_not_trusted" in d.reasons
