"""Security edge cases enforced by the COMPILED transformer-style evaluator.

Each case asserts the compiled evaluator returns the correct decision (and, for capability
cases, that the reference agrees). Grant-level cases use the gated runtime.
"""

from datetime import datetime, timedelta, timezone

import pytest

from capability_transformer import (
    Capability,
    CapabilityBundle,
    CapabilityTransformer,
    CompiledCapabilityTransformer,
    ToolCall,
    crypto,
)
from capability_transformer.delegated_capability import mint_child
from capability_transformer.runtime import GatedToolRuntime, GrantIssuer, ToolGateway

FUT = datetime(2099, 1, 1, tzinfo=timezone.utc)
PAST = datetime(2000, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 6, 23, 12, 0, 0, tzinfo=timezone.utc)


def cap(**over):
    base = dict(id="c1", subject="agent", object="gmail", rights=["draft"],
                issuer="trusted_user", expires_at=FUT)
    base.update(over)
    return Capability(**base)


def bundle(action="draft", object="gmail", provenance="trusted_user", caps=None, **extra):
    return CapabilityBundle(subject="agent", action=action, object=object,
                            source_provenance=provenance,
                            capabilities=caps if caps is not None else [cap()], **extra)


@pytest.fixture
def compiled():
    return CompiledCapabilityTransformer()


@pytest.fixture
def reference():
    return CapabilityTransformer()


def _both(compiled, reference, b):
    c = compiled.evaluate(b).decision
    assert c == reference.evaluate(b).decision
    return c


# ---- capability-level edge cases -----------------------------------------------------
def test_expired_capability(compiled, reference):
    b = bundle(action="read", object="file", caps=[cap(object="file", rights=["read"], expires_at=PAST)])
    assert _both(compiled, reference, b) == "DENY"


def test_revoked_capability(compiled, reference):
    b = bundle(action="read", object="file", caps=[cap(object="file", rights=["read"])],
               revocations=[{"capability_id": "c1"}])
    assert _both(compiled, reference, b) == "DENY"


def test_wrong_issuer(compiled, reference):
    b = bundle(action="read", object="file", caps=[cap(object="file", rights=["read"], issuer="web_page")])
    assert _both(compiled, reference, b) == "DENY"


def test_wrong_subject(compiled, reference):
    b = bundle(action="read", object="file", caps=[cap(subject="user", object="file", rights=["read"])])
    assert _both(compiled, reference, b) == "DENY"


def test_wrong_object(compiled, reference):
    b = bundle(action="read", object="gmail", caps=[cap(object="file", rights=["read"])])
    assert _both(compiled, reference, b) == "DENY"


def test_wrong_right(compiled, reference):
    b = bundle(action="send", object="gmail", caps=[cap(rights=["draft"])])
    assert _both(compiled, reference, b) == "DENY"


def test_unsafe_provenance(compiled, reference):
    # Untrusted data driving a side effect -> DENY even with a matching capability.
    b = bundle(action="send", object="gmail", provenance="retrieved_doc", caps=[cap(rights=["send"])])
    assert _both(compiled, reference, b) == "DENY"


def test_missing_confirmation_escalates(compiled, reference):
    b = bundle(action="send", object="gmail", caps=[cap(rights=["send"])])
    assert _both(compiled, reference, b) == "ESCALATE"


# ---- signed / delegation cases (compiled, signed engine) -----------------------------
def _signed_engines():
    return (CompiledCapabilityTransformer(require_signatures=True),
            CapabilityTransformer(require_signatures=True))


def test_invalid_signature():
    comp, ref = _signed_engines()
    unsigned = cap(object="file", rights=["read"])  # no signature in signed mode
    b = bundle(action="read", object="file", caps=[unsigned])
    assert comp.evaluate(b).decision == "DENY" == ref.evaluate(b).decision


def test_tampered_signature():
    comp, ref = _signed_engines()
    signed = crypto.issue(cap(object="file", rights=["read"]))
    tampered = signed.model_copy(update={"rights": ["read", "send"]})
    b = bundle(action="send", object="file", caps=[tampered])
    assert comp.evaluate(b).decision == "DENY" == ref.evaluate(b).decision


def test_invalid_delegation_no_delegate_right():
    comp, ref = _signed_engines()
    parent = crypto.issue(cap(id="p", subject="user", object="file", rights=["read"]))  # no delegate
    child = mint_child(parent, id="ch", subject="agent", rights=["read"])
    b = CapabilityBundle(subject="agent", action="read", object="file",
                         source_provenance="trusted_user", capabilities=[parent, child])
    assert comp.evaluate(b).decision == "DENY" == ref.evaluate(b).decision


def test_excessive_delegated_rights():
    comp, ref = _signed_engines()
    parent = crypto.issue(cap(id="p", subject="user", object="file",
                              rights=["read", "delegate"], delegatable=True))
    child = mint_child(parent, id="ch", subject="agent", rights=["read", "send"])  # widened
    b = CapabilityBundle(subject="agent", action="send", object="file",
                         source_provenance="trusted_user", capabilities=[parent, child])
    assert comp.evaluate(b).decision == "DENY" == ref.evaluate(b).decision


def test_mismatched_confirmation_action_hash():
    comp = CompiledCapabilityTransformer(require_bound_confirmations=True)
    ref = CapabilityTransformer(require_bound_confirmations=True)
    b = CapabilityBundle(subject="agent", action="send", object="gmail",
                         source_provenance="trusted_user", capabilities=[cap(rights=["send"])],
                         action_hash="approved-X",
                         confirmations=[{"subject": "agent", "object": "gmail", "action": "send",
                                         "issuer": "trusted_user", "action_hash": "approved-Y"}])
    assert comp.evaluate(b).decision == "ESCALATE" == ref.evaluate(b).decision


# ---- grant-level edge cases (gated runtime) ------------------------------------------
def _grant_setup():
    gw, rt = ToolGateway(), GatedToolRuntime()
    c = ToolCall(subject="agent", action="draft", object="gmail", args={"to": "x"})
    b = CapabilityBundle(subject="agent", action="draft", object="gmail",
                         source_provenance="trusted_user", capabilities=[cap(rights=["draft"])])
    return gw, rt, c, b


def test_stale_grant_refused():
    gw, rt, c, b = _grant_setup()
    _, grant = gw.authorize(b, c, now=NOW, nonce="n1", ttl_seconds=30)
    ex = rt.execute(grant, c, now=NOW + timedelta(seconds=31))
    assert ex.executed is False and ex.refused_reason == "grant_expired"


def test_reused_grant_refused():
    gw, rt, c, b = _grant_setup()
    _, grant = gw.authorize(b, c, now=NOW, nonce="n2")
    assert rt.execute(grant, c, now=NOW).executed is True
    replay = rt.execute(grant, c, now=NOW)
    assert replay.executed is False and replay.refused_reason == "grant_replayed"


def test_wrong_grant_action_hash_refused():
    gw, rt, c, b = _grant_setup()
    _, grant = gw.authorize(b, c, now=NOW, nonce="n3")
    other = ToolCall(subject="agent", action="draft", object="gmail", args={"to": "attacker"})
    ex = rt.execute(grant, other, now=NOW)
    assert ex.executed is False and ex.refused_reason == "action_binding_mismatch"
