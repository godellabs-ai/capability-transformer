"""HTTP API tests.

The public API is SECURE by default (signed capabilities + action-bound confirmations).
Tests mint signed capabilities; one test confirms unsigned capabilities are denied.
"""

from fastapi.testclient import TestClient

from capability_transformer import crypto
from capability_transformer.api import SecureCapabilityTransformer, app
from capability_transformer import api as api_module
from capability_transformer.schema import Capability

client = TestClient(app)

FUTURE = "2099-01-01T00:00:00Z"


def signed_cap(rights, object="gmail", subject="agent"):
    cap = Capability(id="cap1", subject=subject, object=object, rights=list(rights),
                     issuer="trusted_user", expires_at=FUTURE)
    return crypto.issue(cap).model_dump(mode="json")


def unsigned_cap(rights, object="gmail", subject="agent", issuer="trusted_user"):
    return {"id": "cap1", "subject": subject, "object": object, "rights": rights,
            "issuer": issuer, "expires_at": FUTURE, "scope": {}, "delegatable": False}


def test_api_is_secure_by_default():
    assert isinstance(api_module._engine, SecureCapabilityTransformer)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["trained"] is False
    assert body["softmax_used"] is False


def test_schema():
    r = client.get("/schema")
    assert r.status_code == 200
    body = r.json()
    assert body["subjects"] and body["objects"] and body["rights"]
    assert "request_schema" in body


def test_examples_endpoint():
    r = client.get("/examples")
    assert r.status_code == 200
    assert "allow_gmail_draft" in r.json()


def test_evaluate_allow_signed():
    r = client.post("/evaluate", json={
        "subject": "agent", "action": "draft", "object": "gmail",
        "source_provenance": "trusted_user", "capabilities": [signed_cap(["draft"])]})
    assert r.status_code == 200
    assert r.json()["decision"] == "ALLOW"
    assert r.json()["reasons"] == ["allowed"]


def test_secure_api_rejects_unsigned_capability():
    r = client.post("/evaluate", json={
        "subject": "agent", "action": "draft", "object": "gmail",
        "source_provenance": "trusted_user", "capabilities": [unsigned_cap(["draft"])]})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "DENY"
    assert "invalid_signature" in body["reasons"]


def test_evaluate_deny():
    r = client.post("/evaluate", json={
        "subject": "agent", "action": "send", "object": "gmail",
        "source_provenance": "retrieved_doc", "capabilities": [signed_cap(["draft"])]})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "DENY"
    assert "right_not_granted" in body["reasons"]
    assert "data_has_no_authority" in body["reasons"]


def test_evaluate_escalate():
    r = client.post("/evaluate", json={
        "subject": "agent", "action": "send", "object": "gmail",
        "source_provenance": "trusted_user", "capabilities": [signed_cap(["send"])]})
    assert r.status_code == 200
    assert r.json()["decision"] == "ESCALATE"
    assert r.json()["reasons"] == ["confirmation_required"]


def test_evaluate_confirmed_allow_bound():
    r = client.post("/evaluate", json={
        "subject": "agent", "action": "send", "object": "gmail",
        "source_provenance": "trusted_user", "capabilities": [signed_cap(["send"])],
        "action_hash": "approved-1",
        "confirmations": [{"subject": "agent", "object": "gmail", "action": "send",
                           "issuer": "trusted_user", "action_hash": "approved-1"}]})
    assert r.status_code == 200
    assert r.json()["decision"] == "ALLOW"


def test_mint_signs_capability():
    cap = {"id": "cap1", "subject": "agent", "object": "file", "rights": ["read"],
           "issuer": "trusted_user", "expires_at": FUTURE, "scope": {}, "delegatable": False}
    r = client.post("/mint", json=cap)
    assert r.status_code == 200
    assert r.json()["signature"]


def test_mint_untrusted_issuer_rejected():
    cap = {"id": "cap1", "subject": "agent", "object": "file", "rights": ["read"],
           "issuer": "web_page", "expires_at": FUTURE, "scope": {}, "delegatable": False}
    r = client.post("/mint", json=cap)
    assert r.status_code == 422


def test_authorize_and_execute_allow():
    body = {
        "bundle": {
            "subject": "agent", "action": "draft", "object": "gmail",
            "source_provenance": "trusted_user", "capabilities": [signed_cap(["draft"])]},
        "args": {"to": "bob@example.com", "body": "hi"},
    }
    r = client.post("/authorize", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["decision"]["decision"] == "ALLOW"
    assert out["grant"] is not None

    call = {"subject": "agent", "action": "draft", "object": "gmail",
            "args": {"to": "bob@example.com", "body": "hi"}}
    r2 = client.post("/execute", json={"grant": out["grant"], "call": call})
    assert r2.status_code == 200
    assert r2.json()["executed"] is True


def test_authorize_deny_then_execute_refused():
    body = {
        "bundle": {
            "subject": "agent", "action": "send", "object": "gmail",
            "source_provenance": "retrieved_doc", "capabilities": [signed_cap(["send"])]},
        "args": {"to": "x"},
    }
    r = client.post("/authorize", json=body)
    assert r.json()["decision"]["decision"] == "DENY"
    assert r.json()["grant"] is None

    call = {"subject": "agent", "action": "send", "object": "gmail", "args": {"to": "x"}}
    r2 = client.post("/execute", json={"grant": None, "call": call})
    assert r2.json()["executed"] is False
    assert r2.json()["refused_reason"] == "no_grant"


def test_audit_endpoints():
    body = {
        "bundle": {
            "subject": "agent", "action": "draft", "object": "gmail",
            "source_provenance": "trusted_user", "capabilities": [signed_cap(["draft"])]},
        "args": {"to": "bob"},
    }
    client.post("/authorize", json=body)

    r = client.get("/audit")
    assert r.status_code == 200
    events = r.json()
    assert len(events) >= 1
    eid = events[-1]["event_id"]

    v = client.get("/audit/verify")
    assert v.status_code == 200
    assert v.json()["ok"] is True

    one = client.get(f"/audit/{eid}")
    assert one.status_code == 200
    assert one.json()["event_id"] == eid
    assert client.get("/audit/does-not-exist").status_code == 404


def test_flow_provenance():
    r = client.post("/flow/provenance", json={"base": "trusted_user", "influences": ["email_body"]})
    assert r.status_code == 200
    body = r.json()
    assert body["effective_provenance"] == "email_body"
    assert body["is_trusted"] is False
    assert body["authorizes_side_effects"] is False

    r2 = client.post("/flow/provenance", json={"base": "trusted_user", "influences": ["system_policy"]})
    assert r2.json()["effective_provenance"] == "trusted_user"
    assert r2.json()["authorizes_side_effects"] is True


def test_invalid_enum_rejected():
    r = client.post("/evaluate", json={
        "subject": "agent", "action": "teleport", "object": "gmail",
        "source_provenance": "trusted_user", "capabilities": []})
    assert r.status_code == 422
