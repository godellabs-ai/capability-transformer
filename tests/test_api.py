"""HTTP API tests."""

from fastapi.testclient import TestClient

from capability_transformer.api import app

client = TestClient(app)

FUTURE = "2099-01-01T00:00:00Z"


def _cap(rights, object="gmail", subject="agent", issuer="trusted_user"):
    return {"id": "cap1", "subject": subject, "object": object, "rights": rights,
            "issuer": issuer, "expires_at": FUTURE, "scope": {}, "delegatable": False}


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


def test_evaluate_allow():
    r = client.post("/evaluate", json={
        "subject": "agent", "action": "draft", "object": "gmail",
        "source_provenance": "trusted_user", "capabilities": [_cap(["draft"])],
        "revocations": [], "confirmations": []})
    assert r.status_code == 200
    assert r.json()["decision"] == "ALLOW"
    assert r.json()["reasons"] == ["allowed"]


def test_evaluate_deny():
    r = client.post("/evaluate", json={
        "subject": "agent", "action": "send", "object": "gmail",
        "source_provenance": "retrieved_doc", "capabilities": [_cap(["draft"])],
        "revocations": [], "confirmations": []})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "DENY"
    assert "right_not_granted" in body["reasons"]
    assert "data_has_no_authority" in body["reasons"]


def test_evaluate_escalate():
    r = client.post("/evaluate", json={
        "subject": "agent", "action": "send", "object": "gmail",
        "source_provenance": "trusted_user", "capabilities": [_cap(["send"])],
        "revocations": [], "confirmations": []})
    assert r.status_code == 200
    assert r.json()["decision"] == "ESCALATE"
    assert r.json()["reasons"] == ["confirmation_required"]


def test_evaluate_confirmed_allow():
    r = client.post("/evaluate", json={
        "subject": "agent", "action": "send", "object": "gmail",
        "source_provenance": "trusted_user", "capabilities": [_cap(["send"])],
        "revocations": [],
        "confirmations": [{"subject": "agent", "object": "gmail", "action": "send", "issuer": "trusted_user"}]})
    assert r.status_code == 200
    assert r.json()["decision"] == "ALLOW"


def test_mint_signs_capability():
    cap = {"id": "cap1", "subject": "agent", "object": "file", "rights": ["read"],
           "issuer": "trusted_user", "expires_at": FUTURE, "scope": {}, "delegatable": False}
    r = client.post("/mint", json=cap)
    assert r.status_code == 200
    assert r.json()["signature"]  # signature populated


def test_mint_untrusted_issuer_rejected():
    cap = {"id": "cap1", "subject": "agent", "object": "file", "rights": ["read"],
           "issuer": "web_page", "expires_at": FUTURE, "scope": {}, "delegatable": False}
    r = client.post("/mint", json=cap)
    assert r.status_code == 422  # no signing key for an untrusted issuer


def test_invalid_enum_rejected():
    r = client.post("/evaluate", json={
        "subject": "agent", "action": "teleport", "object": "gmail",
        "source_provenance": "trusted_user", "capabilities": [], "revocations": [], "confirmations": []})
    assert r.status_code == 422  # nothing silently allowed
