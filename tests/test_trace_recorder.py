"""The step recorder must faithfully replay the compiled forward pass."""

from fastapi.testclient import TestClient

from capability_transformer import CapabilityTransformer, trace_recorder
from capability_transformer.api import app
from capability_transformer.schema import Capability, CapabilityBundle

client = TestClient(app)
FUT = "2099-01-01T00:00:00Z"


def _bundle(action="send", object="gmail", provenance="retrieved_doc", rights=("send",)):
    return CapabilityBundle(subject="agent", action=action, object=object,
                            source_provenance=provenance,
                            capabilities=[Capability(id="c", subject="agent", object=object,
                                                     rights=list(rights), issuer="trusted_user",
                                                     expires_at=FUT)])


def test_trace_structure_and_decision():
    b = _bundle()
    t = trace_recorder.record(b)
    assert t["decision"] == CapabilityTransformer().evaluate(b).decision
    assert t["matches_reference"] is True
    assert t["steps"][0]["layer"] == "embedding"
    assert t["steps"][-1]["kind"] == "output"
    assert t["steps"][-1]["detail"]["decision"] == t["decision"]
    assert {s["layer"] for s in t["steps"]} >= {"embedding", "attention", "ffn_per_cap",
                                                "pool", "ffn_decision", "output"}


def test_every_snapshot_has_consistent_shape():
    t = trace_recorder.record(_bundle())
    n_tokens = len(t["tokens"])
    width = t["layout"]["width"]
    for s in t["steps"]:
        assert len(s["snapshot"]) == n_tokens
        assert all(len(row) == width for row in s["snapshot"])


def test_changed_cells_are_exactly_the_diff():
    # Each step's snapshot differs from the previous one only at the cells it reports.
    t = trace_recorder.record(_bundle())
    steps = t["steps"]
    for i in range(1, len(steps)):
        prev, cur = steps[i - 1]["snapshot"], steps[i]["snapshot"]
        reported = {(r, c) for r, c in steps[i]["changed"]}
        actual = {(r, c) for r in range(len(cur)) for c in range(len(cur[0]))
                  if abs(cur[r][c] - prev[r][c]) > 1e-9}
        assert actual == reported, (i, steps[i]["op"], actual ^ reported)


def test_recorder_is_deterministic():
    b = _bundle()
    a, c = trace_recorder.record(b), trace_recorder.record(b)
    assert [s["snapshot"] for s in a["steps"]] == [s["snapshot"] for s in c["steps"]]


def test_match_step_score_drives_evidence_bit():
    t = trace_recorder.record(_bundle(action="send", object="gmail", rights=["send"]))
    sm = next(s for s in t["steps"] if s["op"] == "head:subject_match")
    for m in sm["detail"]["matches"]:
        assert m["value"] == (1.0 if m["score"] >= 0.5 else 0.0)


# ---- API + UI ------------------------------------------------------------------------
def test_trace_endpoint():
    r = client.post("/trace", json={"bundle": _bundle().model_dump(mode="json")})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in ("ALLOW", "DENY", "ESCALATE")
    assert len(body["steps"]) > 10


def test_trace_examples_endpoint():
    presets = client.get("/trace/examples").json()
    ids = {p["id"] for p in presets}
    assert {"deny_injection", "cross_cap_leak", "delegation"} <= ids
    for p in presets:                       # every preset traces and matches the reference
        r = client.post("/trace", json={"bundle": p["bundle"], **p["config"]})
        assert r.status_code == 200
        assert r.json()["matches_reference"] is True


def test_model_head_endpoint():
    m = client.get("/model/head/subject_match").json()
    assert len(m["Wq"]) == 5 and len(m["Wq"][0]) == 132
    assert client.get("/model/head/nope").status_code == 404


def test_ui_is_served():
    assert "app.js" in client.get("/ui/").text
    assert client.get("/ui/app.js").status_code == 200
    assert client.get("/ui/style.css").status_code == 200
