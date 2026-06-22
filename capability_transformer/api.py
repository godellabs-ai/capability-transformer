"""FastAPI gateway.

The gateway returns a *decision only* — it never performs real tool side effects.
Tool execution must happen downstream and only on ``ALLOW``.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI

from . import compiled_weights as W
from .core import CapabilityTransformer
from .schema import CapabilityBundle, Decision

app = FastAPI(
    title="capability-transformer",
    version="0.1.0",
    description="Attention as Capability Machine — transformer-native capability gateway.",
)

_engine = CapabilityTransformer()
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": W.ENGINE_NAME, "trained": False, "softmax_used": False}


@app.post("/evaluate", response_model=Decision)
def evaluate(bundle: CapabilityBundle) -> Decision:
    """Evaluate a request bundle and return a decision + audit trace."""
    return _engine.evaluate(bundle)


@app.get("/schema")
def schema() -> dict:
    """Return the bounded vocabularies and the request JSON schema."""
    return {
        "subjects": W.SUBJECTS,
        "objects": W.OBJECTS,
        "rights": W.RIGHTS,
        "issuers": W.ISSUERS,
        "trusted_issuers": ["trusted_user", "system"],
        "provenance": W.PROVENANCE,
        "decisions": W.DECISIONS,
        "reason_codes": W.REASON_CODES,
        "high_risk_actions": [
            "gmail.send",
            "slack.post",
            "file.delete",
            "secrets_db.read",
            "browser.invoke",
        ],
        "request_schema": CapabilityBundle.model_json_schema(),
    }


@app.get("/examples")
def examples() -> dict:
    """Return the bundled example request bodies."""
    out = {}
    if _EXAMPLES_DIR.is_dir():
        for path in sorted(_EXAMPLES_DIR.glob("*.json")):
            out[path.stem] = json.loads(path.read_text())
    return out


def main() -> None:  # pragma: no cover - manual entry point
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
