# capability-transformer

**Attention as Capability Machine** — a deterministic, *transformer-native* capability
enforcement gateway that sits in front of an LLM / tool-calling system (e.g. ChatGPT)
and answers `ALLOW` / `DENY` / `ESCALATE` for every tool action, with a machine-readable
reason trace.

---

## What this is

A standalone authorization **gateway**. Given a formalized request
(`subject`, `action`, `object`, `source_provenance`) plus a bundle of capability tokens,
revocations and confirmations, it decides whether the action is authorized — and proves
it with a per-attention-head audit trace.

The enforcement core is a **bounded finite transformer-like machine**. The request is the
attention *query*; capabilities are *keys/values*; the security boundary is a set of
**hard (Boolean) attention masks** computed with `numpy` tensors. No softmax, no trained
weights, no rules engine.

## What this is *not*

- **Not** OPA / Rego / Cedar / Casbin / Prolog / Datalog / any policy engine.
- **Not** a pile of `if/else` authorization checks dressed up as a product — the
  enforcement path is a token matrix processed by hard-attention heads.
- **Not** a trained model: fixed/compiled tensors only, no gradient descent, no softmax
  used as a security boundary.
- **Not** a tool executor — it returns a *decision only* and performs no real
  Gmail/Slack/browser/file side effects.
- **Not** production security. Capability issuance is a *mock* (label check, not a real
  signature). See "Warning" below.

## Why transformer-native

Object-capability security maps cleanly onto attention:

| Capability concept            | Attention concept                          |
|-------------------------------|--------------------------------------------|
| request seeking authority     | **query** token                            |
| possessed capabilities        | **key** tokens                             |
| rights / issuer / expiry bits | **value** tokens                           |
| the security boundary         | **hard attention mask** (Boolean, no softmax) |
| subject/object/right/… checks | **multi-head** attention                   |
| the decision                  | deterministic **reducer** (FFN-like projection) |

Execution shape:

```
bundle ─▶ tokenizer.encode ─▶ X (N×D token matrix)
       ─▶ hard_attention.compute ─▶ head masks
       ─▶ deterministic reducer ─▶ Decision (ALLOW/DENY/ESCALATE + reasons)
       ─▶ trace renderer ─▶ JSON
```

See [`implementation.md`](implementation.md) for the full design, threat model and
phase plan.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

(Requires Python 3.11+. Dependencies: `numpy`, `pydantic`, `fastapi`, `uvicorn`;
`pytest` + `httpx` for tests.)

## Run tests

```bash
pytest
```

The suite includes an **exhaustive bounded** test that enumerates every
subject × object × right combination and prints a coverage summary.

## Run the API

```bash
uvicorn capability_transformer.api:app --reload
# or:
python -m capability_transformer.api
```

Endpoints:

- `POST /evaluate`  — evaluate a request bundle → decision + trace
- `POST /authorize` — (Phase 8c) evaluate + issue a fresh execution grant on ALLOW
- `POST /execute`   — (Phase 8c) run a mock tool, but only for a valid grant (fail-closed)
- `POST /mint`      — (Phase 8a) sign a capability with the demo issuer keyring
- `GET  /health`    — liveness
- `GET  /schema`    — bounded vocabularies + JSON schema
- `GET  /examples`  — bundled example requests

## Example curl commands

Deny (untrusted document tries to send mail; only `draft` is granted):

```bash
curl -s localhost:8000/evaluate -H 'content-type: application/json' -d '{
  "subject":"agent","action":"send","object":"gmail",
  "source_provenance":"retrieved_doc",
  "capabilities":[{"id":"cap1","subject":"agent","object":"gmail",
    "rights":["draft"],"issuer":"trusted_user",
    "expires_at":"2099-01-01T00:00:00Z","scope":{},"delegatable":false}],
  "revocations":[],"confirmations":[]}'
# -> {"decision":"DENY","reasons":["right_not_granted","data_has_no_authority"], ...}
```

Allow (trusted user, `draft` granted, low-risk):

```bash
curl -s localhost:8000/evaluate -H 'content-type: application/json' -d '{
  "subject":"agent","action":"draft","object":"gmail",
  "source_provenance":"trusted_user",
  "capabilities":[{"id":"cap1","subject":"agent","object":"gmail",
    "rights":["draft"],"issuer":"trusted_user",
    "expires_at":"2099-01-01T00:00:00Z","scope":{},"delegatable":false}],
  "revocations":[],"confirmations":[]}'
# -> {"decision":"ALLOW","reasons":["allowed"], ...}
```

Escalate (high-risk `gmail.send` with capability but no confirmation):

```bash
curl -s localhost:8000/evaluate -H 'content-type: application/json' -d '{
  "subject":"agent","action":"send","object":"gmail",
  "source_provenance":"trusted_user",
  "capabilities":[{"id":"cap1","subject":"agent","object":"gmail",
    "rights":["send"],"issuer":"trusted_user",
    "expires_at":"2099-01-01T00:00:00Z","scope":{},"delegatable":false}],
  "revocations":[],"confirmations":[]}'
# -> {"decision":"ESCALATE","reasons":["confirmation_required"], ...}
```

Add a trusted confirmation to the body above and the same request returns `ALLOW`.

## ALLOW / DENY / ESCALATE

- **ALLOW** — a possessed capability matches the request on subject, object and right;
  is issued by a trusted issuer; is not expired and not revoked; the provenance is
  authorized to drive the action; and either the action is low-risk or a trusted
  confirmation is present.
- **DENY** — no such capability exists, or a hard security predicate fails (wrong
  subject/object/right, untrusted issuer, expired, revoked, untrusted data driving a
  side effect, disallowed delegation). All failing reason codes are returned.
- **ESCALATE** — authority exists and all hard checks pass, but the action is
  **high-risk** (`gmail.send`, `slack.post`, `file.delete`, `secrets_db.read`,
  `browser.invoke`) and no trusted confirmation token is present. Route to a human.

## Cryptographically authenticated & attenuable capabilities (Phase 8a/8b)

By default the engine trusts a capability's `issuer` *label* (v1 behavior). Run with
signature enforcement to get **cryptographically authenticated capabilities under a
trusted symmetric-key issuer model**:

```python
from capability_transformer import CapabilityTransformer, Capability, crypto

engine = CapabilityTransformer(require_signatures=True)
cap = Capability(id="c1", subject="agent", object="file", rights=["read", "delegate"],
                 issuer="trusted_user", expires_at="2099-01-01T00:00:00Z")
cap = crypto.issue(cap)   # issuer signs it (populates kid + signature)
# A capability with a missing/forged/tampered signature now DENYs with
# reason "invalid_signature" — even if every other field matches.
```

In signed mode the failure semantics are explicit and unambiguous:

| situation                         | decision / reason          |
|-----------------------------------|----------------------------|
| unsigned cap, trusted issuer      | `DENY [invalid_signature]` |
| malformed / forged signature      | `DENY [invalid_signature]` |
| unknown / untrusted issuer        | `DENY [issuer_not_trusted]`|

**Attenuable delegation (Phase 8b, macaroon-style chained HMAC).** The *holder* of a
capability can mint an attenuated child offline — no issuer key needed — and the gateway
re-derives the chain:

```python
from capability_transformer.delegated_capability import mint_child

child = mint_child(cap, id="c2", subject="agent", rights=["read"])  # weaker-or-equal only
# Child rights ⊆ parent, expiry ≤ parent, scope not widened, subject change needs
# `delegate`. Tampering breaks the signature; revoking/expiring the parent kills the
# child. Trace exposes delegation_chain_valid / attenuation_valid / parent_hash.
```

Each crypto check is reduced to a Boolean bit (`signature`, `chain`, `attenuation`)
consumed by the `head_signature_valid`, `head_chain_valid` and `head_attenuation_valid`
hard-attention heads — so the enforcement path stays a pure tensor pipeline. Run the demo:

```bash
PYTHONPATH=. python examples/signed_capability_demo.py
```

This remains a *mock*: a symmetric, shared per-issuer secret with a single verifier, and
a subset of macaroon semantics (no third-party/discharge caveats). Production should use
asymmetric signatures (Ed25519) or real macaroons — see `implementation.md` §21–§22.

## Gated tool runtime — the enforcement boundary (Phase 8c)

Through Phase 8b the service was an *evaluator*: it returned a decision but gated nothing.
Phase 8c adds the component that actually holds the (mock) tools and **refuses to run
anything without a fresh, action-bound, single-use grant** signed by the gateway:

```python
from capability_transformer import ToolCall
from capability_transformer.runtime import ToolGateway, GatedToolRuntime

gateway, runtime = ToolGateway(), GatedToolRuntime()
call = ToolCall(subject="agent", action="draft", object="gmail",
                args={"to": "bob@example.com", "body": "hi"})

decision, grant = gateway.authorize(bundle, call)   # grant is None unless ALLOW
result = runtime.execute(grant, call)               # runs ONLY for a valid grant
```

The grant is **action-bound** (carries a hash of the exact call + args), **time-bound**
(default 30s TTL) and **single-use** (nonce consumed on execution). Everything fails
closed:

| situation                         | runtime result                          |
|-----------------------------------|-----------------------------------------|
| DENY / ESCALATE                   | no grant → `refused: no_grant`          |
| replay a used grant               | `refused: grant_replayed`               |
| tampered grant (e.g. action swap) | `refused: grant_signature_invalid`      |
| expired grant                     | `refused: grant_expired`                |
| grant args ≠ call args            | `refused: action_binding_mismatch`      |

Run the demo:

```bash
PYTHONPATH=. python examples/gated_runtime_demo.py
```

The runtime trusts only a grant whose HMAC it can verify with the shared gateway↔runtime
secret — never the LLM, the caller, or a bare decision object. Tools are mocks: no real
side effects occur.

**Action-bound confirmations (Phase 8d).** A high-risk confirmation can be bound to the
hash of the *exact* action (subject, action, object, args), so a human approval of "send
to bob" cannot be replayed to authorize "send to attacker":

```python
engine = CapabilityTransformer(require_bound_confirmations=True)  # accept only bound confirmations
```

With `require_bound_confirmations=True`, an unbound or mismatched confirmation yields
`ESCALATE` (and therefore no grant). `ToolGateway.authorize` sets the bundle's
`action_hash` from the concrete `ToolCall`, so binding is enforced end to end.

## ⚠️ Warning — prototype, not production security

This is a **research prototype**. Capability issuance is *mocked*: trust is decided by an
`issuer` label, not by verifying an unforgeable cryptographic grant. Do not rely on it to
secure real systems.

**Production enforcement must happen at the actual tool gateway** — the component that
holds the real Gmail/Slack/file credentials must itself call an enforcement service like
this one and execute *only* on `ALLOW`. The LLM must never be trusted to enforce policy,
and capabilities must be unforgeable (signed) end to end.
