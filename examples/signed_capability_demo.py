"""Phase 8a demo — unforgeable capabilities.

Run:  python examples/signed_capability_demo.py

Shows that with signature enforcement on:
  * a properly issuer-signed capability is honored, but
  * the same capability with a forged issuer label / tampered field is rejected.
"""

from datetime import datetime, timezone

from capability_transformer import Capability, CapabilityBundle, CapabilityTransformer, crypto

FUTURE = datetime(2099, 1, 1, tzinfo=timezone.utc)


def signed(**over) -> Capability:
    base = dict(id="cap1", subject="agent", object="file", rights=["read"],
                issuer="trusted_user", expires_at=FUTURE)
    base.update(over)
    cap = Capability(**base)
    return crypto.issue(cap)  # populates kid + signature


def request(cap: Capability, action="read") -> CapabilityBundle:
    return CapabilityBundle(subject="agent", action=action, object="file",
                            source_provenance="trusted_user", capabilities=[cap])


def main() -> None:
    engine = CapabilityTransformer(require_signatures=True)

    genuine = signed()
    print("genuine signed cap        ->", engine.evaluate(request(genuine)).decision,
          engine.evaluate(request(genuine)).reasons)

    # Attacker forges authority by claiming a trusted issuer but cannot sign for it.
    forged = Capability(id="cap1", subject="agent", object="file", rights=["read"],
                        issuer="trusted_user", expires_at=FUTURE)  # no signature
    d = engine.evaluate(request(forged))
    print("forged (unsigned) cap     ->", d.decision, d.reasons)

    # Attacker keeps a real signature but escalates the granted right: HMAC breaks.
    tampered = genuine.model_copy(update={"rights": ["read", "write"]})
    d = engine.evaluate(request(tampered, action="write"))
    print("tampered (amplified) cap  ->", d.decision, d.reasons)


if __name__ == "__main__":
    main()
