"""Reviewer walkthrough of the compiled transformer-style evaluator.

Run:  PYTHONPATH=. python examples/compiled_transformer_demo.py

Shows: the architecture summary, the attention heads with their Q/K projection shapes,
the feed-forward Boolean gates, and a full decision walked from tokens -> attention heads
-> per-capability conjunction -> existential aggregation -> output projection. The compiled
evaluator's decisions match the reference evaluator (see tests/test_compiled_equivalence).
"""

from capability_transformer import CapabilityTransformer, CompiledCapabilityTransformer, inspection
from capability_transformer.compiler import compile_policy
from capability_transformer.schema import Capability, CapabilityBundle

FUTURE = "2099-01-01T00:00:00Z"


def cap(rights, object="gmail"):
    return Capability(id="c1", subject="agent", object=object, rights=list(rights),
                      issuer="trusted_user", expires_at=FUTURE)


def main() -> None:
    model = compile_policy()
    print("=== compiled architecture ===")
    for k, v in inspection.summary(model).items():
        print(f"  {k}: {v}")

    print("\n=== attention heads (exact capability selectors) ===")
    for h in inspection.describe_heads(model):
        if h["kind"] == "match":
            print(f"  {h['head']:18} Q{h['Wq_shape']} x K{h['Wk_shape']}  "
                  f"{h['query_set']} -> {h['key_token']}  => {h['out_slot']}")

    print("\n=== feed-forward Boolean gates (decision) ===")
    for g in inspection.describe_gates(model):
        if g["token_set"] == "output":
            print(f"  {g['gate']:18} {g['op']:3} {list(g['inputs'])} => {g['out_slot']}")

    print("\n=== walk a decision: untrusted document drives gmail.send ===")
    b = CapabilityBundle(subject="agent", action="send", object="gmail",
                         source_provenance="retrieved_doc", capabilities=[cap(["send"])])
    walk = inspection.inspect_decision(b)
    print("  per-capability evidence:", walk["per_capability_evidence"])
    print("  output evidence:", {k: v for k, v in walk["output_decision_evidence"].items() if v})
    print("  logits:", walk["logits"], "-> decision:", walk["decision"])

    print("\n=== compiled matches reference ===")
    ref, comp = CapabilityTransformer(), CompiledCapabilityTransformer()
    for action, prov in [("draft", "trusted_user"), ("send", "trusted_user"),
                         ("send", "retrieved_doc"), ("read", "retrieved_doc")]:
        bb = CapabilityBundle(subject="agent", action=action, object="gmail",
                              source_provenance=prov, capabilities=[cap([action])])
        r, c = ref.evaluate(bb).decision, comp.evaluate(bb).decision
        print(f"  agent {action:5} gmail (prov={prov:13}) -> reference={r:8} compiled={c:8} "
              f"{'OK' if r == c else 'MISMATCH'}")


if __name__ == "__main__":
    main()
