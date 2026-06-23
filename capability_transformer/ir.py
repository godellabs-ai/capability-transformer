"""Declarative policy IR for the capability authorization machine.

This module describes *what* the authorization policy checks, as a small typed graph,
without committing to *how* it is executed. The reference evaluator
(``CapabilityTransformer``) and the compiled transformer-style evaluator
(``CompiledCapabilityTransformer``) are two executions of the same policy. The compiler
in ``compiler.py`` reads this IR to decide which attention heads and feed-forward gates to
construct; the resulting weights are analytical and deterministic.

The IR makes the security-critical structure explicit, in particular the **existential**
shape of the matching rule:

    ALLOW requires:  ∃ capability c such that  (subject_match(c) ∧ object_match(c) ∧
        right_match(c) ∧ issuer_trusted(c) ∧ not_expired(c) ∧ not_revoked(c) ∧
        signature_valid(c) ∧ chain_ok(c) ∧ attenuation_ok(c))

Evidence is computed **per capability** and only then aggregated. We never combine
``any_subject_match`` with ``any_object_match`` from *different* capabilities — that would
be unsound. The IR encodes the per-capability conjunction as a node whose inputs are all
the same capability's predicates, and a separate existential-aggregation node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeKind(str, Enum):
    # Per-capability predicates produced by attention heads (request/policy comparison).
    CAP_MATCH = "cap_match"            # exact match of a capability field vs request/policy
    # Per-capability predicates that are intrinsic token features (verified upstream).
    CAP_FEATURE = "cap_feature"       # expiry, revoked, signature, chain, attenuation, scope
    # Request-level predicates produced by attention heads.
    REQUEST_MATCH = "request_match"   # provenance trusted, action is read, high-risk, ...
    REQUEST_FEATURE = "request_feature"
    # Per-confirmation predicates.
    CONF_MATCH = "conf_match"
    CONF_FEATURE = "conf_feature"
    # Boolean composition in the feed-forward gates.
    AND = "and"
    OR = "or"
    NOT = "not"
    # Existential aggregation over a token set (∃) — attention max-pool.
    EXISTS = "exists"
    # Final output classes.
    OUTPUT = "output"


class TokenSet(str, Enum):
    CAPABILITIES = "capabilities"
    CONFIRMATIONS = "confirmations"
    REQUEST = "request"


@dataclass(frozen=True)
class Node:
    """A node in the policy graph."""

    name: str
    kind: NodeKind
    inputs: tuple[str, ...] = ()
    # For match nodes: which residual sub-field the query reads and which key it compares to.
    query_field: str | None = None      # e.g. "subject", "rights"
    key_field: str | None = None        # e.g. "subject", or a policy mask name
    key_token: str | None = None        # "request" | "policy" | "self"
    # For EXISTS nodes: the token set being aggregated and the per-token predicate input.
    over: TokenSet | None = None
    # For OUTPUT nodes: the decision class.
    decision_class: str | None = None
    note: str = ""


@dataclass
class PolicyIR:
    """The full policy graph."""

    nodes: list[Node] = field(default_factory=list)

    def add(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def by_name(self, name: str) -> Node:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def of_kind(self, kind: NodeKind) -> list[Node]:
        return [n for n in self.nodes if n.kind == kind]


def build_policy_ir() -> PolicyIR:
    """Construct the policy graph for the bounded object-capability machine."""
    ir = PolicyIR()

    # --- per-capability predicates (attention heads: capability vs request/policy) -----
    ir.add(Node("subject_match", NodeKind.CAP_MATCH, query_field="subject",
                key_field="subject", key_token="request",
                note="cap.subject == request.subject"))
    ir.add(Node("object_match", NodeKind.CAP_MATCH, query_field="object",
                key_field="object", key_token="request",
                note="cap.object == request.object"))
    ir.add(Node("right_match", NodeKind.CAP_MATCH, query_field="rights",
                key_field="action", key_token="request",
                note="request.action in cap.rights (multi-hot dot >= 1)"))
    ir.add(Node("issuer_trusted", NodeKind.CAP_MATCH, query_field="issuer",
                key_field="trusted_issuer_mask", key_token="policy",
                note="cap.issuer in {trusted_user, system}"))
    ir.add(Node("has_delegate", NodeKind.CAP_MATCH, query_field="rights",
                key_field="delegate_mask", key_token="policy",
                note="'delegate' in cap.rights"))
    ir.add(Node("has_target_right", NodeKind.CAP_MATCH, query_field="rights",
                key_field="delegate_right", key_token="request",
                note="request.delegate_right in cap.rights"))

    # --- per-capability intrinsic features (verified at tokenization) ------------------
    for feat, note in [
        ("not_expired", "expiry bit (expires_at > now)"),
        ("not_revoked", "NOT revoked bit"),
        ("signature_valid", "HMAC signature bit"),
        ("chain_ok", "delegation chain bit OR not delegated"),
        ("attenuation_ok", "attenuation bit OR not delegated"),
        ("scope_nonempty", "capability carries a scope constraint"),
        ("scope_mismatch", "request scope does not satisfy capability scope"),
    ]:
        ir.add(Node(feat, NodeKind.CAP_FEATURE, note=note))

    # --- per-capability conjunction (THE security boundary, per capability) ------------
    ir.add(Node("valid_capability", NodeKind.AND,
                inputs=("subject_match", "object_match", "right_match", "issuer_trusted",
                        "not_expired", "not_revoked", "signature_valid", "chain_ok",
                        "attenuation_ok"),
                note="all required predicates hold for THE SAME capability"))
    ir.add(Node("delegator_capability", NodeKind.AND,
                inputs=("subject_match", "object_match", "issuer_trusted", "not_expired",
                        "not_revoked", "has_delegate", "has_target_right"),
                note="a valid delegator holds both delegate and the target right"))
    ir.add(Node("scope_violation", NodeKind.AND,
                inputs=("valid_capability", "scope_nonempty", "scope_mismatch"),
                note="a matched capability whose scope the request violates"))

    # --- existential aggregation over capabilities (∃) ---------------------------------
    ir.add(Node("has_match", NodeKind.EXISTS, inputs=("valid_capability",),
                over=TokenSet.CAPABILITIES, note="∃ a fully valid matching capability"))
    ir.add(Node("delegation_pass", NodeKind.EXISTS, inputs=("delegator_capability",),
                over=TokenSet.CAPABILITIES, note="∃ a valid delegator capability"))
    ir.add(Node("scope_violated", NodeKind.EXISTS, inputs=("scope_violation",),
                over=TokenSet.CAPABILITIES, note="∃ a matched capability violating scope"))

    # --- request-level predicates (attention heads on the request token) ---------------
    ir.add(Node("prov_trusted", NodeKind.REQUEST_MATCH, query_field="provenance",
                key_field="trusted_prov_mask", key_token="policy",
                note="source_provenance in {trusted_user, system_policy}"))
    ir.add(Node("action_is_read", NodeKind.REQUEST_MATCH, query_field="rights",
                key_field="read_mask", key_token="policy",
                note="the action is a passive read"))
    ir.add(Node("high_risk_action", NodeKind.REQUEST_MATCH, query_field="object",
                key_field="high_risk", key_token="self",
                note="object^T HIGH_RISK action"))
    ir.add(Node("action_is_delegate", NodeKind.REQUEST_MATCH, query_field="rights",
                key_field="delegate_mask", key_token="policy",
                note="the action is 'delegate'"))
    ir.add(Node("delegate_present", NodeKind.REQUEST_FEATURE,
                note="bundle.delegate_right is set"))

    # --- per-confirmation predicates ---------------------------------------------------
    ir.add(Node("conf_subject", NodeKind.CONF_MATCH, query_field="subject",
                key_field="subject", key_token="request"))
    ir.add(Node("conf_object", NodeKind.CONF_MATCH, query_field="object",
                key_field="object", key_token="request"))
    ir.add(Node("conf_action", NodeKind.CONF_MATCH, query_field="rights",
                key_field="action", key_token="request"))
    ir.add(Node("conf_issuer", NodeKind.CONF_MATCH, query_field="issuer",
                key_field="trusted_issuer_mask", key_token="policy"))
    ir.add(Node("conf_bind", NodeKind.CONF_FEATURE, note="action-hash binding bit"))
    ir.add(Node("conf_valid", NodeKind.AND,
                inputs=("conf_subject", "conf_object", "conf_action", "conf_issuer",
                        "conf_bind"),
                note="a confirmation that authorizes THIS exact action"))
    ir.add(Node("confirmed", NodeKind.EXISTS, inputs=("conf_valid",),
                over=TokenSet.CONFIRMATIONS, note="∃ a valid action-bound confirmation"))

    # --- decision composition (feed-forward gates) -------------------------------------
    ir.add(Node("prov_ok", NodeKind.OR, inputs=("prov_trusted", "action_is_read")))
    ir.add(Node("scope_ok", NodeKind.NOT, inputs=("scope_violated",)))
    ir.add(Node("is_delegation", NodeKind.AND, inputs=("action_is_delegate", "delegate_present")))
    ir.add(Node("not_is_delegation", NodeKind.NOT, inputs=("is_delegation",)))
    ir.add(Node("delegation_ok", NodeKind.OR, inputs=("not_is_delegation", "delegation_pass")))
    ir.add(Node("required_ok", NodeKind.AND,
                inputs=("has_match", "prov_ok", "scope_ok", "delegation_ok")))
    ir.add(Node("not_required_ok", NodeKind.NOT, inputs=("required_ok",)))
    ir.add(Node("not_confirmed", NodeKind.NOT, inputs=("confirmed",)))
    ir.add(Node("not_high_risk", NodeKind.NOT, inputs=("high_risk_action",)))
    ir.add(Node("escalate_evidence", NodeKind.AND,
                inputs=("required_ok", "high_risk_action", "not_confirmed")))
    ir.add(Node("low_or_confirmed", NodeKind.OR, inputs=("not_high_risk", "confirmed")))
    ir.add(Node("allow_evidence", NodeKind.AND, inputs=("required_ok", "low_or_confirmed")))
    ir.add(Node("deny_evidence", NodeKind.NOT, inputs=("required_ok",)))

    # --- output projection -------------------------------------------------------------
    ir.add(Node("ALLOW", NodeKind.OUTPUT, inputs=("allow_evidence",), decision_class="ALLOW"))
    ir.add(Node("DENY", NodeKind.OUTPUT, inputs=("deny_evidence",), decision_class="DENY"))
    ir.add(Node("ESCALATE", NodeKind.OUTPUT, inputs=("escalate_evidence",),
                decision_class="ESCALATE"))
    return ir
