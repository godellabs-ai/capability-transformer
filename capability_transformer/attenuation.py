"""Phase 8b — attenuation predicates for delegated capabilities.

A child capability may only be *weaker than or equal to* its parent. These are the
structural restriction checks (the "caveats" of our subset). They are recomputed by the
gateway — a self-asserted "rights_subset: true" on the token is never trusted.

Returns (ok, failed_restrictions) so the trace can name the first violated restriction.
"""

from __future__ import annotations

from .util import aware


def _scope_not_widened(parent_scope: dict, child_scope: dict) -> bool:
    """Child must keep every constraint the parent imposed (it may add more)."""
    parent_scope = parent_scope or {}
    child_scope = child_scope or {}
    return all(child_scope.get(k) == v for k, v in parent_scope.items())


def check(parent, child) -> tuple[bool, list[str]]:
    """Verify child ≤ parent. Returns (ok, list_of_failed_restrictions)."""
    failures: list[str] = []
    parent_can_delegate = "delegate" in parent.rights

    # rights: child rights ⊆ parent rights
    if not set(child.rights).issubset(set(parent.rights)):
        failures.append("rights_not_subset")

    # object: must be identical (object change is not an attenuation)
    if child.object != parent.object:
        failures.append("object_changed")

    # scope: child may not widen (remove/loosen) a parent constraint
    if not _scope_not_widened(parent.scope, child.scope):
        failures.append("scope_widened")

    # expiry: child cannot outlive parent
    if aware(child.expires_at) > aware(parent.expires_at):
        failures.append("expiry_extended")

    # subject change requires explicit delegation authority on the parent
    if child.subject != parent.subject and not parent_can_delegate:
        failures.append("subject_change_not_permitted")

    # re-delegation: child may be delegatable only if the parent can delegate and the
    # depth budget still permits another hop
    if child.delegatable:
        depth_budget_ok = (
            child.max_delegation_depth is None
            or (child.delegation_depth or 0) < child.max_delegation_depth
        )
        if not parent_can_delegate or not depth_budget_ok:
            failures.append("redelegation_not_permitted")

    # hard depth ceiling
    if child.max_delegation_depth is not None and (child.delegation_depth or 0) > child.max_delegation_depth:
        failures.append("max_depth_exceeded")

    return (not failures, failures)
