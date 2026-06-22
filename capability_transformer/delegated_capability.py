"""Phase 8b — delegated capability chains (macaroon-style attenuation).

`mint_child` lets the *holder* of a parent capability derive an attenuated child offline
(no issuer key). `verify_bundle` re-derives every chain in a bundle and collapses each
capability to Boolean verdict bits the tokenizer writes onto the token matrix. The
attention core then consumes those bits via `head_signature_valid`, `head_chain_valid`
and `head_attenuation_valid`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from . import attenuation, crypto
from .schema import Capability
from .util import aware, is_revoked

TRUSTED_ISSUERS = {"trusted_user", "system"}


@dataclass
class CapVerdict:
    """Per-capability verification result (all booleans, plus failure labels)."""

    sig_valid: bool
    chain_valid: bool
    atten_valid: bool
    fully_valid: bool
    is_delegated: bool
    parent_id: str | None = None
    parent_hash: str | None = None
    failed_restrictions: list[str] = field(default_factory=list)
    chain_error: str | None = None


def mint_child(
    parent: Capability,
    *,
    id: str,
    subject: str | None = None,
    object: str | None = None,
    rights: list[str] | None = None,
    expires_at: datetime | None = None,
    scope: dict | None = None,
    delegatable: bool = False,
    max_delegation_depth: int | None = None,
) -> Capability:
    """Derive a signed, attenuated child capability from a (signed) parent.

    The child inherits parent fields unless overridden, embeds the parent content hash,
    and is signed with a chained HMAC under the parent's signature.
    """
    if not parent.signature:
        raise ValueError("parent capability must be signed before delegation")
    child = Capability(
        id=id,
        subject=subject if subject is not None else parent.subject,
        object=object if object is not None else parent.object,
        rights=list(rights) if rights is not None else list(parent.rights),
        issuer=parent.issuer,
        expires_at=expires_at if expires_at is not None else parent.expires_at,
        scope=scope if scope is not None else dict(parent.scope or {}),
        delegatable=delegatable,
        kid=parent.kid,
        parent_id=parent.id,
        parent_hash=crypto.capability_hash(parent),
        delegation_depth=(parent.delegation_depth or 0) + 1,
        max_delegation_depth=(
            max_delegation_depth
            if max_delegation_depth is not None
            else parent.max_delegation_depth
        ),
    )
    signature = crypto.sign_child(parent.signature, child)
    return child.model_copy(update={"signature": signature})


def verify_bundle(bundle, *, keyring, now: datetime) -> dict[str, CapVerdict]:
    """Verify every capability (root and delegated) in a bundle.

    Returns a map cap.id -> CapVerdict. Cycles and missing parents fail closed.
    """
    caps = {c.id: c for c in bundle.capabilities}
    memo: dict[str, CapVerdict] = {}
    visiting: set[str] = set()

    def expiry_ok(c) -> bool:
        return aware(c.expires_at) > now

    def trusted(c) -> bool:
        return c.issuer in TRUSTED_ISSUERS

    def evaluate(cid: str) -> CapVerdict:
        if cid in memo:
            return memo[cid]
        cap = caps[cid]
        if cid in visiting:  # cyclic parent reference -> fail closed
            v = CapVerdict(False, False, False, False, cap.parent_id is not None,
                           cap.parent_id, cap.parent_hash, ["cycle"], "cycle")
            memo[cid] = v
            return v
        visiting.add(cid)

        if cap.parent_id is None:
            # Root capability: issuer HMAC signature only.
            sig = crypto.verify(cap, keyring=keyring)
            v = CapVerdict(sig, True, True, sig and expiry_ok(cap) and not is_revoked(cap, bundle.revocations) and trusted(cap),
                           False, None, None, [], None)
        else:
            parent = caps.get(cap.parent_id)
            if parent is None:
                v = CapVerdict(False, False, False, False, True,
                               cap.parent_id, cap.parent_hash, ["parent_missing"], "parent_missing")
            else:
                sig = crypto.verify_child(cap, parent)
                hash_ok = cap.parent_hash == crypto.capability_hash(parent)
                pv = evaluate(parent.id)
                parent_can_delegate = "delegate" in parent.rights
                depth_ok = (cap.max_delegation_depth is None
                            or (cap.delegation_depth or 0) <= cap.max_delegation_depth)
                chain_error = None
                if not hash_ok:
                    chain_error = "parent_hash_mismatch"
                elif not pv.fully_valid:
                    chain_error = "parent_invalid"
                elif not parent_can_delegate:
                    chain_error = "parent_lacks_delegate"
                elif not depth_ok:
                    chain_error = "max_depth_exceeded"
                chain_ok = chain_error is None
                atten_ok, failures = attenuation.check(parent, cap)
                fully = bool(sig and chain_ok and atten_ok
                             and expiry_ok(cap) and not is_revoked(cap, bundle.revocations) and trusted(cap))
                v = CapVerdict(bool(sig), chain_ok, atten_ok, fully, True,
                               cap.parent_id, cap.parent_hash, failures, chain_error)

        visiting.discard(cid)
        memo[cid] = v
        return v

    for cid in caps:
        evaluate(cid)
    return memo
