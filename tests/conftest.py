"""Shared test fixtures and builders."""

from __future__ import annotations

import pytest

from capability_transformer import CapabilityBundle, CapabilityTransformer

FUTURE = "2099-01-01T00:00:00Z"
PAST = "2000-01-01T00:00:00Z"


@pytest.fixture
def engine() -> CapabilityTransformer:
    return CapabilityTransformer()


def cap(
    id="cap1",
    subject="agent",
    object="gmail",
    rights=("draft",),
    issuer="trusted_user",
    expires_at=FUTURE,
    scope=None,
    delegatable=False,
):
    return {
        "id": id,
        "subject": subject,
        "object": object,
        "rights": list(rights),
        "issuer": issuer,
        "expires_at": expires_at,
        "scope": scope or {},
        "delegatable": delegatable,
    }


def bundle(
    subject="agent",
    action="draft",
    object="gmail",
    source_provenance="trusted_user",
    capabilities=None,
    revocations=None,
    confirmations=None,
    **extra,
):
    return CapabilityBundle(
        subject=subject,
        action=action,
        object=object,
        source_provenance=source_provenance,
        capabilities=capabilities if capabilities is not None else [cap()],
        revocations=revocations or [],
        confirmations=confirmations or [],
        **extra,
    )
