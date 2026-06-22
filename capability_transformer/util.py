"""Small shared helpers used by the tokenizer and the delegation verifier."""

from __future__ import annotations

from datetime import datetime, timezone


def aware(dt: datetime) -> datetime:
    """Normalize to a timezone-aware UTC datetime for safe comparison."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_revoked(cap, revocations) -> bool:
    """A capability is revoked if any revocation matches it by id or by fields."""
    for rev in revocations:
        if rev.capability_id is not None:
            if rev.capability_id == cap.id:
                return True
            continue
        subj_ok = rev.subject is None or rev.subject == cap.subject
        obj_ok = rev.object is None or rev.object == cap.object
        if subj_ok and obj_ok and (rev.subject is not None or rev.object is not None):
            return True
    return False
