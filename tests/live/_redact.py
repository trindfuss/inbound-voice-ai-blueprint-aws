"""Redaction of recorded HTTP exchanges for the live contract test (kept apart so a unit test can import it)."""

from __future__ import annotations

from typing import Any

REDACTED = "[REDACTED]"
SENSITIVE_KEY_PARTS = ("token", "secret", "password")


def redact(value: Any) -> Any:
    """A copy of ``value`` (nested JSON) where every value under a key containing ``token``, ``secret`` or
    ``password`` (case-insensitive), at any depth, is replaced by ``[REDACTED]``."""
    if isinstance(value, dict):
        return {k: REDACTED if any(p in str(k).lower() for p in SENSITIVE_KEY_PARTS) else redact(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value
