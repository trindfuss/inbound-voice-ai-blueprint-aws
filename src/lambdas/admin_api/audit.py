"""Audit entries.

One item per change in the ``admin_audit`` table:

* ``pk``: the changed item (``policy#tool#<name>``, ``policy#masking``, ``policy#slot#<name>``,
  ``policy#slot_resolution``, ``policy#redaction``, ``catalog#<key>``, ``user#<username>``); ``sk``: ISO-8601 UTC time with microseconds, ``#`` and a random
  suffix, so entries sort by time and never collide.
* ``ts``, ``action`` (``update``, ``restore``, ``user_create``, ``user_update``,
  ``user_resend``, ``user_delete``), ``target`` (``{"kind", "id"}``), ``actor``
  (``{"sub", "email"}``), ``changes`` (``[{"field", "old", "new"}]``, ``None`` = absent),
  ``security_change``, ``rev`` (the item's new ``rev``, config changes), ``restored_from``
  (``{"pk", "sk"}``, restores) and ``expires_at`` (TTL, now + the stage's retention).

Entries never hold a password, a temporary password or a token: config values and a user's
email, group and status only.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .http import Caller

AUDIT_TABLE = "admin_audit"
#: Fields whose change alters security behaviour: the tool's permission settings; a
#: slot's ``sensitive`` and ``keypad_only`` (phase 1); a
#: slot's ``mode`` and ``confirm``, and ``slot_resolution``'s ``default_mode``, ``flows`` and
#: ``generative_enabled`` (phase 2); and every ``redaction`` field (phase 2),
#: named ``redaction.<field>`` in the entry so a tool's ``mask`` stays a non-security change.
#: ``confirm`` names both a tool's and a slot's setting.
REDACTION_SECURITY_FIELDS: frozenset[str] = frozenset(
    f"redaction.{f}" for f in ("enabled", "entity_types", "min_confidence", "mask", "on_failure"))
SECURITY_FIELDS: frozenset[str] = frozenset({
    "min_level", "confirm", "sensitive", "keypad_only",
    "mode", "default_mode", "flows", "generative_enabled",
}) | REDACTION_SECURITY_FIELDS


def tool_pk(name: str) -> str:
    return f"policy#tool#{name}"


MASKING_PK = "policy#masking"
#: The ``slot_resolution`` policy item (phase 2).
RESOLUTION_PK = "policy#slot_resolution"
#: The transcript ``redaction`` policy item (phase 2).
REDACTION_PK = "policy#redaction"


def slot_pk(name: str) -> str:
    return f"policy#slot#{name}"


def prompt_pk(key: str) -> str:
    return f"catalog#{key}"


def user_pk(username: str) -> str:
    return f"user#{username}"


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def diff(old: Mapping[str, Any], new: Mapping[str, Any], fields: Iterable[str]) -> list[dict[str, Any]]:
    """Field-level changes between two items over ``fields`` (absent = ``None``)."""
    return [{"field": f, "old": old.get(f), "new": new.get(f)} for f in fields
            if old.get(f) != new.get(f) or (f in old) != (f in new)]


@dataclass(frozen=True, slots=True)
class AuditClock:
    """Time and randomness for entries, injectable for tests."""

    now: Callable[[], float]
    suffix: Callable[[], str] = lambda: secrets.token_hex(6)


def entry(*, pk: str, action: str, kind: str, target_id: str, actor: Caller,
          changes: list[dict[str, Any]], security_change: bool, now: float, suffix: str,
          retention_days: int, rev: int | None = None,
          restored_from: Mapping[str, str] | None = None) -> dict[str, Any]:
    ts = iso(now)
    item: dict[str, Any] = {
        "pk": pk, "sk": f"{ts}#{suffix}", "ts": ts, "action": action,
        "target": {"kind": kind, "id": target_id},
        "actor": {"sub": actor.sub, "email": actor.email},
        "changes": changes, "security_change": bool(security_change),
        "expires_at": int(now) + retention_days * 86400,
    }
    if rev is not None:
        item["rev"] = rev
    if restored_from is not None:
        item["restored_from"] = dict(restored_from)
    return item
