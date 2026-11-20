"""Web-activity profile traits and the filter every profile trait passes before the model.

A feed such as Segment writes what a customer recently looked at on the bank's website to the
Memory profile, as the trait group ``WebActivity`` (``TRAITS``). This module is the one source for
those names: the bridge, ``twilio_setup.py`` and the demo trait writer import them.

``bound_profile_traits`` decides, in code rather than in the model, which traits may reach the
prompt:

- only groups on the allow-list;
- per group at most ``MAX_TRAITS_PER_GROUP`` traits whose names match ``TRAIT_NAME_RE``; in
  ``WebActivity`` the ``TRAITS`` names are taken first, so the cap never drops ``lastViewedAt``;
- values: a string (control and format characters removed, surrounding whitespace stripped, cut to
  ``MAX_VALUE_CHARS``, dropped if empty), an int or finite float of magnitude at most
  ``MAX_NUMBER_ABS``, or a bool; anything else is dropped;
- ``WebActivity`` is dropped whole unless ``lastViewedAt`` is an ISO-8601 date and time (a date alone
  or a week or ordinal date doesn't count) no older than ``max_age_hours`` and no more than
  ``FUTURE_SKEW`` ahead of ``now``.

"Control characters" here means the Unicode categories Cc (/C1 controls, newlines, tabs), Cf
(format characters: zero-width spaces and joiners, bidirectional overrides, the BOM), Cs (lone
surrogates) and Zl/Zp (line and paragraph separators). Cf is included because its characters are
invisible and can reorder or hide text; a product name never needs them.
"""

from __future__ import annotations

import itertools
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

GROUP = "WebActivity"
#: trait name -> Memory dataType (the group is created with exactly these)
TRAITS: Mapping[str, str] = {"lastViewedProduct": "STRING", "lastViewedAt": "STRING", "viewCount": "NUMBER"}
FRESHNESS_TRAIT = "lastViewedAt"  # ISO-8601 UTC string; Memory has no date type
DEFAULT_MAX_AGE_HOURS = 24
MAX_TRAITS_PER_GROUP = 20
MAX_VALUE_CHARS = 120
FUTURE_SKEW = timedelta(minutes=5)  # a lastViewedAt this far ahead of the bridge clock still counts
TRAIT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
MAX_NUMBER_ABS = 10**15  # larger ints and floats are dropped (a 4000-digit int is not a view count)

# Bridge settings: the defaults and limits of OWLBANK_MEMORY_TRAIT_GROUPS and
# OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS. infra/owlbank_infra/config.py keeps its own copy (a separate
# project); tests/unit/test_trait_settings_parity.py keeps the two equal.
DEFAULT_TRAIT_GROUPS: tuple[str, ...] = ("Contact", GROUP)
TRAIT_GROUP_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MAX_TRAIT_GROUPS = 10
MAX_AGE_HOURS_RANGE: tuple[int, int] = (1, 8760)

# A calendar date, then 'T' or a space, then hours: fromisoformat alone also takes a date only
# ("2026-10-28", midnight) and week dates ("2026-").
_DATE_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}", re.ASCII)

_REMOVED_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})


def _clean_text(text: str) -> str:
    """``text`` without control characters, stripped and cut to ``MAX_VALUE_CHARS``."""
    kept = "".join(c for c in text if unicodedata.category(c) not in _REMOVED_CATEGORIES)
    return kept.strip()[:MAX_VALUE_CHARS].rstrip()


def _clean_value(value: Any) -> str | int | float | bool | None:
    """The value as it may reach the prompt, or ``None`` when it is dropped."""
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value) if abs(value) <= MAX_NUMBER_ABS else None
    if isinstance(value, float):
        return float(value) if math.isfinite(value) and abs(value) <= MAX_NUMBER_ABS else None
    if isinstance(value, str):
        return _clean_text(value) or None
    return None


def _clean_group(group: Mapping[Any, Any], first: Sequence[str] = ()) -> dict[str, Any]:
    """The group's valid traits, at most ``MAX_TRAITS_PER_GROUP``: the names in ``first`` (those
    present) before the rest, which keep their input order."""
    out: dict[str, Any] = {}
    leading = [(name, group[name]) for name in first if name in group]
    rest = ((name, value) for name, value in group.items() if not (isinstance(name, str) and name in first))
    for name, value in itertools.chain(leading, rest):
        if len(out) >= MAX_TRAITS_PER_GROUP:
            break
        if not isinstance(name, str) or not TRAIT_NAME_RE.fullmatch(name):
            continue
        cleaned = _clean_value(value)
        if cleaned is not None:
            out[str(name)] = cleaned
    return out


def _parse_time(value: Any) -> datetime | None:
    """An aware datetime from an ISO-8601 date-and-time string (``Z`` accepted; no offset means UTC),
    or ``None``. A date without a time, or a week or ordinal date, is ``None``."""
    if not isinstance(value, str) or not _DATE_TIME_RE.match(value):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _is_fresh(group: Mapping[str, Any], now: datetime, max_age_hours: int) -> bool:
    """Whether a cleaned ``WebActivity`` group is inside the window. False on any doubt."""
    if now.tzinfo is None or isinstance(max_age_hours, bool) or not isinstance(max_age_hours, int):
        return False
    at = _parse_time(group.get(FRESHNESS_TRAIT))
    if at is None:
        return False
    try:
        return now - timedelta(hours=max_age_hours) <= at <= now + FUTURE_SKEW
    except (OverflowError, TypeError, ValueError):
        return False


def bound_profile_traits(traits: Mapping[str, Any] | None, *, allow: Sequence[str],
                         now: datetime, max_age_hours: int) -> dict[str, dict[str, Any]]:
    """A new dict: only allow-listed groups; per group at most MAX_TRAITS_PER_GROUP traits whose
    names match TRAIT_NAME_RE (in WebActivity the TRAITS names first, so lastViewedAt is never cut
    by the cap); values str (control characters removed, cut to MAX_VALUE_CHARS, dropped if
    empty), int or finite float with abs at most MAX_NUMBER_ABS, or bool; everything else (None,
    arrays, objects, larger numbers) dropped.
    The WebActivity group is dropped whole when lastViewedAt is missing, unparseable (including a
    date without a time, or a week date), in the
    future by more than FUTURE_SKEW (5 minutes, clock skew), or older than max_age_hours. Groups
    left empty are omitted. Never mutates its input and never raises on trait data: a group that
    can't be read is dropped, and input that isn't a mapping gives ``{}``.

    ``allow`` given as a single ``str`` raises ``TypeError``: that is a programming error in the
    caller (a str is a sequence of characters, which would silently allow nothing), not bad data."""
    if isinstance(allow, str):
        raise TypeError("allow must be a sequence of group names, not a str")
    out: dict[str, dict[str, Any]] = {}
    try:
        if not isinstance(traits, Mapping):
            return out
        allowed = {a for a in allow if isinstance(a, str)}
        for name, group in traits.items():
            try:
                if not isinstance(name, str) or name not in allowed or not isinstance(group, Mapping):
                    continue
                cleaned = _clean_group(group, tuple(TRAITS) if name == GROUP else ())
                if name == GROUP and not _is_fresh(cleaned, now, max_age_hours):
                    continue
                if cleaned:
                    out[str(name)] = cleaned
            except Exception:  # noqa: S112 - an unreadable group is dropped, never raised
                continue
    except Exception:  # fail closed: no traits rather than a failed recall
        return {}
    return out


__all__ = [
    "DEFAULT_MAX_AGE_HOURS",
    "DEFAULT_TRAIT_GROUPS",
    "FRESHNESS_TRAIT",
    "FUTURE_SKEW",
    "GROUP",
    "MAX_AGE_HOURS_RANGE",
    "MAX_NUMBER_ABS",
    "MAX_TRAITS_PER_GROUP",
    "MAX_TRAIT_GROUPS",
    "MAX_VALUE_CHARS",
    "TRAITS",
    "TRAIT_GROUP_NAME_RE",
    "TRAIT_NAME_RE",
    "bound_profile_traits",
]
