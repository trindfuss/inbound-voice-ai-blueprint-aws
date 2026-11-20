"""The shape of each deployed secret (portable secrets).

One definition of "well formed", shared by the runtime (:func:`owlbank.secrets.get_secret`
with ``logical=``), the scripts and ``check_secrets.py``. A :class:`Problem` names the secret,
the field and the kind of fault, **never the value**.

Logical names match ``infra/owlbank_infra/data_stack.SECRETS``. The placeholder CDK writes until
the operator sets a value is ``{"placeholder": true, "unset": "<random>"}``; any value carrying
the ``placeholder`` key is reported as the placeholder.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

__all__ = ["SCHEMAS", "OPTIONAL_PAIRS", "PLACEHOLDER_KEY", "Problem", "ProblemKind", "validate",
           "previous_pair"]

#: The key that marks CDK's placeholder value.
PLACEHOLDER_KEY = "placeholder"

_ACCOUNT_SID = re.compile(r"AC[0-9a-f]{32}")
_API_KEY = re.compile(r"SK[0-9a-f]{32}")
_API_SECRET = re.compile(r"[A-Za-z0-9]{32}")
_AUTH_TOKEN = re.compile(r"[0-9a-f]{32}")
_USER = re.compile(r"[A-Za-z0-9._-]{1,64}")
_PASSWORD = re.compile(r"[0-9a-f]{32,}")

#: logical secret name -> field -> pattern (matched in full, see :func:`validate`).
SCHEMAS: Mapping[str, Mapping[str, re.Pattern[str]]] = MappingProxyType({
    "twilio_api_key": MappingProxyType(
        {"account_sid": _ACCOUNT_SID, "api_key": _API_KEY, "api_secret": _API_SECRET}),
    "twilio_auth_token": MappingProxyType({"auth_token": _AUTH_TOKEN}),
    "basic_auth": MappingProxyType({"user": _USER, "password": _PASSWORD}),
})

#: logical secret name -> (password field, epoch-seconds field). Optional fields that must appear
#: together or not at all: the previous Basic password and the epoch second until which it is
#: still accepted. :func:`validate` reports a lone or malformed member as a problem (for
#: ``check_secrets``); readers use :func:`previous_pair`, which ignores a malformed pair.
OPTIONAL_PAIRS: Mapping[str, tuple[str, str]] = MappingProxyType({
    "basic_auth": ("previous_password", "previous_until"),
})

ProblemKind = Literal["placeholder", "not_object", "missing", "format", "pair"]


@dataclass(frozen=True)
class Problem:
    """One fault in a secret value. ``field`` is ``None`` only for ``not_object``."""

    secret: str
    field: str | None
    kind: ProblemKind

    def describe(self) -> str:
        """A one-line, value-free description, e.g. ``basic_auth: field password: format``."""
        where = f"field {self.field}" if self.field else "value"
        return f"{self.secret}: {where}: {self.kind}"


def validate(logical: str, value: object) -> list[Problem]:
    """Every problem with ``value`` as secret ``logical``; empty if it is well formed.

    Order: placeholder (alone: the other fields are then meaningless), else not an object,
    else each schema field in schema order (``missing`` if absent, ``format`` if not a string
    matching its pattern in full). Extra keys are allowed. ``KeyError`` for an unknown
    ``logical`` (a programming error, not a bad value).
    """
    schema = SCHEMAS[logical]
    if not isinstance(value, Mapping):
        return [Problem(logical, None, "not_object")]
    if PLACEHOLDER_KEY in value:
        return [Problem(logical, PLACEHOLDER_KEY, "placeholder")]
    problems: list[Problem] = []
    for field, pattern in schema.items():
        if field not in value:
            problems.append(Problem(logical, field, "missing"))
            continue
        item = value[field]
        if not isinstance(item, str) or pattern.fullmatch(item) is None:
            problems.append(Problem(logical, field, "format"))
    problems.extend(_pair_problems(logical, value))
    return problems


#: Latest plausible ``previous_until`` (epoch seconds, 2100-01-01 UTC): a larger value is a
#: millisecond epoch or garbage, and would keep a previous password valid for ever.
MAX_EPOCH_S = 4102444800


def _is_epoch(item: object) -> bool:
    return isinstance(item, int) and not isinstance(item, bool) and 0 < item <= MAX_EPOCH_S


def _pair_problems(logical: str, value: Mapping[str, object]) -> list[Problem]:
    pair = OPTIONAL_PAIRS.get(logical)
    if pair is None:
        return []
    pw_field, until_field = pair
    has_pw, has_until = pw_field in value, until_field in value
    if has_pw != has_until:
        return [Problem(logical, until_field if has_pw else pw_field, "pair")]
    if not has_pw:
        return []
    out: list[Problem] = []
    pw = value[pw_field]
    if not isinstance(pw, str) or _PASSWORD.fullmatch(pw) is None:
        out.append(Problem(logical, pw_field, "format"))
    if not _is_epoch(value[until_field]):
        out.append(Problem(logical, until_field, "format"))
    return out


def previous_pair(logical: str, value: Mapping[str, object]) -> tuple[str, int] | None:
    """The well-formed optional pair ``(password, epoch_until)`` of ``value``, else ``None``.

    A missing, lone or malformed pair is ignored (``None``): readers fail closed to "no previous
    password" rather than refusing the secret. ``KeyError`` for a ``logical`` with no pair.
    """
    pw_field, until_field = OPTIONAL_PAIRS[logical]
    if pw_field not in value or until_field not in value:
        return None
    if _pair_problems(logical, value):
        return None
    pw, until = value[pw_field], value[until_field]
    return (pw, until) if isinstance(pw, str) and isinstance(until, int) else None
