"""HTTP Basic credential check for Twilio-facing endpoints.

The webhook URLs configured in Twilio carry ``user:password@``; Twilio sends
them as ``Authorization: Basic base64(user:password)``. Credentials come from
the Basic-auth secret (``owlbank.secrets.get_secret``), never from code.
"""

from __future__ import annotations

import base64
import binascii
import enum
import hmac
import logging
from dataclasses import dataclass

from owlbank.obs import emit

log = logging.getLogger(__name__)

__all__ = ["Match", "BasicResult", "BasicSecret", "match", "check"]


class Match(enum.Enum):
    """Which configured password the request carried."""

    NONE = "none"
    CURRENT = "current"
    PREVIOUS = "previous"


@dataclass(frozen=True)
class BasicResult:
    """``password`` is the password that matched (``None`` for :attr:`Match.NONE`)."""

    match: Match
    password: str | None = None


@dataclass(frozen=True, repr=False)
class BasicSecret:
    """The Basic-auth secret as the request path needs it. ``repr`` carries no password."""

    user: str
    password: str
    previous_password: str | None = None
    previous_until: float | None = None

    def __repr__(self) -> str:
        return "BasicSecret(<redacted>)"


_NO_MATCH = BasicResult(Match.NONE, None)
_DUMMY = b"\x00" * 32


def _decode(authorization_header: str | None) -> tuple[str, str] | None:
    if not isinstance(authorization_header, str):
        return None
    scheme, _, encoded = authorization_header.strip().partition(" ")
    if scheme.lower() != "basic":
        return None
    encoded = encoded.strip()
    if not encoded:
        return None
    try:
        decoded = base64.b64decode(encoded.encode("ascii"), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeError, ValueError) as exc:
        emit(log, logging.DEBUG, "basic_auth_malformed", operation="check",
             error_class=type(exc).__name__, decision="reject")
        return None
    user, sep, password = decoded.partition(":")
    if not sep:
        return None
    return user, password


def match(authorization_header: str | None, expected_user: str, current: str,
          previous: str | None = None, previous_until: float | None = None, *,
          now: float) -> BasicResult:
    """Which password, if any, the header carries.

    ``now`` is the wall clock (epoch seconds, ``time.time()``), because ``previous_until`` is epoch
    seconds; ``owlbank.secrets.clock`` is monotonic and must not be used. ``PREVIOUS`` only while
    ``now < previous_until``; a ``previous`` without an expiry (or an empty one) is never accepted.
    The user, the current and the previous password are always all compared in constant time (a
    fixed dummy when there is no previous), so timing doesn't show which matched. Empty expected
    user or current password never match. Never raises.
    """
    if not expected_user or not current:
        return _NO_MATCH
    creds = _decode(authorization_header)
    if creds is None:
        return _NO_MATCH
    user, password = creds
    pw = password.encode("utf-8")
    prev_active = bool(previous) and previous_until is not None and now < previous_until
    prev_bytes = previous.encode("utf-8") if previous else _DUMMY
    user_ok = hmac.compare_digest(user.encode("utf-8"), expected_user.encode("utf-8"))
    cur_ok = hmac.compare_digest(pw, current.encode("utf-8"))
    prev_ok = hmac.compare_digest(pw, prev_bytes)
    if user_ok and cur_ok:
        return BasicResult(Match.CURRENT, current)
    if user_ok and prev_ok and prev_active:
        return BasicResult(Match.PREVIOUS, previous)
    return _NO_MATCH


def check(authorization_header: str | None, expected_user: str, expected_password: str) -> bool:
    """True only if the header carries exactly the expected Basic credentials (no previous password).

    * Scheme must be ``Basic`` (case-insensitive); any other scheme fails.
    * Missing, empty, non-base64, non-UTF-8 or colon-less credentials fail.
    * Empty expected credentials always fail (fail closed on misconfiguration).
    * User and password are both compared in constant time, and both always
      compared, so timing doesn't reveal which one was wrong.
    Never raises.
    """
    if not expected_user or not expected_password:
        return False
    creds = _decode(authorization_header)
    if creds is None:
        return False
    user, password = creds
    user_ok = hmac.compare_digest(user.encode("utf-8"), expected_user.encode("utf-8"))
    pass_ok = hmac.compare_digest(password.encode("utf-8"), expected_password.encode("utf-8"))
    return user_ok & pass_ok
