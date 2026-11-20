"""Call-bound auth tokens.

* :func:`issue` returns an opaque random token. Only ``sha256(token)`` (hex) is
  stored, in the ``tokens`` table: ``token_hash`` (PK), ``call_sid`` (GSI),
  ``customer_id``, ``level``, ``issued_at``, ``expires_at`` (DynamoDB TTL),
  ``revoked``.
* :func:`validate` returns :class:`TokenClaims` or :class:`AuthError`. Tools take
  the customer ID from the claims only.
* :func:`revoke_call` revokes every token of a call (call-end status callback).
* :func:`upgrade` (step-up) issues a token at a higher level and revokes the old one.

Error precedence, when several apply (checked in this order):
``MISSING`` → ``UNKNOWN`` → ``WRONG_CALL`` → ``REVOKED`` → ``EXPIRED`` →
``AUTH_LEVEL_INSUFFICIENT``. ``WRONG_CALL`` comes early so a token presented on
another call reveals nothing else about itself.

Times are integer epoch seconds passed in as ``now`` (no clock reads here). A
token is valid while ``now < expires_at``. ``MemoryStore`` never expires items;
``DynamoStore`` TTL deletion is lazy, so expiry is always checked here, and a
TTL-deleted token reads as ``UNKNOWN``.

Tokens and hashes are never logged.
"""

from __future__ import annotations

import enum
import hashlib
import logging
import secrets
from dataclasses import dataclass

from owlbank.obs import emit
from owlbank.store import ConditionFailed, Store

log = logging.getLogger(__name__)

__all__ = [
    "AuthError",
    "AuthErrorCode",
    "TokenClaims",
    "TABLE",
    "CALL_INDEX",
    "hash_token",
    "issue",
    "validate",
    "revoke_call",
    "upgrade",
]

TABLE = "tokens"
CALL_INDEX = "call_sid-index"
_TOKEN_BYTES = 32
_ISSUE_ATTEMPTS = 3


class AuthErrorCode(enum.StrEnum):
    MISSING = "MISSING"
    UNKNOWN = "UNKNOWN"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    WRONG_CALL = "WRONG_CALL"
    AUTH_LEVEL_INSUFFICIENT = "AUTH_LEVEL_INSUFFICIENT"


@dataclass(frozen=True, slots=True)
class AuthError:
    code: AuthErrorCode


@dataclass(frozen=True, slots=True)
class TokenClaims:
    call_sid: str
    customer_id: str
    level: int
    issued_at: int
    expires_at: int


def hash_token(token: str) -> str:
    """Deterministic storage key for a token: lower-case hex SHA-256 of its UTF-8 bytes."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _check_int(name: str, value: object, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
    return value


def issue(
    store: Store, *, call_sid: str, customer_id: str, level: int, now: int, max_ttl_s: int
) -> str:
    """Issue a token for this call and customer at ``level``; expires at ``now + max_ttl_s``.

    Returns the opaque token (``secrets.token_urlsafe``); only its hash is stored.
    Raises ``ValueError`` for empty IDs, a negative level/time, or a non-positive TTL.
    """
    if not call_sid or not isinstance(call_sid, str):
        raise ValueError("call_sid is required")
    if not customer_id or not isinstance(customer_id, str):
        raise ValueError("customer_id is required")
    _check_int("level", level, minimum=0)
    _check_int("now", now, minimum=0)
    _check_int("max_ttl_s", max_ttl_s, minimum=1)
    for _ in range(_ISSUE_ATTEMPTS):
        token = secrets.token_urlsafe(_TOKEN_BYTES)
        item = {
            "token_hash": hash_token(token),
            "call_sid": call_sid,
            "customer_id": customer_id,
            "level": level,
            "issued_at": now,
            "expires_at": now + max_ttl_s,
            "revoked": False,
        }
        try:
            store.put(TABLE, item, condition="not_exists")
        except ConditionFailed:  # a 256-bit collision: practically impossible, retry anyway
            emit(log, logging.WARNING, "token_hash_collision", operation="issue", decision="retry")
            continue
        return token
    raise RuntimeError("could not issue a unique token")


def validate(
    store: Store, token: str | None, *, call_sid: str, min_level: int, now: int
) -> TokenClaims | AuthError:
    """Check ``token`` for a tool on ``call_sid`` that needs ``min_level``. See module docs."""
    if not isinstance(token, str) or not token:
        return AuthError(AuthErrorCode.MISSING)
    item = store.get(TABLE, {"token_hash": hash_token(token)})
    if item is None:
        return AuthError(AuthErrorCode.UNKNOWN)
    if not call_sid or item.get("call_sid") != call_sid:
        return AuthError(AuthErrorCode.WRONG_CALL)
    if item.get("revoked") is not False:  # anything but an explicit False is revoked
        return AuthError(AuthErrorCode.REVOKED)
    expires_at = item.get("expires_at")
    if not isinstance(expires_at, int) or now >= expires_at:
        return AuthError(AuthErrorCode.EXPIRED)
    level = item.get("level")
    if not isinstance(level, int) or isinstance(level, bool) or level < min_level:
        return AuthError(AuthErrorCode.AUTH_LEVEL_INSUFFICIENT)
    return TokenClaims(
        call_sid=item["call_sid"],
        customer_id=item["customer_id"],
        level=level,
        issued_at=item["issued_at"],
        expires_at=expires_at,
    )


def revoke_call(store: Store, call_sid: str) -> None:
    """Revoke every token issued on ``call_sid`` (idempotent).

    Uses the ``call_sid-index`` GSI. On DynamoDB a GSI query is eventually
    consistent, so a token issued milliseconds earlier may be missed; call end
    is backstopped by ``expires_at``, and :func:`upgrade` revokes by hash.
    """
    if not call_sid:
        return
    for item in store.query(TABLE, {"call_sid": call_sid}, index=CALL_INDEX):
        if item.get("revoked") is True:
            continue
        try:
            store.update(TABLE, {"token_hash": item["token_hash"]}, {"revoked": True})
        except ConditionFailed:  # deleted (TTL) between query and update
            emit(log, logging.DEBUG, "token_revoke_skipped", operation="revoke_call",
                 decision="ignored_because_token_already_gone")


def upgrade(
    store: Store,
    token: str | None,
    *,
    call_sid: str,
    new_level: int,
    now: int,
    max_ttl_s: int,
) -> str | AuthError:
    """Step-up: replace a valid token on ``call_sid`` with one at ``new_level``.

    The old token must validate for this call (any level); its error is returned
    otherwise. The new token keeps the customer ID, is issued first, and then the
    old one is revoked (by hash, strongly consistent). Levels never decrease:
    ``new_level`` below the current level raises ``ValueError``.

    Call this only after the auth service has verified the factors for
    ``new_level``; it does no verification itself.
    """
    claims = validate(store, token, call_sid=call_sid, min_level=0, now=now)
    if isinstance(claims, AuthError):
        return claims
    _check_int("new_level", new_level, minimum=0)
    if new_level < claims.level:
        raise ValueError("auth level can't decrease")
    new_token = issue(
        store,
        call_sid=call_sid,
        customer_id=claims.customer_id,
        level=new_level,
        now=now,
        max_ttl_s=max_ttl_s,
    )
    if token is None:  # security invariant: validate() passed, so the old token exists (explicit raise survives `python -O`)
        raise RuntimeError("validated token is None")
    try:
        store.update(TABLE, {"token_hash": hash_token(token)}, {"revoked": True})
    except ConditionFailed:
        emit(log, logging.DEBUG, "token_revoke_skipped", operation="upgrade",
             decision="ignored_because_token_already_gone")
    return new_token
