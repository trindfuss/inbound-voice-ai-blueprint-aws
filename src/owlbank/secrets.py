"""Cached Secrets Manager reads (secrets storage).

Secrets are JSON objects in ``SecretString`` (for example
``{"auth_token": "..."}``). Values are cached per ARN for ``ttl_s`` seconds,
so a warm Lambda doesn't call Secrets Manager on every request, and rotation is
picked up within ``ttl_s``.

With ``logical=`` (a name in :data:`owlbank.secret_schema.SCHEMAS`) the value is also checked
against that secret's schema (portable secrets): the placeholder or a malformed value raises
:class:`SecretError` naming the secret and the first problem's field and kind, and logs
``secret_invalid`` with the same, so the caller fails closed. A value that fails the check is not
cached, so a corrected value is picked up on the next read.

**Secret values are never logged** and never appear in exception messages:
logs carry the ARN and cache hit/miss only. botocore's own DEBUG logging
prints raw response bodies (secret values included), so never run deployed
code with the ``botocore`` logger at DEBUG.

Note: this module's name shadows the stdlib ``secrets`` only for code that
does ``import secrets`` from inside ``src/owlbank`` with that directory on
``sys.path``; package imports (``owlbank.secrets``) are unaffected.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from owlbank.obs import emit, log_caught, timed_call
from owlbank.secret_schema import OPTIONAL_PAIRS, Problem, validate

__all__ = ["get_secret", "clear_cache", "SecretError"]

log = logging.getLogger(__name__)

_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_lock = threading.Lock()
_default_client: Any | None = None

#: Cache lifetime (seconds) when the caller passes no ``ttl_s``, per logical secret. The Basic secret
#: is short so a rotation reaches every warm instance within a minute.
DEFAULT_TTL_S = 300.0
TTL_BY_LOGICAL: dict[str, float] = {"basic_auth": 60.0}

#: A forced re-read of one ARN is allowed at most once per this many seconds per instance.
FORCE_MIN_INTERVAL_S = 10.0
_last_forced: dict[str, float] = {}

#: Clock used for cache expiry; tests may replace it.
clock: Callable[[], float] = time.monotonic


class SecretError(Exception):
    """The secret could not be read, is not a JSON object, or (with ``logical``) doesn't match its
    schema. Never carries the value. ``problem`` is the first schema problem, when that was the cause."""

    def __init__(self, message: str, *, problem: Problem | None = None) -> None:
        super().__init__(message)
        self.problem = problem


def clear_cache() -> None:
    """Drop every cached secret (tests, or forced refresh after rotation)."""
    with _lock:
        _cache.clear()
        _last_forced.clear()


def _client() -> Any:
    global _default_client
    if _default_client is None:
        import boto3

        _default_client = boto3.client("secretsmanager")
    return _default_client


def _check(arn: str, logical: str | None, value: dict[str, Any]) -> None:
    """Raise :class:`SecretError` if ``value`` doesn't match ``logical``'s schema (no value logged)."""
    if logical is None:
        return
    optional = OPTIONAL_PAIRS.get(logical, ())
    # a malformed optional pair is ignored by readers (check_secrets reports it)
    problems = [p for p in validate(logical, value) if p.field not in optional]
    if not problems:
        return
    first = problems[0]
    emit(log, logging.WARNING, "secret_invalid", secret=logical, secret_arn=arn,
         field=first.field, problem=first.kind, problems=len(problems), decision="raise_secret_error")
    raise SecretError(f"secret {logical} ({arn}) is not usable: {first.describe()}", problem=first)


def _parse(arn: str, logical: str | None, resp: Any) -> dict[str, Any]:
    """The JSON object in ``resp`` checked against ``logical``; :class:`SecretError` (no value)."""
    raw = resp.get("SecretString")
    if not isinstance(raw, str):
        raise SecretError(f"secret {arn} has no SecretString")
    try:
        value = json.loads(raw)
    except ValueError:
        value = None  # reported just below as SecretError (not a JSON object)
    if not isinstance(value, dict):
        # `from None`: a chained JSONDecodeError would carry the raw value in .doc
        raise SecretError(f"secret {arn} is not a JSON object") from None
    _check(arn, logical, value)
    return value


def get_secret(arn: str, *, client: Any | None = None, ttl_s: float | None = None,
               logical: str | None = None, force: bool = False) -> dict[str, Any]:
    """Return the secret at ``arn`` parsed as a JSON object (a fresh copy).

    ``client`` is a boto3 Secrets Manager client (default: one created lazily).
    ``logical`` names the secret's schema (:data:`owlbank.secret_schema.SCHEMAS`); when given,
    the value is validated and a placeholder or malformed value raises.
    ``ttl_s`` defaults per secret (:data:`TTL_BY_LOGICAL`: 60 s for ``basic_auth``, else 300 s).
    ``force=True`` skips the cache and re-reads, at most once per ARN per
    :data:`FORCE_MIN_INTERVAL_S` per instance (a rate-limited or failed forced read returns the
    cached value, best effort, and so is a forced read that returns unparsable or schema-invalid
    content; with nothing cached a failed forced read raises as a normal read).
    Raises :class:`SecretError` if the read fails, the value isn't a JSON object, or (with
    ``logical``) it doesn't match the schema. The message never carries the value.
    """
    if not arn:
        raise SecretError(f"secret {logical} ARN is not configured" if logical
                          else "secret ARN is not configured")
    if ttl_s is None:
        ttl_s = TTL_BY_LOGICAL.get(logical, DEFAULT_TTL_S) if logical else DEFAULT_TTL_S
    now = clock()
    with _lock:
        hit = _cache.get(arn)
        cached = hit[1] if hit is not None and now - hit[0] < ttl_s else None
        stale = hit[1] if hit is not None else None
        if force:
            last = _last_forced.get(arn)
            if last is not None and now - last < FORCE_MIN_INTERVAL_S:
                force = False  # rate limited: serve the cached value as usual
            else:
                _last_forced[arn] = now
                cached = None
    if cached is not None:
        log.debug("secret cache hit", extra={"secret_arn": arn})
        _check(arn, logical, cached)
        return copy.deepcopy(cached)
    log.debug("secret cache miss", extra={"secret_arn": arn})
    try:
        with timed_call("secretsmanager", "get_secret_value"):
            resp = (client or _client()).get_secret_value(SecretId=arn)
    except Exception as exc:
        if force and stale is not None:
            log_caught(log, "forced secret read failed, using cached value", exc, secret_arn=arn,
                       decision="use_cached")
            _check(arn, logical, stale)
            return copy.deepcopy(stale)
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", type(exc).__name__)
        log_caught(log, "secret read failed", exc, secret_arn=arn, decision="raise_secret_error")
        raise SecretError(f"could not read secret {arn}: {code}") from None
    try:
        value = _parse(arn, logical, resp)
    except SecretError as exc:
        if force and stale is not None:
            log_caught(log, "forced secret read unusable, using cached value", exc, secret_arn=arn,
                       decision="use_cached")
            _check(arn, logical, stale)
            return copy.deepcopy(stale)
        raise
    with _lock:
        _cache[arn] = (now, value)
    return copy.deepcopy(value)
