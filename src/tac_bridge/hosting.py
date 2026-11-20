"""Where the bridge runs: ECS Fargate (default) or AgentCore Runtime.

* :func:`hosting_from_env`: ``OWLBANK_HOSTING`` absent or empty -> ``"fargate"``, ``agentcore``
  -> ``"agentcore"``, anything else -> ``RuntimeError`` naming the variable (never its value).
* :class:`SocketTracker`: the ``/ping`` state. ``HealthyBusy`` while at least one
  ConversationRelay socket is open, otherwise ``Healthy``; ``time_of_last_update`` (epoch
  seconds) changes only when the status changes, as the Runtime contract asks.
* :func:`load_twilio_secrets`: on AgentCore the container gets the two Twilio secrets' ARNs
  (``OWLBANK_SECRET_TWILIO_API_KEY``, ``OWLBANK_SECRET_TWILIO_AUTH_TOKEN``) instead of ECS-injected
  values. They are read from Secrets Manager at start-up and mapped to the names TAC reads,
  the same mapping as the Fargate task definition (``bridge_stack.TASK_SECRETS``). Values never
  reach a log or an exception message; errors name the variable (and the JSON field) only.
* :class:`SessionPin`: the process belongs to the first Runtime session ID it sees (the
  CallSid); every other ID is refused. In ``agentcore`` the relay nonce has one carrier, the
  ConversationRelay ``<Parameter name="n">`` in ``setup.customParameters`` (AgentCore
  forwards no query parameters to the container); Fargate keeps the signed query's ``?n=``.

Importing this module makes no network call.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Literal

Hosting = Literal["fargate", "agentcore"]
HOSTING_ENV = "OWLBANK_HOSTING"
HOSTINGS: tuple[Hosting, ...] = ("fargate", "agentcore")

PING_PATH = "/ping"
INVOCATIONS_PATH = "/invocations"
HEALTHY = "Healthy"
HEALTHY_BUSY = "HealthyBusy"

SECRET_API_KEY_ENV = "OWLBANK_SECRET_TWILIO_API_KEY"  # noqa: S105 (an env var name, not a secret)
SECRET_AUTH_TOKEN_ENV = "OWLBANK_SECRET_TWILIO_AUTH_TOKEN"  # noqa: S105 (an env var name, not a secret)
#: TAC's variable -> (the env var holding the secret's ARN, the JSON field). Same as Fargate's
#: ``TASK_SECRETS`` (``infra/owlbank_infra/bridge_stack.py``).
TWILIO_SECRET_FIELDS: dict[str, tuple[str, str]] = {
    "TWILIO_ACCOUNT_SID": (SECRET_API_KEY_ENV, "account_sid"),
    "TWILIO_API_KEY": (SECRET_API_KEY_ENV, "api_key"),
    "TWILIO_API_SECRET": (SECRET_API_KEY_ENV, "api_secret"),
    "TWILIO_AUTH_TOKEN": (SECRET_AUTH_TOKEN_ENV, "auth_token"),
}
#: The schema (``owlbank.secret_schema.SCHEMAS``) each ARN variable's secret is checked against.
SECRET_LOGICAL: dict[str, str] = {SECRET_API_KEY_ENV: "twilio_api_key", SECRET_AUTH_TOKEN_ENV: "twilio_auth_token"}


def hosting_from_env(env: Mapping[str, str]) -> Hosting:
    """The hosting named by ``OWLBANK_HOSTING`` (absent or empty: ``fargate``)."""
    raw = (env.get(HOSTING_ENV) or "").strip()
    if not raw:
        return "fargate"
    if raw not in HOSTINGS:
        raise RuntimeError(f"{HOSTING_ENV} must be one of: {', '.join(HOSTINGS)}")
    return raw


class SocketTracker:
    """Open ConversationRelay sockets in this process, for ``GET /ping``.

    Used from the event loop only. ``clock`` returns epoch seconds (default ``time.time``).
    A ``closed()`` without a matching ``opened()`` is ignored (the count never goes negative).
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._open = 0
        self._updated = int(clock())

    @property
    def open_count(self) -> int:
        return self._open

    @property
    def status(self) -> str:
        return HEALTHY_BUSY if self._open > 0 else HEALTHY

    def _set(self, count: int) -> None:
        before = self.status
        self._open = count
        if self.status != before:
            self._updated = int(self._clock())

    def opened(self) -> None:
        self._set(self._open + 1)

    def closed(self) -> None:
        self._set(max(0, self._open - 1))

    def ping(self) -> dict[str, Any]:
        return {"status": self.status, "time_of_last_update": self._updated}


def _secret_fault(exc: Any) -> str:
    """What is wrong with a secret, from a ``SecretError``: names the field, never the value."""
    problem = getattr(exc, "problem", None)
    if problem is None:
        return "the secret could not be read or is not a JSON object"
    if problem.kind == "placeholder":
        return "the secret is still the placeholder"
    if problem.kind == "missing":
        return f"the secret has no {problem.field}"
    return f"the secret's {problem.field} is malformed"


def load_twilio_secrets(env: Mapping[str, str], sm_client: Any) -> dict[str, str]:
    """Read the Twilio secrets by ARN and return ``{TAC variable: value}`` for the four names in
    :data:`TWILIO_SECRET_FIELDS`. ``RuntimeError`` (naming the variable and field only) when an ARN
    variable is missing, the secret can't be read or isn't a JSON object, is still the placeholder,
    or a field is missing or malformed (``owlbank.secret_schema``). Each secret is read once."""
    from owlbank.secrets import SecretError, get_secret  # noqa: PLC0415

    arns: dict[str, str] = {}
    for var in dict.fromkeys(v for v, _ in TWILIO_SECRET_FIELDS.values()):  # all present before any read
        arns[var] = (env.get(var) or "").strip()
        if not arns[var]:
            raise RuntimeError(f"{var} is required")
    docs: dict[str, dict[str, Any]] = {}
    for var, arn in arns.items():
        try:
            docs[var] = get_secret(arn, client=sm_client, logical=SECRET_LOGICAL[var])
        except SecretError as exc:
            raise RuntimeError(f"{var}: {_secret_fault(exc)}") from None
    out: dict[str, str] = {}
    for name, (var, field) in TWILIO_SECRET_FIELDS.items():
        value = docs[var].get(field)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"{var}: the secret has no {field}")
        out[name] = value
    return out


# --------------------------------------------------------------------------- call binding

#: The relay nonce's name: Fargate's signed query parameter, ``agentcore``'s ``<Parameter>``.
NONCE_NAME = "n"
#: The header AgentCore puts on every request to the container: the Runtime session ID (= CallSid).
SESSION_ID_HEADER = "x-amzn-bedrock-agentcore-runtime-session-id"


class SessionPin:
    """One Runtime session per process. The first non-empty session ID pins the process;
    afterwards :meth:`check` is ``True`` only for that ID. An absent, empty or non-string ID is
    ``False`` and pins nothing. Used from the event loop only; the ID is never logged."""

    def __init__(self) -> None:
        self._pinned: str | None = None

    @property
    def pinned(self) -> bool:
        return self._pinned is not None

    def check(self, session_id: str | None) -> bool:
        if not isinstance(session_id, str) or not session_id:
            return False
        if self._pinned is None:
            self._pinned = session_id
        return session_id == self._pinned
