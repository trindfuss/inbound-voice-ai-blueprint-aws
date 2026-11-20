"""Lambda entry point for ``runtime_warmer``.

Invoked by the front door only, with ``InvocationType=Event`` on a new call's ``incoming``
(IAM; the function has no API, URL or other trigger, and its async retries are 0). The payload is
``{"call_sid": "CA…"}``. The handler starts the call's AgentCore Runtime session early, so the
microVM boot hides behind the language and ZIP prompts:

``InvokeAgentRuntime(agentRuntimeArn, runtimeSessionId=<CallSid>, payload={"action": "prewarm"})``

with a 2 s connect timeout, a 25 s read timeout and one attempt. The bridge's ``prewarm`` action
does nothing: AgentCore only routes it once ``/ping`` is healthy, which is when the bridge has
finished its start-up work.

One log line per invocation, never carrying the ARN, a URL, a header value or a body:

* ``runtime_prewarm`` (``duration_ms``, ``outcome`` ``ok`` or ``in_progress``). ``in_progress`` is a
  ``RetryableConflictException``: the session is already being provisioned, which is not a failure.
* ``runtime_prewarm_failed`` (``duration_ms``, ``outcome="failed"``, ``error_class``, ``code``) for
  any other error. The call continues and pays the cold start at connect. A metric filter on
  this event with ``code`` ``ThrottlingException``/``ServiceQuotaExceededException`` is alarmed on.
* ``runtime_prewarm_rejected`` (``reason``) when the event carries no valid CallSid: no Runtime call.

The handler never raises (the front door does not wait for it, and a retry would add nothing).

Cold start: importing this module creates no client; the boto3 client is built lazily, on the
first invoke. Environment: ``OWLBANK_RELAY_PARAMETER`` (the Bridge stack's SSM parameter holding the
Runtime's ARN, read once at INIT) or ``OWLBANK_RUNTIME_ARN`` (the ARN itself, set
directly, as the tests do).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from owlbank import coldstart, obs
from owlbank.relay_param import RUNTIME_ARN_ENV, resolve_relay_env
from owlbank.sids import CALL_SID_RE  # one pattern for the signer and the warmer

log = logging.getLogger(__name__)

#: SSM ``GetParameter`` for the relay parameter: once, at INIT.
SSM_CONNECT_TIMEOUT_S = 2
SSM_READ_TIMEOUT_S = 3
#: A Twilio CallSid; also the Runtime session ID, which must be at least 33 characters.

PREWARM_PAYLOAD = json.dumps({"action": "prewarm"}, separators=(",", ":")).encode()
CONNECT_TIMEOUT_S = 2
READ_TIMEOUT_S = 25
#: The session is already being provisioned (a concurrent start): not a failure.
IN_PROGRESS_CODE = "RetryableConflictException"

Outcome = Literal["ok", "in_progress", "failed"]


@dataclass(frozen=True, slots=True)
class PrewarmResult:
    """What one pre-warm attempt did (``code`` only on failure; never a message)."""

    outcome: Outcome
    duration_ms: float
    error_class: str | None = None
    code: str | None = None


@dataclass(frozen=True, slots=True)
class WarmerDeps:
    client: Any
    runtime_arn: str | None
    clock: Callable[[], float] = time.monotonic


def _ms(started: float, clock: Callable[[], float]) -> float:
    return round((clock() - started) * 1000, 1)


def prewarm(client: Any, runtime_arn: str, call_sid: str, *, clock: Callable[[], float]) -> PrewarmResult:
    """Invoke the Runtime's ``prewarm`` action on ``call_sid``'s session and log the outcome.

    Never raises: every exception becomes ``outcome="failed"`` with its class and AWS error code.
    """
    started = clock()
    try:
        response = client.invoke_agent_runtime(
            agentRuntimeArn=runtime_arn,
            runtimeSessionId=call_sid,
            contentType="application/json",
            payload=PREWARM_PAYLOAD,
        )
        body = response.get("response") if isinstance(response, Mapping) else None
        if body is not None and hasattr(body, "read"):
            try:
                body.read()  # drain the stream: the call is done when the bridge has answered
            finally:
                close = getattr(body, "close", None)
                if callable(close):
                    close()
    except Exception as exc:  # noqa: BLE001 - logged below; a pre-warm failure must never raise
        duration = _ms(started, clock)
        code = obs.error_code_of(exc)
        if code == IN_PROGRESS_CODE:
            result = PrewarmResult("in_progress", duration)
            obs.emit(log, logging.INFO, "runtime_prewarm", duration_ms=duration, outcome=result.outcome)
            return result
        result = PrewarmResult("failed", duration, error_class=type(exc).__name__, code=code)
        obs.emit(log, logging.WARNING, "runtime_prewarm_failed", duration_ms=duration, outcome=result.outcome,
                 error_class=result.error_class, code=code, decision="continue_cold")
        return result
    result = PrewarmResult("ok", _ms(started, clock))
    obs.emit(log, logging.INFO, "runtime_prewarm", duration_ms=result.duration_ms, outcome=result.outcome)
    return result


def _reject(reason: str) -> dict[str, Any]:
    obs.emit(log, logging.WARNING, "runtime_prewarm_rejected", reason=reason)
    return {"result": "rejected", "reason": reason}


def handle_event(deps: WarmerDeps, event: Any) -> dict[str, Any]:
    """Validate the event, pre-warm the session, and return ``{"result": <outcome>}``. Never raises."""
    call_sid = event.get("call_sid") if isinstance(event, Mapping) else None
    if not isinstance(call_sid, str) or not CALL_SID_RE.fullmatch(call_sid):
        return _reject("bad_call_sid")
    if not deps.runtime_arn:
        result = PrewarmResult("failed", 0.0, error_class="MissingConfiguration", code=None)
        obs.emit(log, logging.WARNING, "runtime_prewarm_failed", duration_ms=result.duration_ms,
                 outcome=result.outcome, error_class=result.error_class, code=None, decision="continue_cold")
    else:
        result = prewarm(deps.client, deps.runtime_arn, call_sid, clock=deps.clock)
    if result.outcome == "failed":
        obs.degraded("prewarm_failed")
    return {"result": result.outcome}


def _make_client() -> Any:
    import boto3
    from botocore.config import Config

    # ``total_max_attempts`` counts the first try; botocore's ``max_attempts`` counts retries only
    # (``{"max_attempts": 1}`` would be two attempts).
    config = Config(connect_timeout=CONNECT_TIMEOUT_S, read_timeout=READ_TIMEOUT_S,
                    retries={"total_max_attempts": 1})
    return boto3.client("bedrock-agentcore", config=config)


def _make_ssm_client() -> Any:
    import boto3
    from botocore.config import Config

    config = Config(connect_timeout=SSM_CONNECT_TIMEOUT_S, read_timeout=SSM_READ_TIMEOUT_S,
                    retries={"mode": "standard", "total_max_attempts": 2})
    return boto3.client("ssm", config=config)


def build_deps(env: Mapping[str, str] | None = None, *,
               ssm_factory: Callable[[], Any] = _make_ssm_client) -> WarmerDeps:
    """Production dependencies. The Runtime client is created on first use; with
    ``OWLBANK_RELAY_PARAMETER`` the Runtime ARN is read from that parameter now, and an unusable
    parameter raises :class:`owlbank.relay_param.RelayParameterError` (fail closed: the handler logs
    each invoke as failed and tries again on the next)."""
    env = resolve_relay_env(os.environ if env is None else env, hosting="agentcore", ssm_factory=ssm_factory)
    return WarmerDeps(client=coldstart.LazyClient(_make_client), runtime_arn=env.get(RUNTIME_ARN_ENV) or None)


_deps: WarmerDeps | None = None


@obs.instrumented("runtime_warmer", op="prewarm")
def handler(event: Any, context: Any = None) -> dict[str, Any]:
    global _deps
    try:
        if _deps is None:
            _deps = build_deps()
        return handle_event(_deps, event)
    except Exception as exc:  # noqa: BLE001 - never raises; handle_event already catches the invoke
        obs.log_caught(log, "runtime_prewarm_failed", exc, decision="continue_cold",
                       code=obs.error_code_of(exc), outcome="failed")
        return {"result": "failed"}


# Inside Lambda only (a no-op in tests and scripts): wire up now, in the INIT phase.
_deps = coldstart.build_at_init(build_deps)
