"""``agentcore`` call end: stop the call's AgentCore Runtime session.

``/voice/status`` calls :meth:`RuntimeSessions.stop` once per call that reaches an end status, so
the microVM (and its billing) ends with the call instead of after the idle timeout:

``StopRuntimeSession(agentRuntimeArn, runtimeSessionId=<CallSid>)``

with a 2 s connect and read timeout and one attempt (the client is built that way by
``handler.build_app``; it is not the shared ``quick`` config). One log line per attempt,
``runtime_session_stopped`` with ``outcome``:

* ``ok``: the session was stopped;
* ``not_found``: ``ResourceNotFoundException`` (the session had already ended: the idle timeout,
  the max lifetime, or it was never started);
* ``error:<class>``: anything else (the AWS error code goes in ``error_code``); the idle timeout
  stops the session instead.

:meth:`RuntimeSessions.stop` never raises, so a stop never fails the status response. The line
never carries the ARN, a URL, a header value or an error message.

**Exactly one attempt per call.** Twilio may deliver a status callback more than once, and
callbacks can arrive out of order. :func:`claim_stop` writes the claim item
``calls["<CallSid>#runtime-stop"]`` (``#`` keeps it apart from ``CallState``, ``#session`` and the
``#gather#`` claims; same TTL as the call) with a ``not_exists`` condition; only the request whose
write succeeds attempts the stop.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

from owlbank import obs
from owlbank.sids import CALL_SID_RE
from owlbank.store import ConditionFailed, Store

from .relay import RUNTIME_ARN_RE

log = logging.getLogger(__name__)

TABLE = "calls"
SUFFIX = "#runtime-stop"
#: The ``StopRuntimeSession`` client's timeouts: the status webhook waits for it.
CONNECT_TIMEOUT_S = 2
READ_TIMEOUT_S = 2
NOT_FOUND_CODE = "ResourceNotFoundException"


class SessionStopper(Protocol):
    def stop(self, call_sid: str) -> str: ...  # the outcome; never raises


def claim_key(call_sid: str) -> str:
    return f"{call_sid}{SUFFIX}"


def claim_stop(store: Store, call_sid: str, *, expires_at: int) -> bool:
    """Claim the one stop attempt for ``call_sid``: ``True`` for the first caller only.

    A store error other than the condition is logged and counts as not claimed (no attempt; the
    idle timeout stops the session)."""
    try:
        store.put(TABLE, {"call_sid": claim_key(call_sid), "expires_at": int(expires_at)},
                  condition="not_exists")
    except ConditionFailed:
        return False
    except Exception as exc:
        obs.log_caught(log, "runtime_session_stop_claim_failed", exc, call_sid=call_sid,
                       decision="idle_timeout_stops_it")
        return False
    return True


class RuntimeSessions:
    """``StopRuntimeSession`` on our Runtime, for a call's session (session ID = CallSid)."""

    def __init__(self, client: Any, runtime_arn: str, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not isinstance(runtime_arn, str) or not RUNTIME_ARN_RE.match(runtime_arn):
            raise ValueError("runtime_arn is not an AgentCore Runtime ARN")
        self._client = client
        self._arn = runtime_arn
        self._clock = clock

    def stop(self, call_sid: str) -> str:
        """Stop ``call_sid``'s session; return and log the outcome. Never raises."""
        if not isinstance(call_sid, str) or not CALL_SID_RE.match(call_sid):
            outcome = "error:ValueError"  # never sent: the session ID would be refused anyway
            obs.emit(log, logging.WARNING, "runtime_session_stopped", outcome=outcome,
                     decision="skipped_not_a_callsid")
            return outcome
        started = self._clock()
        code: str | None = None
        try:
            self._client.stop_runtime_session(agentRuntimeArn=self._arn, runtimeSessionId=call_sid)
        except Exception as exc:  # noqa: BLE001 - logged below; a stop must never fail the response
            code = obs.error_code_of(exc)
            outcome = "not_found" if code == NOT_FOUND_CODE else f"error:{type(exc).__name__}"
        else:
            outcome = "ok"
        duration = round((self._clock() - started) * 1000, 1)
        if outcome.startswith("error:"):
            obs.emit(log, logging.WARNING, "runtime_session_stopped", call_sid=call_sid,
                     outcome=outcome, error_code=code, duration_ms=duration,
                     decision="idle_timeout_stops_it")
        else:
            obs.emit(log, logging.INFO, "runtime_session_stopped", call_sid=call_sid,
                     outcome=outcome, duration_ms=duration)
        return outcome
