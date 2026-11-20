"""Start the call recap writer once a call is over.

:class:`RecapTrigger` is what the front door and the human-leg close path call with a CallSid.
:class:`EventRecap` is an asynchronous Lambda invoke (``InvocationType=Event``, payload
``{"call_sid": ...}`` and nothing else) of the ``recap_writer`` function named by
:data:`RECAP_FUNCTION_ENV`, the same pattern as the front door's ``EventWarmer``; :class:`NoopRecap`
stands in when that variable is unset. :func:`fire` is how both callers use it: a failed or slow
invoke is logged (``recap_trigger_failed``) and ignored, never raised, so the caller's response is
the same whatever the trigger does.

**When**: (a) the human-leg close, once Orchestrator confirmed it, even if recording
``conversation_closed_at`` then failed (:class:`owlbank.humanleg_close.HumanLegCloser`, the front
door's close and the sweep's); (b) ``/voice/status`` ``completed`` or ``failed`` when there is no
human-leg close to wait for (:func:`at_status`); (c) the sweep giving up on an escalated call
(``sweep_retries`` reached, :func:`owlbank.humanleg.sweep_gave_up`; source ``sweep_gave_up``). Each
path triggers at most once per call; the writer's claim admits one writer across them.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from owlbank import humanleg, obs
from owlbank.obs import timed_call

log = logging.getLogger(__name__)

#: The ``recap_writer`` function's name or ARN (infra sets it on the front door and the sweep).
RECAP_FUNCTION_ENV = "OWLBANK_RECAP_FUNCTION"
#: The invoke's client timeouts (one attempt): an Event invoke is accepted in milliseconds, and the
#: status callback and the close share their Lambda's 10 s with the close and the Runtime stop.
CONNECT_TIMEOUT_S = 0.3
READ_TIMEOUT_S = 0.7

_CALL_SID_RE = re.compile(r"^CA[0-9a-fA-F]{32}$")


class RecapTrigger(Protocol):
    def __call__(self, call_sid: str) -> None:
        """Start the recap writer for ``call_sid``. May raise; callers go through :func:`fire`."""
        ...


class NoopRecap:
    """No writer configured (:data:`RECAP_FUNCTION_ENV` unset): nothing is started."""

    def __call__(self, call_sid: str) -> None:
        obs.emit(log, logging.DEBUG, "recap_trigger_skipped", call_sid=call_sid, reason="no_function",
                 decision="no_op")


class EventRecap:
    """An ``Event`` invoke of the recap writer with ``{"call_sid": ...}``. Raises on any failure
    (a non-202 answer included); :func:`fire` logs it and goes on."""

    def __init__(self, client: Any, function_name: str) -> None:
        if not function_name:
            raise ValueError("the recap function name is empty")
        self._client = client
        self._function = function_name

    def __call__(self, call_sid: str) -> None:
        with timed_call("lambda", "invoke_event:recap"):
            resp = self._client.invoke(FunctionName=self._function, InvocationType="Event",
                                       Payload=json.dumps({"call_sid": call_sid}).encode())
        status = resp.get("StatusCode") if isinstance(resp, Mapping) else None
        if status != 202:
            raise RuntimeError(f"recap invoke returned status {status}")


def client_config() -> Any:
    """The ``botocore`` client config for the recap invoke, shared by the front door and the sweep:
    :data:`CONNECT_TIMEOUT_S`, :data:`READ_TIMEOUT_S` and one attempt in total."""
    from botocore.config import Config

    return Config(connect_timeout=CONNECT_TIMEOUT_S, read_timeout=READ_TIMEOUT_S,
                  retries={"mode": "standard", "total_max_attempts": 1})


def recap_trigger(env: Mapping[str, str], client_factory: Callable[[], Any]) -> RecapTrigger:
    """:class:`EventRecap` over ``client_factory()`` (a Lambda client, built only when the function is
    configured) when :data:`RECAP_FUNCTION_ENV` is set, else :class:`NoopRecap`."""
    function = (env.get(RECAP_FUNCTION_ENV) or "").strip()
    if not function:
        return NoopRecap()
    return EventRecap(client_factory(), function)


def fire(trigger: RecapTrigger | None, call_sid: str, *, source: str) -> bool:
    """Call ``trigger`` for ``call_sid``; ``True`` if it returned. Never raises: a failure is logged
    with the call SID and ``source`` (``status``, ``close``, ``sweep_gave_up``) and ignored. A
    malformed CallSid is not sent."""
    if trigger is None:
        return False
    if not isinstance(call_sid, str) or not _CALL_SID_RE.match(call_sid):
        obs.emit(log, logging.WARNING, "recap_trigger_skipped", reason="bad_call_sid", source=source,
                 decision="no_op")
        return False
    try:
        trigger(call_sid)
    except Exception as exc:  # noqa: BLE001 - the contract: never affect the caller
        obs.log_caught(log, "recap_trigger_failed", exc, call_sid=call_sid, source=source,
                       decision="ignored_no_recap")
        return False
    return True


def at_status(escalation: Mapping[str, Any] | None, *, ai_sessions: int, session_conversation: str | None,
              closer_wired: bool) -> bool:
    """Whether ``/voice/status`` (``completed``/``failed``) starts the writer itself (b): ``True``
    when no human-leg close will start it, that is when the close path isn't wired
    (``humanLegConversation=off``), the call never escalated (``not_escalated``), or the escalation has
    no conversation (none recorded, none known to the session, and no AI session that the closer could
    look one up for). ``False`` when the conversation is already closed (that close started it) or a
    close is coming (now, in this request, or after the transcripts end)."""
    if not closer_wired or not isinstance(escalation, Mapping) or not escalation:
        return True
    if humanleg.conversation_closed(escalation):
        return False
    recorded = escalation.get(humanleg.K_CONVERSATION_ID)
    has_conversation = (isinstance(recorded, str) and bool(recorded)) or bool(session_conversation) \
        or ai_sessions > 0
    return not has_conversation
