"""Lambda entry point for the stale-conversation sweep.

Invoked by an EventBridge Scheduler schedule every 10 minutes (no API, URL or other trigger; never
retried, the next run comes anyway); the event is ignored. The sweep works from our own records,
never from Twilio's conversation list, so its work grows only with escalated calls whose
conversation is not yet closed. One run:

1. one DynamoDB ``Query`` of the ``calls`` table's sparse KEYS_ONLY index ``sweep-due`` for
   ``sweep_due_at <= now``, oldest first (:func:`store_due`). Only escalated calls whose conversation is
   not closed, that can have a conversation (an AI session ran, or an id is recorded) and have fewer
   than 6 retries are in it (``owlbank.callstate.CallState.to_item`` derives its keys for every
   writer, :func:`owlbank.humanleg.sweep_indexed`);
2. per chunk of :data:`STATE_CHUNK` due calls, their call state in one strongly consistent
   ``BatchGetItem`` (:func:`store_states`). A stale index entry (no or expired call state, no longer
   in the index on the fresh read, or no longer due) is skipped before any Twilio read;
3. for a call with no recorded ``call_ended_at`` only, the Calls API status
   (``GET /2010-04-01/Accounts/{AC}/Calls/{CallSid}.json``, :class:`CallStatusReader`): most ended
   calls cost no Twilio read, because ``/voice/status`` recorded their end;
4. :func:`owlbank.humanleg.sweep_step` decides: **ended** more than 5 minutes ago → the close path the
   front door uses: ``HumanLegCloser.end(call_sid, humanleg.SWEEP)``, with the same per-close
   deadline (``CLOSE_DEADLINE_S``), journalling ``conversation_closed`` (``closed_by: sweep``);
   **ended within 5 minutes** → due again just after end + 5 minutes; **live** (in queue or
   with an agent) → due again after half its age since escalation, between 10 and 20 minutes, never
   closed however long it waits; **unknown** (no usable status) and a close **not done** → due again
   in 10 minutes and one more ``sweep_retries`` (6 in a row and the call leaves the index for good;
   a live answer or a wait resets it; the retry that gives up starts the recap writer once,
   ``sweep_gave_up``), so a call that stays unknown never holds the front of the
   queue. Retry and live due times are written
   60 s early (``SWEEP_SLACK_S``) so a run that starts a little early still finds them due; the wait
   after an end is not. Each new due time is written as ``escalation.sweep_next_at`` through ``callstate.mutate`` with a pure function that writes nothing once the conversation is
   closed.

Failures: a failed index query fails the run (``conversation_sweep_query_failed``, then an
exception, so the Lambda's Errors metric and its alarm see it); a failed call-state or status read
leaves those calls due (one call-state item that does not parse leaves only its call); a failed
reschedule leaves its call due; a failed close releases its claim. Orchestrator's closed timeout stays
the last backstop.

The run stops starting new reads or closes when less than :data:`STOP_MARGIN_S` of the function's time
remains (a started close ends within its deadline, well inside that margin). The calls it did not reach
keep their due time, so they come first in the next run, before every rescheduled call. One summary
line, ``conversation_sweep``, with counts only (never a key, a URL or a body); its ``decision`` is
``done`` unless the run stopped for time (``out_of_time``), then ``continue_next_run``.

Environment:

* ``OWLBANK_SECRET_TWILIO_API_KEY`` (required): the ``twilio_api_key`` secret's ARN, read on first use.
* ``OWLBANK_TABLE_CALLS``: the ``calls`` table (the ``sweep-due`` index query; call state batch-read
  by the run, read and written by the reschedule and the close path).
* ``OWLBANK_TABLE_SEGMENTS``: the journal (the close's ``conversation_closed`` event).
* ``OWLBANK_RECORD_TTL_S`` (optional, default 86400): the journal event's lifetime, as on the front door.
* ``OWLBANK_TASKROUTER_WORKSPACE_SID`` (optional): lets the close add a worker whose accept was
  missed, as on the front door.
* ``OWLBANK_RECAP_FUNCTION`` (optional): the ``recap_writer`` function, invoked
  (``Event``) by a confirmed close, and once for a call the sweep gives up on; unset, no recap is
  started (:mod:`owlbank.recap_trigger`).
* ``OWLBANK_LOG_LEVEL``, ``AWS_REGION``.

Cold start: importing this module outside Lambda makes no AWS call; inside Lambda the
dependencies are built once during INIT (``coldstart.build_at_init``). No secret is read at INIT.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from owlbank import callstate, coldstart, humanleg, obs
from owlbank.callstate import CallState
from owlbank.humanleg import CallEnd

# _send and _default_opener: the same https-only request helper and opener as the other Twilio REST
# clients (one place for Basic auth, the https check and the timing metric).
from owlbank.humanleg_client import (
    DEFAULT_API_BASE_URL,
    DEFAULT_TIMEOUT_S,
    Opener,
    OrchestratorHumanLeg,
    TwilioCalls,
    _default_opener,
    _require_https,
    _send,
    credentials_from_secret,
    taskrouter_reservations,
)
from owlbank.humanleg_close import HumanLegCloser
from owlbank.recap_trigger import RecapTrigger, client_config, recap_trigger
from owlbank.recap_trigger import fire as fire_recap
from owlbank.segments import SegmentLog

log = logging.getLogger(__name__)

API_KEY_SECRET_ENV = "OWLBANK_SECRET_TWILIO_API_KEY"  # noqa: S105 - a variable name, not a secret
TASKROUTER_WORKSPACE_ENV = "OWLBANK_TASKROUTER_WORKSPACE_SID"
#: Each Twilio request's timeout (a sweep is not on a call's path, but a run must still end).
REQUEST_TIMEOUT_S = DEFAULT_TIMEOUT_S
#: Stop starting work when less than this remains: one status read (at most 3 s) or one close (its
#: Twilio requests at most ``CLOSE_DEADLINE_S`` in total, plus a few DynamoDB writes).
STOP_MARGIN_S = 20.0
#: Due calls whose state one ``BatchGetItem`` reads (DynamoDB's own limit is 100 keys).
STATE_CHUNK = 100

_CALL_SID_RE = re.compile(r"^CA[0-9a-fA-F]{32}$")
_ACCOUNT_SID_RE = re.compile(r"^AC[0-9a-fA-F]{32}$")


class SweepFailed(RuntimeError):
    """The run could not query the due calls (no detail: the cause is logged before)."""


class DueReader(Protocol):
    def __call__(self, now: int) -> Sequence[str]:
        """The CallSids due at ``now`` (``sweep_due_at <= now``), oldest due first. May raise."""
        ...


class StateReader(Protocol):
    def __call__(self, call_sids: Sequence[str]) -> Mapping[str, CallState]: ...


class StatusReader(Protocol):
    def status(self, call_sid: str) -> CallEnd | None: ...


class Closer(Protocol):
    def end(self, call_sid: str, event: str, *, conversation_id: str | None = None) -> bool: ...


EscalationChange = Callable[[Mapping[str, Any] | None], Mapping[str, Any] | None]


class Writer(Protocol):
    def __call__(self, call_sid: str, change: EscalationChange) -> bool:
        """Apply ``change`` to the call's freshly read ``escalation`` (``None``: write nothing);
        ``True`` if it wrote. May raise."""
        ...


def _end_time(raw: Any) -> float | None:
    """Twilio's RFC 2822 ``end_time`` (``Thu, 05 Nov 2026 10:00:00 +0000``) as epoch seconds; ``None``
    for anything else, a time with no zone included."""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed: datetime = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


class CallStatusReader:
    """A call's status and end time from the Calls API. Never raises: ``None`` on bad input or any
    failure (404 included), which the sweep treats as unknown and leaves alone."""

    def __init__(self, credentials: Callable[[], tuple[str, str, str]], *,
                 base_url: str = DEFAULT_API_BASE_URL, timeout_s: float = REQUEST_TIMEOUT_S,
                 opener: Opener = _default_opener) -> None:
        _require_https(base_url)
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._opener = opener

    def status(self, call_sid: str) -> CallEnd | None:
        if not isinstance(call_sid, str) or not _CALL_SID_RE.match(call_sid):
            obs.emit(log, logging.WARNING, "conversation_sweep_bad_argument", operation="calls.fetch",
                     decision="return_none")
            return None
        try:
            account_sid, key, secret = self._credentials()
            if not _ACCOUNT_SID_RE.match(account_sid or ""):
                obs.emit(log, logging.WARNING, "conversation_sweep_status_failed", call_sid=call_sid,
                         reason="account_sid", decision="leave_for_next_run")
                return None
            url = f"{self._base_url}/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json"
            reply = _send(self._opener, self._timeout_s, "GET", url, "twilio_calls", "calls.fetch",
                          (key, secret), None, "application/json")
            data = reply.data or {}
            status = data.get("status")
            if not reply.ok or not isinstance(status, str) or not status:
                obs.emit(log, logging.WARNING, "conversation_sweep_status_failed", call_sid=call_sid,
                         http_status=reply.status, decision="leave_for_next_run")
                return None
            return CallEnd(status.strip().lower(), _end_time(data.get("end_time")))
        except Exception as exc:  # noqa: BLE001 - the contract: never raise
            obs.log_caught(log, "conversation_sweep_status_failed", exc, call_sid=call_sid,
                           operation="calls.fetch", decision="leave_for_next_run")
            return None


@dataclass(frozen=True, slots=True)
class SweepDeps:
    due: DueReader
    states: StateReader
    calls: StatusReader
    closer: Closer
    write: Writer
    clock: Callable[[], float] = time.time
    #: Started once for a call the sweep gives up on (``sweep_gave_up``).
    recap: RecapTrigger | None = None


@dataclass(frozen=True, slots=True)
class SweepReport:
    """Counts of one run (what the summary line and the handler's return value carry). ``due``: index
    entries due at the run's start; ``skipped``: stale ones (no or expired call state, not escalated,
    closed, or not due after all); ``statuses``: Calls API reads; ``closed``: conversations closed;
    ``rescheduled``: due times moved forward (live, ended within the grace period, close not done);
    ``unknown``: calls with no usable status (due again in 10 minutes) or a failed read or write
    (left due);
    ``out_of_time``: the run stopped for time before the end of the due calls."""

    due: int = 0
    skipped: int = 0
    statuses: int = 0
    closed: int = 0
    rescheduled: int = 0
    unknown: int = 0
    out_of_time: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"due": self.due, "skipped": self.skipped, "statuses": self.statuses, "closed": self.closed,
                "rescheduled": self.rescheduled, "unknown": self.unknown, "out_of_time": self.out_of_time}


def store_due(store: Any) -> DueReader:
    """The due calls from the ``sweep-due`` index (one paginated ``Query``, eventually consistent:
    a call written a moment ago may show next run; the state is re-read strongly before any write)."""
    def read(now: int) -> list[str]:
        items = store.query_range(callstate.TABLE, {callstate.SWEEP_OPEN_ATTR: callstate.SWEEP_OPEN},
                                  low=0, high=int(now), index=callstate.SWEEP_INDEX)
        sids = (i.get("call_sid") for i in items if isinstance(i, Mapping))
        return [s for s in sids if isinstance(s, str) and _CALL_SID_RE.match(s)]
    return read


def store_states(store: Any, *, clock: Callable[[], float] = time.time) -> StateReader:
    """The call states of ``call_sids`` from the ``calls`` table, in ``BatchGetItem`` requests
    (:meth:`owlbank.store.Store.batch_get`, strongly consistent); expired ones are left out (TTL is
    lazy). An item that does not parse is left out on its own (that call is unknown, the rest of the
    chunk is read)."""
    def read(call_sids: Sequence[str]) -> Mapping[str, CallState]:
        now = clock()
        found: dict[str, CallState] = {}
        for item in store.batch_get(callstate.TABLE, [{"call_sid": s} for s in call_sids]):
            try:
                state = CallState.from_item(item)
                if state.is_expired(now):
                    continue
                key = state.call_sid
            except Exception as exc:  # one bad item: that call is left for the next run
                sid = item.get("call_sid") if isinstance(item, Mapping) else None
                obs.log_caught(log, "conversation_sweep_state_unreadable", exc, operation="calls.batch_get",
                               decision="leave_for_next_run",
                               call_sid=sid if isinstance(sid, str) and _CALL_SID_RE.match(sid) else None)
                continue
            found[key] = state
        return found
    return read


def store_writer(store: Any, *, clock: Callable[[], float] = time.time) -> Writer:
    """The sweep's call-state write: ``callstate.mutate`` over ``change`` (the conditional update, so a
    concurrent writer is re-read and ``change`` re-decided; ``None`` writes nothing at all)."""
    def write(call_sid: str, change: EscalationChange) -> bool:
        wrote = [False]

        def fn(s: CallState) -> Mapping[str, Any] | None:
            new = change(s.escalation)
            wrote[0] = new is not None
            return None if new is None else {"escalation": dict(new)}

        callstate.mutate(store, call_sid, fn, now=clock)
        return wrote[0]
    return write


def _write(deps: SweepDeps, call_sid: str, change: EscalationChange) -> bool:
    try:
        return deps.write(call_sid, change)
    except Exception as exc:  # the call keeps its due time: first in line next run
        obs.log_caught(log, "conversation_sweep_reschedule_failed", exc, call_sid=call_sid,
                       operation="calls.mutate", decision="leave_for_next_run")
        return False


def _rescheduled(call_end: CallEnd | None, now: int) -> EscalationChange:
    def change(escalation: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        return humanleg.sweep_reschedule(escalation, call_end, now)
    return change


def _retried(now: int, seen: Mapping[str, Any] | None, gave_up: list[bool]) -> EscalationChange:
    """The retry write; ``gave_up[0]`` says whether its last decision (``callstate.mutate`` may re-run
    it on a fresh read) was the first give-up (:func:`owlbank.humanleg.sweep_gave_up`)."""
    seen_ended_at = (seen or {}).get(humanleg.K_CALL_ENDED_AT)

    def change(escalation: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        new = humanleg.sweep_retry(escalation, now, seen_ended_at=seen_ended_at)
        gave_up[0] = humanleg.sweep_gave_up(escalation, new)
        return new
    return change


def _retry(deps: SweepDeps, call_sid: str, now: int, seen: Mapping[str, Any] | None) -> bool:
    """Write a retry; when that write was the sweep giving up on the call (its last retry, the call
    leaves the index), start the recap writer once (the writer's
    claim dedupes, and its status-free lookup may still find the conversation). ``True`` if it wrote."""
    gave_up = [False]
    wrote = _write(deps, call_sid, _retried(now, seen, gave_up))
    if wrote and gave_up[0]:
        obs.emit(log, logging.INFO, "conversation_sweep_gave_up", call_sid=call_sid,
                 retries=humanleg.SWEEP_MAX_RETRIES, decision="start_recap")
        fire_recap(deps.recap, call_sid, source="sweep_gave_up")
    return wrote


def run_sweep(deps: SweepDeps, *, time_left: Callable[[], float] | None = None) -> SweepReport:
    """One sweep (module docstring). Raises :class:`SweepFailed` only when the index query fails."""
    left = time_left or (lambda: float("inf"))
    now = int(deps.clock())
    try:
        due = list(dict.fromkeys(deps.due(now)))  # each call once, in due order
    except Exception as exc:
        obs.log_caught(log, "conversation_sweep_query_failed", exc, level=logging.ERROR,
                       operation="calls.query", decision="leave_for_next_run")
        raise SweepFailed("sweep-due index query failed") from None
    skipped = statuses = closed = rescheduled = unknown = 0
    short = False
    for first in range(0, len(due), STATE_CHUNK):
        chunk = due[first:first + STATE_CHUNK]
        if left() < STOP_MARGIN_S:
            short = True
            break
        try:
            states = deps.states(chunk)
        except Exception as exc:  # unknown state: these stay due for the next run
            obs.log_caught(log, "conversation_sweep_state_failed", exc, operation="calls.batch_get",
                           decision="leave_for_next_run")
            unknown += len(chunk)
            continue
        for sid in chunk:
            state = states.get(sid)
            escalation = state.escalation if state is not None else None
            if state is None or not humanleg.sweep_indexed(escalation, state.ai_sessions) \
                    or humanleg.sweep_due_at(escalation) > now:
                skipped += 1  # a stale index entry (eventually consistent, or expired state)
                continue
            call_end: CallEnd | None = None
            if humanleg.sweep_needs_status(escalation):
                if left() < STOP_MARGIN_S:
                    short = True
                    break
                call_end = deps.calls.status(sid)
                statuses += 1
            step = humanleg.sweep_step(escalation, call_end, now)
            if step.outcome == humanleg.STEP_CLOSE:
                if left() < STOP_MARGIN_S:
                    short = True
                    break
                recorded = (escalation or {}).get(humanleg.K_CONVERSATION_ID)
                with obs.call_context(sid):
                    done = deps.closer.end(sid, humanleg.SWEEP,
                                           conversation_id=recorded if isinstance(recorded, str) and recorded
                                           else None)
                if done:
                    closed += 1
                elif _retry(deps, sid, now, escalation):
                    rescheduled += 1
                else:
                    unknown += 1
            elif step.outcome in (humanleg.STEP_WAIT, humanleg.STEP_LIVE):
                if _write(deps, sid, _rescheduled(call_end, now)):
                    rescheduled += 1
                else:
                    unknown += 1
            else:  # unknown: back off like a close not done (a failed write leaves it due)
                _retry(deps, sid, now, escalation)
                unknown += 1
        if short:
            break
    report = SweepReport(due=len(due), skipped=skipped, statuses=statuses, closed=closed,
                         rescheduled=rescheduled, unknown=unknown, out_of_time=short)
    obs.emit(log, logging.WARNING if short else logging.INFO, "conversation_sweep",
             decision="continue_next_run" if short else "done", **report.as_dict())
    return report


def _lambda_client() -> Any:  # pragma: no cover - needs boto3 credentials only when invoked
    import boto3

    return coldstart.LazyClient(lambda: boto3.client("lambda", config=client_config()))


def build_deps(env: Mapping[str, str] | None = None, *, store: Any | None = None,
               recap: RecapTrigger | None = None) -> SweepDeps:
    """The sweep over the ``twilio_api_key`` secret (read on first use) and the ``calls`` table. ``recap``
    (default: from ``OWLBANK_RECAP_FUNCTION``) is handed to the closer and started once for a
    call the sweep gives up on."""
    env = dict(os.environ) if env is None else env
    secret = (env.get(API_KEY_SECRET_ENV) or "").strip()
    if not secret:
        raise ValueError(f"{API_KEY_SECRET_ENV} is required")
    if store is None:
        from owlbank.store import DynamoStore

        store = DynamoStore({"calls": env.get("OWLBANK_TABLE_CALLS") or "owlbank-calls",
                             "segments": env.get("OWLBANK_TABLE_SEGMENTS") or "owlbank-segments"},
                            region_name=env.get("AWS_REGION") or None, required=("calls", "segments"))
    credentials = credentials_from_secret(secret)
    client = OrchestratorHumanLeg(credentials, timeout_s=REQUEST_TIMEOUT_S)
    workspace = (env.get(TASKROUTER_WORKSPACE_ENV) or "").strip() or None
    if recap is None:
        recap = recap_trigger(env, _lambda_client)
    closer = HumanLegCloser(
        store, client,
        calls=TwilioCalls(credentials, timeout_s=REQUEST_TIMEOUT_S),
        reservations=(taskrouter_reservations(workspace, credentials, timeout_s=REQUEST_TIMEOUT_S)
                      if workspace else None),
        clock=time.time,
        journal=SegmentLog(store, ttl_s=int(env.get("OWLBANK_RECORD_TTL_S") or "86400")),
        recap=recap)
    return SweepDeps(due=store_due(store), states=store_states(store), calls=CallStatusReader(credentials),
                     closer=closer, write=store_writer(store), recap=recap)


def _time_left(context: Any) -> Callable[[], float] | None:
    get = getattr(context, "get_remaining_time_in_millis", None)
    if not callable(get):
        return None
    return lambda: get() / 1000


_deps: SweepDeps | None = None


@obs.instrumented("conversation_sweep", op="sweep")
def handler(event: Any, context: Any = None) -> dict[str, Any]:
    global _deps
    if _deps is None:
        _deps = build_deps()
    return run_sweep(_deps, time_left=_time_left(context)).as_dict()


# Inside Lambda only (a no-op in tests and scripts): wire up now, in the INIT phase.
_deps = coldstart.build_at_init(build_deps)
