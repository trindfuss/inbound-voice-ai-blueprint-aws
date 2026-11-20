"""Close the call's Orchestrator conversation after the human leg.

:class:`HumanLegCloser` is the front door's ``ConversationCloser`` (``humanLegConversation=on``) and the
close path the stale-conversation sweep reuses. It runs the pure rules of :mod:`owlbank.humanleg`:

1. :func:`~owlbank.humanleg.end_decision` on the call state's ``escalation`` map, written with the
   conditional-update helper (:func:`owlbank.callstate.mutate`), so of two concurrent end signals only
   the one whose write takes the close claim goes on;
2. the task's reservations (TaskRouter), then :func:`~owlbank.humanleg.close_plan`;
3. stop the named transcription if it is still running (best effort), add any missed worker, ``PUT``
   CLOSED, and only then record ``conversation_closed_at`` (nothing is written after it).

A call that never escalated is left alone (AI-only calls are unchanged). Nothing raises: every
failure is logged with the call SID and conversation ID (never a key, never request parameters) and
decided as ``ignored_best_effort``; a close that fails releases its claim, so a later signal or the sweep
retries, and Orchestrator's closed timeout stays the last backstop.

**Deadline.** Every Twilio request of one close (resolve, reservations, stop, adds, GET, PUT) runs
under one :func:`owlbank.humanleg_client.budget` of :data:`CLOSE_DEADLINE_S`: each request's timeout is
cut to what is left, and none is sent once it is spent. A close cut short releases its claim
(``human_leg_close_deadline``) and is left to the next signal or the sweep.

**Journal.** A close that records ``conversation_closed_at`` also appends one
``conversation_closed`` event (``conversation_id``, ``closed_by``; no content) when a journal is given.

**Recap.** A close Orchestrator confirmed then starts the call recap writer
once, when a ``recap`` trigger is given (:func:`owlbank.recap_trigger.fire`: a failure is logged and
ignored), also when recording ``conversation_closed_at`` failed; not when another
close had already recorded it, and never after a close Orchestrator did not confirm. This covers the
front door's close and the sweep's.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from owlbank import callstate, humanleg, obs, recap_trigger
from owlbank.humanleg import ClosePlan, EndDecision
from owlbank.humanleg_client import HumanLegClient, Reservation, budget, budget_left
from owlbank.recap_trigger import RecapTrigger
from owlbank.store import Store

log = logging.getLogger(__name__)

#: One close's Twilio time in total. The front door's Lambda has 10 s (``TIMEOUTS`` in
#: ``infra/owlbank_infra/compute_stack.py``) and after the close ``/voice/status`` may still stop the
#: AgentCore Runtime session (connect 2 s + read 2 s, ``runtime_sessions``): 4 + 4 s leaves 2 s for the
#: DynamoDB reads and writes around them (the claim, the record or release, the journal) and the
#: rest of the request. A healthy close is a few requests of a few hundred ms each, well inside it;
#: 4 s still allows two of the 2 s per-request timeouts (``HUMAN_LEG_TIMEOUT_S``). The sweep reuses
#: it per close, well under its 20 s stop margin.
CLOSE_DEADLINE_S = 4.0

#: ``conversation_closed.closed_by`` for each end signal.
CLOSED_BY: Mapping[str, str] = {humanleg.CALL_ENDED: "status",
                                humanleg.TRANSCRIPTION_STOPPED: "transcription",
                                humanleg.TRANSCRIPTION_ERROR: "transcription",
                                humanleg.SWEEP: "sweep"}


class Journal(Protocol):
    def append(self, call_sid: str, event: Mapping[str, Any]) -> int: ...


class TranscriptionStopper(Protocol):
    def stop_transcription(self, call_sid: str, name: str) -> bool: ...


class ReservationLister(Protocol):
    def list(self, task_sid: str) -> list[Reservation] | None: ...


class HumanLegCloser:
    """The close path of over the call-state ``store`` and the Twilio clients.

    ``calls`` (stop a transcription) and ``reservations`` (the participant repair) are optional:
    without them that step is skipped and logged. ``client`` is the Orchestrator client the
    escalation already uses (it also remembers the ids it has seen CLOSED). ``journal`` (the
    call-observability ``SegmentLog``) gets the ``conversation_closed`` event; without it none is
    written. ``deadline_s`` bounds one close's Twilio requests (:data:`CLOSE_DEADLINE_S`). ``recap``
    starts the recap writer after a confirmed close; without it none is started."""

    def __init__(self, store: Store, client: HumanLegClient, *, calls: TranscriptionStopper | None = None,
                 reservations: ReservationLister | None = None,
                 clock: Callable[[], float] = time.time, journal: Journal | None = None,
                 deadline_s: float = CLOSE_DEADLINE_S, recap: RecapTrigger | None = None) -> None:
        self._store = store
        self._client = client
        self._calls = calls
        self._reservations = reservations
        self._clock = clock
        self._journal = journal
        self._deadline_s = deadline_s
        self._recap = recap

    @property
    def deadline_s(self) -> float:
        """One close's Twilio time in total (cut further only by an enclosing ``budget``)."""
        return self._deadline_s

    # ------------------------------------------------------------------ ConversationCloser

    def close(self, *, call_sid: str, conversation_id: str | None) -> None:
        """``/voice/status`` ``completed`` or ``failed``: close now if no transcription is running,
        else record the call's end and wait for ``transcription-stopped``."""
        self.end(call_sid, humanleg.CALL_ENDED, conversation_id=conversation_id)

    def transcription_ended(self, *, call_sid: str, conversation_id: str | None, event: str,
                            leg: str | None = None) -> None:
        """``/voice/transcription``: :data:`~owlbank.humanleg.TRANSCRIPTION_STOPPED` or
        :data:`~owlbank.humanleg.TRANSCRIPTION_ERROR` for ``leg``'s transcription (``None``: the
        customer leg) of the customer call ``call_sid``, whose state records the legs."""
        self.end(call_sid, event, conversation_id=conversation_id, leg=leg)

    # ------------------------------------------------------------------ the close path

    def end(self, call_sid: str, event: str, *, conversation_id: str | None = None,
            leg: str | None = None) -> bool:
        """Apply the end signal ``event`` to the customer call ``call_sid`` (a transcription's end:
        to its ``leg`` only, :func:`~owlbank.humanleg.end_decision`; a leg the call's state does not
        record changes nothing); ``True`` only if this call closed the conversation. Never raises."""
        try:
            return self._end(call_sid, event, conversation_id, leg)
        except Exception as exc:  # the contract: never raise (a status callback must answer)
            obs.log_caught(log, "human_leg_close_failed", exc, call_sid=call_sid,
                           conversation_id=conversation_id, operation="close",
                           decision="ignored_best_effort")
            return False

    def _end(self, call_sid: str, event: str, conversation_id: str | None, leg: str | None) -> bool:
        now = int(self._clock())
        state = callstate.load(self._store, call_sid)
        first = humanleg.end_decision(state.escalation if state else None, event, now, leg=leg)
        if state is None or first.escalation is None:
            self._log_decision(call_sid, conversation_id, event, first, leg)
            return False
        decided: list[EndDecision] = []

        def decide(s: callstate.CallState) -> Mapping[str, Any]:
            d = humanleg.end_decision(s.escalation, event, now, leg=leg)
            decided[:] = [d]
            return {"escalation": d.escalation} if d.escalation is not None else {}

        state = callstate.mutate(self._store, call_sid, decide, now=self._clock)
        decision = decided[-1]
        self._log_decision(call_sid, conversation_id, event, decision, leg)
        if not decision.close:
            return False
        claim = (state.escalation or {}).get(humanleg.K_CLOSE_CLAIMED_AT)
        closed = out_of_time = False
        try:
            with budget(self._deadline_s):
                closed = self._run(state, conversation_id, now, event)
                out_of_time = not closed and not (budget_left() or 0.0) > 0
        finally:
            if not closed:
                self._release(call_sid, claim)
        if out_of_time:
            obs.emit(log, logging.WARNING, "human_leg_close_deadline", call_sid=call_sid,
                     conversation_id=conversation_id, deadline_s=self._deadline_s,
                     decision="claim_released_retry_later")
        return closed

    def _run(self, state: callstate.CallState, known: str | None, now: int, event: str) -> bool:
        call_sid = state.call_sid
        escalation = state.escalation or {}
        conversation_id = self._conversation_id(state, known)
        if conversation_id is None:
            obs.emit(log, logging.INFO, "human_leg_no_conversation", call_sid=call_sid,
                     operation="close", decision="ignored_best_effort")
            return False
        plan = humanleg.close_plan(state, self._task_reservations(call_sid, conversation_id, escalation),
                                   now=now)
        ids: dict[str, Any] = {"call_sid": call_sid, "conversation_id": conversation_id}
        for leg, name in plan.stop_transcriptions:  # every running leg, each on its own call
            self._stop(leg, conversation_id, name)
        added = self._repair(conversation_id, plan, ids, now)
        if not self._client.close(conversation_id):
            obs.emit(log, logging.WARNING, "human_leg_close_failed", outcome="close_refused",
                     decision="ignored_best_effort", **ids)
            if added:  # repaired all the same: listed, so neither an accept nor a retry adds them again
                self._record(call_sid, conversation_id, added, closed_at=None, ids=ids)
            return False
        recorded = self._record(call_sid, conversation_id, added, closed_at=now, ids=ids)
        if recorded is True:
            self._journal_closed(call_sid, conversation_id, event, ids)
        obs.emit(log, logging.INFO, "human_leg_closed", outcome="closed",
                 transcription_stopped=bool(plan.stop_transcriptions),
                 participants_repaired=len(added), **ids)
        # Orchestrator confirmed the close, so start the writer once for this
        # close, also when recording it failed (None); not when another close had already recorded it
        # (False: that close started it). The writer's claim dedupes the rest.
        if recorded is not False:
            recap_trigger.fire(self._recap, call_sid, source="close")
        return True

    def _journal_closed(self, call_sid: str, conversation_id: str, event: str,
                        ids: Mapping[str, Any]) -> None:
        """The one ``conversation_closed`` journal event: written by the close that recorded
        ``conversation_closed_at``, so at most once per call. Allow-listed fields only. A failure is
        logged; the close stands."""
        if self._journal is None:
            return
        try:
            self._journal.append(call_sid, {"type": "conversation_closed",
                                            "conversation_id": conversation_id,
                                            "closed_by": CLOSED_BY.get(event, "status")})
        except Exception as exc:
            obs.log_caught(log, "human_leg_close_journal_failed", exc, operation="journal",
                           decision="ignored_best_effort", **ids)

    def _record(self, call_sid: str, conversation_id: str, added: Sequence[Mapping[str, Any]], *,
                closed_at: int | None, ids: Mapping[str, Any]) -> bool | None:
        """Record the repaired participants (each worker once) and, after a confirmed close,
        ``conversation_closed_at``; ``True`` only if this write set it, ``False`` if it did not (no
        close to record, or already recorded), ``None`` if the write failed. A failure is logged:
        Orchestrator is closed all the same, and a repeat finds it CLOSED."""
        wrote: list[bool] = [False]

        def record(s: callstate.CallState) -> Mapping[str, Any]:
            wrote[0] = False
            new = dict(s.escalation or {})
            if humanleg.conversation_closed(new):
                return {}
            if closed_at is not None:
                new[humanleg.K_CONVERSATION_CLOSED_AT] = closed_at
                wrote[0] = True
            new.setdefault(humanleg.K_CONVERSATION_ID, conversation_id)
            listed = [dict(p) for p in new.get(humanleg.K_PARTICIPANTS) or [] if isinstance(p, Mapping)]
            have = humanleg.recorded_workers(new)
            fresh = [dict(p) for p in added if p.get("worker_sid") not in have]
            if fresh:
                new[humanleg.K_PARTICIPANTS] = listed + fresh
            return {"escalation": new}

        try:
            callstate.mutate(self._store, call_sid, record, now=self._clock)
        except Exception as exc:
            obs.log_caught(log, "human_leg_close_record_failed", exc, operation="close",
                           decision="ignored_best_effort", **ids)
            return None
        return wrote[0]

    # ------------------------------------------------------------------ steps

    def _conversation_id(self, state: callstate.CallState, known: str | None) -> str | None:
        """The recorded id, else the session's (``known``), else one lookup by CallSid; a call
        that never had an AI session has no conversation, so no lookup is made."""
        recorded = (state.escalation or {}).get(humanleg.K_CONVERSATION_ID)
        if isinstance(recorded, str) and recorded:
            return recorded
        if known:
            return known
        if state.ai_sessions == 0:
            return None
        found = self._client.resolve_conversation(state.call_sid)
        return found if isinstance(found, str) and found else None

    def _task_reservations(self, call_sid: str, conversation_id: str,
                           escalation: Mapping[str, Any]) -> Sequence[Reservation]:
        task_sid = escalation.get(humanleg.K_TASK_SID)
        if self._reservations is None or not isinstance(task_sid, str) or not task_sid:
            obs.emit(log, logging.INFO, "human_leg_repair_skipped", call_sid=call_sid,
                     conversation_id=conversation_id,
                     reason="no_reader" if self._reservations is None else "no_task_sid",
                     decision="ignored_best_effort")
            return ()
        found = self._reservations.list(task_sid)
        if found is None:  # already logged by the reader (human_leg_reservations_failed)
            return ()
        return found

    def _stop(self, call_sid: str, conversation_id: str, name: str) -> None:
        stopped = False
        if self._calls is not None:
            try:
                stopped = self._calls.stop_transcription(call_sid, name) is True
            except Exception as exc:  # the client never raises; this guards the close anyway
                obs.log_caught(log, "human_leg_stop_failed", exc, call_sid=call_sid,
                               conversation_id=conversation_id, operation="stop_transcription",
                               decision="ignored_best_effort")
        # A failed stop does not hold up the close: the call has ended, so the fork ends with it.
        obs.emit(log, logging.INFO if stopped else logging.WARNING, "human_leg_transcription_stop",
                 call_sid=call_sid, conversation_id=conversation_id,
                 outcome="stopped" if stopped else "not_stopped", decision="close_anyway")

    def _repair(self, conversation_id: str, plan: ClosePlan, ids: Mapping[str, Any],
                now: int) -> list[dict[str, Any]]:
        added: list[dict[str, Any]] = []
        for worker in plan.add_workers:
            ok = self._client.add_human_participant(conversation_id, worker, worker)
            obs.emit(log, logging.INFO, "human_leg_participant_repair", outcome="added" if ok else "not_added",
                     decision="close_anyway", **ids)
            if ok:
                added.append({"worker_sid": worker, "added_at": now})
        return added

    def _release(self, call_sid: str, claim: Any) -> None:
        """Drop this attempt's close claim (only if it is still ours), so a retry can close."""
        def fn(s: callstate.CallState) -> Mapping[str, Any]:
            esc = s.escalation or {}
            if esc.get(humanleg.K_CLOSE_CLAIMED_AT) != claim or humanleg.conversation_closed(esc):
                return {}
            new = dict(esc)
            new.pop(humanleg.K_CLOSE_CLAIMED_AT, None)
            return {"escalation": new}

        try:
            callstate.mutate(self._store, call_sid, fn, now=self._clock)
        except Exception as exc:  # the claim then expires after CLOSE_CLAIM_TTL_S
            obs.log_caught(log, "human_leg_close_release_failed", exc, call_sid=call_sid,
                           operation="close", decision="claim_expires")

    @staticmethod
    def _log_decision(call_sid: str, conversation_id: str | None, event: str, d: EndDecision,
                      leg: str | None = None) -> None:
        if d.reason == "not_escalated":
            obs.emit(log, logging.DEBUG, "human_leg_close_skipped", call_sid=call_sid, trigger=event,
                     reason=d.reason, decision="no_op")
            return
        name = "human_leg_after_close" if d.reason == "after_close" else "human_leg_end_signal"
        obs.emit(log, logging.INFO, name, call_sid=call_sid, conversation_id=conversation_id,
                 trigger=event, outcome=d.reason, decision="close" if d.close else "no_op",
                 **({"leg": leg} if leg is not None and leg != call_sid else {}))
