"""Human-leg conversation helpers: pure functions, no I/O."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol, TypeGuard

#: The transcription's name: the start at accept sets it, the close stops it by it.
TRANSCRIPTION_NAME = "humanleg"
CUSTOMER_TRACK_LABEL = "customer"
AGENT_TRACK_LABEL = "agent"
#: A transcription's ``role`` in call state: the customer leg's, or one agent leg's.
ROLE_CUSTOMER = CUSTOMER_TRACK_LABEL
ROLE_AGENT = AGENT_TRACK_LABEL
_ROLES = frozenset({ROLE_CUSTOMER, ROLE_AGENT})
#: Agent legs transcribed per call at most, besides the customer leg.
MAX_AGENT_LEGS = 6
#: The query parameter of an agent leg's status callback that names the customer call.
CALL_QUERY_PARAM = "call"
#: A Twilio call SID.
CALL_SID_RE = re.compile(r"^CA[0-9a-fA-F]{32}$")

#: Catalog ``conversationRelay.transcriptionProvider`` (any case) to Twilio's
#: ``TranscriptionEngine`` value. The catalog names the provider as ConversationRelay does
#: ("Deepgram", "Google"); real-time transcription takes lowercase names.
_ENGINES: Mapping[str, str] = {"google": "google", "deepgram": "deepgram"}


class _HasEnabled(Protocol):
    @property
    def human_leg_transcription(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class SttSettings:
    """The speech-to-text settings the AI segment uses, as real-time transcription settings."""

    engine: str | None
    speech_model: str | None
    language_code: str | None


def transcription_enabled(policy: _HasEnabled) -> bool:
    """The ``human_leg.transcription`` switch; off when the policy can't say."""
    try:
        return policy.human_leg_transcription is True
    except Exception:  # noqa: BLE001 - a policy that can't answer reads as off
        return False


def stt_settings(relay: Mapping[str, Any] | None) -> SttSettings:
    """Engine, speech model and language the AI segment's STT uses (catalog
    ``settings(lang)["conversationRelay"]``), so the human leg is transcribed the same way.

    An unknown or missing provider gives no engine and no model (a model of another provider
    would be wrong); the caller's language then stands alone.
    """
    relay = relay if isinstance(relay, Mapping) else {}
    provider = relay.get("transcriptionProvider")
    engine = _ENGINES.get(provider.strip().lower()) if isinstance(provider, str) else None
    model = relay.get("speechModel")
    language = relay.get("transcriptionLanguage")
    return SttSettings(
        engine=engine,
        speech_model=model if engine and isinstance(model, str) and model else None,
        language_code=language if isinstance(language, str) and language else None,
    )


def transcription_start(
    conversation_id: str | None,
    *,
    enabled: bool,
    language: str | None,
    status_url: str | None,
    engine: str | None = None,
    speech_model: str | None = None,
    agent_legs: bool = False,
) -> Mapping[str, str] | None:
    """The Calls API form parameters that start the customer leg's transcription when the first
    agent accepts (``POST /Calls/{CallSid}/Transcriptions.json``), or ``None``.

    ``None`` when the ``human_leg.transcription`` switch is off or the conversation id is
    unknown or empty (the hand-off then goes on without a transcript).
    Otherwise the start is named (so the close can stop it) and carries the caller's engine,
    language, speech model and status URL; ``None`` or empty values are left out.

    ``agent_legs`` is the ``human_leg.agent_legs`` switch: on, the customer's voice only
    (``inbound_track``, label ``customer``; each agent leg gets :func:`agent_transcription_start`); off, the
    fallback: one ``both_tracks`` transcription with both labels. The
    default is the fallback, so a caller that does not pass the switch keeps today's behaviour.
    """
    if not agent_legs:
        return _start_params(conversation_id, enabled=enabled, language=language, status_url=status_url,
                             engine=engine, speech_model=speech_model, track="both_tracks",
                             inbound_label=CUSTOMER_TRACK_LABEL, outbound_label=AGENT_TRACK_LABEL)
    return _start_params(conversation_id, enabled=enabled, language=language, status_url=status_url,
                         engine=engine, speech_model=speech_model, track="inbound_track",
                         inbound_label=CUSTOMER_TRACK_LABEL)


def agent_transcription_start(
    conversation_id: str | None,
    *,
    enabled: bool,
    customer_call_sid: str | None,
    language: str | None,
    status_url: str | None,
    engine: str | None = None,
    speech_model: str | None = None,
) -> Mapping[str, str] | None:
    """The Calls API form parameters that start one agent leg's transcription, posted to the
    agent leg's ``/Calls/{leg}/Transcriptions.json``, or ``None``.

    The agent's own voice (``inbound_track``, label ``agent``), with the customer leg's name,
    ``ConversationId``, engine, language and model. ``StatusCallbackUrl`` is ``status_url`` with
    ``call=<customer call SID>`` added, so the callback maps back to the call (the query is part of
    the signed URL). ``None`` as :func:`transcription_start`, and also when ``customer_call_sid`` is
    not a call SID (the callback could not be mapped)."""
    if not isinstance(customer_call_sid, str) or not CALL_SID_RE.match(customer_call_sid):
        return None
    url = None
    if isinstance(status_url, str) and status_url:
        url = f"{status_url}{'&' if '?' in status_url else '?'}{CALL_QUERY_PARAM}={customer_call_sid}"
    return _start_params(conversation_id, enabled=enabled, language=language, status_url=url,
                         engine=engine, speech_model=speech_model, track="inbound_track",
                         inbound_label=AGENT_TRACK_LABEL)


def _start_params(conversation_id: str | None, *, enabled: bool, language: str | None,
                  status_url: str | None, engine: str | None, speech_model: str | None, track: str,
                  inbound_label: str, outbound_label: str | None = None) -> Mapping[str, str] | None:
    if not enabled or not conversation_id:
        return None
    params = {
        "Name": TRANSCRIPTION_NAME,
        "Track": track,
        "InboundTrackLabel": inbound_label,
        "OutboundTrackLabel": outbound_label,
        "TranscriptionEngine": engine,
        "LanguageCode": language,
        "SpeechModel": speech_model,
        "StatusCallbackUrl": status_url,
        # UNVERIFIED: the parameter that binds the REST start to the existing
        # Orchestrator conversation. TwiML's attribute is ``conversationId``; whether the Calls
        # API takes it, and under this name, is unknown. If not,'s fallback applies.
        "ConversationId": conversation_id,
    }
    return {k: v for k, v in params.items() if isinstance(v, str) and v}


#: Task attributes longer than this are not parsed (TaskRouter allows far less; a bound on the work).
_MAX_ATTRIBUTES = 64_000


def agent_legs(task_attributes: Any, customer_call_sid: str | None) -> list[str]:
    """The agent legs' call SIDs in the task attributes, pure, never raises.

    Flex writes the conference's legs into ``conference.participants`` (``{"customer": "CA..",
    "worker": "CA.."}``); every value that is a call SID other than ``customer_call_sid`` is an
    agent leg. ``task_attributes`` may be the attributes' JSON string or the parsed map. Each leg
    once, in the order Flex lists them (so the cap keeps the earliest), at most
    :data:`MAX_AGENT_LEGS`. Anything malformed gives ``[]``."""
    attrs: Any = task_attributes
    if isinstance(attrs, (str, bytes)):
        if not attrs or len(attrs) > _MAX_ATTRIBUTES:
            return []
        try:
            attrs = json.loads(attrs)
        except (ValueError, UnicodeDecodeError):
            return []
    conference = attrs.get("conference") if isinstance(attrs, Mapping) else None
    participants = conference.get("participants") if isinstance(conference, Mapping) else None
    if not isinstance(participants, Mapping):
        return []
    legs: list[str] = []
    for value in participants.values():
        if isinstance(value, str) and CALL_SID_RE.match(value) and value != customer_call_sid \
                and value not in legs:
            legs.append(value)
            if len(legs) == MAX_AGENT_LEGS:
                break
    return legs


#: The TaskRouter event that carries the agent legs once Flex has written them.
TASK_UPDATED_EVENT = "task.updated"


@dataclass(frozen=True, slots=True)
class LegsToStart:
    """The legs one event may start (:func:`legs_to_start`): the customer leg (``customer``) and the
    agent legs' call SIDs (``agents``), each not yet recorded. ``reason`` says why there are none."""

    customer: bool
    agents: tuple[str, ...]
    reason: str

    def __bool__(self) -> bool:
        return self.customer or bool(self.agents)


def legs_to_start(escalation: Mapping[str, Any] | None, task_attributes: Any, customer_call_sid: str,
                  *, with_agents: bool) -> LegsToStart:
    """The transcriptions an accept or a ``task.updated`` may start for the call,
    pure, before any write: nothing for a call that never escalated, before the hand-off
    (``handoff_at`` unset: no agent has accepted yet), after the call's end or the close. Otherwise the
    customer leg if it is not recorded (the first start, or a retry after a released failed start),
    and with ``with_agents`` (the ``human_leg.agent_legs`` switch) on, each agent leg in
    ``task_attributes`` (:func:`agent_legs`) not yet recorded, while fewer than
    :data:`MAX_AGENT_LEGS` are. The claims (:func:`claim_transcription`) decide again under the
    conditional write; this only lets an event with nothing new drop early without a write."""
    if not isinstance(escalation, Mapping) or not escalation:
        return LegsToStart(False, (), "not_escalated")
    if conversation_closed(escalation):
        return LegsToStart(False, (), "after_close")
    if escalation.get(K_HANDOFF_AT) is None:
        return LegsToStart(False, (), "before_handoff")
    if escalation.get(K_CALL_ENDED_AT) is not None:
        return LegsToStart(False, (), "call_ended")
    legs = transcriptions(escalation)
    customer = customer_leg(escalation) is None and customer_call_sid not in legs
    agents: tuple[str, ...] = ()
    if with_agents:
        room = MAX_AGENT_LEGS - sum(1 for e in legs.values() if e.get("role") == ROLE_AGENT)
        fresh = [leg for leg in agent_legs(task_attributes, customer_call_sid) if leg not in legs]
        agents = tuple(fresh[:max(room, 0)])
    return LegsToStart(customer, agents, "start" if customer or agents else "nothing_new")


# --------------------------------------------------------------------------- close

#: What ends the human leg, as the close path sees it:
#: ``/voice/status`` ``completed`` or ``failed``; the transcription's ``transcription-stopped`` and
#: ``transcription-error`` (on ``/voice/transcription``); the stale-conversation sweep, which
#: has itself confirmed that the call ended.
CALL_ENDED = "call_ended"
TRANSCRIPTION_STOPPED = "transcription_stopped"
TRANSCRIPTION_ERROR = "transcription_error"
SWEEP = "sweep"
END_EVENTS = frozenset({CALL_ENDED, TRANSCRIPTION_STOPPED, TRANSCRIPTION_ERROR, SWEEP})

#: Twilio's ``TranscriptionEvent`` values that matter to the close; every other one
#: (``transcription-started``, ``transcription-content``) is acknowledged and ignored.
_TRANSCRIPTION_EVENTS: Mapping[str, str] = {
    "transcription-stopped": TRANSCRIPTION_STOPPED, "transcription-error": TRANSCRIPTION_ERROR}

#: Keys of the call state's ``escalation`` map this feature reads and writes.
#: ``conversation_closed_at`` is set once, after Orchestrator confirmed the close; every writer of
#: the conversation checks it first. ``call_ended_at`` records the customer call's end
#: (``/voice/status``, or the customer leg's ``transcription-stopped``); ``transcriptions`` maps each
#: transcribed leg's call SID to ``{name, role, started_at, ended_at}``; ``close_claimed_at``
#: is the one close attempt in progress.
K_CONVERSATION_ID = "conversation_id"
K_TASK_SID = "task_sid"
K_PARTICIPANTS = "participants"
K_TRANSCRIPTIONS = "transcriptions"
K_CALL_ENDED_AT = "call_ended_at"
K_CLOSE_CLAIMED_AT = "close_claimed_at"
K_CONVERSATION_CLOSED_AT = "conversation_closed_at"

#: A close claim older than this is taken to belong to a process that died mid-close, and may be
#: claimed again (a live attempt takes a few requests of at most a few seconds each).
CLOSE_CLAIM_TTL_S = 60

#: Reservation statuses that mean the worker joined the call (TaskRouter: ``accepted``, then
#: ``wrapping`` and ``completed``); ``pending``, ``rejected``, ``timeout``, ``canceled`` and
#: ``rescinded`` never did.
JOINED_RESERVATION_STATUSES = frozenset({"accepted", "wrapping", "completed"})


class _ReservationLike(Protocol):
    @property
    def worker_sid(self) -> str: ...

    @property
    def status(self) -> str: ...


class _StateLike(Protocol):
    @property
    def escalation(self) -> Mapping[str, Any] | None: ...


def transcription_end_event(raw: Any) -> str | None:
    """:data:`TRANSCRIPTION_STOPPED` or :data:`TRANSCRIPTION_ERROR` for Twilio's
    ``TranscriptionEvent``, else ``None`` (nothing to do)."""
    return _TRANSCRIPTION_EVENTS.get(raw.strip().lower()) if isinstance(raw, str) else None


def conversation_closed(state: _StateLike | Mapping[str, Any] | None) -> bool:
    """Whether this feature has closed the call's conversation: a call state, or its
    ``escalation`` map. Every writer of the conversation checks it first and writes nothing if so."""
    escalation = state if isinstance(state, Mapping) or state is None else state.escalation
    return isinstance(escalation, Mapping) and escalation.get(K_CONVERSATION_CLOSED_AT) is not None


def transcriptions(escalation: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    """The recorded transcriptions, leg call SID to ``{name, role, started_at, ended_at}``, in
    the order they were claimed; malformed entries are left out."""
    raw = escalation.get(K_TRANSCRIPTIONS) if isinstance(escalation, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    return {leg: entry for leg, entry in raw.items()
            if isinstance(leg, str) and leg and isinstance(entry, Mapping)
            and isinstance(entry.get("name"), str) and entry.get("name")}


def running_legs(escalation: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    """``(leg call SID, transcription name)`` of every recorded leg whose end was not reported."""
    return tuple((leg, str(entry["name"])) for leg, entry in transcriptions(escalation).items()
                 if entry.get("ended_at") is None)


def transcription_running(escalation: Mapping[str, Any] | None) -> bool:
    """Some leg's transcription was started (claimed) and no end of it was reported."""
    return bool(running_legs(escalation))


def customer_leg(escalation: Mapping[str, Any] | None) -> str | None:
    """The call SID of the recorded customer-leg transcription, if any."""
    return next((leg for leg, entry in transcriptions(escalation).items()
                 if entry.get("role") == ROLE_CUSTOMER), None)


def _mark_call_ended(new: dict[str, Any], now: int) -> None:
    if new.get(K_CALL_ENDED_AT) is None:
        new[K_CALL_ENDED_AT] = now
        new.pop(K_SWEEP_NEXT_AT, None)  # An end always brings the sweep's next look forward
        new.pop(K_SWEEP_RETRIES, None)  # and back into the index with fresh close attempts


def _with_leg(escalation: Mapping[str, Any], leg: str, entry: Mapping[str, Any] | None) -> dict[str, Any]:
    """``escalation`` with the leg's entry replaced (``None``: removed), as a fresh map."""
    legs = {k: dict(v) for k, v in transcriptions(escalation).items()}
    if entry is None:
        legs.pop(leg, None)
    else:
        legs[leg] = dict(entry)
    new = dict(escalation)
    if legs:
        new[K_TRANSCRIPTIONS] = legs
    else:
        new.pop(K_TRANSCRIPTIONS, None)
    return new


@dataclass(frozen=True, slots=True)
class EndDecision:
    """What one end signal does to the call (:func:`end_decision`).

    ``escalation`` is the new ``escalation`` map to write, or ``None`` for no write. ``close``:
    this caller now holds the close claim and runs :func:`close_plan`. ``reason`` names the
    outcome for the log line."""

    escalation: Mapping[str, Any] | None
    close: bool
    reason: str


def end_decision(escalation: Mapping[str, Any] | None, event: str, now: int, *,
                 leg: str | None = None) -> EndDecision:
    """The close rule of as amended, pure: what the end signal ``event``
    (:data:`END_EVENTS`) does. The conversation closes when the customer call has ended **and** no
    recorded leg's transcription is still running.

    * A call that never escalated (``escalation`` empty): nothing (AI-only calls are unchanged). Already
    closed: nothing (``after_close``).
    * :data:`CALL_ENDED` records ``call_ended_at`` (and drops the sweep's ``sweep_next_at``); it
      closes only when no leg is running (switch off, starts failed, or every end already reported),
      else the close waits for the transcripts to end (``waiting_for_transcript``).
    * :data:`TRANSCRIPTION_STOPPED` and :data:`TRANSCRIPTION_ERROR` end ``leg`` only (its
      ``ended_at``); ``leg=None`` means the customer leg. A leg not recorded: nothing
      (``unknown_leg``). The customer leg's ``transcription-stopped`` also means the call ended
      (Twilio sends it when the call ends), so it records ``call_ended_at``; an error does not (the
      transcript is over, the call may not be: ``/voice/status`` then ends it). An agent leg ending
      never closes alone (a transfer): only once the call has ended and it was the last leg running.
    * :data:`SWEEP` closes (the sweep has confirmed the call ended; its plan stops every leg).

    A close also takes the claim (``close_claimed_at``), unless a claim younger than
    :data:`CLOSE_CLAIM_TTL_S` exists (``claimed_elsewhere``): of two concurrent signals that both
    close, the one whose write lands runs the close, so the close request is sent at most once.
    """
    if event not in END_EVENTS:
        raise ValueError(f"unknown end event {event!r}")
    if not isinstance(escalation, Mapping) or not escalation:
        return EndDecision(None, False, "not_escalated")
    if conversation_closed(escalation):
        return EndDecision(None, False, "after_close")
    new = dict(escalation)
    if event == CALL_ENDED:
        _mark_call_ended(new, now)
        wants_close = not transcription_running(new)
        waiting = "waiting_for_transcript"
    elif event in (TRANSCRIPTION_STOPPED, TRANSCRIPTION_ERROR):
        target = leg if leg is not None else customer_leg(new)
        entry = transcriptions(new).get(target) if target is not None else None
        if entry is None or target is None:
            return EndDecision(None, False, "unknown_leg")
        if entry.get("ended_at") is None:
            new = _with_leg(new, target, {**entry, "ended_at": now})
        if event == TRANSCRIPTION_STOPPED and entry.get("role") == ROLE_CUSTOMER:
            _mark_call_ended(new, now)
        ended = new.get(K_CALL_ENDED_AT) is not None
        wants_close = ended and not transcription_running(new)
        waiting = "waiting_for_transcript" if ended else "waiting_for_call_end"
    else:  # SWEEP
        wants_close, waiting = True, ""
    if not wants_close:
        return EndDecision(new if new != escalation else None, False, waiting)
    claimed = new.get(K_CLOSE_CLAIMED_AT)
    if isinstance(claimed, (int, float)) and not isinstance(claimed, bool) \
            and now - claimed < CLOSE_CLAIM_TTL_S:
        return EndDecision(new if new != escalation else None, False, "claimed_elsewhere")
    new[K_CLOSE_CLAIMED_AT] = now
    return EndDecision(new, True, "close")


@dataclass(frozen=True, slots=True)
class ClosePlan:
    """The close, in order (as amended): stop every leg's transcription still running
    (``stop_transcriptions``: ``(leg call SID, name)``, posted to that leg), add the workers
    TaskRouter shows joined but call state does not list (a missed ``reservation.accepted``),
    then close. ``close`` is ``False`` when there is nothing to do."""

    stop_transcriptions: tuple[tuple[str, str], ...]
    add_workers: tuple[str, ...]
    close: bool


NOTHING_TO_CLOSE = ClosePlan((), (), False)


def recorded_workers(escalation: Mapping[str, Any] | None) -> frozenset[str]:
    """Worker SIDs in the ``participants`` list of the ``escalation`` map (writes it)."""
    listed = escalation.get(K_PARTICIPANTS) if isinstance(escalation, Mapping) else None
    if not isinstance(listed, (list, tuple)):
        return frozenset()
    return frozenset(p["worker_sid"] for p in listed
                     if isinstance(p, Mapping) and isinstance(p.get("worker_sid"), str))


def close_plan(state: _StateLike, reservations: Sequence[_ReservationLike], *,
               now: int | None = None) -> ClosePlan:
    """The close of for ``state`` and the task's ``reservations``, pure (no I/O).

    Nothing for a call that never escalated or whose conversation is already closed. Every
    recorded leg still running (its end not yet reported) is stopped by name, in the order the legs
    were claimed, so a stop precedes the close of each running transcription. The sweep's
    close runs this plan too, so a leg whose stop never arrives can't hold the conversation open
    beyond the sweep. Workers are added once
    each, in reservation order, only for a joined reservation (:data:`JOINED_RESERVATION_STATUSES`).
    With ``now``, a worker whose add a live accept delivery has claimed is left to it, so the
    repair never sends a second add for the same worker at the same time."""
    escalation = state.escalation
    if not isinstance(escalation, Mapping) or not escalation or conversation_closed(escalation):
        return NOTHING_TO_CLOSE
    stop = running_legs(escalation)
    present = set(recorded_workers(escalation))
    if now is not None:
        claims = escalation.get(K_PARTICIPANT_CLAIMS)
        if isinstance(claims, Mapping):
            present |= {w for w, at in claims.items() if isinstance(w, str) and _live_claim(at, now)}
    add: list[str] = []
    for r in reservations:
        worker = getattr(r, "worker_sid", None)
        if isinstance(worker, str) and worker and getattr(r, "status", None) in \
                JOINED_RESERVATION_STATUSES and worker not in present:
            present.add(worker)
            add.append(worker)
    return ClosePlan(stop, tuple(add), True)


# --------------------------------------------------------------------------- accepts

#: The TaskRouter event that adds a worker: the first one of an escalated call is the hand-off.
ACCEPTED_EVENT = "reservation.accepted"
#: More ``escalation`` keys: ``handoff_at`` (epoch seconds) is set by the first
#: accept, and is its claim: only the delivery whose write sets it marks the hand-off and starts the
#: transcription. ``taskrouter_events`` lists the events already handled (by ``EventSid``, so a
#: Twilio retry or replay is a no-op). ``participant_claims`` maps a worker to when a delivery
#: claimed its add, so concurrent deliveries for one worker send one add (released after it).
K_HANDOFF_AT = "handoff_at"
K_TASKROUTER_EVENTS = "taskrouter_events"
K_PARTICIPANT_CLAIMS = "participant_claims"
#: The handled events kept in call state (oldest dropped first; a call has a few accepts at most).
MAX_RECORDED_EVENTS = 50
#: A participant claim older than this belongs to a delivery that died mid-add (as the close claim).
PARTICIPANT_CLAIM_TTL_S = CLOSE_CLAIM_TTL_S
#: The conversation metadata the first accept writes: nothing else (8-key limit).
HANDOFF_TO = "human"

_WORKER_SID_RE = re.compile(r"^WK[0-9a-fA-F]{32}$")
_TASK_SID_RE = re.compile(r"^WT[0-9a-fA-F]{32}$")
_EVENT_KEY_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_MAX_IDENTITY = 256


@dataclass(frozen=True, slots=True)
class ParticipantAction:
    """What one ``reservation.accepted`` does (:func:`worker_event_to_action`, :func:`claim_accept`).

    ``first``: the call's first accept, so mark the hand-off and (switch on) start the transcription.
    ``add``: add ``worker_sid`` as a participant (not yet listed). ``identity`` is the participant's
    ``name`` (the worker's friendly name, else the SID; sent to Orchestrator only, never logged).
    ``conversation_id`` is the recorded one, else the task attributes'; ``None`` when neither knows."""

    event_key: str
    worker_sid: str
    identity: str
    task_sid: str | None
    conversation_id: str | None
    first: bool
    add: bool


def accept_event_key(event: Mapping[str, Any]) -> str | None:
    """The idempotence key of a TaskRouter event: its ``EventSid`` (``Sid`` as TaskRouter's callback
    may name it), else ``reservation.accepted:<ReservationSid>`` (a reservation is accepted once)."""
    for name in ("EventSid", "Sid"):
        raw = event.get(name)
        if isinstance(raw, str) and _EVENT_KEY_RE.match(raw.strip()):
            return raw.strip()
    reservation = event.get("ReservationSid")
    if isinstance(reservation, str) and _EVENT_KEY_RE.match(reservation.strip()):
        return f"{ACCEPTED_EVENT}:{reservation.strip()}"
    return None


def _task_conversation_id(event: Mapping[str, Any]) -> str | None:
    raw = event.get("TaskAttributes")
    if not isinstance(raw, str) or not raw or len(raw) > 64_000:
        return None
    try:
        attrs = json.loads(raw)
    except ValueError:
        return None
    found = attrs.get("conversation_id") if isinstance(attrs, dict) else None
    return found if isinstance(found, str) and _ID_RE.match(found) else None


def _seen(escalation: Mapping[str, Any], key: str) -> bool:
    listed = escalation.get(K_TASKROUTER_EVENTS)
    return isinstance(listed, (list, tuple)) and key in listed


def worker_event_to_action(event: Mapping[str, Any], state: _StateLike | None) -> ParticipantAction | None:
    """The participant action of a TaskRouter event for the call ``state``, pure; ``None`` for
    nothing to do: not a ``reservation.accepted``, no worker or no idempotence key, a call that never
    escalated, a closed conversation, or an event already handled."""
    if event.get("EventType") != ACCEPTED_EVENT:
        return None
    escalation = state.escalation if state is not None else None
    if not isinstance(escalation, Mapping) or not escalation or conversation_closed(escalation):
        return None
    worker = event.get("WorkerSid")
    key = accept_event_key(event)
    if not isinstance(worker, str) or not _WORKER_SID_RE.match(worker) or key is None \
            or _seen(escalation, key):
        return None
    task = event.get("TaskSid")
    recorded_task = escalation.get(K_TASK_SID)
    task_sid = task if isinstance(task, str) and _TASK_SID_RE.match(task) else (
        recorded_task if isinstance(recorded_task, str) and recorded_task else None)
    recorded = escalation.get(K_CONVERSATION_ID)
    conversation = recorded if isinstance(recorded, str) and recorded else _task_conversation_id(event)
    name = event.get("WorkerName")
    identity = name.strip()[:_MAX_IDENTITY] if isinstance(name, str) and name.strip() else worker
    return ParticipantAction(
        event_key=key, worker_sid=worker, identity=identity, task_sid=task_sid,
        conversation_id=conversation, first=escalation.get(K_HANDOFF_AT) is None,
        add=worker not in recorded_workers(escalation))


@dataclass(frozen=True, slots=True)
class AcceptClaim:
    """:func:`claim_accept`'s answer: ``escalation`` to write (``None``: no write) and the action this
    delivery runs, with ``first`` and ``add`` as it won them (``None``: nothing). ``reason`` for the log."""

    escalation: Mapping[str, Any] | None
    action: ParticipantAction | None
    reason: str


def _live_claim(claimed: Any, now: int, ttl_s: int = PARTICIPANT_CLAIM_TTL_S) -> bool:
    return isinstance(claimed, (int, float)) and not isinstance(claimed, bool) and now - claimed < ttl_s


def claim_accept(escalation: Mapping[str, Any] | None, action: ParticipantAction, now: int) -> AcceptClaim:
    """``action`` re-decided on the freshly read ``escalation`` (inside a conditional update, so of two
    concurrent deliveries only one wins each claim), pure.

    Records the event as handled, the task SID and the conversation id (first known kept). ``first``
    only if ``handoff_at`` is still unset (this write sets it: one hand-off and one transcription start
    per call). ``add`` only if the worker is neither listed nor claimed by a live delivery (one add
    request per worker). Nothing after close, and nothing while a close is under way
    (``close_claimed_at``, held after the call's end): its repair adds any worker not yet listed."""
    if not isinstance(escalation, Mapping) or not escalation:
        return AcceptClaim(None, None, "not_escalated")
    if conversation_closed(escalation):
        return AcceptClaim(None, None, "after_close")
    if _seen(escalation, action.event_key):
        return AcceptClaim(None, None, "duplicate")
    new = dict(escalation)
    listed = [k for k in new.get(K_TASKROUTER_EVENTS) or [] if isinstance(k, str)]
    new[K_TASKROUTER_EVENTS] = (listed + [action.event_key])[-MAX_RECORDED_EVENTS:]
    if _live_claim(new.get(K_CLOSE_CLAIMED_AT), now, CLOSE_CLAIM_TTL_S):
        # The close is under way (the call has ended): its repair owns the participant set now, so
        # this delivery sends nothing (no second add of a worker the repair is adding).
        return AcceptClaim(new, None, "closing")
    if action.task_sid and not new.get(K_TASK_SID):
        new[K_TASK_SID] = action.task_sid
    if action.conversation_id and not new.get(K_CONVERSATION_ID):
        new[K_CONVERSATION_ID] = action.conversation_id
    first = new.get(K_HANDOFF_AT) is None
    if first:
        new[K_HANDOFF_AT] = now
    add = action.worker_sid not in recorded_workers(new)
    if add:
        claims = dict(new.get(K_PARTICIPANT_CLAIMS) or {})
        if _live_claim(claims.get(action.worker_sid), now):
            add = False
        else:
            claims[action.worker_sid] = now
            new[K_PARTICIPANT_CLAIMS] = claims
    conversation = new.get(K_CONVERSATION_ID)
    won = replace(action, first=first, add=add,
                  conversation_id=conversation if isinstance(conversation, str) and conversation
                  else action.conversation_id)
    return AcceptClaim(new, won, "claimed" if first or add else "nothing_to_do")


def record_accept(escalation: Mapping[str, Any] | None, worker_sid: str, *, added: bool,
                  conversation_id: str | None, now: int) -> Mapping[str, Any] | None:
    """The ``escalation`` map after an accept's requests, pure: the worker listed once in
    ``participants`` (``{worker_sid, added_at}``) if the add succeeded, its claim released either way,
    the conversation id if newly learned. ``None`` (write nothing) once the conversation is closed
     or when nothing changes."""
    if not isinstance(escalation, Mapping) or not escalation or conversation_closed(escalation):
        return None
    new = dict(escalation)
    claims = dict(new.get(K_PARTICIPANT_CLAIMS) or {})
    if claims.pop(worker_sid, None) is not None:
        new[K_PARTICIPANT_CLAIMS] = claims
    if added and worker_sid not in recorded_workers(new):
        listed = [dict(p) for p in new.get(K_PARTICIPANTS) or [] if isinstance(p, Mapping)]
        new[K_PARTICIPANTS] = listed + [{"worker_sid": worker_sid, "added_at": now}]
    if conversation_id and not new.get(K_CONVERSATION_ID):
        new[K_CONVERSATION_ID] = conversation_id
    return new if new != dict(escalation) else None


def handoff_metadata(now: int) -> Mapping[str, str]:
    """The conversation metadata of the hand-off (data model): ``handoff_at``, ``handoff_to``."""
    return {"handoff_at": str(int(now)), "handoff_to": HANDOFF_TO}


def claim_transcription(escalation: Mapping[str, Any] | None, leg: str, *, name: str, role: str,
                        now: int) -> Mapping[str, Any] | None:
    """The ``escalation`` map with ``leg``'s transcription recorded (``{name, role, started_at,
    ended_at: None}`` under ``transcriptions``) **before** its start request, pure, or ``None``
    (do not start): not escalated or closed, the call already ended (``call_ended_at``: nothing
    left to transcribe), ``leg`` not a call SID, an unknown ``role`` or empty ``name``, the leg already
    recorded (one start per leg), a customer leg already recorded (one per call), or
    :data:`MAX_AGENT_LEGS` agent legs already recorded.

    Recording it first, in the same conditional update the close path uses for ``call_ended_at``,
    orders the two: if the call's end lands first, no start; if the claim lands first, the close waits
    for the leg's end and stops it by name, so a running transcription is never left out of
    the close."""
    if not isinstance(escalation, Mapping) or not escalation or conversation_closed(escalation) \
            or escalation.get(K_CALL_ENDED_AT) is not None or role not in _ROLES \
            or not isinstance(leg, str) or not CALL_SID_RE.match(leg) or not isinstance(name, str) or not name:
        return None
    legs = transcriptions(escalation)
    if leg in legs or sum(1 for e in legs.values() if e.get("role") == role) >= \
            (1 if role == ROLE_CUSTOMER else MAX_AGENT_LEGS):
        return None
    return _with_leg(escalation, leg, {"name": name, "role": role, "started_at": now, "ended_at": None})


def _unended_leg(escalation: Mapping[str, Any] | None, leg: str) -> Mapping[str, Any] | None:
    if not isinstance(escalation, Mapping) or conversation_closed(escalation):
        return None
    entry = transcriptions(escalation).get(leg)
    return entry if entry is not None and entry.get("ended_at") is None else None


def release_transcription(escalation: Mapping[str, Any] | None, leg: str) -> Mapping[str, Any] | None:
    """The ``escalation`` map without ``leg``'s claimed transcription after its start failed, pure, or
    ``None`` (no write): not recorded, already ended, or closed. The close then no longer waits for it."""
    if _unended_leg(escalation, leg) is None or escalation is None:
        return None
    return _with_leg(escalation, leg, None)


def end_unstarted_transcription(escalation: Mapping[str, Any] | None, leg: str,
                                now: int) -> Mapping[str, Any] | None:
    """The fallback when :func:`release_transcription` could not be written: ``leg``'s entry
    stays, with ``ended_at`` recorded, so the close no longer waits for an end that will never come.
    Pure; ``None`` (no write): not recorded, already ended, or closed."""
    entry = _unended_leg(escalation, leg)
    if entry is None or escalation is None:
        return None
    return _with_leg(escalation, leg, {**entry, "ended_at": now})


# --------------------------------------------------------------------------- stale-conversation sweep

#: Calls API ``status`` values after which the call is over; ``queued``, ``ringing`` and
#: ``in-progress`` (and anything not listed) are live or unknown, and never closed by the sweep.
ENDED_CALL_STATUSES = frozenset({"completed", "busy", "failed", "no-answer", "canceled"})
#: How long after the call's end the sweep waits before it closes ("more than 5 minutes"),
#: so the normal close path (``transcription-stopped`` or ``/voice/status``) goes first.
SWEEP_GRACE_S = 300


def _finite(value: Any) -> TypeGuard[int | float]:
    """A real number, not a bool, NaN or infinite."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and value == value and abs(value) != float("inf")


@dataclass(frozen=True, slots=True)
class CallEnd:
    """A call as the Calls API reports it: its ``status`` and, once ended, its ``end_time`` in
    epoch seconds (``None`` when the answer has none in a known shape)."""

    status: str
    ended_at: float | None


# --------------------------------------------------------------------------- the sweep from our own records

#: Calls API ``status`` values of a call still going: in queue (``queued``, ``ringing``) or connected
#: (``in-progress``). Any status in neither set is unknown, like no answer at all.
LIVE_CALL_STATUSES = frozenset({"queued", "ringing", "in-progress"})
#: ``escalation`` keys: ``at`` is when the call escalated (the front door writes ISO
#: ``%Y-%m-%dT%H:%M:%SZ``); ``sweep_next_at`` (epoch seconds) is when the sweep next looks at the call,
#: written by the sweep only and dropped when ``call_ended_at`` is recorded (:func:`end_decision`).
K_ESCALATED_AT = "at"
K_SWEEP_NEXT_AT = "sweep_next_at"
#: The first look at an escalated call with no recorded end: this long after it escalated.
SWEEP_FIRST_LOOK_S = 600
#: A live call is looked at again after half its age since escalation, within these bounds.
SWEEP_LIVE_MIN_S = 600
SWEEP_LIVE_MAX_S = 1200
#: A close that was not done (failed, no conversation, claimed elsewhere) or an unknown status is
#: tried again after this.
SWEEP_RETRY_S = 600
#: Retry and live due times are written this much early, so a run that starts a little early on the
#: 10-minute schedule still finds the call due. Not applied to the wait after an end: the grace
#: period must hold.
SWEEP_SLACK_S = 60
#: ``escalation`` key: how many consecutive futile tries (unknown status, close not done) the sweep has
#: written. At :data:`SWEEP_MAX_RETRIES` in a row the call leaves the ``sweep-due`` index for good
#: (about an hour of tries); Orchestrator's own timeout is then the backstop. A live answer or a
#: wait after an end resets it (:func:`sweep_reschedule`).
K_SWEEP_RETRIES = "sweep_retries"
SWEEP_MAX_RETRIES = 6
#: ``escalation`` key: when the sweep gave up on the call (the retry that reached
#: :data:`SWEEP_MAX_RETRIES`), epoch seconds. Written once and kept (an end's fresh retries leave it),
#: so the sweep starts the recap writer for a call it gave up on at most once.
K_SWEEP_GAVE_UP_AT = "sweep_gave_up_at"
#: Due times are kept within epoch 0 and the end of year 9999 (a valid DynamoDB number key).
_DUE_MAX = 253_402_300_799
_ESCALATED_AT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: :class:`SweepStep` outcomes.
STEP_SKIP = "skip"  # not escalated, or the conversation is closed: nothing to do (a stale index entry)
STEP_CLOSE = "close"  # ended more than the grace period ago: the close path
STEP_WAIT = "wait"  # ended within the grace period: due again at the end + grace
STEP_LIVE = "live"  # in queue or with an agent: due again after the back-off
STEP_UNKNOWN = "unknown"  # no usable status: due again after SWEEP_RETRY_S, like a close not done


def _clamp_due(value: float) -> int:
    return int(min(max(math.ceil(value), 0), _DUE_MAX))


def escalated_at(escalation: Mapping[str, Any] | None) -> float | None:
    """When the call escalated (``escalation.at``), epoch seconds; ``None`` if absent or unreadable."""
    raw = escalation.get(K_ESCALATED_AT) if isinstance(escalation, Mapping) else None
    if _finite(raw):
        return float(raw)
    if not isinstance(raw, str):
        return None
    try:
        return datetime.strptime(raw, _ESCALATED_AT_FORMAT).replace(tzinfo=UTC).timestamp()
    except ValueError:
        return None


def sweep_open(escalation: Mapping[str, Any] | None) -> bool:
    """The sweep has work on this call: it escalated and its conversation is not closed (exactly
    the call states in the ``sweep-due`` index)."""
    return isinstance(escalation, Mapping) and bool(escalation) and not conversation_closed(escalation)


def sweep_retries(escalation: Mapping[str, Any] | None) -> int:
    """The retries the sweep has written for this call (anything not a non-negative int counts as 0)."""
    raw = escalation.get(K_SWEEP_RETRIES) if isinstance(escalation, Mapping) else None
    return raw if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0 else 0


def sweep_indexed(escalation: Mapping[str, Any] | None, ai_sessions: Any) -> bool:
    """The call belongs in the ``sweep-due`` index (with), pure: escalated, conversation not
    closed, a conversation possible (an AI session ran, ``ai_sessions > 0``, or a conversation id is
    recorded: a call escalated before any AI session has none and could never be closed), and fewer
    than :data:`SWEEP_MAX_RETRIES` retries."""
    if not sweep_open(escalation) or escalation is None:
        return False
    sessions = ai_sessions if isinstance(ai_sessions, int) and not isinstance(ai_sessions, bool) else 0
    conversation = escalation.get(K_CONVERSATION_ID)
    possible = sessions > 0 or (isinstance(conversation, str) and bool(conversation))
    return possible and sweep_retries(escalation) < SWEEP_MAX_RETRIES


def sweep_due_at(escalation: Mapping[str, Any] | None) -> int:
    """When the sweep next looks at the call, epoch seconds, pure: ``sweep_next_at`` when set,
    else ``call_ended_at`` + :data:`SWEEP_GRACE_S`, else the escalation time + :data:`SWEEP_FIRST_LOOK_S`,
    else 0 (due now: the sweep checks the status anyway). Always an integer in range."""
    esc = escalation if isinstance(escalation, Mapping) else {}
    nxt = esc.get(K_SWEEP_NEXT_AT)
    if _finite(nxt):
        return _clamp_due(nxt)
    ended = esc.get(K_CALL_ENDED_AT)
    if _finite(ended):
        return _clamp_due(ended + SWEEP_GRACE_S)
    at = escalated_at(esc)
    if at is not None:
        return _clamp_due(at + SWEEP_FIRST_LOOK_S)
    return 0


def sweep_needs_status(escalation: Mapping[str, Any] | None) -> bool:
    """The sweep must ask the Calls API about this call: open, and no end recorded (``/voice/status``
    records ``call_ended_at`` for most calls, so most ended calls cost no Twilio read)."""
    return sweep_open(escalation) and not _finite((escalation or {}).get(K_CALL_ENDED_AT))


@dataclass(frozen=True, slots=True)
class SweepStep:
    """What the sweep does with one call (:func:`sweep_step`): ``outcome`` (``STEP_*``) and, for
    :data:`STEP_WAIT` and :data:`STEP_LIVE`, the new ``sweep_next_at`` (else ``None``)."""

    outcome: str
    next_at: int | None = None


def _live_back_off(escalation: Mapping[str, Any], now: float) -> float:
    at = escalated_at(escalation)
    age = max(now - at, 0.0) if at is not None else 0.0
    return min(max(age / 2, SWEEP_LIVE_MIN_S), SWEEP_LIVE_MAX_S)


def sweep_step(escalation: Mapping[str, Any] | None, call_end: CallEnd | None, now: float, *,
               grace_s: float = SWEEP_GRACE_S) -> SweepStep:
    """One call's sweep decision, pure. ``call_end`` is the Calls API's answer (``None``
    when not asked or no answer); a recorded ``call_ended_at`` wins over it.

    * Not escalated or closed: :data:`STEP_SKIP` (nothing is ever written after the close).
    * Ended (recorded, or reported with an end time) more than ``grace_s`` ago: :data:`STEP_CLOSE`.
    * Ended within ``grace_s``: :data:`STEP_WAIT`, due again just after the end + ``grace_s``.
    * Reported live (:data:`LIVE_CALL_STATUSES`): :data:`STEP_LIVE`, due again after half the call's
      age since escalation, between :data:`SWEEP_LIVE_MIN_S` and :data:`SWEEP_LIVE_MAX_S`, less
      :data:`SWEEP_SLACK_S`.
    * Anything else (no answer, an unknown status, ended with no end time): :data:`STEP_UNKNOWN`, due
      again :data:`SWEEP_RETRY_S` less :data:`SWEEP_SLACK_S` after ``now`` (written through
      :func:`sweep_retry`, which counts the retry), so a call that stays unknown never holds the
      front of the queue.
    A live or unknown call is never closed, however long it has waited."""
    if escalation is None or not sweep_open(escalation) or not _finite(now) or not _finite(grace_s) \
            or grace_s < 0:
        return SweepStep(STEP_SKIP)
    recorded = escalation.get(K_CALL_ENDED_AT)
    end: float | None = float(recorded) if _finite(recorded) else None
    reported = call_end if isinstance(call_end, CallEnd) else None
    status = reported.status.strip().lower() if reported is not None and isinstance(reported.status, str) \
        else None
    if end is None and reported is not None and status in ENDED_CALL_STATUSES and _finite(reported.ended_at):
        end = float(reported.ended_at)
    if end is not None:
        if end < now - grace_s:
            return SweepStep(STEP_CLOSE)
        return SweepStep(STEP_WAIT, _clamp_due(math.floor(end) + grace_s + 1))
    if status in LIVE_CALL_STATUSES:
        return SweepStep(STEP_LIVE, _clamp_due(now + _live_back_off(escalation, now) - SWEEP_SLACK_S))
    return SweepStep(STEP_UNKNOWN, _clamp_due(now + SWEEP_RETRY_S - SWEEP_SLACK_S))


def sweep_reschedule(escalation: Mapping[str, Any] | None, call_end: CallEnd | None,
                     now: float) -> Mapping[str, Any] | None:
    """The ``escalation`` map with the sweep's new ``sweep_next_at`` (:func:`sweep_step` re-decided on
    this freshly read map), pure. A live answer or a wait after an end is a useful try, so it also
    resets :data:`K_SWEEP_RETRIES` (the count is of consecutive futile tries only: a call alternating
    between a flaky Calls API and live never runs out of retries while the caller waits). ``None``
    (write nothing) once closed, when the step is not :data:`STEP_WAIT` or :data:`STEP_LIVE` (an end
    recorded meanwhile leaves the call due), or when nothing changes."""
    step = sweep_step(escalation, call_end, now)
    if step.outcome not in (STEP_WAIT, STEP_LIVE) or step.next_at is None or escalation is None:
        return None
    new = {**escalation, K_SWEEP_NEXT_AT: step.next_at}
    new.pop(K_SWEEP_RETRIES, None)
    return None if new == dict(escalation) else new


_UNSEEN: Any = object()


def sweep_retry(escalation: Mapping[str, Any] | None, now: float, *,
                seen_ended_at: Any = _UNSEEN) -> Mapping[str, Any] | None:
    """The ``escalation`` map due again :data:`SWEEP_RETRY_S` less :data:`SWEEP_SLACK_S` after ``now``,
    with :data:`K_SWEEP_RETRIES` one higher (and :data:`K_SWEEP_GAVE_UP_AT` the first time it reaches
    :data:`SWEEP_MAX_RETRIES`), after a close that was not done or an unknown status, pure; ``None`` (write
    nothing) once closed or not escalated, or when ``seen_ended_at`` (the
    ``call_ended_at`` the sweep decided on) differs from the fresh map's: an end recorded meanwhile
    left the call due, and the next run decides on it without a false retry."""
    if not sweep_open(escalation) or not _finite(now) or escalation is None:
        return None
    if seen_ended_at is not _UNSEEN and escalation.get(K_CALL_ENDED_AT) != seen_ended_at:
        return None
    retries = sweep_retries(escalation) + 1
    new = {**escalation, K_SWEEP_NEXT_AT: _clamp_due(now + SWEEP_RETRY_S - SWEEP_SLACK_S),
           K_SWEEP_RETRIES: retries}
    if retries >= SWEEP_MAX_RETRIES and K_SWEEP_GAVE_UP_AT not in escalation:
        new[K_SWEEP_GAVE_UP_AT] = _clamp_due(now)  # the first give-up only (sweep_gave_up)
    return new


def sweep_gave_up(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None) -> bool:
    """The write from ``before`` to ``after`` is the sweep giving up on the call for the first time
    (it set :data:`K_SWEEP_GAVE_UP_AT`), pure: at most once per call, however often it runs out of
    retries."""
    return isinstance(after, Mapping) and K_SWEEP_GAVE_UP_AT in after \
        and not (isinstance(before, Mapping) and K_SWEEP_GAVE_UP_AT in before)
