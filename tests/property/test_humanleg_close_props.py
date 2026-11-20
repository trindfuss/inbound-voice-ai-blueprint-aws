"""Idempotent lifecycle and nothing after close, for the close path.

Over any interleaving of ``/voice/status`` end callbacks, transcription ``stopped``/``error`` callbacks,
sweep runs and ``reservation.accepted`` events (delivered, duplicated or missed), any time between them,
two end signals handled at once (the second while the first one's close request is in flight), and
failing closes injected, the real :class:`~owlbank.humanleg_close.HumanLegCloser` over a real call-state store:

* the conversation is closed at most once, and ``conversation_closed_at`` is set exactly when it was;
* no request of any kind (participant add, hand-off metadata, transcription start or stop, close) is
  issued for the conversation after the close;
* a stop of the transcription precedes the close whenever a transcription was started and Twilio had
  not reported its end ("stop the named transcription first if it is still running");
* every worker TaskRouter shows as joined is a participant when the close is sent (missed accepts are
  repaired), whenever the task is known and TaskRouter answers;
* with healthy dependencies, a call end followed by the transcript's end always leaves it closed;
* the journal holds one ``conversation_closed`` event exactly when the conversation was closed, with ``closed_by`` naming the signal that closed it.

The ``reservation.accepted`` handler is a separate unit; here a model of it stands in (it checks
``conversation_closed`` first, as every writer must), so the properties cover the close path's
interaction with it, not the handler itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank import callstate, humanleg
from owlbank.callstate import CallState
from owlbank.humanleg_client import Reservation
from owlbank.humanleg_close import HumanLegCloser
from owlbank.segments import SegmentLog
from owlbank.store import MemoryStore

CALL = "CA" + "e" * 32
CONV = "conv_" + "a" * 26
TASK = "WT" + "d" * 32
WORKERS = tuple("WK" + c * 32 for c in "abc")
T0 = 1_760_000_000


@dataclass
class World:
    """Twilio as the close path sees it: Orchestrator, Calls, TaskRouter, plus a log of every request."""

    transcription_on: bool
    close_results: list[str]
    taskrouter_down: bool
    requests: list[tuple[Any, ...]] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    joined: list[str] = field(default_factory=list)
    closed: bool = False
    successes: int = 0
    after_close: list[tuple[Any, ...]] = field(default_factory=list)
    # Model facts at the moment of each close request.
    close_snapshots: list[dict[str, Any]] = field(default_factory=list)
    started: bool = False
    end_reported: bool = False
    task_known: bool = False
    now: int = T0
    #: An end signal another Lambda handles while this close request is in flight (concurrency).
    inject: tuple[Any, ...] | None = None
    trigger: str = ""
    closer: Any = None
    store: Any = None

    def request(self, *op: Any) -> None:
        if self.closed:
            self.after_close.append(op)
        self.requests.append(op)

    # -- Orchestrator (HumanLegClient)
    def resolve_conversation(self, call_sid: str) -> str | None:
        return CONV

    def add_human_participant(self, conversation_id: str, worker_sid: str, identity: str) -> bool:
        self.request("add", worker_sid)
        if worker_sid in self.participants:
            return False
        self.participants.append(worker_sid)
        return True

    def mark_handoff(self, conversation_id: str, metadata: Any) -> bool:
        self.request("mark")
        return True

    def close(self, conversation_id: str) -> bool:
        self.close_snapshots.append({"started": self.started, "end_reported": self.end_reported,
                                     "task_known": self.task_known, "joined": list(self.joined),
                                     "participants": list(self.participants),
                                     "index": len(self.requests), "trigger": self.trigger})
        self.request("close")
        if self.inject is not None:  # a concurrent signal lands while the PUT is in flight
            nested, self.inject = self.inject, None
            run(self, self.store, self.closer, nested, gap=0)
        result = self.close_results.pop(0) if self.close_results else "ok"
        if result == "raise":
            raise RuntimeError("orchestrator down")
        if result == "false":
            return False
        self.closed = True
        self.successes += 1
        return True

    # -- Calls
    def stop_transcription(self, call_sid: str, name: str) -> bool:
        self.request("stop", name)
        return True

    # -- TaskRouter (ReservationLister)
    def list(self, task_sid: str) -> list[Reservation] | None:
        if self.taskrouter_down:
            return None
        return [Reservation(w, "accepted") for w in self.joined]


def accept(world: World, store: MemoryStore, worker: str, delivered: bool) -> None:
    """A model of the ``reservation.accepted`` handler: TaskRouter always knows the reservation;
    the front door only if the event was delivered. It writes nothing once the conversation is
    closed, marks the hand-off and starts the transcription on the first accept only."""
    if worker not in world.joined:
        world.joined.append(worker)
    if not delivered:
        return
    state = callstate.load(store, CALL)
    assert state is not None
    if humanleg.conversation_closed(state):
        return
    esc = dict(state.escalation or {})
    if not esc.get(humanleg.K_PARTICIPANTS):
        world.mark_handoff(CONV, {"handoff_to": "human"})
    world.add_human_participant(CONV, worker, worker)
    claimed = humanleg.claim_transcription(esc, CALL, name=humanleg.TRANSCRIPTION_NAME,
                                           role=humanleg.ROLE_CUSTOMER, now=world.now) \
        if world.transcription_on else None
    if claimed is not None:
        world.request("start")
        world.started = True
        esc = dict(claimed)
    listed = [dict(p) for p in esc.get(humanleg.K_PARTICIPANTS) or []]
    if worker not in {p["worker_sid"] for p in listed}:
        listed.append({"worker_sid": worker, "added_at": world.now})
    esc[humanleg.K_PARTICIPANTS] = listed
    esc[humanleg.K_TASK_SID] = TASK
    world.task_known = True
    callstate.mutate(store, CALL, lambda s: {"escalation": esc})


END_SIGNALS = st.sampled_from([("status",), ("stopped",), ("error",), ("sweep",)])
EVENTS = st.one_of(
    END_SIGNALS,
    st.tuples(st.just("accept"), st.sampled_from(WORKERS), st.booleans()),
    # Two end signals at once: the second is handled while the first one's close is in flight.
    st.tuples(st.just("concurrent"), END_SIGNALS, END_SIGNALS),
)
#: Seconds between events: past the close claim's lifetime too (a claim never outlives its close).
GAPS = st.integers(min_value=0, max_value=3 * humanleg.CLOSE_CLAIM_TTL_S)


def run(world: World, store: MemoryStore, closer: HumanLegCloser, event: tuple[Any, ...], *,
        gap: int = 1) -> None:
    world.now += gap
    kind = event[0]
    if kind == "concurrent":
        world.inject = event[2]
        run(world, store, closer, event[1], gap=0)
        world.inject = None
        return
    if kind == "accept":
        accept(world, store, event[1], event[2])
        return
    outer, world.trigger = world.trigger, kind
    if kind == "status":
        closer.close(call_sid=CALL, conversation_id=None)
    elif kind in ("stopped", "error"):
        state = callstate.load(store, CALL)
        if state is not None and not humanleg.conversation_closed(state):
            world.end_reported = True  # Twilio reports the transcript's end
        closer.transcription_ended(call_sid=CALL, conversation_id=None,
                                   event=humanleg.TRANSCRIPTION_STOPPED if kind == "stopped"
                                   else humanleg.TRANSCRIPTION_ERROR)
    else:
        closer.end(CALL, humanleg.SWEEP)
    world.trigger = outer


def setup(transcription_on: bool, close_results: list[str], taskrouter_down: bool
          ) -> tuple[World, MemoryStore, HumanLegCloser]:
    world = World(transcription_on, list(close_results), taskrouter_down)
    store = MemoryStore()
    callstate.create(store, CallState(call_sid=CALL, expires_at=T0 + 86_400, ai_sessions=1,
                                      escalation={"reason": "fraud_suspected", "queue": "q",
                                                  "journal_ref": "jr"}))
    closer = HumanLegCloser(store, world, calls=world, reservations=world, clock=lambda: world.now,
                            journal=SegmentLog(store, now=lambda: world.now))
    world.closer, world.store = closer, store
    return world, store, closer


def check(world: World, store: MemoryStore) -> None:
    state = callstate.load(store, CALL)
    assert state is not None
    # Closed at most once; the state says closed exactly when Orchestrator was closed.
    assert world.successes <= 1
    assert humanleg.conversation_closed(state) == world.closed
    # One journal event, written by the close that succeeded.
    events = [e for e in SegmentLog(store).read(CALL) if e.get("type") == "conversation_closed"]
    assert len(events) == (1 if world.closed else 0)
    assert all(e["closed_by"] in {"status", "transcription", "sweep"} and e["conversation_id"] == CONV
               for e in events)
    # Nothing after the close.
    assert world.after_close == [], world.after_close
    # One participant per worker.
    assert len(world.participants) == len(set(world.participants))
    for snap in world.close_snapshots:
        # While the transcript runs, only the sweep (call confirmed ended) closes, never
        # the status callback alone: a running transcription is stopped before the close.
        if snap["started"] and not snap["end_reported"]:
            assert snap["trigger"] == "sweep", snap
            before = world.requests[: snap["index"]]
            last_start = max(i for i, r in enumerate(before) if r[0] == "start")
            assert any(r[0] == "stop" for r in before[last_start + 1:]), before
        # Missed accepts repaired before the close, when the task is known and TaskRouter answers.
        if snap["task_known"] and not world.taskrouter_down:
            assert set(snap["joined"]) <= set(snap["participants"]), snap


@settings(max_examples=300, deadline=None)
@given(events=st.lists(st.tuples(EVENTS, GAPS), max_size=14), transcription_on=st.booleans(),
       close_results=st.lists(st.sampled_from(["ok", "false", "raise"]), max_size=4),
       taskrouter_down=st.booleans())
def test_close_at_most_once_stop_first_and_nothing_after(events, transcription_on, close_results,
                                                         taskrouter_down) -> None:
    world, store, closer = setup(transcription_on, close_results, taskrouter_down)
    for event, gap in events:
        run(world, store, closer, event, gap=gap)
        check(world, store)


@settings(max_examples=200, deadline=None)
@given(events=st.lists(st.tuples(EVENTS, GAPS), max_size=12), transcription_on=st.booleans(),
       close_results=st.lists(st.sampled_from(["ok", "false", "raise"]), max_size=4),
       taskrouter_down=st.booleans())
def test_call_end_then_transcript_end_always_closes_once_healthy(events, transcription_on, close_results,
                                                                 taskrouter_down) -> None:
    world, store, closer = setup(transcription_on, close_results, taskrouter_down)
    for event, gap in events:
        run(world, store, closer, event, gap=gap)
    world.close_results.clear()  # dependencies healthy from here on
    for event in (("status",), ("stopped",)):
        run(world, store, closer, event)
        check(world, store)
    # Exactly one successful close, however many attempts failed before it.
    assert world.closed and world.successes == 1
    assert humanleg.conversation_closed(callstate.load(store, CALL))
