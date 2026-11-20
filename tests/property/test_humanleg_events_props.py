"""Idempotent lifecycle, switch semantics, the state machine and the close
waiting for every leg, for the participants and per-leg transcriptions from TaskRouter events.

Over any sequence of ``reservation.accepted`` deliveries (duplicated, reordered and replayed, the same
worker on several reservations, some already naming agent legs) and ``task.updated`` deliveries (agent
legs added, repeated and dropped, more than the cap), mixed with ``/voice/status`` call ends, the
customer leg's and each agent leg's ``stopped``/``error`` callbacks through the real
``/voice/transcription`` handler (and callbacks naming a call that doesn't record the leg), with one
delivery handled while another one's request is in flight (concurrent Lambdas), failing adds and starts
injected, both switches drawn, the real front door's TaskRouter handler and the real
:class:`~owlbank.humanleg_close.HumanLegCloser`, over one call-state store:

* Once a worker's add has succeeded, no further add request is issued for that worker, and never
  two at once; call state lists each worker at most once, and only workers Orchestrator has;
* With ``human_leg.transcription`` off, no transcription start; with it on, never a start on a leg
  whose transcription runs (a failed start is released and may be retried), starts only on the
  customer leg and agent legs Flex named, at most :data:`~owlbank.humanleg.MAX_AGENT_LEGS` agent legs,
  none after the call ended, and the customer leg started once when the first accept is handled
  before the call ends with healthy starts; with ``human_leg.agent_legs`` off, the customer leg only,
  ``both_tracks``;
* The conversation is only ever set ACTIVE (never INACTIVE), is closed at most once, never before
  the call has ended; no request of any kind (hand-off metadata, participant, transcription) is issued
  after the close, and the call state's ``escalation`` map does not change after it; ``handoff_at``
  is written once;
* The close happens only once every started leg has ended or been stopped (an agent leg ending
  while the call goes on never closes); a callback naming a call that doesn't record its leg changes
  nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from hypothesis import example, given, settings
from hypothesis import strategies as st

from lambdas.frontdoor import http
from owlbank import callstate, humanleg
from owlbank.callstate import CallState
from owlbank.humanleg_client import Reservation
from owlbank.humanleg_close import HumanLegCloser
from owlbank.policy import Policy
from tests.component.test_frontdoor_support import DATA, HOST, STAGE, build

CALL = "CA" + "e" * 32
CONV = "conv_" + "a" * 26
TASK = "WT" + "d" * 32
OTHER = "CA" + "f" * 32  # a call SID with no state: a forged ``call`` on a leg's callback
W_A, W_B, W_C = ("WK" + c * 32 for c in "abc")
#: Agent legs Flex may write into the task attributes: more than the cap.
AGENTS = tuple("CA" + f"{i:032x}" for i in range(1, 9))
#: The pool of distinct TaskRouter events: (EventSid, worker). The same worker on two reservations
#: (transferred back) is two events.
POOL = (("EA" + "1" * 32, W_A), ("EA" + "2" * 32, W_B), ("EA" + "3" * 32, W_A),
        ("EA" + "4" * 32, W_C), ("EA" + "5" * 32, W_B))
T0 = 1_760_000_000


@dataclass
class World:
    """Orchestrator, the Calls API and TaskRouter as the handler and the closer see them."""

    add_results: list[bool]
    start_results: list[bool]
    close_results: list[bool] = field(default_factory=list)
    close_failed: bool = False
    now: int = T0
    requests: list[tuple[Any, ...]] = field(default_factory=list)
    statuses: list[str] = field(default_factory=lambda: ["ACTIVE"])
    present: list[str] = field(default_factory=list)
    added_ok: set[str] = field(default_factory=set)
    violations: list[str] = field(default_factory=list)
    delivered: list[str] = field(default_factory=list)
    call_ended: bool = False
    #: Legs whose transcription Twilio is running (started, not ended or stopped).
    running: set[str] = field(default_factory=set)
    #: Legs whose start succeeded at some point.
    transcribed: set[str] = field(default_factory=set)
    closes: int = 0
    #: Every start request: (leg, Track).
    starts: list[tuple[str, str]] = field(default_factory=list)
    first_clean: bool = False
    agent_legs: bool = True
    start_ok_always: bool = False
    #: Workers whose add request is issued and not yet answered.
    adding: set[str] = field(default_factory=set)
    #: The delivery or signal another Lambda handles while this request is in flight.
    inject: tuple[Any, ...] | None = None
    #: Which request it lands on: ``mark``, ``add``, ``start``, ``close`` or ``any`` (the first one).
    inject_at: str = "any"
    app: Any = None
    closer: Any = None

    @property
    def closed(self) -> bool:
        return self.statuses[-1] == "CLOSED"

    def request(self, *op: Any) -> None:
        if self.closed:
            self.violations.append(f"{op[0]} after close")
        self.requests.append(op)
        if self.inject is not None and self.inject_at in ("any", op[0]):
            # Issued, not yet landed: the other delivery runs now.
            nested, self.inject = self.inject, None
            run(self, nested)

    # -- Orchestrator
    def resolve_conversation(self, call_sid: str) -> str | None:
        return CONV

    def mark_handoff(self, conversation_id: str, metadata: Any) -> bool:
        self.request("mark")
        if self.closed:
            return False
        self.statuses.append("ACTIVE")
        return True

    def add_human_participant(self, conversation_id: str, worker_sid: str, identity: str) -> bool:
        if worker_sid in self.added_ok:
            self.violations.append(f"add of {worker_sid[:4]} after it succeeded")
        if worker_sid in self.adding:
            self.violations.append(f"two adds of {worker_sid[:4]} at once")
        self.adding.add(worker_sid)
        try:
            self.request("add", worker_sid)
        finally:
            self.adding.discard(worker_sid)
        ok = self.add_results.pop(0) if self.add_results else True
        if not ok or worker_sid in self.present or self.closed:
            return False
        self.present.append(worker_sid)
        self.added_ok.add(worker_sid)
        return True

    def close(self, conversation_id: str) -> bool:
        if not self.call_ended:
            self.violations.append("closed before the call ended")
        if self.running:
            self.violations.append("closed while a leg's transcription runs")
        self.request("close")
        if not (self.close_results.pop(0) if self.close_results else True):
            self.close_failed = True
            return False
        if not self.closed:
            self.statuses.append("CLOSED")
            self.closes += 1
        return True

    # -- Calls
    def start_transcription(self, call_sid: str, params: Any) -> bool:
        if self.call_ended:
            self.violations.append("start after the call ended")
        if call_sid in self.running:
            self.violations.append(f"second start on running leg {call_sid[-4:]}")
        if call_sid != CALL and (call_sid not in AGENTS or not self.agent_legs):
            self.violations.append(f"start on leg {call_sid[-4:]}")
        self.starts.append((call_sid, params.get("Track")))
        self.request("start")
        ok = self.start_results.pop(0) if self.start_results else True
        if ok and not self.call_ended:
            self.running.add(call_sid)
            self.transcribed.add(call_sid)
            return True
        return False

    def stop_transcription(self, call_sid: str, name: str) -> bool:
        self.request("stop")
        self.running.discard(call_sid)
        return True

    # -- TaskRouter reservations (the close's repair)
    def list(self, task_sid: str) -> list[Reservation] | None:
        return [Reservation(w, "accepted") for w in dict.fromkeys(self.delivered)]


def attributes(legs: tuple[int, ...] = ()) -> str:
    attrs: dict[str, Any] = {"call_sid": CALL, "conversation_id": CONV}
    if legs:
        participants = {"customer": CALL, **{f"worker{n}": AGENTS[i] for n, i in enumerate(legs)}}
        attrs["conference"] = {"participants": participants}
    return json.dumps(attrs)


def accept_request(index: int, legs: tuple[int, ...] = ()) -> http.TwilioRequest:
    event_sid, worker = POOL[index]
    params = {"EventType": "reservation.accepted", "EventSid": event_sid, "WorkerSid": worker,
              "WorkerName": f"Agent {index}", "TaskSid": TASK, "ReservationSid": "WR" + event_sid[2:],
              "TaskAttributes": attributes(legs)}
    return http.TwilioRequest(route=http.TASKROUTER_ROUTE, params=params, base_path=STAGE,
                              origin=f"https://{HOST}")


def updated_request(legs: tuple[int, ...]) -> http.TwilioRequest:
    params = {"EventType": "task.updated", "EventSid": "EU" + "0" * 32, "TaskSid": TASK,
              "TaskAttributes": attributes(legs)}
    return http.TwilioRequest(route=http.TASKROUTER_ROUTE, params=params, base_path=STAGE,
                              origin=f"https://{HOST}")


def leg_callback(world: World, leg: str, event: str, call: str | None) -> None:
    """Twilio's status callback for ``leg``'s transcription, through the front door's handler."""
    query = {} if call is None else {humanleg.CALL_QUERY_PARAM: call}
    req = http.TwilioRequest(route=http.TRANSCRIPTION_ROUTE, params={"CallSid": leg}, base_path=STAGE,
                             query=query, origin=f"https://{HOST}")
    world.app.transcription(req, event)


def run(world: World, event: tuple[Any, ...]) -> None:
    kind = event[0]
    if kind == "concurrent":
        world.inject, world.inject_at = event[2], event[3]
        run(world, event[1])
        world.inject = None
        return
    world.now += 1
    if kind == "accept":
        world.delivered.append(POOL[event[1]][1])
        ended_before = world.call_ended
        marks = sum(1 for r in world.requests if r[0] == "mark")
        world.app.taskrouter(accept_request(event[1], event[2] if len(event) > 2 else ()))
        if not ended_before and not world.call_ended and \
                sum(1 for r in world.requests if r[0] == "mark") > marks:
            world.first_clean = True  # the first accept, handled wholly before the call's end
    elif kind == "updated":
        world.app.taskrouter(updated_request(event[1]))
    elif kind == "status":
        world.call_ended = True
        world.closer.close(call_sid=CALL, conversation_id=None)
    elif kind == "stopped":
        # Twilio reports the end of the customer leg's running transcription when the call ends.
        if CALL in world.running and world.call_ended:
            world.running.discard(CALL)
            leg_callback(world, CALL, humanleg.TRANSCRIPTION_STOPPED, None)
    elif kind == "error" and CALL in world.running:
        world.running.discard(CALL)
        leg_callback(world, CALL, humanleg.TRANSCRIPTION_ERROR, None)
    elif kind == "leg_end" and AGENTS[event[1]] in world.running:
        # An agent leg leaves (a transfer, or the conference ending): its transcript ends.
        world.running.discard(AGENTS[event[1]])
        leg_callback(world, AGENTS[event[1]], event[2], CALL)
    elif kind == "forged":
        before, sent = callstate.load(world.app.d.store, CALL), len(world.requests)
        leg_callback(world, AGENTS[event[1]], humanleg.TRANSCRIPTION_STOPPED, OTHER)
        leg_callback(world, AGENTS[event[1]], humanleg.TRANSCRIPTION_STOPPED, None)  # its own SID
        if callstate.load(world.app.d.store, CALL) != before or len(world.requests) != sent:
            world.violations.append("a callback for a call without the leg changed something")


LEG_LISTS = st.lists(st.integers(0, len(AGENTS) - 1), max_size=len(AGENTS), unique=True).map(tuple)
SINGLE = st.one_of(
    st.tuples(st.just("accept"), st.integers(0, len(POOL) - 1)),
    st.tuples(st.just("accept"), st.integers(0, len(POOL) - 1), LEG_LISTS),
    st.tuples(st.just("updated"), LEG_LISTS),
    st.tuples(st.just("leg_end"), st.integers(0, len(AGENTS) - 1),
              st.sampled_from([humanleg.TRANSCRIPTION_STOPPED, humanleg.TRANSCRIPTION_ERROR])),
    st.tuples(st.just("forged"), st.integers(0, len(AGENTS) - 1)),
    st.sampled_from([("status",), ("stopped",), ("error",)]),
)
CONCURRENT = st.tuples(st.just("concurrent"), SINGLE, SINGLE,
                       st.sampled_from(["any", "mark", "add", "start", "close"]))
EVENTS = st.one_of(SINGLE, CONCURRENT, CONCURRENT)  # concurrency drawn twice as often


def setup(switch: bool, add_results: list[bool], start_results: list[bool],
          close_results: list[bool] | None = None, *, agent_legs: bool = True) -> tuple[World, Any]:
    world = World(list(add_results), list(start_results), list(close_results or []), agent_legs=agent_legs)
    h = build()
    doc = json.loads((DATA / "policy.json").read_text())
    doc["human_leg"] = {"transcription": switch, "agent_legs": agent_legs}
    callstate.create(h.store, CallState(call_sid=CALL, expires_at=T0 + 86_400, ai_sessions=1,
                                        language="en-US",
                                        escalation={"reason": "fraud_suspected", "queue": "q",
                                                    "journal_ref": "jr"}))
    closer = HumanLegCloser(h.store, world, calls=world, reservations=world, clock=lambda: world.now)
    h.app.d = replace(h.app.d, policy=Policy.from_dict(doc), human_leg=world, transcriptions=world,
                      closer=closer, clock=lambda: world.now)
    world.app, world.closer = h.app, closer
    return world, h


def check(world: World, h: Any, switch: bool, history: list[dict[str, Any]]) -> None:
    assert world.violations == []
    # Each worker once in call state, and only workers Orchestrator has.
    state = callstate.load(h.store, CALL)
    assert state is not None and state.escalation is not None
    listed = [p["worker_sid"] for p in state.escalation.get(humanleg.K_PARTICIPANTS) or []]
    assert len(listed) == len(set(listed))
    assert set(listed) <= set(world.present)
    assert len(world.present) == len(set(world.present))
    # (The per-leg checks are in World.start_transcription.)
    if not switch:
        assert world.starts == []
    else:
        # The cap counts transcribed legs: a failed start is released and frees its place.
        assert len({leg for leg in world.transcribed if leg != CALL}) <= humanleg.MAX_AGENT_LEGS
        tracks = {track for leg, track in world.starts if leg == CALL}
        assert tracks <= {"inbound_track" if world.agent_legs else "both_tracks"}
        assert all(track == "inbound_track" for leg, track in world.starts if leg != CALL)
        if world.first_clean and world.start_ok_always:
            assert sum(1 for leg, _ in world.starts if leg == CALL) == 1
        recorded = humanleg.transcriptions(state.escalation)
        assert sum(1 for e in recorded.values() if e.get("role") == humanleg.ROLE_AGENT) <= \
            humanleg.MAX_AGENT_LEGS
    # ACTIVE only, then CLOSED at most once, and nothing changes in call state after it.
    assert set(world.statuses) <= {"ACTIVE", "CLOSED"} and world.closes <= 1
    assert "CLOSED" not in world.statuses[:-1]
    handoffs = {e.get(humanleg.K_HANDOFF_AT) for e in history if e.get(humanleg.K_HANDOFF_AT) is not None}
    assert len(handoffs) <= 1
    closed_at = [i for i, e in enumerate(history) if humanleg.conversation_closed(e)]
    if closed_at:
        assert all(e == history[closed_at[0]] for e in history[closed_at[0]:])
        assert world.closed
    # Liveness: with every close accepted, a call that ended with no transcript still running is closed
    # (a failed start never leaves the close waiting for a transcript that does not exist).
    if world.call_ended and not world.running and not world.close_failed:
        assert world.closed


@settings(max_examples=500, deadline=None)
@given(switch=st.booleans(), agent_legs=st.booleans(), events=st.lists(EVENTS, min_size=1, max_size=12),
       add_results=st.lists(st.booleans(), max_size=6), start_results=st.lists(st.booleans(), max_size=6),
       close_results=st.lists(st.booleans(), max_size=3))
# Interleavings found while building the handlers, kept: the close's repair and a late accept adding the
# same worker at once (both orders), a start that fails while the call ends, a transfer whose first
# agent leg ends before the call, and a failed agent-leg start retried by the next update.
@example(switch=True, agent_legs=True,
         events=[("accept", 1), ("status",), ("concurrent", ("accept", 0), ("stopped",), "add")],
         add_results=[], start_results=[], close_results=[])
@example(switch=False, agent_legs=True,
         events=[("accept", 4), ("concurrent", ("status",), ("accept", 1), "add")],
         add_results=[False], start_results=[False], close_results=[])
@example(switch=True, agent_legs=True, events=[("concurrent", ("accept", 0), ("status",), "start")],
         add_results=[], start_results=[False], close_results=[])
@example(switch=True, agent_legs=False, events=[("status",), ("accept", 0)], add_results=[],
         start_results=[], close_results=[False])
@example(switch=True, agent_legs=True,
         events=[("accept", 0), ("updated", (0,)), ("accept", 1, (0,)), ("updated", (0, 1)),
                 ("leg_end", 0, humanleg.TRANSCRIPTION_STOPPED), ("forged", 1), ("status",), ("stopped",),
                 ("leg_end", 1, humanleg.TRANSCRIPTION_STOPPED)],
         add_results=[], start_results=[], close_results=[])
@example(switch=True, agent_legs=True,
         events=[("accept", 0), ("updated", (2,)), ("updated", (2,)),
                 ("concurrent", ("updated", (3,)), ("status",), "start")],
         add_results=[], start_results=[True, False, True, False], close_results=[])
def test_accepts_are_idempotent_switched_and_never_after_close(switch, agent_legs, events, add_results,
                                                               start_results, close_results):
    world, h = setup(switch, add_results, start_results, close_results, agent_legs=agent_legs)
    world.start_ok_always = all(start_results)
    history: list[dict[str, Any]] = []
    for event in events:
        run(world, event)
        state = callstate.load(h.store, CALL)
        assert state is not None
        history.append(dict(state.escalation or {}))
    check(world, h, switch, history)


@settings(max_examples=150, deadline=None)
@given(switch=st.booleans(), order=st.permutations(range(len(POOL))),
       replays=st.lists(st.integers(0, len(POOL) - 1), max_size=6))
def test_every_delivered_worker_is_a_participant_once_whatever_the_order(switch, order, replays):
    """Healthy dependencies, all events delivered in any order with replays: every worker is listed
    exactly once, the hand-off is marked once, and (switch on) the transcription starts once."""
    world, h = setup(switch, [], [])
    for i in list(order) + list(replays):
        run(world, ("accept", i))
    state = callstate.load(h.store, CALL)
    assert state is not None and state.escalation is not None
    listed = [p["worker_sid"] for p in state.escalation[humanleg.K_PARTICIPANTS]]
    assert sorted(listed) == sorted({w for _, w in POOL})
    assert sum(1 for r in world.requests if r[0] == "add") == len({w for _, w in POOL})
    assert sum(1 for r in world.requests if r[0] == "mark") == 1
    assert len(world.starts) == (1 if switch else 0)
    assert state.escalation[humanleg.K_TASK_SID] == TASK
    if switch:
        legs = humanleg.transcriptions(state.escalation)
        assert list(legs) == [CALL] and legs[CALL]["name"] == humanleg.TRANSCRIPTION_NAME
        assert legs[CALL]["role"] == humanleg.ROLE_CUSTOMER


@settings(max_examples=200, deadline=None)
@given(agent_legs=st.booleans(),
       events=st.lists(st.one_of(st.tuples(st.just("accept"), st.integers(0, len(POOL) - 1), LEG_LISTS),
                                 st.tuples(st.just("updated"), LEG_LISTS)), min_size=1, max_size=10))
def test_every_named_agent_leg_starts_once_up_to_the_cap_whatever_the_order(agent_legs, events):
    """Healthy dependencies, accepts and task updates in any order, repeated: the customer leg starts
    once if any accept arrived; each agent leg named after the hand-off (by an update or an accept not
    already handled) starts exactly once, the first
    :data:`~owlbank.humanleg.MAX_AGENT_LEGS` distinct ones in the order Flex named them (none with the
    switch off); an update before any accept starts nothing."""
    world, h = setup(True, [], [], agent_legs=agent_legs)
    expected: list[str] = []
    handed_off = False
    seen: set[int] = set()
    for event in events:
        replay = event[0] == "accept" and event[1] in seen  # the same EventSid: dropped as handled
        if event[0] == "accept":
            seen.add(event[1])
        handed_off = handed_off or event[0] == "accept"
        if handed_off and agent_legs and not replay:
            named_legs = [AGENTS[i] for i in event[-1]][:humanleg.MAX_AGENT_LEGS]
            expected += [leg for leg in named_legs if leg not in expected]
        run(world, event)
    assert world.violations == []
    customer = [leg for leg, _ in world.starts if leg == CALL]
    assert customer == ([CALL] if handed_off else [])
    assert [leg for leg, _ in world.starts if leg != CALL] == expected[:humanleg.MAX_AGENT_LEGS]
    state = callstate.load(h.store, CALL)
    assert state is not None
    assert list(humanleg.transcriptions(state.escalation)) == customer + expected[:humanleg.MAX_AGENT_LEGS]
