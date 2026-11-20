"""Per-leg transcriptions over the pure rules of :mod:`owlbank.humanleg`.

Over random interleavings of leg starts (the customer leg and up to 8 distinct agent legs, repeated, some
starts failing and released or marked ended), the call's end (``/voice/status``), Twilio's per-leg
``transcription-stopped``/``-error`` callbacks (repeated, and for legs never recorded), sweep runs after
the call ended, any time between them and failing closes, a driver that does what the front door and
the closer do with :func:`~owlbank.humanleg.claim_transcription`, :func:`~owlbank.humanleg.end_decision`
and :func:`~owlbank.humanleg.close_plan`:

* At most one transcription per leg (a failed start released may be retried), one customer leg, at most :data:`~owlbank.humanleg.MAX_AGENT_LEGS`
  agent legs; no start once the call's end is recorded or after the close;
* Every leg still running is stopped (on its own call) before the close; nothing is sent after it;
* The conversation closes only after the customer call ended and every started leg has ended or
  been stopped; an agent leg ending before that (a transfer) never closes it; it closes at most once
; with healthy dependencies, the end of every running leg after the call's end closes it, and so
  does the sweep.

The front door's switch (none with ``human_leg.agent_legs`` off) is tested through its own harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from hypothesis import example, given, settings
from hypothesis import strategies as st

from owlbank import humanleg

CUSTOMER = "CA" + "c" * 32
AGENTS = tuple("CA" + f"{i:032x}" for i in range(1, 9))  # more than the cap
LEGS = (CUSTOMER, *AGENTS)
T0 = 1_760_000_000


@dataclass
class World:
    close_results: list[bool]
    esc: dict[str, Any] = field(default_factory=lambda: {"reason": "fraud_suspected", "at": "x"})
    now: int = T0
    requests: list[tuple[Any, ...]] = field(default_factory=list)
    after_close: list[tuple[Any, ...]] = field(default_factory=list)
    running: set[str] = field(default_factory=set)  # started, end not reported, not stopped
    started_ok: set[str] = field(default_factory=set)
    call_ended: bool = False
    closed: bool = False
    closes: int = 0
    violations: list[str] = field(default_factory=list)

    def request(self, *op: Any) -> None:
        if self.closed:
            self.after_close.append(op)
        self.requests.append(op)


def role(leg: str) -> str:
    return humanleg.ROLE_CUSTOMER if leg == CUSTOMER else humanleg.ROLE_AGENT


def start(world: World, leg: str, outcome: str) -> None:
    """The front door: claim the leg in call state, then start it; a failed start is released (or,
    when that write fails, marked ended)."""
    claimed = humanleg.claim_transcription(world.esc, leg, name=humanleg.TRANSCRIPTION_NAME,
                                           role=role(leg), now=world.now)
    if claimed is None:
        return
    if world.closed:
        world.violations.append("claim after close")
    world.esc = dict(claimed)
    world.request("start", leg)
    if world.call_ended:
        world.violations.append("start after the call ended")
    if outcome == "ok":
        world.running.add(leg)
        world.started_ok.add(leg)
        world.request("started", leg)
        return
    undo = humanleg.release_transcription(world.esc, leg) if outcome == "fail_release" \
        else humanleg.end_unstarted_transcription(world.esc, leg, world.now)
    assert undo is not None
    world.esc = dict(undo)


def signal(world: World, event: str, leg: str | None = None, trigger: str = "") -> None:
    """The closer: the end decision on call state, then (if it closes) the plan, its stops, the close."""
    d = humanleg.end_decision(world.esc, event, world.now, leg=leg)
    if d.escalation is not None:
        world.esc = dict(d.escalation)
    if not d.close:
        return
    plan = humanleg.close_plan(SimpleNamespace(escalation=world.esc), [], now=world.now)
    # A close needs the customer call ended, and (unless the sweep) no leg running at all.
    if not world.call_ended:
        world.violations.append(f"close before the call ended ({trigger})")
    if trigger != "sweep" and plan.stop_transcriptions:
        world.violations.append(f"close with legs running ({trigger})")
    for stopped_leg, name in plan.stop_transcriptions:
        assert name == humanleg.TRANSCRIPTION_NAME
        world.request("stop", stopped_leg)
        world.running.discard(stopped_leg)
    # Every leg still running was stopped first.
    if world.running:
        world.violations.append(f"close with unstopped legs {sorted(world.running)}")
    world.request("close")
    ok = world.close_results.pop(0) if world.close_results else True
    if ok:
        world.closed = True
        world.closes += 1
        world.esc[humanleg.K_CONVERSATION_CLOSED_AT] = world.now
    else:
        world.esc.pop(humanleg.K_CLOSE_CLAIMED_AT, None)  # the closer releases its claim


def run(world: World, event: tuple[Any, ...], gap: int) -> None:
    world.now += gap
    kind = event[0]
    if kind == "start":
        start(world, LEGS[event[1]], event[2])
    elif kind == "status":
        world.call_ended = True
        signal(world, humanleg.CALL_ENDED, trigger="status")
    elif kind in ("stopped", "error"):
        leg = LEGS[event[1]]
        recorded = leg in humanleg.transcriptions(world.esc)
        if leg in world.started_ok:
            # Twilio reports a leg's end (repeats too). The customer leg's ``stopped`` comes when the
            # call ends; an agent leg's when that agent's leg ends (a transfer, or the call's end).
            world.running.discard(leg)
            if leg == CUSTOMER and kind == "stopped":
                world.call_ended = True
        elif recorded:
            return  # a claimed leg whose start failed: Twilio has nothing to report
        before = dict(world.esc)
        signal(world, humanleg.TRANSCRIPTION_STOPPED if kind == "stopped" else humanleg.TRANSCRIPTION_ERROR,
               leg=leg, trigger=kind)
        if not recorded:  # a callback for a leg call state never recorded changes nothing
            assert world.esc == before
    elif kind == "sweep" and world.call_ended:  # the sweep closes only calls it saw ended
        signal(world, humanleg.SWEEP, trigger="sweep")


def check(world: World) -> None:
    assert world.violations == []
    assert world.after_close == [], world.after_close
    assert world.closes <= 1
    assert humanleg.conversation_closed(world.esc) == world.closed
    # One transcription per leg. A start that failed and was released may be tried again by a
    # later event ("released if the start fails"); a leg never runs twice.
    starts = [r[1] for r in world.requests if r[0] == "started"]
    assert len(starts) == len(set(starts))
    assert sum(1 for s in starts if s != CUSTOMER) <= humanleg.MAX_AGENT_LEGS
    legs = humanleg.transcriptions(world.esc)
    assert sum(1 for e in legs.values() if e["role"] == humanleg.ROLE_AGENT) <= humanleg.MAX_AGENT_LEGS


LEG = st.integers(0, len(LEGS) - 1)
EVENTS = st.one_of(
    st.tuples(st.just("start"), LEG, st.sampled_from(["ok", "ok", "fail_release", "fail_end"])),
    st.tuples(st.sampled_from(["stopped", "error"]), LEG),
    st.just(("status",)),
    st.just(("sweep",)),
)
GAPS = st.integers(0, 2 * humanleg.CLOSE_CLAIM_TTL_S)


@settings(max_examples=400, deadline=None)
@given(events=st.lists(st.tuples(EVENTS, GAPS), max_size=25),
       close_results=st.lists(st.booleans(), max_size=3))
# A transfer: the first agent's leg stops while the call goes on, a second agent's leg starts; then
# the call ends and the legs stop one by one; only the last closes.
@example(events=[(("start", 0, "ok"), 1), (("start", 1, "ok"), 1), (("stopped", 1), 1),
                 (("start", 2, "ok"), 1), (("status",), 1), (("stopped", 0), 1), (("stopped", 2), 1)],
         close_results=[])
def test_legs_start_once_close_after_the_call_and_every_leg(events, close_results) -> None:
    world = World(list(close_results))
    for event, gap in events:
        run(world, event, gap)
        check(world)


@settings(max_examples=200, deadline=None)
@given(events=st.lists(st.tuples(EVENTS, GAPS), max_size=20),
       close_results=st.lists(st.booleans(), max_size=3), by_sweep=st.booleans())
@example(events=[(("start", 0, "ok"), 0), (("status",), 0), (("sweep",), 0)], close_results=[False],
         by_sweep=False)
def test_after_the_call_ends_the_last_leg_or_the_sweep_closes_once(events, close_results, by_sweep) -> None:
    """Liveness with healthy dependencies: the call ends, then either Twilio reports every running
    leg's end (the last one closes) or the sweep runs (stopping every running leg first)."""
    world = World(list(close_results))
    for event, gap in events:
        run(world, event, gap)
    world.close_results.clear()
    run(world, ("status",), 1)
    if by_sweep:
        run(world, ("sweep",), humanleg.CLOSE_CLAIM_TTL_S)
    else:
        # Twilio reports the end of every leg it ran whose end call state has not recorded: those still
        # running, and those a failed close had already stopped.
        unended = [leg for leg, e in humanleg.transcriptions(world.esc).items()
                   if e.get("ended_at") is None and leg in world.started_ok]
        for leg in unended:
            run(world, ("stopped", LEGS.index(leg)), 1)
        run(world, ("status",), humanleg.CLOSE_CLAIM_TTL_S)
    check(world)
    assert world.closed and world.closes == 1


@settings(max_examples=200, deadline=None)
@given(agent_events=st.lists(st.tuples(
    st.one_of(st.tuples(st.just("start"), st.integers(1, len(LEGS) - 1), st.sampled_from(["ok", "fail_end"])),
              st.tuples(st.sampled_from(["stopped", "error"]), st.integers(1, len(LEGS) - 1))), GAPS),
    max_size=25))
def test_agent_legs_ending_never_close_while_the_call_goes_on(agent_events) -> None:
    """With the customer leg running and the call live, no agent-leg start or end closes."""
    world = World([])
    run(world, ("start", 0, "ok"), 0)
    for event, gap in agent_events:
        run(world, event, gap)
        check(world)
    assert not world.closed and not any(r[0] in ("close", "stop") for r in world.requests)
