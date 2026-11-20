"""The stale-conversation sweep from our own records.

* :func:`owlbank.humanleg.sweep_step` closes only a call whose end is recorded or
  reported more than the grace period ago, never a live or unknown one however long it has waited; a
  second run of :func:`lambdas.conversation_sweep.handler.run_sweep` on the same store closes nothing
  more and sends no further close.
* For any call state, ``to_item`` carries ``sweep_open`` and
  ``sweep_due_at`` exactly when ``escalation`` is set, the conversation is not closed, a conversation
  is possible (``ai_sessions > 0`` or a recorded conversation id) and fewer than 6 retries are counted
; each futile try (unknown, close not done) counts once and a live answer or a wait resets
  the count, so a live call is never dropped while one in every 6 consecutive tries answers live;
  ``sweep_due_at`` is a finite integer in range; ``from_item(to_item(s)) == s`` with the derived
  attributes never in ``_extras`` (a stale copy of them in the item included).
* For any mix of due calls (ended recorded or reported, within or
  beyond grace, ended with no end time, live, unknown, close failing) and any run time budget: each call
  the run handles (unknown included) closes, or its due time moves to at least now + 10 minutes less
  the 60 s slack (live, unknown, close not done) or to just after its end + grace; only the calls it does not reach keep
  their due time, and they stay before every rescheduled call; Calls API reads are at most the due
  calls with no recorded end. Over many runs with a fixed per-run capacity, and any number of calls
  that stay unknown, every ended call closes within ``ceil(n / capacity) + 1`` runs of becoming due,
  and no live or unknown call is ever closed. A futile call (close always fails, status always
  unknown) costs at most 6 tries in total, and one with no possible conversation none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from lambdas.conversation_sweep import handler as sweep
from owlbank import callstate, humanleg
from owlbank.callstate import CallState
from owlbank.humanleg import (
    ENDED_CALL_STATUSES,
    K_CALL_ENDED_AT,
    K_CONVERSATION_CLOSED_AT,
    K_CONVERSATION_ID,
    K_ESCALATED_AT,
    K_SWEEP_NEXT_AT,
    LIVE_CALL_STATUSES,
    STEP_CLOSE,
    SWEEP_GRACE_S,
    CallEnd,
)
from owlbank.humanleg_close import HumanLegCloser
from owlbank.store import MemoryStore

NOW = 1_793_873_730
SLACK = humanleg.SWEEP_SLACK_S
DUE_MAX = 253_402_300_799
STATUSES = [*sorted(ENDED_CALL_STATUSES | LIVE_CALL_STATUSES), "initiated", "", "weird"]


def _sid(i: int) -> str:
    return f"CA{i:032x}"


def _conv(i: int) -> str:
    return f"conv_{i:026x}"


def _iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- strategies

_times = st.integers(min_value=NOW - 10 * 86_400, max_value=NOW + 86_400)
_big = st.integers(min_value=-(10**15), max_value=10**15)
_any_num = st.one_of(st.none(), _times, _big, st.floats(allow_nan=True, allow_infinity=True),
                     st.text(max_size=5), st.booleans())
_at = st.one_of(st.none(), _times.map(_iso), _times, st.text(max_size=8))


@st.composite
def escalations(draw: Any, *, storable: bool = False) -> dict[str, Any] | None:
    if draw(st.integers(0, 9)) == 0:
        return draw(st.sampled_from([None, {}]))
    num = st.one_of(st.none(), _times, _big) if storable else _any_num
    esc: dict[str, Any] = {"reason": draw(st.sampled_from(["fraud_suspected", "caller_request"]))}
    for key, strategy in ((K_ESCALATED_AT, _at), (K_CALL_ENDED_AT, num), (K_SWEEP_NEXT_AT, num),
                          (K_CONVERSATION_CLOSED_AT, st.one_of(st.none(), _times)),
                          (humanleg.K_SWEEP_RETRIES, st.one_of(st.none(), st.integers(-2, 9))
                           if storable else st.one_of(st.none(), st.integers(-2, 9), _any_num)),
                          (K_CONVERSATION_ID, st.one_of(st.none(), st.just("conv_" + "a" * 26)))):
        value = draw(strategy)
        if value is not None:
            esc[key] = value
    return esc


call_ends = st.one_of(st.none(), st.builds(CallEnd, st.sampled_from(STATUSES),
                                           st.one_of(st.none(), _times.map(float),
                                                     st.floats(allow_nan=True, allow_infinity=True))))


# --------------------------------------------------------------------------- (pure)

def _end_of(esc: dict[str, Any], call_end: CallEnd | None) -> float | None:
    recorded = esc.get(K_CALL_ENDED_AT)
    if humanleg._finite(recorded):
        return float(recorded)
    if call_end is not None and call_end.status.strip().lower() in ENDED_CALL_STATUSES \
            and humanleg._finite(call_end.ended_at):
        return float(call_end.ended_at)
    return None


@settings(max_examples=500, deadline=None)
@given(esc=escalations(), call_end=call_ends, now=st.one_of(_times, st.floats()))
def test_closes_only_calls_ended_beyond_grace(esc: dict[str, Any] | None, call_end: CallEnd | None,
                                                now: float) -> None:
    step = humanleg.sweep_step(esc, call_end, now)
    if step.outcome == STEP_CLOSE:
        assert esc and not humanleg.conversation_closed(esc)
        end = _end_of(esc, call_end)
        assert end is not None and end < now - SWEEP_GRACE_S
    if esc and _end_of(esc, call_end) is None:  # live, unknown or no answer: never closed
        assert step.outcome != STEP_CLOSE
    assert humanleg.sweep_step(esc, call_end, now) == step  # deterministic



@settings(max_examples=500, deadline=None)
@given(esc=escalations(storable=True), stale=st.booleans(), ai_sessions=st.integers(0, 3))
def test_index_keys_iff_escalated_open_closable_and_retries_left(esc: dict[str, Any] | None, stale: bool,
                                                                    ai_sessions: int) -> None:
    state = CallState(call_sid=_sid(1), expires_at=NOW + 86_400, ai_sessions=ai_sessions, escalation=esc)
    fresh = state.to_item()
    retries = (esc or {}).get(humanleg.K_SWEEP_RETRIES)
    member = (bool(esc) and not humanleg.conversation_closed(esc)
              and (ai_sessions > 0 or bool((esc or {}).get(K_CONVERSATION_ID)))
              and not (isinstance(retries, int) and retries >= humanleg.SWEEP_MAX_RETRIES))
    assert (callstate.SWEEP_OPEN_ATTR in fresh) is member
    assert (callstate.SWEEP_DUE_ATTR in fresh) is member
    if member:
        due = fresh[callstate.SWEEP_DUE_ATTR]
        assert fresh[callstate.SWEEP_OPEN_ATTR] == callstate.SWEEP_OPEN
        assert type(due) is int and 0 <= due <= DUE_MAX
        assert due == humanleg.sweep_due_at(esc)
    item = {**fresh, callstate.SWEEP_OPEN_ATTR: "open", callstate.SWEEP_DUE_ATTR: 5} if stale else fresh
    back = CallState.from_item(item)
    assert back == state
    assert back.to_item() == fresh
    for attr in (callstate.SWEEP_OPEN_ATTR, callstate.SWEEP_DUE_ATTR):
        assert attr not in (back.to_item().get("_extras") or {})


@settings(max_examples=300, deadline=None)
@given(esc=escalations(), now=_times)
def test_due_time_is_an_int_in_range_and_closed_maps_are_never_written(
        esc: dict[str, Any] | None, now: int) -> None:
    due = humanleg.sweep_due_at(esc)
    assert type(due) is int and 0 <= due <= DUE_MAX
    recorded = (esc or {}).get(K_CALL_ENDED_AT)
    retried = humanleg.sweep_retry(esc, now)
    if retried is not None:  # each retry counts once; a live reschedule doesn't
        assert humanleg.sweep_retries(retried) == humanleg.sweep_retries(esc) + 1
    rescheduled = humanleg.sweep_reschedule(esc, CallEnd("queued", None), now)
    if rescheduled is not None:  # a useful answer resets the count of consecutive futile tries
        assert humanleg.sweep_retries(rescheduled) == 0
    for live, new in ((True, retried),
                      (not humanleg._finite(recorded), humanleg.sweep_reschedule(esc, CallEnd("queued", None), now))):
        if new is None:
            continue
        assert esc and not humanleg.conversation_closed(esc)  # Nothing written once closed
        if live:  # live or retry: 10 minutes on less the slack (or the clamp's ceiling)
            assert humanleg.sweep_due_at(new) >= now + 600 - SLACK or humanleg.sweep_due_at(new) == DUE_MAX
        else:  # a recorded end within grace wins over a live report: just after end + grace
            assert humanleg.sweep_due_at(new) == min(max(math.floor(recorded) + SWEEP_GRACE_S + 1, 0), DUE_MAX)


# --------------------------------------------------------------------------- run_sweep worlds

KINDS = ("ended_old", "ended_recent", "reported_old", "reported_recent", "ended_no_time", "live", "unknown",
         "no_session", "spent")
#: Never in the index: escalated before any AI session with no conversation id, or out of retries.
FUTILE = ("no_session", "spent")
ENDED_OLD = ("ended_old", "reported_old")


@dataclass
class World:
    """Orchestrator and the Calls API, with a simulated clock: each read or close costs time."""

    statuses: dict[str, CallEnd | None] = field(default_factory=dict)
    close_fails: set[str] = field(default_factory=set)
    closes: list[str] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    t: float = 0.0

    def resolve_conversation(self, call_sid: str) -> str | None:
        return None

    def close(self, conversation_id: str) -> bool:
        self.t += 1.0
        self.closes.append(conversation_id)
        return conversation_id not in self.close_fails

    def stop_transcription(self, call_sid: str, name: str) -> bool:
        return True

    def status(self, call_sid: str) -> CallEnd | None:
        self.t += 0.3
        self.asked.append(call_sid)
        return self.statuses.get(call_sid)


def _seed(store: MemoryStore, world: World, i: int, kind: str, due_offset: int, fails: bool) -> None:
    sid = _sid(i)
    esc: dict[str, Any] = {"reason": "r", K_CONVERSATION_ID: _conv(i), K_SWEEP_NEXT_AT: NOW - due_offset,
                           K_ESCALATED_AT: _iso(NOW - 3600)}
    # Some calls with no AI session: still closable because their conversation id is recorded.
    ai_sessions = 0 if due_offset % 3 == 0 else 1
    if kind == "no_session":  # escalated before any AI session: no conversation at all
        del esc[K_CONVERSATION_ID]
        ai_sessions = 0
        esc[K_CALL_ENDED_AT] = NOW - 900
    elif kind == "spent":
        esc[humanleg.K_SWEEP_RETRIES] = humanleg.SWEEP_MAX_RETRIES
        esc[K_CALL_ENDED_AT] = NOW - 900
    if kind == "ended_old":
        esc[K_CALL_ENDED_AT] = NOW - 900
    elif kind == "ended_recent":
        esc[K_CALL_ENDED_AT] = NOW - 100
    world.statuses[sid] = {"reported_old": CallEnd("completed", float(NOW - 900)),
                           "reported_recent": CallEnd("busy", float(NOW - 100)),
                           "ended_no_time": CallEnd("completed", None),
                           "live": CallEnd("in-progress", None)}.get(kind)
    if fails:
        world.close_fails.add(_conv(i))
    callstate.create(store, CallState(call_sid=sid, expires_at=NOW + 86_400, ai_sessions=ai_sessions,
                                      escalation=esc))


def _deps(world: World, store: MemoryStore, now: int) -> sweep.SweepDeps:
    closer = HumanLegCloser(store, world, calls=world, clock=lambda: now)
    return sweep.SweepDeps(due=sweep.store_due(store), states=sweep.store_states(store, clock=lambda: now),
                           calls=world, closer=closer, write=sweep.store_writer(store, clock=lambda: now),
                           clock=lambda: now)


def _escalation(store: MemoryStore, sid: str) -> dict[str, Any]:
    state = callstate.load(store, sid)
    assert state is not None and state.escalation is not None
    return dict(state.escalation)


mixes = st.lists(st.tuples(st.sampled_from(KINDS), st.integers(0, 5000), st.booleans()), min_size=1, max_size=12)


@settings(max_examples=200, deadline=None)
@given(mix=mixes, run_s=st.floats(min_value=0, max_value=40))
def test_each_handled_call_progresses_and_the_rest_keep_their_place(
        mix: list[tuple[str, int, bool]], run_s: float) -> None:
    store, world = MemoryStore(), World()
    for i, (kind, offset, fails) in enumerate(mix):
        _seed(store, world, i, kind, offset, fails)
    before = {_sid(i): _escalation(store, _sid(i)) for i in range(len(mix))}
    end = sweep.STOP_MARGIN_S + run_s
    report = sweep.run_sweep(_deps(world, store, NOW), time_left=lambda: end - world.t)
    kept: list[int] = []
    moved: list[int] = []
    for i, (kind, _offset, fails) in enumerate(mix):
        old, new = before[_sid(i)], _escalation(store, _sid(i))
        if kind in FUTILE:  # never in the index: no read, no close, no write
            assert new == old and _sid(i) not in world.asked and _conv(i) not in world.closes
            continue
        if humanleg.conversation_closed(new):  # Only ended beyond grace
            assert kind in ENDED_OLD and not fails
            continue
        due_old, due_new = humanleg.sweep_due_at(old), humanleg.sweep_due_at(new)
        if new == old:
            kept.append(due_new)
            continue
        moved.append(due_new)
        assert due_new > due_old
        if kind in ("live", "unknown", "ended_no_time") or (kind in ENDED_OLD and fails):
            assert due_new >= NOW + 600 - SLACK
        elif kind in ("ended_recent", "reported_recent"):
            assert due_new == NOW - 100 + SWEEP_GRACE_S + 1
        else:
            raise AssertionError(f"{kind} must not be written")
    assert all(k <= NOW for k in kept) and all(m > NOW for m in moved)  # unreached first next run
    if not report.out_of_time:  # every call was handled: none keeps its place, unknown included
        assert kept == []
    no_end = sum(1 for kind, _o, _f in mix if kind not in ("ended_old", "ended_recent", *FUTILE))
    assert report.statuses == len(world.asked) <= no_end
    recorded = {_sid(i) for i, (k, _o, _f) in enumerate(mix) if k in ("ended_old", "ended_recent")}
    assert not set(world.asked) & recorded  # a recorded end costs no Calls API read
    if not report.out_of_time:
        assert report.closed + report.rescheduled + report.unknown + report.skipped \
            == sum(1 for k, _o, _f in mix if k not in FUTILE)
    assert report.due == sum(1 for k, _o, _f in mix if k not in FUTILE)


@settings(max_examples=150, deadline=None)
@given(mix=mixes)
def test_a_second_run_closes_nothing_more(mix: list[tuple[str, int, bool]]) -> None:
    store, world = MemoryStore(), World()
    for i, (kind, offset, fails) in enumerate(mix):
        _seed(store, world, i, kind, offset, fails)
    first = sweep.run_sweep(_deps(world, store, NOW))
    closes = list(world.closes)
    second = sweep.run_sweep(_deps(world, store, NOW))
    assert second.closed == 0 and world.closes == closes
    assert first.closed == sum(1 for k, _o, f in mix if k in ENDED_OLD and not f)
    never = {_conv(i) for i, (k, _o, _f) in enumerate(mix) if k in ("live", "unknown", "ended_no_time", *FUTILE)}
    assert not never & set(world.closes)


def _record_end(store: MemoryStore, sid: str, ended: int, now: int) -> None:
    """``/voice/status`` recording the end (only that part of: ``call_ended_at``, and it drops
    ``sweep_next_at``), so the front door's own close does not interfere with the sweep here."""
    def fn(s: CallState) -> dict[str, Any] | None:
        esc = dict(s.escalation or {})
        if not esc or K_CALL_ENDED_AT in esc or humanleg.conversation_closed(esc):
            return None
        esc[K_CALL_ENDED_AT] = ended
        esc.pop(K_SWEEP_NEXT_AT, None)
        return {"escalation": esc}
    callstate.mutate(store, sid, fn, now=lambda: now)


def _bounded_runs(n_live: int, n_unknown: int, n_ended: int, capacity: int, data: st.DataObject, *,
                  n_failing: int = 0, n_no_session: int = 0) -> None:
    """Runs every 10 minutes, each with time for about ``capacity`` closes (a close costs 1 s, a status
    read 0.3 s); ended calls end at drawn times and ``/voice/status`` records it; unknown calls never
    get a usable status; failing calls end but their close always fails; no-session calls escalated
    before any AI session and have no conversation. No live or unknown call is ever closed; an ended
    call closes within ``ceil(n / capacity) + 1`` runs of becoming due; a futile call costs at most
    ``SWEEP_MAX_RETRIES`` tries in total, a no-session call none."""
    store, world = MemoryStore(), World()
    groups = (("live", n_live), ("unknown", n_unknown), ("ended", n_ended), ("failing", n_failing),
              ("no_session", n_no_session))
    kind_of = [k for k, count in groups for _ in range(count)]
    n = len(kind_of)
    ends = {i: NOW + data.draw(st.integers(-3000, 6000)) for i, k in enumerate(kind_of)
            if k in ("ended", "failing", "no_session")}
    for i, kind in enumerate(kind_of):
        world.statuses[_sid(i)] = None if kind == "unknown" else CallEnd("in-progress", None)
        esc: dict[str, Any] = {"reason": "r", K_ESCALATED_AT: NOW - data.draw(st.integers(0, 3600))}
        ai_sessions = data.draw(st.integers(0, 2))  # 0 is fine with a recorded conversation id
        if kind == "no_session":
            ai_sessions = 0
        else:
            esc[K_CONVERSATION_ID] = _conv(i)
        if kind == "failing":
            world.close_fails.add(_conv(i))
        callstate.create(store, CallState(call_sid=_sid(i), expires_at=NOW + 10 * 86_400, ai_sessions=ai_sessions,
                                          escalation=esc))
    bound = math.ceil(n / capacity) + 1
    closed_run: dict[int, int] = {}
    for run in range(bound + 12):
        now = NOW + 600 * run
        for i, ended in ends.items():
            if ended <= now:
                world.statuses[_sid(i)] = CallEnd("completed", float(ended))
                _record_end(store, _sid(i), ended, now)
        start = world.t
        budget = sweep.STOP_MARGIN_S + capacity + 0.5
        sweep.run_sweep(_deps(world, store, now), time_left=lambda b=budget, s=start: b - (world.t - s))
        for i in range(n):
            if humanleg.conversation_closed(_escalation(store, _sid(i))):
                assert kind_of[i] == "ended", "a live, unknown or futile call was closed"
                closed_run.setdefault(i, run)
    for i, ended in ends.items():
        if kind_of[i] != "ended":
            continue
        first_due_run = math.ceil(max(ended + SWEEP_GRACE_S + 1 - NOW, 0) / 600)
        assert i in closed_run and closed_run[i] <= first_due_run + bound, (i, ended, closed_run.get(i))
    for i, kind in enumerate(kind_of):
        tries = world.asked.count(_sid(i)) if kind == "unknown" else world.closes.count(_conv(i))
        if kind == "no_session":
            assert _sid(i) not in world.asked and tries == 0
        elif kind in ("unknown", "failing"):
            assert tries <= humanleg.SWEEP_MAX_RETRIES


@settings(max_examples=60, deadline=None)
@given(n_live=st.integers(0, 15), n_ended=st.integers(1, 15), capacity=st.integers(1, 6), data=st.data())
def test_every_ended_call_closes_within_bounded_runs(n_live: int, n_ended: int, capacity: int,
                                                         data: st.DataObject) -> None:
    _bounded_runs(n_live, 0, n_ended, capacity, data)


@settings(max_examples=60, deadline=None)
@given(n_unknown=st.integers(1, 40), n_live=st.integers(0, 8), n_ended=st.integers(1, 10),
       capacity=st.integers(1, 6), data=st.data())
def test_permanent_unknowns_do_not_crowd_out_ended_calls(n_unknown: int, n_live: int, n_ended: int,
                                                             capacity: int, data: st.DataObject) -> None:
    """An unknown call backs off 10 minutes like a close not done, so any number of
    calls that never get a usable status still leave every ended call closed within the bound."""
    _bounded_runs(n_live, n_unknown, n_ended, capacity, data)


@settings(max_examples=60, deadline=None)
@given(n_failing=st.integers(0, 15), n_no_session=st.integers(0, 15), n_unknown=st.integers(0, 10),
       n_ended=st.integers(1, 8), capacity=st.integers(1, 6), data=st.data())
def test_futile_calls_cost_at_most_six_retries(n_failing: int, n_no_session: int, n_unknown: int,
                                                   n_ended: int, capacity: int, data: st.DataObject) -> None:
    """Calls whose close always fails, calls that stay unknown and calls with no conversation
    cost at most SWEEP_MAX_RETRIES tries each (none for no conversation) and never stop an ended
    call from closing within the bound."""
    _bounded_runs(0, n_unknown, n_ended, capacity, data, n_failing=n_failing, n_no_session=n_no_session)


@settings(max_examples=150, deadline=None)
@given(answers=st.lists(st.sampled_from(["unknown", "live"]), min_size=1, max_size=40))
def test_a_live_call_stays_indexed_while_one_in_six_tries_answers_live(answers: list[str]) -> None:
    """Over any sequence of unknown and live answers, the call is in the index exactly
    while it has not had 6 unknowns in a row; once out it is never read again; it is never closed."""
    store, world = MemoryStore(), World()
    callstate.create(store, CallState(call_sid=_sid(0), expires_at=NOW + 10 * 86_400, ai_sessions=1,
                                      escalation={"reason": "r", K_CONVERSATION_ID: _conv(0),
                                                  K_SWEEP_NEXT_AT: NOW, K_ESCALATED_AT: NOW}))
    streak, out, t = 0, False, NOW
    for answer in answers:
        world.statuses[_sid(0)] = None if answer == "unknown" else CallEnd("in-progress", None)
        before = len(world.asked)
        sweep.run_sweep(_deps(world, store, t))
        if out:
            assert len(world.asked) == before  # out of the index: never read again
        else:
            assert len(world.asked) == before + 1
            streak = streak + 1 if answer == "unknown" else 0
            out = streak >= humanleg.SWEEP_MAX_RETRIES
        esc = _escalation(store, _sid(0))
        assert not humanleg.conversation_closed(esc) and world.closes == []
        assert humanleg.sweep_retries(esc) == streak
        indexed = bool(sweep.store_due(store)(DUE_MAX))
        assert indexed is (not out)
        t = max(t + 1, humanleg.sweep_due_at(esc))
