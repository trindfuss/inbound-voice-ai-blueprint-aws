"""Model failover: properties of the in-process decision state, over generated settings and generated sequences of model calls, turns,
fleet views and clock steps (a fake clock: every time is passed in).

* ``choose`` returns the best-ranked healthy entry; with none, the degraded entry with the
  lowest p90 (rank breaks ties, an unknown p90 last); an unavailable entry only when all are, then the
  last. The state it acts on is never better than the local mark or the fleet's.
* Over generated turns (choose, then ``next_after`` while calls fail before
  output), no turn starts on a degraded entry while one is healthy, and no call goes to an unavailable
  entry while a usable one exists.
* ``next_after`` gives the best-ranked usable entry the turn hasn't tried, whatever its
  place in the list (an earlier degraded entry included); each entry gets a turn at most once, and the
  turn runs out only when every usable entry has been tried.
* Two degraded marks of an entry are at least ``dwell_s`` apart; a degraded
  mark lapses no sooner than ``dwell_s`` after it and only once no better-ranked entry is degraded
  (idle reset aside).
* 4a.5: a local unavailable mark never clears on a timer or the idle reset; only after this process has
  seen a fleet view show the entry unavailable and a later one show it not, by ``mark_available``, or by
  the backstop: a fresh view still not showing it unavailable more than ``unavailable_recheck_s`` after
  the mark. ``unconfirmed`` names exactly the marks no view has shown unavailable yet and no older than
  ``unavailable_recheck_s`` (the reporter re-reports them, 4a.1).
* End to end: request errors interleaved anywhere change no state, no window, no choice.
* ``idle_reset_s`` after the last sample every entry not marked unavailable reads healthy.
* Lapses take effect at their due times however often the state is looked at (the uvloop lesson: a
  look that comes early changes nothing, a later one catches up), and memory stays bounded.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.failover.classify import UNAVAILABLE, Classification, Outcome
from owlbank.failover.core import EntryState, FailoverState, FleetView, Transition
from owlbank.failover.settings import EntrySettings, Settings

PROPS = settings(max_examples=300, deadline=None)
H, D, U = EntryState.HEALTHY, EntryState.DEGRADED, EntryState.UNAVAILABLE
SEVERITY = {H: 0, D: 1, U: 2}
TOL = 1e-6

CALL_OUTCOMES: list[Classification] = [*(o for o in Outcome if o is not Outcome.REQUEST_ERROR), UNAVAILABLE]


@st.composite
def failover_settings(draw: st.DrawFn) -> Settings:
    n = draw(st.integers(1, 3))
    entries = []
    for i in range(n):
        trip = draw(st.floats(0.5, 5.0))
        entries.append(EntrySettings(model_id=f"m{i}", trip_ttft_s=trip,
                                     recover_ttft_s=draw(st.floats(0.1, trip, exclude_max=True)),
                                     deadline_s=None if i == n - 1 else draw(st.floats(0.5, 6.0)),
                                     error_rate_limit=draw(st.floats(0.05, 1.0))))
    window = draw(st.integers(1, 12))
    return Settings(entries=tuple(entries), window_calls=window, min_samples=draw(st.integers(1, window)),
                    window_age_s=draw(st.floats(1.0, 120.0)), miss_run=draw(st.integers(1, 4)),
                    dwell_s=draw(st.floats(0.0, 60.0)), idle_reset_s=draw(st.floats(1.0, 200.0)),
                    unavailable_recheck_s=draw(st.floats(1.0, 200.0)))


steps = st.one_of(st.floats(0.0, 3.0), st.floats(0.0, 3.0), st.floats(20.0, 250.0))
ttfts = st.one_of(st.none(), st.floats(0.0, 10.0))


@st.composite
def fleet_views(draw: st.DrawFn, n: int) -> FleetView | None:
    if draw(st.booleans()):
        return None
    return FleetView(states=tuple(draw(st.sampled_from([H, H, D, U])) for _ in range(n)),
                     p90_s=tuple(draw(st.one_of(st.none(), st.floats(0.0, 10.0))) for _ in range(n))
                     if draw(st.booleans()) else ())


@dataclass(frozen=True)
class Op:
    kind: str  # "call", "choose", "available", "turn"
    dt: float
    entry: int = 0
    outcome: Classification = Outcome.OK
    ttft: float | None = None
    fleet: FleetView | None = None
    fails: tuple[Classification | None, ...] = ()


@st.composite
def ops(draw: st.DrawFn, n: int, max_size: int = 60) -> list[Op]:
    out = []
    for _ in range(draw(st.integers(0, max_size))):
        kind = draw(st.sampled_from(["call", "call", "call", "choose", "available", "turn"]))
        out.append(Op(kind=kind, dt=draw(steps), entry=draw(st.integers(0, n - 1)),
                      outcome=draw(st.sampled_from(CALL_OUTCOMES)), ttft=draw(ttfts),
                      fleet=draw(fleet_views(n)),
                      fails=tuple(draw(st.lists(st.one_of(st.none(), st.sampled_from(CALL_OUTCOMES)),
                                                min_size=1, max_size=3)))))
    return out


@st.composite
def scenarios(draw: st.DrawFn) -> tuple[Settings, list[Op]]:
    s = draw(failover_settings())
    return s, draw(ops(len(s)))


def p90_key(fs: FailoverState, i: int, now: float, fleet: FleetView | None) -> tuple[bool, float, int]:
    p90 = fleet.p90(i) if fleet is not None else None
    if p90 is None:
        p90 = fs.stats(i, now=now).p90_s
    return (p90 is None, p90 if p90 is not None else 0.0, i)


def check_choice(fs: FailoverState, now: float, fleet: FleetView | None) -> int:
    """The caller half of ``choose`` and ``next_after`` at one instant."""
    n = len(fs.settings)
    states = [fs.state(i, now=now, fleet=fleet) for i in range(n)]
    for i, st_ in enumerate(states):  # the state acted on is the worse of the two
        assert SEVERITY[st_] >= SEVERITY[fs.local_state(i, now=now)]
        if fleet is not None:
            assert SEVERITY[st_] >= SEVERITY[fleet.states[i]]
    c = fs.choose(now=now, fleet=fleet)
    healthy = [i for i, s in enumerate(states) if s is H]
    degraded = [i for i, s in enumerate(states) if s is D]
    if healthy:
        assert c == healthy[0]
    elif degraded:
        assert c == min(degraded, key=lambda i: p90_key(fs, i, now, fleet))
    else:
        assert c == n - 1
    return c


def ranked(fs: FailoverState, now: float, fleet: FleetView | None) -> list[int]:
    """The usable entries in rank, independently of the code under test: healthy in list order, then
    degraded by p90 (the order ``choose`` uses)."""
    states = [fs.state(i, now=now, fleet=fleet) for i in range(len(fs.settings))]
    healthy = [i for i, s in enumerate(states) if s is H]
    degraded = sorted((i for i, s in enumerate(states) if s is D), key=lambda i: p90_key(fs, i, now, fleet))
    return healthy + degraded


def check_next(fs: FailoverState, now: float, fleet: FleetView | None, tried: set[int], nxt: int | None) -> None:
    """One instant of ``next_after``: the best-ranked untried usable entry, or ``None`` only when none is left."""
    left = [i for i in ranked(fs, now, fleet) if i not in tried]
    assert nxt == (left[0] if left else None)
    assert nxt is None or (nxt not in tried and fs.state(nxt, now=now, fleet=fleet) is not U)


def run(fs: FailoverState, scenario: list[Op], start: float = 0.0, *, check: bool = False,
        on_step: object = None) -> float:
    now = start
    for op in scenario:
        now += op.dt
        if op.kind == "call":
            fs.record(op.entry, op.outcome, op.ttft, now=now)
        elif op.kind == "available":
            fs.mark_available(op.entry, now=now)
        elif op.kind == "choose":
            if check:
                check_choice(fs, now, op.fleet)
            else:
                fs.choose(now=now, fleet=op.fleet)
        else:  # a turn: choose, then move on while calls fail before any output
            n = len(fs.settings)
            c: int | None = check_choice(fs, now, op.fleet) if check else fs.choose(now=now, fleet=op.fleet)
            tried = set()
            for fail in op.fails:
                assert c is not None and c not in tried
                tried.add(c)
                if check:
                    states = [fs.state(i, now=now, fleet=op.fleet) for i in range(n)]
                    if any(s is not U for s in states):
                        assert states[c] is not U  # 4a.3: never an unavailable one while one is usable
                if fail is None:
                    fs.record(c, Outcome.OK, op.ttft, now=now)
                    break
                fs.record(c, fail, now=now)
                nxt = fs.next_after(c, tried=tried, now=now, fleet=op.fleet)
                if check:
                    check_next(fs, now, op.fleet, tried, nxt)
                if nxt is None:
                    break
                c = nxt
        if callable(on_step):
            on_step(now)
        if check:
            assert all(len(fs.health(i).window) <= fs.settings.window_calls for i in range(len(fs.settings)))
    return now




@PROPS
@given(scenarios())
def test_choice_over_generated_turns(scenario: tuple[Settings, list[Op]]) -> None:
    s, sequence = scenario
    run(FailoverState(s, clock=lambda: 0.0), sequence, check=True)




@PROPS
@given(scenarios(), st.data())
def test_request_errors_never_change_any_state(scenario: tuple[Settings, list[Op]],
                                                  data: st.DataObject) -> None:
    s, sequence = scenario
    clean, noisy = FailoverState(s, clock=lambda: 0.0), FailoverState(s, clock=lambda: 0.0)
    now = 0.0
    n = len(s)
    for op in sequence:
        # Request errors at arbitrary times up to this op, on any entry, any number of them.
        for frac in data.draw(st.lists(st.floats(0.0, 1.0), max_size=3)):
            assert noisy.record(data.draw(st.integers(0, n - 1)), Outcome.REQUEST_ERROR,
                                data.draw(ttfts), now=now + frac * op.dt) is None
        now = run(clean, [op], now)
        run(noisy, [op], now - op.dt)
        for i in range(n):
            assert clean.state(i, now=now, fleet=op.fleet) is noisy.state(i, now=now, fleet=op.fleet)
            assert clean.stats(i, now=now) == noisy.stats(i, now=now)
        assert clean.choose(now=now, fleet=op.fleet) == noisy.choose(now=now, fleet=op.fleet)




@PROPS
@given(scenarios())
def test_dwell_and_one_rank_at_a_time(scenario: tuple[Settings, list[Op]]) -> None:
    s, sequence = scenario
    n = len(s)
    fs = FailoverState(s, clock=lambda: 0.0)
    marks: dict[int, float] = {}  # entry -> time of its last degraded mark
    unav_at: dict[int, float] = {}  # entry -> time of its local unavailable mark
    fleet_saw = [False] * n
    last_sample: list[float | None] = [None]

    def is_idle_reset(t: Transition) -> bool:
        last = last_sample[0]
        return last is not None and abs(t.at_s - (last + s.idle_reset_s)) <= TOL

    def check(changes: list[Transition], before: list[EntryState]) -> None:
        local = list(before)
        for t in changes:
            if t.new is D:
                if t.entry in marks:
                    assert t.at_s - marks[t.entry] >= s.dwell_s - TOL  # Marks a dwell apart
                marks[t.entry] = t.at_s
            elif t.new is U:
                fleet_saw[t.entry] = False
                unav_at[t.entry] = t.at_s
            elif t.old is D and not is_idle_reset(t):
                assert t.at_s - marks[t.entry] >= s.dwell_s - TOL
                assert all(local[j] is not D for j in range(t.entry))  # one rank at a time
            local[t.entry] = t.new

    def snapshot() -> list[EntryState]:
        return [fs.health(i).state for i in range(n)]

    now = 0.0
    for op in sequence:
        now += op.dt
        before = snapshot()
        fleet = op.fleet if op.kind in ("choose", "turn") else None
        lapses = fs.advance(now=now, fleet=fleet)
        for t in lapses:
            if t.old is U:  # 4a.5: the fleet's unavailable-then-not, or the backstop, clears it
                assert fleet is not None and fleet.states[t.entry] is not U
                if t.cause == "fleet":
                    assert fleet_saw[t.entry]
                else:
                    assert t.cause == "unconfirmed" and not fleet_saw[t.entry]
                    assert now - unav_at[t.entry] > s.unavailable_recheck_s
        for i in range(n):
            if before[i] is U and fleet is not None and fleet.states[i] is U:
                fleet_saw[i] = True
        check(lapses, before)
        assert fs.advance(now=now, fleet=fleet) == []  # a second look at the same time changes nothing
        before = snapshot()
        if op.kind == "call":
            change = fs.record(op.entry, op.outcome, op.ttft, now=now)
            last_sample[0] = now
            check([change] if change else [], before)
        elif op.kind == "available":
            change = fs.mark_available(op.entry, now=now)
            check([change] if change else [], before)
            if change:
                fleet_saw[op.entry] = False
        else:
            fs.choose(now=now, fleet=op.fleet)




@PROPS
@given(scenarios(), st.floats(0.0, 1000.0))
def test_idle_reset_makes_every_unmarked_entry_healthy(scenario: tuple[Settings, list[Op]], extra: float) -> None:
    s, sequence = scenario
    fs = FailoverState(s, clock=lambda: 0.0)
    last: list[float | None] = [None]
    now = 0.0
    for op in sequence:
        now += op.dt
        if op.kind == "call":
            fs.record(op.entry, op.outcome, op.ttft, now=now)
            last[0] = now
        else:
            run(fs, [Op(kind=op.kind, dt=0.0, entry=op.entry, fleet=op.fleet, fails=())], now)  # no sample
    quiet = (last[0] if last[0] is not None else now) + s.idle_reset_s + extra
    quiet = max(quiet, now)
    marked = [fs.health(i).state is U for i in range(len(s))]
    assert all(fs.state(i, now=quiet) is (U if marked[i] else H) for i in range(len(s)))
    assert all(fs.stats(i, now=quiet).calls == 0 for i in range(len(s)))


# --- lapses at their due times, however often looked at ---------------------------------------


@PROPS
@given(scenarios(), st.lists(st.floats(0.0, 1.0), max_size=6))
def test_the_state_does_not_depend_on_how_often_it_is_read(scenario: tuple[Settings, list[Op]],
                                                           looks: list[float]) -> None:
    s, sequence = scenario
    calls = [op for op in sequence if op.kind in ("call", "available")]
    often, once = FailoverState(s, clock=lambda: 0.0), FailoverState(s, clock=lambda: 0.0)
    now = 0.0
    for op in calls:
        for frac in sorted(looks):  # early looks between events, as an early timer would make
            often.advance(now=now + frac * op.dt)
        now += op.dt
        run(often, [Op(kind=op.kind, dt=0.0, entry=op.entry, outcome=op.outcome, ttft=op.ttft)], now)
        run(once, [Op(kind=op.kind, dt=0.0, entry=op.entry, outcome=op.outcome, ttft=op.ttft)], now)
    for later in (0.0, 30.0, 300.0):
        for i in range(len(s)):
            assert often.local_state(i, now=now + later) is once.local_state(i, now=now + later)
            assert often.stats(i, now=now + later) == once.stats(i, now=now + later)


# --- The reissue order within a turn ----------------------------------------------------


@PROPS
@given(scenarios(), st.data())
def test_a_turn_tries_every_usable_entry_once_in_rank(scenario: tuple[Settings, list[Op]],
                                                          data: st.DataObject) -> None:
    """After any history, a turn whose calls all fail before output (by a deadline miss, which leaves
    the entry usable, or by an error) visits usable entries in rank, each once, and ends only when
    none is left untried."""
    s, sequence = scenario
    fs = FailoverState(s, clock=lambda: 0.0)
    now = run(fs, sequence)
    fleet = data.draw(fleet_views(len(s)))
    c: int | None = fs.choose(now=now, fleet=fleet)
    tried: set[int] = set()
    while c is not None:
        assert c not in tried
        tried.add(c)
        fs.record(c, data.draw(st.sampled_from([Outcome.ABANDONED, Outcome.ERROR, UNAVAILABLE])), now=now)
        c = fs.next_after(c, tried=tried, now=now, fleet=fleet)
        check_next(fs, now, fleet, tried, c)
    assert len(tried) <= len(s)


# --- 4a.1, 4a.5: re-reports and the backstop -------------------------------------------------------


@PROPS
@given(scenarios())
def test_unconfirmed_is_exactly_the_marks_no_view_has_shown(scenario: tuple[Settings, list[Op]]) -> None:
    s, sequence = scenario
    n = len(s)
    fs = FailoverState(s, clock=lambda: 0.0)
    cleared: list[Transition] = []
    fs.on_cleared(cleared.append)
    saw = [False] * n
    now = 0.0
    for op in sequence:
        now += op.dt
        before = [fs.health(i).state for i in range(n)]
        if op.kind == "call":
            change = fs.record(op.entry, op.outcome, op.ttft, now=now)
            if change is not None and change.new is U:
                saw[op.entry] = False
            continue
        if op.kind == "available":
            fs.mark_available(op.entry, now=now)
            continue
        changes = fs.advance(now=now, fleet=op.fleet)
        assert [c for c in changes if c.old is U] == cleared  # every clear reaches the listener
        cleared.clear()
        for i in range(n):
            if before[i] is U and op.fleet is not None and op.fleet.states[i] is U:
                saw[i] = True
        expected = tuple(i for i in range(n) if fs.health(i).state is U and not saw[i]
                         and now - fs.health(i).since_s <= s.unavailable_recheck_s)  # re-reports end
        assert fs.unconfirmed(now=now, fleet=op.fleet) == expected
        assert cleared == []  # a second look at the same time clears nothing more
