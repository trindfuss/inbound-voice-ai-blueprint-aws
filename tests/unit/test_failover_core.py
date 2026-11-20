"""The in-process decision state (``EntryHealth``/``FailoverState``).

A fake clock throughout; every time edge is checked on both sides of an explicit comparison (uvloop timers fire early, so nothing here may rely on a timer firing at or after its due time).
"""

from __future__ import annotations

import pytest

from owlbank.failover.classify import UNAVAILABLE, Outcome
from owlbank.failover.core import EntryState, FailoverState, FleetView, Transition
from owlbank.failover.settings import Settings

H55 = "us.anthropic.claude-haiku-5-5"  # trip 3.5, deadline 4.0
H45 = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # trip 1.6, deadline 3.0
S55 = "us.anthropic.claude-sonnet-5-5"  # last: no deadline
SETTINGS = Settings.for_models([H55, H45, S55])
EPS = 1e-6
HEALTHY, DEGRADED, UNAV = EntryState.HEALTHY, EntryState.DEGRADED, EntryState.UNAVAILABLE


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def make(settings: Settings = SETTINGS) -> tuple[FailoverState, Clock]:
    clock = Clock()
    return FailoverState(settings, clock=clock), clock


def misses(fs: FailoverState, i: int, now: float, n: int | None = None) -> None:
    for k in range(SETTINGS.miss_run if n is None else n):
        fs.record(i, Outcome.ABANDONED, now=now + k)


def fast(fs: FailoverState, i: int, now: float, n: int, ttft: float = 1.0) -> None:
    for k in range(n):
        fs.record(i, Outcome.OK, ttft, now=now + k * 0.1)


def states(fs: FailoverState, now: float, fleet: FleetView | None = None) -> list[EntryState]:
    return [fs.state(i, now=now, fleet=fleet) for i in range(len(SETTINGS))]


# --- trip and choice ---------------------------------------------------------


def test_starts_healthy_on_the_preferred_entry() -> None:
    fs, clock = make()
    assert states(fs, clock.t) == [HEALTHY] * 3
    assert fs.choose() == 0


def test_a_blip_never_degrades() -> None:
    fs, clock = make()
    fast(fs, 0, clock.t, 18)
    fs.record(0, Outcome.OK, 9.0, now=clock.t + 2.0)
    fs.record(0, Outcome.ABANDONED, now=clock.t + 2.1)  # two slow calls of 20, not in a row of three
    fs.record(0, Outcome.OK, 1.0, now=clock.t + 2.2)
    fs.record(0, Outcome.OK, 1.0, now=clock.t + 2.3)
    assert fs.state(0, now=clock.t + 3) is HEALTHY
    assert fs.choose(now=clock.t + 3) == 0


def test_a_run_of_misses_degrades_and_the_entry_is_skipped() -> None:
    fs, clock = make()
    fs.record(0, Outcome.ABANDONED, now=clock.t)
    fs.record(0, Outcome.ABANDONED, now=clock.t + 1)
    assert fs.choose(now=clock.t + 1) == 0
    change = fs.record(0, Outcome.ABANDONED, now=clock.t + 2)
    assert isinstance(change, Transition)
    assert (change.entry, change.old, change.new, change.at_s) == (0, HEALTHY, DEGRADED, clock.t + 2)
    assert change.stats is not None and change.stats.miss_run == 3
    assert fs.choose(now=clock.t + 2) == 1


def test_the_window_trips_on_p90() -> None:
    fs, clock = make()
    fast(fs, 1, clock.t, 17)
    for k in range(3):
        fs.record(1, Outcome.OK, 2.0, now=clock.t + 2 + k)  # three over 1.6 of 20: p90 over
    assert fs.state(1, now=clock.t + 5) is DEGRADED


def test_samples_older_than_the_window_age_do_not_count() -> None:
    fs, clock = make()
    fs.record(0, Outcome.ABANDONED, now=clock.t)
    fs.record(0, Outcome.ABANDONED, now=clock.t + 1)
    # The first miss is exactly window_age_s old: still in; just over: out.
    t = clock.t + SETTINGS.window_age_s
    assert fs.stats(0, now=t).miss_run == 2
    assert fs.stats(0, now=t + EPS).miss_run == 1
    fs.record(0, Outcome.ABANDONED, now=t + EPS)
    assert fs.state(0, now=t + EPS) is HEALTHY


def test_the_window_holds_at_most_window_calls_samples() -> None:
    fs, clock = make()
    fast(fs, 0, clock.t, 50)
    assert fs.stats(0, now=clock.t + 5).calls == SETTINGS.window_calls


def test_all_degraded_picks_the_lowest_p90() -> None:
    fs, clock = make()
    t = clock.t
    for i, slow in ((0, 6.0), (1, 2.5), (2, 9.0)):
        fast(fs, i, t, 17, ttft=slow)
        for k in range(3):
            fs.record(i, Outcome.OK, slow, now=t + 2 + k)
    assert states(fs, t + 5) == [DEGRADED] * 3
    assert fs.choose(now=t + 5) == 1


def test_all_degraded_ties_go_to_the_better_rank_and_no_p90_sorts_last() -> None:
    fs, clock = make()
    for i in range(3):
        misses(fs, i, clock.t)
    assert fs.choose(now=clock.t + 3) == 1  # p90 4.0, 3.0 (the deadlines) and none
    fleet = FleetView(states=(DEGRADED,) * 3, p90_s=(2.0, 2.0, 1.0))
    assert fs.choose(now=clock.t + 3, fleet=FleetView(states=(DEGRADED,) * 3, p90_s=(2.0, 2.0, None))) == 0
    assert fs.choose(now=clock.t + 3, fleet=fleet) == 2


def test_unavailable_never_gets_a_turn_unless_all_are_unavailable() -> None:
    fs, clock = make()
    fs.record(0, UNAVAILABLE, now=clock.t)
    assert fs.state(0, now=clock.t) is UNAV
    misses(fs, 1, clock.t)
    misses(fs, 2, clock.t)
    assert fs.choose(now=clock.t + 3) == 1  # every usable one degraded: never the unavailable
    fs.record(1, UNAVAILABLE, now=clock.t + 3)
    assert fs.choose(now=clock.t + 3) == 2
    fs.record(2, UNAVAILABLE, now=clock.t + 3)
    assert fs.choose(now=clock.t + 3) == 2  # all unavailable: the last, and the turn fails as today


def test_one_unavailable_error_is_enough_and_ignores_dwell() -> None:
    fs, clock = make()
    misses(fs, 0, clock.t)
    change = fs.record(0, UNAVAILABLE, now=clock.t + 3)
    assert change is not None and (change.old, change.new) == (DEGRADED, UNAV)
    assert change.stats is None


def test_a_single_entry_list_always_answers_on_it() -> None:
    fs, clock = make(Settings.for_models([H55]))
    misses(fs, 0, clock.t)
    assert fs.choose(now=clock.t + 3) == 0
    fs.record(0, UNAVAILABLE, now=clock.t + 3)
    assert fs.choose(now=clock.t + 3) == 0
    assert fs.next_after(0, tried={0}, now=clock.t + 3) is None


# --- request errors --------------------------------------------------------------------


def test_request_errors_never_change_state() -> None:
    fs, clock = make()
    for k in range(40):
        assert fs.record(0, Outcome.REQUEST_ERROR, now=clock.t + k) is None
    assert fs.stats(0, now=clock.t + 40).calls == 0
    assert fs.state(0, now=clock.t + 40) is HEALTHY
    fs.record(0, Outcome.ABANDONED, now=clock.t + 41)
    fs.record(0, Outcome.ABANDONED, now=clock.t + 42)
    fs.record(0, Outcome.REQUEST_ERROR, now=clock.t + 43)  # does not break the run of misses
    fs.record(0, Outcome.ABANDONED, now=clock.t + 44)
    assert fs.state(0, now=clock.t + 44) is DEGRADED


# --- next entry within a turn ----------------------------------------------------------


def test_next_after_prefers_a_healthy_entry_in_rank() -> None:
    fs, clock = make()
    assert fs.next_after(0, tried={0}, now=clock.t) == 1
    misses(fs, 1, clock.t)
    assert fs.next_after(0, tried={0}, now=clock.t + 3) == 2  # skip the degraded one while one is healthy
    misses(fs, 2, clock.t)
    assert fs.next_after(0, tried={0}, now=clock.t + 3) == 1  # none healthy left: the next usable in rank
    fs.record(1, UNAVAILABLE, now=clock.t + 3)
    assert fs.next_after(0, tried={0}, now=clock.t + 3) == 2
    fs.record(2, UNAVAILABLE, now=clock.t + 3)
    assert fs.next_after(0, tried={0}, now=clock.t + 3) is None
    assert fs.next_after(2, tried={0, 2}, now=clock.t + 3) is None


def test_next_after_goes_back_to_an_earlier_degraded_entry() -> None:
    """The next entry in rank, not in list position: with entry 0 degraded (so the turn started on
    1) and entry 1 failing, the turn moves to 0 rather than failing or skipping it."""
    fs, clock = make(Settings.for_models([H55, S55]))
    misses(fs, 0, clock.t)
    assert fs.choose(now=clock.t + 3) == 1
    fs.record(1, UNAVAILABLE, now=clock.t + 3)
    assert fs.next_after(1, tried={1}, now=clock.t + 3) == 0
    assert fs.next_after(0, tried={1, 0}, now=clock.t + 3) is None  # each entry once per turn


def test_next_after_ranks_degraded_entries_by_p90_and_never_repeats() -> None:
    fs, clock = make()
    fleet = FleetView(states=(DEGRADED, HEALTHY, DEGRADED), p90_s=(5.0, 1.0, 2.0))
    assert fs.choose(now=clock.t, fleet=fleet) == 1
    assert fs.next_after(1, tried={1}, now=clock.t, fleet=fleet) == 2  # the lower p90 first
    assert fs.next_after(2, tried={1, 2}, now=clock.t, fleet=fleet) == 0
    assert fs.next_after(0, tried={1, 2, 0}, now=clock.t, fleet=fleet) is None
    # a deadline miss leaves the failed entry usable: ``tried`` still keeps it out
    assert fs.next_after(1, tried=(), now=clock.t, fleet=fleet) == 2


# --- fleet verdict ------------------------------------------------------------------


def test_choose_follows_the_fleet_verdict() -> None:
    fs, clock = make()
    fleet = FleetView(states=(DEGRADED, HEALTHY, HEALTHY), p90_s=(5.0, 1.0, 2.0))
    assert fs.choose(now=clock.t, fleet=fleet) == 1
    assert fs.next_after(1, tried={1}, now=clock.t, fleet=fleet) == 2
    all_bad = FleetView(states=(DEGRADED, DEGRADED, DEGRADED), p90_s=(5.0, 3.0, 4.0))
    assert fs.choose(now=clock.t, fleet=all_bad) == 1  # the fleet's p90 decides
    assert fs.choose(now=clock.t, fleet=FleetView(states=(UNAV, HEALTHY, HEALTHY))) == 1


def test_this_calls_evidence_overrides_a_healthy_fleet_verdict() -> None:
    fs, clock = make()
    healthy = FleetView(states=(HEALTHY,) * 3)
    misses(fs, 0, clock.t)
    assert fs.choose(now=clock.t + 3, fleet=healthy) == 1
    fs.record(1, UNAVAILABLE, now=clock.t + 3)
    assert fs.choose(now=clock.t + 3, fleet=healthy) == 2


def test_recovery_follows_the_fleet_verdict() -> None:
    fs, clock = make()
    t = clock.t
    misses(fs, 0, t)
    degraded = FleetView(states=(DEGRADED, HEALTHY, HEALTHY))
    # The local mark lapses after the dwell, but the fleet still says degraded: still skipped.
    assert fs.choose(now=t + SETTINGS.dwell_s + 10, fleet=degraded) == 1
    # The poller's pings recovered it: the next fleet verdict brings it back.
    assert fs.choose(now=t + SETTINGS.dwell_s + 40, fleet=FleetView(states=(HEALTHY,) * 3)) == 0


def test_a_fleet_view_must_cover_every_entry() -> None:
    fs, clock = make()
    with pytest.raises(ValueError):
        fs.choose(now=clock.t, fleet=FleetView(states=(HEALTHY,)))
    with pytest.raises(ValueError):
        FleetView(states=(HEALTHY, HEALTHY), p90_s=(1.0,))


# --- fallback expiry of local marks and dwell ---------------------------------


def test_a_local_degraded_mark_lapses_after_the_dwell() -> None:
    fs, clock = make()
    t = clock.t
    misses(fs, 0, t)
    marked = t + 2
    assert fs.state(0, now=marked + SETTINGS.dwell_s - EPS) is DEGRADED
    assert fs.choose(now=marked + SETTINGS.dwell_s - EPS) == 1
    changes = fs.advance(now=marked + SETTINGS.dwell_s)
    assert [(c.entry, c.old, c.new, c.at_s) for c in changes] == [(0, DEGRADED, HEALTHY, marked + SETTINGS.dwell_s)]
    assert fs.choose(now=marked + SETTINGS.dwell_s) == 0  # tried again, under the deadline's protection


def test_a_lapsed_entry_starts_a_fresh_window_and_can_be_marked_again_at_once() -> None:
    fs, clock = make()
    t = clock.t
    misses(fs, 0, t)
    back = t + 2 + SETTINGS.dwell_s
    assert fs.stats(0, now=back).calls == 0
    fs.record(0, Outcome.ABANDONED, now=back + 1)
    fs.record(0, Outcome.ABANDONED, now=back + 2)
    assert fs.state(0, now=back + 2) is HEALTHY  # the old misses don't count again
    fs.record(0, Outcome.ABANDONED, now=back + 3)
    assert fs.state(0, now=back + 3) is DEGRADED  # still slow: marked again, a dwell after the last mark


def test_recovery_is_one_rank_at_a_time() -> None:
    fs, clock = make()
    t = clock.t
    misses(fs, 1, t)  # the secondary first
    misses(fs, 0, t + 10)  # then the primary
    first = t + 12 + SETTINGS.dwell_s
    assert states(fs, first - EPS)[:2] == [DEGRADED, DEGRADED]  # secondary waits for the primary
    assert states(fs, first)[:2] == [HEALTHY, DEGRADED]
    assert fs.state(1, now=first + SETTINGS.dwell_s - EPS) is DEGRADED
    assert fs.state(1, now=first + SETTINGS.dwell_s) is HEALTHY


def test_lapses_do_not_depend_on_when_they_are_looked_at() -> None:
    a, clock = make()
    b, _ = make()
    t = clock.t
    for fs in (a, b):
        misses(fs, 1, t)
        misses(fs, 0, t + 10)
    for k in range(0, 500, 7):
        a.advance(now=t + k)  # looked at often
    late = b.advance(now=t + 500)  # looked at once
    assert states(a, t + 500) == states(b, t + 500) == [HEALTHY] * 3
    assert [(c.entry, c.at_s) for c in late] == [(0, t + 12 + SETTINGS.dwell_s), (1, t + 12 + 2 * SETTINGS.dwell_s)]


def test_dwell_blocks_a_new_mark_too_soon_after_the_last() -> None:
    settings = Settings.for_models([H55, H45, S55], idle_reset_s=60.0)  # idle reset sooner than dwell
    fs, clock = make(settings)
    t = clock.t
    misses(fs, 0, t)
    marked = t + 2
    later = marked + 100.0  # idle reset has cleared the mark; the dwell has not passed
    assert fs.state(0, now=later) is HEALTHY
    misses(fs, 0, later)
    assert fs.state(0, now=later + 2) is HEALTHY
    fs.record(0, Outcome.ABANDONED, now=marked + settings.dwell_s - 2)  # after another idle reset
    fs.record(0, Outcome.ABANDONED, now=marked + settings.dwell_s - EPS)
    assert fs.state(0, now=marked + settings.dwell_s - EPS) is HEALTHY
    fs.record(0, Outcome.ABANDONED, now=marked + settings.dwell_s)  # exactly the dwell: allowed
    assert fs.state(0, now=marked + settings.dwell_s) is DEGRADED


def test_a_local_unavailable_mark_never_lapses_on_a_timer() -> None:
    fs, clock = make()
    t = clock.t
    fs.record(0, UNAVAILABLE, now=t)
    for k in range(1, 20):
        fast(fs, 1, t + k * 300, 1)  # a busy task, long past the recheck interval
    assert fs.state(0, now=t + 6000) is UNAV
    assert fs.choose(now=t + 6000) == 1


def test_a_healthy_fleet_view_before_it_saw_the_mark_keeps_it() -> None:
    fs, clock = make()
    fs.record(0, UNAVAILABLE, now=clock.t)
    healthy = FleetView(states=(HEALTHY,) * 3)  # the poller hasn't seen this process's reports yet
    assert fs.advance(now=clock.t + 1, fleet=healthy) == []
    assert fs.state(0, now=clock.t + 2, fleet=healthy) is UNAV
    assert fs.choose(now=clock.t + 3, fleet=healthy) == 1


def test_the_fleet_clears_the_mark_after_unavailable_then_available() -> None:
    fs, clock = make()
    fs.record(0, UNAVAILABLE, now=clock.t)
    assert fs.choose(now=clock.t + 30, fleet=FleetView(states=(UNAV, HEALTHY, HEALTHY))) == 1
    assert fs.local_state(0, now=clock.t + 60) is UNAV  # no view: still marked
    back = FleetView(states=(HEALTHY,) * 3)  # the poller's recheck succeeded
    changes = fs.advance(now=clock.t + 900, fleet=back)
    assert [(c.entry, c.old, c.new) for c in changes] == [(0, UNAV, HEALTHY)]
    assert fs.choose(now=clock.t + 900, fleet=back) == 0


def test_degraded_in_the_fleet_after_unavailable_also_clears_the_local_mark() -> None:
    fs, clock = make()
    fs.record(0, UNAVAILABLE, now=clock.t)
    fs.choose(now=clock.t + 30, fleet=FleetView(states=(UNAV, HEALTHY, HEALTHY)))
    slow = FleetView(states=(DEGRADED, HEALTHY, HEALTHY))
    assert fs.local_state(0, now=clock.t + 60, fleet=slow) is HEALTHY
    assert fs.state(0, now=clock.t + 60, fleet=slow) is DEGRADED  # worse-of still holds


def test_a_new_local_mark_needs_the_fleet_to_show_it_unavailable_again() -> None:
    fs, clock = make()
    fs.record(0, UNAVAILABLE, now=clock.t)
    fs.advance(now=clock.t + 30, fleet=FleetView(states=(UNAV, HEALTHY, HEALTHY)))
    fs.advance(now=clock.t + 60, fleet=FleetView(states=(HEALTHY,) * 3))
    fs.record(0, UNAVAILABLE, now=clock.t + 70)  # dead again
    assert fs.state(0, now=clock.t + 80, fleet=FleetView(states=(HEALTHY,) * 3)) is UNAV


def test_unconfirmed_names_marks_until_the_fleet_shows_them() -> None:
    fs, clock = make()
    t = clock.t
    assert fs.unconfirmed(now=t) == ()
    fs.record(0, UNAVAILABLE, now=t)
    fs.record(2, UNAVAILABLE, now=t + 1)
    healthy = FleetView(states=(HEALTHY,) * 3)
    assert fs.unconfirmed(now=t + 10) == (0, 2)  # no view: report them again
    assert fs.unconfirmed(now=t + 20, fleet=healthy) == (0, 2)  # the fleet hasn't taken them yet
    assert fs.unconfirmed(now=t + 30, fleet=FleetView(states=(UNAV, HEALTHY, HEALTHY))) == (2,)
    assert fs.unconfirmed(now=t + 40, fleet=None) == (2,)  # seen once is enough
    assert fs.unconfirmed(now=t + 50, fleet=FleetView(states=(UNAV, HEALTHY, UNAV))) == ()
    fs.record(1, UNAVAILABLE, now=t + 60)
    assert fs.unconfirmed(now=t + 60) == (1,)


def test_the_backstop_drops_a_mark_the_fleet_never_confirmed() -> None:
    fs, clock = make()
    t = clock.t
    seen: list[Transition] = []
    fs.on_cleared(seen.append)
    fs.record(0, UNAVAILABLE, now=t)
    healthy = FleetView(states=(HEALTHY,) * 3)
    recheck = SETTINGS.unavailable_recheck_s
    assert fs.advance(now=t + recheck, fleet=healthy) == []  # exactly the interval: not longer
    assert fs.state(0, now=t + recheck + 1) is UNAV  # no fresh view: the mark stands, however long
    assert fs.state(0, now=t + 10 * recheck) is UNAV
    changes = fs.advance(now=t + 10 * recheck, fleet=healthy)
    assert changes == [Transition(0, UNAV, HEALTHY, t + 10 * recheck, cause="unconfirmed")] == seen
    assert fs.choose(now=t + 10 * recheck, fleet=healthy) == 0  # the next turn tests it again
    assert len(fs.health(0).window) == 0
    change = fs.record(0, UNAVAILABLE, now=t + 10 * recheck + 1)  # still dead: marked again at once
    assert change is not None and change.new is UNAV
    assert fs.unconfirmed(now=t + 10 * recheck + 2) == (0,)


def test_re_reports_stop_after_the_recheck_interval_without_a_view() -> None:
    # A bridge whose list names a model the poller's record lacks has no view at all; its re-reports
    # must stop, or every recheck the poller makes is undone by them. The mark stays.
    fs, clock = make()
    t = clock.t
    fs.record(0, UNAVAILABLE, now=t)
    recheck = SETTINGS.unavailable_recheck_s
    assert fs.unconfirmed(now=t + recheck) == (0,)
    assert fs.unconfirmed(now=t + recheck + 1) == ()
    assert fs.state(0, now=t + 10 * recheck) is UNAV
    fs.record(0, UNAVAILABLE, now=t + 10 * recheck)  # already marked: the clock doesn't restart
    assert fs.unconfirmed(now=t + 10 * recheck + 1) == ()


def test_the_backstop_counts_from_the_mark_and_waits_while_the_fleet_shows_it() -> None:
    fs, clock = make()
    t = clock.t
    recheck = SETTINGS.unavailable_recheck_s
    healthy = FleetView(states=(HEALTHY,) * 3)
    degraded = FleetView(states=(DEGRADED, HEALTHY, HEALTHY))
    fs.record(0, UNAVAILABLE, now=t)
    fs.record(0, UNAVAILABLE, now=t + recheck)  # a repeat is not a new mark
    assert fs.advance(now=t + recheck + 1, fleet=degraded) == [  # not unavailable: degraded counts too
        Transition(0, UNAV, HEALTHY, t + recheck + 1, cause="unconfirmed")]
    assert fs.state(0, now=t + recheck + 1, fleet=degraded) is DEGRADED  # worse-of
    fs.record(1, UNAVAILABLE, now=t + 2000)
    fs.advance(now=t + 2001, fleet=FleetView(states=(HEALTHY, UNAV, HEALTHY)))  # the fleet has it
    fleet_back = fs.advance(now=t + 2000 + recheck + 5, fleet=healthy)  # its recheck succeeded
    assert [(c.entry, c.cause) for c in fleet_back] == [(1, "fleet")]


def test_mark_available_returns_an_unavailable_entry() -> None:
    fs, clock = make()
    fs.record(0, UNAVAILABLE, now=clock.t)
    change = fs.mark_available(0, now=clock.t + 5)
    assert change is not None and (change.old, change.new, change.cause) == (UNAV, HEALTHY, "check")
    assert fs.choose(now=clock.t + 5) == 0
    assert fs.mark_available(0, now=clock.t + 6) is None  # already available


# --- idle reset and bounded memory ------------------------------------------------


def test_idle_reset_makes_every_entry_not_marked_unavailable_healthy() -> None:
    settings = Settings.for_models([H55, H45, S55], idle_reset_s=60.0)  # sooner than the dwell
    fs, clock = make(settings)
    t = clock.t
    misses(fs, 0, t)
    fs.record(1, UNAVAILABLE, now=t + 3)
    fs.record(2, Outcome.REQUEST_ERROR, now=t + 30)  # not traffic
    idle = t + 3 + settings.idle_reset_s
    assert fs.state(0, now=idle - EPS) is DEGRADED
    assert states(fs, idle) == [HEALTHY, UNAV, HEALTHY]  # an unavailable mark survives
    assert all(fs.stats(i, now=idle).calls == 0 for i in range(3))


def test_a_new_state_starts_healthy() -> None:
    fs, clock = make()
    assert all(fs.health(i).state is HEALTHY and fs.health(i).since_s is None for i in range(3))


# --- clock, inputs and settings --------------------------------------------------------------------


def test_the_injected_clock_is_used_when_no_time_is_given() -> None:
    fs, clock = make()
    misses(fs, 0, clock.t)
    clock.t += 3
    assert fs.choose() == 1
    clock.t += SETTINGS.dwell_s
    assert fs.choose() == 0


def test_an_abandoned_call_without_a_ttft_counts_the_deadline() -> None:
    fs, clock = make()
    fs.record(0, Outcome.ABANDONED, now=clock.t)
    assert fs.stats(0, now=clock.t).p90_s == SETTINGS.entry(0).deadline_s


def test_bad_inputs_are_refused() -> None:
    fs, clock = make()
    with pytest.raises(IndexError):
        fs.record(3, Outcome.OK, 1.0, now=clock.t)
    with pytest.raises(IndexError):
        fs.next_after(-1, tried=(), now=clock.t)
    with pytest.raises(ValueError):
        fs.record(0, "slow", now=clock.t)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fs.record(0, Outcome.OK, -1.0, now=clock.t)


def test_settings_can_be_updated_for_the_same_list() -> None:
    fs, clock = make()
    fast(fs, 0, clock.t, 30)
    fs.update_settings(Settings.for_models([H55, H45, S55], window_calls=10, min_samples=10, miss_run=2))
    assert fs.stats(0, now=clock.t + 5).calls == 10
    fs.record(0, Outcome.ABANDONED, now=clock.t + 6)
    fs.record(0, Outcome.ABANDONED, now=clock.t + 7)
    assert fs.state(0, now=clock.t + 7) is DEGRADED
    with pytest.raises(ValueError):
        fs.update_settings(Settings.for_models([H55, S55]))
