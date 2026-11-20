"""The in-process decision state.

:class:`FailoverState` answers "which entry for this turn" (:meth:`~FailoverState.choose`, once per turn) and
"which next, before any output" (:meth:`~FailoverState.next_after`), from what this process
has seen (:meth:`~FailoverState.record`) and, when the bridge has a fresh one, the fleet verdict
(:class:`FleetView`).

* **Trip** uses :func:`~owlbank.failover.rules.is_degraded` over each entry's count window (the
  last ``window_calls`` samples, none older than ``window_age_s``); one unambiguous error marks an entry
  unavailable at once. Request errors are not samples and change nothing.
* **Choice**: the best-ranked healthy entry; if every usable entry is degraded, the one
  with the lowest p90; an unavailable entry only when every entry is (then the last). No caller turn is
  ever sent to a degraded entry to test it, and there is no probe choice: recovery is tested by the
  poller's pings.
* **The fleet verdict** is followed unless this process's own evidence is worse: the state of an entry
  is the worse of its local mark and the fleet's state. Local evidence never makes an entry better than
  the fleet says, so recovery follows the fleet.
* **A local degraded mark lapses** after ``dwell_s`` (with or without a fresh fleet record), one rank at
  a time (the highest-ranked degraded entry first; the next one's dwell counts from that lapse).
  The entry starts a fresh window and is tried again under the deadline's protection. A lapse is not a
  verdict: a new mark may follow at once on fresh evidence, and two marks of one entry are always at
  least ``dwell_s`` apart.
* **A local unavailable mark never lapses on a timer** and survives the idle reset. It clears
  when this process has seen a fleet view mark the entry unavailable and a later one show it not
  unavailable (the poller's recheck succeeded), by :meth:`FailoverState.mark_available`, or on restart.
  A fleet view that says not unavailable before it ever said unavailable may not have seen this
  process's report yet, so the mark stands (worse-of) and :meth:`FailoverState.unconfirmed` names the
  entry, for the reporter to report it again on every flush until the fleet shows it. **Backstop:** if
  a fresh fleet view still says not unavailable more than ``unavailable_recheck_s`` after the local
  mark, the reports never took (the fleet would have marked it by then), and the mark clears: the next
  turn on the entry tests it again, and a real failure marks it again at once.
* **Idle reset**: ``idle_reset_s`` without a sample on any entry clears every window and
  every local degraded mark.

Every time rule is an explicit comparison of the times passed in (uvloop timers fire early, so
nothing here relies on a timer firing at or after its due time). Lapses take effect at their due times
whenever they are looked at, so the state depends only on what was recorded and when, not on how often
it is read. Times are monotonic seconds, non-decreasing across calls. No I/O, no lock, no logging; memory
is bounded by ``window_calls`` samples per entry. A new state is all healthy.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Collection
from dataclasses import dataclass, field

from owlbank.failover.classify import UNAVAILABLE, Classification, Outcome
from owlbank.failover.rules import EntryState, Sample, Stats, dwell_elapsed, is_degraded
from owlbank.failover.settings import Settings

_SEVERITY = {EntryState.HEALTHY: 0, EntryState.DEGRADED: 1, EntryState.UNAVAILABLE: 2}


def _worse(a: EntryState, b: EntryState) -> EntryState:
    return a if _SEVERITY[a] >= _SEVERITY[b] else b


@dataclass(frozen=True)
class FleetView:
    """The fleet verdict as the bridge's reader hands it over: per entry, in list order, its state
    and its fleet p90 TTFT when known. The reader passes ``None`` instead when the record is missing,
    unreadable or stale; freshness is not judged here."""

    states: tuple[EntryState, ...]
    p90_s: tuple[float | None, ...] = ()

    def __post_init__(self) -> None:
        if not all(isinstance(s, EntryState) for s in self.states):
            raise ValueError("fleet states must be EntryState values")
        if self.p90_s and len(self.p90_s) != len(self.states):
            raise ValueError("fleet p90s must match the states")

    def p90(self, index: int) -> float | None:
        return self.p90_s[index] if self.p90_s else None


@dataclass(frozen=True)
class Transition:
    """A change of an entry's local state, for the caller's log line. ``stats`` is the window's
    numbers on a trip, ``None`` otherwise (never caller data)."""

    entry: int
    old: EntryState
    new: EntryState
    at_s: float
    stats: Stats | None = None
    #: Why a local unavailable mark cleared: ``"fleet"`` (the fleet's recheck), ``"unconfirmed"`` (the
    #: backstop) or ``"check"`` (:meth:`FailoverState.mark_available`); ``None`` for other changes.
    cause: str | None = None


@dataclass
class EntryHealth:
    """One entry's local picture: its count window, its local state and when that was set
    (``since_s``), and when it was last marked degraded (the dwell's reference)."""

    window: deque[Sample]
    state: EntryState = EntryState.HEALTHY
    since_s: float | None = None
    last_trip_s: float | None = None
    #: The time the dwell toward a degraded mark's lapse counts from: the mark's time, or the lapse of
    #: the entry above it if later (one rank at a time).
    lapse_from_s: float | None = field(default=None, repr=False)
    #: While locally unavailable: a fleet view has shown this entry unavailable since the mark.
    fleet_saw_unavailable: bool = field(default=False, repr=False)

    def prune(self, now_s: float, settings: Settings) -> None:
        """Drop samples older than ``window_age_s`` (an explicit comparison; exactly the age stays)."""
        while self.window and now_s - self.window[0].at_s > settings.window_age_s:
            self.window.popleft()

    def stats(self, now_s: float, settings: Settings) -> Stats:
        self.prune(now_s, settings)
        return Stats.of(self.window)

    def reset(self, now_s: float | None) -> None:
        """Healthy again with a fresh window; ``last_trip_s`` is kept (the dwell)."""
        self.window.clear()
        self.state = EntryState.HEALTHY
        self.since_s = now_s
        self.lapse_from_s = None
        self.fleet_saw_unavailable = False


class FailoverState:
    """The per-process decision state over an ordered list of one to three entries."""

    def __init__(self, settings: Settings, *, clock: Callable[[], float] = time.monotonic) -> None:
        if not isinstance(settings, Settings):
            raise ValueError("settings must be failover Settings")
        self._settings = settings
        self._clock = clock
        self._on_cleared: Callable[[Transition], None] | None = None
        self._entries = [EntryHealth(window=deque(maxlen=settings.window_calls)) for _ in settings.entries]
        self._last_sample_s: float | None = None

    # --- inputs ------------------------------------------------------------------------------------

    @property
    def settings(self) -> Settings:
        return self._settings

    def update_settings(self, settings: Settings) -> None:
        """New thresholds for the same list (the policy section is read per turn)."""
        if (not isinstance(settings, Settings)
                or [e.model_id for e in settings.entries] != [e.model_id for e in self._settings.entries]):
            raise ValueError("settings must be for the same model list")
        if settings.window_calls != self._settings.window_calls:
            for h in self._entries:
                h.window = deque(h.window, maxlen=settings.window_calls)
        self._settings = settings

    def _now(self, now: float | None) -> float:
        t = self._clock() if now is None else now
        if not math.isfinite(t):
            raise ValueError("now must be a finite time")
        return t

    def on_cleared(self, listener: Callable[[Transition], None] | None) -> None:
        """``listener`` gets every clear of a local unavailable mark by :meth:`advance` (the fleet's
        recheck or the backstop), whichever method looked, for the caller's log line. It must not raise."""
        self._on_cleared = listener

    def _index(self, index: int) -> int:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self._entries):
            raise IndexError("no such model entry")
        return index

    def _fleet(self, fleet: FleetView | None) -> FleetView | None:
        if fleet is not None and len(fleet.states) != len(self._entries):
            raise ValueError("the fleet view must cover every entry")
        return fleet

    def record(self, index: int, outcome: Classification, ttft_s: float | None = None, *,
               now: float | None = None) -> Transition | None:
        """One model call's outcome on entry ``index`` (an :class:`Outcome` or ``UNAVAILABLE``, from
        :func:`~owlbank.failover.classify.classify`), with its TTFT when it had one; an abandoned call
        without one counts the entry's deadline. Returns the local change it caused, if any."""
        i = self._index(index)
        if outcome != UNAVAILABLE and not isinstance(outcome, Outcome):
            raise ValueError("outcome must be an Outcome or UNAVAILABLE")
        if ttft_s is not None and (not math.isfinite(ttft_s) or ttft_s < 0):
            raise ValueError("ttft_s must be a finite, non-negative number")
        if outcome is Outcome.REQUEST_ERROR:
            return None  # the request's fault: not a sample, not traffic, no state
        t = self._now(now)
        self.advance(now=t)
        self._last_sample_s = t
        h, settings = self._entries[i], self._settings
        # UNAVAILABLE: one is enough; a separate state, not subject to the dwell
        if not isinstance(outcome, Outcome):
            if h.state is EntryState.UNAVAILABLE:
                return None
            old = h.state
            h.window.clear()
            h.state, h.since_s, h.lapse_from_s = EntryState.UNAVAILABLE, t, None
            h.fleet_saw_unavailable = False
            return Transition(i, old, EntryState.UNAVAILABLE, t)
        if h.state is EntryState.UNAVAILABLE:
            return None  # a turn on an unavailable entry (all are): nothing to learn
        if outcome is Outcome.ABANDONED and ttft_s is None:
            ttft_s = settings.entry(i).deadline_s
        h.window.append(Sample(at_s=t, outcome=outcome, ttft_s=ttft_s))
        if h.state is not EntryState.HEALTHY:
            return None  # degraded stays until its lapse or the fleet (no in-process recovery)
        stats = h.stats(t, settings)
        if not is_degraded(stats, settings.entry(i), settings):
            return None
        if not dwell_elapsed(since_s=h.last_trip_s, now_s=t, settings=settings):
            return None  # too soon after its last mark
        h.state, h.since_s, h.last_trip_s, h.lapse_from_s = EntryState.DEGRADED, t, t, t
        return Transition(i, EntryState.HEALTHY, EntryState.DEGRADED, t, stats)

    def mark_available(self, index: int, *, now: float | None = None) -> Transition | None:
        """Entry ``index`` is usable again by a check without caller traffic: healthy, fresh window.
        The fleet path clears marks by itself (:meth:`advance`); this is for any other such check."""
        i = self._index(index)
        t = self._now(now)
        self.advance(now=t)
        h = self._entries[i]
        if h.state is not EntryState.UNAVAILABLE:
            return None
        h.reset(t)
        return Transition(i, EntryState.UNAVAILABLE, EntryState.HEALTHY, t, cause="check")

    def unconfirmed(self, *, now: float | None = None, fleet: FleetView | None = None) -> tuple[int, ...]:
        """The entries marked unavailable here that no fleet view has shown unavailable since the mark
        (after applying ``fleet``): the reporter reports each again on every flush, so a lost
        report (a dropped flush, a poller outage longer than the window) is made good. Only for
        ``unavailable_recheck_s`` after the mark: with no view at all (a list the poller's record
        doesn't match, a long poller outage) re-reports would otherwise go on for good and undo every
        recheck the poller makes; the mark itself stays."""
        t = self._now(now)
        self.advance(now=t, fleet=fleet)
        limit = self._settings.unavailable_recheck_s
        return tuple(i for i, h in enumerate(self._entries)
                     if h.state is EntryState.UNAVAILABLE and not h.fleet_saw_unavailable
                     and (h.since_s is None or t - h.since_s <= limit))

    # --- time --------------------------------------------------------------------------------------

    def advance(self, *, now: float | None = None, fleet: FleetView | None = None) -> list[Transition]:
        """Apply the idle reset and the lapses of local degraded marks due by ``now``, at their due
        times, and what ``fleet`` says about local unavailable marks. Called by every other
        method; public so the caller can log the changes (clears of unavailable marks also go to the
        :meth:`on_cleared` listener)."""
        t = self._now(now)
        view = self._fleet(fleet)
        s = self._settings
        changes: list[Transition] = []
        if view is not None:  # Unavailable, then not: the poller's recheck succeeded
            for i, h in enumerate(self._entries):
                if h.state is not EntryState.UNAVAILABLE:
                    continue
                cause = None
                if view.states[i] is EntryState.UNAVAILABLE:
                    h.fleet_saw_unavailable = True
                elif h.fleet_saw_unavailable:
                    cause = "fleet"
                elif h.since_s is not None and t - h.since_s > s.unavailable_recheck_s:
                    cause = "unconfirmed"  # the backstop: the fleet never took this process's reports
                if cause is not None:
                    h.reset(t)
                    change = Transition(i, EntryState.UNAVAILABLE, EntryState.HEALTHY, t, cause=cause)
                    changes.append(change)
                    if self._on_cleared is not None:
                        self._on_cleared(change)
        last = self._last_sample_s
        if last is not None and t >= last + s.idle_reset_s:  # unavailable marks stay
            at = last + s.idle_reset_s
            for i, h in enumerate(self._entries):
                if h.state is EntryState.DEGRADED:
                    changes.append(Transition(i, h.state, EntryState.HEALTHY, at))
                    h.reset(at)
                else:
                    h.window.clear()
            self._last_sample_s = None
            return changes
        while True:  # degraded marks: the highest-ranked first, each a dwell after the one above
            head = next((h for h in self._entries if h.state is EntryState.DEGRADED), None)
            if head is None or head.lapse_from_s is None or t < head.lapse_from_s + s.dwell_s:
                break
            at = head.lapse_from_s + s.dwell_s
            i = self._entries.index(head)
            head.reset(at)
            changes.append(Transition(i, EntryState.DEGRADED, EntryState.HEALTHY, at))
            following = next((h for h in self._entries[i + 1:] if h.state is EntryState.DEGRADED), None)
            if following is not None and following.lapse_from_s is not None:
                following.lapse_from_s = max(following.lapse_from_s, at)
        return changes

    # --- reading -----------------------------------------------------------------------------------

    def health(self, index: int) -> EntryHealth:
        """Entry ``index``'s local picture as last updated (no lapse applied); for tests and logs."""
        return self._entries[self._index(index)]

    def stats(self, index: int, *, now: float | None = None) -> Stats:
        """The numbers of entry ``index``'s current window."""
        i = self._index(index)
        t = self._now(now)
        self.advance(now=t)
        return self._entries[i].stats(t, self._settings)

    def local_state(self, index: int, *, now: float | None = None,
                    fleet: FleetView | None = None) -> EntryState:
        i = self._index(index)
        self.advance(now=self._now(now), fleet=fleet)
        return self._entries[i].state

    def state(self, index: int, *, now: float | None = None, fleet: FleetView | None = None) -> EntryState:
        """The state :meth:`choose` acts on: the worse of the local mark and the fleet's state."""
        local = self.local_state(index, now=now, fleet=fleet)
        view = self._fleet(fleet)
        return local if view is None else _worse(local, view.states[index])

    def _p90(self, index: int, now: float, fleet: FleetView | None) -> float | None:
        if fleet is not None and fleet.p90(index) is not None:
            return fleet.p90(index)
        return self._entries[index].stats(now, self._settings).p90_s

    def _ranked(self, t: float, view: FleetView | None) -> list[int]:
        """The usable entries in rank: healthy ones in list order, then degraded ones by p90 (an
        unknown p90 last, list order breaking ties); never an unavailable one."""
        states = [self.state(i, now=t, fleet=view) for i in range(len(self._entries))]
        healthy = [i for i, st in enumerate(states) if st is EntryState.HEALTHY]

        def key(i: int) -> tuple[bool, float, int]:
            p90 = self._p90(i, t, view)
            return (p90 is None, p90 if p90 is not None else 0.0, i)

        degraded = sorted((i for i, st in enumerate(states) if st is EntryState.DEGRADED), key=key)
        return healthy + degraded

    def choose(self, *, now: float | None = None, fleet: FleetView | None = None) -> int:
        """The entry for a turn, chosen once at its start."""
        ranked = self._ranked(self._now(now), self._fleet(fleet))
        return ranked[0] if ranked else len(self._entries) - 1  # all unavailable: the last, the turn fails as today

    def next_after(self, index: int, *, tried: Collection[int], now: float | None = None,
                   fleet: FleetView | None = None) -> int | None:
        """The entry to reissue a turn to after entry ``index`` failed before any output: the
        best-ranked usable entry not yet tried in this turn, as :meth:`choose` ranks them (healthy
        first, then degraded by p90, wherever it is in the list); never an unavailable one;
        ``None`` when none is left. ``tried`` is the entries the turn has used (``index`` counts as
        tried either way), so each entry gets the turn at most once."""
        i = self._index(index)
        done = {i, *tried}
        return next((j for j in self._ranked(self._now(now), self._fleet(fleet)) if j not in done), None)


__all__ = ["EntryHealth", "EntryState", "FailoverState", "FleetView", "Transition"]
