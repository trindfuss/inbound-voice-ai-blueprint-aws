"""Model failover, property tests of the fleet reporter.

* The stored counters after any sequence of calls, weights and flushes equal the sum of the recorded
  calls, with each sampled OK call counted at its weight and everything else at weight 1.
* A dropped write loses that flush's counters and nothing else.

Both run on asyncio and on uvloop (the production loop).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
import uvloop
from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.failover.classify import UNAVAILABLE, Classification, Outcome
from owlbank.failover.fleet import SAMPLES_PK, Bucket, model_tag
from owlbank.failover.histogram import Histogram, merge
from owlbank.store import MemoryStore, StoreError
from tac_bridge.fleet import TABLE, FleetReporter

T0 = 1_800_000_000.0
MODELS = ("us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5-20251001-v1:0",
          "us.anthropic.claude-sonnet-5-5")
LOOPS = {"asyncio": asyncio.new_event_loop, "uvloop": uvloop.new_event_loop}

OUTCOMES: list[Classification] = [*Outcome, UNAVAILABLE]


@dataclass(frozen=True)
class Call:
    index: int
    outcome: Classification
    ttft_s: float
    draw: float  # the sampling draw this call gets
    every: int  # the weight in force
    dt: float  # wall time before it


calls = st.builds(Call, index=st.integers(0, 2), outcome=st.sampled_from(OUTCOMES),
                  ttft_s=st.floats(0, 12, allow_nan=False), draw=st.floats(0, 1, exclude_max=True),
                  every=st.sampled_from([1, 2, 3, 10, 100]), dt=st.floats(0, 25))
# a step is a call or a flush (whose write fails or not)
steps = st.lists(st.one_of(calls, st.sampled_from(["flush", "fail"])), max_size=60)


class FlakyStore(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail = False

    def add_counters(self, table: str, key: Any, counters: Any, *, defaults: Any = None) -> None:
        if self.fail:
            raise StoreError("dropped")
        super().add_counters(table, key, counters, defaults=defaults)


@dataclass
class Expected:
    calls: int = 0
    raw: int = 0
    errors: int = 0
    misses: int = 0
    unavailable: int = 0
    hist: Histogram = Histogram.empty()

    def add(self, c: Call) -> None:
        if c.outcome is Outcome.REQUEST_ERROR:
            return
        if c.outcome == UNAVAILABLE:
            self.unavailable += 1
            return
        if c.outcome in (Outcome.THROTTLED, Outcome.TIMEOUT, Outcome.ERROR):
            self.calls += 1
            self.errors += 1
            return
        if c.outcome is Outcome.OK:
            if not c.draw < 1 / c.every:
                return
            w = c.every
        else:  # abandoned: always, weight 1
            w = 1
            self.misses += 1
        self.calls += w
        self.raw += 1
        self.hist = self.hist.add(c.ttft_s, weight=w)


def stored(store: MemoryStore, index: int) -> Expected:
    bs = [Bucket.from_item(i) for i in store.query_range(TABLE, {"s": SAMPLES_PK}, low="0", high="9~")]
    bs = [b for b in bs if b.model == model_tag(MODELS[index])]
    return Expected(calls=sum(b.calls for b in bs), raw=sum(b.raw for b in bs),
                    errors=sum(b.errors for b in bs), misses=sum(b.misses for b in bs),
                    unavailable=sum(b.unavailable for b in bs), hist=merge(b.hist for b in bs))


def play(seq: list[Any], loop: str = "asyncio") -> tuple[FlakyStore, dict[int, Expected]]:
    store = FlakyStore()
    wall = {"t": T0}
    current: dict[str, Any] = {"call": None}
    rep = FleetReporter(store, model_ids=MODELS, report_every=lambda: current["call"].every,
                        rng=lambda: current["call"].draw, wall=lambda: wall["t"])
    expected = {i: Expected() for i in range(3)}
    since_flush: dict[int, Expected] = {i: Expected() for i in range(3)}

    def merge_into(target: dict[int, Expected], part: dict[int, Expected]) -> None:
        for i, e in part.items():
            t = target[i]
            t.calls += e.calls
            t.raw += e.raw
            t.errors += e.errors
            t.misses += e.misses
            t.unavailable += e.unavailable
            t.hist = t.hist + e.hist

    async def run() -> None:
        nonlocal since_flush
        for step in [*seq, "flush"]:
            if isinstance(step, Call):
                wall["t"] += step.dt
                current["call"] = step
                rep.record(step.index, step.outcome, step.ttft_s)
                since_flush[step.index].add(step)
                continue
            store.fail = step == "fail"
            await rep.flush()
            if not store.fail:  # a dropped write loses exactly what was pending, nothing more
                merge_into(expected, since_flush)
            since_flush = {i: Expected() for i in range(3)}
        store.fail = False
        await rep.stop(final_flush=False)

    with asyncio.Runner(loop_factory=LOOPS[loop]) as runner:
        runner.run(run())
    return store, expected


@pytest.mark.parametrize("loop", sorted(LOOPS))
@settings(deadline=None, max_examples=150)
@given(steps)
def test_counters_equal_the_sum_of_the_recorded_calls(loop: str, seq: list[Any]) -> None:
    store, expected = play([s for s in seq if s != "fail"], loop)
    for i in range(3):
        assert stored(store, i) == expected[i]


@pytest.mark.parametrize("loop", sorted(LOOPS))
@settings(deadline=None, max_examples=150)
@given(steps)
def test_a_dropped_write_loses_only_its_own_flush(loop: str, seq: list[Any]) -> None:
    store, expected = play(seq, loop)
    for i in range(3):
        assert stored(store, i) == expected[i]


# --- Re-reports of unconfirmed unavailable entries --------------------------------------

#: A re-report step: an unavailable call on an entry, a change of the entries the decision state
#: names as unconfirmed, or a flush (whose write fails or not).
rereport_steps = st.lists(st.one_of(
    st.tuples(st.just("unav"), st.integers(0, 2)),
    st.tuples(st.just("held"), st.frozensets(st.integers(0, 2))),
    st.tuples(st.sampled_from(["flush", "fail"]), st.just(0))), max_size=40)


@pytest.mark.parametrize("loop", sorted(LOOPS))
@settings(deadline=None, max_examples=150)
@given(rereport_steps)
def test_every_flush_reports_each_unconfirmed_entry_once(loop: str, seq: list[Any]) -> None:
    """Each flush whose write goes through adds, per entry, the unavailable detections recorded since
    the last flush, or one when there were none and the entry is unconfirmed: so while an entry stays
    unconfirmed every successful flush reports it, however many flushes before were dropped, and a
    flush never counts a re-report on top of a fresh one."""
    store = FlakyStore()
    held: list[frozenset[int]] = [frozenset()]
    rep = FleetReporter(store, model_ids=MODELS, unconfirmed=lambda: held[0], wall=lambda: T0)
    expected = [0, 0, 0]
    recorded = [0, 0, 0]

    async def run() -> None:
        for kind, arg in [*seq, ("flush", 0)]:
            if kind == "unav":
                rep.record(arg, UNAVAILABLE)
                recorded[arg] += 1
            elif kind == "held":
                held[0] = arg
            else:
                store.fail = kind == "fail"
                before = [stored(store, i).unavailable for i in range(3)]
                await rep.flush()
                for i in range(3):
                    adds = recorded[i] or (1 if i in held[0] else 0)
                    if not store.fail:
                        expected[i] += adds
                        assert stored(store, i).unavailable - before[i] == adds
                recorded[:] = [0, 0, 0]
        store.fail = False
        await rep.stop(final_flush=False)

    with asyncio.Runner(loop_factory=LOOPS[loop]) as runner:
        runner.run(run())
    assert [stored(store, i).unavailable for i in range(3)] == expected
