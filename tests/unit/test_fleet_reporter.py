"""The bridge's fleet reporter.

Against the in-memory store. Every timed test runs on asyncio and on uvloop (the production loop), and
the flush loop is checked against its own clock, never against a timer having fired.
"""

from __future__ import annotations

import asyncio
import random
import re
import threading
import time
from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from owlbank.failover.classify import UNAVAILABLE, Outcome
from owlbank.failover.fleet import ITEM_LIMIT_BYTES, SAMPLES_PK, Bucket, item_size, model_tag, sample_sort_key
from owlbank.failover.histogram import BIN_ATTRS, NUM_BINS, merge
from owlbank.store import MemoryStore, StoreError
from tac_bridge.fleet import FLUSH_INTERVAL_S, TABLE, FleetReporter

T0 = 1_800_000_000.0  # a bucket start
MODELS = ("us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5-20251001-v1:0",
          "us.anthropic.claude-sonnet-5-5")
TAGS = tuple(model_tag(m) for m in MODELS)
ALLOWED_ATTRS = {"s", "sk", "n", "raw", "err", "miss", "unav", "expires_at", *BIN_ATTRS}


# ---------------------------------------------------------------------------------------------- rig


class CountingStore(MemoryStore):
    """A memory store that counts sample writes, can fail them, or block them until released."""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[tuple[dict[str, Any], dict[str, int], dict[str, Any]]] = []
        self.fail = False
        self.gate: threading.Event | None = None

    def add_counters(self, table: str, key: Any, counters: Any, *, defaults: Any = None) -> None:
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail:
            raise StoreError("boom")
        self.writes.append((dict(key), dict(counters), dict(defaults or {})))
        super().add_counters(table, key, counters, defaults=defaults)

    def samples(self) -> list[dict[str, Any]]:
        return self.query_range(TABLE, {"s": SAMPLES_PK}, low="0", high="9~")


class Wall:
    def __init__(self, t: float = T0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def reporter(store: Any, *, entries: int = 3, every: int = 1, rng: Callable[[], float] | None = None,
             wall: Callable[[], float] | None = None, **kw: Any) -> FleetReporter:
    return FleetReporter(store, model_ids=MODELS[:entries], report_every=lambda: every,
                         rng=rng or (lambda: 0.0), wall=wall or Wall(), **kw)


@pytest.fixture(params=["asyncio", "uvloop"])
def run(request: pytest.FixtureRequest) -> Callable[[Callable[[], Coroutine[Any, Any, Any]]], Any]:
    """Run a coroutine on a fresh asyncio or uvloop loop."""
    if request.param == "uvloop":
        uvloop = pytest.importorskip("uvloop")
        factory: Callable[[], asyncio.AbstractEventLoop] = uvloop.new_event_loop
    else:
        factory = asyncio.new_event_loop

    def _run(fn: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
        with asyncio.Runner(loop_factory=factory) as runner:
            return runner.run(fn())

    return _run


def totals(store: CountingStore, index: int) -> Bucket:
    items = [Bucket.from_item(i) for i in store.samples() if i["sk"].endswith(f"#{TAGS[index]}")]
    return Bucket(start=0, model=TAGS[index], calls=sum(b.calls for b in items), raw=sum(b.raw for b in items),
                  errors=sum(b.errors for b in items), misses=sum(b.misses for b in items),
                  unavailable=sum(b.unavailable for b in items), hist=merge(b.hist for b in items))


# ---------------------------------------------------------------------------------------------- writes


def test_intervals_per_hosting() -> None:
    assert FLUSH_INTERVAL_S == {"fargate": 10.0, "agentcore": 30.0}


def test_one_write_per_model_per_flush_and_item_under_one_write_unit(run: Any) -> None:
    store = CountingStore()
    rep = reporter(store)
    for i in range(500):  # every bin, every outcome, on two models
        rep.record(0, Outcome.OK, (i % (NUM_BINS + 4)) * 0.25)
        rep.record(1, [Outcome.OK, Outcome.ABANDONED, Outcome.THROTTLED, Outcome.ERROR, Outcome.TIMEOUT][i % 5],
                   4.0)
    rep.record(1, UNAVAILABLE)
    assert run(rep.flush) == 2
    assert len(store.writes) == 2
    for item in store.samples():
        assert item_size(item) < ITEM_LIMIT_BYTES
        assert set(item) <= ALLOWED_ATTRS
    assert run(rep.flush) == 0  # nothing pending: no write
    assert len(store.writes) == 2


def test_item_key_bucket_and_ttl() -> None:
    store = CountingStore()
    wall = Wall(T0 + 3.7)
    rep = reporter(store, wall=wall)
    rep.record(2, Outcome.OK, 1.0)
    wall.t = T0 + 14.2  # the newest call is in the next bucket
    rep.record(2, Outcome.OK, 1.0)
    asyncio.run(rep.flush())
    key, counters, defaults = store.writes[0]
    assert key == {"s": SAMPLES_PK, "sk": sample_sort_key(int(T0) + 10, TAGS[2])}
    assert defaults == {"expires_at": int(T0) + 10 + 900}
    assert counters == {"n": 2, "raw": 2, "h4": 2}


def test_outage_burst_is_one_error_write_per_process_per_flush() -> None:
    store = CountingStore()
    rep = reporter(store, every=50, rng=lambda: 0.99)  # no OK call sampled
    for _ in range(2000):
        rep.record(0, Outcome.THROTTLED)
        rep.record(0, Outcome.ABANDONED, 4.0)
        rep.record(0, UNAVAILABLE)
        rep.record(0, Outcome.OK, 0.5)
    asyncio.run(rep.flush())
    assert len(store.writes) == 1
    b = totals(store, 0)
    # errors, misses and unavailable detections are always counted, with weight 1
    assert (b.errors, b.misses, b.unavailable, b.calls, b.raw) == (2000, 2000, 2000, 4000, 2000)


def test_request_errors_are_never_counted() -> None:
    store = CountingStore()
    rep = reporter(store)
    for _ in range(10):
        rep.record(0, Outcome.REQUEST_ERROR)
    assert asyncio.run(rep.flush()) == 0
    assert store.writes == []


def test_sampling_weight_applies_to_ok_calls_only() -> None:
    store = CountingStore()
    draws = iter([0.05, 0.5, 0.15, 0.09])  # 1 in 10: the first and the last are sampled
    rep = reporter(store, every=10, rng=lambda: next(draws))
    for _ in range(4):
        rep.record(0, Outcome.OK, 1.1)
    rep.record(0, Outcome.ERROR)
    asyncio.run(rep.flush())
    b = totals(store, 0)
    assert (b.calls, b.raw, b.errors, b.hist.total) == (21, 2, 1, 20)


def test_weighted_totals_unbiased_under_a_changing_report_prob() -> None:
    store = CountingStore()
    rng = random.Random(7)
    every = {"k": 1}
    rep = FleetReporter(store, model_ids=MODELS[:1], report_every=lambda: every["k"], rng=rng.random,
                        wall=Wall())
    calls = 0
    for k in (1, 3, 10, 50, 7, 1, 100, 2):
        every["k"] = k
        for _ in range(6000):
            rep.record(0, Outcome.OK, 0.8)
            calls += 1
        asyncio.run(rep.flush())
    b = totals(store, 0)
    assert b.calls == b.hist.total
    assert abs(b.calls - calls) / calls < 0.05
    assert b.raw < calls  # sampling did reduce what was reported


def test_bad_inputs_never_raise_into_the_caller() -> None:
    store = CountingStore()
    rep = reporter(store, entries=2)
    rep.record(5, Outcome.OK, 1.0)  # no such entry
    rep.record(-1, Outcome.OK, 1.0)
    rep.record(0, "slow", 1.0)  # type: ignore[arg-type]
    rep.record(0, Outcome.OK, None)
    rep.record(0, Outcome.OK, float("nan"))
    rep.record(0, Outcome.OK, -1.0)
    rep.record(0, Outcome.ABANDONED, None)
    assert asyncio.run(rep.flush()) == 0

    broken = reporter(store, entries=1)
    broken._report_every = lambda: 1 // 0  # a broken source of the weight
    broken.record(0, Outcome.OK, 1.0)  # dropped, not raised


def test_a_weight_out_of_range_is_clamped() -> None:
    store = CountingStore()
    rep = reporter(store, entries=1, every=10**9)
    rep.record(0, Outcome.OK, 1.0)  # rng 0.0: sampled
    asyncio.run(rep.flush())
    assert totals(store, 0).calls == 10_000


# ---------------------------------------------------------------------------------------------- failures


def test_a_failed_write_is_dropped_and_later_counters_survive(run: Any) -> None:
    store = CountingStore()
    rep = reporter(store)
    rep.record(0, Outcome.OK, 1.0)
    store.fail = True
    assert run(rep.flush) == 0  # dropped, not raised
    store.fail = False
    rep.record(0, Outcome.ERROR)
    assert run(rep.flush) == 1
    b = totals(store, 0)
    assert (b.calls, b.errors) == (1, 1)  # only the dropped flush's call is lost


def test_one_failed_model_does_not_drop_the_others() -> None:
    class FailModelOne(CountingStore):
        def add_counters(self, table: str, key: Any, counters: Any, *, defaults: Any = None) -> None:
            if key["sk"].endswith(f"#{TAGS[1]}"):
                raise StoreError("boom")
            super().add_counters(table, key, counters, defaults=defaults)

    store = FailModelOne()
    rep = reporter(store)
    for i in range(3):
        rep.record(i, Outcome.OK, 1.0)
    assert asyncio.run(rep.flush()) == 2


def test_a_slow_write_never_blocks_the_caller(run: Any) -> None:
    store = CountingStore()
    gate = store.gate = threading.Event()
    rep = reporter(store, write_timeout_s=0.1)

    async def scenario() -> None:
        rep.record(0, Outcome.OK, 1.0)
        t = time.monotonic()
        assert await rep.flush() == 0  # timed out: dropped
        assert time.monotonic() - t < 1.0
        # meanwhile the loop and the call path run freely
        t = time.monotonic()
        rep.record(0, Outcome.ERROR)
        await asyncio.sleep(0.01)
        assert time.monotonic() - t < 0.5
        # the write is still in flight: the next flush takes nothing and keeps the counters
        assert await rep.flush() == 0
        assert rep._pending[0].counters.errors == 1
        gate.set()
        for _ in range(100):
            if not rep._io.busy:
                break
            await asyncio.sleep(0.01)
        assert await rep.flush() == 1
        await rep.stop(final_flush=False)

    run(scenario)
    b = totals(store, 0)
    assert (b.calls, b.errors) == (2, 1)  # the slow write did land later; never retried


# ---------------------------------------------------------------------------------------------- the loop


def early_timers(loop: asyncio.AbstractEventLoop, early_s: float) -> None:
    """Every ``call_later`` fires ``early_s`` before its due time, as uvloop's millisecond timers can."""
    original = loop.call_later

    def call_later(delay: float, callback: Any, *args: Any, **kwargs: Any) -> asyncio.TimerHandle:
        return original(max(0.0, delay - early_s), callback, *args, **kwargs)

    loop.call_later = call_later  # type: ignore[method-assign, assignment]


@pytest.mark.parametrize("early_s", [0.0, 0.03])
def test_background_flush_never_comes_before_its_interval(run: Any, early_s: float) -> None:
    store = CountingStore()
    times: list[float] = []

    class Timed(FleetReporter):
        async def flush(self) -> int:
            times.append(time.monotonic())
            return await super().flush()

    rep = Timed(store, model_ids=MODELS[:1], interval_s=0.05, wall=Wall())

    made = 0

    async def scenario() -> float:
        nonlocal made
        early_timers(asyncio.get_running_loop(), early_s)
        started = time.monotonic()
        rep.start()
        rep.start()  # idempotent
        end = started + 0.3
        while time.monotonic() < end:  # by the clock: the test's own sleeps fire early too
            rep.record(0, Outcome.OK, 1.0)
            made += 1
            await asyncio.sleep(0.03)
        await rep.stop()
        return started

    started = run(scenario)
    background = times[:-1]  # the last one is stop's final flush
    assert len(background) >= 3
    # the n-th background flush never comes before n intervals have passed
    assert all(t >= started + (n + 1) * 0.05 for n, t in enumerate(background))
    assert totals(store, 0).calls == made  # nothing lost; stop's final flush wrote the rest


def test_stop_flushes_at_call_end_and_never_raises(run: Any) -> None:
    store = CountingStore()
    rep = reporter(store, hosting="agentcore")

    async def scenario() -> None:
        rep.start()
        rep.record(1, Outcome.OK, 2.0)
        store.fail = True
        await rep.stop()  # the write fails: dropped, not raised

    run(scenario)
    assert store.writes == []


# ---------------------------------------------------------------------------------------------- no caller data


def test_no_text_or_digits_beyond_the_counters() -> None:
    store = CountingStore()
    rep = reporter(store)
    for i, o in enumerate([Outcome.OK, Outcome.ABANDONED, Outcome.ERROR, Outcome.THROTTLED, Outcome.TIMEOUT]):
        rep.record(i % 3, o, 1.25 * i)
    rep.record(2, UNAVAILABLE)
    asyncio.run(rep.flush())
    for item in store.samples():
        assert set(item) <= ALLOWED_ATTRS
        assert item["s"] == SAMPLES_PK
        assert re.fullmatch(r"\d{10}#[0-9a-f]{8}", item["sk"]) and item["sk"][11:] in TAGS
        assert all(isinstance(v, int) and not isinstance(v, bool) for k, v in item.items() if k not in {"s", "sk"})


def test_entries_validated() -> None:
    for bad in ((), (*MODELS, "a.fourth"), MODELS[0], (MODELS[0], MODELS[0])):
        with pytest.raises(ValueError, match="model_ids"):
            FleetReporter(MemoryStore(), model_ids=bad)


# ---------------------------------------------------------------------------------------------- re-reports


def test_unconfirmed_entries_are_reported_again_on_every_flush(run: Any) -> None:
    store = CountingStore()
    wall = Wall(T0 + 3)
    held: list[int] = [2]
    rep = reporter(store, wall=wall, unconfirmed=lambda: held)
    rep.record(2, UNAVAILABLE)  # the original report: counted once, not twice, in this flush
    assert run(rep.flush) == 1
    assert store.writes[-1][0]["sk"] == sample_sort_key(int(T0), TAGS[2]) and store.writes[-1][1] == {"unav": 1}
    store.fail = True
    wall.t = T0 + 13
    assert run(rep.flush) == 0  # this flush's write is dropped ...
    store.fail = False
    wall.t = T0 + 23
    assert run(rep.flush) == 1  # ... and the next one reports it again, dated now
    assert store.writes[-1][0]["sk"] == sample_sort_key(int(T0) + 20, TAGS[2]) and store.writes[-1][1] == {"unav": 1}
    held.clear()  # the fleet shows it unavailable: no more re-reports
    assert run(rep.flush) == 0
    assert totals(store, 2).unavailable == 2


def test_a_re_report_joins_the_models_other_counters() -> None:
    store = CountingStore()
    rep = reporter(store, unconfirmed=lambda: (0, 0))
    rep.record(0, Outcome.ERROR)
    assert asyncio.run(rep.flush()) == 1
    assert store.writes[0][1] == {"n": 1, "err": 1, "unav": 1}


def test_a_broken_unconfirmed_source_never_fails_a_flush() -> None:
    store = CountingStore()
    for source in (lambda: 1 // 0, lambda: [7], lambda: ["0"], lambda: None):
        rep = reporter(store, unconfirmed=source)
        rep.record(0, Outcome.OK, 1.0)
        assert asyncio.run(rep.flush()) == 1  # the call's own counters still go
    assert totals(store, 0).unavailable == 0
