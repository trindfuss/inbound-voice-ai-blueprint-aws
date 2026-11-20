"""The bridge's cached fleet verdict and its fallback.

Against the in-memory store; the loop tests run on asyncio and on uvloop.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from owlbank.failover.core import FleetView
from owlbank.failover.fleet import ModelVerdict, State, model_tag, state_key
from owlbank.failover.rules import EntryState
from owlbank.store import MemoryStore, StoreError
from tac_bridge.fleet import READ_INTERVAL_S, STALE_AFTER_S, TABLE, FleetReader

T0 = 1_800_000_000.0
MODELS = ("us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5-20251001-v1:0",
          "us.anthropic.claude-sonnet-5-5")


class Wall:
    def __init__(self, t: float = T0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class CountingStore(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.reads = 0
        self.fail = False
        self.gate: threading.Event | None = None

    def get(self, table: str, key: Any, *, consistent: bool = True) -> Any:
        assert table == TABLE
        assert consistent is False  # half a read unit
        self.reads += 1
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail:
            raise StoreError("boom")
        return super().get(table, key, consistent=consistent)


def state(written_at_s: float, *, states: tuple[EntryState, ...] = (EntryState.HEALTHY, EntryState.DEGRADED,
                                                                         EntryState.HEALTHY),
          every: int = 4, version: int = 3, ids: tuple[str, ...] = MODELS) -> State:
    models = tuple(ModelVerdict(index=i, model=model_tag(ids[i]), state=s, rank=i, since_s=None, p50_s=1.0,
                                p90_s=1.5 + i, error_rate=0.0) for i, s in enumerate(states))
    return State(version=version, written_at_s=written_at_s, models=models, pings=None, report_every=every)


def put(store: MemoryStore, st: State) -> None:
    store.put(TABLE, st.to_item())


@pytest.fixture(params=["asyncio", "uvloop"])
def run(request: pytest.FixtureRequest) -> Callable[[Callable[[], Coroutine[Any, Any, Any]]], Any]:
    if request.param == "uvloop":
        uvloop = pytest.importorskip("uvloop")
        factory: Callable[[], asyncio.AbstractEventLoop] = uvloop.new_event_loop
    else:
        factory = asyncio.new_event_loop

    def _run(fn: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
        with asyncio.Runner(loop_factory=factory) as runner:
            return runner.run(fn())

    return _run


def reader(store: MemoryStore, wall: Wall, **kw: Any) -> FleetReader:
    kw.setdefault("model_ids", MODELS)
    return FleetReader(store, wall=wall, **kw)


def test_defaults() -> None:
    assert (READ_INTERVAL_S, STALE_AFTER_S) == (30.0, 180.0)


def test_serves_the_cached_record_without_reading_per_turn() -> None:
    store, wall = CountingStore(), Wall()
    put(store, state(T0))
    r = reader(store, wall)
    assert r.view() is None  # nothing read yet: in-process picture
    assert r.report_every() == 1
    asyncio.run(r.refresh())
    for _ in range(50):
        assert r.view() == FleetView(states=(EntryState.HEALTHY, EntryState.DEGRADED, EntryState.HEALTHY),
                                     p90_s=(1.5, 2.5, 3.5))
    assert store.reads == 1
    assert r.report_every() == 4


def test_falls_back_when_missing_old_or_unreadable_and_recovers() -> None:
    store, wall = CountingStore(), Wall()
    r = reader(store, wall)
    asyncio.run(r.refresh())  # missing
    assert r.view() is None

    put(store, state(T0))
    asyncio.run(r.refresh())
    assert r.view() is not None

    wall.t = T0 + STALE_AFTER_S + 0.001  # old: the poller stopped writing
    assert r.view() is None
    wall.t = T0 + STALE_AFTER_S  # at the limit, still used
    assert r.view() is not None

    item = state(T0).to_item()
    item["models"] = "garbage"
    store.put(TABLE, item)
    asyncio.run(r.refresh())  # unreadable
    assert r.view() is None
    assert r.report_every() == 4  # the last weight read is kept

    put(store, state(T0 + 30, every=9))  # the record returns
    wall.t = T0 + 31
    asyncio.run(r.refresh())
    assert r.view() is not None
    assert r.report_every() == 9

    store.delete(TABLE, state_key())  # missing again
    asyncio.run(r.refresh())
    assert r.view() is None


def test_a_failed_read_keeps_the_last_good_record_until_it_goes_stale() -> None:
    store, wall = CountingStore(), Wall()
    put(store, state(T0))
    r = reader(store, wall)
    asyncio.run(r.refresh())
    store.fail = True
    wall.t = T0 + 60
    asyncio.run(r.refresh())  # never raises
    assert r.view() is not None
    wall.t = T0 + STALE_AFTER_S + 1
    assert r.view() is None


def test_a_record_for_another_list_is_not_used() -> None:
    store, wall = CountingStore(), Wall()
    put(store, state(T0, states=(EntryState.HEALTHY, EntryState.HEALTHY)))  # no verdict for the third
    r = reader(store, wall)
    asyncio.run(r.refresh())
    assert r.view() is None


def test_the_verdict_is_matched_by_model_not_by_place() -> None:
    a, b, c = MODELS
    store, wall = CountingStore(), Wall()
    put(store, state(T0, states=(EntryState.UNAVAILABLE, EntryState.DEGRADED), ids=(a, b)))
    swapped = reader(store, wall, model_ids=(b, a))
    asyncio.run(swapped.refresh())
    assert swapped.view() == FleetView(states=(EntryState.DEGRADED, EntryState.UNAVAILABLE), p90_s=(2.5, 1.5))
    other = reader(store, wall, model_ids=(c, b))  # C has no verdict: no view (the in-process picture)
    asyncio.run(other.refresh())
    assert other.view() is None
    fewer = reader(store, wall, model_ids=(b,))  # a record naming more models still serves this list's
    asyncio.run(fewer.refresh())
    assert fewer.view() == FleetView(states=(EntryState.DEGRADED,), p90_s=(2.5,))


def test_a_record_from_the_far_future_is_not_used() -> None:
    store, wall = CountingStore(), Wall()
    put(store, state(T0 + STALE_AFTER_S + 5))  # a skewed clock
    r = reader(store, wall)
    asyncio.run(r.refresh())
    assert r.view() is None


def test_an_out_of_range_weight_is_clamped() -> None:
    store, wall = CountingStore(), Wall()
    put(store, state(T0, every=50_000))
    r = reader(store, wall)
    asyncio.run(r.refresh())
    assert r.report_every() == 10_000


def test_a_slow_read_never_blocks_the_caller(run: Any) -> None:
    store, wall = CountingStore(), Wall()
    put(store, state(T0))
    r = reader(store, wall, read_timeout_s=0.1)

    async def scenario() -> None:
        await r.refresh()
        store.gate = threading.Event()
        t = time.monotonic()
        await r.refresh()  # times out
        assert time.monotonic() - t < 1.0
        assert r.view() is not None  # the cached record still serves
        await r.refresh()  # still in flight: skipped, no second thread
        assert store.reads == 2
        store.gate.set()
        await r.stop()

    run(scenario)


def test_the_background_read_runs_at_once_then_every_interval(run: Any) -> None:
    store, wall = CountingStore(), Wall()
    put(store, state(T0))
    r = reader(store, wall, interval_s=0.05)

    async def scenario() -> None:
        started = time.monotonic()
        r.start()
        while r.view() is None and time.monotonic() - started < 1.0:
            await asyncio.sleep(0.005)
        assert r.view() is not None  # the first read came at once, not after an interval
        while time.monotonic() - started < 0.22:
            await asyncio.sleep(0.01)
        await r.stop()
        elapsed = time.monotonic() - started
        # never more often than once an interval (plus the read at start)
        assert 2 <= store.reads <= 1 + int(elapsed / 0.05)

    run(scenario)


def test_entries_validated() -> None:
    for bad in ((), (*MODELS, "a.fourth"), (MODELS[0], MODELS[0])):
        with pytest.raises(ValueError, match="model_ids"):
            FleetReader(MemoryStore(), model_ids=bad)
