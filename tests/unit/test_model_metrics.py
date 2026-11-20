"""The bridge's EMF metrics.

``ActiveRank``, ``Failover`` and ``Unavailable`` in ``OwlBank/<stage>/Model`` with the
``[Service, Environment]`` dimensions; the stage from the fleet table's name; ``ActiveRank`` every
60 s and on change, the timer tested on asyncio and uvloop with timers that may fire early.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
from collections.abc import Callable, Coroutine
from typing import Any

import pytest
import uvloop
from hypothesis import given
from hypothesis import strategies as st

from tac_bridge import model_metrics as mm

LOOPS = {"asyncio": asyncio.new_event_loop, "uvloop": uvloop.new_event_loop}


@pytest.fixture(params=sorted(LOOPS))
def run(request: pytest.FixtureRequest) -> Callable[[Coroutine[Any, Any, Any]], Any]:
    factory = LOOPS[request.param]

    def _run(coro: Coroutine[Any, Any, Any]) -> Any:
        with asyncio.Runner(loop_factory=factory) as runner:
            return runner.run(coro)

    return _run


def lines(out: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(x) for x in out.getvalue().splitlines()]


@pytest.mark.parametrize(("env", "stage"), [
    ({"OWLBANK_TABLE_MODELSTATS": "owlbank-dev-modelstats"}, "dev"),
    ({"OWLBANK_TABLE_MODELSTATS": "owlbank-load-test-modelstats", "OWLBANK_STAGE": "other"}, "load-test"),
    ({"OWLBANK_TABLE_MODELSTATS": "custom-table", "OWLBANK_STAGE": "prod"}, "prod"),
    ({"OWLBANK_STAGE": "dev"}, "dev"),
    ({"OWLBANK_TABLE_MODELSTATS": "custom-table"}, None),
    ({"OWLBANK_STAGE": "Dev Stage"}, None),
    ({}, None),
])
def test_the_stage_comes_from_the_table_name_then_the_stage_setting(env: dict[str, str], stage: str | None) -> None:
    assert mm.metric_stage(env) == stage


def test_a_line_is_one_emf_object_with_the_alarms_names() -> None:
    out = io.StringIO()
    m = mm.ModelMetrics("dev", stream=out, wall=lambda: 1_800_000_000.25)
    m.put(mm.ACTIVE_RANK, 2)
    m.failover()
    m.unavailable()
    rank, failover, unavailable = lines(out)
    assert rank == {
        "_aws": {"Timestamp": 1_800_000_000_250, "CloudWatchMetrics": [{
            "Namespace": "OwlBank/dev/Model", "Dimensions": [["Service", "Environment"]],
            "Metrics": [{"Name": "ActiveRank", "Unit": "None"}]}]},
        "Service": "owlbank-dev-bridge", "Environment": "dev", "ActiveRank": 2}
    assert failover["Failover"] == 1 and failover["_aws"]["CloudWatchMetrics"][0]["Metrics"][0]["Unit"] == "Count"
    assert unavailable["Unavailable"] == 1


@given(st.text(), st.one_of(st.text(), st.floats(), st.booleans(), st.none()))
def test_only_the_three_names_and_integers_are_ever_written(name: str, value: Any) -> None:
    """A line holds a fixed name, the stage and an integer; anything else is dropped."""
    out = io.StringIO()
    mm.ModelMetrics("dev", stream=out).put(name, value)
    assert out.getvalue() == ""


def test_a_broken_stream_never_raises() -> None:
    class Broken(io.StringIO):
        def write(self, s: str) -> int:
            raise OSError("closed")

    mm.ModelMetrics("dev", stream=Broken()).put(mm.FAILOVER, 1)


def test_a_bad_stage_is_refused() -> None:
    with pytest.raises(ValueError):
        mm.ModelMetrics("Dev Stage")


def test_no_stage_means_no_metrics() -> None:
    assert mm.metrics_from_env({}) is None
    assert isinstance(mm.metrics_from_env({"OWLBANK_STAGE": "dev"}), mm.ModelMetrics)


def test_active_rank_is_written_on_change_only() -> None:
    out = io.StringIO()
    mon = mm.ActiveRankMonitor(mm.ModelMetrics("dev", stream=out), lambda: 0)
    for rank in (0, 0, 1, 1, 0):
        mon.observe(rank)
    assert [ln["ActiveRank"] for ln in lines(out)] == [0, 1, 0]
    assert mon.last == 0


def test_active_rank_is_written_every_interval_never_sooner(run: Any) -> None:
    """Once at start, then every interval: tick ``k`` never before ``start + k * interval`` on the
    monotonic clock (a uvloop timer that fires early is seen as early and slept again)."""
    out = io.StringIO()
    stamps: list[float] = []
    interval = 0.03
    started: list[float] = []

    async def go() -> None:
        clock = time.monotonic

        def rank() -> int:
            stamps.append(clock())
            return len(stamps) % 2

        mon = mm.ActiveRankMonitor(mm.ModelMetrics("dev", stream=out), rank, interval_s=interval, clock=clock)
        started.append(clock())
        mon.start()
        await asyncio.sleep(interval * 5.5)
        await mon.stop()

    run(go())
    assert 4 <= len(stamps) <= 7
    assert all(t >= started[0] + k * interval for k, t in enumerate(stamps))
    assert len(lines(out)) == len(stamps)  # each tick writes (the value changes every time here)


def test_an_early_waking_timer_never_ticks_early(run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a clock that lags the loop's timers (as uvloop's fire about 0.4 ms early): every tick is
    still at or after its due time on that clock."""
    stamps: list[float] = []
    interval = 0.02
    lag = 0.005
    started: list[float] = []

    async def go() -> None:
        def clock() -> float:
            return time.monotonic() - lag * (len(stamps) + 1)  # a growing lag: wakes look early

        def rank() -> int:
            stamps.append(clock())
            return 0

        mon = mm.ActiveRankMonitor(mm.ModelMetrics("dev", stream=io.StringIO()), rank, interval_s=interval,
                                   clock=clock)
        started.append(clock())
        mon.start()
        await asyncio.sleep(interval * 6)
        await mon.stop()

    run(go())
    assert len(stamps) >= 2
    assert all(t >= started[0] + k * interval for k, t in enumerate(stamps))


def test_a_failing_rank_source_skips_the_tick(run: Any) -> None:
    out = io.StringIO()

    def broken() -> int:
        raise RuntimeError("state")

    async def go() -> None:
        mon = mm.ActiveRankMonitor(mm.ModelMetrics("dev", stream=out), broken, interval_s=0.01)
        mon.start()
        await asyncio.sleep(0.05)
        await mon.stop()

    run(go())
    assert out.getvalue() == ""
