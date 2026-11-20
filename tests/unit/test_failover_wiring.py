"""The bridge's wiring of the fleet picture and the metrics.

With the fleet table's name and two or more entries the reader, reporter and metrics are built and
attached; without the table (``modelFailover=off``) or with a list of one nothing is. The
reporter samples with the reader's weight; under AgentCore a call's end flushes its report.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Any

import pytest

from owlbank.failover import EntryState, Outcome
from owlbank.failover.fleet import SAMPLES_PK, ModelVerdict, State, model_tag, state_key
from owlbank.store import MemoryStore
from tac_bridge.failover_wiring import FailoverRuntime, log_models_resolved
from tac_bridge.fleet import TABLE
from tests.logutil import lines
from tests.unit.test_strands_failover import IDS, EntryModel, make

TABLE_NAME = "owlbank-dev-modelstats"
ENV = {"OWLBANK_TABLE_MODELSTATS": TABLE_NAME, "AWS_REGION": "us-east-1"}


def runtime(n: int = 2, env: dict[str, str] | None = None, hosting: str = "fargate",
            store: Any = None) -> tuple[Any, FailoverRuntime | None, MemoryStore, io.StringIO]:
    client = make([EntryModel() for _ in range(n)])
    store = store if store is not None else MemoryStore()
    out = io.StringIO()
    rt = FailoverRuntime.from_env(ENV if env is None else env, client, hosting=hosting,  # type: ignore[arg-type]
                                  store=store, metrics_stream=out)
    return client, rt, store, out


def samples(store: MemoryStore) -> list[dict[str, Any]]:
    return list(store.query_range(TABLE, {"s": SAMPLES_PK}, low="0", high="9"))


@pytest.mark.parametrize(("n", "env", "reason"), [
    (1, ENV, "single_entry"),
    (2, {"AWS_REGION": "us-east-1"}, "no_table"),
    (3, {"OWLBANK_TABLE_MODELSTATS": "  "}, "no_table"),
])
def test_without_the_table_or_with_one_entry_nothing_is_built(n: int, env: dict[str, str], reason: str,
                                                             caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    client, rt, _, out = runtime(n, env)
    assert rt is None
    assert client._fleet_reporter is None and client._fleet_view is None  # noqa: SLF001
    assert client._failover_events is None  # noqa: SLF001
    (ln,) = [x for x in lines(caplog.records, "tac_bridge.failover_wiring") if x["event"] == "model_fleet"]
    assert (ln["state"], ln["reason"]) == ("off", reason)
    assert out.getvalue() == ""


def test_with_the_table_the_reader_reporter_and_metrics_are_attached() -> None:
    client, rt, _, _ = runtime(3)
    assert rt is not None and rt.metrics is not None and rt.monitor is not None
    assert rt.metrics.stage == "dev"
    assert client._fleet_reporter is rt.reporter  # noqa: SLF001
    assert client._fleet_view == rt.reader.view  # noqa: SLF001


def test_no_stage_still_runs_the_fleet_without_metrics(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    client, rt, _, _ = runtime(2, {"OWLBANK_TABLE_MODELSTATS": "a-custom-name"})
    assert rt is not None and rt.metrics is None and rt.monitor is None
    assert client._failover_events is None  # noqa: SLF001
    assert any(x["event"] == "model_metrics_off" for x in lines(caplog.records, "tac_bridge.model_metrics"))


def test_the_verdict_and_the_weight_reach_the_client() -> None:
    """The reader's cached verdict drives the choice: the reporter samples with the weight
    the state record publishes."""
    import time

    client, rt, store, _ = runtime(2)
    assert rt is not None
    now = time.time()
    a, b = (model_tag(m) for m in client.settings.model_ids)
    state = State(version=3, written_at_s=now, pings=None, report_every=7, models=(
        ModelVerdict(0, a, EntryState.DEGRADED, 1, now - 5, 4.0, 5.0, 0.0),
        ModelVerdict(1, b, EntryState.HEALTHY, 0, None, 0.6, 0.8, 0.0)))
    store.put(TABLE, state.to_item())

    async def go() -> None:
        await rt.reader.refresh()

    asyncio.run(go())
    assert client.active_rank() == 1
    assert rt.reader.report_every() == 7
    assert store.get(TABLE, state_key()) is not None


@pytest.mark.parametrize("hosting", ["fargate", "agentcore"])
def test_the_report_is_flushed_at_call_end_on_agentcore_and_at_shutdown(hosting: str) -> None:
    client, rt, store, out = runtime(2, hosting=hosting)
    assert rt is not None

    async def go() -> tuple[int, int]:
        await rt.start()
        await asyncio.sleep(0.01)  # the loops' first steps (the reader's read, the first ActiveRank)
        client._on_model_call(0, None, "end_turn", 0.4)  # noqa: SLF001 - one OK call (weight 1)
        await rt.call_ended()
        after_call = len(samples(store))
        await rt.stop()
        return after_call, len(samples(store))

    after_call, after_stop = asyncio.run(go())
    assert after_call == (1 if hosting == "agentcore" else 0)
    assert after_stop == 1
    (item,) = samples(store)
    assert (item["n"], item["raw"]) == (1, 1)
    # ActiveRank once at start (the monitor's first tick), on the preferred entry.
    assert [json.loads(x)["ActiveRank"] for x in out.getvalue().splitlines()] == [0]


def test_a_failing_store_never_reaches_the_call() -> None:
    class Down(MemoryStore):
        def add_counters(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("table down")

        def get(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("table down")

    client, rt, _, _ = runtime(2, store=Down())
    assert rt is not None

    async def go() -> None:
        await rt.start()
        client._on_model_call(0, None, "end_turn", 0.4)  # noqa: SLF001
        await rt.reader.refresh()
        await rt.stop()

    asyncio.run(go())
    assert rt.reader.view() is None and client.active_rank() == 0  # The local picture


def test_models_resolved_lists_the_ids_and_flags_global_entries(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    log_models_resolved(IDS, fleet=True)
    log_models_resolved(("us.anthropic.claude-haiku-5-5", "global.anthropic.claude-sonnet-5-5"), fleet=False)
    log_models_resolved(("us.anthropic.claude-haiku-5-5",), fleet=False)
    recs = [r for r in caplog.records if getattr(r, "event", None) == "models_resolved"]
    a, b, c = (lines([r], r.name)[0] for r in recs)
    assert (a["models"], a["global_entries"], a["failover"], a["fleet"], a["entries"]) == (
        list(IDS), [], "on", "on", 3)
    assert b["global_entries"] == ["global.anthropic.claude-sonnet-5-5"] and recs[1].levelno == logging.WARNING
    assert (c["failover"], c["fleet"], c["preferred"]) == ("off", "off", "us.anthropic.claude-haiku-5-5")


def test_an_abandoned_call_reports_its_deadline() -> None:
    from owlbank.failover import ModelUnusable

    client, rt, _, _ = runtime(2)
    assert rt is not None
    client._on_model_call(0, ModelUnusable(0, Outcome.ABANDONED), None, None)  # noqa: SLF001
    (write,) = rt.reporter.drain()
    assert write.counters["miss"] == 1 and write.counters["raw"] == 1


def test_a_local_unavailable_mark_is_reported_again_until_the_fleet_shows_it(
        caplog: pytest.LogCaptureFixture) -> None:
    """The reporter re-reports an entry this process holds unavailable on every
    flush until the reader's view shows it unavailable; the backstop's clear is logged."""
    import time

    from owlbank.failover import UNAVAILABLE

    caplog.set_level(logging.INFO)
    client, rt, store, _ = runtime(2)
    assert rt is not None
    tags = [model_tag(m) for m in client.settings.model_ids]
    client._failover.record(0, UNAVAILABLE)  # noqa: SLF001 - the mark; its one report was lost
    for _ in range(2):  # no view yet: every flush reports it again
        (write,) = rt.reporter.drain()
        assert write.key["sk"].endswith(tags[0]) and write.counters == {"unav": 1}

    def publish(states: tuple[EntryState, EntryState]) -> None:
        now = time.time()
        store.put(TABLE, State(version=4, written_at_s=now, pings=None, report_every=1, models=tuple(
            ModelVerdict(i, tags[i], st, i, None, None, None, None) for i, st in enumerate(states))).to_item())
        asyncio.run(rt.reader.refresh())

    publish((EntryState.HEALTHY, EntryState.HEALTHY))  # the fleet hasn't taken it: still re-reported
    assert len(rt.reporter.drain()) == 1
    publish((EntryState.UNAVAILABLE, EntryState.HEALTHY))  # now it has
    assert rt.reporter.drain() == []
    assert client.active_rank() == 1

    client._failover.record(1, UNAVAILABLE)  # noqa: SLF001 - a second mark, never confirmed
    publish((EntryState.UNAVAILABLE, EntryState.HEALTHY))
    later = time.monotonic() + client._failover.settings.unavailable_recheck_s + 1  # noqa: SLF001
    client._failover.advance(now=later, fleet=rt.reader.view())  # noqa: SLF001
    (ln,) = [x for x in lines(caplog.records, "tac_bridge.failover_wiring") if x["event"] == "model_available"]
    assert (ln["entry"], ln["model_id"], ln["cause"], ln["scope"]) == (
        1, client.settings.model_ids[1], "unconfirmed", "process")
