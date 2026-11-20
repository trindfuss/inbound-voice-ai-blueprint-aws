"""Model failover, end to end: bridges, the fleet table and the poller.

Several bridge processes (each a real :class:`StrandsAgentClient` over scripted fake models with the
real ``GuardedModel``, a real :class:`FleetReporter` and :class:`FleetReader`) share one in-memory
``modelstats`` table with the real poller (``lambdas.model_poller.service.run_cycle`` over a fake
Bedrock). The bridges' first-token deadlines are real (50 ms, so a stalled first token is a miss); the
fleet's time (bucket times, the record's age, the poller's dwell and ping spacing) is a fake clock
the test moves, the reporter and reader flush and read when the test says. Both event loops run it
(asyncio and uvloop).

* **Degrade, fail over, no caller turn, pings, recover:** the preferred entry stalls; each stalled
  call is reissued to the next entry before anything is spoken; the bridges' reports degrade it in the
  fleet verdict; a bridge that never saw a stall follows the verdict and sends it no caller turn; the
  poller pings it exactly ``pings_per_cycle`` times a cycle while it is degraded and never otherwise;
  fast pings recover it after the dwell, and turns go back to it.
* **Unavailable from the bridges' reports:** while the poller's pings to the degraded entry only see
  hard errors (a ping counts no error as "unavailable"), a bridge that calls it and gets the model's
  ``ResourceNotFoundException`` reports it; the next cycle marks it unavailable: no more pings, and no
  bridge sends it a turn.
* **Leak scan:** the caller's words and digits appear in no log line, metric line, table item,
  state record or span.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from collections.abc import Callable, Coroutine
from dataclasses import replace
from typing import Any

import pytest
import uvloop

from lambdas.model_poller import service
from owlbank.failover import EntryState, FailoverState, Settings
from owlbank.failover.fleet import BUCKET_S, SAMPLES_PK
from owlbank.store import MemoryStore
from tac_bridge.agent_client_strands import ModelEntry, StrandsAgentClient, StrandsSettings
from tac_bridge.failover_wiring import _Events
from tac_bridge.fleet import TABLE, FleetReader, FleetReporter
from tac_bridge.model_metrics import ActiveRankMonitor, ModelMetrics
from tests.model_poller_fakes import FakeBedrock, FakeTime, InlineExecutor, Ping, models_env, states
from tests.strands_fakes import client_error, collect, text_msg
from tests.unit.test_strands_failover import EntryModel, Hang

IDS = ("us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5", "us.anthropic.claude-sonnet-5-5")
DEADLINE_S = 0.05
CYCLE_S = 30.0
CALLER = "My card is 4111 1111 1111 1111, my PIN is 97 53 18 and my name is Jane Roe"
#: Pieces of the caller's words that must never leave the call: digit groups as spoken (a
#: bare short number would also match timestamps and IDs, which are not the caller's).
SECRETS = ("4111 1111", "4111111111", "97 53 18", "975318", "Jane", "Roe", "PIN", "My card")

LOOPS = {"asyncio": asyncio.new_event_loop, "uvloop": uvloop.new_event_loop}


@pytest.fixture(params=sorted(LOOPS))
def run(request: pytest.FixtureRequest) -> Callable[[Coroutine[Any, Any, Any]], Any]:
    factory = LOOPS[request.param]

    def _run(coro: Coroutine[Any, Any, Any]) -> Any:
        with asyncio.Runner(loop_factory=factory) as runner:
            return runner.run(coro)

    return _run


def fleet_settings() -> Settings:
    base = Settings.for_models(IDS)
    return replace(base, entries=tuple(replace(e, deadline_s=None if i == len(IDS) - 1 else DEADLINE_S)
                                       for i, e in enumerate(base.entries)))


SHAPES = [{"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000},
          {"thinking": "off", "text_release": "message", "max_tokens": 400},
          {"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000}]


class Bridge:
    """One bridge process: the client, its fleet reporter and reader, its metrics (fake fleet time)."""

    def __init__(self, store: MemoryStore, ft: FakeTime, name: str) -> None:
        self.models = [EntryModel() for _ in IDS]
        by_id = dict(zip(IDS, self.models, strict=True))
        self.sid = "CA" + name * 32
        settings = StrandsSettings(models=tuple(ModelEntry(id=m, **SHAPES[i])  # type: ignore[arg-type]
                                                for i, m in enumerate(IDS)), region="us-east-1")
        self.client = StrandsAgentClient(settings, model_factory=lambda one, _sid: by_id[one.primary.id],
                                         failover_state=FailoverState(fleet_settings()))
        self.reader = FleetReader(store, model_ids=IDS, wall=ft.wall)
        state = self.client.failover
        self.reporter = FleetReporter(store, model_ids=IDS, report_every=self.reader.report_every,
                                      unconfirmed=lambda: state.unconfirmed(fleet=self.reader.view()),
                                      wall=ft.wall)  # as failover_wiring does
        self.metrics_out = io.StringIO()
        metrics = ModelMetrics("dev", stream=self.metrics_out, wall=ft.wall)
        self.client.use_fleet_view(self.reader.view)
        self.client.use_fleet_reporter(self.reporter)
        self.client.use_failover_events(_Events(metrics, ActiveRankMonitor(metrics, self.client.active_rank)))

    def script(self, entry: int, *steps: Any) -> None:
        self.models[entry].script.extend(steps)

    async def say(self, text: str = CALLER) -> str:
        events = await collect(self.client.turn(self.sid, user_text=text, tool_results=[], system_prompts={"haiku-5-5": "BASE"},
                                                system_suffix="Clock: 12:00"))
        return "".join(getattr(e, "text", "") for e in events if type(e).__name__ == "TextDelta")

    def calls(self, entry: int) -> int:
        return len(self.models[entry].calls)

    async def close(self) -> None:
        await self.reporter.stop(final_flush=False)
        await self.reader.stop()


class Fleet:
    """The table, the poller and its fake Bedrock, on one fake clock."""

    def __init__(self) -> None:
        self.ft = FakeTime()
        self.store = MemoryStore()
        self.bedrock = FakeBedrock(self.ft)
        self.metrics_out = io.StringIO()
        s = fleet_settings()
        self.deps = service.PollerDeps(
            store=self.store, bedrock=self.bedrock, stage="dev", models=service.parse_models(models_env(IDS)),
            settings=lambda: s, fixture=service.load_fixture(), wall=self.ft.wall, mono=self.ft.mono,
            sleep=self.ft.sleep, executor=InlineExecutor(), metrics_out=self.metrics_out)
        self.memory = service.PollerMemory()
        self.next_cycle = 0.0

    async def cycle(self) -> service.CycleReport:
        """One decision cycle, :data:`CYCLE_S` after the previous one's start."""
        self.ft.t = max(self.ft.t, self.next_cycle)
        self.next_cycle = self.ft.t + CYCLE_S
        return await service.run_cycle(self.deps, self.memory)

    def states(self) -> list[EntryState] | None:
        return states(self.store)

    def items(self) -> list[dict[str, Any]]:
        return [*self.store.query_range(TABLE, {"s": SAMPLES_PK}, low="0", high="9"),
                *(i for i in [self.store.get(TABLE, {"s": "state", "sk": "state"})] if i)]


def scan(texts: list[str]) -> None:
    for text in texts:
        for secret in SECRETS:
            assert secret not in text


def test_degrade_fail_over_ping_and_recover(run: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    from tests.strands_fakes import traced

    fleet = Fleet()
    ft, n = fleet.ft, fleet_settings()

    async def go() -> dict[str, Any]:
        a, b, c = (Bridge(fleet.store, ft, x) for x in "123")
        out: dict[str, Any] = {}
        # Healthy: the preferred entry answers; the verdict says so; no ping.
        a.script(0, text_msg("hello there"))
        assert await a.say() == "hello there"
        await a.reporter.flush()
        rep = await fleet.cycle()
        assert fleet.states() == [EntryState.HEALTHY] * 3 and rep.sent == 0
        # The preferred entry stalls: every stalled call moves on before anything is spoken.
        ft.advance(BUCKET_S)  # in a later 10 s bucket than the healthy call
        for bridge in (a, b):
            for _ in range(n.miss_run):
                bridge.script(0, Hang())
                bridge.script(1, text_msg("one moment, here it is"))
                assert await bridge.say() == "one moment, here it is"
            await bridge.reporter.flush()
        rep = await fleet.cycle()
        assert fleet.states() == [EntryState.DEGRADED, EntryState.HEALTHY, EntryState.HEALTHY]
        assert rep.sent == 0  # the cycle that degrades it has not pinged it yet
        # A bridge that never saw a stall follows the verdict: no caller turn to the degraded entry.
        await c.reader.refresh()
        sent_by_cycle: list[int] = []
        pinged: set[str] = set()
        fleet.bedrock.default[IDS[0]] = Ping(ttft_s=0.01)  # it is fast again: only pings can tell
        for _ in range(20):
            c.script(1, text_msg("fine"))
            assert await c.say() == "fine"
            await c.reporter.flush()
            rep = await fleet.cycle()
            sent_by_cycle.append(rep.sent)
            pinged |= {m for m, _, _ in fleet.bedrock.pings}
            await c.reader.refresh()
            if fleet.states() == [EntryState.HEALTHY] * 3:
                break
        out["sent"] = sent_by_cycle
        out["pinged"] = pinged
        assert c.calls(0) == 0
        # Recovered: the next cycle pings nothing and turns go back to the preferred entry.
        rep = await fleet.cycle()
        out["after"] = rep.sent
        await c.reader.refresh()
        c.script(0, text_msg("back on the first"))
        assert await c.say() == "back on the first"
        out["metrics"] = [json.loads(x) for b in (a, b, c) for x in b.metrics_out.getvalue().splitlines()]
        out["metric_text"] = "".join(b.metrics_out.getvalue() for b in (a, b, c)) + fleet.metrics_out.getvalue()
        for bridge in (a, b, c):
            await bridge.close()
        return out

    with traced("off") as t:
        out = run(go())
        spans = t.spans()
    # Pings at the set rate while degraded, to that entry only, and none once it recovered.
    assert out["sent"] and all(s == n.pings_per_cycle for s in out["sent"])
    assert out["pinged"] == {IDS[0]} and out["after"] == 0
    assert len(out["sent"]) * CYCLE_S >= n.dwell_s  # not before the dwell
    assert len(out["sent"]) * n.pings_per_cycle >= n.recover_min_pings
    # The metrics the alarms use: a failover counted per reissue and per verdict-driven start.
    names = [next(k for k in m if k not in ("_aws", "Service", "Environment")) for m in out["metrics"]]
    assert names.count("Failover") == 2 * n.miss_run + 1 and "ActiveRank" in names
    ranks = [m["ActiveRank"] for m in out["metrics"] if "ActiveRank" in m]
    assert 1 in ranks and ranks[-1] == 0
    # nothing of the caller's in any log, metric, table item, state record or span.
    scan([caplog.text, out["metric_text"], json.dumps(fleet.items())])
    assert spans  # Strands' spans went through the bridge's telemetry (content off)
    scan([json.dumps({k: str(v) for k, v in (s.attributes or {}).items()}) for s in spans])
    scan([str(ev.attributes) for s in spans for ev in s.events])


def test_an_entry_goes_unavailable_promptly_from_the_bridges_reports(run: Any,
                                                                     caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    fleet = Fleet()
    ft, n = fleet.ft, fleet_settings()

    async def go() -> dict[str, Any]:
        a, c = Bridge(fleet.store, ft, "4"), Bridge(fleet.store, ft, "5")
        for _ in range(n.miss_run):
            a.script(0, Hang())
            a.script(1, text_msg("ok"))
            await a.say()
        await a.reporter.flush()
        await fleet.cycle()
        assert fleet.states() == [EntryState.DEGRADED, EntryState.HEALTHY, EntryState.HEALTHY]
        # The model is retired: the poller's pings to it see only an error (a hard error to a ping).
        fleet.bedrock.default[IDS[0]] = Ping(error="ResourceNotFoundException", message=CALLER)
        rep = await fleet.cycle()
        assert rep.sent == n.pings_per_cycle and fleet.states()[0] is EntryState.DEGRADED
        # A bridge that has no verdict yet (a new task) calls it and learns what the error means.
        d = Bridge(fleet.store, ft, "6")
        d.script(0, client_error("ResourceNotFoundException", CALLER))
        d.script(1, text_msg("still here"))
        assert await d.say() == "still here"  # moved on before any output (4a.3)
        await d.reporter.flush()
        rep = await fleet.cycle()  # the next cycle: unavailable at once (4a.1), whatever the dwell
        out = {"states": fleet.states(), "sent_then": rep.sent}
        rep = await fleet.cycle()
        out["sent_after"] = rep.sent
        # Out of rotation everywhere: the bridges follow the verdict.
        for bridge in (a, c, d):
            await bridge.reader.refresh()
            bridge.script(1, text_msg("fine"))
            assert await bridge.say() == "fine"
        out["calls0"] = (a.calls(0), c.calls(0), d.calls(0))
        out["metric_text"] = "".join(b.metrics_out.getvalue() for b in (a, c, d))
        for bridge in (a, c, d):
            await bridge.close()
        return out

    out = run(go())
    assert out["states"] == [EntryState.UNAVAILABLE, EntryState.HEALTHY, EntryState.HEALTHY]
    assert out["sent_then"] == n.pings_per_cycle  # the cycle that read the report still pinged it
    assert out["sent_after"] == 0  # never a ping to an unavailable entry (4a.3)
    assert out["calls0"] == (n.miss_run, 0, 1)
    bridge_lines = [r for r in caplog.records if getattr(r, "event", None) == "model_unavailable"
                    and r.name == "tac_bridge.agent_client_strands"]
    assert [(r.model_id, r.error_code) for r in bridge_lines] == [(IDS[0], "ResourceNotFoundException")]
    assert '"Unavailable":1' in out["metric_text"]
    scan([caplog.text, out["metric_text"], json.dumps(fleet.items())])
