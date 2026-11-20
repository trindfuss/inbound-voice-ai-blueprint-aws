"""The model-failover poller: an in-memory store, a scripted Bedrock client and a fake clock. No network."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from collections import deque
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lambdas.model_poller import handler as handler_mod
from lambdas.model_poller import service
from owlbank import obs, security_log, tools_contract
from owlbank.failover.classify import Outcome
from owlbank.failover.fleet import ModelVerdict, State, model_tag, state_key
from owlbank.failover.rules import EntryState, Sample
from owlbank.failover.settings import Settings
from owlbank.store import MemoryStore
from tests.model_poller_fakes import (
    MODEL_IDS,
    STAGE,
    FakeBedrock,
    FakeTime,
    FlakyStore,
    InlineExecutor,
    Ping,
    add_samples,
    ev,
    metrics,
    make_deps,
    models_env,
    run,
    stored_state,
)

H, D, U = EntryState.HEALTHY, EntryState.DEGRADED, EntryState.UNAVAILABLE
LOOPS = ["asyncio", "uvloop"]


def cycle(deps: service.PollerDeps, memory: service.PollerMemory, loop: str = "asyncio") -> service.CycleReport:
    return run(lambda: service.run_cycle(deps, memory), loop)


def seed_state(store: MemoryStore, wall: float, states: list[EntryState], *, since: float | None = None,
               version: int = 3, ids: tuple[str, ...] = MODEL_IDS) -> State:
    models = tuple(ModelVerdict(index=i, model=model_tag(ids[i]), state=s, rank=i, since_s=since, p50_s=None,
                                p90_s=None, error_rate=None)
                   for i, s in enumerate(states))
    state = State(version=version, written_at_s=wall, models=models, pings=None, report_every=1)
    store.put(service.TABLE, state.to_item())
    return state


def degrade_entry0(store: MemoryStore, at_wall: float) -> None:
    add_samples(store, at_wall, 0, [5.0] * 20)


# --------------------------------------------------------------------------- configuration


class TestConfig:
    def test_models_env(self) -> None:
        models = service.parse_models(models_env(MODEL_IDS[:2], thinking="off", max_tokens=512))
        assert [m.id for m in models] == list(MODEL_IDS[:2])
        assert all(m.thinking == "off" and m.max_tokens == 512 for m in models)

    @pytest.mark.parametrize("thinking", [*sorted(service.THINKING_PARAMS), "enabled:1024", "enabled:2047"])
    def test_every_bridge_thinking_setting_is_accepted(self, thinking: str) -> None:
        (model,) = service.parse_models(models_env(MODEL_IDS[:1], thinking=thinking, max_tokens=2048))
        assert model.thinking == thinking

    @pytest.mark.parametrize(("thinking", "max_tokens"), [
        ("enabled:1023", 4096), ("enabled:2048", 2048), ("enabled:4096", 2048), ("enabled:", 4096),
        ("enabled:x", 4096), ("enabled:+1500", 4096), ("enabled:-1500", 4096), ("enabled:1500.0", 4096),
        ("enabled:\uff11\uff15\uff10\uff10", 4096), ("Enabled:1500", 4096), ("enabled", 4096), (5, 4096)])
    def test_bad_thinking_budget_is_refused(self, thinking: Any, max_tokens: int) -> None:
        raw = json.dumps([{"id": "a", "thinking": thinking, "max_tokens": max_tokens}])
        with pytest.raises(service.PollerConfigError) as err:
            service.parse_models(raw)
        assert err.value.name == f"{service.MODELS_ENV}.thinking"
        assert str(thinking) not in str(err.value)  # names the rule, never the value

    @settings(max_examples=300, deadline=None)
    @given(thinking=st.one_of(
               st.sampled_from(sorted(service.THINKING_PARAMS)),
               st.integers(min_value=-10, max_value=10_000).map(lambda n: f"enabled:{n}"),
               st.text(max_size=12).map(lambda t: f"enabled:{t}"),
               st.text(max_size=16)),
           max_tokens=st.integers(min_value=1, max_value=8192))
    def test_thinking_rule_parity_with_the_bridge(self, thinking: str, max_tokens: int) -> None:
        """Every thinking setting the bridge accepts with a max_tokens, the poller accepts, and nothing
        else (the bridge's ``thinking_error``); an accepted one sends the bridge's request fields."""
        strands_client = pytest.importorskip("tac_bridge.agent_client_strands")
        bridge_ok = strands_client.thinking_error(thinking, max_tokens) is None
        raw = json.dumps([{"id": "a", "thinking": thinking, "max_tokens": max_tokens}])
        if not bridge_ok:
            with pytest.raises(service.PollerConfigError):
                service.parse_models(raw)
            return
        (spec,) = service.parse_models(raw)
        req = service.ping_request(spec, service.load_fixture())
        assert req.get("additionalModelRequestFields") == strands_client.thinking_request_fields(thinking)

    @pytest.mark.parametrize("raw", [None, "", "{", "{}", "[]", json.dumps([{"id": "a"}] * 4),
                                     json.dumps([{"id": "", "thinking": "off", "max_tokens": 1}]),
                                     json.dumps([{"id": "a", "thinking": "loud", "max_tokens": 1}]),
                                     json.dumps([{"id": "a", "thinking": "off", "max_tokens": 0}]),
                                     json.dumps([{"id": "a", "thinking": "off", "max_tokens": True}]),
                                     json.dumps([{"id": "a", "thinking": "off", "max_tokens": 1}] * 2)])
    def test_bad_models_env_names_the_setting(self, raw: str | None) -> None:
        with pytest.raises(service.PollerConfigError) as err:
            service.parse_models(raw)
        assert err.value.name.startswith(service.MODELS_ENV)

    @pytest.mark.parametrize(("raw", "stage", "expected"), [
        (None, "dev", None), ("", "prod", None), ("2.5", "dev", 2.5), ("30", "test", 30.0)])
    def test_delay(self, raw: str | None, stage: str, expected: float | None) -> None:
        env = {} if raw is None else {service.FAILOVER_TEST_ENV: raw}
        assert service.test_delay(env, stage=stage) == expected

    @pytest.mark.parametrize(("raw", "stage"), [("2", "prod"), ("0", "dev"), ("31", "dev"), ("x", "dev"),
                                                ("nan", "dev"), ("-1", "dev")])
    def test_delay_refused(self, raw: str, stage: str) -> None:
        with pytest.raises(service.PollerConfigError):
            service.test_delay({service.FAILOVER_TEST_ENV: raw}, stage=stage)

    def test_delay_parity_with_the_bridge(self) -> None:
        gm = pytest.importorskip("tac_bridge.guarded_model")
        for raw in ("", "0.5", "4", "30"):
            env = {service.FAILOVER_TEST_ENV: raw}
            assert service.test_delay(env, stage="dev") == gm.failover_test_delay(env, stage="dev")
        for raw, stage in (("31", "dev"), ("0", "dev"), ("2", "prod")):
            with pytest.raises(service.PollerConfigError):
                service.test_delay({service.FAILOVER_TEST_ENV: raw}, stage=stage)
            with pytest.raises(gm.TestDelayError):
                gm.failover_test_delay({service.FAILOVER_TEST_ENV: raw}, stage=stage)

    def test_thinking_params_parity_with_the_bridge(self) -> None:
        strands_client = pytest.importorskip("tac_bridge.agent_client_strands")
        assert dict(service.THINKING_PARAMS) == strands_client.THINKING_PARAMS

    def test_is_output_parity_with_the_bridge(self) -> None:
        gm = pytest.importorskip("tac_bridge.guarded_model")
        events: list[dict[str, Any]] = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "x"}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}},
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "r"}}}},
            {"contentBlockStart": {"start": {"toolUse": {"name": "get_balance"}}}},
            {"contentBlockStart": {"start": {}}},
            {"contentBlockStop": {}}, {"messageStop": {"stopReason": "end_turn"}}, {"metadata": {}},
            {"contentBlockDelta": "bad"}, {"contentBlockStart": {"start": "bad"}},
        ]
        assert [service.is_output(e) for e in events] == [gm.is_output(e) for e in events]


# --------------------------------------------------------------------------- the fixture and the requests


def _texts(fixture: dict[str, Any]) -> list[str]:
    """Every string in the fixture except the tool names (fixed identifiers shared with the bridge)."""
    out: list[str] = []

    def walk(node: Any, key: str | None = None) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                out.append(k)
                walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, key)
        elif isinstance(node, str) and key != "name":
            out.append(node)

    walk(fixture)
    return out


class TestFixture:
    def test_shape(self) -> None:
        fixture = service.load_fixture()
        assert set(fixture) == {"system", "tools", "messages"}
        assert [m["role"] for m in fixture["messages"]] == ["user"]
        assert [set(b) for b in fixture["messages"][0]["content"]] == [{"text"}]
        for tool in fixture["tools"]:
            assert set(tool) == {"toolSpec"}
            assert set(tool["toolSpec"]) == {"name", "description", "inputSchema"}

    def test_no_digit_outside_tool_names(self) -> None:
        assert not [t for t in _texts(service.load_fixture()) if re.search(r"\d", t)]

    def test_no_placeholder_or_caller_data_slots(self) -> None:
        """The prompt is the unverified one: no template slot is left and no profile or memory block."""
        system = service.load_fixture()["system"]
        assert not re.search(r"\{\w+\}", system)
        assert "<untrusted_data>" not in system.replace("<untrusted_data> ... </untrusted_data>", "")

    def test_tool_names_match_the_bridge(self) -> None:
        names = [t["toolSpec"]["name"] for t in service.load_fixture()["tools"]]
        gateway = [tools_contract.gateway_tool_full_name(t) for t in tools_contract.GATEWAY_LEVEL0_TOOLS]
        assert names == [*tools_contract.INLINE_TOOLS, *gateway]
        strands_client = pytest.importorskip("tac_bridge.agent_client_strands")
        assert names[len(tools_contract.INLINE_TOOLS):] == list(strands_client.ALLOWED_GATEWAY_TOOLS)

    def test_size_is_close_to_a_real_turn(self) -> None:
        """The fixture's prompt and tools are about the size of the bridge's (within 20 %)."""
        catalog = json.loads((service.FIXTURE_PATH.parents[3] / "data" / "prompt_catalog.json").read_text())
        prompt = next(e["text"] for e in catalog["entries"] if e["key"] == "agent.system.haiku-5-5" and e["lang"] == "en-US")
        real = len(prompt) + len(json.dumps(tools_contract.inline_tool_specs()))
        fixture = service.load_fixture()
        inline = fixture["tools"][:len(tools_contract.INLINE_TOOLS)]
        ours = len(fixture["system"]) + len(json.dumps(inline))
        assert 0.8 <= ours / real <= 1.2

    def test_ping_request(self) -> None:
        fixture = service.load_fixture()
        spec = service.ModelSpec(MODEL_IDS[0], "adaptive", 777)
        req = service.ping_request(spec, fixture)
        assert req["modelId"] == MODEL_IDS[0]
        assert req["inferenceConfig"] == {"maxTokens": 777}
        assert req["system"] == [{"text": fixture["system"]}]
        assert req["messages"] == fixture["messages"]
        assert req["toolConfig"] == {"tools": fixture["tools"]}
        assert req["additionalModelRequestFields"]["thinking"]["type"] == "adaptive"
        assert "cachePoint" not in json.dumps(req)  # prompt caching off
        assert "additionalModelRequestFields" not in service.ping_request(
            service.ModelSpec("m", "default", 5), fixture)
        enabled = service.ping_request(service.ModelSpec("m", "enabled:1500", 2000), fixture)
        assert enabled["additionalModelRequestFields"] == {"thinking": {"type": "enabled", "budget_tokens": 1500}}

    def test_recheck_request(self) -> None:
        req = service.recheck_request(service.ModelSpec("m", "adaptive", 9))
        assert req["inferenceConfig"] == {"maxTokens": 1} and "system" not in req and "toolConfig" not in req

    def test_deadlines_and_spacing(self) -> None:
        s = Settings.for_models(list(MODEL_IDS))
        assert [service.ping_deadline(s, i) for i in range(3)] == [4.0, 3.0, 7.0]
        assert service.ping_spacing(s) == 8.0
        assert service.ping_spacing(Settings.for_models(list(MODEL_IDS), pings_per_cycle=5)) == pytest.approx(4.8)
        assert service.cycle_budget_s(s) < service.CYCLE_GAP_S * 2


# --------------------------------------------------------------------------- cycles


class TestCycle:
    def test_first_cycle_writes_a_healthy_record(self) -> None:
        deps, t, bedrock, store = make_deps()
        report = cycle(deps, service.PollerMemory())
        assert report.read and report.written
        state = stored_state(store)
        assert [m.state for m in state.models] == [H, H, H] and state.version == 0
        assert state.written_at_s == pytest.approx(t.wall(), abs=1e-3)
        assert bedrock.pings == [] and bedrock.rechecks == []

    def test_heartbeat_every_cycle_when_nothing_changed(self, caplog: pytest.LogCaptureFixture) -> None:
        deps, t, _, store = make_deps()
        memory = service.PollerMemory()
        cycle(deps, memory)
        first = stored_state(store)
        t.advance(30)
        with caplog.at_level(logging.INFO, logger=service.log.name):
            cycle(deps, memory)
        second = stored_state(store)
        assert second.version == first.version
        assert second.written_at_s - first.written_at_s == pytest.approx(30, abs=1e-3)
        assert [ev(r) for r in caplog.records].count("model_state_written") == 1

    def test_window_read_trips_a_slow_entry(self, caplog: pytest.LogCaptureFixture) -> None:
        deps, t, bedrock, store = make_deps()
        degrade_entry0(store, t.wall() - 20)
        with caplog.at_level(logging.INFO, logger=service.log.name):
            report = cycle(deps, service.PollerMemory())
        state = stored_state(store)
        assert [m.state for m in state.models] == [D, H, H] and state.version == 1
        assert [m.rank for m in state.models] == [2, 0, 1]
        assert report.target is None and bedrock.pings == []  # pings start the cycle after the trip
        degraded = [r for r in caplog.records if ev(r) == "model_degraded"]
        assert len(degraded) == 1 and ev(degraded[0], "entry") == 0 and ev(degraded[0], "samples") == 20

    def test_old_buckets_are_outside_the_window(self) -> None:
        deps, t, _, store = make_deps()
        degrade_entry0(store, t.wall() - 200)
        cycle(deps, service.PollerMemory())
        assert [m.state for m in stored_state(store).models] == [H, H, H]

    def test_weighting_needs_enough_reported_samples(self) -> None:
        """Weighted calls alone don't make a window: 5 reports of weight 10 never trip."""
        deps, t, _, store = make_deps()
        add_samples(store, t.wall() - 20, 0, [5.0] * 5, weight=10)
        cycle(deps, service.PollerMemory())
        assert [m.state for m in stored_state(store).models] == [H, H, H]

    def test_report_prob_rises_and_falls_with_traffic(self) -> None:
        deps, t, _, store = make_deps()
        memory = service.PollerMemory()
        add_samples(store, t.wall() - 20, 1, [0.5] * 1200)  # 1,200 OK calls in 120 s: 10 a second
        cycle(deps, memory)
        assert stored_state(store).report_every == 2
        t.advance(150)  # the busy bucket has left the window
        cycle(deps, memory)
        assert stored_state(store).report_every == 1

    def test_pings_go_to_the_degraded_entry_only(self) -> None:
        deps, t, bedrock, store = make_deps()
        memory = service.PollerMemory()
        seed_state(store, t.wall(), [H, D, D], since=t.wall())
        report = cycle(deps, memory)
        assert bedrock.pinged_models() == [MODEL_IDS[1]] * 3  # the highest-ranked degraded entry
        assert report.sent == 3 and report.target == 1
        spacing = [w for _, w, _ in bedrock.pings]
        assert all(b - a >= 8.0 - 1e-9 for a, b in zip(spacing, spacing[1:], strict=False))
        assert stored_state(store).pings.index == 1 and len(stored_state(store).pings.samples) == 3

    def test_no_ping_to_an_unavailable_entry(self) -> None:
        deps, t, bedrock, store = make_deps()
        seed_state(store, t.wall(), [U, H, H], since=t.wall())
        cycle(deps, service.PollerMemory())
        assert bedrock.pings == []

    def test_no_pings_while_every_entry_is_healthy(self) -> None:
        deps, t, bedrock, store = make_deps()
        memory = service.PollerMemory()
        for _ in range(4):
            cycle(deps, memory)
            t.advance(30)
        assert bedrock.pings == []

    def test_a_ping_past_the_deadline_is_a_miss(self) -> None:
        deps, t, bedrock, store = make_deps()
        seed_state(store, t.wall(), [D, H, H], since=t.wall())
        bedrock.script[MODEL_IDS[0]] = deque([Ping(5.0), Ping(1.0), Ping(error="ThrottlingException")])
        report = cycle(deps, service.PollerMemory())
        assert [(p.outcome, p.ttft_s) for p in report.pings] == [
            (Outcome.ABANDONED, 4.0), (Outcome.OK, pytest.approx(1.0)), (Outcome.THROTTLED, None)]

    def test_a_request_error_ping_is_no_sample(self) -> None:
        deps, t, bedrock, store = make_deps()
        seed_state(store, t.wall(), [D, H, H], since=t.wall())
        bedrock.default[MODEL_IDS[0]] = Ping(error="ValidationException",
                                             message="The model returned the following errors: x")
        report = cycle(deps, service.PollerMemory())
        assert {p.outcome for p in report.pings} == {Outcome.REQUEST_ERROR}

    def test_the_last_entry_is_abandoned_at_twice_its_trip(self) -> None:
        deps, t, bedrock, store = make_deps()
        seed_state(store, t.wall(), [H, H, D], since=t.wall())
        bedrock.script[MODEL_IDS[2]] = deque([Ping(6.9), Ping(7.1), Ping(0.5)])
        report = cycle(deps, service.PollerMemory())
        assert [p.outcome for p in report.pings] == [Outcome.OK, Outcome.ABANDONED, Outcome.OK]
        assert report.pings[1].ttft_s == 7.0

    def test_test_delay_applies_to_the_preferred_entry_only(self) -> None:
        deps, t, bedrock, store = make_deps(test_delay_s=5.0)
        seed_state(store, t.wall(), [D, D, H], since=t.wall())
        report = cycle(deps, service.PollerMemory())
        assert report.target == 0 and {p.outcome for p in report.pings} == {Outcome.ABANDONED}
        deps2, t2, _, store2 = make_deps(test_delay_s=2.0)
        seed_state(store2, t2.wall(), [D, H, H], since=t2.wall())
        assert [p.ttft_s for p in cycle(deps2, service.PollerMemory()).pings] == [2.0, 2.0, 2.0]
        deps3, t3, _, store3 = make_deps(test_delay_s=5.0)
        seed_state(store3, t3.wall(), [H, D, H], since=t3.wall())
        assert {p.outcome for p in cycle(deps3, service.PollerMemory()).pings} == {Outcome.OK}

    def test_recovery_after_enough_good_pings_over_the_period_and_the_dwell(
            self, caplog: pytest.LogCaptureFixture) -> None:
        deps, t, bedrock, store = make_deps()
        memory = service.PollerMemory()
        degrade_entry0(store, t.wall() - 20)
        cycle(deps, memory)
        tripped = stored_state(store).models[0].since_s
        recovered_at = None
        with caplog.at_level(logging.INFO, logger=service.log.name):
            for _ in range(12):
                t.advance(30)
                cycle(deps, memory)
                s = stored_state(store)
                if s.models[0].state is H:
                    recovered_at = s.models[0].since_s
                    break
        assert recovered_at is not None
        assert recovered_at - tripped >= deps.settings().dwell_s  # the dwell
        assert len([m for m in bedrock.pinged_models() if m == MODEL_IDS[0]]) >= 10
        assert [ev(r) for r in caplog.records].count("model_recovered") == 1
        t.advance(30)
        before = len(bedrock.pings)
        cycle(deps, memory)
        assert len(bedrock.pings) == before  # healthy again: no more pings

    def test_model_recovered_logs_the_judged_ping_count(self, caplog: pytest.LogCaptureFixture) -> None:
        """``pings`` is the size of the pruned set the verdict judged: the samples newer than
        ``recover_period_s`` plus the one at its edge, not every ping stored or sent."""
        deps, t, _, store = make_deps()
        memory = service.PollerMemory()
        degrade_entry0(store, t.wall() - 20)
        cycle(deps, memory)
        sent: list[Sample] = []
        report = None
        with caplog.at_level(logging.INFO, logger=service.log.name):
            for _ in range(12):
                t.advance(30)
                report = cycle(deps, memory)
                assert report.target == 0
                sent.extend(report.pings)
                if stored_state(store).models[0].state is H:
                    break
        assert report is not None and report.now_s is not None
        cutoff = report.now_s - deps.settings().recover_period_s
        expected = sum(1 for s in sent if s.at_s > cutoff) + (1 if any(s.at_s <= cutoff for s in sent) else 0)
        logged = [ev(r, "pings") for r in caplog.records if ev(r) == "model_recovered"]
        assert logged == [expected]
        assert expected < len(sent)  # older pings were pruned: the count is not every ping sent

    def test_slow_pings_never_recover(self) -> None:
        deps, t, bedrock, store = make_deps()
        memory = service.PollerMemory()
        seed_state(store, t.wall() - 600, [D, H, H], since=t.wall() - 600)
        bedrock.default[MODEL_IDS[0]] = Ping(3.0)  # under the deadline, over the recovery threshold
        for _ in range(10):
            cycle(deps, memory)
            t.advance(30)
        assert stored_state(store).models[0].state is D

    def test_recheck_marks_an_entry_available(self, caplog: pytest.LogCaptureFixture) -> None:
        deps, t, bedrock, store = make_deps()
        memory = service.PollerMemory()
        seed_state(store, t.wall(), [U, H, H], since=t.wall() - 800)
        cycle(deps, memory)
        assert bedrock.rechecks == []  # not yet due (900 s)
        t.advance(100)
        with caplog.at_level(logging.INFO, logger=service.log.name):
            cycle(deps, memory)
        assert [m for m, _ in bedrock.rechecks] == [MODEL_IDS[0]]
        assert bedrock.rechecks[0][1]["inferenceConfig"] == {"maxTokens": 1}
        assert stored_state(store).models[0].state is H
        assert [ev(r) for r in caplog.records].count("model_available") == 1

    def test_recheck_failure_keeps_it_out(self) -> None:
        deps, t, bedrock, store = make_deps()
        memory = service.PollerMemory()
        bedrock.recheck_fails.add(MODEL_IDS[0])
        seed_state(store, t.wall(), [U, H, H], since=t.wall() - 1000)
        cycle(deps, memory)
        assert len(bedrock.rechecks) == 1 and stored_state(store).models[0].state is U
        for _ in range(10):  # 300 s: no second recheck before 900 s
            t.advance(30)
            cycle(deps, memory)
        assert len(bedrock.rechecks) == 1
        t.advance(900)
        cycle(deps, memory)
        assert len(bedrock.rechecks) == 2 and stored_state(store).models[0].state is U

    def test_an_unavailable_report_marks_it_at_once(self, caplog: pytest.LogCaptureFixture) -> None:
        deps, t, _, store = make_deps()
        memory = service.PollerMemory()
        cycle(deps, memory)
        t.advance(30)
        add_samples(store, t.wall() - 5, 0, [], unavailable=1)
        with caplog.at_level(logging.INFO, logger=service.log.name):
            cycle(deps, memory)
        assert stored_state(store).models[0].state is U
        rec = [r for r in caplog.records if ev(r) == "model_unavailable"]
        assert len(rec) == 1 and ev(rec[0], "preferred") is True and rec[0].levelno == logging.ERROR

    @pytest.mark.parametrize("failure", ["fail_get", "fail_query"])
    def test_table_unreadable_keeps_the_last_state_and_no_heartbeat(
            self, failure: str, caplog: pytest.LogCaptureFixture) -> None:
        deps, t, _, store = make_deps()
        memory = service.PollerMemory()
        cycle(deps, memory)
        before = store.get(service.TABLE, state_key())
        setattr(store, failure, True)
        t.advance(30)
        degrade_entry0(store, t.wall() - 5)
        with caplog.at_level(logging.INFO, logger=service.log.name):
            report = cycle(deps, memory)
        assert not report.written
        setattr(store, failure, False)
        assert store.get(service.TABLE, state_key()) == before
        assert "model_state_written" not in [ev(r) for r in caplog.records]

    def test_failed_write_keeps_the_stored_state_and_the_pings(self) -> None:
        deps, t, bedrock, store = make_deps()
        memory = service.PollerMemory()
        seed_state(store, t.wall() - 600, [D, H, H], since=t.wall() - 600)
        before = store.get(service.TABLE, state_key())
        store.fail_write = True
        first = cycle(deps, memory)
        assert not first.written and store.get(service.TABLE, state_key()) == before
        store.fail_write = False
        t.advance(30)
        second = cycle(deps, memory)
        assert second.written
        assert len(stored_state(store).pings.samples) == 6  # the unwritten cycle's pings are kept

    def test_a_newer_record_is_never_overwritten(self) -> None:
        """Another invocation writes between this cycle's read and write: its record stands."""
        deps, t, bedrock, store = make_deps()
        memory = service.PollerMemory()
        seed_state(store, t.wall(), [D, H, H], since=t.wall(), version=3)
        newer: list[State] = []

        class Racer(FakeBedrock):
            def converse_stream(self, **request: Any) -> dict[str, Any]:
                if not newer:
                    newer.append(seed_state(store, t.wall(), [U, H, H], since=t.wall(), version=4))
                return super().converse_stream(**request)

        deps.bedrock = Racer(t)
        report = cycle(deps, memory)
        assert not report.written
        assert stored_state(store) == State.from_item(newer[0].to_item())

    def test_malformed_record_is_replaced(self, caplog: pytest.LogCaptureFixture) -> None:
        deps, t, _, store = make_deps()
        store.put(service.TABLE, {**state_key(), "version": 9, "models": "x"})
        with caplog.at_level(logging.INFO, logger=service.log.name):
            assert cycle(deps, service.PollerMemory()).written
        state = stored_state(store)
        assert [m.state for m in state.models] == [H, H, H] and state.version == 10
        assert "model_state_malformed" in [ev(r) for r in caplog.records]

    def test_a_record_for_a_shorter_list_keeps_the_listed_models(self) -> None:
        deps, t, _, store = make_deps(ids=MODEL_IDS[:2])
        seed_state(store, t.wall(), [D, H, H], since=t.wall(), version=5)
        cycle(deps, service.PollerMemory())
        state = stored_state(store)
        assert len(state.models) == 2 and state.version == 6
        assert [m.state for m in state.models] == [D, H]
        assert [m.model for m in state.models] == [model_tag(m) for m in MODEL_IDS[:2]]

    def test_state_is_matched_by_model_not_by_place(self, caplog: pytest.LogCaptureFixture) -> None:
        a, b, c = MODEL_IDS
        deps, t, _, store = make_deps(ids=(c, b))
        seed_state(store, t.wall() - 5, [U, D], since=t.wall() - 5, version=5, ids=(a, b))
        with caplog.at_level(logging.WARNING):
            report = cycle(deps, service.PollerMemory())
        state = stored_state(store)
        # C is new to the record: healthy and never changed; B keeps its state and its since.
        assert [m.state for m in state.models] == [H, D]
        assert state.models[0].since_s is None and state.models[1].since_s == pytest.approx(t.wall() - 5)
        assert [m.model for m in state.models] == [model_tag(c), model_tag(b)]
        assert state.version == 6 and report.target == 1  # B is the one pinged
        assert any(ev(r) == "model_state_realigned" for r in caplog.records)

    def test_a_swapped_list_moves_each_state_with_its_model(self) -> None:
        a, b, _ = MODEL_IDS
        deps, t, bedrock, store = make_deps(ids=(b, a))
        seed_state(store, t.wall() - 5, [D, H], since=t.wall() - 5, ids=(a, b))
        report = cycle(deps, service.PollerMemory())
        assert [m.state for m in stored_state(store).models] == [H, D]
        assert report.target == 1 and set(bedrock.pinged_models()) == {a}

    def test_samples_of_a_model_not_listed_are_ignored(self) -> None:
        a, b, c = MODEL_IDS
        deps, t, _, store = make_deps(ids=(a, b))
        add_samples(store, t.wall(), 2, [9.0] * 20, unavailable=1)  # an old-list bridge's model C
        report = cycle(deps, service.PollerMemory())
        assert report.buckets == ()
        assert [m.state for m in stored_state(store).models] == [H, H]

    def test_settings_come_from_the_source(self) -> None:
        deps, t, bedrock, store = make_deps(pings_per_cycle=5)
        seed_state(store, t.wall(), [D, H, H], since=t.wall())
        assert cycle(deps, service.PollerMemory()).sent == 5

    def test_settings_source_never_raises(self) -> None:
        fallback = Settings.for_models(list(MODEL_IDS))
        good = Settings.for_models(list(MODEL_IDS), pings_per_cycle=2)
        calls: list[int] = []

        def read() -> Settings:
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("policy unreadable")
            return good

        source = service.settings_source(read, fallback)
        assert source() is good and source() is good

        def broken() -> Settings:
            raise RuntimeError("policy unreadable")

        assert service.settings_source(broken, fallback)() is fallback


# --------------------------------------------------------------------------- timing and loops


class TestTiming:
    @pytest.mark.parametrize("loop", LOOPS)
    def test_early_wakes_never_start_anything_early(self, loop: str) -> None:
        deps, t, bedrock, store = make_deps(early_s=0.0004)
        seed_state(store, t.wall(), [D, H, H], since=t.wall())
        start = t.wall()
        reports = run(lambda: service.run_invocation(deps, service.PollerMemory()), loop)
        assert len(reports) == 2 and all(r.written for r in reports)
        starts = [w for _, w, _ in bedrock.pings]
        assert len(starts) == 6
        assert all(b - a >= 8.0 for a, b in zip(starts[:3], starts[1:3], strict=False))
        assert starts[3] - start >= service.CYCLE_GAP_S  # the second cycle starts 30 s in, not before
        assert reports[1].now_s - start >= service.CYCLE_GAP_S

    def test_a_cycle_without_time_left_is_skipped(self) -> None:
        deps, t, _, _ = make_deps()
        left = [80.0]

        def remaining() -> float:
            return left[0] - t.mono()

        reports = run(lambda: service.run_invocation(deps, service.PollerMemory(), remaining_s=remaining))
        assert len(reports) == 2
        left[0] = 40.0
        t.t = 0.0
        reports = run(lambda: service.run_invocation(deps, service.PollerMemory(), remaining_s=remaining))
        assert len(reports) == 1

    @pytest.mark.parametrize("loop", LOOPS)
    def test_a_stalled_ping_is_abandoned_at_the_deadline(self, loop: str) -> None:
        """A real thread that never sends an event: the ping is a miss at the deadline by the clock,
        the thread is told to stop, and the stream is closed."""
        settings = Settings.for_models(list(MODEL_IDS[:2]))
        settings = Settings(entries=(settings.entries[0].__class__(
            model_id=MODEL_IDS[0], trip_ttft_s=0.3, recover_ttft_s=0.2, deadline_s=0.15),
            settings.entries[1]))
        release = threading.Event()
        closed = threading.Event()

        class Stall:
            def __iter__(self) -> Any:
                yield {"messageStart": {}}
                release.wait(5)
                yield {"contentBlockDelta": {"delta": {"text": "late"}}}

            def close(self) -> None:
                closed.set()

        class Client:
            def converse_stream(self, **_: Any) -> dict[str, Any]:
                return {"stream": Stall()}

        deps = service.PollerDeps(store=MemoryStore(), bedrock=Client(), stage=STAGE,
                                  models=service.parse_models(models_env(MODEL_IDS[:2])),
                                  settings=lambda: settings, fixture=service.load_fixture())
        started = time.monotonic()
        sample = run(lambda: service.send_ping(deps, settings, 0), loop)
        waited = time.monotonic() - started
        assert sample.outcome is Outcome.ABANDONED and sample.ttft_s == 0.15
        assert waited >= 0.15
        release.set()
        assert closed.wait(2)

    def test_first_output_stops_when_cancelled_and_closes(self) -> None:
        cancel = threading.Event()
        cancel.set()
        bedrock = FakeBedrock(FakeTime())
        result = service.first_output(bedrock, {"modelId": "m"}, cancel=cancel, clock=time.monotonic)
        assert result == service.CallResult() and bedrock.streams[0].closed

    def test_first_output_without_output_is_an_error(self) -> None:
        class Client:
            def converse_stream(self, **_: Any) -> dict[str, Any]:
                return {"stream": iter([{"messageStart": {}}, {"messageStop": {}}])}

        result = service.first_output(Client(), {}, cancel=threading.Event(), clock=time.monotonic)
        assert result.cause is Outcome.ERROR and result.first_at is None


# --------------------------------------------------------------------------- logs

ALLOWED_FIELDS = {"entry", "model_id", "rank", "source", "preferred", "p50_ms", "p90_ms", "error_permille",
                  "samples", "pings", "sent", "ok", "missed", "errors", "version", "changed", "report_every",
                  "pinged", "decision", "items", "entries", "cause", "error_code", "error_class", "stack",
                  "cycle", "reason"}


def test_logs_carry_ids_numbers_and_causes_only(caplog: pytest.LogCaptureFixture) -> None:
    # Production state: the handler's first invocation runs ``obs.configure_logging``, which installs
    # the redacting record factory; every record then carries its private ``_owlbank_redacted`` marker.
    security_log.install_log_redaction()
    deps, t, bedrock, store = make_deps()
    memory = service.PollerMemory()
    bedrock.default[MODEL_IDS[0]] = Ping(error="ValidationException",
                                         message="The model returned the following errors: caller said 4111")
    bedrock.recheck_fails.add(MODEL_IDS[1])
    seed_state(store, t.wall(), [D, U, H], since=t.wall() - 1000)
    with caplog.at_level(logging.DEBUG, logger=service.log.name):
        cycle(deps, memory)
        store.fail_query = True
        t.advance(30)
        cycle(deps, memory)
    records = [r for r in caplog.records if r.name == service.log.name]
    assert records
    base = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"event", "message", "taskName"}
    line_base = {"ts", "level", "component", "event", "message", "logger"}
    formatter = obs.JsonFormatter()
    for r in records:
        # Private ``_``-prefixed record attributes (the redaction and scrub markers) are never
        # written: JsonFormatter skips them. Checked below on the formatted line itself.
        extra = {k for k in set(r.__dict__) - base if not k.startswith("_")}
        assert extra <= ALLOWED_FIELDS, extra
        written = set(json.loads(formatter.format(r))) - line_base
        assert written <= ALLOWED_FIELDS, written
        for k in extra - {"stack"}:
            v = r.__dict__[k]
            assert v is None or isinstance(v, int | float | bool) or k in {
                "model_id", "source", "decision", "cause", "error_code", "error_class", "reason"}
            if k == "model_id":
                assert v in MODEL_IDS
    assert "4111" not in caplog.text and "caller" not in caplog.text


# --------------------------------------------------------------------------- the handler


class TestHandler:
    def env(self, **extra: str) -> dict[str, str]:
        return {"OWLBANK_STAGE": "dev", "OWLBANK_MODELS": models_env(), "AWS_REGION": "us-east-1", **extra}

    def test_build_deps_reads_the_environment(self) -> None:
        deps = handler_mod.build_deps(self.env(OWLBANK_FAILOVER_TEST="3"), bedrock=object(), store=MemoryStore())
        assert deps.stage == "dev" and [m.id for m in deps.models] == list(MODEL_IDS)
        assert deps.test_delay_s == 3.0
        assert deps.settings() == Settings.for_models(list(MODEL_IDS))  # no policy section: defaults

    def test_build_deps_uses_the_policy_section(self) -> None:
        from owlbank.policy import Policy

        doc = json.loads((service.FIXTURE_PATH.parents[3] / "data" / "policy.json").read_text())
        doc["model_failover"] = {"pings_per_cycle": 2, "entries": {MODEL_IDS[0]: {"deadline_s": 3.5}}}
        store = MemoryStore()
        for item in Policy.to_items(doc):
            store.put("policy", item)
        settings = handler_mod.build_deps(self.env(), bedrock=object(), store=store).settings()
        assert settings.pings_per_cycle == 2 and settings.entry(0).deadline_s == 3.5

    def test_dynamo_store_table_names(self) -> None:
        deps = handler_mod.build_deps(self.env(OWLBANK_TABLE_MODELSTATS="ms-x", OWLBANK_TABLE_POLICY="pol-x"),
                                      bedrock=object())
        assert deps.store._names == {"modelstats": "ms-x", "policy": "pol-x"}  # type: ignore[attr-defined]

    @pytest.mark.parametrize(("env", "name"), [
        ({"OWLBANK_MODELS": models_env()}, "OWLBANK_STAGE"),
        ({"OWLBANK_STAGE": "dev"}, "OWLBANK_MODELS"),
        ({"OWLBANK_STAGE": "prod", "OWLBANK_MODELS": models_env(), "OWLBANK_FAILOVER_TEST": "2"},
         "OWLBANK_FAILOVER_TEST")])
    def test_build_deps_refuses(self, env: dict[str, str], name: str) -> None:
        with pytest.raises(service.PollerConfigError) as err:
            handler_mod.build_deps(env, bedrock=object(), store=MemoryStore())
        assert err.value.name == name

    def test_handler_runs_two_cycles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        deps, t, _, store = make_deps()
        monkeypatch.setattr(handler_mod, "_deps", deps)
        monkeypatch.setattr(handler_mod, "_memory", service.PollerMemory())

        class Context:
            def get_remaining_time_in_millis(self) -> int:
                return 120_000 - int(t.mono() * 1000)

        assert handler_mod.handler({}, Context()) == {"cycles": 2, "written": 2}
        assert stored_state(store) is not None

    def test_handler_skips_a_cycle_without_time_left(self, monkeypatch: pytest.MonkeyPatch) -> None:
        deps, t, _, _ = make_deps()
        monkeypatch.setattr(handler_mod, "_deps", deps)
        monkeypatch.setattr(handler_mod, "_memory", service.PollerMemory())

        class Context:  # 60 s left at the start: the second cycle (30 s in) would not fit
            def get_remaining_time_in_millis(self) -> int:
                return 60_000 - int(t.mono() * 1000)

        assert handler_mod.handler({}, Context()) == {"cycles": 1, "written": 1}

    def test_module_path_is_fixed(self) -> None:
        assert handler_mod.__name__ == "lambdas.model_poller.handler" and callable(handler_mod.handler)
        assert asyncio.iscoroutinefunction(service.run_invocation)

    def test_flaky_store_is_a_store(self) -> None:
        assert isinstance(FlakyStore(), MemoryStore) and InlineExecutor()


# --------------------------------------------------------------------------- metrics (EMF) and log fields


def _values(deps: service.PollerDeps, name: str) -> list[Any]:
    return [m[name] for m in metrics(deps) if name in m]


class TestMetrics:
    def test_record_shape(self) -> None:
        deps, t, _, _ = make_deps()
        cycle(deps, service.PollerMemory())
        records = metrics(deps)
        assert {n for r in records for n in r} >= {"_aws", "Service", "Environment"}
        for r in records:
            cw = r["_aws"]["CloudWatchMetrics"]
            assert len(cw) == 1 and cw[0]["Namespace"] == f"OwlBank/{STAGE}/Model"
            assert cw[0]["Dimensions"] == [["Service", "Environment"]]
            (metric,) = cw[0]["Metrics"]
            assert metric["Unit"] == "Count" and isinstance(r[metric["Name"]], int)
            assert r["Service"] == f"owlbank-{STAGE}-model-poller" and r["Environment"] == STAGE
            assert r["_aws"]["Timestamp"] == round(t.wall() * 1000)
            assert set(r) == {"_aws", "Service", "Environment", metric["Name"]}

    def test_heartbeat_once_per_written_cycle_only(self) -> None:
        deps, t, _, store = make_deps()
        memory = service.PollerMemory()
        cycle(deps, memory)
        t.advance(30)
        store.fail_write = True
        cycle(deps, memory)
        store.fail_write = False
        store.fail_get = True
        t.advance(30)
        cycle(deps, memory)
        store.fail_get = False
        store.fail_query = True
        t.advance(30)
        cycle(deps, memory)
        store.fail_query = False
        t.advance(30)
        cycle(deps, memory)
        assert _values(deps, "StateHeartbeat") == [1, 1]

    def test_heartbeat_not_on_a_conflict(self) -> None:
        deps, t, _, store = make_deps()
        seed_state(store, t.wall(), [H, H, H], version=2)

        class Racer(FlakyStore):
            def transact_write(self, puts: Any) -> None:
                seed_state(self, t.wall(), [H, H, H], version=3)
                super().transact_write(puts)

        racer = Racer()
        racer.put(service.TABLE, store.get(service.TABLE, state_key()))
        deps.store = racer
        assert not cycle(deps, service.PollerMemory()).written
        assert _values(deps, "StateHeartbeat") == []

    def test_usable_entries(self) -> None:
        deps, t, _, store = make_deps()
        memory = service.PollerMemory()
        cycle(deps, memory)
        t.advance(30)
        degrade_entry0(store, t.wall() - 5)
        add_samples(store, t.wall() - 5, 2, [], unavailable=1)
        cycle(deps, memory)
        store.fail_query = True
        t.advance(30)
        cycle(deps, memory)  # nothing computed: no value
        assert _values(deps, "UsableEntries") == [3, 1]

    def test_lines_go_to_stdout_as_their_own_json_objects(self, capsys: pytest.CaptureFixture[str]) -> None:
        """EMF bypasses logging: each stdout line is one JSON object with ``_aws`` at its root, even
        with the JSON log formatter on a stdout handler."""
        import sys

        from owlbank.obs import JsonFormatter

        deps, _, _, _ = make_deps()
        deps.metrics_out = None
        logger = logging.getLogger(service.log.name)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        old = logger.level
        logger.setLevel(logging.INFO)
        try:
            cycle(deps, service.PollerMemory())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old)
        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        emf = [line for line in lines if "_aws" in line]
        assert {next(iter(set(e) - {"_aws", "Service", "Environment"})) for e in emf} == {
            "StateHeartbeat", "UsableEntries"}
        assert all("_aws" not in line.get("message", "") for line in lines)


def test_transition_logs_carry_the_model_id(caplog: pytest.LogCaptureFixture) -> None:
    """``model_unavailable``, ``model_degraded``, ``model_recovered`` and ``model_available`` carry
    ``model_id`` (the infra metric filter matches ``$.model_id``)."""
    deps, t, bedrock, store = make_deps()
    memory = service.PollerMemory()
    seed_state(store, t.wall() - 2000, [D, U, H], since=t.wall() - 2000)
    with caplog.at_level(logging.INFO, logger=service.log.name):
        cycle(deps, memory)  # entry 1 rechecked: available
        for _ in range(8):  # entry 0 recovers on fast pings
            t.advance(30)
            cycle(deps, memory)
        t.advance(30)
        degrade_entry0(store, t.wall() - 5)
        add_samples(store, t.wall() - 5, 2, [], unavailable=1)
        cycle(deps, memory)
    seen = {ev(r): ev(r, "model_id") for r in caplog.records
            if ev(r) in {"model_unavailable", "model_degraded", "model_recovered", "model_available"}}
    assert seen == {"model_available": MODEL_IDS[1], "model_recovered": MODEL_IDS[0],
                    "model_degraded": MODEL_IDS[0], "model_unavailable": MODEL_IDS[2]}
