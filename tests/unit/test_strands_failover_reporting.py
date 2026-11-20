"""In the Strands client: the fleet reporter, the log lines and metrics, the
``model_id`` on Strands' TTFT and latency, and the tool loop's budget with a failover.

The rig is the one of ``test_strands_failover``: one scripted fake model per entry with the real ``GuardedModel``. Every
timed test runs on asyncio and uvloop; nothing relies on a timer firing at or after its due time.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from owlbank.failover import UNAVAILABLE, EntryState, FailoverState, FleetView, Outcome
from owlbank.failover.classify import INVALID_MODEL_MESSAGE, REQUEST_ERROR_PREFIX
from tac_bridge import agent_client_strands as acs
from tac_bridge.agent_client import AgentError
from tests.logutil import lines
from tests.strands_fakes import client_error, default_gateway, text_msg, tool_msg
from tests.unit.test_strands_failover import (  # noqa: F401 - ``run`` is a fixture
    DEADLINE_S,
    IDS,
    LOGGER,
    EntryModel,
    Hang,
    entries,
    failover_settings,
    make,
    run,
    spoken,
    turn,
)

CALLER = "my card is 4111 1111 1111 1111 and my name is Jane Roe"
NEW_EVENTS = {"model_failover", "model_degraded", "model_unavailable", "model_request_error"}


class Reporter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, Any, float | None]] = []

    def record(self, index: int, outcome: Any, ttft_s: float | None = None) -> None:
        self.calls.append((index, outcome, ttft_s))


class Events:
    def __init__(self) -> None:
        self.seen: list[tuple[str, Any]] = []

    def failover(self) -> None:
        self.seen.append(("failover", None))

    def unavailable(self) -> None:
        self.seen.append(("unavailable", None))

    def active(self, rank: int) -> None:
        self.seen.append(("active", rank))


def wired(fakes: list[EntryModel], **kw: Any) -> tuple[acs.StrandsAgentClient, Reporter, Events]:
    client = make(fakes, **kw)
    reporter, events = Reporter(), Events()
    client.use_fleet_reporter(reporter)
    client.use_failover_events(events)
    return client, reporter, events


def new_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [ln for ln in lines(caplog.records, LOGGER) if ln.get("event") in NEW_EVENTS]


# --------------------------------------------------------------------- the fleet reporter


def test_every_model_call_is_reported_with_its_outcome(run: Any) -> None:
    """OK with its TTFT; abandoned with the deadline it missed; hard errors without a TTFT."""
    client, reporter, _ = wired([EntryModel([Hang(), client_error("ThrottlingException")]),
                                 EntryModel([text_msg("one"), text_msg("two")])])
    run(turn(client))
    run(turn(client))
    (a, b, c, d) = reporter.calls
    assert a == (0, Outcome.ABANDONED, DEADLINE_S)
    assert b[:2] == (1, Outcome.OK) and b[2] is not None and 0 <= b[2] < 1
    assert c == (0, Outcome.THROTTLED, None)
    assert d[:2] == (1, Outcome.OK)


def test_an_unavailable_detection_is_reported(run: Any) -> None:
    client, reporter, _ = wired([EntryModel([client_error("ResourceNotFoundException")]),
                                 EntryModel([text_msg("ok")])])
    run(turn(client))
    assert reporter.calls[0] == (0, UNAVAILABLE, None)


def test_a_request_error_is_logged_never_reported(run: Any, caplog: pytest.LogCaptureFixture) -> None:
    """4a.2: not a sample, no state, ``model_request_error`` with the code only."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    err = client_error("ValidationException", REQUEST_ERROR_PREFIX + " " + CALLER)
    client, reporter, events = wired([EntryModel([err]), EntryModel()])
    with pytest.raises(AgentError):
        run(turn(client, CALLER))
    assert reporter.calls == []
    (ln,) = new_lines(caplog)
    assert (ln["event"], ln["model_id"], ln["error_code"], ln["entry"]) == (
        "model_request_error", IDS[0], "ValidationException", 0)
    assert client.failover.state(0) is EntryState.HEALTHY
    assert [e for e in events.seen if e[0] != "active"] == []
    assert "4111 1111" not in caplog.text and "Jane" not in caplog.text


def test_a_list_of_one_reports_nothing_and_logs_no_new_line(run: Any, caplog: pytest.LogCaptureFixture) -> None:
    """A list of one runs today's path, whatever is attached."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    client, reporter, events = wired([EntryModel([client_error("ValidationException", INVALID_MODEL_MESSAGE)])])
    with pytest.raises(AgentError):
        run(turn(client))
    assert reporter.calls == [] and events.seen == [] and new_lines(caplog) == []
    assert type(next(iter(client._calls.values())).build.result().agent.event_loop_metrics) \
        is acs.EventLoopMetrics  # noqa: SLF001 - no model_id tagging either


# --------------------------------------------------------------------- the log lines and metrics


def test_a_reissue_logs_model_failover_and_counts_it(run: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    client, _, events = wired([EntryModel([client_error("ModelErrorException", CALLER)]),
                               EntryModel([text_msg("fine")])])
    assert spoken(run(turn(client, CALLER))) == "fine"
    (ln,) = [x for x in new_lines(caplog) if x["event"] == "model_failover"]
    assert {k: ln[k] for k in ("turn", "from_model", "to_model", "cause", "from_entry", "to_entry", "error_code")} == {
        "turn": 1, "from_model": IDS[0], "to_model": IDS[1], "cause": "error", "from_entry": 0, "to_entry": 1,
        "error_code": "ModelErrorException"}
    assert ("failover", None) in events.seen
    assert "4111 1111" not in caplog.text and "Jane" not in caplog.text


def test_a_deadline_reissue_says_deadline_and_turns_count_per_call(run: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    client, _, _ = wired([EntryModel([text_msg("first"), Hang()]), EntryModel([text_msg("second")])])
    run(turn(client))
    run(turn(client))
    (ln,) = [x for x in new_lines(caplog) if x["event"] == "model_failover"]
    assert (ln["turn"], ln["cause"], ln["error_code"]) == (2, "deadline", None)


def test_a_turn_moved_down_by_the_verdict_logs_once_per_call(run: Any, caplog: pytest.LogCaptureFixture) -> None:
    """``cause=degraded``: the first turn below the call's previous entry; ``ActiveRank`` on each
    turn's start (the monitor writes only changes)."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    client, _, events = wired([EntryModel(), EntryModel([text_msg("a"), text_msg("b")])])
    client.use_fleet_view(lambda: FleetView((EntryState.DEGRADED, EntryState.HEALTHY)))
    run(turn(client))
    run(turn(client))
    failovers = [x for x in new_lines(caplog) if x["event"] == "model_failover"]
    assert [(x["turn"], x["cause"], x["from_model"], x["to_model"]) for x in failovers] == [
        (1, "degraded", IDS[0], IDS[1])]
    assert [e for e in events.seen if e[0] == "active"] == [("active", 1), ("active", 1)]
    assert events.seen.count(("failover", None)) == 1
    assert client.active_rank() == 1


def test_an_unavailable_entry_is_logged_for_the_alarm(run: Any, caplog: pytest.LogCaptureFixture) -> None:
    """4a.4: ``model_unavailable`` (ERROR) with ``model_id`` (the alarm's filter field) and the error
    code, once; the ``Unavailable`` metric once."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    client, _, events = wired([EntryModel([client_error("ValidationException", INVALID_MODEL_MESSAGE)]),
                               EntryModel([text_msg("a"), text_msg("b")])])
    run(turn(client))
    run(turn(client))
    (ln,) = [x for x in new_lines(caplog) if x["event"] == "model_unavailable"]
    assert {k: ln[k] for k in ("model_id", "error_code", "entry", "preferred", "scope")} == {
        "model_id": IDS[0], "error_code": "ValidationException", "entry": 0, "preferred": True, "scope": "process"}
    rec = next(r for r in caplog.records if getattr(r, "event", None) == "model_unavailable")
    assert rec.levelno == logging.ERROR
    assert events.seen.count(("unavailable", None)) == 1


def test_a_run_of_misses_logs_model_degraded_with_the_windows_numbers(run: Any,
                                                                     caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    n = failover_settings(2).miss_run
    client, _, _ = wired([EntryModel([Hang() for _ in range(n)]), EntryModel([text_msg("x") for _ in range(n + 1)])])
    for _ in range(n + 1):
        run(turn(client))
    (ln,) = [x for x in new_lines(caplog) if x["event"] == "model_degraded"]
    assert (ln["model_id"], ln["miss_run"], ln["samples"], ln["scope"]) == (IDS[0], n, n, "process")
    assert ln["p90_ms"] == round(DEADLINE_S * 1000)
    assert client.failover.state(0) is EntryState.DEGRADED


def test_a_failing_events_sink_never_breaks_the_turn(run: Any) -> None:
    class Broken:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("metrics down")

    client = make([EntryModel([Hang()]), EntryModel([text_msg("ok")])])
    client.use_failover_events(Broken())  # type: ignore[arg-type]
    assert spoken(run(turn(client))) == "ok"


# --------------------------------------------------------------------- model_id on TTFT and latency


class FakeHistogram:
    def __init__(self) -> None:
        self.records: list[tuple[Any, Any]] = []

    def record(self, amount: Any, attributes: Any = None, context: Any = None) -> None:
        self.records.append((amount, attributes))

    add = record  # Strands' counters


class FakeMetricsClient:
    def __init__(self) -> None:
        self.hists: dict[str, FakeHistogram] = {}

    def __getattr__(self, name: str) -> FakeHistogram:
        if name.startswith("__"):
            raise AttributeError(name)
        return self.hists.setdefault(name, FakeHistogram())


def test_ttft_and_latency_carry_the_calls_model_id(run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMetricsClient()
    monkeypatch.setattr(acs, "MetricsClient", lambda: fake)
    client = make([EntryModel([Hang(), text_msg("a")]), EntryModel([text_msg("b")])])
    run(turn(client))  # entry 0 abandoned (no record), entry 1 answers
    run(turn(client))  # entry 0 answers
    # The fakes report latencyMs only (Strands records TTFT only when the stream gives one).
    assert [attrs for _, attrs in fake.hists["event_loop_latency"].records] == [
        {"model_id": IDS[1]}, {"model_id": IDS[0]}]
    agent = next(iter(client._calls.values())).build.result().agent  # noqa: SLF001
    agent.event_loop_metrics.update_metrics({"latencyMs": 5, "timeToFirstByteMs": 7})
    ttft = fake.hists["model_time_to_first_token"].records
    # Strands records a TTFT only when it has one (whether the fakes' streams give one depends on
    # Strands' streaming state); whatever it recorded carries the answering entry's model_id.
    assert ttft[-1] == (7, {"model_id": IDS[0]})
    assert [attrs for _, attrs in ttft[:-1]] in ([], [{"model_id": IDS[1]}, {"model_id": IDS[0]}],
                                                 [{"model_id": IDS[1]}], [{"model_id": IDS[0]}])
    other = [attrs for n, h in fake.hists.items() if n not in acs.MODEL_TAGGED_INSTRUMENTS for _, attrs in h.records]
    assert all(not attrs or "model_id" not in attrs for attrs in other)


# --------------------------------------------------------------------- the loop budget


TOOL = "level0___faq_lookup"


def test_a_tool_loop_with_a_failover_stays_within_max_iterations(run: Any) -> None:
    """Three tool hops and a failover are five model calls: inside ``max_iterations`` (6)."""
    hops = [tool_msg((TOOL, {"q": f"q{i}"})) for i in range(3)]
    m0, m1 = EntryModel([*hops, Hang()]), EntryModel([text_msg("done")])
    client = make([m0, m1], gateway=default_gateway())
    assert client.settings.max_iterations == 6
    assert spoken(run(turn(client))) == "done"
    assert len(m0.calls) + len(m1.calls) == 5


def test_three_tool_hops_and_two_failovers_use_exactly_the_budget(run: Any) -> None:
    hops = [tool_msg((TOOL, {"q": f"q{i}"})) for i in range(3)]
    m0, m1, m2 = EntryModel([*hops, Hang()]), EntryModel([Hang()]), EntryModel([text_msg("done")])
    client = make([m0, m1, m2], gateway=default_gateway(), models=entries(3))
    assert spoken(run(turn(client))) == "done"
    assert len(m0.calls) + len(m1.calls) + len(m2.calls) == client.settings.max_iterations


def test_the_state_is_shared_with_the_fleet_view_source(run: Any) -> None:
    """``active_rank`` follows the same choice the turn makes (pure, in memory)."""
    state = FailoverState(failover_settings(2))
    client = make([EntryModel(), EntryModel()], state=state)
    assert client.active_rank() == 0
    client.use_fleet_view(lambda: FleetView((EntryState.UNAVAILABLE, EntryState.HEALTHY)))
    assert client.active_rank() == 1
