"""Per-turn model choice and reissue in the Strands client.

The rig is the existing Strands one (``tests/strands_fakes.py``) with one
scripted fake model per list entry, the real ``GuardedModel`` around each, and the fake AgentCore
Memory where history matters. Every timed test runs on asyncio and on uvloop (the production loop);
nothing relies on a timer firing at or after its due time.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Callable, Coroutine
from dataclasses import replace
from typing import Any

import pytest
import uvloop
from botocore.exceptions import EndpointConnectionError
from strands.types.exceptions import ModelThrottledException

from owlbank.failover import UNAVAILABLE, EntryState, FailoverState, FleetView, Outcome, Settings
from owlbank.failover.classify import INVALID_MODEL_MESSAGE, REQUEST_ERROR_PREFIX
from tac_bridge.agent_client import AgentError, TextDelta, ToolResult, TurnEnd
from tac_bridge.agent_client_strands import (
    FIRST_OUTPUT_DEADLINE_CODE,
    ModelEntry,
    StrandsAgentClient,
    StrandsSettings,
)
from tac_bridge.guarded_model import PRODUCERS_KEY, GuardedModel
from tests.logutil import lines
from tests.strands_fakes import (
    Call,
    FakeMemoryData,
    FakeModel,
    ScriptError,
    client_error,
    collect,
    default_gateway,
    light_memory_client,
    memory_factory,
    text_msg,
    tool_msg,
)

SID = "CA" + "8" * 32
LOGGER = "tac_bridge.agent_client_strands"
IDS = ("us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5", "us.anthropic.claude-sonnet-5-5")
#: The first-token deadline of every entry but the last in these tests (seconds).
DEADLINE_S = 0.05
#: A model that never answers within any deadline here.
HANG_S = 30.0

LOOPS = {"asyncio": asyncio.new_event_loop, "uvloop": uvloop.new_event_loop}


@pytest.fixture(params=sorted(LOOPS))
def run(request: pytest.FixtureRequest) -> Callable[[Coroutine[Any, Any, Any]], Any]:
    factory = LOOPS[request.param]

    def _run(coro: Coroutine[Any, Any, Any]) -> Any:
        with asyncio.Runner(loop_factory=factory) as runner:
            return runner.run(coro)

    return _run


class Hang:
    """A scripted model call that sends nothing for ``seconds`` (a stalled first token)."""

    def __init__(self, seconds: float = HANG_S) -> None:
        self.seconds = seconds


class Stubborn:
    """A scripted model call that hangs and, once cancelled, takes ``linger_s`` to wind down."""

    def __init__(self, linger_s: float) -> None:
        self.linger_s = linger_s


class EntryModel(FakeModel):
    """:class:`FakeModel` that also takes :class:`Hang` and :class:`Stubborn` steps and records each
    call's ``invocation_state`` producer record. An unscripted call fails the test."""

    def __init__(self, script: list[Any] = (), **kw: Any) -> None:  # type: ignore[assignment]
        super().__init__(script, **kw)
        self.producers: list[dict[str, int] | None] = []

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, *,
                     tool_choice: Any = None, system_prompt_content: Any = None, **kwargs: Any) -> Any:
        self.calls.append(Call(copy.deepcopy(list(messages)), copy.deepcopy(list(tool_specs or [])),
                               copy.deepcopy(system_prompt_content)))
        state = kwargs.get("invocation_state") or {}
        record = state.get(PRODUCERS_KEY)
        self.producers.append(None if record is None else dict(record))
        if not self.script:
            raise AssertionError("unscripted model call")
        step = self.script.pop(0)
        if isinstance(step, Stubborn):
            try:
                await asyncio.sleep(HANG_S)
            except asyncio.CancelledError:
                await asyncio.sleep(step.linger_s)
                raise
        if isinstance(step, Hang):
            await asyncio.sleep(step.seconds)
            step = text_msg("too late")
        if isinstance(step, BaseException):
            raise step
        for chunk in step:
            await asyncio.sleep(0)
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


def entries(n: int, **overrides: Any) -> tuple[ModelEntry, ...]:
    """``n`` entries with the shipped shapes: Haiku 5.5 adaptive + stream, Haiku 4.5 off + message,
    Sonnet 5.5 adaptive + stream (``overrides`` apply to every entry)."""
    shapes = [{"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000},
              {"thinking": "off", "text_release": "message", "max_tokens": 400},
              {"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000}]
    return tuple(ModelEntry(id=IDS[i], **{**shapes[i], **overrides}) for i in range(n))  # type: ignore[arg-type]


def failover_settings(n: int, deadline_s: float = DEADLINE_S, **kw: Any) -> Settings:
    base = Settings.for_models(IDS[:n], **kw)
    return replace(base, entries=tuple(replace(e, deadline_s=None if i == n - 1 else deadline_s)
                                       for i, e in enumerate(base.entries)))


def make(fakes: list[EntryModel], *, memory: FakeMemoryData | None = None, state: FailoverState | None = None,
         gateway: Any = None, **kw: Any) -> StrandsAgentClient:
    n = len(fakes)
    by_id = dict(zip(IDS, fakes, strict=False))
    s = StrandsSettings(models=kw.pop("models", entries(n)), region="us-east-1",
                        memory_id="mem-1" if memory is not None else "", **kw)
    return StrandsAgentClient(
        s, model_factory=lambda one, _sid: by_id[one.primary.id], gateway_tools=gateway,
        session_manager_factory=memory_factory(memory) if memory is not None else None,
        failover_state=state or FailoverState(failover_settings(n)))


async def turn(client: StrandsAgentClient, text: str | None = "hello", results: Any = ()) -> list[Any]:
    return await collect(client.turn(SID, user_text=text, tool_results=list(results), system_prompts={"haiku-5-5": "BASE"},
                                     system_suffix="Clock: 12:00"))


def spoken(events: list[Any]) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


def user_texts(messages: list[dict[str, Any]]) -> list[str]:
    return [b["text"] for m in messages if m["role"] == "user" for b in m["content"] if "text" in b]


def has_reasoning(call: Call) -> bool:
    return any("reasoningContent" in b for m in call.messages for b in m["content"])


def agent_of(client: StrandsAgentClient) -> Any:
    return client._calls[SID].build.result().agent


# --------------------------------------------------------------------------------- the model list


def test_a_list_of_one_runs_todays_path_without_a_wrapper(run: Any) -> None:
    """A list of one: the agent's model is the bare model (no deadline, no filter), as today."""
    m = EntryModel([text_msg("hi")])
    client = make([m])
    events = run(turn(client))
    assert agent_of(client).model is m and spoken(events) == "hi"
    assert events[-1] == TurnEnd("end_turn") and events[-1].model == IDS[0]


def test_a_list_of_one_fails_a_pre_output_throttle_as_today(run: Any) -> None:
    client = make([EntryModel([client_error("ThrottlingException")])])
    with pytest.raises(AgentError) as ei:
        run(turn(client))
    assert (ei.value.code, ei.value.retryable) == ("ThrottlingException", True)


def test_each_entry_runs_guarded_on_its_own_model(run: Any) -> None:
    models = [EntryModel([text_msg("hi")]), EntryModel(), EntryModel()]
    client = make(models)
    run(turn(client))
    guards = client._calls[SID].build.result().models.models
    assert [type(g) for g in guards] == [GuardedModel] * 3
    assert [g.inner for g in guards] == models and [g.entry for g in guards] == [0, 1, 2]
    assert [g.deadline_s for g in guards] == [DEADLINE_S, DEADLINE_S, None]  # the last has none


def test_the_default_factory_builds_each_entrys_bedrock_model_once() -> None:
    """One ``BedrockModel`` per entry for the process, each with its own parameters."""
    s = StrandsSettings(models=entries(3, prompt_caching=True), region="us-east-1")
    client = StrandsAgentClient(s)
    built = [client._model(SID, i) for i in range(3)]
    assert [client._model("other", i) for i in range(3)] == built
    cfg = [m.get_config() for m in built]
    assert [c["model_id"] for c in cfg] == list(IDS)
    assert [c["max_tokens"] for c in cfg] == [2000, 400, 2000]
    assert cfg[1]["additional_request_fields"] == {"thinking": {"type": "disabled"}}
    assert cfg[0]["additional_request_fields"]["thinking"]["type"] == "adaptive"


@pytest.mark.parametrize(("bad", "match"), [
    ({"models": ()}, "models must be a tuple of 1 to 3"),
    ({"models": entries(3) + (ModelEntry(id="x", thinking="default", max_tokens=10),)}, "1 to 3"),
    ({"models": (entries(1)[0], entries(1)[0])}, "must not repeat a model ID"),
    ({"models": [entries(1)[0]]}, "must be a tuple"),
])
def test_the_list_is_validated(bad: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        StrandsSettings(region="us-east-1", **bad)


def test_each_entry_is_validated_with_the_text_release_rule() -> None:
    """A parameter set the model can't run fails naming the rule (the app adds the entry)."""
    with pytest.raises(ValueError, match="text_release=stream needs thinking adaptive or default"):
        ModelEntry(id=IDS[1], thinking="off", max_tokens=400, text_release="stream")
    for thinking in ("between_tools", "enabled:1024", "default", "adaptive"):
        assert ModelEntry(id=IDS[2], thinking=thinking, max_tokens=2000).text_release == "stream"
    with pytest.raises(ValueError, match="budget must be below max_tokens"):
        ModelEntry(id=IDS[1], thinking="enabled:1024", max_tokens=1024, text_release="message")


def test_the_failover_test_delay_is_refused_at_prod() -> None:
    with pytest.raises(ValueError, match="refused at stage=prod"):
        StrandsSettings(models=entries(2), region="us-east-1", stage="prod", failover_test_s=1.0)
    s = StrandsSettings(models=entries(2), region="us-east-1", stage="dev", failover_test_s=1.0)
    client = StrandsAgentClient(s, model_factory=lambda _s, _sid: EntryModel())
    assert [g.test_delay_s for g in client._call_models(SID).models] == [1.0, None]


# ------------------------------------------------------------------------------ reissue


def test_a_slow_first_token_reissues_to_the_next_entry_before_anything_is_spoken(run: Any) -> None:
    m0, m1 = EntryModel([Hang()]), EntryModel([text_msg("from the second")])
    client = make([m0, m1])
    events = run(turn(client))
    assert spoken(events) == "from the second"
    assert (len(m0.calls), len(m1.calls)) == (1, 1)
    assert user_texts(m1.calls[0].messages) == ["hello"]  # the caller's message, once
    assert events[-1].model == IDS[1]
    assert client.failover.stats(0).calls == 1  # the abandoned call is a sample of the first entry
    assert user_texts(agent_of(client).messages) == ["hello"]


HARD_ERRORS = [
    client_error("ThrottlingException"), client_error("ServiceUnavailableException"),
    client_error("ModelTimeoutException"), client_error("ModelErrorException"),
    client_error("InternalServerException"), client_error("ModelNotReadyException"),
    EndpointConnectionError(endpoint_url="https://bedrock-runtime.invalid"),
    client_error("AccessDeniedException"), client_error("ResourceNotFoundException"),
    client_error("ValidationException", INVALID_MODEL_MESSAGE), ScriptError("no code at all"),
]


@pytest.mark.parametrize("error", HARD_ERRORS, ids=lambda e: getattr(e, "response", {}).get("Error", {}).get(
    "Code", type(e).__name__))
def test_each_pre_output_hard_error_reissues_to_the_next_entry(run: Any, error: BaseException) -> None:
    m0, m1 = EntryModel([[{"messageStart": {"role": "assistant"}}, error]]), EntryModel([text_msg("ok")])
    client = make([m0, m1])
    events = run(turn(client))
    assert spoken(events) == "ok" and (len(m0.calls), len(m1.calls)) == (1, 1)
    assert user_texts(agent_of(client).messages) == ["hello"]


def test_an_unavailable_entry_is_marked_and_skipped_on_the_next_turn(run: Any) -> None:
    m0 = EntryModel([client_error("ValidationException", INVALID_MODEL_MESSAGE)])
    m1 = EntryModel([text_msg("one"), text_msg("two")])
    client = make([m0, m1])
    run(turn(client))
    assert client.failover.state(0) is EntryState.UNAVAILABLE
    assert spoken(run(turn(client, "again"))) == "two" and len(m0.calls) == 1  # 4a.3: no caller traffic


def test_a_request_error_is_never_a_failover(run: Any) -> None:
    """4a.2: the request's fault fails the turn as today and changes no entry's state."""
    m0 = EntryModel([client_error("ValidationException", f"{REQUEST_ERROR_PREFIX} bad field")])
    m1 = EntryModel()
    client = make([m0, m1])
    with pytest.raises(AgentError) as ei:
        run(turn(client))
    assert (ei.value.code, ei.value.retryable) == ("ValidationException", False)
    assert m1.calls == [] and client.failover.stats(0).calls == 0


def test_no_failover_after_the_first_output(run: Any) -> None:
    """Once text was streamed the call completes (or fails) on its model."""
    m0 = EntryModel([[{"messageStart": {"role": "assistant"}}, {"contentBlockStart": {"start": {}}},
                      {"contentBlockDelta": {"delta": {"text": "Partly "}}}, client_error("ThrottlingException")]])
    m1 = EntryModel()
    client = make([m0, m1])

    async def go() -> list[Any]:
        out: list[Any] = []
        with pytest.raises(AgentError) as ei:
            async for ev in client.turn(SID, user_text="hello", tool_results=[], system_prompts={"haiku-5-5": "BASE"}):
                out.append(ev)
        assert ei.value.code == "ThrottlingException"
        return out

    assert spoken(run(go())) == "Partly " and m1.calls == []


def test_a_started_tool_use_is_output_too(run: Any) -> None:
    m0 = EntryModel([[{"messageStart": {"role": "assistant"}},
                      {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": "level0___faq_lookup"}}}},
                      ScriptError("after the tool use started")]])
    client = make([m0, EntryModel()], gateway=default_gateway())
    with pytest.raises(AgentError):
        run(turn(client))


def test_every_entry_unusable_fails_the_turn_with_the_last_error(run: Any) -> None:
    """The last entry's error surfaces as the turn's AgentError (by its code)."""
    models = [EntryModel([Hang()]), EntryModel([client_error("ThrottlingException")]),
              EntryModel([client_error("ServiceUnavailableException")])]
    client = make(models)
    with pytest.raises(AgentError) as ei:
        run(turn(client))
    assert (ei.value.code, ei.value.retryable) == ("ServiceUnavailableException", True)
    assert [len(m.calls) for m in models] == [1, 1, 1]
    assert user_texts(agent_of(client).messages) == ["hello"]


def test_a_deadline_with_nowhere_to_go_fails_retryable(run: Any) -> None:
    m0, m1 = EntryModel([Hang()]), EntryModel()
    state = FailoverState(failover_settings(2))
    state.record(1, UNAVAILABLE)
    client = make([m0, m1], state=state)
    with pytest.raises(AgentError) as ei:
        run(turn(client))
    assert (ei.value.code, ei.value.retryable) == (FIRST_OUTPUT_DEADLINE_CODE, True) and m1.calls == []


def test_an_unavailable_last_resort_fails_not_retryable(run: Any) -> None:
    m0, m1 = EntryModel([client_error("ThrottlingException")]), EntryModel([client_error("AccessDeniedException")])
    with pytest.raises(AgentError) as ei:
        run(turn(make([m0, m1])))
    assert (ei.value.code, ei.value.retryable) == ("AccessDeniedException", False)


def test_a_throttle_on_the_last_entry_keeps_the_throttling_code(run: Any) -> None:
    """Strands' ``ModelThrottledException`` before any output on the last entry: the turn's error is
    ``ThrottlingException``, retryable, as without failover (the journal's value is unchanged)."""
    m0, m1 = EntryModel([Hang()]), EntryModel([ModelThrottledException("slow down")])
    with pytest.raises(AgentError) as ei:
        run(turn(make([m0, m1])))
    assert (ei.value.code, ei.value.retryable) == ("ThrottlingException", True)
    assert len(m1.calls) == 1


def test_a_cancel_while_an_abandoned_call_winds_down_is_not_reissued(run: Any) -> None:
    """The turn is cancelled (barge-in) just after entry 0 missed its deadline, while its read is
    still winding down: the cancellation propagates, entry 1 gets no call and no sample is taken."""
    m0, m1 = EntryModel([Stubborn(linger_s=0.5)]), EntryModel([text_msg("too late")])
    client = make([m0, m1])
    samples: list[Any] = []
    record = client.failover.record

    def spy(*args: Any, **kwargs: Any) -> Any:
        samples.append(args)
        return record(*args, **kwargs)

    client.failover.record = spy  # type: ignore[method-assign]

    async def barge_in() -> None:
        task = asyncio.ensure_future(turn(client))
        while not m0.calls:
            await asyncio.sleep(0.005)
        await asyncio.sleep(DEADLINE_S + 0.05)  # past the deadline: the guard is cleaning up
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(barge_in())
    assert m1.calls == [] and samples == []


# ------------------------------------------------------------------- the choice per turn


def test_the_turn_starts_on_the_choice_and_skips_a_degraded_entry(run: Any) -> None:
    m0, m1 = EntryModel(), EntryModel([text_msg("second")])
    state = FailoverState(failover_settings(2))
    for _ in range(state.settings.miss_run):  # A run of misses degrades the entry
        state.record(0, Outcome.ABANDONED)
    assert state.state(0) is EntryState.DEGRADED
    events = run(turn(make([m0, m1], state=state)))
    assert spoken(events) == "second" and m0.calls == []  # No caller turn to test it


def test_the_fleet_verdict_reaches_the_choice(run: Any) -> None:
    """The seam: ``use_fleet_view`` gives ``choose`` and ``next_after`` the fleet verdict."""
    m0, m1, m2 = EntryModel(), EntryModel([Hang()]), EntryModel([text_msg("third")])
    client = make([m0, m1, m2])
    client.use_fleet_view(lambda: FleetView((EntryState.DEGRADED, EntryState.HEALTHY, EntryState.HEALTHY)))
    assert spoken(run(turn(client))) == "third" and m0.calls == [] and len(m1.calls) == 1


def test_a_failing_or_mismatched_fleet_view_means_the_local_picture(run: Any) -> None:
    def broken() -> FleetView:
        raise RuntimeError("table unreadable")

    for source in (broken, lambda: FleetView((EntryState.DEGRADED,))):
        m0 = EntryModel([text_msg("first")])
        client = make([m0, EntryModel()])
        client.use_fleet_view(source)
        assert spoken(run(turn(client))) == "first"


def test_a_tool_loop_stays_on_its_entry(run: Any) -> None:
    """The turn's later model calls keep the entry chosen at its start."""
    m0 = EntryModel([tool_msg(("level0___faq_lookup", {"q": "hours"}), reasoning="r0"),
                     text_msg("we open at nine")])
    m1 = EntryModel()
    client = make([m0, m1], gateway=default_gateway())
    events = run(turn(client))
    assert spoken(events) == "we open at nine" and len(m0.calls) == 2 and m1.calls == []
    assert m0.producers[0] == {}  # a new agent: nothing produced yet
    assert list(m0.producers[1].values()) == [0]  # the tool use's reasoning, recorded as entry 0's
    assert has_reasoning(m0.calls[1])  # and sent back to entry 0 on the hop


def test_a_tool_loop_call_that_fails_moves_on_and_continues_from_the_tool_result(run: Any) -> None:
    m0 = EntryModel([tool_msg(("level0___faq_lookup", {"q": "hours"})), Hang()])
    m1 = EntryModel([text_msg("we open at nine")])
    client = make([m0, m1], gateway=default_gateway())
    events = run(turn(client))
    assert spoken(events) == "we open at nine"
    last = m1.calls[0].messages[-1]
    assert last["role"] == "user" and "toolResult" in last["content"][0]
    assert user_texts(agent_of(client).messages) == ["hello"]
    roles = [m["role"] for m in agent_of(client).messages]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_a_reissued_turn_after_tool_results_answers_once(run: Any) -> None:
    """An inline tool's result, then the model call on it fails before any output: the next entry
    answers from the history (the result is not sent again)."""
    m0 = EntryModel([tool_msg(("t1", "get_balance", {"account": "checking"})), Hang()])
    m1 = EntryModel([text_msg("you have money")])
    client = make([m0, m1])
    run(turn(client, "my balance?"))
    events = run(turn(client, None, [ToolResult("t1", {"balance": "ok"})]))
    assert spoken(events) == "you have money"
    msgs = agent_of(client).messages
    assert sum(1 for m in msgs for b in m["content"] if "toolResult" in b) == 1
    assert user_texts(msgs) == ["my balance?"]


# ------------------------------------------------------- the rest of the turn on its entry


class Events:
    """The ``FailoverEvents`` seam, recorded."""

    def __init__(self) -> None:
        self.seen: list[tuple[Any, ...]] = []

    def failover(self) -> None:
        self.seen.append(("failover",))

    def unavailable(self) -> None:
        self.seen.append(("unavailable",))

    def active(self, rank: int) -> None:
        self.seen.append(("active", rank))


async def hop(client: StrandsAgentClient, turn_id: int | None, text: str | None = None,
              results: Any = (), *, record: bool = False) -> list[Any]:
    prompts = {"haiku-5-5": "BASE"}
    if record:
        return await collect(client.record_results(SID, list(results), system_prompts=prompts, turn_id=turn_id))
    return await collect(client.turn(SID, user_text=text, tool_results=list(results), system_prompts=prompts,
                                     turn_id=turn_id))


def client_lines(caplog: pytest.LogCaptureFixture, name: str) -> list[dict[str, Any]]:
    return [e for e in lines(caplog.records, LOGGER) if e["event"] == name]


BALANCE_TOOL = ("t1", "get_balance", {"account": "checking"})
BALANCE = [ToolResult("t1", {"balance": "ok"})]


def test_a_later_hop_of_a_moved_turn_stays_and_the_next_turn_chooses_afresh(run: Any) -> None:
    m0, m1 = EntryModel([Hang(), text_msg("preferred again")]), EntryModel([tool_msg(BALANCE_TOOL), text_msg("ok")])
    client = make([m0, m1])
    run(hop(client, 1, "my balance?"))
    assert spoken(run(hop(client, 1, results=BALANCE))) == "ok"
    assert [len(m.calls) for m in (m0, m1)] == [1, 2]
    assert spoken(run(hop(client, 2, "thanks"))) == "preferred again"


def test_a_call_without_a_turn_id_chooses_afresh(run: Any) -> None:
    m0, m1 = EntryModel([Hang(), text_msg("from the first")]), EntryModel([tool_msg(BALANCE_TOOL)])
    client = make([m0, m1])
    run(hop(client, None, "my balance?"))
    assert spoken(run(hop(client, None, results=BALANCE))) == "from the first"


def test_a_later_hop_whose_entry_is_now_unavailable_moves_on_logged_and_counted(
        run: Any, caplog: pytest.LogCaptureFixture) -> None:
    """The turn moved 0 -> 1; before its next hop entry 1 is marked
    unavailable, so the hop goes on to entry 2 (never back to 0), with a ``model_failover`` line
    (cause ``unavailable``) and the ``Failover`` metric. The next turn chooses afresh."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    m0 = EntryModel([Hang(), text_msg("preferred again")])
    m1 = EntryModel([tool_msg(BALANCE_TOOL)])
    m2 = EntryModel([text_msg("from the third")])
    client = make([m0, m1, m2])
    events = Events()
    client.use_failover_events(events)
    run(hop(client, 1, "my balance?"))
    client.failover.record(1, UNAVAILABLE)
    events.seen.clear()
    caplog.clear()
    assert spoken(run(hop(client, 1, results=BALANCE))) == "from the third"
    assert [len(m.calls) for m in (m0, m1, m2)] == [1, 1, 1]
    (line,) = client_lines(caplog, "model_failover")
    assert (line["from_entry"], line["to_entry"], line["cause"], line["decision"]) == (
        1, 2, "unavailable", "start_on_next_model")
    assert ("failover",) in events.seen and ("active", 2) in events.seen
    assert client_lines(caplog, "model_turn_sticky") == []  # it did not stay: a failover, logged once
    assert spoken(run(hop(client, 2, "thanks"))) == "preferred again"


def test_a_record_only_hop_reports_nothing_and_keeps_the_move(run: Any, caplog: pytest.LogCaptureFixture) -> None:
    """a hop that records results calls no model, so no ``model_turn_sticky`` line and no
    ``ActiveRank`` event; the turn's move is kept for its next hop."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    m0, m1 = EntryModel([Hang()]), EntryModel([tool_msg(BALANCE_TOOL), text_msg("next words")])
    client = make([m0, m1])
    events = Events()
    client.use_failover_events(events)
    run(hop(client, 1, "my balance?"))
    events.seen.clear()
    caplog.clear()
    run(hop(client, 1, results=BALANCE, record=True))
    assert events.seen == []
    assert client_lines(caplog, "model_turn_sticky") == [] and client_lines(caplog, "model_failover") == []
    assert spoken(run(hop(client, 1, "and the rest?"))) == "next words"
    assert [len(m.calls) for m in (m0, m1)] == [1, 2]


def test_a_retry_after_every_entry_failed_chooses_afresh(run: Any) -> None:
    """the turn's hop fails on 0 (a miss) and then on 1 (a throttle, nowhere left); the
    bridge's retry of that hop is not pinned to the entry that just failed: every entry is
    abandoned, so it chooses afresh among all."""
    m0 = EntryModel([Hang(), text_msg("from the first")])
    m1 = EntryModel([[{"messageStart": {"role": "assistant"}}, client_error("ThrottlingException")]])
    client = make([m0, m1])
    with pytest.raises(AgentError):
        run(hop(client, 1, "hello"))
    assert spoken(run(hop(client, 1, "hello"))) == "from the first"
    assert [len(m.calls) for m in (m0, m1)] == [2, 1]


def test_a_retry_after_the_last_entry_failed_tries_one_not_abandoned(run: Any) -> None:
    """three entries: 0 misses, 1 throttles with 2 unavailable (nowhere left). Entry 2 is
    usable again by the retry: the retry starts there, not on 1 (failed) or 0 (abandoned)."""
    m0, m2 = EntryModel([Hang()]), EntryModel([text_msg("from the third")])
    m1 = EntryModel([[{"messageStart": {"role": "assistant"}}, client_error("ThrottlingException")]])
    client = make([m0, m1, m2])
    fleet = [FleetView((EntryState.HEALTHY, EntryState.HEALTHY, EntryState.UNAVAILABLE))]
    client.use_fleet_view(lambda: fleet[0])
    with pytest.raises(AgentError):
        run(hop(client, 1, "hello"))
    fleet[0] = FleetView((EntryState.HEALTHY, EntryState.HEALTHY, EntryState.HEALTHY))
    assert spoken(run(hop(client, 1, "hello"))) == "from the third"
    assert [len(m.calls) for m in (m0, m1, m2)] == [1, 1, 1]


def test_a_retry_with_the_rest_unavailable_goes_to_one_not_abandoned(
        run: Any, caplog: pytest.LogCaptureFixture) -> None:
    """0 misses, 1 throttles, 2 is unavailable and stays so. The retry does
    not go back to 0 (abandoned): it follows choose's rule over the entries not abandoned, so the
    unavailable 2 (the last); the change from 1 is a ``model_failover`` (cause ``exhausted``)."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    m0, m2 = EntryModel([Hang(), text_msg("wrong")]), EntryModel([text_msg("from the third")])
    m1 = EntryModel([[{"messageStart": {"role": "assistant"}}, client_error("ThrottlingException")]])
    client = make([m0, m1, m2])
    client.use_fleet_view(lambda: FleetView((EntryState.HEALTHY, EntryState.HEALTHY, EntryState.UNAVAILABLE)))
    with pytest.raises(AgentError):
        run(hop(client, 1, "hello"))
    caplog.clear()
    assert spoken(run(hop(client, 1, "hello"))) == "from the third"
    assert [len(m.calls) for m in (m0, m1, m2)] == [1, 1, 1]
    (line,) = client_lines(caplog, "model_failover")
    assert (line["from_entry"], line["to_entry"], line["cause"]) == (1, 2, "exhausted")


def test_the_move_survives_a_rebuilt_agent(run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """the agent is dropped mid-turn (as after a failed history write) and rebuilt from
    Memory; the turn's next hop still starts on the entry it moved to."""
    light_memory_client(monkeypatch)
    data = FakeMemoryData()
    m0, m1 = EntryModel([Hang(), text_msg("wrong")]), EntryModel([tool_msg(BALANCE_TOOL), text_msg("ok")])
    client = make([m0, m1], memory=data)
    run(hop(client, 1, "my balance?"))
    before = agent_of(client)
    client._drop(SID, client._calls.get(SID))  # noqa: SLF001 - as the failed write does
    assert spoken(run(hop(client, 1, results=BALANCE))) == "ok"
    assert agent_of(client) is not before
    assert [len(m.calls) for m in (m0, m1)] == [1, 2]


def test_a_new_session_forgets_the_move(run: Any) -> None:
    """Turn numbers start again in each ConversationRelay session: ``begin_session`` clears the move."""
    m0, m1 = EntryModel([Hang(), text_msg("from the first")]), EntryModel([tool_msg(BALANCE_TOOL)])
    client = make([m0, m1])
    run(hop(client, 1, "my balance?"))
    run(client.begin_session(SID, history="cached"))
    assert spoken(run(hop(client, 1, results=BALANCE))) == "from the first"


# ------------------------------------------------------------------------- the history filter


def reasoning_texts(call: Call) -> list[str]:
    return [b["reasoningContent"]["reasoningText"]["text"] for m in call.messages for b in m["content"]
            if "reasoningContent" in b]


def test_reasoning_reaches_only_the_entry_that_produced_it(run: Any) -> None:
    """On every call, not only the first after a switch: entry 1 never gets entry 0's reasoning
    (on its first call, or on its tool hop), and entry 0 keeps its own after the return."""
    m0 = EntryModel([text_msg("a", reasoning="r0a"), Hang(), text_msg("d", reasoning="r0d"), text_msg("e")])
    m1 = EntryModel([tool_msg(("level0___faq_lookup", {"q": "hours"}), reasoning="r1b"),
                     text_msg("c", reasoning="r1c")])
    client = make([m0, m1], gateway=default_gateway())
    run(turn(client, "one"))  # entry 0
    run(turn(client, "two"))  # entry 0 abandoned, then entry 1: a tool hop and its answer
    run(turn(client, "three"))  # back on entry 0
    run(turn(client, "four"))  # entry 0 again
    assert [reasoning_texts(c) for c in m0.calls] == [[], ["r0a"], ["r0a"], ["r0a", "r0d"]]
    assert [reasoning_texts(c) for c in m1.calls] == [[], ["r1b"]]
    stored = [b for m in agent_of(client).messages for b in m["content"] if "reasoningContent" in b]
    assert len(stored) == 4  # the stored history keeps every block


def test_reasoning_from_before_a_restart_is_left_out(run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """After a process restart (a rebuild from Memory) a stored message's producer is unknown,
    so its reasoning is left out, even on the entry that made it; new messages are recorded again."""
    light_memory_client(monkeypatch)
    data = FakeMemoryData()
    first = make([EntryModel([text_msg("a", reasoning="r1")]), EntryModel()], memory=data)
    run(turn(first, "one"))
    m0 = EntryModel([text_msg("b", reasoning="r2"), text_msg("c")])
    second = make([m0, EntryModel()], memory=data)  # another process: the agent is rebuilt from Memory
    run(turn(second, "two"))
    run(turn(second, "three"))
    assert [reasoning_texts(c) for c in m0.calls] == [[], ["r2"]]


# --------------------------------------------------------------- Memory: one user message


def stored_user_texts(data: FakeMemoryData) -> list[str]:
    return user_texts(data.stored_messages(SID))


def test_after_a_reissue_memory_holds_the_callers_message_once(run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    light_memory_client(monkeypatch)
    data = FakeMemoryData()
    client = make([EntryModel([Hang()]), EntryModel([text_msg("answer")])], memory=data)
    run(turn(client, "hello"))
    assert stored_user_texts(data) == ["hello"]
    assert [m["role"] for m in data.stored_messages(SID)] == ["user", "assistant"]
    reloaded = make([EntryModel([text_msg("next")]), EntryModel()], memory=data)
    run(turn(reloaded, "and then"))
    assert user_texts(agent_of(reloaded).messages) == ["hello", "and then"]


@pytest.mark.parametrize("retry", [False, True], ids=["new_words", "bridge_retry"])
def test_a_cancelled_call_whose_message_was_flushed_then_a_reissue(run: Any, monkeypatch: pytest.MonkeyPatch,
                                                                   retry: bool) -> None:
    """The dangling-message regression: a barge-in cancels a call after the caller's message was
    flushed; the next turn (new words, or the bridge's retry of the same words) fails over before
    any output. Memory and the history hold each caller message exactly once."""
    light_memory_client(monkeypatch)
    data = FakeMemoryData()
    m0 = EntryModel([Hang(), Hang()])
    m1 = EntryModel([text_msg("answer")])
    state = FailoverState(failover_settings(2, deadline_s=5.0))  # the barge-in comes first
    client = make([m0, m1], memory=data, state=state)

    async def barge_in() -> None:
        task = asyncio.ensure_future(turn(client, "first"))
        while not m0.calls:
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(barge_in())
    assert stored_user_texts(data) == ["first"]  # flushed, unanswered
    client.failover.update_settings(failover_settings(2))  # now the deadline fires first
    second = "first" if retry else "second"
    assert spoken(run(turn(client, second))) == "answer"
    want = ["first"] if retry else ["first", "second"]
    assert stored_user_texts(data) == want
    assert user_texts(agent_of(client).messages) == want
    assert len(m0.calls) == 2 and len(m1.calls) == 1
    assert user_texts(m1.calls[0].messages) == want


# ------------------------------------------------------------------- recording and logging


def test_every_call_is_recorded_with_its_outcome_and_ttft(run: Any) -> None:
    m0 = EntryModel([text_msg("a"), client_error("ThrottlingException")])
    m1 = EntryModel([text_msg("b")])
    client = make([m0, m1])
    run(turn(client, "one"))
    run(turn(client, "two"))
    s0, s1 = client.failover.stats(0), client.failover.stats(1)
    assert (s0.calls, s0.errors, s1.calls, s1.errors) == (2, 1, 1, 0)
    ttfts = [smp.ttft_s for smp in client.failover.health(0).window]
    assert ttfts[0] is not None and 0 <= ttfts[0] < 1 and ttfts[1] is None


def test_the_model_call_line_names_the_model_and_the_cause(run: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = make([EntryModel([Hang()]), EntryModel([text_msg("ok")])])
    run(turn(client, "my card 4111 1111"))
    calls = [ln for ln in lines(caplog.records, LOGGER) if ln.get("event") == "strands_model_call"]
    assert [(c["model_id"], c["outcome"], c.get("cause")) for c in calls] == [
        (IDS[0], "error", "deadline"), (IDS[1], "ok", None)]
    (reissue,) = [ln for ln in lines(caplog.records, LOGGER) if ln.get("event") == "model_failover"]
    assert (reissue["from_model"], reissue["to_model"], reissue["cause"]) == (IDS[0], IDS[1], "deadline")
    assert "4111" not in caplog.text and "my card" not in caplog.text


def test_the_policy_section_is_read_per_turn(run: Any) -> None:
    """New thresholds reach the decision state and each entry's deadline without a deploy."""
    class Section:
        deadline = 4.0

        def settings_for(self, ids: tuple[str, ...]) -> Settings:
            return failover_settings(len(ids), deadline_s=self.deadline)

    section = Section()
    m0 = EntryModel([text_msg("a"), text_msg("b")])
    client = make([m0, EntryModel()])
    client.use_failover_policy(lambda: section)
    run(turn(client, "one"))
    assert client.failover.settings.entry(0).deadline_s == 4.0
    assert agent_of(client).model.deadline_s == 4.0
    section.deadline = 2.5
    run(turn(client, "two"))
    assert agent_of(client).model.deadline_s == 2.5


def test_a_failing_policy_source_keeps_the_settings(run: Any) -> None:
    client = make([EntryModel([text_msg("a")]), EntryModel()])
    before = client.failover.settings

    def broken() -> Any:
        raise RuntimeError("policy table down")

    client.use_failover_policy(broken)
    assert spoken(run(turn(client))) == "a" and client.failover.settings is before


def test_each_entry_releases_text_its_own_way(run: Any) -> None:
    """Haiku 4.5 (off) releases whole messages; the stream entry streams. A switch mid-turn
    uses the new entry's release (nothing of the abandoned call was held by the translator)."""
    m0 = EntryModel([Hang()])
    m1 = EntryModel([tool_msg(("t1", "get_balance", {"account": "checking"}), text="Let me check.")])
    client = make([m0, m1])
    events = run(turn(client))
    assert spoken(events) == ""  # message mode: text in a tool-using message is suppressed
    assert any(type(e).__name__ == "TextSuppressed" for e in events)


def test_the_system_prompt_cache_point_follows_the_entry(run: Any) -> None:
    models = entries(2)
    models = (replace(models[0], prompt_caching=True), models[1])
    m0, m1 = EntryModel([Hang()]), EntryModel([text_msg("ok")])
    client = make([m0, m1], models=models)
    run(turn(client))
    assert any("cachePoint" in b for b in m0.calls[0].system or [])
    assert not any("cachePoint" in b for b in m1.calls[0].system or [])
