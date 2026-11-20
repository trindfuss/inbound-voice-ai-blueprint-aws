"""The guarded model's first-token deadline, history filter and test delay.

(abandoned calls, the cancel signal). Every timed test
runs on asyncio and on uvloop (the production loop), and the deadline is checked against the wrapper's
own clock, never against a timer having fired (uvloop timers can fire early).
"""

from __future__ import annotations

import asyncio
import copy
import threading
import time
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any

import pytest
from botocore.exceptions import ClientError
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException

from owlbank.failover import ModelUnusable, Outcome
from owlbank.failover.classify import INVALID_MODEL_MESSAGE, REQUEST_ERROR_PREFIX, UNAVAILABLE
from tac_bridge.guarded_model import (
    FAILOVER_TEST_ENV,
    MAX_TEST_DELAY_S,
    PRODUCERS_KEY,
    GuardedModel,
    TestDelayError,
    failover_test_delay,
    foreign_reasoning_stripped,
    reasoning_key,
    record_reasoning,
    is_output,
    strip_reasoning,
)

# ---------------------------------------------------------------------------------------------- rig

MSG_START = {"messageStart": {"role": "assistant"}}
REASONING = {"contentBlockDelta": {"delta": {"reasoningContent": {"text": ""}}, "contentBlockIndex": 0}}
TEXT = {"contentBlockDelta": {"delta": {"text": "Hello"}, "contentBlockIndex": 1}}
TOOL_START = {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": "branch_hours"}},
                                    "contentBlockIndex": 1}}
TOOL_INPUT = {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}, "contentBlockIndex": 1}}
BLOCK_STOP = {"contentBlockStop": {"contentBlockIndex": 1}}
MSG_STOP = {"messageStop": {"stopReason": "end_turn"}}
METADATA = {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                         "metrics": {"latencyMs": 1}}}


def client_error(code: str, message: str = "caller said 4111") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, "ConverseStream")


class FakeModel:
    """A scripted stand-in for ``BedrockModel``: ``("sleep", s)``, ``("event", e)``, ``("raise", exc)``.

    It ignores the cancel signal, like a provider that can't abort, so the wrapper must stop on its own.
    """

    def __init__(self, script: list[tuple[str, Any]]) -> None:
        self.script = script
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.config = {"model_id": "fake-model"}

    def update_config(self, **cfg: Any) -> None:
        self.config.update(cfg)

    def get_config(self) -> Any:
        return self.config

    def structured_output(self, *a: Any, **k: Any) -> Any:  # pragma: no cover - not used
        raise NotImplementedError

    @property
    def stateful(self) -> bool:
        return False

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None,
                     **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        self.calls.append({"messages": messages, "tool_specs": tool_specs, "system_prompt": system_prompt,
                           **kwargs})
        try:
            for kind, value in self.script:
                if kind == "sleep":
                    await asyncio.sleep(value)
                elif kind == "event":
                    yield value
                else:
                    raise value
        finally:
            self.closed = True


def early_timers(monkeypatch: pytest.MonkeyPatch, loop: asyncio.AbstractEventLoop, early_s: float) -> None:
    """Every ``call_later`` fires ``early_s`` before its due time, as uvloop's millisecond timers can."""
    original = loop.call_later

    def call_later(delay: float, callback: Any, *args: Any, **kwargs: Any) -> asyncio.TimerHandle:
        return original(max(0.0, delay - early_s), callback, *args, **kwargs)

    monkeypatch.setattr(loop, "call_later", call_later)


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


async def collect(model: GuardedModel, messages: Any = None, **kwargs: Any) -> list[dict[str, Any]]:
    out: list[Any] = []
    async for event in model.stream(messages if messages is not None else [], **kwargs):
        out.append(event)
    return out


def guard(script: list[tuple[str, Any]], *, entry: int = 0, deadline_s: float | None = 0.2,
          **kwargs: Any) -> tuple[GuardedModel, FakeModel]:
    inner = FakeModel(script)
    kwargs.setdefault("stage", "dev")
    return GuardedModel(inner, entry=entry, deadline_s=deadline_s, **kwargs), inner  # type: ignore[arg-type]


FULL = [("event", MSG_START), ("event", TEXT), ("event", BLOCK_STOP), ("event", MSG_STOP),
        ("event", METADATA)]

# ------------------------------------------------------------------------------------- the deadline


def test_a_fast_call_passes_every_event_through_in_order(run: Any) -> None:
    model, inner = guard([("sleep", 0.01), *FULL])
    assert run(lambda: collect(model)) == [e for k, e in FULL if k == "event"]
    assert inner.closed


def test_no_first_token_within_the_deadline_raises_model_unusable(run: Any) -> None:
    model, inner = guard([("event", MSG_START), ("sleep", 5.0), *FULL], entry=1, deadline_s=0.15)
    seen: list[Any] = []

    async def go() -> None:
        async for event in model.stream([]):
            seen.append(event)

    started = time.monotonic()
    with pytest.raises(ModelUnusable) as info:
        run(go)
    elapsed = time.monotonic() - started
    assert info.value.is_deadline and info.value.cause == Outcome.ABANDONED and info.value.entry == 1
    assert 0.15 <= elapsed < 1.5  # at the deadline, not when the model would have answered
    assert seen == []  # the held messageStart never reached Strands
    signal = inner.calls[0]["cancel_signal"]
    assert isinstance(signal, threading.Event) and signal.is_set()  # the reader thread is told to stop
    assert inner.closed


def test_the_agents_own_signal_is_never_set_by_a_deadline(run: Any) -> None:
    model, inner = guard([("sleep", 5.0), *FULL], deadline_s=0.05)
    agent_signal = threading.Event()
    with pytest.raises(ModelUnusable):
        run(lambda: collect(model, cancel_signal=agent_signal))
    assert not agent_signal.is_set()  # setting it would end the call as "Cancelled by user"
    assert inner.calls[0]["cancel_signal"] is not agent_signal


def test_the_per_call_signal_follows_the_agents_signal() -> None:
    model, inner = guard(FULL)
    agent_signal = threading.Event()
    asyncio.run(collect(model, cancel_signal=agent_signal))
    per_call = inner.calls[0]["cancel_signal"]
    assert not per_call.is_set()
    agent_signal.set()
    assert per_call.is_set()  # a barge-in still stops the Bedrock request


def test_a_cancelled_turn_at_the_deadline_is_not_a_failover(run: Any) -> None:
    agent_signal = threading.Event()
    agent_signal.set()
    model, _ = guard([("sleep", 5.0), *FULL], deadline_s=0.05)
    assert run(lambda: collect(model, cancel_signal=agent_signal)) == []  # Strands reports "cancelled"


def test_a_cancelled_turn_with_an_error_is_not_a_failover(run: Any) -> None:
    agent_signal = threading.Event()
    agent_signal.set()
    model, _ = guard([("raise", ModelThrottledException("slow down"))])
    with pytest.raises(ModelThrottledException):
        run(lambda: collect(model, cancel_signal=agent_signal))


def test_the_deadline_is_kept_by_the_clock_when_timers_fire_early(run: Any,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """Every timer fires 30 ms early and the clock reads 20 ms behind: the call is abandoned only when
    the wrapper's clock has reached the deadline."""
    readings: list[float] = []

    def lagging_clock() -> float:
        now = time.monotonic() - 0.02
        readings.append(now)
        return now

    model, _ = guard([("sleep", 5.0), *FULL], deadline_s=0.2, clock=lagging_clock)

    async def go() -> None:
        loop = asyncio.get_running_loop()
        early_timers(monkeypatch, loop, 0.03)
        await collect(model)

    with pytest.raises(ModelUnusable):
        run(go)
    start = readings[0]
    assert readings[-1] - start >= 0.2  # the last reading, at the decision, is at or past the deadline


def test_an_answer_just_before_the_deadline_is_kept_when_timers_fire_early(
        run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Timers 30 ms early must not cut a call whose first token comes before the deadline."""
    model, _ = guard([("sleep", 0.12), *FULL], deadline_s=0.2)

    async def go() -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        early_timers(monkeypatch, loop, 0.03)
        return await collect(model)

    assert len(run(go)) == 5


def test_the_last_entry_has_no_deadline(run: Any) -> None:
    model, _ = guard([("sleep", 0.4), *FULL], entry=2, deadline_s=None)
    assert len(run(lambda: collect(model))) == 5


# --------------------------------------------------------------------------- errors before any output


@pytest.mark.parametrize(("exc", "cause", "code"), [
    # Strands' throttle wrapper keeps Bedrock's code, as without failover (``to_agent_error``).
    (ModelThrottledException("Too many requests"), Outcome.THROTTLED, "ThrottlingException"),
    (client_error("ThrottlingException"), Outcome.THROTTLED, "ThrottlingException"),
    (client_error("ModelErrorException"), Outcome.ERROR, "ModelErrorException"),
    (client_error("ServiceUnavailableException"), Outcome.ERROR, "ServiceUnavailableException"),
    (client_error("modelStreamErrorException"), Outcome.ERROR, "modelStreamErrorException"),
    (client_error("ModelTimeoutException"), Outcome.TIMEOUT, "ModelTimeoutException"),
    (client_error("ValidationException", INVALID_MODEL_MESSAGE), UNAVAILABLE, "ValidationException"),
    (client_error("ResourceNotFoundException"), UNAVAILABLE, "ResourceNotFoundException"),
    (client_error("AccessDeniedException"), UNAVAILABLE, "AccessDeniedException"),
    (RuntimeError("socket gone"), Outcome.ERROR, "RuntimeError"),
])
def test_a_pre_output_error_raises_model_unusable_with_its_cause(run: Any, exc: Exception, cause: Any,
                                                                 code: str) -> None:
    model, inner = guard([("event", MSG_START), ("raise", exc)], entry=0)
    with pytest.raises(ModelUnusable) as info:
        run(lambda: collect(model))
    assert info.value.cause == cause and info.value.entry == 0 and info.value.error_code == code
    assert not info.value.is_deadline
    assert "4111" not in str(info.value) and "socket" not in str(info.value)  # no message kept
    assert info.value.__cause__ is None and info.value.__context__ is None  # the message isn't chained
    assert inner.calls[0]["cancel_signal"].is_set()


@pytest.mark.parametrize("exc", [
    client_error("ValidationException", f"{REQUEST_ERROR_PREFIX} bad tool"),
    client_error("ValidationException", "messages: roles must alternate"),
    ContextWindowOverflowException("too long"),
])
def test_a_request_error_is_raised_unchanged(run: Any, exc: Exception) -> None:
    model, _ = guard([("raise", exc)])
    with pytest.raises(type(exc)) as info:
        run(lambda: collect(model))
    assert info.value is exc  # never a failover: the client logs model_request_error


# ---------------------------------------------------------------------------------- output started


def test_output_already_started_is_never_abandoned(run: Any) -> None:
    script = [("event", MSG_START), ("sleep", 0.02), ("event", TEXT), ("sleep", 0.4), ("event", TEXT),
              ("event", BLOCK_STOP), ("event", MSG_STOP), ("event", METADATA)]
    model, _ = guard(script, deadline_s=0.1)
    assert run(lambda: collect(model)) == [e for k, e in script if k == "event"]


def test_a_tool_use_start_counts_as_output(run: Any) -> None:
    script = [("event", MSG_START), ("sleep", 0.02), ("event", TOOL_START), ("sleep", 0.4),
              ("event", TOOL_INPUT), ("event", BLOCK_STOP), ("event", MSG_STOP), ("event", METADATA)]
    model, _ = guard(script, deadline_s=0.1)
    assert run(lambda: collect(model)) == [e for k, e in script if k == "event"]


def test_an_error_after_output_is_raised_unchanged(run: Any) -> None:
    exc = ModelThrottledException("mid-stream")
    model, _ = guard([("event", TEXT), ("raise", exc)], deadline_s=0.1)
    seen: list[Any] = []

    async def go() -> None:
        async for event in model.stream([]):
            seen.append(event)

    with pytest.raises(ModelThrottledException) as info:
        run(go)
    assert info.value is exc and seen == [TEXT]  # the existing after-output path handles it


def test_reasoning_alone_is_not_output(run: Any) -> None:
    model, _ = guard([("event", MSG_START), ("event", REASONING), ("sleep", 5.0), *FULL], deadline_s=0.1)
    with pytest.raises(ModelUnusable):
        run(lambda: collect(model))


def test_reasoning_then_text_in_time_keeps_every_event_in_order(run: Any) -> None:
    script = [("event", MSG_START), ("event", REASONING), ("sleep", 0.02), ("event", TEXT),
              ("event", MSG_STOP)]
    model, _ = guard(script, deadline_s=0.2)
    assert run(lambda: collect(model)) == [MSG_START, REASONING, TEXT, MSG_STOP]


def test_a_stream_that_ends_without_output_passes_through(run: Any) -> None:
    script = [("event", MSG_START), ("event", MSG_STOP), ("event", METADATA)]
    model, _ = guard(script)
    assert run(lambda: collect(model)) == [MSG_START, MSG_STOP, METADATA]


def test_a_cancelled_task_stops_the_call_and_is_not_a_failover(run: Any) -> None:
    model, inner = guard([("sleep", 5.0), *FULL], deadline_s=None)

    async def go() -> None:
        task = asyncio.ensure_future(collect(model))
        await asyncio.sleep(0.05)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        run(go)
    assert inner.calls[0]["cancel_signal"].is_set() and inner.closed


class StubbornModel(FakeModel):
    """A provider read that takes ``linger_s`` to wind down after it is cancelled."""

    def __init__(self, linger_s: float) -> None:
        super().__init__([])
        self.linger_s = linger_s

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None,
                     **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        self.calls.append(kwargs)
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(self.linger_s)
            raise
        yield TEXT  # pragma: no cover - never reached


def test_a_cancel_during_the_cleanup_after_a_missed_deadline_propagates(run: Any) -> None:
    """The turn is cancelled (barge-in, its timeout) while the abandoned read winds down: the
    cancellation propagates, and the missed deadline is not raised (the turn is not reissued)."""
    inner = StubbornModel(linger_s=0.5)
    model = GuardedModel(inner, entry=0, deadline_s=0.05, stage="dev")  # type: ignore[arg-type]
    outcome: list[str] = []

    async def go() -> None:
        task = asyncio.ensure_future(collect(model))
        while not inner.calls or not inner.calls[0]["cancel_signal"].is_set():
            await asyncio.sleep(0.01)  # the deadline has passed: the wrapper is cleaning up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            outcome.append("cancelled")
        except ModelUnusable:
            outcome.append("unusable")

    started = time.monotonic()
    run(go)
    assert outcome == ["cancelled"]
    assert time.monotonic() - started < 0.45  # did not wait for the read to wind down


def test_the_cleanup_after_a_missed_deadline_ignores_only_the_reads_outcome(run: Any) -> None:
    """Without a cancel, the read's own outcome (its ``CancelledError`` here) is ignored and the
    missed deadline is raised."""
    model = GuardedModel(StubbornModel(linger_s=0.02), entry=1, deadline_s=0.05,  # type: ignore[arg-type]
                         stage="dev")
    with pytest.raises(ModelUnusable) as info:
        run(lambda: collect(model))
    assert info.value.is_deadline and info.value.entry == 1


@pytest.mark.parametrize(("event", "expected"), [
    (TEXT, True), (TOOL_START, True), (TOOL_INPUT, True), (REASONING, False), (MSG_START, False),
    (MSG_STOP, False), (METADATA, False), (BLOCK_STOP, False),
    ({"contentBlockStart": {"start": {}, "contentBlockIndex": 0}}, False),
])
def test_what_counts_as_output(event: dict[str, Any], expected: bool) -> None:
    assert is_output(event) is expected


# ----------------------------------------------------------------------------------- history filter

HISTORY: Any = [
    {"role": "user", "content": [{"text": "What are your hours?"}]},
    {"role": "assistant", "content": [
        {"reasoningContent": {"reasoningText": {"text": "", "signature": "sig-a"}}},
        {"toolUse": {"toolUseId": "t1", "name": "branch_hours", "input": {}}}]},
    {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "9 to 5"}]}}]},
    {"role": "assistant", "content": [{"reasoningContent": {"redactedContent": b"x"}}]},
    {"role": "assistant", "content": [{"text": "We open at nine."}, {"cachePoint": {"type": "default"}}]},
]
SIG_A = reasoning_key(HISTORY[1]["content"][0])
RED_X = reasoning_key(HISTORY[3]["content"][0])


def test_the_filter_removes_every_reasoning_block_and_nothing_else() -> None:
    original = copy.deepcopy(HISTORY)
    filtered = strip_reasoning(HISTORY)
    assert original == HISTORY  # the stored history is untouched
    assert [m["role"] for m in filtered] == [m["role"] for m in HISTORY]
    assert filtered[1]["content"] == [HISTORY[1]["content"][1]]
    assert filtered[3]["content"] == []  # emptied, kept; BedrockModel drops an empty message itself
    assert filtered[0] == HISTORY[0] and filtered[2] == HISTORY[2] and filtered[4] == HISTORY[4]


def test_reasoning_reaches_only_the_entry_that_produced_it(run: Any) -> None:
    model, inner = guard(FULL, entry=1)
    original = copy.deepcopy(HISTORY)
    run(lambda: collect(model, HISTORY, invocation_state={PRODUCERS_KEY: {SIG_A: 1, RED_X: 1}}))
    run(lambda: collect(model, HISTORY, invocation_state={PRODUCERS_KEY: {SIG_A: 1, RED_X: 0}}))
    run(lambda: collect(model, HISTORY, invocation_state={PRODUCERS_KEY: {SIG_A: 0}}))
    run(lambda: collect(model, HISTORY, invocation_state={}))
    run(lambda: collect(model, HISTORY))
    own, mixed, other, *unknown = inner.calls
    assert own["messages"] is HISTORY  # all its own: the request as built (prompt cache intact)
    assert mixed["messages"][1] is HISTORY[1] and mixed["messages"][3]["content"] == []
    assert [m is h for m, h in zip(mixed["messages"], HISTORY, strict=True)] == [True, True, True, False, True]
    assert other["messages"] == strip_reasoning(HISTORY)
    for call in unknown:  # no record: every producer unknown, every reasoning block left out
        assert call["messages"] == strip_reasoning(HISTORY)
    assert original == HISTORY


def test_the_reasoning_key_is_a_digest_of_the_signature_or_redacted_content() -> None:
    signed = {"reasoningContent": {"reasoningText": {"text": "secret thoughts", "signature": "s1"}}}
    assert reasoning_key(signed) == reasoning_key({"reasoningContent": {"reasoningText": {"signature": "s1"}}})
    assert reasoning_key(signed) != reasoning_key({"reasoningContent": {"redactedContent": b"s1"}})
    assert "secret" not in (reasoning_key(signed) or "") and "s1" not in (reasoning_key(signed) or "")[4:]
    for other in ({"text": "hi"}, {"reasoningContent": {"reasoningText": {"text": "no signature"}}},
                  {"reasoningContent": {"redactedContent": b""}}, {"reasoningContent": "odd"}):
        assert reasoning_key(other) is None


def test_an_unkeyed_reasoning_block_is_always_left_out() -> None:
    history: Any = [{"role": "assistant", "content": [{"reasoningContent": {"reasoningText": {"text": "t"}}},
                                                      {"text": "hi"}]}]
    producers: dict[str, int] = {}
    record_reasoning(producers, history[0], 0)
    assert producers == {}
    assert foreign_reasoning_stripped(history, producers, 0)[0]["content"] == [{"text": "hi"}]


def test_recording_names_the_entry_of_every_keyed_block() -> None:
    producers: dict[str, int] = {}
    record_reasoning(producers, HISTORY[1], 2)
    record_reasoning(producers, HISTORY[3], 1)
    record_reasoning(producers, {"role": "assistant"}, 0)  # no content: nothing
    assert producers == {SIG_A: 2, RED_X: 1}


def test_other_arguments_reach_the_inner_model_unchanged(run: Any) -> None:
    model, inner = guard(FULL)
    state = {PRODUCERS_KEY: {}, "x": 1}
    specs = [{"name": "branch_hours"}]
    system = [{"text": "You are a bank."}]
    run(lambda: collect(model, HISTORY, tool_specs=specs, system_prompt="s", system_prompt_content=system,
                        tool_choice={"auto": {}}, invocation_state=state, dynamic_trailing_blocks=1))
    call = inner.calls[0]
    assert call["tool_specs"] is specs and call["system_prompt"] == "s"
    assert call["system_prompt_content"] is system and call["tool_choice"] == {"auto": {}}
    assert call["invocation_state"] is state and call["dynamic_trailing_blocks"] == 1


# -------------------------------------------------------------------------------------- test delay


def test_the_test_delay_holds_the_preferred_entrys_first_token(run: Any) -> None:
    model, _ = guard([("sleep", 0.01), *FULL], entry=0, deadline_s=2.0, test_delay_s=0.25)

    async def go() -> float:
        started = time.monotonic()
        async for event in model.stream([]):
            if is_output(event):
                return time.monotonic() - started
        raise AssertionError("no output")

    assert run(go) >= 0.25


def test_a_test_delay_past_the_deadline_fails_over(run: Any) -> None:
    model, _ = guard([("sleep", 0.01), *FULL], entry=0, deadline_s=0.1, test_delay_s=0.5)
    started = time.monotonic()
    with pytest.raises(ModelUnusable) as info:
        run(lambda: collect(model))
    assert info.value.is_deadline and time.monotonic() - started < 0.45


def test_the_test_delay_applies_to_the_preferred_entry_only(run: Any) -> None:
    model, _ = guard([("sleep", 0.01), *FULL], entry=1, deadline_s=0.1, test_delay_s=0.5)
    assert len(run(lambda: collect(model))) == 5


def test_the_test_delay_is_refused_at_prod() -> None:
    with pytest.raises(TestDelayError):
        guard(FULL, test_delay_s=1.0, stage="prod")
    with pytest.raises(TestDelayError):
        failover_test_delay({FAILOVER_TEST_ENV: "2"}, stage="prod")
    assert failover_test_delay({}, stage="prod") is None
    guard(FULL, test_delay_s=None, stage="prod")  # no delay: fine at prod


@pytest.mark.parametrize(("value", "expected"), [
    (None, None), ("", None), ("  ", None), ("2", 2.0), ("0.5", 0.5), (str(MAX_TEST_DELAY_S), 30.0)])
def test_the_test_delay_setting(value: str | None, expected: float | None) -> None:
    env = {} if value is None else {FAILOVER_TEST_ENV: value}
    assert failover_test_delay(env, stage="dev") == expected


@pytest.mark.parametrize("value", ["abc", "0", "-1", "30.5", "nan", "inf", "1e9"])
def test_a_bad_test_delay_is_refused_naming_the_setting(value: str) -> None:
    with pytest.raises(TestDelayError) as info:
        failover_test_delay({FAILOVER_TEST_ENV: value}, stage="dev")
    assert str(info.value).startswith(f"{FAILOVER_TEST_ENV}: must be")  # the setting, not the value


# -------------------------------------------------------------------------------------- delegation


def test_the_wrapper_delegates_config_to_its_model() -> None:
    model, inner = guard(FULL)
    assert model.get_config() is inner.config and model.config is inner.config
    model.update_config(max_tokens=10)
    assert inner.config["max_tokens"] == 10 and model.stateful is False
    assert model.inner is inner and model.entry == 0 and model.deadline_s == 0.2


def test_bad_construction_is_refused() -> None:
    with pytest.raises(ValueError):
        guard(FULL, entry=-1)
    with pytest.raises(ValueError):
        guard(FULL, deadline_s=0)
    with pytest.raises(ValueError):
        guard(FULL, deadline_s=float("nan"))


# ------------------------------------------------------------------------------- the Strands seam


def test_in_a_strands_agent_a_missed_deadline_continues_on_the_next_entry(run: Any) -> None:
    """``ModelUnusable`` leaves Strands unwrapped with only the caller's message in history, and
    ``stream_async(None)`` on the next entry continues from it: one user message, one answer."""
    from strands import Agent
    from strands.models.model import Model

    class Scripted(FakeModel, Model):  # type: ignore[misc]  # a real ``Model`` for the agent
        pass

    first = GuardedModel(Scripted([("sleep", 5.0), *FULL]), entry=0, deadline_s=0.1, stage="dev")
    second = GuardedModel(Scripted(FULL), entry=1, deadline_s=None, stage="dev")

    async def go() -> list[Any]:
        agent = Agent(model=first, callback_handler=None)
        with pytest.raises(ModelUnusable):
            await agent.invoke_async("What are your hours?")
        assert [m["role"] for m in agent.messages] == ["user"]
        agent.model = second
        async for _ in agent.stream_async(None):
            pass
        return [(m["role"], m["content"]) for m in agent.messages]

    assert run(go) == [("user", [{"text": "What are your hours?"}]), ("assistant", [{"text": "Hello"}])]
