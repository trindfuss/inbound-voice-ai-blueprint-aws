"""Model failover: properties of the guarded model (per-call half).

* Over generated message histories and producer records, the request sent to entry
  ``e`` carries exactly the ``reasoningContent`` blocks recorded as produced by ``e`` and every other
  block of every message in order (and every message and role), and is the unfiltered request
  itself when nothing is left out; the stored history is never changed.
* Over generated streams, a call whose first output comes within the
  deadline is never abandoned and passes every event through in order; a call without output by the
  deadline raises ``ModelUnusable(cause=abandoned)`` once and nothing reaches Strands; an error before
  any output raises ``ModelUnusable`` with ``classify``'s cause (or, for a request error, the error
  itself); an error after output is raised unchanged after the events before it. Run on asyncio and
  uvloop. Times are far from the deadline (a fast event within a quarter of it, a slow one at four
  times it), so the property never depends on a timer's accuracy.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncGenerator, Callable
from typing import Any

import pytest
from botocore.exceptions import ClientError
from hypothesis import HealthCheck, given, settings
from hypothesis import event as note_branch
from hypothesis import strategies as st

from owlbank.failover import ModelUnusable, Outcome, classify
from owlbank.failover.classify import INVALID_MODEL_MESSAGE, REQUEST_ERROR_PREFIX
from tac_bridge.guarded_model import PRODUCERS_KEY, GuardedModel, is_output, reasoning_key, strip_reasoning


texts = st.text(max_size=20)
blocks = st.one_of(
    st.builds(lambda t: {"text": t}, texts),
    st.builds(lambda t, s: {"reasoningContent": {"reasoningText": {"text": t, "signature": s}}}, texts, texts),
    st.builds(lambda b: {"reasoningContent": {"redactedContent": b}}, st.binary(max_size=8)),
    st.builds(lambda i, n: {"toolUse": {"toolUseId": i, "name": n, "input": {}}}, texts, texts),
    st.builds(lambda i, t: {"toolResult": {"toolUseId": i, "content": [{"text": t}]}}, texts, texts),
    st.just({"cachePoint": {"type": "default"}}),
)
histories = st.lists(
    st.builds(lambda r, c: {"role": r, "content": c}, st.sampled_from(["user", "assistant"]),
              st.lists(blocks, max_size=6)),
    max_size=8)
ENTRIES = 3


@st.composite
def produced_histories(draw: st.DrawFn) -> tuple[list[dict[str, Any]], dict[str, int] | None]:
    """A history and a record of the producing entry of most of its keyed reasoning blocks
    (``None``: no record at all)."""
    history = draw(histories)
    producers: dict[str, int] = {}
    for message in history:
        for block in message["content"]:
            key = reasoning_key(block)
            if key is not None and key not in producers and (draw(st.booleans()) or draw(st.booleans())):
                producers[key] = draw(st.integers(0, ENTRIES - 1))
    return history, None if draw(st.integers(0, 9)) == 0 else producers

FP8 = settings(max_examples=300, deadline=None)


@FP8
@given(histories)
def test_the_filter_drops_reasoning_only_and_keeps_order(history: list[dict[str, Any]]) -> None:
    before = copy.deepcopy(history)
    filtered = strip_reasoning(history)  # type: ignore[arg-type]
    assert history == before
    assert [m["role"] for m in filtered] == [m["role"] for m in history]
    for got, original in zip(filtered, history, strict=True):
        assert all("reasoningContent" not in b for b in got["content"])
        assert got["content"] == [b for b in original["content"] if "reasoningContent" not in b]
    assert strip_reasoning(filtered) == filtered  # idempotent


class Recorder:
    """An inner model that records the request and answers at once."""

    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def stream(self, messages: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        self.requests.append(messages)
        yield {"contentBlockDelta": {"delta": {"text": "ok"}, "contentBlockIndex": 0}}


@FP8
@given(produced_histories(), st.integers(0, ENTRIES - 1))
def test_an_entry_gets_reasoning_only_from_messages_it_produced(
        case: tuple[list[dict[str, Any]], dict[str, int] | None], entry: int) -> None:
    history, producers = case
    inner = Recorder()
    model = GuardedModel(inner, entry=entry, deadline_s=5.0, stage="dev")  # type: ignore[arg-type]
    before = copy.deepcopy(history)
    state = None if producers is None else {PRODUCERS_KEY: producers}

    async def go() -> None:
        async for _ in model.stream(history, invocation_state=state):  # type: ignore[arg-type]
            pass

    asyncio.run(go())
    (request,) = inner.requests
    assert history == before
    assert [m["role"] for m in request] == [m["role"] for m in history]
    def own(block: dict[str, Any]) -> bool:
        key = reasoning_key(block)
        return producers is not None and key is not None and producers.get(key) == entry

    dropped = False
    for got, original in zip(request, history, strict=True):
        kept = [b for b in original["content"] if "reasoningContent" not in b or own(b)]
        assert got["content"] == kept
        assert all(own(b) for b in got["content"] if "reasoningContent" in b)
        dropped = dropped or len(kept) != len(original["content"])
    if not dropped:
        assert request is history  # nothing left out: the request Strands built (prompt cache intact)



DEADLINE = 0.08
FAST = DEADLINE / 4
SLOW = DEADLINE * 4

MSG_START = {"messageStart": {"role": "assistant"}}
REASONING = {"contentBlockDelta": {"delta": {"reasoningContent": {"text": ""}}, "contentBlockIndex": 0}}
TEXT = {"contentBlockDelta": {"delta": {"text": "Hi"}, "contentBlockIndex": 1}}
TOOL = {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t", "name": "n"}}, "contentBlockIndex": 1}}
STOP = {"messageStop": {"stopReason": "end_turn"}}


def _error(code: str, message: str = "m") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, "ConverseStream")


ERRORS: list[Callable[[], Exception]] = [
    lambda: _error("ThrottlingException"),
    lambda: _error("ModelErrorException"),
    lambda: _error("ModelTimeoutException"),
    lambda: _error("ValidationException", INVALID_MODEL_MESSAGE),
    lambda: _error("ValidationException", f"{REQUEST_ERROR_PREFIX} bad"),
    lambda: _error("ResourceNotFoundException"),
    lambda: RuntimeError("boom"),
]

# One step: (is the gap before it slow, what comes).
steps = st.lists(
    st.tuples(st.booleans(), st.sampled_from(["start", "reasoning", "text", "tool", "stop", "error"]),
              st.integers(0, len(ERRORS) - 1)),
    min_size=1, max_size=6)
EVENTS: dict[str, dict[str, Any]] = {"start": MSG_START, "reasoning": REASONING, "text": TEXT, "tool": TOOL, "stop": STOP}


class Scripted:
    def __init__(self, script: list[tuple[bool, str, int]]) -> None:
        self.script = script

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        for slow, kind, err in self.script:
            await asyncio.sleep(SLOW if slow else FAST / 4)
            if kind == "error":
                exc = ERRORS[err]()
                raise exc
            yield EVENTS[kind]


def expected(script: list[tuple[bool, str, int]]) -> tuple[str, Any, list[dict[str, Any]]]:
    """The model of the per-call half: ("events", None, events) | ("unusable", cause, []) |
    ("raised", code, events)."""
    elapsed = 0.0
    events: list[dict[str, Any]] = []
    started = False
    for slow, kind, err in script:
        elapsed += SLOW if slow else FAST / 4
        if not started and elapsed > DEADLINE / 2:  # comfortably past nothing yet: abandoned
            return "unusable", Outcome.ABANDONED, []
        if kind == "error":
            exc = ERRORS[err]()
            if started:
                return "raised", type(exc), events
            response = getattr(exc, "response", None)
            code = response["Error"]["Code"] if response else type(exc).__name__
            cause = classify(code, response["Error"]["Message"] if response else None)
            if cause == Outcome.REQUEST_ERROR:
                return "raised", type(exc), []
            return "unusable", cause, []
        event = EVENTS[kind]
        events.append(event)
        started = started or is_output(event)
    return "events", None, events


@pytest.fixture(params=["asyncio", "uvloop"])
def loop_factory(request: pytest.FixtureRequest) -> Callable[[], asyncio.AbstractEventLoop]:
    if request.param == "uvloop":
        factory: Callable[[], asyncio.AbstractEventLoop] = pytest.importorskip("uvloop").new_event_loop
        return factory
    return asyncio.new_event_loop


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(steps)
def test_a_call_with_output_is_never_reissued_and_one_without_is_once(
        loop_factory: Callable[[], asyncio.AbstractEventLoop], script: list[tuple[bool, str, int]]) -> None:
    # No event falls near the deadline: a slow gap is 4x the deadline, and fast gaps add up to at
    # most 6 x FAST/4 < DEADLINE/2.
    model = GuardedModel(Scripted(script), entry=1, deadline_s=DEADLINE, stage="dev")  # type: ignore[arg-type]
    seen: list[Any] = []

    async def go() -> None:
        async for event in model.stream([]):
            seen.append(event)

    kind, what, events = expected(script)
    note_branch(kind)
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        if kind == "events":
            runner.run(go())
            assert seen == events
        elif kind == "unusable":
            with pytest.raises(ModelUnusable) as info:
                runner.run(go())
            assert info.value.cause == what and info.value.entry == 1
            assert seen == []  # nothing reached Strands, so nothing was spoken or stored
        else:
            with pytest.raises(what) as raised:
                runner.run(go())
            assert not isinstance(raised.value, ModelUnusable)
            assert seen == events
