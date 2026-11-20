"""Model failover end to end: over generated model lists,
local entry states and per-call behaviours, one caller turn through the real Strands client, the
real guarded models and the fake AgentCore Memory:

* an entry is tried at most once per turn (the calls on one entry are one contiguous run, and no
  entry has two runs), in rank: the healthy entries in list order, then the degraded ones (an
  earlier degraded entry included), never skipping a usable one;
* a call that produced output is never reissued: a move to another entry follows only a call that
  failed before any output (a missed deadline or a pre-output hard error);
* an entry marked unavailable gets no call; a request error is never a failover;
* nothing of an abandoned call is spoken, and the history and Memory hold the caller's message
  exactly once.

Each example runs on asyncio or uvloop (drawn); nothing waits for a timer to be on time.

Over generated per-entry behaviours across several turns (so the entry of each
call is whatever the failover decisions made it): the request sent to entry ``e`` carries exactly
the reasoning blocks of the history it was built from that ``e`` produced, in order: none from
another entry, on any call (a later tool hop or turn too, not only the first after a switch), and
none of its own lost.
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace
from typing import Any

import pytest
from collections.abc import Callable

import uvloop
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from owlbank.failover import UNAVAILABLE, FailoverState, Outcome, Settings
from owlbank.failover.classify import REQUEST_ERROR_PREFIX
from tac_bridge.agent_client import AgentError, TextDelta
from tac_bridge.agent_client_strands import ModelEntry, StrandsAgentClient, StrandsSettings
from tests.strands_fakes import (
    Call,
    FakeMemoryData,
    FakeModel,
    client_error,
    default_gateway,
    light_memory_client,
    memory_factory,
    text_msg,
    tool_msg,
)

SID = "CA" + "9" * 32
IDS = ("us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5", "us.anthropic.claude-sonnet-5-5")
DEADLINE_S = 0.02
CALLER = "please help me"

#: What one model call does. ``hang``: nothing before the deadline; ``hard``: a pre-output throttle;
#: ``request``: a pre-output request error; ``text``: an answer; ``tool``: a Gateway tool use (the
#: loop calls the model again); ``late``: text, then an error after the output started.
NO_OUTPUT = frozenset({"hang", "hard"})
BEHAVIOURS = ("hang", "hard", "request", "text", "tool", "late")


class LoggedModel(FakeModel):
    """Entry ``index``'s fake: plays ``behaviours`` in order and logs ``(index, behaviour)``."""

    def __init__(self, index: int, behaviours: list[str], log: list[tuple[int, str]]) -> None:
        super().__init__()
        self.index, self.behaviours, self.log = index, list(behaviours), log

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, *,
                     tool_choice: Any = None, system_prompt_content: Any = None, **kwargs: Any) -> Any:
        self.calls.append(Call(copy.deepcopy(list(messages)), [], None))
        b = self.behaviours.pop(0) if self.behaviours else "text"
        self.log.append((self.index, b))
        n = len(self.log)
        if b == "hang":
            await asyncio.sleep(30)
        if b == "hard":
            raise client_error("ThrottlingException")
        if b == "request":
            raise client_error("ValidationException", f"{REQUEST_ERROR_PREFIX} a bad request")
        if b == "tool":
            chunks = tool_msg(("level0___faq_lookup", {"q": "hours"}))
        elif b == "late":
            chunks = [{"messageStart": {"role": "assistant"}}, {"contentBlockStart": {"start": {}}},
                      {"contentBlockDelta": {"delta": {"text": f"[late {self.index}.{n}]"}}},
                      client_error("ServiceUnavailableException")]
        else:
            chunks = text_msg(f"[answer {self.index}.{n}]")
        for chunk in chunks:
            await asyncio.sleep(0)
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


@st.composite
def cases(draw: st.DrawFn) -> dict[str, Any]:
    n = draw(st.integers(2, 3))
    scripts = []
    for i in range(n):
        allowed = [b for b in BEHAVIOURS if not (i == n - 1 and b == "hang")]  # the last has no deadline
        scripts.append(draw(st.lists(st.sampled_from(allowed), min_size=1, max_size=4)))
    marks = draw(st.lists(st.sampled_from(["healthy", "degraded", "unavailable"]), min_size=n, max_size=n))
    return {"n": n, "scripts": scripts, "marks": marks, "uvloop": draw(st.booleans())}


def failover_state(n: int, marks: list[str]) -> FailoverState:
    base = Settings.for_models(IDS[:n])
    s = replace(base, entries=tuple(replace(e, deadline_s=None if i == n - 1 else DEADLINE_S)
                                    for i, e in enumerate(base.entries)))
    state = FailoverState(s)
    for i, mark in enumerate(marks):
        if mark == "unavailable":
            state.record(i, UNAVAILABLE)
        elif mark == "degraded":
            for _ in range(s.miss_run):
                state.record(i, Outcome.ABANDONED)
    return state


def user_texts(messages: list[dict[str, Any]]) -> list[str]:
    return [b["text"] for m in messages if m["role"] == "user" for b in m["content"] if "text" in b]


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_end_to_end(case: dict[str, Any]) -> None:
    n, scripts, marks = case["n"], case["scripts"], case["marks"]
    log: list[tuple[int, str]] = []
    fakes = [LoggedModel(i, scripts[i], log) for i in range(n)]
    shapes = [{"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000},
              {"thinking": "off", "text_release": "message", "max_tokens": 400},
              {"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000}]
    data = FakeMemoryData()
    state = failover_state(n, marks)
    unavailable = {i for i in range(n) if marks[i] == "unavailable"}
    with pytest.MonkeyPatch.context() as mp:
        light_memory_client(mp)
        client = StrandsAgentClient(
            StrandsSettings(models=tuple(ModelEntry(id=IDS[i], **shapes[i]) for i in range(n)),  # type: ignore[arg-type]
                            region="us-east-1", memory_id="mem-1"),
            model_factory=lambda one, _sid: fakes[IDS.index(one.primary.id)], gateway_tools=default_gateway(),
            session_manager_factory=memory_factory(data), failover_state=state)

        async def go() -> tuple[list[Any], BaseException | None]:
            out: list[Any] = []
            try:
                async for ev in client.turn(SID, user_text=CALLER, tool_results=[], system_prompts={"haiku-5-5": "BASE"}):
                    out.append(ev)
            except AgentError as exc:
                return out, exc
            return out, None

        loop_factory = uvloop.new_event_loop if case["uvloop"] else asyncio.new_event_loop
        with asyncio.Runner(loop_factory=loop_factory) as runner:
            events, error = runner.run(go())
            agent = client._calls[SID].build.result().agent
            stored = data.stored_messages(SID)

    entries_called = [i for i, _ in log]
    event(f"entries tried: {len(set(entries_called))}; answered: {error is None}")
    # Each entry tried at most once: its calls are one run, and no entry has two.
    runs = [e for k, e in enumerate(entries_called) if k == 0 or entries_called[k - 1] != e]
    assert len(runs) == len(set(runs)), log
    # In rank: the healthy entries in list order, then the degraded ones (their p90s tie, so in
    # list order); all unavailable: the last alone. A turn that ran out tried every usable entry.
    rank = [i for i in range(n) if marks[i] == "healthy"] + [i for i in range(n) if marks[i] == "degraded"]
    order = rank if rank else [n - 1]
    assert runs == order[:len(runs)], (marks, log)
    # The turn's model-call cap (max_iterations, failover calls included) can end it first; the
    # in-rank prefix check above still holds for such a turn.
    if isinstance(error, AgentError) and error.code == "max_iterations_exceeded":
        assert len(log) == client.settings.max_iterations, log
    elif error is not None and log[-1][1] in NO_OUTPUT:
        assert runs == order, (marks, log)
    # A move to another entry follows only a call that produced no output.
    for (a, ba), (b, _) in zip(log, log[1:], strict=False):
        if a != b:
            assert ba in NO_OUTPUT, log
    # No call to an entry marked unavailable, unless every entry is (then the last, as today).
    if len(unavailable) < n:
        assert not unavailable & set(entries_called), (marks, log)
    # A request error or output followed by an error ends the turn on that call.
    for k, (_, b) in enumerate(log):
        if b in ("request", "late"):
            assert k == len(log) - 1 and error is not None, log
    # Spoken: only what calls with output said; never an abandoned call's words.
    said = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "too late" not in said
    for k, (i, b) in enumerate(log):
        if b in NO_OUTPUT:
            assert f"{i}.{k + 1}]" not in said
    if error is None:
        assert log[-1][1] == "text", log
    # The caller's message once, in the history and in Memory.
    assert user_texts(agent.messages) == [CALLER]
    assert user_texts(stored) == [CALLER], json.dumps(stored)[:400]


# ------------------------------------------------------------------------------------- Reasoning


def reasoning_texts(messages: list[dict[str, Any]]) -> list[str]:
    return [b["reasoningContent"]["reasoningText"]["text"] for m in messages for b in m["content"]
            if "reasoningContent" in b]


class ReasoningModel(LoggedModel):
    """:class:`LoggedModel` whose answers and tool uses carry reasoning ``r<entry>.<call>`` (signed
    per text), and which keeps, per call, the request's reasoning and the history it was built from."""

    def __init__(self, index: int, behaviours: list[str], log: list[tuple[int, str]],
                 history: Callable[[], list[dict[str, Any]]]) -> None:
        super().__init__(index, behaviours, log)
        self.history = history
        self.seen: list[tuple[list[str], list[str]]] = []

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, *,
                     tool_choice: Any = None, system_prompt_content: Any = None, **kwargs: Any) -> Any:
        self.seen.append((reasoning_texts(list(messages)), reasoning_texts(copy.deepcopy(self.history()))))
        b = self.behaviours[0] if self.behaviours else "text"
        if b not in ("text", "tool"):
            async for chunk in super().stream(messages, tool_specs, system_prompt, tool_choice=tool_choice,
                                              system_prompt_content=system_prompt_content, **kwargs):
                yield chunk
            return
        if self.behaviours:
            self.behaviours.pop(0)
        self.log.append((self.index, b))
        mark = f"r{self.index}.{len(self.log)}"
        chunks = (tool_msg(("level0___faq_lookup", {"q": "hours"}), reasoning=mark) if b == "tool"
                  else text_msg(f"[answer {self.index}.{len(self.log)}]", reasoning=mark))
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk


@st.composite
def reasoning_cases(draw: st.DrawFn) -> dict[str, Any]:
    n = draw(st.integers(2, 3))
    scripts = []
    for i in range(n):
        allowed = [b for b in ("hang", "hard", "text", "tool") if not (i == n - 1 and b == "hang")]
        scripts.append(draw(st.lists(st.sampled_from(allowed), min_size=1, max_size=8)))
    return {"n": n, "scripts": scripts, "turns": draw(st.integers(1, 4)), "uvloop": draw(st.booleans())}


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=reasoning_cases())
def test_an_entry_is_sent_only_the_reasoning_it_produced(case: dict[str, Any]) -> None:
    n, scripts = case["n"], case["scripts"]
    log: list[tuple[int, str]] = []
    holder: dict[str, Any] = {}

    def history() -> list[dict[str, Any]]:
        return list(holder["client"]._calls[SID].build.result().agent.messages)

    fakes = [ReasoningModel(i, scripts[i], log, history) for i in range(n)]
    shapes = [{"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000},
              {"thinking": "adaptive", "text_release": "message", "max_tokens": 400},
              {"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000}]
    client = StrandsAgentClient(
        StrandsSettings(models=tuple(ModelEntry(id=IDS[i], **shapes[i]) for i in range(n)),  # type: ignore[arg-type]
                        region="us-east-1"),
        model_factory=lambda one, _sid: fakes[IDS.index(one.primary.id)], gateway_tools=default_gateway(),
        failover_state=failover_state(n, ["healthy"] * n))
    holder["client"] = client

    async def go() -> None:
        for k in range(case["turns"]):
            try:
                async for _ in client.turn(SID, user_text=f"{CALLER} {k}", tool_results=[], system_prompts={"haiku-5-5": "BASE"}):
                    pass
            except AgentError:
                pass  # every entry unusable this turn: the next turn goes on from the history

    loop_factory = uvloop.new_event_loop if case["uvloop"] else asyncio.new_event_loop
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(go())

    switched = any(a != b for (a, _), (b, _) in zip(log, log[1:], strict=False))
    event(f"entries called: {len({i for i, _ in log})}; switched: {switched}")
    for fake in fakes:
        own = f"r{fake.index}."
        for sent, built_from in fake.seen:
            assert all(t.startswith(own) for t in sent), (fake.index, sent, log)  # nothing foreign
            assert sent == [t for t in built_from if t.startswith(own)], (fake.index, sent, built_from, log)
