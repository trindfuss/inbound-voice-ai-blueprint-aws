"""The bridge suites through the real agent client.

The bridge's component and end-to-end tests script the agent at the protocol level (``TextDelta``,
``ToolCall``, ``TurnEnd``, an exception, or raw ConverseStream events of one model message). The
rigs' agent (``new_agent``, ``new_scripted_agent``) is :class:`StrandsDriven`: each step is turned
into the model stream that would have produced it (:func:`step_to_model_calls`) and played by a
:class:`ParityModel` under a **real** :class:`~tac_bridge.agent_client_strands.StrandsAgentClient`,
with history in AgentCore Memory on the in-memory :class:`~tests.strands_fakes.FakeMemoryData`
(shared by every rig on the same store, so two "processes" share one Memory). The bridge then sees
whatever the real client makes of it: held text, interrupts and resumes, deferred hand-offs, error
mapping, the session ownership protocol. A test marked ``protocol_fake`` gets the plain scripted
fake instead (``FakeAgent``, ``ScriptedAgent``): its steps go straight to the bridge, for bridge
behaviour on inputs the real client never produces (its own defences).

How a step becomes a model stream:

* one step is one model message (a scripted step never has a Gateway tool between two messages). ``TextDelta``\\ s are text blocks,
  ``ToolCall``\\ s tool-use blocks with the same ID, name and input, ``TextSuppressed(n)`` is ``n``
  characters of text in the tool-using message, ``TurnEnd`` the ``messageStop`` (``tool_use`` when
  the message uses a tool) and its usage the metadata;
* an exception in the step is raised by the model at that point; an exception as the whole step is
  raised by the model call;
* ``delay_s`` (read while streaming, as the fake does) pauses before each event's chunks;
* a step the model can't produce fails loudly (:class:`UnrealisticStep`): text spoken before a tool
  call in one message (a real client suppresses it), or a tool the client doesn't offer.

An ``await_caller`` result is never asked about (the client records it with no model call,
alone through ``record_results`` or with the caller's words in one ``turn``). The scripts have one
step per agent input, so the recorded result has a step too: it is recorded in ``calls`` and
skipped (a callable step still runs, for its checks of the envelope; its reply is never asked for;
:class:`PerInputScript`); any model call the client makes for it fails the test.

A result for a tool use the client never emitted (a test that writes ``pending_tool`` into call
state directly, as if an earlier session had run) first gets that earlier session played into the
agent's history (:meth:`StrandsDriven._prime`), so the result resumes a real interrupt. A result for
an ID the client did emit is never primed: if it is no longer pending, the client's error is real.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from tac_bridge.agent_client import TextDelta, TextSuppressed, ToolCall, ToolResult, TurnEnd
from tac_bridge.agent_client_strands import StrandsAgentClient, inline_tools
from tests.strands_fakes import Call, FakeMemoryData, FakeModel, default_gateway, memory_factory, settings, tool_msg

INLINE_NAMES = frozenset(t.tool_name for t in inline_tools())


#: The fake AgentCore Memory per store, reset before each test (root ``conftest``).
_MEMORIES: dict[int, tuple[Any, FakeMemoryData]] = {}
#: Which scripted agent the rigs build: ``strands`` (the real client) or ``plain`` (``protocol_fake``).
RIG: dict[str, str] = {"agent": "strands"}


def reset(*, plain: bool = False) -> None:
    """Before each test: no Memory shared with an earlier test; ``plain`` for ``protocol_fake``."""
    _MEMORIES.clear()
    RIG["agent"] = "plain" if plain else "strands"


def plain_fake() -> bool:
    """The test is marked ``protocol_fake``: the rigs build the plain scripted fake."""
    return RIG["agent"] == "plain"


def memory_for(store: Any) -> FakeMemoryData:
    """The one fake AgentCore Memory of a store (rigs sharing a store are processes of one stack)."""
    entry = _MEMORIES.get(id(store))
    if entry is None or entry[0] is not store:
        entry = (store, FakeMemoryData())
        _MEMORIES[id(store)] = entry
    return entry[1]


__all__ = ["PerInputScript", "StrandsDriven", "memory_for", "plain_fake", "reset", "step_to_model_calls"]


class UnrealisticStep(AssertionError):
    """A scripted step no model stream can produce through a real client (see the module docs)."""


class UnscriptedModelCall(AssertionError):
    """The Strands loop called the model more often than the step scripts."""


@dataclass(frozen=True)
class _Pause:
    seconds: Callable[[], float]


class ParityModel(FakeModel):
    """:class:`FakeModel` that also honours :class:`_Pause` markers, and fails on an unscripted call."""

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, *,
                     tool_choice: Any = None, system_prompt_content: Any = None, **kwargs: Any) -> Any:
        self.calls.append(Call(copy.deepcopy(list(messages)), copy.deepcopy(list(tool_specs or [])),
                               copy.deepcopy(system_prompt_content)))
        entry = self.script.pop(0) if self.script else UnscriptedModelCall("no scripted step left")
        if isinstance(entry, BaseException):
            raise entry
        for chunk in entry:
            if isinstance(chunk, _Pause):
                seconds = chunk.seconds()
                if seconds:
                    await asyncio.sleep(seconds)
                continue
            await asyncio.sleep(0)
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


def _usage_chunk(usage: Mapping[str, int] | None) -> dict[str, Any]:
    u = dict(usage or {})
    inp, out = int(u.get("input_tokens", 10)), int(u.get("output_tokens", 5))
    raw: dict[str, int] = {"inputTokens": inp, "outputTokens": out, "totalTokens": inp + out}
    if "cache_read_tokens" in u:
        raw["cacheReadInputTokens"] = int(u["cache_read_tokens"])
    if "cache_write_tokens" in u:
        raw["cacheWriteInputTokens"] = int(u["cache_write_tokens"])
    return {"metadata": {"usage": raw, "metrics": {"latencyMs": 1}}}


def _raw_step(step: Sequence[Any]) -> list[Any]:
    """Raw events (``h_msg``): already ConverseStream chunks, one message."""
    starts = sum(1 for ev in step if isinstance(ev, dict) and "messageStart" in ev)
    if starts != 1:
        raise UnrealisticStep(f"a raw step must be one model message, got {starts}")
    chunks: list[Any] = []
    for ev in step:
        if isinstance(ev, dict):
            start = ((ev.get("contentBlockStart") or {}).get("start") or {}).get("toolUse")
            if start is not None and start.get("name") not in INLINE_NAMES:
                raise UnrealisticStep(f"tool {start.get('name')!r} is not offered to the model")
        chunks.append(ev)
        if isinstance(ev, dict) and "messageStop" in ev:
            chunks.append(_usage_chunk(None))
    return chunks


def step_to_model_calls(step: Sequence[Any], delay: Callable[[], float]) -> list[Any]:
    """The model calls (each a list of chunks) that make the real client emit ``step``."""
    if any(isinstance(ev, dict) for ev in step):
        return [_raw_step(step)]
    chunks: list[Any] = [{"messageStart": {"role": "assistant"}}]
    text_open = False
    has_tool = False
    spoken = False
    ended = False

    def close_text() -> None:
        nonlocal text_open
        if text_open:
            chunks.append({"contentBlockStop": {}})
            text_open = False

    for ev in step:
        if ended and not isinstance(ev, BaseException):
            raise UnrealisticStep(f"event after TurnEnd: {ev!r}")
        chunks.append(_Pause(delay))
        if isinstance(ev, TextDelta | TextSuppressed):
            if has_tool:
                raise UnrealisticStep("text after a tool call in one message")
            if isinstance(ev, TextDelta):
                spoken = True
            if not text_open:
                chunks.append({"contentBlockStart": {"start": {}}})
                text_open = True
            text = ev.text if isinstance(ev, TextDelta) else "x" * ev.chars
            if text:
                chunks.append({"contentBlockDelta": {"delta": {"text": text}}})
        elif isinstance(ev, ToolCall):
            if spoken:
                raise UnrealisticStep("text spoken before a tool call in one message (a real client "
                                      "suppresses it)")
            if ev.name not in INLINE_NAMES:
                raise UnrealisticStep(f"tool {ev.name!r} is not offered to the model")
            close_text()
            has_tool = True
            chunks += [{"contentBlockStart": {"start": {"toolUse": {"toolUseId": ev.tool_use_id, "name": ev.name}}}},
                       {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(ev.input)}}}},
                       {"contentBlockStop": {}}]
        elif isinstance(ev, TurnEnd):
            close_text()
            stop = "tool_use" if has_tool else (ev.stop_reason or "end_turn")
            chunks += [{"messageStop": {"stopReason": stop}}, _usage_chunk(ev.usage)]
            ended = True
        elif isinstance(ev, BaseException):
            chunks.append(ev)
            return [chunks]
        else:
            raise UnrealisticStep(f"unknown scripted event {ev!r}")
    if not ended:
        close_text()
        chunks += [{"messageStop": {"stopReason": "tool_use" if has_tool else "end_turn"}}, _usage_chunk(None)]
    if not any(isinstance(c, dict) and "contentBlockStart" in c for c in chunks):
        # A message with no content block at all: Strands needs one, so give it an empty text block
        # (nothing is spoken).
        stop_at = next(i for i, c in enumerate(chunks) if isinstance(c, dict) and "messageStop" in c)
        chunks[stop_at:stop_at] = [{"contentBlockStart": {"start": {}}}, {"contentBlockStop": {}}]
    return [chunks]


def preferred_prompt(system_prompts: Mapping[str, str] | None) -> str | None:
    """The preferred entry's prompt of the bridge's ``{model key: prompt}`` map (in list
    order, so the first), as a one-model agent sees it; ``None`` for no map."""
    return next(iter(system_prompts.values()), None) if system_prompts else None


class PerInputScript:
    """One script step per agent input (see the module docs): a recorded ``await_caller`` result
    consumes its step without a model call. Mixed into every scripted agent fake.

    The class provides ``_record(...)`` (records the call, returns the record) and ``_next_step
    (rec)``."""

    def _skip_record_step(self, session_id: str, tool_results: Sequence[ToolResult],
                          system_prompts: Mapping[str, str] | None, system_suffix: str | None) -> None:
        rec = self._record(session_id, None, tool_results, system_prompts,  # type: ignore[attr-defined]
                           system_suffix)
        rec["recorded"] = True
        step = self._next_step(rec)  # type: ignore[attr-defined]
        if callable(step) and not isinstance(step, list):
            step(rec)  # its checks of what the agent was given still run; its reply is never asked for


class StrandsDriven(PerInputScript):
    """Mixin for a scripted agent fake (``FakeAgent``, ``ScriptedAgent``): the same interface,
    but every turn runs through a real :class:`StrandsAgentClient` (see the module docs).

    The base class provides ``_record(...)`` (records the call, returns the record), ``_next_step
    (rec)``, ``calls``, ``sessions``, ``histories`` and (optionally) ``delay_s`` and ``closed``."""

    def _attach(self, memory: FakeMemoryData) -> None:
        self.model = ParityModel()
        self.memory_data = memory
        self.client = StrandsAgentClient(
            # Message mode: the step-to-stream mapping above is message mode's
            # contract (text in a tool-using message is TextSuppressed); stream mode has its own tests.
            settings(memory_id="mem-1", text_release="message"), model_factory=lambda _s, _sid: self.model,
            gateway_tools=default_gateway(), session_manager_factory=memory_factory(memory))
        self.emitted: set[str] = set()

    async def begin_session(self, session_id: str, *, history: str = "reload") -> None:
        self.sessions.append(session_id)  # type: ignore[attr-defined]
        self.histories.append(history)  # type: ignore[attr-defined]
        await self.client.begin_session(session_id, history=history)  # type: ignore[arg-type]

    async def record_exchange(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """Through the real client (no model call, no script step consumed)."""
        exchanges = getattr(self, "exchanges", None)
        if exchanges is not None:
            exchanges.append((session_id, user_text, assistant_text))
        self.model.script[:] = []  # any model call here fails the test (UnscriptedModelCall)
        await self.client.record_exchange(session_id, user_text, assistant_text)

    def turn(self, session_id: str, *, user_text: str | None, tool_results: Sequence[ToolResult],
             system_prompts: Mapping[str, str] | None, system_suffix: str | None = None,
             turn_id: int | None = None) -> AsyncGenerator[Any, None]:
        if tool_results and user_text:
            # An await_caller result with the caller's words, in one turn. The script
            # has a step for each (the result, then the words); the model is never asked about
            # the result, so its step is recorded and skipped.
            self._skip_record_step(session_id, tool_results, system_prompts, system_suffix)
            rec = self._record(session_id, user_text, [], system_prompts,  # type: ignore[attr-defined]
                               system_suffix)
        else:
            rec = self._record(session_id, user_text, tool_results, system_prompts,  # type: ignore[attr-defined]
                               system_suffix)
        step = self._next_step(rec)  # type: ignore[attr-defined]
        return self._drive(step, rec, session_id, user_text, list(tool_results), system_prompts, system_suffix,
                           turn_id)

    def record_results(self, session_id: str, tool_results: Sequence[ToolResult], *,
                       system_prompts: Mapping[str, str] | None, system_suffix: str | None = None,
                       turn_id: int | None = None) -> AsyncGenerator[Any, None]:
        """The result hop's scripted step is skipped (no model call); any model call the
        client makes anyway fails the test (:class:`UnscriptedModelCall`)."""
        self._skip_record_step(session_id, tool_results, system_prompts, system_suffix)
        return self._drive_record(session_id, list(tool_results), system_prompts, system_suffix, turn_id)

    async def _drive_record(self, session_id: str, tool_results: list[ToolResult],
                            system_prompts: Mapping[str, str] | None, system_suffix: str | None,
                            turn_id: int | None = None) -> AsyncGenerator[Any, None]:
        try:
            await self._prime(session_id, tool_results)
            self.model.script[:] = []
            events = self.client.record_results(session_id, tool_results, system_prompts=system_prompts,
                                                system_suffix=system_suffix, turn_id=turn_id)
            async with aclosing(events):
                async for ev in events:
                    yield ev
        finally:
            if hasattr(self, "closed"):
                self.closed += 1  # type: ignore[has-type]

    def _delay(self) -> float:
        return float(getattr(self, "delay_s", 0.0) or 0.0)

    async def _drive(self, step: Any, rec: dict[str, Any], session_id: str, user_text: str | None,
                     tool_results: list[ToolResult], system_prompts: Mapping[str, str] | None,
                     system_suffix: str | None, turn_id: int | None = None) -> AsyncGenerator[Any, None]:
        try:
            if callable(step) and not isinstance(step, list):
                step = step(rec)
            await self._prime(session_id, tool_results)
            if isinstance(step, BaseException):
                self.model.script[:] = [step]
            else:
                self.model.script[:] = step_to_model_calls(step, self._delay)
            events = self.client.turn(session_id, user_text=user_text, tool_results=tool_results,
                                      system_prompts=system_prompts, system_suffix=system_suffix,
                                      turn_id=turn_id)
            async with aclosing(events):
                async for ev in events:
                    if isinstance(ev, ToolCall):
                        self.emitted.add(ev.tool_use_id)
                    yield ev
        finally:
            if hasattr(self, "closed"):
                self.closed += 1  # type: ignore[has-type]

    async def _prime(self, session_id: str, tool_results: Sequence[ToolResult]) -> None:
        """Play the earlier session a test assumed (a ``pending_tool`` written straight into call
        state): the model called the tool, the client emitted it. Only for IDs never emitted."""
        from tac_bridge.agent_client_strands import pending_interrupts

        built = await self.client._built(session_id, "cached")  # noqa: SLF001 - test seam
        pending = pending_interrupts(built.agent)
        missing = [r for r in tool_results if r.tool_use_id not in pending and r.tool_use_id not in self.emitted]
        if not missing:
            return
        uses = []
        for r in missing:
            name = r.content.get("tool") if isinstance(r.content, Mapping) else None
            uses.append((r.tool_use_id, name if name in INLINE_NAMES else "get_balance", {}))
        self.model.script[:] = [tool_msg(*uses)]
        async with aclosing(self.client.turn(session_id, user_text="(an earlier session)", tool_results=[],
                                             system_prompts=None)) as gen:
            async for ev in gen:
                if isinstance(ev, ToolCall):
                    self.emitted.add(ev.tool_use_id)

    def model_inputs_text(self) -> str:
        """Everything the real client sent the model (messages with history, system blocks)."""
        return json.dumps([{"messages": c.messages, "system": c.system} for c in self.model.calls], default=str)
