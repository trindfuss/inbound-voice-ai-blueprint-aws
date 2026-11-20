"""Model failover: sticky turns as a state machine.

Over generated model lists of one to three entries and generated sequences of steps against the real
Strands client (the real guarded models with a short first-token deadline, the fake AgentCore
Memory): hops of the current bridge turn (the caller's words, tool results for a pending inline tool
use, or record-only results), turn boundaries, the bridge's retry of a failed hop (the same inputs,
the same turn), fleet-verdict changes between hops (degraded, unavailable, recovered) and agent
rebuilds from Memory. Each model call answers, uses an inline tool, misses its deadline or throttles
before any output, as drawn.

Invariants:

* **No return:** within a turn, no model call starts on an entry the turn has left, unless
  every entry has been left (then the turn starts again from all of them).
* **A new turn chooses afresh:** the first model call of a new turn id is on the entry
  ``FailoverState.choose`` names just before it.
* **Every change of entry within a turn is a failover:** each move between consecutive model calls
  of a turn (within a hop, or from one hop to the next) is one ``model_failover`` line from and to
  those entries, and each line counts the ``Failover`` metric once.
* **Record-only hops change nothing:** no model call, no line, no metric, and the turn's move is
  as it was.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import replace
from typing import Any

import pytest
from hypothesis import HealthCheck, event, example, given, settings
from hypothesis import strategies as st

from owlbank.failover import EntryState, FailoverState, FleetView, Settings
from tac_bridge.agent_client import AgentError, ToolCall, ToolResult
from tac_bridge.agent_client_strands import ModelEntry, StrandsAgentClient, StrandsSettings
from tests.logutil import lines
from tests.strands_fakes import (
    Call,
    FakeMemoryData,
    FakeModel,
    client_error,
    light_memory_client,
    memory_factory,
    new_id,
    text_msg,
    tool_msg,
)

SID = "CA" + "7" * 32
LOGGER = "tac_bridge.agent_client_strands"
IDS = ("us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5", "us.anthropic.claude-sonnet-5-5")
DEADLINE_S = 0.02
PROMPTS = {"haiku-5-5": "BASE"}
STATES = {"healthy": EntryState.HEALTHY, "degraded": EntryState.DEGRADED, "unavailable": EntryState.UNAVAILABLE}


class StepModel(FakeModel):
    """Entry ``index``: plays its drawn behaviours in order, then answers; logs each call."""

    def __init__(self, index: int, behaviours: list[str], log: list[int]) -> None:
        super().__init__()
        self.index, self.behaviours, self.log = index, list(behaviours), log

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, *,
                     tool_choice: Any = None, system_prompt_content: Any = None, **kwargs: Any) -> Any:
        self.calls.append(Call(copy.deepcopy(list(messages)), [], None))
        self.log.append(self.index)
        b = self.behaviours.pop(0) if self.behaviours else "text"
        if b == "hang":
            await asyncio.sleep(30)
        if b == "hard":
            raise client_error("ThrottlingException")
        chunks = (tool_msg((new_id(), "get_balance", {"account": "checking"})) if b == "tool"
                  else text_msg(f"answer {self.index}"))
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk


class Events:
    def __init__(self) -> None:
        self.failovers = 0

    def failover(self) -> None:
        self.failovers += 1

    def unavailable(self) -> None:
        pass

    def active(self, rank: int) -> None:
        pass


class Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


STEP = st.one_of(
    st.just(("hop",)), st.just(("results",)), st.just(("record",)), st.just(("new_turn",)),
    st.just(("rebuild",)),
    st.tuples(st.just("fleet"), st.integers(0, 2), st.sampled_from(sorted(STATES))))


@st.composite
def cases(draw: st.DrawFn) -> dict[str, Any]:
    n = draw(st.integers(1, 3))
    scripts = []
    for i in range(n):
        allowed = ["text", "tool", "hard"] + ([] if i == n - 1 else ["hang"])  # the last has no deadline
        scripts.append(draw(st.lists(st.sampled_from(allowed), max_size=6)))
    return {"n": n, "scripts": scripts, "steps": draw(st.lists(STEP, min_size=1, max_size=14))}


def make_client(n: int, fakes: list[StepModel], data: FakeMemoryData) -> StrandsAgentClient:
    base = Settings.for_models(IDS[:n])
    state = FailoverState(replace(base, entries=tuple(
        replace(e, deadline_s=None if i == n - 1 else DEADLINE_S) for i, e in enumerate(base.entries))))
    shapes = [{"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000},
              {"thinking": "off", "text_release": "message", "max_tokens": 400},
              {"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000}]
    by_id = {IDS[i]: fakes[i] for i in range(n)}
    return StrandsAgentClient(
        StrandsSettings(models=tuple(ModelEntry(id=IDS[i], **shapes[i]) for i in range(n)),  # type: ignore[arg-type]
                        region="us-east-1", memory_id="mem-1"),
        model_factory=lambda one, _sid: by_id[one.primary.id], session_manager_factory=memory_factory(data),
        failover_state=state)


def move_of(client: StrandsAgentClient) -> Any:
    rec = client._sessions.get(SID)  # noqa: SLF001
    return None if rec is None else rec.move


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
# Two unavailable entries: the hop fails on 0 (a miss) then on 1 (a throttle) with
# nowhere left; the retry must not go back to 0.
@example(case={"n": 3, "scripts": [["hang", "text"], ["hard"], []],
               "steps": [("fleet", 2, "unavailable"), ("hop",), ("hop",)]})
# A live call: a miss, a tool use on the next entry, the hop after the tool result, a new turn.
@example(case={"n": 2, "scripts": [["hang", "text"], ["tool", "text"]],
               "steps": [("hop",), ("results",), ("new_turn",), ("hop",)]})
def test_3_3a_entry_choice_over_hops_turns_and_state_changes(case: dict[str, Any]) -> None:
    n = case["n"]
    calls: list[int] = []
    fakes = [StepModel(i, case["scripts"][i], calls) for i in range(n)]
    data = FakeMemoryData()
    handler = Records()
    logger = logging.getLogger(LOGGER)
    with pytest.MonkeyPatch.context() as mp:
        light_memory_client(mp)
        client = make_client(n, fakes, data)
        events = Events()
        client.use_failover_events(events)
        fleet = [EntryState.HEALTHY] * n
        client.use_fleet_view(lambda: FleetView(tuple(fleet)))
        logger.addHandler(handler)
        old_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            asyncio.run(_drive(case, client, fakes, calls, fleet, handler, events))
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)


async def _drive(case: dict[str, Any], client: StrandsAgentClient, fakes: list[StepModel], calls: list[int],
                 fleet: list[EntryState], handler: Records, events: Events) -> None:
    n = case["n"]
    turn_id = 1
    turn_calls: list[int] = []  # the model calls of the current turn, in order
    left: set[int] = set()  # the entries the current turn has left (since its last full reset)
    pending: list[str] = []  # tool uses waiting for a result
    retry: tuple[str, Any] | None = None  # a failed hop's inputs: the next hop repeats them
    fresh = True  # the next hop with a model call is the first of its turn
    await client.begin_session(SID, history="new")
    for step in case["steps"]:
        kind = step[0]
        if kind == "fleet":
            if step[1] < n:
                fleet[step[1]] = STATES[step[2]]
            continue
        if kind == "rebuild":
            client._drop(SID, client._calls.get(SID))  # noqa: SLF001 - as a failed history write does
            continue
        if kind == "new_turn" and retry is None:
            turn_id += 1
            turn_calls, left, fresh = [], set(), True
            continue
        if retry is not None:
            hop_kind, inputs = retry
        elif kind == "record" and pending:
            hop_kind, inputs = "record", [ToolResult(t, {"balance": "ok"}) for t in pending]
        elif kind == "results" and pending:
            hop_kind, inputs = "results", [ToolResult(t, {"balance": "ok"}) for t in pending]
        else:
            hop_kind, inputs = "words", f"words {len(calls)}"

        before_calls, before_lines, before_failovers = len(calls), len(handler.records), events.failovers
        before_move = move_of(client)
        expected_first = client.failover.choose(fleet=FleetView(tuple(fleet))) if fresh and n > 1 else None
        emitted: list[str] = []
        failed = False
        try:
            if hop_kind == "record":
                gen = client.record_results(SID, inputs, system_prompts=PROMPTS, turn_id=turn_id)
            elif hop_kind == "results":
                gen = client.turn(SID, user_text=None, tool_results=inputs, system_prompts=PROMPTS, turn_id=turn_id)
            else:
                gen = client.turn(SID, user_text=inputs, tool_results=[], system_prompts=PROMPTS, turn_id=turn_id)
            async for ev in gen:
                if isinstance(ev, ToolCall):
                    emitted.append(ev.tool_use_id)
        except AgentError:
            failed = True
        hop_calls = calls[before_calls:]
        hop_lines = lines(handler.records[before_lines:], LOGGER)

        if hop_kind == "record":
            event("a record-only hop")
            assert not failed
            assert hop_calls == []
            assert not [e for e in hop_lines if e["event"] == "model_failover"]
            assert events.failovers == before_failovers
            assert not [e for e in hop_lines if e["event"] == "model_turn_sticky"]
            assert move_of(client) == before_move
            pending = []
            continue

        # A new turn starts where choose says.
        if expected_first is not None and hop_calls:
            assert hop_calls[0] == expected_first, (expected_first, hop_calls)
        # No return to an entry the turn has left, unless it has left every entry; each change of
        # entry within the turn is one model_failover line (the verdict's start lines aside).
        changes: list[tuple[int, int]] = []
        prev = turn_calls[-1] if turn_calls else None
        for e in hop_calls:
            if prev is not None and e != prev:
                changes.append((prev, e))
                left.add(prev)
                if e in left:
                    assert len(left) == n, (turn_calls, hop_calls, left)
                    event("every entry left: the turn starts again")
                    left = set()
            prev = e
        turn_calls.extend(hop_calls)
        # (A verdict line, cause ``degraded``, only ever opens a turn: it is no change within one.)
        failovers = [e for e in hop_lines if e["event"] == "model_failover"]
        assert all(e["cause"] != "degraded" for e in failovers) or fresh, failovers
        within = [(e["from_entry"], e["to_entry"]) for e in failovers if e["cause"] != "degraded"]
        assert within == changes, (turn_calls, hop_calls, within)
        assert events.failovers - before_failovers == len(failovers)  # each line counted once
        if changes:
            event("a move within a turn")
        if changes and not fresh and changes[0][1] == hop_calls[0]:
            event("a later hop started on another entry than the turn's last")
        if hop_calls:
            fresh = False
        if failed:
            retry = (hop_kind, inputs)
            event("a failed hop, retried")
        else:
            retry = None
            pending = emitted
