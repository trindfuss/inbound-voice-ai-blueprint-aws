"""Model failover 3.3a through the bridge: the rest of a turn stays on the entry it switched to.

The live call (``failoverTest=6``): turn 1's first model call on Haiku 5.5 missed its first-token
deadline and moved to Haiku 4.5, which called ``list_recent_transactions``; the bridge ran the tool
and called the agent again with the result (hop 2), and that call went back to Haiku 5.5, missed the
deadline again and failed over again. Here the component rig (the simulator, the real bridge, a real
:class:`StrandsAgentClient` over one fake model per entry with the real guarded models and a 50 ms
first-token deadline) replays that sequence:

* once a turn has moved to another entry, the turn's later hops start on that entry (the preferred
  entry gets one call that turn), and ``model_failover`` is logged once, ``model_turn_sticky`` for
  the later hop;
* the next caller turn chooses afresh: the preferred entry again, as it is not degraded;
* a turn whose switched-to entry also fails on a later hop moves on and stays there, never
  going back to an entry it abandoned;
* a turn with no switch is unchanged.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

from owlbank.failover import FailoverState, Settings
from tac_bridge.agent_client import AgentError
from tac_bridge.agent_client_strands import ModelEntry, StrandsAgentClient, StrandsSettings
from tac_bridge.bridge import BridgeConfig
from tests.component.test_bridge_fakes import FakeBackend, Rig, make_rig, spoken, write_call_state
from tests.logutil import lines
from tests.strands_fakes import Call, FakeModel, text_msg, tool_msg

IDS = ("us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5", "us.anthropic.claude-sonnet-5-5")
DEADLINE_S = 0.05
CLIENT_LOG = "tac_bridge.agent_client_strands"
SHAPES = {IDS[0]: {"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000},
          IDS[1]: {"thinking": "off", "text_release": "message", "max_tokens": 400},
          IDS[2]: {"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000}}
TXNS = {"tool": "list_recent_transactions",
        "data": {"accountType": "checking", "last4": "4821",
                 "transactions": [{"date": "2026-10-09", "description": "Coffee Example Co", "amount": "-4.50"}]}}


class StepModel(FakeModel):
    """An entry's fake: plays ``steps`` in order, then answers. ``hang``: no first token before any
    deadline (the guard abandons the call); ``tool``: an inline ``list_recent_transactions`` call;
    ``text``: an answer naming the entry."""

    def __init__(self, index: int, steps: Sequence[str] = ()) -> None:
        super().__init__()
        self.index = index
        self.steps = list(steps)

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, *,
                     tool_choice: Any = None, system_prompt_content: Any = None, **kwargs: Any) -> Any:
        self.calls.append(Call(copy.deepcopy(list(messages)), [], None))
        step = self.steps.pop(0) if self.steps else "text"
        if step == "hang":
            await asyncio.sleep(30)
        chunks = (tool_msg(("list_recent_transactions", {"account": "checking"})) if step == "tool"
                  else text_msg(f"Answer from entry {self.index}."))
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk


def rig_for(fakes: list[StepModel]) -> Rig:
    ids = IDS[:len(fakes)]
    base = Settings.for_models(list(ids))
    state = FailoverState(replace(base, entries=tuple(
        replace(e, deadline_s=None if i == len(ids) - 1 else DEADLINE_S) for i, e in enumerate(base.entries))))
    rig = make_rig(config=BridgeConfig(model_ids=ids),
                   backend=FakeBackend({"list_recent_transactions": [TXNS] * 4}))
    by_id = dict(zip(ids, fakes, strict=True))
    rig.bridge._agent = StrandsAgentClient(  # noqa: SLF001 - the real client, over the fake models
        StrandsSettings(models=tuple(ModelEntry(id=m, **SHAPES[m]) for m in ids),  # type: ignore[arg-type]
                        region="us-east-1"),
        model_factory=lambda one, _sid: by_id[one.primary.id], failover_state=state)
    write_call_state(rig.store)
    return rig


def converse(rig: Rig, prompts: Sequence[str], fakes: list[StepModel]) -> tuple[list[str], list[list[int]]]:
    """What the caller heard per turn, and each entry's model calls after each turn."""
    async def go() -> tuple[list[str], list[list[int]]]:
        sim = rig.sim()
        await sim.connect()
        said, counts = [], []
        for p in prompts:
            said.append(spoken(await sim.prompt(p, timeout=5.0)))
            counts.append([len(f.calls) for f in fakes])
        await sim.hangup()
        return said, counts

    return asyncio.run(go())


def client_events(caplog: pytest.LogCaptureFixture, name: str) -> list[dict[str, Any]]:
    return [e for e in lines(caplog.records, CLIENT_LOG) if e["event"] == name]


def test_the_sequence_the_tool_hop_stays_on_the_switched_entry(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=CLIENT_LOG)
    fakes = [StepModel(0, ["hang", "text"]), StepModel(1, ["tool", "text"])]
    rig = rig_for(fakes)
    said, counts = converse(rig, ["What did I spend?", "Thanks, and my savings?"], fakes)
    assert "Answer from entry 1." in said[0]
    # Turn 1: the preferred entry once (the miss), then Haiku 4.5 for the tool call and for the hop
    # after the tool result: no second trip to the preferred entry.
    assert counts[0] == [1, 2]
    last = fakes[1].calls[1].messages[-1]
    assert last["role"] == "user" and "toolResult" in last["content"][0]
    # Turn 2 chooses afresh: the preferred entry is not degraded (one miss), so it answers.
    assert "Answer from entry 0." in said[1]
    assert counts[1] == [2, 2]
    failovers = client_events(caplog, "model_failover")
    assert [(e["from_entry"], e["to_entry"], e["cause"]) for e in failovers] == [(0, 1, "deadline")]
    (sticky,) = client_events(caplog, "model_turn_sticky")
    assert (sticky["entry"], sticky["model_id"], sticky["decision"]) == (1, IDS[1], "stay_on_switched_entry")
    assert sticky["abandoned"] == [0]


def test_a_switched_entry_that_fails_on_a_later_hop_moves_on_and_stays() -> None:
    fakes = [StepModel(0, ["hang", "text"]), StepModel(1, ["tool", "hang"]), StepModel(2, ["tool", "text"])]
    rig = rig_for(fakes)
    said, counts = converse(rig, ["What did I spend?", "Thanks"], fakes)
    assert "Answer from entry 2." in said[0]
    # Hop 1: 0 (miss) -> 1 (tool); hop 2: 1 (miss) -> 2 (tool); hop 3: 2, never back to 0 or 1.
    assert counts[0] == [1, 2, 2]
    assert "Answer from entry 0." in said[1] and counts[1] == [2, 2, 2]


def test_a_turn_with_no_switch_is_unchanged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=CLIENT_LOG)
    fakes = [StepModel(0, ["tool", "text"]), StepModel(1)]
    rig = rig_for(fakes)
    said, counts = converse(rig, ["What did I spend?"], fakes)
    assert "Answer from entry 0." in said[0] and counts[0] == [2, 0]
    assert client_events(caplog, "model_failover") == []
    assert client_events(caplog, "model_turn_sticky") == []


def test_a_turn_whose_first_hop_fails_early_does_not_inherit_the_last_turns_move(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn 2's first hop fails before the agent is called (here the session-ready step);
    the bridge retries the hop. The retry is still turn 2 (its own turn number), so it chooses afresh
 and does not start on the entry turn 1 moved to."""
    fakes = [StepModel(0, ["hang", "text"]), StepModel(1, ["tool", "text"])]
    rig = rig_for(fakes)
    real = rig.bridge._session_ready  # noqa: SLF001
    seen = {"n": 0}

    async def flaky(call_sid: str) -> None:
        seen["n"] += 1
        if seen["n"] == 3:  # turn 1 has two hops; turn 2's first try fails
            raise AgentError("agent owner write failed", retryable=True, code="owner_write_failed")
        await real(call_sid)

    monkeypatch.setattr(rig.bridge, "_session_ready", flaky)
    said, counts = converse(rig, ["What did I spend?", "Thanks, and my savings?"], fakes)
    assert counts[0] == [1, 2]
    assert seen["n"] >= 4 and "Answer from entry 0." in said[1]
    assert counts[1] == [2, 2]
