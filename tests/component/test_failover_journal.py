"""Model failover journal through the bridge.

The bridge runs a real :class:`~tac_bridge.agent_client_strands.StrandsAgentClient` over the
simulator; each turn's ``ai_turn`` journal event names the model that answered (the entry's ID,
nothing else), also when the turn moved to the next entry before anything was spoken, and the
caller hears only the answering model's words.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from owlbank.failover import FailoverState, Settings
from tac_bridge.agent_client_strands import ModelEntry, StrandsAgentClient, StrandsSettings
from tests.component.test_bridge_fakes import CALL_SID, Rig, make_rig, run, spoken, write_call_state
from tests.component.test_stream_release import StreamAgent, chunks, msg
from tests.strands_fakes import FakeMemoryData, FakeModel, client_error, default_gateway, memory_factory
from tests.strands_parity import ParityModel, memory_for

IDS = ("us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5")


class Throttled(FakeModel):
    """An entry whose every call is throttled before any output."""

    async def stream(self, *a: Any, **k: Any) -> Any:
        self.calls.append(None)  # type: ignore[arg-type]
        raise client_error("ThrottlingException")
        yield  # pragma: no cover


@dataclass
class TwoModelAgent(StreamAgent):
    """The stream rig's agent with a list of two: a throttled preferred entry, then the scripted one."""

    def _attach(self, memory: FakeMemoryData) -> None:
        self.model = ParityModel()
        self.first = Throttled()
        self.memory_data = memory
        base = Settings.for_models(IDS)
        models = (ModelEntry(id=IDS[0], thinking="adaptive", max_tokens=2000),
                  ModelEntry(id=IDS[1], thinking="default", max_tokens=400))
        by_id = {IDS[0]: self.first, IDS[1]: self.model}
        self.client = StrandsAgentClient(
            StrandsSettings(models=models, region="us-east-1", memory_id="mem-1"),
            model_factory=lambda one, _sid: by_id[one.primary.id], gateway_tools=default_gateway(),
            session_manager_factory=memory_factory(memory),
            failover_state=FailoverState(replace(base, entries=(replace(base.entries[0], deadline_s=1.0),
                                                                base.entries[1]))))
        self.emitted: set[str] = set()


def rig_with(agent_cls: type[StreamAgent], script: list[Any]) -> Rig:
    rig = make_rig()
    agent = agent_cls(list(script), memory=memory_for(rig.store))
    rig.agent = agent
    rig.bridge._agent = agent  # noqa: SLF001 - the rig's agent, swapped
    write_call_state(rig.store)
    return rig


def converse(rig: Rig, prompt: str) -> list[dict[str, Any]]:
    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt(prompt, timeout=5.0)
        await sim.hangup()
        return frames

    return run(go())


def ai_turns(rig: Rig) -> list[dict[str, Any]]:
    return [e for e in rig.events() if e["type"] == "ai_turn" and "text" in e]


def test_the_journal_names_the_model_of_a_list_of_one() -> None:
    rig = rig_with(StreamAgent, [msg(chunks("Your savings balance is fine."))])
    frames = converse(rig, "How is my savings?")
    assert spoken(frames) == "Your savings balance is fine."
    (ev,) = ai_turns(rig)
    assert ev["model"] == "us.anthropic.claude-haiku-5-5"


def test_a_failover_is_unheard_and_the_journal_names_the_model_that_answered() -> None:
    rig = rig_with(TwoModelAgent, [msg(chunks("Your savings balance is fine."))])
    frames = converse(rig, "How is my savings?")
    assert spoken(frames) == "Your savings balance is fine."  # no apology: nothing failed for the caller
    (ev,) = ai_turns(rig)
    assert ev["model"] == IDS[1]
    assert set(ev) <= {"type", "text", "model", "call_sid", "seq", "ts", "expires_at"}
    agent = rig.agent
    assert len(agent.first.calls) == 1 and len(agent.model.calls) == 1
    messages = agent.memory_data.stored_messages(CALL_SID)
    assert sum(1 for m in messages if m["role"] == "user" and any("text" in b for b in m["content"])) == 1

