"""Segment-traits: ``StrandsAgentClient.record_exchange`` and the ``escalate``
tool's live reason descriptions, offline.

The persistence path: both messages go through ``Agent._append_messages`` (Strands' own message
path), so ``MessageAddedEvent`` reaches the session manager, which appends and syncs as for any turn;
the AgentCore manager's batched writes are flushed by the next invocation (or by a turn that makes
none). A resumed session reads the pair back, under the AgentCore Memory session manager on the
fake data plane and under Strands' own ``FileSessionManager``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest
from strands.session.file_session_manager import FileSessionManager

from owlbank.tools_contract import DEFAULT_REASON_DESCRIPTIONS
from tac_bridge.agent_client import AgentClient, ToolCall, ToolResult
from tac_bridge.agent_client_strands import StrandsAgentClient
from tests.logutil import lines
from tests.strands_fakes import (
    FakeMemoryData,
    FakeModel,
    client_error,
    collect,
    content_of,
    light_memory_client,
    make_memory_client,
    settings,
    text_msg,
    tool_msg,
)

SID = "CA" + "8" * 32
LOGGER = "tac_bridge.agent_client_strands"
STARTED = "(The call is connected.)"
WELCOME = "Hi Thomas, you're verified. Are you calling about the HELOC you looked at on our website?"
PAIR = [{"role": "user", "content": [{"text": STARTED}]},
        {"role": "assistant", "content": [{"text": WELCOME}]}]


@pytest.fixture(autouse=True)
def _light(monkeypatch: pytest.MonkeyPatch) -> None:
    light_memory_client(monkeypatch)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


async def say(client: StrandsAgentClient, text: str | None = None, results: Any = ()) -> list[Any]:
    return await collect(client.turn(SID, user_text=text, tool_results=list(results),
                                     system_prompts={"haiku-5-5": "BASE"}, system_suffix="Clock: 12:00"))


def agent_of(client: StrandsAgentClient) -> Any:
    return client._calls[SID].build.result().agent


def shape(messages: list[Any]) -> list[dict[str, Any]]:
    return [content_of(m) for m in messages]


def test_the_client_satisfies_the_protocol() -> None:
    assert isinstance(StrandsAgentClient(settings(), model_factory=lambda _s, _sid: FakeModel()), AgentClient)


# --------------------------------------------------------------------------- AgentCore Memory


def test_the_pair_is_recorded_with_no_model_call_and_precedes_the_reply() -> None:
    data = FakeMemoryData()
    m = FakeModel([text_msg("Great, let's look at the HELOC.")])
    client = make_memory_client(m, data)

    async def go() -> None:
        await client.begin_session(SID, history="new")
        await client.record_exchange(SID, STARTED, WELCOME)
        assert m.calls == []  # no model call
        await say(client, "Yes")

    run(go())
    assert shape(m.calls[0].messages) == [*PAIR, {"role": "user", "content": [{"text": "Yes"}]}]
    stored = data.stored_messages(SID)
    assert stored[:2] == PAIR and stored[2] == {"role": "user", "content": [{"text": "Yes"}]}
    assert len(stored) == 4


def test_a_resumed_agentcore_session_reads_the_pair_back() -> None:
    data = FakeMemoryData()
    first = make_memory_client(FakeModel([text_msg("Sure.")]), data)

    async def one() -> None:
        await first.begin_session(SID, history="new")
        await first.record_exchange(SID, STARTED, WELCOME)
        await say(first, "Yes")

    run(one())
    m = FakeModel([text_msg("Anything else?")])
    second = make_memory_client(m, data)

    async def two() -> None:
        await second.begin_session(SID, history="reload")
        await say(second, "No")

    run(two())
    seen = shape(m.calls[0].messages)
    assert seen[:2] == PAIR and seen.count(PAIR[1]) == 1


def test_the_pair_is_written_by_close_when_no_turn_followed() -> None:
    """Shut down between the welcome and the caller's reply: ``close()`` writes the buffered
    messages. (The agent-state snapshot of an agent that never ran an invocation is dropped, as
    before: written late it could override another process's newer state.)"""
    data = FakeMemoryData()
    first = make_memory_client(FakeModel(), data)

    async def one() -> None:
        await first.begin_session(SID, history="new")
        await first.record_exchange(SID, STARTED, WELCOME)
        await first.aclose()

    run(one())
    assert data.stored_messages(SID) == PAIR


def test_a_turn_with_no_invocation_writes_a_buffered_pair() -> None:
    """Blank words make no invocation, so the client flushes the pair itself."""
    data = FakeMemoryData()
    client = make_memory_client(FakeModel(), data)

    async def go() -> None:
        await client.begin_session(SID, history="new")
        await client.record_exchange(SID, STARTED, WELCOME)
        assert client._calls[SID].build.result().manager.pending() > 0
        await say(client, "   ")
        assert client._calls[SID].build.result().manager.pending() == 0

    run(go())
    assert data.stored_messages(SID) == PAIR


def test_after_an_unanswered_user_message_only_the_reply_is_added(caplog: pytest.LogCaptureFixture) -> None:
    """A failed turn left the caller's words unanswered: the welcome alone keeps the roles alternating."""
    data = FakeMemoryData()
    m = FakeModel([client_error("ThrottlingException"), text_msg("ok")])
    client = make_memory_client(m, data)

    async def go() -> None:
        await client.begin_session(SID, history="new")
        with pytest.raises(Exception):  # noqa: B017 - AgentError (retryable throttle)
            await say(client, "hello?")
        with caplog.at_level(logging.INFO, logger=LOGGER):
            await client.record_exchange(SID, STARTED, WELCOME)
        await say(client, "Yes")

    run(go())
    roles = [msg["role"] for msg in m.calls[-1].messages]
    assert roles == ["user", "assistant", "user"]
    assert shape(m.calls[-1].messages)[1] == PAIR[1]
    assert STARTED not in str(m.calls[-1].messages)
    (rec,) = [e for e in lines(caplog.records, LOGGER) if e["event"] == "strands_exchange_recorded"]
    assert rec["messages"] == 1


def test_a_pending_tool_use_leaves_the_history_unchanged(caplog: pytest.LogCaptureFixture) -> None:
    data = FakeMemoryData()
    m = FakeModel([tool_msg(("t1", "make_payment", {"amount": "5.00"})), text_msg("paid")])
    client = make_memory_client(m, data)

    async def go() -> None:
        await client.begin_session(SID, history="new")
        out = await say(client, "pay five")
        assert out[0] == ToolCall("t1", "make_payment", {"amount": "5.00"})
        before = shape(agent_of(client).messages)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            await client.record_exchange(SID, STARTED, WELCOME)
        assert shape(agent_of(client).messages) == before
        await say(client, results=[ToolResult("t1", {"ok": True, "welcome": WELCOME})])

    run(go())
    (skip,) = [e for e in lines(caplog.records, LOGGER) if e["event"] == "strands_exchange_skipped"]
    assert skip["reason"] == "tool_use_pending"
    last = m.calls[-1].messages[-1]["content"][0]["toolResult"]
    assert WELCOME in last["content"][0]["text"]


@pytest.mark.parametrize(("user", "reply"), [("", WELCOME), (STARTED, "  "), ("\n", "")])
def test_blank_text_adds_nothing(user: str, reply: str) -> None:
    client = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: FakeModel())

    async def go() -> None:
        await client.begin_session(SID, history="new")
        await client.record_exchange(SID, user, reply)

    run(go())
    assert agent_of(client).messages == []


# --------------------------------------------------------------------------- Strands' own session manager


def test_a_resumed_file_session_reads_the_pair_back(tmp_path: Path) -> None:
    """Strands' ``FileSessionManager`` (RepositorySessionManager's ``MessageAddedEvent`` hook)
    persists the pair through the same message path."""

    def factory(session_id: str, _mode: str) -> Any:
        return FileSessionManager(session_id=session_id, storage_dir=str(tmp_path))

    first = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: FakeModel(),
                               session_manager_factory=factory)

    async def one() -> None:
        await first.begin_session(SID, history="new")
        await first.record_exchange(SID, STARTED, WELCOME)

    run(one())
    m = FakeModel([text_msg("Sure.")])
    second = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: m, session_manager_factory=factory)

    async def two() -> None:
        await second.begin_session(SID, history="reload")
        await say(second, "Yes")

    run(two())
    assert shape(m.calls[0].messages) == [*PAIR, {"role": "user", "content": [{"text": "Yes"}]}]


def test_in_memory_history_keeps_the_pair_for_the_next_turn() -> None:
    m = FakeModel([text_msg("Sure.")])
    client = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: m)

    async def go() -> None:
        await client.begin_session(SID, history="new")
        await client.record_exchange(SID, STARTED, WELCOME)
        await say(client, "Yes")

    run(go())
    assert shape(m.calls[0].messages)[:2] == PAIR


# --------------------------------------------------------------------------- reason descriptions


def _reason_description(m: FakeModel) -> str:
    (spec,) = [t for t in m.calls[0].tool_specs if t["name"] == "escalate"]
    return str(spec["inputSchema"]["json"]["properties"]["reason"].get("description", ""))


def _first_turn(client: StrandsAgentClient, m: FakeModel, sid: str = SID) -> str:
    async def go() -> None:
        await collect(client.turn(sid, user_text="hi", tool_results=[], system_prompts={"haiku-5-5": "BASE"}))

    run(go())
    return _reason_description(m)


def test_the_escalate_spec_carries_the_given_descriptions() -> None:
    m = FakeModel([text_msg("hi")])
    live = {"card_dispute": "Zebraquartz dispute line.", "fraud_suspected": "Heronbasalt fraud line."}
    client = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: m, reason_descriptions=live)
    desc = _first_turn(client, m)
    assert "card_dispute: Zebraquartz dispute line." in desc
    assert "fraud_suspected: Heronbasalt fraud line." in desc
    assert DEFAULT_REASON_DESCRIPTIONS["card_dispute"] not in desc


def test_the_live_source_is_read_at_each_agent_build() -> None:
    m = FakeModel([text_msg("one"), text_msg("two")])
    current = {"card_dispute": "First wording."}
    client = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: m)
    client.use_reason_descriptions(lambda: current)
    assert "card_dispute: First wording." in _first_turn(client, m, "CA" + "1" * 32)
    current = {"card_dispute": "Second wording."}
    m.calls.clear()
    assert "card_dispute: Second wording." in _first_turn(client, m, "CA" + "2" * 32)  # a new call's agent


@pytest.mark.parametrize("source", [lambda: (_ for _ in ()).throw(RuntimeError("store down")),
                                    lambda: ["not", "a", "mapping"], lambda: {"card_dispute": 3}])
def test_a_failing_source_falls_back_to_the_seed_descriptions(source: Any,
                                                             caplog: pytest.LogCaptureFixture) -> None:
    m = FakeModel([text_msg("hi")])
    with caplog.at_level(logging.INFO, logger=LOGGER):
        client = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: m, reason_descriptions=source)
        desc = _first_turn(client, m)
    assert DEFAULT_REASON_DESCRIPTIONS["card_dispute"] in desc
    assert any(e["event"] == "reason descriptions unavailable" for e in lines(caplog.records, LOGGER))
