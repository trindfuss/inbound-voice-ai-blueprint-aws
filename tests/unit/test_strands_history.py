"""``StrandsAgentClient`` history in AgentCore Memory, offline.

AgentCore Memory is :class:`tests.strands_fakes.FakeMemoryData` under the real session manager
(``_NewestStateSessionManager``, or the unpatched library class where a test proves a fix is still
needed). No test touches the network.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import pytest
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

from owlbank.tools_contract import INLINE_TOOLS
from tac_bridge.agent_client import AgentError, TextDelta, ToolCall, ToolResult, ToolStarted, TurnEnd
from tac_bridge.agent_client_strands import (
    MEMORY_ACTOR_ID,
    MEMORY_BATCH_SIZE,
    McpGatewayTools,
    StrandsAgentClient,
    _NewestStateSessionManager,
    _SharedBotoClients,
    memory_session_manager_factory,
)
from tests.logutil import lines, text_of
from tests.strands_fakes import (
    Builds,
    FakeBotoSession,
    FakeGatewayProvider,
    FakeMemoryData,
    FakeModel,
    client_error,
    collect,
    content_of,
    default_gateway,
    gateway_tool,
    light_memory_client,
    make_memory_client,
    memory_factory,
    settings,
    text_msg,
    tool_msg,
)

SID = "CA" + "6" * 32
LOGGER = "tac_bridge.agent_client_strands"


@pytest.fixture(autouse=True)
def _light(monkeypatch: pytest.MonkeyPatch) -> None:
    light_memory_client(monkeypatch)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


async def say(client: StrandsAgentClient, text: str | None = None, results: Any = ()) -> list[Any]:
    return await collect(client.turn(SID, user_text=text, tool_results=list(results),
                                     system_prompts={"haiku-5-5": "BASE"}, system_suffix="Clock: 12:00"))


def agent_of(client: StrandsAgentClient, sid: str = SID) -> Any:
    return client._calls[sid].build.result().agent


def manager_of(client: StrandsAgentClient, sid: str = SID) -> Any:
    return client._calls[sid].build.result().manager


# --------------------------------------------------------------------------- the session manager


def test_default_factory_from_settings_is_lazy():
    client = StrandsAgentClient(settings(memory_id="mem-1"), model_factory=lambda _s, _sid: FakeModel())
    assert client._session_manager_factory is not None
    shared = client._session_manager_factory._clients  # type: ignore[attr-defined]
    assert isinstance(shared, _SharedBotoClients)
    assert shared._clients == {}  # no boto client until the first build (or the prewarm)
    # without a Memory ID, history stays in memory
    assert StrandsAgentClient(settings(), model_factory=lambda _s, _sid: FakeModel())._session_manager_factory is None
    with pytest.raises(ValueError):
        memory_session_manager_factory("", "us-east-1")


def test_factory_config():
    data = FakeMemoryData()
    manager = memory_session_manager_factory("mem-1", "us-east-1", boto_session=FakeBotoSession(data))(SID, "new")
    assert isinstance(manager, _NewestStateSessionManager)
    c = manager.config
    assert (c.memory_id, c.session_id, c.actor_id) == ("mem-1", SID, MEMORY_ACTOR_ID)
    assert (c.batch_size, c.async_mode, c.retrieval_config) == (MEMORY_BATCH_SIZE, True, None)
    assert MEMORY_BATCH_SIZE == 100 and MEMORY_ACTOR_ID == "owlbank-voice-strands"
    assert manager.memory_client.gmdp_client is data


def test_shared_boto_clients_make_one_client_per_service():
    made: list[str] = []

    class Session:
        def client(self, service: str, region_name: str | None = None, config: Any = None) -> object:
            made.append(service)
            return object()

    shared = _SharedBotoClients("us-east-1", session_factory=Session)
    threads = [threading.Thread(target=shared.client, args=("bedrock-agentcore",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert shared.client("bedrock-agentcore") is shared.client("bedrock-agentcore")
    assert made == ["bedrock-agentcore"]


def test_new_does_no_list_events():
    data = FakeMemoryData()
    client = make_memory_client(FakeModel([text_msg("hello")]), data)

    async def go() -> None:
        await client.begin_session(SID, history="new")
        assert data.count("ListEvents") == 0
        assert data.count("CreateEvent") == 1  # the session event, so a later reload finds it
        await say(client, "hi")

    run(go())
    assert data.count("ListEvents") == 0
    assert data.stored_messages(SID) == [{"role": "user", "content": [{"text": "hi"}]},
                                         {"role": "assistant", "content": [{"text": "hello"}]}]


def test_unpatched_new_session_reads_memory():
    """Why ``fresh`` exists: the library reads (twice) even for a session it creates."""
    data = FakeMemoryData()
    client = make_memory_client(FakeModel(), data, cls=AgentCoreMemorySessionManager)
    run(client.begin_session(SID, history="new"))
    assert data.count("ListEvents") == 2


def test_writes_are_batched_one_flush_per_invocation():
    data = FakeMemoryData()
    m = FakeModel([tool_msg(("level0___faq_lookup", {"q": "x"})), text_msg("answer")])
    client = make_memory_client(m, data, gateway=default_gateway())

    async def go() -> list[Any]:
        await client.begin_session(SID, history="new")
        before = data.count("CreateEvent")
        seen: list[int] = []
        data.on_op = lambda name, kw: seen.append(len(m.calls))
        out = await say(client, "hours?")
        assert data.count("CreateEvent") - before == 2  # one messages event + one agent-state event
        assert seen == [2, 2]  # both after the last model call: nothing before any model call
        return out

    assert [type(e) for e in run(go())] == [ToolStarted, TextDelta, TurnEnd]  # the Gateway tool start
    assert len(data.stored_messages(SID)) == 4  # user, toolUse, toolResult, answer


def test_reload_restores_history_and_logs_the_rebuild(caplog):
    data = FakeMemoryData()
    first = make_memory_client(FakeModel([text_msg("one"), text_msg("two")]), data)

    async def go() -> None:
        await first.begin_session(SID, history="new")
        await say(first, "a")
        await say(first, "b")

    run(go())
    m = FakeModel([text_msg("three")])
    second = make_memory_client(m, data)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        run(second.begin_session(SID, history="reload"))
        run(say(second, "c"))
    assert [content_of(x) for x in m.calls[0].messages] == data.stored_messages(SID)[:-1]
    assert len(m.calls[0].messages) == 5
    reload_lines = [f for f in lines(caplog.records, LOGGER) if f["event"] == "agent_history_reload"]
    assert len(reload_lines) == 1
    line = reload_lines[0]
    assert line["reason"] == "owner_changed" and line["outcome"] == "ok" and line["messages"] == 4
    assert isinstance(line["duration_ms"], float)
    assert "one" not in text_of(caplog.records) and "two" not in text_of(caplog.records)


def test_cached_uses_the_cache_and_reloads_when_evicted(caplog):
    data = FakeMemoryData()
    builds = Builds()
    client = make_memory_client(FakeModel(), data, builds=builds)

    async def go() -> None:
        await client.begin_session(SID, history="new")
        await say(client, "a")
        agent = agent_of(client)
        await client.begin_session(SID, history="cached")
        assert agent_of(client) is agent
        reads = data.count("ListEvents")
        await say(client, "b")
        assert data.count("ListEvents") == reads  # no read on a cached turn
        del client._calls[SID]  # evicted
        await client.begin_session(SID, history="cached")
        assert agent_of(client) is not agent
        assert len(agent_of(client).messages) == 4

    with caplog.at_level(logging.INFO, logger=LOGGER):
        run(go())
    assert [mode for _sid, mode in builds.made] == ["new", "reload"]
    reasons = [f["reason"] for f in lines(caplog.records, LOGGER) if f["event"] == "agent_history_reload"]
    assert reasons == ["cache_miss"]


def test_turn_without_session_start_reloads():
    data = FakeMemoryData()
    first = make_memory_client(FakeModel(), data)
    run(first.begin_session(SID, history="new"))
    run(say(first, "a"))
    builds = Builds()
    m = FakeModel()
    second = make_memory_client(m, data, builds=builds)
    run(say(second, "b"))
    assert builds.made == [(SID, "reload")]
    assert len(m.calls[0].messages) == 3


def test_reload_replaces_a_cached_agent():
    data = FakeMemoryData()
    client = make_memory_client(FakeModel(), data)
    run(client.begin_session(SID, history="new"))
    run(say(client, "a"))
    agent = agent_of(client)
    run(client.begin_session(SID, history="reload"))
    assert agent_of(client) is not agent
    assert [content_of(x) for x in agent_of(client).messages] == [content_of(x) for x in agent.messages]


# --------------------------------------------------------------------------- support: the newest state


def _interrupted_session(data: FakeMemoryData, cls: Any = None) -> None:
    client = make_memory_client(FakeModel([tool_msg(("t1", "make_payment", {"amount": "5.00"}))]), data, cls=cls)

    async def go() -> None:
        await client.begin_session(SID, history="new")
        out = await say(client, "pay five")
        assert out[0] == ToolCall("t1", "make_payment", {"amount": "5.00"})

    run(go())


def test_reload_restores_the_newest_agent_state_and_resumes():
    data = FakeMemoryData()
    _interrupted_session(data)
    m = FakeModel([text_msg("paid")])
    client = make_memory_client(m, data)
    run(client.begin_session(SID, history="reload"))
    assert agent_of(client)._interrupt_state.activated is True
    out = run(say(client, results=[ToolResult("t1", {"ok": True})]))
    assert out == [TextDelta("paid"), TurnEnd("end_turn", {"input_tokens": 10, "output_tokens": 5})]
    assert "toolResult" in m.calls[0].messages[-1]["content"][0]


def test_library_bug_unpatched_manager_restores_stale_interrupt_state():
    """Documents the bedrock-agentcore 1.24.1 bug the subclass fixes: with batched writes the
    library restores the OLDEST agent-state snapshot of a flush (``payload[0]``), so a rebuilt agent
    is not interrupted and can't take the tool result. If this test fails, the library has fixed
    the bug: re-check ``_NewestStateSessionManager.read_agent`` before removing it."""
    data = FakeMemoryData()
    _interrupted_session(data, cls=AgentCoreMemorySessionManager)
    assert data.newest_agent_state(SID)["_internal_state"]["interrupt_state"]["activated"] is True
    client = make_memory_client(FakeModel(), data, cls=AgentCoreMemorySessionManager)
    run(client.begin_session(SID, history="reload"))
    assert agent_of(client)._interrupt_state.activated is False  # stale: the bug
    with pytest.raises(AgentError) as e:
        run(say(client, results=[ToolResult("t1", {"ok": True})]))
    assert e.value.code == "unknown_tool_use"


# --------------------------------------------------------------------------- unknown tool use recovery


def _recovery_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [f for f in lines(caplog.records, LOGGER) if f.get("event") == "agent_history_reload"
            and f.get("reason") == "unknown_tool_use_recovery"]


def test_unknown_tool_use_on_a_new_agent_reloads_once_from_memory(caplog: pytest.LogCaptureFixture):
    """The owner was lost, so the session started ``new`` (empty) while
    Memory holds the hand-off; the result reloads the agent once and the turn goes on."""
    data = FakeMemoryData()
    _interrupted_session(data)
    builds = Builds()
    m = FakeModel([text_msg("paid")])
    client = make_memory_client(m, data, builds=builds)
    run(client.begin_session(SID, history="new"))
    assert agent_of(client).messages == []
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = run(say(client, results=[ToolResult("t1", {"ok": True})]))
    assert out == [TextDelta("paid"), TurnEnd("end_turn", {"input_tokens": 10, "output_tokens": 5})]
    assert [mode for _sid, mode in builds.made] == ["new", "reload"]
    sent = m.calls[0].messages
    assert sent[-1]["content"][0]["toolResult"]["toolUseId"] == "t1"
    assert sent[-2]["content"][0]["toolUse"]["name"] == "make_payment"  # the history came back
    (line,) = _recovery_lines(caplog)
    assert line["outcome"] == "ok" and line["interrupted"] is True


def test_unknown_tool_use_on_a_stale_cached_agent_reloads(caplog: pytest.LogCaptureFixture):
    """``cached`` but the cache is stale (the call ran elsewhere and the owner was lost)."""
    data = FakeMemoryData()
    builds = Builds()
    stale = make_memory_client(FakeModel([text_msg("hello")]), data, builds=builds)
    run(stale.begin_session(SID, history="new"))
    run(collect(stale.turn(SID, user_text="hi", tool_results=[], system_prompts={"haiku-5-5": "p"})))
    other = make_memory_client(FakeModel([tool_msg(("t7", "get_balance", {"account": "checking"}))]), data)
    run(other.begin_session(SID, history="reload"))
    run(collect(other.turn(SID, user_text="balance", tool_results=[], system_prompts={"haiku-5-5": "p"})))
    run(stale.begin_session(SID, history="cached"))  # the wrong decision: its agent misses t7
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = run(collect(stale.turn(SID, user_text=None, tool_results=[ToolResult("t7", {"ok": 1})],
                                     system_prompts={"haiku-5-5": "p"})))
    assert out[-1].stop_reason == "end_turn"
    assert [mode for _sid, mode in builds.made] == ["new", "reload"]
    assert len(_recovery_lines(caplog)) == 1


def test_still_unknown_after_the_recovery_fails_without_looping():
    data = FakeMemoryData()
    _interrupted_session(data)
    builds = Builds()
    m = FakeModel()
    client = make_memory_client(m, data, builds=builds)
    run(client.begin_session(SID, history="new"))
    with pytest.raises(AgentError) as e:
        run(say(client, results=[ToolResult("t-other", {"ok": True})]))
    assert (e.value.code, e.value.retryable) == ("unknown_tool_use", False)
    with pytest.raises(AgentError) as e:  # the agent was reloaded in this session: no second reload
        run(say(client, results=[ToolResult("t-other", {"ok": True})]))
    assert e.value.code == "unknown_tool_use"
    assert [mode for _sid, mode in builds.made] == ["new", "reload"]
    assert m.calls == []


def test_no_recovery_after_a_reload_in_this_session():
    data = FakeMemoryData()
    _interrupted_session(data)
    builds = Builds()
    client = make_memory_client(FakeModel(), data, builds=builds)
    run(client.begin_session(SID, history="reload"))
    with pytest.raises(AgentError) as e:
        run(say(client, results=[ToolResult("t-other", {"ok": True})]))
    assert e.value.code == "unknown_tool_use"
    assert [mode for _sid, mode in builds.made] == ["reload"]


def test_a_new_session_on_the_cached_agent_allows_the_recovery_again():
    data = FakeMemoryData()
    _interrupted_session(data)
    builds = Builds()
    client = make_memory_client(FakeModel([text_msg("paid")]), data, builds=builds)
    run(client.begin_session(SID, history="reload"))
    with pytest.raises(AgentError):
        run(say(client, results=[ToolResult("t-other", {})]))
    run(client.begin_session(SID, history="reload"))  # a reload session: still no recovery
    with pytest.raises(AgentError):
        run(say(client, results=[ToolResult("t-other", {})]))
    run(client.begin_session(SID, history="cached"))  # a later session kept the cached agent
    with pytest.raises(AgentError):
        run(say(client, results=[ToolResult("t-other", {})]))
    assert [mode for _sid, mode in builds.made] == ["reload", "reload", "reload"]


def test_aclose_drops_the_snapshot_of_an_agent_that_never_ran():
    """A process holding a stale, never-run agent (its session started, the call moved
    on and handed off elsewhere) must not write that agent's creation snapshot at shutdown: it would
    become the newest agent state and the next reload would lose the pending hand-off."""
    data = FakeMemoryData()
    idle = make_memory_client(FakeModel(), data)
    run(idle.begin_session(SID, history="new"))  # never runs a turn
    _interrupted_session(data)  # another process: the call's hand-off (t1) is in Memory
    run(idle.aclose())
    assert data.newest_agent_state(SID)["_internal_state"]["interrupt_state"]["activated"] is True
    client = make_memory_client(FakeModel([text_msg("paid")]), data)
    run(client.begin_session(SID, history="reload"))
    out = run(say(client, results=[ToolResult("t1", {"ok": True})]))
    assert out[-1].stop_reason == "end_turn"


# --------------------------------------------------------------------------- failed reads


@pytest.mark.parametrize("nth", [0, 1, 2])  # read_session, read_agent, list_messages
def test_failed_read_is_history_unavailable_never_empty(nth: int):
    data = FakeMemoryData()
    first = make_memory_client(FakeModel(), data)
    run(first.begin_session(SID, history="new"))
    run(say(first, "a"))
    calls = {"n": 0}

    def fail(_kw: dict[str, Any]) -> Exception | None:
        calls["n"] += 1
        return client_error("ServiceUnavailableException", op="ListEvents") if calls["n"] == nth + 1 else None

    data.fail["list_events"] = fail
    m = FakeModel()
    client = make_memory_client(m, data)
    with pytest.raises(AgentError) as e:
        run(client.begin_session(SID, history="reload"))
    assert (e.value.code, e.value.retryable) == ("history_unavailable", True)
    assert SID not in client._calls
    data.fail.clear()
    run(say(client, "b"))  # the next turn reads again, and gets the whole history
    assert len(m.calls[0].messages) == 3


def test_unpatched_failed_read_continues_with_empty_history():
    """Why ``read_agent`` and ``list_messages`` raise: the library returns ``[]`` on a failed read."""
    data = FakeMemoryData()
    first = make_memory_client(FakeModel(), data)
    run(first.begin_session(SID, history="new"))
    run(say(first, "a"))
    calls = {"n": 0}

    def fail(_kw: dict[str, Any]) -> Exception | None:
        calls["n"] += 1
        return client_error("ServiceUnavailableException", op="ListEvents") if calls["n"] == 3 else None

    data.fail["list_events"] = fail
    m = FakeModel()
    client = make_memory_client(m, data, cls=AgentCoreMemorySessionManager)
    run(say(client, "b"))
    assert len(m.calls[0].messages) == 1  # the history is silently gone


# --------------------------------------------------------------------------- section 4: flush and hand-offs


def test_handoff_tool_call_comes_after_the_flush():
    data = FakeMemoryData()
    client = make_memory_client(FakeModel([tool_msg(("t9", "escalate", {"reason": "x", "summary": "y"}))]), data)

    async def go() -> None:
        await client.begin_session(SID, history="new")
        async for ev in client.turn(SID, user_text="agent please", tool_results=[], system_prompts={"haiku-5-5": "p"}):
            if isinstance(ev, ToolCall):
                assert manager_of(client).pending() == 0
                state = data.newest_agent_state(SID)["_internal_state"]["interrupt_state"]
                assert state["activated"] is True
                assert data.stored_messages(SID)[-1]["content"][0]["toolUse"]["toolUseId"] == "t9"

    run(go())


def test_inline_bank_call_is_emitted_at_the_interrupt():
    data = FakeMemoryData()
    client = make_memory_client(FakeModel([tool_msg(("t2", "list_recent_transactions", {}))]), data)

    async def go() -> None:
        await client.begin_session(SID, history="new")
        async for ev in client.turn(SID, user_text="what did I spend", tool_results=[], system_prompts={"haiku-5-5": "p"}):
            if isinstance(ev, ToolCall):
                assert manager_of(client).pending() > 0  # not deferred: the flush comes later

    run(go())


def _fail_flushes(data: FakeMemoryData) -> None:
    data.fail["create_event"] = lambda kw: client_error("ThrottlingException", op="CreateEvent")


@pytest.mark.parametrize("script", [
    [tool_msg(("t3", "authenticate", {"target": "get_balance"}))],  # a hand-off
    [text_msg("plain")],
    [tool_msg(("t4", "transfer_funds", {"amount": "1.00"}))],  # money moves: never on a lost history
])
def test_flush_failure_fails_the_turn_and_drops_the_agent(script: list[Any], caplog):
    data = FakeMemoryData()
    m = FakeModel(script)
    client = make_memory_client(m, data)
    run(client.begin_session(SID, history="new"))
    _fail_flushes(data)
    seen: list[Any] = []

    async def go() -> None:
        async for ev in client.turn(SID, user_text="go", tool_results=[], system_prompts={"haiku-5-5": "p"}):
            seen.append(ev)

    with caplog.at_level(logging.INFO, logger=LOGGER), pytest.raises(AgentError) as e:
        run(go())
    assert (e.value.code, e.value.retryable) == ("history_write_failed", True)
    assert not any(isinstance(ev, ToolCall) and ev.name in ("authenticate", "escalate") for ev in seen)
    assert not any(isinstance(ev, TurnEnd) for ev in seen)
    assert SID not in client._calls  # the retry rebuilds from what Memory holds
    rec = [f for f in lines(caplog.records, LOGGER) if f.get("operation") == "memory_flush"]
    assert rec and rec[0]["decision"] == "fail_turn_retryable"
    data.fail.clear()
    m.script = [text_msg("again")]
    out = run(say(client, "go"))
    assert out[0] == TextDelta("again")
    assert len(m.calls[-1].messages) == 1  # the failed invocation never reached Memory


class _NoAutoFlush(_NewestStateSessionManager):
    """A session manager whose end-of-invocation flush does nothing (as if a library version
    stopped flushing before ``result``), to prove the client flushes itself before a hand-off."""

    def register_hooks(self, registry: Any, **kwargs: Any) -> None:
        self._flush_messages = list  # type: ignore[method-assign]  # what the hook binds: a no-op
        try:
            super().register_hooks(registry, **kwargs)
        finally:
            del self._flush_messages


def test_client_flushes_itself_if_the_library_did_not(caplog):
    data = FakeMemoryData()
    client = make_memory_client(FakeModel([tool_msg(("t5", "escalate", {"reason": "x", "summary": "y"}))]),
                                data, cls=_NoAutoFlush)

    async def go() -> None:
        await client.begin_session(SID, history="reload")
        async for ev in client.turn(SID, user_text="help", tool_results=[], system_prompts={"haiku-5-5": "p"}):
            if isinstance(ev, ToolCall):
                assert manager_of(client).pending() == 0
                assert data.newest_agent_state(SID)["_internal_state"]["interrupt_state"]["activated"]

    with caplog.at_level(logging.INFO, logger=LOGGER):
        run(go())
    assert any(f["event"] == "memory_flush_explicit" for f in lines(caplog.records, LOGGER))


def test_auto_flush_is_what_normally_guarantees_the_order(caplog):
    data = FakeMemoryData()
    client = make_memory_client(FakeModel([tool_msg(("t6", "get_balance", {"account": "checking"}))]), data)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        run(client.begin_session(SID, history="new"))
        out = run(say(client, "balance"))
    assert out[0].name == "get_balance"
    assert manager_of(client).flushes == 1
    assert not any(f["event"] == "memory_flush_explicit" for f in lines(caplog.records, LOGGER))


# --------------------------------------------------------------------------- cache and shutdown


def test_aclose_closes_every_session_manager_and_the_gateway():
    data = FakeMemoryData()
    builds = Builds()
    gateway = default_gateway()
    client = make_memory_client(FakeModel(), data, builds=builds, gateway=gateway)
    closed: list[str] = []

    async def go() -> None:
        for sid in ("CA" + "1" * 32, "CA" + "2" * 32):
            await client.begin_session(sid, history="new")
        for manager in builds.managers:
            real = manager.close
            manager.close = lambda real=real, sid=manager.config.session_id: (closed.append(sid), real())
        await client.aclose()

    run(go())
    assert sorted(closed) == ["CA" + "1" * 32, "CA" + "2" * 32]
    assert gateway.closed and not client._calls


def test_aclose_survives_a_failing_close():
    data = FakeMemoryData()
    builds = Builds()
    client = make_memory_client(FakeModel(), data, builds=builds)

    async def go() -> None:
        await client.begin_session(SID, history="new")

        def boom() -> None:
            raise RuntimeError("x")

        builds.managers[0].close = boom
        await client.aclose()

    run(go())


# --------------------------------------------------------------------------- Gateway: late recovery


def test_gateway_tools_bound_at_a_later_turn_once_available():
    m = FakeModel([text_msg("a"), text_msg("b"), text_msg("c")])
    provider = FakeGatewayProvider([gateway_tool("level0___faq_lookup"), gateway_tool("level0___get_branch_hours"),
                                    gateway_tool("level0___delete_everything")], fail=True)
    client = make_memory_client(m, FakeMemoryData(), gateway=provider)
    run(client.begin_session(SID, history="new"))
    run(say(client, "1"))
    assert m.calls[0].tool_names == list(INLINE_TOOLS)
    provider.fail = False
    run(say(client, "2"))
    run(say(client, "3"))
    allowed = [*INLINE_TOOLS, "level0___faq_lookup", "level0___get_branch_hours"]
    assert sorted(m.calls[1].tool_names) == sorted(allowed)  # never the unapproved one
    assert sorted(m.calls[2].tool_names) == sorted(allowed)  # bound once


def test_mcp_gateway_retries_in_the_background():
    attempts: list[str] = []

    class FlakyMcp:
        def start(self) -> None:
            attempts.append("start")
            if len(attempts) == 1:
                raise ConnectionError("down")

        def list_tools_sync(self) -> list[Any]:
            return [gateway_tool("level0___faq_lookup")]

        def stop(self, *a: Any) -> None:
            attempts.append("stop")

    now = {"t": 0.0}
    provider = McpGatewayTools(FlakyMcp(), retry_s=10.0, clock=lambda: now["t"])
    with pytest.raises(ConnectionError):
        provider.start()
    with pytest.raises(RuntimeError):
        provider.tools()  # too soon: no retry yet
    assert attempts == ["start"]
    now["t"] = 11.0
    with pytest.raises(RuntimeError):
        provider.tools()  # starts a retry in a thread, answers at once
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            listed = provider.tools()
            break
        except RuntimeError:
            time.sleep(0.01)
    assert [t.tool_name for t in listed] == ["level0___faq_lookup"]
    provider.close()
    assert attempts == ["start", "start", "stop"]


def test_memory_factory_records_builds():
    data = FakeMemoryData()
    builds = Builds()
    factory = memory_factory(data, builds)
    factory(SID, "new")
    assert builds.made == [(SID, "new")]


def test_metadata_filter_matches_the_library_builder():
    from bedrock_agentcore.memory.models.filters import (
        EventMetadataFilter,
        LeftExpression,
        OperatorType,
        RightExpression,
    )

    from tac_bridge.agent_client_strands import _meta_equals

    built = EventMetadataFilter.build_expression(left_operand=LeftExpression.build("agentId"),
                                                 operator=OperatorType.EQUALS_TO,
                                                 right_operand=RightExpression.build("owlbank_voice"))
    assert _meta_equals("agentId", "owlbank_voice") == built


def test_a_rebuilt_manager_never_writes_before_what_it_read():
    """The library's event clock is per manager, bumped 1 ms ahead on a
    same-millisecond tie, so a manager built right after a busy one could stamp its events earlier
    and Memory would order them first. The rebuilt manager's clock starts after the newest event."""
    data = FakeMemoryData()
    client = make_memory_client(FakeModel(default="ok"), data)

    async def go() -> None:
        await client.begin_session(SID, history="new")
        for i in range(12):  # many events within a few ms: the clock runs ahead of the wall clock
            await say(client, f"u{i}")
        del client._calls[SID]  # evicted: the next turn reloads with a new manager
        await say(client, "after")

    run(go())
    texts = [m["content"][0]["text"] for m in data.stored_messages(SID) if m["role"] == "user"]
    assert texts == [*(f"u{i}" for i in range(12)), "after"]
