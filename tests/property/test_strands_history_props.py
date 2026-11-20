"""Properties of ``StrandsAgentClient`` history and the
cache bound, offline.

AgentCore Memory is :class:`tests.strands_fakes.FakeMemoryData` under the real session manager
(batched writes); the model is a scripted ``FakeModel``. The ownership check is a Hypothesis state machine over
simulated bridge processes that share one call-state store (the bridge's own
``CallStateRepo.claim_agent_owner``) and one Memory.
"""

from __future__ import annotations

import asyncio
import string
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from owlbank import callstate
from owlbank.store import MemoryStore
from owlbank.tools_contract import HANDOFF_TOOLS, INLINE_TOOLS
from tac_bridge.agent_client import AgentError, TextDelta, ToolCall, ToolResult, TurnEnd
from tac_bridge.agent_client_strands import ALLOWED_GATEWAY_TOOLS, StrandsAgentClient, pending_interrupts
from tac_bridge.bridge import CallStateRepo
from tests.strands_fakes import (
    Builds,
    FakeMemoryData,
    FakeModel,
    client_error,
    collect,
    content_of,
    default_gateway,
    light_memory_client,
    make_memory_client,
    text_msg,
    tool_msg,
)
from tests.strands_fakes import settings as client_settings

SID = "CA" + "8" * 32
PROPS = settings(max_examples=30, deadline=None,
                 suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])


@pytest.fixture(autouse=True)
def _light(monkeypatch: pytest.MonkeyPatch) -> None:
    light_memory_client(monkeypatch)


# Caller words have a visible character: speech recognition never sends a blank final, and Memory
# drops blank text blocks (the library's converter filters them), so a blank-only message would not
# round-trip (the bridge never sends one).
words = st.text(alphabet=string.ascii_letters + " ", min_size=1, max_size=12).filter(str.strip)
inline_names = st.sampled_from(INLINE_TOOLS)
gateway_names = st.sampled_from(ALLOWED_GATEWAY_TOOLS)


async def turn(client: StrandsAgentClient, text: str | None = None, results: Any = ()) -> list[Any]:
    return await collect(client.turn(SID, user_text=text, tool_results=list(results), system_prompts={"haiku-5-5": "BASE"},
                                     system_suffix="Clock"))


# --------------------------------------------------------------------------- Round trip


# A conversation: text turns (some with a Gateway tool run inside the loop), then one turn that
# stops on 1-2 inline calls (optionally with a Gateway call in the same message).
text_turn = st.tuples(words, st.booleans(), st.one_of(st.none(), gateway_names))
final_turn = st.tuples(words, st.lists(inline_names, min_size=1, max_size=2, unique=True),
                       st.one_of(st.none(), gateway_names), st.booleans())


def _script(turns: list[tuple[str, bool, str | None]], final: tuple[str, list[str], str | None, bool]) -> list[Any]:
    out: list[Any] = []
    for i, (_text, reasoning, gw) in enumerate(turns):
        if gw is not None:
            out.append(tool_msg((f"g{i}", gw, {"q": "x"})))
        out.append(text_msg(f"reply {i}", reasoning="" if reasoning else None))
    _text, inline, gw, reasoning = final
    uses: list[tuple[str, str, dict[str, Any]]] = [(f"i{j}", n, {"n": j}) for j, n in enumerate(inline)]
    if gw is not None:
        uses.append(("gfinal", gw, {"q": "y"}))
    out.append(tool_msg(*uses, reasoning="" if reasoning else None))
    return out


@PROPS
@given(st.lists(text_turn, max_size=3), final_turn)
def test_rebuild_round_trips_messages_and_interrupts(turns: list[Any], final: Any) -> None:
    data = FakeMemoryData()
    first = make_memory_client(FakeModel(_script(turns, final)), data, gateway=default_gateway())

    async def build_history() -> Any:
        await first.begin_session(SID, history="new")
        for text, _r, _g in turns:
            await turn(first, text)
        await turn(first, final[0])
        return first._calls[SID].build.result().agent

    original = asyncio.run(build_history())
    assert original._interrupt_state.activated

    m = FakeModel([text_msg("resumed")])
    second = make_memory_client(m, data, gateway=default_gateway())

    async def rebuild_and_resume() -> tuple[Any, list[Any]]:
        await second.begin_session(SID, history="reload")
        rebuilt = second._calls[SID].build.result().agent
        pending = pending_interrupts(rebuilt)
        out = await turn(second, results=[ToolResult(tid, {"ok": tid}) for tid in pending])
        return rebuilt, out

    _rebuilt, out = asyncio.run(rebuild_and_resume())
    assert out[-1] == TurnEnd("end_turn", {"input_tokens": 10, "output_tokens": 5})
    assert TextDelta("resumed") in out
    # the model, resumed, saw exactly the original history plus the tool results
    sent = [content_of(x) for x in m.calls[0].messages]
    assert sent[:len(original.messages)] == [content_of(x) for x in original.messages]
    results = [c["toolResult"]["toolUseId"] for c in sent[len(original.messages)]["content"] if "toolResult" in c]
    # the inline results the bridge sent, plus the Gateway result held in the interrupt state
    assert sorted(results) == sorted(pending_ids(original) | ({"gfinal"} if final[2] else set()))


def pending_ids(agent: Any) -> set[str]:
    return set(pending_interrupts(agent))


@PROPS
@given(st.lists(text_turn, max_size=2), final_turn)
def test_rebuilt_state_equals_the_original(turns: list[Any], final: Any) -> None:
    data = FakeMemoryData()
    first = make_memory_client(FakeModel(_script(turns, final)), data, gateway=default_gateway())
    second = make_memory_client(FakeModel(), data, gateway=default_gateway())

    async def go() -> tuple[Any, Any]:
        await first.begin_session(SID, history="new")
        for text, _r, _g in turns:
            await turn(first, text)
        await turn(first, final[0])
        await second.begin_session(SID, history="reload")
        return first._calls[SID].build.result().agent, second._calls[SID].build.result().agent

    original, rebuilt = asyncio.run(go())
    assert [content_of(x) for x in rebuilt.messages] == [content_of(x) for x in original.messages]
    assert rebuilt._interrupt_state.activated is True
    assert pending_interrupts(rebuilt) == pending_interrupts(original)
    assert rebuilt._interrupt_state.to_dict() == original._interrupt_state.to_dict()


# --------------------------------------------------------------------------- Failed loads


@PROPS
@given(nth=st.integers(0, 2), cached_miss=st.booleans(), script=st.lists(
    st.one_of(words.map(text_msg), inline_names.map(lambda n: tool_msg((n, {})))), min_size=1, max_size=3))
def test_failed_load_yields_no_text_or_tool_call(nth: int, cached_miss: bool, script: list[Any]) -> None:
    data = FakeMemoryData()
    seed = make_memory_client(FakeModel([text_msg("seed")]), data)
    asyncio.run(seed.begin_session(SID, history="new"))
    asyncio.run(turn(seed, "seed"))
    counter = {"n": 0}

    def fail(_kw: dict[str, Any]) -> Exception | None:
        counter["n"] += 1
        return client_error("InternalServerException", op="ListEvents") if counter["n"] > nth else None

    data.fail["list_events"] = fail
    m = FakeModel(script)
    client = make_memory_client(m, data)
    seen: list[Any] = []

    async def go() -> None:
        if not cached_miss:
            with pytest.raises(AgentError):
                await client.begin_session(SID, history="reload")
        async for ev in client.turn(SID, user_text="hello", tool_results=[], system_prompts={"haiku-5-5": "p"}):
            seen.append(ev)

    with pytest.raises(AgentError) as e:
        asyncio.run(go())
    assert (e.value.code, e.value.retryable) == ("history_unavailable", True)
    assert not any(isinstance(ev, (TextDelta, ToolCall)) for ev in seen)
    assert m.calls == []  # the model never ran on a partial history


# --------------------------------------------------------------------------- Hand-offs after the flush

message = st.one_of(
    words.map(lambda t: ("text", t)),
    st.tuples(st.just("gateway"), gateway_names),
    st.tuples(st.just("inline"), st.lists(inline_names, min_size=1, max_size=3)),
)


def _render(spec: Any, n: int) -> Any:
    if spec[0] == "text":
        return text_msg(spec[1])
    if spec[0] == "gateway":
        return tool_msg((f"g{n}", spec[1], {"q": "x"}))
    return tool_msg(*[(f"t{n}_{j}", name, {}) for j, name in enumerate(spec[1])])


@PROPS
@given(st.lists(message, min_size=1, max_size=5))  # at most 6 model calls in one turn
def test_no_handoff_tool_call_before_the_flush(specs: list[Any]) -> None:
    data = FakeMemoryData()
    m = FakeModel([_render(s, i) for i, s in enumerate(specs)])
    client = make_memory_client(m, data, gateway=default_gateway())
    handoffs = 0

    async def go() -> None:
        nonlocal handoffs
        await client.begin_session(SID, history="new")
        results: list[ToolResult] = []
        text: str | None = "start"
        for _ in range(len(specs) + 1):
            calls: list[ToolCall] = []
            async for ev in client.turn(SID, user_text=text, tool_results=results, system_prompts={"haiku-5-5": "p"}):
                if isinstance(ev, ToolCall):
                    calls.append(ev)
                    if ev.name in HANDOFF_TOOLS:
                        handoffs += 1
                        manager = client._calls[SID].build.result().manager
                        assert manager.pending() == 0  # nothing of the invocation is only in memory
                        state = data.newest_agent_state(SID)["_internal_state"]["interrupt_state"]
                        assert state["activated"] is True
                        stored = {i["reason"]["toolUseId"] for i in state["interrupts"].values()}
                        assert ev.tool_use_id in stored
            if not calls:
                break
            results, text = [ToolResult(c.tool_use_id, {"ok": True}) for c in calls], None

    asyncio.run(go())
    reached = [s for s in specs if s[0] == "inline" and any(n in HANDOFF_TOOLS for n in s[1])]
    if specs[0][0] == "inline" and reached and reached[0] is specs[0]:
        assert handoffs > 0  # the first message's hand-off was reached and checked


# --------------------------------------------------------------------------- the cache bound


@PROPS
@given(st.integers(1, 4), st.lists(st.tuples(st.integers(0, 6), st.sampled_from(["new", "cached", "reload"])),
                                   max_size=25))
def test_cache_never_exceeds_its_bound(bound: int, ops: list[tuple[int, str]]) -> None:
    client = StrandsAgentClient(client_settings(cache_max_calls=bound), model_factory=lambda _s, _sid: FakeModel())

    async def go() -> None:
        for n, history in ops:
            await client.begin_session(f"CA{n:032d}", history=history)  # type: ignore[arg-type]
            assert len(client._calls) <= bound

    asyncio.run(go())


# --------------------------------------------------------------------------- Session ownership


class _Process:
    """One simulated bridge process: its ``instance_id``, its client (own cache) and its model."""

    def __init__(self, name: str, data: FakeMemoryData, store: MemoryStore) -> None:
        self.instance_id = name
        self.model = FakeModel(default="ok")
        self.builds = Builds()
        self.client = make_memory_client(self.model, data, builds=self.builds,
                                         window_messages=10_000, cache_max_calls=10)
        self.calls = CallStateRepo(store, now=lambda: 1_000)

    def agent(self) -> Any:
        entry = self.client._calls.get(SID)
        if entry is None or not entry.build.done() or entry.build.exception() is not None:
            return None
        return entry.build.result().agent


class SessionOwnership(RuleBasedStateMachine):
    """Ownership over sessions of one call across two processes: moves, returns, restarts, evictions,
    hand-offs to the front door, and front-door writes, some by older code.

    Model state: which process ran the latest session, and for each process whether its cached agent
    is still current (no other process ran a session since its last one, no restart, no eviction).
    The invariant behind it all: no session or turn ever runs on an agent missing the history
    Memory holds for the call (in particular never on an empty agent)."""

    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.data = FakeMemoryData()
        self.store = MemoryStore()
        callstate.create(self.store, callstate.CallState(call_sid=SID, expires_at=10_000))
        self.generation = [0, 0]
        self.procs = [self._spawn(0), self._spawn(1)]
        self.current: int | None = None  # the process running the active session
        self.current_valid = [False, False]  # the process's cache holds the current history
        self.sessions = 0
        self.owner_id: str | None = None  # the stored agent_owner: the latest session's process
        self.turns = 0
        self.handoffs = 0
        self.session_turns = 0  # turns in the active session

    def _spawn(self, i: int) -> _Process:
        self.generation[i] += 1
        return _Process(f"p{i}-{self.generation[i]}", self.data, self.store)

    def teardown(self) -> None:
        self.loop.close()

    def run(self, coro: Any) -> Any:
        return self.loop.run_until_complete(coro)

    def _state(self) -> callstate.CallState:
        state = callstate.load(self.store, SID)
        assert state is not None
        return state

    def _awaiting_frontdoor(self) -> bool:
        state = self._state()
        return state.pending_tool is not None and state.pending_result is None

    @precondition(lambda self: not self._awaiting_frontdoor())
    @rule(i=st.integers(0, 1))
    def new_session(self, i: int) -> None:
        proc = self.procs[i]
        before = proc.agent()
        reads = self.data.count("ListEvents", SID)
        owner_kept = self.owner_id is not None
        history = proc.calls.claim_agent_owner(SID, proc.instance_id)  # the bridge's decision and write
        if self.sessions == 0:
            assert history == "new"
        elif self.current_valid[i] and owner_kept:
            assert history == "cached"
        else:
            assert history != "new"  # never ``new`` after a session, even with the owner lost
            if not owner_kept:
                assert history == "reload"
        self.run(proc.client.begin_session(SID, history=history))
        after = proc.agent()
        if self.sessions == 0:
            assert self.data.count("ListEvents", SID) == reads  # the first session never reads
        rebuilt = after is not before
        # The cached agent is used iff no other process ran a session since this one's last
        # (and it was neither restarted nor evicted, and the owner wasn't lost); else rebuilt.
        assert rebuilt == (not (self.current_valid[i] and owner_kept))
        # the session's agent holds exactly what Memory holds for the call
        assert [content_of(x) for x in after.messages] == self.data.stored_messages(SID)
        self.sessions += 1
        self.session_turns = 0
        self.owner_id = proc.instance_id
        self.current = i
        self.current_valid = [j == i for j in range(2)]

    @precondition(lambda self: self.current is not None)
    @rule(text=words, handoff=st.booleans())
    def turn(self, text: str, handoff: bool) -> None:
        i = self.current
        assert i is not None
        proc = self.procs[i]
        before = proc.agent()
        stored = self.data.stored_messages(SID)
        self.turns += 1
        user_text: str | None = f"{text} {self.turns}"
        results: list[ToolResult] = []
        state = self._state()
        if self.session_turns == 0 and state.pending_tool is not None and state.pending_result is not None:
            # the session's first hop sends the hand-off's result (the bridge's _claim_result)
            results = [ToolResult(state.pending_tool.tool_use_id, dict(state.pending_result))]
            proc.calls.patch(SID, None, ["pending_tool", "pending_result"])
            user_text = None
        tid = f"bal{self.turns}"
        if handoff:
            proc.model.script.append(tool_msg((tid, "get_balance", {"account": "checking"})))
        n0 = len(proc.model.calls)
        out = self.run(collect(proc.client.turn(SID, user_text=user_text, tool_results=results,
                                                system_prompts={"haiku-5-5": "p"})))
        after = proc.agent()
        assert (after is before) == (before is not None)  # only an evicted agent is rebuilt mid-session
        # the model always sees the whole persisted history, never a stale, partial or empty one
        assert [content_of(x) for x in proc.model.calls[n0].messages][:len(stored)] == stored
        now = self.data.stored_messages(SID)
        assert now[:len(stored)] == stored and len(now) > len(stored)
        self.session_turns += 1
        self.current_valid[i] = True
        if handoff:
            assert out[-1].stop_reason == "tool_use"
            assert [e.tool_use_id for e in out if isinstance(e, ToolCall)] == [tid]
            # the bridge stores the pending tool and hands off: the session ends
            proc.calls.patch(SID, {"pending_tool": {"tool_use_id": tid, "name": "get_balance", "args": {}}})
            self.handoffs += 1
            self.current = None
        else:
            assert out[-1].stop_reason == "end_turn"

    @precondition(lambda self: self.current is None and self.sessions > 0)
    @rule(writer=st.sampled_from(["current", "drops_owner", "pre_fix"]))
    def frontdoor_write(self, writer: str) -> None:
        """The front door between sessions: a fresh relay nonce, and a pending hand-off's result.
        ``current`` code goes through ``callstate``; older code rewrote the whole item without the
        fields it didn't know: ``drops_owner`` without ``agent_owner``; ``pre_fix`` (the incident's
        front door, before the fix) without ``agent_owner`` and ``ai_sessions`` too, and
        only while a hand-off is pending (a pre-fix writer on a plain reconnect is the residual gap
        the design notes: it must never run beside this bridge)."""
        state = self._state()
        changes: dict[str, Any] = {"ws_nonce_hash": "ab" * 32}
        if state.pending_tool is not None and state.pending_result is None:
            changes["pending_result"] = {"status": "ok", "balance": "12.34"}
        if writer == "current":
            callstate.mutate(self.store, SID, lambda _s: changes)
            return
        if writer == "pre_fix" and state.pending_tool is None:
            writer = "drops_owner"
        item = dict(self.store.get(callstate.TABLE, {"call_sid": SID}) or {})
        item.update(changes, version=item["version"] + 1)
        item.pop(callstate.F_AGENT_OWNER, None)
        if writer == "pre_fix":
            item.pop(callstate.F_AI_SESSIONS, None)
        self.store.put(callstate.TABLE, item)  # the legacy whole-item put
        self.owner_id = None

    @rule(i=st.integers(0, 1))
    def restart(self, i: int) -> None:
        self.run(self.procs[i].client.aclose())
        self.procs[i] = self._spawn(i)  # a new instance_id, an empty cache
        self.current_valid[i] = False
        if self.current == i:
            self.current = None  # the WebSocket session died with the process

    @rule(i=st.integers(0, 1))
    def evict(self, i: int) -> None:
        self.procs[i].client._calls.pop(SID, None)
        self.current_valid[i] = False

    @invariant()
    def owner_is_the_latest_session(self) -> None:
        assert self._state().agent_owner == self.owner_id  # None before the first session, or lost


TestSessionOwnership = SessionOwnership.TestCase
TestSessionOwnership.settings = settings(max_examples=40, stateful_step_count=20, deadline=None,
                                         suppress_health_check=[HealthCheck.too_slow])
