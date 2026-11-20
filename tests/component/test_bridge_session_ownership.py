"""Strands sprint: the bridge's session ownership.

At each ConversationRelay ``setup`` the bridge decides ``history`` from the call's ``agent_owner``
(none → ``new``, this process → ``cached``, another → ``reload``), records ``agent_owner =
instance_id`` and calls the agent's ``begin_session(call_sid, history=...)`` in the background. The
session's first agent turn waits for both. Real TAC and the ConversationRelay simulator; the
agent is the real Strands client playing the scripts (``tests.strands_parity``), or a gated
``FakeAgent`` for the wait itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from owlbank import callstate as cs
from owlbank import ws_nonce
from tac_bridge import bridge as bridge_mod
from tac_bridge.agent_client_strands import StrandsAgentClient
from tac_bridge.bridge import CallStateRepo, had_ai_session, session_history
from tests.component.test_bridge_fakes import (
    CALL_SID,
    FakeAgent,
    end_frame,
    make_rig,
    rearm_nonce,
    relay_nonce,
    run,
    say,
    seeded_store,
    spoken,
    wait_for,
    write_call_state,
)
from tests.logutil import lines
from tests.strands_fakes import (
    FakeMemoryData,
    FakeModel,
    default_gateway,
    light_memory_client,
    memory_factory,
    tool_msg,
)
from tests.strands_fakes import settings as strands_settings


@dataclass
class GatedAgent(FakeAgent):
    """``begin_session`` waits for ``gate``; every step is recorded with the stored owner."""

    gate: asyncio.Event | None = None
    order: list[tuple[str, Any]] = field(default_factory=list)
    store: Any = None

    def _owner(self) -> Any:
        return (self.store.get("calls", {"call_sid": CALL_SID}) or {}).get("agent_owner")

    async def begin_session(self, session_id: str, *, history: str = "reload") -> None:
        self.order.append(("begin", history))
        if self.gate is not None:
            await self.gate.wait()
        await super().begin_session(session_id, history=history)
        self.order.append(("begun", history))

    def turn(self, session_id: str, **kw: Any) -> Any:
        self.order.append(("turn", self._owner()))
        return super().turn(session_id, **kw)


def test_session_history_decision():
    assert session_history(None, "p1") == "new"
    assert session_history("", "p1") == "new"
    assert session_history("p1", "p1") == "cached"
    assert session_history("p2", "p1") == "reload"


def test_claim_writes_the_owner_once():
    store = seeded_store()
    write_call_state(store)
    repo = CallStateRepo(store, now=lambda: 1_800_000_000)
    v0 = cs.load(store, CALL_SID).version  # type: ignore[union-attr]
    assert repo.claim_agent_owner(CALL_SID, "p1") == "new"
    assert repo.claim_agent_owner(CALL_SID, "p1") == "cached"
    state = cs.load(store, CALL_SID)
    assert state is not None and state.agent_owner == "p1"
    assert state.version == v0 + 1  # no write when this process already is the owner
    assert repo.claim_agent_owner(CALL_SID, "p2") == "reload"
    assert cs.load(store, CALL_SID).agent_owner == "p2"  # type: ignore[union-attr]


@pytest.mark.protocol_fake("a gated FakeAgent replaces the rig's agent: the bridge's wait itself")
def test_first_turn_waits_for_the_owner_write_and_begin_session():
    rig = make_rig()
    gate = asyncio.Event()
    agent = GatedAgent([say("Hello.")], gate=gate, store=rig.store)
    rig.bridge._agent = agent
    write_call_state(rig.store)

    async def go() -> str:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt_nowait("Hi")
        await wait_for(lambda: bool(agent.order))
        for _ in range(50):
            await asyncio.sleep(0)
        assert agent.calls == []  # the turn waits while begin_session runs
        gate.set()
        frames = await sim.wait_reply()
        await sim.hangup()
        return spoken(frames)

    assert run(go()) == "Hello."
    owner = rig.bridge.instance_id
    assert agent.order == [("begin", "new"), ("begun", "new"), ("turn", owner)]
    assert rig.call()["agent_owner"] == owner


def test_sessions_across_two_processes():
    store = seeded_store()
    write_call_state(store)
    a = make_rig(store=store)
    b = make_rig(store=store)
    assert a.bridge.instance_id != b.bridge.instance_id

    async def session(rig: Any) -> None:
        rearm_nonce(store)
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("Hello")
        await sim.hangup()

    async def go() -> None:
        await session(a)  # the first AI session: no history to read
        await session(a)  # the same process again: its cache is current
        await session(b)  # moved: rebuild from Memory
        await session(a)  # back: b ran a session since, so a's cache is stale
        await session(a)

    run(go())
    assert a.agent.histories == ["new", "cached", "reload", "cached"]
    assert b.agent.histories == ["reload"]
    assert store.get("calls", {"call_sid": CALL_SID})["agent_owner"] == a.bridge.instance_id


def test_failed_owner_write_reloads_and_the_first_turn_retries_it():
    rig = make_rig(script=[say("Hi there.")], )
    write_call_state(rig.store)
    real = rig.bridge._calls.claim_agent_owner
    attempts: list[str] = []

    def flaky(call_sid: str, instance_id: str) -> str:
        attempts.append(call_sid)
        if len(attempts) == 1:
            raise cs.VersionConflict("busy")
        return real(call_sid, instance_id)

    rig.bridge._calls.claim_agent_owner = flaky  # type: ignore[method-assign]

    async def go() -> str:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("Hello")
        await sim.hangup()
        return spoken(frames)

    assert run(go()) == "Hi there."
    assert rig.agent.histories == ["reload"]  # unknown owner: always safe
    assert len(attempts) == 2 and rig.call()["agent_owner"] == rig.bridge.instance_id


def test_owner_write_that_keeps_failing_fails_the_turn_retryably():
    rig = make_rig(script=[say("never")], )
    write_call_state(rig.store)

    def broken(call_sid: str, instance_id: str) -> str:
        raise cs.VersionConflict("busy")

    rig.bridge._calls.claim_agent_owner = broken  # type: ignore[method-assign]

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("Hello")
        await sim.hangup()

    run(go())
    assert rig.agent.calls == []  # never ran a turn without the owner recorded
    errors = [e for e in rig.events() if e.get("type") == "ai_turn" and e.get("outcome") == "error"]
    assert [e["error"] for e in errors] == ["session_owner_unavailable", "session_owner_unavailable"]


# --------------------------------------------------------------------------- the lost owner


def test_the_new_rule_needs_no_sign_of_an_earlier_ai_session():
    """No owner is ``new`` only when call state shows no earlier AI session."""
    base = cs.CallState(call_sid=CALL_SID, expires_at=10_000)
    assert not had_ai_session(base)
    signs = [{"ai_sessions": 1},
             {"pending_tool": cs.PendingTool("tu_1", "get_balance")},
             {"pending_result": {"status": "completed"}},
             {"prefetched": {"tool": "get_balance"}},
             {"step_up_target": "make_payment"}]
    for sign in signs:
        state = replace(base, **sign)
        assert had_ai_session(state), sign
        assert session_history(None, "p1", earlier_session=True) == "reload"
    assert session_history("p1", "p1", earlier_session=True) == "cached"


def test_claim_counts_sessions_and_a_lost_owner_reloads():
    store = seeded_store()
    write_call_state(store)
    repo = CallStateRepo(store, now=lambda: 1_800_000_000)
    assert repo.claim_agent_owner(CALL_SID, "p1") == "new"
    assert repo.claim_agent_owner(CALL_SID, "p1") == "cached"
    assert repo.claim_agent_owner(CALL_SID, "p2") == "reload"
    assert cs.load(store, CALL_SID).ai_sessions == 2  # type: ignore[union-attr]
    # an older writer erases the owner (but, with call state keeping unknown fields, not the count)
    item = dict(store.get("calls", {"call_sid": CALL_SID}))
    item.pop("agent_owner")
    store.put("calls", item)
    assert repo.claim_agent_owner(CALL_SID, "p1") == "reload"
    # the incident's pre-fix writer erased both, but the hand-off in flight still shows
    cs.mutate(store, CALL_SID, lambda _s: {"pending_tool": {"tool_use_id": "tu_1", "name": "get_balance"},
                                           "pending_result": {"status": "completed"}})
    item = dict(store.get("calls", {"call_sid": CALL_SID}))
    item.pop("agent_owner")
    item.pop("ai_sessions")
    store.put("calls", item)
    assert repo.claim_agent_owner(CALL_SID, "p1") == "reload"


BAL = {"tool_use_id": "tu_bal", "name": "get_balance", "args": {"account": "checking"}}
BAL_RESULT = {"status": "completed", "tool": "get_balance", "next": "await_caller",
              "spoken": [{"template": "balance.read", "text": "Your checking balance is $1,234.56."}],
              "data": {"account": "checking", "last4": "4821", "available": "1234.56"}}


def _legacy_frontdoor_write(store: Any) -> None:
    """The incident's front door (pre-strands code, before call state kept unknown attributes):
    it answers the hand-off and rewrites the whole item without the fields it doesn't know."""
    item = dict(store.get("calls", {"call_sid": CALL_SID}))
    item.pop("agent_owner", None)
    item.pop("ai_sessions", None)
    item.update(pending_result=BAL_RESULT, version=item["version"] + 1,
                ws_nonce_hash=ws_nonce.nonce_hash(relay_nonce()))
    store.put("calls", item)


def _incident(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
              ) -> tuple[Any, FakeModel, list[dict[str, Any]]]:
    light_memory_client(monkeypatch)
    rig = make_rig()
    data = FakeMemoryData()
    model = FakeModel([tool_msg(("tu_bal", "get_balance", {"account": "checking"}))], default="<silent/>")
    client = StrandsAgentClient(strands_settings(memory_id="mem-1"), model_factory=lambda _s, _sid: model,
                                gateway_tools=default_gateway(), session_manager_factory=memory_factory(data))
    rig.bridge._agent = client
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        first = await sim.prompt("What's my checking balance?")  # session 1: new
        assert end_frame(first)["data"]["tool"] == "get_balance"  # type: ignore[index]
        await sim.hangup()
        assert rig.call()["pending_tool"]["tool_use_id"] == "tu_bal"
        assert data.newest_agent_state(CALL_SID)["_internal_state"]["interrupt_state"]["activated"]
        _legacy_frontdoor_write(rig.store)  # the owner field is erased
        assert "agent_owner" not in rig.call()
        sim = rig.sim()
        await sim.connect()  # session 2
        frames = await sim.prompt("Thanks")
        await sim.hangup()
        return frames

    with caplog.at_level(logging.INFO, logger="tac_bridge.agent_client_strands"):
        frames = run(go())
    return rig, model, frames


def _model_got_the_balance(model: FakeModel) -> None:
    resumed = next(call.messages for call in model.calls
                   if any("toolResult" in c for m in call.messages for c in m["content"]))
    results = [c["toolResult"] for m in resumed for c in m["content"] if "toolResult" in c]
    assert [r["toolUseId"] for r in results] == ["tu_bal"]
    assert "1234.56" in json.dumps(results[0], default=str)
    uses = [c["toolUse"]["name"] for m in resumed for c in m["content"] if "toolUse" in c]
    assert uses == ["get_balance"]  # the history (session 1's hand-off) came back with it


def _turn_ok(rig: Any, frames: list[dict[str, Any]]) -> None:
    errors = [e for e in rig.events() if e.get("type") == "ai_turn" and e.get("outcome") == "error"]
    assert errors == []
    end = end_frame(frames)
    assert end is None or end["data"].get("reason") != "escalate"


def test_incident_lost_owner_reloads_at_session_start(caplog: pytest.LogCaptureFixture,
                                                      monkeypatch: pytest.MonkeyPatch):
    """CA900e…9439, with the fixes: the hand-off in flight shows an earlier AI session, so session
    2 reloads from Memory and the get_balance result reaches the model."""
    rig, model, frames = _incident(caplog, monkeypatch)
    _turn_ok(rig, frames)
    _model_got_the_balance(model)
    reloads = [f for f in lines(caplog.records, "tac_bridge.agent_client_strands")
               if f.get("event") == "agent_history_reload"]
    assert [f["reason"] for f in reloads] == ["owner_changed"]


def test_incident_safety_net_recovers_a_wrong_new(caplog: pytest.LogCaptureFixture,
                                                 monkeypatch: pytest.MonkeyPatch):
    """The same sequence with the session decision still ``new`` (as before fix 2, or if the signs
    were cleared first): the unknown tool use reloads the agent once and the turn succeeds."""
    monkeypatch.setattr(bridge_mod, "had_ai_session", lambda _state: False)
    rig, model, frames = _incident(caplog, monkeypatch)
    _turn_ok(rig, frames)
    _model_got_the_balance(model)
    reloads = [f for f in lines(caplog.records, "tac_bridge.agent_client_strands")
               if f.get("event") == "agent_history_reload"]
    assert [f["reason"] for f in reloads] == ["unknown_tool_use_recovery"]
