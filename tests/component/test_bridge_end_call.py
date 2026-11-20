"""End-call and the closed login: the bridge's part.

* An allowed ``end_call`` hands off with HandoffData ``{reason: caller_done, tool: end_call}``
  (the silence hang-up's shape: no pending tool, no escalation), one journal ``handoff`` event and
  an ``end_call_requested`` log line, with no second model call.
* It is refused with ``CALL_IN_PROGRESS`` and the pending step for each pending state, with
  ``TOOL_DISABLED`` when the policy switch is off (and then it is not offered to the model), and
  with ``CALL_ALONE`` next to any other call.
* Once the login is closed, ``authenticate`` is refused with ``VERIFICATION_CLOSED`` and a
  front-door tool below its level gets its level refusal with ``"verification": "closed"`` instead
  of the step-up shortcut.

Each bridge test runs on asyncio and on uvloop (production's loop, whose timers can fire early).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from types import SimpleNamespace
from typing import Any

import pytest

from owlbank.callstate import PendingTool
from owlbank.catalog import PromptCatalog
from owlbank.pii import FakePiiRedactor, PiiGuard
from owlbank.policy import Policy
from owlbank.segments import SegmentLog
from owlbank.tools_contract import (
    AUTH_LEVEL_INSUFFICIENT,
    CALL_IN_PROGRESS,
    CALLER_DONE,
    END_CALL,
    TOOL_DISABLED,
    VERIFICATION_CLOSED,
)
from tac_bridge.agent_client import TextDelta, ToolCall, TurnEnd
from tac_bridge.agent_client_strands import StrandsAgentClient
from tac_bridge.bridge import Bridge, CallView, _pending_step
from tests.component.masking_fixtures import unmasked_policy_doc
from tests.component.test_bridge_fakes import (
    CALL_SID,
    NOW,
    SEALER,
    FakeAgent,
    FakeBackend,
    _inline_io,
    load_json,
    seeded_store,
    write_call_state,
)
from tests.logutil import lines
from tests.strands_fakes import FakeModel, collect, light_memory_client, settings, text_msg

CATALOG = PromptCatalog.from_dict(load_json("prompt_catalog.json"))
BRIDGE_LOG = "tac_bridge.bridge"
PAY = {"from_account": "checking", "to": "red", "amount": "50.00"}
DONE = [TextDelta("Anything else?"), TurnEnd("end_turn")]


def policy(*, end_call: bool = True) -> Policy:
    doc = unmasked_policy_doc()
    doc["end_call"] = {"enabled": end_call}
    return Policy.from_dict(doc)


@pytest.fixture(params=["asyncio", "uvloop"])
def loop_factory(request: pytest.FixtureRequest) -> Callable[[], asyncio.AbstractEventLoop]:
    if request.param == "uvloop":
        factory: Callable[[], asyncio.AbstractEventLoop] = pytest.importorskip("uvloop").new_event_loop
        return factory
    return asyncio.new_event_loop


def run_on(loop_factory: Callable[[], asyncio.AbstractEventLoop], coro: Coroutine[Any, Any, Any]) -> Any:
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        return runner.run(coro)


def rig(script: list[Any], *, level: int = 1, end_call: bool = True, **state: Any) -> SimpleNamespace:
    store = seeded_store()
    write_call_state(store, verified=level > 0, level=max(level, 1), **state)
    pol = policy(end_call=end_call)
    journal = SegmentLog(store, now=lambda: NOW)
    agent = FakeAgent(list(script))
    backend = FakeBackend({"list_recent_transactions": {"tool": "list_recent_transactions", "data": {}}})
    bridge = Bridge(store=store, agent=agent, backend=backend, memory=None, catalog=CATALOG,
                    policy=pol, pii=PiiGuard.for_policy(FakePiiRedactor(), pol), journal=journal,
                    sealer=SEALER, now=lambda: NOW, io=_inline_io)
    ctx = SimpleNamespace(call_sid=CALL_SID, conversation_id="conv_end", metadata={},
                          profile_id=None, profile=None, pending_handoff_data=None)
    return SimpleNamespace(store=store, journal=journal, agent=agent, backend=backend, bridge=bridge, ctx=ctx)


async def turn(r: SimpleNamespace, text: str = "That's all, thanks.") -> str:
    return "".join([chunk async for chunk in r.bridge.run_turn(r.ctx, text)])


def handoff(r: SimpleNamespace) -> dict[str, Any] | None:
    data = r.ctx.pending_handoff_data
    return json.loads(data.handoff_data) if data is not None else None


def call_state(r: SimpleNamespace) -> dict[str, Any]:
    return r.store.get("calls", {"call_sid": CALL_SID}) or {}


def results(r: SimpleNamespace, hop: int = 1) -> list[Any]:
    return [res.content for res in r.agent.calls[hop]["tool_results"]]


def end_call(tu: str = "tu_end") -> list[Any]:
    return [ToolCall(tu, END_CALL, {}), TurnEnd("tool_use")]


# --------------------------------------------------------------------------- allowed


def test_an_allowed_end_call_hands_off_caller_done_like_the_silence_hang_up(
        loop_factory: Callable[[], asyncio.AbstractEventLoop], caplog: pytest.LogCaptureFixture) -> None:
    r = rig([end_call()])
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        spoken = run_on(loop_factory, turn(r))
    data = handoff(r)
    assert data is not None
    assert {k: data[k] for k in ("reason", "tool")} == {"reason": CALLER_DONE, "tool": END_CALL}
    assert data["conversationId"] == "conv_end" and data["journal"]
    assert set(data) <= {"reason", "tool", "journal", "conversationId", "profileId"}
    assert len(r.agent.calls) == 1  # no second model call: the front door says the goodbye
    assert spoken == ""
    handoffs = [e for e in r.journal.read(CALL_SID) if e["type"] == "handoff"]
    assert len(handoffs) == 1
    assert {k: handoffs[0][k] for k in ("tool", "reason", "outcome")} == {
        "tool": END_CALL, "reason": CALLER_DONE, "outcome": "requested"}
    state = call_state(r)
    assert state.get("pending_tool") is None and state.get("escalation") is None
    assert state.get("step_up_target") is None
    events = [e["event"] for e in lines(caplog.records, BRIDGE_LOG)]
    assert events.count("end_call_requested") == 1
    assert "escalation_requested" not in events


# --------------------------------------------------------------------------- refused


def _view(**fields: Any) -> CallView:
    return CallView(call_sid=CALL_SID, level=1, verified_customer_id="C-1001", **fields)


#: case -> (view fields, pending_handoff_data set, expected ``pending``)
PENDING_STATES: dict[str, tuple[dict[str, Any], bool, str]] = {
    "tool": ({"pending_tool": PendingTool("tu_bal", "get_balance", {"account": "checking"})}, False, "tool"),
    # a step-up in progress: the model's call is the pending tool, the target is set
    "step_up_in_progress": ({"pending_tool": PendingTool("tu_pay", "make_payment", PAY),
                             "step_up_target": "make_payment"}, False, "tool"),
    "result": ({"pending_result": {"status": "completed", "tool": "get_balance"}}, False, "result"),
    "handoff": ({}, True, "handoff"),
}


@pytest.mark.parametrize("case", sorted(PENDING_STATES))
def test_end_call_is_refused_while_a_step_is_pending(
        case: str, loop_factory: Callable[[], asyncio.AbstractEventLoop]) -> None:
    fields, handoff_set, pending = PENDING_STATES[case]
    r = rig([])
    view = _view(**fields)
    marker = object()
    r.ctx.pending_handoff_data = marker if handoff_set else None
    assert _pending_step(view, r.ctx) == pending
    hop = run_on(loop_factory, r.bridge._handle_calls(r.ctx, view, [ToolCall("tu_end", END_CALL, {})]))
    assert hop is not None
    (res,) = hop.tool_results
    assert res.status == "error"
    assert res.content == {"error": CALL_IN_PROGRESS, "tool": END_CALL, "pending": pending}
    assert r.ctx.pending_handoff_data is (marker if handoff_set else None)  # nothing new set
    assert not [e for e in r.journal.read(CALL_SID) if e.get("reason") == CALLER_DONE]


#: case -> call state the front door left for the AI: a finished tool's result, or an interrupt's
RESUMED_STATES: dict[str, dict[str, Any]] = {
    "result": {"pending_tool": PendingTool("tu_bal", "get_balance", {"account": "checking"}),
               "pending_result": {"status": "completed", "tool": "get_balance", "data": {}}},
    "interrupted": {"pending_result": {"status": "interrupted", "next": "handle_utterance"}},
}


@pytest.mark.parametrize("case", sorted(RESUMED_STATES))
def test_end_call_right_after_a_resumed_result_hands_off(
        case: str, loop_factory: Callable[[], asyncio.AbstractEventLoop],
        caplog: pytest.LogCaptureFixture) -> None:
    """The first hop claims the pending result (clears it in the store), so the
    model's ``end_call`` in the same turn ("anything else?" - "No, that's all") isn't pending."""
    # a result is recorded in its own agent call first, then the caller's words go
    script = ([[]] if "pending_tool" in RESUMED_STATES[case] else []) + [end_call(), DONE]
    r = rig(script, **RESUMED_STATES[case])
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        run_on(loop_factory, turn(r, "No, that's all, thanks."))
    data = handoff(r)
    assert data is not None and data["reason"] == CALLER_DONE and data["tool"] == END_CALL
    events = [e["event"] for e in lines(caplog.records, BRIDGE_LOG)]
    assert "end_call_refused" not in events


def test_a_refusal_is_logged_once(loop_factory: Callable[[], asyncio.AbstractEventLoop],
                                  caplog: pytest.LogCaptureFixture) -> None:
    """The masking planner runs the same check; only the real refusal logs."""
    r = rig([end_call(), DONE], end_call=False)
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        run_on(loop_factory, turn(r))
    events = [e["event"] for e in lines(caplog.records, BRIDGE_LOG)]
    assert events.count("end_call_refused") == 1


@pytest.mark.parametrize("fields", [{"step_up_target": "make_payment"},
                                    {"escalation": {"reason": "fraud_suspected", "at": NOW}},
                                    {"step_up_target": "make_payment",
                                     "escalation": {"reason": "fraud_suspected", "at": NOW}}])
def test_a_finished_step_up_or_an_earlier_escalation_does_not_block_end_call(
        fields: dict[str, Any], loop_factory: Callable[[], asyncio.AbstractEventLoop]) -> None:
    """``step_up_target`` and ``escalation`` stay in call state after the
    step is over, so they don't count as pending; only a pending tool or result does."""
    r = rig([end_call()], **fields)
    run_on(loop_factory, turn(r))
    data = handoff(r)
    assert data is not None and data["reason"] == CALLER_DONE and data["tool"] == END_CALL
    assert len(r.agent.calls) == 1


def test_with_the_switch_off_end_call_is_refused_tool_disabled(
        loop_factory: Callable[[], asyncio.AbstractEventLoop]) -> None:
    r = rig([end_call(), DONE], end_call=False)
    run_on(loop_factory, turn(r))
    assert handoff(r) is None
    assert results(r) == [{"error": TOOL_DISABLED, "tool": END_CALL}]


@pytest.mark.parametrize("other", [ToolCall("tu_tx", "list_recent_transactions", {"account": "checking"}),
                                   ToolCall("tu_esc", "escalate", {"reason": "caller_requested_human"}),
                                   ToolCall("tu_end2", END_CALL, {})])
def test_end_call_with_any_other_call_is_call_alone(
        other: ToolCall, loop_factory: Callable[[], asyncio.AbstractEventLoop]) -> None:
    r = rig([[ToolCall("tu_end", END_CALL, {}), other, TurnEnd("tool_use")], DONE])
    run_on(loop_factory, turn(r))
    assert handoff(r) is None
    assert {(c["tool"], c.get("error")) for c in results(r) if c.get("tool") == END_CALL} == {
        (END_CALL, "CALL_ALONE")}
    assert not [e for e in r.journal.read(CALL_SID) if e["type"] == "handoff"]


# --------------------------------------------------------------------------- the switch (agent client)


def _offered(client: StrandsAgentClient, model: FakeModel, sid: str) -> set[str]:
    async def go() -> None:
        await collect(client.turn(sid, user_text="hi", tool_results=[], system_prompts={"haiku-5-5": "BASE"}))

    asyncio.run(go())
    return {t["name"] for t in model.calls[-1].tool_specs}


def test_end_call_is_offered_only_while_the_switch_is_on_read_at_each_agent_build(
        monkeypatch: pytest.MonkeyPatch) -> None:
    light_memory_client(monkeypatch)
    model = FakeModel([text_msg("one"), text_msg("two"), text_msg("three")])
    client = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: model)
    assert END_CALL in _offered(client, model, "CA" + "1" * 32)  # default: on
    switch = {"on": False}
    client.use_end_call_switch(lambda: switch["on"])
    off = _offered(client, model, "CA" + "2" * 32)
    assert END_CALL not in off and "escalate" in off
    switch["on"] = True
    assert END_CALL in _offered(client, model, "CA" + "3" * 32)  # a new call's agent


@pytest.mark.parametrize("source", [lambda: (_ for _ in ()).throw(RuntimeError("store down")),
                                    lambda: "no"])
def test_a_failing_or_malformed_switch_source_offers_end_call(source: Any,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    light_memory_client(monkeypatch)
    model = FakeModel([text_msg("hi")])
    client = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: model)
    client.use_end_call_switch(source)
    assert END_CALL in _offered(client, model, "CA" + "4" * 32)


# --------------------------------------------------------------------------- login closed


def test_after_a_closed_login_authenticate_is_refused_verification_closed(
        loop_factory: Callable[[], asyncio.AbstractEventLoop]) -> None:
    r = rig([[ToolCall("tu_auth", "authenticate", {"tool": "make_payment"}), TurnEnd("tool_use")], DONE],
            level=0, login_closed=True)
    run_on(loop_factory, turn(r, "Let me try again."))
    assert handoff(r) is None
    assert results(r) == [{"error": VERIFICATION_CLOSED, "tool": "authenticate"}]
    assert call_state(r).get("step_up_target") is None


@pytest.mark.parametrize("name,args", [("make_payment", PAY), ("get_balance", {"account": "checking"})])
def test_after_a_closed_login_a_protected_tool_gets_the_level_refusal_not_a_step_up(
        name: str, args: dict[str, Any], loop_factory: Callable[[], asyncio.AbstractEventLoop]) -> None:
    r = rig([[ToolCall("tu_x", name, dict(args)), TurnEnd("tool_use")], DONE], level=0, login_closed=True)
    run_on(loop_factory, turn(r, "Pay my card."))
    assert handoff(r) is None
    assert results(r) == [{"error": AUTH_LEVEL_INSUFFICIENT, "tool": name, "verification": "closed"}]
    state = call_state(r)
    assert state.get("step_up_target") is None and state.get("pending_tool") is None
    assert r.backend.calls == []


@pytest.mark.parametrize("source", ["no_token", "bank_refuses"])
@pytest.mark.parametrize("closed", [True, False])
def test_a_bank_tool_level_refusal_says_verification_closed_after_a_closed_login(
        closed: bool, source: str, loop_factory: Callable[[], asyncio.AbstractEventLoop],
        caplog: pytest.LogCaptureFixture) -> None:
    """An inline bank tool refused for the
    caller's level (no token, or the bank's own refusal) carries the same marker, so the model
    doesn't try ``authenticate`` first."""
    name = "list_recent_transactions"
    state: dict[str, Any] = {"token": None} if source == "no_token" else {}
    r = rig([[ToolCall("tu_x", name, {"account": "checking"}), TurnEnd("tool_use")], DONE],
            level=0, login_closed=closed, **state)
    r.backend.responses[name] = {"error": AUTH_LEVEL_INSUFFICIENT, "tool": name}
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        run_on(loop_factory, turn(r, "What did I spend last week?"))
    expected: dict[str, Any] = {"error": AUTH_LEVEL_INSUFFICIENT, "tool": name}
    if closed:
        expected["verification"] = "closed"
    assert results(r) == [expected]
    events = [e["event"] for e in lines(caplog.records, BRIDGE_LOG)]
    assert events.count("verification_closed") == (1 if closed else 0)


@pytest.mark.parametrize("name,args", [("make_payment", PAY), ("get_balance", {"account": "checking"})])
def test_without_a_closed_login_the_step_up_shortcut_is_unchanged(
        name: str, args: dict[str, Any], loop_factory: Callable[[], asyncio.AbstractEventLoop]) -> None:
    r = rig([[ToolCall("tu_x", name, dict(args)), TurnEnd("tool_use")]], level=0)
    run_on(loop_factory, turn(r, "Pay my card."))
    data = handoff(r)
    assert data is not None and data["reason"] == "step-up" and data["tool"] == name
