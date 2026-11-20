"""End-call and keyed-phone login through the bridge.

The bridge runs with its real catalog, policy, journal and call state, a scripted agent and fake
bank tools. Each property runs on asyncio and on uvloop (production's loop, whose timers can fire
early).

* For generated ``CallView`` / context states and the policy switch,
  ``end_call`` is allowed iff the switch is on and:func:`_pending_step` is ``None``, and
  ``_pending_step`` is ``None`` iff none of ``pending_tool``, ``pending_result``,
  ``pending_handoff_data`` is set (``step_up_target`` and ``escalation`` don't count). Allowed, ``_handle_calls``
  sets the ``caller_done`` hand-off; refused, it sets nothing and the model gets the error.
* For generated model responses (any tools, any arguments, one to three calls each)
  through ``run_turn``, then a second turn on the same session, at most one ``caller_done``
  hand-off is set, and only by a response that was ``end_call`` alone.
* No caller turn (any words, ``goodbye`` included, with any model reply that doesn't
  call ``end_call``), keypad key sequence or silence expiry produces ``caller_done``.
* Once ``login_closed`` is set, no generated sequence of model tool calls (any
  tool, any arguments, any level) starts a step-up.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Coroutine
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank import keypad
from owlbank.callstate import PendingTool
from owlbank.catalog import PromptCatalog
from owlbank.pii import FakePiiRedactor, PiiGuard
from owlbank.policy import Policy
from owlbank.segments import SegmentLog
from owlbank.tools_contract import CALL_IN_PROGRESS, CALLER_DONE, END_CALL, TOOL_DISABLED
from tac_bridge.agent_client import TextDelta, ToolCall, TurnEnd
from tac_bridge.bridge import Bridge, BridgeConfig, CallView, _pending_step
from tac_bridge.silence import Expiry
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

CATALOG = PromptCatalog.from_dict(load_json("prompt_catalog.json"))
SUPPRESS = [HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
VALID: dict[str, dict[str, Any]] = {
    END_CALL: {},
    "escalate": {"reason": "caller_requested_human"},
    "get_balance": {"account": "checking"},
    "make_payment": {"from_account": "checking", "to": "red", "amount": "50.00"},
    "transfer_funds": {"from_account": "checking", "to_account": "savings", "amount": "20.00"},
    "change_address": {"street": "1 Main St", "city": "Springfield", "state": "IL", "postal_code": "62701"},
    "list_recent_transactions": {"account": "checking"},
    "no_such_tool": {},
}
PROTECTED = ("get_balance", "make_payment", "transfer_funds", "change_address", "list_recent_transactions")


def _policy(end_call: bool = True) -> Policy:
    doc = unmasked_policy_doc()
    doc["end_call"] = {"enabled": end_call}
    return Policy.from_dict(doc)


POLICIES = {True: _policy(True), False: _policy(False)}


@pytest.fixture(params=["asyncio", "uvloop"])
def loop_factory(request: pytest.FixtureRequest) -> Callable[[], asyncio.AbstractEventLoop]:
    if request.param == "uvloop":
        factory: Callable[[], asyncio.AbstractEventLoop] = pytest.importorskip("uvloop").new_event_loop
        return factory
    return asyncio.new_event_loop


def run_on(loop_factory: Callable[[], asyncio.AbstractEventLoop], coro: Coroutine[Any, Any, Any]) -> Any:
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        return runner.run(coro)


class _Voice:
    """One TAC-like session; every send recorded as the text it carried."""

    def __init__(self, context: Any) -> None:
        self._context = context
        self.sent: list[str] = []

    def get_conversation_session_by_call_sid(self, call_sid: str) -> Any:
        return self._context if self._context.call_sid == call_sid else None

    def get_websocket(self, conversation_id: str) -> Any:
        return object()

    async def send_response(self, conversation_id: str, gen: Any) -> None:
        self.sent.append(gen if isinstance(gen, str) else "".join([c async for c in gen]))


def rig(script: list[Any], *, level: int = 1, end_call: bool = True, config: BridgeConfig | None = None,
        **state: Any) -> SimpleNamespace:
    store = seeded_store()
    write_call_state(store, verified=level > 0, level=max(level, 1), **state)
    pol = POLICIES[end_call]
    journal = SegmentLog(store, now=lambda: NOW)
    agent = FakeAgent(list(script))
    backend = FakeBackend({name: (lambda name=name, **_: {"tool": name, "data": {}}) for name in PROTECTED})
    bridge = Bridge(store=store, agent=agent, backend=backend, memory=None, catalog=CATALOG,
                    policy=pol, pii=PiiGuard.for_policy(FakePiiRedactor(), pol), journal=journal,
                    sealer=SEALER, config=config or BridgeConfig(), now=lambda: NOW, io=_inline_io)
    ctx = SimpleNamespace(call_sid=CALL_SID, conversation_id="conv_props", metadata={},
                          profile_id=None, profile=None, pending_handoff_data=None)
    return SimpleNamespace(store=store, journal=journal, agent=agent, bridge=bridge, ctx=ctx)


async def turn(r: SimpleNamespace, text: str) -> str:
    return "".join([chunk async for chunk in r.bridge.run_turn(r.ctx, text)])


def handoff(r: SimpleNamespace) -> dict[str, Any] | None:
    data = r.ctx.pending_handoff_data
    return json.loads(data.handoff_data) if data is not None and hasattr(data, "handoff_data") else None


def caller_done_events(r: SimpleNamespace) -> list[dict[str, Any]]:
    return [e for e in r.journal.read(CALL_SID) if e.get("reason") == CALLER_DONE]


# Strategies ---------------------------------------------------------------------------------------

def _call(name: str, valid: bool, target: str) -> ToolCall:
    """``authenticate`` with any target (a protected tool, or not); others valid or invalid."""
    if name == "authenticate":
        return ToolCall("", name, {"tool": target} if valid else {"junk": "x"})
    return ToolCall("", name, dict(VALID[name]) if valid else {"junk": "x"})


tool_calls = st.builds(_call, st.sampled_from([*VALID, "authenticate"]), st.booleans(),
                       st.sampled_from([*PROTECTED, "escalate", "nope"]))
responses = st.lists(tool_calls, min_size=1, max_size=3)
scripts = st.lists(responses, min_size=1, max_size=3)


def numbered(script: list[list[ToolCall]], prefix: str = "a") -> list[list[Any]]:
    """Each response with unique tool_use_ids (``prefix`` apart), then ``TurnEnd("tool_use")``."""
    out: list[list[Any]] = []
    for i, calls in enumerate(script):
        out.append([*(ToolCall(f"tu_{prefix}{i}_{j}", c.name, dict(c.input)) for j, c in enumerate(calls)),
                    TurnEnd("tool_use")])
    return out



views = st.fixed_dictionaries({}, optional={
    "pending_tool": st.just(PendingTool("tu_p", "get_balance", {"account": "checking"})),
    "pending_result": st.just({"status": "completed", "tool": "get_balance"}),
    "step_up_target": st.sampled_from(PROTECTED),
    "escalation": st.just({"reason": "fraud_suspected", "at": NOW}),
})


@settings(max_examples=120, deadline=None, suppress_health_check=SUPPRESS)
@given(fields=views, handoff_set=st.booleans(), switch=st.booleans(), level=st.integers(0, 3),
       login_closed=st.booleans())
def test_end_call_is_allowed_iff_the_switch_is_on_and_nothing_is_pending(
        loop_factory: Callable[[], asyncio.AbstractEventLoop], fields: dict[str, Any], handoff_set: bool,
        switch: bool, level: int, login_closed: bool) -> None:
    r = rig([], end_call=switch)
    marker = object()
    r.ctx.pending_handoff_data = marker if handoff_set else None
    view = CallView(call_sid=CALL_SID, level=level, verified_customer_id="C-1001" if level else None,
                    login_closed=login_closed, **fields)
    # Only these count; step_up_target and escalation (generated too) never do
    nothing_set = not ({"pending_tool", "pending_result"} & set(fields)) and not handoff_set
    assert (_pending_step(view, r.ctx) is None) == nothing_set
    call = ToolCall("tu_end", END_CALL, {})
    refusal = r.bridge._pre_handoff_check(view, call, r.ctx)
    allowed = switch and nothing_set
    assert (refusal is None) == allowed
    if refusal is not None:
        expected = {"error": TOOL_DISABLED, "tool": END_CALL} if not switch else {
            "error": CALL_IN_PROGRESS, "tool": END_CALL, "pending": _pending_step(view, r.ctx)}
        assert refusal.content == expected and refusal.status == "error"
    hop = run_on(loop_factory, r.bridge._handle_calls(r.ctx, view, [call]))
    if allowed:
        assert hop is None
        data = handoff(r)
        assert data is not None and data["reason"] == CALLER_DONE and data["tool"] == END_CALL
        assert len(caller_done_events(r)) == 1
    else:
        assert hop is not None and [res.content for res in hop.tool_results] == [refusal.content]
        assert r.ctx.pending_handoff_data is (marker if handoff_set else None)
        assert caller_done_events(r) == []




@settings(max_examples=150, deadline=None, suppress_health_check=SUPPRESS)
@given(first=scripts, second=scripts, level=st.integers(0, 3))
def test_at_most_one_caller_done_and_only_from_end_call_alone(
        loop_factory: Callable[[], asyncio.AbstractEventLoop], first: list[list[ToolCall]],
        second: list[list[ToolCall]], level: int) -> None:
    script = [*numbered(first), [TextDelta("Anything else?"), TurnEnd("end_turn")],
              *numbered(second, "b"), [TextDelta("Anything else?"), TurnEnd("end_turn")]]
    r = rig(script, level=level)
    after: list[dict[str, Any] | None] = []

    async def go() -> None:
        await turn(r, "That's all, thanks.")
        after.append(handoff(r))
        await turn(r, "Bye.")  # a later turn on the same session (e.g. a barge-in kept the end back)
        after.append(handoff(r))

    run_on(loop_factory, go())
    done = caller_done_events(r)
    assert len(done) <= 1
    events = r.journal.read(CALL_SID)
    called = {e["tool_use_id"] for e in events if e["type"] == "tool_call" and e.get("tool") == END_CALL}
    answered = {e["tool_use_id"] for e in events if e["type"] == "tool_result" and e.get("tool") == END_CALL}
    handed_off = called - answered  # an end_call the bridge took as the hand-off gets no result
    if done:
        assert any(d is not None and d["reason"] == CALLER_DONE and d["tool"] == END_CALL for d in after)
        (winner,) = handed_off
        (resp,) = [resp for resp in script if any(isinstance(c, ToolCall) and c.tool_use_id == winner
                                                  for c in resp)]
        assert [c.name for c in resp if isinstance(c, ToolCall)] == [END_CALL]  # end_call alone
    else:
        assert handed_off == set()
    for resp in script:
        calls = [c for c in resp if isinstance(c, ToolCall)]
        if len(calls) > 1:  # end_call next to any other call: CALL_ALONE, never the hand-off
            ids = {c.tool_use_id for c in calls if c.name == END_CALL}
            assert not ids & handed_off
            for rec in r.agent.calls:
                for res in rec["tool_results"]:
                    if res.tool_use_id in ids:
                        assert res.content["error"] == "CALL_ALONE"



utterances = st.one_of(
    st.sampled_from(["Goodbye.", "That's all, thanks.", "No, thanks, bye!", "Adiós, gracias.", "end call",
                     "hang up", "I'm done.", "end_call", "caller_done"]),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz ,.?!'", min_size=1, max_size=40).filter(str.strip))
non_end_calls = st.lists(tool_calls.filter(lambda c: c.name != END_CALL), min_size=1, max_size=3)
replies = st.lists(st.one_of(st.just("text"), non_end_calls), min_size=1, max_size=3)


@settings(max_examples=100, deadline=None, suppress_health_check=SUPPRESS)
@given(words=utterances, plan=replies, level=st.integers(0, 3))
def test_no_caller_turn_without_the_tool_call_produces_caller_done(
        loop_factory: Callable[[], asyncio.AbstractEventLoop], words: str, plan: list[Any], level: int) -> None:
    script: list[list[Any]] = []
    for i, step in enumerate(plan):
        if step == "text":
            script.append([TextDelta("Goodbye!"), TurnEnd("end_turn")])
        else:
            script.append([*(ToolCall(f"tu_{i}_{j}", c.name, dict(c.input)) for j, c in enumerate(step)),
                           TurnEnd("tool_use")])
    r = rig(script, level=level)

    async def go() -> None:
        for _ in plan:
            if r.ctx.pending_handoff_data is not None:
                break
            await turn(r, words)

    run_on(loop_factory, go())
    assert caller_done_events(r) == []
    data = handoff(r)
    assert data is None or data["reason"] != CALLER_DONE


KEYS = list("0123456789*#")


@settings(max_examples=80, deadline=None, suppress_health_check=SUPPRESS)
@given(keys=st.lists(st.tuples(st.sampled_from(KEYS), st.sampled_from([0.0, 0.2, keypad.WINDOW_S + 0.01, 3.0])),
                     min_size=1, max_size=10))
def test_no_keypad_sequence_produces_caller_done(
        loop_factory: Callable[[], asyncio.AbstractEventLoop], keys: list[tuple[str, float]]) -> None:
    r = rig([], config=BridgeConfig(ai_keypad=True))
    r.bridge.voice = _Voice(r.ctx)
    r.bridge.session_start(CALL_SID)
    speech = r.bridge._speech[CALL_SID]
    speech.language, speech.text, speech.label = "en-US", "Anything else?", "agent"

    async def go() -> None:
        buffer = r.bridge._keypad.setdefault(CALL_SID, keypad.KeypadBuffer())
        at = time.monotonic() - 1000.0
        for key, gap in keys:
            at += gap
            buffer.on_digit(key, at)
        await r.bridge._keypad_due(CALL_SID, buffer)

    run_on(loop_factory, go())
    assert caller_done_events(r) == []
    data = handoff(r)
    assert data is None or data["reason"] != CALLER_DONE


@settings(max_examples=20, deadline=None, suppress_health_check=SUPPRESS)
@given(action=st.sampled_from(["hangup", "escalate"]), repeat=st.integers(0, 3))
def test_no_silence_expiry_produces_caller_done(
        loop_factory: Callable[[], asyncio.AbstractEventLoop], action: str, repeat: int) -> None:
    r = rig([])
    r.bridge.voice = _Voice(r.ctx)
    r.bridge.session_start(CALL_SID)
    speech = r.bridge._speech[CALL_SID]
    run_on(loop_factory, r.bridge._silence_action(CALL_SID, speech, r.ctx,
                                                  Expiry("action", repeat, action=action)))
    data = handoff(r)
    assert data is not None and data["reason"] != CALLER_DONE and data.get("tool") != END_CALL
    assert caller_done_events(r) == []




@settings(max_examples=150, deadline=None, suppress_health_check=SUPPRESS)
@given(script=scripts, level=st.integers(0, 3))
def test_once_the_login_is_closed_no_tool_call_sequence_starts_a_step_up(
        loop_factory: Callable[[], asyncio.AbstractEventLoop], script: list[list[ToolCall]], level: int) -> None:
    r = rig([*numbered(script), [TextDelta("What else?"), TurnEnd("end_turn")]], level=level,
            login_closed=True)

    async def go() -> None:
        for words in ("Let me verify.", "Try again please."):
            if r.ctx.pending_handoff_data is not None:
                break
            await turn(r, words)

    run_on(loop_factory, go())
    data = handoff(r)
    assert data is None or data["reason"] != "step-up"
    state = r.store.get("calls", {"call_sid": CALL_SID}) or {}
    assert not state.get("step_up_target")
    pending = state.get("pending_tool") or {}
    assert pending.get("name") != "authenticate"
    assert not [e for e in r.journal.read(CALL_SID) if e["type"] == "handoff" and e.get("reason") == "step-up"]
    for rec in r.agent.calls:
        for res in rec["tool_results"]:
            content = res.content if isinstance(res.content, dict) else {}
            if content.get("tool") == "authenticate" and content.get("error") not in ("CALL_ALONE",
                                                                                     "ONE_HANDOFF_AT_A_TIME"):
                assert content["error"] == "VERIFICATION_CLOSED"
