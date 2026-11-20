"""The deterministic step-up shortcut.

Bridge: a hand-off tool the front door runs (``get_balance``, or a deterministic-confirm tool)
refused only for the caller's level hands off for step-up at once, with the model's own call as
``pending_tool`` (no AUTH_LEVEL_INSUFFICIENT round trip). Front door: after a verified step-up
for such a pending tool it goes straight to the tool's scripted step (the confirmation
read-back, or the balance) with no return to the AI, and the final envelope answers the
model's ORIGINAL call with everything the caller heard.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from owlbank import callstate
from owlbank.callstate import PendingTool
from owlbank.envelope import CANCELLED_BY_CALLER, ResultEnvelope
from owlbank.slots import FakeResolver, ResolverOutput
from owlbank.tools_contract import AUTH_LEVEL_INSUFFICIENT
from tests.component.masking_fixtures import masked_policy_doc
from tests.component.test_bridge_fakes import (
    CALL_TOKEN,
    FakeBackend,
    end_frame,
    make_rig,
    run,
    say,
    spoken,
    tool,
    write_call_state,
)
from tests.component.test_bridge_masking import collect, context, direct, en
from tests.logutil import lines
from tests.component.test_frontdoor_support import (
    OTP,
    TEST_FACTORS,
    THOMAS,
    Harness,
    build,
    fd,  # noqa: F401 - fixture
    greeting,
    relay,
    says,
    sid,
)

PAY = {"from_account": "checking", "to": "purple", "amount": "50.00"}
READBACK = ("To confirm, you'd like to pay fifty dollars toward your OwlBank Purple Card from "
            "your checking account. Is that right? Say yes or no, or press 1 for yes and 2 for no.")
SUCCESS = "Thanks, you're verified."
INTENT_SWITCH = ResolverOutput(kind="intent_switch")
S = TEST_FACTORS["C-1001"]["ssn4"]
Z = TEST_FACTORS["C-1001"]["zip"]
XFER = {"from_account": "checking", "to_account": "savings", "amount": "20.00"}
MAKE_PAYMENT_PHRASE = en("mask.make_payment")
TRANSFER_PHRASE = en("mask.transfer_funds")
BRIDGE_LOG = "tac_bridge.bridge"


# =========================================================================== bridge


def _one_prompt(rig: Any, text: str) -> list[dict[str, Any]]:
    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt(text)
        await sim.hangup()
        return frames

    return run(go())


def test_level1_make_payment_hands_off_for_step_up_in_one_model_hop() -> None:
    backend = FakeBackend({})
    rig = make_rig(backend=backend, script=[tool("make_payment", "tu_pay", **PAY)])
    write_call_state(rig.store)  # level 1

    frames = _one_prompt(rig, "Pay fifty dollars on my card from checking")
    assert len(rig.agent.calls) == 1  # one model hop: no AUTH_LEVEL_INSUFFICIENT round trip
    assert AUTH_LEVEL_INSUFFICIENT not in json.dumps(
        [r.content for c in rig.agent.calls for r in c["tool_results"]])
    # The tool's hand-off phrase, not silence, before the step-up prompt: it is wanted
    # (data/policy.json masks make_payment.
    assert spoken(frames) == MAKE_PAYMENT_PHRASE + " "
    end = end_frame(frames)
    assert end is not None
    data = end["data"]
    # The same hand-off ``authenticate`` produces.
    assert data["reason"] == "step-up" and data["tool"] == "make_payment"
    assert data["args"] == {"tool": "make_payment"}
    for secret in (CALL_TOKEN, "50.00", "purple"):
        assert secret not in end["frame"]["handoffData"]
    call = rig.call()
    # ...but the pending tool is the model's own call, so it gets the final result.
    assert call["pending_tool"] == {"tool_use_id": "tu_pay", "name": "make_payment", "args": PAY}
    assert call["step_up_target"] == "make_payment"
    assert backend.calls == []
    handoffs = [e for e in rig.events() if e["type"] == "handoff"]
    assert [(h["tool"], h["reason"], h["tool_use_id"]) for h in handoffs] == [
        ("make_payment", "step-up", "tu_pay")]


def test_level2_transfer_funds_takes_the_shortcut_too() -> None:
    xfer = {"from_account": "checking", "to_account": "savings", "amount": "20.00"}
    rig = make_rig(script=[tool("transfer_funds", "tu_x", **xfer)])
    write_call_state(rig.store, level=2)
    end = end_frame(_one_prompt(rig, "Move twenty dollars to savings"))
    assert end is not None and end["data"]["reason"] == "step-up"
    assert end["data"]["tool"] == "transfer_funds"
    assert rig.call()["pending_tool"]["args"] == xfer
    assert len(rig.agent.calls) == 1


def test_argument_errors_still_go_back_to_the_model() -> None:
    rig = make_rig(script=[tool("make_payment", "tu_pay", from_account="checking", to="purple",
                                amount="-5"),
                           say("How much would you like to pay?")])
    write_call_state(rig.store)  # level 1: below make_payment's level too
    frames = _one_prompt(rig, "Pay my card")
    assert end_frame(frames) is None
    (res,) = rig.agent.calls[1]["tool_results"]
    assert res.content["error"] == "INVALID_ARGUMENTS" and res.content["tool"] == "make_payment"
    assert spoken(frames) == "How much would you like to pay?"
    assert rig.call()["pending_tool"] is None


def test_a_model_called_authenticate_still_works_as_before() -> None:
    rig = make_rig(script=[tool("authenticate", "tu_auth", tool="change_address")])
    write_call_state(rig.store)
    end = end_frame(_one_prompt(rig, "I moved"))
    assert end is not None and end["data"] == {**end["data"], "reason": "step-up",
                                               "tool": "change_address"}
    assert rig.call()["pending_tool"] == {"tool_use_id": "tu_auth", "name": "authenticate",
                                          "args": {"tool": "change_address"}}


def test_an_inline_tool_below_its_level_is_still_refused_to_the_model() -> None:
    """Only front-door tools take the shortcut; an inline bank tool keeps today's answer."""
    backend = FakeBackend({"change_address": [{"error": AUTH_LEVEL_INSUFFICIENT,
                                               "tool": "change_address", "required_level": 2}]})
    rig = make_rig(backend=backend, script=[tool("change_address", "tu_addr", street="1 Main St", city="Springfield",
                                state="IL", postal_code="62701"),
                           tool("authenticate", "tu_auth", tool="change_address")])
    write_call_state(rig.store)  # level 1 < change_address's 2
    end = end_frame(_one_prompt(rig, "Change my address"))
    assert rig.agent.calls[1]["tool_results"][0].content == {"error": AUTH_LEVEL_INSUFFICIENT,
                                                              "tool": "change_address"}
    assert end is not None and rig.call()["pending_tool"]["name"] == "authenticate"


def test_a_lost_shortcut_handoff_is_resent_as_a_step_up() -> None:
    """Barge-in cancelled the stream before ``end``: the rebuilt hand-off is the step-up again."""
    rig = make_rig(script=[])
    write_call_state(rig.store, pending_tool={"tool_use_id": "tu_pay", "name": "make_payment",
                                              "args": PAY})
    end = end_frame(_one_prompt(rig, "hello?"))
    assert end is not None and end["data"]["reason"] == "step-up"
    assert end["data"]["tool"] == "make_payment"
    assert rig.agent.calls == []


def test_a_failed_step_up_result_answers_the_original_call() -> None:
    env = ResultEnvelope(status="not_verified", tool="make_payment", spoken=[], data={"level": 1},
                         next="offer_alternative").to_json()
    rig = make_rig(script=[say("<silent/>"), say("I can't do that today, but a specialist can.")])
    write_call_state(rig.store, pending_tool={"tool_use_id": "tu_pay", "name": "make_payment",
                                              "args": PAY}, pending_result=env)
    frames = _one_prompt(rig, "Okay, what now?")
    (res,) = rig.agent.calls[0]["tool_results"]
    assert res.tool_use_id == "tu_pay" and res.content["status"] == "not_verified"
    assert res.content["tool"] == "make_payment"
    assert spoken(frames) == "I can't do that today, but a specialist can."


# --------------------------------------------------------------------------- the phrase on a step-up
# A step-up shortcut speaks the tool's hand-off
# phrase and holds ``end`` for it like any hand-off phrase, with no prefetch. Message mode (the
# phrase does not depend on text release); stream mode is in test_stream_release.py.


def _events(caplog: pytest.LogCaptureFixture, name: str) -> list[dict[str, Any]]:
    return [ln for ln in lines(caplog.records, BRIDGE_LOG) if ln["event"] == name]


@pytest.mark.parametrize(("name", "args", "phrase", "level"), [
    ("make_payment", PAY, MAKE_PAYMENT_PHRASE, 1),
    ("transfer_funds", XFER, TRANSFER_PHRASE, 2),
])
def test_step_up_shortcut_speaks_the_handoff_phrase_then_holds_end_for_it(
        caplog: pytest.LogCaptureFixture, name: str, args: dict[str, Any], phrase: str,
        level: int) -> None:
    backend = FakeBackend({})
    d = direct([tool(name, "tu_x", **args)], backend=backend, level=level,
               policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=5000, chars_per_s=30, end_mode="wait"))
    ctx = context()
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, times = run(collect(d, "do it", ctx))
    assert chunks == [phrase + " "]
    estimate = 0.3 + len(phrase) / 30
    assert estimate - 0.05 <= times["__end__"] - times[phrase] <= estimate + 0.15
    (spoke,) = _events(caplog, "mask_spoken")
    assert (spoke["key"], spoke["tool"], spoke["kind"]) == (f"mask.{name}", name, "tool")
    (hold,) = _events(caplog, "handoff_hold")
    assert (hold["tool"], hold["source"], hold["prefetched"]) == (name, "mask", False)
    assert 0 < hold["wait_ms"] <= estimate * 1000
    (short,) = _events(caplog, "step_up_shortcut")
    assert short["tool"] == name
    # No prefetch: the bank is never called, nothing stored.
    assert backend.calls == [] and d.call().get("prefetched") is None
    # The step-up hand-off data are unchanged by the phrase.
    data = json.loads(ctx.pending_handoff_data.handoff_data)
    assert (data["reason"], data["tool"], data["args"]) == ("step-up", name, {"tool": name})
    assert d.call()["pending_tool"] == {"tool_use_id": "tu_x", "name": name, "args": args}
    assert d.call()["step_up_target"] == name
    assert len(d.agent.calls) == 1


def test_step_up_shortcut_with_end_mode_immediate_does_not_hold(
        caplog: pytest.LogCaptureFixture) -> None:
    d = direct([tool("make_payment", "tu_pay", **PAY)], backend=FakeBackend({}),
               policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=2500, end_mode="immediate"))
    ctx = context()
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, times = run(collect(d, "pay my card", ctx))
    assert chunks == [MAKE_PAYMENT_PHRASE + " "]
    assert times["__end__"] - times[MAKE_PAYMENT_PHRASE] < 0.1
    (hold,) = _events(caplog, "handoff_hold")
    assert hold["source"] == "mask" and hold["wait_ms"] == 0
    assert json.loads(ctx.pending_handoff_data.handoff_data)["reason"] == "step-up"


@pytest.mark.parametrize("level", [1, 3])
def test_a_handoff_refused_for_its_arguments_gets_no_phrase(
        caplog: pytest.LogCaptureFixture, level: int) -> None:
    # Below the level or at it: invalid arguments are refused to the model, so nothing is said
    # for the hand-off (only the model's own next words).
    d = direct([tool("make_payment", "tu_pay", from_account="checking", to="purple", amount="-5"),
                say("How much would you like to pay?")], backend=FakeBackend({}), level=level,
               policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=2500))
    ctx = context()
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, _ = run(collect(d, "pay my card", ctx))
    assert "".join(chunks) == "How much would you like to pay?"
    assert MAKE_PAYMENT_PHRASE not in "".join(chunks)
    assert _events(caplog, "mask_spoken") == [] and _events(caplog, "handoff_hold") == []
    assert _events(caplog, "step_up_shortcut") == []
    assert ctx.pending_handoff_data is None
    assert d.agent.calls[1]["tool_results"][0].content["error"] == "INVALID_ARGUMENTS"


def test_at_its_level_make_payment_still_speaks_the_phrase_before_the_confirmation(
        caplog: pytest.LogCaptureFixture) -> None:
    d = direct([tool("make_payment", "tu_pay", **PAY)], backend=FakeBackend({}), level=3,
               policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=5000, chars_per_s=30, end_mode="wait"))
    ctx = context()
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, times = run(collect(d, "pay my card", ctx))
    assert chunks == [MAKE_PAYMENT_PHRASE + " "]
    estimate = 0.3 + len(MAKE_PAYMENT_PHRASE) / 30
    assert estimate - 0.05 <= times["__end__"] - times[MAKE_PAYMENT_PHRASE] <= estimate + 0.15
    (hold,) = _events(caplog, "handoff_hold")
    assert (hold["source"], hold["prefetched"]) == ("mask", False)
    assert _events(caplog, "step_up_shortcut") == []
    data = json.loads(ctx.pending_handoff_data.handoff_data)
    assert (data["reason"], data["tool"]) == ("confirm", "make_payment")


# =========================================================================== front door


def pending(h: Harness, call_sid: str, name: str, args: dict[str, Any], tool_use_id: str) -> None:
    callstate.mutate(h.store, call_sid, lambda s: {
        "pending_tool": PendingTool(tool_use_id=tool_use_id, name=name, args=dict(args))})


def shortcut_handoff(h: Harness, call_sid: str, name: str = "make_payment",
                     args: dict[str, Any] | None = None, tool_use_id: str = "tu_pay"):
    """What the bridge leaves for a refused-for-level front-door tool."""
    pending(h, call_sid, name, args if args is not None else PAY, tool_use_id)
    return h.handoff(call_sid, {"reason": "step-up", "tool": name, "args": {"tool": name},
                                "journal": "jr-1"})


def bank_ops(h: Harness, op: str) -> list[dict[str, Any]]:
    return [p for p in h.lam.payloads if p.get("op") == op]


def test_step_up_handoff_for_a_pending_make_payment_starts_the_step_up(fd: Harness) -> None:
    call = sid(1)
    fd.login_thomas(call)
    root = shortcut_handoff(fd, call)
    assert says(root) == [
        "I can help you make that payment. First I need to ask you a few more questions "
        "to verify your identity.",
        "Please say or enter the last four digits of your Social Security number. If you can't, press pound."]
    assert fd.state(call).step_up_target == "make_payment"


def test_verified_step_up_goes_straight_to_the_read_back_then_pays(fd: Harness) -> None:
    call = sid(2)
    fd.login_thomas(call)
    shortcut_handoff(fd, call)
    fd.factor(call, S)
    root = fd.factor(call, OTP)
    assert fd.state(call).level == 3
    # The next TwiML is the read-back: no ConversationRelay connect in between.
    assert root.find(".//ConversationRelay") is None
    assert root.find("Gather") is not None
    assert says(root) == [SUCCESS, READBACK]
    assert fd.state(call).pending_result is None
    assert bank_ops(fd, "make_payment") == []
    root = fd.factor(call, "1")  # yes
    assert relay(root) is not None
    # the outcome is a <Say> before the connect; the greeting is only the follow-up.
    assert says(root)[0].startswith("Done. Your payment of fifty dollars")
    assert greeting(root) == "Is there anything else I can help you with?"
    (paid,) = bank_ops(fd, "make_payment")
    assert paid["idempotency_key"] == "tu_pay"  # the model's original call
    env = fd.pending_result(call)
    assert env.status == "completed" and env.tool == "make_payment" and env.next == "await_caller"
    assert env.data["executed"] is True
    assert [s.template for s in env.spoken] == [
        "stepup.intro", "factor.ssn4", "factor.ack", "factor.otp", "stepup.success",
        "confirm.make_payment", "confirm.make_payment.done", "common.anything_else"]
    # ...and the bridge sends it as the result for that same tool_use_id.
    assert fd.state(call).pending_tool.tool_use_id == "tu_pay"


def test_verified_step_up_then_no_cancels_with_the_step_up_in_spoken(fd: Harness) -> None:
    call = sid(3)
    fd.login_thomas(call)
    shortcut_handoff(fd, call)
    fd.factor(call, S)
    fd.factor(call, OTP)
    fd.factor(call, "2")  # no
    env = fd.pending_result(call)
    assert env.status == CANCELLED_BY_CALLER and env.tool == "make_payment"
    assert [s.template for s in env.spoken][:5] == [
        "stepup.intro", "factor.ssn4", "factor.ack", "factor.otp", "stepup.success"]
    assert bank_ops(fd, "make_payment") == []


def test_failed_step_up_returns_not_verified_for_make_payment(fd: Harness) -> None:
    call = sid(4)
    fd.login_thomas(call)
    root = shortcut_handoff(fd, call)
    for _ in range(10):
        if root.find(".//ConversationRelay") is not None or root.find(".//Enqueue") is not None:
            break
        root = fd.factor(call, "0000")
    env = fd.pending_result(call)
    assert env.status == "not_verified" and env.tool == "make_payment"
    assert env.next == "offer_alternative" and env.forTool is None
    assert fd.state(call).pending_tool.tool_use_id == "tu_pay"
    assert bank_ops(fd, "make_payment") == []


def test_interrupted_step_up_returns_interrupted_for_make_payment() -> None:
    h = build(resolver=FakeResolver([INTENT_SWITCH]), config_kw={"accept_spoken_factors": True})
    call = sid(5)
    h.login_thomas(call)
    shortcut_handoff(h, call)
    h.xml("factor", call, SpeechResult="actually what are your hours", Confidence="0.9")
    env = h.pending_result(call)
    assert env.status == "interrupted" and env.tool == "make_payment"
    assert env.next == "handle_utterance" and env.callerUtterance


def test_get_balance_pending_after_step_up_reads_the_balance_directly(fd: Harness) -> None:
    call = sid(6)
    fd.start(call, THOMAS)
    # Unverified, limited ai, login not closed: the caller asks something else at the ZIP
    # question (a cancel there is a decline and closes the login).
    fd.resolver.default = ResolverOutput(kind="intent_switch")
    fd.xml("factor", call, SpeechResult="what are your hours", Confidence="0.9")
    fd.resolver.default = ResolverOutput(kind="none")
    root = shortcut_handoff(fd, call, "get_balance", {"account": "checking"}, "tu_bal")
    assert says(root)[0].startswith("I can help you check your balance.")
    root = fd.factor(call, Z)
    assert fd.state(call).level == 1
    # Straight to the read-out: the success line and the balance in one greeting.
    assert says(root) == []
    text = greeting(root)
    assert text.startswith(SUCCESS + " Your checking account ending in four, eight, two, one")
    assert text.endswith("Is there anything else I can help you with?")
    env = fd.pending_result(call)
    assert env.status == "completed" and env.tool == "get_balance" and env.next == "await_caller"
    assert env.data["available"] == "1234.56"
    assert [s.template for s in env.spoken][:3] == ["stepup.intro", "factor.zip", "stepup.success"]
    assert fd.state(call).pending_tool.tool_use_id == "tu_bal"


def test_a_model_called_authenticate_still_returns_to_the_ai(fd: Harness) -> None:
    call = sid(7)
    fd.login_thomas(call)
    fd.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    fd.handoff(call, {"reason": "step-up", "tool": "make_payment", "journal": "jr-1"})
    fd.factor(call, S)
    root = fd.factor(call, OTP)
    assert greeting(root) == SUCCESS
    env = fd.pending_result(call)
    assert env.status == "verified" and env.tool == "authenticate"
    assert env.next == "retry_original_tool" and env.forTool == "make_payment"
