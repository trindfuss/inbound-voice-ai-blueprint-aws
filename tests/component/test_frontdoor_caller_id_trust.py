"""The front door's caller-ID trust gate and deterministic confirmation.

Signed fake webhooks into the real front door, with the real auth service and bank
tools behind the fake Lambda client (``test_frontdoor_support.build``).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank import callstate
from owlbank.callstate import PendingTool
from owlbank.envelope import CANCELLED_BY_CALLER
from owlbank.segments import clear_sensitive
from owlbank.slots import FakeResolver, ResolverOutput

from tests.component.test_frontdoor_support import (
    OTP,
    TEST_FACTORS,
    THOMAS,
    Harness,
    all_fixture_factor_values,
    assert_no_leak,
    assert_outcome,
    assert_scripted,
    build,
    fd,  # noqa: F401 - fixture
    relay,
    says,
    sid,
)

PAY = {"from_account": "checking", "to": "red", "amount": "50.00"}
READBACK = ("To confirm, you'd like to pay fifty dollars toward your OwlBank Red Card from your "
            "checking account. Is that right? Say yes or no, or press 1 for yes and 2 for no.")
INTENT_SWITCH = ResolverOutput(kind="intent_switch")
Z, S = TEST_FACTORS["C-1001"]["zip"], TEST_FACTORS["C-1001"]["ssn4"]


def bank_ops(h: Harness, op: str) -> list[dict[str, Any]]:
    return [p for p in h.lam.payloads if p.get("op") == op]


def to_level3(h: Harness, call_sid: str) -> None:
    h.login_thomas(call_sid)
    h.set_pending_tool(call_sid, "authenticate", {"tool": "make_payment"})
    h.handoff(call_sid, {"reason": "step-up", "tool": "make_payment", "journal": "jr-1"})
    h.factor(call_sid, S)
    h.factor(call_sid, OTP)
    assert h.state(call_sid).level == 3
    callstate.mutate(h.store, call_sid, lambda s: {"pending_result": None, "pending_tool": None})


def confirm_handoff(h: Harness, call_sid: str, tool: str = "make_payment",
                    args: dict[str, Any] | None = None, tool_use_id: str = "tu_pay") -> ET.Element:
    callstate.mutate(h.store, call_sid, lambda s: {
        "pending_tool": PendingTool(tool_use_id=tool_use_id, name=tool, args=dict(args or PAY))})
    return h.handoff(call_sid, {"reason": "confirm", "tool": tool, "journal": "jr-2"})


def answers(h: Harness, call_sid: str) -> list[str]:
    return [e["answer"] for e in h.journal(call_sid) if e["type"] == "confirm_answer"]


# --------------------------------------------------------------------------- trust gate


def test_untrusted_ani_asks_the_next_levels_factors_and_sends_them_together(fd: Harness) -> None:
    call = sid(1)
    fd.xml("incoming", call, From=THOMAS, To="+12025550100", CallStatus="ringing")  # no StirVerstat
    st = fd.state(call)
    assert st.stir_verstat is None and st.ani_trusted is False
    start = next(e for e in fd.journal(call) if e["type"] == "call_start")
    assert start["stir_verstat"] is None and start["ani_trusted"] is False
    assert start["ani_match_counts"] is False
    root = fd.xml("language", call, Digits="")
    assert says(root)[-1] == "Please say or enter your five-digit ZIP code. If you can't, press pound."
    root = fd.factor(call, Z)
    # the held ZIP is acknowledged before the next factor.
    assert says(root) == ["Got it.",
                          "Please say or enter the last four digits of your Social Security number. If you can't, press pound."]
    assert bank_ops(fd, "verify") == []  # nothing verified on the ZIP alone
    root = fd.factor(call, S)
    assert relay(root).get("welcomeGreeting").startswith("Hi Thomas.")
    (v,) = bank_ops(fd, "verify")
    assert sorted(v["factors"]) == ["ssn4", "zip"] and v["target_level"] == 1
    assert v["stir_verstat"] is None
    st = fd.state(call)
    assert st.level == 1 and set(st.factors_verified) == {"zip", "ssn4"}  # no ani_match
    assert "held" not in (fd.session(call).flow or {})


def test_trusted_ani_keeps_the_ani_match_factor(fd: Harness) -> None:
    call = sid(2)
    fd.login_thomas(call)
    (v,) = bank_ops(fd, "verify")
    assert v["stir_verstat"] == "TN-Validation-Passed-A" and sorted(v["factors"]) == ["zip"]
    assert set(fd.state(call).factors_verified) == {"ani_match", "zip"}
    assert fd.state(call).ani_trusted is True


def test_untrusted_step_up_to_level_2_asks_the_otp(fd: Harness) -> None:
    call = sid(3)
    fd.xml("incoming", call, From=THOMAS, To="+12025550100", CallStatus="ringing",
           StirVerstat="TN-Validation-Passed-C")
    fd.xml("language", call, Digits="")
    fd.factor(call, Z)
    fd.factor(call, S)
    fd.set_pending_tool(call, "authenticate", {"tool": "change_address"})
    root = fd.handoff(call, {"reason": "step-up", "tool": "change_address", "journal": "jr"})
    assert says(root)[-1].startswith("We just sent a six-digit code")  # ZIP and SSN aren't re-asked
    fd.factor(call, OTP)
    assert fd.state(call).level >= 2


def test_malformed_verstat_is_untrusted_not_an_error(fd: Harness) -> None:
    call = sid(4)
    fd.xml("incoming", call, From=THOMAS, To="+12025550100", CallStatus="ringing",
           StirVerstat="<script>TN-Validation-Passed-A")
    assert fd.state(call).stir_verstat is None and fd.state(call).ani_trusted is False


def test_untrusted_failures_never_feed_the_customer_lockout(fd: Harness) -> None:
    for n in range(3):
        call = sid(10 + n)
        fd.xml("incoming", call, From=THOMAS, To="+12025550100", CallStatus="ringing")
        fd.xml("language", call, Digits="")
        for _ in range(3):
            fd.factor(call, "00000")
            fd.factor(call, "0000")
    keys = [k[0] for k in fd.store._tables["calls"]]  # noqa: SLF001
    assert not [k for k in keys if str(k).startswith("ratelimit#customer#")]
    call = sid(20)
    fd.login_thomas(call)  # the real customer, trusted ANI: not locked out
    assert fd.state(call).level == 1


# --------------------------------------------------------------------------- confirmation


def test_confirm_reads_back_and_runs_once_on_yes(fd: Harness) -> None:
    call = sid(30)
    to_level3(fd, call)
    root = confirm_handoff(fd, call)
    assert says(root) == [READBACK]
    gather = root.find("Gather")
    assert gather is not None and "speech" in gather.get("input")
    assert bank_ops(fd, "make_payment") == []
    root = fd.xml("factor", call, Digits="1")
    # the result is a plain <Say> before the connect (it can't be cut off);
    # only the follow-up is the interruptible greeting.
    done = ("Done. Your payment of fifty dollars to your OwlBank Red Card from your checking "
            "account is complete. Your confirmation number is ")
    (said,) = says(root)
    assert said.startswith(done)
    assert_outcome(root, said, follow="Is there anything else I can help you with?")
    (pay,) = bank_ops(fd, "make_payment")
    assert pay["idempotency_key"] == "tu_pay" and pay["args"] == PAY
    env = fd.pending_result(call)
    assert env.status == "completed" and env.tool == "make_payment" and env.next == "await_caller"
    assert env.data["executed"] is True and env.data["amount"] == "50.00"
    assert [s.template for s in env.spoken] == ["confirm.make_payment", "confirm.make_payment.done",
                                                "common.anything_else"]
    assert answers(fd, call) == ["yes"]


@pytest.mark.parametrize("answer", [{"Digits": "2"}, {"SpeechResult": "No.", "Confidence": "0.9"},
                                    {"SpeechResult": "cancel", "Confidence": "0.9"}])
def test_confirm_no_or_cancel_does_not_run(fd: Harness, answer: dict[str, str]) -> None:
    call = sid(31)
    to_level3(fd, call)
    confirm_handoff(fd, call)
    root = fd.xml("factor", call, **answer)
    assert_outcome(root, "Okay, I haven't done that.", follow="Is there anything else I can help you with?")
    assert bank_ops(fd, "make_payment") == []
    env = fd.pending_result(call)
    assert env.status == CANCELLED_BY_CALLER and env.next == "await_caller"
    assert env.data["executed"] is False and env.completed is False


def test_confirm_silence_reprompts_once_then_cancels(fd: Harness) -> None:
    call = sid(32)
    to_level3(fd, call)
    confirm_handoff(fd, call)
    root = fd.xml("factor", call)
    assert says(root) == ["Please say yes or no.", READBACK]
    root = fd.xml("factor", call)
    assert_outcome(root, "Okay, I haven't done that.", follow="Is there anything else I can help you with?")
    assert fd.pending_result(call).status == CANCELLED_BY_CALLER
    assert fd.pending_result(call).data["answer"] == "no_answer"
    assert bank_ops(fd, "make_payment") == []


def test_confirm_intent_switch_is_interrupted_and_does_not_run() -> None:
    h = build(resolver=FakeResolver([INTENT_SWITCH]))
    call = sid(33)
    to_level3(h, call)
    confirm_handoff(h, call)
    root = h.xml("factor", call, SpeechResult="Actually what's my savings balance?", Confidence="0.9")
    assert relay(root).get("welcomeGreeting") == "One moment while I get that for you."
    env = h.pending_result(call)
    assert env.status == "interrupted" and env.next == "handle_utterance"
    assert env.callerUtterance == "Actually what's my savings balance?" and env.completed is False
    assert bank_ops(h, "make_payment") == []


def test_replayed_yes_webhook_does_not_pay_twice(fd: Harness) -> None:
    call = sid(34)
    to_level3(fd, call)
    confirm_handoff(fd, call)
    fd.xml("factor", call, Digits="1")
    fd.xml("factor", call, Digits="1")  # Twilio retry: no flow in progress now
    assert len(bank_ops(fd, "make_payment")) == 1


def test_confirm_insufficient_funds_is_failed_and_said(fd: Harness) -> None:
    call = sid(35)
    to_level3(fd, call)
    confirm_handoff(fd, call, args={**PAY, "amount": "999999.00"})
    root = fd.xml("factor", call, Digits="1")
    assert_outcome(root,
                   "I'm sorry, there isn't enough available in that account, so I didn't do that.",
                   follow="Is there anything else I can help you with?")
    env = fd.pending_result(call)
    assert env.status == "failed" and env.data == {"executed": False, "error": "INSUFFICIENT_FUNDS"}


def test_transfer_readback_in_spanish(fd: Harness) -> None:
    call = sid(36)
    fd.start(call, THOMAS, lang_digit="2")
    fd.factor(call, Z)
    fd.set_pending_tool(call, "authenticate", {"tool": "transfer_funds"})
    fd.handoff(call, {"reason": "step-up", "tool": "transfer_funds", "journal": "jr"})
    fd.factor(call, S)
    fd.factor(call, OTP)
    root = confirm_handoff(fd, call, tool="transfer_funds",
                           args={"from_account": "savings", "to_account": "checking", "amount": "20.05"})
    assert says(root) == ["Para confirmar, desea transferir veinte dólares con cinco centavos de su "
                          "cuenta de ahorros a su cuenta de cheques. ¿Es correcto? Diga sí o no, u "
                          "oprima 1 para sí y 2 para no."]
    root = fd.xml("factor", call, SpeechResult="sí", Confidence="0.9")
    (said,) = says(root)  # the outcome as <Say>, the follow-up as the greeting
    assert said.startswith("Listo. Su transferencia de veinte dólares con cinco centavos")
    assert_outcome(root, said, follow="¿Hay algo más en lo que pueda ayudarle?")
    assert len(bank_ops(fd, "transfer_funds")) == 1


def test_confirm_below_the_tools_level_steps_up_first(fd: Harness) -> None:
    call = sid(37)
    fd.login_thomas(call)  # level 1
    root = confirm_handoff(fd, call)
    assert says(root)[0].startswith("I can help you make that payment.")
    assert bank_ops(fd, "make_payment") == []


def test_invalid_pending_args_are_never_read_or_run(fd: Harness) -> None:
    call = sid(38)
    to_level3(fd, call)
    root = confirm_handoff(fd, call, args={**PAY, "amount": "-5", "customer_id": "C-1002"})
    assert_scripted(root, "I'm sorry, something went wrong on our end.")
    assert fd.pending_result(call).status == "failed"
    assert bank_ops(fd, "make_payment") == []


def test_confirm_handoff_without_a_pending_tool_is_ignored(fd: Harness) -> None:
    """HandoffData alone never starts a confirmation (only call state, written by the bridge)."""
    call = sid(39)
    to_level3(fd, call)
    root = fd.handoff(call, {"reason": "confirm", "tool": "make_payment", "args": PAY})
    assert root.find("Gather") is None
    assert bank_ops(fd, "make_payment") == []


# --------------------------------------------------------------------------- property test

ANSWER = st.sampled_from(["yes", "yes_speech", "no", "no_speech", "silence", "off_script",
                          "cancel", "repeat"])
_PARAMS = {"yes": {"Digits": "1"}, "yes_speech": {"SpeechResult": "yes", "Confidence": "0.9"},
           "no": {"Digits": "2"}, "no_speech": {"SpeechResult": "no", "Confidence": "0.9"},
           "silence": {}, "off_script": {"SpeechResult": "what is my balance", "Confidence": "0.9"},
           "cancel": {"SpeechResult": "never mind", "Confidence": "0.9"},
           "repeat": {"SpeechResult": "repeat", "Confidence": "0.9"}}


def expected_decision(seq: list[str]) -> str | None:
    """The model of the flow: the first decisive answer (or None if the caller hung up)."""
    tries = repeats = 0
    for a in seq:
        if a.startswith("yes"):
            return "yes"
        if a.startswith("no") or a == "cancel":
            return "no"
        if a == "off_script":
            return "interrupted"
        if a == "silence":
            tries += 1
            if tries > 1:
                return "no_answer"
        if a == "repeat":
            repeats += 1
            if repeats > 3:
                return "no_answer"
    return None


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(ANSWER, min_size=0, max_size=7), st.integers(min_value=0, max_value=2))
def test_deterministic_tool_runs_iff_the_recorded_answer_is_yes_and_at_most_once(
        caplog: pytest.LogCaptureFixture, seq: list[str], replays: int) -> None:
    """For any sequence of caller answers (then up to two replayed webhooks), the payment
    runs exactly when the recorded answer is yes, and never more than once."""
    caplog.set_level(logging.DEBUG)
    h = build(resolver=FakeResolver(default=INTENT_SWITCH))
    call = sid(50)
    try:
        to_level3(h, call)
        confirm_handoff(h, call)
        sent: list[str] = []
        for a in seq:
            flow = h.session(call).flow
            if not flow or flow.get("kind") != "confirm":
                break
            sent.append(a)
            h.xml("factor", call, **_PARAMS[a])
        for _ in range(replays):  # Twilio retries of the last webhook
            if sent:
                h.post("factor", call, **_PARAMS[sent[-1]])
        decision = expected_decision(seq)
        ran = len(bank_ops(h, "make_payment"))
        recorded = answers(h, call)
        assert ran <= 1
        assert (ran == 1) == ("yes" in recorded) == (decision == "yes")
        assert recorded.count("yes") <= 1
        if decision is not None and decision != "yes":
            st_ = h.state(call)
            env = st_.pending_result
            assert env is not None
            assert env["status"] == ("interrupted" if decision == "interrupted" else CANCELLED_BY_CALLER)
        assert_no_leak(h, caplog.records, all_fixture_factor_values() | h.sent_factors)
    finally:
        caplog.clear()
        clear_sensitive(call)
