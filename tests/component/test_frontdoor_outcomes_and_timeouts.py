"""Outcome read-outs spoken before ConversationRelay,
``last_spoken`` sealed in call state, the yes/no speech timeout, and the silence hand-off actions.

Signed fake webhooks into the real front door (``test_frontdoor_support.build``). Expected words
come from the prompt catalog."""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.frontdoor import http
from lambdas.frontdoor.app import BARGE_IN_MODES, FrontDoorConfig
from lambdas.frontdoor.backend import ToolResult
from owlbank import callstate, last_spoken, sensitive
from owlbank.callstate import PendingTool
from owlbank.segments import clear_sensitive
from owlbank.token_seal import FakeSealer, UnsealError

from tests.component.test_frontdoor_caller_id_trust import PAY, S, Z, bank_ops, confirm_handoff, to_level3
from tests.component.test_frontdoor_confirmation_readout import balance_handoff, expected_checking_readout
from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    CLOCK,
    OTP,
    THOMAS,
    WORKFLOW,
    Harness,
    assert_outcome,
    assert_scripted,
    build,
    fd,
    greeting,
    relay,
    says,
    sid,
)

EN, ES = "en-US", "es-US"
WS = "wss://bridge.owlbank.example/ws"
TRANSFER = {"from_account": "savings", "to_account": "checking", "amount": "20.05"}
#: Outcome kinds of a deterministic confirmation.
OUTCOMES = ("payment_done", "transfer_done", "cancelled", "failed", "insufficient_funds")


def _to_level3(h: Harness, call: str, lang: str) -> None:
    h.start(call, THOMAS, lang_digit="2" if lang == ES else "")
    h.factor(call, Z)
    h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    h.handoff(call, {"reason": "step-up", "tool": "make_payment", "journal": "jr-1"})
    h.factor(call, S)
    h.factor(call, OTP)
    assert h.state(call).level == 3 and h.state(call).language == lang
    callstate.mutate(h.store, call, lambda s: {"pending_result": None, "pending_tool": None})


def _outcome(h: Harness, call: str, kind: str) -> ET.Element:
    """Run a deterministic confirmation to the outcome ``kind``; the outcome's TwiML."""
    tool, args = ("transfer_funds", TRANSFER) if kind == "transfer_done" else ("make_payment", PAY)
    if kind == "insufficient_funds":
        args = {**PAY, "amount": "999999.00"}
    if kind == "failed":
        h.app.d.backend.run_tool = lambda **_: ToolResult("error", code="UNAVAILABLE")  # type: ignore[method-assign]
    confirm_handoff(h, call, tool=tool, args=args)
    return h.xml("factor", call, Digits="2" if kind == "cancelled" else "1")


def _unseal(h: Harness, call: str, *, now: float = CLOCK) -> str | None:
    return last_spoken.unseal(h.sealer, h.state(call).last_spoken, call_sid=call, now=now)




@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(kind=st.sampled_from(OUTCOMES), lang=st.sampled_from((EN, ES)),
       mode=st.sampled_from(BARGE_IN_MODES))
def test_an_outcome_is_never_inside_an_interruptible_greeting(kind: str, lang: str,
                                                                    mode: str) -> None:
    """Over outcome kinds x languages x ``scriptedBargeIn``, the outcome text is a
    plain <Say> before <Connect> and never in the welcome greeting; the follow-up always is the
    greeting, interruptible as configured; ``last_spoken`` is the follow-up."""
    h = build(config_kw={"scripted_barge_in": mode})
    call = sid(1)
    try:
        _to_level3(h, call, lang)
        root = _outcome(h, call, kind)
        outcome, follow = h.pending_result(call).spoken[-2:]
        assert follow.template == "common.anything_else"
        assert follow.text == h.app.d.catalog.text("common.anything_else", lang)
        assert outcome.template.startswith("confirm.") and outcome.text
        assert_outcome(root, outcome.text, follow=follow.text, interruptible=mode)
        assert outcome.text not in greeting(root)
        assert _unseal(h, call) == follow.text
    finally:
        # each Hypothesis example is a new call on the same CallSid: forget what this process
        # knows of it (sealed entries, unverifiable) as well as the scrub's registered values
        sensitive.forget(call)




def test_payment_done_is_said_then_the_greeting_is_anything_else(fd: Harness) -> None:
    call = sid(2)
    to_level3(fd, call)
    root = _outcome(fd, call, "payment_done")
    (said,) = says(root)
    assert said.startswith("Done. Your payment of fifty dollars to your OwlBank Red Card from your checking account")
    assert [c.tag for c in root] == ["Say", "Connect"]
    assert greeting(root) == "Is there anything else I can help you with?"
    assert relay(root).get("welcomeGreetingInterruptible") == "any"
    assert relay(root).get("events", "").split() == ["speaker-events"]  # every connect
    assert len(bank_ops(fd, "make_payment")) == 1
    # The envelope still lists everything the caller heard, in order.
    assert [s.template for s in fd.pending_result(call).spoken] == [
        "confirm.make_payment", "confirm.make_payment.done", "common.anything_else"]


def test_balance_readout_is_still_the_interruptible_greeting(fd: Harness) -> None:
    """Read-outs that are not outcomes are still spoken in the welcome greeting."""
    fd.login_thomas(sid(3))
    root = balance_handoff(fd, sid(3))
    texts = expected_checking_readout(fd)
    assert_scripted(root, *texts, interruptible="any")
    assert _unseal(fd, sid(3)) == " ".join(texts)  # what the caller hears at session start


# --------------------------------------------------------------------------- last_spoken


def test_last_spoken_is_sealed_for_the_call_and_expires(fd: Harness) -> None:
    call = sid(4)
    to_level3(fd, call)
    _outcome(fd, call, "payment_done")
    rec = fd.state(call).last_spoken
    assert rec is not None and set(rec) == {"sealed", "expires_at"}
    assert rec["expires_at"] == int(CLOCK) + last_spoken.TTL_S == int(CLOCK) + 600
    text = "Is there anything else I can help you with?"
    assert text not in json.dumps(rec)
    # The sealer with the CallSid as context: unseals for this call only.
    assert fd.sealer.unseal(rec["sealed"], call_sid=call) == text
    with pytest.raises(UnsealError):
        fd.sealer.unseal(rec["sealed"], call_sid=sid(5))
    assert last_spoken.unseal(fd.sealer, rec, call_sid=sid(5), now=CLOCK) is None
    assert _unseal(fd, call, now=CLOCK + 599) == text
    assert _unseal(fd, call, now=CLOCK + 600) is None


def test_every_connect_replaces_last_spoken(fd: Harness) -> None:
    call = sid(6)
    root = fd.login_thomas(call)
    assert _unseal(fd, call) == greeting(root)  # the personal greeting
    root = balance_handoff(fd, call)
    assert _unseal(fd, call) == greeting(root)


def test_an_empty_greeting_clears_last_spoken(fd: Harness) -> None:
    call = sid(7)
    fd.login_thomas(call)
    assert fd.state(call).last_spoken is not None
    req = http.TwilioRequest(route="connect-action", params={"CallSid": call}, base_path="/prod")
    fd.app._relay(req, fd.state(call), EN, "")  # noqa: SLF001 - no flow connects with no greeting today
    assert fd.state(call).last_spoken is None


def test_last_spoken_text_is_never_logged(fd: Harness, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    call = sid(8)
    to_level3(fd, call)
    _outcome(fd, call, "payment_done")
    rec = fd.state(call).last_spoken
    last_spoken.unseal(fd.sealer, rec, call_sid=call, now=CLOCK + 601)  # expired: logged
    last_spoken.unseal(fd.sealer, {**rec, "sealed": "AAAA"}, call_sid=call, now=CLOCK)  # bad seal
    blob = "\n".join(r.getMessage() + " " + json.dumps(vars(r), default=str) for r in caplog.records
                     if r.name == "owlbank.last_spoken")
    assert "last_spoken_unusable" in blob
    texts = [s.text for s in fd.pending_result(call).spoken]
    for text in texts:
        assert text not in blob


def test_seal_failure_stores_nothing_and_the_call_goes_on(fd: Harness) -> None:
    class Broken(FakeSealer):
        def seal(self, token: str, *, call_sid: str) -> str:
            if " " in token:  # the greeting (call tokens have no spaces): KMS fails for it only
                raise RuntimeError("kms down: " + token)
            return super().seal(token, call_sid=call_sid)

    fd.app.d.sealer = Broken()
    root = fd.login_thomas(sid(9))
    assert relay(root) is not None
    assert fd.state(sid(9)).last_spoken is None


@pytest.mark.parametrize("record", [None, "x", {}, {"sealed": "", "expires_at": 1},
                                    {"sealed": "abc", "expires_at": True},
                                    {"sealed": "abc", "expires_at": float("nan")},
                                    {"sealed": "abc", "expires_at": "9999999999"}])
def test_unseal_of_a_bad_record_is_none(record: Any) -> None:
    assert last_spoken.unseal(FakeSealer(), record, call_sid=sid(1), now=CLOCK) is None


def test_seal_of_empty_text_is_none() -> None:
    assert last_spoken.seal(FakeSealer(), "", call_sid=sid(1), now=CLOCK) is None
    assert last_spoken.seal(FakeSealer(), "  ", call_sid=sid(1), now=CLOCK) is None


@settings(max_examples=50)
@given(text=st.text(min_size=1).filter(lambda t: t.strip()),
       now=st.integers(min_value=0, max_value=2**40))
def test_seal_unseal_round_trip(text: str, now: int) -> None:
    sealer = FakeSealer()
    rec = last_spoken.seal(sealer, text, call_sid=sid(1), now=now)
    assert rec is not None and rec["expires_at"] == now + last_spoken.TTL_S
    assert last_spoken.unseal(sealer, rec, call_sid=sid(1), now=now) == text
    assert last_spoken.unseal(sealer, rec, call_sid=sid(1), now=now + last_spoken.TTL_S) is None
    assert last_spoken.unseal(sealer, rec, call_sid=sid(2), now=now) is None


def test_unknown_field_preservation_keeps_last_spoken() -> None:
    """An item from code without the field round-trips it, and this code keeps it."""
    rec = {"sealed": "abc", "expires_at": 5}
    state = callstate.CallState.from_item({"call_sid": sid(1), "expires_at": 10, "last_spoken": rec})
    assert state.last_spoken == rec and state.to_item()["last_spoken"] == rec


# --------------------------------------------------------------------------- yes/no timeout


def test_yes_no_gathers_use_two_seconds_with_the_speech_model(fd: Harness) -> None:
    call = sid(10)
    to_level3(fd, call)
    gather = confirm_handoff(fd, call).find("Gather")
    assert gather is not None
    assert (gather.get("speechTimeout"), gather.get("speechModel")) == ("2", "deepgram_nova-3")


def test_digit_factors_keep_their_own_timeout() -> None:
    h = build(config_kw={"accept_spoken_factors": True})
    g = h.start(sid(11), THOMAS).find("Gather")  # the ZIP factor, spoken answers allowed
    assert g is not None and "speech" in g.get("input")
    assert (g.get("speechTimeout"), g.get("speechModel")) == ("3", "deepgram_nova-3")
    clear_sensitive(sid(11))


def test_auto_confirm_timeout_drops_the_speech_model_only_for_yes_no() -> None:
    h = build(config_kw={"accept_spoken_factors": True, "confirm_speech_timeout_s": "auto"})
    g = h.start(sid(13), THOMAS).find("Gather")
    assert (g.get("speechTimeout"), g.get("speechModel")) == ("3", "deepgram_nova-3")
    h.factor(sid(13), Z)
    h.set_pending_tool(sid(13), "authenticate", {"tool": "make_payment"})
    h.handoff(sid(13), {"reason": "step-up", "tool": "make_payment", "journal": "jr-1"})
    h.factor(sid(13), S)
    h.factor(sid(13), OTP)
    callstate.mutate(h.store, sid(13), lambda s: {"pending_result": None, "pending_tool": None})
    g = confirm_handoff(h, sid(13)).find("Gather")
    assert g.get("speechTimeout") == "auto" and "speechModel" not in g.attrib
    clear_sensitive(sid(13))


@pytest.mark.parametrize("raw, expected", [(None, 2), ("", 2), ("auto", "auto"), (" AUTO ", "auto"),
                                           ("1", 1), ("10", 10)])
def test_confirm_speech_timeout_from_env(raw: str | None, expected: int | str) -> None:
    env = {"OWLBANK_RELAY_WS_URL": WS}
    if raw is not None:
        env["OWLBANK_CONFIRM_SPEECH_TIMEOUT"] = raw
    cfg = FrontDoorConfig.from_env(env)
    assert cfg.confirm_speech_timeout_s == expected and cfg.speech_timeout_s == 3


@pytest.mark.parametrize("raw", ["0", "11", "1.5", "fast"])
def test_bad_confirm_speech_timeout_from_env_is_refused(raw: str) -> None:
    with pytest.raises(ValueError, match="confirm_speech_timeout_s|OWLBANK_CONFIRM_SPEECH_TIMEOUT"):
        FrontDoorConfig.from_env({"OWLBANK_RELAY_WS_URL": WS, "OWLBANK_CONFIRM_SPEECH_TIMEOUT": raw})


# --------------------------------------------------------------------------- silence actions


@pytest.mark.parametrize("lang", [EN, ES])
@pytest.mark.parametrize("data", [{"reason": "no_input_hangup"},
                                  {"reason": "escalate", "escalationReason": "no_input_hangup"}])
def test_no_input_hangup_says_goodbye_and_hangs_up(fd: Harness, lang: str, data: dict) -> None:
    call = sid(14)
    fd.start(call, THOMAS, lang_digit="2" if lang == ES else "")
    fd.factor(call, Z)
    root = fd.handoff(call, {**data, "journal": "jr-9"})
    assert [c.tag for c in root] == ["Say", "Hangup"]
    assert says(root) == [fd.app.d.catalog.text("common.goodbye_no_input", lang)]
    st = fd.state(call)
    assert st.sealed_token is None and st.last_spoken is None
    assert all(t["revoked"] for t in fd.store.query("tokens", {"call_sid": call},
                                                    index="call_sid-index"))
    keys = [e.get("prompt_key") for e in fd.journal(call) if e["type"] == "prompt"]
    assert keys[-1] == "common.goodbye_no_input"


def test_no_input_hangup_wins_over_a_pending_tool(fd: Harness) -> None:
    call = sid(15)
    fd.login_thomas(call)
    fd.set_pending_tool(call, "get_balance", {"account": "checking"})
    root = fd.handoff(call, {"reason": "no_input_hangup"})
    assert [c.tag for c in root] == ["Say", "Hangup"]
    assert bank_ops(fd, "get_balance") == []


@pytest.mark.parametrize("data", [{"reason": "escalate", "escalationReason": "no_input"},
                                  {"reason": "no_input"}])
def test_no_input_escalates_with_that_reason(fd: Harness, data: dict) -> None:
    call = sid(16)
    fd.login_thomas(call)
    root = fd.handoff(call, {**data, "journal": "jr-10"})
    enq = root.find("Enqueue")
    assert enq is not None and enq.get("workflowSid") == WORKFLOW
    attrs = json.loads(root.find("Enqueue/Task").text)
    assert attrs["escalation_reason"] == "no_input"
    assert fd.state(call).escalation["reason"] == "no_input"
    assert says(root) == [fd.app.d.catalog.text(
        fd.app.d.policy.escalation_route("no_input").phrase_key, EN)]


def test_no_input_escalation_wins_over_a_pending_tool(fd: Harness) -> None:
    call = sid(17)
    fd.login_thomas(call)
    callstate.mutate(fd.store, call, lambda s: {
        "pending_tool": PendingTool(tool_use_id="tu_bal", name="get_balance",
                                    args={"account": "checking"})})
    root = fd.handoff(call, {"reason": "escalate", "escalationReason": "no_input"})
    assert json.loads(root.find("Enqueue/Task").text)["escalation_reason"] == "no_input"
    assert bank_ops(fd, "get_balance") == []


@pytest.mark.parametrize("raw, expected", [(None, 3), ("", 3), ("  ", 3), ("5", 5), (" 10 ", 10), ("1", 1)])
def test_keypad_timeout_from_env(raw: str | None, expected: int) -> None:
    """Keyed-phone: ``OWLBANK_KEYPAD_TIMEOUT``, whole seconds 1-10, default 3."""
    env = {"OWLBANK_RELAY_WS_URL": WS}
    if raw is not None:
        env["OWLBANK_KEYPAD_TIMEOUT"] = raw
    assert FrontDoorConfig.from_env(env).factor_timeout_s == expected


@pytest.mark.parametrize("raw", ["abc", "0", "11", "1.5", "-2"])
def test_bad_keypad_timeout_from_env_is_refused(raw: str) -> None:
    with pytest.raises(ValueError, match="OWLBANK_KEYPAD_TIMEOUT") as exc:
        FrontDoorConfig.from_env({"OWLBANK_RELAY_WS_URL": WS, "OWLBANK_KEYPAD_TIMEOUT": raw})
    assert raw not in str(exc.value).replace("OWLBANK_KEYPAD_TIMEOUT", "").replace("1 to 10", "")
