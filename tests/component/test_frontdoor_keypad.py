"""AI keypad shortcuts: the AI session's ConversationRelay
TwiML per ``OWLBANK_AI_KEYPAD``, and the bridge's keypad hand-off (0 pressed) routed by the front
door to the policy's queue for ``caller_requested_human``, whatever tool is pending.

Signed fake webhooks into the real front door (``test_frontdoor_support.build``)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pytest

from lambdas.frontdoor.app import AI_KEYPAD_ENV, FrontDoorConfig
from owlbank import callstate
from owlbank.callstate import PendingTool
from owlbank.segments import clear_sensitive
from tests.component.test_frontdoor_caller_id_trust import bank_ops
from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    WORKFLOW,
    Harness,
    build,
    fd,
    relay,
    says,
    sid,
)

WS = "wss://bridge.owlbank.example/ws"
EN = "en-US"
SUMMARY = "The caller pressed 0 to speak with an agent."
KEYPAD_HANDOFF = {"reason": "escalate", "tool": "escalate",
                  "escalationReason": "caller_requested_human", "trigger": "keypad",
                  "journal": "jr-keypad", "conversationId": "conv_" + "a" * 26}

#: The relay TwiML after a verified login, captured before the keypad shortcuts, with the nonce masked.
TODAY = (
    '<?xml version="1.0" encoding="UTF-8"?><Response><Connect action="/prod/voice/connect-action">'
    '<ConversationRelay conversationConfiguration="conv_configuration_dddddddddddddddddddddddddd" '
    'dtmfDetection="false" events="speaker-events" interruptible="speech" language="en-US" '
    'speechModel="nova-3-general" transcriptionLanguage="en-US" transcriptionProvider="Deepgram" '
    'ttsProvider="ElevenLabs" url="wss://bridge.owlbank.example/ws?n=NONCE" '
    'voice="UgBBYS2sOqTuMpoF3BR0-v4_turbo" '
    'welcomeGreeting="Hi Thomas. I am the OwlBank virtual agent. How can I help you today?">'
    '<Parameter name="callSid" value="CAeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeab" />'
    "</ConversationRelay></Connect></Response>"
)


def _masked(body: str) -> str:
    return re.sub(r"n=[A-Za-z0-9_\-]+", "n=NONCE", body)


@pytest.fixture
def keypad_on(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    h = build(config_kw={"ai_keypad": True})
    yield h
    for call in {k[0].split("#")[0] for k in h.store._tables["calls"]}:  # noqa: SLF001
        clear_sensitive(call)


# --------------------------------------------------------------------------- TwiML


def test_off_gives_today_s_relay_twiml_byte_for_byte(fd: Harness) -> None:
    assert fd.app.cfg.ai_keypad is False
    fd.login_thomas(sid(1))
    assert _masked(fd.bodies[-1]) == TODAY


def test_on_reports_keys_also_during_agent_speech(keypad_on: Harness) -> None:
    cr = relay(keypad_on.login_thomas(sid(1)))
    assert cr.get("dtmfDetection") == "true"
    assert cr.get("interruptible") == "any"
    assert cr.get("reportInputDuringAgentSpeech") == "dtmf"
    # Only those three attributes differ from today's.
    expected = (TODAY.replace('dtmfDetection="false"', 'dtmfDetection="true"')
                .replace('interruptible="speech"', 'interruptible="any"')
                .replace('language="en-US" speechModel', 'language="en-US" '
                         'reportInputDuringAgentSpeech="dtmf" speechModel'))
    assert _masked(keypad_on.bodies[-1]) == expected


# --------------------------------------------------------------------------- env


@pytest.mark.parametrize(("raw", "expected"), [(None, True), ("", True), ("on", True),
                                               ("ON", True), (" off ", False), ("off", False)])
def test_env_parsing(raw: str | None, expected: bool) -> None:
    env = {"OWLBANK_RELAY_WS_URL": WS}
    if raw is not None:
        env[AI_KEYPAD_ENV] = raw
    assert FrontDoorConfig.from_env(env).ai_keypad is expected


@pytest.mark.parametrize("raw", ["true", "1", "yes", "enabled"])
def test_bad_env_value_stops_start_up_naming_the_variable(raw: str) -> None:
    with pytest.raises(ValueError, match="OWLBANK_AI_KEYPAD") as exc:
        FrontDoorConfig.from_env({"OWLBANK_RELAY_WS_URL": WS, AI_KEYPAD_ENV: raw})
    assert raw not in str(exc.value)


# --------------------------------------------------------------------------- the hand-off


def _bridge_escalated(h: Harness, call: str) -> None:
    """What the bridge's ``_keypad_escalate`` writes before ending the session."""
    callstate.mutate(h.store, call, lambda s: {"escalation": {
        "reason": "caller_requested_human", "summary": SUMMARY, "source": "keypad",
        "journal_ref": "jr-keypad"}})


PENDING: list[tuple[str, dict[str, Any]] | None] = [
    None,
    ("get_balance", {"account": "checking"}),                       # a deterministic tool
    ("make_payment", {"payee": "electric", "amount": "10.00"}),     # a confirmed bank tool
    ("authenticate", {"tool": "make_payment"}),                     # a step-up
    ("escalate", {"reason": "fraud_suspected", "summary": "x"}),    # the model's own escalation
]


@pytest.mark.parametrize("ai_keypad", [True, False])
@pytest.mark.parametrize("pending", PENDING, ids=lambda p: p[0] if p else "none")
def test_keypad_handoff_goes_to_the_support_queue_whatever_is_pending(
        pending: tuple[str, dict[str, Any]] | None, ai_keypad: bool,
        caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    h = build(config_kw={"ai_keypad": ai_keypad})
    call = sid(2)
    try:
        h.login_thomas(call)
        _bridge_escalated(h, call)
        if pending is not None:
            name, args = pending
            callstate.mutate(h.store, call, lambda s: {
                "pending_tool": PendingTool(tool_use_id="tu_1", name=name, args=args)})
        root = h.handoff(call, KEYPAD_HANDOFF)
        enq = root.find("Enqueue")
        assert enq is not None and enq.get("workflowSid") == WORKFLOW
        attrs = json.loads(root.find("Enqueue/Task").text)  # type: ignore[union-attr,arg-type]
        assert attrs["escalation_reason"] == "caller_requested_human"
        assert attrs["ai_summary"] == SUMMARY
        assert attrs["journal_ref"] == "jr-keypad"
        assert attrs["conversation_id"] == KEYPAD_HANDOFF["conversationId"]
        st = h.state(call)
        assert st.escalation["reason"] == "caller_requested_human"
        assert st.escalation["queue"] == h.app.d.policy.queue_name("support", EN)
        assert st.escalation["queue"] == "OwlBank Support EN"
        assert says(root) == [h.app.d.catalog.text(
            h.app.d.policy.escalation_route("caller_requested_human").phrase_key, EN)]
        assert bank_ops(h, "get_balance") == [] and bank_ops(h, "make_payment") == []
        assert st.sealed_token is None
    finally:
        for c in {k[0].split("#")[0] for k in h.store._tables["calls"]}:  # noqa: SLF001
            clear_sensitive(c)


def test_keypad_handoff_with_an_unknown_reason_takes_the_usual_path(fd: Harness) -> None:
    """An unknown reason is not the keypad branch: the usual hand-off path runs (here the pending
    tool wins, as it does for any other hand-off)."""
    call = sid(3)
    fd.login_thomas(call)
    fd.set_pending_tool(call, "get_balance", {"account": "checking"})
    root = fd.handoff(call, {**KEYPAD_HANDOFF, "escalationReason": "not_a_reason"})
    assert root.find("Enqueue") is None
    assert len(bank_ops(fd, "get_balance")) == 1


def test_non_keypad_escalate_handoff_behaves_as_before(fd: Harness) -> None:
    """Without ``trigger`` the pending tool still wins over HandoffData (``_classify_handoff``)."""
    call = sid(4)
    fd.login_thomas(call)
    fd.set_pending_tool(call, "get_balance", {"account": "checking"})
    root = fd.handoff(call, {k: v for k, v in KEYPAD_HANDOFF.items() if k != "trigger"})
    assert root.find("Enqueue") is None
    assert len(bank_ops(fd, "get_balance")) == 1


def test_non_keypad_model_escalation_keeps_its_reason_and_summary(fd: Harness) -> None:
    call = sid(5)
    fd.login_thomas(call)
    fd.set_pending_tool(call, "escalate", {"reason": "fraud_suspected", "summary": "Card stolen."})
    root = fd.handoff(call, {"reason": "escalate", "tool": "escalate", "journal": "jr-5"})
    attrs = json.loads(root.find("Enqueue/Task").text)  # type: ignore[union-attr,arg-type]
    assert attrs["escalation_reason"] == "fraud_suspected"
    assert fd.state(call).escalation["queue"] == fd.app.d.policy.queue_name("fraud", EN)
