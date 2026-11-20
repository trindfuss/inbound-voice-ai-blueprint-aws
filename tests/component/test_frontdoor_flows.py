"""``<Connect action>`` flows: balance, step-up 1 -> 3, escalation, failure paths,
and the status callback. The no-leak scan runs at teardown."""

from __future__ import annotations

import json

from owlbank import tokens
from owlbank.slots import FakeResolver, ResolverOutput

from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    CLOCK, ENQUEUE_ACTION_URL, MARIA, NOBODY, OTP, TEST_FACTORS, THOMAS, WORKFLOW, RecordingCloser,
    build, fd,
    assert_scripted, greeting, relay, says, sid,
)


def _balance_handoff(fd, call_sid, account="checking"):
    fd.set_pending_tool(call_sid, "get_balance", {"account": account})
    return fd.handoff(call_sid, {"reason": "tool", "tool": "get_balance",
                                 "args": {"account": account}, "journal": "jr-1"})




def test_balance_read_out_then_reconnect_with_follow_up(fd):
    fd.login_thomas(sid(1))
    root = _balance_handoff(fd, sid(1))
    # the read-out is the interruptible ConversationRelay greeting, not <Say> verbs.
    assert_scripted(
        root,
        "Your checking account ending in four, eight, two, one has an available balance of "
        "one thousand two hundred thirty-four dollars and fifty-six cents. Your current balance "
        "is one thousand three hundred ten dollars and six cents.",
        "That includes seventy-five dollars and fifty cents in pending transactions.",
        "Is there anything else I can help you with?",
    )
    assert root.find("Connect").get("action") == "/prod/voice/connect-action"
    env = fd.pending_result(sid(1))
    assert env.status == "completed" and env.next == "await_caller" and env.tool == "get_balance"
    assert [s.template for s in env.spoken] == ["balance.read", "balance.read.pending",
                                                "common.anything_else"]
    assert env.data == {"account": "checking", "last4": "4821", "available": "1234.56",
                        "current": "1310.06", "pending": "75.50", "currency": "USD",
                        "asOf": "2026-10-10T14:58:00Z"}
    assert fd.state(sid(1)).interrupted_utterance is None
    # Journal: what was said, from which template.
    keys = [e.get("prompt_key") for e in fd.journal(sid(1)) if e["type"] == "prompt"]
    assert keys[-3:] == ["balance.read", "balance.read.pending", "common.anything_else"]
    # Follow-up: savings, same level, no pending sentence (pending is zero).
    root = _balance_handoff(fd, sid(1), "savings")
    assert says(root) == [] and "seven, seven, three, zero" in greeting(root)
    assert "pending" not in greeting(root)
    assert greeting(root).endswith(". Is there anything else I can help you with?")
    assert fd.pending_result(sid(1)).data["available"] == "8420.00"


def test_balance_no_account_is_not_found_with_follow_up(fd):
    fd.start(sid(1), MARIA, lang_digit="2")
    fd.factor(sid(1), TEST_FACTORS["C-1004"]["zip"])
    root = _balance_handoff(fd, sid(1), "savings")
    assert_scripted(root, "No veo una cuenta de ahorros en su perfil.",
                    "¿Hay algo más en lo que pueda ayudarle?")
    env = fd.pending_result(sid(1))
    assert env.status == "not_found" and env.next == "await_caller"


def test_balance_bank_unavailable_is_failed_offer_alternative(fd):
    fd.login_thomas(sid(1))
    fd.lam.overrides["get_balance"] = {"error": "UNAVAILABLE", "tool": "get_balance"}
    root = _balance_handoff(fd, sid(1))
    # No welcome after it: the greeting is the apology alone.
    assert_scripted(root, "I'm sorry, I can't look up your account right now.")
    env = fd.pending_result(sid(1))
    assert env.status == "failed" and env.next == "offer_alternative"


def test_balance_without_a_token_steps_up_instead(fd):
    fd.start(sid(1), THOMAS)
    # Unverified, limited ai, login not closed: the caller asks something else at the ZIP
    # question (a cancel there is a decline and closes the login).
    fd.resolver.default = ResolverOutput(kind="intent_switch")
    fd.xml("factor", sid(1), SpeechResult="what are your hours", Confidence="0.9")
    fd.resolver.default = ResolverOutput(kind="none")
    root = _balance_handoff(fd, sid(1))
    assert says(root)[0].startswith("I can help you check your balance.")
    assert "get_balance" not in fd.lam.ops()




def test_step_up_level1_to_3_asks_only_missing_factors(fd):
    fd.login_thomas(sid(1))
    fd.set_pending_tool(sid(1), "authenticate", {"tool": "make_payment"})
    root = fd.handoff(sid(1), {"reason": "step-up", "tool": "make_payment", "journal": "jr-2"})
    assert says(root) == [
        "I can help you make that payment. First I need to ask you a few more questions "
        "to verify your identity.",
        "Please say or enter the last four digits of your Social Security number. If you can't, press pound.",
    ]
    assert fd.state(sid(1)).step_up_target == "make_payment"
    root = fd.factor(sid(1), TEST_FACTORS["C-1001"]["ssn4"])
    assert fd.state(sid(1)).level == 2
    # "Got it." acknowledges the verified SSN4 before the next factor's prompt.
    assert says(root) == ["Got it.",
                          "We just sent a six-digit code to your phone by text message. "
                          "Please say or enter it now, or press pound if you didn't get it."]
    assert fd.verify.sent == [(THOMAS, "sms")]
    otp_send = [e for e in fd.lam.payloads if e.get("op") == "send_otp"][0]
    assert otp_send["ani"] == THOMAS
    root = fd.factor(sid(1), OTP)
    cr = relay(root)
    assert cr.get("welcomeGreeting") == "Thanks, you're verified."
    st = fd.state(sid(1))
    assert st.level == 3 and set(st.factors_verified) == {"ani_match", "zip", "ssn4", "otp"}
    env = fd.pending_result(sid(1))
    assert env.status == "verified" and env.next == "retry_original_tool"
    assert env.tool == "authenticate" and env.forTool == "make_payment"
    assert env.data == {"level": 3}
    assert [s.template for s in env.spoken] == ["stepup.intro", "factor.ssn4", "factor.ack",
                                                "factor.otp", "stepup.success"]
    # Only the missing factors were sent, one level at a time; ZIP was never asked again.
    verifies = [e for e in fd.lam.payloads if e.get("op") == "verify"]
    assert [sorted(v["factors"]) for v in verifies] == [["zip"], ["ssn4"], ["otp"]]
    assert [v["target_level"] for v in verifies] == [1, 2, 3]
    assert verifies[-1]["ani"] == THOMAS
    asked = [e["factor"] for e in fd.journal(sid(1)) if e["type"] == "factor_asked"]
    assert asked == ["zip", "ssn4", "otp"]
    # The upgraded, sealed token now passes a level-3 check; the old one is revoked.
    token = fd.sealer.unseal(st.sealed_token, call_sid=sid(1))
    assert not isinstance(tokens.validate(fd.store, token, call_sid=sid(1), min_level=3,
                                          now=int(CLOCK)), tokens.AuthError)


def test_step_up_wrong_ssn_fails_with_offer_alternative(fd):
    fd.login_thomas(sid(1))
    fd.set_pending_tool(sid(1), "authenticate", {"tool": "change_address"})
    fd.handoff(sid(1), {"reason": "step-up", "tool": "change_address"})
    root = fd.factor(sid(1), "9998")
    assert says(root)[0] == "Sorry, that didn't match our records. Let's try again."
    root = fd.factor(sid(1), "9997")  # auth service: stepup_attempts_per_call = 2
    assert relay(root).get("welcomeGreeting").startswith("I'm sorry, I wasn't able to verify")
    env = fd.pending_result(sid(1))
    assert env.status == "not_verified" and env.next == "offer_alternative"
    assert fd.state(sid(1)).level == 1


def test_step_up_unknown_tool_fails_closed_with_generic_phrase(fd):
    fd.login_thomas(sid(1))
    fd.set_pending_tool(sid(1), "authenticate", {"tool": "wire_to_offshore"})
    root = fd.handoff(sid(1), {"reason": "step-up"})
    assert says(root)[0].startswith("I can help with that.")
    assert fd.session(sid(1)).flow["target_level"] == 3  # max level


def test_step_up_when_level_already_sufficient_is_verified_at_once(fd):
    fd.login_thomas(sid(1))
    fd.set_pending_tool(sid(1), "authenticate", {"tool": "list_recent_transactions"})
    root = fd.handoff(sid(1), {"reason": "step-up", "tool": "list_recent_transactions"})
    assert relay(root).get("welcomeGreeting") == "Thanks, you're verified."
    assert fd.pending_result(sid(1)).next == "retry_original_tool"


def test_intent_switch_during_step_up_returns_interrupted_for_tool():
    h = build(resolver=FakeResolver([ResolverOutput(kind="intent_switch")]))
    h.login_thomas(sid(1))
    h.set_pending_tool(sid(1), "authenticate", {"tool": "make_payment"})
    h.handoff(sid(1), {"reason": "step-up", "tool": "make_payment"})
    root = h.xml("factor", sid(1),
                 SpeechResult="Actually, can you give me my checking balance instead?")
    assert relay(root).get("welcomeGreeting") == "One moment while I get that for you."
    env = h.pending_result(sid(1))
    assert env.status == "interrupted" and env.next == "handle_utterance"
    assert env.forTool == "make_payment" and env.completed is False
    assert env.callerUtterance == "Actually, can you give me my checking balance instead?"
    assert h.state(sid(1)).level == 1


def test_cancel_during_step_up_is_declined(fd):
    fd.login_thomas(sid(1))
    fd.set_pending_tool(sid(1), "authenticate", {"tool": "make_payment"})
    fd.handoff(sid(1), {"reason": "step-up", "tool": "make_payment"})
    fd.xml("factor", sid(1), SpeechResult="never mind")
    env = fd.pending_result(sid(1))
    assert env.status == "declined" and env.next == "offer_alternative"


def test_step_up_attempts_are_capped(fd):
    fd.login_thomas(sid(1))
    for _ in range(2):
        fd.set_pending_tool(sid(1), "authenticate", {"tool": "make_payment"})
        fd.handoff(sid(1), {"reason": "step-up", "tool": "make_payment"})
        fd.xml("factor", sid(1), SpeechResult="never mind")
    fd.set_pending_tool(sid(1), "authenticate", {"tool": "make_payment"})
    root = fd.handoff(sid(1), {"reason": "step-up", "tool": "make_payment"})
    assert root.find("Gather") is None
    assert fd.pending_result(sid(1)).status == "not_verified"




def test_escalation_enqueues_with_server_side_task_attributes(fd):
    fd.login_thomas(sid(1))
    summary = "Caller reports two card transactions he doesn't recognize.\x07 Wants the card blocked."
    fd.set_pending_tool(sid(1), "escalate", {"reason": "fraud_suspected", "summary": summary})
    root = fd.handoff(sid(1), {"reason": "escalate", "escalationReason": "fraud_suspected",
                               "journal": "jr-7f3a", "conversationId": "CH" + "f" * 32,
                               "profileId": "mem_profile_abc"})
    assert says(root) == ["I'll connect you with our fraud team right away."]
    enq = root.find("Enqueue")
    assert enq.get("workflowSid") == WORKFLOW and enq.get("waitUrl") is None
    assert enq.get("action") == ENQUEUE_ACTION_URL
    task = enq.find("Task")
    assert task.get("priority") == "100" and task.get("timeout") is None  # no queue time limit
    attrs = json.loads(task.text)
    assert attrs["escalation_reason"] == "fraud_suspected"
    assert attrs["ai_summary"] == ("Caller reports two card transactions he doesn't recognize. "
                                   "Wants the card blocked.")
    assert attrs["auth_level"] == 1 and attrs["customer_id"] == "C-1001"
    assert attrs["language"] == "en-US" and attrs["from"] == THOMAS
    assert attrs["call_sid"] == sid(1) and attrs["type"] == "inbound"
    assert attrs["factors_verified"] == ["ani_match", "zip"]
    assert attrs["conversation_id"] == "CH" + "f" * 32
    assert attrs["profile_id"] == "mem_profile_abc"
    assert attrs["journal_ref"] == "jr-7f3a" and attrs["test"] is False
    st = fd.state(sid(1))
    # The Flex Context API finds the call by this journal ref.
    assert st.escalation["journal_ref"] == attrs["journal_ref"] == "jr-7f3a"
    assert st.escalation["reason"] == "fraud_suspected"
    assert st.sealed_token is None
    assert all(t["revoked"] for t in fd.store.query("tokens", {"call_sid": sid(1)},
                                                    index="call_sid-index"))


def test_escalation_unknown_reason_routes_as_default_in_spanish(fd):
    fd.start(sid(1), MARIA, lang_digit="2")
    fd.factor(sid(1), TEST_FACTORS["C-1004"]["zip"])
    fd.set_pending_tool(sid(1), "escalate", {"reason": "made_up", "summary": "x"})
    root = fd.handoff(sid(1), {"reason": "escalate", "escalationReason": "made_up"})
    assert says(root) == ["Le comunico con un especialista que puede ayudarle. "
                          "Por favor, espere un momento."]
    attrs = json.loads(root.find("Enqueue/Task").text)
    assert attrs["escalation_reason"] == "unable_to_resolve"
    assert root.find("Enqueue").get("action") == ENQUEUE_ACTION_URL  # the language comes from call state
    ref = fd.state(sid(1)).escalation["journal_ref"]
    assert ref and ref == attrs["journal_ref"]  # generated when the handoff carried none


# --------------------------------------------------------------------------- failure paths


def test_websocket_failure_reconnects_once_then_escalates(fd):
    fd.login_thomas(sid(1))
    params = {"CallStatus": "in-progress", "SessionStatus": "failed", "ErrorCode": "64105"}
    root = fd.xml("connect-action", sid(1), **params)
    assert says(root) == ["I'm sorry, something went wrong on our end."]
    assert relay(root).get("welcomeGreeting") == "Is there anything else I can help you with?"
    root = fd.xml("connect-action", sid(1), **params)
    assert root.find("Enqueue") is not None
    assert json.loads(root.find("Enqueue/Task").text)["escalation_reason"] == "unable_to_resolve"


def test_ended_without_handoff_data_is_treated_as_a_drop(fd):
    fd.login_thomas(sid(1))
    root = fd.xml("connect-action", sid(1), CallStatus="in-progress", SessionStatus="ended")
    assert relay(root) is not None


def test_caller_hangup_on_connect_action_returns_empty(fd):
    fd.login_thomas(sid(1))
    resp = fd.post("connect-action", sid(1), CallStatus="completed", SessionStatus="completed")
    assert resp["statusCode"] == 200 and "<Response />" in resp["body"]


# --------------------------------------------------------------------------- status


def test_status_completed_revokes_marks_ended_journals_and_closes(fd):
    fd.login_thomas(sid(1))
    fd.handoff(sid(1), {"reason": "noise", "conversationId": "CH" + "a" * 32})  # learns the conv
    resp = fd.post("status", sid(1), CallStatus="completed", CallDuration="95")
    assert resp["statusCode"] == 200 and "<Response />" in resp["body"]
    assert all(t["revoked"] for t in fd.store.query("tokens", {"call_sid": sid(1)},
                                                    index="call_sid-index"))
    assert fd.session(sid(1)).ended is True and fd.state(sid(1)).sealed_token is None
    end = [e for e in fd.journal(sid(1)) if e["type"] == "call_end"]
    assert len(end) == 1 and end[0]["outcome"] == "completed" and end[0]["duration_s"] == 95
    assert fd.closer.closed == [(sid(1), "CH" + "a" * 32)]
    fd.post("status", sid(1), CallStatus="completed")  # idempotent
    assert len([e for e in fd.journal(sid(1)) if e["type"] == "call_end"]) == 1


def test_status_no_answer_does_not_close_conversation_and_non_final_is_ignored(fd):
    fd.start(sid(1), NOBODY)
    fd.post("status", sid(1), CallStatus="ringing")
    assert not [e for e in fd.journal(sid(1)) if e["type"] == "call_end"]
    fd.post("status", sid(1), CallStatus="no-answer")
    assert fd.session(sid(1)).ended and fd.closer.closed == []


def test_status_close_failure_is_logged_and_response_still_ok(caplog):
    h = build(closer=RecordingCloser(fail=True))
    h.login_thomas(sid(1))
    resp = h.post("status", sid(1), CallStatus="failed")
    assert resp["statusCode"] == 200 and "<Response />" in resp["body"]
    assert any(r.getMessage() == "conversation close failed" for r in caplog.records)
    assert h.session(sid(1)).ended


def test_failed_verification_is_logged_without_the_value(fd, caplog):
    """A misheard factor left no log line. Each verification outcome logs
    ``verification_result`` with the outcome and factor names, never the value."""
    import logging
    fd.login_thomas(sid(1))
    fd.set_pending_tool(sid(1), "authenticate", {"tool": "change_address"})
    fd.handoff(sid(1), {"reason": "step-up", "tool": "change_address"})
    with caplog.at_level(logging.INFO):
        caplog.clear()  # the login's own "verified" line came before
        fd.factor(sid(1), "9998")
    lines = [r for r in caplog.records if r.getMessage() == "verification_result"]
    assert [getattr(r, "outcome", None) for r in lines] == ["not_verified"]
    assert getattr(lines[0], "factors", None) == ["ssn4"]
    assert not any("9998" in r.getMessage() or "9998" in str(r.__dict__) for r in caplog.records)
