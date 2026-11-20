"""Front door: the one-time code goes only to the calling number, in the
caller's language; when it can't, a neutral phrase and a human queue (``otp_unavailable``)."""

from __future__ import annotations

import json

from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    MARIA,
    TEST_FACTORS,
    THOMAS,
    fd,
    says,
    sid,
)

EN_PHRASE = "For your security, I'll connect you with a specialist who can help finish this."
ES_PHRASE = "Por su seguridad, le comunico con un especialista que puede ayudarle a terminar este trámite."
#: the SSN-4 was verified, so its acknowledgement plays while the code is being sent, before
#: the auth service's answer is known (the harness follows the redirect, as Twilio does).
EN_ACK, ES_ACK = "Got it.", "Entendido."


def _step_up_to_otp(fd, call_sid, ani, cid, *, lang_digit=""):
    """Login at level 1, then a step-up to level 3; the SSN-4 answer leads to the OTP send."""
    fd.start(call_sid, ani, lang_digit=lang_digit)
    fd.factor(call_sid, TEST_FACTORS[cid]["zip"])
    fd.set_pending_tool(call_sid, "authenticate", {"tool": "make_payment"})
    fd.handoff(call_sid, {"reason": "step-up", "tool": "make_payment", "journal": "jr-r5"})
    return fd.factor(call_sid, TEST_FACTORS[cid]["ssn4"])


def test_the_call_language_goes_to_verify_as_the_locale(fd):
    _step_up_to_otp(fd, sid(1), MARIA, "C-1004", lang_digit="2")
    assert fd.verify.sent == [(MARIA, "sms")]
    assert fd.verify.locales == ["es-US"]
    send = [e for e in fd.lam.payloads if e.get("op") == "send_otp"][0]
    assert send["language"] == "es-US" and send["ani"] == MARIA


def test_english_call_sends_the_english_locale(fd):
    _step_up_to_otp(fd, sid(1), THOMAS, "C-1001")
    assert fd.verify.locales == ["en-US"]


def test_verify_delivery_failure_transfers_to_a_human_with_a_neutral_phrase(fd):
    fd.verify.fail = True
    root = _step_up_to_otp(fd, sid(1), THOMAS, "C-1001")
    assert says(root) == [EN_ACK, EN_PHRASE]
    attrs = json.loads(root.find("Enqueue/Task").text)
    assert attrs["escalation_reason"] == "otp_unavailable"
    assert attrs["auth_level"] == 2 and attrs["customer_id"] == "C-1001"  # the level reached stays
    assert fd.verify.sent == []
    st = fd.state(sid(1))
    assert st.escalation["reason"] == "otp_unavailable" and st.escalation["queue"] == "OwlBank Support EN"
    assert st.level == 2 and st.verified_customer_id == "C-1001"
    assert all(t["revoked"] for t in fd.store.query("tokens", {"call_sid": sid(1)},
                                                    index="call_sid-index"))  # AI session is over
    # The same path as every escalation: no second send was attempted, the flow is closed.
    assert len([e for e in fd.lam.payloads if e.get("op") == "send_otp"]) == 1


def test_the_neutral_phrase_and_queue_follow_the_callers_language(fd):
    fd.verify.fail = True
    root = _step_up_to_otp(fd, sid(1), MARIA, "C-1004", lang_digit="2")
    assert says(root) == [ES_ACK, ES_PHRASE]
    assert fd.state(sid(1)).escalation["queue"] == "OwlBank Support ES"
    assert root.find("Enqueue").get("action") is not None


def test_a_calling_number_with_no_record_for_the_customer_sends_nothing(fd):
    """Withheld, malformed or unrecorded: the auth service says escalate, nothing is sent."""
    from owlbank import callstate

    for n, ani in enumerate(("anonymous", "+12025550177", "")):
        call_sid = sid(10 + n)
        fd.start(call_sid, THOMAS)
        fd.factor(call_sid, TEST_FACTORS["C-1001"]["zip"])
        callstate.mutate(fd.store, call_sid, lambda s, ani=ani: {"ani": ani})  # the call's ANI as the front door holds it
        fd.set_pending_tool(call_sid, "authenticate", {"tool": "make_payment"})
        fd.handoff(call_sid, {"reason": "step-up", "tool": "make_payment", "journal": "jr-r5"})
        root = fd.factor(call_sid, TEST_FACTORS["C-1001"]["ssn4"])
        assert says(root) == [EN_ACK, EN_PHRASE]
        assert json.loads(root.find("Enqueue/Task").text)["escalation_reason"] == "otp_unavailable"
    assert fd.verify.sent == []
