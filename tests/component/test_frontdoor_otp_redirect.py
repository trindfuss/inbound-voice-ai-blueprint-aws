""""Got it." plays as soon as the SSN-4 is verified; the one-time code is sent while it plays.

After a verified factor whose next factor is the OTP, the front door starts the auth service's
``send_otp`` asynchronously (``Event`` invoke, ``record``), answers with only the acknowledgement
and a ``<Redirect>`` to ``/voice/factor``, and the redirect waits (up to 2.5 s since the send
started) for the auth service's ``otp_send`` in call state.

Signed fake webhooks into the real front door and the real auth service
(``test_frontdoor_support.build``). ``follow=False`` shows each response as Twilio receives it.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from typing import Any

import pytest

from lambdas.frontdoor import http
from lambdas.frontdoor.app import OTP_WAIT_S
from owlbank import callstate

from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    CLOCK,
    OTP,
    STAGE,
    TEST_FACTORS,
    THOMAS,
    Harness,
    build,
    fd,
    greeting,
    relay,
    says,
    sid,
)

EN = "en-US"
ACK = "Got it."
OTP_PROMPT = "We just sent a six-digit code to your phone by text message. Please say or enter it now, or press pound if you didn't get it."
OTP_UNAVAILABLE_PHRASE = ("For your security, I'll connect you with a specialist who can help "
                          "finish this.")
FAILED = "I'm sorry, I wasn't able to verify your identity, so I can't do that right now. Is there anything else I can help you with?"
S = TEST_FACTORS["C-1001"]["ssn4"]
Z = TEST_FACTORS["C-1001"]["zip"]


class FakeTime:
    """The front door's clock and sleep: sleeping advances the clock and lets the fake Lambda run
    the queued asynchronous invokes that are due (a slow auth service)."""

    def __init__(self, h: Harness, t: float = CLOCK) -> None:
        self.h, self.t = h, t
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds
        self.h.lam.run_deferred(self.t)


def timed(h: Harness, *, delay_s: float | None = None) -> FakeTime:
    """Give ``h`` a fake clock; with ``delay_s`` the async send lands that long after it starts."""
    t = FakeTime(h)
    h.app.d.clock = t
    h.app.d.sleep = t.sleep
    if delay_s is not None:
        h.lam.defer_events, h.lam.event_delay_s, h.lam.clock = True, delay_s, t
    return t


def step_up(h: Harness, call_sid: str) -> None:
    """Log Thomas in (ZIP), then a level-3 step-up for a payment: SSN-4 is asked, then the OTP."""
    h.start(call_sid, THOMAS)
    h.factor(call_sid, Z)
    h.set_pending_tool(call_sid, "authenticate", {"tool": "make_payment"})
    h.handoff(call_sid, {"reason": "step-up", "tool": "make_payment", "journal": "jr-23"})


def sends(h: Harness) -> list[str]:
    """``send_otp`` invokes, by invocation type."""
    return [kind for (_, op), kind in zip(h.lam.calls, h.lam.invocation_types, strict=True)
            if op == "send_otp"]


def prompt_keys(h: Harness, call_sid: str) -> list[str]:
    return [e["prompt_key"] for e in h.journal(call_sid) if e["type"] == "prompt"]


def assert_cleared(h: Harness, call_sid: str) -> None:
    assert h.state(call_sid).otp_send is None
    assert "otp_wait" not in (h.session(call_sid).flow or {})


def first_response(h: Harness, call_sid: str) -> ET.Element:
    return h.factor(call_sid, S, follow=False)


def redirect(h: Harness, call_sid: str) -> ET.Element:
    """The ``<Redirect>`` as Twilio follows it: a POST with the call's parameters, no input."""
    return h.xml("factor", call_sid, follow=False)


# --------------------------------------------------------------------------- the first response


def test_ssn_verified_answers_with_only_got_it_and_a_redirect(fd: Harness) -> None:
    step_up(fd, sid(1))
    root = first_response(fd, sid(1))
    assert fd.state(sid(1)).level == 2
    assert [c.tag for c in root] == ["Say", "Redirect"]
    assert says(root) == [ACK]
    red = root.find("Redirect")
    assert red is not None and red.get("method") == "POST"
    assert red.text == f"{STAGE}/voice/factor"
    assert root.find(".//Gather") is None
    assert sends(fd) == ["Event"]  # invoked once, asynchronously
    (payload,) = [p for p in fd.lam.payloads if p.get("op") == "send_otp"]
    assert payload["record"] is True and payload["ani"] == THOMAS and payload["language"] == EN
    flow = fd.session(sid(1)).flow
    assert flow["otp_wait"]["factor"] == "otp" and isinstance(flow["otp_wait"]["started_ms"], int)
    assert prompt_keys(fd, sid(1))[-1] == "factor.ack"


def test_redirect_with_the_code_sent_asks_the_otp_without_a_second_send_or_ack(
        fd: Harness, caplog: pytest.LogCaptureFixture) -> None:
    t = timed(fd)
    step_up(fd, sid(2))
    first_response(fd, sid(2))
    assert fd.state(sid(2)).otp_send["status"] == "sent"  # the auth service wrote it
    root = redirect(fd, sid(2))
    assert [c.tag for c in root] == ["Gather"]
    assert says(root) == [OTP_PROMPT]  # no second "Got it."
    assert sends(fd) == ["Event"] and fd.verify.sent == [(THOMAS, "sms")]  # no second send
    assert t.slept == []  # the answer was already there
    assert_cleared(fd, sid(2))
    ready = [r for r in caplog.records if r.getMessage() == "otp_send_ready"]
    assert len(ready) == 1 and ready[0].status == "sent" and ready[0].wait_ms == 0
    assert [r.getMessage() for r in caplog.records].count("otp_send_async") == 1
    # The journal and the envelope read as before the redirect: ack, then the code prompt.
    assert prompt_keys(fd, sid(2))[-2:] == ["factor.ack", "factor.otp"]
    fd.factor(sid(2), OTP)
    assert fd.state(sid(2)).level == 3
    env = fd.pending_result(sid(2))
    assert [s.template for s in env.spoken][1:4] == ["factor.ssn4", "factor.ack", "factor.otp"]


def test_a_send_that_lands_within_the_window_is_waited_for(
        fd: Harness, caplog: pytest.LogCaptureFixture) -> None:
    t = timed(fd, delay_s=1.2)
    step_up(fd, sid(3))
    first_response(fd, sid(3))
    assert fd.state(sid(3)).otp_send is None and fd.verify.sent == []  # still sending
    root = redirect(fd, sid(3))
    assert says(root) == [OTP_PROMPT]
    assert fd.verify.sent == [(THOMAS, "sms")]
    assert 1.2 <= sum(t.slept) < OTP_WAIT_S and max(t.slept) <= 0.1 + 1e-9
    (ready,) = [r for r in caplog.records if r.getMessage() == "otp_send_ready"]
    assert ready.status == "sent" and 1200 <= ready.wait_ms < 2500
    assert_cleared(fd, sid(3))


def test_escalate_goes_to_otp_unavailable(fd: Harness) -> None:
    fd.verify.fail = True  # Verify can't deliver: the auth service answers escalate
    step_up(fd, sid(4))
    first_response(fd, sid(4))
    assert fd.state(sid(4)).otp_send["status"] == "escalate"
    root = redirect(fd, sid(4))
    assert says(root) == [OTP_UNAVAILABLE_PHRASE]
    attrs = json.loads(root.find("Enqueue/Task").text)
    assert attrs["escalation_reason"] == "otp_unavailable" and attrs["auth_level"] == 2
    assert sends(fd) == ["Event"] and fd.verify.sent == []
    assert fd.state(sid(4)).otp_send is None


def test_error_takes_the_give_up_path(fd: Harness) -> None:
    def broken(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("verify client bug")  # the auth service answers unavailable

    fd.verify.send = broken  # type: ignore[method-assign]
    step_up(fd, sid(5))
    first_response(fd, sid(5))
    assert fd.state(sid(5)).otp_send["status"] == "error"
    root = redirect(fd, sid(5))
    assert says(root) == [] and greeting(root) == FAILED
    env = fd.pending_result(sid(5))
    assert env.status == "failed" and env.next == "offer_alternative"
    assert [s.template for s in env.spoken][-2:] == ["factor.ack", "stepup.failed"]
    assert sends(fd) == ["Event"]
    assert_cleared(fd, sid(5))


def test_no_answer_within_the_window_gives_up_with_a_warning(
        fd: Harness, caplog: pytest.LogCaptureFixture) -> None:
    t = timed(fd, delay_s=10.0)
    step_up(fd, sid(6))
    first_response(fd, sid(6))
    t.t += 0.8  # "Got it." plays before Twilio follows the redirect
    root = redirect(fd, sid(6))
    assert greeting(root) == FAILED and fd.pending_result(sid(6)).status == "failed"
    assert t.t - CLOCK == pytest.approx(OTP_WAIT_S, abs=0.01)  # waited to 2.5 s after the send started
    assert fd.verify.sent == []
    (warn,) = [r for r in caplog.records if r.getMessage() == "otp_send_timeout"]
    assert warn.levelno == logging.WARNING and warn.decision == "give_up"
    assert 2500 <= warn.wait_ms <= 2600
    assert not [r for r in caplog.records if r.getMessage() == "otp_send_ready"]
    assert_cleared(fd, sid(6))
    outcomes = [e.get("outcome") for e in fd.journal(sid(6)) if e["type"] == "verification"]
    assert "otp_send_timeout" in outcomes


def test_async_invoke_failure_falls_back_to_the_synchronous_send(fd: Harness) -> None:
    fd.lam.fail_event_ops.add("send_otp")
    step_up(fd, sid(7))
    root = first_response(fd, sid(7))
    # Exactly the response with the redirect removed: the ack and the prompt inside one OTP Gather.
    assert [c.tag for c in root] == ["Gather"]
    assert [e.text for e in root.find("Gather").iter("Say")] == [ACK, OTP_PROMPT]
    assert sends(fd) == ["Event", "RequestResponse"]
    assert fd.verify.sent == [(THOMAS, "sms")]
    assert_cleared(fd, sid(7))
    assert prompt_keys(fd, sid(7))[-2:] == ["factor.ack", "factor.otp"]
    fd.factor(sid(7), OTP)
    assert fd.state(sid(7)).level == 3


def test_a_stale_otp_send_is_cleared_before_the_send(fd: Harness) -> None:
    timed(fd, delay_s=1.0)
    step_up(fd, sid(8))
    callstate.mutate(fd.store, sid(8), lambda s: {"otp_send": {"status": "sent", "at": 1}})
    first_response(fd, sid(8))
    assert fd.state(sid(8)).otp_send is None  # not read as this send's answer
    root = redirect(fd, sid(8))
    assert says(root) == [OTP_PROMPT] and fd.verify.sent == [(THOMAS, "sms")]


def test_untrusted_caller_before_login_never_gets_a_code(fd: Harness) -> None:
    """The ``otp_before_login`` gate runs before any send, asynchronous or not."""
    call = sid(9)
    fd.xml("incoming", call, From=THOMAS, To="+12025550100", CallStatus="ringing")  # untrusted
    st, sess = fd.app._load(call)  # noqa: SLF001
    assert st.level == 0 and not st.ani_trusted
    sess.flow = {"kind": "stepup", "tool": "make_payment", "target_level": 3, "factor": None,
                 "tries": 0, "repeats": 0, "spoken": []}
    req = http.TwilioRequest("factor", {"CallSid": call}, STAGE)
    body = fd.app._ask(req, st, sess, "otp", pre=[fd.app._say("factor.ack", EN)],  # noqa: SLF001
                       ack_first=True)
    root = ET.fromstring(body.encode())
    attrs = json.loads(root.find("Enqueue/Task").text)
    assert attrs["escalation_reason"] == "verification_risk"
    assert sends(fd) == [] and fd.verify.sent == []
    assert "otp_wait" not in (fd.session(call).flow or {})


def test_an_otp_asked_first_without_an_ack_is_still_sent_synchronously(fd: Harness) -> None:
    """An untrusted level-1 caller stepping up to 2 is asked the OTP first (no ack): unchanged."""
    call = sid(10)
    fd.xml("incoming", call, From=THOMAS, To="+12025550100", CallStatus="ringing",
           StirVerstat="TN-Validation-Passed-C")
    fd.xml("language", call, Digits="")
    fd.factor(call, Z)
    fd.factor(call, S)
    fd.set_pending_tool(call, "authenticate", {"tool": "change_address"})
    root = fd.handoff(call, {"reason": "step-up", "tool": "change_address", "journal": "jr"})
    assert root.find("Gather") is not None and says(root)[-1] == OTP_PROMPT
    assert ACK not in says(root)
    assert sends(fd) == ["RequestResponse"]


def test_a_login_flow_is_unaffected(fd: Harness) -> None:
    root = fd.start(sid(11), THOMAS)
    assert [c.tag for c in root][0] == "Gather"
    root = fd.factor(sid(11), Z, follow=False)
    assert root.find("Redirect") is None
    assert relay(root).get("welcomeGreeting", "").startswith("Hi Thomas.")
    assert sends(fd) == [] and fd.lam.event_ops() == []


def test_digits_on_the_redirect_are_never_taken_as_the_code(fd: Harness) -> None:
    """No prompt was open, so input arriving with the redirect is ignored, never verified."""
    step_up(fd, sid(12))
    first_response(fd, sid(12))
    root = fd.xml("factor", sid(12), follow=False, Digits=OTP)
    assert says(root) == [OTP_PROMPT]
    assert fd.state(sid(12)).level == 2
    assert [p for p in fd.lam.payloads if p.get("op") == "verify"
            and "otp" in (p.get("factors") or {})] == []
    fd.sent_factors.add(OTP)


def test_spanish_step_up_acks_with_entendido_then_redirects() -> None:
    h = build()
    h.start(sid(13), "+12025550103", lang_digit="2")  # Maria
    h.factor(sid(13), TEST_FACTORS["C-1004"]["zip"])
    h.set_pending_tool(sid(13), "authenticate", {"tool": "make_payment"})
    h.handoff(sid(13), {"reason": "step-up", "tool": "make_payment", "journal": "jr"})
    root = h.factor(sid(13), TEST_FACTORS["C-1004"]["ssn4"], follow=False)
    assert says(root) == ["Entendido."] and root.find("Say").get("language") == "es-US"
    assert root.find("Redirect") is not None


# --------------------------------------------------------------------------- the wire


def test_send_otp_async_is_an_event_invoke_with_record_and_reports_failures() -> None:
    from lambdas.frontdoor.backend import LambdaBackendClient
    from lambdas.frontdoor.fakes import FakeLambdaClient

    seen: list[dict[str, Any]] = []
    lam = FakeLambdaClient({"auth": seen.append, "bank": seen.append})
    client = LambdaBackendClient(lam, auth_function="auth", bank_function="bank")
    kw = {"call_sid": sid(30), "ani": THOMAS, "language": EN, "stir_verstat": None}
    assert client.send_otp_async(**kw) is True
    assert lam.calls == [("auth", "send_otp")] and lam.invocation_types == ["Event"]
    # An ANI call's event has no identity_source (absent = ani) ...
    assert seen == [{"op": "send_otp", **kw, "record": True}]
    assert client.send_otp_async(call_sid=sid(30), ani=THOMAS, language=EN, stir_verstat=None,
                                 identity_source="keyed_phone") is True  # ... a keyed one has
    assert seen[-1] == {"op": "send_otp", **kw, "identity_source": "keyed_phone", "record": True}
    seen.clear()
    lam.fail_event_ops.add("send_otp")
    assert client.send_otp_async(**kw) is False  # throttled, timed out: send synchronously

    class NotAccepted:
        def invoke(self, **_kw: Any) -> dict[str, Any]:
            return {"StatusCode": 200}

    assert LambdaBackendClient(NotAccepted(), auth_function="a", bank_function="b") \
        .send_otp_async(**kw) is False
