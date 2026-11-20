"""The ``otp_wait`` redirect after "Got it." ends in exactly one of {OTP prompt,
otp_unavailable, give-up}, never sends a code itself, and always clears ``otp_wait`` and
``otp_send``, whatever the auth service writes (or doesn't) and whenever Twilio follows the
redirect.

The asynchronous send is queued and never run (``defer_events``); instead the auth service's
``otp_send`` write is simulated at ``land_s`` seconds after the send started, on the front door's
fake clock. A stale ``otp_send`` may be present before the SSN-4 answer: it must never count."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from lambdas.frontdoor.app import OTP_WAIT_S
from owlbank import callstate
from owlbank.segments import clear_sensitive
from tests.component.test_frontdoor_support import CLOCK, TEST_FACTORS, THOMAS, Harness, build, sid

OTP_PROMPT = "We just sent a six-digit code to your phone by text message. Please say or enter it now, or press pound if you didn't get it."
FAILED = "I'm sorry, I wasn't able to verify your identity, so I can't do that right now. Is there anything else I can help you with?"
CALL = sid(23)

statuses = st.sampled_from([None, "sent", "escalate", "error", "bogus"])
stale = st.sampled_from([None, "sent", "escalate", "error"])
seconds = st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)


class Sim:
    """The front door's clock and sleep, with the auth service's write landing at ``land_at``."""

    def __init__(self, h: Harness, status: str | None, land_s: float) -> None:
        self.h, self.t = h, CLOCK
        self.status, self.land_at = status, CLOCK + land_s
        self.landed = False

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds
        if self.status is not None and not self.landed and self.t >= self.land_at:
            self.landed = True
            rec = {"status": self.status, "at": int(self.t)}
            callstate.mutate(self.h.store, CALL, lambda _s: {"otp_send": rec})


def _outcome(root: ET.Element) -> str:
    tags = [c.tag for c in root]
    if tags == ["Gather"] and [e.text for e in root.iter("Say")] == [OTP_PROMPT]:
        return "otp_prompt"
    if tags[-1:] == ["Enqueue"]:
        attrs: dict[str, Any] = json.loads(root.find("Enqueue/Task").text or "{}")  # type: ignore[union-attr]
        return "otp_unavailable" if attrs.get("escalation_reason") == "otp_unavailable" else "other"
    cr = root.find("./Connect/ConversationRelay")
    if cr is not None and cr.get("welcomeGreeting") == FAILED:
        return "give_up"
    return "other"


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(status=statuses, land_s=seconds, redirect_s=seconds, stale_status=stale)
def test_the_redirect_ends_in_exactly_one_outcome_and_clears_its_state(
        status: str | None, land_s: float, redirect_s: float, stale_status: str | None) -> None:
    # Keep clear of the exact deadline and of a landing at the redirect instant (float ties).
    assume(abs(land_s - OTP_WAIT_S) > 0.15 and abs(land_s - redirect_s) > 0.15)
    h = build()
    sim = Sim(h, status, land_s)
    h.app.d.clock = sim
    h.app.d.sleep = sim.advance
    h.lam.defer_events, h.lam.clock = True, sim
    try:
        h.start(CALL, THOMAS)
        h.factor(CALL, TEST_FACTORS["C-1001"]["zip"])
        h.set_pending_tool(CALL, "authenticate", {"tool": "make_payment"})
        h.handoff(CALL, {"reason": "step-up", "tool": "make_payment", "journal": "jr"})
        if stale_status is not None:
            callstate.mutate(h.store, CALL, lambda _s: {"otp_send": {"status": stale_status, "at": 1}})
        first = h.factor(CALL, TEST_FACTORS["C-1001"]["ssn4"], follow=False)
        assert [c.tag for c in first] == ["Say", "Redirect"]
        sends_before = h.lam.ops().count("send_otp")
        assert sends_before == 1 and h.lam.event_ops() == ["send_otp"]

        sim.advance(redirect_s)  # "Got it." plays; then Twilio follows the redirect
        root = h.xml("factor", CALL, follow=False)

        outcome = _outcome(root)
        landed_in_time = status is not None and (land_s < redirect_s or land_s < OTP_WAIT_S)
        expected = ("otp_prompt" if landed_in_time and status == "sent"
                    else "otp_unavailable" if landed_in_time and status == "escalate"
                    else "give_up")
        assert outcome == expected, (outcome, ET.tostring(root))
        # Never sends a code itself: no further send_otp invoke, and no code went out at all.
        assert h.lam.ops().count("send_otp") == sends_before and h.verify.sent == []
        # Always clears its state.
        st_ = h.state(CALL)
        assert st_.otp_send is None
        assert "otp_wait" not in (h.session(CALL).flow or {})
    finally:
        clear_sensitive(CALL)
