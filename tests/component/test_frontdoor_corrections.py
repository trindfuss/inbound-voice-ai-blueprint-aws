"""Corrections in the front door.

* A correction of a factor collected in this flow and not yet sent to the auth service replaces its
  held value (sealed), is confirmed (``slot.corrected`` with the digits, or
  ``slot.corrected_sensitive`` with no value) and the interrupted slot is asked again.
* A correction and an answer in one breath: the correction is applied, the current slot is asked
  again and nothing is pre-filled.
* A correction whose value fails is re-asked once for the corrected slot, then the interrupted
  slot resumes.
* A correction after verification is re-asked, then goes to the AI; no ``verify`` call and
  ``level``, ``attempts``, ``stepups``, the risk counters and ``factors_verified`` unchanged.
* A correction naming an earlier flow's slot: re-asked, and twice is an intent switch.
* No corrections after an API call in the flow: a slot collected before the flow's
  last API call (any of them, through ``SlotFlow.api_call``) is handled as a finished flow's slot;
  the untrusted ZIP stays correctable at the SSN prompt; the card until ``make_payment``.
* Journal: each applied correction writes its own ``slot_resolution`` (``correction: true``,
  ``correction_of``), and the original answer's event stays.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from typing import Any

import pytest

from lambdas.auth_service import service as auth_service
from lambdas.frontdoor import state as sessions
from owlbank import segments
from owlbank.policy import Policy
from owlbank.slots import FakeResolver, ResolverOutput
from tests.component.test_frontdoor_caller_id_trust import S, Z
from tests.component.test_frontdoor_support import (
    CLOCK,
    DATA,
    FAILED_AI,
    OTP,
    THOMAS,
    Harness,
    assert_no_leak,
    build,
    relay,
    says,
    sid,
)

ZIP_PROMPT = "Please say or enter your five-digit ZIP code. If you can't, press pound."
SSN_PROMPT = "Please say or enter the last four digits of your Social Security number. If you can't, press pound."
ZIP_CLARIFY = "Please enter the 5 digits of your ZIP code on your keypad."
SSN_CLARIFY = "Please enter the 4 digits of your Social Security number on your keypad."
OTP_CLARIFY = "Please enter the 6 digits of your one-time code on your keypad."
ZIP_CHANGED = "Okay, I've changed that to nine, four, one, zero, five."
SSN_UPDATED = "Okay, I've updated your Social Security number."
WRONG_ZIP = "98101"   # Sam's ZIP: a real ZIP, but not Thomas's
WRONG_SSN = "9999"


def correction(slot: str, value: str | None) -> ResolverOutput:
    return ResolverOutput(kind="correction", slot=slot, value=value)


def speak(h: Harness, call: str, text: str) -> ET.Element:
    return h.xml("factor", call, SpeechResult=text, Confidence="0.9")


def verifies(h: Harness) -> list[dict[str, Any]]:
    return [p for p in h.lam.payloads if p.get("op") == "verify"]


def resolutions(h: Harness, call: str) -> list[dict[str, Any]]:
    return [e for e in h.journal(call) if e["type"] == "slot_resolution"]


def held(h: Harness, call: str) -> dict[str, str]:
    """The held factors, unsealed."""
    raw = (h.session(call).flow or {}).get("held") or {}
    return {k: h.sealer.unseal(v, call_sid=call) for k, v in raw.items()}


def untrusted_start(h: Harness, call: str) -> ET.Element:
    """An untrusted caller ID (no StirVerstat): level 1 needs the next level's factors, so the
    first one is held, sealed, until the second is collected."""
    h.xml("incoming", call, From=THOMAS, To="+12025550100", CallStatus="ringing")
    return h.xml("language", call, Digits="")


def ssn_first(h: Harness) -> Harness:
    """Level 1 is ``ani_match`` + SSN last four and level 2 adds the ZIP, so an untrusted caller is
    asked the SSN last four first and it is held while the ZIP is asked."""
    raw = json.loads((DATA / "policy.json").read_text())
    raw["levels"] = {"1": ["ani_match", "ssn4"], "2": ["ani_match", "ssn4", "zip"],
                     "3": ["ani_match", "ssn4", "zip", "otp"]}
    policy = Policy.from_dict(raw)
    h.app.d.policy = policy
    deps = auth_service.AuthDeps(store=h.store, policy=policy, verify_client=h.verify,
                                 clock=lambda: int(CLOCK))
    h.lam.functions["owlbank-auth"] = lambda e: auth_service.handle_event(deps, e)  # type: ignore[index]
    return h


def counters(h: Harness, call: str) -> dict[str, Any]:
    """Everything a post-verification correction never changes."""
    st = h.state(call)
    risk = sorted(json.dumps(v, sort_keys=True, default=str)
                  for v in h.store._tables.get("risk", {}).values())  # noqa: SLF001
    limits = sorted(str(k) for k in h.store._tables["calls"]  # noqa: SLF001
                    if "ratelimit" in str(k[0]))
    return {"level": st.level, "attempts": dict(st.attempts), "stepups": h.session(call).stepups,
            "factors_verified": tuple(st.factors_verified), "risk": risk, "limits": limits,
            "verify_calls": len(verifies(h))}


@pytest.fixture(autouse=True)
def _forget_sensitive():
    yield
    for n in range(40):
        segments.clear_sensitive(sid(n))




def test_a_zip_correction_at_the_ssn_prompt_replaces_the_held_zip_and_re_asks_the_ssn() -> None:
    h = build(resolver=FakeResolver([correction("zip", Z)]))
    call = sid(1)
    untrusted_start(h, call)
    assert says(h.factor(call, WRONG_ZIP)) == ["Got it.", SSN_PROMPT]
    assert held(h, call) == {"zip": WRONG_ZIP}
    root = speak(h, call, "actually the ZIP should be 94105")
    assert says(root) == [ZIP_CHANGED, SSN_PROMPT]
    assert root.find(".//Gather").get("numDigits") == "4"   # the interrupted slot's Gather
    assert held(h, call) == {"zip": Z} and verifies(h) == []
    root = h.factor(call, S)
    assert relay(root).get("welcomeGreeting").startswith("Hi Thomas.")
    (v,) = verifies(h)
    assert sorted(v["factors"]) == ["ssn4", "zip"]
    assert h.state(call).level == 1   # the corrected ZIP was the one sent


def test_the_correction_writes_its_own_slot_resolution_and_the_original_stays() -> None:
    h = build(resolver=FakeResolver([correction("zip", Z)]))
    call = sid(2)
    untrusted_start(h, call)
    h.factor(call, WRONG_ZIP)
    speak(h, call, "actually the ZIP should be 94105")
    events = resolutions(h, call)
    original, turn, fix = events
    assert original["slot"] == "zip" and original["value"] == WRONG_ZIP
    assert original["correction"] is False
    assert turn["slot"] == "ssn4" and turn["match"] == "correction" and "value" not in turn
    assert fix["slot"] == "zip" and fix["correction"] is True and fix["correction_of"] == "ssn4"
    assert fix["value"] == Z and fix["match"] == "correction" and fix["tries"] == 1
    assert fix["effective_mode"] == "assisted"


def test_a_correction_of_ssn4_says_corrected_sensitive_and_no_value_anywhere(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    h = ssn_first(build(resolver=FakeResolver([correction("ssn4", S)])))
    call = sid(3)
    assert says(untrusted_start(h, call))[-1] == SSN_PROMPT
    assert says(h.factor(call, WRONG_SSN)) == ["Got it.", ZIP_PROMPT]
    root = speak(h, call, "sorry, my last four are one one one one")
    assert says(root) == [SSN_UPDATED, ZIP_PROMPT]
    assert held(h, call) == {"ssn4": S}
    st = h.state(call)
    assert len(st.sealed_sensitive) == 2   # the new value remembered; the old one stays
    root = h.factor(call, Z)
    assert relay(root) is not None and h.state(call).level == 1
    fix = next(e for e in resolutions(h, call) if e["correction"])
    assert fix["slot"] == "ssn4" and fix["correction_of"] == "zip" and "value" not in fix
    assert_no_leak(h, caplog.records, {S, WRONG_SSN})
    texts = [*h.bodies, *h.stored_strings(), *(r.getMessage() for r in caplog.records)]
    assert not [t for t in texts if "one one one one" in t]   # nor the caller's words


def test_a_correction_and_an_answer_in_one_breath_re_asks_the_current_slot() -> None:
    h = build(resolver=FakeResolver([correction("zip", "12345")]))
    call = sid(4)
    untrusted_start(h, call)
    h.factor(call, WRONG_ZIP)
    # The sentence names the ZIP's label, so the guard leaves it to the resolver. With
    # "the ZIP is ..." the guard reads 6789 as the answer.
    root = speak(h, call, "the ZIP code is 12345 and the last four are 6789")
    assert len(h.resolver.calls) == 1
    assert says(root) == ["Okay, I've changed that to one, two, three, four, five.", SSN_PROMPT]
    assert held(h, call) == {"zip": "12345"}            # no SSN pre-filled
    assert [c["name"] for c in h.session(call).collected] == ["zip"]
    assert verifies(h) == [] and h.state(call).sealed_sensitive == ()


def test_a_correction_that_fails_is_re_asked_once_then_the_interrupted_slot_resumes() -> None:
    h = build(resolver=FakeResolver([correction("zip", "12345")]))  # not in the words
    call = sid(5)
    untrusted_start(h, call)
    h.factor(call, WRONG_ZIP)
    root = speak(h, call, "oh wait, my zip code is different")
    assert says(root) == [ZIP_CLARIFY]
    assert root.find(".//Gather").get("numDigits") == "5"  # the corrected slot's Gather
    assert held(h, call) == {"zip": WRONG_ZIP}
    root = h.factor(call, Z)
    assert says(root) == [ZIP_CHANGED, SSN_PROMPT]
    assert held(h, call) == {"zip": Z}
    flow = h.session(call).flow or {}
    assert flow["tries"] == {"zip": 1, "ssn4": 1} and flow["pending"] is None
    h.factor(call, S)
    assert h.state(call).level == 1


def test_a_failed_correction_re_ask_answered_badly_resumes_unchanged() -> None:
    h = build(resolver=FakeResolver([correction("zip", "12345")]))
    call = sid(6)
    untrusted_start(h, call)
    h.factor(call, WRONG_ZIP)
    speak(h, call, "oh wait, my zip code is different")
    root = h.factor(call, "12")                         # not a ZIP
    assert says(root) == [SSN_PROMPT]
    assert held(h, call) == {"zip": WRONG_ZIP}
    assert not [e for e in resolutions(h, call) if e["correction"]]


def test_corrections_use_tries_and_give_up_like_wrong_answers() -> None:
    h = build(resolver=FakeResolver(default=correction("zip", Z)))
    call = sid(7)
    untrusted_start(h, call)
    h.factor(call, WRONG_ZIP)
    for _ in range(2):   # maxRetries 2
        assert says(speak(h, call, "actually my zip is 94105")) == [ZIP_CHANGED, SSN_PROMPT]
    root = speak(h, call, "actually my zip is 94105")
    assert relay(root).get("welcomeGreeting") == FAILED_AI  # the AI by default
    assert verifies(h) == []




def _stepup_at_otp(h: Harness, call: str) -> None:
    """Thomas at level 2 (SSN last four verified in this step-up), asked the code."""
    h.login_thomas(call)
    h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    h.handoff(call, {"reason": "step-up", "tool": "make_payment"})
    h.factor(call, S)
    assert h.state(call).level == 2
    assert (h.session(call).flow or {}).get("factor") == "otp"


def test_a_correction_after_verification_is_re_asked_then_goes_to_the_ai() -> None:
    h = build(resolver=FakeResolver(default=correction("ssn4", "2222")))
    call = sid(8)
    _stepup_at_otp(h, call)
    before = counters(h, call)
    root = speak(h, call, "actually my last four are two two two two")
    assert says(root) == [OTP_CLARIFY]                    # unclear for the code
    assert counters(h, call) == before
    root = speak(h, call, "no really, my last four are two two two two")
    assert relay(root) is not None                        # an intent switch: the AI
    env = h.pending_result(call)
    assert env.status == "interrupted" and env.callerUtterance is not None
    assert "2222" not in env.callerUtterance and "two two two two" not in env.callerUtterance
    assert counters(h, call) == before                    # still nothing changed
    assert not [e for e in resolutions(h, call) if e["correction"]]
    assert held(h, call) == {}


def test_a_held_factor_already_sent_is_not_corrected() -> None:
    """The household's shared ZIP was sent and is held for the disambiguation: it was sent to the
    auth service, so it is no longer correctable: re-asked, nothing replaced."""
    h = build(resolver=FakeResolver([correction("zip", "98102")]))
    jordan = h.store.get("customers", {"customer_id": "C-1003"})
    sam = h.store.get("customers", {"customer_id": "C-1002"})
    jordan["factors"]["zip"] = sam["factors"]["zip"]
    h.store.put("customers", jordan)
    call = sid(9)
    h.start(call, "+12025550102")
    h.factor(call, "98101")
    before, sealed = counters(h, call), held(h, call)
    assert says(speak(h, call, "actually my zip is 98102")) == [SSN_CLARIFY]
    assert counters(h, call) == before and held(h, call) == sealed




def test_a_correction_naming_an_earlier_flows_slot_twice_is_an_intent_switch() -> None:
    h = build(resolver=FakeResolver(default=correction("zip", Z)))
    call = sid(10)
    h.login_thomas(call)
    h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    h.handoff(call, {"reason": "step-up", "tool": "make_payment"})
    before = counters(h, call)
    assert says(speak(h, call, "actually my zip is 94105")) == [SSN_CLARIFY]
    assert (h.session(call).flow or {})["earlier_strikes"] == 1
    root = speak(h, call, "I said my zip is 94105")
    assert relay(root) is not None
    assert h.pending_result(call).status == "interrupted"
    assert counters(h, call) == before


def test_an_answer_between_earlier_flow_corrections_resets_the_strikes() -> None:
    h = build(resolver=FakeResolver([correction("zip", Z), ResolverOutput(kind="none")]))
    call = sid(11)
    h.login_thomas(call)
    h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    h.handoff(call, {"reason": "step-up", "tool": "make_payment"})
    speak(h, call, "actually my zip is 94105")
    assert (h.session(call).flow or {})["earlier_strikes"] == 1
    assert says(speak(h, call, "hmm let me look")) == [SSN_CLARIFY]
    flow = h.session(call).flow or {}
    assert flow["earlier_strikes"] == 0 and flow["tries"]["ssn4"] == 2   # "consecutive"




@pytest.mark.parametrize("target", ["ssn4", "otp", "card", "nope"])
def test_a_correction_of_an_unasked_slot_is_none_and_stores_nothing(target: str) -> None:
    h = build(resolver=FakeResolver([correction(target, "1111")]))
    call = sid(12)
    untrusted_start(h, call)
    root = speak(h, call, "actually my last four are 1111")
    assert says(root) == [ZIP_CLARIFY]
    sess = h.session(call)
    assert sess.collected == [] and "held" not in (sess.flow or {})
    assert h.state(call).sealed_sensitive == ()
    assert set((sess.flow or {})["tries"]) == {"zip"}


def test_otp_spoken_at_the_otp_prompt_still_verifies_after_a_correction_attempt() -> None:
    h = build(resolver=FakeResolver([correction("ssn4", "2222")]))
    call = sid(13)
    _stepup_at_otp(h, call)
    speak(h, call, "actually my last four are two two two two")
    h.factor(call, OTP)
    assert h.state(call).level == 3


def test_an_untrusted_caller_correcting_a_sent_factor_changes_no_risk_counter() -> None:
    """A failed attempt from an untrusted caller ID wrote a risk counter; correcting the SSN last
    four that was sent with it changes none of the counters and calls nothing."""
    h = build(resolver=FakeResolver(default=correction("ssn4", S)))
    call = sid(14)
    untrusted_start(h, call)
    h.factor(call, WRONG_ZIP)
    root = h.factor(call, WRONG_SSN)
    assert says(root)[-1] == ZIP_PROMPT and len(verifies(h)) == 1     # not verified: again
    before = counters(h, call)
    assert before["risk"] != [] and before["attempts"] != {}
    assert says(speak(h, call, "wait, my last four are one one one one")) == [ZIP_CLARIFY]
    assert counters(h, call) == before
    root = speak(h, call, "my last four are one one one one I said")
    assert relay(root) is not None and counters(h, call) == before


def test_repeat_during_a_correction_re_ask_repeats_the_corrected_slots_question() -> None:
    h = build(resolver=FakeResolver([correction("zip", "12345")]))
    call = sid(15)
    untrusted_start(h, call)
    h.factor(call, WRONG_ZIP)
    speak(h, call, "oh wait, my zip code is different")
    tries = dict((h.session(call).flow or {})["tries"])
    root = speak(h, call, "repeat")
    assert says(root) == [ZIP_CLARIFY] and root.find(".//Gather").get("numDigits") == "5"
    assert (h.session(call).flow or {})["tries"] == tries         # a repeat uses no try
    assert says(h.factor(call, Z)) == [ZIP_CHANGED, SSN_PROMPT]




def _api_call(h: Harness, call: str) -> None:
    """An API call the front door makes inside the current flow (through ``SlotFlow.api_call``,
    the one helper every call site uses), e.g. a ``send_otp`` between two factors."""
    sess = h.session(call)
    h.app.slots.api_call(sess, lambda: None)
    sessions.save(h.store, sess)


def _row(h: Harness, call: str, name: str) -> dict[str, Any]:
    return next(r for r in h.session(call).collected if r["name"] == name)


def test_the_untrusted_zip_stays_correctable_at_the_ssn_prompt() -> None:
    """No API call happens between the held ZIP and the SSN prompt (no code before
    login), so the ZIP was collected after the flow's last API call."""
    h = build(resolver=FakeResolver([correction("zip", Z)]))
    call = sid(16)
    untrusted_start(h, call)
    h.factor(call, WRONG_ZIP)
    flow = h.session(call).flow or {}
    assert _row(h, call, "zip")["at_api_call"] == flow["api_calls"]
    assert says(speak(h, call, "actually the ZIP should be 94105")) == [ZIP_CHANGED, SSN_PROMPT]
    assert held(h, call) == {"zip": Z}


def test_a_held_factor_is_not_correctable_after_an_api_call_in_the_flow() -> None:
    """Once any API call has happened in the flow, the ZIP held before it is handled as a
    finished flow's slot: re-asked, nothing replaced, and twice an intent switch."""
    h = build(resolver=FakeResolver(default=correction("zip", Z)))
    call = sid(17)
    untrusted_start(h, call)
    h.factor(call, WRONG_ZIP)
    _api_call(h, call)
    assert _row(h, call, "zip")["verified"] is False       # not sent: only the call locks it
    before = counters(h, call)
    assert says(speak(h, call, "actually the ZIP should be 94105")) == [SSN_CLARIFY]
    assert held(h, call) == {"zip": WRONG_ZIP}
    assert (h.session(call).flow or {})["earlier_strikes"] == 1
    root = speak(h, call, "I said the ZIP should be 94105")
    assert relay(root) is not None and h.pending_result(call).status == "interrupted"
    assert counters(h, call) == before
    assert not [e for e in resolutions(h, call) if e["correction"]]


def test_a_factor_held_after_an_ambiguous_verify_is_not_correctable() -> None:
    """The shared ZIP was sent with the ambiguous ``verify`` (an API call) and is
    held for the disambiguation. It isn't correctable even if its row didn't say it was sent:
    the API call alone locks it."""
    h = build(resolver=FakeResolver([correction("zip", "98102")]))
    jordan = h.store.get("customers", {"customer_id": "C-1003"})
    sam = h.store.get("customers", {"customer_id": "C-1002"})
    jordan["factors"]["zip"] = sam["factors"]["zip"]
    h.store.put("customers", jordan)
    call = sid(18)
    h.start(call, "+12025550102")
    h.factor(call, "98101")
    sess = h.session(call)
    flow = sess.flow or {}
    assert len(verifies(h)) == 1 and flow["api_calls"] >= 1 and "zip" in flow["held"]
    assert _row(h, call, "zip")["at_api_call"] < flow["api_calls"]
    for row in sess.collected:          # take away the "sent" mark: the rule doesn't need it
        row["verified"] = False
    sessions.save(h.store, sess)
    before, sealed = counters(h, call), held(h, call)
    assert says(speak(h, call, "actually my zip is 98102")) == [SSN_CLARIFY]
    assert counters(h, call) == before and held(h, call) == sealed


def test_a_slot_collected_after_the_api_call_stays_correctable() -> None:
    """The counter only locks what came before the call."""
    h = ssn_first(build(resolver=FakeResolver([correction("ssn4", S)])))
    call = sid(19)
    untrusted_start(h, call)
    _api_call(h, call)                   # before anything is collected
    h.factor(call, WRONG_SSN)
    assert says(speak(h, call, "sorry, my last four are one one one one")) == [SSN_UPDATED,
                                                                                ZIP_PROMPT]
    assert held(h, call) == {"ssn4": S}


def test_a_re_collected_slot_keeps_its_first_at_api_call() -> None:
    """Fail safe: collecting a slot again in the same flow never makes it correctable again."""
    from lambdas.frontdoor import slotflow

    h = build()
    call = sid(20)
    untrusted_start(h, call)
    h.factor(call, WRONG_ZIP)
    first = _row(h, call, "zip")["at_api_call"]
    _api_call(h, call)
    sess = h.session(call)
    slotflow.collect(sess, "zip", "digits", verified=False)
    (row,) = [r for r in sess.collected if r["name"] == "zip"]
    assert row["at_api_call"] == first and (sess.flow or {})["api_calls"] == first + 1
    assert not slotflow.fresh(sess, "zip")


def test_every_backend_call_in_a_call_goes_through_the_helper() -> None:
    """One helper wraps every API call the front door makes (``precheck``, ``verify``,
    ``send_otp``, ``make_payment`` ...): the backend sees no call the helper didn't count."""
    from lambdas.frontdoor import slotflow

    seen: list[str] = []
    real = slotflow.SlotFlow.api_call

    def spy(self: Any, sess: Any, op: Any) -> Any:
        before = len(h.lam.payloads)
        out = real(self, sess, op)
        seen.extend(str(p.get("op")) for p in h.lam.payloads[before:])
        return out

    h = build()
    h.app.slots.api_call = spy.__get__(h.app.slots)          # type: ignore[method-assign]
    call = sid(21)
    untrusted_start(h, call)
    h.factor(call, Z)
    h.factor(call, S)
    h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    h.handoff(call, {"reason": "step-up", "tool": "make_payment"})
    h.factor(call, OTP)
    ops = [str(p.get("op")) for p in h.lam.payloads]
    assert ops and sorted(seen) == sorted(ops), (seen, ops)


def test_the_card_is_correctable_in_the_read_back_until_an_api_call() -> None:
    """No API call happens between the card and ``make_payment``, so a card correction in the
    read-back applies; once the flow has made one, the same correction changes nothing."""
    from tests.component.test_frontdoor_card import RED, pending_to, propose, readback, to_level3

    out = ResolverOutput(kind="correction", slot="card", value="red")
    h = build(resolver=FakeResolver([out]))
    call = sid(22)
    to_level3(h, call)
    propose(h, call, "purple")
    assert says(speak(h, call, "no, the red one")) == [f"Okay, I've changed that to {RED}.",
                                                       readback(RED)]
    assert pending_to(h, call) == "red"

    h = build(resolver=FakeResolver([out]))
    call = sid(23)
    to_level3(h, call)
    propose(h, call, "purple")
    _api_call(h, call)
    root = speak(h, call, "no, the red one")
    assert pending_to(h, call) == "purple"
    assert says(root)[-1].startswith("To confirm, you'd like to pay")
    assert not [e for e in resolutions(h, call) if e["correction"]]
    assert not [p for p in h.lam.payloads if p.get("op") == "make_payment"]
