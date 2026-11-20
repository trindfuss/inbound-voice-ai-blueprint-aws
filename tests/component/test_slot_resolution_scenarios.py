"""The slot-resolution scenarios, each in ``deterministic``, ``assisted`` and ``generative``.

Signed fake webhooks into the real front door with the real auth service and bank tools
(``test_frontdoor_support.build``) and ``FakeResolver`` for the model. The modes:

* ``deterministic``: the running ceiling is ``deterministic``; the resolver is never called.
* ``assisted``: the seed policy (``default_mode: assisted``, redaction enabled).
* ``generative``: the ``generative`` ceiling, ``generative_enabled`` and ``default_mode:
  generative``, so the card (a non-sensitive choice slot) is generative and every digit or sensitive
  slot is capped to ``assisted``.

C-1001 (Thomas) holds the Red and Purple cards; C-1002 (Sam) the Rewards card only.

Recorded behaviours, not defects: "the one that's purple" is an exact
deterministic match under the carrier trimming in every mode; the label
guard makes "the ZIP is 12345 and the last four are 6789" an SSN answer. The row "no, I meant the
red one" can't be reached as such in ``deterministic``: no inferred card is ever confirmed there, so
"Did you mean ...?" is never asked; the words alone are unclear there.

The corrected slot's re-ask follows the slot's rules (the ZIP read back, a card named
or confirmed), end to end through the front door.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from typing import Any

import pytest

from owlbank import sensitive
from owlbank.policy import Policy
from owlbank.segments import clear_sensitive
from owlbank.slots import FakeResolver, ResolverOutput
from scripts.hash_tool import hash_factor
from tests.component.test_frontdoor_card import (
    PROMPT,
    PURPLE,
    RED,
    REWARDS,
    YES_NO,
    paid_with,
    payments,
    pending_to,
    propose,
    readback,
    resolutions,
    speak,
    to_level3,
)
from tests.component.test_frontdoor_support import (
    DATA,
    OTP,
    TEST_FACTORS,
    THOMAS,
    Harness,
    assert_no_leak,
    build,
    relay,
    says,
    sid,
)

SEED = json.loads((DATA / "policy.json").read_text())
MODES = ("deterministic", "assisted", "generative")
EN = "en-US"

Z, S = TEST_FACTORS["C-1001"]["zip"], TEST_FACTORS["C-1001"]["ssn4"]
ZIP_PROMPT = "Please say or enter your five-digit ZIP code. If you can't, press pound."
ZIP_CLARIFY = "Please enter the 5 digits of your ZIP code on your keypad."
SSN_PROMPT = "Please say or enter the last four digits of your Social Security number. If you can't, press pound."
SSN_CLARIFY = "Please enter the 4 digits of your Social Security number on your keypad."
DISAMBIG = f"Which one did you mean: {RED} or {PURPLE}?"
CONFIRM_PURPLE = f"Did you mean the {PURPLE}?"
NOT_AVAILABLE_ONE = (f"I don't see an OwlBank Purple Card on your profile. You have the {REWARDS}. "
                     "Would you like to use that?")
PROMPT_SAM = f"Which card would you like to pay: your {REWARDS}?"


def confirm_digits(words: str) -> str:
    return f"Did you mean {words}?"


@pytest.fixture(autouse=True)
def _forget_calls() -> Any:
    yield
    for n in range(80):
        clear_sensitive(sid(300 + n))
        sensitive.forget(sid(300 + n))


def harness(mode: str, *outs: ResolverOutput, default: ResolverOutput | None = None) -> Harness:
    """The front door in ``mode`` (module docstring), the resolver answering ``outs`` in order."""
    kw: dict[str, Any] = {}
    raw = json.loads(json.dumps(SEED))
    if mode == "deterministic":
        kw["slot_mode_ceiling"] = "deterministic"
    elif mode == "generative":
        kw["slot_mode_ceiling"] = "generative"
        raw["slot_resolution"].update(generative_enabled=True, default_mode="generative")
    h = build(resolver=FakeResolver(list(outs), default=default), config_kw=kw)
    h.app.d.policy = Policy.from_dict(raw)
    return h


def effective_modes(h: Harness, call: str, slot: str) -> set[str]:
    return {e["effective_mode"] for e in resolutions(h, call) if e["slot"] == slot}


def at_card_question(h: Harness, n: int, who: str = "C-1001") -> str:
    call = sid(300 + n)
    to_level3(h, call, who)
    if who == "C-1001":
        assert says(propose(h, call, absent=True)) == [PROMPT]
    else:
        # Sam's one card needs no card question; it is asked after a "no" to the
        # not-available question about the card the AI named (one try used).
        assert says(propose(h, call, "purple")) == [NOT_AVAILABLE_ONE]
        assert says(h.xml("factor", call, Digits="2")) == [PROMPT_SAM]
    return call


def at_read_back(h: Harness, n: int, to: str = "purple") -> str:
    call = sid(300 + n)
    to_level3(h, call)
    assert says(propose(h, call, to)) == [readback(PURPLE if to == "purple" else RED)]
    return call


def at_ssn(h: Harness, n: int) -> str:
    """Thomas logged in, the step-up for ``change_address`` asking the SSN last four."""
    call = sid(300 + n)
    h.start(call, THOMAS)
    h.factor(call, Z)
    h.set_pending_tool(call, "authenticate", {"tool": "change_address"})
    assert says(h.handoff(call, {"reason": "step-up", "tool": "change_address"}))[-1] == SSN_PROMPT
    return call


def set_zip(h: Harness, value: str, who: str = "C-1001") -> None:
    item = h.store.get("customers", {"customer_id": who})
    assert item is not None
    item["factors"] = {**item.get("factors", {}), "zip": hash_factor(value)}
    h.store.put("customers", item)


def _no_model_in_deterministic(h: Harness, mode: str) -> None:
    if mode == "deterministic":
        assert h.resolver.calls == []


# --------------------------------------------------------------------------- the card rows


@pytest.mark.parametrize("mode", MODES)
def test_the_ai_names_a_card_the_caller_holds_and_yes_pays_it(mode: str) -> None:
    """"make a $25 payment to the purple card from checking": read back by name, yes pays."""
    h = harness(mode)
    call = at_read_back(h, 1)
    paid_with(h, call, "purple", PURPLE)
    assert h.resolver.calls == []


@pytest.mark.parametrize("mode", MODES)
def test_the_one_thats_purple_is_an_exact_match_in_every_mode(mode: str) -> None:
    """"the one that's purple" when asked (recorded: exact under the carrier trimming, so no model call)."""
    h = harness(mode, ResolverOutput(kind="inferred", value="purple"))
    call = at_card_question(h, 2)
    assert says(speak(h, call, "the one that's purple")) == [readback(PURPLE)]
    assert h.resolver.calls == []
    paid_with(h, call, "purple", PURPLE)


@pytest.mark.parametrize("mode", MODES)
def test_an_inferred_card_is_confirmed_then_yes(mode: str) -> None:
    """The intent of "the one that's purple", with words that don't name it: unclear in
    ``deterministic``; inferred, confirmed, a deterministic yes in ``assisted``; a generated
    confirmation in ``generative``."""
    say = "You'd like the {option:purple}, right?"
    h = harness(mode, ResolverOutput(kind="inferred", value="purple", say=say, say_lang=EN))
    call = at_card_question(h, 3)
    root = speak(h, call, "the violet one")
    if mode == "deterministic":
        assert says(root) == [PROMPT]  # the question lists the cards; no "Please say" first and pending_to(h, call) is None
        _no_model_in_deterministic(h, mode)
        return
    question = f"You'd like the {PURPLE}, right?" if mode == "generative" else CONFIRM_PURPLE
    assert says(root) == [question] and pending_to(h, call) is None
    assert h.resolver.calls[0]["generative"] is (mode == "generative")
    assert says(speak(h, call, "yes")) == [readback(PURPLE)]
    paid_with(h, call, "purple", PURPLE)
    assert effective_modes(h, call, "card") == {mode}


@pytest.mark.parametrize("mode", MODES)
def test_the_owlbank_card_is_ambiguous(mode: str) -> None:
    """"the OwlBank card": unclear; a disambiguation over the caller's cards only; generated."""
    say = "Sure, the {option:red} or the {option:purple}?"
    h = harness(mode, ResolverOutput(kind="ambiguous", candidates=("red", "purple"), say=say,
                                     say_lang=EN))
    call = at_card_question(h, 4)
    root = speak(h, call, "the OwlBank card")
    if mode == "deterministic":
        assert says(root) == [PROMPT]  # the question lists the cards; no "Please say" first
        _no_model_in_deterministic(h, mode)
        return
    spoken = f"Sure, the {RED} or the {PURPLE}?" if mode == "generative" else DISAMBIG
    assert says(root) == [spoken] and REWARDS not in spoken
    assert says(speak(h, call, "the red one")) == [readback(RED)]
    paid_with(h, call, "red", RED)


@pytest.mark.parametrize("mode", MODES)
def test_no_i_meant_the_red_one_to_did_you_mean_purple(mode: str) -> None:
    """"no, I meant the red one" in reply to "Did you mean the OwlBank Purple Card?": Red is
    selected directly. In ``deterministic`` the question is never asked (module docstring):
    the words alone are unclear."""
    h = harness(mode, ResolverOutput(kind="inferred", value="purple"),
                ResolverOutput(kind="exact", value="red"))
    call = at_card_question(h, 5)
    if mode == "deterministic":
        assert says(speak(h, call, "no, I meant the red one")) == [PROMPT]  # the question lists the cards; no "Please say" first
        _no_model_in_deterministic(h, mode)
        return
    assert says(speak(h, call, "the violet one")) == [CONFIRM_PURPLE]
    root = speak(h, call, "no, I meant the red one")
    assert says(root) == [readback(RED)] and pending_to(h, call) == "red"
    assert len(h.resolver.calls) == 2
    paid_with(h, call, "red", RED)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("answer", ["yes", "no"])
def test_the_purple_card_for_a_rewards_only_customer_is_not_available(mode: str,
                                                                      answer: str) -> None:
    """The purple card for Sam: not available (no model); yes selects Rewards, no re-asks.
    The AI naming it asks at once, never replacing it with his one card."""
    say = "No {option:purple} here; shall I use your {option:rewards}?"
    h = harness(mode, ResolverOutput(kind="exact", value="purple", say=say, say_lang=EN))
    call = sid(300 + 6 + (answer == "no"))
    to_level3(h, call, "C-1002")
    root = propose(h, call, "purple")
    # No model call in any mode, so there is no generated text to speak (a "generated
    # not-available question" needs a turn the model resolves; below).
    assert says(root) == [NOT_AVAILABLE_ONE] and h.resolver.calls == []
    assert pending_to(h, call) == "purple"
    if answer == "yes":
        root = h.xml("factor", call, Digits="1")
        assert says(root) == [readback(REWARDS)]
        h.xml("factor", call, Digits="1")
        (pay,) = payments(h)                    # Sam's checking is overdrawn: the bank refuses it
        assert pay["args"]["to"] == "rewards"
    else:
        root = h.xml("factor", call, Digits="2")
        assert says(root) == [PROMPT_SAM] and pending_to(h, call) == "purple" and payments(h) == []


@pytest.mark.parametrize("mode", MODES)
def test_a_card_not_on_file_the_model_resolves_gets_generated_wording_in_generative(
        mode: str) -> None:
    """"the lilac card" from Sam: the model's purple (not named, so inferred) isn't on file: not
    available; the generated question in ``generative``; unclear in ``deterministic``."""
    say = "No {option:purple} here; shall I use your {option:rewards}?"
    h = harness(mode, ResolverOutput(kind="exact", value="purple", say=say, say_lang=EN))
    call = at_card_question(h, 22, who="C-1002")
    root = speak(h, call, "the lilac card")
    if mode == "deterministic":
        assert says(root) == [PROMPT_SAM]
        _no_model_in_deterministic(h, mode)
        return
    expected = (f"No {PURPLE} here; shall I use your {REWARDS}?" if mode == "generative"
                else NOT_AVAILABLE_ONE)
    assert says(root) == [expected] and pending_to(h, call) == "purple"
    assert says(h.xml("factor", call, Digits="1")) == [readback(REWARDS)]


@pytest.mark.parametrize("mode", MODES)
def test_no_the_red_one_during_the_read_back(mode: str) -> None:
    """A correction: ``slot.corrected`` (catalog text in every mode), a new read-back, yes.
    ``deterministic``: unclear, re-asked."""
    h = harness(mode, ResolverOutput(kind="correction", slot="card", value="red"))
    call = at_read_back(h, 8)
    root = speak(h, call, "no, the red one")
    if mode == "deterministic":
        assert says(root) == [YES_NO, readback(PURPLE)] and pending_to(h, call) == "purple"
        _no_model_in_deterministic(h, mode)
        paid_with(h, call, "purple", PURPLE)
        return
    assert says(root) == [f"Okay, I've changed that to {RED}.", readback(RED)]
    assert pending_to(h, call) == "red"
    paid_with(h, call, "red", RED)


@pytest.mark.parametrize("mode", MODES)
def test_whats_my_balance_during_the_read_back(mode: str) -> None:
    """An intent switch to the AI; ``deterministic``: unclear, re-asked."""
    h = harness(mode, ResolverOutput(kind="intent_switch"))
    call = at_read_back(h, 9)
    root = speak(h, call, "what's my balance?")
    assert payments(h) == []
    if mode == "deterministic":
        assert says(root) == [YES_NO, readback(PURPLE)]
        _no_model_in_deterministic(h, mode)
        return
    assert relay(root) is not None
    env = h.pending_result(call)
    assert env.status == "interrupted" and env.callerUtterance == "what's my balance?"


# --------------------------------------------------------------------------- the digit rows


@pytest.mark.parametrize("mode", MODES)
def test_zip_23456_oh_no_actually_it_is_12345(mode: str, caplog: pytest.LogCaptureFixture) -> None:
    """Resolves to 12345 and is read back (ZIP isn't sensitive), yes verifies; ``deterministic``:
    unclear. Digits are never generative: ``generative`` runs the ZIP in ``assisted``."""
    caplog.set_level(logging.INFO)
    h = harness(mode, ResolverOutput(kind="inferred", value="12345"))
    set_zip(h, "12345")
    call = sid(310)
    assert says(h.start(call, THOMAS))[-1] == ZIP_PROMPT
    root = h.xml("factor", call, SpeechResult="23456, oh, no actually it is 12345",
                 Confidence="0.9")
    if mode == "deterministic":
        assert says(root) == [ZIP_CLARIFY]
        _no_model_in_deterministic(h, mode)
        return
    assert says(root) == [confirm_digits("one, two, three, four, five")]
    assert h.resolver.calls[0]["generative"] is False
    root = h.xml("factor", call, Digits="1")
    assert relay(root) is not None and h.state(call).verified_customer_id == "C-1001"
    assert effective_modes(h, call, "zip") == {"assisted"}
    (first, *_) = resolutions(h, call)
    assert first["value"] == "12345"             # a non-sensitive value may be journaled


@pytest.mark.parametrize("mode", MODES)
def test_a_messy_ssn_is_never_read_back_or_logged(mode: str,
                                                  caplog: pytest.LogCaptureFixture) -> None:
    """SSN last four spoken messily: accepted unconfirmed (``confirm: never``), never read back,
    in no log line, response or stored string; ``deterministic``: unclear."""
    caplog.set_level(logging.DEBUG)
    h = harness(mode, ResolverOutput(kind="inferred", value=S))
    call = at_ssn(h, 11)
    words = "eh, one, hold on... one, one and one"
    root = h.xml("factor", call, SpeechResult=words, Confidence="0.9")
    if mode == "deterministic":
        assert says(root) == [SSN_CLARIFY]
        _no_model_in_deterministic(h, mode)
    else:
        assert relay(root) is not None and h.state(call).level == 2
        assert not [t for t in says(root) if "one" in t.casefold()]       # never read back
        assert effective_modes(h, call, "ssn4") == {"assisted"}
    assert_no_leak(h, caplog.records, {S})
    texts = [*h.bodies, *h.stored_strings(), *(r.getMessage() for r in caplog.records)]
    assert not [t for t in texts if words in t]                           # nor the words


@pytest.mark.parametrize("mode", MODES)
def test_an_intent_switch_on_a_sensitive_answer_gets_the_placeholder(mode: str) -> None:
    """"yes my last four are 1111, and what's my balance?": the AI receives the placeholder text
    ``deterministic``: unclear."""
    h = harness(mode, ResolverOutput(kind="intent_switch"))
    call = at_ssn(h, 12)
    root = h.xml("factor", call, SpeechResult=f"yes my last four are {S}, and what's my balance?",
                 Confidence="0.9")
    if mode == "deterministic":
        assert says(root) == [SSN_CLARIFY]
        _no_model_in_deterministic(h, mode)
        return
    assert relay(root) is not None
    env = h.pending_result(call)
    assert env.status == "interrupted"
    assert env.callerUtterance == "yes my last four are [ssn4], and what's my balance?"


# --------------------------------------------------------------------------- rows 1 and 1a


@pytest.mark.parametrize("mode", MODES)
def test_row_1_a_value_with_a_request_is_an_answer(mode: str) -> None:
    """Row 1: "I want my balance and my last four are 1 1 1 1": the deterministic parse reads the
    value, which is accepted and verified, in every mode, with no model call."""
    h = harness(mode, ResolverOutput(kind="intent_switch"))
    call = at_ssn(h, 13)
    root = h.xml("factor", call, SpeechResult="I want my balance and my last four are 1 1 1 1",
                 Confidence="0.9")
    assert relay(root) is not None and h.state(call).level == 2
    assert h.resolver.calls == []
    assert "1 1 1 1" not in (h.state(call).interrupted_utterance or "")


@pytest.mark.parametrize("mode", MODES)
def test_row_1_with_the_resolver_answer_first(mode: str) -> None:
    """Row 1 with the resolver: the parse refuses a value followed by more than two words, so
    the resolver runs and calls it an intent switch carrying the value; the value grounds, so it
    is the answer (inferred; ``confirm: never`` accepts it). ``deterministic``: unclear."""
    h = harness(mode, ResolverOutput(kind="intent_switch", value=S))
    call = at_ssn(h, 14)
    root = h.xml("factor", call,
                 SpeechResult=f"my last four are {S} and I also want my balance please",
                 Confidence="0.9")
    if mode == "deterministic":
        assert says(root) == [SSN_CLARIFY]
        _no_model_in_deterministic(h, mode)
        return
    assert len(h.resolver.calls) == 1
    assert relay(root) is not None and h.state(call).level == 2
    (ev,) = [e for e in resolutions(h, call) if e["slot"] == "ssn4"]
    assert ev["match"] == "inferred" and "value" not in ev


@pytest.mark.parametrize("mode", MODES)
def test_row_1a_no_value_is_an_intent_switch(mode: str) -> None:
    """Row 1a: "actually, can I get my recent transactions": no value, so an intent switch with the
    words unchanged; ``deterministic`` re-asks."""
    said = "actually, can I get my recent transactions"
    h = harness(mode, ResolverOutput(kind="intent_switch"))
    call = at_ssn(h, 15)
    root = h.xml("factor", call, SpeechResult=said, Confidence="0.9")
    if mode == "deterministic":
        assert says(root) == [SSN_CLARIFY]
        _no_model_in_deterministic(h, mode)
        return
    assert relay(root) is not None
    env = h.pending_result(call)
    assert env.status == "interrupted" and env.callerUtterance == said


# ------------------------------------------- the corrected slot's re-ask follows its rules


def _untrusted_zip_then_ssn(h: Harness, call: str, wrong_zip: str = "98101") -> None:
    """An untrusted caller ID: the ZIP is held, sealed, while the SSN last four is asked."""
    h.xml("incoming", call, From=THOMAS, To="+12025550100", CallStatus="ringing")
    h.xml("language", call, Digits="")
    assert says(h.factor(call, wrong_zip)) == ["Got it.", SSN_PROMPT]


def _held_zip(h: Harness, call: str) -> str:
    raw = (h.session(call).flow or {}).get("held") or {}
    return h.sealer.unseal(raw["zip"], call_sid=call)


@pytest.mark.parametrize("mode", ["assisted", "generative"])
def test_an_inferred_zip_at_its_re_ask_is_read_back_then_yes_applies(mode: str) -> None:
    h = harness(mode, ResolverOutput(kind="correction", slot="zip", value=None),
                ResolverOutput(kind="inferred", value=Z))
    call = sid(316)
    _untrusted_zip_then_ssn(h, call)
    root = speak(h, call, "wait, my zip code is different")
    assert says(root) == [ZIP_CLARIFY]
    root = speak(h, call, REASK_WORDS)
    assert says(root) == [confirm_digits("nine, four, one, zero, five")]
    assert _held_zip(h, call) == "98101"                       # not applied before the yes
    g = root.find(".//Gather")
    # A yes/no follow-up in the ZIP's input settings: no digit count, speech allowed.
    assert g is not None and g.get("numDigits") is None and "speech" in (g.get("input") or "")
    root = h.xml("factor", call, Digits="1")
    assert says(root) == ["Okay, I've changed that to nine, four, one, zero, five.", SSN_PROMPT]
    assert _held_zip(h, call) == Z
    root = h.factor(call, S)
    assert relay(root) is not None and h.state(call).level == 1


def test_a_no_to_the_zip_read_back_at_its_re_ask_drops_the_correction() -> None:
    h = harness("assisted", ResolverOutput(kind="correction", slot="zip", value=None),
                ResolverOutput(kind="inferred", value=Z))
    call = sid(317)
    _untrusted_zip_then_ssn(h, call)
    speak(h, call, "wait, my zip code is different")
    speak(h, call, REASK_WORDS)
    root = h.xml("factor", call, Digits="2")
    assert says(root) == [SSN_PROMPT]                          # the SSN resumes
    assert _held_zip(h, call) == "98101"


def test_a_repeat_at_the_zip_read_back_repeats_the_read_back() -> None:
    h = harness("assisted", ResolverOutput(kind="correction", slot="zip", value=None),
                ResolverOutput(kind="inferred", value=Z), ResolverOutput(kind="repeat"))
    call = sid(318)
    _untrusted_zip_then_ssn(h, call)
    speak(h, call, "wait, my zip code is different")
    speak(h, call, REASK_WORDS)
    root = speak(h, call, "sorry, come again")
    assert says(root) == [confirm_digits("nine, four, one, zero, five")]
    assert says(h.xml("factor", call, Digits="1"))[-1] == SSN_PROMPT
    assert _held_zip(h, call) == Z


#: The ZIP at its re-ask, with more than two words after it: the parse leaves it to the resolver.
REASK_WORDS = "nine four one oh five, or so I believe, honestly"


def _gather_of(root: ET.Element) -> ET.Element:
    g = root.find(".//Gather")
    assert g is not None
    return g


def test_a_card_correction_the_caller_did_not_name_is_never_applied() -> None:
    """End to end: the model proposes Red, the words name no card: re-asked (the front
    door's guard), and an inferred card at the re-ask is never applied."""
    h = harness("assisted", ResolverOutput(kind="correction", slot="card", value="red"),
                ResolverOutput(kind="inferred", value="red"))
    call = at_read_back(h, 19)
    root = speak(h, call, "no, the other one")
    assert says(root) == [PROMPT]  # the question lists the cards; no "Please say" first
    assert _gather_of(root).get("hints")                      # the card's Gather
    root = speak(h, call, "the scarlet one")
    assert says(root) == [readback(PURPLE)] and pending_to(h, call) == "purple"
    paid_with(h, call, "purple", PURPLE)


def test_at_the_card_re_ask_an_ambiguous_answer_disambiguates_then_a_name_applies() -> None:
    h = harness("assisted", ResolverOutput(kind="correction", slot="card", value=None),
                ResolverOutput(kind="ambiguous", candidates=("red", "purple")))
    call = at_read_back(h, 20)
    speak(h, call, "no, the other card")
    root = speak(h, call, "the OwlBank card")
    assert says(root) == [DISAMBIG] and pending_to(h, call) == "purple"
    root = speak(h, call, "the red one")
    assert says(root) == [f"Okay, I've changed that to {RED}.", readback(RED)]
    paid_with(h, call, "red", RED)


def test_the_otp_flow_is_untouched_by_a_correction_follow_up() -> None:
    """A step-up through the code, with a ZIP correction read back at the SSN prompt first."""
    h = harness("assisted", ResolverOutput(kind="correction", slot="zip", value=None),
                ResolverOutput(kind="inferred", value=Z))
    call = sid(321)
    _untrusted_zip_then_ssn(h, call)
    speak(h, call, "wait, my zip code is different")
    speak(h, call, REASK_WORDS)
    h.xml("factor", call, Digits="1")
    h.factor(call, S)
    h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    h.handoff(call, {"reason": "step-up", "tool": "make_payment"})
    root = h.factor(call, OTP)
    assert h.state(call).level == 3 and relay(root) is not None
