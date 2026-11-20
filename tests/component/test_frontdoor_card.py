"""The card in the payment confirmation.

Signed fake webhooks into the real front door, with the real auth service and bank tools behind
the fake Lambda client (``test_frontdoor_support.build``) and ``FakeResolver`` for the model.
C-1001 (Thomas) holds the OwlBank Red Card and the OwlBank Purple Card; C-1002 (Sam) holds the
OwlBank Rewards Card only.

* A ``to`` that is one of the caller's cards is read back by name. A missing or unknown one with one
  card on file uses that card with no question; with two or more, ``prompt.card`` lists them
  ("your ... or your ...", policy order) and that order is the one ordinals pick from. A
  known card the caller doesn't hold asks the not-available question and is never mapped to
  another card.
* The ``card`` answer goes through ``decide``: confirm, disambiguate, not available, "no, I meant Y".
* During the read-back a card correction replaces ``pending_tool.args.to``, says ``slot.corrected``
  and reads back again; a change to the amount or the from-account is an intent switch.
* "The one that's purple" is unclear in ``deterministic`` and inferred in ``assisted`` in the
  original wording; the carrier trimming makes it an exact deterministic match for Purple in every
  mode, so it is accepted with no model call and the read-back confirms it. Tested that way.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from typing import Any

import pytest

from owlbank import callstate, sensitive
from owlbank.callstate import PendingTool
from owlbank.catalog import PromptCatalog
from owlbank.envelope import CANCELLED_BY_CALLER
from owlbank.policy import Policy
from owlbank.segments import clear_sensitive
from owlbank.slots import FakeResolver, ResolverOutput
from owlbank.slots.resolver import request_payload
from tests.component.test_frontdoor_support import (
    DATA,
    LEES,
    MARIA,
    OTP,
    TEST_FACTORS,
    THOMAS,
    Harness,
    all_fixture_factor_values,
    assert_no_leak,
    build,
    relay,
    says,
    sid,
)

ASSISTED, DETERMINISTIC = "assisted", "deterministic"
MODES = (ASSISTED, DETERMINISTIC)

RED, PURPLE, REWARDS = "OwlBank Red Card", "OwlBank Purple Card", "OwlBank Rewards Card"
PROMPT = f"Which card would you like to pay: your {RED} or your {PURPLE}?"
PROMPT_SAM = f"Which card would you like to pay: your {REWARDS}?"
NOT_AVAILABLE_SAM = (f"I don't see an {PURPLE} on your profile. You have the {REWARDS}. "
                     "Would you like to use that?")
CLARIFY_THOMAS = f"Please say {RED} or {PURPLE}."
YES_NO = "Please say yes or no."


@pytest.fixture(autouse=True)
def _forget_calls() -> Any:
    """Each test's calls leave no sensitive registry entry behind (the registry is per process and
    keyed by CallSid, which other modules' tests reuse)."""
    yield
    for n in range(64):
        clear_sensitive(sid(n))
        sensitive.forget(sid(n))


def readback(card: str, amount: str = "twenty-five dollars") -> str:
    return (f"To confirm, you'd like to pay {amount} toward your {card} from your checking account. "
            "Is that right? Say yes or no, or press 1 for yes and 2 for no.")


def build_h(mode: str, *outs: ResolverOutput, card: dict[str, Any] | None = None,
            default: ResolverOutput | None = None) -> Harness:
    """A harness in ``mode`` (the running ceiling), with ``card``'s settings changed."""
    kw = {"slot_mode_ceiling": DETERMINISTIC} if mode == DETERMINISTIC else {}
    h = build(resolver=FakeResolver(list(outs), default=default), config_kw=kw)
    if card:
        raw = json.loads((DATA / "policy.json").read_text())
        raw["slots"]["card"].update(card)
        h.app.d.policy = Policy.from_dict(raw)
    return h


def to_level3(h: Harness, call: str, who: str = "C-1001") -> None:
    """The caller verified to level 3 (``make_payment``), with no tool pending."""
    ani = {"C-1001": THOMAS, "C-1002": LEES, "C-1004": MARIA}[who]
    h.start(call, ani)
    h.factor(call, TEST_FACTORS[who]["zip"])
    assert h.state(call).verified_customer_id == who
    h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    h.handoff(call, {"reason": "step-up", "tool": "make_payment", "journal": "jr-1"})
    h.factor(call, TEST_FACTORS[who]["ssn4"])
    h.factor(call, OTP)
    assert h.state(call).level == 3
    callstate.mutate(h.store, call, lambda s: {"pending_result": None, "pending_tool": None})


def propose(h: Harness, call: str, to: Any = None, *, absent: bool = False,
            amount: str = "25.00") -> ET.Element:
    """The AI proposes ``make_payment`` (``to`` left out when ``absent``)."""
    args: dict[str, Any] = {"from_account": "checking", "amount": amount}
    if not absent:
        args["to"] = to
    callstate.mutate(h.store, call, lambda s: {
        "pending_tool": PendingTool(tool_use_id="tu_pay", name="make_payment", args=args)})
    return h.handoff(call, {"reason": "confirm", "tool": "make_payment", "journal": "jr-2"})


def speak(h: Harness, call: str, text: str) -> ET.Element:
    return h.xml("factor", call, SpeechResult=text, Confidence="0.9")


def payments(h: Harness) -> list[dict[str, Any]]:
    return [p for p in h.lam.payloads if p.get("op") == "make_payment"]


def pending_to(h: Harness, call: str) -> Any:
    pt = h.state(call).pending_tool
    return None if pt is None else pt.args.get("to")


def gather(root: ET.Element) -> ET.Element:
    g = root.find("Gather")
    assert g is not None, ET.tostring(root)
    return g


def paid_with(h: Harness, call: str, card: str, spoken: str) -> None:
    """A deterministic yes ran the payment once, to ``card``, and the outcome names it."""
    root = h.xml("factor", call, Digits="1")
    (pay,) = payments(h)
    assert pay["args"]["to"] == card
    assert says(root)[0].startswith(
        f"Done. Your payment of twenty-five dollars to your {spoken} from your checking account")
    assert relay(root) is not None
    env = h.pending_result(call)
    assert env.status == "completed" and env.data["executed"] is True


def resolutions(h: Harness, call: str) -> list[dict[str, Any]]:
    return [e for e in h.journal(call) if e["type"] == "slot_resolution"]


# --------------------------------------------------------------------------- the AI names the card


@pytest.mark.parametrize("mode", MODES)
def test_a_card_the_caller_holds_is_read_back_by_name_and_yes_pays_it(mode: str) -> None:
    """"make a $25 payment to the purple card from checking" (the AI proposes ``purple``)."""
    h = build_h(mode)
    call = sid(1)
    to_level3(h, call)
    root = propose(h, call, "purple")
    assert says(root) == [readback(PURPLE)] and payments(h) == []
    paid_with(h, call, "purple", PURPLE)
    assert h.resolver.calls == []


@pytest.mark.parametrize("to", [None, "credit_card", "", 7])
def test_a_missing_or_unknown_card_lists_the_callers_cards_and_is_never_mapped(to: Any) -> None:
    """No card, or a value that isn't a known card, with two cards on file."""
    h = build_h(ASSISTED)
    call = sid(2)
    to_level3(h, call)
    root = propose(h, call, to, absent=to is None)
    assert says(root) == [PROMPT]
    assert pending_to(h, call) == to and payments(h) == []
    g = gather(root)
    assert g.get("input") == "dtmf speech"
    hints = (g.get("hints") or "").split(", ")
    assert {RED, "red", PURPLE, "purple", REWARDS, "rewards"} <= set(hints)
    assert [e["factor"] for e in h.journal(call) if e["type"] == "factor_asked"][-1] == "card"


def test_the_spoken_name_answers_the_card_question_then_the_read_back_names_it() -> None:
    h = build_h(ASSISTED)
    call = sid(3)
    to_level3(h, call)
    propose(h, call, absent=True)
    root = speak(h, call, "the OwlBank Red Card")
    assert says(root) == [readback(RED)] and pending_to(h, call) == "red"
    assert h.resolver.calls == []
    paid_with(h, call, "red", RED)


def test_a_key_selects_the_card_by_its_position_among_the_callers_cards() -> None:
    h = build_h(ASSISTED)
    call = sid(4)
    to_level3(h, call)
    propose(h, call, absent=True)
    root = h.xml("factor", call, Digits="2")
    assert says(root) == [readback(PURPLE)] and pending_to(h, call) == "purple"




@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("to", [None, "credit_card", 7])
def test_one_card_on_file_is_used_with_no_question_and_the_read_back_names_it(
        mode: str, to: Any) -> None:
    """C-1002 holds only the Rewards card; no card (or an unknown value) needs no question."""
    h = build_h(mode)
    call = sid(60)
    to_level3(h, call, "C-1002")
    root = propose(h, call, to, absent=to is None)
    assert says(root) == [readback(REWARDS)] and pending_to(h, call) == "rewards"
    assert "card" not in [e["factor"] for e in h.journal(call) if e["type"] == "factor_asked"]
    assert h.resolver.calls == [] and payments(h) == []
    h.xml("factor", call, Digits="1")
    (pay,) = payments(h)                    # Sam's checking is overdrawn: the bank refuses it
    assert pay["args"]["to"] == "rewards"


def test_the_single_card_stays_correctable_in_the_read_back() -> None:
    """The card set without a question is collected like a named one: "no" cancels."""
    h = build_h(DETERMINISTIC)
    call = sid(61)
    to_level3(h, call, "C-1002")
    propose(h, call, absent=True)
    assert [c["name"] for c in h.session(call).collected if c["name"] == "card"] == ["card"]
    root = speak(h, call, "no")
    assert relay(root) is not None and payments(h) == []


@pytest.mark.parametrize("mode", MODES)
def test_the_card_question_lists_the_callers_cards_and_the_second_one_selects_purple(
        mode: str) -> None:
    """Policy order, each with "your"; an ordinal picks from that order, no model."""
    h = build_h(mode, ResolverOutput(kind="exact", value="red"))
    call = sid(62)
    to_level3(h, call)
    root = propose(h, call, absent=True)
    assert says(root) == [PROMPT]
    assert h.session(call).flow["listed"] == {"slot": "card", "values": ["red", "purple"]}
    root = speak(h, call, "the second one")
    assert says(root) == [readback(PURPLE)] and pending_to(h, call) == "purple"
    assert h.resolver.calls == []
    (res,) = [e for e in resolutions(h, call) if e["slot"] == "card"]
    assert res["match"] == "exact" and res["source"] == "ordinal" and res["value"] == "purple"
    paid_with(h, call, "purple", PURPLE)


@pytest.mark.parametrize("text, card", [("the first one", "red"), ("the last one", "purple"),
                                        ("second", "purple")])
def test_ordinals_over_the_card_question(text: str, card: str) -> None:
    h = build_h(DETERMINISTIC)
    call = sid(63)
    to_level3(h, call)
    propose(h, call, absent=True)
    assert pending_to(h, call) is None
    speak(h, call, text)
    assert pending_to(h, call) == card


@pytest.mark.parametrize("text", ["the third one", "the first one or the second one"])
def test_an_ordinal_beyond_the_list_or_two_ordinals_select_nothing(text: str) -> None:
    h = build_h(DETERMINISTIC)
    call = sid(64)
    to_level3(h, call)
    propose(h, call, absent=True)
    root = speak(h, call, text)
    assert says(root) == [PROMPT]  # the question lists the cards; no "Please say" first and pending_to(h, call) is None


def test_an_ordinal_after_the_clarify_picks_from_the_cards_it_listed() -> None:
    h = build_h(DETERMINISTIC)
    call = sid(65)
    to_level3(h, call)
    propose(h, call, absent=True)
    speak(h, call, "hmm")
    assert h.session(call).flow["listed"]["values"] == ["red", "purple"]
    speak(h, call, "the first one")
    assert pending_to(h, call) == "red"


def test_an_ordinal_after_a_disambiguation_picks_from_its_candidates() -> None:
    h = build_h(ASSISTED, ResolverOutput(kind="ambiguous", candidates=["purple", "red"]))
    call = sid(66)
    to_level3(h, call)
    propose(h, call, absent=True)
    root = speak(h, call, "the OwlBank card")
    assert says(root) == [f"Which one did you mean: {PURPLE} or {RED}?"]
    assert h.session(call).flow["pending"]["listed"] == ["purple", "red"]   # the order read
    root = speak(h, call, "the second one")
    assert says(root) == [readback(RED)] and pending_to(h, call) == "red"
    assert len(h.resolver.calls) == 1


@pytest.mark.parametrize("mode", MODES)
def test_a_known_card_thomas_doesnt_hold_asks_not_available_and_an_ordinal_answers_it(
        mode: str) -> None:
    """Never replaced; the not-available question lists his cards."""
    h = build_h(mode)
    call = sid(67)
    to_level3(h, call)
    root = propose(h, call, "rewards")
    assert says(root) == [f"I don't see an {REWARDS} on your profile. You can use {RED} or "
                          f"{PURPLE}. Which would you like?"]
    assert pending_to(h, call) == "rewards" and payments(h) == []
    assert h.session(call).flow["pending"]["listed"] == ["red", "purple"]
    assert gather(root).get("hints")                         # a choice Gather, not a yes/no
    root = speak(h, call, "the second one")
    assert says(root) == [readback(PURPLE)] and pending_to(h, call) == "purple"
    assert h.resolver.calls == []


def test_repeat_says_the_listing_card_question_again_and_keeps_the_order() -> None:
    h = build_h(DETERMINISTIC)
    call = sid(68)
    to_level3(h, call)
    propose(h, call, absent=True)
    assert says(speak(h, call, "say that again")) == [PROMPT]
    speak(h, call, "the second one")
    assert pending_to(h, call) == "purple"


def test_an_ordinal_with_nothing_listed_selects_nothing() -> None:
    """The read-back lists nothing, so "the second one" there is no card."""
    h = build_h(DETERMINISTIC)
    call = sid(69)
    to_level3(h, call)
    propose(h, call, "red")
    root = speak(h, call, "the second one")
    assert says(root) == [YES_NO, readback(RED)] and pending_to(h, call) == "red"


def test_the_spanish_card_question_lists_the_cards_and_la_segunda_selects_morada() -> None:
    h = build_h(ASSISTED)
    call = sid(70)
    h.xml("incoming", call, From=THOMAS, To="+12025550100", CallStatus="ringing",
          StirVerstat="TN-Validation-Passed-A")
    h.xml("language", call, Digits="2")
    if h.state(call).language != "es-US":
        pytest.skip("the language menu's Spanish key isn't 2")
    h.factor(call, TEST_FACTORS["C-1001"]["zip"])
    h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    h.handoff(call, {"reason": "step-up", "tool": "make_payment"})
    h.factor(call, TEST_FACTORS["C-1001"]["ssn4"])
    h.factor(call, OTP)
    callstate.mutate(h.store, call, lambda s: {"pending_result": None, "pending_tool": None})
    assert says(propose(h, call, absent=True)) == [
        "¿A qué tarjeta desea hacer el pago: su Tarjeta OwlBank Roja o su Tarjeta OwlBank Morada?"]
    (text,) = says(speak(h, call, "la segunda"))
    assert "a su Tarjeta OwlBank Morada desde" in text and pending_to(h, call) == "purple"


# --------------------------------------------------------------------------- assisted rows


def test_the_one_thats_purple_is_an_exact_match_in_assisted_with_no_model_call() -> None:
    """The carrier trimming makes it exact, not inferred (module docstring)."""
    h = build_h(ASSISTED, ResolverOutput(kind="inferred", value="purple"))
    call = sid(5)
    to_level3(h, call)
    propose(h, call, absent=True)
    root = speak(h, call, "the one that's purple")
    assert says(root) == [readback(PURPLE)] and h.resolver.calls == []
    paid_with(h, call, "purple", PURPLE)


def test_an_inferred_card_is_confirmed_and_a_deterministic_yes_selects_it() -> None:
    h = build_h(ASSISTED, ResolverOutput(kind="inferred", value="purple"))
    call = sid(6)
    to_level3(h, call)
    propose(h, call, absent=True)
    root = speak(h, call, "the violet one")
    assert says(root) == [f"Did you mean the {PURPLE}?"] and pending_to(h, call) is None
    g = gather(root)
    assert g.get("hints") is None and g.get("numDigits") is None    # a yes/no follow-up
    root = speak(h, call, "yes")
    assert says(root) == [readback(PURPLE)]
    paid_with(h, call, "purple", PURPLE)


def test_a_model_yes_to_the_card_confirmation_is_unclear() -> None:
    """Only a deterministic yes selects the option a question names."""
    h = build_h(ASSISTED, ResolverOutput(kind="inferred", value="purple"),
                ResolverOutput(kind="none", reply="yes"))
    call = sid(7)
    to_level3(h, call)
    propose(h, call, absent=True)
    speak(h, call, "the violet one")
    root = speak(h, call, "go with that")
    assert says(root) == [YES_NO, f"Did you mean the {PURPLE}?"]
    assert pending_to(h, call) is None


def test_the_owlbank_card_is_ambiguous_and_asks_between_the_callers_cards_only() -> None:
    h = build_h(ASSISTED, ResolverOutput(kind="ambiguous", candidates=["red", "purple"]))
    call = sid(8)
    to_level3(h, call)
    propose(h, call, absent=True)
    root = speak(h, call, "the OwlBank card")
    assert says(root) == [f"Which one did you mean: {RED} or {PURPLE}?"]
    root = speak(h, call, "the red one")
    assert says(root) == [readback(RED)]
    paid_with(h, call, "red", RED)


def test_no_i_meant_the_red_one_selects_red_directly() -> None:
    """In reply to "Did you mean the OwlBank Purple Card?"."""
    h = build_h(ASSISTED, ResolverOutput(kind="inferred", value="purple"),
                ResolverOutput(kind="exact", value="red"))
    call = sid(9)
    to_level3(h, call)
    propose(h, call, absent=True)
    assert says(speak(h, call, "the violet one")) == [f"Did you mean the {PURPLE}?"]
    root = speak(h, call, "no, I meant the red one")
    assert says(root) == [readback(RED)] and pending_to(h, call) == "red"
    paid_with(h, call, "red", RED)


@pytest.mark.parametrize("mode", MODES)
def test_the_purple_card_for_sam_is_not_available_then_yes_selects_rewards(mode: str) -> None:
    """A known card Sam doesn't hold is asked about, never replaced by his one card
; only the explicit yes selects the card on file. No model call."""
    h = build_h(mode)
    call = sid(10)
    to_level3(h, call, "C-1002")
    root = propose(h, call, "purple")
    assert says(root) == [NOT_AVAILABLE_SAM]
    g = gather(root)
    assert g.get("hints") is None and g.get("numDigits") is None    # a yes/no follow-up
    assert pending_to(h, call) == "purple"
    root = speak(h, call, "yes")
    assert says(root) == [readback(REWARDS)] and pending_to(h, call) == "rewards"
    h.xml("factor", call, Digits="1")
    (pay,) = payments(h)                    # Sam's checking is overdrawn: the bank refuses it
    assert pay["args"]["to"] == "rewards" and h.resolver.calls == []


@pytest.mark.parametrize("mode", MODES)
def test_the_purple_card_for_sam_then_no_asks_the_card_again(mode: str) -> None:
    h = build_h(mode)
    call = sid(11)
    to_level3(h, call, "C-1002")
    propose(h, call, "purple")
    root = speak(h, call, "no")
    assert says(root) == [PROMPT_SAM]
    assert pending_to(h, call) == "purple" and payments(h) == []
    assert h.session(call).flow["tries"]["card"] == 1       # the no (the AI named purple, not Sam)


def test_a_model_yes_to_not_available_is_unclear_and_repeats_the_question() -> None:
    h = build_h(ASSISTED, ResolverOutput(kind="none", reply="yes"))
    call = sid(12)
    to_level3(h, call, "C-1002")
    propose(h, call, "purple")
    root = speak(h, call, "go with that")
    assert says(root) == [YES_NO, NOT_AVAILABLE_SAM]
    assert pending_to(h, call) == "purple"


# --------------------------------------------------------------------------- the read-back


def test_no_the_red_one_during_the_read_back_corrects_the_card_and_reads_back_again() -> None:
    h = build_h(ASSISTED, ResolverOutput(kind="correction", slot="card", value="red"))
    call = sid(13)
    to_level3(h, call)
    propose(h, call, "purple")
    root = speak(h, call, "no, the red one")
    assert says(root) == [f"Okay, I've changed that to {RED}.", readback(RED)]
    assert pending_to(h, call) == "red" and payments(h) == []
    (corr,) = [e for e in resolutions(h, call) if e["correction"]]
    assert corr["slot"] == "card" and corr["value"] == "red" and corr["correction_of"] == "yes_no"
    paid_with(h, call, "red", RED)


def test_a_card_correction_the_caller_didnt_name_is_not_applied_and_the_card_is_asked() -> None:
    """The model's card isn't in the words ("the other one"): nothing is changed; the card is
    re-asked once; a named answer then corrects it."""
    h = build_h(ASSISTED, ResolverOutput(kind="correction", slot="card", value="red"))
    call = sid(14)
    to_level3(h, call)
    propose(h, call, "purple")
    root = speak(h, call, "no, the other one")
    assert says(root) == [PROMPT]  # the question lists the cards; no "Please say" first and pending_to(h, call) == "purple"
    assert h.session(call).flow["pending"]["kind"] == "correction_reask"
    root = speak(h, call, "red")
    assert says(root) == [f"Okay, I've changed that to {RED}.", readback(RED)]
    assert pending_to(h, call) == "red"


def _card_entry(h: Harness) -> dict[str, Any]:
    """The collected card as the last resolver request sent it (``request_payload``)."""
    payload = request_payload(h.resolver.requests[-1])
    (entry,) = [c for c in payload["collected"] if c["name"] == "card"]
    return entry


def test_actually_make_that_the_purple_card_at_the_read_back_corrects_the_card() -> None:
    """The real-call defect: card question, "red", the read-back, then "actually make that
    the purple card". The request lists the collected card's options and the caller's cards (never
    the card chosen), so the model can name ``purple``; the caller named it, so it is applied."""
    h = build_h(ASSISTED, ResolverOutput(kind="correction", slot="card", value="purple"))
    call = sid(71)
    to_level3(h, call)
    assert says(propose(h, call, absent=True)) == [PROMPT]
    root = speak(h, call, "red")
    assert says(root) == [readback(RED)] and pending_to(h, call) == "red"
    assert h.resolver.calls == []
    root = speak(h, call, "actually make that the purple card")
    assert says(root) == [f"Okay, I've changed that to {PURPLE}.", readback(PURPLE)]
    assert pending_to(h, call) == "purple" and payments(h) == []
    (req,) = h.resolver.requests
    assert req.slot.slot_type == "yes_no" and req.known_options == ()
    entry = _card_entry(h)
    assert set(entry) == {"name", "type", "label", "scope", "options", "caller_options"}
    assert entry["type"] == "choice" and entry["caller_options"] == ["red", "purple"]
    assert {o["value"] for o in entry["options"]} >= {"red", "purple", "rewards"}
    assert {o["name"] for o in entry["options"]} >= {RED, PURPLE}
    (corr,) = [e for e in resolutions(h, call) if e["correction"]]
    assert corr["slot"] == "card" and corr["value"] == "purple"
    paid_with(h, call, "purple", PURPLE)


def test_a_model_card_correction_the_caller_didnt_say_is_not_applied_at_the_read_back() -> None:
    """"change it" names no card; the model's ``purple`` is not applied. The card is re-asked
    once and stays red."""
    h = build_h(ASSISTED, ResolverOutput(kind="correction", slot="card", value="purple"))
    call = sid(72)
    to_level3(h, call)
    propose(h, call, absent=True)
    speak(h, call, "red")
    root = speak(h, call, "change it")
    assert says(root) == [PROMPT] and pending_to(h, call) == "red"
    assert h.session(call).flow["pending"]["kind"] == "correction_reask"
    assert not [e for e in resolutions(h, call) if e.get("correction")]
    assert payments(h) == []


def test_a_correction_to_a_card_the_caller_doesnt_hold_is_re_asked_never_applied() -> None:
    h = build_h(ASSISTED, ResolverOutput(kind="correction", slot="card", value="rewards"))
    call = sid(15)
    to_level3(h, call)
    propose(h, call, "purple")
    root = speak(h, call, "no, the rewards card")
    assert says(root) == [PROMPT]  # the question lists the cards; no "Please say" first and pending_to(h, call) == "purple"
    # Still not theirs. The re-ask's reply follows the card's rules: not available,
    # in the card's Gather; nothing is applied yet.
    root = speak(h, call, "the rewards card")
    assert says(root) == [f"I don't see an {REWARDS} on your profile. You can use {RED} or "
                          f"{PURPLE}. Which would you like?"]
    assert pending_to(h, call) == "purple" and payments(h) == []
    # Named on file: applied, and the payment read back with it (the read-back confirms).
    root = speak(h, call, "the red card")
    assert says(root) == [f"Okay, I've changed that to {RED}.", readback(RED)]
    assert pending_to(h, call) == "red"
    paid_with(h, call, "red", RED)


def test_a_card_not_on_file_at_the_re_ask_then_unclear_resumes_the_read_back() -> None:
    h = build_h(ASSISTED, ResolverOutput(kind="correction", slot="card", value="rewards"))
    call = sid(95)
    to_level3(h, call)
    propose(h, call, "purple")
    speak(h, call, "no, the rewards card")
    speak(h, call, "the rewards card")             # not available
    root = speak(h, call, "hmm")                   # unclear: the correction is dropped
    assert says(root) == [readback(PURPLE)] and pending_to(h, call) == "purple"


def test_an_inferred_card_at_the_card_re_ask_is_not_applied() -> None:
    h = build_h(ASSISTED, ResolverOutput(kind="correction", slot="card", value="red"),
                ResolverOutput(kind="inferred", value="red"))
    call = sid(16)
    to_level3(h, call)
    propose(h, call, "purple")
    speak(h, call, "no, the other one")
    root = speak(h, call, "the scarlet one")
    assert says(root) == [readback(PURPLE)] and pending_to(h, call) == "purple"


def test_whats_my_balance_during_the_read_back_is_an_intent_switch_in_assisted() -> None:
    h = build_h(ASSISTED, ResolverOutput(kind="intent_switch"))
    call = sid(17)
    to_level3(h, call)
    propose(h, call, "purple")
    root = speak(h, call, "what's my balance?")
    assert relay(root) is not None and payments(h) == []
    env = h.pending_result(call)
    assert env.status == "interrupted" and env.callerUtterance == "what's my balance?"


@pytest.mark.parametrize("target", ["amount", "from_account"])
def test_a_change_to_the_amount_or_the_from_account_is_an_intent_switch(target: str) -> None:
    """The AI proposed it; it proposes again and a new read-back confirms it."""
    h = build_h(ASSISTED, ResolverOutput(kind="correction", slot=target, value="50.00"))
    call = sid(18)
    to_level3(h, call)
    propose(h, call, "purple")
    root = speak(h, call, "actually make it fifty dollars from savings")
    assert relay(root) is not None and payments(h) == []
    env = h.pending_result(call)
    assert env.status == "interrupted" and env.data["executed"] is False


# --------------------------------------------------------------------------- deterministic rows


def test_deterministic_the_one_thats_purple_is_accepted_by_the_alias() -> None:
    h = build_h(DETERMINISTIC)
    call = sid(20)
    to_level3(h, call)
    propose(h, call, absent=True)
    assert says(speak(h, call, "the one that's purple")) == [readback(PURPLE)]
    paid_with(h, call, "purple", PURPLE)
    assert h.resolver.calls == []


@pytest.mark.parametrize("text", ["the OwlBank card", "no, I meant the red one", "the violet one"])
def test_deterministic_unclear_card_answers_re_ask_listing_the_callers_cards(text: str) -> None:
    h = build_h(DETERMINISTIC, ResolverOutput(kind="exact", value="red"))
    call = sid(21)
    to_level3(h, call)
    propose(h, call, absent=True)
    root = speak(h, call, text)
    assert says(root) == [PROMPT]  # the question lists the cards; no "Please say" first
    assert pending_to(h, call) is None and h.resolver.calls == []
    assert h.session(call).flow["tries"]["card"] == 1


@pytest.mark.parametrize("text", ["no, the red one", "what's my balance?"])
def test_deterministic_read_back_answers_that_arent_yes_or_no_re_ask(text: str) -> None:
    h = build_h(DETERMINISTIC, ResolverOutput(kind="correction", slot="card", value="red"))
    call = sid(22)
    to_level3(h, call)
    propose(h, call, "purple")
    root = speak(h, call, text)
    assert says(root) == [YES_NO, readback(PURPLE)]
    assert pending_to(h, call) == "purple" and h.resolver.calls == [] and payments(h) == []


# --------------------------------------------------------------------------- retries, commands


def test_the_card_question_gives_up_after_its_retries_and_nothing_runs() -> None:
    h = build_h(DETERMINISTIC)
    call = sid(23)
    to_level3(h, call)
    propose(h, call, absent=True)
    for _ in range(2):                     # card maxRetries: 2
        assert says(speak(h, call, "hmm"))[0] == PROMPT
    root = speak(h, call, "hmm")
    assert relay(root) is not None and payments(h) == []
    env = h.pending_result(call)
    assert env.status == CANCELLED_BY_CALLER and env.data["answer"] == "no_answer"


def test_repeat_says_the_card_question_again_without_a_try() -> None:
    h = build_h(DETERMINISTIC)
    call = sid(24)
    to_level3(h, call)
    propose(h, call, absent=True)
    assert says(speak(h, call, "say that again")) == [PROMPT]
    assert h.session(call).flow["tries"].get("card", 0) == 0


def test_cancel_at_the_card_question_cancels_the_payment() -> None:
    h = build_h(DETERMINISTIC)
    call = sid(25)
    to_level3(h, call)
    propose(h, call, absent=True)
    root = speak(h, call, "cancel")
    assert relay(root) is not None and payments(h) == []
    assert h.pending_result(call).data["answer"] == "cancel"


def test_a_customer_with_no_card_on_file_hears_the_apology_and_nothing_runs() -> None:
    h = build_h(ASSISTED)
    call = sid(26)
    to_level3(h, call, "C-1004")
    root = propose(h, call, "red")
    assert relay(root) is not None and payments(h) == []
    env = h.pending_result(call)
    assert env.status == "failed" and [s.template for s in env.spoken] == ["error.generic"]


def test_the_payment_never_runs_with_a_card_the_caller_doesnt_hold() -> None:
    """Defence in depth: call state changed under the read-back."""
    h = build_h(ASSISTED)
    call = sid(27)
    to_level3(h, call)
    propose(h, call, "purple")
    callstate.mutate(h.store, call, lambda s: {
        "pending_tool": PendingTool(tool_use_id="tu_pay", name="make_payment",
                                    args={**s.pending_tool.args, "to": "rewards"})})
    h.xml("factor", call, Digits="1")
    assert payments(h) == [] and h.pending_result(call).status == "failed"


# --------------------------------------------------------------------------- Gather settings


def test_a_keypad_only_card_hears_a_keypad_menu_and_never_calls_the_model() -> None:
    h = build_h(ASSISTED, ResolverOutput(kind="exact", value="red"), card={"keypad_only": True})
    call = sid(28)
    to_level3(h, call)
    root = propose(h, call, absent=True)
    assert says(root) == [PROMPT, f"Press 1 for the {RED}, 2 for the {PURPLE}."]
    g = gather(root)
    assert g.get("input") == "dtmf" and g.get("hints") is None
    root = speak(h, call, "the red one")         # speech can't answer a keypad-only slot
    assert says(root)[0] == PROMPT and h.resolver.calls == []
    assert says(root)[-1] == f"Press 1 for the {RED}, 2 for the {PURPLE}."
    root = h.xml("factor", call, Digits="2")
    assert says(root) == [readback(PURPLE)]


def test_a_keypad_only_not_available_question_is_a_keypad_yes_no() -> None:
    h = build_h(ASSISTED, card={"keypad_only": True})
    call = sid(29)
    to_level3(h, call, "C-1002")
    root = propose(h, call, "purple")
    assert says(root) == [NOT_AVAILABLE_SAM, "Press 1 for yes, or 2 for no."]
    assert gather(root).get("input") == "dtmf"
    assert says(h.xml("factor", call, Digits="1")) == [readback(REWARDS)]


def test_a_barge_in_false_card_plays_its_questions_before_the_gather_path_b() -> None:
    h = build_h(ASSISTED, ResolverOutput(kind="ambiguous", candidates=["red", "purple"]),
                card={"barge_in": False})
    call = sid(30)
    to_level3(h, call)
    for root in (propose(h, call, absent=True), speak(h, call, "the OwlBank card")):
        tags = [c.tag for c in root]
        assert tags[0] in ("Say", "Play") and tags[-1] == "Gather"
        assert len(gather(root)) == 0                             # nothing nested


# --------------------------------------------------------------------------- catalog, Spanish


@pytest.mark.parametrize("lang, card, start, done", [
    ("en-US", RED, "toward your OwlBank Red Card from", "to your OwlBank Red Card from"),
    ("es-US", "Tarjeta OwlBank Roja", "a su Tarjeta OwlBank Roja desde",
     "a su Tarjeta OwlBank Roja desde"),
])
def test_both_confirm_templates_name_the_card(lang: str, card: str, start: str, done: str) -> None:
    cat = PromptCatalog.from_dict(json.loads((DATA / "prompt_catalog.json").read_text()))
    values = {"amount": "x", "fromAccount": "y", "toCard": card, "confirmation": "z"}
    assert start in cat.text("confirm.make_payment", lang, **values)
    assert done in cat.text("confirm.make_payment.done", lang, **values)


def test_the_spanish_read_back_names_the_card() -> None:
    h = build_h(ASSISTED)
    call = sid(31)
    h.xml("incoming", call, From=THOMAS, To="+12025550100", CallStatus="ringing",
          StirVerstat="TN-Validation-Passed-A")
    h.xml("language", call, Digits="2")
    if h.state(call).language != "es-US":
        pytest.skip("the language menu's Spanish key isn't 2")
    h.factor(call, TEST_FACTORS["C-1001"]["zip"])
    h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    h.handoff(call, {"reason": "step-up", "tool": "make_payment"})
    h.factor(call, TEST_FACTORS["C-1001"]["ssn4"])
    h.factor(call, OTP)
    callstate.mutate(h.store, call, lambda s: {"pending_result": None, "pending_tool": None})
    (text,) = says(propose(h, call, "purple"))
    assert "a su Tarjeta OwlBank Morada desde su cuenta de cheques" in text


# --------------------------------------------------------------------------- no leaks


def test_a_card_flow_leaks_no_factor_value(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    h = build_h(ASSISTED, ResolverOutput(kind="ambiguous", candidates=["red", "purple"]),
                ResolverOutput(kind="correction", slot="card", value="purple"))
    call = sid(32)
    try:
        to_level3(h, call)
        propose(h, call, absent=True)
        speak(h, call, "the OwlBank card")
        speak(h, call, "red")
        speak(h, call, "no, the purple one")
        h.xml("factor", call, Digits="1")
        assert payments(h)[0]["args"]["to"] == "purple"
        h.sent_factors |= {TEST_FACTORS["C-1001"]["zip"], TEST_FACTORS["C-1001"]["ssn4"], OTP}
        assert_no_leak(h, caplog.records, all_fixture_factor_values() | h.sent_factors)
    finally:
        clear_sensitive(call)
