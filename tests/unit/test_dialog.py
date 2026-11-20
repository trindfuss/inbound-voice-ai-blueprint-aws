"""Unit tests for the dialog policy.

One example per rule and per arrow of the state diagram. Property tests are in
``tests/property/test_dialog_machine.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest

from owlbank.slot_policy import ConfirmPolicy, OptionDef, SlotSettings
from owlbank.slots.dialog import (
    Accept,
    ApplyCorrection,
    Command,
    ConfirmOption,
    Disambiguate,
    GiveUp,
    IntentSwitch,
    NotAvailable,
    ReadBackDigits,
    Reask,
    Turn,
    decide,
)
from owlbank.slots.types import CollectedSlot, OptionView, Pending, Resolution, SlotDecl

KNOWN = (OptionView("red", "OwlBank Red Card", ("red",)),
         OptionView("purple", "OwlBank Purple Card", ("purple",)),
         OptionView("rewards", "OwlBank Rewards Card", ("rewards",)))
C1001 = ("red", "purple")        # two cards on file
C1002 = ("rewards",)             # one card on file


def digits(name: str, length: int, *, sensitive: bool = False, confirm: ConfirmPolicy = "when_inferred",
           max_retries: int = 2) -> SlotSettings:
    return SlotSettings(name=name, decl=SlotDecl("digits", {"length": length}, max_retries=max_retries),
                        is_factor=True, sensitive=sensitive, confirm=confirm)


def card(confirm: ConfirmPolicy = "when_inferred", max_retries: int = 2) -> SlotSettings:
    opts = [{"value": o.value} for o in KNOWN]
    return SlotSettings(name="card", decl=SlotDecl("choice", {"options": opts}, max_retries=max_retries),
                        is_factor=False, sensitive=False, flow="confirm", confirm=confirm,
                        options=tuple(OptionDef(o.value) for o in KNOWN))


def yes_no(name: str = "confirm_payment") -> SlotSettings:
    return SlotSettings(name=name, decl=SlotDecl("yes_no", max_retries=2), is_factor=False,
                        sensitive=False, flow="confirm")


ZIP = digits("zip", 5)
SSN4 = digits("ssn4", 4, sensitive=True, confirm="never")
DOB = digits("dob", 8)


def turn(slot: SlotSettings, *, mode: str = "assisted", tries: Mapping[str, int] | None = None,
         pending: Pending | None = None, collected: Sequence[CollectedSlot] = (),
         strikes: int = 0, owned: Sequence[str] | None = None,
         settings: Mapping[str, SlotSettings] | None = None,
         caller_options: Mapping[str, tuple[str, ...]] | None = None) -> Turn:
    names = [slot.name, *(c.name for c in collected)]
    return Turn(slot=slot, mode=mode,  # type: ignore[arg-type]
                tries=dict(tries or {}), pending=pending, collected=tuple(collected),
                earlier_strikes=strikes,
                owned=frozenset(names if owned is None else owned),
                settings=dict(settings or {}), caller_options=dict(caller_options or {}))


def res(match: str, source: str = "model", **kw: Any) -> Resolution:
    if match in ("exact", "inferred", "none", "ambiguous", "correction", "intent_switch") \
            and source == "model":
        kw.setdefault("model", "ok")
    return Resolution(match=match, source=source, **kw)  # type: ignore[arg-type]


def collected(name: str, slot_type: str = "digits", *, scope: str = "current",
              verified: bool = False) -> CollectedSlot:
    return CollectedSlot(name=name, slot_type=slot_type, label=name,  # type: ignore[arg-type]
                         scope=scope, verified=verified)  # type: ignore[arg-type]


def choose(t: Turn, r: Resolution, caller: Sequence[str] = C1001) -> tuple[Any, Turn]:
    return decide(t, r, known=KNOWN, caller=caller)


def plain(t: Turn, r: Resolution) -> tuple[Any, Turn]:
    return decide(t, r, known=(), caller=())


# ------------------------------------------------------------------ Asking -> Accepted


def test_exact_digits_from_the_keypad_accepts() -> None:
    action, nxt = plain(turn(ZIP), res("exact", "dtmf", value="94105"))
    assert action == Accept("zip", "94105")
    assert nxt.tries == {} and nxt.pending is None


def test_exact_digits_from_the_model_accept_with_when_inferred() -> None:
    action, _ = plain(turn(ZIP), res("exact", value="94105"))
    assert action == Accept("zip", "94105")


def test_model_exact_with_confirm_always_is_read_back() -> None:
    action, nxt = plain(turn(replace(ZIP, confirm="always")), res("exact", value="94105"))
    assert action == ReadBackDigits("zip", "94105")
    assert nxt.pending == Pending("confirm_digits", "zip", sealed_value="94105")


def test_keypad_exact_with_confirm_always_still_accepts() -> None:
    action, _ = plain(turn(replace(ZIP, confirm="always")), res("exact", "dtmf", value="94105"))
    assert action == Accept("zip", "94105")


def test_inferred_with_confirm_never_accepts_unread_sensitive_seed() -> None:
    # SSN last-4 spoken messily: grounded, accepted unconfirmed, never read back.
    action, _ = plain(turn(SSN4), res("inferred", value="1234"))
    assert action == Accept("ssn4", "1234")


def test_exact_choice_by_alias_accepts() -> None:
    action, _ = choose(turn(card()), res("exact", "alias", value="purple"))
    assert action == Accept("card", "purple")


def test_alias_exact_with_confirm_always_accepts_but_model_exact_confirms() -> None:
    t = turn(card(confirm="always"))
    assert choose(t, res("exact", "alias", value="red"))[0] == Accept("card", "red")
    assert choose(t, res("exact", value="red"))[0] == ConfirmOption("card", "red")


# -------------------------------------------------------------- Asking -> ConfirmOption


def test_inferred_choice_is_confirmed() -> None:
    # "the one that's purple" (assisted).
    action, nxt = choose(turn(card()), res("inferred", value="purple"))
    assert action == ConfirmOption("card", "purple")
    assert nxt.pending == Pending("confirm_option", "card", option="purple")
    assert nxt.tries == {"card": 1}


def test_inferred_choice_is_confirmed_even_with_confirm_never() -> None:
    # An option the caller didn't name needs a yes.
    action, _ = choose(turn(card(confirm="never")), res("inferred", value="purple"))
    assert action == ConfirmOption("card", "purple")


# -------------------------------------------------------------- Asking -> ConfirmDigits


def test_inferred_digits_not_sensitive_are_read_back() -> None:
    # ZIP "23456, oh, no actually it is 12345".
    action, nxt = plain(turn(ZIP), res("inferred", value="12345"))
    assert action == ReadBackDigits("zip", "12345")
    assert action.key == "slot.confirm_digits"
    assert nxt.pending == Pending("confirm_digits", "zip", sealed_value="12345")
    assert nxt.tries == {"zip": 1}


def test_inferred_digits_sensitive_are_reasked_never_read_back() -> None:
    t = turn(replace(SSN4, confirm="when_inferred"))
    action, nxt = plain(t, res("inferred", value="1234"))
    assert action == Reask("clarify.digits", "ssn4")
    assert nxt.pending is None and nxt.tries == {"ssn4": 1}
    action, _ = plain(replace(t, slot=replace(SSN4, confirm="always")), res("exact", value="1234"))
    assert action == Reask("clarify.digits", "ssn4")


def test_model_digits_of_the_wrong_length_are_unclear() -> None:
    action, _ = plain(turn(ZIP), res("inferred", value="1234"))
    assert action == Reask("clarify.digits", "zip")


# --------------------------------------------------------------- Asking -> Disambiguate


def test_ambiguous_choice_lists_the_candidates_only() -> None:
    # "the OwlBank card".
    action, nxt = choose(turn(card()), res("ambiguous", candidates=("red", "purple")))
    assert action == Disambiguate("card", ("red", "purple"))
    assert nxt.pending == Pending("disambiguate", "card", candidates=("red", "purple"))
    assert nxt.tries == {"card": 1}


def test_ambiguous_choice_with_a_candidate_not_on_file_is_unclear() -> None:
    action, _ = choose(turn(card()), res("ambiguous", candidates=("red", "rewards")))
    assert action == Reask("clarify.choice", "card")


def test_ambiguous_digits_reask_and_never_list_candidates() -> None:
    action, nxt = plain(turn(ZIP), res("ambiguous", candidates=("12345", "54321")))
    assert action == Reask("clarify.digits", "zip")
    assert nxt.pending is None


# --------------------------------------------------------------- Asking -> NotAvailable


def test_known_option_not_on_file_one() -> None:
    # "the purple card", customer with only Rewards.
    action, nxt = choose(turn(card()), res("exact", "alias", value="purple"), caller=C1002)
    assert action == NotAvailable("card", "purple", ("rewards",))
    assert action.key == "slot.not_available_one"
    assert nxt.pending == Pending("not_available_one", "card", option="rewards")
    assert nxt.tries == {"card": 1}


def test_known_option_not_on_file_many() -> None:
    action, nxt = choose(turn(card()), res("inferred", value="rewards"))
    assert action == NotAvailable("card", "rewards", C1001)
    assert action.key == "slot.not_available_many"
    assert nxt.pending == Pending("not_available_many", "card", candidates=C1001)


def test_unknown_option_is_none() -> None:
    action, _ = choose(turn(card()), res("exact", value="platinum"))
    assert action == Reask("clarify.choice", "card")


# ------------------------------------------------------------- Asking -> CorrectionDone


def test_correction_of_a_current_slot_is_applied_and_the_slot_reasked() -> None:
    # "actually the ZIP should be 94105" at the SSN prompt.
    t = turn(SSN4, collected=[collected("zip")], settings={"zip": ZIP}, tries={"zip": 0})
    action, nxt = plain(t, res("correction", target="zip", value="94105"))
    assert action == ApplyCorrection("zip", "94105", False, then=Reask("prompt", "ssn4"))
    assert action.key == "slot.corrected"
    assert nxt.tries == {"zip": 1, "ssn4": 1}           # The interrupted re-ask
    assert nxt.pending is None


def test_correction_of_a_sensitive_slot_says_corrected_sensitive() -> None:
    t = turn(DOB, collected=[collected("ssn4")], settings={"ssn4": SSN4})
    action, _ = plain(t, res("correction", target="ssn4", value="6789"))
    assert isinstance(action, ApplyCorrection) and action.key == "slot.corrected_sensitive"


def test_correction_value_that_fails_is_reasked_once_for_the_target_then_resumes() -> None:
    t = turn(SSN4, collected=[collected("zip")], settings={"zip": ZIP})
    action, nxt = plain(t, res("correction", target="zip", value="123"))
    assert action == Reask("clarify.digits", "zip")
    assert nxt.pending == Pending("correction_reask", "zip")
    assert nxt.tries == {"zip": 1, "ssn4": 1}
    # The reply is valid: applied, and the interrupted slot resumes without another try.
    action2, nxt2 = plain(nxt, res("exact", "speech", value="94105"))
    assert action2 == ApplyCorrection("zip", "94105", False,
                                      then=Reask("prompt", "ssn4", consume=False))
    assert nxt2.tries == nxt.tries and nxt2.pending is None
    # The reply is the model's correction of the same slot: a model value, so inferred, and the
    # ZIP's confirm (when_inferred) reads it back first (the re-ask follows the slot's rules).
    action4, nxt4 = plain(nxt, res("correction", target="zip", value="94105"))
    assert action4 == ReadBackDigits("zip", "94105")
    assert nxt4.pending == Pending("confirm_digits", "zip", sealed_value="94105")
    assert nxt4.tries == {"zip": 2, "ssn4": 1}
    # The reply fails again: no correction, the interrupted slot resumes.
    action3, nxt3 = plain(nxt, res("none", "none"))
    assert action3 == Reask("prompt", "ssn4", consume=False)
    assert nxt3.tries == nxt.tries


def test_correction_of_an_unasked_slot_is_none() -> None:
    # "the last four are 6789" before ssn4 is asked stores nothing.
    action, nxt = plain(turn(ZIP), res("correction", target="ssn4", value="6789"))
    assert action == Reask("clarify.digits", "zip")
    assert nxt.tries == {"zip": 1}


def test_correction_of_an_unknown_slot_is_none() -> None:
    action, _ = plain(turn(ZIP), res("correction", target="favourite_colour", value="12345"))
    # 12345 grounds for zip, so it is the answer (inferred -> read back).
    assert action == ReadBackDigits("zip", "12345")
    action, _ = plain(turn(ZIP), res("correction", target="favourite_colour", value="red"))
    assert action == Reask("clarify.digits", "zip")


def test_correction_of_an_earlier_flow_slot_reasks_then_switches() -> None:
    t = turn(SSN4, collected=[collected("zip", scope="earlier")], settings={"zip": ZIP})
    action, nxt = plain(t, res("correction", target="zip", value="94105"))
    assert action == Reask("clarify.digits", "ssn4")
    assert nxt.earlier_strikes == 1 and nxt.tries == {"ssn4": 1}
    action2, nxt2 = plain(nxt, res("correction", target="zip", value="94105"))
    assert action2 == IntentSwitch("earlier_correction")
    assert nxt2.tries == nxt.tries


def test_earlier_strikes_count_consecutive_corrections_only() -> None:
    t = turn(SSN4, collected=[collected("zip", scope="earlier")], strikes=1)
    action, nxt = plain(t, res("none", "none"))
    assert isinstance(action, Reask) and nxt.earlier_strikes == 0


def test_correction_of_a_verified_slot_never_verifies() -> None:
    # Re-asked like a finished flow's slot; only the question's counter moves.
    t = turn(DOB, collected=[collected("zip", verified=True)], settings={"zip": ZIP},
             tries={"zip": 1, "dob": 0})
    action, nxt = plain(t, res("correction", target="zip", value="94105"))
    assert action == Reask("clarify.digits", "dob")
    assert nxt.tries == {"zip": 1, "dob": 1}


def test_confirm_flow_correction_of_the_card_is_applied() -> None:
    # "no, the red one" during the read-back.
    t = turn(yes_no(), collected=[collected("card", "choice")], owned=["card"],
             settings={"card": card()}, caller_options={"card": C1001})
    # The caller named the red card (the front door checked the words: ``named``).
    t = replace(t, named=frozenset({"card"}))
    action, nxt = plain(t, res("correction", target="card", value="red"))
    assert action == ApplyCorrection("card", "red", False, then=Reask("prompt", "confirm_payment"))
    assert nxt.tries == {"card": 1, "confirm_payment": 1}


def test_confirm_flow_correction_to_a_card_not_on_file_is_reasked() -> None:
    t = turn(yes_no(), collected=[collected("card", "choice")], owned=["card"],
             settings={"card": card()}, caller_options={"card": C1002})
    action, nxt = plain(t, res("correction", target="card", value="red"))
    assert action == Reask("clarify.choice", "card")
    assert nxt.pending == Pending("correction_reask", "card")


def test_confirm_flow_change_to_an_argument_it_does_not_own_is_an_intent_switch() -> None:
    # The amount is the AI's argument.
    t = turn(yes_no(), collected=[collected("card", "choice"), collected("amount")], owned=["card"],
             settings={"card": card()})
    action, _ = plain(t, res("correction", target="amount", value="30"))
    assert action == IntentSwitch("not_owned")


# --------------------------------------------------------------- Asking -> Asking (none)


def test_none_reasks_and_never_selects() -> None:
    action, nxt = choose(turn(card()), res("none"))
    assert action == Reask("clarify.choice", "card")
    assert nxt.tries == {"card": 1}


def test_none_with_a_value_still_never_selects() -> None:
    action, _ = choose(turn(card()), res("none", value="red"))
    assert action == Reask("clarify.choice", "card")


# --------------------------------------------------------------------- Asking -> AI


def test_intent_switch_with_no_value_goes_to_the_ai() -> None:
    # "actually, can I get my recent transactions" at the SSN prompt.
    action, nxt = plain(turn(SSN4, pending=None), res("intent_switch"))
    assert action == IntentSwitch()
    assert nxt.tries == {}


def test_answer_first_a_grounded_value_beats_the_intent_switch() -> None:
    # "yes my last four are 1234, and what's my balance?" -> 1234 is the answer.
    action, _ = plain(turn(SSN4), res("intent_switch", value="1234"))
    assert action == Accept("ssn4", "1234")


def test_answer_first_choice_value_is_confirmed() -> None:
    action, _ = choose(turn(card()), res("intent_switch", value="purple"))
    assert action == ConfirmOption("card", "purple")


@pytest.mark.parametrize("kind", ["escalate", "cancel"])
def test_commands_end_the_flow(kind: str) -> None:
    action, nxt = plain(turn(ZIP, pending=Pending("confirm_digits", "zip", sealed_value="1")),
                        res(kind, "command"))
    assert action == Command(kind)  # type: ignore[arg-type]
    assert nxt.pending is None


def test_repeat_keeps_the_question_and_uses_no_try() -> None:
    p = Pending("confirm_option", "card", option="red")
    action, nxt = choose(turn(card(), pending=p, tries={"card": 1}, strikes=1),
                         res("repeat", "command"))
    assert action == Command("repeat")
    assert nxt.pending == p and nxt.tries == {"card": 1} and nxt.earlier_strikes == 1


# ------------------------------------------------------------------ ConfirmOption -> *


def _confirming(option: str = "purple") -> Turn:
    return turn(card(), pending=Pending("confirm_option", "card", option=option), tries={"card": 1})


def test_confirm_option_deterministic_yes_accepts() -> None:
    action, nxt = choose(_confirming(), res("exact", "speech", reply="yes", reply_deterministic=True))
    assert action == Accept("card", "purple")
    assert nxt.pending is None and nxt.tries == {"card": 1}


def test_confirm_option_model_yes_is_unclear() -> None:
    action, nxt = choose(_confirming(), res("inferred", reply="yes"))
    assert action == Reask("clarify.yes_no", "card")
    assert nxt.pending == _confirming().pending and nxt.tries == {"card": 2}


def test_confirm_option_no_reasks_the_choice() -> None:
    action, nxt = choose(_confirming(), res("exact", "speech", reply="no", reply_deterministic=True))
    assert action == Reask("prompt", "card")
    assert nxt.pending is None and nxt.tries == {"card": 2}


def test_confirm_option_no_i_meant_y_exact_selects_y() -> None:
    # "no, I meant the red one" to "Did you mean the OwlBank Purple Card?".
    r = res("exact", "alias", value="red", reply="no", reply_deterministic=True)
    action, nxt = choose(_confirming(), r)
    assert action == Accept("card", "red")
    assert nxt.tries == {"card": 2}                    # The reply uses one try


def test_confirm_option_no_i_meant_y_inferred_confirms_y() -> None:
    action, nxt = choose(_confirming(), res("inferred", value="red", reply="no"))
    assert action == ConfirmOption("card", "red")
    assert nxt.pending == Pending("confirm_option", "card", option="red")


def test_confirm_option_y_matching_several_disambiguates() -> None:
    action, _ = choose(_confirming(), res("ambiguous", candidates=("red", "purple"), reply="no"))
    assert action == Disambiguate("card", ("red", "purple"))


def test_confirm_option_y_known_not_on_file_is_not_available() -> None:
    action, _ = choose(_confirming(), res("exact", "alias", value="rewards", reply="no"))
    assert action == NotAvailable("card", "rewards", C1001)


def test_confirm_option_unclear_reply_repeats_the_question() -> None:
    action, nxt = choose(_confirming(), res("none"))
    assert action == Reask("clarify.yes_no", "card")
    assert nxt.pending == _confirming().pending


def test_a_yes_on_a_none_is_ignored() -> None:
    # Fail closed: a none never selects; the engine sends a deterministic reply as exact.
    action, _ = choose(_confirming(), res("none", "speech", reply="yes", reply_deterministic=True))
    assert action == Reask("clarify.yes_no", "card")


# ------------------------------------------------------------------ ConfirmDigits -> *


def _reading_back() -> Turn:
    return turn(ZIP, pending=Pending("confirm_digits", "zip", sealed_value="12345"), tries={"zip": 1})


def test_read_back_deterministic_yes_accepts_the_pending_value() -> None:
    action, nxt = plain(_reading_back(), res("exact", "speech", value="yes"))
    assert action == Accept("zip", "12345")
    assert nxt.pending is None
    action, _ = plain(_reading_back(), res("exact", "dtmf", reply="yes", reply_deterministic=True))
    assert action == Accept("zip", "12345")


def test_read_back_model_yes_is_unclear() -> None:
    action, nxt = plain(_reading_back(), res("inferred", value="yes"))
    assert action == Reask("clarify.yes_no", "zip")
    assert nxt.pending == _reading_back().pending


def test_read_back_no_reasks_with_one_try() -> None:
    action, nxt = plain(_reading_back(), res("exact", "speech", value="no"))
    assert action == Reask("prompt", "zip")
    assert nxt.pending is None and nxt.tries == {"zip": 2}


def test_read_back_new_value_is_a_fresh_answer() -> None:
    # "no, it's 54321": inferred -> read back again, one try.
    action, nxt = plain(_reading_back(), res("inferred", value="54321", reply="no"))
    assert action == ReadBackDigits("zip", "54321")
    assert nxt.tries == {"zip": 2}
    action, _ = plain(_reading_back(), res("exact", "dtmf", value="54321"))
    assert action == Accept("zip", "54321")


# ------------------------------------------------------------------- Disambiguate -> *


def _disambiguating() -> Turn:
    return turn(card(), pending=Pending("disambiguate", "card", candidates=C1001), tries={"card": 1})


def test_disambiguate_one_candidate_named_exactly_accepts() -> None:
    action, _ = choose(_disambiguating(), res("exact", "alias", value="red"))
    assert action == Accept("card", "red")


def test_disambiguate_one_candidate_inferred_confirms() -> None:
    action, nxt = choose(_disambiguating(), res("inferred", value="red"))
    assert action == ConfirmOption("card", "red")
    assert nxt.tries == {"card": 2}


def test_disambiguate_none_reasks() -> None:
    action, nxt = choose(_disambiguating(), res("none"))
    assert action == Reask("clarify.choice", "card")
    assert nxt.pending is None and nxt.tries == {"card": 2}


def test_disambiguate_yes_is_not_an_answer() -> None:
    action, _ = choose(_disambiguating(), res("exact", "speech", reply="yes", reply_deterministic=True))
    assert action == Reask("clarify.choice", "card")


# ------------------------------------------------------------------- NotAvailable -> *


def _not_available_one() -> Turn:
    return turn(card(), pending=Pending("not_available_one", "card", option="rewards"),
                tries={"card": 1})


def test_not_available_one_deterministic_yes_selects_the_card_on_file() -> None:
    action, _ = choose(_not_available_one(),
                       res("exact", "speech", reply="yes", reply_deterministic=True), caller=C1002)
    assert action == Accept("card", "rewards")


def test_not_available_one_model_yes_is_unclear() -> None:
    action, _ = choose(_not_available_one(), res("inferred", reply="yes"), caller=C1002)
    assert action == Reask("clarify.yes_no", "card")


def test_not_available_one_no_reasks() -> None:
    action, nxt = choose(_not_available_one(),
                         res("exact", "speech", reply="no", reply_deterministic=True), caller=C1002)
    assert action == Reask("prompt", "card")
    assert nxt.tries == {"card": 2}


def test_not_available_many_reply_is_a_fresh_answer() -> None:
    t = turn(card(), pending=Pending("not_available_many", "card", candidates=C1001),
             tries={"card": 1})
    assert choose(t, res("exact", "alias", value="red"))[0] == Accept("card", "red")
    action, nxt = choose(t, res("none"))
    assert action == Reask("clarify.choice", "card") and nxt.tries == {"card": 2}


# --------------------------------------------------------------- yes_no slots


def test_yes_no_slot_deterministic_yes_accepts_model_yes_does_not() -> None:
    t = turn(yes_no())
    assert plain(t, res("exact", "speech", value="yes"))[0] == Accept("confirm_payment", "yes")
    assert plain(t, res("inferred", value="yes"))[0] == Reask("clarify.yes_no", "confirm_payment")
    assert plain(t, res("inferred", value="no"))[0] == Accept("confirm_payment", "no")


def test_what_is_my_balance_during_the_read_back_is_an_intent_switch() -> None:
    action, _ = plain(turn(yes_no()), res("intent_switch"))
    assert action == IntentSwitch()


# ------------------------------------------------------------------ counters


def test_over_max_retries_gives_up() -> None:
    action, nxt = plain(turn(ZIP, tries={"zip": 2}), res("none"))
    assert action == GiveUp("zip")
    assert nxt.tries == {"zip": 3} and nxt.pending is None


def test_a_correction_over_the_targets_retries_gives_up() -> None:
    t = turn(SSN4, collected=[collected("zip")], settings={"zip": ZIP}, tries={"zip": 2})
    action, _ = plain(t, res("correction", target="zip", value="94105"))
    assert action == GiveUp("zip")


def test_no_action_decrements_a_counter() -> None:
    t = turn(card(), tries={"card": 1, "zip": 2})
    for r in (res("exact", "alias", value="red"), res("none"), res("intent_switch"),
              res("repeat", "command")):
        _, nxt = choose(t, r)
        assert all(nxt.tries.get(k, 0) >= v for k, v in t.tries.items())


# ------------------------------------------------------------ deterministic (fail closed)


@pytest.mark.parametrize("r", [res("intent_switch"), res("correction", target="zip", value="94105"),
                               res("inferred", value="12345")])
def test_deterministic_mode_treats_model_kinds_as_none(r: Resolution) -> None:
    t = turn(ZIP, mode="deterministic", collected=[collected("dob")], settings={"dob": DOB})
    action, _ = plain(t, r)
    assert action == Reask("clarify.digits", "zip")


# ------------------------------------------------------------------------- Turn checks


def test_turn_refuses_a_follow_up_about_another_slot() -> None:
    with pytest.raises(ValueError):
        turn(ZIP, pending=Pending("confirm_digits", "dob", sealed_value="1"))
    with pytest.raises(ValueError):
        turn(ZIP, pending=Pending("correction_reask", "dob"))      # no settings for dob
    with pytest.raises(ValueError):
        turn(ZIP, strikes=-1)


# ------------------------------------- the corrected slot's re-ask follows its rules


def _zip_reask(zip_: SlotSettings = ZIP, **kw: Any) -> Turn:
    """At the SSN prompt, "actually the ZIP is wrong": the ZIP's re-ask is pending."""
    t = turn(SSN4, collected=[collected("zip")], settings={"zip": zip_}, **kw)
    action, nxt = plain(t, res("correction", target="zip", value="123"))
    assert action == Reask("clarify.digits", "zip")
    return nxt


RESUME = Reask("prompt", "ssn4", consume=False)


def test_reask_inferred_digits_are_read_back_then_a_deterministic_yes_applies() -> None:
    nxt = _zip_reask()
    action, nxt2 = plain(nxt, res("inferred", value="94105"))
    assert action == ReadBackDigits("zip", "94105")
    assert nxt2.pending == Pending("confirm_digits", "zip", sealed_value="94105")
    assert nxt2.tries == {"zip": 2, "ssn4": 1}          # the read-back uses one try of the ZIP
    # A deterministic yes: applied, and the SSN prompt resumes with no new try.
    action2, nxt3 = plain(nxt2, res("exact", "dtmf", reply="yes", reply_deterministic=True))
    assert action2 == ApplyCorrection("zip", "94105", False, then=RESUME)
    assert nxt3.pending is None and nxt3.tries == nxt2.tries


def test_reask_read_back_model_yes_or_no_drops_the_correction() -> None:
    nxt = _zip_reask()
    _, nxt2 = plain(nxt, res("inferred", value="94105"))
    for r in (res("inferred", reply="yes"),                          # a model yes
              res("exact", "dtmf", reply="no", reply_deterministic=True),
              res("none", "none")):
        action, nxt3 = plain(nxt2, r)
        assert action == RESUME, r
        assert nxt3.pending is None and nxt3.tries == nxt2.tries


def test_reask_read_back_new_value_is_read_back_again() -> None:
    nxt = _zip_reask(digits("zip", 5, max_retries=3))
    _, nxt2 = plain(nxt, res("inferred", value="94105"))
    action, nxt3 = plain(nxt2, res("inferred", value="12345"))
    assert action == ReadBackDigits("zip", "12345")
    assert nxt3.pending == Pending("confirm_digits", "zip", sealed_value="12345")


def test_reask_read_backs_stop_at_the_targets_retries_and_resume() -> None:
    nxt = _zip_reask(digits("zip", 5, max_retries=1))      # the re-ask used the ZIP's one retry
    action, nxt2 = plain(nxt, res("inferred", value="94105"))
    assert action == RESUME                                # no read-back left: dropped
    assert nxt2.pending is None


def test_reask_inferred_digits_with_confirm_never_are_applied() -> None:
    nxt = _zip_reask(digits("zip", 5, confirm="never"))
    action, _ = plain(nxt, res("inferred", value="94105"))
    assert action == ApplyCorrection("zip", "94105", False, then=RESUME)


def test_reask_inferred_sensitive_digits_are_never_read_back_and_dropped() -> None:
    t = turn(ZIP, collected=[collected("ssn4")], settings={"ssn4": digits("ssn4", 4, sensitive=True)})
    _, nxt = plain(t, res("correction", target="ssn4", value="12"))
    assert nxt.pending == Pending("correction_reask", "ssn4")
    action, nxt2 = plain(nxt, res("inferred", value="6789"))
    assert action == Reask("prompt", "zip", consume=False)     # Never spoken, re-asked once
    assert nxt2.pending is None


def _card_reask(caller_opts: tuple[str, ...] = C1001) -> Turn:
    t = turn(yes_no(), collected=[collected("card", "choice")], owned=["card"],
             settings={"card": card()}, caller_options={"card": caller_opts})
    _, nxt = plain(t, res("correction", target="card", value="gold"))      # not an option
    assert nxt.pending == Pending("correction_reask", "card")
    return nxt


def test_reask_inferred_card_is_confirmed_then_a_deterministic_yes_applies() -> None:
    nxt = _card_reask()
    action, nxt2 = choose(nxt, res("inferred", value="purple"))
    assert action == ConfirmOption("card", "purple")
    assert nxt2.pending == Pending("confirm_option", "card", option="purple")
    action2, nxt3 = choose(nxt2, res("exact", "dtmf", reply="yes", reply_deterministic=True))
    assert action2 == ApplyCorrection("card", "purple", False,
                                      then=Reask("prompt", "confirm_payment", consume=False))
    assert nxt3.pending is None
    # A model yes selects nothing.
    action3, _ = choose(nxt2, res("inferred", reply="yes"))
    assert action3 == Reask("prompt", "confirm_payment", consume=False)


def test_reask_named_card_is_applied() -> None:
    action, _ = choose(_card_reask(), res("exact", "alias", value="red"))
    assert action == ApplyCorrection("card", "red", False,
                                     then=Reask("prompt", "confirm_payment", consume=False))


def test_reask_ambiguous_card_disambiguates_then_an_exact_name_applies() -> None:
    nxt = _card_reask()
    action, nxt2 = choose(nxt, res("ambiguous", candidates=("red", "purple")))
    assert action == Disambiguate("card", ("red", "purple"))
    action2, _ = choose(nxt2, res("exact", "speech", value="red"))
    assert isinstance(action2, ApplyCorrection) and action2.value == "red"


def test_reask_card_not_on_file_asks_not_available() -> None:
    nxt = _card_reask(C1002)
    action, nxt2 = choose(nxt, res("exact", "alias", value="purple"), caller=C1002)
    assert action == NotAvailable("card", "purple", C1002)
    action2, _ = choose(nxt2, res("exact", "dtmf", reply="yes", reply_deterministic=True),
                        caller=C1002)
    assert action2 == ApplyCorrection("card", "rewards", False,
                                      then=Reask("prompt", "confirm_payment", consume=False))


def test_a_choice_correction_the_caller_did_not_name_is_confirmed_first() -> None:
    # The model proposes the red card, but the words didn't name it (``named`` empty).
    t = turn(yes_no(), collected=[collected("card", "choice")], owned=["card"],
             settings={"card": card()}, caller_options={"card": C1001})
    action, nxt = plain(t, res("correction", target="card", value="red"))
    assert action == ConfirmOption("card", "red")
    assert nxt.pending == Pending("confirm_option", "card", option="red")
    assert nxt.tries == {"card": 1, "confirm_payment": 1}
    action2, _ = choose(nxt, res("exact", "dtmf", reply="yes", reply_deterministic=True))
    assert action2 == ApplyCorrection("card", "red", False,
                                      then=Reask("prompt", "confirm_payment", consume=False))
    action3, _ = choose(nxt, res("inferred", reply="yes"))
    assert not isinstance(action3, (Accept, ApplyCorrection))


def test_turn_accepts_a_follow_up_about_a_collected_slot_with_settings() -> None:
    t = turn(SSN4, collected=[collected("zip")], settings={"zip": ZIP},
             pending=Pending("confirm_digits", "zip", sealed_value="94105"))
    assert t.correcting() and t.answered() is ZIP


# ------------------------------------------------------------------ decline


@pytest.mark.parametrize("mode", ["assisted", "generative"])
def test_a_model_decline_on_a_declinable_slot_is_the_decline_command(mode: str) -> None:
    t = turn(replace(ZIP, declinable=True), mode=mode, tries={"zip": 1})
    action, nxt = plain(t, res("decline", model="ok"))
    assert action == Command("decline")
    assert nxt.tries == {"zip": 1}


@pytest.mark.parametrize("mode", ["deterministic", "assisted", "generative"])
def test_a_keypad_decline_on_a_declinable_slot_is_the_decline_command(mode: str) -> None:
    action, _ = plain(turn(replace(SSN4, declinable=True), mode=mode), res("decline", "dtmf"))
    assert action == Command("decline")


def test_a_decline_with_a_read_back_pending_is_still_the_decline_command() -> None:
    p = Pending("confirm_digits", "zip", sealed_value="94105")
    action, _ = plain(turn(replace(ZIP, declinable=True), pending=p), res("decline", "dtmf"))
    assert action == Command("decline")


def test_a_decline_on_a_slot_that_isnt_declinable_is_none() -> None:
    t = turn(ZIP)
    assert plain(t, res("decline", model="ok")) == plain(t, res("none", model="ok"))
    assert plain(t, res("decline", "dtmf")) == plain(t, res("none", "dtmf"))


@pytest.mark.parametrize("source", ["model", "speech", "command"])
def test_deterministic_mode_takes_a_decline_only_from_the_keypad(source: str) -> None:
    action, _ = plain(turn(replace(ZIP, declinable=True), mode="deterministic"), res("decline", source))
    assert action != Command("decline")
