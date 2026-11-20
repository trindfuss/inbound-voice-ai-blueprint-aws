"""``sensitive.placeholder_text``, the answering turn's text for a sensitive slot.

The fixed cases, plus the rules' edges: a run of 2 or more number tokens,
or one that supplied digits of the value (``value_runs``), or one that doesn't read, becomes
``[<slot name>]``; with a value but no located runs (a keypad answer with speech) every one-token
run whose digit occurs in the value does too (fallback); a ``yes_no`` or ``choice`` span becomes
the token, and with no span the whole utterance does; the value itself is replaced wherever it
occurs; adjacent tokens collapse to one.
"""

from __future__ import annotations

import unicodedata

import pytest

from owlbank.sensitive import placeholder_text
from owlbank.slot_policy import SlotSettings
from owlbank.slots import number_runs
from owlbank.slots.types import SlotDecl

SSN4 = SlotSettings(name="ssn4", decl=SlotDecl("digits", {"length": 4}), is_factor=True,
                    sensitive=True)
ZIP = SlotSettings(name="zip", decl=SlotDecl("digits", {"length": 5}), is_factor=True,
                   sensitive=True)
YES_NO = SlotSettings(name="yes_no", decl=SlotDecl("yes_no"), is_factor=False, sensitive=True,
                      flow="confirm")
ACCOUNT = SlotSettings(
    name="account",
    decl=SlotDecl("choice", {"options": [{"value": "savings", "labelKey": "label.savings"},
                                         {"value": "checking", "labelKey": "label.checking"}]}),
    is_factor=False, sensitive=True, flow="confirm")


def _run_index(text: str, piece: str) -> int:
    """Index (in ``number_runs(text, ("en", "es"))``) of the run that is exactly ``piece``."""
    s = unicodedata.normalize("NFC", text)
    for i, run in enumerate(number_runs(s, ("en", "es"))):
        if s[run.start:run.end] == piece:
            return i
    raise AssertionError(f"no run {piece!r}")


# --------------------------------------------------------------------------- phase 1

def test_lone_word_that_supplied_nothing_is_kept() -> None:
    text = "yes my last four are 1234"
    assert placeholder_text(text, SSN4, "1234", value_runs=[_run_index(text, "1234")]) \
        == "yes my last four are [ssn4]"


def test_value_split_across_two_runs() -> None:
    text = "one, uh, two three four"
    runs = [_run_index(text, "one"), _run_index(text, "two three four")]
    assert placeholder_text(text, SSN4, "1234", value_runs=runs) == "[ssn4], uh, [ssn4]"


def test_two_token_run_that_supplied_nothing_is_replaced() -> None:
    text = "1 2, no, 1234"
    assert placeholder_text(text, SSN4, "1234", value_runs=[_run_index(text, "1234")]) \
        == "[ssn4], no, [ssn4]"


def test_spanish_number_words() -> None:
    text = "uno dos tres cuatro"
    assert placeholder_text(text, SSN4, "1234", value_runs=[0]) == "[ssn4]"


def test_no_value_keeps_a_lone_word() -> None:
    text = "what's my balance? I can't find my four digits"
    assert placeholder_text(text, SSN4, None) == text


def test_keypad_value_with_speech_uses_the_fallback() -> None:
    # The value can't be located in the words, so a lone word whose digit is in it goes.
    assert placeholder_text("four", SSN4, "1234") == "[ssn4]"


def test_fallback_keeps_a_lone_word_whose_digit_is_not_in_the_value() -> None:
    assert placeholder_text("my last four", SSN4, "1235") == "my last four"
    assert placeholder_text("my last four", SSN4, "1234") == "my last [ssn4]"


def test_spanish_lone_o_and_una_survive() -> None:
    # A lone "o" (English zero) or "una" stays unless it supplied a digit of the value.
    text = "quiero una tarjeta o la otra"
    assert placeholder_text(text, SSN4, None) == text
    located = "quiero una tarjeta o la otra, son 5678"
    assert placeholder_text(located, SSN4, "5678", value_runs=[_run_index(located, "5678")]) \
        == "quiero una tarjeta o la otra, son [ssn4]"


def test_spanish_three_token_run_is_replaced() -> None:
    assert placeholder_text("uno o dos", SSN4, None) == "[ssn4]"


def test_run_that_does_not_read_is_replaced() -> None:
    # Fail closed: "first" doesn't read as digits.
    text = "it was the first time, 1234"
    assert placeholder_text(text, SSN4, "1234", value_runs=[_run_index(text, "1234")]) \
        == "it was the [ssn4] time, [ssn4]"


def test_one_word_run_with_two_digits_is_replaced() -> None:
    # Outside the tokens no number run of 2 or more digits survives ("fifteen" is 15).
    assert placeholder_text("about fifteen minutes", SSN4, None) == "about [ssn4] minutes"


def test_sensitive_yes_no_replaces_the_span() -> None:
    text = "yes that's right"
    assert placeholder_text(text, YES_NO, "yes", span=(0, 3)) == "[yes_no] that's right"


def test_sensitive_choice_replaces_the_label_match() -> None:
    text = "the savings account please"
    start = text.index("savings")
    assert placeholder_text(text, ACCOUNT, "savings", span=(start, start + len("savings"))) \
        == "the [account] account please"


@pytest.mark.parametrize("slot,value,text", [
    (YES_NO, "yes", "yeah sure why not"),
    (ACCOUNT, "checking", "the one I use every day"),
])
def test_yes_no_or_choice_without_a_span_replaces_the_whole_utterance(
        slot: SlotSettings, value: str, text: str) -> None:
    assert placeholder_text(text, slot, value) == f"[{slot.name}]"


def test_yes_no_with_no_value_and_no_span_applies_the_run_rules_only() -> None:
    assert placeholder_text("what's my balance", YES_NO, None) == "what's my balance"


# --------------------------------------------------------------------------- the other rules

def test_value_replaced_wherever_it_occurs() -> None:
    # Rule 4: the value itself, with any case, spaces or hyphens.
    text = "SAVINGS, I said sav-ings"
    assert placeholder_text(text, ACCOUNT, "savings", span=(0, 7)) == "[account], I said [account]"


def test_adjacent_tokens_collapse() -> None:
    text = "1234; 5678"
    assert placeholder_text(text, SSN4, "1234", value_runs=[0]) == "[ssn4]"
    assert placeholder_text("94105! 9 4 1 0 5", ZIP, "94105", value_runs=[0, 1]) == "[zip]"


def test_tokens_separated_by_a_word_do_not_collapse() -> None:
    assert placeholder_text("1234 or 5678", SSN4, "5678", value_runs=[1]) == "[ssn4] or [ssn4]"


def test_value_runs_out_of_range_are_ignored() -> None:
    assert placeholder_text("four", SSN4, "1234", value_runs=[7]) == "four"


def test_empty_and_runless_text() -> None:
    assert placeholder_text("", SSN4, "1234") == ""
    assert placeholder_text("I don't know", SSN4, "1234") == "I don't know"


def test_text_is_nfc_and_span_indexes_the_nfc_text() -> None:
    # "sí" decomposed is 3 code points; the span (0, 2) indexes the composed text.
    decomposed = unicodedata.normalize("NFD", "sí, cuatro")
    assert placeholder_text(decomposed, YES_NO, "yes", span=(0, 2)) == "[yes_no], cuatro"
    assert placeholder_text(decomposed, SSN4, None) == unicodedata.normalize("NFC", decomposed)


def test_slot_name_is_the_token() -> None:
    otp = SlotSettings(name="otp", decl=SlotDecl("digits", {"length": 6}), is_factor=True,
                       sensitive=True)
    assert placeholder_text("it's 482 913", otp, "482913", value_runs=[0]) == "it's [otp]"


def test_a_mixed_language_run_the_value_came_from_is_still_cut() -> None:
    """A property test found that "four 0 0 0 cero" reads as no digits over en+es, but the
    located parse points at it, and the placeholder cuts it (rule 1, fail closed)."""
    from owlbank.slots.validators import parse_speech_located

    speech = "four 0 0 0 cero"
    got = parse_speech_located(SSN4.decl, speech, "en-US")
    assert got.value is not None and got.value_runs == (0,)
    assert number_runs(speech, ("en", "es"))[0].digits is None
    assert placeholder_text(speech, SSN4, got.value, value_runs=got.value_runs) == "[ssn4]"


@pytest.mark.parametrize("speech", [
    "the first is one and then two and then three and four",
    "uh 7, wait, 3 hold on 9 then 1",
    "uno y luego dos y tres y cuatro",
])
def test_every_run_cuts_single_digits_split_by_filler(speech: str) -> None:
    """Rule 5: with no value known yet, one-token runs survive rule 1, so a
    value said digit by digit with filler would get through; ``every_run`` cuts them all."""
    out = placeholder_text(speech, SSN4, None, every_run=True)
    assert number_runs(out.replace("[ssn4]", " "), ("en", "es")) == []
    assert "[ssn4]" in out


def test_every_run_off_keeps_the_existing_rule() -> None:
    speech = "the first is one and then two and then three and four"
    assert "one" in placeholder_text(speech, SSN4, None)
