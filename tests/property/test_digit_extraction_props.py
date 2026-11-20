"""Digits spoken inside a longer sentence are found; an ambiguous answer is refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.policy import Policy
from owlbank.slot_policy import SlotSettings
from owlbank.slots.engine import make_context, resolve

POLICY = json.loads((Path(__file__).resolve().parents[2] / "data" / "policy.json").read_text())
EN_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
#: Words that carry meaning (kept by normalization, not filler), for "the caller kept talking".
CONTENT = ["hold", "on", "what's", "balance", "transfer", "agent", "wait", "actually", "checking",
           "savings", "can", "check"]
FILLER = ["yeah", "the", "last", "are", "my", "is", "code", "it's", "um", "so", "okay", "zip", "number",
          "thanks", "I", "think", "that's", "right", "here", "you", "go"]


def _slot(name: str) -> SlotSettings:
    """The seed's settings for a factor (``resolve`` takes them)."""
    return Policy.from_dict(POLICY).slot(name)


def _say(slot: SlotSettings, text: str, lang: str = "en-US"):
    return resolve(slot, digits=None, speech=text, confidence=0.9, lang=lang, mode="deterministic",
                   ctx=make_context(question_text="q", labels={}, commands={}), resolver=None)


@pytest.mark.parametrize("factor,text,want", [
    ("ssn4", "Yeah the last four are 1111", "1111"),
    ("ssn4", "yeah the last four are one one one one", "1111"),
    ("ssn4", "the last four 1111", "1111"),
    ("zip", "my zip is 94105", "94105"),
    ("zip", "I live at 12 Main, zip 94105", "94105"),
    ("otp", "okay it's 472195 thanks", "472195"),
])
def test_digits_inside_a_sentence(factor, text, want) -> None:
    r = _say(_slot(factor), text)
    assert (r.match, r.value) == ("exact", want)


@pytest.mark.parametrize("factor,text", [
    ("ssn4", "1111 or maybe 2222"),
    ("ssn4", "I don't know"),
    ("zip", "my zip is 9410"),
])
def test_ambiguous_or_missing_is_not_guessed(factor, text) -> None:
    assert _say(_slot(factor), text).match != "exact"


def test_spanish_sentence() -> None:
    r = _say(_slot("ssn4"), "sí, los últimos cuatro son uno uno uno uno", "es-US")
    assert (r.match, r.value) == ("exact", "1111")


@settings(max_examples=300)
@given(factor=st.sampled_from(["ssn4", "zip", "otp"]), data=st.data(),
       as_words=st.booleans())
def test_property_one_answer_in_any_filler_is_found(factor, data, as_words) -> None:
    """For any digit string of the factor's length, said as digits or digit words, after any
    filler words and followed by at most two (the caller stopped after the number), the
    parser returns exactly that string."""
    n = POLICY["factors"][factor]["constraints"]["length"]
    digits = data.draw(st.text("0123456789", min_size=n, max_size=n))
    spoken = " ".join(EN_WORDS[int(d)] for d in digits) if as_words else digits
    before = data.draw(st.lists(st.sampled_from(FILLER), max_size=5))
    after = data.draw(st.lists(st.sampled_from(FILLER), max_size=2))
    r = _say(_slot(factor), " ".join([*before, spoken, *after]))
    assert (r.match, r.value) == ("exact", digits)


@settings(max_examples=200)
@given(factor=st.sampled_from(["ssn4", "zip", "otp"]), data=st.data())
def test_property_two_different_answers_are_never_guessed(factor, data) -> None:
    """Two different digit strings of the factor's length in one sentence: never a value."""
    n = POLICY["factors"][factor]["constraints"]["length"]
    a = data.draw(st.text("0123456789", min_size=n, max_size=n))
    b = data.draw(st.text("0123456789", min_size=n, max_size=n).filter(lambda x: x != a))
    sep = data.draw(st.lists(st.sampled_from(FILLER), min_size=1, max_size=3))
    assert _say(_slot(factor), " ".join([a, *sep, b])).match != "exact"


@settings(max_examples=200)
@given(factor=st.sampled_from(["ssn4", "zip", "otp"]), data=st.data())
def test_property_a_caller_who_keeps_talking_is_left_to_the_resolver(factor, data) -> None:
    """Three or more meaningful words after the number: not taken from the sentence (it may be a
    change of subject, which the resolver decides; with no resolver the answer stays unclear).
    Hesitations ("yeah", "um") are removed by normalization and don't count."""
    n = POLICY["factors"][factor]["constraints"]["length"]
    digits = data.draw(st.text("0123456789", min_size=n, max_size=n))
    before = data.draw(st.lists(st.sampled_from(FILLER), min_size=1, max_size=3))
    after = data.draw(st.lists(st.sampled_from(CONTENT), min_size=3, max_size=6))
    assert _say(_slot(factor), " ".join([*before, digits, *after])).match != "exact"
