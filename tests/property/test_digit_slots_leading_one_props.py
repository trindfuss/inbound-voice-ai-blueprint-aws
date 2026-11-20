"""The leading-one drop is the phone slot's only.

Dropping a spoken leading "one" before any digit run would make an over-long
reading on other digit slots lose its first digit ("one 2 3 4 5" read as 2345 for the last four of an SSN). Only the
phone slot reads a leading one as a country code; every other digits slot reads an over-long
reading starting with 1 as at 4b52773: no value (the caller is asked again, or the resolver reads
it). P: for a non-phone digits slot, an over-long run that starts with a 1 never gives an exact
value (never one shorter than the caller's digits).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.policy import Policy
from owlbank.slot_policy import SlotSettings
from owlbank.slots import SlotDecl
from owlbank.slots.engine import make_context, resolve

POLICY = Policy.from_dict(json.loads(
    (Path(__file__).resolve().parents[2] / "data" / "policy.json").read_text()))
EN_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
#: A ten-digit slot that is not the phone slot: the phone reading must not apply to it.
ACCOUNT10 = SlotSettings(name="account10", decl=SlotDecl(slot_type="digits", constraints={"length": 10}),
                         is_factor=False, sensitive=False)
SLOTS = {"zip": POLICY.slot("zip"), "ssn4": POLICY.slot("ssn4"), "otp": POLICY.slot("otp"),
         "account10": ACCOUNT10}
BEFORE = ["yeah", "the", "number", "is", "my", "code", "it's", "um", "okay"]
AFTER = ["thanks", "I", "think", "that's", "right"]


def _say(slot: SlotSettings, text: str):
    return resolve(slot, digits=None, speech=text, confidence=0.9, lang="en-US", mode="deterministic",
                   ctx=make_context(question_text="q", labels={}, commands={}), resolver=None)


@pytest.mark.parametrize(("slot", "text"), [
    ("ssn4", "one 2 3 4 5"),
    ("ssn4", "one two three four five"),
    ("otp", "code 1 2 3 4 5 6 7"),
    ("zip", "one nine four one oh five"),
    ("zip", "1 94105"),
    ("account10", "one 407 544 5555"),
])
def test_an_over_long_reading_starting_with_one_is_not_a_value(slot, text) -> None:
    assert _say(SLOTS[slot], text).match != "exact"


@pytest.mark.parametrize(("slot", "text", "want"), [
    ("ssn4", "the last four 1111", "1111"),
    ("ssn4", "the last 4 1111", "1111"),        # A numeral count
    ("ssn4", "1234", "1234"),
    ("zip", "my zip is 94105", "94105"),
    ("otp", "code 1 2 3 4 5 6", "123456"),
    ("zip", "Oh, 94105.", "94105"),             # A leading interjection "oh"
    ("ssn4", "oh 1234", "1234"),
    ("otp", "Oh, 123456", "123456"),
])
def test_ordinary_readings_still_parse(slot, text, want) -> None:
    r = _say(SLOTS[slot], text)
    assert (r.match, r.value) == ("exact", want)


@st.composite
def rendering(draw, digits: str) -> str:
    """``digits`` in groups, each one digit token or digit words (zero as "zero": a trailing "oh"
    is an interjection, a separate rule)."""
    cuts = sorted(draw(st.sets(st.integers(1, max(1, len(digits) - 1)), max_size=4)))
    groups = [digits[a:b] for a, b in zip([0, *cuts], [*cuts, len(digits)], strict=True) if digits[a:b]]
    out: list[str] = []
    for g in groups:
        out.extend([g] if draw(st.booleans()) else [EN_WORDS[int(d)] for d in g])
    return " ".join(out)


@settings(max_examples=400, deadline=None)
@given(data=st.data(), name=st.sampled_from(sorted(SLOTS)), lead=st.sampled_from(["one", "1"]),
       extra=st.integers(0, 3))
def test_property_no_leading_one_drop_off_the_phone_slot(data, name, lead, extra) -> None:
    """A run of 1 + n digits (n at least the slot's length) after carrier words, followed by at
    most two words: never an exact value on a non-phone slot."""
    slot = SLOTS[name]
    length = slot.decl.constraints["length"]
    rest = data.draw(st.text("0123456789", min_size=length + extra, max_size=length + extra))
    before = data.draw(st.lists(st.sampled_from(BEFORE), max_size=3))
    after = data.draw(st.lists(st.sampled_from(AFTER), max_size=2))
    text = " ".join([*before, lead, data.draw(rendering(rest)), *after])
    r = _say(slot, text)
    assert r.match != "exact", (text, r.value)
