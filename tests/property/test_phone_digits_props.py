"""keyed-phone-login: the deterministic parser on the ``phone`` slot (ten digits).

A spoken leading one ("one 407 ...", "one four oh seven ...") is dropped in words as in digits,
and the parser never reports an ``exact`` value that is the caller's digits with some missing
(found live: "one four oh seven five four four five five five five" read as 1407544555).

The phone slot reads a North American number: the leading one is a country code only before
ten digits whose first is 2-9; a ten-digit reading starting with 0 or 1 is never exact, nor is one
followed by another number run (a possible correction: the resolver decides). Other digit slots
never drop a leading one (``tests/property/test_digit_slots_leading_one_props.py``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from owlbank.policy import Policy
from owlbank.slots.normalize import normalize, spoken_digits
from owlbank.slots.engine import make_context, resolve
from owlbank.slots.validators import _is_number_token

POLICY = Policy.from_dict(json.loads(
    (Path(__file__).resolve().parents[2] / "data" / "policy.json").read_text()))
PHONE = POLICY.slot("phone")
EN_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
#: Words before the number (none of them a number word).
BEFORE = ["yeah", "the", "number", "is", "my", "it's", "um", "so", "okay", "phone"]
AFTER = ["thanks", "I", "think", "that's", "right"]


def _say(text: str):
    return resolve(PHONE, digits=None, speech=text, confidence=0.9, lang="en-US", mode="deterministic",
                   ctx=make_context(question_text="q", labels={}, commands={}), resolver=None)


# ------------------------------------------------------------------------------- unit cases


@pytest.mark.parametrize("text", [
    "one four oh seven five four four five five five five",   # (was none; the model then 1407544555)
    "one 407 544 5555",
    "1 407 544 5555",
    "my number is one four zero seven five four four five five five five",
    "one 4075445555",
    "oh 407 544 5555",                                            # a leading interjection 
])
def test_a_spoken_leading_one_is_dropped(text) -> None:
    assert (_say(text).match, _say(text).value) == ("exact", "4075445555")


@pytest.mark.parametrize("text", [
    "407 544 5555 5",          # eleven digits: was read as its first ten
    "four oh seven five four four five five five five five",
    "2 407 544 5555",          # a leading digit other than one is not dropped
    "my number is 407 544 5555 1",
])
def test_digits_are_never_cut_off(text) -> None:
    assert _say(text).match != "exact"


@pytest.mark.parametrize("speech", ["the last four 1111", "the last 4 1111"])
def test_the_count_word_still_falls_away(speech) -> None:
    # A numeral count ("4") as well as a word.
    ssn4 = POLICY.slot("ssn4")
    r = resolve(ssn4, digits=None, speech=speech, confidence=0.9, lang="en-US",
                mode="deterministic", ctx=make_context(question_text="q", labels={}, commands={}),
                resolver=None)
    assert (r.match, r.value) == ("exact", "1111")


@pytest.mark.parametrize("text", [
    "0407544555", "1407544555", "one four oh seven five four four five five five",   # 0/1 first
    "oh four oh seven five four four five five five",
    "212 555 1234 no 1235",                  # Another number follows: the resolver decides
    "212 555 1234 sorry 1235",
    "one 212 555 1234 no 1235",
])
def test_phone_readings_that_are_never_exact(text) -> None:
    assert _say(text).match != "exact"


def test_a_number_before_the_value_is_not_a_correction_of_it() -> None:
    # The caller's last number is the value; only a number after it defers to the resolver.
    assert (_say("407 no 212 555 1234").match, _say("407 no 212 555 1234").value) \
        == ("exact", "2125551234")


def test_eleven_digits_keyed_in_one_token_read_as_the_ten() -> None:
    # The final parse reads "12025550101" as early_phone does (was none, the model read it).
    assert (_say("12025550101").match, _say("12025550101").value) == ("exact", "2025550101")


# ------------------------------------------------------------------------------ properties

ten_digits = st.text("0123456789", min_size=10, max_size=10)
#: A US number: no area code starts with 0 or 1.
nanp = st.tuples(st.sampled_from("23456789"), st.text("0123456789", min_size=9, max_size=9)).map("".join)


@st.composite
def rendering(draw, number: str) -> str:
    """``number`` said in groups, each group as one digit token or as digit words ("oh" for
    some zeros)."""
    cuts = sorted(draw(st.sets(st.integers(1, 9), max_size=4)))
    groups = [number[a:b] for a, b in zip([0, *cuts], [*cuts, 10], strict=True)]
    out: list[str] = []
    for g in groups:
        if draw(st.booleans()):
            out.append(g)
        else:
            out.extend(draw(st.sampled_from(["oh", "zero"])) if d == "0" else EN_WORDS[int(d)] for d in g)
    return " ".join(out)


@settings(max_examples=400, deadline=None)
@given(data=st.data(), number=nanp, lead=st.sampled_from([None, "one", "1"]))
def test_property_any_rendering_with_or_without_a_leading_one_is_exact(data, number, lead) -> None:
    """(a) Any spoken-word or digit rendering of a ten-digit US number, with or without a separate
    leading one, after carrier words and followed by at most two words, parses to exactly it."""
    spoken = data.draw(rendering(number))
    before = data.draw(st.lists(st.sampled_from(BEFORE), max_size=4))
    after = data.draw(st.lists(st.sampled_from(AFTER), max_size=2))
    text = " ".join([*before, *([lead] if lead else []), spoken, *after])
    r = _say(text)
    assert (r.match, r.value) == ("exact", number), text


def _run_readings(run: list[str]) -> set[str]:
    """The readings of one run: whole, without a leading count of the digits after it (a single
    digit 2-9, word or numeral, before a digit token: "the last four 1111"), without a first
    "oh" before a digit token ("oh 407 ..."), or without a
    last "oh" after a digit token (an interjection, "23456, oh, no ..."). Never the run with a
    digit cut off its end."""
    out: set[str] = set()
    whole = spoken_digits(" ".join(run), "en-US")
    if whole:
        out.add(whole)
    count = spoken_digits(run[0], "en-US") if run else None
    if len(run) > 1 and run[1].isdigit() and count is not None and len(count) == 1 \
            and count not in "01" and (not run[0].isdigit() or len(run[0]) == 1):
        rest = spoken_digits(" ".join(run[1:]), "en-US")
        if rest and len(rest) == int(count):
            assert whole is None or whole.endswith(rest)   # only ever a leading token
            out.add(rest)
    if len(run) > 1 and run[0] == "oh" and run[1].isdigit():   # a leading interjection
        rest = spoken_digits(" ".join(run[1:]), "en-US")
        if rest:
            assert whole is None or whole == "0" + rest
            out.add(rest)
    if len(run) > 1 and run[-1] == "oh" and run[-2].isdigit():
        head = spoken_digits(" ".join(run[:-1]), "en-US")
        if head:
            assert whole is None or whole == head + "0"   # only the interjection's 0
            out.add(head)
    return out


def _runs(norm: str) -> list[list[str]]:
    toks = norm.split()
    runs: list[list[str]] = []
    i = 0
    while i < len(toks):
        if not _is_number_token(toks[i], "en-US"):
            i += 1
            continue
        j = i
        while j < len(toks) and _is_number_token(toks[j], "en-US"):
            j += 1
        runs.append(toks[i:j])
        i = j
    return runs


def _allowed_readings(norm: str) -> set[str]:
    """Every reading the phone parser may give as exact: a reading of the *last* run of number
    tokens (a run after the value may be a correction), or that reading without its leading 1
    when it is eleven digits (the country code)."""
    runs = _runs(norm)
    readings = _run_readings(runs[-1]) if runs else set()
    return readings | {r[1:] for r in readings if len(r) == 11 and r[0] == "1"}


TOKENS = st.one_of(
    st.sampled_from(EN_WORDS + ["oh", "double", "twenty", "no", "sorry", "is", "my", "number", "um",
                                "and", "what", "was", "it"]),
    st.text("0123456789", min_size=1, max_size=4),
)


@settings(max_examples=600, deadline=None)
@given(toks=st.lists(TOKENS, max_size=16))
def test_property_no_wrong_exact(toks) -> None:
    """(b) For any input, an exact value is the caller's last run of number words read whole, or
    with a leading country-code one (or a count word) dropped: never digits missing at the end,
    never a digit the caller didn't say in that run, never a first digit of 0 or 1."""
    text = " ".join(toks)
    r = _say(text)
    if r.match == "exact":
        assert r.value in _allowed_readings(normalize(text, "en-US")), text
        assert r.value[0] in "23456789", text


@settings(max_examples=300, deadline=None)
@given(number=ten_digits, extra=st.text("0123456789", min_size=1, max_size=1),
       as_words=st.booleans(), at_end=st.booleans())
def test_property_eleven_digits_are_not_cut_to_ten(number, extra, as_words, at_end) -> None:
    """An extra digit before or after ten digits: never exact, unless the eleven start with a
    one said on its own (the leading one, dropped)."""
    digits = number + extra if at_end else extra + number
    assume(digits[0] != "1")
    spoken = " ".join(EN_WORDS[int(d)] for d in digits) if as_words else " ".join(digits)
    assert _say(spoken).match != "exact"
