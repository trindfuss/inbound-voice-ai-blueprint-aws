"""Property tests for spoken verification factors.

For the real factor declarations in ``data/policy.json`` (zip 5, ssn4 4, otp 6):

* a digit string read digit by digit in en-US or es-US (as words, or as the digits an
  ASR engine may return) resolves back to the same digits, from speech;
* when the keypad gave a valid answer, it wins over any speech in the same gather.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from owlbank.policy import Policy
from owlbank.slot_policy import SlotSettings
from owlbank.slots import make_context, resolve

POLICY = json.loads((Path(__file__).resolve().parents[2] / "data" / "policy.json").read_text())
#: ``resolve`` takes the slot's settings; the seed's, for each factor.
FACTORS = {name: Policy.from_dict(POLICY).slot(name) for name in POLICY["factors"]}
TODAY = date(2026, 10, 13)

WORDS = {
    "en-US": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
    "es-US": ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"],
}

factor_names = st.sampled_from(sorted(FACTORS))
langs = st.sampled_from(sorted(WORDS))


def _resolve(slot: SlotSettings, lang: str, *, digits: str | None, speech: str | None):
    return resolve(slot, digits=digits, speech=speech, confidence=0.9, lang=lang, mode="deterministic",
                   ctx=make_context(question_text="q", labels={}, commands={}), resolver=None, today=TODAY)


@st.composite
def factor_and_digits(draw: st.DrawFn) -> tuple[str, str]:
    name = draw(factor_names)
    length = FACTORS[name].decl.constraints["length"]
    return name, draw(st.text("0123456789", min_size=length, max_size=length))


@st.composite
def spoken_form(draw: st.DrawFn, digits: str, lang: str) -> str:
    """How a caller (or the ASR transcript of one) may read ``digits`` one by one."""
    style = draw(st.sampled_from(["words", "digits", "spaced", "commas", "mixed"]))
    words = WORDS[lang]
    if style == "words":
        return " ".join(words[int(c)] for c in digits)
    if style == "digits":
        return digits
    if style == "spaced":
        return " ".join(digits)
    if style == "commas":
        return ", ".join(words[int(c)] for c in digits) + "."
    return " ".join(words[int(c)] if draw(st.booleans()) else c for c in digits)


@given(case=factor_and_digits(), lang=langs, data=st.data())
def test_digit_by_digit_speech_round_trips(case, lang, data):
    name, digits = case
    text = data.draw(spoken_form(digits, lang))
    out = _resolve(FACTORS[name], lang, digits=None, speech=text)
    assert (out.match, out.value, out.source) == ("exact", digits, "speech"), text


@given(case=factor_and_digits(), lang=langs, other=st.text("0123456789", min_size=4, max_size=6),
       data=st.data())
def test_valid_keypad_wins_over_speech(case, lang, other, data):
    name, digits = case
    speech = data.draw(st.one_of(spoken_form(other, lang), st.text(max_size=30)))
    keyed = data.draw(st.sampled_from([digits, digits + "#"]))
    out = _resolve(FACTORS[name], lang, digits=keyed, speech=speech)
    assert (out.match, out.value, out.source) == ("exact", digits, "dtmf")


def test_policy_factors_allow_two_retries():
    assert {n: s.decl.max_retries for n, s in FACTORS.items()} == {"zip": 2, "ssn4": 2, "otp": 2}
