"""A digit answer cut short is asked again without the resolver (owner test call 2026-10-15; 4.6)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.unit.test_slots import SSN4, FakeResolver, run

EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ES = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]


@pytest.mark.parametrize("speech", ["one", "One.", "one one", "my one one one", "nine four one"])
def test_short_digit_answer_skips_the_resolver(speech: str) -> None:
    fake = FakeResolver()
    r = run(SSN4, speech=speech, resolver=fake)
    assert (r.match, r.source) == ("none", "short_digits")
    assert fake.calls == []


def test_spanish_short_answer_skips_the_resolver() -> None:
    fake = FakeResolver()
    r = run(SSN4, speech="uno uno", lang="es-US", resolver=fake)
    assert (r.match, r.source) == ("none", "short_digits")
    assert fake.calls == []


@pytest.mark.parametrize("speech", ["one moment please", "I don't know", "what is my balance"])
def test_sentences_still_go_to_the_resolver(speech: str) -> None:
    fake = FakeResolver()
    run(SSN4, speech=speech, resolver=fake)
    assert fake.calls, speech


def test_a_complete_answer_is_still_a_value() -> None:
    assert run(SSN4, speech="one one one one").match == "exact"


@settings(max_examples=150, deadline=None)
@given(st.lists(st.integers(0, 9), min_size=1, max_size=8), st.sampled_from(["en-US", "es-US"]))
def test_property_only_short_number_runs_are_short(digits: list[int], lang: str) -> None:
    words = EN if lang == "en-US" else ES
    speech = " ".join(words[d] for d in digits)
    fake = FakeResolver()
    r = run(SSN4, speech=speech, lang=lang, resolver=fake)
    if len(digits) < 4:
        assert (r.match, r.source) == ("none", "short_digits") and fake.calls == []
    elif len(digits) == 4:
        assert r.match == "exact" and r.value == "".join(map(str, digits))
    else:
        assert r.source != "short_digits"
