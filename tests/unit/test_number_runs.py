"""``number_runs``: the number-run finder of ``placeholder_text`` (task)."""

from __future__ import annotations

import pytest

from owlbank.slots import NumberRun, number_runs

BOTH = ("en-US", "es-US")


def shape(text: str, langs=BOTH) -> list[tuple[str, int, str | None]]:
    """Each run as (its text, tokens, digits)."""
    return [(text[r.start:r.end], r.tokens, r.digits) for r in number_runs(text, langs)]


def test_filler_ends_a_run():
    assert shape("one, uh, two three four") == [("one", 1, "1"), ("two three four", 3, "234")]


def test_correction_gives_two_runs():
    assert shape("1 2, no, 1234") == [("1 2", 2, "12"), ("1234", 4, "1234")]


def test_mixed_language_run_counts_every_token():
    # "o" is an English unit, "uno" and "dos" Spanish: one run of 3 tokens over both families.
    # Neither language reads it as a whole, so its digits are unknown.
    assert shape("uno o dos") == [("uno o dos", 3, None)]


@pytest.mark.parametrize(("text", "digits"), [("o", "0"), ("una", "1")])
def test_lone_spanish_word_is_a_one_token_run(text, digits):
    assert shape(f"quiero {text} cosa") == [(text, 1, digits)]


def test_digit_token_of_k_digits_counts_k():
    assert shape("my zip is 94105") == [("94105", 5, "94105")]
    assert shape("7") == [("7", 1, "7")]


@pytest.mark.parametrize(
    ("text", "langs", "expected"),
    [
        ("one and two", ("en-US",), [("one and two", 2, None)]),
        ("treinta y uno", ("es-US",), [("treinta y uno", 2, "31")]),
        ("cinco con dos", ("es-US",), [("cinco con dos", 2, None)]),
        ("uno y dos", BOTH, [("uno y dos", 2, None)]),
    ],
)
def test_joiners_are_excluded_from_tokens(text, langs, expected):
    assert shape(text, langs) == expected


def test_run_that_does_not_read_has_no_digits():
    assert shape("the first one", ("en-US",)) == [("first one", 2, None)]


def test_joiner_only_run():
    # Punctuation outside the run separators ends a run, so a joiner promoted between two numeric
    # tokens can stand alone. It reads as no digits and counts as 1 token (fail closed: never a run
    # of 0 tokens).
    assert shape("five; and; six", ("en-US",)) == [("five", 1, "5"), ("and", 1, None), ("six", 1, "6")]


def test_languages_select_the_word_lists():
    assert number_runs("uno dos", ("en-US",)) == []
    assert shape("uno dos", ("es-US",)) == [("uno dos", 2, "12")]
    assert shape("one two", ("es-US",)) == []
    assert shape("uno dos", BOTH) == [("uno dos", 2, "12")]


def test_offsets_index_the_text_as_given():
    text = "Sí, es Ó uno"
    runs = number_runs(text, ("es-US",))
    assert runs == [NumberRun(start=9, end=12, tokens=1, digits="1")]
    assert [text[r.start:r.end] for r in number_runs(text, BOTH)] == ["Ó uno"]


def test_accents_and_case_are_ignored():
    assert shape("NUEVE cuatro", BOTH) == [("NUEVE cuatro", 2, "94")]
    assert shape("ó") == [("ó", 1, "0")]


def test_no_runs():
    assert number_runs("", BOTH) == []
    assert number_runs("Agent please", BOTH) == []
