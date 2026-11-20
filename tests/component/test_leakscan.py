"""The factor leak scan matches whole values, not slices of generated digit runs.

Regression for a flaky property test: a random 8-digit confirmation number that
happened to contain a test SSN's last four tripped the old substring check.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.component.test_frontdoor_support import assert_no_leak
from tests.leakscan import compile_values, leaks, value_pattern

VALUE = "1111"  # a synthetic factor value, not a fixture
DIGITS = st.text(alphabet="0123456789", min_size=1, max_size=8)
VALUES = st.text(alphabet="0123456789", min_size=4, max_size=6)
SEP = st.sampled_from(["", " ", "-", "  ", " - "])


@pytest.mark.parametrize("text", [
    "confirmation 09191111", '{"confirmation": "11110919"}', "CA0e1111222", "t=1760001111000",
    "line 21111", "id 111111",
])
def test_a_generated_value_that_contains_the_factor_is_not_a_leak(text: str) -> None:
    assert not leaks(compile_values([VALUE]), text)


@pytest.mark.parametrize("text", [
    "1111", "ssn4 1111 ok", '{"ssn": "1111"}', '"ssn":1111', "ssn=1111", "ssn1111",
    "digits 1 1 1 1", "11-11", "the code is 1111.", "zip 94105 ssn 1111", "x\n1111\n",
])
def test_the_value_as_its_own_value_is_a_leak(text: str) -> None:
    assert leaks(compile_values([VALUE]), text)


def test_letters_match_case_insensitively_at_word_edges() -> None:
    p = [value_pattern("s3cret")]
    assert leaks(p, "pw=S3CRET;")
    assert not leaks(p, "xs3cret")


@given(VALUES, DIGITS, DIGITS)
def test_a_value_inside_a_longer_digit_run_never_matches(v: str, pre: str, post: str) -> None:
    assert not leaks(compile_values([v]), f"no {pre}{v} here and none {v}{post} here")


@given(VALUES, st.data())
def test_a_value_on_its_own_always_matches(v: str, data: st.DataObject) -> None:
    spaced = v[0] + "".join(data.draw(SEP) + c for c in v[1:])
    shape = data.draw(st.sampled_from([
        "{}", "ssn {} end", '{{"ssn": "{}"}}', '"ssn":{}', "ssn={}", "ssn{}", "{}.",
    ]))
    pad = data.draw(DIGITS)
    text = f"{pad} " + shape.format(spaced)  # a digit run nearby, but not touching the value
    assert leaks(compile_values([v]), text)


def _harness(bodies: list[str], stored: list[str]) -> SimpleNamespace:
    return SimpleNamespace(bodies=bodies, stored_strings=lambda: stored)


def test_assert_no_leak_ignores_a_confirmation_number_containing_a_factor() -> None:
    stored = [json.dumps({"confirmation": "0919" + VALUE})]
    rec = logging.LogRecord("x", logging.INFO, "x", 1111, "confirmation %s", ("1111" + "0413",), None)
    assert_no_leak(_harness([], stored), [rec], {VALUE})  # type: ignore[arg-type]


@pytest.mark.parametrize("where", ["response", "log", "store"])
def test_assert_no_leak_still_catches_a_real_leak(where: str) -> None:
    text = json.dumps({"ssn": VALUE})
    rec = logging.LogRecord("x", logging.INFO, "x", 1, text if where == "log" else "ok", None, None)
    h = _harness([text] if where == "response" else [], [text] if where == "store" else [])
    with pytest.raises(AssertionError, match=f"leaked into {where}"):
        assert_no_leak(h, [rec], {VALUE})  # type: ignore[arg-type]


@pytest.mark.parametrize(("text", "leak"), [
    ('"duration_ms": 2035.6', False),   # a measured time that happens to start with the value
    ("x 3.2035 y", False),              # the fractional part of a decimal
    ("SSN 2035.", True),                # a sentence-ending full stop is still a boundary
    ('"ssn": "2035"', True),
])
def test_a_decimal_number_is_not_a_whole_value(text: str, leak: bool) -> None:
    assert leaks(compile_values(["2035"]), text) is leak


@pytest.mark.parametrize(("text", "leak"), [
    ('"escalated_at":"2025-10-09T08:53:20Z"', False),  # a past flake: 10-09 read as 1009
    ("at 2025-10-09t08:53:20.123+00:00 ok", False),
    ("2026-01-02T10:09Z", False),
    ("2025-10-09T08:53:20Z ssn 1009", True),           # a value next to a timestamp still leaks
    ("on 2025-10-09 ssn", True),                       # a bare date is not a server timestamp
])
def test_a_generated_timestamp_is_not_a_whole_value(text: str, leak: bool) -> None:
    assert leaks(compile_values(["1009"]), text) is leak


# ---- random tokens the server minted (a nonce can hold a short value between letters)

NONCE = "zA1111leQrStUvWxYzAbCdEfGhIjKlM"  # 32 characters of the token_urlsafe alphabet


@pytest.fixture
def minted() -> object:
    """Record ``NONCE`` as minted for one test, then forget it."""
    from tests import leakscan

    before = (set(leakscan._MINTED), list(leakscan._MINTED_LENGTHS))  # noqa: SLF001
    leakscan.record_minted(NONCE)
    yield NONCE
    leakscan._MINTED.clear()  # noqa: SLF001
    leakscan._MINTED.update(before[0])
    leakscan._MINTED_LENGTHS[:] = before[1]


@pytest.mark.parametrize("shape", [
    '<Parameter name="n" value="{}"/>', "n={}&x=1", '{{"n": "{}"}}', "nonce {} end",
])
def test_a_minted_token_holding_a_factor_is_not_a_leak(minted: str, shape: str) -> None:
    assert not leaks(compile_values([VALUE]), shape.format(minted))


def test_the_same_text_leaks_when_the_token_was_not_minted() -> None:
    assert leaks(compile_values([VALUE]), f'<Parameter name="n" value="{NONCE}"/>')


@pytest.mark.parametrize("shape", [
    '<Parameter name="n" value="{}"/> ssn 1111', "ssn 1111 n={}", '{{"ssn": "1111", "n": "{}"}}',
    "1111{}", "{}1111", "1111-{}", "{}-1111", "1 1 1 1 {}",
])
def test_a_real_leak_in_or_next_to_a_minted_token_is_still_found(minted: str, shape: str) -> None:
    assert leaks(compile_values([VALUE]), shape.format(minted))


def test_only_the_exact_minted_value_is_left_out(minted: str) -> None:
    other = minted[:-1] + ("a" if minted[-1] != "a" else "b")  # one character off: not minted
    assert leaks(compile_values([VALUE]), other)
