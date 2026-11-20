"""``safe_traceback`` never contains an exception message, an argument or a local
variable, whatever they hold (a phone number, a digit run, a token, any text), and is bounded."""

from __future__ import annotations

import base64
import string
from typing import Any

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from owlbank.obs import safe_traceback

CASES = settings(max_examples=200, deadline=None)

phones = st.from_regex(r"\+[1-9][0-9]{7,14}", fullmatch=True)
digit_runs = st.from_regex(r"[0-9]{6,24}", fullmatch=True)
tokens = st.binary(min_size=18, max_size=48).map(lambda b: base64.urlsafe_b64encode(b).decode().rstrip("="))
words = st.text(min_size=12, max_size=200)  # arbitrary text, newlines and non-ASCII included
messages = st.one_of(phones, digit_runs, tokens, words,
                     st.text(alphabet=string.ascii_letters + " ", min_size=12, max_size=80))


def _raise(message: str, extra: Any) -> None:
    local_secret = message + "-local"  # noqa: F841 (a local variable: it must never be logged)
    raise ValueError(message, extra)


def _raise_chained(message: str, extra: Any) -> None:
    try:
        _raise(message, extra)
    except ValueError as inner:
        raise RuntimeError(f"wrapped {message}", extra) from inner


def _caught(fn: Any, message: str, extra: Any) -> BaseException:
    try:
        fn(message, extra)
    except BaseException as exc:  # noqa: BLE001
        return exc
    raise AssertionError("did not raise")


BASELINE = safe_traceback(_caught(_raise_chained, "", ""))


def _clean_of(out: str, message: str, extra: Any) -> None:
    assert message not in out
    assert f"wrapped {message}" not in out
    assert message + "-local" not in out
    for piece in (str(extra), repr(extra)):
        if len(piece) >= 12:
            assert piece not in out


@CASES
@given(messages, messages)
def test_the_stack_never_contains_the_message_or_an_argument(message: str, arg: str) -> None:
    assume(message not in BASELINE and arg not in BASELINE)
    for fn in (_raise, _raise_chained):
        exc = _caught(fn, message, {"k": arg, "n": [arg]})
        out = safe_traceback(exc)
        _clean_of(out, message, {"k": arg, "n": [arg]})
        assert arg not in out
        assert out.startswith("Traceback (most recent call last):") and "ValueError" in out


@CASES
@given(messages, st.integers(min_value=1, max_value=12))
def test_a_chain_of_any_depth_is_bounded_and_names_only_classes(message: str, depth: int) -> None:
    assume(message not in BASELINE)
    exc: BaseException | None = None
    for i in range(depth):
        try:
            raise (ValueError if i % 2 else KeyError)(f"{message} #{i}") from exc
        except BaseException as caught:  # noqa: BLE001
            exc = caught
    assert exc is not None
    out = safe_traceback(exc)
    assert message not in out
    assert out.count("Traceback (most recent call last):") == min(depth, 5)
    assert len(out) <= 8192


@CASES
@given(messages, st.integers(min_value=0, max_value=900))  # below the recursion limit
def test_the_output_is_bounded_however_deep_the_stack(message: str, frames: int) -> None:
    assume(message not in BASELINE)

    def recurse(n: int) -> None:
        if n == 0:
            raise RuntimeError(message)
        recurse(n - 1)

    try:
        recurse(frames)
    except RuntimeError as exc:
        out = safe_traceback(exc)
    assert len(out) <= 8192 and message not in out
    assert out.rstrip().endswith("RuntimeError")  # the end (the raise) is what is kept


@CASES
@given(messages)
def test_message_hooks_are_never_called(message: str) -> None:
    called: list[str] = []

    class Hostile(Exception):
        def __str__(self) -> str:
            called.append("str")
            return message

        def __repr__(self) -> str:
            called.append("repr")
            return message

    try:
        raise Hostile(message)
    except Hostile as exc:
        out = safe_traceback(exc)
    assert called == [] and "Hostile" in out


def test_a_recursion_error_is_bounded_too() -> None:
    def forever() -> None:
        forever()

    try:
        forever()
    except RecursionError as exc:
        out = safe_traceback(exc)
    assert len(out) <= 8192 and out.rstrip().endswith("RecursionError")
