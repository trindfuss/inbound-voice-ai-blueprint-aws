"""Rotation as the operator script writes it and the request path reads it: over any sequence of
rotations, clears and clock advances, exactly the current password is accepted, plus the previous one
only while ``now < previous_until``; every other password matches nothing."""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from lambdas.frontdoor.handler import basic_secret_of
from owlbank.basic_auth import Match, match
from scripts import populate_secrets as ps
from tests.component.test_frontdoor_support import basic_header

USER = "twilio-hooks"
START = 1_800_000_000.0


def password(n: int) -> str:
    return f"{n:048x}"


steps = st.lists(
    st.one_of(
        st.tuples(st.just("rotate"), st.floats(min_value=0.01, max_value=48, allow_nan=False)),
        st.tuples(st.just("clear"), st.none()),
        st.tuples(st.just("advance"), st.floats(min_value=0, max_value=20 * 3600, allow_nan=False)),
    ),
    max_size=25,
)


@settings(max_examples=150, deadline=None)
@given(sequence=steps)
def test_only_the_current_and_the_unexpired_previous_password_are_accepted(
        sequence: list[tuple[str, float | None]]) -> None:
    now = START
    issued = 0
    stored: dict[str, Any] = {"user": USER, "password": password(issued)}
    # the model, kept apart from the stored value: what must be accepted
    current, previous, until = password(issued), None, None

    def check() -> None:
        secret = basic_secret_of(stored)
        for n in range(issued + 2):  # every password ever issued, and one never issued
            pw = password(n)
            got = match(basic_header(USER, pw), secret.user, secret.password, secret.previous_password,
                        secret.previous_until, now=now)
            if pw == current:
                assert got.match is Match.CURRENT
            elif pw == previous and until is not None and now < until:
                assert got.match is Match.PREVIOUS
            else:
                assert got.match is Match.NONE and got.password is None
        assert match(basic_header("other", current), secret.user, secret.password, secret.previous_password,
                     secret.previous_until, now=now).match is Match.NONE

    check()
    for kind, arg in sequence:
        if kind == "rotate":
            assert arg is not None
            issued += 1
            stored = ps.rotate_basic_value(stored, password(issued), now=now, overlap_hours=arg)
            previous, until, current = current, int(now + arg * 3600), password(issued)
        elif kind == "clear":
            stored = ps.clear_previous_value(stored)
            previous, until = None, None
        else:
            assert arg is not None
            now += arg
        check()
