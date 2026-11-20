"""Unit tests for owlbank.basic_auth."""

from __future__ import annotations

import base64

import pytest
from hypothesis import given
from hypothesis import strategies as st

from owlbank.basic_auth import BasicResult, Match, check, match


def header(user: str, password: str, scheme: str = "Basic") -> str:
    return f"{scheme} " + base64.b64encode(f"{user}:{password}".encode()).decode()


def test_correct_credentials_pass() -> None:
    assert check(header("twilio", "s3cret"), "twilio", "s3cret")


def test_scheme_is_case_insensitive_and_whitespace_tolerant() -> None:
    assert check(header("twilio", "s3cret", "basic"), "twilio", "s3cret")
    assert check("  " + header("twilio", "s3cret", "BASIC") + " ", "twilio", "s3cret")


def test_password_may_contain_colons_and_unicode() -> None:
    assert check(header("u", "a:b:ç"), "u", "a:b:ç")


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "Basic",
        "Basic ",
        "Bearer abc",
        header("twilio", "s3cret", "Bearer"),
        "Basic !!!notbase64!!!",
        "Basic " + base64.b64encode(b"nocolon").decode(),
        "Basic " + base64.b64encode(b"\xff\xfe:\xff").decode(),
        "Basic dHdpbGlvOnMzY3JldA",  # missing padding
        header("twilio", "wrong"),
        header("wrong", "s3cret"),
        header("twilio", "s3cret "),
        header("Twilio", "s3cret"),
        header("", ""),
    ],
)
def test_bad_headers_fail(value) -> None:
    assert not check(value, "twilio", "s3cret")


def test_non_string_header_fails() -> None:
    assert not check(b"Basic x", "twilio", "s3cret")  # type: ignore[arg-type]


@pytest.mark.parametrize("user,password", [("", "p"), ("u", ""), ("", "")])
def test_empty_expected_credentials_fail_closed(user: str, password: str) -> None:
    assert not check(header(user, password), user, password)


def test_both_fields_compared_in_constant_time(monkeypatch: pytest.MonkeyPatch) -> None:
    import owlbank.basic_auth as mod

    calls: list[tuple[bytes, bytes]] = []
    real = mod.hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(mod.hmac, "compare_digest", spy)
    assert not check(header("wrong", "s3cret"), "twilio", "s3cret")
    assert len(calls) == 2  # password compared even though the user was wrong


# --- match(): current / previous / none (rotation overlap) ---

CUR = "c" * 32
PREV = "d" * 32
UNTIL = 1_000_000


def _m(pw: str, now: float, *, previous: str | None = PREV, until: float | None = UNTIL,
       user: str = "twilio") -> BasicResult:
    return match(header(user, pw), "twilio", CUR, previous, until, now=now)


def test_match_current() -> None:
    assert _m(CUR, 0) == BasicResult(Match.CURRENT, CUR)


def test_match_previous_only_before_expiry() -> None:
    assert _m(PREV, UNTIL - 1) == BasicResult(Match.PREVIOUS, PREV)
    assert _m(PREV, UNTIL) == BasicResult(Match.NONE, None)
    assert _m(PREV, UNTIL + 1).match is Match.NONE


def test_match_neither_is_none() -> None:
    assert _m("e" * 32, 0) == BasicResult(Match.NONE, None)


def test_match_wrong_user_is_none_for_both() -> None:
    assert _m(CUR, 0, user="other").match is Match.NONE
    assert _m(PREV, 0, user="other").match is Match.NONE


def test_match_no_previous_or_no_expiry_never_previous() -> None:
    assert _m(PREV, 0, previous=None).match is Match.NONE
    assert _m(PREV, 0, until=None).match is Match.NONE
    assert match(header("twilio", ""), "twilio", CUR, "", UNTIL, now=0).match is Match.NONE


def test_match_malformed_headers_and_empty_expected_never_raise() -> None:
    for bad in (None, "", "Bearer x", "Basic", "Basic !!!", "Basic " + base64.b64encode(b"nocolon").decode()):
        assert match(bad, "twilio", CUR, PREV, UNTIL, now=0).match is Match.NONE
    assert match(header("twilio", CUR), "", CUR, now=0).match is Match.NONE
    assert match(header("twilio", ""), "twilio", "", now=0).match is Match.NONE


def test_check_still_a_plain_wrapper() -> None:
    assert check(header("twilio", CUR), "twilio", CUR)
    assert not check(header("twilio", PREV), "twilio", CUR)


def test_match_compares_user_current_and_previous_every_time(monkeypatch: pytest.MonkeyPatch) -> None:
    import hmac as _hmac

    calls: list[tuple[bytes, bytes]] = []
    real = _hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr("owlbank.basic_auth.hmac.compare_digest", spy)
    for previous in (PREV, None):
        calls.clear()
        _m(CUR, 0, previous=previous)  # even on an early CURRENT match: three compares
        assert len(calls) == 3
        calls.clear()
        _m("zzz", 0, previous=previous, user="nope")
        assert len(calls) == 3


@given(
    current=st.text(min_size=1, max_size=12),
    previous=st.one_of(st.none(), st.text(min_size=0, max_size=12)),
    until=st.one_of(st.none(), st.integers(0, 2_000)),
    now=st.integers(0, 2_000),
    sent=st.one_of(st.none(), st.text(max_size=12)),
)
def test_p7a_acceptance_rule(current: str, previous: str | None, until: int | None, now: int,
                             sent: str | None) -> None:
    """Accepted iff current, or previous and unexpired; at most two."""
    candidates = {current, previous, "never-a-password", "", *(x for x in [sent] if x is not None)}
    candidates.discard(None)
    accepted: set[str] = set()
    for pw in candidates:
        assert pw is not None
        res = match(header("twilio", pw), "twilio", current, previous, until, now=float(now))
        prev_live = bool(previous) and until is not None and now < until
        expect_current = pw == current
        expect_previous = (not expect_current) and prev_live and pw == previous
        if expect_current:
            assert res == BasicResult(Match.CURRENT, current)
        elif expect_previous:
            assert res == BasicResult(Match.PREVIOUS, previous)
        else:
            assert res == BasicResult(Match.NONE, None)  # neither: the same result as today's 401
        if res.match is not Match.NONE:
            assert res.password == pw
            accepted.add(pw)
    assert len(accepted) <= 2
