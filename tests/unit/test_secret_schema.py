"""``owlbank.secret_schema`` unit tests."""

from __future__ import annotations

import pytest

from owlbank.secret_schema import SCHEMAS, Problem, previous_pair, validate

# Fictional, well-formed values.
GOOD: dict[str, dict[str, str]] = {
    "twilio_api_key": {"account_sid": "AC" + "0123456789abcdef" * 2, "api_key": "SK" + "fedcba9876543210" * 2,
                       "api_secret": "AbCdEfGhIjKlMnOpQrStUvWxYz012345"},
    "twilio_auth_token": {"auth_token": "0123456789abcdef" * 2},
    "basic_auth": {"user": "owlbank", "password": "a1b2c3d4" * 6},
}
# CDK's placeholder (data_stack.py: template {"placeholder": true} plus a generated "unset" key).
PLACEHOLDER = {"placeholder": True, "unset": "Rk2mQ9xZpL7vT4nW8yB3cF6hJ1dS5gA0"}


def test_schemas_cover_the_three_deployed_secrets() -> None:
    assert set(SCHEMAS) == {"twilio_api_key", "twilio_auth_token", "basic_auth"}
    assert set(SCHEMAS) == set(GOOD)
    for logical, fields in SCHEMAS.items():
        assert set(fields) == set(GOOD[logical])


@pytest.mark.parametrize("logical", sorted(GOOD))
def test_a_valid_value_has_no_problem(logical: str) -> None:
    assert validate(logical, GOOD[logical]) == []


@pytest.mark.parametrize("logical", sorted(GOOD))
def test_extra_keys_are_allowed(logical: str) -> None:
    assert validate(logical, {**GOOD[logical], "note": "rotated 2026-11"}) == []


@pytest.mark.parametrize("logical", sorted(GOOD))
def test_the_placeholder_is_one_named_problem(logical: str) -> None:
    assert validate(logical, PLACEHOLDER) == [Problem(logical, "placeholder", "placeholder")]


@pytest.mark.parametrize("logical", sorted(GOOD))
def test_a_placeholder_flag_on_a_full_value_is_still_the_placeholder(logical: str) -> None:
    assert validate(logical, {**GOOD[logical], "placeholder": True}) == [
        Problem(logical, "placeholder", "placeholder")]


@pytest.mark.parametrize("value", [None, "x", 42, ["auth_token"], True])
def test_not_an_object_is_one_problem_without_a_field(value: object) -> None:
    assert validate("twilio_auth_token", value) == [Problem("twilio_auth_token", None, "not_object")]


@pytest.mark.parametrize(("logical", "field"), [(lg, f) for lg in sorted(GOOD) for f in GOOD[lg]])
def test_a_missing_field_is_named(logical: str, field: str) -> None:
    value = {k: v for k, v in GOOD[logical].items() if k != field}
    assert validate(logical, value) == [Problem(logical, field, "missing")]


@pytest.mark.parametrize(("logical", "field", "bad"), [
    ("twilio_api_key", "account_sid", "SK" + "0" * 32),  # wrong prefix
    ("twilio_api_key", "account_sid", "AC" + "0" * 31),  # short
    ("twilio_api_key", "account_sid", "AC" + "A" * 32),  # upper-case hex
    ("twilio_api_key", "api_key", "AC" + "0" * 32),
    ("twilio_api_key", "api_key", "SK" + "0" * 33),
    ("twilio_api_key", "api_secret", "a" * 31),
    ("twilio_api_key", "api_secret", "a" * 31 + "-"),
    ("twilio_auth_token", "auth_token", "0" * 32 + "\n"),  # a trailing newline is not accepted
    ("twilio_auth_token", "auth_token", "g" * 32),
    ("twilio_auth_token", "auth_token", ""),
    ("basic_auth", "user", ""),
    ("basic_auth", "user", "a" * 65),
    ("basic_auth", "user", "own bank"),
    ("basic_auth", "user", "owl:bank"),
    ("basic_auth", "password", "a" * 31),
    ("basic_auth", "password", "A" * 40),
    ("basic_auth", "password", "a" * 40 + "%"),
    ("basic_auth", "password", 12345678901234567890123456789012),
    ("basic_auth", "password", None),
])
def test_a_bad_format_is_named(logical: str, field: str, bad: object) -> None:
    assert validate(logical, {**GOOD[logical], field: bad}) == [Problem(logical, field, "format")]


def test_every_bad_field_is_reported_in_schema_order() -> None:
    value = {"api_secret": "!"}
    assert validate("twilio_api_key", value) == [
        Problem("twilio_api_key", "account_sid", "missing"),
        Problem("twilio_api_key", "api_key", "missing"),
        Problem("twilio_api_key", "api_secret", "format"),
    ]


def test_problems_never_carry_the_value() -> None:
    secret = "a1b2c3d4" * 3  # too short for a password: a format problem
    problems = validate("basic_auth", {"user": "owlbank", "password": secret})
    assert problems and all(secret not in repr(p) and secret not in p.describe() for p in problems)


def test_describe_names_secret_field_and_kind() -> None:
    assert Problem("basic_auth", "password", "format").describe() == "basic_auth: field password: format"
    assert Problem("basic_auth", None, "not_object").describe() == "basic_auth: value: not_object"


def test_an_unknown_logical_name_is_a_programming_error() -> None:
    with pytest.raises(KeyError):
        validate("twilio_api_kye", GOOD["twilio_api_key"])


# --- optional previous-password pair (basic_auth) ---

PW = "a1b2c3d4" * 6
PAIR = {"previous_password": "0f" * 16, "previous_until": 1_900_000_000}


def _basic(**extra: object) -> dict[str, object]:
    return {**GOOD["basic_auth"], **extra}


def test_pair_absent_is_valid() -> None:
    assert validate("basic_auth", _basic()) == []
    assert previous_pair("basic_auth", _basic()) is None


def test_pair_present_is_valid_and_readable() -> None:
    assert validate("basic_auth", _basic(**PAIR)) == []
    assert previous_pair("basic_auth", _basic(**PAIR)) == ("0f" * 16, 1_900_000_000)


@pytest.mark.parametrize("only", ["previous_password", "previous_until"])
def test_one_without_the_other_is_a_pair_problem_and_ignored(only: str) -> None:
    value = _basic(**{only: PAIR[only]})
    other = "previous_until" if only == "previous_password" else "previous_password"
    assert validate("basic_auth", value) == [Problem("basic_auth", other, "pair")]
    assert previous_pair("basic_auth", value) is None


@pytest.mark.parametrize("bad_pw", ["short", "", "ZZ" * 16, 5, None])
def test_bad_previous_password_is_format_and_ignored(bad_pw: object) -> None:
    value = _basic(previous_password=bad_pw, previous_until=1_900_000_000)
    assert validate("basic_auth", value) == [Problem("basic_auth", "previous_password", "format")]
    assert previous_pair("basic_auth", value) is None


@pytest.mark.parametrize("bad_until", ["1900000000", 1.5, True, 0, -5, None])
def test_bad_previous_until_is_format_and_ignored(bad_until: object) -> None:
    value = _basic(previous_password="0f" * 16, previous_until=bad_until)
    assert validate("basic_auth", value) == [Problem("basic_auth", "previous_until", "format")]
    assert previous_pair("basic_auth", value) is None


def test_other_secrets_have_no_pair() -> None:
    assert validate("twilio_auth_token", {**GOOD["twilio_auth_token"], "previous_until": 5}) == []
    with pytest.raises(KeyError):
        previous_pair("twilio_auth_token", {})


def test_pair_problem_describe_names_field_and_kind() -> None:
    assert Problem("basic_auth", "previous_until", "pair").describe() == \
        "basic_auth: field previous_until: pair"


@pytest.mark.parametrize("bad_until", [1_900_000_000_000, 4_102_444_801, 10**18])
def test_implausible_previous_until_is_format_and_ignored(bad_until: int) -> None:
    """A millisecond epoch or a far-future value would keep the old password valid for ever."""
    value = _basic(previous_password="0f" * 16, previous_until=bad_until)
    assert validate("basic_auth", value) == [Problem("basic_auth", "previous_until", "format")]
    assert previous_pair("basic_auth", value) is None


def test_previous_until_at_the_bound_is_valid() -> None:
    value = _basic(previous_password="0f" * 16, previous_until=4_102_444_800)
    assert validate("basic_auth", value) == []
    assert previous_pair("basic_auth", value) == ("0f" * 16, 4_102_444_800)
