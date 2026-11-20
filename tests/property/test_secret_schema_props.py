"""Portable secrets, property (schema is exact).

``validate`` returns no problem for any value built from the format patterns, and exactly one
problem, on the mutated field, for any single mutation: field removed, prefix changed, length
changed (out of range), a non-allowed character, or the placeholder flag added.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.secret_schema import SCHEMAS, Problem, previous_pair, validate


@dataclass(frozen=True)
class Spec:
    """A field's format as the design states it: prefix + alphabet{lo,hi}."""

    prefix: str
    alphabet: str
    lo: int
    hi: int | None  # None: unbounded


HEX = "0123456789abcdef"
ALNUM = string.ascii_letters + string.digits
# The secret formats, written out independently of the module, so the two must agree.
SPECS: dict[str, dict[str, Spec]] = {
    "twilio_api_key": {"account_sid": Spec("AC", HEX, 32, 32), "api_key": Spec("SK", HEX, 32, 32),
                       "api_secret": Spec("", ALNUM, 32, 32)},
    "twilio_auth_token": {"auth_token": Spec("", HEX, 32, 32)},
    "basic_auth": {"user": Spec("", ALNUM + "._-", 1, 64), "password": Spec("", HEX, 32, None)},
}
PATTERNS = {"account_sid": r"AC[0-9a-f]{32}", "api_key": r"SK[0-9a-f]{32}", "api_secret": r"[A-Za-z0-9]{32}",
            "auth_token": r"[0-9a-f]{32}", "user": r"[A-Za-z0-9._-]{1,64}", "password": r"[0-9a-f]{32,}"}


def test_specs_match_the_module_and_the_design() -> None:
    assert {lg: set(f) for lg, f in SPECS.items()} == {lg: set(f) for lg, f in SCHEMAS.items()}
    for fields in SCHEMAS.values():
        for field, pattern in fields.items():
            assert pattern.pattern == PATTERNS[field]


def body(spec: Spec, lo: int, hi: int) -> st.SearchStrategy[str]:
    return st.text(alphabet=spec.alphabet, min_size=lo, max_size=hi)


def field_value(spec: Spec) -> st.SearchStrategy[str]:
    return body(spec, spec.lo, spec.hi if spec.hi is not None else spec.lo + 40).map(lambda b: spec.prefix + b)


@st.composite
def secret(draw: st.DrawFn) -> tuple[str, dict[str, str]]:
    logical = draw(st.sampled_from(sorted(SPECS)))
    return logical, {f: draw(field_value(s)) for f, s in SPECS[logical].items()}


@settings(max_examples=500)
@given(secret())
def test_values_built_from_the_patterns_validate(case: tuple[str, dict[str, str]]) -> None:
    logical, value = case
    assert validate(logical, value) == []


@settings(max_examples=300)
@given(secret(), st.dictionaries(st.text(min_size=1).filter(lambda k: k not in {"placeholder"}),
                                 st.text(), max_size=3))
def test_extra_keys_never_matter(case: tuple[str, dict[str, str]], extra: dict[str, str]) -> None:
    logical, value = case
    assert validate(logical, {**extra, **value}) == []


def _mutate(draw: st.DrawFn, spec: Spec, current: str) -> str:
    """One format-breaking mutation of ``current`` (a valid value of ``spec``)."""
    kinds = ["length", "char"] + (["prefix"] if spec.prefix else [])
    kind = draw(st.sampled_from(kinds))
    b = current[len(spec.prefix):]
    if kind == "prefix":
        letters = st.text(alphabet=string.ascii_letters + string.digits, min_size=2, max_size=2)
        new = draw(letters.filter(lambda p: p != spec.prefix))
        return new + b
    if kind == "length":
        options = [st.integers(0, spec.lo - 1)] if spec.lo > 0 else []
        if spec.hi is not None:
            options.append(st.integers(spec.hi + 1, spec.hi + 40))
        n = draw(st.one_of(options))
        return spec.prefix + draw(body(spec, n, n))
    # a non-allowed character, replacing or inserted at any position (prefix included)
    bad = draw(st.characters(codec="utf-8").filter(lambda c: c not in spec.alphabet))
    pos = draw(st.integers(0, len(current)))
    # A prefix letter is outside the body alphabet, so replacing it with itself would change nothing
    if draw(st.booleans()) and pos < len(current) and current[pos] != bad:
        return current[:pos] + bad + current[pos + 1:]
    return current[:pos] + bad + current[pos:]


@st.composite
def mutated(draw: st.DrawFn) -> tuple[str, dict[str, object], Problem]:
    logical, value = draw(secret())
    mutation = draw(st.sampled_from(["remove", "format", "placeholder"]))
    if mutation == "placeholder":
        flag = draw(st.one_of(st.just(True), st.booleans(), st.none(), st.text()))
        return logical, {**value, "placeholder": flag}, Problem(logical, "placeholder", "placeholder")
    field = draw(st.sampled_from(sorted(value)))
    if mutation == "remove":
        out: dict[str, object] = {k: v for k, v in value.items() if k != field}
        return logical, out, Problem(logical, field, "missing")
    return logical, {**value, field: _mutate(draw, SPECS[logical][field], value[field])}, \
        Problem(logical, field, "format")


@settings(max_examples=2000)
@given(mutated())
def test_any_single_mutation_is_caught_on_the_right_field(case: tuple[str, dict[str, object], Problem]) -> None:
    logical, value, want = case
    assert validate(logical, value) == [want]


@settings(max_examples=300)
@given(secret(), st.sampled_from([None, 0, 1.5, True, [], {}]))
def test_a_non_string_field_is_a_format_problem(case: tuple[str, dict[str, str]], bad: object) -> None:
    logical, value = case
    field = sorted(value)[0]
    assert validate(logical, {**value, field: bad}) == [Problem(logical, field, "format")]


# --- optional pair: both or neither ---

_PREV_PW = st.text(alphabet=HEX, min_size=32, max_size=64)
_UNTIL = st.integers(min_value=1, max_value=4_000_000_000)


@given(pw=_PREV_PW, until=_UNTIL)
def test_a_well_formed_pair_validates_and_is_read_back(pw: str, until: int) -> None:
    value = {"user": "owlbank", "password": "a" * 32, "previous_password": pw, "previous_until": until}
    assert validate("basic_auth", value) == []
    assert previous_pair("basic_auth", value) == (pw, until)


@given(pw=_PREV_PW, until=_UNTIL, keep=st.sampled_from(["previous_password", "previous_until"]))
def test_one_member_alone_is_always_a_pair_problem_and_never_read(pw: str, until: int, keep: str) -> None:
    value: dict[str, object] = {"user": "owlbank", "password": "a" * 32}
    value[keep] = pw if keep == "previous_password" else until
    problems = validate("basic_auth", value)
    assert [p.kind for p in problems] == ["pair"]
    assert previous_pair("basic_auth", value) is None


@given(pw=st.one_of(st.none(), st.integers(), st.text(max_size=31)),
       until=st.one_of(st.none(), st.text(max_size=5), st.floats(), st.integers(max_value=0)))
def test_a_malformed_pair_is_reported_and_ignored(pw: object, until: object) -> None:
    value = {"user": "owlbank", "password": "a" * 32, "previous_password": pw, "previous_until": until}
    assert validate("basic_auth", value)
    assert previous_pair("basic_auth", value) is None
