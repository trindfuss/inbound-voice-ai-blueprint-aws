"""Segment-traits properties of ``bound_profile_traits``.

- Only allow-listed groups; at most 20 traits per group; names match the pattern; every value a
  str of at most 120 chars with no control characters, an int or finite float with abs at most
  ``MAX_NUMBER_ABS``, or a bool.
- Idempotent.
- Never raises for arbitrary JSON-like input (nested, huge strings, odd Unicode); never mutates it.
- ``WebActivity`` kept exactly when ``lastViewedAt`` parses and is at most ``max_age_hours`` old and
  not more than ``FUTURE_SKEW`` ahead; kept sets only shrink as ``max_age_hours`` shrinks; the order
  of the group's traits doesn't change whether it is kept, and a kept group keeps ``lastViewedAt``.
"""

from __future__ import annotations

import copy
import math
import unicodedata
from datetime import UTC, datetime, timedelta
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank.web_activity import (
    FRESHNESS_TRAIT,
    FUTURE_SKEW,
    GROUP,
    MAX_NUMBER_ABS,
    MAX_TRAITS_PER_GROUP,
    MAX_VALUE_CHARS,
    TRAIT_NAME_RE,
    bound_profile_traits,
)

PROPS = settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
GROUP_NAMES = ("Contact", GROUP, "Secret", "Other", "")
CONTROL_CATEGORIES = {"Cc", "Cf", "Cs", "Zl", "Zp"}

odd_text = st.text(alphabet=st.characters(exclude_categories=()), max_size=40) | st.sampled_from([
    "‮LOCEH", "​", "\x00\x1b[2J", "  ", "é", "\U0001f3e0‍", "﻿", " \t\n ",
])
huge_text = st.builds(lambda c, n: c * n, st.sampled_from(["x", "\x00", "é", "\U0001f600", " "]),
                      st.integers(100, 5000))
big_numbers = st.integers(-(10**20), 10**20) | st.sampled_from(
    [MAX_NUMBER_ABS, -MAX_NUMBER_ABS, MAX_NUMBER_ABS + 1, int("9" * 4000), 1e15, 1.0000001e15, -1e300])
scalars = (st.none() | st.booleans() | st.integers() | big_numbers
           | st.floats(allow_nan=True, allow_infinity=True)
           | odd_text | huge_text | st.binary(max_size=8))
json_like = st.recursive(scalars, lambda inner: st.lists(inner, max_size=4)
                         | st.dictionaries(st.text(max_size=8), inner, max_size=4), max_leaves=20)
trait_names = (st.from_regex(TRAIT_NAME_RE, fullmatch=True) | st.text(max_size=70)
               | st.sampled_from([FRESHNESS_TRAIT, "lastViewedProduct", "viewCount"]) | st.integers())
offsets = st.integers(-60 * 24 * 10, 60 * 24 * 2).map(lambda m: timedelta(minutes=m))
timestamps = st.one_of(
    offsets.map(lambda d: (NOW + d).strftime("%Y-%m-%dT%H:%M:%SZ")),
    offsets.map(lambda d: (NOW + d).isoformat()),
    st.sampled_from(["", "yesterday", "2026-13-01T00:00:00Z", "nope", "2026-10-28", "2026-W44-3",
                     "2026-W44-3T11:00:00Z"]),
    scalars,
)


@st.composite
def groups(draw: st.DrawFn) -> dict[Any, Any]:
    g: dict[Any, Any] = draw(st.dictionaries(trait_names, json_like, max_size=30))
    if draw(st.booleans()):
        g[FRESHNESS_TRAIT] = draw(timestamps)
    return g


profiles = st.dictionaries(st.sampled_from(GROUP_NAMES) | st.text(max_size=10), groups() | json_like, max_size=6)
inputs = profiles | json_like
allows = st.lists(st.sampled_from(GROUP_NAMES), max_size=4)
ages = st.integers(1, 24 * 400)


def _run(traits: Any, allow: list[str], age: int) -> dict[str, dict[str, Any]]:
    return bound_profile_traits(traits, allow=allow, now=NOW, max_age_hours=age)


@PROPS
@given(inputs, allows, ages)
def test_output_within_limits(traits: Any, allow: list[str], age: int) -> None:
    out = _run(traits, allow, age)
    assert type(out) is dict
    for name, group in out.items():
        assert name in allow
        assert type(group) is dict and 0 < len(group) <= MAX_TRAITS_PER_GROUP
        for key, value in group.items():
            assert isinstance(key, str) and TRAIT_NAME_RE.fullmatch(key)
            if isinstance(value, str):
                assert 0 < len(value) <= MAX_VALUE_CHARS
                assert not any(unicodedata.category(c) in CONTROL_CATEGORIES for c in value)
            else:
                assert type(value) in (int, float, bool)
                if type(value) is float:
                    assert math.isfinite(value)
                if type(value) is not bool:
                    assert abs(value) <= MAX_NUMBER_ABS


@PROPS
@given(inputs, allows, ages)
def test_bound_traits_are_idempotent(traits: Any, allow: list[str], age: int) -> None:
    once = _run(traits, allow, age)
    assert _run(once, allow, age) == once


@PROPS
@given(inputs, allows, ages)
def test_never_raises_never_mutates(traits: Any, allow: list[str], age: int) -> None:
    before = copy.deepcopy(traits)
    _run(traits, allow, age)
    assert repr(traits) == repr(before)  # repr: NaN != NaN, but its repr is stable


@PROPS
@given(offsets, ages, st.booleans())
def test_kept_exactly_inside_window(delta: timedelta, age: int, z_form: bool) -> None:
    at = NOW + delta
    stamp = at.strftime("%Y-%m-%dT%H:%M:%SZ") if z_form else at.isoformat()
    out = _run({GROUP: {FRESHNESS_TRAIT: stamp, "viewCount": 3}}, [GROUP], age)
    inside = -timedelta(hours=age) <= delta <= FUTURE_SKEW
    assert (GROUP in out) == inside


@PROPS
@given(profiles, allows, ages, ages)
def test_kept_sets_shrink_with_age(traits: Any, allow: list[str], a: int, b: int) -> None:
    small, large = sorted((a, b))
    out_small, out_large = _run(traits, allow, small), _run(traits, allow, large)
    assert set(out_small) <= set(out_large)
    for name, group in out_small.items():
        assert group.items() <= out_large[name].items()


valid_names = st.from_regex(TRAIT_NAME_RE, fullmatch=True).filter(lambda n: n != FRESHNESS_TRAIT)


@st.composite
def ordered_web_groups(draw: st.DrawFn) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """A WebActivity group with ``lastViewedAt`` and up to 30 other valid traits, and the same group
    with its traits in another order."""
    stamp = draw(timestamps)
    others = draw(st.dictionaries(valid_names, st.integers(0, 9) | st.text(min_size=1, max_size=5),
                                  max_size=30))
    items = [*others.items(), (FRESHNESS_TRAIT, stamp)]
    return dict(items), dict(draw(st.permutations(items))), stamp


@PROPS
@given(ordered_web_groups(), ages)
def test_trait_order_does_not_change_keeping(case: tuple[dict[str, Any], dict[str, Any], Any], age: int) -> None:
    group, shuffled, stamp = case
    out, out_shuffled = _run({GROUP: group}, [GROUP], age), _run({GROUP: shuffled}, [GROUP], age)
    alone = _run({GROUP: {FRESHNESS_TRAIT: stamp}}, [GROUP], age)
    assert (GROUP in out) == (GROUP in out_shuffled) == (GROUP in alone)
    for result in (out, out_shuffled):
        if GROUP in result:
            assert result[GROUP][FRESHNESS_TRAIT] == alone[GROUP][FRESHNESS_TRAIT]
