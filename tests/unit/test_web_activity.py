"""The profile-trait filter."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from owlbank.web_activity import (
    DEFAULT_MAX_AGE_HOURS,
    FRESHNESS_TRAIT,
    GROUP,
    MAX_NUMBER_ABS,
    MAX_TRAITS_PER_GROUP,
    MAX_VALUE_CHARS,
    TRAITS,
    bound_profile_traits,
)

NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
ALLOW = ("Contact", "WebActivity")


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _web(at: str | None = _iso(NOW - timedelta(hours=1)), **extra: Any) -> dict[str, Any]:
    group: dict[str, Any] = {"lastViewedProduct": "HELOC", "viewCount": 3, **extra}
    if at is not None:
        group[FRESHNESS_TRAIT] = at
    return group


def _bound(traits: Any, *, allow: tuple[str, ...] = ALLOW, max_age_hours: int = DEFAULT_MAX_AGE_HOURS) -> dict[str, Any]:
    return bound_profile_traits(traits, allow=allow, now=NOW, max_age_hours=max_age_hours)


def test_constants_match_the_design() -> None:
    assert GROUP == "WebActivity"
    assert dict(TRAITS) == {"lastViewedProduct": "STRING", "lastViewedAt": "STRING", "viewCount": "NUMBER"}
    assert FRESHNESS_TRAIT == "lastViewedAt"
    assert (MAX_TRAITS_PER_GROUP, MAX_VALUE_CHARS, DEFAULT_MAX_AGE_HOURS) == (20, 120, 24)


def test_allow_listed_group_kept_and_another_dropped() -> None:
    out = _bound({"Contact": {"email": "a@example.com"}, "Secret": {"ssn": "123"}, GROUP: _web()})
    assert set(out) == {"Contact", GROUP}
    assert out["Contact"] == {"email": "a@example.com"}
    assert out[GROUP]["lastViewedProduct"] == "HELOC"
    assert _bound({GROUP: _web()}, allow=("Contact",)) == {}


def test_value_cut_to_max_chars() -> None:
    out = _bound({"Contact": {"note": "x" * 500}})
    assert out["Contact"]["note"] == "x" * MAX_VALUE_CHARS


def test_control_characters_removed() -> None:
    out = _bound({"Contact": {"note": "HE\x00LOC\n\t\x1b[31m‮​\x7f!"}})
    assert out["Contact"]["note"] == "HELOC[31m!"


def test_value_empty_after_cleaning_dropped() -> None:
    assert _bound({"Contact": {"note": "\x00\n ​", "email": "a@b"}}) == {"Contact": {"email": "a@b"}}


def test_none_arrays_objects_and_odd_numbers_dropped() -> None:
    out = _bound({"Contact": {"a": None, "b": [1, 2], "c": {"d": 1}, "e": float("nan"), "f": float("inf"),
                              "g": 7, "h": 2.5, "i": True, "j": b"bytes"}})
    assert out == {"Contact": {"g": 7, "h": 2.5, "i": True}}


def test_bad_trait_name_dropped() -> None:
    out = _bound({"Contact": {"1bad": "x", "has space": "x", "ok_Name1": "x", "x" * 65: "x", 5: "x"}})
    assert out == {"Contact": {"ok_Name1": "x"}}


def test_at_most_max_traits_per_group() -> None:
    out = _bound({"Contact": {f"t{i}": i for i in range(50)}})
    assert list(out["Contact"]) == [f"t{i}" for i in range(MAX_TRAITS_PER_GROUP)]


def test_last_viewed_at_missing_drops_group() -> None:
    assert _bound({GROUP: _web(at=None)}) == {}


def test_last_viewed_at_unparseable_drops_group() -> None:
    assert _bound({GROUP: _web(at="yesterday")}) == {}
    assert _bound({GROUP: {**_web(), FRESHNESS_TRAIT: 1730000000}}) == {}


def test_last_viewed_at_offset_forms_accepted() -> None:
    at = (NOW - timedelta(hours=2)).isoformat()  # +00:00
    assert GROUP in _bound({GROUP: _web(at=at)})
    assert GROUP in _bound({GROUP: _web(at="2026-10-28T07:00:00-04:00")})  # 1 h ago


def test_age_window_default_24_hours() -> None:
    assert GROUP in _bound({GROUP: _web(at=_iso(NOW - timedelta(hours=23)))})
    assert GROUP not in _bound({GROUP: _web(at=_iso(NOW - timedelta(hours=25)))})
    assert GROUP in _bound({GROUP: _web(at=_iso(NOW - timedelta(hours=25)))}, max_age_hours=26)


def test_future_skew() -> None:
    assert GROUP in _bound({GROUP: _web(at=_iso(NOW + timedelta(minutes=4)))})
    assert GROUP not in _bound({GROUP: _web(at=_iso(NOW + timedelta(minutes=6)))})


def test_only_web_activity_is_age_checked() -> None:
    old = _iso(NOW - timedelta(days=30))
    assert _bound({"Contact": {"lastViewedAt": old}}) == {"Contact": {"lastViewedAt": old}}


def test_empty_group_omitted() -> None:
    assert _bound({"Contact": {"a": None, "b": []}}) == {}
    assert _bound({"Contact": "not a mapping"}) == {}


def test_input_unchanged() -> None:
    traits = {"Contact": {"note": "x" * 500, "a": None}, "Other": {"k": "v"}, GROUP: _web(at="bad")}
    before = copy.deepcopy(traits)
    out = _bound(traits)
    assert traits == before
    assert out is not traits


def test_non_mapping_input_gives_empty() -> None:
    for bad in (None, [], "x", 3, object()):
        assert _bound(bad) == {}


def test_naive_now_fails_closed_for_web_activity() -> None:
    out = bound_profile_traits({"Contact": {"a": "b"}, GROUP: _web()}, allow=ALLOW,
                               now=datetime(2026, 10, 28, 12, 0), max_age_hours=24)
    assert out == {"Contact": {"a": "b"}}


def test_numbers_bounded_by_magnitude() -> None:
    out = _bound({"Contact": {"a": 10**15, "b": -(10**15), "c": 10**15 + 1, "d": int("9" * 4000),
                              "e": -int("9" * 4000), "f": 1e15, "g": -1e15, "h": 1.5e15, "i": 1e308}})
    assert out == {"Contact": {"a": 10**15, "b": -(10**15), "f": 1e15, "g": -1e15}}
    assert MAX_NUMBER_ABS == 10**15


def test_huge_view_count_dropped_but_group_kept() -> None:
    out = _bound({GROUP: _web(viewCount=int("9" * 4000))})
    assert out[GROUP] == {"lastViewedProduct": "HELOC", FRESHNESS_TRAIT: _iso(NOW - timedelta(hours=1))}


def test_last_viewed_at_kept_when_it_comes_after_the_cap() -> None:
    group: dict[str, Any] = {f"t{i}": i for i in range(24)}
    group[FRESHNESS_TRAIT] = _iso(NOW - timedelta(hours=1))  # 25th and last
    out = _bound({GROUP: group})
    assert GROUP in out
    assert len(out[GROUP]) == MAX_TRAITS_PER_GROUP
    assert out[GROUP][FRESHNESS_TRAIT] == group[FRESHNESS_TRAIT]
    assert list(out[GROUP]) == [FRESHNESS_TRAIT, *[f"t{i}" for i in range(MAX_TRAITS_PER_GROUP - 1)]]


def test_web_activity_trait_names_taken_first() -> None:
    group: dict[str, Any] = {f"t{i}": i for i in range(30)}
    group.update(_web())
    out = _bound({GROUP: group})
    assert list(out[GROUP])[:3] == ["lastViewedProduct", FRESHNESS_TRAIT, "viewCount"]


def test_date_only_last_viewed_at_drops_group() -> None:
    assert _bound({GROUP: _web(at="2026-10-28")}) == {}


def test_week_date_last_viewed_at_drops_group() -> None:
    assert _bound({GROUP: _web(at="2026-W44-3")}) == {}
    assert _bound({GROUP: _web(at="2026-W44-3T11:00:00Z")}) == {}  # fromisoformat takes it; we don't


def test_space_separated_time_accepted() -> None:
    assert GROUP in _bound({GROUP: _web(at="2026-10-28 11:00:00+00:00")})


def test_allow_as_str_raises_type_error() -> None:
    with pytest.raises(TypeError, match="allow"):
        bound_profile_traits({"Contact": {"a": "b"}}, allow="Contact", now=NOW, max_age_hours=24)
