""":class:`EntrySettings`, :class:`Settings` and their defaults.

* The defaults per known model, whatever the profile prefix (``us.``, ``global.``, a bare
  ``anthropic.`` ID) and the dated version tail.
* Any other model: trip 2.3 x baseline p90, recover 1.5 x baseline, deadline max(4.0, 2.7 x baseline).
* The last entry has no first-token deadline; a list of one is today's single model.
* The global defaults; a bad value is refused with an error naming the field.
"""

from __future__ import annotations

import dataclasses

import pytest

from owlbank.failover.settings import (
    DEFAULT_BASELINE_P90_S,
    EntrySettings,
    Settings,
    SettingsError,
    entry_defaults,
    series_key,
    series_parts,
)

HAIKU_55 = "us.anthropic.claude-haiku-5-5"
HAIKU_45 = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET_55 = "us.anthropic.claude-sonnet-5-5"
DEFAULT_LIST = (HAIKU_55, HAIKU_45, SONNET_55)


# --------------------------------------------------------------------------- per-entry defaults


@pytest.mark.parametrize("model_id", [HAIKU_55, "global.anthropic.claude-haiku-5-5", "anthropic.claude-haiku-5-5",
                                      "us.anthropic.claude-haiku-5-5-20260901-v1:0"])
def test_haiku_55_defaults(model_id: str) -> None:
    e = entry_defaults(model_id)
    assert (e.model_id, e.trip_ttft_s, e.recover_ttft_s, e.deadline_s, e.error_rate_limit) == (
        model_id, 3.5, 2.5, 4.0, 0.2)


@pytest.mark.parametrize("model_id", [HAIKU_45, "global.anthropic.claude-haiku-4-5-20251001-v1:0",
                                      "us.anthropic.claude-haiku-4-5"])
def test_haiku_45_defaults(model_id: str) -> None:
    e = entry_defaults(model_id)
    assert (e.trip_ttft_s, e.recover_ttft_s, e.deadline_s, e.error_rate_limit) == (1.6, 1.1, 3.0, 0.2)


def test_sonnet_55_defaults() -> None:
    e = entry_defaults(SONNET_55)
    assert (e.trip_ttft_s, e.recover_ttft_s, e.error_rate_limit) == (3.5, 2.5, 0.2)
    # gives Sonnet 5.5 no deadline because it is last; placed earlier it needs one
    # The same as the other adaptive model's.
    assert e.deadline_s == 4.0
    assert entry_defaults(SONNET_55, last=True).deadline_s is None


@pytest.mark.parametrize("model_id", ["us.anthropic.claude-sonnet-5", "us.anthropic.claude-sonnet-4-6",
                                      "us.anthropic.claude-3-haiku-20240307-v1:0", "us.amazon.nova-pro-v1:0",
                                      "us.anthropic.claude-haiku-5-55", "not a model id"])
def test_any_other_model_gets_baseline_defaults(model_id: str) -> None:
    e = entry_defaults(model_id)
    assert DEFAULT_BASELINE_P90_S == 1.5
    assert e.trip_ttft_s == pytest.approx(2.3 * 1.5)
    assert e.recover_ttft_s == pytest.approx(1.5 * 1.5)
    assert e.deadline_s == pytest.approx(4.05)
    assert e.error_rate_limit == 0.2


@pytest.mark.parametrize(("baseline", "deadline"), [(1.0, 4.0), (1.48, 4.0), (2.0, 5.4)])
def test_baseline_scales_the_defaults(baseline: float, deadline: float) -> None:
    e = entry_defaults("us.anthropic.claude-sonnet-5", baseline_p90_s=baseline)
    assert e.trip_ttft_s == pytest.approx(2.3 * baseline)
    assert e.recover_ttft_s == pytest.approx(1.5 * baseline)
    assert e.deadline_s == pytest.approx(deadline)


def test_known_models_ignore_the_baseline() -> None:
    assert entry_defaults(HAIKU_45, baseline_p90_s=9.0) == entry_defaults(HAIKU_45)


@pytest.mark.parametrize("baseline", [0.0, -1.0, float("nan"), float("inf")])
def test_a_bad_baseline_is_refused(baseline: float) -> None:
    with pytest.raises(SettingsError) as err:
        entry_defaults("us.anthropic.claude-sonnet-5", baseline_p90_s=baseline)
    assert err.value.field == "baseline_p90_s"


# --------------------------------------------------------------------------- the list


def test_default_list() -> None:
    s = Settings.for_models(DEFAULT_LIST)
    assert [e.model_id for e in s.entries] == list(DEFAULT_LIST)
    assert [e.deadline_s for e in s.entries] == [4.0, 3.0, None]  # none on the last
    assert [e.trip_ttft_s for e in s.entries] == [3.5, 1.6, 3.5]
    assert [e.recover_ttft_s for e in s.entries] == [2.5, 1.1, 2.5]


def test_global_defaults() -> None:
    s = Settings.for_models(DEFAULT_LIST)
    assert (s.window_calls, s.window_age_s, s.min_samples, s.miss_run, s.pings_per_cycle, s.recover_min_pings,
            s.recover_period_s, s.dwell_s, s.idle_reset_s, s.unavailable_recheck_s) == (
        20, 120.0, 20, 3, 3, 10, 120.0, 180.0, 600.0, 900.0)


def test_a_list_of_one_has_no_deadline() -> None:
    s = Settings.for_models([HAIKU_55])
    assert len(s.entries) == 1
    assert s.entries[0].deadline_s is None


def test_sonnet_first_keeps_a_deadline() -> None:
    s = Settings.for_models([SONNET_55, HAIKU_55])
    assert [e.deadline_s for e in s.entries] == [4.0, None]


def test_global_settings_pass_through() -> None:
    s = Settings.for_models(DEFAULT_LIST, min_samples=5, pings_per_cycle=4)
    assert (s.min_samples, s.pings_per_cycle) == (5, 4)


def test_entry_lookup_by_index() -> None:
    s = Settings.for_models(DEFAULT_LIST)
    assert s.entry(1).model_id == HAIKU_45
    assert len(s) == 3


def test_settings_are_frozen() -> None:
    s = Settings.for_models(DEFAULT_LIST)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.min_samples = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.entries[0].trip_ttft_s = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize("ids", [[], [HAIKU_55, HAIKU_45, SONNET_55, "us.anthropic.claude-sonnet-5"]])
def test_one_to_three_entries(ids: list[str]) -> None:
    with pytest.raises(SettingsError) as err:
        Settings.for_models(ids)
    assert err.value.field == "entries"


def test_duplicate_entries_refused() -> None:
    with pytest.raises(SettingsError) as err:
        Settings.for_models([HAIKU_55, HAIKU_55])
    assert err.value.field == "entries"


def test_only_the_last_entry_has_no_deadline() -> None:
    a, b = entry_defaults(HAIKU_55), entry_defaults(HAIKU_45, last=True)
    with pytest.raises(SettingsError) as err:  # a deadline on the last entry
        Settings(entries=(a, dataclasses.replace(b, deadline_s=3.0)))
    assert err.value.field == "deadline_s"
    with pytest.raises(SettingsError) as err:  # no deadline before the last
        Settings(entries=(dataclasses.replace(a, deadline_s=None), b))
    assert err.value.field == "deadline_s"
    assert Settings(entries=(a, b)).entries == (a, b)


@pytest.mark.parametrize(("field", "value"), [
    ("trip_ttft_s", 0.0), ("trip_ttft_s", float("nan")), ("trip_ttft_s", float("inf")),
    ("recover_ttft_s", 0.0), ("recover_ttft_s", -1.0),
    ("recover_ttft_s", 3.5),  # not below trip: no hysteresis
    ("recover_ttft_s", 4.0),
    ("deadline_s", 0.0), ("deadline_s", float("nan")),
    ("error_rate_limit", 0.0), ("error_rate_limit", 1.5), ("error_rate_limit", float("nan")),
])
def test_entry_bounds(field: str, value: float) -> None:
    with pytest.raises(SettingsError) as err:
        dataclasses.replace(entry_defaults(HAIKU_55), **{field: value})
    assert err.value.field == field


def test_entry_needs_a_model_id() -> None:
    with pytest.raises(SettingsError) as err:
        EntrySettings(model_id="", trip_ttft_s=3.5, recover_ttft_s=2.5, deadline_s=4.0)
    assert err.value.field == "model_id"


@pytest.mark.parametrize(("field", "value"), [
    ("window_calls", 0), ("window_age_s", 0.0), ("window_age_s", float("nan")),
    ("min_samples", 0), ("min_samples", 21),  # can never be met by a window of 20
    ("miss_run", 0), ("pings_per_cycle", 0), ("pings_per_cycle", 6), ("pings_per_cycle", 2.5), ("pings_per_cycle", True),
    ("recover_min_pings", 0), ("recover_period_s", -1.0), ("dwell_s", -1.0), ("dwell_s", float("inf")),
    ("idle_reset_s", 0.0), ("unavailable_recheck_s", 0.0),
])
def test_global_bounds(field: str, value: float) -> None:
    with pytest.raises(SettingsError) as err:
        Settings.for_models(DEFAULT_LIST, **{field: value})
    assert err.value.field == field


@pytest.mark.parametrize(("field", "value"), [("pings_per_cycle", 1), ("pings_per_cycle", 5), ("dwell_s", 0.0),
                                              ("recover_period_s", 0.0), ("min_samples", 20)])
def test_global_bounds_inclusive(field: str, value: float) -> None:
    assert getattr(Settings.for_models(DEFAULT_LIST, **{field: value}), field) == value


def test_settings_error_is_a_value_error_naming_the_field() -> None:
    with pytest.raises(ValueError, match="pings_per_cycle"):
        Settings.for_models(DEFAULT_LIST, pings_per_cycle=9)


# --------------------------------------------------------------------------- series keys


@pytest.mark.parametrize(("model_id", "key"), [
    (HAIKU_55, "haiku-5-5"), ("global.anthropic.claude-haiku-5-5", "haiku-5-5"), ("anthropic.claude-haiku-5-5", "haiku-5-5"),
    ("us.anthropic.claude-haiku-5-5-20260901-v1:0", "haiku-5-5"), (HAIKU_45, "haiku-4-5"), (SONNET_55, "sonnet-5-5"),
    ("eu.anthropic.claude-opus-12-10", "opus-12-10"),
])
def test_series_key_is_family_major_minor_whatever_the_prefix_and_tail(model_id: str, key: str) -> None:
    """The per-model prompt's key, shared by every profile of one model."""
    assert series_key(model_id) == key
    family, major, minor = key.split("-")
    assert series_parts(model_id) == (family, int(major), int(minor))


@pytest.mark.parametrize("model_id", ["amazon.nova-pro-v1:0", "us.anthropic.claude-haiku", "", "claude-haiku-5-5"])
def test_another_model_has_no_series_key(model_id: str) -> None:
    assert series_key(model_id) is None and series_parts(model_id) is None
