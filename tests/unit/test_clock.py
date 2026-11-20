"""The bank's local date and time line for the agent's system prompt."""

from __future__ import annotations

from datetime import UTC, datetime

from hypothesis import given
from hypothesis import strategies as st

from tac_bridge.clock import DEFAULT_TIMEZONE, local_now_line, zone


def test_english_line_in_new_york_time() -> None:
    now = datetime(2026, 10, 14, 15, 42, tzinfo=UTC)  # 11:42 AM EDT
    assert local_now_line(now, zone("America/New_York"), "en-US") == \
        "Current date and time at the bank (America/New_York): Wednesday, October 14, 2026, 11:42 AM."


def test_spanish_line() -> None:
    now = datetime(2026, 12, 6, 23, 5, tzinfo=UTC)  # 6:05 PM EST, Sunday
    assert local_now_line(now, zone("America/New_York"), "es-US") == \
        "Fecha y hora actuales del banco (America/New_York): domingo 6 de diciembre de 2026, 18:05."


def test_the_local_date_can_differ_from_utc() -> None:
    now = datetime(2026, 10, 15, 2, 30, tzinfo=UTC)  # still the 14th in New York
    assert "Wednesday, October 14" in local_now_line(now, zone("America/New_York"), "en-US")


def test_unknown_or_empty_zone_falls_back_to_the_default() -> None:
    assert zone("Not/AZone").key == DEFAULT_TIMEZONE and zone(None).key == DEFAULT_TIMEZONE


@given(st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2100, 1, 1), timezones=st.just(UTC)),
       st.sampled_from(["en-US", "es-US"]), st.sampled_from(["America/New_York", "America/Chicago", "Europe/Madrid", "UTC"]))
def test_property_the_line_names_the_local_day_and_zone(now, lang, tz) -> None:
    line = local_now_line(now, zone(tz), lang)
    local = now.astimezone(zone(tz))
    assert tz in line and str(local.year) in line and str(local.day) in line
