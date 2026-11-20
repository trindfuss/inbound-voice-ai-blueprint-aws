"""tests/component/slot_world.py: the leak scan's log surface drops only numeric size and timing
fields (a generated value equal to a byte count or a duration is a collision, not a leak)."""

from __future__ import annotations

import json

from tests.component.slot_world import _without_measurements


def test_only_numeric_size_and_timing_fields_are_dropped():
    line = json.dumps({"event": "invocation_start", "event_bytes": 1137, "duration_ms": 12.5,
                       "ssn4": "1137", "attempts": 1137, "ok_ms": True})
    kept = json.loads(_without_measurements(line))
    assert "event_bytes" not in kept and "duration_ms" not in kept
    assert kept["ssn4"] == "1137"      # a string under any key is still scanned
    assert kept["attempts"] == 1137    # a number under any other key is still scanned
    assert kept["ok_ms"] is True       # booleans aren't measurements


def test_nested_timing_maps_and_fields_are_dropped():
    line = json.dumps({"event": "factor_answer", "timings_ms": {"resolve": 123, "save": 4.5},
                       "inner": {"wait_ms": 123, "zip": "123", "list": [{"took_ms": 123, "n": 123}]},
                       "mixed_ms": {"resolve": 123, "said": "123"}, "empty_ms": {}})
    kept = json.loads(_without_measurements(line))
    assert "timings_ms" not in kept                       # a map of timings is a measurement
    assert kept["inner"] == {"zip": "123", "list": [{"n": 123}]}  # nested fields go, the rest stays
    assert kept["mixed_ms"] == {"resolve": 123, "said": "123"}   # a map with any non-number is scanned
    assert kept["empty_ms"] == {}


def test_a_non_json_line_is_unchanged():
    assert _without_measurements("plain 1137 text") == "plain 1137 text"
