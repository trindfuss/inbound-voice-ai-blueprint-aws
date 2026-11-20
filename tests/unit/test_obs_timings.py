"""``obs.Timings``: named spans in whole milliseconds."""

from __future__ import annotations

import json

import pytest

from owlbank.obs import Timings


class FakeClock:
    """Seconds, advanced by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def tick(self, ms: float) -> None:
        self.now += ms / 1000


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def test_empty_timings_give_an_empty_dict():
    assert Timings().to_dict() == {}


def test_one_span_records_its_milliseconds(clock):
    t = Timings(clock)
    with t.span("state_load"):
        clock.tick(12)
    assert t.to_dict() == {"state_load": 12}


def test_a_name_used_again_accumulates(clock):
    t = Timings(clock)
    for ms in (3, 4, 5):
        with t.span("catalog"):
            clock.tick(ms)
    assert t.to_dict() == {"catalog": 12}


def test_rounding_happens_once_on_the_total(clock):
    """Ten 0.4 ms spans are 4 ms, not ten zeros."""
    t = Timings(clock)
    for _ in range(10):
        with t.span("render"):
            clock.tick(0.4)
    assert t.to_dict() == {"render": 4}


def test_spans_nest_and_the_parent_includes_the_child(clock):
    t = Timings(clock)
    with t.span("resolve"):
        clock.tick(2)
        with t.span("policy"):
            clock.tick(5)
        clock.tick(1)
    assert t.to_dict() == {"resolve": 8, "policy": 5}


def test_the_same_name_re_entered_is_counted_once(clock):
    t = Timings(clock)
    with t.span("catalog"):
        clock.tick(3)
        with t.span("catalog"):
            clock.tick(4)
    assert t.to_dict() == {"catalog": 7}
    with t.span("catalog"):  # closed again: an ordinary span adds normally
        clock.tick(1)
    assert t.to_dict() == {"catalog": 8}


def test_a_span_that_raises_still_records_and_the_error_propagates(clock):
    t = Timings(clock)
    with pytest.raises(RuntimeError, match="boom"), t.span("state_save"):
        clock.tick(9)
        raise RuntimeError("boom")
    assert t.to_dict() == {"state_save": 9}
    with t.span("state_save"):  # the name is not left open by the failure
        clock.tick(1)
    assert t.to_dict() == {"state_save": 10}


def test_a_raising_inner_span_records_both_levels(clock):
    t = Timings(clock)
    with pytest.raises(ValueError), t.span("resolve"), t.span("model"):
        clock.tick(6)
        raise ValueError
    assert t.to_dict() == {"resolve": 6, "model": 6}


def test_names_keep_the_order_they_were_first_opened(clock):
    t = Timings(clock)
    for name in ("state_load", "resolve", "render", "state_load", "state_save"):
        with t.span(name):
            clock.tick(1)
    assert list(t.to_dict()) == ["state_load", "resolve", "render", "state_save"]


def test_an_open_span_counts_zero_until_it_closes(clock):
    t = Timings(clock)
    with t.span("resolve"):
        clock.tick(5)
        assert t.to_dict() == {"resolve": 0}
    assert t.to_dict() == {"resolve": 5}


@pytest.mark.parametrize("bad", ["", "State", "card 1234", "zip=94105", "a" * 41, "1st", "x-y", None, 3])
def test_a_name_that_could_carry_a_value_is_refused(bad):
    t = Timings()
    with pytest.raises(ValueError), t.span(bad):  # type: ignore[arg-type]
        pass
    assert t.to_dict() == {}


def test_the_output_is_names_and_ints_only_and_json_safe(clock):
    t = Timings(clock)
    with t.span("state_load"):
        clock.tick(1.6)
    out = t.to_dict()
    assert all(isinstance(k, str) and type(v) is int for k, v in out.items())
    assert json.loads(json.dumps(out)) == {"state_load": 2}


def test_the_default_clock_measures_real_time():
    t = Timings()
    with t.span("noop"):
        pass
    assert t.to_dict()["noop"] >= 0
