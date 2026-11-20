"""The fixed-bin TTFT histogram."""

from __future__ import annotations

import math

import pytest

from owlbank.failover.histogram import (
    BIN_ATTRS,
    BIN_S,
    NUM_BINS,
    RANGE_S,
    Histogram,
    bin_index,
    covers,
    merge,
)
from owlbank.failover.rules import percentile as raw_percentile
from owlbank.failover.settings import Settings, entry_defaults


def test_bins_are_quarter_seconds_with_one_open_bin():
    assert BIN_S == 0.25
    assert int(RANGE_S / BIN_S) + 1 == NUM_BINS
    assert tuple(f"h{i}" for i in range(NUM_BINS)) == BIN_ATTRS
    assert bin_index(0.0) == 0
    assert bin_index(0.2499) == 0
    assert bin_index(0.25) == 1
    assert bin_index(3.5) == 14
    assert bin_index(RANGE_S - 1e-9) == NUM_BINS - 2
    assert bin_index(RANGE_S) == NUM_BINS - 1
    assert bin_index(120.0) == NUM_BINS - 1


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
def test_bin_index_refuses_bad_ttft(bad):
    with pytest.raises(ValueError):
        bin_index(bad)


def test_every_default_threshold_and_deadline_is_inside_the_range():
    """A threshold in the open bin could not be judged; the 12 bins of (0 to 3 s) could not."""
    s = Settings.for_models(["us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                             "us.anthropic.claude-sonnet-5-5"])
    for e in s.entries:
        assert covers(e.trip_ttft_s) and covers(e.recover_ttft_s)
        assert e.deadline_s is None or covers(e.deadline_s)
    other = entry_defaults("us.anthropic.claude-opus-9-1")
    assert covers(other.trip_ttft_s) and other.deadline_s is not None and covers(other.deadline_s)
    assert not covers(RANGE_S)


def test_weighted_add_and_merge():
    h = Histogram.empty().add(0.3, weight=5).add(0.3).add(9.0, weight=2)
    assert h.total == 8
    assert h.counts[1] == 6 and h.counts[NUM_BINS - 1] == 2
    m = merge([h, Histogram.of([0.1, 0.1]), Histogram.empty()])
    assert m.total == 10 and m.counts[0] == 2
    assert h + Histogram.of([0.1]) == merge([h, Histogram.of([0.1])])


@pytest.mark.parametrize("weight", [0, -1, 1.5, True])
def test_weight_must_be_a_positive_integer(weight):
    with pytest.raises(ValueError):
        Histogram.empty().add(1.0, weight=weight)


def test_counts_must_be_non_negative_integers():
    with pytest.raises(ValueError):
        Histogram((0,) * (NUM_BINS - 1) + (-1,))
    with pytest.raises(ValueError):
        Histogram((0,) * 3)
    with pytest.raises(ValueError):
        Histogram((0.5,) + (0,) * (NUM_BINS - 1))


def test_percentile_is_the_upper_edge_of_the_nearest_rank_bin():
    h = Histogram.of([0.1] * 9 + [2.6])
    assert h.percentile(50) == 0.25
    assert h.percentile(90) == 0.25
    assert h.percentile(100) == 2.75
    assert Histogram.empty().percentile(90) is None
    assert Histogram.of([20.0]).percentile(90) == math.inf
    with pytest.raises(ValueError):
        h.percentile(0)


def test_percentile_within_one_bin_of_the_raw_percentile():
    values = [0.05 * i for i in range(1, 140)]
    h = Histogram.of(values)
    for p in (50, 90, 99):
        raw = raw_percentile(values, p)
        assert raw is not None
        got = h.percentile(p)
        assert got is not None and raw < got <= raw + BIN_S


def test_attrs_round_trip_and_sparse():
    h = Histogram.of([0.1, 0.1, 3.9]).add(30.0, weight=7)
    attrs = h.to_attrs()
    assert attrs == {"h0": 2, "h15": 1, f"h{NUM_BINS - 1}": 7}
    assert Histogram.from_attrs({"s": "dev", **attrs}) == h
    assert Histogram.from_attrs({}) == Histogram.empty()


@pytest.mark.parametrize("bad", [-1, "3", 1.5, True])
def test_from_attrs_refuses_bad_counters(bad):
    with pytest.raises(ValueError):
        Histogram.from_attrs({"h3": bad})
