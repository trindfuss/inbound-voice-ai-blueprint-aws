"""The pure trip and recovery rules.

* :meth:`Stats.of` over raw samples: calls, latency samples, hard errors, nearest-rank p50/p90, the
  trailing run of deadline misses, the time spanned; request errors are not samples.
* Trip: p90 strictly over the trip threshold with at least ``min_samples`` latency samples, never
  below the minimum; a run of ``miss_run`` misses even below the minimum; an error rate strictly over
  the limit with at least ``min_samples`` calls.
* Recovery: at least ``recover_min_pings`` ping samples spanning at least ``recover_period_s``, p90 strictly
  below the recovery threshold, error rate strictly under the limit.
* Dwell: a change is allowed once ``dwell_s`` has passed, by explicit comparison.
"""

from __future__ import annotations

import pytest

from owlbank.failover.classify import Outcome
from owlbank.failover.rules import Sample, Stats, dwell_elapsed, is_degraded, is_recovered, percentile
from owlbank.failover.settings import EntrySettings, Settings

HAIKU_55 = "us.anthropic.claude-haiku-5-5"  # trip 3.5, recover 2.5, deadline 4.0
SETTINGS = Settings.for_models([HAIKU_55, "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                                "us.anthropic.claude-sonnet-5-5"])
ENTRY = SETTINGS.entry(0)


def ok(t: float, ttft: float) -> Sample:
    return Sample(at_s=t, outcome=Outcome.OK, ttft_s=ttft)


def miss(t: float) -> Sample:
    return Sample(at_s=t, outcome=Outcome.ABANDONED, ttft_s=ENTRY.deadline_s)


def err(t: float, outcome: Outcome = Outcome.THROTTLED) -> Sample:
    return Sample(at_s=t, outcome=outcome)


def req(t: float) -> Sample:
    return Sample(at_s=t, outcome=Outcome.REQUEST_ERROR)


def oks(ttfts: list[float], start: float = 0.0, step: float = 1.0) -> list[Sample]:
    return [ok(start + i * step, x) for i, x in enumerate(ttfts)]


# --------------------------------------------------------------------------- percentile


def test_percentile_is_nearest_rank() -> None:
    assert percentile([], 90) is None
    assert percentile([2.0], 90) == 2.0
    values = [float(i) for i in range(1, 11)]  # 1..10
    assert percentile(values, 90) == 9.0
    assert percentile(values, 50) == 5.0
    assert percentile(list(reversed(values)), 90) == 9.0
    assert percentile([float(i) for i in range(1, 21)], 90) == 18.0
    assert percentile([1.0] * 7 + [4.0], 90) == 4.0  # n=8: rank 8, the largest


@pytest.mark.parametrize("p", [0, 101, -5])
def test_percentile_bounds(p: int) -> None:
    with pytest.raises(ValueError, match="percentile"):
        percentile([1.0], p)


# --------------------------------------------------------------------------- Stats.of


def test_stats_of_mixed_samples() -> None:
    samples = [ok(0, 1.0), err(1), ok(2, 2.0), req(3), miss(4), miss(5)]
    s = Stats.of(samples)
    assert (s.calls, s.latency, s.errors, s.miss_run) == (5, 4, 1, 2)
    assert s.p90_s == 4.0
    assert s.p50_s == 2.0
    assert s.span_s == 5.0  # 0 to 5; the request error at 3 is not a sample
    assert s.error_rate == pytest.approx(0.2)


def test_stats_of_nothing() -> None:
    s = Stats.of([])
    assert (s.calls, s.latency, s.errors, s.p50_s, s.p90_s, s.miss_run, s.span_s, s.error_rate) == (
        0, 0, 0, None, None, 0, 0.0, 0.0)


def test_miss_run_is_the_trailing_run() -> None:
    assert Stats.of([miss(0), miss(1), ok(2, 1.0), miss(3)]).miss_run == 1
    assert Stats.of([miss(0), miss(1), err(2)]).miss_run == 0  # a hard error ends the run
    assert Stats.of([miss(0), miss(1), req(2), miss(3)]).miss_run == 3  # a request error doesn't


def test_stats_order_by_time_not_by_list_position() -> None:
    assert Stats.of([miss(5), ok(1, 1.0), miss(3)]) == Stats.of([ok(1, 1.0), miss(3), miss(5)])


def test_ok_without_a_ttft_counts_as_a_call_only() -> None:
    s = Stats.of([Sample(at_s=0, outcome=Outcome.OK)])
    assert (s.calls, s.latency, s.p90_s) == (1, 0, None)


def test_request_errors_are_not_samples() -> None:
    base = [ok(0, 1.0), miss(1), err(2)]
    assert Stats.of(base + [req(3), req(4)]) == Stats.of(base)


def test_stats_refuses_an_unavailable_or_unknown_outcome() -> None:
    with pytest.raises(ValueError, match="outcome"):
        Sample(at_s=0, outcome="unavailable")  # type: ignore[arg-type]


def test_sample_refuses_a_negative_or_non_finite_ttft() -> None:
    with pytest.raises(ValueError, match="ttft_s"):
        ok(0, -0.1)
    with pytest.raises(ValueError, match="ttft_s"):
        ok(0, float("nan"))


# --------------------------------------------------------------------------- trip: the window


def test_trips_on_p90_over_threshold_with_enough_samples() -> None:
    stats = Stats.of(oks([1.0] * 17 + [3.6] * 3))  # 20 samples, p90 3.6 > 3.5
    assert is_degraded(stats, ENTRY, SETTINGS)


def test_no_trip_below_the_minimum_samples() -> None:
    stats = Stats.of(oks([3.9] * 19))  # every sample slow, but only 19 of 20
    assert not is_degraded(stats, ENTRY, SETTINGS)


def test_no_trip_at_exactly_the_threshold() -> None:
    assert not is_degraded(Stats.of(oks([3.5] * 20)), ENTRY, SETTINGS)


def test_two_slow_of_twenty_do_not_trip() -> None:
    # Nearest-rank p90 of 20 is the 18th value: still fast with two slow samples.
    assert not is_degraded(Stats.of(oks([1.0] * 18 + [3.9, 3.9])), ENTRY, SETTINGS)
    assert not is_degraded(Stats.of(oks([1.0] * 18) + [miss(20), ok(21, 1.0), miss(22)][-2:]), ENTRY, SETTINGS)


def test_three_slow_of_twenty_trip() -> None:
    assert is_degraded(Stats.of(oks([1.0] * 17 + [3.9, 3.9, 3.9])), ENTRY, SETTINGS)
    assert is_degraded(Stats.of(oks([3.9, 1.0, 3.9] + [1.0] * 16 + [3.9])), ENTRY, SETTINGS)


def test_hard_errors_do_not_count_toward_the_latency_minimum() -> None:
    samples = oks([3.9] * 15) + [err(20 + i) for i in range(5)]  # 20 calls, 15 latency, 25% errors
    stats = Stats.of(samples)
    assert stats.p90_s == 3.9
    assert is_degraded(stats, ENTRY, SETTINGS)  # by the error rate, not the window p90
    # With the error rate under the limit, the p90 alone does not trip on 16 latency samples.
    lenient = Stats.of(oks([3.9] * 16) + [err(20 + i) for i in range(4)])  # 20 calls, 20% errors
    assert not is_degraded(lenient, ENTRY, SETTINGS)


# --------------------------------------------------------------------------- trip: the run of misses


def test_trips_on_a_run_of_misses_below_the_minimum() -> None:
    assert is_degraded(Stats.of([miss(0), miss(1), miss(2)]), ENTRY, SETTINGS)


def test_two_misses_in_a_row_do_not_trip() -> None:
    assert not is_degraded(Stats.of([miss(0), miss(1)]), ENTRY, SETTINGS)
    assert not is_degraded(Stats.of([miss(0), miss(1), ok(2, 1.0), miss(3), miss(4)]), ENTRY, SETTINGS)


def test_one_or_two_slow_calls_alone_do_not_trip() -> None:
    assert not is_degraded(Stats.of([ok(0, 3.9)]), ENTRY, SETTINGS)
    assert not is_degraded(Stats.of([miss(0), ok(1, 3.9)]), ENTRY, SETTINGS)


# --------------------------------------------------------------------------- trip: the error rate


def test_trips_on_the_error_rate_with_enough_calls() -> None:
    samples = oks([1.0] * 15) + [err(20 + i, Outcome.TIMEOUT) for i in range(5)]  # 20 calls, 25%
    assert is_degraded(Stats.of(samples), ENTRY, SETTINGS)


def test_error_rate_at_the_limit_does_not_trip() -> None:
    samples = oks([1.0] * 16) + [err(20 + i, Outcome.ERROR) for i in range(4)]  # 20 calls, exactly 20%
    assert not is_degraded(Stats.of(samples), ENTRY, SETTINGS)


def test_error_rate_needs_the_minimum_calls() -> None:
    assert not is_degraded(Stats.of([err(0), err(1), err(2)]), ENTRY, SETTINGS)


# --------------------------------------------------------------------------- recovery


def pings(ttfts: list[float], period: float = 120.0) -> list[Sample]:
    """Ping samples from 0 to exactly ``period``."""
    step = period / (len(ttfts) - 1)
    times = [i * step for i in range(len(ttfts) - 1)] + [period]
    return [ok(t, x) for t, x in zip(times, ttfts, strict=True)]


def ping_stats(samples: list[Sample], entry: EntrySettings = ENTRY) -> Stats:
    """Ping samples' stats as the fleet verdict builds them: the span counts good samples only."""
    return Stats.of(samples, good_under_s=entry.recover_ttft_s)


def test_recovers_strictly_below_the_lower_threshold() -> None:
    assert is_recovered(ping_stats(pings([1.0] * 9 + [2.4])), ENTRY, SETTINGS)


def test_no_recovery_at_the_lower_threshold() -> None:
    assert not is_recovered(ping_stats(pings([2.5] * 10)), ENTRY, SETTINGS)


def test_no_recovery_between_the_thresholds() -> None:
    stats = ping_stats(pings([3.0] * 10))  # under trip 3.5, over recover 2.5
    assert not is_recovered(stats, ENTRY, SETTINGS)
    assert not is_degraded(stats, ENTRY, SETTINGS)


def test_recovery_needs_the_minimum_pings() -> None:
    assert not is_recovered(ping_stats(pings([1.0] * 9)), ENTRY, SETTINGS)


def test_recovery_needs_the_period() -> None:
    assert not is_recovered(ping_stats(pings([1.0] * 10, period=119.9)), ENTRY, SETTINGS)
    assert is_recovered(ping_stats(pings([1.0] * 10, period=120.0)), ENTRY, SETTINGS)


def test_recovery_needs_the_error_rate_under_the_limit() -> None:
    good = pings([1.0] * 12)
    assert is_recovered(ping_stats(good), ENTRY, SETTINGS)
    with_errors = good + [err(130), err(131), err(132)]  # 3 of 15: 20%, not under the limit
    assert not is_recovered(ping_stats(with_errors), ENTRY, SETTINGS)
    assert is_recovered(ping_stats(good + [err(130), err(131)]), ENTRY, SETTINGS)  # 2 of 14


def test_recovery_refused_while_a_run_of_misses_would_trip() -> None:
    stats = Stats(calls=20, latency=20, errors=0, p50_s=1.0, p90_s=1.0, miss_run=3, span_s=200.0,
                  good_span_s=200.0)
    assert not is_recovered(stats, ENTRY, SETTINGS)


def test_a_deadline_miss_among_pings_blocks_recovery() -> None:
    # Abandoned pings count at the deadline (4.0), above the recovery threshold.
    assert not is_recovered(ping_stats(pings([1.0] * 8 + [4.0, 4.0])), ENTRY, SETTINGS)


def test_the_recovery_span_counts_from_the_oldest_good_sample() -> None:
    """An abandoned or failed sample at the period's edge never supplies the span."""
    good = [ok(30.0 + i * 10, 1.0) for i in range(10)]  # 30 to 120: 90 s of good pings
    for edge in (miss(0.0), err(0.0), ok(0.0, 2.5), Sample(at_s=0.0, outcome=Outcome.OK)):
        stats = ping_stats([edge, *good])
        assert (stats.span_s, stats.good_span_s) == (120.0, 90.0)
        assert not is_recovered(stats, ENTRY, SETTINGS)
    assert ping_stats([ok(0.0, 2.4), *good]).good_span_s == 120.0
    assert is_recovered(ping_stats([ok(0.0, 2.4), *good]), ENTRY, SETTINGS)
    assert ping_stats([*good, miss(130.0)]).good_span_s == 100.0  # to the newest sample, whatever it is


def test_no_threshold_no_good_span() -> None:
    stats = Stats.of(pings([1.0] * 10))
    assert stats.span_s == 120.0 and stats.good_span_s == 0.0
    assert not is_recovered(stats, ENTRY, SETTINGS)


def test_last_entry_without_a_deadline_uses_the_same_rules() -> None:
    last = SETTINGS.entry(2)
    assert last.deadline_s is None
    assert is_degraded(Stats.of(oks([6.0] * 20)), last, SETTINGS)
    assert is_recovered(ping_stats(pings([1.0] * 10), last), last, SETTINGS)


# --------------------------------------------------------------------------- dwell


def test_dwell_by_explicit_comparison() -> None:
    assert not dwell_elapsed(since_s=100.0, now_s=279.999, settings=SETTINGS)
    assert dwell_elapsed(since_s=100.0, now_s=280.0, settings=SETTINGS)
    assert dwell_elapsed(since_s=None, now_s=0.0, settings=SETTINGS)  # never changed
    # A clock that read earlier than ``since`` (a timer that fired early) is not elapsed.
    assert not dwell_elapsed(since_s=100.0, now_s=99.0, settings=SETTINGS)
