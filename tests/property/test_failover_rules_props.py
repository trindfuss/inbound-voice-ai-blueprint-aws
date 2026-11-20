"""Model failover: properties of the classification and the trip/recovery rules.

* Base: any ``ValidationException`` whose message starts with the request-error prefix
  classifies as a request error, never unavailable or a hard error; request errors are not samples, so
  interleaving them anywhere changes no statistic and no verdict.
* With the default ``min_samples`` (20), one or two slow samples among any
  number of fast ones (a window of at least ``min_samples``), and so fewer than ``miss_run`` misses in
  a row, never trip; a run of ``miss_run`` misses always does.
* Hysteresis: a recovered verdict implies not tripped, for any statistics and any valid settings.
* A recovered verdict over raw samples implies the samples under the recovery
  threshold (OK, not abandoned or failed) span at least ``recover_period_s`` to the newest sample.
* ``classify`` never raises over arbitrary codes and messages, and returns one known class.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from owlbank.failover.classify import REQUEST_ERROR_PREFIX, UNAVAILABLE, Outcome, classify
from owlbank.failover.rules import Sample, Stats, is_degraded, is_recovered, percentile
from owlbank.failover.settings import EntrySettings, Settings

SETTINGS = settings(max_examples=300, deadline=None)

KNOWN_CODES = ["ValidationException", "validationException", "ThrottlingException", "throttlingException",
               "ResourceNotFoundException", "AccessDeniedException", "ModelErrorException",
               "ModelStreamErrorException", "ModelNotReadyException", "ModelTimeoutException",
               "ServiceUnavailableException", "ReadTimeoutError", "EndpointConnectionError"]
codes = st.one_of(st.none(), st.text(max_size=40), st.sampled_from(KNOWN_CODES))
messages = st.one_of(st.none(), st.text(max_size=200))
CLASSES = {*Outcome, UNAVAILABLE}

pos = st.floats(min_value=0.05, max_value=30.0, allow_nan=False, allow_infinity=False)
ttfts = st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False)
SAMPLE_OUTCOMES = [o for o in Outcome if o is not Outcome.REQUEST_ERROR]


@st.composite
def entries(draw: st.DrawFn) -> EntrySettings:
    trip = draw(pos)
    recover = draw(st.floats(min_value=0.01, max_value=trip, exclude_max=True, allow_nan=False))
    return EntrySettings(model_id="m", trip_ttft_s=trip, recover_ttft_s=recover, deadline_s=None,
                         error_rate_limit=draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False)))


@st.composite
def rule_settings(draw: st.DrawFn) -> Settings:
    window = draw(st.integers(1, 40))
    return Settings(entries=(draw(entries()),), window_calls=window, min_samples=draw(st.integers(1, window)),
                    miss_run=draw(st.integers(1, 6)), recover_min_pings=draw(st.integers(1, 30)),
                    recover_period_s=draw(st.floats(min_value=0.0, max_value=600.0, allow_nan=False)))


@st.composite
def samples(draw: st.DrawFn, max_size: int = 40) -> list[Sample]:
    out = []
    for t in sorted(draw(st.lists(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
                                  max_size=max_size))):
        outcome = draw(st.sampled_from(SAMPLE_OUTCOMES))
        ttft = draw(ttfts) if outcome in (Outcome.OK, Outcome.ABANDONED) and draw(st.booleans()) else None
        out.append(Sample(at_s=t, outcome=outcome, ttft_s=ttft))
    return out


@st.composite
def any_stats(draw: st.DrawFn) -> Stats:
    latency = draw(st.integers(0, 60))
    errors = draw(st.integers(0, 60))
    calls = latency + errors + draw(st.integers(0, 5))
    p90 = draw(ttfts) if latency else None
    p50 = draw(st.floats(min_value=0.0, max_value=p90, allow_nan=False)) if p90 is not None else None
    return Stats(calls=calls, latency=latency, errors=errors, p50_s=p50, p90_s=p90,
                 miss_run=draw(st.integers(0, min(latency, 10))),
                 span_s=(span := draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False))),
                 good_span_s=draw(st.floats(min_value=0.0, max_value=span, allow_nan=False)))


# --------------------------------------------------------------------------- classify


@SETTINGS
@given(st.text(max_size=200), st.sampled_from(["ValidationException", "validationException"]),
       st.sampled_from(["", " ", "\n"]))
def test_request_error_prefix_is_never_unavailable_or_hard(rest: str, code: str, lead: str) -> None:
    assert classify(code, lead + REQUEST_ERROR_PREFIX + rest) is Outcome.REQUEST_ERROR


@SETTINGS
@given(codes, messages)
def test_classify_never_raises_and_returns_a_known_class(code: str | None, message: str | None) -> None:
    result = classify(code, message)
    assert result in CLASSES
    assert result not in (Outcome.OK, Outcome.ABANDONED)  # those are never errors


@SETTINGS
@given(messages)
def test_not_found_and_denied_are_unavailable_whatever_the_message(message: str | None) -> None:
    assert classify("ResourceNotFoundException", message) == UNAVAILABLE
    assert classify("AccessDeniedException", message) == UNAVAILABLE


# --------------------------------------------------------------------------- At the rules


@SETTINGS
@given(samples(), st.lists(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False), max_size=10),
       rule_settings())
def test_request_errors_change_no_statistic_or_verdict(base: list[Sample], at: list[float],
                                                          rules: Settings) -> None:
    noisy = base + [Sample(at_s=t, outcome=Outcome.REQUEST_ERROR) for t in at]
    entry = rules.entry(0)
    under = entry.recover_ttft_s
    assert Stats.of(noisy) == Stats.of(base)
    assert Stats.of(noisy, good_under_s=under) == Stats.of(base, good_under_s=under)
    assert is_degraded(Stats.of(noisy), entry, rules) == is_degraded(Stats.of(base), entry, rules)
    assert (is_recovered(Stats.of(noisy, good_under_s=under), entry, rules)
            == is_recovered(Stats.of(base, good_under_s=under), entry, rules))


# --------------------------------------------------------------------------- The rule half


DEFAULTS = Settings.for_models(["us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                                "us.anthropic.claude-sonnet-5-5"])


@SETTINGS
@given(st.integers(0, len(DEFAULTS) - 1), st.integers(0, 60), st.integers(1, 2), st.data())
def test_one_or_two_slow_samples_among_fast_ones_never_trip(index: int, extra: int, slow: int,
                                                                  data: st.DataObject) -> None:
    """The default settings (``min_samples`` 20): a window of at least ``min_samples`` calls with one or
    two slow samples (over the trip threshold, or deadline misses) among any number of fast ones, and
    so fewer than ``miss_run`` misses in a row, never trips."""
    entry = DEFAULTS.entry(index)
    assert DEFAULTS.min_samples == 20 and DEFAULTS.miss_run > 2
    n = DEFAULTS.min_samples - slow + extra  # fast samples; the window has n + slow >= min_samples
    fast = data.draw(st.lists(st.floats(min_value=0.0, max_value=entry.trip_ttft_s, allow_nan=False),
                              min_size=n, max_size=n))
    window = [(Outcome.OK, x) for x in fast]
    for _ in range(slow):
        miss = entry.deadline_s is not None and data.draw(st.booleans())
        sample = ((Outcome.ABANDONED, entry.deadline_s) if miss else
                  (Outcome.OK, data.draw(st.floats(min_value=entry.trip_ttft_s, max_value=60.0,
                                                   exclude_min=True, allow_nan=False))))
        window.insert(data.draw(st.integers(0, len(window))), sample)
    stats = Stats.of([Sample(at_s=float(i), outcome=o, ttft_s=t) for i, (o, t) in enumerate(window)])
    assert stats.calls >= DEFAULTS.min_samples and stats.miss_run < DEFAULTS.miss_run
    assert not is_degraded(stats, entry, DEFAULTS)


@SETTINGS
@given(rule_settings(), st.integers(0, 40))
def test_a_run_of_misses_trips_even_below_the_minimum(rules: Settings, fast: int) -> None:
    entry = rules.entry(0)
    window = [Sample(at_s=float(i), outcome=Outcome.OK, ttft_s=0.0) for i in range(fast)]
    window += [Sample(at_s=float(fast + i), outcome=Outcome.ABANDONED, ttft_s=None)
               for i in range(rules.miss_run)]
    assert is_degraded(Stats.of(window), entry, rules)


# --------------------------------------------------------------------------- hysteresis


@SETTINGS
@given(rule_settings(), any_stats())
def test_recovered_implies_not_tripped(rules: Settings, stats: Stats) -> None:
    entry = rules.entry(0)
    if is_recovered(stats, entry, rules):
        assert not is_degraded(stats, entry, rules)


@SETTINGS
@given(rule_settings(), samples())
def test_recovered_implies_not_tripped_over_samples(rules: Settings, window: list[Sample]) -> None:
    entry = rules.entry(0)
    stats = Stats.of(window, good_under_s=entry.recover_ttft_s)
    if is_recovered(stats, entry, rules):
        assert not is_degraded(stats, entry, rules)


@SETTINGS
@given(rule_settings(), samples())
def test_recovered_implies_good_samples_span_the_period(rules: Settings, window: list[Sample]) -> None:
    entry = rules.entry(0)
    if not is_recovered(Stats.of(window, good_under_s=entry.recover_ttft_s), entry, rules):
        return
    good = [s.at_s for s in window if s.outcome is Outcome.OK and s.ttft_s is not None
            and s.ttft_s < entry.recover_ttft_s]
    counted = [s.at_s for s in window if s.outcome is not Outcome.REQUEST_ERROR]
    assert good and max(counted) - min(good) >= rules.recover_period_s


@SETTINGS
@given(rule_settings(), st.integers(1, 30), st.floats(min_value=0.0, max_value=0.999, allow_nan=False))
def test_a_steady_fast_window_recovers_and_never_trips(rules: Settings, extra: int, frac: float) -> None:
    """Every TTFT strictly below the recovery threshold, enough pings over the period: recovered."""
    entry = rules.entry(0)
    n = rules.recover_min_pings + extra
    ttft = entry.recover_ttft_s * frac
    step = rules.recover_period_s / (n - 1)
    stats = Stats.of([Sample(at_s=i * step, outcome=Outcome.OK, ttft_s=ttft) for i in range(n)],
                     good_under_s=entry.recover_ttft_s)
    assume(stats.good_span_s >= rules.recover_period_s)  # float rounding of i * step
    assert is_recovered(stats, entry, rules)
    assert not is_degraded(stats, entry, rules)


# --------------------------------------------------------------------------- percentile


@SETTINGS
@given(st.lists(ttfts, min_size=1, max_size=60))
def test_percentile_is_a_sample_and_ordered(values: list[float]) -> None:
    p50, p90 = percentile(values, 50), percentile(values, 90)
    assert p50 in values and p90 in values
    assert p50 is not None and p90 is not None and p50 <= p90
    # Nearest rank: at least 90% of the samples are at or below p90, and fewer than 90% below it.
    n = len(values)
    assert sum(v <= p90 for v in values) * 10 >= 9 * n
    assert sum(v < p90 for v in values) * 10 < 9 * n
