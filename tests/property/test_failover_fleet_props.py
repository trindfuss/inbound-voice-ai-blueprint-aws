"""Model failover: properties of the fleet picture's core.

* FP-fleet: the verdict from weighted, bucketed histograms equals the raw-sample verdict for any
  generated samples and weights, except when the raw p90 is within one bin under the trip threshold;
  the fleet's run of misses never exceeds the true run, so it never trips where the raw samples don't.
* A histogram percentile is never below the raw one and at most one bin above it.
* Weights never produce a negative or non-integer counter, and the counters add up (unbiased totals).
* Sort keys order by (bucket, model); the window range holds exactly the window's buckets.
* However large the weights, fewer than ``min_samples`` reported samples never trip the window rule.
* ``report_every`` is 1 at or below the budget and never falls as traffic rises.
* Across generated cycles: no state change within the dwell (except to or from unavailable), recovery
  only for the pinged entry and at most one a cycle, never a ping target that is unavailable, and every
  state record round-trips and stays one write unit.
* Over generated ping sequences: whenever a degraded entry recovers, the ping samples under
  its recovery threshold span at least ``recover_period_s`` (from the oldest good one to the newest
  sample), in the pruned set judged and among every ping sent; an abandoned or failed sample at the
  period's edge never supplies the span.
"""

from __future__ import annotations

import math

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from owlbank.failover.classify import Outcome
from owlbank.failover.fleet import (
    BUCKET_S,
    ITEM_LIMIT_BYTES,
    Bucket,
    Counters,
    EntryState,
    State,
    align,
    fleet_verdict,
    initial_state,
    item_size,
    judged_pings,
    model_tag,
    ping_target,
    report_every,
    sample_key,
    sample_sort_key,
    window_buckets,
    window_range,
    window_stats,
)
from owlbank.failover.histogram import BIN_S, RANGE_S, Histogram
from owlbank.failover.rules import Sample, Stats, is_degraded, percentile
from owlbank.failover.settings import EntrySettings, Settings

PROPS = settings(max_examples=300, deadline=None)
T0 = 1_800_000_000

ttfts = st.floats(min_value=0.0, max_value=12.0, allow_nan=False, allow_infinity=False)
weights = st.integers(min_value=1, max_value=50)
MAX_W = 10_000  # MAX_REPORT_EVERY
M = model_tag("m")  # the single-entry tests' model
tags = st.text("0123456789abcdef", min_size=8, max_size=8)


@st.composite
def entries(draw: st.DrawFn) -> EntrySettings:
    trip = draw(st.floats(min_value=0.3, max_value=RANGE_S - 0.01, allow_nan=False))
    recover = draw(st.floats(min_value=0.05, max_value=trip, exclude_max=True, allow_nan=False))
    rate = draw(st.floats(min_value=0.05, max_value=1.0, allow_nan=False))
    return EntrySettings("m", trip_ttft_s=trip, recover_ttft_s=recover, deadline_s=None, error_rate_limit=rate)


#: A process's sampled calls in one bucket: the weight in force there, and (outcome, ttft) per call.
calls = st.tuples(st.sampled_from([Outcome.OK, Outcome.OK, Outcome.OK, Outcome.THROTTLED, Outcome.ABANDONED]),
                  ttfts)
buckets_of_calls = st.lists(st.tuples(weights, st.lists(calls, max_size=25)), min_size=1, max_size=12)


def build(spec, deadline: float = 4.0) -> tuple[list[Bucket], list[Sample]]:
    """Buckets as the poller reads them, and the raw samples they stand for (an OK call of weight k is
    k raw samples, as unbiased sampling means), in time order."""
    buckets, raw = [], []
    for k, (weight, cs) in enumerate(spec):
        start = T0 + BUCKET_S * k
        c = Counters()
        for j, (outcome, ttft) in enumerate(cs):
            at = start + j * BUCKET_S / (len(cs) + 1)
            if outcome is Outcome.OK:
                c.record(outcome, ttft, weight=weight)
                raw += [Sample(at, outcome, ttft)] * weight
            elif outcome is Outcome.ABANDONED:
                c.record(outcome, deadline)
                raw.append(Sample(at, outcome, deadline))
            else:
                c.record(outcome, None)
                raw.append(Sample(at, outcome, None))
        buckets.append(Bucket.from_item({**sample_key(start, M), **c.increments()}))
    return buckets, raw


@PROPS
@given(spec=buckets_of_calls, entry=entries(), min_samples=st.integers(1, 20),
       deadline=st.floats(min_value=0.5, max_value=RANGE_S - 0.01, allow_nan=False))
def test_fleet_verdict_equals_raw_verdict_except_within_one_bin_of_the_threshold(spec, entry, min_samples, deadline):
    s = Settings(entries=(entry,), window_calls=max(20, min_samples), min_samples=min_samples)
    buckets, raw = build(spec, deadline)
    fleet, exact = window_stats(buckets), Stats.of(raw)
    assert (fleet.calls, fleet.latency, fleet.errors) == (exact.calls, exact.latency, exact.errors)
    assert fleet.miss_run <= exact.miss_run
    if exact.miss_run < s.miss_run:
        assume(exact.p90_s is None or not entry.trip_ttft_s - BIN_S < exact.p90_s <= entry.trip_ttft_s)
        assert is_degraded(fleet, entry, s) == is_degraded(exact, entry, s)
    else:  # the raw run trips; the buckets may not see all of it (they never see more)
        assert is_degraded(exact, entry, s)


@PROPS
@given(spec=st.lists(st.tuples(st.integers(1, MAX_W), st.lists(ttfts, max_size=19)), min_size=1, max_size=12),
       data=st.data())
def test_fewer_than_min_samples_reports_never_trip_the_window_rule(spec, data):
    """However large the weights, a window with fewer than min_samples reported samples (and no run of
    misses or error rate to trip on) stays healthy."""
    s = Settings.for_models(["us.anthropic.claude-haiku-5-5"])
    total = sum(len(ts) for _, ts in spec)
    assume(total < s.min_samples)
    now = T0 + BUCKET_S * len(spec)
    window = []
    for k, (w, ts) in enumerate(spec):
        c = Counters()
        for t in ts:
            c.record(Outcome.OK, t, weight=w)
        if not c.empty:
            window.append(Bucket.from_item({**sample_key(T0 + BUCKET_S * k, model_tag(s.entry(0).model_id)),
                                            **c.increments()}))
    st_ = fleet_verdict(window, s, now, None)
    assert st_.models[0].state is EntryState.HEALTHY


@PROPS
@given(values=st.lists(ttfts, min_size=1, max_size=200), weight=weights, p=st.integers(1, 100))
def test_histogram_percentile_within_one_bin_above_the_raw_one(values, weight, p):
    raw = percentile(values, p)
    got = Histogram.of(values, weight=weight).percentile(p)
    assert raw is not None and got is not None
    if raw >= RANGE_S:
        assert got == math.inf
    else:
        assert raw < got <= raw + BIN_S


recorded = st.lists(st.tuples(st.sampled_from(list(Outcome)), ttfts, weights), max_size=60)


@PROPS
@given(seq=recorded, unav=st.integers(0, 3))
def test_weights_never_produce_negative_or_non_integer_counters(seq, unav):
    c = Counters()
    for outcome, ttft, w in seq:
        c.record(outcome, ttft, weight=w)
    for _ in range(unav):
        c.record_unavailable()
    inc = c.increments()
    assert all(type(v) is int and v > 0 for v in inc.values())
    ok_weight = sum(w for o, _, w in seq if o is Outcome.OK)
    hard = sum(1 for o, _, _ in seq if o in (Outcome.THROTTLED, Outcome.TIMEOUT, Outcome.ERROR))
    miss = sum(1 for o, _, _ in seq if o is Outcome.ABANDONED)
    assert inc.get("n", 0) == ok_weight + hard + miss
    assert inc.get("err", 0) == hard and inc.get("miss", 0) == miss and inc.get("unav", 0) == unav
    assert c.hist.total == ok_weight + miss


@PROPS
@given(parts=st.lists(recorded, min_size=1, max_size=5))
def test_processes_adding_to_one_item_equal_one_combined_record(parts):
    """ADD updates from several processes into one sample item lose and invent nothing."""
    item: dict[str, int] = {}
    together = Counters()
    for seq in parts:
        c = Counters()
        for outcome, ttft, w in seq:
            c.record(outcome, ttft, weight=w)
            together.record(outcome, ttft, weight=w)
        for k, v in c.increments().items():
            item[k] = item.get(k, 0) + v
    assert item == together.increments()
    b = Bucket.from_item({**sample_key(T0, M), **item})
    assert (b.calls, b.errors, b.misses, b.hist) == (together.calls, together.errors, together.misses,
                                                     together.hist)


bucket_starts = st.integers(0, 10**9 - 1).map(lambda x: x * BUCKET_S)


@PROPS
@given(a=st.tuples(bucket_starts, tags), b=st.tuples(bucket_starts, tags))
def test_sort_keys_order_by_bucket_then_model(a, b):
    ka, kb = sample_sort_key(*a), sample_sort_key(*b)
    assert (ka < kb) == (a < b) and (ka == kb) == (a == b)


@PROPS
@given(now=st.floats(min_value=1000.0, max_value=9.0e9, allow_nan=False),
       age=st.floats(min_value=10.0, max_value=900.0, allow_nan=False),
       probe=st.integers(-200, 200), tag=tags)
def test_window_range_holds_exactly_the_window_buckets(now, age, probe, tag):
    s = Settings.for_models(["a.anthropic.claude-haiku-5-5"], window_age_s=age)
    first, last = window_buckets(now, s)
    low, high = window_range(now, s)
    b = max(0, first + probe * BUCKET_S)
    key = sample_sort_key(b, tag)
    assert (low <= key <= high) == (first <= b <= last)
    assert now - age - 1e-3 <= first < now - age + BUCKET_S and last <= now


@PROPS
@given(lo=st.integers(0, 5_000), hi=st.integers(0, 5_000), budget=st.floats(min_value=0.5, max_value=20.0))
def test_report_every_never_falls_as_traffic_rises(lo, hi, budget):
    s = Settings.for_models(["a.anthropic.claude-haiku-5-5"])
    lo, hi = min(lo, hi), max(lo, hi)

    def window(total: int) -> list[Bucket]:
        c = Counters()
        if total:
            c.record(Outcome.OK, 0.5, weight=total)
        return [Bucket.from_item({**sample_key(T0, M), **c.increments()})]

    k_lo, k_hi = report_every(window(lo), s, budget_per_s=budget), report_every(window(hi), s, budget_per_s=budget)
    assert 1 <= k_lo <= k_hi
    if lo / s.window_age_s <= budget:
        assert k_lo == 1
    assert hi / s.window_age_s / k_hi <= budget or k_hi == 10_000


# --------------------------------------------------------------------------- cycles

MODELS = ["us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5-20251001-v1:0",
          "us.anthropic.claude-sonnet-5-5"]
CYCLE = st.fixed_dictionaries({
    "dt": st.sampled_from([30.0, 30.0, 30.0, 0.5, 29.9, 61.0, 400.0]),
    # per model: no samples, fast, slow, misses only, or an unavailable report
    # (weighted toward quiet cycles and good pings, so recoveries happen as well as trips)
    "traffic": st.lists(st.sampled_from(["none"] * 6 + ["fast"] * 3 + ["slow", "miss", "unav"]),
                        min_size=3, max_size=3),
    "ping_ttft": st.sampled_from([0.4, 0.4, 1.0, 1.0, 1.0, 2.4, 3.0, 5.0, None]),
    "recheck_ok": st.booleans(),
})


def _traffic(kind: str, now: float, index: int) -> list[Bucket]:
    c = Counters()
    if kind == "fast":
        for _ in range(25):
            c.record(Outcome.OK, 0.6)
    elif kind == "slow":
        for _ in range(25):
            c.record(Outcome.OK, 5.5)
    elif kind == "miss":
        for _ in range(3):
            c.record(Outcome.ABANDONED, 4.0)
    elif kind == "unav":
        c.record_unavailable()
    if c.empty:
        return []
    start = int(now // BUCKET_S) * BUCKET_S
    return [Bucket.from_item({**sample_key(start, model_tag(MODELS[index])), **c.increments()})]


@settings(max_examples=150, deadline=None)
@given(cycles=st.lists(CYCLE, min_size=1, max_size=40))
def test_cycles_keep_dwell_one_rank_recovery_and_valid_records(cycles):
    s = Settings.for_models(MODELS)
    now = float(T0)
    state: State | None = None
    for cyc in cycles:
        now += cyc["dt"]
        window = [b for i, kind in enumerate(cyc["traffic"]) for b in _traffic(kind, now, i)]
        target = ping_target(state) if state is not None else None
        pings = []
        if target is not None:
            t = cyc["ping_ttft"]
            pings = [Sample(now - 24 + 8 * k, Outcome.ABANDONED if t is None else Outcome.OK,
                            4.0 if t is None else t) for k in range(s.pings_per_cycle)]
        available = {m.index for m in state.models if m.state is EntryState.UNAVAILABLE} \
            if state is not None and cyc["recheck_ok"] else set()
        new = fleet_verdict(window, s, now, state, pings=pings, available=available)

        before = state.models if state is not None else None
        recovered = 0
        for i, m in enumerate(new.models):
            old_state = before[i].state if before else EntryState.HEALTHY
            old_since = before[i].since_s if before else None
            if m.state is old_state:
                assert m.since_s == old_since
                continue
            assert m.since_s == now
            if EntryState.UNAVAILABLE not in (m.state, old_state):  # 4a moves ignore the dwell
                assert old_since is None or now - old_since >= s.dwell_s
            if old_state is EntryState.DEGRADED and m.state is EntryState.HEALTHY:
                recovered += 1
                assert i == target  # only the pinged entry recovers (one rank at a time)
            if old_state is EntryState.UNAVAILABLE:
                assert m.state is EntryState.HEALTHY and i in available
        assert recovered <= 1
        assert sorted(m.rank for m in new.models) == [0, 1, 2]
        nt = ping_target(new)
        assert nt is None or new.models[nt].state is EntryState.DEGRADED
        if new.pings is not None:
            assert new.pings.index == nt
            assert len(new.pings.samples) <= 24
        item = new.to_item()
        assert item_size(item) < ITEM_LIMIT_BYTES
        assert State.from_item(item).to_item() == item
        state = new


#: One recovery ping: its outcome and TTFT (fast, between the thresholds, abandoned at the deadline,
#: or a hard error), and its start in the cycle (seconds before the decision).
PING = st.tuples(st.sampled_from(["fast", "fast", "fast", "slow", "miss", "error"]),
                 st.floats(min_value=0.0, max_value=29.0, allow_nan=False))
#: A cycle: its gap from the last, and its pings: mostly the poller's three, 8 s apart and alike
#: (as in a run of abandoned pings and then good ones), sometimes any mix.
PING_CYCLE = st.tuples(
    st.sampled_from([30.0, 30.0, 30.0, 0.5, 29.9, 61.0, 150.0]),
    st.one_of(st.sampled_from(["fast", "fast", "slow", "miss", "error"]).map(
                  lambda kind: [(kind, 20.0), (kind, 12.0), (kind, 4.0)]),
              st.sampled_from(["fast", "miss"]).map(lambda kind: [(kind, 20.0), (kind, 12.0), (kind, 4.0)]),
              st.lists(PING, max_size=4)))


def _ping(kind: str, at: float, entry: EntrySettings) -> Sample:
    if kind == "fast":
        return Sample(at, Outcome.OK, entry.recover_ttft_s * 0.5)
    if kind == "slow":
        return Sample(at, Outcome.OK, entry.recover_ttft_s)  # at the threshold: not under it
    if kind == "miss":
        return Sample(at, Outcome.ABANDONED, 4.0)
    return Sample(at, Outcome.THROTTLED)


def _good_span(samples: list[Sample], entry: EntrySettings) -> float:
    good = [x.at_s for x in samples if x.outcome is Outcome.OK and x.ttft_s is not None
            and x.ttft_s < entry.recover_ttft_s]
    return max(x.at_s for x in samples) - min(good) if good else 0.0


@settings(max_examples=300, deadline=None)
@given(cycles=st.lists(PING_CYCLE, min_size=1, max_size=40))
def test_recovery_only_when_good_pings_span_the_period(cycles):
    s = Settings.for_models(MODELS)
    entry = s.entry(0)
    now = float(T0)
    slow = Counters()
    for _ in range(25):
        slow.record(Outcome.OK, 5.5)
    state = fleet_verdict([Bucket.from_item({**sample_key(int(now // BUCKET_S) * BUCKET_S, model_tag(MODELS[0])),
                                             **slow.increments()})], s, now, None)
    assert state.models[0].state is EntryState.DEGRADED
    sent: list[Sample] = []
    for dt, spec in cycles:
        now += dt
        assert ping_target(state) == 0
        pings = [_ping(kind, now - before, entry) for kind, before in spec]
        judged = judged_pings(state, s, now, pings)
        sent.extend(pings)
        assert set(judged) <= set(sent)
        new = fleet_verdict([], s, now, state, pings=pings)
        if new.models[0].state is EntryState.HEALTHY:
            assert _good_span(list(judged), entry) >= s.recover_period_s
            assert _good_span(sent, entry) >= s.recover_period_s
            break
        state = new


# --------------------------------------------------------------------------- list changes

NEW_MODEL = "us.anthropic.claude-opus-5-5"


@st.composite
def lists(draw: st.DrawFn) -> list[str]:
    """A deployed list after a change: some of MODELS, maybe a new model, in any order."""
    pool = [*MODELS, NEW_MODEL]
    return draw(st.permutations(pool).map(lambda p: p[:draw(st.integers(1, 3))]))


@settings(max_examples=150, deadline=None)
@given(cycles=st.lists(CYCLE, min_size=1, max_size=15), new_list=lists(), data=st.data())
def test_a_changed_list_keeps_every_models_state_and_starts_new_ones_healthy(cycles, new_list, data):
    """Whatever the stored verdict and however the list changes (reordered, shortened, a model swapped
    in), each listed model keeps its stored verdict and a model the record doesn't name starts healthy
; the pings stay with their model; the record stays valid and within one write unit; samples
    are counted for their own model only."""
    s = Settings.for_models(MODELS)
    now = float(T0)
    state = None
    for cyc in cycles:  # some stored verdict, as the cycles make it
        now += cyc["dt"]
        window = [b for i, kind in enumerate(cyc["traffic"]) for b in _traffic(kind, now, i)]
        target = ping_target(state) if state is not None else None
        pings = [Sample(now - 24 + 8 * k, Outcome.OK, 0.4) for k in range(3)] if target is not None else []
        state = fleet_verdict(window, s, now, state, pings=pings)
    stored = State.from_item(state.to_item())
    other = Settings.for_models(new_list)
    aligned = align(stored, other)
    by_model = {m.model: m for m in stored.models}
    for i, (model_id, m) in enumerate(zip(new_list, aligned.models, strict=True)):
        assert m.index == i and m.model == model_tag(model_id)
        was = by_model.get(m.model)
        if was is None:
            assert (m.state, m.since_s, m.p90_s) == (EntryState.HEALTHY, None, None)
        else:
            assert (m.state, m.since_s, m.p50_s, m.p90_s, m.error_rate) == (
                was.state, was.since_s, was.p50_s, was.p90_s, was.error_rate)
    assert sorted(m.rank for m in aligned.models) == list(range(len(new_list)))
    if aligned.pings is not None:
        assert stored.pings is not None and aligned.pings.samples == stored.pings.samples
        assert aligned.models[aligned.pings.index].model == stored.models[stored.pings.index].model
    elif stored.pings is not None:
        assert stored.models[stored.pings.index].model not in {model_tag(x) for x in new_list}
    assert align(aligned, other) == aligned
    # The next cycle under the new list: an unavailable report of a model only the old list has changes
    # nothing; one of a listed model marks that model, wherever it is listed.
    now += 30
    reported = data.draw(st.sampled_from(MODELS))
    unav = Counters()
    unav.record_unavailable()
    window = [Bucket.from_item({**sample_key(int(now // BUCKET_S) * BUCKET_S, model_tag(reported)),
                                **unav.increments()})]
    new = fleet_verdict(window, other, now, stored)
    for model_id, m, a in zip(new_list, new.models, aligned.models, strict=True):
        if model_id == reported:
            assert m.state is EntryState.UNAVAILABLE
        elif a.state is not EntryState.DEGRADED:  # a degraded one may recover on carried pings
            assert m.state is a.state
    item = new.to_item()
    assert item_size(item) < ITEM_LIMIT_BYTES
    assert State.from_item(item).to_item() == item
    assert new.version >= stored.version


def test_initial_state_names_each_model():
    s = Settings.for_models(MODELS)
    assert [m.model for m in initial_state(s, T0).models] == [model_tag(x) for x in MODELS]
