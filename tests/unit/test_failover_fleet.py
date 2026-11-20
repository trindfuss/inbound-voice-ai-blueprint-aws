"""The fleet picture's pure core.

Table keys and their order, the weighted counters of a sample item, window statistics from histograms,
``report_prob`` against the write budget, the state item and its size, and ``fleet_verdict`` across
cycles: the trip, hysteresis and dwell of :mod:`owlbank.failover.rules`, recovery from the poller's
pings one rank at a time, and the unavailable state.
"""

from __future__ import annotations

import math
import random
import re
from pathlib import Path

import pytest

from owlbank.failover.classify import Outcome
from owlbank.failover.fleet import (
    BUCKET_S,
    ITEM_LIMIT_BYTES,
    MAX_PING_SAMPLES,
    SAMPLE_TTL_S,
    SAMPLES_PK,
    STATE_PK,
    STATE_SK,
    Bucket,
    Counters,
    EntryState,
    ModelVerdict,
    Pings,
    State,
    align,
    bucket_start,
    fleet_verdict,
    initial_state,
    item_size,
    model_tag,
    model_tags,
    parse_sample_sort_key,
    ping_target,
    report_every,
    sample_expires_at,
    sample_key,
    sample_sort_key,
    state_key,
    window_buckets,
    window_range,
    window_stats,
)
from owlbank.failover.histogram import NUM_BINS, Histogram
from owlbank.failover.rules import Sample, Stats, is_degraded
from owlbank.failover.settings import Settings

MODELS = ["us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5-20251001-v1:0",
          "us.anthropic.claude-sonnet-5-5"]
S = Settings.for_models(MODELS)
TAGS = model_tags(S)
T0 = 1_800_000_000.0  # a wall-clock epoch, as the poller's


def bucket(start: float, index: int, ttfts=(), *, weight: int = 1, errors: int = 0, misses: int = 0,
           deadline: float = 4.0, unavailable: int = 0, tags: tuple[str, ...] = TAGS) -> Bucket:
    c = Counters()
    for t in ttfts:
        c.record(Outcome.OK, t, weight=weight)
    for _ in range(errors):
        c.record(Outcome.THROTTLED, None)
    for _ in range(misses):
        c.record(Outcome.ABANDONED, deadline)
    for _ in range(unavailable):
        c.record_unavailable()
    item = {**sample_key(bucket_start(start), tags[index]), **c.increments()}
    return Bucket.from_item(item)


def bucket_for(settings: Settings, start: float, index: int, ttfts=(), **kw) -> Bucket:
    """A sample of ``settings``' entry ``index`` (another list than ``S``)."""
    return bucket(start, index, ttfts, tags=model_tags(settings), **kw)


def pings_ok(start: float, n: int, ttft: float, step: float = 10.0) -> list[Sample]:
    return [Sample(start + i * step, Outcome.OK, ttft) for i in range(n)]


# --------------------------------------------------------------------------- keys


def test_bucket_start_floors_to_ten_seconds():
    assert BUCKET_S == 10
    assert bucket_start(T0 + 9.999) == int(T0)
    assert bucket_start(T0 + 10) == int(T0) + 10
    with pytest.raises(ValueError):
        bucket_start(-1)


def test_infra_leading_keys_name_the_same_partitions():
    """The IAM leading-key conditions (infra ``model_failover``) name the partitions these keys use.
    infra/ is its own project (CDK), so its constants are read from the source."""
    src = (Path(__file__).resolve().parents[2] / "infra" / "owlbank_infra" / "model_failover.py").read_text()
    consts = dict(re.findall(r'^(SAMPLES_PARTITION|STATE_PARTITION) = "([^"]*)"$', src, re.M))
    assert consts == {"SAMPLES_PARTITION": SAMPLES_PK, "STATE_PARTITION": STATE_PK}


def test_model_tags_are_short_fixed_and_name_the_model():
    assert tuple(model_tag(m) for m in MODELS) == TAGS
    assert all(re.fullmatch(r"[0-9a-f]{8}", t) for t in TAGS) and len(set(TAGS)) == 3
    assert model_tag(MODELS[0]) == model_tag(MODELS[0]) != model_tag(MODELS[0] + "x")
    for bad in ("", None, 3):
        with pytest.raises(ValueError):
            model_tag(bad)  # type: ignore[arg-type]


def test_sample_and_state_keys():
    tag = TAGS[2]
    assert sample_sort_key(1_800_000_000, tag) == f"1800000000#{tag}"
    assert sample_sort_key(10, TAGS[0]) == f"0000000010#{TAGS[0]}"
    # fixed partitions, not the stage: each stage has its own table
    assert sample_key(1_800_000_000, tag) == {"s": "samples", "sk": f"1800000000#{tag}"}
    assert state_key() == {"s": "state", "sk": STATE_SK}
    assert parse_sample_sort_key(f"1800000000#{tag}") == (1_800_000_000, tag)
    for bad in ("state", "18#x", "1800000000#", f"1800000001#{tag}", f"-180000000#{tag}", "1800000000#1",
                f"1800000000#{tag.upper()}", f"1800000000#{tag}0"):
        with pytest.raises(ValueError):
            parse_sample_sort_key(bad)
    for bad_tag in (0, "1", MODELS[0], tag[:7], tag.upper()):
        with pytest.raises(ValueError):
            sample_sort_key(10, bad_tag)  # type: ignore[arg-type]
    assert sample_expires_at(1_800_000_000) == 1_800_000_000 + SAMPLE_TTL_S
    assert SAMPLE_TTL_S == 900


def test_sort_keys_order_by_time_then_model():
    keys = [sample_sort_key(b, t) for b in (990, 1000, 10_000, 1_800_000_000) for t in sorted(TAGS)]
    assert sorted(keys) == keys
    assert sorted([*keys[3:], *keys[:3]]) == keys


def test_window_range_excludes_older_buckets_and_includes_the_current_one():
    now = T0 + 7  # inside bucket
    first, last = window_buckets(now, S)
    assert last == int(T0)
    assert first == int(T0) - 110  # the oldest bucket starting within window_age_s (120) of now
    low, high = window_range(now, S)
    edges = ("00000000", "ffffffff", *TAGS)
    inside = [sample_sort_key(b, t) for b in range(first, last + 1, BUCKET_S) for t in edges]
    assert all(low <= k <= high for k in inside)
    older = [sample_sort_key(first - BUCKET_S, t) for t in edges] + [sample_sort_key(first - 900, TAGS[2])]
    assert all(k < low for k in older)
    assert sample_sort_key(last + BUCKET_S, "00000000") > high


# --------------------------------------------------------------------------- counters


def test_counters_weight_sampled_latency_and_count_errors_and_misses_once():
    c = Counters()
    c.record(Outcome.OK, 0.6, weight=4)
    c.record(Outcome.OK, 0.6, weight=4)
    c.record(Outcome.ABANDONED, 4.0, weight=4)  # always reported: the weight is ignored
    c.record(Outcome.THROTTLED, None, weight=4)
    c.record(Outcome.ERROR, None)
    c.record(Outcome.REQUEST_ERROR, None)  # never a sample
    c.record_unavailable()
    inc = c.increments()
    assert inc == {"n": 11, "raw": 3, "err": 2, "miss": 1, "unav": 1, "h2": 8, "h16": 1}
    assert all(isinstance(v, int) and v > 0 for v in inc.values())
    assert not c.empty
    assert Counters().increments() == {} and Counters().empty


def test_counters_refuse_bad_input():
    c = Counters()
    with pytest.raises(ValueError):
        c.record(Outcome.OK, None)
    with pytest.raises(ValueError):
        c.record(Outcome.OK, 1.0, weight=0)
    with pytest.raises(ValueError):
        c.record(Outcome.ABANDONED, -1.0)


def test_bucket_from_item_round_trip_and_refusals():
    b = bucket(T0, 1, [0.5, 0.7], weight=3, errors=1, misses=2)
    assert (b.start, b.model, b.calls, b.raw, b.errors, b.misses, b.unavailable) == (int(T0), TAGS[1], 9, 4, 1, 2, 0)
    assert b.hist.total == 8
    with pytest.raises(ValueError):
        Bucket.from_item({"s": "samples", "sk": "state"})
    with pytest.raises(ValueError):
        Bucket.from_item({"s": "samples", "sk": f"1800000000#{TAGS[0]}", "n": -1})
    with pytest.raises(ValueError):
        Bucket.from_item({"s": "samples", "sk": "1800000000#0", "err": 1.5})


# --------------------------------------------------------------------------- window statistics


def test_window_stats_from_histograms():
    bs = [bucket(T0, 0, [0.4] * 18), bucket(T0 + 10, 0, [3.9, 3.9], errors=1)]
    st = window_stats(bs)
    assert (st.calls, st.latency, st.errors, st.miss_run) == (21, 20, 1, 0)
    assert st.p90_s == 0.5 and st.p50_s == 0.5
    assert st.span_s == 10
    assert window_stats([]) == Stats(calls=0, latency=0, errors=0, p50_s=None, p90_s=None, miss_run=0, span_s=0.0)


def test_window_miss_run_counts_trailing_all_miss_buckets_only():
    assert window_stats([bucket(T0, 0, [0.5]), bucket(T0 + 10, 0, misses=2),
                         bucket(T0 + 20, 0, misses=1)]).miss_run == 3
    # a bucket with another call ends the run, its own misses unknown in order: not counted
    assert window_stats([bucket(T0, 0, misses=2), bucket(T0 + 10, 0, [0.5], misses=2)]).miss_run == 0
    assert window_stats([bucket(T0, 0, misses=2), bucket(T0 + 10, 0, errors=1)]).miss_run == 0


def test_window_verdict_matches_the_raw_sample_verdict():
    """FP-fleet, one example: the same samples judged raw and through bucketed histograms."""
    rng = random.Random(7)
    entry = S.entry(0)
    for slow_share in (0.0, 0.05, 0.3, 0.6):
        ttfts = [rng.uniform(3.8, 6.0) if rng.random() < slow_share else rng.uniform(0.6, 2.0)
                 for _ in range(60)]
        raw = Stats.of(Sample(T0 + i, Outcome.OK, t) for i, t in enumerate(ttfts))
        bs = [bucket(T0 + 10 * k, 0, ttfts[10 * k:10 * k + 10]) for k in range(6)]
        assert is_degraded(window_stats(bs), entry, S) == is_degraded(raw, entry, S), slow_share


# --------------------------------------------------------------------------- report_prob


def test_report_every_is_one_below_the_budget_and_rises_with_traffic():
    def rate(per_s: float) -> list[Bucket]:  # per_s ok calls a second over the 120 s window
        per_bucket = int(per_s * BUCKET_S)
        return [bucket(T0 + 10 * k, 0, [0.5] * per_bucket) for k in range(12)]

    assert report_every([], S) == 1
    assert report_every(rate(4.0), S) == 1
    assert report_every(rate(5.0), S) == 1
    assert report_every(rate(5.1), S) == 2
    assert report_every(rate(50.0), S) == 10
    ks = [report_every(rate(r), S) for r in (1, 5, 10, 20, 40, 80)]
    assert ks == sorted(ks)
    assert report_every(rate(10.0), S, budget_per_s=1.0) == 10
    # errors and misses are always reported, so they don't drive the sampling of latency samples
    assert report_every([bucket(T0, 0, errors=500, misses=500)], S) == 1


def test_weighted_totals_stay_unbiased_when_report_prob_changes_between_buckets():
    """Each bucket uses the report_prob in force; E[weight x Bernoulli(1/weight)] is exactly 1."""
    rng = random.Random(11)
    true_per_bucket = 200
    ks = [1, 2, 5, 10, 25, 3]
    trials, totals = 400, []
    for _ in range(trials):
        total = 0
        for k in ks:
            c = Counters()
            for _ in range(true_per_bucket):
                if rng.random() < 1 / k:
                    c.record(Outcome.OK, 0.5, weight=k)
            total += c.increments().get("n", 0)
        totals.append(total)
    mean = sum(totals) / trials
    expected = true_per_bucket * len(ks)
    assert abs(mean - expected) < 0.02 * expected


# --------------------------------------------------------------------------- state item


def test_initial_state_and_item_round_trip():
    st = initial_state(S, T0)
    assert [m.state for m in st.models] == [EntryState.HEALTHY] * 3
    assert [m.rank for m in st.models] == [0, 1, 2]
    assert st.report_every == 1 and st.report_prob == 1.0 and st.pings is None
    item = st.to_item()
    assert {k: item[k] for k in ("s", "sk")} == state_key()
    assert State.from_item(item) == st


def test_state_item_holds_only_integers_strings_lists_maps_and_nulls():
    st = State(version=4, written_at_s=T0 + 0.1234, report_every=7,
               models=tuple(initial_state(S, T0).models),
               pings=Pings(0, (Sample(T0 - 3.3, Outcome.OK, 1.234), Sample(T0, Outcome.ABANDONED, 4.0))))

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        else:
            assert v is None or isinstance(v, str | int) and not isinstance(v, bool), v

    item = st.to_item()
    walk(item)
    again = State.from_item(item)
    assert again.to_item() == item
    assert again.pings is not None and again.pings.samples[1].outcome is Outcome.ABANDONED


def test_worst_case_items_are_under_one_write_unit():
    sample = {**sample_key(9_999_999_990, TAGS[2]), "n": 10**9, "raw": 10**9, "err": 10**9,
              "miss": 10**9, "unav": 10**9, "expires_at": 9_999_999_999,
              **{f"h{i}": 10**9 for i in range(NUM_BINS)}}
    assert item_size(sample) < ITEM_LIMIT_BYTES
    models = tuple(ModelVerdict(index=m.index, model=m.model, state=EntryState.UNAVAILABLE, rank=m.rank, since_s=T0,
                               p50_s=math.inf, p90_s=7.777, error_rate=0.333) for m in initial_state(S, T0).models)
    pings = Pings(2, tuple(Sample(T0 + i, Outcome.ABANDONED, 7.0) for i in range(MAX_PING_SAMPLES)))
    st = State(version=10**9, written_at_s=T0, models=models, pings=pings, report_every=10_000)
    assert item_size(st.to_item()) < ITEM_LIMIT_BYTES


def test_from_item_refuses_malformed_records():
    item = initial_state(S, T0).to_item()
    for broken in ({**item, "models": "x"}, {**item, "version": -1}, {**item, "report_every": 0},
                   {**item, "models": [{**item["models"][0], "state": "sick"}]},
                   {**item, "pings": {"index": 0, "samples": [[1, 2]]}}):
        with pytest.raises(ValueError):
            State.from_item(broken)


# --------------------------------------------------------------------------- fleet_verdict


def slow_window(now: float, index: int = 0, ttft: float = 4.5, n: int = 20) -> list[Bucket]:
    return [bucket(now - 10, index, [ttft] * n)]


def slow_window_for(settings: Settings, now: float, index: int = 0, ttft: float = 4.5, n: int = 20) -> list[Bucket]:
    return [bucket_for(settings, now - 10, index, [ttft] * n)]


def fast_window(now: float, index: int = 0, n: int = 20) -> list[Bucket]:
    return [bucket(now - 10, index, [0.8] * n)]


def test_no_data_keeps_everything_healthy():
    st = fleet_verdict([], S, T0, None)
    assert [m.state for m in st.models] == [EntryState.HEALTHY] * 3
    assert st.written_at_s == T0 and st.version == 0
    assert ping_target(st) is None


def test_trip_on_p90_over_threshold_with_enough_samples():
    st = fleet_verdict(slow_window(T0), S, T0, None)
    assert st.models[0].state is EntryState.DEGRADED and st.models[0].since_s == T0
    assert st.version == 1
    assert [m.rank for m in st.models] == [2, 0, 1]  # healthy first, in order
    assert ping_target(st) == 0


def test_too_few_samples_or_a_blip_never_trip():
    assert fleet_verdict(slow_window(T0, n=19), S, T0, None).models[0].state is EntryState.HEALTHY
    blip = [bucket(T0 - 20, 0, [0.8] * 18), bucket(T0 - 10, 0, [9.0, 9.0])]
    assert fleet_verdict(blip, S, T0, None).models[0].state is EntryState.HEALTHY


def test_a_few_heavily_weighted_reports_never_trip_the_p90_rule():
    """With sampling: 19 reported samples of weight 50 are 950 weighted ones, still not a window."""
    st = fleet_verdict([bucket(T0 - 10, 0, [4.5] * 19, weight=50)], S, T0, None)
    assert st.models[0].state is EntryState.HEALTHY
    assert st.models[0].p90_s is not None and st.models[0].p90_s > 3.5  # still shown
    st = fleet_verdict([bucket(T0 - 10, 0, [4.5] * 20, weight=50)], S, T0, None)
    assert st.models[0].state is EntryState.DEGRADED


def test_trip_on_a_run_of_misses_and_on_the_error_rate():
    st = fleet_verdict([bucket(T0 - 10, 1, misses=3, deadline=3.0)], S, T0, None)
    assert st.models[1].state is EntryState.DEGRADED
    errs = [bucket(T0 - 10, 0, [0.8] * 15, errors=5)]  # 25% > 20%
    assert fleet_verdict(errs, S, T0, None).models[0].state is EntryState.DEGRADED


def test_buckets_outside_the_window_are_ignored():
    old = [bucket(T0 - 130, 0, [4.5] * 20)]
    assert fleet_verdict(old, S, T0, None).models[0].state is EntryState.HEALTHY
    future = [bucket(T0 + 10, 0, [4.5] * 20)]
    assert fleet_verdict(future, S, T0, None).models[0].state is EntryState.HEALTHY


def test_recovery_needs_enough_good_pings_over_the_period_and_the_dwell():
    st = fleet_verdict(slow_window(T0), S, T0, None)  # degraded
    t = T0
    # 3 good pings per 30 s cycle; recover needs 10 pings over 120 s and 180 s of dwell
    states = []
    for cycle in range(1, 12):
        t = T0 + 30 * cycle
        new = [Sample(t - 24 + 8 * k, Outcome.OK, 1.0) for k in range(3)]
        st = fleet_verdict([], S, t, st, pings=new)
        states.append((t - T0, st.models[0].state))
        if st.models[0].state is EntryState.HEALTHY:
            break
    recovered_at = next(dt for dt, s in states if s is EntryState.HEALTHY)
    assert recovered_at == 180  # pings span 120 s by 150 s; the dwell holds it until 180 s
    assert st.models[0].since_s == t and st.pings is None


def test_slow_or_missed_pings_never_recover():
    st = fleet_verdict(slow_window(T0), S, T0, None)
    for cycle in range(1, 20):
        t = T0 + 30 * cycle
        new = [Sample(t - 24, Outcome.OK, 1.0), Sample(t - 16, Outcome.OK, 3.0),
               Sample(t - 8, Outcome.ABANDONED, 4.0)]
        st = fleet_verdict([], S, t, st, pings=new)
        assert st.models[0].state is EntryState.DEGRADED


def test_pings_are_bounded_to_the_recovery_period_with_one_sample_at_its_edge():
    st = fleet_verdict(slow_window(T0), S, T0, None)
    st = fleet_verdict([], S, T0 + 30, st, pings=pings_ok(T0 - 170, 40, 1.0, step=5.0))  # to t0 + 25
    assert st.pings is not None and st.pings.index == 0
    times = [p.at_s for p in st.pings.samples]
    assert times == sorted(times)
    assert len(times) <= MAX_PING_SAMPLES
    cutoff = T0 + 30 - S.recover_period_s
    assert sum(1 for x in times if x <= cutoff) <= 1


def test_dwell_holds_a_trip_after_a_recovery():
    st = fleet_verdict(slow_window(T0), S, T0, None)
    st = State(version=st.version, written_at_s=st.written_at_s, report_every=1, pings=None,
               models=tuple(ModelVerdict(index=m.index, model=m.model, state=EntryState.HEALTHY, rank=m.rank,
                                        since_s=T0, p50_s=None, p90_s=None, error_rate=None)
                            for m in st.models))  # recovered
    held = fleet_verdict(slow_window(T0 + 179), S, T0 + 179, st)
    assert held.models[0].state is EntryState.HEALTHY
    tripped = fleet_verdict(slow_window(T0 + 180), S, T0 + 180, held)
    assert tripped.models[0].state is EntryState.DEGRADED


def test_recovery_is_one_rank_at_a_time_and_pings_go_to_the_highest_ranked_degraded():
    both = slow_window(T0, 0) + slow_window(T0, 1, ttft=2.5)
    st = fleet_verdict(both, S, T0, None)
    assert [m.state for m in st.models[:2]] == [EntryState.DEGRADED] * 2
    assert ping_target(st) == 0
    assert [m.rank for m in st.models] == [2, 1, 0]  # healthy first, then degraded by p90
    # good pings are for the target only; entry 1 stays degraded while entry 0 recovers
    t = T0
    while st.models[0].state is EntryState.DEGRADED:
        t += 30
        st = fleet_verdict([], S, t, st, pings=[Sample(t - 24 + 8 * k, Outcome.OK, 0.5) for k in range(3)])
        assert st.models[1].state is EntryState.DEGRADED
    assert ping_target(st) == 1 and st.pings is None


def test_all_degraded_ranks_by_p90():
    window = slow_window(T0, 0, ttft=6.0) + slow_window(T0, 1, ttft=2.5) + slow_window(T0, 2, ttft=4.5)
    st = fleet_verdict(window, S, T0, None)
    assert all(m.state is EntryState.DEGRADED for m in st.models)
    assert [m.rank for m in st.models] == [2, 0, 1]


def test_unavailable_at_once_and_back_only_through_a_recheck():
    st = fleet_verdict(slow_window(T0), S, T0, None)  # entry 0 degraded; dwell running
    st = fleet_verdict([bucket(T0 + 5, 0, unavailable=1), bucket(T0 + 5, 1, unavailable=1)], S, T0 + 20, st)
    assert st.models[0].state is EntryState.UNAVAILABLE and st.models[1].state is EntryState.UNAVAILABLE
    assert [m.rank for m in st.models] == [1, 2, 0]
    assert ping_target(st) is None  # never a ping to an unavailable entry
    # good traffic doesn't bring it back; a successful recheck does, and older reports don't undo it
    reports = [bucket(T0 + 5, 0, [0.5] * 30, unavailable=1)]
    st = fleet_verdict(reports, S, T0 + 50, st)
    assert st.models[0].state is EntryState.UNAVAILABLE
    st = fleet_verdict(reports, S, T0 + 60, st, available={0})
    assert st.models[0].state is EntryState.HEALTHY and st.models[0].since_s == T0 + 60
    assert st.models[1].state is EntryState.UNAVAILABLE
    st = fleet_verdict(reports, S, T0 + 90, st)
    assert st.models[0].state is EntryState.HEALTHY
    # a new report after the recheck marks it again, dwell or not
    st = fleet_verdict(reports + [bucket(T0 + 90, 0, unavailable=1)], S, T0 + 95, st)
    assert st.models[0].state is EntryState.UNAVAILABLE


def test_available_ignored_for_an_entry_that_is_not_unavailable():
    st = fleet_verdict(slow_window(T0), S, T0, None)
    again = fleet_verdict([], S, T0 + 30, st, available={0, 1})
    assert again.models[0].state is EntryState.DEGRADED and again.models[1].since_s is None


def test_version_bumps_only_on_change_and_written_at_always_moves():
    st = fleet_verdict(fast_window(T0), S, T0, None)
    same = fleet_verdict(fast_window(T0 + 30), S, T0 + 30, st)
    assert same.version == st.version and same.written_at_s == T0 + 30
    changed = fleet_verdict(slow_window(T0 + 60), S, T0 + 60, same)
    assert changed.version == st.version + 1


def test_a_changed_entry_list_keeps_each_listed_models_state():
    other = Settings.for_models(MODELS[:2])
    st = fleet_verdict(slow_window(T0), S, T0, None)
    assert st.models[0].state is EntryState.DEGRADED
    again = fleet_verdict([], other, T0 + 30, st)
    assert len(again.models) == 2 and [m.state for m in again.models] == [EntryState.DEGRADED, EntryState.HEALTHY]
    assert again.version == st.version + 1  # the stored record changed
    new = fleet_verdict([], Settings.for_models(["new.model", MODELS[1]]), T0 + 30, st)
    assert [m.state for m in new.models] == [EntryState.HEALTHY, EntryState.HEALTHY]
    assert new.models[0].since_s is None and new.models[0].model == model_tag("new.model")


def test_state_for_a_b_read_with_c_b_starts_c_healthy_and_keeps_b():
    a, b, c = "model.a", "model.b", "model.c"
    ab, cb = Settings.for_models([a, b]), Settings.for_models([c, b])
    st = fleet_verdict(slow_window_for(ab, T0, 0, ttft=9.0) + slow_window_for(ab, T0, 1, ttft=9.0), ab, T0, None)
    st = fleet_verdict([bucket_for(ab, T0 + 5, 0, unavailable=1)], ab, T0 + 20, st)
    assert [m.state for m in st.models] == [EntryState.UNAVAILABLE, EntryState.DEGRADED]
    item = st.to_item()
    stored = State.from_item(item)
    aligned = align(stored, cb)
    assert [m.state for m in aligned.models] == [EntryState.HEALTHY, EntryState.DEGRADED]
    assert aligned.models[0].since_s is None and aligned.models[1] == ModelVerdict(
        index=1, model=model_tag(b), state=EntryState.DEGRADED, rank=aligned.models[1].rank,
        since_s=st.models[1].since_s, p50_s=st.models[1].p50_s, p90_s=st.models[1].p90_s,
        error_rate=st.models[1].error_rate)
    # C's own reports go to C; A's (an old-list bridge's) are nobody's in the new list
    later = fleet_verdict([bucket_for(ab, T0 + 40, 0, unavailable=1)], cb, T0 + 45, stored)
    assert [m.state for m in later.models] == [EntryState.HEALTHY, EntryState.DEGRADED]
    assert ping_target(later) == 1
    assert align(aligned, cb) == aligned  # idempotent


def test_a_swapped_list_moves_state_and_pings_with_the_model():
    st = fleet_verdict(slow_window(T0), S, T0, None)
    st = fleet_verdict([], S, T0 + 30, st, pings=pings_ok(T0 + 6, 3, 0.5, step=8.0))
    assert st.pings is not None and st.pings.index == 0
    swapped = Settings.for_models([MODELS[1], MODELS[0], MODELS[2]])
    moved = align(st, swapped)
    assert [m.state for m in moved.models] == [EntryState.HEALTHY, EntryState.DEGRADED, EntryState.HEALTHY]
    assert moved.pings == Pings(1, st.pings.samples) and ping_target(moved) == 1
    assert [m.rank for m in moved.models] == [0, 2, 1]
    again = fleet_verdict(slow_window_for(swapped, T0 + 60, 0, ttft=0.5), swapped, T0 + 60, st)
    assert again.models[1].state is EntryState.DEGRADED and again.models[0].state is EntryState.HEALTHY


def test_a_record_without_model_tags_is_malformed():
    item = initial_state(S, T0).to_item()
    for broken in ({**item, "models": [{k: v for k, v in m.items() if k != "m"} for m in item["models"]]},
                   {**item, "models": [{**m, "m": MODELS[0]} for m in item["models"]]},
                   {**item, "models": [{**m, "m": TAGS[0]} for m in item["models"]]}):  # duplicates
        with pytest.raises(ValueError):
            State.from_item(broken)


def test_report_prob_is_published_in_the_state():
    window = [bucket(T0 - 10 * k, 1, [0.5] * 100) for k in range(1, 12)]  # 1,100 calls over 120 s
    st = fleet_verdict(window, S, T0, None)
    assert st.report_every == 2 and st.report_prob == 0.5
    assert Histogram.from_attrs(st.to_item()) == Histogram.empty()  # no bins in the state item
