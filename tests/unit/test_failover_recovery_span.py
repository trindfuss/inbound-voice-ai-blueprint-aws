"""The recovery span.

A degraded entry recovers only after at least ``recover_min_pings`` ping samples *spanning at
least* ``recover_period_s`` *under the recovery threshold*. In a live run Haiku 5.5 recovered at
23:40:14 with its good pings spanning about 103 s (the first good ping started about 23:38:29; its
cycle's log line is 23:38:46). ``fleet._prune_pings`` keeps one sample at the period's edge so the kept
samples can span the period; that edge sample was the last ping abandoned while the test delay was on
(TTFT = the 4 s deadline), and ``Stats.of`` measures ``span_s`` from it. Its 4 s is the largest of
13 latency samples, so the nearest-rank p90 (the 12th) is a good ping's: the abandoned sample counts
in the p90 but only supplies the span.

The test replays that sequence with the poller's timings (a ``model_pings`` line is written when a
cycle's last ping ends; a ping's ``at_s`` is when it started).
"""

from __future__ import annotations

from owlbank.failover.classify import Outcome
from owlbank.failover.fleet import (
    Bucket,
    Counters,
    EntryState,
    State,
    bucket_start,
    fleet_verdict,
    model_tags,
    sample_key,
)
from owlbank.failover.rules import Sample
from owlbank.failover.settings import Settings

MODELS = ["us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5-20251001-v1:0",
          "us.anthropic.claude-sonnet-5-5"]
S = Settings.for_models(MODELS)
T0 = 1_800_000_000.0  # a wall-clock epoch, as the poller's
DEADLINE = 4.0  # Haiku 5.5's first-token deadline; an abandoned ping's TTFT

# Cycle times after the last cycle of abandoned pings (23:38:14.0), from the poller's log.
GOOD_CYCLES = (32.8, 61.1, 89.7, 120.0, 150.0, 180.0)
# Ping starts in a cycle, before the cycle's decision (8 s apart; a good ping takes ~1.3 s, an
# abandoned one the 4 s deadline).
GOOD_STARTS = (-17.3, -9.3, -1.3)
ABANDONED_STARTS = (-20.0, -12.0, -4.0)
GOOD_TTFTS = (1.13, 1.17, 1.33)  # p90 ~1.33 s, under Haiku 5.5's recovery threshold


def degraded_at(now: float) -> State:
    c = Counters()
    for _ in range(20):
        c.record(Outcome.OK, 4.5)
    item = {**sample_key(bucket_start(now - 10), model_tags(S)[0]), **c.increments()}
    st = fleet_verdict([Bucket.from_item(item)], S, now, None)
    assert st.models[0].state is EntryState.DEGRADED
    return st


def test_recovery_needs_good_pings_spanning_the_recovery_period():
    assert max(GOOD_TTFTS) < S.entry(0).recover_ttft_s
    st = degraded_at(T0)
    last_bad = T0 + 30 * 14  # 7 minutes of abandoned pings: well past the dwell
    for cycle in range(1, 15):
        now = T0 + 30 * cycle
        st = fleet_verdict([], S, now, st,
                           pings=[Sample(now + d, Outcome.ABANDONED, DEADLINE) for d in ABANDONED_STARTS])
        assert st.models[0].state is EntryState.DEGRADED

    first_good: float | None = None
    recovered: tuple[float, float, list[Sample]] | None = None  # (now, good span, samples judged)
    for offset in GOOD_CYCLES:
        now = last_bad + offset
        new = [Sample(now + d, Outcome.OK, t) for d, t in zip(GOOD_STARTS, GOOD_TTFTS, strict=True)]
        first_good = new[0].at_s if first_good is None else first_good
        judged = [*(st.pings.samples if st.pings is not None else ()), *new]
        st = fleet_verdict([], S, now, st, pings=new)
        if st.models[0].state is EntryState.HEALTHY:
            recovered = (now, new[-1].at_s - first_good, judged)
            break

    assert recovered is not None, "never recovered, though good pings span the period by the last cycle"
    now, good_span, judged = recovered
    # the sample _prune_pings keeps at the period's edge, from which Stats.of measures span_s
    edge = max((s for s in judged if s.at_s <= now - S.recover_period_s), key=lambda s: s.at_s)
    assert good_span >= S.recover_period_s, (
        f"recovered {now - first_good:.1f} s after the first good ping with good pings spanning only "
        f"{good_span:.1f} s < recover_period_s {S.recover_period_s:.0f} s; the span was measured from "
        f"the edge sample at {edge.at_s - first_good:+.1f} s ({edge.outcome.value}, ttft {edge.ttft_s})")


def test_a_zero_period_still_needs_one_good_ping() -> None:
    """Hypothesis found it: with ``recover_period_s`` 0 and one abandoned ping, nothing was good yet
    the entry counted as recovered."""
    import dataclasses

    from owlbank.failover.rules import Stats, is_recovered
    zero = dataclasses.replace(S, recover_period_s=0.0, recover_min_pings=1)
    entry = zero.entry(0)
    abandoned = Stats.of([Sample(at_s=0.0, outcome=Outcome.ABANDONED, ttft_s=0.0)],
                         good_under_s=entry.recover_ttft_s)
    assert not is_recovered(abandoned, entry, zero)
    good = Stats.of([Sample(at_s=0.0, outcome=Outcome.OK, ttft_s=0.0)], good_under_s=entry.recover_ttft_s)
    assert is_recovered(good, entry, zero)
