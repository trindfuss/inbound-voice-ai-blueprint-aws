"""Slot model bench maths: ``scripts/bench/slot_model_bench.py``.

* cost per 1,000 calls is 1000 x the mean over the calls of (in x p_in + out x p_out), for any token
  counts and prices, and doesn't depend on the order of the calls;
* the percentiles are order statistics of the samples (nearest rank), order-invariant, and p50 <= p90;
* the summary's per-group answer, model-call and error counts partition the answers;
* per mode, the same maths hold over each mode's own samples: the modes' model calls add up
  to the run's, each mode's cost and percentiles are those of its own calls, a mode with no model
  call (``deterministic``) has no latency or cost, and the stop is exactly "some p90 > 1 s".
"""

from __future__ import annotations

import math

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scripts.bench import slot_model_bench as smb

PROPS = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])

latency = st.floats(min_value=0.0, max_value=60_000.0, allow_nan=False, allow_infinity=False)
samples = st.lists(latency, min_size=1, max_size=60)
tokens = st.integers(min_value=0, max_value=200_000)
usage = st.lists(st.tuples(tokens, tokens), min_size=1, max_size=40)
per_mtok = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)


def price(p_in: float, p_out: float) -> smb.Price:
    return smb.Price(model_id="m", input_per_mtok=p_in, output_per_mtok=p_out, currency="USD",
                     source="test", effective="2026-10-01", looked_up="2026-10-19")


@PROPS
@given(usage, per_mtok, per_mtok, st.randoms(use_true_random=False))
def test_cost_per_1000_is_the_mean_per_call_cost_times_1000(calls, p_in, p_out, rnd):
    pr = price(p_in, p_out)
    expect = 1000 * sum(i * p_in / 1e6 + o * p_out / 1e6 for i, o in calls) / len(calls)
    got = smb.cost_per_1000(calls, pr)
    assert got is not None and math.isclose(got, expect, rel_tol=1e-9, abs_tol=1e-12)
    shuffled = list(calls)
    rnd.shuffle(shuffled)
    assert math.isclose(smb.cost_per_1000(shuffled, pr) or 0.0, got, rel_tol=1e-9, abs_tol=1e-12)


@PROPS
@given(samples, st.floats(min_value=0.0, max_value=1.0))
def test_pct_is_an_order_statistic(xs, p):
    got = smb.pct(xs, p)
    ys = sorted(xs)
    assert got in xs
    assert got == ys[max(1, math.ceil(p * len(ys))) - 1]


@PROPS
@given(samples, st.randoms(use_true_random=False))
def test_pct_is_order_invariant_and_p50_le_p90(xs, rnd):
    ys = list(xs)
    rnd.shuffle(ys)
    p50, p90 = smb.pct(xs, 0.5), smb.pct(xs, 0.9)
    assert p50 == smb.pct(ys, 0.5) and p90 == smb.pct(ys, 0.9)
    assert p50 is not None and p90 is not None
    assert min(xs) <= p50 <= p90 <= max(xs)


case_ids = st.sampled_from([c.id for c in smb.CASES])
sample = st.builds(
    lambda cid, lat, tin, tout, kind, failed, calls, answered: (
        smb.Sample.failed(smb.CASE_BY_ID[cid], latency_ms=lat, error="E") if failed
        else smb.Sample.of(smb.CASE_BY_ID[cid], latency_ms=lat,
                           input_tokens=tin if calls and answered else None,
                           output_tokens=tout if calls and answered else None,
                           server_latency_ms=None, kind=kind, value=None, model_calls=calls,
                           model=("ok" if answered else "timeout") if calls else "not_called")),
    case_ids, latency, tokens, tokens, st.sampled_from(smb.KINDS), st.booleans(), st.integers(0, 1),
    st.booleans())


@PROPS
@given(st.lists(sample, min_size=1, max_size=40))
def test_groups_partition_the_calls_and_p50_le_p90(ss):
    summary = smb.summarize(ss, price(1.0, 5.0))
    overall = summary["overall"]
    assert overall["answers"] == len(ss)
    assert overall["model_calls"] == sum(s.model_calls for s in ss)
    assert overall["errors"] == sum(1 for s in ss if s.error)
    for group in ("by_slot_type", "by_lang", "by_category"):
        assert sum(g["answers"] for g in summary[group].values()) == len(ss)
        assert sum(g["model_calls"] for g in summary[group].values()) == overall["model_calls"]
        assert sum(g["errors"] for g in summary[group].values()) == overall["errors"]
    for g in (overall, *summary["by_slot_type"].values()):
        if g["latency_p50_ms"] is not None:
            assert g["latency_p50_ms"] <= g["latency_p90_ms"]


def _det(s: smb.Sample) -> smb.Sample:
    """``s`` as a deterministic answer: no model call, no tokens."""
    return smb.Sample.of(smb.CASE_BY_ID[s.case_id], latency_ms=s.latency_ms, input_tokens=None,
                         output_tokens=None, server_latency_ms=None, kind=s.kind, value=None,
                         model_calls=0, model="not_called", mode="deterministic")


@PROPS
@given(st.lists(sample, min_size=1, max_size=30), st.lists(sample, min_size=1, max_size=30),
       st.lists(sample, min_size=1, max_size=30), per_mtok, per_mtok)
def test_per_mode_totals(det, assisted, gen, p_in, p_out):
    pr = price(p_in, p_out)
    runs = {"deterministic": [_det(s) for s in det], "assisted": assisted, "generative": gen}
    summaries = {m: smb.summarize(ss, pr) for m, ss in runs.items()}
    total = smb.summarize([s for ss in runs.values() for s in ss], pr)["overall"]
    assert sum(v["overall"]["model_calls"] for v in summaries.values()) == total["model_calls"]
    assert sum(v["overall"]["answers"] for v in summaries.values()) == total["answers"]
    d = summaries["deterministic"]["overall"]
    assert d["model_calls"] == 0 and d["latency_p90_ms"] is None and d["cost_per_1000_usd"] is None
    for m, ss in runs.items():
        o = summaries[m]["overall"]
        called = [s for s in ss if s.model_calls and s.error is None]
        lat = [s.latency_ms for s in called]
        assert o["latency_p50_ms"] == smb.pct(lat, 0.5) and o["latency_p90_ms"] == smb.pct(lat, 0.9)
        pairs = [(s.input_tokens, s.output_tokens) for s in called
                 if s.input_tokens is not None and s.output_tokens is not None]
        expect = smb.cost_per_1000(pairs, pr)
        if expect is None:
            assert o["cost_per_1000_usd"] is None
        else:
            assert math.isclose(o["cost_per_1000_usd"], round(expect, 6), rel_tol=1e-9, abs_tol=1e-12)
    verdict = smb.latency_stop(summaries)
    over = [m for m, v in summaries.items()
            if v["overall"]["latency_p90_ms"] is not None and v["overall"]["latency_p90_ms"] > 1000.0]
    assert verdict["over"] == over and verdict["stop"] == bool(over)
    assert (verdict["message"] or "").startswith("STOP:") == bool(over)
