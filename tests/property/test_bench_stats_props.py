"""Benchmark statistics: the percentile and summary functions of
``scripts/bench/agent_bench.py`` and ``scripts/bench/call_latency.py`` over arbitrary samples.

* ``pct`` is order-invariant, monotone in ``p`` (p50 <= p90), within the sample's range, the sample
  itself for one sample, and ``None`` for none; the two scripts' copies agree.
* the summaries don't depend on the order of the runs or calls, and their ratios stay in [0, 1].
* per hosting: grouping by hosting partitions the calls and their turn samples (none lost
  or duplicated); every group's p50 <= p90; warm + cold = paired connects; each mint is one connect
  and a session is paired with at most one mint; the pre-warm hit rate is in [0, 1].
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scripts.bench import agent_bench as ab
from scripts.bench import call_latency as cl

PROPS = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])

secs = st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False)
samples = st.lists(secs, min_size=1, max_size=40)


@PROPS
@given(samples, st.randoms(use_true_random=False))
def test_pct_is_order_invariant(xs, rnd):
    ys = list(xs)
    rnd.shuffle(ys)
    for p in (0.0, 0.5, 0.9, 1.0):
        assert ab.pct(xs, p) == ab.pct(ys, p) == cl.pct(ys, p)


@PROPS
@given(samples, st.floats(0, 1), st.floats(0, 1))
def test_pct_is_monotone_and_in_range(xs, p, q):
    lo, hi = sorted((p, q))
    assert ab.pct(xs, lo) <= ab.pct(xs, hi)
    assert ab.pct(xs, 0.5) <= ab.pct(xs, 0.9)
    assert round(min(xs), 3) <= ab.pct(xs, lo) <= round(max(xs), 3)
    assert ab.pct(xs, 0.0) == round(min(xs), 3) and ab.pct(xs, 1.0) == round(max(xs), 3)


@PROPS
@given(secs, st.floats(0, 1))
def test_pct_of_one_sample_is_that_sample(x, p):
    assert ab.pct([x], p) == cl.pct([x], p) == round(x, 3)


def test_pct_of_no_sample_is_none():
    assert ab.pct([], 0.5) is None and cl.pct([], 0.9) is None


# ----------------------------------------------------------------------------- agent_bench summaries

tokens = st.integers(min_value=0, max_value=5000)
usage = st.one_of(st.none(), st.fixed_dictionaries(dict.fromkeys(ab.USAGE_KEYS, tokens)))
model_call = st.fixed_dictionaries({"model_latency_s": secs, "model_ttfb_s": secs})


@st.composite
def hops(draw):
    end = draw(secs)
    text = draw(st.one_of(st.none(), st.floats(0, 1).map(lambda f: round(end * f, 3))))
    hop = {"first_event_s": text if text is not None else end, "first_text_s": text, "tool_call_s": None,
           "end_s": end, "tools": [], "usage": draw(usage)}
    if draw(st.booleans()):
        hop["model_calls"] = draw(st.lists(model_call, min_size=1, max_size=3))
    return hop


@st.composite
def runs(draw):
    out = []
    for _ in range(draw(st.integers(1, 8))):
        name = draw(st.sampled_from(["greeting", "transactions_inline", "payment_stepup"]))
        steps = []
        for _ in range(draw(st.integers(1, 3))):
            hs = draw(st.lists(hops(), min_size=1, max_size=3))
            steps.append({"hops": hs, "response_s": draw(secs), "timer_eligible": draw(st.booleans())})
        out.append({"scenario": name, "steps": steps})
    return out


@PROPS
@given(runs(), st.randoms(use_true_random=False))
def test_bench_summaries_ignore_run_order(rs, rnd):
    shuffled = list(rs)
    rnd.shuffle(shuffled)
    assert ab.summarize(rs) == ab.summarize(shuffled)
    assert ab.overall(rs) == ab.overall(shuffled)


@PROPS
@given(runs())
def test_bench_summary_bounds(rs):
    for v in ab.summarize(rs).values():
        if v["response_p50"] is not None:
            assert v["response_p50"] <= v["response_p90"]
    o = ab.overall(rs)
    assert o["hops"] == sum(len(s["hops"]) for r in rs for s in r["steps"])
    assert o["hop_end_p50"] <= o["hop_end_p90"]
    if o["model_latency_p50"] is not None:
        assert o["model_latency_p50"] <= o["model_latency_p90"]
    if o["cache_hit_ratio"] is not None:
        assert 0.0 <= o["cache_hit_ratio"] <= 1.0


@PROPS
@given(tokens, tokens, tokens)
def test_cache_hit_ratio_bounds(read, write, uncached):
    r = ab.cache_hit_ratio(read, write, uncached)
    if read + write + uncached == 0:
        assert r is None
    else:
        assert 0.0 <= r <= 1.0
        assert r == round(read / (read + write + uncached), 3)
        assert (read == 0) <= (r == 0.0)  # no reads, no hits


# ----------------------------------------------------------------------------- call_latency summaries


@st.composite
def strands_calls(draw):
    calls = []
    for i in range(draw(st.integers(1, 5))):
        lines = []
        for _ in range(draw(st.integers(0, 4))):
            lines.append((0.0, {"event": "strands_model_call", "model_latency_s": draw(secs),
                                "model_ttfb_s": draw(secs), "input_tokens": draw(tokens),
                                "output_tokens": draw(tokens), "cache_read_tokens": draw(tokens),
                                "cache_write_tokens": draw(tokens)}))
        for _ in range(draw(st.integers(1, 3))):
            lines.append((0.0, {"event": "turn", "duration_ms": draw(secs) * 1000, "hops": 1,
                                "tools": draw(st.sampled_from([[], ["list_recent_transactions"], ["escalate"]]))}))
        for _ in range(draw(st.integers(0, 2))):
            lines.append((0.0, {"event": "agent_history_reload", "reason": draw(st.sampled_from(["other_owner", "evicted"])),
                                "duration_ms": draw(secs) * 1000, "outcome": "ok"}))
        harness = (draw(st.lists(secs, max_size=3)), [])
        calls.append(cl.build_call(f"CA{i:032d}", 0, cl.parse_bridge(lines), [], {},
                                   harness_times=lambda sid, h=harness: (h[0], [0.0] * len(h[0]))))
    return calls


@PROPS
@given(strands_calls(), st.randoms(use_true_random=False))
def test_call_summary_ignores_call_order(calls, rnd):
    shuffled = list(calls)
    rnd.shuffle(shuffled)
    a, b = cl.summarize(calls), cl.summarize(shuffled)
    assert a == b
    s = a.get("strands")
    if s and s["cache_hit_ratio"] is not None:
        assert 0.0 <= s["cache_hit_ratio"] <= 1.0
    for v in a["turn_s"].values():
        assert v["p50"] <= v["p90"]


# ----------------------------------------------------------------------------- per hosting

times = st.floats(min_value=0.0, max_value=600.0, allow_nan=False, allow_infinity=False)
STREAMS = ("s1", "s2", "s3")


@st.composite
def connect_inputs(draw):
    """Mints, sessions and process start-ups for one call, at arbitrary times."""
    mints = [(t, {"event": "relay_minted", "hosting": draw(st.sampled_from([None, "fargate", "agentcore"]))})
             for t in draw(st.lists(times, max_size=5))]
    sessions = sorted(({"ts": t, "stream": draw(st.sampled_from(STREAMS)), "hosting": None,
                        "history": draw(st.sampled_from([None, "new", "cached", "reload"])),
                        "first_word_ts": t + draw(st.floats(0, 5)),
                        "first_word_source": draw(st.sampled_from(["agent_speaking", "ws_session_start"]))}
                       for t in draw(st.lists(times, max_size=5))), key=lambda s: s["ts"])
    startups = {s: draw(st.lists(times, max_size=3)) for s in STREAMS}
    return mints, sessions, startups


@PROPS
@given(connect_inputs())
def test_pair_connects(inp):
    mints, sessions, startups = inp
    cs = cl.pair_connects(mints, sessions, startups)
    assert len(cs) == len(mints)
    assert [c["position"] for c in cs] == ["first", *["reentry"] * (len(cs) - 1)][:len(cs)]
    paired = [c for c in cs if c["paired"]]
    assert all(c["start"] in ("warm", "cold") for c in paired)
    assert all(c["connect_to_first_word_s"] >= 0 for c in paired)
    # mint windows [mint, next mint) don't overlap, so no session is paired twice
    keys = sorted(mints, key=lambda m: m[0])
    picked = []
    for i, (ts, _) in enumerate(keys):
        nxt = keys[i + 1][0] if i + 1 < len(keys) else float("inf")
        s = next((x for x in sessions if ts <= x["ts"] < nxt), None)
        if s is not None:
            picked.append(id(s))
    assert len(picked) == len(set(picked)) == len(paired) <= len(sessions)


@st.composite
def hosted_calls(draw):
    calls = []
    for i in range(draw(st.integers(0, 8))):
        lines = [(0.0, {"event": "strands_model_call", "model_latency_s": draw(secs), "model_ttfb_s": draw(secs)})]
        for _ in range(draw(st.integers(0, 3))):
            lines.append((0.0, {"event": "turn", "duration_ms": draw(secs) * 1000, "hops": 1,
                                "tools": draw(st.sampled_from([[], ["list_recent_transactions"], ["get_balance"]]))}))
        mints, sessions, startups = draw(connect_inputs())
        connects = cl.pair_connects(mints, sessions, startups)
        prewarm = draw(st.lists(st.fixed_dictionaries({"outcome": st.sampled_from(["ok", "in_progress", "failed"]),
                                                       "duration_ms": secs.map(lambda x: x * 1000)}), max_size=2))
        c = cl.build_call(f"CA{i:032d}", 0, cl.parse_bridge(lines), [], {}, harness_times=lambda sid: ([], []),
                          hosting=draw(st.sampled_from(cl.HOSTINGS)), connects=connects, prewarm=prewarm,
                          dial_to_first_word_s=draw(st.one_of(st.none(), secs)))
        if draw(st.booleans()) and c["hosting"] == "fargate":
            c.pop("hosting")  # a saved call from before the hosting field
        calls.append(c)
    return calls


@PROPS
@given(hosted_calls(), st.randoms(use_true_random=False))
def test_grouping_by_hosting_partitions_the_samples(calls, rnd):
    groups = cl.group_by_hosting(calls)
    ids = sorted(id(c) for cs in groups.values() for c in cs)
    assert ids == sorted(id(c) for c in calls)  # no call lost or duplicated
    s = cl.summarize(calls)
    assert sum(h["calls"] for h in s["hostings"].values()) == s["calls"] == len(calls)
    for kind, v in s["turn_s"].items():
        assert sum(h["turn_s"].get(kind, {"n": 0})["n"] for h in s["hostings"].values()) == v["n"]
    total_paired = sum(1 for c in calls for x in c["connects"] if x["paired"])
    assert sum(h["connects"] for h in s["hostings"].values()) == total_paired
    for h, v in s["hostings"].items():
        ctfw = v["connect_to_first_word_s"]
        assert ctfw["warm"]["n"] + ctfw["cold"]["n"] == v["connects"] == ctfw["all"]["n"]
        assert ctfw["first"]["n"] + ctfw["reentry"]["n"] == v["connects"]
        for part in (*ctfw.values(), *v["turn_s"].values(), v["dial_to_first_word_s"]):
            if part["p50"] is not None:
                assert part["p50"] <= part["p90"]
        if h == "agentcore" and v["prewarm_hit_rate"] is not None:
            assert 0.0 <= v["prewarm_hit_rate"] <= 1.0
    if set(s["hostings"]) == set(cl.HOSTINGS):
        assert "hosting_difference" in s
    shuffled = list(calls)
    rnd.shuffle(shuffled)
    assert cl.summarize(shuffled) == s

