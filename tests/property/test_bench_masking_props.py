"""The benchmark's masking timeline over arbitrary steps.

* phrase delay is never negative, and exactly 0 for an answer step in which no phrase is spoken;
  ``None`` only for a hand-off step;
* the longest silence and the first audio are never negative;
* for a fixed run, the number of tool phrases spoken never grows as ``tool_after_ms`` grows, and
  the number of acknowledgements never grows as ``after_ms`` grows;
* a v2 tool phrase is spoken only before the answer is ready (or for a hand-off), and never before
  the acknowledgement's estimated end.
"""

from __future__ import annotations

from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scripts.bench import agent_bench as ab
from tests.unit.scripts.test_bench_masking import REV22_POLICY, REV22_TEXTS, _catalog

PROPS = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])

secs = st.floats(min_value=0.0, max_value=8.0, allow_nan=False, allow_infinity=False).map(lambda f: round(f, 3))
delay_ms = st.integers(0, 3000)
TOOLS = ["list_recent_transactions", "change_address", "get_balance", "faq_lookup", "escalate"]
HANDOFF = {"get_balance", "escalate"}
MODELS = st.sampled_from(ab.MASKING_MODELS)


@st.composite
def a_step(draw) -> dict[str, Any]:
    """Hops as ``run_scenario`` records them, with ``response_s`` and ``handoff`` computed the same way."""
    hops: list[dict[str, Any]] = []
    handoff = None
    for _ in range(draw(st.integers(1, 4))):
        end = draw(secs)
        text = draw(st.one_of(st.none(), secs.map(lambda f, e=end: round(min(f, e), 3))))
        tools = [] if text is not None and draw(st.booleans()) else draw(st.lists(st.sampled_from(TOOLS), max_size=2))
        call = draw(secs.map(lambda f, e=end: round(min(f, e), 3))) if tools else None
        marks = [t for t in (text, call) if t is not None]
        hops.append({"first_event_s": min(marks) if marks else None, "first_text_s": text, "tool_call_s": call,
                     "end_s": end, "tools": tools})
        hand = [t for t in tools if t in HANDOFF]
        if hand:
            handoff = hand[0]
            break
        if not tools:
            break
    response = 0.0
    for h in hops:
        if h["first_text_s"] is not None:
            response += h["first_text_s"]
            break
        if handoff is not None and h is hops[-1]:
            response += h["tool_call_s"] or h["end_s"]
            break
        response += h["end_s"]
    return {"hops": hops, "handoff": handoff, "response_s": round(response, 3), "timer_eligible": draw(st.booleans())}


runs = st.lists(a_step(), min_size=1, max_size=12).map(lambda steps: [{"scenario": "x", "steps": steps}])


def mk(model: str = "v2", **kw: Any) -> ab.Masking:
    return ab.masking_settings(model, policy=REV22_POLICY, catalog=_catalog(REV22_TEXTS), **kw)


@PROPS
@given(a_step(), MODELS, delay_ms, delay_ms)
def test_phrase_delay_nonnegative_and_zero_without_phrases(s, model, after, tool_after):
    a = ab.step_audio(s, mk(model, after_ms=after, tool_after_ms=tool_after))
    assert (a.phrase_delay_s is None) == (s["handoff"] is not None)
    if a.phrase_delay_s is not None:
        assert a.phrase_delay_s >= 0
        if not a.phrases:
            assert a.phrase_delay_s == 0
    assert a.max_silence_s >= 0 and a.first_audio_s >= 0
    assert a.answer_s >= a.ready_s


@PROPS
@given(a_step(), delay_ms, delay_ms)
def test_tool_phrase_only_while_waiting_and_after_the_ack(s, after, tool_after):
    m = mk(after_ms=after, tool_after_ms=tool_after)
    a = ab.step_audio(s, m)
    ack = [s0 + d for s0, d, k in a.phrases if k == "ack"]
    for s0, _, k in a.phrases:
        if k == "tool":
            assert s["handoff"] is not None or s0 < a.ready_s
            if ack:
                assert s0 >= ack[0] - 1e-9
    assert sum(k == "tool" for _, _, k in a.phrases) <= 1 and sum(k == "ack" for _, _, k in a.phrases) <= 1


@PROPS
@given(runs, delay_ms, delay_ms, delay_ms)
def test_tool_phrases_never_grow_with_tool_after_ms(rs, after, t1, t2):
    lo, hi = sorted((t1, t2))
    a = ab.masking_overall(rs, mk(after_ms=after, tool_after_ms=lo))
    b = ab.masking_overall(rs, mk(after_ms=after, tool_after_ms=hi))
    assert b["tool_phrase_share"] <= a["tool_phrase_share"]
    rows = ab.sweep(rs, {"tool_after_ms": sorted({lo, hi, 0, 3000})}, after_ms=after)
    shares = [r["tool_phrase_share"] for r in rows]
    assert shares == sorted(shares, reverse=True)


@PROPS
@given(runs, MODELS, delay_ms, delay_ms)
def test_acks_never_grow_with_after_ms(rs, model, a1, a2):
    lo, hi = sorted((a1, a2))
    if lo == 0:  # 0 turns the timer off
        lo = 1
        hi = max(hi, 1)
    a = ab.masking_overall(rs, mk(model, after_ms=lo))
    b = ab.masking_overall(rs, mk(model, after_ms=hi))
    assert b["ack_share"] <= a["ack_share"]
    for o in (a, b):
        assert o["max_silence_p50"] >= 0
        if o["phrase_delay_p50"] is not None:
            assert 0 <= o["phrase_delay_p50"] <= o["phrase_delay_p90"]
