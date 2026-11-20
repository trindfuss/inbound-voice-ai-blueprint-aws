"""scripts/bench/agent_bench.py: the masking timeline, offline.

Hand-built steps against fixed settings (a policy and catalog given here, so later tuning of
``data/`` doesn't move these numbers), the legacy model against a saved run's summary, and the
``--resummarize`` / ``--sweep`` command line.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.bench import agent_bench as ab

RESULTS = Path(__file__).resolve().parents[2] / "fixtures" / "bench"
NOCACHE = RESULTS / "strands-nocache-20261015T154853Z.json"
CACHE = RESULTS / "strands-cache-20261015T155407Z.json"

#: The en-US masking phrases and policy values the summaries were computed with.
REV22_TEXTS = {
    "mask.generic.1": "One moment.", "mask.generic.2": "Just a second.", "mask.generic.3": "Bear with me.",
    "mask.generic.4": "Give me a moment.", "mask.generic.5": "Let me take a look.",
    "mask.list_recent_transactions": "I'm pulling up your recent transactions now.",
    "mask.change_address": "I'm updating your address now.",
    "mask.get_balance": "I'm getting your balance now.",
}
REV22_POLICY = {"masking": {"after_ms": 1500, "handoff_wait_ms": 2500, "end_mode": "wait", "chars_per_s": 15},
                "tools": {"get_balance": {"mask": {"phrase_key": "mask.get_balance"}},
                          "list_recent_transactions": {"mask": {"phrase_key": "mask.list_recent_transactions"}},
                          "change_address": {"mask": {"phrase_key": "mask.change_address"}},
                          "faq_lookup": {"mask": None}, "escalate": {"mask": None}}}


def _catalog(texts: dict[str, str]) -> dict[str, Any]:
    return {"entries": [{"key": k, "lang": "en-US", "text": t} for k, t in texts.items()]
            + [{"key": "mask.generic.1", "lang": "es-US", "text": "Un momento, por favor, un momento."}]}


def settings(model: str = "v2", policy: dict[str, Any] | None = None, texts: dict[str, str] | None = None,
             **kw: Any) -> ab.Masking:
    return ab.masking_settings(model, policy=policy or REV22_POLICY, catalog=_catalog(texts or REV22_TEXTS), **kw)


def hop(end: float, *, text: float | None = None, call: float | None = None, tools: tuple[str, ...] = ()) -> dict[str, Any]:
    marks = [t for t in (text, call) if t is not None]
    return {"first_event_s": min(marks) if marks else None, "first_text_s": text, "tool_call_s": call,
            "end_s": end, "tools": list(tools)}


def step(*hops: dict[str, Any], response: float, handoff: str | None = None, timer: bool = True) -> dict[str, Any]:
    return {"hops": list(hops), "response_s": response, "handoff": handoff, "timer_eligible": timer}


def dur(text: str) -> float:
    return 0.3 + len(text) / 15


TXN = "list_recent_transactions"


# ----------------------------------------------------------------------------- settings


def test_v2_uses_the_longest_phrase_of_each_pool():
    texts = {**REV22_TEXTS, f"mask.{TXN}.2": "One moment while I pull up your recent transactions, please."}
    m = settings(texts=texts)
    assert m.ack_s == pytest.approx(dur("Let me take a look."))
    assert m.phrase_s[TXN] == pytest.approx(dur(texts[f"mask.{TXN}.2"]))
    legacy = settings("legacy", texts=texts)  # legacy: the pool's mean, the single phrase_key text
    generic = [t for k, t in REV22_TEXTS.items() if k.startswith("mask.generic.")]
    assert legacy.ack_s == pytest.approx(sum(dur(t) for t in generic) / len(generic))
    assert legacy.phrase_s[TXN] == pytest.approx(dur(REV22_TEXTS[f"mask.{TXN}"]))


def test_delays_per_tool_policy_default_and_handoff():
    policy = json.loads(json.dumps(REV22_POLICY))
    policy["masking"]["tool_after_ms"] = 400
    policy["tools"]["change_address"]["mask"]["after_ms"] = 900
    policy["tools"]["get_balance"]["mask"]["after_ms"] = 900  # a hand-off: ignored, always 0
    m = settings(policy=policy)
    assert m.delay_s == {TXN: 0.4, "change_address": 0.9, "get_balance": 0.0}
    assert m.tool_after_ms == 400 and m.timer_s == 1.5
    m = settings(policy=policy, after_ms=2000, tool_after_ms=300, tool_delays={"change_address": 0})
    assert m.delay_s == {TXN: 0.3, "change_address": 0.0, "get_balance": 0.0} and m.timer_s == 2.0
    assert settings(after_ms=0).timer_s is None
    assert settings().tool_after_ms == ab.DEFAULT_TOOL_AFTER_MS  # absent from the policy


@pytest.mark.parametrize("kw", [{"tool_delays": {"faq_lookup": 300}}, {"tool_delays": {"nope": 1}},
                                {"after_ms": -1}, {"tool_after_ms": 10_001}])
def test_bad_settings_are_refused(kw):
    with pytest.raises(ValueError):
        settings(**kw)


# ----------------------------------------------------------------------------- the v2 timeline


def test_fast_tool_gets_no_tool_phrase():
    m = settings(after_ms=2000)  # tool_after_ms 600
    st = step(hop(0.5, call=0.5, tools=(TXN,)), hop(0.6, text=0.4), response=0.9)
    a = ab.step_audio(st, m)
    assert a.phrases == [] and not a.ack and not a.tool_phrase
    assert a.phrase_delay_s == 0 and a.first_audio_s == 0.9 and a.max_silence_s == 0.9
    # the legacy model speaks it at once, and the answer waits behind it
    old = ab.step_audio(st, settings("legacy", after_ms=2000))
    assert old.tool_phrase and old.phrase_delay_s == pytest.approx(0.5 + dur(REV22_TEXTS[f"mask.{TXN}"]) - 0.9, abs=1e-3)


def test_slow_model_hop_gets_the_ack_then_the_tool_phrase_after_the_ack_ends():
    m = settings(after_ms=2000)
    st = step(hop(2.1, call=2.1, tools=(TXN,)), hop(3.0, text=2.0), response=4.1)
    a = ab.step_audio(st, m)
    ack_end = 2.0 + m.ack_s
    assert ack_end > 2.1 + 0.6  # the acknowledgement's end is the later of the two
    assert [k for _, _, k in a.phrases] == ["ack", "tool"]
    assert a.phrases[0][0] == 2.0
    assert a.phrases[1][0] == pytest.approx(ack_end)
    tool_end = ack_end + m.phrase_s[TXN]
    assert a.answer_s == pytest.approx(max(4.1, tool_end))
    assert a.phrase_delay_s == pytest.approx(max(0.0, tool_end - 4.1), abs=1e-3)
    assert a.first_audio_s == 2.0


def test_tool_phrase_after_its_delay_when_the_ack_ended_earlier():
    m = settings(after_ms=1500, tool_after_ms=600)
    st = step(hop(3.0, call=3.0, tools=(TXN,)), hop(3.0, text=2.5), response=5.5)
    a = ab.step_audio(st, m)
    assert [(k, round(s0, 3)) for s0, _, k in a.phrases] == [("ack", 1.5), ("tool", 3.6)]


def test_handoff_phrase_is_immediate_and_has_no_phrase_delay():
    m = settings(after_ms=2000, tool_after_ms=900)
    st = step(hop(1.2, call=1.0, tools=("get_balance",)), response=1.0, handoff="get_balance")
    a = ab.step_audio(st, m)
    assert a.phrases == [(1.0, m.phrase_s["get_balance"], "tool")]
    assert a.phrase_delay_s is None and a.first_audio_s == 1.0


def test_handoff_phrase_waits_for_a_playing_ack():
    m = settings(after_ms=1500)
    st = step(hop(1.8, call=1.6, tools=("get_balance",)), response=1.6, handoff="get_balance")
    a = ab.step_audio(st, m)
    assert [k for _, _, k in a.phrases] == ["ack", "tool"]
    assert a.phrases[1][0] == pytest.approx(1.5 + m.ack_s)


def test_answer_before_the_deadline_speaks_no_phrase():
    st = step(hop(0.5, call=0.5, tools=(TXN,)), hop(1.0, text=1.0), response=1.5)
    assert ab.step_audio(st, settings(after_ms=2000, tool_delays={TXN: 1200})).phrases == []
    spoken = ab.step_audio(st, settings(after_ms=2000, tool_delays={TXN: 600}))  # deadline 1.1 < 1.5
    assert [(k, round(s0, 3)) for s0, _, k in spoken.phrases] == [("tool", 1.1)]
    assert spoken.phrase_delay_s == pytest.approx(1.1 + spoken_len() - 1.5, abs=1e-3)


def spoken_len() -> float:
    return dur(REV22_TEXTS[f"mask.{TXN}"])


def test_text_in_the_first_hop_stops_the_ack_and_the_tool_phrase():
    m = settings(after_ms=1500)
    st = step(hop(3.0, text=1.0, call=2.9, tools=(TXN,)), response=1.0)
    assert ab.step_audio(st, m).phrases == []


def test_ack_not_on_an_await_caller_resume():
    m = settings(after_ms=1500)
    st = step(hop(3.0, text=3.0), response=3.0, timer=False)
    assert ab.step_audio(st, m).phrases == []
    st["timer_eligible"] = True
    assert ab.step_audio(st, m).ack


def test_masking_overall_counts_shares():
    m = settings(after_ms=2000)
    runs = [{"scenario": "x", "steps": [
        step(hop(3.0, text=3.0), response=3.0),                                   # ack
        step(hop(1.0, call=1.0, tools=(TXN,)), hop(2.0, text=2.0), response=3.0),  # tool phrase at 1.6
        step(hop(1.0, text=1.0), response=1.0),                                   # nothing
        step(hop(1.2, call=1.0, tools=("get_balance",)), response=1.0, handoff="get_balance"),
    ]}, {"scenario": "x", "error": "boom", "steps": []}]
    o = ab.masking_overall(runs, m)
    assert o["steps"] == 4 and o["ack_share"] == 0.25 and o["tool_phrase_share"] == 0.5
    assert o["phrase_delay_p50"] is not None  # over the three answer steps


# ----------------------------------------------------------------------------- regression


@pytest.mark.parametrize("path", [NOCACHE, CACHE])
def test_legacy_model_reproduces_the_saved_summaries(path):
    data = ab.prepare_saved(json.loads(path.read_text()))
    m = settings("legacy")
    assert m.timer_s == data["masking"]["timer_s"]
    s = ab.summarize([r for r in data["raw"] if "error" not in r], m)
    for k, v in data["summary"].items():
        for metric in ("first_audio_p50", "first_audio_p90", "max_silence_p50", "max_silence_p90"):
            assert s[k][metric] == v[metric], (k, metric)


# ----------------------------------------------------------------------------- the command line


def test_sweep_prints_one_row_per_combination(capsys):
    asyncio.run(ab.main(["--resummarize", str(CACHE), "--sweep", "after_ms=1500,2000", "tool_after_ms=0,300,600"]))
    out = capsys.readouterr().out.splitlines()
    head = next(i for i, x in enumerate(out) if x.lstrip().startswith("after_ms"))
    rows = [x.split() for x in out[head + 1:] if x.strip()]
    assert [(r[0], r[1]) for r in rows] == [(a, t) for a in ("1500", "2000") for t in ("0", "300", "600")]
    assert all(r[-2].endswith("%") and r[-1].endswith("%") for r in rows)


def test_sweep_rows_follow_the_grid():
    data = ab.prepare_saved(json.loads(CACHE.read_text()))
    rows = ab.sweep(data["raw"], {"tool_after_ms": [0, 900]}, after_ms=2500)
    assert [(r["after_ms"], r["tool_after_ms"]) for r in rows] == [(2500, 0), (2500, 900)]
    assert rows[0]["tool_phrase_share"] >= rows[1]["tool_phrase_share"]


def test_resummarize_with_candidate_settings(capsys):
    asyncio.run(ab.main(["--resummarize", str(CACHE), "--after-ms", "2000", "--tool-after-ms", "300",
                         "--tool-delay", f"{TXN}=900"]))
    out = capsys.readouterr().out
    assert "v2 model, after_ms 2000, tool_after_ms 300" in out and f"'{TXN}': 900" in out
    assert "phrase delay p50" in out and "all steps: longest silence" in out


@pytest.mark.parametrize("argv", [
    ["--sweep", "after_ms=1500"],                                         # no --resummarize
    ["--resummarize", str(CACHE), "--sweep", "bogus=1"],
    ["--resummarize", str(CACHE), "--sweep", "after_ms=x"],
    ["--resummarize", str(CACHE), "--tool-delay", "faq_lookup=300"],      # not masked
    ["--resummarize", str(CACHE), "--tool-delay", "list_recent_transactions"],
    ["--resummarize", str(CACHE), "--masking-model", "v3"],
])
def test_bad_masking_arguments_are_usage_errors(argv):
    with pytest.raises(SystemExit):
        asyncio.run(ab.main(argv))


def test_result_file_records_masking_and_phrase_delay(tmp_path):
    args = ab.build_parser().parse_args(["--agent", "strands", "--runs", "1"])
    run = {"scenario": "transactions_inline", "steps": [
        step(hop(1.0, call=1.0, tools=(TXN,)), hop(2.0, text=2.0), response=3.0)]}
    path = ab.write_result(tmp_path, label="x", args=args, backend={}, runs=[run], masking=settings(after_ms=2000))
    data = json.loads(path.read_text())
    assert data["masking"]["model"] == "v2" and data["masking"]["tool_delay_ms"][TXN] == 600
    assert data["masking_overall"]["tool_phrase_share"] == 1.0
    assert data["summary"]["transactions_inline#1"]["phrase_delay_p50"] is not None


def test_compare_shows_phrase_delay_and_masking_rows():
    data = ab.prepare_saved(json.loads(CACHE.read_text()))
    m = settings()
    new = {**data, "summary": ab.summarize(data["raw"], m), "masking_overall": ab.masking_overall(data["raw"], m)}
    lines = ab.compare_lines([data, new])
    assert any("phrase_delay_p50" in x and x.startswith("transactions_inline#1") for x in lines)
    row = next(x for x in lines if x.startswith("overall (masking)") and "tool_phrase_share" in x)
    assert row.split()[3] == "-"  # the saved file predates it
