"""scripts/bench/call_latency.py: real-call metrics for Strands calls from the bridge's own timing
lines; older calls from before the harness removal are still
summarized. Offline: CloudWatch-shaped events are built here, and one sample is produced by the real Strands client through the bridge's JSON
formatter, so the field names the script reads are the ones the client writes.

Per hosting: recorded-style logs for a Fargate and an AgentCore call
(fictional SIDs and stream names) are read through a fake CloudWatch Logs client; no AWS call.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest
from botocore.exceptions import ClientError

from owlbank import obs
from scripts.bench import call_latency as cl
from tests.strands_fakes import FakeModel, make_client, text_msg, tool_msg, usage

SID = "CA" + "0" * 31 + "1"


def _ev(ts: float, **m: Any) -> dict[str, Any]:
    return {"timestamp": int(ts * 1000), "message": json.dumps({"call_sid": SID, "level": "INFO", **m})}


#: A recorded-style sample: two sessions, a reload on the second (another task), three turns (no tool,
#: inline tool over two model calls, a hand-off), a line that isn't JSON, and a JSON non-object.
STRANDS_SAMPLE = [
    _ev(100.0, event="ws_session_start"),
    _ev(100.1, event="agent_history_new", reason="new", duration_ms=3.0, outcome="ok", messages=0),
    _ev(103.0, event="strands_model_call", call=1, model_ttfb_s=0.6, model_latency_s=0.9, input_tokens=1200,
        output_tokens=40, cache_read_tokens=0, cache_write_tokens=1000, stop_reason="end_turn"),
    _ev(103.1, event="turn", turn=1, duration_ms=1100.0, hops=1, tools=[], stop_reason="end_turn"),
    _ev(110.0, event="strands_model_call", call=1, model_ttfb_s=0.5, model_latency_s=0.8, input_tokens=200,
        output_tokens=30, cache_read_tokens=1000, cache_write_tokens=0, stop_reason="tool_use"),
    _ev(111.0, event="strands_model_call", call=2, model_ttfb_s=0.7, model_latency_s=1.0, input_tokens=300,
        output_tokens=50, cache_read_tokens=1000, cache_write_tokens=0, stop_reason="end_turn"),
    _ev(111.2, event="turn", turn=2, duration_ms=2200.0, hops=2, tools=["list_recent_transactions"]),
    {"timestamp": 112000, "message": "not json"},
    {"timestamp": 112001, "message": "[1, 2]"},
    _ev(130.0, event="ws_session_start"),
    _ev(130.2, event="agent_history_reload", reason="other_owner", duration_ms=180.0, outcome="ok", messages=8),
    _ev(132.0, event="strands_model_call", call=1, model_ttfb_s=0.4, model_latency_s=0.6, input_tokens=100,
        output_tokens=20, cache_read_tokens=1000, cache_write_tokens=0, stop_reason="tool_use"),
    _ev(132.1, event="turn", turn=3, duration_ms=1000.0, hops=1, tools=["escalate"]),
]


def _no_memory(sid: str):
    pytest.fail("a Strands call must not read AgentCore Memory metrics")


def test_strands_call_metrics_from_the_bridge_lines():
    parsed = cl.parse_bridge(cl._json_lines(STRANDS_SAMPLE))
    c = cl.build_call(SID, 100, parsed, [5.0], {"incoming": [700.0]}, harness_times=_no_memory)
    assert c["backend"] == "strands"
    assert [t["kind"] for t in c["turns"]] == ["no_tool", "inline_tool", "handoff"]
    assert c["segments"] == 2
    assert c["model_calls"] == 4
    assert c["model_latency_s"] == [0.9, 0.8, 1.0, 0.6]
    assert c["model_ttfb_s"] == [0.6, 0.5, 0.7, 0.4]
    # (turn time - model time) / model calls = (4.3 - 3.3) / 4
    assert c["overhead_s"] == pytest.approx(0.25)
    assert (c["cache_read_tokens"], c["cache_write_tokens"], c["input_tokens"]) == (3000, 1000, 1800)
    assert c["cache_hit_ratio"] == round(3000 / (1800 + 3000 + 1000), 3)
    assert c["history_reloads"] == 1 and c["history_reload_reasons"] == {"other_owner": 1}
    assert c["history_reload_ms"] == [180.0] and c["history_new"] == 1
    assert c["harness_overhead_s"] is None


def test_strands_summary():
    parsed = cl.parse_bridge(cl._json_lines(STRANDS_SAMPLE))
    c = cl.build_call(SID, 100, parsed, [5.0], {"incoming": [700.0]}, harness_times=_no_memory)
    s = cl.summarize([c, c])
    assert s["backends"] == ["strands"]
    st = s["strands"]
    assert st["calls"] == 2 and st["model_calls"] == 8
    assert st["model_latency_s"]["n"] == 8 and st["model_latency_s"]["p50"] == cl.pct([0.9, 0.8, 1.0, 0.6] * 2, .5)
    assert st["overhead_s_per_model_call"]["mean"] == pytest.approx(0.25)
    assert st["tokens"]["cache_read_tokens"] == 6000
    assert st["history_reloads"] == {"n": 2, "reasons": {"other_owner": 2}, "errors": 0,
                                     "duration_ms": {"n": 2, "p50": 180.0, "p90": 180.0}}
    assert s["turn_s"]["inline_tool"]["p50"] == 2.2
    assert s["harness_overhead_s_per_model_call"] == {"n": 0, "mean": None}


def test_harness_call_is_unchanged():
    lines = [e for e in STRANDS_SAMPLE if '"strands_model_call"' not in e["message"]
             and "agent_history" not in e["message"]]
    parsed = cl.parse_bridge(cl._json_lines(lines))
    seen = []

    def memory(sid):
        seen.append(sid)
        return [1.0, 1.1, 0.9, 1.2], [0.5, 0.6, 0.4, 0.7]

    c = cl.build_call(SID, 100, parsed, [], {}, harness_times=memory)
    assert seen == [SID]
    assert c["backend"] == "harness"
    assert c["model_calls_match_hops"] is True  # 1 + 2 + 1 hops, 4 model calls
    assert c["harness_overhead_s"] == pytest.approx((4.3 - 4.2) / 4, abs=1e-3)
    s = cl.summarize([c])
    assert "strands" not in s and s["backends"] == ["harness"]
    assert s["model_latency_s"]["n"] == 4 and s["harness_overhead_s_per_model_call"]["n"] == 1


def test_harness_overhead_needs_matching_hops():
    parsed = cl.parse_bridge(cl._json_lines([STRANDS_SAMPLE[3]]))
    c = cl.build_call(SID, 100, parsed, [], {}, harness_times=lambda sid: ([1.0, 1.0], [0.5, 0.5]))
    assert c["model_calls_match_hops"] is False and c["harness_overhead_s"] is None


def test_call_without_agent_turns_is_not_a_harness_call():
    """A call that reached the bridge but had no turn (the caller never spoke) is ``none``, not an
    old harness call, and AgentCore Memory isn't read for it."""
    parsed = cl.parse_bridge(cl._json_lines([STRANDS_SAMPLE[0]]))
    c = cl.build_call(SID, 100, parsed, [], {}, harness_times=_no_memory)
    assert c["backend"] == "none" and c["turns"] == [] and c["model_latency_s"] == []
    s = cl.summarize([c])
    assert s["backends"] == ["none"] and "strands" not in s


def test_lines_written_by_the_real_client_are_read():
    """The Strands client's own ``strands_model_call`` lines, through the bridge's JSON formatter."""
    out: list[str] = []

    class ToList(logging.Handler):
        def emit(self, record):
            out.append(self.format(record))

    handler = ToList()
    handler.addFilter(obs.ContextFilter())
    handler.setFormatter(obs.JsonFormatter())
    logger = logging.getLogger("tac_bridge.agent_client_strands")
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    model = FakeModel([tool_msg(("list_recent_transactions", {"accountType": "checking"}),
                                use=usage(100, 10, cache_read=0, cache_write=500)),
                       text_msg("Here they are.", use=usage(50, 5, cache_read=500, cache_write=0))])
    client = make_client(model)

    async def go():
        from tac_bridge.agent_client import ToolCall, ToolResult

        with obs.call_context(SID, component="bridge"):
            calls = [ev async for ev in client.turn(SID, user_text="transactions?", tool_results=[], system_prompts={"haiku-5-5": "p"})
                     if isinstance(ev, ToolCall)]
            _ = [ev async for ev in client.turn(SID, user_text=None, system_prompts={"haiku-5-5": "p"},
                                                tool_results=[ToolResult(calls[0].tool_use_id, {"data": []})])]
        await client.aclose()

    try:
        asyncio.run(go())
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    events = [{"timestamp": 1000 * i, "message": m} for i, m in enumerate(out)]
    events.append({"timestamp": 9000, "message": json.dumps({"event": "turn", "duration_ms": 3000.0, "hops": 2,
                                                             "tools": ["list_recent_transactions"]})})
    assert all(json.loads(e["message"]).get("call_sid") == SID for e in events[:-1])
    c = cl.build_call(SID, 0, cl.parse_bridge(cl._json_lines(events)), [], {}, harness_times=_no_memory)
    assert c["backend"] == "strands" and c["model_calls"] == 2
    assert all(x >= 0 for x in c["model_latency_s"]) and len(c["model_ttfb_s"]) == 2
    assert (c["cache_read_tokens"], c["cache_write_tokens"], c["input_tokens"]) == (500, 500, 150)
    assert c["overhead_s"] is not None


# ----------------------------------------------------------------------------- per hosting

FG_SID = "CA" + "f" * 30 + "01"
AC_SID = "CA" + "a" * 30 + "02"
RUNTIME_GROUP = "/aws/bedrock-agentcore/runtimes/owlbank_sandbox_bridge-AbCdEf0123-DEFAULT"
T0 = 1_800_000_000  # the segments table's call_start (s)


def _cw(ts: float, stream: str, sid: str | None = None, **m: Any) -> dict[str, Any]:
    body = {"level": "INFO", **m}
    if sid is not None:
        body["call_sid"] = sid
    return {"timestamp": round(ts * 1000), "message": json.dumps(body), "logStreamName": stream}


class FakeLogs:
    """``filter_log_events`` over recorded events per group: a quoted term must appear in the
    message; ``?"a" ?"b"`` matches either; two events per page; an unknown group raises
    ``ResourceNotFoundException`` as CloudWatch does."""

    def __init__(self, groups: dict[str, list[dict[str, Any]]]):
        self.groups = groups
        self.calls: list[tuple[str, str]] = []

    def filter_log_events(self, *, logGroupName, startTime, endTime, filterPattern, nextToken=None):
        self.calls.append((logGroupName, filterPattern))
        if logGroupName not in self.groups:
            raise ClientError({"Error": {"Code": "ResourceNotFoundException", "Message": "x"}}, "FilterLogEvents")
        if filterPattern.startswith("?"):
            terms = [t.strip('?"') for t in filterPattern.split(" ?")]
        else:
            terms = [filterPattern.strip('"')]
        hits = [e for e in self.groups[logGroupName] if startTime <= e["timestamp"] <= endTime
                and any(t in e["message"] for t in terms)]
        i = int(nextToken or 0)
        page: dict[str, Any] = {"events": hits[i:i + 2]}
        if i + 2 < len(hits):
            page["nextToken"] = str(i + 2)
        return page


def _turn(ts, stream, sid, ms, tools=()):
    return _cw(ts, stream, sid, event="turn", duration_ms=ms, hops=1, tools=list(tools))


def _model(ts, stream, sid, ttfb, lat):
    return _cw(ts, stream, sid, event="strands_model_call", model_ttfb_s=ttfb, model_latency_s=lat,
               input_tokens=10, output_tokens=5, cache_read_tokens=0, cache_write_tokens=0)


def _speak(ts, stream, sid):
    return _cw(ts, stream, sid, event="relay_speaker_event", recognised=True, speaker_event="agentSpeaking", value="on")


#: Recorded-style logs (fictional SIDs and stream names). Fargate: one long-running task, two
#: connects (first, then a re-entry after a hand-off). AgentCore: a first connect on a pre-warmed
#: microVM (started before relay_minted), then a re-entry that landed on a new microVM started after
#: its relay_minted (cold, history reload), with no speaker event (first word = session start).
FG_TASK = "tac-bridge/bridge/0123456789abcdef"
VM1, VM2 = "2026/10/20/[runtime-logs]01a666a1", "2026/10/20/[runtime-logs]1995cd9c"
FIXTURE = {
    cl.BRIDGE: [
        _cw(T0 - 3600, FG_TASK, event="agent_settings", prompt_caching=True, thinking=False),
        _cw(T0 + 1.6, FG_TASK, FG_SID, event="ws_session_start"),
        _cw(T0 + 1.7, FG_TASK, FG_SID, event="agent_session_start", history="new", owner_written=True),
        _speak(T0 + 2.4, FG_TASK, FG_SID),
        _model(T0 + 6.0, FG_TASK, FG_SID, 0.6, 0.9),
        _turn(T0 + 6.1, FG_TASK, FG_SID, 1000.0),
        _turn(T0 + 9.0, FG_TASK, FG_SID, 2000.0, ["get_balance"]),
        _cw(T0 + 20.3, FG_TASK, FG_SID, event="ws_session_start"),
        _cw(T0 + 20.4, FG_TASK, FG_SID, event="agent_session_start", history="cached", owner_written=True),
        _speak(T0 + 20.9, FG_TASK, FG_SID),
        _model(T0 + 25.0, FG_TASK, FG_SID, 0.5, 0.7),
        _turn(T0 + 25.1, FG_TASK, FG_SID, 900.0),
    ],
    RUNTIME_GROUP: [
        _cw(T0 + 0.5, VM1, event="agent_settings", prompt_caching=True, thinking=False),
        _cw(T0 + 2.0, VM1, AC_SID, event="ws_session_start", hosting="agentcore"),
        _cw(T0 + 2.1, VM1, AC_SID, event="agent_session_start", history="new", owner_written=True),
        _speak(T0 + 2.9, VM1, AC_SID),
        _model(T0 + 7.0, VM1, AC_SID, 0.7, 1.0),
        _turn(T0 + 7.1, VM1, AC_SID, 1200.0),
        _turn(T0 + 10.0, VM1, AC_SID, 2500.0, ["get_balance"]),
        _cw(T0 + 31.0, VM2, event="agent_settings", prompt_caching=True, thinking=False),
        _cw(T0 + 33.5, VM2, AC_SID, event="ws_session_start", hosting="agentcore"),
        _cw(T0 + 33.6, VM2, AC_SID, event="agent_session_start", history="reload", owner_written=True),
        _cw(T0 + 33.8, VM2, AC_SID, event="agent_history_reload", reason="owner_changed", duration_ms=253.0,
            outcome="ok", messages=4),
        _model(T0 + 37.0, VM2, AC_SID, 0.8, 1.1),
        _turn(T0 + 37.1, VM2, AC_SID, 1300.0),
    ],
    cl.FRONTDOOR: [
        _cw(T0 + 0.0, "fd", FG_SID, event="invocation_start", op="incoming"),
        _cw(T0 + 0.4, "fd", FG_SID, event="relay_minted", hosting="fargate", nonce_carrier="query"),
        _cw(T0 + 0.5, "fd", FG_SID, event="invocation_end", op="incoming", duration_ms=480.0),
        _cw(T0 + 19.0, "fd", FG_SID, event="relay_minted", hosting="fargate", nonce_carrier="query"),
        _cw(T0 + 0.0, "fd", AC_SID, event="invocation_start", op="incoming"),
        _cw(T0 + 0.6, "fd", AC_SID, event="relay_minted", hosting="agentcore", nonce_carrier="parameter"),
        _cw(T0 + 0.7, "fd", AC_SID, event="invocation_end", op="incoming", duration_ms=650.0),
        _cw(T0 + 30.5, "fd", AC_SID, event="relay_minted", hosting="agentcore", nonce_carrier="parameter"),
    ],
    cl.WARMER: [
        _cw(T0 + 0.3, "w", AC_SID, event="runtime_prewarm", duration_ms=1800.0, outcome="ok"),
    ],
}


def _analyze(sid, groups=None, fixture=FIXTURE):
    logs = FakeLogs(fixture)
    return cl.analyze_call(sid, T0, groups or cl.bridge_groups(RUNTIME_GROUP), logs, harness_times=_no_memory), logs


def test_runtime_log_group_from_outputs():
    outs = {"OwlBankData": {"TableCalls": "t"},
            "OwlBankBridge": {"BridgeHosting": "agentcore", "RuntimeLogGroup": RUNTIME_GROUP}}
    assert cl.runtime_log_group(outs) == RUNTIME_GROUP
    assert cl.runtime_log_group({"OwlBankBridge": {"BridgeHosting": "fargate"}}) is None
    assert cl.bridge_groups(None) == [(cl.BRIDGE, "fargate")]
    assert cl.bridge_groups(RUNTIME_GROUP) == [(cl.BRIDGE, "fargate"), (RUNTIME_GROUP, "agentcore")]


def test_fargate_call_connects():
    c, _ = _analyze(FG_SID)
    assert c["hosting"] == "fargate" and c["backend"] == "strands" and c["segments"] == 2
    first, again = c["connects"]
    assert first == {"position": "first", "hosting": "fargate", "paired": True, "start": "warm", "history": "new",
                     "first_word_source": "agent_speaking", "connect_to_first_word_s": 2.0}
    assert again["position"] == "reentry" and again["start"] == "warm" and again["history"] == "cached"
    assert again["connect_to_first_word_s"] == 1.9
    assert c["dial_to_first_word_s"] == 2.4
    assert c["prewarm"] == [] and c["frontdoor_ms"] == {"incoming": [480.0]}


def test_agentcore_call_connects_warm_then_cold():
    c, logs = _analyze(AC_SID)
    assert c["hosting"] == "agentcore" and c["segments"] == 2
    first, again = c["connects"]
    # pre-warmed: VM1 started before relay_minted, so warm
    assert first["start"] == "warm" and first["connect_to_first_word_s"] == 2.3
    # the re-entry waited for VM2's start-up: cold; no speaker event, so the session start counts
    assert again == {"position": "reentry", "hosting": "agentcore", "paired": True, "start": "cold",
                     "history": "reload", "first_word_source": "ws_session_start", "connect_to_first_word_s": 3.0}
    assert c["prewarm"] == [{"outcome": "ok", "duration_ms": 1800.0}]
    assert c["history_reloads"] == 1
    assert (RUNTIME_GROUP, cl.STARTUP_PATTERN) in logs.calls


def test_summary_per_hosting_and_difference():
    calls = [_analyze(FG_SID)[0], _analyze(AC_SID)[0]]
    s = cl.summarize(calls)
    assert s["calls"] == 2 and sorted(s["hostings"]) == ["agentcore", "fargate"]
    fg, ac = s["hostings"]["fargate"], s["hostings"]["agentcore"]
    assert fg["calls"] == ac["calls"] == 1
    assert fg["connects"] == 2 and fg["unpaired_connects"] == 0
    assert fg["connect_to_first_word_s"]["warm"] == {"n": 2, "p50": 1.95, "p90": 1.99}
    assert fg["connect_to_first_word_s"]["cold"] == {"n": 0, "p50": None, "p90": None}
    assert "prewarm_hit_rate" not in fg and "cold_start_s" not in fg
    assert ac["connect_to_first_word_s"]["warm"]["n"] == 1 and ac["connect_to_first_word_s"]["cold"]["n"] == 1
    assert ac["session_history"] == {"new": 1, "reload": 1}
    assert ac["first_word_source"] == {"agent_speaking": 1, "ws_session_start": 1}
    assert ac["prewarm_hit_rate"] == 1.0
    assert ac["prewarm"] == {"outcomes": {"ok": 1}, "duration_s": {"n": 1, "p50": 1.8, "p90": 1.8}}
    assert ac["cold_start_s"]["connect_cold_minus_warm_p50"] == 0.7
    assert ac["turn_s"]["handoff"]["p50"] == 2.5 and fg["turn_s"]["handoff"]["p50"] == 2.0
    d = s["hosting_difference"]
    assert d["turn_s"]["handoff"] == {"p50": 0.5, "p90": 0.5}
    assert d["connect_to_first_word_s"]["first"] == {"p50": 0.3, "p90": 0.3}
    assert d["connect_to_first_word_s"]["reentry"] == {"p50": 1.1, "p90": 1.1}
    assert d["connect_to_first_word_s"]["cold"] == {"p50": None, "p90": None}
    assert d["dial_to_first_word_s"] == {"p50": 0.5, "p90": 0.5}
    json.dumps(s)  # the saved file is JSON


def test_one_hosting_has_no_difference():
    s = cl.summarize([_analyze(FG_SID)[0]])
    assert list(s["hostings"]) == ["fargate"] and "hosting_difference" not in s


def test_missing_runtime_group_and_no_relay_minted():
    """A Runtime group that doesn't exist (not deployed) is skipped; a call with no ``relay_minted``
    (from before the AgentCore hosting option) has no connects and is a Fargate call."""
    fixture = {k: v for k, v in FIXTURE.items() if k != RUNTIME_GROUP}
    fixture[cl.FRONTDOOR] = [e for e in FIXTURE[cl.FRONTDOOR] if "relay_minted" not in e["message"]]
    c, logs = _analyze(FG_SID, fixture=fixture)
    assert (RUNTIME_GROUP, f'"{FG_SID}"') in logs.calls
    assert c["hosting"] == "fargate" and c["connects"] == [] and c["dial_to_first_word_s"] is None
    assert cl.summarize([c])["hostings"]["fargate"]["connects"] == 0


def test_hosting_from_the_bridge_line_and_the_group():
    """No ``hosting`` on the mint: the session line's field, else the group the lines came from."""
    fixture = dict(FIXTURE)
    fixture[cl.FRONTDOOR] = [_cw(T0 + 0.6, "fd", AC_SID, event="relay_minted", nonce_carrier="parameter")]
    assert _analyze(AC_SID, fixture=fixture)[0]["hosting"] == "agentcore"
    fixture[RUNTIME_GROUP] = [_cw(T0 + 2.0, VM1, AC_SID, event="ws_session_start"),
                              _model(T0 + 4.0, VM1, AC_SID, 0.5, 0.8)]
    assert _analyze(AC_SID, fixture=fixture)[0]["hosting"] == "agentcore"
    fixture[cl.FRONTDOOR] = []
    assert _analyze(AC_SID, fixture=fixture)[0]["hosting"] == "agentcore"


def test_unpaired_connect():
    """A mint whose connect never reached the bridge: counted, no time, and the earlier mints still pair."""
    fixture = dict(FIXTURE)
    fixture[cl.FRONTDOOR] = [*FIXTURE[cl.FRONTDOOR],
                             _cw(T0 + 50.0, "fd", FG_SID, event="relay_minted", hosting="fargate")]
    c, _ = _analyze(FG_SID, fixture=fixture)
    assert [x["paired"] for x in c["connects"]] == [True, True, False]
    assert "connect_to_first_word_s" not in c["connects"][2]
    assert cl.summarize([c])["hostings"]["fargate"]["unpaired_connects"] == 1


def test_other_client_errors_are_raised():
    class Denied:
        def filter_log_events(self, **kw):
            raise ClientError({"Error": {"Code": "AccessDeniedException", "Message": "x"}}, "FilterLogEvents")

    with pytest.raises(ClientError):
        cl.log_events(cl.BRIDGE, 0, 1, '"x"', Denied())


def test_saved_calls_without_hosting_fields_still_summarize():
    """Result files saved before have no ``hosting``/``connects``: they count as Fargate."""
    parsed = cl.parse_bridge(cl._json_lines(STRANDS_SAMPLE))
    c = cl.build_call(SID, 100, parsed, [5.0], {"incoming": [700.0]}, harness_times=_no_memory)
    for k in ("hosting", "connects", "prewarm", "dial_to_first_word_s"):
        c.pop(k)
    s = cl.summarize([c])
    assert s["hostings"]["fargate"]["calls"] == 1 and s["hostings"]["fargate"]["connects"] == 0
