"""Per-turn latency of real calls, from the deployed system's own logs.

The end-to-end companion to ``agent_bench.py``: for each call since ``--since`` it reads the bridge's
``turn``, ``ws_session_*`` and ``strands_model_call`` lines and TAC's conversation lines (CloudWatch),
and the front door's ``invocation_end`` lines. Nothing here reads what the caller or the AI said.

Reported, as p50/p90 over the calls:

* ``turn_s`` by kind (no tool, inline tool, handoff): the bridge's time from the caller's prompt to the
  end of the agent's reply (``turn`` ``duration_ms``).
* ``model_latency_s`` and ``model_ttfb_s`` per model call, from the bridge's ``strands_model_call``
  lines.
* ``overhead_s`` per model call: (sum of turn time - sum of model latency) / model calls. Model calls
  are counted from the lines, since a turn can run several (a Gateway tool runs inside the loop).
* ``cache_read_tokens``, ``cache_write_tokens`` and ``cache_hit_ratio`` (reads / (uncached input +
  reads + writes)), and ``agent_history_reload`` count, reasons and durations.
* ``conversation_wait_s``: ``ws_session_start`` to TAC's "Started VOICE conversation" (Orchestrator
  creating the call's conversation; with ``reuseConversation`` only the first segment pays it).
* ``frontdoor_ms`` by route.

**Per hosting.** Every call gets a ``hosting`` (``fargate`` or
``agentcore``): the front door's ``relay_minted`` line's ``hosting`` field, else the bridge's
``ws_session_start`` line's, else the log group the bridge lines came from (``/owlbank/ecs/tac-bridge``
is ``fargate``; the Runtime log group, read from the ``RuntimeLogGroup`` stack output or
``--runtime-log-group``, is ``agentcore``), else ``fargate`` (calls from before the AgentCore hosting option). The summary's
``hostings`` section repeats the whole summary for each hosting's calls, and adds:

* ``connect_to_first_word_s``: each connect's time from the front door's ``relay_minted`` to the first
  ``relay_speaker_event`` ``agentSpeaking on`` of the bridge session it opened (the next
  ``ws_session_start`` after it), or to that ``ws_session_start`` when the session logged no speaker
  event (the greeting starts at ``setup``); ``first_word_source`` counts which. Split ``warm``/``cold``:
  cold when the process that served the session (its log stream) logged a start-up line
  (``startup_complete`` or ``agent_settings``) after ``relay_minted`` and no later than the session
  start, i.e. the connect waited for a new process; else warm. Also split ``first`` (a call's first
  connect) and ``reentry`` (later connects), with the bridge's ``agent_session_start`` ``history``
  (``new``, ``cached``, ``reload``) counted per connect.
* ``dial_to_first_word_s``: the front door's first line for the call to its first connect's first word.
* ``prewarm`` (``agentcore``): the warmer's ``runtime_prewarm`` outcomes and ``duration_ms`` (a cold
  boot as the warmer saw it), and ``prewarm_hit_rate``: the share of first connects that were warm.
* ``cold_start_s``: the warmer's ``runtime_prewarm`` ``ok`` durations, and the cold-minus-warm p50 of
  connect-to-first-word.

``hosting_difference`` is ``agentcore`` minus ``fargate`` for each p50 and p90 that both hostings
have (positive: ``agentcore`` is slower). No pass/fail bar: you decide.

**Older calls on the AgentCore harness.** The harness was removed; the bridge runs Strands
only. A call with no ``strands_model_call`` lines is taken to be a call from before the removal
(``backend: harness``): its per-model-call times are read from the harness session in AgentCore
Memory (``timeToFirstByteMs``, ``latencyMs``) and its overhead is reported as ``harness_overhead_s``,
so the saved harness measurements can still be summarized while those logs and events exist.

Usage::

    uv run python scripts/bench/call_latency.py --since 2026-10-16T00:00:00Z --label strands
    uv run python scripts/bench/call_latency.py --since 2026-10-20T00:00:00Z --label hostings \\
        --runtime-log-group /aws/bedrock-agentcore/runtimes/<runtime id>-DEFAULT
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[2]
BRIDGE = "/owlbank/ecs/tac-bridge"
FRONTDOOR = "/owlbank/lambda/frontdoor"
WARMER = "/owlbank/lambda/runtime_warmer"
HOSTINGS = ("fargate", "agentcore")
#: The bridge process's start-up lines (no CallSid): ``startup_complete`` (if logged)
#: or ``agent_settings`` (logged once per process before it listens).
STARTUP_EVENTS = ("startup_complete", "agent_settings")
STARTUP_PATTERN = " ".join(f'?"{e}"' for e in STARTUP_EVENTS)
HANDOFF = {"get_balance", "authenticate", "make_payment", "transfer_funds", "escalate", "end_call"}


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def outputs() -> dict[str, Any]:
    return json.loads((ROOT / "infra" / "cdk-outputs.json").read_text())


def runtime_log_group(outs: dict[str, Any]) -> str | None:
    """The ``RuntimeLogGroup`` output of any stack (the Bridge stack in ``agentcore``), else ``None``."""
    for stack in outs.values():
        if isinstance(stack, dict) and isinstance(stack.get("RuntimeLogGroup"), str):
            return str(stack["RuntimeLogGroup"])
    return None


def bridge_groups(runtime_group: str | None) -> list[tuple[str, str]]:
    """The bridge log groups to read, each with the hosting its lines come from."""
    groups = [(BRIDGE, "fargate")]
    if runtime_group:
        groups.append((runtime_group, "agentcore"))
    return groups


def table_name(prefix: str) -> str:
    names = boto3.client("dynamodb").list_tables()["TableNames"]
    return next(n for n in names if n.startswith(prefix))


def memory_id() -> str:
    for k, v in outputs().get("OwlBankAgent", {}).items():
        if "MemoryId" in k:
            return str(v)
    raise SystemExit("no MemoryId in infra/cdk-outputs.json")


def calls_since(since: datetime) -> list[tuple[str, int]]:
    t = boto3.resource("dynamodb").Table(table_name("OwlBankData-SegmentsTable"))
    items, kw = [], {"FilterExpression": Attr("type").eq("call_start") & Attr("ts").gte(int(since.timestamp()))}
    while True:
        r = t.scan(**kw)
        items += r["Items"]
        if "LastEvaluatedKey" not in r:
            break
        kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]
    return sorted(((i["call_sid"], int(i["ts"])) for i in items), key=lambda x: x[1])


def log_events(group: str, start_ms: int, end_ms: int, pattern: str, logs: Any = None) -> list[dict[str, Any]]:
    """Matching events of one log group; none if the group doesn't exist (a hosting not deployed)."""
    logs = logs if logs is not None else boto3.client("logs")
    out: list[dict[str, Any]] = []
    kw = {"logGroupName": group, "startTime": start_ms, "endTime": end_ms, "filterPattern": pattern}
    while True:
        try:
            r = logs.filter_log_events(**kw)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                return out
            raise
        out += r["events"]
        if "nextToken" not in r:
            break
        kw["nextToken"] = r["nextToken"]
    return out


def _json_lines(events: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    out = []
    for e in events:
        try:
            m = json.loads(e["message"])
        except (ValueError, TypeError):
            continue
        if isinstance(m, dict):
            out.append((e["timestamp"] / 1000, m))
    return out


def _stream_lines(events: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any], str]]:
    """Like :func:`_json_lines`, with each event's log stream (a bridge process: a Fargate task or a
    Runtime microVM)."""
    out = []
    for e in events:
        try:
            m = json.loads(e["message"])
        except (ValueError, TypeError):
            continue
        if isinstance(m, dict):
            out.append((e["timestamp"] / 1000, m, str(e.get("logStreamName") or "")))
    return out


def parse_sessions(lines: list[tuple[float, dict[str, Any], str]], default_hosting: str | None = None,
                   ) -> list[dict[str, Any]]:
    """The bridge sessions of one call, in time order, from its lines ``(ts, json, stream)``: each
    ``ws_session_start`` with its stream, its line's ``hosting`` (else ``default_hosting``), the
    session's ``agent_session_start`` ``history``, and its first word: the first ``agentSpeaking on``
    speaker event before the next session start (same stream), else the session start itself."""
    lines = sorted(lines, key=lambda x: x[0])
    starts = [(ts, m, st) for ts, m, st in lines if m.get("event") == "ws_session_start"]
    out = []
    for i, (ts, m, stream) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else float("inf")
        within = [(t, x) for t, x, s in lines if ts <= t < end and (not stream or not s or s == stream)]
        spoke = next((t for t, x in within if x.get("event") == "relay_speaker_event"
                      and x.get("speaker_event") == "agentSpeaking" and x.get("value") == "on"), None)
        history = next((x.get("history") for _, x in within if x.get("event") == "agent_session_start"), None)
        out.append({"ts": ts, "stream": stream, "hosting": m.get("hosting") or default_hosting,
                    "history": history if isinstance(history, str) else None,
                    "first_word_ts": spoke if spoke is not None else ts,
                    "first_word_source": "agent_speaking" if spoke is not None else "ws_session_start"})
    return out


def pair_connects(minted: list[tuple[float, dict[str, Any]]], sessions: list[dict[str, Any]],
                  startups: dict[str, list[float]]) -> list[dict[str, Any]]:
    """One record per front-door ``relay_minted`` (a connect), in time order: the session it opened
    (the first session start after it and before the next mint), its ``connect_to_first_word_s``,
    ``start`` ``warm`` or ``cold`` (cold: the session's process logged a start-up line in
    ``(minted, session start]``), ``position`` (``first`` or ``reentry``) and ``history``. A mint with
    no session is ``paired: False`` (the connect never reached the bridge) and has no time."""
    mints = sorted(((ts, m) for ts, m in minted if m.get("event") == "relay_minted"), key=lambda x: x[0])
    out = []
    for i, (ts, m) in enumerate(mints):
        nxt = mints[i + 1][0] if i + 1 < len(mints) else float("inf")
        s = next((x for x in sessions if ts <= x["ts"] < nxt), None)
        rec: dict[str, Any] = {"position": "first" if i == 0 else "reentry",
                               "hosting": m.get("hosting") or (s or {}).get("hosting"), "paired": s is not None}
        if s is not None:
            cold = any(ts < t <= s["ts"] for t in startups.get(s["stream"], []))
            rec.update({"start": "cold" if cold else "warm", "history": s["history"],
                        "first_word_source": s["first_word_source"],
                        "connect_to_first_word_s": round(s["first_word_ts"] - ts, 3)})
        out.append(rec)
    return out


def call_hosting(connects: list[dict[str, Any]], sessions: list[dict[str, Any]]) -> str:
    """The call's hosting: its first connect's, else its first session's, else ``fargate``."""
    for x in [*connects, *sessions]:
        if x.get("hosting") in HOSTINGS:
            return str(x["hosting"])
    return "fargate"


def parse_bridge(lines: list[tuple[float, dict[str, Any]]]) -> dict[str, Any]:
    """The bridge's lines for one call, as ``(timestamp_s, json)``: turns, session starts, and the
    Strands client's timing lines (``strands_model_call``, ``agent_history_reload``/``_new``)."""
    turns, sessions, model_calls, reloads, news = [], [], [], [], []
    for ts, m in lines:
        ev = m.get("event")
        if ev == "turn":
            tools = m.get("tools") or []
            kind = "handoff" if any(t in HANDOFF for t in tools) else ("inline_tool" if tools else "no_tool")
            turns.append({"kind": kind, "s": m["duration_ms"] / 1000, "hops": m.get("hops", 1)})
        elif ev == "ws_session_start":
            sessions.append(ts)
        elif ev == "strands_model_call":
            model_calls.append(m)
        elif ev == "agent_history_reload":
            reloads.append(m)
        elif ev == "agent_history_new":
            news.append(m)
    return {"turns": turns, "sessions": sessions, "model_calls": model_calls, "reloads": reloads, "news": news}


def strands_metrics(parsed: dict[str, Any]) -> dict[str, Any]:
    """Per-model-call time and cache use of a Strands call, from its ``strands_model_call`` lines."""
    mc = parsed["model_calls"]
    lat = [c["model_latency_s"] for c in mc if c.get("model_latency_s") is not None]
    ttfb = [c["model_ttfb_s"] for c in mc if c.get("model_ttfb_s") is not None]
    turn_s = sum(t["s"] for t in parsed["turns"])
    overhead = (turn_s - sum(lat)) / len(lat) if lat and len(lat) == len(mc) and parsed["turns"] else None
    tok = {k: sum(c.get(k) or 0 for c in mc) for k in ("input_tokens", "output_tokens",
                                                       "cache_read_tokens", "cache_write_tokens")}
    prompt = tok["input_tokens"] + tok["cache_read_tokens"] + tok["cache_write_tokens"]
    reasons: dict[str, int] = {}
    for r in parsed["reloads"]:
        reasons[str(r.get("reason"))] = reasons.get(str(r.get("reason")), 0) + 1
    return {"model_latency_s": lat, "model_ttfb_s": ttfb, "model_calls": len(mc),
            "overhead_s": round(overhead, 3) if overhead is not None else None, **tok,
            "cache_hit_ratio": round(tok["cache_read_tokens"] / prompt, 3) if prompt else None,
            "history_reloads": len(parsed["reloads"]), "history_reload_reasons": reasons,
            "history_reload_ms": [r["duration_ms"] for r in parsed["reloads"] if r.get("duration_ms") is not None],
            "history_reload_errors": sum(1 for r in parsed["reloads"] if r.get("outcome") == "error"),
            "history_new": len(parsed["news"])}


def harness_model_times(sid: str) -> tuple[list[float], list[float]]:
    """Per-model-call latency and time to first byte of a call from before the harness removal,
    read from its harness session in AgentCore Memory."""
    lat, ttfb = [], []
    mem = boto3.client("bedrock-agentcore")
    kw = {"memoryId": memory_id(), "sessionId": sid, "actorId": sid, "includePayloads": True, "maxResults": 100}
    try:
        while True:
            r = mem.list_events(**kw)
            for ev in r["events"]:
                for p in ev.get("payload", []):
                    c = p.get("conversational")
                    if c and c["role"] == "ASSISTANT":
                        metrics = json.loads(c["content"]["text"])["message"].get("metadata", {}).get("metrics", {})
                        if "latencyMs" in metrics:
                            lat.append(metrics["latencyMs"] / 1000)
                            ttfb.append(metrics.get("timeToFirstByteMs", 0) / 1000)
            if not r.get("nextToken"):
                break
            kw["nextToken"] = r["nextToken"]
    except Exception as exc:  # noqa: BLE001 - memory may have expired; the rest still counts
        print(f"{sid}: no memory events ({type(exc).__name__})")
    return lat, ttfb


def build_call(sid: str, t0: int, parsed: dict[str, Any], conv_waits: list[float], fd: dict[str, list[float]],
               harness_times: Callable[[str], tuple[list[float], list[float]]] = harness_model_times, *,
               hosting: str = "fargate", connects: list[dict[str, Any]] | None = None,
               prewarm: list[dict[str, Any]] | None = None, dial_to_first_word_s: float | None = None,
               ) -> dict[str, Any]:
    """One call's record. Strands if the bridge logged ``strands_model_call`` lines, ``none`` if the
    call had no agent turn, else a call from before the harness removal (model times from AgentCore
    Memory). ``hosting``, ``connects``
    (:func:`pair_connects`), ``prewarm`` (the warmer's ``runtime_prewarm*`` lines: ``outcome``,
    ``duration_ms``) and ``dial_to_first_word_s`` are the per-hosting fields."""
    turns = parsed["turns"]
    base = {"call_sid": sid, "at": datetime.fromtimestamp(t0, UTC).isoformat(), "hosting": hosting,
            "turns": turns, "segments": len(parsed["sessions"]), "conversation_wait_s": conv_waits,
            "frontdoor_ms": fd, "connects": connects or [], "prewarm": prewarm or [],
            "dial_to_first_word_s": dial_to_first_word_s}
    if parsed["model_calls"]:
        sm = strands_metrics(parsed)
        return {**base, "backend": "strands", **sm, "harness_overhead_s": None,
                "model_calls_match_hops": None}
    if not turns:
        # No agent turn at all (the caller hung up or never spoke): nothing to time, and it is not a
        # harness call, so AgentCore Memory isn't read.
        return {**base, "backend": "none", "model_latency_s": [], "model_ttfb_s": [],
                "harness_overhead_s": None, "model_calls_match_hops": None}
    lat, ttfb = harness_times(sid)
    hops = sum(t["hops"] for t in turns)
    overhead = (sum(t["s"] for t in turns) - sum(lat)) / len(lat) if lat and hops == len(lat) else None
    return {**base, "backend": "harness", "model_latency_s": lat, "model_ttfb_s": ttfb,
            "harness_overhead_s": round(overhead, 3) if overhead is not None else None,
            "model_calls_match_hops": hops == len(lat)}


def analyze_call(sid: str, t0: int, groups: list[tuple[str, str]] | None = None, logs: Any = None,
                 harness_times: Callable[[str], tuple[list[float], list[float]]] = harness_model_times,
                 ) -> dict[str, Any]:
    """One call from its logs: the bridge lines from each of ``groups`` (``(log group, hosting)``,
    default the Fargate bridge only), and the front door's and the warmer's lines."""
    groups = groups or bridge_groups(None)
    start, end = (t0 - 5) * 1000, (t0 + 1800) * 1000
    bridge_events: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    startups: dict[str, list[float]] = {}
    tac: list[dict[str, Any]] = []
    for group, group_hosting in groups:
        evs = log_events(group, start, end, f'"{sid}"', logs)
        if not evs:
            continue
        bridge_events += evs
        sessions += parse_sessions(_stream_lines(evs), group_hosting)
        for ts, m, stream in _stream_lines(log_events(group, start, end, STARTUP_PATTERN, logs)):
            if m.get("event") in STARTUP_EVENTS:
                startups.setdefault(stream, []).append(ts)
        tac += log_events(group, start, end, '"Started VOICE conversation"', logs)
    sessions.sort(key=lambda x: x["ts"])
    parsed = parse_bridge(_json_lines(sorted(bridge_events, key=lambda e: e["timestamp"])))
    conv_waits = []
    # TAC's line carries no CallSid: count it for this call only if it follows one of this call's
    # session starts within 60 s (one bridge task in the sandbox; with several, calls can overlap).
    for e in tac:
        ts = e["timestamp"] / 1000
        prior = [s for s in parsed["sessions"] if s <= ts]
        if prior and ts - prior[-1] < 60:
            conv_waits.append(round(ts - prior[-1], 3))
    fd: dict[str, list[float]] = {}
    fd_lines = _json_lines(log_events(FRONTDOOR, start, end, f'"{sid}"', logs))
    for _, m in fd_lines:
        if m.get("event") == "invocation_end" and m.get("op"):
            fd.setdefault(m["op"], []).append(m["duration_ms"])
    connects = pair_connects(fd_lines, sessions, startups)
    prewarm = [{"outcome": m.get("outcome"), "duration_ms": m.get("duration_ms")}
               for _, m in _json_lines(log_events(WARMER, start, end, f'"{sid}"', logs))
               if m.get("event") in ("runtime_prewarm", "runtime_prewarm_failed")]
    dial = None
    if connects and connects[0]["paired"]:
        minted0 = min(ts for ts, m in fd_lines if m.get("event") == "relay_minted")
        dial = round(minted0 + connects[0]["connect_to_first_word_s"] - min(ts for ts, _ in fd_lines), 3)
    return build_call(sid, t0, parsed, conv_waits, fd, harness_times, hosting=call_hosting(connects, sessions),
                      connects=connects, prewarm=prewarm, dial_to_first_word_s=dial)


def _p(xs: list[float]) -> dict[str, Any]:
    return {"n": len(xs), "p50": pct(xs, .5), "p90": pct(xs, .9)}


def summarize(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """The summary of all calls, plus ``hostings`` (the same summary and the connect metrics for each
    hosting's calls) and ``hosting_difference`` (agentcore minus fargate, when both have calls)."""
    out = _summarize_calls(calls)
    groups = group_by_hosting(calls)
    out["hostings"] = {h: {**_summarize_calls(cs), **connect_summary(cs, h)} for h, cs in groups.items()}
    if all(h in groups for h in HOSTINGS):
        out["hosting_difference"] = hosting_difference(out["hostings"]["fargate"], out["hostings"]["agentcore"])
    return out


def group_by_hosting(calls: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """The calls by ``hosting`` (``fargate`` when absent: saved calls from before the AgentCore hosting option), sorted by key."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in calls:
        groups.setdefault(str(c.get("hosting") or "fargate"), []).append(c)
    return dict(sorted(groups.items()))


def connect_summary(calls: list[dict[str, Any]], hosting: str) -> dict[str, Any]:
    """Connect-to-first-word (all, warm, cold, first, re-entry), dial to first word, and for
    ``agentcore`` the pre-warm outcomes, hit rate and cold-start time, over ``calls``."""
    cs = [x for c in calls for x in c.get("connects", [])]
    paired = [x for x in cs if x.get("paired")]

    def ctfw(key: str | None = None, value: str | None = None) -> list[float]:
        return [x["connect_to_first_word_s"] for x in paired if key is None or x.get(key) == value]

    def counts(key: str) -> dict[str, int]:
        return dict(sorted(collections.Counter(str(x.get(key)) for x in paired).items()))

    warm, cold = ctfw("start", "warm"), ctfw("start", "cold")
    out: dict[str, Any] = {
        "connects": len(paired), "unpaired_connects": len(cs) - len(paired),
        "connect_to_first_word_s": {"all": _p(ctfw()), "warm": _p(warm), "cold": _p(cold),
                                    "first": _p(ctfw("position", "first")),
                                    "reentry": _p(ctfw("position", "reentry"))},
        "first_word_source": counts("first_word_source"), "session_history": counts("history"),
        "dial_to_first_word_s": _p([c["dial_to_first_word_s"] for c in calls
                                    if c.get("dial_to_first_word_s") is not None]),
    }
    if hosting == "agentcore":
        pw = [p for c in calls for p in c.get("prewarm", [])]
        ok_s = [p["duration_ms"] / 1000 for p in pw if p.get("outcome") == "ok" and p.get("duration_ms") is not None]
        firsts = [x for x in paired if x["position"] == "first"]
        out["prewarm"] = {"outcomes": dict(sorted(collections.Counter(str(p.get("outcome")) for p in pw).items())),
                          "duration_s": _p(ok_s)}
        out["prewarm_hit_rate"] = (round(sum(x["start"] == "warm" for x in firsts) / len(firsts), 3)
                                   if firsts else None)
        w50, c50 = pct(warm, .5), pct(cold, .5)
        out["cold_start_s"] = {"warmer_prewarm_ok": _p(ok_s),
                               "connect_cold_minus_warm_p50": (round(c50 - w50, 3)
                                                               if w50 is not None and c50 is not None else None)}
    return out


def _diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """``b - a`` over the p50/p90 leaves both summaries have, recursively (``None`` where either has none)."""
    out: dict[str, Any] = {}
    for k in sorted(set(a) & set(b)):
        if k in ("p50", "p90"):
            out[k] = round(b[k] - a[k], 3) if a[k] is not None and b[k] is not None else None
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            d = _diff(a[k], b[k])
            if d:
                out[k] = d
    return out


def hosting_difference(fargate: dict[str, Any], agentcore: dict[str, Any]) -> dict[str, Any]:
    """``agentcore`` minus ``fargate`` for per-turn time, connect-to-first-word, dial to first word and
    model time (positive: ``agentcore`` is slower)."""
    keys = ("turn_s", "connect_to_first_word_s", "dial_to_first_word_s", "model_latency_s", "model_ttfb_s")
    return {k: _diff(fargate.get(k) or {}, agentcore.get(k) or {}) for k in keys}


def _summarize_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[float]] = {}
    for c in calls:
        for t in c["turns"]:
            by_kind.setdefault(t["kind"], []).append(t["s"])
    allv = lambda key, cs=calls: [x for c in cs for x in c.get(key, [])]  # noqa: E731
    fd: dict[str, list[float]] = {}
    for c in calls:
        for op, xs in c["frontdoor_ms"].items():
            fd.setdefault(op, []).extend(xs)
    over = [c["harness_overhead_s"] for c in calls if c.get("harness_overhead_s") is not None]
    out = {
        "calls": len(calls),
        "backends": sorted({c.get("backend", "harness") for c in calls}),
        "turn_s": {k: _p(v) for k, v in by_kind.items()},
        "model_latency_s": _p(allv("model_latency_s")),
        "model_ttfb_s": {"p50": pct(allv("model_ttfb_s"), .5), "p90": pct(allv("model_ttfb_s"), .9)},
        "harness_overhead_s_per_model_call": {"n": len(over), "mean": round(statistics.mean(over), 3) if over else None},
        "conversation_wait_s": {"n": len(allv("conversation_wait_s")), "p50": pct(allv("conversation_wait_s"), .5), "max": max(allv("conversation_wait_s"), default=None)},
        "frontdoor_ms": {k: _p(v) for k, v in fd.items()},
    }
    strands = [c for c in calls if c.get("backend") == "strands"]
    if strands:
        so = [c["overhead_s"] for c in strands if c.get("overhead_s") is not None]
        tok = {k: sum(c.get(k, 0) for c in strands) for k in ("input_tokens", "output_tokens",
                                                             "cache_read_tokens", "cache_write_tokens")}
        prompt = tok["input_tokens"] + tok["cache_read_tokens"] + tok["cache_write_tokens"]
        reasons: dict[str, int] = {}
        for c in strands:
            for r, n in c.get("history_reload_reasons", {}).items():
                reasons[r] = reasons.get(r, 0) + n
        out["strands"] = {
            "calls": len(strands),
            "model_calls": sum(c["model_calls"] for c in strands),
            "model_latency_s": _p(allv("model_latency_s", strands)),
            "model_ttfb_s": _p(allv("model_ttfb_s", strands)),
            "overhead_s_per_model_call": {"n": len(so), "mean": round(statistics.mean(so), 3) if so else None,
                                          "p50": pct(so, .5)},
            "tokens": tok, "cache_hit_ratio": round(tok["cache_read_tokens"] / prompt, 3) if prompt else None,
            "history_reloads": {"n": sum(c["history_reloads"] for c in strands), "reasons": reasons,
                                "errors": sum(c["history_reload_errors"] for c in strands),
                                "duration_ms": _p(allv("history_reload_ms", strands))},
            "history_new": sum(c["history_new"] for c in strands),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--since", required=True, help="ISO time, e.g. 2026-10-14T03:45:00Z")
    ap.add_argument("--label", default="calls")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results"))
    ap.add_argument("--runtime-log-group", help="the Runtime's log group (default: the RuntimeLogGroup "
                    "output in infra/cdk-outputs.json, if any)")
    args = ap.parse_args()
    since = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
    runtime_group = args.runtime_log_group
    if runtime_group is None:
        try:
            runtime_group = runtime_log_group(outputs())
        except (OSError, ValueError):
            runtime_group = None
    groups = bridge_groups(runtime_group)
    logs = boto3.client("logs")
    calls = [analyze_call(sid, t0, groups, logs) for sid, t0 in calls_since(since)]
    summary = summarize(calls)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"calls-{args.label}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps({"label": args.label, "since": args.since, "summary": summary, "calls": calls}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


if __name__ == "__main__":
    main()
