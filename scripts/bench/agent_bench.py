"""Latency benchmark for the agent stage, comparable across architectures.

Runs fixed caller scenarios through an :class:`tac_bridge.agent_client.AgentClient`, the interface the
bridge itself uses, and times every streamed event the bridge would act on. An alternative architecture
(a direct-model loop, a Strands agent, AgentCore Runtime) registers its client in ``AGENTS`` and runs the
same scenarios unchanged, so results from different branches compare like for like.

What is measured, per caller utterance ("step"):

* ``first_text_s``: time to the first words the bridge would send to ConversationRelay (``TextDelta``
  through the bridge's ``<silent/>`` filter; when they come depends on ``--text-release``, below).
* ``handoff_s``: for a step that ends in a scripted step (``get_balance``, ``authenticate``,
  ``make_payment``, ``escalate``, ``end_call``), the time to that tool call: the caller hears the front door next.
* ``response_s``: what the caller waits for: ``first_text_s``, else ``handoff_s``, summed over the hops
  of the step (inline tool hops included; stub tool results take 0 s, real ones add ~0.1 to 0.3 s,
  see ``call_latency.py``).
* per hop: ``first_event_s``, ``tool_call_s``, ``end_s``.
* ``first_audio_s``: the first thing the caller hears with latency masking, computed from the
  measured timings and ``data/policy.json``'s masking settings (timer phrase, masked tool call, or
  the answer). ``--resummarize FILE`` recomputes a saved run's summary with the current settings,
  or with candidates (``--after-ms``, ``--tool-after-ms``, ``--tool-delay TOOL=MS``); ``--sweep``
  tabulates a grid of them. ``--masking-model v2`` (default) models scheduled tool phrases;
  ``legacy`` the earlier bridge.
* ``phrase_delay_s``: how long a ready answer waited behind phrases still playing.

* per hop, when the client reports them: the ``TurnEnd`` token usage (``usage``: input, output, cache
  read and cache write tokens) and, for the Strands client, each model call's ``strands_model_call``
  timing line (``model_calls``: time to first byte, latency, tokens, stop reason). ``overall`` in the
  result file summarizes them over every hop: per-hop time, model time, overhead per model call and
  the cache hit ratio.

Tool results are stubs built from the fictional fixtures, so no bank data or token is involved. Each run
uses a new session, warmed first the way production warms it unless ``--cold``: the Strands client gets
``begin_session(history="new")``, the agent pre-build the bridge starts at session start.

Agents (``--agent``):

* ``strands`` (default): a :class:`tac_bridge.agent_client_strands.StrandsAgentClient` run in this
  process with your AWS credentials, against the deployed Gateway (``GatewayUrl``) and Memory
  (``MemoryId``) from ``infra/cdk-outputs.json``, with the bridge's model, max tokens and reasoning
  setting (read from the synthesized ``infra/cdk.out/OwlBankBridge.template.json``, or given with
  ``--model-id``, ``--max-tokens`` and ``--thinking``). ``--prompt-caching on|off`` (default off)
  chooses the two Strands runs of 4a.2; no redeploy is needed for either. Sessions (Memory session
  IDs) are ``bench-<hex>``; their events expire with the Memory resource's event expiry, and the
  bench deletes none.

``--tracing on|off`` (default off) is the tracing-overhead comparison of call observability: ``on`` installs the
bridge's telemetry in this process exactly as the bridge does (``owlbank.telemetry.bridge.install_bridge``,
sample rate 1.0, ``traceContent`` off, stage ``dev``), exporting to the deployed traces log group
(``--traces-log-group``, default ``/owlbank/traces/bridge``) with your AWS credentials and ``--region``.
Each bench session's spans carry its ``bench-<hex>`` ID as their ``session.id``, so
``scripts/call_timeline.py --call-sid bench-<hex>`` shows the call. The providers are flushed at the end.
``off`` never imports ``owlbank.telemetry``. The result file and the default label record the setting
(``strands-nocache-tracing-on``); run on and off with the same ``--runs`` and compare with ``--compare``.

``--text-release stream|message`` sets ``StrandsSettings.text_release`` for the run; the default is the bridge's ``OWLBANK_TEXT_RELEASE`` from the synthesized
template, like the model settings (``stream`` when all three model settings are given). Stream
release with ``thinking`` off stops the run, as it stops synth. Run both modes with the same
``--runs`` and compare. The result file records it as ``text_release`` and the default label adds
``-message`` for the non-default mode. Each hop records ``text_before_tool`` (how many times the
agent spoke and then called a tool, stream mode only) and its ``text_before_tool_chars``;
``overall`` counts those hops (``text_before_tool_hops``). ``first_text_s`` and the hop's ``text``
are what the bridge would send (``<silent/>`` removed, even split across chunks). Files from before
these fields still load, ``--compare`` and ``--resummarize``.

The AgentCore harness was benchmarked too, before it was removed: its saved results
(``baseline-harness-*.json`` in the ``--out`` directory) are the historical baseline, still readable by
``--compare`` and ``--resummarize``.

Usage (repository root, AWS credentials for the sandbox)::

    uv run python scripts/bench/agent_bench.py --runs 8                                     # strands-nocache-*.json
    uv run python scripts/bench/agent_bench.py --agent strands --prompt-caching on --runs 8 # strands-cache-*.json
    uv run python scripts/bench/agent_bench.py --runs 8 --tracing on                        # strands-nocache-tracing-on-*.json
    uv run python scripts/bench/agent_bench.py --compare a.json b.json [c.json]
    uv run python scripts/bench/agent_bench.py --resummarize x.json \\
        --sweep after_ms=1500,2000,2500 tool_after_ms=0,300,600,900                     # offline
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import inspect
import json
import logging
import secrets
import statistics
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from owlbank import agent_prompt  # noqa: E402
from owlbank.catalog import PromptCatalog  # noqa: E402
from tac_bridge.agent_client import (  # noqa: E402
    AgentClient,
    SilentFilter,
    TextBeforeTool,
    TextDelta,
    TextSuppressed,
    ToolCall,
    ToolResult,
    TurnEnd,
)
from tac_bridge.clock import local_now_line, zone  # noqa: E402

HANDOFF_TOOLS = {"get_balance", "authenticate", "make_payment", "transfer_funds", "escalate", "end_call"}


#: Masking models (``--masking-model``): ``legacy`` is the earlier bridge (a tool phrase the moment
#: a masked tool call arrives); ``v2`` is the current one (scheduled tool phrases).
MASKING_MODELS = ("legacy", "v2")
#: Policy-wide default delay of a tool phrase, when ``masking.tool_after_ms`` is absent.
DEFAULT_TOOL_AFTER_MS = 600
DELAY_RANGE_MS = (0, 10_000)
TTS_START_S = 0.3  # as tac_bridge.masking.TTS_START_S


@dataclass(frozen=True)
class Masking:
    """Masking settings the audio timeline is computed with.

    ``ack_s`` and ``phrase_s`` are estimated phrase durations (``0.3 s + chars / chars_per_s``). The
    ``legacy`` model uses the acknowledgement pool's mean and each tool's single ``phrase_key`` text
    (what the earlier summaries were computed with); ``v2`` uses the **longest** phrase of each pool
    (the acknowledgement pool, and a tool's ``phrase_key`` plus any ``phrase_key.<n>``), so the
    estimated phrase delay is on the conservative side. ``delay_s`` is each masked tool's phrase
    delay (``v2`` only; hand-off tools are always 0)."""

    model: str
    timer_s: float | None
    ack_s: float
    phrase_s: dict[str, float]
    delay_s: dict[str, float]
    handoff: frozenset[str] = frozenset(HANDOFF_TOOLS)
    after_ms: int = 0
    tool_after_ms: int = DEFAULT_TOOL_AFTER_MS
    chars_per_s: float = 15.0

    def record(self) -> dict[str, Any]:
        """What a result file records about the settings its summary was computed with."""
        v2 = self.model == "v2"
        return {"model": self.model, "timer_s": self.timer_s, "after_ms": self.after_ms,
                "tool_after_ms": self.tool_after_ms if v2 else None,
                "tool_delay_ms": {t: round(d * 1000) for t, d in sorted(self.delay_s.items())} if v2 else None,
                "masked_tools": sorted(self.phrase_s), "chars_per_s": self.chars_per_s,
                "phrase_length": "longest of each pool" if v2 else "mean acknowledgement, single tool phrase"}


def _check_ms(name: str, v: Any) -> int:
    lo, hi = DELAY_RANGE_MS
    if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
        raise ValueError(f"{name}: {v!r} is not an integer from {lo} to {hi} ms")
    return v


def masking_settings(model: str = "v2", *, after_ms: int | None = None, tool_after_ms: int | None = None,
                     tool_delays: dict[str, int] | None = None, policy: dict[str, Any] | None = None,
                     catalog: dict[str, Any] | None = None) -> Masking:
    """From ``data/policy.json`` and the catalog, overridden by the candidate values given:
    the timer (``after_ms``, 0 = off), the policy-wide ``tool_after_ms``, and per-tool delays
    (``tool_delays``, else the tool's ``mask.after_ms`` in the policy, else ``tool_after_ms``)."""
    if model not in MASKING_MODELS:
        raise ValueError(f"masking model {model!r}: one of {', '.join(MASKING_MODELS)}")
    policy = policy if policy is not None else json.loads((ROOT / "data" / "policy.json").read_text())
    catalog = catalog if catalog is not None else json.loads((ROOT / "data" / "prompt_catalog.json").read_text())
    texts = {e["key"]: e["text"] for e in catalog["entries"] if e["lang"] == "en-US"}
    m = policy.get("masking") or {}
    after = _check_ms("after_ms", after_ms if after_ms is not None else m.get("after_ms", 1500))
    tool_after = _check_ms("tool_after_ms", tool_after_ms if tool_after_ms is not None
                           else m.get("tool_after_ms", DEFAULT_TOOL_AFTER_MS))
    rate = float(m.get("chars_per_s", 15)) or 15.0

    def dur(text: str) -> float:
        return TTS_START_S + len(text) / rate

    def pool(key: str) -> list[str]:
        return [t for k, t in texts.items() if k == key or (k.startswith(key + ".") and k[len(key) + 1:].isdigit())]

    masks = {name: row["mask"] for name, row in policy.get("tools", {}).items()
             if isinstance(row, dict) and isinstance(row.get("mask"), dict)}
    generic = [t for k, t in texts.items() if k.startswith("mask.generic.")]
    if model == "legacy":
        phrase_s = {n: dur(texts.get(mk["phrase_key"], "")) for n, mk in masks.items()}
        ack_s = sum(dur(t) for t in generic) / len(generic) if generic else 0.0
    else:
        phrase_s = {n: max((dur(t) for t in pool(mk["phrase_key"])), default=dur("")) for n, mk in masks.items()}
        ack_s = max((dur(t) for t in generic), default=0.0)
    overrides = dict(tool_delays or {})
    unknown = sorted(set(overrides) - set(masks))
    if unknown:
        raise ValueError(f"tool delay for tools that aren't masked in the policy: {', '.join(unknown)}")
    delay_s: dict[str, float] = {}
    for n, mk in masks.items():
        own = overrides.get(n, mk.get("after_ms"))
        if n in HANDOFF_TOOLS:
            delay_s[n] = 0.0  # A hand-off phrase plays at once
        elif own is not None:
            delay_s[n] = _check_ms(f"{n}.after_ms", own) / 1000
        else:
            delay_s[n] = tool_after / 1000
    return Masking(model=model, timer_s=after / 1000 if after else None, ack_s=ack_s, phrase_s=phrase_s,
                   delay_s=delay_s, after_ms=after, tool_after_ms=tool_after, chars_per_s=rate)


@dataclass(frozen=True)
class StepAudio:
    """What the caller hears in one step with masking.

    ``phrases`` are ``(start, duration, kind)``, kind ``ack`` or ``tool``. ``ready_s`` is when the
    answer (or hand-off) was ready (``response_s``); ``answer_s`` when it is heard: ConversationRelay
    queues text, so no earlier than the end of the phrases sent before it. ``phrase_delay_s`` is
    ``answer_s - ready_s`` (how long a ready answer waited behind a phrase; 0 if none was
    playing) for a step that ends in answer text, ``None`` for a hand-off step (no answer in this
    session; its phrase hold is not part of this)."""

    phrases: list[tuple[float, float, str]]
    ready_s: float
    answer_s: float
    phrase_delay_s: float | None

    def timeline(self) -> list[tuple[float, float]]:
        return [(s0, d) for s0, d, _ in self.phrases] + [(self.answer_s, 0.0)]

    @property
    def ack(self) -> bool:
        return any(k == "ack" for _, _, k in self.phrases)

    @property
    def tool_phrase(self) -> bool:
        return any(k == "tool" for _, _, k in self.phrases)

    @property
    def first_audio_s(self) -> float:
        return round(min([self.answer_s] + [s0 for s0, _, _ in self.phrases]), 3)

    @property
    def max_silence_s(self) -> float:
        """The longest stretch of silence: from the step's start to the first audio, and between the
        end of one phrase and the start of the next thing heard."""
        gaps, end = [], 0.0
        for s0, d in sorted(self.timeline()):
            gaps.append(max(0.0, s0 - end))
            end = max(end, s0 + d)
        return round(max(gaps), 3)


def _phrases_legacy(step: dict[str, Any], m: Masking) -> list[tuple[float, float, str]]:
    """The timer acknowledgement if the first hop has nothing by then (not on an
    ``await_caller`` resume); the first masked tool call's phrase at once if no answer text came first."""
    hops = step["hops"]
    out: list[tuple[float, float, str]] = []
    if m.timer_s is not None and step.get("timer_eligible", True):
        first = hops[0]["first_event_s"] or hops[0]["end_s"]
        if first > m.timer_s:
            out.append((m.timer_s, m.ack_s, "ack"))
    t0 = 0.0
    for h in hops:
        if h["first_text_s"] is not None:
            break
        hit = [t for t in h["tools"] if t in m.phrase_s]
        if h["tool_call_s"] is not None and hit:
            out.append((t0 + h["tool_call_s"], m.phrase_s[hit[0]], "tool"))
            break
        t0 += h["end_s"]
    return out


def _phrases_v2(step: dict[str, Any], m: Masking) -> list[tuple[float, float, str]]:
    """The timer acknowledgement at ``after_ms`` if the first hop has no text
    and no tool call by then (not on an ``await_caller`` resume); the first masked tool call's
    phrase scheduled at ``max(tool call + its delay, acknowledgement's estimated end)`` and spoken
    only if the answer isn't ready by then (``response_s`` > deadline); a hand-off tool's phrase at
    once (delay 0, after a playing acknowledgement), always spoken."""
    hops = step["hops"]
    ready = step["response_s"]
    out: list[tuple[float, float, str]] = []
    ack_end = 0.0
    if m.timer_s is not None and step.get("timer_eligible", True):
        h0 = hops[0]
        marks = [t for t in (h0["first_text_s"], h0["tool_call_s"]) if t is not None]
        if (min(marks) if marks else h0["end_s"]) > m.timer_s:
            out.append((m.timer_s, m.ack_s, "ack"))
            ack_end = m.timer_s + m.ack_s
    t0 = 0.0
    for h in hops:
        if h["first_text_s"] is not None:
            break
        hit = [t for t in h["tools"] if t in m.phrase_s]
        if h["tool_call_s"] is not None and hit:
            tool, call = hit[0], t0 + h["tool_call_s"]
            handoff = tool in m.handoff
            delay = 0.0 if handoff else m.delay_s.get(tool, m.tool_after_ms / 1000)
            deadline = max(call + delay, ack_end)
            if handoff or ready > deadline:
                out.append((deadline, m.phrase_s[tool], "tool"))
            break
        t0 += h["end_s"]
    return out


def step_audio(step: dict[str, Any], m: Masking) -> StepAudio | None:
    """The step's audio under masking model ``m.model``; ``None`` for a step without hops."""
    if not step.get("hops"):
        return None
    phrases = (_phrases_legacy if m.model == "legacy" else _phrases_v2)(step, m)
    ready = step["response_s"]
    answer = max([ready] + [s0 + d for s0, d, _ in phrases])
    delay = round(answer - ready, 3) if step.get("handoff") is None else None
    return StepAudio(phrases, ready, answer, delay)


def audio_timeline(step: dict[str, Any], m: Masking) -> list[tuple[float, float]]:
    """(start, duration) of what the caller hears in this step; the answer or hand-off last, with
    duration 0 (only its start matters here)."""
    a = step_audio(step, m)
    return a.timeline() if a else []


def first_audio_s(step: dict[str, Any], m: Masking) -> float | None:
    a = step_audio(step, m)
    return a.first_audio_s if a else None


def max_silence_s(step: dict[str, Any], m: Masking) -> float | None:
    a = step_audio(step, m)
    return a.max_silence_s if a else None


def masking_overall(runs: list[dict[str, Any]], m: Masking) -> dict[str, Any]:
    """Over every step of every successful run: longest silence p50/p90, first audio p50/p90,
    phrase delay p50/p90 (answer steps only), and the share of steps with an acknowledgement and
    with a tool phrase."""
    audio = [a for r in runs if "error" not in r for st in r.get("steps", []) if "response_s" in st
             for a in [step_audio(st, m)] if a is not None]
    sil = [a.max_silence_s for a in audio]
    fa = [a.first_audio_s for a in audio]
    pd = [a.phrase_delay_s for a in audio if a.phrase_delay_s is not None]
    n = len(audio)
    return {"steps": n, "max_silence_p50": pct(sil, .5), "max_silence_p90": pct(sil, .9),
            "first_audio_p50": pct(fa, .5), "first_audio_p90": pct(fa, .9),
            "phrase_delay_p50": pct(pd, .5), "phrase_delay_p90": pct(pd, .9),
            "ack_share": round(sum(a.ack for a in audio) / n, 3) if n else None,
            "tool_phrase_share": round(sum(a.tool_phrase for a in audio) / n, 3) if n else None}


# ----------------------------------------------------------------------------- the system prompt


#: The verified fictional customer's profile line, as ``Bridge._system_prompt`` passes it.
BENCH_TIMEZONE = "America/New_York"  # ``BridgeConfig.timezone``'s default
BENCH_PROFILE = "name Thomas Doe; accounts: checking (Everyday Checking), savings, credit_card (OwlBank Rewards Card)"


def system_prompt(lang: str = "en-US", model_key: str = agent_prompt.DEFAULT_MODEL_KEY) -> agent_prompt.RenderedPrompt:
    """The bridge's prompt for the verified fictional customer with no memory, rendered by
    :func:`owlbank.agent_prompt.render` from the seed catalog, so the bench sends what callers get
    (the placeholder rule included): the static part and the per-call part."""
    catalog = PromptCatalog.from_dict(json.loads((ROOT / "data" / "prompt_catalog.json").read_text()))
    return agent_prompt.render(catalog, model_key, lang, caller=agent_prompt.CallerFacts("Thomas", BENCH_PROFILE, None),
                               web_block=None)


def system_prompts(model_ids: Sequence[str] = (), lang: str = "en-US") -> dict[str, str]:
    """The static part of :func:`system_prompt` per model key of ``model_ids``, as the bridge
    sends them (before the cache point); no IDs: the default model key's alone."""
    keys = dict.fromkeys(agent_prompt.model_key(m) for m in model_ids) or {agent_prompt.DEFAULT_MODEL_KEY: None}
    return {k: system_prompt(lang, k).static for k in keys}


def system_suffix(lang: str = "en-US", now: datetime | None = None) -> str:
    """What the bridge sends after the cache point: the per-call part, a newline and the clock line
    (the bank's local time; the bridge's default zone)."""
    per_call = system_prompt(lang).per_call  # the same for every model key
    return f"{per_call}\n{local_now_line(now or datetime.now(UTC), zone(BENCH_TIMEZONE), lang)}"


# ----------------------------------------------------------------------------- scenarios
# A step is one caller utterance. ``results`` answers the tool calls the agent makes during that step:
# tool name -> result content (or a callable of the call's input). A handoff tool's result is delivered
# at the start of the NEXT step, the way the front door's pending result reaches the bridge.

TXNS = {"tool": "list_recent_transactions", "data": {"accountType": "checking", "last4": "4821", "transactions": [
    {"amount": "-4.50", "date": "2026-10-09", "description": "Coffee Example Co", "status": "pending"},
    {"amount": "-71.00", "date": "2026-10-09", "description": "Grocery Sample Mart", "status": "pending"},
    {"amount": "2100.00", "date": "2026-10-08", "description": "Payroll Example Corp", "status": "posted"},
    {"amount": "-88.20", "date": "2026-10-07", "description": "Utility Sample Power", "status": "posted"}]}}
BALANCE = {"tool": "get_balance", "status": "completed", "data": {"account": "savings", "available": "8420.00",
           "current": "8420.00", "last4": "7730"},
           "spoken": [{"template": "balance.read", "text": "Your savings account ending in seven, seven, three, zero has an available balance of eight thousand four hundred twenty dollars."},
                      {"template": "common.anything_else", "text": "Is there anything else I can help you with?"}]}
STEPUP_OK = {"tool": "authenticate", "status": "verified", "forTool": "make_payment", "next": "retry_original_tool",
             "data": {"level": 3}, "spoken": [{"template": "stepup.success", "text": "Thanks, you're verified."}]}
PAID = {"tool": "make_payment", "status": "completed", "data": {"amount": "25.00", "confirmation": "85027343", "executed": True},
        "spoken": [{"template": "confirm.make_payment.done", "text": "Done. Your payment of twenty-five dollars from your checking account is complete."},
                   {"template": "common.anything_else", "text": "Is there anything else I can help you with?"}]}


@dataclass
class Step:
    say: str | None                      # the caller's words (None: the step starts from a pending result)
    pending: dict[str, Any] | None = None  # the handoff tool's result to deliver first
    results: dict[str, Any] = field(default_factory=dict)


SCENARIOS: dict[str, list[Step]] = {
    "greeting": [Step("Hi, what can you help me with today?")],
    "faq_gateway": [Step("What are your branch hours on Saturday?")],
    "transactions_inline": [Step("What are my recent transactions on checking?",
                                 results={"list_recent_transactions": TXNS})],
    "balance_then_bargein": [
        Step("What's my savings balance?"),
        Step(None, pending={**BALANCE, "next": "handle_utterance",
                            "callerUtterance": "And what are my recent transactions on checking?"},
             results={"list_recent_transactions": TXNS}),
    ],
    "payment_stepup": [
        Step("I want to pay 25 dollars to my credit card from checking.",
             results={"make_payment": {"error": "AUTH_LEVEL_INSUFFICIENT", "tool": "make_payment", "_status": "error"}}),
        Step(None, pending=STEPUP_OK),
        Step(None, pending={**PAID, "next": "handle_utterance", "callerUtterance": "Thanks. Transfer me to an agent, please."}),
    ],
}
# In payment_stepup step 1 the inline make_payment returns AUTH_LEVEL_INSUFFICIENT and the agent then calls
# authenticate (a handoff). Step 2 delivers the step-up result; the agent retries make_payment (a handoff,
# the scripted confirmation). Step 3 delivers the paid result with the caller asking for a person.


# ----------------------------------------------------------------------------- agents

#: ``warm(session_id)``: run before a session's first turn, sync (run in a thread) or async.
Warm = Callable[[str], None] | Callable[[str], Awaitable[None]]
#: A factory returns the client, its warm function (or None) and, optionally, what to record about
#: the backend in the result file (``backend``).
AgentFactory = Callable[[argparse.Namespace],
                        tuple[AgentClient, Warm | None] | tuple[AgentClient, Warm | None, dict[str, Any]]]

OUTPUTS = ROOT / "infra" / "cdk-outputs.json"
BRIDGE_TEMPLATE = ROOT / "infra" / "cdk.out" / "OwlBankBridge.template.json"
#: Session-ID prefix, so benchmark sessions are recognisable in Memory.
SESSION_PREFIX = "bench-"
#: The bridge task's environment variable that holds the agent's model list (a
#: JSON list of entries, the preferred first; the benchmark runs the preferred entry).
MODELS_ENV = "OWLBANK_MODELS"
STRANDS_LOGGER = "tac_bridge.agent_client_strands"
PACKAGES = ("strands-agents", "bedrock-agentcore", "mcp-proxy-for-aws")


def deployed_model_settings(template: Path = BRIDGE_TEMPLATE) -> dict[str, Any]:
    """The bridge's preferred model entry (the first of ``OWLBANK_MODELS`` in the bridge task
    definition: model ID, max tokens, reasoning setting and text release), from the synthesized
    bridge template: what the last synth or deploy from this checkout produced, ``-c`` flags
    included. ``source`` names the file; ``models`` lists every entry's ID (model failover)."""
    from tac_bridge.agent_client_strands import TEXT_RELEASE_MODES, thinking_error  # noqa: PLC0415

    if not template.is_file():
        raise SystemExit(f"{_rel(template)} not found: synthesize the stacks (cdk synth) or pass "
                         "--model-id, --max-tokens and --thinking")
    resources = json.loads(template.read_text()).get("Resources", {})
    found = []
    for r in resources.values():
        if r.get("Type") != "AWS::ECS::TaskDefinition":
            continue
        for c in r.get("Properties", {}).get("ContainerDefinitions", []):
            env = {e.get("Name"): e.get("Value") for e in c.get("Environment", []) if isinstance(e, dict)}
            if MODELS_ENV in env:
                found.append(env)
    if len(found) != 1:
        raise SystemExit(f"{_rel(template)}: no single bridge container with {MODELS_ENV}; pass "
                         "--model-id, --max-tokens and --thinking")
    unrecognised = (f"{_rel(template)}: unrecognised agent model settings in {MODELS_ENV}; pass --model-id, "
                    "--thinking and --max-tokens")
    try:
        entries = json.loads(found[0][MODELS_ENV])
    except (TypeError, ValueError):
        raise SystemExit(unrecognised) from None
    if not isinstance(entries, list) or not entries or not all(isinstance(e, dict) for e in entries):
        raise SystemExit(unrecognised)
    first = entries[0]
    model_id, max_tokens, thinking = first.get("id"), first.get("max_tokens"), first.get("thinking")
    text_release = first.get("text_release")
    if (not isinstance(model_id, str) or not model_id or isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int) or not isinstance(thinking, str)
            or thinking_error(thinking, max_tokens) is not None):
        raise SystemExit(unrecognised)
    if text_release not in TEXT_RELEASE_MODES:
        raise SystemExit(f"{_rel(template)}: unrecognised text_release {text_release!r} in {MODELS_ENV}; "
                         f"pass --text-release ({' or '.join(sorted(TEXT_RELEASE_MODES))})")
    return {"model_id": model_id, "max_tokens": max_tokens, "thinking": thinking,
            "text_release": text_release, "models": [e.get("id") for e in entries], "source": _rel(template)}


def _model_info(m: dict[str, Any]) -> dict[str, Any]:
    return {"model_id": m["model_id"], "max_tokens": m["max_tokens"], "thinking": m["thinking"],
            "model_settings_source": m.get("source")}


def package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


class LogCapture(logging.Handler):
    """Collects the Strands client's ``strands_model_call`` and ``agent_history_*`` lines (their
    fields are record attributes, :func:`owlbank.obs.emit`). Runs are sequential, so one buffer."""

    EVENTS = {"strands_model_call", "agent_history_new", "agent_history_reload"}
    FIELDS = ("call", "model_ttfb_s", "model_first_text_s", "model_first_sentence_s", "model_latency_s","input_tokens", "output_tokens", "cache_read_tokens",
              "cache_write_tokens", "stop_reason", "reason", "duration_ms", "outcome", "messages", "interrupted")

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        event = getattr(record, "event", None)
        if event in self.EVENTS:
            row = {"event": event}
            row.update({k: getattr(record, k) for k in self.FIELDS if hasattr(record, k)})
            self.records.append(row)

    def drain(self) -> list[dict[str, Any]]:
        out, self.records = self.records, []
        return out


#: Installed by :func:`make_strands`; ``None`` for other agents (no per-model-call lines).
CAPTURE: LogCapture | None = None


def install_capture() -> LogCapture:
    global CAPTURE  # noqa: PLW0603 - one benchmark process, one capture
    if CAPTURE is None:
        CAPTURE = LogCapture()
        logger = logging.getLogger(STRANDS_LOGGER)
        logger.setLevel(logging.INFO)  # emit() checks isEnabledFor before building the line
        logger.addHandler(CAPTURE)
    return CAPTURE


def make_strands(args: argparse.Namespace) -> tuple[AgentClient, Warm | None, dict[str, Any]]:
    """A real :class:`StrandsAgentClient` in this process, on the deployed Gateway and
    Memory, with the bridge's model settings and text release (overridable with ``--model-id``,
    ``--thinking``, ``--max-tokens``, ``--text-release``; with the first three the template isn't
    read and the text release defaults to ``stream``) and ``--prompt-caching``. Stream release with
    reasoning off stops the run, as it stops synth and the bridge."""
    import tac_bridge.agent_client_strands as acs  # noqa: PLC0415 - only this agent needs Strands

    overridden = [k for k in ("model_id", "max_tokens", "thinking") if getattr(args, k)]
    m = deployed_model_settings() if len(overridden) < 3 else {"source": "command line", "text_release": "stream"}
    model = {"model_id": args.model_id or m["model_id"], "max_tokens": args.max_tokens or m["max_tokens"],
             "thinking": args.thinking or m["thinking"]}
    if len(overridden) == 3:
        overridden = []
    elif args.text_release:
        overridden.append("text_release")
    text_release = args.text_release or m["text_release"]
    if text_release is None:
        raise SystemExit(f"{m['source']}: no text release; synthesize the stacks again or pass --text-release")
    conflict = acs.text_release_conflict(text_release, model["thinking"])
    if conflict is not None:
        raise SystemExit(f"{conflict} (here: --text-release message, or --thinking adaptive)")
    entry = acs.ModelEntry(id=model["model_id"], thinking=model["thinking"], max_tokens=model["max_tokens"],
                           text_release=text_release, prompt_caching=args.prompt_caching == "on")
    settings = acs.StrandsSettings(
        models=(entry,), region=args.region,
        gateway_url=args.gateway_url or _from_outputs("OwlBankAgent", "GatewayUrl"),
        memory_id=args.memory_id or _from_outputs("OwlBankAgent", "MemoryId"))
    install_capture()
    client = acs.StrandsAgentClient(
        settings,
        gateway_tools=acs.make_gateway_tool_provider(settings.gateway_url, settings.region),
        session_manager_factory=acs.memory_session_manager_factory(settings.memory_id, settings.region))

    async def warm(session_id: str) -> None:  # what the bridge starts at session start
        await client.begin_session(session_id, history="new")

    info = {"backend": "strands", "prompt_caching": entry.prompt_caching,
            "text_release": entry.text_release, **model,
            "model_settings_source": m.get("source") + (f" (overridden: {', '.join(overridden)})" if overridden else ""),
            "gateway_url": settings.gateway_url, "memory_id": settings.memory_id,
            "memory_actor_id": acs.MEMORY_ACTOR_ID, "session_prefix": SESSION_PREFIX,
            "packages": package_versions()}
    return client, warm, info


#: name -> factory(args) -> (agent, warm(session_id) or None[, backend info]). Add alternatives here.
AGENTS: dict[str, AgentFactory] = {
    "strands": make_strands,
}


#: The deployed traces log group (``call_observability.TRACES_LOG_GROUP_NAME``).
TRACES_LOG_GROUP = "/owlbank/traces/bridge"
TRACING_STAGE = "dev"


def tracing_environment(region: str, log_group: str = TRACES_LOG_GROUP) -> dict[str, str]:
    """The environment the bridge's telemetry reads (``owlbank.telemetry.settings``) for a bench process:
    on, every call sampled, no conversation text, the deployed traces group."""
    return {"OWLBANK_CALL_OBSERVABILITY": "on", "OWLBANK_TRACE_SAMPLE_RATE": "1.0",
            "OWLBANK_TRACE_CONTENT": "off", "OWLBANK_TRACES_LOG_GROUP": log_group,
            "OWLBANK_STAGE": TRACING_STAGE, "OWLBANK_HOSTING": "fargate", "AWS_REGION": region}


def start_tracing(args: argparse.Namespace, *, install: Callable[..., Any] | None = None) -> Any:
    """``--tracing on``: install the bridge telemetry as ``tac_bridge.app._start_telemetry`` does, before the
    Strands client is built (so it finds the providers). Returns the handle, or stops the run when the
    install is refused or fails (an untraced "on" run would make the comparison meaningless). ``install`` is the
    installer, for tests. Imported here so that ``--tracing off`` never loads ``owlbank.telemetry``."""
    from owlbank.telemetry import TelemetrySettings  # noqa: PLC0415

    env = tracing_environment(args.region, args.traces_log_group)
    settings = TelemetrySettings.from_env(env, service="bridge")
    if settings is None:
        raise SystemExit("--tracing on: the telemetry settings were refused (see the log line above)")
    if install is None:
        from owlbank.telemetry.bridge import install_bridge  # noqa: PLC0415

        install = install_bridge
    handle = install(settings, environ=env)
    if handle is None:
        raise SystemExit("--tracing on: the telemetry install was refused or failed (see the log line above; "
                         "an OTEL_EXPORTER_OTLP_*ENDPOINT variable in the environment refuses it)")
    print(f"tracing on: spans to {settings.traces_log_group} in {settings.region}, sample rate "
          f"{settings.sample_rate}, content {settings.content}", flush=True)
    return handle


def flush_tracing(handle: Any, timeout_ms: int = 15_000) -> None:
    """Export what is still queued, then stop the providers. Never raises."""
    try:
        handle.tracer_provider.force_flush(timeout_ms)
        handle.shutdown()
    except Exception as exc:  # noqa: BLE001 - a flush failure must not lose the benchmark's result
        print(f"tracing flush failed: {type(exc).__name__}", file=sys.stderr)


def _from_outputs(stack: str, key_part: str) -> str:
    outputs = json.loads(OUTPUTS.read_text()) if OUTPUTS.is_file() else {}
    values = outputs.get(stack, {})
    if key_part in values:  # the stack's own output (e.g. GatewayUrl, MemoryId)
        return str(values[key_part])
    for k, v in values.items():
        if key_part in k and "Runtime" not in k:
            return str(v)
    raise SystemExit(f"no {key_part} in infra/cdk-outputs.json[{stack}]; deploy the stack with "
                     "--outputs-file cdk-outputs.json, or pass it explicitly")


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


# ----------------------------------------------------------------------------- running

async def run_hop(agent: AgentClient, sid: str, prompts: Mapping[str, str], *, user_text: str | None,
                  results: list[ToolResult], suffix: str | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    hop: dict[str, Any] = {"first_event_s": None, "first_text_s": None, "tool_call_s": None, "end_s": None,
                           "tools": [], "text": "", "suppressed_chars": 0, "text_before_tool": 0,
                           "text_before_tool_chars": 0, "stop": None, "usage": None}
    calls: list[ToolCall] = []
    silent = SilentFilter()  # what the bridge would send: no <silent/>, even split across chunks
    if CAPTURE is not None:
        CAPTURE.drain()  # only this hop's lines
    async for ev in agent.turn(sid, user_text=user_text, tool_results=results, system_prompts=prompts,
                                   system_suffix=suffix):
        t = round(time.perf_counter() - t0, 3)
        hop["first_event_s"] = hop["first_event_s"] or t
        if isinstance(ev, TextDelta):
            hop["text"] += silent.feed(ev.text)
            if hop["first_text_s"] is None and hop["text"].strip():
                hop["first_text_s"] = t
        elif isinstance(ev, TextBeforeTool):  # stream mode: the agent spoke, then called a tool
            hop["text_before_tool"] += 1
            hop["text_before_tool_chars"] += ev.chars
        elif isinstance(ev, TextSuppressed):
            hop["suppressed_chars"] += ev.chars
        elif isinstance(ev, ToolCall):
            hop["tool_call_s"] = hop["tool_call_s"] or t
            hop["tools"].append(ev.name)
            calls.append(ev)
        elif isinstance(ev, TurnEnd):
            hop["stop"] = ev.stop_reason
            hop["usage"] = dict(ev.usage) if ev.usage else None
    hop["end_s"] = round(time.perf_counter() - t0, 3)
    hop["text"] += silent.flush()  # a held marker prefix that never became a marker: sent at the hop's end
    if hop["first_text_s"] is None and hop["text"].strip():
        hop["first_text_s"] = hop["end_s"]
    hop["_calls"] = calls
    if CAPTURE is not None:
        lines = CAPTURE.drain()
        hop["model_calls"] = [{k: v for k, v in r.items() if k != "event"} for r in lines
                              if r["event"] == "strands_model_call"]
        hop["_history"] = [r for r in lines if r["event"] != "strands_model_call"]
    return hop


def _result_for(step: Step, call: ToolCall) -> ToolResult:
    spec = step.results.get(call.name, {"tool": call.name, "status": "completed", "data": {}})
    content = dict(spec(call.input) if callable(spec) else spec)
    status = content.pop("_status", "success")
    return ToolResult(call.tool_use_id, content, status)


async def run_scenario(agent: AgentClient, warm: Warm | None, name: str,
                       steps: list[Step], *, cold: bool, max_hops: int = 6,
                       session_prefix: str = SESSION_PREFIX, settle_s: float = 3.0) -> dict[str, Any]:
    sid = session_prefix + secrets.token_hex(14)
    history: list[dict[str, Any]] = []
    if warm and not cold:
        if inspect.iscoroutinefunction(warm):
            await warm(sid)
        else:
            await asyncio.to_thread(warm, sid)
        if CAPTURE is not None:
            history += CAPTURE.drain()
        await asyncio.sleep(settle_s)  # the caller is still on the language and ZIP prompts in production
    prompts = system_prompts(getattr(getattr(agent, "settings", None), "model_ids", ()))
    suffix = system_suffix()
    pending_call: ToolCall | None = None
    out_steps = []
    for step in steps:
        hops = []
        user_text = step.say
        results: list[ToolResult] = []
        if step.pending is not None:
            if pending_call is None:
                out_steps.append({"error": "no pending handoff call to answer"})
                break
            results = [ToolResult(pending_call.tool_use_id, step.pending)]
            user_text = None
        handoff = None
        for _ in range(max_hops):
            hop = await run_hop(agent, sid, prompts, user_text=user_text, results=results, suffix=suffix)
            calls = hop.pop("_calls")
            history += hop.pop("_history", [])
            hops.append(hop)
            handoffs = [c for c in calls if c.name in HANDOFF_TOOLS and c.name not in step.results]
            if handoffs:
                handoff = handoffs[0]
                break
            if not calls:
                break
            user_text, results = None, [_result_for(step, c) for c in calls]
        pending_call = handoff
        response = 0.0
        for h in hops:
            if h["first_text_s"] is not None:
                response += h["first_text_s"]
                break
            if handoff is not None and h is hops[-1]:
                response += h["tool_call_s"] or h["end_s"]
                break
            response += h["end_s"]
        out_steps.append({"say": step.say, "hops": hops, "handoff": handoff.name if handoff else None,
                          "timer_eligible": step.pending is None or step.pending.get("next") != "await_caller",
                          "response_s": round(response, 3), "first_text_s": hops[-1]["first_text_s"] if hops else None})
    out = {"scenario": name, "session": sid, "steps": out_steps}
    if CAPTURE is not None:
        out["history_builds"] = [{k: v for k, v in r.items() if k in ("event", "reason", "duration_ms", "outcome")}
                                 for r in history]
    return out


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def summarize(runs: list[dict[str, Any]], masking: Masking | None = None) -> dict[str, Any]:
    """Per step (``scenario#n``): answer time, masking figures (with ``masking``, default the
    policy's settings under the ``v2`` model) and tokens."""
    mk = masking or masking_settings()
    by: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        for i, st in enumerate(r["steps"]):
            key = f"{r['scenario']}#{i + 1}"
            m = by.setdefault(key, {"response_s": [], "hop1_first_event_s": [], "hops": [], "hop_end_s": [],
                                    "first_audio_s": [], "max_silence_s": [], "phrase_delay_s": [],
                                    "ack": [], "tool_phrase": [], **{t: [] for t in USAGE_KEYS}})
            if "response_s" in st:
                m["response_s"].append(st["response_s"])
                a = step_audio(st, mk)
                if a is not None:
                    m["first_audio_s"].append(a.first_audio_s)
                    m["max_silence_s"].append(a.max_silence_s)
                    m["ack"].append(float(a.ack))
                    m["tool_phrase"].append(float(a.tool_phrase))
                    if a.phrase_delay_s is not None:
                        m["phrase_delay_s"].append(a.phrase_delay_s)
                m["hops"].append(len(st["hops"]))
                if st["hops"] and st["hops"][0]["first_event_s"] is not None:
                    m["hop1_first_event_s"].append(st["hops"][0]["first_event_s"])
                m["hop_end_s"].extend(h["end_s"] for h in st["hops"])
                used = [h["usage"] for h in st["hops"] if h.get("usage")]
                if used:
                    for t in USAGE_KEYS:
                        m[t].append(sum(u.get(t, 0) or 0 for u in used))
    return {k: {"n": len(v["response_s"]),
                "response_p50": pct(v["response_s"], .5), "response_p90": pct(v["response_s"], .9),
                "first_audio_p50": pct(v["first_audio_s"], .5), "first_audio_p90": pct(v["first_audio_s"], .9),
                "max_silence_p50": pct(v["max_silence_s"], .5), "max_silence_p90": pct(v["max_silence_s"], .9),
                "phrase_delay_p50": pct(v["phrase_delay_s"], .5), "phrase_delay_p90": pct(v["phrase_delay_s"], .9),
                "ack_share": round(statistics.mean(v["ack"]), 3) if v["ack"] else None,
                "tool_phrase_share": round(statistics.mean(v["tool_phrase"]), 3) if v["tool_phrase"] else None,
                "hop_end_p50": pct(v["hop_end_s"], .5), "hops_mean": round(statistics.mean(v["hops"]), 2) if v["hops"] else None,
                "hop1_first_event_p50": pct(v["hop1_first_event_s"], .5),
                # tokens per step (all its hops), mean over the runs; None when the client sent no usage
                **{f"{t}_mean": round(statistics.mean(v[t]), 1) if v[t] else None for t in USAGE_KEYS}}
            for k, v in by.items()}


#: ``TurnEnd`` usage keys (both clients use these names; the cache ones only from Strands).
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


def cache_hit_ratio(read: float, write: float, uncached: float) -> float | None:
    """Share of the prompt read from the cache. Bedrock reports the uncached input (``inputTokens``)
    apart from the cached reads and writes, so the prompt is their sum."""
    total = read + write + uncached
    return round(read / total, 3) if total else None


def overall(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Over every hop of every successful run (computed from ``raw``, so files from before these
    fields get the hop numbers too): per-hop time, model time per model call, overhead per model call
    ((hop time - its model time) / its model calls, hops with timing lines only), token totals and the
    cache hit ratio."""
    hops = [h for r in runs if "error" not in r for st in r.get("steps", []) for h in st.get("hops", [])]
    ends = [h["end_s"] for h in hops if h.get("end_s") is not None]
    mc = [c for h in hops for c in h.get("model_calls") or []]
    lat = [c["model_latency_s"] for c in mc if c.get("model_latency_s") is not None]
    ttfb = [c["model_ttfb_s"] for c in mc if c.get("model_ttfb_s") is not None]
    overhead = []
    for h in hops:
        calls = h.get("model_calls") or []
        if calls and h.get("end_s") is not None and all(c.get("model_latency_s") is not None for c in calls):
            overhead.append(round((h["end_s"] - sum(c["model_latency_s"] for c in calls)) / len(calls), 3))
    used = [h["usage"] for h in hops if h.get("usage")]
    totals = {t: sum(u.get(t, 0) or 0 for u in used) for t in USAGE_KEYS} if used else None
    # Stream mode: hops whose agent spoke before calling a tool (0 in message mode; files from before
    # the text-release setting have no such hop field and count 0).
    narrated = [h for h in hops if h.get("text_before_tool")]
    # Answer calls (no tool use) that timed their text: what releasing the first sentence would save.
    answers = [c for c in mc if c.get("stop_reason") == "end_turn" and c.get("model_first_sentence_s") is not None
               and c.get("model_latency_s") is not None]
    first_text = [c["model_first_text_s"] for c in answers if c.get("model_first_text_s") is not None]
    to_text = [round(c["model_first_text_s"] - c["model_ttfb_s"], 3) for c in answers
               if c.get("model_first_text_s") is not None and c.get("model_ttfb_s") is not None]
    saving = [round(c["model_latency_s"] - c["model_first_sentence_s"], 3) for c in answers]
    stream_saving = [round(c["model_latency_s"] - c["model_first_text_s"], 3) for c in answers
                     if c.get("model_first_text_s") is not None]
    return {
        "hops": len(ends), "hop_end_p50": pct(ends, .5), "hop_end_p90": pct(ends, .9),
        "model_calls": len(mc), "model_latency_p50": pct(lat, .5), "model_latency_p90": pct(lat, .9),
        "model_ttfb_p50": pct(ttfb, .5), "model_ttfb_p90": pct(ttfb, .9),
        "overhead_per_model_call_p50": pct(overhead, .5),
        "overhead_per_model_call_mean": round(statistics.mean(overhead), 3) if overhead else None,
        "answer_calls_timed": len(answers),
        "text_before_tool_hops": len(narrated),
        "text_before_tool_chars": sum(h.get("text_before_tool_chars", 0) for h in narrated),
        "first_text_p50": pct(first_text, .5), "first_text_p90": pct(first_text, .9),
        "first_byte_to_text_p50": pct(to_text, .5), "first_byte_to_text_p90": pct(to_text, .9),
        "sentence_release_saving_p50": pct(saving, .5), "sentence_release_saving_p90": pct(saving, .9),
        "stream_release_saving_p50": pct(stream_saving, .5), "stream_release_saving_p90": pct(stream_saving, .9),
        "usage": totals,
        "cache_hit_ratio": cache_hit_ratio(totals["cache_read_tokens"], totals["cache_write_tokens"],
                                           totals["input_tokens"]) if totals else None,
        "history_builds": collections.Counter(b["event"] + ":" + str(b.get("reason")) for r in runs
                                              for b in r.get("history_builds", [])) or None,
    }


def print_table(summary: dict[str, Any], title: str, masking: dict[str, Any] | None = None) -> None:
    print(f"\n{title}\n{'step':30} {'n':>3} {'answer p50':>11} {'p90':>6} {'1st audio':>10} "
          f"{'longest silence p50':>20} {'p90':>6} {'phrase delay p50':>17} {'p90':>6}")
    for k, v in summary.items():
        print(f"{k:30} {v['n']:>3} {v['response_p50'] or 0:>11.2f} {v['response_p90'] or 0:>6.2f} "
              f"{v.get('first_audio_p50') or 0:>10.2f} {v.get('max_silence_p50') or 0:>20.2f} {v.get('max_silence_p90') or 0:>6.2f} "
              f"{_fmt(v.get('phrase_delay_p50'), 17)} {_fmt(v.get('phrase_delay_p90'), 6)}")
    if masking:
        print(f"all steps: longest silence p50 {masking['max_silence_p50']} p90 {masking['max_silence_p90']}, "
              f"first audio p50 {masking['first_audio_p50']}, phrase delay p50 {masking['phrase_delay_p50']} "
              f"p90 {masking['phrase_delay_p90']}, ack in {masking['ack_share']}, tool phrase in {masking['tool_phrase_share']} "
              f"of {masking['steps']} steps")


SWEEP_KEYS = ("after_ms", "tool_after_ms")


def parse_sweep(items: list[str]) -> dict[str, list[int]]:
    """``after_ms=1500,2000 tool_after_ms=0,300`` -> ``{"after_ms": [1500, 2000], ...}``."""
    out: dict[str, list[int]] = {}
    for item in items:
        key, sep, vals = item.partition("=")
        if not sep or key not in SWEEP_KEYS:
            raise ValueError(f"--sweep {item!r}: expected KEY=MS[,MS...] with KEY one of {', '.join(SWEEP_KEYS)}")
        try:
            out[key] = [_check_ms(key, int(v)) for v in vals.split(",") if v.strip()]
        except ValueError as exc:
            raise ValueError(f"--sweep {item!r}: {exc}") from None
        if not out[key]:
            raise ValueError(f"--sweep {item!r}: no values")
    return out


def parse_tool_delays(items: list[str] | None) -> dict[str, int]:
    """``--tool-delay TOOL=MS`` (repeatable) -> ``{tool: ms}``."""
    out: dict[str, int] = {}
    for item in items or []:
        tool, sep, ms = item.partition("=")
        if not sep or not tool:
            raise ValueError(f"--tool-delay {item!r}: expected TOOL=MS")
        try:
            out[tool] = _check_ms(f"--tool-delay {tool}", int(ms))
        except ValueError as exc:
            raise ValueError(str(exc)) from None
    return out


def sweep(runs: list[dict[str, Any]], grid: dict[str, list[int]], *, model: str = "v2",
          after_ms: int | None = None, tool_after_ms: int | None = None,
          tool_delays: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """:func:`masking_overall` for every combination of the grid's values (the other settings as
    given, else the policy's)."""
    afters: list[int | None] = list(grid["after_ms"]) if grid.get("after_ms") else [after_ms]
    tools: list[int | None] = list(grid["tool_after_ms"]) if grid.get("tool_after_ms") else [tool_after_ms]
    rows = []
    for a in afters:
        for t in tools:
            m = masking_settings(model, after_ms=a, tool_after_ms=t, tool_delays=tool_delays)
            rows.append({"after_ms": m.after_ms, "tool_after_ms": m.tool_after_ms, **masking_overall(runs, m)})
    return rows


def sweep_lines(rows: list[dict[str, Any]]) -> list[str]:
    out = [f"{'after_ms':>8} {'tool_after_ms':>13} {'silence p50':>11} {'p90':>6} {'1st audio p50':>13} "
           f"{'phrase delay p50':>16} {'p90':>6} {'ack':>6} {'tool phrase':>11}"]
    for r in rows:
        out.append(f"{r['after_ms']:>8} {r['tool_after_ms']:>13} {_fmt(r['max_silence_p50'], 11)} "
                   f"{_fmt(r['max_silence_p90'], 6)} {_fmt(r['first_audio_p50'], 13)} "
                   f"{_fmt(r['phrase_delay_p50'], 16)} {_fmt(r['phrase_delay_p90'], 6)} "
                   f"{_pct_str(r['ack_share'], 6)} {_pct_str(r['tool_phrase_share'], 11)}")
    return out


def _pct_str(x: float | None, width: int) -> str:
    return f"{x * 100:{width - 1}.0f}%" if x is not None else f"{'-':>{width}}"


def describe(data: dict[str, Any]) -> str:
    """``label [backend, caching]`` for a result file (files from before ``backend`` show ``agent``)."""
    b = data.get("backend") or {}
    parts = [b.get("backend") or data.get("agent") or "?"]
    if "prompt_caching" in b:
        parts.append("cache on" if b["prompt_caching"] else "cache off")
    if data.get("cold"):
        parts.append("cold")
    if data.get("tracing") in ("on", "off"):
        parts.append(f"tracing {data['tracing']}")
    if data.get("text_release") in ("stream", "message"):  # absent in files from before the text-release setting
        parts.append(f"text {data['text_release']}")
    return f"{data.get('label', '?')} [{', '.join(parts)}]"


def _fmt(x: Any, width: int = 8) -> str:
    return f"{x:{width}.2f}" if isinstance(x, int | float) and not isinstance(x, bool) else f"{'-':>{width}}"


def _gain(x: Any, y: Any) -> str:
    if isinstance(x, int | float) and isinstance(y, int | float) and x:
        return f"{(y - x) / x * 100:+6.0f}%"
    return f"{'-':>7}"


STEP_METRICS = ("response_p50", "response_p90", "first_audio_p50", "max_silence_p50",
                "phrase_delay_p50", "phrase_delay_p90")
MASKING_METRICS = ("max_silence_p50", "max_silence_p90", "first_audio_p50", "phrase_delay_p50",
                   "phrase_delay_p90", "ack_share", "tool_phrase_share")
OVERALL_METRICS = ("hop_end_p50", "hop_end_p90", "model_latency_p50", "model_ttfb_p50",
                   "overhead_per_model_call_p50", "cache_hit_ratio")


def compare_lines(files: list[dict[str, Any]]) -> list[str]:
    """Two or three result files side by side (harness, Strands without caching, Strands with
    caching, 4a.2): each step's p50 and p90 per file, then each step's gain against the file before
    it, and, for three files, last against first. A metric a file lacks (older files) shows ``-``."""
    if not 2 <= len(files) <= 3:
        raise ValueError("compare takes two or three result files")
    letters = "ABC"[:len(files)]
    pairs = [(i - 1, i) for i in range(1, len(files))] + ([(0, len(files) - 1)] if len(files) == 3 else [])
    out = [f"{x}: {describe(f)}" for x, f in zip(letters, files, strict=True)]
    out.append("(seconds; first audio uses each file's own masking settings)")
    head = f"{'step':30} {'metric':28} " + " ".join(f"{x:>8}" for x in letters) + " " + \
        " ".join(f"{letters[j] + ' vs ' + letters[i]:>7}" for i, j in pairs)
    out.append(head)

    def row(name: str, metric: str, vals: list[Any]) -> None:
        if all(v is None for v in vals):
            return
        out.append(f"{name:30} {metric:28} " + " ".join(_fmt(v) for v in vals) + " " +
                   " ".join(_gain(vals[i], vals[j]) for i, j in pairs))

    for k in files[0].get("summary", {}):
        if all(k in f.get("summary", {}) for f in files):
            for metric in STEP_METRICS:
                row(k, metric, [f["summary"][k].get(metric) for f in files])
    overs = [f.get("overall") or overall(f.get("raw", [])) for f in files]
    for metric in OVERALL_METRICS:
        row("overall", metric, [o.get(metric) for o in overs])
    for metric in MASKING_METRICS:  # each file's own masking_overall; older files lack it
        row("overall (masking)", metric, [(f.get("masking_overall") or {}).get(metric) for f in files])
    for t in USAGE_KEYS:
        row("overall (tokens, total)", t, [(o.get("usage") or {}).get(t) for o in overs])
    return out


def compare(*paths: str) -> None:
    files = [json.loads(Path(p).read_text()) for p in paths]
    print()
    print("\n".join(compare_lines(files)))


def default_label(args: argparse.Namespace) -> str:
    """``strands-nocache`` / ``strands-cache`` for the Strands agent, else the agent name;
    ``-cold`` appended for a cold run, ``-tracing-on`` for a traced one, ``-message`` for
    ``--text-release message`` (``stream``, the default, adds nothing, so existing labels stay the same)."""
    base = args.agent
    if args.agent == "strands":
        base = "strands-cache" if args.prompt_caching == "on" else "strands-nocache"
    return (f"{base}{'-cold' if args.cold else ''}{'-tracing-on' if getattr(args, 'tracing', 'off') == 'on' else ''}"
            f"{'-message' if args.text_release == 'message' else ''}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent", default="strands", choices=sorted(AGENTS))
    ap.add_argument("--label", default=None, help="name for this run, e.g. strands-nocache")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--cold", action="store_true", help="no warm-up (or Strands pre-build) before each run")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--text-release", choices=("stream", "message"), default=None,
                    help="how the agent client releases answer text: stream (as it arrives) or message "
                         "(whole message per hop); the stream-versus-message comparison (default: the "
                         "bridge's, else stream with --model-id, --max-tokens and --thinking)")
    ap.add_argument("--tracing", choices=("on", "off"), default="off",
                    help="on: install the bridge's telemetry and export spans to the traces log group")
    ap.add_argument("--traces-log-group", default=TRACES_LOG_GROUP, help="with --tracing on (default: %(default)s)")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results"))
    ap.add_argument("--compare", nargs="+", metavar="FILE.json", help="two or three result files")
    ap.add_argument("--resummarize", metavar="FILE", help="recompute a saved run's summary (e.g. with masking)")
    g = ap.add_argument_group("masking (summaries, --resummarize, --sweep)")
    g.add_argument("--masking-model", choices=MASKING_MODELS, default="v2",
                   help="v2: scheduled tool phrases (default); legacy: a tool phrase at once")
    g.add_argument("--after-ms", type=int, default=None, help="timed acknowledgement delay (default: the policy's; 0 = off)")
    g.add_argument("--tool-after-ms", type=int, default=None,
                   help=f"policy-wide tool phrase delay (default: the policy's, else {DEFAULT_TOOL_AFTER_MS})")
    g.add_argument("--tool-delay", action="append", metavar="TOOL=MS", help="one tool's phrase delay (repeatable)")
    g.add_argument("--sweep", nargs="+", metavar="KEY=MS,MS", help="with --resummarize: a table over "
                   "every combination, e.g. after_ms=1500,2000,2500 tool_after_ms=0,300,600,900")
    g = ap.add_argument_group("strands")
    g.add_argument("--prompt-caching", choices=("on", "off"), default="off")
    g.add_argument("--gateway-url", default=None, help="default: OwlBankAgent GatewayUrl output")
    g.add_argument("--memory-id", default=None, help="default: OwlBankAgent MemoryId output")
    g.add_argument("--model-id", default=None, help="default: the bridge's (synthesized bridge template)")
    g.add_argument("--max-tokens", type=int, default=None, help="default: the bridge's")
    g.add_argument("--thinking", choices=("adaptive", "off", "default"), default=None, help="default: the bridge's")
    return ap


def prepare_saved(data: dict[str, Any]) -> dict[str, Any]:
    """Fill in step fields a saved file may predate (``timer_eligible``, from the scenario)."""
    for r in data["raw"]:
        for st, sc in zip(r.get("steps", []), SCENARIOS.get(r["scenario"], []), strict=False):
            st.setdefault("timer_eligible", sc.pending is None or sc.pending.get("next") != "await_caller")
    return data


async def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.compare:
        if not 2 <= len(args.compare) <= 3:
            ap.error("--compare takes two or three result files")
        compare(*args.compare)
        return
    try:
        delays = parse_tool_delays(args.tool_delay)
        grid = parse_sweep(args.sweep) if args.sweep else None
        mk = masking_settings(args.masking_model, after_ms=args.after_ms, tool_after_ms=args.tool_after_ms,
                              tool_delays=delays)
    except ValueError as exc:
        ap.error(str(exc))
    if grid is not None and not args.resummarize:
        ap.error("--sweep needs --resummarize FILE")
    if args.resummarize:
        data = json.loads(Path(args.resummarize).read_text())
        prepare_saved(data)
        ok = [r for r in data["raw"] if "error" not in r]
        if grid is not None:
            print(f"\n{data['label']}: masking sweep ({args.masking_model} model, {len(ok)} runs; "
                  "silence, first audio and phrase delay in seconds over all steps; phrase delay over answer steps)")
            print("\n".join(sweep_lines(sweep(ok, grid, model=args.masking_model, after_ms=args.after_ms,
                                               tool_after_ms=args.tool_after_ms, tool_delays=delays))))
            return
        data["summary"] = summarize(ok, mk)
        data["overall"] = overall(data["raw"])
        print_table(data["summary"], f"{data['label']} (re-summarized: {args.masking_model} model, after_ms "
                    f"{mk.after_ms}, tool_after_ms {mk.tool_after_ms}, tool delays (ms) {mk.record()['tool_delay_ms']})",
                    masking_overall(ok, mk))
        return
    tracing = start_tracing(args) if args.tracing == "on" else None  # before the client: it reads the providers
    made = AGENTS[args.agent](args)
    agent, warm = made[0], made[1]
    backend: dict[str, Any] = made[2] if len(made) > 2 else {"backend": args.agent}  # type: ignore[misc]
    # The run's one text release: the client's (the flag, else the synthesized bridge's), resolved
    # once by the agent factory into ``backend``; the label (here) and the result file use it.
    args.text_release = backend.get("text_release") or args.text_release or "stream"
    prefix = SESSION_PREFIX
    names = [s for s in args.scenarios.split(",") if s]
    runs = []
    try:
        for i in range(args.runs):
            for name in names:
                try:
                    runs.append(await run_scenario(agent, warm, name, SCENARIOS[name], cold=args.cold,
                                                   session_prefix=prefix))
                except Exception as exc:  # noqa: BLE001 - a failed run is recorded, the benchmark goes on
                    runs.append({"scenario": name, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "steps": []})
                print(f"run {i + 1}/{args.runs} {name}: " + ", ".join(
                    f"{s.get('response_s')}s" for s in runs[-1]["steps"]) + (f" ERROR {runs[-1]['error']}" if "error" in runs[-1] else ""),
                    flush=True)
    finally:
        aclose = getattr(agent, "aclose", None)
        if aclose is not None:
            await aclose()  # flushes Memory, stops the Gateway MCP client
        if tracing is not None:
            await asyncio.to_thread(flush_tracing, tracing)
    label = args.label or default_label(args)
    path = write_result(Path(args.out), label=label, args=args, backend=backend, runs=runs, masking=mk)
    data = json.loads(path.read_text())
    print_table(data["summary"], f"{label} ({args.runs} runs, {'cold' if args.cold else 'warm'}, {data['errors']} errors)",
                data["masking_overall"])
    o = data["overall"]
    print(f"\nper hop p50 {o['hop_end_p50']} s over {o['hops']} hops; model calls {o['model_calls']}, "
          f"model latency p50 {o['model_latency_p50']}, overhead per model call p50 {o['overhead_per_model_call_p50']}, "
          f"cache hit ratio {o['cache_hit_ratio']}, tokens {o['usage']}")
    print(f"text spoken before a tool call: {o['text_before_tool_hops']} hops, "
          f"{o['text_before_tool_chars']} characters ({args.text_release} release)")
    print(f"answer calls {o['answer_calls_timed']}: first text p50 {o['first_text_p50']} p90 {o['first_text_p90']}, "
          f"first byte to text p50 {o['first_byte_to_text_p50']}, saving from releasing the first sentence "
          f"p50 {o['sentence_release_saving_p50']} p90 {o['sentence_release_saving_p90']}, "
          f"from streaming the text p50 {o['stream_release_saving_p50']} p90 {o['stream_release_saving_p90']}")
    print(f"\nsaved {_rel(path)}")


def write_result(out: Path, *, label: str, args: argparse.Namespace, backend: dict[str, Any],
                 runs: list[dict[str, Any]], masking: Masking | None = None) -> Path:
    mk = masking or masking_settings()
    ok = [r for r in runs if "error" not in r]
    summary = summarize(ok, mk)
    result = {"label": label, "agent": args.agent, "cold": args.cold, "tracing": getattr(args, "tracing", "off"),
              "text_release": args.text_release, "runs": args.runs, "backend": backend,
              "masking": mk.record(), "masking_overall": masking_overall(ok, mk),
              "at": datetime.now(UTC).isoformat(timespec="seconds"), "summary": summary,
              "overall": overall(runs), "raw": runs, "errors": sum(1 for r in runs if "error" in r)}
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{label}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(result, indent=1))
    return path


if __name__ == "__main__":
    asyncio.run(main())
