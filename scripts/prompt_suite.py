"""scripts/prompt_suite.py: the per-model prompt validation suite (model failover).

Runs the cases of ``data/prompt_suite.json`` (``en-US`` and ``es-US``; no case covers a result with
``next`` = ``await_caller``, which no bridge path sends the model) against each model
of the list, on Bedrock, through the real agent client, and writes a validation record
(``data/prompt_validation.json``) for each model and language whose every run of every case
passed (all runs, no tolerance), with the entry's parameters' hash
(:func:`~owlbank.prompt_validation.params_hash`).

Each run is one caller turn on a fresh :class:`StrandsAgentClient` with a one-entry list (the model
under test with its own parameters: thinking, max tokens, text release, prompt caching), so failover
is off and the request is the bridge's (thinking fields, text release, cache points):

* the system prompt is :func:`owlbank.agent_prompt.render` of the model's own catalog text in the
  case's language, for the verified fixture customer (Thomas Doe, ``C-1001`` in
  ``data/fixtures.json``; profile by the bridge's ``_profile_text``, no Memory, no web activity,
  unless the case's ``context`` says otherwise: unverified, a memory recap, a web block), sent as
  the bridge sends it: the static part as the model's system prompt, the per-call part and the
  bridge's clock line (at a fixed time) as the suffix after the cache point;
* the case's history is injected through the client's session manager seam (:class:`SeededHistory`,
  no Memory). A history ``toolResult`` may give its content as ``{"json": ...}``: it becomes the JSON
  text the bridge sends (:meth:`ToolResult.to_block`), a bank tool's result wrapped as the bridge
  wraps it (``untrusted.wrap_result``);
* the Gateway tools are fakes with the deployed specs (``infra/owlbank_infra/contract.py``
  ``level0_tool_definitions``) that answer from the case's ``tool_results`` (no Gateway call);
* the turn: the caller's words, or, with ``caller`` null, the model continues from the history (its
  last message is a tool result, as when the bridge resumes with one). An inline tool call with a
  scripted result in ``tool_results`` gets it and the turn goes on; a hand-off tool
  (``get_balance``, ``authenticate``, ``escalate``, ``end_call``) or an unscripted call ends the
  run, as a hand-off ends the session.

The outcome is the text the caller would hear (after the ``<silent/>`` filter), whether the
model wrote ``<silent/>``, the tool calls with their arguments in order, whether heard text came
before the first tool call, and the last stop reason; :func:`owlbank.prompt_validation.failed`
checks it against the case's ``expect``.

Errors: a throttle or 5xx (:data:`RETRY_CODES`) is retried with backoff and never counts as a
failed run. A stop the model's own output causes (:data:`MODEL_STOP_CODES`: the answer reached
``max_tokens``; in a call the bridge escalates) is a failed run, not an error.
Any other error, or one still failing after ``--attempts``, stops the suite with no record (exit 2).

Output: a table of passes per case, language and model, then each failed run (case, model,
failed checks, the outcome: synthetic text only) and the records written. Exit 0 when every model
and language passed, 1 when any failed, 2 on a stopping error. ``--case`` (a subset), ``--runs``
below the case file's ``runs`` (preflight and the seed refuse such a record anyway) and
``--no-write`` never write records.

Cost of one full run for the default list: 14 cases x 2 languages x 5 runs x 3 models =
420 runs; the two cases with a scripted inline result take two model calls (that result, then the
answer), the others one: 480 model calls. Each request is about 3 thousand input tokens (system prompt about 1.2
thousand, tool specs about 1.4 thousand, history; measured offline from the requests, chars / 4),
and up to a few hundred output tokens: about 1.5 million input tokens a run, about half a million
per model. The default list has ``prompt_caching`` on, so cache reads make it cheaper still. At
Bedrock's us-east-1 on-demand input prices before any cache read ($0.11, $1.10 and $2.20 per million)
that is about $0.06 for Haiku 5.5, $0.55 for Haiku 4.5 and $1.10 for Sonnet 5.5, so about $1.70.
The run's measured token totals per model go into the record file's header (``last_run``).

Usage (repository root; fixture data only, no caller data)::

    AWS_PROFILE=<dev profile> uv run python scripts/prompt_suite.py                 # every model
    AWS_PROFILE=<dev profile> uv run python scripts/prompt_suite.py --model haiku-4-5 --case <id>
    AWS_PROFILE=<dev profile> uv run python scripts/prompt_suite.py --models "$OWLBANK_MODELS"
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import datetime
import json
import os
import random
import sys
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):  # run as a file: make `scripts`, `infra` and `src` importable
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from strands import tool  # noqa: E402
from strands.models.model import Model  # noqa: E402
from strands.session.session_manager import SessionManager  # noqa: E402
from strands.types.tools import ToolContext  # noqa: E402

from infra.owlbank_infra.config import InfraConfig  # noqa: E402 - the default list and its parameters
from infra.owlbank_infra.contract import level0_tool_definitions  # noqa: E402 - the Gateway's own specs
from owlbank import agent_prompt, untrusted  # noqa: E402
from owlbank.catalog import CatalogKeyError, PromptCatalog  # noqa: E402
from owlbank.prompt_validation import (  # noqa: E402
    PARAM_KEYS,
    Case,
    CaseTally,
    Check,
    Outcome,
    Record,
    SchemaError,
    Suite,
    ToolCall,
    ValidationFile,
    failed,
    load_records,
    load_suite,
    params_hash,
    save_records,
)
from owlbank.tools_contract import (  # noqa: E402
    BANK_TOOLS,
    GATEWAY_LEVEL0_TOOLS,
    HANDOFF_TOOLS,
    INLINE_TOOLS,
    gateway_tool_full_name,
)
from tac_bridge import agent_client as ac  # noqa: E402
from tac_bridge.agent_client_strands import (  # noqa: E402
    ALLOWED_GATEWAY_TOOLS,
    ModelEntry,
    StrandsAgentClient,
    StrandsSettings,
    default_model_factory,
)
from tac_bridge.app import MODELS_ENV, models_from_env  # noqa: E402
from tac_bridge.bridge import _profile_text  # noqa: E402 - the bridge's own profile line
from tac_bridge.clock import local_now_line, zone  # noqa: E402

SUITE_PATH: Final = ROOT / "data" / "prompt_suite.json"
RECORDS_PATH: Final = ROOT / "data" / "prompt_validation.json"
CATALOG_PATH: Final = ROOT / "data" / "prompt_catalog.json"
FIXTURES_PATH: Final = ROOT / "data" / "fixtures.json"
#: The verified fixture customer every case runs for (Thomas Doe).
FIXTURE_CUSTOMER_ID: Final = "C-1001"
#: ``BridgeConfig.timezone``'s default, and a fixed time just after the fixture balances' ``asOf``.
BANK_TIMEZONE: Final = "America/New_York"
CLOCK_AT: Final = datetime.datetime(2026, 10, 10, 15, 0, tzinfo=datetime.UTC)
DEFAULT_REGION: Final = "us-east-1"
DEFAULT_CONCURRENCY: Final = 6
#: Tries of one run when Bedrock throttles or fails server-side, and the backoff between them.
DEFAULT_ATTEMPTS: Final = 6
BACKOFF_BASE_S: Final = 2.0
BACKOFF_MAX_S: Final = 60.0
#: Error codes retried (throttles and 5xx); any other error stops the suite.
RETRY_CODES: Final = frozenset({
    "ThrottlingException", "ServiceQuotaExceededException", "ModelNotReadyException",
    "InternalServerException", "ServiceUnavailableException", "ModelErrorException",
    "ModelTimeoutException",
})
#: Error codes the model's own output causes: a failed run (stop reason the code), never retried.
MODEL_STOP_CODES: Final = frozenset({"max_tokens"})
#: Agent turns in one run (the caller's turn plus the scripted inline results); more is a loop.
MAX_HOPS: Final = 6
SESSION_PREFIX: Final = "prompt-suite"
_GATEWAY_PREFIX: Final = gateway_tool_full_name("")
_KNOWN_TOOLS: Final = frozenset(INLINE_TOOLS) | frozenset(GATEWAY_LEVEL0_TOOLS)
#: Tool names a history ``toolUse`` may carry: the inline tools and the Gateway's full names.
_HISTORY_TOOLS: Final = frozenset(INLINE_TOOLS) | frozenset(ALLOWED_GATEWAY_TOOLS)
_USAGE_KEYS: Final = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


class SuiteError(RuntimeError):
    """An error that stops the suite with no record: not a throttle or 5xx, or one still failing
    after every attempt. The message names the model, case and language, and the error code."""


# --------------------------------------------------------------------------- the case's inputs


@dataclass(frozen=True, slots=True)
class Prepared:
    """A case ready to run: the history as Converse messages in the bridge's form, and the tool-use
    IDs its last message answers (the continuation of a ``caller``-null turn)."""

    case: Case
    history: tuple[Mapping[str, Any], ...]
    resume_ids: tuple[str, ...]


def result_status(result: Mapping[str, Any]) -> Literal["success", "error"]:
    """The ``toolResult`` status the bridge gives a result: ``error`` for a bank error or a failed
    envelope, else ``success``."""
    return "error" if "error" in result or result.get("status") == "failed" else "success"


def model_view(tool_name: str, result: Any) -> Any:
    """A tool result as the model gets it from the bridge: a bank tool's wrapped
    (``untrusted.wrap_result``), anything else as is."""
    return untrusted.wrap_result(result) if tool_name in BANK_TOOLS else result


def _block(block: Any, uses: dict[str, str], where: str) -> dict[str, Any]:
    if not isinstance(block, Mapping) or len(block) != 1:
        raise SchemaError(f"{where}: a content block has exactly one key")
    kind, body = next(iter(block.items()))
    if kind == "text":
        if not isinstance(body, str) or not body.strip():
            raise SchemaError(f"{where}: text must be a non-empty string")
        return {"text": body}
    if kind == "toolUse":
        if not (isinstance(body, Mapping) and set(body) == {"toolUseId", "name", "input"}
                and isinstance(body["toolUseId"], str) and body["name"] in _HISTORY_TOOLS
                and isinstance(body["input"], Mapping)):
            raise SchemaError(f"{where}: toolUse is {{toolUseId, name (a known tool), input: {{...}}}}")
        uses[body["toolUseId"]] = body["name"]
        return {"toolUse": copy.deepcopy(dict(body))}
    if kind == "toolResult":
        if not (isinstance(body, Mapping) and set(body) == {"toolUseId", "status", "content"}
                and body["status"] in ("success", "error") and isinstance(body["content"], list)
                and len(body["content"]) == 1 and isinstance(body["content"][0], Mapping)):
            raise SchemaError(f"{where}: toolResult is {{toolUseId, status, content: [one block]}}")
        name = uses.pop(body["toolUseId"], None)
        if name is None:
            raise SchemaError(f"{where}: toolResult for a tool use not made just before it")
        (ckind, cbody), = body["content"][0].items()
        if ckind == "json":
            content: Any = model_view(name, cbody)
        elif ckind == "text" and isinstance(cbody, str):
            content = cbody
        else:
            raise SchemaError(f"{where}: toolResult content is {{json: ...}} or {{text: ...}}")
        return ac.ToolResult(body["toolUseId"], content, body["status"]).to_block()
    raise SchemaError(f"{where}: unknown content block {kind!r}")


def prepare(case: Case) -> Prepared:
    """The case's history in the form the agent's history holds it (:class:`Prepared`). Raises
    :class:`SchemaError` for a block the runner can't send: a tool use left unanswered before the
    next assistant message, a result for an unknown tool use, or ``caller`` null on a history that
    doesn't end with tool results."""
    uses: dict[str, str] = {}
    history: list[dict[str, Any]] = []
    for i, m in enumerate(case.history):
        where = f"{case.id} {case.lang} history[{i}]"
        if m["role"] == "assistant" and uses:
            raise SchemaError(f"{where}: an earlier tool use has no result")
        history.append({"role": m["role"], "content": [_block(b, uses, where) for b in m["content"]]})
    if uses:
        raise SchemaError(f"{case.id} {case.lang}: a tool use has no result")
    tail = history[-1] if history else {"role": "", "content": []}
    resume = tuple(b["toolResult"]["toolUseId"] for b in tail["content"] if "toolResult" in b) \
        if tail["role"] == "user" else ()
    if case.caller is None and not resume:
        raise SchemaError(f"{case.id} {case.lang}: caller null needs a history ending with tool results")
    if case.caller is not None and not case.caller.strip():
        raise SchemaError(f"{case.id} {case.lang}: caller must be words or null")
    unknown = set(case.tool_results) - _KNOWN_TOOLS
    if unknown:
        raise SchemaError(f"{case.id} {case.lang}: tool_results for unknown tools {sorted(unknown)}")
    return Prepared(case=case, history=tuple(history), resume_ids=resume)


class SeededHistory(SessionManager):
    """A session manager that only starts the agent on the case's history (the client's seam for
    Memory: no Memory here). Nothing is stored."""

    def __init__(self, history: Sequence[Mapping[str, Any]]) -> None:
        self.session_id = uuid.uuid4().hex
        self._history = history

    def initialize(self, agent: Any, **kwargs: Any) -> None:
        agent.messages = copy.deepcopy(list(self._history))

    def append_message(self, message: Any, agent: Any, **kwargs: Any) -> None:
        return None

    def redact_latest_message(self, redact_message: Any, agent: Any, **kwargs: Any) -> None:
        return None

    def sync_agent(self, agent: Any, **kwargs: Any) -> None:
        return None


def _gateway_tool(name: str, description: str, schema: Mapping[str, Any], results: Mapping[str, Any]) -> Any:
    @tool(name=gateway_tool_full_name(name), description=description, inputSchema={"json": dict(schema)},
          context=True)
    def _gw(tool_context: ToolContext) -> str:
        return json.dumps(results.get(name, {"error": "UNAVAILABLE", "tool": name}), ensure_ascii=False)

    return _gw


class ScriptedGateway:
    """The :class:`GatewayToolProvider` of a run: the Level-0 tools with the Gateway's specs,
    answering from the case's ``tool_results`` (``UNAVAILABLE`` when the case scripts none)."""

    def __init__(self, results: Mapping[str, Any]) -> None:
        self._tools = [_gateway_tool(d["name"], d["description"], d["input_schema"], results)
                       for d in level0_tool_definitions()]

    def tools(self) -> Sequence[Any]:
        return list(self._tools)


# --------------------------------------------------------------------------- one run


@dataclass
class _Collector:
    """The run's events, folded into an :class:`Outcome`. Text goes through the bridge's
    :class:`ac.ReasoningFilter`, flushed per hop and model message (``MessageEnd``) as the bridge does,
    so ``text`` is what the caller would hear (before the ``<silent/>`` filter)."""

    text: list[str] = field(default_factory=list)
    reasoning: ac.ReasoningFilter = field(default_factory=ac.ReasoningFilter)
    calls: list[ToolCall] = field(default_factory=list)
    before_tool: bool = False
    stop: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    def tool(self, name: str, args: Mapping[str, Any]) -> None:
        if not self.calls and ac.strip_silent("".join(self.text)).strip():
            self.before_tool = True  # heard text released in an earlier message
        self.calls.append(ToolCall(name, args))

    def feed(self, ev: Any) -> None:
        if isinstance(ev, ac.TextDelta):
            self.text.append(self.reasoning.feed(ev.text))
            return
        if isinstance(ev, (ac.MessageEnd, ac.ToolCall)):
            self.hop_end()  # the model message ended, as in the bridge
        if isinstance(ev, ac.TextBeforeTool):
            self.before_tool = True
        elif isinstance(ev, ac.ToolCall):
            self.tool(ev.name, ev.input)
        elif isinstance(ev, ac.ToolStarted):
            self.tool(ev.name.removeprefix(_GATEWAY_PREFIX), ev.input)
        elif isinstance(ev, ac.TurnEnd):
            self.stop = ev.stop_reason
            for k, v in (ev.usage or {}).items():
                self.usage[k] = self.usage.get(k, 0) + v

    def hop_end(self) -> None:
        """A model message or the hop ended: an unclosed reasoning block is dropped, a held tail released."""
        self.text.append(self.reasoning.flush())

    def outcome(self) -> Outcome:
        self.hop_end()  # a run that raised mid-hop: nothing held is lost or spoken twice
        raw = "".join(self.text)
        return Outcome(text=ac.strip_silent(raw), silent=ac.SILENT_MARK in raw, tool_calls=tuple(self.calls),
                       text_before_tool=self.before_tool and bool(self.calls), stop_reason=self.stop,
                       reasoning_removed=self.reasoning.blocks)


@dataclass(frozen=True, slots=True)
class SuiteModel:
    """One model under test: its entry (with its own parameters), its prompt key and its Strands
    model (built once, shared by its runs as the bridge shares one per entry)."""

    entry: ModelEntry
    key: str
    model: Model


_WEB_BLOCKS: Final = {"offer": agent_prompt.WEB_ACTIVITY_KEY, "context": agent_prompt.WEB_ACTIVITY_CONTEXT_KEY}


def case_prompt(catalog: PromptCatalog, key: str, case: Case, customer: Mapping[str, Any]) -> agent_prompt.RenderedPrompt:
    """The case's system prompt for model ``key``, as the bridge renders it: the
    fixture customer verified with no memory and no web block, unless the case's ``context`` says
    otherwise (unverified; a memory recap wrapped as the bridge wraps it; a web block)."""
    ctx = case.context or {}
    caller: agent_prompt.CallerFacts | None = None
    if ctx.get("verified", True):
        memory = untrusted.wrap("memory", ctx["memory"]) if ctx.get("memory") else None
        caller = agent_prompt.CallerFacts(str(customer.get("first_name") or ""), _profile_text(customer, None), memory)
    return agent_prompt.render(catalog, key, case.lang, caller=caller, web_block=_WEB_BLOCKS.get(ctx.get("web_block", "")))


async def run_once(m: SuiteModel, prepared: Prepared, prompt: agent_prompt.RenderedPrompt, *, region: str) -> tuple[Outcome, dict[str, int]]:
    """One run of a case on one model: a fresh one-entry client, the history, the turn and its
    scripted results. Returns the outcome and the run's token usage; raises :class:`ac.AgentError`."""
    case = prepared.case
    client = StrandsAgentClient(
        StrandsSettings(models=(m.entry,), region=region),
        model_factory=lambda _s, _sid: m.model,
        gateway_tools=ScriptedGateway(case.tool_results),
        session_manager_factory=lambda _sid, _mode: SeededHistory(prepared.history))
    sid = f"{SESSION_PREFIX}-{m.key}-{case.id}-{case.lang}-{uuid.uuid4().hex[:8]}"
    clock = local_now_line(CLOCK_AT, zone(BANK_TIMEZONE), case.lang)
    collector = _Collector()
    user_text = case.caller
    results = [ac.ToolResult(tid, {}) for tid in prepared.resume_ids] if user_text is None else []
    try:
        await client.begin_session(sid, history="new")
        for _ in range(MAX_HOPS):
            following: list[ac.ToolResult] = []
            ended = False
            async for ev in client.turn(sid, user_text=user_text, tool_results=results,
                                        system_prompts={m.key: prompt.static},
                                        system_suffix=f"{prompt.per_call}\n{clock}"):
                collector.feed(ev)
                if isinstance(ev, ac.ToolCall):
                    if ev.name in HANDOFF_TOOLS or ev.name not in case.tool_results:
                        ended = True  # a hand-off, or a call the case doesn't script: the run ends
                    else:
                        scripted = case.tool_results[ev.name]
                        following.append(ac.ToolResult(ev.tool_use_id, model_view(ev.name, scripted),
                                                       result_status(scripted)))
            collector.hop_end()
            if ended or not following:
                break
            user_text, results = None, following
    finally:
        await client.aclose()
    return collector.outcome(), collector.usage


# --------------------------------------------------------------------------- the suite


@dataclass
class Tally:
    """Runs and passes of one case for one model and language, and the failed runs."""

    runs: int = 0
    passes: int = 0
    failures: list[tuple[Outcome, list[Check]]] = field(default_factory=list)


@dataclass
class SuiteResult:
    """``tallies[(model key, lang, case id)]``, token usage per model key, and whether every case of
    the suite ran (no ``--case`` subset): only then may records be written."""

    models: tuple[SuiteModel, ...]
    langs: tuple[str, ...]
    case_ids: tuple[str, ...]
    tallies: dict[tuple[str, str, str], Tally]
    usage: dict[str, dict[str, int]]
    complete: bool
    runs: int

    def passed(self, key: str, lang: str) -> bool:
        ts = [t for (k, la, _), t in self.tallies.items() if k == key and la == lang]
        return bool(ts) and all(t.runs >= 1 and t.passes == t.runs for t in ts)


Sleep = Callable[[float], Awaitable[None]]


def _code(exc: BaseException) -> str:
    return getattr(exc, "code", None) or type(exc).__name__


def _prompt_key(m: SuiteModel, case: Case) -> tuple[str, str, frozenset[tuple[str, Any]]]:
    """What a case's rendered prompt depends on: the model, the language and the context, so cases
    that share a context share one render."""
    return (m.key, case.lang, frozenset((case.context or {}).items()))


async def run_suite(suite: Suite, models: Sequence[SuiteModel], catalog: PromptCatalog, *,
                    langs: Sequence[str] | None = None, case_ids: Sequence[str] | None = None,
                    runs: int | None = None, concurrency: int = DEFAULT_CONCURRENCY,
                    attempts: int = DEFAULT_ATTEMPTS, region: str = DEFAULT_REGION,
                    sleep: Sleep = asyncio.sleep, customer: Mapping[str, Any] | None = None,
                    progress: Callable[[str], None] | None = None) -> SuiteResult:
    """Every run of every selected case on every model, at most ``concurrency`` at a time. Raises
    :class:`SuiteError` (the other runs cancelled) for an error that isn't retried or still fails
    after ``attempts``; a retried error never counts as a run."""
    runs = suite.runs if runs is None else runs
    langs = tuple(dict.fromkeys(c.lang for c in suite.cases)) if langs is None else tuple(langs)
    cases = [c for c in suite.cases if c.lang in langs and (case_ids is None or c.id in case_ids)]
    if not cases:
        raise SuiteError("no case selected")
    prepared = {(c.id, c.lang): prepare(c) for c in cases}
    customer = customer if customer is not None else fixture_customer()
    prompts: dict[tuple[str, str, frozenset[tuple[str, Any]]], agent_prompt.RenderedPrompt] = {}
    for m in models:
        for c in cases:
            key = _prompt_key(m, c)
            if key in prompts:
                continue
            try:
                prompts[key] = case_prompt(catalog, m.key, c, customer)
            except CatalogKeyError as exc:
                raise SuiteError(f"{m.key} {c.lang}: no {agent_prompt.system_key(m.key)} text in the catalog") from exc
    tallies = {(m.key, c.lang, c.id): Tally() for m in models for c in cases}
    usage: dict[str, dict[str, int]] = {m.key: dict.fromkeys(_USAGE_KEYS, 0) | {"runs": 0} for m in models}
    gate = asyncio.Semaphore(concurrency)

    async def one(m: SuiteModel, case: Case) -> None:
        async with gate:
            stopped = False
            for attempt in range(1, attempts + 1):
                try:
                    outcome, used = await run_once(m, prepared[(case.id, case.lang)],
                                                   prompts[_prompt_key(m, case)], region=region)
                    break
                except Exception as exc:  # noqa: BLE001 - classified below; never a failed run
                    code = _code(exc)
                    if isinstance(exc, ac.AgentError) and code in MODEL_STOP_CODES:
                        outcome, used, stopped = Outcome(text="", silent=False, stop_reason=code), {}, True
                        break
                    retry = isinstance(exc, ac.AgentError) and code in RETRY_CODES
                    if not retry or attempt == attempts:
                        why = "after retries" if retry else "not retried"
                        raise SuiteError(f"{m.key} {case.id} {case.lang}: {code} ({why}, attempt {attempt}); "
                                         "no record written") from exc
                    delay = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** (attempt - 1)) * (0.5 + random.random() / 2)  # noqa: S311 - jitter
                    if progress is not None:
                        progress(f"retry {m.key} {case.id} {case.lang}: {code}, attempt {attempt}, "
                                 f"waiting {delay:.1f} s")
                    await sleep(delay)
        bad = list(case.expect) if stopped else failed(case.expect, outcome)
        t = tallies[(m.key, case.lang, case.id)]
        t.runs += 1
        t.passes += not bad
        if bad:
            t.failures.append((outcome, bad))
        u = usage[m.key]
        u["runs"] += 1
        for k in _USAGE_KEYS:
            u[k] += used.get(k, 0)

    try:
        async with asyncio.TaskGroup() as group:
            for m in models:
                for case in cases:
                    for _ in range(runs):
                        group.create_task(one(m, case))
    except BaseExceptionGroup as eg:
        stop = eg.subgroup(SuiteError)
        if stop is None:
            raise
        raise _first(stop) from None
    return SuiteResult(models=tuple(models), langs=langs, case_ids=tuple(dict.fromkeys(c.id for c in cases)),
                       tallies=tallies, usage=usage,
                       complete=case_ids is None or {c.id for c in suite.cases} <= set(case_ids), runs=runs)


def _first(group: BaseExceptionGroup) -> BaseException:
    exc: BaseException = group
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return exc


def entry_params(entry: ModelEntry) -> dict[str, Any]:
    """The entry's model parameters a record is validated with: the ``OWLBANK_MODELS`` fields
    of :data:`~owlbank.prompt_validation.PARAM_KEYS`."""
    return {k: getattr(entry, k) for k in PARAM_KEYS}


def records_for(result: SuiteResult, suite: Suite, catalog: PromptCatalog, date: str) -> list[Record]:
    """One :class:`Record` per model and language that passed every run of every case. None
    for a ``--case`` subset or fewer runs than the case file's (a full run is every one of them)."""
    if not result.complete or result.runs < suite.runs:
        return []
    out = []
    for m in result.models:
        for lang in result.langs:
            if not result.passed(m.key, lang):
                continue
            cases = {cid: CaseTally(runs=t.runs, passes=t.passes)
                     for (k, la, cid), t in result.tallies.items() if k == m.key and la == lang}
            out.append(Record(model_key=m.key, lang=lang, prompt_hash=agent_prompt.prompt_hash(catalog, m.key, lang),
                              suite_hash=suite.hash, params_hash=params_hash(entry_params(m.entry)),
                              model_id=m.entry.id, date=date, cases=cases))
    return out


def write_records(path: Path, records: Sequence[Record], result: SuiteResult, date: str) -> None:
    """Add ``records`` to the record file (replacing each model and language's earlier record) and
    put the run's token totals per model in its header (``last_run``)."""
    if not records:
        return
    current = load_records(path) if path.exists() else ValidationFile.empty()
    header = json.loads(json.dumps(dict(current.header)))
    last = header.setdefault("last_run", {})
    for key in dict.fromkeys(r.model_key for r in records):
        last[key] = {"date": date, **result.usage[key]}
    updated = ValidationFile(header=header, records=current.records)
    for r in records:
        updated = updated.add(r)
    save_records(path, updated)


# --------------------------------------------------------------------------- output


def table(result: SuiteResult) -> str:
    """Passes per case and language (rows) and model (columns), ``passes/runs``."""
    keys = [m.key for m in result.models]
    rows = [["case", "lang", *keys]]
    for lang in result.langs:
        for cid in result.case_ids:
            cells = []
            for k in keys:
                t = result.tallies.get((k, lang, cid))
                cells.append(f"{t.passes}/{t.runs}" if t is not None else "-")
            rows.append([cid, lang, *cells])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows)


def failure_lines(result: SuiteResult, limit: int = 300) -> list[str]:
    out = []
    for (key, lang, cid), t in result.tallies.items():
        for outcome, bad in t.failures:
            calls = ", ".join(f"{c.name}({json.dumps(dict(c.args), ensure_ascii=False)})" for c in outcome.tool_calls)
            text = outcome.text if len(outcome.text) <= limit else outcome.text[:limit] + "..."
            out.append(f"FAIL {cid} {lang} {key}: {', '.join(c.kind for c in bad)} | silent={outcome.silent} "
                       f"stop={outcome.stop_reason} tools=[{calls}] text={text!r}")
    return out


# --------------------------------------------------------------------------- inputs and main


def fixture_customer(path: Path = FIXTURES_PATH) -> Mapping[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return next(c for c in doc["customers"] if c.get("customer_id") == FIXTURE_CUSTOMER_ID)


def default_entries() -> tuple[ModelEntry, ...]:
    """The default list with each model's own parameters, as synth puts it in ``OWLBANK_MODELS``."""
    return parse_models(json.dumps([m.as_env() for m in InfraConfig().models]))


def parse_models(raw: str) -> tuple[ModelEntry, ...]:
    """An ``OWLBANK_MODELS`` value (the bridge's parser and rules)."""
    try:
        return models_from_env({MODELS_ENV: raw})
    except RuntimeError as exc:
        raise SystemExit(f"--models: {exc}") from None


ModelFactory = Callable[[StrandsSettings], Model]


def build_models(entries: Iterable[ModelEntry], *, region: str,
                 model_factory: ModelFactory = default_model_factory) -> tuple[SuiteModel, ...]:
    out = []
    for e in entries:
        out.append(SuiteModel(entry=e, key=agent_prompt.model_key(e.id),
                          model=model_factory(StrandsSettings(models=(e,), region=region))))
    return tuple(out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--models", default=None,
                    help=f"an {MODELS_ENV} value (JSON list of entries); default: the default list with "
                         "each model's synth parameters")
    ap.add_argument("--model", action="append", default=None,
                    help="only this model key (haiku-5-5, haiku-4-5, sonnet-5-5); repeatable")
    ap.add_argument("--lang", action="append", default=None, help="only this language; repeatable")
    ap.add_argument("--case", action="append", default=None,
                    help="only this case ID; repeatable (writes no record)")
    ap.add_argument("--runs", type=int, default=None, help="runs per case (default: the case file's, 5)")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help=f"runs in flight (default {DEFAULT_CONCURRENCY})")
    ap.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS,
                    help=f"tries of a run on a throttle or 5xx (default {DEFAULT_ATTEMPTS})")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", DEFAULT_REGION))
    ap.add_argument("--suite", type=Path, default=SUITE_PATH)
    ap.add_argument("--records", type=Path, default=RECORDS_PATH)
    ap.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    ap.add_argument("--no-write", action="store_true", help="print the results, write no record")
    return ap


def main(argv: list[str] | None = None, *, model_factory: ModelFactory = default_model_factory,
         sleep: Sleep = asyncio.sleep, today: Callable[[], datetime.date] = datetime.date.today) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    for name in ("runs", "concurrency", "attempts"):
        value = getattr(args, name)
        if value is not None and value < 1:
            ap.error(f"--{name} must be at least 1")
    try:
        suite = load_suite(args.suite)
        for c in suite.cases:
            prepare(c)
    except (OSError, ValueError) as exc:
        print(f"case file {args.suite}: {exc}", file=sys.stderr)
        return 2
    entries = parse_models(args.models) if args.models else default_entries()
    if args.model:
        unknown = set(args.model) - {agent_prompt.model_key(e.id) for e in entries}
        if unknown:
            ap.error(f"--model {sorted(unknown)} not in the list")
        entries = tuple(e for e in entries if agent_prompt.model_key(e.id) in args.model)
    catalog = PromptCatalog.from_dict(json.loads(args.catalog.read_text(encoding="utf-8")))
    models = build_models(entries, region=args.region, model_factory=model_factory)
    print(f"suite {suite.hash[:12]} runs {args.runs or suite.runs} concurrency {args.concurrency} region "
          f"{args.region}; models " + ", ".join(f"{m.key} ({m.entry.id}, thinking {m.entry.thinking}, "
                                                f"{m.entry.text_release}, caching "
                                                f"{'on' if m.entry.prompt_caching else 'off'})" for m in models),
          flush=True)
    try:
        result = asyncio.run(run_suite(suite, models, catalog, langs=args.lang, case_ids=args.case,
                                       runs=args.runs, concurrency=args.concurrency, attempts=args.attempts,
                                       region=args.region, sleep=sleep,
                                       progress=lambda line: print(line, file=sys.stderr, flush=True)))
    except SuiteError as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    print(table(result))
    for line in failure_lines(result):
        print(line)
    for key, u in result.usage.items():
        print(f"tokens {key}: " + ", ".join(f"{k} {v}" for k, v in u.items()))
    date = today().isoformat()
    records = records_for(result, suite, catalog, date)
    if args.no_write or not result.complete or result.runs < suite.runs:
        why = ("--no-write" if args.no_write else "a --case subset" if not result.complete
               else f"--runs {result.runs} is below the case file's {suite.runs}")
        print(f"no record written ({why})")
    else:
        write_records(args.records, records, result, date)
        for r in records:
            print(f"record {r.model_key} {r.lang}: prompt {r.prompt_hash[:12]} suite {r.suite_hash[:12]} "
                  f"params {r.params_hash[:12]}")
    ok = all(result.passed(m.key, lang) for m in result.models for lang in result.langs)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
