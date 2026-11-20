"""``StrandsAgentClient``: the bridge's :class:`~tac_bridge.agent_client.AgentClient` on a Strands loop.

One Strands ``Agent`` per call, cached in the process (a bounded LRU with idle eviction). It
calls Claude on Bedrock directly through ``BedrockModel`` and offers the model exactly:

* the **inline tools** (``INLINE_TOOLS``, specs from ``tools_contract``). Each raises a Strands
  interrupt, so the loop pauses and the client yields a :class:`ToolCall`; the bridge runs the
  tool and the next :meth:`StrandsAgentClient.turn` resumes the agent with an ``interruptResponse``.
  The tool body has no side effects: Strands runs it again on resume.
* the **approved Gateway tools** (``GATEWAY_LEVEL0_TOOLS`` by full name), already listed by a
  process-wide MCP client (:func:`make_gateway_tool_provider`). They run inside the loop.
  Anything else a Gateway lists never reaches the model.

Events follow the ``AgentClient`` contract. How text is released is
each model entry's :attr:`ModelEntry.text_release` (:class:`_Translator`):
``stream`` (the default) yields each text chunk as :class:`TextDelta` at once until a tool-use
block starts in the message; if text the caller hears came before it, one :class:`TextBeforeTool`
follows (its length and the tool), and text after the tool-use block is reported as
:class:`TextSuppressed`. ``message`` holds text per model message and releases it at its
``messageStop``, or reports it as :class:`TextSuppressed` if the message also uses a tool.
``stream`` needs thinking ``adaptive`` or ``default`` (:class:`ModelEntry`). Exactly one
:class:`TurnEnd` comes last, with usage including the cache token counts. Failures raise
:class:`AgentError` (:func:`to_agent_error`).

History: AgentCore Memory through
:class:`_NewestStateSessionManager` (batched, asynchronous writes, actor ``owlbank-voice-strands``),
flushed when each invocation ends. :meth:`StrandsAgentClient.begin_session` takes the bridge's
``history`` decision: ``new`` builds an empty agent without reading Memory, ``cached`` keeps the
cached agent (or reloads if it was evicted), ``reload`` rebuilds from Memory. A failed read is a
retryable ``history_unavailable`` error, never an empty history. Hand-off ``ToolCall``\\ s are
emitted only after the flush; a failed flush fails the turn (``history_write_failed``).
A tool result for a tool use the agent never made, on an agent not rebuilt from Memory in the current
session, rebuilds it from Memory once (``agent_history_reload``, reason ``unknown_tool_use_recovery``)
before the turn fails with the non-retryable ``unknown_tool_use``.
Without a Memory ID (and no injected factory) history stays in memory.

Blank caller words are no words (no model call when nothing is pending). A turn that resends
the input of a turn that failed after recording it (the bridge's retry of a failed hop) continues
from the history instead of adding the input again (:func:`retry_continues`). :meth:`prewarm`
builds the Bedrock and Memory clients at start-up, and OpenTelemetry's harmless "Failed to detach
context" error on a closed stream is filtered (:class:`OtelDetachNoise`).

Model failover: :attr:`StrandsSettings.models` is an
ordered list of one to three :class:`ModelEntry` (each with its own thinking, max tokens, text
release and prompt caching). With two or more, each entry's model runs as a
:class:`~tac_bridge.guarded_model.GuardedModel` (first-token deadline, history filter); each turn
starts on the decision state's choice (:class:`~owlbank.failover.FailoverState`, the fleet verdict
through :meth:`StrandsAgentClient.use_fleet_view`) and keeps it for its tool loop; a call that fails
before any output moves the turn to the next entry by continuing from the history, so the caller's
message is never sent or stored twice. Every model call's outcome and time to first output feed the
decision state and, when the bridge has the fleet picture, the fleet reporter
(:meth:`StrandsAgentClient.use_fleet_reporter`); ``strands_model_call`` names the entry's
``model_id``. A reissued call counts toward ``max_iterations``. A list of one runs exactly the
single-model path (no wrapper, no new line, metric or attribute).

Model failover's log lines (model IDs, entry indexes, numbers, causes and error codes
only): ``model_failover`` (``turn``: the call's agent turn in this process, ``from_model``,
``to_model``, ``cause`` ``deadline`` or ``error`` for a reissue before any output, ``degraded`` for a
turn that starts below its call's previous entry because of the verdict, ``unavailable`` or
``exhausted`` for a later hop of a turn that starts on another entry than the turn's because that
entry is now unavailable or failed with nowhere left); ``model_turn_sticky`` (3.3a: a later
hop of a turn that moved starts on the entry it moved to: ``entry``, ``model_id``, ``abandoned``);
``model_degraded`` and
``model_unavailable`` when this process's own evidence marks an entry (``scope=process``; the
fleet's verdicts are the poller's lines); ``model_request_error`` (never a state change). The
``Failover``, ``Unavailable`` and ``ActiveRank`` metrics go through
:meth:`StrandsAgentClient.use_failover_events`. With two or more entries, Strands' TTFT and
latency histograms carry the call's ``model_id`` attribute (:class:`_ModelTaggedLoopMetrics`).

Never logged: caller text, tool inputs or results, model text. Logged: counts, timings, token counts,
error classes and codes, model IDs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, Callable, Collection, Iterator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal, Protocol

from bedrock_agentcore.memory.integrations.strands import session_manager as _acm
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from strands import Agent, tool
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.hooks import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)
from strands.models import BedrockModel, CacheConfig
from strands.models.model import Model
from strands.telemetry.metrics import EventLoopMetrics, MetricsClient
from strands.types.exceptions import (
    ContextWindowOverflowException,
    MaxTokensReachedException,
    ModelThrottledException,
    SessionException,
)
from strands.types.session import SessionAgent, SessionMessage
from strands.types.tools import ToolContext

from owlbank.failover import (
    EntryState,
    FailoverState,
    FleetView,
    ModelUnusable,
    Outcome,
    Settings,
    Transition,
    classify,
)
from owlbank.failover.classify import Classification
from owlbank.obs import emit, log_caught
from owlbank.tools_contract import (
    DEFAULT_CARDS,
    DEFAULT_REASON_DESCRIPTIONS,
    END_CALL,
    GATEWAY_LEVEL0_TOOLS,
    HANDOFF_TOOLS,
    gateway_tool_full_name,
    inline_tool_specs,
    tool_error_code,
    tool_outcome,
)

from .agent_client import (
    _NOT_RETRYABLE,
    AgentError,
    AgentEvent,
    HistoryMode,
    MessageEnd,
    TextBeforeTool,
    TextDelta,
    TextSuppressed,
    ToolCall,
    ToolResult,
    ToolStarted,
    TurnEnd,
    agent_error_from_aws,
    prompt_for,
    strip_reasoning,
    strip_silent,
)
from .guarded_model import PRODUCERS_KEY, GuardedModel, check_test_delay, record_reasoning

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- constants

#: The model request's reasoning fields per thinking setting (an ``OWLBANK_MODELS`` entry's
#: ``thinking``). ``enabled:<budget>`` is the one parameterised setting
#: (:data:`ENABLED_THINKING_PREFIX`, :func:`thinking_request_fields`).
THINKING_BETA = "thinking-binding-controls-2026-08-01"
THINKING_PARAMS: dict[str, dict | None] = {
    "adaptive": {"additionalModelRequestFields": {
        "anthropic_beta": [THINKING_BETA],
        "thinking": {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}}},
    "off": {"additionalModelRequestFields": {"thinking": {"type": "disabled"}}},
    "default": None,
    # Sonnet 5.5 refuses ``disabled`` and names ``between_tools`` instead.
    "between_tools": {"additionalModelRequestFields": {"thinking": {"type": "between_tools"}}},
}
#: ``enabled:<budget>``: extended thinking with a token budget (at least :data:`MIN_THINKING_BUDGET`,
#: below the entry's ``max_tokens``).
ENABLED_THINKING_PREFIX = "enabled:"
MIN_THINKING_BUDGET = 1024
#: A thinking setting: a :data:`THINKING_PARAMS` key or ``enabled:<budget>``.
ThinkingMode = str
#: Entries in a model list (one to three).
MAX_MODEL_ENTRIES = 3
#: When a model message's text is released as ``TextDelta``:
#: ``stream``, each chunk at once until a tool-use block starts in the message; ``message``, at
#: ``messageStop`` and only if the message has no tool use (today's behaviour).
TextRelease = Literal["stream", "message"]
TEXT_RELEASE_MODES: tuple[str, ...] = ("stream", "message")
#: The TTL of the tools and static-system cache points: ``5m``
#: writes no ``ttl`` (Bedrock's default); ``1h`` writes ``ttl: "1h"`` on those two points, and the
#: message point keeps the default, so TTLs never increase (tools, system, messages).
PromptCacheTtl = Literal["5m", "1h"]
PROMPT_CACHE_TTLS: tuple[str, ...] = ("5m", "1h")
HISTORY_MODES: tuple[str, ...] = ("new", "cached", "reload")
#: The span around each rebuild from Memory.
SPAN_MEMORY_RELOAD = "memory.reload"
#: How an agent is built: empty without a Memory read, or restored from Memory.
BuildMode = Literal["new", "reload"]

#: The Gateway tools the model may see, by the full name the Gateway lists. The single source
#: is ``tools_contract``; anything else a provider lists is dropped.
ALLOWED_GATEWAY_TOOLS: tuple[str, ...] = tuple(gateway_tool_full_name(t) for t in GATEWAY_LEVEL0_TOOLS)
_GATEWAY_NAMES: frozenset[str] = frozenset(ALLOWED_GATEWAY_TOOLS)
#: A sentence end in streamed text: ``.``, ``?`` or ``!`` (closing quotes or brackets allowed) then
#: whitespace, so "3.50" and a trailing "." still being written don't count.
_SENTENCE_END = re.compile(r"[.?!][\"')\]]*\s")
#: How much of a message's text before a new chunk the sentence-end search keeps: an end mark and
#: its closing quotes or brackets split from the whitespace after them (a longer run of closers
#: than this is not seen as a sentence end, which only delays the timing).
_SENTENCE_TAIL = 16

#: Prefix of the interrupt names the inline tools raise (``inline:<tool>``).
INTERRUPT_PREFIX = "inline:"
#: Strands ``agent_id`` of the call's agent (the session manager keys its agent state by it).
AGENT_ID = "owlbank_voice"
#: AgentCore Memory actor of the call histories. The session is the CallSid. The value
#: is kept from when a second agent backend wrote its own histories under actor = CallSid.
MEMORY_ACTOR_ID = "owlbank-voice-strands"
#: Messages buffered per flush: the library's maximum, so in practice one flush per invocation.
MEMORY_BATCH_SIZE = 100
#: What a pending inline call gets when the bridge sends no result for it (step 3).
CANCELLED: dict[str, Any] = {"error": "cancelled"}

#: Strands stop reasons reported in :class:`TurnEnd` under the Bedrock name, so logs and the saved
#: benchmark baselines compare like with like: a pause for inline tools is ``tool_use``.
_STOP_REASON_NAMES = {"interrupt": "tool_use"}


# --------------------------------------------------------------------------- settings


def text_release_conflict(text_release: str, thinking: str) -> str | None:
    """Why ``text_release`` can't run with ``thinking``, or ``None``:
    the one rule and message for :class:`StrandsSettings` (``ValueError``), the bridge's
    environment (``RuntimeError``, prefixed with the ``OWLBANK_*`` names) and the latency bench
    (``SystemExit``). Infra's per-entry ``modelParams`` check says the same in its own names."""
    if text_release == "stream" and thinking == "off":
        return ("text_release=stream needs thinking adaptive or default: with reasoning off the model "
                "narrates before tool calls; set text_release=message to turn reasoning off")
    return None


def _thinking_budget(thinking: str) -> int | None:
    """The budget of ``enabled:<budget>``; ``None`` for any other setting or a malformed budget."""
    if not thinking.startswith(ENABLED_THINKING_PREFIX):
        return None
    raw = thinking[len(ENABLED_THINKING_PREFIX):]
    return int(raw) if raw.isascii() and raw.isdigit() else None


def thinking_error(thinking: Any, max_tokens: int) -> str | None:
    """Why ``thinking`` is not a usable setting with ``max_tokens``, or ``None``. Never echoes
    the value."""
    if not isinstance(thinking, str):
        return "thinking must be a string"
    if thinking in THINKING_PARAMS:
        return None
    if not thinking.startswith(ENABLED_THINKING_PREFIX):
        return f"thinking must be one of {sorted(THINKING_PARAMS)} or {ENABLED_THINKING_PREFIX}<budget>"
    budget = _thinking_budget(thinking)
    if budget is None or budget < MIN_THINKING_BUDGET:
        return f"the thinking budget must be an integer of at least {MIN_THINKING_BUDGET}"
    if budget >= max_tokens:
        return "the thinking budget must be below max_tokens"
    return None


@dataclass(frozen=True)
class ModelEntry:
    """One entry of the model list: the model (or
    inference profile) ID and the request parameters it needs. The ``OWLBANK_MODELS`` entry shape."""

    id: str
    thinking: ThinkingMode
    max_tokens: int
    #: How this model's text is released (:data:`TextRelease`).
    text_release: TextRelease = "stream"
    prompt_caching: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("id is required")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        error = thinking_error(self.thinking, self.max_tokens)
        if error is not None:
            raise ValueError(error)
        if self.text_release not in TEXT_RELEASE_MODES:
            raise ValueError(f"text_release must be one of {list(TEXT_RELEASE_MODES)}")
        conflict = text_release_conflict(self.text_release, self.thinking)
        if conflict is not None:
            raise ValueError(conflict)
        if not isinstance(self.prompt_caching, bool):
            raise ValueError("prompt_caching must be true or false")


@dataclass(frozen=True)
class StrandsSettings:
    """``models`` is the ordered model list (one to three entries,
    the preferred first); a list of one runs exactly today's single-model path. ``gateway_url`` and
    ``memory_id`` are used by the wiring. ``stage`` and ``failover_test_s`` (the
    ``OWLBANK_FAILOVER_TEST`` delay of the first entry, refused at ``stage=prod``) are the guarded
    models' (:mod:`tac_bridge.guarded_model`)."""

    models: tuple[ModelEntry, ...]
    region: str
    gateway_url: str = ""
    memory_id: str = ""
    max_iterations: int = 6
    turn_timeout_s: float = 30.0
    window_messages: int = 40
    cache_max_calls: int = 500
    cache_idle_s: float = 900.0
    stage: str = ""
    failover_test_s: float | None = None
    prompt_cache_ttl: PromptCacheTtl = "5m"

    def __post_init__(self) -> None:
        models = self.models
        if (not isinstance(models, tuple) or not 1 <= len(models) <= MAX_MODEL_ENTRIES
                or not all(isinstance(m, ModelEntry) for m in models)):
            raise ValueError(f"models must be a tuple of 1 to {MAX_MODEL_ENTRIES} model entries")
        if len({m.id for m in models}) != len(models):
            raise ValueError("models must not repeat a model ID")
        for name in ("max_iterations", "window_messages", "cache_max_calls"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.turn_timeout_s <= 0 or self.cache_idle_s <= 0:
            raise ValueError("turn_timeout_s and cache_idle_s must be positive")
        check_test_delay(self.failover_test_s, stage=self.stage)
        if not isinstance(self.prompt_cache_ttl, str) or self.prompt_cache_ttl not in PROMPT_CACHE_TTLS:
            raise ValueError(f"prompt_cache_ttl must be one of {list(PROMPT_CACHE_TTLS)}")

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(m.id for m in self.models)

    @property
    def primary(self) -> ModelEntry:
        """The preferred entry (the first)."""
        return self.models[0]

    def only(self, index: int) -> StrandsSettings:
        """These settings with entry ``index`` as the only model (what a model factory builds from)."""
        return replace(self, models=(self.models[index],), failover_test_s=None)


def thinking_request_fields(thinking: str) -> dict[str, Any] | None:
    """The ``additionalModelRequestFields`` for a thinking setting (``None``: send nothing)."""
    budget = _thinking_budget(thinking)
    if budget is not None:
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    params = THINKING_PARAMS[thinking]
    return None if params is None else dict(params["additionalModelRequestFields"])


def default_model_factory(settings: StrandsSettings) -> Model:
    """The Bedrock model for the settings' first entry. With
    ``prompt_cache_ttl`` ``1h`` the tools point carries its own ``ttl: "1h"``;
    ``cache_config.ttl`` stays unset, so the message point keeps the default 5 minutes."""
    entry = settings.primary
    kwargs: dict[str, Any] = {"model_id": entry.id, "max_tokens": entry.max_tokens}
    fields = thinking_request_fields(entry.thinking)
    if fields is not None:
        kwargs["additional_request_fields"] = fields
    if entry.prompt_caching:
        tools_ttl: bool | str = "1h" if settings.prompt_cache_ttl == "1h" else True
        kwargs["cache_config"] = CacheConfig(strategy="anthropic", tools_ttl=tools_ttl)
    return BedrockModel(region_name=settings.region, **kwargs)


def system_blocks(base: str | None, suffix: str | None, *, caching: bool,
                  ttl: str | None = None) -> list[dict[str, Any]] | None:
    """The system prompt as content blocks: the stable base, a cache point when caching is on, and
    the suffix after it: the text after the cache point (the per-call part and the clock line), so
    it never invalidates the cache. The point carries ``ttl`` only when it is ``1h``
    (``5m`` is Bedrock's default and writes none). Empty parts are left
    out (Bedrock rejects empty text blocks)."""
    blocks: list[dict[str, Any]] = []
    if base:
        blocks.append({"text": base})
        if caching:
            point: dict[str, Any] = {"type": "default"}
            if ttl == "1h":
                point["ttl"] = ttl
            blocks.append({"cachePoint": point})
    if suffix:
        blocks.append({"text": suffix})
    return blocks or None


# --------------------------------------------------------------------------- tools


def _inline_tool(name: str, description: str, input_schema: dict[str, Any]) -> Any:
    """One inline tool: the only parameter is the tool context, so any input
    passes and the explicit spec is what the model sees. It raises an interrupt carrying the
    tool-use ID and input; on resume it returns the bridge's result. No side effects (it runs twice)."""

    @tool(name=name, description=description, inputSchema={"json": input_schema}, context=True)
    def _inline(tool_context: ToolContext) -> Any:
        use = tool_context.tool_use
        return tool_context.interrupt(f"{INTERRUPT_PREFIX}{name}",
                                      reason={"toolUseId": use["toolUseId"], "input": use.get("input")})

    return _inline


def inline_tools(cards: Collection[str] = DEFAULT_CARDS,
                 reason_descriptions: Mapping[str, str] = DEFAULT_REASON_DESCRIPTIONS,
                 exclude: Collection[str] = ()) -> list[Any]:
    """The inline tools with their specs from ``tools_contract.inline_tool_specs()``. ``cards``:
    the ``card`` slot's options for ``make_payment.to`` (value -> spoken name).
    ``reason_descriptions``: ``{reason: one line}`` for ``escalate.reason``
    (the live policy's, :attr:`owlbank.policy.Policy.reason_descriptions`). ``exclude``: tools
    left out (``{END_CALL}`` when the policy's ``end_call`` switch is off)."""
    out = []
    for entry in inline_tool_specs(cards=cards, reason_descriptions=reason_descriptions,
                                   exclude=exclude):
        fn = entry["config"]["inlineFunction"]
        out.append(_inline_tool(entry["name"], fn["description"], fn["inputSchema"]))
    return out


class GatewayToolProvider(Protocol):
    """Gives the already-listed Gateway tools for a new agent. Raises if unavailable."""

    def tools(self) -> Sequence[Any]: ...


class McpGatewayTools:
    """The process-wide Gateway MCP client: started once, listed once, never handed to an
    Agent itself (a started client can't be). Each Agent gets the listed tools.

    If the start or the listing failed, :meth:`tools` raises at once and, at most every
    ``retry_s``, retries in a daemon thread (never in the caller's path), so a later agent or turn
    gets the tools once the Gateway is back."""

    def __init__(self, mcp_client: Any, *, retry_s: float | None = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._mcp = mcp_client
        self._tools: list[Any] | None = None
        self._started = False
        self._closed = False
        self._retry_s = retry_s
        self._clock = clock
        self._last_attempt: float | None = None
        self._retrying = False
        self._lock = threading.Lock()

    def start(self) -> None:
        self._last_attempt = self._clock()
        if not self._started:
            self._mcp.start()
            self._started = True
        self._tools = list(self._mcp.list_tools_sync())
        emit(log, logging.INFO, "gateway_tools_listed", tools=len(self._tools))

    def tools(self) -> Sequence[Any]:
        tools = self._tools
        if tools is None:
            self._retry_later()
            raise RuntimeError("the Gateway MCP client is not started")
        return list(tools)

    def _retry_later(self) -> None:
        with self._lock:
            if self._closed or self._retrying or self._retry_s is None:
                return
            if self._last_attempt is not None and self._clock() - self._last_attempt < self._retry_s:
                return
            self._retrying = True
            self._last_attempt = self._clock()
        threading.Thread(target=self._retry, name="gateway-mcp-retry", daemon=True).start()

    def _retry(self) -> None:
        try:
            self.start()
        except Exception as exc:  # noqa: BLE001 - tried again after retry_s
            log_caught(log, "gateway MCP client retry failed", exc, operation="gateway_start",
                       decision="retry_later")
        finally:
            with self._lock:
                self._retrying = False

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._tools = None
        if not self._started:
            return
        self._started = False
        self._mcp.stop(None, None, None)


def make_gateway_tool_provider(url: str, region: str, *,
                               mcp_client_factory: Callable[..., Any] | None = None) -> McpGatewayTools:
    """Build and start the Gateway MCP client (SigV4, service ``bedrock-agentcore``) with the
    allow-list filter (full names, no prefix), and list its tools. A failure to start is logged and
    leaves the provider without tools (each agent then omits them), never fatal."""
    def transport() -> Any:
        from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client  # noqa: PLC0415 (bridge only)

        return aws_iam_streamablehttp_client(url, aws_service="bedrock-agentcore", aws_region=region)

    if mcp_client_factory is None:
        from strands.tools.mcp import MCPClient  # noqa: PLC0415

        mcp_client_factory = MCPClient
    client = mcp_client_factory(transport, tool_filters={"allowed": list(ALLOWED_GATEWAY_TOOLS)}, prefix=None)
    provider = McpGatewayTools(client)
    try:
        provider.start()
    except Exception as exc:  # noqa: BLE001 - not fatal: the Level-0 tools are left out
        log_caught(log, "gateway MCP client failed to start", exc, operation="gateway_start",
                   decision="omit_gateway_tools")
    return provider


# --------------------------------------------------------------------------- history (AgentCore Memory)


class HistoryUnavailable(SessionException):
    """Reading the call's history from Memory failed (never continue with an empty one)."""


class _NewestStateSessionManager(AgentCoreMemorySessionManager):
    """``AgentCoreMemorySessionManager`` with the fixes this client needs.

    **Depends on internals of the exactly pinned ``bedrock-agentcore==1.24.1``** (the event layout of
    ``_flush_agent_states_only``, the metadata keys, ``memory_client.list_events``, the
    ``_flush_messages`` hook and ``read_session`` being the only read in ``__init__``). The tests run
    the unpatched class against the same fake Memory to prove each fix is still needed, and a round
    trip proves the restore, so a version bump that changes any of this fails the suite.

    * ``read_agent`` restores the **newest** agent-state snapshot. With ``batch_size > 1`` the
      library packs every buffered snapshot of a flush into one event, oldest first, and reads
      ``payload[0]``: a stale interrupt state (``activated`` False) that can't be resumed. The newest
      event comes first from ``ListEvents``; its newest snapshot is the last blob.
    * ``read_agent`` and ``list_messages`` **raise** :class:`HistoryUnavailable` when the read fails
      (the library logs and returns ``None`` / ``[]``, which would continue with an empty history).
    * ``fresh=True`` (history ``new``): ``read_session`` returns "no session" without a
      ``ListEvents`` call, so the agent starts empty and nothing is read (the session event is still
      written, so a later ``reload`` finds the history).
    * A restored manager **never writes an event older than one it read**: its clock (the library's
      per-instance monotonic event timestamps) starts after the newest event read. Memory orders a
      session's events by ``eventTimestamp``; without this, a manager built after another one (a
      move between tasks, or a reload after eviction) takes its timestamps from the wall clock,
      which can be behind the previous manager's (it bumps same-millisecond ties ahead), and its
      first events would sort before the previous manager's last ones (found by the test).
    * ``_flush_messages`` (the library's flush, run automatically at the end of each invocation)
      records its outcome: ``flush_error`` after a failure, ``flushes`` after each success.
    """

    def __init__(self, config: AgentCoreMemoryConfig, *, fresh: bool = False, **kwargs: Any) -> None:
        self._fresh = fresh  # read by read_session, which the base __init__ calls
        self.flush_error: BaseException | None = None
        self.flushes = 0
        super().__init__(config, **kwargs)
        self._fresh = False

    def read_session(self, session_id: str, **kwargs: Any) -> Any:
        if self._fresh:
            return None
        try:
            return super().read_session(session_id, **kwargs)
        except Exception as exc:
            raise HistoryUnavailable(f"reading the session failed: {type(exc).__name__}") from exc

    def read_agent(self, session_id: str, agent_id: str, **kwargs: Any) -> SessionAgent | None:
        if session_id != self.config.session_id:
            return None
        try:
            events = self.memory_client.list_events(
                memory_id=self.config.memory_id, actor_id=self.config.actor_id, session_id=session_id,
                event_metadata=[_meta_equals(_acm.STATE_TYPE_KEY, _acm.StateType.AGENT.value),
                                _meta_equals(_acm.AGENT_ID_KEY, agent_id)],
                max_results=1)
        except Exception as exc:
            raise HistoryUnavailable(f"reading the agent state failed: {type(exc).__name__}") from exc
        if not events:
            return None
        self._seed_clock(events)
        blobs = [p["blob"] for p in events[0].get("payload") or () if isinstance(p, Mapping) and "blob" in p]
        if not blobs:
            raise HistoryUnavailable("the agent-state event has no snapshot")
        agent = SessionAgent.from_dict(json.loads(blobs[-1]))  # the newest snapshot of the flush
        if agent.created_at:  # as the library does (it re-sends created_at with each update)
            created: dict[str, Any] = self._agent_created_at_cache
            created[agent_id] = agent.created_at
        return agent

    def list_messages(self, session_id: str, agent_id: str, limit: int | None = None, offset: int = 0,
                      **kwargs: Any) -> list[SessionMessage]:
        if session_id != self.config.session_id:
            raise SessionException("session ID mismatch")
        try:
            events = self.memory_client.list_events(
                memory_id=self.config.memory_id, actor_id=self.config.actor_id, session_id=session_id,
                max_results=(limit + offset) if limit else _acm.MAX_FETCH_ALL_RESULTS)
            messages = list(self.converter.events_to_messages(events))
        except Exception as exc:
            raise HistoryUnavailable(f"reading the messages failed: {type(exc).__name__}") from exc
        self._seed_clock(events)
        return messages[offset:offset + limit] if limit is not None else messages[offset:]

    def _seed_clock(self, events: Sequence[Mapping[str, Any]]) -> None:
        """Start this manager's event clock after the newest event read (see the class docstring)."""
        stamps = [e["eventTimestamp"] for e in events if isinstance(e.get("eventTimestamp"), datetime)]
        if not stamps:
            return
        newest = max(stamps)
        with self._timestamp_lock:
            if self._last_timestamp is None or newest > self._last_timestamp:
                self._last_timestamp = newest

    def _flush_messages(self) -> list[dict[str, Any]]:
        try:
            out = super()._flush_messages()
        except BaseException as exc:
            self.flush_error = exc
            raise
        self.flush_error = None
        self.flushes += 1
        return out

    def pending(self) -> int:
        """Messages and agent states still buffered (not yet in Memory)."""
        return self.pending_message_count() + self.pending_agent_state_count()

    def drop_unrun_agent_state(self) -> int:
        """Before a late ``close()``: if no invocation ever flushed, the buffer holds only the
        snapshot of an agent that never ran (its creation). Written late, it would become the
        newest agent state after another process moved the call on and handed off, so the next
        reload would lose that hand-off's pending tool use (found by the test). Returns how
        many snapshots were dropped."""
        if self.flushes:
            return 0
        with self._agent_state_lock:
            dropped = len(self._agent_state_buffer)
            self._agent_state_buffer.clear()
        return dropped


class _AsyncModeNotice(logging.Filter):
    """Drops the library's WARNING on every session manager built with ``async_mode=True`` ("the
    agent must be invoked via the async path"): this client only ever calls ``stream_async``, so it
    would be one false warning per call."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.msg).startswith("AgentCoreMemorySessionManager async_mode=True")


_acm.logger.addFilter(_AsyncModeNotice())

#: The logger OpenTelemetry's ``context.detach`` logs to.
OTEL_CONTEXT_LOGGER = "opentelemetry.context"
_OTEL_DETACH_MESSAGE = "Failed to detach context"


class OtelDetachNoise(logging.Filter):
    """Drops OpenTelemetry's ``Failed to detach context`` ERROR (with a ``ValueError`` traceback),
    logged when a Strands stream is closed early (``aclose()`` on barge-in): Strands' span context
    was attached in another ``contextvars`` context. Harmless (logged, not raised), but it would be an ERROR
    line on every barge-in. Every other record passes."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.name == OTEL_CONTEXT_LOGGER and record.getMessage() == _OTEL_DETACH_MESSAGE)


def install_otel_detach_filter() -> None:
    """Put :class:`OtelDetachNoise` on the OpenTelemetry context logger, once (idempotent)."""
    logger = logging.getLogger(OTEL_CONTEXT_LOGGER)
    if not any(isinstance(f, OtelDetachNoise) for f in logger.filters):
        logger.addFilter(OtelDetachNoise())


install_otel_detach_filter()


def _meta_equals(key: str, value: str) -> Any:
    """An ``EventMetadataFilter`` expression, as ``EventMetadataFilter.build_expression`` makes it."""
    return {"left": {"metadataKey": key}, "operator": "EQUALS_TO",
            "right": {"metadataValue": {"stringValue": value}}}


class _SharedBotoClients:
    """A stand-in for the ``boto3.Session`` the session manager takes: one client per service for
    the whole process, created on first use under a lock (a boto3 session isn't thread-safe; the
    clients are, and agents are built in worker threads)."""

    def __init__(self, region: str, session_factory: Callable[[], Any] | None = None) -> None:
        self.region_name = region
        self._session_factory = session_factory
        self._lock = threading.Lock()
        self._clients: dict[str, Any] = {}

    def client(self, service: str, region_name: str | None = None, config: Any = None) -> Any:
        with self._lock:
            c = self._clients.get(service)
            if c is None:
                if self._session_factory is None:
                    import boto3  # noqa: PLC0415 - only when a call is first built

                    self._session_factory = lambda: boto3.Session(region_name=self.region_name)
                c = self._session_factory().client(service, region_name=region_name or self.region_name,
                                                   config=config)
                self._clients[service] = c
            return c


class MemorySessionManagerFactory:
    """The call's session manager, ``session_id = CallSid``, the Strands actor, batched
    asynchronous writes, no retrieval (short-term only). Nothing is created until the first build
    or :meth:`prewarm`. ``boto_session`` is for tests (an object with ``client()`` and
    ``region_name``)."""

    def __init__(self, memory_id: str, region: str, *, boto_session: Any = None) -> None:
        if not memory_id:
            raise ValueError("memory_id is required")
        self.memory_id = memory_id
        self.region = region
        self._clients = boto_session if boto_session is not None else _SharedBotoClients(region)

    def __call__(self, session_id: str, history: BuildMode) -> _NewestStateSessionManager:
        config = AgentCoreMemoryConfig(memory_id=self.memory_id, session_id=session_id, actor_id=MEMORY_ACTOR_ID,
                                       batch_size=MEMORY_BATCH_SIZE, async_mode=True)
        return _NewestStateSessionManager(config, fresh=history == "new", region_name=self.region,
                                          boto_session=self._clients)

    def prewarm(self) -> None:
        """Pay the Memory clients' first-use cost now, not on a call's first turn (the first
        ``MemoryClient`` built in a process takes about 4 s cold): one throwaway library
        ``MemoryClient`` (what every session manager builds first) and the shared data-plane and
        control-plane clients, with the configuration the session manager asks for. No AWS call
        (building a boto3 client is local)."""
        from botocore.config import Config as BotocoreConfig  # noqa: PLC0415

        _acm.MemoryClient(region_name=self.region)
        config = BotocoreConfig(user_agent_extra="strands-agents")  # as the session manager's
        for service in ("bedrock-agentcore-control", "bedrock-agentcore"):
            self._clients.client(service, region_name=self.region, config=config)


def memory_session_manager_factory(memory_id: str, region: str, *,
                                   boto_session: Any = None) -> MemorySessionManagerFactory:
    """The AgentCore Memory session-manager factory (see :class:`MemorySessionManagerFactory`)."""
    return MemorySessionManagerFactory(memory_id, region, boto_session=boto_session)


# --------------------------------------------------------------------------- per-turn state and hooks


@dataclass
class _TurnState:
    """What one ``turn()`` has seen so far (shared by the hook and the event translation)."""

    model_calls: int = 0
    call_started: float | None = None
    first_byte: float | None = None
    #: When the call's first answer-text delta, and the end of its first sentence, came in the stream
    #: (measurement for releasing text by sentence; timings only, never the text).
    first_text: float | None = None
    first_sentence: float | None = None
    #: When the call's first output in the sense of model failover came: a text delta or a tool
    #: use (its start or its input); the call's TTFT sample (reasoning is not output).
    first_output: float | None = None
    call_usage: dict[str, int] = field(default_factory=dict)
    #: The model call's provider timings from the stream's ``metadata`` chunk (``latencyMs``, and
    #: ``timeToFirstByteMs`` if the provider sends it).
    call_metrics: dict[str, int] = field(default_factory=dict)
    call_stop: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class _LoopLimits(HookProvider):
    """Count model calls per ``turn()``; the one past ``max_iterations`` raises a
    non-retryable ``max_iterations_exceeded`` before it reaches the model. The call's timing is
    :class:`_ModelCallTelemetry`'s (registered after this one, so a refused call is never timed)."""

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        self.state: _TurnState | None = None

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self._before_model_call)

    def _before_model_call(self, event: BeforeModelCallEvent) -> None:
        st = self.state
        if st is None:
            return
        if st.model_calls >= self.max_iterations:
            raise AgentError(f"more than {self.max_iterations} model calls in one turn",
                             retryable=False, code="max_iterations_exceeded")
        st.model_calls += 1


#: Called once per model call that reached the model: ``(entry, exception, stop reason, TTFT s)``.
ModelCallObserver = Callable[[int, BaseException | None, str | None, float | None], None]


def entry_of(model: Any) -> int:
    """The list entry a model the agent uses stands for: a :class:`GuardedModel`'s own index; the
    bare model of a list of one is entry 0."""
    return model.entry if isinstance(model, GuardedModel) else 0


@dataclass
class _CallModels:
    """Per agent build ("per call"): the model each list entry runs as on
    this agent (a :class:`GuardedModel` per entry for a list of two or more, the bare model for a
    list of one), and the entry that produced each reasoning block a model call of this build
    returned, by :func:`~tac_bridge.guarded_model.reasoning_key` (a digest; never the reasoning). A
    new build (a process restart, a rebuild from Memory) starts empty: the producer of a block it
    did not see returned is unknown. Kept in the bridge only, never stored; it lives as long as the
    build, so it holds at most one key per reasoning block of the call."""

    models: list[Model]
    producers: dict[str, int] = field(default_factory=dict)
    #: The entry the previous turn of this agent started on (``None``: no turn yet), for the
    #: ``model_failover`` line of a turn that starts lower because of the verdict.
    turn_entry: int | None = None


@dataclass(frozen=True)
class _TurnMove:
    """Model failover 3.3a: where the bridge turn ``turn_id`` stands after its last hop that called a
    model. ``entry``: the entry that hop ended on (the turn's later hops start there); ``failed``:
    that hop failed on it with no entry left; ``abandoned``: the entries the turn left (with
    ``entry`` when ``failed``), never used again in the turn unless every entry is."""

    turn_id: int
    entry: int
    failed: bool
    abandoned: frozenset[int]


@dataclass
class _SessionFailover:
    """Model failover, per session in this process: the agent calls run (``model_failover``'s
    ``turn``) and the current bridge turn's move. Kept outside the agent build, so a dropped or
    rebuilt agent mid-turn keeps it; ``begin_session`` clears the move (a session's turn numbers
    start again); dropped only once idle as long as a cached agent (a closed call), never by count."""

    last_used: float
    turns: int = 0
    move: _TurnMove | None = None


#: The two Strands histograms that carry the call's ``model_id``.
MODEL_TAGGED_INSTRUMENTS = ("event_loop_latency", "model_time_to_first_token")


class _TaggedHistogram:
    """A Strands histogram whose every record carries ``model_id`` (the model of the call it measures)."""

    def __init__(self, inner: Any, model_id: Callable[[], str | None]) -> None:
        self._inner, self._model_id = inner, model_id

    def record(self, amount: Any, attributes: Mapping[str, Any] | None = None, context: Any = None) -> None:
        attrs = dict(attributes or {})
        try:
            mid = self._model_id()
        except Exception:  # noqa: BLE001 - the value still goes out, untagged
            mid = None
        if mid:
            attrs["model_id"] = mid
        self._inner.record(amount, attributes=attrs or None, context=context)


class _TaggedMetricsClient:
    """Strands' :class:`MetricsClient` with :data:`MODEL_TAGGED_INSTRUMENTS` tagged; everything else as is."""

    def __init__(self, inner: Any, model_id: Callable[[], str | None]) -> None:
        self._inner, self._model_id = inner, model_id

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._inner, name)
        return _TaggedHistogram(value, self._model_id) if name in MODEL_TAGGED_INSTRUMENTS else value


class _ModelTaggedLoopMetrics(EventLoopMetrics):
    """Model failover: an agent's loop metrics whose TTFT and latency records carry the
    ``model_id`` of the model call they measure (at most three values, one per entry). Strands
    records them after the call, before the client can move the agent to another entry, so
    ``agent.model`` is still the call's model then."""

    _model_id_source: Callable[[], str | None] | None = None

    @property
    def _metrics_client(self) -> Any:
        inner = MetricsClient()
        source = self._model_id_source
        return inner if source is None else _TaggedMetricsClient(inner, source)


def tag_model_metrics(agent: Any, model_ids: Sequence[str]) -> None:
    """Give ``agent`` loop metrics that tag its TTFT and latency with the call's ``model_id``."""
    ids = tuple(model_ids)
    metrics = _ModelTaggedLoopMetrics()

    def current() -> str | None:
        entry = entry_of(agent.model)
        return ids[entry] if 0 <= entry < len(ids) else None

    metrics._model_id_source = current
    agent.event_loop_metrics = metrics


class _ReasoningProducers(HookProvider):
    """Model failover: which entry produced each reasoning block, so the guarded model sends a
    reasoning block only to the entry whose model signed it. A model call that returns a message
    records its reasoning blocks as its entry's (``AfterModelCallEvent``, the message Strands then
    adds to the history); before each model call ``invocation_state["reasoning_producers"]`` gets a
    copy of the record. Never raises: on a failure the copy is empty, so every reasoning block is
    left out (always accepted); a block a failed hook did not record is of unknown
    origin, left out the same way."""

    def __init__(self, track: _CallModels) -> None:
        self._track = track

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self._before_model_call)
        registry.add_callback(AfterModelCallEvent, self._after_model_call)

    def _before_model_call(self, event: BeforeModelCallEvent) -> None:
        try:
            event.invocation_state[PRODUCERS_KEY] = dict(self._track.producers)
        except Exception as exc:  # noqa: BLE001 - the safe side: filter every reasoning block
            with contextlib.suppress(Exception):
                event.invocation_state[PRODUCERS_KEY] = {}
            _hook_failed("reasoning_producers", exc)

    def _after_model_call(self, event: AfterModelCallEvent) -> None:
        try:
            response = getattr(event, "stop_response", None)
            message = getattr(response, "message", None) if response is not None else None
            if isinstance(message, Mapping):
                record_reasoning(self._track.producers, message, entry_of(event.agent.model))
        except Exception as exc:  # noqa: BLE001 - the blocks are then of unknown origin
            _hook_failed("reasoning_producers", exc)


class _ModelCallTelemetry(HookProvider):
    """The ``strands_model_call`` line from ``Before``/``AfterModelCallEvent``, so a
    **failed** model call gets one too (``outcome`` ``error``, ``error_class``). The line keeps the
    bridge's wall-clock timings (``model_ttfb_s``, ``model_latency_s``, from the stream the
    :class:`_Translator` sees) beside Bedrock's ``latencyMs`` (``bedrock_latency_ms``) and Strands'
    ``timeToFirstByteMs`` (``strands_ttfb_ms``) when the call succeeded, and the entry's
    ``model_id`` (with ``cause`` when the call was abandoned before any output).
    ``observer`` gets every call's entry, outcome and TTFT (the failover decision state).
    Shares the turn's :class:`_TurnState` with ``limits``. Never raises: a hook failure loses
    only the line."""

    def __init__(self, limits: _LoopLimits, *, model_ids: Sequence[str] = (),
                 observer: ModelCallObserver | None = None) -> None:
        self._limits = limits
        self._model_ids = tuple(model_ids)
        self._observer = observer

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self._before_model_call)
        registry.add_callback(AfterModelCallEvent, self._after_model_call)

    def _before_model_call(self, event: BeforeModelCallEvent) -> None:
        try:
            st = self._limits.state
            if st is None:
                return
            st.call_started = time.monotonic()
            st.first_byte = None
            st.first_text = None
            st.first_sentence = None
            st.first_output = None
            st.call_usage = {}
            st.call_metrics = {}
            st.call_stop = None
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a turn
            _hook_failed("model_call_before", exc)

    def _after_model_call(self, event: AfterModelCallEvent) -> None:
        try:
            st = self._limits.state
            if st is None or st.call_started is None:
                return
            exc = getattr(event, "exception", None)
            # The stream's metadata chunk (:class:`_Translator`), overlaid with the non-zero metrics
            # Strands attaches to the message (its own ``timeToFirstByteMs`` is only there; a zero
            # there is its default, not a measurement).
            metrics: dict[str, Any] = dict(st.call_metrics)
            stop = st.call_stop
            response = getattr(event, "stop_response", None)
            if response is not None:
                meta = (response.message or {}).get("metadata") if isinstance(response.message, Mapping) else None
                raw = (meta or {}).get("metrics") if isinstance(meta, Mapping) else None
                if isinstance(raw, Mapping):
                    metrics.update({k: v for k, v in raw.items() if _int_metric(raw, k)})
                stop = stop or response.stop_reason
            entry = entry_of(event.agent.model)
            if self._observer is not None:
                ttft = st.first_output - st.call_started if st.first_output is not None else None
                try:
                    self._observer(entry, exc, stop, None if ttft is None else max(0.0, ttft))
                except Exception as obs_exc:  # noqa: BLE001 - the line still goes out
                    _hook_failed("model_call_observer", obs_exc)
            model_id = self._model_ids[entry] if 0 <= entry < len(self._model_ids) else None
            _log_model_call(st, stop=stop, metrics=metrics, exc=exc, model_id=model_id)
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a turn
            _hook_failed("model_call_after", exc)


#: A failed telemetry hook is logged at most this often.
_HOOK_FAILURE_LOG_S = 60.0
_hook_failure_logged_at: float | None = None


def _hook_failed(hook: str, exc: BaseException) -> None:
    """A telemetry hook raised: ``telemetry_hook_failed`` (WARNING) at most once a minute per
    process, class and stack only (``log_caught``). Never raises."""
    global _hook_failure_logged_at  # noqa: PLW0603 - one rate limit for the process
    try:
        now = time.monotonic()
        if _hook_failure_logged_at is not None and now - _hook_failure_logged_at < _HOOK_FAILURE_LOG_S:
            return
        _hook_failure_logged_at = now
        log_caught(log, "telemetry_hook_failed", exc, level=logging.WARNING, operation=hook,
                   decision="continue_untraced")
    except Exception:  # noqa: BLE001, S110 - logging must not break the turn either
        pass


def _ms(seconds: float | None) -> int | None:
    """Whole milliseconds; ``None`` for no value or the histogram's open bin."""
    if seconds is None or seconds != seconds or seconds in (float("inf"), float("-inf")):
        return None
    return round(seconds * 1000)


def _int_metric(metrics: Mapping[str, Any], key: str) -> int | None:
    value = metrics.get(key)
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _log_model_call(st: _TurnState, *, stop: str | None, metrics: Mapping[str, Any],
                    exc: BaseException | None, model_id: str | None = None) -> None:
    """One line per model call: timings and token counts, never content; the
    entry's ``model_id`` and, for a call abandoned before any output, the
    ``cause`` (an outcome name)."""
    now = time.monotonic()
    started = st.call_started if st.call_started is not None else now
    u = st.call_usage
    fields: dict[str, Any] = {
        "call": st.model_calls,
        "model_id": model_id,
        "model_ttfb_s": round(st.first_byte - started, 3) if st.first_byte is not None else None,
        "model_first_text_s": round(st.first_text - started, 3) if st.first_text is not None else None,
        "model_first_sentence_s": (round(st.first_sentence - started, 3)
                                   if st.first_sentence is not None else None),
        "model_latency_s": round(now - started, 3),
        "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
        "cache_read_tokens": u.get("cache_read_tokens", 0), "cache_write_tokens": u.get("cache_write_tokens", 0),
        "stop_reason": stop,
        "outcome": "ok" if exc is None else "error",
    }
    for name, key in (("bedrock_latency_ms", "latencyMs"), ("strands_ttfb_ms", "timeToFirstByteMs")):
        value = _int_metric(metrics, key)  # the provider's own timings, when the call reported them
        if value is not None:
            fields[name] = value
    if exc is not None:
        fields["error_class"] = type(exc).__name__
        unusable = model_unusable(exc)
        if unusable is not None:
            fields["cause"] = "deadline" if unusable.is_deadline else str(unusable.cause)
    st.call_started = None
    emit(log, logging.INFO if exc is None else logging.WARNING, "strands_model_call", **fields)


#: ``owlbank.tool_kind`` on a tool span.
TOOL_KIND_INLINE, TOOL_KIND_GATEWAY, TOOL_KIND_HANDOFF = "inline", "gateway", "handoff"


def tool_kind(name: str) -> str:
    """``handoff`` (the front door runs it after the session ends), ``gateway`` (an approved
    Level-0 Gateway tool, run inside the loop) or ``inline`` (run by the bridge)."""
    if name in HANDOFF_TOOLS:
        return TOOL_KIND_HANDOFF
    if name in _GATEWAY_NAMES:
        return TOOL_KIND_GATEWAY
    return TOOL_KIND_INLINE


def _log_gateway_result(event: AfterToolCallEvent) -> None:
    """One ``tool_result`` line for a Gateway tool call that reached an outcome, classified by
    :func:`owlbank.tools_contract.tool_outcome` from the hook's ``result``, ``exception`` and
    ``cancel_message``. Inline tools are logged by the bridge, which runs them; an interrupt has no
    after-hook. A Gateway call cancelled by a barge-in or hang-up raises ``CancelledError`` through
    Strands, so it gets no after-hook either and logs nothing. No arguments or result text. Never
    raises."""
    try:
        name = str(event.tool_use.get("name", ""))
        if tool_kind(name) != TOOL_KIND_GATEWAY:
            return
        result = getattr(event, "result", None)
        exc = getattr(event, "exception", None)
        outcome = tool_outcome(TOOL_KIND_GATEWAY, result, exception=exc,
                               cancel_message=getattr(event, "cancel_message", None))
        fields: dict[str, Any] = {"tool": name[:64], "tool_kind": TOOL_KIND_GATEWAY, "outcome": outcome}
        code = tool_error_code(TOOL_KIND_GATEWAY, result) if exc is None else None
        if code is not None and outcome != "cancelled":
            fields["error_code"] = code
        emit(log, logging.INFO, "tool_result", **fields)
    except Exception as exc:  # noqa: BLE001 - telemetry never breaks a turn
        _hook_failed("tool_result", exc)


class _ToolTelemetry(HookProvider):
    """Our attributes on Strands' ``execute_tool`` span, which is the current span
    while the tool hooks run: ``owlbank.tool_kind`` (at ``BeforeToolCallEvent``, so an inline tool
    that ends on an interrupt has it too) and, at ``AfterToolCallEvent``, ``owlbank.outcome``
    (``success`` | ``error``) and ``owlbank.error_class``. A current span that isn't this tool
    call's (no matching ``gen_ai.tool.call.id``) is left alone. With telemetry off the current span
    is the no-op span. Never raises."""

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)
        registry.add_callback(AfterToolCallEvent, self._after_tool_call)

    @staticmethod
    def _span(tool_use: Mapping[str, Any]) -> Any:
        from opentelemetry import trace  # noqa: PLC0415 - Strands imports it already

        span = trace.get_current_span()
        if not span.is_recording():
            return None
        attributes = getattr(span, "attributes", None) or {}
        if attributes.get("gen_ai.tool.call.id") != tool_use.get("toolUseId"):
            return None
        return span

    def _before_tool_call(self, event: BeforeToolCallEvent) -> None:
        try:
            span = self._span(event.tool_use)
            if span is not None:
                span.set_attribute("owlbank.tool_kind", tool_kind(str(event.tool_use.get("name", ""))))
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a turn
            _hook_failed("tool_call_before", exc)

    def _after_tool_call(self, event: AfterToolCallEvent) -> None:
        _log_gateway_result(event)
        try:
            span = self._span(event.tool_use)
            if span is None:
                return
            exc = getattr(event, "exception", None)
            result = getattr(event, "result", None)
            status = result.get("status") if isinstance(result, Mapping) else None
            outcome = "error" if exc is not None or status == "error" else "success"
            span.set_attribute("owlbank.tool_kind", tool_kind(str(event.tool_use.get("name", ""))))
            span.set_attribute("owlbank.outcome", outcome)
            if exc is not None:
                span.set_attribute("owlbank.error_class", type(exc).__name__)
        except Exception as hook_exc:  # noqa: BLE001 - telemetry never breaks a turn
            _hook_failed("tool_call_after", hook_exc)


_USAGE_KEYS = (("inputTokens", "input_tokens"), ("outputTokens", "output_tokens"),
               ("cacheReadInputTokens", "cache_read_tokens"), ("cacheWriteInputTokens", "cache_write_tokens"))


class _Translator:
    """Strands stream events → protocol events.

    ``release="message"``: text is held per model message (raw ``event`` chunks) and released at
    ``messageStop`` unless the message has a tool use, in which case its length is reported as
    :class:`TextSuppressed`. Text still held when the message ends without a stop is suppressed too
    (nothing proves it had no tool use).

    ``release="stream"``: text before any
    tool-use block in the message is yielded as :class:`TextDelta` at once. At the message's first
    tool use after released text the caller hears (not blank once ``<silent/>`` markers are
    removed), one :class:`TextBeforeTool` reports that text's length and the tool's name. Text after a
    tool-use block is held and suppressed as in message
    mode; released text stays released, whatever follows.

    Each model message ends with one :class:`MessageEnd` at its ``messageStop`` (or, with none, at
    Strands' ``message`` event or the next ``messageStart``), after the message's text and tool
    starts; a tool use the bridge never sees (a name neither inline nor an approved
    Gateway tool, which Strands answers as unknown) still ends its message there.

    Reasoning is never spoken (either mode). A :class:`ToolCall` comes from the interrupt event, or,
    for a call the client defers (hand-offs), from the invocation's ``result``.

    A tool-use block naming an approved Gateway tool yields one
    :class:`ToolStarted` when the block completes in the stream (``contentBlockStop``, or the
    ``messageStop`` if the block was never closed). That is the earliest point its input is known,
    and Strands runs the tool only after the message ends, so it always precedes the tool's run.
    Once per tool-use ID; never for inline tools (they yield :class:`ToolCall`) or other names.
    """

    def __init__(self, state: _TurnState, defer: Callable[[str], bool], *, release: TextRelease) -> None:
        self.state = state
        self.defer = defer
        self.stream = release == "stream"
        self.held: list[str] = []
        self.tool_message = False
        # Stream mode, per message (reset at messageStart): the text already yielded, and whether
        # its TextBeforeTool was decided (sent, or not needed: nothing the caller hears).
        self.released: list[str] = []
        self.before_tool_sent = False
        # Per message: the end of its text so far, for a sentence end split across chunks.
        self.tail = ""
        self.deferred: list[ToolCall] = []
        self.open_tool: dict[str, str] | None = None  # the tool-use block being streamed
        self.started: set[str] = set()  # tool-use IDs already reported as ToolStarted
        self.open_message = False  # a messageStart came and its MessageEnd is still to send

    def feed(self, ev: Mapping[str, Any]) -> Iterator[AgentEvent]:
        if "event" in ev:
            yield from self._chunk(ev["event"])
        elif "message" in ev:
            msg = ev["message"]
            if isinstance(msg, Mapping) and msg.get("role") == "assistant":
                yield from self._suppress_held()
                yield from self._end_message()  # a message with no messageStop ends here
        elif "tool_interrupt_event" in ev:
            for call in _tool_calls(ev["tool_interrupt_event"]):
                if self.defer(call.name):
                    self.deferred.append(call)
                else:
                    yield call
        elif "result" in ev:
            yield from self.deferred
            self.deferred = []

    def _chunk(self, chunk: Mapping[str, Any]) -> Iterator[AgentEvent]:
        st = self.state
        if "messageStart" in chunk:
            yield from self._suppress_held()
            yield from self._end_message()  # the last one never ended (no messageStop, no message)
            self.open_message = True
            self.tool_message = False
            self.open_tool = None
            self.released, self.before_tool_sent, self.tail = [], False, ""
        elif "contentBlockStart" in chunk:
            self._first_byte()
            yield from self._close_tool()
            start = (chunk["contentBlockStart"] or {}).get("start") or {}
            if "toolUse" in start:
                self._first_output()
                self.tool_message = True
                use = start["toolUse"] or {}
                self.open_tool = {"id": str(use.get("toolUseId", "")), "name": str(use.get("name", "")),
                                  "input": ""}
                yield from self._before_tool()
        elif "contentBlockDelta" in chunk:
            self._first_byte()
            delta = (chunk["contentBlockDelta"] or {}).get("delta") or {}
            if "text" in delta:
                text = delta["text"]
                if text:
                    self._first_output()
                if text and self.stream and not self.tool_message:
                    self.released.append(text)
                    self._time_text(text)  # before the yield: the bridge may take a while to come back
                    yield TextDelta(text)
                elif text:
                    self.held.append(text)
                    self._time_text(text)
            elif "toolUse" in delta:
                self._first_output()
                self.tool_message = True
                if self.open_tool is not None:
                    piece = (delta["toolUse"] or {}).get("input", "")
                    if isinstance(piece, str):
                        self.open_tool["input"] += piece
                yield from self._before_tool()  # a tool-use delta without its block start
            # reasoningContent (and anything else) is never spoken
        elif "contentBlockStop" in chunk:
            yield from self._close_tool()
        elif "messageStop" in chunk:
            yield from self._close_tool()
            stop = (chunk["messageStop"] or {}).get("stopReason")
            st.call_stop = stop
            if (self.held or self.released) and st.first_sentence is None:
                st.first_sentence = time.monotonic()  # a one-sentence answer ends with the message
            if self.tool_message or stop == "tool_use":
                yield from self._before_tool()  # a tool_use stop with no tool-use block seen
                yield from self._suppress_held()
            else:
                for text in self.held:
                    yield TextDelta(text)
                self.held = []
            self.tool_message = False
            yield from self._end_message()
        elif "metadata" in chunk:
            meta = chunk["metadata"] or {}
            raw = meta.get("usage")
            if isinstance(raw, Mapping):
                for src, dst in _USAGE_KEYS:
                    value = raw.get(src)
                    if isinstance(value, int) and not isinstance(value, bool):
                        st.call_usage[dst] = st.call_usage.get(dst, 0) + value
                        st.usage[dst] = st.usage.get(dst, 0) + value
            timings = meta.get("metrics")
            if isinstance(timings, Mapping):  # The provider's timings for strands_model_call
                for key in ("latencyMs", "timeToFirstByteMs"):
                    value = _int_metric(timings, key)
                    if value is not None:
                        st.call_metrics[key] = value

    def _close_tool(self) -> Iterator[AgentEvent]:
        """End the open tool-use block: a :class:`ToolStarted` if it names an approved Gateway tool
        not yet reported. The input is the model's JSON; anything else is reported as
        ``{"_unparseable": True}``, as for an inline tool."""
        block, self.open_tool = self.open_tool, None
        if block is None or block["name"] not in _GATEWAY_NAMES:
            return
        if block["id"]:
            if block["id"] in self.started:
                return
            self.started.add(block["id"])
        try:
            args = json.loads(block["input"] or "{}")
        except ValueError:
            args = None
        if not isinstance(args, dict):
            args = {"_unparseable": True}
        yield ToolStarted(block["name"], args)

    def _before_tool(self) -> Iterator[AgentEvent]:
        """Stream mode: the message's one :class:`TextBeforeTool`, if text the caller hears was
        already released: ``chars`` counts it without ``<silent/>``
        markers or reasoning and markup (the bridge never speaks them), and whitespace,
        markers or reasoning alone send none. Its tool
        is the open tool-use block's name (``""`` if none). Text released stays fixed once a
        tool-use block started, so this is decided once."""
        if self.stream and self.released and not self.before_tool_sent:
            self.before_tool_sent = True
            spoken = strip_silent(strip_reasoning("".join(self.released)))
            if spoken.strip():
                yield TextBeforeTool(len(spoken),
                                     self.open_tool["name"] if self.open_tool is not None else "")

    def _end_message(self) -> Iterator[AgentEvent]:
        """The open model message's one :class:`MessageEnd` (the bridge's boundary,
        whatever the message's tool use)."""
        if self.open_message:
            self.open_message = False
            yield MessageEnd()

    def _suppress_held(self) -> Iterator[AgentEvent]:
        if self.held:
            yield TextSuppressed(sum(map(len, self.held)))
        self.held = []

    def _first_byte(self) -> None:
        if self.state.first_byte is None:
            self.state.first_byte = time.monotonic()

    def _first_output(self) -> None:
        if self.state.first_output is None:
            self.state.first_output = time.monotonic()

    def _time_text(self, text: str) -> None:
        """When the call's first answer text, and the end of its first sentence, arrived. Only the
        new chunk and the message's last few characters are searched (linear in the text)."""
        st = self.state
        if st.first_text is None:
            st.first_text = time.monotonic()
        if st.first_sentence is None:
            window = self.tail + text
            if _SENTENCE_END.search(window):
                st.first_sentence = time.monotonic()
            self.tail = window[-_SENTENCE_TAIL:]


def _tool_calls(event: Mapping[str, Any]) -> Iterator[ToolCall]:
    """The :class:`ToolCall` for an inline tool's interrupt event."""
    use = event.get("tool_use") or {}
    for interrupt in event.get("interrupts") or ():
        reason = getattr(interrupt, "reason", None)
        name = str(getattr(interrupt, "name", ""))
        if not (isinstance(reason, Mapping) and name.startswith(INTERRUPT_PREFIX)):
            continue
        args = reason.get("input", use.get("input"))
        if isinstance(args, str):  # Strands normally parses it; keep a fallback
            try:
                args = json.loads(args or "{}")
            except ValueError:
                args = None
        if not isinstance(args, dict):
            log.warning("strands tool input is not a JSON object", extra={
                "tool": name[len(INTERRUPT_PREFIX):][:64], "operation": "parse_tool_input",
                "decision": "fallback_unparseable_args"})
            args = {"_unparseable": True}
        yield ToolCall(str(reason.get("toolUseId", use.get("toolUseId", ""))),
                       name[len(INTERRUPT_PREFIX):], args)


def pending_interrupts(agent: Any) -> dict[str, str]:
    """``tool_use_id → interrupt id`` for the inline calls the agent is waiting on, read from its
    interrupt state at the start of every turn (a resume cancelled early leaves the state
    active, so the previous turn's view can't be trusted). Matched through ``reason["toolUseId"]``,
    never by parsing the opaque interrupt id."""
    state = agent._interrupt_state
    if not state.activated:
        return {}
    out: dict[str, str] = {}
    for interrupt in state.interrupts.values():
        reason = interrupt.reason
        if (str(interrupt.name).startswith(INTERRUPT_PREFIX) and isinstance(reason, Mapping)
                and isinstance(reason.get("toolUseId"), str)):
            out[reason["toolUseId"]] = interrupt.id
    return out


def _response(result: ToolResult) -> dict[str, Any]:
    """The interrupt response that becomes the ``toolResult``: ``{"status", "content"}``,
    in the JSON-text form of :meth:`ToolResult.to_block`. Never ``None`` (that re-interrupts)."""
    block = result.to_block()["toolResult"]
    return {"status": block["status"], "content": block["content"]}


Invocation = tuple[Any, dict[str, int] | None]


def unanswered_input(agent: Any) -> Mapping[str, Any] | None:
    """The history's last message if it is the caller's (``user``) with no reply after it: the
    previous invocation recorded its input, then failed before the model answered (an error, a
    timeout, or a barge-in that cancelled the model call). ``None`` otherwise."""
    messages = agent.messages
    if messages and messages[-1].get("role") == "user":
        return messages[-1]
    return None


def _answered(message: Mapping[str, Any]) -> set[Any]:
    """The tool-use IDs a ``user`` message answers with ``toolResult`` blocks."""
    return {b["toolResult"].get("toolUseId") for b in message.get("content") or ()
            if isinstance(b, Mapping) and "toolResult" in b}


def results_recorded(agent: Any, pending: Mapping[str, str], results: Mapping[str, ToolResult],
                     user_text: str | None) -> bool:
    """Results with caller words, where the results are already the history's unanswered
    last message (an earlier try recorded them with no model call, then failed before the words
    were added). The retry then sends only the words."""
    tail = unanswered_input(agent)
    return bool(results) and user_text is not None and tail is not None and not pending \
        and set(results) <= _answered(tail)


def retry_continues(agent: Any, pending: Mapping[str, str], results: Mapping[str, ToolResult],
                    user_text: str | None) -> bool:
    """Whether this turn resends the input of a turn that failed after recording it (the
    bridge's retry sends the same hop again, once). Then the client continues from the history
    instead of adding the input again:

    * results for tool uses no longer pending, all answered in the unanswered last message (the
      results were delivered, then the model call failed): without this the retry would be an
      ``unknown_tool_use``, and the call would escalate instead of retrying;
    * the same caller words as the unanswered last message, alone: the model would see them twice;
    *: results and caller words together, both recorded (the words last, the results just
      before them), then the model call failed.
    """
    tail = unanswered_input(agent)
    if tail is None or pending:
        return False
    content = list(tail.get("content") or ())
    if results:
        if user_text is not None and content == [{"text": user_text}]:
            messages = agent.messages
            before = messages[-2] if len(messages) >= 2 else {}
            return before.get("role") == "user" and set(results) <= _answered(before)
        return set(results) <= _answered(tail) and user_text is None
    return user_text is not None and content == [{"text": user_text}]


def plan_invocations(pending: Mapping[str, str], results: Mapping[str, ToolResult],
                     user_text: str | None, *, respond: bool = True) -> list[Invocation]:
    """The Strands invocations for one turn (step 3), as ``(input, limits)``.

    * Pending interrupts are all answered: with the bridge's result, or a synthesized cancelled
      result. Strands refuses text mixed with interrupt responses, so caller words
      follow as a second invocation, and the first is capped at one loop turn: the results are
      appended and no model call happens, so the history reads ``[..., toolResult, user text]``.
    * ``respond=False`` (:meth:`StrandsAgentClient.record_results`) and no words: the
      results invocation alone, capped the same way, so the results are recorded with no model call.
    * No pending interrupt: the caller's words.
    """
    if pending:
        responses = [{"interruptResponse": {
            "interruptId": iid,
            "response": _response(results.get(tid) or ToolResult(tid, CANCELLED, "error"))}}
            for tid, iid in pending.items()]
        if user_text:
            return [(responses, {"turns": 1}), (user_text, None)]
        if not respond:
            return [(responses, {"turns": 1})]
        return [(responses, None)]
    if user_text is None:
        raise ValueError("user_text or tool_results is required")
    return [(user_text, None)]


# --------------------------------------------------------------------------- errors


def _causes(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and what it wraps (Strands' ``EventLoopException.original_exception``, ``__cause__``,
    ``__context__``), breadth first, each once."""
    seen: set[int] = set()
    queue: list[BaseException] = [exc]
    while queue and len(seen) < 20:
        e = queue.pop(0)
        if id(e) in seen:
            continue
        seen.add(id(e))
        yield e
        for nxt in (getattr(e, "original_exception", None), e.__cause__, e.__context__):
            if isinstance(nxt, BaseException):
                queue.append(nxt)


def model_unusable(exc: BaseException) -> ModelUnusable | None:
    """The :class:`ModelUnusable` a failed model call or turn carries (Strands wraps it in an
    ``EventLoopException``), or ``None``."""
    return next((e for e in _causes(exc) if isinstance(e, ModelUnusable)), None)


def _error_code(exc: BaseException) -> tuple[str, str | None]:
    """A failed model call's error code (a boto ``Error.Code``, else the exception's class name) and
    message, for :func:`owlbank.failover.classify` only: never logged or kept."""
    for e in _causes(exc):
        response = getattr(e, "response", None)
        if isinstance(response, Mapping) and isinstance(response.get("Error"), Mapping):
            error = response["Error"]
            if error.get("Code"):
                message = error.get("Message")
                return str(error["Code"]), message if isinstance(message, str) else None
    return type(exc).__name__, None


#: The code of a turn that lost its last usable entry to a missed first-token deadline.
FIRST_OUTPUT_DEADLINE_CODE = "first_token_deadline"


def unusable_error(exc: ModelUnusable) -> AgentError:
    """ "all entries unusable": the last entry's failure as the turn's :class:`AgentError`,
    by its error code as for any Bedrock error (retryable unless a validation or access error); a
    missed deadline is retryable (:data:`FIRST_OUTPUT_DEADLINE_CODE`). Only the code reaches the error."""
    code = exc.error_code or (FIRST_OUTPUT_DEADLINE_CODE if exc.is_deadline else None)
    return AgentError(f"Bedrock ConverseStream failed before any output: {type(exc).__name__}", code=code,
                      retryable=code not in _NOT_RETRYABLE)


def to_agent_error(exc: BaseException) -> AgentError:
    """AWS errors by code (:func:`agent_error_from_aws`: throttling and 5xx
    retryable, validation and access errors not), with Strands' own exceptions mapped to codes.
    Only class names and AWS error codes reach the error; never model or caller text."""
    chain = list(_causes(exc))
    for e in chain:
        if isinstance(e, AgentError):
            return e
    for e in chain:
        if isinstance(e, ContextWindowOverflowException):
            return AgentError("model context window exceeded", retryable=False,
                              code="model_context_window_exceeded")
        if isinstance(e, MaxTokensReachedException):
            return AgentError("model stopped at max_tokens", retryable=False, code="max_tokens")
    for e in chain:
        response = getattr(e, "response", None)
        if isinstance(response, Mapping) and isinstance(response.get("Error"), Mapping):
            return agent_error_from_aws(e, operation="Bedrock ConverseStream")
    for e in chain:
        if isinstance(e, ModelThrottledException):
            return AgentError("model throttled", retryable=True, code="ThrottlingException")
    for e in chain:
        if isinstance(e, ModelUnusable):
            return unusable_error(e)
    return AgentError(f"Strands turn failed: {type(exc).__name__}", retryable=True)


# --------------------------------------------------------------------------- the client


@dataclass
class _Built:
    agent: Any
    limits: _LoopLimits
    #: The call's Memory session manager (``None``: history in memory only).
    manager: Any = None
    #: Whether the approved Gateway tools are on the agent (else a later turn tries again).
    gateway_bound: bool = False
    #: The model per list entry on this agent and the entry that produced each reasoning block.
    models: _CallModels = field(default_factory=lambda: _CallModels([]))


@dataclass
class _CallAgent:
    """One cached call: the agent's build (the session-start future) and last use."""

    build: asyncio.Task[_Built]
    last_used: float
    pending: dict[str, str] = field(default_factory=dict)
    #: The agent was rebuilt from Memory in the current session (2.4a recovery: a tool result for
    #: an unknown tool use triggers one reload only when this is False).
    reloaded_this_session: bool = False


def _succeeded(task: asyncio.Task[Any]) -> bool:
    """A finished task that returned (not cancelled, not failed)."""
    if task.cancelled():
        return False
    try:
        task.result()
    except Exception:  # noqa: BLE001 - the failure was already logged by whoever awaited it
        return False
    return True


#: ``(settings for one entry, session_id)`` → that entry's model (:meth:`StrandsSettings.only`).
ModelFactory = Callable[[StrandsSettings, str], Model]
#: The live ``model_failover`` policy section: an object with ``settings_for(model_ids)`` giving the
#: failover :class:`~owlbank.failover.Settings` (``owlbank.policy.ModelFailoverConfig``).
FailoverPolicySource = Callable[[], Any]
#: The fleet verdict for this turn, or ``None`` when there is none (no fleet
#: picture, or a missing, unreadable or stale record: the in-process picture alone).
FleetViewSource = Callable[[], FleetView | None]


class FleetRecorder(Protocol):
    """This process's report to the fleet (``tac_bridge.fleet.FleetReporter``): synchronous, no I/O,
    never raises."""

    def record(self, index: int, outcome: Classification, ttft_s: float | None = None) -> None: ...


class FailoverEvents(Protocol):
    """Model failover's metrics (``tac_bridge.model_metrics``): never raise."""

    def failover(self) -> None: ...

    def unavailable(self) -> None: ...

    def active(self, rank: int) -> None:
        """A turn starts on entry ``rank`` (``ActiveRank`` on a change)."""
        ...


#: ``model_failover`` causes of a call that starts on another entry (no call was abandoned): the
#: verdict (``degraded``), or a later hop of a turn whose entry is now unavailable or failed
#: with nowhere left.
_START_CAUSES = frozenset({"degraded", "unavailable", "exhausted"})
#: The ``escalate`` reasons' descriptions, or a zero-argument callable giving them (the live policy's).
ReasonDescriptions = Mapping[str, str] | Callable[[], Mapping[str, str]]
#: The ``end_call`` switch, or a zero-argument callable giving it (the live policy's).
EndCallSwitch = bool | Callable[[], bool]
#: ``(session_id, build mode)`` → a Strands session manager, or ``None`` for in-memory history.
SessionManagerFactory = Callable[[str, BuildMode], Any]


def _write_failed(built: _Built, exc: BaseException) -> bool:
    """The turn failed because a Memory write (the invocation's flush) failed."""
    if getattr(built.manager, "flush_error", None) is not None:
        return True
    return built.manager is not None and any(
        isinstance(e, SessionException) and not isinstance(e, HistoryUnavailable) for e in _causes(exc))


class StrandsAgentClient:
    """:class:`~tac_bridge.agent_client.AgentClient` on a Strands ``Agent`` per call.

    ``model_factory(settings, session_id)`` builds the model (default: one shared ``BedrockModel``).
    ``gateway_tools`` gives the listed Gateway tools (default: none). ``session_manager_factory
    (session_id, mode)`` makes the call's session manager; by default it is the AgentCore Memory one
    (:func:`memory_session_manager_factory`) when ``settings.memory_id`` is set, else history stays in
    memory. ``clock`` is for the cache's idle eviction (tests). ``cards``: the ``card`` options that
    ``make_payment.to`` offers (default: the seed's); the bridge checks ``to`` against the live policy.
    ``reason_descriptions``: the ``escalate`` reasons' descriptions, a mapping or a zero-argument
    callable read at each agent build (the live policy's: :meth:`use_reason_descriptions`); default
    the seed's.

    **Model list**. ``model_factory`` is called once per entry
    (with :meth:`StrandsSettings.only`); without one, each entry's ``BedrockModel`` is built once for
    the process. With two or more entries each runs as a :class:`GuardedModel` (the first-token
    deadline, the history filter); each turn starts on ``failover_state.choose`` and keeps that entry
    for its later model calls; a call that fails before any output (:class:`ModelUnusable`) moves the
    turn to ``next_after`` by continuing from the history (``stream_async(None)``), so the caller's
    message is never sent twice. A list of one runs today's path (no wrapper). ``failover_state``
    (default: one for the list, with the default settings) records every model call's outcome and
    TTFT; :meth:`use_failover_policy` gives the live ``model_failover`` section (read per turn) and
    :meth:`use_fleet_view` the fleet verdict."""

    def __init__(self, settings: StrandsSettings, *,
                 model_factory: ModelFactory | None = None,
                 gateway_tools: GatewayToolProvider | None = None,
                 session_manager_factory: SessionManagerFactory | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 cards: Collection[str] = DEFAULT_CARDS,
                 reason_descriptions: ReasonDescriptions = DEFAULT_REASON_DESCRIPTIONS,
                 failover_state: FailoverState | None = None,
                 failover_policy: FailoverPolicySource | None = None,
                 fleet_view: FleetViewSource | None = None,
                 fleet_reporter: FleetRecorder | None = None,
                 failover_events: FailoverEvents | None = None) -> None:
        self.settings = settings
        self._model_factory = model_factory
        self._shared_models: dict[int, Model] = {}
        if failover_state is None:
            failover_state = FailoverState(Settings.for_models(settings.model_ids))
        elif [e.model_id for e in failover_state.settings.entries] != list(settings.model_ids):
            raise ValueError("failover_state must be for the settings' model list")
        self._failover = failover_state
        self._failover_policy = failover_policy
        self._fleet_view = fleet_view
        self._fleet_reporter = fleet_reporter
        self._failover_events = failover_events
        #: Per session: its agent-call count and its current turn's move (:class:`_SessionFailover`).
        self._sessions: OrderedDict[str, _SessionFailover] = OrderedDict()  # least recently used first
        self._gateway = gateway_tools
        if session_manager_factory is None and settings.memory_id:
            session_manager_factory = memory_session_manager_factory(settings.memory_id, settings.region)
        self._session_manager_factory = session_manager_factory
        self._clock = clock
        self._cards = cards
        self._reason_descriptions: ReasonDescriptions = reason_descriptions
        self._end_call: EndCallSwitch = True
        #: The inline tools for the descriptions and switch last read (rebuilt only when they change).
        self._inline_for: tuple[tuple[tuple[str, str], ...], bool] | None = None
        self._inline: list[Any] = self._inline_tools()
        self._calls: OrderedDict[str, _CallAgent] = OrderedDict()
        #: The bridge's span helper (``tac_bridge.bridge._Tracing``), set by :meth:`enable_tracing`
        #: when telemetry is on; ``None`` otherwise.
        self._tracing: Any = None

    def use_reason_descriptions(self, source: ReasonDescriptions) -> None:
        """The ``escalate`` tool's reason descriptions from now on (a mapping,
        or a zero-argument callable such as ``lambda: policy.reason_descriptions``, read at each
        agent build so a policy change reaches the next call). Agents already built keep theirs."""
        self._reason_descriptions = source
        self._inline = self._inline_tools()

    def use_end_call_switch(self, source: EndCallSwitch) -> None:
        """Whether the model is offered ``end_call``, a bool or a zero-argument
        callable such as ``policy.end_call_enabled``, read at each agent build, so a switch change
        applies from the next call. Agents already built keep their tools."""
        self._end_call = source
        self._inline = self._inline_tools()

    def _end_call_enabled(self) -> bool:
        """The switch; a source that fails or gives a non-bool reads as on (the policy's default)."""
        source = self._end_call
        try:
            value = source() if callable(source) else source
        except Exception as exc:  # noqa: BLE001 - on, as a malformed policy section reads
            log_caught(log, "end_call switch unavailable", exc, operation="end_call_switch",
                       decision="offer_end_call")
            return True
        return value if isinstance(value, bool) else True

    def _inline_tools(self) -> list[Any]:
        """The inline tools with the current reason descriptions and ``end_call`` switch. A source
        that fails or gives something other than a mapping of strings: the seed's descriptions,
        one warning."""
        source = self._reason_descriptions
        try:
            raw = source() if callable(source) else source
            if not isinstance(raw, Mapping) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
                raise TypeError("reason descriptions must map strings to strings")
            descriptions: Mapping[str, str] = raw
        except Exception as exc:  # noqa: BLE001 - the seed's descriptions still tell the reasons apart
            log_caught(log, "reason descriptions unavailable", exc, operation="reason_descriptions",
                       decision="use_seed_descriptions")
            descriptions = DEFAULT_REASON_DESCRIPTIONS
        enabled = self._end_call_enabled()
        key = (tuple(descriptions.items()), enabled)
        if key != self._inline_for or not getattr(self, "_inline", None):
            self._inline = inline_tools(self._cards, descriptions,
                                        exclude=() if enabled else (END_CALL,))
            self._inline_for = key
        return self._inline

    @property
    def failover(self) -> FailoverState:
        """The in-process decision state over the model list."""
        return self._failover

    def use_failover_policy(self, source: FailoverPolicySource | None) -> None:
        """Model failover: the live ``model_failover`` section from now on, read at the start
        of each turn (``lambda: policy.model_failover``, cached by the policy object)."""
        self._failover_policy = source

    @property
    def multi_model(self) -> bool:
        """Two or more entries: failover is on (a list of one runs today's path)."""
        return len(self.settings.models) > 1

    def use_fleet_reporter(self, reporter: FleetRecorder | None) -> None:
        """Model failover: every model call's outcome also goes to ``reporter`` (with two or
        more entries)."""
        self._fleet_reporter = reporter

    def use_failover_events(self, events: FailoverEvents | None) -> None:
        """Model failover: the ``Failover``, ``Unavailable`` and ``ActiveRank`` metrics."""
        self._failover_events = events

    def active_rank(self) -> int:
        """The entry a turn would start on now (``ActiveRank``; 0 for a list of one). Pure, in memory."""
        if not self.multi_model:
            return 0
        return self._failover.choose(fleet=self._fleet())

    def use_fleet_view(self, source: FleetViewSource | None) -> None:
        """Model failover (wired): the fleet verdict each turn's choice and each
        reissue follow, overridden by this process's own evidence (``FailoverState.choose``)."""
        self._fleet_view = source

    def _refresh_failover(self) -> None:
        """The policy section's settings for the list, if they changed. A failing source keeps the
        current settings (logged at most once a minute, field and class only)."""
        source = self._failover_policy
        if source is None:
            return
        try:
            new = source().settings_for(self.settings.model_ids)
            if new != self._failover.settings:
                self._failover.update_settings(new)
        except Exception as exc:  # noqa: BLE001 - the thresholds in use stay
            _hook_failed("failover_policy", exc)

    def _fleet(self) -> FleetView | None:
        source = self._fleet_view
        if source is None:
            return None
        try:
            view = source()
        except Exception as exc:  # noqa: BLE001 - no fleet verdict: the in-process picture
            _hook_failed("fleet_view", exc)
            return None
        if view is not None and len(view.states) != len(self.settings.models):
            return None
        return view

    def _on_model_call(self, entry: int, exc: BaseException | None, stop: str | None,
                       ttft_s: float | None) -> None:
        """Every model call's outcome and TTFT into the decision state: a success with
        its time to the first output; an abandoned or failed call by its cause or error class. A call
        the turn's cancellation ended is not a sample. With two or more entries the same outcome goes
        to the fleet reporter (an abandoned call with the deadline it missed), a request error
        is logged (``model_request_error``) and a mark this process sets is logged
        (``model_degraded``, ``model_unavailable``). Never raises into the hook."""
        code: str | None = None
        if exc is None:
            if stop == "cancelled":
                return
            outcome: Classification = Outcome.OK
        else:
            unusable = model_unusable(exc)
            if unusable is not None:
                outcome, ttft_s, code = unusable.cause, None, unusable.error_code
            else:
                code, message = _error_code(exc)
                outcome, ttft_s = classify(code, message), None
        change = self._failover.record(entry, outcome, ttft_s)
        if not self.multi_model:
            return  # today's path: no report, no new line
        model_id = self.settings.models[entry].id
        if outcome is Outcome.REQUEST_ERROR:
            emit(log, logging.WARNING, "model_request_error", entry=entry, model_id=model_id,
                 error_code=code, decision="no_state_change")
            return
        reporter = self._fleet_reporter
        if reporter is not None:
            reported = ttft_s
            if outcome is Outcome.ABANDONED:
                reported = self._failover.settings.entry(entry).deadline_s
            reporter.record(entry, outcome, reported)
        if change is not None:
            self._log_mark(change, code)

    def _log_mark(self, change: Transition, code: str | None) -> None:
        """``model_unavailable`` (4a.4: the entry and the error code; the bridge log group's alarm
        fires on it) or ``model_degraded`` (5.1: the window's numbers) for a mark this process set."""
        model_id = self.settings.models[change.entry].id
        if change.new is EntryState.UNAVAILABLE:
            emit(log, logging.ERROR, "model_unavailable", entry=change.entry, model_id=model_id,
                 error_code=code, preferred=change.entry == 0, scope="process")
            self._event("unavailable")
        elif change.new is EntryState.DEGRADED:
            st = change.stats
            emit(log, logging.WARNING, "model_degraded", entry=change.entry, model_id=model_id,
                 p50_ms=_ms(st.p50_s) if st else None, p90_ms=_ms(st.p90_s) if st else None,
                 error_rate=round(st.error_rate, 3) if st else None,
                 samples=st.calls if st else None, miss_run=st.miss_run if st else None, scope="process")

    def _event(self, name: str, *args: Any) -> None:
        events = self._failover_events
        if events is None:
            return
        try:
            getattr(events, name)(*args)
        except Exception as exc:  # noqa: BLE001 - a metric never fails a turn
            _hook_failed("failover_events", exc)

    def _session_failover(self, session_id: str) -> _SessionFailover:
        rec = self._sessions.get(session_id)
        if rec is None:
            rec = self._sessions[session_id] = _SessionFailover(last_used=self._clock())
        rec.last_used = self._clock()
        self._sessions.move_to_end(session_id)  # kept in last-used order, so _evict stops early
        return rec

    def _log_failover(self, turn: int, src: int, dst: int, cause: str, *, model_calls: int,
                      error_code: str | None = None) -> None:
        """``model_failover`` (5.1: the turn, from, to and cause; never text) and the ``Failover``
        metric."""
        emit(log, logging.WARNING if cause != "degraded" else logging.INFO, "model_failover", turn=turn,
             from_model=self.settings.models[src].id, to_model=self.settings.models[dst].id, cause=cause,
             from_entry=src, to_entry=dst, error_code=error_code, model_calls=model_calls,
             decision="start_on_next_model" if cause in _START_CAUSES else "continue_on_next_model")
        self._event("failover")

    def enable_tracing(self, tracing: Any) -> None:
        """A ``memory.reload`` span around each rebuild from Memory
        (``tracing.child``), so the reload's botocore spans nest under it. Called by the bridge's
        ``enable_tracing``."""
        self._tracing = tracing

    # ------------------------------------------------------------------ build and cache

    def _model(self, session_id: str, index: int = 0) -> Model:
        """Entry ``index``'s model: the factory's, or the process's one ``BedrockModel`` for it."""
        if self._model_factory is not None:
            return self._model_factory(self.settings.only(index), session_id)
        model = self._shared_models.get(index)
        if model is None:  # one boto client per entry for every call
            model = self._shared_models[index] = default_model_factory(self.settings.only(index))
        return model

    def _guard(self, inner: Model, index: int) -> GuardedModel:
        s = self.settings
        return GuardedModel(inner, entry=index, deadline_s=self._failover.settings.entry(index).deadline_s,
                            stage=s.stage, test_delay_s=s.failover_test_s if index == 0 else None)

    def _call_models(self, session_id: str) -> _CallModels:
        """The agent's model per entry: the bare model for a list of one (today's path), else a
        :class:`GuardedModel` per entry."""
        n = len(self.settings.models)
        inner = [self._model(session_id, i) for i in range(n)]
        if n == 1:
            return _CallModels(inner)
        return _CallModels([self._guard(m, i) for i, m in enumerate(inner)])

    def _model_for(self, built: _Built, index: int) -> Model:
        """Entry ``index``'s model on ``built``, its guard renewed if the policy changed its deadline."""
        model = built.models.models[index]
        if isinstance(model, GuardedModel) and model.deadline_s != self._failover.settings.entry(index).deadline_s:
            model = built.models.models[index] = self._guard(model.inner, index)
        return model

    def prewarm(self) -> None:
        """Pay the clients' first-use cost at start-up, not on a call's first turn: the shared
        Bedrock model (its ``bedrock-runtime`` client) and the Memory clients. Blocking (run it in a
        thread); no AWS call. Best effort: a failure is logged and the cost moves to the first call."""
        started = time.monotonic()
        try:
            if self._model_factory is None:
                for i in range(len(self.settings.models)):
                    self._model("", i)
            warm = getattr(self._session_manager_factory, "prewarm", None)
            if warm is not None:
                warm()
        except Exception as exc:  # noqa: BLE001 - only a latency cost if it fails
            log_caught(log, "strands client prewarm failed", exc, operation="prewarm",
                       decision="continue_cold")
            return
        emit(log, logging.INFO, "strands_prewarmed",
             duration_ms=round((time.monotonic() - started) * 1000, 1))

    def _gateway_tools(self, *, quiet: bool = False) -> list[Any]:
        """The approved Gateway tools, or none if the provider fails (warn, carry on).
        ``quiet``: a retry at a later turn, logged at DEBUG."""
        if self._gateway is None:
            return []
        try:
            listed = list(self._gateway.tools())
        except Exception as exc:  # noqa: BLE001 - the model answers without the Level-0 tools
            log_caught(log, "gateway tools unavailable", exc, level=logging.DEBUG if quiet else logging.WARNING,
                       operation="gateway_tools", decision="omit_gateway_tools")
            return []
        keep: list[Any] = []
        names: set[str] = set()
        for t in listed:
            name = getattr(t, "tool_name", None)
            if name in ALLOWED_GATEWAY_TOOLS and name not in names:
                keep.append(t)
                names.add(name)
        if len(keep) != len(listed):
            emit(log, logging.WARNING, "gateway_tools_dropped", listed=len(listed), kept=len(keep),
                 decision="drop_unapproved")
        return keep

    def _build_agent(self, session_id: str, mode: BuildMode, reason: str, history: str | None = None) -> _Built:
        """ step 1. Runs in a worker thread: the session manager reads Memory while the
        Agent is constructed (``reload``), synchronously even in async mode. A failed read or
        session write is a retryable ``history_unavailable``. With tracing, a reload runs in a
        ``memory.reload`` span (``owlbank.history``: the bridge's decision, ``owlbank.reason``) under
        the caller's context: the turn span, or the session span at session start."""
        tracing = self._tracing
        if tracing is None or mode != "reload" or self._session_manager_factory is None:
            return self._build(session_id, mode, reason)
        with tracing.child(SPAN_MEMORY_RELOAD, session_id, {"owlbank.history": history, "owlbank.reason": reason}):
            return self._build(session_id, mode, reason)

    def _build(self, session_id: str, mode: BuildMode, reason: str) -> _Built:
        s = self.settings
        limits = _LoopLimits(s.max_iterations)
        gateway = self._gateway_tools()
        started = time.monotonic()
        manager: Any = None
        try:
            if self._session_manager_factory is not None:
                manager = self._session_manager_factory(session_id, mode)
            models = self._call_models(session_id)
            agent = Agent(
                model=models.models[0],
                tools=[*self._inline_tools(), *gateway],
                callback_handler=None,
                # per_turn: trim before every model call, so no request carries more than the window
                conversation_manager=SlidingWindowConversationManager(window_size=s.window_messages, per_turn=True),
                hooks=[limits, _ModelCallTelemetry(limits, model_ids=s.model_ids, observer=self._on_model_call),
                       _ReasoningProducers(models), _ToolTelemetry()],
                agent_id=AGENT_ID,
                # Every Strands span carries the call, as attributes and as baggage (the
                # bridge's turn span normally supplies the baggage; this covers a span outside it).
                trace_attributes={"session.id": session_id, "call_sid": session_id},
                baggage_attributes={"session.id": session_id, "call_sid": session_id},
                # No retries inside the loop: a throttle or 5xx surfaces as a retryable AgentError and the
                # bridge's own retry and apology path handles it, within the turn budget.
                retry_strategy=None,
                **({"session_manager": manager} if manager is not None else {}),
            )
        except Exception as exc:
            if self._session_manager_factory is None:
                raise
            self._log_history(mode, reason, started, None)
            raise AgentError("call history unavailable", retryable=True, code="history_unavailable") from exc
        if self._session_manager_factory is not None:
            self._log_history(mode, reason, started, agent)
        if self.multi_model:
            tag_model_metrics(agent, s.model_ids)  # TTFT and latency per model_id
        return _Built(agent, limits, manager, gateway_bound=bool(gateway), models=models)

    @staticmethod
    def _log_history(mode: BuildMode, reason: str, started: float, agent: Any) -> None:
        """``agent_history_reload`` for every rebuild from Memory, ``agent_history_new`` for
        an empty start: the reason and the time taken, counts only (never content)."""
        fields: dict[str, Any] = {"reason": reason,
                                  "duration_ms": round((time.monotonic() - started) * 1000, 1),
                                  "outcome": "ok" if agent is not None else "error"}
        if agent is not None:
            fields["messages"] = len(agent.messages)
            fields["interrupted"] = bool(agent._interrupt_state.activated)
        emit(log, logging.INFO if agent is not None else logging.WARNING,
             "agent_history_reload" if mode == "reload" else "agent_history_new", **fields)

    def _evict(self) -> None:
        """Idle entries first, then the least recently used past the bound. An evicted
        agent's Memory buffer is empty (each invocation flushes); it is dropped, never flushed late
        (a late flush could write a stale state after the call moved on)."""
        now = self._clock()
        while self._calls:
            sid, entry = next(iter(self._calls.items()))
            if now - entry.last_used <= self.settings.cache_idle_s:
                break
            del self._calls[sid]
            emit(log, logging.INFO, "strands_agent_evicted", reason="idle", cached=len(self._calls))
        while len(self._calls) > self.settings.cache_max_calls:
            self._calls.popitem(last=False)
            emit(log, logging.INFO, "strands_agent_evicted", reason="lru", cached=len(self._calls))
        while self._sessions:  # a closed call (a live turn is never idle that long); oldest first
            sid, rec = next(iter(self._sessions.items()))
            if now - rec.last_used <= self.settings.cache_idle_s:
                break
            del self._sessions[sid]

    def _plan(self, entry: _CallAgent | None, history: HistoryMode) -> tuple[BuildMode, str] | None:
        """How to get the call's agent: ``None`` uses the cached one; else ``(mode, reason)``.

        * ``new``: always an empty agent, no Memory read (the call's first AI session).
        * ``cached``: the cached agent; if it was evicted (or its build failed), reload.
        * ``reload``: rebuild from Memory (another process ran a session since). Without a store
          there is nothing to reload from, so the cached agent stays (behaviour)."""
        if history == "new":
            return "new", "new_session"
        if entry is None:
            return "reload", "owner_changed" if history == "reload" else "cache_miss"
        if entry.build.done() and not _succeeded(entry.build):
            return "reload", "build_failed"
        if history == "reload" and self._session_manager_factory is not None:
            return "reload", "owner_changed"
        return None

    def _entry(self, session_id: str, history: HistoryMode, *, session_start: bool = False,
               force: tuple[BuildMode, str] | None = None) -> _CallAgent:
        """The call's cache entry for ``history`` (:meth:`_plan`), or a new build (``force``: this
        build whatever the cache holds). ``session_start``: a new ConversationRelay session, so a
        kept agent was not rebuilt from Memory in it."""
        self._evict()
        entry = self._calls.get(session_id)
        plan = force or self._plan(entry, history)
        if entry is not None and plan is None:
            entry.last_used = self._clock()
            self._calls.move_to_end(session_id)
            if session_start:
                entry.reloaded_this_session = False
            return entry
        mode, reason = plan or ("reload", "cache_miss")
        task = asyncio.create_task(asyncio.to_thread(self._build_agent, session_id, mode, reason, history))
        entry = _CallAgent(build=task, last_used=self._clock(),
                           reloaded_this_session=mode == "reload" and self._session_manager_factory is not None)
        self._calls[session_id] = entry
        self._calls.move_to_end(session_id)
        self._evict()
        return entry

    def _drop(self, session_id: str, entry: _CallAgent | None) -> None:
        if entry is not None and self._calls.get(session_id) is entry:
            del self._calls[session_id]

    async def _built(self, session_id: str, history: HistoryMode, *, session_start: bool = False,
                     force: tuple[BuildMode, str] | None = None) -> _Built:
        entry = self._entry(session_id, history, session_start=session_start, force=force)
        try:
            return await asyncio.shield(entry.build)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._drop(session_id, entry)  # the next attempt builds again
            err = to_agent_error(exc)
            log_caught(log, "strands agent build failed", exc, operation="agent_build",
                       decision="fail_turn", agent_error=err.code)
            raise err from exc

    async def begin_session(self, session_id: str, *, history: HistoryMode = "reload") -> None:
        """Prepare the call's agent at session start (the bridge runs this in the
        background, and the session's first turn awaits it). ``history`` is the bridge's session
        ownership decision (see :meth:`_plan`). A failed history read raises a retryable
        ``history_unavailable`` :class:`AgentError`; the next turn then tries the read again."""
        if history not in HISTORY_MODES:
            raise ValueError(f"history must be one of {HISTORY_MODES}")
        rec = self._sessions.get(session_id)
        if rec is not None:
            rec.move = None  # A new session's turn numbers start again
        await self._built(session_id, history, session_start=True)

    async def aclose(self) -> None:
        """Drop every cached agent, flushing its Memory session manager (``close()``: anything still
        buffered, except the snapshot of an agent that never ran, which could be stale:
        :meth:`_NewestStateSessionManager.drop_unrun_agent_state`), and stop the Gateway client.
        Best effort."""
        entries = list(self._calls.values())
        self._calls.clear()
        self._sessions.clear()
        for entry in entries:
            if not (entry.build.done() and _succeeded(entry.build)):
                continue
            manager = entry.build.result().manager
            close = getattr(manager, "close", None)
            if close is None:
                continue
            try:
                drop = getattr(manager, "drop_unrun_agent_state", None)
                if drop is not None and drop():
                    emit(log, logging.DEBUG, "memory_unrun_state_dropped", operation="memory_close",
                         decision="drop_stale_snapshot")
                await asyncio.to_thread(close)
            except Exception as exc:  # noqa: BLE001 - shutdown is best effort
                log_caught(log, "memory session manager close failed", exc, operation="memory_close",
                           decision="ignored_best_effort")
        close_gw = getattr(self._gateway, "close", None)
        if close_gw is not None:
            try:
                close_gw()
            except Exception as exc:  # noqa: BLE001 - shutdown is best effort
                log_caught(log, "gateway MCP client stop failed", exc, operation="gateway_stop",
                           decision="ignored_best_effort")

    def _defer_tool_call(self, name: str) -> bool:
        """A hand-off's :class:`ToolCall` is emitted at the invocation's ``result``, after the
        Memory flush, because the next session may run on another task; other inline calls
        at their interrupt."""
        return name in HANDOFF_TOOLS

    def _bind_late_gateway(self, built: _Built) -> None:
        """An agent built while the Gateway was unavailable gets the approved Gateway tools at
        the start of a later turn, once the provider has them (it caches its listing). The same
        allow-list filter applies."""
        if built.gateway_bound or self._gateway is None:
            return
        late = self._gateway_tools(quiet=True)
        if not late:
            return
        built.agent.tool_registry.process_tools(late)
        built.gateway_bound = True
        emit(log, logging.INFO, "gateway_tools_recovered", tools=len(late))

    def _may_recover(self, entry: _CallAgent | None) -> bool:
        """The unknown-tool-use recovery applies: there is a Memory to reload from, and the call's
        agent was not already rebuilt from it in this session."""
        return self._session_manager_factory is not None and entry is not None \
            and not entry.reloaded_this_session

    @staticmethod
    async def _ensure_flushed(built: _Built) -> None:
        """At the invocation's ``result``: nothing of the invocation may still be buffered.
        The library flushes automatically when the invocation ends, before ``result``, so
        this normally does nothing; if anything is still buffered (a library change), flush it here
        (the library has no public flush; ``_flush_messages`` is what its own hook calls)."""
        manager = built.manager
        pending = getattr(manager, "pending", None)
        if pending is None or not pending():
            return
        emit(log, logging.WARNING, "memory_flush_explicit", pending=pending(), decision="flush_before_result")
        await asyncio.to_thread(manager._flush_messages)

    # ------------------------------------------------------------------ the turn

    async def record_exchange(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """The session's welcome as the agent's own turn, with no
        model call. Both messages go through Strands' own message path (``Agent._append_messages``,
        what the event loop uses for every message): each is appended to ``agent.messages`` and
        fires ``MessageAddedEvent``, on which the AgentCore Memory session manager appends the
        message and syncs the agent state, as for any other turn. With its batched writes they are
        buffered and written by the next invocation's flush (``AfterInvocationEvent``, which
        Strands fires in a ``finally``); the bridge always runs the hop's agent call right after
        this, and a turn that makes no invocation flushes them itself (:meth:`_flush_buffered`).

        A history ending with an unanswered user message (a failed turn's input, or results
        recorded with no reply): only the reply is added, so the roles still alternate. A tool use
        still waiting for its result: nothing is added (its ``toolResult`` must come next; the
        bridge puts the welcome in the result's envelope instead), one warning."""
        if not user_text.strip() or not assistant_text.strip():
            emit(log, logging.INFO, "strands_exchange_skipped", reason="blank_text")
            return
        built = await self._built(session_id, "cached")  # the session start's agent, or a reload
        agent = built.agent
        if pending_interrupts(agent):
            emit(log, logging.WARNING, "strands_exchange_skipped", reason="tool_use_pending",
                 decision="history_unchanged")
            return
        reply: dict[str, Any] = {"role": "assistant", "content": [{"text": assistant_text}]}
        messages: list[Any] = [reply]
        if unanswered_input(agent) is None:
            messages.insert(0, {"role": "user", "content": [{"text": user_text}]})
        try:
            await agent._append_messages(*messages)  # noqa: SLF001 - Strands' message path (see above)
        except Exception as exc:
            self._drop(session_id, self._calls.get(session_id))  # the retry rebuilds from Memory
            err = to_agent_error(exc)
            log_caught(log, "recording the exchange failed", exc, operation="record_exchange",
                       decision="raise_agent_error", agent_error=err.code)
            raise err from exc
        emit(log, logging.INFO, "strands_exchange_recorded", messages=len(messages))

    @staticmethod
    async def _flush_buffered(built: _Built) -> None:
        """A turn that ends with no invocation: anything buffered (an exchange recorded just before
        it, :meth:`record_exchange`) is written now, as an invocation's own flush would have."""
        manager = built.manager
        pending = getattr(manager, "pending", None)
        if pending is None or not pending():
            return
        emit(log, logging.DEBUG, "memory_flush_without_invocation", pending=pending())
        await asyncio.to_thread(manager._flush_messages)

    def record_results(self, session_id: str, tool_results: Sequence[ToolResult], *,
                       system_prompts: Mapping[str, str] | None,
                       system_suffix: str | None = None,
                       turn_id: int | None = None) -> AsyncGenerator[AgentEvent, None]:
        """Answer the pending interrupts with ``tool_results`` (the caller already heard
        the outcome) and stop before the model is called: the history gets the ``toolResult``
        message, the stream is just :class:`TurnEnd`. The next turn's words follow it."""
        return self._turn(session_id, user_text=None, tool_results=tool_results,
                          system_prompts=system_prompts, system_suffix=system_suffix, respond=False,
                          turn_id=turn_id)

    def turn(self, session_id: str, *, user_text: str | None,
             tool_results: Sequence[ToolResult],
             system_prompts: Mapping[str, str] | None,
             system_suffix: str | None = None,
             turn_id: int | None = None) -> AsyncGenerator[AgentEvent, None]:
        return self._turn(session_id, user_text=user_text, tool_results=tool_results,
                          system_prompts=system_prompts, system_suffix=system_suffix, respond=True,
                          turn_id=turn_id)

    async def _turn(self, session_id: str, *, user_text: str | None,
                    tool_results: Sequence[ToolResult],
                    system_prompts: Mapping[str, str] | None,
                    system_suffix: str | None,
                    respond: bool, turn_id: int | None = None) -> AsyncGenerator[AgentEvent, None]:
        built = await self._built(session_id, "cached")  # the session start's agent, or a reload
        agent = built.agent
        entry = self._calls.get(session_id)
        self._bind_late_gateway(built)
        pending = pending_interrupts(agent)
        if entry is not None:
            entry.pending = pending
        results = {r.tool_use_id: r for r in tool_results}
        unknown = [tid for tid in results if tid not in pending]
        retry = retry_continues(agent, pending, results, user_text)
        if not retry and unknown and results_recorded(agent, pending, results, user_text):
            # An earlier try recorded these results (no model call), then failed before the
            # words: send only the words.
            emit(log, logging.INFO, "strands_results_already_recorded", results=len(results),
                 decision="send_words_only")
            results, unknown = {}, []
        if unknown and not retry and self._may_recover(entry):
            # Safety net: a result for a tool use this agent never made, on an agent not
            # rebuilt from Memory in this session (``new``, or ``cached``): the history decision
            # may have been wrong (an owner lost to an older writer). Rebuild once from
            # Memory and match again; never more than once per turn.
            emit(log, logging.WARNING, "strands_unknown_tool_use", results=len(results),
                 unknown=len(unknown), pending=len(pending), decision="reload_history")
            self._drop(session_id, entry)
            built = await self._built(session_id, "reload", force=("reload", "unknown_tool_use_recovery"))
            agent = built.agent
            entry = self._calls.get(session_id)
            self._bind_late_gateway(built)
            pending = pending_interrupts(agent)
            if entry is not None:
                entry.pending = pending
            unknown = [tid for tid in results if tid not in pending]
            retry = retry_continues(agent, pending, results, user_text)
        if retry and not respond:
            # The results are already recorded; recording adds nothing and calls no model.
            emit(log, logging.INFO, "strands_retry_continues", results=len(results),
                 text=False, decision="already_recorded")
            yield TurnEnd("end_turn", None)
            return
        if retry:
            emit(log, logging.INFO, "strands_retry_continues", results=len(results),
                 text=user_text is not None, decision="continue_from_history")
        elif unknown:
            emit(log, logging.WARNING, "strands_unknown_tool_use", results=len(results),
                 unknown=len(unknown), pending=len(pending), decision="reject")
            raise AgentError("tool result for an unknown tool use", retryable=False, code="unknown_tool_use")
        if user_text is not None and not user_text.strip():
            # Blank words are no words: Memory drops a whitespace-only text block, so it
            # would not survive a reload, and Bedrock rejects an empty one. As the bridge does with
            # blank words after a hand-off result, nothing is sent to the model for them.
            emit(log, logging.INFO, "strands_blank_text_skipped", chars=len(user_text), pending=len(pending))
            user_text = None
            if not pending:
                await self._flush_buffered(built)
                yield TurnEnd("end_turn", None)
                return
        invocations = [(None, None)] if retry else plan_invocations(pending, results, user_text,
                                                                     respond=respond)

        # Model failover: the turn's entry is chosen once, here, and kept for its later model
        # calls (a tool loop) unless a call on it fails before any output (below). 3.3a: a later
        # hop of the same bridge turn (``turn_id``) starts where the turn's move left it.
        # A record-only hop (``respond=False``) calls no model: it neither chooses nor reports, and
        # leaves the turn's move as it is.
        rec = self._session_failover(session_id) if self.multi_model else None
        move = rec.move if rec is not None and turn_id is not None else None
        if move is not None and move.turn_id != turn_id:
            move = None  # a new turn chooses afresh
        abandoned: frozenset[int] = frozenset()
        turn_no = 0
        fleet: FleetView | None = None  # read only for a hop that calls a model (record-only: no read)
        if rec is None:
            current = 0
        elif not respond:
            current = move.entry if move is not None else (built.models.turn_entry or 0)
        else:
            self._refresh_failover()
            fleet = self._fleet()
            rec.turns += 1
            turn_no = rec.turns
            if move is None:
                current = self._failover.choose(fleet=fleet)
                self._event("active", current)
                before = built.models.turn_entry if built.models.turn_entry is not None else 0
                if current > before:  # the verdict moved this call down
                    self._log_failover(turn_no, before, current, "degraded", model_calls=0)
                built.models.turn_entry = current
            else:
                current, abandoned = self._continue_entry(move, fleet)
                self._event("active", current)
                if current != move.entry:  # every change of entry in a turn is a failover
                    self._log_failover(turn_no, move.entry, current,
                                       "exhausted" if move.failed else "unavailable", model_calls=0)
                elif abandoned:
                    emit(log, logging.INFO, "model_turn_sticky", turn=turn_no, entry=current,
                         model_id=self.settings.models[current].id, abandoned=sorted(abandoned),
                         decision="stay_on_switched_entry")
        state = _TurnState()
        built.limits.state = state
        translator = _Translator(state, self._defer_tool_call, release=self.settings.models[current].text_release)
        self._use_entry(built, current, translator, system_prompts, system_suffix)
        deadline = asyncio.get_running_loop().time() + self.settings.turn_timeout_s
        stop: str | None = None
        queue = list(invocations)
        tried = {current, *abandoned}  # Each entry gets the turn at most once (3.3a: all its hops)
        exhausted = False
        try:
            while queue:
                prompt, limits = queue.pop(0)
                stream = agent.stream_async(prompt, limits=limits)
                try:
                    async with aclosing(stream):
                        while True:
                            try:
                                # The deadline covers only the awaits on Strands, never a yield: the
                                # bridge reads each event from its own task (the timer race), so a
                                # timeout scope spanning a yield could cancel the bridge instead.
                                async with asyncio.timeout_at(deadline):
                                    ev = await anext(stream)
                                    if "result" in ev:
                                        await self._ensure_flushed(built)  # before any deferred hand-off
                            except StopAsyncIteration:
                                break
                            for out in translator.feed(ev):
                                yield out
                            if "result" in ev:
                                stop = getattr(ev["result"], "stop_reason", None)
                except Exception as exc:
                    unusable = model_unusable(exc)
                    if unusable is None or _write_failed(built, exc):
                        raise
                    # The call produced nothing (the guard held it back), so nothing was spoken
                    # and no assistant message was added; the caller's message (or the tool result)
                    # is already the history's last message, flushed to Memory. The next entry
                    # continues from it: never re-sent, never removed (each entry once).
                    nxt = self._failover.next_after(unusable.entry, tried=tried, fleet=fleet)
                    if nxt is None:
                        exhausted = True
                        raise unusable_error(unusable) from None
                    self._log_failover(turn_no, unusable.entry, nxt,
                                       "deadline" if unusable.is_deadline else "error",
                                       model_calls=state.model_calls, error_code=unusable.error_code)
                    current = nxt
                    tried.add(nxt)
                    self._use_entry(built, current, translator, system_prompts, system_suffix)
                    queue.insert(0, (None, None))
        except TimeoutError as exc:
            emit(log, logging.WARNING, "strands_turn_timeout", timeout_s=self.settings.turn_timeout_s,
                 model_calls=state.model_calls, decision="fail_turn_retryable")
            raise AgentError("turn timed out", retryable=True, code="timeout_exceeded") from exc
        except AgentError:
            raise
        except Exception as exc:
            if _write_failed(built, exc):
                # The history isn't durable, so nothing of this invocation may be acted on
                # (a hand-off above all). Drop the agent: the retry rebuilds from what Memory holds.
                self._drop(session_id, entry)
                log_caught(log, "memory write failed; failing the turn", exc, level=logging.ERROR,
                           operation="memory_flush", decision="fail_turn_retryable",
                           deferred_handoffs=len(translator.deferred))
                raise AgentError("call history write failed", retryable=True,
                                 code="history_write_failed") from exc
            err = to_agent_error(exc)
            log_caught(log, "strands turn failed", exc, level=logging.DEBUG, operation="strands_turn",
                       decision="raise_agent_error", agent_error=err.code)
            raise err from exc
        finally:
            built.limits.state = None
            if rec is not None and respond and turn_id is not None:
                rec.move = _TurnMove(turn_id, current, exhausted,
                                     frozenset(tried) if exhausted else frozenset(tried - {current}))
        if not respond and state.model_calls == 0:
            stop = "end_turn"  # recorded (the turn cap stopped the loop before any model call)
        yield TurnEnd(_STOP_REASON_NAMES.get(stop, stop) if stop else stop, dict(state.usage) or None,
                      model=self.settings.models[current].id if state.model_calls else None)

    def _continue_entry(self, move: _TurnMove, fleet: FleetView | None) -> tuple[int, frozenset[int]]:
        """The entry a later hop of turn ``move.turn_id`` starts on, and the entries the turn has
        abandoned. The entry the turn is on, unless it failed with nowhere left or is now unavailable
        Then :meth:`FailoverState.choose`'s rule over the entries not abandoned: the
        best-ranked usable one, else (all unavailable) the last. Only when every entry is abandoned
        does the hop choose afresh among all (the abandoned set starts again)."""
        n = len(self.settings.models)
        if len(move.abandoned) >= n:
            return self._failover.choose(fleet=fleet), frozenset()
        state = self._failover
        if not move.failed and state.state(move.entry, fleet=fleet) is not EntryState.UNAVAILABLE:
            return move.entry, move.abandoned
        nxt = state.next_after(move.entry, tried=move.abandoned, fleet=fleet)
        if nxt is None:  # every entry not abandoned is unavailable: the last of them, as choose does
            nxt = max(i for i in range(n) if i not in move.abandoned)
        return nxt, (move.abandoned | {move.entry}) - {nxt}

    def _use_entry(self, built: _Built, index: int, translator: _Translator,
                   system_prompts: Mapping[str, str] | None, system_suffix: str | None) -> None:
        """Run the agent's next model calls on entry ``index``: its model, its text release,
        its own system prompt (``system_prompts`` by the entry's model key, :func:`prompt_for`)
        and that prompt's cache point with the settings' TTL. Only
        between model calls (nothing of a call is held)."""
        entry = self.settings.models[index]
        prompt = prompt_for(system_prompts, self.settings.model_ids, index)
        built.agent.model = self._model_for(built, index)
        translator.stream = entry.text_release == "stream"
        built.agent.system_prompt = system_blocks(prompt, system_suffix, caching=entry.prompt_caching,
                                                  ttl=self.settings.prompt_cache_ttl)
