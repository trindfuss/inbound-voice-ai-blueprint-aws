"""The escalation briefing: one model call under one overall deadline.

The briefing is a short clause read to the caller when a call would have reached an agent. The
model's input is the structured facts (registry names, never a digit) plus, when present, the
agent's cleaned summary and the scrubbed tail of the transcript (both prepared by the caller); its
output is checked structurally (:func:`check_output`) before it is spoken; and any failure gives
``Briefing(None, "fallback", 0)``, for which the caller reads a deterministic catalog line.

:func:`generate` enforces the deadline itself on a module-level executor, so the wait on the model
never exceeds the budget and :func:`generate` never raises. Nothing here logs the text.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import threading
import time
import unicodedata
from collections.abc import Callable, Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from owlbank.obs import log_caught, timed_call
from owlbank.policy import ESTABLISHED_FACTORS
from owlbank.tools_contract import (
    DEFAULT_ESCALATION_REASONS,
    FALLBACK_REASON,
    GATEWAY_LEVEL0_TOOLS,
    HANDOFF_TOOLS,
    PROTECTED_TOOLS,
)

if TYPE_CHECKING:
    from owlbank.callstate import CallState

__all__ = [
    "BRIEFING_BUDGET_MS", "client_for_budget", "prewarm", "MAX_WORDS", "SYSTEM_PROMPT_KEY", "Briefing", "BriefingFacts",
    "BriefingModel", "BedrockBriefingModel", "build_facts", "check_output", "generate",
    "known_tools", "plain_words",
]

log = logging.getLogger(__name__)

#: The limit of the model's summary clause (about 12 seconds spoken).
MAX_WORDS = 30
#: Default overall deadline of one briefing; the caller reads ``OWLBANK_BRIEFING_BUDGET_MS``.
BRIEFING_BUDGET_MS = 4000
#: Catalog key of the system prompt (en-US only); read by name, never hard-coded here.
SYSTEM_PROMPT_KEY = "agent.briefing.system"
#: The system prompt is written in English; the briefing language goes in the facts and the prompt.
PROMPT_LANG = "en-US"
#: The resolver's model id and its environment variable (same model, no new setting).
DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-5-5"
MODEL_ID_ENV = "OWLBANK_RESOLVER_MODEL_ID"
MAX_TOKENS = 256
SUMMARY_MAX = 1000
TRANSCRIPT_MAX = 1500
CONNECT_TIMEOUT_S = 1.0

_FORBIDDEN = frozenset("<>&{}`")

#: Workers of the module-level pool. A call past its deadline keeps its worker until the client's
#: own timeouts end it; a full pool just makes the next briefing a fallback.
_POOL_WORKERS = 2
_POOL = ThreadPoolExecutor(max_workers=_POOL_WORKERS, thread_name_prefix="briefing")


@dataclass(frozen=True)
class BriefingFacts:
    """What the briefing is built from. ``factors`` and ``tools`` are plain words of registry
    members (digits removed, underscores as spaces); ``reason`` is the policy key. ``summary`` and
    ``transcript`` are the only free text, each prepared (cleaned, PII-masked, scrubbed) by the
    caller and kept to :data:`SUMMARY_MAX` and :data:`TRANSCRIPT_MAX` characters here too."""

    language: str
    level: int
    reason: str
    factors: tuple[str, ...]
    tools: tuple[str, ...]
    #: The agent's escalation summary, already prepared (``task_attributes.prepare_summary``).
    summary: str | None = None
    #: The last turns of the conversation, already scrubbed (``briefing_transcript.fetch``).
    transcript: str | None = None


@dataclass(frozen=True)
class Briefing:
    summary: str | None
    source: Literal["model", "fallback"]
    words: int


_FALLBACK = Briefing(None, "fallback", 0)


# ---------------------------------------------------------------------------- facts


def known_tools() -> frozenset[str]:
    """The tool registry: every tool name the call can record."""
    return frozenset((*GATEWAY_LEVEL0_TOOLS, *PROTECTED_TOOLS, *HANDOFF_TOOLS))


def plain_words(name: str) -> str:
    """``name`` as plain words: Unicode decimal digits removed, underscores as spaces."""
    kept = "".join(" " if c == "_" else c for c in name if unicodedata.category(c) != "Nd")
    return " ".join(kept.split())


def _plain_members(values: Any, allowed: Collection[str]) -> tuple[str, ...]:
    out: list[str] = []
    if not isinstance(values, (list, tuple)):
        return ()
    for v in values:
        if isinstance(v, str) and v in allowed:
            word = plain_words(v)
            if word and word not in out:
                out.append(word)
    return tuple(out)


def build_facts(state: CallState, journal_summary: Mapping[str, Any], *, reason: str,
                known_factors: Collection[str] = (),
                known_reasons: Collection[str] = (),
                tools: Collection[str] | None = None,
                summary: str | None = None,
                transcript: str | None = None) -> BriefingFacts:
    """Facts for ``state`` and the journal's ``summary_for_resume`` result.

    Tool names come from ``journal_summary['toolsCalled']`` and factor names from
    ``state.factors_verified`` (``factorsAsked`` is not used: asked is not verified).
    A name that is not a member of the tool registry (``tools``, default :func:`known_tools`) or of
    the policy's factors (``known_factors`` plus the established ones) is dropped. ``reason`` must
    be a policy reason (``known_reasons`` plus the defaults), else the fallback reason is used.
    """
    allowed_tools = known_tools() if tools is None else frozenset(tools)
    allowed_factors = frozenset(known_factors) | ESTABLISHED_FACTORS
    allowed_reasons = frozenset(known_reasons) | frozenset(DEFAULT_ESCALATION_REASONS) | {FALLBACK_REASON}
    safe_reason = reason if reason in allowed_reasons else FALLBACK_REASON
    # The level is the state's only: the journal's levelReached is the highest ever logged, which
    # overstates it after a verification failure resets the level to 0.
    level = state.level if isinstance(state.level, int) else 0
    return BriefingFacts(
        language=state.language,
        level=level,
        reason=safe_reason,
        factors=_plain_members(state.factors_verified, allowed_factors),
        tools=_plain_members(journal_summary.get("toolsCalled"), allowed_tools),
        summary=_free_text(summary, SUMMARY_MAX, keep="head"),
        transcript=_free_text(transcript, TRANSCRIPT_MAX, keep="tail"),
    )


def _free_text(text: Any, limit: int, *, keep: str) -> str | None:
    """``text`` (already prepared by the caller) kept to ``limit`` characters, or ``None`` when it
    is not a non-blank string. A transcript keeps its end, the most recent turns."""
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    if len(text) > limit:
        text = text[-limit:] if keep == "tail" else text[:limit]
    return text


def facts_payload(facts: BriefingFacts) -> dict[str, Any]:
    """The model's input: the facts as JSON-ready data. No numeric level; ``agent_summary`` and
    ``conversation`` only when present."""
    payload: dict[str, Any] = {
        "language": facts.language,
        "escalation_reason": plain_words(facts.reason),
        "factors_verified": list(facts.factors),
        "tools_run": list(facts.tools),
    }
    if facts.summary:
        payload["agent_summary"] = facts.summary
    if facts.transcript:
        payload["conversation"] = facts.transcript
    return payload


# ---------------------------------------------------------------------------- output check


def check_output(text: str, *, max_words: int = MAX_WORDS) -> str | None:
    """The whitespace-normalised ``text`` if it passes the structural checks, else ``None``.

    Rejects: not a string, empty, any Unicode number (category N*: digits, numerals such as
    ``\u00b2 \u00bd \u2460 \u216b``), more than ``max_words`` words, any of
    ``< > & { }`` or a backtick, a newline, any control or format character (category C*). No phrase matching.
    """
    if not isinstance(text, str):
        return None
    if "\n" in text or "\r" in text:
        return None
    if any(
        c in _FORBIDDEN
        or unicodedata.category(c)[0] in "NC"
        for c in text
        if c != "\t"
    ):
        return None
    words = text.split()
    if not words or len(words) > max_words:
        return None
    return " ".join(words)


# ---------------------------------------------------------------------------- the model


class BriefingModel(Protocol):
    def summarise(self, facts: BriefingFacts, *, deadline: float) -> str | None:
        """The summary clause for ``facts``; ``deadline`` is on the ``now`` clock of
        :func:`generate`. May raise; :func:`generate` handles it."""
        ...


_CLIENTS: dict[int, Any] = {}
_CLIENTS_LOCK = threading.Lock()


def _make_client(budget_ms: int = BRIEFING_BUDGET_MS) -> Any:
    import boto3
    from botocore.config import Config

    config = Config(connect_timeout=CONNECT_TIMEOUT_S, read_timeout=budget_ms / 1000,
                    retries={"mode": "standard", "total_max_attempts": 1})
    return boto3.client("bedrock-runtime", config=config)


def client_for_budget(budget_ms: int = BRIEFING_BUDGET_MS) -> Any:
    """The cached, lazily built ``bedrock-runtime`` client whose read timeout is ``budget_ms``
    (one per budget value). No SDK import or client is built before first use."""
    from owlbank.coldstart import LazyClient

    with _CLIENTS_LOCK:
        client = _CLIENTS.get(budget_ms)
        if client is None:
            client = _CLIENTS[budget_ms] = LazyClient(lambda: _make_client(budget_ms))
        return client


def prewarm(budget_ms: int = BRIEFING_BUDGET_MS) -> None:
    """Build the client for ``budget_ms`` now (call at INIT), so the first briefing does not pay
    for loading the SDK inside its deadline. The connection still opens on the first call."""
    _ = client_for_budget(budget_ms).meta  # a LazyClient builds on first attribute access


class BedrockBriefingModel:
    """Plain Converse over ``bedrock-runtime``: ``system`` (catalog ``agent.briefing.system``) and
    one user message of the facts as JSON. One attempt, ``maxTokens`` 256 and no sampling
    parameters: the current Claude models reject ``temperature`` (a 400 ``ValidationException``).
    The 30-word limit is enforced by :func:`check_output`, the cap only bounds a runaway reply."""

    def __init__(self, catalog: Any, *, client: Any | None = None, model_id: str | None = None,
                 budget_ms: int = BRIEFING_BUDGET_MS) -> None:
        self._catalog = catalog
        self._client = client if client is not None else client_for_budget(budget_ms)
        self.model_id = model_id or os.environ.get(MODEL_ID_ENV) or DEFAULT_MODEL_ID

    def request(self, facts: BriefingFacts) -> dict[str, Any]:
        """The exact Converse kwargs (exposed for tests)."""
        system = self._catalog.text(SYSTEM_PROMPT_KEY, PROMPT_LANG, language=facts.language)
        user = json.dumps(facts_payload(facts), sort_keys=True)
        return {
            "modelId": self.model_id,
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {"maxTokens": MAX_TOKENS},
        }

    def summarise(self, facts: BriefingFacts, *, deadline: float) -> str | None:
        kwargs = self.request(facts)
        with timed_call("bedrock", "converse"):
            response = self._client.converse(**kwargs)
        blocks = response["output"]["message"]["content"]
        parts = [b["text"] for b in blocks if isinstance(b, Mapping) and isinstance(b.get("text"), str)]
        return "".join(parts) if parts else None


# ---------------------------------------------------------------------------- generate


def generate(model: BriefingModel | None, facts: BriefingFacts, *,
             now: Callable[[], float] = time.monotonic,
             budget_ms: int = BRIEFING_BUDGET_MS) -> Briefing:
    """The briefing for ``facts``, within one overall deadline of ``budget_ms``; never raises.

    A ``None`` model, an exception, a timeout (the late result is discarded) or a failed check
    all give ``Briefing(None, "fallback", 0)``.
    """
    try:
        if model is None:
            return _FALLBACK
        started = now()
        deadline = started + budget_ms / 1000
        remaining = deadline - now()
        if remaining <= 0:
            return _FALLBACK
        ctx = contextvars.copy_context()
        future = _POOL.submit(ctx.run, model.summarise, facts, deadline=deadline)
        try:
            raw = future.result(timeout=max(0.0, deadline - now()))
        except FutureTimeout:
            future.cancel()
            log.warning("briefing_timeout")
            return _FALLBACK
        checked = check_output(raw) if raw is not None else None
        if checked is None:
            return _FALLBACK
        return Briefing(checked, "model", len(checked.split()))
    except Exception as exc:  # noqa: BLE001 - the briefing never fails the call
        with contextlib.suppress(Exception):
            log_caught(log, "briefing_error", exc, decision="fallback")
        return _FALLBACK
