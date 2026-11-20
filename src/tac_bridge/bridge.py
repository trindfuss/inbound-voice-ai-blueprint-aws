"""The per-call loop: prompt -> agent turn -> stream -> inline tool / handoff.

TAC calls :meth:`Bridge.on_message_ready` for each final ``prompt``. The bridge
streams the agent's text with ``vc.send_response(conversation_id, gen)`` and
returns ``None``. TAC sends ``last: true`` itself and, if the session has
``pending_handoff_data``, the ``end`` frame after it.

Per prompt:

1. **Call state** is read from the ``calls`` table by ``CallSid`` (never
   from TAC's in-process session). See :class:`CallView` for the fields.
2. **System prompt** (first prompt, or when the verified customer changes):
   catalog ``agent.system.<model>`` in the call's language, one per entry of the model list
    + the back-end profile + TAC Memory. Memory is loaded only for a verified caller, through TAC's Memory
   client, for the customer's ``user_id`` (never the ANI profile). The bridge sets
   ``context.profile_id`` to the verified profile and clears ``context.profile``.
3. **Input:**
   * ``pending_result`` + ``pending_tool`` (back from a deterministic flow): the
     envelope is sent as the ``toolResult`` for ``pending_tool``, then cleared. If the caller
     spoke (this prompt, or the interrupted words), the result goes with ``next`` =
     ``await_caller`` and no ``callerUtterance``, and the words go with it as the turn's
     ``user_text``. A result with ``next`` = ``await_caller`` (the front door spoke the
     outcome) is recorded, never answered: the agent adds it to the history with no model call
     (:meth:`~tac_bridge.agent_client.AgentClient.record_results`), or together with the words in
     one ``turn`` when they came with it (the model answers only the words). The ``<silent/>``
     rule stays as a fallback. For
     ``handle_utterance`` with an ``interrupted_utterance``, and for
     ``retry_original_tool`` (step-up succeeded), the resume starts at ``setup``,
     without waiting for a prompt (:meth:`Bridge.on_setup`). With no pending tool
     (login intent switch) the utterance is sent as the caller's text.
   * otherwise the prompt text.
4. **Tool calls:** customer-data tools are run against the back end with the
   call's token, unsealed in memory from ``sealed_token`` per call, and the result goes back to the model
   (including ``AUTH_LEVEL_INSUFFICIENT``). Handoff tools (``authenticate``,
   ``get_balance``, ``escalate``) store ``pending_tool`` (and the escalation reason
   and summary) in call state, set ``context.pending_handoff_data`` (reason, tool,
   args, journal ref; checked with ``segments.assert_clean``) and end the stream.
   A front-door tool (``get_balance``, a deterministic-confirm tool) with valid
   arguments but below its ``min_level`` is not refused to the model: the bridge hands off
   for step-up at once (the ``authenticate`` hand-off), with the model's own call as
   ``pending_tool``; the front door runs it after a verified step-up and its envelope
   answers that call.
5. **Journal:** ``ai_turn``, ``tool_call`` and ``tool_result`` events.
6. **Errors**: an apology, one retry, then an escalation handoff.
7. **PII**: every ``ai_turn`` text and the escalation summary pass the
   injected :class:`owlbank.pii.PiiGuard` (Comprehend detection, our masking) before
   they are journaled or stored. If the check is unavailable the turn text becomes
   ``[WITHHELD]`` and the summary the catalog's ``escalation.summary_withheld``; the
   escalation still proceeds. Before that, the exact-value scrub
   (``segments.redact_text``) removes the call's sensitive values. The bridge learns them
   from call state's ``sealed_sensitive``: every call-state read calls
   ``sensitive.restore``, and ``session_end`` calls ``sensitive.forget`` (with tracing, after the
   trace redactor snapshots them for the call's unexported spans). A value that can't be
   unsealed makes the call unverifiable: both texts are then withheld (fail closed). None
   of this changes what the agent receives.
8. **Relay nonce**: :meth:`Bridge.accept_connection` checks the
   one-time nonce from the relay URL against ``ws_nonce_hash`` and consumes it,
   before TAC sees ``setup``.
9. **Idempotency**: ``make_payment`` and ``transfer_funds`` carry the
   model's ``tool_use_id`` as ``idempotency_key``; a failed invoke is retried once
   with the same key, so the bank moves money at most once.
10. **Deterministic confirmation**: a bank tool whose policy ``confirm``
    is ``deterministic`` is never run here. After the argument and level checks the
    bridge stores it as ``pending_tool`` (validated arguments only) and hands off with
    reason ``confirm``; ``handoffData`` carries no arguments. The front door reads it
    back, asks yes or no, and runs it only on yes; its envelope comes back as the
    ``toolResult`` (``completed``, ``CANCELLED_BY_CALLER`` or ``interrupted``).
11. **Untrusted content**: Memory recall and account nicknames in the system
    prompt, and transaction descriptions (and nicknames, memos) in tool results, are
    wrapped in ``<untrusted_data source="...">`` blocks with their content escaped
    (:mod:`owlbank.untrusted`).
12. **No narration of tool use**: the caller hears only text from a model
    message without tool use. The agent client holds each message's text until its
    ``messageStop`` and hands back text from a tool-using message as
    :class:`~tac_bridge.agent_client.TextSuppressed` (a length); the bridge logs it as
    ``ai_text_suppressed`` and neither speaks nor journals it. Structural only, no
    phrase matching. In stream mode (the default) the client yields each
    text chunk at once; when a tool-use block starts after text was already spoken it yields
    :class:`~tac_bridge.agent_client.TextBeforeTool`: the bridge logs
    ``ai_text_spoken_before_tool`` (WARNING, a length and the tool), counts it on the ``turn``
    line (``text_before_tool``), journals the spoken part as ``ai_turn``, runs the tool call as
    usual and, for a hand-off, holds ``end`` by the one hand-off hold rule (:func:`_handoff_hold_s`):
    until the latest of the silence watch's estimate of all speech sent (item 14), the hop's
    released text from its first chunk and a hand-off phrase; at most ``handoff_wait_ms`` from the
    hand-off, 0 with ``end_mode=immediate``; ``handoff_hold`` ``source=released_text`` (``mask``
    after a phrase) and ``basis`` naming the estimate that won.
13. **Latency masking**: scripted catalog phrases (``mask.<tool>``,
    ``mask.generic``) cover model and tool latency, decided per turn by
    :class:`~tac_bridge.masking.MaskPlanner`. The first hop of a turn whose input is the
    caller's words (never a resume without them) is raced against ``masking.after_ms`` (``asyncio.wait``, the
    agent stream is never
    cancelled) and speaks ``mask.generic`` if nothing came by then. A masked tool call's
    phrase is **scheduled**: rendered at once from the tool's pool
    (``mask.<tool>`` and ``mask.<tool>.<n>``, rotated per call and tool, ``{argument}``
    placeholders filled with catalog ``arg.*`` words), then spoken when its deadline
    (tool call + the tool's ``after_ms`` or ``masking.tool_after_ms``, not before the
    acknowledgement's estimated end) passes while the turn is still waiting (the tool or the next
    model call; :func:`_race_deadlines`), and dropped if answer text comes first. A hand-off's
    phrase is spoken at once. A Gateway tool start (``ToolStarted``) is
    masked the same way. The inline bank tools of one message run concurrently (results and
    journal in call order) and nothing waits for a phrase. A masked hand-off starts the
    read-only ``get_balance`` prefetch at once, stores its result as ``prefetched`` in call
    state for the front door, and holds ``end`` until the phrase's estimated end, or later speech
    queued ahead of it (item 12's rule; at most the tool's or the policy's ``handoff_wait_ms``;
    not at all with ``end_mode=immediate``).
    Phrases are journaled as ``prompt`` events and logged as ``mask_spoken``, phrases not
    spoken as ``mask_skipped`` (with the reason); they never reach the agent (not in its
    input, not in ``ai_turn``).
14. **Caller silence**: each WebSocket session has a
    :class:`~tac_bridge.silence.SilenceWatch`. It arms after the last outgoing speech of a turn
    (the turn's last message: agent text, or a scripted phrase that ended the turn's speech; a turn
    that said nothing re-arms from the last message the caller heard) and at session start from
    the welcome greeting, unsealed from call state's ``last_spoken`` (missing: no greeting repeat).
    The timeout counts from the end of the agent's speech: reported by ConversationRelay's
    speaker events (``agentSpeaking off``) when the session has them, else the
    estimated end of all speech sent (``0.3 s + chars / chars_per_s``; a repeat's from when it
    was actually sent), plus ``masking.silence_timeout_ms`` (read per turn through the policy cache; 0 = off), except
    the greeting arm of the call's first AI session, which waits
    ``masking.greeting_silence_timeout_ms`` instead (= no greeting repeat). "First" is
    :func:`had_ai_session` false on the call state read when the socket's nonce is consumed (before
    this session's own ``ai_sessions`` write can land; the same signs as history ``new``), or, when
    no nonce check recorded it, on the state loaded at ``setup``. Any
    ``prompt``, ``dtmf`` or ``interrupt`` frame (:meth:`Bridge.on_inbound`, from the app's
    ``PeekingWebSocket``) resets the count and re-counts the timeout from the input itself (so a key
    press or a barge-in with no turn after it still gets a repeat); a turn disarms, re-arms at its
    end, and nothing fires while one runs. On expiry the same text is sent again straight to ConversationRelay (never to the agent,
    not a turn), journaled ``prompt_repeat`` and logged without text, up to
    ``silence_max_repeats``; after the last repeat's timeout, ``silence_action`` once: ``escalate``
    (the escalation path, reason ``no_input``) or ``hangup`` (hand-off reason
    ``no_input_hangup``: the front door says goodbye and hangs up). An ``interrupt`` within
    ``REPEAT_GRACE_MS`` (500 ms) of a repeat being sent means the caller was already speaking: the
    repeat doesn't count and is logged ``prompt_repeat_collided``; so does a
    ``clientSpeaking on`` within the grace.
    Speaker events (:meth:`Bridge.on_speaker_event`, from the app's parser): while the
    caller is reported speaking (``clientSpeaking on``) nothing fires; ``clientSpeaking off`` with
    no prompt after it restarts the wait from that moment with the repeat count unchanged;
    nothing fires while the agent is reported speaking. ``silence_timing`` is logged once per
    session: ``speaker_events`` at the first recognised event, else ``estimate`` at session end.
    The first ``agentSpeaking on`` after a turn sent text logs ``agent_speech_started`` with
    ``ms_after_first_text``, once per turn: only within
    ``AGENT_SPEECH_START_MAX_MS`` of the turn's first chunk and before a silence repeat goes out.
15. **AI keypad shortcuts** (only with
    ``BridgeConfig.ai_keypad``): :meth:`Bridge.on_keypad` feeds each key to the call's
    :class:`~owlbank.keypad.KeypadBuffer` and wakes when the entry is due. Under the call's lock,
    with a live session and no pending hand-off: ``1``/``2`` become a caller turn with the
    catalog's ``keypad.yes``/``keypad.no`` in the call's language (the turn itself waits for the
    lock, so a key during a turn acts after it, and is checked again once it has the lock; it is
    TAC's stream task, so a barge-in or a new prompt cancels it like a spoken turn); ``0``
    escalates with ``caller_requested_human`` and no model call (hand-off ``trigger`` ``keypad``);
    ``9`` repeats the last message as the silence repeat does (the silence count unchanged);
    anything else, or a multi-key entry, speaks ``keypad.hint`` (not the last message 9 repeats).
    A ``prompt`` drops keys not yet acted on. Keys while a hand-off is pending or with no live
    session are dropped; a hand-off whose ``end`` a key or barge-in kept from going out gets it
    sent. Keypad work is cancelled when the session ends. One
    ``keypad_action`` line per action (``action`` and ``keys``, a count); a digit is never logged,
    journaled, traced or given to the agent.

The token never reaches the model: it is held only as ``sealed_token`` in
call state, unsealed in memory for one bank-tool call, never cached, logged or
stored. Every model input passes :func:`_assert_no_token` against the sealed
value, and every bank result is checked against the raw and sealed token.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import aclosing, contextmanager, nullcontext, suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

from owlbank import agent_prompt, callstate, obs, segments, sensitive, untrusted, ws_nonce
from owlbank import keypad as keypads
from owlbank import last_spoken as last_spoken_seal
from owlbank.callstate import (
    F_AGENT_OWNER,
    F_AI_SESSIONS,
    F_LAST_SPOKEN,
    F_PREFETCHED,
    WELCOME_OFFER_MADE,
    CallState,
    PendingTool,
)
from owlbank.catalog import CatalogKeyError, PromptCatalog
from owlbank.masking_config import placeholder_names, pool_keys, render, words_for
from owlbank.obs import emit, log_caught, timed_call
from owlbank.pii import SURFACE_AI_TURN, SURFACE_ESCALATION_SUMMARY, WITHHELD_TURN, PiiGuard
from owlbank.policy import MASK_GENERIC_KEYS, Policy
from owlbank.segments import RedactionError, SegmentLog
from owlbank.store import Store
from owlbank.task_attributes import clean_summary, normalize_reason
from owlbank.token_seal import Sealer, UnsealError
from owlbank.tools_contract import (
    AUTH_LEVEL_INSUFFICIENT,
    BANK_TOOLS,
    CALL_IN_PROGRESS,
    CALLER_DONE,
    END_CALL,
    FALLBACK_REASON,
    GATEWAY_LEVEL0_TOOLS,
    HANDOFF_TOOLS,
    IDEMPOTENT_OPS,
    SUMMARY_MAX_LEN,
    TOOL_DISABLED,
    VERIFICATION_CLOSED,
    auth_level_insufficient,
    card_values,
    gateway_tool_full_name,
    tool_error_code,
    tool_outcome,
    tool_schemas,
    validate_args,
)
from owlbank.web_activity import DEFAULT_MAX_AGE_HOURS, DEFAULT_TRAIT_GROUPS, bound_profile_traits
from owlbank.web_activity import GROUP as WEB_ACTIVITY_GROUP
from tac_bridge.agent_client import (
    ERROR_STOP_REASONS,
    SILENT_MARK,
    AgentClient,
    AgentError,
    HistoryMode,
    MessageEnd,
    ReasoningFilter,
    SilentFilter,
    TextBeforeTool,
    TextDelta,
    TextSuppressed,
    ToolCall,
    ToolResult,
    ToolStarted,
    TurnEnd,
    strip_silent,
)
from tac_bridge.clock import local_now_line, zone
from tac_bridge.masking import (
    SKIP_ANSWER_STARTED,
    SKIP_NO_RENDERABLE,
    SKIP_TURN_ENDED,
    EndMode,
    MaskPlanner,
    ScheduledPhrase,
    speech_end_chars,
)
from tac_bridge.silence import AGENT, Expiry, SilenceWatch

log = logging.getLogger(__name__)

CALLS = "calls"

# --------------------------------------------------------------------------- call state

#: ``CallState`` fields the bridge clears or sets (all typed fields of
#: :class:`owlbank.callstate.CallState`; no raw store attributes).
F_PENDING_TOOL = "pending_tool"
F_PENDING_RESULT = "pending_result"
F_INTERRUPTED = "interrupted_utterance"
F_ESCALATION = "escalation"
F_STEP_UP = "step_up_target"


@dataclass(frozen=True, slots=True)
class CallView:
    """What the bridge reads from :class:`CallState`. ``sealed_token`` is never printed."""

    call_sid: str
    language: str = "en-US"
    level: int = 0
    verified_customer_id: str | None = None
    pending_tool: PendingTool | None = None
    pending_result: dict[str, Any] | None = None
    interrupted_utterance: str | None = None
    step_up_target: str | None = None
    escalation: dict[str, Any] | None = None
    agent_owner: str | None = None
    version: int = 0
    sealed_token: str | None = field(default=None, repr=False)
    #: The session's welcome greeting, sealed (``owlbank.last_spoken``); never printed.
    last_spoken: Mapping[str, Any] | None = field(default=None, repr=False)
    #: :func:`had_ai_session` for this state (fallback, when no nonce check recorded it).
    earlier_ai_session: bool = False
    #: The call's sensitive values, KMS-sealed by the front door; never printed. Passed to
    #: ``sensitive.restore`` on every read (:meth:`CallStateRepo.load`).
    sealed_sensitive: tuple[str, ...] = field(default=(), repr=False)
    #: :data:`~owlbank.callstate.WELCOME_OFFER_MADE` when this session's welcome
    #: made the web-activity offer, else ``None`` (the prompt block).
    welcome_offer: str | None = None
    #: This call's login ended as ``verify_failed``, ``locked`` or
    #: ``declined``; no second verification (``authenticate`` refused, no step-up shortcut).
    login_closed: bool = False

    @property
    def verified(self) -> bool:
        return bool(self.verified_customer_id) and self.level >= 1

    @classmethod
    def from_state(cls, state: CallState) -> CallView:
        pr: Any = state.pending_result
        if isinstance(pr, (str, bytes)):  # tolerate a serialized envelope
            try:
                pr = json.loads(pr)
            except ValueError:
                log.warning("pending_result is not JSON; ignoring it", extra={
                    "operation": "load_call_state", "decision": "ignored_because_malformed_pending_result"})
                pr = None
        return cls(
            call_sid=state.call_sid,
            language=state.language or "en-US",
            level=state.level,
            verified_customer_id=state.verified_customer_id or None,
            pending_tool=state.pending_tool,
            pending_result=dict(pr) if isinstance(pr, Mapping) else None,
            interrupted_utterance=state.interrupted_utterance or None,
            step_up_target=state.step_up_target,
            escalation=dict(state.escalation) if state.escalation is not None else None,
            agent_owner=state.agent_owner or None,
            version=state.version,
            sealed_token=state.sealed_token or None,
            last_spoken=state.last_spoken,
            earlier_ai_session=had_ai_session(state),
            sealed_sensitive=tuple(state.sealed_sensitive),
            welcome_offer=state.welcome_offer or None,
            login_closed=bool(state.login_closed),
        )


class CallStateRepo:
    """Reads with ``callstate.load`` and writes with ``callstate.mutate`` (versioned, guarded).

    With a ``sealer``, every :meth:`load` also registers the call's sealed sensitive values in this
    process (``sensitive.restore``), so the exact-value scrub of journal text and log lines
    has them. ``restore`` never raises: an entry that doesn't unseal marks the call unverifiable."""

    def __init__(self, store: Store, *, now: Callable[[], float] = time.time,
                 sealer: Sealer | None = None) -> None:
        self._store = store
        self._now = now
        self._sealer = sealer

    def load(self, call_sid: str) -> CallView | None:
        state = callstate.load(self._store, call_sid, now=self._now())
        if state is None:
            return None
        if self._sealer is not None and state.sealed_sensitive:
            sensitive.restore(call_sid, state.sealed_sensitive, self._sealer)
        return CallView.from_state(state)

    def patch(self, call_sid: str, updates: Mapping[str, Any] | None = None,
              remove: Sequence[str] = ()) -> None:
        """Set ``updates`` and reset the ``remove`` fields to ``None``, as one versioned write."""
        changes = dict(updates or {})
        if "sealed_token" in changes or "sealed_token" in remove:
            raise ValueError("the bridge never writes the token")
        changes.update(dict.fromkeys(remove))
        callstate.mutate(self._store, call_sid, lambda _state: changes, now=self._now)

    def claim_agent_owner(self, call_sid: str, instance_id: str) -> HistoryMode:
        """Session ownership: the ``history`` decision for this session, from
        the call's ``agent_owner`` (and :func:`had_ai_session`), and ``agent_owner = instance_id``
        with ``ai_sessions + 1``, in one versioned read-modify-write (no write when this process
        already is the owner). Raises ``KeyError`` when the call has no state, and the store's
        errors."""
        decided: list[HistoryMode] = []

        def fn(state: CallState) -> Mapping[str, Any]:
            mode = session_history(state.agent_owner, instance_id, earlier_session=had_ai_session(state))
            decided.append(mode)  # the last attempt's decision (mutate retries on a conflict)
            if mode == "cached":
                raise _OwnerUnchanged
            return {F_AGENT_OWNER: instance_id, F_AI_SESSIONS: max(0, int(state.ai_sessions or 0)) + 1}

        with suppress(_OwnerUnchanged):
            callstate.mutate(self._store, call_sid, fn, now=self._now)
        return decided[-1]


class _OwnerUnchanged(Exception):  # noqa: N818 - control flow, never raised out of claim_agent_owner
    """This process already owns the call's agent: nothing to write."""


def had_ai_session(state: CallState) -> bool:
    """Call state shows the call has been in a ConversationRelay AI session before, whatever
    ``agent_owner`` says. A missing owner alone is not proof of a first session: a writer running
    older code once rewrote the item without it. Signs, each set only once
    the AI has run (or a hand-off from it is in flight): an earlier owner write (``ai_sessions``), a
    hand-off's ``pending_tool`` or its ``pending_result`` (a result envelope with no pending tool is
    the login intent switch, before any AI session; ``reload`` is still safe there, it only reads),
    the hand-off prefetch, and a step-up target (only an AI hand-off starts a step-up)."""
    return bool(state.ai_sessions) or state.pending_tool is not None or state.pending_result is not None \
        or state.prefetched is not None or bool(state.step_up_target)


def session_history(owner: str | None, instance_id: str, *, earlier_session: bool = False) -> HistoryMode:
    """No owner and no sign of an earlier AI session (``earlier_session``, from
    :func:`had_ai_session`) → ``new`` (a call's first AI session has no history, so nothing is
    read); this process → ``cached``; any other, or no owner after an earlier session → ``reload``
    (the call ran elsewhere, we restarted, or the owner was lost)."""
    if not owner:
        return "reload" if earlier_session else "new"
    return "cached" if owner == instance_id else "reload"


# --------------------------------------------------------------------------- injected clients


class BackendError(RuntimeError):
    """The back end (``bank_tools``) could not be reached or failed."""


@runtime_checkable
class BackendClient(Protocol):
    """``bank_tools``: ``{"op", "token", "call_sid", "args"[, "idempotency_key"]}`` -> result."""

    async def call(self, op: str, *, token: str | None, call_sid: str,
                   args: Mapping[str, Any], idempotency_key: str | None = None) -> dict[str, Any]: ...


class LambdaBackendClient:
    """:class:`BackendClient` by direct Lambda invoke (IAM; no network endpoint)."""

    def __init__(self, function_name: str, *, client: Any | None = None,
                 region_name: str | None = None) -> None:
        self._fn = function_name
        self._client = client
        self._region = region_name

    def _lambda(self) -> Any:
        if self._client is None:
            import boto3  # noqa: PLC0415

            self._client = boto3.client("lambda", region_name=self._region)
        return self._client

    def _invoke(self, payload: bytes, op: str = "?") -> dict[str, Any]:
        with timed_call("lambda", f"invoke:{op}"):
            resp = self._lambda().invoke(FunctionName=self._fn, InvocationType="RequestResponse",
                                         Payload=payload)
        if resp.get("FunctionError"):
            raise BackendError("bank_tools returned a function error")
        body = resp["Payload"].read()
        result = json.loads(body or b"{}")
        if not isinstance(result, dict):
            raise BackendError("bank_tools returned a non-object")
        return result

    async def call(self, op: str, *, token: str | None, call_sid: str,
                   args: Mapping[str, Any], idempotency_key: str | None = None) -> dict[str, Any]:
        event: dict[str, Any] = {"op": op, "token": token, "call_sid": call_sid, "args": dict(args)}
        if idempotency_key is not None:
            event["idempotency_key"] = idempotency_key
        payload = json.dumps(event).encode()
        return await asyncio.to_thread(self._invoke, payload, op)


@runtime_checkable
class MemoryClient(Protocol):
    """A thin wrapper over TAC's Memory client."""

    async def lookup_profile_ids(self, identifier: str) -> list[str]:
        """Profile IDs for the verified customer's identifier (canonical first). Raises on HTTP errors."""
        ...

    async def create_profile(self, identifier: str) -> str:
        """Create (or resolve) the profile that carries ``identifier``; returns its ID. Raises on errors."""
        ...

    async def retrieve_prompt(self, context: Any) -> str | None:
        """Recall for ``context.profile_id`` rendered as a prompt section, or ``None``."""
        ...


class ConversationCloser(Protocol):
    """Closes the Orchestrator conversation(s) a finished ConversationRelay segment left ACTIVE."""

    async def close_active(self, call_sid: str) -> int: ...


class TACConversationCloser:
    """:class:`ConversationCloser` over TAC's Orchestrator client.

    Every ``<Connect><ConversationRelay>`` makes Twilio create a NEW conversation for the same
    CallSid, and TAC's lookup on ``setup`` insists on exactly one ACTIVE conversation for it
    ("Expected exactly 1 conversation for call_sid ..., but found 3"). A call that leaves
    ConversationRelay for a scripted step and comes back (step-up, balance read-out, confirm) therefore
    broke on its second connection (found at the first real step-up call). Closing the previous
    conversation when its segment ends leaves exactly one ACTIVE conversation for the next connect.
    Twilio: close with PUT /v2/Conversations/{id} status CLOSED (a final state).
    """

    def __init__(self, tac: Any, *, timeout_s: float = 5.0) -> None:
        self._tac = tac
        self._timeout_s = timeout_s

    async def close_active(self, call_sid: str) -> int:
        client = self._tac.conversation_orchestrator_client
        if client is None:
            return 0
        conversations = await asyncio.wait_for(
            client.list_conversations(channel_id=call_sid, status=["ACTIVE"]), self._timeout_s)
        for conversation in conversations:
            await asyncio.wait_for(client.update_conversation(conversation.id, "CLOSED"), self._timeout_s)
        return len(conversations)


class SingleConversationGuard:
    """Wraps TAC's Orchestrator client so a call never has more than one ACTIVE conversation.

    TAC's ``setup`` lookup lists the call's ACTIVE conversations and accepts only exactly one; with
    two or more it polls until it gives up (about 30 s with the stopgap, then Twilio 64105). That
    happens when the two halves of ``reuseConversation`` disagree (a deploy in progress: a front
    door still creating a conversation per segment while the bridge no longer closes them), or if
    an earlier close failed. More than one never resolves by waiting, so this keeps the **newest**
    (the one ConversationRelay just attached to), closes the others, and returns it, so TAC
    proceeds at once. Only the call-scoped ACTIVE lookup is touched; everything else passes through.
    """

    def __init__(self, client: Any, *, timeout_s: float = 5.0) -> None:
        self._client = client
        self._timeout_s = timeout_s

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def list_conversations(self, *args: Any, **kwargs: Any) -> list[Any]:
        conversations = list(await self._client.list_conversations(*args, **kwargs))
        if len(conversations) < 2 or not kwargs.get("channel_id") or kwargs.get("status") != ["ACTIVE"]:
            return conversations
        ordered = sorted(conversations, key=lambda c: (getattr(c, "created_at", None) or "", c.id))
        keep, extras = ordered[-1], ordered[:-1]
        closed = 0
        for conversation in extras:
            try:
                await asyncio.wait_for(self._client.update_conversation(conversation.id, "CLOSED"),
                                       self._timeout_s)
                closed += 1
            except Exception as exc:  # noqa: BLE001 - still return one: TAC can proceed either way
                log_caught(log, "closing an extra conversation failed", exc,
                           operation="close_extra_conversation", decision="return_newest_anyway")
        emit(log, logging.WARNING, "extra_conversations_closed", found=len(conversations), closed=closed,
             decision="keep_newest")
        return [keep]


# Twilio's Memory API accepts only these identifier types (a live 400/20400 listed them: chat,
# email, phone, pushUserID, whatsapp). ``phone`` is the ANI and is shared by the Doe household, so
# verified customers are keyed on ``email`` (fictional ``@owlbank-demo.example`` in the fixtures).
MEMORY_ID_TYPE = "email"
MEMORY_TRAIT_GROUP = "Contact"  # Twilio's default trait group; its ``email`` trait is the identifier
MEMORY_IDENTIFIER_FIELD = "memory_email"  # the customers-table attribute that holds the value


class TACMemoryClient:
    """:class:`MemoryClient` over a ``tac.TAC`` instance in Orchestrator mode.

    TAC fetches every trait group of the profile (no
    ``TWILIO_MEMORY_PROFILE_TRAIT_GROUPS``), and :meth:`retrieve_prompt` bounds them with
    :func:`owlbank.web_activity.bound_profile_traits` (the ``allow`` groups only, values cleaned,
    stale web activity dropped) before TAC's prompt builder renders them."""

    def __init__(self, tac: Any, *, allow: Sequence[str], max_age_hours: int,
                 now: Callable[[], float]) -> None:
        if tac.conversation_memory_client is None:
            raise ValueError("TAC must run in Orchestrator mode (no Memory client)")
        self._tac = tac
        self._allow = list(allow)
        self._max_age_hours = max_age_hours
        self._now = now

    async def lookup_profile_ids(self, identifier: str) -> list[str]:
        with timed_call("tac_memory", "lookup_profile"):
            res = await self._tac.conversation_memory_client.lookup_profile(
                id_type=MEMORY_ID_TYPE, value=identifier)
        return list(res.profiles or [])

    async def create_profile(self, identifier: str) -> str:
        # Identity resolution upserts on the identifier trait, so a repeat returns the same profile.
        # Only the verified identifier is sent: never the ANI (a phone trait would merge the
        # household profile in).
        with timed_call("tac_memory", "create_profile"):
            return str(await self._tac.conversation_memory_client.create_profile(
                {MEMORY_TRAIT_GROUP: {MEMORY_ID_TYPE: identifier}}))

    async def retrieve_prompt(self, context: Any) -> str | None:
        from tac.adapters import AdapterOptions, MemoryPromptBuilder  # noqa: PLC0415

        if not context.profile_id:  # never let TAC fall back to the ANI lookup
            return None
        with timed_call("tac_memory", "retrieve_memory"):
            mem = await self._tac.retrieve_memory(context)
        self._bound_traits(context)
        return MemoryPromptBuilder.build(mem, context, AdapterOptions(profile_traits=self._allow))

    def _bound_traits(self, context: Any) -> None:
        """Replace ``context.profile.traits`` with the bounded groups and log
        ``traits_loaded``: the kept groups' names and the number of traits, never a value.
        If the filter raises (a bug), only ``Contact`` is kept, as before this feature (fail closed)."""
        profile = context.profile
        traits = getattr(profile, "traits", None)
        if profile is None or not traits:
            return
        try:
            kept = bound_profile_traits(traits, allow=self._allow, max_age_hours=self._max_age_hours,
                                        now=datetime.fromtimestamp(self._now(), UTC))
        except Exception as exc:
            log_caught(log, "trait filter failed; keeping Contact only", exc, operation="trait_filter",
                       decision="contact_only")
            contact = traits.get(MEMORY_TRAIT_GROUP) if isinstance(traits, Mapping) else None
            kept = {MEMORY_TRAIT_GROUP: contact} if contact and MEMORY_TRAIT_GROUP in self._allow else {}
        profile.traits = kept
        emit(log, logging.INFO, "traits_loaded", groups=[g for g in self._allow if g in kept],
             count=sum(len(v) for v in kept.values() if isinstance(v, Mapping)))


class VoiceChannelLike(Protocol):
    """The two ``tac.channels.voice.VoiceChannel`` methods the bridge uses."""

    async def send_response(self, conversation_id: str, response: Any, role: str | None = None) -> None: ...

    def get_conversation_session_by_call_sid(self, call_sid: str) -> Any: ...


def tac_sessions_for_call(voice: Any, call_sid: str) -> list[Any]:
    """Every TAC session for ``call_sid``, oldest first (task S30A).

    TAC 2.5.0 has no public way to list a channel's sessions: ``get_conversation_session_by_call_sid``
    returns the first (oldest) match of ``BaseChannel._conversations`` (``tac/channels/base.py`` sets it,
    ``tac/channels/voice/channel.py`` iterates it), a dict keyed by conversation id in insertion order. A
    session that TAC hasn't yet ended (its ``CLOSED`` webhook lost the race against the next call's
    sessions) stays in that dict, so the oldest match can be stale. This is the one place that reads the
    private attribute; ``tests/unit/test_bridge_session_lookup.py`` fails if TAC changes its shape. A
    channel without it (a test fake, or a TAC whose shape changed) falls back to the public lookup.
    """
    conversations = getattr(voice, "_conversations", None)
    if isinstance(conversations, Mapping):
        return [s for s in list(conversations.values()) if getattr(s, "call_sid", None) == call_sid]
    one = voice.get_conversation_session_by_call_sid(call_sid)
    return [] if one is None else [one]


def tac_stream_state(voice: Any, conversation_id: str) -> Any | None:
    """TAC's per-conversation ``SessionState`` (its ``stream_task``, the turn a ``prompt`` or an
    ``interrupt`` cancels), or ``None`` when there is none (no session manager, or a test fake).

    TAC 2.5.0 has no public accessor: ``VoiceChannel._provider`` is the ConversationRelay provider and
    its ``session_manager`` (``ThreadSafeSessionManager`` by default) holds one state per conversation,
    created at the socket's ``setup``. A keypad turn is made that conversation's stream task, so
    a barge-in or a new prompt cancels it as it cancels a spoken turn.
    ``tests/unit/test_bridge_session_lookup.py`` fails if TAC changes this shape."""
    manager = getattr(getattr(voice, "_provider", None), "session_manager", None)
    has_session = getattr(manager, "has_session", None)
    if manager is None or not callable(has_session) or not has_session(conversation_id):
        return None
    return manager.get_or_create_session(conversation_id)


def pick_session(sessions: Sequence[Any], has_socket: Callable[[Any], bool]) -> tuple[Any | None, bool]:
    """The newest session with a registered WebSocket, else the newest; and whether it has one.

    ``sessions`` is oldest first. Returns ``(None, False)`` when there is none.
    """
    for session in reversed(sessions):
        if has_socket(session):
            return session, True
    return (sessions[-1] if sessions else None), False


HandoffFactory = Callable[[str], Any]


def _default_handoff_factory(handoff_json: str) -> Any:
    from tac.models.handoff import PendingHandoffData  # noqa: PLC0415

    return PendingHandoffData(handoffData=handoff_json)  # the alias; the SDK model has no mypy plugin


# --------------------------------------------------------------------------- helpers


class _NonceRejected(Exception):
    """Internal: the relay nonce did not match (or the call state is gone)."""


class TokenLeakError(RuntimeError):
    """A model input contained the call token. Never sent."""


def _assert_no_token(token: str | None, payload: Any) -> None:
    if not token:
        return
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str)
    if token in text:
        raise TokenLeakError("model input contained the call token")


@dataclass(slots=True)
class _Hop:
    user_text: str | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    #: The hop's input carries the caller's words, so the timed acknowledgement
    #: may be armed. Never for a resume right after a scripted line with no caller words.
    timer: bool = False
    #: The hop's results are the front door's envelope with ``next`` = ``await_caller``
    #: (the caller already heard the outcome) and the caller said nothing: recorded in the agent's
    #: history with no model call (:meth:`AgentClient.record_results`), never answered.
    record_only: bool = False
    #: Call observability (``evalshape.HopInfo``): the step the hop resumes after (the pending
    #: tool's name) and the front door's ``callerUtterance`` (or the interrupted words) it carried.
    step: str | None = None
    utterance: str | None = None
    #: The session's welcome, recorded in the agent's history (the synthetic
    #: ``agent.call_started`` / welcome pair) before this hop's agent call; ``None`` once recorded,
    #: or when the welcome rides in the hop's tool result instead.
    welcome: str | None = None


#: Hand-off tools whose read-only work the bridge starts while the phrase plays.
#: State-changing tools are never here: they run only after the scripted confirmation.
PREFETCH_TOOLS: frozenset[str] = frozenset({"get_balance"})

#: What :func:`_race_timer` yields when the timer fires before the next agent event.
_TIMER = object()


@dataclass(frozen=True, slots=True)
class _Mask:
    """A masking phrase yielded this turn for a tool call."""

    key: str
    tool: str
    text: str  # as yielded (with the trailing space)
    at: float  # time.monotonic() once it was on the wire


#: The estimates a hand-off hold is the latest of (``handoff_hold`` ``basis``), in tie order.
HOLD_BASES = ("phrase", "hop_text", "watch")


def _hold_estimates(planner: MaskPlanner, now: float, *, speech_end: float,
                    hop_text: tuple[float, int] | None, mask: _Mask | None) -> dict[str, float]:
    """The estimated remaining speech (seconds from ``now``, possibly negative) a hand-off hold
    covers, by :data:`HOLD_BASES` name:

    * ``watch``: ``speech_end`` (the session's :attr:`SilenceWatch.playing_until`, all speech
      queued so far) less ``now``; absent when ``-inf`` (no speech session, or a new one);
    * ``hop_text``: the current hop's released agent text, ``(first chunk sent at, chars)``,
      ``0.3 s + chars / chars_per_s`` from its first chunk (a speaker report can't cut it short);
    * ``phrase``: the hand-off phrase ``mask`` from when it was sent."""
    out: dict[str, float] = {}
    if mask is not None and mask.text.strip():
        out["phrase"] = speech_end_chars(len(mask.text.strip()), mask.at, planner.chars_per_s) - now
    if hop_text is not None and hop_text[1] > 0:
        out["hop_text"] = speech_end_chars(hop_text[1], hop_text[0], planner.chars_per_s) - now
    if speech_end != float("-inf"):
        out["watch"] = speech_end - now
    return out


def _handoff_hold_s(planner: MaskPlanner, tool: str | None,
                    estimates: Mapping[str, float]) -> tuple[float, str]:
    """Seconds to hold a hand-off's ``end``, and which estimate set it (``basis``, a
    :data:`HOLD_BASES` name, ``none`` without any): the one rule for every hand-off hold, after a
    hand-off phrase or released agent text. The latest of the
    ``estimates`` (:func:`_hold_estimates`), never negative, never more than the tool's
    ``handoff_wait_ms`` from the hand-off (its own, else the policy's), 0 with
    ``end_mode=immediate``."""
    known = [b for b in HOLD_BASES if b in estimates]
    basis = max(known, key=lambda b: estimates[b]) if known else "none"  # ties: HOLD_BASES order
    if planner.end_mode == "immediate":
        return 0.0, basis
    cap = planner.handoff_wait_ms_for(tool) / 1000.0  # The tool's own, else policy
    return min(cap, max([0.0, *(estimates[b] for b in known)])), basis


class _Deadlines:
    """Named deadlines (``time.monotonic()`` scale) raced with a stream by
    :func:`_race_deadlines`. The turn arms ``ack`` (the timed acknowledgement) and ``tool``
    (a scheduled tool phrase); the consumer may arm or cancel between events."""

    __slots__ = ("_at",)

    def __init__(self) -> None:
        self._at: dict[str, float] = {}

    def arm(self, name: str, at: float) -> None:
        self._at[name] = at

    def cancel(self, name: str) -> None:
        self._at.pop(name, None)

    def armed(self, name: str) -> bool:
        return name in self._at

    def earliest(self) -> tuple[str, float] | None:
        if not self._at:
            return None
        name = min(self._at, key=self._at.__getitem__)
        return name, self._at[name]

    def __bool__(self) -> bool:
        return bool(self._at)


@dataclass(frozen=True, slots=True)
class _Fired:
    """What :func:`_race_deadlines` yields when deadline ``name`` passes before the next event."""

    name: str


async def _race_deadlines(events: AsyncIterator[Any], deadlines: _Deadlines) -> AsyncGenerator[Any, None]:
    """``events``, interleaved with one :class:`_Fired` per deadline that passes first.

    One pending ``anext`` runs as a task raced with the earliest armed deadline in a single
    ``asyncio.wait`` (never ``wait_for``), so the stream is never cancelled or delayed by a
    deadline; after a deadline fires (it is disarmed first) the same task is raced with the next
    one, then awaited. ``deadlines`` may change between events (the consumer runs while this
    generator is suspended at a yield). If this generator is closed (barge-in) the pending read is
    cancelled and settled before the caller closes ``events``.
    """
    while True:
        task: asyncio.Future[Any] = asyncio.ensure_future(anext(events))
        try:
            while not task.done():
                nxt = deadlines.earliest()
                if nxt is None:
                    break
                name, at = nxt
                remaining = at - time.monotonic()
                if remaining > 0:
                    done, _ = await asyncio.wait({task}, timeout=remaining)
                    if done:
                        break
                    if deadlines.earliest() != nxt or at > time.monotonic():
                        continue  # re-armed meanwhile, or woke a hair early: race again
                deadlines.cancel(name)
                yield _Fired(name)
            try:
                ev = await task
            except StopAsyncIteration:
                return
        finally:
            if not task.done():
                task.cancel()
                await asyncio.wait({task})
        yield ev


async def _race_timer(events: AsyncIterator[Any], timeout_s: float | None) -> AsyncGenerator[Any, None]:
    """``events``, plus one :data:`_TIMER` if no text or tool call arrived within ``timeout_s``
    (the timer race, now a thin wrapper of :func:`_race_deadlines`)."""
    deadlines = _Deadlines()
    if timeout_s is not None:
        deadlines.arm("timer", time.monotonic() + timeout_s)
    async with aclosing(_race_deadlines(events, deadlines)) as stream:
        async for ev in stream:
            if isinstance(ev, _Fired):
                yield _TIMER
                continue
            if isinstance(ev, (TextDelta, ToolCall)):
                deadlines.cancel("timer")  # speech or a tool call came first: no acknowledgement
            yield ev


async def _one(aw: Awaitable[Any]) -> AsyncGenerator[Any, None]:
    """A one-event stream: the result of ``aw`` (to race a coroutine with deadlines)."""
    yield await aw


@dataclass(slots=True)
class _PendingPhrase:
    """A scheduled tool phrase, rendered at tool-call time."""

    sched: ScheduledPhrase
    key: str  # the pool key chosen
    text: str  # rendered, with the trailing space


@dataclass(slots=True)
class _TurnMask:
    """One turn's masking state: the planner, its deadlines and the rendered pending phrase."""

    planner: MaskPlanner
    deadlines: _Deadlines = field(default_factory=_Deadlines)
    pending: _PendingPhrase | None = None


@dataclass(frozen=True, slots=True)
class _TextFilters:
    """One turn's filters for the agent's text, in order: reasoning and markup, then the
    ``<silent/>`` marker. Each hop's model text goes through both; both flush per hop."""

    reasoning: ReasoningFilter
    silent: SilentFilter


def _reasoning_removed(reasoning: ReasoningFilter, stats: _TurnStats) -> None:
    """The turn's removed reasoning and markup on its stats (the turn span), and one
    ``reasoning_text_removed`` line if anything was removed. Counts only, never the text. WARNING
    when, on the turn's normal completion, its last model message removed reasoning (or ended in
    an over-long tag dropped to its end) and the caller hears nothing of that message
    unintentionally: no words of the agent's own and no ``<silent/>`` marker (an unclosed or misspelled close
    tag swallows the whole answer). Decided per model
    message: an earlier message's tool call, a Gateway tool's or an unknown tool's in the same hop
    included, hides nothing. A message that calls a tool (a hand-off included) is never the last:
    its end (``MessageEnd``, or the inline ``ToolCall``) starts a new message, so a turn that ends
    on a tool call removed nothing in its last message and stays INFO. Else INFO too (a barge-in
    or an error escalation ends the turn early, not silently)."""
    reasoning.flush()  # a barge-in closed the turn mid-block: what was removed so far counts, never spoken
    stats.reasoning_blocks, stats.reasoning_chars, stats.markup_tags = reasoning.blocks, reasoning.chars, reasoning.tags
    stats.reasoning_text = reasoning.removed
    if reasoning.blocks or reasoning.tags:
        # the last model message removed reasoning, or an over-long tag ran to its end
        last_msg = reasoning.blocks > stats.msg_blocks or reasoning.skips > stats.msg_skips
        unintended = stats.completed and not (stats.spoke or stats.silenced)
        level = logging.WARNING if last_msg and unintended else logging.INFO
        emit(log, level, "reasoning_text_removed", blocks=reasoning.blocks, chars=reasoning.chars,
             tags=reasoning.tags, turn=stats.number)


@dataclass(frozen=True, slots=True)
class BridgeConfig:
    max_hops: int = 6
    retries: int = 1
    session_wait_s: float = 12.0
    session_poll_s: float = 0.05
    memory_timeout_s: float = 4.0
    backend_timeout_s: float = 8.0
    #: Keep the call's Orchestrator conversation open across segments (the
    #: front door reconnects with ``conversationId``), so segment ends don't close it. Must match
    #: the front door's ``OWLBANK_REUSE_CONVERSATION``; Orchestrator's inactive/closed timeouts
    #: close it after the call.
    reuse_conversation: bool = False
    #: The bank's IANA time zone for the date and time line in the system prompt.
    timezone: str = "America/New_York"
    #: Memory trait groups whose traits may reach the model (``OWLBANK_MEMORY_TRAIT_GROUPS``).
    memory_trait_groups: tuple[str, ...] = DEFAULT_TRAIT_GROUPS
    #: Oldest web activity offered, in hours (``OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS``).
    web_activity_max_age_hours: int = DEFAULT_MAX_AGE_HOURS
    #: AI keypad shortcuts (``OWLBANK_AI_KEYPAD``): keys during the AI session act. The code
    #: default is off (tests and a bare config keep today's behaviour); the env default is on.
    ai_keypad: bool = False
    #: Model failover: the agent's model list (``OWLBANK_MODELS``' IDs, in order). The bridge
    #: renders one system prompt per distinct model key of it. Empty (tests, a bare config): one
    #: prompt, for :data:`~owlbank.agent_prompt.DEFAULT_MODEL_KEY`.
    model_ids: tuple[str, ...] = ()


META_KEY = "owlbank"

#: ``(model key, language)`` pairs already logged as ``model_prompt_missing`` in this process.
_PROMPT_MISSING_LOGGED: set[tuple[str, str]] = set()

# The placeholder rule and the prompt's rendering live in owlbank.agent_prompt.


def _has_web_activity(context: Any) -> bool:
    """Whether the recall kept a ``WebActivity`` group: ``TACMemoryClient`` leaves only the
    bounded groups on ``context.profile.traits``."""
    traits = getattr(getattr(context, "profile", None), "traits", None)
    return isinstance(traits, Mapping) and bool(traits.get(WEB_ACTIVITY_GROUP))


def _log_rendered(statics: Mapping[str, str], per_call: str, lang: str, *, verified: bool,
                  web_block: str | None) -> None:
    """``system_prompt_rendered`` once per rendered prompt (each model key of a render), so a
    cache miss can be told from a changed static part. Sizes in characters and the static part's
    hash only (the first 16 hex characters of its SHA-256): no prompt text, no caller value.
    ``web_block`` is the block's catalog key or ``None``."""
    for model_key, static in statics.items():
        emit(log, logging.INFO, "system_prompt_rendered", model_key=model_key, language=lang,
             static_sha=hashlib.sha256(static.encode("utf-8")).hexdigest()[:16],
             static_chars=len(static), per_call_chars=len(per_call), verified=verified,
             web_block=web_block)


def _trait_count(context: Any) -> int:
    """How many bounded traits the recall kept (``owlbank.trait_count`` on ``bridge.turn``)."""
    traits = getattr(getattr(context, "profile", None), "traits", None)
    if not isinstance(traits, Mapping):
        return 0
    return sum(len(v) for v in traits.values() if isinstance(v, Mapping))


@dataclass(slots=True)
class _Session:
    """Counters for one WebSocket session (the ``ws_session_end`` line)."""

    started: float
    prompts: int = 0
    tool_calls: int = 0
    turns: int = 0
    #: The session's token totals (``ws_session_end``;'s tokens-per-call alarm).
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(slots=True)
class _TurnStats:
    """What one turn did (the ``turn`` line). No message text, ever."""

    number: int
    started: float
    prompt: bool
    stop_reason: str | None = None
    hops: int = 0
    tools: list[str] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    outcome: str = "cancelled"
    #: ``time.monotonic()`` of the turn's first chunk to ConversationRelay of any kind (``silence_ms``:
    #: what the caller hears first, a masking phrase included), and of its first chunk of the agent's
    #: own answer (``wait_ms``: no masking phrase, no apology). 7.3.
    first_out: float | None = None
    answered_at: float | None = None
    #: Set by the hop loop just before it yields agent answer text (``answered_at``'s marker).
    answering: bool = False
    #: Call observability: the turn's ``bridge.turn`` span and the tracing that made it (else ``None``).
    span: Any = None
    tracing: Any = None
    #: Tool calls that started after text of their message was spoken
    #: (``ai_text_spoken_before_tool``); the ``turn`` line's ``text_before_tool``.
    text_before_tool: int = 0
    #: Reasoning blocks (and their characters) and other tags the
    #: :class:`ReasoningFilter` removed this turn; the blocks' text only for the turn span.
    reasoning_blocks: int = 0
    reasoning_chars: int = 0
    markup_tags: int = 0
    reasoning_text: str = field(default="", repr=False)
    #: The filter's ``blocks`` and ``skips`` when the current (in the end, last) model message
    #: started, so the WARNING reads what that message alone removed.
    msg_blocks: int = 0
    msg_skips: int = 0
    #: The turn's current (in the end, last) model message sent the caller some of the agent's own
    #: words (not whitespace only; no masking phrase or apology), or the agent wrote ``<silent/>``:
    #: a last message that removed reasoning with neither logs at WARNING. A tool call needs
    #: no flag: it ends its message (:meth:`new_message`), so that message is never the last.
    spoke: bool = False
    silenced: bool = False
    #: The hop loop ended normally (not a barge-in, which closes it, nor an error escalation).
    completed: bool = False
    #: ``agent_speech_started`` was logged for this turn, or can no longer be
    #: (once per turn; :data:`AGENT_SPEECH_START_MAX_MS`).
    speech_logged: bool = False

    def new_message(self, reasoning: ReasoningFilter) -> None:
        """A model message starts (a hop's start, or after the client's ``MessageEnd`` or an
        inline ``ToolCall`` ended the last one): the WARNING's per-message flags start again."""
        self.msg_blocks, self.msg_skips = reasoning.blocks, reasoning.skips
        self.spoke = self.silenced = False


#: ``agent_speech_started`` is logged only within this long of the turn's
#: first chunk out. Speech normally starts about 0.15 s (p90 0.19 s) after it; an
#: ``agentSpeaking on`` later than this is not this turn's first text being spoken (a later burst,
#: or the turn's text was never spoken), so it would only log a misleading figure.
AGENT_SPEECH_START_MAX_MS = 3000.0

#: The silence action's hand-off reasons (the front door's connect-action knows both).
NO_INPUT = "no_input"
NO_INPUT_HANGUP = "no_input_hangup"
#: The escalation summary for a silence escalation (fixed, like the bridge's own error summary).
NO_INPUT_SUMMARY = "The caller did not respond after the last message was repeated."
#: AI keypad shortcuts: the hand-off ``trigger`` and escalation ``source`` of a 0, its
#: reason, the summary used when the catalog has no ``keypad.agent_summary``, the turn words used
#: when it has no ``keypad.yes``/``keypad.no``, and the repeat triggers (``prompt_repeat``).
KEYPAD_TRIGGER = "keypad"
KEYPAD_REASON = "caller_requested_human"
KEYPAD_SUMMARY_KEY = "keypad.agent_summary"
KEYPAD_SUMMARY_FALLBACK = "The caller pressed 0 to speak with an agent."
KEYPAD_HINT_KEY = "keypad.hint"
#: TAC's session for the socket is still being set up; keys are held, not dropped.
KEYPAD_NOT_READY = "session_not_ready"
KEYPAD_WORDS: Mapping[str, tuple[str, str]] = MappingProxyType({
    keypads.YES: ("keypad.yes", "Yes"), keypads.NO: ("keypad.no", "No")})
REPEAT_SILENCE = "silence"
REPEAT_KEYPAD = "keypad"
#: What ``relay_interrupt`` reports as playing before the session has sent anything.
PLAYING_GREETING = "greeting"
PLAYING_AGENT = "agent"
#: ``silence_timing`` modes. ``speaker_events`` is logged at the session's first
#: recognised speaker event; ``estimate`` at the end of a session that had none.
TIMING_SPEAKER_EVENTS = "speaker_events"
TIMING_ESTIMATE = "estimate"


@dataclass(slots=True)
class _CallSpeech:
    """One WebSocket session's outgoing speech and its :class:`SilenceWatch`."""

    watch: SilenceWatch
    handle: asyncio.TimerHandle | None = None
    #: What was sent last: a catalog key, ``agent`` or ``greeting`` (``relay_interrupt``).
    playing: str = PLAYING_GREETING
    #: The last message (what a repeat speaks), its label and when it started going out.
    text: str | None = None
    label: str = PLAYING_GREETING
    at: float = 0.0
    #: The last message is agent text this turn is still streaming (chunks join it).
    open: bool = False
    turn_spoke: bool = False
    language: str = "en-US"
    #: The session started with a sealed greeting in call state (cleared once a turn speaks).
    clear_last_spoken: bool = False
    #: The session's current (or last) turn, for ``agent_speech_started``.
    turn: _TurnStats | None = None
    #: This is the call's first AI session, as call state showed it when the socket's
    #: nonce was consumed (``None``: not known, e.g. no nonce check ran; ``_arm_greeting`` then
    #: uses the state it loads at ``setup``).
    first_session: bool | None = None
    #: The session's welcome as ``_arm_greeting`` unsealed it (``None``: none,
    #: or not unsealable), once ``welcome_known``; taken (``welcome_taken``) only once it is in the
    #: agent's history: the synthetic pair recorded, or placed in a pending result's envelope. A turn
    #: that fails before that leaves it for the session's next turn (exactly once).
    welcome: str | None = field(default=None, repr=False)
    welcome_known: bool = False
    welcome_taken: bool = False
    #: The session is ending on a hand-off the bridge set (TAC clears the context's
    #: ``pending_handoff_data`` when it sends the ``end``; this stays set until the session ends).
    handoff: bool = False
    #: A hand-off's ``end`` the bridge sent itself after a key or a barge-in cut the turn that
    #: would have sent it (:meth:`Bridge._keypad_end_pending`): at most once per session, and the
    #: task doing it while it runs.
    end_resent: bool = False
    end_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    #: The socket has started sending an ``end`` frame (``PeekingWebSocket`` saw it go out:
    #: :meth:`Bridge.on_relay_end`). A turn cancelled while suspended in that send leaves TAC's
    #: ``pending_handoff_data`` set, but the ``end`` went out: it is never resent.
    end_out: bool = False
    #: The session's ``_send`` calls still running (each one's done future and its task), whoever
    #: runs them: TAC's stream task, a resume at ``setup``, keypad and silence work. Once its
    #: generator ends TAC still sends ``last: true`` and, with a hand-off pending, the ``end``,
    #: after the turn has released the call's lock; the ``end`` resend waits for these.
    sends: dict[asyncio.Future[None], asyncio.Task[Any] | None] = field(default_factory=dict, repr=False)
    #: The session's keypad ``yes``/``no`` turn while it runs (cancelled by a barge-in or a prompt).
    keypad_turn: asyncio.Task[Any] | None = field(default=None, repr=False)
    #: A ``prompt`` has reached TAC on this socket. Before that TAC 2.5.0 has no conversation for the
    #: socket and ignores an ``interrupt`` (no cancel, no acknowledgement, no ``on_interrupt``).
    prompted: bool = False


@dataclass(slots=True)
class _KeypadTurn:
    """A keypad ``yes``/``no`` turn as :meth:`Bridge.run_turn` runs it: refused (no model call)
    when, with the call's lock taken, a hand-off is pending or the session is gone."""

    refused: bool = False


# --------------------------------------------------------------------------- call observability

#: Span names.
SPAN_SESSION = "relay.session"
SPAN_TURN = "bridge.turn"
#: A failed span call is logged at most this often.
_TRACE_FAILURE_LOG_S = 60.0
#: How long a session end's flush of the session's spans may take (it never delays the forget).
TRACE_FLUSH_TIMEOUT_S = 3.5


def _ms_since(at: float | None, start: float) -> float | None:
    return round((at - start) * 1000, 1) if at is not None else None


class _Tracing:
    """Call observability's spans in the bridge ("Bridge integration").

    It exists only when telemetry is installed (:meth:`Bridge.enable_tracing`): with the switch off
    the bridge holds ``None``, every call site is a ``None`` check, and none of the imports below
    run. Each ``relay.session`` and ``bridge.turn`` span is a **new root** (an empty context with the
    call in baggage), so each turn is its own trace in the call's session; the
    ``BaggageSpanProcessor`` copies ``session.id`` and ``call_sid`` onto it and onto everything
    under it. Attribute and event keys are ``owlbank.*`` identifiers, counts and timings only,
    never text (the allow-list's ``owlbank.*`` entry).

    No method raises: a failure is counted in :attr:`failures`, logged at DEBUG at most
    once a minute, and the span, event or attribute is skipped.
    """

    def __init__(self, tracer: Any) -> None:
        from opentelemetry import baggage, context, trace  # noqa: PLC0415 - only when tracing is on
        from opentelemetry.context import Context  # noqa: PLC0415

        from owlbank.telemetry import evalshape  # noqa: PLC0415

        self._tracer = tracer
        self._baggage = baggage
        self._context = context
        self._trace = trace
        self._empty = Context
        self._hop_var = evalshape.HOP
        self._hop_info = evalshape.HopInfo
        self.failures = 0
        self._logged_at: float | None = None
        self._flush_logged_at: float | None = None
        self._retain_logged_at: float | None = None

    def _failed(self, what: str, exc: BaseException) -> None:
        with suppress(Exception):
            self.failures += 1
            now = time.monotonic()
            if self._logged_at is None or now - self._logged_at >= _TRACE_FAILURE_LOG_S:
                self._logged_at = now
                emit(log, logging.DEBUG, "telemetry_span_failed", operation=what,
                     error_class=type(exc).__name__, failures=self.failures, decision="continue_untraced")

    def flush_failed(self, exc: BaseException | None) -> None:
        """A session end's flush raised, timed out or returned ``False``: counted in
        :attr:`failures` and logged as ``trace_flush_incomplete`` (WARNING) at most once a minute.
        The spans still export later, with their call's values (the redactor's snapshot)."""
        with suppress(Exception):
            self.failures += 1
            now = time.monotonic()
            if self._flush_logged_at is None or now - self._flush_logged_at >= _TRACE_FAILURE_LOG_S:
                self._flush_logged_at = now
                emit(log, logging.WARNING, "trace_flush_incomplete",
                     error_class=type(exc).__name__ if exc is not None else "FlushIncomplete",
                     failures=self.failures, decision="export_later")

    def retain_failed(self, exc: BaseException) -> None:
        """The redactor's ``retain`` raised: the call's spans exported after the forget lose its
        values' scrub. Counted in :attr:`failures` and logged as ``trace_retain_failed`` (WARNING,
        no values) at most once a minute."""
        with suppress(Exception):
            self.failures += 1
            now = time.monotonic()
            if self._retain_logged_at is None or now - self._retain_logged_at >= _TRACE_FAILURE_LOG_S:
                self._retain_logged_at = now
                log_caught(log, "trace_retain_failed", exc, decision="continue",
                           failures=self.failures)

    @staticmethod
    def _clean(attributes: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in attributes.items():
            if value is None:
                continue
            if isinstance(value, list | tuple):
                value = tuple(str(v) for v in value)
            out[key] = value
        return out

    def start(self, name: str, call_sid: str, attributes: Mapping[str, Any] | None = None) -> Any:
        """A new root span for ``call_sid`` (``None`` on failure)."""
        try:
            ctx = self._baggage.set_baggage("session.id", call_sid, context=self._empty())
            ctx = self._baggage.set_baggage("call_sid", call_sid, context=ctx)
            return self._tracer.start_span(name, context=ctx, attributes=self._clean(attributes or {}))
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a call
            self._failed("start_span", exc)
            return None

    def attach(self, span: Any, call_sid: str) -> Any:
        """Make ``span`` current, with the call in baggage; the token for :meth:`detach`."""
        if span is None:
            return None
        try:
            ctx = self._baggage.set_baggage("session.id", call_sid, context=self._empty())
            ctx = self._baggage.set_baggage("call_sid", call_sid, context=ctx)
            return self._context.attach(self._trace.set_span_in_context(span, ctx))
        except Exception as exc:  # noqa: BLE001
            self._failed("attach", exc)
            return None

    def detach(self, token: Any) -> None:
        if token is None:
            return
        try:
            self._context.detach(token)
        except Exception as exc:  # noqa: BLE001
            self._failed("detach", exc)

    @contextmanager
    def child(self, name: str, call_sid: str, attributes: Mapping[str, Any] | None = None) -> Iterator[Any]:
        """A span under the current one (a new root when there is none), current while the block
        runs, with the call in baggage either way, so the sampler keeps it with its call and its
        children (botocore's) nest under it. Ends with ``owlbank.outcome`` and, if the
        block raised, ``owlbank.error_class``; the exception propagates. Yields the span (``None``
        on failure)."""
        span = token = None
        try:
            ctx = self._context.get_current()
            if self._baggage.get_baggage("call_sid", ctx) != call_sid:
                ctx = self._baggage.set_baggage("session.id", call_sid, context=ctx)
                ctx = self._baggage.set_baggage("call_sid", call_sid, context=ctx)
            span = self._tracer.start_span(name, context=ctx, attributes=self._clean(attributes or {}))
            token = self._context.attach(self._trace.set_span_in_context(span, ctx))
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a call
            self._failed("start_child", exc)
        outcome: dict[str, Any] = {"owlbank.outcome": "ok"}
        try:
            yield span
        except BaseException as exc:
            outcome = {"owlbank.outcome": "error", "owlbank.error_class": type(exc).__name__}
            raise
        finally:
            self.detach(token)
            self.end(span, outcome)

    def event(self, span: Any, name: str, attributes: Mapping[str, Any]) -> None:
        if span is None:
            return
        try:
            span.add_event(name, self._clean(attributes))
        except Exception as exc:  # noqa: BLE001
            self._failed("add_event", exc)

    def set(self, span: Any, attributes: Mapping[str, Any]) -> None:
        if span is None:
            return
        try:
            span.set_attributes(self._clean(attributes))
        except Exception as exc:  # noqa: BLE001
            self._failed("set_attributes", exc)

    def end(self, span: Any, attributes: Mapping[str, Any]) -> None:
        if span is None:
            return
        self.set(span, attributes)
        try:
            span.end()
        except Exception as exc:  # noqa: BLE001
            self._failed("end_span", exc)

    async def within(self, gen: AsyncIterator[Any], span: Any, call_sid: str) -> AsyncGenerator[Any, None]:
        """``gen``'s items, each produced with ``span`` current: the turn's work (and every task or
        thread it starts, which copy the context) runs under the turn span. The span is attached and
        detached around each step, so it never leaks into the consumer between items."""
        while True:
            token = self.attach(span, call_sid)
            try:
                item = await anext(gen)
            except StopAsyncIteration:
                return
            finally:
                self.detach(token)
            yield item

    async def with_hop(self, gen: AsyncIterator[Any], hop: _Hop, step: str | None) -> AsyncGenerator[Any, None]:
        """``gen``'s items with ``evalshape.HOP`` set while each is produced: Strands
        starts the hop's ``invoke_agent`` span inside one of these steps, and the
        ``EvalShapeProcessor`` gives it a user message when the hop has no caller words."""
        info: Any = None
        try:
            info = self._hop_info(caller_words=bool(hop.user_text and hop.user_text.strip()),
                                  caller_utterance=hop.utterance or None, step=hop.step or step)
        except Exception as exc:  # noqa: BLE001
            self._failed("hop_info", exc)
        while True:
            token = None
            if info is not None:
                try:
                    token = self._hop_var.set(info)
                except Exception as exc:  # noqa: BLE001
                    self._failed("hop_set", exc)
            try:
                item = await anext(gen)
            except StopAsyncIteration:
                return
            finally:
                if token is not None:
                    try:
                        self._hop_var.reset(token)
                    except Exception as exc:  # noqa: BLE001
                        self._failed("hop_reset", exc)
            yield item


def _relay_code(value: Any) -> str | int | None:
    """A ConversationRelay ``info``/``error`` frame's code, if it is an identifier or a number."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and 0 < len(value) <= 64 and all(c.isalnum() or c in "._-" for c in value):
        return value
    return None


class Bridge:
    """The per-call loop. All collaborators are injected."""

    #: Call observability: set by :meth:`enable_tracing` when telemetry is installed.
    tracing: _Tracing | None = None

    def __init__(
        self,
        *,
        store: Store,
        agent: AgentClient,
        backend: BackendClient,
        memory: MemoryClient | None,
        catalog: PromptCatalog,
        policy: Policy,
        sealer: Sealer,
        pii: PiiGuard,
        journal: SegmentLog | None = None,
        voice: VoiceChannelLike | None = None,
        handoff_factory: HandoffFactory = _default_handoff_factory,
        config: BridgeConfig | None = None,
        now: Callable[[], float] = time.time,
        io: Callable[..., Awaitable[Any]] | None = None,
        conversations: ConversationCloser | None = None,
        instance_id: str | None = None,
    ) -> None:
        self._store = store
        #: This process's ID in call state's ``agent_owner``: random at every start, so a
        #: restarted process never trusts a cache it no longer has.
        self.instance_id = instance_id or secrets.token_hex(8)
        #: The pending session start per call (owner write, then the agent's ``begin_session``);
        #: the session's first agent turn awaits it. Its result: the owner write landed.
        self._session_starts: dict[str, asyncio.Task[bool]] = {}
        self._conversations = conversations
        self._calls = CallStateRepo(store, now=now, sealer=sealer)
        self._agent = agent
        self._backend = backend
        self._memory = memory
        self._catalog = catalog
        self._policy = policy
        self._sealer = sealer
        self._pii = pii
        self._journal = journal or SegmentLog(store, now=now)
        self.voice = voice
        self._handoff_factory = handoff_factory
        self._cfg = config if config is not None else BridgeConfig()
        self._now = now
        self._io = io or asyncio.to_thread
        self._locks: dict[str, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        #: call SID -> {pool slot -> last phrase spoken}: the acknowledgements and each
        #: tool's pool. Bounded by MASK_ROTATION_MAX_CALLS.
        self._mask_rotation: dict[str, dict[str, str]] = {}
        self._sessions: dict[str, _Session] = {}
        #: call SID -> the session's speech and silence watch; set by session_start.
        self._speech: dict[str, _CallSpeech] = {}
        #: call SID -> the catalog key of the scripted chunk about to be yielded (``_mark``).
        self._scripted: dict[str, str] = {}
        #: Silence expiries in flight. Not in ``_tasks``: ``wait_idle`` never waits for silence.
        self._silence_tasks: set[asyncio.Task[Any]] = set()
        #: call SID -> whether call state showed no earlier AI session when the socket's nonce was
        #: consumed; taken by ``session_start``.
        self._first_session: dict[str, bool] = {}
        #: AI keypad shortcuts: call SID -> the session's key buffer, and its pending wake-up with
        #: the due time it was set for. Dropped with the session (``session_start``/``_end``).
        self._keypad: dict[str, keypads.KeypadBuffer] = {}
        self._keypad_wakes: dict[str, tuple[asyncio.TimerHandle, float]] = {}
        #: call SID -> the session's keypad work in flight (wake-ups' actions, turns, a re-sent
        #: ``end``): cancelled when the session starts or ends. Not in ``_tasks``: ``wait_idle``
        #: never waits for them.
        self._keypad_tasks: dict[str, set[asyncio.Task[Any]]] = {}

    @property
    def keypad_enabled(self) -> bool:
        """Whether keys during the AI session act (``OWLBANK_AI_KEYPAD``)."""
        return bool(self._cfg.ai_keypad)

    # ------------------------------------------------------------------ call observability

    def enable_tracing(self, tracer: Any, *, flush: Callable[[], Any] | None = None,
                       retain: Callable[[str], None] | None = None) -> None:
        """Turn on the bridge's spans (``relay.session``, ``bridge.turn``, ``memory.reload`` and
        their events) with ``tracer`` (the installed telemetry's). Called once at start-up, only
        when the switch is on. An agent client with an ``enable_tracing`` hook gets the same
        tracing, for its ``memory.reload`` spans.

        With ``traceContent=redacted`` the exact-value scrub runs at export, so ``retain`` (the
        redactor's :meth:`~owlbank.telemetry.bridge.TranscriptRedactor.retain`) is called with the
        call just before its sensitive values are forgotten: it keeps a snapshot for
        ``RETAIN_TTL_S``, for the call's spans exported after the forget. ``flush`` (blocking; the
        tracer provider's bounded ``force_flush``) then runs in a thread at each session end, so
        the session's spans don't wait for the next batch (it frees nothing); it never delays the
        forget, and a flush that fails or times out is counted and logged."""
        self._trace_flush = flush
        self._trace_retain = retain
        self._relay_spans: dict[str, Any] = {}
        self._turn_spans: dict[str, Any] = {}
        self._handoff_tools: dict[str, str] = {}
        self.tracing = _Tracing(tracer)
        hook = getattr(self._agent, "enable_tracing", None)
        if callable(hook):
            hook(self.tracing)

    def _note_handoff(self, call_sid: str, tool: str | None) -> None:
        """The session is ending on a hand-off (``relay.session``'s ``owlbank.handoff_tool``): keys
        no longer act, also once TAC has sent the ``end`` and cleared the context's
        ``pending_handoff_data``."""
        speech = self._speech.get(call_sid)
        if speech is not None:
            speech.handoff = True
        if self.tracing is not None and tool:
            self._handoff_tools[call_sid] = str(tool)

    def trace_relay_frame(self, call_sid: str | None, kind: str, code: Any) -> None:
        """A ConversationRelay ``info`` or ``error`` frame: a ``relay.<kind>`` event on the session
        span with the frame's type and code only. No-op when tracing is off."""
        tracing = self.tracing
        if tracing is None or not call_sid:
            return
        tracing.event(self._relay_spans.get(call_sid), f"relay.{kind}",
                      {"owlbank.type": kind, "owlbank.code": _relay_code(code)})

    def trace_interrupt(self, call_sid: str | None, interrupt_ms: Any, chars_played: int | None) -> None:
        """A barge-in: a ``relay.interrupt`` event on the session span, and the same two
        attributes on the turn span if a turn is open. Never the utterance text."""
        tracing = self.tracing
        if tracing is None or not call_sid:
            return
        ms = interrupt_ms if isinstance(interrupt_ms, int | float) and not isinstance(interrupt_ms, bool) else None
        attributes = {"owlbank.interrupt_ms": ms, "owlbank.chars_played": chars_played}
        tracing.event(self._relay_spans.get(call_sid), "relay.interrupt", attributes)
        tracing.set(self._turn_spans.get(call_sid), attributes)

    def _trace_session_start(self, call_sid: str, first_session: bool | None) -> None:
        tracing = self.tracing
        if tracing is None:
            return
        tracing.end(self._relay_spans.pop(call_sid, None), {"owlbank.end_reason": "replaced"})
        self._handoff_tools.pop(call_sid, None)
        resumed = None if first_session is None else not first_session
        span = tracing.start(SPAN_SESSION, call_sid, {"owlbank.resumed": resumed})
        if span is not None:
            self._relay_spans[call_sid] = span

    def _trace_session_end(self, call_sid: str, reason: str, close_code: int | None, session: _Session) -> None:
        tracing = self.tracing
        if tracing is None:
            return
        tracing.end(self._relay_spans.pop(call_sid, None), {
            "owlbank.end_reason": reason, "owlbank.turns": session.turns,
            "owlbank.close_code": close_code, "owlbank.handoff_tool": self._handoff_tools.pop(call_sid, None),
            "owlbank.input_tokens": session.input_tokens, "owlbank.output_tokens": session.output_tokens})

    # ------------------------------------------------------------------ TAC hooks

    async def on_message_ready(self, message: str, context: Any, memory: Any = None) -> None:
        """TAC ``on_message_ready`` callback. ``memory`` is always ``None`` (``memory_mode="never"``)."""
        with obs.call_context(getattr(context, "call_sid", None), component="bridge"):
            await self._send(context, self.run_turn(context, message))
        return

    # -- WebSocket session lines: started by the app on a verified ``setup``, ended when the socket closes

    def session_start(self, call_sid: str) -> None:
        with obs.call_context(call_sid, component="bridge"):
            self._sessions[call_sid] = _Session(started=time.monotonic())
            self._silence_drop(call_sid)
            self._keypad_drop(call_sid)
            # The session's lock exists from its start, so keypad work can tell an ended session
            # (``session_end`` pops it) from one that has had no turn yet.
            self._locks.setdefault(call_sid, asyncio.Lock())
            watch = SilenceWatch()
            self._silence_configure(watch)
            first_session = self._first_session.pop(call_sid, None)
            self._speech[call_sid] = _CallSpeech(watch, first_session=first_session)
            emit(log, logging.INFO, "ws_session_start")
            self._trace_session_start(call_sid, first_session)

    def session_end(self, call_sid: str, reason: str, close_code: int | None = None) -> None:
        session = self._sessions.pop(call_sid, None) or _Session(started=time.monotonic())
        self._forget_sensitive(call_sid, self._locks.pop(call_sid, None))
        speech = self._speech.get(call_sid)
        if speech is not None and not speech.watch.speaker_events:
            with obs.call_context(call_sid, component="bridge"):  # Once per session
                emit(log, logging.INFO, "silence_timing", mode=TIMING_ESTIMATE)
        self._silence_drop(call_sid)
        self._keypad_drop(call_sid)
        self._first_session.pop(call_sid, None)
        start = self._session_starts.get(call_sid)
        if start is not None and start.done() and not start.cancelled() and start.result():
            del self._session_starts[call_sid]  # a pending or failed one stays for the next turn
        with obs.call_context(call_sid, component="bridge"):
            emit(log, logging.INFO, "ws_session_end", reason=reason,
                 duration_ms=round((time.monotonic() - session.started) * 1000, 1),
                 prompts=session.prompts, tool_calls=session.tool_calls, turns=session.turns,
                 input_tokens=session.input_tokens, output_tokens=session.output_tokens)
            self._trace_session_end(call_sid, reason, close_code, session)
            if self._conversations is not None and not self._cfg.reuse_conversation:
                task = asyncio.create_task(self._close_conversations(call_sid))  # copies the bound call SID
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

    def _forget_sensitive(self, call_sid: str, lock: asyncio.Lock | None) -> None:
        """Drop this process's copy of the call's sensitive values at session end, so
        per-call state doesn't grow. A turn still running (it holds the call's lock) keeps them
        until it ends; a new session of the call, started meanwhile, keeps them (it restores
        them on its own reads anyway). With tracing, the redactor keeps its own snapshot of the values
        for ``RETAIN_TTL_S`` after the forget (:meth:`enable_tracing`), so the forget never waits
        for the export; the session's spans are flushed in the background."""
        self._flush_traces()
        self._forget_now(call_sid, lock)

    def _flush_traces(self) -> None:
        """The session end's bounded flush, in the background (:meth:`enable_tracing`)."""
        tracing = self.tracing
        flush = getattr(self, "_trace_flush", None) if tracing is not None else None
        if tracing is None or flush is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no loop (tests): the batch processor exports on its own schedule
            return

        async def flushed() -> None:
            try:
                done = await asyncio.wait_for(asyncio.to_thread(flush), TRACE_FLUSH_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001 - the spans export later, still scrubbed
                tracing.flush_failed(exc)
                return
            if done is False:  # force_flush timed out inside
                tracing.flush_failed(None)

        task = loop.create_task(flushed())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _forget_values(self, call_sid: str) -> None:
        """``sensitive.forget``, after the trace redactor took its snapshot of the values."""
        retain = getattr(self, "_trace_retain", None) if self.tracing is not None else None
        if retain is not None:
            try:
                retain(call_sid)
            except Exception as exc:  # noqa: BLE001 - its spans fall back to the registered values
                if self.tracing is not None:
                    self.tracing.retain_failed(exc)
        sensitive.forget(call_sid)

    def _forget_now(self, call_sid: str, lock: asyncio.Lock | None) -> None:
        if lock is None or not lock.locked():
            self._forget_values(call_sid)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no loop: nothing can still be running a turn
            self._forget_values(call_sid)
            return

        async def after_turn() -> None:
            async with lock:
                if call_sid not in self._sessions:
                    self._forget_values(call_sid)

        task = loop.create_task(after_turn())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _close_conversations(self, call_sid: str) -> None:
        """Best effort: a failure here must never break the call (the closed-timeout is the backstop)."""
        assert self._conversations is not None  # noqa: S101  (only scheduled when injected)
        try:
            closed = await self._conversations.close_active(call_sid)
            emit(log, logging.INFO, "conversations_closed", count=closed)
        except Exception as exc:  # noqa: BLE001
            log_caught(log, "closing the finished segment's conversation failed", exc,
                       operation="close_conversations", decision="ignored_best_effort")

    def _session(self, call_sid: str) -> _Session:
        return self._sessions.setdefault(call_sid, _Session(started=time.monotonic()))

    async def accept_connection(self, call_sid: str | None, nonce: str | None) -> bool:
        """``True`` only if ``nonce`` matches the call's ``ws_nonce_hash``; it is then
        consumed (a second use fails). Unknown or expired call state, or no nonce: ``False``."""
        if not call_sid or not isinstance(call_sid, str) or not nonce:
            return False
        try:
            return bool(await self._io(self._consume_nonce, call_sid, nonce))
        except Exception as exc:
            log_caught(log, "relay nonce check failed", exc, level=logging.ERROR,
                       operation="consume_nonce", decision="reject")
            return False

    def _consume_nonce(self, call_sid: str, nonce: str) -> bool:
        first: list[bool] = []

        def fn(state: CallState) -> Mapping[str, Any]:
            if state.is_expired(self._now()) or not ws_nonce.matches(state.ws_nonce_hash, nonce):
                raise _NonceRejected
            # Read here, before this session's own owner write (``ai_sessions + 1``)
            # can land, so the decision can't race it.
            first.append(not had_ai_session(state))
            return {"ws_nonce_hash": None}

        try:
            callstate.mutate(self._store, call_sid, fn, now=self._now)
        except (_NonceRejected, KeyError, ValueError, callstate.VersionConflict) as exc:
            emit(log, logging.DEBUG, "nonce_rejected", operation="consume_nonce",
                 error_class=type(exc).__name__, decision="reject")
            return False
        if first:
            self._first_session[call_sid] = first[-1]  # the attempt that wrote
        return True

    async def on_setup(self, call_sid: str | None, custom_parameters: Mapping[str, Any] | None = None) -> None:
        """Called by the WebSocket wrapper on ``setup`` (TAC has no hook). Idempotent."""
        if not call_sid:
            return
        setup_at = time.monotonic()  # the welcome greeting starts playing about now
        speech = self._speech.get(call_sid)
        epoch = speech.watch.input_epoch if speech is not None else 0  # caller input / turns only
        with obs.call_context(call_sid, component="bridge"):
            self._begin_session(call_sid)
            await self._on_setup(call_sid, custom_parameters, setup_at=setup_at, epoch=epoch)

    def _begin_session(self, call_sid: str) -> None:
        """The session start, in the background so it never delays the greeting: the
        session-ownership decision and write, then the agent's ``begin_session`` hook with that
        decision; the session's first turn waits for it. A client without the hook is skipped (no
        ownership to track)."""
        hook = getattr(self._agent, "begin_session", None)
        if hook is None:
            return
        task = asyncio.create_task(self._run_begin_session(hook, call_sid))  # copies the bound call SID
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._session_starts[call_sid] = task

    async def _run_begin_session(self, hook: Callable[..., Awaitable[None]], call_sid: str) -> bool:
        """Best effort, never raises: a failure here must never break the call. Returns whether the
        ``agent_owner`` write landed; if not, ``history`` is ``reload`` (always safe: it reads) and
        the first turn tries the write again (:meth:`_session_ready`)."""
        owner_ok, history = False, "reload"
        try:
            history = await self._io(self._calls.claim_agent_owner, call_sid, self.instance_id)
            owner_ok = True
        except Exception as exc:  # noqa: BLE001
            log_caught(log, "agent owner write failed", exc, operation="claim_agent_owner",
                       decision="reload_history")
        emit(log, logging.INFO, "agent_session_start", history=history, owner_written=owner_ok)
        tracing = self.tracing
        # A history reload at session start (the client's ``memory.reload`` span) nests
        # under the session span, with the call in baggage; this task has no OTel context of its own.
        token = tracing.attach(self._relay_spans.get(call_sid), call_sid) if tracing is not None else None
        try:
            await hook(call_sid, history=history)
        except Exception as exc:  # noqa: BLE001
            log_caught(log, "agent begin_session failed", exc,
                       operation="begin_session", decision="ignored_best_effort")
        finally:
            if tracing is not None:
                tracing.detach(token)
        return owner_ok

    async def _session_started(self, call_sid: str) -> None:
        """Wait for the pending session start (if any) to finish, whatever its outcome; never
        raises its failure (:meth:`_session_ready` handles that before the agent turn)."""
        task = self._session_starts.get(call_sid)
        if task is None or task.done():
            return
        try:
            await asyncio.shield(task)  # a barge-in cancels this turn, not the start
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
        except Exception as exc:  # noqa: BLE001 - _run_begin_session never raises; defensive
            log_caught(log, "session start failed", exc, operation="session_start",
                       decision="continue_to_session_ready")

    async def _session_ready(self, call_sid: str) -> None:
        """The session's first agent turn waits for the session start (the owner write and the
        agent's ``begin_session``), so another process can never see an older owner once this
        session has run a turn, and the turn runs on the agent the decision chose. A failed owner
        write is tried once more here; if that fails too, the turn fails (retryable, so the bridge's
        retry tries again, then escalates)."""
        task = self._session_starts.get(call_sid)
        if task is None:
            return
        try:
            owner_ok = await asyncio.shield(task)  # a barge-in cancels this turn, not the start
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
            owner_ok = False
        if not owner_ok:
            try:
                await self._io(self._calls.claim_agent_owner, call_sid, self.instance_id)
            except Exception as exc:
                raise AgentError("agent owner write failed", retryable=True,
                                 code="session_owner_unavailable") from exc
        if self._session_starts.get(call_sid) is task:
            del self._session_starts[call_sid]

    async def _on_setup(self, call_sid: str, custom_parameters: Mapping[str, Any] | None, *,
                        setup_at: float | None = None, epoch: int = 0) -> None:
        params = dict(custom_parameters or {})
        expected = params.get("callSid")
        if expected and expected != call_sid:  # defence in depth only
            log.warning("setup customParameters callSid does not match the socket's callSid",
                        extra={"decision": "continue_with_socket_call_sid"})
        try:
            view = await self._io(self._calls.load, call_sid)
        except Exception as exc:
            log_caught(log, "call state read failed on setup", exc, level=logging.ERROR,
                       operation="load_call_state", decision="skip_resume")
            return
        if view is None:
            return
        await self._arm_greeting(call_sid, view, setup_at if setup_at is not None else time.monotonic(),
                                 epoch)
        if not self._is_immediate(view):
            return
        task = asyncio.create_task(self._resume_on_setup(call_sid))  # copies the bound call SID
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def on_interrupt(self, context: Any, interrupt: Any) -> None:
        """TAC ``on_interrupt`` callback: journal what the caller heard."""
        call_sid = getattr(context, "call_sid", None)
        heard = getattr(interrupt, "utterance_until_interrupt", None) or ""
        if call_sid:
            view = None
            try:
                view = await self._io(self._calls.load, call_sid)
            except Exception as exc:
                log_caught(log, "call state read failed on interrupt", exc, call_sid=call_sid,
                           operation="load_call_state", decision="fallback_default_language")
            lang = view.language if view is not None else "en-US"
            text = await self._turn_text(call_sid, heard, lang, 2000)
            await self._append(call_sid, {"type": "ai_turn", "interrupted": True, "text": text})

    async def wait_idle(self, timeout: float = 30.0) -> None:
        """Wait for background resumes (tests). Raises instead of spinning if they never settle."""
        end = time.monotonic() + timeout
        while self._tasks:
            if time.monotonic() > end:
                raise TimeoutError(f"background tasks never settled: {sorted(map(repr, self._tasks))}")
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # ------------------------------------------------------------------ resume on setup

    @staticmethod
    def _is_immediate(view: CallView) -> bool:
        env = view.pending_result
        if env and view.pending_tool and env.get("next") == "retry_original_tool":
            return True  # step-up succeeded: retry and answer now, no dead air (only await_caller waits)
        return bool(env and env.get("next") == "handle_utterance"
                    and (view.interrupted_utterance or env.get("callerUtterance")))

    async def _resume_on_setup(self, call_sid: str) -> None:
        if self.voice is None:
            return
        deadline = time.monotonic() + self._cfg.session_wait_s
        context = None
        while time.monotonic() < deadline:
            context, _ = self._session_for_call(call_sid)
            if context is not None:
                break
            await asyncio.sleep(self._cfg.session_poll_s)
        if context is None:
            log.warning("no TAC session for the resume; it will run on the first prompt", extra={
                "decision": "defer_to_first_prompt"})
            return
        # On a reconnect (after a step-up or a scripted flow) the TAC session from the first connection
        # already exists, so the wait above ends at once, but the NEW WebSocket is only registered after
        # TAC finishes its own setup (a conversation lookup that takes seconds). Sending before that makes
        # TAC log "No websocket connection" and drop the send, and the turn never runs (found at the first
        # real step-up call: the caller heard "Thanks, you're verified" and then silence). Wait for the socket.
        # The session is re-chosen on every poll, so a stale session (no socket, its CLOSED webhook not
        # yet in) never holds the resume while a newer session for the same CallSid gets its socket.
        live = await self._wait_for_socket(call_sid, context)
        if live is None:
            log.warning("TAC never registered the new WebSocket; the resume will run on the first prompt",
                        extra={"decision": "defer_to_first_prompt", "operation": "resume_on_setup"})
            return
        await self._send(live, self.run_turn(live, None))

    def _session_for_call(self, call_sid: str) -> tuple[Any | None, bool]:
        """The TAC session for this CallSid that has a registered WebSocket, else the newest (TAC's own
        lookup returns the oldest, which can be stale); and whether it has a socket."""
        if self.voice is None:
            return None, False
        get_socket = getattr(self.voice, "get_websocket", None)

        def has_socket(session: Any) -> bool:
            return get_socket is not None and get_socket(session.conversation_id) is not None

        return pick_session(tac_sessions_for_call(self.voice, call_sid), has_socket)

    async def _wait_for_socket(self, call_sid: str, context: Any) -> Any | None:
        """The session for ``call_sid`` once TAC has a WebSocket registered for it (bounded by
        ``session_wait_s``); ``None`` if none gets one in time."""
        if getattr(self.voice, "get_websocket", None) is None:  # no accessor: nothing to wait for
            return context
        deadline = time.monotonic() + self._cfg.session_wait_s
        while True:
            current, has_socket = self._session_for_call(call_sid)
            if has_socket:
                return current
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(self._cfg.session_poll_s)

    async def _send(self, context: Any, gen: AsyncIterator[str] | str) -> None:
        if self.voice is None:
            raise RuntimeError("Bridge.voice is not set")
        speech = self._speech.get(getattr(context, "call_sid", None) or "")
        if speech is None:
            await self.voice.send_response(context.conversation_id, gen)
            return
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        speech.sends[done] = asyncio.current_task()  # until TAC's last (and end) are sent too
        try:
            await self.voice.send_response(context.conversation_id, gen)
        finally:
            speech.sends.pop(done, None)
            done.set_result(None)

    def on_relay_end(self, call_sid: str | None) -> None:
        """``PeekingWebSocket``: an ``end`` frame is going out on the session's socket (TAC's, or the
        bridge's resend through TAC). Set before the frame's send is awaited, so a turn cancelled
        while suspended in it still counts the ``end`` as sent (keys stay dropped)."""
        speech = self._speech.get(call_sid or "")
        if speech is not None:
            speech.end_out = speech.handoff = True

    # ------------------------------------------------------------------ the turn

    async def run_turn(self, context: Any, message: str | None, *,
                       keypad: _KeypadTurn | None = None) -> AsyncGenerator[str, None]:
        """Text chunks for the caller. ``message is None`` means a resume started at ``setup``.
        ``keypad``: a keypad ``yes``/``no`` turn, checked again once it has the call's lock:
        with a hand-off pending or the session gone it yields nothing and calls no model."""
        call_sid = context.call_sid
        speech = self._speech.get(call_sid)
        if speech is not None:  # Never fires during a turn; the turn's speech re-arms it
            speech.watch.turn_started()
            speech.open = speech.turn_spoke = False
            self._silence_reschedule(call_sid)
        self._scripted.pop(call_sid, None)
        stats: _TurnStats | None = None
        try:
            lock = self._locks.setdefault(call_sid, asyncio.Lock())
            async with lock:
                if keypad is not None and self._keypad_refuse(call_sid, context):
                    keypad.refused = True
                    return
                session = self._session(call_sid)
                stats = _TurnStats(number=session.turns + 1, started=time.monotonic(),
                                   prompt=message is not None)
                if speech is not None:
                    speech.turn = stats  # Agent_speech_started
                tracing = self.tracing
                if tracing is not None:  # Each turn is its own trace in the call's session
                    stats.tracing = tracing
                    stats.span = tracing.start(SPAN_TURN, call_sid, {"owlbank.turn": stats.number,
                                                                     "owlbank.resumed": message is None})
                    if stats.span is not None:
                        self._turn_spans[call_sid] = stats.span
                try:
                    async with aclosing(self._turn_locked(context, call_sid, message, stats)) as chunks, \
                            (nullcontext(chunks) if tracing is None else
                             aclosing(tracing.within(chunks, stats.span, call_sid))) as traced:
                        async for chunk in traced:
                            self._note_out(call_sid, chunk)
                            if chunk:
                                now = time.monotonic()
                                if stats.first_out is None:
                                    stats.first_out = now
                                if stats.answering and stats.answered_at is None:
                                    stats.answered_at = now
                            yield chunk
                    if stats.outcome == "cancelled":
                        stats.outcome = "ok"
                except Exception as exc:
                    stats.outcome = "error"
                    log_caught(log, "bridge turn failed; escalating", exc, level=logging.ERROR,
                               turn=stats.number, operation="run_turn", decision="escalate")
                    apology = self._phrase("error.generic", "en-US")
                    self._mark(call_sid, "error.generic")
                    self._note_out(call_sid, apology)
                    if apology and stats.first_out is None:
                        stats.first_out = time.monotonic()
                    yield apology
                    self._set_handoff(context, call_sid, {"reason": "escalate", "tool": "escalate",
                                                          "escalationReason": FALLBACK_REASON})
                finally:
                    self._end_turn(call_sid, session, stats)
        finally:
            if speech is not None:
                self._silence_after_turn(context, call_sid, speech, stats)

    def _end_turn(self, call_sid: str, session: _Session, stats: _TurnStats) -> None:
        """The ``turn`` line: ids, counts and timings only (no prompt, reply or tool arguments)."""
        session.turns += 1
        session.prompts += 1 if stats.prompt else 0
        session.tool_calls += len(stats.tools)
        session.input_tokens += stats.input_tokens or 0
        session.output_tokens += stats.output_tokens or 0
        fields: dict[str, Any] = {
            "turn": stats.number, "duration_ms": round((time.monotonic() - stats.started) * 1000, 1),
            "stop_reason": stats.stop_reason, "hops": stats.hops, "tools": list(stats.tools),
            "outcome": stats.outcome, "resumed": not stats.prompt}
        if stats.input_tokens is not None:
            fields["input_tokens"] = stats.input_tokens
        if stats.output_tokens is not None:
            fields["output_tokens"] = stats.output_tokens
        # The caller's silence (anything heard, a masking phrase included) and wait (the agent's
        # own answer) from the turn's start; null when nothing of that kind went out.
        fields["silence_ms"] = _ms_since(stats.first_out, stats.started)
        fields["wait_ms"] = _ms_since(stats.answered_at, stats.started)
        fields["text_before_tool"] = stats.text_before_tool
        with obs.call_context(call_sid, component="bridge"):
            emit(log, logging.INFO, "turn", **fields)
        if stats.tracing is not None and stats.span is not None:
            if self._turn_spans.get(call_sid) is stats.span:
                del self._turn_spans[call_sid]
            stats.tracing.end(stats.span, {
                "owlbank.turn": stats.number, "owlbank.stop_reason": stats.stop_reason,
                "owlbank.hops": stats.hops, "owlbank.tools": list(stats.tools),
                "owlbank.outcome": stats.outcome, "owlbank.resumed": not stats.prompt,
                "owlbank.input_tokens": stats.input_tokens, "owlbank.output_tokens": stats.output_tokens,
                "owlbank.silence_ms": fields["silence_ms"], "owlbank.wait_ms": fields["wait_ms"],
                "owlbank.text_before_tool": stats.text_before_tool,
                # Counts for every content mode; the text is a TEXT key (traceContent)
                "owlbank.reasoning_blocks": stats.reasoning_blocks, "owlbank.reasoning_chars": stats.reasoning_chars,
                "owlbank.markup_tags": stats.markup_tags, "owlbank.reasoning_text": stats.reasoning_text or None})

    async def _turn_locked(self, context: Any, call_sid: str, message: str | None,
                           stats: _TurnStats | None = None) -> AsyncGenerator[str, None]:
        stats = stats or _TurnStats(number=0, started=time.monotonic(), prompt=message is not None)
        try:
            view = await self._io(self._calls.load, call_sid)
        except Exception as exc:
            log_caught(log, "call state read failed", exc, level=logging.ERROR,
                       operation="load_call_state", decision="treat_as_no_call_state")
            view = None
        if view is None:
            if message is None:
                return
            log.warning("no call state for this call; escalating", extra={"decision": "escalate"})
            emit(log, logging.INFO, "escalation_requested", reason=FALLBACK_REASON, source="bridge")
            self._mark(call_sid, "error.generic")
            yield self._phrase("error.generic", "en-US")
            self._set_handoff(context, call_sid, {"reason": "escalate",
                                                  "escalationReason": FALLBACK_REASON})
            return

        if view.pending_tool is not None or view.pending_result is not None:
            # The session's history decision reads these as signs of an earlier AI session;
            # the first hop clears them, so let the decision read them first.
            await self._session_started(call_sid)
        hop = await self._first_hop(context, view, message)
        if hop is None:
            return
        # The first hop claimed (cleared in the store) whatever was pending; the rest of the turn
        # must see that too, or end_call's pending check refuses on a finished step.
        view = replace(view, pending_tool=None, pending_result=None)
        welcome = await self._session_welcome(call_sid, view)
        if welcome:
            self._place_welcome(call_sid, hop, welcome)
        try:
            system_prompts, per_call, clock = await self._system_prompt(context, view)
        except Exception as exc:
            log_caught(log, "system prompt build failed", exc, level=logging.ERROR,
                       operation="system_prompt", decision="fallback_system_prompt")
            (system_prompts, per_call), clock = (self._fallback_system_prompt(view),
                                                 self._clock_line(context, view.call_sid, view.language))
        # Agent prompt caching: the static part per model key goes before the system cache point,
        # the per-call part and the clock line (the call's, :meth:`_clock_line`) after it.
        system_suffix = f"{per_call}\n{clock}"

        # Reasoning and markup come out first, then the <silent/> marker
        reasoning = ReasoningFilter(keep=stats.span is not None)
        silent = SilentFilter()
        planner = self._planner(call_sid)  # One per turn, from the current policy
        tm = _TurnMask(planner)
        try:
            async with aclosing(self._hops(context, call_sid, view, hop, system_prompts, system_suffix,
                                           _TextFilters(reasoning, silent), tm, stats)) as out:
                async for chunk in out:
                    yield chunk
        finally:
            for p in planner.end_turn():  # the turn ended with the phrase still waiting
                self._log_skip(p.tool, p.phrase_key, SKIP_TURN_ENDED, stats, p)
            _reasoning_removed(reasoning, stats)

    def _place_welcome(self, call_sid: str, hop: _Hop, welcome: str) -> None:
        """Where the session's welcome goes in the agent's history. A tool result pending (a
        re-entry after a hand-off): a ``welcome`` field in that result's envelope, since the agent's
        ``toolUse`` must be followed directly by its ``toolResult``; taken now. Else the synthetic
        pair, recorded before the hop's agent call (:meth:`_record_welcome`), and taken only once
        recorded."""
        if not hop.tool_results:
            hop.welcome = welcome
            return
        self._welcome_taken(call_sid)
        first = hop.tool_results[0]
        if not isinstance(first.content, Mapping):
            emit(log, logging.WARNING, "welcome_not_recorded", reason="text_result",
                 decision="welcome_not_recorded")
            return
        hop.tool_results[0] = ToolResult(first.tool_use_id, {**first.content, "welcome": welcome}, first.status)
        emit(log, logging.INFO, "welcome_recorded", path="envelope", chars=len(welcome))

    async def _hops(self, context: Any, call_sid: str, view: CallView, hop: _Hop | None,
                    system_prompts: Mapping[str, str], system_suffix: str, filters: _TextFilters,
                    tm: _TurnMask, stats: _TurnStats) -> AsyncGenerator[str, None]:
        """The turn's agent hops and tool calls (the body of :meth:`_turn_locked`). Every hop sends
        the same ``system_prompts`` (the static part per model key) and ``system_suffix`` (the
        per-call part and the call's clock line), so a turn's system blocks never change."""
        planner = tm.planner
        reasoning, silent = filters.reasoning, filters.silent
        failures = 0
        hops = 0
        last_tool: str | None = None  # the step a tool-result hop resumes after
        # The agent's own words were spoken this turn (any hop): a hand-off holds for them.
        released = False
        while hop is not None and hops < self._cfg.max_hops:
            hops += 1
            stats.hops = hops
            # The WARNING is decided on the last model message alone (reset each hop, a
            # retry included, and for each later model message of the hop, below)
            stats.new_message(reasoning)
            ended = False  # a MessageEnd came: the next event starts a new model message
            text_parts: list[str] = []  # the agent's own words only, never a masking phrase
            # This hop's released text, (first chunk sent at, chars), for a hold
            hop_first: float | None = None
            hop_chars = 0
            calls: list[ToolCall] = []
            stop: str | None = None
            hop_model: str | None = None  # The journal's ``model`` (the entry's ID)
            # Only the turn's first hop, and only for the caller's words, is raced against
            # the timed acknowledgement. Tool-result hops never set ``timer``.
            timer_s = planner.timer_s() if hop.timer else None
            if timer_s is not None:
                # Counted from when the caller's words reached the bridge (the turn's start),
                # not from after the turn's preparation (system prompt, memory).
                timer_s = max(0.05, timer_s - (time.monotonic() - stats.started))
            try:
                await self._session_ready(call_sid)  # First hop only; then a no-op
                if hop.welcome is not None:  # Before the hop's agent call, exactly once
                    await self._record_welcome(call_sid, view, hop.welcome)
                    hop.welcome = None
                    self._welcome_taken(call_sid)  # no await since the record: never twice
                _assert_no_token(view.sealed_token, list(system_prompts.values()))
                # Agent prompt caching: every part of the system text, the per-call part
                # and the clock line (the suffix) as well as the static parts
                _assert_no_token(view.sealed_token, system_suffix)
                _assert_no_token(view.sealed_token, hop.user_text or "")
                _assert_no_token(view.sealed_token, [r.content for r in hop.tool_results])
                if hop.record_only:
                    events = self._agent.record_results(call_sid, list(hop.tool_results),
                                                        system_prompts=system_prompts,
                                                        system_suffix=system_suffix, turn_id=stats.number)
                else:
                    # Every hop of this turn (and each retry) carries the turn's
                    # number, so a later hop keeps the entry the turn moved to; a new turn chooses
                    # afresh
                    events = self._agent.turn(call_sid, user_text=hop.user_text,
                                              tool_results=list(hop.tool_results),
                                              system_prompts=system_prompts, system_suffix=system_suffix,
                                              turn_id=stats.number)
                if timer_s is not None:
                    tm.deadlines.arm("ack", time.monotonic() + timer_s)
                hop_events = None if self.tracing is None else self.tracing.with_hop(events, hop, last_tool)
                # The stream is raced with the acknowledgement and a scheduled tool phrase;
                # closes the agent stream on barge-in (the race first, then the stream)
                async with aclosing(events), (nullcontext() if hop_events is None else aclosing(hop_events)), \
                        aclosing(_race_deadlines(events if hop_events is None else hop_events,
                                                 tm.deadlines)) as stream:
                    async for ev in stream:
                        if isinstance(ev, _Fired):
                            async for text in self._on_fired(ev.name, tm, view, stats):
                                yield text
                            continue
                        if ended and not isinstance(ev, TurnEnd):
                            # The next model message of the hop (after a Gateway tool or an
                            # unknown tool name) is judged on its own. Not at MessageEnd itself: the
                            # hop's last message is the one the WARNING reads.
                            stats.new_message(reasoning)
                            ended = False
                        if isinstance(ev, (ToolCall, ToolStarted)):
                            tm.deadlines.cancel("ack")  # a tool came first
                        # What the caller hears, and the journal keeps, is the text
                        # without reasoning or markup; reasoning alone is no speech for the ack. The
                        # client's MessageEnd ends each model message, whatever its tool use (one
                        # the bridge never sees included), so the filter is flushed there: an
                        # unclosed block never swallows the next model message of the hop. An inline
                        # ToolCall comes after its message's end (a no-op then: nothing in between);
                        # it ends the message too for a client that sends no MessageEnd. A tool-use
                        # block ends the message's text too (TextBeforeTool, ToolStarted): words held
                        # as a possible tag are spoken before the tool's masking phrase, not after.
                        said = ""
                        if isinstance(ev, TextDelta):
                            said = reasoning.feed(ev.text)
                        elif isinstance(ev, (MessageEnd, ToolCall, TextBeforeTool, ToolStarted)):
                            said = reasoning.flush()
                        if said:
                            tm.deadlines.cancel("ack")  # speech came first
                            text_parts.append(said)
                            # the filter releases the marker whole, in one piece
                            stats.silenced = stats.silenced or SILENT_MARK in said
                            out = silent.feed(said)
                            if out:
                                stats.answering = True  # the chunk the turn sends next is the agent's answer
                                stats.spoke = stats.spoke or not out.isspace()
                                released = True
                                hop_first = time.monotonic() if hop_first is None else hop_first
                                hop_chars += len(out)
                                yield out
                                self._on_answer(tm, stats)
                        if isinstance(ev, TextBeforeTool):
                            # A tool-use block started after this message's
                            # text was spoken. The text is in text_parts (journaled below); the
                            # tool call follows as usual. Counts and the tool only, never text.
                            stats.text_before_tool += 1
                            emit(log, logging.WARNING, "ai_text_spoken_before_tool", chars=ev.chars,
                                 tool=ev.tool, hop=hops, turn=stats.number)
                        elif isinstance(ev, TextSuppressed):
                            # Text from a message that also called a tool. The agent client
                            # never yields it as TextDelta, so it is neither spoken nor journaled.
                            # Log the length only, never the text.
                            emit(log, logging.INFO, "ai_text_suppressed", chars=ev.chars, hop=hops,
                                 turn=stats.number)
                        elif isinstance(ev, ToolCall):
                            calls.append(ev)
                        elif isinstance(ev, ToolStarted):
                            # A Gateway tool started inside the agent loop.
                            # Masked like an inline tool call, and it counts as a tool call
                            # for the acknowledgement; never run, answered or counted
                            # otherwise (no ToolResult, no hand-off, not in the turn's tools).
                            tool = _gateway_tool_name(ev.name)
                            if tool is not None and not hop.record_only:
                                due = self._schedule_tool_phrase(tm, view, stats, [(tool, ev.input)])
                                if due is not None:
                                    self._log_mask(due.key, due.sched.tool, "tool", stats,
                                                   after_ms=due.sched.after_ms, source=due.sched.source)
                                    self._mark(call_sid, due.key)
                                    yield due.text
                                    await self._tool_phrase_spoken(call_sid, due)
                            else:
                                planner.saw_tool_call = True
                        elif isinstance(ev, TurnEnd):
                            stop = ev.stop_reason
                            stats.stop_reason = stop
                            hop_model = ev.model
                            if ev.usage:
                                stats.input_tokens = (stats.input_tokens or 0) + ev.usage.get("input_tokens", 0)
                                stats.output_tokens = (stats.output_tokens or 0) + ev.usage.get("output_tokens", 0)
                        if isinstance(ev, MessageEnd):
                            ended = True
                        elif isinstance(ev, ToolCall):
                            # A message that calls an inline tool (a hand-off included) is
                            # never the turn's last: its reasoning hides no answer
                            stats.new_message(reasoning)
                tm.deadlines.cancel("ack")
                if stop in ERROR_STOP_REASONS and not calls:
                    raise AgentError(f"turn ended with {stop}", retryable=stop == "timeout_exceeded")
            except (AgentError, TokenLeakError) as exc:
                tm.deadlines.cancel("ack")
                reasoning.flush()  # a held tag prefix or an unclosed block of the failed call: never spoken
                dropped = silent.flush()  # stream mode: a held marker prefix of the failed call is never spoken
                failures += 1
                # Stream mode: words of the failed call the caller already heard are journaled like any
                # spoken text before the error, so the journal and the Flex timeline match the call.
                spoken = strip_silent("".join(text_parts))
                if dropped and spoken.endswith(dropped):
                    spoken = spoken[:-len(dropped)]
                if spoken.strip():
                    await self._append(call_sid, {"type": "ai_turn",
                                                  "text": await self._turn_text(call_sid, spoken,
                                                                                view.language, 4000)})
                retry = isinstance(exc, AgentError) and exc.retryable and failures <= self._cfg.retries
                log_caught(log, "agent turn failed", exc, operation="agent_turn", attempt=failures,
                           decision="retry" if retry else "escalate",
                           service_message=getattr(exc, "service_message", None))
                await self._append(call_sid, {"type": "ai_turn", "outcome": "error",
                                              "error": exc.code if isinstance(exc, AgentError) else "token_guard"})
                if failures == 1:
                    self._mark(call_sid, "error.generic")
                    yield self._phrase("error.generic", view.language)
                    self._on_answer(tm, stats)
                if retry:
                    continue  # same hop again, once
                stats.outcome = "escalated"
                await self._escalate_on_error(context, view)
                return
            rest = reasoning.flush()  # the hop's message ends: an unclosed block is dropped
            text_parts.append(rest)
            stats.silenced = stats.silenced or SILENT_MARK in rest
            tail = silent.feed(rest) + silent.flush()
            if tail:
                stats.answering = True
                stats.spoke = stats.spoke or not tail.isspace()
                released = True
                hop_first = time.monotonic() if hop_first is None else hop_first
                hop_chars += len(tail)
                yield tail
                self._on_answer(tm, stats)
            text = strip_silent("".join(text_parts))
            if text.strip():
                await self._append(call_sid, {"type": "ai_turn",
                                              "text": await self._turn_text(call_sid, text,
                                                                            view.language, 4000),
                                              **({"model": hop_model} if hop_model else {})})
            if not calls:
                stats.completed = True
                return
            stats.tools.extend(c.name for c in calls)
            last_tool = calls[0].name
            mask: _Mask | None = None
            inputs: dict[str, Mapping[str, Any]] = {}
            for c in calls:
                inputs.setdefault(c.name, c.input)
            planner.saw_tool_call = True  # A tool call, even one that gets no phrase
            due = self._schedule_tool_phrase(tm, view, stats,
                                             [(n, inputs[n]) for n in self._maskable(view, calls, context)])
            if due is not None:
                # /.1a: a phrase due now (delay 0, or a hand-off) goes on
                # the wire, then the tools start in the same step (no await in between); its
                # prompt event is journaled by _handle_calls after the tools started.
                self._log_mask(due.key, due.sched.tool, "tool", stats,
                               after_ms=due.sched.after_ms, source=due.sched.source)
                self._mark(call_sid, due.key)
                yield due.text
                self._rotation_set(call_sid, due.sched.tool, due.key)
                mask = _Mask(key=due.key, tool=due.sched.tool, text=due.text, at=time.monotonic())
            # Text spoken this turn before its tool call (stream mode only) holds a hand-off.
            work = self._handle_calls(context, view, calls, mask=mask, planner=planner, released=released,
                                      hop_text=None if hop_first is None else (hop_first, hop_chars))
            if not tm.deadlines.armed("tool"):
                hop = await work
                continue
            # A scheduled tool phrase may fall due while the tools run.
            hop = None
            async with aclosing(_race_deadlines(_one(work), tm.deadlines)) as racing:
                async for ev in racing:
                    if isinstance(ev, _Fired):
                        async for text in self._on_fired(ev.name, tm, view, stats):
                            yield text
                    else:
                        hop = ev
        if hop is not None:
            log.warning("tool loop hit max_hops; ending the turn", extra={
                "max_hops": self._cfg.max_hops, "decision": "end_turn"})
        stats.completed = True

    # ------------------------------------------------------------------ masking

    async def _on_fired(self, name: str, tm: _TurnMask, view: CallView,
                        stats: _TurnStats) -> AsyncGenerator[str, None]:
        """A deadline passed: the acknowledgement (``ack``) or the scheduled tool phrase (``tool``)."""
        call_sid = view.call_sid
        planner = tm.planner
        if name == "ack":
            key = planner.on_timer(planner.spoken_text)
            text = self._mask_text(key, view.language) if key else ""
            if key and text:
                self._log_mask(key, None, "timer", stats, after_ms=planner.after_ms, source="default")
                self._mark(call_sid, key)
                yield text
                planner.note_ack(text.strip(), time.monotonic())  # A tool phrase follows it
                # The next turn of this call uses the next one. Kept across the call's
                # reconnects; bounded (oldest calls dropped) so it never grows.
                self._rotation_set(call_sid, _GENERIC_SLOT, key)
                await self._append(call_sid, {"type": "prompt", "prompt_key": key,
                                              "rendered_text": text.strip()})
            return
        pending = tm.pending
        due = planner.on_deadline(time.monotonic())
        tm.pending = None
        if due is None or pending is None or pending.sched != due:
            return
        self._log_mask(pending.key, due.tool, "tool", stats, after_ms=due.after_ms, source=due.source)
        self._mark(call_sid, pending.key)
        yield pending.text
        await self._tool_phrase_spoken(call_sid, pending)

    async def _tool_phrase_spoken(self, call_sid: str, p: _PendingPhrase) -> None:
        """After a scheduled tool phrase went on the wire: rotation and the journal."""
        self._rotation_set(call_sid, p.sched.tool, p.key)
        await self._append(call_sid, {"type": "prompt", "prompt_key": p.key,
                                      "rendered_text": p.text.strip()})

    def _on_answer(self, tm: _TurnMask, stats: _TurnStats) -> None:
        """The turn sent answer text: a pending tool phrase is cancelled."""
        tm.deadlines.cancel("tool")
        tm.pending = None
        for p in tm.planner.on_text():
            self._log_skip(p.tool, p.phrase_key, SKIP_ANSWER_STARTED, stats, p)

    def _schedule_tool_phrase(self, tm: _TurnMask, view: CallView, stats: _TurnStats,
                              calls: Sequence[tuple[str, Mapping[str, Any]]]) -> _PendingPhrase | None:
        """Tool calls (or a Gateway tool start) reached the bridge. The phrase is
        rendered now; a phrase due at once is returned for the caller to yield, a later one is
        armed as the ``tool`` deadline. Each tool that gets no phrase is logged (``mask_skipped``)."""
        planner = tm.planner
        now = time.monotonic()
        scheduled = planner.on_tool_calls([n for n, _ in calls], now)
        for tool, reason in planner.last_skips:
            mask = planner.mask_of(tool)
            self._log_skip(tool, mask.phrase_key if mask else None, reason, stats)
        if not scheduled:
            return None
        sched = scheduled[0]
        if tm.pending is not None:  # superseded by a hand-off (logged above)
            tm.pending = None
            tm.deadlines.cancel("tool")
        args = next((a for n, a in calls if n == sched.tool), {})
        rendered = self._render_tool_phrase(view.call_sid, sched.tool, sched.phrase_key, args,
                                            view.language)
        if rendered is None:
            planner.discard(sched)
            self._log_skip(sched.tool, sched.phrase_key, SKIP_NO_RENDERABLE, stats, sched)
            return None
        pending = _PendingPhrase(sched, *rendered)
        if sched.deadline <= now:
            return pending if planner.on_deadline(sched.deadline) is not None else None
        tm.pending = pending
        tm.deadlines.arm("tool", sched.deadline)
        return None

    def _render_tool_phrase(self, call_sid: str, tool: str, phrase_key: str, args: Any,
                            lang: str) -> tuple[str, str] | None:
        """``(pool key, text + " ")`` for the tool's next phrase in rotation that renders in full
        for these arguments, or ``None`` with an ERROR naming the key.
        Never the generic acknowledgement as a substitute."""
        keys = pool_keys(phrase_key, self._catalog.keys)
        order = rotation_order(keys, self._rotation_get(call_sid, tool))
        reasons = self._policy.reasons
        words = words_for(tool, self._catalog, lang, reasons=reasons)
        for key in order:
            try:
                template = self._catalog.template(key, lang).strip()
            except CatalogKeyError:
                continue
            if placeholder_names(template) and not self._catalog.has(key, lang):
                continue  # Words only in the call's language, so no other language's template
            text = render(tool, template, args, words, reasons=reasons)
            if text and text.strip():
                return key, text.strip() + " "
        emit(log, logging.ERROR, "mask_phrase_unrenderable", key=phrase_key[:128], tool=tool,
             language=lang, pool=[k[:128] for k in order], decision="no_phrase")
        return None

    @staticmethod
    def _log_skip(tool: str, key: str | None, reason: str, stats: _TurnStats,
                  sched: ScheduledPhrase | None = None) -> None:
        """A tool phrase not spoken, and why."""
        fields: dict[str, Any] = {"tool": tool, "key": key, "reason": reason, "turn": stats.number,
                                  "ms_since_turn_start": round((time.monotonic() - stats.started) * 1000, 1)}
        if sched is not None:
            fields.update(after_ms=sched.after_ms, source=sched.source)
        emit(log, logging.INFO, "mask_skipped", **fields)
        if stats.tracing is not None:  # The same identifiers, never the phrase text
            stats.tracing.event(stats.span, "mask_skipped", {f"owlbank.{k}": v for k, v in fields.items()})

    def _rotation_get(self, call_sid: str, slot: str) -> str | None:
        return self._mask_rotation.get(call_sid, {}).get(slot)

    def _rotation_set(self, call_sid: str, slot: str, key: str) -> None:
        """Remember the last phrase of a pool (``slot``: a tool, or the acknowledgements) for this
        call. Bounded: the oldest calls are dropped."""
        per_call = self._mask_rotation.pop(call_sid, {})
        per_call[slot] = key
        self._mask_rotation[call_sid] = per_call
        while len(self._mask_rotation) > MASK_ROTATION_MAX_CALLS:
            self._mask_rotation.pop(next(iter(self._mask_rotation)))

    # ------------------------------------------------------------------ inputs

    async def _first_hop(self, context: Any, view: CallView, message: str | None) -> _Hop | None:
        if view.pending_tool and view.pending_result is not None:
            return await self._claim_result(view, message)
        if view.pending_tool is None and view.pending_result is not None:
            return await self._claim_utterance(view, message)
        if view.pending_tool and view.pending_result is None:
            if message is None:
                return None
            # A handoff was requested but the session didn't end (e.g. barge-in
            # cancelled the stream before TAC sent `end`): send it again.
            pt = view.pending_tool
            rebuilt = _handoff_for_pending(
                view, step_up=self._front_door_tool(pt.name)
                and view.level < self._policy.min_level(pt.name))
            if rebuilt is not None:
                log.info("re-sending a handoff that did not complete", extra={"decision": "resend_handoff"})
                if view.pending_tool.name == "escalate" and \
                        rebuilt["journal"] != (view.escalation or {}).get("journal_ref"):
                    await self._io(self._calls.patch, view.call_sid, {
                        F_ESCALATION: {**(view.escalation or {}), "journal_ref": rebuilt["journal"]}})
                rebuilt.update(_context_ids(context))
                context.pending_handoff_data = self._handoff_factory(_clean_handoff(view.call_sid, rebuilt))
                self._note_handoff(view.call_sid, pt.name)
                return None
            await self._io(self._calls.patch, view.call_sid, None, [F_PENDING_TOOL])
            return _Hop(tool_results=[ToolResult(view.pending_tool.tool_use_id, {
                "error": "HANDOFF_NOT_COMPLETED", "tool": view.pending_tool.name,
                "next": "handle_utterance", "callerUtterance": message}, "error")], timer=True)
        if message is None:
            return None
        return _Hop(user_text=message, timer=True)

    async def _claim_utterance(self, view: CallView, message: str | None) -> _Hop | None:
        """An envelope with no pending tool: the caller switched intent before the AI ever
        ran (login ``interrupted``). There is no tool call to complete, so the
        caller's (redacted) words become the first user turn."""
        if message is None and not self._is_immediate(view):
            return None
        env = dict(view.pending_result or {})
        try:
            await self._io(self._calls.patch, view.call_sid, None, [F_PENDING_RESULT, F_INTERRUPTED])
        except Exception as exc:
            log_caught(log, "could not clear pending_result; not resuming", exc, level=logging.ERROR,
                       operation="claim_pending_result", decision="skip_resume")
            return None
        words = message or ""
        if env.get("next") == "handle_utterance":
            words = f"{view.interrupted_utterance or env.get('callerUtterance') or ''} {words}".strip()
        return _Hop(user_text=words, timer=True, step=_envelope_step(env), utterance=_utterance(view, env)) \
            if words else None

    async def _claim_result(self, view: CallView, message: str | None) -> _Hop | None:
        env = dict(view.pending_result or {})
        if view.pending_tool is None:
            raise RuntimeError("claim_result without a pending tool")
        if message is None and not self._is_immediate(view):
            return None
        # Claim first, so each result is sent exactly once.
        try:
            await self._io(self._calls.patch, view.call_sid, None,
                           [F_PENDING_RESULT, F_PENDING_TOOL, F_INTERRUPTED])
        except Exception as exc:
            log_caught(log, "could not clear pending_result; not resuming", exc, level=logging.ERROR,
                       operation="claim_pending_result", decision="skip_resume")
            return None
        words = message or ""
        utterance = _utterance(view, env)
        if env.get("next") == "handle_utterance":
            words = f"{view.interrupted_utterance or env.get('callerUtterance') or ''} {words}".strip()
        env.pop("callerUtterance", None)
        if words:
            # The caller's words (a barge-in, or a prompt that came with the result) reach
            # the agent as a real caller turn: the result with ``next`` = ``await_caller`` is
            # recorded, and the model answers the words, in one turn. Inside a tool
            # result the model misread them ("I haven't made any payment").
            env["next"] = "await_caller"
        elif env.get("next") == "handle_utterance":
            env["callerUtterance"] = ""  # no words at all: as before
        results: list[ToolResult] = []
        status: Literal["success", "error"] = "error" if env.get("status") == "failed" else "success"
        results.append(ToolResult(view.pending_tool.tool_use_id, env, status))
        await self._append(view.call_sid, {
            "type": "tool_result", "tool": view.pending_tool.name,
            "tool_use_id": view.pending_tool.tool_use_id, "outcome": env.get("status"),
            "next": env.get("next")})
        # Only the caller's words arm the timer. A resume right after the system's
        # own scripted line ("Thanks, you're verified.") does not: an acknowledgement there sounds
        # like the bridge answering itself.
        step = view.pending_tool.name
        if words.strip():
            return _Hop(user_text=words, tool_results=results, timer=True, step=step, utterance=utterance)
        # Blank words are no words for the agent: the result is only recorded.
        return _Hop(tool_results=results, timer=bool(words), record_only=env.get("next") == "await_caller",
                    step=step, utterance=utterance)

    # ------------------------------------------------------------------ system prompt

    async def _system_prompt(self, context: Any, view: CallView) -> tuple[dict[str, str], str, str]:
        """``(static by model key, per_call, clock_line)``: each model
        key's static part (:func:`~owlbank.agent_prompt.render_static`, :meth:`_per_model`),
        the same for every caller, which the client puts before the system cache point; the
        per-call part (caller context, web block), rendered once for every model key; and
        the call's clock line (:meth:`_clock_line`). The turn sends ``per_call + "\\n" + clock`` as
        the suffix, after the cache point. The data is fetched once; both parts are cached per
        call under one key."""
        meta = context.metadata.setdefault(META_KEY, {})
        # The block follows this session's ``welcome_offer``, and TAC keeps the context
        # (and so this cache) across the call's sessions. The call SID keeps a later call on a reused
        # conversation from taking this call's per-call part (its Memory read predates the recap).
        key = [view.call_sid, view.verified_customer_id if view.verified else None, view.language,
               view.welcome_offer]
        if meta.get("prompt_key") == key and meta.get("system_prompt") and isinstance(meta.get("per_call"), str):
            clock = self._clock_line(context, view.call_sid, view.language)
            return meta["system_prompt"], meta["per_call"], clock

        caller: agent_prompt.CallerFacts | None = None
        web_activity = False
        if view.verified:
            customer = await self._io(self._store.get, "customers",
                                      {"customer_id": view.verified_customer_id})
            first_name, profile = "", ""  # a missing record reads as the empty values
            if customer:
                first_name = str(customer.get("first_name") or "")
                profile = _profile_text(customer, view)
            memory_section = await self._load_memory(context, view, customer)
            if memory_section:
                memory_section = untrusted.wrap("memory", memory_section)
                web_activity = _has_web_activity(context)
            caller = agent_prompt.CallerFacts(first_name, profile, memory_section or None)
        else:
            # Memory gating: never bind or load a profile before verification.
            context.profile_id = None
            context.profile = None
            meta.pop("verified_profile_id", None)
        # A block only when the filter kept web activity, else today's prompt.
        # WebActivity stays in the recall either way; the block says whether the welcome already
        # asked (``agent.web_activity``) or no offer was made (``..._context``). A block missing
        # from the catalog is left out by the renderer (memory kept, ``web_activity_block_missing``).
        web_block = None
        if web_activity:
            web_block = agent_prompt.WEB_ACTIVITY_KEY if view.welcome_offer == WELCOME_OFFER_MADE \
                else agent_prompt.WEB_ACTIVITY_CONTEXT_KEY
        if view.verified and self.tracing is not None:
            self.tracing.set(self._turn_spans.get(view.call_sid), {"owlbank.trait_count": _trait_count(context)})
        # Agent prompt caching: the static part per model key (no caller values at all)
        # and one per-call part, which takes no model key.
        lang = view.language
        statics = self._per_model(lang, lambda model_key: agent_prompt.render_static(self._catalog, model_key, lang))
        per_call = agent_prompt.render_per_call(self._catalog, lang, caller=caller, web_block=web_block)
        meta["prompt_key"], meta["system_prompt"], meta["per_call"] = key, statics, per_call
        _log_rendered(statics, per_call, lang, verified=caller is not None, web_block=web_block)
        return statics, per_call, self._clock_line(context, view.call_sid, lang)

    def _model_keys(self) -> tuple[str, ...]:
        """The distinct model keys of the model list, preferred first."""
        keys = [agent_prompt.model_key(m) for m in self._cfg.model_ids] or [agent_prompt.DEFAULT_MODEL_KEY]
        return tuple(dict.fromkeys(keys))

    def _per_model(self, lang: str, build: Callable[[str], str]) -> dict[str, str]:
        """``{model key: build(model key)}`` for :meth:`_model_keys`, in list order; ``build`` runs
        once per key. A key whose catalog text is missing (a bridge deployed before the
        reseed) gets the preferred key's prompt, with ``model_prompt_missing`` (WARNING, once per
        process per key and language); the preferred key missing raises ``CatalogKeyError``."""
        preferred, *others = self._model_keys()
        prompts = {preferred: build(preferred)}
        for model_key in others:
            try:
                prompts[model_key] = build(model_key)
            except CatalogKeyError:
                prompts[model_key] = prompts[preferred]
                if (model_key, lang) not in _PROMPT_MISSING_LOGGED:
                    _PROMPT_MISSING_LOGGED.add((model_key, lang))
                    emit(log, logging.WARNING, "model_prompt_missing", model_key=model_key, language=lang,
                         key=agent_prompt.system_key(model_key), decision="use_preferred_prompt",
                         preferred=preferred)
        return prompts

    def _clock_line(self, context: Any, call_sid: str, lang: str) -> str:
        """The bank's local date and time for the call. Set at the call's first turn, rebuilt
        only on a new bank-local day or a language change (it is not rebuilt every turn): the line sits before
        the history, so a line
        that moved every minute broke the messages cache point. Kept under ``clock`` in the TAC
        context's metadata with the call's SID, so a later call on the same context (a reused
        conversation, ``reuse_conversation``) builds its own. TAC keeps contexts per worker, in
        process: a re-entry on this worker keeps the line; one on another task or session
        builds a new line, which costs one cache miss. Accepted limit: the time of day drifts
        during a long call. Metadata that can't hold it (missing, ``None``, read-only; a fallback
        turn whose build failed on it) gets a fresh line, not kept."""
        tz = zone(self._cfg.timezone)
        now = datetime.fromtimestamp(self._now(), UTC)
        day = now.astimezone(tz).date().isoformat()  # the bank's date, not UTC's
        try:
            meta = context.metadata.setdefault(META_KEY, {})
            held = meta.get("clock")
            if isinstance(held, dict) and held.get("call_sid") == call_sid and held.get("day") == day \
                    and held.get("lang") == lang and isinstance(held.get("line"), str):
                return str(held["line"])
            line = local_now_line(now, tz, lang)
            meta["clock"] = {"call_sid": call_sid, "day": day, "lang": lang, "line": line}
            return line
        except (AttributeError, TypeError):
            return local_now_line(now, tz, lang)

    async def _load_memory(self, context: Any, view: CallView, customer: Mapping[str, Any] | None) -> str | None:
        context.profile_id = None
        context.profile = None  # REQUIRED: else TAC reuses the ANI profile's traits
        context.metadata.setdefault(META_KEY, {}).pop("verified_profile_id", None)
        if self._memory is None:
            return None
        identifier = (customer or {}).get(MEMORY_IDENTIFIER_FIELD)
        if not isinstance(identifier, str) or not identifier:
            return None  # no stable verified identifier: no memory, never the ANI
        created = False
        try:
            ids = await asyncio.wait_for(self._memory.lookup_profile_ids(identifier),
                                         self._cfg.memory_timeout_s)
        except Exception as exc:
            log_caught(log, "memory profile lookup failed; continuing without memory", exc,
                       operation="memory_lookup", decision="continue_without_memory")
            return None
        if not ids:
            # First call from this verified customer: create the profile so the next call finds it.
            # Best effort and bounded; a failure never blocks the call (no ANI fallback either).
            try:
                ids = [await asyncio.wait_for(self._memory.create_profile(identifier),
                                              self._cfg.memory_timeout_s)]
                created = True
            except Exception as exc:
                log_caught(log, "memory profile create failed; continuing without memory", exc,
                           operation="memory_create_profile", decision="continue_without_memory")
                return None
        context.profile_id = ids[0]
        context.profile = None
        context.metadata.setdefault(META_KEY, {})["verified_profile_id"] = ids[0]
        if created:
            return None  # a brand-new profile has nothing to recall
        try:
            return await asyncio.wait_for(self._memory.retrieve_prompt(context), self._cfg.memory_timeout_s)
        except Exception as exc:
            # TAC may have loaded the profile before the failure, unfiltered: drop it (fail closed).
            context.profile = None
            log_caught(log, "memory recall failed; continuing without memory", exc,
                       operation="memory_recall", decision="continue_without_memory")
            return None

    def _fallback_system_prompt(self, view: CallView) -> tuple[dict[str, str], str]:
        """``(static by model key, per_call)`` when :meth:`_system_prompt` failed: each model's
        static part (the fallback as there) and the fallback per-call part, with no caller data
        and no web block: :data:`agent_prompt.FALLBACK_CONTEXT` for
        every caller, which claims no verification either way. Not cached."""
        lang = view.language
        statics = self._per_model(lang, lambda model_key: agent_prompt.render_static(self._catalog, model_key, lang))
        per_call = agent_prompt.render_per_call(self._catalog, lang, caller=None, web_block=None, fallback=True)
        _log_rendered(statics, per_call, lang, verified=view.verified, web_block=None)
        return statics, per_call

    # ------------------------------------------------------------------ tools

    async def _handle_calls(self, context: Any, view: CallView, calls: list[ToolCall], *,
                            mask: _Mask | None = None,
                            planner: MaskPlanner | None = None,
                            released: bool = False,
                            hop_text: tuple[float, int] | None = None) -> _Hop | None:
        """Run one model message's tool calls.

        Every decision is made first, without awaiting, and the inline bank tools (and a masked
        hand-off's prefetch) are started at once as eager tasks, so they run concurrently and
        nothing waits for a masking phrase. Results and journal events stay in
        the model's call order. A hand-off holds ``end`` for its masking phrase (the step-up
        shortcut too, without a prefetch) or, with none, for the agent's
        text ``released`` before it this turn, by one rule
        (:func:`_handoff_hold_s`): until the latest of the session's speech estimate, this hop's
        released text ``hop_text`` (``(first chunk sent at, chars)``) and the phrase.
        """
        call_sid = view.call_sid
        entries: list[ToolResult | asyncio.Task[ToolResult] | None] = []  # None: the handoff
        tasks: list[asyncio.Task[ToolResult]] = []
        handoff: ToolCall | None = None
        step_up = False  # The hand-off is the step-up for ``handoff`` itself
        prefetch: asyncio.Task[ToolResult] | None = None
        try:
            for call in calls:
                is_handoff = self._is_handoff(call.name)
                entry: ToolResult | asyncio.Task[ToolResult] | None
                if is_handoff and len(calls) > 1:
                    # A handoff ends the session, so it must be the only call in its response:
                    # the other results can't be held across the deterministic flow.
                    entry = ToolResult(call.tool_use_id, {"error": "CALL_ALONE", "tool": call.name},
                                       "error")
                elif is_handoff and handoff is None:
                    entry = self._pre_handoff_check(view, call, context)
                    if entry is None:
                        handoff = call
                    elif self._step_up_shortcut(view, call, entry):
                        # Refused only for the caller's level: start the step-up now, for
                        # this call, instead of a model round trip (AUTH_LEVEL_INSUFFICIENT, then
                        # authenticate). The front door answers this tool_use_id.
                        handoff, step_up, entry = call, True, None
                elif is_handoff:
                    entry = ToolResult(call.tool_use_id, {"error": "ONE_HANDOFF_AT_A_TIME",
                                                          "tool": call.name}, "error")
                elif call.name in BANK_TOOLS:
                    entry = _eager_task(self._run_bank_tool(view, call))
                    tasks.append(entry)
                else:
                    entry = ToolResult(call.tool_use_id, {"error": "UNKNOWN_TOOL", "tool": call.name},
                                       "error")
                entries.append(entry)
            if handoff is not None and not step_up and mask is not None \
                    and mask.tool == handoff.name and handoff.name in PREFETCH_TOOLS:
                # Read-only work for the front door, started with the phrase. Never on a
                # step-up: the caller's token can't run the tool yet (level too low).
                prefetch = _eager_task(self._run_bank_tool(view, handoff, raw=True))
                tasks.append(prefetch)
            if mask is not None:
                await self._append(call_sid, {"type": "prompt", "prompt_key": mask.key,
                                              "rendered_text": mask.text.strip()})
            results: list[ToolResult] = []
            for call, entry in zip(calls, entries, strict=True):
                await self._append(call_sid, _tool_call_event(call_sid, call))
                if entry is None:
                    continue
                try:
                    r = await entry if isinstance(entry, asyncio.Task) else entry
                except BaseException as exc:  # a barge-in or hang-up cancels it: ``cancelled``
                    _log_tool_result(call.name, exception=exc)
                    raise
                _log_tool_result(call.name, r.content)
                results.append(r)
                await self._append(call_sid, {"type": "tool_result", "tool": call.name,
                                              "tool_use_id": call.tool_use_id, "outcome": r.status,
                                              "error": r.content.get("error") if isinstance(r.content, dict)
                                              else None})
            if handoff is not None:
                await self._start_handoff(context, view, handoff, step_up=step_up)
                if planner is None:
                    return None
                phrase = mask if mask is not None and mask.tool == handoff.name else None
                if phrase is not None or released:
                    # One hold for the hand-off phrase (also
                    # on the step-up shortcut, where ``prefetch`` is always None) and for
                    # the agent's text released this turn: the latest of the session's speech
                    # estimate, the current hop's released text and the phrase. The two sources
                    # never meet: the planner gives no phrase once the answer started.
                    now = time.monotonic()
                    estimates = _hold_estimates(planner, now, speech_end=self._speech_end(call_sid),
                                                hop_text=hop_text, mask=phrase)
                    hold, basis = _handoff_hold_s(planner, handoff.name, estimates)
                    source = "mask" if phrase is not None else "released_text"
                    if (await self._hold_handoff(view, handoff, hold,
                                                 prefetch if phrase is not None else None, planner,
                                                 source=source, basis=basis)
                            and prefetch in tasks):
                        tasks.remove(prefetch)  # handed to a background store: don't cancel it below
                return None
            return _Hop(tool_results=results)
        finally:
            for task in tasks:  # barge-in, an error, or a late prefetch: abandon what still runs
                if not task.done():
                    task.cancel()
                elif not task.cancelled():
                    with suppress(Exception):  # retrieved, so an unawaited failure is not reported twice
                        task.result()

    def _maskable(self, view: CallView, calls: Sequence[ToolCall], context: Any = None) -> list[str]:
        """The tool names of a message that will actually run or hand off: a
        hand-off that is refused (``CALL_ALONE``, arguments, a level refusal that is not the
        step-up shortcut) and a bank tool the caller's level can't use (it returns
        ``AUTH_LEVEL_INSUFFICIENT`` at once) get no phrase, since nothing would be fetched or
        changed. A tool the caller confirms in a deterministic flow (``make_payment``,
        ``transfer_funds``), refused only for the caller's level, hands off for step-up at once
        (:meth:`_step_up_shortcut`) and keeps its tool's phrase, which is worded to fit a step-up
        as well ("Let me set that up."): the caller hears it instead of silence before the step-up
        prompt. ``get_balance`` gets no phrase on
        a step-up: its phrases announce the balance ("Fetching your balance now."), which a failed
        verification would never deliver."""
        names: list[str] = []
        for call in calls:
            if self._is_handoff(call.name):
                if len(calls) > 1:
                    continue
                refusal = self._pre_handoff_check(view, call, context, log_refusal=False)
                if refusal is not None and not (self._needs_confirm(call.name)
                                                and self._step_up_shortcut(view, call, refusal)):
                    continue
            elif call.name not in BANK_TOOLS or view.level < self._policy.min_level(call.name):
                continue
            names.append(call.name)
        return names

    async def _hold_handoff(self, view: CallView, call: ToolCall, hold: float,
                            prefetch: asyncio.Task[ToolResult] | None, planner: MaskPlanner, *,
                            source: str, basis: str) -> bool:
        """Never wait for the work, only (in ``wait`` mode) for the words.

        ``hold`` (seconds from now, :func:`_handoff_hold_s`) is for what the caller hears last
        before the hand-off: after the hand-off's masking phrase (``source="mask"``) or the agent's
        text released this turn (``source="released_text"``); ``basis``
        names the estimate that set it. The prefetch (only with a phrase) gets at most that same hold: if it is done by then it is
        stored now; if not, it is stored from the background when it completes, and the front door
        uses it only if it lands first (else it fetches itself). Returns True when the prefetch was
        handed to that background store."""
        deadline = time.monotonic() + hold
        stored = detached = False
        if prefetch is not None:
            if not prefetch.done() and hold > 0:
                await asyncio.wait({prefetch}, timeout=hold)
            if prefetch.done():
                stored = await self._store_prefetch(view, call, prefetch)
            else:
                task = asyncio.create_task(self._store_prefetch_late(view, call, prefetch))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
                detached = True
        wait_s = max(0.0, deadline - time.monotonic())
        emit(log, logging.INFO, "handoff_hold", tool=call.name, wait_ms=round(hold * 1000, 1),
             end_mode=planner.end_mode, prefetched=stored, source=source, basis=basis)
        if wait_s > 0:
            await asyncio.sleep(wait_s)
        return detached

    async def _store_prefetch_late(self, view: CallView, call: ToolCall,
                                   prefetch: asyncio.Task[ToolResult]) -> None:
        """The prefetch outlived the hold: store it when it completes (best effort)."""
        try:
            await asyncio.wait_for(asyncio.shield(prefetch), self._cfg.backend_timeout_s)
        except Exception:  # noqa: BLE001 - logged below; the front door fetches as today
            emit(log, logging.WARNING, "prefetch_unused", tool=call.name, cause="late",
                 operation="prefetch", decision="front_door_fetches")
            return
        await self._store_prefetch(view, call, prefetch)

    async def _store_prefetch(self, view: CallView, call: ToolCall,
                              prefetch: asyncio.Task[ToolResult]) -> bool:
        """Write a successful prefetch to call state (``prefetched``); False if not usable."""
        try:
            res = prefetch.result()
        except Exception as exc:
            log_caught(log, "hand-off prefetch failed", exc, tool=call.name, operation="prefetch",
                       decision="front_door_fetches")
            return False
        content = res.content if isinstance(res.content, Mapping) else None
        if res.status != "success" or content is None or "error" in content:
            emit(log, logging.WARNING, "prefetch_unused", tool=call.name, cause="error",
                 error_code=str(content.get("error"))[:40] if content else None,
                 operation="prefetch", decision="front_door_fetches")
            return False
        record = {"tool_use_id": call.tool_use_id, "tool": call.name,
                  "args": {"account": call.input.get("account")},
                  "result": _no_floats(dict(content)), "at": int(self._now())}
        try:
            await self._io(self._calls.patch, view.call_sid, {F_PREFETCHED: record})
        except Exception as exc:
            log_caught(log, "could not store the hand-off prefetch", exc, tool=call.name,
                       operation="store_prefetch", decision="front_door_fetches")
            return False
        return True

    def _needs_confirm(self, name: str) -> bool:
        """A bank tool the caller must confirm in a deterministic flow."""
        return name in BANK_TOOLS and self._policy.confirm(name) == "deterministic"

    def _is_handoff(self, name: str) -> bool:
        return name in HANDOFF_TOOLS or self._needs_confirm(name)

    def _front_door_tool(self, name: str) -> bool:
        """A hand-off tool the front door runs itself (``get_balance``, or a deterministic confirm)."""
        return name == "get_balance" or self._needs_confirm(name)

    def _step_up_shortcut(self, view: CallView, call: ToolCall, refusal: ToolResult) -> bool:
        """``refusal`` (from :meth:`_pre_handoff_check`) is for the caller's level only, on a
        front-door tool with valid arguments: the bridge hands off for step-up directly. Never once
        the call's login is closed: that refusal goes back to the model."""
        content = refusal.content if isinstance(refusal.content, Mapping) else {}
        return (not view.login_closed and self._front_door_tool(call.name)
                and content.get("error") == AUTH_LEVEL_INSUFFICIENT
                and view.level < self._policy.min_level(call.name))

    def _pre_handoff_check(self, view: CallView, call: ToolCall, context: Any = None, *,
                           log_refusal: bool = True) -> ToolResult | None:
        """Level check before any handoff, and argument validation. ``context``: the TAC
        session, for a hand-off already set this session (``end_call``'s pending check).
        ``log_refusal``: false for the masking planner's look-ahead, so each refusal logs once.

        ``end_call``: refused with ``TOOL_DISABLED`` when the policy switch is off, and
        with ``CALL_IN_PROGRESS`` while a step is pending (:func:`_pending_step`). Once the login
        is closed, ``authenticate`` is refused with ``VERIFICATION_CLOSED``
        and a front-door tool's level refusal says ``"verification": "closed"``."""
        if call.name == "escalate":
            return None  # never refuse an escalation; reason/summary are normalized server-side
        if call.name == END_CALL:
            if not self._policy.end_call_enabled():
                if log_refusal:
                    emit(log, logging.INFO, "end_call_refused", cause="disabled")
                return ToolResult(call.tool_use_id, {"error": TOOL_DISABLED, "tool": END_CALL}, "error")
            pending = _pending_step(view, context)
            if pending is not None:
                if log_refusal:
                    emit(log, logging.INFO, "end_call_refused", cause="pending", pending=pending)
                return ToolResult(call.tool_use_id, {"error": CALL_IN_PROGRESS, "tool": END_CALL,
                                                     "pending": pending}, "error")
            return None
        if call.name == "authenticate" and view.login_closed:
            if log_refusal:
                emit(log, logging.INFO, "verification_closed", tool=call.name)
            return ToolResult(call.tool_use_id, {"error": VERIFICATION_CLOSED, "tool": "authenticate"},
                              "error")
        errors = validate_args(call.name, call.input, reasons=self._policy.reasons,
                               cards=card_values(self._policy))
        if errors:
            return ToolResult(call.tool_use_id, {"error": "INVALID_ARGUMENTS", "tool": call.name,
                                                 "details": errors}, "error")
        if self._front_door_tool(call.name) and view.level < self._policy.min_level(call.name):
            # Once closed, no step-up shortcut; the model offers what it can
            return ToolResult(call.tool_use_id, _level_refusal(view, call.name, log_refusal), "error")
        return None

    async def _run_bank_tool(self, view: CallView, call: ToolCall, *, raw: bool = False) -> ToolResult:
        """One bank call with the call's token. ``raw``: the scrubbed result as the bank returned
        it, without the untrusted-data wrapping meant for the model (the hand-off prefetch)."""
        errors = validate_args(call.name, call.input, reasons=self._policy.reasons,
                               cards=card_values(self._policy))
        if errors:
            return ToolResult(call.tool_use_id, {"error": "INVALID_ARGUMENTS", "tool": call.name,
                                                 "details": errors}, "error")
        # Reload the call state (step-up replaces sealed_token) and unseal in memory,
        # for this tool call only. Missing or unsealable -> level 0 -> AUTH_LEVEL_INSUFFICIENT.
        try:
            fresh = await self._io(self._calls.load, view.call_sid)
        except Exception as exc:
            log_caught(log, "call state reload failed before a bank tool", exc, tool=call.name,
                       operation="load_call_state", decision="treat_as_level_0")
            fresh = None
        sealed = fresh.sealed_token if fresh is not None else None
        token: str | None = None
        if sealed:
            try:
                token = await self._io(self._sealer.unseal, sealed, call_sid=view.call_sid)
            except UnsealError as exc:
                log_caught(log, "sealed token did not unseal for this call", exc, tool=call.name,
                           operation="unseal_token", decision="treat_as_level_0")
            except Exception as exc:
                log_caught(log, "token unseal failed", exc, tool=call.name,
                           operation="unseal_token", decision="treat_as_level_0")
        if not token:
            # Same answer the tool gives without a token; the API still enforces it.
            return ToolResult(call.tool_use_id, _level_refusal(fresh or view, call.name, not raw), "error")
        idem = call.tool_use_id if call.name in IDEMPOTENT_OPS else None
        attempts = 2 if idem else 1  # a retry is safe only with the same idempotency key
        result = None
        for attempt in range(attempts):
            try:
                kw = {"idempotency_key": idem} if idem else {}
                result = await asyncio.wait_for(
                    self._backend.call(call.name, token=token, call_sid=view.call_sid,
                                       args=call.input, **kw),
                    self._cfg.backend_timeout_s)
                break
            except Exception as exc:
                log_caught(log, "bank_tools call failed", exc, tool=call.name, attempt=attempt + 1,
                           operation="bank_tools",
                           decision="retry" if attempt + 1 < attempts else "return_unavailable")
        if result is None:
            return ToolResult(call.tool_use_id, {"error": "UNAVAILABLE", "tool": call.name}, "error")
        result = _scrub_result(token, call.name, result, sealed=sealed)
        del token
        if not raw:
            result = untrusted.wrap_result(result)  # Descriptions, nicknames, memos
        if result.get("error") == AUTH_LEVEL_INSUFFICIENT:
            # nothing beyond code + tool, and the closed-login marker (b)
            result = _level_refusal(fresh or view, call.name, not raw)
        return ToolResult(call.tool_use_id, result, "error" if "error" in result else "success")

    async def _start_handoff(self, context: Any, view: CallView, call: ToolCall, *,
                             step_up: bool = False) -> None:
        call_sid = view.call_sid
        updates: dict[str, Any] = {}
        ref = new_journal_ref()  # opaque; for an escalation it is also the Flex Context API key
        if call.name == END_CALL:
            # The silence hang-up's shape (:meth:`_silence_action`). No pending
            # tool (the front door answers nothing) and no escalation: it says ``common.goodbye``
            # and hangs up.
            await self._append(call_sid, {"type": "handoff", "tool": END_CALL, "reason": CALLER_DONE,
                                          "outcome": "requested"})
            emit(log, logging.INFO, "end_call_requested", level=view.level)
            data_end: dict[str, Any] = {"reason": CALLER_DONE, "tool": END_CALL, "journal": ref,
                                        **_context_ids(context)}
            context.pending_handoff_data = self._handoff_factory(_clean_handoff(call_sid, data_end))
            self._note_handoff(call_sid, END_CALL)
            return
        if step_up:
            # The same hand-off ``authenticate`` produces, but ``pending_tool`` is the model's
            # own call (validated arguments only), so its tool_use_id gets the final result.
            data: dict[str, Any] = {"reason": "step-up", "tool": call.name, "args": {"tool": call.name}}
            updates[F_STEP_UP] = call.name
            args: dict[str, Any] = _tool_args(call)
            emit(log, logging.INFO, "step_up_shortcut", tool=call.name, level=view.level,
                 min_level=self._policy.min_level(call.name))
        elif call.name == "authenticate":
            target = str(call.input.get("tool"))
            data = {"reason": "step-up", "tool": target, "args": {"tool": target}}
            updates[F_STEP_UP] = target
            args = {"tool": target}
        elif call.name == "get_balance":
            args = {"account": call.input.get("account")}
            data = {"reason": "tool", "tool": "get_balance", "args": args}
        elif self._needs_confirm(call.name):
            # Validated (``_pre_handoff_check``): only the tool's own parameters. They stay in
            # call state; handoffData (Twilio-visible) gets no amounts or accounts.
            args = _tool_args(call)
            data = {"reason": "confirm", "tool": call.name}
        else:  # escalate
            reason, known = normalize_reason(call.input.get("reason"), self._policy.reasons)
            if not known:
                log.warning("unknown escalation reason from the model", extra={
                    "reason": reason, "decision": "fallback_reason"})
            emit(log, logging.INFO, "escalation_requested", reason=reason, source="agent")
            summary = await self._summary_text(call_sid, call.input.get("summary"), view.language)
            args = {"reason": reason}
            data = {"reason": "escalate", "tool": "escalate", "escalationReason": reason}
            updates[F_ESCALATION] = {"reason": reason, "summary": summary, "at": int(self._now()),
                                     "source": "agent", "journal_ref": ref}
            await self._append(call_sid, {"type": "escalation", "reason": reason, "summary": summary})
        await self._append(call_sid, {"type": "handoff", "tool": call.name, "reason": data["reason"],
                                      "tool_use_id": call.tool_use_id, "outcome": "requested"})
        data["journal"] = ref
        data.update(_context_ids(context))
        handoff_json = _clean_handoff(call_sid, data)
        updates[F_PENDING_TOOL] = {"tool_use_id": call.tool_use_id, "name": call.name, "args": args}
        try:
            await self._io(self._calls.patch, call_sid, updates)
        except Exception as exc:
            log_caught(log, "could not store the pending tool; handing off anyway", exc,
                       level=logging.ERROR, tool=call.name, operation="store_pending_tool",
                       decision="handoff_anyway")
        context.pending_handoff_data = self._handoff_factory(handoff_json)
        self._note_handoff(call_sid, call.name)

    async def _escalate_on_error(self, context: Any, view: CallView) -> None:
        reason = FALLBACK_REASON
        ref = new_journal_ref()
        emit(log, logging.INFO, "escalation_requested", reason=reason, source="bridge")
        await self._append(view.call_sid, {"type": "escalation", "reason": reason,
                                           "summary": "The AI agent was unavailable."})
        data = {"reason": "escalate", "tool": "escalate", "escalationReason": reason,
                "journal": ref, **_context_ids(context)}
        handoff_json = _clean_handoff(view.call_sid, data)
        try:
            await self._io(self._calls.patch, view.call_sid, {
                F_ESCALATION: {"reason": reason, "summary": "The AI agent was unavailable.",
                               "at": int(self._now()), "source": "bridge", "journal_ref": ref}})
        except Exception as exc:
            log_caught(log, "could not store the escalation; handing off anyway", exc,
                       level=logging.ERROR, operation="store_escalation", decision="handoff_anyway")
        context.pending_handoff_data = self._handoff_factory(handoff_json)
        self._note_handoff(view.call_sid, "escalate")

    def _set_handoff(self, context: Any, call_sid: str, data: dict[str, Any]) -> None:
        ref = new_journal_ref()
        data = {**data, "journal": ref, **_context_ids(context)}
        if data.get("reason") == "escalate":
            try:  # best effort: there may be no call state at all
                self._calls.patch(call_sid, {F_ESCALATION: {
                    "reason": data.get("escalationReason") or FALLBACK_REASON,
                    "at": int(self._now()), "source": "bridge", "journal_ref": ref}})
            except Exception as exc:
                log_caught(log, "could not store the escalation ref; handing off anyway", exc,
                           operation="store_escalation", decision="ignored_because_best_effort")
        context.pending_handoff_data = self._handoff_factory(_clean_handoff(call_sid, data))
        self._note_handoff(call_sid, data.get("tool") or data.get("reason"))

    # ------------------------------------------------------------------ silence

    def _mark(self, call_sid: str, key: str) -> None:
        """The next chunk yielded for this call is the scripted catalog phrase ``key``."""
        self._scripted[call_sid] = key

    def _note_out(self, call_sid: str, chunk: str) -> None:
        """A chunk is going to ConversationRelay: the speaking estimate, the session's last message
        (agent chunks in a row join one message; a scripted phrase is a message of its own) and
        what is playing (``relay_interrupt``)."""
        label = self._scripted.pop(call_sid, None)
        speech = self._speech.get(call_sid)
        if speech is None or not chunk:
            return
        now = time.monotonic()
        # A scripted phrase's trailing space only separates it from what follows (it is not
        # spoken), so the estimate counts the phrase as the hand-off hold does.
        speech.watch.note_speech(chunk if label is None else chunk.rstrip(), now)
        if label is None and speech.open and speech.text is not None:
            speech.text += chunk
        else:
            speech.text, speech.at, speech.label = chunk, now, label or PLAYING_AGENT
            speech.open = label is None
        speech.playing = speech.label
        speech.turn_spoke = True

    def _speech_end(self, call_sid: str) -> float:
        """When all speech sent to this session so far is estimated to end (``time.monotonic()``
        scale): the silence watch's estimate (fed by :meth:`_note_out`), cut short by a
        reported speech end or a barge-in. No session: ``-inf`` (nothing is known to be playing)."""
        speech = self._speech.get(call_sid)
        return speech.watch.playing_until if speech is not None else float("-inf")

    def now_playing(self, call_sid: str | None) -> str:
        """What the session sent last (a catalog key, ``agent``, or ``greeting`` before anything),
        for ``relay_interrupt``. Never text."""
        speech = self._speech.get(call_sid or "")
        return speech.playing if speech is not None else "unknown"

    def on_inbound(self, kind: str, call_sid: str | None) -> None:
        """A ``prompt``, ``dtmf`` or ``interrupt`` frame arrived (``PeekingWebSocket``, before TAC):
        caller input, so the silence count resets and the pending expiry is cancelled.
        If no turn starts within ``silence_timeout_ms`` the last message is repeated, counting from
        this input (a key press or a barge-in that no prompt follows). No content. A ``prompt`` or
        ``interrupt`` also stops keypad work (:meth:`_keypad_barge`).

        An ``interrupt`` within ``REPEAT_GRACE_MS`` of a repeat being sent means the
        caller was already speaking: that repeat doesn't count (``prompt_repeat_collided``, no
        text); the reset and re-arm are as for any input."""
        speech = self._speech.get(call_sid or "")
        if speech is None or call_sid is None:
            return
        if kind in ("prompt", "interrupt"):
            with obs.call_context(call_sid, component="bridge"):
                self._keypad_barge(call_sid, speech, kind)
        watch = speech.watch
        now = time.monotonic()
        repeats, repeat_at = watch.repeats, watch.repeat_at
        collided = watch.on_inbound(now, str(kind))
        self._silence_reschedule(call_sid)
        counted = repeats - (1 if collided else 0)
        with obs.call_context(call_sid, component="bridge"):
            if collided and repeat_at is not None:
                emit(log, logging.INFO, "prompt_repeat_collided", repeat=repeats, what=speech.label,
                     ms_after_repeat=round((now - repeat_at) * 1000, 1), grace_ms=watch.repeat_grace_ms,
                     cause="interrupt")
            if counted:
                emit(log, logging.INFO, "silence_reset", kind=str(kind)[:16], repeats=counted)

    def on_speaker_event(self, call_sid: str | None, who: str, speaking: bool) -> None:
        """ConversationRelay reported that ``who`` (``caller`` or ``agent``) started or
        stopped speaking (``PeekingWebSocket``, before TAC; never caller input to TAC). The caller
        speaking pauses the silence wait and stopping restarts it, the count unchanged; the
        agent's speech end starts the wait. The session's first one logs ``silence_timing``
        ``speaker_events``. A ``clientSpeaking on`` within ``REPEAT_GRACE_MS`` of a repeat is a
        collision, like an ``interrupt`` (``prompt_repeat_collided``). No content."""
        speech = self._speech.get(call_sid or "")
        if speech is None or call_sid is None:
            return
        watch = speech.watch
        now = time.monotonic()
        first = not watch.speaker_events
        repeats, repeat_at = watch.repeats, watch.repeat_at
        collided = watch.on_speaker(who, speaking, now)
        self._silence_reschedule(call_sid)
        turn = speech.turn
        with obs.call_context(call_sid, component="bridge"):
            if first:
                emit(log, logging.INFO, "silence_timing", mode=TIMING_SPEAKER_EVENTS)
            if (who == AGENT and speaking and turn is not None and turn.first_out is not None
                    and not turn.speech_logged):
                # Text-to-speech's part of the caller's wait (wait_ms is
                # the bridge's): from the turn's first chunk out (``first_out``: a masking phrase
                # or the apology, if one went first) to the agent's speech starting. Not logged
                # past the cap: then this speech is not that chunk's.
                turn.speech_logged = True
                after_ms = (now - turn.first_out) * 1000
                if after_ms <= AGENT_SPEECH_START_MAX_MS:
                    emit(log, logging.INFO, "agent_speech_started", turn=turn.number,
                         ms_after_first_text=round(after_ms, 1))
            if collided and repeat_at is not None:
                emit(log, logging.INFO, "prompt_repeat_collided", repeat=repeats, what=speech.label,
                     ms_after_repeat=round((now - repeat_at) * 1000, 1), grace_ms=watch.repeat_grace_ms,
                     cause="client_speaking")

    def _silence_configure(self, watch: SilenceWatch) -> None:
        """The policy's current silence values, through its cache (lenient: bad fields were
        defaulted with an ERROR by the loader)."""
        try:
            cfg = self._policy.masking
            watch.configure(timeout_ms=cfg.silence_timeout_ms, max_repeats=cfg.silence_max_repeats,
                            action=cfg.silence_action, chars_per_s=float(cfg.chars_per_s))
        except Exception as exc:  # noqa: BLE001 - keep the watch's current values
            log_caught(log, "silence settings unavailable", exc, operation="silence_settings",
                       decision="keep_current_settings")

    def _silence_drop(self, call_sid: str) -> None:
        speech = self._speech.pop(call_sid, None)
        self._scripted.pop(call_sid, None)
        if speech is not None and speech.handle is not None:
            speech.handle.cancel()
            speech.handle = None

    def _silence_reschedule(self, call_sid: str) -> None:
        """One wake-up per session, at the watch's deadline (none when disarmed)."""
        speech = self._speech.get(call_sid)
        if speech is None:
            return
        if speech.handle is not None:
            speech.handle.cancel()
            speech.handle = None
        deadline = speech.watch.wake_at  # never before speech is estimated to end (no busy wake-ups)
        if deadline is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        delay = max(0.005, deadline - time.monotonic())
        speech.handle = loop.call_later(delay, self._silence_due, call_sid, speech)

    def _silence_due(self, call_sid: str, speech: _CallSpeech) -> None:
        speech.handle = None
        if self._speech.get(call_sid) is not speech:
            return
        task = asyncio.create_task(self._silence_fire(call_sid, speech))
        self._silence_tasks.add(task)
        task.add_done_callback(self._silence_tasks.discard)

    async def _silence_fire(self, call_sid: str, speech: _CallSpeech) -> None:
        with obs.call_context(call_sid, component="bridge"):
            try:
                await self._silence_expired(call_sid, speech)
            except Exception as exc:  # noqa: BLE001 - silence handling must never break the call
                log_caught(log, "silence handling failed", exc, level=logging.ERROR,
                           operation="silence_expired", decision="ignored")

    async def _silence_expired(self, call_sid: str, speech: _CallSpeech) -> None:
        watch = speech.watch
        waited_ms = watch.arm_timeout_ms  # the wait that just ran out (the greeting's own)
        exp = watch.poll(time.monotonic())
        if exp is None:
            if watch.turns == 0:
                self._silence_reschedule(call_sid)  # woke early, or the deadline moved
            return  # a running turn re-arms the watch when it ends
        epoch = watch.epoch
        lock = self._locks.setdefault(call_sid, asyncio.Lock())
        async with lock:
            context, _ = self._session_for_call(call_sid)  # The live session, not TAC's oldest
            usable = (context is not None and getattr(context, "pending_handoff_data", None) is None)
            if watch.epoch != epoch or self._speech.get(call_sid) is not speech or not usable:
                # The caller spoke, a turn started, the session ended, or a hand-off is under way.
                if exp.kind == "action":
                    watch.retract()
                if exp.kind == "repeat":
                    same = watch.epoch == epoch
                    watch.unrepeat()  # never sent: it doesn't count (count unchanged)
                    if same:
                        watch.disarm()
                emit(log, logging.INFO, "silence_skipped", kind=exp.kind,
                     cause="no_session" if not usable else "input_or_turn", decision="no_action")
                return
            if exp.kind == "repeat":
                await self._silence_repeat(call_sid, speech, context, exp, waited_ms)
            else:
                await self._silence_action(call_sid, speech, context, exp)
        self._silence_reschedule(call_sid)

    async def _silence_repeat(self, call_sid: str, speech: _CallSpeech, context: Any, exp: Expiry,
                              waited_ms: int | None = None) -> None:
        """3: the same text again, straight to ConversationRelay (like a masking phrase:
        scripted, never in the agent's history, not a turn). Logged without the text."""
        text = exp.text or ""
        fields: dict[str, Any] = {
            "repeat": exp.repeat, "max_repeats": speech.watch.max_repeats, "what": speech.label,
            "chars": len(text), "timeout_ms": speech.watch.timeout_ms,
            "waited_ms": speech.watch.timeout_ms if waited_ms is None else waited_ms}
        await self._repeat_last(call_sid, context, speech, text, trigger=REPEAT_SILENCE, fields=fields,
                                event={"repeat": exp.repeat})

    async def _repeat_last(self, call_sid: str, context: Any, speech: _CallSpeech, text: str, *,
                           trigger: str, fields: Mapping[str, Any], event: Mapping[str, Any]) -> None:
        """Send the session's last message ``text`` again, straight to ConversationRelay (never in
        the agent's history, not a turn), then log ``prompt_repeat`` (``fields``, never the text)
        and journal it (``event`` plus what was repeated). ``trigger`` :data:`REPEAT_SILENCE`
        The watch's grace and estimate count from the send. :data:`REPEAT_KEYPAD`
        (the caller pressed 9): the silence wait counts from the repeat's end, the silence
        repeat count unchanged."""
        if speech.turn is not None:  # Speech from here on is the repeat's, not the turn's
            speech.turn.speech_logged = True
        await self._send(context, text)
        if trigger == REPEAT_SILENCE:
            speech.watch.repeat_sent(time.monotonic())  # The grace counts from the send
        else:
            speech.watch.note_speech(text, time.monotonic())
        speech.playing = speech.label
        emit(log, logging.INFO, "prompt_repeat", **fields)
        if self.tracing is not None:  # No turn is open between turns, so on the session span
            span = self._turn_spans.get(call_sid) or self._relay_spans.get(call_sid)
            name = "silence_repeat" if trigger == REPEAT_SILENCE else "keypad_repeat"
            self.tracing.event(span, name, {f"owlbank.{k}": v for k, v in fields.items()})
        entry: dict[str, Any] = {"type": "prompt_repeat", **event, "what": speech.label}
        if speech.label == PLAYING_AGENT:  # the agent's words: journaled like an ai_turn
            entry["text"] = await self._turn_text(call_sid, text, speech.language, 4000)
        else:  # a catalog phrase or the front door's greeting: as their prompt events were
            if speech.label != PLAYING_GREETING:
                entry["prompt_key"] = speech.label
            entry["rendered_text"] = text.strip()
        await self._append(call_sid, entry)

    async def _silence_action(self, call_sid: str, speech: _CallSpeech, context: Any, exp: Expiry) -> None:
        """Exactly once: ``escalate`` through the escalation path with reason
        ``no_input``, or ``hangup`` (reason ``no_input_hangup``: the front door says
        ``common.goodbye_no_input`` and hangs up). The session then ends (``end`` frame)."""
        ref = new_journal_ref()
        if exp.action == "hangup":
            data: dict[str, Any] = {"reason": NO_INPUT_HANGUP}
            await self._append(call_sid, {"type": "handoff", "reason": NO_INPUT_HANGUP,
                                          "outcome": "requested"})
        else:
            emit(log, logging.INFO, "escalation_requested", reason=NO_INPUT, source="bridge")
            await self._append(call_sid, {"type": "escalation", "reason": NO_INPUT,
                                          "summary": NO_INPUT_SUMMARY})
            await self._append(call_sid, {"type": "handoff", "tool": "escalate", "reason": "escalate",
                                          "outcome": "requested"})
            try:
                await self._io(self._calls.patch, call_sid, {
                    F_ESCALATION: {"reason": NO_INPUT, "summary": NO_INPUT_SUMMARY, "at": int(self._now()),
                                   "source": "bridge", "journal_ref": ref}})
            except Exception as exc:
                log_caught(log, "could not store the escalation; handing off anyway", exc,
                           level=logging.ERROR, operation="store_escalation", decision="handoff_anyway")
            data = {"reason": "escalate", "tool": "escalate", "escalationReason": NO_INPUT}
        data["journal"] = ref
        data.update(_context_ids(context))
        emit(log, logging.INFO, "silence_action", action=exp.action, repeats=exp.repeat,
             timeout_ms=speech.watch.timeout_ms, what=speech.label)
        context.pending_handoff_data = self._handoff_factory(_clean_handoff(call_sid, data))
        self._note_handoff(call_sid, data.get("tool") or data.get("reason"))
        await self._send(context, "")  # TAC sends last: true, then the end frame

    # ------------------------------------------------------------------ AI keypad shortcuts

    def on_keypad(self, call_sid: str | None, digit: str) -> None:
        """A key pressed during the AI session (``PeekingWebSocket``, the frame swallowed): fed to
        the call's :class:`~owlbank.keypad.KeypadBuffer`, with a wake-up at the entry's due time.
        With a hand-off pending or no session, the call's keys are dropped; a hand-off whose
        ``end`` a key's barge-in kept from going out gets it sent (:meth:`_keypad_end_pending`).
        While TAC is still setting the session up (its conversation lookup, seconds after the
        welcome started), keys are held: they act once it is there, or are dropped once
        ``session_wait_s`` has passed since the socket's session started.
        The digit goes to the buffer only: never logged, journaled, traced or stored."""
        if not call_sid:
            return
        with obs.call_context(call_sid, component="bridge"):
            _, cause = self._keypad_context(call_sid)
            if cause is not None and cause != KEYPAD_NOT_READY:
                self._keypad_ignore(call_sid, cause)
                return
            buffer = self._keypad.setdefault(call_sid, keypads.KeypadBuffer())
            now = time.monotonic()
            due = buffer.on_digit(digit, now)
            self._keypad_schedule(call_sid, buffer, now, due)

    def _keypad_context(self, call_sid: str) -> tuple[Any | None, str | None]:
        """The live TAC session and ``None``, or ``None`` and why keys can't act:
        ``no_session`` (the socket's session is gone, or TAC's never came within
        ``session_wait_s`` of its start), :data:`KEYPAD_NOT_READY` (TAC's session is still being
        set up: keys are held) or ``handoff_pending``."""
        speech = self._speech.get(call_sid)
        if speech is None:
            return None, "no_session"
        context, _ = self._session_for_call(call_sid)
        if context is None:
            session = self._sessions.get(call_sid)
            if session is not None and time.monotonic() < session.started + self._cfg.session_wait_s:
                return None, KEYPAD_NOT_READY
            return None, "no_session"
        if speech.handoff or getattr(context, "pending_handoff_data", None) is not None:
            return None, "handoff_pending"
        return context, None

    def _keypad_ignore(self, call_sid: str, cause: str) -> None:
        buffer = self._keypad.get(call_sid)
        if buffer is not None:
            buffer.clear()
        with obs.call_context(call_sid, component="bridge"):
            emit(log, logging.INFO, "keypad_ignored", cause=cause, decision="dropped")
        if cause == "handoff_pending":
            self._keypad_end_pending(call_sid, cause="keypad")

    @staticmethod
    def _keypad_lost(call_sid: str, buffer: keypads.KeypadBuffer, cause: str) -> None:
        """Keys not yet acted on are dropped from ``buffer`` (the session ended or was replaced, a
        spoken prompt took over): logged with the cause, never silently (and never a digit)."""
        if not buffer.pending:
            return
        buffer.clear()
        with obs.call_context(call_sid, component="bridge"):
            emit(log, logging.INFO, "keypad_ignored", cause=cause, decision="dropped")

    def _keypad_refuse(self, call_sid: str, context: Any) -> bool:
        """ for a keypad turn that has just taken the call's lock (``run_turn``): whether a
        hand-off is now pending or the session is gone (then the turn calls no model)."""
        live, cause = self._keypad_context(call_sid)
        if cause is None and live is context:
            return False
        self._keypad_ignore(call_sid, cause or "no_session")
        return True

    def _keypad_barge(self, call_sid: str, speech: _CallSpeech, kind: str) -> None:
        """A ``prompt`` or ``interrupt`` reached the socket (``on_inbound``, before TAC sees it).

        * A ``prompt`` drops the keys not yet acted on and their wake-up: pressing 1 and saying
          "yes" is one turn, the spoken one.
        * Either cancels a keypad turn still running (as a barge-in or a new prompt
          cancels a spoken turn). TAC cancels it too, as its stream task, once the socket has had a
          prompt; before that TAC ignores an ``interrupt``, so the bridge cancels it here and sends
          TAC's acknowledgement (``last: true``) itself (:meth:`_keypad_ack`).
        * With shortcuts on, an ``interrupt`` that cut a turn after its hand-off was set gets the
          ``end`` sent (:meth:`_keypad_end_pending`)."""
        if kind == "prompt":
            self._keypad_forget(call_sid, cause="prompt")
        turn = speech.keypad_turn
        if turn is not None and not turn.done():
            turn.cancel()
            context, _ = self._session_for_call(call_sid)
            if kind == "interrupt" and not speech.prompted and context is not None:
                self._keypad_spawn(call_sid, self._keypad_ack(call_sid, speech, context, turn))
        if kind == "interrupt" and self.keypad_enabled:
            self._keypad_end_pending(call_sid, cause="interrupt")
        if kind == "prompt":
            speech.prompted = True

    async def _keypad_ack(self, call_sid: str, speech: _CallSpeech, context: Any, turn: asyncio.Task[Any]) -> None:
        """TAC's barge-in acknowledgement (an empty ``last: true``, then the ``end`` if a hand-off is
        pending), for an ``interrupt`` TAC ignored, once the cancelled keypad turn has unwound."""
        try:
            await asyncio.wait({turn})
            lock = self._locks.get(call_sid)
            if lock is None:
                return
            async with lock:
                if self._speech.get(call_sid) is not speech or speech.end_out:
                    return  # session over, or its ``end`` (after a ``last: true``) went out
                if getattr(context, "pending_handoff_data", None) is None:
                    await self._send(context, "")
                    return
            # A hand-off is pending: TAC would send its ``end`` after this ``last: true``. That goes
            # through the one ``end`` path (sent at most once, ``handoff_end_resent``), whose send
            # (or the turn still sending the ``end``) carries the ``last: true`` too.
            self._keypad_end_pending(call_sid, cause="interrupt")
        except Exception as exc:  # noqa: BLE001 - never breaks the call
            log_caught(log, "keypad barge-in acknowledgement failed", exc, operation="keypad_ack",
                       decision="ignored")

    def _keypad_end_pending(self, call_sid: str, *, cause: str) -> None:
        """The session's hand-off is set (``pending_handoff_data``) but its ``end`` hasn't gone
        out: a key (``interruptible="any"``) or a barge-in cancelled the turn that would have sent
        it, while it held for its last words (``_hold_handoff``). Send it, at most once per session,
        once no send is left that would send it (TAC's stream task, a resume, any ``_send``) and
        the socket hasn't sent one (``end_out``): TAC sends ``last: true``, then the ``end``."""
        speech = self._speech.get(call_sid)
        context, _ = self._session_for_call(call_sid)
        if (speech is None or context is None or speech.end_resent or speech.end_out
                or getattr(context, "pending_handoff_data", None) is None
                or (speech.end_task is not None and not speech.end_task.done())):
            return
        speech.end_task = self._keypad_spawn(call_sid, self._keypad_send_end(call_sid, speech, context, cause))

    async def _keypad_send_end(self, call_sid: str, speech: _CallSpeech, context: Any, cause: str) -> None:
        try:
            # A turn still streaming (TAC's stream task, or any other send, e.g. a resume at
            # ``setup``) sends the end itself when it finishes, after it released the lock.
            await self._stream_done(context)
            await self._sends_done(speech)
            lock = self._locks.get(call_sid)
            if lock is None:
                return  # the session ended
            async with lock:
                if (self._speech.get(call_sid) is not speech or speech.end_resent or speech.end_out
                        or getattr(context, "pending_handoff_data", None) is None
                        or not self._stream_done_now(context) or self._sending(speech)):
                    return  # sent, or a send still running sends it (a later key or barge-in re-checks)
                speech.end_resent = speech.handoff = True  # Keys stay dropped after the end
                emit(log, logging.INFO, "handoff_end_resent", cause=cause)
                await self._send(context, "")
        except Exception as exc:  # noqa: BLE001 - never breaks the call
            log_caught(log, "could not send the hand-off end", exc, level=logging.ERROR,
                       operation="handoff_end", decision="ignored")

    def _stream_done_now(self, context: Any) -> bool:
        """No turn is streaming to the caller for this session (TAC's stream task is done)."""
        state = tac_stream_state(self.voice, context.conversation_id)
        task = getattr(state, "stream_task", None)
        return task is None or task.done() or task is asyncio.current_task()

    async def _stream_done(self, context: Any) -> None:
        """Wait (without cancelling it) for the turn TAC is streaming for this session, if any."""
        for _ in range(4):  # a new prompt may replace it meanwhile; bounded
            state = tac_stream_state(self.voice, context.conversation_id)
            task = getattr(state, "stream_task", None)
            if task is None or task.done() or task is asyncio.current_task():
                return
            await asyncio.wait({task})

    @staticmethod
    def _sending(speech: _CallSpeech) -> set[asyncio.Future[None]]:
        """The session's sends still running, other than the current task's own."""
        current = asyncio.current_task()
        return {done for done, task in speech.sends.items() if task is not current and not done.done()}

    async def _sends_done(self, speech: _CallSpeech) -> None:
        """Wait (without cancelling them) for the session's other sends, TAC's ``last``/``end`` included."""
        for _ in range(4):  # new sends may start meanwhile; bounded
            running = self._sending(speech)
            if not running:
                return
            await asyncio.wait(running)

    def _keypad_schedule(self, call_sid: str, buffer: keypads.KeypadBuffer, now: float, due: float) -> None:
        """One wake-up per call at the open entry's due time. A wake-up set for an entry that was
        already complete when this key came (``now >= due``) is kept, so its held action isn't
        delayed by the new entry's window."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        previous = self._keypad_wakes.pop(call_sid, None)
        if previous is not None and now < previous[1]:
            previous[0].cancel()
        delay = max(0.005, due - time.monotonic())
        handle = loop.call_later(delay, self._keypad_wake, call_sid, buffer)
        self._keypad_wakes[call_sid] = (handle, due)

    def _keypad_rewake(self, call_sid: str, buffer: keypads.KeypadBuffer, now: float) -> None:
        """The wake-up ran before the open entry was complete (uvloop's millisecond timers can run
        a ``call_later`` callback up to about 1 ms before ``time.monotonic()`` reaches its due
        time; the lost a key this way): wake again at the due time."""
        due = buffer.due
        if due is None:
            return
        emit(log, logging.DEBUG, "keypad_wake_early", ms_early=round((due - now) * 1000, 3))
        self._keypad_schedule(call_sid, buffer, now, due)

    def _keypad_wake(self, call_sid: str, buffer: keypads.KeypadBuffer) -> None:
        if self._keypad.get(call_sid) is not buffer:
            self._keypad_lost(call_sid, buffer, "no_session")  # the session ended or restarted
            return
        self._keypad_spawn(call_sid, self._keypad_fire(call_sid, buffer))

    def _keypad_spawn(self, call_sid: str, work: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        """Keypad work for the call's current session: cancelled by :meth:`_keypad_drop`."""
        task = asyncio.create_task(work)  # copies the bound call SID
        self._keypad_tasks.setdefault(call_sid, set()).add(task)

        def untrack(done: asyncio.Task[Any]) -> None:
            tasks = self._keypad_tasks.get(call_sid)
            if tasks is not None:
                tasks.discard(done)
                if not tasks:
                    del self._keypad_tasks[call_sid]

        task.add_done_callback(untrack)
        return task

    def _keypad_forget(self, call_sid: str, *, cause: str) -> None:
        """The call's keys not yet acted on (logged ``keypad_ignored`` with ``cause`` if there
        were any), and their wake-up, are dropped (the buffer stays)."""
        buffer = self._keypad.get(call_sid)
        if buffer is not None:
            self._keypad_lost(call_sid, buffer, cause)
        wake = self._keypad_wakes.pop(call_sid, None)
        if wake is not None:
            wake[0].cancel()

    def _keypad_drop(self, call_sid: str) -> None:
        """The session started or ended: its keys, wake-up and keypad work in flight (an action
        waiting for the lock, a keypad turn) are dropped."""
        self._keypad_forget(call_sid, cause="session_end")
        self._keypad.pop(call_sid, None)
        try:
            current = asyncio.current_task()
        except RuntimeError:  # no loop: nothing can be in flight
            current = None
        for task in self._keypad_tasks.pop(call_sid, set()):
            if task is not current:
                task.cancel()

    async def _keypad_fire(self, call_sid: str, buffer: keypads.KeypadBuffer) -> None:
        with obs.call_context(call_sid, component="bridge"):
            try:
                await self._keypad_due(call_sid, buffer)
            except Exception as exc:  # noqa: BLE001 - keypad handling must never break the call
                log_caught(log, "keypad handling failed", exc, level=logging.ERROR,
                           operation="keypad_due", decision="ignored")

    async def _keypad_due(self, call_sid: str, buffer: keypads.KeypadBuffer) -> None:
        """Run every completed entry's action, oldest first, each under the call's lock with the
        session and hand-off checked again. A turn (``yes``/``no``) runs after the lock is
        released: :meth:`run_turn` takes it itself, so it waits for any turn still running. No lock
        means the session ended: nothing acts (and no lock is created for it).

        Every exit that drops a key not yet acted on logs ``keypad_ignored`` with its cause. While
        TAC's session is still being set up (:data:`KEYPAD_NOT_READY`) the keys are held: checked
        again every ``session_poll_s`` (the lock released meanwhile) until it is there, or dropped
        (``session_not_ready``) once ``session_wait_s`` has passed since the socket's session
        started. A wake-up that ran before the open entry was complete wakes again (it is not a drop)."""
        while True:
            turn: tuple[Any, str, str] | None = None
            wait = False
            lock = self._locks.get(call_sid)
            if lock is None:
                self._keypad_lost(call_sid, buffer, "no_session")  # the session ended
                return
            async with lock:
                if self._keypad.get(call_sid) is not buffer or self._locks.get(call_sid) is not lock:
                    self._keypad_lost(call_sid, buffer, "no_session")  # the session ended or was replaced
                    return
                context, cause = self._keypad_context(call_sid)
                speech = self._speech.get(call_sid)
                if context is None or speech is None:
                    if not buffer.pending:
                        return  # nothing left to act on (a prompt or another wake-up took it)
                    if cause == KEYPAD_NOT_READY:
                        wait = True
                    else:
                        if cause == "no_session" and speech is not None:
                            cause = KEYPAD_NOT_READY  # held for TAC's session, which didn't come in time
                        self._keypad_ignore(call_sid, cause or "no_session")
                        return
                else:
                    now = time.monotonic()
                    action = buffer.poll(now)
                    if action is None:
                        if buffer.pending:
                            self._keypad_rewake(call_sid, buffer, now)
                        return
                    turn = self._keypad_act_prepare(call_sid, action, context, speech)
                    if turn is None:
                        if action.kind == keypads.AGENT:
                            await self._keypad_agent(call_sid, context)
                            return  # a hand-off is pending: nothing more acts
                        if action.kind == keypads.REPEAT:
                            await self._keypad_guarded(action.kind, self._keypad_repeat(call_sid, context, speech))
                        else:
                            await self._keypad_guarded(action.kind, self._keypad_hint(call_sid, context, speech))
            if wait:
                await asyncio.sleep(self._cfg.session_poll_s)
                continue
            if turn is not None:
                await self._keypad_turn(call_sid, *turn)

    def _keypad_act_prepare(self, call_sid: str, action: keypads.Action, context: Any,
                            speech: _CallSpeech) -> tuple[Any, str, str] | None:
        """Log (and trace) the action; for ``yes``/``no`` the turn to run once the lock is released."""
        emit(log, logging.INFO, "keypad_action", action=action.kind, keys=action.keys)
        if self.tracing is not None:
            self.tracing.event(self._relay_spans.get(call_sid), "keypad_action",
                               {"owlbank.action": action.kind, "owlbank.keys": action.keys})
        if action.kind in KEYPAD_WORDS:
            return context, self._keypad_word(action.kind, speech.language), action.kind
        return None

    async def _keypad_turn(self, call_sid: str, context: Any, word: str, kind: str) -> None:
        """Exactly as if the caller had said the word (TAC's prompt path). The turn is the
        session's TAC stream task, so a barge-in (speech or a key) or a new prompt cancels it as it
        cancels a spoken turn (and TAC sends its ``last: true``; before the socket's first prompt,
        when TAC ignores a barge-in, :meth:`_keypad_barge` does both); it starts once a turn TAC is
        still streaming has finished (that turn is not cancelled: a key waits for the turn)."""
        state = tac_stream_state(self.voice, context.conversation_id)
        previous = getattr(state, "stream_task", None)
        task = self._keypad_spawn(call_sid, self._keypad_turn_run(context, word, kind, previous))
        if previous is not None:
            # Cancelled before its first step (TAC's prompt/interrupt or ``_keypad_barge`` in this
            # same loop iteration), the coroutine never runs its own ``previous.cancel()``.
            task.add_done_callback(lambda done: previous.cancel() if done.cancelled() else None)
        if state is not None:
            state.stream_task = task
        speech = self._speech.get(call_sid)
        if speech is not None:
            speech.keypad_turn = task  # a barge-in before the socket's first prompt: cancelled here
        try:
            await asyncio.wait({task})  # the next entry acts after this one (a cancelled turn included)
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def _keypad_turn_run(self, context: Any, word: str, kind: str, previous: asyncio.Task[Any] | None) -> None:
        if previous is not None and not previous.done():
            try:
                await asyncio.wait({previous})
            except asyncio.CancelledError:  # a barge-in or prompt cancelled the keypad turn: that turn too
                previous.cancel()
                raise
        await self._keypad_guarded(kind, self._keypad_send_turn(context, word))

    async def _keypad_send_turn(self, context: Any, word: str) -> None:
        """``run_turn`` for the keypad word, sent through TAC. A turn refused at the lock sends
        nothing, not even TAC's ``last: true``."""
        guard = _KeypadTurn()
        gen = self.run_turn(context, word, keypad=guard)
        async with aclosing(gen):
            try:
                first: str | None = await anext(gen)
            except StopAsyncIteration:
                if guard.refused:
                    return
                first = None

            async def chunks() -> AsyncGenerator[str, None]:
                if first is not None:
                    yield first
                async for chunk in gen:
                    yield chunk

            await self._send(context, chunks())

    @staticmethod
    async def _keypad_guarded(kind: str, work: Awaitable[Any]) -> None:
        """A failed turn, repeat or hint leaves the call as it was (``keypad_action_failed``)."""
        try:
            await work
        except Exception as exc:  # noqa: BLE001
            log_caught(log, "keypad_action_failed", exc, action=kind, decision="call_unchanged")

    def _keypad_word(self, kind: str, lang: str) -> str:
        """``keypad.yes``/``keypad.no`` in the call's language; the English word (and
        ``keypad_phrase_missing``) when the catalog has none."""
        key, fallback = KEYPAD_WORDS[kind]
        try:
            text = self._catalog.text(key, lang).strip()
        except CatalogKeyError:
            text = ""
        if not text:
            emit(log, logging.WARNING, "keypad_phrase_missing", key=key, language=lang,
                 decision="english_word")
            return fallback
        return text

    async def _keypad_repeat(self, call_sid: str, context: Any, speech: _CallSpeech) -> None:
        """The session's last message again, as the silence repeat sends it; the silence
        repeat count is unchanged and the silence wait counts from the repeat's end."""
        text = speech.text or ""
        if not text.strip():
            emit(log, logging.INFO, "keypad_action_skipped", action=keypads.REPEAT, cause="nothing_spoken")
            return
        fields = {"trigger": REPEAT_KEYPAD, "what": speech.label, "chars": len(text)}
        await self._repeat_last(call_sid, context, speech, text, trigger=REPEAT_KEYPAD, fields=fields,
                                event={"trigger": REPEAT_KEYPAD})
        self._silence_reschedule(call_sid)

    async def _keypad_hint(self, call_sid: str, context: Any, speech: _CallSpeech) -> None:
        """``keypad.hint`` out of turn (scripted, never in the agent's history), then the
        silence watch re-armed as at a turn's end. The session's last message (what 9 and the
        silence repeat speak) stays the one before the hint, unless nothing was said yet: then it
        is the hint. The watch's speaking estimate counts the hint either way, so the silence wait
        starts at its end. No hint in the catalog: skipped."""
        try:
            text = self._catalog.text(KEYPAD_HINT_KEY, speech.language).strip()
        except CatalogKeyError:
            text = ""
        if not text:
            emit(log, logging.WARNING, "keypad_phrase_missing", key=KEYPAD_HINT_KEY,
                 language=speech.language, decision="hint_skipped")
            return
        if speech.turn is not None:  # Speech from here on is the hint's, not the turn's
            speech.turn.speech_logged = True
        if (speech.text or "").strip():  # a message to repeat: it stays the last message
            speech.watch.note_speech(text, time.monotonic())
            speech.playing = KEYPAD_HINT_KEY
        else:
            self._mark(call_sid, KEYPAD_HINT_KEY)
            self._note_out(call_sid, text)
        await self._send(context, text)
        if self._speech.get(call_sid) is speech:
            self._silence_configure(speech.watch)
            speech.watch.arm(speech.text, speech.at)
            self._silence_reschedule(call_sid)

    def _keypad_summary(self) -> str:
        """The fixed English summary (the human agent's language, like other summaries)."""
        try:
            text = self._catalog.text(KEYPAD_SUMMARY_KEY, "en-US").strip()
        except CatalogKeyError:
            text = ""
        if not text:
            emit(log, logging.WARNING, "keypad_phrase_missing", key=KEYPAD_SUMMARY_KEY, language="en-US",
                 decision="fixed_summary")
            return KEYPAD_SUMMARY_FALLBACK
        return text

    async def _keypad_agent(self, call_sid: str, context: Any) -> None:
        """0 pressed. The keypad escalation; if it fails before the hand-off is set, the
        bridge's error escalation (``_escalate_on_error``) instead. The session then ends."""
        try:
            await self._keypad_escalate(call_sid, context)
            return
        except Exception as exc:  # noqa: BLE001
            log_caught(log, "keypad_action_failed", exc, level=logging.ERROR, action=keypads.AGENT,
                       decision="escalate_on_error" if getattr(context, "pending_handoff_data", None) is None
                       else "handoff_set")
        try:
            if getattr(context, "pending_handoff_data", None) is None:
                view = await self._io(self._calls.load, call_sid)
                if view is not None:
                    await self._escalate_on_error(context, view)
                else:  # no call state: the bare escalation hand-off (the front door's fallback queue)
                    self._set_handoff(context, call_sid, {"reason": "escalate", "tool": "escalate",
                                                          "escalationReason": FALLBACK_REASON})
            await self._send(context, "")
        except Exception as exc:  # noqa: BLE001 - the call stays with the AI; the caller can ask again
            log_caught(log, "keypad escalation fallback failed", exc, level=logging.ERROR,
                       operation="escalate_on_error", decision="call_unchanged")

    async def _keypad_escalate(self, call_sid: str, context: Any) -> None:
        """Modelled on the silence ``no_input`` escalation: no model call; the reason
        ``caller_requested_human``, a fixed summary, ``source`` and hand-off ``trigger``
        ``keypad`` (the front door routes it before ``_classify_handoff``); then the ``end``."""
        ref = new_journal_ref()
        summary = self._keypad_summary()
        emit(log, logging.INFO, "escalation_requested", reason=KEYPAD_REASON, source=KEYPAD_TRIGGER)
        await self._append(call_sid, {"type": "escalation", "reason": KEYPAD_REASON, "summary": summary})
        await self._append(call_sid, {"type": "handoff", "tool": "escalate", "reason": "escalate",
                                      "trigger": KEYPAD_TRIGGER, "outcome": "requested"})
        try:
            await self._io(self._calls.patch, call_sid, {
                F_ESCALATION: {"reason": KEYPAD_REASON, "summary": summary, "at": int(self._now()),
                               "source": KEYPAD_TRIGGER, "journal_ref": ref}})
        except Exception as exc:
            log_caught(log, "could not store the escalation; handing off anyway", exc,
                       level=logging.ERROR, operation="store_escalation", decision="handoff_anyway")
        data: dict[str, Any] = {"reason": "escalate", "tool": "escalate", "escalationReason": KEYPAD_REASON,
                                "trigger": KEYPAD_TRIGGER, "journal": ref, **_context_ids(context)}
        context.pending_handoff_data = self._handoff_factory(_clean_handoff(call_sid, data))
        self._note_handoff(call_sid, "escalate")
        buffer = self._keypad.get(call_sid)
        if buffer is not None:
            self._keypad_lost(call_sid, buffer, "handoff_pending")  # Nothing pressed after this acts
        await self._send(context, "")  # TAC sends last: true, then the end frame

    async def _arm_greeting(self, call_sid: str, view: CallView, setup_at: float, epoch: int) -> None:
        """The session's welcome greeting, unsealed from call state, is the first message
        the watch counts from. Missing, expired or not unsealable: no greeting repeat (the next
        turn's speech still arms the watch). Skipped if anything happened since ``setup``."""
        speech = self._speech.get(call_sid)
        if speech is None:
            return
        speech.language = view.language
        if view.last_spoken is None:
            speech.welcome_known = True  # No welcome to record for this session
            return
        speech.clear_last_spoken = True
        text = await self._unseal_welcome(call_sid, view)
        if self._speech.get(call_sid) is speech and not speech.welcome_known:
            # The session's first agent hop records it, whatever happened since setup.
            speech.welcome, speech.welcome_known = text or None, True
        if self._speech.get(call_sid) is not speech or speech.watch.input_epoch != epoch or not text:
            return  # caller input or a turn since setup (speaker events don't count)
        self._silence_configure(speech.watch)
        speech.watch.note_speech(text, setup_at)
        speech.text, speech.at, speech.label, speech.open = text, setup_at, PLAYING_GREETING, False
        speech.playing = PLAYING_GREETING
        first = speech.first_session if speech.first_session is not None else not view.earlier_ai_session
        greeting_ms = self._greeting_timeout_ms() if first else None
        if speech.watch.arm(text, setup_at, timeout_ms=greeting_ms):
            self._silence_reschedule(call_sid)

    async def _unseal_welcome(self, call_sid: str, view: CallView) -> str | None:
        """The session's welcome, unsealed from call state's ``last_spoken`` (``None``: missing,
        expired or not unsealable)."""
        if view.last_spoken is None:
            return None
        try:
            text = await self._io(last_spoken_seal.unseal, self._sealer, view.last_spoken,
                                  call_sid=call_sid, now=self._now())
        except Exception as exc:  # noqa: BLE001 - unseal never raises; defensive
            log_caught(log, "greeting unseal failed", exc, operation="unseal_last_spoken",
                       decision="no_greeting_repeat")
            return None
        return text if isinstance(text, str) and text.strip() else None

    async def _session_welcome(self, call_sid: str, view: CallView) -> str | None:
        """The session's welcome for this turn's first agent hop, until it is
        taken (``None`` after that, or when there was none). Normally what ``_arm_greeting``
        unsealed at ``setup``; if the turn got here first, unsealed now from the state the turn
        loaded (and kept, so a later turn doesn't unseal it again). Not taken here: a turn that
        fails before the welcome is in the history leaves it for the next turn
        (:meth:`_welcome_taken`)."""
        speech = self._speech.get(call_sid)
        if speech is None or speech.welcome_taken:
            return None
        if speech.welcome_known:
            return speech.welcome
        text = await self._unseal_welcome(call_sid, view)
        if self._speech.get(call_sid) is speech and not speech.welcome_known:
            speech.welcome, speech.welcome_known = text, True
        return text

    def _welcome_taken(self, call_sid: str) -> None:
        """The session's welcome is in the agent's history (or will never be): no later turn
        of the session places it again."""
        speech = self._speech.get(call_sid)
        if speech is not None:
            speech.welcome_taken = True

    async def _record_welcome(self, call_sid: str, view: CallView, welcome: str) -> None:
        """No tool result pending: the synthetic pair (user ``agent.call_started``, assistant
        the welcome) goes into the agent's history with no model call, so the caller's reply
        answers the agent's own turn. A catalog without the key (a bridge deployed before the
        reseed): nothing recorded, one warning. A failure is a retryable :class:`AgentError`."""
        try:
            started = self._catalog.text("agent.call_started", view.language)
        except CatalogKeyError:
            emit(log, logging.WARNING, "call_started_missing", language=view.language,
                 decision="welcome_not_recorded")
            return
        _assert_no_token(view.sealed_token, welcome)
        try:
            await self._agent.record_exchange(call_sid, started, welcome)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"record_exchange failed: {type(exc).__name__}", retryable=True,
                             code="record_exchange_failed") from exc
        emit(log, logging.INFO, "welcome_recorded", path="exchange", chars=len(welcome))

    def _greeting_timeout_ms(self) -> int | None:
        """``masking.greeting_silence_timeout_ms`` through the policy cache (``None``,
        i.e. ``silence_timeout_ms``, if the policy can't be read)."""
        try:
            return int(self._policy.masking.greeting_silence_timeout_ms)
        except Exception as exc:  # noqa: BLE001 - fall back to the ordinary timeout
            log_caught(log, "greeting silence setting unavailable", exc, operation="silence_settings",
                       decision="use_silence_timeout")
            return None

    def _silence_after_turn(self, context: Any, call_sid: str, speech: _CallSpeech,
                            stats: _TurnStats | None) -> None:
        """A turn ended: arm from its last message (or, if it said nothing, the last message the
        caller heard). Not after a barge-in, an error or a hand-off (the session is ending)."""
        speech.watch.turn_ended()
        if self._speech.get(call_sid) is not speech:
            return
        spoke = speech.turn_spoke
        if stats is not None and stats.outcome == "ok" \
                and getattr(context, "pending_handoff_data", None) is None:
            self._silence_configure(speech.watch)
            speech.watch.arm(speech.text, speech.at if spoke else time.monotonic())
        self._silence_reschedule(call_sid)
        if spoke and speech.clear_last_spoken:
            speech.clear_last_spoken = False  # The greeting is no longer the last message
            task = asyncio.create_task(self._clear_last_spoken(call_sid))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _clear_last_spoken(self, call_sid: str) -> None:
        with obs.call_context(call_sid, component="bridge"):
            try:
                await self._io(self._calls.patch, call_sid, None, [F_LAST_SPOKEN])
            except Exception as exc:  # noqa: BLE001 - best effort; it expires anyway
                log_caught(log, "could not clear last_spoken", exc, operation="clear_last_spoken",
                           decision="ignored_it_expires")

    # ------------------------------------------------------------------ small things

    def _planner(self, call_sid: str) -> MaskPlanner:
        """A fresh per-turn :class:`MaskPlanner` from the current policy, with the call's
        next acknowledgement from the rotated pool (never the one spoken last).

        Settings are read per turn through the policy's cache (a table edit applies
        within 60 s, on the next turn); bad fields were already defaulted with an ERROR by the
        lenient loader."""
        cfg = self._policy.masking
        end_mode: EndMode = "immediate" if cfg.end_mode == "immediate" else "wait"
        masks: dict[str, Any] = {}

        def tool_mask(tool: str) -> Any:  # one read per tool per turn
            if tool not in masks:
                masks[tool] = self._policy.tool_mask(tool)
            return masks[tool]

        return MaskPlanner(after_ms=cfg.after_ms, handoff_wait_ms=cfg.handoff_wait_ms,
                           end_mode=end_mode, chars_per_s=float(cfg.chars_per_s),
                           tool_after_ms=cfg.tool_after_ms, tool_mask=tool_mask,
                           is_handoff=self._is_handoff,
                           generic_key=next_generic_key(self._rotation_get(call_sid, _GENERIC_SLOT)))

    def _mask_text(self, key: str, lang: str) -> str:
        """A masking phrase from the catalog, or ``""`` (and a WARNING) if missing."""
        try:
            text = self._catalog.text(key, lang).strip()
        except CatalogKeyError:
            text = ""
        if not text:
            emit(log, logging.WARNING, "mask_phrase_missing", key=key[:128], language=lang,
                 decision="no_phrase")
            return ""
        return text + " "

    @staticmethod
    def _log_mask(key: str, tool: str | None, kind: str, stats: _TurnStats, *,
                  after_ms: int | None = None, source: str | None = None) -> None:
        """19: the ``mask_spoken`` line (key, tool, kind, ms since the turn started, the
        delay used and where it came from: ``tool``, ``default`` or ``handoff``)."""
        fields: dict[str, Any] = {
            "key": key, "tool": tool, "kind": kind, "turn": stats.number,
            "ms_since_turn_start": round((time.monotonic() - stats.started) * 1000, 1),
            "after_ms": after_ms, "source": source}
        emit(log, logging.INFO, "mask_spoken", **fields)
        if stats.tracing is not None:  # The same identifiers, never the phrase text
            stats.tracing.event(stats.span, "mask_spoken", {f"owlbank.{k}": v for k, v in fields.items()})

    def _phrase(self, key: str, lang: str) -> str:
        try:
            return self._catalog.text(key, lang) + " "
        except CatalogKeyError:
            log.debug("catalog phrase missing", extra={"key": key, "decision": "ignored_because_no_phrase"})
            return ""

    async def _turn_text(self, call_sid: str, text: str, lang: str, cap: int) -> str:
        """An ``ai_turn`` text for the journal: factor values, then PII masked. Withheld
        when a sealed value of the call couldn't be unsealed (fail closed)."""
        if sensitive.is_unverifiable(call_sid):
            return WITHHELD_TURN[:cap]
        text = segments.redact_text(call_sid, text)
        try:
            text = await self._io(self._pii.text, SURFACE_AI_TURN, text, language=lang,
                                  withheld=WITHHELD_TURN)
        except Exception as exc:
            log_caught(log, "PII guard failed; withholding the turn text", exc, level=logging.ERROR,
                       operation="pii_turn_text", decision="withhold_text")
            text = WITHHELD_TURN
        return text[:cap]

    async def _summary_text(self, call_sid: str, summary: Any, lang: str) -> str:
        """The escalation summary: cleaned, then PII masked, or the withheld phrase (also
        when a sealed value of the call couldn't be unsealed: fail closed)."""
        withheld = self._phrase("escalation.summary_withheld", lang).strip() or WITHHELD_TURN
        if sensitive.is_unverifiable(call_sid):
            return withheld[:SUMMARY_MAX_LEN].rstrip()
        text = clean_summary(call_sid, summary)
        try:
            text = await self._io(self._pii.text, SURFACE_ESCALATION_SUMMARY, text, language=lang,
                                  withheld=withheld)
        except Exception as exc:
            log_caught(log, "PII guard failed; withholding the summary", exc, level=logging.ERROR,
                       operation="pii_summary", decision="withhold_text")
            text = withheld
        return text[:SUMMARY_MAX_LEN].rstrip()

    async def _append(self, call_sid: str, event: Mapping[str, Any]) -> int | None:
        try:
            return await self._io(self._journal.append, call_sid, event)
        except RedactionError as exc:
            log_caught(log, "journal event rejected by the redaction guard", exc,
                       event_type=str(event.get("type"))[:40], decision="ignored_event_not_journaled")
            return None
        except Exception as exc:
            log_caught(log, "journal append failed", exc, level=logging.ERROR,
                       event_type=str(event.get("type"))[:40], operation="journal_append",
                       decision="ignored_because_journal_is_best_effort")
            return None


# --------------------------------------------------------------------------- module helpers


def _log_tool_result(tool: str, content: Any = None, *, exception: BaseException | None = None) -> None:
    """One ``tool_result`` line for an inline tool call that reached an outcome (a bank tool, or
    a call the bridge refused before running it), classified by
    :func:`owlbank.tools_contract.tool_outcome`. No arguments or result text. Hand-offs and the
    hand-off prefetch log none. Logged with ``callObservability`` off too (a plain line; the metric
    filters exist only when it is on). Never raises."""
    with suppress(Exception):  # logging never breaks a turn
        outcome = tool_outcome("inline", content, exception=exception)
        fields: dict[str, Any] = {"tool": tool[:64], "tool_kind": "inline", "outcome": outcome}
        code = tool_error_code("inline", content) if exception is None else None
        if code is not None:
            fields["error_code"] = code
        emit(log, logging.INFO, "tool_result", **fields)


def _eager_task(coro: Coroutine[Any, Any, ToolResult]) -> asyncio.Task[ToolResult]:
    """Start ``coro`` now: it runs up to its first suspension before this returns."""
    return asyncio.Task(coro, loop=asyncio.get_running_loop(), eager_start=True)


def _no_floats(value: Any) -> Any:
    """A copy with every float or Decimal as a decimal string (the store rejects them)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {k: _no_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_no_floats(v) for v in value]
    return value



#: Calls whose last acknowledgement the bridge remembers; far above one task's load.
MASK_ROTATION_MAX_CALLS = 5000


def next_generic_key(last: str | None, pool: Sequence[str] = MASK_GENERIC_KEYS) -> str:
    """The acknowledgement after ``last`` in the pool, in rotation; the first if none."""
    if last in pool:
        return pool[(pool.index(last) + 1) % len(pool)]
    return pool[0]


#: The rotation slot of the acknowledgements (tool pools use the tool's name).
_GENERIC_SLOT = ""


def rotation_order(pool: Sequence[str], last: str | None) -> list[str]:
    """The order to try a tool's pool in: the keys after ``last`` in rotation,
    with ``last`` itself tried only at the end (if it is the only phrase that renders for this
    call's arguments, a repeat beats silence). ``last`` unknown or ``None``: the pool's order."""
    keys = list(dict.fromkeys(pool))
    if last not in keys:
        return keys
    i = keys.index(last)
    return [*keys[i + 1:], *keys[:i], last]


_GATEWAY_NAMES: dict[str, str] = {gateway_tool_full_name(t): t for t in GATEWAY_LEVEL0_TOOLS}


def _gateway_tool_name(name: str) -> str | None:
    """The policy name of a Gateway tool from a :class:`ToolStarted` (``level0___faq_lookup`` →
    ``faq_lookup``; a bare known name is accepted), or ``None`` for anything else."""
    if name in GATEWAY_LEVEL0_TOOLS:
        return name
    return _GATEWAY_NAMES.get(name)


def _profile_text(customer: Mapping[str, Any], view: CallView) -> str:
    """The back-end profile for the prompt: name and account types. No numbers or balances."""
    name = " ".join(str(customer.get(k) or "") for k in ("first_name", "last_name")).strip()
    accounts = []
    for a in customer.get("accounts") or []:
        if isinstance(a, Mapping) and a.get("type"):
            nick = f" ({untrusted.wrap('account_nickname', a['nickname'])})" if a.get("nickname") else ""
            accounts.append(f"{a['type']}{nick}")
    parts = [f"name {name}" if name else "", f"accounts: {', '.join(accounts)}" if accounts else ""]
    return "; ".join(p for p in parts if p)


def _context_ids(context: Any) -> dict[str, str]:
    """TAC's Orchestrator ``conversationId`` and the verified Memory ``profileId``, for the
    front door's task attributes. IDs only; the profile is set only after verification."""
    out: dict[str, str] = {}
    conversation_id = getattr(context, "conversation_id", None)
    if isinstance(conversation_id, str) and conversation_id:
        out["conversationId"] = conversation_id
    meta = getattr(context, "metadata", None)
    profile_id = (meta.get(META_KEY) or {}).get("verified_profile_id") if isinstance(meta, dict) else None
    if isinstance(profile_id, str) and profile_id:  # set by _load_memory only, never TAC's ANI profile
        out["profileId"] = profile_id
    return out


def _tool_args(call: ToolCall) -> dict[str, Any]:
    """The call's validated arguments: only the tool's own parameters (they stay in call state;
    handoffData, which Twilio sees, gets no amounts or accounts)."""
    if call.name == "get_balance":
        return {"account": call.input.get("account")}
    return {k: call.input[k] for k in tool_schemas()[call.name]["properties"] if k in call.input}


def _tool_call_event(call_sid: str, call: ToolCall) -> dict[str, Any]:
    event = {"type": "tool_call", "tool": call.name, "tool_use_id": call.tool_use_id,
             "args": dict(call.input)}
    if call.name == "escalate":
        event["args"] = {"reason": call.input.get("reason")}  # the summary is journaled as cleaned text
    try:
        segments.assert_clean(call_sid, event)
    except RedactionError:
        log.debug("tool call args not journaled", extra={
            "tool": call.name, "decision": "fallback_arg_names_only"})
        event["args"] = {"keys": sorted(str(k) for k in call.input)}
    return event


def _scrub_result(token: str | None, tool: str, result: Mapping[str, Any], *,
                  sealed: str | None = None) -> dict[str, Any]:
    """Defence in depth: a result that echoes the (sealed) token, or has a denied key, is withheld."""
    out = dict(result)
    try:
        _assert_no_token(token, out)
        _assert_no_token(sealed, out)
        segments.assert_clean("", out)  # deny-listed keys only (no registered values here)
    except (TokenLeakError, RedactionError) as exc:
        log_caught(log, "bank_tools result withheld from the model", exc, tool=tool,
                   decision="return_unavailable")
        return {"error": "UNAVAILABLE", "tool": tool}
    return out


def new_journal_ref() -> str:
    """An opaque, unguessable journal reference (one per handoff; stored for escalations)."""
    return secrets.token_urlsafe(16)


def _handoff_for_pending(view: CallView, *, step_up: bool = False) -> dict[str, Any] | None:
    """Rebuild ``handoffData`` from call state, for a handoff whose ``end`` was lost (barge-in).
    ``step_up``: the pending front-door tool still needs a step-up."""
    pt = view.pending_tool
    if pt is None:
        return None
    stored = (view.escalation or {}).get("journal_ref")
    ref = stored if pt.name == "escalate" and isinstance(stored, str) and stored else new_journal_ref()
    if step_up and pt.name not in ("authenticate", "escalate"):
        return {"reason": "step-up", "tool": pt.name, "args": {"tool": pt.name}, "journal": ref}
    if pt.name == "authenticate":
        target = str(pt.args.get("tool") or view.step_up_target or "")
        return {"reason": "step-up", "tool": target, "args": {"tool": target}, "journal": ref}
    if pt.name == "get_balance":
        return {"reason": "tool", "tool": "get_balance", "args": {"account": pt.args.get("account")},
                "journal": ref}
    if pt.name in BANK_TOOLS:  # a confirm handoff; the args stay in call state
        return {"reason": "confirm", "tool": pt.name, "journal": ref}
    if pt.name == "escalate":
        reason = (view.escalation or {}).get("reason") or pt.args.get("reason") or FALLBACK_REASON
        return {"reason": "escalate", "tool": "escalate", "escalationReason": reason, "journal": ref}
    return None


def _level_refusal(view: CallView, tool: str, log_refusal: bool = True) -> dict[str, Any]:
    """``AUTH_LEVEL_INSUFFICIENT`` for ``tool``; once the call's login is closed (keyed-phone login) with
    ``"verification": "closed"``, so
    the model offers what it can instead of calling ``authenticate`` first. ``log_refusal``:
    false for a look-ahead or the hand-off prefetch, so each refusal the model sees logs once."""
    refusal: dict[str, Any] = dict(auth_level_insufficient(tool))
    if view.login_closed:
        if log_refusal:
            emit(log, logging.INFO, "verification_closed", tool=tool)
        refusal["verification"] = "closed"
    return refusal


def _pending_step(view: CallView, context: Any) -> str | None:
    """The step a hand-off is waiting on, the single definition of "pending"
    for ``end_call``: ``tool`` / ``result`` (a front-door hand-off's ``pending_tool`` /
    ``pending_result``; a step-up in progress has its tool as ``pending_tool``) or ``handoff``
    (this session's ``pending_handoff_data`` is already set); ``None`` when nothing is pending.
    ``step_up_target`` and ``escalation`` don't count: both stay in call state after their step
    is over (``had_ai_session``, the human leg), so they would block ``end_call`` for the rest of
    the call."""
    if view.pending_tool is not None:
        return "tool"
    if view.pending_result is not None:
        return "result"
    if getattr(context, "pending_handoff_data", None) is not None:
        return "handoff"
    return None


def _clean_handoff(call_sid: str, data: dict[str, Any]) -> str:
    """``handoffData`` JSON: reason, tool, args, journal ref; never customer data."""
    try:
        segments.assert_clean(call_sid, data)
    except RedactionError:
        log.warning("handoff args dropped by the redaction guard", extra={"decision": "fallback_drop_args"})
        data = {k: v for k, v in data.items() if k != "args"}
        segments.assert_clean(call_sid, data)
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def _utterance(view: CallView, env: Mapping[str, Any]) -> str | None:
    """The caller's words a front-door result carried: the interrupted utterance or the
    envelope's ``callerUtterance``, for the resumed hop's ``gen_ai.user.message``."""
    words = view.interrupted_utterance or env.get("callerUtterance")
    return words.strip() or None if isinstance(words, str) else None


def _envelope_step(env: Mapping[str, Any]) -> str | None:
    """The step a front-door result with no pending tool resumes after (8.0's placeholder)."""
    step = env.get("tool") or env.get("next")
    return str(step) if step else None
