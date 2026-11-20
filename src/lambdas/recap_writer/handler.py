"""Lambda entry point for the call recap writer.

Invoked asynchronously (``InvocationType=Event``) once a call is over: by the front door at call end
when there is no human-leg close to wait for, and after the human-leg close. The payload is
``{"call_sid": "CA…"}`` and nothing else. Lambda retries a failed async invoke once; this
handler raises only before it has claimed the call, so the retry covers a cold failure and never
duplicates a write. One run (:func:`write_recap`):

1. the policy switch ``memory.write`` off → ``skipped switch_off``; redaction off in policy →
   ``skipped redaction_off`` (off isn't "passing"). Nothing is claimed.
2. the claim: ``recap`` in call state, written with :func:`owlbank.callstate.mutate` from
   :func:`owlbank.recap.recap_claim`. No claim (another writer has it, or an outcome is recorded) →
   stop, one DEBUG line.
3. the call state and the verified customer's record → :func:`owlbank.recap.recap_target`:
   the verified customer's ``memory_email`` profile (created at step 10 if missing, as the bridge
   does), else an existing profile matching a real E.164 caller ID that passed STIR/SHAKEN (never
   created; an untrusted one → ``skipped ani_untrusted``), else ``skipped``.
4. the conversation's communications and participants → transcript lines; none →
   ``skipped no_transcript``; an Orchestrator read failure (or more than ``READ_MAX_PAGES`` pages) →
   ``failed orchestrator_error``. Read before the profile, so a call with nothing to recap never
   creates one.
5. the call's sealed factor values (``sealed_sensitive``) unsealed one blob at a time with
   :func:`owlbank.sensitive.restore`; any failure → ``skipped redaction_failed``.
6. the Memory profile: ``Profiles/Lookup`` by email or phone; none and a phone target → ``skipped
   no_profile``; more than one → ``skipped ambiguous_profile`` (never guess); none and an email
   target → created at step 10.
7. the transcript scrubbed (:func:`owlbank.recap.scrub`: factor values with and without commas,
   digit runs, the PII check failing closed) → ``skipped redaction_failed`` when it can't be.
8. one Bedrock Converse call with the forced tool ``write_recap {summary, observations}`` (catalog key ``agent.recap.system``,
   the transcript as untrusted data in a delimited block, 15 s connect and 15 s read, one attempt) → ``failed model_error|model_timeout|model_malformed``.
9. each output piece scrubbed again, then :func:`owlbank.recap.bounded`, then a last factor-value
   check on what is about to be written.
10. a missing email profile created (``POST Profiles`` with the ``Contact.email`` trait, the
    bridge's ``create_profile``; only now, so a run that stops earlier leaves no empty profile),
    ``recap.writing_at`` marked (:func:`owlbank.recap.recap_mark`, conditional on still holding the
    claim), then one summary and at most 5 observations in one batch (``source`` ``owlbank-voice``,
    the conversation ID, ``occurredAt``). A failed observation write after the summary →
    ``failed partial_write``, never a second attempt.
11. the outcome ``{outcome, reason, profile_id, at}`` recorded in ``recap`` (while still the holder).

**The deadline.** The run has the time the Lambda context gives
(``get_remaining_time_in_millis``; :data:`DEFAULT_RUN_S` without a context) less
:data:`DEADLINE_MARGIN_S` to record the outcome. Each step's bound is what its client's configuration
allows (connect plus read timeouts, times the attempts): a Memory or Orchestrator request 6 s
(:data:`REQUEST_BOUND_S`), a KMS unseal 6 s per blob, a Comprehend check 6 s per 99,000-byte chunk
(two attempts), the model 30 s. Before each step (each Orchestrator page, each blob, each text
checked) whose bound would pass the deadline, the run records ``failed deadline`` and makes no
further call; the checks are against the clock, one step at a time, so a call with normal latency is
never stopped by the bounds of steps still to come. The last check is before the profile create and
``writing_at`` and covers the create and both Memory writes: once the write has started it is never
stopped (nor ever taken over).

Every exception after the claim is caught (``log_caught``), recorded as ``failed`` with the step's
reason and never re-raised. One log line per call: ``recap_written`` or ``recap_skipped`` with the
reason, the call SID and the profile ID, never any text.

Environment:

* ``OWLBANK_SECRET_TWILIO_API_KEY`` (required): the ``twilio_api_key`` secret (Memory and
  Orchestrator), read on first use.
* ``OWLBANK_MEMORY_STORE_ID`` (required): the Memory store.
* ``OWLBANK_RECAP_MODEL_ID`` (optional): the Bedrock model or inference profile (default the slot
  resolver's Haiku 5.5, :data:`owlbank.slots.resolver.DEFAULT_MODEL_ID`).
* ``OWLBANK_TOKEN_KMS_KEY``: the sealing key of ``sealed_sensitive`` (unset: a call with sealed values
  is skipped ``redaction_failed``).
* ``OWLBANK_TABLE_CALLS``, ``OWLBANK_TABLE_CUSTOMERS``, ``OWLBANK_TABLE_POLICY``,
  ``OWLBANK_TABLE_CATALOG`` (the ``owlbank.config`` names), ``OWLBANK_CACHE_TTL_S``, ``AWS_REGION``.

Cold start: importing this module outside Lambda makes no AWS call; inside Lambda the
dependencies are built during INIT (``coldstart.build_at_init``; the policy and catalog are read
then, no secret is).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Protocol

from owlbank import callstate, coldstart, humanleg, memory_http, obs, segments, sensitive
from owlbank.callstate import CallState
from owlbank.pii import (
    COMPREHEND_CONNECT_TIMEOUT_S,
    COMPREHEND_MAX_BYTES,
    COMPREHEND_READ_TIMEOUT_S,
    PiiGuard,
)
from owlbank.recap import (
    MAX_OBSERVATION_CHARS,
    MAX_OBSERVATIONS,
    MAX_SUMMARY_CHARS,
    Skip,
    bounded,
    has_factor_value,
    recap_claim,
    recap_mark,
    recap_target,
    scrub,
    transcript_lines,
)
from owlbank.token_seal import Sealer

log = logging.getLogger(__name__)

API_KEY_SECRET_ENV: Final = "OWLBANK_SECRET_TWILIO_API_KEY"  # noqa: S105 - a variable name
MEMORY_STORE_ENV: Final = "OWLBANK_MEMORY_STORE_ID"
MODEL_ID_ENV: Final = "OWLBANK_RECAP_MODEL_ID"

#: The system prompt: single-language, English.
CATALOG_KEY: Final = "agent.recap.system"
PROMPT_LANG: Final = "en-US"
TOOL_NAME: Final = "write_recap"
#: The ``source`` of every item this writer puts in Memory.
SOURCE: Final = "owlbank-voice"
#: The timeout of each Memory and Orchestrator request, and of the model call.
REQUEST_TIMEOUT_S: Final = 3.0
MODEL_TIMEOUT_S: Final = 15.0
MODEL_MAX_TOKENS: Final = 1024
#: What one step can take at most, from its client's configuration (the step checks are against
#: these). A Memory (``requests``) or Orchestrator (``urllib``) request: the timeout is the connect
#: timeout and again the read timeout, one attempt.
REQUEST_BOUND_S: Final = 2 * REQUEST_TIMEOUT_S
#: The model call: client_for_budget sets connect and read to the budget, one attempt.
MODEL_BOUND_S: Final = 2 * MODEL_TIMEOUT_S
#: build_deps' KMS client (connect, read, total attempts); a blob's unseal is bounded by all of them.
KMS_CONNECT_TIMEOUT_S: Final = 1
KMS_READ_TIMEOUT_S: Final = 2
KMS_ATTEMPTS: Final = 2
UNSEAL_BOUND_S: Final = KMS_ATTEMPTS * (KMS_CONNECT_TIMEOUT_S + KMS_READ_TIMEOUT_S)
#: owlbank.pii's Comprehend client: ``retries={"max_attempts": 1}`` is botocore's legacy setting,
#: one retry after the first attempt, so two attempts in all.
COMPREHEND_ATTEMPTS: Final = 2
#: A Comprehend request, per chunk of the text checked.
PII_BOUND_S: Final = COMPREHEND_ATTEMPTS * (COMPREHEND_CONNECT_TIMEOUT_S + COMPREHEND_READ_TIMEOUT_S)
#: The run's time without a Lambda context (tests, scripts): the function's 60 s timeout.
DEFAULT_RUN_S: Final = 60.0
#: Kept back from the deadline to record the outcome (one conditional call-state write).
DEADLINE_MARGIN_S: Final = 2.0
CUSTOMERS: Final = "customers"

# Outcomes.
WRITTEN: Final = "written"
SKIPPED: Final = "skipped"
FAILED: Final = "failed"

# Reasons (``not_verified`` and ``no_identifier`` come from owlbank.recap).
SWITCH_OFF: Final = "switch_off"
REDACTION_OFF: Final = "redaction_off"
REDACTION_FAILED: Final = "redaction_failed"
NO_STATE: Final = "no_state"
NO_PROFILE: Final = "no_profile"
AMBIGUOUS_PROFILE: Final = "ambiguous_profile"
NO_TRANSCRIPT: Final = "no_transcript"
MODEL_ERROR: Final = "model_error"
MODEL_TIMEOUT: Final = "model_timeout"
MODEL_MALFORMED: Final = "model_malformed"
MEMORY_ERROR: Final = "memory_error"
ORCHESTRATOR_ERROR: Final = "orchestrator_error"
PARTIAL_WRITE: Final = "partial_write"
STATE_ERROR: Final = "state_error"
NOT_CLAIMED: Final = "not_claimed"
CLAIM_LOST: Final = "claim_lost"
DEADLINE: Final = "deadline"

_CALL_SID_RE = re.compile(r"^CA[0-9a-fA-F]{32}$")
_TIMEOUT_ERRORS = frozenset({"ReadTimeoutError", "ConnectTimeoutError", "ConnectTimeout", "ReadTimeout"})


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one run did: ``outcome`` (``written``/``skipped``/``failed``), the reason when not
    written, and the Memory profile ID when one was found."""

    outcome: str
    reason: str | None = None
    profile_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"result": self.outcome, "reason": self.reason}


class RecapModelError(Exception):
    """The model gave no usable ``write_recap`` call; ``reason`` is ``model_error``,
    ``model_timeout`` or ``model_malformed`` (no detail: never a response body)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------- dependencies


class MemoryClient(Protocol):
    def lookup(self, id_type: str, value: str) -> Sequence[str]: ...

    def create(self, id_type: str, value: str) -> str: ...

    def write_summary(self, profile_id: str, item: Mapping[str, Any]) -> None: ...

    def write_observations(self, profile_id: str, items: Sequence[Mapping[str, Any]]) -> None: ...


class Conversations(Protocol):
    def latest_conversation(self, call_sid: str) -> str | None: ...

    def communications(self, conversation_id: str, *,
                       more: Callable[[], bool] | None = None) -> list[Mapping[str, Any]] | None: ...

    def participants(self, conversation_id: str, *,
                     more: Callable[[], bool] | None = None) -> list[Mapping[str, Any]] | None: ...


class RecapModel(Protocol):
    def recap(self, system: str, user: str) -> Mapping[str, Any]:
        """The ``write_recap`` tool input; raises :class:`RecapModelError`."""
        ...


class Catalog(Protocol):
    def text(self, key: str, lang: str, **values: Any) -> str: ...


@dataclass(frozen=True, slots=True)
class RecapDeps:
    store: Any
    policy: Any  # .memory_write and .redaction (owlbank.policy.Policy)
    pii: PiiGuard
    sealer: Sealer | None
    memory: MemoryClient
    conversations: Conversations
    model: RecapModel
    catalog: Catalog
    clock: Callable[[], float] = time.time


class HttpMemory:
    """:class:`MemoryClient` over :mod:`owlbank.memory_http` (``client`` returns a
    ``twilio.rest.Client``, built on first use)."""

    def __init__(self, client: Callable[[], Any], store_id: str, *,
                 timeout_s: float = REQUEST_TIMEOUT_S) -> None:
        self._client = client
        self._store_id = store_id
        self._timeout_s = timeout_s

    def lookup(self, id_type: str, value: str) -> list[str]:
        return memory_http.lookup_profile_by(self._client(), self._store_id, id_type, value,
                                             timeout_s=self._timeout_s)

    def create(self, id_type: str, value: str) -> str:
        """Identity resolution upserts on the identifier trait (``POST Profiles``, the bridge's
        ``create_profile`` through TAC): only that one trait is sent, never the ANI."""
        _, doc = memory_http.request(
            self._client(), "POST", f"{memory_http.store_url(self._store_id)}/Profiles",
            {"traits": {"Contact": {id_type: value}}}, what="profile create", timeout_s=self._timeout_s)
        profile_id = doc.get("id") if isinstance(doc, Mapping) else None
        if not isinstance(profile_id, str) or not profile_id:
            raise memory_http.MemoryApiError("Memory profile create answered without a profile ID",
                                             status=200)
        return profile_id

    def write_summary(self, profile_id: str, item: Mapping[str, Any]) -> None:
        memory_http.create_summaries(self._client(), self._store_id, profile_id, [item],
                                     timeout_s=self._timeout_s)

    def write_observations(self, profile_id: str, items: Sequence[Mapping[str, Any]]) -> None:
        memory_http.create_observations(self._client(), self._store_id, profile_id, items,
                                        timeout_s=self._timeout_s)


def tool_spec() -> dict[str, Any]:
    """The forced ``write_recap`` tool."""
    return {"toolSpec": {
        "name": TOOL_NAME,
        "description": "Record the recap of this call for the next agent.",
        "inputSchema": {"json": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "maxLength": MAX_SUMMARY_CHARS,
                            "description": "Why the customer called and the outcome."},
                "observations": {"type": "array", "maxItems": MAX_OBSERVATIONS,
                                 "items": {"type": "string", "maxLength": MAX_OBSERVATION_CHARS},
                                 "description": "Topics and open follow-ups, one fact each."},
            },
            "required": ["summary", "observations"],
        }},
    }}


class BedrockRecapModel:
    """:class:`RecapModel` over ``bedrock-runtime`` Converse with ``toolChoice`` forced to
    ``write_recap`` (the slot resolver's pattern). ``client`` has the 15 s timeouts and one attempt
    (:func:`owlbank.slots.resolver.client_for_budget`)."""

    def __init__(self, client: Any, *, model_id: str | None = None,
                 max_tokens: int = MODEL_MAX_TOKENS) -> None:
        from owlbank.slots.resolver import DEFAULT_MODEL_ID

        self._client = client
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.max_tokens = max_tokens

    def request(self, system: str, user: str) -> dict[str, Any]:
        return {
            "modelId": self.model_id,
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "toolConfig": {"tools": [tool_spec()], "toolChoice": {"tool": {"name": TOOL_NAME}}},
            "inferenceConfig": {"maxTokens": self.max_tokens},
        }

    def recap(self, system: str, user: str) -> Mapping[str, Any]:
        try:
            with obs.timed_call("bedrock", "converse"):
                response = self._client.converse(**self.request(system, user))
        except Exception as exc:
            timeout = isinstance(exc, TimeoutError) or type(exc).__name__ in _TIMEOUT_ERRORS
            raise RecapModelError(MODEL_TIMEOUT if timeout else MODEL_ERROR) from exc
        return parse_response(response)


def parse_response(response: Any) -> Mapping[str, Any]:
    """The input of the one ``write_recap`` call in a Converse response; :class:`RecapModelError`
    ``model_malformed`` for anything else (another stop reason, no or several tool calls, another
    tool, input that isn't an object)."""
    if not isinstance(response, Mapping) or response.get("stopReason") != "tool_use":
        raise RecapModelError(MODEL_MALFORMED)
    output = response.get("output")
    message = output.get("message") if isinstance(output, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, list):
        raise RecapModelError(MODEL_MALFORMED)
    uses = [b.get("toolUse") for b in content if isinstance(b, Mapping) and "toolUse" in b]
    if len(uses) != 1 or not isinstance(uses[0], Mapping) or uses[0].get("name") != TOOL_NAME:
        raise RecapModelError(MODEL_MALFORMED)
    raw = uses[0].get("input")
    if not isinstance(raw, Mapping):
        raise RecapModelError(MODEL_MALFORMED)
    return raw


def user_message(transcript: str) -> str:
    """The transcript as untrusted data in a delimited block. Angle brackets in it are
    replaced, so no text can close the block early."""
    safe = transcript.replace("<", "‹").replace(">", "›")
    return f"<transcript>\n{safe}\n</transcript>\nCall {TOOL_NAME} with the recap of this call."


# --------------------------------------------------------------------------- the run


@dataclass(slots=True)
class _Progress:
    """The step a run is in: the reason recorded if it raises, and the profile found so far."""

    reason: str = STATE_ERROR
    profile_id: str | None = None
    summary_written: bool = False
    owner: bool = True


@dataclass(frozen=True, slots=True)
class _Budget:
    """The run's deadline on ``clock``'s scale."""

    deadline: float
    clock: Callable[[], float]

    def fits(self, bound_s: float) -> bool:
        """A step bounded by ``bound_s`` ends before the deadline, the margin kept."""
        return self.clock() + bound_s <= self.deadline - DEADLINE_MARGIN_S


class _PageGate:
    """The ``more`` of an Orchestrator read: one more page only while its request fits; remembers
    whether it said no, so a read the deadline cut is told from a failed one."""

    def __init__(self, budget: _Budget) -> None:
        self._budget = budget
        self.cut = False

    def __call__(self) -> bool:
        if self._budget.fits(REQUEST_BOUND_S):
            return True
        self.cut = True
        return False


def _pii_bound(text: str) -> float:
    """The PII check's bound for ``text``: one Comprehend request per chunk."""
    return PII_BOUND_S * max(1, -(-len(text.encode()) // COMPREHEND_MAX_BYTES))


def _now(deps: RecapDeps) -> int:
    return int(deps.clock())


def _log_line(call_sid: str, outcome: Outcome) -> None:
    """One line per call, identifiers and the reason only."""
    if outcome.outcome == WRITTEN:
        obs.emit(log, logging.INFO, "recap_written", call_sid=call_sid, profile_id=outcome.profile_id,
                 outcome=WRITTEN)
        return
    obs.emit(log, logging.WARNING if outcome.outcome == FAILED else logging.INFO, "recap_skipped",
             call_sid=call_sid, profile_id=outcome.profile_id, outcome=outcome.outcome,
             reason=outcome.reason)


def _claim(deps: RecapDeps, call_sid: str) -> tuple[CallState, int] | None:
    """Claim the call's recap; ``None`` when another writer holds it or it is done. Raises
    ``KeyError`` for a call with no state, and store errors (before the claim a raise is allowed)."""
    taken: list[dict[str, Any] | None] = [None]
    now = _now(deps)

    def fn(s: CallState) -> Mapping[str, Any] | None:
        taken[0] = recap_claim(s.recap, now)
        return None if taken[0] is None else {callstate.F_RECAP: taken[0]}

    state = callstate.mutate(deps.store, call_sid, fn, now=deps.clock)
    claim = taken[0]
    return None if claim is None else (state, int(claim["claimed_at"]))


def _mark(deps: RecapDeps, call_sid: str, claimed_at: int) -> bool:
    marked = [False]
    now = _now(deps)

    def fn(s: CallState) -> Mapping[str, Any] | None:
        new = recap_mark(s.recap, claimed_at, now)
        marked[0] = new is not None
        return None if new is None else {callstate.F_RECAP: new}

    callstate.mutate(deps.store, call_sid, fn, now=deps.clock)
    return marked[0]


def _record(deps: RecapDeps, call_sid: str, claimed_at: int, outcome: Outcome) -> None:
    """Write the outcome into ``recap`` if this writer still holds the claim. Never raises."""
    now = _now(deps)

    def fn(s: CallState) -> Mapping[str, Any] | None:
        recap = s.recap
        if not isinstance(recap, Mapping) or recap.get("claimed_at") != claimed_at \
                or recap.get("outcome") is not None:
            return None  # taken over (or done) meanwhile: the holder records its own
        new: dict[str, Any] = {**dict(recap), "outcome": outcome.outcome, "at": now}
        if outcome.reason is not None:
            new["reason"] = outcome.reason
        if outcome.profile_id is not None:
            new["profile_id"] = outcome.profile_id
        return {callstate.F_RECAP: new}

    try:
        callstate.mutate(deps.store, call_sid, fn, now=deps.clock)
    except Exception as exc:  # noqa: BLE001 - the outcome is logged either way
        obs.log_caught(log, "recap_outcome_not_recorded", exc, call_sid=call_sid,
                       operation="calls.mutate", decision="ignored_logged")


def _recorded_conversation(state: CallState) -> str | None:
    recorded = (state.escalation or {}).get(humanleg.K_CONVERSATION_ID)
    return recorded if isinstance(recorded, str) and recorded else None


def _latest_conversation(state: CallState, conversations: Conversations) -> str | None:
    found = conversations.latest_conversation(state.call_sid)
    return found if isinstance(found, str) and found else None


def _stamp(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(call_sid: str, state: CallState, claimed_at: int, deps: RecapDeps, p: _Progress,
         budget: _Budget) -> Outcome:
    def deadline() -> Outcome:
        return Outcome(FAILED, DEADLINE, p.profile_id)

    customer = None
    if state.verified_customer_id and state.level >= 1:
        customer = deps.store.get(CUSTOMERS, {"customer_id": state.verified_customer_id})
    target = recap_target(state, customer)
    if isinstance(target, Skip):
        return Outcome(SKIPPED, target.reason)

    # The transcript before the profile: a call with nothing to recap never creates one.
    # The conversation recorded at escalation, else (a call that had an AI session) the newest one
    # Orchestrator lists for the CallSid, whatever its status (by now it may
    # be INACTIVE or CLOSED); a call with no AI session has none.
    p.reason = ORCHESTRATOR_ERROR
    conversation_id = _recorded_conversation(state)
    if conversation_id is None and state.ai_sessions:
        if not budget.fits(REQUEST_BOUND_S):
            return deadline()
        conversation_id = _latest_conversation(state, deps.conversations)
    if conversation_id is None:
        return Outcome(SKIPPED, NO_TRANSCRIPT)
    # Each page is asked for only while its request fits (not a page count worked out up front, so
    # a normal call reads all it has); a read the deadline cut is ``deadline``, any other None
    # (a failure or READ_MAX_PAGES) ``orchestrator_error``.
    gate = _PageGate(budget)
    communications = deps.conversations.communications(conversation_id, more=gate)
    if communications is None:
        return deadline() if gate.cut else Outcome(FAILED, ORCHESTRATOR_ERROR)
    participants = deps.conversations.participants(conversation_id, more=gate)
    if participants is None:
        return deadline() if gate.cut else Outcome(FAILED, ORCHESTRATOR_ERROR)
    lines = transcript_lines(communications, participants)
    if not lines:
        return Outcome(SKIPPED, NO_TRANSCRIPT)

    # The sealed factor values, one blob at a time while its unseal fits (sensitive.restore of one
    # blob is the same as of all of them, blob by blob); the first that fails stops the run.
    p.reason = REDACTION_FAILED
    if state.sealed_sensitive:
        if deps.sealer is None:
            return Outcome(SKIPPED, REDACTION_FAILED)
        for blob in state.sealed_sensitive:
            if not budget.fits(UNSEAL_BOUND_S):
                return deadline()
            sensitive.restore(call_sid, [blob], deps.sealer)
            if sensitive.is_unverifiable(call_sid):
                return Outcome(SKIPPED, REDACTION_FAILED)
    factors = segments.sensitive_values(call_sid)

    # Looked up now, so a call with no profile (phone) or two stops early; created only just before
    # the write (below), so a later failure never leaves an empty profile.
    p.reason = MEMORY_ERROR
    if not budget.fits(REQUEST_BOUND_S):
        return deadline()
    ids = list(dict.fromkeys(deps.memory.lookup(target.id_type, target.value)))
    if len(ids) > 1:
        return Outcome(SKIPPED, AMBIGUOUS_PROFILE)
    if not ids and not target.create:
        return Outcome(SKIPPED, NO_PROFILE)  # Never create a phone profile
    p.profile_id = profile_id = ids[0] if ids else None

    p.reason = REDACTION_FAILED
    joined = "\n".join(lines)
    if not budget.fits(_pii_bound(joined)):
        return deadline()
    transcript = scrub(joined, factor_values=factors, pii=deps.pii, language=state.language)
    if transcript is None:
        return Outcome(SKIPPED, REDACTION_FAILED, profile_id)

    p.reason = MODEL_ERROR
    if not budget.fits(MODEL_BOUND_S):
        return deadline()
    system = deps.catalog.text(CATALOG_KEY, PROMPT_LANG)
    try:
        raw = deps.model.recap(system, user_message(transcript))
    except RecapModelError as exc:
        obs.log_caught(log, "recap_model_failed", exc, call_sid=call_sid, reason=exc.reason,
                       operation="bedrock.converse", decision="record_failed")
        return Outcome(FAILED, exc.reason, profile_id)
    if not isinstance(raw, Mapping):
        return Outcome(FAILED, MODEL_MALFORMED, profile_id)

    # Scrub each piece, then bound it (scrub first: a cut never decides what the scrub sees).
    p.reason = REDACTION_FAILED
    summary = raw.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return Outcome(FAILED, MODEL_MALFORMED, profile_id)
    items = raw.get("observations")
    wanted = [i for i in items if isinstance(i, str) and i.strip()][:MAX_OBSERVATIONS] \
        if isinstance(items, list | tuple) else []
    if not budget.fits(_pii_bound(summary)):
        return deadline()
    clean_summary = scrub(summary, factor_values=factors, pii=deps.pii, language=PROMPT_LANG)
    if clean_summary is None:
        return Outcome(SKIPPED, REDACTION_FAILED, profile_id)
    clean_items: list[str] = []
    for item in wanted:
        if not budget.fits(_pii_bound(item)):
            return deadline()
        clean = scrub(item, factor_values=factors, pii=deps.pii, language=PROMPT_LANG)
        if clean is None:
            return Outcome(SKIPPED, REDACTION_FAILED, profile_id)
        clean_items.append(clean)
    recap = bounded({"summary": clean_summary, "observations": clean_items})
    if recap is None:
        return Outcome(FAILED, MODEL_MALFORMED, profile_id)
    if any(has_factor_value(t, factors) for t in (recap.summary, *recap.observations)):
        return Outcome(SKIPPED, REDACTION_FAILED, profile_id)  # a cut that formed a value

    # The last check: the profile create (if any) and both writes must fit, since a started write
    # is never stopped.
    creates = 0 if profile_id is not None else 1
    if not budget.fits(REQUEST_BOUND_S * (creates + (2 if recap.observations else 1))):
        return deadline()
    if profile_id is None:
        p.reason = MEMORY_ERROR
        p.profile_id = profile_id = deps.memory.create(target.id_type, target.value)
    p.reason = STATE_ERROR
    if not _mark(deps, call_sid, claimed_at):
        p.owner = False
        return Outcome(SKIPPED, CLAIM_LOST, profile_id)

    p.reason = MEMORY_ERROR
    occurred_at = _stamp(_now(deps))
    deps.memory.write_summary(profile_id, {"content": recap.summary, "conversationId": conversation_id,
                                           "source": SOURCE, "occurredAt": occurred_at})
    p.summary_written = True
    if recap.observations:
        p.reason = PARTIAL_WRITE
        deps.memory.write_observations(profile_id, [
            {"content": text, "source": SOURCE, "conversationIds": [conversation_id],
             "occurredAt": occurred_at} for text in recap.observations])
    return Outcome(WRITTEN, None, profile_id)


def write_recap(call_sid: str, deps: RecapDeps, *, deadline: float | None = None) -> Outcome:
    """Write the recap of ``call_sid``'s call, at most once (module docstring), by ``deadline`` (on
    ``deps.clock``'s scale; default :data:`DEFAULT_RUN_S` from now). Raises only before the claim
    (policy or store errors); after it, every failure is recorded and logged."""
    budget = _Budget(deps.clock() + DEFAULT_RUN_S if deadline is None else deadline, deps.clock)
    if not deps.policy.memory_write:
        outcome = Outcome(SKIPPED, SWITCH_OFF)
        _log_line(call_sid, outcome)
        return outcome
    if not deps.policy.redaction.enabled:
        outcome = Outcome(SKIPPED, REDACTION_OFF)
        _log_line(call_sid, outcome)
        return outcome
    try:
        claimed = _claim(deps, call_sid)
    except KeyError:  # no call state (expired or never written): nothing to claim, no retry helps
        outcome = Outcome(SKIPPED, NO_STATE)
        _log_line(call_sid, outcome)
        return outcome
    if claimed is None:
        obs.emit(log, logging.DEBUG, "recap_not_claimed", call_sid=call_sid, decision="stop")
        return Outcome(SKIPPED, NOT_CLAIMED)
    state, claimed_at = claimed

    progress = _Progress()
    sensitive.forget(call_sid)  # unseal afresh: nothing left over from an earlier run here
    try:
        outcome = _run(call_sid, state, claimed_at, deps, progress, budget)
    except Exception as exc:  # noqa: BLE001 - after the claim nothing is re-raised
        reason = PARTIAL_WRITE if progress.summary_written else progress.reason
        obs.log_caught(log, "recap_failed", exc, call_sid=call_sid, profile_id=progress.profile_id,
                       reason=reason, decision="record_failed")
        outcome = Outcome(FAILED, reason, progress.profile_id)
    finally:
        sensitive.forget(call_sid)
    if progress.owner:
        _record(deps, call_sid, claimed_at, outcome)
    _log_line(call_sid, outcome)
    return outcome


# --------------------------------------------------------------------------- wiring


def _once(factory: Callable[[], Any]) -> Callable[[], Any]:
    """``factory()`` on first use, then reused (the API-key secret is read once)."""
    lock = threading.Lock()
    box: list[Any] = []

    def get() -> Any:
        if not box:
            with lock:
                if not box:
                    box.append(factory())
        return box[0]

    return get


def build_deps(env: Mapping[str, str] | None = None, *, store: Any | None = None) -> RecapDeps:
    """Production dependencies: no secret read and no Twilio, Bedrock, KMS or Comprehend call until
    first use (the policy and catalog are read now)."""
    import boto3
    from botocore.config import Config

    from lambdas.auth_service.verify_client import twilio_client_factory
    from owlbank.catalog import PromptCatalog
    from owlbank.config import Settings
    from owlbank.humanleg_client import OrchestratorHumanLeg, credentials_from_secret
    from owlbank.pii import ComprehendPiiRedactor, make_comprehend_client
    from owlbank.policy import Policy
    from owlbank.slots.resolver import client_for_budget
    from owlbank.store import DynamoStore
    from owlbank.token_seal import KmsSealer

    env = dict(os.environ) if env is None else env
    settings = Settings.from_env(env)
    secret = (settings.secrets.twilio_api_key or "").strip()
    if not secret:
        raise ValueError(f"{API_KEY_SECRET_ENV} is required")
    store_id = (env.get(MEMORY_STORE_ENV) or "").strip()
    if not store_id:
        raise ValueError(f"{MEMORY_STORE_ENV} is required")
    if store is None:
        store = DynamoStore.from_settings(settings)
    policy = Policy.from_store(store, ttl_s=settings.cache_ttl_s)
    catalog = PromptCatalog.from_store(store, ttl_s=settings.cache_ttl_s)
    region = settings.aws_region
    pii = PiiGuard.for_policy(
        ComprehendPiiRedactor(coldstart.LazyClient(lambda: make_comprehend_client(region))), policy)
    sealer: Sealer | None = None
    if settings.token_kms_key:
        quick = Config(connect_timeout=KMS_CONNECT_TIMEOUT_S, read_timeout=KMS_READ_TIMEOUT_S,
                       retries={"total_max_attempts": KMS_ATTEMPTS})
        sealer = KmsSealer(coldstart.LazyClient(lambda: boto3.client("kms", region_name=region, config=quick)),
                           settings.token_kms_key)
    model_client = client_for_budget(region_name=region)(int(MODEL_TIMEOUT_S * 1000))
    return RecapDeps(
        store=store, policy=policy, pii=pii, sealer=sealer,
        memory=HttpMemory(_once(twilio_client_factory(secret)), store_id),
        conversations=OrchestratorHumanLeg(credentials_from_secret(secret), timeout_s=REQUEST_TIMEOUT_S),
        model=BedrockRecapModel(model_client, model_id=(env.get(MODEL_ID_ENV) or "").strip() or None),
        catalog=catalog)


_deps: RecapDeps | None = None


def _remaining_s(context: Any) -> float | None:
    """The time Lambda leaves this invocation, or None without a context (tests, scripts)."""
    get = getattr(context, "get_remaining_time_in_millis", None)
    return get() / 1000 if callable(get) else None


@obs.instrumented("recap_writer", op="recap")
def handler(event: Any, context: Any = None) -> dict[str, Any]:
    global _deps
    call_sid = event.get("call_sid") if isinstance(event, Mapping) else None
    if not isinstance(call_sid, str) or not _CALL_SID_RE.match(call_sid):
        obs.emit(log, logging.WARNING, "recap_bad_request", decision="ignored")
        return {"result": "bad_request", "reason": "call_sid"}
    if _deps is None:
        _deps = build_deps()
    left = _remaining_s(context)
    deadline = None if left is None else _deps.clock() + left
    with obs.call_context(call_sid):
        return write_recap(call_sid, _deps, deadline=deadline).as_dict()


# Inside Lambda only (a no-op in tests and scripts): wire up now, in the INIT phase.
_deps = coldstart.build_at_init(build_deps)


__all__ = ["BedrockRecapModel", "HttpMemory", "Outcome", "RecapDeps", "RecapModelError", "build_deps",
           "handler", "parse_response", "tool_spec", "user_message", "write_recap"]
