"""The front door: deterministic TwiML flows around the AI stage.

Routes (API Gateway REST proxy, all ``POST``, all signed by Twilio and Basic-authenticated):

* ``/voice/incoming``: create the call state, ANI lookup, language ``<Gather>``.
* ``/voice/language``: record the language, ask the level-1 factors (or fall back).
* ``/voice/factor``: resolve each answer with the slot engine, verify, continue,
  connect ConversationRelay, or return an envelope to the AI.
* ``/voice/connect-action``: the ``<Connect action>`` callback: step-up, deterministic
  tools (``get_balance``), deterministic confirmation, escalation (``<Enqueue>``),
  WebSocket failure.
* ``/voice/status``: call end: revoke tokens, mark the call ended, journal ``call_end``; in
  ``agentcore``, stop the call's Runtime session once (:mod:`.runtime_sessions`).
* ``/voice/partial``: a digit Gather's, or a choice question's (the card question and its follow-ups
  that list options), ``partialResultCallback`` (answered 204): when a partial
  already holds the whole answer, claim the Gather, seal the answer and end the Gather through a
  Twilio REST call update that redirects to ``/voice/factor?early=<id>``. One answer per Gather: see :mod:`lambdas.frontdoor.gathers`.
* ``/voice/enqueue-action``: the ``<Enqueue>`` ``action`` (the
  caller left the queue). It never answers with an escalation and doesn't act on how long the
  caller has waited (no queue time limit). The ``<Enqueue>`` has no ``waitUrl``: Twilio plays
  its default hold music.
* ``/voice/orchestrator`` (``agentcore`` only): the
  Conversation Configuration's status callback. Basic (the WAF has challenged) and then the
  signature (:func:`lambdas.frontdoor.http.authenticate_callback`), then
  ``orchestrator_event_dropped`` (``event_type``, ``lag_ms``, ``reason=no_relay``) and an empty
  200: no Runtime call, no call-state write, and never the body, a credential or a header value
  in a log line. On ``fargate`` the route doesn't exist (403, ``route``).

Every handler is wrapped: an unexpected error gives the catalog apology plus a
fallback (escalate, else goodbye), never an exception to Twilio.
Factor values are never logged, stored or echoed. A sensitive slot's value
 is sealed into call state and registered for the log scrub the moment it is parsed
(``sensitive.remember``), and every call-state load registers the values another instance
sealed (``sensitive.restore``). Its answering turn reaches the AI as one text with the
slot's ``[<name>]`` placeholder. The escalation summary is PII-checked again
here (defence in depth), and every ConversationRelay connect carries a fresh
one-time nonce, stored only as a hash: in its URL on ``fargate``, as
``<Parameter name="n">`` on ``agentcore``.

**Caller-ID trust.** ``StirVerstat`` from the incoming webhook is kept in
call state with an ``ani_trusted`` flag and journaled. It is passed to every auth
service ``verify``, which applies the gate (whether ``ani_match`` counts, and that an
untrusted ANI never feeds the customer lockout). The front door only uses the same
policy rule (:meth:`Policy.effective_factors`) to know which factors to ask; when no
new level is reached by one factor, it holds it sealed (as for disambiguation)
and asks the next one.

**Untrusted-attempt cap.** For an untrusted caller ID the front door
asks the auth service's ``precheck`` before the first factor of a login or step-up, and
``verify`` applies the same gate. Either answering ``escalate`` sends the call to a human
with reason ``verification_risk`` (``<Enqueue>``, the neutral catalog phrase
``escalation.verification_risk``); no further factor is asked. The call's level, verified
customer and factors are cleared first, so the task carries ``auth_level`` 0 and the
agent verifies from scratch. A one-time code is never sent to an untrusted caller before
a token exists: that factor escalates instead.

**One-time code to the calling number only.** The front door passes the
call's ANI and language to the auth service's ``send_otp``. When it answers ``escalate``
(the ANI is withheld, malformed or not on file for the verified customer, or Verify can't
deliver) no code is sent; the caller hears the neutral phrase ``escalation.otp_unavailable``
and is enqueued with reason ``otp_unavailable`` (default support queue). Unlike
``verification_risk``, the call's level and verified customer are kept: the caller did
verify up to the level reached.

**Deterministic confirmation.** A tool whose policy ``confirm`` is
``deterministic`` comes back from the bridge as a ``confirm`` handoff with the model's
call in ``pending_tool``. The front door re-validates the arguments, checks the token,
reads the action back from ``confirm.<tool>`` and gathers a ``yes_no`` slot. Only on a
recorded yes does it run the tool on ``bank_tools`` (token unsealed, the
``tool_use_id`` as idempotency key), then speaks ``confirm.<tool>.done`` and
returns ``await_caller``, as ``get_balance`` does. No, no answer, retries exhausted or
cancel: the tool doesn't run and the model gets ``CANCELLED_BY_CALLER``. An intent
switch returns ``interrupted``/``handle_utterance``.

**"Got it." while the code is sent.** When a verified factor is acknowledged and the
next factor is the one-time code, the front door doesn't wait for the send (Twilio Verify through
the auth service, about 0.6 s). It clears any stale ``otp_send``, starts ``send_otp``
asynchronously (``Event`` invoke, ``record``) and answers at once with only ``<Say>Got it.</Say>``
and a ``<Redirect>`` back to ``/voice/factor``, marking the flow ``otp_wait``. The redirect (no
``Digits``, no ``SpeechResult``) polls the call state for the auth service's ``otp_send`` for up to
:data:`OTP_WAIT_S` since the send started, then clears both and: ``sent`` asks the code (no second
send, no second ack); ``escalate`` is ``otp_unavailable``; ``error`` or no answer in time is the
error path. If the asynchronous invoke can't be queued, the send is synchronous, as before.
The untrusted-caller gate (no code before login) runs before either.

**Deterministic step-up shortcut.** A ``step-up`` hand-off whose ``pending_tool`` is
``get_balance`` or a deterministic-confirm tool (the bridge refused it only for the level) starts
the step-up for that tool. Once verified, the front door goes straight to the tool's scripted
step, with no return to the AI: the confirmation read-back (after ``stepup.success``) or the
balance read-out. The envelope answers the model's original call, and its ``spoken`` list starts
with the step-up's prompts. A failed, declined or interrupted step-up returns the usual step-up
status with ``tool`` = the pending tool.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import math
import os
import re
import secrets as _stdlib_secrets
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, Protocol

from owlbank import (
    briefing,
    briefing_transcript,
    callstate,
    humanleg,
    last_spoken,
    obs,
    recap_trigger,
    segments,
    sensitive,
    task_attributes,
    tokens,
    twiml,
    web_activity,
    ws_nonce,
)
from owlbank.basic_auth import BasicSecret
from owlbank.callstate import CallState
from owlbank.catalog import PromptCatalog
from owlbank.envelope import CANCELLED_BY_CALLER, ResultEnvelope, Spoken
from owlbank.humanleg_client import HumanLegClient
from owlbank.humanleg_client import budget as human_leg_budget
from owlbank.humanleg_client import budget_left as human_leg_budget_left
from owlbank.pii import ON_FAILURE_WITHHOLD, WITHHELD_TURN, PiiGuard
from owlbank.policy import ESTABLISHED_FACTORS, Policy
from owlbank.recap_trigger import RecapTrigger
from owlbank.security_log import auth_failed
from owlbank.segments import SegmentLog
from owlbank.sids import CALL_SID_RE
from owlbank.slot_policy import MODES, Mode, SlotSettings, gather_input, is_mode
from owlbank.slots import Resolver, early_digits, early_phone, hints
from owlbank.slots.validators import early_choice
from owlbank.store import Store
from owlbank.task_attributes import build_task_attributes
from owlbank.telemetry import frontdoor as telemetry
from owlbank.token_seal import Sealer, UnsealError
from owlbank.tools_contract import (
    BANK_TOOLS,
    CALLER_DONE,
    CARD_SLOT,
    PHONE_SLOT,
    card_values,
    validate_args,
)
from owlbank.welcome import GREETING_KEY as OFFER_GREETING_KEY
from owlbank.welcome import MATCHED as OFFER_MATCHED
from owlbank.welcome import render_offer

from . import gathers, http, slotflow
from . import state as sessions
from .backend import BackendClient, ToolResult, parse_tool
from .calls import CallUpdater
from .relay import PresignError, RelayUrls, StaticRelayUrl
from .runtime_sessions import SessionStopper, claim_stop
from .speech import digits_words, money_words
from .state import Session

log = logging.getLogger(__name__)

WITHHELD_ANIS = frozenset({"", "anonymous", "restricted", "unknown", "private", "unavailable",
                           "+266696687", "+86282452253", "+7378742833", "+2562533"})
END_STATUSES = frozenset({"completed", "failed", "no-answer", "busy", "canceled"})
FALLBACK_ACTIONS = frozenset({"limited_ai", "escalate"})
#: The bank's default choice per outcome; ``OWLBANK_FALLBACKS`` overrides single entries.
#: A failed (or locked) and a declined login go to the AI by default.
DEFAULT_FALLBACKS: Mapping[str, str] = {
    "no_match": "limited_ai", "lookup_error": "limited_ai",
    "verify_failed": "limited_ai", "cancelled": "limited_ai", "declined": "limited_ai",
}
#: Login outcomes that close verification for the call (``login_closed``) and, when
#: they land in the AI, are welcomed with :data:`FAILED_AI_KEY` alone.
CLOSING_OUTCOMES = frozenset({"verify_failed", "declined"})
FAILED_AI_KEY = "verify.failed_ai"
_SAFE_REF = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")
MAX_REPEATS = 3
#: "Use your keypad" reminders a factor prompt gives before a spoken answer counts as a try
#: : the reminder itself is free, but a caller can't loop on it forever.
MAX_KEYPAD_REMINDERS = MAX_REPEATS
_VERSTAT = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
#: Keyed-in phone login: the call state's ``identity_source`` for a call identified
#: by a keyed-in number; the sensitive store's record name for that number once it matched (one
#: per call, read back by :meth:`FrontDoor._auth_identity`); the keyed lookups a call may make
#:  and the question asked after a miss. The slot accepts 10 digits only.
KEYED_PHONE = "keyed_phone"
IDENTITY_ANI = "ani"
KEYED_NUMBER_RECORD = "keyed_phone"
MAX_KEYED_TRIES = 2
PHONE_RETRY_KEY = "login.phone_number_retry"
_PHONE_DIGITS = re.compile(r"^[0-9]{10}$")
#: Escalation reason for the untrusted-caller gate.
VERIFICATION_RISK = "verification_risk"
OTP_UNAVAILABLE = "otp_unavailable"
#: The bridge's silence watch hands off with one of these after the caller stayed
#: silent through the repeats: ``no_input`` escalates (a policy reason), ``no_input_hangup`` says
#: ``common.goodbye_no_input`` and hangs up. The model's ``end_call`` hands off
#: with ``caller_done`` and gets the same hang-up with ``common.goodbye``.
#: ConversationRelay reports when the caller and the agent start and stop speaking, so
#: the bridge's silence wait pauses while the caller speaks and counts from the agent's speech end.
RELAY_EVENTS = "speaker-events"
NO_INPUT = "no_input"
NO_INPUT_HANGUP = "no_input_hangup"
#: The hang-up reasons and the goodbye each says (:meth:`FrontDoor._hangup`).
_HANGUP_PHRASES: Mapping[str, str] = {NO_INPUT_HANGUP: "common.goodbye_no_input",
                                      CALLER_DONE: "common.goodbye"}
#: The bridge's hand-off prefetch is used only this many seconds after it was taken.
PREFETCH_MAX_AGE_S = 30
#: How long the ``otp_wait`` redirect waits for the asynchronous send's outcome, counted
#: from when the send started, and how often it looks.
OTP_WAIT_S = 2.5
OTP_POLL_S = 0.1
#: How long a request for a Gather another request is answering waits for that
#: answer's TwiML, and how often it looks.
GATHER_REPLAY_WAIT_S = 4.0
GATHER_REPLAY_POLL_S = 0.1
#: Gather ids are letters only (no digit run in a URL or log can look like a factor).
_GATHER_ID = re.compile(r"^[A-Za-z]{16}$")
_GATHER_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
#: Balance fields that must be decimal strings for a prefetched result to be read out.
_BALANCE_AMOUNTS = ("available", "current", "pending")
#: AgentCore hosting: AgentCore closes a Runtime WebSocket at about 3,657 s (code 1008), and
#: ConversationRelay reports that to ``<Connect action>`` only as ``failed`` with this ErrorCode and
#: no ``HandoffData``, like any dropped socket. A close that old (since the URL was minted) is the
#: cap: the call reconnects silently, at most ``MAX_STREAM_RECONNECTS`` times (Twilio's default
#: 4 h ``timeLimit``).
STREAM_LIMIT_ERROR_CODE = "64105"
STREAM_LIMIT_MIN_AGE_S = 3540
MAX_STREAM_RECONNECTS = 4
CloseKind = Literal["stream_limit", "other"]


@dataclass(frozen=True, slots=True)
class GeneratedSpoken(Spoken):
    """A prompt whose text the model wrote and :func:`owlbank.slots.generative.check` accepted.
    ``template`` is the catalog key it replaced; the ``prompt``
    journal event gets ``generated: true``. ``fallback`` is the catalog template's verb and
    spoken text, said instead when the response doesn't parse as XML."""

    fallback: tuple[Any, Spoken] = field(compare=False, repr=False)


def twiml_parses(xml: str) -> bool:
    """The rendered response parses as XML (``xml.etree.ElementTree``; never repaired)."""
    try:
        ET.fromstring(xml)  # noqa: S314 - our own rendered response, not untrusted input
    except ET.ParseError:
        return False
    return True


# --------------------------------------------------------------------------- config


#: ``welcomeGreetingInterruptible`` values (see the ConversationRelay TwiML reference).
BARGE_IN_MODES = ("any", "speech", "dtmf", "none")
#: Where the bridge runs: ``OWLBANK_HOSTING`` absent or empty is ``fargate``.
HOSTING_ENV = "OWLBANK_HOSTING"
HOSTINGS = ("fargate", "agentcore")
#: The escalation briefing's overall deadline in ms (absent or empty is the default).
BRIEFING_BUDGET_ENV = "OWLBANK_BRIEFING_BUDGET_MS"
MIN_BRIEFING_BUDGET_MS = 100
#: The front door's Lambda timeout in ms, used when the handler passes no remaining-time clock (keep
#: equal to ``TIMEOUTS["frontdoor"]`` in infra/owlbank_infra/compute_stack.py; a test checks it).
REQUEST_TIMEOUT_MS = 10_000
#: Left of the request's time for the render, the journal write and the response after the briefing.
BRIEFING_MARGIN_MS = 1500
#: Below this much budget the model is not called at all (the fixed line is spoken).
BRIEFING_FLOOR_MS = 300
MAX_BRIEFING_BUDGET_MS = 5000
#: The transcript read and scrub share at most this much of the briefing budget (and at most
#: :data:`BRIEFING_TRANSCRIPT_SHARE` of it); the model gets the rest.
BRIEFING_TRANSCRIPT_MS = 1200
BRIEFING_TRANSCRIPT_SHARE = 0.4
#: Catalog keys of the spoken briefing (a lead-in, the model's clause, the fixed line, and the labels).
BRIEFING_LEAD_KEY = "escalate.briefing.lead"
BRIEFING_CLAUSE_KEY = "escalate.briefing.clause"
BRIEFING_FALLBACK_KEY = "escalate.briefing.fallback"
BRIEFING_LEVEL_PREFIX = "escalate.briefing.level."
BRIEFING_REASON_PREFIX = "escalate.briefing.reason."
BRIEFING_REASON_DEFAULT = "default"
#: ``escalation.destination`` of a call that was read its briefing instead of being queued.
BRIEFING_DESTINATION = "briefing"
#: The slot mode ceiling: absent or empty is ``assisted``.
SLOT_MODE_CEILING_ENV = "OWLBANK_SLOT_MODE_CEILING"
#: The speech log of Gather turns (``on`` | ``off``, absent or empty is ``on``) and its
#: redaction (``on`` | ``off``, absent or empty is ``off``).
SPEECH_LOG_ENV = "OWLBANK_SPEECH_LOG"
SPEECH_LOG_REDACTION_ENV = "OWLBANK_SPEECH_LOG_REDACTION"
#: The Memory trait groups (the bridge's and the auth service's setting too).
MEMORY_TRAIT_GROUPS_ENV = "OWLBANK_MEMORY_TRAIT_GROUPS"
#: AI keypad shortcuts: ``on`` | ``off``, absent or empty is ``on`` (the bridge's setting
#: too; deploy both together).
AI_KEYPAD_ENV = "OWLBANK_AI_KEYPAD"
#: The phone slot's settle wait in ms (absent or empty is the default), and its
#: ceiling (the wait, the claim and its REST update stay well inside the front door's 10 s timeout).
#: Default 0 (low latency first: a complete phone number
#: ends the Gather at once, like a ZIP, and a correction after it is not caught). 1500 lets a
#: caller correct after a pause of up to 1.5 s.
PHONE_SETTLE_ENV = "OWLBANK_PHONE_SETTLE_MS"
KEYPAD_TIMEOUT_ENV = "OWLBANK_KEYPAD_TIMEOUT"
DEFAULT_KEYPAD_TIMEOUT_S = 3
DEFAULT_PHONE_SETTLE_MS = 0
MAX_PHONE_SETTLE_MS = 3000
#: With shortcuts on, a key stops
#: the agent's speech and keys pressed during speech are still delivered (speech during agent
#: speech stays unreported, as with Twilio's default ``none``).
AI_KEYPAD_INTERRUPTIBLE = "any"
AI_KEYPAD_REPORT_INPUT = "dtmf"
#: ``trigger`` of the bridge's keypad hand-off (0 pressed: a human, no model call).
KEYPAD_TRIGGER = "keypad"
#: ``<Enqueue action>``'s ``QueueResult`` values (Twilio's Enqueue reference) the action handler
#: answers with the goodbye: the caller left the queue (``leave``, also how a task cancelled by
#: hand is expected to arrive) or the agent leg ended with the caller still on the line
#: (``bridged``). ``hangup``: nobody to speak to. The enqueue failures get the apology first. Any
#: other value (``redirected``, ``bridging-in-process``, ``redirected-from-bridged``, unknown)
#: means the call is, or is about to be, somewhere else: an empty answer, never a ``<Hangup>``.
QUEUE_LEFT_RESULTS = frozenset({"leave", "bridged"})
QUEUE_FAILED_RESULTS = frozenset({"queue-full", "error", "system-error"})


@dataclass(frozen=True, slots=True)
class FrontDoorConfig:
    """Deployment settings (no secrets). ``fallbacks`` is the bank's per-outcome choice."""

    #: ``fargate``: the bridge's ``wss://.../ws`` URL (required). ``agentcore``: unused (``None``);
    #: each ``<Connect>`` gets a pre-signed Runtime URL from ``Deps.relay``.
    relay_ws_url: str | None = None
    conversation_configuration: str | None = None
    workflow_sid: str | None = None
    language_timeout_s: int = 3
    #: Whole seconds a keypad factor or confirm Gather waits for the next key (``OWLBANK_KEYPAD_TIMEOUT``,
    #: 1-10). Default 3 s: a caller who can't enter the answer waits
    #: less before the prompt is asked again.
    factor_timeout_s: int = DEFAULT_KEYPAD_TIMEOUT_S
    #: What ends a spoken answer. An int is seconds of silence after speech. Twilio's
    #: Gather docs: a ``speechModel`` "requires" an integer ``speechTimeout`` ("Don't use auto"), so
    #: with "auto" the Gather omits ``speechModel`` (Twilio's default model) and with an int it keeps
    #: the catalog's model (Deepgram). An answer cut short arrives with too few digits and is asked
    #: again like any unclear answer. Default 3 s with Deepgram: early finish moves on as soon as all digits are heard, so the timeout only applies when the caller stops short (a 1 s timeout cuts callers off after one digit).
    speech_timeout_s: int | str = 3
    #: The same setting for spoken yes/no confirmations, which callers answer in
    #: phrases ("Yeah, that's correct"); same values and "auto" rule. Default 2 s.
    confirm_speech_timeout_s: int | str = 2
    #: The deploy-time ceiling; when false every ``factors`` slot is keypad-only.
    accept_spoken_factors: bool = True
    fallbacks: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_FALLBACKS))
    max_relay_reconnects: int = 1
    bridge_phrase_key: str = "common.one_moment"
    unverified_greeting_keys: tuple[str, ...] = ("greeting.unverified", "greeting.generic")
    interruptible: str = "speech"
    #: How the caller may interrupt a scripted read-out (balance, results) that is spoken
    #: as the ConversationRelay welcome greeting: ``any``, ``speech``, ``dtmf`` or ``none``.
    scripted_barge_in: str = "any"
    #: Reconnect ConversationRelay to the call's existing Orchestrator
    #: conversation (``conversationId``) instead of letting Orchestrator create a new one per
    #: segment. The code default is false (tests and a bare config); infra sets it through
    #: ``OWLBANK_REUSE_CONVERSATION`` (default true). The bridge's setting must match.
    reuse_conversation: bool = False
    #: A spoken digit factor ends as soon as a partial result holds a complete
    #: answer (``partialResultCallback`` + a REST call update). Needs ``Deps.calls``; off (or no
    #: updater) gives today's Gathers exactly.
    early_digits: bool = True
    #: A partial holding a complete phone number (the phone slot only) is claimed
    #: only if no newer partial for its Gather arrives within this many ms
    #: (``OWLBANK_PHONE_SETTLE_MS``, 0-3000; 0 claims at once). A caller correcting themselves is
    #: not cut off; the wait holds the partial's request (the Lambda timeout leaves room).
    phone_settle_ms: int = DEFAULT_PHONE_SETTLE_MS
    record_ttl_s: int = 86400
    test_traffic: bool = False
    #: AgentCore hosting: ``fargate`` (today's behaviour exactly) or ``agentcore`` (pre-signed Runtime
    #: URLs, the Runtime warmer on a new call's ``incoming``).
    hosting: str = "fargate"
    #: Model-assisted slots: the deploy-time mode ceiling, ``OWLBANK_SLOT_MODE_CEILING``.
    #: Every answer's mode is the frozen mode restricted by it; ``deterministic`` means no
    #: resolver is built.
    slot_mode_ceiling: Mode = "assisted"
    #: The briefing's one overall deadline in ms (``OWLBANK_BRIEFING_BUDGET_MS``); the wait on the
    #: model never exceeds it.
    briefing_budget_ms: int = briefing.BRIEFING_BUDGET_MS
    #: ``factor_answer`` carries the Gather turn's ``speech`` (a sensitive slot's value as
    #: its placeholder). On by default, as deployed.
    speech_log: bool = True
    #: That text also passes through the transcript redaction (a Comprehend call). On by
    #: default, as deployed.
    speech_log_redaction: bool = True
    #: The Memory trait groups the deployment lets reach the agent
    #: (``OWLBANK_MEMORY_TRAIT_GROUPS``, the bridge's setting). The welcome offer is prepared only
    #: when it lists ``WebActivity``. The code default is none (tests and a bare config).
    memory_trait_groups: tuple[str, ...] = ()
    #: AI keypad shortcuts: the AI session reports keys (``dtmfDetection``), a key
    #: stops the agent's speech and keys during speech are delivered. Off gives today's
    #: ConversationRelay TwiML exactly. The code default is off (tests and a bare config);
    #: ``OWLBANK_AI_KEYPAD`` defaults to on.
    ai_keypad: bool = False

    def __post_init__(self) -> None:
        if self.hosting not in HOSTINGS:
            raise ValueError(f"hosting must be one of {HOSTINGS}")
        if not is_mode(self.slot_mode_ceiling):
            raise ValueError(f"slot_mode_ceiling must be one of {list(MODES)}")  # INIT fails
        if (self.hosting == "fargate" or self.relay_ws_url is not None) and not (
                self.relay_ws_url or "").startswith("wss://"):
            raise ValueError("relay_ws_url must be a wss:// URL")
        bad = {k: v for k, v in self.fallbacks.items() if v not in FALLBACK_ACTIONS}
        if bad:
            raise ValueError(f"unknown fallback actions {sorted(bad)}")
        if self.scripted_barge_in not in BARGE_IN_MODES:
            raise ValueError(f"scripted_barge_in must be one of {BARGE_IN_MODES}")
        for name in ("speech_timeout_s", "confirm_speech_timeout_s"):
            value = getattr(self, name)
            if value != "auto" and not (
                    isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 10):
                raise ValueError(f"{name} must be 'auto' or an integer from 1 to 10")
        if isinstance(self.factor_timeout_s, bool) or not isinstance(self.factor_timeout_s, int) \
                or not 1 <= self.factor_timeout_s <= 10:
            raise ValueError(f"{KEYPAD_TIMEOUT_ENV} must be an integer from 1 to 10")
        if isinstance(self.phone_settle_ms, bool) or not isinstance(self.phone_settle_ms, int) \
                or not 0 <= self.phone_settle_ms <= MAX_PHONE_SETTLE_MS:
            raise ValueError(f"{PHONE_SETTLE_ENV} must be an integer from 0 to {MAX_PHONE_SETTLE_MS}")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> FrontDoorConfig:
        env = os.environ if env is None else env
        fallbacks = dict(DEFAULT_FALLBACKS)
        fallbacks.update(json.loads(env.get("OWLBANK_FALLBACKS", "{}") or "{}"))
        hosting = (env.get(HOSTING_ENV) or "").strip() or "fargate"
        if hosting not in HOSTINGS:
            raise ValueError(f"{HOSTING_ENV} must be one of {HOSTINGS}")  # never echo the value
        ceiling = (env.get(SLOT_MODE_CEILING_ENV) or "").strip() or "assisted"
        if not is_mode(ceiling):
            raise ValueError(f"{SLOT_MODE_CEILING_ENV} must be one of {list(MODES)}")  # never echo it
        return cls(
            hosting=hosting,
            slot_mode_ceiling=ceiling,
            briefing_budget_ms=_briefing_budget(env.get(BRIEFING_BUDGET_ENV)),
            speech_log=_on_off(env, SPEECH_LOG_ENV, default=True),
            speech_log_redaction=_on_off(env, SPEECH_LOG_REDACTION_ENV, default=True),
            # Fargate needs it (KeyError as before); agentcore has none (no relay domain).
            relay_ws_url=env["OWLBANK_RELAY_WS_URL"] if hosting == "fargate" else None,
            conversation_configuration=env.get("OWLBANK_CONVERSATION_CONFIGURATION_ID") or None,
            workflow_sid=env.get("OWLBANK_WORKFLOW_SID") or None,
            accept_spoken_factors=env.get("OWLBANK_ACCEPT_SPOKEN_FACTORS", "true").lower() == "true",
            fallbacks=fallbacks,
            record_ttl_s=int(env.get("OWLBANK_RECORD_TTL_S", "86400")),
            scripted_barge_in=(env.get("OWLBANK_SCRIPTED_BARGE_IN") or "any").lower(),
            reuse_conversation=env.get("OWLBANK_REUSE_CONVERSATION", "false").lower() == "true",
            early_digits=env.get("OWLBANK_EARLY_DIGITS", "true").strip().lower() != "false",
            phone_settle_ms=_settle_ms(env.get(PHONE_SETTLE_ENV)),
            factor_timeout_s=_keypad_timeout(env.get(KEYPAD_TIMEOUT_ENV)),
            speech_timeout_s=_speech_timeout(env.get("OWLBANK_SPEECH_TIMEOUT")),
            confirm_speech_timeout_s=_speech_timeout(env.get("OWLBANK_CONFIRM_SPEECH_TIMEOUT"),
                                                     name="OWLBANK_CONFIRM_SPEECH_TIMEOUT", default=2),
            memory_trait_groups=_trait_groups(env.get(MEMORY_TRAIT_GROUPS_ENV)),
            ai_keypad=_on_off(env, AI_KEYPAD_ENV, default=True),
        )

    @property
    def welcome_offers(self) -> bool:
        """Prepare the web-activity welcome offer (``WebActivity`` listed)."""
        return web_activity.GROUP in self.memory_trait_groups


def _briefing_budget(raw: str | None) -> int:
    """``OWLBANK_BRIEFING_BUDGET_MS``: absent or empty is the default; anything but an integer in
    range fails INIT, naming the variable and never echoing the value."""
    text = (raw or "").strip()
    if not text:
        return briefing.BRIEFING_BUDGET_MS
    message = (f"{BRIEFING_BUDGET_ENV} must be an integer from {MIN_BRIEFING_BUDGET_MS} "
               f"to {MAX_BRIEFING_BUDGET_MS}")
    try:
        value = int(text)
    except ValueError:
        raise ValueError(message) from None
    if not MIN_BRIEFING_BUDGET_MS <= value <= MAX_BRIEFING_BUDGET_MS:
        raise ValueError(message)
    return value


def _settle_ms(raw: str | None) -> int:
    """``OWLBANK_PHONE_SETTLE_MS``: absent or empty is the default; anything but an integer
    fails INIT, naming the variable and never echoing the value (the range is checked by the config)."""
    text = (raw or "").strip()
    if not text:
        return DEFAULT_PHONE_SETTLE_MS
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{PHONE_SETTLE_ENV} must be an integer from 0 to {MAX_PHONE_SETTLE_MS}") from None


def _trait_groups(raw: str | None) -> tuple[str, ...]:
    """``OWLBANK_MEMORY_TRAIT_GROUPS``: comma-separated group names (infra validates them). Only
    membership is used here, so anything else is read as given; unset or empty is none."""
    return tuple(dict.fromkeys(g.strip() for g in (raw or "").split(",") if g.strip()))


def _on_off(env: Mapping[str, str], name: str, *, default: bool) -> bool:
    """``on`` or ``off`` (any case; absent or empty is ``default``). Anything else fails INIT,
    naming the variable and never echoing the value."""
    raw = (env.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw not in ("on", "off"):
        raise ValueError(f"{name} must be 'on' or 'off'")
    return raw == "on"


def _keypad_timeout(raw: str | None) -> int:
    """``OWLBANK_KEYPAD_TIMEOUT``: whole seconds; unset or blank means
    :data:`DEFAULT_KEYPAD_TIMEOUT_S`. The range (1-10) is checked with the rest of the config.
    Never echoes the value."""
    if raw is None or raw.strip() == "":
        return DEFAULT_KEYPAD_TIMEOUT_S
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(f"{KEYPAD_TIMEOUT_ENV} must be an integer from 1 to 10") from None


def _speech_timeout(raw: str | None, *, name: str = "OWLBANK_SPEECH_TIMEOUT",
                    default: int = 3) -> int | str:
    """``OWLBANK_SPEECH_TIMEOUT`` or ``OWLBANK_CONFIRM_SPEECH_TIMEOUT``: "auto"
    or whole seconds; unset means ``default``."""
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip().lower()
    if raw == "auto":
        return "auto"
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be 'auto' or an integer") from None


@dataclass(slots=True)
class Deps:
    """Everything the front door talks to, injected."""

    store: Store
    catalog: PromptCatalog
    policy: Policy
    backend: BackendClient
    auth_token: Callable[[], str]
    basic_credentials: Callable[[], tuple[str, str]]
    #: The Basic secret with its previous password, for the request check (``force``: re-read).
    basic_secret: Callable[[bool], BasicSecret]
    sealer: Sealer
    pii: PiiGuard
    #: The slot resolver. ``None``: no model call (the deterministic
    #: ceiling builds none).
    resolver: Resolver | None = None
    #: The escalation briefing's model (without Flex). ``None``: no model call, the caller hears the
    #: fixed line (the ``deterministic`` ceiling builds none).
    briefing_model: briefing.BriefingModel | None = None
    #: The escalation briefing's conversation reader (the Orchestrator client; ``resolve_conversation``,
    #: ``communications`` and ``participants``). ``None``: the briefing has no transcript.
    transcripts: Any | None = None
    clock: Callable[[], float] = time.time
    closer: ConversationCloser = field(default_factory=lambda: NoopConversationCloser())
    #: How the ``otp_wait`` redirect sleeps between polls; tests pair it with ``clock``.
    sleep: Callable[[float], None] = time.sleep
    #: Twilio REST call updates for the early finish of spoken digits; ``None`` = off.
    calls: CallUpdater | None = None
    #: The ConversationRelay URL for each ``<Connect>``. ``None``: ``fargate``'s
    #: :class:`StaticRelayUrl` over ``FrontDoorConfig.relay_ws_url`` (today's URL byte for byte);
    #: ``agentcore`` requires a signer (:class:`~lambdas.frontdoor.relay.RuntimeUrlSigner`).
    relay: RelayUrls | None = None
    #: ``agentcore``: starts the call's Runtime session early, given the CallSid; an
    #: asynchronous (Event) invoke of ``runtime_warmer``. ``None``: no pre-warm.
    warmer: Callable[[str], None] | None = None
    #: ``agentcore``: stops the call's Runtime session at call end
    #: (:class:`~lambdas.frontdoor.runtime_sessions.RuntimeSessions`). ``None``: no stop.
    runtime_sessions: SessionStopper | None = None
    #: Human-leg (``humanLegConversation=on``): the Orchestrator client the escalation uses to find
    #: the call's conversation when it doesn't know it. ``None``: no lookup.
    human_leg: HumanLegClient | None = None
    #: Human-leg: starts the customer leg's transcription on the first accept (the Calls API,
    #: :class:`owlbank.humanleg_client.TwilioCalls`). ``None``: no start, whatever the policy switch.
    transcriptions: TranscriptionStarter | None = None
    #: Memory writes: starts the call recap writer at call end when no human-leg close
    #: will (:func:`owlbank.recap_trigger.at_status`); the closer holds its own for the close.
    #: ``None``: no recap is started here.
    recap: RecapTrigger | None = None


class TranscriptionStarter(Protocol):
    def start_transcription(self, call_sid: str, params: Mapping[str, str]) -> bool:
        """``True`` if Twilio started it; ``False`` on any failure (never raises)."""
        ...


#: The ``reservation.accepted`` handler's total time for its Twilio requests: the hand-off
#: metadata, the participant add and the transcription start share it, then the event gets its 204.
HUMAN_LEG_ACCEPT_BUDGET_S = 2.0


class ConversationCloser(Protocol):
    """Closes the call's Orchestrator Conversation at call end (the Conversation
    Configuration's timeout is in minutes, so call end must close it explicitly)."""

    def close(self, *, call_sid: str, conversation_id: str | None) -> None:
        """``/voice/status`` ``completed`` or ``failed``. ``conversation_id`` is ``None`` when the
        front door never learned it; an implementation must then find the conversation by
        ``call_sid`` or do nothing."""
        ...

    def transcription_ended(self, *, call_sid: str, conversation_id: str | None, event: str,
                            leg: str | None = None) -> None:
        """``/voice/transcription``: ``event`` is ``transcription_stopped`` or
        ``transcription_error`` (:mod:`owlbank.humanleg`) for the transcription on ``leg`` (the leg's
        call SID; ``None`` means the customer leg) of the customer call ``call_sid``."""
        ...


class NoopConversationCloser:
    """Default with ``humanLegConversation=off``: nothing closes the conversation here (with
    ``on``, the handler injects :class:`owlbank.humanleg_close.HumanLegCloser`)."""

    def close(self, *, call_sid: str, conversation_id: str | None) -> None:
        log.info("conversation close skipped (no closer configured)",
                 extra={"call_sid": call_sid, "conversation_known": conversation_id is not None})

    def transcription_ended(self, *, call_sid: str, conversation_id: str | None, event: str,
                            leg: str | None = None) -> None:
        log.info("conversation close skipped (no closer configured)",
                 extra={"call_sid": call_sid, "conversation_known": conversation_id is not None})


class _Abort(Exception):
    """Internal: the request can't be served normally (e.g. no call state)."""


class _AgentUnavailable(Exception):
    """Internal: the AI stage can't be reached (a relay URL couldn't be signed). Takes the
    apology-then-human path: the catalog apology, then a human."""


# --------------------------------------------------------------------------- the app


class FrontDoor:
    def __init__(self, deps: Deps, config: FrontDoorConfig) -> None:
        self.d = deps
        self.cfg = config
        self.journal = SegmentLog(deps.store, now=deps.clock, ttl_s=config.record_ttl_s)
        self.slots = slotflow.SlotFlow(self)
        #: The answer request's timings (:meth:`factor`); ``None`` outside one.
        self._timings: obs.Timings | None = None
        #: ``keyed_phone_unavailable`` was logged on this instance (once).
        self._phone_unready_logged = False
        #: The request in progress: its start (monotonic) and, from the Lambda context, its remaining ms.
        self._request_started = time.monotonic()
        self._remaining_ms: Callable[[], int | float] | None = None
        if deps.relay is not None:
            self.relay_urls: RelayUrls = deps.relay
        elif config.relay_ws_url is not None:
            self.relay_urls = StaticRelayUrl(config.relay_ws_url)
        else:
            raise ValueError("agentcore hosting needs Deps.relay (a RuntimeUrlSigner)")

    # ------------------------------------------------------------------ entry point

    def handle(self, event: Mapping[str, Any], *,
               remaining_ms: Callable[[], int | float] | None = None) -> dict[str, Any]:
        """API Gateway proxy event -> proxy response. Never raises. ``remaining_ms`` is the Lambda
        context's ``get_remaining_time_in_millis`` (the briefing's budget is capped by it). With call observability on, an
        authenticated webhook is one span (3.1; a no-op when off). Twilio's 401 challenge and
        rejected requests get none: they would double the span count and trust an
        unauthenticated CallSid (their logs and metric filters stay the record)."""
        self._request_started = time.monotonic()
        self._remaining_ms = remaining_ms
        return self._handle(event, http.route_name(event))

    def _handle(self, event: Mapping[str, Any], route: str | None) -> dict[str, Any]:
        if route == http.ORCHESTRATOR_ROUTE:
            return self.orchestrator_event(event)
        try:
            req = http.authenticate(event, auth_token=self.d.auth_token,
                                    basic_secret=self.d.basic_secret, clock=self.d.clock)
        except (http.Challenge, http.Malformed) as exc:  # 401 challenge, or the generic 400
            return http.challenge() if isinstance(exc, http.Challenge) else http.bad_request()
        except http.Rejected:
            return http.forbidden()
        except Exception as exc:  # pragma: no cover - authenticate already fails closed
            obs.log_caught(log, "security: rejected request", exc, decision="reject_403")
            return http.forbidden()
        call_sid = None
        if telemetry.installed():  # a TaskRouter event names its call in the task's attributes
            if req.route == http.TASKROUTER_ROUTE:
                call_sid = taskrouter_call_sid(req.params)
            elif req.route == http.TRANSCRIPTION_ROUTE:  # an agent leg's span is its customer call's
                call_sid = transcription_call_sid(req) or req.call_sid
            else:
                call_sid = req.call_sid
        if req.route == http.TRANSCRIPTION_ROUTE:
            # Human leg: only the end of the transcript matters (and gets a span); every other
            # event (``transcription-content`` carries the transcript) is acknowledged untouched.
            end = humanleg.transcription_end_event(req.params.get("TranscriptionEvent"))
            if end is not None:
                with telemetry.webhook_span(route, call_sid):
                    self.transcription(req, end)
            return http.no_content()
        with telemetry.webhook_span(route, call_sid):
            if req.route == "partial":  # Twilio ignores any answer to a partial result
                self.partial(req)
                return http.no_content()
            if req.route == http.TASKROUTER_ROUTE:
                self.taskrouter(req)
                return http.event_ack()
            return http.twiml_response(self.dispatch(req))

    def orchestrator_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """The Orchestrator status-callback sink: check, log the event type and timing, answer 200.
        Never relays, never touches the store. Never raises."""
        try:
            if self.cfg.hosting != "agentcore":
                auth_failed("frontdoor", "route")
                return http.forbidden()
            req = http.authenticate_callback(event, auth_token=self.d.auth_token,
                                             basic_secret=self.d.basic_secret, clock=self.d.clock)
        except http.Unauthorized:
            return http.unauthorized()
        except http.Rejected:
            return http.forbidden()
        except Exception as exc:  # pragma: no cover - authenticate_callback already fails closed
            obs.log_caught(log, "security: rejected request", exc, decision="reject_403")
            return http.forbidden()
        try:
            described = describe_orchestrator_event(req, received_ms=int(self.d.clock() * 1000))
        except Exception as exc:  # the acknowledgement never depends on reading the event
            obs.log_caught(log, "orchestrator_event_unreadable", exc, decision="acknowledged")
            described = {}
        obs.emit(log, logging.INFO, "orchestrator_event_dropped", reason="no_relay", **described)
        return http.ok_empty()

    def dispatch(self, req: http.TwilioRequest) -> str:
        routes: dict[str, Callable[[http.TwilioRequest], str]] = {
            "incoming": self.incoming, "language": self.language, "factor": self.factor,
            "connect-action": self.connect_action, "tool-flow": self.factor,
            "status": self.status,
            http.ENQUEUE_ACTION_ROUTE: self.enqueue_action,
        }
        if req.route == http.ENQUEUE_ACTION_ROUTE:
            # Never the apology path: it escalates, and the caller is already queued (or leaving).
            try:
                return routes[req.route](req)
            except Exception as exc:
                obs.log_caught(log, "handler error", exc, level=logging.ERROR, route=req.route,
                               call_sid=req.call_sid, decision="empty")
                return twiml.empty()
        try:
            return routes[req.route](req)
        except _AgentUnavailable:  # already logged (relay_presign_failed); apology-then-human path
            obs.degraded("relay_presign_failed")
            return self._apology(req)
        except Exception as exc:
            # Class and frames only: the message could carry caller input.
            obs.log_caught(log, "handler error", exc, level=logging.ERROR, route=req.route,
                           call_sid=req.call_sid,
                           decision="status_empty" if req.route == "status" else "apology_and_escalate")
            obs.degraded("handler_error")
            if req.route == "status":
                return twiml.empty()
            return self._apology(req)

    # ------------------------------------------------------------------ helpers

    def _now(self) -> int:
        return int(self.d.clock())

    def _span(self, name: str) -> contextlib.AbstractContextManager[None]:
        """A ``timings_ms`` span of the answer request in progress; nothing outside one."""
        t = self._timings
        return t.span(name) if t is not None else contextlib.nullcontext()

    def _render(self, *verbs: Any) -> str:
        with self._span("render"):
            return twiml.render(*verbs)

    def _save(self, sess: Session) -> None:
        with self._span("state_save"):
            sessions.save(self.d.store, sess)

    def _say_voice(self, key: str, lang: str, *, voice_lang: str, voice: str | None = None,
                   **values: Any) -> tuple[twiml.Say | twiml.Play, Spoken]:
        """``key`` in ``lang`` (text and pronunciation), spoken with ``voice`` if given, else with
        ``voice_lang``'s voice."""
        text = self.d.catalog.text(key, lang, **values)
        audio = self.d.catalog.audio_url(key, lang)
        if audio:
            return twiml.Play(audio), Spoken(key, text)
        voice = voice or self.d.catalog.settings(voice_lang).get("say", {}).get("voice")
        language = self.d.catalog.settings(lang).get("say", {}).get("language")
        return twiml.Say(text, voice=voice, language=language), Spoken(key, text)

    def _say(self, key: str, lang: str, **values: Any) -> tuple[twiml.Say | twiml.Play, Spoken]:
        text = self.d.catalog.text(key, lang, **values)
        audio = self.d.catalog.audio_url(key, lang)
        if audio:
            return twiml.Play(audio), Spoken(key, text)
        say = self.d.catalog.settings(lang).get("say", {})
        return twiml.Say(text, voice=say.get("voice"), language=say.get("language")), Spoken(key, text)

    def _say_generated(self, key: str, lang: str, text: str, *,
                       template: tuple[Any, Spoken]) -> tuple[twiml.Say, Spoken]:
        """Model wording that :func:`owlbank.slots.generative.check` accepted, already rendered
        in ``lang``'s voice. ``key`` is the catalog template it stands in for; the journal
        marks it ``generated``. Never a recording: the text is this call's. ``template`` is
        that template, spoken instead if the response holding the text doesn't parse
        (:meth:`_reprompt`)."""
        say = self.d.catalog.settings(lang).get("say", {})
        return (twiml.Say(text, voice=say.get("voice"), language=say.get("language")),
                GeneratedSpoken(key, text, template))

    def _log_event(self, call_sid: str, event: dict[str, Any]) -> None:
        self.journal.append(call_sid, event)

    def _prompts(self, call_sid: str, items: list[tuple[Any, Spoken]]) -> None:
        for _, sp in items:
            extra = {"generated": True} if isinstance(sp, GeneratedSpoken) else {}
            self._log_event(call_sid, {"type": "prompt", "prompt_key": sp.template,
                                       "rendered_text": sp.text, **extra})

    def _state(self, call_sid: str) -> CallState | None:
        """The call state, with the call's sealed sensitive values registered in this process
        (``sensitive.restore``): another front-door instance or the bridge may have sealed
        them. ``restore`` never raises; an entry that doesn't unseal marks the call unverifiable,
        and its journal surfaces then withhold."""
        with self._span("state_load"):
            st = callstate.load(self.d.store, call_sid)
        if st is not None and st.sealed_sensitive:
            sensitive.restore(call_sid, st.sealed_sensitive, self.d.sealer)
        return st

    def _remember(self, call_sid: str, slot: SlotSettings, value: str) -> None:
        """A sensitive slot's value: sealed into call state and registered here. A
        ``yes_no`` or ``choice`` value is sealed but never registered; one under
        ``MIN_SENSITIVE_LEN`` is sealed and logged ``sensitive_scrub_skipped``."""
        sensitive.remember(self.d.store, self.d.sealer, call_sid, value, slot=slot.name,
                           slot_type=slot.decl.slot_type, now=self.d.clock)

    def _load(self, call_sid: str) -> tuple[CallState, Session]:
        if not call_sid:
            raise _Abort("no CallSid")
        st = self._state(call_sid)
        if st is None:
            raise _Abort("no call state")
        with self._span("state_load"):
            sess = sessions.load(self.d.store, call_sid) or Session(call_sid, st.expires_at)
        return st, sess

    def _update(self, st: CallState, **changes: Any) -> CallState:
        def fn(_: CallState) -> Mapping[str, Any]:
            return changes
        with self._span("state_save"):
            return callstate.mutate(self.d.store, st.call_sid, fn, now=self.d.clock)

    def _slot(self, name: str) -> SlotSettings:
        """A slot's settings (``slot#<name>``): its shape (``decl``), input and whether it
        is sensitive, which alone decides what the AI sees and what is registered."""
        try:
            return self.d.policy.slot(name)
        except KeyError:
            raise _Abort("unknown slot") from None

    def _ani_counts(self, st: CallState) -> bool:
        """Whether the ANI match counts as a factor on this call (the auth service
        applies the same policy rule and is the one that enforces it). Never on a call identified
        by a keyed-in number, also with ``on_untrusted: allow``."""
        if st.identity_source == KEYED_PHONE:
            return False
        return self.d.policy.ani_match_counts(st.stir_verstat)

    def _have(self, st: CallState) -> set[str]:
        have = set(st.factors_verified) - set(ESTABLISHED_FACTORS)
        if st.candidate_ids and self._ani_counts(st):
            have.add("ani_match")
        return have

    def _eff(self, st: CallState, level: int) -> list[str]:
        return self.d.policy.effective_factors(level, ani_counts=self._ani_counts(st))

    def _missing(self, st: CallState, target: int, extra: set[str] | None = None) -> list[str]:
        """Collected factors still needed for ``target`` on this call, in policy order."""
        have = self._have(st) | (extra or set())
        return self._collectable([f for f in self._eff(st, target) if f not in have])

    def _level_for(self, have: set[str], st: CallState) -> int:
        level = 0
        for n in range(1, self.d.policy.max_level + 1):
            if set(self._eff(st, n)) <= have:
                level = n
        return level

    def _collectable(self, names: list[str]) -> list[str]:
        return [f for f in names if f not in ESTABLISHED_FACTORS]

    # ------------------------------------------------------------------ ConversationRelay

    def _relay(self, req: http.TwilioRequest, st: CallState, lang: str,
               welcome: str, *, scripted: bool = False,
               conversation_id: str | None = None,
               welcome_offer: str | None = None) -> twiml.ConversationRelay:
        """The ``<Connect action><ConversationRelay>``, with TAC ``generate_twiml``'s attribute set
        (integration notes 1e). ``welcomeGreeting`` is always set, even empty.

        Every connect gets a fresh one-time nonce and a fresh URL from ``relay_urls``:
        today's bridge URL with the nonce (``fargate``) or a pre-signed Runtime URL
        (``agentcore``). The nonce's hash goes to call state (replacing any unused one), and the
        same write stores the greeting sealed as ``last_spoken``, so the bridge can repeat it
        after caller silence; an empty greeting clears it. When this greeting is the
        web-activity offer, the same write sets ``welcome_offer`` to ``made``;
        any other connect leaves it as it is (once spoken, the offer is in the
        agent's history for the rest of the call; only call end and the silence hangup clear it).
        In ``agentcore`` that write
        also stores ``relay_minted_at`` (the warm/cold metric). If the URL can't be signed
         nothing is written: ``relay_presign_failed`` (error class only), then
        :class:`_AgentUnavailable`. The text, the URL and the nonce are never logged."""
        nonce, digest = ws_nonce.new_nonce()
        try:
            url = self.relay_urls.mint(call_sid=st.call_sid, nonce=nonce)
        except PresignError as exc:
            obs.emit(log, logging.ERROR, "relay_presign_failed", call_sid=st.call_sid,
                     error_class=exc.error_class, hosting=self.cfg.hosting,
                     decision="apology_and_escalate")
            raise _AgentUnavailable from None
        now = self.d.clock()
        spoken = last_spoken.seal(self.d.sealer, welcome, call_sid=st.call_sid, now=now)
        minted: dict[str, Any] = {"relay_minted_at": int(now)} if self.cfg.hosting == "agentcore" else {}
        offer = ({callstate.F_WELCOME_OFFER: callstate.WELCOME_OFFER_MADE}
                 if welcome_offer == callstate.WELCOME_OFFER_MADE else {})
        self._update(st, ws_nonce_hash=digest, last_spoken=spoken, **offer, **minted)
        obs.emit(log, logging.INFO, "relay_minted", call_sid=st.call_sid, hosting=self.cfg.hosting)
        parameters = {"callSid": req.call_sid}
        if self.cfg.hosting == "agentcore":
            # AgentCore forwards no query parameter, so the nonce's one carrier is
            # <Parameter name="n"> (the bridge reads customParameters). Fargate keeps ``?n=``.
            parameters[ws_nonce.QUERY_PARAM] = nonce
        cr = self.d.catalog.settings(lang).get("conversationRelay", {})
        keypad = self.cfg.ai_keypad
        extra: dict[str, str] = {}
        if self.cfg.reuse_conversation and conversation_id:
            # Route this segment straight to the call's conversation (ConversationRelay
            # ``conversationId`` bypasses the configuration's grouping rules).
            extra["conversationId"] = conversation_id
        elif self.cfg.conversation_configuration:
            extra["conversationConfiguration"] = self.cfg.conversation_configuration
        return twiml.ConversationRelay(
            url=url,
            action=req.url("connect-action"),
            welcome_greeting=welcome,
            language=lang,
            tts_provider=cr.get("ttsProvider"),
            voice=cr.get("voice"),
            transcription_provider=cr.get("transcriptionProvider"),
            transcription_language=cr.get("transcriptionLanguage"),
            speech_model=cr.get("speechModel"),
            # With keypad shortcuts on, keys reach the bridge, also during the agent's speech.
            interruptible=AI_KEYPAD_INTERRUPTIBLE if keypad else self.cfg.interruptible,
            welcome_greeting_interruptible=self.cfg.scripted_barge_in if scripted else None,
            dtmf_detection=keypad,
            report_input_during_agent_speech=AI_KEYPAD_REPORT_INPUT if keypad else None,
            events=RELAY_EVENTS,
            parameters=parameters,
            extra_attrs=extra,
        )

    def _connect(self, req: http.TwilioRequest, st: CallState, sess: Session,
                 say_now: list[tuple[Any, Spoken]], welcome: Spoken | None,
                 envelope: ResultEnvelope | None, *, protect_outcome: bool = False,
                 welcome_offer: str | None = None) -> str:
        """Return to (or start) the AI stage. The envelope becomes ``pending_result``.

        Any prepared welcome (``welcome_prepared``) is discarded here: only
        :meth:`_login_complete` may use it, and it does so before calling this. ``welcome_offer``
         goes to :meth:`_relay`, into the write of ``last_spoken``.

        ``protect_outcome``: ``say_now`` is an outcome read-out (a payment or
        transfer done, cancelled, failed) the caller must hear whole, so it is spoken as plain
        ``<Say>`` before ``<Connect>`` (not interruptible) and only ``welcome`` (the follow-up,
        "Anything else?") is the interruptible welcome greeting. ConversationRelay's
        interruptibility is per greeting, not per part, so this is the requirement's fallback."""
        sess.flow = None
        sess.relay_failures = 0
        self._save(sess)
        utterance = envelope.callerUtterance if envelope and envelope.next == "handle_utterance" \
            else None
        discard = {callstate.F_WELCOME_PREPARED: None} if st.welcome_prepared is not None else {}
        st = self._update(st, pending_result=envelope.to_dict() if envelope else None,
                          interrupted_utterance=utterance, **discard)
        self._prompts(st.call_sid, say_now + ([(None, welcome)] if welcome else []))
        self._log_event(st.call_sid, {"type": "handoff", "to": "ai",
                                      "outcome": envelope.status if envelope else "connect",
                                      "level": st.level})
        # A scripted read-out (balance, a result) is spoken by ConversationRelay as its
        # welcome greeting, so the caller can interrupt it and what they say reaches the AI as the
        # next turn (the envelope above carries the full text). The words stay the catalog's.
        # Pre-recorded audio can't be a greeting, so any <Play> item keeps the old <Say>/<Play> path.
        if say_now and not protect_outcome and all(isinstance(v, twiml.Say) for v, _ in say_now):
            greeting = " ".join([s.text for _, s in say_now] + ([welcome.text] if welcome else []))
            return self._render(self._relay(req, st, st.language, greeting, scripted=True,
                                            conversation_id=sess.conversation_id,
                                            welcome_offer=welcome_offer))
        verbs: list[Any] = [v for v, _ in say_now]
        verbs.append(self._relay(req, st, st.language, welcome.text if welcome else "",
                                 scripted=protect_outcome and bool(say_now),
                                 conversation_id=sess.conversation_id, welcome_offer=welcome_offer))
        return self._render(*verbs)

    # ------------------------------------------------------------------ escalation

    def _escalate(self, req: http.TwilioRequest, st: CallState, sess: Session, reason: Any,
                  summary: Any, *, pre: list[tuple[Any, Spoken]] | None = None,
                  journal_ref: str | None = None, conversation_id: str | None = None,
                  profile_id: str | None = None) -> str:
        if not self.cfg.workflow_sid:
            return self._escalate_default(req, st, sess, reason, pre=pre, journal_ref=journal_ref,
                                          summary=summary,
                                          conversation_id=conversation_id or sess.conversation_id)
        policy = self.d.policy
        route = policy.escalation_route(reason if isinstance(reason, str) else "")
        if route.reason == VERIFICATION_RISK:
            # Nothing this call verified is vouched for; the agent starts at 0.
            st = self._update(st, level=0, verified_customer_id=None, factors_verified=())
        lang = st.language
        ref = journal_ref if journal_ref and _SAFE_REF.match(journal_ref) else (
            _stdlib_secrets.token_urlsafe(16))  # opaque, one per escalation
        now = self._now()
        queue = policy.queue_name(route.queue_key, lang, default_lang=self.d.catalog.default_language)
        withheld = self._withheld_summary(lang)
        if sensitive.is_unverifiable(st.call_sid):
            summary = withheld  # a sealed value couldn't be unsealed: fail closed
        conversation_id = self._handoff_conversation(st, conversation_id or sess.conversation_id)
        attrs = build_task_attributes(
            st, self.journal.summary_for_resume(st.call_sid), route.reason, summary,
            now=now, reasons=policy.reasons, conversation_id=conversation_id,
            profile_id=profile_id, journal_ref=ref, test=self.cfg.test_traffic,
            store=self.d.store, pii=self.d.pii, withheld_summary=withheld,
        )
        at = datetime.fromtimestamp(now, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        # The AI session is over: customer-data tools stop accepting the call's token.
        discard = {callstate.F_WELCOME_PREPARED: None} if st.welcome_prepared is not None else {}
        st = self._update(st, escalation={"reason": route.reason, "queue": queue, "at": at,
                                          "journal_ref": ref}, sealed_token=None, **discard)
        tokens.revoke_call(self.d.store, st.call_sid)
        sess.flow = None
        self._save(sess)
        phrase = self._say(route.phrase_key, lang)
        items = (pre or []) + [phrase]
        self._prompts(st.call_sid, items)
        self._log_event(st.call_sid, {"type": "escalation", "reason": route.reason,
                                      "queue": queue, "journal_ref": ref})
        self._journal_handoff(st, conversation_id)
        obs.emit(log, logging.INFO, "escalation", reason=route.reason, source="frontdoor", queue=queue,
                 level_reached=st.level)
        # No <Task> timeout, so the task lives for TaskRouter's default lifetime;
        # Twilio's default hold music plays (no waitUrl) and the action URL sees them leave.
        enqueue = twiml.Enqueue(workflow_sid=self.cfg.workflow_sid, task_attributes=attrs,
                                action=req.url(http.ENQUEUE_ACTION_ROUTE),
                                task_priority=route.priority)
        return self._render(*[v for v, _ in items], enqueue)

    def _escalate_default(self, req: http.TwilioRequest, st: CallState, sess: Session, reason: Any,
                          *, pre: list[tuple[Any, Spoken]] | None, journal_ref: str | None,
                          summary: Any = None, conversation_id: str | None = None) -> str:
        """Escalation without Flex: :meth:`_escalate_briefing`, and if that raises anything, the
        apology, the goodbye and a hang-up (never an empty or half-built response)."""
        try:
            return self._escalate_briefing(req, st, sess, reason, pre=pre, journal_ref=journal_ref,
                                           summary=summary, conversation_id=conversation_id)
        except Exception as exc:
            obs.log_caught(log, "briefing_escalation_failed", exc, level=logging.ERROR,
                           call_sid=st.call_sid, decision="apology_and_hangup")
            with contextlib.suppress(Exception):
                tokens.revoke_call(self.d.store, st.call_sid)
            verbs: list[Any] = []
            for key in ("error.generic", "common.goodbye"):
                try:
                    verbs.append(self._say(key, st.language)[0])
                except Exception as text_exc:
                    obs.log_caught(log, "briefing_fail_safe_text_unavailable", text_exc,
                                   operation="catalog_text", decision="skipped")
            return self._render(*verbs, twiml.Hangup())

    def _escalate_briefing(self, req: http.TwilioRequest, st: CallState, sess: Session, reason: Any,
                           *, pre: list[tuple[Any, Spoken]] | None = None,
                           journal_ref: str | None = None, summary: Any = None,
                           conversation_id: str | None = None) -> str:
        """Escalation without a Flex workflow: nothing is queued, dialled or transferred. The
        caller hears ``pre``, then what an agent would have received (the briefing), then the
        goodbye, and the call ends. The route phrase is neither spoken nor journaled.

        The first entry (no ``destination: briefing`` in ``st.escalation``) works out the route,
        resets the level on a verification risk, and records ``escalation``. A repeat keeps that
        record's ``ref`` and ``at``. Every entry revokes the call's token and journals the
        ``escalation`` event only if none holds this ``ref``. The read-out is built afresh each time,
        from the agent's summary (kept in the escalation record, so a repeat has it) and the
        conversation's scrubbed transcript tail. Nothing here logs the briefing text."""
        policy = self.d.policy
        route = policy.escalation_route(reason if isinstance(reason, str) else "")
        prior = st.escalation if isinstance(st.escalation, Mapping) \
            and st.escalation.get("destination") == BRIEFING_DESTINATION else None
        prior_ref = prior.get("journal_ref") if prior is not None else None
        if prior is not None and isinstance(prior_ref, str) and _SAFE_REF.match(prior_ref):
            ref = prior_ref
            route_reason = str(prior.get("reason") or route.reason)
            summary = summary or prior.get("summary")
        else:
            prior = None
            route_reason = route.reason
            if route_reason == VERIFICATION_RISK:
                # Nothing this call verified is vouched for; the level starts again at 0.
                st = self._update(st, level=0, verified_customer_id=None, factors_verified=())
            ref = journal_ref if journal_ref and _SAFE_REF.match(journal_ref) else (
                _stdlib_secrets.token_urlsafe(16))  # opaque, one per escalation
            at = datetime.fromtimestamp(self._now(), tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            discard = {callstate.F_WELCOME_PREPARED: None} if st.welcome_prepared is not None else {}
            if not summary and isinstance(st.escalation, Mapping):
                summary = st.escalation.get("summary")  # the bridge's, when the handoff has none
            record: dict[str, Any] = {"reason": route_reason, "destination": BRIEFING_DESTINATION,
                                      "at": at, "journal_ref": ref}
            if isinstance(summary, str) and summary:
                record["summary"] = summary  # the bridge's (cleaned) text, kept for a repeat
            st = self._update(st, escalation=record, sealed_token=None, **discard)
        lang = st.language
        tokens.revoke_call(self.d.store, st.call_sid)  # idempotent: a retry still revokes
        sess.flow = None
        self._save(sess)
        if not self._escalation_journaled(st.call_sid, ref):
            self._log_event(st.call_sid, {"type": "escalation", "reason": route_reason,
                                          "destination": BRIEFING_DESTINATION, "journal_ref": ref})
        items = list(pre or [])
        self._prompts(st.call_sid, items)  # the apology and the like, before the wait
        read_out = self._briefing_read_out(st, lang, route_reason, summary=summary,
                                           conversation_id=conversation_id or sess.conversation_id)
        self._prompts(st.call_sid, read_out)
        goodbye = self._say("common.goodbye", lang)
        self._prompts(st.call_sid, [goodbye])
        obs.emit(log, logging.INFO, "escalation", reason=route_reason, source="frontdoor",
                 destination=BRIEFING_DESTINATION, level_reached=st.level)
        return self._render(*[v for v, _ in (*items, *read_out, goodbye)], twiml.Hangup())

    def _escalation_journaled(self, call_sid: str, ref: str) -> bool:
        """Whether the journal already holds an ``escalation`` event with this ``ref``. A journal
        that can't be read counts as not journaled (a second event beats none)."""
        try:
            events = self.journal.read(call_sid)
        except Exception as exc:
            obs.log_caught(log, "escalation_journal_unreadable", exc, call_sid=call_sid,
                           operation="journal_read", decision="journal_again")
            return False
        return any(e.get("type") == "escalation" and e.get("journal_ref") == ref for e in events)

    def _briefing_summary(self, st: CallState, summary: Any) -> str | None:
        """The agent's escalation summary as the Flex task's ``ai_summary`` is prepared (cleaned,
        factor values redacted, PII-masked), or ``None`` for none, a withheld one, or a call whose
        sealed values could not be unsealed."""
        if not isinstance(summary, str) or not summary.strip() or sensitive.is_unverifiable(st.call_sid):
            return None
        try:
            # The model's input is withheld on an outage whatever the policy says (the Flex task's
            # own summary follows the policy).
            return task_attributes.prepare_summary(st.call_sid, summary, self.d.pii, st.language,
                                                   self._withheld_summary(st.language),
                                                   on_failure=ON_FAILURE_WITHHOLD)
        except Exception as exc:
            obs.log_caught(log, "briefing_summary_failed", exc, call_sid=st.call_sid,
                           operation="prepare_summary", decision="without_summary")
            return None

    def _briefing_transcript(self, st: CallState, conversation_id: str | None,
                             seconds: float) -> str | None:
        """The scrubbed tail of the call's conversation, read and scrubbed within ``seconds``, else
        ``None``. Needs a reader, an AI session and a conversation (the given one, else one lookup by
        CallSid)."""
        reader = self.d.transcripts
        if reader is None or st.ai_sessions == 0 or sensitive.is_unverifiable(st.call_sid):
            return None
        if seconds <= 0:
            return None
        deadline = time.monotonic() + seconds
        try:
            with human_leg_budget(seconds):
                conv = conversation_id or reader.resolve_conversation(st.call_sid)
                return briefing_transcript.fetch(
                    reader, conv, call_sid=st.call_sid,
                    factor_values=segments.sensitive_values(st.call_sid), pii=self.d.pii,
                    language=st.language, deadline=deadline)
        except Exception as exc:
            obs.log_caught(log, "briefing_transcript_failed", exc, call_sid=st.call_sid,
                           operation="transcript", decision="without_transcript")
            return None

    def _briefing_facts(self, st: CallState, reason: str, *, summary: str | None = None,
                        transcript: str | None = None) -> briefing.BriefingFacts:
        policy = self.d.policy
        try:
            journal_summary: Mapping[str, Any] = self.journal.summary_for_resume(st.call_sid)
        except Exception as exc:
            obs.log_caught(log, "briefing_summary_unavailable", exc, call_sid=st.call_sid,
                           operation="journal_summary", decision="facts_without_tools")
            journal_summary = {}
        factors = {f for n in range(1, policy.max_level + 1) for f in policy.factors(n)}
        return briefing.build_facts(st, journal_summary, reason=reason, known_factors=factors,
                                    known_reasons=policy.reasons, summary=summary,
                                    transcript=transcript)

    def _briefing_label(self, key: str, lang: str, default_key: str | None = None) -> str:
        """The catalog label ``key`` in ``lang`` (the catalog falls back to its default language
        for a missing language). A key the catalog lacks entirely reads ``default_key`` and logs a
        warning; with no default the lookup raises."""
        if key not in self.d.catalog.keys and default_key is not None:
            log.warning("briefing label missing; using the default label (key=%s)", key)
            key = default_key
        return self.d.catalog.text(key, lang)

    def _briefing_budget_ms(self) -> float:
        """The configured briefing budget, capped by what is left of this request (the Lambda
        context's remaining time, else the timeout less the time since the request began) less
        :data:`BRIEFING_MARGIN_MS`."""
        left: float | None = None
        if self._remaining_ms is not None:
            try:
                left = float(self._remaining_ms())
            except Exception:  # noqa: BLE001 - fall back to the request's own clock
                left = None
        if left is None:
            left = REQUEST_TIMEOUT_MS - (time.monotonic() - self._request_started) * 1000
        return min(float(self.cfg.briefing_budget_ms), left - BRIEFING_MARGIN_MS)

    def _briefing_read_out(self, st: CallState, lang: str, reason: str, *, summary: Any = None,
                           conversation_id: str | None = None) -> list[tuple[Any, Spoken]]:
        """The lead-in and then the model's clause (else the fixed line): one generation under the
        one deadline, never raising for a model failure. Logs ``briefing_generated`` with the source,
        the word count and the latency, never the text."""
        started = time.monotonic()
        budget_ms = self._briefing_budget_ms()
        model = self.d.briefing_model
        if budget_ms < BRIEFING_FLOOR_MS:
            obs.emit(log, logging.WARNING, "briefing_skipped", call_sid=st.call_sid,
                     budget_ms=max(0, int(budget_ms)), decision="fixed_line")
            model = None  # not enough of the request left: the fixed line, no model call
        prepared: str | None = None
        transcript: str | None = None
        inputs_ms = 0
        if model is not None:
            # The summary check and the transcript read each make a PII call; they run together under
            # one hard deadline (a late one is dropped), and the model gets what is left of the budget.
            seconds = min(BRIEFING_TRANSCRIPT_MS, BRIEFING_TRANSCRIPT_SHARE * budget_ms) / 1000
            inputs_started = time.monotonic()
            got = briefing_transcript.gather({
                "summary": lambda: self._briefing_summary(st, summary),
                "transcript": lambda: self._briefing_transcript(st, conversation_id, seconds),
            }, seconds)
            prepared, transcript = got["summary"], got["transcript"]
            inputs_ms = round((time.monotonic() - inputs_started) * 1000)
        transcript_ms = inputs_ms
        facts = self._briefing_facts(st, reason, summary=prepared, transcript=transcript)
        left_ms = budget_ms - inputs_ms
        if model is not None and left_ms < BRIEFING_FLOOR_MS:
            model = None  # the transcript used the budget: the fixed line
        result = briefing.generate(model, facts, budget_ms=max(1, round(left_ms)))
        obs.emit(log, logging.INFO, "briefing_generated", call_sid=st.call_sid, source=result.source,
                 words=result.words, latency_ms=int((time.monotonic() - started) * 1000),
                 transcript_ms=transcript_ms, has_summary=prepared is not None,
                 has_transcript=transcript is not None)
        level = self._briefing_label(f"{BRIEFING_LEVEL_PREFIX}{facts.level}", lang)
        lead = self._say(BRIEFING_LEAD_KEY, lang, level=level)
        reason_label = self._briefing_label(f"{BRIEFING_REASON_PREFIX}{facts.reason}", lang,
                                            f"{BRIEFING_REASON_PREFIX}{BRIEFING_REASON_DEFAULT}")
        fixed = self._say(BRIEFING_FALLBACK_KEY, lang, level=level, reason=reason_label)
        if result.source == "model" and result.summary:
            clause = self._say_generated(BRIEFING_CLAUSE_KEY, lang, result.summary, template=fixed)
            # The structural check keeps markup out, but a control character is not valid XML: the
            # response must parse, or the caller hears the fixed line instead.
            if twiml_parses(self._render(lead[0], clause[0])):
                return [lead, clause]
            obs.emit(log, logging.WARNING, "briefing_unparseable", call_sid=st.call_sid,
                     decision="fixed_line")
        return [fixed]

    def _handoff_conversation(self, st: CallState, known: str | None) -> str | None:
        """The call's Orchestrator conversation for the task attributes: ``known`` (the
        bridge's hand-off, else the session's), else one lookup by CallSid when the human-leg
        client is wired (``resolve_conversation``: ``None`` for zero or several ACTIVE ones, and
        under its own short timeout). A call that never had an AI session has no conversation, so
        no lookup is made. ``None`` still escalates; nothing is written."""
        call_sid = st.call_sid
        if known or self.d.human_leg is None or st.ai_sessions == 0:
            return known
        try:
            found = self.d.human_leg.resolve_conversation(call_sid)
        except Exception as exc:  # the client never raises; this guards the escalation anyway
            obs.log_caught(log, "human_leg_resolve_failed", exc, call_sid=call_sid,
                           operation="resolve", decision="ignored_best_effort")
            return None
        return found if isinstance(found, str) and found else None

    def _journal_handoff(self, st: CallState, conversation_id: str | None) -> None:
        """Journal ``handoff`` (``to: human``) and log it with the conversation id.
        A failure is logged and ignored: the ``<Enqueue>`` never depends on it."""
        try:
            self._log_event(st.call_sid, {"type": "handoff", "to": "human", "level": st.level,
                                          "conversation_id": conversation_id})
            outcome = "journaled"
        except Exception as exc:
            obs.log_caught(log, "human_leg_handoff_journal_failed", exc, call_sid=st.call_sid,
                           operation="journal_handoff", decision="ignored_best_effort")
            outcome = "journal_failed"
        obs.emit(log, logging.INFO, "human_leg_handoff", call_sid=st.call_sid,
                 conversation_id=conversation_id, outcome=outcome)

    # ------------------------------------------------------------------ the queue

    def enqueue_action(self, req: http.TwilioRequest) -> str:
        """The ``<Enqueue action>``: Twilio requests it when the caller is out of the
        queue. Decided by ``QueueResult`` alone (never by how long the caller waited): the
        goodbye when the caller left the queue or the agent leg ended (:data:`QUEUE_LEFT_RESULTS`),
        the apology and goodbye when the enqueue failed, and an empty answer otherwise (hung up,
        or the call is on its way elsewhere), which never adds a ``<Hangup>``. Never escalates."""
        result = (req.params.get("QueueResult") or "").strip().lower()
        known = result in QUEUE_LEFT_RESULTS or result in QUEUE_FAILED_RESULTS or result in (
            "hangup", "redirected", "bridging-in-process", "redirected-from-bridged")
        obs.emit(log, logging.ERROR if result in QUEUE_FAILED_RESULTS else logging.INFO,
                 "queue_exit", call_sid=req.call_sid, result=result if known else "unknown")
        if result not in QUEUE_LEFT_RESULTS and result not in QUEUE_FAILED_RESULTS:
            return twiml.empty()
        lang = self.d.catalog.default_language
        try:
            st = self._state(req.call_sid) if req.call_sid else None
            if st is not None and st.language:
                lang = st.language
        except Exception as exc:
            obs.log_caught(log, "queue_exit_state_unavailable", exc, call_sid=req.call_sid,
                           operation="load_call_state", decision="default_language")
        keys = (["error.generic"] if result in QUEUE_FAILED_RESULTS else []) + ["common.goodbye"]
        verbs: list[Any] = []
        for key in keys:
            try:
                verbs.append(self._say(key, lang)[0])
            except Exception as exc:
                obs.log_caught(log, "queue_exit_text_unavailable", exc, call_sid=req.call_sid,
                               operation="catalog_text", decision="skipped")
        return self._render(*verbs, twiml.Hangup())

    def _untrusted(self, st: CallState) -> bool:
        """An untrusted caller ID, or a keyed-in number, which is never trusted
        whatever the call's ``StirVerstat``."""
        return st.identity_source == KEYED_PHONE or not self.d.policy.ani_trusted(st.stir_verstat)

    def _auth_identity(self, st: CallState) -> tuple[str | None, str | None, str]:
        """``(ani, stir_verstat, identity_source)`` for every auth-service call: the call's own ANI and ``StirVerstat`` with ``"ani"``, or on a call identified
        by a keyed-in number that number (unsealed from the call's sensitive store,
        :func:`sensitive.recall`), ``None`` and ``"keyed_phone"``, so the auth service treats the
        call as untrusted with no ``ani_match`` whatever ``on_untrusted`` says, and sends a code
        only to that number, which is on file for the customer. A keyed call whose number can't be
        recalled fails closed (:class:`_Abort`: the apology and a human). The number is never
        logged."""
        if st.identity_source != KEYED_PHONE:
            return st.ani, st.stir_verstat, IDENTITY_ANI
        number = sensitive.recall(st.call_sid, st.sealed_sensitive, slot=KEYED_NUMBER_RECORD)
        if number is None:
            obs.emit(log, logging.ERROR, "keyed_number_unavailable", call_sid=st.call_sid,
                     decision="apology_and_escalate")
            raise _Abort("keyed number unavailable")
        return number, None, KEYED_PHONE

    def _risk_escalate(self, req: http.TwilioRequest, st: CallState, sess: Session,
                       cause: str) -> str:
        """Transfer to a human with reason ``verification_risk``."""
        self._log_event(st.call_sid, {"type": "verification", "outcome": VERIFICATION_RISK,
                                      "cause": cause, "level": st.level})
        return self._escalate(req, st, sess, VERIFICATION_RISK, "")

    def _otp_unavailable(self, req: http.TwilioRequest, st: CallState, sess: Session) -> str:
        """Transfer to a human with reason ``otp_unavailable``: no code was sent.
        The call's level stays as reached; the neutral phrase comes from the policy route."""
        self._log_event(st.call_sid, {"type": "verification", "outcome": OTP_UNAVAILABLE,
                                      "level": st.level})
        return self._escalate(req, st, sess, OTP_UNAVAILABLE, "")

    def _risk_check(self, req: http.TwilioRequest, st: CallState, sess: Session,
                    target: int) -> str | None:
        """The auth service's untrusted-caller gate before the first factor is asked.

        ``None`` to go on; otherwise the TwiML to return (escalation, or the error path)."""
        if not self._untrusted(st):
            return None  # a trusted caller never touches the untrusted-attempt counter
        ani, verstat, source = self._auth_identity(st)
        res = self.slots.api_call(sess, lambda: self.d.backend.precheck(
            call_sid=st.call_sid, ani=ani, target_level=target, stir_verstat=verstat,
            identity_source=source))
        if res == "ok":
            return None
        if res == "escalate":
            return self._risk_escalate(req, st, sess, "precheck")
        self._log_event(st.call_sid, {"type": "verification", "outcome": "precheck_error",
                                      "level": st.level})
        return self._give_up(req, st, sess, "error")

    def _withheld_summary(self, lang: str) -> str:
        try:
            return self.d.catalog.text("escalation.summary_withheld", lang)
        except Exception as exc:
            obs.log_caught(log, "withheld-summary text unavailable", exc, operation="catalog_text",
                           decision="fallback_builtin_text")
            return WITHHELD_TURN

    def _apology(self, req: http.TwilioRequest) -> str:
        lang = self.d.catalog.default_language
        st = sess = None
        try:
            st, sess = self._load(req.call_sid)
            lang = st.language
        except Exception as exc:
            obs.log_caught(log, "apology without call state", exc, operation="load_call_state",
                           decision="fallback_default_language_no_escalation")
        try:
            apology = self._say("error.generic", lang)
        except Exception as exc:
            obs.log_caught(log, "apology text unavailable", exc, level=logging.ERROR,
                           operation="catalog_text", decision="hangup")
            return self._render(twiml.Hangup())
        if st is not None and sess is not None:
            try:
                return self._escalate(req, st, sess, "unable_to_resolve", "", pre=[apology])
            except Exception as exc:
                obs.log_caught(log, "fallback escalation failed", exc, level=logging.ERROR,
                               call_sid=req.call_sid, decision="apology_and_hangup")
        try:
            bye = self._say("common.goodbye", lang)
            return self._render(apology[0], bye[0], twiml.Hangup())
        except Exception as exc:
            obs.log_caught(log, "goodbye text unavailable", exc, operation="catalog_text",
                           decision="apology_and_hangup")
            return self._render(apology[0], twiml.Hangup())

    # ------------------------------------------------------------------ /voice/incoming

    def incoming(self, req: http.TwilioRequest) -> str:
        sid = req.call_sid
        if not sid:
            raise _Abort("no CallSid")
        raw_from = (req.params.get("From") or "").strip()
        ani = None if raw_from.lower() in WITHHELD_ANIS else raw_from
        raw_verstat = (req.params.get("StirVerstat") or "").strip()
        verstat = raw_verstat if _VERSTAT.match(raw_verstat) else None  # absent: untrusted
        trusted = self.d.policy.ani_trusted(verstat)
        now = self._now()
        st = self._state(sid)
        if st is None:
            st = callstate.create(self.d.store, CallState(
                call_sid=sid, expires_at=now + self.cfg.record_ttl_s,
                language=self.d.catalog.default_language, ani=ani, stir_verstat=verstat,
                ani_trusted=trusted))
            self._prewarm(sid)
            sess = Session(sid, st.expires_at)
            self._log_event(sid, {"type": "call_start", "ani_present": ani is not None,
                                  "stir_verstat": verstat, "ani_trusted": trusted,
                                  "ani_match_counts": self.d.policy.ani_match_counts(verstat)})
            st, sess = self._lookup(st, sess)
            self._save(sess)
        return self._language_gather(req)

    def _prewarm(self, sid: str) -> None:
        """``agentcore``: start the new call's Runtime session now, so the microVM
        boots behind the language and ZIP prompts. Only after ``callstate.create`` and only for a
        CallSid. Asynchronous (an Event invoke): the TwiML never waits for it, and a
        dispatch error is logged and ignored (the call pays the cold start at connect)."""
        if self.cfg.hosting != "agentcore" or self.d.warmer is None or not CALL_SID_RE.match(sid):
            return
        try:
            self.d.warmer(sid)
        except Exception as exc:
            obs.log_caught(log, "runtime_prewarm_dispatch_failed", exc, call_sid=sid,
                           decision="ignored_cold_start_at_connect")

    def _lookup(self, st: CallState, sess: Session) -> tuple[CallState, Session]:
        """ANI dip: candidates only, never identity."""
        if st.ani is None:
            sess.lookup = "withheld"
            outcome, count = "withheld", 0
        else:
            ani = st.ani
            res = self.slots.api_call(sess, lambda: self.d.backend.ani_lookup(
                call_sid=st.call_sid, ani=ani))
            if res.status == "ok":
                sess.lookup = "ok"
                st = self._update(st, candidate_ids=res.candidate_ids)
                count = len(res.candidate_ids)
                outcome = "no_match" if count == 0 else ("match" if count == 1 else "multiple")
            else:
                sess.lookup, outcome, count = "error", "error", 0
        self._log_event(st.call_sid, {"type": "lookup", "outcome": outcome, "count": count})
        return st, sess

    def _language_gather(self, req: http.TwilioRequest) -> str:
        """The welcome first, so callers know they reached the bank, then the menu
        for the other language. Both in the default language's voice (one voice for the opening);
        the menu keeps its own language for pronunciation."""
        default = self.d.catalog.default_language
        sel_lang = "es-US" if "es-US" in self.d.catalog.languages else default
        welcome = self._say("greeting.generic", default)
        # The menu line in a male Spanish voice (``say.menuVoice``) when the catalog sets one;
        # else the default language's voice.
        menu = self._say_voice("language.select", sel_lang, voice_lang=default,
                               voice=self.d.catalog.settings(sel_lang).get("say", {}).get("menuVoice"))
        self._prompts(req.call_sid, [welcome, menu])
        return self._render(
            twiml.Gather(action=req.url("language"), prompts=[welcome[0], menu[0]], input="dtmf",
                         num_digits=1, timeout=self.cfg.language_timeout_s,
                         action_on_empty_result=True),
            twiml.Redirect(req.url("language")),
        )

    # ------------------------------------------------------------------ /voice/language

    def language(self, req: http.TwilioRequest) -> str:
        st, sess = self._load(req.call_sid)
        digits = (req.params.get("Digits") or "").strip()
        lang = "es-US" if digits == "2" and "es-US" in self.d.catalog.languages \
            else self.d.catalog.default_language
        st = self._update(st, language=lang)
        self._log_event(st.call_sid, {"type": "prompt", "prompt_key": "language.select",
                                      "outcome": lang})
        if sess.lookup in ("pending", "error"):  # one retry, now the caller has chosen
            st, sess = self._lookup(st, sess)
        if sess.lookup == "error":
            return self._fallback(req, st, sess, "lookup_error",
                                  pre=[self._say("error.lookup_unavailable", lang)])
        # The opening already welcomed the caller in the default language; a caller
        # who chose another language hears the welcome once more, in theirs.
        pre = [] if lang == self.d.catalog.default_language else [self._say("greeting.generic", lang)]
        if not st.candidate_ids:  # no match, or ANI withheld
            if self._ask_phone(st, sess):  # Ask the account number first
                slotflow.start_flow(self, sess, "login", tool=None, target_level=1)
                return self._ask(req, st, sess, PHONE_SLOT, pre=pre)
            return self._fallback(req, st, sess, "no_match")
        factors = self._missing(st, 1)
        if not factors:
            return self._fallback(req, st, sess, "no_match")
        slotflow.start_flow(self, sess, "login", tool=None, target_level=1)
        gate = self._risk_check(req, st, sess, 1)
        if gate is not None:
            return gate
        return self._ask(req, st, sess, factors[0], pre=pre)

    # ------------------------------------------------------------------ keyed-in phone login

    def _ask_phone(self, st: CallState, sess: Session) -> bool:
        """Ask for the account phone number when the switch is on,
        the ANI lookup ran and found no candidate (or the ANI is withheld), and no keyed lookup has
        run on this call yet. Never while the ``phone`` slot or its prompts aren't loaded (new code
        with an old policy or prompt table, before the reseed): the call takes today's ``no_match``
        path instead, logged once per instance."""
        return (self.d.policy.ani_trust.ask_phone_on_no_match
                and sess.lookup in ("ok", "withheld") and not st.candidate_ids
                and sess.keyed_tries == 0 and st.identity_source != KEYED_PHONE
                and self._phone_ready())

    def _phone_ready(self) -> bool:
        """Whether the ``phone`` slot, its question and its retry question are loaded."""
        try:
            slot = self.d.policy.slot(PHONE_SLOT)
        except KeyError:
            missing = f"slot {PHONE_SLOT}"
        else:
            keys = self.d.catalog.keys
            missing = next((k for k in (slot.decl.prompt_key or f"factor.{PHONE_SLOT}",
                                        PHONE_RETRY_KEY) if k not in keys), "")
        if not missing:
            return True
        if not self._phone_unready_logged:
            self._phone_unready_logged = True
            obs.emit(log, logging.WARNING, "keyed_phone_unavailable", missing=missing,
                     decision="no_match")
        return False

    def _keyed_event(self, call_sid: str, outcome: str, count: int) -> None:
        """``keyed_lookup``: the outcome and the candidate count only, never the number."""
        obs.emit(log, logging.INFO, "keyed_lookup", call_sid=call_sid, outcome=outcome, count=count)
        self._log_event(call_sid, {"type": "keyed_lookup", "outcome": outcome, "count": count})

    def _claim_phone(self, req: http.TwilioRequest, st: CallState, sess: Session,
                     digits: str) -> str:
        """The ``phone`` slot's accepted value: ``"+1"`` + the 10 digits,
        looked up with the existing ``ani_lookup`` (retried once on error, as the ANI lookup is).
        Candidates: the number is sealed (``sensitive.remember``), the call is marked
        ``keyed_phone`` (never trusted) and login goes on as for a matched ANI, with the
        untrusted factor set. No match: asked once more (``login.phone_number_retry``, which says
        nothing about the number), then the ``no_match`` fallback. The number is never logged
        or journaled; ``keyed_lookup`` carries the outcome and count only."""
        sid = st.call_sid
        if not _PHONE_DIGITS.fullmatch(digits):  # the slot engine accepts 10 digits only
            return self._give_up(req, st, sess, "unclear")
        sess.keyed_tries = int(sess.keyed_tries) + 1
        if st.ani_trusted:
            # A caller who keyed in a number is untrusted whatever
            # the lookup finds (match, no match, error), so the recap never goes to the caller-ID
            # profile. Only the recap reads this field; verification reads the verstat and
            # identity_source, so nothing else changes for the call.
            st = self._update(st, ani_trusted=False)
        number = "+1" + digits
        # Registered for the exact-value scrub and sealed, also when it finds no one.
        self._remember(sid, self._slot(PHONE_SLOT), number)
        res = None
        for _ in range(2):  # one retry on error, as the ANI lookup
            res = self.slots.api_call(sess, lambda: self.d.backend.ani_lookup(call_sid=sid,
                                                                             ani=number))
            if res.status == "ok":
                break
        if res is None or res.status != "ok":
            self._keyed_event(sid, "error", 0)
            return self._fallback(req, st, sess, "lookup_error",
                                  pre=[self._say("error.lookup_unavailable", st.language)])
        count = len(res.candidate_ids)
        if count == 0:
            self._keyed_event(sid, "no_match", 0)
            if sess.keyed_tries < MAX_KEYED_TRIES:
                return self._ask(req, st, sess, PHONE_SLOT, prompt_key=PHONE_RETRY_KEY)
            return self._fallback(req, st, sess, "no_match")
        self._keyed_event(sid, "match" if count == 1 else "multiple", count)
        if not sensitive.remember(self.d.store, self.d.sealer, sid, number,
                                  slot=KEYED_NUMBER_RECORD, slot_type="digits", now=self.d.clock):
            return self._give_up(req, st, sess, "error")  # not sealed for later webhooks: fail closed
        st = self._update(st, candidate_ids=res.candidate_ids, identity_source=KEYED_PHONE,
                          ani_trusted=False)
        flow = sess.flow or {}
        factors = self._missing(st, int(flow.get("target_level", 1)))
        if not factors:
            return self._fallback(req, st, sess, "no_match")
        gate = self._risk_check(req, st, sess, 1)
        if gate is not None:
            return gate
        return self._ask(req, st, sess, factors[0])

    # ------------------------------------------------------------------ fallbacks

    def _fallback(self, req: http.TwilioRequest, st: CallState, sess: Session, outcome: str,
                  *, pre: list[tuple[Any, Spoken]] | None = None,
                  envelope: ResultEnvelope | None = None) -> str:
        """The bank's choice for a login that ended with ``outcome``. A failed (or locked)
        or declined login (:data:`CLOSING_OUTCOMES`) closes verification for the call
        and, in the AI, is welcomed with :data:`FAILED_AI_KEY` alone; escalated, it keeps
        ``verify.failed`` before the hand-off."""
        action = self.cfg.fallbacks.get(outcome, "limited_ai")
        self._log_event(st.call_sid, {"type": "verification", "outcome": f"fallback_{outcome}",
                                      "action": action, "level": st.level})
        closing = outcome in CLOSING_OUTCOMES
        if closing:
            st = self._close_login(st)
        if action == "escalate":
            reason = "step_up_failed" if outcome == "verify_failed" else "unable_to_resolve"
            if closing:
                pre = (pre or []) + [self._say("verify.failed", st.language)]
            return self._escalate(req, st, sess, reason, "", pre=pre)
        if closing and FAILED_AI_KEY not in self.d.catalog.keys:
            # A catalog seeded before the failed-AI key existed: today's apology and the unverified greeting.
            return self._limited_ai(req, st, sess, envelope=envelope,
                                    pre=(pre or []) + [self._say("verify.failed", st.language)])
        return self._limited_ai(req, st, sess, pre=pre or [], envelope=envelope,
                                welcome_key=FAILED_AI_KEY if closing else None)

    def _close_login(self, st: CallState) -> CallState:
        """``login_closed`` in call state, so no second verification starts in this call.
        A failed write is logged and the login still ends on its fallback; the bridge
        then can't see the flag, and the per-customer lockout still caps guesses."""
        try:
            return self._update(st, **{callstate.F_LOGIN_CLOSED: True})
        except Exception as exc:
            obs.log_caught(log, "login_closed not saved", exc, level=logging.ERROR,
                           operation="state_save", call_sid=st.call_sid, decision="fallback_anyway")
            return st

    def _limited_ai(self, req: http.TwilioRequest, st: CallState, sess: Session, *,
                    pre: list[tuple[Any, Spoken]], envelope: ResultEnvelope | None = None,
                    welcome_key: str | None = None) -> str:
        """Unauthenticated AI with level-0 tools only. ``welcome_key`` replaces the
        unverified greeting (``verify.failed_ai`` after a failed or declined login)."""
        lang = st.language
        if envelope is not None and envelope.next == "handle_utterance":
            welcome = self._say(self.cfg.bridge_phrase_key, lang)[1]
        elif welcome_key is not None:
            welcome = self._say(welcome_key, lang)[1]
        else:
            key = next(k for k in self.cfg.unverified_greeting_keys if k in self.d.catalog.keys)
            welcome = self._say(key, lang)[1]
        if envelope is not None:
            envelope = replace(envelope, spoken=[*envelope.spoken, *[s for _, s in pre], welcome])
        return self._connect(req, st, sess, pre, welcome, envelope)

    # ------------------------------------------------------------------ factor prompts

    def _early_eligible(self, slot: SlotSettings) -> bool:
        """A Gather that may finish early, with early finish on and a Twilio updater wired:
        spoken digits of one fixed length, or a choice slot's question. Yes/no and every
        other slot: never. A keypad-only slot never."""
        decl = slot.decl
        length = decl.constraints.get("length")
        fixed_digits = (decl.slot_type == "digits" and isinstance(length, int)
                        and not isinstance(length, bool))
        choice = decl.slot_type == "choice" and bool(slot.options)
        return (self.cfg.early_digits and self.d.calls is not None and (fixed_digits or choice)
                and not gather_input(slot, self.cfg.accept_spoken_factors).keypad_only)

    @staticmethod
    def _early_flow(flow: Mapping[str, Any], slot: SlotSettings) -> bool:
        """Whether a partial for ``slot``'s Gather may be taken in this flow state.
        Digits, as before: not in the confirm flow, not while a follow-up is pending. A choice: the
        slot the flow is asking (in the confirm flow only ``card``), with no follow-up pending or
        one that lists options (a disambiguation, not available with several cards; the clarify
        that lists them has none). Never while a code is being sent."""
        if flow.get("otp_wait") or flow.get("factor") != slot.name:
            return False
        pending = flow.get("pending")
        if slot.decl.slot_type != "choice":
            return flow.get("kind") != "confirm" and not pending
        if flow.get("kind") == "confirm" and slot.name != CARD_SLOT:
            return False
        if pending is None:
            return True
        return (isinstance(pending, Mapping) and pending.get("slot") == slot.name
                and pending.get("kind") in slotflow.LISTING_KINDS)

    def _new_gather(self, flow: dict[str, Any], slot: SlotSettings) -> str | None:
        """A fresh Gather id in ``flow["gather"]`` (with its start time) for an eligible slot,
        else none (and any old one removed). A choice question gets one only when its partials
        could be taken (:meth:`_early_flow`). The caller saves the session."""
        if not self._early_eligible(slot) or (
                slot.decl.slot_type == "choice" and not self._early_flow(flow, slot)):
            flow.pop("gather", None)
            return None
        gid = "".join(_stdlib_secrets.choice(_GATHER_ID_ALPHABET) for _ in range(16))
        flow["gather"] = {"id": gid, "started_ms": int(self.d.clock() * 1000)}
        return gid

    def _gather(self, req: http.TwilioRequest, lang: str, slot: SlotSettings,
                prompts: list[Any], gather_id: str | None = None, *,
                follow_up: bool = False) -> list[twiml.Verb]:
        """The slot's prompts and ``<Gather>``. ``barge_in: true``: the prompts nested in
        the Gather, as before. ``barge_in: false`` (``bargeIn="false"`` doesn't stop
        a key press): the prompts play first and the Gather nests none, with the same attributes,
        so the caller can't cut the prompt (keys pressed during it may not be collected).
        ``follow_up``: a yes/no follow-up about the slot (a read-back): the slot's input and
        barge-in settings, but no digit count and the confirmation's speech timeout."""
        shape = gather_input(slot, self.cfg.accept_spoken_factors)
        nested, before = (prompts, []) if shape.barge_in else ([], prompts)
        return [*before, self._gather_verb(req, lang, slot, nested, gather_id, shape.keypad_only,
                                           follow_up=follow_up)]

    def _gather_verb(self, req: http.TwilioRequest, lang: str, slot: SlotSettings,
                     prompts: list[Any], gather_id: str | None, keypad_only: bool, *,
                     follow_up: bool = False) -> twiml.Gather:
        g = self.d.catalog.settings(lang).get("gather", {})
        decl = slot.decl
        length = decl.constraints.get("length")
        num_digits = length if isinstance(length, int) and not follow_up else None
        if keypad_only:
            # A keypad-only slot (its own ``keypad_only``, or a verification factor
            # while spoken factors are off) listens for the keypad only, so speech or line noise
            # can't end the gather before the caller has finished typing. No speech settings,
            # hints or Gather id (no early finish): Twilio would ignore them.
            return twiml.Gather(
                action=req.url("factor"), prompts=prompts, input="dtmf", num_digits=num_digits,
                timeout=self.cfg.factor_timeout_s, finish_on_key="#", action_on_empty_result=True,
            )
        labels = {o["label_key"]: self.d.catalog.text(o["label_key"], lang)
                  for o in decl.options() if o["label_key"]}
        # Yes/no answers come as phrases ("Yeah, that's correct"), so they wait longer
        # for the end of speech than digit factors do.
        speech_timeout = (self.cfg.confirm_speech_timeout_s if decl.slot_type == "yes_no" or follow_up
                          else self.cfg.speech_timeout_s)
        # An eligible Gather reports partial results, and its id travels in both URLs so
        # the final webhook and the partials name the same Gather.
        early = {"partialResultCallback": req.url("partial", g=gather_id),
                 "partialResultCallbackMethod": "POST"} if gather_id else {}
        return twiml.Gather(
            action=req.url("factor", g=gather_id) if gather_id else req.url("factor"),
            prompts=prompts, input="dtmf speech",
            num_digits=num_digits, speech_timeout=speech_timeout,
            timeout=self.cfg.factor_timeout_s, language=g.get("language", lang),
            # Twilio requires an integer speechTimeout with a speechModel, so "auto" runs on
            # Twilio's default model rather than send a combination the docs rule out.
            speech_model=None if speech_timeout == "auto" else g.get("speechModel"),
            hints=[] if follow_up else self._hints(slot, labels, lang),
            finish_on_key="#", action_on_empty_result=True, extra_attrs=early,
        )

    def _hints(self, slot: SlotSettings, labels: Mapping[str, str], lang: str) -> list[str]:
        """Speech hints: a ``constraints.options`` slot's labels, and a policy choice slot's
        spoken names and aliases in ``lang`` (each once, in policy order)."""
        out = hints(slot.decl, labels)
        if slot.decl.slot_type == "choice":
            for view in self.slots.option_views(slot, lang):
                out.extend([view.name, *view.aliases])
        return list(dict.fromkeys(h for h in out if h))

    def _ask(self, req: http.TwilioRequest, st: CallState, sess: Session, factor: str, *,
             pre: list[tuple[Any, Spoken]] | None = None, ack_first: bool = False,
             otp_sent: bool = False, prompt_key: str | None = None) -> str:
        """Ask one factor (sending the OTP first for ``verify_sms`` factors).

        ``ack_first``: ``pre`` is the acknowledgement, which may be spoken on its own while
        the code is sent. ``otp_sent``: the code already went out (the ``otp_wait``
        redirect saw ``sent``): ask without sending. ``prompt_key``: another question than the
        slot's own (the keyed-phone retry)."""
        flow = sess.flow or {}
        lang = st.language
        definition = self.d.policy.factor(factor) or {}
        sms = definition.get("channel") == "verify_sms" and not otp_sent
        if sms and st.level == 0 and self._untrusted(st):
            # Never send a code to an untrusted caller before login.
            return self._risk_escalate(req, st, sess, "otp_before_login")
        self._request_welcome(st, sess, factor)
        if sms:
            if ack_first and pre:
                started = self._ack_while_sending(req, st, sess, factor, pre)
                if started is not None:
                    return started
            ani, verstat, source = self._auth_identity(st)   # A number on file only
            sent = self.slots.api_call(sess, lambda: self.d.backend.send_otp(
                call_sid=st.call_sid, ani=ani, language=lang, stir_verstat=verstat,
                identity_source=source))
            if sent == "escalate":  # No code to any number but the caller's own
                return self._otp_unavailable(req, st, sess)
            if sent != "sent":
                return self._give_up(req, st, sess, "error", pre=pre)
        slot = self._slot(factor)
        prompt = self._say(self.slots._prompt_key_for(slot, lang, prompt_key), lang)
        items = (pre or []) + [prompt]
        flow.update(factor=factor, repeats=0, reminders=0, pending=None)
        slotflow.set_tries(flow, factor, 0)
        flow["spoken"] = [*flow.get("spoken", []), *[{"template": s.template, "text": s.text}
                                                      for _, s in items]]
        gid = self._new_gather(flow, slot)
        sess.flow = flow
        self._save(sess)
        self._prompts(st.call_sid, items)
        self._log_event(st.call_sid, {"type": "factor_asked", "factor": factor})
        return self._render(*self._gather(req, lang, slot, [v for v, _ in items], gather_id=gid))

    def _ack_while_sending(self, req: http.TwilioRequest, st: CallState, sess: Session,
                           factor: str, ack: list[tuple[Any, Spoken]]) -> str | None:
        """Start the code send asynchronously and speak only the acknowledgement, then
        redirect back to ``/voice/factor`` (:meth:`_otp_wait`). ``None`` when the asynchronous
        invoke couldn't be queued: the caller sends synchronously instead."""
        sid = st.call_sid
        if st.otp_send is not None:  # a stale result from an earlier send must not be read
            st = self._update(st, **{callstate.F_OTP_SEND: None})
        started_ms = int(self.d.clock() * 1000)
        ani, verstat, source = self._auth_identity(st)   # Same ani as send, verify
        lang = st.language
        if not self.slots.api_call(sess, lambda: self.d.backend.send_otp_async(
                call_sid=sid, ani=ani, language=lang, stir_verstat=verstat,
                identity_source=source)):
            obs.emit(log, logging.WARNING, "otp_send_async_unavailable", call_sid=sid,
                     decision="synchronous_send")
            return None
        flow = sess.flow or {}
        flow.update(factor=None, otp_wait={"started_ms": started_ms, "factor": factor})
        flow["spoken"] = [*flow.get("spoken", []), *[{"template": s.template, "text": s.text}
                                                      for _, s in ack]]
        sess.flow = flow
        self._save(sess)
        self._prompts(sid, ack)
        obs.emit(log, logging.INFO, "otp_send_async", call_sid=sid, factor=factor)
        return self._render(*[v for v, _ in ack], twiml.Redirect(req.url("factor"), method="POST"))

    def _otp_wait(self, req: http.TwilioRequest, st: CallState, sess: Session) -> str:
        """The redirect after "Got it.": wait for the asynchronous send's outcome (up to
        :data:`OTP_WAIT_S` since it started), clear it, then ask the code, escalate
        (``otp_unavailable``) or take the error path. Never sends a code itself."""
        flow = sess.flow or {}
        wait = flow.pop("otp_wait", None) or {}
        p = req.params
        if (p.get("Digits") or "").strip() or (p.get("SpeechResult") or "").strip():
            # No <Gather> was open: this can't be an answer to the code prompt. Never resolved.
            log.debug("input while waiting for the code send", extra={
                "call_sid": st.call_sid, "decision": "ignored_input_during_otp_wait"})
        raw_factor = wait.get("factor")
        factor = raw_factor if isinstance(raw_factor, str) and raw_factor else "otp"
        started_ms = wait.get("started_ms")
        started_ms = started_ms if isinstance(started_ms, int) and not isinstance(started_ms, bool) \
            else int(self.d.clock() * 1000)
        deadline_ms = started_ms + int(OTP_WAIT_S * 1000)
        status = self._otp_status(st)
        while status is None:
            left_ms = deadline_ms - int(self.d.clock() * 1000)
            if left_ms <= 0:
                break
            self.d.sleep(min(OTP_POLL_S, left_ms / 1000))
            fresh = self._state(st.call_sid)
            if fresh is None:
                raise _Abort("no call state")
            st = fresh
            status = self._otp_status(st)
        wait_ms = max(0, int(self.d.clock() * 1000) - started_ms)
        if st.otp_send is not None:
            st = self._update(st, **{callstate.F_OTP_SEND: None})
        sess.flow = flow
        self._save(sess)
        if status is None:
            obs.emit(log, logging.WARNING, "otp_send_timeout", call_sid=st.call_sid,
                     wait_ms=wait_ms, decision="give_up")
            self._log_event(st.call_sid, {"type": "verification", "outcome": "otp_send_timeout",
                                          "level": st.level})
            return self._give_up(req, st, sess, "error")
        obs.emit(log, logging.INFO, "otp_send_ready", call_sid=st.call_sid, status=status,
                 wait_ms=wait_ms)
        if status == "sent":
            return self._ask(req, st, sess, factor, otp_sent=True)
        if status == "escalate":
            return self._otp_unavailable(req, st, sess)
        return self._give_up(req, st, sess, "error")

    @staticmethod
    def _otp_status(st: CallState) -> str | None:
        """The ``otp_send`` status, or ``None`` while there is none. Junk counts as ``error``."""
        rec = st.otp_send
        if rec is None:
            return None
        status = rec.get("status") if isinstance(rec, Mapping) else None
        return status if status in callstate.OTP_SEND_STATUSES else "error"

    def _reprompt(self, req: http.TwilioRequest, st: CallState, sess: Session, slot: SlotSettings,
                  items: list[tuple[Any, Spoken]], *, follow_up: bool = False) -> str:
        """The question again (or a follow-up) in ``slot``'s Gather. A response holding generated
        text (:class:`GeneratedSpoken`) is parsed once before anything is saved or journaled; if it
        doesn't parse (a rendering defect, not model content), each generated text is replaced by
        its catalog template and ``generative_fallback`` (``twiml_invalid``) is logged at ERROR,
        never the text."""
        flow = sess.flow or {}
        if follow_up:
            flow.pop("gather", None)   # a yes/no follow-up never finishes early
            gid = None
        else:
            gid = self._new_gather(flow, slot)

        def render(its: list[tuple[Any, Spoken]]) -> str:
            return self._render(*self._gather(req, st.language, slot, [v for v, _ in its],
                                              gather_id=gid, follow_up=follow_up))

        out = render(items)
        generated = [sp for _, sp in items if isinstance(sp, GeneratedSpoken)]
        if generated and not twiml_parses(out):
            for sp in generated:
                obs.emit(log, logging.ERROR, "generative_fallback", call_sid=st.call_sid,
                         flow=flow.get("kind") if isinstance(flow.get("kind"), str) else None,
                         slot=slot.name, template=sp.template, reason="twiml_invalid",
                         decision="template")
            items = [it[1].fallback if isinstance(it[1], GeneratedSpoken) else it for it in items]
            out = render(items)
        flow["spoken"] = [*flow.get("spoken", []), *[{"template": s.template, "text": s.text}
                                                      for _, s in items]]
        sess.flow = flow
        self._save(sess)
        self._prompts(st.call_sid, items)
        return out

    # ------------------------------------------------------------------ /voice/partial

    def partial(self, req: http.TwilioRequest) -> None:
        """A ``partialResultCallback``: finish the Gather early when the partial already holds
        the whole answer. Never raises (the route always answers 204); never logs or
        journals the partial's text or digits."""
        try:
            self._partial(req)
        except Exception as exc:
            obs.log_caught(log, "factor_partial_failed", exc, route="partial",
                           call_sid=req.call_sid, decision="ignored")

    def _partial(self, req: http.TwilioRequest) -> None:
        if not (self.cfg.early_digits and self.d.calls is not None):
            return
        sid = req.call_sid
        gid = _gather_id(req.query.get("g"))
        text = _partial_text(req.params)
        now_ms = int(self.d.clock() * 1000)
        st = self._state(sid) if sid and gid else None
        if st is not None and gid is not None and telemetry.installed():
            self._count_partial(sid, gid, len(text), now_ms, st.expires_at)
        sess = sessions.load(self.d.store, sid) if st is not None else None
        flow = (sess.flow if sess else None) or {}
        raw_gather = flow.get("gather")
        current: Mapping[str, Any] = raw_gather if isinstance(raw_gather, Mapping) else {}
        name = flow.get("factor") if isinstance(flow.get("factor"), str) else None
        started = current.get("started_ms")
        started = started if isinstance(started, int) and not isinstance(started, bool) else None
        try:
            slot_type: str | None = self.d.policy.slot(name).decl.slot_type if name else None
        except KeyError:
            slot_type = None
        # Shape only: never the partial's text or a value.
        shape: dict[str, Any] = {"slot": name, "slot_type": slot_type, "chars": len(text),
                                 "ms_since_gather": None if started is None else now_ms - started}
        live = (st is not None and sess is not None and gid is not None and name is not None
                and slot_type is not None and current.get("id") == gid)
        slot = self._slot(name) if live and name else None
        if st is None or sess is None or gid is None or slot is None \
                or not self._early_eligible(slot) or not self._early_flow(flow, slot):
            obs.emit(log, logging.INFO, "factor_partial", call_sid=sid, outcome="stale", **shape)
            return
        decl = slot.decl
        lang = st.language
        # Every live partial of a phone Gather is newer than any partial before it, whatever
        # it holds, so it is recorded before the parse, in Twilio's order (SequenceNumber),
        # not arrival order: an older partial delivered late changes nothing and never claims; one
        # without a valid SequenceNumber stops any earlier one claiming and never claims itself.
        settle_id: str | None = None
        seq: int | None = None
        if self._settles(slot):
            seq = _sequence_number(req.params)
            settle_id, newest = self._mark_partial(sid, gid, st, seq)
            if not newest:
                obs.emit(log, logging.INFO, "factor_partial", call_sid=sid,
                         outcome="superseded" if seq is not None else "unsequenced", **shape)
                return
        label = self.d.catalog.text(decl.label_key, lang) if decl.label_key else None
        # The 2b label guard applies to an early finish too.
        others = tuple(c.label for c in self.slots.collected(sess, lang) if c.name != name)
        if decl.slot_type == "choice":
            outcome, value = early_choice(
                decl, text, lang, options=self.slots.caller_views(sess, slot, lang),
                listed=self.slots.listed_now(flow, slot),
                labels={decl.label_key: label} if decl.label_key and label is not None else {},
                other_labels=others, known=self.slots.option_views(slot, lang))
        elif slot.name == PHONE_SLOT:
            outcome, value = early_phone(decl, text, lang, label=label, other_labels=others)
        else:
            outcome, value = early_digits(decl, text, lang, label=label, other_labels=others)
        if outcome == "complete" and value is not None and settle_id is not None:
            # Wait for a newer partial (a correction); claim only if none came.
            obs.emit(log, logging.INFO, "factor_partial", call_sid=sid, outcome="settling",
                     settle_ms=self.cfg.phone_settle_ms, **shape)
            # Open the Twilio connection inside the wait, so the claim's update doesn't pay
            # its set-up (about 1 s on an environment that hasn't updated a call yet).
            waited_from = self.d.clock()
            self._warm_calls(sid)
            left = self.cfg.phone_settle_ms / 1000 - (self.d.clock() - waited_from)
            if left > 0:
                self.d.sleep(left)
            shape["ms_since_gather"] = (None if started is None
                                        else int(self.d.clock() * 1000) - started)
        if outcome == "complete" and value is not None:
            outcome = self._claim_early(req, st, gid, slot, value, started, settle_id=settle_id,
                                        settle_seq=seq)
        obs.emit(log, logging.INFO, "factor_partial", call_sid=sid, outcome=outcome, **shape)

    def _warm_calls(self, sid: str) -> None:
        """Make sure the call updater's connection is open (once per environment). A failure
        is logged and ignored: the claim's update just opens the connection itself."""
        try:
            if self.d.calls is not None:
                self.d.calls.ensure_warm()
        except Exception as exc:
            obs.log_caught(log, "twilio_connection_warm_failed", exc, call_sid=sid, decision="ignored")

    def _settles(self, slot: SlotSettings) -> bool:
        """The phone slot's partials wait ``phone_settle_ms`` before a claim (none at 0)."""
        return slot.name == PHONE_SLOT and self.cfg.phone_settle_ms > 0

    def _mark_partial(self, sid: str, gid: str, st: CallState, seq: int | None) -> tuple[str, bool]:
        """Record this partial (a random id, never the text); whether it is now the
        Gather's newest (:func:`gathers.mark_partial`)."""
        settle_id = "".join(_stdlib_secrets.choice(_GATHER_ID_ALPHABET) for _ in range(16))
        newest = gathers.mark_partial(self.d.store, sid, gid, settle_id=settle_id, seq=seq,
                                      expires_at=st.expires_at)
        return settle_id, newest

    def _count_partial(self, sid: str, gid: str, length: int, now_ms: int, expires_at: int) -> None:
        """Call observability: count the partial on its Gather's item; the length only,
        never the text. A failure is logged and ignored (the span then has no partial fields)."""
        try:
            gathers.count_partial(self.d.store, sid, gid, length=length, now_ms=now_ms,
                                  expires_at=expires_at)
        except Exception as exc:  # telemetry never changes the call
            obs.log_caught(log, "partial_count_failed", exc, call_sid=sid, decision="ignored")

    def _claim_early(self, req: http.TwilioRequest, st: CallState, gid: str, slot: SlotSettings,
                     value: str, started_ms: int | None, *, settle_id: str | None = None,
                     settle_seq: int | None = None) -> str:
        """Claim the Gather for its early answer, seal it (the digits, or a choice's option value)
        next to the session and end the Gather with a REST call update (one per Gather). ``complete`` or ``already_claimed``.
        With ``settle_id``, only while it is still the Gather's newest partial (its id and
        its ``settle_seq``; the id is only ever written with its sequence, so the CAS on the
        id covers both), else ``superseded`` (the newer partial owns the decision).
        A sensitive slot's value is remembered first, so the scrub has it from here on."""
        sid = st.call_sid
        name = slot.name
        store = self.d.store
        loaded = gathers.load(store, sid, gid)
        if gathers.state_of(loaded) != gathers.NONE:
            return "already_claimed"  # a partial, or the final webhook, got there first
        if settle_id is not None and (gathers.settle_id_of(loaded) != settle_id
                                      or gathers.settle_seq_of(loaded) != settle_seq):
            return "superseded"
        if slot.sensitive:
            self._remember(sid, slot, value)
        now = self._now()
        item = {"state": gathers.EARLY, "sealed_value": self.d.sealer.seal(value, call_sid=sid),
                "digits_expires_at": now + gathers.DIGITS_TTL_S, "started_ms": started_ms,
                "early_ms": int(self.d.clock() * 1000), "expires_at": st.expires_at,
                # The partial counters (present only with telemetry on) survive the claim.
                **gathers.partial_counters(loaded)}
        if not gathers.cas(store, sid, gid, gathers.NONE, item, settle_id=settle_id):
            if settle_id is not None and gathers.state_of(now_item := gathers.load(store, sid, gid)) \
                    == gathers.NONE and gathers.settle_id_of(now_item) != settle_id:
                return "superseded"
            return "already_claimed"
        try:
            if self.d.calls is None:
                raise RuntimeError("no call updater")
            url = req.absolute_url("factor", credentials=self.d.basic_credentials(), early=gid)
            self.d.calls.update_twiml(sid, twiml.render(twiml.Redirect(url, method="POST")))
        except Exception as exc:
            # A failed shortcut never breaks the flow; the Gather ends on its own timeout
            # and its final webhook takes the answer. The answer stays sealed (60 s) in case the
            # update did reach Twilio and its redirect still comes.
            obs.log_caught(log, "factor_early_update_failed", exc, call_sid=sid, slot=name,
                           decision="gather_finishes_on_its_own")
            gathers.take(store, sid, gid, allowed=(gathers.EARLY,),
                         changes={"state": gathers.RELEASED})
        return "complete"

    # ------------------------------------------------------------------ /voice/factor

    def factor(self, req: http.TwilioRequest) -> str:
        """A factor answer. With a Gather id (``early`` from the early redirect, ``g`` from
        the Gather's own action URL) exactly one request per Gather is answered; the others get
        its TwiML again. Without one, as before.

        18.12: the request is timed (:class:`owlbank.obs.Timings`): state load, the first policy
        and catalog access (a cache reload shows up there), the slot engine, render and state
        save. Each ``factor_answer`` line is written at the end with ``timings_ms``."""
        if slotflow.EARLY_VALUE in req.params:   # only a sealed claim carries an early value
            req = replace(req, params={k: v for k, v in req.params.items()
                                       if k != slotflow.EARLY_VALUE})
        timings = obs.Timings()
        self._timings = timings
        try:
            with timings.span("policy"):
                _ = self.d.policy.version
            with timings.span("catalog"):
                _ = self.d.catalog.keys
            gid = _gather_id(req.query.get("early"))
            early = gid is not None
            gid = gid or _gather_id(req.query.get("g"))
            if gid is None:
                return self._factor(req)
            return self._factor_once(req, gid, early=early)
        finally:
            self._timings = None
            self.slots.flush_answers(timings.to_dict())

    def _factor_once(self, req: http.TwilioRequest, gid: str, *, early: bool) -> str:
        sid = req.call_sid
        st0 = self._state(sid) if sid else None
        if st0 is None:
            raise _Abort("no call state")
        now_ms = int(self.d.clock() * 1000)
        allowed = (gathers.EARLY, gathers.RELEASED) if early else \
            (gathers.NONE, gathers.EARLY, gathers.RELEASED)
        won, prev = gathers.take(
            self.d.store, sid, gid, allowed=allowed,
            changes={"state": gathers.PROCESSING, "owner": "early" if early else "final",
                     "claimed_ms": now_ms},
            drop=("sealed_value", "digits_expires_at"), defaults={"expires_at": st0.expires_at})
        if not won:
            return self._gather_replay(req, gid, prev, early=early)
        if telemetry.installed():
            self._speech_timing(prev, now_ms, early=early)
        if early:
            req = self._early_request(req, prev or {}, now_ms)
        try:
            xml = self._factor(req)
        except Exception as exc:  # as dispatch does, but the answer is stored for the replays
            obs.log_caught(log, "handler error", exc, level=logging.ERROR, route=req.route,
                           call_sid=sid, decision="apology_and_escalate")
            obs.degraded("handler_error")
            xml = self._apology(req)
        gathers.take(self.d.store, sid, gid, allowed=(gathers.PROCESSING,),
                     changes={"state": gathers.DONE,
                              "sealed_twiml": gathers.seal_twiml(self.d.sealer, xml, call_sid=sid)})
        return xml

    @staticmethod
    def _speech_timing(item: Mapping[str, Any] | None, now_ms: int, *, early: bool) -> None:
        """Call observability on the ``gather.<slot>`` span: the partials counted on the Gather's
        item, the time from the last change in their length to this (the answering) request, and
        whether the Gather finished early. Absent counters (none came, or counting failed) are left out."""
        counters = gathers.partial_counters(item)
        last = counters.get("last_change_ms")
        telemetry.add_fields(partials=counters.get("partial_count"),
                             end_of_speech_ms=None if last is None else max(0, now_ms - last),
                             finished_early=early)

    def _early_request(self, req: http.TwilioRequest, claim: Mapping[str, Any],
                       now_ms: int) -> http.TwilioRequest:
        """The early redirect as the factor handler sees it. Digits: the sealed digits as
        ``SpeechResult`` (exactly as if Twilio had sent them). A choice: the option value
        as :data:`slotflow.EARLY_VALUE`, which the slot flow answers as ``match: exact, source:
        early`` with no parse or model call. Expired or unreadable: no answer (asked again like
        any unclear answer)."""
        sid = req.call_sid
        params = {k: v for k, v in req.params.items()
                  if k not in ("Digits", "SpeechResult", "Confidence", "FinishedOnKey",
                               slotflow.EARLY_VALUE)}
        with self._span("state_load"):
            sess = sessions.load(self.d.store, sid)
        flow = (sess.flow if sess else None) or {}
        name = flow.get("factor") if isinstance(flow.get("factor"), str) else None
        try:
            slot_type = self.d.policy.slot(name).decl.slot_type if name else None
        except KeyError:
            slot_type = None
        sealed = claim.get("sealed_value")
        expires = claim.get("digits_expires_at")
        if isinstance(sealed, str) and isinstance(expires, int) and self._now() <= expires:
            try:
                value = self.d.sealer.unseal(sealed, call_sid=sid)
            except UnsealError as exc:
                obs.log_caught(log, "early answer could not be unsealed", exc,
                               operation="unseal_early_value", decision="treat_as_no_answer")
            else:
                params[slotflow.EARLY_VALUE if slot_type == "choice" else "SpeechResult"] = value
        started, early_ms = claim.get("started_ms"), claim.get("early_ms")
        timeout = self.cfg.speech_timeout_s
        obs.emit(log, logging.INFO, "factor_early", call_sid=sid, slot=name, slot_type=slot_type,
                 ms_since_gather=now_ms - started if isinstance(started, int) else None,
                 ms_since_partial=now_ms - early_ms if isinstance(early_ms, int) else None,
                 speech_timeout_ms=timeout * 1000 if isinstance(timeout, int) else None,
                 digits_present="SpeechResult" in params,
                 value_present="SpeechResult" in params or slotflow.EARLY_VALUE in params)
        return replace(req, params=params)

    def _gather_replay(self, req: http.TwilioRequest, gid: str, item: Mapping[str, Any] | None,
                       *, early: bool) -> str:
        """Another request owns this Gather's answer: wait for its TwiML and return it.
        Never verifies."""
        sid = req.call_sid
        deadline = self.d.clock() + GATHER_REPLAY_WAIT_S
        while gathers.state_of(item) == gathers.PROCESSING and self.d.clock() < deadline:
            self.d.sleep(GATHER_REPLAY_POLL_S)
            item = gathers.load(self.d.store, sid, gid)
        state = gathers.state_of(item)
        obs.emit(log, logging.INFO, "factor_replay", call_sid=sid,
                 via="early" if early else "final", state=state)
        chunks = (item or {}).get("sealed_twiml")
        if state != gathers.DONE or not isinstance(chunks, list):
            raise _Abort("gather answer unavailable")
        try:
            return gathers.unseal_twiml(self.d.sealer, chunks, call_sid=sid)
        except Exception as exc:
            raise _Abort("stored gather answer unreadable") from exc

    def _factor(self, req: http.TwilioRequest) -> str:
        st, sess = self._load(req.call_sid)
        flow = sess.flow
        if flow and flow.get("factor") and not flow.get("otp_wait"):
            telemetry.set_step(f"gather.{flow['factor']}")
        if flow and flow.get("kind") == "confirm":
            return self._confirm_answer(req, st, sess)
        if flow and flow.get("otp_wait"):  # The redirect after "Got it."; no prompt is open
            return self._otp_wait(req, st, sess)
        if not flow or not flow.get("factor"):
            raise _Abort("no flow in progress")
        return self.slots.factor(req, st, sess, str(flow["factor"]))

    def _gather_ms(self, flow: Mapping[str, Any]) -> int | None:
        """Ms since the current early-finish Gather was sent (``None`` without one)."""
        g = flow.get("gather")
        started = g.get("started_ms") if isinstance(g, Mapping) else None
        if not isinstance(started, int) or isinstance(started, bool):
            return None
        return int(self.d.clock() * 1000) - started

    def _keypad_reminder(self, req: http.TwilioRequest, st: CallState, sess: Session,
                         slot: SlotSettings) -> str:
        """A spoken answer to a keypad-only slot (17.1; every factor while spoken factors are off):
        remind without using a try, up to :data:`MAX_KEYPAD_REMINDERS`; after that each
        one counts as a try."""
        flow = sess.flow or {}
        flow["reminders"] = int(flow.get("reminders", 0)) + 1
        if flow["reminders"] > MAX_KEYPAD_REMINDERS:
            return self._retry(req, st, sess, slot, "common.use_keypad")
        lang = st.language
        return self._reprompt(req, st, sess, slot,
                              [self._say("common.use_keypad", lang),
                               self._say(self.slots._prompt_key_for(slot, lang), lang)])

    def _retry(self, req: http.TwilioRequest, st: CallState, sess: Session, slot: SlotSettings,
               key: str) -> str:
        """One try of ``slot`` used outside the dialog policy (a keypad reminder past its cap)."""
        flow = sess.flow or {}
        tries = slotflow.tries_of(flow, slot.name) + 1
        slotflow.set_tries(flow, slot.name, tries)
        decl = slot.decl
        if tries > decl.max_retries:
            return self._give_up(req, st, sess, "unclear")
        lang = st.language
        if key == "clarify.digits":
            c = decl.constraints
            length = c.get("length") or c.get("maxLength") or ""
            label = self.d.catalog.text(decl.label_key, lang) if decl.label_key else ""
            items = [self._say(key, lang, length=length, label=label)]
        else:
            items = [self._say(key, lang), self._say(self.slots._prompt_key_for(slot, lang), lang)]
        return self._reprompt(req, st, sess, slot, items)

    def _off_script(self, req: http.TwilioRequest, st: CallState, sess: Session, kind: str,
                    utterance: str | None) -> str:
        """escalate / cancel / intent_switch: the flow ends. ``utterance`` is the answering
        turn's text, built once by the engine (9.2(a)): the caller's words for a
        non-sensitive slot, the ``[<slot name>]`` placeholder text for a sensitive one. It becomes
        the envelope's ``callerUtterance`` (and so call state's ``interrupted_utterance``); no
        front-door journal event or log line carries it.

        ``decline``: the caller can't or won't give a declinable slot's value.
        At the login phone question it is the ``no_match`` path; at a login security
        question the ``declined`` fallback (a ``cancel`` there too); at a
        step-up the ``declined`` result. No ``verify`` call and no attempt is recorded; the log and journal carry the factor name only."""
        flow = sess.flow or {}
        turn = utterance or None
        self._log_event(st.call_sid, {"type": "verification", "outcome": f"off_script_{kind}",
                                      "level": st.level})
        factor = flow.get("factor")
        if flow.get("kind") == "login":
            if kind == "escalate":
                return self._escalate(req, st, sess, "caller_requested_human", "")
            if factor == PHONE_SLOT and kind == "decline":
                self._keyed_event(st.call_sid, "declined", 0)
                return self._fallback(req, st, sess, "no_match")
            if factor != PHONE_SLOT and kind in ("decline", "cancel"):
                self._verification(st.call_sid, {"outcome": "declined", "factors": _names(factor),
                                                 "level": st.level})
                return self._fallback(req, st, sess, "declined")
            if kind == "cancel" or not turn:
                return self._fallback(req, st, sess, "cancelled")
            env = ResultEnvelope(status="interrupted", tool="authenticate",
                                 spoken=self._flow_spoken(flow),
                                 callerUtterance=turn, completed=False,
                                 next="handle_utterance")
            return self._limited_ai(req, st, sess, pre=[], envelope=env)
        if kind == "decline":
            self._verification(st.call_sid, {"outcome": "declined", "factors": _names(factor),
                                             "level": st.level})
        if kind in ("cancel", "decline") or not turn:
            return self._stepup_finish(req, st, sess, "declined")
        return self._stepup_finish(req, st, sess, "interrupted", utterance=turn)

    def _flow_spoken(self, flow: Mapping[str, Any]) -> list[Spoken]:
        return [Spoken(s["template"], s["text"]) for s in flow.get("spoken", [])]

    # ------------------------------------------------------------------ verification

    def _verify(self, req: http.TwilioRequest, st: CallState, sess: Session, name: str,
                value: str) -> str:
        telemetry.set_step("verify")
        flow = sess.flow or {}
        lang = st.language
        sid = st.call_sid
        factors = {name: value}
        held = flow.pop("held", None) or {}
        for held_name, sealed in held.items():  # disambiguation: re-send the earlier factor
            try:
                factors[held_name] = self.d.sealer.unseal(sealed, call_sid=sid)
            except UnsealError as exc:
                obs.log_caught(log, "held factor could not be unsealed", exc,
                               operation="unseal_held_factor", decision="give_up")
                return self._give_up(req, st, sess, "error")
        reach = self._level_for(self._have(st) | set(factors), st)
        pending = self._missing(st, int(flow.get("target_level", 1)), set(factors))
        if reach <= st.level and pending:
            # No level is reached yet (untrusted ANI: the next level's factors are needed
            # together). Hold what was collected, sealed, and ask the next factor.
            for k in factors:
                slotflow.collect(sess, k, self._slot(k).decl.slot_type, verified=False)
            flow["held"] = {k: self.d.sealer.seal(v, call_sid=sid) for k, v in factors.items()}
            self._verification(sid, {"outcome": "factor_held",
                                  "factors": sorted(factors), "level": st.level})
            return self._ask(req, st, sess, pending[0], pre=[self._say("factor.ack", lang)],
                             ack_first=True)
        target = reach or int(flow.get("target_level", 1))
        for k in factors:  # sent to the auth service: no longer correctable
            slotflow.collect(sess, k, self._slot(k).decl.slot_type, verified=True)
        ani, verstat, source = self._auth_identity(st)   # Keyed, untrusted
        res = self.slots.api_call(sess, lambda: self.d.backend.verify(
            call_sid=sid, ani=ani, factors=factors, target_level=target,
            stir_verstat=verstat, identity_source=source))
        if res.status == "escalate":  # verification_risk
            return self._risk_escalate(req, st, sess, "verify")
        if res.status == "verified":
            if res.token is None:  # security invariant: a verified result always carries a token
                raise RuntimeError("verified result without a token")
            fv = res.factors_verified or tuple(self.d.policy.factors(res.level))
            if st.identity_source == KEYED_PHONE:
                # Never ``ani_match``. The auth service never reports it for a
                # ``keyed_phone`` call; kept as a second check. The factors it checked are verified.
                fv = tuple(dict.fromkeys(
                    [f for f in fv if f not in ESTABLISHED_FACTORS] + sorted(factors)))
            st = self._update(st, level=max(st.level, res.level),
                              verified_customer_id=res.customer_id,
                              factors_verified=tuple(dict.fromkeys([*st.factors_verified, *fv])),
                              candidate_ids=(res.customer_id,) if res.customer_id else st.candidate_ids,
                              sealed_token=self.d.sealer.seal(res.token, call_sid=sid))
            self._verification(sid, {"outcome": "verified",
                                  "level": st.level, "factors": sorted(factors)})
            missing = self._missing(st, int(flow.get("target_level", 1)))
            if missing:
                return self._ask(req, st, sess, missing[0], pre=[self._say("factor.ack", lang)],
                                 ack_first=True)
            if flow.get("kind") == "login":
                return self._login_complete(req, st, sess)
            return self._stepup_finish(req, st, sess, "verified")
        if res.status == "ambiguous" and not held:
            self._verification(sid, {"outcome": "ambiguous",
                                  "factors": sorted(factors), "count": len(st.candidate_ids)})
            nxt = self._disambiguator(self._have(st) | set(factors))
            if nxt is None:
                return self._give_up(req, st, sess, "verify_failed")
            factor, level = nxt
            flow["target_level"] = max(int(flow.get("target_level", 1)), level)
            flow["held"] = {k: self.d.sealer.seal(v, call_sid=sid) for k, v in factors.items()}
            return self._ask(req, st, sess, factor)
        if res.status in ("not_verified", "ambiguous"):
            attempts = dict(st.attempts)
            attempts[name] = attempts.get(name, 0) + 1
            st = self._update(st, attempts=attempts)
            self._verification(sid, {"outcome": "not_verified",
                                  "factors": sorted(factors), "level": st.level})
            if sum(attempts.values()) >= self.d.policy.limit("verify_attempts_per_call") \
                    or res.attempts_remaining == 0:
                return self._give_up(req, st, sess, "verify_failed")
            # Re-ask from the first factor of this attempt (a held factor is asked again too).
            first = next(iter(held), name)
            return self._ask(req, st, sess, first, pre=[self._say("verify.retry", lang)])
        self._verification(sid, {"outcome": res.status,
                              "factors": sorted(factors), "level": st.level})
        return self._give_up(req, st, sess, "verify_failed" if res.status == "locked" else "error")

    def _verification(self, call_sid: str, fields: dict[str, Any]) -> None:
        """A verification outcome, journaled and logged as ``verification_result``: the outcome,
        the factor names and the level only, never a value (a misheard ZIP left no line)."""
        obs.emit(log, logging.INFO, "verification_result", call_sid=call_sid, **fields)
        self._log_event(call_sid, {"type": "verification", **fields})

    def _disambiguator(self, have: set[str]) -> tuple[str, int] | None:
        """The next keypad factor not yet used, and the level it belongs to."""
        policy = self.d.policy
        for n in range(1, policy.max_level + 1):
            for f in self._collectable(policy.factors(n)):
                if f not in have and (policy.factor(f) or {}).get("channel") == "dtmf":
                    return f, n
        return None

    def _give_up(self, req: http.TwilioRequest, st: CallState, sess: Session, why: str, *,
                 pre: list[tuple[Any, Spoken]] | None = None) -> str:
        flow = sess.flow or {}
        lang = st.language
        if flow.get("kind") == "login" and flow.get("factor") == PHONE_SLOT and why != "error":
            # Keyed-phone: no usable number is a caller we can't identify, not a
            # failed verification; it sounds like any other no-match call.
            self._keyed_event(st.call_sid, "unclear", 0)
            return self._fallback(req, st, sess, "no_match", pre=pre)
        if flow.get("kind") == "login":
            if why == "error":
                return self._fallback(req, st, sess, "lookup_error",
                                      pre=(pre or []) + [self._say("error.generic", lang)])
            # ``verify.failed`` (escalated) or ``verify.failed_ai`` (the AI) is said by _fallback.
            return self._fallback(req, st, sess, "verify_failed", pre=pre)
        return self._stepup_finish(req, st, sess, "failed" if why == "error" else "not_verified",
                                   pre=pre)

    def _login_complete(self, req: http.TwilioRequest, st: CallState, sess: Session) -> str:
        first = None
        if st.verified_customer_id:
            cust = self.d.store.get("customers", {"customer_id": st.verified_customer_id})
            if cust and isinstance(cust.get("first_name"), str):
                first = cust["first_name"]
        offer = self._offer_welcome(st, first)
        if offer is not None:
            return self._connect(req, st, sess, [], offer, None,
                                 welcome_offer=callstate.WELCOME_OFFER_MADE)
        welcome = self._say("greeting.personal", st.language, firstName=first)[1]
        return self._connect(req, st, sess, [], welcome, None)

    # ------------------------------------------------------------------ welcome offer (segment traits)

    def _request_welcome(self, st: CallState, sess: Session, factor: str) -> None:
        """When the last login factor is asked for an identified customer and
        ``WebActivity`` is listed, start the auth service's ``prepare_welcome`` (an ``Event``
        invoke), once per call (``Session.welcome_requested``, saved with the session by the
        caller). An identified customer is the verified one, else the only ANI candidate: an
        unknown ANI, or several candidates, triggers nothing. Never waits, never raises."""
        flow = sess.flow or {}
        if (not self.cfg.welcome_offers or sess.welcome_requested or flow.get("kind") != "login"
                or st.verified_customer_id is None and len(st.candidate_ids) != 1):
            return
        customer_id = st.verified_customer_id or st.candidate_ids[0]
        held = flow.get("held")
        extra = set(held) if isinstance(held, Mapping) else set()
        try:
            last = self._missing(st, int(flow.get("target_level", 1)), extra) == [factor]
        except Exception as exc:
            obs.log_caught(log, "welcome_request_check_failed", exc, call_sid=st.call_sid,
                           decision="fallback_welcome")
            return
        if not last:
            return
        sess.welcome_requested = True
        try:
            queued = self.d.backend.prepare_welcome_async(call_sid=st.call_sid, customer_id=customer_id,
                                                          language=st.language)
        except Exception as exc:
            obs.log_caught(log, "welcome_request_failed", exc, call_sid=st.call_sid,
                           decision="fallback_welcome")
            queued = False
        obs.emit(log, logging.INFO if queued else logging.WARNING, "welcome_requested",
                 call_sid=st.call_sid, factor=factor, queued=queued)

    def _offer_welcome(self, st: CallState, first_name: str | None) -> Spoken | None:
        """The offer welcome (``greeting.offer`` with ``offer.<id>``) when the
        prepared record unseals for this call, is ``matched``, was made for the verified customer
        and the call's language, and renders; else ``None`` (the caller gets ``greeting.personal``
        exactly). The record is cleared by :meth:`_connect` either way. Logs one ``welcome_used``
        line with the ``outcome`` (``made``, ``not_ready``, ``fallback``, ``mismatch``,
        ``no_phrase``), never a text or an offer. Never raises."""
        record = st.welcome_prepared
        if record is None and not self.cfg.welcome_offers:
            return None  # the feature is off: no line
        outcome, reason, spoken = "not_ready", None, None
        try:
            if record is not None:
                prepared = callstate.unseal_welcome(self.d.sealer, record, call_sid=st.call_sid,
                                                    now=self.d.clock())
                if prepared is None:
                    outcome, reason = "fallback", "unusable"
                elif (prepared.customer_id != st.verified_customer_id
                      or prepared.language != st.language):
                    outcome = "mismatch"
                elif prepared.reason != OFFER_MATCHED or prepared.offer_id is None:
                    outcome, reason = "fallback", prepared.reason
                else:
                    text = render_offer(self.d.catalog, st.language, first_name, prepared.offer_id)
                    if text is None:
                        outcome = "no_phrase"
                    else:
                        outcome, spoken = "made", Spoken(OFFER_GREETING_KEY, text)
        except Exception as exc:
            obs.log_caught(log, "welcome_use_failed", exc, call_sid=st.call_sid,
                           decision="fallback_welcome")
            outcome, reason, spoken = "fallback", "error", None
        fields = {"reason": reason} if reason is not None else {}
        obs.emit(log, logging.INFO, "welcome_used", call_sid=st.call_sid, outcome=outcome, **fields)
        return spoken

    # ------------------------------------------------------------------ step-up

    def _stepup_start(self, req: http.TwilioRequest, st: CallState, sess: Session,
                      tool: str | None) -> str:
        policy, lang = self.d.policy, st.language
        tool = tool if isinstance(tool, str) and tool else None
        known = tool is not None and policy.has_tool(tool)
        target = policy.min_level(tool or "")  # unknown tool -> max level (fail closed)
        phrase_key = policy.phrase_key(tool) if known and tool else None
        if not known:
            log.warning("step-up for unknown tool; failing closed", extra={"call_sid": st.call_sid})
        intro = (self._say("stepup.intro", lang, phrase=self.d.catalog.text(phrase_key, lang))
                 if phrase_key else self._say("stepup.intro.generic", lang))
        st = self._update(st, step_up_target=tool)
        slotflow.start_flow(self, sess, "stepup", tool=tool or "unknown", target_level=target)
        sess.stepups += 1
        self._log_event(st.call_sid, {"type": "tool_call", "tool": "authenticate",
                                      "for_tool": tool or "unknown", "target_level": target})
        if sess.stepups > policy.limit("stepup_attempts_per_call"):
            return self._stepup_finish(req, st, sess, "not_verified")
        if st.level >= target:
            return self._stepup_finish(req, st, sess, "verified")
        if st.login_closed:
            # The login failed or was declined; no second try in this call, even
            # if a hand-off got past the bridge's own check. Nothing is asked.
            self._log_event(st.call_sid, {"type": "verification", "outcome": "login_closed",
                                          "level": st.level})
            return self._stepup_finish(req, st, sess, "not_verified")
        if not st.candidate_ids:
            # No candidate record (no match, or ANI withheld): the no-match path, the bank's
            # configured fallback, with no step-up intro (there are no questions to ask).
            if self.cfg.fallbacks.get("no_match", "limited_ai") == "escalate":
                return self._fallback(req, st, sess, "no_match")
            self._log_event(st.call_sid, {"type": "verification", "outcome": "fallback_no_match",
                                          "action": "limited_ai", "level": st.level})
            return self._stepup_finish(req, st, sess, "not_verified")
        missing = self._missing(st, target)
        if not missing:
            return self._stepup_finish(req, st, sess, "not_verified", pre=[intro])
        gate = self._risk_check(req, st, sess, target)
        if gate is not None:
            return gate
        return self._ask(req, st, sess, missing[0], pre=[intro])

    def _stepup_finish(self, req: http.TwilioRequest, st: CallState, sess: Session, status: str,
                       *, utterance: str | None = None,
                       pre: list[tuple[Any, Spoken]] | None = None) -> str:
        flow = sess.flow or {}
        lang = st.language
        for_tool = flow.get("tool") or None
        pre = pre or []
        direct = self._direct_tool(st, for_tool)
        if status == "verified" and direct is not None:
            return self._stepup_then_tool(req, st, sess, direct, pre)
        if status == "verified":
            welcome, nxt = self._say("stepup.success", lang)[1], "retry_original_tool"
        elif status in ("not_verified", "failed"):
            welcome, nxt = self._say("stepup.failed", lang)[1], "offer_alternative"
        elif status == "interrupted":
            welcome, nxt = self._say(self.cfg.bridge_phrase_key, lang)[1], "handle_utterance"
        else:  # declined
            welcome, nxt = None, "offer_alternative"
        spoken = [*self._flow_spoken(flow), *[s for _, s in pre], *([welcome] if welcome else [])]
        # A step-up the bridge started for a front-door tool answers that tool's own call.
        tool = direct.name if direct is not None else "authenticate"
        env = ResultEnvelope(
            status=status, tool=tool, forTool=for_tool if direct is None else None, spoken=spoken,
            data={"level": st.level}, next=nxt, callerUtterance=utterance,
            completed=False if status in ("interrupted", "declined") else None,
        )
        self._log_event(st.call_sid, {"type": "tool_result", "tool": "authenticate",
                                      "outcome": status, "level": st.level})
        return self._connect(req, st, sess, pre, welcome, env)

    def _front_door_tool(self, name: str) -> bool:
        """A tool the front door runs itself: ``get_balance`` or a deterministic confirm."""
        return name == "get_balance" or (name in BANK_TOOLS
                                         and self.d.policy.confirm(name) == "deterministic")

    def _direct_tool(self, st: CallState, for_tool: str | None) -> Any:
        """The pending tool call this step-up is for, when it is a front-door tool (the model
        called it, not ``authenticate``); ``None`` otherwise."""
        pt = st.pending_tool
        if pt is None or pt.name in ("authenticate", "escalate") or pt.name != for_tool:
            return None
        return pt if self._front_door_tool(pt.name) else None

    def _stepup_then_tool(self, req: http.TwilioRequest, st: CallState, sess: Session, pt: Any,
                          pre: list[tuple[Any, Spoken]]) -> str:
        """The step-up is verified; go straight to the pending tool's scripted step (the
        confirmation read-back, or the balance), with no return to the AI in between. What the
        caller heard in the step-up (intro, factor prompts, ``stepup.success``) leads the final
        envelope's ``spoken`` list."""
        flow = sess.flow or {}
        success = self._say("stepup.success", st.language)
        heard = [*self._flow_spoken(flow), *[s for _, s in pre]]
        self._log_event(st.call_sid, {"type": "tool_result", "tool": "authenticate",
                                      "outcome": "verified", "level": st.level, "for_tool": pt.name})
        sess.flow = None
        if pt.name == "get_balance":
            return self._run_tool(req, st, sess, pt.name, pt.args, pre=[*pre, success],
                                  heard=heard)
        return self._confirm_start(req, st, sess, pt, pre=[*pre, success], heard=heard)

    # ------------------------------------------------------------------ deterministic tools

    def _run_tool(self, req: http.TwilioRequest, st: CallState, sess: Session, name: str,
                  args: Mapping[str, Any], *, pre: list[tuple[Any, Spoken]] | None = None,
                  heard: list[Spoken] | None = None) -> str:
        """``pre``: said first, in this response (``stepup.success``); ``heard``: spoken
        earlier in this flow. Both lead the envelope's ``spoken`` list."""
        lang = st.language
        pre = pre or []
        before = [*(heard or []), *[s for _, s in pre]]
        self._log_event(st.call_sid, {"type": "tool_call", "tool": name})
        # The bridge's prefetch is good for this one pass at most; clear it
        # whatever happens next (used, ignored, step-up or failure).
        prefetched = st.prefetched
        if prefetched is not None:
            st = self._update(st, **{callstate.F_PREFETCHED: None})
        if name != "get_balance" or validate_args(name, dict(args)):
            return self._tool_failed(req, st, sess, name, "error.generic", pre=pre, heard=heard)
        token: str | None = None
        if st.sealed_token:
            try:
                token = self.d.sealer.unseal(st.sealed_token, call_sid=st.call_sid)
            except UnsealError as exc:
                obs.log_caught(log, "call token could not be unsealed", exc, operation="unseal_token",
                               tool=name, decision="step_up")
                token = None
        claims = tokens.validate(self.d.store, token, call_sid=st.call_sid,
                                 min_level=self.d.policy.min_level(name), now=self._now())
        if isinstance(claims, tokens.AuthError) or token is None:
            if heard is not None:  # Right after a verified step-up; never loop back into it
                return self._tool_failed(req, st, sess, name, "error.generic", pre=pre, heard=heard)
            return self._stepup_start(req, st, sess, name)
        account = str(args["account"])
        pt = st.pending_tool
        res, reason, age = prefetched_balance(
            prefetched, tool=name, tool_use_id=pt.tool_use_id if pt is not None else None,
            args=args, now=self._now())
        if res is not None:
            obs.emit(log, logging.INFO, "prefetch_used", call_sid=st.call_sid, tool=name, age_s=age)
        else:
            obs.emit(log, logging.INFO, "prefetch_ignored", call_sid=st.call_sid, tool=name,
                     reason=reason)
            res = self.slots.api_call(sess, lambda: self.d.backend.get_balance(
                token=token, call_sid=st.call_sid, account=account))
        account_label = self.d.catalog.text(f"account.{account}", lang)
        if res.status == "auth_insufficient":
            if heard is not None:
                return self._tool_failed(req, st, sess, name, "error.generic", pre=pre, heard=heard)
            return self._stepup_start(req, st, sess, name)
        follow = self._say("common.anything_else", lang)
        if res.status == "not_found":
            items = [self._say("balance.no_account", lang, accountType=account_label)]
            env = ResultEnvelope(status="not_found", tool=name,
                                 spoken=[*before, *[s for _, s in items], follow[1]],
                                 data={"account": account}, next="await_caller")
            self._log_event(st.call_sid, {"type": "tool_result", "tool": name, "outcome": "not_found"})
            return self._connect(req, st, sess, [*pre, *items], follow[1], env)
        if res.status != "ok":
            return self._tool_failed(req, st, sess, name, "error.lookup_unavailable", pre=pre,
                                     heard=heard)
        try:
            items, data = self._balance_speech(account, account_label, res.data, lang)
        except (KeyError, ValueError, TypeError, ArithmeticError) as exc:  # Decimal raises InvalidOperation
            obs.log_caught(log, "balance result could not be spoken", exc, tool=name,
                           operation="balance_speech", decision="tool_failed_apology")
            return self._tool_failed(req, st, sess, name, "error.lookup_unavailable", pre=pre,
                                     heard=heard)
        env = ResultEnvelope(status="completed", tool=name,
                             spoken=[*before, *[s for _, s in items], follow[1]], data=data,
                             next="await_caller")
        self._log_event(st.call_sid, {"type": "tool_result", "tool": name, "outcome": "completed"})
        return self._connect(req, st, sess, [*pre, *items], follow[1], env)

    def _balance_speech(self, account: str, label: str, raw: Mapping[str, Any],
                        lang: str) -> tuple[list[tuple[Any, Spoken]], dict[str, Any]]:
        from decimal import Decimal
        data = {"account": account}
        for k in ("last4", "available", "current", "pending", "currency", "asOf"):
            if isinstance(raw.get(k), str):
                data[k] = raw[k]
        last4 = digits_words(data["last4"], lang)
        available = Decimal(data["available"])
        current = Decimal(data.get("current", data["available"]))
        pending = Decimal(data.get("pending", "0"))
        if available < 0 or current < 0:
            over = -min(available, current)
            items = [self._say("balance.overdrawn", lang, accountType=label, last4=last4,
                               overdrawn=money_words(over, lang))]
        else:
            items = [self._say("balance.read", lang, accountType=label, last4=last4,
                               available=money_words(available, lang),
                               current=money_words(current, lang))]
        if pending != 0:
            items.append(self._say("balance.read.pending", lang,
                                   pending=money_words(abs(pending), lang)))
        return items, data

    def _tool_failed(self, req: http.TwilioRequest, st: CallState, sess: Session, name: str,
                     key: str, *, pre: list[tuple[Any, Spoken]] | None = None,
                     heard: list[Spoken] | None = None) -> str:
        items = [*(pre or []), self._say(key, st.language)]
        env = ResultEnvelope(status="failed", tool=name,
                             spoken=[*(heard or []), *[s for _, s in items]], data={},
                             next="offer_alternative")
        self._log_event(st.call_sid, {"type": "tool_result", "tool": name, "outcome": "failed"})
        return self._connect(req, st, sess, items, None, env)

    # ------------------------------------------------------------------ confirmation

    def _token(self, st: CallState, tool: str) -> str | None:
        """The call token, unsealed in memory, if it is valid for ``tool``'s level."""
        token: str | None = None
        if st.sealed_token:
            try:
                token = self.d.sealer.unseal(st.sealed_token, call_sid=st.call_sid)
            except UnsealError as exc:
                obs.log_caught(log, "call token could not be unsealed", exc, operation="unseal_token",
                               tool=tool, decision="treated_as_no_token")
                token = None
        claims = tokens.validate(self.d.store, token, call_sid=st.call_sid,
                                 min_level=self.d.policy.min_level(tool), now=self._now())
        return None if isinstance(claims, tokens.AuthError) else token

    def _confirm_values(self, tool: str, args: Mapping[str, Any], lang: str) -> dict[str, str]:
        """Read-back placeholders: validated arguments only, formatted server-side."""
        def label(account: Any) -> str:
            return self.d.catalog.text(f"account.{account}", lang)
        values = {"amount": money_words(str(args["amount"]), lang),
                  "fromAccount": label(args["from_account"])}
        to = args.get("to_account")
        if to is not None:
            values["toAccount"] = label(to)
        # ``make_payment.to`` is a card option, not an account: its catalog spoken name.
        card = args.get("to") if tool == "make_payment" else None
        key = f"option.{CARD_SLOT}.{card}"
        if isinstance(card, str) and card in card_values(self.d.policy) and key in self.d.catalog.keys:
            values["toCard"] = self.d.catalog.text(key, lang)
        return values

    def _caller_cards(self, st: CallState) -> tuple[str, ...]:
        """The verified customer's cards: the ``card`` slot's known options whose
        product the customer's record holds, in policy order. ``()`` without a verified customer
        or a readable record (fail closed: nothing can be paid)."""
        cid = st.verified_customer_id
        cust = self.d.store.get("customers", {"customer_id": cid}) if cid else None
        accounts = cust.get("accounts") if isinstance(cust, Mapping) else None
        held = {a.get("product") for a in accounts if isinstance(a, Mapping)
                and a.get("type") == "credit_card"} if isinstance(accounts, list) else set()
        return tuple(v for v in card_values(self.d.policy) if v in held)

    def _set_card(self, st: CallState, tool: str, card: str) -> CallState:
        """``pending_tool.args.to = card`` in call state (``callstate.mutate``). The
        caller (``SlotFlow``) checked that ``card`` is one of the caller's cards."""
        def fn(s: CallState) -> Mapping[str, Any]:
            pt = s.pending_tool
            if pt is None or pt.name != tool:
                raise _Abort("card set without its pending tool")
            return {"pending_tool": replace(pt, args={**pt.args, "to": card})}
        with self._span("state_save"):
            return callstate.mutate(self.d.store, st.call_sid, fn, now=self.d.clock)

    def _confirm_start(self, req: http.TwilioRequest, st: CallState, sess: Session,
                       pt: Any, *, pre: list[tuple[Any, Spoken]] | None = None,
                       heard: list[Spoken] | None = None) -> str:
        """Read the model's tool call back and ask yes or no. The tool hasn't run.

        ``make_payment``: a ``to`` that is one of the caller's cards is read back by
        name. A missing or unknown ``to`` with exactly one card on file uses that card with no
        question, and the read-back names it; with two or more, ``prompt.card`` lists them. A
        known card the caller doesn't hold asks the not-available question, even with one
        card on file: it is never mapped to another card. A customer with no card on file
        can't pay: the generic apology.

        ``pre``: said before the read-back in this response (``stepup.success``);
        ``heard``: spoken earlier (the step-up). Both lead the envelope's ``spoken`` list."""
        name = pt.name
        args = dict(pt.args)
        self._log_event(st.call_sid, {"type": "tool_call", "tool": name, "confirm": "deterministic"})
        # ``to`` is checked against the caller's cards below, not by the schema (an unknown card
        # is asked, not refused).
        checked = {k: v for k, v in args.items() if k != "to"} if name == "make_payment" else args
        if name not in BANK_TOOLS or validate_args(name, checked, reasons=self.d.policy.reasons):
            return self._tool_failed(req, st, sess, name, "error.generic", pre=pre, heard=heard)
        if self._token(st, name) is None:
            if heard is not None:  # Right after a verified step-up; never loop back into it
                return self._tool_failed(req, st, sess, name, "error.generic", pre=pre, heard=heard)
            return self._stepup_start(req, st, sess, name)
        cards = self._caller_cards(st) if name == "make_payment" else ()
        if name == "make_payment" and not cards:
            return self._tool_failed(req, st, sess, name, "error.generic", pre=pre, heard=heard)
        slotflow.start_flow(self, sess, "confirm", tool=name, cards=list(cards),
                            spoken=[{"template": s.template, "text": s.text} for s in heard or []])
        if name == "make_payment":
            to = args.get("to")
            if to not in cards:
                if isinstance(to, str) and to in card_values(self.d.policy):
                    return self.slots.ask_not_available(req, st, sess, to, pre=list(pre or []))
                if len(cards) != 1:
                    return self.slots.ask_card(req, st, sess, pre=list(pre or []))
                st = self._set_card(st, name, cards[0])     # The one card, no question
            slotflow.collect(sess, CARD_SLOT, "choice", verified=False)   # correctable
        return self._confirm_readback(req, st, sess, name, pre=list(pre or []))

    def _confirm_readback(self, req: http.TwilioRequest, st: CallState, sess: Session, name: str,
                          *, pre: list[tuple[Any, Spoken]] | None = None) -> str:
        """The read-back of the pending tool call as call state holds it now, asked as yes or no
        (``slot#yes_no``). ``pre`` is spoken first."""
        telemetry.set_step("confirm.readback")
        flow = sess.flow or {}
        flow.update(factor="yes_no", pending=None)
        sess.flow = flow
        self._log_event(st.call_sid, {"type": "confirm_asked", "tool": name})
        return self._confirm_ask(req, st, sess, [*(pre or []), self._readback(st, name)])

    def _confirm_ask(self, req: http.TwilioRequest, st: CallState, sess: Session,
                     items: list[tuple[Any, Spoken]]) -> str:
        flow = sess.flow or {}
        flow["spoken"] = [*flow.get("spoken", []), *[{"template": s.template, "text": s.text}
                                                      for _, s in items]]
        sess.flow = flow
        self._save(sess)
        self._prompts(st.call_sid, items)
        slot = self._slot("yes_no")  # The confirmation's settings are ``slot#yes_no``
        return self._render(*self._gather(req, st.language, slot, [v for v, _ in items]))

    def _readback(self, st: CallState, name: str) -> tuple[Any, Spoken]:
        pt = st.pending_tool
        args = dict(pt.args) if pt is not None else {}
        return self._say(f"confirm.{name}", st.language,
                         **self._confirm_values(name, args, st.language))

    def _confirm_answer(self, req: http.TwilioRequest, st: CallState, sess: Session) -> str:
        flow = sess.flow or {}
        name = str(flow.get("tool") or "")
        pt = st.pending_tool
        if pt is None or pt.name != name:
            sess.flow = None
            self._save(sess)
            raise _Abort("confirmation without its pending tool")
        if flow.get("factor") == CARD_SLOT:
            return self.slots.card(req, st, sess, name)
        return self.slots.confirm(req, st, sess, name)

    def _confirm_interrupted(self, req: http.TwilioRequest, st: CallState, sess: Session, name: str,
                             utterance: str) -> str:
        """escalate / intent_switch with words during the read-back: the AI handles the
        utterance (the answering turn's text). The tool doesn't run."""
        flow = sess.flow or {}
        lang = st.language
        self._log_event(st.call_sid, {"type": "confirm_answer", "tool": name,
                                      "answer": "interrupted"})
        welcome = self._say(self.cfg.bridge_phrase_key, lang)[1]
        env = ResultEnvelope(status="interrupted", tool=name,
                             spoken=[*self._flow_spoken(flow), welcome],
                             data={"executed": False}, next="handle_utterance",
                             callerUtterance=utterance, completed=False)
        self._log_event(st.call_sid, {"type": "tool_result", "tool": name, "outcome": "interrupted"})
        return self._connect(req, st, sess, [], welcome, env)

    def _confirm_cancel(self, req: http.TwilioRequest, st: CallState, sess: Session, name: str,
                        answer: str) -> str:
        """No, no answer, retries exhausted or cancel: the tool does not run."""
        flow = sess.flow or {}
        lang = st.language
        self._log_event(st.call_sid, {"type": "confirm_answer", "tool": name, "answer": answer})
        items = [self._say("confirm.cancelled", lang)]
        follow = self._say("common.anything_else", lang)
        env = ResultEnvelope(status=CANCELLED_BY_CALLER, tool=name,
                             spoken=[*self._flow_spoken(flow), *[s for _, s in items], follow[1]],
                             data={"executed": False, "answer": answer}, next="await_caller",
                             completed=False)
        self._log_event(st.call_sid, {"type": "tool_result", "tool": name,
                                      "outcome": CANCELLED_BY_CALLER})
        return self._connect(req, st, sess, items, follow[1], env, protect_outcome=True)

    def _confirm_execute(self, req: http.TwilioRequest, st: CallState, sess: Session,
                         name: str) -> str:
        """The caller said yes: run the tool once (idempotency key = ``tool_use_id``)."""
        flow = sess.flow or {}
        lang = st.language
        pt = st.pending_tool
        if pt is None:
            raise RuntimeError("confirm_execute without a pending tool")
        self._log_event(st.call_sid, {"type": "confirm_answer", "tool": name, "answer": "yes"})
        sess.flow = None  # the answer is consumed: a replayed webhook can't run it again
        self._save(sess)
        args = dict(pt.args)
        if validate_args(name, args, reasons=self.d.policy.reasons,
                         cards=card_values(self.d.policy)):
            return self._tool_failed(req, st, sess, name, "error.generic")
        if name == "make_payment" and args.get("to") not in self._caller_cards(st):
            # Never a card the caller doesn't hold (the flow only reads back theirs).
            return self._tool_failed(req, st, sess, name, "error.generic")
        token = self._token(st, name)
        if token is None:
            return self._stepup_start(req, st, sess, name)
        res = None
        for _ in range(2):  # one retry, safe only because the key is the same
            res = self.slots.api_call(sess, functools.partial(
                self.d.backend.run_tool, token=token, call_sid=st.call_sid, tool=name, args=args,
                idempotency_key=pt.tool_use_id))
            if not (res.status == "error" and res.code is None):
                break
        del token
        if res is None:
            raise RuntimeError("tool run produced no result")
        self._log_event(st.call_sid, {"type": "tool_result", "tool": name,
                                      "outcome": res.status, "error": res.code})
        if res.status == "auth_insufficient":
            return self._stepup_start(req, st, sess, name)
        spoken_before = self._flow_spoken(flow)
        follow = self._say("common.anything_else", lang)
        if res.status != "ok":
            key = "confirm.insufficient_funds" if res.code == "INSUFFICIENT_FUNDS" else "confirm.failed"
            items = [self._say(key, lang)]
            env = ResultEnvelope(status="failed", tool=name,
                                 spoken=[*spoken_before, *[s for _, s in items], follow[1]],
                                 data={"executed": False, "error": res.code or "UNAVAILABLE"},
                                 next="await_caller")
            return self._connect(req, st, sess, items, follow[1], env, protect_outcome=True)
        data = json.loads(json.dumps(dict(res.data), default=str))
        conf = data.get("confirmation")
        values = self._confirm_values(name, args, lang)
        values["confirmation"] = (digits_words(conf, lang) if isinstance(conf, str)
                                  and conf.isascii() and conf.isdigit() else str(conf or ""))
        items = [self._say(f"confirm.{name}.done", lang, **values)]
        env = ResultEnvelope(status="completed", tool=name,
                             spoken=[*spoken_before, *[s for _, s in items], follow[1]],
                             data={"executed": True, **data}, next="await_caller")
        return self._connect(req, st, sess, items, follow[1], env, protect_outcome=True)

    # ------------------------------------------------------------------ /voice/connect-action

    def connect_action(self, req: http.TwilioRequest) -> str:
        p = req.params
        if p.get("CallStatus") == "completed" or p.get("SessionStatus") == "completed":
            return twiml.empty()  # caller hung up
        handoff = _parse_handoff(p.get("HandoffData"))
        ended = p.get("SessionStatus") == "ended"
        hangup = _hangup_reason(handoff) if ended and handoff is not None else None
        if handoff is not None and hangup is not None and hangup in _HANGUP_PHRASES:
            # Before the load, so no store failure takes the apology path (it escalates)
            return self._hangup_ended(req.call_sid, handoff, hangup)
        st, sess = self._load(req.call_sid)
        conv = _safe_id(handoff.get("conversationId")) if handoff else None
        if conv and conv != sess.conversation_id:
            sess.conversation_id = conv
            self._save(sess)
        if ended and handoff is not None:
            if hangup == NO_INPUT:
                # The bridge's silence watch hands the call to a human through the usual
                # escalation path, whatever tool call may still be pending.
                ref = (st.escalation or {}).get("journal_ref") or handoff.get("journal")
                return self._escalate(
                    req, st, sess, NO_INPUT, (st.escalation or {}).get("summary") or "",
                    journal_ref=ref if isinstance(ref, str) else None,
                    conversation_id=_safe_id(handoff.get("conversationId")),
                    profile_id=_safe_id(handoff.get("profileId")),
                )
            keypad_reason = _keypad_reason(handoff, self.d.policy.reasons)
            if keypad_reason is not None:
                # The caller pressed 0; the bridge ended the session with a fixed
                # summary in call state. Like the silence escalation, a pending tool can't win.
                ref = (st.escalation or {}).get("journal_ref") or handoff.get("journal")
                return self._escalate(
                    req, st, sess, keypad_reason, (st.escalation or {}).get("summary") or "",
                    journal_ref=ref if isinstance(ref, str) else None,
                    conversation_id=_safe_id(handoff.get("conversationId")),
                    profile_id=_safe_id(handoff.get("profileId")),
                )
            kind, name, args = self._classify_handoff(st, handoff)
            telemetry.set_step(f"handoff.{kind or 'none'}")
            if kind == "stepup":
                return self._stepup_start(req, st, sess, args.get("tool"))
            if kind == "tool" and name:
                return self._run_tool(req, st, sess, name, args)
            if kind == "confirm" and st.pending_tool is not None:
                return self._confirm_start(req, st, sess, st.pending_tool)
            if kind == "escalate":
                # The bridge's stored ref wins; HandoffData's is the same value (Twilio-signed).
                ref = (st.escalation or {}).get("journal_ref") or handoff.get("journal") \
                    or handoff.get("journal_ref")
                # The bridge keeps the agent's (cleaned) summary in call state, never in HandoffData.
                summary = args.get("summary") or (st.escalation or {}).get("summary") or ""
                return self._escalate(
                    req, st, sess, args.get("reason"), summary,
                    journal_ref=ref if isinstance(ref, str) else None,
                    conversation_id=_safe_id(handoff.get("conversationId")),
                    profile_id=_safe_id(handoff.get("profileId")),
                )
        if self.classify_close(p, st, self.d.clock()) == "stream_limit":
            if sess.stream_reconnects < MAX_STREAM_RECONNECTS:
                return self._stream_reconnect(req, st, sess)
            obs.emit(log, logging.WARNING, "stream_reconnect_cap_reached", call_sid=st.call_sid,
                     stream_reconnects=sess.stream_reconnects, decision="drop_path")
        # WebSocket dropped, session error, or an unusable handoff: reconnect once, then escalate.
        sess.relay_failures += 1
        self._log_event(st.call_sid, {"type": "handoff", "to": "frontdoor",
                                      "outcome": f"relay_{p.get('SessionStatus') or 'unknown'}",
                                      "error_code": (p.get("ErrorCode") or "")[:16]})
        if sess.relay_failures <= self.cfg.max_relay_reconnects:
            self._save(sess)
            apology = self._say("error.generic", st.language)
            welcome = self._say("common.anything_else", st.language)[1]
            self._prompts(st.call_sid, [apology, (None, welcome)])
            return self._render(apology[0], self._relay(req, st, st.language, welcome.text,
                                                        conversation_id=sess.conversation_id))
        return self._escalate(req, st, sess, "unable_to_resolve", "")

    def classify_close(self, params: Mapping[str, str], st: CallState, now: float) -> CloseKind:
        """``stream_limit`` iff ``agentcore``, the session ``failed`` with ErrorCode
        64105 and no ``HandoffData``, and at least 3,540 s have passed since the URL was minted
        (``relay_minted_at``). ConversationRelay passes no close code, so the duration is the
        only sign of the 60-minute cap. Everything else, and every close on ``fargate``, is
        ``other`` (today's drop path)."""
        if self.cfg.hosting != "agentcore" or st.relay_minted_at is None:
            return "other"
        if params.get("SessionStatus") != "failed" or (params.get("HandoffData") or "").strip():
            return "other"
        if (params.get("ErrorCode") or "").strip() != STREAM_LIMIT_ERROR_CODE:
            return "other"
        return "stream_limit" if now - st.relay_minted_at >= STREAM_LIMIT_MIN_AGE_S else "other"

    def _stream_reconnect(self, req: http.TwilioRequest, st: CallState, sess: Session) -> str:
        """The 60-minute cap closed a healthy session. Reconnect to the same Runtime session
        (CallSid) with a fresh URL and nonce, an empty greeting and no apology; not a relay
        failure. The count is saved first, so a reconnect that can't be signed still counts (and
        takes the apology-then-human path). No URL or nonce is logged."""
        sess.stream_reconnects += 1
        self._save(sess)
        self._log_event(st.call_sid, {"type": "handoff", "to": "frontdoor",
                                      "outcome": "relay_stream_limit"})
        cr = self._relay(req, st, st.language, "", conversation_id=sess.conversation_id)
        obs.emit(log, logging.INFO, "relay_reconnect", call_sid=st.call_sid, reason="stream_limit",
                 stream_reconnects=sess.stream_reconnects, hosting=self.cfg.hosting)
        return self._render(cr)

    def _hangup_ended(self, call_sid: str, handoff: Mapping[str, Any], reason: str) -> str:
        """:meth:`_hangup` for a session that ended with a hang-up ``reason``: reads the call state
        and the session each on its own, so a failed read only drops what it couldn't read.
        The new conversation id is saved with the session in ``_hangup``. A session that couldn't
        be read is never saved (a fresh one would overwrite it); a fresh one is made only when the
        read worked, found none, and call state gives its expiry. No CallSid: just the goodbye."""
        st: CallState | None = None
        sess: Session | None = None
        if call_sid:
            try:
                st = self._state(call_sid)
                if st is None:
                    obs.emit(log, logging.ERROR, "hangup_without_call_state", call_sid=call_sid,
                             decision="hangup_anyway")
            except Exception as exc:
                obs.log_caught(log, "hangup without call state", exc, level=logging.ERROR,
                               operation="load_call_state", call_sid=call_sid, decision="hangup_anyway")
            try:
                with self._span("state_load"):
                    sess = sessions.load(self.d.store, call_sid)
                if sess is None and st is not None:
                    sess = Session(call_sid, st.expires_at)
            except Exception as exc:
                obs.log_caught(log, "hangup without session", exc, level=logging.ERROR,
                               operation="session_load", call_sid=call_sid, decision="hangup_anyway")
        conv = _safe_id(handoff.get("conversationId"))
        if conv and sess is not None:
            sess.conversation_id = conv
        return self._hangup(call_sid, st, sess, reason)

    def _hangup(self, sid: str, st: CallState | None, sess: Session | None, reason: str) -> str:
        """The AI session ended with a hang-up ``reason`` (:data:`_HANGUP_PHRASES`):
        ``silence_action = hangup`` (``no_input_hangup``) or the model's ``end_call``
        (``caller_done``). The reason's catalog goodbye in the call's language, then
        ``<Hangup/>``. The AI session is over, so the call's token is revoked as at call end.

        Failures never escalate and never leave the caller in silence: a failed
        state clear or revocation is logged and the call still hangs up (the token expires on its
        TTL); a missing goodbye says ``error.generic`` instead, and with that missing too the
        response is a bare ``<Hangup/>``. ``st`` / ``sess`` are ``None`` when they couldn't be read
        (:meth:`_hangup_ended`): the goodbye is then in the default language, and the rest still
        runs on the call SID."""
        lang = st.language if st is not None and st.language else self.d.catalog.default_language
        level = st.level if st is not None else None
        if st is not None:
            try:
                self._update(st, sealed_token=None, last_spoken=None,
                             **{callstate.F_WELCOME_OFFER: None, callstate.F_WELCOME_PREPARED: None})
            except Exception as exc:
                obs.log_caught(log, "hangup state clear failed", exc, level=logging.ERROR,
                               operation="state_save", call_sid=sid, decision="hangup_anyway")
        if sid:
            try:
                tokens.revoke_call(self.d.store, sid)
            except Exception as exc:
                obs.log_caught(log, "hangup token revoke failed", exc, level=logging.ERROR,
                               operation="revoke_token", call_sid=sid, decision="hangup_anyway")
        if sess is not None:
            try:
                sess.flow = None
                self._save(sess)
            except Exception as exc:
                obs.log_caught(log, "hangup session save failed", exc, operation="session_save",
                               call_sid=sid, decision="hangup_anyway")
        verbs: list[Any] = []
        for key in (_HANGUP_PHRASES[reason], "error.generic"):
            try:
                said = self._say(key, lang)
            except Exception as exc:
                obs.log_caught(log, "hangup text unavailable", exc, operation="catalog_text",
                               call_sid=sid, prompt_key=key, decision="next_text_or_bare_hangup")
                continue
            verbs = [said[0]]
            try:
                if sid:
                    self._prompts(sid, [said])
            except Exception as exc:
                obs.log_caught(log, "hangup prompt not journaled", exc, operation="journal",
                               call_sid=sid, decision="hangup_anyway")
            break
        if not sid:  # no call to journal or count: just the goodbye
            return self._render(*verbs, twiml.Hangup())
        try:
            self._log_event(sid, {"type": "handoff", "to": "frontdoor", "outcome": reason})
        except Exception as exc:
            obs.log_caught(log, "hangup not journaled", exc, operation="journal", call_sid=sid,
                           decision="hangup_anyway")
        obs.emit(log, logging.INFO, "caller_done_hangup" if reason == CALLER_DONE
                 else "no_input_hangup", call_sid=sid, source="frontdoor",
                 **({"level_reached": level} if level is not None else {}))
        return self._render(*verbs, twiml.Hangup())

    def _classify_handoff(self, st: CallState, handoff: Mapping[str, Any]
                          ) -> tuple[str | None, str | None, dict[str, Any]]:
        """``(kind, tool, args)``. The pending tool in call state wins over ``HandoffData``."""
        reason = str(handoff.get("reason", "")).lower().replace("-", "_")
        pt = st.pending_tool
        if pt is not None:
            if reason in ("step_up", "stepup", "authenticate") and pt.name not in (
                    "authenticate", "escalate") and self._front_door_tool(pt.name):
                # The bridge started the step-up for the model's own call (it was refused
                # only for the caller's level); the pending tool runs after a verified step-up.
                return "stepup", pt.name, {"tool": pt.name}
            if pt.name == "authenticate":
                kind: str | None = "stepup"
            elif pt.name == "escalate":
                kind = "escalate"
            elif self.d.policy.kind(pt.name) == "deterministic":
                kind = "tool"
            elif pt.name in BANK_TOOLS and self.d.policy.confirm(pt.name) == "deterministic":
                kind = "confirm"  # Only from call state, never from HandoffData
            else:
                kind = None
            expected = {"stepup": {"step_up", "stepup", "authenticate"}, "tool": {"tool"},
                        "escalate": {"escalate"}, "confirm": {"confirm"}}.get(kind or "", set())
            if reason not in expected:
                log.warning("handoff reason doesn't match the pending tool",
                            extra={"call_sid": st.call_sid})
            return kind, pt.name, dict(pt.args)
        tool = handoff.get("tool") if isinstance(handoff.get("tool"), str) else None
        if reason in ("step_up", "stepup", "authenticate"):
            return "stepup", "authenticate", {"tool": tool}
        if reason == "tool":
            raw_args = handoff.get("args")
            return "tool", tool, dict(raw_args) if isinstance(raw_args, dict) else {}
        if reason == "escalate":
            return "escalate", "escalate", {"reason": handoff.get("escalationReason"),
                                            "summary": ""}
        return None, None, {}

    # ------------------------------------------------------------------ /voice/status

    def status(self, req: http.TwilioRequest) -> str:
        telemetry.set_step("call.end")
        sid = req.call_sid
        call_status = req.params.get("CallStatus", "")
        duration = req.params.get("CallDuration", "")
        # Twilio's status and duration; never a phone number.
        telemetry.add_fields(call_status=call_status or None,
                             call_duration_s=int(duration) if duration.isdigit() else None)
        if not sid or call_status not in END_STATUSES:
            return twiml.empty()
        tokens.revoke_call(self.d.store, sid)
        st = self._state(sid)
        sess = sessions.load(self.d.store, sid)
        if telemetry.installed():
            telemetry.add_fields(last_step=last_step(st, sess))
        first_end = sess is None or not sess.ended  # this request is the call's first end status
        if sess is None and st is not None:
            sess = Session(sid, st.expires_at)
        if st is not None and (st.sealed_token or st.last_spoken or st.welcome_offer
                               or st.welcome_prepared is not None):
            st = self._update(st, sealed_token=None, last_spoken=None,
                              **{callstate.F_WELCOME_OFFER: None, callstate.F_WELCOME_PREPARED: None})
        if sess is not None and not sess.ended:
            sess.ended, sess.ended_at = True, self._now()
            sess.flow = None
            self._save(sess)
            self._log_event(sid, {"type": "call_end", "outcome": call_status,
                                  "duration_s": int(duration) if duration.isdigit() else None})
        if call_status in ("completed", "failed"):
            try:
                self.d.closer.close(call_sid=sid,
                                    conversation_id=sess.conversation_id if sess else None)
            except Exception as exc:  # never breaks the status response
                obs.log_caught(log, "conversation close failed", exc, call_sid=sid,
                               decision="ignored_because_status_must_succeed")
            self._recap_at_status(st, sess, first_end)
        sensitive.forget(sid)
        self._stop_runtime_session(st)
        if self.cfg.hosting == "agentcore":
            # Refresh the signing credentials when under 15 min is left, here where nobody
            # waits, so AssumeRole rarely falls on a connect. Never raises.
            refresh = getattr(self.relay_urls, "refresh_if_due", None)
            if callable(refresh):
                refresh()
        return twiml.empty()

    def _recap_at_status(self, st: CallState | None, sess: Session | None, first_end: bool) -> None:
        """Start the recap writer when no human-leg close will (:func:`recap_trigger.at_status`,
        on the state read before the close), once per call (the first end status: a repeat finds the
        session ended). Never raises; a failed trigger is logged and the response is unchanged."""
        if self.d.recap is None or st is None or not first_end:
            return
        try:
            start = recap_trigger.at_status(
                st.escalation, ai_sessions=st.ai_sessions,
                session_conversation=sess.conversation_id if sess else None,
                closer_wired=not isinstance(self.d.closer, NoopConversationCloser))
        except Exception as exc:  # pure, but the status response must go out whatever happens
            obs.log_caught(log, "recap_trigger_failed", exc, call_sid=st.call_sid, source="status",
                           decision="ignored_no_recap")
            return
        if start:
            recap_trigger.fire(self.d.recap, st.call_sid, source="status")

    # ------------------------------------------------------------------ /voice/transcription

    def transcription(self, req: http.TwilioRequest, event: str) -> None:
        """One leg's transcription ended: ``event`` is
        ``transcription_stopped`` or ``transcription_error``. ``CallSid`` is the leg; the signed
        ``call`` query parameter names the customer call whose state records it (an agent leg's
        status URL carries it; absent, the leg is the customer call itself). The closer ends
        that leg only, and closes once the call has ended and no leg is running; a ``call`` whose
        state does not record the leg changes nothing (``unknown_leg``), and nothing is recorded after
        the close. Never raises (the route always answers 204); no request parameter is logged."""
        leg = req.call_sid
        if not leg or not CALL_SID_RE.match(leg):
            obs.emit(log, logging.INFO, "transcription_event_dropped", reason="no_call_sid",
                     decision="ignored")
            return
        call = transcription_call_sid(req)
        if call is None:
            obs.emit(log, logging.INFO, "transcription_event_dropped", reason="bad_call_param",
                     decision="ignored", call_sid=leg)
            return
        try:
            sess = sessions.load(self.d.store, call)
            self.d.closer.transcription_ended(
                call_sid=call, conversation_id=sess.conversation_id if sess else None, event=event, leg=leg)
        except Exception as exc:
            obs.log_caught(log, "conversation close failed", exc, call_sid=call,
                           decision="ignored_because_status_must_succeed")

    # ------------------------------------------------------------------ /voice/taskrouter

    def taskrouter(self, req: http.TwilioRequest) -> None:
        """A TaskRouter workspace event: the queue telemetry (:meth:`_taskrouter_telemetry`), then the
        human leg's participant on ``reservation.accepted`` (:meth:`_human_leg_accept`) and the
        agent legs' transcriptions on ``task.updated`` (:meth:`_human_leg_task_updated`). Each is
        independent of the other's switch. Never raises (the route always answers 204)."""
        self._taskrouter_telemetry(req)
        self._human_leg_accept(req)
        self._human_leg_task_updated(req)

    def _taskrouter_telemetry(self, req: http.TwilioRequest) -> None:
        """Call observability (opt-in through ``twilio_setup.py --taskrouter-events``): for an
        accepted reservation, a cancelled or a timed-out task, the ``handoff.queue`` span of the task's
        call (``call_sid`` from the task attributes) with the queue, the wait (``TaskAge``, whole
        seconds since the task was created by ``<Enqueue>``) and the outcome. Never the worker's or the
        caller's identity. Other events, or telemetry off: nothing. Never raises."""
        try:
            outcome = taskrouter_outcome(req.params)
            if outcome is None or not telemetry.installed():
                return
            if taskrouter_call_sid(req.params) is None:
                obs.emit(log, logging.INFO, "taskrouter_event_dropped", reason="no_call_sid",
                         event_type=req.params.get("EventType"))
                return
            age = req.params.get("TaskAge", "")
            telemetry.set_step("handoff.queue")
            telemetry.add_fields(
                queue=req.params.get("TaskQueueName") or req.params.get("TaskQueueSid") or None,
                wait_ms=int(age) * 1000 if age.isdigit() else None, outcome=outcome)
        except Exception as exc:
            obs.log_caught(log, "taskrouter_event_failed", exc, decision="ignored")

    # ------------------------------------------------------------------ human leg: accepts

    def _human_leg_accept(self, req: http.TwilioRequest) -> None:
        """``reservation.accepted`` for an escalated call, with
        the human-leg client wired (``humanLegConversation=on``). Synchronously, all Twilio requests
        within :data:`HUMAN_LEG_ACCEPT_BUDGET_S` in total (no background work: the Lambda freezes after
        the 204): on the call's first accept, the hand-off metadata (``mark_handoff``); the worker as a
        ``HUMAN_AGENT`` participant (once per worker, 409 = present); with the
        ``human_leg.transcription`` switch on, the transcriptions not yet recorded (:meth:`_start_legs`:
        the customer leg, and any agent legs the task attributes already hold;
        ``/voice/transcription`` is their status callback, with the Basic credentials). Idempotent by
        ``EventSid`` and worker (claims in call state, :func:`owlbank.humanleg.claim_accept`); a later
        accept (an onward transfer) adds its worker and starts only what is not recorded yet. Ignored for a call that never escalated, and after the close
        (``human_leg_after_close``, nothing written). Never raises; never logs a worker's identity."""
        client = self.d.human_leg
        if client is None or req.params.get("EventType") != humanleg.ACCEPTED_EVENT:
            return
        call_sid = taskrouter_call_sid(req.params)
        if call_sid is None:
            obs.emit(log, logging.INFO, "human_leg_accept_dropped", reason="no_call_sid", decision="ignored")
            return
        started = time.monotonic()
        close_after: list[str] = []
        outcome: dict[str, Any] | None = None
        try:
            with human_leg_budget(HUMAN_LEG_ACCEPT_BUDGET_S):
                outcome = self._accept(req, call_sid, client, close_after)
        except Exception as exc:
            obs.log_caught(log, "human_leg_accept_failed", exc, call_sid=call_sid, operation="accept",
                           decision="ignored_best_effort")
        if outcome is not None:
            obs.emit(log, logging.INFO, "human_leg_accept", call_sid=call_sid,
                     elapsed_ms=int((time.monotonic() - started) * 1000), **outcome)
        if close_after:
            self._close_after_failed_start(call_sid, close_after[0])

    def _close_after_failed_start(self, call_sid: str, conversation_id: str) -> None:
        """The call ended while the transcription start was in flight, and the start failed: the
        close that waited for this transcript runs now, after the accept's budget, so it gets
        its own full deadline (``CLOSE_DEADLINE_S``), not what is left of the accept's 2 s."""
        closer = self.d.closer
        own = getattr(closer, "deadline_s", None)
        outer = human_leg_budget_left()
        effective = own if outer is None or not isinstance(own, (int, float)) else min(own, outer)
        obs.emit(log, logging.INFO, "human_leg_close_after_start", call_sid=call_sid,
                 conversation_id=conversation_id, deadline_s=effective, decision="close")
        try:
            closer.close(call_sid=call_sid, conversation_id=conversation_id)
        except Exception as exc:  # the closer never raises; the 204 must go out anyway
            obs.log_caught(log, "conversation close failed", exc, call_sid=call_sid,
                           decision="ignored_best_effort")

    def _accept(self, req: http.TwilioRequest, call_sid: str, client: HumanLegClient,
                close_after: list[str]) -> dict[str, Any] | None:
        """One accept under the budget. ``close_after`` gets the conversation id when the call ended
        during a failed transcription start (the caller closes it after the budget). Once the
        claim is taken, ``record`` always runs (a failed request in between releases the
        participant claim)."""
        st = callstate.load(self.d.store, call_sid)
        action = humanleg.worker_event_to_action(req.params, st)
        if action is None:
            self._log_accept_ignored(call_sid, st)
            return None
        now = int(self._now())
        claimed: list[humanleg.AcceptClaim] = []

        def claim(s: CallState) -> Mapping[str, Any]:
            c = humanleg.claim_accept(s.escalation, action, now)
            claimed[:] = [c]
            return {"escalation": c.escalation} if c.escalation is not None else {}

        st = callstate.mutate(self.d.store, call_sid, claim, now=self._now)
        won = claimed[-1]
        if won.action is None:
            self._log_accept_ignored(call_sid, st, reason=won.reason)
            return None
        act = won.action
        if not act.first and not act.add:  # a worker already listed accepts again (transferred back)
            started: list[str] = []
            if act.conversation_id:  # the attributes may name a new agent leg
                started, close_now = self._start_legs(req, call_sid, act.conversation_id)
                if close_now:
                    close_after[:] = [act.conversation_id]
            return {"conversation_id": act.conversation_id, "first": False,
                    "transcriptions_started": len(started), "outcome": won.reason}
        conversation_id = act.conversation_id or None
        if conversation_id is None:
            try:  # after the claim: an error means no conversation, and `record` still runs
                conversation_id = self._accept_conversation(st, client)
            except Exception as exc:
                obs.log_caught(log, "human_leg_accept_failed", exc, operation="accept_conversation",
                               decision="record_anyway", call_sid=call_sid)
        ids: dict[str, Any] = {"call_sid": call_sid, "conversation_id": conversation_id}
        marked = added = False
        started = []
        if conversation_id is None:
            obs.emit(log, logging.INFO, "human_leg_no_conversation", operation="accept",
                     decision="ignored_best_effort", call_sid=call_sid)
        else:
            try:
                if act.first and self._still_open(call_sid):
                    marked = client.mark_handoff(conversation_id, humanleg.handoff_metadata(now)) is True
                    if not marked:
                        obs.emit(log, logging.WARNING, "human_leg_handoff_not_marked",
                                 decision="ignored_best_effort", **ids)
                if act.add and self._still_open(call_sid):
                    added = client.add_human_participant(
                        conversation_id, act.worker_sid, act.identity) is True
            except Exception as exc:  # the clients never raise; a call-state read can
                obs.log_caught(log, "human_leg_accept_failed", exc, operation="accept_requests",
                               decision="record_anyway", **ids)
            started, close_now = self._start_legs(req, call_sid, conversation_id)
            if close_now:
                close_after[:] = [conversation_id]

        def record(s: CallState) -> Mapping[str, Any]:
            new = humanleg.record_accept(s.escalation, act.worker_sid, added=added,
                                         conversation_id=conversation_id, now=now)
            return {"escalation": new} if new is not None else {}

        after = callstate.mutate(self.d.store, call_sid, record, now=self._now)
        if humanleg.conversation_closed(after):
            obs.emit(log, logging.INFO, "human_leg_after_close", operation="accept_record",
                     decision="no_op", **ids)
        return {"conversation_id": conversation_id, "first": act.first, "handoff_marked": marked,
                "participant": "added" if added else ("not_added" if act.add else "listed"),
                "transcriptions_started": len(started), "outcome": won.reason}

    def _still_open(self, call_sid: str) -> bool:
        """The conversation is not closed (re-read before each write to it)."""
        st = callstate.load(self.d.store, call_sid)
        if st is not None and humanleg.conversation_closed(st):
            obs.emit(log, logging.INFO, "human_leg_after_close", operation="accept", decision="no_op",
                     call_sid=call_sid)
            return False
        return st is not None

    def _accept_conversation(self, st: CallState, client: HumanLegClient) -> str | None:
        """Neither call state nor the task attributes know the conversation: the session's, else one
        lookup by CallSid (within the budget) when the call had an AI session."""
        sess = sessions.load(self.d.store, st.call_sid)
        if sess is not None and sess.conversation_id:
            return sess.conversation_id
        if st.ai_sessions == 0:
            return None
        found = client.resolve_conversation(st.call_sid)
        return found if isinstance(found, str) and found else None

    def _agent_legs_on(self) -> bool:
        """The ``human_leg.agent_legs`` switch; a policy that can't answer reads as off (the
        fallback: one ``both_tracks`` transcription, no agent legs)."""
        try:
            return self.d.policy.human_leg_agent_legs is True
        except Exception:  # noqa: BLE001 - a policy that can't answer reads as off
            return False

    def _start_legs(self, req: http.TwilioRequest, call_sid: str,
                    conversation_id: str) -> tuple[list[str], bool]:
        """The transcriptions an accept or a ``task.updated`` starts: only
        with the ``human_leg.transcription`` switch on and a starter wired, and only legs not yet
        recorded (:func:`owlbank.humanleg.legs_to_start`): the customer leg (first, or again after a
        released failed start), then with ``human_leg.agent_legs`` on each agent leg in the task
        attributes (``inbound_track`` each; off: one ``both_tracks`` on the customer leg, no agent
        legs). Each leg is claimed before its request and released if it fails (:meth:`_start_leg`).
        Contained: any error is logged and counts as not started.

        Returns ``(legs started, close_now)``: ``close_now`` when a start failed and the call ended
        meanwhile (its close was waiting for this transcript); the caller closes after its budget,
        so the close gets its own deadline."""
        ids: dict[str, Any] = {"call_sid": call_sid, "conversation_id": conversation_id}
        plan: list[tuple[str, str, Mapping[str, str] | None]] = []
        try:
            starter = self.d.transcriptions
            if starter is None or not humanleg.transcription_enabled(self.d.policy):
                return [], False
            st = callstate.load(self.d.store, call_sid)
            if st is None:
                return [], False
            with_agents = self._agent_legs_on()
            todo = humanleg.legs_to_start(st.escalation, req.params.get("TaskAttributes"), call_sid,
                                          with_agents=with_agents)
            if not todo:
                return [], False
            lang = st.language or self.d.catalog.default_language
            stt = humanleg.stt_settings(self.d.catalog.settings(lang).get("conversationRelay"))
            common: dict[str, Any] = {
                "enabled": True, "language": stt.language_code or lang, "engine": stt.engine,
                "speech_model": stt.speech_model,
                "status_url": req.absolute_url(http.TRANSCRIPTION_ROUTE, credentials=self.d.basic_credentials())}
            if todo.customer:
                plan.append((call_sid, humanleg.ROLE_CUSTOMER, humanleg.transcription_start(
                    conversation_id, agent_legs=with_agents, **common)))
            plan.extend((leg, humanleg.ROLE_AGENT, humanleg.agent_transcription_start(
                conversation_id, customer_call_sid=call_sid, **common)) for leg in todo.agents)
        except Exception as exc:
            obs.log_caught(log, "human_leg_transcription_start_failed", exc, operation="start_transcription",
                           decision="handoff_without_transcript", **ids)
            return [], False
        started: list[str] = []
        close_now = False
        for leg, role, params in plan:
            if params is None:
                continue
            ok, close = self._start_leg(starter, call_sid, leg, role, params, ids)
            if ok:
                started.append(leg)
            close_now = close_now or close
        return started, close_now

    def _start_leg(self, starter: TranscriptionStarter, call_sid: str, leg: str, role: str,
                   params: Mapping[str, str], ids: Mapping[str, Any]) -> tuple[bool, bool]:
        """One leg's start: claimed in the call's state (:func:`owlbank.humanleg.claim_transcription`:
        not after the call's end or the close, once per leg, the agent-leg cap), posted to the leg's
        call, and released if it fails (:meth:`_release_leg`). ``(started, close_now)``; never raises."""
        leg_ids = {**ids, "leg": leg, "role": role}
        claim_sent = False
        try:
            claimed: list[bool] = []

            def claim(s: CallState) -> Mapping[str, Any]:
                new = humanleg.claim_transcription(s.escalation, leg, name=params["Name"], role=role,
                                                  now=int(self._now()))
                claimed[:] = [new is not None]
                return {"escalation": new} if new is not None else {}

            claim_sent = True  # from here an error may leave the leg claimed: release it below
            callstate.mutate(self.d.store, call_sid, claim, now=self._now)
            if not claimed[-1]:
                obs.emit(log, logging.INFO, "human_leg_transcription_skipped",
                         reason="recorded_ended_or_closed", decision="no_op", **leg_ids)
                return False, False
            if starter.start_transcription(leg, params) is True:
                return True, False
            obs.emit(log, logging.WARNING, "human_leg_transcription_not_started",
                     decision="handoff_without_transcript", **leg_ids)
        except Exception as exc:
            obs.log_caught(log, "human_leg_transcription_start_failed", exc, operation="start_transcription",
                           decision="handoff_without_transcript", **leg_ids)
            if not claim_sent:
                return False, False
        return False, self._release_leg(call_sid, leg, leg_ids)

    def _release_leg(self, call_sid: str, leg: str, ids: Mapping[str, Any]) -> bool:
        """Release ``leg``'s claimed transcription after a failed start, so a later event may start it
        again; if that write fails, record the leg's ``ended_at`` instead, so the close
        never waits for a transcript that will not come. ``True`` when the call has ended and the
        conversation is still open (close now). Never raises: if both writes fail, the sweep closes
        the conversation later."""
        def release(s: CallState) -> Mapping[str, Any]:
            new = humanleg.release_transcription(s.escalation, leg)
            return {"escalation": new} if new is not None else {}

        def mark_ended(s: CallState) -> Mapping[str, Any]:
            new = humanleg.end_unstarted_transcription(s.escalation, leg, int(self._now()))
            return {"escalation": new} if new is not None else {}

        after: CallState | None = None
        for fn, failure in ((release, "human_leg_transcription_release_failed"),
                            (mark_ended, "human_leg_transcription_end_record_failed")):
            try:
                after = callstate.mutate(self.d.store, call_sid, fn, now=self._now)
                break
            except Exception as exc:
                obs.log_caught(log, failure, exc, operation="start_transcription",
                               decision="record_transcript_end" if fn is release else "sweep_closes_later",
                               **ids)
        if after is None:
            return False
        return (after.escalation or {}).get(humanleg.K_CALL_ENDED_AT) is not None \
            and not humanleg.conversation_closed(after)

    # ------------------------------------------------------------------ human leg: task.updated

    def _human_leg_task_updated(self, req: http.TwilioRequest) -> None:
        """``task.updated``: Flex writes the conference's legs into the task
        attributes after the conference forms, and again on a transfer. For an escalated call past
        its hand-off, with the human-leg client and starter wired and the ``human_leg.transcription``
        switch on, start the transcriptions not yet recorded (:meth:`_start_legs`: new agent legs with
        ``human_leg.agent_legs`` on, and the customer leg again if its start failed), within
        :data:`HUMAN_LEG_ACCEPT_BUDGET_S`. The workspace's filter sends every task update here, so an
        event for a task that isn't ours (no call SID or no call state), a call not escalated or
        closed, or with nothing new is dropped after one read, with no write. Never raises."""
        if self.d.human_leg is None or self.d.transcriptions is None \
                or req.params.get("EventType") != humanleg.TASK_UPDATED_EVENT:
            return
        call_sid = taskrouter_call_sid(req.params)
        if call_sid is None:  # not one of our calls' tasks: nothing logged (workspace-wide events)
            return
        try:
            if not humanleg.transcription_enabled(self.d.policy):
                return
            st = callstate.load(self.d.store, call_sid)
            escalation = st.escalation if st is not None else None
            todo = humanleg.legs_to_start(escalation, req.params.get("TaskAttributes"), call_sid,
                                          with_agents=self._agent_legs_on())
            conversation_id = (escalation or {}).get(humanleg.K_CONVERSATION_ID)
            if not todo or not isinstance(conversation_id, str) or not conversation_id:
                obs.emit(log, logging.DEBUG, "human_leg_task_update_ignored", call_sid=call_sid,
                         reason=todo.reason if not todo else "no_conversation", decision="no_op")
                return
        except Exception as exc:
            obs.log_caught(log, "human_leg_task_update_failed", exc, call_sid=call_sid,
                           operation="task_updated", decision="ignored_best_effort")
            return
        started_at = time.monotonic()
        started: list[str] = []
        close_now = False
        try:
            with human_leg_budget(HUMAN_LEG_ACCEPT_BUDGET_S):
                started, close_now = self._start_legs(req, call_sid, conversation_id)
        except Exception as exc:
            obs.log_caught(log, "human_leg_task_update_failed", exc, call_sid=call_sid,
                           operation="task_updated", decision="ignored_best_effort")
        obs.emit(log, logging.INFO, "human_leg_task_updated", call_sid=call_sid,
                 conversation_id=conversation_id, customer_leg=todo.customer, agent_legs=len(todo.agents),
                 transcriptions_started=len(started),
                 elapsed_ms=int((time.monotonic() - started_at) * 1000))
        if close_now:
            self._close_after_failed_start(call_sid, conversation_id)

    @staticmethod
    def _log_accept_ignored(call_sid: str, st: CallState | None, *, reason: str | None = None) -> None:
        if humanleg.conversation_closed(st) or reason == "after_close":
            obs.emit(log, logging.INFO, "human_leg_after_close", operation="accept", decision="no_op",
                     call_sid=call_sid)
            return
        if reason is None:
            reason = "not_escalated" if st is None or not st.escalation else "duplicate_or_no_worker"
        obs.emit(log, logging.DEBUG, "human_leg_accept_ignored", call_sid=call_sid, reason=reason,
                 decision="no_op")

    def _stop_runtime_session(self, st: CallState | None) -> None:
        """``agentcore``: stop the call's Runtime session, exactly once per call
        (the ``#runtime-stop`` claim absorbs repeated and out-of-order callbacks). Only for
        a call in state with a CallSid, as every session we start is. Never raises."""
        stopper = self.d.runtime_sessions
        if self.cfg.hosting != "agentcore" or stopper is None or st is None:
            return
        if not CALL_SID_RE.match(st.call_sid):
            return
        if claim_stop(self.d.store, st.call_sid, expires_at=st.expires_at):
            try:
                stopper.stop(st.call_sid)
            except Exception as exc:  # the contract says never; status must answer anyway
                obs.log_caught(log, "runtime_session_stop_failed", exc, call_sid=st.call_sid,
                               decision="idle_timeout_stops_it")


# --------------------------------------------------------------------------- parsing helpers


#: TaskRouter events that end a hand-off's wait in the queue, and the outcome each records.
TASKROUTER_OUTCOMES: Mapping[str, str] = {
    "reservation.accepted": "accepted", "task.canceled": "cancelled", "workflow.timeout": "timeout"}


def taskrouter_outcome(params: Mapping[str, str]) -> str | None:
    """``accepted``, ``cancelled`` (``hangup`` when TaskRouter says the caller hung up) or
    ``timeout`` for the events :data:`TASKROUTER_OUTCOMES` lists, else ``None``."""
    outcome = TASKROUTER_OUTCOMES.get(params.get("EventType", ""))
    if outcome == "cancelled" and params.get("TaskCanceledReason", "").strip().lower() == "hangup":
        return "hangup"
    return outcome


def taskrouter_call_sid(params: Mapping[str, str]) -> str | None:
    """The ``call_sid`` in a TaskRouter event's ``TaskAttributes`` (``build_task_attributes`` and
    ``<Enqueue>`` both put it there), if it is a CallSid; else ``None``."""
    raw = params.get("TaskAttributes")
    if not raw or len(raw) > 64_000:
        return None
    try:
        attrs = json.loads(raw)
    except ValueError:
        return None
    sid = attrs.get("call_sid") if isinstance(attrs, dict) else None
    return sid if isinstance(sid, str) and CALL_SID_RE.match(sid) else None


def transcription_call_sid(req: http.TwilioRequest) -> str | None:
    """The customer call a ``/voice/transcription`` callback belongs to: the signed
    ``call`` query parameter of an agent leg's status URL, else (absent) the leg itself (the
    customer leg's URL has none). ``None`` when ``call`` is present but not a call SID, or neither is."""
    call = req.query.get(humanleg.CALL_QUERY_PARAM)
    if call is None:
        call = req.call_sid
    return call if isinstance(call, str) and CALL_SID_RE.match(call) else None


def last_step(st: CallState | None, sess: Session | None) -> str | None:
    """Where the call was when it ended, from the state the front door
    keeps (no step is stored): the open question's step name (``gather.<slot>``,
    ``confirm.readback``, ``verify`` while a code is awaited), else ``handoff`` after an
    escalation, else ``ai`` after a ConversationRelay session, else ``None``."""
    flow = (sess.flow if sess is not None else None) or {}
    if flow.get("kind") == "confirm":
        return "confirm.readback"
    if flow.get("otp_wait"):
        return "verify"
    factor = flow.get("factor")
    if isinstance(factor, str) and factor:
        return f"gather.{factor}"
    if st is not None and st.escalation:
        return "handoff"
    if st is not None and st.ai_sessions > 0:
        return "ai"
    return None


#: An event type is logged as-is only when it looks like an enum (``COMMUNICATION_CREATED``).
_EVENT_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}\Z")
UNRECOGNISED_EVENT = "unrecognised"
_EVENT_TIME_FIELDS = ("timestamp", "updatedAt", "createdAt")


def _event_epoch_ms(value: Any) -> float | None:
    if not isinstance(value, str) or len(value) > 40:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.timestamp() * 1000 if parsed.tzinfo is not None else None


def describe_orchestrator_event(req: http.CallbackRequest, *, received_ms: int) -> dict[str, Any]:
    """What ``orchestrator_event_dropped`` may carry: ``event_type`` (an enum-shaped value, else
    ``unrecognised``; absent when the event has none) and ``lag_ms`` (``received_ms`` minus the
    event's own time, when one parses as ISO-8601 with a zone). Nothing else from the event: no
    field value, ID, address, transcript or URL."""
    payload: Any
    if req.is_json:
        try:
            payload = json.loads(req.body) if req.body else {}
        except ValueError:
            payload = {}
    else:
        payload = http.parse_form(req.body)
    if not isinstance(payload, Mapping):
        return {}
    out: dict[str, Any] = {}
    raw_type = payload.get("eventType", payload.get("EventType"))
    if raw_type is not None:
        out["event_type"] = (raw_type if isinstance(raw_type, str) and _EVENT_TYPE_RE.match(raw_type)
                             else UNRECOGNISED_EVENT)
    data = payload.get("data")
    for source in (payload, data if isinstance(data, Mapping) else {}):
        stamp = next((t for t in (_event_epoch_ms(source.get(f)) for f in _EVENT_TIME_FIELDS)
                      if t is not None), None)
        if stamp is not None and math.isfinite(stamp):
            out["lag_ms"] = round(received_ms - stamp)
            break
    return out


def prefetched_balance(prefetched: Mapping[str, Any] | None, *, tool: str,
                       tool_use_id: str | None, args: Mapping[str, Any], now: float
                       ) -> tuple[ToolResult | None, str | None, float | None]:
    """The bridge's hand-off prefetch as the bank's ``ok`` result, if it may be used.

    Returns ``(result, None, age_s)`` when the record is for ``tool``, the pending tool call's
    ``tool_use_id`` and the same ``args``, at most :data:`PREFETCH_MAX_AGE_S` old and shaped
    like a balance; otherwise ``(None, reason, age_s or None)`` with ``reason`` one of
    ``missing``, ``other_call``, ``other_args``, ``stale`` or ``bad_shape``. Pure: the caller
    validates the token first and clears the record either way.
    """
    if prefetched is None:
        return None, "missing", None
    if not isinstance(prefetched, Mapping):
        return None, "bad_shape", None
    if (prefetched.get("tool") != tool or tool_use_id is None
            or prefetched.get("tool_use_id") != tool_use_id):
        return None, "other_call", None
    pre_args = prefetched.get("args")
    if not isinstance(pre_args, Mapping) or dict(pre_args) != dict(args):
        return None, "other_args", None
    at = prefetched.get("at")
    if isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at):
        return None, "bad_shape", None
    age = float(now) - float(at)
    if age > PREFETCH_MAX_AGE_S or age < -2:  # a timestamp from the future (beyond clock skew) is not trusted
        return None, "stale", age
    raw = prefetched.get("result")
    if not isinstance(raw, Mapping) or raw.get("status", "ok") != "ok":
        return None, "bad_shape", age
    res = parse_tool(raw)
    if res.status != "ok" or not _balance_shaped(res.data):
        return None, "bad_shape", age
    return res, None, age


def _balance_shaped(data: Mapping[str, Any]) -> bool:
    """``data`` has what :meth:`FrontDoor._balance_speech` reads: ``last4`` and decimal amounts."""
    if not isinstance(data.get("last4"), str) or not isinstance(data.get("available"), str):
        return False
    for k in _BALANCE_AMOUNTS:
        v = data.get(k)
        if v is None:
            continue
        if not isinstance(v, str):
            return False
        try:
            if not Decimal(v).is_finite():
                return False
        except ArithmeticError:
            return False
    return True


def _hangup_reason(handoff: Mapping[str, Any]) -> str | None:
    """:data:`NO_INPUT` (the silence escalation), :data:`NO_INPUT_HANGUP` (the silence hang-up)
    or :data:`CALLER_DONE` (the model's ``end_call``) when ``handoff`` is
    one of them, else ``None``. Accepted forms: ``{"reason": "no_input_hangup"}``, ``{"reason":
    "no_input"}``, the escalation form ``{"reason": "escalate", "escalationReason": "no_input"}``
    (also with ``no_input_hangup``), and ``{"reason": "caller_done"}`` (only as the reason itself:
    an escalation never ends as ``caller_done``)."""
    def norm(value: Any) -> str:
        return value.lower().replace("-", "_") if isinstance(value, str) else ""
    reason = norm(handoff.get("reason"))
    if reason == CALLER_DONE:
        return CALLER_DONE
    if reason == "escalate":
        reason = norm(handoff.get("escalationReason"))
    return reason if reason in (NO_INPUT, NO_INPUT_HANGUP) else None


def _names(factor: Any) -> list[str]:
    """A decline's factor, as the ``factors`` list of a verification event (name only)."""
    return [factor] if isinstance(factor, str) and factor else []


def _keypad_reason(handoff: Mapping[str, Any], reasons: tuple[str, ...]) -> str | None:
    """The escalation reason of the bridge's keypad hand-off (``{"reason": "escalate",
    "escalationReason": "caller_requested_human", "trigger": "keypad", ...}``), else ``None``.
    Only a reason the policy knows is accepted; anything else takes the usual hand-off path."""
    if handoff.get("trigger") != KEYPAD_TRIGGER:
        return None
    if str(handoff.get("reason", "")).lower() != "escalate":
        return None
    reason = handoff.get("escalationReason")
    return reason if isinstance(reason, str) and reason in reasons else None


def _parse_handoff(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        log.warning("handoff payload is not JSON", extra={
            "operation": "parse_handoff", "decision": "ignored_because_malformed_handoff"})
        return None
    return data if isinstance(data, dict) else None


def _gather_id(value: Any) -> str | None:
    """A well-formed Gather id, else ``None``."""
    return value if isinstance(value, str) and _GATHER_ID.match(value) else None


#: The longest ``SequenceNumber`` read (digits); anything else is not a valid one.
_MAX_SEQUENCE_DIGITS = 9


def _sequence_number(params: Mapping[str, str]) -> int | None:
    """A partial callback's ``SequenceNumber`` (Twilio numbers a Gather's partials in order), or
    ``None`` when missing or not a plain non-negative integer of at most nine digits."""
    raw = params.get("SequenceNumber")
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not (0 < len(raw) <= _MAX_SEQUENCE_DIGITS and raw.isascii() and raw.isdigit()):
        return None
    return int(raw)


def _partial_text(params: Mapping[str, str]) -> str:
    """The partial transcript: ``UnstableSpeechResult`` (Twilio's documented parameter, the whole
    hypothesis so far); ``StableSpeechResult`` only when that is missing. The two are never
    joined, so no word can be counted twice."""
    return (params.get("UnstableSpeechResult") or params.get("StableSpeechResult") or "").strip()


def _safe_id(value: Any) -> str | None:
    return value if isinstance(value, str) and _SAFE_REF.match(value) else None
