"""Lambda entry point for the front door (API Gateway REST proxy integration).

Outside Lambda, importing this module creates no AWS clients and reads no secrets: the wiring
(``build_app``) runs lazily on the first request. Inside Lambda it runs once during the INIT
phase (:func:`owlbank.coldstart.build_at_init`), which a provisioned-concurrency
environment completes before any call arrives. The wiring reads no secret (the credentials are
callables); what it reads is the policy and catalog tables, as the first request did. Under
provisioned concurrency or SnapStart, INIT then builds the early-finish Twilio client, reading
its API key secret (:func:`prepare_early_finish`), and opens its HTTPS connection with
one account read (a failure is logged and ignored). Clients the ``incoming`` route never
uses (KMS, Bedrock, Comprehend) are created on first use. Environment (besides ``owlbank.config``):

* ``OWLBANK_HOSTING``: absent or ``fargate`` (default), or ``agentcore``.
* ``OWLBANK_RELAY_PARAMETER``: the Bridge stack's SSM String parameter
  (``/owlbank/<stage>/bridge/relay``), read once in :func:`build_app` (INIT) with
  ``ssm:GetParameter``. On ``fargate`` its value is the relay host and fills
  ``OWLBANK_RELAY_WS_URL`` with ``wss://<host>/ws``; on ``agentcore`` it is the Runtime ARN and fills
  ``OWLBANK_RUNTIME_ARN`` (:func:`owlbank.relay_param.resolve_relay_env`). A missing, empty or
  malformed parameter fails closed: no app, a ``relay_parameter_unavailable`` log line, and every
  request fails until it is fixed. Without this variable the two below are read as given.
* ``OWLBANK_RELAY_WS_URL`` (required on ``fargate``): the TAC bridge ``wss://.../ws`` URL.
* ``agentcore`` only: ``OWLBANK_RUNTIME_ARN``, ``OWLBANK_PRESIGN_ROLE_ARN`` (the
  relay URLs are signed locally with that role's credentials; the STS client is built on
  first use and the credentials are fetched at INIT, :func:`prepare_relay`),
  ``OWLBANK_URL_TTL_S`` (default 300), ``OWLBANK_WARMER_FUNCTION`` (``runtime_warmer``, invoked
  with ``InvocationType=Event`` on a new call's ``incoming``; its Lambda client is built on
  first use). The nonce always travels as ``<Parameter name="n">`` there. At call end
  ``/voice/status`` calls ``StopRuntimeSession`` on ``OWLBANK_RUNTIME_ARN`` (a
  ``bedrock-agentcore`` client built on first use, 2 s connect and read timeouts, one attempt).
* ``OWLBANK_CONVERSATION_CONFIGURATION_ID``, ``OWLBANK_WORKFLOW_SID``, ``OWLBANK_FALLBACKS``
  (JSON), ``OWLBANK_ACCEPT_SPOKEN_FACTORS``. The ``<Enqueue>`` has no ``waitUrl``: Twilio plays its
  default hold music.
* ``OWLBANK_HUMAN_LEG_CONVERSATION`` (``on`` | ``off``, absent is ``off``): with ``on`` and
  ``OWLBANK_SECRET_TWILIO_API_KEY``, an escalation that doesn't know the call's Orchestrator
  conversation looks it up once (:func:`human_leg_client`; the secret is read on first use), and
  the conversation is closed after the human leg (:func:`human_leg_closer`: on
  ``/voice/transcription`` ``transcription-stopped``, or on ``/voice/status`` when no transcription
  runs); ``OWLBANK_TASKROUTER_WORKSPACE_SID`` lets the close add a worker whose accept was missed.
  ``reservation.accepted`` on ``/voice/taskrouter`` then adds the worker as a participant, and the
  first one marks the hand-off; with policy ``human_leg.transcription`` on, accepts and
  ``task.updated`` start the customer leg's transcription and one per agent leg (policy
  ``human_leg.agent_legs``; off: one ``both_tracks`` transcription) (:func:`human_leg_transcriptions`).
* ``OWLBANK_RECAP_FUNCTION``: the ``recap_writer`` function, invoked with
  ``InvocationType=Event`` and ``{"call_sid": ...}`` on ``/voice/status`` when no human-leg close will
  follow, and by the closer after the close (:mod:`owlbank.recap_trigger`; a Lambda client built on
  first use, short timeouts, one attempt; a failure is logged and ignored). Unset: no recap.
* ``OWLBANK_TOKEN_KMS_KEY`` (``settings.token_kms_key``): seals the call token.
* ``OWLBANK_AUTH_SERVICE_FUNCTION``, ``OWLBANK_BANK_TOOLS_FUNCTION``: function names/ARNs (the
  ``live`` alias ARNs under provisioned concurrency).
* ``OWLBANK_EARLY_DIGITS`` (default true) and ``OWLBANK_SECRET_TWILIO_API_KEY``: spoken digits
  finish early through a Twilio REST call update, with a client built from the API key
  secret: at INIT under provisioned concurrency or SnapStart (the one secret
  read at INIT), otherwise on first use. Without the secret the feature is off.
* ``slot_input``: one INFO line per cold start, when the app is built, with
  ``acceptSpokenFactors`` and each slot's effective keypad-only and ``barge_in`` (names and
  booleans only).
* ``OWLBANK_SLOT_MODE_CEILING``: ``deterministic``, ``assisted``
  (default) or ``generative``; anything else fails INIT. The slot resolver
  (:class:`~owlbank.slots.BedrockResolver`) gets a ``bedrock-runtime`` client per budget, each a
  ``LazyClient`` built on first use; with the ``deterministic`` ceiling no resolver and no
  client exist at all. ``OWLBANK_RESOLVER_MODEL_ID`` overrides the model.
  ``slot_modes`` and ``slot_mode_capped`` (and any ``slot_mode_invalid``) are logged once per cold
  start, with the ``slot_input`` line (:func:`owlbank.slots.modes.log_mode_table`).
* ``OWLBANK_WORKFLOW_SID``: set only in Flex mode (an escalation enqueues to it). Unset, an
  escalation reads the caller its briefing and ends the call (:meth:`FrontDoor._escalate_briefing`);
  its model (:func:`build_briefing_model`) is a :class:`~owlbank.briefing.BedrockBriefingModel` over a
  client built at INIT (:func:`prepare_briefing`), or ``None`` with the ``deterministic`` ceiling
  (the fixed line is read; no client exists). ``OWLBANK_BRIEFING_BUDGET_MS`` (default 4000, 100 to
  5000) is the one overall deadline.
* ``OWLBANK_SPEECH_LOG`` (``on`` default, or ``off``) and ``OWLBANK_SPEECH_LOG_REDACTION`` (``on``
  default, or ``off``): the Gather turn's transcript on ``factor_answer``
  (a sensitive or login-factor slot's value as its placeholder), and its redaction; anything else
  fails INIT.
* ``OWLBANK_AI_KEYPAD`` (``on`` default, or ``off``): the AI session's
  ConversationRelay reports keys, also during the agent's speech (``dtmfDetection``,
  ``interruptible="any"``, ``reportInputDuringAgentSpeech="dtmf"``); ``off`` gives today's TwiML.
  Anything else fails INIT. The bridge reads the same variable.
* Slot preload: every ``owlbank.slots`` submodule is imported with this module
  (:data:`SLOT_MODULES`), and :func:`preload_slots` builds the slot mode table and reads the
  ``option.*`` and ``slot.*`` catalog texts once per cold start; a failure is logged
  (``slot_preload_failed``) and never fails INIT.
* PII check: Comprehend ``DetectPiiEntities`` in the Lambda's region
  (role: ``comprehend:DetectPiiEntities``); config from the policy ``redaction`` section.
* ``OWLBANK_CALL_OBSERVABILITY`` (``off`` default) and the other
  :class:`~owlbank.telemetry.settings.TelemetrySettings` variables: with ``on``, the front door's
  tracer provider is built once per process, at INIT or with the first app
  (:func:`install_telemetry`, :mod:`owlbank.telemetry.frontdoor`). With ``off`` the provider is
  never built and OpenTelemetry is never imported (the span hooks the app calls are no-ops).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

from owlbank import briefing, coldstart, obs
from owlbank.basic_auth import BasicSecret
from owlbank.briefing import BedrockBriefingModel, BriefingModel
from owlbank.humanleg_client import (
    OrchestratorHumanLeg,
    TwilioCalls,
    credentials_from_secret,
    taskrouter_reservations,
)
from owlbank.humanleg_close import HumanLegCloser
from owlbank.recap_trigger import RecapTrigger, recap_trigger
from owlbank.recap_trigger import client_config as recap_client_config
from owlbank.relay_param import resolve_relay_env
from owlbank.secret_schema import previous_pair
from owlbank.segments import SegmentLog
from owlbank.slot_policy import gather_input
from owlbank.slots import (
    BedrockResolver,
    Resolver,
    client_for_budget,
    dialog,
    generative,
    grounding,
    modes,
    resolver,
)
from owlbank.telemetry.settings import TelemetrySettings

from . import calls, http
from .app import HOSTING_ENV, HOSTINGS, Deps, FrontDoor, FrontDoorConfig, NoopConversationCloser
from .relay import RuntimeUrlSigner
from .runtime_sessions import CONNECT_TIMEOUT_S as STOP_CONNECT_TIMEOUT_S
from .runtime_sessions import READ_TIMEOUT_S as STOP_READ_TIMEOUT_S
from .runtime_sessions import RuntimeSessions

#: The Runtime warmer dispatch: an Event invoke is accepted in milliseconds; one attempt,
#: since the call goes on without it. Not the shared ``quick`` config.
WARMER_CONNECT_TIMEOUT_S = 1
WARMER_READ_TIMEOUT_S = 2
#: STS ``AssumeRole`` for the presign role: at INIT or, rarely, on a webhook.
STS_CONNECT_TIMEOUT_S = 2
STS_READ_TIMEOUT_S = 3
#: SSM ``GetParameter`` for the relay parameter: once, at INIT.
SSM_CONNECT_TIMEOUT_S = 2
SSM_READ_TIMEOUT_S = 3
#: Human leg (``on`` | ``off``, absent is ``off``): the escalation looks up the call's
#: conversation when it doesn't know it. One Orchestrator request, at most this long.
HUMAN_LEG_ENV = "OWLBANK_HUMAN_LEG_CONVERSATION"
HUMAN_LEG_TIMEOUT_S = 2.0
#: The TaskRouter workspace (set with the flag on): the close reads the task's reservations
#: from it to add a worker whose ``reservation.accepted`` was missed.
TASKROUTER_WORKSPACE_ENV = "OWLBANK_TASKROUTER_WORKSPACE_SID"

log = logging.getLogger(__name__)

#: The ``owlbank.slots`` submodules a card answer uses, imported here so the import is paid at cold
#: start: ``modes`` and ``generative`` are lazy attributes of the package (PEP 562), and
#: ``dialog``, ``grounding`` and ``resolver`` are named so the guarantee doesn't rest on what
#: ``app.py`` happens to import.
SLOT_MODULES = (dialog, generative, grounding, modes, resolver)
#: Catalog key prefixes read once at cold start: the options' spoken names and the slot
#: questions (confirmations, disambiguation, corrections, keypad menus).
PRELOAD_PREFIXES = ("option.", "slot.")

_app: FrontDoor | None = None
#: Whether :func:`install_telemetry` has run in this process (it runs once).
_telemetry_checked = False


class EventWarmer:
    """``Deps.warmer``: ``runtime_warmer`` invoked asynchronously with ``{"call_sid": ...}``.
    Raises on any failure; the front door logs ``runtime_prewarm_dispatch_failed`` and goes on."""

    def __init__(self, client: Any, function_name: str) -> None:
        if not function_name:
            raise ValueError("the warmer function name is empty")
        self._client = client
        self._function = function_name

    def __call__(self, call_sid: str) -> None:
        resp = self._client.invoke(FunctionName=self._function, InvocationType="Event",
                                   Payload=json.dumps({"call_sid": call_sid}).encode())
        status = resp.get("StatusCode")
        if status != 202:
            raise RuntimeError(f"warmer invoke returned status {status}")


def basic_secret_of(value: Mapping[str, Any]) -> BasicSecret:
    """The Basic secret from its parsed value; a missing or malformed previous pair means no
    previous password (fail closed)."""
    pair = previous_pair("basic_auth", value)
    return BasicSecret(str(value["user"]), str(value["password"]),
                       pair[0] if pair else None, pair[1] if pair else None)


def relay_signer(env: Mapping[str, str], sts: Any) -> RuntimeUrlSigner:
    """The ``agentcore`` relay URL signer from the environment. ``sts``: a (lazy) STS client."""
    ttl = (env.get("OWLBANK_URL_TTL_S") or "300").strip()
    if not ttl.isdigit():
        raise ValueError("OWLBANK_URL_TTL_S must be whole seconds")
    return RuntimeUrlSigner(runtime_arn=env["OWLBANK_RUNTIME_ARN"], region=env["AWS_REGION"],
                            role_arn=env["OWLBANK_PRESIGN_ROLE_ARN"], sts=sts, clock=time.time,
                            ttl_s=int(ttl))


def relay_env(env: Mapping[str, str], ssm_factory: Callable[[], Any]) -> dict[str, str]:
    """``env`` with its hosting's relay variable filled from the relay parameter. An unknown
    hosting is left for ``FrontDoorConfig.from_env`` to refuse, with no SSM read."""
    hosting = (env.get(HOSTING_ENV) or "").strip() or "fargate"
    if hosting not in HOSTINGS:
        return dict(env)
    return resolve_relay_env(env, hosting=hosting, ssm_factory=ssm_factory)


def human_leg_client(env: Mapping[str, str], api_key_secret: str | None) -> OrchestratorHumanLeg | None:
    """The escalation's conversation lookup: an :class:`OrchestratorHumanLeg`
    over the ``twilio_api_key`` secret when :data:`HUMAN_LEG_ENV` is ``on`` and the secret is
    configured, else ``None`` (no lookup; the escalation is unchanged). No secret is read here: the
    credentials are read on the first lookup. Its timeout is :data:`HUMAN_LEG_TIMEOUT_S`, inside the
    webhook's budget."""
    if (env.get(HUMAN_LEG_ENV) or "").strip().lower() != "on" or not api_key_secret:
        return None
    return OrchestratorHumanLeg(credentials_from_secret(api_key_secret), timeout_s=HUMAN_LEG_TIMEOUT_S)


def briefing_transcripts(model: BriefingModel | None, api_key_secret: str | None,
                         client: OrchestratorHumanLeg | None) -> OrchestratorHumanLeg | None:
    """The conversation reader of the escalation briefing: the human-leg client when there is one,
    else an :class:`OrchestratorHumanLeg` over the ``twilio_api_key`` secret (read on first use)
    when the briefing has a model, else ``None`` (no transcript). The human-leg switch does not
    matter: reading the transcript is not the human leg. Each request is under
    :data:`HUMAN_LEG_TIMEOUT_S` and the briefing's own transcript budget."""
    if model is None or not api_key_secret:
        return None
    return client or OrchestratorHumanLeg(credentials_from_secret(api_key_secret),
                                          timeout_s=HUMAN_LEG_TIMEOUT_S)


def human_leg_closer(env: Mapping[str, str], api_key_secret: str | None, store: Any,
                     client: OrchestratorHumanLeg | None, *,
                     journal: SegmentLog | None = None,
                     recap: RecapTrigger | None = None) -> HumanLegCloser | None:
    """The real ``ConversationCloser`` when the human-leg client is wired
    (:func:`human_leg_client`), else ``None`` (the front door keeps its no-op closer and AI-only
    behaviour is unchanged). It shares that client, stops the transcription through the Calls API
    and reads reservations from :data:`TASKROUTER_WORKSPACE_ENV` (absent: no repair), all over the
    ``twilio_api_key`` secret, read on first use; each request under :data:`HUMAN_LEG_TIMEOUT_S`
    and the whole close under ``CLOSE_DEADLINE_S``. ``journal`` gets the close's
    ``conversation_closed`` event; ``recap`` starts the recap writer after the close."""
    if client is None or not api_key_secret:
        return None
    credentials = credentials_from_secret(api_key_secret)
    workspace = (env.get(TASKROUTER_WORKSPACE_ENV) or "").strip() or None
    return HumanLegCloser(
        store, client,
        calls=TwilioCalls(credentials, timeout_s=HUMAN_LEG_TIMEOUT_S),
        reservations=(taskrouter_reservations(workspace, credentials, timeout_s=HUMAN_LEG_TIMEOUT_S)
                      if workspace else None),
        clock=time.time, journal=journal, recap=recap)


def human_leg_transcriptions(api_key_secret: str | None,
                             client: OrchestratorHumanLeg | None) -> TwilioCalls | None:
    """The Calls API client that starts the customer leg's transcription on the first accept
     when the human-leg client is wired, else ``None`` (no start). Over the
    ``twilio_api_key`` secret, read on first use; each request under :data:`HUMAN_LEG_TIMEOUT_S`
    (and the accept's 2 s total)."""
    if client is None or not api_key_secret:
        return None
    return TwilioCalls(credentials_from_secret(api_key_secret), timeout_s=HUMAN_LEG_TIMEOUT_S)


def build_app(env: dict[str, str] | None = None) -> FrontDoor:  # pragma: no cover - needs AWS
    import boto3
    from botocore.config import Config

    from owlbank import secrets
    from owlbank.catalog import PromptCatalog
    from owlbank.config import Settings
    from owlbank.pii import ComprehendPiiRedactor, PiiGuard, make_comprehend_client
    from owlbank.policy import Policy
    from owlbank.store import DynamoStore
    from owlbank.token_seal import KmsSealer

    from .backend import LambdaBackendClient
    from .calls import TwilioCallUpdater, twilio_client_factory

    env = dict(os.environ if env is None else env)
    ssm_config = Config(connect_timeout=SSM_CONNECT_TIMEOUT_S, read_timeout=SSM_READ_TIMEOUT_S,
                        retries={"mode": "standard", "total_max_attempts": 2})
    env = relay_env(env, lambda: boto3.client("ssm", config=ssm_config))
    settings = Settings.from_env(env)
    store = DynamoStore.from_settings(settings)
    # One attempt in total: botocore's max_attempts counts retries, so {"max_attempts": 1} meant two
    # attempts (up to ~12 s with backoff, near Twilio's 15 s webhook limit). A failure takes the
    # front door's own apology-and-escalate path instead.
    quick = Config(connect_timeout=2, read_timeout=4, retries={"total_max_attempts": 1})
    backend = LambdaBackendClient(
        boto3.client("lambda", config=quick),
        auth_function=env["OWLBANK_AUTH_SERVICE_FUNCTION"],
        bank_function=env["OWLBANK_BANK_TOOLS_FUNCTION"],
    )

    def auth_token() -> str:
        return str(secrets.get_secret(settings.secrets.twilio_auth_token or "",
                                      logical="twilio_auth_token")["auth_token"])

    def basic_secret(force: bool = False) -> BasicSecret:
        value = secrets.get_secret(settings.secrets.basic_auth or "", logical="basic_auth",
                                   force=force)
        return basic_secret_of(value)

    def basic_credentials() -> tuple[str, str]:
        """The current ``(user, password)``, for the callback URLs this function builds."""
        current = basic_secret()
        return current.user, current.password

    cfg = FrontDoorConfig.from_env(env)
    relay = warmer = runtime_sessions = None
    if cfg.hosting == "agentcore":
        region = env["AWS_REGION"]
        sts_config = Config(connect_timeout=STS_CONNECT_TIMEOUT_S, read_timeout=STS_READ_TIMEOUT_S,
                            retries={"mode": "standard", "total_max_attempts": 2})
        relay = relay_signer(env, coldstart.LazyClient(
            lambda: boto3.client("sts", region_name=region, config=sts_config)))
        warmer_config = Config(connect_timeout=WARMER_CONNECT_TIMEOUT_S,
                               read_timeout=WARMER_READ_TIMEOUT_S,
                               retries={"mode": "standard", "total_max_attempts": 1})
        warmer = EventWarmer(coldstart.LazyClient(lambda: boto3.client("lambda", config=warmer_config)),
                             env["OWLBANK_WARMER_FUNCTION"])
        stop_config = Config(connect_timeout=STOP_CONNECT_TIMEOUT_S, read_timeout=STOP_READ_TIMEOUT_S,
                             retries={"mode": "standard", "total_max_attempts": 1})
        runtime_sessions = RuntimeSessions(
            coldstart.LazyClient(lambda: boto3.client("bedrock-agentcore", region_name=region,
                                                      config=stop_config)),
            env["OWLBANK_RUNTIME_ARN"])

    policy = Policy.from_store(store, ttl_s=settings.cache_ttl_s)
    human_leg = human_leg_client(env, settings.secrets.twilio_api_key)
    recap_config = recap_client_config()
    recap = recap_trigger(env, lambda: coldstart.LazyClient(
        lambda: boto3.client("lambda", config=recap_config)))
    closer = human_leg_closer(env, settings.secrets.twilio_api_key, store, human_leg,
                              journal=SegmentLog(store, ttl_s=cfg.record_ttl_s), recap=recap)
    catalog = PromptCatalog.from_store(store, ttl_s=settings.cache_ttl_s)
    briefing_model = build_briefing_model(cfg, catalog)
    deps = Deps(
        store=store,
        catalog=catalog,
        policy=policy,
        backend=backend,
        auth_token=auth_token,
        basic_credentials=basic_credentials,
        basic_secret=basic_secret,
        # Not used by /voice/incoming: built on first use, so the first request doesn't pay for them.
        sealer=KmsSealer(coldstart.LazyClient(lambda: boto3.client("kms", config=quick)),
                         settings.token_kms_key or ""),
        pii=PiiGuard.for_policy(ComprehendPiiRedactor(coldstart.LazyClient(make_comprehend_client)), policy),
        resolver=build_resolver(cfg),
        briefing_model=briefing_model,
        transcripts=briefing_transcripts(briefing_model, settings.secrets.twilio_api_key, human_leg),
        # No secret read until the first early finish (or INIT).
        calls=(TwilioCallUpdater(twilio_client_factory(settings.secrets.twilio_api_key))
               if settings.secrets.twilio_api_key else None),
        relay=relay,
        warmer=warmer,
        runtime_sessions=runtime_sessions,
        human_leg=human_leg,
        closer=closer if closer is not None else NoopConversationCloser(),
        transcriptions=human_leg_transcriptions(settings.secrets.twilio_api_key, human_leg),
        recap=recap,
    )
    return FrontDoor(deps, cfg)


def build_resolver(cfg: FrontDoorConfig, client_factory: Callable[..., Any] | None = None
                   ) -> Resolver | None:
    """The slot resolver for this deployment: a :class:`BedrockResolver` whose
    ``bedrock-runtime`` clients are built on first use, one per budget. ``None`` with the
    ``deterministic`` ceiling: no resolver and no client factory call ever."""
    if cfg.slot_mode_ceiling == "deterministic":
        return None
    return BedrockResolver(client_for_budget(client_factory))


def build_briefing_model(cfg: FrontDoorConfig, catalog: Any,
                         client: Any | None = None) -> BriefingModel | None:
    """The escalation briefing's model, built once per cold start: a :class:`BedrockBriefingModel`
    only when the briefing is on (no Flex workflow configured, so the escalation reads the briefing)
    and the slot mode ceiling lets the front door call a model. ``None`` otherwise (Flex mode, or the
    ``deterministic`` ceiling): no model call and no ``bedrock-runtime`` client is ever built."""
    if cfg.workflow_sid or cfg.slot_mode_ceiling == "deterministic":
        return None
    return BedrockBriefingModel(catalog, client=client, budget_ms=cfg.briefing_budget_ms)


def prepare_briefing(app: FrontDoor) -> None:
    """At INIT, with the briefing on: build the model's client now, so the first briefing does not
    pay for loading the SDK inside its deadline. Elsewhere nothing. Never raises."""
    if app.d.briefing_model is None:
        return
    try:
        briefing.prewarm(app.cfg.briefing_budget_ms)
    except Exception as exc:
        obs.log_caught(log, "briefing_prewarm_failed", exc, decision="ignored_built_on_first_use")


def log_mode_table(app: FrontDoor) -> None:
    """``slot_modes``, ``slot_mode_capped`` and any ``slot_mode_invalid``, once per
    cold start. Names and modes only. Never raises (a failure is logged and the app is used)."""
    try:
        modes.log_mode_table(app.d.policy, app.cfg.slot_mode_ceiling)
    except Exception as exc:
        obs.log_caught(log, "slot_modes_failed", exc, decision="ignored")


def prepare_relay(app: FrontDoor) -> None:
    """At INIT: fetch the ``agentcore`` signing credentials now, so ``AssumeRole`` rarely
    falls on a call's webhook. On ``fargate``, nothing. Never raises (a failure is logged and the
    first ``<Connect>`` retries)."""
    if app.cfg.hosting != "agentcore":
        return
    refresh = getattr(app.relay_urls, "refresh_if_due", None)
    if callable(refresh):
        refresh()


def prepare_early_finish(app: FrontDoor, env: Mapping[str, str] | None = None) -> None:
    """At INIT under provisioned concurrency or SnapStart: import the Twilio SDK and
    build the REST client for early finish, reading the API key secret now, and warm its
    connection with one account read (inside ``prebuild``). Elsewhere, or with
    early finish off, nothing: the client is built on first use. Never raises."""
    if not (calls.build_at_init(env) and app.cfg.early_digits and app.d.calls is not None):
        return
    from lambdas.auth_service.verify_client import preload_twilio_sdk

    preload_twilio_sdk()
    prebuild = getattr(app.d.calls, "prebuild", None)
    if callable(prebuild):
        prebuild()  # logs and falls back to first use on failure


def log_slot_input(app: FrontDoor) -> None:
    """``slot_input``: the spoken-factors ceiling and, per slot, the effective
    keypad-only and ``barge_in`` the Gathers use (:func:`~owlbank.slot_policy.gather_input`).
    Names and booleans only. Never raises (a failure is logged and the app is used as is)."""
    try:
        accept = app.cfg.accept_spoken_factors
        policy = app.d.policy
        slots: dict[str, dict[str, bool]] = {}
        for name in policy.slots():
            shape = gather_input(policy.slot(name), accept)
            slots[name] = {"keypad_only": shape.keypad_only, "barge_in": shape.barge_in}
        obs.emit(log, logging.INFO, "slot_input", acceptSpokenFactors=accept, slots=slots)
    except Exception as exc:
        obs.log_caught(log, "slot_input_failed", exc, decision="ignored")


def preload_slots(app: FrontDoor) -> None:
    """Once per cold start: build the slot mode table and each slot's Gather
    input, and read every ``option.*`` and ``slot.*`` catalog template in every language, with
    the universal commands, so a call's first card answer pays no first read. (The modules are
    imported with this one, :data:`SLOT_MODULES`.) Logs ``slot_preload`` at INFO with counts
    and ``duration_ms``. Never raises: a failure is logged as ``slot_preload_failed`` and the app
    is used as is, the reads then happening on first use."""
    started = time.perf_counter()
    try:
        policy, catalog = app.d.policy, app.d.catalog
        modes.mode_table(policy, app.cfg.slot_mode_ceiling)
        names = policy.slots()
        for name in names:
            gather_input(policy.slot(name), app.cfg.accept_spoken_factors)
        texts = 0
        langs = catalog.languages
        for key in sorted(k for k in catalog.keys if k.startswith(PRELOAD_PREFIXES)):
            for lang in langs:
                if catalog.has(key, lang):  # no fallback lookup: it would log a warning per gap
                    catalog.template(key, lang)
                    texts += 1
        for lang in langs:
            catalog.commands(lang)
        warm = getattr(app.d.resolver, "warm", None)   # None with the deterministic ceiling
        if callable(warm):
            warm(policy.slot_resolution.resolver_budget_ms)
    except Exception as exc:
        obs.log_caught(log, "slot_preload_failed", exc, decision="ignored")
        return
    obs.emit(log, logging.INFO, "slot_preload", slots=len(names), texts=texts,
             languages=len(langs), duration_ms=round((time.perf_counter() - started) * 1000, 1))


def install_telemetry(env: Mapping[str, str] | None = None) -> bool:
    """Call observability: once per process, build the front door's tracer provider when
    ``OWLBANK_CALL_OBSERVABILITY`` is ``on``. ``True`` when it was installed by this call. With the
    switch off (or invalid) nothing is built and OpenTelemetry is not imported. Never raises."""
    global _telemetry_checked
    if _telemetry_checked:
        return False
    _telemetry_checked = True
    settings = TelemetrySettings.from_env(os.environ if env is None else env, service="frontdoor")
    if settings is None:
        return False
    from owlbank.telemetry import frontdoor as telemetry  # noqa: PLC0415 - only when on

    return telemetry.install_frontdoor(settings)


def new_app() -> FrontDoor:
    """The app for this execution environment (once per cold start), with its ``slot_input`` and
    ``slot_modes`` lines and the slot preload (:func:`preload_slots`). Telemetry is installed first
    when on (:func:`install_telemetry`)."""
    install_telemetry()
    app = build_app()
    log_slot_input(app)
    log_mode_table(app)
    preload_slots(app)
    return app


def _build_at_init() -> FrontDoor:  # pragma: no cover - Lambda INIT only
    install_telemetry()
    app = new_app()
    prepare_early_finish(app)
    prepare_relay(app)
    prepare_briefing(app)
    return app


@obs.instrumented("frontdoor", op=lambda event, _ctx: http.route_name(event))
def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    global _app
    if _app is None:
        _app = new_app()
    return _app.handle(event, remaining_ms=getattr(context, "get_remaining_time_in_millis", None))


# Kept for existing callers; the Lambda entry point configured in CDK is `handler`.
lambda_handler = handler

# Inside Lambda only: do the wiring now, in the INIT phase. Elsewhere (tests, scripts) a no-op.
_app = coldstart.build_at_init(_build_at_init)
