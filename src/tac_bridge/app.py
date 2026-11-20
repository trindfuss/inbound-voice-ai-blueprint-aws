"""Our FastAPI app: TAC's ``/ws`` and ``/webhook`` handlers only, plus ``/health``.

* **Not** ``TACFastAPIServer``: it always registers ``/twiml``, which must not be
  reachable. Only these routes exist: ``WS /ws``, ``POST /webhook``,
  ``GET /health``. No docs or OpenAPI routes.
* TAC runs in **Orchestrator mode** (``TWILIO_CONVERSATION_CONFIGURATION_ID`` is
  required), voice ``memory_mode="never"``, ``TAC_ANALYTICS_DISABLED=true``.
* **Signature validation** on ``/ws`` (before ``accept``; 1008 on failure) and
  ``/webhook``, with TAC's own dependencies.
* :class:`ForwardedHeadersMiddleware`: behind the ALB, ``X-Forwarded-Host`` is
  overwritten with ``TWILIO_VOICE_PUBLIC_DOMAIN`` and ``X-Forwarded-Proto`` forced to
  ``https``, so the URL TAC rebuilds is the one Twilio signed (``wss://<domain>/ws``).
* :class:`PeekingWebSocket`: reads ``setup`` (``callSid``, ``customParameters``)
  before TAC sees it, since TAC has no ``setup`` hook It also enforces
  the one-time relay nonce: the ``n`` query parameter of the signed URL must
  match the call's ``ws_nonce_hash``, which is then consumed. Otherwise the socket is
  closed before TAC creates a session, so no tool can run.
* **Logging:** every signature or nonce failure emits the stable
  :func:`owlbank.security_log.auth_failed` line
  (``{"event":"twilio_auth_failed","component":"bridge","reason":"signature|nonce"}``), and
  :func:`owlbank.security_log.install_log_redaction` strips query strings from every log
  record (TAC, uvicorn and ours), so the nonce never reaches a log. TAC installs its own
  handler; :func:`owlbank.obs.route_to_root` removes it, so TAC's records go through the root
  handler's PII scrubber and JSON formatter.
* :func:`build_ws_signature_dependency`: the ``/ws`` signature check. It accepts
  Twilio's signature over the full URL including the query (``wss://host/ws?n=...``)
  or TAC's convention (URL without query, query as parameters), so the nonce in the
  query can't break validation whichever form Twilio uses. Never logs the URL.
* **Webhook de-duplication**: :class:`IdempotencyCache` remembers each
  ``I-Twilio-Idempotency-Token`` (any header case) for 10 minutes, at most 10,000 of them (oldest
  dropped first). A repeated token is answered 200 without reaching TAC
  (``webhook_duplicate``, with an 8-character hash prefix, never the token); a webhook without
  a token is processed as before. Only SHA-256 digests are kept, never the tokens.

* **Agent loop**: always :class:`StrandsAgentClient`, calling Bedrock
  directly. Its settings come from ``OWLBANK_MODELS`` (a JSON list of one to
  three entries ``{"id", "thinking", "max_tokens", "text_release", "prompt_caching"}``, the
  preferred first; the first entry's text release is logged at start-up as ``agent_text_release``),
  ``OWLBANK_GATEWAY_URL``, ``OWLBANK_MEMORY_ID`` and ``AWS_REGION``, plus the test-only
  ``OWLBANK_FAILOVER_TEST`` (infra never sets it at prod, which is the real guard; the bridge also refuses it at ``OWLBANK_STAGE=prod``, but only when that is set, with call observability on); a missing or invalid one fails at
  start-up naming the variable (and the entry, by index), never its value. The ``model_failover``
  policy section is read per turn through the policy cache. ``models_resolved`` logs the list at
  start-up. With two or more entries and ``OWLBANK_TABLE_MODELSTATS`` (
  ``tac_bridge.failover_wiring``) the fleet reader, reporter and metrics are built; their loops
  start in the app's lifespan and stop at shutdown (with a final flush), and under AgentCore each
  call's end flushes the report. Without the table, or with a list of one, none is built.
  The Gateway MCP
  client is started once, and the Bedrock and Memory clients are built (their cold first-use cost
  paid) before the app is returned, so before uvicorn listens and ``/health`` answers. Shutdown
  (the app's lifespan) calls the client's ``aclose()``.

* **AI keypad shortcuts** ``OWLBANK_AI_KEYPAD`` (``on``,
  the default when absent or empty, or ``off``; anything else fails start-up naming the variable).
  On, :class:`PeekingWebSocket` hands each ``dtmf`` frame's digit to ``Bridge.on_keypad`` and
  swallows the frame; off, ``dtmf`` frames reach TAC as before.

* **Hosting** (:mod:`tac_bridge.hosting`): ``OWLBANK_HOSTING`` absent
  is Fargate, exactly as above. ``agentcore`` (AgentCore Runtime) adds ``GET /ping``
  (``HealthyBusy`` while a relay socket is open) and ``POST /invocations`` (``{"action":
  "prewarm"}`` -> ``{"status": "ready"}`` touching nothing; anything else 400), and drops
  ``/webhook``, the ``/ws`` signature dependency and :class:`ForwardedHeadersMiddleware` (the
  start-up line ``ws_signature_check`` says ``state=off``, ``reason=agentcore_hosting``). The
  Twilio secrets are read from Secrets Manager by ARN before ``TACConfig.from_env()``.
  **Call binding**: the process is pinned to the first Runtime session ID it sees
  (:class:`tac_bridge.hosting.SessionPin`). On ``setup`` the bridge first checks that the session-ID
  header, ``setup.callSid`` and ``customParameters.callSid`` are equal and the pin agrees, with no
  state write (``twilio_auth_failed`` ``reason=binding``, so a mismatched socket can't burn the
  call's nonce); then the nonce, whose only carrier is ``setup.customParameters["n"]`` (the query
  is never read: AgentCore forwards none), goes to the unchanged check
  (``reason=nonce``). ``n`` is removed from the parameters before ``on_setup`` and TAC see them.
  **Log parity**: both hostings log the same events with the same fields, plus ``hosting``
  (``fargate`` | ``agentcore``) on the start-up lines (``agent_settings``, ``startup_complete``) and
  the ``ws_session_*`` lines (stamped through :func:`owlbank.obs.set_event_fields`). The only
  hosting-specific line is ``agentcore``'s ``ws_signature_check``. ``startup_complete`` is logged
  once per process (uvicorn runs the app's lifespan once), when start-up is done and the server is
  about to accept sockets; ``scripts/bench/call_latency.py`` classifies cold connects by it.

Run: ``python -m tac_bridge.serve`` (TLS on Fargate), or locally
``uvicorn --factory tac_bridge.app:build_app_from_env --host 0.0.0.0 --port 8080``.
Importing this module makes no network call; ``build_app_from_env`` does
(``TAC(...)`` fetches the Conversation Configuration; the Gateway MCP client lists its tools).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, NoReturn

# A bank deployment never sends TAC usage telemetry.
os.environ["TAC_ANALYTICS_DISABLED"] = "true"

from fastapi import Depends, FastAPI, Request, WebSocket  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from owlbank import obs  # noqa: E402
from owlbank.security_log import auth_failed, install_log_redaction  # noqa: E402
from owlbank.web_activity import MAX_AGE_HOURS_RANGE, MAX_TRAIT_GROUPS, TRAIT_GROUP_NAME_RE  # noqa: E402
from tac_bridge.bridge import Bridge, BridgeConfig, SingleConversationGuard, TACConversationCloser  # noqa: E402
from tac_bridge.hosting import (  # noqa: E402
    INVOCATIONS_PATH,
    NONCE_NAME,
    PING_PATH,
    SESSION_ID_HEADER,
    Hosting,
    SessionPin,
    SocketTracker,
    hosting_from_env,
    load_twilio_secrets,
)
from tac_bridge.speaker_events import is_speaker_frame, parse_speaker_event, shape_of  # noqa: E402

if TYPE_CHECKING:
    from tac_bridge.agent_client_strands import ModelEntry, StrandsAgentClient, StrandsSettings

log = logging.getLogger(__name__)

WS_PATH = "/ws"
#: The start-up line logged when the app is ready to accept sockets (call_latency's cold start).
STARTUP_COMPLETE_EVENT = "startup_complete"
#: Bridge lines stamped with ``hosting``; the start-up lines carry it explicitly.
HOSTING_STAMPED_EVENTS = ("ws_session_start", "ws_session_end")
#: Inbound ConversationRelay frames that are caller input (the silence watch).
INBOUND_KINDS: frozenset[str] = frozenset({"prompt", "dtmf", "interrupt"})
#: AI keypad shortcuts: the ConversationRelay frame of a key press.
KEYPAD_FRAME = "dtmf"
#: An outbound ``end`` frame as TAC writes it (compact JSON; confirmed by parsing): the bridge notes
#: it went out (``Bridge.on_relay_end``), so a hand-off's ``end`` is never sent twice.
_END_FRAME_MARK = '"type":"end"'
#: AI keypad shortcuts 4.1: ``on`` | ``off``; absent or empty is ``on`` (the front door's too).
AI_KEYPAD_ENV = "OWLBANK_AI_KEYPAD"
#: ConversationRelay's own ``info`` and ``error`` frames: span events with type and code.
RELAY_SIGNAL_KINDS: frozenset[str] = frozenset({"info", "error"})
WEBHOOK_PATH = "/webhook"
HEALTH_PATH = "/health"
#: How long a webhook idempotency token is remembered, and how many at most.
IDEMPOTENCY_TTL_S = 600.0
IDEMPOTENCY_MAX_ENTRIES = 10_000
IDEMPOTENCY_HEADER = "i-twilio-idempotency-token"


def token_ref(token: str) -> str:
    """A short, non-reversible reference to a token for logs (8 hex characters of SHA-256)."""
    return hashlib.sha256(token.encode("utf-8", "surrogatepass")).hexdigest()[:8]


class IdempotencyCache:
    """Tokens seen in the last ``ttl_s`` seconds, at most ``max_entries``.

    Keyed by the token's SHA-256 digest (the token itself is never kept). An entry expires
    ``ttl_s`` after it was first seen; a repeat within that time doesn't extend it. Entries are
    kept in first-seen order, which is also expiry order, so pruning drops expired entries and,
    beyond ``max_entries``, the oldest first.
    """

    def __init__(self, *, ttl_s: float = IDEMPOTENCY_TTL_S, max_entries: int = IDEMPOTENCY_MAX_ENTRIES,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if max_entries < 1 or ttl_s <= 0:
            raise ValueError("max_entries and ttl_s must be positive")
        self.ttl_s = float(ttl_s)
        self.max_entries = int(max_entries)
        self._clock = clock
        self._seen: OrderedDict[bytes, float] = OrderedDict()  # digest -> expires at

    def __len__(self) -> int:
        return len(self._seen)

    def check_and_add(self, token: str) -> bool:
        """``True`` if ``token`` was seen and hasn't expired (a duplicate); otherwise it is
        recorded now and ``False`` is returned."""
        now = self._clock()
        key = hashlib.sha256(token.encode("utf-8", "surrogatepass")).digest()
        expires = self._seen.get(key)
        if expires is not None and now < expires:
            return True
        self._seen.pop(key, None)
        self._seen[key] = now + self.ttl_s
        self._prune(now)
        return False

    def _prune(self, now: float) -> None:
        # The oldest entries first: expired ones, and any beyond the bound.
        while self._seen:
            key, expires = next(iter(self._seen.items()))
            if expires <= now or len(self._seen) > self.max_entries:
                del self._seen[key]
            else:
                break


# --------------------------------------------------------------------------- ASGI middleware


class ForwardedHeadersMiddleware:
    """Pin ``X-Forwarded-Host``/``-Proto`` to the public domain (overwrite, never trust)."""

    def __init__(self, app: Any, public_domain: str) -> None:
        if not public_domain:
            raise ValueError("public_domain is required")
        self.app = app
        self._host = public_domain.encode("latin-1")

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") in ("http", "websocket"):
            headers = [(k, v) for k, v in scope.get("headers", [])
                       if k not in (b"x-forwarded-host", b"x-forwarded-proto")]
            headers += [(b"x-forwarded-host", self._host), (b"x-forwarded-proto", b"https")]
            scope = {**scope, "headers": headers}
        await self.app(scope, receive, send)


# --------------------------------------------------------------------------- WebSocket wrapper


class PeekingWebSocket:
    """A ``WebSocketProtocol`` wrapper that sees every inbound frame before TAC.

    On ``setup`` it records ``callSid`` and ``customParameters`` and schedules
    ``bridge.on_setup`` (idempotent: ``setup`` arrives again on reconnect). ``dtmf``
    frames are never logged (factors are collected outside the AI).

    After a verified ``setup``, each ``prompt``, ``dtmf`` and ``interrupt`` frame is reported to
    ``bridge.on_inbound(kind, call_sid)`` (caller input for the silence watch; no content),
    and each ``interrupt`` is logged as ``relay_interrupt``: what was playing (a catalog
    key, ``agent`` or ``greeting``) and, when ConversationRelay sends ``durationUntilInterruptMs``,
    how long it had played. Never the utterance text.

    AI keypad shortcuts (only when ``bridge.keypad_enabled``, ``OWLBANK_AI_KEYPAD``):
    after a verified ``setup`` a ``dtmf`` frame is reported to ``on_inbound`` as before, its digit
    goes to ``bridge.on_keypad(call_sid, digit)``, and the frame is swallowed: the next frame is
    read and returned instead, so TAC never sees it. The digit is never logged or kept here.
    Off: ``dtmf`` frames go on to TAC exactly as before.

    Speaker events (:mod:`tac_bridge.speaker_events`) go to
    ``bridge.on_speaker_event`` and are logged as ``relay_speaker_event`` (name and value only);
    TAC gets ``{"type": "ignored"}`` in their place, so they never reach it as caller input.

    Call observability (only when the bridge traces): every ``info`` and ``error`` frame
    becomes a ``relay.info``/``relay.error`` event on the session span with its type and code only
    (``Bridge.trace_relay_frame``), and an ``interrupt`` adds ``durationUntilInterruptMs`` and the
    length of ``utteranceUntilInterrupt`` (never its text) to the session and turn spans
    (``Bridge.trace_interrupt``). Frames reach TAC exactly as before.

    ``nonce_from_parameters`` (``agentcore``): the nonce is ``setup.customParameters["n"]``
    instead of ``nonce``, and ``n`` is removed from the parameters before ``on_setup`` or TAC sees
    them. ``expected_call_sid`` (``agentcore``): the call this socket may bind to (the Runtime
    session ID when the process pin agrees, else ``""``, which matches nothing). At ``setup``,
    ``callSid`` and ``customParameters.callSid`` must both equal it, or the socket is closed with
    ``reason=binding`` before the nonce is looked at (no store write). ``None`` (Fargate): no
    binding check, today's behaviour.
    """

    def __init__(self, inner: Any, bridge: Bridge, *, nonce: str | None = None,
                 nonce_from_parameters: bool = False, expected_call_sid: str | None = None) -> None:
        self._inner = inner
        self._bridge = bridge
        self._nonce = nonce
        self._nonce_from_parameters = nonce_from_parameters
        self._expected_call_sid = expected_call_sid
        self.verified = False
        self.call_sid: str | None = None
        self.custom_parameters: dict[str, Any] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        #: AI keypad shortcuts: ``dtmf`` frames go to the bridge and never reach TAC.
        self._keypad = bool(getattr(bridge, "keypad_enabled", False))

    async def accept(self) -> None:
        await self._inner.accept()

    async def send_text(self, data: str) -> None:
        if _END_FRAME_MARK in data:
            self._end_frame(data)
        await self._inner.send_text(data)

    def _end_frame(self, data: str) -> None:
        """An ``end`` frame is going out (TAC's hand-off): the bridge notes it before the send is
        awaited, so a turn cancelled inside that send never gets its ``end`` resent."""
        try:
            frame = json.loads(data)
        except ValueError:
            return
        note = getattr(self._bridge, "on_relay_end", None)
        if isinstance(frame, Mapping) and frame.get("type") == "end" and callable(note):
            note(self.call_sid)

    async def close(self) -> None:
        await self._inner.close()

    async def receive_json(self) -> Any:
        while True:
            msg = await self._inner.receive_json()
            if (self._keypad and self.verified and isinstance(msg, Mapping)
                    and msg.get("type") == KEYPAD_FRAME):
                # AI keypad shortcuts: the key goes to the bridge's buffer and the
                # frame is swallowed (TAC drops ``dtmf`` anyway; 3.2: it never sees the digit).
                self._inbound(msg)  # caller input: the silence watch resets
                self._keypad_key(msg)
                continue
            return await self._receive(msg)

    async def _receive(self, msg: Any) -> Any:
        if isinstance(msg, Mapping) and msg.get("type") == "setup":
            if self.verified:  # one setup per socket; a second one is ignored, never resumed
                log.warning("security: second setup on a relay socket ignored")
                return {"type": "ignored"}
            sid = msg.get("callSid")
            obs.bind_call_sid(sid)  # the socket's call SID is on every line from here, nonce failures included
            nonce, self._nonce = self._nonce, None
            raw_params = msg.get("customParameters")
            if self._nonce_from_parameters:  # one nonce source: the <Parameter>, never the query
                params = dict(raw_params) if isinstance(raw_params, Mapping) else {}
                carried = params.pop(NONCE_NAME, None)
                nonce = carried if isinstance(carried, str) and carried else None
                msg = {**msg, "customParameters": params}  # TAC never sees the nonce either
            call_sid = sid if isinstance(sid, str) and sid else None
            bound = self._bound(call_sid, raw_params)
            # Binding first, with no state write, so a mismatched socket never burns the nonce.
            accepted = bound and call_sid is not None and await self._bridge.accept_connection(call_sid, nonce)
            if not bound:
                log.warning("security: relay setup rejected (call binding mismatch)")
                await self._reject("binding")
            if call_sid is None or not accepted:
                log.warning("security: relay setup rejected (nonce missing, wrong or used)")
                await self._reject("nonce")
            self.verified = True
            self.call_sid = call_sid
            self._bridge.session_start(call_sid)
            params = msg.get("customParameters")
            self.custom_parameters = dict(params) if isinstance(params, Mapping) else {}
            task = asyncio.create_task(self._bridge.on_setup(call_sid, self.custom_parameters))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        elif isinstance(msg, Mapping) and self.verified and msg.get("type") in INBOUND_KINDS:
            self._inbound(msg)
        elif isinstance(msg, Mapping) and self.verified and is_speaker_frame(msg):
            if msg.get("type") in RELAY_SIGNAL_KINDS:  # every info frame is a session event
                self._relay_signal(msg)
            self._speaker(msg)
            return {"type": "ignored"}  # never caller input for TAC
        elif isinstance(msg, Mapping) and self.verified and msg.get("type") in RELAY_SIGNAL_KINDS:
            self._relay_signal(msg)
        return msg

    def _keypad_key(self, msg: Mapping[str, Any]) -> None:
        """A key pressed during the AI session: the digit goes to ``bridge.on_keypad`` and
        nowhere else (3.1: never logged, stored or traced here). Never breaks the socket."""
        try:
            digit = msg.get("digit")
            self._bridge.on_keypad(self.call_sid, digit if isinstance(digit, str) else "")
        except Exception as exc:  # noqa: BLE001
            obs.log_caught(log, "keypad bookkeeping failed", exc, operation="on_keypad",
                           decision="ignored")

    def _relay_signal(self, msg: Mapping[str, Any]) -> None:
        """Call observability: an ``info`` or ``error`` frame as a session-span event, its
        type and code only (never a description). The frame goes on to TAC unchanged."""
        try:
            kind = str(msg.get("type"))
            code = msg.get("code") if msg.get("code") is not None else msg.get("name")
            self._bridge.trace_relay_frame(self.call_sid, kind, code)
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks the socket
            obs.log_caught(log, "relay signal bookkeeping failed", exc, level=logging.DEBUG,
                           operation="trace_relay_frame", decision="ignored")

    def _bound(self, call_sid: str | None, params: Any) -> bool:
        """With an expected call, ``setup.callSid`` and ``customParameters.callSid`` both equal
        it (an empty expected call matches nothing). Without one (Fargate): always ``True``."""
        expected = self._expected_call_sid
        if expected is None:
            return True
        carried = params.get("callSid") if isinstance(params, Mapping) else None
        return bool(expected) and call_sid == expected and carried == call_sid

    async def _reject(self, reason: str) -> NoReturn:
        """Log the auth failure, close the socket and end it before TAC sees the ``setup``."""
        auth_failed("bridge", reason)
        try:
            await self._inner.close()
        finally:
            from tac.channels.websocket_protocol import WebSocketDisconnectError  # noqa: PLC0415

            raise WebSocketDisconnectError("relay setup rejected")

    def _speaker(self, msg: Mapping[str, Any]) -> None:
        """A speaker event (``relay_speaker_event``: ``speaker_event``, the event name, and ``value`` only)
        for the
        silence watch; an unrecognised shape is logged with its type and key names only and
        ignored (the watch keeps estimate timing). Never breaks the socket."""
        try:
            event = parse_speaker_event(msg)
            if event is None:
                obs.emit(log, logging.INFO, "relay_speaker_event", recognised=False, **shape_of(msg))
                return
            obs.emit(log, logging.INFO, "relay_speaker_event", recognised=True, speaker_event=event.name,
                     value=event.value)
            self._bridge.on_speaker_event(self.call_sid, event.who, event.speaking)
        except Exception as exc:  # noqa: BLE001
            obs.log_caught(log, "speaker event bookkeeping failed", exc, operation="on_speaker_event",
                           decision="ignored")

    def _inbound(self, msg: Mapping[str, Any]) -> None:
        """Caller input reached the socket (never its content). Never breaks the socket."""
        kind = str(msg.get("type"))
        try:
            if kind == "interrupt":
                ms = msg.get("durationUntilInterruptMs")
                played = ms if isinstance(ms, int | float) and not isinstance(ms, bool) else None
                obs.emit(log, logging.INFO, "relay_interrupt",
                         playing=self._bridge.now_playing(self.call_sid), ms_played=played)
                heard = msg.get("utteranceUntilInterrupt")
                self._bridge.trace_interrupt(self.call_sid, played,
                                             len(heard) if isinstance(heard, str) else None)  # Never the text
            self._bridge.on_inbound(kind, self.call_sid)
        except Exception as exc:  # noqa: BLE001
            obs.log_caught(log, "inbound frame bookkeeping failed", exc, operation="on_inbound",
                           decision="ignored")


# --------------------------------------------------------------------------- app


def build_ws_signature_dependency(auth_token: str) -> Any:
    """FastAPI dependency: validate ``X-Twilio-Signature`` on the ``/ws`` upgrade (1008 on failure)."""
    from starlette.websockets import WebSocketDisconnect  # noqa: PLC0415
    from tac.server.signature_validation import _build_websocket_url_and_params  # noqa: PLC0415
    from twilio.request_validator import RequestValidator  # noqa: PLC0415

    validator = RequestValidator(auth_token)

    async def _validate(websocket: WebSocket) -> None:
        signature = websocket.headers.get("x-twilio-signature")
        ok = False
        if signature:
            url, params = _build_websocket_url_and_params(websocket)
            query = websocket.url.query
            ok = validator.validate(url, params, signature) or bool(
                query and validator.validate(f"{url}?{query}", {}, signature))
        if not ok:
            log.warning("security: relay WebSocket signature missing or invalid")
            auth_failed("bridge", "signature")
            await websocket.close(code=1008, reason="Invalid Twilio signature")
            raise WebSocketDisconnect(code=1008)

    return _validate


def build_webhook_signature_dependency(auth_token: str) -> Any:
    """TAC's ``/webhook`` signature check, plus our auth-failure line."""
    from fastapi import HTTPException  # noqa: PLC0415
    from tac.server import build_http_signature_dependency  # noqa: PLC0415

    inner = build_http_signature_dependency(auth_token)

    async def _validate(request: Request) -> None:
        try:
            await inner(request)
        except HTTPException as exc:
            if exc.status_code == 403:
                auth_failed("bridge", "signature")
            raise

    return _validate


def build_app(tac: Any, voice_channel: Any, bridge: Bridge, *, public_domain: str | None = None,
              auth_token: str | None = None,
              on_shutdown: Sequence[Callable[[], Awaitable[Any]]] = (),
              idempotency: IdempotencyCache | None = None, hosting: Hosting = "fargate",
              tracker: SocketTracker | None = None, pin: SessionPin | None = None,
              on_startup: Sequence[Callable[[], Awaitable[Any]]] = (),
              on_call_end: Sequence[Callable[[], Awaitable[Any]]] = ()) -> FastAPI:
    """Wire TAC, the voice channel and the bridge into our FastAPI app (no network calls).
    ``on_startup`` runs, in order and best effort, when the app's lifespan starts (on the running
    loop: background loops such as model failover's fleet picture); ``on_shutdown`` runs, in order
    and best effort, when the app shuts down (e.g. the agent client's ``aclose()``);
    ``on_call_end`` (``agentcore`` only, where a process is one call) runs, best effort, when a
    relay socket closes. ``idempotency``: the webhook de-duplication cache (default a
    new :class:`IdempotencyCache`). ``hosting``: ``fargate`` (today's routes, dependencies and
    middleware) or ``agentcore`` (see the module docs); ``tracker``: the ``/ping`` state
    (``agentcore`` only; default a new :class:`SocketTracker`). ``pin``: the process's session pin
    (``agentcore`` only; default a new :class:`SessionPin`)."""
    if hosting not in ("fargate", "agentcore"):
        raise ValueError("hosting must be fargate or agentcore")
    obs.set_event_fields(HOSTING_STAMPED_EVENTS, hosting=hosting)  # 8.1
    if hosting == "agentcore":
        return _build_agentcore_app(tac, voice_channel, bridge, on_shutdown=on_shutdown,
                                    tracker=tracker if tracker is not None else SocketTracker(),
                                    pin=pin if pin is not None else SessionPin(),
                                    on_startup=on_startup, on_call_end=on_call_end)
    if tracker is not None or pin is not None or on_call_end:
        raise ValueError("tracker, pin and on_call_end are for hosting agentcore only")
    install_log_redaction()  # the relay nonce rides in the /ws query string: keep it out of logs
    obs.route_to_root(logging.getLogger("tac"))  # TAC's own handler would bypass the root scrubber

    domain = public_domain or tac.config.voice_public_domain
    if not domain:
        raise ValueError("TWILIO_VOICE_PUBLIC_DOMAIN is required (signature URL reconstruction)")
    token = auth_token or tac.config.auth_token
    if not token:
        raise ValueError("the Twilio auth token is required for signature validation")
    ws_path = tac.config.voice_websocket_path or WS_PATH

    bridge.voice = voice_channel
    tac.on_message_ready(bridge.on_message_ready)
    tac.on_interrupt(bridge.on_interrupt)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=_lifespan(on_shutdown, hosting="fargate", on_startup=on_startup))
    http_sig = build_webhook_signature_dependency(token)
    ws_sig = build_ws_signature_dependency(token)
    background: set[asyncio.Task[Any]] = set()
    seen = idempotency if idempotency is not None else IdempotencyCache()

    @app.get(HEALTH_PATH)
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.websocket(ws_path)
    async def ws(websocket: WebSocket, _: None = Depends(ws_sig)) -> None:
        await _relay_session(websocket, voice_channel, bridge)

    @app.post(WEBHOOK_PATH, dependencies=[Depends(http_sig)])
    async def conversation_webhook(request: Request) -> JSONResponse:
        data = await request.json()
        if not isinstance(data, dict):
            return JSONResponse({"status": "error"}, status_code=400)
        token = request.headers.get(IDEMPOTENCY_HEADER) or None  # headers are case-insensitive
        if token is not None and seen.check_and_add(token):
            obs.emit(log, logging.INFO, "webhook_duplicate", token_ref=token_ref(token),
                     decision="ignored")
            return JSONResponse({"status": "ok"})
        task = asyncio.create_task(voice_channel.process_webhook(data, token))
        background.add(task)
        task.add_done_callback(background.discard)
        return JSONResponse({"status": "ok"})

    app.add_middleware(ForwardedHeadersMiddleware, public_domain=domain)
    app.state.bridge = bridge
    app.state.tac = tac
    app.state.voice_channel = voice_channel
    app.state.idempotency = seen
    return app


def _lifespan(on_shutdown: Sequence[Callable[[], Awaitable[Any]]], *, hosting: Hosting,
              on_startup: Sequence[Callable[[], Awaitable[Any]]] = ()) -> Any:
    """The app's lifespan: at start-up, run ``on_startup`` in order, best effort, then the
    ``startup_complete`` line (all other start-up work is done before the app is built, and uvicorn
    starts listening right after the lifespan starts); on shutdown, run ``on_shutdown`` in order,
    best effort."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        for hook in on_startup:
            try:
                await hook()
            except Exception as exc:  # noqa: BLE001 - a background loop never stops the bridge
                obs.log_caught(log, "startup hook failed", exc, operation="startup",
                               decision="ignored_best_effort")
        obs.emit(log, logging.INFO, STARTUP_COMPLETE_EVENT, hosting=hosting)
        yield
        for hook in on_shutdown:
            try:
                await hook()
            except Exception as exc:  # noqa: BLE001 - shutdown is best effort
                obs.log_caught(log, "shutdown hook failed", exc, operation="shutdown",
                               decision="ignored_best_effort")

    return lifespan


async def _run_best_effort(hooks: Sequence[Callable[[], Awaitable[Any]]], operation: str) -> None:
    for hook in hooks:
        try:
            await hook()
        except Exception as exc:  # noqa: BLE001 - best effort
            obs.log_caught(log, f"{operation} hook failed", exc, operation=operation,
                           decision="ignored_best_effort")


async def _relay_session(websocket: WebSocket, voice_channel: Any, bridge: Bridge, *,
                         expected_call_sid: str | None = None) -> None:
    """One ConversationRelay socket, checked on ``setup`` before TAC. ``expected_call_sid`` ``None``
    (Fargate): the nonce is the signed query's ``n``. Otherwise (``agentcore``): the
    call binding to ``expected_call_sid``, then the nonce from ``customParameters["n"]`` only."""
    from tac.server import FastAPIWebSocketAdapter  # noqa: PLC0415

    agentcore = expected_call_sid is not None
    nonce = None if agentcore else websocket.query_params.get(NONCE_NAME)
    peeking = PeekingWebSocket(
        FastAPIWebSocketAdapter(websocket), bridge, nonce=nonce,
        nonce_from_parameters=agentcore, expected_call_sid=expected_call_sid)
    reason = "closed"
    close_code: int | None = None
    try:
        await voice_channel.handle_websocket(peeking)
    except asyncio.CancelledError:
        reason = "cancelled"
        raise
    except Exception as exc:
        reason = "disconnect" if type(exc).__name__.endswith("Disconnect") or \
            type(exc).__name__.endswith("DisconnectError") else "error"
        close_code = _close_code(exc)
        if reason == "error":
            obs.log_caught(log, "relay session failed", exc, level=logging.ERROR, decision="reraise")
        raise
    finally:
        if peeking.verified and peeking.call_sid:
            if close_code is None:
                bridge.session_end(peeking.call_sid, reason)
            else:
                bridge.session_end(peeking.call_sid, reason, close_code=close_code)


def _close_code(exc: BaseException) -> int | None:
    """The WebSocket close code a disconnect carried (``relay.session``'s ``owlbank.close_code``),
    from the exception or its cause; ``None`` when none is known."""
    for candidate in (exc, exc.__cause__, exc.__context__):
        code = getattr(candidate, "code", None)
        if isinstance(code, int) and not isinstance(code, bool):
            return code
    return None


PREWARM_ACTION = "prewarm"


def _build_agentcore_app(tac: Any, voice_channel: Any, bridge: Bridge, *,
                         on_shutdown: Sequence[Callable[[], Awaitable[Any]]],
                         tracker: SocketTracker, pin: SessionPin,
                         on_startup: Sequence[Callable[[], Awaitable[Any]]] = (),
                         on_call_end: Sequence[Callable[[], Awaitable[Any]]] = ()) -> FastAPI:
    """The AgentCore Runtime app: ``/health``, ``/ws`` (no Twilio signature dependency;
    the call binding and the ``<Parameter>`` nonce instead), ``GET /ping`` and ``POST /invocations``.
    No ``/webhook`` and no :class:`ForwardedHeadersMiddleware`."""
    install_log_redaction()  # no query reaches the container, but keep every query out of logs anyway
    obs.route_to_root(logging.getLogger("tac"))  # TAC's own handler would bypass the root scrubber
    ws_path = tac.config.voice_websocket_path or WS_PATH

    bridge.voice = voice_channel
    tac.on_message_ready(bridge.on_message_ready)
    tac.on_interrupt(bridge.on_interrupt)
    obs.emit(log, logging.INFO, "ws_signature_check", state="off", reason="agentcore_hosting")

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=_lifespan(on_shutdown, hosting="agentcore", on_startup=on_startup))

    @app.get(HEALTH_PATH)
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get(PING_PATH)
    async def ping() -> dict[str, Any]:
        return tracker.ping()

    @app.post(INVOCATIONS_PATH)
    async def invocations(request: Request) -> JSONResponse:
        # Start-up work is done before uvicorn listens, so prewarm itself touches nothing
        # (no store, TAC, Memory, Bedrock or Gateway call). The body is never logged.
        try:
            data = await request.json()
        except ValueError:
            data = None
        if isinstance(data, dict) and data.get("action") == PREWARM_ACTION:
            session_id = request.headers.get(SESSION_ID_HEADER)
            if session_id and not pin.check(session_id):  # Normally the call's first request pins
                log.warning("security: prewarm for a session other than the pinned one")
            return JSONResponse({"status": "ready"})
        return JSONResponse({"status": "error"}, status_code=400)

    @app.websocket(ws_path)
    async def ws(websocket: WebSocket) -> None:
        session_id = websocket.headers.get(SESSION_ID_HEADER)
        expected = session_id if session_id and pin.check(session_id) else ""  # "" binds to nothing
        tracker.opened()
        try:
            await _relay_session(websocket, voice_channel, bridge, expected_call_sid=expected)
        finally:
            tracker.closed()
            await _run_best_effort(on_call_end, "call_end")  # The call's report

    app.state.bridge = bridge
    app.state.tac = tac
    app.state.voice_channel = voice_channel
    app.state.tracker = tracker
    app.state.pin = pin
    app.state.hosting = "agentcore"
    return app


#: How many times TAC polls Conversation Orchestrator for the call's conversation when a
#: ConversationRelay session starts (TAC 2.5.0 default 10, about 15 s with its backoff capped at
#: 1.5 s). Observed: the conversation took 4 to 12 s to appear, and once more than 15 s,
#: so TAC gave up (Twilio 64105, "something went wrong"). 20 tries is about 30 s: a slow creation
#: becomes a pause, not an apology. Stopgap until one conversation per call.
TAC_CONVERSATION_POLL_ATTEMPTS = 20


def set_tac_conversation_polling(attempts: int = TAC_CONVERSATION_POLL_ATTEMPTS) -> None:
    """Set TAC's conversation polling count. TAC exposes no setting for it, so this sets the
    provider module's constant; ``attempts`` must be positive."""
    from tac.channels.voice.conversation_relay import provider  # noqa: PLC0415

    if attempts < 1:
        raise ValueError("attempts must be positive")
    if not hasattr(provider, "_POLL_ATTEMPTS"):
        raise RuntimeError("TAC no longer has _POLL_ATTEMPTS: review this stopgap for the TAC upgrade")
    provider._POLL_ATTEMPTS = attempts  # noqa: SLF001 - no public setting in TAC 2.5.0


_ON_OFF = {"on": True, "off": False}


MODELS_ENV = "OWLBANK_MODELS"
#: Agent prompt caching: the tools and static-system cache points' TTL (infra ``promptCacheTtl``).
PROMPT_CACHE_TTL_ENV = "OWLBANK_PROMPT_CACHE_TTL"
#: The keys of an ``OWLBANK_MODELS`` entry (infra ``ModelEntry.as_env``).
MODEL_ENTRY_KEYS = ("id", "thinking", "max_tokens", "text_release", "prompt_caching")
#: Model failover: the fleet table's name (``tac_bridge.fleet.TABLE_ENV``), set only with
#: ``modelFailover=on`` and two or more entries.
FLEET_TABLE_ENV = "OWLBANK_TABLE_MODELSTATS"


def models_from_env(env: Mapping[str, str]) -> tuple[ModelEntry, ...]:
    """``OWLBANK_MODELS``: a JSON list of one to three entries,
    each exactly ``{"id", "thinking", "max_tokens", "text_release", "prompt_caching"}``. Anything
    else raises ``RuntimeError`` naming the variable and the entry by index (``OWLBANK_MODELS[1]``),
    never a value; so does a parameter set the entry's model can't run (``text_release=stream`` with
    ``thinking=off``)."""
    from tac_bridge.agent_client_strands import MAX_MODEL_ENTRIES, ModelEntry  # noqa: PLC0415

    raw = (env.get(MODELS_ENV) or "").strip()
    if not raw:
        raise RuntimeError(f"{MODELS_ENV} is required")
    try:
        doc = json.loads(raw)
    except ValueError:
        raise RuntimeError(f"{MODELS_ENV} must be a JSON list of model entries") from None
    if not isinstance(doc, list) or not 1 <= len(doc) <= MAX_MODEL_ENTRIES:
        raise RuntimeError(f"{MODELS_ENV} must be a JSON list of 1 to {MAX_MODEL_ENTRIES} model entries")
    entries: list[ModelEntry] = []
    for i, item in enumerate(doc):
        where = f"{MODELS_ENV}[{i}]"
        if not isinstance(item, dict) or set(item) != set(MODEL_ENTRY_KEYS):
            raise RuntimeError(f"{where} must be an object with exactly the keys {', '.join(MODEL_ENTRY_KEYS)}")
        if not isinstance(item["id"], str) or not item["id"].strip():
            raise RuntimeError(f"{where}: id must be a model or inference profile ID")
        if isinstance(item["max_tokens"], bool) or not isinstance(item["max_tokens"], int):
            raise RuntimeError(f"{where}: max_tokens must be a positive integer")
        if not isinstance(item["thinking"], str) or not isinstance(item["text_release"], str):
            raise RuntimeError(f"{where}: thinking and text_release must be strings")
        if not isinstance(item["prompt_caching"], bool):
            raise RuntimeError(f"{where}: prompt_caching must be true or false")
        try:
            entries.append(ModelEntry(id=item["id"].strip(), thinking=item["thinking"].strip().lower(),
                                      max_tokens=item["max_tokens"],
                                      text_release=item["text_release"].strip().lower(),  # type: ignore[arg-type]
                                      prompt_caching=item["prompt_caching"]))
        except ValueError as exc:  # ModelEntry's rule, named by entry; its messages carry no value
            raise RuntimeError(f"{where}: {exc}") from None
    if len({e.id for e in entries}) != len(entries):
        raise RuntimeError(f"{MODELS_ENV} must not list a model twice")
    return tuple(entries)


def strands_settings_from_env(env: Mapping[str, str]) -> StrandsSettings:
    """:class:`StrandsSettings` from the bridge stack's environment (model failover). A missing or invalid
    variable raises ``RuntimeError`` naming it, never echoing its value
    (:func:`models_from_env` for the model list). ``OWLBANK_FAILOVER_TEST`` (optional, seconds) is
    refused at ``OWLBANK_STAGE=prod``. ``OWLBANK_PROMPT_CACHE_TTL``:
    absent or empty is ``5m``; ``5m`` or ``1h`` in any case."""
    from tac_bridge.agent_client_strands import PROMPT_CACHE_TTLS, StrandsSettings  # noqa: PLC0415
    from tac_bridge.guarded_model import TestDelayError, failover_test_delay  # noqa: PLC0415

    def required(name: str) -> str:
        value = (env.get(name) or "").strip()
        if not value:
            raise RuntimeError(f"{name} is required")
        return value

    models = models_from_env(env)
    gateway_url = required("OWLBANK_GATEWAY_URL")
    if not gateway_url.startswith("https://"):
        raise RuntimeError("OWLBANK_GATEWAY_URL must be an https:// URL")
    memory_id = required("OWLBANK_MEMORY_ID")
    region = required("AWS_REGION")
    stage = (env.get("OWLBANK_STAGE") or "").strip()
    try:
        test_delay = failover_test_delay(env, stage=stage)
    except TestDelayError as exc:  # names the variable and the rule, never the value
        raise RuntimeError(str(exc)) from None
    ttl = (env.get(PROMPT_CACHE_TTL_ENV) or "").strip().lower() or "5m"
    if ttl not in PROMPT_CACHE_TTLS:
        raise RuntimeError(f"{PROMPT_CACHE_TTL_ENV} must be 5m or 1h")
    return StrandsSettings(models=models, region=region, gateway_url=gateway_url, memory_id=memory_id,
                           stage=stage, failover_test_s=test_delay,
                           prompt_cache_ttl=ttl)  # type: ignore[arg-type]


def keypad_setting_from_env(env: Mapping[str, str]) -> bool:
    """AI keypad shortcuts 4.1: ``OWLBANK_AI_KEYPAD`` ``on`` or ``off`` (any case; absent or
    empty is ``on``, as on the front door). Anything else raises ``RuntimeError`` naming the
    variable, never echoing its value."""
    raw = (env.get(AI_KEYPAD_ENV) or "").strip().lower()
    if not raw:
        return True
    if raw not in _ON_OFF:
        raise RuntimeError(f"{AI_KEYPAD_ENV} must be one of: on, off")
    return _ON_OFF[raw]


def trait_settings_from_env(env: Mapping[str, str]) -> dict[str, Any]:
    """``BridgeConfig`` keyword arguments for the Memory trait settings.
    Unset or blank means the default. An invalid value raises ``RuntimeError`` naming the variable,
    never echoing its value."""
    out: dict[str, Any] = {}
    groups_raw = (env.get("OWLBANK_MEMORY_TRAIT_GROUPS") or "").strip()
    if groups_raw:
        names = tuple(part.strip() for part in groups_raw.split(","))
        if (not 1 <= len(names) <= MAX_TRAIT_GROUPS or len(set(names)) != len(names)
                or not all(TRAIT_GROUP_NAME_RE.fullmatch(n) for n in names)):
            raise RuntimeError(f"OWLBANK_MEMORY_TRAIT_GROUPS must list 1 to {MAX_TRAIT_GROUPS} distinct group "
                               f"names matching {TRAIT_GROUP_NAME_RE.pattern}")
        out["memory_trait_groups"] = names
    hours_raw = (env.get("OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS") or "").strip()
    if hours_raw:
        try:
            hours = int(hours_raw)
        except ValueError:
            hours = 0
        low, high = MAX_AGE_HOURS_RANGE
        if not low <= hours <= high or not hours_raw.isascii() or not hours_raw.lstrip("+").isdigit():
            raise RuntimeError(f"OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS must be an integer from {low} to {high}")
        out["web_activity_max_age_hours"] = hours
    return out


def build_strands_client(settings: StrandsSettings) -> StrandsAgentClient:
    """The Strands agent client, ready for calls: the Gateway MCP client started
    and listed once (a failure is logged and the Level-0 tools are left out until it recovers),
    and the Bedrock and Memory clients prewarmed. Blocking: run it off the event loop."""
    from tac_bridge.agent_client_strands import (  # noqa: PLC0415
        StrandsAgentClient,
        install_otel_detach_filter,
        make_gateway_tool_provider,
    )

    install_otel_detach_filter()  # barge-in closes Strands streams: drop OpenTelemetry's detach noise
    gateway = make_gateway_tool_provider(settings.gateway_url, settings.region)
    client = StrandsAgentClient(settings, gateway_tools=gateway)
    client.prewarm()
    return client


def _start_telemetry(env: Mapping[str, str]) -> tuple[Any, Any]:
    """Call observability's start-up: ``(handle, redactor)``, each ``None`` when not in use.

    With the switch off, :mod:`owlbank.telemetry.bridge` (and so our OpenTelemetry providers and
    exporters) is never imported. Never raises: a failure leaves the bridge untraced.
    """
    from owlbank.telemetry import TelemetrySettings  # noqa: PLC0415  (stdlib only)

    settings = TelemetrySettings.from_env(env, service="bridge")
    if settings is None:
        return None, None
    from owlbank.telemetry.bridge import TranscriptRedactor, install_bridge  # noqa: PLC0415

    redactor = TranscriptRedactor() if settings.content == "redacted" else None
    handle = install_bridge(settings, redact=redactor, environ=env)
    if handle is None:
        return None, None
    obs.emit(log, logging.INFO, "telemetry_started", trace_content=settings.content,
             sample_rate=settings.sample_rate)
    return handle, redactor


def build_app_from_env(env: Mapping[str, str] | None = None) -> FastAPI:
    """Production wiring from environment variables (see the module docs).

    Required: ``TWILIO_ACCOUNT_SID``, ``TWILIO_AUTH_TOKEN``, ``TWILIO_API_KEY``,
    ``TWILIO_API_SECRET``, ``TWILIO_PHONE_NUMBER`` (ECS task-definition secrets),
    ``TWILIO_CONVERSATION_CONFIGURATION_ID``, ``TWILIO_VOICE_PUBLIC_DOMAIN``,
    ``OWLBANK_BANK_TOOLS_FUNCTION``, ``OWLBANK_TOKEN_KMS_KEY`` (decrypt the front door's
    sealed token) and the agent settings (:func:`strands_settings_from_env`). Optional:
    ``OWLBANK_TABLE_*``, ``OWLBANK_HOSTING`` (:func:`tac_bridge.hosting.hosting_from_env`).
    With ``agentcore``, ``TWILIO_VOICE_PUBLIC_DOMAIN`` isn't required and the four Twilio values
    come from Secrets Manager (``OWLBANK_SECRET_TWILIO_API_KEY``, ``OWLBANK_SECRET_TWILIO_AUTH_TOKEN``:
    ARNs, read with the execution role) into ``os.environ`` before ``TACConfig.from_env()``.
    The PII guard uses Comprehend
    ``DetectPiiEntities`` in ``AWS_REGION`` (task role: ``comprehend:DetectPiiEntities``).
    """
    from tac import TAC, TACConfig  # noqa: PLC0415
    from tac.channels.voice import ConversationRelayProviderConfig, VoiceChannel  # noqa: PLC0415

    from owlbank.catalog import PromptCatalog  # noqa: PLC0415
    from owlbank.config import Settings  # noqa: PLC0415
    from owlbank.pii import ComprehendPiiRedactor, PiiGuard, make_comprehend_client  # noqa: PLC0415
    from owlbank.policy import Policy  # noqa: PLC0415
    from owlbank.segments import SegmentLog  # noqa: PLC0415
    from owlbank.store import DynamoStore  # noqa: PLC0415
    from owlbank.token_seal import KmsSealer  # noqa: PLC0415
    from tac_bridge.bridge import LambdaBackendClient, TACMemoryClient  # noqa: PLC0415

    obs.configure_logging("bridge")  # JSON lines, OWLBANK_LOG_LEVEL, scrubber; idempotent
    set_tac_conversation_polling()
    env = os.environ if env is None else env
    # Call observability: our providers exist before the Strands client starts, so its
    # tracer and meter find them. Off (the default): the telemetry module isn't even imported.
    telemetry, trace_redactor = _start_telemetry(env)
    hosting = hosting_from_env(env)
    required: tuple[str, ...] = ("TWILIO_CONVERSATION_CONFIGURATION_ID", "TWILIO_VOICE_PUBLIC_DOMAIN",
                "OWLBANK_BANK_TOOLS_FUNCTION", "OWLBANK_TOKEN_KMS_KEY")
    if hosting == "agentcore":  # no signature URL to rebuild
        required = tuple(n for n in required if n != "TWILIO_VOICE_PUBLIC_DOMAIN")
    for name in required:
        if not env.get(name):
            raise RuntimeError(f"{name} is required")
    strands_settings = strands_settings_from_env(env)
    ai_keypad = keypad_setting_from_env(env)
    if hosting == "agentcore":
        # Read by ARN at start-up, before anything else starts; TAC reads os.environ.
        # Values are never logged; a failure names the variable only.
        import boto3  # noqa: PLC0415

        os.environ.update(load_twilio_secrets(
            env, boto3.client("secretsmanager", region_name=strands_settings.region)))
    primary = strands_settings.primary
    from tac_bridge.failover_wiring import FailoverRuntime, log_models_resolved  # noqa: PLC0415

    log_models_resolved(strands_settings.model_ids,
                        fleet=len(strands_settings.models) > 1 and bool(env.get(FLEET_TABLE_ENV)))
    obs.emit(log, logging.INFO, "agent_settings", prompt_caching=primary.prompt_caching,
             prompt_cache_ttl=strands_settings.prompt_cache_ttl,
             thinking=primary.thinking, hosting=hosting, models=len(strands_settings.models))
    obs.emit(log, logging.INFO, "agent_text_release", mode=primary.text_release)
    obs.emit(log, logging.INFO, "ai_keypad", state="on" if ai_keypad else "off")
    # The Strands client's start-up (Gateway listing, client prewarm) overlaps TAC's configuration
    # fetch below; both finish before the app is returned, so before uvicorn listens.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-start")
    strands_start = pool.submit(build_strands_client, strands_settings)
    pool.shutdown(wait=False)

    import boto3  # noqa: PLC0415

    settings = Settings.from_env(env)
    sealer = KmsSealer(boto3.client("kms", region_name=settings.aws_region),
                       settings.token_kms_key or env["OWLBANK_TOKEN_KMS_KEY"])
    store = DynamoStore.from_settings(settings)
    tac_config = TACConfig.from_env()
    if not tac_config.conversation_configuration_id:
        raise RuntimeError("Orchestrator mode is required")
    tac = TAC(tac_config)  # network: GET the Conversation Configuration
    # TAC replaced the "tac" logger's handlers with its own and stopped propagation: undo both, so
    # its records are scrubbed JSON lines like ours. Its level (TWILIO_LOG_LEVEL) stays on the logger.
    obs.route_to_root(logging.getLogger("tac"))
    # Never more than one ACTIVE conversation for a call (see SingleConversationGuard).
    tac.conversation_orchestrator_client = SingleConversationGuard(  # type: ignore[assignment]
        tac.conversation_orchestrator_client)  # a duck-typed wrapper: TAC only calls its methods
    voice_channel = VoiceChannel(tac, config=ConversationRelayProviderConfig(memory_mode="never"))

    agent = strands_start.result()
    on_shutdown: list[Callable[[], Awaitable[Any]]] = [agent.aclose]
    policy = Policy.from_store(store, ttl_s=settings.cache_ttl_s)
    # The escalate tool's reason descriptions are the live policy's, read at
    # each call's agent build (through the policy cache), not the seed's defaults.
    agent.use_reason_descriptions(lambda: policy.reason_descriptions)
    # end_call is offered only while the policy switch is on (read per build).
    agent.use_end_call_switch(policy.end_call_enabled)
    # Model failover: the thresholds, window and deadlines, read per turn (policy cache).
    agent.use_failover_policy(lambda: policy.model_failover)
    # Model failover: the fleet picture and the metrics, only with the table
    # and two or more entries; otherwise None and today's single-model path.
    failover = FailoverRuntime.from_env(env, agent, hosting=hosting)
    on_startup: list[Callable[[], Awaitable[Any]]] = []
    on_call_end: list[Callable[[], Awaitable[Any]]] = []
    if failover is not None:
        on_startup.append(failover.start)
        on_shutdown.insert(0, failover.stop)  # the final flush before the agent closes
        if hosting == "agentcore":
            on_call_end.append(failover.call_ended)
    pii = PiiGuard.for_policy(ComprehendPiiRedactor(make_comprehend_client(settings.aws_region)), policy)
    if trace_redactor is not None:
        trace_redactor.attach(pii)  # traceContent=redacted: the transcripts' redaction
    if telemetry is not None:
        handle = telemetry

        async def stop_telemetry() -> None:  # flushes for at most 3 s
            await asyncio.to_thread(handle.shutdown)

        on_shutdown.append(stop_telemetry)
    bridge_config = BridgeConfig(
        reuse_conversation=env.get("OWLBANK_REUSE_CONVERSATION", "false").lower() == "true",
        timezone=env.get("OWLBANK_TIMEZONE") or "America/New_York",
        ai_keypad=ai_keypad,
        model_ids=strands_settings.model_ids,  # A prompt per entry
        **trait_settings_from_env(env))
    bridge = Bridge(
        store=store,
        agent=agent,
        backend=LambdaBackendClient(env["OWLBANK_BANK_TOOLS_FUNCTION"], region_name=settings.aws_region),
        memory=TACMemoryClient(tac, allow=bridge_config.memory_trait_groups,
                               max_age_hours=bridge_config.web_activity_max_age_hours, now=time.time),
        conversations=TACConversationCloser(tac),
        sealer=sealer,
        catalog=PromptCatalog.from_store(store, ttl_s=settings.cache_ttl_s),
        policy=policy,
        pii=pii,
        journal=SegmentLog(store, ttl_s=settings.record_ttl_s),
        config=bridge_config,
    )
    if telemetry is not None:
        provider = telemetry.tracer_provider
        # relay.session, bridge.turn and memory.reload spans. With redacted content the
        # redactor keeps a snapshot of each call's values for RETAIN_TTL_S after the forget, and
        # each session end flushes in the background.
        bridge.enable_tracing(telemetry.tracer,
                              flush=(lambda: provider.force_flush(3000)) if trace_redactor is not None else None,
                              retain=trace_redactor.retain if trace_redactor is not None else None)
    return build_app(tac, voice_channel, bridge, on_shutdown=on_shutdown, hosting=hosting,
                     on_startup=on_startup, on_call_end=on_call_end)
