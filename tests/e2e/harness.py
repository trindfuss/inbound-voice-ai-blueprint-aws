"""One in-memory OwlBank world for end-to-end tests (offline).

Everything real except the network edges:

* **Store:** one :class:`~owlbank.store.MemoryStore`, seeded with ``scripts/seed.py``'s
  own plan (``build_plan`` + ``apply_plan``): fixtures, policy table and prompt catalog.
  The front door and the bridge read the policy and catalog *from the store*.
* **Front door:** the real :class:`~lambdas.frontdoor.app.FrontDoor`, driven by signed fake
  Twilio webhooks (API Gateway proxy events) with Twilio's Basic challenge-response: every
  webhook is first sent without ``Authorization`` (401 with ``WWW-Authenticate``), then repeated
  with the credentials (``tests.component.test_frontdoor_support.twilio_post``).
* **Back end:** the real ``auth_service`` and ``bank_tools`` behind one fake Lambda
  client, used by the front door and the bridge (``lambda:InvokeFunction`` wire shapes).
* **Bridge:** the real :class:`~tac_bridge.bridge.Bridge` on real TAC v2.5.0 (fakes for
  Orchestrator and Memory), driven by a ConversationRelay simulator.
* **Agent:** :class:`ScriptedAgent`'s scripts played through the real Strands client
  (``tests.strands_parity``); level-0 tools a step calls run through :class:`FakeGateway` (the
  real ``level0_tools`` handler).
* **KMS:** one shared :class:`~owlbank.token_seal.FakeSealer`.

:class:`Call` follows Twilio's behaviour: POST ``incoming``; answer each ``<Gather>``
(digits or speech) by POSTing its ``action``; on ``<Connect><ConversationRelay>`` open
the bridge WebSocket (``setup``, ``prompt``...); on the bridge's ``end`` POST the
``<Connect action>`` with ``HandoffData``; follow redirects and reconnects until
``<Enqueue>`` or hang-up; then POST the status callback ``completed``.

:class:`World` records every surface a secret could leak to (TwiML, WebSocket frames,
logs, store, model inputs, task attributes, Context API responses) for
:meth:`World.assert_no_leaks`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

from tac.channels.voice import ConversationRelayProviderConfig, VoiceChannel

from lambdas.auth_service import service as auth_service
from lambdas.bank_tools import service as bank_service
from lambdas.context_api import handler as context_api
from lambdas.frontdoor import http as frontdoor_http
from lambdas.frontdoor import state as sessions
from lambdas.frontdoor.app import Deps, FrontDoor, FrontDoorConfig
from lambdas.frontdoor.backend import LambdaBackendClient as FrontDoorBackend
from lambdas.frontdoor.fakes import FakeLambdaClient
from lambdas.level0_tools import handler as level0_handler
from owlbank.basic_auth import BasicSecret
from owlbank import callstate, obs
from owlbank.catalog import PromptCatalog
from owlbank.envelope import ResultEnvelope
from owlbank.pii import FakePiiRedactor, PiiGuard
from owlbank.policy import Policy
from owlbank.segments import SegmentLog, clear_sensitive
from owlbank.slots import FakeResolver
from owlbank.store import MemoryStore
from owlbank.token_seal import FakeSealer
from scripts import seed
from tac_bridge.agent_client import AgentError, TextDelta, ToolCall, ToolResult, TurnEnd, join_system
from tac_bridge.app import PeekingWebSocket
from tac_bridge.bridge import Bridge, BridgeConfig, TACMemoryClient
from tac_bridge.bridge import LambdaBackendClient as BridgeBackend

from tests.context_api_fakes import TASK_SID, PermissiveTaskRouter
from tests.component.test_bridge_fakes import _CLOSE, SimSocket, make_tac, wait_for
from tests.strands_parity import PerInputScript, StrandsDriven, memory_for, plain_fake, preferred_prompt
from tests.component.test_frontdoor_support import (
    AUTH_TOKEN,
    BASIC,
    OTP,
    TEST_FACTORS,
    WORKFLOW,
    FakeVerify,
    RecordingCloser,
    twilio_post,
)
from tests.leakscan import compile_values, leaks

DATA = Path(__file__).resolve().parents[2] / "data"
T0 = 1_760_000_000.0
RELAY_URL = "wss://bridge.owlbank.example/ws"
CONV_CONFIG = "conv_configuration_test"
OWLBANK_NUMBER = "+12025550100"
AUTH_FN, BANK_FN = "owlbank-auth", "owlbank-bank"
TRUSTED_VERSTAT = "TN-Validation-Passed-A"
CONFIRMATION = "80706050"  # fixed payment confirmation (random digits could collide with factors)

THOMAS, LEES, MARIA, NOBODY = "+12025550101", "+12025550102", "+12025550103", "+12025550199"

#: Store attributes holding server-generated IDs, hashes or times: not scanned for factor
#: values (random digits there could collide; ``factors`` are salted hashes;
#: ``agent_owner`` is the bridge's random hex ``instance_id``).
_SKIP_KEYS = frozenset({"token_hash", "claimed_at", "ended_at", "at", "issued_at", "factors",
                        "seq", "ts", "expires_at", "version", "tool_use_id", "call_sid",
                        "salt", "sealed_token", "held", "ws_nonce_hash", "agent_owner"})
_STD_LOG_ATTRS = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {"message", "asctime"}


def call_sid(n: int) -> str:
    """A valid ``CA`` + 32 hex CallSid with no digits (so it can't collide with a factor)."""
    hexl = "abcdef"
    return "CA" + "e" * 29 + hexl[n // 36 % 6] + hexl[n // 6 % 6] + hexl[n % 6]


def factor(customer_id: str, name: str) -> str:
    return TEST_FACTORS[customer_id][name]


class RecordingSealer(FakeSealer):
    """The shared fake KMS; remembers every sealed value (tokens and held factors) for the scan."""

    def __init__(self) -> None:
        super().__init__()
        self.sealed: list[str] = []

    def seal(self, token: str, *, call_sid: str) -> str:
        out = super().seal(token, call_sid=call_sid)
        self.sealed.append(out)
        return out


# --------------------------------------------------------------------------- clock


class Clock:
    """One shared, settable clock (epoch seconds) for every component."""

    def __init__(self, t: float = T0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


# --------------------------------------------------------------------------- the agent


Step = list[Any] | Callable[[dict[str, Any]], list[Any]] | Exception


def say(*parts: str) -> list[Any]:
    return [*(TextDelta(p) for p in parts), TurnEnd("end_turn")]


def tool(name: str, tool_use_id: str, **args: Any) -> list[Any]:
    return [ToolCall(tool_use_id, name, dict(args)), TurnEnd("tool_use")]


def result_then_words(agent: ScriptedAgent, words: str, on_words: Step,
                      on_result: Callable[[dict[str, Any]], None] | None = None) -> list[Step]:
    """A pending result that comes with the caller's words is two agent inputs in one turn:
    the result (``next`` = ``await_caller``, no ``callerUtterance``), recorded with no model call
    (its step only checks the envelope, with ``on_result``), then ``words`` as the caller's
    text, answered by ``on_words``."""

    def result(rec: dict[str, Any]) -> list[Any]:
        env = agent.envelope(rec)
        assert env["next"] == "await_caller" and "callerUtterance" not in env
        if on_result is not None:
            on_result(env)
        return say("<silent/>")

    def caller(rec: dict[str, Any]) -> Any:
        assert rec["user_text"] == words and rec["tool_results"] == []
        return on_words(rec) if callable(on_words) and not isinstance(on_words, list) else on_words

    return [result, caller]


class FakeGateway:
    """AgentCore Gateway with the real ``level0_tools`` Lambda target (the Level-0 tools)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.results: list[dict[str, Any]] = []

    def call(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        ctx = SimpleNamespace(client_context=SimpleNamespace(
            custom={level0_handler.TOOL_NAME_KEY: f"level0{level0_handler.DELIMITER}{name}"}))
        result = level0_handler.handler(dict(args), ctx)
        self.calls.append((name, dict(args)))
        self.results.append(result)
        return result


@dataclass
class ScriptedAgent(PerInputScript):
    """A scripted ``AgentClient`` (one step per agent input; a recorded ``await_caller`` result's
    step is never asked for).

    A step is a list of events, an exception, or a callable that gets the turn's inputs
    (``user_text``, ``tool_results``, ``system_prompts``) and returns the events, so a
    step can check what the model was given and call Gateway tools.
    """

    gateway: FakeGateway
    script: list[Step] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    unexpected: int = 0
    sessions: list[str] = field(default_factory=list)

    histories: list[str] = field(default_factory=list)
    #: Segment-traits: ``(session_id, user_text, assistant_text)`` per ``record_exchange``.
    exchanges: list[tuple[str, str, str]] = field(default_factory=list)

    async def begin_session(self, session_id: str, *, history: str = "reload") -> None:
        self.sessions.append(session_id)
        self.histories.append(history)  # The bridge's session-ownership decision

    async def record_exchange(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """Recorded with no model call; no script step is consumed."""
        self.exchanges.append((session_id, user_text, assistant_text))

    def queue(self, *steps: Step) -> None:
        self.script.extend(steps)

    def _record(self, session_id: str, user_text: str | None, tool_results: Sequence[ToolResult],
                system_prompts: Mapping[str, str] | None, system_suffix: str | None) -> dict[str, Any]:
        # What the model sees (the two parts joined), and the suffix on its own.
        rec = {"session_id": session_id, "user_text": user_text, "tool_results": list(tool_results),
               "system_prompt": join_system(preferred_prompt(system_prompts), system_suffix),
               "system_prompts": system_prompts, "system_suffix": system_suffix}
        self.calls.append(rec)
        return rec

    def _next_step(self, rec: dict[str, Any]) -> Step:
        if self.script:
            return self.script.pop(0)
        self.unexpected += 1
        return say("Unscripted turn.")

    def turn(self, session_id: str, *, user_text: str | None, tool_results: Sequence[ToolResult],
             system_prompts: Mapping[str, str] | None, system_suffix: str | None = None,
             turn_id: int | None = None) -> Any:
        if tool_results and user_text:  # the results are recorded, the words answered
            self._skip_record_step(session_id, tool_results, system_prompts, system_suffix)
            tool_results = ()
        rec = self._record(session_id, user_text, tool_results, system_prompts, system_suffix)
        return self._run(self._next_step(rec), rec)

    def record_results(self, session_id: str, tool_results: Sequence[ToolResult], *,
                       system_prompts: Mapping[str, str] | None, system_suffix: str | None = None,
                       turn_id: int | None = None) -> Any:
        """Recorded with no model call: the step is skipped, the stream is ``TurnEnd``."""
        self._skip_record_step(session_id, tool_results, system_prompts, system_suffix)
        return self._run([TurnEnd("end_turn")], {})

    async def _run(self, step: Step, rec: dict[str, Any]) -> Any:
        if callable(step) and not isinstance(step, list):
            step = step(rec)
        if isinstance(step, Exception):
            raise step
        for ev in step:
            yield ev

    def inputs_text(self) -> str:
        """Everything the model was ever given (system prompt, messages, tool results)."""
        return json.dumps([{**c, "tool_results": [r.content for r in c["tool_results"]]}
                           for c in self.calls] + self.gateway.results, default=str)

    def envelope(self, rec: dict[str, Any]) -> dict[str, Any]:
        (res,) = rec["tool_results"]
        assert isinstance(res.content, dict)
        return res.content


@dataclass
class StrandsScriptedAgent(StrandsDriven, ScriptedAgent):
    """``ScriptedAgent``'s scripts through a real ``StrandsAgentClient`` (``tests.strands_parity``);
    the model input it reports includes what the client sent the model."""

    memory: Any = None

    def __post_init__(self) -> None:
        self._attach(self.memory)

    def inputs_text(self) -> str:
        return ScriptedAgent.inputs_text(self) + self.model_inputs_text()


def new_scripted_agent(gateway: FakeGateway, store: Any) -> ScriptedAgent:
    """The world's scripted agent (``tests.strands_parity``): through the real Strands client; the
    plain ``ScriptedAgent`` for a ``protocol_fake`` test."""
    if plain_fake():
        return ScriptedAgent(gateway)
    return StrandsScriptedAgent(gateway, memory=memory_for(store))


# --------------------------------------------------------------------------- ConversationRelay


class RelaySession:
    """One ConversationRelay WebSocket (Twilio's side), into TAC through ``PeekingWebSocket``."""

    def __init__(self, world: World, sid: str, ani: str, params: Mapping[str, str],
                 nonce: str | None = None) -> None:
        self.world = world
        self.sid = sid
        self.ani = ani
        self.params = dict(params)
        self.nonce = nonce
        self.socket = SimSocket()
        self.peek = PeekingWebSocket(self.socket, world.bridge, nonce=nonce)
        self.task: asyncio.Task[None] | None = None
        self._mark = 0

    async def connect(self) -> list[dict[str, Any]]:
        voice = self.world.voice
        self.task = asyncio.create_task(self._serve(voice))
        await self.socket.inbound.put({
            "type": "setup", "sessionId": "VX" + "c" * 32, "accountSid": "AC" + "b" * 32,
            "callSid": self.sid, "from": self.ani, "to": OWLBANK_NUMBER, "direction": "inbound",
            "callType": "PSTN", "callStatus": "IN-PROGRESS", "customParameters": self.params})
        await wait_for(lambda: voice.get_conversation_session_by_call_sid(self.sid) is not None)
        await self.settle()
        return self.frames()

    async def _serve(self, voice: VoiceChannel) -> None:
        """What ``tac_bridge.app``'s ``/ws`` endpoint does around the socket: the session start and end lines."""
        try:
            await voice.handle_websocket(self.peek)
        finally:
            if self.peek.verified and self.peek.call_sid:
                self.world.bridge.session_end(self.peek.call_sid, "closed")

    async def settle(self) -> None:
        """Let ``on_setup`` and any resume it started run to completion."""
        for _ in range(3):
            for _ in range(20):
                await asyncio.sleep(0)
            if self.peek._tasks:  # noqa: SLF001 - test introspection
                await asyncio.gather(*list(self.peek._tasks), return_exceptions=True)  # noqa: SLF001
            await self.world.bridge.wait_idle()

    def frames(self) -> list[dict[str, Any]]:
        out = self.socket.sent[self._mark:]
        self._mark = len(self.socket.sent)
        return out

    async def prompt(self, text: str, lang: str = "en-US") -> list[dict[str, Any]]:
        start = len(self.socket.sent)
        self.world.said.append(text)
        await self.socket.inbound.put({"type": "prompt", "voicePrompt": text, "lang": lang, "last": True})
        await wait_for(lambda: any(f.get("type") == "text" and f.get("last")
                                   for f in self.socket.sent[start:]), 5.0)
        await self.settle()
        return self.frames()

    async def close(self) -> None:
        if self.socket.closed and self.task is not None and self.task.done():
            return
        await self.socket.inbound.put(_CLOSE)
        if self.task is not None:
            await asyncio.wait_for(self.task, 5)


def spoken(frames: Iterable[Mapping[str, Any]]) -> str:
    return "".join(f.get("token", "") for f in frames if f.get("type") == "text")


# --------------------------------------------------------------------------- the call driver


class Call:
    """One phone call, driven the way Twilio drives the webhooks and the WebSocket."""

    def __init__(self, world: World, sid: str, ani: str) -> None:
        self.world = world
        self.sid = sid
        self.ani = ani
        self.state = "new"  # new | gather | relay | queued | ended
        self.gather: ET.Element | None = None
        self.relay: RelaySession | None = None
        self.connect_action: str | None = None
        self.relay_opens = 0
        self.heard: list[tuple[str, str]] = []  # (source, text): say | welcome | ai
        self.handoffs: list[dict[str, Any]] = []
        self.task: dict[str, Any] | None = None
        self.enqueue: ET.Element | None = None
        self.sent_digits: list[str] = []

    # -- what the caller heard
    def texts(self, source: str | None = None) -> list[str]:
        return [t for s, t in self.heard if source is None or s == source]

    def mark(self) -> int:
        return len(self.heard)

    def since(self, mark: int, source: str | None = None) -> list[str]:
        return [t for s, t in self.heard[mark:] if source is None or s == source]

    # -- webhooks
    def _base(self) -> dict[str, str]:
        return {"CallSid": self.sid, "AccountSid": "AC" + "b" * 32, "From": self.ani or "anonymous",
                "To": OWLBANK_NUMBER, "Direction": "inbound", "ApiVersion": "2010-04-01"}

    def _post(self, url: str, params: Mapping[str, str]) -> str:
        path = urlparse(url).path
        assert "/voice/" in path, url
        route = path.rsplit("/voice/", 1)[1]
        body = {**self._base(), **params}
        resp = twilio_post(self.world.frontdoor_handler, route, body, form=self.world.url_form)
        self.world.sids.add(self.sid)
        assert resp["statusCode"] == 200, resp
        self.world.twiml.append(resp["body"])
        return resp["body"]

    async def start(self, stir_verstat: str | None = TRUSTED_VERSTAT) -> None:
        """``stir_verstat=None`` leaves ``StirVerstat`` out (no SHAKEN PASSporT: untrusted)."""
        params = {"CallStatus": "ringing"}
        if stir_verstat is not None:
            params["StirVerstat"] = stir_verstat
        await self._follow(self._post("/prod/voice/incoming", params))

    async def press(self, digits: str) -> None:
        """Answer the current ``<Gather>`` with keypad digits (``""`` = timeout, no input)."""
        assert self.state == "gather" and self.gather is not None, self.state
        if digits:
            self.sent_digits.append(digits)
            self.world.sent_factors.add(digits)
        params = {"CallStatus": "in-progress"}
        if digits:
            params["Digits"] = digits
        await self._follow(self._post(self.gather.get("action") or "", params))

    async def speak_to_gather(self, speech: str, confidence: str = "0.92") -> None:
        assert self.state == "gather" and self.gather is not None, self.state
        assert "speech" in (self.gather.get("input") or ""), "this <Gather> doesn't take speech"
        await self._follow(self._post(self.gather.get("action") or "",
                                      {"CallStatus": "in-progress", "SpeechResult": speech,
                                       "Confidence": confidence}))

    async def say(self, text: str, lang: str = "en-US") -> str:
        """Say something to the AI; returns what the AI said back (streamed text)."""
        assert self.state == "relay" and self.relay is not None, self.state
        frames = await self.relay.prompt(text, lang)
        reply = spoken(frames)
        if reply:
            self.heard.append(("ai", reply))
            self.world.said.append(reply)
        await self._after_frames(frames)
        return reply

    async def hangup(self, duration: str = "95") -> None:
        if self.relay is not None:
            await self.relay.close()
            self.relay = None
        self.state = "ended"
        self._post("/prod/voice/status", {"CallStatus": "completed", "CallDuration": duration})

    # -- following TwiML
    async def _follow(self, body: str) -> None:
        root = ET.fromstring(body.encode())
        for el in root:
            if el.tag == "Say":
                self.heard.append(("say", el.text or ""))
            elif el.tag == "Play":
                self.heard.append(("play", el.text or ""))
            elif el.tag == "Gather":
                for child in el:
                    if child.tag == "Say":
                        self.heard.append(("say", child.text or ""))
                self.gather, self.state = el, "gather"
                return  # actionOnEmptyResult: a timeout posts the action, so <Redirect> isn't reached
            elif el.tag == "Redirect":
                await self._follow(self._post(el.text or "", {"CallStatus": "in-progress"}))
                return
            elif el.tag == "Connect":
                cr = el.find("ConversationRelay")
                assert cr is not None, body
                self.connect_action = el.get("action")
                await self._open_relay(cr)
                return
            elif el.tag == "Enqueue":
                task = el.find("Task")
                assert task is not None and task.text
                self.enqueue, self.task, self.state = el, json.loads(task.text), "queued"
                self.world.task_attributes.append(self.task)
                return
            elif el.tag == "Hangup":
                self.state = "ended"
                return
        self.state = "ended"  # end of document: Twilio hangs up

    async def _open_relay(self, cr: ET.Element) -> None:
        url = urlparse(cr.get("url") or "")
        assert url._replace(query="").geturl() == RELAY_URL
        (nonce,) = parse_qs(url.query)["n"]  # one-time relay nonce
        self.world.relay_nonces.append(nonce)
        assert cr.get("welcomeGreeting") is not None, "welcomeGreeting must always be set"
        welcome = cr.get("welcomeGreeting") or ""
        if welcome:
            self.heard.append(("welcome", welcome))
        params = {p.get("name") or "": p.get("value") or "" for p in cr.findall("Parameter")}
        self.relay = RelaySession(self.world, self.sid, self.ani, params, nonce)
        self.world.relays.append(self.relay)
        self.relay_opens += 1
        self.gather, self.state = None, "relay"
        frames = await self.relay.connect()
        reply = spoken(frames)
        if reply:
            self.heard.append(("ai", reply))
        await self._after_frames(frames)

    async def _after_frames(self, frames: list[dict[str, Any]]) -> None:
        ends = [f for f in frames if f.get("type") == "end"]
        if not ends:
            return
        assert len(ends) == 1
        raw = ends[0]["handoffData"]
        self.handoffs.append(json.loads(raw))
        assert self.relay is not None
        await self.relay.close()  # ConversationRelay closes the socket after `end`
        self.relay = None
        body = self._post(self.connect_action or "", {
            "CallStatus": "in-progress", "SessionStatus": "ended", "SessionId": "VX" + "c" * 32,
            "SessionDuration": "30", "HandoffData": raw})
        await self._follow(body)


# --------------------------------------------------------------------------- the world


class World:
    def __init__(self, *, resolver: FakeResolver | None = None,
                 config: Mapping[str, Any] | None = None, bridge_config: BridgeConfig | None = None,
                 policy_doc: Mapping[str, Any] | None = None, flex: bool = True,
                 briefing_model: Any = None) -> None:
        self.clock = Clock()
        self.store = MemoryStore()
        plan = seed.build_plan(seed.load_json(DATA / "fixtures.json"),
                               policy_doc or seed.load_json(DATA / "policy.json"),
                               seed.load_json(DATA / "prompt_catalog.json"), stage="test")
        self.seed_counts = seed.apply_plan(plan, self.store)
        self.sealer = RecordingSealer()
        self.verify = FakeVerify()
        self.closer = RecordingCloser()
        self.resolver = resolver or FakeResolver()
        self.raw_tokens: list[str] = []
        self.sent_factors: set[str] = set()
        self.twiml: list[str] = []
        self.task_attributes: list[dict[str, Any]] = []
        self.context_responses: list[dict[str, Any]] = []
        self.relays: list[RelaySession] = []
        self.relay_nonces: list[str] = []
        self.pii = FakePiiRedactor()  # Comprehend stand-in, shared by the front door and the bridge
        self.url_form = "plain"  # URL form the fake Twilio signs: "plain" or "credentials"

        now_int = lambda: int(self.clock())  # noqa: E731
        policy = Policy.from_store(self.store, ttl_s=60, clock=self.clock)
        self.auth_deps = auth_service.AuthDeps(store=self.store, policy=policy, verify_client=self.verify,
                                               clock=now_int)
        self.bank_deps = bank_service.BankDeps(store=self.store, policy=policy, clock=now_int,
                                               confirmation=lambda: CONFIRMATION)
        self.sids: set[str] = set()
        self.said: list[str] = []  # what the caller and the AI said: never in a log line
        # The three Lambda entry points run through ``obs.instrumented``, as in production, so the
        # call SID, component and request ID are bound on every line they cause.
        self._auth_entry = obs.instrumented("auth_service")(
            lambda event, context=None: auth_service.handle_event(self.auth_deps, event))
        self._bank_entry = obs.instrumented("bank_tools")(
            lambda event, context=None: bank_service.handle_event(self.bank_deps, event))
        self.lam = FakeLambdaClient({AUTH_FN: self._auth, BANK_FN: self._bank})

        catalog = PromptCatalog.from_store(self.store, ttl_s=60, clock=self.clock)
        cfg = {"relay_ws_url": RELAY_URL, "conversation_configuration": CONV_CONFIG,
               "workflow_sid": WORKFLOW if flex else None, "test_traffic": True, **(config or {})}
        self.config = FrontDoorConfig(**cfg)
        self.frontdoor = FrontDoor(Deps(
            store=self.store, catalog=catalog, policy=policy,
            backend=FrontDoorBackend(self.lam, auth_function=AUTH_FN, bank_function=BANK_FN),
            auth_token=lambda: AUTH_TOKEN, basic_credentials=lambda: BASIC,
                basic_secret=lambda force=False: BasicSecret(*BASIC), sealer=self.sealer,
            pii=PiiGuard.for_policy(self.pii, policy), resolver=self.resolver, clock=self.clock, closer=self.closer,
            briefing_model=briefing_model), self.config)

        self.frontdoor_handler = obs.instrumented("frontdoor", op=lambda e, _c: frontdoor_http.route_name(e))(
            lambda event, context=None: self.frontdoor.handle(event))
        self.gateway = FakeGateway()
        self.agent = new_scripted_agent(self.gateway, self.store)
        self.tac = make_tac()
        self.voice = VoiceChannel(self.tac, config=ConversationRelayProviderConfig(memory_mode="never"))
        bridge_policy = Policy.from_store(self.store, ttl_s=60, clock=self.clock)
        bridge_config = bridge_config or BridgeConfig()
        self.bridge = Bridge(
            store=self.store, agent=self.agent, pii=PiiGuard.for_policy(self.pii, bridge_policy),
            backend=BridgeBackend(BANK_FN, client=self.lam),
            memory=TACMemoryClient(self.tac, allow=bridge_config.memory_trait_groups,
                                   max_age_hours=bridge_config.web_activity_max_age_hours, now=self.clock),
            catalog=PromptCatalog.from_store(self.store, ttl_s=60, clock=self.clock),
            policy=bridge_policy,
            sealer=self.sealer, journal=SegmentLog(self.store, now=self.clock),
            config=bridge_config, now=self.clock, io=_inline_io)
        self.tac.on_message_ready(self.bridge.on_message_ready)
        self.tac.on_interrupt(self.bridge.on_interrupt)
        self.bridge.voice = self.voice

    # -- back end (real services; the token each verify issues is recorded for the leak scan)
    def _auth(self, event: dict[str, Any]) -> Any:
        result = self._auth_entry(event, SimpleNamespace(aws_request_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"))
        if isinstance(result, dict) and isinstance(result.get("token"), str):
            self.raw_tokens.append(result["token"])
        return result

    def _bank(self, event: dict[str, Any]) -> Any:
        return self._bank_entry(event, SimpleNamespace(aws_request_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"))

    def bank_api(self, op: str, *, token: str | None, sid: str, args: Mapping[str, Any],
                 **extra: Any) -> dict[str, Any]:
        """Call ``bank_tools`` directly, as anything holding the IAM permission could."""
        return bank_service.handle_event(self.bank_deps, {"op": op, "token": token, "call_sid": sid,
                                                          "args": dict(args), **extra})

    # -- calls
    def dial(self, ani: str, sid: str) -> Call:
        return Call(self, sid, ani)

    def state(self, sid: str) -> callstate.CallState:
        st = callstate.load(self.store, sid)
        assert st is not None
        return st

    def session(self, sid: str) -> sessions.Session | None:
        return sessions.load(self.store, sid)

    def journal(self, sid: str) -> list[dict[str, Any]]:
        return SegmentLog(self.store).read(sid)

    def prompt_keys(self, sid: str) -> list[str]:
        return [e["prompt_key"] for e in self.journal(sid) if e["type"] == "prompt" and e.get("prompt_key")]

    def envelope(self, sid: str) -> ResultEnvelope | None:
        raw = self.state(sid).pending_result
        return ResultEnvelope.from_dict(dict(raw)) if raw else None

    def customer(self, customer_id: str) -> dict[str, Any]:
        item = self.store.get("customers", {"customer_id": customer_id})
        assert item is not None
        return item

    def context(self, journal_ref: str, sid: str, *, worker: str | None = "WK" + "a" * 32,
                origin: str = context_api.ALLOWED_ORIGIN) -> dict[str, Any]:
        """``GET /context/{journal_ref}?call_sid=`` as the Flex plugin calls it (after the authorizer)."""
        event = {"httpMethod": "GET", "headers": {"Origin": origin},
                 "pathParameters": {"journal_ref": journal_ref},
                 "queryStringParameters": {"call_sid": sid, "task_sid": TASK_SID},
                 "requestContext": {"authorizer": {"worker_sid": worker, "roles": "supervisor"}
                                    if worker else {}}}
        resp = context_api.get_context(event, self.store, now=self.clock,
                                       taskrouter=PermissiveTaskRouter(sid, journal_ref))
        self.context_responses.append(resp)
        return resp

    # -- the security scan
    def factor_values(self) -> set[str]:
        values = {v for f in TEST_FACTORS.values() for v in f.values()} | {OTP} | self.sent_factors
        return {v for v in values if len(v) >= 4}

    def nonsensitive_slots(self) -> frozenset[str]:
        """The slots whose value a ``slot_resolution`` event or a ``factor_answer`` line may carry:
        the ones not flagged sensitive."""
        policy = self.frontdoor.d.policy
        return frozenset(n for n in policy.slots() if not policy.slot(n).sensitive)

    def surfaces(self, records: Sequence[logging.LogRecord]) -> dict[str, list[str]]:
        logs: list[str] = []
        open_slots = self.nonsensitive_slots()
        for r in records:
            logs.append(r.getMessage())
            # A non-sensitive slot's value may be in its own factor_answer line's ``value``.
            sanctioned = getattr(r, "event", None) == "factor_answer" and getattr(r, "factor", None) in open_slots
            logs.extend(str(v) for k, v in r.__dict__.items()
                        if k not in _STD_LOG_ATTRS and not (sanctioned and k == "value"))
        frames = [json.dumps(f, ensure_ascii=False) for s in self.relays for f in s.socket.sent]
        return {
            "twiml": list(self.twiml),
            "relay frames (transcript, handoffData)": frames,
            "log": logs,
            "store": self.stored_strings(),
            "model input": [self.agent.inputs_text()],
            "task attributes": [json.dumps(t, ensure_ascii=False) for t in self.task_attributes],
            "context api": [r.get("body") or "" for r in self.context_responses],
        }

    def stored_strings(self, *, skip: frozenset[str] = _SKIP_KEYS) -> list[str]:
        out: list[str] = []
        open_slots = self.nonsensitive_slots()

        def walk(v: Any) -> None:
            if isinstance(v, dict):
                # A non-sensitive slot's value is journaled in its slot_resolution event only.
                sanctioned = v.get("type") == "slot_resolution" and v.get("slot") in open_slots
                for k, x in v.items():
                    if k not in skip and not (sanctioned and k == "value"):
                        out.append(str(k))
                        walk(x)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    walk(x)
            elif v is not None:
                out.append(str(v))

        for table in self.store._tables.values():  # noqa: SLF001 - every table, every item
            for item in table.values():
                walk(item)
        return out

    def assert_no_leaks(self, records: Sequence[logging.LogRecord]) -> None:
        """No plaintext factor value, test OTP, raw token, or sealed value (outside the store)
        on any surface."""
        factors = compile_values(self.factor_values())  # whole values (tests.leakscan)
        surfaces = self.surfaces(records)
        for where, texts in surfaces.items():
            for text in texts:
                assert not leaks(factors, text), f"a factor value leaked into {where}"
        sealed = self.sealed_values()
        everything = {**surfaces, "store (all attributes)": self.stored_strings(skip=frozenset())}
        for where, texts in everything.items():
            for text in texts:
                for tok in self.raw_tokens:
                    assert tok not in text, f"a raw call token leaked into {where}"
                if where.startswith("store"):
                    continue  # sealed values live (only) in call state and the sidecar
                for s in sealed:
                    assert s not in text, f"a sealed token leaked into {where}"

    def log_texts(self, records: Sequence[logging.LogRecord]) -> list[str]:
        out: list[str] = []
        for r in records:
            out.append(r.getMessage())
            out.extend(str(v) for k, v in r.__dict__.items() if k not in _STD_LOG_ATTRS)
        return out

    def assert_logs_clean(self, records: Sequence[logging.LogRecord]) -> None:
        """Leak scan over every record of the run (DEBUG included): no full phone number, no
        confirmation code, no caller or AI message text."""
        joined = "\n".join(self.log_texts(records))
        phones = re.findall(r"\+\d{7,}", joined)
        assert not phones, f"a full phone number is in the logs: {phones[:3]}"
        assert CONFIRMATION not in joined, "a payment confirmation code is in the logs"
        folded = joined.casefold()
        for said in self.said:
            if len(said.strip()) >= 10:
                assert said.strip().casefold() not in folded, "message text is in the logs"

    def assert_correlated(self, records: Sequence[logging.LogRecord]) -> None:
        """Each call's SID is on records from the front door, from the auth service when it was
        called for that call, and from the bridge when a relay session ran for it."""
        by_sid: dict[str, set[str]] = {}
        for r in records:
            sid, comp = getattr(r, "call_sid", None), getattr(r, "component", None)
            if sid and comp:
                by_sid.setdefault(sid, set()).add(comp)
        auth_sids = {p.get("call_sid") for (fn, op), p in zip(self.lam.calls, self.lam.payloads, strict=True)
                     if fn == AUTH_FN}
        relay_sids = {s.sid for s in self.relays}
        for sid in self.sids:
            comps = by_sid.get(sid, set())
            assert "frontdoor" in comps, f"{sid}: no front door record carries the call SID ({comps})"
            if sid in auth_sids:
                assert "auth_service" in comps, f"{sid}: no auth service record carries the call SID ({comps})"
            if sid in relay_sids:
                assert "bridge" in comps, f"{sid}: no bridge record carries the call SID ({comps})"

    def sealed_values(self) -> set[str]:
        return set(self.sealer.sealed)

    def cleanup(self) -> None:
        for key in list(self.store._tables["calls"]):  # noqa: SLF001
            clear_sensitive(str(key[0]).split("#")[0])


async def _inline_io(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Store I/O inline (``MemoryStore`` isn't thread-safe; production uses a thread)."""
    return fn(*args, **kwargs)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


__all__ = [
    "AgentError", "CONFIRMATION", "Call", "Clock", "FakeGateway", "LEES", "MARIA", "NOBODY", "OTP",
    "RelaySession", "ScriptedAgent", "THOMAS", "TRUSTED_VERSTAT", "World", "call_sid", "factor", "run", "say",
    "spoken", "tool",
]
