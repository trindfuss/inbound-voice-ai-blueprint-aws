"""Fakes and a ConversationRelay simulator for the TAC bridge tests.

Everything here is offline:

* **Real TAC v2.5.0** in Orchestrator mode. The constructor's configuration fetch
  is patched, and the Conversation Orchestrator and Memory clients are replaced
  with in-process fakes (:class:`FakeOrchestrator`, :class:`FakeTwilioMemory`).
  ``TAC_ANALYTICS_DISABLED=true``.
* :class:`SimSocket` satisfies TAC's ``WebSocketProtocol``; :class:`CRSimulator`
  speaks the ConversationRelay protocol (``setup``, ``prompt``, ``interrupt``,
  hang-up) to ``VoiceChannel.handle_websocket`` through our ``PeekingWebSocket``.
* :class:`FakeAgent` scripts the agent (rigs play its scripts through the real Strands client,
  :func:`new_agent`); :class:`FakeBackend` is ``bank_tools``;
  :class:`FakeMemory` is the bridge's ``MemoryClient`` wrapper.

This module also holds a few self-tests of the fakes.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

os.environ["TAC_ANALYTICS_DISABLED"] = "true"

import httpx  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("tac")

from tac import TAC, TACConfig  # noqa: E402
from tac.channels.voice import ConversationRelayProviderConfig, VoiceChannel  # noqa: E402
from tac.channels.websocket_protocol import WebSocketDisconnectError  # noqa: E402
from tac.models.conversation import (  # noqa: E402
    ConversationResponse,
    ParticipantAddress,
    ParticipantResponse,
)
from tac.models.memory import (  # noqa: E402
    MemoryRetrievalResponse,
    ObservationInfo,
    ProfileLookupResponse,
    ProfileResponse,
)

from owlbank import callstate as cs  # noqa: E402
from owlbank import ws_nonce  # noqa: E402
from owlbank.catalog import PromptCatalog  # noqa: E402
from owlbank.pii import FakePiiRedactor, PiiGuard  # noqa: E402
from owlbank.policy import Policy  # noqa: E402
from owlbank.segments import SegmentLog  # noqa: E402
from owlbank.store import MemoryStore  # noqa: E402
from owlbank.token_seal import FakeSealer  # noqa: E402
from tac_bridge.agent_client import (  # noqa: E402
    AgentError,
    TextDelta,
    ToolCall,
    ToolResult,
    TurnEnd,
    join_system,
)
from tac_bridge.app import PeekingWebSocket  # noqa: E402
from tac_bridge.bridge import Bridge, BridgeConfig  # noqa: E402
from tac_bridge.clock import local_now_line, zone  # noqa: E402
from tac_bridge.hosting import SessionPin  # noqa: E402
from tests.strands_fakes import FakeMemoryData  # noqa: E402
from tests.strands_parity import PerInputScript, StrandsDriven, memory_for, plain_fake, preferred_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

CALL_SID = "CA" + "1" * 32
ACCOUNT_SID = "AC" + "0" * 32
AUTH_TOKEN = "test-auth-token-not-a-secret"
PUBLIC_DOMAIN = "relay.owlbank.example"
ANI = "+12025550101"
ANI_PROFILE = "mem_profile_ani_household"
VERIFIED_PROFILE = "mem_profile_verified_c1001"
VERIFIED_EMAIL = "thomas.doe@owlbank-demo.example"  # C-1001's memory_email in data/fixtures.json
VALID_ID_TYPES = ("chat", "email", "phone", "pushUserID", "whatsapp")  # Twilio's 400/20400 list
CALL_TOKEN = "tok_" + "Zx9" * 14  # the opaque call token (what the front door holds)
NOW = 1_800_000_000
SEALER = FakeSealer()


def load_json(name: str) -> Any:
    return json.loads((DATA / name).read_text())


# --------------------------------------------------------------------------- store seeding


def seeded_store() -> MemoryStore:
    store = MemoryStore()
    for c in load_json("fixtures.json")["customers"]:
        store.put("customers", c)
    return store


def write_call_state(store: MemoryStore, call_sid: str = CALL_SID, *, verified: bool = True,
                     level: int = 1, language: str = "en-US", token: str | None = CALL_TOKEN,
                     **fields: Any) -> None:
    """What the front door leaves in call state before connecting the AI.

    Only typed ``CallState`` fields (no raw attributes). The token is stored
    sealed. ``fields`` sets e.g. ``pending_tool``, ``pending_result`` (an
    envelope object, or its JSON) and ``interrupted_utterance``.
    """
    if isinstance(fields.get("pending_result"), str):
        fields["pending_result"] = json.loads(fields["pending_result"])
    fields.setdefault("ws_nonce_hash", ws_nonce.nonce_hash(relay_nonce(call_sid)))
    state = cs.CallState(call_sid=call_sid, expires_at=NOW + 3600, language=language, ani=ANI,
                         sealed_token=SEALER.seal(token, call_sid=call_sid) if token else None,
                         candidate_ids=("C-1001",),
                         verified_customer_id="C-1001" if verified else None,
                         level=level if verified else 0,
                         factors_verified=("ani_match", "zip") if verified else (), **fields)
    cs.create(store, state)


def clock_line_at(bridge: Bridge, lang: str, at: float | None = None) -> str:
    """The clock line built fresh at ``at`` (default: the bridge's clock now), in the bridge's
    zone: what :meth:`Bridge._clock_line` holds for a call whose line was set then."""
    now = bridge._now() if at is None else at  # noqa: SLF001
    return local_now_line(datetime.fromtimestamp(now, UTC), zone(bridge._cfg.timezone), lang)  # noqa: SLF001


def relay_nonce(call_sid: str = CALL_SID) -> str:
    """The test relay nonce for a call (what the front door put in the relay URL)."""
    return "nonce-" + call_sid


def rearm_nonce(store: MemoryStore, call_sid: str = CALL_SID) -> None:
    """A front-door reconnect: a fresh nonce (here: the same test value) for the next socket."""
    cs.mutate(store, call_sid, lambda _s: {"ws_nonce_hash": ws_nonce.nonce_hash(relay_nonce(call_sid))},
              now=lambda: NOW)


def front_door_update(store: MemoryStore, call_sid: str = CALL_SID, **changes: Any) -> None:
    """A later front-door write (e.g. step-up), through ``callstate.mutate``."""
    cs.mutate(store, call_sid, lambda _s: changes, now=lambda: NOW)


# --------------------------------------------------------------------------- fake agent


Step = list[Any] | Exception


@dataclass
class FakeAgent(PerInputScript):
    """Scripted ``AgentClient``. Each agent input consumes the next step of ``script``: a ``turn``,
    and a recorded ``await_caller`` result (its step is never asked for, no text)."""

    script: list[Step] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    delay_s: float = 0.0
    closed: int = 0
    sessions: list[str] = field(default_factory=list)

    histories: list[str] = field(default_factory=list)
    #: Segment-traits: ``(session_id, user_text, assistant_text)`` per ``record_exchange``.
    exchanges: list[tuple[str, str, str]] = field(default_factory=list)

    async def begin_session(self, session_id: str, *, history: str = "reload") -> None:
        self.sessions.append(session_id)
        self.histories.append(history)  # the bridge's session-ownership decision

    async def record_exchange(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """Recorded with no model call; no script step is consumed."""
        self.exchanges.append((session_id, user_text, assistant_text))

    def _record(self, session_id: str, user_text: str | None, tool_results: Sequence[ToolResult],
                system_prompts: Mapping[str, str] | None, system_suffix: str | None) -> dict[str, Any]:
        # What the model sees (the preferred static part and the suffix joined), and both
        # parts on their own: the static part per model key, the per-call part and clock line after
        # the cache point.
        rec = {"session_id": session_id, "user_text": user_text, "tool_results": list(tool_results),
               "system_prompt": join_system(preferred_prompt(system_prompts), system_suffix),
               "system_prompts": system_prompts, "system_suffix": system_suffix}
        self.calls.append(rec)
        return rec

    def _next_step(self, rec: dict[str, Any]) -> Step:
        return self.script.pop(0) if self.script else [TextDelta("OK."), TurnEnd("end_turn")]

    def turn(self, session_id: str, *, user_text: str | None, tool_results: Sequence[ToolResult],
             system_prompts: Mapping[str, str] | None, system_suffix: str | None = None,
             turn_id: int | None = None) -> AsyncIterator[Any]:
        if tool_results and user_text:  # the results are recorded, the words answered
            self._skip_record_step(session_id, tool_results, system_prompts, system_suffix)
            tool_results = ()
        rec = self._record(session_id, user_text, tool_results, system_prompts, system_suffix)
        return self._run(self._next_step(rec))

    def record_results(self, session_id: str, tool_results: Sequence[ToolResult], *,
                       system_prompts: Mapping[str, str] | None, system_suffix: str | None = None,
                       turn_id: int | None = None) -> AsyncIterator[Any]:
        """Recorded with no model call: the step is skipped, the stream is ``TurnEnd``."""
        self._skip_record_step(session_id, tool_results, system_prompts, system_suffix)
        return self._run([TurnEnd("end_turn")])

    async def _run(self, step: Step) -> AsyncIterator[Any]:
        try:
            if isinstance(step, Exception):
                raise step
            if any(isinstance(ev, dict) for ev in step):
                raise TypeError("raw model events need the real client: build the agent with new_agent")
            for ev in step:
                if self.delay_s:
                    await asyncio.sleep(self.delay_s)
                if isinstance(ev, Exception):
                    raise ev
                yield ev
        finally:
            self.closed += 1

    def all_inputs_text(self) -> str:
        """Everything the model was ever given, serialized (for token-leak checks)."""
        return json.dumps([{**c, "tool_results": [r.content for r in c["tool_results"]]}
                           for c in self.calls], default=str)


@dataclass
class StrandsFakeAgent(StrandsDriven, FakeAgent):
    """``FakeAgent``'s scripts through a real ``StrandsAgentClient`` (``tests.strands_parity``)."""

    memory: Any = None

    def __post_init__(self) -> None:
        self._attach(self.memory if self.memory is not None else FakeMemoryData())

    def all_inputs_text(self) -> str:
        return FakeAgent.all_inputs_text(self) + self.model_inputs_text()


def new_agent(script: Sequence[Step] = (), *, store: Any = None, delay_s: float = 0.0) -> FakeAgent:
    """The rigs' scripted agent (``tests.strands_parity``): the scripts through the real Strands
    client, with the store's Memory; the plain ``FakeAgent`` for a ``protocol_fake`` test."""
    if plain_fake():
        return FakeAgent(list(script), delay_s=delay_s)
    return StrandsFakeAgent(list(script), delay_s=delay_s,
                            memory=memory_for(store) if store is not None else None)


def say(*parts: str) -> list[Any]:
    return [*(TextDelta(p) for p in parts), TurnEnd("end_turn")]


def tool(name: str, tool_use_id: str = "tu_1", **args: Any) -> list[Any]:
    return [ToolCall(tool_use_id, name, dict(args)), TurnEnd("tool_use")]


# Raw ConverseStream events of one model message, for steps that need the message structure
# (text in a message that also calls a tool is never spoken). Only the real client plays
# them (new_agent).


def h_text(t: str, i: int = 0) -> list[dict[str, Any]]:
    """A text content block."""
    return [{"contentBlockStart": {"contentBlockIndex": i, "start": {}}},
            {"contentBlockDelta": {"contentBlockIndex": i, "delta": {"text": t}}},
            {"contentBlockStop": {"contentBlockIndex": i}}]


def h_reasoning(t: str, i: int = 0) -> list[dict[str, Any]]:
    """A reasoning content block (never spoken)."""
    return [{"contentBlockStart": {"contentBlockIndex": i, "start": {}}},
            {"contentBlockDelta": {"contentBlockIndex": i, "delta": {"reasoningContent": {"text": t}}}},
            {"contentBlockStop": {"contentBlockIndex": i}}]


def h_tool(name: str, tool_use_id: str = "tu_1", i: int = 0, **args: Any) -> list[dict[str, Any]]:
    """A toolUse content block."""
    return [{"contentBlockStart": {"contentBlockIndex": i, "start": {"toolUse": {
                "toolUseId": tool_use_id, "name": name, "type": "tool_use"}}}},
            {"contentBlockDelta": {"contentBlockIndex": i, "delta": {"toolUse": {"input": json.dumps(args)}}}},
            {"contentBlockStop": {"contentBlockIndex": i}}]


def h_msg(*blocks: list[dict[str, Any]], stop: str | None = None) -> list[dict[str, Any]]:
    """One model message; ``stop`` defaults to ``tool_use`` if any block is a toolUse."""
    events = [ev for b in blocks for ev in b]
    if stop is None:
        has_tool = any("toolUse" in (ev.get("contentBlockStart", {}).get("start") or {}) for ev in events)
        stop = "tool_use" if has_tool else "end_turn"
    return [{"messageStart": {"role": "assistant"}}, *events, {"messageStop": {"stopReason": stop}}]


# --------------------------------------------------------------------------- fake back end


@dataclass
class FakeBackend:
    """``bank_tools``: ``{"op","token","call_sid","args"}``; ``responses[op]`` is a list or a callable."""

    responses: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def call(self, op: str, *, token: str | None, call_sid: str,
                   args: Mapping[str, Any], idempotency_key: str | None = None) -> dict[str, Any]:
        rec = {"op": op, "token": token, "call_sid": call_sid, "args": dict(args)}
        if idempotency_key is not None:
            rec["idempotency_key"] = idempotency_key
        self.calls.append(rec)
        r = self.responses.get(op, {"error": "NOT_FOUND", "tool": op})
        if isinstance(r, list):
            r = r.pop(0)
        if callable(r):
            r = r(token=token, call_sid=call_sid, args=args)
        if isinstance(r, Exception):
            raise r
        return dict(r)


# --------------------------------------------------------------------------- fake memory (wrapper)


@dataclass
class FakeMemory:
    """The bridge's ``MemoryClient``; records what it saw at each call."""

    profiles: dict[str, list[str]] = field(default_factory=lambda: {VERIFIED_EMAIL: [VERIFIED_PROFILE]})
    prompts: dict[str, str] = field(default_factory=lambda: {
        VERIFIED_PROFILE: "## Key Observations\n- Asked about a card replacement by SMS last week"})
    lookups: list[str] = field(default_factory=list)
    creates: list[str] = field(default_factory=list)
    retrievals: list[dict[str, Any]] = field(default_factory=list)
    lookup_error: Exception | None = None
    create_error: Exception | None = None

    async def lookup_profile_ids(self, identifier: str) -> list[str]:
        self.lookups.append(identifier)
        if self.lookup_error is not None:
            raise self.lookup_error
        return list(self.profiles.get(identifier, []))

    async def create_profile(self, identifier: str) -> str:
        self.creates.append(identifier)
        if self.create_error is not None:
            raise self.create_error
        pid = f"mem_profile_created_{len(self.creates)}"
        self.profiles[identifier] = [pid]
        return pid

    async def retrieve_prompt(self, context: Any) -> str | None:
        self.retrievals.append({"profile_id": context.profile_id, "profile": context.profile})
        return self.prompts.get(context.profile_id)


# --------------------------------------------------------------------------- fake Twilio services


class FakeOrchestrator:
    """Conversation Orchestrator: one ACTIVE conversation per CallSid, CUSTOMER bound to the ANI profile."""

    def __init__(self) -> None:
        self.list_conversations_calls = 0

    async def list_conversations(self, channel_id: str | None = None, status: Any = None, **_: Any):
        self.list_conversations_calls += 1
        return [ConversationResponse(id=f"conv_{channel_id[-8:]}", accountId=ACCOUNT_SID, status="ACTIVE")]

    async def list_participants(self, conversation_id: str):
        return [
            ParticipantResponse(id="pa_customer", conversationId=conversation_id, accountId=ACCOUNT_SID,
                                name="caller", type="CUSTOMER", profileId=ANI_PROFILE,
                                addresses=[ParticipantAddress(channel="VOICE", address=ANI)]),
            ParticipantResponse(id="pa_agent", conversationId=conversation_id, accountId=ACCOUNT_SID,
                                name="owlbank", type="AI_AGENT",
                                addresses=[ParticipantAddress(channel="VOICE", address="+12025550199")]),
        ]

    async def list_communications(self, conversation_id: str | None = None, **_: Any):
        return []


class FakeTwilioMemory:
    """Stands in for ``tac.context.memory.MemoryClient`` (Twilio Memory REST)."""

    store_id = "mem_store_test"

    def __init__(self) -> None:
        self.lookups: list[tuple[str, str]] = []
        self.creates: list[dict[str, Any]] = []
        self.lookup_error: Exception | None = None
        self.create_error: Exception | None = None
        self.get_profile_ids: list[str] = []
        self.recall_ids: list[str] = []
        self.by_email = {VERIFIED_EMAIL: [VERIFIED_PROFILE]}
        self.by_phone = {ANI: [ANI_PROFILE]}
        #: More trait groups per profile ID, beside ``Contact``: what a feed such
        #: as Segment or ``scripts/web_activity.py`` wrote, returned as stored (no filtering).
        self.trait_groups: dict[str, dict[str, Any]] = {}
        #: Seconds the recall waits before answering (a slow Memory).
        self.recall_delay_s = 0.0
        #: Segment-traits: a profile's whole recall (observations and summaries) in place of
        #: the default single observation.
        self.recalls: dict[str, MemoryRetrievalResponse] = {}

    async def lookup_profile(self, id_type: str, value: str) -> ProfileLookupResponse:
        """Twilio accepts only these identifier types (live 400, code 20400); anything else raises."""
        self.lookups.append((id_type, value))
        if self.lookup_error is not None:
            raise self.lookup_error
        if id_type not in VALID_ID_TYPES:
            raise httpx.HTTPStatusError(
                f"invalid identifier type '{id_type}'", request=httpx.Request("POST", "https://memory.test"),
                response=httpx.Response(400, json={"code": 20400}))
        table = self.by_email if id_type == "email" else self.by_phone
        return ProfileLookupResponse(normalizedValue=value, profiles=table.get(value, []))

    async def create_profile(self, traits: dict[str, dict[str, Any]]) -> str:
        self.creates.append(traits)
        if self.create_error is not None:
            raise self.create_error
        pid = f"mem_profile_created_{len(self.creates)}"
        email = traits.get("Contact", {}).get("email")
        if email is not None:
            self.by_email[email] = [pid]
        return pid

    async def get_profile(self, profile_id: str, trait_groups: list[str] | None = None) -> ProfileResponse:
        self.get_profile_ids.append(profile_id)
        name = "Thomas Doe" if profile_id == VERIFIED_PROFILE else "Household (ANI) Profile"
        return ProfileResponse(id=profile_id, createdAt="2026-01-01T00:00:00Z",
                               traits={"Contact": {"name": name}, **self.trait_groups.get(profile_id, {})})

    async def retrieve_memory(self, profile_id: str, **_: Any) -> MemoryRetrievalResponse:
        self.recall_ids.append(profile_id)
        if self.recall_delay_s:
            await asyncio.sleep(self.recall_delay_s)
        if profile_id in self.recalls:
            return self.recalls[profile_id]
        content = ("Verified customer asked about card replacement by SMS"
                   if profile_id == VERIFIED_PROFILE else "ANI HOUSEHOLD OBSERVATION")
        return MemoryRetrievalResponse(observations=[ObservationInfo(
            content=content, source="conversation-intelligence", id="obs_1",
            createdAt="2026-10-01T00:00:00Z", updatedAt="2026-10-01T00:00:00Z")])


def make_tac() -> TAC:
    """Real TAC in Orchestrator mode, with the network seams replaced by fakes."""
    config = TACConfig(account_sid=ACCOUNT_SID, auth_token=AUTH_TOKEN, api_key="SK" + "0" * 32,
                       api_secret="test-api-secret", phone_number="+12025550199",
                       conversation_configuration_id="conv_configuration_test",
                       voice_public_domain=PUBLIC_DOMAIN, log_level="WARNING")
    with patch("tac.context.conversation.ConversationClient.get_configuration",
               return_value=SimpleNamespace(memory_store_id="mem_store_test")):
        tac = TAC(config)
    tac.conversation_orchestrator_client = FakeOrchestrator()  # type: ignore[assignment]
    tac.conversation_memory_client = FakeTwilioMemory()  # type: ignore[assignment]
    return tac


# --------------------------------------------------------------------------- ConversationRelay simulator


_CLOSE = object()


class SimSocket:
    """In-memory ``WebSocketProtocol``: Twilio's side of a ConversationRelay socket."""

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[Any] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.accepted = False
        self.closed = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive_json(self) -> Any:
        msg = await self.inbound.get()
        if msg is _CLOSE:
            raise WebSocketDisconnectError("caller hung up")
        return msg

    async def send_text(self, data: str) -> None:
        if self.closed:
            raise WebSocketDisconnectError("closed")
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        self.closed = True


async def wait_for(pred: Callable[[], bool], timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not pred():
        if loop.time() > end:
            raise AssertionError("timed out waiting for the simulator")
        await asyncio.sleep(0.002)


#: How :class:`CRSimulator` hands the socket to the bridge: ``fargate`` (the nonce as the query's
#: ``n``) or ``agentcore`` (the Runtime session ID = the CallSid through the process's
#: :class:`SessionPin`, the call binding, and the nonce as ``customParameters["n"]``). A test module
#: re-runs a suite on ``agentcore`` by setting this (``test_bridge_agentcore_suites.py``).
SIM_HOSTING = "fargate"


class CRSimulator:
    """Drives TAC's ``VoiceChannel.handle_websocket`` the way ConversationRelay does."""

    def __init__(self, voice: VoiceChannel, bridge: Bridge, *, pin: SessionPin | None = None) -> None:
        self.voice = voice
        self.bridge = bridge
        self.pin = pin if pin is not None else SessionPin()  # one per process: share the rig's
        self.socket = SimSocket()
        self.task: asyncio.Task[None] | None = None
        self._mark = 0

    async def connect(self, call_sid: str = CALL_SID, custom_parameters: dict[str, Any] | None = None,
                      *, nonce: str | None | object = ...) -> None:
        self.call_sid = call_sid
        if nonce is ...:
            nonce = relay_nonce(call_sid)
        params = dict(custom_parameters or {"callSid": call_sid})
        if SIM_HOSTING == "agentcore":  # the front door's <Parameter>s; the session header = CallSid
            params.setdefault("callSid", call_sid)
            if nonce is not None:
                params["n"] = nonce
            self.peek = PeekingWebSocket(self.socket, self.bridge, nonce_from_parameters=True,
                                         expected_call_sid=call_sid if self.pin.check(call_sid) else "")
        else:
            self.peek = PeekingWebSocket(self.socket, self.bridge, nonce=nonce)  # type: ignore[arg-type]
        self.task = asyncio.create_task(self.voice.handle_websocket(self.peek))
        await self.socket.inbound.put({
            "type": "setup", "sessionId": "VX" + "2" * 32, "accountSid": ACCOUNT_SID, "callSid": call_sid,
            "from": ANI, "to": "+12025550199", "direction": "inbound", "callType": "PSTN",
            "callStatus": "IN-PROGRESS", "customParameters": params})
        await wait_for(lambda: self.voice.get_conversation_session_by_call_sid(call_sid) is not None)

    @property
    def session(self) -> Any:
        return self.voice.get_conversation_session_by_call_sid(self.call_sid)

    def frames(self) -> list[dict[str, Any]]:
        """Frames sent since the last ``frames()``/``prompt()``."""
        out = self.socket.sent[self._mark:]
        self._mark = len(self.socket.sent)
        return out

    async def wait_reply(self, timeout: float = 3.0) -> list[dict[str, Any]]:
        start = self._mark
        await wait_for(lambda: any(f.get("type") == "text" and f.get("last") for f in self.socket.sent[start:]),
                       timeout)
        await self.idle()
        return self.frames()

    async def prompt(self, text: str, *, timeout: float = 3.0) -> list[dict[str, Any]]:
        self._mark = len(self.socket.sent)
        await self.socket.inbound.put({"type": "prompt", "voicePrompt": text, "lang": "en-US", "last": True})
        return await self.wait_reply(timeout)

    async def prompt_nowait(self, text: str) -> None:
        self._mark = len(self.socket.sent)
        await self.socket.inbound.put({"type": "prompt", "voicePrompt": text, "lang": "en-US", "last": True})

    async def interrupt(self, heard: str = "", ms: int = 300) -> None:
        await self.socket.inbound.put({"type": "interrupt", "utteranceUntilInterrupt": heard,
                                       "durationUntilInterruptMs": ms})

    async def dtmf(self, digit: str) -> None:
        """A key press, as ConversationRelay reports it (``dtmfDetection``)."""
        await self.socket.inbound.put({"type": "dtmf", "digit": digit})

    async def idle(self) -> None:
        for _ in range(20):
            await asyncio.sleep(0)
        await self.bridge.wait_idle()

    async def hangup(self) -> None:
        await self.socket.inbound.put(_CLOSE)
        if self.task is not None:
            await asyncio.wait_for(self.task, 3)


def spoken(frames: Sequence[Mapping[str, Any]]) -> str:
    return "".join(f.get("token", "") for f in frames if f.get("type") == "text")


def end_frame(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    ends = [f for f in frames if f.get("type") == "end"]
    assert len(ends) <= 1
    if not ends:
        return None
    return {"frame": ends[0], "data": json.loads(ends[0]["handoffData"])}


# --------------------------------------------------------------------------- a whole rig


@dataclass
class Rig:
    store: MemoryStore
    agent: FakeAgent
    backend: FakeBackend
    memory: Any
    tac: TAC
    voice: VoiceChannel
    bridge: Bridge
    journal: SegmentLog
    pii: FakePiiRedactor = field(default_factory=FakePiiRedactor)
    pin: SessionPin = field(default_factory=SessionPin)  # the process's session pin (agentcore)

    def sim(self) -> CRSimulator:
        return CRSimulator(self.voice, self.bridge, pin=self.pin)

    def call(self, call_sid: str = CALL_SID) -> dict[str, Any]:
        return self.store.get("calls", {"call_sid": call_sid}) or {}

    def events(self, call_sid: str = CALL_SID) -> list[dict[str, Any]]:
        return self.journal.read(call_sid)


def inline_money_policy() -> dict[str, Any]:
    """``policy.json`` with ``confirm: none`` on the money tools: the bridge runs them inline
    (the path a deployment that turns deterministic confirmation off gets)."""
    doc = load_json("policy.json")
    for name in ("make_payment", "transfer_funds"):
        doc["tools"][name]["confirm"] = "none"
    return doc


def make_rig(*, script: list[Step] | None = None, backend: FakeBackend | None = None,
             memory: Any = "fake", config: BridgeConfig | None = None, store: MemoryStore | None = None,
             policy_doc: dict[str, Any] | None = None) -> Rig:
    from tac_bridge.bridge import TACMemoryClient

    store = store or seeded_store()
    tac = make_tac()
    voice = VoiceChannel(tac, config=ConversationRelayProviderConfig(memory_mode="never"))
    agent = new_agent(script or [], store=store)
    backend = backend or FakeBackend()
    config = config or BridgeConfig()
    if memory == "tac":
        mem: Any = TACMemoryClient(tac, allow=config.memory_trait_groups,
                                   max_age_hours=config.web_activity_max_age_hours, now=lambda: NOW)
    else:
        mem = FakeMemory() if memory == "fake" else memory
    journal = SegmentLog(store, now=lambda: NOW)
    policy = Policy.from_dict(policy_doc or load_json("policy.json"))
    pii = FakePiiRedactor()
    bridge = Bridge(store=store, agent=agent, backend=backend, memory=mem,
                    catalog=PromptCatalog.from_dict(load_json("prompt_catalog.json")),
                    policy=policy, pii=PiiGuard.for_policy(pii, policy), journal=journal,
                    sealer=SEALER, config=config, now=lambda: NOW, io=_inline_io)
    tac.on_message_ready(bridge.on_message_ready)
    tac.on_interrupt(bridge.on_interrupt)
    bridge.voice = voice
    return Rig(store, agent, backend, mem, tac, voice, bridge, journal, pii)


async def _inline_io(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run store I/O inline (MemoryStore isn't thread-safe; production uses a thread)."""
    return fn(*args, **kwargs)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# --------------------------------------------------------------------------- self-tests of the fakes


def test_fake_agent_records_and_closes() -> None:
    agent = FakeAgent([say("Hi"), AgentError("boom")])

    async def go() -> list[Any]:
        out = [ev async for ev in agent.turn("s", user_text="x", tool_results=(), system_prompts={"haiku-5-5": "p"})]
        with pytest.raises(AgentError):
            async for _ in agent.turn("s", user_text="y", tool_results=(), system_prompts={"haiku-5-5": "p"}):
                pass
        return out

    out = run(go())
    assert out == [TextDelta("Hi"), TurnEnd("end_turn")]
    assert [c["user_text"] for c in agent.calls] == ["x", "y"]
    assert agent.closed == 2


def test_simulator_reaches_real_tac_orchestrator_session() -> None:
    rig = make_rig(script=[say("Hello", " there.")])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        assert sim.socket.accepted
        assert sim.session.conversation_id.startswith("conv_")
        assert sim.session.call_sid == CALL_SID
        frames = await sim.prompt("hi")
        await sim.hangup()
        return frames

    frames = run(go())
    assert frames == [{"type": "text", "token": "Hello", "last": False},
                      {"type": "text", "token": " there.", "last": False},
                      {"type": "text", "token": "", "last": True}]
