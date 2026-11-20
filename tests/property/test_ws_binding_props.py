"""AgentCore hosting: the bridge's call binding and session pin on ``agentcore``.

* A state machine over the in-memory store. Sockets arrive with any
  combination of Runtime session ID, ``setup.callSid``, ``customParameters.callSid`` and
  ``customParameters.n`` (fresh, stale, reused, foreign, absent, not a string), while the front
  door mints new nonces. A socket is accepted iff the three IDs are equal, the process pin agrees
  and the nonce matches the call's current unconsumed hash; each minted nonce is accepted at most
  once; a rejected socket never reaches ``on_setup`` or TAC (the ``setup`` is never returned); a
  binding rejection writes nothing; the auth-failure reason is ``binding`` before ``nonce``.
* With the process pinned, any sequence of re-entries (each a fresh nonce)
  interleaved with refused sockets for other sessions keeps session ownership exactly as on
  Fargate (``new`` once, then ``cached``; the owner is this process), and the single-conversation
  guard still hands TAC exactly one ACTIVE conversation however many the Orchestrator lists.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule
from tac.channels.websocket_protocol import WebSocketDisconnectError
from tac.models.conversation import ConversationResponse

from owlbank import callstate as cs
from owlbank import ws_nonce
from tac_bridge.app import PeekingWebSocket
from tac_bridge.bridge import SingleConversationGuard
from tac_bridge.hosting import SessionPin
from tests.component import test_bridge_fakes as fakes
from tests.component.test_bridge_fakes import (
    ACCOUNT_SID,
    CALL_SID,
    NOW,
    FakeOrchestrator,
    make_rig,
    relay_nonce,
    say,
    write_call_state,
)
from tests.logutil import fields

CALL_B = "CA" + "3" * 32
FOREIGN = "CA" + "9" * 32  # a well-formed CallSid with no call state
CALLS = (CALL_SID, CALL_B)
SAME = object()     # customParameters.callSid = setup.callSid
ABSENT = object()   # the key is left out

ids = st.one_of(st.sampled_from([CALL_SID, CALL_B, FOREIGN, None, ""]), st.text(max_size=40))
carried_ids = st.one_of(st.just(SAME), st.just(ABSENT), ids, st.integers())
nonce_choices = st.one_of(st.integers(0, 64), st.just(ABSENT), st.just(""), st.integers().map(lambda i: -i - 1000),
                          st.text(max_size=24).map(lambda t: "never-minted-" + t))


class Inner:
    """The inner socket: one ``setup`` frame, records closes."""

    def __init__(self, frame: dict[str, Any]) -> None:
        self.frame = frame
        self.closed = False

    async def receive_json(self) -> Any:
        return self.frame

    async def close(self) -> None:
        self.closed = True


class Reasons(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.reasons: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() == "twilio_auth_failed":
            self.reasons.append(fields(record)["reason"])


def capture_reasons() -> tuple[Reasons, logging.Logger]:
    logger = logging.getLogger("owlbank.security")
    handler = Reasons()
    logger.addHandler(handler)
    return handler, logger


def snapshot(store: Any) -> str:
    return json.dumps([store.get("calls", {"call_sid": sid}) for sid in CALLS], sort_keys=True, default=str)


async def offer(bridge: Any, pin: SessionPin, header: Any, frame: dict[str, Any]) -> tuple[bool, Any, Inner]:
    """One socket the way the ``agentcore`` ``/ws`` route builds it: ``(accepted, message, inner)``."""
    inner = Inner(frame)
    expected = header if isinstance(header, str) and header and pin.check(header) else ""
    peek = PeekingWebSocket(inner, bridge, nonce_from_parameters=True, expected_call_sid=expected)
    try:
        msg = await peek.receive_json()
    except WebSocketDisconnectError:
        return False, None, inner
    return peek.verified, msg, inner




class BindingMachine(RuleBasedStateMachine):
    @initialize()
    def start(self) -> None:
        self.rig = make_rig()
        for sid in CALLS:
            write_call_state(self.rig.store, sid)
        self.current: dict[str, str | None] = {sid: relay_nonce(sid) for sid in CALLS}
        self.minted: list[str] = [relay_nonce(sid) for sid in CALLS]
        self.accepted: list[str] = []
        self.pin = SessionPin()
        self.pinned: str | None = None  # the model of the pin
        self.setups: list[str] = []
        self.counter = 0

        def on_setup(call_sid: Any, params: Any = None) -> Any:
            self.setups.append(call_sid)  # recorded at the call, before the task runs
            return asyncio.sleep(0)

        self.rig.bridge.on_setup = on_setup  # type: ignore[method-assign]

    @rule(call=st.sampled_from(CALLS))
    def mint(self, call: str) -> None:
        """The front door's next ``<Connect>`` for the call: a fresh nonce, the old one stale."""
        self.counter += 1
        nonce = f"nonce-{call[-4:]}-{self.counter}"
        cs.mutate(self.rig.store, call, lambda _s: {"ws_nonce_hash": ws_nonce.nonce_hash(nonce)}, now=lambda: NOW)
        self.current[call] = nonce
        self.minted.append(nonce)

    @rule(call=st.sampled_from(CALLS))
    def connect_as_the_front_door_would(self, call: str) -> None:
        """A well-formed socket for ``call`` with its latest nonce (consumed already, or not):
        accepted only while the nonce is unconsumed and the pin is ``call``'s."""
        latest = [n for n in self.minted if n == relay_nonce(call) or n.startswith(f"nonce-{call[-4:]}-")][-1]
        self._connect(call, call, SAME, latest)

    @rule(header=ids, setup_sid=ids, carried=carried_ids, choice=nonce_choices)
    def connect(self, header: Any, setup_sid: Any, carried: Any, choice: Any) -> None:
        if isinstance(choice, int) and choice >= 0:
            n: Any = self.minted[choice % len(self.minted)]
        else:
            n = choice  # absent, empty, not a string, or never minted
        self._connect(header, setup_sid, carried, n)

    def _connect(self, header: Any, setup_sid: Any, carried: Any, n: Any) -> None:
        params: dict[str, Any] = {}
        if carried is not ABSENT:
            params["callSid"] = setup_sid if carried is SAME else carried
        if n is not ABSENT:
            params["n"] = n
        frame: dict[str, Any] = {"type": "setup", "accountSid": ACCOUNT_SID, "customParameters": params}
        if setup_sid is not None:
            frame["callSid"] = setup_sid

        if isinstance(header, str) and header and self.pinned is None:
            self.pinned = header
        pin_ok = isinstance(header, str) and bool(header) and header == self.pinned
        ids_ok = pin_ok and setup_sid == header and "callSid" in params and params["callSid"] == setup_sid
        nonce_ok = isinstance(n, str) and bool(n) and setup_sid in self.current and self.current[setup_sid] == n
        expect = ids_ok and nonce_ok

        before, setups_before = snapshot(self.rig.store), len(self.setups)
        handler, logger = capture_reasons()
        try:
            accepted, msg, inner = asyncio.run(offer(self.rig.bridge, self.pin, header, frame))
        finally:
            logger.removeHandler(handler)

        event("accepted" if accepted else "binding" if not ids_ok else "nonce")
        assert accepted is expect
        if accepted:
            assert n not in self.accepted  # each minted nonce at most once
            self.accepted.append(n)
            self.current[setup_sid] = None
            assert self.setups[setups_before:] == [setup_sid]
            assert "n" not in msg["customParameters"] and handler.reasons == []
            assert not inner.closed
            self.rig.bridge.session_end(setup_sid, "closed")
        else:
            assert self.setups[setups_before:] == []  # never on_setup, and TAC never got the setup
            assert inner.closed
            assert handler.reasons == ["nonce" if ids_ok else "binding"]
            if not ids_ok:
                assert snapshot(self.rig.store) == before  # a binding rejection writes nothing

    @invariant()
    def store_matches_the_model(self) -> None:
        if not hasattr(self, "rig"):
            return
        for sid in CALLS:
            stored = self.rig.call(sid)["ws_nonce_hash"]
            want = self.current[sid]
            assert stored == (ws_nonce.nonce_hash(want) if want is not None else None)


TestBindingMachine = BindingMachine.TestCase
TestBindingMachine.settings = settings(max_examples=150, stateful_step_count=25, deadline=None,
                                       suppress_health_check=[HealthCheck.too_slow])




class ManyActive(FakeOrchestrator):
    """The Orchestrator listing ``count[0]`` ACTIVE conversations for the call (the failure seen live)."""

    def __init__(self) -> None:
        super().__init__()
        self.count = [1]
        self.closed: list[str] = []

    async def list_conversations(self, channel_id: str | None = None, status: Any = None, **_: Any):
        self.list_conversations_calls += 1
        return [ConversationResponse(id=f"conv_{i}", accountId=ACCOUNT_SID, status="ACTIVE",
                                     createdAt=f"2026-10-19T00:00:0{i}Z") for i in range(self.count[0])]

    async def update_conversation(self, conversation_id: str, status: str) -> Any:
        self.closed.append(conversation_id)


steps = st.lists(st.one_of(st.tuples(st.just("reenter"), st.integers(1, 3)),
                           st.tuples(st.just("intruder"), st.sampled_from(["header", "setup", "param"]))),
                 min_size=1, max_size=6)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(plan=steps)
def test_ownership_and_single_conversation_hold_with_the_pin(plan: list[tuple[str, Any]]) -> None:
    old = fakes.SIM_HOSTING
    fakes.SIM_HOSTING = "agentcore"
    try:
        _run_plan(plan)
    finally:
        fakes.SIM_HOSTING = old


def _run_plan(plan: list[tuple[str, Any]]) -> None:
    reentries = sum(1 for kind, _ in plan if kind == "reenter")
    rig = make_rig(script=[say("Hello.") for _ in range(reentries)])
    orchestrator = ManyActive()
    rig.tac.conversation_orchestrator_client = SingleConversationGuard(orchestrator)  # type: ignore[assignment]
    write_call_state(rig.store)
    other = "CA" + "5" * 32
    write_call_state(rig.store, other)
    owner = rig.bridge.instance_id
    assert rig.pin.check(CALL_SID)  # the call's prewarm pinned the process

    async def go() -> None:
        for kind, arg in plan:
            if kind == "reenter":
                fakes.rearm_nonce(rig.store)
                orchestrator.count[0] = arg
                sim = rig.sim()
                await sim.connect()  # TAC resolved exactly one conversation, or this would time out
                frames = await sim.prompt("Hi", timeout=15)
                assert fakes.spoken(frames) == "Hello."
                await sim.hangup()
                assert rig.call()["agent_owner"] == owner
            else:
                before = json.dumps([rig.call(), rig.call(other)], sort_keys=True, default=str)
                header, setup_sid, param_sid = {
                    "header": (other, CALL_SID, CALL_SID),   # another session on this pinned process
                    "setup": (CALL_SID, other, other),        # a setup for another call
                    "param": (CALL_SID, CALL_SID, other),     # the parameters name another call
                }[arg]
                frame = {"type": "setup", "callSid": setup_sid,
                         "customParameters": {"callSid": param_sid, "n": relay_nonce(setup_sid)}}
                accepted, _msg, inner = await offer(rig.bridge, rig.pin, header, frame)
                assert not accepted and inner.closed
                assert json.dumps([rig.call(), rig.call(other)], sort_keys=True, default=str) == before

    asyncio.run(go())
    histories = rig.agent.histories
    if reentries == 0:
        assert histories == [] and rig.call().get("agent_owner") is None
    else:
        assert histories == ["new"] + ["cached"] * (reentries - 1)
    assert len(rig.agent.calls) == reentries  # a refused socket never reached the agent
    assert rig.call(other).get("agent_owner") is None
