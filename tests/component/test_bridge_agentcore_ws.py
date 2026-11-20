"""The bridge's ``/ws`` on AgentCore (call binding, nonce).

* Call binding: the Runtime session-ID header, ``setup.callSid`` and
  ``customParameters.callSid`` must be equal and the process pin must agree; otherwise the socket
  closes before TAC with ``reason=binding`` and nothing is written (the nonce stays unconsumed).
* The nonce: only ``setup.customParameters["n"]`` (removed before ``on_setup`` and TAC
  see it); a query ``n`` never counts. A valid nonce is accepted once; a missing, wrong or reused
  one closes the socket before TAC with ``reason=nonce`` (as on Fargate).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tac_bridge.app import PeekingWebSocket, build_app
from tac_bridge.hosting import SESSION_ID_HEADER, SessionPin
from tests.component.test_bridge_fakes import CALL_SID, make_rig, relay_nonce, say, write_call_state
from tests.logutil import lines as lines_of

OTHER_SID = "CA" + "2" * 32


def client_for(*, hosting: str = "agentcore", script: Any = None, pin: SessionPin | None = None):
    rig = make_rig(script=script or [say("Hello ", "from the bridge.")])
    app = build_app(rig.tac, rig.voice, rig.bridge, hosting=hosting,  # type: ignore[arg-type]
                    pin=pin)
    write_call_state(rig.store)
    return TestClient(app), rig


#: The header AgentCore sets on the upgrade: the Runtime session ID (= the CallSid).
SESSION = {SESSION_ID_HEADER: CALL_SID}
NONCE = {"n": relay_nonce()}


def setup_frame(params: dict[str, Any] | None = None, call_sid: str = CALL_SID) -> dict[str, Any]:
    return {"type": "setup", "callSid": call_sid, "from": "+12025550101", "to": "+12025550199",
            "customParameters": {"callSid": call_sid, **(params or {})}}


PROMPT = {"type": "prompt", "voicePrompt": "hi", "lang": "en-US", "last": True}


def talk(client: TestClient, path: str = "/ws", params: dict[str, Any] | None = None,
         headers: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """One accepted socket: setup, one prompt, the reply frames."""
    with client.websocket_connect(path, headers=SESSION if headers is None else headers) as ws:
        ws.send_json(setup_frame(NONCE if params is None else params))
        ws.send_json(PROMPT)
        return [json.loads(ws.receive_text()) for _ in range(3)]


def rejected(client: TestClient, path: str = "/ws", params: dict[str, Any] | None = None, *,
             headers: dict[str, str] | None = None, frame: dict[str, Any] | None = None) -> int:
    """A socket the bridge closes on setup: the close code."""
    with pytest.raises(WebSocketDisconnect) as ei, client.websocket_connect(
            path, headers=SESSION if headers is None else headers) as ws:
        ws.send_json(frame if frame is not None else setup_frame(NONCE if params is None else params))
        ws.receive_text()
    return ei.value.code


def auth_failures(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [ln["reason"] for ln in lines_of(caplog.records, "owlbank.security") if ln["event"] == "twilio_auth_failed"]


def record_setups(rig: Any) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    inner = rig.bridge.on_setup

    async def on_setup(call_sid: str | None, params: Any = None) -> None:
        seen.append(dict(params or {}))
        await inner(call_sid, params)

    rig.bridge.on_setup = on_setup
    return seen


# --------------------------------------------------------------------------- the nonce


def test_parameter_nonce_is_accepted_once_and_stripped() -> None:
    client, rig = client_for()
    setups = record_setups(rig)
    with client:
        frames = talk(client, params={**NONCE, "lang": "en-US"})
        assert "Hello from the bridge." in "".join(str(f.get("token", "")) for f in frames)
        assert rejected(client) == 1000  # reused
    assert len(rig.agent.calls) == 1
    assert setups == [{"callSid": CALL_SID, "lang": "en-US"}]  # on_setup never sees "n"


@pytest.mark.parametrize("path,params", [
    ("/ws", {}),                                                 # missing
    ("/ws", {"n": "wrong"}),                                     # wrong
    ("/ws", {"n": ""}),                                          # empty
    ("/ws", {"n": 12345}),                                       # not a string
    (f"/ws?n={relay_nonce()}", {}),                              # the query never counts
])
def test_missing_wrong_or_query_nonce_closes_before_tac(path: str, params: Any,
                                                        caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    client, rig = client_for()
    setups = record_setups(rig)
    with client:
        assert rejected(client, path, params) == 1000
    assert rig.agent.calls == [] and setups == []
    assert rig.call()["ws_nonce_hash"]  # not consumed
    assert auth_failures(caplog) == ["nonce"]


# --------------------------------------------------------------------------- the call binding


def _frame(call_sid: Any = CALL_SID, carried: Any = CALL_SID, *, carry: bool = True) -> dict[str, Any]:
    params: dict[str, Any] = {**NONCE}
    if carry:
        params["callSid"] = carried
    return {"type": "setup", "callSid": call_sid, "from": "+12025550101", "to": "+12025550199",
            "customParameters": params}


@pytest.mark.parametrize("headers,frame", [
    ({SESSION_ID_HEADER: OTHER_SID}, _frame()),                     # header differs from both
    ({}, _frame()),                                                 # no session header
    ({SESSION_ID_HEADER: ""}, _frame()),                            # empty session header
    (SESSION, _frame(call_sid=OTHER_SID)),                          # setup.callSid differs
    (SESSION, _frame(carried=OTHER_SID)),                           # customParameters.callSid differs
    (SESSION, _frame(carry=False)),                                 # customParameters.callSid absent
    (SESSION, _frame(call_sid=None)),                               # setup.callSid absent
    (SESSION, _frame(carried=12345)),                               # not a string
    ({SESSION_ID_HEADER: OTHER_SID}, _frame(OTHER_SID, OTHER_SID)), # all equal, but pinned to CALL_SID
])
def test_binding_mismatch_closes_before_tac_and_writes_nothing(headers: dict[str, str], frame: dict[str, Any],
                                                               caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    pin = SessionPin()
    assert pin.check(CALL_SID)  # the process belongs to CALL_SID (e.g. its prewarm came first)
    client, rig = client_for(pin=pin)
    setups = record_setups(rig)
    before = json.dumps(rig.call(), sort_keys=True, default=str)
    with client:
        assert rejected(client, headers=headers, frame=frame) == 1000
    assert rig.agent.calls == [] and setups == []
    assert json.dumps(rig.call(), sort_keys=True, default=str) == before  # nonce not consumed, nothing written
    assert auth_failures(caplog) == ["binding"]
    with client:  # the call's own socket still gets in with the same nonce
        talk(client)
    assert len(rig.agent.calls) == 1


def test_first_session_id_pins_the_process(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    client, rig = client_for()
    write_call_state(rig.store, OTHER_SID)
    with client:
        talk(client)  # pins CALL_SID
        frame = {**_frame(OTHER_SID, OTHER_SID), "customParameters": {"callSid": OTHER_SID,
                                                                       "n": relay_nonce(OTHER_SID)}}
        assert rejected(client, headers={SESSION_ID_HEADER: OTHER_SID}, frame=frame) == 1000
    assert auth_failures(caplog) == ["binding"]
    assert rig.call(OTHER_SID)["ws_nonce_hash"]  # the other call's nonce is untouched
    assert client.app.state.pin.check(CALL_SID) and not client.app.state.pin.check(OTHER_SID)


def test_prewarm_pins_the_process() -> None:
    client, rig = client_for()
    r = client.post("/invocations", json={"action": "prewarm"}, headers={SESSION_ID_HEADER: CALL_SID})
    assert r.status_code == 200
    r = client.post("/invocations", json={"action": "prewarm"}, headers={SESSION_ID_HEADER: OTHER_SID})
    assert r.status_code == 200 and r.json() == {"status": "ready"}  # still ready, touches nothing
    assert client.app.state.pin.check(CALL_SID) and not client.app.state.pin.check(OTHER_SID)
    with client:
        talk(client)
    assert len(rig.agent.calls) == 1


def test_a_binding_failure_never_reaches_the_nonce_check(monkeypatch: pytest.MonkeyPatch) -> None:
    client, rig = client_for()
    checked: list[Any] = []

    async def accept_connection(call_sid: Any, nonce: Any) -> bool:
        checked.append(call_sid)
        return True

    monkeypatch.setattr(rig.bridge, "accept_connection", accept_connection)
    with client:
        assert rejected(client, headers={SESSION_ID_HEADER: OTHER_SID}) == 1000
    assert checked == []


def test_reentry_on_the_pinned_session_is_accepted() -> None:
    from tests.component.test_bridge_fakes import rearm_nonce

    client, rig = client_for(script=[say("One."), say("Two.")])
    with client:
        talk(client)
        rearm_nonce(rig.store)
        talk(client)
    assert len(rig.agent.calls) == 2


def test_fargate_has_no_binding_check() -> None:
    rig = make_rig(script=[say("Hi.")])
    write_call_state(rig.store)

    async def go() -> Any:
        peek = PeekingWebSocket(FakeInner([_frame(carried=OTHER_SID)]), rig.bridge, nonce=relay_nonce())
        return peek, await peek.receive_json()

    peek, msg = asyncio.run(go())
    assert peek.verified and peek.call_sid == CALL_SID and msg["customParameters"]["n"] == relay_nonce()  # untouched on Fargate


class FakeInner:
    """The inner socket: hands out frames, records closes."""

    def __init__(self, frames: list[Any]) -> None:
        self.frames = list(frames)
        self.closed = False

    async def receive_json(self) -> Any:
        return self.frames.pop(0)

    async def close(self) -> None:
        self.closed = True


def test_tac_never_sees_the_parameter_nonce() -> None:
    rig = make_rig(script=[say("Hi.")])
    write_call_state(rig.store)

    async def go() -> Any:
        peek = PeekingWebSocket(FakeInner([setup_frame({"n": relay_nonce(), "x": "1"})]), rig.bridge,
                                nonce="ignored-query-value", nonce_from_parameters=True,
                                expected_call_sid=CALL_SID)
        msg = await peek.receive_json()
        await asyncio.sleep(0)
        return peek, msg

    peek, msg = asyncio.run(go())
    assert msg["customParameters"] == {"callSid": CALL_SID, "x": "1"}
    assert peek.custom_parameters == {"callSid": CALL_SID, "x": "1"}
    assert peek.verified and peek.call_sid == CALL_SID


def test_tracker_and_pin_are_for_agentcore_only() -> None:
    rig = make_rig(script=[say("Hi.")])
    with pytest.raises(ValueError):
        build_app(rig.tac, rig.voice, rig.bridge, pin=SessionPin())


def test_session_pin() -> None:
    pin = SessionPin()
    assert not pin.check(None) and not pin.check("") and not pin.pinned  # nothing pins
    assert pin.check(CALL_SID) and pin.pinned
    assert pin.check(CALL_SID)
    assert not pin.check(OTHER_SID) and not pin.check(None) and not pin.check(CALL_SID.lower())
    assert pin.check(CALL_SID)


def test_startup_and_call_end_hooks_run(caplog: pytest.LogCaptureFixture) -> None:
    """Model failover: ``on_startup`` runs in the lifespan (the fleet loops start on the running
    loop); under AgentCore ``on_call_end`` runs when the relay socket closes (the call's report is
    flushed). A failing hook is logged and ignored."""
    rig = make_rig(script=[say("Hello ", "from the bridge.")])
    seen: list[str] = []

    async def started() -> None:
        asyncio.get_running_loop()  # on the running loop
        seen.append("startup")

    async def ended() -> None:
        seen.append("call_end")

    async def broken() -> None:
        raise RuntimeError("flush failed")

    app = build_app(rig.tac, rig.voice, rig.bridge, hosting="agentcore",
                    on_startup=[started, broken], on_call_end=[broken, ended])
    write_call_state(rig.store)
    with TestClient(app) as client:
        assert seen == ["startup"]
        talk(client)
    assert seen == ["startup", "call_end"]
    events = [getattr(r, "event", None) for r in caplog.records]
    assert "startup hook failed" in events and "call_end hook failed" in events


def test_call_end_hooks_are_for_agentcore_only() -> None:
    rig = make_rig(script=[say("Hi.")])

    async def ended() -> None:
        return None

    with pytest.raises(ValueError):
        build_app(rig.tac, rig.voice, rig.bridge, on_call_end=[ended])
