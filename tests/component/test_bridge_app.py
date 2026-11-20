"""Our FastAPI app over real TAC: routes, signature validation, ALB headers, a signed WS call."""

from __future__ import annotations

from tests.logutil import lines as lines_of, text_of

import hashlib
import json
import os
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from twilio.request_validator import RequestValidator

from tests.component.test_bridge_fakes import (
    AUTH_TOKEN,
    CALL_SID,
    PUBLIC_DOMAIN,
    make_rig,
    relay_nonce,
    say,
    write_call_state,
)

from tac_bridge.app import ForwardedHeadersMiddleware, build_app, build_app_from_env

VALIDATOR = RequestValidator(AUTH_TOKEN)
WS_URL = f"wss://{PUBLIC_DOMAIN}/ws"


def client_and_rig(script=None):
    rig = make_rig(script=script or [say("Hello from the bridge.")])
    rig.bridge._io = __import__("asyncio").to_thread  # production I/O path (worker thread)
    app = build_app(rig.tac, rig.voice, rig.bridge)
    return TestClient(app), rig


def test_analytics_disabled_and_only_our_routes_exist() -> None:
    assert os.environ["TAC_ANALYTICS_DISABLED"] == "true"
    client, _ = client_and_rig()
    assert client.get("/health").json() == {"ok": True}
    for path in ("/twiml", "/docs", "/redoc", "/openapi.json", "/conversation-relay-callback",
                 "/twilio/call-events/status"):
        assert client.post(path).status_code in (404, 405)
        assert client.get(path).status_code in (404, 405)
    paths = {getattr(r, "path", None) for r in client.app.routes}
    assert paths == {"/health", "/ws", "/webhook"}


def test_webhook_requires_a_valid_signature() -> None:
    client, _ = client_and_rig()
    body = json.dumps({"eventType": "CONVERSATION_UPDATED", "data": {"id": "conv_x", "status": "CLOSED"}})
    assert client.post("/webhook", content=body, headers={"content-type": "application/json"}).status_code == 403
    sha = hashlib.sha256(body.encode()).hexdigest()
    url = f"https://{PUBLIC_DOMAIN}/webhook?bodySHA256={sha}"
    sig = VALIDATOR.compute_signature(url, {})
    ok = client.post(f"/webhook?bodySHA256={sha}", content=body,
                     headers={"content-type": "application/json", "X-Twilio-Signature": sig,
                              "X-Forwarded-Host": "evil.example"})  # overwritten by our middleware
    assert ok.status_code == 200 and ok.json() == {"status": "ok"}
    bad = client.post(f"/webhook?bodySHA256={sha}", content=body + " ",
                      headers={"content-type": "application/json", "X-Twilio-Signature": sig})
    assert bad.status_code == 403


def test_unsigned_websocket_is_closed_before_accept() -> None:
    client, rig = client_and_rig()
    for headers in ({}, {"X-Twilio-Signature": "bm90LWEtc2lnbmF0dXJl"}):
        with pytest.raises(WebSocketDisconnect) as ei, client.websocket_connect("/ws", headers=headers) as ws:
            ws.receive_text()
        assert ei.value.code == 1008
    assert rig.agent.calls == []


def _signed(nonce: str, *, full_url: bool = True) -> tuple[str, dict[str, str]]:
    """The ``/ws?n=`` path and headers. ``full_url``: Twilio signs ``wss://host/ws?n=...``;
    otherwise TAC's convention (URL without query, query as parameters)."""
    query = urlencode({"n": nonce})
    sig = (VALIDATOR.compute_signature(f"{WS_URL}?{query}", {}) if full_url
           else VALIDATOR.compute_signature(WS_URL, {"n": nonce}))
    return f"/ws?{query}", {"X-Twilio-Signature": sig}


@pytest.mark.parametrize("full_url", [True, False])
def test_signed_websocket_call_end_to_end(full_url: bool) -> None:
    client, rig = client_and_rig([say("Hello ", "Thomas.")])
    write_call_state(rig.store)
    path, headers = _signed(relay_nonce(), full_url=full_url)
    with client.websocket_connect(path, headers=headers) as ws:
        ws.send_json({"type": "setup", "callSid": CALL_SID, "from": "+12025550101", "to": "+12025550199",
                      "customParameters": {"callSid": CALL_SID}})
        ws.send_json({"type": "prompt", "voicePrompt": "hi", "lang": "en-US", "last": True})
        frames = [json.loads(ws.receive_text()) for _ in range(3)]
    assert frames == [{"type": "text", "token": "Hello ", "last": False},
                      {"type": "text", "token": "Thomas.", "last": False},
                      {"type": "text", "token": "", "last": True}]
    assert rig.agent.calls[0]["session_id"] == CALL_SID


def test_forwarded_headers_are_overwritten() -> None:
    seen = {}

    async def inner(scope, receive, send):
        seen.update({k.decode(): v.decode() for k, v in scope["headers"]})

    mw = ForwardedHeadersMiddleware(inner, PUBLIC_DOMAIN)
    __import__("asyncio").run(mw({"type": "websocket", "headers": [
        (b"x-forwarded-host", b"evil.example"), (b"x-forwarded-proto", b"http"), (b"host", b"10.0.0.5")]},
        None, None))
    assert seen["x-forwarded-host"] == PUBLIC_DOMAIN and seen["x-forwarded-proto"] == "https"
    with pytest.raises(ValueError):
        ForwardedHeadersMiddleware(inner, "")


def test_build_app_from_env_requires_orchestrator_and_bridge_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="TWILIO_CONVERSATION_CONFIGURATION_ID"):
        build_app_from_env({})
    env = {"TWILIO_CONVERSATION_CONFIGURATION_ID": "conv_configuration_x",
           "TWILIO_VOICE_PUBLIC_DOMAIN": PUBLIC_DOMAIN, "OWLBANK_HARNESS_ARN": "arn:x"}
    with pytest.raises(RuntimeError, match="OWLBANK_BANK_TOOLS_FUNCTION"):
        build_app_from_env(env)


def _setup_and_prompt(client: TestClient, path: str, headers: dict[str, str], call_sid: str = CALL_SID):
    """Open the socket, send ``setup`` and one prompt; return the frames, or the close code."""
    try:
        with client.websocket_connect(path, headers=headers) as ws:
            ws.send_json({"type": "setup", "callSid": call_sid, "from": "+12025550101",
                          "to": "+12025550199", "customParameters": {"callSid": call_sid}})
            ws.send_json({"type": "prompt", "voicePrompt": "pay my card", "lang": "en-US", "last": True})
            return [json.loads(ws.receive_text()) for _ in range(2)]
    except WebSocketDisconnect as exc:
        return exc.code


def test_relay_nonce_is_single_use() -> None:
    """A replayed socket (same signed URL, same nonce) is closed and runs nothing."""
    client, rig = client_and_rig([say("Hi.")])
    write_call_state(rig.store)
    path, headers = _signed(relay_nonce())
    assert isinstance(_setup_and_prompt(client, path, headers), list)
    calls = len(rig.agent.calls)
    assert rig.store.get("calls", {"call_sid": CALL_SID})["ws_nonce_hash"] is None  # consumed
    assert _setup_and_prompt(client, path, headers) == 1000  # closed by the bridge
    assert len(rig.agent.calls) == calls and rig.backend.calls == []


def test_relay_nonce_for_another_call_is_rejected() -> None:
    other = "CA" + "2" * 32
    client, rig = client_and_rig([say("Hi.")])
    write_call_state(rig.store)
    write_call_state(rig.store, other)
    path, headers = _signed(relay_nonce(other))  # a valid nonce, but for the attacker's own call
    assert _setup_and_prompt(client, path, headers, CALL_SID) == 1000
    assert rig.agent.calls == [] and rig.backend.calls == []
    # Neither nonce was consumed by the failed attempt.
    assert rig.store.get("calls", {"call_sid": CALL_SID})["ws_nonce_hash"]
    assert rig.store.get("calls", {"call_sid": other})["ws_nonce_hash"]


@pytest.mark.parametrize("setup", [
    {"type": "setup", "from": "+12025550101"},  # callSid missing
    {"type": "setup", "callSid": None},
    {"type": "setup", "callSid": ""},
    {"type": "setup", "callSid": 12345},  # not a string
])
def test_relay_setup_without_a_usable_call_sid_closes_the_socket(setup: dict) -> None:
    """A setup frame with no string callSid is closed, never passed on as ``None``."""
    client, rig = client_and_rig([say("Hi.")])
    write_call_state(rig.store)
    path, headers = _signed(relay_nonce())
    with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect(path, headers=headers) as ws:
        ws.send_json(setup)
        ws.send_json({"type": "prompt", "voicePrompt": "pay my card", "lang": "en-US", "last": True})
        ws.receive_text()
    assert exc.value.code == 1000
    assert rig.bridge._sessions == {}  # session_start was never called
    assert rig.agent.calls == [] and rig.backend.calls == []
    assert rig.store.get("calls", {"call_sid": CALL_SID})["ws_nonce_hash"]  # the nonce was not consumed


def test_relay_missing_nonce_or_unknown_call_is_rejected() -> None:
    client, rig = client_and_rig([say("Hi.")])
    write_call_state(rig.store)
    sig = VALIDATOR.compute_signature(WS_URL, {})
    assert _setup_and_prompt(client, "/ws", {"X-Twilio-Signature": sig}) == 1000  # no nonce
    path, headers = _signed(relay_nonce("CA" + "9" * 32))
    assert _setup_and_prompt(client, path, headers, "CA" + "9" * 32) == 1000  # no call state
    path, headers = _signed("not-the-nonce")
    assert _setup_and_prompt(client, path, headers) == 1000  # wrong nonce
    assert rig.agent.calls == []


def test_tampered_nonce_breaks_the_signature() -> None:
    client, rig = client_and_rig([say("Hi.")])
    write_call_state(rig.store)
    _, headers = _signed(relay_nonce())
    assert _setup_and_prompt(client, "/ws?n=other", headers) == 1008


def auth_failed_reasons(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [ln["reason"] for ln in lines_of(caplog.records, "owlbank.security")
            if ln["event"] == "twilio_auth_failed" and ln["component"] == "bridge"]


def test_failed_signature_with_a_nonce_never_logs_the_nonce(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level(logging.DEBUG)
    client, rig = client_and_rig([say("Hi.")])
    write_call_state(rig.store)
    nonce = relay_nonce()
    path, _ = _signed(nonce)
    assert _setup_and_prompt(client, path, {"X-Twilio-Signature": "bm90LWEtc2lnbmF0dXJl"}) == 1008
    assert _setup_and_prompt(client, path, {}) == 1008
    # TAC's own line, if any version logs it, is redacted too.
    logging.getLogger("tac.server.signature_validation").warning(
        "WebSocket signature validation failed. URL used: %s", f"{WS_URL}?n={nonce}")
    texts = [r.getMessage() + str(r.args) + str(getattr(r, "__dict__", {})) for r in caplog.records]
    assert texts and all(nonce not in t for t in texts)
    assert auth_failed_reasons(caplog) == ["signature", "signature"]


def test_nonce_and_webhook_failures_emit_the_stable_line(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level(logging.WARNING)
    client, rig = client_and_rig([say("Hi.")])
    write_call_state(rig.store)
    path, headers = _signed("wrong-nonce")
    assert _setup_and_prompt(client, path, headers) == 1000
    body = json.dumps({"eventType": "CONVERSATION_UPDATED", "data": {"id": "conv_x"}})
    assert client.post("/webhook", content=body, headers={"content-type": "application/json"}).status_code == 403
    reasons = auth_failed_reasons(caplog)
    assert "nonce" in reasons and "signature" in reasons
    assert "wrong-nonce" not in text_of(caplog.records)


# --------------------------------------------------------------------------- webhook de-duplication

WEBHOOK_BODY = json.dumps({"eventType": "CONVERSATION_UPDATED", "data": {"id": "conv_x", "status": "CLOSED"}})


def _post_signed(client: TestClient, body: str, headers: dict[str, str] | None = None):
    sha = hashlib.sha256(body.encode()).hexdigest()
    sig = VALIDATOR.compute_signature(f"https://{PUBLIC_DOMAIN}/webhook?bodySHA256={sha}", {})
    return client.post(f"/webhook?bodySHA256={sha}", content=body,
                       headers={"content-type": "application/json", "X-Twilio-Signature": sig,
                                **(headers or {})})


def _recording_client(idempotency=None):
    rig = make_rig(script=[say("Hello from the bridge.")])
    seen: list[str | None] = []

    async def process_webhook(data, token=None):
        seen.append(token)

    rig.voice.process_webhook = process_webhook
    app = build_app(rig.tac, rig.voice, rig.bridge, idempotency=idempotency)
    return TestClient(app), seen


def test_a_repeated_idempotency_token_reaches_tac_once(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level(logging.INFO)
    client, seen = _recording_client()
    token = "idem-secret-token-123"
    with client:
        first = _post_signed(client, WEBHOOK_BODY, {"I-Twilio-Idempotency-Token": token})
        again = _post_signed(client, WEBHOOK_BODY, {"i-twilio-idempotency-token": token})  # any case
        third = _post_signed(client, WEBHOOK_BODY, {"I-TWILIO-IDEMPOTENCY-TOKEN": token})
        other = _post_signed(client, WEBHOOK_BODY, {"I-Twilio-Idempotency-Token": "idem-other"})
    assert [r.status_code for r in (first, again, third, other)] == [200, 200, 200, 200]
    assert again.json() == {"status": "ok"}
    assert seen == [token, "idem-other"]
    dups = [ln for ln in lines_of(caplog.records, "tac_bridge.app") if ln["event"] == "webhook_duplicate"]
    assert len(dups) == 2 and all(len(d["token_ref"]) == 8 for d in dups)
    assert token not in text_of(caplog.records)


def test_a_webhook_without_a_token_is_always_processed() -> None:
    client, seen = _recording_client()
    with client:
        for _ in range(2):
            assert _post_signed(client, WEBHOOK_BODY).status_code == 200
        assert _post_signed(client, WEBHOOK_BODY, {"I-Twilio-Idempotency-Token": ""}).status_code == 200
    assert seen == [None, None, None]


def test_an_expired_token_is_processed_again() -> None:
    from tac_bridge.app import IdempotencyCache

    now = [0.0]
    client, seen = _recording_client(IdempotencyCache(clock=lambda: now[0]))
    with client:
        _post_signed(client, WEBHOOK_BODY, {"I-Twilio-Idempotency-Token": "tok"})
        now[0] = 599.0
        _post_signed(client, WEBHOOK_BODY, {"I-Twilio-Idempotency-Token": "tok"})
        now[0] = 600.0
        _post_signed(client, WEBHOOK_BODY, {"I-Twilio-Idempotency-Token": "tok"})
    assert seen == ["tok", "tok"]
