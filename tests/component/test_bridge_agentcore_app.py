"""The bridge app with ``hosting="agentcore"``.

Routes (``/ping``, ``/invocations``, no ``/webhook``), no Twilio signature on ``/ws`` (the call
binding and nonce check instead), ``/ping`` following open and closed relay sockets, the ``ws_signature_check``
start-up line, and ``build_app_from_env`` reading the Twilio secrets by ARN before TAC.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from owlbank import secrets as owl_secrets
from tac_bridge.app import ForwardedHeadersMiddleware, build_app, build_app_from_env
from tac_bridge.hosting import SESSION_ID_HEADER, SocketTracker
from tests.component.test_bridge_fakes import CALL_SID, make_rig, relay_nonce, say, write_call_state
from tests.logutil import lines as lines_of, text_of


def agentcore_client(script=None, tracker: SocketTracker | None = None):
    rig = make_rig(script=script or [say("Hello from the bridge.")])
    app = build_app(rig.tac, rig.voice, rig.bridge, hosting="agentcore", tracker=tracker)
    return TestClient(app), rig


def test_routes_and_no_forwarded_headers_middleware() -> None:
    client, _ = agentcore_client()
    paths = {getattr(r, "path", None) for r in client.app.routes}
    assert paths == {"/health", "/ws", "/ping", "/invocations"}
    assert all(m.cls is not ForwardedHeadersMiddleware for m in client.app.user_middleware)
    for path in ("/webhook", "/twiml", "/docs", "/openapi.json"):
        assert client.post(path).status_code in (404, 405)
    assert client.get("/health").json() == {"ok": True}


def test_fargate_is_the_default_and_unchanged() -> None:
    rig = make_rig(script=[say("Hi.")])
    app = build_app(rig.tac, rig.voice, rig.bridge)
    assert {getattr(r, "path", None) for r in app.routes} == {"/health", "/ws", "/webhook"}
    assert [m.cls for m in app.user_middleware] == [ForwardedHeadersMiddleware]


def test_agentcore_needs_no_public_domain_or_auth_token() -> None:
    rig = make_rig(script=[say("Hi.")])
    rig.tac.config.voice_public_domain = None
    rig.tac.config.auth_token = ""
    build_app(rig.tac, rig.voice, rig.bridge, hosting="agentcore")
    with pytest.raises(ValueError):
        build_app(rig.tac, rig.voice, rig.bridge)  # Fargate still requires them


def test_unknown_hosting_is_rejected() -> None:
    rig = make_rig(script=[say("Hi.")])
    with pytest.raises(ValueError):
        build_app(rig.tac, rig.voice, rig.bridge, hosting="lambda")  # type: ignore[arg-type]


def test_startup_line_says_the_signature_check_is_off(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    agentcore_client()
    found = [ln for ln in lines_of(caplog.records, "tac_bridge.app") if ln["event"] == "ws_signature_check"]
    assert len(found) == 1
    assert found[0]["state"] == "off" and found[0]["reason"] == "agentcore_hosting"


def test_fargate_has_no_signature_check_line(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    rig = make_rig(script=[say("Hi.")])
    build_app(rig.tac, rig.voice, rig.bridge)
    assert not [ln for ln in lines_of(caplog.records, "tac_bridge.app") if ln["event"] == "ws_signature_check"]


# --------------------------------------------------------------------------- /invocations


@pytest.mark.parametrize("payload", [{"action": "prewarm"}, {"action": "prewarm", "extra": 1}])
def test_prewarm_is_ready_and_touches_nothing(payload: dict[str, Any]) -> None:
    client, rig = agentcore_client()
    write_call_state(rig.store)
    before = json.dumps(rig.store.get("calls", {"call_sid": CALL_SID}), sort_keys=True, default=str)
    r = client.post("/invocations", json=payload,
                    headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": CALL_SID})
    assert r.status_code == 200 and r.json() == {"status": "ready"}
    assert rig.agent.calls == [] and rig.backend.calls == []
    assert json.dumps(rig.store.get("calls", {"call_sid": CALL_SID}), sort_keys=True, default=str) == before
    assert rig.bridge._sessions == {}


@pytest.mark.parametrize("body", [b"", b"not json", b"[]", b'"prewarm"', b"{}", b'{"action": "PREWARM"}',
                                  b'{"action": "orchestrator_webhook", "body": {}}', b'{"action": null}'])
def test_anything_else_is_400(body: bytes, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    client, rig = agentcore_client()
    r = client.post("/invocations", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json() == {"status": "error"}
    assert rig.agent.calls == [] and rig.backend.calls == []
    if body:
        assert body.decode() not in text_of(caplog.records).replace("{}", "")  # the body is never logged


# --------------------------------------------------------------------------- /ws and /ping


def _wait_ping(client: TestClient, status: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while True:
        body = client.get("/ping").json()
        if body["status"] == status or time.monotonic() > deadline:
            return body
        time.sleep(0.01)


def test_ping_follows_relay_sockets() -> None:
    now = [1_800_000_000.0]
    tracker = SocketTracker(clock=lambda: now[0])
    client, rig = agentcore_client([say("Hello ", "Thomas.")], tracker=tracker)
    write_call_state(rig.store)
    with client:
        assert client.get("/ping").json() == {"status": "Healthy", "time_of_last_update": 1_800_000_000}
        now[0] += 5
        # No X-Twilio-Signature: AgentCore authenticates the upgrade; the nonce still applies.
        with client.websocket_connect("/ws", headers={SESSION_ID_HEADER: CALL_SID}) as ws:
            ws.send_json({"type": "setup", "callSid": CALL_SID, "from": "+12025550101", "to": "+12025550199",
                          "customParameters": {"callSid": CALL_SID, "n": relay_nonce()}})
            ws.send_json({"type": "prompt", "voicePrompt": "hi", "lang": "en-US", "last": True})
            frames = [json.loads(ws.receive_text()) for _ in range(3)]
            assert client.get("/ping").json() == {"status": "HealthyBusy",
                                                  "time_of_last_update": 1_800_000_005}
            now[0] += 5
        assert _wait_ping(client, "Healthy") == {"status": "Healthy", "time_of_last_update": 1_800_000_010}
    assert frames[-1] == {"type": "text", "token": "", "last": True}
    assert rig.agent.calls[0]["session_id"] == CALL_SID
    assert tracker.open_count == 0


def test_rejected_socket_still_returns_to_healthy() -> None:
    tracker = SocketTracker()
    client, rig = agentcore_client([say("Hi.")], tracker=tracker)
    write_call_state(rig.store)
    with client:
        for params in ({}, {"n": "wrong-nonce"}, {"n": relay_nonce()}):  # missing, wrong; then no session header
            headers = {SESSION_ID_HEADER: CALL_SID} if params.get("n") != relay_nonce() else {}
            with pytest.raises(WebSocketDisconnect) as ei, client.websocket_connect("/ws", headers=headers) as ws:
                ws.send_json({"type": "setup", "callSid": CALL_SID,
                              "customParameters": {"callSid": CALL_SID, **params}})
                ws.receive_text()
            assert ei.value.code == 1000
        assert _wait_ping(client, "Healthy")["status"] == "Healthy"
    assert rig.agent.calls == [] and tracker.open_count == 0


# --------------------------------------------------------------------------- build_app_from_env


AGENT_ENV = {
    "OWLBANK_HOSTING": "agentcore",
    "TWILIO_CONVERSATION_CONFIGURATION_ID": "conv_configuration_x",
    "OWLBANK_BANK_TOOLS_FUNCTION": "owlbank-dev-bank-tools",
    "OWLBANK_TOKEN_KMS_KEY": "alias/owlbank-test",
    # stream needs thinking adaptive or default
    "OWLBANK_MODELS": json.dumps([{"id": "us.anthropic.claude-test", "thinking": "off", "max_tokens": 512,
                                   "text_release": "message", "prompt_caching": True}], separators=(",", ":")),
    "OWLBANK_GATEWAY_URL": "https://gateway.example/mcp",
    "OWLBANK_MEMORY_ID": "mem-test",
    "AWS_REGION": "us-east-1",
}
API_ARN = "arn:aws:secretsmanager:us-east-1:111111111111:secret:owlbank/dev/twilio-api-key-AbCdEf"
TOKEN_ARN = "arn:aws:secretsmanager:us-east-1:111111111111:secret:owlbank/dev/twilio-auth-token-GhIjKl"
SECRET_VALUES = {"account_sid": "AC" + "d" * 32, "api_key": "SK" + "d" * 32,
                 "api_secret": "FromSmApiSecretZz" + "z" * 15, "auth_token": "fa11" * 8}
TAC_NAMES = ("TWILIO_ACCOUNT_SID", "TWILIO_API_KEY", "TWILIO_API_SECRET", "TWILIO_AUTH_TOKEN")


def test_bad_hosting_fails_at_start_up() -> None:
    with pytest.raises(RuntimeError, match="^OWLBANK_HOSTING must be one of"):
        build_app_from_env({**AGENT_ENV, "OWLBANK_HOSTING": "ecs"})


@pytest.mark.parametrize("missing", ["OWLBANK_SECRET_TWILIO_API_KEY", "OWLBANK_SECRET_TWILIO_AUTH_TOKEN"])
def test_agentcore_missing_secret_arn_names_the_variable(missing: str, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[Any] = []
    monkeypatch.setattr("tac_bridge.app.build_strands_client", lambda s: started.append(s))
    import boto3

    reads: list[str] = []

    class NoReads:
        def get_secret_value(self, *, SecretId: str) -> dict[str, str]:  # noqa: N803
            reads.append(SecretId)
            return {"SecretString": "{}"}

    monkeypatch.setattr(boto3, "client", lambda service, **kw: NoReads())
    env = {**AGENT_ENV, "OWLBANK_SECRET_TWILIO_API_KEY": API_ARN, "OWLBANK_SECRET_TWILIO_AUTH_TOKEN": TOKEN_ARN}
    del env[missing]
    # TWILIO_VOICE_PUBLIC_DOMAIN is absent: not required in agentcore.
    with pytest.raises(RuntimeError, match=f"^{missing} is required$"):
        build_app_from_env(env)
    assert started == [] and reads == []  # failed before the agent client started or any secret read


def test_agentcore_reads_secrets_into_the_environment_before_tac(monkeypatch: pytest.MonkeyPatch,
                                                                 caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    owl_secrets.clear_cache()
    for name in TAC_NAMES:
        monkeypatch.setenv(name, "placeholder")  # restored (removed) after the test

    class FakeSM:
        def get_secret_value(self, *, SecretId: str) -> dict[str, str]:  # noqa: N803
            if SecretId == API_ARN:
                return {"SecretString": json.dumps({k: SECRET_VALUES[k]
                                                    for k in ("account_sid", "api_key", "api_secret")})}
            return {"SecretString": json.dumps({"auth_token": SECRET_VALUES["auth_token"]})}

    import boto3

    real_client = boto3.client
    monkeypatch.setattr(boto3, "client", lambda service, **kw: FakeSM() if service == "secretsmanager"
                        else real_client(service, **kw))
    monkeypatch.setattr("tac_bridge.app.build_strands_client", lambda s: None)
    seen: dict[str, str] = {}

    class Stop(Exception):
        pass

    def from_env(*_a: Any, **_k: Any) -> Any:
        import os

        seen.update({n: os.environ[n] for n in TAC_NAMES})
        raise Stop

    monkeypatch.setattr("tac.TACConfig.from_env", from_env)
    env = {**AGENT_ENV, "OWLBANK_SECRET_TWILIO_API_KEY": API_ARN, "OWLBANK_SECRET_TWILIO_AUTH_TOKEN": TOKEN_ARN}
    try:
        with pytest.raises(Stop):
            build_app_from_env(env)
    finally:
        owl_secrets.clear_cache()
    assert seen == {"TWILIO_ACCOUNT_SID": SECRET_VALUES["account_sid"], "TWILIO_API_KEY": SECRET_VALUES["api_key"],
                    "TWILIO_API_SECRET": SECRET_VALUES["api_secret"],
                    "TWILIO_AUTH_TOKEN": SECRET_VALUES["auth_token"]}
    logged = text_of(caplog.records)
    assert all(v not in logged for v in SECRET_VALUES.values())


def test_fargate_still_requires_the_public_domain() -> None:
    env = {k: v for k, v in AGENT_ENV.items() if k != "OWLBANK_HOSTING"}
    with pytest.raises(RuntimeError, match="^TWILIO_VOICE_PUBLIC_DOMAIN is required$"):
        build_app_from_env(env)
