"""The bridge's entry point serves TLS on its port (Fargate) and plain HTTP on AgentCore.

A real uvicorn server on a free local port, built by ``tac_bridge.serve.make_server`` exactly as the
container builds it, with a small app standing in for ``build_app_from_env``: ``/health`` answers
over HTTPS (self-signed, so the client doesn't verify), and a WebSocket upgrades over ``wss``.
"""

from __future__ import annotations

import asyncio
import os
import ssl
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest
import websockets
from cryptography import x509
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from fastapi import FastAPI, WebSocket

from tac_bridge import serve

APP = "tests.component.test_bridge_tls_serve:build_test_app"


def build_test_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        await sock.send_text("hello " + await sock.receive_text())
        await sock.close()

    return app


@contextmanager
def running(env: dict[str, str]) -> Iterator[tuple[object, int]]:
    server = serve.make_server(env, app=APP, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not (server.started and server.servers):
        assert time.monotonic() < deadline, "server didn't start"
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield server, port
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_fargate_serves_health_over_https_and_refuses_plain_http():
    with running({}) as (server, port):
        assert server.config.ssl is not None
        resp = httpx.get(f"https://127.0.0.1:{port}/health", verify=False)  # noqa: S501 (self-signed)
        assert resp.status_code == 200 and resp.json() == {"status": "ok"}
        assert "server" not in resp.headers
        with pytest.raises(httpx.HTTPError):
            httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
        assert serve.healthcheck({}, port=port) == 0


def test_fargate_websocket_upgrades_over_wss():
    ctx = ssl._create_unverified_context()  # noqa: S323 (self-signed)

    async def talk(port: int) -> str:
        async with websockets.connect(f"wss://127.0.0.1:{port}/ws", ssl=ctx) as sock:
            await sock.send("bridge")
            return str(await sock.recv())

    with running({"OWLBANK_HOSTING": "fargate"}) as (_, port):
        assert asyncio.run(talk(port)) == "hello bridge"


def test_agentcore_serves_plain_http():
    with running({"OWLBANK_HOSTING": "agentcore"}) as (server, port):
        assert server.config.ssl is None
        assert httpx.get(f"http://127.0.0.1:{port}/health").status_code == 200
        assert serve.healthcheck({"OWLBANK_HOSTING": "agentcore"}, port=port) == 0


def test_server_options_match_the_former_command_line():
    server = serve.make_server({}, app=APP, host="127.0.0.1", port=0)
    cfg = server.config
    assert (cfg.ws_ping_interval, cfg.ws_ping_timeout) == (20.0, 20.0)
    assert cfg.server_header is False and cfg.access_log is False


def test_healthcheck_fails_when_nothing_listens():
    assert serve.healthcheck({}, port=1) == 1


def test_self_signed_pem_is_a_matching_key_and_certificate():
    pem = serve.self_signed_pem()
    cert = x509.load_pem_x509_certificate(pem[pem.index(b"-----BEGIN CERTIFICATE-----"):])
    key = load_pem_private_key(pem, password=None)
    assert cert.public_key().public_numbers() == key.public_key().public_numbers()  # type: ignore[union-attr]
    assert cert.subject == cert.issuer


def test_pem_path_leaves_nothing_behind():
    with serve.pem_path(b"pem bytes") as path:
        with open(path, "rb") as fh:
            assert fh.read() == b"pem bytes"
        if not path.startswith("/proc/self/fd/"):
            assert (os.stat(path).st_mode & 0o777) == 0o600
    assert not os.path.exists(path)


def test_two_contexts_get_different_certificates():
    assert serve.self_signed_pem() != serve.self_signed_pem()
