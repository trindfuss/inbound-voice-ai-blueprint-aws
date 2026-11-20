"""scripts/bench/handshake_timer.py (handshake p50/p90): the dry run makes no
call; the apply path times new and pre-warmed sessions against a local WebSocket server with a fake
Runtime client; and a leak test over everything it prints and logs. Nothing here reaches AWS. No PBT
(I/O glue).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

websockets = pytest.importorskip("websockets")
from websockets.asyncio.client import connect as ws_connect  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402
from websockets.datastructures import Headers  # noqa: E402
from websockets.http11 import Response  # noqa: E402

from lambdas.frontdoor.relay import RuntimeUrlSigner, SigningCredentials  # noqa: E402
from lambdas.runtime_warmer.handler import PREWARM_PAYLOAD  # noqa: E402
from owlbank.sids import CALL_SID_RE  # noqa: E402
from scripts._common import ScriptError  # noqa: E402
from scripts.bench import handshake_timer  # noqa: E402
from tests.logutil import text_of  # noqa: E402

REGION = "us-east-1"
RUNTIME_ARN = f"arn:aws:bedrock-agentcore:{REGION}:123456789012:runtime/owlbank_dev_bridge-AbC123"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET = "aws-secret-key-0123456789abcdefghij"
AWS_TOKEN = "aws-session-token-ZZZZ-0123456789"
SECRET_VALUES = (AWS_SECRET, AWS_TOKEN)


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    path = tmp_path / "cdk-outputs.json"
    path.write_text(json.dumps({"OwlBankBridge": {"BridgeHosting": "agentcore", "RuntimeArn": RUNTIME_ARN}}))
    return path


def forbidden(*_: Any, **__: Any) -> Any:
    raise AssertionError("a dry run must make no AWS call")


class FakeRuntime:
    def __init__(self) -> None:
        self.invokes: list[dict[str, Any]] = []

    def invoke_agent_runtime(self, **kw: Any) -> dict[str, Any]:
        self.invokes.append(kw)
        return {"response": SimpleNamespace(read=lambda: b"{}")}


class FakeSession:
    """A boto3.Session stand-in: refreshable-looking credentials, a fake Runtime client."""

    def __init__(self) -> None:
        self.runtime = FakeRuntime()
        frozen = SimpleNamespace(access_key=AWS_KEY, secret_key=AWS_SECRET, token=AWS_TOKEN)
        self.creds = SimpleNamespace(get_frozen_credentials=lambda: frozen,
                                     _expiry_time=datetime.now(UTC) + timedelta(minutes=50))

    def get_credentials(self) -> Any:
        return self.creds

    def client(self, name: str) -> Any:
        return {"bedrock-agentcore": self.runtime}[name]


class LocalServer:
    """A websockets server on 127.0.0.1 in its own thread. ``mode``: ``accept`` (accept and wait for
    the client's close) or ``reject`` (HTTP 403 at the handshake)."""

    def __init__(self) -> None:
        self.mode = "accept"
        self.paths: list[str] = []
        self.loop = asyncio.new_event_loop()
        self.port = 0
        self._ready = threading.Event()
        self._stop: asyncio.Future[None] | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    async def _handler(self, ws: Any) -> None:
        self.paths.append(ws.request.path)
        await ws.wait_closed()

    def _process_request(self, ws: Any, request: Any) -> Any:
        if self.mode == "reject":
            self.paths.append(request.path)
            return Response(403, "Forbidden", Headers(), b"no")
        return None

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)

        async def main() -> None:
            async with serve(self._handler, "127.0.0.1", 0, process_request=self._process_request) as server:
                self.port = server.sockets[0].getsockname()[1]
                self._stop = self.loop.create_future()
                self._ready.set()
                await self._stop
        try:
            self.loop.run_until_complete(main())
        finally:
            self.loop.close()

    def start(self) -> LocalServer:
        self.thread.start()
        assert self._ready.wait(5)
        return self

    def stop(self) -> None:
        assert self._stop is not None
        self.loop.call_soon_threadsafe(self._stop.set_result, None)
        self.thread.join(5)

    def connector(self, seen: list[str]) -> Any:
        """``connect(url)``: records the signed URL, connects to this server with its path and query."""
        def connect(url: str) -> Any:
            seen.append(url)
            parts = urlsplit(url)
            return ws_connect(f"ws://127.0.0.1:{self.port}{parts.path}?{parts.query}", open_timeout=5)
        return connect


@pytest.fixture
def server() -> Iterator[LocalServer]:
    srv = LocalServer().start()
    yield srv
    srv.stop()


def captured(capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> str:
    out = capsys.readouterr()
    return out.out + out.err + text_of(caplog.records)


def assert_no_leak(text: str, *extra: str) -> None:
    for value in (*SECRET_VALUES, *extra):
        assert value not in text, "a credential, nonce or URL reached the output"
    for marker in ("X-Amz-Signature", "X-Amz-Credential", "X-Amz-Security-Token", "wss://", "ws://", "https://"):
        assert marker not in text


def test_dry_run_makes_no_call(outputs: Path, capsys: Any, caplog: Any) -> None:
    caplog.set_level(logging.DEBUG)
    assert handshake_timer.main(["--cdk-outputs", str(outputs)], session_factory=forbidden,
                                connect=forbidden) == 0
    assert "DRY RUN" in (text := captured(capsys, caplog))
    assert "owlbank_dev_bridge-AbC123" in text
    assert_no_leak(text)
    assert handshake_timer.main(["--cdk-outputs", str(outputs), "--count", "0"], session_factory=forbidden,
                                connect=forbidden) == 2


def test_needs_a_runtime_arn(tmp_path: Path, capsys: Any) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"OwlBankBridge": {"RelayDomain": "relay.owlbank.example"}}))
    assert handshake_timer.main(["--cdk-outputs", str(empty)], session_factory=forbidden, connect=forbidden) == 2
    assert "RuntimeArn" in capsys.readouterr().err


def test_apply_times_new_and_warm_sessions(server: LocalServer, outputs: Path, tmp_path: Path,
                                           capsys: Any, caplog: Any) -> None:
    caplog.set_level(logging.DEBUG)
    session, seen = FakeSession(), []
    out_file = tmp_path / "hs.json"
    assert handshake_timer.main(["--apply", "--cdk-outputs", str(outputs), "--count", "3", "--output",
                                 str(out_file)], session_factory=lambda region: session,
                                connect=server.connector(seen)) == 0
    doc = json.loads(out_file.read_text())
    for kind in ("new", "warm", "prewarm"):
        assert doc[kind]["count"] == 3 and doc[kind]["ok"] == 3 and doc[kind]["failures"] == {}
        assert doc[kind]["p50_ms"] is not None and doc[kind]["p90_ms"] >= doc[kind]["p50_ms"]
    ids = doc["session_ids"]["new"] + doc["session_ids"]["warm"]
    assert len(set(ids)) == 6 and all(CALL_SID_RE.match(s) for s in ids)
    assert [i["runtimeSessionId"] for i in session.runtime.invokes] == doc["session_ids"]["warm"]
    assert all(i["payload"] == PREWARM_PAYLOAD and i["agentRuntimeArn"] == RUNTIME_ARN
               for i in session.runtime.invokes)
    assert len(seen) == 6
    text = captured(capsys, caplog) + out_file.read_text()
    assert_no_leak(text, *seen)
    assert all("n" not in parse_qs(urlsplit(u).query) for u in seen)


def test_counts_failures(server: LocalServer) -> None:
    server.mode = "reject"
    signer = RuntimeUrlSigner(runtime_arn=RUNTIME_ARN, region=REGION, clock=lambda: 0.0,
                              credential_provider=lambda: SigningCredentials(AWS_KEY, AWS_SECRET, None, 1e6))
    res = handshake_timer.run_sessions(signer=signer, runtime_client=FakeRuntime(), runtime_arn=RUNTIME_ARN,
                                       count=2, connect=server.connector([]))
    assert res["new"]["ok"] == 0 and res["new"]["failures"] == {"InvalidStatus/403": 2}
    assert res["new"]["p50_ms"] is None


def test_percentile() -> None:
    assert handshake_timer.percentile([], 0.5) is None
    assert handshake_timer.percentile([5.0], 0.9) == 5.0
    assert handshake_timer.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
    assert handshake_timer.percentile([float(i) for i in range(1, 11)], 0.9) == 9.1


def test_operator_credentials_use_the_session_expiry() -> None:
    session = FakeSession()
    creds = handshake_timer.operator_credential_provider(session, lambda: 0.0)()
    assert creds.access_key == AWS_KEY and creds.expires_at == session.creds._expiry_time.timestamp()
    assert AWS_SECRET not in repr(creds)
    static = SimpleNamespace(get_credentials=lambda: SimpleNamespace(get_frozen_credentials=lambda: SimpleNamespace(
        access_key="k", secret_key="s", token=None)))
    assert handshake_timer.operator_credential_provider(static, lambda: 100.0)().expires_at == 3700.0
    with pytest.raises(ScriptError):
        handshake_timer.operator_credential_provider(SimpleNamespace(get_credentials=lambda: None), lambda: 0.0)()


def test_error_code_names_the_class_and_aws_code_only() -> None:
    exc = Exception("secret detail")
    exc.response = {"Error": {"Code": "ThrottlingException", "Message": "secret detail"}}  # type: ignore[attr-defined]
    assert handshake_timer.error_code(exc) == "Exception/ThrottlingException"
    assert handshake_timer.error_code(ValueError("x")) == "ValueError"
