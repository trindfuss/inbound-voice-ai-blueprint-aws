"""The early-finish Twilio client is built at INIT under provisioned concurrency or
SnapStart, lazily otherwise; a failure at INIT is logged without secret values and falls back."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from lambdas.frontdoor import calls, handler
from lambdas.frontdoor.calls import TwilioCallUpdater
from tests.logutil import lines, text_of
from tests.unit.test_coldstart import aws_env  # noqa: F401 - fixture

SECRET = {"account_sid": "AC" + "0" * 32, "api_key": "SK" + "1" * 32, "api_secret": "S3cr3tValue" + "v" * 21}
ARN = "arn:aws:secretsmanager:us-east-1:1:secret:twilio-api-key"


@pytest.mark.parametrize("init_type,want", [
    ("provisioned-concurrency", True), ("snap-start", True), ("on-demand", False), (None, False),
])
def test_build_at_init_only_ahead_of_time(init_type: str | None, want: bool) -> None:
    env = {} if init_type is None else {"AWS_LAMBDA_INITIALIZATION_TYPE": init_type}
    assert calls.build_at_init(env) is want


def _warm_url() -> str:
    return calls.WARM_URL.format(account_sid=SECRET["account_sid"])


class _Resp:
    def __init__(self, status: int) -> None:
        self.status_code = status


@pytest.fixture
def app(aws_env, monkeypatch):  # noqa: F811
    """The real front door wiring, with the Twilio API key secret configured and its reads counted.
    The real ``twilio.rest.Client`` is built; its requests are recorded, never sent."""
    from twilio.rest import Client

    from owlbank import secrets

    reads: list[str] = []
    sent: list[tuple[str, str, float | None]] = []

    def request(self: Any, method: str, uri: str, *a: Any, timeout: float | None = None, **k: Any) -> _Resp:
        sent.append((method, uri, timeout))
        return _Resp(200)

    monkeypatch.setattr(Client, "request", request)

    def get_secret(arn: str, *, logical: str | None = None) -> dict[str, str]:
        assert logical == "twilio_api_key"  # the read is checked against the API key schema
        reads.append(arn)
        return dict(SECRET)

    monkeypatch.setattr(secrets, "get_secret", get_secret)
    monkeypatch.setenv("OWLBANK_SECRET_TWILIO_API_KEY", ARN)
    monkeypatch.setenv("OWLBANK_EARLY_DIGITS", "true")
    fd = handler.build_app()
    fd.reads = reads  # type: ignore[attr-defined]
    fd.sent = sent  # type: ignore[attr-defined]
    assert reads == []  # building the app reads no secret
    return fd


@pytest.mark.parametrize("init_type", ["provisioned-concurrency", "snap-start"])
def test_client_is_built_at_init_ahead_of_time(app, init_type: str) -> None:
    handler.prepare_early_finish(app, {"AWS_LAMBDA_INITIALIZATION_TYPE": init_type})
    assert app.reads == [ARN]
    client = app.d.calls._client  # noqa: SLF001
    assert client is not None and client.username == SECRET["api_key"]
    assert app.d.calls._get() is client and app.reads == [ARN]  # noqa: SLF001 - reused, not re-read
    # One authenticated account read at INIT opens the pooled connection
    assert app.sent == [("GET", _warm_url(),
                         calls.WARM_TIMEOUT_S)]


@pytest.mark.parametrize("env", [{"AWS_LAMBDA_INITIALIZATION_TYPE": "on-demand"}, {}])
def test_client_stays_lazy_otherwise(app, env: dict[str, str]) -> None:
    handler.prepare_early_finish(app, env)
    assert app.reads == [] and app.d.calls._client is None  # noqa: SLF001
    app.d.calls._get()  # noqa: SLF001 - the first early finish builds it
    assert app.reads == [ARN]
    assert app.sent == []  # no warm-up read on demand


def test_nothing_built_when_early_finish_is_off(aws_env, monkeypatch) -> None:  # noqa: F811
    from owlbank import secrets

    monkeypatch.setattr(secrets, "get_secret", lambda *a, **k: pytest.fail("secret read"))
    monkeypatch.delenv("OWLBANK_SECRET_TWILIO_API_KEY", raising=False)
    fd = handler.build_app()
    assert fd.d.calls is None
    handler.prepare_early_finish(fd, {"AWS_LAMBDA_INITIALIZATION_TYPE": "provisioned-concurrency"})


def test_init_failure_is_logged_without_secrets_and_falls_back_to_lazy(caplog) -> None:
    attempts: list[int] = []

    class _Client:
        pass

    def factory() -> Any:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError(f"cannot read {SECRET['api_secret']}")
        return _Client()

    upd = TwilioCallUpdater(factory)
    with caplog.at_level(logging.WARNING, logger="lambdas.frontdoor.calls"):
        assert upd.prebuild() is False
    recs = lines(caplog.records, "lambdas.frontdoor.calls")
    assert [r["event"] for r in recs] == ["twilio_client_init_failed"]
    assert recs[0]["decision"] == "build_on_first_use" and recs[0]["error_class"] == "RuntimeError"
    assert SECRET["api_secret"] not in text_of(caplog.records)
    assert upd._client is None  # noqa: SLF001
    assert isinstance(upd._get(), _Client) and len(attempts) == 2  # noqa: SLF001 - built on first use


def test_prepare_early_finish_never_raises_on_init_failure(app, monkeypatch) -> None:
    from owlbank import secrets

    def boom(arn: str, **kw: Any) -> Any:
        raise RuntimeError("secrets manager down")

    monkeypatch.setattr(secrets, "get_secret", boom)
    handler.prepare_early_finish(app, {"AWS_LAMBDA_INITIALIZATION_TYPE": "provisioned-concurrency"})
    assert app.d.calls._client is None  # noqa: SLF001


# --------------------------------------------------------------------------- connection warm-up


class _FakeClient:
    account_sid = SECRET["account_sid"]

    def __init__(self, result: Any = 200) -> None:
        self.result = result
        self.requests: list[tuple[str, str, float | None]] = []
        self.updates: list[tuple[str, str]] = []

    def request(self, method: str, uri: str, *, timeout: float | None = None) -> _Resp:
        self.requests.append((method, uri, timeout))
        if isinstance(self.result, BaseException):
            raise self.result
        return _Resp(self.result)

    def calls(self, sid: str) -> Any:
        client = self

        class _Call:
            def update(self, twiml: str) -> None:
                client.updates.append((sid, twiml))

        return _Call()


def _app_with(monkeypatch, client: _FakeClient) -> Any:
    fd = handler.build_app()
    monkeypatch.setattr(fd.d, "calls", TwilioCallUpdater(lambda: client))
    return fd


@pytest.mark.parametrize("init_type", ["provisioned-concurrency", "snap-start"])
def test_warm_read_once_at_init_ahead_of_time(aws_env, monkeypatch, caplog, init_type: str) -> None:  # noqa: F811
    monkeypatch.setenv("OWLBANK_EARLY_DIGITS", "true")
    monkeypatch.setenv("OWLBANK_SECRET_TWILIO_API_KEY", ARN)
    client = _FakeClient()
    fd = _app_with(monkeypatch, client)
    with caplog.at_level(logging.INFO, logger="lambdas.frontdoor.calls"):
        handler.prepare_early_finish(fd, {"AWS_LAMBDA_INITIALIZATION_TYPE": init_type})
    assert client.requests == [("GET", _warm_url(),
                                calls.WARM_TIMEOUT_S)]
    assert calls.WARM_TIMEOUT_S <= 2
    assert [r["event"] for r in lines(caplog.records, "lambdas.frontdoor.calls")] == ["twilio_connection_warmed"]
    fd.d.calls.update_twiml("CA1", "<Response/>")
    assert client.requests[1:] == [] and client.updates == [("CA1", "<Response/>")]  # once, at INIT only


@pytest.mark.parametrize("env", [{"AWS_LAMBDA_INITIALIZATION_TYPE": "on-demand"}, {}])
def test_no_warm_read_on_demand(aws_env, monkeypatch, env: dict[str, str]) -> None:  # noqa: F811
    monkeypatch.setenv("OWLBANK_EARLY_DIGITS", "true")
    monkeypatch.setenv("OWLBANK_SECRET_TWILIO_API_KEY", ARN)
    client = _FakeClient()
    fd = _app_with(monkeypatch, client)
    handler.prepare_early_finish(fd, env)
    fd.d.calls.update_twiml("CA1", "<Response/>")
    assert client.requests == [] and client.updates == [("CA1", "<Response/>")]


@pytest.mark.parametrize("result,status", [
    (TimeoutError(f"slow {SECRET['api_secret']}"), None), (ConnectionError("reset"), None), (401, 401), (503, 503),
])
def test_warm_failure_is_logged_by_class_and_ignored(caplog, result: Any, status: int | None) -> None:
    client = _FakeClient(result)
    upd = TwilioCallUpdater(lambda: client)
    with caplog.at_level(logging.INFO, logger="lambdas.frontdoor.calls"):
        assert upd.prebuild() is True  # the client is built; the warm-up failure doesn't matter
    (rec,) = lines(caplog.records, "lambdas.frontdoor.calls")
    assert rec["event"] == "twilio_connection_warm_failed" and rec["decision"] == "ignored"
    assert rec["error_class"] == (type(result).__name__ if status is None else "HttpStatus")
    assert rec.get("http_status") == status
    assert SECRET["api_secret"] not in text_of(caplog.records)
    upd.update_twiml("CA1", "<Response/>")  # early finish still works
    assert client.updates == [("CA1", "<Response/>")]


# --------------------------------------------------------------------------- warm in the settle wait (keyed-in phone login)


def test_ensure_warm_warms_once_while_fresh() -> None:
    """The settle wait opens the connection; a later wait inside ``WARM_FRESH_S`` sends nothing."""
    now = [100.0]
    client = _FakeClient()
    updater = TwilioCallUpdater(lambda: client, monotonic=lambda: now[0])
    updater.ensure_warm()
    now[0] += calls.WARM_FRESH_S
    updater.ensure_warm()
    assert len(client.requests) == 1


def test_ensure_warm_after_an_update_sends_nothing() -> None:
    client = _FakeClient()
    updater = TwilioCallUpdater(lambda: client)
    updater.update_twiml("CA1", "<Response/>")
    updater.ensure_warm()
    assert client.requests == [] and len(client.updates) == 1


def test_ensure_warm_warms_again_once_the_connection_may_have_gone_idle() -> None:
    """The far end closes an idle keep-alive connection, so after ``WARM_FRESH_S``
    without a warm-up or an update the next wait warms again."""
    now = [100.0]
    client = _FakeClient()
    updater = TwilioCallUpdater(lambda: client, monotonic=lambda: now[0])
    updater.update_twiml("CA1", "<Response/>")
    now[0] += calls.WARM_FRESH_S + 0.1
    updater.ensure_warm()
    assert len(client.requests) == 1


def test_ensure_warm_tries_again_after_a_failed_warm_up() -> None:
    client = _FakeClient(503)
    updater = TwilioCallUpdater(lambda: client)
    updater.ensure_warm()
    client.result = 200
    updater.ensure_warm()
    updater.ensure_warm()
    assert len(client.requests) == 2


class _RecordingHttp:
    """A ``twilio.http`` client that records the requests the real ``Client`` hands it."""

    def __init__(self, status: int = 200, error: BaseException | None = None) -> None:
        self.status, self.error = status, error
        self.sent: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kw: Any) -> Any:
        from twilio.http.response import Response

        self.sent.append({"method": method, "url": url, **kw})
        if self.error is not None:
            raise self.error
        return Response(self.status, "{}")


def _real_client_updater(http: _RecordingHttp) -> TwilioCallUpdater:
    from twilio.rest import Client

    return TwilioCallUpdater(lambda: Client(SECRET["api_key"], SECRET["api_secret"], SECRET["account_sid"],
                                            http_client=http))  # type: ignore[arg-type]


def test_warm_up_is_one_calls_list_get_with_the_standard_key(caplog) -> None:
    """R 5.2: exactly one GET of a one-row Calls list under the account, Basic auth from the key."""
    http = _RecordingHttp()
    upd = _real_client_updater(http)
    with caplog.at_level(logging.DEBUG, logger="lambdas.frontdoor.calls"):
        assert upd.warm() is True
    (req,) = http.sent
    assert (req["method"], req["url"], req["timeout"]) == (
        "GET", _warm_url(),
        calls.WARM_TIMEOUT_S)
    assert req["auth"] == (SECRET["api_key"], SECRET["api_secret"])
    assert SECRET["api_secret"] not in text_of(caplog.records)


@pytest.mark.parametrize("make_http,status", [
    (lambda: _RecordingHttp(error=TimeoutError(f"slow {SECRET['api_secret']}")), None),
    (lambda: _RecordingHttp(status=401), 401),
    (lambda: _RecordingHttp(status=403), 403),
], ids=["timeout", "401", "403"])
def test_warm_up_failure_is_swallowed_and_never_logs_credentials(caplog, make_http: Any, status: int | None) -> None:
    http = make_http()  # a fresh recorder per test run
    upd = _real_client_updater(http)
    with caplog.at_level(logging.DEBUG):
        assert upd.warm() is False  # no raise
    assert len(http.sent) == 1
    if status is not None:
        (rec,) = [r for r in lines(caplog.records, "lambdas.frontdoor.calls")
                  if r["event"] == "twilio_connection_warm_failed"]
        assert rec["error_class"] == "HttpStatus" and rec["http_status"] == status
        assert rec["decision"] == "ignored"
    text = text_of(caplog.records)
    assert SECRET["api_secret"] not in text and SECRET["api_key"] not in text
