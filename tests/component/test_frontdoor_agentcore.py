"""The front door's hosting wiring.

``fargate`` keeps today's URL byte for byte and never pre-warms or stops a session; ``agentcore``
mints a fresh pre-signed Runtime URL and nonce for every ``<Connect>`` (the nonce only as
``<Parameter name="n">``, never in the URL), stores ``relay_minted_at``, dispatches the
Runtime warmer once per new call, turns a signing failure into the failure path (the catalog
apology, then a human), and stops the call's Runtime session once at call end. The real
:class:`RuntimeUrlSigner` signs with fixed fake credentials (no AWS call); the warmer and the
session stopper are recorders.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pytest

from lambdas.frontdoor import handler
from lambdas.frontdoor.app import FrontDoor, FrontDoorConfig
from lambdas.frontdoor.relay import PresignError, RuntimeUrlSigner, SigningCredentials
from lambdas.frontdoor.runtime_sessions import RuntimeSessions, claim_key
from owlbank import ws_nonce
from tests.component.test_frontdoor_support import (
    CLOCK,
    Harness,
    all_fixture_factor_values,
    assert_no_leak,
    build,
    relay,
    says,
    sid,
)
from tests.logutil import lines, text_of
from tests.unit.test_coldstart import aws_env  # noqa: F401 - fixture

REGION = "us-east-1"
RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/owlbank_dev_bridge-AbCdEfGhIj"
AKID = "AKIAIOSFODNN7EXAMPLE"
SECRET_KEY = "fake/Secret+Key=abcdefghijklmnopqrstuvwxyz"
SESSION_TOKEN = "fake+Session/Token=abcdefghijklmnopqrstuvwxyz"
FD_LOGGER = "lambdas.frontdoor.app"


# ---------------------------------------------------------------------------- fakes


class RecordingWarmer:
    """``Deps.warmer``: records each dispatch and whether the call state existed at that moment."""

    def __init__(self, h: Harness | None = None, error: Exception | None = None) -> None:
        self.h = h
        self.error = error
        self.calls: list[str] = []
        self.state_existed: list[bool] = []

    def __call__(self, call_sid: str) -> None:
        self.calls.append(call_sid)
        if self.h is not None:
            self.state_existed.append(self.h.store.get("calls", {"call_sid": call_sid}) is not None)
        if self.error is not None:
            raise self.error


def fake_credentials() -> SigningCredentials:
    return SigningCredentials(access_key=AKID, secret_key=SECRET_KEY, token=SESSION_TOKEN,
                              expires_at=CLOCK + 3600)


def signer(*, provider: Any = fake_credentials) -> RuntimeUrlSigner:
    return RuntimeUrlSigner(runtime_arn=RUNTIME_ARN, region=REGION, clock=lambda: CLOCK,
                            credential_provider=provider)


class RecordingStopper:
    """``Deps.runtime_sessions``: records each stop and whether the call state existed then."""

    def __init__(self, h: Harness | None = None, outcome: str = "ok",
                 error: Exception | None = None) -> None:
        self.h, self.outcome, self.error = h, outcome, error
        self.calls: list[str] = []
        self.state_existed: list[bool] = []

    def stop(self, call_sid: str) -> str:
        self.calls.append(call_sid)
        if self.h is not None:
            self.state_existed.append(self.h.store.get("calls", {"call_sid": call_sid}) is not None)
        if self.error is not None:
            raise self.error
        return self.outcome


class CheckingSigner:
    """Wraps a signer: records every mint and whether the call state existed when it was asked."""

    def __init__(self, inner: Any, h: Harness) -> None:
        self.inner, self.h = inner, h
        self.mints: list[tuple[str, str, str]] = []
        self.state_existed: list[bool] = []
        self.refreshes = 0

    def mint(self, *, call_sid: str, nonce: str) -> str:
        self.state_existed.append(self.h.store.get("calls", {"call_sid": call_sid}) is not None)
        url = str(self.inner.mint(call_sid=call_sid, nonce=nonce))
        self.mints.append((call_sid, nonce, url))
        return url

    def refresh_if_due(self, *, margin_s: int = 900) -> None:
        self.refreshes += 1


def agentcore(h: Harness, relay_urls: Any, warmer: Any = None, stopper: Any = None) -> Harness:
    """Switch ``h`` to ``agentcore`` hosting with ``relay_urls``, ``warmer`` and ``stopper``
    (same store)."""
    cfg = replace(h.app.cfg, hosting="agentcore", relay_ws_url=None)
    h.app = FrontDoor(replace(h.app.d, relay=relay_urls, warmer=warmer, runtime_sessions=stopper), cfg)
    return h


def parameters(cr: Any) -> dict[str, str]:
    return {p.get("name"): p.get("value") for p in cr.iter("Parameter")}


def query(url: str) -> dict[str, str]:
    return dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))


@pytest.fixture
def leakcheck(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    made: list[Harness] = []

    def make(**kw: Any) -> Harness:
        h = build(**kw)
        made.append(h)
        return h

    yield make
    for h in made:
        assert_no_leak(h, caplog.records, all_fixture_factor_values() | h.sent_factors)


# ---------------------------------------------------------------------------- config


def test_hosting_defaults_to_fargate_and_needs_the_bridge_url() -> None:
    env = {"OWLBANK_RELAY_WS_URL": "wss://bridge.example/ws"}
    assert FrontDoorConfig.from_env(env).hosting == "fargate"
    assert FrontDoorConfig.from_env({**env, "OWLBANK_HOSTING": ""}).hosting == "fargate"
    with pytest.raises(KeyError):
        FrontDoorConfig.from_env({})
    with pytest.raises(ValueError, match="relay_ws_url"):
        FrontDoorConfig(relay_ws_url=None)


def test_agentcore_needs_no_bridge_url_and_ignores_one_in_the_environment() -> None:
    cfg = FrontDoorConfig.from_env({"OWLBANK_HOSTING": "agentcore",
                                    "OWLBANK_RELAY_WS_URL": "wss://stale.example/ws"})
    assert cfg.hosting == "agentcore" and cfg.relay_ws_url is None


def test_unknown_hosting_is_rejected_without_echoing_the_value() -> None:
    with pytest.raises(ValueError) as err:
        FrontDoorConfig.from_env({"OWLBANK_HOSTING": "lambda-xyz", "OWLBANK_RELAY_WS_URL": "wss://b/ws"})
    assert "OWLBANK_HOSTING" in str(err.value) and "lambda-xyz" not in str(err.value)
    with pytest.raises(ValueError, match="hosting"):
        FrontDoorConfig(relay_ws_url="wss://b/ws", hosting="ecs")


def test_agentcore_front_door_needs_a_signer() -> None:
    h = build()
    with pytest.raises(ValueError, match="Deps.relay"):
        FrontDoor(h.app.d, replace(h.app.cfg, hosting="agentcore", relay_ws_url=None))


# ---------------------------------------------------------------------------- fargate unchanged


def test_fargate_url_is_todays_bridge_url_with_the_nonce(leakcheck) -> None:
    h = leakcheck()
    warmer = RecordingWarmer()
    h.app = FrontDoor(replace(h.app.d, warmer=warmer), h.app.cfg)  # given, but fargate never warms
    cr = relay(h.login_thomas(sid(1)))
    url = cr.get("url") or ""
    nonce = query(url)["n"]
    assert url == ws_nonce.with_nonce("wss://bridge.owlbank.example/ws", nonce)
    st = h.state(sid(1))
    assert st.ws_nonce_hash == ws_nonce.nonce_hash(nonce) and st.relay_minted_at is None
    assert {p.get("name") for p in cr.iter("Parameter")} == {"callSid"}
    assert warmer.calls == []


# ---------------------------------------------------------------------------- warmer


def test_warmer_dispatched_once_per_new_call_after_the_state_exists(leakcheck) -> None:
    h = leakcheck()
    warmer = RecordingWarmer()
    agentcore(h, signer(), warmer)
    warmer.h = h
    h.start(sid(1), "+12025550101")
    h.xml("incoming", sid(1), From="+12025550101", To="+12025550100", CallStatus="ringing")
    h.xml("incoming", sid(2), From="+12025550101", To="+12025550100", CallStatus="ringing")
    assert warmer.calls == [sid(1), sid(2)]
    assert warmer.state_existed == [True, True]


def test_warmer_skipped_for_an_id_that_is_not_a_callsid(leakcheck) -> None:
    h = leakcheck()
    warmer = RecordingWarmer()
    agentcore(h, signer(), warmer)
    h.xml("incoming", "CA" + "E" * 32, From="+12025550101", To="+12025550100")  # upper case
    h.xml("incoming", "CAfake", From="+12025550101", To="+12025550100")
    assert warmer.calls == []


def test_warmer_dispatch_error_is_logged_and_the_twiml_is_unchanged(leakcheck, caplog) -> None:
    plain = leakcheck()
    want = plain.post("incoming", sid(1), From="+12025550101", To="+12025550100")["body"]
    h = agentcore(leakcheck(), signer(), RecordingWarmer(error=RuntimeError("throttled")))
    got = h.post("incoming", sid(1), From="+12025550101", To="+12025550100")["body"]
    assert got == want
    failed = [x for x in lines(caplog.records, FD_LOGGER)
              if x["event"] == "runtime_prewarm_dispatch_failed"]
    assert len(failed) == 1 and failed[0]["error_class"] == "RuntimeError"
    assert failed[0]["call_sid"] == sid(1)


# ---------------------------------------------------------------------------- minting


def test_each_connect_mints_a_fresh_url_and_nonce(leakcheck, caplog) -> None:
    h = leakcheck(config_kw={"max_relay_reconnects": 3})
    s = CheckingSigner(signer(), h)
    agentcore(h, s)
    sent = [relay(h.login_thomas(sid(1)))]
    for _ in range(2):  # dropped sockets: reconnect with a fresh URL and nonce
        root = h.xml("connect-action", sid(1), CallStatus="in-progress", SessionStatus="failed")
        sent.append(relay(root))
    urls = [cr.get("url") or "" for cr in sent]
    assert [u for _, _, u in s.mints] == urls
    nonces = [n for _, n, _ in s.mints]
    assert len(set(nonces)) == 3  # the URLs may match (same second); the nonces never do
    assert [parameters(cr) for cr in sent] == [{"callSid": sid(1), "n": n} for n in nonces]
    for u, n in zip(urls, nonces, strict=True):
        assert "n" not in query(u) and n not in u  # never in the URL
    for u in urls:
        parts = urlsplit(u)
        q = query(u)
        assert parts.scheme == "wss" and parts.netloc == f"bedrock-agentcore.{REGION}.amazonaws.com"
        assert q["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] == sid(1)
        assert int(q["X-Amz-Expires"]) <= 300
    st = h.state(sid(1))
    assert st.ws_nonce_hash == ws_nonce.nonce_hash(nonces[-1])
    assert st.relay_minted_at == int(CLOCK)
    assert s.state_existed == [True, True, True]
    minted = [x for x in lines(caplog.records, FD_LOGGER) if x["event"] == "relay_minted"]
    assert len(minted) == 3 and all(m["hosting"] == "agentcore" for m in minted)
    logs = text_of(caplog.records)
    for _, n, u in s.mints:  # no URL, signature, credential or nonce in any log
        q = query(u)
        for secret in (n, u, q["X-Amz-Signature"], q["X-Amz-Credential"], q["X-Amz-Security-Token"]):
            assert secret not in logs
    assert SESSION_TOKEN not in logs and SECRET_KEY not in logs


def test_agentcore_sends_the_nonce_only_as_a_custom_parameter(leakcheck) -> None:
    h = leakcheck()
    s = CheckingSigner(signer(), h)
    agentcore(h, s)
    cr = relay(h.login_thomas(sid(1)))
    (_, nonce, url), = s.mints
    assert "n" not in query(cr.get("url") or "") and cr.get("url") == url and nonce not in url
    assert parameters(cr) == {"callSid": sid(1), "n": nonce}
    assert h.state(sid(1)).ws_nonce_hash == ws_nonce.nonce_hash(nonce)


# ---------------------------------------------------------------------------- presign failure


#: An error message that must never reach a log (the traceback shows only the ``raise`` line).
STS_ERROR = "sts unreachable " + "x9y8z7"


def _failing_provider() -> SigningCredentials:
    raise ConnectionError(STS_ERROR)


def test_presign_failure_gives_the_catalog_apology_then_a_human(leakcheck, caplog) -> None:
    h = agentcore(leakcheck(), signer(provider=_failing_provider))
    root = h.login_thomas(sid(1))
    assert root.find("./Connect") is None
    enqueue = root.find("Enqueue")
    assert enqueue is not None
    assert json.loads(enqueue.find("Task").text or "{}")["escalation_reason"] == "unable_to_resolve"
    apology = h.app.d.catalog.text("error.generic", "en-US")
    assert says(root)[0] == apology
    failed = [x for x in lines(caplog.records, FD_LOGGER) if x["event"] == "relay_presign_failed"]
    assert len(failed) == 1 and failed[0]["error_class"] == "ConnectionError"
    assert STS_ERROR not in text_of(caplog.records)
    assert not [x for x in lines(caplog.records, FD_LOGGER) if x["event"] == "handler error"]
    st = h.state(sid(1))
    assert st.ws_nonce_hash is None and st.relay_minted_at is None  # nothing written for it


def test_presign_failure_on_a_reconnect_also_escalates(leakcheck) -> None:
    h = leakcheck(config_kw={"max_relay_reconnects": 3})

    class FailSecond:
        def __init__(self) -> None:
            self.n = 0
            self.inner = signer()

        def mint(self, *, call_sid: str, nonce: str) -> str:
            self.n += 1
            if self.n > 1:
                raise PresignError("ExpiredToken")
            return self.inner.mint(call_sid=call_sid, nonce=nonce)

    agentcore(h, FailSecond())
    relay(h.login_thomas(sid(1)))
    root = h.xml("connect-action", sid(1), CallStatus="in-progress", SessionStatus="failed")
    assert root.find("./Connect") is None and root.find("Enqueue") is not None


# ---------------------------------------------------------------------------- credential refresh


def test_status_refreshes_the_signing_credentials_on_agentcore_only(leakcheck) -> None:
    h = leakcheck()
    s = CheckingSigner(signer(), h)
    agentcore(h, s)
    h.login_thomas(sid(1))
    h.post("status", sid(1), CallStatus="ringing")  # not an end status: nothing
    assert s.refreshes == 0
    h.post("status", sid(1), CallStatus="completed")
    assert s.refreshes == 1

    f = leakcheck()
    fs = CheckingSigner(signer(), f)
    f.app = FrontDoor(replace(f.app.d, relay=fs), f.app.cfg)  # fargate with an explicit relay
    f.login_thomas(sid(2))
    f.post("status", sid(2), CallStatus="completed")
    assert fs.refreshes == 0


# ---------------------------------------------------------------------------- handler wiring


class FakeLambda:
    def __init__(self, status: int = 202) -> None:
        self.status = status
        self.invokes: list[dict[str, Any]] = []

    def invoke(self, **kwargs: Any) -> dict[str, Any]:
        self.invokes.append(kwargs)
        return {"StatusCode": self.status}


def test_event_warmer_invokes_the_warmer_asynchronously() -> None:
    client = FakeLambda()
    handler.EventWarmer(client, "owlbank-dev-runtime-warmer")(sid(1))
    (call,) = client.invokes
    assert call["FunctionName"] == "owlbank-dev-runtime-warmer"
    assert call["InvocationType"] == "Event"
    assert json.loads(call["Payload"]) == {"call_sid": sid(1)}
    with pytest.raises(RuntimeError):
        handler.EventWarmer(FakeLambda(status=500), "w")(sid(1))
    with pytest.raises(ValueError):
        handler.EventWarmer(client, "")


AGENTCORE_ENV = {"OWLBANK_HOSTING": "agentcore", "OWLBANK_RUNTIME_ARN": RUNTIME_ARN,
                 "OWLBANK_PRESIGN_ROLE_ARN": "arn:aws:iam::123456789012:role/owlbank-dev-relay-presign",
                 "OWLBANK_WARMER_FUNCTION": "owlbank-dev-runtime-warmer", "AWS_REGION": REGION}


def test_relay_signer_from_env() -> None:
    s = handler.relay_signer({**AGENTCORE_ENV, "OWLBANK_URL_TTL_S": "120"}, sts=object())
    assert s.ttl_s == 120
    assert handler.relay_signer(AGENTCORE_ENV, sts=object()).ttl_s == 300
    with pytest.raises(ValueError, match="OWLBANK_URL_TTL_S"):
        handler.relay_signer({**AGENTCORE_ENV, "OWLBANK_URL_TTL_S": "5m"}, sts=object())
    with pytest.raises(ValueError):
        handler.relay_signer({**AGENTCORE_ENV, "OWLBANK_URL_TTL_S": "301"}, sts=object())


def test_build_app_agentcore_wires_a_lazy_signer_and_warmer(aws_env, monkeypatch) -> None:  # noqa: F811
    for k, v in AGENTCORE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("OWLBANK_RELAY_WS_URL")
    app = handler.build_app()
    assert aws_env == ["lambda"]  # the backend client only: STS and the warmer's client are lazy
    assert app.cfg.hosting == "agentcore" and isinstance(app.relay_urls, RuntimeUrlSigner)
    assert isinstance(app.d.warmer, handler.EventWarmer)
    assert isinstance(app.d.runtime_sessions, RuntimeSessions)  # its client is lazy too


def test_build_app_fargate_wires_neither(aws_env) -> None:  # noqa: F811
    app = handler.build_app()
    assert aws_env == ["lambda"]
    assert app.cfg.hosting == "fargate" and app.d.relay is None and app.d.warmer is None
    assert app.d.runtime_sessions is None


def test_prepare_relay_refreshes_at_init_on_agentcore_only() -> None:
    h = build()
    s = CheckingSigner(signer(), h)
    handler.prepare_relay(FrontDoor(replace(h.app.d, relay=s), h.app.cfg))  # fargate
    assert s.refreshes == 0
    handler.prepare_relay(agentcore(h, s).app)
    assert s.refreshes == 1


def test_signer_refresh_failure_at_init_never_raises(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    h = agentcore(build(), signer(provider=_failing_provider))
    handler.prepare_relay(h.app)
    assert STS_ERROR not in text_of(caplog.records)


# ---------------------------------------------------------------------------- call end


def test_call_end_stops_the_runtime_session_once(leakcheck) -> None:
    h = leakcheck()
    stopper = RecordingStopper()
    agentcore(h, CheckingSigner(signer(), h), stopper=stopper)
    stopper.h = h
    h.login_thomas(sid(1))
    h.post("status", sid(1), CallStatus="in-progress")  # not an end status
    assert stopper.calls == []
    for status in ("completed", "completed", "failed", "canceled"):  # Twilio retries, late ones
        assert h.post("status", sid(1), CallStatus=status)["statusCode"] == 200
    assert stopper.calls == [sid(1)] and stopper.state_existed == [True]
    claim = h.store.get("calls", {"call_sid": claim_key(sid(1))})
    assert claim is not None and claim["expires_at"] == h.state(sid(1)).expires_at


@pytest.mark.parametrize("status", ["completed", "failed", "no-answer", "busy", "canceled"])
def test_every_end_status_stops_the_session(leakcheck, status: str) -> None:
    h = leakcheck()
    stopper = RecordingStopper()
    agentcore(h, signer(), stopper=stopper)
    h.xml("incoming", sid(1), From="+12025550101", To="+12025550100", CallStatus="ringing")
    h.post("status", sid(1), CallStatus=status)
    assert stopper.calls == [sid(1)]


def test_no_stop_for_a_call_without_state_or_a_callsid(leakcheck) -> None:
    h = leakcheck()
    stopper = RecordingStopper()
    agentcore(h, signer(), stopper=stopper)
    h.post("status", sid(1), CallStatus="completed")  # never seen: no session was started
    h.xml("incoming", "CAfake", From="+12025550101", To="+12025550100")
    h.post("status", "CAfake", CallStatus="completed")
    assert stopper.calls == []


def test_a_stop_error_never_fails_the_status_response(leakcheck, caplog) -> None:
    plain = leakcheck()
    plain.login_thomas(sid(1))
    want = plain.post("status", sid(1), CallStatus="completed")
    h = leakcheck()
    stopper = RecordingStopper(error=RuntimeError("boom"))  # a stopper that breaks its contract
    agentcore(h, signer(), stopper=stopper)
    h.login_thomas(sid(1))
    got = h.post("status", sid(1), CallStatus="completed")
    assert got["statusCode"] == want["statusCode"] == 200 and got["body"] == want["body"]
    assert h.post("status", sid(1), CallStatus="completed")["statusCode"] == 200
    assert stopper.calls == [sid(1)]  # claimed: a failed attempt is not repeated
    failed = [x for x in lines(caplog.records, FD_LOGGER) if x["event"] == "runtime_session_stop_failed"]
    assert len(failed) == 1 and failed[0]["error_class"] == "RuntimeError"


class FakeAgentCore:
    """A ``bedrock-agentcore`` client: ``stop_runtime_session`` records and raises ``error``."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def stop_runtime_session(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"runtimeSessionId": kwargs["runtimeSessionId"], "statusCode": 200}


def test_status_with_the_real_stopper_logs_the_outcome(leakcheck, caplog) -> None:
    h = leakcheck()
    client = FakeAgentCore()
    agentcore(h, signer(), stopper=RuntimeSessions(client, RUNTIME_ARN))
    h.login_thomas(sid(1))
    h.post("status", sid(1), CallStatus="completed")
    assert client.calls == [{"agentRuntimeArn": RUNTIME_ARN, "runtimeSessionId": sid(1)}]
    [line] = [x for x in lines(caplog.records, "lambdas.frontdoor.runtime_sessions")
              if x["event"] == "runtime_session_stopped"]
    assert line["outcome"] == "ok" and line["call_sid"] == sid(1)
    assert RUNTIME_ARN not in text_of(caplog.records)


def test_fargate_never_stops_a_session(leakcheck) -> None:
    h = leakcheck()
    stopper = RecordingStopper()
    h.app = FrontDoor(replace(h.app.d, runtime_sessions=stopper), h.app.cfg)  # given, but fargate
    h.login_thomas(sid(1))
    h.post("status", sid(1), CallStatus="completed")
    assert stopper.calls == []
    assert h.store.get("calls", {"call_sid": claim_key(sid(1))}) is None


# ---------------------------------------------------------------------------- 60-minute cap

#: What ConversationRelay sent to ``connect-action`` on a forced 1008 close.
CAP_CLOSE: dict[str, Any] = {"CallStatus": "in-progress", "SessionStatus": "failed", "ErrorCode": "64105"}


class Clock:
    """A settable ``Deps.clock`` (the signer keeps its own fixed clock)."""

    def __init__(self) -> None:
        self.now = CLOCK

    def __call__(self) -> float:
        return self.now


def timed(h: Harness, clock: Clock) -> Harness:
    h.app = FrontDoor(replace(h.app.d, clock=clock), h.app.cfg)
    return h


def reconnects(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [x for x in lines(caplog.records, FD_LOGGER) if x["event"] == "relay_reconnect"]


def test_cap_close_reconnects_silently_to_the_same_session(leakcheck, caplog) -> None:
    clock = Clock()
    h = leakcheck()
    s = CheckingSigner(signer(), h)
    timed(agentcore(h, s), clock)
    relay(h.login_thomas(sid(1)))
    first = h.state(sid(1))
    clock.now = CLOCK + 3657  # the measured cap
    root = h.xml("connect-action", sid(1), **CAP_CLOSE)
    cr = relay(root)
    assert says(root) == [] and root.find("Play") is None  # no apology
    assert cr.get("welcomeGreeting") == ""
    (_, n1, _), (_, n2, url2) = s.mints
    assert n1 != n2 and cr.get("url") == url2 and parameters(cr) == {"callSid": sid(1), "n": n2}
    assert query(url2)["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] == sid(1)  # same session
    st = h.state(sid(1))
    assert st.ws_nonce_hash == ws_nonce.nonce_hash(n2) != first.ws_nonce_hash
    assert st.relay_minted_at == int(CLOCK + 3657) and st.last_spoken is None
    sess = h.session(sid(1))
    assert sess.relay_failures == 0 and sess.stream_reconnects == 1
    [line] = reconnects(caplog)
    assert line["reason"] == "stream_limit" and line["call_sid"] == sid(1)
    assert line["stream_reconnects"] == 1
    # The reconnected session still gets today's drop path for an ordinary drop.
    clock.now += 30
    root = h.xml("connect-action", sid(1), **CAP_CLOSE)
    assert says(root) == [h.app.d.catalog.text("error.generic", "en-US")]
    assert h.session(sid(1)).relay_failures == 1
    assert h.post("status", sid(1), CallStatus="completed")["statusCode"] == 200
    logs = text_of(caplog.records)
    for _, n, u in s.mints:
        assert n not in logs and u not in logs and query(u)["X-Amz-Signature"] not in logs


def test_a_drop_before_the_cap_takes_the_apology_and_reconnect_once_path(leakcheck, caplog) -> None:
    clock = Clock()
    h = timed(agentcore(leakcheck(), signer()), clock)
    h.login_thomas(sid(1))
    clock.now = CLOCK + 60  # a forced close: what a real call heard
    root = h.xml("connect-action", sid(1), **CAP_CLOSE)
    assert says(root) == [h.app.d.catalog.text("error.generic", "en-US")]
    assert relay(root).get("welcomeGreeting") == h.app.d.catalog.text("common.anything_else", "en-US")
    clock.now += 60
    root = h.xml("connect-action", sid(1), **CAP_CLOSE)
    assert root.find("./Connect") is None and root.find("Enqueue") is not None
    assert h.session(sid(1)).stream_reconnects == 0 and reconnects(caplog) == []


def test_another_connect_failure_after_the_cap_age_takes_the_apology_then_human_path(
        leakcheck) -> None:
    # a quota or throttle refusal is a failed connect; only 64105 is the cap's signature.
    clock = Clock()
    h = timed(agentcore(leakcheck(), signer()), clock)
    h.login_thomas(sid(1))
    clock.now = CLOCK + 4000
    params = {**CAP_CLOSE, "ErrorCode": "64107"}
    root = h.xml("connect-action", sid(1), **params)
    assert says(root) == [h.app.d.catalog.text("error.generic", "en-US")]
    root = h.xml("connect-action", sid(1), **params)
    task = root.find("Enqueue/Task")
    assert task is not None
    assert json.loads(task.text or "{}")["escalation_reason"] == "unable_to_resolve"


def test_at_most_four_silent_reconnects_per_call(leakcheck, caplog) -> None:
    clock = Clock()
    h = timed(agentcore(leakcheck(), signer()), clock)
    h.login_thomas(sid(1))
    apology = h.app.d.catalog.text("error.generic", "en-US")
    for n in range(1, 5):
        clock.now += 3657
        root = h.xml("connect-action", sid(1), **CAP_CLOSE)
        assert says(root) == [] and h.session(sid(1)).stream_reconnects == n
    clock.now += 3657
    root = h.xml("connect-action", sid(1), **CAP_CLOSE)  # the fifth: today's drop path
    assert says(root) == [apology] and relay(root) is not None
    assert h.session(sid(1)).relay_failures == 1
    assert len(reconnects(caplog)) == 4
    capped = [x for x in lines(caplog.records, FD_LOGGER)
              if x["event"] == "stream_reconnect_cap_reached"]
    assert len(capped) == 1


def test_a_cap_reconnect_that_cannot_be_signed_takes_the_apology_then_human_path(leakcheck) -> None:
    clock = Clock()
    h = leakcheck()

    class FailSecond:
        def __init__(self) -> None:
            self.n, self.inner = 0, signer()

        def mint(self, *, call_sid: str, nonce: str) -> str:
            self.n += 1
            if self.n > 1:
                raise PresignError("ExpiredToken")
            return self.inner.mint(call_sid=call_sid, nonce=nonce)

    timed(agentcore(h, FailSecond()), clock)
    h.login_thomas(sid(1))
    clock.now = CLOCK + 3657
    root = h.xml("connect-action", sid(1), **CAP_CLOSE)
    assert root.find("./Connect") is None and root.find("Enqueue") is not None
    assert says(root)[0] == h.app.d.catalog.text("error.generic", "en-US")
    assert h.session(sid(1)).stream_reconnects == 1  # counted before the mint


def test_fargate_never_reconnects_silently(leakcheck, caplog) -> None:
    clock = Clock()
    h = timed(leakcheck(), clock)
    h.login_thomas(sid(1))
    clock.now = CLOCK + 3657
    root = h.xml("connect-action", sid(1), **CAP_CLOSE)
    assert says(root) == [h.app.d.catalog.text("error.generic", "en-US")]
    assert h.session(sid(1)).stream_reconnects == 0 and reconnects(caplog) == []
