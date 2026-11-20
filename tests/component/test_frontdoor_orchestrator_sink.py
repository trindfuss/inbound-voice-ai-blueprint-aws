"""The front door's Orchestrator sink, ``POST /voice/orchestrator``.

In ``agentcore`` the Conversation Configuration's status callback lands here. The WAF has already
answered a request without ``Authorization`` with the Basic challenge (API Gateway REST renames a
Lambda's ``WWW-Authenticate``), so the handler checks Basic (missing or wrong: 401,
``twilio_auth_failed reason=basic``), then ``X-Twilio-Signature`` (403, ``reason=signature``), then
logs ``orchestrator_event_dropped`` (event type, timing, ``reason=no_relay``) and answers an empty
200. It never calls the Runtime, never reads or writes call state, and never logs the event body,
the credentials or a header value. On ``fargate`` the route doesn't exist. No PBT: a fixed handler
with no state; the cases below cover it.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import replace
from typing import Any
from urllib.parse import urlencode

import pytest
from twilio.request_validator import RequestValidator

from lambdas.frontdoor import http
from lambdas.frontdoor.app import FrontDoor
from owlbank.basic_auth import BasicSecret
from owlbank.security_log import AUTH_FAILED_EVENT, BASIC_CHALLENGE_EVENT
from tests.component.test_frontdoor_agentcore import RecordingStopper, RecordingWarmer, signer
from tests.component.test_frontdoor_support import (
    AUTH_TOKEN,
    BASIC,
    CLOCK,
    HOST,
    STAGE,
    basic_header,
    build,
)
from tests.logutil import fields, text_of

PATH = f"{STAGE}/voice/orchestrator"
CONV_ID = "conv_conversation_01k7private0000000000"
ADDRESS = "+12025550177"
TRANSCRIPT = "my account number is nine nine one two"
IDEMPOTENCY = "idem-token-QwErTy-1234567890"
EVENT_TIME = "2025-10-09T08:53:19.400Z"  # CLOCK is 2025-10-09T08:53:20Z
BODY = json.dumps({
    "eventType": "COMMUNICATION_CREATED",
    "timestamp": EVENT_TIME,
    "data": {"id": CONV_ID, "status": "ACTIVE", "content": TRANSCRIPT,
             "participants": [{"address": ADDRESS}]},
})
SECRETS = (BASIC[0], BASIC[1], AUTH_TOKEN)
PRIVATE = (CONV_ID, ADDRESS, TRANSCRIPT, IDEMPOTENCY, EVENT_TIME)


# ---------------------------------------------------------------------------- fakes and events


class RecordingStore:
    """Wraps the harness store: records every call made after ``arm()``."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[str] = []
        self.armed = False

    def arm(self) -> None:
        self.armed = True

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kw: Any) -> Any:
            if self.armed:
                self.calls.append(name)
            return attr(*args, **kw)

        return call


class RecordingSigner:
    def __init__(self) -> None:
        self.inner = signer()
        self.mints: list[str] = []

    def mint(self, *, call_sid: str, nonce: str) -> str:
        self.mints.append(call_sid)
        return str(self.inner.mint(call_sid=call_sid, nonce=nonce))

    def refresh_if_due(self, *, margin_s: int = 900) -> None:
        self.mints.append("refresh")


class Sink:
    """An ``agentcore`` front door whose store, Lambda client, signer, warmer and session stopper
    all record."""

    def __init__(self, *, hosting: str = "agentcore", auth_token: Any = None) -> None:
        h = build()
        self.store = RecordingStore(h.store)
        self.lam = h.lam
        self.relay = RecordingSigner()
        self.warmer = RecordingWarmer()
        self.stopper = RecordingStopper()
        cfg = replace(h.app.cfg, hosting=hosting,
                      relay_ws_url=None if hosting == "agentcore" else h.app.cfg.relay_ws_url)
        deps = replace(h.app.d, store=self.store, relay=self.relay, warmer=self.warmer,
                       runtime_sessions=self.stopper,
                       auth_token=auth_token or h.app.d.auth_token)
        self.app = FrontDoor(deps, cfg)
        self.store.arm()
        self.lam_calls_before = len(self.lam.calls)

    def assert_untouched(self) -> None:
        """No store, Lambda, Runtime URL, warmer or session-stop call."""
        assert self.store.calls == []
        assert len(self.lam.calls) == self.lam_calls_before
        assert self.relay.mints == [] and self.warmer.calls == [] and self.stopper.calls == []


def sink_event(body: str = BODY, *, auth: str | None = "ok", signature: str | None = None,
               url_credentials: bool = False, json_body: bool = True, method: str = "POST",
               base64_body: bool = False, path: str = PATH) -> dict[str, Any]:
    """An API Gateway REST proxy event for one Orchestrator callback, after the WAF's challenge.
    ``auth="ok"`` carries the right Basic credentials, ``None`` none, any other string as is."""
    validator = RequestValidator(AUTH_TOKEN)
    query = {"bodySHA256": validator.compute_hash(body)} if json_body else None
    authority = f"{BASIC[0]}:{BASIC[1]}@{HOST}" if url_credentials else HOST
    url = f"https://{authority}{path}" + (f"?{urlencode(query)}" if query else "")
    params = {} if json_body else dict(_form(body))
    headers = {
        "Host": HOST, "X-Forwarded-Proto": "https", "X-Forwarded-Port": "443",
        "Content-Type": "application/json" if json_body else "application/x-www-form-urlencoded",
        "I-Twilio-Idempotency-Token": IDEMPOTENCY,
        "X-Twilio-Signature": signature if signature is not None else validator.compute_signature(url, params),
    }
    if auth == "ok":
        headers["Authorization"] = basic_header(*BASIC)
    elif auth is not None:
        headers["Authorization"] = auth
    raw = base64.b64encode(body.encode()).decode() if base64_body else body
    return {
        "resource": "/voice/orchestrator", "path": "/voice/orchestrator", "httpMethod": method,
        "headers": headers, "multiValueHeaders": None,
        "queryStringParameters": query,
        "multiValueQueryStringParameters": {k: [v] for k, v in query.items()} if query else None,
        "requestContext": {"path": path, "stage": "prod", "domainName": HOST},
        "body": raw, "isBase64Encoded": base64_body,
    }


def _form(body: str) -> list[tuple[str, str]]:
    from urllib.parse import parse_qsl

    return parse_qsl(body, keep_blank_values=True)


def named(caplog: pytest.LogCaptureFixture, event: str) -> list[dict[str, Any]]:
    return [fields(r) for r in caplog.records if r.getMessage() == event]


def assert_nothing_private_logged(caplog: pytest.LogCaptureFixture, *responses: dict[str, Any],
                                  event: dict[str, Any] | None = None) -> None:
    """No body content, ID, address, credential or header value in any log line or response."""
    text = text_of(caplog.records) + json.dumps(responses)
    for value in (*PRIVATE, *SECRETS):
        assert value not in text, "private value logged or answered"
    if event is not None:
        for name, value in event["headers"].items():
            if value and name.lower() not in ("host", "x-forwarded-proto", "x-forwarded-port", "content-type"):
                assert value not in text, f"header {name} value logged"
    assert "bodySHA256" not in text


@pytest.fixture(autouse=True)
def _debug(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)


# ---------------------------------------------------------------------------- accepted


@pytest.mark.parametrize("url_credentials", [False, True], ids=["plain", "credentials"])
def test_valid_callback_is_logged_and_acknowledged_and_goes_nowhere(
        caplog: pytest.LogCaptureFixture, url_credentials: bool) -> None:
    s = Sink()
    event = sink_event(url_credentials=url_credentials)
    resp = s.app.handle(event)
    assert resp == {"statusCode": 200, "headers": {}, "body": ""}
    (line,) = named(caplog, "orchestrator_event_dropped")
    assert line["reason"] == "no_relay"
    assert line["event_type"] == "COMMUNICATION_CREATED"
    assert line["lag_ms"] == 600
    assert set(line) <= {"event", "reason", "event_type", "lag_ms"}
    assert not named(caplog, AUTH_FAILED_EVENT)
    s.assert_untouched()
    assert_nothing_private_logged(caplog, resp, event=event)


def test_every_event_type_seen_in_testing_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    s = Sink()
    for kind in ("CONVERSATION_CREATED", "PARTICIPANT_ADDED", "COMMUNICATION_CREATED",
                 "CONVERSATION_UPDATED"):
        body = json.dumps({"eventType": kind, "data": {"id": CONV_ID}})
        assert s.app.handle(sink_event(body))["statusCode"] == 200
    assert [x["event_type"] for x in named(caplog, "orchestrator_event_dropped")] == [
        "CONVERSATION_CREATED", "PARTICIPANT_ADDED", "COMMUNICATION_CREATED", "CONVERSATION_UPDATED"]
    s.assert_untouched()


@pytest.mark.parametrize("raw_type", [TRANSCRIPT, ADDRESS, "conversation created", 7, "X" * 65])
def test_an_event_type_that_is_not_an_enum_is_logged_as_unrecognised(
        caplog: pytest.LogCaptureFixture, raw_type: Any) -> None:
    body = json.dumps({"eventType": raw_type, "data": {"id": CONV_ID}})
    assert Sink().app.handle(sink_event(body))["statusCode"] == 200
    (line,) = named(caplog, "orchestrator_event_dropped")
    assert line["event_type"] == "unrecognised" and "lag_ms" not in line
    assert_nothing_private_logged(caplog)


@pytest.mark.parametrize("body", ["", "not json at all " + TRANSCRIPT, "[1, 2]", '"text"',
                                  json.dumps({"timestamp": "yesterday", "data": "x"})])
def test_an_unreadable_or_odd_body_is_still_acknowledged(caplog: pytest.LogCaptureFixture,
                                                         body: str) -> None:
    s = Sink()
    assert s.app.handle(sink_event(body))["statusCode"] == 200
    (line,) = named(caplog, "orchestrator_event_dropped")
    assert line["reason"] == "no_relay" and "event_type" not in line and "lag_ms" not in line
    s.assert_untouched()
    assert_nothing_private_logged(caplog)


def test_a_form_encoded_callback_is_checked_as_a_form(caplog: pytest.LogCaptureFixture) -> None:
    body = urlencode({"EventType": "CONVERSATION_UPDATED", "ConversationSid": CONV_ID})
    s = Sink()
    assert s.app.handle(sink_event(body, json_body=False))["statusCode"] == 200
    (line,) = named(caplog, "orchestrator_event_dropped")
    assert line["event_type"] == "CONVERSATION_UPDATED"
    s.assert_untouched()
    assert_nothing_private_logged(caplog)


def test_a_base64_body_is_decoded_before_the_check(caplog: pytest.LogCaptureFixture) -> None:
    assert Sink().app.handle(sink_event(base64_body=True))["statusCode"] == 200


# ---------------------------------------------------------------------------- Basic


@pytest.mark.parametrize("auth", [None, "", basic_header(BASIC[0], "wrong-password"),
                                  basic_header("someone", BASIC[1]), "Bearer abc.def.ghi",
                                  "Basic !!!not-base64!!!"],
                         ids=["missing", "empty", "wrong-password", "wrong-user", "bearer", "garbage"])
def test_missing_or_wrong_basic_is_401_reason_basic(caplog: pytest.LogCaptureFixture,
                                                    auth: str | None) -> None:
    s = Sink()
    event = sink_event(auth=auth)
    resp = s.app.handle(event)
    assert resp == {"statusCode": 401, "headers": {}, "body": ""}  # the WAF sends the challenge
    (failed,) = named(caplog, AUTH_FAILED_EVENT)
    assert failed["reason"] == "basic" and failed["component"] == "frontdoor"
    assert not named(caplog, BASIC_CHALLENGE_EVENT)
    assert not named(caplog, "orchestrator_event_dropped")
    s.assert_untouched()
    assert_nothing_private_logged(caplog, resp, event=event)


# ---------------------------------------------------------------------------- signature


@pytest.mark.parametrize("case", ["bad", "missing", "body_changed", "other_token", "other_path"])
def test_a_bad_signature_is_403_reason_signature(caplog: pytest.LogCaptureFixture, case: str) -> None:
    s = Sink()
    if case == "bad":
        event = sink_event(signature="c2lnbmF0dXJlLW5vdC12YWxpZA==")
    elif case == "missing":
        event = sink_event()
        del event["headers"]["X-Twilio-Signature"]
    elif case == "body_changed":  # bodySHA256 and the signature are for BODY; another body arrives
        event = sink_event()
        event["body"] = BODY.replace("ACTIVE", "CLOSED")
    elif case == "other_token":
        good = RequestValidator("another-account-token-0000000000")
        query = {"bodySHA256": good.compute_hash(BODY)}
        event = sink_event(signature=good.compute_signature(
            f"https://{HOST}{PATH}?{urlencode(query)}", {}))
    else:  # signed for another route of the same API
        event = sink_event()
        other = sink_event(path=f"{STAGE}/voice/status")
        event["headers"]["X-Twilio-Signature"] = other["headers"]["X-Twilio-Signature"]
    resp = s.app.handle(event)
    assert resp == {"statusCode": 403, "headers": {}, "body": ""}
    (failed,) = named(caplog, AUTH_FAILED_EVENT)
    assert failed["reason"] == "signature"
    assert not named(caplog, "orchestrator_event_dropped")
    s.assert_untouched()
    assert_nothing_private_logged(caplog, resp, event=event)


def test_an_unreadable_body_is_403_reason_signature(caplog: pytest.LogCaptureFixture) -> None:
    event = sink_event()
    event["isBase64Encoded"] = True
    event["body"] = "%%% not base64 " + TRANSCRIPT
    assert Sink().app.handle(event)["statusCode"] == 403
    (failed,) = named(caplog, AUTH_FAILED_EVENT)
    assert failed["reason"] == "signature"
    assert_nothing_private_logged(caplog)


# ---------------------------------------------------------------------------- fail closed


def test_an_unreadable_secret_is_403_unverifiable(caplog: pytest.LogCaptureFixture) -> None:
    def broken() -> str:
        raise RuntimeError(f"cannot read the secret {AUTH_TOKEN}")

    s = Sink(auth_token=broken)
    assert s.app.handle(sink_event())["statusCode"] == 403
    (failed,) = named(caplog, AUTH_FAILED_EVENT)
    assert failed["reason"] == "unverifiable"
    s.assert_untouched()
    assert_nothing_private_logged(caplog)


def test_only_post_is_served(caplog: pytest.LogCaptureFixture) -> None:
    s = Sink()
    assert s.app.handle(sink_event(method="GET"))["statusCode"] == 403
    (failed,) = named(caplog, AUTH_FAILED_EVENT)
    assert failed["reason"] == "route"
    s.assert_untouched()


def test_fargate_has_no_orchestrator_route(caplog: pytest.LogCaptureFixture) -> None:
    """Fargate is unchanged: its status callback is the bridge's ``/webhook``; this route is refused."""
    s = Sink(hosting="fargate")
    assert s.app.handle(sink_event())["statusCode"] == 403
    (failed,) = named(caplog, AUTH_FAILED_EVENT)
    assert failed["reason"] == "route"
    assert not named(caplog, "orchestrator_event_dropped")
    s.assert_untouched()


def test_the_voice_webhook_check_never_accepts_the_orchestrator_route() -> None:
    """Defence in depth: :func:`http.authenticate` (form webhooks with a TwiML answer) refuses it."""
    with pytest.raises(http.Rejected):
        http.authenticate(sink_event(), auth_token=lambda: AUTH_TOKEN, basic_secret=lambda force=False: BasicSecret(*BASIC))


def test_the_route_is_named_for_the_invocation_log() -> None:
    assert http.route_name(sink_event()) == "orchestrator"


def test_the_voice_routes_still_challenge_for_basic() -> None:
    """The other routes keep their auth challenge (the sink's 401 has no ``WWW-Authenticate``)."""
    from tests.component.test_frontdoor_support import NO_AUTH, make_event

    s = Sink()
    resp = s.app.handle(make_event("incoming", {"CallSid": "CA" + "1" * 32}, auth=NO_AUTH))
    assert resp["statusCode"] == 401 and resp["headers"]["WWW-Authenticate"].startswith("Basic ")


def test_lag_is_measured_against_the_front_door_clock(caplog: pytest.LogCaptureFixture) -> None:
    s = Sink()
    s.app.d = replace(s.app.d, clock=lambda: CLOCK + 2)
    assert s.app.handle(sink_event())["statusCode"] == 200
    (line,) = named(caplog, "orchestrator_event_dropped")
    assert line["lag_ms"] == 2600
