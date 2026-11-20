"""The front door accepts every API Gateway REST proxy request shape, and answers a body it
can't read with a generic 400 (valid empty TwiML, nothing echoed, logged without content)."""

from __future__ import annotations

import base64
import logging
from typing import Any

import pytest

from lambdas.frontdoor import http
from owlbank.basic_auth import BasicSecret
from tests.component.test_frontdoor_support import (
    AUTH_TOKEN, BASIC, NO_AUTH, basic_header, build, make_event,
)

PARAMS = {"CallSid": "CA" + "c" * 32, "AccountSid": "AC" + "b" * 32, "From": "+12025550101"}


def auth(event: dict[str, Any]) -> http.TwilioRequest:
    return http.authenticate(event, auth_token=lambda: AUTH_TOKEN, basic_secret=lambda force=False: BasicSecret(*BASIC))


@pytest.mark.parametrize("body", [None, "", "missing"])
def test_missing_or_null_body_is_an_empty_form(body: str | None) -> None:
    event = make_event("status", {})
    if body == "missing":
        del event["body"]
    else:
        event["body"] = body
    assert auth(event).params == {}


def test_base64_body_is_decoded() -> None:
    event = make_event("status", PARAMS)
    event["body"] = base64.b64encode(event["body"].encode()).decode()
    event["isBase64Encoded"] = True
    assert auth(event).params == PARAMS


@pytest.mark.parametrize("case", [str.lower, str.upper, str.title, lambda s: s.swapcase()])
def test_header_names_in_any_case(case) -> None:
    event = make_event("status", PARAMS)
    event["headers"] = {case(k): v for k, v in event["headers"].items()}
    assert auth(event).params == PARAMS


def test_multi_value_headers_alone_and_null_maps() -> None:
    event = make_event("factor?g=abc", PARAMS)
    event["multiValueHeaders"] = {k.upper(): [v] for k, v in event["headers"].items()}
    event["headers"] = None
    event["multiValueQueryStringParameters"] = None  # only the single-value map
    req = auth(event)
    assert req.params == PARAMS and req.query == {"g": "abc"}
    event = make_event("status", PARAMS)
    event.update(queryStringParameters=None, multiValueQueryStringParameters=None,
                 multiValueHeaders=None)
    assert auth(event).params == PARAMS


def test_headers_normalized_once_and_shared_with_the_url() -> None:
    event = make_event("status", PARAMS)
    headers = http._headers(event)  # noqa: SLF001
    assert all(k == k.lower() for k in headers)
    assert http.public_url(event, headers) == http.public_url(event)


MALFORMED = {
    "bad base64": ("@@not base64@@", True),
    "base64 of non-UTF-8": (base64.b64encode(b"CallSid=\xff\xfe").decode(), True),
    "bad percent-escape": ("CallSid=%FF%FE&Digits=94105", False),
    "too many fields": ("&".join(f"f{i}=1" for i in range(http.MAX_FORM_FIELDS + 5)), False),
    "not text": ({"CallSid": "x"}, False),
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_malformed_body_is_a_generic_400_logged_without_content(name: str, caplog) -> None:
    body, b64 = MALFORMED[name]
    fd = build()
    event = make_event("factor", PARAMS)
    event.update(body=body, isBase64Encoded=b64)
    caplog.set_level(logging.DEBUG)
    resp = fd.app.handle(event)
    assert resp == http.bad_request()
    assert resp["statusCode"] == 400 and resp["body"] == http.BAD_REQUEST_TWIML
    rec = [r for r in caplog.records if r.getMessage() == "frontdoor: malformed request"]
    assert rec and rec[0].reason == "malformed_body" and rec[0].decision == "reject_400"
    logged = " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert "94105" not in logged and "@@not" not in logged and "Traceback" not in logged


def test_malformed_body_with_wrong_basic_is_still_403() -> None:
    fd = build()
    event = make_event("factor", PARAMS, auth=basic_header(BASIC[0], "wrong"))
    event["body"] = "CallSid=%FF"
    assert fd.app.handle(event)["statusCode"] == 403


def test_malformed_body_without_credentials_is_still_the_challenge() -> None:
    fd = build()
    event = make_event("factor", PARAMS, auth=NO_AUTH)
    event["body"] = "CallSid=%FF"
    assert fd.app.handle(event)["statusCode"] == 401


@pytest.mark.parametrize("field,value", [("headers", ["x"]), ("requestContext", "x"),
                                         ("multiValueHeaders", {"Host": None})])
def test_odd_maps_do_not_crash(field: str, value: Any) -> None:
    fd = build()
    event = make_event("status", PARAMS)
    event[field] = value
    assert fd.app.handle(event)["statusCode"] in (200, 204, 401, 403)
