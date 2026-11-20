"""Every Lambda handler logs ``invocation_start`` and ``invocation_end``; an unhandled
exception logs ``invocation_error`` with the call SID (where the event has one) and re-raises;
no sensitive value reaches a log record."""

from __future__ import annotations

import base64
import logging
from types import SimpleNamespace
from collections.abc import Callable
from typing import Any

import pytest

from lambdas.auth_service import handler as auth_handler
from lambdas.bank_tools import handler as bank_handler
from lambdas.context_api import handler as context_handler
from lambdas.flex_authorizer import handler as flex_handler
from lambdas.frontdoor import handler as frontdoor_handler
from lambdas.level0_tools import handler as level0_handler
from tests.logutil import fields, text_of, with_context

SID = "CA" + "0123456789abcdef" * 2
SECRET_ZIP, SECRET_TOKEN = "94105", "Zq9secretCallTokenValue_abcdefgh123456"
REQUEST_ID = "8f507cfc-1234-4697-b07a-ac58fc914c95"
CTX = SimpleNamespace(aws_request_id=REQUEST_ID,
                      client_context=SimpleNamespace(custom={"bedrockAgentCoreToolName": "level0___faq_lookup"}))


def api_event(route: str, body: str) -> dict[str, Any]:
    return {"httpMethod": "POST", "path": f"/voice/{route}", "body": body, "headers": {}}


# name -> (entry point, event, patch that makes it work, patch that makes it raise, expected op, call SID in event?)
def _ok(monkeypatch: pytest.MonkeyPatch, module: Any, attr: str, result: Any) -> None:
    monkeypatch.setattr(module, attr, lambda *a, **k: result)


def _boom(monkeypatch: pytest.MonkeyPatch, module: Any, attr: str) -> None:
    def raise_(*a: Any, **k: Any) -> Any:
        raise RuntimeError(f"boom with {SECRET_ZIP} and {SECRET_TOKEN}")

    monkeypatch.setattr(module, attr, raise_)


class Case:
    def __init__(self, name: str, entry: Callable[..., Any], event: dict[str, Any], op: str | None,
                 has_sid: bool, setup: Callable[[pytest.MonkeyPatch, bool], None]) -> None:
        self.name, self.entry, self.event, self.op, self.has_sid, self.setup = name, entry, event, op, has_sid, setup

    def __repr__(self) -> str:
        return self.name


def _auth(mp: pytest.MonkeyPatch, fail: bool) -> None:
    mp.setattr(auth_handler, "_deps", object())
    (_boom if fail else _ok)(mp, auth_handler.service, "handle_event", *(() if fail else ({"result": "verified"},)))


def _bank(mp: pytest.MonkeyPatch, fail: bool) -> None:
    mp.setattr(bank_handler, "_deps", object())
    (_boom if fail else _ok)(mp, bank_handler.service, "handle_event", *(() if fail else ({"data": {}, "tool": "get_balance"},)))


def _frontdoor(mp: pytest.MonkeyPatch, fail: bool) -> None:
    def handle(event: Any, **_kw: Any) -> Any:
        if fail:
            raise RuntimeError(f"boom with {SECRET_ZIP}")
        return {"statusCode": 200, "headers": {}, "body": "<Response/>"}

    mp.setattr(frontdoor_handler, "_app", SimpleNamespace(handle=handle))


def _context(mp: pytest.MonkeyPatch, fail: bool) -> None:
    mp.setattr(context_handler, "_store", object())
    if fail:
        _boom(mp, context_handler, "get_context")
    else:
        _ok(mp, context_handler, "get_context", {"statusCode": 404, "headers": {}, "body": ""})


def _flex(mp: pytest.MonkeyPatch, fail: bool) -> None:
    mp.setattr(flex_handler, "_default_validator", object())
    if fail:
        _boom(mp, flex_handler, "authorize")
    else:
        _ok(mp, flex_handler, "authorize", {"principalId": "u", "policyDocument": {"Statement": [{"Effect": "Allow"}]}})


def _level0(mp: pytest.MonkeyPatch, fail: bool) -> None:
    if fail:
        calls = iter(["faq_lookup"])  # the op name is read first; the handler's own lookup then raises

        def tool_name(context: Any) -> str:
            try:
                return next(calls)
            except StopIteration:
                raise RuntimeError(f"boom with {SECRET_ZIP} and {SECRET_TOKEN}") from None

        mp.setattr(level0_handler, "tool_name", tool_name)
    else:
        _ok(mp, level0_handler.service, "load_content", {})
        _ok(mp, level0_handler.service, "dispatch", {"tool": "faq_lookup", "data": {}})


CASES = [
    Case("auth_service", auth_handler.handler,
         {"op": "verify", "call_sid": SID, "factors": {"zip": SECRET_ZIP}, "token": SECRET_TOKEN}, "verify", True, _auth),
    Case("bank_tools", bank_handler.handler,
         {"op": "get_balance", "call_sid": SID, "token": SECRET_TOKEN, "args": {"account": "checking"}},
         "get_balance", True, _bank),
    Case("frontdoor", frontdoor_handler.lambda_handler,
         api_event("factor", f"CallSid={SID}&Digits={SECRET_ZIP}"), "factor", True, _frontdoor),
    Case("frontdoor (base64 body)", frontdoor_handler.lambda_handler,
         {**api_event("incoming", base64.b64encode(f"CallSid={SID}&From=%2B12025550123".encode()).decode()),
          "isBase64Encoded": True}, "incoming", True, _frontdoor),
    Case("context_api", context_handler.handler,
         {"httpMethod": "GET", "pathParameters": {"journal_ref": SECRET_TOKEN},
          "queryStringParameters": {"call_sid": SID}}, "get_context", True, _context),
    Case("flex_authorizer", flex_handler.handler,
         {"methodArn": "arn:aws:execute-api:us-east-1:111111111111:abc/prod/GET/context/x",
          "headers": {"X-Flex-Token": SECRET_TOKEN}}, "authorize", False, _flex),
    Case("level0_tools", level0_handler.handler, {"question": f"my zip is {SECRET_ZIP}"}, "faq_lookup", False, _level0),
]


@pytest.mark.parametrize("case", CASES, ids=repr)
def test_a_handler_logs_start_and_end(case: Case, monkeypatch: pytest.MonkeyPatch,
                                      caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    with_context(caplog)
    case.setup(monkeypatch, False)
    case.entry(case.event, CTX)
    inv = {fields(r)["event"]: r for r in caplog.records if r.name == "owlbank.invocation"}
    assert set(inv) == {"invocation_start", "invocation_end"}
    assert fields(inv["invocation_start"])["op"] == case.op
    assert fields(inv["invocation_end"])["outcome"] in ("ok", "degraded") and "duration_ms" in fields(inv["invocation_end"])
    assert all(getattr(r, "request_id", None) == REQUEST_ID for r in inv.values())
    if case.has_sid:
        assert all(getattr(r, "call_sid", None) == SID for r in inv.values())
    assert SECRET_ZIP not in text_of(caplog.records) and SECRET_TOKEN not in text_of(caplog.records)


@pytest.mark.parametrize("case", CASES, ids=repr)
def test_a_forced_exception_logs_an_error_line_with_the_call_sid_and_reraises(
        case: Case, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    with_context(caplog)
    case.setup(monkeypatch, True)
    with pytest.raises(RuntimeError):
        case.entry(case.event, CTX)
    errors = [r for r in caplog.records if r.name == "owlbank.invocation" and fields(r)["event"] == "invocation_error"]
    assert len(errors) == 1
    err = errors[0]
    assert err.levelno == logging.ERROR and fields(err)["op"] == case.op
    assert err.exc_info is not None and err.exc_info[0] is RuntimeError
    if case.has_sid:
        assert getattr(err, "call_sid", None) == SID
    end = [fields(r) for r in caplog.records if fields(r)["event"] == "invocation_end"] if False else [
        fields(r) for r in caplog.records if getattr(r, "event", None) == "invocation_end"]
    assert end and end[-1]["outcome"] == "error"
    # The record keeps the exception for the formatter, which scrubs it; nothing sensitive is in a field.
    flat = text_of(caplog.records)
    assert SECRET_TOKEN not in flat.replace(f"boom with {SECRET_ZIP} and {SECRET_TOKEN}", "")


@pytest.mark.parametrize("case", CASES, ids=repr)
def test_the_formatted_error_line_has_class_scrubbed_message_and_stack(
        case: Case, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from owlbank import obs

    caplog.set_level(logging.DEBUG)
    with_context(caplog)
    case.setup(monkeypatch, True)
    with pytest.raises(RuntimeError):
        case.entry(case.event, CTX)
    rec = next(r for r in caplog.records if getattr(r, "event", None) == "invocation_error")
    scrub = obs.ScrubFilter()
    scrub.filter(rec)
    line = obs.JsonFormatter().format(rec)
    assert '"error_class":"RuntimeError"' in line and "stack_trace" in line
    assert SECRET_TOKEN not in line  # the token-like value in the exception message is masked
