"""The places that catch an error and carry on log it with ``log_caught``: class, code,
the decision taken and a stack of frames, never the exception message. Each case forces a real failure
whose message holds a phone number, a code and a token, and checks the record and that none of it is
anywhere in the logs. The excluded, expected outcomes keep logging without a stack."""

from __future__ import annotations

import logging
import types
from typing import Any

import pytest

from lambdas.auth_service import service as auth_service
from lambdas.auth_service.verify_client import TwilioVerifyClient, VerifyUnavailable
from lambdas.bank_tools import service as bank_service
from lambdas.flex_authorizer import handler as flex_handler
from lambdas.frontdoor.backend import LambdaBackendClient as FrontDoorBackend
from lambdas.level0_tools import handler as level0_handler
from owlbank import secrets
from owlbank.token_seal import KmsSealer, UnsealError
from tests.logutil import fields, text_of
from tests.unit.test_dependency_logging import MESSAGE, FakeTwilio, Raises, TwilioRestException, boom

SID = "CA" + "0123456789abcdef" * 2


# The messages come from a constant, not a literal in the raising line: a stack carries source lines.


def caught(caplog: pytest.LogCaptureFixture, event: str) -> dict[str, Any]:
    (rec,) = [r for r in caplog.records if getattr(r, "event", None) == event or r.getMessage() == event]
    f = fields(rec) | {"_level": rec.levelno}
    assert f["stack"].startswith("Traceback (most recent call last):") and f["decision"]
    assert "error_message" not in f and rec.exc_info is None
    flat = text_of(caplog.records)
    assert "94105" not in flat and "+12025550123" not in flat and "Zq9secret" not in flat and "denied for" not in flat
    return f


def test_front_door_backend_invoke_failure(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    backend = FrontDoorBackend(Raises(boom("TooManyRequestsException")), auth_function="a", bank_function="b")
    assert backend.precheck(call_sid=SID, ani="+12025550123", target_level=1) == "error"
    f = caught(caplog, "backend invoke failed")
    assert f["decision"] == "fallback_error_result" and f["op"] == "precheck" and f["_level"] == logging.WARNING
    assert f["error_class"] == "ClientError" and f["error_code"] == "TooManyRequestsException"


def test_secret_read_failure(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    secrets.clear_cache()
    with pytest.raises(secrets.SecretError):
        secrets.get_secret("arn:aws:secretsmanager:us-east-1:111111111111:secret:x",
                           client=Raises(boom("ResourceNotFoundException")))
    f = caught(caplog, "secret read failed")
    assert f["decision"] == "raise_secret_error" and f["error_code"] == "ResourceNotFoundException"
    assert "ClientError" in f["stack"]


def test_twilio_verify_failures_have_a_stack_and_the_no_pending_check_does_not(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    client = TwilioVerifyClient("VA" + "0" * 32, lambda: FakeTwilio(TwilioRestException(429, 60203)))
    with pytest.raises(VerifyUnavailable):
        client.send("+12025550123", "sms")
    f = caught(caplog, "verify_send_failed")
    assert f["decision"] == "raise_verify_unavailable" and f["error_code"] == "60203" and f["http_status"] == 429
    assert "TwilioRestException" in f["stack"]
    caplog.clear()
    assert TwilioVerifyClient("VA" + "0" * 32, lambda: FakeTwilio(TwilioRestException(404, 20404))).check(
        "+12025550123", "123456") is False  # expected: no pending verification
    (rec,) = [r for r in caplog.records if r.name == "lambdas.auth_service.verify_client"]
    assert fields(rec)["decision"] == "ignored_because_no_pending_verification" and "stack" not in fields(rec)


def test_unseal_failure_names_the_cause_class(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    with pytest.raises(UnsealError):
        KmsSealer(Raises(boom("InvalidCiphertextException")), "key").unseal("YWJj", call_sid=SID)
    f = caught(caplog, "unseal_failed")
    assert f["decision"] == "raise_unseal_error" and f["error_code"] == "InvalidCiphertextException"


def test_flex_authorizer_unexpected_error_and_the_deliberate_denials(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    class Exploding:
        def validate(self, token: str) -> Any:
            raise RuntimeError(MESSAGE + token)

    event = {"methodArn": "arn:aws:execute-api:us-east-1:111111111111:abc/prod/GET/context/x",
             "headers": {"X-Flex-Token": "Zq9secretCallTokenValue_abcdefgh123456"}}
    assert flex_handler.authorize(event, Exploding())["policyDocument"]["Statement"][0]["Effect"] == "Deny"
    f = caught(caplog, "flex authorizer: unexpected error")
    assert f["decision"] == "deny" and f["_level"] == logging.ERROR and f["error_class"] == "RuntimeError"
    caplog.clear()

    class Invalid:
        def validate(self, token: str) -> Any:
            raise flex_handler.InvalidToken("token rejected (HTTP 400)")

    flex_handler.authorize(event, Invalid())  # normal traffic: a rejected token has no stack
    assert all("stack" not in fields(r) for r in caplog.records)


def test_level0_tool_failure(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    def load_content(path: Any = None) -> Any:
        raise RuntimeError(MESSAGE)

    monkeypatch.setattr(level0_handler.service, "load_content", load_content)
    ctx = types.SimpleNamespace(client_context=types.SimpleNamespace(custom={"bedrockAgentCoreToolName": "level0___faq_lookup"}))
    assert level0_handler.handler({"question": "hours?"}, ctx) == {"error": "UNAVAILABLE"}
    f = caught(caplog, "level0_tools tool failed")
    assert f["decision"] == "return_unavailable" and f["tool"] == "faq_lookup" and f["_level"] == logging.ERROR


def test_bank_tools_operation_failure(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    def ani_lookup(deps: Any, ani: Any) -> Any:
        raise RuntimeError(MESSAGE)

    monkeypatch.setattr(bank_service, "ani_lookup", ani_lookup)
    out = bank_service.handle_event(object(), {"op": "ani_lookup", "call_sid": SID, "args": {"ani": "+12025550123"}})
    assert out["error"] == "UNAVAILABLE"
    f = caught(caplog, "bank_tools_operation_failed")
    assert f["decision"] == "return_unavailable" and f["op"] == "ani_lookup" and f["_level"] == logging.ERROR
    assert f["call_sid"] == SID


def test_auth_service_operation_failure(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    class Policy:
        max_level = 3

        def ani_trusted(self, verstat: Any) -> bool:
            raise RuntimeError(MESSAGE)

    deps = auth_service.AuthDeps(store=None, policy=Policy())  # type: ignore[arg-type]
    out = auth_service.precheck(deps, call_sid=SID, ani="+12025550123", target_level=1)
    assert out == {"result": "unavailable"}
    f = caught(caplog, "auth_service_operation_failed")
    assert f["decision"] == "return_unavailable" and f["operation"] == "precheck" and f["_level"] == logging.ERROR
    assert f["call_sid"] == SID


@pytest.mark.parametrize(("exc", "transient"), [
    (boom("ThrottlingException"), True), (boom("KMSInternalException"), True),
    (boom("DependencyTimeoutException"), True), (TimeoutError("read timeout"), True),
    (boom("InvalidCiphertextException"), False),  # a bad blob or another call's context
    (boom("IncorrectKeyException"), False), (boom("AccessDeniedException"), False)])
def test_unseal_error_says_whether_a_retry_may_succeed(exc: Exception, transient: bool,
                                                        caplog: pytest.LogCaptureFixture) -> None:
    """keyed-phone item 3: sensitive.restore retries only a transient unseal failure."""
    caplog.set_level(logging.DEBUG)
    with pytest.raises(UnsealError) as info:
        KmsSealer(Raises(exc), "key").unseal("YWJj", call_sid=SID)
    assert info.value.transient is transient
    assert caught(caplog, "unseal_failed")["transient"] is transient


def test_bad_base64_is_not_a_transient_unseal_failure() -> None:
    class Never:
        def decrypt(self, **_: Any) -> Any:
            raise AssertionError("not reached")

    with pytest.raises(UnsealError) as info:
        KmsSealer(Never(), "key").unseal("not base64!", call_sid=SID)
    assert info.value.transient is False
