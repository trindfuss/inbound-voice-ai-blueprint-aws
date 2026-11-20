"""Every dependency call is timed and logged (``dependency_call``) with the AWS or
Twilio error code on failure and never the message."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from botocore.exceptions import ClientError

from lambdas.auth_service.verify_client import TwilioVerifyClient, VerifyUnavailable
from lambdas.frontdoor.backend import LambdaBackendClient as FrontDoorBackend
from owlbank import secrets
from owlbank.pii import ComprehendPiiRedactor, PiiCheckUnavailable
from owlbank.slots import BedrockResolver, ResolverRequest
from owlbank.slots.resolver import SlotView
from owlbank.store import ConditionFailed, DynamoStore, SCHEMAS
from owlbank.token_seal import KmsSealer, UnsealError
from tac_bridge.bridge import LambdaBackendClient as BridgeBackend
from tests.logutil import fields, text_of

MESSAGE = "denied for +12025550123 zip 94105 token Zq9secretCallTokenValue_abcdefgh123456"


def boom(code: str = "ThrottlingException") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": MESSAGE},
                        "ResponseMetadata": {"HTTPStatusCode": 400}}, "Op")


class Raises:
    """Any attribute is a callable that raises the given error (a stub boto3 client or resource)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)

        def call(*a: Any, **k: Any) -> Any:
            raise self._exc

        return call

    def Table(self, name: str) -> Raises:  # noqa: N802 (boto3 resource API)
        return self


def deps(records: list[logging.LogRecord]) -> list[dict[str, Any]]:
    return [fields(r) | {"_level": r.levelname} for r in records if r.name == "owlbank.dependency"]


def assert_failure_logged(caplog: pytest.LogCaptureFixture, dependency: str, operation: str, code: str) -> dict:
    (line,) = [d for d in deps(caplog.records) if d["outcome"] == "error"]
    assert (line["dependency"], line["operation"], line["error_code"]) == (dependency, operation, code), line
    assert line["_level"] == "WARNING" and line["error_class"] and "duration_ms" in line
    flat = text_of(caplog.records)
    assert "94105" not in flat and "+12025550123" not in flat and "Zq9secret" not in flat
    return line


def store(exc: Exception) -> DynamoStore:
    return DynamoStore({t: f"owlbank-{t}" for t in SCHEMAS}, resource=Raises(exc))


@pytest.mark.parametrize("operation, call", [
    ("get_item:calls", lambda s: s.get("calls", {"call_sid": "CA1"})),
    ("put_item:calls", lambda s: s.put("calls", {"call_sid": "CA1"})),
    ("query:tokens", lambda s: s.query("tokens", {"call_sid": "CA1"}, index="call_sid-index")),
    ("update_item:calls", lambda s: s.update("calls", {"call_sid": "CA1"}, {"x": 1})),
    ("delete_item:calls", lambda s: s.delete("calls", {"call_sid": "CA1"})),
    ("batch_get_item:policy", lambda s: s.batch_get("policy", [{"pk": "meta"}])),
    ("scan:catalog", lambda s: s.scan("catalog")),
])
def test_every_dynamodb_operation_is_timed(operation: str, call: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    with pytest.raises(ClientError):
        call(store(boom("ProvisionedThroughputExceededException")))
    assert_failure_logged(caplog, "dynamodb", operation, "ProvisionedThroughputExceededException")


def test_a_failed_condition_is_a_debug_line_not_a_failure(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    s = store(boom("ConditionalCheckFailedException"))
    with pytest.raises(ConditionFailed):
        s.put("calls", {"call_sid": "CA1"}, condition="not_exists")
    (line,) = deps(caplog.records)
    assert line["outcome"] == "condition_failed" and line["_level"] == "DEBUG"


def test_successful_store_reads_are_debug_not_info(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    class Table:
        def get_item(self, **k: Any) -> dict[str, Any]:
            return {}

    class Resource:
        def Table(self, name: str) -> Table:  # noqa: N802
            return Table()

    DynamoStore({t: f"o-{t}" for t in SCHEMAS}, resource=Resource()).get("calls", {"call_sid": "CA1"})
    (line,) = deps(caplog.records)
    assert line["outcome"] == "ok" and line["_level"] == "DEBUG"


def test_lambda_invoke_from_the_front_door(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    backend = FrontDoorBackend(Raises(boom("TooManyRequestsException")), auth_function="a", bank_function="b")
    assert backend.precheck(call_sid="CA1", ani="+12025550123", target_level=1) == "error"
    assert_failure_logged(caplog, "lambda", "invoke:precheck", "TooManyRequestsException")


def test_lambda_invoke_from_the_bridge(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    backend = BridgeBackend("bank", client=Raises(boom("ServiceException")))
    with pytest.raises(ClientError):
        asyncio.run(backend.call("get_balance", token="t", call_sid="CA1", args={"account": "checking"}))
    assert_failure_logged(caplog, "lambda", "invoke:get_balance", "ServiceException")


def test_bedrock_converse_in_the_resolver(caplog: pytest.LogCaptureFixture) -> None:
    """The resolver replaced the classifier; its Converse call is a ``dependency_call``."""
    caplog.set_level(logging.DEBUG)
    resolver = BedrockResolver(lambda budget_ms: Raises(boom("ModelTimeoutException")))
    req = ResolverRequest(language="en-US", question="q",
                          slot=SlotView(name="zip", slot_type="digits", label="ZIP code"),
                          known_options=(), caller_options=(), collected=(), pending=None,
                          utterance="utterance with 94105")
    out = resolver.resolve(req, budget_ms=1500, generative=False)
    assert (out.kind, out.outcome) == ("none", "error")
    line = assert_failure_logged(caplog, "bedrock", "converse", "ModelTimeoutException")
    assert line["error_class"] == "ClientError"
    warning = [r for r in caplog.records if r.name == "owlbank.slots.resolver"][0]
    assert fields(warning)["decision"] == "fallback_none"


def test_kms_encrypt_and_decrypt(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    sealer = KmsSealer(Raises(boom("KMSInternalException")), "key")
    with pytest.raises(ClientError):
        sealer.seal("Zq9secretCallTokenValue_abcdefgh123456", call_sid="CA1")
    assert_failure_logged(caplog, "kms", "encrypt", "KMSInternalException")
    caplog.clear()
    with pytest.raises(UnsealError):
        sealer.unseal("YWJj", call_sid="CA1")
    kinds = {d["operation"]: d for d in deps(caplog.records)}
    assert kinds["decrypt"]["error_code"] == "KMSInternalException"
    assert any(getattr(r, "event", "") == "unseal_failed" and fields(r)["decision"] == "raise_unseal_error"
               for r in caplog.records)


def test_comprehend_in_the_pii_redactor(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    with pytest.raises(PiiCheckUnavailable):
        ComprehendPiiRedactor(Raises(boom("ThrottlingException"))).redact("my name is Jo", language="en-US")
    assert_failure_logged(caplog, "comprehend", "detect_pii_entities", "ThrottlingException")


def test_secrets_manager_reads(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    secrets.clear_cache()
    with pytest.raises(secrets.SecretError):
        secrets.get_secret("arn:aws:secretsmanager:us-east-1:111111111111:secret:x", client=Raises(boom("ResourceNotFoundException")))
    assert_failure_logged(caplog, "secretsmanager", "get_secret_value", "ResourceNotFoundException")


class TwilioRestException(Exception):
    def __init__(self, status: int, code: int) -> None:
        super().__init__(MESSAGE)
        self.status, self.code = status, code


class FakeTwilio:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.verify = self

    @property
    def v2(self) -> FakeTwilio:
        return self

    def services(self, sid: str) -> FakeTwilio:
        return self

    @property
    def verifications(self) -> Raises:
        return Raises(self.exc)

    @property
    def verification_checks(self) -> Raises:
        return Raises(self.exc)


def test_twilio_verify_send_logs_the_twilio_code(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    client = TwilioVerifyClient("VA" + "0" * 32, lambda: FakeTwilio(TwilioRestException(429, 60203)))
    with pytest.raises(VerifyUnavailable):
        client.send("+12025550123", "sms")
    line = assert_failure_logged(caplog, "twilio_verify", "verifications.create", "60203")
    assert line["http_status"] == 429


def test_twilio_verify_check_404_is_expected_and_other_errors_fail(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    assert TwilioVerifyClient("VA" + "0" * 32, lambda: FakeTwilio(TwilioRestException(404, 20404))).check(
        "+12025550123", "123456") is False
    assert [d for d in deps(caplog.records) if d["outcome"] == "error"] == []
    caplog.clear()
    with pytest.raises(VerifyUnavailable):
        TwilioVerifyClient("VA" + "0" * 32, lambda: FakeTwilio(TwilioRestException(500, 20500))).check(
            "+12025550123", "123456")
    assert_failure_logged(caplog, "twilio_verify", "verification_checks.create", "20500")
