"""``StopRuntimeSession`` at call end.

One outcome per attempt (``ok``, ``not_found``, ``error:<class>``), never an exception, never the
ARN or an error message in a log; the ``#runtime-stop`` claim admits exactly one attempt per call.
No AWS call: the client is a fake, the errors are botocore's own classes.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError

from lambdas.frontdoor import runtime_sessions
from lambdas.frontdoor.runtime_sessions import RuntimeSessions, claim_key, claim_stop
from owlbank.store import MemoryStore, StoreError
from tests.logutil import lines, text_of

ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/owlbank_dev_bridge-AbC123xyz0"
SID = "CA" + "0123456789abcdef" * 2
SECRET_MESSAGE = "detail-that-must-not-be-logged-x9y8"
LOGGER = "lambdas.frontdoor.runtime_sessions"


class FakeClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def stop_runtime_session(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"runtimeSessionId": kwargs["runtimeSessionId"], "statusCode": 200}


def client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": SECRET_MESSAGE}}, "StopRuntimeSession")


def stopped(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [x for x in lines(caplog.records, LOGGER) if x["event"] == "runtime_session_stopped"]


def test_timeouts_are_two_seconds() -> None:
    assert runtime_sessions.CONNECT_TIMEOUT_S == 2 and runtime_sessions.READ_TIMEOUT_S == 2


def test_ok(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    client = FakeClient()
    assert RuntimeSessions(client, ARN).stop(SID) == "ok"
    assert client.calls == [{"agentRuntimeArn": ARN, "runtimeSessionId": SID}]
    [line] = stopped(caplog)
    assert line["outcome"] == "ok" and line["call_sid"] == SID
    assert "duration_ms" in line
    assert ARN not in text_of(caplog.records)


def test_not_found(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    client = FakeClient(client_error("ResourceNotFoundException"))
    assert RuntimeSessions(client, ARN).stop(SID) == "not_found"
    [line] = stopped(caplog)
    assert line["outcome"] == "not_found"
    assert SECRET_MESSAGE not in text_of(caplog.records)


@pytest.mark.parametrize(("error", "outcome", "code"), [
    (client_error("ThrottlingException"), "error:ClientError", "ThrottlingException"),
    (client_error("AccessDeniedException"), "error:ClientError", "AccessDeniedException"),
    (ConnectTimeoutError(endpoint_url="https://bedrock-agentcore.us-east-1.amazonaws.com"),
     "error:ConnectTimeoutError", None),
    (ReadTimeoutError(endpoint_url="https://bedrock-agentcore.us-east-1.amazonaws.com"),
     "error:ReadTimeoutError", None),
    (RuntimeError(SECRET_MESSAGE), "error:RuntimeError", None),
])
def test_errors_are_logged_by_class_and_never_raise(caplog: pytest.LogCaptureFixture, error: Exception,
                                                    outcome: str, code: str | None) -> None:
    caplog.set_level(logging.DEBUG)
    assert RuntimeSessions(FakeClient(error), ARN).stop(SID) == outcome
    [line] = stopped(caplog)
    assert line["outcome"] == outcome
    assert line.get("error_code") == code
    assert line["decision"] == "idle_timeout_stops_it"
    text = text_of(caplog.records)
    assert SECRET_MESSAGE not in text and "bedrock-agentcore.us-east-1" not in text


def test_a_modeled_error_class_is_its_own_name() -> None:
    import boto3

    client = boto3.client("bedrock-agentcore", region_name="us-east-1",
                          aws_access_key_id="x", aws_secret_access_key="y")  # noqa: S106 - fake, no call
    throttled = client.exceptions.ThrottlingException(
        {"Error": {"Code": "ThrottlingException", "Message": SECRET_MESSAGE}}, "StopRuntimeSession")
    missing = client.exceptions.ResourceNotFoundException(
        {"Error": {"Code": "ResourceNotFoundException", "Message": SECRET_MESSAGE}}, "StopRuntimeSession")
    assert RuntimeSessions(FakeClient(throttled), ARN).stop(SID) == "error:ThrottlingException"
    assert RuntimeSessions(FakeClient(missing), ARN).stop(SID) == "not_found"


@pytest.mark.parametrize("sid", ["", "CAfake", "CA" + "A" * 32, "CA" + "0" * 32 + "\n"])
def test_not_a_callsid_is_never_sent(sid: str) -> None:
    client = FakeClient()
    assert RuntimeSessions(client, ARN).stop(sid) == "error:ValueError"
    assert client.calls == []


@pytest.mark.parametrize("arn", ["", "arn:aws:lambda:us-east-1:123456789012:function:x",
                                 ARN + "/runtime-endpoint/DEFAULT"])
def test_constructor_rejects_anything_but_a_runtime_arn(arn: str) -> None:
    with pytest.raises(ValueError) as err:
        RuntimeSessions(FakeClient(), arn)
    assert arn == "" or arn not in str(err.value)


# ---------------------------------------------------------------------------- the claim


def test_claim_admits_the_first_request_only() -> None:
    store = MemoryStore()
    assert claim_stop(store, SID, expires_at=1_900_000_000)
    assert not claim_stop(store, SID, expires_at=1_900_000_000)
    assert store.get("calls", {"call_sid": claim_key(SID)}) == {
        "call_sid": SID + "#runtime-stop", "expires_at": 1_900_000_000}
    assert claim_stop(store, "CA" + "f" * 32, expires_at=1)  # per call


def test_claim_key_is_apart_from_the_other_call_items() -> None:
    key = claim_key(SID)
    assert key != SID and not key.endswith("#session") and "#gather#" not in key


def test_a_store_error_counts_as_not_claimed(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    class Broken(MemoryStore):
        def put(self, table: str, item: Any, *, condition: Any = None) -> None:
            raise StoreError("dynamodb unavailable")

    assert not claim_stop(Broken(), SID, expires_at=1)
    [line] = [x for x in lines(caplog.records, LOGGER) if x["event"] == "runtime_session_stop_claim_failed"]
    assert line["decision"] == "idle_timeout_stops_it" and line["error_class"] == "StoreError"
