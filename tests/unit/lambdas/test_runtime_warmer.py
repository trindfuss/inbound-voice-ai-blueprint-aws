"""``runtime_warmer`` (warmer part).

Stubbed ``bedrock-agentcore`` client only: no network, no AWS account.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.runtime_warmer import handler as warmer
from lambdas.runtime_warmer.handler import CALL_SID_RE, PrewarmResult, WarmerDeps, handle_event, prewarm

from tests.logutil import lines, text_of

ARN = "arn:aws:bedrock-agentcore:us-east-1:111122223333:runtime/owlbank_dev_bridge-AbCdEf1234"
SID = "CA" + "0123456789abcdef" * 2
LOGGER = "lambdas.runtime_warmer.handler"
BODY_MARKER = "body-content-never-logged"


class FakeBody:
    def __init__(self, *, fail: BaseException | None = None) -> None:
        self.read_calls = 0
        self.closed = False
        self._fail = fail

    def read(self) -> bytes:
        self.read_calls += 1
        if self._fail is not None:
            raise self._fail
        return json.dumps({"status": "ready", "note": BODY_MARKER}).encode()

    def close(self) -> None:
        self.closed = True


class FakeAgentCore:
    """Records ``invoke_agent_runtime`` calls; raises ``error`` or returns a streaming body."""

    def __init__(self, *, error: BaseException | None = None, body: FakeBody | None = None,
                 clock: FakeClock | None = None, elapsed_s: float = 0.0) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error
        self.body = body or FakeBody()
        self.clock = clock
        self.elapsed_s = elapsed_s

    def invoke_agent_runtime(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.clock is not None:
            self.clock.now += self.elapsed_s
        if self.error is not None:
            raise self.error
        return {"statusCode": 200, "contentType": "application/json", "runtimeSessionId": kwargs["runtimeSessionId"],
                "response": self.body}


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def client_error(code: str, status: int = 400) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": f"message with {ARN} never logged"},
                        "ResponseMetadata": {"HTTPStatusCode": status}}, "InvokeAgentRuntime")


@pytest.fixture
def warm_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    yield caplog


def warmer_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [f for f in lines(caplog.records, LOGGER) if f["event"].startswith("runtime_prewarm")]


def assert_no_secrets_logged(caplog: pytest.LogCaptureFixture) -> None:
    text = text_of(caplog.records)
    assert ARN not in text and "never logged" not in text and BODY_MARKER not in text


# --------------------------------------------------------------------------- prewarm()


def test_ok_invokes_once_with_the_call_session_and_prewarm_payload(warm_logs: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    client = FakeAgentCore(clock=clock, elapsed_s=1.2345)
    result = prewarm(client, ARN, SID, clock=clock)
    assert result == PrewarmResult("ok", 1234.5)
    assert client.calls == [{"agentRuntimeArn": ARN, "runtimeSessionId": SID, "contentType": "application/json",
                             "payload": b'{"action":"prewarm"}'}]
    assert json.loads(client.calls[0]["payload"]) == {"action": "prewarm"}
    assert client.body.read_calls == 1 and client.body.closed
    [line] = warmer_lines(warm_logs)
    assert line["event"] == "runtime_prewarm" and line["outcome"] == "ok" and line["duration_ms"] == 1234.5
    assert_no_secrets_logged(warm_logs)


def test_retryable_conflict_is_in_progress_not_a_failure(warm_logs: pytest.LogCaptureFixture) -> None:
    client = FakeAgentCore(error=client_error("RetryableConflictException", 409))
    result = prewarm(client, ARN, SID, clock=FakeClock())
    assert result.outcome == "in_progress" and result.code is None
    [line] = warmer_lines(warm_logs)
    assert line["event"] == "runtime_prewarm" and line["outcome"] == "in_progress"
    assert all(r.levelno == logging.INFO for r in warm_logs.records if r.name == LOGGER)
    assert_no_secrets_logged(warm_logs)


@pytest.mark.parametrize("code", ["ThrottlingException", "ServiceQuotaExceededException", "AccessDeniedException",
                                  "ResourceNotFoundException", "RuntimeClientError", "InternalServerException"])
def test_client_errors_log_failed_with_class_and_code(code: str, warm_logs: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    client = FakeAgentCore(error=client_error(code), clock=clock, elapsed_s=0.05)
    result = prewarm(client, ARN, SID, clock=clock)
    assert result == PrewarmResult("failed", 50.0, error_class="ClientError", code=code)
    assert len(client.calls) == 1  # one attempt, no retry
    [line] = warmer_lines(warm_logs)
    assert line["event"] == "runtime_prewarm_failed"
    assert (line["outcome"], line["error_class"], line["code"], line["duration_ms"]) == ("failed", "ClientError",
                                                                                         code, 50.0)
    assert_no_secrets_logged(warm_logs)


@pytest.mark.parametrize("exc", [
    ReadTimeoutError(endpoint_url="https://bedrock-agentcore.us-east-1.amazonaws.com/x"),
    EndpointConnectionError(endpoint_url="https://bedrock-agentcore.us-east-1.amazonaws.com/x"),
    RuntimeError(f"unexpected {ARN}"),
])
def test_non_client_errors_log_failed_without_code(exc: BaseException, warm_logs: pytest.LogCaptureFixture) -> None:
    result = prewarm(FakeAgentCore(error=exc), ARN, SID, clock=FakeClock())
    assert result.outcome == "failed" and result.error_class == type(exc).__name__ and result.code is None
    [line] = warmer_lines(warm_logs)
    assert line["event"] == "runtime_prewarm_failed" and line["code"] is None
    assert "amazonaws.com" not in text_of(warm_logs.records)
    assert_no_secrets_logged(warm_logs)


def test_body_read_failure_is_failed_and_the_stream_is_closed(warm_logs: pytest.LogCaptureFixture) -> None:
    body = FakeBody(fail=ReadTimeoutError(endpoint_url="https://bedrock-agentcore.us-east-1.amazonaws.com/x"))
    client = FakeAgentCore(body=body)
    result = prewarm(client, ARN, SID, clock=FakeClock())
    assert result.outcome == "failed" and result.error_class == "ReadTimeoutError"
    assert body.closed


# --------------------------------------------------------------------------- handle_event()


@pytest.mark.parametrize("event", [
    {}, {"call_sid": None}, {"call_sid": 12}, {"call_sid": ""}, {"call_sid": "CA123"},
    {"call_sid": "XX" + "0" * 32}, {"call_sid": "CA" + "0" * 31}, {"call_sid": "CA" + "0" * 33},
    {"call_sid": "CA" + "A" * 32}, {"call_sid": SID + "\n"}, {"CallSid": SID}, [SID], SID, None, 7,
])
def test_bad_call_sid_is_rejected_without_a_runtime_call(event: Any, warm_logs: pytest.LogCaptureFixture) -> None:
    client = FakeAgentCore()
    out = handle_event(WarmerDeps(client=client, runtime_arn=ARN, clock=FakeClock()), event)
    assert out == {"result": "rejected", "reason": "bad_call_sid"}
    assert client.calls == []
    [line] = warmer_lines(warm_logs)
    assert line["event"] == "runtime_prewarm_rejected" and line["reason"] == "bad_call_sid"


def test_missing_runtime_arn_is_a_failure_without_a_call(warm_logs: pytest.LogCaptureFixture) -> None:
    client = FakeAgentCore()
    out = handle_event(WarmerDeps(client=client, runtime_arn=None), {"call_sid": SID})
    assert out == {"result": "failed"} and client.calls == []
    [line] = warmer_lines(warm_logs)
    assert line["event"] == "runtime_prewarm_failed" and line["error_class"] == "MissingConfiguration"


@pytest.mark.parametrize(("error", "expected"), [
    (None, "ok"), (client_error("RetryableConflictException", 409), "in_progress"),
    (client_error("ThrottlingException", 429), "failed"),
])
def test_handle_event_returns_the_outcome(error: BaseException | None, expected: str) -> None:
    client = FakeAgentCore(error=error)
    assert handle_event(WarmerDeps(client=client, runtime_arn=ARN), {"call_sid": SID}) == {"result": expected}
    assert [c["runtimeSessionId"] for c in client.calls] == [SID]


# --------------------------------------------------------------------------- handler() and wiring


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> FakeAgentCore:
    client = FakeAgentCore()
    monkeypatch.setattr(warmer, "_deps", WarmerDeps(client=client, runtime_arn=ARN))
    return client


def test_handler_ok_and_invocation_lines(installed: FakeAgentCore, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    assert warmer.handler({"call_sid": SID}) == {"result": "ok"}
    assert len(installed.calls) == 1
    [end] = [f for f in lines(caplog.records, "owlbank.invocation") if f["event"] == "invocation_end"]
    assert end["outcome"] == "ok" and end["op"] == "prewarm"


def test_handler_failure_is_degraded_not_raised(installed: FakeAgentCore, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    installed.error = client_error("ThrottlingException", 429)
    assert warmer.handler({"call_sid": SID}) == {"result": "failed"}
    [end] = [f for f in lines(caplog.records, "owlbank.invocation") if f["event"] == "invocation_end"]
    assert end["outcome"] == "degraded"


def test_handler_never_raises_even_if_wiring_fails(monkeypatch: pytest.MonkeyPatch,
                                                    caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    def boom(env: Any = None) -> WarmerDeps:
        raise RuntimeError(f"wiring failed for {ARN}")

    monkeypatch.setattr(warmer, "_deps", None)
    monkeypatch.setattr(warmer, "build_deps", boom)
    assert warmer.handler({"call_sid": SID}) == {"result": "failed"}
    failed = [f for f in lines(caplog.records, LOGGER) if f["event"] == "runtime_prewarm_failed"]
    assert failed and failed[0]["error_class"] == "RuntimeError" and failed[0]["decision"] == "continue_cold"
    assert ARN not in text_of(caplog.records)


def test_build_deps_reads_the_arn_and_creates_no_client() -> None:
    deps = warmer.build_deps({"OWLBANK_RUNTIME_ARN": ARN})
    assert deps.runtime_arn == ARN
    assert warmer.build_deps({}).runtime_arn is None
    assert deps.client._client is None  # LazyClient: nothing built until the first invoke


def test_client_timeouts_and_single_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    client = warmer._make_client()  # building a client makes no network call
    cfg = client.meta.config
    assert (cfg.connect_timeout, cfg.read_timeout) == (2, 25)
    assert cfg.retries["total_max_attempts"] == 1  # one attempt in all, no retry
    assert client.meta.service_model.service_name == "bedrock-agentcore"


# --------------------------------------------------------------------------- PBT: (warmer part)

_valid_sids = st.text(alphabet="0123456789abcdef", min_size=32, max_size=32).map(lambda h: "CA" + h)
_json_leaf = st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=True), st.text(max_size=40),
                       _valid_sids)
_json = st.recursive(_json_leaf, lambda children: st.one_of(
    st.lists(children, max_size=4), st.dictionaries(st.text(max_size=12), children, max_size=4)), max_leaves=12)
_events = st.one_of(
    _json,
    st.fixed_dictionaries({"call_sid": st.one_of(_valid_sids, _json)}, optional={"extra": _json}),
    st.builds(lambda sid, junk: {"call_sid": sid + junk}, _valid_sids, st.text(min_size=1, max_size=4)),
)
_errors = st.one_of(
    st.none(),
    st.sampled_from(["RetryableConflictException", "ThrottlingException", "ServiceQuotaExceededException",
                     "ValidationException", "InternalServerException", ""]).map(client_error),
    st.sampled_from([RuntimeError("x"), ValueError("y"), KeyError("z"),
                     ReadTimeoutError(endpoint_url="https://example.invalid/x")]),
)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(event=_events, error=_errors, arn=st.one_of(st.just(ARN), st.none()))
def test_any_event_returns_and_invokes_only_for_a_call_sid(event: Any, error: BaseException | None,
                                                                 arn: str | None,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """ (warmer part): for any payload the handler returns without raising, and the Runtime is
    invoked (once, on that session) only when the event's ``call_sid`` matches ``CALL_SID_RE``."""
    client = FakeAgentCore(error=error)
    monkeypatch.setattr(warmer, "_deps", WarmerDeps(client=client, runtime_arn=arn))
    out = warmer.handler(event)
    assert isinstance(out, dict) and out["result"] in {"ok", "in_progress", "failed", "rejected"}
    sid = event.get("call_sid") if isinstance(event, dict) else None
    valid = isinstance(sid, str) and CALL_SID_RE.fullmatch(sid) is not None
    if valid and arn:
        assert [c["runtimeSessionId"] for c in client.calls] == [sid]
        assert all(c["agentRuntimeArn"] == arn and c["payload"] == b'{"action":"prewarm"}' for c in client.calls)
    else:
        assert client.calls == []
        assert out["result"] == ("failed" if valid else "rejected")
