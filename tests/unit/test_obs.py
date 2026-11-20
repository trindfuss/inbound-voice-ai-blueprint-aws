"""owlbank.obs: JSON lines, correlation, levels, scrubber wiring, the Lambda runtime
handler, timed_call and instrumented."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from owlbank import obs, security_log
from tests.logutil import fields, with_context

ROOT = Path(__file__).resolve().parents[2]
SID = "CA" + "0123456789abcdef" * 2


@pytest.fixture
def stream() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def configured(stream: io.StringIO, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(obs.LOG_LEVEL_ENV, raising=False)
    obs.configure_logging("unit", stream=stream)
    yield stream


def lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(ln) for ln in stream.getvalue().splitlines() if ln.strip()]


# --------------------------------------------------------------------------- the JSON line


def test_a_line_is_one_json_object_with_the_standard_keys(configured: io.StringIO) -> None:
    with obs.call_context(SID, component="frontdoor", request_id="11111111-aaaa-bbbb-cccc-111111111111"):
        logging.getLogger("x").info("hello", extra={"route": "incoming", "n": 3})
    (line,) = lines(configured)
    assert line["level"] == "INFO" and line["component"] == "frontdoor" and line["event"] == "hello"
    assert line["call_sid"] == SID and line["request_id"] == "11111111-aaaa-bbbb-cccc-111111111111"
    assert line["route"] == "incoming" and line["n"] == 3 and re.fullmatch(r"\d{4}-\d\d-\d\dT[\d:.]+Z", line["ts"])


def test_event_key_names_the_event_and_the_message_follows_it(configured: io.StringIO) -> None:
    obs.emit(logging.getLogger("x"), logging.WARNING, "dependency_call", dependency="kms", level=7, logger="z")
    (line,) = lines(configured)
    assert line["event"] == "dependency_call" and "message" not in line
    assert line["dependency"] == "kms" and line["level"] == "WARNING"  # a clashing field is renamed, not lost
    assert line["level_"] == 7 and line["logger_"] == "z"


def test_exception_becomes_scrubbed_fields_and_never_the_runtime_text(configured: io.StringIO) -> None:
    try:
        raise ValueError("bad value for +12025550123 with token Zq9secretNonceValue_abcdefgh123456")
    except ValueError:
        logging.getLogger("x").exception("it broke")
    (line,) = lines(configured)
    assert line["error_class"] == "ValueError" and "Traceback" in line["stack_trace"]
    assert "+12025550123" not in configured.getvalue() and "Zq9secret" not in configured.getvalue()
    assert "+*******0123" in line["error_message"]


def test_message_args_and_extra_are_scrubbed(configured: io.StringIO) -> None:
    logging.getLogger("x").warning("caller %s said %s", "+12025550123", "call me on 2025550123",
                                   extra={"url": "wss://r.example/ws?n=abc123", "nested": {"p": "+447700900123"}})
    text = configured.getvalue()
    for leaked in ("+12025550123", "2025550123", "n=abc123", "+447700900123"):
        assert leaked not in text
    assert lines(configured)[0]["url"] == "wss://r.example/ws?[redacted]"


def test_log_formatting_never_raises(configured: io.StringIO) -> None:
    logging.getLogger("x").info("bad %d format", "not a number")
    assert lines(configured)[0]["event"].startswith("bad")


# --------------------------------------------------------------------------- levels


def test_default_level_is_info_and_the_variable_sets_it(monkeypatch: pytest.MonkeyPatch, stream: io.StringIO) -> None:
    monkeypatch.delenv(obs.LOG_LEVEL_ENV, raising=False)
    assert obs.configure_logging("unit", stream=stream) == logging.INFO
    monkeypatch.setenv(obs.LOG_LEVEL_ENV, "debug")
    assert obs.configure_logging("unit", stream=stream) == logging.DEBUG
    monkeypatch.setenv(obs.LOG_LEVEL_ENV, "WARN")
    assert obs.configure_logging("unit", stream=stream) == logging.WARNING
    assert logging.getLogger().level == logging.WARNING


def test_an_invalid_level_falls_back_to_info_and_warns_once(monkeypatch: pytest.MonkeyPatch,
                                                            stream: io.StringIO) -> None:
    monkeypatch.setenv(obs.LOG_LEVEL_ENV, "loud")
    assert obs.configure_logging("unit", stream=stream) == logging.INFO
    assert obs.configure_logging("unit", stream=stream) == logging.INFO
    warnings = [ln for ln in lines(stream) if ln["event"] == "invalid_log_level"]
    assert len(warnings) == 1 and warnings[0]["value"] == "loud" and warnings[0]["level"] == "WARNING"


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
def test_third_party_debug_loggers_are_pinned_at_every_level(level: str, stream: io.StringIO) -> None:
    obs.configure_logging("unit", level=level, stream=stream)
    for name in ("botocore", "boto3", "urllib3", "s3transfer", "hpack", "botocore.hooks",
                 "botocore.endpoint", "urllib3.connectionpool"):
        logger = logging.getLogger(name)
        assert logger.getEffectiveLevel() >= logging.WARNING, name
        assert not logger.isEnabledFor(logging.DEBUG)


def test_debug_cannot_reenable_a_pinned_logger_even_if_a_library_sets_its_level(stream: io.StringIO) -> None:
    obs.configure_logging("unit", level="DEBUG", stream=stream)
    logging.getLogger("botocore").setLevel(logging.DEBUG)  # what boto3.set_stream_logger does
    logging.getLogger("botocore.parsers").debug("Response body: b'{\"SecretString\": \"hunter2hunter2\"}'")
    assert "hunter2" not in stream.getvalue() and stream.getvalue() == ""
    logging.getLogger("botocore.parsers").warning("a real problem")
    assert "a real problem" in stream.getvalue()  # warnings still pass


def test_configure_logging_is_idempotent(stream: io.StringIO) -> None:
    root = logging.getLogger()
    before = len(root.handlers)
    obs.configure_logging("unit", stream=stream)
    after_first = len(root.handlers)
    obs.configure_logging("unit", stream=stream)
    obs.configure_logging("unit", stream=stream)
    assert after_first == before + 1 and len(root.handlers) == after_first
    logging.getLogger("x").info("once")
    assert [ln["event"] for ln in lines(stream)] == ["once"]


def test_the_stripping_of_query_strings_stays_installed(configured: io.StringIO) -> None:
    logging.getLogger("tac.server").warning("URL used: %s", "wss://relay.example/ws?n=Zq9secretNonceValue")
    assert "Zq9secret" not in configured.getvalue()


# --------------------------------------------------------------------------- the Lambda runtime handler


class LambdaLoggerHandler(logging.StreamHandler):
    """Stands in for the Python runtime's handler (class name and JSON output as documented)."""


class _JsonFormatter(logging.Formatter):
    """The runtime's JSON format: timestamp, level, message, logger, requestId, and every ``extra``
    key at the top level; an exception as errorType/errorMessage/stackTrace."""

    def format(self, record: logging.LogRecord) -> str:
        std = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {"message", "asctime"}
        out = {"timestamp": "2026-10-12T00:00:00Z", "level": record.levelname, "message": record.getMessage(),
               "logger": record.name, "requestId": "req"}
        out.update({k: v for k, v in record.__dict__.items() if k not in std and not k.startswith("_")})
        if record.exc_info:
            out.update(errorType=record.exc_info[0].__name__, errorMessage=str(record.exc_info[1]),
                       stackTrace=["frame"])
        return json.dumps(out, default=str)


class _TextFormatter(logging.Formatter):
    pass


@pytest.fixture
def runtime_handler(stream: io.StringIO):
    root = logging.getLogger()
    handler = LambdaLoggerHandler(stream)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    yield handler
    root.removeHandler(handler)


def test_under_the_json_runtime_the_runtime_handler_is_kept_and_extras_are_top_level(
        runtime_handler: logging.Handler, stream: io.StringIO) -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    obs.configure_logging("auth_service")
    obs.configure_logging("auth_service")
    assert root.handlers == before  # no second handler: no duplicate lines
    assert isinstance(runtime_handler.formatter, _JsonFormatter)  # the runtime's format is untouched
    with obs.call_context(SID, component="auth_service"):
        obs.emit(logging.getLogger("owlbank.audit"), logging.WARNING, "rate_limited", audit="rate_limited",
                 scope="call", phone="+12025550123")
    out = [json.loads(ln) for ln in stream.getvalue().splitlines()]
    assert len(out) == 1
    (line,) = out
    assert line["event"] == "rate_limited" and line["scope"] == "call" and line["call_sid"] == SID
    assert line["component"] == "auth_service" and line["requestId"] == "req"
    assert "+12025550123" not in stream.getvalue()


def test_under_the_json_runtime_an_exception_is_scrubbed_before_the_runtime_formats_it(
        runtime_handler: logging.Handler, stream: io.StringIO) -> None:
    obs.configure_logging("auth_service")
    try:
        raise KeyError("+12025550123")
    except KeyError:
        logging.getLogger("x").exception("failed")
    (line,) = [json.loads(ln) for ln in stream.getvalue().splitlines()]
    assert "errorMessage" not in line and "errorType" not in line  # the runtime never saw exc_info
    assert line["error_class"] == "KeyError" and "+12025550123" not in stream.getvalue()


def test_under_the_text_runtime_the_handler_gets_the_json_formatter(stream: io.StringIO) -> None:
    root = logging.getLogger()
    handler = LambdaLoggerHandler(stream)
    handler.setFormatter(_TextFormatter("%(message)s"))
    root.addHandler(handler)
    try:
        before = list(root.handlers)
        obs.configure_logging("frontdoor")
        assert root.handlers == before and isinstance(handler.formatter, obs.JsonFormatter)
        logging.getLogger("x").info("hi", extra={"a": 1})
        assert lines(stream)[0]["event"] == "hi"
    finally:
        root.removeHandler(handler)


# --------------------------------------------------------------------------- the stable lines


def test_the_auth_failed_and_untrusted_ani_lines_still_match_the_infra_literal_terms(
        configured: io.StringIO) -> None:
    infra = (ROOT / "infra" / "owlbank_infra" / "observability.py").read_text()
    assert 'AUTH_FAILED_EVENT = "twilio_auth_failed"' in infra
    assert 'OTP_UNTRUSTED_ANI_EVENT = "otp_sent_untrusted_ani"' in infra
    security_log.auth_failed("frontdoor", "signature")
    security_log.otp_sent_untrusted_ani()
    first, second = configured.getvalue().splitlines()
    assert '"twilio_auth_failed"' in first and '"otp_sent_untrusted_ani"' in second  # a quoted literal term
    assert json.loads(first)["component"] == "frontdoor" and json.loads(first)["reason"] == "signature"
    assert json.loads(first)["level"] == "WARNING" and json.loads(second)["component"] == "auth_service"


def test_the_literal_terms_also_match_under_the_lambda_json_format(
        runtime_handler: logging.Handler, stream: io.StringIO) -> None:
    obs.configure_logging("frontdoor")
    security_log.auth_failed("frontdoor", "basic")
    security_log.otp_sent_untrusted_ani()
    text = stream.getvalue()
    assert '"twilio_auth_failed"' in text and '"otp_sent_untrusted_ani"' in text


# --------------------------------------------------------------------------- correlation


class Ctx:
    aws_request_id = "8f507cfc-1234-4697-b07a-ac58fc914c95"


def test_bind_lambda_reads_the_context_the_trace_env_var_and_the_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("_X_AMZN_TRACE_ID", "Root=1-67891233-abcdef012345678912345678;Parent=53995c3f42cd8ad8;Sampled=1")
    tokens = obs.bind_lambda({"call_sid": SID}, Ctx(), component="auth_service")
    try:
        rec = logging.LogRecord("x", logging.INFO, "f", 1, "m", (), None)
        obs.ContextFilter().filter(rec)
        assert (rec.call_sid, rec.request_id, rec.component) == (SID, Ctx.aws_request_id, "auth_service")
        assert rec.trace_id == "1-67891233-abcdef012345678912345678"
    finally:
        obs.unbind(tokens)
    assert obs.current_call_sid() is None and obs.current_request_id() is None


@pytest.mark.parametrize("event", [
    {"call_sid": SID},
    {"queryStringParameters": {"call_sid": SID}},
    {"body": f"AccountSid=AC1&CallSid={SID}&From=%2B1", "httpMethod": "POST"},
    {"body": __import__("base64").b64encode(f"CallSid={SID}&x=1".encode()).decode(), "isBase64Encoded": True},
])
def test_the_call_sid_is_found_in_every_event_shape(event: dict) -> None:
    assert obs.extract_call_sid(event) == SID


@pytest.mark.parametrize("event", [None, "x", {}, {"call_sid": "bad sid with spaces"}, {"call_sid": 5},
                                   {"body": "CallSid=%0AINJECTED"}])
def test_malformed_ids_are_never_bound(event) -> None:
    assert obs.extract_call_sid(event) is None


def test_an_explicit_call_sid_extra_does_not_break_logging_and_is_vetted(configured: io.StringIO) -> None:
    with obs.call_context(SID):
        logging.getLogger("x").info("m", extra={"call_sid": SID})  # a record factory would raise KeyError here
    logging.getLogger("x").info("m", extra={"call_sid": "+12025550123"})
    first, second = lines(configured)
    assert first["call_sid"] == SID and "+12025550123" not in json.dumps(second)


# --------------------------------------------------------------------------- timed_call


def client_error(code: str, message: str = "") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message},
                        "ResponseMetadata": {"HTTPStatusCode": 400, "RequestId": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}},
                       "GetItem")


def test_timed_call_logs_the_error_code_and_never_the_message(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    secret = "User +12025550123 is not authorized; zip 94105"
    with pytest.raises(ClientError), obs.timed_call("dynamodb", "get_item:calls"):
        raise client_error("ProvisionedThroughputExceededException", secret)
    (rec,) = [r for r in caplog.records if r.name == "owlbank.dependency"]
    f = fields(rec)
    assert rec.levelno == logging.WARNING and f["event"] == "dependency_call" and f["outcome"] == "error"
    assert f["dependency"] == "dynamodb" and f["operation"] == "get_item:calls"
    assert f["error_class"] == "ClientError" and f["error_code"] == "ProvisionedThroughputExceededException"
    assert f["http_status"] == 400 and f["aws_request_id"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert "duration_ms" in f
    assert secret not in caplog.text and "94105" not in str(f) and "+12025550123" not in str(f)


def test_timed_call_reads_twilio_style_status_and_code() -> None:
    class TwilioRestException(Exception):
        status, code = 404, 20404

    assert obs.error_code_of(TwilioRestException("never shown")) == "20404"

    class OnlyStatus(Exception):
        status = 503

    assert obs.error_code_of(OnlyStatus()) == "503"
    assert obs.error_code_of(ValueError("x")) is None


def test_timed_call_success_level_slow_calls_and_the_decorators(caplog: pytest.LogCaptureFixture,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    caplog.set_level(logging.DEBUG)

    @obs.timed_call("kms", "decrypt")
    def sync(x: int) -> int:
        return x + 1

    @obs.timed_call("bedrock", "converse", ok_level=logging.DEBUG)
    async def coro() -> str:
        return "ok"

    assert sync(1) == 2 and asyncio.run(coro()) == "ok"
    kms, bedrock = [r for r in caplog.records if r.name == "owlbank.dependency"]
    assert kms.levelno == logging.INFO and bedrock.levelno == logging.DEBUG
    assert fields(kms)["outcome"] == "ok" and fields(bedrock)["operation"] == "converse"

    clock = iter([0.0, 0.6])
    monkeypatch.setattr(obs.time, "perf_counter", lambda: next(clock))
    caplog.clear()
    with obs.timed_call("comprehend", "detect_pii_entities"):
        pass
    (slow,) = caplog.records
    assert slow.levelno == logging.WARNING and fields(slow)["outcome"] == "slow"
    assert fields(slow)["duration_ms"] == 600.0


def test_timed_call_expected_codes_are_debug_not_failures(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    with pytest.raises(ClientError), obs.timed_call(
            "dynamodb", "put_item:calls", expected_codes=("ConditionalCheckFailedException",)):
        raise client_error("ConditionalCheckFailedException")
    (rec,) = caplog.records
    assert rec.levelno == logging.DEBUG and fields(rec)["outcome"] == "condition_failed"


# --------------------------------------------------------------------------- instrumented


def run_handler(handler, event, ctx=None, caplog=None):
    return handler(event, ctx or Ctx())


def invocation(caplog: pytest.LogCaptureFixture) -> dict[str, dict]:
    return {fields(r)["event"]: fields(r) | {"_level": r.levelname, "_call_sid": getattr(r, "call_sid", None)}
            for r in caplog.records if r.name == "owlbank.invocation"}


def test_instrumented_logs_start_and_end_without_the_body(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    with_context(caplog)

    @obs.instrumented("auth_service")
    def handler(event, context=None):
        return {"result": "verified", "token": "t"}

    event = {"op": "verify", "call_sid": SID, "factors": {"zip": "94105"}}
    assert handler(event, Ctx()) == {"result": "verified", "token": "t"}  # return value untouched
    inv = invocation(caplog)
    assert inv["invocation_start"]["op"] == "verify" and inv["invocation_start"]["event_bytes"] == len(
        json.dumps(event, separators=(",", ":")))
    end = inv["invocation_end"]
    assert end["outcome"] == "ok" and end["result_code"] == "verified" and end["duration_ms"] >= 0
    assert inv["invocation_start"]["_call_sid"] == SID == end["_call_sid"]
    assert "94105" not in caplog.text and "factors" not in str([fields(r) for r in caplog.records])


@pytest.mark.parametrize("response, outcome, code", [
    ({"result": "not_verified"}, "ok", "not_verified"),
    ({"result": "unavailable"}, "degraded", "unavailable"),
    ({"error": "AUTH_LEVEL_INSUFFICIENT", "tool": "x"}, "ok", "AUTH_LEVEL_INSUFFICIENT"),
    ({"error": "UNAVAILABLE"}, "degraded", "UNAVAILABLE"),
    ({"statusCode": 200, "body": "<Response/>"}, "ok", "200"),
    ({"statusCode": 403, "body": ""}, "ok", "403"),
    ({"statusCode": 500, "body": ""}, "error", "500"),
    ({"principalId": "u", "policyDocument": {"Statement": [{"Effect": "Deny"}]}}, "ok", "deny"),
    ({"unrelated": 1}, "ok", None),
])
def test_instrumented_classifies_the_response(response: dict, outcome: str, code: str | None,
                                              caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    obs.instrumented("x")(lambda event, context=None: response)({}, Ctx())
    end = invocation(caplog)["invocation_end"]
    assert end["outcome"] == outcome and end.get("result_code") == code
    assert end["_level"] == ("INFO" if outcome == "ok" else "WARNING")


def test_note_and_degraded_reach_the_end_line(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    def handler(event, context=None):
        obs.note(route="factor")
        obs.degraded("validator_error")
        return {"statusCode": 200}

    obs.instrumented("x")(handler)({}, Ctx())
    end = invocation(caplog)["invocation_end"]
    assert end["outcome"] == "degraded" and end["degraded_by"] == "validator_error" and end["route"] == "factor"
    obs.degraded("outside an invocation")  # no-op, no error


def test_instrumented_logs_an_error_line_with_the_call_sid_and_reraises(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    with_context(caplog)

    @obs.instrumented("bank_tools")
    def handler(event, context=None):
        raise RuntimeError("boom for +12025550123")

    with pytest.raises(RuntimeError, match="boom"):
        handler({"op": "get_balance", "call_sid": SID, "args": {"account": "checking"}}, Ctx())
    inv = invocation(caplog)
    err, end = inv["invocation_error"], inv["invocation_end"]
    assert err["_level"] == "ERROR" and err["_call_sid"] == SID and err["op"] == "get_balance"
    rec = next(r for r in caplog.records if fields(r)["event"] == "invocation_error")
    assert rec.exc_info is not None and rec.exc_info[0] is RuntimeError  # the stack trace is on the record
    assert end["outcome"] == "error" and end["result_code"] == "RuntimeError"
    # the formatted line carries the class, the scrubbed message and the stack
    formatted = json.loads(obs.JsonFormatter().format(rec))
    assert formatted["error_class"] == "RuntimeError" and "+12025550123" not in json.dumps(formatted)
    assert "stack_trace" in formatted and formatted["call_sid"] == SID


def test_instrumented_does_not_leak_context_between_invocations(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    with_context(caplog)
    obs.instrumented("x")(lambda event, context=None: {})({"call_sid": SID}, Ctx())
    assert obs.current_call_sid() is None
    obs.instrumented("x")(lambda event, context=None: {})({}, None)
    last = [r for r in caplog.records if r.name == "owlbank.invocation"][-1]
    assert getattr(last, "call_sid", None) in (None, "")


def test_op_can_be_a_function(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    obs.instrumented("level0_tools", op=lambda e, c: "faq_lookup")(lambda e, c=None: {})({}, Ctx())
    assert invocation(caplog)["invocation_start"]["op"] == "faq_lookup"


def test_instrumented_configures_logging_on_the_first_lambda_invocation_only(
        runtime_handler: logging.Handler, stream: io.StringIO, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "owlbank-dev-frontdoor")
    before = list(logging.getLogger().handlers)

    @obs.instrumented("frontdoor")
    def handler(event, context=None):
        return {"statusCode": 200}

    assert not any(isinstance(f, obs.ScrubFilter) for f in runtime_handler.filters)  # not at import
    handler({}, Ctx())
    handler({}, Ctx())
    assert logging.getLogger().handlers == before
    assert sum(isinstance(f, obs.ScrubFilter) for f in runtime_handler.filters) == 1
    events = [json.loads(ln)["event"] for ln in stream.getvalue().splitlines()]
    assert events == ["invocation_start", "invocation_end"] * 2  # one line each, never duplicated
