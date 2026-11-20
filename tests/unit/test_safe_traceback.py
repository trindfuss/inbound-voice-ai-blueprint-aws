"""Caught errors log a stack trace without the exception message.

``safe_traceback`` (frames and class names only), ``log_caught`` (class, code, decision, stack), the
stack on a failed ``timed_call``, the unhandled-error line, and a structural guard: no code outside
``owlbank.obs`` logs an exception with ``log.exception`` or ``exc_info``."""

from __future__ import annotations

import ast
import io
import json
import logging
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from owlbank import obs, security_log
from owlbank.obs import log_caught, safe_traceback, timed_call
from tests.logutil import fields, text_of

ROOT = Path(__file__).resolve().parents[2]
MESSAGE = "denied for +12025550123 zip 94105 token Zq9secretCallTokenValue_abcdefgh123456"


class RootCauseError(Exception):
    pass


class WrapperError(Exception):
    pass


def _fail_inner(secret_local: str) -> None:
    other_local = "local-value-that-must-not-appear"  # noqa: F841
    raise RootCauseError(secret_local)


def _fail_outer() -> None:
    try:
        _fail_inner(MESSAGE)
    except RootCauseError as inner:
        raise WrapperError(MESSAGE) from inner


def _caught() -> BaseException:
    try:
        _fail_outer()
    except WrapperError as exc:
        return exc
    raise AssertionError


def _client_error() -> ClientError:
    return ClientError({"Error": {"Code": "ThrottlingException", "Message": MESSAGE},
                        "ResponseMetadata": {"HTTPStatusCode": 400, "RequestId": "8f507cfc-1234-4697-b07a-ac58fc914c95"}},
                       "GetItem")


# --------------------------------------------------------------------------- safe_traceback


def test_frames_are_present_with_file_line_function_and_source() -> None:
    out = safe_traceback(_caught())
    assert out.startswith("Traceback (most recent call last):")
    assert "test_safe_traceback.py" in out and ", in _fail_inner" in out and ", in _fail_outer" in out
    assert "raise RootCauseError(secret_local)" in out and "raise WrapperError(MESSAGE) from inner" in out
    assert "line " in out


def test_a_chain_gives_class_names_only_oldest_first() -> None:
    out = safe_traceback(_caught())
    qual = f"{RootCauseError.__module__}.RootCauseError"
    assert qual in out and f"{WrapperError.__module__}.WrapperError" in out
    assert out.index(qual) < out.index("WrapperError")  # the cause comes first
    assert "The above exception led to this one" in out
    assert "94105" not in out and "12025550123" not in out and "Zq9secret" not in out and "denied for" not in out


def test_builtin_classes_are_not_prefixed_and_a_missing_traceback_gives_the_class_only() -> None:
    assert safe_traceback(ValueError("x")) == "Traceback (most recent call last):\nValueError"
    assert safe_traceback(_client_error()).endswith("botocore.exceptions.ClientError")


def test_no_local_variable_value_appears() -> None:
    assert "local-value-that-must-not-appear" not in safe_traceback(_caught())


def test_the_context_is_followed_even_after_raise_from_none() -> None:
    def go() -> None:
        try:
            raise RootCauseError(MESSAGE)
        except RootCauseError:
            raise WrapperError(MESSAGE) from None

    with pytest.raises(WrapperError) as ei:
        go()
    out = safe_traceback(ei.value)
    assert "RootCauseError" in out and "WrapperError" in out and MESSAGE not in out


def test_the_chain_is_capped_at_five_and_a_cycle_ends() -> None:
    exc: BaseException | None = None
    for i in range(9):
        try:
            raise ValueError(f"m{i}") from exc
        except ValueError as caught:
            exc = caught
    assert exc is not None
    assert safe_traceback(exc).count("Traceback (most recent call last):") == 5
    a, b = ValueError("a"), ValueError("b")
    a.__cause__, b.__cause__ = b, a
    assert safe_traceback(a).count("Traceback (most recent call last):") == 2


def test_the_output_is_capped_at_8kb_and_keeps_the_end() -> None:
    def deep(n: int) -> None:
        if n == 0:
            raise RootCauseError("x")
        deep(n - 1)

    try:
        deep(500)
    except RootCauseError as exc:
        out = safe_traceback(exc)
    assert len(out) <= 8192 and out.endswith("RootCauseError") and "deep" in out


def test_source_lines_go_through_the_scrubber() -> None:
    def go() -> None:
        raise RootCauseError() if "+12025550123" else None  # a literal number in the source line

    with pytest.raises(RootCauseError) as ei:
        go()
    out = safe_traceback(ei.value)
    assert "+12025550123" not in out and "+*******0123" in out


def test_an_alarm_event_name_in_a_source_line_cannot_match_a_metric_filter() -> None:
    def go() -> None:
        raise RootCauseError  # twilio_auth_failed otp_sent_untrusted_ani

    with pytest.raises(RootCauseError) as ei:
        go()
    out = safe_traceback(ei.value)
    assert security_log.AUTH_FAILED_EVENT not in out and security_log.OTP_UNTRUSTED_EVENT not in out
    assert "twilio-auth_failed" in out


def test_the_alarm_terms_cover_every_alarm_event() -> None:
    assert {security_log.AUTH_FAILED_EVENT, security_log.OTP_UNTRUSTED_EVENT} <= set(obs._ALARM_TERMS)  # noqa: SLF001


# --------------------------------------------------------------------------- log_caught


def test_log_caught_logs_class_code_decision_identifiers_and_stack(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    log = logging.getLogger("t.caught")
    log_caught(log, "thing failed", _client_error(), decision="fallback_x", operation="get_item", call_sid="CA" + "1" * 32)
    (rec,) = caplog.records
    f = fields(rec)
    assert rec.levelno == logging.WARNING and f["event"] == "thing failed"
    assert (f["error_class"], f["error_code"], f["http_status"]) == ("ClientError", "ThrottlingException", 400)
    assert f["decision"] == "fallback_x" and f["operation"] == "get_item" and f["call_sid"] == "CA" + "1" * 32
    assert f["stack"].startswith("Traceback") and "ClientError" in f["stack"]
    assert rec.exc_info is None  # nothing for a formatter to turn into a message
    flat = text_of(caplog.records)
    assert "94105" not in flat and "Zq9secret" not in flat and "denied for" not in flat


def test_log_caught_takes_a_level_and_skips_the_work_when_it_is_off(caplog: pytest.LogCaptureFixture,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    log = logging.getLogger("t.caught.level")
    caplog.set_level(logging.ERROR)
    calls: list[int] = []
    monkeypatch.setattr(obs, "safe_traceback", lambda exc: calls.append(1) or "")
    log_caught(log, "quiet", ValueError("x"), decision="d")  # WARNING is below ERROR: nothing, no stack built
    assert caplog.records == [] and calls == []
    log_caught(log, "loud", ValueError("x"), decision="d", level=logging.ERROR)
    assert [r.levelno for r in caplog.records] == [logging.ERROR] and calls == [1]


def test_log_caught_in_the_json_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(obs.LOG_LEVEL_ENV, raising=False)
    stream = io.StringIO()
    obs.configure_logging("unit", stream=stream)
    try:
        _fail_outer()
    except WrapperError as exc:
        log_caught(logging.getLogger("t.json"), "wrapped failure", exc, decision="retry", level=logging.ERROR)
    (line,) = [json.loads(ln) for ln in stream.getvalue().splitlines() if ln.strip()]
    assert line["event"] == "wrapped failure" and line["level"] == "ERROR" and line["decision"] == "retry"
    assert line["error_class"] == "WrapperError" and "error_message" not in line and "stack_trace" not in line
    assert "RootCauseError" in line["stack"] and "_fail_inner" in line["stack"]
    assert "94105" not in stream.getvalue() and "Zq9secret" not in stream.getvalue()


# --------------------------------------------------------------------------- timed_call and instrumented


def test_a_failed_timed_call_logs_the_stack_without_the_message(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    with pytest.raises(ClientError), timed_call("dynamodb", "get_item"):
        raise _client_error()
    (rec,) = [r for r in caplog.records if r.name == "owlbank.dependency"]
    f = fields(rec)
    assert f["outcome"] == "error" and f["error_code"] == "ThrottlingException"
    assert "ClientError" in f["stack"] and "raise _client_error()" in f["stack"]
    assert "94105" not in text_of(caplog.records)


def test_an_expected_failure_and_a_success_have_no_stack(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    cond = ClientError({"Error": {"Code": "ConditionalCheckFailedException", "Message": "x"}}, "PutItem")
    with pytest.raises(ClientError), timed_call("dynamodb", "put_item", expected_codes=("ConditionalCheckFailedException",)):
        raise cond
    with timed_call("dynamodb", "get_item"):
        pass
    recs = [fields(r) for r in caplog.records if r.name == "owlbank.dependency"]
    assert [r["outcome"] for r in recs] == ["condition_failed", "ok"]
    assert all("stack" not in r for r in recs)


def test_an_unhandled_error_keeps_its_scrubbed_message_but_the_stack_has_none(configured_stream) -> None:
    @obs.instrumented("unit", op="boom")
    def handler(event: object, context: object = None) -> None:
        raise RootCauseError(MESSAGE)

    with pytest.raises(RootCauseError):
        handler({})
    (line,) = [json.loads(ln) for ln in configured_stream.getvalue().splitlines()
               if '"invocation_error"' in ln]
    assert "+*******0123" in line["error_message"]  # design: the scrubbed message, for unhandled errors only
    assert "Zq9secret" not in configured_stream.getvalue() and "Zq9secret" not in line["error_message"]
    assert "RootCauseError" in line["stack_trace"] and "94105" not in line["stack_trace"]
    assert "+12025550123" not in line["stack_trace"] and "denied for" not in line["stack_trace"]


@pytest.fixture
def configured_stream(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(obs.LOG_LEVEL_ENV, raising=False)
    stream = io.StringIO()
    obs.configure_logging("unit", stream=stream)
    return stream


# --------------------------------------------------------------------------- the guard


def _python_files() -> list[Path]:
    return [p for d in ("src", "scripts") for p in (ROOT / d).rglob("*.py") if p != ROOT / "src/owlbank/obs.py"]


def test_no_code_logs_a_caught_exception_with_its_message() -> None:
    """Structural: ``log.exception(...)``, ``exc_info=`` and ``traceback.format_exc`` would log the message
    (through the formatter's ``error_message``). Caught errors go through ``obs.log_caught``."""
    offenders: list[str] = []
    for path in _python_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute) and fn.attr in ("exception", "format_exc", "print_exc"):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} .{fn.attr}()")
                offenders += [f"{path.relative_to(ROOT)}:{node.lineno} exc_info=" for kw in node.keywords if kw.arg == "exc_info"]
    assert not offenders, offenders
