"""The exact-value step of ``obs.ScrubFilter``.

With a call bound, every string of a log record (message, ``%`` args, ``extra`` fields at any
depth, the formatted exception) goes through ``segments.redact_text(current_call_sid(), ...)``
before the pattern scrub. No call bound, or nothing registered for it: the record is unchanged by
this step. Checked through both logger set-ups: the bridge's (our handler and JSON formatter) and
the front door's (the Lambda JSON runtime's handler, which doesn't scrub on its own).
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator

import pytest

from owlbank import obs, segments

SID = "CA" + "0123456789abcdef" * 2
OTHER_SID = "CA" + "fedcba9876543210" * 2
VALUE = "4821"  # 4 digits: the pattern scrub alone keeps it
WORD = "zqxv"  # letters: the pattern scrub alone keeps it


class LambdaLoggerHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Stands in for the Lambda Python runtime's handler (matched by class name)."""


class _RuntimeJson(logging.Formatter):
    """The runtime's JSON format: message plus every ``extra`` key at the top level, no scrubbing."""

    def format(self, record: logging.LogRecord) -> str:
        std = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {"message", "asctime"}
        out = {"level": record.levelname, "message": record.getMessage(), "logger": record.name}
        out.update({k: v for k, v in record.__dict__.items() if k not in std and not k.startswith("_")})
        if record.exc_info:
            out["errorMessage"] = str(record.exc_info[1])
        return json.dumps(out, default=str)


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    yield
    segments.clear_sensitive(SID)
    segments.clear_sensitive(OTHER_SID)


@pytest.fixture(params=["bridge", "frontdoor"])
def setup(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[io.StringIO]:
    """A configured logger set-up writing to a stream: the bridge's, or the front door's Lambda one."""
    monkeypatch.delenv(obs.LOG_LEVEL_ENV, raising=False)
    stream = io.StringIO()
    if request.param == "bridge":
        obs.configure_logging("bridge", stream=stream)
        yield stream
        return
    root = logging.getLogger()
    handler = LambdaLoggerHandler(stream)
    handler.setFormatter(_RuntimeJson())
    root.addHandler(handler)
    try:
        obs.configure_logging("frontdoor")
        yield stream
    finally:
        root.removeHandler(handler)


def _norm(text: str) -> str:
    return segments._norm_text(text)


def test_message_args_extra_nested_and_exception_are_scrubbed(setup: io.StringIO) -> None:
    segments.register_sensitive(SID, [VALUE, WORD])
    log = logging.getLogger("x")
    with obs.call_context(SID):
        log.warning("said 48-21 and %s", "Z Q X V",
                    extra={"field": f"v={VALUE}", "nested": {"deep": [f"a {WORD.upper()} b", {"k": "4 8 2 1"}]}})
        try:
            raise ValueError(f"bad {VALUE} {WORD}")
        except ValueError:
            log.error("boom", exc_info=True)
    text = setup.getvalue()
    assert VALUE not in _norm(text) and WORD not in _norm(text)
    assert "[redacted]" in text


def test_no_call_bound_leaves_the_text_alone(setup: io.StringIO) -> None:
    segments.register_sensitive(SID, [WORD])
    logging.getLogger("x").warning("unbound %s", WORD)
    assert WORD in setup.getvalue()


def test_another_calls_values_are_not_applied(setup: io.StringIO) -> None:
    segments.register_sensitive(OTHER_SID, [VALUE])
    with obs.call_context(SID):
        logging.getLogger("x").warning("mine %s", VALUE, extra={"f": VALUE})
    text = setup.getvalue()
    assert text.count(VALUE) >= 2 and "[redacted]" not in text


def test_the_pattern_scrub_still_runs_after(setup: io.StringIO) -> None:
    segments.register_sensitive(SID, ["123456"])
    with obs.call_context(SID):
        logging.getLogger("x").warning("v 123456 phone +12025550123 run 987654321")
    text = setup.getvalue()
    assert "[redacted]" in text  # the exact step ran first: the value isn't "[num]"
    assert "+12025550123" not in text and "+*******0123" in text
    assert "987654321" not in text and "[num]" in text


def test_the_callers_objects_are_not_changed(setup: io.StringIO) -> None:
    segments.register_sensitive(SID, [VALUE])
    nested = {"deep": [f"x {VALUE}"]}
    args = (f"a {VALUE}",)
    with obs.call_context(SID):
        logging.getLogger("x").warning("m %s", *args, extra={"nested": nested})
    assert nested == {"deep": [f"x {VALUE}"]} and args == (f"a {VALUE}",)
    assert VALUE not in setup.getvalue()


def test_identifiers_are_not_touched() -> None:
    """The bound call SID (exempt from scrubbing) is kept even when a value occurs inside it."""
    segments.register_sensitive(SID, ["abcdef"])
    record = logging.LogRecord("x", logging.WARNING, __file__, 1, "hello abcdef", None, None)
    with obs.call_context(SID):
        obs.ScrubFilter().filter(record)
    assert record.call_sid == SID and record.getMessage() == "hello [redacted]"
