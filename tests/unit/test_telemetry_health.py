"""Call observability: :class:`ExportHealth` and :class:`OncePer`.

On a fake clock: a failing or raising inner exporter is counted, ``trace_export_failed`` is logged
at most once per 60 s with the error class and the count dropped since the last line, and nothing
is ever raised.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from owlbank.telemetry.health import ExportHealth, OncePer

from tests.logutil import lines

LOGGER = "test.telemetry.health"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Inner(SpanExporter):
    def __init__(self) -> None:
        self.mode: str = "ok"
        self.calls = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.calls += 1
        if self.mode == "raise":
            raise ConnectionError("xray.us-east-1 unreachable")
        return SpanExportResult.FAILURE if self.mode == "fail" else SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        raise RuntimeError("shutdown")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        raise RuntimeError("flush")


def _spans(n: int) -> list[Any]:
    return [ReadableSpan(name=f"s{i}") for i in range(n)]


@pytest.fixture
def rig(caplog: pytest.LogCaptureFixture) -> tuple[ExportHealth, Inner, Clock]:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    inner, clock = Inner(), Clock()
    return ExportHealth(inner, logging.getLogger(LOGGER), clock=clock), inner, clock


def _logged(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str, int]]:
    return [(ln["event"], ln["error_class"], ln["dropped"]) for ln in lines(caplog.records, LOGGER)]


def test_success_passes_through_silently(rig: Any, caplog: pytest.LogCaptureFixture) -> None:
    health, _, _ = rig
    assert health.export(_spans(3)) is SpanExportResult.SUCCESS
    assert health.dropped == 0 and _logged(caplog) == []


def test_logs_once_per_minute_with_the_count_since_the_last_line(rig: Any,
                                                                 caplog: pytest.LogCaptureFixture) -> None:
    health, inner, clock = rig
    inner.mode = "raise"
    assert health.export(_spans(2)) is SpanExportResult.FAILURE
    clock.now += 30
    assert health.export(_spans(5)) is SpanExportResult.FAILURE
    clock.now += 29.9
    inner.mode = "fail"
    assert health.export(_spans(1)) is SpanExportResult.FAILURE
    assert _logged(caplog) == [("trace_export_failed", "ConnectionError", 2)]
    clock.now += 0.1  # 60 s after the first line
    assert health.export(_spans(4)) is SpanExportResult.FAILURE
    assert _logged(caplog) == [("trace_export_failed", "ConnectionError", 2),
                               ("trace_export_failed", "ExportFailure", 10)]
    assert (health.dropped, health.failures) == (12, 4)
    # The exception's message is never logged.
    assert all("unreachable" not in r.getMessage() and "unreachable" not in str(r.__dict__)
               for r in caplog.records)


def test_recovery_then_failure_logs_again_after_the_interval(rig: Any, caplog: pytest.LogCaptureFixture) -> None:
    health, inner, clock = rig
    inner.mode = "fail"
    health.export(_spans(1))
    inner.mode = "ok"
    clock.now += 120
    assert health.export(_spans(1)) is SpanExportResult.SUCCESS
    inner.mode = "fail"
    health.export(_spans(3))
    assert _logged(caplog) == [("trace_export_failed", "ExportFailure", 1),
                               ("trace_export_failed", "ExportFailure", 3)]


def test_never_raises(rig: Any) -> None:
    health, inner, _ = rig
    inner.mode = "raise"
    for _ in range(5):
        assert health.export(_spans(1)) is SpanExportResult.FAILURE
    health.shutdown()
    assert health.force_flush() is False


def test_a_failing_logger_does_not_raise() -> None:
    class BadLogger(logging.Logger):
        def log(self, *_: Any, **__: Any) -> None:  # type: ignore[override]
            raise RuntimeError("log")

    inner = Inner()
    inner.mode = "raise"
    health = ExportHealth(inner, BadLogger("bad"), clock=Clock())
    assert health.export(_spans(1)) is SpanExportResult.FAILURE
    assert health.dropped == 1


def test_once_per() -> None:
    clock = Clock()
    once = OncePer(60, clock)
    assert once.ready()
    assert not once.ready()
    clock.now += 59.999
    assert not once.ready()
    clock.now += 0.001
    assert once.ready()
    assert not once.ready()
