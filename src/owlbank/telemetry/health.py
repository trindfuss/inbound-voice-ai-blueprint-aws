"""Export failures: counted, logged at most once a minute, never raised.

:class:`ExportHealth` is the outermost span exporter in both processes
(``BatchSpanProcessor(ExportHealth(ContentFilterExporter(...)))``). When the exporter
it wraps returns ``FAILURE`` or raises (throttled, unreachable, 4xx), the spans of that batch are
counted as dropped and ``trace_export_failed`` is logged with the error class and the count
dropped since the last such line, at most once per :data:`LOG_INTERVAL_S`. Nothing propagates
into the batch thread, the turn or the webhook.

:class:`OncePer` is the rate limit, shared with the other components that log their own caught
errors.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from owlbank.obs import emit

LOG_INTERVAL_S = 60.0
FAILURE_RESULT = "ExportFailure"  # error_class when the inner exporter returns FAILURE without raising


class OncePer:
    """``ready()`` is true at most once per ``interval_s`` (the first call is always ready)."""

    def __init__(self, interval_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._interval = interval_s
        self._clock = clock
        self._last: float | None = None
        self._lock = threading.Lock()

    def ready(self) -> bool:
        with self._lock:
            now = self._clock()
            if self._last is not None and now - self._last < self._interval:
                return False
            self._last = now
            return True


class ExportHealth(SpanExporter):
    """Wraps a span exporter: failures are counted and logged once a minute, never raised."""

    def __init__(self, inner: SpanExporter, log: logging.Logger, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._inner = inner
        self._log = log
        self._once = OncePer(LOG_INTERVAL_S, clock)
        self._lock = threading.Lock()
        self._pending = 0  # dropped since the last trace_export_failed line
        self.dropped = 0  # dropped since start
        self.failures = 0  # failed export calls since start

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            result = self._inner.export(spans)
        except Exception as exc:  # noqa: BLE001 - an exporter error must never reach the caller
            self._failed(len(spans), type(exc).__name__)
            return SpanExportResult.FAILURE
        if result is not SpanExportResult.SUCCESS:
            self._failed(len(spans), FAILURE_RESULT)
        return result

    def _failed(self, count: int, error_class: str) -> None:
        try:
            with self._lock:
                self.failures += 1
                self.dropped += count
                self._pending += count
                if not self._once.ready():
                    return
                dropped, self._pending = self._pending, 0
            emit(self._log, logging.WARNING, "trace_export_failed", error_class=error_class, dropped=dropped)
        except Exception:  # noqa: BLE001, S110 - logging must not turn a failed export into a raise
            pass

    def shutdown(self) -> None:
        with contextlib.suppress(Exception):  # shutdown is best effort
            self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            return self._inner.force_flush(timeout_millis)
        except Exception:  # noqa: BLE001 - a flush failure is reported as False, not raised
            return False
