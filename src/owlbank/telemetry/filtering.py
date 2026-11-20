"""The content filter: an exporter wrapper that copies each span with only what may leave.

Ended spans are immutable in the OpenTelemetry SDK, so :class:`ContentFilterExporter`
builds a new ``ReadableSpan`` for each span and hands the copies to the exporter it wraps. It is
the one choke point for Strands', botocore's and our own spans. Per ``traceContent``:

=========  =====================================================================================
``off``    span, event and link attributes: allow-listed keys only (:func:`allowlist.allowed`);
           events keep their names; ``exception`` events keep ``exception.type`` only; the status
           description (an exception's message) is dropped
``redacted`` as ``off``, plus the text keys (:data:`allowlist.TEXT_KEYS`) passed through
           ``redact``; a key whose redaction returns ``None`` or raises is dropped
``full``   unchanged
=========  =====================================================================================

``aws.auth.*`` is dropped in every mode. Before filtering, each span goes through
:func:`~owlbank.telemetry.evalshape.fill_interrupted_tool`.

**Fails closed**: if filtering a span raises, that span is dropped and counted
(:attr:`ContentFilterExporter.dropped`), never passed on unfiltered; the other spans of the batch
still go. Errors from the inner exporter propagate to :class:`~owlbank.telemetry.health.ExportHealth`.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Link
from opentelemetry.trace.status import Status

from owlbank.obs import log_caught
from owlbank.telemetry.allowlist import TEXT_KEYS, allowed, never
from owlbank.telemetry.evalshape import fill_interrupted_tool
from owlbank.telemetry.health import LOG_INTERVAL_S, OncePer
from owlbank.telemetry.settings import CONTENTS, Content

log = logging.getLogger(__name__)

Redact = Callable[[str], "str | None"]
EXCEPTION_EVENT = "exception"
EXCEPTION_KEYS = frozenset({"exception.type"})


class ContentFilterExporter(SpanExporter):
    """Exports filtered copies of spans to ``inner``."""

    def __init__(self, inner: SpanExporter, *, mode: Content, redact: Redact | None = None) -> None:
        if mode not in CONTENTS:
            raise ValueError("mode must be off, redacted or full")
        self._inner = inner
        self._mode = mode
        self._redact = redact
        self._once = OncePer(LOG_INTERVAL_S)
        self._lock = threading.Lock()
        self.dropped = 0  # spans dropped because filtering them raised

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        out: list[ReadableSpan] = []
        for span in spans:
            try:
                out.append(self._copy(fill_interrupted_tool(span)))
            except Exception as exc:  # noqa: BLE001 - fail closed: the span is dropped, not sent raw
                self._failed(exc)
        if not out:  # an empty batch is a success; a batch that all failed filtering is not
            return SpanExportResult.FAILURE if spans else SpanExportResult.SUCCESS
        return self._inner.export(out)

    def _failed(self, exc: Exception) -> None:
        try:
            with self._lock:
                self.dropped += 1
            if self._once.ready():
                log_caught(log, "trace_filter_failed", exc, decision="drop_span")
        except Exception:  # noqa: S110 - counting must not raise into the export
            pass

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)

    # -- copying ------------------------------------------------------------------------------

    def _copy(self, span: ReadableSpan) -> ReadableSpan:
        full = self._mode == "full"
        status = span.status
        if not full and status.description:
            status = Status(status.status_code)
        return ReadableSpan(
            name=span.name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=self._attributes(span.attributes),
            events=tuple(self._event(e) for e in span.events),
            links=tuple(Link(link.context, self._attributes(link.attributes)) for link in span.links),
            kind=span.kind,
            status=status,
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )

    def _event(self, event: Event) -> Event:
        raw: Mapping[str, Any] = event.attributes or {}
        if event.name == EXCEPTION_EVENT and self._mode != "full":
            # Type only; the message and stack never reach the redaction either.
            raw = {k: v for k, v in raw.items() if k in EXCEPTION_KEYS}
        return Event(event.name, self._attributes(raw), timestamp=event.timestamp)

    def _attributes(self, attributes: Mapping[str, Any] | None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in (attributes or {}).items():
            if never(key):
                continue
            if self._mode == "full" or allowed(key):
                out[key] = value
            elif self._mode == "redacted" and key in TEXT_KEYS:
                clean = self._redacted(value)
                if clean is not None:
                    out[key] = clean
        return out

    def _redacted(self, value: Any) -> Any:
        """``value`` through ``redact``, or ``None`` (drop the key) when that fails."""
        if self._redact is None:
            return None
        try:
            if isinstance(value, str):
                return self._redact(value)
            if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
                cleaned = [self._redact(v) for v in value]
                return None if any(c is None for c in cleaned) else tuple(cleaned)
        except Exception:  # noqa: S110 - a failed redaction drops the key
            pass
        return None
