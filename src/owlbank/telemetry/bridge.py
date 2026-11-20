"""The bridge's tracer and meter providers, built in code (``bridge.py``).

:func:`install_bridge` is called once by ``tac_bridge.app.build_app_from_env``, after logging is set
up and before the Strands client starts (so Strands' tracer and meter find our providers). It sets:

* the **resource**: ``service.name`` (``owlbank-<stage>-bridge``), ``deployment.environment``,
  ``owlbank.hosting``, ``aws.log.group.names`` (the traces log group), ``aws.service.type`` =
  ``gen_ai_agent`` and ``aws.local.service`` = the service name (what the GenAI Observability
  console lists agents by);
* a **tracer provider** with the per-call sampler, the X-Ray ID generator and, in order:
  a baggage processor copying ``session.id`` and ``call_sid`` onto every span, the
  evaluation-shape processor, and a batch processor (queue 2048) over
  ``ExportHealth(ContentFilterExporter(SigV4OtlpSpanExporter))``;
* the global **propagators**: baggage, X-Ray, W3C trace context;
* a **meter provider** whose only reader writes EMF lines on stdout every 60 s, with a
  view that keeps ``tool_name`` and ``model_id`` and drops every other metric attribute
  (``model_id`` is on Strands' TTFT and latency histograms only, set by the Strands client when the
  model list has two or more entries: at most three values);
* **botocore instrumentation** with ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``
  forced to ``false``. It gets a no-op meter provider, so the EMF lines stay Strands' own metrics
  And no logger provider is ever set, so nothing goes out as OTLP logs.

No other instrumentation is installed. It **refuses** (``telemetry_refused``, nothing
installed) when the traces log group or the Region is missing, or when any
``OTEL_EXPORTER_OTLP_*ENDPOINT`` variable is set: Strands reads those and would add its own exporter
to another endpoint. It **never raises**: any error is logged and it returns ``None``,
and the bridge runs untraced.

``redacted`` content: the text keys go through :class:`TranscriptRedactor`, the transcripts'
redaction (``owlbank.pii.PiiGuard``: Comprehend and our masking) after the exact-value scrub of the
span's own call's values (``owlbank.segments``), kept for :data:`RETAIN_TTL_S` past the call's
forget (:class:`CallScopedContentFilter`, :meth:`TranscriptRedactor.retain`). The guard is built after
the Strands client starts, so it is attached later; until then every text key is dropped.

This module imports the OpenTelemetry SDK; the bridge imports it only when the switch is on.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import baggage, metrics, propagate, trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.context import Context
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.propagators.aws import AwsXRayPropagator
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.propagators.textmap import TextMapPropagator
from opentelemetry.sdk.extension.aws.trace import AwsXRayIdGenerator
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from owlbank.obs import emit, log_caught
from owlbank.telemetry.evalshape import EvalShapeProcessor
from owlbank.telemetry.exporters import EmfStdoutMetricExporter, SigV4OtlpSpanExporter
from owlbank.telemetry.filtering import ContentFilterExporter, Redact
from owlbank.telemetry.health import LOG_INTERVAL_S, ExportHealth, OncePer
from owlbank.telemetry.sampling import make_sampler
from owlbank.telemetry.settings import TelemetrySettings

log = logging.getLogger(__name__)

SESSION_ID_KEY = "session.id"
CALL_SID_KEY = "call_sid"
BAGGAGE_KEYS = (SESSION_ID_KEY, CALL_SID_KEY)
METRICS_NAMESPACE = "bedrock-agentcore"
METRICS_INTERVAL_MS = 60_000
#: The metric attributes the view keeps: every other one is dropped.
METRIC_ATTRIBUTE_KEYS = ("tool_name", "model_id")
MAX_QUEUE_SIZE = 2048
SHUTDOWN_TIMEOUT_S = 3.0
CAPTURE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
#: OTEL_EXPORTER_OTLP_ENDPOINT and every per-signal OTEL_EXPORTER_OTLP_<SIGNAL>_ENDPOINT.
_FOREIGN_ENDPOINT_RE = re.compile(r"^OTEL_EXPORTER_OTLP_(?:[A-Z0-9]+_)*ENDPOINT$")


# --------------------------------------------------------------------------------------------
# Processors and the redaction adapter
# --------------------------------------------------------------------------------------------


class BaggageSpanProcessor(SpanProcessor):
    """Copies the listed baggage entries onto each span as it starts.

    The same job as ``opentelemetry-processor-baggage``'s processor, for a fixed set of keys,
    without the extra dependency. ``session.id`` is the CallSid, so when only
    ``call_sid`` is in baggage it fills ``session.id`` too. Never raises.
    """

    def __init__(self, keys: Iterable[str] = BAGGAGE_KEYS) -> None:
        self._keys = tuple(keys)

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        try:
            values = {key: baggage.get_baggage(key, parent_context) for key in self._keys}
            if SESSION_ID_KEY in self._keys and not values.get(SESSION_ID_KEY):
                values[SESSION_ID_KEY] = values.get(CALL_SID_KEY)
            for key, value in values.items():
                if isinstance(value, str) and value:
                    span.set_attribute(key, value)
        except Exception:  # noqa: BLE001, S110 - a failed enrichment leaves the span as it was
            pass

    def on_end(self, span: ReadableSpan) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class TranscriptRedactor:
    """The ``redacted`` content mode's redaction: the transcripts', ``None`` on any failure.

    Calls :meth:`attach` with the bridge's ``PiiGuard`` once it exists; until then every text is
    ``None`` (the key is dropped). Each text first goes through the exact-value scrub, then
    through the guard. A guard that can't check the text (Comprehend unavailable) gives ``None``.

    **Whose values.** The scrub of a span's text uses that span's call's values only: the
    ones registered in ``owlbank.segments`` now plus the snapshot :meth:`retain` took when the bridge
    forgot them. So the export's timing doesn't matter: a span exported after its session ended (a
    turn still running at hang-up, a slow or failed flush) is scrubbed with the same values as one
    exported before. The span's call is set by :meth:`scope` (:class:`CallScopedContentFilter`); a
    span with no call, or a text outside any scope, is scrubbed with every call's values (fail
    closed). A snapshot always lives :data:`RETAIN_TTL_S` after its call's latest :meth:`retain`
    (no early release: it fails closed), at most :data:`MAX_RETAINED_CALLS` of them, oldest dropped
    first. Losing one (that cap, or a failed :meth:`retain`) logs a WARNING at most once a minute;
    expiry is normal and not logged. The values are memory only and never logged.
    """

    _WITHHELD = "\x00owlbank-telemetry-withheld\x00"

    def __init__(self, guard: Any = None, *, language: str = "en-US",
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._guard = guard
        self._language = language
        self._clock = clock
        self._lock = threading.Lock()
        #: call -> (normalized values, expiry): the snapshot taken at :meth:`retain`, in the order
        #: of each call's latest retain (so oldest first is also soonest to expire).
        self._retained: dict[str, tuple[frozenset[str], float]] = {}
        self._evicted_once = OncePer(LOG_INTERVAL_S, clock)
        #: snapshots evicted at the cap since the last ``trace_snapshot_evicted`` line, and ever.
        self._evicted_pending = 0
        self.evicted = 0
        self._failed_once = OncePer(LOG_INTERVAL_S, clock)
        self._scope = threading.local()

    def attach(self, guard: Any) -> None:
        self._guard = guard

    def __call__(self, text: str) -> str | None:
        guard = self._guard
        if guard is None or not isinstance(text, str):
            return None
        try:
            text = self._scrub(text)
            out = guard.text("trace", text, language=self._language, withheld=self._WITHHELD,
                             on_failure="withhold")
        except Exception:  # noqa: BLE001 - a failed redaction drops the key
            return None
        if not isinstance(out, str) or out == self._WITHHELD:
            return None
        return out

    # -- the span's call -------------------------------------------------------------------------

    @contextlib.contextmanager
    def scope(self, call_sid: str | None) -> Iterator[None]:
        """Texts redacted inside are a span of ``call_sid`` (``None``: unknown, fail closed)."""
        previous = getattr(self._scope, "call", _UNSCOPED)
        self._scope.call = call_sid
        try:
            yield
        finally:
            self._scope.call = previous

    def _scrub(self, text: str) -> str:
        segments = sys.modules.get("owlbank.segments")
        if segments is None:  # never imported: nothing was registered or retained
            return text
        scope = getattr(self._scope, "call", _UNSCOPED)
        call = scope if isinstance(scope, str) and scope else None
        # The registry first, then the snapshot: the bridge retains before it forgets, so a value
        # gone from the registry by the first read is in the snapshot by the second.
        values: set[str] = set()
        if call is not None:
            values |= segments.sensitive_values(call)
        else:
            for other in segments.registered_calls():  # a snapshot: calls register and forget concurrently
                values |= segments.sensitive_values(other)
        with self._lock:
            self._expire()
            if call is not None:
                held = self._retained.get(call)
                if held is not None:
                    values |= held[0]
            else:
                for held_values, _expiry in self._retained.values():
                    values |= held_values
        return segments.redact_values(values, text) if values else text

    # -- the snapshot --------------------------------------------------------------------------

    def retain(self, call_sid: str) -> None:
        """Keep a snapshot of ``call_sid``'s registered values (merged with any it holds) until
        :data:`RETAIN_TTL_S` from now, for its spans exported later; the bridge calls it just before
        it forgets them. Never raises."""
        evicted = 0
        try:
            segments = sys.modules.get("owlbank.segments")
            values = segments.sensitive_values(call_sid) if segments is not None else frozenset()
            with self._lock:
                self._expire()
                held = self._retained.pop(call_sid, None)
                if held is not None:
                    values = values | held[0]
                if not values:
                    return  # nothing to scrub
                self._retained[call_sid] = (frozenset(values), self._clock() + RETAIN_TTL_S)
                while len(self._retained) > MAX_RETAINED_CALLS:  # oldest first (latest retain order)
                    self._retained.pop(next(iter(self._retained)))
                    self._evicted_pending += 1
                    self.evicted += 1
                    evicted = 1
        except Exception as exc:  # noqa: BLE001 - its spans fall back to the registered values
            with contextlib.suppress(Exception):
                if self._failed_once.ready():
                    log_caught(log, "trace_retain_failed", exc, decision="continue")
        if evicted:
            with contextlib.suppress(Exception):
                if self._evicted_once.ready():
                    with self._lock:
                        evicted, self._evicted_pending = self._evicted_pending, 0
                    emit(log, logging.WARNING, "trace_snapshot_evicted", evicted=evicted, total=self.evicted,
                         max_retained_calls=MAX_RETAINED_CALLS, decision="drop_oldest")

    def retained_calls(self) -> tuple[str, ...]:
        """The calls with a snapshot held (tests and diagnostics; never the values)."""
        with self._lock:
            self._expire()
            return tuple(self._retained)

    def _expire(self) -> None:
        """Drop snapshots past their expiry (the lock is held). Insertion order is expiry order."""
        now = self._clock()
        while self._retained:
            call, (_values, expiry) = next(iter(self._retained.items()))
            if expiry > now:
                return
            del self._retained[call]


#: A redaction outside any :meth:`TranscriptRedactor.scope`.
_UNSCOPED = object()
#: How long :meth:`TranscriptRedactor.retain` keeps a call's snapshot after its latest retain: spans
#: exported later than this (far beyond the batch processor's schedule) lose the forgotten values.
RETAIN_TTL_S = 600.0
#: The most calls with a snapshot held at once (oldest dropped first); a bridge task holds few calls.
MAX_RETAINED_CALLS = 1024


def span_call(span: Any) -> str | None:
    """The call a span belongs to: its ``session.id`` (the CallSid), else ``call_sid``."""
    attributes = getattr(span, "attributes", None) or {}
    for key in (SESSION_ID_KEY, CALL_SID_KEY):
        value = attributes.get(key)
        if isinstance(value, str) and value:
            return value
    return None


class CallScopedContentFilter(ContentFilterExporter):
    """The content filter with each span's text redacted in its call's scope
    (:meth:`TranscriptRedactor.scope`). Used for ``redacted`` content only."""

    def __init__(self, inner: SpanExporter, *, redactor: TranscriptRedactor) -> None:
        super().__init__(inner, mode="redacted", redact=redactor)
        self._redactor = redactor

    def _copy(self, span: ReadableSpan) -> ReadableSpan:
        with self._redactor.scope(span_call(span)):
            return super()._copy(span)


# --------------------------------------------------------------------------------------------
# The handle
# --------------------------------------------------------------------------------------------


@dataclass
class TelemetryHandle:
    """What :func:`install_bridge` installed; :meth:`shutdown` flushes and stops it."""

    tracer_provider: TracerProvider
    meter_provider: MeterProvider
    instrumentor: Any = None
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _started: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def tracer(self) -> trace.Tracer:
        return self.tracer_provider.get_tracer("owlbank.bridge")

    def shutdown(self, timeout_s: float = SHUTDOWN_TIMEOUT_S) -> bool:
        """Flush spans and metrics and stop, within ``timeout_s``. Never raises.

        Returns whether everything finished in time; what hasn't by then is abandoned (the work
        runs on a daemon thread, so it can't hold the process open). Idempotent.
        """
        with self._lock:
            if not self._started:
                self._started = True
                threading.Thread(target=self._shutdown, name="telemetry-shutdown", daemon=True).start()
        return self._done.wait(timeout_s)

    def _shutdown(self) -> None:
        timeout_ms = int(SHUTDOWN_TIMEOUT_S * 1000)
        steps: list[tuple[str, Callable[[], Any]]] = [
            ("uninstrument", lambda: self.instrumentor and self.instrumentor.uninstrument()),
            ("flush_spans", lambda: self.tracer_provider.force_flush(timeout_ms)),
            ("stop_spans", self.tracer_provider.shutdown),
            ("stop_metrics", lambda: self.meter_provider.shutdown(timeout_millis=timeout_ms)),
        ]
        try:
            for step, run in steps:
                try:
                    run()
                except Exception as exc:  # noqa: BLE001 - shutdown is best effort
                    log_caught(log, "telemetry_shutdown_failed", exc, decision="continue", step=step)
        finally:
            self._done.set()


# --------------------------------------------------------------------------------------------
# install_bridge
# --------------------------------------------------------------------------------------------


def foreign_endpoints(environ: Mapping[str, str]) -> list[str]:
    """Names (never values) of the ``OTEL_EXPORTER_OTLP_*ENDPOINT`` variables set in ``environ``."""
    return sorted(name for name in environ if _FOREIGN_ENDPOINT_RE.match(name))


def bridge_resource(settings: TelemetrySettings) -> Resource:
    """The bridge's resource attributes."""
    attributes: dict[str, Any] = {
        "service.name": settings.service_name,
        "deployment.environment": settings.stage,
        "owlbank.hosting": settings.hosting,
        "aws.service.type": "gen_ai_agent",
        "aws.local.service": settings.service_name,
    }
    if settings.traces_log_group:
        attributes["aws.log.group.names"] = (settings.traces_log_group,)
    return Resource.create(attributes)


def propagator() -> TextMapPropagator:
    """Baggage, X-Ray (``X-Amzn-Trace-Id``) and W3C trace context."""
    return CompositePropagator([W3CBaggagePropagator(), AwsXRayPropagator(), TraceContextTextMapPropagator()])


def _refusal(settings: TelemetrySettings, environ: Iterable[Mapping[str, str]]) -> dict[str, Any] | None:
    if not settings.traces_log_group:
        return {"reason": "no_traces_log_group"}
    if not settings.region:
        return {"reason": "no_region"}
    names = sorted({name for env in environ for name in foreign_endpoints(env)})
    if names:
        return {"reason": "foreign_otlp_endpoint", "variables": names}
    return None


class _ParamSafeBotocore:
    """Botocore instrumentation whose extensions see a copy of each call's parameters.

    Some service extensions of opentelemetry-instrumentation-botocore rewrite the request to carry
    trace context: Lambda ``Invoke`` re-serialises ``Payload`` with an added ``headers`` key (and
    the baggage, so the call SID), SNS ``Publish``/``PublishBatch`` add ``MessageAttributes``.
    ``bank_tools`` rejects unknown top-level keys, so with tracing on every tool call failed. No version
    offers a switch for this, so the library's ``_patched_api_call`` runs on a
    deep copy of the parameters (spans, attributes and hooks as before) while botocore gets the
    caller's own, untouched. This covers every service, including ones a later version starts to
    rewrite. Trace context still reaches AWS in the ``X-Amzn-Trace-Id`` HTTP header, which is not a
    request parameter. If the parameters can't be copied, the call goes out untraced, unchanged.

    The override is an attribute on the instrumentor (a per-class singleton), set before
    ``instrument()`` wraps it and removed by :meth:`uninstrument`.
    """

    def __init__(self, instrumentor: Any) -> None:
        self._inner = instrumentor

    def instrument(self, **kwargs: Any) -> None:
        inner = self._inner
        library_call = type(inner)._patched_api_call  # noqa: SLF001 - the library's own wrapper

        def api_call(original: Callable[..., Any], instance: Any, args: tuple[Any, ...],
                     kwargs: dict[str, Any]) -> Any:
            if len(args) != 2 or not isinstance(args[1], dict):
                return library_call(inner, original, instance, args, kwargs)
            try:
                shadow = (args[0], copy.deepcopy(args[1]))
            except Exception:  # noqa: BLE001 - e.g. a file object; send it as given, untraced
                return original(*args, **kwargs)
            return library_call(inner, lambda *_a, **_k: original(*args, **kwargs), instance, shadow, kwargs)

        inner._patched_api_call = api_call  # noqa: SLF001 - wrapped by instrument() below
        inner.instrument(**kwargs)

    def uninstrument(self) -> None:
        try:
            self._inner.uninstrument()
        finally:
            vars(self._inner).pop("_patched_api_call", None)


def install_bridge(
    settings: TelemetrySettings,
    *,
    redact: Redact | None = None,
    environ: Mapping[str, str] | None = None,
    set_tracer_provider: Callable[[Any], None] | None = None,
    set_meter_provider: Callable[[Any], None] | None = None,
    set_propagator: Callable[[TextMapPropagator], None] | None = None,
    instrument_botocore: bool = True,
) -> TelemetryHandle | None:
    """Install the bridge's telemetry (module docs), or return ``None`` (refused or failed).

    ``redact`` is the ``redacted`` mode's redaction (normally a :class:`TranscriptRedactor`); with
    ``off`` and ``full`` it is ignored. ``environ`` is checked for foreign OTLP endpoints in addition
    to ``os.environ`` (which Strands reads). The ``set_*`` callables default to OpenTelemetry's
    global setters, looked up at call time; tests pass their own, since a global provider can be set
    only once per process.
    """
    handle: TelemetryHandle | None = None
    try:
        envs = [os.environ] if environ is None else [os.environ, environ]
        refused = _refusal(settings, envs)
        if refused is not None:
            emit(log, logging.WARNING, "telemetry_refused", **refused)
            return None
        region, log_group = settings.region or "", settings.traces_log_group or ""  # both checked above

        resource = bridge_resource(settings)
        # Build everything before touching a global, so a failure installs nothing.
        sink = SigV4OtlpSpanExporter(region, log_group)
        filtered: SpanExporter
        if settings.content == "redacted" and isinstance(redact, TranscriptRedactor):
            # Each span's text is scrubbed with its own call's values, whenever it's exported.
            filtered = CallScopedContentFilter(sink, redactor=redact)
        else:
            filtered = ContentFilterExporter(sink, mode=settings.content,
                                             redact=redact if settings.content == "redacted" else None)
        exporter = ExportHealth(filtered, log)
        tracer_provider = TracerProvider(
            resource=resource, sampler=make_sampler(settings.sample_rate), id_generator=AwsXRayIdGenerator()
        )
        tracer_provider.add_span_processor(BaggageSpanProcessor(BAGGAGE_KEYS))
        tracer_provider.add_span_processor(EvalShapeProcessor())
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter, max_queue_size=MAX_QUEUE_SIZE))

        reader = PeriodicExportingMetricReader(
            EmfStdoutMetricExporter(METRICS_NAMESPACE, service=settings.service_name, environment=settings.stage),
            export_interval_millis=METRICS_INTERVAL_MS,
        )
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[reader],
            views=[View(instrument_name="*", attribute_keys=set(METRIC_ATTRIBUTE_KEYS))],
        )
        handle = TelemetryHandle(tracer_provider=tracer_provider, meter_provider=meter_provider)

        if instrument_botocore:
            from opentelemetry.instrumentation.botocore import BotocoreInstrumentor  # noqa: PLC0415

            os.environ[CAPTURE_CONTENT_ENV] = "false"  # Never prompts or answers on botocore spans
            instrumentor = _ParamSafeBotocore(BotocoreInstrumentor())
            handle.instrumentor = instrumentor  # set first, so a failed instrument() is still undone
            instrumentor.instrument(tracer_provider=tracer_provider, meter_provider=NoOpMeterProvider())

        (set_propagator or propagate.set_global_textmap)(propagator())
        (set_tracer_provider or trace.set_tracer_provider)(tracer_provider)
        (set_meter_provider or metrics.set_meter_provider)(meter_provider)
        return handle
    except Exception as exc:  # noqa: BLE001 - telemetry never stops the bridge
        with contextlib.suppress(Exception):
            if handle is not None and handle.instrumentor is not None:
                handle.instrumentor.uninstrument()
        with contextlib.suppress(Exception):
            log_caught(log, "telemetry_install_failed", exc, decision="run_untraced")
        return None
