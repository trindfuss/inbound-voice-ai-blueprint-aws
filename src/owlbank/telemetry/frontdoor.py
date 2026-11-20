"""Front-door spans: one per webhook, named by step, plus downstream Lambda calls.

:func:`install_frontdoor` builds the front door's own tracer provider at INIT, only when
``callObservability`` is on (``handler.py``): the resource, the per-call sampler
X-Ray trace and span IDs and
``SimpleSpanProcessor(ExportHealth(ContentFilterExporter(UdpSpanExporter, mode="off")))``. The
filter's mode is always ``off``: front-door spans never carry conversation text, whatever
``traceContent`` says. UDP is non-blocking, so there is no batching thread to freeze between
invocations.

The app calls the other functions unconditionally. Until :func:`install_frontdoor` has run, each is
a no-op after one module-global check, and nothing here imports OpenTelemetry: its imports happen
inside the functions, after that check, so a process with the switch off imports none of it. None of
them ever raises into the webhook: a telemetry error is swallowed, and only
the webhook's own exception passes through :func:`webhook_span`.

* :func:`webhook_span` (``FrontDoor.handle``): a new root span ``voice.<route>`` under a context
  whose baggage holds ``session.id`` and ``call_sid``, with ``owlbank.route`` and the
  Lambda's own X-Ray trace ID as ``aws.xray.lambda_trace_id``.
* :func:`set_step`: renames the webhook span to the step (``gather.<slot>``, ``verify``, ...) and
  sets ``owlbank.step``. The last step of a webhook names it.
* :func:`add_fields`: ``owlbank.<name>`` attributes on the webhook span. ``value`` and
  ``speech`` are dropped here, whatever the caller passes.
* :func:`child_span`: a child of the current span (``lambda.invoke <op>``).
* :func:`lambda_trace_header`: the ``X-Amzn-Trace-Id`` header for the current span, so the called
  function's segment joins the webhook's trace.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from collections.abc import Iterator
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from owlbank import obs

if TYPE_CHECKING:  # pragma: no cover
    from opentelemetry.sdk.trace.export import SpanExporter

    from owlbank.telemetry.settings import TelemetrySettings

log = logging.getLogger(__name__)

#: The front door's tracer once :func:`install_frontdoor` has run; ``None`` means every function is a no-op.
_tracer: Any = None

#: The webhook span of the request in progress (``add_fields`` and ``set_step`` target it even
#: inside a child span).
_WEBHOOK: ContextVar[Any] = ContextVar("owlbank_frontdoor_webhook_span", default=None)

SESSION_KEY = "session.id"
CALL_SID_KEY = "call_sid"
PREFIX = "owlbank."
#: Never span attributes, whatever the caller passes (never a slot value or the caller's speech).
FORBIDDEN_FIELDS = frozenset({"value", "speech"})
_TRACE_ENV = "_X_AMZN_TRACE_ID"
_ROOT_RE = re.compile(r"^1-[0-9a-f]{8}-[0-9a-f]{24}$")
_PRIMITIVES = (str, bool, int, float)


class _NoSpan:
    """What :func:`child_span` yields when telemetry is off or the span couldn't start."""

    def set(self, key: str, value: Any) -> None:
        """Set ``key`` (a full attribute name) to ``value``; a no-op here."""


_NO_SPAN = _NoSpan()


class _ChildSpan(_NoSpan):
    def __init__(self, span: Any) -> None:
        self._span = span

    def set(self, key: str, value: Any) -> None:
        if value is None or not isinstance(value, _PRIMITIVES):
            return
        with contextlib.suppress(Exception):
            self._span.set_attribute(key, value)


def install_frontdoor(settings: TelemetrySettings, *, exporter: SpanExporter | None = None) -> bool:
    """Build the front door's tracer provider; ``True`` when spans will be exported.

    ``exporter`` replaces the UDP exporter (tests); it is still wrapped by the content filter and
    :class:`~owlbank.telemetry.health.ExportHealth`. Never raises: a failure logs
    ``telemetry_install_failed`` and returns ``False``, and the front door runs untraced.
    """
    global _tracer
    try:
        from opentelemetry.baggage.propagation import W3CBaggagePropagator  # noqa: PLC0415
        from opentelemetry.propagate import set_global_textmap  # noqa: PLC0415
        from opentelemetry.propagators.aws import AwsXRayPropagator  # noqa: PLC0415
        from opentelemetry.propagators.composite import CompositePropagator  # noqa: PLC0415
        from opentelemetry.sdk.extension.aws.trace import AwsXRayIdGenerator  # noqa: PLC0415
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: PLC0415

        from owlbank.telemetry.exporters import UdpSpanExporter  # noqa: PLC0415
        from owlbank.telemetry.filtering import ContentFilterExporter  # noqa: PLC0415
        from owlbank.telemetry.health import ExportHealth  # noqa: PLC0415
        from owlbank.telemetry.sampling import make_sampler  # noqa: PLC0415

        inner = exporter if exporter is not None else UdpSpanExporter()
        provider = TracerProvider(
            resource=Resource.create({
                "service.name": settings.service_name,
                "deployment.environment": settings.stage,
                "owlbank.hosting": settings.hosting,
            }),
            sampler=make_sampler(settings.sample_rate),
            id_generator=AwsXRayIdGenerator(),
        )
        # Front-door spans never carry text: the filter is "off" whatever traceContent says.
        provider.add_span_processor(SimpleSpanProcessor(
            ExportHealth(ContentFilterExporter(inner, mode="off"), log)))
        set_global_textmap(CompositePropagator([AwsXRayPropagator(), W3CBaggagePropagator()]))
        _tracer = provider.get_tracer("owlbank.frontdoor")
    except Exception as exc:  # noqa: BLE001 - telemetry must never stop the front door
        obs.log_caught(log, "telemetry_install_failed", exc, decision="untraced")
        return False
    obs.emit(log, logging.INFO, "telemetry_installed", service=settings.service_name,
             sample_rate=settings.sample_rate)
    return True


def installed() -> bool:
    return _tracer is not None


def webhook_span(route: str | None, call_sid: str | None) -> contextlib.AbstractContextManager[None]:
    """The webhook's span (a new root, ``voice.<route>``), current for the request; a no-op when off."""
    if _tracer is None:
        return contextlib.nullcontext()
    return _webhook_span(route or "unknown", call_sid)


@contextlib.contextmanager
def _webhook_span(route: str, call_sid: str | None) -> Iterator[None]:
    started = _start_webhook(route, call_sid)
    try:
        yield
    except BaseException as exc:
        if started is not None:
            _record_exception(started[0], exc)
        raise
    finally:
        if started is not None:
            _end_webhook(*started)


def _start_webhook(route: str, call_sid: str | None) -> tuple[Any, Any, Any] | None:
    try:
        from opentelemetry import baggage, context, trace  # noqa: PLC0415

        ctx = context.Context()  # empty: a new root, never a parent from the caller
        attributes: dict[str, Any] = {"owlbank.route": route}
        if call_sid:
            ctx = baggage.set_baggage(SESSION_KEY, call_sid, ctx)
            ctx = baggage.set_baggage(CALL_SID_KEY, call_sid, ctx)
            attributes[SESSION_KEY] = call_sid
            attributes[CALL_SID_KEY] = call_sid
        root = lambda_trace_id()
        if root is not None:
            attributes["aws.xray.lambda_trace_id"] = root
        span = _tracer.start_span(f"voice.{route}", context=ctx, kind=trace.SpanKind.SERVER,
                                  attributes=attributes)
        token = context.attach(trace.set_span_in_context(span, ctx))
        hook = _WEBHOOK.set(span)
        return span, token, hook
    except Exception:  # noqa: BLE001 - the webhook runs untraced
        return None


def _end_webhook(span: Any, token: Any, hook: Any) -> None:
    with contextlib.suppress(Exception):
        _WEBHOOK.reset(hook)
    with contextlib.suppress(Exception):
        from opentelemetry import context  # noqa: PLC0415

        context.detach(token)
    with contextlib.suppress(Exception):
        span.end()


def _record_exception(span: Any, exc: BaseException) -> None:
    """The exception's class only: its message can hold caller input."""
    with contextlib.suppress(Exception):
        from opentelemetry.trace import Status, StatusCode  # noqa: PLC0415

        span.add_event("exception", {"exception.type": type(exc).__name__})
        span.set_status(Status(StatusCode.ERROR))


def lambda_trace_id() -> str | None:
    """The Lambda invocation's own X-Ray trace ID (``Root`` of ``_X_AMZN_TRACE_ID``), if valid."""
    for part in os.environ.get(_TRACE_ENV, "").split(";"):
        key, _, value = part.strip().partition("=")
        if key == "Root" and _ROOT_RE.match(value):
            return value
    return None


def set_step(name: str) -> None:
    """Name the webhook span after the step (``owlbank.step``); a no-op when off or outside one."""
    if _tracer is None:
        return
    try:
        span = _WEBHOOK.get()
        if span is None:
            return
        span.update_name(name)
        span.set_attribute("owlbank.step", name)
    except Exception:  # noqa: BLE001, S110 - never into the webhook
        pass


def add_fields(**fields: Any) -> None:
    """``owlbank.<name>`` attributes on the webhook span. ``None`` and non-scalar values are skipped;
    ``value`` and ``speech`` are always dropped. A no-op when off or outside a webhook."""
    if _tracer is None:
        return
    try:
        span = _WEBHOOK.get()
        if span is None:
            return
        for name, value in fields.items():
            if name in FORBIDDEN_FIELDS or value is None or not isinstance(value, _PRIMITIVES):
                continue
            span.set_attribute(PREFIX + name, value)
    except Exception:  # noqa: BLE001, S110 - never into the webhook
        pass


def child_span(name: str, **attributes: Any) -> contextlib.AbstractContextManager[_NoSpan]:
    """A child span of the current span (attributes given by full name; ``None`` skipped). Yields
    an object whose ``set(key, value)`` adds attributes later (an outcome). A no-op when off."""
    if _tracer is None:
        return contextlib.nullcontext(_NO_SPAN)
    return _child_span(name, attributes)


@contextlib.contextmanager
def _child_span(name: str, attributes: dict[str, Any]) -> Iterator[_NoSpan]:
    started = _start_child(name, attributes)
    try:
        yield _ChildSpan(started[0]) if started is not None else _NO_SPAN
    except BaseException as exc:
        if started is not None:
            _record_exception(started[0], exc)
        raise
    finally:
        if started is not None:
            span, token = started
            with contextlib.suppress(Exception):
                from opentelemetry import context  # noqa: PLC0415

                context.detach(token)
            with contextlib.suppress(Exception):
                span.end()


def _start_child(name: str, attributes: dict[str, Any]) -> tuple[Any, Any] | None:
    try:
        from opentelemetry import baggage, context, trace  # noqa: PLC0415

        attrs = {k: v for k, v in attributes.items() if v is not None and isinstance(v, _PRIMITIVES)}
        for key in (SESSION_KEY, CALL_SID_KEY):  # every span carries the call
            value = baggage.get_baggage(key)
            if isinstance(value, str) and value:
                attrs[key] = value
        span = _tracer.start_span(name, kind=trace.SpanKind.CLIENT, attributes=attrs)
        token = context.attach(trace.set_span_in_context(span))
        return span, token
    except Exception:  # noqa: BLE001 - the call runs untraced
        return None


def lambda_trace_header(existing: str | None = None) -> str | None:
    """``Root=1-<8 hex>-<24 hex>;Parent=<16 hex>;Sampled=1[;Lineage=...]`` for the current span.

    ``None`` when off, with no valid current span, or when the span is not sampled: the caller
    then leaves botocore's own header alone, so Lambda's active tracing in the called function is
    not switched off. ``existing`` is the header already on the request; its ``Lineage=``
    field (Lambda's recursive-loop detection) is kept, once."""
    if _tracer is None:
        return None
    try:
        from opentelemetry import trace  # noqa: PLC0415

        sc = trace.get_current_span().get_span_context()
        if not sc.is_valid or not sc.trace_flags.sampled:
            return None
        trace_hex = format(sc.trace_id, "032x")
        header = f"Root=1-{trace_hex[:8]}-{trace_hex[8:]};Parent={format(sc.span_id, '016x')};Sampled=1"
        lineage = next((p.strip() for p in (existing or "").split(";") if p.strip().startswith("Lineage=")), None)
        return f"{header};{lineage}" if lineage else header
    except Exception:  # noqa: BLE001 - no header rather than a failed invoke
        return None
