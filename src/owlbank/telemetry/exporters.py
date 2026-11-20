"""Our own small OpenTelemetry exporters (call observability).

No ADOT package is a dependency: ADOT pins an older OpenTelemetry than the one Strands locks. We use
only ADOT's wire formats, which a test checked byte for byte:

* :class:`SigV4OtlpSpanExporter` (bridge): the upstream OTLP/HTTP protobuf span exporter, with a
  ``requests`` session that signs every request with SigV4 for the ``xray`` service and adds the
  unified-mode headers ``x-aws-log-group`` and ``x-aws-log-stream``, so spans land in our own traces
  log group. It exports **only** to ``https://xray.<region>.amazonaws.com/v1/traces`` for its own
  Region; any other endpoint is refused at construction.
* :class:`UdpSpanExporter` (front door): the ADOT ``OTLPUdpSpanExporter`` framing to Lambda's X-Ray
  daemon, ``{"format":"json","version":1}\\n`` + ``T1S`` + base64 OTLP protobuf, one datagram per
  export call on a non-blocking socket. It never raises; an error returns ``FAILURE``. The
  address must be loopback unless it comes from ``AWS_XRAY_DAEMON_ADDRESS`` (the daemon address
  Lambda itself sets).
* :class:`EmfStdoutMetricExporter` (bridge): one CloudWatch Embedded Metric Format JSON line per
  data point on stdout, which lands in the bridge's existing log group. Dimensions are bounded
  to ``Service``, ``Environment`` and ``tool_name``; every other point attribute is dropped.

The front door's Lambda layer holds only the SDK and the OTLP protobuf encoder (no ``requests``, no
OTLP/HTTP exporter), so the OTLP/HTTP and ``requests`` imports happen inside
:class:`SigV4OtlpSpanExporter` and this module imports cleanly in either process.
"""

from __future__ import annotations

import base64
import contextlib
import ipaddress
import json
import logging
import math
import os
import re
import socket
import sys
from collections.abc import Mapping, Sequence
from typing import Any, TextIO

from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.metrics import (
    Counter,
    Histogram,
    ObservableCounter,
    ObservableGauge,
    ObservableUpDownCounter,
    UpDownCounter,
    _Gauge,
)
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    ExponentialHistogramDataPoint,
    HistogramDataPoint,
    MetricExporter,
    MetricExportResult,
    MetricsData,
    NumberDataPoint,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from owlbank.obs import log_caught
from owlbank.telemetry.health import LOG_INTERVAL_S, OncePer

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------
# SigV4 OTLP/HTTP span exporter (bridge)
# --------------------------------------------------------------------------------------------

_REGION_RE = re.compile(r"^[a-z]{2}(-[a-z]+)+-\d{1,2}$")
_SIGV4_SERVICE = "xray"


def xray_traces_endpoint(region: str) -> str:
    """The only endpoint the bridge exports spans to: X-Ray's OTLP traces endpoint in ``region``."""
    return f"https://xray.{region}.amazonaws.com/v1/traces"


def _signing_session_class() -> type:
    """A ``requests.Session`` subclass that signs each prepared request (imported lazily)."""
    import requests  # noqa: PLC0415  (not in the front door's layer)
    from botocore.auth import SigV4Auth  # noqa: PLC0415
    from botocore.awsrequest import AWSRequest  # noqa: PLC0415

    class _SigV4Session(requests.Session):
        """Signs every request just before it is sent, with credentials frozen per request.

        Signing in :meth:`send` covers every path the upstream transport takes
        (``Session.request`` prepares the request and calls ``send``) and signs the exact
        method, URL, headers and body that go on the wire. Credentials are re-read from the
        botocore session for each request, so a refreshed task-role credential is picked up.
        """

        def __init__(self, botocore_session: Any, region: str, extra_headers: Mapping[str, str]) -> None:
            super().__init__()
            self._botocore_session = botocore_session
            self._region = region
            self._extra_headers = dict(extra_headers)

        def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
            credentials = self._botocore_session.get_credentials()
            if credentials is None:
                raise RuntimeError("no AWS credentials for SigV4 span export")
            frozen = credentials.get_frozen_credentials()
            for name, value in self._extra_headers.items():
                request.headers[name] = value
            body = request.body
            if isinstance(body, str):
                body = body.encode("utf-8")
            aws_request = AWSRequest(
                method=request.method,
                url=request.url,
                data=body or b"",
                headers=dict(request.headers),
            )
            SigV4Auth(frozen, _SIGV4_SERVICE, self._region).add_auth(aws_request)
            for name, value in aws_request.headers.items():
                request.headers[name] = value
            return super().send(request, **kwargs)

    return _SigV4Session


class SigV4OtlpSpanExporter(SpanExporter):
    """OTLP/HTTP protobuf to X-Ray, signed with SigV4, into our traces log group.

    ``region`` is the deployment's own Region; ``log_group`` is the traces log group and
    ``stream`` its log stream (unified-mode headers ``x-aws-log-group``/``x-aws-log-stream``).
    ``session`` is a botocore session (a default one when omitted). ``endpoint``, when given, must be
    exactly :func:`xray_traces_endpoint` for ``region``; anything else raises :class:`ValueError`
    So a stray ``OTEL_EXPORTER_OTLP_*`` value can never redirect spans.
    """

    def __init__(
        self,
        region: str,
        log_group: str,
        *,
        stream: str = "spans",
        session: Any = None,
        endpoint: str | None = None,
        timeout: float | None = None,
    ) -> None:
        if not isinstance(region, str) or not _REGION_RE.fullmatch(region):
            raise ValueError("SigV4OtlpSpanExporter: region is not an AWS Region name")
        if not log_group or not stream:
            raise ValueError("SigV4OtlpSpanExporter: log_group and stream are required")
        expected = xray_traces_endpoint(region)
        if endpoint is not None and endpoint != expected:
            raise ValueError(
                "SigV4OtlpSpanExporter: endpoint must be X-Ray's OTLP endpoint in the own Region"
            )
        if session is None:
            import botocore.session  # noqa: PLC0415

            session = botocore.session.Session()
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: PLC0415

        self.endpoint = expected
        self.region = region
        headers = {"x-aws-log-group": log_group, "x-aws-log-stream": stream}
        self._session = _signing_session_class()(session, region, headers)
        self._inner = OTLPSpanExporter(
            endpoint=expected, headers=headers, session=self._session, timeout=timeout
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export one batch; any error (including signing) is ``FAILURE``, never an exception."""
        try:
            return self._inner.export(spans)
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


# --------------------------------------------------------------------------------------------
# OTLP over UDP span exporter (front door)
# --------------------------------------------------------------------------------------------

UDP_HEADER = b'{"format":"json","version":1}\n'
UDP_SIGNAL_PREFIX = b"T1S"
DEFAULT_DAEMON_ADDRESS = "127.0.0.1:2000"
_DAEMON_ENV = "AWS_XRAY_DAEMON_ADDRESS"


def _parse_addr(addr: str) -> tuple[str, int]:
    """``host:port`` with an IPv4 literal (or ``localhost``) and a port in 1-65535; else ValueError."""
    host, sep, port_text = addr.strip().rpartition(":")
    if not sep or not port_text.isdigit():
        raise ValueError("UdpSpanExporter: address must be host:port")
    port = int(port_text)
    if not 0 < port < 65536:
        raise ValueError("UdpSpanExporter: port out of range")
    if host == "localhost":
        host = "127.0.0.1"
    try:
        ip = ipaddress.IPv4Address(host)
    except ValueError:
        raise ValueError("UdpSpanExporter: host must be an IPv4 address") from None
    return str(ip), port


def udp_datagram(spans: Sequence[ReadableSpan]) -> bytes:
    """The datagram for one export call: header, ``T1S``, base64 of the OTLP protobuf request."""
    return UDP_HEADER + UDP_SIGNAL_PREFIX + base64.b64encode(encode_spans(spans).SerializeToString())


class UdpSpanExporter(SpanExporter):
    """ADOT's OTLP-over-UDP wire format to Lambda's X-Ray daemon; never raises.

    ``addr`` defaults to ``AWS_XRAY_DAEMON_ADDRESS`` when set, else ``127.0.0.1:2000``. An address
    that is not loopback is refused (:class:`ValueError`) unless it is the value of
    ``AWS_XRAY_DAEMON_ADDRESS``. ``sock`` replaces the socket (tests).
    """

    def __init__(self, addr: str | None = None, *, sock: Any = None) -> None:
        env_addr = os.environ.get(_DAEMON_ENV)
        if addr is None:
            addr = env_addr or DEFAULT_DAEMON_ADDRESS
        host, port = _parse_addr(addr)
        if not ipaddress.IPv4Address(host).is_loopback and addr != env_addr:
            raise ValueError("UdpSpanExporter: address must be loopback or AWS_XRAY_DAEMON_ADDRESS")
        self.addr = (host, port)
        if sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
        self._sock = sock
        self.failures = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            self._sock.sendto(udp_datagram(spans), self.addr)
        except Exception:
            self.failures += 1
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        with contextlib.suppress(Exception):  # shutdown must not raise
            self._sock.close()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# --------------------------------------------------------------------------------------------
# EMF on stdout metric exporter (bridge)
# --------------------------------------------------------------------------------------------

BASE_DIMENSIONS = ("Service", "Environment")
#: ``tool_name`` on the tool metrics; ``model_id`` on Strands' TTFT and latency histograms when
#: the model list has two or more entries (at most three values).
ALLOWED_EXTRA_DIMENSIONS = frozenset({"tool_name", "model_id"})

_UNITS = {
    "ms": "Milliseconds",
    "s": "Seconds",
    "us": "Microseconds",
    "By": "Bytes",
    "Count": "Count",
    "1": "Count",
    "token": "Count",
    "{token}": "Count",
}
_RESERVED_KEYS = frozenset({"_aws", *BASE_DIMENSIONS, *ALLOWED_EXTRA_DIMENSIONS})

_DELTA = AggregationTemporality.DELTA
_DELTA_TEMPORALITY: dict[type, AggregationTemporality] = {
    Counter: _DELTA,
    UpDownCounter: _DELTA,
    Histogram: _DELTA,
    ObservableCounter: _DELTA,
    ObservableUpDownCounter: _DELTA,
    ObservableGauge: _DELTA,
    _Gauge: _DELTA,  # a gauge line only for a period in which it was set
}


def cloudwatch_unit(otel_unit: str | None) -> str:
    """CloudWatch unit for an OpenTelemetry unit string (``None`` when unknown)."""
    return _UNITS.get(otel_unit or "", "None")


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _point_value(point: Any) -> Any:
    """The EMF value of one data point, or ``None`` to skip it.

    Sums and gauges give their number. Histograms give a CloudWatch statistic set
    (``{"Count", "Sum", "Min", "Max"}``): it keeps average, minimum and maximum per period, which
    the anomaly and ceiling alarms use, at one metric per point; an empty histogram is skipped.
    """
    if isinstance(point, NumberDataPoint):
        return point.value if _finite(point.value) else None
    if isinstance(point, HistogramDataPoint | ExponentialHistogramDataPoint):
        if not point.count:
            return None
        stats = {"Count": point.count, "Sum": point.sum, "Min": point.min, "Max": point.max}
        return stats if all(_finite(v) for v in stats.values()) else None
    return None


class EmfStdoutMetricExporter(MetricExporter):
    """One CloudWatch EMF JSON line per metric data point, on stdout.

    Each line: ``_aws.Timestamp`` (ms), ``_aws.CloudWatchMetrics`` with ``namespace``, the dimension
    sets ``["Service","Environment"]`` and, only when the point has ``tool_name`` or ``model_id``
    One more set with each of them present added (``["Service","Environment",
    "tool_name"]``, ``[..., "model_id"]`` or ``[..., "tool_name", "model_id"]``); top-level ``Service``,
    ``Environment``, ``tool_name`` or ``model_id`` when present, and the metric value under the
    metric's name. Every other point attribute is dropped. ``allowed_dimensions`` may only
    name ``tool_name`` and ``model_id``. Temporality is delta, so
    each line is that period's value. ``stream`` defaults to ``sys.stdout`` at write time. Export
    never raises; a failure returns ``FAILURE``.
    """

    def __init__(
        self,
        namespace: str,
        *,
        service: str,
        environment: str,
        allowed_dimensions: Sequence[str] = ("tool_name", "model_id"),
        stream: TextIO | None = None,
    ) -> None:
        extra = tuple(dict.fromkeys(allowed_dimensions))
        if not set(extra) <= ALLOWED_EXTRA_DIMENSIONS:
            raise ValueError("EmfStdoutMetricExporter: dimensions are bounded to tool_name and model_id")
        if not namespace or not service or not environment:
            raise ValueError("EmfStdoutMetricExporter: namespace, service and environment are required")
        super().__init__(preferred_temporality=dict(_DELTA_TEMPORALITY))
        self.namespace = namespace
        self.service = service
        self.environment = environment
        self._extra = extra
        self._stream = stream
        self._once = OncePer(LOG_INTERVAL_S)

    def _failed(self, exc: Exception) -> None:
        """A rate-limited WARNING with the error class only (never a value); never raises."""
        try:
            if self._once.ready():
                log_caught(log, "metric_export_failed", exc, decision="drop_metrics")
        except Exception:  # noqa: S110 - logging must not raise into the export
            pass

    def emf_record(self, name: str, unit: str | None, point: Any) -> dict[str, Any] | None:
        """The EMF object for one data point, or ``None`` when the point has no exportable value."""
        if not name or name in _RESERVED_KEYS:
            return None
        value = _point_value(point)
        if value is None:
            return None
        attributes = point.attributes or {}
        dims: dict[str, str] = {}
        for key in self._extra:
            raw = attributes.get(key)
            if raw is not None and str(raw):
                dims[key] = str(raw)
        dimension_sets = [list(BASE_DIMENSIONS)]
        if dims:
            dimension_sets.append([*BASE_DIMENSIONS, *dims])
        timestamp = getattr(point, "time_unix_nano", 0) // 1_000_000
        return {
            "_aws": {
                "Timestamp": timestamp,
                "CloudWatchMetrics": [
                    {
                        "Namespace": self.namespace,
                        "Dimensions": dimension_sets,
                        "Metrics": [{"Name": name, "Unit": cloudwatch_unit(unit)}],
                    }
                ],
            },
            "Service": self.service,
            "Environment": self.environment,
            **dims,
            name: value,
        }

    def export(
        self, metrics_data: MetricsData, timeout_millis: float = 10_000, **kwargs: Any
    ) -> MetricExportResult:
        ok = True
        lines: list[str] = []
        try:
            for resource_metrics in metrics_data.resource_metrics:
                for scope_metrics in resource_metrics.scope_metrics:
                    for metric in scope_metrics.metrics:
                        for point in getattr(metric.data, "data_points", ()):
                            try:
                                record = self.emf_record(metric.name, metric.unit, point)
                                if record is not None:
                                    lines.append(json.dumps(record, separators=(",", ":"), allow_nan=False))
                            except Exception as exc:  # noqa: BLE001
                                ok = False
                                self._failed(exc)
            if lines:
                stream = self._stream if self._stream is not None else sys.stdout
                stream.write("".join(line + "\n" for line in lines))
                stream.flush()
        except Exception as exc:  # noqa: BLE001 - the SDK ignores the result, so say it here
            self._failed(exc)
            return MetricExportResult.FAILURE
        return MetricExportResult.SUCCESS if ok else MetricExportResult.FAILURE

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        return True

    def shutdown(self, timeout_millis: float = 30_000, **kwargs: Any) -> None:
        return None
