"""Call observability exporters.

* SigV4: the request is signed for ``xray`` in the exporter's Region (checked against an independent
  header-signature computation from the SigV4 spec), carries ``x-aws-log-group``/``x-aws-log-stream``,
  re-reads credentials per request, and never raises; a foreign endpoint is refused. No network: a
  ``requests`` adapter captures the request.
* UDP: the datagram is byte-exact to ADOT's framing (a real socket on 127.0.0.1:0), a
  ``sendto`` error returns ``FAILURE``, and only loopback or ``AWS_XRAY_DAEMON_ADDRESS`` is accepted.
* EMF: the line shape matches.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import logging
import socket
from urllib.parse import urlsplit

import pytest
import requests
from botocore.credentials import Credentials
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricExportResult,
    NumberDataPoint,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from owlbank.telemetry import exporters as ex
from tests.property.test_telemetry_emf_props import metrics_data_for

REGION = "us-west-2"
LOG_GROUP = "/owlbank/traces/bridge"


def _spans(names=("bridge.turn",)):
    mem = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "owlbank-test"}))
    provider.add_span_processor(SimpleSpanProcessor(mem))
    tracer = provider.get_tracer("test")
    for name in names:
        with tracer.start_as_current_span(name, attributes={"call_sid": "CA" + "0" * 32}):
            pass
    return mem.get_finished_spans()


# ---------------------------------------------------------------- SigV4


class FakeBotocoreSession:
    """Hands out the current credentials; counts reads (refresh per request)."""

    def __init__(self, creds):
        self.creds = creds
        self.reads = 0

    def get_credentials(self):
        self.reads += 1
        return self.creds


class CaptureAdapter(requests.adapters.BaseAdapter):
    def __init__(self, status=200):
        super().__init__()
        self.requests = []
        self.status = status

    def send(self, request, **kwargs):
        self.requests.append(request)
        resp = requests.Response()
        resp.status_code = self.status
        resp.reason = "OK" if self.status == 200 else "Bad Request"
        resp._content = b""
        resp.request = request
        resp.url = request.url
        return resp

    def close(self):
        pass


def _exporter(creds=None, status=200):
    session = FakeBotocoreSession(creds or Credentials("AKIDEXAMPLE", "secretkey", "sessiontoken"))
    exp = ex.SigV4OtlpSpanExporter(REGION, LOG_GROUP, session=session, timeout=1)
    adapter = CaptureAdapter(status)
    exp._session.mount("https://", adapter)
    return exp, session, adapter


def _hmac(key, msg):
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _verify_header_signature(request, secret, region, service):
    """Recompute the Authorization signature from the SigV4 spec (header form) and compare."""
    auth = request.headers["Authorization"]
    algo, rest = auth.split(" ", 1)
    assert algo == "AWS4-HMAC-SHA256"
    parts = dict(p.strip().split("=", 1) for p in rest.split(","))
    signed = parts["SignedHeaders"].split(";")
    amz_date = request.headers["X-Amz-Date"]
    url = urlsplit(request.url)
    headers = {k.lower(): " ".join(str(v).split()) for k, v in request.headers.items()}
    headers.setdefault("host", url.netloc)
    canonical = "\n".join(
        [
            request.method,
            url.path,
            url.query,
            "".join(f"{h}:{headers[h]}\n" for h in signed),
            ";".join(signed),
            hashlib.sha256(request.body or b"").hexdigest(),
        ]
    )
    scope = f"{amz_date[:8]}/{region}/{service}/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    key = _hmac(
        _hmac(_hmac(_hmac(("AWS4" + secret).encode(), amz_date[:8]), region), service), "aws4_request"
    )
    assert parts["Credential"].endswith("/" + scope)
    assert hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest() == parts["Signature"]
    return parts, signed


def test_sigv4_request_signed_for_xray_with_log_group_headers():
    exp, _, adapter = _exporter()
    spans = _spans()
    assert exp.export(spans) == SpanExportResult.SUCCESS
    (req,) = adapter.requests
    assert req.method == "POST"
    assert req.url == f"https://xray.{REGION}.amazonaws.com/v1/traces"
    assert req.headers["x-aws-log-group"] == LOG_GROUP
    assert req.headers["x-aws-log-stream"] == "spans"
    assert req.headers["Content-Type"] == "application/x-protobuf"
    assert req.headers["X-Amz-Security-Token"] == "sessiontoken"
    parts, signed = _verify_header_signature(req, "secretkey", REGION, "xray")
    assert parts["Credential"].startswith("AKIDEXAMPLE/")
    assert {"host", "x-amz-date", "x-aws-log-group", "x-aws-log-stream", "content-type"} <= set(signed)
    # The body is the OTLP protobuf of the spans.
    decoded = ExportTraceServiceRequest.FromString(req.body)
    assert decoded.resource_spans[0].scope_spans[0].spans[0].name == "bridge.turn"


def test_sigv4_custom_stream_name():
    session = FakeBotocoreSession(Credentials("AKID", "s"))
    exp = ex.SigV4OtlpSpanExporter(REGION, LOG_GROUP, stream="other", session=session, timeout=1)
    adapter = CaptureAdapter()
    exp._session.mount("https://", adapter)
    exp.export(_spans())
    assert adapter.requests[0].headers["x-aws-log-stream"] == "other"
    assert "X-Amz-Security-Token" not in adapter.requests[0].headers


def test_sigv4_credentials_reread_per_request():
    exp, session, adapter = _exporter(Credentials("AKIDONE", "s1", "t1"))
    exp.export(_spans())
    session.creds = Credentials("AKIDTWO", "s2", "t2")
    exp.export(_spans())
    assert session.reads == 2
    assert "Credential=AKIDONE/" in adapter.requests[0].headers["Authorization"]
    assert "Credential=AKIDTWO/" in adapter.requests[1].headers["Authorization"]
    _verify_header_signature(adapter.requests[1], "s2", REGION, "xray")


def test_sigv4_http_error_is_failure_not_raise():
    exp, _, _ = _exporter(status=400)
    assert exp.export(_spans()) == SpanExportResult.FAILURE


def test_sigv4_missing_credentials_is_failure_not_raise():
    session = FakeBotocoreSession(None)
    exp = ex.SigV4OtlpSpanExporter(REGION, LOG_GROUP, session=session, timeout=1)
    adapter = CaptureAdapter()
    exp._session.mount("https://", adapter)
    assert exp.export(_spans()) == SpanExportResult.FAILURE
    assert adapter.requests == []


def test_sigv4_default_endpoint_is_own_region():
    exp = ex.SigV4OtlpSpanExporter("eu-central-1", LOG_GROUP, session=FakeBotocoreSession(None))
    assert exp.endpoint == "https://xray.eu-central-1.amazonaws.com/v1/traces"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://xray.us-east-1.amazonaws.com/v1/traces",  # another Region
        "http://xray.us-west-2.amazonaws.com/v1/traces",
        "https://xray.us-west-2.amazonaws.com/v1/traces/",
        "https://collector.example.com/v1/traces",
        "https://xray.us-west-2.amazonaws.com.evil.example/v1/traces",
        "http://localhost:4318/v1/traces",
        "",
    ],
)
def test_sigv4_foreign_endpoint_refused(endpoint):
    with pytest.raises(ValueError, match="endpoint"):
        ex.SigV4OtlpSpanExporter(REGION, LOG_GROUP, session=FakeBotocoreSession(None), endpoint=endpoint)


def test_sigv4_exact_endpoint_accepted():
    exp = ex.SigV4OtlpSpanExporter(
        REGION, LOG_GROUP, session=FakeBotocoreSession(None), endpoint=ex.xray_traces_endpoint(REGION)
    )
    assert exp.endpoint == f"https://xray.{REGION}.amazonaws.com/v1/traces"


@pytest.mark.parametrize("region", ["", "US-WEST-2", "us-west-2.evil", "xray", "us-west-2/x"])
def test_sigv4_bad_region_refused(region):
    with pytest.raises(ValueError, match="region"):
        ex.SigV4OtlpSpanExporter(region, LOG_GROUP, session=FakeBotocoreSession(None))


def test_sigv4_requires_log_group():
    with pytest.raises(ValueError, match="log_group"):
        ex.SigV4OtlpSpanExporter(REGION, "", session=FakeBotocoreSession(None))


# ---------------------------------------------------------------- UDP


def test_udp_datagram_byte_exact_to_adot_framing(monkeypatch):
    monkeypatch.delenv("AWS_XRAY_DAEMON_ADDRESS", raising=False)
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2)
    try:
        port = receiver.getsockname()[1]
        exp = ex.UdpSpanExporter(f"127.0.0.1:{port}")
        spans = _spans(("gather.card", "verify"))
        assert exp.export(spans) == SpanExportResult.SUCCESS
        data, _ = receiver.recvfrom(65535)
    finally:
        receiver.close()
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

    expected = (
        b'{"format":"json","version":1}\n'
        + b"T1S"
        + base64.b64encode(encode_spans(spans).SerializeToString())
    )
    assert data == expected
    prefix = b'{"format":"json","version":1}\nT1S'
    assert data.startswith(prefix)
    decoded = ExportTraceServiceRequest.FromString(base64.b64decode(data[len(prefix) :]))
    names = [s.name for s in decoded.resource_spans[0].scope_spans[0].spans]
    assert names == ["gather.card", "verify"]
    exp.shutdown()


class FakeSock:
    def __init__(self, error=None):
        self.error = error
        self.sent = []

    def sendto(self, data, addr):
        if self.error:
            raise self.error
        self.sent.append((data, addr))

    def close(self):
        raise OSError("close fails")


def test_udp_one_datagram_per_export(monkeypatch):
    monkeypatch.delenv("AWS_XRAY_DAEMON_ADDRESS", raising=False)
    sock = FakeSock()
    exp = ex.UdpSpanExporter(sock=sock)
    exp.export(_spans(("a", "b", "c")))
    assert len(sock.sent) == 1
    assert sock.sent[0][1] == ("127.0.0.1", 2000)
    exp.shutdown()  # a failing close is swallowed


@pytest.mark.parametrize("error", [OSError("unreachable"), BlockingIOError(), ValueError("x")])
def test_udp_sendto_error_is_failure(monkeypatch, error):
    monkeypatch.delenv("AWS_XRAY_DAEMON_ADDRESS", raising=False)
    exp = ex.UdpSpanExporter(sock=FakeSock(error))
    assert exp.export(_spans()) == SpanExportResult.FAILURE
    assert exp.failures == 1


def test_udp_default_from_env_non_loopback_allowed(monkeypatch):
    monkeypatch.setenv("AWS_XRAY_DAEMON_ADDRESS", "169.254.79.129:2000")
    exp = ex.UdpSpanExporter(sock=FakeSock())
    assert exp.addr == ("169.254.79.129", 2000)
    exp2 = ex.UdpSpanExporter("169.254.79.129:2000", sock=FakeSock())
    assert exp2.addr == ("169.254.79.129", 2000)


@pytest.mark.parametrize("addr", ["10.0.0.5:2000", "8.8.8.8:2000", "169.254.79.129:2000"])
def test_udp_non_loopback_refused_unless_env(monkeypatch, addr):
    monkeypatch.delenv("AWS_XRAY_DAEMON_ADDRESS", raising=False)
    with pytest.raises(ValueError, match="loopback"):
        ex.UdpSpanExporter(addr, sock=FakeSock())


@pytest.mark.parametrize(
    "addr", ["127.0.0.1", "example.com:2000", "127.0.0.1:0", "127.0.0.1:70000", "::1:2000"]
)
def test_udp_malformed_address_refused(monkeypatch, addr):
    monkeypatch.delenv("AWS_XRAY_DAEMON_ADDRESS", raising=False)
    with pytest.raises(ValueError):
        ex.UdpSpanExporter(addr, sock=FakeSock())


def test_udp_localhost_maps_to_loopback(monkeypatch):
    monkeypatch.delenv("AWS_XRAY_DAEMON_ADDRESS", raising=False)
    assert ex.UdpSpanExporter("localhost:2000", sock=FakeSock()).addr == ("127.0.0.1", 2000)


# ---------------------------------------------------------------- EMF


def _emf_lines(record):
    out = io.StringIO()
    exp = ex.EmfStdoutMetricExporter(
        "bedrock-agentcore", service="owlbank-dev-bridge", environment="dev", stream=out
    )
    reader = PeriodicExportingMetricReader(exp, export_interval_millis=3_600_000)
    provider = MeterProvider(metric_readers=[reader], resource=Resource.create({"service.name": "x"}))
    record(provider.get_meter("strands"))
    provider.force_flush()
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    provider.shutdown()
    return lines


def test_emf_line_shape():
    def record(meter):
        meter.create_gauge("strands.model.time_to_first_token", unit="ms").set(812)

    (line,) = _emf_lines(record)
    ts = line["_aws"]["Timestamp"]
    assert isinstance(ts, int) and ts > 1_600_000_000_000  # milliseconds
    assert line == {
        "_aws": {
            "Timestamp": ts,
            "CloudWatchMetrics": [
                {
                    "Namespace": "bedrock-agentcore",
                    "Dimensions": [["Service", "Environment"]],
                    "Metrics": [{"Name": "strands.model.time_to_first_token", "Unit": "Milliseconds"}],
                }
            ],
        },
        "Service": "owlbank-dev-bridge",
        "Environment": "dev",
        "strands.model.time_to_first_token": 812,
    }


def test_emf_tool_name_dimension_and_unbounded_attributes_dropped():
    def record(meter):
        counter = meter.create_counter("strands.tool.call_count", unit="Count")
        counter.add(1, {"tool_name": "get_balance", "tool_use_id": "tooluse_1", "event_loop_cycle_id": "c1"})
        counter.add(1, {"tool_name": "get_balance", "tool_use_id": "tooluse_2"})

    lines = _emf_lines(record)
    # tool_use_id differs, but it is dropped from the output, not from aggregation: one line per point.
    for line in lines:
        assert line["_aws"]["CloudWatchMetrics"][0]["Dimensions"] == [
            ["Service", "Environment"],
            ["Service", "Environment", "tool_name"],
        ]
        assert line["tool_name"] == "get_balance"
        assert set(line) == {"_aws", "Service", "Environment", "tool_name", "strands.tool.call_count"}
        assert line["_aws"]["CloudWatchMetrics"][0]["Metrics"][0]["Unit"] == "Count"
    assert sum(line["strands.tool.call_count"] for line in lines) == 2


def test_emf_histogram_is_statistic_set():
    def record(meter):
        hist = meter.create_histogram("strands.tool.duration", unit="s")
        for v in (0.5, 1.5, 1.0):
            hist.record(v, {"tool_name": "lookup"})

    (line,) = _emf_lines(record)
    assert line["strands.tool.duration"] == {"Count": 3, "Sum": 3.0, "Min": 0.5, "Max": 1.5}
    assert line["_aws"]["CloudWatchMetrics"][0]["Metrics"][0]["Unit"] == "Seconds"


def test_emf_rejects_unbounded_dimension_config():
    with pytest.raises(ValueError, match="tool_name"):
        ex.EmfStdoutMetricExporter("ns", service="s", environment="e", allowed_dimensions=("tool_use_id",))


def test_emf_write_error_is_failure_not_raise():
    class Broken(io.StringIO):
        def write(self, s):
            raise OSError("closed")

    exp = ex.EmfStdoutMetricExporter("ns", service="s", environment="e", stream=Broken())
    data = metrics_data_for("c", "1", [NumberDataPoint({}, 0, 1, 1)])
    assert exp.export(data) == MetricExportResult.FAILURE


def test_emf_failure_is_a_rate_limited_warning_with_the_error_class_only(caplog):
    class Broken(io.StringIO):
        def write(self, s):
            raise OSError("closed: " + "secret" + "-value")  # not in the source line the stack shows

    exp = ex.EmfStdoutMetricExporter("ns", service="s", environment="e", stream=Broken())
    data = metrics_data_for("c", "1", [NumberDataPoint({}, 0, 1, 1)])
    with caplog.at_level(logging.WARNING, logger="owlbank.telemetry.exporters"):
        assert exp.export(data) == MetricExportResult.FAILURE
        assert exp.export(data) == MetricExportResult.FAILURE
    (record,) = [r for r in caplog.records if r.getMessage() == "metric_export_failed"]
    assert record.levelno == logging.WARNING and record.error_class == "OSError"
    assert "secret-value" not in caplog.text and "secret-value" not in str(record.__dict__)


def test_emf_a_point_that_fails_to_encode_is_a_warning_too(caplog, monkeypatch):
    exp = ex.EmfStdoutMetricExporter("ns", service="s", environment="e", stream=io.StringIO())
    monkeypatch.setattr(exp, "emf_record", lambda *a: 1 / 0)
    data = metrics_data_for("c", "1", [NumberDataPoint({}, 0, 1, 1)])
    with caplog.at_level(logging.WARNING, logger="owlbank.telemetry.exporters"):
        assert exp.export(data) == MetricExportResult.FAILURE
    assert [r.error_class for r in caplog.records if r.getMessage() == "metric_export_failed"] == [
        "ZeroDivisionError"]


def test_cloudwatch_units():
    assert ex.cloudwatch_unit("ms") == "Milliseconds"
    assert ex.cloudwatch_unit("s") == "Seconds"
    assert ex.cloudwatch_unit("token") == "Count"
    assert ex.cloudwatch_unit("furlong") == "None"
    assert ex.cloudwatch_unit(None) == "None"
