"""Call observability exporter properties:

* For any metric point with arbitrary attributes, the EMF line's dimension keys, and its
  top-level keys besides ``_aws`` and the metric name, are within ``{Service, Environment, tool_name}``;
  ``tool_name`` is a dimension exactly when the point has a non-empty one.
* Any configured span endpoint other than ``https://xray.<region>.amazonaws.com/v1/traces``
  for the exporter's own Region is refused; any non-loopback UDP address not from
  ``AWS_XRAY_DAEMON_ADDRESS`` is refused.
"""

from __future__ import annotations

import io
import ipaddress
import json

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from opentelemetry.sdk.metrics.export import (
    Gauge,
    Histogram,
    HistogramDataPoint,
    Metric,
    MetricExportResult,
    MetricsData,
    NumberDataPoint,
    ResourceMetrics,
    ScopeMetrics,
    Sum,
    AggregationTemporality,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.util.instrumentation import InstrumentationScope

from owlbank.telemetry import exporters as ex

BOUNDED = {"Service", "Environment", "tool_name"}


def metrics_data_for(name: str, unit: str, points: list, kind: str = "gauge") -> MetricsData:
    """A one-metric ``MetricsData`` holding ``points`` (gauge, sum or histogram)."""
    if kind == "histogram":
        data = Histogram(data_points=points, aggregation_temporality=AggregationTemporality.DELTA)
    elif kind == "sum":
        data = Sum(
            data_points=points, aggregation_temporality=AggregationTemporality.DELTA, is_monotonic=True
        )
    else:
        data = Gauge(data_points=points)
    metric = Metric(name=name, description="", unit=unit, data=data)
    scope = ScopeMetrics(scope=InstrumentationScope("strands"), metrics=[metric], schema_url="")
    return MetricsData(
        resource_metrics=[ResourceMetrics(resource=Resource.create({}), scope_metrics=[scope], schema_url="")]
    )


_attr_keys = st.one_of(
    st.sampled_from(
        ["tool_name", "tool_use_id", "event_loop_cycle_id", "Service", "Environment", "_aws", "call_sid"]
    ),
    st.text(min_size=1, max_size=20),
)
_attr_values = st.one_of(st.text(max_size=30), st.integers(), st.floats(allow_nan=False), st.booleans())
_attrs = st.dictionaries(_attr_keys, _attr_values, max_size=8)
_names = st.one_of(
    st.sampled_from(
        ["strands.tool.call_count", "strands.model.time_to_first_token", "gen_ai.client.token.usage"]
    ),
    st.text(min_size=1, max_size=30),
)


def _number_point(attrs, value):
    return NumberDataPoint(
        attributes=attrs, start_time_unix_nano=0, time_unix_nano=1_700_000_000_000_000_000, value=value
    )


def _hist_point(attrs, values):
    return HistogramDataPoint(
        attributes=attrs,
        start_time_unix_nano=0,
        time_unix_nano=1_700_000_000_000_000_000,
        count=len(values),
        sum=sum(values),
        bucket_counts=[len(values)],
        explicit_bounds=[],
        min=min(values),
        max=max(values),
    )


@settings(deadline=None, max_examples=300)
@given(
    name=_names,
    unit=st.sampled_from(["ms", "s", "Count", "token", "", "weird"]),
    kind=st.sampled_from(["gauge", "sum", "histogram"]),
    points=st.lists(
        st.tuples(_attrs, st.lists(st.floats(min_value=0, max_value=1e9), min_size=1, max_size=5)),
        min_size=1,
        max_size=4,
    ),
)
def test_emf_dimensions_bounded(name, unit, kind, points):
    built = [_hist_point(a, v) if kind == "histogram" else _number_point(a, v[0]) for a, v in points]
    out = io.StringIO()
    exp = ex.EmfStdoutMetricExporter("bedrock-agentcore", service="svc", environment="dev", stream=out)
    assert exp.export(metrics_data_for(name, unit, built, kind)) == MetricExportResult.SUCCESS
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    if name in {"_aws", *BOUNDED}:
        assert lines == []
        return
    assert len(lines) == len(built)
    for line, (attrs, _) in zip(lines, points, strict=True):
        directive = line["_aws"]["CloudWatchMetrics"][0]
        dim_keys = {key for dims in directive["Dimensions"] for key in dims}
        assert dim_keys <= BOUNDED
        assert set(line) - {"_aws", name} <= BOUNDED
        has_tool = attrs.get("tool_name") is not None and str(attrs["tool_name"]) != ""
        assert ("tool_name" in dim_keys) == has_tool
        assert directive["Dimensions"][0] == ["Service", "Environment"]
        assert line["Service"] == "svc" and line["Environment"] == "dev"
        # Every dimension named is present at the top level (EMF requirement).
        assert dim_keys <= set(line)


@settings(deadline=None)
@given(extra=st.lists(st.text(min_size=1, max_size=15), min_size=1, max_size=3))
def test_dimension_config_bounded(extra):
    assume(set(extra) - {"tool_name"})
    with pytest.raises(ValueError):
        ex.EmfStdoutMetricExporter("ns", service="s", environment="e", allowed_dimensions=extra)


class _NoCreds:
    def get_credentials(self):
        return None


_REGIONS = st.sampled_from(["us-east-1", "us-west-2", "eu-central-1", "ap-southeast-2", "us-gov-west-1"])


def _near_misses(region: str):
    good = ex.xray_traces_endpoint(region)
    return st.one_of(
        st.text(max_size=80),
        _REGIONS.map(ex.xray_traces_endpoint),
        st.sampled_from(
            [
                good.replace("https://", "http://"),
                good + "/",
                good + "?x=1",
                good.upper(),
                " " + good,
                good + " ",
                good.replace("xray.", "xray-fips."),
                good.replace("amazonaws.com", "amazonaws.com.cn"),
                good.replace("/v1/traces", "/v1/logs"),
                good.replace("xray.", "evil.example/xray."),
            ]
        ),
        st.builds(lambda pre, post: pre + good + post, st.text(max_size=10), st.text(max_size=10)),
    )


@settings(deadline=None, max_examples=300)
@given(data=st.data(), region=_REGIONS)
def test_foreign_span_endpoint_refused(data, region):
    endpoint = data.draw(_near_misses(region))
    assume(endpoint != ex.xray_traces_endpoint(region))
    with pytest.raises(ValueError):
        ex.SigV4OtlpSpanExporter(region, "/owlbank/traces/bridge", session=_NoCreds(), endpoint=endpoint)


@settings(deadline=None)
@given(region=_REGIONS)
def test_own_endpoint_accepted(region):
    exp = ex.SigV4OtlpSpanExporter(
        region, "/owlbank/traces/bridge", session=_NoCreds(), endpoint=ex.xray_traces_endpoint(region)
    )
    assert exp.endpoint == f"https://xray.{region}.amazonaws.com/v1/traces"


class _FakeSock:
    def sendto(self, data, addr):
        pass

    def close(self):
        pass


# The fixture only clears AWS_XRAY_DAEMON_ADDRESS once; it holds for every example.
@settings(deadline=None, max_examples=300, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(ip=st.ip_addresses(v=4), port=st.integers(1, 65535))
def test_udp_non_loopback_refused(monkeypatch_env, ip, port):
    addr = f"{ip}:{port}"
    if ipaddress.IPv4Address(ip).is_loopback:
        assert ex.UdpSpanExporter(addr, sock=_FakeSock()).addr == (str(ip), port)
    else:
        with pytest.raises(ValueError):
            ex.UdpSpanExporter(addr, sock=_FakeSock())


@pytest.fixture
def monkeypatch_env(monkeypatch):
    monkeypatch.delenv("AWS_XRAY_DAEMON_ADDRESS", raising=False)
    return monkeypatch
