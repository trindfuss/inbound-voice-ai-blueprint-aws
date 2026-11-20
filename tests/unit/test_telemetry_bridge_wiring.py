"""Call observability: the bridge's tracer and meter providers and their start-up wiring.

``build_app_from_env`` runs offline with the fixture
(``tests/unit/test_agent_wiring.py`` ``offline_start``). OpenTelemetry's global providers can be set
only once per process, so every test passes or patches in recording setters and never touches the
real globals; the SigV4 exporter is replaced by an in-memory one.

Mostly wiring (the filter, sampler and EMF properties are tested elsewhere); one property checks that botocore
instrumentation sends any JSON-object Lambda payload unchanged.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any

import boto3
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st
from opentelemetry import baggage, context, trace
from opentelemetry.sdk.extension.aws.trace import AwsXRayIdGenerator
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased

import tac_bridge.app as bridge_app
from owlbank.telemetry import allowlist
from owlbank.telemetry import bridge as tb
from owlbank.telemetry.settings import TelemetrySettings
from tac_bridge.app import build_app_from_env
from tests.logutil import fields, lines, text_of
from tests.unit.test_agent_wiring import _BRIDGE_ENV, STRANDS_ENV, offline_start  # noqa: F401  (fixture)

SID = "CA" + "5" * 32
LOG_GROUP = "/owlbank/traces/bridge"
ON_ENV = {
    "OWLBANK_CALL_OBSERVABILITY": "on",
    "OWLBANK_STAGE": "dev",
    "OWLBANK_TRACES_LOG_GROUP": LOG_GROUP,
    "OWLBANK_TRACE_SAMPLE_RATE": "1.0",
}


def _settings(**overrides: Any) -> TelemetrySettings:
    base: dict[str, Any] = {
        "stage": "dev", "service_name": "owlbank-dev-bridge", "hosting": "fargate", "sample_rate": 1.0,
        "content": "off", "traces_log_group": LOG_GROUP, "region": "us-east-1",
    }
    return TelemetrySettings(**{**base, **overrides})


class Globals:
    """Recording replacements for OpenTelemetry's global setters (set-once in the real API)."""

    def __init__(self, order: list[str] | None = None) -> None:
        self.order = order if order is not None else []
        self.tracer_provider: Any = None
        self.meter_provider: Any = None
        self.propagator: Any = None

    def set_tracer_provider(self, provider: Any) -> None:
        self.order.append("tracer_provider")
        self.tracer_provider = provider

    def set_meter_provider(self, provider: Any) -> None:
        self.order.append("meter_provider")
        self.meter_provider = provider

    def set_propagator(self, propagator: Any) -> None:
        self.propagator = propagator

    def kwargs(self) -> dict[str, Any]:
        return {"set_tracer_provider": self.set_tracer_provider, "set_meter_provider": self.set_meter_provider,
                "set_propagator": self.set_propagator}


@pytest.fixture
def exported(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """The SigV4 exporter replaced by an in-memory one; records its constructor arguments."""
    seen: dict[str, Any] = {"exporter": InMemorySpanExporter(), "args": []}

    def factory(region: str, log_group: str, **kwargs: Any) -> SpanExporter:
        seen["args"].append((region, log_group))
        return seen["exporter"]

    monkeypatch.setattr(tb, "SigV4OtlpSpanExporter", factory)
    for name in [n for n in os.environ if tb.foreign_endpoints({n: ""})]:
        monkeypatch.delenv(name)
    monkeypatch.setenv(tb.CAPTURE_CONTENT_ENV, "true")  # install_bridge must force it; restored after
    yield seen


@pytest.fixture
def handles() -> Iterator[list[tb.TelemetryHandle]]:
    """Handles to shut down after the test (uninstruments botocore, stops the batch thread)."""
    out: list[tb.TelemetryHandle] = []
    yield out
    for handle in out:
        handle.shutdown()


@pytest.fixture
def patched_globals(monkeypatch: pytest.MonkeyPatch) -> Globals:
    """OpenTelemetry's global setters replaced (install_bridge looks them up at call time)."""
    rec = Globals()
    monkeypatch.setattr("opentelemetry.trace.set_tracer_provider", rec.set_tracer_provider)
    monkeypatch.setattr("opentelemetry.metrics.set_meter_provider", rec.set_meter_provider)
    monkeypatch.setattr("opentelemetry.propagate.set_global_textmap", rec.set_propagator)
    return rec


def _events(caplog: Any, logger: str, name: str) -> list[dict[str, Any]]:
    return [f for f in lines(caplog.records, logger) if f.get("event") == name]


def _install(handles: list[tb.TelemetryHandle], rec: Globals, settings: TelemetrySettings | None = None,
             **kwargs: Any) -> tb.TelemetryHandle:
    handle = tb.install_bridge(settings or _settings(), **rec.kwargs(), **kwargs)
    assert handle is not None
    handles.append(handle)
    return handle


def _span_in_call(handle: tb.TelemetryHandle, name: str = "bridge.turn", **attributes: Any) -> None:
    token = context.attach(baggage.set_baggage("call_sid", SID))
    try:
        with handle.tracer.start_as_current_span(name, attributes=attributes):
            pass
    finally:
        context.detach(token)


# --------------------------------------------------------------------------- install_bridge


def test_install_sets_the_resource_sampler_id_generator_and_propagators(
        exported: dict[str, Any], handles: list[tb.TelemetryHandle]) -> None:
    rec = Globals()
    handle = _install(handles, rec, _settings(hosting="agentcore", stage="test", service_name="owlbank-test-bridge"))
    assert rec.tracer_provider is handle.tracer_provider
    assert rec.meter_provider is handle.meter_provider
    attrs = dict(handle.tracer_provider.resource.attributes)
    assert attrs["service.name"] == "owlbank-test-bridge"
    assert attrs["deployment.environment"] == "test"
    assert attrs["owlbank.hosting"] == "agentcore"
    assert attrs["aws.service.type"] == "gen_ai_agent"
    assert attrs["aws.local.service"] == "owlbank-test-bridge"
    assert tuple(attrs["aws.log.group.names"]) == (LOG_GROUP,)
    assert isinstance(handle.tracer_provider.id_generator, AwsXRayIdGenerator)
    assert isinstance(handle.tracer_provider.sampler, ParentBased)
    assert {"baggage", "traceparent", "x-amzn-trace-id"} <= {f.lower() for f in rec.propagator.fields}
    assert exported["args"] == [("us-east-1", LOG_GROUP)]


def test_a_span_in_a_call_is_exported_through_the_filter_with_the_session(
        exported: dict[str, Any], handles: list[tb.TelemetryHandle]) -> None:
    handle = _install(handles, Globals())
    _span_in_call(handle, **{"gen_ai.agent.tools": ("lookup_balance",),
                             "gen_ai.input.messages": "my ssn is 123-45-6789", "owlbank.turn": 1})
    assert handle.shutdown() is True
    (span,) = exported["exporter"].get_finished_spans()
    assert span.name == "bridge.turn"
    assert span.attributes["session.id"] == SID
    assert span.attributes["call_sid"] == SID
    assert span.attributes["owlbank.turn"] == 1
    assert tuple(span.attributes["gen_ai.agent.tools"]) == ("lookup_balance",)
    assert "gen_ai.input.messages" not in span.attributes  # content off: text dropped by the filter


def test_a_span_outside_a_call_is_not_exported(exported: dict[str, Any], handles: list[tb.TelemetryHandle]) -> None:
    handle = _install(handles, Globals())
    with handle.tracer.start_as_current_span("orphan"):
        pass
    handle.shutdown()
    assert exported["exporter"].get_finished_spans() == ()


def test_redacted_mode_passes_text_through_the_redaction(
        exported: dict[str, Any], handles: list[tb.TelemetryHandle]) -> None:
    handle = _install(handles, Globals(), _settings(content="redacted"), redact=lambda text: text.upper())
    _span_in_call(handle, **{"gen_ai.input.messages": "hello"})
    handle.shutdown()
    (span,) = exported["exporter"].get_finished_spans()
    assert span.attributes["gen_ai.input.messages"] == "HELLO"


def test_shutdown_flushes_within_three_seconds(exported: dict[str, Any], handles: list[tb.TelemetryHandle]) -> None:
    handle = _install(handles, Globals())
    _span_in_call(handle)
    assert exported["exporter"].get_finished_spans() == ()  # batched, not yet exported
    started = time.monotonic()
    assert handle.shutdown() is True
    assert time.monotonic() - started < 3.0
    assert len(exported["exporter"].get_finished_spans()) == 1


def test_shutdown_is_bounded_when_the_exporter_hangs(
        exported: dict[str, Any], monkeypatch: pytest.MonkeyPatch, handles: list[tb.TelemetryHandle]) -> None:
    release = threading.Event()

    class Hanging(SpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            release.wait(10)
            return SpanExportResult.SUCCESS

    monkeypatch.setattr(tb, "SigV4OtlpSpanExporter", lambda *_a, **_k: Hanging())
    handle = _install(handles, Globals())
    _span_in_call(handle)
    started = time.monotonic()
    try:
        assert handle.shutdown(timeout_s=0.3) is False
        assert time.monotonic() - started < 1.5
    finally:
        release.set()


def test_botocore_is_instrumented_without_message_content(
        exported: dict[str, Any], handles: list[tb.TelemetryHandle]) -> None:
    from opentelemetry.instrumentation.botocore import BotocoreInstrumentor  # noqa: PLC0415

    handle = _install(handles, Globals())
    assert os.environ[tb.CAPTURE_CONTENT_ENV] == "false"
    assert BotocoreInstrumentor().is_instrumented_by_opentelemetry
    handle.shutdown()
    assert not BotocoreInstrumentor().is_instrumented_by_opentelemetry


def test_the_meter_provider_writes_emf_with_bounded_dimensions(
        exported: dict[str, Any], handles: list[tb.TelemetryHandle], capsys: Any) -> None:
    import json  # noqa: PLC0415

    handle = _install(handles, Globals())
    counter = handle.meter_provider.get_meter("strands").create_counter("strands.tool.call_count", unit="Count")
    counter.add(1, {"tool_name": "lookup_balance", "tool_use_id": "tooluse_1", "event_loop_cycle_id": "c1"})
    handle.shutdown()  # the reader's final collection
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    (record,) = [r for r in records if "strands.tool.call_count" in r]
    assert record["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "bedrock-agentcore"
    assert set(record) == {"_aws", "Service", "Environment", "tool_name", "strands.tool.call_count"}
    assert (record["Service"], record["Environment"]) == ("owlbank-dev-bridge", "dev")


def test_the_view_keeps_model_id_on_the_ttft_and_latency(
        exported: dict[str, Any], handles: list[tb.TelemetryHandle], capsys: Any) -> None:
    """Model failover: the TTFT and latency histograms keep the ``model_id`` the Strands client sets
    (a dimension set of its own beside the ``[Service, Environment]`` aggregate); other attributes go."""
    import json  # noqa: PLC0415

    handle = _install(handles, Globals())
    meter = handle.meter_provider.get_meter("strands")
    for name in ("strands.model.time_to_first_token", "strands.event_loop.latency"):
        meter.create_histogram(name, unit="ms").record(800, {"model_id": "us.anthropic.claude-haiku-4-5",
                                                             "event_loop_cycle_id": "c1"})
    handle.shutdown()
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    for name in ("strands.model.time_to_first_token", "strands.event_loop.latency"):
        (record,) = [r for r in records if name in r]
        assert set(record) == {"_aws", "Service", "Environment", "model_id", name}
        assert record["model_id"] == "us.anthropic.claude-haiku-4-5"
        assert record["_aws"]["CloudWatchMetrics"][0]["Dimensions"] == [
            ["Service", "Environment"], ["Service", "Environment", "model_id"]]


@pytest.mark.parametrize("name", ["OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
                                  "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"])
def test_a_foreign_otlp_endpoint_is_refused(exported: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                            caplog: Any, name: str) -> None:
    caplog.set_level(logging.INFO, logger=tb.__name__)
    monkeypatch.setenv(name, "https://collector.example:4318")
    rec = Globals()
    assert tb.install_bridge(_settings(), **rec.kwargs()) is None
    assert rec.order == [] and rec.propagator is None
    (refused,) = _events(caplog, tb.__name__, "telemetry_refused")
    assert refused["reason"] == "foreign_otlp_endpoint" and refused["variables"] == [name]
    assert "collector.example" not in caplog.text
    assert exported["args"] == []


def test_a_foreign_endpoint_in_the_passed_env_is_refused(exported: dict[str, Any], caplog: Any) -> None:
    caplog.set_level(logging.INFO, logger=tb.__name__)
    rec = Globals()
    assert tb.install_bridge(_settings(), environ={"OTEL_EXPORTER_OTLP_ENDPOINT": ""}, **rec.kwargs()) is None
    assert rec.order == []
    assert _events(caplog, tb.__name__, "telemetry_refused")


@pytest.mark.parametrize(("field", "reason"), [("traces_log_group", "no_traces_log_group"), ("region", "no_region")])
def test_missing_log_group_or_region_is_refused(exported: dict[str, Any], caplog: Any, field: str, reason: str) -> None:
    caplog.set_level(logging.INFO, logger=tb.__name__)
    rec = Globals()
    assert tb.install_bridge(_settings(**{field: None}), **rec.kwargs()) is None
    assert rec.order == []
    (refused,) = _events(caplog, tb.__name__, "telemetry_refused")
    assert refused["reason"] == reason


def test_an_install_error_is_logged_and_installs_nothing(
        exported: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: Any) -> None:
    caplog.set_level(logging.INFO, logger=tb.__name__)

    def broken(*_a: Any, **_k: Any) -> SpanExporter:
        raise ValueError("boom")

    monkeypatch.setattr(tb, "SigV4OtlpSpanExporter", broken)
    rec = Globals()
    assert tb.install_bridge(_settings(), **rec.kwargs()) is None
    assert rec.order == []
    (failed,) = _events(caplog, tb.__name__, "telemetry_install_failed")
    assert failed["error_class"] == "ValueError"


def test_gen_ai_agent_tools_is_allowed() -> None:
    assert allowlist.allowed("gen_ai.agent.tools")
    assert allowlist.ATTRIBUTES["gen_ai.agent.tools"]
    assert not allowlist.allowed("gen_ai.tool.description")
    assert not allowlist.allowed("gen_ai.tool.json_schema")


def test_transcript_redactor_drops_until_attached_and_on_failure() -> None:
    class Guard:
        def __init__(self, out: str | None = None, fail: bool = False) -> None:
            self.out, self.fail = out, fail

        def text(self, surface: str, text: str, *, language: str, withheld: str, on_failure: str) -> str:
            if self.fail:
                raise RuntimeError("down")
            return withheld if self.out is None else self.out

    redactor = tb.TranscriptRedactor()
    assert redactor("hi") is None
    redactor.attach(Guard(out="[NAME] called"))
    assert redactor("Ann called") == "[NAME] called"
    redactor.attach(Guard(out=None))  # the check was unavailable: withheld
    assert redactor("Ann called") is None
    redactor.attach(Guard(fail=True))
    assert redactor("Ann called") is None


def test_transcript_redactor_runs_the_exact_value_scrub_first() -> None:
    from owlbank import segments  # noqa: PLC0415

    seen: list[str] = []

    class Guard:
        def text(self, surface: str, text: str, **_k: Any) -> str:
            seen.append(text)
            return text

    segments.register_sensitive(SID, ["94105"])
    try:
        assert tb.TranscriptRedactor(Guard())("my zip is 94105") == "my zip is [redacted]"
    finally:
        segments.clear_sensitive(SID)
    assert seen == ["my zip is [redacted]"]


class _PassGuard:
    def text(self, surface: str, text: str, **_k: Any) -> str:
        return text


def test_a_retained_snapshot_scrubs_until_the_ttl_after_the_latest_retain_and_not_after(
        caplog: Any) -> None:
    from owlbank import segments, sensitive  # noqa: PLC0415

    now = [0.0]
    redactor = tb.TranscriptRedactor(_PassGuard(), clock=lambda: now[0])
    caplog.set_level(logging.DEBUG, logger=tb.__name__)
    segments.register_sensitive(SID, ["94105"])
    try:
        redactor.retain(SID)
        sensitive.forget(SID)
        with redactor.scope(SID):
            assert redactor("zip 94105") == "zip [redacted]"  # forgotten, still in the snapshot
        now[0] = tb.RETAIN_TTL_S - 1
        segments.register_sensitive(SID, ["4821"])  # a new session, then its end: merged, TTL restarts
        redactor.retain(SID)
        sensitive.forget(SID)
        now[0] = tb.RETAIN_TTL_S + 1  # past the first retain's TTL, within the second's
        with redactor.scope(SID):
            assert redactor("zip 94105 ssn 4821") == "zip [redacted] ssn [redacted]"
        assert redactor("zip 94105") == "zip [redacted]"  # no call: every call's values
        assert redactor.retained_calls() == (SID,)
        now[0] = 2 * tb.RETAIN_TTL_S  # the TTL after the latest retain
        assert redactor.retained_calls() == ()
        with redactor.scope(SID):
            assert redactor("zip 94105") == "zip 94105"  # expired: dropped
    finally:
        sensitive.forget(SID)
    assert not _events(caplog, tb.__name__, "trace_snapshot_evicted")  # expiry is normal: no log


def test_the_snapshot_cap_drops_the_oldest_and_warns_without_values(
        monkeypatch: pytest.MonkeyPatch, caplog: Any) -> None:
    from owlbank import segments, sensitive  # noqa: PLC0415

    monkeypatch.setattr(tb, "MAX_RETAINED_CALLS", 2)
    now = [0.0]
    redactor = tb.TranscriptRedactor(_PassGuard(), clock=lambda: now[0])
    calls = [f"CA{i}" + "0" * 31 for i in range(5)]
    caplog.set_level(logging.INFO, logger=tb.__name__)
    try:
        for i, call in enumerate(calls[:3]):
            segments.register_sensitive(call, [f"9{i}9{i}9{i}"])
            redactor.retain(call)
            sensitive.forget(call)
        assert redactor.retained_calls() == (calls[1], calls[2])  # the oldest went
        with redactor.scope(calls[0]):
            assert redactor("909090") == "909090"
        with redactor.scope(calls[2]):
            assert redactor("929292") == "[redacted]"
        segments.register_sensitive(calls[3], ["939393"])
        redactor.retain(calls[3])  # a second eviction within the minute: not logged, but counted
        now[0] = tb.LOG_INTERVAL_S + 1
        segments.register_sensitive(calls[4], ["949494"])
        redactor.retain(calls[4])  # the next line carries both evictions since the last one
    finally:
        for call in calls:
            sensitive.forget(call)
    first, second = [r for r in caplog.records if r.getMessage() == "trace_snapshot_evicted"]
    assert first.levelno == logging.WARNING and fields(first)["evicted"] == 1
    assert fields(second)["evicted"] == 2 and fields(second)["total"] == 3
    assert redactor.evicted == 3
    logged = text_of(caplog.records)
    assert not [v for v in ("909090", "919191", "929292", "939393", "949494") if v in logged]


def test_a_second_retain_moves_the_call_to_the_back_for_expiry_and_eviction(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from owlbank import segments, sensitive  # noqa: PLC0415

    monkeypatch.setattr(tb, "MAX_RETAINED_CALLS", 2)
    now = [0.0]
    redactor = tb.TranscriptRedactor(_PassGuard(), clock=lambda: now[0])
    a, b, c = (f"CA{x}" + "0" * 31 for x in "abc")
    try:
        for call, at in ((a, 0.0), (b, 1.0), (a, 500.0)):
            now[0] = at
            segments.register_sensitive(call, [f"{call[2]}{call[2]}7777"])
            redactor.retain(call)
        assert redactor.retained_calls() == (b, a)  # A's latest retain is the newest
        now[0] = tb.RETAIN_TTL_S + 100  # past B's TTL, within A's second one
        assert redactor.retained_calls() == (a,)
        now[0] = 500.0
        redactor = tb.TranscriptRedactor(_PassGuard(), clock=lambda: now[0])
        for call, at in ((a, 0.0), (b, 1.0), (a, 2.0), (c, 3.0)):
            now[0] = at
            segments.register_sensitive(call, [f"{call[2]}{call[2]}7777"])
            redactor.retain(call)
        assert redactor.retained_calls() == (a, c)  # the cap dropped B, not the re-retained A
    finally:
        for call in (a, b, c):
            sensitive.forget(call)


def test_a_failed_retain_warns_once_a_minute_without_values(monkeypatch: pytest.MonkeyPatch, caplog: Any) -> None:
    from owlbank import segments  # noqa: PLC0415

    secret = "94105"

    def boom(_call: str) -> frozenset[str]:
        raise RuntimeError(f"registry {secret} broken")  # a message holding a value: never logged

    monkeypatch.setattr(segments, "sensitive_values", boom)
    redactor = tb.TranscriptRedactor(_PassGuard(), clock=lambda: 0.0)
    caplog.set_level(logging.INFO, logger=tb.__name__)
    redactor.retain(SID)  # never raises
    redactor.retain(SID)
    (warn,) = [r for r in caplog.records if r.getMessage() == "trace_retain_failed"]
    assert warn.levelno == logging.WARNING and fields(warn)["error_class"] == "RuntimeError"
    assert secret not in text_of(caplog.records)


# --------------------------------------------------------------------------- build_app_from_env


def test_off_imports_no_telemetry_module_and_sets_no_provider(
        offline_start: dict[str, Any], patched_globals: Globals,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch, caplog: Any) -> None:
    caplog.set_level(logging.INFO, logger="tac_bridge.app")
    monkeypatch.delitem(sys.modules, "owlbank.telemetry.bridge", raising=False)
    monkeypatch.delenv("OWLBANK_CALL_OBSERVABILITY", raising=False)
    app = build_app_from_env({**_BRIDGE_ENV, **STRANDS_ENV})
    with TestClient(app):
        pass
    assert "owlbank.telemetry.bridge" not in sys.modules
    assert patched_globals.order == [] and patched_globals.propagator is None
    assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)
    assert not _events(caplog, "tac_bridge.app", "telemetry_started")
    assert getattr(app.state.bridge, "_trace_retain", None) is None
    assert getattr(app.state.bridge, "_trace_flush", None) is None


def test_on_installs_before_the_strands_start_and_shuts_down_with_the_app(
        offline_start: dict[str, Any], exported: dict[str, Any], patched_globals: Globals,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch, caplog: Any) -> None:
    caplog.set_level(logging.INFO, logger="tac_bridge.app")
    order = patched_globals.order
    real_start = bridge_app.build_strands_client

    def start(settings: Any) -> Any:
        order.append("strands_start")
        return real_start(settings)

    monkeypatch.setattr(bridge_app, "build_strands_client", start)
    stopped: list[bool] = []
    real_install = tb.install_bridge

    def install(*args: Any, **kwargs: Any) -> tb.TelemetryHandle | None:
        handle = real_install(*args, **kwargs)
        assert handle is not None
        real_shutdown = handle.shutdown
        monkeypatch.setattr(handle, "shutdown", lambda timeout_s=3.0: stopped.append(True) or real_shutdown(timeout_s))
        return handle

    monkeypatch.setattr(tb, "install_bridge", install)
    app = build_app_from_env({**_BRIDGE_ENV, **STRANDS_ENV, **ON_ENV})
    assert order.index("tracer_provider") < order.index("strands_start")
    assert order.index("meter_provider") < order.index("strands_start")
    (started,) = _events(caplog, "tac_bridge.app", "telemetry_started")
    assert started["trace_content"] == "off" and started["sample_rate"] == 1.0
    with TestClient(app):
        pass
    assert stopped == [True]
    assert patched_globals.tracer_provider.resource.attributes["service.name"] == "owlbank-dev-bridge"
    assert app.state.bridge.tracing is not None  # traced, content off: no snapshot, no session-end flush
    assert app.state.bridge._trace_retain is None
    assert app.state.bridge._trace_flush is None


def test_on_with_redacted_attaches_the_bridge_pii_guard(
        offline_start: dict[str, Any], exported: dict[str, Any], patched_globals: Globals,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    real_install = tb.install_bridge

    def install(settings: TelemetrySettings, **kwargs: Any) -> tb.TelemetryHandle | None:
        seen["redact"] = kwargs.get("redact")
        return real_install(settings, **kwargs)

    monkeypatch.setattr(tb, "install_bridge", install)
    app = build_app_from_env({**_BRIDGE_ENV, **STRANDS_ENV, **ON_ENV, "OWLBANK_TRACE_CONTENT": "redacted"})
    assert isinstance(seen["redact"], tb.TranscriptRedactor)
    assert seen["redact"]._guard is app.state.bridge._pii
    # The production wiring: the bridge retains through this redactor and flushes at session end.
    assert app.state.bridge._trace_retain == seen["redact"].retain
    assert app.state.bridge._trace_flush is not None
    with TestClient(app):
        pass


def test_on_with_full_content_retains_nothing_and_does_not_flush_at_session_end(
        offline_start: dict[str, Any], exported: dict[str, Any], patched_globals: Globals,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_app_from_env({**_BRIDGE_ENV, **STRANDS_ENV, **ON_ENV, "OWLBANK_TRACE_CONTENT": "full"})
    assert app.state.bridge.tracing is not None
    assert app.state.bridge._trace_retain is None
    assert app.state.bridge._trace_flush is None
    with TestClient(app):
        pass


def test_on_with_a_foreign_endpoint_starts_untraced(
        offline_start: dict[str, Any], exported: dict[str, Any], patched_globals: Globals,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "https://collector.example:4318/v1/traces")
    app = build_app_from_env({**_BRIDGE_ENV, **STRANDS_ENV, **ON_ENV})
    assert _events(caplog, tb.__name__, "telemetry_refused")
    assert not _events(caplog, "tac_bridge.app", "telemetry_started")
    assert patched_globals.order == []
    with TestClient(app):
        pass


# --------------------------------------------------------------------------- botocore leaves requests alone
# the Lambda extension of opentelemetry-instrumentation-botocore rewrote each Invoke
# Payload (adding a ``headers`` key), and bank_tools rejects unknown keys, so every tool call failed
# with tracing on. 4.11: botocore spans stay, request parameters and payloads never change.

_JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(),
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(), inner, max_size=4),
    max_leaves=12,
)


def _stubbed(service: str) -> tuple[Any, Stubber]:
    client = boto3.client(service, region_name="us-east-1", aws_access_key_id="AKIDEXAMPLE",
                          aws_secret_access_key="secret")  # noqa: S106 - fake, never sent (stubbed)
    return client, Stubber(client)


def _invoke_in_call(client: Any, stubber: Stubber, payload: bytes) -> None:
    stubber.add_response(
        "invoke", {"StatusCode": 200, "Payload": StreamingBody(io.BytesIO(b"{}"), 2)},
        expected_params={"FunctionName": "owlbank-dev-bank-tools", "InvocationType": "RequestResponse",
                         "Payload": payload},
    )
    token = context.attach(baggage.set_baggage("call_sid", SID))
    try:
        with stubber:
            client.invoke(FunctionName="owlbank-dev-bank-tools", InvocationType="RequestResponse", Payload=payload)
        stubber.assert_no_pending_responses()
    finally:
        context.detach(token)


def _client_spans(exported: dict[str, Any], handle: tb.TelemetryHandle, name: str) -> list[ReadableSpan]:
    handle.tracer_provider.force_flush()
    return [s for s in exported["exporter"].get_finished_spans()
            if s.name == name and s.kind is trace.SpanKind.CLIENT]


def test_a_lambda_invoke_sends_its_payload_unchanged_and_still_gets_a_client_span(
        exported: dict[str, Any], handles: list[tb.TelemetryHandle]) -> None:
    _install(handles, Globals())
    client, stubber = _stubbed("lambda")
    payload = json.dumps({"tool": "get_balance", "args": {"account": "chk"}, "call_sid": SID}).encode()
    _invoke_in_call(client, stubber, payload)  # Stubber fails on any change to the Payload bytes
    spans = _client_spans(exported, handles[0], "Lambda.Invoke")
    assert len(spans) == 1
    assert spans[0].attributes is not None and spans[0].attributes.get("rpc.method") == "Invoke"


def test_any_json_object_payload_is_sent_exactly_as_given(
        exported: dict[str, Any], handles: list[tb.TelemetryHandle]) -> None:
    handle = _install(handles, Globals())
    client, stubber = _stubbed("lambda")

    @settings(max_examples=60, deadline=None)
    @given(body=st.dictionaries(st.text(), _JSON, max_size=6))
    def sent_unchanged(body: dict[str, Any]) -> None:
        _invoke_in_call(client, stubber, json.dumps(body).encode())

    sent_unchanged()
    assert len(_client_spans(exported, handle, "Lambda.Invoke")) >= 1


def test_an_sns_publish_keeps_its_message_attributes(
        exported: dict[str, Any], handles: list[tb.TelemetryHandle]) -> None:
    _install(handles, Globals())  # the bridge makes no SNS calls; the guard covers every service anyway
    client, stubber = _stubbed("sns")
    topic = "arn:aws:sns:us-east-1:123456789012:t"
    stubber.add_response("publish", {"MessageId": "m-1"}, expected_params={"TopicArn": topic, "Message": "hi"})
    token = context.attach(baggage.set_baggage("call_sid", SID))
    try:
        with stubber:
            client.publish(TopicArn=topic, Message="hi")
    finally:
        context.detach(token)
    handles[0].tracer_provider.force_flush()  # SNS Publish is a PRODUCER span named after the topic
    assert [s.attributes.get("rpc.method") for s in exported["exporter"].get_finished_spans()
            if s.attributes is not None and s.attributes.get("rpc.service") == "SNS"] == ["Publish"]


def test_uninstrument_removes_the_payload_guard(exported: dict[str, Any], handles: list[tb.TelemetryHandle]) -> None:
    from opentelemetry.instrumentation.botocore import BotocoreInstrumentor  # noqa: PLC0415

    handle = _install(handles, Globals())
    assert "_patched_api_call" in vars(BotocoreInstrumentor())
    assert handle.shutdown()
    assert "_patched_api_call" not in vars(BotocoreInstrumentor())
    assert not BotocoreInstrumentor().is_instrumented_by_opentelemetry
