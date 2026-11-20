"""Front-door telemetry install (call observability).

* Off (the default, or an invalid setting): the provider is never built and OpenTelemetry is never
  imported by the front door, at INIT or on a request (checked in a fresh interpreter).
* On: installed at INIT exactly once per process, however many times the app is built.
* The real wiring's boto3 Lambda client carries the ``X-Amzn-Trace-Id`` header of the current span
  on ``Invoke`` (the ``before-sign`` handler), and none when off. Only a sampled span sets it, and
  it keeps botocore's ``Lineage=``; an unsampled span leaves botocore's header untouched.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lambdas.frontdoor import backend, handler
from owlbank.telemetry import frontdoor
from owlbank.telemetry.settings import TelemetrySettings
from tests.unit.test_coldstart import aws_env  # noqa: F401 - fixture

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test: telemetry not installed, and ``install_telemetry`` not yet run in this "process"."""
    monkeypatch.setattr(frontdoor, "_tracer", None)
    monkeypatch.setattr(handler, "_telemetry_checked", False)
    monkeypatch.setattr(handler, "_app", None)
    for name in ("OWLBANK_CALL_OBSERVABILITY", "OWLBANK_TRACE_SAMPLE_RATE", "_X_AMZN_TRACE_ID"):
        monkeypatch.delenv(name, raising=False)


def _spy(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Count ``install_frontdoor`` calls; each really installs, against an in-memory exporter."""
    calls: list[Any] = []
    real = frontdoor.install_frontdoor

    def spy(settings: Any, **kw: Any) -> bool:
        calls.append(settings)
        return real(settings, exporter=InMemorySpanExporter())

    monkeypatch.setattr(frontdoor, "install_frontdoor", spy)
    return calls


def test_off_in_a_fresh_interpreter_imports_no_opentelemetry() -> None:
    """Importing the front door and running its telemetry install with the switch off (unset,
    ``off`` or invalid) imports no ``opentelemetry`` module and installs nothing."""
    code = textwrap.dedent("""
        import json, sys
        from lambdas.frontdoor import handler
        from owlbank.telemetry import frontdoor
        out = []
        for env in ({}, {"OWLBANK_CALL_OBSERVABILITY": "off"},
                    {"OWLBANK_CALL_OBSERVABILITY": "yes"},
                    {"OWLBANK_CALL_OBSERVABILITY": "on", "OWLBANK_TRACE_SAMPLE_RATE": "2"}):
            handler._telemetry_checked = False
            out.append(handler.install_telemetry(env))
        with frontdoor.webhook_span("factor", "CA" + "f" * 32):
            frontdoor.set_step("verify")
            frontdoor.add_fields(factor="zip")
            with frontdoor.child_span("lambda.invoke verify") as span:
                span.set("owlbank.outcome", "ok")
        print(json.dumps({"installed": out, "tracer": frontdoor.installed(),
                          "header": frontdoor.lambda_trace_header(),
                          "otel": sorted(m for m in sys.modules if m.startswith("opentelemetry"))}))
    """)
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,  # noqa: S603
                          cwd=ROOT, check=True, timeout=120)
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result == {"installed": [False, False, False, False], "tracer": False, "header": None,
                      "otel": []}


def test_off_never_calls_install_frontdoor(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy(monkeypatch)
    assert handler.install_telemetry() is False
    assert calls == [] and not frontdoor.installed()


def test_an_invalid_setting_leaves_the_front_door_untraced(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy(monkeypatch)
    monkeypatch.setenv("OWLBANK_CALL_OBSERVABILITY", "on")
    monkeypatch.setenv("OWLBANK_TRACE_SAMPLE_RATE", "1.5")
    assert handler.install_telemetry() is False
    assert calls == [] and not frontdoor.installed()


def test_on_installs_at_init_once_per_process(aws_env, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    calls = _spy(monkeypatch)
    monkeypatch.setenv("OWLBANK_CALL_OBSERVABILITY", "on")
    monkeypatch.setenv("OWLBANK_STAGE", "dev")
    app = handler._build_at_init()                 # INIT
    assert frontdoor.installed()
    assert [s.service_name for s in calls] == ["owlbank-dev-frontdoor"]
    handler.new_app()                              # a rebuild in the same process
    assert handler.install_telemetry() is False
    assert len(calls) == 1
    assert app is not None


def test_install_failure_is_logged_and_untraced(monkeypatch: pytest.MonkeyPatch) -> None:
    from owlbank.telemetry.settings import TelemetrySettings

    settings = TelemetrySettings.from_env({"OWLBANK_CALL_OBSERVABILITY": "on"}, service="frontdoor")
    assert settings is not None
    monkeypatch.setattr("owlbank.telemetry.exporters.UdpSpanExporter.__init__",
                        lambda self, *a, **k: (_ for _ in ()).throw(OSError("no socket")))
    assert frontdoor.install_frontdoor(settings) is False
    assert not frontdoor.installed()


class _Raw:
    """The raw HTTP body botocore reads a streaming ``Payload`` from."""

    def __init__(self, data: bytes) -> None:
        self._io = io.BytesIO(data)

    def read(self, *a: Any, **k: Any) -> bytes:
        return self._io.read(*a)

    def stream(self, **_k: Any) -> Any:
        yield self._io.read()


def _capturing_client(app: Any) -> list[dict[str, str]]:
    """Answer the real boto3 Lambda client's ``Invoke`` locally, keeping each request's headers."""
    from botocore.awsrequest import AWSResponse

    seen: list[dict[str, str]] = []
    client = app.d.backend._client

    def send(request: Any, **_k: Any) -> AWSResponse:
        seen.append(dict(request.headers))
        return AWSResponse(request.url, 200, {"X-Amz-Executed-Version": "1"}, _Raw(b'{"result": "ok"}'))

    client.meta.events.register("before-send.lambda.Invoke", send)
    return seen


def _header(headers: dict[str, Any]) -> str | None:
    value = next((v for k, v in headers.items() if k.lower() == "x-amzn-trace-id"), None)
    return value.decode() if isinstance(value, bytes) else value


def test_the_real_lambda_client_sends_the_trace_header_when_on(aws_env, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setenv("OWLBANK_CALL_OBSERVABILITY", "on")
    _spy(monkeypatch)
    app = handler._build_at_init()
    seen = _capturing_client(app)
    with frontdoor.webhook_span("factor", "CA" + "f" * 32):
        assert app.d.backend.precheck(call_sid="CA" + "f" * 32, ani=None, target_level=1) == "ok"
    (headers,) = seen
    header = _header(headers)
    assert header is not None and header.startswith("Root=1-") and header.endswith(";Sampled=1")


def test_the_real_lambda_client_sends_no_trace_header_when_off(aws_env) -> None:  # noqa: F811
    app = handler._build_at_init()
    seen = _capturing_client(app)
    assert app.d.backend.precheck(call_sid="CA" + "f" * 32, ani=None, target_level=1) == "ok"
    (headers,) = seen
    assert _header(headers) is None


# ---------------------------------------------------------------- the before-sign handler

LINEAGE = "Lineage=1a2b3c4d:0"
BOTOCORE_HEADER = f"Root=1-5759e988-bd862e3fe1be46a994272793;Parent=53995c3f42cd8ad8;Sampled=1;{LINEAGE}"


class _Request:
    def __init__(self, header: str | None) -> None:
        self.headers: dict[str, str] = {} if header is None else {backend.TRACE_HEADER: header}


def _install(rate: str) -> None:
    env = {"OWLBANK_CALL_OBSERVABILITY": "on", "OWLBANK_STAGE": "dev", "OWLBANK_TRACE_SAMPLE_RATE": rate}
    settings = TelemetrySettings.from_env(env, service="frontdoor")
    assert settings is not None and frontdoor.install_frontdoor(settings, exporter=InMemorySpanExporter())


def _sent(existing: str | None) -> tuple[str | None, str]:
    """The header the handler leaves on a request inside a span, and that span's own Root/Parent."""
    request = _Request(existing)
    with frontdoor.webhook_span("factor", "CA" + "f" * 32):
        from opentelemetry import trace

        sc = trace.get_current_span().get_span_context()
        backend._add_trace_header(request)
        trace_hex = format(sc.trace_id, "032x")
        own = f"Root=1-{trace_hex[:8]}-{trace_hex[8:]};Parent={format(sc.span_id, '016x')}"
    return request.headers.get(backend.TRACE_HEADER), own


def test_a_sampled_span_sets_root_and_parent_with_sampled_1_and_keeps_lineage() -> None:
    _install("1.0")
    header, own = _sent(BOTOCORE_HEADER)
    assert header == f"{own};Sampled=1;{LINEAGE}"


def test_a_sampled_span_without_an_existing_header_adds_no_lineage() -> None:
    _install("1.0")
    header, own = _sent(None)
    assert header == f"{own};Sampled=1"
    header, own = _sent("Root=1-5759e988-bd862e3fe1be46a994272793;Sampled=1")
    assert header == f"{own};Sampled=1"


def test_an_unsampled_span_leaves_the_existing_header_untouched() -> None:
    _install("0.0")
    header, _ = _sent(BOTOCORE_HEADER)
    assert header == BOTOCORE_HEADER
    header, _ = _sent(None)
    assert header is None
    with frontdoor.webhook_span("factor", "CA" + "f" * 32):
        assert frontdoor.lambda_trace_header(BOTOCORE_HEADER) is None


def test_the_real_lambda_client_keeps_lineage_and_leaves_unsampled_alone(aws_env, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setenv("OWLBANK_CALL_OBSERVABILITY", "on")
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "frontdoor")  # botocore then adds the Lambda header
    monkeypatch.setenv("_X_AMZN_TRACE_ID", BOTOCORE_HEADER)
    _spy(monkeypatch)
    app = handler._build_at_init()
    seen = _capturing_client(app)
    call = "CA" + "f" * 32
    with frontdoor.webhook_span("factor", call):
        assert app.d.backend.precheck(call_sid=call, ani=None, target_level=1) == "ok"
    header = _header(seen[0])
    assert header is not None and header.endswith(f";Sampled=1;{LINEAGE}")
    assert header.count("Lineage=") == 1 and header != BOTOCORE_HEADER
    monkeypatch.setattr(frontdoor, "_tracer", None)
    _install("0.0")
    seen.clear()
    with frontdoor.webhook_span("factor", call):
        assert app.d.backend.precheck(call_sid=call, ani=None, target_level=1) == "ok"
    assert _header(seen[0]) == BOTOCORE_HEADER
