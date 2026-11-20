"""The bridge never exports telemetry to a third party.

Strands and the OpenTelemetry SDK export spans when an OTLP endpoint or the console exporter is
configured through the environment, so the bridge's containers (plain environment variables or
secrets) are checked in both modes:

* ``callObservability=off`` (every existing context): no OpenTelemetry variable at all.
* ``callObservability=on``: only the documented content-control variables
  (``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``, and ``OTEL_SEMCONV_STABILITY_OPT_IN`` with
  ``traceContent=off``). The exporter is built in code and sends only to X-Ray in the deployment's
  own Region, naming our traces log group through ``OWLBANK_TRACES_LOG_GROUP``; no
  ``OTEL_EXPORTER_*`` endpoint or header variable, and never ``STRANDS_OTEL_ENABLE_CONSOLE_EXPORT``.
"""

from __future__ import annotations

import warnings

import pytest
from conftest import _synth

from owlbank_infra.call_observability import TRACES_LOG_GROUP_NAME

FORBIDDEN = {
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "STRANDS_OTEL_ENABLE_CONSOLE_EXPORT",
}
#: The only OpenTelemetry variables the bridge may carry with the switch on.
ALLOWED_OTEL_ON = {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "OTEL_SEMCONV_STABILITY_OPT_IN"}


def _otel_like(name: str) -> bool:
    return name.startswith(("OTEL_", "STRANDS_OTEL_"))


def _bridge_containers(synth) -> list[dict[str, str | None]]:
    """Per container: variable name -> value (``None`` for a secret)."""
    tpl = synth.templates["OwlBankBridge"]
    per_container = []
    for res in tpl["Resources"].values():
        if res["Type"] != "AWS::ECS::TaskDefinition":
            continue
        for container in res["Properties"]["ContainerDefinitions"]:
            env: dict[str, str | None] = {e["Name"]: e["Value"] for e in container.get("Environment", [])}
            env |= {s["Name"]: None for s in container.get("Secrets", [])}
            per_container.append(env)
    return per_container


def test_bridge_sets_no_otel_exporter(synth_any) -> None:
    """Off (every existing context): no exporter variable, and no OpenTelemetry variable at all."""
    containers = _bridge_containers(synth_any)
    assert containers, "no ECS task definition found in the OwlBankBridge stack"
    for env in containers:
        names = set(env)
        assert not names & FORBIDDEN, f"OTEL exporter variables set on the bridge: {sorted(names & FORBIDDEN)}"
        assert not {n for n in names if n.startswith("OTEL_EXPORTER_")}, sorted(names)
        assert not {n for n in names if _otel_like(n)}, sorted(names)


@pytest.fixture(scope="module", params=["off", "redacted"])
def synth_on(request, tmp_path_factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # traceContent != off warns
        return _synth(tmp_path_factory, f"no-otel-export-on-{request.param}",
                      {"callObservability": "on", "traceContent": request.param})


def test_bridge_on_allows_only_documented_otel_vars(synth_on) -> None:
    """On: only the content controls, the traces log group named, nothing else."""
    containers = _bridge_containers(synth_on)
    assert containers
    for env in containers:
        names = set(env)
        assert not names & FORBIDDEN, sorted(names & FORBIDDEN)
        assert not {n for n in names if n.startswith("OTEL_EXPORTER_")}, sorted(names)
        assert {n for n in names if _otel_like(n)} <= ALLOWED_OTEL_ON, sorted(names)
        assert env["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] == "false"
        assert env["OWLBANK_TRACES_LOG_GROUP"] == TRACES_LOG_GROUP_NAME
        # No endpoint or URL in any OpenTelemetry variable.
        for n in names & ALLOWED_OTEL_ON:
            assert "://" not in (env[n] or ""), n
        if synth_on.cfg.trace_content == "off":
            assert env["OTEL_SEMCONV_STABILITY_OPT_IN"] == "gen_ai_unredacted_attributes="
        else:
            assert "OTEL_SEMCONV_STABILITY_OPT_IN" not in names
