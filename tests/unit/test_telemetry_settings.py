"""Call observability: :meth:`TelemetrySettings.from_env`.

Off (absent, ``off``, anything but ``on``) gives ``None``; on gives the defaults; an invalid value
gives ``None`` and one ``telemetry_disabled`` line naming the setting, never its value. The module
imports no OpenTelemetry.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from owlbank.telemetry.settings import TelemetrySettings

from tests.logutil import lines, text_of

LOGGER = "owlbank.telemetry.settings"
ON = {"OWLBANK_CALL_OBSERVABILITY": "on"}


@pytest.mark.parametrize("env", [{}, {"OWLBANK_CALL_OBSERVABILITY": "off"},
                                 {"OWLBANK_CALL_OBSERVABILITY": ""}])
def test_off_is_none_and_silent(env: dict[str, str], caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    assert TelemetrySettings.from_env(env, service="bridge") is None
    assert lines(caplog.records, LOGGER) == []


def test_unknown_switch_value_is_off_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    assert TelemetrySettings.from_env({"OWLBANK_CALL_OBSERVABILITY": "yes-please"}, service="bridge") is None
    assert [ln["setting"] for ln in lines(caplog.records, LOGGER)] == ["OWLBANK_CALL_OBSERVABILITY"]
    assert "yes-please" not in text_of(caplog.records)


def test_defaults() -> None:
    s = TelemetrySettings.from_env(ON, service="bridge")
    assert s == TelemetrySettings(stage="dev", service_name="owlbank-dev-bridge", hosting="fargate",
                                  sample_rate=1.0, content="off", traces_log_group=None, region=None)


def test_all_values_read() -> None:
    env = {**ON, "OWLBANK_STAGE": "prod", "OWLBANK_HOSTING": "agentcore", "OWLBANK_TRACE_SAMPLE_RATE": "0.25",
           "OWLBANK_TRACE_CONTENT": "redacted", "OWLBANK_TRACES_LOG_GROUP": "/owlbank/traces/bridge",
           "AWS_REGION": "us-east-1"}
    s = TelemetrySettings.from_env(env, service="frontdoor")
    assert s is not None
    assert (s.stage, s.service_name, s.hosting, s.sample_rate, s.content, s.traces_log_group, s.region) == (
        "prod", "owlbank-prod-frontdoor", "agentcore", 0.25, "redacted", "/owlbank/traces/bridge", "us-east-1")


def test_settings_are_frozen() -> None:
    s = TelemetrySettings.from_env(ON, service="bridge")
    assert s is not None
    with pytest.raises(AttributeError):
        s.sample_rate = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(("name", "value"), [
    ("OWLBANK_TRACE_SAMPLE_RATE", "1.5"),
    ("OWLBANK_TRACE_SAMPLE_RATE", "-0.1"),
    ("OWLBANK_TRACE_SAMPLE_RATE", "nan"),
    ("OWLBANK_TRACE_SAMPLE_RATE", "inf"),
    ("OWLBANK_TRACE_SAMPLE_RATE", "half"),
    ("OWLBANK_TRACE_CONTENT", "everything"),
    ("OWLBANK_TRACE_CONTENT", "OFF"),
    ("OWLBANK_HOSTING", "lambda"),
    ("OWLBANK_STAGE", "Prod Stage!"),
])
def test_invalid_value_disables_and_names_the_setting(name: str, value: str,
                                                      caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    assert TelemetrySettings.from_env({**ON, name: value}, service="bridge") is None
    logged = lines(caplog.records, LOGGER)
    assert [(ln["event"], ln["setting"]) for ln in logged] == [("telemetry_disabled", name)]
    assert value not in text_of(caplog.records)


@pytest.mark.parametrize("rate", ["0", "0.0", "1", "1.0", "0.5"])
def test_rate_bounds_inclusive(rate: str) -> None:
    s = TelemetrySettings.from_env({**ON, "OWLBANK_TRACE_SAMPLE_RATE": rate}, service="bridge")
    assert s is not None and s.sample_rate == float(rate)


def test_settings_module_imports_no_opentelemetry() -> None:
    code = ("import sys; import owlbank.telemetry, owlbank.telemetry.settings, owlbank.telemetry.allowlist; "
            "print([m for m in sys.modules if m.startswith('opentelemetry')])")
    src = Path(__file__).resolve().parents[2] / "src"
    env = {**os.environ, "PYTHONPATH": str(src)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)  # noqa: S603
    assert out.stdout.strip() == "[]"
