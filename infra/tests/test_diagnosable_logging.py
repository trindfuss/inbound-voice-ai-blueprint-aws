"""Diagnosable logging (infra): JSON Lambda logging, log levels, saved Insights queries."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from conftest import _synth
from owlbank_infra.config import LOG_LEVELS, from_context, normalize_log_level
from owlbank_infra.observability import APP_LOG_GROUP_NAMES, SAVED_QUERIES

FUNCTIONS = ("frontdoor", "auth_service", "bank_tools", "level0_tools", "context_api", "flex_authorizer")
QUERY_NAMES = (
    "OwlBank/Trace a call by call SID",
    "OwlBank/Errors by component",
    "OwlBank/Slowest invocations and dependency calls",
    "OwlBank/Auth failures",
    "OwlBank/Escalations by reason",
)


@pytest.fixture(scope="module")
def synth_debug(tmp_path_factory):
    return _synth(tmp_path_factory, "debug", {"logLevel": "debug"})


@pytest.fixture(scope="module")
def synth_error(tmp_path_factory):
    return _synth(tmp_path_factory, "error", {"logLevel": "ERROR"})


def _lambdas(synth):
    """Lambda functions of the app handlers: the six, the model-failover poller when on, the
    human-leg conversation sweep when on, and the recap writer when on (not CDK's
    custom-resource Lambdas)."""
    out = [r["Properties"] for _, _, r in synth.resources("AWS::Lambda::Function")
           if r["Properties"].get("Handler", "").startswith("lambdas.")]
    assert len(out) == 6 + (1 if synth.cfg.failover_on else 0) + (1 if synth.cfg.human_leg_conversation_on else 0) \
        + (1 if synth.cfg.recap_writer_on else 0)
    return out


def _check(synth, app_level, system_level, env_level):
    for props in _lambdas(synth):
        cfg = props["LoggingConfig"]
        assert cfg["LogFormat"] == "JSON"
        assert cfg["ApplicationLogLevel"] == app_level
        assert cfg["SystemLogLevel"] == system_level
        assert "LogGroup" in cfg  # the explicit, CMK-encrypted group is kept
        assert props["Environment"]["Variables"]["OWLBANK_LOG_LEVEL"] == env_level


def _bridge_env(synth):
    (_, _, td), = synth.resources("AWS::ECS::TaskDefinition")
    (container,) = td["Properties"]["ContainerDefinitions"]
    return {e["Name"]: e["Value"] for e in container["Environment"]}


def test_default_log_level_is_info(synth_any):
    _check(synth_any, "INFO", "INFO", "INFO")
    assert _bridge_env(synth_any)["OWLBANK_LOG_LEVEL"] == "INFO"
    assert _bridge_env(synth_any)["TWILIO_LOG_LEVEL"] == "INFO"


def test_debug_log_level(synth_debug):
    _check(synth_debug, "DEBUG", "DEBUG", "DEBUG")
    assert _bridge_env(synth_debug)["OWLBANK_LOG_LEVEL"] == "DEBUG"
    assert _bridge_env(synth_debug)["TWILIO_LOG_LEVEL"] == "INFO"  # untouched


def test_error_level_maps_system_to_warn(synth_error):
    # Lambda system logs have no ERROR level.
    _check(synth_error, "ERROR", "WARN", "ERROR")
    assert _bridge_env(synth_error)["OWLBANK_LOG_LEVEL"] == "ERROR"


def test_six_functions_have_explicit_log_groups_and_names(synth_default):
    groups = {r["Properties"]["LogGroupName"] for _, _, r in synth_default.resources("AWS::Logs::LogGroup")}
    for name in FUNCTIONS:
        assert f"/owlbank/lambda/{name}" in groups


def test_five_saved_queries_with_exact_names(synth_any):
    defs = [r["Properties"] for _, _, r in synth_any.resources("AWS::Logs::QueryDefinition")]
    assert sorted(d["Name"] for d in defs) == sorted(QUERY_NAMES)
    assert [q.name for q in SAVED_QUERIES] == list(QUERY_NAMES)


def test_saved_queries_are_created_once(synth_default):
    stacks = {s for s, _, _ in synth_default.resources("AWS::Logs::QueryDefinition")}
    assert len(stacks) == 1


def test_trace_query_covers_every_app_log_group_and_is_parameterised(synth_default):
    defs = {r["Properties"]["Name"]: r["Properties"]
            for _, _, r in synth_default.resources("AWS::Logs::QueryDefinition")}
    trace = defs["OwlBank/Trace a call by call SID"]
    expected = {f"/owlbank/lambda/{n}" for n in FUNCTIONS} | {"/owlbank/ecs/tac-bridge"}
    assert set(trace["LogGroupNames"]) == expected == set(APP_LOG_GROUP_NAMES)
    q = trace["QueryString"]
    assert "filter call_sid = {{call_sid}}" in q
    assert q.index("filter") < q.index("sort @timestamp asc")
    (param,) = trace["Parameters"]
    assert param["Name"] == "call_sid"
    assert param["DefaultValue"].startswith('"CA')


def test_every_query_group_exists_in_the_app(synth_default):
    """Query log-group names are strings, so check they match real log groups."""
    groups = {r["Properties"]["LogGroupName"] for _, _, r in synth_default.resources("AWS::Logs::LogGroup")}
    for _, _, r in synth_default.resources("AWS::Logs::QueryDefinition"):
        assert set(r["Properties"]["LogGroupNames"]) <= groups


@pytest.mark.parametrize("name,needles", [
    ("OwlBank/Errors by component",
     ['level in ["ERROR", "WARNING", "WARN", "CRITICAL"]', 'outcome = "error"', "stats count(*) as occurrences by component, event",
      "sort occurrences desc"]),
    ("OwlBank/Slowest invocations and dependency calls",
     ['event in ["invocation_end", "dependency_call"]', "sort duration_ms desc", "limit 50"]),
    ("OwlBank/Auth failures",
     ["twilio_auth_failed", "verification_failed", "rate_limited", "verification_risk",
      "otp_sent_untrusted_ani", "by bin(5m), event, component, reason"]),
    ("OwlBank/Escalations by reason",
     ['event = "escalation"', "stats count(*) as escalations by reason, source"]),
])
def test_query_text(synth_default, name, needles):
    q = next(r["Properties"]["QueryString"] for _, _, r in synth_default.resources("AWS::Logs::QueryDefinition")
             if r["Properties"]["Name"] == name)
    for n in needles:
        assert n in q
    # Structure: starts with fields, each stage is a pipe.
    assert q.startswith("fields ")
    assert all(line.startswith("| ") for line in q.splitlines()[1:])


def test_existing_metric_filters_untouched(synth_default):
    patterns = {r["Properties"]["FilterPattern"] for _, _, r in synth_default.resources("AWS::Logs::MetricFilter")}
    assert '"twilio_auth_failed"' in patterns
    assert '"otp_sent_untrusted_ani"' in patterns


# ---- logLevel validation ---------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("DEBUG", "DEBUG"), ("INFO", "INFO"), ("WARN", "WARN"), ("ERROR", "ERROR"),
    ("debug", "DEBUG"), ("Info", "INFO"), (" warn ", "WARN"), ("WARNING", "WARN"), ("warning", "WARN"),
    ("error", "ERROR"),
])
def test_log_level_accepted(raw, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert normalize_log_level(raw) == expected


@pytest.mark.parametrize("raw", [None, ""])
def test_log_level_unset_is_info_without_warning(raw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert normalize_log_level(raw) == "INFO"


@pytest.mark.parametrize("raw", ["verbose", "TRACE", "FATAL", "  ", 7, True, ["DEBUG"]])
def test_invalid_log_level_falls_back_to_info_with_warning(raw):
    with pytest.warns(UserWarning, match="logLevel"):
        assert normalize_log_level(raw) == "INFO"


def test_from_context_log_level_and_helpers():
    base = Path(".")
    assert from_context({}, base_dir=base).log_level == "INFO"
    cfg = from_context({"logLevel": "warning"}, base_dir=base)
    assert (cfg.log_level, cfg.python_log_level) == ("WARN", "WARNING")
    assert (cfg.lambda_application_log_level, cfg.lambda_system_log_level) == ("WARN", "WARN")
    cfg = from_context({"logLevel": "error"}, base_dir=base)
    assert (cfg.python_log_level, cfg.lambda_application_log_level, cfg.lambda_system_log_level) == (
        "ERROR", "ERROR", "WARN")
    with pytest.warns(UserWarning):
        assert from_context({"logLevel": "loud"}, base_dir=base).log_level == "INFO"


def test_every_level_maps_to_real_lambda_enums():
    import owlbank_infra.node_bootstrap  # noqa: F401
    from aws_cdk import aws_lambda as lambda_
    for level in LOG_LEVELS:
        cfg = from_context({"logLevel": level}, base_dir=Path("."))
        assert hasattr(lambda_.ApplicationLogLevel, cfg.lambda_application_log_level)
        assert hasattr(lambda_.SystemLogLevel, cfg.lambda_system_log_level)
