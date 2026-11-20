"""Model-assisted slots: the mode ceiling.

``slotModeCeiling`` (``deterministic`` | ``assisted`` | ``generative``, default ``assisted``) reaches
``OWLBANK_SLOT_MODE_CEILING`` on the front door and on ``admin_api``; any other value fails synth.
With ``deterministic`` the front-door role has no ``bedrock:InvokeModel`` statement (and no Bedrock
nag acknowledgement); with the other two it has exactly the inference profile's ARNs. A metric
filter on the front door's log group counts ``slot_mode_invalid`` and alarms at 1 per 5 minutes.

The three values are checked exhaustively by synth; the Hypothesis test covers the parser's
"anything else is refused" over arbitrary strings (config only, no synth).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import _synth
from hypothesis import given, settings
from hypothesis import strategies as st
from test_synth_and_nag import _nag_report

from owlbank_infra.app_builder import build_app
from owlbank_infra.config import DEFAULT_SLOT_MODE_CEILING, SLOT_MODES, ConfigError, from_context
from owlbank_infra.observability import SLOT_MODE_INVALID_EVENT

BASE = Path(__file__).resolve().parents[1]
CEILING_ENV = "OWLBANK_SLOT_MODE_CEILING"
PROFILE = "arn:aws:bedrock:us-east-1:111111111111:inference-profile/us.anthropic.claude-haiku-5-5"
VIA_PROFILE = "arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-5-5"


@pytest.fixture(scope="module", params=SLOT_MODES)
def ceiling_synth(request, tmp_path_factory):
    """Each ceiling, with the admin console on so ``admin_api`` is synthesized too."""
    synth = _synth(tmp_path_factory, f"ceiling-{request.param}",
                   {"slotModeCeiling": request.param, "adminConsole": "true"})
    return request.param, synth


# ---------------------------------------------------------------------------------- helpers

def _flat(value):
    """A CFN string value as text: ``Fn::Join`` joined, ``AWS::Partition`` as ``aws``."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "Fn::Join" in value:
        sep, parts = value["Fn::Join"]
        return sep.join(_flat(p) for p in parts)
    if value == {"Ref": "AWS::Partition"}:
        return "aws"
    return repr(value)


def _fn_env(synth, stack: str, prefix: str) -> dict:
    res = synth.templates[stack]["Resources"]
    fns = [r for lid, r in res.items() if r["Type"] == "AWS::Lambda::Function" and lid.startswith(prefix)]
    assert len(fns) == 1, (stack, prefix, len(fns))
    return fns[0]["Properties"]["Environment"]["Variables"]


def _frontdoor_role_statements(synth) -> list[dict]:
    res = synth.templates["OwlBankCompute"]["Resources"]
    role_ids = [lid for lid, r in res.items() if r["Type"] == "AWS::IAM::Role"
                and r["Properties"].get("RoleName") == synth.cfg.frontdoor_role_name]
    assert len(role_ids) == 1, role_ids
    out = []
    for r in res.values():
        if r["Type"] == "AWS::IAM::Policy" and {"Ref": role_ids[0]} in r["Properties"].get("Roles", []):
            out.extend(r["Properties"]["PolicyDocument"]["Statement"])
    for r in res.values():  # inline policies on the role itself, if any
        if r["Type"] == "AWS::IAM::Role" and r["Properties"].get("RoleName") == synth.cfg.frontdoor_role_name:
            for p in r["Properties"].get("Policies", []):
                out.extend(p["PolicyDocument"]["Statement"])
    return out


def _actions(statement: dict) -> list[str]:
    a = statement.get("Action", [])
    return [a] if isinstance(a, str) else list(a)


def _bedrock_invoke(statements: list[dict]) -> list[dict]:
    return [s for s in statements if any(a.startswith("bedrock:Invoke") or a in ("bedrock:*", "*")
                                         for a in _actions(s))]


# ---------------------------------------------------------------------------------- config

def test_default_ceiling_is_assisted():
    assert DEFAULT_SLOT_MODE_CEILING == "assisted"
    assert from_context({}, base_dir=BASE).slot_mode_ceiling == "assisted"


@pytest.mark.parametrize("mode", SLOT_MODES)
def test_each_mode_is_accepted(mode):
    assert from_context({"slotModeCeiling": mode}, base_dir=BASE).slot_mode_ceiling == mode


def test_synth_refuses_a_bogus_ceiling(tmp_path):
    with pytest.raises(ConfigError, match="slotModeCeiling"):
        build_app({"slotModeCeiling": "bogus"}, outdir=str(tmp_path))


def test_a_non_string_ceiling_is_refused():
    with pytest.raises(ConfigError, match="slotModeCeiling"):
        from_context({"slotModeCeiling": 1}, base_dir=BASE)


@settings(max_examples=200, deadline=None)
@given(st.text(max_size=20))
def test_any_other_value_is_refused(raw):
    """1.3: anything that isn't one of the three values (blank means the default) fails."""
    if not raw.strip():
        assert from_context({"slotModeCeiling": raw}, base_dir=BASE).slot_mode_ceiling == "assisted"
    elif raw.strip().lower() in SLOT_MODES:
        assert from_context({"slotModeCeiling": raw}, base_dir=BASE).slot_mode_ceiling == raw.strip().lower()
    else:
        with pytest.raises(ConfigError, match="slotModeCeiling"):
            from_context({"slotModeCeiling": raw}, base_dir=BASE)


# ---------------------------------------------------------------------------------- env

def test_ceiling_reaches_front_door_and_admin_api(ceiling_synth):
    mode, synth = ceiling_synth
    assert _fn_env(synth, "OwlBankCompute", "FrontdoorFn")[CEILING_ENV] == mode
    assert _fn_env(synth, "OwlBankAdmin", "ApiFn")[CEILING_ENV] == mode


def test_default_synth_carries_assisted(synth_default):
    assert _fn_env(synth_default, "OwlBankCompute", "FrontdoorFn")[CEILING_ENV] == "assisted"


def test_only_front_door_and_admin_api_get_the_ceiling(ceiling_synth):
    _, synth = ceiling_synth
    for stack, lid, r in synth.resources("AWS::Lambda::Function"):
        env = r["Properties"].get("Environment", {}).get("Variables", {})
        if (stack, lid.startswith("FrontdoorFn")) == ("OwlBankCompute", True) or \
                (stack, lid.startswith("ApiFn")) == ("OwlBankAdmin", True):
            continue
        assert CEILING_ENV not in env, (stack, lid)


# ---------------------------------------------------------------------------------- IAM

def test_front_door_bedrock_statement_follows_the_ceiling(ceiling_synth):
    mode, synth = ceiling_synth
    invoke = _bedrock_invoke(_frontdoor_role_statements(synth))
    if mode == "deterministic":
        assert invoke == [], invoke
        return
    assert {s["Sid"] for s in invoke} == {"SlotResolverProfile", "SlotResolverViaProfile"}
    for s in invoke:
        assert _actions(s) == ["bedrock:InvokeModel"], s
    by_sid = {s["Sid"]: s for s in invoke}
    assert _flat(by_sid["SlotResolverProfile"]["Resource"]) == PROFILE
    via = by_sid["SlotResolverViaProfile"]
    assert _flat(via["Resource"]) == VIA_PROFILE
    assert _flat(via["Condition"]["StringEquals"]["bedrock:InferenceProfileArn"]) == PROFILE


def test_briefing_follows_the_ceiling_and_adds_no_iam(ceiling_synth):
    """The escalation briefing (no Flex) reuses the resolver's grant: on iff a model may be involved,
    so ``deterministic`` has no briefing and still no Bedrock statement. The budget env is set
    either way and the Lambda timeout leaves room for it."""
    mode, synth = ceiling_synth
    assert synth.cfg.briefing_on == (mode != "deterministic")
    assert not _bedrock_invoke(_frontdoor_role_statements(synth)) or mode != "deterministic"
    env = _fn_env(synth, "OwlBankCompute", "FrontdoorFn")
    budget_ms = int(env["OWLBANK_BRIEFING_BUDGET_MS"])
    res = synth.templates["OwlBankCompute"]["Resources"]
    (timeout,) = [r["Properties"]["Timeout"] for lid, r in res.items()
                  if r["Type"] == "AWS::Lambda::Function" and lid.startswith("FrontdoorFn")]
    assert timeout >= 10 and 0 < budget_ms <= 5000


def test_briefing_is_off_in_flex_mode(synth_flex):
    """Flex mode hands the caller to an agent: no briefing, even with a model-capable ceiling."""
    assert synth_flex.cfg.flex_mode and not synth_flex.cfg.briefing_on


def test_bedrock_nag_ack_only_with_the_statement(ceiling_synth):
    mode, synth = ceiling_synth
    fd_bedrock_acks = [(p, r) for p, r, _ in synth.acks
                       if p.startswith("OwlBankCompute/FrontdoorRole") and "bedrock" in r]
    if mode == "deterministic":
        assert fd_bedrock_acks == []
    else:
        assert len(fd_bedrock_acks) == 1, fd_bedrock_acks


def test_no_slot_classifier_sid_left(ceiling_synth):
    _, synth = ceiling_synth
    for s in _frontdoor_role_statements(synth):
        assert not s.get("Sid", "").startswith("SlotClassifier"), s


# ---------------------------------------------------------------------------------- alarm

def test_slot_mode_invalid_filter_and_alarm(ceiling_synth):
    _, synth = ceiling_synth
    res = synth.templates["OwlBankCompute"]["Resources"]
    fd_group = next(lid for lid, r in res.items() if r["Type"] == "AWS::Logs::LogGroup"
                    and r["Properties"].get("LogGroupName") == "/owlbank/lambda/frontdoor")
    filters = [r["Properties"] for r in res.values() if r["Type"] == "AWS::Logs::MetricFilter"
               and SLOT_MODE_INVALID_EVENT in r["Properties"]["FilterPattern"]]
    assert len(filters) == 1, filters
    mf = filters[0]
    assert mf["LogGroupName"] == {"Ref": fd_group}
    assert mf["FilterPattern"] == f'"{SLOT_MODE_INVALID_EVENT}"'
    (transform,) = mf["MetricTransformations"]
    alarms = [r["Properties"] for r in res.values() if r["Type"] == "AWS::CloudWatch::Alarm"
              and r["Properties"].get("MetricName") == transform["MetricName"]
              and r["Properties"].get("Namespace") == transform["MetricNamespace"]]
    assert len(alarms) == 1, alarms
    alarm = alarms[0]
    assert alarm["Threshold"] == 1
    assert alarm["Period"] == 300
    assert alarm["Statistic"] == "Sum"
    assert alarm["EvaluationPeriods"] == 1
    assert alarm["ComparisonOperator"] == "GreaterThanOrEqualToThreshold"
    assert alarm["TreatMissingData"] == "notBreaching"
    # The same alarm topic as the front door's other alarms (the Data stack's, imported).
    auth = next(r["Properties"] for lid, r in res.items() if r["Type"] == "AWS::CloudWatch::Alarm"
                and lid.startswith("FrontdoorAuthRejectedAlarm"))
    assert len(alarm["AlarmActions"]) == 1
    assert alarm["AlarmActions"] == auth["AlarmActions"]
    assert "AlarmTopic" in alarm["AlarmActions"][0]["Fn::ImportValue"]


# ---------------------------------------------------------------------------------- cdk-nag

def test_cdk_nag_clean(ceiling_synth):
    _, synth = ceiling_synth
    report = _nag_report(synth)
    if report is not None:
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [(v["ruleName"], v.get("violatingResources")) for v in errors]
        assert report["conclusion"] == "success"
    for path, rule, reason in synth.acks:
        assert rule.startswith("AwsSolutions-") and len(reason.strip()) >= 40, (path, rule)
