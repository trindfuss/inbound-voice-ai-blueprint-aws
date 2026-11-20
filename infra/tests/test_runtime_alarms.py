"""AgentCore hosting: alarms, metric filters and saved queries for ``bridgeHosting=agentcore``.

The matrix is ``test_bridge_agentcore``'s: ``agentcore`` x {``vpc``, ``public``} x {dev, prod}.

* 4.4: the ``twilio_auth_failed`` reject filter is on the Runtime's log group, with the Fargate
  filter's pattern, metric and alarm threshold (any reason: signature, nonce, binding).
* 8.3: an ERROR/CRITICAL filter on the Runtime's log group; ``AWS/Bedrock-AgentCore`` ``Throttles``
  >= 1 and ``SystemErrors`` >= 5 per 5 minutes on this Runtime's DEFAULT endpoint, and the
  account's ``ActiveSessionCount`` (``Service=AgentCore.Runtime``) >= 80 % of
  ``agentcoreSessionQuota``. Every alarm notifies the alarm topic. No ECS, ALB, CloudFront or WAF
  alarm exists in this mode.
* 8.2: the five saved queries cover the six Lambdas, the warmer and the Runtime's log group (not the
  dormant ECS group), and every group they name exists in the app.
* Fargate: no AgentCore metric anywhere (the snapshot tests hold the rest unchanged).

No PBT (fixed alarm set; assertions). cdk-nag runs on the same synths in
``test_bridge_agentcore.test_cdk_nag_clean``.
"""

from __future__ import annotations

import json

import pytest
from test_bridge_agentcore import _ctx, _quiet_synth

from owlbank_infra import observability as obs

BRIDGE, COMPUTE = "OwlBankBridge", "OwlBankCompute"
RUNTIME = "AWS::BedrockAgentCore::Runtime"
ALARM, FILTER, QUERY = "AWS::CloudWatch::Alarm", "AWS::Logs::MetricFilter", "AWS::Logs::QueryDefinition"


@pytest.fixture(scope="module", params=[("vpc", "dev"), ("public", "dev"), ("vpc", "prod"), ("public", "prod")],
                ids=lambda p: f"{p[0]}-{p[1]}")
def ac(request, tmp_path_factory):
    mode, stage = request.param
    synth = _quiet_synth(tmp_path_factory, f"ra-{mode}-{stage}", _ctx(mode, stage))
    synth.mode, synth.stage = mode, stage  # type: ignore[attr-defined]
    return synth


@pytest.fixture(scope="module")
def ac_quota(tmp_path_factory):
    return _quiet_synth(tmp_path_factory, "ra-quota", {**_ctx("vpc", "dev"), "agentcoreSessionQuota": 1234})


@pytest.fixture
def fargate(ac, request):
    return request.getfixturevalue("synth_prod" if ac.stage == "prod" else "synth_default")


# --------------------------------------------------------------------------- helpers

def _res(synth, stack, rtype):
    return {lid: r for lid, r in synth.templates[stack]["Resources"].items() if r["Type"] == rtype}


def _runtime_lid(synth):
    (lid,) = _res(synth, BRIDGE, RUNTIME)
    return lid


def _runtime_log_group_lid(synth):
    (lid,) = [lid for lid, r in _res(synth, BRIDGE, "AWS::Logs::LogGroup").items()
              if obs.RUNTIME_LOG_GROUP_PREFIX in json.dumps(r["Properties"]["LogGroupName"])]
    return lid


def _filter(synth, stack, metric_name):
    (mf,) = [r["Properties"] for r in _res(synth, stack, FILTER).values()
             if r["Properties"]["MetricTransformations"][0]["MetricName"] == metric_name]
    return mf


def _alarm(synth, stack, metric_name):
    (a,) = [r["Properties"] for r in _res(synth, stack, ALARM).values()
            if r["Properties"].get("MetricName") == metric_name]
    return a


def _topic_actions(synth):
    """The Fargate bridge's alarm actions (one reference to the Data stack's alarm topic)."""
    (actions,) = {json.dumps(r["Properties"]["AlarmActions"]) for r in _res(synth, BRIDGE, ALARM).values()}
    return json.loads(actions)


# --------------------------------------------------------------------------- log filters

def test_reject_filter_on_the_runtime_log_group_with_the_fargate_threshold(ac, fargate):
    mf = _filter(ac, BRIDGE, obs.BRIDGE_REJECT_METRIC)
    assert mf["LogGroupName"] == {"Ref": _runtime_log_group_lid(ac)}
    want = _filter(fargate, BRIDGE, obs.BRIDGE_REJECT_METRIC)
    assert mf["FilterPattern"] == want["FilterPattern"] == f'"{obs.AUTH_FAILED_EVENT}"'
    assert mf["MetricTransformations"] == want["MetricTransformations"]
    alarm, fg_alarm = _alarm(ac, BRIDGE, obs.BRIDGE_REJECT_METRIC), _alarm(fargate, BRIDGE, obs.BRIDGE_REJECT_METRIC)
    for k in ("Namespace", "Threshold", "Period", "Statistic", "EvaluationPeriods", "ComparisonOperator"):
        assert alarm[k] == fg_alarm[k], k
    assert alarm["Threshold"] == 10


def test_error_filter_on_the_runtime_log_group(ac):
    mf = _filter(ac, BRIDGE, "RuntimeAppErrors")
    assert mf["LogGroupName"] == {"Ref": _runtime_log_group_lid(ac)}
    for level in obs.ERROR_LEVELS:
        assert f'$.level = "{level}"' in mf["FilterPattern"]
    assert mf["MetricTransformations"][0]["MetricNamespace"] == f"OwlBank/{ac.stage}/Security"
    alarm = _alarm(ac, BRIDGE, "RuntimeAppErrors")
    assert (alarm["Threshold"], alarm["Period"], alarm["Statistic"], alarm["EvaluationPeriods"]) == (3, 300, "Sum", 1)
    assert alarm["ComparisonOperator"] == "GreaterThanOrEqualToThreshold"


# --------------------------------------------------------------------------- AgentCore metrics

@pytest.mark.parametrize("metric, threshold", [("Throttles", 1), ("SystemErrors", 5)])
def test_runtime_metric_alarm(ac, metric, threshold):
    alarm = _alarm(ac, BRIDGE, metric)
    assert alarm["Namespace"] == "AWS/Bedrock-AgentCore"
    dims = {d["Name"]: d["Value"] for d in alarm["Dimensions"]}
    assert dims == {"Operation": "InvokeAgentRuntime",
                    "Name": f"{ac.stacks['compute'].cfg.runtime_name}::DEFAULT",
                    "Resource": {"Fn::GetAtt": [_runtime_lid(ac), "AgentRuntimeArn"]}}
    assert (alarm["Threshold"], alarm["Period"], alarm["Statistic"], alarm["EvaluationPeriods"]) == (
        threshold, 300, "Sum", 1)
    assert alarm["ComparisonOperator"] == "GreaterThanOrEqualToThreshold"
    assert alarm["TreatMissingData"] == "notBreaching"


def test_active_sessions_at_80_percent_of_the_default_quota(ac):
    alarm = _alarm(ac, BRIDGE, "ActiveSessionCount")
    assert alarm["Namespace"] == "AWS/Bedrock-AgentCore"
    assert alarm["Dimensions"] == [{"Name": "Service", "Value": "AgentCore.Runtime"}]
    assert ac.stacks["compute"].cfg.agentcore_session_quota == 5000
    assert (alarm["Threshold"], alarm["Statistic"], alarm["Period"]) == (4000, "Maximum", 300)
    assert alarm["ComparisonOperator"] == "GreaterThanOrEqualToThreshold"


def test_active_sessions_follow_the_configured_quota(ac_quota):
    assert _alarm(ac_quota, BRIDGE, "ActiveSessionCount")["Threshold"] == 988  # ceil(0.8 * 1234)


@pytest.mark.parametrize("quota, expected", [(1, 1), (2, 2), (5, 4), (1234, 988), (5000, 4000)])
def test_active_session_threshold(quota, expected):
    assert obs.active_session_threshold(quota) == expected


def test_exactly_these_alarms_and_all_notify_the_topic(ac, fargate):
    alarms = [r["Properties"] for r in _res(ac, BRIDGE, ALARM).values()]
    # Model failover on (the default list): its three bridge alarms (test_model_failover_infra).
    failover = (["ActiveRank", "ModelUnavailable", "PreferredModelUnavailable"]
                if ac.cfg.failover_on else [])
    assert sorted(a["MetricName"] for a in alarms) == sorted(
        ["ActiveSessionCount", obs.BRIDGE_REJECT_METRIC, "RuntimeAppErrors", "SystemErrors", "Throttles",
         *failover])
    topic = _topic_actions(fargate)
    assert all(a["AlarmActions"] == topic for a in alarms)
    # No ECS, ALB, CloudFront or WAF alarm in this mode.
    assert {a["Namespace"] for a in alarms} == {"AWS/Bedrock-AgentCore", f"OwlBank/{ac.stage}/Security",
                                                *([f"OwlBank/{ac.stage}/Model"] if failover else [])}


def test_fargate_has_no_agentcore_metric(synth_default, synth_prod):
    for synth in (synth_default, synth_prod):
        for stack in synth.templates.values():
            assert "AWS/Bedrock-AgentCore" not in json.dumps(stack)


# --------------------------------------------------------------------------- saved queries

def test_saved_queries_cover_lambdas_warmer_and_runtime(ac):
    defs = [r["Properties"] for r in _res(ac, BRIDGE, QUERY).values()]
    assert sorted(d["Name"] for d in defs) == sorted(q.name for q in obs.SAVED_QUERIES)
    stacks = {name for name, t in ac.templates.items() if _res(ac, name, QUERY)}
    assert stacks == {BRIDGE}
    rt_lid = _runtime_lid(ac)
    runtime_group = {"Fn::Join": ["", [obs.RUNTIME_LOG_GROUP_PREFIX,
                                       {"Fn::GetAtt": [rt_lid, "AgentRuntimeId"]}, "-DEFAULT"]]}
    for d in defs:
        names = d["LogGroupNames"]
        assert names[:-1] == [*obs.LAMBDA_LOG_GROUP_NAMES, obs.RUNTIME_WARMER_LOG_GROUP_NAME]
        assert names[-1] == runtime_group
        assert obs.BRIDGE_LOG_GROUP_NAME not in names


def test_every_query_group_exists_in_the_app(ac):
    defined = {json.dumps(r["Properties"]["LogGroupName"])
               for t in ac.templates for r in _res(ac, t, "AWS::Logs::LogGroup").values()}
    for d in _res(ac, BRIDGE, QUERY).values():
        for name in d["Properties"]["LogGroupNames"]:
            assert json.dumps(name) in defined, name
    assert json.dumps(obs.RUNTIME_WARMER_LOG_GROUP_NAME) in {
        json.dumps(r["Properties"]["LogGroupName"]) for r in _res(ac, COMPUTE, "AWS::Logs::LogGroup").values()}


def test_trace_query_is_the_same_text_on_both_hostings(ac, fargate):
    name = f"{obs.QUERY_NAME_PREFIX}Trace a call by call SID"

    def trace(synth):
        (q,) = [r["Properties"] for r in _res(synth, BRIDGE, QUERY).values() if r["Properties"]["Name"] == name]
        return q["QueryString"], q.get("QueryParameters")

    assert trace(ac) == trace(fargate)
