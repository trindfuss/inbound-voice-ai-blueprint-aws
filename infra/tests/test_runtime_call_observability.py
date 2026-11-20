"""Call observability under the AgentCore hosting option.

``bridgeHosting=agentcore`` x {``vpc``, ``public``} x {dev, prod} with ``callObservability=on``,
plus dev ``vpc`` with ``traceContent=redacted`` for the retention rule. The switch-off side is the
session fixtures ``synth_agentcore_{vpc,public}_{dev,prod}`` (the same contexts without the switch),
which the agentcore snapshot and assertion tests already pin; here they are checked for having
none of it, and the on templates are checked against them for what changes.

With the AgentCore hosting the bridge exports to the same traces log group as on Fargate
(``/owlbank/traces/bridge`` with the TracesKey, ``spans`` stream, X-Ray resource policy and
``TraceReader``); the Runtime's execution role gets the X-Ray write grant; the Runtime sets
``UNIFIED_TRACES_DESTINATION_ENABLED=true``; the Runtime's own log group is unchanged.

No property-based tests: fixed template assertions over a small example matrix of contexts, with
no generated input space ("PBT: none").
"""

from __future__ import annotations

import json

import pytest
from test_bridge_agentcore import _ctx, _quiet_synth
from test_synth_and_nag import _nag_report

from owlbank_infra.call_observability import (
    TRACE_READER_LOG_ACTIONS,
    TRACES_LOG_GROUP_NAME,
    TRACES_STREAM_NAME,
    XRAY_OTLP_REASON,
    XRAY_WRITE_ACTIONS,
)
from owlbank_infra.runtime import STAR_REASON, UNIFIED_TRACES_ENV
from owlbank_infra.runtime import star_reason as runtime_star_reason

BRIDGE = "OwlBankBridge"
ACCOUNT, REGION = "111111111111", "us-east-1"
TRACES_ARN = f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:{TRACES_LOG_GROUP_NAME}"
XRAY_SOURCE = f"arn:aws:xray:{REGION}:{ACCOUNT}:*"
RUNTIME = "AWS::BedrockAgentCore::Runtime"
STAR_RULE = "AwsSolutions-IAM5[Resource::*]"
MATRIX = [("vpc", "dev"), ("public", "dev"), ("vpc", "prod"), ("public", "prod")]
#: What the switch adds to the Runtime's environment (5.12 allows no other OTel variable).
ON_KEYS = {"OWLBANK_CALL_OBSERVABILITY", "OWLBANK_STAGE", "OWLBANK_TRACE_SAMPLE_RATE",
           "OWLBANK_TRACE_CONTENT", "OWLBANK_TRACES_LOG_GROUP",
           "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", UNIFIED_TRACES_ENV}
#: Resources only the switch adds to the Bridge template (with the traces group itself), TracesKey
#: and its alias included.
ADDED_TYPES = {"AWS::Logs::LogStream", "AWS::Logs::ResourcePolicy", "AWS::KMS::Key", "AWS::KMS::Alias"}


def _on_ctx(mode: str, stage: str, **extra) -> dict:
    return {**_ctx(mode, stage), "callObservability": "on", **extra}


@pytest.fixture(scope="module")
def ac_on(tmp_path_factory) -> dict:
    synths = {f"{m}-{s}": _quiet_synth(tmp_path_factory, f"acobs-{m}-{s}", _on_ctx(m, s)) for m, s in MATRIX}
    synths["vpc-dev-redacted"] = _quiet_synth(tmp_path_factory, "acobs-redacted",
                                              _on_ctx("vpc", "dev", traceContent="redacted"))
    return synths


@pytest.fixture(params=[f"{m}-{s}" for m, s in MATRIX])
def on(request, ac_on):
    return ac_on[request.param]


@pytest.fixture
def off(on, request):
    """The switch-off synth with the same mode and stage as ``on``."""
    mode, stage = on.stacks["compute"].cfg.agentcore_network_mode, on.stacks["compute"].cfg.stage
    return request.getfixturevalue(f"synth_agentcore_{mode}_{stage}")


@pytest.fixture(params=[f"{m}-{s}" for m, s in MATRIX])
def off_any(request):
    mode, stage = request.param.split("-")
    return request.getfixturevalue(f"synth_agentcore_{mode}_{stage}")


# --------------------------------------------------------------------------- helpers

def _flat(value) -> str:
    """A CloudFormation string value with ``Fn::Join`` resolved and the partition as ``aws``."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("Ref") == "AWS::Partition":
            return "aws"
        if "Fn::Join" in value:
            sep, parts = value["Fn::Join"]
            return sep.join(_flat(p) for p in parts)
        return json.dumps(value)
    raise TypeError(value)


def _as_list(v):
    return v if isinstance(v, list) else [v]


def _res(synth, rtype=None) -> dict:
    return {lid: r for lid, r in synth.templates[BRIDGE]["Resources"].items()
            if rtype is None or r["Type"] == rtype}


def _one(synth, rtype, pred=lambda lid, res: True):
    found = [(lid, r) for lid, r in _res(synth, rtype).items() if pred(lid, r)]
    assert len(found) == 1, (rtype, [lid for lid, _ in found])
    return found[0]


def _cfg(synth):
    return synth.stacks["compute"].cfg


def _runtime_env(synth) -> dict:
    _, rt = _one(synth, RUNTIME)
    return rt["Properties"]["EnvironmentVariables"]


def _traces_group(synth):
    return _one(synth, "AWS::Logs::LogGroup",
                lambda _l, r: r["Properties"].get("LogGroupName") == TRACES_LOG_GROUP_NAME)


def _runtime_group(synth):
    return _one(synth, "AWS::Logs::LogGroup", lambda lid, _r: lid.startswith("RuntimeLogs"))


def _statements(synth, role_lid) -> list[dict]:
    out = []
    for r in _res(synth, "AWS::IAM::Policy").values():
        if {"Ref": role_lid} in r["Properties"].get("Roles", []):
            out.extend(_as_list(r["Properties"]["PolicyDocument"]["Statement"]))
    return out


def _exec_role_lid(synth) -> str:
    lid, _ = _one(synth, "AWS::IAM::Role",
                  lambda _l, r: r["Properties"].get("RoleName") == _cfg(synth).bridge_task_role_name)
    return lid


def _star_reasons(synth) -> list[str]:
    return [r for path, rule, r in synth.acks if path.endswith(f"{BRIDGE}/TaskRole") and rule == STAR_RULE]


# --------------------------------------------------------------------------- Runtime env

def test_runtime_env(on, off):
    env, off_env = _runtime_env(on), _runtime_env(off)
    expected = set(ON_KEYS)
    if _cfg(on).trace_content == "off":
        expected.add("OTEL_SEMCONV_STABILITY_OPT_IN")
    assert set(env) - set(off_env) == expected
    # Everything the Runtime had before is still there, with the same values.
    assert {k: env[k] for k in off_env} == off_env
    assert env[UNIFIED_TRACES_ENV] == "true"
    assert env["OWLBANK_TRACES_LOG_GROUP"] == TRACES_LOG_GROUP_NAME
    assert env["OWLBANK_CALL_OBSERVABILITY"] == "on"
    assert env["OWLBANK_STAGE"] == _cfg(on).stage
    assert env["OWLBANK_TRACE_CONTENT"] == _cfg(on).trace_content
    assert env["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] == "false"
    assert not [k for k in env if k.startswith("OTEL_EXPORTER_")]


def test_redacted_env_has_no_source_redaction(ac_on):
    env = _runtime_env(ac_on["vpc-dev-redacted"])
    assert env["OWLBANK_TRACE_CONTENT"] == "redacted"
    assert "OTEL_SEMCONV_STABILITY_OPT_IN" not in env
    assert env[UNIFIED_TRACES_ENV] == "true"


# --------------------------------------------------------------------------- traces group

def _traces_key(synth):
    return _one(synth, "AWS::KMS::Key",
                lambda lid, _r: lid.startswith("TracesKey") and not lid.startswith("TracesKeyAlias"))


def test_traces_key_in_bridge(on):
    """TracesKey lives in OwlBankBridge with the group: rotation on, retained at prod,
    CloudWatch Logs scoped to the traces group, and OwlBankData has none of it."""
    lid, key = _traces_key(on)
    assert key["Properties"]["EnableKeyRotation"] is True
    assert key["DeletionPolicy"] == ("Retain" if _cfg(on).is_prod else "Delete")
    (logs_st,) = [s for s in key["Properties"]["KeyPolicy"]["Statement"]
                  if s.get("Principal", {}).get("Service") == f"logs.{REGION}.amazonaws.com"]
    cond = logs_st["Condition"]["ArnEquals"]["kms:EncryptionContext:aws:logs:arn"]
    assert TRACES_LOG_GROUP_NAME in json.dumps(cond)
    _, alias = _one(on, "AWS::KMS::Alias", lambda _l, r: r["Properties"]["AliasName"] == "alias/owlbank-traces")
    assert alias["Properties"]["TargetKeyId"] == {"Fn::GetAtt": [lid, "Arn"]}
    data = on.templates["OwlBankData"]
    assert not [k for k, r in data["Resources"].items() if r["Type"] == "AWS::KMS::Key" and k.startswith("TracesKey")]
    assert "TracesKey" not in json.dumps(data.get("Outputs", {}))
    assert not [v for v in _imports(on) if "TracesKey" in json.dumps(v)]


def _imports(synth) -> list:
    out, stack = [], [synth.templates[BRIDGE]]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            if "Fn::ImportValue" in v:
                out.append(v["Fn::ImportValue"])
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
    return out


def test_traces_group_on_the_traces_key(on):
    key_lid, _ = _traces_key(on)
    _, group = _traces_group(on)
    assert group["Properties"]["KmsKeyId"] == {"Fn::GetAtt": [key_lid, "Arn"]}
    assert not [v for v in _imports(on) if "TracesKey" in json.dumps(v)]


@pytest.mark.parametrize(("name", "days", "policy"), [
    ("vpc-dev", 30, "Delete"),            # traceContent=off: the stage's retention
    ("public-dev", 30, "Delete"),
    ("vpc-dev-redacted", 7, "Delete"),    # traceContent != off: traceContentRetentionDays (7)
    ("vpc-prod", 365, "Retain"),
    ("public-prod", 365, "Retain"),
])
def test_traces_group_retention(ac_on, name, days, policy):
    _, group = _traces_group(ac_on[name])
    assert group["Properties"]["RetentionInDays"] == days
    assert group["DeletionPolicy"] == policy


def test_spans_stream(on):
    group_lid, _ = _traces_group(on)
    _, stream = _one(on, "AWS::Logs::LogStream")
    assert stream["Properties"] == {"LogGroupName": TRACES_LOG_GROUP_NAME, "LogStreamName": TRACES_STREAM_NAME}
    assert group_lid in stream["DependsOn"]


def test_xray_resource_policy(on):
    group_lid, _ = _traces_group(on)
    _, res = _one(on, "AWS::Logs::ResourcePolicy")
    assert group_lid in res["DependsOn"]
    (st,) = json.loads(_flat(res["Properties"]["PolicyDocument"]))["Statement"]
    assert st["Effect"] == "Allow"
    assert st["Principal"] == {"Service": "xray.amazonaws.com"}  # never "*"
    assert _as_list(st["Action"]) == ["logs:PutLogEvents"]
    assert _as_list(st["Resource"]) == [f"{TRACES_ARN}:*"]
    assert st["Condition"] == {"StringEquals": {"aws:SourceAccount": ACCOUNT},
                               "ArnLike": {"aws:SourceArn": XRAY_SOURCE}}


def test_trace_reader_role(on):
    lid, _ = _one(on, "AWS::IAM::Role",
                  lambda _l, r: r["Properties"].get("RoleName") == f"owlbank-{_cfg(on).stage}-trace-reader")
    by_sid = {s["Sid"]: s for s in _statements(on, lid)}
    assert set(by_sid) == {"ReadTracesGroup", "TracesQueryResults", "DecryptTraces"}
    assert by_sid["ReadTracesGroup"]["Action"] == list(TRACE_READER_LOG_ACTIONS)
    assert _flat(by_sid["ReadTracesGroup"]["Resource"]) == f"{TRACES_ARN}:*"
    key_lid, _ = _traces_key(on)
    assert by_sid["DecryptTraces"]["Action"] == "kms:Decrypt"
    assert by_sid["DecryptTraces"]["Resource"] == {"Fn::GetAtt": [key_lid, "Arn"]}


# --------------------------------------------------------------------------- execution role

def test_execution_role_xray_grant_and_ack(on, off):
    sts = _statements(on, _exec_role_lid(on))
    xray = [s for s in sts if s.get("Sid") == "CallObservabilityXRay"]
    assert len(xray) == 1
    assert _as_list(xray[0]["Action"]) == list(XRAY_WRITE_ACTIONS)
    assert xray[0]["Resource"] == "*"
    assert not [s for s in sts if s is not xray[0] and any(a.startswith("xray:") for a in _as_list(s["Action"]))]
    # 6.4 as reworded: the stack creates the resource policy; the role can't.
    assert not [s for s in sts if "logs:PutResourcePolicy" in _as_list(s["Action"])]
    # Only the X-Ray statement is new on the role.
    off_sids = {s.get("Sid") for s in _statements(off, _exec_role_lid(off))}
    assert {s.get("Sid") for s in sts} - off_sids == {"CallObservabilityXRay"}
    # One IAM5 ack, naming every X-Ray action, plus the Runtime's existing reasons.
    (reason,) = _star_reasons(on)
    assert reason == runtime_star_reason(_cfg(on))
    assert XRAY_OTLP_REASON in reason and reason != STAR_REASON
    assert reason.startswith(STAR_REASON.removesuffix(" Nothing else uses '*'."))
    assert reason.endswith(" Nothing else uses '*'.")
    for action in XRAY_WRITE_ACTIONS:
        assert action in reason


# --------------------------------------------------------------------------- Runtime log group

def test_runtime_log_group_unchanged(on, off):
    """Its name, key, retention and removal are as with the switch off (6.4: it keeps its logs)."""
    on_lid, on_group = _runtime_group(on)
    off_lid, off_group = _runtime_group(off)
    assert on_lid == off_lid
    assert on_group == off_group
    # And its write policy.
    lid = next(lid for lid in _res(off, "AWS::IAM::Policy") if lid.startswith("RuntimeLogsPolicy"))
    assert _res(on)[lid] == _res(off)[lid]


def test_only_traces_resources_are_added(on, off):
    on_res, off_res = _res(on), _res(off)
    added = set(on_res) - set(off_res)
    assert not set(off_res) - set(on_res)
    types = {on_res[lid]["Type"] for lid in added}
    # The metric filters, anomaly and ceiling alarms and the call timeline query.
    alarms = {"AWS::Logs::MetricFilter", "AWS::CloudWatch::Alarm", "AWS::CloudWatch::AnomalyDetector",
              "AWS::Logs::QueryDefinition"}
    assert types == ADDED_TYPES | alarms | {"AWS::Logs::LogGroup", "AWS::IAM::Role", "AWS::IAM::Policy"}
    group_lid, _ = _traces_group(on)
    assert group_lid in added


# --------------------------------------------------------------------------- cdk-nag

def test_cdk_nag_clean(on):
    report = _nag_report(on)
    if report is not None:
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [(v["ruleName"], v.get("violatingResources")) for v in errors]
        assert report["conclusion"] == "success"


def test_one_ack_per_construct_and_rule(on):
    assert len({(p, r) for p, r, _ in on.acks}) == len(on.acks)


# --------------------------------------------------------------------------- off: none of it

def test_off_has_none_of_it(off_any):
    env = _runtime_env(off_any)
    assert UNIFIED_TRACES_ENV not in env
    assert not {k for k in env if k.startswith(("OTEL_", "OWLBANK_TRACE", "OWLBANK_CALL_OBS"))}
    assert "OWLBANK_STAGE" not in env
    assert not _res(off_any, "AWS::Logs::LogStream")
    assert not _res(off_any, "AWS::Logs::ResourcePolicy")
    for stack in off_any.templates.values():
        assert not [lid for lid, r in stack.get("Resources", {}).items()
                    if r["Type"] == "AWS::KMS::Key" and lid.startswith("TracesKey")]
    for r in _res(off_any, "AWS::Logs::LogGroup").values():
        assert r["Properties"].get("LogGroupName") != TRACES_LOG_GROUP_NAME
    for r in _res(off_any, "AWS::IAM::Role").values():
        assert not str(r["Properties"].get("RoleName", "")).endswith("-trace-reader")
    sts = _statements(off_any, _exec_role_lid(off_any))
    assert not [s for s in sts if any(a.startswith("xray:") for a in _as_list(s["Action"]))]
    assert _star_reasons(off_any) == [STAR_REASON]
