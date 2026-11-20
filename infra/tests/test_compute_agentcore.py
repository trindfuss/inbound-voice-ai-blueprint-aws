"""AgentCore hosting: ``OwlBankCompute`` with ``bridgeHosting=agentcore``, and the Bridge→Compute relay
parameter on both hostings.

The matrix is ``test_bridge_agentcore``'s: ``agentcore`` x {``vpc``, ``public``} x {dev (CloudFront
mode), prod with a domain}. Each Compute template is compared with the Fargate synth of the same stage.

* Front door: ``OWLBANK_HOSTING=agentcore``, the presign role ARN, the warmer, a 300 s URL TTL
  and the relay parameter's name; no ``OWLBANK_RELAY_WS_URL``, no Runtime ARN in the template and no
  nonce-carrier setting. Its statements are the Fargate front door's plus exactly three:
  ``sts:AssumeRole`` on the presign role only, ``lambda:InvokeFunction`` on the warmer only and
  ``StopRuntimeSession`` on {R, E}.
* ``runtime_warmer``: 256 MB, 30 s, async retries 0; its role holds only its own logs, X-Ray,
  the relay parameter and ``InvokeAgentRuntime`` on {R, E}. R and E name the Runtime by its
  fixed name prefix and DEFAULT endpoint, and the prefix is the Runtime's ``AgentRuntimeName``.
* The ``runtime_prewarm_failed`` throttle/quota alarm, threshold 1 in 5 minutes.
* One SSM String parameter with the same logical ID and name in both hostings (Fargate: the relay
  host; agentcore: the Runtime ARN); ``ssm:GetParameter`` on that ARN only; no ``Fn::ImportValue``
  between Bridge and Compute and no export from Bridge in either hosting, so a switch never has to
  remove an export another stack imports. ``OwlBankBridgeImage`` is in both hostings.
* cdk-nag clean.

No PBT: IaC, checked by assertions. The front door's and warmer's parameter reading
is tested in ``tests/unit/test_relay_param.py`` and ``tests/property/test_relay_param_props.py``.
"""

from __future__ import annotations

import json
import sys

import pytest
from conftest import REPO
from test_bridge_agentcore import _ctx, _quiet_synth
from test_synth_and_nag import _nag_report

from owlbank_infra import compute_stack, grants, runtime
from owlbank_infra.bridge_stack import RELAY_PARAMETER_ID
from owlbank_infra.nag import ACKS

COMPUTE, BRIDGE, IMAGE = "OwlBankCompute", "OwlBankBridge", "OwlBankBridgeImage"
ACCOUNT, REGION = "111111111111", "us-east-1"
ADDED_FD_SIDS = {"AssumePresignRole", "InvokeRuntimeWarmer", "StopRuntimeSession"}
WARMER_SIDS = {"WriteOwnLogs", "ReadRelayParameter", "InvokeRuntime"}


@pytest.fixture(scope="module", params=[("vpc", "dev"), ("public", "dev"), ("vpc", "prod"), ("public", "prod")],
                ids=lambda p: f"{p[0]}-{p[1]}")
def ac(request, tmp_path_factory):
    mode, stage = request.param
    synth = _quiet_synth(tmp_path_factory, f"cac-{mode}-{stage}", _ctx(mode, stage))
    synth.mode, synth.stage = mode, stage  # type: ignore[attr-defined]
    return synth


@pytest.fixture
def fargate(ac, request):
    return request.getfixturevalue("synth_prod" if ac.stage == "prod" else "synth_default")


@pytest.fixture(params=["fargate-dev", "fargate-prod", "agentcore"])
def any_hosting(request, ac):
    if request.param == "agentcore":
        return ac
    return request.getfixturevalue("synth_prod" if request.param == "fargate-prod" else "synth_default")


# --------------------------------------------------------------------------- helpers

def _res(synth, stack, rtype=None):
    return {lid: r for lid, r in synth.templates[stack]["Resources"].items()
            if rtype is None or r["Type"] == rtype}


def _fn(synth, prefix):
    (item,) = [(lid, r) for lid, r in _res(synth, COMPUTE, "AWS::Lambda::Function").items()
               if lid.startswith(prefix)]
    return item


def _env(synth, prefix):
    return _fn(synth, prefix)[1]["Properties"]["Environment"]["Variables"]


def _role_lid(synth, fn_prefix):
    return _fn(synth, fn_prefix)[1]["Properties"]["Role"]["Fn::GetAtt"][0]


def _statements(synth, role_lid):
    return [s for p in _res(synth, COMPUTE, "AWS::IAM::Policy").values()
            if {"Ref": role_lid} in p["Properties"].get("Roles", [])
            for s in p["Properties"]["PolicyDocument"]["Statement"]]


def _by_sid(stmts):
    out = {s["Sid"]: s for s in stmts}
    assert len(out) == len(stmts), "duplicate or missing Sid"
    return out


def _arn(tail):
    return {"Fn::Join": ["", ["arn:", {"Ref": "AWS::Partition"}, tail]]}


def _r_and_e(synth):
    name = synth.stacks["compute"].cfg.runtime_name
    r = f":bedrock-agentcore:{REGION}:{ACCOUNT}:runtime/{name}-*"
    return [_arn(r), _arn(r + "/runtime-endpoint/DEFAULT")]


def _parameter(synth):
    found = _res(synth, BRIDGE, "AWS::SSM::Parameter")
    assert len(found) == 1, sorted(found)
    return next(iter(found.items()))


# --------------------------------------------------------------------------- Front door

def test_front_door_environment(ac):
    env = _env(ac, "FrontdoorFn")
    cfg = ac.stacks["compute"].cfg
    assert env["OWLBANK_HOSTING"] == "agentcore"
    assert env["OWLBANK_PRESIGN_ROLE_ARN"] == _arn(f":iam::{ACCOUNT}:role/{cfg.presign_role_name}")
    warmer_lid, _ = _fn(ac, "RuntimeWarmerFn")
    assert env["OWLBANK_WARMER_FUNCTION"] == {"Fn::GetAtt": [warmer_lid, "Arn"]}
    assert env["OWLBANK_URL_TTL_S"] == "300"
    assert env["OWLBANK_RELAY_PARAMETER"] == cfg.relay_parameter_name == f"/owlbank/{ac.stage}/bridge/relay"
    for absent in ("OWLBANK_RELAY_WS_URL", "OWLBANK_RUNTIME_ARN", "OWLBANK_NONCE_CARRIER"):
        assert absent not in env, absent


def test_front_door_env_is_fargates_plus_the_hosting_settings(ac, fargate):
    added = set(_env(ac, "FrontdoorFn")) - set(_env(fargate, "FrontdoorFn"))
    assert added == {"OWLBANK_HOSTING", "OWLBANK_PRESIGN_ROLE_ARN", "OWLBANK_WARMER_FUNCTION",
                     "OWLBANK_URL_TTL_S"}
    assert set(_env(fargate, "FrontdoorFn")) <= set(_env(ac, "FrontdoorFn"))


def test_front_door_grants_are_fargates_plus_three(ac, fargate):
    """AssumeRole on the presign role only, InvokeFunction on the warmer only, StopRuntimeSession
    on {R, E}; every other statement exactly as on Fargate."""
    stmts = _statements(ac, _role_lid(ac, "FrontdoorFn"))
    added = [s for s in stmts if s.get("Sid") in ADDED_FD_SIDS]
    rest = [s for s in stmts if s.get("Sid") not in ADDED_FD_SIDS]
    theirs = _statements(fargate, _role_lid(fargate, "FrontdoorFn"))
    assert sorted(json.dumps(s, sort_keys=True) for s in rest) == sorted(
        json.dumps(s, sort_keys=True) for s in theirs)
    mine = _by_sid(added)
    assert set(mine) == ADDED_FD_SIDS
    cfg = ac.stacks["compute"].cfg
    assume = mine["AssumePresignRole"]
    assert (assume["Action"], assume["Resource"]) == (
        "sts:AssumeRole", _arn(f":iam::{ACCOUNT}:role/{cfg.presign_role_name}"))
    warmer_lid, _ = _fn(ac, "RuntimeWarmerFn")
    inv = mine["InvokeRuntimeWarmer"]
    assert (inv["Action"], inv["Resource"]) == ("lambda:InvokeFunction", {"Fn::GetAtt": [warmer_lid, "Arn"]})
    stop = mine["StopRuntimeSession"]
    assert (stop["Action"], stop["Resource"]) == ("bedrock-agentcore:StopRuntimeSession", _r_and_e(ac))
    agentcore_actions = {a for s in stmts for a in ([s["Action"]] if isinstance(s["Action"], str)
                                                    else s["Action"]) if a.startswith("bedrock-agentcore:")}
    assert agentcore_actions == {"bedrock-agentcore:StopRuntimeSession"}


# --------------------------------------------------------------------------- runtime_warmer

def test_warmer_function_settings(ac):
    lid, fn = _fn(ac, "RuntimeWarmerFn")
    props = fn["Properties"]
    assert props["Handler"] == "lambdas.runtime_warmer.handler.handler"
    assert (props["MemorySize"], props["Timeout"]) == (256, 30)
    assert props["Environment"]["Variables"]["OWLBANK_RELAY_PARAMETER"] == ac.stacks["compute"].cfg.relay_parameter_name
    assert "OWLBANK_RUNTIME_ARN" not in props["Environment"]["Variables"]
    assert not any(k.startswith("OWLBANK_SECRET_") for k in props["Environment"]["Variables"])
    cfgs = [r["Properties"] for r in _res(ac, COMPUTE, "AWS::Lambda::EventInvokeConfig").values()
            if r["Properties"]["FunctionName"] == {"Ref": lid}]
    assert len(cfgs) == 1 and cfgs[0]["MaximumRetryAttempts"] == 0
    assert "FunctionUrl" not in json.dumps(ac.templates[COMPUTE])  # no Function URL anywhere


def test_warmer_role_holds_only_invoke_runtime(ac):
    """Its own logs, X-Ray (no resource-level permissions), the relay parameter, and
    InvokeAgentRuntime on {R, E}; no tables, secrets or other AgentCore action."""
    stmts = _statements(ac, _role_lid(ac, "RuntimeWarmerFn"))
    sids = {s.get("Sid") for s in stmts}
    assert sids == WARMER_SIDS | {None}  # the unnamed one is X-Ray's
    (xray,) = [s for s in stmts if "Sid" not in s]
    assert {a.split(":")[0] for a in xray["Action"]} == {"xray"}
    by = {s["Sid"]: s for s in stmts if "Sid" in s}
    assert (by["InvokeRuntime"]["Action"], by["InvokeRuntime"]["Resource"]) == (
        "bedrock-agentcore:InvokeAgentRuntime", _r_and_e(ac))
    assert not any(a.startswith(("dynamodb:", "secretsmanager:", "kms:")) for s in stmts
                   for a in ([s["Action"]] if isinstance(s["Action"], str) else s["Action"]))


def test_runtime_prefix_matches_the_runtime_name(ac):
    """The grants' prefix is the CfnRuntime's AgentRuntimeName (its ID is ``<name>-<suffix>``)."""
    (rt,) = list(_res(ac, BRIDGE, "AWS::BedrockAgentCore::Runtime").values())
    assert rt["Properties"]["AgentRuntimeName"] == ac.stacks["compute"].cfg.runtime_name
    assert runtime.ENDPOINT_SUFFIX == grants.RUNTIME_ENDPOINT_SUFFIX


def test_prewarm_throttle_alarm(ac):
    (mf,) = [r["Properties"] for r in _res(ac, COMPUTE, "AWS::Logs::MetricFilter").values()
             if "runtime_prewarm_failed" in r["Properties"]["FilterPattern"]]
    pattern = mf["FilterPattern"]
    assert '$.event = "runtime_prewarm_failed"' in pattern
    for code in ("ThrottlingException", "ServiceQuotaExceededException"):
        assert f'$.code = "{code}"' in pattern
    warmer_logs = [lid for lid, r in _res(ac, COMPUTE, "AWS::Logs::LogGroup").items()
                   if r["Properties"]["LogGroupName"] == "/owlbank/lambda/runtime_warmer"]
    assert mf["LogGroupName"] == {"Ref": warmer_logs[0]}
    metric = mf["MetricTransformations"][0]
    (alarm,) = [r["Properties"] for r in _res(ac, COMPUTE, "AWS::CloudWatch::Alarm").values()
                if r["Properties"].get("MetricName") == metric["MetricName"]]
    assert alarm["Namespace"] == metric["MetricNamespace"]
    assert (alarm["Threshold"], alarm["Period"], alarm["EvaluationPeriods"], alarm["Statistic"]) == (1, 300, 1, "Sum")
    assert alarm["ComparisonOperator"] == "GreaterThanOrEqualToThreshold" and alarm["AlarmActions"]
    descriptions = {r["Properties"].get("AlarmDescription") for r in _res(ac, COMPUTE, "AWS::CloudWatch::Alarm").values()}
    assert {"runtime_warmer Lambda errors", "runtime_warmer Lambda throttles"} <= descriptions


def test_fargate_has_no_warmer(synth_default, synth_prod):
    for synth in (synth_default, synth_prod):
        assert not [lid for lid in _res(synth, COMPUTE) if lid.startswith("RuntimeWarmer")]
        assert "runtime_prewarm_failed" not in json.dumps(synth.templates[COMPUTE])


# --------------------------------------------------------------------------- Both hostings

def test_relay_parameter_in_both_hostings(any_hosting):
    """One String parameter with the same logical ID and fixed name in either hosting, so a switch
    updates its value in place: the relay host on Fargate, the Runtime ARN on agentcore."""
    lid, param = _parameter(any_hosting)
    cfg = any_hosting.stacks["compute"].cfg
    assert lid.startswith(RELAY_PARAMETER_ID)
    props = param["Properties"]
    assert (props["Name"], props["Type"], props["Tier"]) == (cfg.relay_parameter_name, "String", "Standard")
    if cfg.is_agentcore:
        (rt_lid,) = list(_res(any_hosting, BRIDGE, "AWS::BedrockAgentCore::Runtime"))
        assert props["Value"] == {"Fn::GetAtt": [rt_lid, "AgentRuntimeArn"]}
    elif cfg.custom_domain_mode:
        assert props["Value"] == cfg.bridge_host
    else:
        (dist,) = list(_res(any_hosting, BRIDGE, "AWS::CloudFront::Distribution"))
        assert props["Value"] == {"Fn::GetAtt": [dist, "DomainName"]}


def test_same_parameter_logical_id_in_both_hostings(ac, fargate):
    assert _parameter(ac)[0] == _parameter(fargate)[0]


def test_front_door_reads_only_the_relay_parameter(any_hosting):
    cfg = any_hosting.stacks["compute"].cfg
    want = _arn(f":ssm:{REGION}:{ACCOUNT}:parameter{cfg.relay_parameter_name}")
    readers = ["FrontdoorFn"] + (["RuntimeWarmerFn"] if cfg.is_agentcore else [])
    for prefix in readers:
        (s,) = [s for s in _statements(any_hosting, _role_lid(any_hosting, prefix))
                if "ssm:GetParameter" in json.dumps(s["Action"])]
        assert (s["Sid"], s["Action"], s["Resource"]) == ("ReadRelayParameter", "ssm:GetParameter", want)
    ssm_users = [lid for lid, p in _res(any_hosting, COMPUTE, "AWS::IAM::Policy").items()
                 if '"ssm:' in json.dumps(p["Properties"]["PolicyDocument"])]
    assert len(ssm_users) == len(readers)


def _imports(node, out):
    if isinstance(node, dict):
        if "Fn::ImportValue" in node:
            out.append(json.dumps(node["Fn::ImportValue"]))
        for v in node.values():
            _imports(v, out)
    elif isinstance(node, list):
        for v in node:
            _imports(v, out)
    return out


def test_a_hosting_switch_removes_no_export(any_hosting):
    """No Fn::ImportValue of a Bridge export in any stack, and no export from Bridge, in
    either hosting, so switching (or rolling back) never asks CloudFormation to delete an export
    that another stack still imports."""
    bridge_outputs = any_hosting.templates[BRIDGE].get("Outputs", {})
    assert not [n for n, o in bridge_outputs.items() if "Export" in o]
    for name, tpl in any_hosting.templates.items():
        assert not [i for i in _imports(tpl, []) if BRIDGE in i], name
    assert BRIDGE in {d.stack_name for d in any_hosting.stacks["compute"].dependencies}  # order only


def test_image_stack_in_both_hostings(any_hosting):
    """``OwlBankBridgeImage`` is in the app on both hostings."""
    assert IMAGE in any_hosting.templates


def test_env_name_matches_the_lambda_side():
    sys.path.insert(0, str(REPO / "src"))
    try:
        from owlbank import relay_param
    finally:
        sys.path.remove(str(REPO / "src"))
    assert compute_stack.RELAY_PARAMETER_ENV == relay_param.RELAY_PARAMETER_ENV
    assert compute_stack.URL_TTL_S == 300


def test_cdk_nag_clean(ac):
    report = _nag_report(ac)
    if report is not None:
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [(v["ruleName"], v.get("violatingResources")) for v in errors]
        assert report["conclusion"] == "success"
    mine = [(p, r, why) for p, r, why in ACKS if p.startswith(f"{COMPUTE}/") and "bedrock-agentcore" in r]
    assert mine and all(why == grants.RUNTIME_PREFIX_REASON for _, _, why in mine)
