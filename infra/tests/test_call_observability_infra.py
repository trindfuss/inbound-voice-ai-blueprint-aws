"""Call observability infrastructure, switch on.

Template assertions over offline synths with ``callObservability=on``: dev with ``traceContent=off``
(the CloudFront mode), dev with ``traceContent=redacted``, prod (custom domain) with ``off``, and
dev with ``tracesEnableTransactionSearch=true``. The switch-off side is the existing snapshot tests
 plus the "none of this exists" checks below over ``synth_any``.

No property-based tests: these are fixed template assertions over a small example matrix of
contexts, with no generated input space ("PBT: none").
"""

from __future__ import annotations

import json
import warnings

import pytest
from conftest import FAKE_CERT, _synth
from test_synth_and_nag import _nag_report

from owlbank_infra.app_builder import build_app
from owlbank_infra.call_observability import (
    TRACE_READER_LOG_ACTIONS,
    TRACE_READER_QUERY_ACTIONS,
    TRACES_KEY_ALIAS,
    TRACES_LOG_GROUP_NAME,
    TRACES_STREAM_NAME,
    XRAY_WRITE_ACTIONS,
)
from owlbank_infra.config import ConfigError
from owlbank_infra.grants import BRIDGE_STAR_REASON

ACCOUNT = "111111111111"
REGION = "us-east-1"
TRACES_ARN = f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:{TRACES_LOG_GROUP_NAME}"
XRAY_SOURCE = f"arn:aws:xray:{REGION}:{ACCOUNT}:*"

CONTEXTS = {
    "on-dev": {"callObservability": "on"},
    "on-redacted": {"callObservability": "on", "traceContent": "redacted"},
    "on-prod": {"callObservability": "on", "stage": "prod", "domainName": "owlbank.example.com",
                "certificateArn": FAKE_CERT, "bridgeDesiredCount": 2},
    "on-txsearch": {"callObservability": "on", "tracesEnableTransactionSearch": "true"},
}
#: The bridge variables the switch adds (5.12 allows no other OTel variable).
BRIDGE_ON_KEYS = {"OWLBANK_CALL_OBSERVABILITY", "OWLBANK_STAGE", "OWLBANK_TRACE_SAMPLE_RATE",
                  "OWLBANK_TRACE_CONTENT", "OWLBANK_TRACES_LOG_GROUP",
                  "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"}
FRONTDOOR_ON_KEYS = {"OWLBANK_CALL_OBSERVABILITY", "OWLBANK_STAGE", "OWLBANK_TRACE_SAMPLE_RATE"}


def _quiet(factory, name, ctx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # traceContent != off warns
        return _synth(factory, name, ctx)


@pytest.fixture(scope="module")
def obs(tmp_path_factory) -> dict:
    """One synth per on-context, built once for the module."""
    return {name: _quiet(tmp_path_factory, f"callobs-{name}", ctx) for name, ctx in CONTEXTS.items()}


@pytest.fixture(params=sorted(CONTEXTS))
def on(request, obs):
    return obs[request.param]


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


def _one(synth, stack: str, rtype: str, pred=lambda lid, res: True):
    found = [(lid, res) for lid, res in synth.templates[stack]["Resources"].items()
             if res["Type"] == rtype and pred(lid, res)]
    assert len(found) == 1, (stack, rtype, [lid for lid, _ in found])
    return found[0]


def _traces_group(synth):
    return _one(synth, "OwlBankBridge", "AWS::Logs::LogGroup",
                lambda _l, r: r["Properties"].get("LogGroupName") == TRACES_LOG_GROUP_NAME)


def _bridge_env(synth) -> list[tuple[str, str]]:
    _, td = _one(synth, "OwlBankBridge", "AWS::ECS::TaskDefinition")
    (container,) = td["Properties"]["ContainerDefinitions"]
    return [(e["Name"], e["Value"]) for e in container["Environment"]]


def _frontdoor(synth):
    return _one(synth, "OwlBankCompute", "AWS::Lambda::Function",
                lambda lid, _r: lid.startswith("FrontdoorFn"))


def _statements(synth, stack: str, role_prefix: str) -> list[dict]:
    """Every identity-policy statement attached to the role whose logical ID starts with
    ``role_prefix`` (exactly one such role)."""
    resources = synth.templates[stack]["Resources"]
    (role_lid,) = [lid for lid, r in resources.items()
                   if r["Type"] == "AWS::IAM::Role" and lid.startswith(role_prefix)]
    out = []
    for res in resources.values():
        if res["Type"] == "AWS::IAM::Policy" and {"Ref": role_lid} in res["Properties"].get("Roles", []):
            st = res["Properties"]["PolicyDocument"]["Statement"]
            out.extend(st if isinstance(st, list) else [st])
    return out


def _as_list(v):
    return v if isinstance(v, list) else [v]


# --------------------------------------------------------------------------- Bridge: TracesKey
#
# TracesKey lives in OwlBankBridge with the traces group and TraceReader: an export
# from OwlBankData would block a redeploy with callObservability=off (Data deploys first and
# CloudFormation won't delete an export another stack still imports).

def _is_traces_key(lid: str, _r=None) -> bool:
    return lid.startswith("TracesKey") and not lid.startswith("TracesKeyAlias")


def _walk(value):
    """Every dict nested in a JSON value (the value itself included)."""
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)


def _traces_key(synth):
    return _one(synth, "OwlBankBridge", "AWS::KMS::Key", _is_traces_key)


def test_traces_key(on):
    lid, key = _traces_key(on)
    assert key["Properties"]["EnableKeyRotation"] is True
    expected = "Retain" if on.cfg.is_prod else "Delete"
    assert key["DeletionPolicy"] == expected
    _, alias = _one(on, "OwlBankBridge", "AWS::KMS::Alias",
                    lambda _l, r: r["Properties"]["AliasName"] == TRACES_KEY_ALIAS)
    assert alias["Properties"]["TargetKeyId"] == {"Fn::GetAtt": [lid, "Arn"]}
    statements = key["Properties"]["KeyPolicy"]["Statement"]
    logs_st = [s for s in statements
               if s.get("Principal", {}).get("Service") == f"logs.{REGION}.amazonaws.com"]
    assert len(logs_st) == 1
    cond = logs_st[0]["Condition"]
    assert list(cond) == ["ArnEquals"]
    assert _flat(cond["ArnEquals"]["kms:EncryptionContext:aws:logs:arn"]) == TRACES_ARN
    # No statement opens the key to anyone ('*' principal).
    for st in statements:
        assert st.get("Principal") != "*" and st.get("Principal", {}).get("AWS") != "*", st


def test_data_has_no_traces_key_or_export(on):
    data = on.templates["OwlBankData"]
    assert not [lid for lid, r in data["Resources"].items()
                if r["Type"] in ("AWS::KMS::Key", "AWS::KMS::Alias") and lid.startswith("TracesKey")]
    assert not [r for r in data["Resources"].values()
                if r["Type"] == "AWS::KMS::Alias" and r["Properties"]["AliasName"] == TRACES_KEY_ALIAS]
    assert not [name for name, out in data.get("Outputs", {}).items()
                if "TracesKey" in name or "TracesKey" in json.dumps(out)]


def test_bridge_imports_no_traces_key(on):
    imports = [d["Fn::ImportValue"] for d in _walk(on.templates["OwlBankBridge"]) if "Fn::ImportValue" in d]
    assert not [i for i in imports if "TracesKey" in json.dumps(i)], imports


# --------------------------------------------------------------------------- Bridge: traces group

def test_traces_log_group_uses_traces_key(on):
    key_lid, _ = _traces_key(on)
    _, group = _traces_group(on)
    assert group["Properties"]["KmsKeyId"] == {"Fn::GetAtt": [key_lid, "Arn"]}


@pytest.mark.parametrize(("name", "days", "policy"), [
    ("on-dev", 30, "Delete"),        # traceContent=off: the stage's retention (dev: 1 month)
    ("on-txsearch", 30, "Delete"),
    ("on-redacted", 7, "Delete"),    # traceContent != off: traceContentRetentionDays (default 7)
    ("on-prod", 365, "Retain"),      # off at prod: the stage's 1 year, retained
])
def test_traces_log_group_retention(obs, name, days, policy):
    _, group = _traces_group(obs[name])
    assert group["Properties"]["RetentionInDays"] == days
    assert group["DeletionPolicy"] == policy


def test_traces_retention_setting_is_used(tmp_path_factory):
    synth = _quiet(tmp_path_factory, "callobs-retention-14",
                   {"callObservability": "on", "traceContent": "redacted", "traceContentRetentionDays": 14})
    _, group = _traces_group(synth)
    assert group["Properties"]["RetentionInDays"] == 14


def test_traces_retention_not_a_cloudwatch_value_fails_synth(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ConfigError, match="traceContentRetentionDays"):
            build_app({"adminConsole": "false", "callObservability": "on", "traceContent": "redacted",
                       "traceContentRetentionDays": 2}, outdir=str(tmp_path))


def test_spans_stream(on):
    group_lid, _ = _traces_group(on)
    _, stream = _one(on, "OwlBankBridge", "AWS::Logs::LogStream")
    assert stream["Properties"] == {"LogGroupName": TRACES_LOG_GROUP_NAME,
                                    "LogStreamName": TRACES_STREAM_NAME}
    assert group_lid in stream["DependsOn"]


# --------------------------------------------------------------------------- resource policies

def _policy_doc(res) -> dict:
    return json.loads(_flat(res["Properties"]["PolicyDocument"]))


def test_traces_resource_policy(on):
    _, res = _one(on, "OwlBankBridge", "AWS::Logs::ResourcePolicy")
    (st,) = _policy_doc(res)["Statement"]
    assert st["Effect"] == "Allow"
    assert st["Principal"] == {"Service": "xray.amazonaws.com"}
    assert _as_list(st["Action"]) == ["logs:PutLogEvents"]
    assert _as_list(st["Resource"]) == [f"{TRACES_ARN}:*"]
    assert st["Condition"] == {"StringEquals": {"aws:SourceAccount": ACCOUNT},
                               "ArnLike": {"aws:SourceArn": XRAY_SOURCE}}


def _principals(st) -> list:
    p = st.get("Principal")
    if p is None:
        return []
    if p == "*":
        return ["*"]
    return [v for vals in p.values() for v in _as_list(vals)]


def test_no_star_principal_in_any_logs_resource_policy(on):
    policies = list(on.resources("AWS::Logs::ResourcePolicy"))
    assert policies
    for stack, lid, res in policies:
        for st in _policy_doc(res)["Statement"]:
            assert "*" not in _principals(st), (stack, lid)
            assert "NotPrincipal" not in st, (stack, lid)


def test_transaction_search_only_with_flag(obs):
    for name, synth in obs.items():
        configs = list(synth.resources("AWS::XRay::TransactionSearchConfig"))
        data_policies = [r for _s, _l, r in synth.resources("AWS::Logs::ResourcePolicy") if _s == "OwlBankData"]
        if name == "on-txsearch":
            assert len(configs) == 1 and len(data_policies) == 1, name
        else:
            assert not configs and not data_policies, name


def test_transaction_search_policy(obs):
    synth = obs["on-txsearch"]
    plid, res = _one(synth, "OwlBankData", "AWS::Logs::ResourcePolicy")
    (st,) = _policy_doc(res)["Statement"]
    assert st["Principal"] == {"Service": "xray.amazonaws.com"}
    assert _as_list(st["Action"]) == ["logs:PutLogEvents"]
    assert sorted(_as_list(st["Resource"])) == sorted([
        f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:aws/spans:*",
        f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:/aws/application-signals/data:*"])
    assert st["Condition"] == {"StringEquals": {"aws:SourceAccount": ACCOUNT},
                               "ArnLike": {"aws:SourceArn": XRAY_SOURCE}}
    _, config = _one(synth, "OwlBankData", "AWS::XRay::TransactionSearchConfig")
    assert plid in config["DependsOn"]


# --------------------------------------------------------------------------- TraceReader

def test_trace_reader_role(on):
    lid, role = _one(on, "OwlBankBridge", "AWS::IAM::Role",
                     lambda _l, r: r["Properties"].get("RoleName") == f"owlbank-{on.cfg.stage}-trace-reader")
    (trust,) = role["Properties"]["AssumeRolePolicyDocument"]["Statement"]
    assert _flat(trust["Principal"]["AWS"]) == f"arn:aws:iam::{ACCOUNT}:root"
    assert not role["Properties"].get("ManagedPolicyArns")
    sts = _statements(on, "OwlBankBridge", lid)
    by_sid = {s["Sid"]: s for s in sts}
    assert set(by_sid) == {"ReadTracesGroup", "TracesQueryResults", "DecryptTraces"}
    assert by_sid["ReadTracesGroup"]["Action"] == list(TRACE_READER_LOG_ACTIONS)
    assert _flat(by_sid["ReadTracesGroup"]["Resource"]) == f"{TRACES_ARN}:*"
    assert by_sid["TracesQueryResults"]["Action"] == list(TRACE_READER_QUERY_ACTIONS)
    assert by_sid["TracesQueryResults"]["Resource"] == "*"
    assert by_sid["DecryptTraces"]["Action"] == "kms:Decrypt"
    key_lid, _ = _traces_key(on)
    assert by_sid["DecryptTraces"]["Resource"] == {"Fn::GetAtt": [key_lid, "Arn"]}


# --------------------------------------------------------------------------- task role

def test_task_role_xray_grant_and_ack(on):
    sts = _statements(on, "OwlBankBridge", "TaskRole")
    xray = [s for s in sts if s.get("Sid") == "CallObservabilityXRay"]
    assert len(xray) == 1
    assert _as_list(xray[0]["Action"]) == list(XRAY_WRITE_ACTIONS)
    assert xray[0]["Resource"] == "*"
    # No other statement grants an X-Ray action.
    assert not [s for s in sts if s is not xray[0] and any(a.startswith("xray:") for a in _as_list(s["Action"]))]
    reasons = [r for path, rule, r in on.acks
               if path.endswith("/TaskRole") and rule == "AwsSolutions-IAM5[Resource::*]"]
    assert len(reasons) == 1
    for action in XRAY_WRITE_ACTIONS:
        assert action in reasons[0]


def test_off_task_role_has_no_xray(synth_any):
    sts = _statements(synth_any, "OwlBankBridge", "TaskRole")
    assert not [s for s in sts if any(a.startswith("xray:") for a in _as_list(s["Action"]))]
    reasons = [r for path, rule, r in synth_any.acks
               if path.endswith("/TaskRole") and rule == "AwsSolutions-IAM5[Resource::*]"]
    assert reasons == [BRIDGE_STAR_REASON]


# --------------------------------------------------------------------------- Env

def test_bridge_env(on, synth_default):
    env = _bridge_env(on)
    names = [n for n, _ in env]
    values = dict(env)
    assert len(names) == len(set(names))
    expected = set(BRIDGE_ON_KEYS)
    if on.cfg.trace_content == "off":
        expected.add("OTEL_SEMCONV_STABILITY_OPT_IN")
    off_names = {n for n, _ in _bridge_env(synth_default)}
    assert set(names) - off_names == expected
    assert values["OWLBANK_CALL_OBSERVABILITY"] == "on"
    assert values["OWLBANK_STAGE"] == on.cfg.stage
    assert float(values["OWLBANK_TRACE_SAMPLE_RATE"]) == on.cfg.trace_sample_rate
    assert values["OWLBANK_TRACE_CONTENT"] == on.cfg.trace_content
    assert values["OWLBANK_TRACES_LOG_GROUP"] == TRACES_LOG_GROUP_NAME
    assert values["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] == "false"
    if on.cfg.trace_content == "off":
        assert values["OTEL_SEMCONV_STABILITY_OPT_IN"] == "gen_ai_unredacted_attributes="
    assert not [n for n in names if n.startswith("OTEL_EXPORTER_")]


def test_frontdoor_env_and_layer(on):
    _, fd = _frontdoor(on)
    env = fd["Properties"]["Environment"]["Variables"]
    for k in FRONTDOOR_ON_KEYS:
        assert k in env, k
    assert env["OWLBANK_CALL_OBSERVABILITY"] == "on"
    assert env["OWLBANK_STAGE"] == on.cfg.stage
    assert not [k for k in env if k.startswith("OTEL_")]
    layer_lid, layer = _one(on, "OwlBankCompute", "AWS::Lambda::LayerVersion",
                            lambda _l, r: "OpenTelemetry" in r["Properties"].get("Description", ""))
    assert layer["Properties"]["CompatibleArchitectures"] == ["arm64"]
    assert {"Ref": layer_lid} in fd["Properties"]["Layers"]
    for lid, res in on.templates["OwlBankCompute"]["Resources"].items():
        if res["Type"] == "AWS::Lambda::Function" and not lid.startswith("FrontdoorFn"):
            assert {"Ref": layer_lid} not in res["Properties"].get("Layers", []), lid


def test_frontdoor_role_gets_nothing_new(on, synth_default):
    """Spans go to the daemon over UDP: the X-Ray grant active tracing already makes covers
    it, so the front-door role's statements are the same as with the switch off."""
    def actions(s):
        return sorted(a for st in _statements(s, "OwlBankCompute", "FrontdoorRole")
                      for a in _as_list(st["Action"]))
    if on.cfg.stage == "dev" and not on.domain_mode:
        assert actions(on) == actions(synth_default)
    assert "xray:PutTraceSegments" in actions(on)


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

def test_off_creates_none_of_it(synth_any):
    assert not list(synth_any.resources("AWS::Logs::LogStream"))
    assert not list(synth_any.resources("AWS::Logs::ResourcePolicy"))
    assert not list(synth_any.resources("AWS::XRay::TransactionSearchConfig"))
    for _s, _l, res in synth_any.resources("AWS::Logs::LogGroup"):
        assert res["Properties"].get("LogGroupName") != TRACES_LOG_GROUP_NAME
    for _s, _l, res in synth_any.resources("AWS::KMS::Alias"):
        assert res["Properties"]["AliasName"] != TRACES_KEY_ALIAS
    assert not [(s, lid) for s, lid, _r in synth_any.resources("AWS::KMS::Key") if _is_traces_key(lid)]
    for _s, _l, res in synth_any.resources("AWS::IAM::Role"):
        assert not str(res["Properties"].get("RoleName", "")).endswith("-trace-reader")
    for _s, _l, res in synth_any.resources("AWS::Lambda::LayerVersion"):
        assert "OpenTelemetry" not in res["Properties"].get("Description", "")
    names = {n for n, _ in _bridge_env(synth_any)}
    assert not {n for n in names if n.startswith(("OTEL_", "OWLBANK_TRACE", "OWLBANK_CALL_OBS"))}
    assert "OWLBANK_STAGE" not in names
    _, fd = _frontdoor(synth_any)
    fd_env = fd["Properties"]["Environment"]["Variables"]
    assert not {k for k in fd_env if k.startswith(("OTEL_", "OWLBANK_TRACE", "OWLBANK_CALL_OBS"))}


# --------------------------------------------------------------------------- 3.9 taskrouter route

def _api_paths(synth) -> set[str]:
    return {r["Properties"]["PathPart"] for r in synth.templates["OwlBankApi"]["Resources"].values()
            if r["Type"] == "AWS::ApiGateway::Resource"}


def _waf_regexes(synth) -> str:
    return json.dumps([r for r in synth.templates["OwlBankApi"]["Resources"].values()
                       if r["Type"] in ("AWS::WAFv2::RegexPatternSet", "AWS::WAFv2::WebACL")])


@pytest.fixture(scope="module")
def human_leg_off(tmp_path_factory) -> dict:
    """``humanLegConversation=off``, with call observability off and on: the route's
    other user, so this module's 3.9 check still sees call observability's own effect."""
    tw = tmp_path_factory.mktemp("callobs-tw") / "twilio.context.json"
    tw.write_text(json.dumps({"workflowSid": "WW" + "2" * 32, "workspaceSid": "WS" + "4" * 32}))
    return {mode: _quiet(tmp_path_factory, f"callobs-hl-off-{mode}",
                         {"humanLegConversation": "off", "callObservability": mode, "twilioOutputs": str(tw)})
            for mode in ("off", "on")}


def test_taskrouter_route_only_when_on(human_leg_off, on) -> None:
    """3.9: TaskRouter's workspace events reach ``/voice/taskrouter`` (API resource and WAF
    allow-list) with the switch on. With ``humanLegConversation=off`` (its other user) the route exists only with call observability on."""
    assert "taskrouter" not in _api_paths(human_leg_off["off"])
    assert "taskrouter" not in _waf_regexes(human_leg_off["off"])
    assert "taskrouter" in _api_paths(human_leg_off["on"])
    assert "taskrouter" in _waf_regexes(human_leg_off["on"])
    # No Flex (the fixture has no workspace and workflow SIDs): no TaskRouter events, so no route.
    assert "taskrouter" not in _api_paths(on)
    assert "taskrouter" not in _waf_regexes(on)
