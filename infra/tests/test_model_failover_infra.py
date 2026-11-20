"""Model failover's fleet picture in infra.

The ``modelstats`` table is always in ``OwlBankData`` (retained, deletion-protected at prod like the
other tables). On (the default list of three): its write-budget alarm, the poller in ``OwlBankCompute`` with its schedule, role and alarms, and the bridge's table env,
item grants and alarms in ``OwlBankBridge`` (both hostings). Off, or a list of one: exactly today's
single-model Bridge and Compute templates, and Data plus only the table, compared with
``snapshots/failover-off/`` (captured from the single-model synth before this task's change).

No property test: the switch is a two-valued setting and the templates are compared whole; the IAM
checks enumerate every DynamoDB statement in every role.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import snapshot_util as su
from conftest import _synth
from owlbank_infra import model_failover as mf
from owlbank_infra.bridge_stack import FAILOVER_TEST_ENV, MODELS_ENV
from owlbank_infra.config import ConfigError, InfraConfig, from_context
from test_model_list import H55, _as_list, _fargate_env, _flat, _runtime_env
from test_synth_and_nag import _nag_report

BASE = Path(__file__).resolve().parents[1]
README = BASE / "README.md"
SNAPSHOT = Path(__file__).resolve().parent / "snapshots" / "failover-off"
TODAY_STACKS = ("OwlBankBridge", "OwlBankCompute", "OwlBankData")
DATA, COMPUTE, BRIDGE = "OwlBankData", "OwlBankCompute", "OwlBankBridge"
TABLE_ARN = "arn:aws:dynamodb:us-east-1:111111111111:table/owlbank-dev-modelstats"


def cfg(**ctx: Any) -> InfraConfig:
    return from_context(ctx, base_dir=BASE)


@pytest.fixture(scope="module")
def synth_off(tmp_path_factory):
    return _synth(tmp_path_factory, "failover-off", {"modelFailover": "off"})


@pytest.fixture(scope="module")
def synth_one(tmp_path_factory):
    return _synth(tmp_path_factory, "failover-one", {"modelIds": [H55]})


@pytest.fixture(scope="module")
def synth_test_delay(tmp_path_factory):
    return _synth(tmp_path_factory, "failover-delay", {"failoverTest": "6"})


# ---- helpers


def _resources(synth, stack: str, rtype: str) -> dict[str, dict]:
    return {lid: r for lid, r in synth.templates[stack]["Resources"].items() if r["Type"] == rtype}


def _poller_id(synth) -> str:
    (lid,) = [lid for lid, r in _resources(synth, COMPUTE, "AWS::Lambda::Function").items()
              if r["Properties"]["Handler"] == "lambdas.model_poller.handler.handler"]
    return lid


def _poller(synth) -> dict:
    return synth.templates[COMPUTE]["Resources"][_poller_id(synth)]


def _poller_role_id(synth) -> str:
    return _poller(synth)["Properties"]["Role"]["Fn::GetAtt"][0]


def _statements(synth, stack: str, role_id: str | None = None) -> list[dict]:
    out = []
    for res in _resources(synth, stack, "AWS::IAM::Policy").values():
        roles = [r["Ref"] for r in res["Properties"].get("Roles", [])]
        if role_id is None or role_id in roles:
            out += res["Properties"]["PolicyDocument"]["Statement"]
    return _flat(out)


def _dynamodb_statements(statements: list[dict]) -> list[dict]:
    return [st for st in statements if any(a.startswith("dynamodb:") for a in _as_list(st["Action"]))]


def _modelstats_statements(statements: list[dict]) -> list[dict]:
    return [st for st in _dynamodb_statements(statements)
            if any("modelstats" in str(r) for r in _as_list(st["Resource"]))]


def _leading(st: dict) -> list[str]:
    return st["Condition"]["ForAllValues:StringEquals"]["dynamodb:LeadingKeys"]


def _alarms(synth, stack: str) -> dict[str, dict]:
    return {lid: r["Properties"] for lid, r in _resources(synth, stack, "AWS::CloudWatch::Alarm").items()}


def _alarm(synth, stack: str, prefix: str) -> dict:
    (props,) = [p for lid, p in _alarms(synth, stack).items() if lid.startswith(prefix)]
    return props


def _bridge_role_ids(synth) -> list[str]:
    return [lid for lid, r in _resources(synth, BRIDGE, "AWS::IAM::Role").items()
            if r["Properties"].get("RoleName") == synth.cfg.bridge_task_role_name]


# ---- the switch


def test_switch_defaults_on_and_off_keeps_only_the_first_entry():
    on = cfg()
    assert on.model_failover == "on" and on.failover_on and len(on.models) == 3
    off = cfg(modelFailover="off")
    assert off.model_failover == "off" and not off.failover_on
    assert off.models == on.models[:1]
    assert not cfg(modelIds=[H55]).failover_on
    assert cfg(modelFailover=" ON ").failover_on


@pytest.mark.parametrize("raw", ["yes", "true", "disabled", "1"])
def test_bad_switch_value_is_refused_naming_the_key(raw):
    with pytest.raises(ConfigError, match="modelFailover"):
        cfg(modelFailover=raw)


@pytest.mark.parametrize("ctx", [{"modelFailover": "off"}, {"modelIds": [H55]}])
def test_failover_test_needs_the_feature(ctx):
    with pytest.raises(ConfigError, match="failoverTest needs modelFailover=on"):
        cfg(failoverTest="6", **ctx)


def _without_table(template: dict) -> tuple[dict, dict[str, dict]]:
    """The normalized Data template minus the modelstats table, and the table resources removed."""
    t = json.loads(json.dumps(su.normalize(template)))
    removed = {lid: r for lid, r in t["Resources"].items()
               if r["Type"] == "AWS::DynamoDB::Table" and r["Properties"].get("TableName") == "owlbank-dev-modelstats"}
    for lid in removed:
        del t["Resources"][lid]
    return t, removed


@pytest.mark.parametrize("fixture", ["synth_off", "synth_one"])
def test_off_and_a_list_of_one_synthesize_todays_bridge_and_compute(fixture, request):
    synth = request.getfixturevalue(fixture)
    problems = []
    for name in (BRIDGE, COMPUTE):
        expected = su.load(SNAPSHOT / f"{name}.json")
        problems += [f"{name}: {line}" for line in su.diff(expected, su.normalize(synth.templates[name]))]
    assert not problems, "\n".join(problems[:50])


@pytest.mark.parametrize("fixture", ["synth_off", "synth_one"])
def test_off_and_a_list_of_one_add_only_the_table_to_data(fixture, request):
    synth = request.getfixturevalue(fixture)
    rest, removed = _without_table(synth.templates[DATA])
    assert len(removed) == 1
    problems = su.diff(su.load(SNAPSHOT / f"{DATA}.json"), rest)
    assert not problems, "\n".join(problems[:50])


@pytest.mark.parametrize("fixture", ["synth_off", "synth_one"])
def test_off_has_no_fleet_picture_beyond_the_table(fixture, request):
    templates = request.getfixturevalue(fixture).templates
    rest, _ = _without_table(templates[DATA])
    text = json.dumps([rest, templates[COMPUTE], templates[BRIDGE]])
    for needle in ("modelstats", "model_poller", mf.TABLE_ENV, "OwlBank/dev/Model", mf.UNAVAILABLE_EVENT):
        assert needle not in text, needle
    assert "Modelstats" not in json.dumps(rest)  # no write-budget alarm either


# ---- the table and its write guard


@pytest.fixture(scope="module")
def synth_test_stage(tmp_path_factory):
    return _synth(tmp_path_factory, "failover-stage-test", {"stage": "test"})


@pytest.mark.parametrize("fixture,protected", [("synth_default", False), ("synth_test_stage", False),
                                               ("synth_prod", True)])
def test_table_is_retained_and_protected_at_prod_only(fixture, protected, request):
    """Same settings as the other tables (AWS-M7): RETAIN in every stage, deletion protection at prod."""
    synth = request.getfixturevalue(fixture)
    (res,) = [r for r in _resources(synth, DATA, "AWS::DynamoDB::Table").values()
              if r["Properties"].get("TableName", "").endswith("-modelstats")]
    assert res["DeletionPolicy"] == "Retain" and res["UpdateReplacePolicy"] == "Retain"
    assert res["Properties"].get("DeletionProtectionEnabled", False) is protected


def test_table_properties(synth_default):
    tables = {lid: r["Properties"] for lid, r in _resources(synth_default, DATA, "AWS::DynamoDB::Table").items()
              if r["Properties"].get("TableName") == "owlbank-dev-modelstats"}
    (props,) = tables.values()
    assert props["KeySchema"] == [{"AttributeName": "s", "KeyType": "HASH"},
                                  {"AttributeName": "sk", "KeyType": "RANGE"}]
    assert props["AttributeDefinitions"] == [{"AttributeName": "s", "AttributeType": "S"},
                                             {"AttributeName": "sk", "AttributeType": "S"}]
    assert props["BillingMode"] == "PAY_PER_REQUEST"
    assert props.get("TableClass", "STANDARD") == "STANDARD"
    assert props["TimeToLiveSpecification"] == {"AttributeName": "expires_at", "Enabled": True}
    sse = props["SSESpecification"]
    assert sse["SSEEnabled"] is True and sse["SSEType"] == "KMS"
    assert sse["KMSMasterKeyId"]["Fn::GetAtt"][0].startswith("DataKey")
    for absent in ("StreamSpecification", "GlobalSecondaryIndexes", "LocalSecondaryIndexes",
                   "KinesisStreamSpecification"):
        assert absent not in props, absent
    pitr = props.get("PointInTimeRecoverySpecification", {})
    assert not pitr.get("PointInTimeRecoveryEnabled", False)


def test_write_budget_alarm(synth_default):
    a = _alarm(synth_default, DATA, "ModelstatsWriteBudgetAlarm")
    assert a["Namespace"] == "AWS/DynamoDB" and a["MetricName"] == "ConsumedWriteCapacityUnits"
    assert a["Dimensions"][0]["Name"] == "TableName"
    assert a["Statistic"] == "Sum" and a["Period"] == 300
    assert a["Threshold"] == 2 * 5 * 300
    assert a["ComparisonOperator"] == "GreaterThanThreshold"
    assert a["AlarmActions"]


def test_write_budget_matches_the_fleet_module():
    from owlbank.failover.fleet import WRITE_BUDGET_PER_S
    assert mf.WRITE_BUDGET_PER_S == WRITE_BUDGET_PER_S


def test_table_is_not_in_the_shared_table_env(synth_default):
    """Only the bridge and the poller get the table's name."""
    for res in _resources(synth_default, COMPUTE, "AWS::Lambda::Function").values():
        env = res["Properties"].get("Environment", {}).get("Variables", {})
        is_poller = res["Properties"]["Handler"].startswith("lambdas.model_poller.")
        assert (mf.TABLE_ENV in env) == is_poller, res["Properties"]["Handler"]


# ---- the poller


def test_poller_function_settings(synth_default):
    p = _poller(synth_default)["Properties"]
    assert p["Architectures"] == ["arm64"] and p["MemorySize"] == 128 and p["Runtime"] == "python3.12"
    assert p["Timeout"] >= 30 + 5 * 8 + 8  # two cycles 30 s apart, up to five pings 8 s apart, a deadline
    assert p["Timeout"] <= 900
    env = p["Environment"]["Variables"]
    assert env[mf.TABLE_ENV] == "owlbank-dev-modelstats"
    assert env[mf.STAGE_ENV] == "dev"
    assert env[MODELS_ENV] == _fargate_env(synth_default)[MODELS_ENV]
    assert FAILOVER_TEST_ENV not in env
    assert env[mf.POLICY_TABLE_ENV] == {"Fn::ImportValue": env[mf.POLICY_TABLE_ENV]["Fn::ImportValue"]}
    assert "PolicyTable" in env[mf.POLICY_TABLE_ENV]["Fn::ImportValue"]
    for key in env:
        assert not key.startswith("OWLBANK_SECRET_"), key
        assert not key.startswith("OWLBANK_TABLE_") or key in (mf.TABLE_ENV, mf.POLICY_TABLE_ENV), key


def test_poller_is_never_retried(synth_default):
    lid = _poller_id(synth_default)
    (cfg_,) = [r["Properties"] for r in _resources(synth_default, COMPUTE, "AWS::Lambda::EventInvokeConfig").values()
               if r["Properties"]["FunctionName"] == {"Ref": lid}]
    assert cfg_["MaximumRetryAttempts"] == 0


def test_poller_gets_the_test_delay_with_the_bridge(synth_test_delay):
    env = _poller(synth_test_delay)["Properties"]["Environment"]["Variables"]
    assert env[FAILOVER_TEST_ENV] == "6" == _fargate_env(synth_test_delay)[FAILOVER_TEST_ENV]


def test_schedule_is_every_minute(synth_default):
    (rule,) = [r["Properties"] for r in _resources(synth_default, COMPUTE, "AWS::Events::Rule").values()
               if r["Properties"].get("ScheduleExpression")]
    assert rule["ScheduleExpression"] == "rate(1 minute)"
    assert rule["State"] == "ENABLED"
    (target,) = rule["Targets"]
    assert target["Arn"] == {"Fn::GetAtt": [_poller_id(synth_default), "Arn"]}
    assert target["RetryPolicy"]["MaximumRetryAttempts"] == 0


# ---- permissions


def test_poller_dynamodb_least_privilege(synth_default):
    sts = _modelstats_statements(_statements(synth_default, COMPUTE, _poller_role_id(synth_default)))
    by_sid = {st["Sid"]: st for st in sts}
    assert set(by_sid) == {"ModelstatsQuerySamples", "ModelstatsWriteState"}
    q, w = by_sid["ModelstatsQuerySamples"], by_sid["ModelstatsWriteState"]
    assert _as_list(q["Action"]) == ["dynamodb:Query"] and _leading(q) == ["samples"]
    assert sorted(_as_list(w["Action"])) == ["dynamodb:GetItem", "dynamodb:PutItem"]
    assert _leading(w) == ["state"]
    for st in sts:
        assert _as_list(st["Resource"]) == [TABLE_ARN]
    actions = {a for st in sts for a in _as_list(st["Action"])}
    assert not actions & {"dynamodb:UpdateItem", "dynamodb:Scan", "dynamodb:DeleteItem",
                          "dynamodb:BatchWriteItem"}


def test_poller_reads_the_policy_table_only(synth_default):
    """The model_failover policy section (Policy.from_store): GetItem and BatchGetItem on the policy
    table, nothing else on any Data-stack table; KMS through the one ViaService statement."""
    sts = _statements(synth_default, COMPUTE, _poller_role_id(synth_default))
    other = [st for st in _dynamodb_statements(sts)
             if not any("modelstats" in str(r) for r in _as_list(st["Resource"]))]
    (st,) = other
    assert st["Sid"] == "PolicyRead"
    assert sorted(_as_list(st["Action"])) == ["dynamodb:BatchGetItem", "dynamodb:GetItem"]
    (res,) = _as_list(st["Resource"])
    assert "PolicyTable" in json.dumps(res) and "/index" not in json.dumps(res)
    kms_sts = [s for s in sts if any(a.startswith("kms:") for a in _as_list(s["Action"]))]
    assert len(kms_sts) == 1
    assert kms_sts[0]["Condition"]["StringEquals"]["kms:ViaService"] == "dynamodb.us-east-1.amazonaws.com"


def test_poller_invokes_every_entry_through_its_profile(synth_default):
    sts = _statements(synth_default, COMPUTE, _poller_role_id(synth_default))
    bedrock = [st for st in sts if any(a.startswith("bedrock:") for a in _as_list(st["Action"]))]
    assert {st["Sid"] for st in bedrock} == {f"{p}{part}" for p in ("PollerInvokeModel", "PollerInvokeModel2",
                                                                     "PollerInvokeModel3")
                                             for part in ("Profile", "ViaProfile")}
    for st in bedrock:
        assert sorted(_as_list(st["Action"])) == ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
    profiles = sorted(st["Resource"] for st in bedrock if st["Sid"].endswith("Profile")
                      and not st["Sid"].endswith("ViaProfile"))
    assert profiles == sorted(f"arn:aws:bedrock:us-east-1:111111111111:inference-profile/{m.id}"
                              for m in synth_default.cfg.models)
    for st in bedrock:
        if st["Sid"].endswith("ViaProfile"):
            assert "bedrock:InferenceProfileArn" in st["Condition"]["StringEquals"]


def test_poller_role_has_no_other_access(synth_default):
    sts = _statements(synth_default, COMPUTE, _poller_role_id(synth_default))
    prefixes = {a.split(":")[0] for st in sts for a in _as_list(st["Action"])}
    assert prefixes <= {"dynamodb", "bedrock", "kms", "logs", "xray"}, prefixes
    kms_sts = [st for st in sts if any(a.startswith("kms:") for a in _as_list(st["Action"]))]
    assert len(kms_sts) == 1
    assert "kms:ViaService" in kms_sts[0]["Condition"]["StringEquals"]
    for st in sts:
        for a in _as_list(st["Action"]):
            assert "*" not in a, st


@pytest.mark.parametrize("fixture", ["synth_default", "synth_agentcore_vpc_dev"])
def test_bridge_dynamodb_least_privilege(fixture, request):
    synth = request.getfixturevalue(fixture)
    (role_id,) = _bridge_role_ids(synth)
    sts = _modelstats_statements(_statements(synth, BRIDGE, role_id))
    by_sid = {st["Sid"]: st for st in sts}
    assert set(by_sid) == {"ModelstatsReportSamples", "ModelstatsReadState"}
    r, g = by_sid["ModelstatsReportSamples"], by_sid["ModelstatsReadState"]
    assert _as_list(r["Action"]) == ["dynamodb:UpdateItem"] and _leading(r) == ["samples"]
    assert _as_list(g["Action"]) == ["dynamodb:GetItem"] and _leading(g) == ["state"]
    for st in sts:
        assert _as_list(st["Resource"]) == [TABLE_ARN]


@pytest.mark.parametrize("fixture", ["synth_default", "synth_agentcore_vpc_dev"])
def test_bridge_env_names_the_table(fixture, request):
    synth = request.getfixturevalue(fixture)
    env = _runtime_env(synth) if synth.cfg.is_agentcore else _fargate_env(synth)
    assert env[mf.TABLE_ENV] == "owlbank-dev-modelstats"
    # No stage on the bridge for failover: the partitions are fixed values. Call observability
    # (off here) is what sets OWLBANK_STAGE on the bridge.
    assert mf.STAGE_ENV not in env


def test_no_other_role_touches_the_table(synth_default):
    poller_role = _poller_role_id(synth_default)
    for stack in synth_default.templates:
        for res in _resources(synth_default, stack, "AWS::IAM::Policy").values():
            roles = [r.get("Ref") for r in res["Properties"].get("Roles", [])]
            sts = _modelstats_statements(_flat(res["Properties"]["PolicyDocument"]["Statement"]))
            if not sts:
                continue
            assert (stack == COMPUTE and roles == [poller_role]) or (
                stack == BRIDGE and roles == _bridge_role_ids(synth_default)), (stack, roles)


# ---- Alarms (4a.)


def test_state_stale_alarm(synth_default):
    a = _alarm(synth_default, COMPUTE, "ModelStateStaleAlarm")
    assert a["Namespace"] == "OwlBank/dev/Model" and a["MetricName"] == mf.HEARTBEAT_METRIC
    assert {d["Name"]: d["Value"] for d in a["Dimensions"]} == {"Service": "owlbank-dev-model-poller",
                                                                "Environment": "dev"}
    assert a["Statistic"] == "Sum" and a["Period"] == 60 and a["EvaluationPeriods"] == 3
    assert a["Threshold"] == 1 and a["ComparisonOperator"] == "LessThanThreshold"
    assert a["TreatMissingData"] == "breaching"


def test_all_entries_down_alarm(synth_default):
    a = _alarm(synth_default, COMPUTE, "ModelAllEntriesDownAlarm")
    assert a["MetricName"] == mf.USABLE_METRIC and a["Statistic"] == "Maximum"
    assert a["Threshold"] == 1 and a["ComparisonOperator"] == "LessThanThreshold"


def test_poller_errors_alarm(synth_default):
    a = _alarm(synth_default, COMPUTE, "ModelPollerErrorsAlarm")
    assert a["MetricName"] == "Errors"


@pytest.mark.parametrize("fixture", ["synth_default", "synth_agentcore_vpc_dev"])
def test_bridge_alarms(fixture, request):
    synth = request.getfixturevalue(fixture)
    preferred = synth.cfg.models[0].id
    rank = _alarm(synth, BRIDGE, "ModelActiveRankAlarm")
    assert rank["Namespace"] == "OwlBank/dev/Model" and rank["MetricName"] == "ActiveRank"
    assert {d["Name"]: d["Value"] for d in rank["Dimensions"]} == {"Service": "owlbank-dev-bridge",
                                                                   "Environment": "dev"}
    assert rank["Statistic"] == "Maximum" and rank["Period"] == 300 and rank["EvaluationPeriods"] == 3
    assert rank["Threshold"] == 0 and rank["ComparisonOperator"] == "GreaterThanThreshold"
    filters = {r["Properties"]["MetricTransformations"][0]["MetricName"]: r["Properties"]["FilterPattern"]
               for r in _resources(synth, BRIDGE, "AWS::Logs::MetricFilter").values()}
    assert '$.event = "model_unavailable"' in filters["ModelUnavailable"]
    assert f'$.model_id = "{preferred}"' in filters["PreferredModelUnavailable"]
    for prefix, metric in (("ModelUnavailableAlarm", "ModelUnavailable"),
                           ("PreferredModelUnavailableAlarm", "PreferredModelUnavailable")):
        a = _alarm(synth, BRIDGE, prefix)
        assert a["MetricName"] == metric and a["Threshold"] == 1 and a["EvaluationPeriods"] == 1
        assert a["ComparisonOperator"] == "GreaterThanOrEqualToThreshold" and a["AlarmActions"]
    assert preferred in _alarm(synth, BRIDGE, "PreferredModelUnavailableAlarm")["AlarmDescription"]


# ---- cdk-nag and the README


@pytest.mark.parametrize("fixture", ["synth_default", "synth_agentcore_vpc_dev", "synth_test_delay"])
def test_nag_clean(fixture, request):
    report = _nag_report(request.getfixturevalue(fixture))
    if report is not None:
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [(v["ruleName"], v.get("violatingResources")) for v in errors]


def test_pitr_ack_has_a_reason(synth_default):
    acks = [a for a in synth_default.acks if a[1] == "AwsSolutions-DDB3"]
    assert [a[0] for a in acks] == ["OwlBankData/ModelstatsTable"]


def test_settings_table_documents_the_switch():
    rows = [ln for ln in README.read_text().splitlines() if ln.startswith("| `modelFailover` |")]
    assert len(rows) == 1
    _, default, meaning = (c.strip() for c in rows[0].strip("|").split("|"))
    assert default == "`on`"
    for needle in ("`off`", "list of one", "modelstats", "poller", mf.TABLE_ENV, "first entry",
                   "failoverTest"):
        assert needle in meaning, needle
