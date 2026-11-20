"""The Bedrock invocation-logging change alarm.

An EventBridge rule on the default bus matches CloudTrail's ``PutModelInvocationLoggingConfiguration``
from Bedrock and notifies the CMK-encrypted alarm topic. The topic policy lets
``events.amazonaws.com`` publish, and the CMK's key policy lets it use the key, each only with
``aws:SourceArn`` = the rule's ARN. No ``*`` principal is added. The rule exists in every hosting
mode and stage.

The rule is in the deployment Region only; whether the US inference profile needs a
rule per routed Region is an open question. No PBT: CDK wiring, covered by assertions.
"""

from __future__ import annotations

import json

import pytest
from test_synth_and_nag import _nag_report

DATA = "OwlBankData"
REGION, ACCOUNT = "us-east-1", "111111111111"

#: The pattern, written out here so a drift in the stack fails this test.
PATTERN = {
    "source": ["aws.bedrock"],
    "detail-type": ["AWS API Call via CloudTrail"],
    "detail": {"eventSource": ["bedrock.amazonaws.com"],
               "eventName": ["PutModelInvocationLoggingConfiguration"]},
}
EVENTS = {"Service": "events.amazonaws.com"}


def _rule_arn(stage: str) -> dict:
    return {"Fn::Join": ["", ["arn:", {"Ref": "AWS::Partition"},
                              f":events:{REGION}:{ACCOUNT}:rule/owlbank-{stage}-bedrock-logging-change"]]}


def _only(template: dict, rtype: str, prefix: str = "") -> tuple[str, dict]:
    found = [(lid, r) for lid, r in template["Resources"].items()
             if r["Type"] == rtype and lid.startswith(prefix)]
    assert len(found) == 1, (rtype, prefix, [lid for lid, _ in found])
    return found[0]


def _topic_id(template: dict) -> str:
    return _only(template, "AWS::SNS::Topic", "AlarmTopic")[0]


def _events_statements(statements: list[dict]) -> list[dict]:
    return [st for st in statements if st.get("Principal") == EVENTS]


@pytest.fixture(params=["synth_default", "synth_prod", "synth_agentcore_vpc_dev"])
def case(request):
    synth = request.getfixturevalue(request.param)
    return synth.cfg.stage, synth


def test_rule_pattern_and_target(case):
    stage, synth = case
    tpl = synth.templates[DATA]
    _, rule = _only(tpl, "AWS::Events::Rule")
    props = rule["Properties"]
    assert props["Name"] == f"owlbank-{stage}-bedrock-logging-change"
    assert props.get("EventBusName", "default") == "default"
    assert props["State"] == "ENABLED"
    assert props["EventPattern"] == PATTERN
    assert len(props["Targets"]) == 1
    target = props["Targets"][0]
    assert target["Arn"] == {"Ref": _topic_id(tpl)}
    # SNS takes a resource policy, not a role (no RoleArn, so no extra IAM role).
    assert "RoleArn" not in target
    # No other rule anywhere in the app, but model failover's poller schedule (on with the default list).
    others = [(s, lid) for s, lid, _ in synth.resources("AWS::Events::Rule") if s != DATA]
    assert others == ([("OwlBankCompute", others[0][1])] if synth.cfg.failover_on else [])
    if others:
        assert others[0][1].startswith("ModelPollerSchedule")
    assert [s for s, _, _ in synth.resources("AWS::Events::Rule")].count(DATA) == 1


def test_topic_policy_allows_only_this_rule(case):
    stage, synth = case
    tpl = synth.templates[DATA]
    _, policy = _only(tpl, "AWS::SNS::TopicPolicy", "AlarmTopic")
    statements = policy["Properties"]["PolicyDocument"]["Statement"]
    events = _events_statements(statements)
    assert events == [{
        "Sid": "AllowBedrockLoggingRulePublish",
        "Effect": "Allow",
        "Principal": EVENTS,
        "Action": "sns:Publish",
        "Resource": {"Ref": _topic_id(tpl)},
        "Condition": {"ArnEquals": {"aws:SourceArn": _rule_arn(stage)}},
    }]
    # The TLS-only Deny is kept.
    assert any(st["Sid"] == "AllowPublishThroughSSLOnly" and st["Effect"] == "Deny" for st in statements)


def test_key_policy_allows_events_only_for_this_rule(case):
    stage, synth = case
    tpl = synth.templates[DATA]
    _, key = _only(tpl, "AWS::KMS::Key", "DataKey")
    events = _events_statements(key["Properties"]["KeyPolicy"]["Statement"])
    assert events == [{
        "Sid": "AllowBedrockLoggingRuleToEncryptedTopic",
        "Effect": "Allow",
        "Principal": EVENTS,
        "Action": ["kms:GenerateDataKey*", "kms:Decrypt"],
        "Resource": "*",  # in a key policy, "*" means this key
        "Condition": {"ArnEquals": {"aws:SourceArn": _rule_arn(stage)}},
    }]
    # The token key never names EventBridge.
    _, token = _only(tpl, "AWS::KMS::Key", "TokenKey")
    assert not _events_statements(token["Properties"]["KeyPolicy"]["Statement"])


def test_rule_arn_in_policies_matches_the_rule_name(case):
    """The policies build the rule's ARN from its fixed name (a GetAtt would make a cycle: key ->
    rule -> topic -> key); the name in the ARN must be the rule's actual name."""
    stage, synth = case
    tpl = synth.templates[DATA]
    _, rule = _only(tpl, "AWS::Events::Rule")
    arn_tail = _rule_arn(stage)["Fn::Join"][1][2]
    assert arn_tail.endswith("/" + rule["Properties"]["Name"])


def _statements(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get("Statement"), list):
            yield from obj["Statement"]
        for v in obj.values():
            yield from _statements(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _statements(v)


def _star(principal) -> bool:
    if principal == "*":
        return True
    if isinstance(principal, dict):
        return any(v == "*" or (isinstance(v, list) and "*" in v) for v in principal.values())
    return isinstance(principal, list) and "*" in principal


def test_no_star_principal_in_any_allow(synth_matrix):
    """Every statement of every synthesized template: no Allow has a ``*`` principal. Deny guards
    with ``*`` (TLS-only, token key) are held to their allow-list by tests/test_aws_defaults.py."""
    for stack, tpl in synth_matrix.templates.items():
        for st in _statements(tpl):
            if st.get("Effect") == "Allow":
                assert not _star(st.get("Principal")), (stack, st)
                assert "NotPrincipal" not in st, (stack, st)
    # The two new statements are Allows with a service principal, never "*".
    text = json.dumps(synth_matrix.templates[DATA])
    assert "AllowBedrockLoggingRulePublish" in text and "AllowBedrockLoggingRuleToEncryptedTopic" in text


def test_cdk_nag_clean(case):
    _, synth = case
    report = _nag_report(synth)
    if report is not None:
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [(v["ruleName"], v.get("violatingResources")) for v in errors]
        assert report["conclusion"] == "success"
