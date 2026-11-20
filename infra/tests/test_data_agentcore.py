"""AgentCore hosting: the Data stack with ``bridgeHosting=agentcore``.

The CMK gains one statement, ``RuntimeLogsCmk``, letting CloudWatch Logs use it on the live
Runtime's log group (``/aws/bedrock-agentcore/runtimes/<runtime_name>-*``). The
token key's policy is identical across modes: the execution role keeps the fixed ``bridge-task``
name, so nothing else in Data changes. Fargate is held byte for byte by the snapshot test
. No PBT: one fixed statement, asserted exactly.
"""

from __future__ import annotations

import json
import warnings

import pytest
from conftest import _synth
from test_synth_and_nag import _nag_report

DATA = "OwlBankData"
ACCOUNT, REGION = "111111111111", "us-east-1"
DIGEST = "sha256:" + "ab" * 32
LOGS_ACTIONS = ["kms:Encrypt*", "kms:Decrypt*", "kms:ReEncrypt*", "kms:GenerateDataKey*", "kms:Describe*"]
PROD = {"stage": "prod", "domainName": "owlbank.example.com",
        "certificateArn": "arn:aws:acm:us-east-1:111111111111:certificate/"
                          "11111111-2222-3333-4444-555555555555",
        "bridgeDesiredCount": 2}
#: The cached AZ name -> AZ ID map the Bridge stack needs in VPC mode; fictional values.
AZ_MAP = {"owlbank:az-ids:111111111111:us-east-1": {"us-east-1a": "use1-az1", "us-east-1b": "use1-az2"}}
AGENTCORE = {"bridgeHosting": "agentcore", "bridgeImageDigest": DIGEST, "twilioPhoneNumber": "+15555550123",
             **AZ_MAP}

#: (fixture name, stage, context): agentcore x {vpc, public} x {dev, prod-with-ack}.
CASES = {
    "dev-vpc": ("dev", {**AGENTCORE, "agentcoreNetworkMode": "vpc"}),
    "dev-public": ("dev", {**AGENTCORE, "agentcoreNetworkMode": "public"}),
    "prod-vpc": ("prod", {**AGENTCORE, **PROD, "agentcoreNetworkMode": "vpc"}),
    "prod-public": ("prod", {**AGENTCORE, **PROD, "agentcoreNetworkMode": "public",
                             "agentcoreNetworkModeAck": "public-at-prod"}),
}


def _log_group_arn(name: str) -> dict:
    return {"Fn::Join": ["", ["arn:", {"Ref": "AWS::Partition"},
                              f":logs:{REGION}:{ACCOUNT}:log-group:/aws/bedrock-agentcore/runtimes/{name}-*"]]}


def _key(template: dict, construct_id: str) -> tuple[str, dict]:
    keys = [(lid, res) for lid, res in template["Resources"].items()
            if res["Type"] == "AWS::KMS::Key" and lid.startswith(construct_id)]
    assert len(keys) == 1, keys
    return keys[0]


def _sids(template: dict, construct_id: str) -> dict[str, dict]:
    # CDK's default root statement on the CMK has no Sid; key it by its content instead.
    return {st.get("Sid") or json.dumps(st, sort_keys=True): st
            for st in _key(template, construct_id)[1]["Properties"]["KeyPolicy"]["Statement"]}


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request, tmp_path_factory):
    stage, ctx = CASES[request.param]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # public mode warns by design
        synth = _synth(tmp_path_factory, f"data-agentcore-{request.param}", ctx)
    return stage, synth


@pytest.fixture(scope="module")
def synth_fargate_prod(tmp_path_factory):
    return _synth(tmp_path_factory, "data-fargate-prod", PROD)


# --- CMK: the RuntimeLogsCmk statement ------------------------------------------------------------

def test_cmk_allows_logs_on_the_runtime_log_group_only(case, synth_default, synth_fargate_prod):
    stage, synth = case
    fargate = synth_fargate_prod if stage == "prod" else synth_default
    on, off = _sids(synth.templates[DATA], "DataKey"), _sids(fargate.templates[DATA], "DataKey")
    assert set(on) - set(off) == {"RuntimeLogsCmk"}
    assert {sid: on[sid] for sid in off} == off  # every existing statement is unchanged
    st = on["RuntimeLogsCmk"]
    assert st["Effect"] == "Allow"
    assert st["Principal"] == {"Service": f"logs.{REGION}.amazonaws.com"}
    assert st["Action"] == LOGS_ACTIONS and st["Resource"] == "*"
    assert st["Condition"] == {"ArnLike": {
        "kms:EncryptionContext:aws:logs:arn": _log_group_arn(f"owlbank_{stage}_bridge")}}


def test_fargate_has_no_runtime_logs_statement(synth_default, synth_fargate_prod):
    for synth in (synth_default, synth_fargate_prod):
        data = json.dumps(synth.templates[DATA])
        assert "RuntimeLogsCmk" not in data and "bedrock-agentcore" not in data


# --- token key and everything else in Data: identical across modes --------------------------------

def test_token_key_policy_identical_across_modes(case, synth_default, synth_fargate_prod):
    stage, synth = case
    fargate = synth_fargate_prod if stage == "prod" else synth_default
    on, off = _key(synth.templates[DATA], "TokenKey"), _key(fargate.templates[DATA], "TokenKey")
    assert on == off
    bridge = json.dumps(on[1]["Properties"]["KeyPolicy"])
    assert f"role/owlbank-{stage}-bridge-task" in bridge  # The execution role keeps the name


def test_agentcore_changes_only_the_cmk_policy(case, synth_default, synth_fargate_prod):
    stage, synth = case
    fargate = synth_fargate_prod if stage == "prod" else synth_default
    on, off = synth.templates[DATA], fargate.templates[DATA]
    cmk = _key(off, "DataKey")[0]
    assert set(on["Resources"]) == set(off["Resources"])
    for lid, res in off["Resources"].items():
        if lid == cmk:
            strip = lambda r: {**r, "Properties": {k: v for k, v in r["Properties"].items()  # noqa: E731
                                                   if k != "KeyPolicy"}}
            assert strip(on["Resources"][lid]) == strip(res), lid
        else:
            assert on["Resources"][lid] == res, lid
    assert {k: v for k, v in on.items() if k != "Resources"} == \
        {k: v for k, v in off.items() if k != "Resources"}


def test_no_star_principal_outside_deny(case):
    _, synth = case
    for lid, res in synth.templates[DATA]["Resources"].items():
        for st in (res.get("Properties", {}).get("KeyPolicy") or {}).get("Statement", []):
            if '"*"' in json.dumps(st.get("Principal")):
                assert st["Effect"] == "Deny" and st["Sid"] == "DenyCryptoExceptSealingRoles", (lid, st)


def test_cdk_nag_clean_in_agentcore(case):
    _, synth = case
    report = _nag_report(synth)
    if report is not None:  # on success cdk-nag may leave no AwsSolutions entry at all
        assert not [v["ruleName"] for v in report["violations"] if v["severity"] == "error"]
        assert report["conclusion"] == "success"
