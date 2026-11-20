"""AgentCore hosting: ``OwlBankBridge`` with ``bridgeHosting=agentcore`` and the stack layout per mode.

The matrix is ``agentcore`` x {``vpc``, ``public``} x {dev (CloudFront mode), prod with a domain and,
for ``public``, ``agentcoreNetworkModeAck``}. Each is compared with the Fargate synth of the same
stage and domain (``synth_default``, ``synth_prod``). Assertions:

* No ECS, ELBv2, CloudFront, bridge WAF or ALB log bucket in any template,
  and exactly one ``AWS::BedrockAgentCore::Runtime``, in ``OwlBankBridge``.
* The retained fixed-name resources (origin-verify secret in the CloudFront mode; at prod
  the bridge and WAF log groups) are present with the Fargate logical IDs and properties.
* ``TaskRole`` keeps its logical ID and name; its trust is the AgentCore service only; its
  statements are the Fargate task role's plus the Runtime's additions, nothing else (exact IAM).
* No ``*`` principal outside a Deny, no Function URL, no authorizer or header allow-list on
  the Runtime; in ``vpc`` the security group has egress on 443 only and no ingress.
* The image by digest from our repository; the digest is required by this stack, and the
  image stack synthesizes without it.
* No ``OwlBankNetwork`` in ``public``; in ``vpc`` the subnets chosen by AZ ID (one subnet
  when only one AZ ID is supported) and a Network template identical to Fargate's (exports kept).
* The log group on the CMK, the outputs, and cdk-nag clean.

PBT: the template part of the Runtime-only layout over generated valid ``agentcore`` contexts (one synth per example).
"""

from __future__ import annotations

import json
import tempfile
import warnings

import pytest
import test_aws_defaults as cp11
from aws_cdk import App, Environment
from conftest import FAKE_CERT, INFRA, _synth
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from test_synth_and_nag import _nag_report

from owlbank_infra.app_builder import build_app
from owlbank_infra.config import ConfigError, from_context
from owlbank_infra.image_stack import BridgeImageStack
from owlbank_infra.nag import ACKS
from owlbank_infra.runtime import MAX_LIFETIME_S

BRIDGE, DATA, NETWORK, IMAGE = "OwlBankBridge", "OwlBankData", "OwlBankNetwork", "OwlBankBridgeImage"
ACCOUNT, REGION = "111111111111", "us-east-1"
DIGEST = "sha256:" + "ef" * 32
AZ_KEY = f"owlbank:az-ids:{ACCOUNT}:{REGION}"
AZ_MAP = {AZ_KEY: {"us-east-1a": "use1-az1", "us-east-1b": "use1-az2"}}
AGENTCORE = {"bridgeHosting": "agentcore", "bridgeImageDigest": DIGEST, "twilioPhoneNumber": "+15555550123",
             **AZ_MAP}
#: ``synth_prod``'s context (conftest) without the alarm email, which only the Data stack uses.
PROD = {"stage": "prod", "domainName": "owlbank.example.com", "certificateArn": FAKE_CERT,
        "bridgeDesiredCount": 2, "owner": "voice-team", "costCenter": "cc-1234",
        "dataClassification": "restricted"}
#: Resource types that exist only for the Fargate bridge.
FARGATE_ONLY_PREFIXES = ("AWS::ECS::", "AWS::ElasticLoadBalancingV2::", "AWS::CloudFront::")
RUNTIME = "AWS::BedrockAgentCore::Runtime"
ADDED_SIDS = {"ReadTwilioSecrets", "DecryptSecretsViaSecretsManager", "PullBridgeImage",
              "EcrAuthToken", "RuntimeMetrics"}
LOG_SIDS = {"RuntimeLogStreams", "RuntimeLogGroup"}
OUTPUTS = {"BridgeHosting", "RuntimeArn", "RuntimeId", "RuntimeLogGroup", "PresignRoleArn",
           "AgentcoreNetworkMode", "OrchestratorCallbackPath"}


def _ctx(mode: str, stage: str) -> dict:
    ctx = {**AGENTCORE, "agentcoreNetworkMode": mode}
    if stage == "prod":
        ctx.update(PROD)
        if mode == "public":
            ctx["agentcoreNetworkModeAck"] = "public-at-prod"
    return ctx


def _quiet_synth(factory, name, ctx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # public mode's trade-off warning
        return _synth(factory, name, ctx)


@pytest.fixture(scope="module", params=[("vpc", "dev"), ("public", "dev"), ("vpc", "prod"), ("public", "prod")],
                ids=lambda p: f"{p[0]}-{p[1]}")
def ac(request, tmp_path_factory):
    mode, stage = request.param
    synth = _quiet_synth(tmp_path_factory, f"ac-{mode}-{stage}", _ctx(mode, stage))
    synth.mode, synth.stage = mode, stage  # type: ignore[attr-defined]
    return synth


@pytest.fixture
def fargate(ac, request):
    """The Fargate synth with the same stage and domain as ``ac``."""
    return request.getfixturevalue("synth_prod" if ac.stage == "prod" else "synth_default")


# --------------------------------------------------------------------------- helpers

def _res(synth, stack, rtype=None):
    return {lid: r for lid, r in synth.templates[stack]["Resources"].items()
            if rtype is None or r["Type"] == rtype}


def _one(synth, stack, rtype):
    found = _res(synth, stack, rtype)
    assert len(found) == 1, (rtype, sorted(found))
    return next(iter(found.items()))


def _role_lid(synth, stack, name):
    (lid,) = [lid for lid, r in _res(synth, stack, "AWS::IAM::Role").items()
              if r["Properties"].get("RoleName") == name]
    return lid


def _statements(synth, stack, role_lid):
    return [s for p in _res(synth, stack, "AWS::IAM::Policy").values()
            if {"Ref": role_lid} in p["Properties"].get("Roles", [])
            for s in p["Properties"]["PolicyDocument"]["Statement"]]


def _cfg(synth):
    return synth.stacks["compute"].cfg


def _role_arn(name):
    return {"Fn::Join": ["", ["arn:", {"Ref": "AWS::Partition"}, f":iam::{ACCOUNT}:role/{name}"]]}


def _r_and_e(rt_lid):
    arn = {"Fn::GetAtt": [rt_lid, "AgentRuntimeArn"]}
    return [arn, {"Fn::Join": ["", [arn, "/runtime-endpoint/DEFAULT"]]}]


def _imports(node, out=None):
    out = set() if out is None else out
    if isinstance(node, dict):
        if "Fn::ImportValue" in node:
            out.add(json.dumps(node["Fn::ImportValue"], sort_keys=True))
        for v in node.values():
            _imports(v, out)
    elif isinstance(node, list):
        for v in node:
            _imports(v, out)
    return out


def _exports(tpl):
    return {o["Export"]["Name"] for o in tpl.get("Outputs", {}).values() if "Export" in o}


def assert_cph2_template(templates: dict[str, dict]) -> None:
    """Nothing of the Fargate bridge anywhere, exactly one Runtime."""
    runtimes = []
    for stack, tpl in templates.items():
        for lid, r in tpl.get("Resources", {}).items():
            t = r["Type"]
            assert not t.startswith(FARGATE_ONLY_PREFIXES), (stack, lid, t)
            if t == RUNTIME:
                runtimes.append((stack, lid))
            if stack == BRIDGE:
                # The bridge WAF (web ACL, association, logging) and the ALB log bucket.
                assert not t.startswith("AWS::WAFv2::"), (lid, t)
                assert t != "AWS::S3::Bucket", (lid, t)
            if t == "AWS::WAFv2::WebACL":
                name = r["Properties"].get("Name", "")
                assert not name.endswith(("-alb", "-cloudfront")), (stack, name)
    assert len(runtimes) == 1 and runtimes[0][0] == BRIDGE, runtimes


# --------------------------------------------------------------------------- Digest

def test_synth_fails_without_the_digest_in_agentcore(tmp_path):
    ctx = {"adminConsole": "false", **{k: v for k, v in AGENTCORE.items() if k != "bridgeImageDigest"}}
    with pytest.raises(ConfigError, match="bridgeImageDigest.*bridgeHosting=agentcore"):
        build_app(ctx, outdir=str(tmp_path))


def test_the_image_stack_synthesizes_without_the_digest(tmp_path):
    """Push-then-deploy: config doesn't need the digest, and the repository stack doesn't."""
    cfg = from_context({"bridgeHosting": "agentcore"}, base_dir=INFRA)
    assert cfg.is_agentcore and cfg.bridge_image_digest is None
    app = App(outdir=str(tmp_path))
    stack = BridgeImageStack(app, IMAGE, cfg=cfg, env=Environment(account=cfg.account, region=cfg.region))
    template = app.synth().get_stack_by_name(stack.stack_name).template
    assert [r["Type"] for r in template["Resources"].values()] == ["AWS::ECR::Repository"]


# --------------------------------------------------------------------------- Layout

def test_no_fargate_resources_and_one_runtime(ac):
    assert_cph2_template(ac.templates)


def test_stack_layout(ac):
    expected = {DATA, "OwlBankAgent", IMAGE, BRIDGE, "OwlBankCompute", "OwlBankApi"}
    if ac.mode == "vpc":
        expected.add(NETWORK)
    assert set(ac.templates) == expected
    assert ("network" in ac.stacks) == (ac.mode == "vpc")
    bridge_deps = {d.stack_name for d in ac.stacks["bridge"].dependencies}
    assert IMAGE in bridge_deps  # BridgeImage before Bridge (the digest is pushed in between)
    assert (NETWORK in bridge_deps) == (ac.mode == "vpc")
    assert DATA in {d.stack_name for d in ac.stacks["image"].dependencies}


def test_no_network_stack_and_no_vpc_in_public(ac):
    if ac.mode != "public":
        pytest.skip("vpc mode")
    for tpl in ac.templates.values():
        types = {r["Type"] for r in tpl["Resources"].values()}
        assert not {"AWS::EC2::VPC", "AWS::EC2::NatGateway", "AWS::EC2::FlowLog",
                    "AWS::EC2::SecurityGroup"} & types


def test_network_template_unchanged_in_vpc_so_a_switch_keeps_its_exports(ac, fargate):
    """A switch updates Network first: it must not drop an export the other hosting's Bridge imports."""
    if ac.mode != "vpc":
        pytest.skip("public mode has no Network stack")
    assert ac.templates[NETWORK] == fargate.templates[NETWORK]


def test_bridge_imports_only_exports_that_fargate_also_has(ac, fargate):
    """Rollback: Data and Network deploy before Bridge, so every export the agentcore Bridge
    imports must exist in the Fargate templates too, and Data's exports must not change."""
    assert _imports(ac.templates[BRIDGE]) <= _imports(fargate.templates[BRIDGE])
    assert _exports(ac.templates[DATA]) == _exports(fargate.templates[DATA])


# --------------------------------------------------------------------------- Retained resources

def _retained_ids(synth):
    return {lid: r for lid, r in _res(synth, BRIDGE).items() if r.get("DeletionPolicy") == "Retain"}


def test_retained_fixed_name_resources_stay_with_their_logical_ids(ac, fargate):
    fixed_name_retained = {
        lid: r for lid, r in _retained_ids(fargate).items()
        if r["Type"] in ("AWS::SecretsManager::Secret", "AWS::Logs::LogGroup")
        and ("Name" in r["Properties"] or "LogGroupName" in r["Properties"])}
    if ac.stage == "prod":
        assert {r["Properties"]["LogGroupName"] for r in fixed_name_retained.values()} == {
            "/owlbank/ecs/tac-bridge", "aws-waf-logs-owlbank-prod-alb"}
    else:
        assert [r["Properties"]["Name"] for r in fixed_name_retained.values()] == [
            "owlbank/dev/bridge/origin-verify"]
    resources = _res(ac, BRIDGE)
    for lid, r in fixed_name_retained.items():
        assert resources.get(lid) == r, lid  # same logical ID, properties and policies


def test_only_the_runtime_log_group_is_added_besides_the_dormant_ones(ac, fargate):
    groups = _res(ac, BRIDGE, "AWS::Logs::LogGroup")
    dormant = {lid for lid in groups if lid in _res(fargate, BRIDGE)}
    assert len(groups) - len(dormant) == 1
    assert dormant == ({lid for lid, r in _res(fargate, BRIDGE, "AWS::Logs::LogGroup").items()}
                       if ac.stage == "prod" else set())


# --------------------------------------------------------------------------- Execution role, IAM

def test_task_role_keeps_logical_id_and_name(ac, fargate):
    cfg = _cfg(ac)
    lid = _role_lid(ac, BRIDGE, cfg.bridge_task_role_name)
    assert lid == _role_lid(fargate, BRIDGE, cfg.bridge_task_role_name)
    assert lid.startswith("TaskRole")
    trust = _res(ac, BRIDGE)[lid]["Properties"]["AssumeRolePolicyDocument"]["Statement"]
    assert trust == [{
        "Action": "sts:AssumeRole", "Effect": "Allow",
        "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
        "Condition": {"StringEquals": {"aws:SourceAccount": ACCOUNT},
                      "ArnLike": {"aws:SourceArn": {"Fn::Join": ["", [
                          "arn:", {"Ref": "AWS::Partition"}, f":bedrock-agentcore:{REGION}:{ACCOUNT}:*"]]}}}}]


def test_execution_role_is_the_task_role_plus_the_runtime_additions(ac, fargate):
    cfg = _cfg(ac)
    lid = _role_lid(ac, BRIDGE, cfg.bridge_task_role_name)
    mine = _statements(ac, BRIDGE, lid)
    theirs = _statements(fargate, BRIDGE, _role_lid(fargate, BRIDGE, cfg.bridge_task_role_name))
    by_sid = {s["Sid"]: s for s in mine}
    assert len(by_sid) == len(mine), "every statement has its own Sid"
    for st_ in theirs:
        assert by_sid.get(st_["Sid"]) == st_, st_["Sid"]
    assert set(by_sid) - {s["Sid"] for s in theirs} == ADDED_SIDS | LOG_SIDS
    rt_lid, _ = _one(ac, BRIDGE, RUNTIME)
    assert by_sid["PullBridgeImage"]["Resource"] == {"Fn::Join": ["", [
        "arn:", {"Ref": "AWS::Partition"}, f":ecr:{REGION}:{ACCOUNT}:repository/{cfg.image_repo_name}"]]}
    assert sorted(by_sid["PullBridgeImage"]["Action"]) == ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"]
    assert by_sid["EcrAuthToken"] == {"Sid": "EcrAuthToken", "Effect": "Allow",
                                      "Action": "ecr:GetAuthorizationToken", "Resource": "*"}
    assert by_sid["RuntimeMetrics"]["Condition"] == {
        "StringEquals": {"cloudwatch:namespace": "bedrock-agentcore"}}
    assert by_sid["ReadTwilioSecrets"]["Action"] == "secretsmanager:GetSecretValue"
    assert len(by_sid["ReadTwilioSecrets"]["Resource"]) == 2
    log_group = {"Fn::Join": ["", ["/aws/bedrock-agentcore/runtimes/",
                                   {"Fn::GetAtt": [rt_lid, "AgentRuntimeId"]}, "-DEFAULT"]]}
    group_arn = json.dumps(by_sid["RuntimeLogGroup"]["Resource"])
    assert json.dumps({"Fn::GetAtt": [rt_lid, "AgentRuntimeId"]}) in group_arn
    assert ":log-stream:*" in json.dumps(by_sid["RuntimeLogStreams"]["Resource"])
    groups = [r for r in _res(ac, BRIDGE, "AWS::Logs::LogGroup").values()
              if r["Properties"]["LogGroupName"] == log_group]
    assert len(groups) == 1


def test_presign_role_one_statement_and_front_door_trust_only(ac):
    cfg = _cfg(ac)
    lid = _role_lid(ac, BRIDGE, cfg.presign_role_name)
    rt_lid, _ = _one(ac, BRIDGE, RUNTIME)
    assert _statements(ac, BRIDGE, lid) == [{
        "Sid": "InvokeRuntimeWebSocket", "Effect": "Allow",
        "Action": "bedrock-agentcore:InvokeAgentRuntimeWithWebSocketStream",
        "Resource": _r_and_e(rt_lid)}]
    role = _res(ac, BRIDGE)[lid]["Properties"]
    assert role["MaxSessionDuration"] == 3600
    (trust,) = role["AssumeRolePolicyDocument"]["Statement"]
    assert trust["Condition"] == {"ArnEquals": {"aws:PrincipalArn": _role_arn(cfg.frontdoor_role_name)}}
    assert trust["Principal"] == {"AWS": {"Fn::Join": ["", [
        "arn:", {"Ref": "AWS::Partition"}, f":iam::{ACCOUNT}:root"]]}}


def test_only_two_roles_and_no_other_policies(ac):
    cfg = _cfg(ac)
    names = sorted(r["Properties"].get("RoleName") for r in _res(ac, BRIDGE, "AWS::IAM::Role").values())
    assert names == sorted([cfg.bridge_task_role_name, cfg.presign_role_name])
    task, presign = (_role_lid(ac, BRIDGE, n) for n in (cfg.bridge_task_role_name, cfg.presign_role_name))
    for p in _res(ac, BRIDGE, "AWS::IAM::Policy").values():
        assert p["Properties"]["Roles"] in ([{"Ref": task}], [{"Ref": presign}])


@pytest.mark.parametrize("check", [
    cp11.test_no_wildcard_principal, cp11.test_deny_allowlist_entries_are_used,
    cp11.test_no_policy_encoded_as_string, cp11.test_lambda_permissions_scoped_to_source,
    cp11.test_resource_policies_have_no_star_allow,
    cp11.test_no_lambda_function_urls, cp11.test_every_log_group_encrypted_with_cmk,
    cp11.test_lambdas_have_explicit_encrypted_log_groups, cp11.test_no_secret_values_anywhere,
    cp11.test_lambda_env_has_no_secret_values, cp11.test_secrets_encrypted_with_cmk_and_empty,
    cp11.test_tables_encrypted_with_cmk_with_pitr,
], ids=lambda f: f.__name__)
def test_holds_in_agentcore(check, ac):
    """The mode-independent AWS-defaults checks. The bucket, port-443 and policy-type ones expect the ALB
    and its bucket policy; this mode's versions are below."""
    check(ac)


def test_nothing_open_to_the_internet_but_https_egress(ac):
    """No internet-open ingress anywhere (no ALB in this mode); open egress only on 443."""
    rules = list(cp11._open_cidr_rules(ac))
    for lid, _, node in rules:
        assert node.get("FromPort") == 443 and node.get("ToPort") == 443, (lid, node)
    for _, lid, res in ac.resources("AWS::EC2::SecurityGroup"):
        assert not [r for r in res["Properties"].get("SecurityGroupIngress", [])
                    if r.get("CidrIp") == "0.0.0.0/0" or r.get("CidrIpv6") == "::/0"], lid
    assert not [lid for _, lid, r in ac.resources("AWS::EC2::SecurityGroupIngress")
                if r["Properties"].get("CidrIp") == "0.0.0.0/0" or r["Properties"].get("CidrIpv6") == "::/0"]


def test_policy_bearing_types_are_the_fargate_ones_without_the_bucket_policy(ac):
    found = {t for _, _, t, _ in cp11._statements_typed(ac)}
    assert found == cp11.POLICY_BEARING_TYPES - {"AWS::S3::BucketPolicy"}, found


def test_no_s3_bucket_in_any_template(ac):
    """The ALB access-log bucket was the only bucket of the call path."""
    assert not list(ac.resources("AWS::S3::Bucket"))


def test_no_star_principal_and_no_function_url_in_any_template(ac):
    for stack, lid, _, st_ in cp11._statements_typed(ac):
        if cp11._has_star(st_.get("Principal")):
            assert st_["Effect"] == "Deny", (stack, lid, st_)
    assert not list(ac.resources("AWS::Lambda::Url"))
    for _, _, r in ac.resources("AWS::Lambda::Permission"):
        assert r["Properties"].get("FunctionUrlAuthType") is None


# --------------------------------------------------------------------------- the Runtime

def test_runtime_properties(ac):
    cfg = _cfg(ac)
    rt_lid, rt = _one(ac, BRIDGE, RUNTIME)
    p = rt["Properties"]
    assert p["AgentRuntimeName"] == cfg.runtime_name == f"owlbank_{ac.stage}_bridge"
    assert "AuthorizerConfiguration" not in p and "RequestHeaderConfiguration" not in p
    assert p["ProtocolConfiguration"] == "HTTP"
    assert p["LifecycleConfiguration"] == {"IdleRuntimeSessionTimeout": 900, "MaxLifetime": MAX_LIFETIME_S}
    uri = p["AgentRuntimeArtifact"]["ContainerConfiguration"]["ContainerUri"]
    assert uri == {"Fn::Join": ["", [f"{ACCOUNT}.dkr.ecr.{REGION}.", {"Ref": "AWS::URLSuffix"},
                                     f"/{cfg.image_repo_name}@{DIGEST}"]]}
    task = _role_lid(ac, BRIDGE, cfg.bridge_task_role_name)
    assert p["RoleArn"] == {"Fn::GetAtt": [task, "Arn"]}
    assert not _res(ac, BRIDGE, "AWS::BedrockAgentCore::RuntimeEndpoint")  # DEFAULT only


def test_runtime_environment(ac):
    env = _one(ac, BRIDGE, RUNTIME)[1]["Properties"]["EnvironmentVariables"]
    assert env["OWLBANK_HOSTING"] == "agentcore"
    assert "OWLBANK_NONCE_CARRIER" not in env
    assert "TWILIO_VOICE_PUBLIC_DOMAIN" not in env  # no relay host
    for var in ("OWLBANK_SECRET_TWILIO_API_KEY", "OWLBANK_SECRET_TWILIO_AUTH_TOKEN"):
        assert "Fn::ImportValue" in env[var], var
    for var in ("TWILIO_ACCOUNT_SID", "TWILIO_API_KEY", "TWILIO_API_SECRET", "TWILIO_AUTH_TOKEN"):
        assert var not in env


def test_network_configuration_and_security_group(ac):
    rt_lid, rt = _one(ac, BRIDGE, RUNTIME)
    net = rt["Properties"]["NetworkConfiguration"]
    sgs = _res(ac, BRIDGE, "AWS::EC2::SecurityGroup")
    if ac.mode == "public":
        assert net == {"NetworkMode": "PUBLIC"} and not sgs
        return
    (sg_lid, sg), = sgs.items()
    assert net["NetworkMode"] == "VPC"
    assert net["NetworkModeConfig"]["SecurityGroups"] == [{"Fn::GetAtt": [sg_lid, "GroupId"]}]
    subnets = net["NetworkModeConfig"]["Subnets"]
    assert len(subnets) == 2 and all("Fn::ImportValue" in s for s in subnets)
    assert all("privateSubnet" in s["Fn::ImportValue"] for s in subnets)
    assert sg["Properties"]["SecurityGroupEgress"] == [{
        "CidrIp": "0.0.0.0/0", "Description": "HTTPS to Twilio and AWS APIs",
        "FromPort": 443, "IpProtocol": "tcp", "ToPort": 443}]
    assert "SecurityGroupIngress" not in sg["Properties"]
    assert not _res(ac, BRIDGE, "AWS::EC2::SecurityGroupIngress")
    assert not _res(ac, BRIDGE, "AWS::EC2::SecurityGroupEgress")


def test_runtime_log_group_on_the_cmk(ac):
    rt_lid, _ = _one(ac, BRIDGE, RUNTIME)
    name = {"Fn::Join": ["", ["/aws/bedrock-agentcore/runtimes/",
                              {"Fn::GetAtt": [rt_lid, "AgentRuntimeId"]}, "-DEFAULT"]]}
    ((lid, group),) = [(lid, r) for lid, r in _res(ac, BRIDGE, "AWS::Logs::LogGroup").items()
                       if r["Properties"]["LogGroupName"] == name]
    assert group["Properties"]["KmsKeyId"] in cp11._cmk_refs(ac)
    if ac.stage == "prod":
        assert group["Properties"]["RetentionInDays"] == 365 and group["DeletionPolicy"] == "Retain"
    else:
        assert group["Properties"]["RetentionInDays"] == 30 and group["DeletionPolicy"] == "Delete"


def test_single_supported_az_gives_one_subnet_and_a_warning(tmp_path_factory):
    """Only the private subnet whose AZ ID is supported."""
    ctx = {**AGENTCORE, AZ_KEY: {"us-east-1a": "use1-az3", "us-east-1b": "use1-az4"}}
    synth = _quiet_synth(tmp_path_factory, "ac-single-az", ctx)
    subnets = _one(synth, BRIDGE, RUNTIME)[1]["Properties"]["NetworkConfiguration"]["NetworkModeConfig"]["Subnets"]
    assert len(subnets) == 1 and "privateSubnet2" in subnets[0]["Fn::ImportValue"]
    # The other private subnet is still exported (the Network template doesn't change, 1.6).
    names = _exports(synth.templates[NETWORK])
    assert any("privateSubnet1" in n for n in names) and any("privateSubnet2" in n for n in names)


def test_vpc_mode_needs_the_az_map(tmp_path):
    ctx = {"adminConsole": "false", **{k: v for k, v in AGENTCORE.items() if k != AZ_KEY}}
    with pytest.raises(ConfigError, match="preflight"):
        build_app(ctx, outdir=str(tmp_path))


def test_public_mode_needs_no_az_map(tmp_path):
    ctx = {"adminConsole": "false", "agentcoreNetworkMode": "public",
           **{k: v for k, v in AGENTCORE.items() if k != AZ_KEY}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, stacks = build_app(ctx, outdir=str(tmp_path))
    assert "network" not in stacks and stacks["bridge"].runtime is not None


def test_idle_timeout_from_context(tmp_path_factory):
    synth = _quiet_synth(tmp_path_factory, "ac-idle", {**AGENTCORE, "agentcoreIdleTimeoutSeconds": "120"})
    life = _one(synth, BRIDGE, RUNTIME)[1]["Properties"]["LifecycleConfiguration"]
    assert life == {"IdleRuntimeSessionTimeout": 120, "MaxLifetime": MAX_LIFETIME_S}


# --------------------------------------------------------------------------- outputs, nag

def test_outputs(ac):
    outputs = ac.templates[BRIDGE]["Outputs"]
    mine = {k for k in outputs if not k.startswith("ExportsOutput")}
    assert mine == OUTPUTS and "RelayDomain" not in outputs
    assert outputs["BridgeHosting"]["Value"] == "agentcore"
    assert outputs["AgentcoreNetworkMode"]["Value"] == ac.mode
    assert outputs["OrchestratorCallbackPath"]["Value"] == "/voice/orchestrator"
    rt_lid, _ = _one(ac, BRIDGE, RUNTIME)
    assert outputs["RuntimeArn"]["Value"] == {"Fn::GetAtt": [rt_lid, "AgentRuntimeArn"]}
    assert outputs["RuntimeId"]["Value"] == {"Fn::GetAtt": [rt_lid, "AgentRuntimeId"]}
    assert not [k for k, o in outputs.items() if "Export" in o and not k.startswith("ExportsOutput")]


def test_fargate_gets_no_new_output(synth_default, synth_prod):
    for synth in (synth_default, synth_prod):
        assert not OUTPUTS & set(synth.templates[BRIDGE]["Outputs"])


def test_cdk_nag_clean(ac):
    report = _nag_report(ac)
    if report is not None:  # on success cdk-nag may leave no AwsSolutions entry at all
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [(v["ruleName"], v.get("violatingResources")) for v in errors]
        assert report["conclusion"] == "success"
    mine = [(p, r, why) for p, r, why in ACKS if p.startswith(f"{BRIDGE}/")]
    assert all(len(why) >= 40 for _, _, why in mine)


# --------------------------------------------------------------------------- PBT: template part

_az_ids = st.sampled_from(["use1-az1", "use1-az2", "use1-az4", "use1-az3", "use1-az5", "use1-az6"])


@st.composite
def agentcore_contexts(draw) -> dict:
    """Valid ``agentcore`` contexts over the hosting keys' domains."""
    mode = draw(st.sampled_from(["vpc", "public"]))
    stage = draw(st.sampled_from(["dev", "test", "prod"]))
    ctx = {"bridgeHosting": "agentcore", "agentcoreNetworkMode": mode,
           "bridgeImageDigest": "sha256:" + draw(st.text("0123456789abcdef", min_size=64, max_size=64)),
           "agentcoreIdleTimeoutSeconds": str(draw(st.integers(60, MAX_LIFETIME_S - 1))),
           "agentcoreSessionQuota": str(draw(st.integers(1, 1_000_000))), "stage": stage,
           "twilioPhoneNumber": "+15555550123"}
    if stage == "prod" or draw(st.booleans()):  # prod needs the custom-domain mode
        ctx.update({"domainName": "owlbank.example.com", "certificateArn": FAKE_CERT})
    if stage == "prod" and mode == "public":
        ctx["agentcoreNetworkModeAck"] = "public-at-prod"
    if mode == "vpc":
        a, b = draw(_az_ids), draw(_az_ids)
        supported = {"use1-az1", "use1-az2", "use1-az4"}
        if a not in supported and b not in supported:
            a = "use1-az1"
        ctx[AZ_KEY] = {"us-east-1a": a, "us-east-1b": b}
        if stage == "prod" and (a not in supported or b not in supported):
            ctx["agentcoreSingleAzAck"] = "true"
    return ctx


@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ctx=agentcore_contexts())
def test_pbt_template_part(ctx):
    with tempfile.TemporaryDirectory() as out, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        app, stacks = build_app({"adminConsole": "false", **ctx}, outdir=out)
        templates = {s.stack_name: s.template for s in app.synth().stacks}
    assert_cph2_template(templates)
    assert (NETWORK in templates) == (ctx["agentcoreNetworkMode"] == "vpc")
    p = next(r for r in templates[BRIDGE]["Resources"].values() if r["Type"] == RUNTIME)["Properties"]
    assert p["LifecycleConfiguration"]["IdleRuntimeSessionTimeout"] == int(ctx["agentcoreIdleTimeoutSeconds"])
    uri_parts = p["AgentRuntimeArtifact"]["ContainerConfiguration"]["ContainerUri"]["Fn::Join"][1]
    assert uri_parts[-1].endswith(f'-bridge@{ctx["bridgeImageDigest"]}')
    assert p["NetworkConfiguration"]["NetworkMode"] == ("VPC" if ctx["agentcoreNetworkMode"] == "vpc" else "PUBLIC")
    assert "AuthorizerConfiguration" not in p and "RequestHeaderConfiguration" not in p
    bridge_types = {r["Type"] for r in templates[BRIDGE]["Resources"].values()}
    assert ("AWS::EC2::SecurityGroup" in bridge_types) == (ctx["agentcoreNetworkMode"] == "vpc")


def test_agentcore_synth_requires_the_twilio_phone_number():
    """The bridge exits at start-up without TWILIO_PHONE_NUMBER and a Runtime has no circuit
    breaker, so an agentcore synth without -c twilioPhoneNumber fails, naming the key."""
    from owlbank_infra.config import ConfigError

    ctx = dict(AGENTCORE)
    ctx.pop("twilioPhoneNumber", None)
    with pytest.raises(ConfigError, match="twilioPhoneNumber is required by bridgeHosting=agentcore"):
        build_app(ctx)
