"""AgentCore hosting: ``runtime.build_runtime``.

The builder is exercised in two test stacks added to the real app (the Data, Network and Agent
stacks supply the keys, tables, secrets, VPC, Memory and Gateway): ``RuntimeTestVpc`` (VPC mode,
the private subnets) and ``RuntimeTestPublic`` (PUBLIC). Assertions:

* No ``AuthorizerConfiguration``, no ``RequestHeaderConfiguration``, no
  ``CfnRuntimeEndpoint``; the security group has no ingress and egress only on TCP 443;
  no ``*`` principal and no Function URL in the test stacks.
* The presign role has exactly one Allow statement, ``InvokeAgentRuntimeWithWebSocketStream``
  on exactly ``{R, R/runtime-endpoint/DEFAULT}``, and trusts only the named front-door role; the
  execution role's statements are the Fargate ``TaskRole``'s (same synth) plus the Runtime's additions,
  each with its exact actions and resources, and nothing else.
* The log group is ``/aws/bedrock-agentcore/runtimes/<RuntimeId>-DEFAULT`` on the CMK.
* cdk-nag (AwsSolutions) reports no unacknowledged error over the whole app.

The Fargate templates are unchanged by the refactor (``grants.grant_bridge_backend``,
``bridge_stack.bridge_base_env``): ``test_fargate_snapshot.py`` covers that.
The property test checks ``NetworkConfiguration`` and the security group for any subnet list and
network mode.
"""

from __future__ import annotations

import json

import pytest
from aws_cdk import App, Environment, Stack
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_kms as kms
from aws_cdk import aws_secretsmanager as sm
from aws_cdk.assertions import Template
from conftest import INFRA
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank_infra.app_builder import build_app
from owlbank_infra.bridge_stack import TASK_SECRETS, bridge_base_env
from owlbank_infra.config import from_context
from owlbank_infra.grants import BRIDGE_TABLES_RO, BRIDGE_TABLES_RW, function_arn
from owlbank_infra.nag import ACKS
from owlbank_infra.runtime import (
    CREDENTIAL_ENV,
    MAX_LIFETIME_S,
    RuntimeIds,
    RuntimeNames,
    build_runtime,
)

ACCOUNT, REGION = "111111111111", "us-east-1"
DIGEST = "sha256:" + "cd" * 32
IDLE = 60
ADDED_SIDS = {"ReadTwilioSecrets", "DecryptSecretsViaSecretsManager", "PullBridgeImage",
              "EcrAuthToken", "RuntimeMetrics"}


# --- the app with two test stacks -----------------------------------------------------------------

def _build(app_stacks, sid: str, *, vpc_mode: bool):
    cfg = app_stacks["compute"].cfg
    data, network, agent = app_stacks["data"], app_stacks["network"], app_stacks["agent"]
    stack = Stack(data.node.scope, sid, env=Environment(account=ACCOUNT, region=REGION))
    vpc = network.vpc if vpc_mode else None
    subnets = [s.subnet_id for s in network.vpc.private_subnets] if vpc_mode else []
    # Exactly what app_builder hands the Bridge stack (the ``live`` alias).
    bank_tools = cfg.invoke_arn("bank_tools", function_arn(data, cfg.function_name("bank_tools")))
    env = bridge_base_env(stack, cfg, tables=data.tables, token_key=data.token_key,
                          bank_tools_arn=bank_tools, memory_id=agent.memory.attr_memory_id,
                          gateway_url=agent.gateway.attr_gateway_url)
    parts = build_runtime(
        stack, cfg=cfg, names=RuntimeNames.bridge(cfg), key=data.key, token_key=data.token_key,
        tables=data.tables, secrets=data.secrets, repo_name=cfg.image_repo_name, image_digest=DIGEST,
        bank_tools_arn=bank_tools, memory_arn=agent.memory.attr_memory_arn,
        gateway_arn=agent.gateway.attr_gateway_arn, environment=env, idle_timeout_s=IDLE,
        vpc=vpc, subnet_ids=subnets, ids=RuntimeIds(role="TestRuntimeRole"))
    return stack, parts, bank_tools


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("runtime-builder")
    app, stacks = build_app({"adminConsole": "false"}, outdir=str(outdir))
    vpc_stack, vpc_parts, bank_tools = _build(stacks, "RuntimeTestVpc", vpc_mode=True)
    pub_stack, pub_parts, _ = _build(stacks, "RuntimeTestPublic", vpc_mode=False)
    assembly = app.synth()
    templates = {s.stack_name: s.template for s in assembly.stacks}
    return {"templates": templates, "outdir": outdir, "stacks": stacks, "bank_tools": bank_tools,
            "vpc": (templates["RuntimeTestVpc"], vpc_parts),
            "public": (templates["RuntimeTestPublic"], pub_parts)}


def _of_type(tpl, rtype):
    return {lid: r for lid, r in tpl["Resources"].items() if r["Type"] == rtype}


def _one(tpl, rtype):
    found = _of_type(tpl, rtype)
    assert len(found) == 1, (rtype, sorted(found))
    return next(iter(found.items()))


def _role_lid(tpl, name):
    return next(lid for lid, r in _of_type(tpl, "AWS::IAM::Role").items()
                if r["Properties"].get("RoleName") == name)


def _policies_of(tpl, role_lid):
    return {lid: r for lid, r in _of_type(tpl, "AWS::IAM::Policy").items()
            if {"Ref": role_lid} in r["Properties"].get("Roles", [])}


def _statements(policies):
    return [s for p in policies.values() for s in p["Properties"]["PolicyDocument"]["Statement"]]


def _by_sid(statements):
    out: dict[str, list] = {}
    for s in statements:
        out.setdefault(s.get("Sid"), []).append(s)
    return out


@pytest.fixture(params=["vpc", "public"])
def mode(request, built):
    return request.param


# --- the Runtime ----------------------------------------------------------------------

def test_runtime_properties(built, mode):
    tpl, _ = built[mode]
    stacks = built["stacks"]
    cfg = stacks["compute"].cfg
    _, rt = _one(tpl, "AWS::BedrockAgentCore::Runtime")
    p = rt["Properties"]
    assert "AuthorizerConfiguration" not in p
    assert "RequestHeaderConfiguration" not in p
    assert p["AgentRuntimeName"] == cfg.runtime_name
    assert p["ProtocolConfiguration"] == "HTTP"
    assert p["LifecycleConfiguration"] == {"IdleRuntimeSessionTimeout": IDLE, "MaxLifetime": MAX_LIFETIME_S}
    uri = p["AgentRuntimeArtifact"]["ContainerConfiguration"]["ContainerUri"]
    flat = json.dumps(uri)
    assert f"/{cfg.image_repo_name}@{DIGEST}" in flat and ".dkr.ecr." in flat
    assert not _of_type(tpl, "AWS::BedrockAgentCore::RuntimeEndpoint")  # DEFAULT only
    role_lid = _role_lid(tpl, cfg.bridge_task_role_name)
    assert p["RoleArn"] == {"Fn::GetAtt": [role_lid, "Arn"]}


def test_runtime_waits_for_the_execution_policy(built, mode):
    tpl, _ = built[mode]
    cfg = built["stacks"]["compute"].cfg
    rt_lid, rt = _one(tpl, "AWS::BedrockAgentCore::Runtime")
    role_lid = _role_lid(tpl, cfg.bridge_task_role_name)
    default = [lid for lid in _policies_of(tpl, role_lid) if "DefaultPolicy" in lid]
    assert len(default) == 1 and default[0] in rt.get("DependsOn", [])


def test_runtime_environment_has_secret_arns_never_values(built, mode):
    tpl, _ = built[mode]
    env = _one(tpl, "AWS::BedrockAgentCore::Runtime")[1]["Properties"]["EnvironmentVariables"]
    assert env["OWLBANK_HOSTING"] == "agentcore"
    assert "OWLBANK_NONCE_CARRIER" not in env  # The nonce is always <Parameter name="n">
    for var in ("OWLBANK_SECRET_TWILIO_API_KEY", "OWLBANK_SECRET_TWILIO_AUTH_TOKEN"):
        assert "Fn::ImportValue" in env[var], var  # the Data stack's secret ARN
    assert not set(CREDENTIAL_ENV) & set(env)
    # Everything else the task gets today, with the same keys.
    bridge_env = {e["Name"] for e in _one(built["templates"]["OwlBankBridge"],
                                          "AWS::ECS::TaskDefinition")[1]["Properties"]
                  ["ContainerDefinitions"][0]["Environment"]}
    assert bridge_env - {"TWILIO_VOICE_PUBLIC_DOMAIN"} <= set(env)


def test_credential_env_matches_the_fargate_task_secrets():
    assert set(CREDENTIAL_ENV) == set(TASK_SECRETS)


def test_network_configuration(built):
    tpl, parts = built["vpc"]
    net = _one(tpl, "AWS::BedrockAgentCore::Runtime")[1]["Properties"]["NetworkConfiguration"]
    sg_lid, _ = _one(tpl, "AWS::EC2::SecurityGroup")
    assert net["NetworkMode"] == "VPC"
    assert net["NetworkModeConfig"]["SecurityGroups"] == [{"Fn::GetAtt": [sg_lid, "GroupId"]}]
    assert len(net["NetworkModeConfig"]["Subnets"]) == 2  # the two private subnets
    assert all("Fn::ImportValue" in s for s in net["NetworkModeConfig"]["Subnets"])
    tpl_pub, parts_pub = built["public"]
    net_pub = _one(tpl_pub, "AWS::BedrockAgentCore::Runtime")[1]["Properties"]["NetworkConfiguration"]
    assert net_pub == {"NetworkMode": "PUBLIC"}
    assert not _of_type(tpl_pub, "AWS::EC2::SecurityGroup") and parts_pub.security_group is None


def test_security_group_egress_443_only_no_ingress(built):
    tpl, _ = built["vpc"]
    sg_lid, sg = _one(tpl, "AWS::EC2::SecurityGroup")
    assert "SecurityGroupIngress" not in sg["Properties"]
    assert sg["Properties"]["SecurityGroupEgress"] == [{
        "CidrIp": "0.0.0.0/0", "Description": "HTTPS to Twilio and AWS APIs",
        "FromPort": 443, "IpProtocol": "tcp", "ToPort": 443}]
    assert not _of_type(tpl, "AWS::EC2::SecurityGroupIngress")
    assert not _of_type(tpl, "AWS::EC2::SecurityGroupEgress")


# --- log group -------------------------------------------------------------------------------

def test_log_group_name_and_cmk(built, mode):
    tpl, parts = built[mode]
    rt_lid, _ = _one(tpl, "AWS::BedrockAgentCore::Runtime")
    _, lg = _one(tpl, "AWS::Logs::LogGroup")
    assert lg["Properties"]["LogGroupName"] == {"Fn::Join": ["", [
        "/aws/bedrock-agentcore/runtimes/", {"Fn::GetAtt": [rt_lid, "AgentRuntimeId"]}, "-DEFAULT"]]}
    assert "Fn::ImportValue" in lg["Properties"]["KmsKeyId"]  # the Data stack's CMK
    assert lg["Properties"]["RetentionInDays"] == 30 and lg["DeletionPolicy"] == "Delete"  # dev


# --- presign role ---------------------------------------------------------------------------

def _endpoint_resources(rt_lid):
    arn = {"Fn::GetAtt": [rt_lid, "AgentRuntimeArn"]}
    return [arn, {"Fn::Join": ["", [arn, "/runtime-endpoint/DEFAULT"]]}]


def test_presign_role_one_statement_on_runtime_and_default_endpoint(built, mode):
    tpl, _ = built[mode]
    cfg = built["stacks"]["compute"].cfg
    rt_lid, _ = _one(tpl, "AWS::BedrockAgentCore::Runtime")
    lid = _role_lid(tpl, cfg.presign_role_name)
    role = tpl["Resources"][lid]["Properties"]
    assert role["MaxSessionDuration"] == 3600
    assert "ManagedPolicyArns" not in role and "Policies" not in role
    stmts = _statements(_policies_of(tpl, lid))
    assert stmts == [{
        "Sid": "InvokeRuntimeWebSocket", "Effect": "Allow",
        "Action": "bedrock-agentcore:InvokeAgentRuntimeWithWebSocketStream",
        "Resource": _endpoint_resources(rt_lid)}]
    trust = role["AssumeRolePolicyDocument"]["Statement"]
    assert trust == [{
        "Action": "sts:AssumeRole", "Effect": "Allow",
        "Principal": {"AWS": {"Fn::Join": ["", ["arn:", {"Ref": "AWS::Partition"}, f":iam::{ACCOUNT}:root"]]}},
        "Condition": {"ArnEquals": {"aws:PrincipalArn": {"Fn::Join": ["", [
            "arn:", {"Ref": "AWS::Partition"}, f":iam::{ACCOUNT}:role/{cfg.frontdoor_role_name}"]]}}}}]


def test_parts_name_r_and_e(built):
    _, parts = built["vpc"]
    assert parts.invoke_resources == [parts.runtime_arn, parts.endpoint_arn]
    assert parts.endpoint_arn.endswith("/runtime-endpoint/DEFAULT")


# --- execution role ------------------------------------------------------------------

def _fargate_task_statements(built):
    tpl = built["templates"]["OwlBankBridge"]
    cfg = built["stacks"]["compute"].cfg
    return _statements(_policies_of(tpl, _role_lid(tpl, cfg.bridge_task_role_name)))


def test_fargate_task_role_still_has_the_application_statements(built):
    sids = {s["Sid"] for s in _fargate_task_statements(built)}
    assert {"InvokeBankTools", "TokenUnseal", "TablesRead", "TablesReadWrite", "TablesKmsViaDynamoDB",
            "DetectPiiForRedaction", "StrandsGateway", "StrandsMemory", "StrandsMemoryCmk"} <= sids
    assert not sids & ADDED_SIDS


def test_execution_role_is_task_role_plus_listed_additions(built, mode):
    tpl, _ = built[mode]
    cfg = built["stacks"]["compute"].cfg
    rt_lid, _ = _one(tpl, "AWS::BedrockAgentCore::Runtime")
    lid = _role_lid(tpl, cfg.bridge_task_role_name)
    role = tpl["Resources"][lid]["Properties"]
    assert "ManagedPolicyArns" not in role and "Policies" not in role
    policies = _policies_of(tpl, lid)
    assert len(policies) == 2  # the default policy and the log-group policy
    stmts = _statements(policies)
    task = _fargate_task_statements(built)
    added = [s for s in stmts if s.get("Sid") in ADDED_SIDS | {"RuntimeLogStreams", "RuntimeLogGroup"}]
    rest = [s for s in stmts if s not in added]
    assert sorted(map(json.dumps, rest)) == sorted(map(json.dumps, task))
    by = _by_sid(added)
    assert sorted(by) == sorted(ADDED_SIDS | {"RuntimeLogStreams", "RuntimeLogGroup"})
    assert all(len(v) == 1 for v in by.values())
    by1 = {k: v[0] for k, v in by.items()}
    assert all(s["Effect"] == "Allow" for s in added)

    secrets = by1["ReadTwilioSecrets"]
    assert secrets["Action"] == "secretsmanager:GetSecretValue"
    assert len(secrets["Resource"]) == 2 and all("Fn::ImportValue" in r for r in secrets["Resource"])
    assert "Condition" not in secrets

    decrypt = by1["DecryptSecretsViaSecretsManager"]
    assert decrypt["Action"] == "kms:Decrypt"
    assert decrypt["Condition"] == {"StringEquals": {"kms:ViaService": f"secretsmanager.{REGION}.amazonaws.com"}}

    pull = by1["PullBridgeImage"]
    assert pull["Action"] == ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"]
    assert pull["Resource"] == {"Fn::Join": ["", [
        "arn:", {"Ref": "AWS::Partition"}, f":ecr:{REGION}:{ACCOUNT}:repository/{cfg.image_repo_name}"]]}

    assert by1["EcrAuthToken"]["Action"] == "ecr:GetAuthorizationToken"
    assert by1["EcrAuthToken"]["Resource"] == "*" and "Condition" not in by1["EcrAuthToken"]

    metrics = by1["RuntimeMetrics"]
    assert metrics["Action"] == "cloudwatch:PutMetricData" and metrics["Resource"] == "*"
    assert metrics["Condition"] == {"StringEquals": {"cloudwatch:namespace": "bedrock-agentcore"}}

    group_arn = ["arn:", {"Ref": "AWS::Partition"}, f":logs:{REGION}:{ACCOUNT}:log-group:/aws/bedrock-agentcore/runtimes/",
                 {"Fn::GetAtt": [rt_lid, "AgentRuntimeId"]}, "-DEFAULT"]
    streams = by1["RuntimeLogStreams"]
    assert streams["Action"] == ["logs:CreateLogStream", "logs:PutLogEvents"]
    assert streams["Resource"] == {"Fn::Join": ["", [*group_arn[:-1], "-DEFAULT:log-stream:*"]]}
    group = by1["RuntimeLogGroup"]
    assert group["Action"] == ["logs:DescribeLogStreams", "logs:CreateLogGroup"]
    assert group["Resource"] == {"Fn::Join": ["", group_arn]}


def test_execution_role_trust(built, mode):
    tpl, _ = built[mode]
    cfg = built["stacks"]["compute"].cfg
    trust = tpl["Resources"][_role_lid(tpl, cfg.bridge_task_role_name)]["Properties"]["AssumeRolePolicyDocument"]
    assert trust["Statement"] == [{
        "Action": "sts:AssumeRole", "Effect": "Allow",
        "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
        "Condition": {"StringEquals": {"aws:SourceAccount": ACCOUNT},
                      "ArnLike": {"aws:SourceArn": {"Fn::Join": ["", [
                          "arn:", {"Ref": "AWS::Partition"}, f":bedrock-agentcore:{REGION}:{ACCOUNT}:*"]]}}}}]


# --- Across the test stacks -------------------------------------------------------------

def _policy_statements(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "Statement":
                yield from v if isinstance(v, list) else [v]
            yield from _policy_statements(v)
    elif isinstance(node, list):
        for v in node:
            yield from _policy_statements(v)


def test_no_star_principal_and_no_function_url(built, mode):
    tpl, _ = built[mode]
    for s in _policy_statements(tpl):
        p = s.get("Principal")
        if p == "*" or (isinstance(p, dict) and "*" in (p.get("AWS"), [p.get("AWS")])):
            assert s["Effect"] == "Deny", s
    assert not _of_type(tpl, "AWS::Lambda::Url")


def test_cdk_nag_clean_and_acks_have_reasons(built):
    path = built["outdir"] / "validation-report.json"
    if path.exists():
        report = next((p for p in json.loads(path.read_text())["pluginReports"]
                       if p["pluginName"] == "AwsSolutions"), None)
        if report is not None:
            errors = [v for v in report["violations"] if v["severity"] == "error"]
            assert not errors, [(v["ruleName"], v.get("violatingResources")) for v in errors]
    mine = [(p, r, why) for p, r, why in ACKS if p.startswith("RuntimeTest")]
    rules = {r for _, r, _ in mine}
    assert "AwsSolutions-IAM5[Resource::*]" in rules
    assert any(r.startswith("AwsSolutions-IAM5[Resource::arn:") and r.endswith("-DEFAULT:log-stream:*]")
               for r in rules), rules
    star = next(why for p, r, why in mine if r == "AwsSolutions-IAM5[Resource::*]")
    assert "ecr:GetAuthorizationToken" in star and "cloudwatch:PutMetricData" in star


# --- argument checks -------------------------------------------------------------------------------

def _light(vpc_mode: bool):
    """A stack with imported Data/Network resources: fast to synthesize, no cross-stack refs."""
    app = App()
    stack = Stack(app, "Light", env=Environment(account=ACCOUNT, region=REGION))
    cfg = from_context({}, base_dir=INFRA)
    key = kms.Key.from_key_arn(stack, "Key", f"arn:aws:kms:{REGION}:{ACCOUNT}:key/11111111-1111-1111-1111-111111111111")
    token = kms.Key.from_key_arn(stack, "Tok", f"arn:aws:kms:{REGION}:{ACCOUNT}:key/22222222-2222-2222-2222-222222222222")
    tables = {n: ddb.Table.from_table_arn(stack, f"T{n}", f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/{n}")
              for n in (*BRIDGE_TABLES_RO, *BRIDGE_TABLES_RW)}
    secrets = {n: sm.Secret.from_secret_complete_arn(
        stack, f"S{n}", f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:owlbank/dev/{n}-AbCdEf")
        for n in ("twilio_api_key", "twilio_auth_token")}
    vpc = (ec2.Vpc.from_vpc_attributes(stack, "Vpc", vpc_id="vpc-0123456789abcdef0",
                                       availability_zones=[f"{REGION}a"]) if vpc_mode else None)
    common = {
        "cfg": cfg, "names": RuntimeNames.bridge(cfg), "key": key, "token_key": token,
        "tables": tables, "secrets": secrets, "repo_name": cfg.image_repo_name,
        "image_digest": DIGEST,
        "bank_tools_arn": f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:owlbank-dev-bank-tools",
        "memory_arn": f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}:memory/m-1",
        "gateway_arn": f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}:gateway/g-1",
        "environment": {"AWS_REGION": REGION}, "idle_timeout_s": 900, "vpc": vpc}
    return stack, common


@pytest.mark.parametrize("override,msg", [
    ({"image_digest": "latest"}, "image_digest"),
    ({"image_digest": "sha256:" + "AB" * 32}, "image_digest"),
    ({"idle_timeout_s": 59}, "idle_timeout_s"),
    ({"idle_timeout_s": MAX_LIFETIME_S}, "idle_timeout_s"),
    ({"subnet_ids": []}, "at least one subnet"),
    ({"subnet_ids": ["subnet-1", "subnet-1"]}, "duplicates"),
    ({"environment": {"TWILIO_AUTH_TOKEN": "x"}}, "TWILIO_AUTH_TOKEN"),
    ({"environment": {"OWLBANK_HOSTING": "fargate"}}, "OWLBANK_HOSTING"),
    ({"environment": {"OWLBANK_SECRET_TWILIO_API_KEY": "arn"}}, "OWLBANK_SECRET_TWILIO_API_KEY"),
])
def test_bad_arguments_are_refused(override, msg):
    stack, common = _light(vpc_mode=True)
    with pytest.raises(ValueError, match=msg):
        build_runtime(stack, **{**common, "subnet_ids": ["subnet-a"], **override})


def test_public_mode_refuses_subnets():
    stack, common = _light(vpc_mode=False)
    with pytest.raises(ValueError, match="without a vpc"):
        build_runtime(stack, **common, subnet_ids=["subnet-a"])


# --- property: NetworkConfiguration names exactly the given subnets; the SG never gains ingress ----

SUBNET_IDS = st.lists(st.from_regex(r"subnet-[0-9a-f]{8,17}", fullmatch=True),
                      min_size=1, max_size=4, unique=True)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(subnets=SUBNET_IDS, vpc_mode=st.booleans())
def test_network_configuration_property(subnets, vpc_mode):
    stack, common = _light(vpc_mode=vpc_mode)
    build_runtime(stack, **common, subnet_ids=subnets if vpc_mode else [])
    tpl = Template.from_stack(stack).to_json()
    net = _one(tpl, "AWS::BedrockAgentCore::Runtime")[1]["Properties"]["NetworkConfiguration"]
    sgs = _of_type(tpl, "AWS::EC2::SecurityGroup")
    assert not _of_type(tpl, "AWS::EC2::SecurityGroupIngress")
    if vpc_mode:
        assert net["NetworkMode"] == "VPC"
        assert net["NetworkModeConfig"]["Subnets"] == subnets
        (sg_lid, sg), = sgs.items()
        assert net["NetworkModeConfig"]["SecurityGroups"] == [{"Fn::GetAtt": [sg_lid, "GroupId"]}]
        assert "SecurityGroupIngress" not in sg["Properties"]
        assert [(e["IpProtocol"], e["FromPort"], e["ToPort"]) for e in sg["Properties"]["SecurityGroupEgress"]] \
            == [("tcp", 443, 443)]
    else:
        assert net == {"NetworkMode": "PUBLIC"} and not sgs
