"""The project AWS defaults, asserted over every synthesized template.

The mode-independent checks run over ``synth_matrix`` (conftest ``MATRIX``): the four Fargate
context sets and ``agentcore`` x {vpc, public} x {dev, prod-with-ack}. Checks of resources only one hosting has branch on
``synth.cfg.is_agentcore``; those of the ALB,
target group and ECS tasks stay on the Fargate ``synth_any``.
"""

from __future__ import annotations

import json
import re

import pytest

DATA = "OwlBankData"


# --------------------------------------------------------------------------- helpers

def _walk(node, path=()):
    yield path, node
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, path + (i,))


def _logical_id(synth, stack: str, construct_id: str, rtype: str) -> str:
    ids = [lid for s, lid, r in synth.resources(rtype) if s == stack and lid.startswith(construct_id)]
    assert len(ids) == 1, (construct_id, ids)
    return ids[0]


def _key_refs(synth, construct_id: str) -> list:
    """Every template expression that means "the ARN or ID of this Data-stack key"."""
    lid = _logical_id(synth, DATA, construct_id, "AWS::KMS::Key")
    local = [{"Fn::GetAtt": [lid, "Arn"]}, {"Ref": lid}]
    exports = [o["Export"]["Name"] for o in synth.templates[DATA].get("Outputs", {}).values()
               if "Export" in o and o["Value"] in local]
    return local + [{"Fn::ImportValue": name} for name in exports]


def _cmk_refs(synth) -> list:
    return _key_refs(synth, "DataKey")


def _statements(synth):
    """(stack, logical id, statement) for every IAM statement anywhere (policies, key policies...)."""
    for stack, lid, res in synth.resources():
        for _, node in _walk(res.get("Properties", {})):
            if isinstance(node, dict) and "Statement" in node and isinstance(node["Statement"], list):
                for st in node["Statement"]:
                    yield stack, lid, st


def _policies_of_role(synth, stack: str, role_prefix: str) -> list[dict]:
    role = _logical_id(synth, stack, role_prefix, "AWS::IAM::Role")
    out = []
    for s, _, res in synth.resources("AWS::IAM::Policy"):
        if s == stack and {"Ref": role} in res["Properties"].get("Roles", []):
            out += res["Properties"]["PolicyDocument"]["Statement"]
    return out



def _is_tls_only_deny(st) -> bool:
    """The "deny all non-TLS requests" statement (enforce_ssl on a bucket or topic).

    A Deny can only apply to everyone with a "*" principal; it grants nothing.
    """
    return (st.get("Effect") == "Deny"
            and st.get("Condition") == {"Bool": {"aws:SecureTransport": "false"}})


def _role_arn_join(name: str) -> dict:
    return {"Fn::Join": ["", ["arn:", {"Ref": "AWS::Partition"}, f":iam::111111111111:role/{name}"]]}


def _is_token_key_guard(synth, stack: str, lid: str, st) -> bool:
    """AWS-H1: the token key's Deny on crypto for everyone but the sealing roles.

    Matched exactly: on the TokenKey only, Deny, the four crypto actions, and a single
    ArnNotEquals aws:PrincipalArn condition naming exactly the fixed role names (the front door,
    the auth service, which seals the prepared welcome, the bridge task, and, when on, the recap
    writer, which unseals the call's factor values to scrub them).
    """
    if stack != DATA or lid != _logical_id(synth, DATA, "TokenKey", "AWS::KMS::Key"):
        return False
    stage = synth.stacks["data"].node.try_get_context("stage") or "dev"
    want = [_role_arn_join(f"owlbank-{stage}-frontdoor"), _role_arn_join(f"owlbank-{stage}-auth-service"),
            _role_arn_join(f"owlbank-{stage}-bridge-task")]
    if synth.cfg.recap_writer_on:
        want.append(_role_arn_join(f"owlbank-{stage}-recap-writer"))
    return (st.get("Effect") == "Deny"
            and st.get("Principal") == {"AWS": "*"}
            and sorted(st.get("Action", [])) == sorted(
                ["kms:Decrypt", "kms:Encrypt", "kms:ReEncrypt*", "kms:GenerateDataKey*"])
            and st.get("Condition") == {"ArnNotEquals": {"aws:PrincipalArn": want}})


#: The only Deny statements allowed a "*" principal. A new one needs a reviewed change here.
DENY_STAR_ALLOWLIST = {
    "tls-only deny (log bucket, alarm topic)": lambda synth, stack, lid, st, rtype:
        _is_tls_only_deny(st) and rtype in ("AWS::S3::BucketPolicy", "AWS::SNS::TopicPolicy"),
    "token key guard (AWS-H1)": lambda synth, stack, lid, st, rtype: _is_token_key_guard(synth, stack, lid, st),
}


def _has_star(p) -> bool:
    if p == "*":
        return True
    if isinstance(p, dict):
        return any(v == "*" or (isinstance(v, list) and "*" in v) for v in p.values())
    return isinstance(p, list) and "*" in p


def _statements_typed(synth):
    for stack, lid, res in synth.resources():
        for _, node in _walk(res.get("Properties", {})):
            if isinstance(node, dict) and "Statement" in node and isinstance(node["Statement"], list):
                for st in node["Statement"]:
                    yield stack, lid, res["Type"], st


def test_no_wildcard_principal(synth_matrix):
    """No Allow ever has a "*" principal. A Deny with "*" must be on the named allow-list."""
    for stack, lid, rtype, st in _statements_typed(synth_matrix):
        effect = st.get("Effect")
        assert effect in ("Allow", "Deny"), (stack, lid, st)
        if effect == "Allow":
            assert "NotPrincipal" not in st and "NotAction" not in st, (stack, lid, st)
            assert not _has_star(st.get("Principal")), (stack, lid, st)
            continue
        if _has_star(st.get("Principal")) or "NotPrincipal" in st:
            assert "NotPrincipal" not in st, (stack, lid, st)
            assert any(ok(synth_matrix, stack, lid, st, rtype) for ok in DENY_STAR_ALLOWLIST.values()), \
                ("Deny with '*' principal not on the allow-list", stack, lid, st)


def test_deny_allowlist_entries_are_used(synth_matrix):
    """Each allow-list entry matches something, so a stale entry can't linger."""
    for name, ok in DENY_STAR_ALLOWLIST.items():
        assert any(ok(synth_matrix, s, lid, st, t) for s, lid, t, st in _statements_typed(synth_matrix)), name


def test_no_policy_encoded_as_string(synth_matrix):
    """A JSON-encoded policy would hide from the statement walker."""
    for stack, lid, res in synth_matrix.resources():
        for path, node in _walk(res.get("Properties", {})):
            if isinstance(node, str):
                assert "Statement" not in node and "Principal" not in node, (stack, lid, path)


def test_lambda_permissions_scoped_to_source(synth_matrix):
    perms = list(synth_matrix.resources("AWS::Lambda::Permission"))
    assert perms
    for stack, lid, res in perms:
        p = res["Properties"]
        assert "SourceArn" in p, (stack, lid)
        assert "FunctionUrlAuthType" not in p and p["Principal"] != "*", (stack, lid)


#: Every resource type that carries a policy document today. A new type needs a review here.
POLICY_BEARING_TYPES = {"AWS::IAM::Role", "AWS::IAM::Policy", "AWS::KMS::Key", "AWS::S3::BucketPolicy",
                        "AWS::SNS::TopicPolicy"}
RESOURCE_POLICY_TYPES = {"AWS::KMS::Key", "AWS::S3::BucketPolicy", "AWS::SNS::TopicPolicy",
                         "AWS::SecretsManager::ResourcePolicy", "AWS::Logs::ResourcePolicy",
                         "AWS::ApiGateway::RestApi"}


def test_policy_bearing_types_are_known(synth_matrix):
    found = {t for _, _, t, _ in _statements_typed(synth_matrix)}
    # agentcore: no ALB, so no access-log bucket and no bucket policy.
    assert found == POLICY_BEARING_TYPES - ({"AWS::S3::BucketPolicy"} if synth_matrix.cfg.is_agentcore else set())


def test_resource_policies_have_no_star_allow(synth_matrix):
    for stack, lid, rtype, st in _statements_typed(synth_matrix):
        if rtype in RESOURCE_POLICY_TYPES and st["Effect"] == "Allow":
            assert not _has_star(st.get("Principal")), (stack, lid, st)
            assert "Principal" in st, (stack, lid, st)
    for rtype in ("AWS::SecretsManager::ResourcePolicy", "AWS::Logs::ResourcePolicy"):
        assert not list(synth_matrix.resources(rtype)), rtype
    for _, lid, res in synth_matrix.resources("AWS::ApiGateway::RestApi"):
        assert "Policy" not in res["Properties"], lid


def test_no_lambda_function_urls(synth_matrix):
    assert not list(synth_matrix.resources("AWS::Lambda::Url"))
    for stack, lid, res in synth_matrix.resources("AWS::Lambda::Permission"):
        assert "FunctionUrlAuthType" not in res["Properties"], (stack, lid)
        assert res["Properties"]["Principal"] != "*", (stack, lid)


BPA_ALL_ON = {"BlockPublicAcls": True, "BlockPublicPolicy": True,
              "IgnorePublicAcls": True, "RestrictPublicBuckets": True}


def _assert_bucket_is_private(synth):
    """What holds in every ``s3PublicAccessBlockMode``: ownership enforced, no ACL, no website,
    encryption, an SSL-only Deny, and no Allow for a ``*`` principal. Returns the buckets."""
    buckets = list(synth.resources("AWS::S3::Bucket"))
    assert buckets, "expected the ALB access-log bucket"
    for stack, lid, res in buckets:
        props = res["Properties"]
        assert "AccessControl" not in props, (stack, lid)
        assert props["OwnershipControls"]["Rules"] == [{"ObjectOwnership": "BucketOwnerEnforced"}]
        assert "WebsiteConfiguration" not in props
        sse = props["BucketEncryption"]["ServerSideEncryptionConfiguration"]
        assert sse and all("ServerSideEncryptionByDefault" in r for r in sse), (stack, lid)
    # The exact set of buckets: a new bucket needs a reviewed change here.
    assert [(s, lid[:len("AlbLogs")]) for s, lid, _ in buckets] == [("OwlBankBridge", "AlbLogs")]
    policies = list(synth.resources("AWS::S3::BucketPolicy"))
    assert len(policies) == 1
    for stack, lid, res in policies:
        statements = res["Properties"]["PolicyDocument"]["Statement"]
        assert any(_is_tls_only_deny(st) for st in statements), (stack, lid)
        for st in statements:
            if st["Effect"] == "Deny":
                assert _is_tls_only_deny(st), st
                continue
            assert not _has_star(st.get("Principal")), (stack, lid, st)
            assert _is_allowed_log_delivery(st), (stack, lid, st)
    return buckets


def test_no_public_s3(synth_matrix):
    """``explicit`` (the default): every bucket also sets all four Block Public Access flags;
    ``account``: none does (the next test checks the rest). ``agentcore``: the ALB access-log
    bucket was the call path's only bucket, so there is none."""
    if synth_matrix.cfg.is_agentcore:
        assert not list(synth_matrix.resources("AWS::S3::Bucket"))
        assert not list(synth_matrix.resources("AWS::S3::BucketPolicy"))
        return
    for stack, lid, res in _assert_bucket_is_private(synth_matrix):
        if synth_matrix.bpa_mode == "account":  # the account setting holds it (next test)
            assert "PublicAccessBlockConfiguration" not in res["Properties"], (stack, lid)
        else:
            assert res["Properties"]["PublicAccessBlockConfiguration"] == BPA_ALL_ON, (stack, lid)


def test_no_public_s3_account_mode(synth_bpa_account):
    """``account`` (an SCP denies s3:PutBucketPublicAccessBlock): the property is absent, everything
    else that keeps the bucket private is still there. Preflight --post-deploy reads the live result."""
    assert synth_bpa_account.bpa_mode == "account"
    for stack, lid, res in _assert_bucket_is_private(synth_bpa_account):
        assert "PublicAccessBlockConfiguration" not in res["Properties"], (stack, lid)
    # No stack sets Block Public Access some other way (a custom resource or a second bucket type).
    assert "PublicAccessBlock" not in json.dumps(synth_bpa_account.templates)
    assert not list(synth_bpa_account.resources("AWS::S3::AccessPoint"))


def test_account_mode_changes_only_the_bucket_property(synth_default, synth_bpa_account):
    """The two modes differ in the bucket's Block Public Access property and nothing else."""
    def strip(tpls):
        out = json.loads(json.dumps(tpls))
        for tpl in out.values():
            for res in tpl.get("Resources", {}).values():
                if res["Type"] == "AWS::S3::Bucket":
                    res["Properties"].pop("PublicAccessBlockConfiguration", None)
        return out
    a, b = strip(synth_default.templates), strip(synth_bpa_account.templates)
    assert a.keys() == b.keys()
    for name in a:
        assert a[name]["Resources"] == b[name]["Resources"], name


#: us-east-1's regional ELB account (CDK RegionInfo). A cross-account principal: it sends
#: no aws:SourceAccount, so it is scoped by action and object path instead.
ELB_ACCOUNT = "127311923021"
LOG_DELIVERY_SERVICES = {"delivery.logs.amazonaws.com", "logdelivery.elasticloadbalancing.amazonaws.com"}


def _is_allowed_log_delivery(st) -> bool:
    p = st.get("Principal", {})
    path = json.dumps(st["Resource"])
    if p.get("Service") in LOG_DELIVERY_SERVICES:
        return (st["Condition"]["StringEquals"].get("aws:SourceAccount") == "111111111111"
                and st["Action"] in ("s3:PutObject", "s3:GetBucketAcl"))
    if "AWS" in p:
        return (f":iam::{ELB_ACCOUNT}:root" in json.dumps(p["AWS"]) and st["Action"] == "s3:PutObject"
                and "/alb/AWSLogs/111111111111/*" in path)
    return False


def test_tables_encrypted_with_cmk_with_pitr(synth_matrix):
    refs = _cmk_refs(synth_matrix)
    tables = list(synth_matrix.resources("AWS::DynamoDB::Table"))
    # Plus modelstats (always built, 6.3), which has no PITR on purpose.
    assert len(tables) == 8
    for stack, lid, res in tables:
        p = res["Properties"]
        assert p["SSESpecification"]["SSEEnabled"] is True and p["SSESpecification"]["SSEType"] == "KMS"
        assert p["SSESpecification"]["KMSMasterKeyId"] in refs, lid
        if lid.startswith("ModelstatsTable"):
            assert "PointInTimeRecoverySpecification" not in p, lid
            continue
        assert p["PointInTimeRecoverySpecification"]["PointInTimeRecoveryEnabled"] is True, lid


def test_ttl_on_expiring_tables(synth_matrix):
    by_pk = {}
    for _, lid, res in synth_matrix.resources("AWS::DynamoDB::Table"):
        by_pk[lid] = res["Properties"].get("TimeToLiveSpecification")
    ttl = {lid for lid, spec in by_pk.items() if spec and spec["Enabled"]}
    assert {re.match(r"[A-Za-z]+?Table", lid).group(0) for lid in ttl} == {
        "TokensTable", "CallsTable", "SegmentsTable", "RiskTable",
        "ModelstatsTable"}
    for lid in ttl:
        assert by_pk[lid]["AttributeName"] == "expires_at"


def test_secrets_encrypted_with_cmk_and_empty(synth_matrix):
    refs = _cmk_refs(synth_matrix)
    secrets = list(synth_matrix.resources("AWS::SecretsManager::Secret"))
    names = {res["Properties"]["Name"] for _, _, res in secrets}
    # Plus the Context API's own Restricted-key secret (contextApiRestrictedKey, on by default).
    twilio = {"owlbank/twilio/api-key", "owlbank/twilio/auth-token", "owlbank/twilio/basic-auth",
              "owlbank/twilio/context-api-key"}
    # CloudFront mode: plus the generated origin-verify secret, in the Bridge stack.
    assert names == twilio | (set() if synth_matrix.domain_mode else {"owlbank/dev/bridge/origin-verify"})
    for stack, lid, res in secrets:
        assert res["Properties"]["KmsKeyId"] in refs, lid
        assert "SecretString" not in res["Properties"], f"{lid}: secret value in the template"


def test_every_log_group_encrypted_with_cmk(synth_matrix):
    refs = _cmk_refs(synth_matrix)
    groups = list(synth_matrix.resources("AWS::Logs::LogGroup"))
    assert len(groups) >= 9  # 6 Lambdas, flow logs, API access logs, bridge
    for stack, lid, res in groups:
        assert res["Properties"].get("KmsKeyId") in refs, (stack, lid)


def test_lambdas_have_explicit_encrypted_log_groups(synth_matrix):
    for stack, lid, res in synth_matrix.resources("AWS::Lambda::Function"):
        assert "LoggingConfig" in res["Properties"], lid


def test_waf_on_api_stage_and_alb(synth_matrix):
    """API stage: regional ACL. Custom domain: a regional ACL on the public ALB. CloudFront
    mode: no association on the ALB (it's internal); a CLOUDFRONT ACL is on the distribution."""
    assocs = {s: res["Properties"] for s, _, res in synth_matrix.resources("AWS::WAFv2::WebACLAssociation")}
    # agentcore: no ALB and no CloudFront (the Runtime endpoint is AWS-owned, 4.6), the API ACL only.
    alb_acl = synth_matrix.domain_mode and not synth_matrix.cfg.is_agentcore
    assert set(assocs) == ({"OwlBankApi", "OwlBankBridge"} if alb_acl else {"OwlBankApi"})
    stage_arn = json.dumps(assocs["OwlBankApi"]["ResourceArn"])
    stage = _logical_id(synth_matrix, "OwlBankApi", "ApiDeploymentStage", "AWS::ApiGateway::Stage")
    assert stage in stage_arn and "/stages/" in stage_arn
    if alb_acl:
        alb = _logical_id(synth_matrix, "OwlBankBridge", "Alb", "AWS::ElasticLoadBalancingV2::LoadBalancer")
        assert assocs["OwlBankBridge"]["ResourceArn"] == {"Ref": alb}
    for s, props in assocs.items():
        acl_lid = props["WebACLArn"]["Fn::GetAtt"][0]
        acl = synth_matrix.templates[s]["Resources"][acl_lid]["Properties"]
        names = {r["Name"] for r in acl["Rules"]}
        assert {"AllowedPathsAndMethodsOnly", "RateLimitPerIp", "BodySizeLimit",
                "AWSManagedRulesCommonRuleSet", "AWSManagedRulesKnownBadInputsRuleSet"} <= names
        assert acl["Scope"] == "REGIONAL"


_SECRETISH = re.compile(r"(TOKEN|SECRET|PASSWORD|API_KEY)", re.I)


def test_no_plaintext_secrets_in_task_definitions(synth_matrix):
    secret_lids = {lid for _, lid, _ in synth_matrix.resources("AWS::SecretsManager::Secret")}
    tds = list(synth_matrix.resources("AWS::ECS::TaskDefinition"))
    if synth_matrix.cfg.is_agentcore:
        assert not tds
        _assert_runtime_env_has_no_secret_values(synth_matrix)
        return
    assert len(tds) == 1
    for stack, lid, res in tds:
        for c in res["Properties"]["ContainerDefinitions"]:
            for e in c.get("Environment", []):
                # Names that look like credentials must be secrets, never environment values.
                if _SECRETISH.search(e["Name"]) and not e["Name"].startswith("OWLBANK_TABLE_"):
                    # A key ARN and an output-token count, not secrets.
                    assert e["Name"] in ("OWLBANK_TOKEN_KMS_KEY", "OWLBANK_MAX_TOKENS"), e["Name"]
            secrets = {s["Name"]: s["ValueFrom"] for s in c.get("Secrets", [])}
            assert {"TWILIO_AUTH_TOKEN", "TWILIO_API_KEY", "TWILIO_API_SECRET",
                    "TWILIO_ACCOUNT_SID"} == set(secrets)
            for name, value_from in secrets.items():
                text = json.dumps(value_from)
                assert "ImportValue" in text and any(s in text for s in secret_lids), (name, text)


def _assert_runtime_env_has_no_secret_values(synth):
    """agentcore: a Runtime has no ``Secrets`` injection, so the bridge reads the two Twilio secrets
    itself: the environment carries their ARNs, never a value, and no Twilio credential name."""
    ((_, _, rt),) = synth.resources("AWS::BedrockAgentCore::Runtime")
    env = rt["Properties"]["EnvironmentVariables"]
    for name, value in env.items():
        if not _SECRETISH.search(name) or name.startswith("OWLBANK_TABLE_"):
            continue
        if name.startswith("OWLBANK_SECRET_"):
            assert "Fn::ImportValue" in value, name  # a secret ARN from the Data stack
        else:
            assert name in ("OWLBANK_TOKEN_KMS_KEY", "OWLBANK_MAX_TOKENS"), name
    assert {n for n in env if n.startswith("OWLBANK_SECRET_")} == {
        "OWLBANK_SECRET_TWILIO_API_KEY", "OWLBANK_SECRET_TWILIO_AUTH_TOKEN"}
    for name in ("TWILIO_AUTH_TOKEN", "TWILIO_API_KEY", "TWILIO_API_SECRET", "TWILIO_ACCOUNT_SID"):
        assert name not in env, name


def test_no_secret_values_anywhere(synth_matrix):
    """No template carries anything that looks like a Twilio credential value."""
    blob = json.dumps(synth_matrix.templates)
    assert not re.search(r"\bSK[0-9a-f]{32}\b", blob)
    assert not re.search(r'"auth_token"\s*:\s*"[^"]', blob)


def test_lambda_env_has_no_secret_values(synth_matrix):
    for _, lid, res in synth_matrix.resources("AWS::Lambda::Function"):
        env = res["Properties"].get("Environment", {}).get("Variables", {})
        for name, value in env.items():
            if name.startswith("OWLBANK_SECRET_"):
                assert isinstance(value, dict), f"{lid}.{name} must be an ARN reference"


# --------------------------------------------------------------------------- Exposure

def test_alb_https_only(synth_domain_any):
    """Custom-domain mode: HTTPS only on the ALB. (The CloudFront mode's ALB is internal, with
    one HTTP listener behind the VPC origin: see test_exposure_modes.py.)"""
    synth_any = synth_domain_any
    listeners = [r["Properties"] for _, _, r in synth_any.resources("AWS::ElasticLoadBalancingV2::Listener")]
    assert listeners and all(li["Protocol"] == "HTTPS" and li["Port"] == 443 for li in listeners)
    for li in listeners:
        assert li["SslPolicy"].startswith("ELBSecurityPolicy-TLS13")


def test_target_group_health_and_deregistration(synth_any):
    tgs = [r["Properties"] for _, _, r in synth_any.resources("AWS::ElasticLoadBalancingV2::TargetGroup")]
    assert len(tgs) == 1
    tg = tgs[0]
    assert tg["HealthCheckPath"] == "/health"
    attrs = {a["Key"]: a["Value"] for a in tg["TargetGroupAttributes"]}
    assert attrs["deregistration_delay.timeout_seconds"] == "300"


def test_tasks_private_and_reachable_only_from_alb(synth_any):
    svc = [r["Properties"] for _, _, r in synth_any.resources("AWS::ECS::Service")][0]
    net = svc["NetworkConfiguration"]["AwsvpcConfiguration"]
    assert net["AssignPublicIp"] == "DISABLED"
    task_sg = net["SecurityGroups"][0]["Fn::GetAtt"][0]
    ingress = synth_any.templates["OwlBankBridge"]["Resources"][task_sg]["Properties"].get("SecurityGroupIngress", [])
    rules = ingress + [r["Properties"] for _, _, r in synth_any.resources("AWS::EC2::SecurityGroupIngress")
                       if json.dumps(r["Properties"].get("GroupId")).find(task_sg) >= 0]
    assert rules and all("CidrIp" not in r and "SourceSecurityGroupId" in r for r in rules)


def test_task_egress_https_only(synth_any):
    tpl = synth_any.templates["OwlBankBridge"]["Resources"]
    sg = next(r for lid, r in tpl.items() if lid.startswith("TaskSg") and r["Type"] == "AWS::EC2::SecurityGroup")
    assert sg["Properties"]["SecurityGroupEgress"] == [
        {"CidrIp": "0.0.0.0/0", "Description": "HTTPS to Twilio and AWS APIs",
         "FromPort": 443, "IpProtocol": "tcp", "ToPort": 443}]


def _open_cidr_rules(synth):
    """(logical id, rule) for every security-group rule with an open (0.0.0.0/0 or ::/0) CIDR."""
    for _, lid, res in synth.resources():
        for _, node in _walk(res.get("Properties", {})):
            if isinstance(node, dict) and (node.get("CidrIp") == "0.0.0.0/0" or node.get("CidrIpv6") == "::/0") \
                    and "IpProtocol" in node:
                yield lid, res["Type"], node


def test_only_443_open_to_the_internet(synth_matrix):
    """No rule opens anything but 443 to the internet; the exact set depends on the mode.

    Custom domain: the one internet-open ingress rule is the public ALB's HTTPS (443), plus
    the tasks' HTTPS egress to Twilio and AWS APIs. CloudFront mode: nothing is open to the
    internet for ingress at all (the only public entry points, API Gateway and CloudFront, are
    not security groups); the only open rule is the tasks' HTTPS egress. agentcore: no ALB, so no
    open ingress in either domain mode; in ``vpc`` the Runtime's HTTPS egress.
    """
    for lid, _, node in _open_cidr_rules(synth_matrix):
        if "FromPort" in node:
            assert node["FromPort"] == 443 and node["ToPort"] == 443, (lid, node)

    def is_open(rule) -> bool:
        return rule.get("CidrIp") == "0.0.0.0/0" or rule.get("CidrIpv6") == "::/0"

    ingress = [(lid, rule) for _, lid, res in synth_matrix.resources("AWS::EC2::SecurityGroup")
               for rule in res["Properties"].get("SecurityGroupIngress", []) if is_open(rule)]
    ingress += [(lid, r["Properties"]) for _, lid, r in synth_matrix.resources("AWS::EC2::SecurityGroupIngress")
                if is_open(r["Properties"])]
    if synth_matrix.domain_mode and not synth_matrix.cfg.is_agentcore:
        assert len(ingress) == 1, ingress
        ((lid, rule),) = ingress
        assert lid.startswith("AlbSg") and rule["FromPort"] == 443 and rule["ToPort"] == 443
    else:
        assert ingress == [], ingress


def test_backend_lambdas_have_no_api_permissions(synth_matrix):
    """auth_service, bank_tools, level0_tools: no resource policy at all (IAM-invoked only). The
    model-failover poller's one permission is its schedule's (EventBridge, that rule only)."""
    perms = [r["Properties"] for _, _, r in synth_matrix.resources("AWS::Lambda::Permission")]
    schedule = [p for p in perms if p["Principal"] == "events.amazonaws.com"]
    assert len(schedule) == (1 if synth_matrix.cfg.failover_on else 0)
    for p in schedule:
        assert p["FunctionName"]["Fn::GetAtt"][0].startswith("ModelPollerFn")
        assert p["SourceArn"]["Fn::GetAtt"][0].startswith("ModelPollerSchedule")
    perms = [p for p in perms if p not in schedule]
    blob = json.dumps(perms)
    for name in ("AuthService", "BankTools", "Level0Tools"):
        assert f"{name}Fn" not in blob, name
    assert all(p["Principal"] == "apigateway.amazonaws.com" for p in perms)


def test_auth_token_readable_only_by_twilio_facing_roles(synth_matrix):
    """Only components that receive Twilio requests may read the Auth Token."""
    readers = set()
    for stack, lid, res in synth_matrix.resources("AWS::IAM::Policy"):
        for st in res["Properties"]["PolicyDocument"]["Statement"]:
            actions = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
            if "secretsmanager:GetSecretValue" in actions and "AuthToken" in json.dumps(st["Resource"]):
                readers.add(f"{stack}/{lid}")
    # Fargate: the ECS execution role injects the secret. agentcore: the Runtime execution role
    # (``TaskRole``) reads it itself.
    bridge = "TaskRole" if synth_matrix.cfg.is_agentcore else "TaskDefExecutionRole"
    assert len(readers) == 2 and any("Frontdoor" in r for r in readers) \
        and any(f"OwlBankBridge/{bridge}" in r for r in readers), readers


def test_custom_domain_disables_execute_api(synth_domain, synth_default):
    def api(s):
        return [r["Properties"] for _, _, r in s.resources("AWS::ApiGateway::RestApi")][0]
    assert api(synth_domain)["DisableExecuteApiEndpoint"] is True
    # CloudFront mode (no domain): execute-api stays on, the only HTTPS name for the API.
    assert api(synth_default).get("DisableExecuteApiEndpoint") in (None, False)
    domains = [r["Properties"] for _, _, r in synth_domain.resources("AWS::ApiGateway::DomainName")]
    assert domains[0]["DomainName"] == "voice.owlbank.example.com"
    assert domains[0]["SecurityPolicy"] == "TLS_1_2"


def test_flex_context_route_uses_authorizer(synth_matrix):
    auths = [r["Properties"] for _, _, r in synth_matrix.resources("AWS::ApiGateway::Authorizer")]
    assert len(auths) == 1 and auths[0]["Type"] == "REQUEST"
    assert auths[0]["IdentitySource"] == "method.request.header.X-Flex-Token"
    assert auths[0]["AuthorizerResultTtlInSeconds"] == 0
    methods = [r["Properties"] for _, _, r in synth_matrix.resources("AWS::ApiGateway::Method")]
    by_verb = {}
    for m in methods:
        by_verb.setdefault(m["HttpMethod"], []).append(m)
    assert all(m["AuthorizationType"] == "CUSTOM" for m in by_verb["GET"])
    # The seven voice routes (partial, enqueue-action), plus
    # taskrouter and transcription with the default
    # humanLegConversation=on; agentcore adds POST /voice/orchestrator.
    from owlbank_infra.api_stack import voice_routes
    assert len(voice_routes(synth_matrix.cfg)) == (9 if synth_matrix.cfg.flex_mode else 7)  # Flex adds both
    assert len(by_verb["POST"]) == len(voice_routes(synth_matrix.cfg)) + synth_matrix.cfg.is_agentcore
    responses = [r["Properties"] for _, _, r in synth_matrix.resources("AWS::ApiGateway::GatewayResponse")]
    assert {r["ResponseType"] for r in responses} == {"UNAUTHORIZED", "ACCESS_DENIED"}
    for r in responses:
        assert r["ResponseParameters"]["gatewayresponse.header.Access-Control-Allow-Origin"] == \
            "'https://flex.twilio.com'"


# --------------------------------------------------------------------------- token key

def test_token_key_grants(synth_matrix):
    refs = _key_refs(synth_matrix, "TokenKey")
    users = {}
    for stack, lid, res in synth_matrix.resources("AWS::IAM::Policy"):
        for st in res["Properties"]["PolicyDocument"]["Statement"]:
            if st["Resource"] in refs or (isinstance(st["Resource"], list) and any(r in refs for r in st["Resource"])):
                actions = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
                assert st["Condition"] == {"ForAnyValue:StringEquals": {"kms:EncryptionContextKeys": ["call_sid"]}}
                users[lid] = sorted(actions)
    assert len(users) == 3 + synth_matrix.cfg.recap_writer_on, users
    if synth_matrix.cfg.recap_writer_on:  # Unseal only
        assert next(v for k, v in users.items() if k.startswith("RecapWriterRole")) == ["kms:Decrypt"]
    fd = next(v for k, v in users.items() if k.startswith("FrontdoorRole"))
    auth = next(v for k, v in users.items() if k.startswith("AuthServiceRole"))
    task = next(v for k, v in users.items() if k.startswith("TaskRole"))
    assert fd == ["kms:Decrypt", "kms:Encrypt"] and auth == ["kms:Encrypt"] and task == ["kms:Decrypt"]
    key = synth_matrix.templates[DATA]["Resources"][_logical_id(synth_matrix, DATA, "TokenKey", "AWS::KMS::Key")]
    assert key["Properties"]["EnableKeyRotation"] is True


def _token_key_policy(synth) -> dict:
    key = synth.templates[DATA]["Resources"][_logical_id(synth, DATA, "TokenKey", "AWS::KMS::Key")]
    return {st["Sid"]: st for st in key["Properties"]["KeyPolicy"]["Statement"]}


CRYPTO = {"kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey", "kms:GenerateDataKeyWithoutPlaintext",
          "kms:ReEncryptFrom", "kms:ReEncryptTo", "kms:CreateGrant"}


def test_token_key_policy_limits_crypto_to_sealing_roles(synth_matrix):
    """The key policy, not just IAM, limits who can use the token key."""
    stage = synth_matrix.stacks["data"].node.try_get_context("stage") or "dev"
    fd, br = _role_arn_join(f"owlbank-{stage}-frontdoor"), _role_arn_join(f"owlbank-{stage}-bridge-task")
    auth = _role_arn_join(f"owlbank-{stage}-auth-service")
    stmts = _token_key_policy(synth_matrix)
    recap = synth_matrix.cfg.recap_writer_on
    assert set(stmts) == {"KeyAdministration", "FrontdoorSealUnseal", "AuthServiceSeal", "BridgeUnseal",
                          "DenyCryptoExceptSealingRoles", *(["RecapWriterUnseal"] if recap else [])}
    admin = stmts["KeyAdministration"]
    assert "root" in json.dumps(admin["Principal"]) and "Condition" not in admin
    admin_actions = set(admin["Action"])
    assert not admin_actions & CRYPTO and not any("*" in a for a in admin_actions), admin_actions
    call_sid = {"ForAnyValue:StringEquals": {"kms:EncryptionContextKeys": ["call_sid"]}}
    assert stmts["FrontdoorSealUnseal"]["Condition"] == {"ArnEquals": {"aws:PrincipalArn": fd}, **call_sid}
    assert sorted(stmts["FrontdoorSealUnseal"]["Action"]) == ["kms:Decrypt", "kms:Encrypt"]
    assert stmts["BridgeUnseal"]["Condition"] == {"ArnEquals": {"aws:PrincipalArn": br}, **call_sid}
    assert stmts["BridgeUnseal"]["Action"] == "kms:Decrypt"
    assert stmts["AuthServiceSeal"]["Condition"] == {"ArnEquals": {"aws:PrincipalArn": auth}, **call_sid}
    assert stmts["AuthServiceSeal"]["Action"] == "kms:Encrypt"
    if recap:  # The recap writer unseals the call's factor values
        rw = _role_arn_join(f"owlbank-{stage}-recap-writer")
        assert stmts["RecapWriterUnseal"]["Condition"] == {"ArnEquals": {"aws:PrincipalArn": rw}, **call_sid}
        assert stmts["RecapWriterUnseal"]["Action"] == "kms:Decrypt"
    deny = stmts["DenyCryptoExceptSealingRoles"]
    assert _is_token_key_guard(synth_matrix, DATA, _logical_id(synth_matrix, DATA, "TokenKey", "AWS::KMS::Key"), deny)


def test_sealing_roles_have_the_fixed_names(synth_matrix):
    stage = synth_matrix.stacks["data"].node.try_get_context("stage") or "dev"
    names = {r["Properties"].get("RoleName") for _, _, r in synth_matrix.resources("AWS::IAM::Role")}
    assert {f"owlbank-{stage}-frontdoor", f"owlbank-{stage}-auth-service", f"owlbank-{stage}-bridge-task"} <= names


def test_token_key_env(synth_matrix):
    fd = [r for _, lid, r in synth_matrix.resources("AWS::Lambda::Function") if lid.startswith("FrontdoorFn")][0]
    assert "OWLBANK_TOKEN_KMS_KEY" in fd["Properties"]["Environment"]["Variables"]


# --------------------------------------------------------------------------- least privilege

@pytest.mark.parametrize("role,expected", [
    ("FrontdoorRole", {"InvokeAuthService", "InvokeBankTools"}),
    ("AuthServiceRole", set()),
    ("ContextApiRole", set()),
])
def test_lambda_invoke_grants(synth_default, role, expected):
    sids = {st.get("Sid") for st in _policies_of_role(synth_default, "OwlBankCompute", role)}
    assert {s for s in sids if s and s.startswith("Invoke")} == expected


def _alias_ref(synth, prefix: str) -> dict:
    """``{"Ref": <logical id>}`` of a function's ``live`` alias (an alias Ref is its ARN)."""
    lid = _logical_id(synth, "OwlBankCompute", f"{prefix}FnAliaslive", "AWS::Lambda::Alias")
    return {"Ref": lid}


def test_frontdoor_invokes_the_live_aliases_exactly(synth_default):
    """The grants name the qualified alias ARNs (``...:live``), not the unqualified function
    ARNs, and never a ``:*`` pattern."""
    stmts = {st["Sid"]: st for st in _policies_of_role(synth_default, "OwlBankCompute", "FrontdoorRole")
             if str(st.get("Sid", "")).startswith("Invoke")}
    assert stmts["InvokeAuthService"]["Resource"] == _alias_ref(synth_default, "AuthService")
    assert stmts["InvokeBankTools"]["Resource"] == _alias_ref(synth_default, "BankTools")
    assert set(stmts) == {"InvokeAuthService", "InvokeBankTools"}
    for st in stmts.values():
        assert st["Action"] == "lambda:InvokeFunction" and isinstance(st["Resource"], dict)
        assert ":*" not in json.dumps(st["Resource"])


def test_gateway_role_invokes_level0_only(synth_default):
    stmts = _policies_of_role(synth_default, "OwlBankAgent", "GatewayRole")
    invokes = [st for st in stmts if "lambda:InvokeFunction" in json.dumps(st["Action"])]
    assert len(invokes) == 1 and ":function:owlbank-dev-level0-tools" in json.dumps(invokes[0]["Resource"])


def test_bridge_task_role(synth_default):
    stmts = _policies_of_role(synth_default, "OwlBankBridge", "TaskRole")
    blob = json.dumps(stmts)
    assert ":function:owlbank-dev-bank-tools" in blob and "bedrock-agentcore:InvokeHarness" not in blob
    assert "bedrock-agentcore:InvokeGateway" in blob  # the agent loop runs here
    invoke = [st for st in stmts if st.get("Action") == "lambda:InvokeFunction"]
    assert len(invoke) == 1 and invoke[0]["Sid"] == "InvokeBankTools"
    # Exactly the alias ARN (built as a string, no reference to Compute): no ":*", no
    # unqualified fallback.
    assert json.dumps(invoke[0]["Resource"]).endswith(':function:owlbank-dev-bank-tools:live"]]}')
    assert "secretsmanager:" not in blob  # the execution role injects secrets
    assert "level0" not in blob and "auth-service" not in blob and "AuthService" not in blob


def test_context_api_read_only(synth_default):
    stmts = _policies_of_role(synth_default, "OwlBankCompute", "ContextApiRole")
    actions = {a for st in stmts for a in (st["Action"] if isinstance(st["Action"], list) else [st["Action"]])}
    assert not {a for a in actions if a.startswith("dynamodb:") and
                any(w in a for w in ("Put", "Update", "Delete", "Write"))}
    # It reads exactly one secret (the Twilio key for the per-task
    # TaskRouter check), by its ARN; tests/test_human_leg_infra.py pins which one.
    secret_stmts = [st for st in stmts if any(str(a).startswith("secretsmanager:") for a in
                                              (st["Action"] if isinstance(st["Action"], list) else [st["Action"]]))]
    assert len(secret_stmts) == 1 and "*" not in json.dumps(secret_stmts[0]["Resource"])


def test_no_iam_wildcard_actions(synth_matrix):
    """No wildcard actions, except CDK's default key-policy root statement (kms:* for the account)."""
    keys = {lid for _, lid, _ in synth_matrix.resources("AWS::KMS::Key")}
    for stack, lid, st in _statements(synth_matrix):
        if lid in keys and st.get("Action") == "kms:*":
            assert "root" in json.dumps(st["Principal"]), (lid, st)
            continue
        if _is_tls_only_deny(st):
            continue
        actions = st.get("Action", [])
        actions = actions if isinstance(actions, list) else [actions]
        for a in actions:
            assert a != "*" and not a.endswith(":*"), (stack, lid, a)


def _risk_table_refs(synth) -> list[str]:
    """Logical ID and export names that mean the Data stack's risk table."""
    lid = _logical_id(synth, DATA, "RiskTable", "AWS::DynamoDB::Table")
    out = [lid]
    for o in synth.templates[DATA].get("Outputs", {}).values():
        if "Export" in o and lid in json.dumps(o["Value"]):
            out.append(o["Export"]["Name"])
    return out


def test_risk_table_is_reachable_only_by_auth_service(synth_matrix):
    """Only the auth_service role has any IAM statement naming the risk table,
    with read and write; the others only see its name in their environment."""
    refs = _risk_table_refs(synth_matrix)
    holders = set()
    for stack, lid, res in synth_matrix.resources("AWS::IAM::Policy"):
        for st in res["Properties"]["PolicyDocument"]["Statement"]:
            if any(r in json.dumps(st.get("Resource")) for r in refs):
                roles = res["Properties"].get("Roles", [])
                holders.update(json.dumps(r) for r in roles)
                acts = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
                assert {"dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"} <= set(acts)
    compute = "OwlBankCompute"
    auth_role = _logical_id(synth_matrix, compute, "AuthServiceRole", "AWS::IAM::Role")
    assert holders == {json.dumps({"Ref": auth_role})}, holders


def test_risk_table_settings(synth_matrix):
    lid = _logical_id(synth_matrix, DATA, "RiskTable", "AWS::DynamoDB::Table")
    res = synth_matrix.templates[DATA]["Resources"][lid]
    p = res["Properties"]
    assert p["KeySchema"] == [{"AttributeName": "risk_key", "KeyType": "HASH"}]
    assert p["BillingMode"] == "PAY_PER_REQUEST"
    assert p["SSESpecification"]["KMSMasterKeyId"] in _cmk_refs(synth_matrix)
    assert p["PointInTimeRecoverySpecification"]["PointInTimeRecoveryEnabled"] is True
    assert p["TimeToLiveSpecification"] == {"AttributeName": "expires_at", "Enabled": True}
    assert res["DeletionPolicy"] == "Retain" and res["UpdateReplacePolicy"] == "Retain"
    stage = synth_matrix.stacks["data"].node.try_get_context("stage") or "dev"
    assert p.get("DeletionProtectionEnabled", False) is (stage == "prod")


def test_auth_service_env_names_the_risk_table(synth_matrix):
    fn = [r for _, lid, r in synth_matrix.resources("AWS::Lambda::Function")
          if lid.startswith("AuthServiceFn")][0]
    assert "OWLBANK_TABLE_RISK" in fn["Properties"]["Environment"]["Variables"]
