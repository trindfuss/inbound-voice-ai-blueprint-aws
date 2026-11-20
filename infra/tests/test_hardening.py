"""The AWS best-practices review fixes (AWS-M2..M7, lows, new IAM), asserted on synth."""

from __future__ import annotations

import json

import pytest

from owlbank_infra.app_builder import build_app
from owlbank_infra.config import ConfigError

DATA = "OwlBankData"


def _stmts_of_role(synth, stack: str, role_prefix: str) -> list[dict]:
    roles = [lid for s, lid, _ in synth.resources("AWS::IAM::Role") if s == stack and lid.startswith(role_prefix)]
    assert len(roles) == 1, roles
    out = []
    for s, _, res in synth.resources("AWS::IAM::Policy"):
        if s == stack and {"Ref": roles[0]} in res["Properties"].get("Roles", []):
            out += res["Properties"]["PolicyDocument"]["Statement"]
    return out


def _actions(st) -> list[str]:
    return st["Action"] if isinstance(st["Action"], list) else [st["Action"]]


def _key_lid(synth, prefix: str) -> str:
    return next(lid for s, lid, _ in synth.resources("AWS::KMS::Key") if s == DATA and lid.startswith(prefix))


def _data_key_policy(synth) -> dict:
    res = synth.templates[DATA]["Resources"][_key_lid(synth, "DataKey")]
    return {st.get("Sid", ""): st for st in res["Properties"]["KeyPolicy"]["Statement"]}


# ------------------------------------------------------------------ AWS-M2

def _single_key_grant(synth, stack: str, role: str, sid: str) -> dict:
    """The role's one ``sid`` statement: exactly four named actions on exactly one key."""
    stmts = _stmts_of_role(synth, stack, role)
    kms = [st for st in stmts if st.get("Sid") == sid]
    if role == "GatewayRole":
        assert [st for st in stmts if any(a.startswith("kms:") for a in _actions(st))] == kms
    assert len(kms) == 1
    assert sorted(_actions(kms[0])) == ["kms:Decrypt", "kms:DescribeKey", "kms:Encrypt", "kms:GenerateDataKey"]
    res = kms[0]["Resource"]
    assert not isinstance(res, list) or len(res) == 1
    assert res != "*" and "*" not in json.dumps(res)
    return kms[0]


def test_gateway_cmk_grant_is_narrow_and_direct(synth_any):
    """AgentCore calls KMS directly as the Gateway's service role (verified at the first deploy: a
    ViaService condition never matches and Gateway creation fails), so that grant has no condition."""
    assert "Condition" not in _single_key_grant(synth_any, "OwlBankAgent", "GatewayRole", "GatewayCmk")


def test_bridge_memory_cmk_grant_is_only_through_agentcore(synth_any):
    """The bridge's task role runs application code, so its grant for the CMK-encrypted Memory
    holds only through the AgentCore service in this Region, never a direct KMS call."""
    st = _single_key_grant(synth_any, "OwlBankBridge", "TaskRole", "StrandsMemoryCmk")
    assert set(st["Condition"]) == {"StringEquals"}
    (via,) = st["Condition"]["StringEquals"].values()
    assert list(st["Condition"]["StringEquals"]) == ["kms:ViaService"]
    assert via == f"bedrock-agentcore.{synth_any.cfg.region}.amazonaws.com"


#: The one direct (no ViaService) data-key grant, see grants.grant_key_use: the Gateway's service
#: role (a verified direct caller).
AGENTCORE_DIRECT_KMS_SIDS = {"GatewayCmk"}


def test_every_data_key_identity_grant_has_via_service(synth_any):
    """No role may use the data CMK directly: always through a named service, except the Gateway's
    service role (AgentCore uses it directly), asserted narrowly above."""
    lid = _key_lid(synth_any, "DataKey")
    refs = [{"Fn::GetAtt": [lid, "Arn"]}]
    refs += [{"Fn::ImportValue": o["Export"]["Name"]} for o in synth_any.templates[DATA]["Outputs"].values()
             if "Export" in o and o["Value"] in refs]
    for stack, plid, res in synth_any.resources("AWS::IAM::Policy"):
        for st in res["Properties"]["PolicyDocument"]["Statement"]:
            resources = st["Resource"] if isinstance(st["Resource"], list) else [st["Resource"]]
            if any(r in refs for r in resources):
                if st.get("Sid") in AGENTCORE_DIRECT_KMS_SIDS:
                    continue
                assert "kms:ViaService" in st["Condition"]["StringEquals"], (stack, plid, st)


# ------------------------------------------------------------------ AWS-M3

def test_waf_logging_both_acls(synth_any):
    """Both web ACLs log, with credentials redacted: the API's, and the bridge's (the public
    ALB's in the custom-domain mode, the CloudFront distribution's in the CloudFront mode)."""
    stage = synth_any.stacks["data"].node.try_get_context("stage") or "dev"
    confs = {s: r["Properties"] for s, _, r in synth_any.resources("AWS::WAFv2::LoggingConfiguration")}
    assert set(confs) == {"OwlBankApi", "OwlBankBridge"}
    groups = {r["Properties"]["LogGroupName"]: r["Properties"]
              for _, _, r in synth_any.resources("AWS::Logs::LogGroup")}
    bridge_which = "alb" if synth_any.domain_mode else "cloudfront"
    for stack, which in (("OwlBankApi", "api"), ("OwlBankBridge", bridge_which)):
        c = confs[stack]
        name = f"aws-waf-logs-owlbank-{stage}-{which}"
        dest = json.dumps(c["LogDestinationConfigs"])
        assert dest.endswith(":log-group:" + name + '"]]}]'), dest  # no trailing ":*"
        assert name in groups and "KmsKeyId" in groups[name] and "RetentionInDays" in groups[name]
        redacted = c["RedactedFields"]
        assert {"SingleHeader": {"Name": "authorization"}} in redacted
        assert {"SingleHeader": {"Name": "cookie"}} in redacted
        assert {"SingleHeader": {"Name": "x-twilio-signature"}} in redacted
        assert ({"QueryString": {}} in redacted) == (which != "api")  # the WebSocket nonce
        acl = synth_any.templates[stack]["Resources"][c["ResourceArn"]["Fn::GetAtt"][0]]
        assert acl["Type"] == "AWS::WAFv2::WebACL"


# ------------------------------------------------------------------ AWS-M4

def test_alarm_topic_encrypted_without_subscriptions_by_default(synth_default, synth_prod):
    topics = [r["Properties"] for _, _, r in synth_default.resources("AWS::SNS::Topic")]
    assert len(topics) == 1
    assert topics[0]["KmsMasterKeyId"] == {"Fn::GetAtt": [_key_lid(synth_default, "DataKey"), "Arn"]}
    assert not list(synth_default.resources("AWS::SNS::Subscription"))
    subs = [r["Properties"] for _, _, r in synth_prod.resources("AWS::SNS::Subscription")]
    assert subs == [{"Endpoint": "oncall@example.com", "Protocol": "email",
                     "TopicArn": {"Ref": next(lid for _, lid, _ in synth_prod.resources("AWS::SNS::Topic"))}}]


def test_cloudwatch_may_use_cmk_for_the_topic_only_via_sns(synth_any):
    st = _data_key_policy(synth_any)["AllowCloudWatchAlarmsToEncryptedTopic"]
    assert st["Principal"] == {"Service": "cloudwatch.amazonaws.com"}
    assert sorted(st["Action"]) == ["kms:Decrypt", "kms:GenerateDataKey"]
    assert st["Condition"]["StringEquals"]["kms:ViaService"] == "sns.us-east-1.amazonaws.com"


def _alarms(synth, stack=None):
    return [(s, r["Properties"]) for s, _, r in synth.resources("AWS::CloudWatch::Alarm") if stack in (None, s)]


def test_every_alarm_notifies_the_topic(synth_any):
    alarms = _alarms(synth_any)
    assert alarms
    for stack, a in alarms:
        assert len(a["AlarmActions"]) == 1, (stack, a)
        text = json.dumps(a["AlarmActions"])
        assert "AlarmTopic" in text, (stack, text)


def test_required_alarms_exist(synth_any):
    def metrics(stack):
        return {(a.get("Namespace"), a.get("MetricName")) for _, a in _alarms(synth_any, stack)}
    compute = _alarms(synth_any, "OwlBankCompute")
    for metric in ("Errors", "Throttles"):
        fns = {json.dumps(a["Dimensions"]) for _, a in compute if a.get("MetricName") == metric}
        # The six functions, the model-failover poller when on (the default list), and the human-leg
        # conversation sweep when on (its own 30-minute alarms, test_human_leg_infra), and the recap
        # writer when on.
        assert len(fns) == 6 + (1 if synth_any.cfg.failover_on else 0) + (
            1 if synth_any.cfg.human_leg_conversation_on else 0) + (
            1 if synth_any.cfg.recap_writer_on else 0), (metric, fns)
    stage = synth_any.stacks["data"].node.try_get_context("stage") or "dev"
    sec = f"OwlBank/{stage}/Security"
    assert (sec, "FrontdoorRejected") in metrics("OwlBankCompute")
    assert {("AWS/ApiGateway", "5XXError"), ("AWS/WAFV2", "BlockedRequests")} <= metrics("OwlBankApi")
    assert {("AWS/ApplicationELB", "HTTPCode_ELB_5XX_Count"), ("AWS/ApplicationELB", "HTTPCode_Target_5XX_Count"),
            ("AWS/ApplicationELB", "UnHealthyHostCount"), ("ECS/ContainerInsights", "RunningTaskCount"),
            ("AWS/WAFV2", "BlockedRequests"), (sec, "BridgeWsRejected")} <= metrics("OwlBankBridge")


def test_running_task_alarm_tracks_desired_count(synth_default, synth_domain):
    for synth, want in ((synth_default, 1), (synth_domain, 2)):
        a = next(a for _, a in _alarms(synth, "OwlBankBridge") if a.get("MetricName") == "RunningTaskCount")
        assert a["Threshold"] == want and a["ComparisonOperator"] == "LessThanThreshold"
        assert a["TreatMissingData"] == "breaching"


def test_waf_blocked_alarm_uses_acl_names(synth_any):
    stage = synth_any.stacks["data"].node.try_get_context("stage") or "dev"
    names = {r["Properties"]["Name"] for _, _, r in synth_any.resources("AWS::WAFv2::WebACL")}
    bridge_acl = f"owlbank-{stage}-alb" if synth_any.domain_mode else f"owlbank-{stage}-cloudfront"
    assert names == {f"owlbank-{stage}-api", bridge_acl}
    dims = [{d["Name"]: d["Value"] for d in a["Dimensions"]} for _, a in _alarms(synth_any)
            if a.get("MetricName") == "BlockedRequests"]
    assert sorted(d["WebACL"] for d in dims) == sorted(names)
    # A CLOUDFRONT-scope ACL reports its metrics with Region=Global; a REGIONAL one with its Region.
    want_region = {f"owlbank-{stage}-api": "us-east-1", bridge_acl: "us-east-1" if synth_any.domain_mode else "Global"}
    assert all(d["Rule"] == "ALL" and d["Region"] == want_region[d["WebACL"]] for d in dims)


def test_auth_failure_metric_filters_match_source_log_lines(synth_default):
    """Both filters match the stable event the code emits (src/owlbank/security_log.py)."""
    from owlbank_infra.contract import REPO_ROOT
    src = (REPO_ROOT / "src" / "owlbank" / "security_log.py").read_text()
    assert 'AUTH_FAILED_EVENT = "twilio_auth_failed"' in src
    filters = {s: r["Properties"] for s, _, r in synth_default.resources("AWS::Logs::MetricFilter")
               if r["Properties"]["FilterPattern"] == '"twilio_auth_failed"'}
    assert set(filters) == {"OwlBankCompute", "OwlBankBridge"}
    for f in filters.values():
        assert f["MetricTransformations"][0]["DefaultValue"] == 0
    assert "Frontdoor" in json.dumps(filters["OwlBankCompute"]["LogGroupName"])
    assert "BridgeLogs" in json.dumps(filters["OwlBankBridge"]["LogGroupName"])


def test_basic_challenge_event_matches_no_metric_filter(synth_default):
    """Every Twilio webhook starts with a request without credentials, answered 401 and logged as
    ``twilio_basic_challenge`` (INFO). No metric filter may match that line, or the auth-failure
    alarm would fire on every call. The filters are literal quoted terms."""
    from owlbank_infra.contract import REPO_ROOT
    src = (REPO_ROOT / "src" / "owlbank" / "security_log.py").read_text()
    assert 'BASIC_CHALLENGE_EVENT = "twilio_basic_challenge"' in src
    lines = ['{"event":"twilio_basic_challenge","component":"frontdoor","reason":"basic","challenge":true}',
             "twilio_basic_challenge",
             '{"event":"twilio_signature_form","component":"frontdoor","form":"credentials"}']
    patterns = [r["Properties"]["FilterPattern"] for _, _, r in synth_default.resources("AWS::Logs::MetricFilter")]
    assert patterns
    for pattern in patterns:
        if pattern.startswith("{"):
            # Model failover's JSON filters match one other ``event`` value exactly.
            assert '$.event = "model_unavailable"' in pattern and "twilio_basic_challenge" not in pattern, pattern
            continue
        assert pattern.startswith('"') and pattern.endswith('"'), pattern  # a literal term, not a JSON filter
        assert not any(pattern.strip('"') in line for line in lines), pattern


def test_otp_untrusted_ani_filter_and_alarm(synth_default):
    """The auth service's untrusted-ANI code line is filtered and alarmed on."""
    from owlbank_infra.contract import REPO_ROOT
    src = (REPO_ROOT / "src" / "owlbank" / "security_log.py").read_text()
    assert 'OTP_UNTRUSTED_EVENT = "otp_sent_untrusted_ani"' in src
    filters = [(s, r["Properties"]) for s, _, r in synth_default.resources("AWS::Logs::MetricFilter")
               if r["Properties"]["FilterPattern"] == '"otp_sent_untrusted_ani"']
    (stack, f), = filters
    assert stack == "OwlBankCompute"
    assert "AuthService" in json.dumps(f["LogGroupName"])
    mt = f["MetricTransformations"][0]
    assert mt["MetricName"] == "OtpSentUntrustedAni" and mt["DefaultValue"] == 0
    alarms = [a for _, a in _alarms(synth_default, "OwlBankCompute")
              if a.get("MetricName") == "OtpSentUntrustedAni"]
    (a,) = alarms
    assert a["Threshold"] == 1 and a["Statistic"] == "Sum" and a["Period"] == 300
    assert a["TreatMissingData"] == "notBreaching"
    assert "AlarmTopic" in json.dumps(a["AlarmActions"])  # the CMK-encrypted SNS topic


def test_otp_keyed_phone_is_counted_without_an_alarm(synth_default):
    """A code sent on a keyed-in number's call is counted
    (``OtpSentKeyedPhone``), never alarmed, and never matches the untrusted-ANI filter."""
    from owlbank_infra.contract import REPO_ROOT
    src = (REPO_ROOT / "src" / "owlbank" / "security_log.py").read_text()
    assert 'OTP_KEYED_PHONE_EVENT = "otp_sent_keyed_phone"' in src
    (stack, f), = [(s, r["Properties"]) for s, _, r in synth_default.resources("AWS::Logs::MetricFilter")
                   if r["Properties"]["FilterPattern"] == '"otp_sent_keyed_phone"']
    assert stack == "OwlBankCompute" and "AuthService" in json.dumps(f["LogGroupName"])
    mt = f["MetricTransformations"][0]
    assert mt["MetricName"] == "OtpSentKeyedPhone" and mt["DefaultValue"] == 0
    assert not [a for _, a in _alarms(synth_default, "OwlBankCompute")
                if a.get("MetricName") == "OtpSentKeyedPhone"]
    assert "otp_sent_untrusted_ani" not in '{"event":"otp_sent_keyed_phone","component":"auth_service"}'


# ------------------------------------------------------------------ AWS-M5

def _rate(synth, stack):
    acl = next(r for s, _, r in synth.resources("AWS::WAFv2::WebACL") if s == stack)
    rule = next(r for r in acl["Properties"]["Rules"] if r["Name"] == "RateLimitPerIp")
    return rule["Statement"]["RateBasedStatement"]["Limit"]


def test_api_waf_rate_limit_is_high_and_configurable(synth_default, synth_domain, tmp_path):
    assert _rate(synth_default, "OwlBankApi") == 2000  # Sandbox default, raise after a load test
    assert _rate(synth_default, "OwlBankBridge") == 10000  # CloudFront ACL: high, like the API's
    assert _rate(synth_domain, "OwlBankBridge") == 1000  # the public ALB's ACL, unchanged
    for ctx, want in (({"wafApiRateLimit": 20000, "wafCloudFrontRateLimit": 4000},
                       {"OwlBankApi": 20000, "OwlBankBridge": 4000}),
                      ({"wafApiRateLimit": 20000, "wafAlbRateLimit": 3000, "domainName": "owlbank.example.com",
                        "certificateArn": "arn:aws:acm:us-east-1:111111111111:certificate/11111111-2222-3333-4444-555555555555"},
                       {"OwlBankApi": 20000, "OwlBankBridge": 3000})):
        app, _ = build_app({**ctx, "adminConsole": "false"},  # the admin ACLs: test_admin_stack.py
                           outdir=str(tmp_path / str(len(want)) / str(want["OwlBankBridge"])))
        tpl = {s.stack_name: s.template for s in app.synth().stacks}
        limits = {s: next(r for r in res["Properties"]["Rules"] if r["Name"] == "RateLimitPerIp")
                  ["Statement"]["RateBasedStatement"]["Limit"]
                  for s, t in tpl.items() for res in t["Resources"].values() if res["Type"] == "AWS::WAFv2::WebACL"}
        assert limits == want


def test_api_stage_throttling_is_the_volume_control(synth_any):
    stage = next(r for _, _, r in synth_any.resources("AWS::ApiGateway::Stage"))["Properties"]
    ms = stage["MethodSettings"][0]
    assert ms["ThrottlingRateLimit"] == 50 and ms["ThrottlingBurstLimit"] == 100


# ------------------------------------------------------------------ AWS-M6

def test_dynamodb_gateway_endpoint_on_private_route_tables(synth_any):
    eps = [r["Properties"] for _, _, r in synth_any.resources("AWS::EC2::VPCEndpoint")]
    assert len(eps) == 1 and eps[0]["VpcEndpointType"] == "Gateway"
    assert "dynamodb" in json.dumps(eps[0]["ServiceName"])
    net = synth_any.templates["OwlBankNetwork"]["Resources"]
    private_rts = {lid for lid, r in net.items() if r["Type"] == "AWS::EC2::RouteTable" and "private" in lid}
    assert len(private_rts) == 2
    assert {rt["Ref"] for rt in eps[0]["RouteTableIds"]} == private_rts



def test_every_log_group_has_a_cmk_and_retention_and_every_lambda_one_of_them(synth_matrix):
    """No log group is left to Lambda to create (no CMK, no expiry): every template's groups carry
    the CMK and a retention, and every function logs to one of them."""
    groups = {(s, lid): r["Properties"] for s, lid, r in synth_matrix.resources("AWS::Logs::LogGroup")}
    assert groups
    for key, g in groups.items():
        assert "KmsKeyId" in g and "RetentionInDays" in g, key
    for s, lid, r in synth_matrix.resources("AWS::Lambda::Function"):
        assert (s, r["Properties"]["LoggingConfig"]["LogGroup"]["Ref"]) in groups, (s, lid)


# ------------------------------------------------------------------ AWS-M7

def test_prod_hardening(synth_prod):
    groups = [r for _, _, r in synth_prod.resources("AWS::Logs::LogGroup")]
    assert groups
    for g in groups:
        assert g["DeletionPolicy"] == "Retain" and g["Properties"]["RetentionInDays"] == 365
    for _, lid, r in synth_prod.resources("AWS::DynamoDB::Table"):
        assert r["Properties"]["DeletionProtectionEnabled"] is True, lid
    alb = next(r for _, _, r in synth_prod.resources("AWS::ElasticLoadBalancingV2::LoadBalancer"))
    attrs = {a["Key"]: a["Value"] for a in alb["Properties"]["LoadBalancerAttributes"]}
    assert attrs["deletion_protection.enabled"] == "true"


def test_dev_keeps_demo_defaults(synth_default):
    for g in (r for _, _, r in synth_default.resources("AWS::Logs::LogGroup")):
        assert g["DeletionPolicy"] == "Delete" and g["Properties"]["RetentionInDays"] == 30
    for _, lid, r in synth_default.resources("AWS::DynamoDB::Table"):
        assert not r["Properties"].get("DeletionProtectionEnabled"), lid
    alb = next(r for _, _, r in synth_default.resources("AWS::ElasticLoadBalancingV2::LoadBalancer"))
    attrs = {a["Key"]: a["Value"] for a in alb["Properties"]["LoadBalancerAttributes"]}
    assert attrs.get("deletion_protection.enabled", "false") == "false"


# ------------------------------------------------------------------ New IAM

@pytest.mark.parametrize("stack,role", [("OwlBankCompute", "FrontdoorRole"), ("OwlBankBridge", "TaskRole")])
def test_comprehend_pii_detect(synth_default, stack, role):
    st = [s for s in _stmts_of_role(synth_default, stack, role) if s.get("Sid") == "DetectPiiForRedaction"]
    assert len(st) == 1 and st[0]["Action"] == "comprehend:DetectPiiEntities" and st[0]["Resource"] == "*"


#: The only actions allowed on Resource "*", each with no resource-level permissions.
STAR_RESOURCE_ACTIONS = {"xray:PutTraceSegments", "xray:PutTelemetryRecords", "ecr:GetAuthorizationToken",
                         "comprehend:DetectPiiEntities"}


def test_star_resources_only_for_unscopable_actions(synth_any):
    for stack, lid, res in synth_any.resources("AWS::IAM::Policy"):
        for st in res["Properties"]["PolicyDocument"]["Statement"]:
            if st["Resource"] == "*" or (isinstance(st["Resource"], list) and "*" in st["Resource"]):
                assert set(_actions(st)) <= STAR_RESOURCE_ACTIONS, (stack, lid, st)


def test_comprehend_not_granted_elsewhere(synth_any):
    holders = set()
    for stack, lid, res in synth_any.resources("AWS::IAM::Policy"):
        if "comprehend:" in json.dumps(res["Properties"]["PolicyDocument"]):
            holders |= {(stack, r["Ref"].rstrip("0123456789ABCDEF")) for r in res["Properties"]["Roles"]}
    # ContextApiRole: the screen-pop timeline's read-time redaction.
    # RecapWriterRole: the recap's scrub, when the writer is on.
    want = {("OwlBankCompute", "FrontdoorRole"), ("OwlBankCompute", "ContextApiRole"),
            ("OwlBankBridge", "TaskRole")}
    if synth_any.cfg.recap_writer_on:
        want.add(("OwlBankCompute", "RecapWriterRole"))
    assert holders == want, holders


# ------------------------------------------------------------------ Bedrock (Region wildcard, one model)

# The bridge holds one pair per entry of the default three-model list; the front door only the primary.
@pytest.mark.parametrize("stack,role,pairs", [("OwlBankBridge", "TaskRole", 3), ("OwlBankCompute", "FrontdoorRole", 1)])
def test_bedrock_statements_shape(synth_any, stack, role, pairs):
    bedrock = [st for st in _stmts_of_role(synth_any, stack, role)
               if any(a.startswith("bedrock:") for a in _actions(st))]
    profiles = [st for st in bedrock if "Condition" not in st]
    vias = [st for st in bedrock if "Condition" in st]
    assert len(profiles) == len(vias) == pairs, bedrock
    for v in vias:
        assert v["Condition"]["StringEquals"]["bedrock:InferenceProfileArn"] in [p["Resource"] for p in profiles]
    profile = next(p for p in profiles
                   if ":us-east-1:111111111111:inference-profile/us.anthropic.claude-haiku-5-5" in json.dumps(p["Resource"]))
    via = next(v for v in vias
               if v["Condition"] == {"StringEquals": {"bedrock:InferenceProfileArn": profile["Resource"]}})
    res = via["Resource"]
    assert res == {"Fn::Join": ["", ["arn:", {"Ref": "AWS::Partition"},
                                     ":bedrock:*::foundation-model/anthropic.claude-haiku-5-5"]]}, res


def test_no_unshaped_bedrock_statement(synth_any):
    for stack, lid, res in synth_any.resources("AWS::IAM::Policy"):
        for st in res["Properties"]["PolicyDocument"]["Statement"]:
            if not any(a.startswith("bedrock:") for a in _actions(st)):
                continue
            text = json.dumps(st["Resource"])
            assert "foundation-model/*" not in text and "inference-profile/*" not in text, (lid, st)
            if "foundation-model/" in text:
                assert "bedrock:InferenceProfileArn" in json.dumps(st.get("Condition", {})), (lid, st)
            else:
                assert "inference-profile/us." in text, (lid, st)


def test_model_destination_regions_param_is_gone():
    from owlbank_infra.config import InfraConfig
    assert not hasattr(InfraConfig, "model_destination_regions")


# ------------------------------------------------------------------ lows

def test_cmk_logs_statement_scoped_to_owlbank_groups(synth_any):
    st = _data_key_policy(synth_any)["AllowCloudWatchLogs"]
    pats = st["Condition"]["ArnLike"]["kms:EncryptionContext:aws:logs:arn"]
    tails = sorted(json.dumps(p).rsplit(":log-group:", 1)[1].split('"')[0] for p in pats)
    assert tails == ["/owlbank/*", "aws-waf-logs-owlbank-*"], pats
    for _, lid, r in synth_any.resources("AWS::Logs::LogGroup"):
        name = r["Properties"]["LogGroupName"]
        assert name.startswith(("/owlbank/", "aws-waf-logs-owlbank-")), (lid, name)


def test_default_tags(synth_default, synth_prod):
    def tags(synth):
        t = next(r for _, _, r in synth.resources("AWS::DynamoDB::Table"))["Properties"]["Tags"]
        return {x["Key"]: x["Value"] for x in t}
    assert {"owner": "unassigned", "cost-center": "unassigned",
            "data-classification": "confidential"}.items() <= tags(synth_default).items()
    assert {"owner": "voice-team", "cost-center": "cc-1234",
            "data-classification": "restricted", "stage": "prod"}.items() <= tags(synth_prod).items()


def test_access_analyzer(synth_any):
    analyzers = [(s, r["Properties"]) for s, _, r in synth_any.resources("AWS::AccessAnalyzer::Analyzer")]
    assert len(analyzers) == 1 and analyzers[0][0] == DATA and analyzers[0][1]["Type"] == "ACCOUNT"


@pytest.mark.parametrize("ctx,msg", [
    ({"alarmEmail": "not-an-email"}, "alarmEmail"),
    ({"owner": "bad<tag>"}, "tag value"),
    ({"wafApiRateLimit": 5}, "between"),
])
def test_bad_context_is_rejected(tmp_path, ctx, msg):
    with pytest.raises(ConfigError, match=msg):
        build_app(ctx, outdir=str(tmp_path))
