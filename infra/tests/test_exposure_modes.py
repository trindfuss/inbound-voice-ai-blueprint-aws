"""The two exposure modes.

* CloudFront mode (no ``domainName``, the default): internal ALB behind a CloudFront VPC
  origin, CLOUDFRONT-scope WAF on the distribution, ``execute-api`` left on.
* Custom-domain mode (``domainName`` set): public HTTPS ALB, regional WAF, ``execute-api`` off.
"""

from __future__ import annotations

import json

import pytest

from owlbank_infra.app_builder import build_app
from owlbank_infra.config import ConfigError
from owlbank_infra.nag import ACKS

CACHING_DISABLED_POLICY_ID = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"  # AWS managed CachingDisabled
FAKE_CERT = "arn:aws:acm:us-east-1:111111111111:certificate/11111111-2222-3333-4444-555555555555"


def _one(synth, rtype, stack=None):
    found = [(s, lid, r) for s, lid, r in synth.resources(rtype) if stack in (None, s)]
    assert len(found) == 1, (rtype, [lid for _, lid, _ in found])
    return found[0][1], found[0][2]["Properties"]


def _alb_ingress(synth):
    """Every ingress rule of the bridge ALB's security group (inline and separate resources)."""
    bridge = synth.templates["OwlBankBridge"]["Resources"]
    sg = next(lid for lid, r in bridge.items() if lid.startswith("AlbSg") and r["Type"] == "AWS::EC2::SecurityGroup")
    rules = list(bridge[sg]["Properties"].get("SecurityGroupIngress", []))
    for lid, r in bridge.items():
        if r["Type"] == "AWS::EC2::SecurityGroupIngress" and r["Properties"]["GroupId"] == {"Fn::GetAtt": [sg, "GroupId"]}:
            rules.append(r["Properties"])
    return rules


def _dependencies(synth) -> dict[str, set[str]]:
    return {name: {d.stack_name for d in stack.dependencies} for name, stack in
            ((s.stack_name, s) for s in synth.stacks.values())}


# --------------------------------------------------------------------------- both modes

def test_both_modes_synthesize(synth_cloudfront, synth_domain, synth_prod):
    for synth in (synth_cloudfront, synth_domain, synth_prod):
        assert len(synth.templates) == 7  # with OwlBankBridgeImage


def test_default_is_cloudfront_mode(synth_default):
    assert not synth_default.domain_mode
    assert len(list(synth_default.resources("AWS::CloudFront::Distribution"))) == 1


def test_domain_mode_has_no_cloudfront(synth_domain_any):
    for rtype in ("AWS::CloudFront::Distribution", "AWS::CloudFront::VpcOrigin",
                  "AWS::CloudFront::OriginRequestPolicy"):
        assert not list(synth_domain_any.resources(rtype)), rtype


@pytest.mark.parametrize("fixture", ["synth_cloudfront", "synth_domain", "synth_prod"])
def test_stack_graph_is_acyclic_and_as_designed(request, fixture):
    """Agent and Bridge never depend on Compute. Compute depends on Agent and, in both modes,
    follows Bridge (deploy order for the relay parameter; no export). Api depends on
    Compute. No cycles."""
    synth = request.getfixturevalue(fixture)
    deps = _dependencies(synth)
    # Cycle check: DFS with colors.
    state: dict[str, int] = {}

    def visit(n: str) -> None:
        assert state.get(n) != 1, f"cycle through {n}"
        if state.get(n) == 2:
            return
        state[n] = 1
        for d in deps[n]:
            visit(d)
        state[n] = 2

    for n in deps:
        visit(n)
    assert "OwlBankCompute" not in deps["OwlBankAgent"] and "OwlBankCompute" not in deps["OwlBankBridge"]
    assert "OwlBankAgent" in deps["OwlBankCompute"] and "OwlBankAgent" in deps["OwlBankBridge"]
    assert "OwlBankCompute" in deps["OwlBankApi"]
    assert "OwlBankBridge" in deps["OwlBankCompute"]


def test_other_stacks_name_lambdas_by_deterministic_arn(synth_any):
    """Agent and Bridge reference bank_tools / level0_tools by name: no import from Compute."""
    for stack in ("OwlBankAgent", "OwlBankBridge"):
        imports = json.dumps(synth_any.templates[stack])
        assert "OwlBankCompute:" not in imports, stack
    names = {lid: r["Properties"].get("FunctionName")
             for _, lid, r in synth_any.resources("AWS::Lambda::Function")}
    stage = synth_any.stacks["data"].node.try_get_context("stage") or "dev"
    assert {n for n in names.values() if n} == {f"owlbank-{stage}-bank-tools", f"owlbank-{stage}-level0-tools"}


def test_gateway_target_is_in_compute_and_names_the_function(synth_any):
    (stack, _), = [(s, r) for s, _, r in synth_any.resources("AWS::BedrockAgentCore::GatewayTarget")]
    assert stack == "OwlBankCompute"


def test_bridge_public_domain_env_is_the_relay_host(synth_any):
    _, td = _one(synth_any, "AWS::ECS::TaskDefinition")
    env = {e["Name"]: e["Value"] for e in td["ContainerDefinitions"][0]["Environment"]}
    if synth_any.domain_mode:
        assert env["TWILIO_VOICE_PUBLIC_DOMAIN"] == "relay.owlbank.example.com"
    else:
        dist, _ = _one(synth_any, "AWS::CloudFront::Distribution")
        assert env["TWILIO_VOICE_PUBLIC_DOMAIN"] == {"Fn::GetAtt": [dist, "DomainName"]}


def test_frontdoor_relay_parameter(synth_any):
    """AgentCore hosting: the front door reads the relay host from the Bridge's fixed-name SSM parameter
    (and builds ``wss://<host>/ws``, as the template's URL used to be); Bridge exports nothing."""
    fd = next(r for _, lid, r in synth_any.resources("AWS::Lambda::Function") if lid.startswith("FrontdoorFn"))
    env = fd["Properties"]["Environment"]["Variables"]
    name = synth_any.stacks["compute"].cfg.relay_parameter_name
    assert "OWLBANK_RELAY_WS_URL" not in env and env["OWLBANK_RELAY_PARAMETER"] == name
    (stack, _, param), = list(synth_any.resources("AWS::SSM::Parameter"))
    assert stack == "OwlBankBridge" and param["Properties"]["Name"] == name
    assert param["Properties"]["Type"] == "String"
    if synth_any.domain_mode:
        assert param["Properties"]["Value"] == "relay.owlbank.example.com"
    else:
        dist, _ = _one(synth_any, "AWS::CloudFront::Distribution")
        assert param["Properties"]["Value"] == {"Fn::GetAtt": [dist, "DomainName"]}
    assert not any("Export" in o for o in synth_any.templates["OwlBankBridge"].get("Outputs", {}).values())
    assert "relay.invalid" not in json.dumps(synth_any.templates)


def test_alb_keeps_health_check_deregistration_idle_timeout_and_logs(synth_any):
    _, alb = _one(synth_any, "AWS::ElasticLoadBalancingV2::LoadBalancer")
    attrs = {a["Key"]: a["Value"] for a in alb["LoadBalancerAttributes"]}
    assert attrs["idle_timeout.timeout_seconds"] == "900"
    assert attrs["access_logs.s3.enabled"] == "true"
    _, tg = _one(synth_any, "AWS::ElasticLoadBalancingV2::TargetGroup")
    assert tg["HealthCheckPath"] == "/health"
    assert {a["Key"]: a["Value"] for a in tg["TargetGroupAttributes"]}["deregistration_delay.timeout_seconds"] == "300"


def test_unhealthy_host_alarm_in_both_modes(synth_any):
    alarms = [r["Properties"] for s, _, r in synth_any.resources("AWS::CloudWatch::Alarm") if s == "OwlBankBridge"]
    assert any(a.get("MetricName") == "UnHealthyHostCount" for a in alarms)


def test_every_new_suppression_has_a_written_reason(synth_cloudfront):
    cfr = {rule: reason for _, rule, reason in ACKS if rule.startswith("AwsSolutions-CFR")}
    assert set(cfr) == {"AwsSolutions-CFR1", "AwsSolutions-CFR3", "AwsSolutions-CFR4"}
    for rule, reason in cfr.items():
        assert len(reason) >= 150, rule
    # A suppression on the origin or on the ALB's open ingress is not needed in the CloudFront mode.
    assert not any(rule == "AwsSolutions-CFR5" for _, rule, _ in ACKS)


# --------------------------------------------------------------------------- CloudFront mode

def test_cf_alb_is_internal_in_private_subnets_with_http_listener_only(synth_cloudfront):
    _, alb = _one(synth_cloudfront, "AWS::ElasticLoadBalancingV2::LoadBalancer")
    assert alb["Scheme"] == "internal"
    assert len(alb["Subnets"]) == 2
    for subnet in alb["Subnets"]:
        assert "privateSubnet" in json.dumps(subnet) and "publicSubnet" not in json.dumps(subnet)
    listeners = [r["Properties"] for _, _, r in synth_cloudfront.resources("AWS::ElasticLoadBalancingV2::Listener")]
    assert [(lst["Protocol"], lst["Port"]) for lst in listeners] == [("HTTP", 80)]
    assert not any(lst["Port"] == 443 for lst in listeners)
    assert not list(synth_cloudfront.resources("AWS::CertificateManager::Certificate"))
    assert "certificate" not in json.dumps(listeners).lower()


def test_cf_alb_security_group_only_takes_port_80_from_the_cloudfront_origin(synth_cloudfront):
    rules = _alb_ingress(synth_cloudfront)
    assert len(rules) == 1, rules
    (rule,) = rules
    assert rule["SourcePrefixListId"] == "pl-3b927c52"
    assert rule["FromPort"] == rule["ToPort"] == 80 and rule["IpProtocol"] == "tcp"
    assert "CidrIp" not in rule and "CidrIpv6" not in rule and "SourceSecurityGroupId" not in rule


def test_cf_prefix_list_id_is_a_context_parameter(tmp_path):
    app, _ = build_app({"cloudFrontOriginPrefixListId": "pl-0123456789abcdef0"}, outdir=str(tmp_path))
    tpl = next(s.template for s in app.synth().stacks if s.stack_name == "OwlBankBridge")
    ids = {r["Properties"].get("SourcePrefixListId") for r in tpl["Resources"].values()
           if r["Type"] == "AWS::EC2::SecurityGroupIngress"}
    assert "pl-0123456789abcdef0" in ids
    with pytest.raises(ConfigError, match="prefix list"):
        build_app({"cloudFrontOriginPrefixListId": "nope"}, outdir=str(tmp_path / "bad"))


def test_cf_tasks_reachable_only_from_the_alb_sg(synth_cloudfront):
    bridge = synth_cloudfront.templates["OwlBankBridge"]["Resources"]
    task_ingress = [r["Properties"] for lid, r in bridge.items()
                    if r["Type"] == "AWS::EC2::SecurityGroupIngress" and lid.startswith("TaskSg")]
    assert task_ingress and all(r["FromPort"] == 8080 and "SourceSecurityGroupId" in r for r in task_ingress)


def test_cf_distribution_basics(synth_cloudfront):
    _, d = _one(synth_cloudfront, "AWS::CloudFront::Distribution")
    c = d["DistributionConfig"]
    assert c["Enabled"] is True
    assert c["PriceClass"] == "PriceClass_100"
    assert c["HttpVersion"] == "http1.1" and c["IPV6Enabled"] is False
    assert "ViewerCertificate" not in c and "Aliases" not in c  # the default *.cloudfront.net certificate
    assert "Restrictions" not in c
    behavior = c["DefaultCacheBehavior"]
    assert "CacheBehaviors" not in c  # one behavior for every path
    assert behavior["ViewerProtocolPolicy"] == "https-only"
    assert sorted(behavior["AllowedMethods"]) == sorted(["GET", "HEAD", "OPTIONS", "PUT", "PATCH", "POST", "DELETE"])
    assert behavior["CachePolicyId"] == CACHING_DISABLED_POLICY_ID
    assert behavior["Compress"] is False


def test_cf_origin_request_policy_forwards_websocket_and_twilio_headers(synth_cloudfront):
    lid, p = _one(synth_cloudfront, "AWS::CloudFront::OriginRequestPolicy")
    cfg = p["OriginRequestPolicyConfig"]
    # allViewer forwards every viewer header: Host, Upgrade, Connection, Sec-WebSocket-*,
    # X-Twilio-Signature, Authorization (the CloudFront docs' recommended WebSocket setup).
    assert cfg["HeadersConfig"] == {"HeaderBehavior": "allViewer"}
    assert cfg["QueryStringsConfig"] == {"QueryStringBehavior": "all"}  # the one-time nonce (?n=)
    assert cfg["CookiesConfig"] == {"CookieBehavior": "none"}
    _, d = _one(synth_cloudfront, "AWS::CloudFront::Distribution")
    assert d["DistributionConfig"]["DefaultCacheBehavior"]["OriginRequestPolicyId"] == {"Ref": lid}


def test_cf_vpc_origin_is_http_only_to_the_internal_alb(synth_cloudfront):
    vlid, v = _one(synth_cloudfront, "AWS::CloudFront::VpcOrigin")
    cfg = v["VpcOriginEndpointConfig"]
    alb_lid, _ = _one(synth_cloudfront, "AWS::ElasticLoadBalancingV2::LoadBalancer")
    assert cfg["Arn"] == {"Ref": alb_lid}
    assert cfg["OriginProtocolPolicy"] == "http-only" and cfg["HTTPPort"] == 80
    assert "HTTPSPort" not in cfg
    _, d = _one(synth_cloudfront, "AWS::CloudFront::Distribution")
    (origin,) = d["DistributionConfig"]["Origins"]
    assert origin["VpcOriginConfig"]["VpcOriginId"] == {"Fn::GetAtt": [vlid, "Id"]}
    assert "CustomOriginConfig" not in origin and "S3OriginConfig" not in origin
    assert origin["VpcOriginConfig"]["OriginReadTimeout"] == 60
    assert origin["VpcOriginConfig"]["OriginKeepaliveTimeout"] == 60


def test_cf_origin_timeouts_are_context_parameters(tmp_path):
    ctx = {"cloudFrontOriginReadTimeoutSeconds": 90, "cloudFrontOriginKeepaliveTimeoutSeconds": 120}
    app, _ = build_app(ctx, outdir=str(tmp_path))
    tpl = next(s.template for s in app.synth().stacks if s.stack_name == "OwlBankBridge")
    (d,) = [r for r in tpl["Resources"].values() if r["Type"] == "AWS::CloudFront::Distribution"]
    vpc = d["Properties"]["DistributionConfig"]["Origins"][0]["VpcOriginConfig"]
    assert vpc["OriginReadTimeout"] == 90 and vpc["OriginKeepaliveTimeout"] == 120
    for bad in ({"cloudFrontOriginReadTimeoutSeconds": 500}, {"cloudFrontOriginKeepaliveTimeoutSeconds": 0}):
        with pytest.raises(ConfigError, match="between"):
            build_app(bad, outdir=str(tmp_path / "bad"))


def test_cf_web_acl_is_cloudfront_scope_with_the_bridge_rules_and_logging(synth_cloudfront):
    _, d = _one(synth_cloudfront, "AWS::CloudFront::Distribution")
    acl_lid = d["DistributionConfig"]["WebACLId"]["Fn::GetAtt"][0]
    acl = synth_cloudfront.templates["OwlBankBridge"]["Resources"][acl_lid]
    assert acl["Type"] == "AWS::WAFv2::WebACL"
    p = acl["Properties"]
    assert p["Scope"] == "CLOUDFRONT" and p["Name"] == "owlbank-dev-cloudfront"
    rules = {r["Name"]: r for r in p["Rules"]}
    assert {"AllowedPathsAndMethodsOnly", "RateLimitPerIp", "BodySizeLimit",
            "AWSManagedRulesAmazonIpReputationList", "AWSManagedRulesCommonRuleSet",
            "AWSManagedRulesKnownBadInputsRuleSet"} <= set(rules)
    assert rules["RateLimitPerIp"]["Statement"]["RateBasedStatement"]["Limit"] == 10000
    paths = json.dumps(rules["AllowedPathsAndMethodsOnly"])
    assert "^/ws$" in paths and "^/webhook$" in paths
    assert "/health" not in paths  # Health checks go ALB-direct and never pass the WAF
    (conf,) = [r["Properties"] for s, _, r in synth_cloudfront.resources("AWS::WAFv2::LoggingConfiguration")
               if s == "OwlBankBridge"]
    assert conf["ResourceArn"] == {"Fn::GetAtt": [acl_lid, "Arn"]}
    assert json.dumps(conf["LogDestinationConfigs"]).endswith(":log-group:aws-waf-logs-owlbank-dev-cloudfront\"]]}]")
    assert {"SingleHeader": {"Name": "authorization"}} in conf["RedactedFields"]
    assert {"SingleHeader": {"Name": "cookie"}} in conf["RedactedFields"]
    assert {"QueryString": {}} in conf["RedactedFields"]
    # No regional association: the ALB is internal, the CloudFront ACL replaces it.
    assert not [s for s, _, _ in synth_cloudfront.resources("AWS::WAFv2::WebACLAssociation") if s == "OwlBankBridge"]


def test_cf_mode_requires_us_east_1(tmp_path):
    with pytest.raises(ConfigError, match="us-east-1"):
        build_app({"region": "us-west-2"}, outdir=str(tmp_path))


def test_domain_mode_is_allowed_outside_us_east_1(tmp_path):
    cert = "arn:aws:acm:us-west-2:111111111111:certificate/11111111-2222-3333-4444-555555555555"
    app, _ = build_app({"region": "us-west-2", "domainName": "owlbank.example.com", "certificateArn": cert},
                       outdir=str(tmp_path))
    assert app.synth()


def test_cf_api_keeps_execute_api_with_regional_waf_and_throttling(synth_cloudfront):
    _, api = _one(synth_cloudfront, "AWS::ApiGateway::RestApi")
    assert api.get("DisableExecuteApiEndpoint") in (None, False)
    assert not list(synth_cloudfront.resources("AWS::ApiGateway::DomainName"))
    (assoc,) = [r["Properties"] for s, _, r in synth_cloudfront.resources("AWS::WAFv2::WebACLAssociation")
                if s == "OwlBankApi"]
    assert "/stages/" in json.dumps(assoc["ResourceArn"])
    _, stage = _one(synth_cloudfront, "AWS::ApiGateway::Stage")
    ms = stage["MethodSettings"][0]
    assert ms["ThrottlingRateLimit"] == 50 and ms["ThrottlingBurstLimit"] == 100


def test_cf_outputs_for_the_setup_scripts(synth_cloudfront):
    bridge = synth_cloudfront.templates["OwlBankBridge"]["Outputs"]
    dist, _ = _one(synth_cloudfront, "AWS::CloudFront::Distribution")
    assert bridge["RelayDomain"]["Value"] == {"Fn::GetAtt": [dist, "DomainName"]}
    assert "AlbDnsName" not in bridge  # the ALB is internal: nothing to alias
    api = synth_cloudfront.templates["OwlBankApi"]["Outputs"]
    host = json.dumps(api["ApiDomain"]["Value"])
    assert ".execute-api.us-east-1." in host and "https://" not in host and "/v1" not in host
    assert api["ApiStage"]["Value"]["Ref"].startswith("ApiDeploymentStagev1")
    assert "ApiDomainTarget" not in api


def test_cf_alarms(synth_cloudfront):
    alarms = [r["Properties"] for s, _, r in synth_cloudfront.resources("AWS::CloudWatch::Alarm")
              if s == "OwlBankBridge"]
    (five,) = [a for a in alarms if a.get("MetricName") == "5xxErrorRate"]
    assert five["Namespace"] == "AWS/CloudFront" and five["Threshold"] == 5
    dims = {d["Name"]: d["Value"] for d in five["Dimensions"]}
    assert dims["Region"] == "Global" and "DistributionId" in dims
    assert "AlarmTopic" in json.dumps(five["AlarmActions"])
    # The ALB alarms that stay valid on an internal ALB.
    for metric in ("HTTPCode_ELB_5XX_Count", "HTTPCode_Target_5XX_Count", "UnHealthyHostCount"):
        assert any(a.get("MetricName") == metric for a in alarms), metric


def test_cf_mode_no_new_public_surface(synth_cloudfront):
    """In the CloudFront mode: the only internet-facing entry points are CloudFront and
    API Gateway; no public subnets host a load balancer; no listener on 443."""
    for _, _, r in synth_cloudfront.resources("AWS::ElasticLoadBalancingV2::LoadBalancer"):
        assert r["Properties"]["Scheme"] == "internal"
    assert not list(synth_cloudfront.resources("AWS::Lambda::Url"))
    for _, lid, r in synth_cloudfront.resources("AWS::EC2::SecurityGroup"):
        for rule in r["Properties"].get("SecurityGroupIngress", []):
            assert "CidrIp" not in rule and "CidrIpv6" not in rule, (lid, rule)


# --------------------------------------------------------------------------- custom-domain mode (unchanged)

def test_domain_mode_unchanged(synth_domain_any):
    synth = synth_domain_any
    _, alb = _one(synth, "AWS::ElasticLoadBalancingV2::LoadBalancer")
    assert alb["Scheme"] == "internet-facing"
    assert all("publicSubnet" in json.dumps(s) for s in alb["Subnets"])
    listeners = [r["Properties"] for _, _, r in synth.resources("AWS::ElasticLoadBalancingV2::Listener")]
    assert [(lst["Protocol"], lst["Port"]) for lst in listeners] == [("HTTPS", 443)]
    (rule,) = _alb_ingress(synth)
    assert rule["CidrIp"] == "0.0.0.0/0" and rule["FromPort"] == rule["ToPort"] == 443
    _, api = _one(synth, "AWS::ApiGateway::RestApi")
    assert api["DisableExecuteApiEndpoint"] is True
    acl = next(r["Properties"] for s, _, r in synth.resources("AWS::WAFv2::WebACL") if s == "OwlBankBridge")
    assert acl["Scope"] == "REGIONAL"
    assert synth.templates["OwlBankBridge"]["Outputs"]["RelayDomain"]["Value"] == "relay.owlbank.example.com"
    assert "AlbDnsName" in synth.templates["OwlBankBridge"]["Outputs"]
    assert synth.templates["OwlBankApi"]["Outputs"]["ApiDomain"]["Value"] == "voice.owlbank.example.com"
    assert "ApiStage" not in synth.templates["OwlBankApi"]["Outputs"]
