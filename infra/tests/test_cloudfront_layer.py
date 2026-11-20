"""The CloudFront layer's second ALB control (secret origin header), WAF managed-rule
modes and body sizes, the auth-header rule, the prod guard and the WebSocket keepalive flags."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from conftest import FAKE_CERT, REPO

from owlbank_infra.app_builder import build_app
from owlbank_infra.config import ConfigError, from_context

DOMAIN = {"domainName": "owlbank.example.com", "certificateArn": FAKE_CERT}
_cache: dict[str, dict] = {}


def _tpl(tmp_path_factory, **ctx) -> dict[str, dict]:
    key = json.dumps(ctx, sort_keys=True)
    if key not in _cache:
        app, _ = build_app(ctx, outdir=str(tmp_path_factory.mktemp("r9")))
        _cache[key] = {s.stack_name: s.template for s in app.synth().stacks}
    return _cache[key]


def _acls(tpl: dict, stack: str) -> dict:
    (acl,) = [r["Properties"] for r in tpl[stack]["Resources"].values() if r["Type"] == "AWS::WAFv2::WebACL"]
    return acl


def _rules(tpl: dict, stack: str) -> dict:
    return {r["Name"]: r for r in _acls(tpl, stack)["Rules"]}


MANAGED = ("AWSManagedRulesAmazonIpReputationList", "AWSManagedRulesCommonRuleSet",
           "AWSManagedRulesKnownBadInputsRuleSet")


def _override(rule: dict) -> str:
    (k,) = rule["OverrideAction"]
    return k


# ---------------------------------------------------------------- 1. origin verify header

def _bridge(synth):
    return synth.templates["OwlBankBridge"]["Resources"]


def _resolve_ref(value) -> bool:
    """True if ``value`` is a CloudFormation Join that starts with a secretsmanager dynamic reference."""
    parts = value["Fn::Join"][1]
    return parts[0] == "{{resolve:secretsmanager:" and "Ref" in parts[1] and parts[2].startswith(":SecretString:")


def test_origin_header_is_a_dynamic_reference_never_plaintext(synth_cloudfront):
    res = _bridge(synth_cloudfront)
    (dist,) = [r for r in res.values() if r["Type"] == "AWS::CloudFront::Distribution"]
    (origin,) = dist["Properties"]["DistributionConfig"]["Origins"]
    (hdr,) = origin["OriginCustomHeaders"]
    assert hdr["HeaderName"] == "X-Origin-Verify"
    assert _resolve_ref(hdr["HeaderValue"])
    (secret_lid,) = [k for k, r in res.items() if r["Type"] == "AWS::SecretsManager::Secret"]
    assert hdr["HeaderValue"]["Fn::Join"][1][1] == {"Ref": secret_lid}
    secret = res[secret_lid]["Properties"]
    assert "SecretString" not in secret and "GenerateSecretString" in secret
    assert secret["GenerateSecretString"]["ExcludePunctuation"] is True  # no ALB wildcards (* ?)
    assert "KmsKeyId" in secret  # the CMK
    # The header name appears exactly twice (origin and rule), each with a dynamic reference.
    assert json.dumps(synth_cloudfront.templates["OwlBankBridge"]).count('"X-Origin-Verify"') == 2
    assert not any("X-Origin-Verify" in json.dumps(t) for n, t in synth_cloudfront.templates.items()
                   if n != "OwlBankBridge")


def test_alb_default_is_403_and_one_rule_forwards_on_the_header(synth_cloudfront):
    res = _bridge(synth_cloudfront)
    (lst,) = [r["Properties"] for r in res.values() if r["Type"] == "AWS::ElasticLoadBalancingV2::Listener"]
    (default,) = lst["DefaultActions"]
    assert default["Type"] == "fixed-response" and default["FixedResponseConfig"]["StatusCode"] == "403"
    rules = [r["Properties"] for r in res.values() if r["Type"] == "AWS::ElasticLoadBalancingV2::ListenerRule"]
    (rule,) = rules
    (cond,) = rule["Conditions"]
    assert cond["Field"] == "http-header"
    assert cond["HttpHeaderConfig"]["HttpHeaderName"] == "X-Origin-Verify"
    (value,) = cond["HttpHeaderConfig"]["Values"]
    assert _resolve_ref(value)
    (action,) = rule["Actions"]
    assert action["Type"] == "forward"
    # The same secret feeds the distribution and the rule.
    (dist,) = [r for r in res.values() if r["Type"] == "AWS::CloudFront::Distribution"]
    (origin,) = dist["Properties"]["DistributionConfig"]["Origins"]
    assert origin["OriginCustomHeaders"][0]["HeaderValue"] == value


def test_alb_ingress_is_unchanged_prefix_list_port_80_only(synth_cloudfront):
    res = _bridge(synth_cloudfront)
    sg = next(k for k, r in res.items() if k.startswith("AlbSg") and r["Type"] == "AWS::EC2::SecurityGroup")
    rules = list(res[sg]["Properties"].get("SecurityGroupIngress", []))
    rules += [r["Properties"] for r in res.values() if r["Type"] == "AWS::EC2::SecurityGroupIngress"
              and r["Properties"]["GroupId"] == {"Fn::GetAtt": [sg, "GroupId"]}]
    (rule,) = rules
    assert rule["SourcePrefixListId"] == "pl-3b927c52"
    assert rule["FromPort"] == rule["ToPort"] == 80 and rule["IpProtocol"] == "tcp"
    assert "CidrIp" not in rule


def test_target_group_health_check_is_still_health(synth_cloudfront):
    (tg,) = [r["Properties"] for r in _bridge(synth_cloudfront).values()
             if r["Type"] == "AWS::ElasticLoadBalancingV2::TargetGroup"]
    assert tg["HealthCheckPath"] == "/health"  # health checks bypass listener rules


def test_secret_version_pin(tmp_path_factory):
    version = "0123456789abcdef0123456789abcdef"
    tpl = _tpl(tmp_path_factory, originVerifySecretVersion=version)
    (dist,) = [r for r in tpl["OwlBankBridge"]["Resources"].values() if r["Type"] == "AWS::CloudFront::Distribution"]
    value = dist["Properties"]["DistributionConfig"]["Origins"][0]["OriginCustomHeaders"][0]["HeaderValue"]
    assert value["Fn::Join"][1][2] == f":SecretString:::{version}}}}}"
    with pytest.raises(ConfigError, match="version ID"):
        from_context({"originVerifySecretVersion": "x"}, base_dir=Path("."))


def test_custom_domain_mode_has_no_origin_verify(synth_domain):
    res = _bridge(synth_domain)
    assert not [r for r in res.values() if r["Type"] == "AWS::SecretsManager::Secret"]
    assert not [r for r in res.values() if r["Type"] == "AWS::ElasticLoadBalancingV2::ListenerRule"]
    (lst,) = [r["Properties"] for r in res.values() if r["Type"] == "AWS::ElasticLoadBalancingV2::Listener"]
    assert lst["DefaultActions"][0]["Type"] == "forward"


# ---------------------------------------------------------------- 2. WAF managed-rule mode

def test_mode_default_follows_the_stage():
    assert from_context({}, base_dir=Path(".")).waf_managed_mode == "count"
    assert from_context({"stage": "test"}, base_dir=Path(".")).waf_managed_mode == "count"
    assert from_context({"stage": "prod", **DOMAIN}, base_dir=Path(".")).waf_managed_mode == "block"
    assert from_context({"wafManagedRuleMode": "BLOCK"}, base_dir=Path(".")).waf_managed_mode == "block"
    assert from_context({"stage": "prod", "wafManagedRuleMode": "count", **DOMAIN},
                        base_dir=Path(".")).waf_managed_mode == "count"
    with pytest.raises(ConfigError, match="count"):
        from_context({"wafManagedRuleMode": "off"}, base_dir=Path("."))


def test_count_mode_is_the_dev_default_on_both_acls(synth_default):
    for stack in ("OwlBankBridge", "OwlBankApi"):
        rules = _rules(synth_default.templates, stack)
        assert {_override(rules[m]) for m in MANAGED} == {"Count"}, stack
    # The label rule counts too, so nothing the managed rules matched is blocked.
    assert "Block" in rules["AuthorizationHeaderRequired"]["Action"]  # not a managed rule


def test_block_mode_on_both_acls_and_the_size_rule_override(tmp_path_factory):
    tpl = _tpl(tmp_path_factory, wafManagedRuleMode="block")
    for stack in ("OwlBankBridge", "OwlBankApi"):
        rules = _rules(tpl, stack)
        assert {_override(rules[m]) for m in MANAGED} == {"None"}, stack
    cf = _rules(tpl, "OwlBankBridge")
    crs = cf["AWSManagedRulesCommonRuleSet"]["Statement"]["ManagedRuleGroupStatement"]
    assert crs["RuleActionOverrides"] == [{"Name": "SizeRestrictions_BODY", "ActionToUse": {"Count": {}}}]
    oversize = cf["OversizeBodyOutsideLargeBodyPaths"]
    assert oversize["Action"] == {"Block": {}}
    text = json.dumps(oversize["Statement"])
    assert "awswaf:managed:aws:core-rule-set:SizeRestrictions_Body" in text
    assert "^/webhook$" in text and "NotStatement" in text
    # The API has no large-body path: the CRS keeps its own 8 KB rule untouched.
    api = _rules(tpl, "OwlBankApi")
    assert "RuleActionOverrides" not in api["AWSManagedRulesCommonRuleSet"]["Statement"]["ManagedRuleGroupStatement"]
    assert "OversizeBodyOutsideLargeBodyPaths" not in api


def test_count_mode_label_rule_counts(synth_default):
    cf = _rules(synth_default.templates, "OwlBankBridge")
    assert cf["OversizeBodyOutsideLargeBodyPaths"]["Action"] == {"Count": {}}
    crs = cf["AWSManagedRulesCommonRuleSet"]["Statement"]["ManagedRuleGroupStatement"]
    assert "RuleActionOverrides" not in crs  # the whole group is Count already


def test_prod_defaults_to_block_and_can_be_counted(synth_prod, tmp_path_factory):
    rules = _rules(synth_prod.templates, "OwlBankApi")
    assert {_override(rules[m]) for m in MANAGED} == {"None"}
    tpl = _tpl(tmp_path_factory, stage="prod", wafManagedRuleMode="count", **DOMAIN)
    assert {_override(_rules(tpl, "OwlBankApi")[m]) for m in MANAGED} == {"Count"}
    assert {_override(_rules(tpl, "OwlBankBridge")[m]) for m in MANAGED} == {"Count"}  # the ALB's ACL


def test_cloudfront_body_inspection_limit_covers_the_custom_size_cap(synth_cloudfront):
    acl = _acls(synth_cloudfront.templates, "OwlBankBridge")
    assert acl["AssociationConfig"] == {"RequestBody": {"CLOUDFRONT": {"DefaultSizeInspectionLimit": "KB_64"}}}
    size = _rules(synth_cloudfront.templates, "OwlBankBridge")["BodySizeLimit"]["Statement"]["SizeConstraintStatement"]
    assert size["Size"] == 65536 and size["FieldToMatch"]["Body"]["OversizeHandling"] == "MATCH"
    # Regional ACLs: API Gateway's 16 KB default covers 8 KB; an ALB's limit is fixed. No config.
    assert "AssociationConfig" not in _acls(synth_cloudfront.templates, "OwlBankApi")


def test_alb_acl_has_no_association_config_or_large_body_rule(synth_domain):
    assert "AssociationConfig" not in _acls(synth_domain.templates, "OwlBankBridge")
    assert "OversizeBodyOutsideLargeBodyPaths" not in _rules(synth_domain.templates, "OwlBankBridge")


# ---------------------------------------------------------------- 4. low fixes

def test_health_is_not_an_allowed_path_on_any_bridge_acl(synth_any):
    allowed = json.dumps(_rules(synth_any.templates, "OwlBankBridge")["AllowedPathsAndMethodsOnly"])
    assert "^/ws$" in allowed and "^/webhook$" in allowed and "/health" not in allowed


def test_api_blocks_voice_requests_without_authorization(synth_any):
    rule = _rules(synth_any.templates, "OwlBankApi")["AuthorizationHeaderRequired"]
    # A block that answers with the Basic challenge (Twilio retries with credentials only after it)
    assert rule["Action"] == {"Block": {"CustomResponse": {
        "ResponseCode": 401,
        "ResponseHeaders": [{"Name": "WWW-Authenticate", "Value": 'Basic realm="owlbank"'}]}}}
    assert "CustomResponseBodyKey" not in json.dumps(rule["Action"])
    (path, neg) = rule["Statement"]["AndStatement"]["Statements"]
    assert path["RegexMatchStatement"]["RegexString"] == "^(/v1)?/voice/"
    assert path["RegexMatchStatement"]["FieldToMatch"] == {"UriPath": {}}
    size = neg["NotStatement"]["Statement"]["SizeConstraintStatement"]
    assert size["FieldToMatch"] == {"SingleHeader": {"Name": "authorization"}}
    assert size["ComparisonOperator"] == "GT" and size["Size"] == 0
    assert re.match(path["RegexMatchStatement"]["RegexString"], "/v1/voice/incoming")
    assert not re.match(path["RegexMatchStatement"]["RegexString"], "/v1/context/abc")
    assert "AuthorizationHeaderRequired" not in _rules(synth_any.templates, "OwlBankBridge")


def test_only_the_authorization_rule_has_a_custom_response(synth_any):
    for stack in ("OwlBankBridge", "OwlBankApi"):
        acl = _acls(synth_any.templates, stack)
        assert "CustomResponseBodies" not in acl
        custom = {r["Name"] for r in acl["Rules"] if "CustomResponse" in json.dumps(r.get("Action", {}))}
        assert custom == ({"AuthorizationHeaderRequired"} if stack == "OwlBankApi" else set())
        assert "CustomResponse" not in json.dumps(acl["DefaultAction"])


def test_waf_priorities_are_unique(synth_any):
    for stack in ("OwlBankBridge", "OwlBankApi"):
        prios = [r["Priority"] for r in _acls(synth_any.templates, stack)["Rules"]]
        assert len(prios) == len(set(prios))


def test_api_rate_limit_default_and_context(tmp_path_factory):
    assert from_context({}, base_dir=Path(".")).waf_api_rate_limit == 2000
    assert from_context({"wafApiRateLimit": "5000"}, base_dir=Path(".")).waf_api_rate_limit == 5000


def test_default_waf_logging_redacts_the_signature_header(synth_any):
    for _, _, r in synth_any.resources("AWS::WAFv2::LoggingConfiguration"):
        redacted = r["Properties"]["RedactedFields"]
        for name in ("authorization", "cookie", "x-twilio-signature"):
            assert {"SingleHeader": {"Name": name}} in redacted


# ---------------------------------------------------------------- guard and keepalive

def test_prod_without_a_domain_fails_synth_and_points_to_the_custom_domain_mode(tmp_path):
    with pytest.raises(ConfigError, match=r"custom-domain mode.*domainName and certificateArn"):
        build_app({"stage": "prod"}, outdir=str(tmp_path))


@pytest.mark.parametrize("stage", ["dev", "test"])
def test_sandbox_mode_still_works_below_prod(stage):
    assert from_context({"stage": stage}, base_dir=Path(".")).custom_domain_mode is False


def test_prod_with_a_domain_is_accepted(synth_prod):
    assert synth_prod.domain_mode


def test_dockerfile_cmd_has_explicit_websocket_keepalive():
    text = (REPO / "src" / "tac_bridge" / "Dockerfile").read_text()
    cmd = text[text.rindex("CMD ["):]
    argv = json.loads(cmd[3:].replace("\\\n", " ").strip())
    assert argv == ["python", "-m", "tac_bridge.serve"]
    serve = (REPO / "src" / "tac_bridge" / "serve.py").read_text()
    assert "ws_ping_interval=20.0" in serve and "ws_ping_timeout=20.0" in serve
