"""The ``OwlBankAdmin`` stack.

The synthesized stack has no public bucket, an ``/api/*`` behaviour to the API with the
origin-verify header and a WAF rule requiring it, no ``AuthType: NONE`` (no Function URL, no
unauthenticated method), no method without the Cognito authorizer, MFA ``REQUIRED`` iff
``stage=prod``, and the CSP and HSTS headers. Every route is a method
with the authorizer and a Lambda proxy integration. The Lambda's role reaches only the
policy, catalog and audit tables and the one user pool. Also: the first user exists only with
``adminEmail``, the stack is optional (``adminConsole``), no CORS is left, and cdk-nag is clean.
"""

from __future__ import annotations

import json
import re

import pytest
from conftest import FAKE_CERT, _synth
from test_synth_and_nag import _nag_report

from owlbank_infra.admin_stack import GROUPS, HSTS_SECONDS, MAX_BODY_BYTES, PROMPTS_MAX_BODY_BYTES
from owlbank_infra.app_builder import build_app
from owlbank_infra.config import ConfigError

ADMIN = "OwlBankAdmin"
EMAIL = "first.admin@example.com"
PROD = {"stage": "prod", "domainName": "owlbank.example.com", "certificateArn": FAKE_CERT,
        "bridgeDesiredCount": 2}


@pytest.fixture(scope="module")
def admin_dev(tmp_path_factory):
    """The dev default (adminConsole unset in the real app means on; named here because the
    shared ``_synth`` leaves the console out unless asked)."""
    return _synth(tmp_path_factory, "admin-dev", {"adminConsole": "true"})


@pytest.fixture(scope="module")
def admin_email(tmp_path_factory):
    return _synth(tmp_path_factory, "admin-email", {"adminConsole": "true", "adminEmail": EMAIL})


@pytest.fixture(scope="module")
def admin_prod(tmp_path_factory):
    return _synth(tmp_path_factory, "admin-prod", {**PROD, "adminConsole": "true", "adminEmail": EMAIL})


@pytest.fixture(scope="module")
def admin_bpa_account(tmp_path_factory):
    return _synth(tmp_path_factory, "admin-bpa", {"adminConsole": "true", "s3PublicAccessBlockMode": "account"})


@pytest.fixture(scope="module", params=["admin_dev", "admin_email", "admin_prod", "admin_bpa_account"])
def admin_any(request):
    return request.getfixturevalue(request.param)


def _res(synth, rtype):
    return [r for s, _, r in synth.resources(rtype) if s == ADMIN]


def _one(synth, rtype):
    found = _res(synth, rtype)
    assert len(found) == 1, (rtype, len(found))
    return found[0]["Properties"]


def _tpl(synth):
    return synth.templates[ADMIN]


# ---- optional stack ----------------------------------------------------------

def test_stack_on_by_default_in_dev(tmp_path):
    _, stacks = build_app({}, outdir=str(tmp_path))
    assert stacks["admin"].stack_name == ADMIN


def test_stack_absent_when_turned_off(tmp_path):
    _, stacks = build_app({"adminConsole": "false"}, outdir=str(tmp_path))
    assert "admin" not in stacks


def test_stack_absent_in_prod_by_default(tmp_path):
    _, stacks = build_app(PROD, outdir=str(tmp_path))
    assert "admin" not in stacks


def test_stack_off_by_default_outside_us_east_1_and_refused_when_asked(tmp_path):
    ctx = {"region": "us-west-2", "domainName": "owlbank.example.com", "certificateArn": FAKE_CERT}
    _, stacks = build_app(ctx, outdir=str(tmp_path / "a"))
    assert "admin" not in stacks
    with pytest.raises(ConfigError, match="us-east-1"):
        build_app({**ctx, "adminConsole": "true"}, outdir=str(tmp_path / "b"))


def test_call_path_stacks_do_not_depend_on_admin(admin_dev):
    admin = admin_dev.stacks["admin"]
    for key, stack in admin_dev.stacks.items():
        if key != "admin":
            assert admin not in stack.dependencies, key


@pytest.mark.parametrize("bad", ["not-an-email", "a@b", "two@example.com three@example.com", "x" * 250 + "@example.com"])
def test_bad_admin_email_is_rejected_without_echoing_it(tmp_path, bad):
    with pytest.raises(ConfigError) as err:
        build_app({"adminEmail": bad}, outdir=str(tmp_path))
    assert bad not in str(err.value)


# ---- Cognito ---------------------------------------------------------------

def test_user_pool_admin_created_users_and_password_policy(admin_any):
    pool = _one(admin_any, "AWS::Cognito::UserPool")
    assert pool["AdminCreateUserConfig"]["AllowAdminCreateUserOnly"] is True
    assert pool["UsernameAttributes"] == ["email"]
    assert pool["Policies"]["PasswordPolicy"] == {
        "MinimumLength": 12, "RequireLowercase": True, "RequireUppercase": True,
        "RequireNumbers": True, "RequireSymbols": True, "TemporaryPasswordValidityDays": 3}
    assert pool["AccountRecoverySetting"]["RecoveryMechanisms"] == [{"Name": "verified_email", "Priority": 1}]


@pytest.mark.parametrize("fixture,prod", [("admin_dev", False), ("admin_email", False),
                                          ("admin_bpa_account", False), ("admin_prod", True)])
def test_mfa_required_iff_prod(request, fixture, prod):
    pool = _one(request.getfixturevalue(fixture), "AWS::Cognito::UserPool")
    if prod:
        assert pool["MfaConfiguration"] == "ON"  # CloudFormation's word for REQUIRED
        assert pool["EnabledMfas"] == ["SOFTWARE_TOKEN_MFA"]  # TOTP only, no SMS
    else:
        assert pool["MfaConfiguration"] == "OFF"
        assert "EnabledMfas" not in pool


def test_retention_per_stage(admin_dev, admin_prod):
    for synth, policy, protection in ((admin_dev, "Delete", "INACTIVE"), (admin_prod, "Retain", "ACTIVE")):
        pool = _res(synth, "AWS::Cognito::UserPool")[0]
        assert pool["DeletionPolicy"] == policy and pool["Properties"]["DeletionProtection"] == protection
        for rtype in ("AWS::DynamoDB::Table", "AWS::S3::Bucket"):
            assert _res(synth, rtype)[0]["DeletionPolicy"] == policy, rtype
        assert _one(synth, "AWS::DynamoDB::Table")["DeletionProtectionEnabled"] is (policy == "Retain")


def test_groups_exist(admin_any):
    names = {g["Properties"]["GroupName"] for g in _res(admin_any, "AWS::Cognito::UserPoolGroup")}
    assert names == set(GROUPS) == {"admin", "admin-editor", "admin-viewer"}


def test_app_client_is_public_code_flow_with_console_callback(admin_any):
    client = _one(admin_any, "AWS::Cognito::UserPoolClient")
    assert client["GenerateSecret"] is False  # a public client: PKCE in the browser
    assert client["AllowedOAuthFlows"] == ["code"]
    assert client["AllowedOAuthFlowsUserPoolClient"] is True
    assert sorted(client["AllowedOAuthScopes"]) == ["email", "openid", "profile"]
    assert client["ExplicitAuthFlows"] == ["ALLOW_REFRESH_TOKEN_AUTH"]
    assert client["SupportedIdentityProviders"] == ["COGNITO"]
    assert client["PreventUserExistenceErrors"] == "ENABLED"
    assert (client["AccessTokenValidity"], client["IdTokenValidity"], client["RefreshTokenValidity"]) == (60, 60, 480)
    assert set(client["TokenValidityUnits"].values()) == {"minutes"}
    dist_id = next(lid for s, lid, _ in admin_any.resources("AWS::CloudFront::Distribution") if s == ADMIN)
    want = [{"Fn::Join": ["", ["https://", {"Fn::GetAtt": [dist_id, "DomainName"]}, "/"]]}]
    assert client["CallbackURLs"] == want and client["LogoutURLs"] == want


def test_managed_login_prefix_domain_is_unique_per_account_and_stage(admin_dev, admin_prod):
    for synth, stage in ((admin_dev, "dev"), (admin_prod, "prod")):
        domain = _one(synth, "AWS::Cognito::UserPoolDomain")
        assert domain["Domain"] == f"owlbank-{stage}-admin-111111111111"
        assert domain["ManagedLoginVersion"] == 2
        assert not any(w in domain["Domain"] for w in ("aws", "amazon", "cognito"))
        assert _one(synth, "AWS::Cognito::ManagedLoginBranding")["UseCognitoProvidedValues"] is True


def test_first_user_only_with_admin_email(admin_dev, admin_bpa_account, admin_email, admin_prod):
    for synth in (admin_dev, admin_bpa_account):
        assert not _res(synth, "AWS::Cognito::UserPoolUser")
        assert not _res(synth, "AWS::Cognito::UserPoolUserToGroupAttachment")
    for synth in (admin_email, admin_prod):
        user = _one(synth, "AWS::Cognito::UserPoolUser")
        assert user["Username"] == EMAIL
        assert user["DesiredDeliveryMediums"] == ["EMAIL"]
        assert {a["Name"]: a["Value"] for a in user["UserAttributes"]} == {"email": EMAIL, "email_verified": "true"}
        assert "MessageAction" not in user  # Cognito sends the temporary password
        attach = _one(synth, "AWS::Cognito::UserPoolUserToGroupAttachment")
        assert attach["Username"] == EMAIL and attach["GroupName"] == "admin"


def test_admin_email_appears_only_in_the_user_resources(admin_email):
    tpl = _tpl(admin_email)
    for lid, res in tpl["Resources"].items():
        if res["Type"] not in ("AWS::Cognito::UserPoolUser", "AWS::Cognito::UserPoolUserToGroupAttachment"):
            assert EMAIL not in json.dumps(res), lid
    assert EMAIL not in json.dumps(tpl.get("Outputs", {}))


# ---- site -------------------------------------------------------------------

def test_site_bucket_is_private(admin_any):
    bucket = _one(admin_any, "AWS::S3::Bucket")
    assert bucket["OwnershipControls"] == {"Rules": [{"ObjectOwnership": "BucketOwnerEnforced"}]}
    assert "AccessControl" not in bucket
    sse = bucket["BucketEncryption"]["ServerSideEncryptionConfiguration"][0]["ServerSideEncryptionByDefault"]
    assert sse == {"SSEAlgorithm": "AES256"}
    if admin_any.bpa_mode == "explicit":
        assert bucket["PublicAccessBlockConfiguration"] == {
            "BlockPublicAcls": True, "BlockPublicPolicy": True, "IgnorePublicAcls": True, "RestrictPublicBuckets": True}
    else:  # S3's default and the account setting protect it (preflight checks both)
        assert "PublicAccessBlockConfiguration" not in bucket
    assert "WebsiteConfiguration" not in bucket


def test_bucket_policy_tls_only_and_cloudfront_only(admin_any):
    statements = _one(admin_any, "AWS::S3::BucketPolicy")["PolicyDocument"]["Statement"]
    tls = [s for s in statements if s["Effect"] == "Deny"
           and s.get("Condition") == {"Bool": {"aws:SecureTransport": "false"}}]
    assert tls and tls[0]["Action"] == "s3:*"
    for s in statements:
        if s["Effect"] == "Allow":
            # Only CloudFront (OAC, this distribution) and CDK's auto-delete role; never "*".
            assert s["Principal"] != "*" and s["Principal"].get("AWS") != "*", s
            if "Service" in s["Principal"]:
                assert s["Principal"]["Service"] == "cloudfront.amazonaws.com"
                assert s["Action"] == "s3:GetObject"
                assert "AWS:SourceArn" in s["Condition"]["StringEquals"]


def test_distribution_uses_oac_https_and_index(admin_any):
    cfg = _one(admin_any, "AWS::CloudFront::Distribution")["DistributionConfig"]
    assert cfg["DefaultRootObject"] == "index.html"
    behavior = cfg["DefaultCacheBehavior"]
    assert behavior["ViewerProtocolPolicy"] == "redirect-to-https"
    assert behavior["AllowedMethods"] == ["GET", "HEAD"]
    assert "ResponseHeadersPolicyId" in behavior
    origin = next(o for o in cfg["Origins"] if o["Id"] == behavior["TargetOriginId"])
    assert "OriginAccessControlId" in origin and origin["S3OriginConfig"]["OriginAccessIdentity"] == ""
    oac = _one(admin_any, "AWS::CloudFront::OriginAccessControl")["OriginAccessControlConfig"]
    assert oac["OriginAccessControlOriginType"] == "s3" and oac["SigningBehavior"] == "always"


def test_csp_and_hsts_headers(admin_any):
    sec = _one(admin_any, "AWS::CloudFront::ResponseHeadersPolicy")["ResponseHeadersPolicyConfig"]["SecurityHeadersConfig"]
    csp = json.dumps(sec["ContentSecurityPolicy"]["ContentSecurityPolicy"])
    stage = "prod" if "prod" in json.dumps(_one(admin_any, "AWS::Cognito::UserPoolDomain")) else "dev"
    login = f"https://owlbank-{stage}-admin-111111111111.auth.us-east-1.amazoncognito.com"
    assert "default-src 'self'" in csp
    # The API is same-origin (/api/*), so connect-src is 'self' plus the token endpoint.
    assert f"connect-src 'self' {login};" in csp
    assert "execute-api" not in csp
    assert f"form-action 'self' {login};" in csp
    assert "frame-ancestors 'none'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp and "*" not in csp
    assert sec["ContentSecurityPolicy"]["Override"] is True
    assert HSTS_SECONDS == 365 * 24 * 3600
    assert sec["StrictTransportSecurity"] == {"AccessControlMaxAgeSec": HSTS_SECONDS,
                                              "IncludeSubdomains": True, "Override": True}
    assert sec["ContentTypeOptions"] == {"Override": True}
    assert sec["FrameOptions"] == {"FrameOption": "DENY", "Override": True}
    assert sec["ReferrerPolicy"] == {"ReferrerPolicy": "strict-origin", "Override": True}


# ---- audit table ----------------------------------------------------------------------

def test_audit_table(admin_any):
    table = _one(admin_any, "AWS::DynamoDB::Table")
    assert table["KeySchema"] == [{"AttributeName": "pk", "KeyType": "HASH"},
                                  {"AttributeName": "sk", "KeyType": "RANGE"}]
    assert {a["AttributeName"]: a["AttributeType"] for a in table["AttributeDefinitions"]} == {"pk": "S", "sk": "S"}
    assert table["SSESpecification"]["SSEEnabled"] is True and table["SSESpecification"]["SSEType"] == "KMS"
    # The project CMK, imported from the Data stack.
    assert "Fn::ImportValue" in json.dumps(table["SSESpecification"]["KMSMasterKeyId"])
    assert "OwlBankData" in json.dumps(table["SSESpecification"]["KMSMasterKeyId"])
    assert table["PointInTimeRecoverySpecification"] == {"PointInTimeRecoveryEnabled": True}
    assert table["TimeToLiveSpecification"] == {"AttributeName": "expires_at", "Enabled": True}


# ---- API --------------------------------------------------------------------

def test_every_method_has_the_cognito_authorizer(admin_any):
    authorizers = {lid: r for s, lid, r in admin_any.resources("AWS::ApiGateway::Authorizer") if s == ADMIN}
    assert len(authorizers) == 1
    (auth_id, auth), = authorizers.items()
    assert auth["Properties"]["Type"] == "COGNITO_USER_POOLS"
    assert auth["Properties"]["IdentitySource"] == "method.request.header.Authorization"
    pool_id = next(lid for s, lid, _ in admin_any.resources("AWS::Cognito::UserPool") if s == ADMIN)
    assert auth["Properties"]["ProviderARNs"] == [{"Fn::GetAtt": [pool_id, "Arn"]}]
    methods = _res(admin_any, "AWS::ApiGateway::Method")
    assert methods
    for m in methods:
        assert m["Properties"]["AuthorizationType"] == "COGNITO_USER_POOLS", m
        assert m["Properties"]["AuthorizerId"] == {"Ref": auth_id}


def test_no_auth_type_none_anywhere(admin_any):
    tpl = json.dumps(_tpl(admin_any))
    assert '"AuthorizationType": "NONE"' not in tpl
    assert '"AuthType": "NONE"' not in tpl
    assert not _res(admin_any, "AWS::Lambda::Url")  # API Gateway only, never a Function URL


def test_api_is_regional_with_access_logs_and_throttling(admin_any):
    api = _one(admin_any, "AWS::ApiGateway::RestApi")
    assert api["EndpointConfiguration"] == {"Types": ["REGIONAL"]}
    stage = _one(admin_any, "AWS::ApiGateway::Stage")
    assert stage["StageName"] == "v1"
    assert "DestinationArn" in stage["AccessLogSetting"]
    (settings,) = stage["MethodSettings"]
    assert settings["DataTraceEnabled"] is False and settings["LoggingLevel"] == "ERROR"
    assert (settings["ThrottlingRateLimit"], settings["ThrottlingBurstLimit"]) == (10, 20)
    assert not _res(admin_any, "AWS::ApiGateway::Account")  # OwlBankApi owns the account setting


def test_admin_follows_api_for_deploy_order(admin_dev):
    assert admin_dev.stacks["api"] in admin_dev.stacks["admin"].dependencies


def test_no_cors_left(admin_any):
    """Same origin through CloudFront, so the CORS gateway responses are gone, no method
    sets CORS headers and there is no OPTIONS method (none without the authorizer)."""
    assert not _res(admin_any, "AWS::ApiGateway::GatewayResponse")
    tpl = json.dumps(_tpl(admin_any))
    assert "Access-Control-" not in tpl
    assert "'*'" not in tpl
    assert all(m["Properties"]["HttpMethod"] != "OPTIONS" for m in _res(admin_any, "AWS::ApiGateway::Method"))


# ---- Routes, Lambda, role --------------------------------------------------------------

def _resource_paths(synth) -> dict[str, str]:
    """Logical ID of each API resource (and the root) -> its full path."""
    api_id = next(lid for s, lid, _ in synth.resources("AWS::ApiGateway::RestApi") if s == ADMIN)
    resources = {lid: r["Properties"] for s, lid, r in synth.resources("AWS::ApiGateway::Resource") if s == ADMIN}
    paths: dict[str, str] = {}

    def path_of(lid: str) -> str:
        if lid not in paths:
            p = resources[lid]
            parent = p["ParentId"]
            base = "" if parent == {"Fn::GetAtt": [api_id, "RootResourceId"]} else path_of(parent["Ref"])
            paths[lid] = f"{base}/{p['PathPart']}"
        return paths[lid]

    for lid in resources:
        path_of(lid)
    paths["__root__"] = "/"
    return paths


def _routes_in(synth) -> set[tuple[str, str]]:
    paths = _resource_paths(synth)
    out = set()
    for m in _res(synth, "AWS::ApiGateway::Method"):
        ref = m["Properties"]["ResourceId"]
        out.add((m["Properties"]["HttpMethod"], paths[ref["Ref"]] if "Ref" in ref else "/"))
    return out


def _api_fn_id(synth) -> str:
    (lid,) = [lid for s, lid, r in synth.resources("AWS::Lambda::Function")
              if s == ADMIN and r["Properties"].get("Handler") == "lambdas.admin_api.handler.handler"]
    return lid


def test_every_route_is_an_api_method_under_api(admin_any):
    """Every route of the handler's own table (lambdas.admin_api.app.ROUTES) is a method under
    /api, and nothing else is (there is no /health MOCK)."""
    from owlbank_infra.contract import admin_api_routes

    want = {(m, f"/api{t}") for m, t in admin_api_routes()}
    assert len(want) == 21  # 15 + the slot routes (GET /slots, PUT /slots/{name},
    # PUT /slots/resolution, POST /slots/preview) + GET and PUT /redaction
    assert _routes_in(admin_any) == want


def test_every_method_is_a_lambda_proxy_with_the_authorizer(admin_any):
    fn = _api_fn_id(admin_any)
    for m in _res(admin_any, "AWS::ApiGateway::Method"):
        p = m["Properties"]
        assert p["AuthorizationType"] == "COGNITO_USER_POOLS"
        assert "AuthorizerId" in p
        integ = p["Integration"]
        assert integ["Type"] == "AWS_PROXY" and integ["IntegrationHttpMethod"] == "POST"
        assert fn in json.dumps(integ["Uri"])
    # Only API Gateway invokes it, per method ARN (no test-invoke permissions, no Function URL).
    perms = _res(admin_any, "AWS::Lambda::Permission")
    api_perms = [p["Properties"] for p in perms if fn in json.dumps(p["Properties"]["FunctionName"])]
    assert len(api_perms) == 21  # one per route (the slot and redaction routes included)
    for p in api_perms:
        assert p["Principal"] == "apigateway.amazonaws.com"
        arn = json.dumps(p["SourceArn"])
        assert "ApiDeploymentStagev1" in arn and "test-invoke-stage" not in arn  # the deployed stage only


def test_api_lambda_is_packaged_like_the_others(admin_any):
    fn = next(r["Properties"] for s, lid, r in admin_any.resources("AWS::Lambda::Function")
              if s == ADMIN and lid == _api_fn_id(admin_any))
    assert fn["Runtime"] == "python3.12" and fn["Architectures"] == ["arm64"]
    assert fn["TracingConfig"] == {"Mode": "Active"}
    assert fn["LoggingConfig"]["LogFormat"] == "JSON"
    log_id = fn["LoggingConfig"]["LogGroup"]["Ref"]
    log = _tpl(admin_any)["Resources"][log_id]["Properties"]
    assert log["LogGroupName"] == "/owlbank/lambda/admin_api"
    assert "OwlBankData" in json.dumps(log["KmsKeyId"])  # the project CMK
    env = fn["Environment"]["Variables"]
    assert set(env) >= {"OWLBANK_ADMIN_USER_POOL_ID", "OWLBANK_TABLE_POLICY", "OWLBANK_TABLE_CATALOG",
                        "OWLBANK_TABLE_ADMIN_AUDIT", "OWLBANK_STAGE"}
    assert "PolicyTable" in json.dumps(env["OWLBANK_TABLE_POLICY"])
    assert "CatalogTable" in json.dumps(env["OWLBANK_TABLE_CATALOG"])
    assert "AuditTable" in json.dumps(env["OWLBANK_TABLE_ADMIN_AUDIT"])
    assert env["OWLBANK_STAGE"] in ("dev", "prod")
    assert "Layers" not in fn  # needs only the runtime's boto3
    assert not _res(admin_any, "AWS::Lambda::Url")


def _api_env(synth) -> dict:
    fn = next(r["Properties"] for s, lid, r in synth.resources("AWS::Lambda::Function")
              if s == ADMIN and lid == _api_fn_id(synth))
    return fn["Environment"]["Variables"]


@pytest.fixture(scope="module")
def admin_spoken_off(tmp_path_factory):
    return _synth(tmp_path_factory, "admin-spoken-off",
                  {"adminConsole": "true", "acceptSpokenFactors": "false"})


def test_api_env_carries_accept_spoken_factors(admin_dev, admin_prod, admin_spoken_off):
    """The admin API gets the deploy's ``acceptSpokenFactors``, as the front
    door does: ``"true"`` by default in every stage, ``"false"`` with the override."""
    assert _api_env(admin_dev)["OWLBANK_ACCEPT_SPOKEN_FACTORS"] == "true"
    assert _api_env(admin_prod)["OWLBANK_ACCEPT_SPOKEN_FACTORS"] == "true"
    assert _api_env(admin_spoken_off)["OWLBANK_ACCEPT_SPOKEN_FACTORS"] == "false"


def test_override_changes_only_the_env(admin_dev, admin_spoken_off):
    """The override adds no permission: the API role's statements are the same (the start-up
    entry is a ``PutItem`` on the audit table, which the role already has)."""
    def statements(synth):
        fn = next(r["Properties"] for s, lid, r in synth.resources("AWS::Lambda::Function")
                  if s == ADMIN and lid == _api_fn_id(synth))
        role_id = fn["Role"]["Fn::GetAtt"][0]
        return [p["Properties"]["PolicyDocument"] for p in _res(synth, "AWS::IAM::Policy")
                if {"Ref": role_id} in p["Properties"]["Roles"]]
    assert statements(admin_dev) == statements(admin_spoken_off)


def test_cdk_nag_clean_with_the_override(admin_spoken_off):
    test_cdk_nag_clean(admin_spoken_off)


def test_api_role_reaches_only_policy_catalog_audit_and_the_pool(admin_any):
    fn = next(r["Properties"] for s, lid, r in admin_any.resources("AWS::Lambda::Function")
              if s == ADMIN and lid == _api_fn_id(admin_any))
    role_id = fn["Role"]["Fn::GetAtt"][0]
    role = _tpl(admin_any)["Resources"][role_id]["Properties"]
    assert "ManagedPolicyArns" not in role and "Policies" not in role
    assert role["AssumeRolePolicyDocument"]["Statement"][0]["Principal"] == {"Service": "lambda.amazonaws.com"}
    policies = [r["Properties"] for r in _res(admin_any, "AWS::IAM::Policy")
                if {"Ref": role_id} in r["Properties"]["Roles"]]
    statements = [s for p in policies for s in p["PolicyDocument"]["Statement"]]
    audit_id = next(lid for s, lid, _ in admin_any.resources("AWS::DynamoDB::Table") if s == ADMIN)
    pool_id = next(lid for s, lid, _ in admin_any.resources("AWS::Cognito::UserPool") if s == ADMIN)
    seen_actions = set()
    for s in statements:
        assert s["Effect"] == "Allow"
        actions = s["Action"] if isinstance(s["Action"], list) else [s["Action"]]
        seen_actions |= set(actions)
        resources = s["Resource"] if isinstance(s["Resource"], list) else [s["Resource"]]
        services = {a.split(":")[0] for a in actions}
        assert len(services) == 1, s
        (service,) = services
        if service == "dynamodb" and s.get("Sid") == "PolicyBatchRead":
            # 18.14: the policy reload's one BatchGetItem, on the policy table only.
            assert actions == ["dynamodb:BatchGetItem"]
            assert len(resources) == 1 and "PolicyTable" in json.dumps(resources[0])
            assert "index" not in json.dumps(resources)
        elif service == "dynamodb" and s.get("Sid") == "CatalogScan":
            # 18.14: the catalog reload's one Scan, on the catalog table only.
            assert actions == ["dynamodb:Scan"]
            assert len(resources) == 1 and "CatalogTable" in json.dumps(resources[0])
            assert "index" not in json.dumps(resources)
        elif service == "dynamodb":
            assert s.get("Sid") == "ConfigAndAuditTables", s
            assert sorted(actions) == ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:Query"]
            names = [json.dumps(r) for r in resources]
            assert len(names) == 3
            assert any("PolicyTable" in n and "ImportValue" in n for n in names)
            assert any("CatalogTable" in n and "ImportValue" in n for n in names)
            assert {"Fn::GetAtt": [audit_id, "Arn"]} in resources
            assert not any("index" in n for n in names)
        elif service == "cognito-idp":
            assert resources == [{"Fn::GetAtt": [pool_id, "Arn"]}]
            assert set(actions) == {
                "cognito-idp:ListUsers", "cognito-idp:ListUsersInGroup", "cognito-idp:AdminGetUser",
                "cognito-idp:AdminListGroupsForUser", "cognito-idp:AdminCreateUser",
                "cognito-idp:AdminAddUserToGroup", "cognito-idp:AdminRemoveUserFromGroup",
                "cognito-idp:AdminDisableUser", "cognito-idp:AdminEnableUser", "cognito-idp:AdminDeleteUser",
                "cognito-idp:AdminUserGlobalSignOut"}
        elif service == "kms":
            assert set(actions) == {"kms:Decrypt", "kms:DescribeKey", "kms:Encrypt", "kms:GenerateDataKey"}
            assert s["Condition"] == {"StringEquals": {"kms:ViaService": "dynamodb.us-east-1.amazonaws.com"}}
            assert "OwlBankData" in json.dumps(resources)
        elif service == "logs":
            assert set(actions) == {"logs:CreateLogStream", "logs:PutLogEvents"}
            assert "*" not in json.dumps(resources)
        elif service == "xray":
            assert set(actions) == {"xray:PutTraceSegments", "xray:PutTelemetryRecords"} and resources == ["*"]
        else:
            raise AssertionError(f"unexpected service {service}: {s}")
    assert not any(a.endswith("*") for a in seen_actions)
    assert not {"dynamodb:DeleteItem", "dynamodb:UpdateItem", "dynamodb:BatchWriteItem"} & seen_actions
    sids = [s.get("Sid") for s in statements if any(
        a.startswith("dynamodb:") for a in (s["Action"] if isinstance(s["Action"], list) else [s["Action"]]))]
    assert sorted(sids) == ["CatalogScan", "ConfigAndAuditTables", "PolicyBatchRead"]


# ---- The API through the console's CloudFront ------------------------------------------

def _api_behavior(synth):
    cfg = _one(synth, "AWS::CloudFront::Distribution")["DistributionConfig"]
    (behavior,) = cfg["CacheBehaviors"]
    origin = next(o for o in cfg["Origins"] if o["Id"] == behavior["TargetOriginId"])
    return behavior, origin


def _origin_verify_ref(synth) -> dict:
    secret_id = next(lid for s, lid, _ in synth.resources("AWS::SecretsManager::Secret") if s == ADMIN)
    return {"Fn::Join": ["", ["{{resolve:secretsmanager:", {"Ref": secret_id}, ":SecretString:::}}"]]}


def test_api_behaviour_goes_to_the_api_with_the_origin_verify_header(admin_any):
    behavior, origin = _api_behavior(admin_any)
    assert behavior["PathPattern"] == "/api/*"
    assert behavior["ViewerProtocolPolicy"] == "https-only"
    assert set(behavior["AllowedMethods"]) == {"GET", "HEAD", "OPTIONS", "PUT", "PATCH", "POST", "DELETE"}
    assert "ResponseHeadersPolicyId" in behavior  # CSP, HSTS and nosniff on API responses too
    api_id = next(lid for s, lid, _ in admin_any.resources("AWS::ApiGateway::RestApi") if s == ADMIN)
    assert origin["DomainName"] == {"Fn::Join": ["", [{"Ref": api_id}, ".execute-api.us-east-1.",
                                                      {"Ref": "AWS::URLSuffix"}]]}
    # /api/tools -> https://<api>/v1/api/tools, the API resource /api/tools.
    assert origin["OriginPath"] == "/v1"
    assert origin["CustomOriginConfig"] == {"OriginProtocolPolicy": "https-only", "OriginSSLProtocols": ["TLSv1.2"]}
    assert origin["OriginCustomHeaders"] == [{"HeaderName": "X-Origin-Verify",
                                              "HeaderValue": _origin_verify_ref(admin_any)}]


def test_api_behaviour_forwards_authorization_query_and_content_type_never_cookies(admin_any):
    behavior, _ = _api_behavior(admin_any)
    cache = _tpl(admin_any)["Resources"][behavior["CachePolicyId"]["Ref"]]["Properties"]["CachePolicyConfig"]
    assert (cache["MinTTL"], cache["DefaultTTL"], cache["MaxTTL"]) == (0, 0, 1)
    key = cache["ParametersInCacheKeyAndForwardedToOrigin"]
    assert key["HeadersConfig"] == {"HeaderBehavior": "whitelist", "Headers": ["Authorization"]}
    assert key["QueryStringsConfig"] == {"QueryStringBehavior": "all"}
    assert key["CookiesConfig"] == {"CookieBehavior": "none"}
    orp = _tpl(admin_any)["Resources"][behavior["OriginRequestPolicyId"]["Ref"]]["Properties"]["OriginRequestPolicyConfig"]
    assert orp["HeadersConfig"] == {"HeaderBehavior": "whitelist", "Headers": ["Content-Type"]}
    assert orp["QueryStringsConfig"] == {"QueryStringBehavior": "all"}
    assert orp["CookiesConfig"] == {"CookieBehavior": "none"}


def test_origin_verify_secret_is_generated_with_the_cmk(admin_any):
    secret = _one(admin_any, "AWS::SecretsManager::Secret")
    assert secret["GenerateSecretString"] == {"ExcludePunctuation": True, "PasswordLength": 48}
    assert "SecretString" not in secret
    assert "OwlBankData" in json.dumps(secret["KmsKeyId"])
    # The value is only ever a dynamic reference.
    assert json.dumps(_tpl(admin_any)).count("{{resolve:secretsmanager:") == 2


def test_api_waf_blocks_requests_without_the_origin_verify_header(admin_any):
    acls = {r["Properties"]["Scope"]: r["Properties"] for s, _, r in admin_any.resources("AWS::WAFv2::WebACL") if s == ADMIN}
    rules = {r["Name"]: r for r in acls["REGIONAL"]["Rules"]}
    rule = rules["OriginVerifyHeaderRequired"]
    assert rule["Action"] == {"Block": {}}
    match = rule["Statement"]["NotStatement"]["Statement"]["ByteMatchStatement"]
    assert match["FieldToMatch"] == {"SingleHeader": {"Name": "x-origin-verify"}}
    assert match["PositionalConstraint"] == "EXACTLY"
    assert match["SearchString"] == _origin_verify_ref(admin_any)
    assert match["TextTransformations"] == [{"Priority": 0, "Type": "NONE"}]
    assert "OriginVerifyHeaderRequired" not in {r["Name"] for r in acls["CLOUDFRONT"]["Rules"]}
    for conf in _res(admin_any, "AWS::WAFv2::LoggingConfiguration"):
        assert {"SingleHeader": {"Name": "x-origin-verify"}} in conf["Properties"]["RedactedFields"]


def _allowed(acl: dict) -> list[tuple[re.Pattern, set[str]]]:
    rule = next(r for r in acl["Rules"] if r["Name"] == "AllowedPathsAndMethodsOnly")
    inner = rule["Statement"]["NotStatement"]["Statement"]
    stmts = inner["OrStatement"]["Statements"] if "OrStatement" in inner else [inner]
    out = []
    for st in stmts:
        path, method = st["AndStatement"]["Statements"]
        ms = method["OrStatement"]["Statements"] if "OrStatement" in method else [method]
        out.append((re.compile(path["RegexMatchStatement"]["RegexString"]),
                    {m["ByteMatchStatement"]["SearchString"] for m in ms}))
    return out


def _passes(allowed, method: str, path: str) -> bool:
    return any(rx.search(path) and method in ms for rx, ms in allowed)


@pytest.mark.parametrize("method,path,ok", [
    ("GET", "/api/tools", True), ("PUT", "/api/tools/get_balance", True),
    ("GET", "/api/prompts", True), ("PUT", "/api/prompts/mask.get_balance.1", True),
    ("GET", "/api/placeholders/get_balance", True), ("GET", "/api/audit", True),
    ("POST", "/api/restore", True), ("GET", "/api/users", True), ("POST", "/api/users", True),
    ("PATCH", "/api/users/ops%40example.com", True), ("DELETE", "/api/users/ops%40example.com", True),
    ("POST", "/api/users/ops%40example.com/resend", True), ("PUT", "/api/masking", True),
    ("GET", "/api/redaction", True), ("PUT", "/api/redaction", True), ("POST", "/api/redaction", False),
    ("OPTIONS", "/api/tools", False), ("DELETE", "/api/tools/get_balance", False),
    ("GET", "/api/health", False), ("GET", "/api/customers", False), ("POST", "/api/tools", False),
])
def test_waf_allow_lists_cover_exactly_the_api_routes(admin_dev, method, path, ok):
    acls = {r["Properties"]["Scope"]: r["Properties"] for s, _, r in admin_dev.resources("AWS::WAFv2::WebACL") if s == ADMIN}
    assert _passes(_allowed(acls["REGIONAL"]), method, "/v1" + path) is ok  # API Gateway: stage first
    assert _passes(_allowed(acls["CLOUDFRONT"]), method, path) is ok  # CloudFront: as the viewer sends it


def test_site_waf_still_serves_the_static_files(admin_dev):
    acls = {r["Properties"]["Scope"]: r["Properties"] for s, _, r in admin_dev.resources("AWS::WAFv2::WebACL") if s == ADMIN}
    site = _allowed(acls["CLOUDFRONT"])
    for path in ("/", "/index.html", "/app.js", "/views/tools.js", "/config.js",
                 # the admin-console-look files.
                 "/fonts/Inter-4.1-latin-var.woff2", "/icons.svg", "/owl.svg", "/fonts/OFL.txt",
                 "/theme.js", "/tokens.css"):
        assert _passes(site, "GET", path) and _passes(site, "HEAD", path), path
    assert not _passes(site, "POST", "/index.html")



def _rules(acl: dict) -> dict[str, dict]:
    return {r["Name"]: r for r in acl["Rules"]}


def _size(stmt: dict) -> int:
    return stmt["SizeConstraintStatement"]["Size"]


def _path_regex(stmt: dict) -> re.Pattern:
    return re.compile(stmt["RegexMatchStatement"]["RegexString"])


@pytest.mark.parametrize("scope,prefix", [("CLOUDFRONT", ""), ("REGIONAL", "/v1")])
def test_prompts_write_gets_the_large_body_cap_on_both_acls(admin_any, scope, prefix):
    """The prompts write may send 64 KB (every language of a long agent.* text); every other route
    keeps 8 KB, and the Common Rule Set's own 8 KB limit is lifted on that one path only."""
    acls = {r["Properties"]["Scope"]: r["Properties"] for s, _, r in admin_any.resources("AWS::WAFv2::WebACL") if s == ADMIN}
    rules = _rules(acls[scope])
    small, not_big = rules["BodySizeLimit"]["Statement"]["AndStatement"]["Statements"]
    assert _size(small) == MAX_BODY_BYTES == 8192
    (big_path,) = [_path_regex(not_big["NotStatement"]["Statement"])]
    large_size, large_path = rules["LargeBodySizeLimit"]["Statement"]["AndStatement"]["Statements"]
    assert _size(large_size) == PROMPTS_MAX_BODY_BYTES == 65536
    assert _path_regex(large_path).pattern == big_path.pattern
    oversize = rules["OversizeBodyOutsideLargeBodyPaths"]["Statement"]["AndStatement"]["Statements"]
    assert _path_regex(oversize[1]["NotStatement"]["Statement"]).pattern == big_path.pattern
    for path, on in ((f"{prefix}/api/prompts/agent.system.haiku-5-5", True),
                     (f"{prefix}/api/prompts/mask.get_balance.1", True),
                     (f"{prefix}/api/prompts", False), (f"{prefix}/api/restore", False),
                     (f"{prefix}/api/tools/get_balance", False), (f"{prefix}/api/users", False),
                     (f"{prefix}/api/prompts/a/b", False)):
        assert bool(big_path.search(path)) is on, path
    crs = rules["AWSManagedRulesCommonRuleSet"]
    if crs["OverrideAction"] == {"None": {}}:  # block mode; in count mode the whole group counts
        assert crs["Statement"]["ManagedRuleGroupStatement"]["RuleActionOverrides"] == [
            {"Name": "SizeRestrictions_BODY", "ActionToUse": {"Count": {}}}]
    # Each ACL inspects the whole capped body (API Gateway's default would be 16 KB).
    (limits,) = acls[scope]["AssociationConfig"]["RequestBody"].values()
    assert limits == {"DefaultSizeInspectionLimit": "KB_64"}
    assert set(acls[scope]["AssociationConfig"]["RequestBody"]) == (
        {"CLOUDFRONT"} if scope == "CLOUDFRONT" else {"API_GATEWAY"})


def _max_prompt_body(languages: list[str]) -> str:
    """The console's prompts write (``{rev, texts}``, ``JSON.stringify``: compact, non-ASCII not
    escaped) with every language at the API's longest agent.* text. The worst case for these
    languages' text is two UTF-8 bytes per character (accented Latin letters; a quote or a line
    break is escaped to two bytes too)."""
    from owlbank_infra.contract import _import_from_src
    chars = _import_from_src("lambdas.admin_api.config_routes").MAX_AGENT_PROMPT_CHARS
    texts = dict.fromkeys(languages, "\u00f1" * chars)
    return json.dumps({"rev": 999_999_999, "texts": texts}, separators=(",", ":"), ensure_ascii=False)


def test_max_size_agent_prompt_write_fits_under_the_waf_and_server_caps():
    from owlbank_infra.contract import DATA_DIR, _import_from_src
    languages = list(json.loads((DATA_DIR / "prompt_catalog.json").read_text())["languages"])
    assert len(languages) == 2
    body = _max_prompt_body(languages)
    assert len(json.loads(body)["texts"][languages[0]]) == 16_000
    size = len(body.encode("utf-8"))
    assert 64_000 < size <= PROMPTS_MAX_BODY_BYTES
    http = _import_from_src("lambdas.admin_api.http")
    assert http.MAX_BODY_BYTES == PROMPTS_MAX_BODY_BYTES  # the WAF and the handler agree
    assert http.json_body({"body": body})["texts"].keys() == set(languages)


def test_auto_delete_provider_logs_to_a_cmk_group_with_retention(admin_dev, admin_prod):
    """Dev empties the site bucket on delete through CDK's provider Lambda; its log group is ours,
    with the CMK and the stack's retention, and exists before the Lambda. Prod has no provider."""
    tpl = _tpl(admin_dev)["Resources"]
    (fn_id,) = [lid for lid, r in tpl.items() if r["Type"] == "AWS::Lambda::Function"
                and lid.startswith("CustomS3AutoDeleteObjectsCustomResourceProviderHandler")]
    group_id = tpl[fn_id]["Properties"]["LoggingConfig"]["LogGroup"]["Ref"]
    assert group_id in tpl[fn_id]["DependsOn"]
    group = tpl[group_id]["Properties"]
    assert group["LogGroupName"] == "/owlbank/admin/site-auto-delete"
    assert "OwlBankData" in json.dumps(group["KmsKeyId"]) and group["RetentionInDays"] == 30
    assert not [r for s, _, r in admin_prod.resources("Custom::S3AutoDeleteObjects") if s == ADMIN]
    assert "site-auto-delete" not in json.dumps(_tpl(admin_prod))


def test_every_admin_log_group_has_a_cmk_and_retention_and_every_lambda_one_of_them(admin_any):
    groups = {lid: r["Properties"] for s, lid, r in admin_any.resources("AWS::Logs::LogGroup") if s == ADMIN}
    for lid, g in groups.items():
        assert "KmsKeyId" in g and "RetentionInDays" in g, lid
    for s, lid, r in admin_any.resources("AWS::Lambda::Function"):
        if s == ADMIN:
            assert r["Properties"]["LoggingConfig"]["LogGroup"]["Ref"] in groups, lid


# ---- the site's files and config.js ----------------------------------------------------

#: The font's file name: its version is in the name, so a one-year immutable cache is safe.
FONT = "Inter-4.1-latin-var.woff2"


def _deployments(synth):
    """``(main, font)``: the two ``Custom::CDKBucketDeployment``s, told apart by the font's
    ``fonts/`` key prefix."""
    deps = _res(synth, "Custom::CDKBucketDeployment")
    assert len(deps) == 2, len(deps)
    (font,) = [d["Properties"] for d in deps if d["Properties"].get("DestinationBucketKeyPrefix") == "fonts/"]
    (main,) = [d["Properties"] for d in deps if "DestinationBucketKeyPrefix" not in d["Properties"]]
    return main, font


def test_site_deployment_with_generated_config_js(admin_any):
    main, font = _deployments(admin_any)
    bucket_id = next(lid for s, lid, _ in admin_any.resources("AWS::S3::Bucket") if s == ADMIN)
    dist_id = next(lid for s, lid, _ in admin_any.resources("AWS::CloudFront::Distribution") if s == ADMIN)
    for p in (main, font):
        assert p["DestinationBucketName"] == {"Ref": bucket_id}
        assert p["DistributionId"] == {"Ref": dist_id}
        assert p["Prune"] is True
    assert main["DistributionPaths"] == ["/*"]
    assert len(main["SourceObjectKeys"]) == 2  # the site's files and config.js
    # The main deployment's prune must leave the font alone: excluded keys are never deleted.
    assert "fonts/*.woff2" in main["Exclude"]
    markers = json.dumps(main["SourceMarkers"])
    pool_id = next(lid for s, lid, _ in admin_any.resources("AWS::Cognito::UserPool") if s == ADMIN)
    client_id = next(lid for s, lid, _ in admin_any.resources("AWS::Cognito::UserPoolClient") if s == ADMIN)
    assert pool_id in markers and client_id in markers and dist_id in markers


def test_font_deployment_serves_woff2_with_a_one_year_immutable_cache(admin_any):
    """An explicit type, not the handler's MIME guess; ``fonts/OFL.txt`` (the main
    deployment's) is excluded so this deployment's prune keeps it."""
    _, font = _deployments(admin_any)
    assert len(font["SourceObjectKeys"]) == 1
    assert font["SystemMetadata"]["content-type"] == "font/woff2"
    assert font["SystemMetadata"]["cache-control"] == "public, max-age=31536000, immutable"
    assert font["Exclude"] == ["*.txt"]
    assert font["DistributionPaths"] == ["/fonts/*"]


def _config_doc(text: str) -> dict:
    return json.loads(text.split("Object.freeze(", 1)[1].rsplit(");", 1)[0])


def test_config_js_is_an_es_module_with_the_api_base():
    from owlbank_infra.admin_stack import config_js

    text = config_js(region="us-east-1", user_pool_id="us-east-1_abc", client_id="client123",
                     login_domain="owlbank-dev-admin-1.auth.us-east-1.amazoncognito.com",
                     redirect_uri="https://d111.cloudfront.net/", stage="dev")
    assert _config_doc(text) == {"region": "us-east-1", "userPoolId": "us-east-1_abc",
                                 "clientId": "client123",
                                 "loginDomain": "owlbank-dev-admin-1.auth.us-east-1.amazoncognito.com",
                                 "redirectUri": "https://d111.cloudfront.net/", "apiBase": "/api",
                                 "stage": "dev"}
    assert text.lstrip().startswith("//") and "export default" in text


@pytest.mark.parametrize("stage", ["dev", "test", "prod"])
def test_cpl15_config_js_round_trips_the_stage(stage):
    """The console's stage badge reads ``stage`` from config.js.
    Three values, so the exhaustive check stands in for a property test."""
    from owlbank_infra.admin_stack import config_js

    text = config_js(region="us-east-1", user_pool_id="p", client_id="c", login_domain="l",
                     redirect_uri="https://d111.cloudfront.net/", stage=stage)
    assert _config_doc(text)["stage"] == stage


def test_cpl15_site_deployment_config_js_carries_the_stack_stage(admin_dev, admin_prod):
    """The call site passes ``cfg.stage``: the staged ``config.js`` asset names the stage."""
    for synth, stage in ((admin_dev, "dev"), (admin_prod, "prod")):
        configs = [d / "config.js" for d in synth.outdir.glob("asset.*") if (d / "config.js").exists()]
        assert len(configs) == 1, "config.js is a single data asset"
        assert _config_doc(configs[0].read_text())["stage"] == stage


#: The console's CSP as it was before the admin-console-look feature, captured from
#: ``content_security_policy``. It must not change.
PINNED_CSP = ("default-src 'self'; "
              "connect-src 'self' https://owlbank-dev-admin-1.auth.us-east-1.amazoncognito.com; "
              "form-action 'self' https://owlbank-dev-admin-1.auth.us-east-1.amazoncognito.com; "
              "frame-ancestors 'none'; base-uri 'self'; object-src 'none'; img-src 'self' data:")


def test_cpl13_csp_is_pinned_to_the_exact_pre_feature_string():
    """No new origin, no inline script or style, no 'unsafe-inline'."""
    from owlbank_infra.admin_stack import content_security_policy

    csp = content_security_policy(
        login_origin="https://owlbank-dev-admin-1.auth.us-east-1.amazoncognito.com")
    assert csp == PINNED_CSP


def test_site_asset_has_the_console_without_its_tests(admin_dev):
    sites = [d for d in admin_dev.outdir.glob("asset.*") if (d / "index.html").exists()
             and (d / "app.js").exists()]
    assert sites, "the console's files are an asset"
    for d in sites:
        assert not (d / "tests").exists()
        assert not (d / "config.js").exists()  # generated at deploy, never shipped from the repo
        for name in ("icons.svg", "owl.svg", "theme.js", "tokens.css", "fonts/OFL.txt"):
            assert (d / name).is_file(), name
        assert not (d / "fonts" / "README.md").exists()
        assert not list(d.rglob("*.woff2")), "the font is the font deployment's"
    fonts = [d for d in admin_dev.outdir.glob("asset.*") if (d / FONT).exists()]
    assert fonts, "the font is an asset"
    for d in fonts:
        assert sorted(f.name for f in d.rglob("*") if f.is_file()) == [FONT]


def test_admin_origin_verify_secret_version_pins_the_references(tmp_path_factory):
    version = "a1b2c3d4-0000-1111-2222-333344445555"
    synth = _synth(tmp_path_factory, "admin-pinned", {"adminConsole": "true",
                                                      "adminOriginVerifySecretVersion": version})
    assert json.dumps(_tpl(synth)).count(f":SecretString:::{version}}}}}") == 2


def test_bad_admin_origin_verify_secret_version_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="adminOriginVerifySecretVersion"):
        build_app({"adminOriginVerifySecretVersion": "nope"}, outdir=str(tmp_path))


# ---- WAF --------------------------------------------------------------------------

def test_waf_on_distribution_and_api(admin_any):
    acls = {lid: r["Properties"] for s, lid, r in admin_any.resources("AWS::WAFv2::WebACL") if s == ADMIN}
    scopes = {p["Scope"]: lid for lid, p in acls.items()}
    assert set(scopes) == {"CLOUDFRONT", "REGIONAL"}
    cfg = _one(admin_any, "AWS::CloudFront::Distribution")["DistributionConfig"]
    assert cfg["WebACLId"] == {"Fn::GetAtt": [scopes["CLOUDFRONT"], "Arn"]}
    (assoc,) = _res(admin_any, "AWS::WAFv2::WebACLAssociation")
    assert assoc["Properties"]["WebACLArn"] == {"Fn::GetAtt": [scopes["REGIONAL"], "Arn"]}
    assert "ApiDeploymentStagev1" in json.dumps(assoc["Properties"]["ResourceArn"])
    for lid, p in acls.items():
        rules = {r["Name"]: r for r in p["Rules"]}
        assert "AWSManagedRulesCommonRuleSet" in rules
        assert rules["RateLimitPerIp"]["Statement"]["RateBasedStatement"]["AggregateKeyType"] == "IP"
    logged = {json.dumps(r["Properties"]["ResourceArn"]) for r in _res(admin_any, "AWS::WAFv2::LoggingConfiguration")}
    assert logged == {json.dumps({"Fn::GetAtt": [lid, "Arn"]}) for lid in acls}


# ---- outputs, properties, cdk-nag ------------------------------------------------------------

def test_outputs(admin_any):
    outputs = _tpl(admin_any)["Outputs"]
    for name in ("ConsoleUrl", "UserPoolId", "UserPoolClientId", "ManagedLoginDomain", "InternalApiEndpoint"):
        assert name in outputs, name
    # The API is used through the console's origin, not its execute-api URL.
    assert "execute-api" not in json.dumps(outputs["InternalApiEndpoint"])


def test_properties(admin_dev):
    admin = admin_dev.stacks["admin"]
    for attr in ("user_pool", "user_pool_client", "user_pool_domain", "site_bucket", "distribution",
                 "audit_table", "api", "authorizer", "console_url", "console_origin", "login_domain",
                 "groups", "site_waf", "api_waf", "headers_policy", "api_function", "origin_verify",
                 "api_cache_policy", "api_origin_request_policy", "site_deployment"):
        assert getattr(admin, attr) is not None, attr


def test_cdk_nag_clean(admin_any):
    report = _nag_report(admin_any)
    if report is not None:
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [(v["ruleName"], [c["constructPath"] for c in v["violatingConstructs"]]) for v in errors]
        assert report["conclusion"] == "success"


def test_invitation_body_names_the_console_url_output(admin_any):
    pool = _one(admin_any, "AWS::Cognito::UserPool")
    body = pool["AdminCreateUserConfig"]["InviteMessageTemplate"]["EmailMessage"]
    assert "ConsoleUrl" in body and "{####}" in body and "{username}" in body
    assert "InternalApiEndpoint" not in body


def test_invalidation_policy_is_scoped_to_the_distribution_arn(admin_any):
    """8.7: no cloudfront action is granted on '*' anywhere in the Admin stack."""
    dist_id = next(lid for s, lid, _ in admin_any.resources("AWS::CloudFront::Distribution") if s == ADMIN)
    found = 0
    for _s, _lid, res in admin_any.resources("AWS::IAM::Policy"):
        if _s != ADMIN:
            continue
        for st in res["Properties"]["PolicyDocument"]["Statement"]:
            actions = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
            if any(a.startswith("cloudfront:") for a in actions):
                found += 1
                assert st["Resource"] != "*", st
                assert dist_id in json.dumps(st["Resource"]) and "distribution/" in json.dumps(st["Resource"])
    assert found == 1



def test_deployment_handler_policy_keeps_s3_statements_and_scopes_cloudfront(admin_any):
    """The handler's policy is rewritten from a snapshot; the S3 asset and site-bucket grants must survive."""
    (policy,) = [r["Properties"] for s, lid, r in admin_any.resources("AWS::IAM::Policy")
                 if s == ADMIN and lid.startswith("CustomCDKBucketDeployment") and "DefaultPolicy" in lid]
    statements = policy["PolicyDocument"]["Statement"]

    def acts(st):
        return st["Action"] if isinstance(st["Action"], list) else [st["Action"]]

    def res(st):
        return json.dumps(st["Resource"])

    s3 = [st for st in statements if any(a.startswith("s3:") for a in acts(st))]
    assert any("s3:GetObject*" in acts(st) and "cdk-" in res(st) and "assets" in res(st) for st in s3), s3
    site_id = next(lid for s, lid, _ in admin_any.resources("AWS::S3::Bucket") if s == ADMIN)
    assert any({"s3:DeleteObject*", "s3:Abort*"} <= set(acts(st)) and site_id in res(st) for st in s3), s3
    (cf,) = [st for st in statements if "cloudfront:CreateInvalidation" in acts(st)]
    assert cf["Resource"] != "*" and "distribution/" in res(cf)
