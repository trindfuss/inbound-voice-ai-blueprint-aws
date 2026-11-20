"""Admin: the admin console.

* **Cognito**: an admin-created-users-only pool (self sign-up off, email sign-in, a strong
  password policy, TOTP MFA required at ``stage=prod`` and off otherwise), the groups ``admin``,
  ``admin-editor`` and ``admin-viewer``, a public app client (authorization code + PKCE, no secret)
  whose callback and logout URL is the console's CloudFront URL, and a Cognito prefix domain with
  managed login. With ``adminEmail`` the first user is created and put in ``admin``; Cognito
  emails them a temporary password.
* **Site**: a private S3 bucket behind CloudFront with origin access control, HTTPS only,
  a response-headers policy (CSP, HSTS, no framing, nosniff) and a CLOUDFRONT-scope web ACL. The
  files in ``src/admin_console`` and a generated ``config.js`` are deployed with
  ``BucketDeployment``.
* **Audit table**: ``admin_audit``, pk/sk strings, the project CMK, TTL, PITR.
* **API**: a regional REST API, one Lambda (``lambdas.admin_api``) behind a Lambda
  proxy integration on every route of :func:`.contract.admin_api_routes`, each method with the
  Cognito authorizer; a regional web ACL, access logs and throttling.
* **Same origin**: the distribution's second behaviour, ``/api/*``, goes to the API's
  regional endpoint with origin path ``/<stage>``, so ``/api/tools`` reaches the API resource
  ``/api/tools`` (the handler strips ``/api``). No caching, every method, ``Authorization``,
  ``Content-Type`` and the query string forwarded, never cookies. The browser sees one origin: no
  CORS, no preflight, no unauthenticated method. CloudFront adds ``X-Origin-Verify`` (a generated
  Secrets Manager secret) and the API's web ACL blocks any request without it, so the
  API is reachable only through the distribution.

Nothing in the call path depends on this stack, so ``cdk destroy OwlBankAdmin`` removes it
without touching calls. In ``dev`` and ``test`` its stateful resources are destroyed
with it; at ``prod`` they are retained.

Dependency order inside the stack, so there is no cycle: the managed-login domain is a string
built from its prefix; the API (and its Lambda) needs only the pool and the tables; the
distribution references the API's ID and the domain string; the app client references the
distribution's domain name; the site deployment references the client.
"""

from __future__ import annotations

import json
import re
from typing import cast

from aws_cdk import CfnOutput, CfnResource, Duration, RemovalPolicy, SecretValue, Stack
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_cloudfront as cf
from aws_cdk import aws_cloudfront_origins as cf_origins
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3deploy
from aws_cdk import aws_secretsmanager as sm
from aws_cdk import aws_wafv2 as waf
from constructs import Construct

from .compute_stack import LAMBDA_SRC_DIR, stage_lambda_source
from .config import InfraConfig
from .contract import REPO_ROOT, TTL_ATTRIBUTE, admin_api_routes
from .grants import CATALOG_TABLE, catalog_scan_statement
from .nag import ack
from .observability import log_group, waf_logging
from .waf import web_acl

#: The console's groups: ``admin`` manages users and edits, ``admin-editor``
#: views and edits, ``admin-viewer`` views only.
GROUPS = {
    "admin": "Admin console: everything an editor can do, plus manage users",
    "admin-editor": "Admin console: view and edit configuration",
    "admin-viewer": "Admin console: view only",
}
FIRST_USER_GROUP = "admin"
API_STAGE = "v1"
#: The console's path prefix for its API: the CloudFront behaviour ``/api/*`` and the
#: API's resource tree both start with it, and the site's ``config.js`` names it.
API_PREFIX = "/api"
API_BEHAVIOR = f"{API_PREFIX}/*"
ORIGIN_VERIFY_HEADER = "X-Origin-Verify"
SITE_DIR = REPO_ROOT / "src" / "admin_console"
#: The font files: their own deployment (``FontDeployment``) gives them ``font/woff2`` and a
#: one-year immutable cache (their names carry the version). The main deployment excludes them
#: from both its asset and its prune.
FONT_GLOB = "fonts/*.woff2"
FONT_CACHE_CONTROL = "public, max-age=31536000, immutable"
#: Not uploaded by the main deployment: the Node tests, the package.json that marks the files as
#: ES modules for them, the docs, and the font (``FontDeployment`` uploads it).
SITE_EXCLUDE = ["tests", "tests/*", "package.json", "*.md", FONT_GLOB]
#: One path segment of a route parameter as a WAF sees it (raw, percent-encoded): a tool name, a
#: prompt key (``mask.get_balance.1``) or a username (an email; the console sends ``@`` as ``%40``).
PATH_SEGMENT = r"[A-Za-z0-9%._~+@-]{1,256}"
#: The construct id CDK gives the auto-delete provider (one per stack, ``s3.Bucket``).
AUTO_DELETE_PROVIDER_ID = "Custom::S3AutoDeleteObjectsCustomResourceProvider"
#: Token lifetimes (tokens expire within an hour).
ACCESS_ID_TOKEN_MINUTES = 60
REFRESH_TOKEN_HOURS = 8
#: The API's stage throttling: a handful of operators, so low limits.
API_RATE_LIMIT, API_BURST_LIMIT = 10, 20
HSTS_SECONDS = 365 * 24 * 3600
#: Both admin web ACLs' rate rule, requests per 5 minutes per source IP. The console's users are
#: a handful of operators (a page load is a few requests), unlike Twilio's shared egress IPs that
#: the call path's ACLs must allow for, so the limit is low and not tied to those settings.
WAF_RATE_LIMIT = 1000
#: Every route's body cap but the prompts write: a tool row, a spoken prompt (600 characters per
#: language), a user or a restore are small JSON objects.
MAX_BODY_BYTES = 8192
#: The prompts write (``PUT /api/prompts/{key}``) sends every language's text in one body, and an
#: ``agent.*`` text may be 16,000 characters, so its body passes 8 KB (a shipped system prompt in
#: two languages is 10-13 KB). It gets the server's own cap (``admin_api.http.MAX_BODY_BYTES``),
#: which is also the most a WAF body inspection can cover.
PROMPTS_MAX_BODY_BYTES = 65536
#: The admin API Lambda: a request reads a few items and writes one transaction; Cognito
#: calls for user management are the slowest part.
API_FN_TIMEOUT_S = 15
API_FN_MEMORY_MB = 512
#: Exactly what the admin API's role may do. DynamoDB on the policy, catalog and
#: audit tables only (writes are ``TransactWriteItems`` of conditional Puts, which IAM authorizes
#: as ``PutItem``); Cognito on the one user pool.
API_TABLE_ACTIONS = ("dynamodb:GetItem", "dynamodb:Query", "dynamodb:PutItem")
API_TABLES = ("policy", "catalog")
#: The policy reload's one ``BatchGetItem``, on the policy table only.
API_POLICY_BATCH_SID = "PolicyBatchRead"
API_POLICY_BATCH_ACTIONS = ("dynamodb:BatchGetItem",)
API_COGNITO_ACTIONS = (
    "cognito-idp:ListUsers", "cognito-idp:ListUsersInGroup", "cognito-idp:AdminGetUser",
    "cognito-idp:AdminListGroupsForUser", "cognito-idp:AdminCreateUser",
    "cognito-idp:AdminAddUserToGroup", "cognito-idp:AdminRemoveUserFromGroup",
    "cognito-idp:AdminDisableUser", "cognito-idp:AdminEnableUser", "cognito-idp:AdminDeleteUser",
    "cognito-idp:AdminUserGlobalSignOut",
)


def _route_regex(template: str) -> str:
    return re.sub(r"\{[A-Za-z_]+\}", lambda _: PATH_SEGMENT, template)


def api_routes() -> list[tuple[str, list[str]]]:
    """``(path template under /api, methods)`` for every admin API route, in the handler's order."""
    out: dict[str, list[str]] = {}
    for method, template in admin_api_routes():
        out.setdefault(f"{API_PREFIX}{template}", []).append(method)
    return list(out.items())


#: (path regex, methods) the admin API's regional web ACL lets through, as API Gateway sees the
#: path (the stage first). Everything else is blocked.
API_ROUTES: list[tuple[str, list[str]]] = [
    (rf"^(/{API_STAGE})?{_route_regex(path)}$", methods) for path, methods in api_routes()]
#: The site's static files: ``/`` (the OAuth redirect lands there with its query string) and
#: files with a static extension, so an unknown ``/api/...`` path never passes as a "file".
SITE_FILES = r"^/(([A-Za-z0-9_-]+/)*[A-Za-z0-9_.-]+\.(html|js|css|svg|png|ico|txt|woff2))?$"
#: The CloudFront web ACL's allow-list: the site's static files (GET/HEAD) and the API routes as
#: the viewer sends them (no stage).
SITE_ROUTES: list[tuple[str, list[str]]] = [
    (SITE_FILES, ["GET", "HEAD"]),
    *[(rf"^{_route_regex(path)}$", methods) for path, methods in api_routes()]]
#: The prompts write as each web ACL sees it: the CloudFront one without the stage, the API one
#: with it (``large_body_paths``).
PROMPT_WRITE_PATH = f"{API_PREFIX}/prompts/{{key}}"
SITE_LARGE_BODY_PATHS = (rf"^{_route_regex(PROMPT_WRITE_PATH)}$",)
API_LARGE_BODY_PATHS = (rf"^(/{API_STAGE})?{_route_regex(PROMPT_WRITE_PATH)}$",)


def login_domain_prefix(cfg: InfraConfig) -> str:
    """The Cognito prefix domain: unique per account and stage. Cognito rejects prefixes that
    contain ``aws``, ``amazon`` or ``cognito``, so none of those words is used."""
    return f"owlbank-{cfg.stage}-admin-{cfg.account}"


def content_security_policy(*, login_origin: str) -> str:
    """The console's CSP: only its own files and its own origin (the API is
    same-origin under ``/api``), plus the managed login origin for the token exchange and the
    sign-in form; no framing, no plugins, no inline script or style."""
    return "; ".join([
        "default-src 'self'",
        f"connect-src 'self' {login_origin}",
        f"form-action 'self' {login_origin}",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "object-src 'none'",
        "img-src 'self' data:",
    ])


def config_js(*, region: str, user_pool_id: str, client_id: str, login_domain: str,
              redirect_uri: str, stage: str) -> str:
    """The site's ``config.js`` (an ES module), generated at deploy; the values may be tokens,
    which ``BucketDeployment`` fills in. ``stage`` drives the console's environment badge."""
    doc = {"region": region, "userPoolId": user_pool_id, "clientId": client_id,
           "loginDomain": login_domain, "redirectUri": redirect_uri, "apiBase": API_PREFIX,
           "stage": stage}
    return ("// Generated at deploy by OwlBankAdmin (infra/owlbank_infra/admin_stack.py).\n"
            f"export default Object.freeze({json.dumps(doc, indent=2)});\n")


class AdminStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, cfg: InfraConfig, key: kms.IKey,
                 tables: dict[str, ddb.ITable], **kw) -> None:
        super().__init__(scope, cid, **kw)
        removal = RemovalPolicy.RETAIN if cfg.is_prod else RemovalPolicy.DESTROY
        self.cfg = cfg

        # ---- Cognito user pool
        self.user_pool = cognito.UserPool(
            self, "UserPool",
            user_pool_name=f"{cfg.name_prefix}-admin",
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True),
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(required=True, mutable=True)),
            password_policy=cognito.PasswordPolicy(
                min_length=12, require_lowercase=True, require_uppercase=True,
                require_digits=True, require_symbols=True,
                temp_password_validity=Duration.days(3)),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            mfa=cognito.Mfa.REQUIRED if cfg.is_prod else cognito.Mfa.OFF,
            mfa_second_factor=(cognito.MfaSecondFactor(otp=True, sms=False) if cfg.is_prod else None),
            feature_plan=cognito.FeaturePlan.ESSENTIALS,
            user_invitation=cognito.UserInvitationConfig(
                email_subject="Your OwlBank admin console account",
                email_body=("You have been given access to the OwlBank admin console. Your username "
                            "is {username} and your temporary password is {####}. You must change "
                            "it when you first sign in; it expires in 3 days. The console's "
                            "address is the ConsoleUrl output of the OwlBankAdmin stack.")),
            deletion_protection=cfg.is_prod,
            removal_policy=removal,
        )
        ack(self.user_pool, "AwsSolutions-COG8",
            "The Plus feature plan (Cognito threat protection, a per-user charge) is not used: "
            "Essentials covers managed login, TOTP MFA and admin-created users, which is what the "
            "console needs. The pool has no self sign-up, a 12-character password policy, TOTP MFA "
            "at stage prod, and WAF in front of the console and its API; the production guide "
            "lists Plus as a production option.")
        if not cfg.is_prod:
            ack(self.user_pool, "AwsSolutions-COG2",
                "MFA is required at stage prod only; dev and test pools have admin-created "
                "users only, with a strong password policy.")
        self.groups = {
            name: cognito.CfnUserPoolGroup(self, f"Group{''.join(p.capitalize() for p in name.split('-'))}",
                                           user_pool_id=self.user_pool.user_pool_id,
                                           group_name=name, description=desc)
            for name, desc in GROUPS.items()
        }

        # The managed-login domain (a Cognito prefix domain). Its URL is a plain string, so the
        # distribution's CSP can name it without referencing the domain resource.
        prefix = login_domain_prefix(cfg)
        self.user_pool_domain = self.user_pool.add_domain(
            "LoginDomain",
            cognito_domain=cognito.CognitoDomainOptions(domain_prefix=prefix),
            managed_login_version=cognito.ManagedLoginVersion.NEWER_MANAGED_LOGIN)
        self.login_domain = f"{prefix}.auth.{self.region}.amazoncognito.com"
        login_origin = f"https://{self.login_domain}"

        # ---- Audit table
        self.audit_table = ddb.Table(
            self, "AuditTable",
            table_name=f"{cfg.name_prefix}-admin-audit",
            partition_key=ddb.Attribute(name="pk", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="sk", type=ddb.AttributeType.STRING),
            billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
            encryption=ddb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=key,
            point_in_time_recovery_specification=ddb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True),
            time_to_live_attribute=TTL_ATTRIBUTE,
            deletion_protection=cfg.is_prod,
            removal_policy=removal,
        )

        # ---- Origin verify: a generated secret, never in a template.
        self.origin_verify = sm.Secret(
            self, "OriginVerifySecret",
            secret_name=f"owlbank/{cfg.stage}/admin/origin-verify",
            description="Value of the X-Origin-Verify header that the admin console's CloudFront "
                        "/api/* origin sends and the admin API's web ACL requires. Generated; read "
                        "by CloudFormation through a dynamic reference.",
            encryption_key=key,
            generate_secret_string=sm.SecretStringGenerator(exclude_punctuation=True, password_length=48),
            removal_policy=RemovalPolicy.RETAIN if cfg.is_prod else RemovalPolicy.DESTROY,
        )
        ack(self.origin_verify, "AwsSolutions-SMG4",
            "A shared header value, not a service credential: it is rotated by writing a new "
            "version and redeploying with adminOriginVerifySecretVersion (infra/README.md), "
            "because the distribution's origin header and the web ACL rule must change together; "
            "a rotation Lambda can't do that.")
        # ``{{resolve:secretsmanager:<arn>:SecretString:::[<version>]}}``: CloudFormation reads the
        # value when it creates or updates the referencing resource.
        origin_verify_value = SecretValue.secrets_manager(
            self.origin_verify.secret_arn,
            version_id=cfg.admin_origin_verify_secret_version).unsafe_unwrap()

        # ---- API
        self.api_function = self._api_function(cfg, key, tables)
        self.api, self.authorizer = self._api(cfg, key, origin_verify_value)
        self._routes()

        # ---- Site
        self.site_bucket = s3.Bucket(
            self, "SiteBucket",
            # s3PublicAccessBlockMode=account: an organization SCP can deny
            # s3:PutBucketPublicAccessBlock, so the property is left out and S3's default for new
            # buckets plus the account setting protect it (checked by preflight --post-deploy).
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL if cfg.explicit_public_access_block else None,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=True,
            removal_policy=removal,
            auto_delete_objects=not cfg.is_prod,
        )
        if not cfg.is_prod:
            self._auto_delete_log_group(cfg, key)
        ack(self.site_bucket, "AwsSolutions-S1",
            "The bucket holds only the console's static files, read by CloudFront through origin "
            "access control. Server access logging needs another bucket; every viewer request is "
            "logged by the CloudFront web ACL, and CloudTrail data events cover the bucket if needed.")
        self.site_waf = web_acl(
            self, "SiteWebAcl", name=f"{cfg.name_prefix}-admin-site", metric="owlbank-admin-site",
            allowed=SITE_ROUTES, rate_limit=WAF_RATE_LIMIT, max_body_bytes=MAX_BODY_BYTES,
            scope_type="CLOUDFRONT", managed_mode=cfg.waf_managed_mode,
            large_body_paths=SITE_LARGE_BODY_PATHS, large_body_max_bytes=PROMPTS_MAX_BODY_BYTES)
        waf_logging(self, "SiteWafLogging", cfg=cfg, key=key, acl=self.site_waf, which="admin-site",
                    redact_query_string=True)  # OAuth redirects carry the code in the query string
        headers = cf.ResponseHeadersPolicy(
            self, "SecurityHeaders",
            comment="OwlBank admin console: CSP, HSTS, no framing, nosniff",
            security_headers_behavior=cf.ResponseSecurityHeadersBehavior(
                content_security_policy=cf.ResponseHeadersContentSecurityPolicy(
                    content_security_policy=content_security_policy(login_origin=login_origin),
                    override=True),
                strict_transport_security=cf.ResponseHeadersStrictTransportSecurity(
                    access_control_max_age=Duration.seconds(HSTS_SECONDS),
                    include_subdomains=True, override=True),
                content_type_options=cf.ResponseHeadersContentTypeOptions(override=True),
                frame_options=cf.ResponseHeadersFrameOptions(
                    frame_option=cf.HeadersFrameOption.DENY, override=True),
                referrer_policy=cf.ResponseHeadersReferrerPolicy(
                    referrer_policy=cf.HeadersReferrerPolicy.STRICT_ORIGIN, override=True),
            ))
        self.distribution = cf.Distribution(
            self, "Distribution",
            comment=f"OwlBank admin console ({cfg.stage})",
            default_root_object="index.html",
            price_class=cf.PriceClass.PRICE_CLASS_100,
            http_version=cf.HttpVersion.HTTP2_AND_3,
            # No minimum_protocol_version: CloudFront ignores it on the default *.cloudfront.net
            # certificate (see the CFR4 acknowledgement below).
            web_acl_id=self.site_waf.attr_arn,
            default_behavior=cf.BehaviorOptions(
                origin=cf_origins.S3BucketOrigin.with_origin_access_control(self.site_bucket),
                viewer_protocol_policy=cf.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cf.AllowedMethods.ALLOW_GET_HEAD,
                cache_policy=cf.CachePolicy.CACHING_OPTIMIZED,
                response_headers_policy=headers,
                compress=True),
            additional_behaviors={API_BEHAVIOR: self._api_behavior(cfg, headers, origin_verify_value)},
        )
        ack(self.distribution, "AwsSolutions-CFR1",
            "No geo restriction: operators may sign in from anywhere. The console is behind "
            "Cognito sign-in, and the CloudFront web ACL (managed rules, rate rule, path "
            "allow-list) fronts it.")
        ack(self.distribution, "AwsSolutions-CFR3",
            "CloudFront standard logging needs an S3 bucket with ACLs enabled, which the project "
            "rules out (BucketOwnerEnforced, no ACLs). Every request is logged by the CloudFront "
            "web ACL to a CMK-encrypted log group, with authorization, cookie and query string "
            "redacted.")
        ack(self.distribution, "AwsSolutions-CFR4",
            "The default *.cloudfront.net certificate fixes the minimum TLS version (TLSv1) "
            "whatever the distribution asks for, and a stricter policy needs a custom domain and "
            "an ACM certificate. Browsers negotiate TLS 1.2 or later; the production guide "
            "recommends a custom domain for the console.")
        self.headers_policy = headers
        self.console_url = f"https://{self.distribution.distribution_domain_name}/"
        self.console_origin = self.console_url.rstrip("/")

        # ---- App client: public, authorization code + PKCE, callback = the console URL.
        self.user_pool_client = self.user_pool.add_client(
            "ConsoleClient",
            user_pool_client_name=f"{cfg.name_prefix}-admin-console",
            generate_secret=False,
            auth_flows=cognito.AuthFlow(),  # see ExplicitAuthFlows below
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True, implicit_code_grant=False,
                                         client_credentials=False),
                scopes=[cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL, cognito.OAuthScope.PROFILE],
                callback_urls=[self.console_url],
                logout_urls=[self.console_url]),
            supported_identity_providers=[cognito.UserPoolClientIdentityProvider.COGNITO],
            access_token_validity=Duration.minutes(ACCESS_ID_TOKEN_MINUTES),
            id_token_validity=Duration.minutes(ACCESS_ID_TOKEN_MINUTES),
            refresh_token_validity=Duration.hours(REFRESH_TOKEN_HOURS),
            prevent_user_existence_errors=True,
            enable_token_revocation=True,
        )
        # Without ExplicitAuthFlows Cognito would allow SRP and custom auth from any holder of the
        # (public) client ID. Managed login doesn't use these API flows, so only refresh is left.
        # [unverified at deploy] sign-in through managed login with refresh-only explicit flows.
        cfn_client = self.user_pool_client.node.default_child
        cfn_client.add_property_override("ExplicitAuthFlows", ["ALLOW_REFRESH_TOKEN_AUTH"])
        # The newer managed login shows a page only for a client with a branding style; Cognito's
        # own default style is enough.
        cognito.CfnManagedLoginBranding(
            self, "LoginBranding", user_pool_id=self.user_pool.user_pool_id,
            client_id=self.user_pool_client.user_pool_client_id, use_cognito_provided_values=True
        ).node.add_dependency(self.user_pool_domain)

        # ---- First user: only with -c adminEmail=...
        self.first_user: cognito.CfnUserPoolUser | None = None
        if cfg.admin_email:
            user = cognito.CfnUserPoolUser(
                self, "FirstAdminUser", user_pool_id=self.user_pool.user_pool_id,
                username=cfg.admin_email,
                user_attributes=[
                    cognito.CfnUserPoolUser.AttributeTypeProperty(name="email", value=cfg.admin_email),
                    cognito.CfnUserPoolUser.AttributeTypeProperty(name="email_verified", value="true")],
                desired_delivery_mediums=["EMAIL"])
            attach = cognito.CfnUserPoolUserToGroupAttachment(
                self, "FirstAdminUserGroup", user_pool_id=self.user_pool.user_pool_id,
                username=cfg.admin_email, group_name=FIRST_USER_GROUP)
            # Retained: a later deploy without adminEmail must not delete the first admin
            # (removing the resource from the template would). Removing the user is done in
            # the console or with Cognito directly; the pool's own removal policy still applies.
            user.apply_removal_policy(RemovalPolicy.RETAIN)
            attach.apply_removal_policy(RemovalPolicy.RETAIN)
            attach.add_dependency(user)
            attach.add_dependency(self.groups[FIRST_USER_GROUP])
            self.first_user = user

        # ---- The site's files and its generated config.js
        self.site_deployment = self._site_deployment(cfg, key)

        CfnOutput(self, "ConsoleUrl", value=self.console_url, description="The admin console")
        CfnOutput(self, "UserPoolId", value=self.user_pool.user_pool_id)
        CfnOutput(self, "UserPoolClientId", value=self.user_pool_client.user_pool_client_id)
        CfnOutput(self, "ManagedLoginDomain", value=self.login_domain,
                  description="Cognito managed login (prefix domain)")
        CfnOutput(self, "InternalApiEndpoint", value=f"{self.console_origin}{API_PREFIX}",
                  description="The console's API, same origin (reachable only through CloudFront)")
        CfnOutput(self, "AuditTableName", value=self.audit_table.table_name)
        CfnOutput(self, "SiteBucketName", value=self.site_bucket.bucket_name)

    # ------------------------------------------------------------------ the API Lambda

    def _api_function(self, cfg: InfraConfig, key: kms.IKey,
                      tables: dict[str, ddb.ITable]) -> lambda_.Function:
        """``lambdas.admin_api.handler.handler``, packaged like the Compute stack's functions (the
        same staged ``src`` bundle; no third-party layer: it needs only the runtime's boto3)."""
        if not LAMBDA_SRC_DIR.exists():  # Compute stages it first in the app; not when built alone
            stage_lambda_source()
        logs = log_group(self, "ApiFnLogs", cfg=cfg, key=key, name="/owlbank/lambda/admin_api")
        role = iam.Role(self, "ApiFnRole", assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
                        description="OwlBank admin console API Lambda (least privilege)")
        role.add_to_policy(iam.PolicyStatement(
            sid="WriteOwnLogs", actions=["logs:CreateLogStream", "logs:PutLogEvents"],
            resources=[logs.log_group_arn]))
        table_arns = [tables[n].table_arn for n in API_TABLES] + [self.audit_table.table_arn]
        role.add_to_policy(iam.PolicyStatement(
            sid="ConfigAndAuditTables", actions=list(API_TABLE_ACTIONS), resources=table_arns))
        # The whole-policy read is one BatchGetItem of the policy table,
        # and the catalog read one Scan of the catalog table; each on that one table only.
        role.add_to_policy(iam.PolicyStatement(
            sid=API_POLICY_BATCH_SID, actions=list(API_POLICY_BATCH_ACTIONS),
            resources=[tables["policy"].table_arn]))
        role.add_to_policy(catalog_scan_statement(tables[CATALOG_TABLE].table_arn))
        role.add_to_policy(iam.PolicyStatement(
            sid="TablesKmsViaDynamoDB",
            actions=["kms:Decrypt", "kms:DescribeKey", "kms:Encrypt", "kms:GenerateDataKey"],
            resources=[key.key_arn],
            conditions={"StringEquals": {"kms:ViaService": f"dynamodb.{self.region}.amazonaws.com"}}))
        role.add_to_policy(iam.PolicyStatement(
            sid="ConsoleUserPool", actions=list(API_COGNITO_ACTIONS),
            resources=[self.user_pool.user_pool_arn]))
        fn = lambda_.Function(
            self, "ApiFn",
            runtime=lambda_.Runtime.PYTHON_3_12,  # Python 3.12 everywhere
            architecture=lambda_.Architecture.ARM_64,
            handler="lambdas.admin_api.handler.handler",
            code=lambda_.Code.from_asset(str(LAMBDA_SRC_DIR)),
            role=role,
            log_group=logs,
            timeout=Duration.seconds(API_FN_TIMEOUT_S),
            memory_size=API_FN_MEMORY_MB,
            environment={
                "OWLBANK_ADMIN_USER_POOL_ID": self.user_pool.user_pool_id,
                "OWLBANK_TABLE_POLICY": tables["policy"].table_name,
                "OWLBANK_TABLE_CATALOG": tables["catalog"].table_name,
                "OWLBANK_TABLE_ADMIN_AUDIT": self.audit_table.table_name,
                "OWLBANK_STAGE": cfg.stage,
                "OWLBANK_LOG_LEVEL": cfg.python_log_level,
                # Shown read-only by GET /slots; a change is audited at cold start.
                "OWLBANK_ACCEPT_SPOKEN_FACTORS": "true" if cfg.accept_spoken_factors else "false",
                # The deploy-time slot-mode ceiling, shown read-only by the console.
                "OWLBANK_SLOT_MODE_CEILING": cfg.slot_mode_ceiling,
            },
            tracing=lambda_.Tracing.ACTIVE,
            logging_format=lambda_.LoggingFormat.JSON,
            application_log_level_v2=getattr(lambda_.ApplicationLogLevel, cfg.lambda_application_log_level),
            system_log_level_v2=getattr(lambda_.SystemLogLevel, cfg.lambda_system_log_level),
        )
        ack(role, "AwsSolutions-IAM5[Resource::*]",
            "X-Ray tracing: xray:PutTraceSegments and xray:PutTelemetryRecords don't support "
            "resource-level permissions, so their resource must be '*'. Nothing else uses '*'.")
        ack(role, f"AwsSolutions-IAM5[Resource::<{self.get_logical_id(logs.node.default_child)}.Arn>:*]",
            "Log stream names are created at runtime, so the log-stream ARN under this "
            "function's own log group needs a trailing wildcard.")
        ack(fn, "AwsSolutions-L1",
            "Python 3.12 is pinned on purpose (one Python version for the Lambdas, "
            "the bridge and the tests). Move all components together when upgrading.")
        return fn

    def _auto_delete_log_group(self, cfg: InfraConfig, key: kms.IKey) -> None:
        """``auto_delete_objects`` adds CDK's ``Custom::S3AutoDeleteObjects`` provider, a Lambda with
        no logging settings, so Lambda would create its log group on first run with no CMK and no
        expiry. Point it at a group of ours instead, made first, with the CMK and the stack's
        retention. Its role's ``AWSLambdaBasicExecutionRole`` already covers writing to it."""
        provider = Stack.of(self).node.try_find_child(AUTO_DELETE_PROVIDER_ID)
        handler = provider.node.try_find_child("Handler") if provider is not None else None
        if not isinstance(handler, CfnResource):
            raise RuntimeError(f"no {AUTO_DELETE_PROVIDER_ID} handler to give a log group")
        logs = log_group(self, "SiteAutoDeleteLogs", cfg=cfg, key=key,
                         name="/owlbank/admin/site-auto-delete")
        handler.add_property_override("LoggingConfig", {"LogGroup": logs.log_group_name})
        handler.add_dependency(cast(CfnResource, logs.node.default_child))

    # ------------------------------------------------------------------ the REST API

    def _api(self, cfg: InfraConfig, key: kms.IKey, origin_verify_value: str
             ) -> tuple[apigw.RestApi, apigw.CognitoUserPoolsAuthorizer]:
        access_logs = log_group(self, "ApiAccessLogs", cfg=cfg, key=key, name="/owlbank/admin/apigw-access-logs")
        api = apigw.RestApi(
            self, "Api",
            rest_api_name=f"{cfg.name_prefix}-admin",
            description="OwlBank admin console API (Cognito authorizer on every method; reachable "
                        "only through the console's CloudFront distribution)",
            endpoint_types=[apigw.EndpointType.REGIONAL],
            # The account-level CloudWatch role is set by OwlBankApi (app_builder orders the
            # stacks); a second AWS::ApiGateway::Account here would fight over that singleton.
            cloud_watch_role=False,
            deploy_options=apigw.StageOptions(
                stage_name=API_STAGE,
                access_log_destination=apigw.LogGroupLogDestination(access_logs),
                access_log_format=apigw.AccessLogFormat.json_with_standard_fields(
                    caller=False, http_method=True, ip=True, protocol=True, request_time=True,
                    resource_path=True, response_length=True, status=True, user=True),
                logging_level=apigw.MethodLoggingLevel.ERROR,
                data_trace_enabled=False,  # never log request bodies or tokens
                tracing_enabled=True,
                throttling_rate_limit=API_RATE_LIMIT,
                throttling_burst_limit=API_BURST_LIMIT,
            ),
        )
        authorizer = apigw.CognitoUserPoolsAuthorizer(
            self, "Authorizer", cognito_user_pools=[self.user_pool],
            identity_source=apigw.IdentitySource.header("Authorization"))
        self._validator = api.add_request_validator(
            "Validator", validate_request_body=True, validate_request_parameters=True)

        self.api_waf = web_acl(
            self, "ApiWebAcl", name=f"{cfg.name_prefix}-admin-api", metric="owlbank-admin-api",
            allowed=API_ROUTES, rate_limit=WAF_RATE_LIMIT, max_body_bytes=MAX_BODY_BYTES,
            managed_mode=cfg.waf_managed_mode,
            # API Gateway's default 16 KB body inspection would cut the prompts cap short.
            large_body_paths=API_LARGE_BODY_PATHS, large_body_max_bytes=PROMPTS_MAX_BODY_BYTES,
            inspection_resource="API_GATEWAY",
            required_header=(ORIGIN_VERIFY_HEADER, origin_verify_value))
        waf_logging(self, "ApiWafLogging", cfg=cfg, key=key, acl=self.api_waf, which="admin-api",
                    redact_query_string=False)
        waf.CfnWebACLAssociation(self, "ApiWebAclAssociation",
                                 resource_arn=api.deployment_stage.stage_arn,
                                 web_acl_arn=self.api_waf.attr_arn)
        return api, authorizer

    def _routes(self) -> None:
        """Every route under ``/api``, each with the Cognito authorizer and a Lambda
        proxy integration. No OPTIONS method and no CORS: the console calls the API from its own
        origin through CloudFront. A request without a valid token gets API Gateway's 401."""
        integration = apigw.LambdaIntegration(self.api_function, proxy=True, allow_test_invoke=False)
        for path, methods in api_routes():
            resource = self.api.root
            for part in path.strip("/").split("/"):
                resource = resource.get_resource(part) or resource.add_resource(part)
            params = {f"method.request.path.{p}": True for p in re.findall(r"\{([A-Za-z_]+)\}", path)}
            for method in methods:
                resource.add_method(
                    method, integration,
                    authorizer=self.authorizer, authorization_type=apigw.AuthorizationType.COGNITO,
                    request_validator=self._validator, request_parameters=params or None)

    # ------------------------------------------------------------------ CloudFront /api/*

    def _api_behavior(self, cfg: InfraConfig, headers: cf.IResponseHeadersPolicy,
                      origin_verify_value: str) -> cf.BehaviorOptions:
        """``/api/*`` to the API's regional endpoint, origin path = the stage.

        CloudFront refuses ``Authorization`` in an origin request policy; a header is forwarded
        when it is in the cache policy's key. So the cache policy keys on ``Authorization`` and
        the query string (a cached entry could only ever be served to the same token) with a
        1-second maximum TTL, the smallest that allows a header in the key, and the API sends
        ``Cache-Control: no-store`` on every response, so nothing is stored. The origin request
        policy adds ``Content-Type`` and the query string; no cookies anywhere."""
        domain = f"{self.api.rest_api_id}.execute-api.{self.region}.{self.url_suffix}"
        origin = cf_origins.HttpOrigin(
            domain, origin_path=f"/{API_STAGE}",
            protocol_policy=cf.OriginProtocolPolicy.HTTPS_ONLY,
            origin_ssl_protocols=[cf.OriginSslPolicy.TLS_V1_2],
            custom_headers={ORIGIN_VERIFY_HEADER: origin_verify_value})
        cache = cf.CachePolicy(
            self, "ApiCachePolicy",
            cache_policy_name=f"{cfg.name_prefix}-admin-api",
            comment="Admin API: no caching (API sends no-store); forwards Authorization and the query string",
            default_ttl=Duration.seconds(0), min_ttl=Duration.seconds(0), max_ttl=Duration.seconds(1),
            header_behavior=cf.CacheHeaderBehavior.allow_list("Authorization"),
            query_string_behavior=cf.CacheQueryStringBehavior.all(),
            cookie_behavior=cf.CacheCookieBehavior.none(),
            enable_accept_encoding_gzip=False, enable_accept_encoding_brotli=False)
        forward = cf.OriginRequestPolicy(
            self, "ApiOriginRequestPolicy",
            origin_request_policy_name=f"{cfg.name_prefix}-admin-api",
            comment="Admin API: Content-Type and the query string; never cookies",
            header_behavior=cf.OriginRequestHeaderBehavior.allow_list("Content-Type"),
            query_string_behavior=cf.OriginRequestQueryStringBehavior.all(),
            cookie_behavior=cf.OriginRequestCookieBehavior.none())
        self.api_cache_policy, self.api_origin_request_policy = cache, forward
        return cf.BehaviorOptions(
            origin=origin,
            viewer_protocol_policy=cf.ViewerProtocolPolicy.HTTPS_ONLY,
            allowed_methods=cf.AllowedMethods.ALLOW_ALL,
            cached_methods=cf.CachedMethods.CACHE_GET_HEAD,
            cache_policy=cache,
            origin_request_policy=forward,
            response_headers_policy=headers,
            compress=False)

    # ------------------------------------------------------------------ the site's files

    def _site_deployment(self, cfg: InfraConfig, key: kms.IKey) -> s3deploy.BucketDeployment:
        """``src/admin_console`` (without its tests) plus a generated ``config.js``; the
        distribution is invalidated so a deploy shows at once."""
        config = config_js(region=self.region, user_pool_id=self.user_pool.user_pool_id,
                           client_id=self.user_pool_client.user_pool_client_id,
                           login_domain=self.login_domain, redirect_uri=self.console_url,
                           stage=cfg.stage)
        logs = log_group(self, "SiteDeploymentLogs", cfg=cfg, key=key,
                         name="/owlbank/admin/site-deployment")
        deployment = s3deploy.BucketDeployment(
            self, "SiteDeployment",
            sources=[s3deploy.Source.asset(str(SITE_DIR), exclude=SITE_EXCLUDE),
                     s3deploy.Source.data("config.js", config)],
            destination_bucket=self.site_bucket,
            distribution=self.distribution, distribution_paths=["/*"],
            # Excluded keys are neither uploaded nor pruned, so the font deployment's files stay.
            exclude=[FONT_GLOB],
            prune=True, memory_limit=256, log_group=logs)
        # The font, with an explicit type (the handler's MIME table may not know .woff2). Its
        # prune is limited to ``fonts/`` and skips ``fonts/OFL.txt``, the main deployment's.
        # Same memory and log group, so it reuses the one singleton handler (and its acks).
        s3deploy.BucketDeployment(
            self, "FontDeployment",
            sources=[s3deploy.Source.asset(str(SITE_DIR / "fonts"), exclude=["*.txt", "*.md"])],
            destination_bucket=self.site_bucket, destination_key_prefix="fonts/",
            content_type="font/woff2",
            cache_control=[s3deploy.CacheControl.from_string(FONT_CACHE_CONTROL)],
            exclude=["*.txt"],
            distribution=self.distribution, distribution_paths=["/fonts/*"],
            prune=True, memory_limit=256, log_group=logs)
        # CDK's deployment function (a singleton custom resource handler) and its role.
        handler = next(c for c in self.node.children
                       if c.node.id.startswith("Custom::CDKBucketDeployment"))
        ack(handler, "AwsSolutions-IAM4[Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/"
                     "AWSLambdaBasicExecutionRole]",
            "CDK's BucketDeployment handler role (aws-cdk-lib's own construct) uses "
            "AWSLambdaBasicExecutionRole for its logs; the function runs only at deploy time, "
            "copying the console's static files into the site bucket.")
        s3_reason = ("CDK's BucketDeployment handler copies the CDK asset (the console's files) into "
                     "the site bucket: aws-cdk-lib grants it read on the asset objects and "
                     "read/write/delete on the site bucket's objects with these action groups. "
                     "It runs only at deploy time and touches only those two buckets.")
        for finding in ("Action::s3:GetObject*", "Action::s3:GetBucket*", "Action::s3:List*",
                        "Action::s3:DeleteObject*", "Action::s3:Abort*",
                        f"Resource::arn:<AWS::Partition>:s3:::cdk-hnb659fds-assets-{cfg.account}-{cfg.region}/*",
                        f"Resource::<{self.get_logical_id(self.site_bucket.node.default_child)}.Arn>/*"):
            ack(handler, f"AwsSolutions-IAM5[{finding}]", s3_reason)
        # aws-cdk-lib grants cloudfront:CreateInvalidation/GetInvalidation on '*'; scope that
        # statement to this distribution's ARN (the deployment already references the
        # distribution, so no new dependency or cycle).
        policy = handler.node.find_child("ServiceRole").node.find_child("DefaultPolicy")
        cfn_policy = policy.node.default_child
        dist_arn = Stack.of(self).format_arn(
            service="cloudfront", region="", resource="distribution",
            resource_name=self.distribution.distribution_id)
        statements = policy.document.to_json()["Statement"]
        for st in statements:
            acts = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
            if "cloudfront:CreateInvalidation" in acts:
                st["Resource"] = dist_arn
        cfn_policy.policy_document = iam.PolicyDocument.from_json({"Version": "2012-10-17", "Statement": statements})
        ack(handler, "AwsSolutions-L1",
            "The BucketDeployment handler is aws-cdk-lib's own function; its runtime is chosen by "
            "the CDK release and moves with CDK upgrades. It runs only at deploy time.")
        return deployment
