"""Api: the REST API for Twilio's webhooks (front door) and the Flex Context API, behind WAF.

* ``POST /voice/{incoming,language,factor,connect-action,status,partial,enqueue-action}`` ->
  ``frontdoor`` (``enqueue-action``: the ``<Enqueue>`` ``action``; the ``<Enqueue>``
  has no ``waitUrl``).
  Twilio can't sign SigV4 and a REST authorizer can't see the signed body, so the
  handler validates ``X-Twilio-Signature`` and HTTP Basic credentials first.
* ``agentcore`` hosting only: ``POST /voice/orchestrator`` ->
  ``frontdoor``, the Conversation Configuration's status callback (the sink: Basic, then
  ``X-Twilio-Signature``, then 200). Same stage throttling, logging and method settings as the voice
  routes. The WAF sends its Basic challenge (API Gateway REST renames a Lambda's
  ``WWW-Authenticate`` to ``x-amzn-Remapped-WWW-Authenticate``), and the path alone gets a 64 KB body
  cap (Orchestrator events carry transcripts), with API Gateway's WAF body inspection raised to 64 KB.
  The other paths keep 8 KB. Nothing changes in ``fargate``.
* ``POST /voice/taskrouter`` (``callObservability=on`` or ``humanLegConversation=on``) and
  ``POST /voice/transcription`` (``humanLegConversation=on``) -> ``frontdoor``: both also need
  Flex mode (``workspaceSid`` and ``workflowSid`` in ``twilioOutputs``). Twilio webhooks
  like the voice routes (signature and Basic checked in the handler), same WAF and throttling.
* ``GET /context/{journal_ref}`` -> ``context_api``, with the ``flex_authorizer``
  REQUEST authorizer on ``X-Flex-Token``. ``OPTIONS`` reaches the Lambda without the
  authorizer, which answers CORS for the Flex origin only.
* With ``domainName`` set, the API is served on ``voice.<domain>`` and the default
  ``execute-api`` endpoint is turned off.
* Without a domain (the CloudFront mode) the default ``execute-api`` endpoint
  stays on: it is the only HTTPS name. Compensating controls: the regional WAF, stage
  throttling, HTTP Basic and the Twilio signature check in the handler. Twilio's webhook
  URLs then carry the stage: ``https://<id>.execute-api.<region>.amazonaws.com/v1/voice/...``.
"""

from __future__ import annotations

from typing import Any

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_certificatemanager as acm
from aws_cdk import aws_kms as kms
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sns as sns
from aws_cdk import aws_wafv2 as waf
from constructs import Construct

from .config import InfraConfig
from .nag import ack
from .observability import Alarms, log_group, waf_logging
from .waf import web_acl

#: ``partial``: Gather ``partialResultCallback``, answered 204. ``enqueue-action``:
#: the ``<Enqueue>`` ``action``, on every escalation whatever the flags.
VOICE_ROUTES = ("incoming", "language", "factor", "connect-action", "status", "partial",
                "enqueue-action")
#: TaskRouter workspace events: the queue-wait span and the human leg's
#: participants. Present when either switch is on.
TASKROUTER_ROUTE = "taskrouter"
#: The human-leg ``<Transcription>`` status callback: signed like
#: ``/voice/status`` and checked in the front door. Only with ``humanLegConversation=on`` and Flex
#: mode (``workspaceSid`` and ``workflowSid`` in ``twilioOutputs``).
TRANSCRIPTION_ROUTE = "transcription"


def voice_routes(cfg: InfraConfig) -> tuple[str, ...]:
    """The front door's Twilio webhook routes: ``taskrouter`` with ``callObservability=on`` or
    ``humanLegConversation=on``; ``transcription`` with ``humanLegConversation=on`` only. Both also
    need Flex mode (``workspaceSid`` and ``workflowSid`` in ``twilioOutputs``)."""
    return (VOICE_ROUTES
            + ((TASKROUTER_ROUTE,) if cfg.taskrouter_route_on else ())
            + ((TRANSCRIPTION_ROUTE,) if cfg.human_leg_conversation_on else ()))
#: ``agentcore`` only: the Orchestrator status-callback sink, under ``/voice/``.
ORCHESTRATOR_ROUTE = "orchestrator"
#: Its body cap (Orchestrator events carry transcripts; WAF inspects at most 64 KB).
ORCHESTRATOR_MAX_BODY_BYTES = 65536
STAGE = "v1"


#: Why the Twilio webhook methods have no API Gateway authorizer (one wording for every such route).
TWILIO_APIG4_REASON = (
    "Twilio webhooks can't use IAM or a REST authorizer (no SigV4, and an authorizer "
    "can't see the signed POST body). The handler checks X-Twilio-Signature and HTTP "
    "Basic credentials before anything else; WAF and throttling sit in front.")
TWILIO_COG4_REASON = (
    "Twilio is not a Cognito user; the front door authenticates with X-Twilio-Signature "
    "and HTTP Basic credentials in the handler.")
#: The API's 8 KB body cap (API Gateway's default 16 KB WAF inspection limit covers it).
MAX_BODY_BYTES = 8192


def voice_stage_options(access_logs: logs.ILogGroup) -> apigw.StageOptions:
    """The ``v1`` stage of a Twilio-facing REST API: JSON access logs without caller or user,
    ERROR execution logs, never request bodies, X-Ray, and stage throttling (50 rps, burst 100)."""
    return apigw.StageOptions(
        stage_name=STAGE,
        access_log_destination=apigw.LogGroupLogDestination(access_logs),
        access_log_format=apigw.AccessLogFormat.json_with_standard_fields(
            caller=False, http_method=True, ip=True, protocol=True, request_time=True,
            resource_path=True, response_length=True, status=True, user=False),
        logging_level=apigw.MethodLoggingLevel.ERROR,
        data_trace_enabled=False,  # never log request bodies (factor values)
        tracing_enabled=True,
        throttling_rate_limit=50,
        throttling_burst_limit=100,
    )


def twilio_webhook(resource: apigw.IResource, integration: apigw.Integration,
                   validator: apigw.IRequestValidator) -> apigw.Method:
    """``POST`` on ``resource`` for a Twilio webhook: no API Gateway authorizer (the handler checks
    the Twilio signature and HTTP Basic credentials first), the request validator on."""
    m = resource.add_method("POST", integration, request_validator=validator,
                            authorization_type=apigw.AuthorizationType.NONE)
    ack(m, "AwsSolutions-APIG4", TWILIO_APIG4_REASON)
    ack(m, "AwsSolutions-COG4", TWILIO_COG4_REASON)
    return m


def api_waf(stack: Stack, *, cfg: InfraConfig, key: kms.IKey, api: apigw.RestApi, name: str,
            metric: str, which: str, allowed: list[tuple[str, list[str]]],
            authorization_required_path: str, **web_acl_options: Any) -> waf.CfnWebACL:
    """The regional web ACL of a Twilio-facing REST API (``WebAcl``), its logging (``WafLogging``)
    and its association with the stage (``WebAclAssociation``): the allow-list, the per-IP rate
    rule (``wafApiRateLimit``), the 8 KB body cap, the managed rule groups (``wafManagedRuleMode``)
    and the Basic challenge on ``authorization_required_path``. ``web_acl_options`` go to
    :func:`waf.web_acl` (a larger cap on named paths, for example)."""
    acl = web_acl(
        stack, "WebAcl", name=name, metric=metric, allowed=allowed,
        rate_limit=cfg.waf_api_rate_limit, max_body_bytes=MAX_BODY_BYTES,
        managed_mode=cfg.waf_managed_mode,
        authorization_required_path=authorization_required_path, **web_acl_options)
    # No query strings carry credentials here (the webhooks are POST bodies; the only query
    # values are the opaque Gather ids ``g`` and ``early``).
    waf_logging(stack, "WafLogging", cfg=cfg, key=key, acl=acl, which=which, redact_query_string=False)
    waf.CfnWebACLAssociation(stack, "WebAclAssociation",
                             resource_arn=api.deployment_stage.stage_arn,
                             web_acl_arn=acl.attr_arn)
    return acl


def api_alarms(alarms: Alarms, api: apigw.RestApi, acl_name: str, region: str) -> None:
    """5XX responses and a burst of WAF blocks."""
    alarms.alarm("Api5xxAlarm", api.metric_server_error(period=Duration.minutes(5), statistic="Sum"),
                 threshold=5, description="API Gateway 5XX responses")
    alarms.waf_blocked("ApiWafBlockedAlarm", acl_name, region)


class ApiStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, cfg: InfraConfig, key: kms.IKey,
                 functions: dict[str, lambda_.IFunction], alarm_topic: sns.ITopic, **kw) -> None:
        super().__init__(scope, cid, **kw)

        # Import by ARN so the invoke permissions live in this stack (no Compute<->Api cycle).
        # ``functions`` is Compute's ``invoke_targets``: for the front door that is the ``live``
        # alias, so the integration URI and the invoke permission name the alias ARN.
        def imported(name: str) -> lambda_.IFunction:
            return lambda_.Function.from_function_attributes(
                self, f"{name}Ref", function_arn=functions[name].function_arn, same_environment=True)

        frontdoor, context_api, authorizer_fn = (imported(n) for n in
                                                 ("frontdoor", "context_api", "flex_authorizer"))

        access_logs = log_group(self, "AccessLogs", cfg=cfg, key=key, name="/owlbank/apigw/access-logs")
        self.api = apigw.RestApi(
            self, "Api",
            rest_api_name="owlbank-voice",
            description="OwlBank Twilio webhooks and Flex Context API",
            endpoint_types=[apigw.EndpointType.REGIONAL],
            disable_execute_api_endpoint=cfg.custom_domain_mode,  # on in the CloudFront mode
            cloud_watch_role=True,
            cloud_watch_role_removal_policy=RemovalPolicy.DESTROY,
            deploy_options=voice_stage_options(access_logs),
        )
        validator = self.api.add_request_validator(
            "Validator", validate_request_body=True, validate_request_parameters=True)

        voice = self.api.root.add_resource("voice")
        fd_integration = apigw.LambdaIntegration(frontdoor, proxy=True)
        for route in voice_routes(cfg):
            twilio_webhook(voice.add_resource(route), fd_integration, validator)
        if cfg.is_agentcore:
            twilio_webhook(voice.add_resource(ORCHESTRATOR_ROUTE), fd_integration, validator)

        flex_auth = apigw.RequestAuthorizer(
            self, "FlexAuthorizer",
            handler=authorizer_fn,
            identity_sources=[apigw.IdentitySource.header("X-Flex-Token")],
            # No cache, so a token is never trusted past its
            # expiry or sign-out; the panel makes one request per accepted task.
            results_cache_ttl=Duration.seconds(0),
        )
        ctx = self.api.root.add_resource("context").add_resource("{journal_ref}")
        ctx_integration = apigw.LambdaIntegration(context_api, proxy=True)
        get = ctx.add_method(
            "GET", ctx_integration, authorizer=flex_auth,
            authorization_type=apigw.AuthorizationType.CUSTOM,
            request_validator=validator,
            request_parameters={"method.request.path.journal_ref": True,
                                "method.request.querystring.call_sid": True})
        ack(get, "AwsSolutions-COG4",
            "Callers are Flex agents with Flex (Twilio) tokens, not Cognito users; the "
            "flex_authorizer Lambda validates the token with Twilio.")
        options = ctx.add_method("OPTIONS", ctx_integration,
                                 authorization_type=apigw.AuthorizationType.NONE)
        for rule in ("AwsSolutions-APIG4", "AwsSolutions-COG4"):
            ack(options, rule,
                "CORS preflight can't carry the Flex token; context_api answers OPTIONS with CORS "
                f"headers for {cfg.flex_origin} only and returns no data.")

        cors = {"Access-Control-Allow-Origin": f"'{cfg.flex_origin}'", "Vary": "'Origin'"}
        for rtype in (apigw.ResponseType.UNAUTHORIZED, apigw.ResponseType.ACCESS_DENIED):
            self.api.add_gateway_response(f"Gw{rtype.response_type}", type=rtype, response_headers=cors)

        # Account-level role that lets API Gateway write execution logs.
        ack(self.api, "AwsSolutions-IAM4[Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AmazonAPIGatewayPushToCloudWatchLogs]",
            "This is the AWS-documented managed policy for API Gateway's account-level "
            "CloudWatch role; it grants only log-writing actions.")

        if cfg.domain_name:
            cert = acm.Certificate.from_certificate_arn(self, "Cert", cfg.effective_certificate_arn())
            domain = apigw.DomainName(
                self, "Domain", domain_name=cfg.api_host, certificate=cert,
                endpoint_type=apigw.EndpointType.REGIONAL, security_policy=apigw.SecurityPolicy.TLS_1_2)
            domain.add_base_path_mapping(self.api, stage=self.api.deployment_stage)
            # Read by scripts/twilio_setup.py (webhook URLs). Only with a custom domain: the
            # execute-api endpoint is off then, and the scripts expect a bare host.
            CfnOutput(self, "ApiDomain", value=cfg.api_host,
                      description=f"Alias this to {domain.domain_name_alias_domain_name}")
            CfnOutput(self, "ApiDomainTarget", value=domain.domain_name_alias_domain_name)

        else:
            # The default endpoint's host and stage, read by scripts/twilio_setup.py: the webhook
            # URL is https://<ApiDomain>/<ApiStage>/voice/incoming. ``ApiDomain`` is the host only.
            CfnOutput(self, "ApiDomain", value=f"{self.api.rest_api_id}.execute-api.{self.region}.{self.url_suffix}",
                      description="API Gateway default endpoint host (execute-api stays on without a domain)")
            CfnOutput(self, "ApiStage", value=self.api.deployment_stage.stage_name,
                      description="Stage name: the first path segment of the webhook URLs")

        # Every Twilio webhook arrives from Twilio's shared egress IPs, so a per-IP
        # rate rule throttles all of OwlBank's calls together at volume. wafApiRateLimit
        # (default 2000 per 5 minutes per IP) is sized for the sandbox, where Twilio calls
        # from a handful of IPs at low volume; production raises it after a load test. The
        # volume control is the stage throttling above (50 rps, burst 100), which doesn't care
        # which IP the requests come from; API Gateway applies it per method (each route's POST
        # has its own bucket; developer guide, "Throttle requests to your REST APIs").
        # /voice/transcription (every transcript event of a human-leg call, both tracks) is
        # counted by its own per-IP rate rule, not the voice webhooks', so transcript callbacks
        # cannot use up the count that the call's own webhooks need (production guide, human leg).
        # Twilio always sends HTTP Basic credentials to /voice/*, so a request without an
        # Authorization header is blocked at the edge (the handler still checks the credentials).
        # Managed rule groups follow wafManagedRuleMode (count first, then block). API Gateway's
        # default 16 KB body inspection limit already covers this ACL's 8 KB body cap.
        # agentcore: /voice/orchestrator is allowed (POST), already under the Basic challenge's
        # /voice/ regex, and the one large-body path (64 KB, inspection raised; 8 KB elsewhere).
        stage_re = rf"^(/{STAGE})?"
        orchestrator_re = rf"{stage_re}/voice/{ORCHESTRATOR_ROUTE}$"
        large_body: dict[str, Any] = ({"large_body_paths": (orchestrator_re,),
                                       "large_body_max_bytes": ORCHESTRATOR_MAX_BODY_BYTES,
                                       "inspection_resource": "API_GATEWAY"} if cfg.is_agentcore else {})
        acl_name = f"{cfg.name_prefix}-api"
        self.web_acl = api_waf(
            self, cfg=cfg, key=key, api=self.api, name=acl_name, metric="owlbank-api", which="api",
            allowed=[(rf"{stage_re}/voice/({'|'.join(voice_routes(cfg))})$", ["POST"]),
                     *([(orchestrator_re, ["POST"])] if cfg.is_agentcore else []),
                     (rf"{stage_re}/context/[A-Za-z0-9_-]{{1,128}}$", ["GET", "OPTIONS"])],
            authorization_required_path=rf"{stage_re}/voice/",
            own_rate_paths=((rf"{stage_re}/voice/{TRANSCRIPTION_ROUTE}$",)
                            if cfg.human_leg_conversation_on else ()),
            **large_body)

        # ---- Alarms
        api_alarms(Alarms(self, alarm_topic, cfg), self.api, acl_name, self.region)
