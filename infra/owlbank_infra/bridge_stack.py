"""Bridge: ECS Fargate service for ``src/tac_bridge`` behind an ALB, in one of two exposure
modes, chosen by whether ``domainName`` is set.

* **Custom domain:** a public, HTTPS-only ALB (ACM certificate) with a regional WAF.
* **CloudFront (no domain, the default):** an internal ALB in the private subnets with an
  HTTP listener on port 80, reached only through a CloudFront VPC origin. The distribution
  (default ``*.cloudfront.net`` domain) carries a CLOUDFRONT-scope WAF. The ALB's security
  group accepts port 80 only from the CloudFront origin-facing managed prefix list.
* **Second control on the internal ALB:** the prefix list admits any CloudFront
  distribution, so the VPC origin also sends a secret header (``X-Origin-Verify``, a generated
  Secrets Manager secret, read through a ``{{resolve:secretsmanager:...}}`` dynamic reference, so
  the value is never in the template), and the HTTP listener forwards only requests that carry
  it; its default action is a fixed 403.
* Tasks run in private subnets and accept traffic only from the ALB's security group.
* Twilio secrets reach the container as task-definition ``secrets`` (``valueFrom`` a
  Secrets Manager ARN), never as plaintext ``environment`` values.
* The task role may invoke ``bank_tools`` and use the tables it needs, and runs the agent loop
  itself: the model, the Gateway and the Memory.

With ``bridgeHosting=agentcore`` the same stack holds the bridge as an AgentCore
Runtime instead (:meth:`BridgeStack._agentcore`): none of the above is created, except the retained
fixed-name resources, which stay dormant.
"""

from __future__ import annotations

import json

from aws_cdk import CfnOutput, Duration, RemovalPolicy, SecretValue, Stack
from aws_cdk import aws_certificatemanager as acm
from aws_cdk import aws_cloudfront as cf
from aws_cdk import aws_cloudfront_origins as cf_origins
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as sm
from aws_cdk import aws_sns as sns
from aws_cdk import aws_ssm as ssm
from aws_cdk import aws_wafv2 as waf
from aws_cdk.region_info import RegionInfo
from constructs import Construct

from . import call_observability, model_failover
from .azs import select_runtime_subnets
from .compute_stack import grant_secret_decrypt, import_secret, table_env
from .config import InfraConfig
from .contract import REPO_ROOT, TABLES
from .grants import BRIDGE_STAR_REASON, TOKEN_KEY_ENV, grant_bridge_backend
from .nag import ack
from .observability import (
    BRIDGE_AUTH_REJECTED_ID,
    BRIDGE_LOG_GROUP_NAME,
    BRIDGE_REJECT_MESSAGES,
    BRIDGE_REJECT_METRIC,
    Alarms,
    agentcore_log_group_names,
    saved_queries,
    waf_logging,
)
from .observability import log_group as log_group_fn
from .runtime import RuntimeNames, RuntimeParts, build_runtime
from .waf import web_acl

CONTAINER_PORT = 8080  # src/tac_bridge/Dockerfile
DOCKERFILE = REPO_ROOT / "src" / "tac_bridge" / "Dockerfile"
#: Build context = repo root, narrowed to what the Dockerfile COPYs (``src/owlbank``,
#: ``src/tac_bridge``), so the asset hash ignores the admin console, Lambdas, scripts and data
#: (tests/test_bridge_image_context.py).
IMAGE_EXCLUDE = ["**", "!src/owlbank/**", "!src/tac_bridge/**",
                 "**/__pycache__/**", "**/*.pyc"]

#: Paths and methods the bridge serves; the web ACLs block everything else. ``/health`` is not
#: here on purpose: the ALB's health checks go straight to the tasks and never pass the WAF.
ALLOWED_PATHS = [(r"^/ws$", ["GET"]), (r"^/webhook$", ["POST"])]
#: Paths whose bodies can legitimately exceed 8 KB (Conversation Orchestrator event callbacks
#: with transcripts); the CRS's 8 KB body rule is scoped around them (see ``waf.web_acl``).
LARGE_BODY_PATHS = (r"^/webhook$",)
#: Custom header the CloudFront VPC origin sends and the internal ALB's listener rule requires.
ORIGIN_VERIFY_HEADER = "X-Origin-Verify"

#: Construct IDs of the bridge's log group and WAF logging (``<id>Logs`` is the WAF log group). The
#: ``agentcore`` branch keeps the retained ones under the same IDs, so their logical IDs match.
BRIDGE_LOGS_ID = "BridgeLogs"
WAF_LOGGING_ID = "WafLogging"
ORIGIN_VERIFY_SECRET_ID = "OriginVerifySecret"  # noqa: S105 (a construct ID, not a secret)
#: Where the Orchestrator status callback goes in ``agentcore`` (the front-door sink on
#: ``OwlBankApi``); ``scripts/twilio_setup.py`` reads it from the ``OrchestratorCallbackPath`` output.
ORCHESTRATOR_CALLBACK_PATH = "/voice/orchestrator"
#: Construct ID of the relay parameter, the same in both hostings, so a switch
#: updates its value in place.
RELAY_PARAMETER_ID = "RelayParameter"


def waf_log_which(cfg: InfraConfig) -> str:
    """The bridge WAF log group's suffix: ``alb`` (custom domain) or ``cloudfront``."""
    return "alb" if cfg.custom_domain_mode else "cloudfront"


#: Env var -> (secret logical name, JSON field). TAC reads these names.
TASK_SECRETS = {
    "TWILIO_ACCOUNT_SID": ("twilio_api_key", "account_sid"),
    "TWILIO_API_KEY": ("twilio_api_key", "api_key"),
    "TWILIO_API_SECRET": ("twilio_api_key", "api_secret"),
    "TWILIO_AUTH_TOKEN": ("twilio_auth_token", "auth_token"),
}


#: The agent's model list: a JSON list of entries
#: ``{"id", "thinking", "max_tokens", "text_release", "prompt_caching"}`` in preference order.
MODELS_ENV = "OWLBANK_MODELS"
#: Set only with ``failoverTest=<seconds>``: the test-only first-token delay of the first entry, in
#: seconds.
FAILOVER_TEST_ENV = "OWLBANK_FAILOVER_TEST"


def models_env_value(cfg: InfraConfig) -> str:
    """``OWLBANK_MODELS``: compact JSON, entries in ``modelIds`` order."""
    return json.dumps([m.as_env() for m in cfg.models], separators=(",", ":"))


def strands_env(cfg: InfraConfig, gateway_url: str) -> dict[str, str]:
    """Container environment the Strands agent client reads: the
    model list with each entry's parameters (``modelIds``, ``modelParams`` over ``promptCaching`` and
    each model's known defaults), ``OWLBANK_PROMPT_CACHE_TTL`` (always written), the Gateway URL,
    and ``OWLBANK_FAILOVER_TEST=<seconds>`` only when ``failoverTest`` is set."""
    return {
        MODELS_ENV: models_env_value(cfg),
        "OWLBANK_PROMPT_CACHE_TTL": cfg.prompt_cache_ttl,
        "OWLBANK_GATEWAY_URL": gateway_url,
        **({FAILOVER_TEST_ENV: f"{cfg.failover_test:g}"} if cfg.failover_test is not None else {}),
    }


def bridge_base_env(stack: Stack, cfg: InfraConfig, *, tables: dict[str, ddb.ITable],
                    token_key: kms.IKey, bank_tools_arn: str, memory_id: str,
                    gateway_url: str, traces_log_group: str | None = None) -> dict[str, str]:
    """The bridge container's non-secret environment, the same in both hostings.

    Fargate adds ``TWILIO_VOICE_PUBLIC_DOMAIN`` (the relay host) and gets the Twilio secrets as
    task-definition ``secrets``; the AgentCore Runtime adds its hosting variables and the secrets'
    ARNs (``runtime.build_runtime``). Key order is the Fargate task definition's.

    With ``callObservability=on`` the call-observability variables come last
    (:func:`call_observability.bridge_env`), so with it off the keys and their order are unchanged.
    ``traces_log_group`` is the traces log group, the same in both hostings."""
    return {
        # Every table name (DynamoStore needs them all); IAM limits what is reachable.
        **table_env(tables, tuple(TABLES)),
        # Model failover: the fleet table name, only when on (none when off, so the
        # bridge builds no reporter or reader).
        **model_failover.bridge_env(cfg),
        "AWS_REGION": stack.region,
        "TAC_ANALYTICS_DISABLED": "true",
        "TWILIO_VOICE_WEBSOCKET_PATH": "/ws",
        "TWILIO_LOG_LEVEL": "INFO",
        "OWLBANK_LOG_LEVEL": cfg.python_log_level,  # The app's own log level
        "OWLBANK_BANK_TOOLS_FUNCTION": bank_tools_arn,
        TOKEN_KEY_ENV: token_key.key_arn,
        "OWLBANK_MEMORY_ID": memory_id,
        "OWLBANK_DEFAULT_LANGUAGE": "en-US",
        "OWLBANK_SUPPORTED_LANGUAGES": "en-US,es-US",
        **({"TWILIO_CONVERSATION_CONFIGURATION_ID": cfg.twilio_outputs["conversationConfigurationId"]}
           if "conversationConfigurationId" in cfg.twilio_outputs else {}),
        **({"TWILIO_PHONE_NUMBER": cfg.twilio_phone_number} if cfg.twilio_phone_number else {}),
        "OWLBANK_REUSE_CONVERSATION": "true" if cfg.reuse_conversation else "false",
        "OWLBANK_AI_KEYPAD": cfg.ai_keypad_shortcuts,  # keypad shortcuts
        "OWLBANK_TIMEZONE": cfg.timezone,
        # Web activity from Memory traits.
        "OWLBANK_MEMORY_TRAIT_GROUPS": ",".join(cfg.memory_trait_groups),
        "OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS": str(cfg.web_activity_max_age_hours),
        # The Strands agent loop's settings; OWLBANK_MEMORY_ID is above.
        **strands_env(cfg, gateway_url),
        **(call_observability.bridge_env(cfg, traces_log_group=traces_log_group)
           if cfg.call_observability_on else {}),
    }


class BridgeStack(Stack):
    """``OwlBankBridge``: the Fargate bridge (default) or, with ``bridgeHosting=agentcore``, the
    bridge as an AgentCore Runtime (one stack, two shapes; see :meth:`_agentcore`)."""

    def __init__(self, scope: Construct, cid: str, *, cfg: InfraConfig, vpc: ec2.IVpc | None, key: kms.IKey,
                 token_key: kms.IKey,
                 tables: dict[str, ddb.ITable], secrets: dict[str, sm.ISecret],
                 bank_tools_arn: str, memory_id: str, memory_arn: str, gateway_arn: str,
                 gateway_url: str, alarm_topic: sns.ITopic, **kw) -> None:
        super().__init__(scope, cid, **kw)
        custom_domain = cfg.custom_domain_mode
        self.distribution: cf.Distribution | None
        self.relay_domain: str
        #: The Runtime's parts (``agentcore`` only; ``None`` on Fargate).
        self.runtime: RuntimeParts | None = None
        if cfg.is_agentcore:
            # Returns before any ECS, ALB, CloudFront or WAF construct exists.
            self._agentcore(cfg, vpc=vpc, key=key, token_key=token_key, tables=tables,
                            secrets=secrets, bank_tools_arn=bank_tools_arn, memory_id=memory_id,
                            memory_arn=memory_arn, gateway_arn=gateway_arn, gateway_url=gateway_url,
                            alarm_topic=alarm_topic)
            return
        if vpc is None:
            raise ValueError("bridgeHosting=fargate needs the VPC (OwlBankNetwork)")

        # ---- ALB access logs (ALB log delivery supports SSE-S3 only, not a CMK).
        # s3PublicAccessBlockMode=account: an organization SCP can deny
        # s3:PutBucketPublicAccessBlock, which fails any bucket that sets the property, so the
        # property is omitted; S3's default for new buckets (all four on) and the account-level
        # setting protect it, and preflight --post-deploy reads the result back.
        log_bucket = s3.Bucket(
            self, "AlbLogs",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL if cfg.explicit_public_access_block else None,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,
            lifecycle_rules=[s3.LifecycleRule(expiration=Duration.days(90))],
            removal_policy=RemovalPolicy.RETAIN,
        )
        self._alb_log_delivery_policy(log_bucket, prefix="alb")
        ack(log_bucket, "AwsSolutions-S1",
            "This bucket receives ALB access logs; logging access to the log bucket itself would "
            "need another bucket with the same question. CloudTrail data events cover it if needed.")

        # ---- Security groups. Custom domain: the ALB is public on 443 only. CloudFront mode:
        # the ALB is internal and takes port 80 from the CloudFront origin-facing prefix list
        # only. Tasks are reachable only from the ALB in both.
        alb_sg = ec2.SecurityGroup(
            self, "AlbSg", vpc=vpc, allow_all_outbound=False,
            description=("OwlBank bridge ALB: HTTPS from Twilio" if custom_domain
                         else "OwlBank bridge internal ALB: HTTP from the CloudFront VPC origin"))
        if custom_domain:
            alb_sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(443), "Twilio ConversationRelay (dynamic IPs)")
            ack(alb_sg, "AwsSolutions-EC23",
                "Twilio connects the ConversationRelay WebSocket from dynamic IPs, so 443 must be "
                "open. HTTPS only, WAF in front, TAC validates X-Twilio-Signature. Static "
                "Proxy allowlisting is deferred to the production guide.")
        else:
            # Docs option 1 for VPC origins ("Restrict access with VPC origins"). Option 2, the
            # service-managed SG CloudFront-VPCOrigins-Service-SG, has no CloudFormation
            # attribute: it needs a custom resource (a Node.js Lambda with an unencrypted log
            # group), which the project rules out (see network_stack). The ALB is internal,
            # so only the VPC origin's network interfaces can reach it in any case.
            alb_sg.add_ingress_rule(
                ec2.Peer.prefix_list(cfg.cloudfront_prefix_list_id), ec2.Port.tcp(80),
                "CloudFront VPC origin (managed prefix list com.amazonaws.global.cloudfront.origin-facing)")
        task_sg = ec2.SecurityGroup(self, "TaskSg", vpc=vpc, allow_all_outbound=False,
                                    description="OwlBank bridge tasks: from the ALB only")
        # Outbound HTTPS only, through NAT: Twilio (TAC calls it at startup) and AWS APIs.
        task_sg.add_egress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(443), "HTTPS to Twilio and AWS APIs")
        task_sg.add_ingress_rule(alb_sg, ec2.Port.tcp(CONTAINER_PORT), "ALB to bridge")
        alb_sg.add_egress_rule(task_sg, ec2.Port.tcp(CONTAINER_PORT), "ALB to bridge")

        # ---- ECS cluster, task definition, service.
        cluster = ecs.Cluster(self, "Cluster", vpc=vpc,
                              container_insights_v2=ecs.ContainerInsights.ENABLED)
        log_group = log_group_fn(self, BRIDGE_LOGS_ID, cfg=cfg, key=key, name=BRIDGE_LOG_GROUP_NAME)
        # The task role's name is fixed: the token key policy names it.
        task_role = iam.Role(
            self, "TaskRole", role_name=cfg.bridge_task_role_name,
            description="OwlBank bridge task role (least privilege)",
            assumed_by=iam.ServicePrincipal(
                "ecs-tasks.amazonaws.com",
                conditions={"StringEquals": {"aws:SourceAccount": self.account},
                            "ArnLike": {"aws:SourceArn": f"arn:{self.partition}:ecs:{self.region}:{self.account}:*"}}))
        task_def = ecs.FargateTaskDefinition(
            self, "TaskDef", cpu=512, memory_limit_mib=1024, task_role=task_role,
            runtime_platform=ecs.RuntimePlatform(cpu_architecture=ecs.CpuArchitecture.ARM64,
                                                 operating_system_family=ecs.OperatingSystemFamily.LINUX))

        imported = {s: import_secret(self, f"Secret-{s}", secrets[s])
                    for s in sorted({s for s, _ in TASK_SECRETS.values()})}
        container = task_def.add_container(
            "bridge",
            image=self._image(),
            logging=ecs.LogDriver.aws_logs(stream_prefix="bridge", log_group=log_group),
            port_mappings=[ecs.PortMapping(container_port=CONTAINER_PORT)],
            environment=bridge_base_env(self, cfg, tables=tables, token_key=token_key,
                                        bank_tools_arn=bank_tools_arn, memory_id=memory_id,
                                        gateway_url=gateway_url,
                                        traces_log_group=call_observability.TRACES_LOG_GROUP_NAME),
            secrets={env: ecs.Secret.from_secrets_manager(imported[s], field)
                     for env, (s, field) in TASK_SECRETS.items()},
            readonly_root_filesystem=True,
            health_check=None,  # the ALB target group health check covers /health
        )
        grant_secret_decrypt(self, task_def.obtain_execution_role(), key)
        ack(task_def.obtain_execution_role(), "AwsSolutions-IAM5[Resource::*]",
            "ecr:GetAuthorizationToken doesn't support resource-level permissions, so ECS's "
            "image pull needs it on '*'. Image pulls themselves are scoped to the one repository.")
        task_def.add_volume(name="tmp")
        # S108: /tmp is the container path of the tmpfs volume the read-only root filesystem needs, not a host temp file.
        container.add_mount_points(
            ecs.MountPoint(container_path="/tmp", source_volume="tmp", read_only=False))  # noqa: S108
        ack(task_def, "AwsSolutions-ECS2",
            "The environment holds non-secret configuration only (table names, ARNs, ports). "
            "Every credential is a task-definition secret from Secrets Manager.")

        # Task role: bank_tools, its tables, and the agent loop's model, Gateway and Memory (below).
        # No secrets (the execution role injects them).
        # The same builder gives the AgentCore Runtime's execution role these statements.
        grant_bridge_backend(self, task_def.task_role, cfg=cfg, key=key, token_key=token_key,
                             tables=tables, bank_tools_arn=bank_tools_arn,
                             gateway_arn=gateway_arn, memory_arn=memory_arn,
                             star_reason=call_observability.bridge_star_reason(cfg, BRIDGE_STAR_REASON))
        # Call observability: the traces log group, its stream and X-Ray policy, the
        # TraceReader role and the task role's X-Ray grant. Nothing when off.
        if cfg.call_observability_on:
            call_observability.bridge_traces(self, cfg, task_role=task_def.task_role)

        service = ecs.FargateService(
            self, "Service", cluster=cluster, task_definition=task_def,
            desired_count=cfg.bridge_desired_count,
            assign_public_ip=False,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            security_groups=[task_sg],
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            min_healthy_percent=100, max_healthy_percent=200,
            health_check_grace_period=Duration.seconds(60),
        )

        # ---- ALB. Custom domain: public, HTTPS only, no HTTP listener. CloudFront mode:
        # internal, in the private subnets, HTTP on 80 (a sandbox trade-off).
        self.alb = elbv2.ApplicationLoadBalancer(
            self, "Alb", vpc=vpc, internet_facing=custom_domain, security_group=alb_sg,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC if custom_domain
                                            else ec2.SubnetType.PRIVATE_WITH_EGRESS),
            idle_timeout=Duration.seconds(cfg.alb_idle_timeout_s),
            drop_invalid_header_fields=True,
            deletion_protection=cfg.is_prod,
        )
        # Access logs; the bucket policy is _alb_log_delivery_policy (with aws:SourceAccount).
        self.alb.set_attribute("access_logs.s3.enabled", "true")
        self.alb.set_attribute("access_logs.s3.bucket", log_bucket.bucket_name)
        self.alb.set_attribute("access_logs.s3.prefix", "alb")
        self.alb.node.add_dependency(log_bucket)
        route: dict = {}
        if custom_domain:
            cert = acm.Certificate.from_certificate_arn(self, "Cert", cfg.effective_certificate_arn())
            listener = self.alb.add_listener("Https", port=443, protocol=elbv2.ApplicationProtocol.HTTPS,
                                             certificates=[cert], ssl_policy=elbv2.SslPolicy.TLS13_RES,
                                             open=False)
        else:
            # The default action is a fixed 403; the one forwarding rule needs the secret
            # origin header the VPC origin adds. The ALB's own health checks go straight from
            # the load balancer to the targets and never pass through listener rules, so the
            # target group's /health check is unaffected.
            self.origin_verify = self._origin_verify_secret(cfg, key)
            origin_verify_value = self._origin_verify_value(cfg)
            listener = self.alb.add_listener(
                "Http", port=80, protocol=elbv2.ApplicationProtocol.HTTP, open=False,
                default_action=elbv2.ListenerAction.fixed_response(
                    403, content_type="text/plain", message_body="Forbidden"))
            route = {"priority": 1, "conditions": [
                elbv2.ListenerCondition.http_header(ORIGIN_VERIFY_HEADER, [origin_verify_value])]}
        # HTTPS to the task in every mode: the container serves TLS with a self-signed certificate
        # made at start-up (src/tac_bridge/serve.py). The ALB doesn't validate target certificates.
        tg = listener.add_targets(
            "Bridge", port=CONTAINER_PORT, protocol=elbv2.ApplicationProtocol.HTTPS,
            targets=[service],
            deregistration_delay=Duration.seconds(cfg.deregistration_delay_s),
            health_check=elbv2.HealthCheck(path="/health", healthy_http_codes="200",
                                           protocol=elbv2.Protocol.HTTPS,
                                           interval=Duration.seconds(15), timeout=Duration.seconds(5)),
            **route,
        )

        if custom_domain:
            acl_name = f"{cfg.name_prefix}-alb"
            acl = web_acl(self, "WebAcl", name=acl_name, metric="owlbank-bridge",
                          allowed=ALLOWED_PATHS, rate_limit=cfg.waf_alb_rate_limit, max_body_bytes=65536,
                          managed_mode=cfg.waf_managed_mode)
            # An ALB hands WAF at most 8 KB of a body (fixed, not configurable), so bodies over
            # 8 KB are oversize here and BodySizeLimit (oversize matches) blocks them whatever the
            # 65536 says. Raise nothing here: the custom-domain mode's real limit is 8 KB.
            # The /ws query string carries the one-time WebSocket nonce: redact it.
            waf_logging(self, WAF_LOGGING_ID, cfg=cfg, key=key, acl=acl, which=waf_log_which(cfg),
                        redact_query_string=True)
            waf.CfnWebACLAssociation(self, "WebAclAssociation",
                                     resource_arn=self.alb.load_balancer_arn, web_acl_arn=acl.attr_arn)
            self.distribution = None
            #: The host Twilio connects to: the Compute stack builds the relay URL from it.
            self.relay_domain = cfg.bridge_host or ""
            CfnOutput(self, "RelayDomain", value=self.relay_domain,
                      description="Read by scripts/twilio_setup.py; alias it to AlbDnsName")
            CfnOutput(self, "AlbDnsName", value=self.alb.load_balancer_dns_name,
                      description=f"Point {cfg.bridge_host} (CNAME/alias) here")
        else:
            acl_name = f"{cfg.name_prefix}-cloudfront"
            acl = web_acl(self, "WebAcl", name=acl_name, metric="owlbank-bridge-cf",
                          allowed=ALLOWED_PATHS, rate_limit=cfg.waf_cloudfront_rate_limit,
                          max_body_bytes=65536, scope_type="CLOUDFRONT",
                          managed_mode=cfg.waf_managed_mode, large_body_paths=LARGE_BODY_PATHS)
            waf_logging(self, WAF_LOGGING_ID, cfg=cfg, key=key, acl=acl, which=waf_log_which(cfg),
                        redact_query_string=True)
            self.distribution = self._distribution(cfg, acl)
            self.relay_domain = self.distribution.distribution_domain_name
            CfnOutput(self, "RelayDomain", value=self.relay_domain,
                      description="CloudFront default domain; read by scripts/twilio_setup.py")
        self.web_acl = acl
        self.task_definition = task_def
        # TAC rebuilds the signed wss:// URL from this host; the bridge
        # pins X-Forwarded-Host to it. Same stack as the distribution: no cycle.
        container.add_environment("TWILIO_VOICE_PUBLIC_DOMAIN", self.relay_domain)
        # The front door builds wss://<host>/ws from this parameter (no export to Compute).
        self._relay_parameter(cfg, self.relay_domain,
                              "OwlBank relay WebSocket host (fargate); the front door builds wss://<host>/ws")

        # ---- Alarms
        alarms = Alarms(self, alarm_topic, cfg)
        five = Duration.minutes(5)
        alarms.alarm("AlbElb5xxAlarm",
                     self.alb.metrics.http_code_elb(elbv2.HttpCodeElb.ELB_5XX_COUNT, period=five, statistic="Sum"),
                     threshold=5, description="ALB-generated 5XX responses")
        alarms.alarm("AlbTarget5xxAlarm",
                     self.alb.metrics.http_code_target(elbv2.HttpCodeTarget.TARGET_5XX_COUNT, period=five,
                                                       statistic="Sum"),
                     threshold=5, description="Bridge (target) 5XX responses")
        alarms.alarm("UnhealthyHostsAlarm",
                     tg.metrics.unhealthy_host_count(period=Duration.minutes(1), statistic="Maximum"),
                     threshold=1, evaluation_periods=3, description="Bridge targets failing /health")
        if cfg.bridge_desired_count > 0:
            running = cw.Metric(namespace="ECS/ContainerInsights", metric_name="RunningTaskCount",
                                dimensions_map={"ClusterName": cluster.cluster_name,
                                                "ServiceName": service.service_name},
                                statistic="Minimum", period=Duration.minutes(1))
            alarms.alarm("RunningTasksAlarm", running, threshold=cfg.bridge_desired_count,
                         comparison=cw.ComparisonOperator.LESS_THAN_THRESHOLD, evaluation_periods=5,
                         missing=cw.TreatMissingData.BREACHING,
                         description=f"Fewer than {cfg.bridge_desired_count} bridge tasks running")
        if self.distribution is None:
            alarms.waf_blocked("AlbWafBlockedAlarm", acl_name, self.region)
        else:
            alarms.waf_blocked("CloudFrontWafBlockedAlarm", acl_name, "Global")
            # Rate of 5xx responses to viewers. CloudFront metrics live in us-east-1 and carry
            # Region=Global; the L2 helper omits that dimension, so the metric is spelled out.
            alarms.alarm("CloudFront5xxAlarm",
                         cw.Metric(namespace="AWS/CloudFront", metric_name="5xxErrorRate",
                                   dimensions_map={"DistributionId": self.distribution.distribution_id,
                                                   "Region": "Global"},
                                   statistic="Average", period=five),
                         threshold=5, evaluation_periods=2,
                         description="CloudFront 5xx error rate (percent of requests) above 5")
        alarms.reject_filter(BRIDGE_AUTH_REJECTED_ID, log_group, BRIDGE_REJECT_MESSAGES, BRIDGE_REJECT_METRIC,
                             "Bridge WebSocket signature failures (TAC log lines)")
        # Saved Logs Insights queries over all the app log groups (Lambdas and this bridge).
        # Here, not in Compute: this stack deploys last of the two, so every group exists. The
        # queries name the groups as strings, so there are no cross-stack references.
        saved_queries(self)
        # Model failover: model_unavailable and ActiveRank, only when on.
        model_failover.bridge_alarms(self, cfg, alarms, log_group)
        if cfg.call_observability_on:
            call_observability.CallAlarms(self, cfg, alarm_topic).build(log_group)
            call_observability.call_timeline_query(
                self, traces_log_group_name=call_observability.TRACES_LOG_GROUP_NAME)

    # ------------------------------------------------------------------ agentcore

    def _agentcore(self, cfg: InfraConfig, *, vpc: ec2.IVpc | None, key: kms.IKey, token_key: kms.IKey,
                   tables: dict[str, ddb.ITable], secrets: dict[str, sm.ISecret], bank_tools_arn: str,
                   memory_id: str, memory_arn: str, gateway_arn: str, gateway_url: str,
                   alarm_topic: sns.ITopic) -> None:
        """The bridge as an AgentCore Runtime, from :func:`runtime.build_runtime`.

        * The image by digest from our repository (``OwlBankBridgeImage``), named by its fixed
          name: the digest is required here, not by config, so the repository can deploy first.
        * The execution role keeps the Fargate task role's construct ID ``TaskRole`` and fixed name
          : the Data stack's token-key policy names it, and a switch updates it in place.
        * The DEFAULT endpoint only. ``vpc`` mode: the private subnets whose AZ ID AgentCore
          supports (``azs.select_runtime_subnets``) and a security group with egress on 443
          only; ``public`` mode: no VPC (``OwlBankNetwork`` isn't in the app).
        * The retained fixed-name Fargate resources stay, dormant, under their construct IDs.
        * Alarms and saved queries: the bridge's reject and error filters on the Runtime's
          log group, the ``AWS/Bedrock-AgentCore`` throttle, system-error and active-session alarms,
          and the saved queries over the Lambdas, the warmer and the Runtime's log group. No ECS,
          ALB, CloudFront or WAF alarm exists in this mode.
        * Call observability, only with ``callObservability=on``: the same traces log group,
          ``spans`` stream, X-Ray resource policy, ``TraceReader`` role and X-Ray grant as on
          Fargate (:func:`call_observability.bridge_traces`, with its TracesKey, on the execution role), and the same
          variables naming that group. The Runtime's own log group is unchanged.
        """
        digest = cfg.require_bridge_image_digest("bridgeHosting=agentcore")
        # The bridge exits at start-up without TAC's TWILIO_PHONE_NUMBER, and a Runtime has no
        # deployment circuit breaker, so fail synth rather than deploy sessions that can't start.
        cfg.require_twilio_phone_number("bridgeHosting=agentcore")
        runtime_vpc = vpc if cfg.agentcore_in_vpc else None
        if cfg.agentcore_in_vpc and runtime_vpc is None:
            raise ValueError("agentcoreNetworkMode=vpc needs the VPC (OwlBankNetwork)")
        subnets = (select_runtime_subnets(self, cfg, runtime_vpc.private_subnets)
                   if runtime_vpc is not None else [])
        # Same environment as the Fargate task, without the relay host (there is none).
        env = bridge_base_env(self, cfg, tables=tables, token_key=token_key, bank_tools_arn=bank_tools_arn,
                              memory_id=memory_id, gateway_url=gateway_url,
                              traces_log_group=call_observability.TRACES_LOG_GROUP_NAME)
        rt = build_runtime(
            self, cfg=cfg, names=RuntimeNames.bridge(cfg), key=key, token_key=token_key, tables=tables,
            secrets=secrets, repo_name=cfg.image_repo_name, image_digest=digest,
            bank_tools_arn=bank_tools_arn, memory_arn=memory_arn, gateway_arn=gateway_arn,
            environment=env, idle_timeout_s=cfg.agentcore_idle_timeout_s, vpc=runtime_vpc,
            subnet_ids=[s.subnet_id for s in subnets])
        self.runtime = rt
        if cfg.call_observability_on:
            call_observability.bridge_traces(self, cfg, task_role=rt.role)
        self.distribution = None
        # No relay host in this mode: the front door and the warmer read the Runtime ARN from the
        # relay parameter and name the Runtime by its fixed name prefix in IAM.
        self._relay_parameter(cfg, rt.runtime_arn,
                              "OwlBank bridge AgentCore Runtime ARN (agentcore); read by the front door and warmer")
        self._dormant_fargate_resources(cfg, key)

        rt_alarms = Alarms(self, alarm_topic, cfg)
        rt_alarms.runtime_alarms(
            runtime_name=cfg.runtime_name, runtime_arn=rt.runtime_arn, group=rt.log_group,
            session_quota=cfg.agentcore_session_quota)
        model_failover.bridge_alarms(self, cfg, rt_alarms, rt.log_group)  # when on
        # Here as on Fargate (this stack deploys after Compute, so every group exists).
        saved_queries(self, log_group_names=agentcore_log_group_names(rt.log_group_name))
        if cfg.call_observability_on:
            # The bridge's EMF and log lines are in the Runtime's log group.
            call_observability.CallAlarms(self, cfg, alarm_topic).build(rt.log_group)
            # Spans go to the traces group under the AgentCore hosting option too; the logs (and so the
            # alarms' metric filters above) stay in the Runtime's log group.
            call_observability.call_timeline_query(
                self, traces_log_group_name=call_observability.TRACES_LOG_GROUP_NAME)

        CfnOutput(self, "BridgeHosting", value="agentcore",
                  description="Bridge hosting; scripts treat a missing output as fargate")
        CfnOutput(self, "RuntimeArn", value=rt.runtime_arn)
        CfnOutput(self, "RuntimeId", value=rt.runtime.attr_agent_runtime_id)
        CfnOutput(self, "RuntimeLogGroup", value=rt.log_group_name,
                  description="The Runtime's application log group (scripts/bench/call_latency.py)")
        CfnOutput(self, "PresignRoleArn", value=rt.presign_role.role_arn)
        CfnOutput(self, "AgentcoreNetworkMode", value=cfg.agentcore_network_mode)
        CfnOutput(self, "OrchestratorCallbackPath", value=ORCHESTRATOR_CALLBACK_PATH,
                  description="Orchestrator status-callback path on the voice API (scripts/twilio_setup.py)")

    def _relay_parameter(self, cfg: InfraConfig, value: str, description: str) -> ssm.StringParameter:
        """The one Bridge→Compute value, as an SSM String parameter with a fixed name
        (``cfg.relay_parameter_name``) and the same construct ID in both hostings. Compute's Lambdas
        read it at start-up with ``ssm:GetParameter`` on this name only, so there is no
        CloudFormation export, and a hosting switch or rollback updates the value in place. Not a
        secret (a host name or an ARN), so a String, not a SecureString."""
        self.relay_parameter = ssm.StringParameter(
            self, RELAY_PARAMETER_ID, parameter_name=cfg.relay_parameter_name, string_value=value,
            description=description, tier=ssm.ParameterTier.STANDARD)
        return self.relay_parameter

    def _dormant_fargate_resources(self, cfg: InfraConfig, key: kms.IKey) -> None:
        """The Fargate resources that are ``RETAIN`` with a fixed name, kept in the template
        with the same construct IDs and properties, so a switch back to Fargate updates them instead
        of failing on "already exists". The origin-verify secret exists in the CloudFront mode
        only; the bridge and WAF log groups are retained only at ``stage=prod``."""
        if not cfg.custom_domain_mode:
            self.origin_verify = self._origin_verify_secret(cfg, key)
        if cfg.is_prod:
            log_group_fn(self, BRIDGE_LOGS_ID, cfg=cfg, key=key, name=BRIDGE_LOG_GROUP_NAME)
            log_group_fn(self, f"{WAF_LOGGING_ID}Logs", cfg=cfg, key=key,
                         name=f"aws-waf-logs-{cfg.name_prefix}-{waf_log_which(cfg)}")

    # ------------------------------------------------------------------ Fargate helpers

    def _distribution(self, cfg: InfraConfig, acl: waf.CfnWebACL) -> cf.Distribution:
        """The CloudFront mode's front door: default domain, WAF, a VPC origin to the internal ALB."""
        # CloudFront origin: HTTP only, because the internal ALB has no certificate.
        origin = cf_origins.VpcOrigin.with_application_load_balancer(
            self.alb, http_port=80, protocol_policy=cf.OriginProtocolPolicy.HTTP_ONLY,
            custom_headers={ORIGIN_VERIFY_HEADER: self._origin_verify_value(cfg)},
            read_timeout=Duration.seconds(cfg.cloudfront_origin_read_timeout_s),
            keepalive_timeout=Duration.seconds(cfg.cloudfront_origin_keepalive_timeout_s))
        # All viewer headers (Host, Upgrade, Connection, Sec-WebSocket-*, X-Twilio-Signature),
        # all query strings (the WebSocket nonce), no cookies. The CloudFront docs name
        # "forward all viewer headers" as the way to make WebSockets work.
        origin_request = cf.OriginRequestPolicy(
            self, "OriginRequestPolicy",
            comment="OwlBank bridge: all viewer headers and query strings, no cookies (WebSocket + Twilio signature)",
            header_behavior=cf.OriginRequestHeaderBehavior.all(),
            query_string_behavior=cf.OriginRequestQueryStringBehavior.all(),
            cookie_behavior=cf.OriginRequestCookieBehavior.none())
        dist = cf.Distribution(
            self, "Distribution",
            comment=f"OwlBank bridge ({cfg.stage}): ConversationRelay WebSocket and TAC webhook",
            price_class=cf.PriceClass.PRICE_CLASS_100,
            http_version=cf.HttpVersion.HTTP1_1,  # CloudFront WebSockets are HTTP/1.1 only
            enable_ipv6=False,
            web_acl_id=acl.attr_arn,
            default_behavior=cf.BehaviorOptions(
                origin=origin,
                viewer_protocol_policy=cf.ViewerProtocolPolicy.HTTPS_ONLY,  # Twilio uses wss://
                allowed_methods=cf.AllowedMethods.ALLOW_ALL,
                cache_policy=cf.CachePolicy.CACHING_DISABLED,
                origin_request_policy=origin_request,
                compress=False),
        )
        ack(dist, "AwsSolutions-CFR1",
            "No geo restriction: Twilio opens the WebSocket and webhooks from its own regions "
            "(US1 today, other regions by configuration), not from the caller's country, so a "
            "country allow-list adds risk of blocking Twilio and no protection. The CloudFront "
            "web ACL (managed rules, path allow-list, rate rule) and the Twilio signature and "
            "nonce checks protect the endpoint.")
        ack(dist, "AwsSolutions-CFR3",
            "CloudFront standard logging needs an S3 bucket with ACLs enabled, which the project "
            "rules out (BucketOwnerEnforced, no ACLs). Every request is logged by the CloudFront "
            "web ACL to a CMK-encrypted log group, with authorization, cookie and query string "
            "redacted, and the ALB writes access logs.")
        ack(dist, "AwsSolutions-CFR4",
            "The default *.cloudfront.net certificate fixes the minimum TLS version (TLSv1), "
            "and a stricter policy needs a custom domain and an ACM certificate. This is the "
            "no-domain sandbox mode's trade-off; Twilio negotiates TLS 1.2 or later. "
            "The production guide recommends the custom-domain mode with a TLS 1.3 ALB policy.")
        return dist

    def _origin_verify_secret(self, cfg: InfraConfig, key: kms.IKey) -> sm.Secret:
        """A generated secret for the ``X-Origin-Verify`` header. No script populates it: Secrets
        Manager generates it at creation (48 characters, no punctuation, so no ``*`` or ``?``,
        which an ALB header condition treats as wildcards), and the value is never in the template."""
        secret = sm.Secret(
            self, ORIGIN_VERIFY_SECRET_ID,
            secret_name=f"owlbank/{cfg.stage}/bridge/origin-verify",
            description="Value of the X-Origin-Verify header that the CloudFront VPC origin sends and "
                        "the internal ALB's listener rule requires. Generated; read by CloudFormation "
                        "through a dynamic reference.",
            encryption_key=key,
            generate_secret_string=sm.SecretStringGenerator(exclude_punctuation=True, password_length=48),
            removal_policy=RemovalPolicy.RETAIN,
        )
        ack(secret, "AwsSolutions-SMG4",
            "A shared header value, not a service credential: it is rotated by writing a new "
            "version and redeploying with originVerifySecretVersion (docs/production-guide.md), "
            "because the distribution and the ALB rule must change together; a rotation Lambda "
            "can't do that.")
        return secret

    def _origin_verify_value(self, cfg: InfraConfig) -> str:
        """``{{resolve:secretsmanager:<arn>:SecretString:::[<version>]}}``: CloudFormation reads the
        value when it creates or updates the resource, so the template holds only this reference.
        It is used in the CloudFront origin's custom header and the ALB rule's header condition."""
        return SecretValue.secrets_manager(
            self.origin_verify.secret_arn, version_id=cfg.origin_verify_secret_version).unsafe_unwrap()

    def _alb_log_delivery_policy(self, bucket: s3.Bucket, prefix: str) -> None:
        """What ``log_access_logs`` adds, plus ``aws:SourceAccount`` where the principal supports it.

        The regional ELB account (older Regions) is a cross-account principal: it sends no
        aws:SourceAccount, so that statement is scoped by the object path only. The
        service principals (newer Regions; delivery.logs) get aws:SourceAccount.
        """
        objects = bucket.arn_for_objects(f"{prefix}/AWSLogs/{self.account}/*")
        source = {"StringEquals": {"aws:SourceAccount": self.account}}
        elb_account = RegionInfo.get(self.region).elbv2_account
        if elb_account:
            bucket.add_to_resource_policy(iam.PolicyStatement(
                sid="AlbLogDeliveryElbAccount", actions=["s3:PutObject"], resources=[objects],
                principals=[iam.AccountPrincipal(elb_account)]))
        else:
            bucket.add_to_resource_policy(iam.PolicyStatement(
                sid="AlbLogDeliveryService", actions=["s3:PutObject"], resources=[objects],
                principals=[iam.ServicePrincipal("logdelivery.elasticloadbalancing.amazonaws.com")],
                conditions=source))
        bucket.add_to_resource_policy(iam.PolicyStatement(
            sid="LogDeliveryWrite", actions=["s3:PutObject"], resources=[objects],
            principals=[iam.ServicePrincipal("delivery.logs.amazonaws.com")],
            conditions={"StringEquals": {"s3:x-amz-acl": "bucket-owner-full-control",
                                         "aws:SourceAccount": self.account}}))
        bucket.add_to_resource_policy(iam.PolicyStatement(
            sid="LogDeliveryAclCheck", actions=["s3:GetBucketAcl"], resources=[bucket.bucket_arn],
            principals=[iam.ServicePrincipal("delivery.logs.amazonaws.com")], conditions=source))

    def _image(self) -> ecs.ContainerImage:
        if not DOCKERFILE.is_file():
            raise FileNotFoundError(f"bridge Dockerfile not found: {DOCKERFILE}")
        # Staged at synth (a directory copy); built and pushed only by a deploy.
        return ecs.ContainerImage.from_asset(
            str(REPO_ROOT), file=str(DOCKERFILE.relative_to(REPO_ROOT)), exclude=IMAGE_EXCLUDE,
            platform=ecr_assets.Platform.LINUX_ARM64)
