"""The bridge as an AgentCore Runtime.

:func:`build_runtime` adds, to any stack, the pieces one Runtime needs: the L1 ``CfnRuntime``
(``AWS::BedrockAgentCore::Runtime``), its execution role, its security group (VPC mode), its
pre-created log group and the presign role the front door assumes to mint WebSocket URLs. It is
parametrized by construct IDs (:class:`RuntimeIds`) and names (:class:`RuntimeNames`); the
``agentcore`` branch of ``OwlBankBridge`` uses it.

* **No inbound authorizer and no header allow-list**: the Runtime's default inbound auth is
  IAM SigV4, and the only callers are the pre-signed URL (presign role) and the warmer. No
  ``AuthorizerConfiguration`` and no ``RequestHeaderConfiguration``.
* **The DEFAULT endpoint**: no ``CfnRuntimeEndpoint``. Grants name the Runtime ARN ``R`` and
  ``R/runtime-endpoint/DEFAULT`` (:attr:`RuntimeParts.endpoint_arn`).
* **Execution role**: the bridge's application statements from
  :func:`grants.grant_bridge_backend` (the Fargate task role's, unchanged), plus what the Runtime
  needs itself: the two Twilio secrets (read at start-up by ARN), the image in the one
  repository, its own log group and the ``bedrock-agentcore`` metric namespace. No workload
  identity (no AgentCore Identity). X-Ray is optional: with ``callObservability=on`` the
  Bridge stack adds the X-Ray OTLP write grant (``call_observability.bridge_traces``, the same as
  the Fargate task role's) and the role's one IAM5 reason names those actions too
  (:func:`star_reason`); with it off there is no X-Ray.
* **Network**: ``vpc`` given → ``VPC`` mode in exactly the given subnets with a new security
  group (egress TCP 443 only, no ingress); no ``vpc`` → ``PUBLIC``.
* **Log group**: ``/aws/bedrock-agentcore/runtimes/<RuntimeId>-DEFAULT``, the name AgentCore
  writes to, created here with the CMK and the stage's retention and removal policy. The CMK's key
  policy must allow it (the Data stack's ``RuntimeLogsCmk`` statement).
* **Environment**: the caller's non-secret base environment (``bridge_stack.bridge_base_env``)
  plus ``OWLBANK_HOSTING=agentcore`` and the Twilio secrets' **ARNs**. No nonce-carrier setting: in
  ``agentcore`` the nonce always travels as ``<Parameter name="n">``. A Twilio credential
  variable in the caller's environment is refused: values never go in the template.
* **Call observability** (optional): with ``callObservability=on`` the
  Runtime also sets ``UNIFIED_TRACES_DESTINATION_ENABLED=true``. The bridge's own exporter
  sends its spans to the same traces log group as on Fargate (``/owlbank/traces/bridge``, named by
  ``OWLBANK_TRACES_LOG_GROUP`` in the caller's environment); this Runtime's log group keeps its
  logs, key and retention unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from aws_cdk import Duration, Stack, Token
from aws_cdk import aws_bedrockagentcore as ac
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from aws_cdk import aws_secretsmanager as sm

from .call_observability import XRAY_OTLP_REASON
from .compute_stack import grant_secret_decrypt
from .config import InfraConfig
from .data_stack import role_arn
from .grants import PII_DETECT_REASON, grant_bridge_backend
from .nag import ack
from .observability import RUNTIME_LOG_GROUP_PREFIX
from .observability import log_group as log_group_fn

#: The Runtime's fixed maximum session lifetime (the CFN maximum, 8 hours).
MAX_LIFETIME_S = 28800
#: ``IdleRuntimeSessionTimeout`` bounds (CFN).
IDLE_TIMEOUT_RANGE = (60, MAX_LIFETIME_S)
#: Environment variables the bridge reads in ``agentcore`` hosting (``src/tac_bridge/hosting.py``).
HOSTING_ENV = "OWLBANK_HOSTING"
#: Set to ``true`` with ``callObservability=on``.
UNIFIED_TRACES_ENV = "UNIFIED_TRACES_DESTINATION_ENABLED"
SECRET_ENV = {  # env var -> Data stack secret logical name
    "OWLBANK_SECRET_TWILIO_API_KEY": "twilio_api_key",
    "OWLBANK_SECRET_TWILIO_AUTH_TOKEN": "twilio_auth_token",
}
#: Twilio credential variables (Fargate's task-definition ``secrets``): never a Runtime env value.
CREDENTIAL_ENV = ("TWILIO_ACCOUNT_SID", "TWILIO_API_KEY", "TWILIO_API_SECRET", "TWILIO_AUTH_TOKEN")
#: The metric namespace the Runtime role may publish to (the AgentCore execution-role docs).
METRIC_NAMESPACE = "bedrock-agentcore"
ENDPOINT_SUFFIX = "/runtime-endpoint/DEFAULT"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

STAR_REASON = (
    PII_DETECT_REASON + " ecr:GetAuthorizationToken has no resource-level permissions either, so the "
    "image pull needs it on '*' (the pull itself, BatchGetImage and GetDownloadUrlForLayer, names the "
    "one repository). cloudwatch:PutMetricData has no resource types; it is limited by the "
    f"cloudwatch:namespace condition to '{METRIC_NAMESPACE}'. Nothing else uses '*'.")
_NOTHING_ELSE = " Nothing else uses '*'."


def star_reason(cfg: InfraConfig) -> str:
    """The execution role's one ``AwsSolutions-IAM5[Resource::*]`` reason: :data:`STAR_REASON` when
    call observability is off; when on, it also names the X-Ray OTLP write actions."""
    if not cfg.call_observability_on:
        return STAR_REASON
    return STAR_REASON.removesuffix(_NOTHING_ELSE) + " Also " + XRAY_OTLP_REASON + _NOTHING_ELSE


LOG_STREAMS_REASON = (
    "Log streams are created by the AgentCore service at run time with names we don't choose, so "
    "CreateLogStream and PutLogEvents name this Runtime's one log group and any stream in it "
    "(':log-stream:*'); the AgentCore docs' example grants every Runtime's group, which is narrowed here.")


@dataclass(frozen=True)
class RuntimeIds:
    """Construct IDs. The Bridge stack keeps ``TaskRole`` for the execution role."""

    runtime: str = "Runtime"
    role: str = "TaskRole"
    presign_role: str = "PresignRole"
    log_group: str = "RuntimeLogs"
    logs_policy: str = "RuntimeLogsPolicy"
    security_group: str = "RuntimeSg"


@dataclass(frozen=True)
class RuntimeNames:
    """Fixed names: the Runtime, its execution role, the presign role, and the one front-door role
    the presign role trusts."""

    runtime: str
    role: str
    presign_role: str
    frontdoor_role: str

    @classmethod
    def bridge(cls, cfg: InfraConfig) -> RuntimeNames:
        """The live ``agentcore`` bridge: the task role's name stays."""
        return cls(runtime=cfg.runtime_name, role=cfg.bridge_task_role_name,
                   presign_role=cfg.presign_role_name, frontdoor_role=cfg.frontdoor_role_name)


@dataclass(frozen=True)
class RuntimeParts:
    runtime: ac.CfnRuntime
    role: iam.Role
    presign_role: iam.Role
    log_group: logs.LogGroup
    security_group: ec2.SecurityGroup | None
    #: ``R``: the Runtime ARN.
    runtime_arn: str
    #: ``E``: the DEFAULT endpoint's ARN, ``R/runtime-endpoint/DEFAULT``.
    endpoint_arn: str
    log_group_name: str

    @property
    def invoke_resources(self) -> list[str]:
        """``[R, E]``: what every ``bedrock-agentcore:Invoke*`` grant names."""
        return [self.runtime_arn, self.endpoint_arn]


def build_runtime(stack: Stack, *, cfg: InfraConfig, names: RuntimeNames, key: kms.IKey,
                  token_key: kms.IKey, tables: dict[str, ddb.ITable], secrets: Mapping[str, sm.ISecret],
                  repo_name: str, image_digest: str, bank_tools_arn: str, memory_arn: str,
                  gateway_arn: str, environment: Mapping[str, str], idle_timeout_s: int,
                  vpc: ec2.IVpc | None = None, subnet_ids: Sequence[str] = (),
                  ids: RuntimeIds = RuntimeIds(),  # noqa: B008
                  description: str = "OwlBank ConversationRelay bridge") -> RuntimeParts:
    """Add one bridge Runtime and its roles, security group and log group to ``stack``."""
    _check(environment=environment, image_digest=image_digest, idle_timeout_s=idle_timeout_s,
           vpc=vpc, subnet_ids=subnet_ids)
    part, region, account = stack.partition, stack.region, stack.account

    # ---- Execution role: trust the AgentCore service for this account only.
    role = iam.Role(
        stack, ids.role, role_name=names.role,
        description="OwlBank bridge Runtime execution role (least privilege)",
        assumed_by=iam.ServicePrincipal(
            "bedrock-agentcore.amazonaws.com",
            conditions={"StringEquals": {"aws:SourceAccount": account},
                        "ArnLike": {"aws:SourceArn": f"arn:{part}:bedrock-agentcore:{region}:{account}:*"}}))
    grant_bridge_backend(stack, role, cfg=cfg, key=key, token_key=token_key, tables=tables,
                         bank_tools_arn=bank_tools_arn, gateway_arn=gateway_arn,
                         memory_arn=memory_arn, star_reason=star_reason(cfg))
    # Added for the Runtime: the secrets Fargate's execution role injected ...
    role.add_to_principal_policy(iam.PolicyStatement(
        sid="ReadTwilioSecrets", actions=["secretsmanager:GetSecretValue"],
        resources=[secrets[s].secret_arn for s in SECRET_ENV.values()]))
    grant_secret_decrypt(stack, role, key)
    # ... the image, in the one repository ...
    role.add_to_principal_policy(iam.PolicyStatement(
        sid="PullBridgeImage", actions=["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
        resources=[f"arn:{part}:ecr:{region}:{account}:repository/{repo_name}"]))
    role.add_to_principal_policy(iam.PolicyStatement(
        sid="EcrAuthToken", actions=["ecr:GetAuthorizationToken"], resources=["*"]))
    # ... and metrics in the AgentCore namespace only.
    role.add_to_principal_policy(iam.PolicyStatement(
        sid="RuntimeMetrics", actions=["cloudwatch:PutMetricData"], resources=["*"],
        conditions={"StringEquals": {"cloudwatch:namespace": METRIC_NAMESPACE}}))

    # ---- Security group (VPC mode): HTTPS out, nothing in (same egress rule as the task today).
    sg: ec2.SecurityGroup | None = None
    if vpc is not None:
        sg = ec2.SecurityGroup(stack, ids.security_group, vpc=vpc, allow_all_outbound=False,
                               description="OwlBank bridge Runtime: HTTPS out only, no ingress")
        sg.add_egress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(443), "HTTPS to Twilio and AWS APIs")
        network = ac.CfnRuntime.NetworkConfigurationProperty(
            network_mode="VPC", network_mode_config=ac.CfnRuntime.VpcConfigProperty(
                subnets=list(subnet_ids), security_groups=[sg.security_group_id]))
    else:
        network = ac.CfnRuntime.NetworkConfigurationProperty(network_mode="PUBLIC")

    # ---- The Runtime: HTTP protocol, IAM (SigV4) inbound auth, no header allow-list.
    env = {**environment, HOSTING_ENV: "agentcore",
           **{var: secrets[s].secret_arn for var, s in SECRET_ENV.items()},
           **({UNIFIED_TRACES_ENV: "true"} if cfg.call_observability_on else {})}
    runtime = ac.CfnRuntime(
        stack, ids.runtime,
        agent_runtime_name=names.runtime,
        description=description,
        agent_runtime_artifact=ac.CfnRuntime.AgentRuntimeArtifactProperty(
            container_configuration=ac.CfnRuntime.ContainerConfigurationProperty(
                container_uri=f"{account}.dkr.ecr.{region}.{stack.url_suffix}/{repo_name}@{image_digest}")),
        protocol_configuration="HTTP",
        lifecycle_configuration=ac.CfnRuntime.LifecycleConfigurationProperty(
            idle_runtime_session_timeout=idle_timeout_s, max_lifetime=MAX_LIFETIME_S),
        network_configuration=network,
        role_arn=role.role_arn,
        environment_variables=env,
    )
    # The role's policy (image pull, secrets) exists before AgentCore creates the Runtime.
    runtime.node.add_dependency(role)
    runtime_arn = runtime.attr_agent_runtime_arn
    endpoint_arn = f"{runtime_arn}{ENDPOINT_SUFFIX}"

    # ---- Log group, and the role's right to write to it. A separate policy, not the role's
    # default one: it names the group, which names the Runtime, which waits for the default policy.
    log_group_name = f"{RUNTIME_LOG_GROUP_PREFIX}{runtime.attr_agent_runtime_id}-DEFAULT"
    group = log_group_fn(stack, ids.log_group, cfg=cfg, key=key, name=log_group_name)
    group_arn = f"arn:{part}:logs:{region}:{account}:log-group:{log_group_name}"
    logs_policy = iam.Policy(stack, ids.logs_policy, roles=[role], statements=[
        iam.PolicyStatement(sid="RuntimeLogStreams", actions=["logs:CreateLogStream", "logs:PutLogEvents"],
                            resources=[f"{group_arn}:log-stream:*"]),
        iam.PolicyStatement(sid="RuntimeLogGroup", actions=["logs:DescribeLogStreams", "logs:CreateLogGroup"],
                            resources=[group_arn]),
    ])
    logs_policy.node.add_dependency(group)
    ack(logs_policy, f"AwsSolutions-IAM5[Resource::{_nag_stream_resource(stack, runtime)}]",
        LOG_STREAMS_REASON)

    # ---- Presign role: the front door assumes it to sign URLs; one statement, {R, E}.
    presign = iam.Role(
        stack, ids.presign_role, role_name=names.presign_role,
        description="OwlBank: pre-signs the bridge Runtime's WebSocket URLs for the front door",
        max_session_duration=Duration.hours(1),
        # The token-key pattern: the account root with an aws:PrincipalArn condition, so this role
        # can be created before the front-door role exists, and only that role can assume it.
        assumed_by=iam.AccountRootPrincipal().with_conditions(
            {"ArnEquals": {"aws:PrincipalArn": role_arn(stack, names.frontdoor_role)}}))
    presign.add_to_principal_policy(iam.PolicyStatement(
        sid="InvokeRuntimeWebSocket",
        actions=["bedrock-agentcore:InvokeAgentRuntimeWithWebSocketStream"],
        resources=[runtime_arn, endpoint_arn]))

    return RuntimeParts(runtime=runtime, role=role, presign_role=presign, log_group=group,
                        security_group=sg, runtime_arn=runtime_arn, endpoint_arn=endpoint_arn,
                        log_group_name=log_group_name)


def _nag_stream_resource(stack: Stack, runtime: ac.CfnRuntime) -> str:
    """The log-stream resource as cdk-nag spells it in the IAM5 finding ID (tokens as ``<...>``)."""
    def lit(value: str, pseudo: str) -> str:
        return f"<{pseudo}>" if Token.is_unresolved(value) else value

    rid = stack.resolve(runtime.logical_id)
    return (f"arn:<AWS::Partition>:logs:{lit(stack.region, 'AWS::Region')}:"
            f"{lit(stack.account, 'AWS::AccountId')}:log-group:{RUNTIME_LOG_GROUP_PREFIX}"
            f"<{rid}.AgentRuntimeId>-DEFAULT:log-stream:*")


def _check(*, environment: Mapping[str, str], image_digest: str, idle_timeout_s: int,
           vpc: ec2.IVpc | None, subnet_ids: Sequence[str]) -> None:
    if not _DIGEST.fullmatch(image_digest):
        raise ValueError("image_digest must be sha256: and 64 lowercase hex digits")
    lo, hi = IDLE_TIMEOUT_RANGE
    if not lo <= idle_timeout_s < hi:
        raise ValueError(f"idle_timeout_s must be at least {lo} and below the maximum lifetime {hi}")
    if vpc is None and subnet_ids:
        raise ValueError("subnet_ids given without a vpc (PUBLIC mode has no subnets)")
    if vpc is not None and not subnet_ids:
        raise ValueError("VPC mode needs at least one subnet")
    if len(set(subnet_ids)) != len(subnet_ids):
        raise ValueError("subnet_ids has duplicates")
    reserved = {HOSTING_ENV, UNIFIED_TRACES_ENV, *SECRET_ENV, *CREDENTIAL_ENV}
    clash = sorted(reserved & set(environment))
    if clash:
        raise ValueError(f"environment must not set {', '.join(clash)} (set here, or a credential)")
