"""Build the OwlBank CDK app: seven stacks, the bridge image repository included on both hostings
(plus the admin console; with ``agentcore`` + ``public``, no ``OwlBankNetwork``),
cdk-nag AwsSolutions on all."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aws_cdk import App, Environment, Tags, Validations
from cdk_nag import AwsSolutionsChecks

from . import node_bootstrap  # noqa: F401  (must run before aws_cdk loads jsii)
from .admin_stack import AdminStack
from .agent_stack import AgentStack
from .api_stack import ApiStack
from .bridge_stack import BridgeStack
from .compute_stack import ComputeStack
from .config import InfraConfig, check_context_keys, from_context
from .data_stack import DataStack
from .grants import function_arn
from .image_stack import BridgeImageStack
from .network_stack import NetworkStack

INFRA_DIR = Path(__file__).resolve().parents[1]
PREFIX = "OwlBank"

#: Every context key the app reads. Synth refuses any other key outside
#: CDK's own namespaces (``config.check_context_keys``), so a secret can't arrive as context and a
#: typo isn't silently ignored. Adding a setting means adding it here, where the reviewer decides
#: it isn't secret. Removed keys stay listed so ``from_context`` can refuse them with their own
#: message.
CONTEXT_KEYS: tuple[str, ...] = (
    "accountId", "region", "stage", "domainName", "certificateArn", "developerPhone",
    "testNumbers", "bridgeDesiredCount", "resolverModelId", "twilioOutputs",
    # The agent's model list; ``modelId`` is read only to refuse it.
    "modelIds", "modelParams", "failoverTest", "modelFailover", "modelId",
    # Removed in favour of modelParams; read only to refuse them.
    "agentThinking", "agentTextRelease", "agentMaxTokens",
    "albIdleTimeoutSeconds", "deregistrationDelaySeconds",
    "natGateways", "flexOrigin",
    "twilioPhoneNumber", "verifyServiceSid", "alarmEmail", "wafApiRateLimit", "wafAlbRateLimit",
    "owner", "costCenter", "dataClassification", "logLevel", "wafCloudFrontRateLimit",
    "cloudFrontOriginReadTimeoutSeconds", "cloudFrontOriginKeepaliveTimeoutSeconds",
    "cloudFrontOriginPrefixListId", "wafManagedRuleMode", "originVerifySecretVersion",
    "s3PublicAccessBlockMode", "lambdaMemoryMb", "lambdaProvisionedConcurrency",
    "acceptSpokenFactors", "scriptedBargeIn", "speechTimeout", "confirmSpeechTimeout", "keypadTimeout", "reuseConversation", "earlyDigits", "aiKeypadShortcuts", "timezone",
    "memoryTraitGroups", "webActivityMaxAgeHours",
    "promptCaching", "promptCacheTtl", "adminConsole", "adminEmail", "adminOriginVerifySecretVersion",
    # Bridge image and AZ keys; absent keys change no template.
    "bridgeImageDigest", "agentcoreSingleAzAck", "availabilityZones",
    # AgentCore hosting; absent keys change no template.
    "bridgeHosting", "agentcoreNetworkMode", "agentcoreNetworkModeAck",
    "agentcoreIdleTimeoutSeconds", "agentcoreSessionQuota",
    # Model-assisted slots.
    "slotModeCeiling",
    # Speech log of Gather turns.
    "speechLog", "speechLogRedaction",
    # Call observability; the defaults change no template.
    "callObservability", "traceSampleRate", "traceContent", "traceContentRetentionDays",
    "tracesEnableTransactionSearch", "ttftCeilingSeconds", "turnWaitCeilingSeconds",
    "silenceCeilingSeconds", "tokensPerCallCeiling",
    # Portable secrets: the user in the generated Basic auth secret; not a secret.
    "basicAuthUser",
    # Human-leg conversation.
    "humanLegConversation", "contextApiRestrictedKey",
)


def keep_network_exports(network: NetworkStack, *, custom_domain: bool) -> None:
    """``agentcore`` with the VPC: export everything the Fargate bridge imports from the Network
    stack (the VPC, every private subnet and, with a custom domain, the public subnets the ALB
    uses), so ``OwlBankNetwork``'s template is the same in both hostings. Without this, a switch
    would update Network first and fail removing an export the other hosting's Bridge still
    imports. Exports nothing imports are allowed."""
    vpc = network.vpc
    network.export_value(vpc.vpc_id)
    subnets = list(vpc.private_subnets) + (list(vpc.public_subnets) if custom_domain else [])
    for subnet in subnets:
        network.export_value(subnet.subnet_id)


def build_app(context: Mapping[str, Any] | None = None, *, outdir: str | None = None) -> tuple[App, dict]:
    """Return ``(app, stacks)``. ``context`` overrides ``cdk.json`` context."""
    ctx_in = dict(context or {})
    app = App(context=ctx_in, outdir=outdir)
    # The full context, CDK's own keys included, against the known-keys list.
    check_context_keys(app.node.get_all_context().keys(), CONTEXT_KEYS)
    ctx = {k: app.node.try_get_context(k) for k in CONTEXT_KEYS}
    cfg: InfraConfig = from_context(ctx, base_dir=INFRA_DIR)
    env = Environment(account=cfg.account, region=cfg.region)
    # Pre-seed the AZ context so no construct triggers an availability-zones lookup.
    app.node.set_context(f"availability-zones:account={cfg.account}:region={cfg.region}", cfg.azs)

    data = DataStack(app, f"{PREFIX}Data", cfg=cfg, env=env)
    # With agentcore + public nothing uses the VPC, so OwlBankNetwork is left out (no NAT or
    # flow logs to pay for).
    network = (NetworkStack(app, f"{PREFIX}Network", cfg=cfg, key=data.key, env=env)
               if not cfg.is_agentcore or cfg.agentcore_in_vpc else None)
    # Stack order: Agent and Bridge name the Lambdas they need by deterministic
    # ARN, so neither depends on Compute. Compute depends on Agent (the Gateway it adds the
    # level0 target to) and follows Bridge for deploy order only: the front door reads the relay
    # from Bridge's fixed-name SSM parameter at start-up, with no export. Api depends on
    # Compute. No cycles.
    # The bridge calls bank_tools through its ``live`` alias while provisioned concurrency is on
    #: built as a string like the unqualified ARN, so the graph stays acyclic. The Gateway's
    # level-0 target keeps the unqualified ARN (level0_tools has no alias).
    bank_tools_arn = cfg.invoke_arn("bank_tools", function_arn(data, cfg.function_name("bank_tools")))
    level0_arn = function_arn(data, cfg.function_name("level0_tools"))
    agent = AgentStack(app, f"{PREFIX}Agent", cfg=cfg, key=data.key, level0_arn=level0_arn, env=env)
    # AgentCore hosting: the bridge image's ECR repository, on both hostings, so a fresh account can
    # deploy the repository, push the image, then deploy agentcore. It names the CMK by its fixed
    # alias, so it follows Data for deploy order only, without an export; nothing depends on it in
    # Fargate, so the other templates don't change.
    image = BridgeImageStack(app, f"{PREFIX}BridgeImage", cfg=cfg, env=env)
    image.add_stack_dependency(data)
    bridge = BridgeStack(app, f"{PREFIX}Bridge", cfg=cfg, vpc=network.vpc if network else None,
                         key=data.key,
                         token_key=data.token_key,
                         tables=data.tables, secrets=data.secrets,
                         bank_tools_arn=bank_tools_arn,
                         memory_id=agent.memory.attr_memory_id,
                         memory_arn=agent.memory.attr_memory_arn,
                         gateway_arn=agent.gateway.attr_gateway_arn,
                         gateway_url=agent.gateway.attr_gateway_url, alarm_topic=data.alarm_topic,
                         env=env)
    compute = ComputeStack(app, f"{PREFIX}Compute", cfg=cfg, key=data.key, token_key=data.token_key,
                           tables=data.tables, secrets=data.secrets,
                           alarm_topic=data.alarm_topic,
                           gateway_identifier=agent.gateway.attr_gateway_identifier, env=env)
    # The relay parameter exists before the front door and the warmer start (they fail closed
    # without it). An ordering dependency only: no export, so either hosting deploys either way.
    compute.add_stack_dependency(bridge)
    api = ApiStack(app, f"{PREFIX}Api", cfg=cfg, key=data.key, functions=compute.invoke_targets,
                   alarm_topic=data.alarm_topic, env=env)
    stacks = {"data": data, **({"network": network} if network is not None else {}),
              "compute": compute, "agent": agent, "bridge": bridge, "api": api, "image": image}
    if cfg.is_agentcore:
        # The image is pushed between the two deploys.
        bridge.add_stack_dependency(image)
        if network is not None:
            keep_network_exports(network, custom_domain=cfg.custom_domain_mode)
    if cfg.admin_console:
        # Optional; nothing in the call path depends on it. It follows Api only
        # for deploy order (Api sets the account-level API Gateway CloudWatch role its stage needs).
        admin = AdminStack(app, f"{PREFIX}Admin", cfg=cfg, key=data.key, tables=data.tables, env=env)
        admin.add_stack_dependency(api)
        stacks["admin"] = admin

    for stack in stacks.values():
        Tags.of(stack).add("project", "owlbank-blueprint")
        Tags.of(stack).add("stage", cfg.stage)
        Tags.of(stack).add("owner", cfg.tag_owner)
        Tags.of(stack).add("cost-center", cfg.tag_cost_center)
        Tags.of(stack).add("data-classification", cfg.tag_data_classification)
    Validations.of(app).add_plugins(AwsSolutionsChecks(app))
    return app, stacks
