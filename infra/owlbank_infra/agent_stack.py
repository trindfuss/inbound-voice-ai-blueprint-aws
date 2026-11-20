"""Agent: AgentCore Memory (short-term only) and Gateway (IAM inbound),
used by the bridge's Strands agent loop. The Gateway's ``level0_tools`` target is created
by the Compute stack (``gateway_target.py``); this stack names the function by its deterministic
ARN only."""

from __future__ import annotations

from aws_cdk import CfnOutput, Stack
from aws_cdk import aws_bedrockagentcore as ac
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from constructs import Construct

from .config import InfraConfig
from .grants import grant_invoke, grant_key_use

MEMORY_EXPIRY_DAYS = 3  # the API minimum with no strategies


def _service_role(scope: Construct, cid: str, description: str, stack: Stack) -> iam.Role:
    """A role only AgentCore can assume, and only on behalf of this account (confused deputy)."""
    return iam.Role(
        scope, cid,
        description=description,
        assumed_by=iam.ServicePrincipal(
            "bedrock-agentcore.amazonaws.com",
            conditions={"StringEquals": {"aws:SourceAccount": stack.account},
                        "ArnLike": {"aws:SourceArn":
                                    f"arn:{stack.partition}:bedrock-agentcore:{stack.region}:{stack.account}:*"}}),
    )


class AgentStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, cfg: InfraConfig, key: kms.IKey,
                 level0_arn: str, **kw) -> None:
        super().__init__(scope, cid, **kw)

        # ---- Memory: no strategies (no long-term extraction), shortest expiry, CMK.
        self.memory = ac.CfnMemory(
            self, "Memory",
            name="owlbank_voice_stm",
            event_expiry_duration=MEMORY_EXPIRY_DAYS,
            encryption_key_arn=key.key_arn,
            description="OwlBank per-call short-term memory only; no strategies",
        )

        # ---- Gateway: IAM inbound; level0_tools is its only Lambda target (added by Compute).
        gw_role = _service_role(self, "GatewayRole", "OwlBank AgentCore Gateway: invoke level0_tools only", self)
        grant_invoke(gw_role, level0_arn, sid="InvokeLevel0Tools")
        grant_key_use(self, gw_role, key, sid="GatewayCmk")
        self.gateway = ac.CfnGateway(
            self, "Gateway",
            name="owlbank-level0",
            authorizer_type="AWS_IAM",
            protocol_type="MCP",
            role_arn=gw_role.role_arn,
            kms_key_arn=key.key_arn,
            # Wording kept from the harness era on purpose: changing it would update the deployed
            # Gateway, which this leaves unchanged.
            description="Level-0 tools (no customer data) for the OwlBank harness",
        )
        self.gateway.node.add_dependency(gw_role)
        # The bridge's Strands agent loop and the benchmark connect to the Gateway's MCP URL and the
        # Memory directly, so both are stack outputs.
        CfnOutput(self, "GatewayUrl", value=self.gateway.attr_gateway_url,
                  description="AgentCore Gateway MCP URL (level-0 tools), for the Strands agent client")
        CfnOutput(self, "MemoryId", value=self.memory.attr_memory_id,
                  description="AgentCore Memory ID (short-term history)")
