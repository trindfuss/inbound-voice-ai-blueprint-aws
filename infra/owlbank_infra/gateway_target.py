"""The AgentCore Gateway's ``level0_tools`` Lambda target.

Lives in its own module, and is created by the Compute stack (not the Agent stack): the
target needs the ``level0_tools`` function to exist, and Compute is deployed after Agent in
the CloudFront mode. The Gateway itself stays in the Agent stack, which refers to the
function only by its deterministic ARN.
"""

from __future__ import annotations

from typing import Any

from aws_cdk import aws_bedrockagentcore as ac
from constructs import Construct

from .contract import gateway_target_name, level0_tool_definitions

T = ac.CfnGatewayTarget
#: Also the prefix of every tool name the Gateway exposes (``level0___get_branch_hours``), which the
#: bridge's Strands client maps back to the tool. Defined once in ``tools_contract``.
GATEWAY_TOOL_NAME = gateway_target_name()


def _schema(s: dict[str, Any]) -> T.SchemaDefinitionProperty:
    props = s.get("properties")
    return T.SchemaDefinitionProperty(
        type=s["type"],
        description=s.get("description"),
        properties={k: _schema(v) for k, v in props.items()} if props else None,
        required=s.get("required") or None,
        items=_schema(s["items"]) if "items" in s else None,
    )


def level0_target(scope: Construct, cid: str, *, gateway_identifier: str, lambda_arn: str) -> T:
    tools = [T.ToolDefinitionProperty(name=d["name"], description=d["description"],
                                      input_schema=_schema(d["input_schema"]))
             for d in level0_tool_definitions()]
    return T(
        scope, cid,
        gateway_identifier=gateway_identifier,
        name=GATEWAY_TOOL_NAME,
        description="level0_tools Lambda: branch hours and FAQ",
        credential_provider_configurations=[T.CredentialProviderConfigurationProperty(
            credential_provider_type="GATEWAY_IAM_ROLE")],
        target_configuration=T.TargetConfigurationProperty(mcp=T.McpTargetConfigurationProperty(
            lambda_=T.McpLambdaTargetConfigurationProperty(
                lambda_arn=lambda_arn,
                tool_schema=T.ToolSchemaProperty(inline_payload=tools)))),
    )
