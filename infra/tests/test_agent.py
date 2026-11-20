"""AgentCore resources: Memory, Gateway, Gateway target. No harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import FAKE_CERT

SRC = Path(__file__).resolve().parents[2] / "src"


def _one(synth, rtype):
    found = [r["Properties"] for _, _, r in synth.resources(rtype)]
    assert len(found) == 1, rtype
    return found[0]


def test_memory_short_term_only(synth_default):
    m = _one(synth_default, "AWS::BedrockAgentCore::Memory")
    assert m["EventExpiryDuration"] == 3
    assert "MemoryStrategies" not in m
    assert "EncryptionKeyArn" in m


def test_gateway_iam_inbound_and_level0_target(synth_default):
    g = _one(synth_default, "AWS::BedrockAgentCore::Gateway")
    assert g["AuthorizerType"] == "AWS_IAM" and g["ProtocolType"] == "MCP" and "KmsKeyArn" in g
    t = _one(synth_default, "AWS::BedrockAgentCore::GatewayTarget")
    lam = t["TargetConfiguration"]["Mcp"]["Lambda"]
    assert "Level0Tools" in json.dumps(lam["LambdaArn"])
    names = {tool["Name"] for tool in lam["ToolSchema"]["InlinePayload"]}
    assert names == {"get_branch_hours", "faq_lookup"}
    assert t["CredentialProviderConfigurations"] == [{"CredentialProviderType": "GATEWAY_IAM_ROLE"}]


def _non_strings(node, path="$"):
    if isinstance(node, dict):
        return [x for k, v in node.items() for x in _non_strings(v, f"{path}.{k}")]
    if isinstance(node, list):
        return [x for i, v in enumerate(node) for x in _non_strings(v, f"{path}[{i}]")]
    return [] if isinstance(node, str) else [(path, node)]


def test_gateway_target_tool_schemas_have_no_bool_or_number(synth_default):
    """The Gateway target's tool schemas are CloudFormation Json properties."""
    t = _one(synth_default, "AWS::BedrockAgentCore::GatewayTarget")
    payload = t["TargetConfiguration"]["Mcp"]["Lambda"]["ToolSchema"]["InlinePayload"]
    assert _non_strings(payload) == []


def test_agentcore_roles_trust_only_agentcore_with_source_account(synth_default):
    roles = [r for _, lid, r in synth_default.resources("AWS::IAM::Role") if lid.startswith("GatewayRole")]
    assert len(roles) == 1
    for r in roles:
        st = r["Properties"]["AssumeRolePolicyDocument"]["Statement"]
        assert len(st) == 1 and st[0]["Principal"] == {"Service": "bedrock-agentcore.amazonaws.com"}
        assert st[0]["Condition"]["StringEquals"]["aws:SourceAccount"] == "111111111111"


def _bedrock_statements(synth):
    out = []
    for _, _, res in synth.resources("AWS::IAM::Policy"):
        for st in res["Properties"]["PolicyDocument"]["Statement"]:
            acts = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
            if any(a.startswith("bedrock:InvokeModel") for a in acts):
                out.append(st)
    return out


def test_foundation_model_only_through_inference_profile(synth_default):
    sts = _bedrock_statements(synth_default)
    assert sts, "no Bedrock invoke statements found"
    blob = json.dumps(sts)
    assert "inference-profile/us.anthropic.claude-haiku-5-5" in blob
    for st in sts:
        res = st["Resource"] if isinstance(st["Resource"], list) else [st["Resource"]]
        if any("foundation-model" in json.dumps(r) for r in res):
            cond = st.get("Condition", {}).get("StringEquals", {})
            assert "bedrock:InferenceProfileArn" in cond, st


@pytest.mark.parametrize("stage", ["dev", "test", "prod"])
@pytest.mark.parametrize("ctx,want", [({}, True),
                                      ({"acceptSpokenFactors": "false"}, False),
                                      ({"acceptSpokenFactors": False}, False),
                                      ({"acceptSpokenFactors": "true"}, True),
                                      ({"acceptSpokenFactors": True}, True)])
def test_accept_spoken_factors_stage_default_and_override(stage, ctx, want):
    """On by default in every stage; ``acceptSpokenFactors=false`` turns it off in each."""
    from owlbank_infra.config import from_context

    base = Path(__file__).resolve().parents[1]
    prod = {"domainName": "owlbank.example.com", "certificateArn": FAKE_CERT} if stage == "prod" else {}
    assert from_context({"stage": stage, **prod, **ctx}, base_dir=base).accept_spoken_factors is want


@pytest.mark.parametrize("stage", ["dev", "test", "prod"])
def test_accept_spoken_factors_is_on_in_every_stage(stage):
    from owlbank_infra.config import InfraConfig

    assert InfraConfig(stage=stage).accept_spoken_factors is True  # type: ignore[call-arg]


def test_gateway_target_name_is_the_pinned_prefix(synth_default):
    """The Gateway target name is ``tools_contract.GATEWAY_TARGET_NAME``, the prefix of every tool
    name the Gateway exposes (``level0___<tool>``), which the bridge's Strands client maps back."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from owlbank.tools_contract import GATEWAY_LEVEL0_TOOLS, GATEWAY_TARGET_NAME

    t = _one(synth_default, "AWS::BedrockAgentCore::GatewayTarget")
    assert t["Name"] == GATEWAY_TARGET_NAME
    payload = t["TargetConfiguration"]["Mcp"]["Lambda"]["ToolSchema"]["InlinePayload"]
    exposed = {f"{t['Name']}___{d['Name']}" for d in payload}
    assert exposed == {f"{GATEWAY_TARGET_NAME}___{n}" for n in GATEWAY_LEVEL0_TOOLS}


def test_no_harness_anywhere(synth_any):
    """No harness resource, execution role, warm-up Lambda or harness grant in any stack."""
    types = {r["Type"] for _, _, r in synth_any.resources()}
    assert "AWS::BedrockAgentCore::Harness" not in types
    assert not [lid for _, lid, _ in synth_any.resources() if "Harness" in lid or "Warmer" in lid]
    blob = json.dumps(synth_any.templates)
    for word in ("InvokeHarness", "InvokeAgentRuntime", "harness_warmer", "OWLBANK_PREWARM_FUNCTION",
                 "OWLBANK_HARNESS", "OWLBANK_AGENT_BACKEND", "BotoLayer"):
        assert word not in blob, word


def test_agent_stack_holds_only_memory_gateway_and_its_role(synth_any):
    """The Agent stack after: Memory, Gateway, the Gateway's role and its policy. Nothing else."""
    tpl = synth_any.templates["OwlBankAgent"]
    types = sorted(r["Type"] for r in tpl["Resources"].values() if r["Type"] != "AWS::CDK::Metadata")
    assert types == ["AWS::BedrockAgentCore::Gateway", "AWS::BedrockAgentCore::Memory",
                     "AWS::IAM::Policy", "AWS::IAM::Role"]
