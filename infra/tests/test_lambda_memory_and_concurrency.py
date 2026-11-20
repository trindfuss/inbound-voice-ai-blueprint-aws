"""Lambda memory and provisioned concurrency on a ``live`` alias. The harness warmer
is gone with the harness: the last tests here check nothing of it is left.

Offline: synthesizes the app and reads the templates. The graph being acyclic is shown by the
synth itself succeeding (CDK refuses a cyclic cross-stack reference).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from owlbank_infra.config import ConfigError, from_context

PC_FUNCTIONS = ("frontdoor", "auth_service", "bank_tools")
CID = {"frontdoor": "Frontdoor", "auth_service": "AuthService", "bank_tools": "BankTools",
       "level0_tools": "Level0Tools", "context_api": "ContextApi", "flex_authorizer": "FlexAuthorizer"}
COMPUTE = "OwlBankCompute"


def _ctx(**kv):
    return from_context(kv, base_dir=Path("."))


def _functions(synth) -> dict[str, tuple[str, dict]]:
    """name -> (logical id, properties) of the app's Lambda functions, without model failover's
    poller, the human-leg conversation sweep and the recap writer (off the call path;
    test_model_failover_infra, test_human_leg_infra and test_recap_writer_infra cover them)."""
    out = {}
    for _, lid, r in synth.resources("AWS::Lambda::Function"):
        handler = r["Properties"].get("Handler", "")
        if handler.startswith("lambdas.") and not handler.startswith(
                ("lambdas.model_poller.", "lambdas.conversation_sweep.", "lambdas.recap_writer.")):
            out[handler.split(".")[1]] = (lid, r["Properties"])
    return out


def _env(synth, name: str) -> dict:
    return _functions(synth)[name][1].get("Environment", {}).get("Variables", {})


def _aliases(synth) -> dict[str, dict]:
    return {lid: r["Properties"] for _, lid, r in synth.resources("AWS::Lambda::Alias")}


def _role_statements(synth, stack: str, role_prefix: str) -> list[dict]:
    roles = [lid for s, lid, r in synth.resources("AWS::IAM::Role")
             if s == stack and lid.startswith(role_prefix)]
    assert len(roles) == 1, roles
    out = []
    for s, _, res in synth.resources("AWS::IAM::Policy"):
        if s == stack and {"Ref": roles[0]} in res["Properties"].get("Roles", []):
            out += res["Properties"]["PolicyDocument"]["Statement"]
    return out


def _invoke_resources(synth) -> list:
    """Every Resource of every lambda:InvokeFunction statement in the app's IAM policies."""
    found = []
    for _, _, res in synth.resources("AWS::IAM::Policy"):
        for st in res["Properties"]["PolicyDocument"]["Statement"]:
            if st.get("Action") == "lambda:InvokeFunction":
                found.append(st["Resource"])
    return found


# --------------------------------------------------------------------------- context values

def test_defaults():
    cfg = _ctx()
    assert (cfg.lambda_memory_mb, cfg.lambda_provisioned_concurrency) == (1024, 1)
    assert cfg.memory_mb("frontdoor") == cfg.memory_mb("auth_service") == cfg.memory_mb("bank_tools") == 1024
    assert [cfg.memory_mb(n) for n in ("level0_tools", "context_api", "flex_authorizer")] == [512] * 3


@pytest.mark.parametrize("value", [127, 10241, 0, -1, "abc", 1.5e3 + 0.5])
def test_memory_out_of_range_is_refused(value):
    with pytest.raises(ConfigError, match="lambdaMemoryMb"):
        _ctx(lambdaMemoryMb=value)


@pytest.mark.parametrize("value", [128, 1024, "2048", 10240])
def test_memory_in_range(value):
    assert _ctx(lambdaMemoryMb=value).lambda_memory_mb == int(value)


@pytest.mark.parametrize("value", [-1, 101, "many", 2.5])
def test_provisioned_concurrency_out_of_range_is_refused(value):
    with pytest.raises(ConfigError, match="lambdaProvisionedConcurrency"):
        _ctx(lambdaProvisionedConcurrency=value)


def test_provisioned_concurrency_zero_is_off_and_unqualified():
    cfg = _ctx(lambdaProvisionedConcurrency=0)
    assert not cfg.provisioned_concurrency_on
    assert cfg.invoke_arn("bank_tools", "arn:f") == "arn:f"
    on = _ctx(lambdaProvisionedConcurrency="2")
    assert on.invoke_arn("bank_tools", "arn:f") == "arn:f:live"
    assert on.invoke_arn("level0_tools", "arn:f") == "arn:f"  # the Gateway target stays unqualified


# --------------------------------------------------------------------------- memory

def test_memory_sizes(synth_default):
    mem = {n: p["MemorySize"] for n, (_, p) in _functions(synth_default).items()}
    assert mem == {"frontdoor": 1024, "auth_service": 1024, "bank_tools": 1024, "level0_tools": 512,
                   "context_api": 512, "flex_authorizer": 512}


def test_memory_context_applies_to_the_three_only(synth_tuned):
    mem = {n: p["MemorySize"] for n, (_, p) in _functions(synth_tuned).items()}
    assert {mem[n] for n in PC_FUNCTIONS} == {2048}
    assert mem["level0_tools"] == mem["context_api"] == mem["flex_authorizer"] == 512


# --------------------------------------------------------------------------- the live alias

def test_live_alias_with_provisioned_concurrency_on_exactly_the_three(synth_default):
    fns = _functions(synth_default)
    aliases = _aliases(synth_default)
    assert len(aliases) == 3
    by_fn = {json.dumps(a["FunctionName"]): a for a in aliases.values()}
    for name in PC_FUNCTIONS:
        a = by_fn[json.dumps({"Ref": fns[name][0]})]
        assert a["Name"] == "live"
        assert a["ProvisionedConcurrencyConfig"] == {"ProvisionedConcurrentExecutions": 1}
        assert "FunctionVersion" in a  # points at a published version, never $LATEST
    # no provisioned concurrency anywhere else (level0, context_api, authorizer)
    assert "ProvisionedConcurrencyConfig" not in json.dumps(
        [r["Properties"] for _, _, r in synth_default.resources("AWS::Lambda::Function")])


def test_provisioned_concurrency_count_follows_the_context(synth_tuned):
    counts = [a["ProvisionedConcurrencyConfig"]["ProvisionedConcurrentExecutions"]
              for a in _aliases(synth_tuned).values()]
    assert counts == [3, 3, 3]


def test_alias_is_not_created_when_concurrency_is_zero(synth_no_pc):
    assert not _aliases(synth_no_pc)
    assert not list(synth_no_pc.resources("AWS::Lambda::Version"))
    # nothing references an alias: every invoke is unqualified
    blob = json.dumps(synth_no_pc.templates)
    assert "Aliaslive" not in blob and ":live" not in blob


def test_no_pc_invokers_use_the_unqualified_arns(synth_no_pc):
    fd = _env(synth_no_pc, "frontdoor")
    fns = _functions(synth_no_pc)
    assert fd["OWLBANK_AUTH_SERVICE_FUNCTION"] == {"Fn::GetAtt": [fns["auth_service"][0], "Arn"]}
    assert fd["OWLBANK_BANK_TOOLS_FUNCTION"] == {"Fn::GetAtt": [fns["bank_tools"][0], "Arn"]}
    assert not json.dumps(_invoke_resources(synth_no_pc)).count(":*")


def test_frontdoor_calls_use_the_aliases(synth_default):
    fns = _functions(synth_default)
    fd = _env(synth_default, "frontdoor")
    alias = {n: next(lid for lid, a in _aliases(synth_default).items()
                     if a["FunctionName"] == {"Ref": fns[n][0]}) for n in PC_FUNCTIONS}
    assert fd["OWLBANK_AUTH_SERVICE_FUNCTION"] == {"Ref": alias["auth_service"]}
    assert fd["OWLBANK_BANK_TOOLS_FUNCTION"] == {"Ref": alias["bank_tools"]}


def test_api_integrations_and_permissions_use_the_frontdoor_alias(synth_flex):
    synth_default = synth_flex  # Flex mode: taskrouter and transcription routes exist
    stage_alias = next(lid for lid, a in _aliases(synth_default).items()
                       if a["FunctionName"] == {"Ref": _functions(synth_default)["frontdoor"][0]})
    api = synth_default.templates["OwlBankApi"]["Resources"]
    methods = [r["Properties"] for r in api.values()
               if r["Type"] == "AWS::ApiGateway::Method"
               and "FrontdoorFnAliaslive" in json.dumps(r["Properties"].get("Integration", {}))]
    # The seven voice routes (partial, enqueue-action) plus taskrouter
    # and transcription (humanLegConversation=on).
    assert len(methods) == 9
    for m in methods:
        text = json.dumps(m["Integration"]["Uri"])
        assert f"OwlBankCompute:ExportsOutputRef{stage_alias}" in text and "FrontdoorFnAliaslive" in text
    perms = [r["Properties"] for r in api.values() if r["Type"] == "AWS::Lambda::Permission"
             and "FrontdoorFnAliaslive" in json.dumps(r["Properties"]["FunctionName"])]
    assert len(perms) == 18  # nine routes, each with the deployed-stage and the Test permission


def test_every_invoke_grant_is_exact_and_alias_where_it_should_be(synth_any):
    """No ``:*``, no wildcard; the three provisioned functions are granted by alias ARN only;
    the Gateway's level0_tools target keeps the unqualified function."""
    resources = _invoke_resources(synth_any)
    assert resources
    blob = json.dumps(resources)
    assert ":*" not in blob and '"*"' not in blob
    for r in resources:
        text = json.dumps(r)
        if "level0" in text.lower() or "Level0" in text:
            assert ":live" not in text and "Alias" not in text, text
    alias_lids = set(_aliases(synth_any))
    assert any(r == {"Ref": lid} for r in resources for lid in alias_lids)
    # the three functions' own (unqualified) ARNs appear in no invoke grant
    for name in PC_FUNCTIONS:
        lid = _functions(synth_any)[name][0]
        assert {"Fn::GetAtt": [lid, "Arn"]} not in resources, name
    # bridge: the qualified string ARN, not the unqualified one
    bridge_ok = [r for r in resources if isinstance(r, dict) and "owlbank-" in json.dumps(r)
                 and "bank-tools" in json.dumps(r)]
    stage = synth_any.stacks["data"].node.try_get_context("stage") or "dev"
    assert len(bridge_ok) == 1
    assert json.dumps(bridge_ok[0]).endswith(f':function:owlbank-{stage}-bank-tools:live"]]}}')


def test_bridge_env_names_the_bank_tools_alias(synth_any):
    (_, td), = [(lid, r["Properties"]) for _, lid, r in synth_any.resources("AWS::ECS::TaskDefinition")]
    env = {e["Name"]: e["Value"] for e in td["ContainerDefinitions"][0]["Environment"]}
    stage = synth_any.stacks["data"].node.try_get_context("stage") or "dev"
    assert json.dumps(env["OWLBANK_BANK_TOOLS_FUNCTION"]).endswith(f':function:owlbank-{stage}-bank-tools:live"]]}}')


def test_bridge_env_is_unqualified_without_pc(synth_no_pc):
    (_, td), = [(lid, r["Properties"]) for _, lid, r in synth_no_pc.resources("AWS::ECS::TaskDefinition")]
    env = {e["Name"]: e["Value"] for e in td["ContainerDefinitions"][0]["Environment"]}
    assert json.dumps(env["OWLBANK_BANK_TOOLS_FUNCTION"]).endswith(':function:owlbank-dev-bank-tools"]]}')


def test_gateway_level0_target_stays_unqualified(synth_default):
    (_, _, target), = list(synth_default.resources("AWS::BedrockAgentCore::GatewayTarget"))
    lam = target["Properties"]["TargetConfiguration"]["Mcp"]["Lambda"]
    assert "Level0Tools" in json.dumps(lam["LambdaArn"]) and ":live" not in json.dumps(lam["LambdaArn"])


def test_agent_and_bridge_do_not_import_from_compute(synth_default):
    """The alias ARN is a string in Bridge: the stack graph stays acyclic."""
    for stack in ("OwlBankAgent", "OwlBankBridge"):
        assert "OwlBankCompute:" not in json.dumps(synth_default.templates[stack]), stack


def test_function_names_stay_deterministic_for_the_two_named_functions(synth_default):
    names = {lid: r["Properties"].get("FunctionName")
             for _, lid, r in synth_default.resources("AWS::Lambda::Function")}
    assert {n for n in names.values() if n} == {"owlbank-dev-bank-tools", "owlbank-dev-level0-tools"}


# --------------------------------------------------------------------------- no warmer

def test_no_warmer_function_env_grant_or_layer(synth_any):
    """The six functions only, all on the one deps layer; no warmer, no env var, no grant."""
    fns = _functions(synth_any)
    assert set(fns) == set(CID)
    for _, p in fns.values():
        assert "DepsLayer" in json.dumps(p["Layers"]) and len(p["Layers"]) == 1
    assert "OWLBANK_PREWARM_FUNCTION" not in _env(synth_any, "frontdoor")
    sids = {st.get("Sid") for st in _role_statements(synth_any, COMPUTE, "FrontdoorRole")}
    assert "InvokeHarnessWarmer" not in sids
    blob = json.dumps(synth_any.templates)
    assert "HarnessWarmer" not in blob and "harness_warmer" not in blob and "BotoLayer" not in blob
    layers = [lid for _, lid, _ in synth_any.resources("AWS::Lambda::LayerVersion")]
    assert len(layers) == 1 and layers[0].startswith("DepsLayer")


def test_only_auth_service_has_an_async_config(synth_any):
    """The warmer's ``MaximumRetryAttempts=0`` config went with it; only auth_service's is left."""
    fn, _ = _functions(synth_any)["auth_service"]
    alias = [lid for lid, a in _aliases(synth_any).items() if a["FunctionName"] == {"Ref": fn}]
    cfgs = [r["Properties"] for _, _, r in synth_any.resources("AWS::Lambda::EventInvokeConfig")
            if not r["Properties"]["FunctionName"]["Ref"].startswith(
                ("ModelPollerFn", "ConversationSweepFn", "RecapWriterFn"))]
    # The poller and the sweep are never retried (scheduled); the recap writer's one retry is
    # test_recap_writer_infra's.
    assert len(cfgs) == 1 and cfgs[0]["FunctionName"] == {"Ref": fn} and alias


def test_auth_service_background_send_is_never_retried(synth_default):
    """The async invoke of auth_service (the background code send) goes to the ``live`` alias,
    whose async config has no retries, so a slow send can't text a second code."""
    fn, _ = _functions(synth_default)["auth_service"]
    cfgs = [r["Properties"] for _, _, r in synth_default.resources("AWS::Lambda::EventInvokeConfig")
            if r["Properties"]["FunctionName"] == {"Ref": fn}]
    # The qualifier is the alias (CloudFormation resolves it from the alias ARN), not $LATEST.
    assert any(c["Qualifier"] != "$LATEST" and "Fn::Select" in json.dumps(c["Qualifier"])
               and c["MaximumRetryAttempts"] == 0 for c in cfgs), cfgs
