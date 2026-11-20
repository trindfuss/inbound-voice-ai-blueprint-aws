"""The agent's model list in infra.

``modelIds`` (one to three inference profiles, default the Region's geography profiles of Haiku 5.5,
Haiku 4.5 and Sonnet 5.5), ``modelParams`` (per-entry overrides) and ``failoverTest`` (never at
prod) are parsed at synth; the bridge role gets one statement pair per entry, each model only
through its profile (``global.`` gets AWS's empty-Region shape); both hostings get the same
``OWLBANK_MODELS``.

The statement property runs exhaustively over every ordered list of one to three entries drawn
from five profiles (85 lists) instead of with Hypothesis, which this uv project doesn't have: the
input space is small enough to enumerate, so nothing is sampled.
"""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as hst
from aws_cdk import App, Environment, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from conftest import _synth
from owlbank_infra.bridge_stack import FAILOVER_TEST_ENV, MODELS_ENV, strands_env
from owlbank_infra.compute_stack import BEDROCK_REGION_WILDCARD_REASON, bedrock_invoke_statements
from owlbank_infra.config import (
    DEFAULT_MODEL_BASES,
    ConfigError,
    InfraConfig,
    ModelEntry,
    default_model_ids,
    from_context,
)
from owlbank_infra.grants import STRANDS_MODEL_ACTIONS, grant_strands_backend, strands_model_sid
from owlbank_infra.nag import ACKS
from test_synth_and_nag import _nag_report

BASE = Path(__file__).resolve().parents[1]
README = BASE / "README.md"
BRIDGE = "OwlBankBridge"
REGION, ACCOUNT = "us-east-1", "111111111111"
H55, H45, S55 = (f"us.{b}" for b in DEFAULT_MODEL_BASES)
G_H55, G_S55 = "global.anthropic.claude-haiku-5-5", "global.anthropic.claude-sonnet-5-5"
FAKE_CERT = "arn:aws:acm:eu-west-1:111111111111:certificate/11111111-2222-3333-4444-555555555555"
#: Today's (pre-failover) two Strands model statements, from the snapshot before this change.
TODAY_PROFILE = "arn:aws:bedrock:us-east-1:111111111111:inference-profile/us.anthropic.claude-haiku-5-5"
TODAY_STATEMENTS = {
    "StrandsInvokeModelProfile": {"Action": list(STRANDS_MODEL_ACTIONS), "Effect": "Allow",
                                  "Resource": TODAY_PROFILE},
    "StrandsInvokeModelViaProfile": {
        "Action": list(STRANDS_MODEL_ACTIONS), "Effect": "Allow",
        "Condition": {"StringEquals": {"bedrock:InferenceProfileArn": TODAY_PROFILE}},
        "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-5-5"},
}


def cfg(**ctx: Any) -> InfraConfig:
    return from_context(ctx, base_dir=BASE)


def _flat(value: Any) -> Any:
    """A resolved CloudFormation value with ``Fn::Join`` and the partition ``Ref`` flattened."""
    if isinstance(value, dict):
        if set(value) == {"Fn::Join"}:
            sep, parts = value["Fn::Join"]
            return sep.join(str(_flat(p)) for p in parts)
        if value == {"Ref": "AWS::Partition"}:
            return "aws"
        return {k: _flat(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_flat(v) for v in value]
    return value


def _as_list(v: Any) -> list:
    return v if isinstance(v, list) else [v]


def _model_statements(statements: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for st in statements:
        if st.get("Sid", "").startswith("StrandsInvokeModel"):
            assert st["Sid"] not in out, st["Sid"]
            out[st["Sid"]] = st
    return out


def _bare_grant(models: tuple[ModelEntry, ...]) -> tuple[dict[str, dict], list[tuple[str, str, str]]]:
    """``grant_strands_backend`` on a bare stack: the role's model statements and its acks."""
    app = App()
    stack = Stack(app, "S", env=Environment(account=ACCOUNT, region=REGION))
    role = iam.Role(stack, "R", assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"))
    first = len(ACKS)
    grant_strands_backend(stack, role, cfg=InfraConfig(models=models), key=kms.Key(stack, "K"),
                          gateway_arn="arn:aws:bedrock-agentcore:us-east-1:111111111111:gateway/g",
                          memory_arn="arn:aws:bedrock-agentcore:us-east-1:111111111111:memory/m")
    doc = stack.resolve(role.node.find_child("DefaultPolicy").document.to_json())
    return _model_statements(_flat(doc["Statement"])), ACKS[first:]


def _entry(model_id: str) -> ModelEntry:
    return ModelEntry(id=model_id, thinking="off", max_tokens=400, text_release="message",
                      prompt_caching=False)


def _template_model_statements(synth, stack: str = BRIDGE) -> dict[str, dict]:
    """Sid -> statement of the model statements of the role(s) in ``stack`` (one role per hosting)."""
    statements = [st for res in synth.templates[stack]["Resources"].values()
                  if res["Type"] == "AWS::IAM::Policy"
                  for st in res["Properties"]["PolicyDocument"]["Statement"]]
    return _model_statements(_flat(statements))


def _fargate_env(synth) -> dict[str, Any]:
    td = next(r for r in synth.templates[BRIDGE]["Resources"].values()
              if r["Type"] == "AWS::ECS::TaskDefinition")
    (container,) = td["Properties"]["ContainerDefinitions"]
    return {e["Name"]: e["Value"] for e in container["Environment"]}


def _runtime_env(synth) -> dict[str, Any]:
    rt = next(r for r in synth.templates[BRIDGE]["Resources"].values()
              if r["Type"] == "AWS::BedrockAgentCore::Runtime")
    return rt["Properties"]["EnvironmentVariables"]


# ---- the list: defaults and parsing


def test_default_list_is_the_regions_geography_profiles():
    c = cfg()
    assert [m.id for m in c.models] == [H55, H45, S55]
    assert c.models == InfraConfig().models
    # Defaults: Haiku 5.5 and Sonnet 5.5 adaptive + stream; Haiku 4.5 off + message.
    assert [(m.thinking, m.text_release, m.max_tokens) for m in c.models] == [
        ("adaptive", "stream", 2000), ("off", "message", 1000), ("adaptive", "stream", 2000)]
    assert all(m.prompt_caching for m in c.models) and c.failover_test is None


def test_default_entries_have_caching_on_and_off_turns_every_entry_off():
    assert [m.as_env()["prompt_caching"] for m in cfg().models] == [True, True, True]
    assert [m.prompt_caching for m in cfg(promptCaching="off").models] == [False, False, False]
    c = cfg(promptCaching="off", modelParams={H45: {"prompt_caching": "on"}})
    assert [m.prompt_caching for m in c.models] == [False, True, False]
    c = cfg(modelParams={S55: {"prompt_caching": "off"}})
    assert [m.prompt_caching for m in c.models] == [True, True, False]


def test_one_hour_ttl_with_every_entry_caching_off_warns():
    with pytest.warns(UserWarning, match="promptCacheTtl=1h has no effect"):
        cfg(promptCaching="off", promptCacheTtl="1h")
    with pytest.warns(UserWarning, match="promptCacheTtl=1h has no effect"):  # modelParams off on the only entry
        cfg(modelIds=[H55], modelParams={H55: {"prompt_caching": "off"}}, promptCacheTtl="1h")


@pytest.mark.parametrize("ctx", [
    {"promptCaching": "off", "promptCacheTtl": "5m"}, {"promptCacheTtl": "1h"}, {"promptCaching": "off"},
    {"promptCaching": "off", "promptCacheTtl": "1h", "modelParams": {S55: {"prompt_caching": "on"}}}])
def test_no_ttl_warning_otherwise(ctx):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg(**ctx)


@settings(max_examples=60, deadline=None)
@given(caching=hst.sampled_from([None, "on", "off"]), ttl=hst.sampled_from([None, "5m", "1h"]),
       overrides=hst.dictionaries(hst.sampled_from([H55, H45, S55]),
                                 hst.sampled_from(["on", "off", True, False]), max_size=3))
def test_synth_half_entries_follow_precedence_and_env_ttl_is_chosen(caching, ttl, overrides):
    """``modelParams`` over ``promptCaching`` (default on) per entry, and the env
    TTL is the chosen ``promptCacheTtl`` (default 5m)."""
    ctx: dict[str, Any] = {}
    if caching is not None:
        ctx["promptCaching"] = caching
    if ttl is not None:
        ctx["promptCacheTtl"] = ttl
    if overrides:
        ctx["modelParams"] = {mid: {"prompt_caching": v} for mid, v in overrides.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        c = cfg(**ctx)
    base = (caching or "on") == "on"
    want = {mid: (v if isinstance(v, bool) else v == "on") for mid, v in overrides.items()}
    assert [m.prompt_caching for m in c.models] == [want.get(m.id, base) for m in c.models]
    env = strands_env(c, "https://gw.example")
    assert env["OWLBANK_PROMPT_CACHE_TTL"] == (ttl or "5m")
    assert [m["prompt_caching"] for m in json.loads(env[MODELS_ENV])] == [m.prompt_caching for m in c.models]


def test_default_list_follows_the_region():
    eu = cfg(region="eu-west-1", domainName="owlbank.example.com", certificateArn=FAKE_CERT)
    assert [m.id for m in eu.models] == [f"eu.{b}" for b in DEFAULT_MODEL_BASES]
    assert default_model_ids("ap-southeast-2")[0] == "apac.anthropic.claude-haiku-5-5"


def test_region_without_geography_needs_an_explicit_global_list():
    ctx = {"region": "ca-central-1", "domainName": "owlbank.example.com",
           "certificateArn": FAKE_CERT.replace("eu-west-1", "ca-central-1")}
    with pytest.raises(ConfigError, match="modelIds"):
        cfg(**ctx)
    with pytest.warns(UserWarning):
        assert [m.id for m in cfg(**ctx, modelIds=[G_H55]).models] == [G_H55]


@pytest.mark.parametrize("raw", [[H55, S55], json.dumps([H55, S55]), f"{H55}, {S55}", (H55, S55)])
def test_list_forms(raw):
    assert [m.id for m in cfg(modelIds=raw).models] == [H55, S55]


@pytest.mark.parametrize("raw", [None, "", "  "])
def test_unset_means_default(raw):
    assert cfg(modelIds=raw).models == InfraConfig().models


@pytest.mark.parametrize(("raw", "needle"), [
    ([], "1 to 3"), ([H55, H45, S55, G_H55], "1 to 3"), ([H55, H55], "repeat"),
    ("[not json", "JSON"), ({"a": 1}, "list"), ([H55, 3], "list"),
    (["anthropic.claude-haiku-5-5"], "not an inference-profile ID"),
    (["us.anthropic.claude haiku"], "not an inference-profile ID"),
    (["eu.anthropic.claude-haiku-5-5"], "another geography"),
    ([H55, "apac.anthropic.claude-sonnet-5-5"], "another geography"),
])
def test_bad_list_is_refused_naming_the_key(raw, needle):
    with pytest.raises(ConfigError, match="modelIds") as exc:
        cfg(modelIds=raw)
    assert needle in str(exc.value)


def test_another_geography_is_refused_at_synth(tmp_path):
    from owlbank_infra.app_builder import build_app
    with pytest.raises(ConfigError, match=r"modelIds entry 'eu\.anthropic\.claude-haiku-5-5'"):
        build_app({"modelIds": '["eu.anthropic.claude-haiku-5-5"]'}, outdir=str(tmp_path))


def test_global_entry_warns_naming_it():
    with pytest.warns(UserWarning, match=r"global\.anthropic\.claude-haiku-5-5.*any commercial Region"):
        cfg(modelIds=[G_H55, H45])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg(modelIds=[H55])  # a geographic list doesn't warn


def test_old_model_id_key_is_refused():
    with pytest.raises(ConfigError, match="modelId was replaced by modelIds"):
        cfg(modelId=H55)
    assert "model_id" not in InfraConfig.__dataclass_fields__


# ---- per-entry parameters


def test_model_params_override_one_entry():
    c = cfg(modelParams=json.dumps({H45: {"thinking": "enabled:2048", "max_tokens": 4000,
                                          "prompt_caching": "on"}}))
    h45 = c.models[1]
    assert (h45.thinking, h45.max_tokens, h45.text_release, h45.prompt_caching) == (
        "enabled:2048", 4000, "message", True)
    assert c.models[0] == InfraConfig().models[0] and c.models[2] == InfraConfig().models[2]


def test_caching_applies_to_every_entry_and_model_params_win():
    c = cfg(modelIds=[H55, S55, H45], promptCaching="on",
            modelParams={H55: {"thinking": "off", "text_release": "message", "max_tokens": 3000},
                         H45: {"max_tokens": 500}})
    assert [(m.thinking, m.text_release, m.max_tokens, m.prompt_caching) for m in c.models] == [
        ("off", "message", 3000, True), ("adaptive", "stream", 2000, True), ("off", "message", 500, True)]


@pytest.mark.parametrize("params", [{"thinking": "adaptive"}, {"text_release": "stream"},
                                    {"thinking": "off", "text_release": "message"},
                                    {"thinking": "default", "max_tokens": 64}])
def test_first_entry_params_with_the_default_list_are_accepted(params):
    """Each value the removed agent-wide keys accepted still synthesizes through ``modelParams`` on
    Haiku 5.5, and the fallbacks keep the settings their models need."""
    c = cfg(modelParams={H55: params})
    assert c.models[1:] == InfraConfig().models[1:]


def test_list_of_one_with_model_params():
    for params, caching, want in (({}, "off", ("adaptive", 2000, "stream", False)),
                                  ({"thinking": "off", "text_release": "message"}, "off",
                                   ("off", 1000, "message", False)),
                                  ({"thinking": "default", "max_tokens": 3000}, "on",
                                   ("default", 3000, "stream", True))):
        c = cfg(modelIds=[H55], promptCaching=caching, modelParams={H55: params})
        (m,) = c.models
        assert (m.thinking, m.max_tokens, m.text_release, m.prompt_caching) == want


@pytest.mark.parametrize(("ctx", "entry", "needle"), [
    ({"modelParams": {H55: {"thinking": "off"}}}, H55, "text_release=stream needs reasoning on"),
    ({"modelParams": {H45: {"thinking": "adaptive", "text_release": "stream"}}}, H45, "refused by this model"),
    ({"modelParams": {S55: {"thinking": "off", "text_release": "message"}}}, S55, "refused by this model"),
    ({"modelParams": {H55: {"thinking": "enabled:2048"}}}, H55, "refused by this model"),
    ({"modelParams": {H45: {"thinking": "enabled:512"}}}, H45, "at least 1024"),
    ({"modelParams": {H45: {"thinking": "enabled:x"}}}, H45, "at least 1024"),
    ({"modelParams": {H45: {"thinking": "enabled:4000", "max_tokens": 4000}}}, H45, "below max_tokens"),
    ({"modelParams": {H45: {"thinking": "deep"}}}, H45, "must be one of"),
    ({"modelParams": {H45: {"text_release": "chunk"}}}, H45, "text_release"),
    ({"modelIds": [H45], "modelParams": {H45: {"text_release": "stream"}}}, H45,
     "text_release=stream needs reasoning on"),
])
def test_bad_parameter_set_is_refused_naming_the_entry(ctx, entry, needle):
    with pytest.raises(ConfigError, match=f"modelIds entry '{entry}'") as exc:
        cfg(**ctx)
    assert needle in str(exc.value)


@pytest.mark.parametrize(("raw", "needle"), [
    ("[1]", "JSON object"), ({G_H55: {}}, "not in modelIds"), ({H55: "off"}, "must be an object"),
    ({H55: {"temperature": 1}}, "unknown field"), ({H55: {"thinking": 1}}, "must be a string"),
    ({H55: {"max_tokens": 10}}, "64 to 8192"), ({H55: {"max_tokens": True}}, "64 to 8192"),
    ({H55: {"max_tokens": "2000"}}, "64 to 8192"), ({H55: {"prompt_caching": "yes"}}, "prompt_caching"),
    ("{bad", "JSON"),
])
def test_bad_model_params_are_refused_naming_the_key(raw, needle):
    with pytest.raises(ConfigError, match="modelParams") as exc:
        cfg(modelParams=raw)
    assert needle in str(exc.value)


# ---- failoverTest


@pytest.mark.parametrize(("raw", "want"), [(None, None), ("", None), ("off", None), (" OFF ", None),
                                            ("6", 6.0), (6, 6.0), ("0.5", 0.5), (30, 30.0), ("30", 30.0)])
def test_failover_test_values(raw, want):
    assert cfg(failoverTest=raw).failover_test == want


@pytest.mark.parametrize("raw", ["on", "yes", "0", 0, -1, "30.01", 31, "nan", "inf", True, [6]])
def test_failover_test_bad_values_are_refused(raw):
    with pytest.raises(ConfigError, match="failoverTest must be off or a delay in seconds"):
        cfg(failoverTest=raw)


def test_failover_test_is_refused_at_prod():
    prod = {"stage": "prod", "domainName": "owlbank.example.com",
            "certificateArn": FAKE_CERT.replace("eu-west-1", "us-east-1")}
    assert cfg(**prod, failoverTest="off").failover_test is None
    with pytest.raises(ConfigError, match="failoverTest is not allowed at stage=prod"):
        cfg(**prod, failoverTest="6")


# ---- IAM: one statement pair per entry, each model only through its profile


def _expected_pair(model_id: str) -> tuple[str, list[str]]:
    prefix, base = model_id.split(".", 1)
    profile = f"arn:aws:bedrock:{REGION}:{ACCOUNT}:inference-profile/{model_id}"
    if prefix == "global":
        return profile, [f"arn:aws:bedrock:::foundation-model/{base}",
                         f"arn:aws:bedrock:{REGION}::foundation-model/{base}"]
    return profile, [f"arn:aws:bedrock:*::foundation-model/{base}"]


POOL = (H55, H45, S55, G_H55, G_S55)
ALL_LISTS = [lst for n in (1, 2, 3) for lst in itertools.permutations(POOL, n)]


def test_statements_for_every_valid_list():
    """Exhaustive over all 85 ordered lists from five profiles: exactly one Profile/ViaProfile pair
    per entry under that entry's Sid, the profile in the deployment Region, the foundation model
    only through that profile, only the two model actions, no other model, no wildcard action;
    one IAM5 acknowledgement per geographic entry and none for ``global.``."""
    assert len(ALL_LISTS) == 85
    for lst in ALL_LISTS:
        sts, acks = _bare_grant(tuple(_entry(m) for m in lst))
        assert len(sts) == 2 * len(lst), lst
        for i, mid in enumerate(lst):
            profile, models = _expected_pair(mid)
            p, v = sts[f"{strands_model_sid(i)}Profile"], sts[f"{strands_model_sid(i)}ViaProfile"]
            for st in (p, v):
                assert st["Effect"] == "Allow" and sorted(_as_list(st["Action"])) == sorted(STRANDS_MODEL_ACTIONS)
            assert _as_list(p["Resource"]) == [profile] and "Condition" not in p
            assert _as_list(v["Resource"]) == models
            assert v["Condition"] == {"StringEquals": {"bedrock:InferenceProfileArn": profile}}
        bases = {r.rsplit("/", 1)[1] for st in sts.values() for r in _as_list(st["Resource"])
                 if ":foundation-model/" in r}
        assert bases == {m.split(".", 1)[1] for m in lst}
        model_acks = [a for a in acks if a[1].startswith("AwsSolutions-IAM5[Resource::arn:<AWS::Partition>:bedrock:")]
        want = sorted(f"AwsSolutions-IAM5[Resource::arn:<AWS::Partition>:bedrock:*::foundation-model/"
                      f"{m.split('.', 1)[1]}]" for m in lst if not m.startswith("global."))
        assert sorted(a[1] for a in model_acks) == want, lst
        assert all(a[2] == BEDROCK_REGION_WILDCARD_REASON for a in model_acks)


def test_global_entry_shape():
    """AWS's documented global-profile shape: the profile here, and the one model as the
    empty-Region ARN and this Region's ARN, both only through the profile; no Region wildcard."""
    stack = Stack(App(), "G", env=Environment(account=ACCOUNT, region=REGION))
    sts = [_flat(stack.resolve(s.to_statement_json()))
           for s in bedrock_invoke_statements(stack, G_S55, sid="X", actions=list(STRANDS_MODEL_ACTIONS))]
    profile = f"arn:aws:bedrock:{REGION}:{ACCOUNT}:inference-profile/{G_S55}"
    assert sts == [
        {"Sid": "XProfile", "Effect": "Allow", "Action": list(STRANDS_MODEL_ACTIONS), "Resource": profile},
        {"Sid": "XViaProfile", "Effect": "Allow", "Action": list(STRANDS_MODEL_ACTIONS),
         "Resource": ["arn:aws:bedrock:::foundation-model/anthropic.claude-sonnet-5-5",
                      f"arn:aws:bedrock:{REGION}::foundation-model/anthropic.claude-sonnet-5-5"],
         "Condition": {"StringEquals": {"bedrock:InferenceProfileArn": profile}}},
    ]


@pytest.fixture(scope="module")
def synth_one(tmp_path_factory):
    return _synth(tmp_path_factory, "model-list-one", {"modelIds": [H55]})


@pytest.fixture(scope="module")
def synth_global_test(tmp_path_factory):
    with pytest.warns(UserWarning, match="global"):
        return _synth(tmp_path_factory, "model-list-global",
                      {"modelIds": json.dumps([G_H55, H45, S55]), "failoverTest": "6"})


@pytest.fixture(scope="module")
def synth_agentcore_failover_test(tmp_path_factory):
    from test_bridge_agentcore import _ctx, _quiet_synth
    return _quiet_synth(tmp_path_factory, "model-list-agentcore-test", {**_ctx("vpc", "dev"), "failoverTest": 4.5})


def test_list_of_one_equals_todays_statements(synth_one):
    sts = _template_model_statements(synth_one)
    assert {sid: {k: v for k, v in st.items() if k != "Sid"} for sid, st in sts.items()} == TODAY_STATEMENTS
    model_acks = [a for a in synth_one.acks if "foundation-model" in a[1]
                  and a[0].startswith(f"{BRIDGE}/")]
    assert [a[1] for a in model_acks] == [
        "AwsSolutions-IAM5[Resource::arn:<AWS::Partition>:bedrock:*::foundation-model/anthropic.claude-haiku-5-5]"]


def test_default_synth_grants_exactly_the_three_entries(synth_default):
    sts = _template_model_statements(synth_default)
    assert set(sts) == {f"{strands_model_sid(i)}{part}" for i in range(3) for part in ("Profile", "ViaProfile")}
    for i, mid in enumerate((H55, H45, S55)):
        profile, models = _expected_pair(mid)
        assert sts[f"{strands_model_sid(i)}Profile"]["Resource"] == profile
        assert _as_list(sts[f"{strands_model_sid(i)}ViaProfile"]["Resource"]) == models


def test_no_other_bridge_statement_names_bedrock_models(synth_default, synth_agentcore_vpc_dev):
    """Only the Strands model statements touch ``bedrock:InvokeModel*`` in the Bridge stack, in
    both hostings, and no statement anywhere in it has a wildcard Bedrock action."""
    for synth in (synth_default, synth_agentcore_vpc_dev):
        for res in synth.templates[BRIDGE]["Resources"].values():
            if res["Type"] != "AWS::IAM::Policy":
                continue
            for st in res["Properties"]["PolicyDocument"]["Statement"]:
                actions = _as_list(st["Action"])
                assert not any(a.startswith("bedrock:") and "*" in a for a in actions), st
                if any(a.startswith("bedrock:InvokeModel") for a in actions):
                    assert st["Sid"].startswith("StrandsInvokeModel"), st


def test_agentcore_role_gets_the_same_model_statements(synth_default, synth_agentcore_vpc_dev):
    assert _template_model_statements(synth_agentcore_vpc_dev) == _template_model_statements(synth_default)


def test_global_synth_statements_and_nag_clean(synth_global_test):
    sts = _template_model_statements(synth_global_test)
    profile, models = _expected_pair(G_H55)
    assert sts["StrandsInvokeModelProfile"]["Resource"] == profile
    assert sts["StrandsInvokeModelViaProfile"]["Resource"] == models
    report = _nag_report(synth_global_test)
    if report is not None:
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [(v["ruleName"], v.get("violatingResources")) for v in errors]


def test_list_of_one_is_nag_clean(synth_one):
    report = _nag_report(synth_one)
    if report is not None:
        assert not [v for v in report["violations"] if v["severity"] == "error"]


# ---- environment on both hostings


def test_models_env_on_fargate(synth_default):
    env = _fargate_env(synth_default)
    assert json.loads(env[MODELS_ENV]) == [m.as_env() for m in InfraConfig().models]
    assert FAILOVER_TEST_ENV not in env


def test_models_env_is_the_same_on_agentcore(synth_default, synth_agentcore_vpc_dev):
    assert _runtime_env(synth_agentcore_vpc_dev)[MODELS_ENV] == _fargate_env(synth_default)[MODELS_ENV]
    assert FAILOVER_TEST_ENV not in _runtime_env(synth_agentcore_vpc_dev)


def test_failover_test_env_on_agentcore(synth_agentcore_failover_test):
    assert _runtime_env(synth_agentcore_failover_test)[FAILOVER_TEST_ENV] == "4.5"


def test_failover_test_env_only_when_set(synth_global_test):
    env = _fargate_env(synth_global_test)
    assert env[FAILOVER_TEST_ENV] == "6"
    assert [m["id"] for m in json.loads(env[MODELS_ENV])] == [G_H55, H45, S55]


# ---- the settings table


def _row(key: str) -> list[str]:
    rows = [ln for ln in README.read_text().splitlines() if ln.startswith(f"| `{key}` |")]
    assert len(rows) == 1, (key, rows)
    return [c.strip() for c in rows[0].strip("|").split("|")]


def test_settings_table_documents_the_model_list():
    assert not [ln for ln in README.read_text().splitlines() if ln.startswith("| `modelId` |")]
    _, default, meaning = _row("modelIds")
    assert "Haiku 5.5, Haiku 4.5 and Sonnet 5.5" in default
    for needle in ("OWLBANK_MODELS", "one to three", "global.", "any commercial Region",
                   "Another geography's profile", "exactly the listed entries"):
        assert needle in meaning, needle
    _, default, meaning = _row("modelParams")
    assert default == "unset"
    for needle in ("thinking", "max_tokens", "text_release", "prompt_caching", "enabled:<budget>",
                   "naming the entry"):
        assert needle in meaning, needle
    _, default, meaning = _row("failoverTest")
    assert default == "`off`"
    for needle in ("OWLBANK_FAILOVER_TEST=<seconds>", "at most 30", "first", "stage=prod", "fails synth"):
        assert needle in meaning, needle


def test_readme_documents_prompt_caching_default_and_ttl():
    _, default, meaning = _row("promptCaching")
    assert default == "`on`" and "`off`" in meaning
    _, default, meaning = _row("promptCacheTtl")
    assert default == "`5m`"
    for needle in ("`1h`", "OWLBANK_PROMPT_CACHE_TTL", "2x", "1.25x", "0.1x", "tools", "system"):
        assert needle in meaning, needle
    text = README.read_text()
    admin = text.split("## The admin console", 1)[1].split("```\n", 2)[1]
    assert "promptCaching" not in admin
    assert "reseed, then the bridge" in text
