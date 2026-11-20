"""The bridge runs the agent loop itself.

There is no backend choice: the bridge task role always has exactly the scoped statements (the
listed models through their inference profiles, the one Gateway, the one Memory, the CMK) and the
container always gets the Strands settings. Each entry of ``OWLBANK_MODELS`` is ``modelParams`` over
``promptCaching`` and its model's known defaults (the list itself is tested in
``test_model_list.py``). Synth rejects invalid values, and the harness's old settings and the
agent-wide ``agentThinking``, ``agentTextRelease`` and ``agentMaxTokens`` are gone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import _synth
from owlbank_infra.config import ConfigError, InfraConfig, from_context
from owlbank_infra.grants import STRANDS_MEMORY_ACTIONS
from test_synth_and_nag import _nag_report

BASE = Path(__file__).resolve().parents[1]
BRIDGE = "OwlBankBridge"
MODEL_ACTIONS = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
#: The default model list has three entries: one statement pair each.
STRANDS_SIDS = {"StrandsInvokeModelProfile", "StrandsInvokeModelViaProfile",
                "StrandsInvokeModel2Profile", "StrandsInvokeModel2ViaProfile",
                "StrandsInvokeModel3Profile", "StrandsInvokeModel3ViaProfile", "StrandsGateway",
                "StrandsMemory", "StrandsMemoryCmk"}
#: The task role's full statement set: bank_tools, the token key, its tables, Comprehend, the loop.
#: ``CatalogScan``: the catalog reload's one Scan of the catalog table.
TASK_ROLE_SIDS = {"InvokeBankTools", "TokenUnseal", "TablesRead", "TablesReadWrite", "CatalogScan",
                  "TablesKmsViaDynamoDB", "DetectPiiForRedaction", *STRANDS_SIDS}
AGENT_ENV = {"OWLBANK_MODELS", "OWLBANK_GATEWAY_URL", "OWLBANK_MEMORY_ID"}
#: The single-model variables ``OWLBANK_MODELS`` replaced: never set any more.
SINGLE_MODEL_ENV = {"OWLBANK_MODEL_ID", "OWLBANK_MAX_TOKENS", "OWLBANK_THINKING",
                    "OWLBANK_TEXT_RELEASE", "OWLBANK_PROMPT_CACHING"}
#: Environment and context keys of the removed harness: never set any more.
HARNESS_ENV = {"OWLBANK_AGENT_BACKEND", "OWLBANK_HARNESS_ARN", "OWLBANK_HARNESS_QUALIFIER",
               "OWLBANK_HARNESS_ALLOWED_TOOLS"}
HARNESS_CONTEXT_KEYS = ("agentBackend", "harnessPrewarm", "harnessThinking", "harnessMaxTokens",
                        "harnessIdleTimeoutSeconds", "harnessMaxLifetimeSeconds", "harnessAllowedTools")


@pytest.fixture(scope="module")
def synth_caching(tmp_path_factory):
    return _synth(tmp_path_factory, "caching", {"promptCaching": "on", "modelParams": json.dumps(
        {"us.anthropic.claude-haiku-5-5": {"max_tokens": 3000}})})


def _task_role_statements(synth) -> dict[str, dict]:
    """Sid -> statement for every policy attached to the bridge's task role."""
    tpl = synth.templates[BRIDGE]
    role_ids = {lid for lid, r in tpl["Resources"].items()
                if r["Type"] == "AWS::IAM::Role" and lid.startswith("TaskRole")}
    assert len(role_ids) == 1, role_ids
    out: dict[str, dict] = {}
    for _, res in tpl["Resources"].items():
        if res["Type"] != "AWS::IAM::Policy":
            continue
        if not any(r.get("Ref") in role_ids for r in res["Properties"].get("Roles", [])):
            continue
        for stmt in res["Properties"]["PolicyDocument"]["Statement"]:
            assert stmt.get("Sid") and stmt["Sid"] not in out, stmt
            out[stmt["Sid"]] = stmt
    return out


def _container_env(synth) -> dict[str, object]:
    tpl = synth.templates[BRIDGE]
    td = next(r for r in tpl["Resources"].values() if r["Type"] == "AWS::ECS::TaskDefinition")
    (container,) = td["Properties"]["ContainerDefinitions"]
    return {e["Name"]: e["Value"] for e in container["Environment"]}


def _as_list(v):
    return v if isinstance(v, list) else [v]


# ---- settings parsing


def test_defaults():
    cfg = from_context({}, base_dir=BASE)
    assert cfg.prompt_caching == "on"
    assert cfg.prompt_caching_on
    assert cfg.prompt_cache_ttl == "5m"
    assert [(m.thinking, m.max_tokens, m.text_release) for m in cfg.models][0] == ("adaptive", 2000, "stream")


def test_prompt_cache_ttl_values():
    for value in ("5m", "1h"):
        assert from_context({"promptCacheTtl": value}, base_dir=BASE).prompt_cache_ttl == value


@pytest.mark.parametrize("bad", ["2h", "30m", "1H5", "true", "0"])
def test_invalid_prompt_cache_ttl_fails_synth(bad, tmp_path):
    from owlbank_infra.app_builder import build_app
    with pytest.raises(ConfigError, match="promptCacheTtl"):
        build_app({"promptCacheTtl": bad}, outdir=str(tmp_path))


@pytest.mark.parametrize(("key", "value"), [("promptCaching", "true"), ("promptCaching", "1"),
                                            ("promptCaching", "enabled")])
def test_invalid_values_are_rejected(key, value):
    with pytest.raises(ConfigError, match=key):
        from_context({key: value}, base_dir=BASE)


def test_build_app_rejects_junk(tmp_path):
    from owlbank_infra.app_builder import build_app
    with pytest.raises(ConfigError, match="promptCaching"):
        build_app({"promptCaching": "junk"}, outdir=str(tmp_path))


@pytest.mark.parametrize(("key", "value"), [("agentThinking", "off"), ("agentTextRelease", "message"),
                                            ("agentMaxTokens", 3000)])
def test_removed_agent_keys_fail_synth_pointing_at_model_params(key, value, tmp_path):
    from owlbank_infra.app_builder import build_app
    with pytest.raises(ConfigError, match=key) as exc:
        build_app({key: value}, outdir=str(tmp_path))
    assert "removed" in str(exc.value) and "modelParams" in str(exc.value)


# ---- Text release and thinking per entry


def _first(**params):
    return from_context({"modelParams": json.dumps({"us.anthropic.claude-haiku-5-5": params})},
                        base_dir=BASE).models[0]


def test_text_release_default_is_stream():
    assert from_context({}, base_dir=BASE).models[0].text_release == "stream"


@pytest.mark.parametrize("raw,want", [("stream", "stream"), ("message", "message"), ("MESSAGE", "message"),
                                      (" Stream ", "stream")])
def test_text_release_values(raw, want):
    assert _first(text_release=raw).text_release == want


@pytest.mark.parametrize("bad", ["on", "true", "chunk"])
def test_text_release_invalid_names_key_and_values(bad):
    with pytest.raises(ConfigError, match="text_release") as exc:
        _first(text_release=bad)
    assert "stream" in str(exc.value) and "message" in str(exc.value)


@pytest.mark.parametrize("bad", [1, True])
def test_text_release_non_string_is_rejected(bad):
    with pytest.raises(ConfigError, match="text_release"):
        _first(text_release=bad)


def test_text_release_stream_with_thinking_off_is_refused():
    for params in ({"thinking": "off"}, {"thinking": "off", "text_release": "stream"}):
        with pytest.raises(ConfigError) as exc:
            _first(**params)
        assert "text_release=stream" in str(exc.value) and "modelParams" in str(exc.value)


def test_text_release_message_with_thinking_off_is_accepted():
    m = _first(thinking="off", text_release="message")
    assert (m.thinking, m.text_release, m.max_tokens) == ("off", "message", 1000)


def test_max_tokens_default_follows_thinking():
    """2000 with reasoning on (it spends output tokens; 400 ran out once in testing), 1000 with off."""
    assert _first(thinking="off", text_release="message").max_tokens == 1000
    assert _first(thinking="default").max_tokens == 2000


def test_max_tokens_override_and_bounds():
    assert _first(max_tokens=3000).max_tokens == 3000
    assert _first(max_tokens=64, thinking="off", text_release="message").max_tokens == 64
    for bad in (10, 63, 8193, 100000, "x", 2.5, True):
        with pytest.raises(ConfigError, match="max_tokens"):
            _first(max_tokens=bad)


def test_harness_settings_are_gone():
    """The old names are removed outright, with no compatibility shim."""
    fields = set(InfraConfig.__dataclass_fields__)
    assert not {f for f in fields if "harness" in f or f == "agent_backend"}
    assert not hasattr(InfraConfig, "strands_backend") and not hasattr(InfraConfig, "harness_max_tokens")
    # An old key in the context is simply not read: same config as without it.
    for key in HARNESS_CONTEXT_KEYS:
        assert from_context({key: "off"}, base_dir=BASE) == from_context({}, base_dir=BASE), key


def test_app_builder_reads_no_harness_keys():
    import inspect

    from owlbank_infra import app_builder
    src = inspect.getsource(app_builder)
    for key in HARNESS_CONTEXT_KEYS:
        assert f'"{key}"' not in src, key
    assert '"promptCaching"' in src
    assert '"modelParams"' in src


# The Hypothesis property over arbitrary strings is in
# tests/property/test_agent_settings_props.py: this uv project has no Hypothesis, and the
# parser module (config.py) is stdlib-only, so the root project's test suite runs it.


# ---- the task role: exactly the scoped statements, in every configuration


def test_task_role_has_exactly_the_scoped_statements(synth_any):
    sts = _task_role_statements(synth_any)
    # Model failover on (the default list): the two modelstats item grants (test_model_failover_infra).
    failover = {"ModelstatsReportSamples", "ModelstatsReadState"} if synth_any.cfg.failover_on else set()
    assert set(sts) == TASK_ROLE_SIDS | failover

    profile = sts["StrandsInvokeModelProfile"]
    assert profile["Effect"] == "Allow" and sorted(_as_list(profile["Action"])) == MODEL_ACTIONS
    profile_arn = _as_list(profile["Resource"])
    assert len(profile_arn) == 1
    assert "inference-profile/us.anthropic.claude-haiku-5-5" in json.dumps(profile_arn)
    assert "Condition" not in profile

    via = sts["StrandsInvokeModelViaProfile"]
    assert sorted(_as_list(via["Action"])) == MODEL_ACTIONS
    (fm,) = _as_list(via["Resource"])
    assert json.dumps(fm).endswith('::foundation-model/anthropic.claude-haiku-5-5"]]}')
    # The Region wildcard on the one model is usable only through our profile.
    assert via["Condition"] == {"StringEquals": {"bedrock:InferenceProfileArn": profile_arn[0]}}

    gw = sts["StrandsGateway"]
    assert _as_list(gw["Action"]) == ["bedrock-agentcore:InvokeGateway"]
    (gw_res,) = _as_list(gw["Resource"])
    assert "Fn::ImportValue" in gw_res and "GatewayGatewayArn" in json.dumps(gw_res)

    mem = sts["StrandsMemory"]
    assert sorted(_as_list(mem["Action"])) == sorted(STRANDS_MEMORY_ACTIONS)
    (mem_res,) = _as_list(mem["Resource"])
    assert "Fn::ImportValue" in mem_res and "MemoryMemoryArn" in json.dumps(mem_res)

    cmk = sts["StrandsMemoryCmk"]
    assert sorted(_as_list(cmk["Action"])) == sorted(
        ["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey", "kms:DescribeKey"])
    (key_res,) = _as_list(cmk["Resource"])
    assert "Fn::ImportValue" in key_res
    assert "bedrock-agentcore." in json.dumps(cmk["Condition"]["StringEquals"]["kms:ViaService"])

    for sid in STRANDS_SIDS:
        for r in _as_list(sts[sid]["Resource"]):
            assert r != "*", sid
        for a in _as_list(sts[sid]["Action"]):
            assert "*" not in a, sid
        assert sts[sid]["Effect"] == "Allow"


def test_task_role_has_no_harness_grants(synth_any):
    actions = {a for s in _task_role_statements(synth_any).values() for a in _as_list(s["Action"])}
    assert not actions & {"bedrock-agentcore:InvokeHarness", "bedrock-agentcore:InvokeAgentRuntime",
                          "bedrock-agentcore:InvokeAgentRuntimeCommand"}


# ---- container environment


def _models(env) -> list[dict]:
    return json.loads(env["OWLBANK_MODELS"])


def test_env_defaults(synth_any):
    env = _container_env(synth_any)
    assert set(env) >= AGENT_ENV
    assert not HARNESS_ENV & set(env)
    assert not SINGLE_MODEL_ENV & set(env)
    first = _models(env)[0]
    assert first == {"id": "us.anthropic.claude-haiku-5-5", "thinking": "adaptive", "max_tokens": 2000,
                     "text_release": "stream", "prompt_caching": True}
    assert all(m["prompt_caching"] is True for m in _models(env))
    assert env["OWLBANK_PROMPT_CACHE_TTL"] == "5m"
    assert "GatewayGatewayUrl" in json.dumps(env["OWLBANK_GATEWAY_URL"])
    assert "MemoryMemoryId" in json.dumps(env["OWLBANK_MEMORY_ID"])


def test_env_caching_off_keeps_every_entry_off(tmp_path_factory):
    """``promptCaching=off`` is still honoured; ``modelParams`` still wins per entry."""
    synth = _synth(tmp_path_factory, "caching-off", {"promptCaching": "off", "modelParams": json.dumps(
        {"us.anthropic.claude-sonnet-5-5": {"prompt_caching": "on"}})})
    assert [m["prompt_caching"] for m in _models(_container_env(synth))] == [False, False, True]


def test_env_prompt_cache_ttl_1h_on_fargate(tmp_path_factory):
    synth = _synth(tmp_path_factory, "ttl-1h", {"promptCacheTtl": "1h"})
    assert _container_env(synth)["OWLBANK_PROMPT_CACHE_TTL"] == "1h"


def _runtime_env(synth) -> dict:
    tpl = synth.templates[BRIDGE]
    rt = next(r for r in tpl["Resources"].values() if r["Type"] == "AWS::BedrockAgentCore::Runtime")
    return rt["Properties"]["EnvironmentVariables"]


def test_env_prompt_cache_ttl_on_agentcore_runtime(synth_agentcore_vpc_dev):
    """Both hostings use ``strands_env``: the default is written on the Runtime too."""
    env = _runtime_env(synth_agentcore_vpc_dev)
    assert env["OWLBANK_PROMPT_CACHE_TTL"] == "5m"
    assert all(m["prompt_caching"] is True for m in _models(env))


def test_env_prompt_cache_ttl_1h_on_agentcore_runtime(tmp_path_factory):
    from test_bridge_agentcore import _ctx, _quiet_synth
    synth = _quiet_synth(tmp_path_factory, "ac-ttl-1h", {**_ctx("vpc", "dev"), "promptCacheTtl": "1h"})
    assert _runtime_env(synth)["OWLBANK_PROMPT_CACHE_TTL"] == "1h"


def test_env_thinking_off(synth_thinking_off):
    """``modelParams`` on the first entry; the fallbacks keep their defaults."""
    entry, *fallbacks = _models(_container_env(synth_thinking_off))
    assert (entry["thinking"], entry["max_tokens"], entry["text_release"]) == ("off", 1000, "message")
    assert [m["thinking"] for m in fallbacks] == ["off", "adaptive"]


def test_env_text_release_message(tmp_path_factory):
    from conftest import _synth
    params = {"modelParams": json.dumps({"us.anthropic.claude-haiku-5-5": {"text_release": "message"}})}
    env = _container_env(_synth(tmp_path_factory, "text-message", params))
    assert [m["text_release"] for m in _models(env)] == ["message", "message", "stream"]


def test_env_text_release_agentcore_runtime(synth_agentcore_vpc_dev):
    """The AgentCore Runtime gets the same variable (both hostings use ``bridge_base_env``)."""
    tpl = synth_agentcore_vpc_dev.templates[BRIDGE]
    rt = next(r for r in tpl["Resources"].values() if r["Type"] == "AWS::BedrockAgentCore::Runtime")
    assert _models(rt["Properties"]["EnvironmentVariables"])[0]["text_release"] == "stream"


def test_env_caching_and_max_tokens(synth_caching):
    """``promptCaching`` applies to every entry, ``modelParams`` ``max_tokens`` to its entry."""
    models = _models(_container_env(synth_caching))
    assert [(m["prompt_caching"], m["max_tokens"]) for m in models] == [(True, 3000), (True, 1000), (True, 2000)]
    assert models[0]["thinking"] == "adaptive"


def test_cdk_nag_clean_with_caching(synth_caching):
    report = _nag_report(synth_caching)
    if report is not None:
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [v["ruleName"] for v in errors]
        assert report["conclusion"] == "success"


# ---- outputs


def test_agent_stack_outputs_gateway_url_and_memory_id(synth_any):
    outputs = synth_any.templates["OwlBankAgent"]["Outputs"]
    assert outputs["GatewayUrl"]["Value"] == {"Fn::GetAtt": ["Gateway", "GatewayUrl"]}
    assert outputs["MemoryId"]["Value"] == {"Fn::GetAtt": ["Memory", "MemoryId"]}
