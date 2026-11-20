"""The agent settings, Strands start-up and shutdown, the client's
blank-text rule, the OpenTelemetry detach-noise filter, and the start-up prewarm. Offline: TAC's
configuration fetch, the Gateway MCP client and the prewarm's AWS clients are faked."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tac_bridge import agent_client_strands as acs
from tac_bridge.agent_client import AgentError, ToolCall, ToolResult, TurnEnd
from tac_bridge.agent_client_strands import OtelDetachNoise, StrandsAgentClient, install_otel_detach_filter
from tac_bridge.app import build_app_from_env, strands_settings_from_env, trait_settings_from_env
from tac_bridge.bridge import BridgeConfig
from owlbank.store import MemoryStore
from scripts import seed
from tests.logutil import lines
from tests.strands_fakes import (
    FakeGatewayProvider,
    FakeMemoryData,
    FakeModel,
    collect,
    light_memory_client,
    make_memory_client,
    settings,
    text_msg,
    tool_msg,
)

SID = "CA" + "7" * 32
PRIMARY = {"id": "us.anthropic.claude-haiku-5-5", "thinking": "adaptive", "max_tokens": 2000,
           "text_release": "stream", "prompt_caching": False}
SECOND = {"id": "us.anthropic.claude-haiku-4-5", "thinking": "off", "max_tokens": 400,
          "text_release": "message", "prompt_caching": False}
THIRD = {"id": "us.anthropic.claude-sonnet-5-5", "thinking": "between_tools", "max_tokens": 2000,
         "text_release": "stream", "prompt_caching": True}


def models_env(*entries: dict[str, Any]) -> str:
    return json.dumps(list(entries or [PRIMARY]), separators=(",", ":"))


STRANDS_ENV = {
    "OWLBANK_MODELS": models_env(PRIMARY),
    "OWLBANK_GATEWAY_URL": "https://gw.example/mcp",
    "OWLBANK_MEMORY_ID": "mem-1",
    "AWS_REGION": "us-east-1",
}


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def events(caplog: Any, logger: str, name: str) -> list[dict[str, Any]]:
    return [f for f in lines(caplog.records, logger) if f.get("event") == name]


# --------------------------------------------------------------------------- the agent settings


def test_strands_settings_from_env() -> None:
    s = strands_settings_from_env(STRANDS_ENV)
    (m,) = s.models
    assert (m.id, s.region, m.max_tokens, m.thinking, m.prompt_caching, s.gateway_url, s.memory_id) == (
        "us.anthropic.claude-haiku-5-5", "us-east-1", 2000, "adaptive", False, "https://gw.example/mcp", "mem-1")
    assert m.text_release == "stream" and s.failover_test_s is None
    cached = strands_settings_from_env({**STRANDS_ENV, "OWLBANK_MODELS": models_env({**PRIMARY, "prompt_caching": True})})
    assert cached.primary.prompt_caching is True


def test_the_model_list_is_read_in_order_with_each_entrys_parameters() -> None:
    """Model failover: the infra ``OWLBANK_MODELS`` shape, every entry kept as given."""
    s = strands_settings_from_env({**STRANDS_ENV, "OWLBANK_MODELS": models_env(PRIMARY, SECOND, THIRD)})
    assert [(m.id, m.thinking, m.max_tokens, m.text_release, m.prompt_caching) for m in s.models] == [
        tuple(e[k] for k in ("id", "thinking", "max_tokens", "text_release", "prompt_caching"))
        for e in (PRIMARY, SECOND, THIRD)]
    budget = strands_settings_from_env({**STRANDS_ENV, "OWLBANK_MODELS": models_env(
        {**SECOND, "thinking": "enabled:1024", "max_tokens": 2000})})
    assert budget.primary.thinking == "enabled:1024"


def test_the_infra_env_value_is_read_unchanged() -> None:
    """The exact string ``bridge_stack.models_env_value`` writes (compact JSON, ``as_env`` keys)."""
    infra = json.dumps([PRIMARY, SECOND], separators=(",", ":"))
    s = strands_settings_from_env({**STRANDS_ENV, "OWLBANK_MODELS": infra})
    assert s.model_ids == (PRIMARY["id"], SECOND["id"])


@pytest.mark.parametrize("name", ["OWLBANK_MODELS", "OWLBANK_GATEWAY_URL", "OWLBANK_MEMORY_ID", "AWS_REGION"])
def test_a_missing_strands_variable_is_named(name: str) -> None:
    env = {k: v for k, v in STRANDS_ENV.items() if k != name}
    with pytest.raises(RuntimeError, match=f"^{name} is required$"):
        strands_settings_from_env(env)
    with pytest.raises(RuntimeError, match=f"^{name} is required"):
        strands_settings_from_env({**env, name: "  "})


def test_the_single_model_variables_are_gone() -> None:
    """Model failover: ``OWLBANK_MODELS`` replaced ``OWLBANK_MODEL_ID`` and its companions."""
    env = {k: v for k, v in STRANDS_ENV.items() if k != "OWLBANK_MODELS"}
    with pytest.raises(RuntimeError, match="^OWLBANK_MODELS is required$"):
        strands_settings_from_env({**env, "OWLBANK_MODEL_ID": "us.anthropic.claude-haiku-5-5",
                                   "OWLBANK_MAX_TOKENS": "2000", "OWLBANK_THINKING": "adaptive",
                                   "OWLBANK_TEXT_RELEASE": "stream", "OWLBANK_PROMPT_CACHING": "off"})


SECRET_ID = "us.anthropic.claude-secret-9-9"


@pytest.mark.parametrize(("value", "message"), [
    ("not json", r"^OWLBANK_MODELS must be a JSON list of model entries$"),
    ("{}", r"^OWLBANK_MODELS must be a JSON list of 1 to 3 model entries$"),
    ("[]", r"^OWLBANK_MODELS must be a JSON list of 1 to 3 model entries$"),
    (models_env(PRIMARY, SECOND, THIRD, {**PRIMARY, "id": SECRET_ID}), r"^OWLBANK_MODELS must be a JSON list of 1 to 3"),
    (models_env(PRIMARY, {"id": SECRET_ID}), r"^OWLBANK_MODELS\[1\] must be an object with exactly the keys"),
    (models_env({**PRIMARY, "extra": SECRET_ID}), r"^OWLBANK_MODELS\[0\] must be an object with exactly"),
    (models_env({**PRIMARY, "id": " "}), r"^OWLBANK_MODELS\[0\]: id must be"),
    (models_env({**PRIMARY, "id": SECRET_ID, "max_tokens": "2000"}), r"^OWLBANK_MODELS\[0\]: max_tokens must be"),
    (models_env({**PRIMARY, "id": SECRET_ID, "max_tokens": True}), r"^OWLBANK_MODELS\[0\]: max_tokens must be"),
    (models_env({**PRIMARY, "id": SECRET_ID, "max_tokens": 0}), r"^OWLBANK_MODELS\[0\]: max_tokens must be"),
    (models_env({**PRIMARY, "id": SECRET_ID, "thinking": "deep"}), r"^OWLBANK_MODELS\[0\]: thinking must be one of"),
    (models_env({**PRIMARY, "id": SECRET_ID, "thinking": "enabled:999"}), r"^OWLBANK_MODELS\[0\]: the thinking budget"),
    (models_env({**PRIMARY, "id": SECRET_ID, "thinking": "enabled:4000"}), r"^OWLBANK_MODELS\[0\]: the thinking budget must be below"),
    (models_env({**PRIMARY, "id": SECRET_ID, "text_release": "chunks"}), r"^OWLBANK_MODELS\[0\]: text_release must be one of"),
    (models_env({**PRIMARY, "id": SECRET_ID, "prompt_caching": "on"}), r"^OWLBANK_MODELS\[0\]: prompt_caching must be"),
    (models_env(PRIMARY, {**PRIMARY}), r"^OWLBANK_MODELS must not list a model twice$"),
])
def test_an_invalid_model_list_is_named_without_its_value(value: str, message: str) -> None:
    with pytest.raises(RuntimeError, match=message) as err:
        strands_settings_from_env({**STRANDS_ENV, "OWLBANK_MODELS": value})
    assert SECRET_ID not in str(err.value)


@pytest.mark.parametrize(("name", "value"), [("OWLBANK_GATEWAY_URL", "http://gw.example/mcp")])
def test_an_invalid_strands_variable_is_named_without_its_value(name: str, value: str) -> None:
    with pytest.raises(RuntimeError, match=f"^{name} must be") as err:
        strands_settings_from_env({**STRANDS_ENV, name: value})
    assert value not in str(err.value)


@pytest.mark.parametrize(("value", "ttl"), [(None, "5m"), ("", "5m"), ("  ", "5m"), ("5m", "5m"),
                                          ("1h", "1h"), ("1H", "1h"), (" 5M ", "5m")])
def test_the_prompt_cache_ttl_is_read(value: str | None, ttl: str) -> None:
    """Agent prompt caching: absent or empty is ``5m``; ``5m`` or ``1h`` in any case."""
    env = dict(STRANDS_ENV) if value is None else {**STRANDS_ENV, "OWLBANK_PROMPT_CACHE_TTL": value}
    assert strands_settings_from_env(env).prompt_cache_ttl == ttl


@pytest.mark.parametrize("value", ["2h", "300s", "15", "on", "1h " + SECRET_ID, SECRET_ID])
def test_an_invalid_prompt_cache_ttl_is_named_without_its_value(value: str) -> None:
    with pytest.raises(RuntimeError, match="^OWLBANK_PROMPT_CACHE_TTL must be 5m or 1h$") as err:
        strands_settings_from_env({**STRANDS_ENV, "OWLBANK_PROMPT_CACHE_TTL": value})
    assert value not in str(err.value) and SECRET_ID not in str(err.value)


def test_the_failover_test_delay_is_read_and_refused_at_prod() -> None:
    """``OWLBANK_FAILOVER_TEST`` (seconds) is the first entry's test delay; at
    ``OWLBANK_STAGE=prod`` the start fails naming the variable."""
    s = strands_settings_from_env({**STRANDS_ENV, "OWLBANK_FAILOVER_TEST": "2.5", "OWLBANK_STAGE": "dev"})
    assert s.failover_test_s == 2.5 and s.stage == "dev"
    with pytest.raises(RuntimeError, match="^OWLBANK_FAILOVER_TEST: refused at stage=prod$"):
        strands_settings_from_env({**STRANDS_ENV, "OWLBANK_FAILOVER_TEST": "2.5", "OWLBANK_STAGE": "prod"})
    with pytest.raises(RuntimeError, match="^OWLBANK_FAILOVER_TEST: must be") as err:
        strands_settings_from_env({**STRANDS_ENV, "OWLBANK_FAILOVER_TEST": "1234567"})
    assert "1234567" not in str(err.value)


# --------------------------------------------------------------------------- the app starts


_TWILIO_ENV = {
    "TWILIO_ACCOUNT_SID": "AC" + "0" * 32, "TWILIO_AUTH_TOKEN": "test-auth-token-not-a-secret",
    "TWILIO_API_KEY": "SK" + "0" * 32, "TWILIO_API_SECRET": "test-api-secret",
    "TWILIO_PHONE_NUMBER": "+12025550199", "TWILIO_CONVERSATION_CONFIGURATION_ID": "conv_configuration_x",
    "TWILIO_VOICE_PUBLIC_DOMAIN": "relay.owlbank.example",
}
_BRIDGE_ENV = {
    **_TWILIO_ENV, "AWS_REGION": "us-east-1", "OWLBANK_BANK_TOOLS_FUNCTION": "arn:aws:lambda:us-east-1:1:function:b",
    "OWLBANK_TOKEN_KMS_KEY": "arn:aws:kms:us-east-1:1:key/k",
}


@pytest.fixture
def offline_start(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """TAC's configuration fetch, the Gateway MCP client and the client prewarm, faked."""
    seen: dict[str, Any] = {"gateway": [], "prewarmed": 0, "closed": 0}
    for k, v in _TWILIO_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    # boto3 resolves credentials when a client is built: dummy ones, no instance-metadata lookup.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setattr("tac.context.conversation.ConversationClient.get_configuration",
                        lambda *_a, **_k: SimpleNamespace(memory_store_id="mem_store_test"))
    store = MemoryStore()  # instead of DynamoDB (the policy is read at start-up)
    root = Path(__file__).resolve().parents[2] / "data"
    seed.apply_plan(seed.build_plan(seed.load_json(root / "fixtures.json"), seed.load_json(root / "policy.json"),
                                    seed.load_json(root / "prompt_catalog.json"), stage="test"), store)
    monkeypatch.setattr("owlbank.store.DynamoStore.from_settings", classmethod(lambda _cls, _s: store))

    def gateway(url: str, region: str) -> FakeGatewayProvider:
        seen["gateway"].append((url, region))
        provider = FakeGatewayProvider()
        seen["provider"] = provider
        return provider

    def prewarm(self: StrandsAgentClient) -> None:
        seen["prewarmed"] += 1

    real_aclose = StrandsAgentClient.aclose

    async def aclose(self: StrandsAgentClient) -> None:
        seen["closed"] += 1
        await real_aclose(self)

    monkeypatch.setattr(acs, "make_gateway_tool_provider", gateway)
    monkeypatch.setattr(StrandsAgentClient, "prewarm", prewarm)
    monkeypatch.setattr(StrandsAgentClient, "aclose", aclose)
    return seen


def _health(app: Any) -> Any:
    with TestClient(app) as client:  # runs the lifespan: start-up, then shutdown on exit
        return client.get("/health").json()


def test_the_app_starts_with_the_strands_client(offline_start: dict[str, Any], caplog: Any) -> None:
    caplog.set_level(logging.INFO, logger="tac_bridge.app")
    app = build_app_from_env({**_BRIDGE_ENV, **STRANDS_ENV})
    agent = app.state.bridge._agent
    assert isinstance(agent, StrandsAgentClient)
    assert agent.settings == strands_settings_from_env(STRANDS_ENV)
    # Started once, prewarmed once, both before the app was returned (so before uvicorn listens).
    assert offline_start["gateway"] == [("https://gw.example/mcp", "us-east-1")]
    assert offline_start["prewarmed"] == 1
    assert agent._gateway is offline_start["provider"]
    assert _health(app) == {"ok": True}
    assert offline_start["closed"] == 1 and offline_start["provider"].closed  # aclose on shutdown
    (ln,) = events(caplog, "tac_bridge.app", "agent_settings")
    assert (ln["thinking"], ln["prompt_caching"], ln["prompt_cache_ttl"]) == ("adaptive", False, "5m")


def test_the_prompt_cache_ttl_is_logged_and_a_bad_one_fails_the_start(offline_start: dict[str, Any],
                                                                      caplog: Any) -> None:
    caplog.set_level(logging.INFO, logger="tac_bridge.app")
    app = build_app_from_env({**_BRIDGE_ENV, **STRANDS_ENV, "OWLBANK_PROMPT_CACHE_TTL": "1h"})
    assert app.state.bridge._agent.settings.prompt_cache_ttl == "1h"
    (ln,) = events(caplog, "tac_bridge.app", "agent_settings")
    assert ln["prompt_cache_ttl"] == "1h"
    with pytest.raises(RuntimeError, match="^OWLBANK_PROMPT_CACHE_TTL must be 5m or 1h$"):
        build_app_from_env({**_BRIDGE_ENV, **STRANDS_ENV, "OWLBANK_PROMPT_CACHE_TTL": "1d"})


@pytest.mark.parametrize("mode", ["stream", "message"])
def test_each_text_release_mode_starts_and_is_logged(offline_start: dict[str, Any], caplog: Any, mode: str) -> None:
    caplog.set_level(logging.INFO, logger="tac_bridge.app")
    env = {**STRANDS_ENV, "OWLBANK_MODELS": models_env({**PRIMARY, "text_release": mode.upper()})}  # any case
    assert strands_settings_from_env(env).primary.text_release == mode
    app = build_app_from_env({**_BRIDGE_ENV, **env})
    assert app.state.bridge._agent.settings.primary.text_release == mode
    (ln,) = events(caplog, "tac_bridge.app", "agent_text_release")
    assert ln["mode"] == mode


def test_a_bad_text_release_names_the_allowed_values() -> None:
    with pytest.raises(RuntimeError, match=r"^OWLBANK_MODELS\[0\]: text_release must be one of \['stream', 'message'\]$"):
        strands_settings_from_env({**STRANDS_ENV, "OWLBANK_MODELS": models_env({**PRIMARY, "text_release": "chunks"})})


@pytest.mark.parametrize("mode", ["stream", "STREAM"])
def test_stream_release_with_thinking_off_names_the_entry(mode: str) -> None:
    # A RuntimeError naming the entry (as for any bad value), not
    # ModelEntry's ValueError; message mode with thinking off still starts.
    bad = {**SECOND, "text_release": mode}
    with pytest.raises(RuntimeError, match=r"^OWLBANK_MODELS\[1\]: text_release=stream needs "
                                           "thinking adaptive or default: .*set text_release=message"):
        strands_settings_from_env({**STRANDS_ENV, "OWLBANK_MODELS": models_env(PRIMARY, bad)})
    s = strands_settings_from_env({**STRANDS_ENV, "OWLBANK_MODELS": models_env(PRIMARY, SECOND)})
    assert (s.models[1].text_release, s.models[1].thinking) == ("message", "off")


def test_the_app_reads_the_failover_policy_per_turn(offline_start: dict[str, Any]) -> None:
    """The client reads the live ``model_failover`` section through the policy."""
    app = build_app_from_env({**_BRIDGE_ENV, **STRANDS_ENV, "OWLBANK_MODELS": models_env(PRIMARY, SECOND)})
    agent = app.state.bridge._agent
    cfg = agent._failover_policy()
    assert cfg.settings_for(agent.settings.model_ids).entries[0].model_id == PRIMARY["id"]


def test_removed_backend_settings_are_ignored(offline_start: dict[str, Any]) -> None:
    """No backend choice; stale harness variables change nothing."""
    app = build_app_from_env({**_BRIDGE_ENV, **STRANDS_ENV, "OWLBANK_AGENT_BACKEND": "harness",
                              "OWLBANK_HARNESS_ARN": "arn:aws:bedrock-agentcore:us-east-1:1:harness/h"})
    assert isinstance(app.state.bridge._agent, StrandsAgentClient)


def test_the_agent_settings_are_required(offline_start: dict[str, Any]) -> None:
    with pytest.raises(RuntimeError, match="^OWLBANK_MODELS is required$"):
        build_app_from_env(dict(_BRIDGE_ENV))


def test_a_missing_strands_variable_fails_the_start(offline_start: dict[str, Any]) -> None:
    env = {k: v for k, v in {**_BRIDGE_ENV, **STRANDS_ENV}.items() if k != "OWLBANK_GATEWAY_URL"}
    with pytest.raises(RuntimeError, match="OWLBANK_GATEWAY_URL is required"):
        build_app_from_env(env)
    assert offline_start["gateway"] == []  # failed before anything started


def test_a_failing_shutdown_hook_is_logged(offline_start: dict[str, Any], caplog: Any) -> None:
    async def broken(self: StrandsAgentClient) -> None:
        raise RuntimeError("stop failed")

    with patch.object(StrandsAgentClient, "aclose", broken):
        app = build_app_from_env({**_BRIDGE_ENV, **STRANDS_ENV})  # the hook is bound at start-up
    caplog.set_level(logging.DEBUG, logger="tac_bridge.app")
    assert _health(app) == {"ok": True}  # shutdown doesn't raise
    rec = [r for r in caplog.records if getattr(r, "operation", None) == "shutdown"]
    assert rec and rec[0].decision == "ignored_best_effort"


# --------------------------------------------------------------------------- prewarm


class _Clients:
    region_name = "us-east-1"

    def __init__(self) -> None:
        self.made: list[tuple[str, Any]] = []

    def client(self, service: str, region_name: str | None = None, config: Any = None) -> Any:
        self.made.append((service, getattr(config, "user_agent_extra", None)))
        return object()


def test_memory_factory_prewarm_builds_the_clients_the_session_manager_uses(monkeypatch: Any) -> None:
    light_memory_client(monkeypatch)
    built: list[str | None] = []
    real = acs._acm.MemoryClient

    class Recording(real):  # type: ignore[misc, valid-type]
        def __init__(self, region_name: str | None = None, **kw: Any) -> None:
            built.append(region_name)
            super().__init__(region_name=region_name, **kw)

    monkeypatch.setattr(acs._acm, "MemoryClient", Recording)
    clients = _Clients()
    factory = acs.memory_session_manager_factory("mem-1", "us-east-1", boto_session=clients)
    factory.prewarm()
    assert built == ["us-east-1"]
    assert clients.made == [("bedrock-agentcore-control", "strands-agents"), ("bedrock-agentcore", "strands-agents")]


def test_shared_clients_are_reused_after_the_prewarm() -> None:
    made: list[str] = []

    class Session:
        def client(self, service: str, region_name: str | None = None, config: Any = None) -> Any:
            made.append(service)
            return SimpleNamespace(service=service)

    shared = acs._SharedBotoClients("us-east-1", session_factory=Session)
    factory = acs.MemorySessionManagerFactory("mem-1", "us-east-1", boto_session=shared)
    with patch.object(acs._acm, "MemoryClient", lambda **_k: None):
        factory.prewarm()
    first = shared.client("bedrock-agentcore")
    assert shared.client("bedrock-agentcore") is first
    assert made == ["bedrock-agentcore-control", "bedrock-agentcore"]  # nothing new after the prewarm


def test_client_prewarm_builds_the_shared_model_and_the_memory_clients(monkeypatch: Any, caplog: Any) -> None:
    models: list[Any] = []

    def model_factory(s: Any) -> Any:
        models.append(FakeModel())
        return models[-1]

    warmed: list[int] = []

    class Factory:
        def __call__(self, session_id: str, mode: str) -> Any:  # pragma: no cover - not built here
            raise AssertionError

        def prewarm(self) -> None:
            warmed.append(1)

    monkeypatch.setattr(acs, "default_model_factory", model_factory)
    caplog.set_level(logging.INFO, logger="tac_bridge.agent_client_strands")
    client = StrandsAgentClient(settings(memory_id="mem-1"), session_manager_factory=Factory())
    client.prewarm()
    assert len(models) == 1 and warmed == [1]
    assert client._model(SID) is models[0]  # the first call reuses it
    assert len(events(caplog, "tac_bridge.agent_client_strands", "strands_prewarmed")) == 1


def test_a_failed_prewarm_is_logged_not_raised(monkeypatch: Any, caplog: Any) -> None:
    def broken(s: Any) -> Any:
        raise RuntimeError("no region")

    monkeypatch.setattr(acs, "default_model_factory", broken)
    caplog.set_level(logging.DEBUG, logger="tac_bridge.agent_client_strands")
    StrandsAgentClient(settings()).prewarm()
    rec = [r for r in caplog.records if getattr(r, "operation", None) == "prewarm"]
    assert rec and rec[0].decision == "continue_cold"


# --------------------------------------------------------------------------- blank caller text


def test_blank_text_ends_the_turn_without_a_model_call() -> None:
    model = FakeModel([text_msg("Hello.")])
    data = FakeMemoryData()

    async def go() -> list[Any]:
        client = make_memory_client(model, data)
        await client.begin_session(SID, history="new")
        out = await collect(client.turn(SID, user_text="  \n\t", tool_results=[], system_prompts={"haiku-5-5": "p"}))
        out += await collect(client.turn(SID, user_text="Hi", tool_results=[], system_prompts={"haiku-5-5": "p"}))
        return out

    with pytest.MonkeyPatch.context() as mp:
        light_memory_client(mp)
        out = run(go())
    assert out[0] == TurnEnd("end_turn", None)
    assert len(model.calls) == 1  # only the real words reached the model
    assert [m["content"] for m in model.calls[0].messages] == [[{"text": "Hi"}]]
    texts = [c.get("text") for m in data.stored_messages(SID) for c in m["content"]]
    assert all(t is None or t.strip() for t in texts)  # no whitespace-only block stored


def test_blank_text_with_a_pending_tool_answers_it_without_the_text() -> None:
    model = FakeModel([tool_msg(("tu-1", "list_recent_transactions", {"account": "checking"})),
                       text_msg("Never mind then.")])

    async def go() -> tuple[list[Any], list[Any]]:
        client = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: model)
        first = await collect(client.turn(SID, user_text="recent", tool_results=[], system_prompts={"haiku-5-5": "p"}))
        second = await collect(client.turn(SID, user_text="   ", tool_results=[], system_prompts={"haiku-5-5": "p"}))
        return first, second

    first, second = run(go())
    assert isinstance(first[0], ToolCall) and first[-1].stop_reason == "tool_use"
    assert second[-1] == TurnEnd("end_turn", second[-1].usage)
    last = model.calls[-1].messages
    # The pending call got the cancelled result; no blank text message followed it.
    assert last[-1]["role"] == "user" and "toolResult" in last[-1]["content"][0]
    assert last[-1]["content"][0]["toolResult"]["toolUseId"] == "tu-1"
    assert not any(c.get("text") == "   " for m in last for c in m["content"])


def test_blank_text_alongside_results_is_dropped() -> None:
    model = FakeModel([tool_msg(("tu-1", "list_recent_transactions", {"account": "checking"})),
                       text_msg("Nothing recent.")])

    async def go() -> list[Any]:
        client = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: model)
        await collect(client.turn(SID, user_text="recent", tool_results=[], system_prompts={"haiku-5-5": "p"}))
        return await collect(client.turn(SID, user_text=" ", system_prompts={"haiku-5-5": "p"},
                                         tool_results=[ToolResult("tu-1", {"transactions": []})]))

    out = run(go())
    assert out[-1].stop_reason == "end_turn"
    assert len(model.calls) == 2 and model.calls[-1].messages[-1]["content"][0]["toolResult"]["status"] == "success"


# --------------------------------------------------------------------------- OpenTelemetry detach noise


def _record(name: str, msg: str, level: int = logging.ERROR) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, msg, None, None)


def test_the_filter_drops_exactly_the_detach_error() -> None:
    f = OtelDetachNoise()
    assert f.filter(_record("opentelemetry.context", "Failed to detach context")) is False
    assert f.filter(_record("opentelemetry.context", "Failed to detach context: other")) is True
    assert f.filter(_record("opentelemetry.context", "Failed to load context: %s")) is True
    assert f.filter(_record("tac_bridge.bridge", "Failed to detach context")) is True
    assert f.filter(_record("opentelemetry.trace", "Failed to detach context")) is True


def test_the_filter_is_installed_once_and_silences_the_real_logger(caplog: Any) -> None:
    install_otel_detach_filter()
    install_otel_detach_filter()
    logger = logging.getLogger("opentelemetry.context")
    assert sum(isinstance(f, OtelDetachNoise) for f in logger.filters) == 1
    caplog.set_level(logging.DEBUG, logger="opentelemetry.context")
    try:
        raise ValueError("<Token> was created in a different Context")
    except ValueError:
        logger.exception("Failed to detach context")
        logger.exception("something else broke")
    assert [r.getMessage() for r in caplog.records if r.name == "opentelemetry.context"] == ["something else broke"]


def test_otel_detach_on_a_closed_strands_stream_is_not_logged(caplog: Any, monkeypatch: Any) -> None:
    """Barge-in: the bridge reads the stream in its timer-race task and closes it from its own, so
    Strands' span context is detached in another context. OpenTelemetry's ERROR is dropped."""
    model = FakeModel([text_msg("A long answer that the caller interrupts.", chunks=8)], delay=0.01)
    caplog.set_level(logging.DEBUG)
    dropped: list[str] = []
    real = OtelDetachNoise.filter

    def counting(self: OtelDetachNoise, record: logging.LogRecord) -> bool:
        keep = real(self, record)
        if not keep:
            dropped.append(record.getMessage())
        return keep

    monkeypatch.setattr(OtelDetachNoise, "filter", counting)

    async def go() -> Any:
        client = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: model)
        gen = client.turn(SID, user_text="hi", tool_results=[], system_prompts={"haiku-5-5": "p"})
        first = await asyncio.create_task(anext(gen))  # read in another task (the bridge's timer race)
        await gen.aclose()  # closed from this one (barge-in)
        return first

    assert run(go()).text == "A long"
    assert dropped and set(dropped) == {"Failed to detach context"}  # it happened, and was dropped
    assert not [r for r in caplog.records if r.name == "opentelemetry.context"]


# --------------------------------------------------------------------------- the bridge's retry of a failed hop


def _throttled() -> AgentError:
    return AgentError("throttled", retryable=True, code="ThrottlingException")


def test_a_retried_result_hop_continues_from_the_history() -> None:
    """The model call after a resume failed: the results are in the history and nothing is pending.
    The bridge sends the same results again; the client continues instead of refusing them."""
    model = FakeModel([tool_msg(("tu-1", "list_recent_transactions", {"account": "checking"})),
                       _throttled(), text_msg("Nothing recent.")])
    data = FakeMemoryData()
    results = [ToolResult("tu-1", {"transactions": []})]

    async def go() -> list[Any]:
        client = make_memory_client(model, data)
        await client.begin_session(SID, history="new")
        await collect(client.turn(SID, user_text="recent", tool_results=[], system_prompts={"haiku-5-5": "p"}))
        with pytest.raises(AgentError) as err:
            await collect(client.turn(SID, user_text=None, tool_results=results, system_prompts={"haiku-5-5": "p"}))
        assert err.value.code == "ThrottlingException" and err.value.retryable
        return await collect(client.turn(SID, user_text=None, tool_results=results, system_prompts={"haiku-5-5": "p"}))

    with pytest.MonkeyPatch.context() as mp:
        light_memory_client(mp)
        out = run(go())
    assert out[-1].stop_reason == "end_turn" and out[0].text == "Nothing recent."  # type: ignore[union-attr]
    shapes = [[next(iter(b)) for b in m["content"]] for m in data.stored_messages(SID)]
    assert shapes == [["text"], ["toolUse"], ["toolResult"], ["text"]]  # the result once


def test_a_retried_text_hop_sends_the_words_once() -> None:
    model = FakeModel([_throttled(), text_msg("Hi.")])

    async def go() -> list[Any]:
        client = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: model)
        with pytest.raises(AgentError):
            await collect(client.turn(SID, user_text="hello", tool_results=[], system_prompts={"haiku-5-5": "p"}))
        return await collect(client.turn(SID, user_text="hello", tool_results=[], system_prompts={"haiku-5-5": "p"}))

    assert run(go())[0].text == "Hi."  # type: ignore[union-attr]
    assert [m["content"] for m in model.calls[-1].messages] == [[{"text": "hello"}]]


def test_new_words_after_a_failed_turn_are_added() -> None:
    """Different words are the caller's next turn, not a retry (Bedrock accepts two user messages)."""
    model = FakeModel([_throttled(), text_msg("Sure.")])

    async def go() -> None:
        client = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: model)
        with pytest.raises(AgentError):
            await collect(client.turn(SID, user_text="hello", tool_results=[], system_prompts={"haiku-5-5": "p"}))
        await collect(client.turn(SID, user_text="are you there?", tool_results=[], system_prompts={"haiku-5-5": "p"}))

    run(go())
    assert [m["content"] for m in model.calls[-1].messages] == [[{"text": "hello"}], [{"text": "are you there?"}]]


def test_results_for_a_tool_use_never_answered_are_still_refused() -> None:
    model = FakeModel([_throttled()])

    async def go() -> None:
        client = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: model)
        with pytest.raises(AgentError):
            await collect(client.turn(SID, user_text="hello", tool_results=[], system_prompts={"haiku-5-5": "p"}))
        await collect(client.turn(SID, user_text=None, tool_results=[ToolResult("tu-x", {})], system_prompts={"haiku-5-5": "p"}))

    with pytest.raises(AgentError) as err:
        run(go())
    assert err.value.code == "unknown_tool_use"


# --------------------------------------------------------------------------- Memory trait settings


def test_trait_settings_default_when_unset_or_blank() -> None:
    for env in ({}, {"OWLBANK_MEMORY_TRAIT_GROUPS": " ", "OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS": ""}):
        assert trait_settings_from_env(env) == {}
    cfg = BridgeConfig()
    assert (cfg.memory_trait_groups, cfg.web_activity_max_age_hours) == (("Contact", "WebActivity"), 24)


def test_trait_settings_are_parsed() -> None:
    got = trait_settings_from_env({"OWLBANK_MEMORY_TRAIT_GROUPS": "Contact, Other.1",
                                   "OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS": "8760"})
    assert got == {"memory_trait_groups": ("Contact", "Other.1"), "web_activity_max_age_hours": 8760}


@pytest.mark.parametrize(("name", "value"), [
    ("OWLBANK_MEMORY_TRAIT_GROUPS", "A,A"), ("OWLBANK_MEMORY_TRAIT_GROUPS", "bad name"),
    ("OWLBANK_MEMORY_TRAIT_GROUPS", "A,,B"), ("OWLBANK_MEMORY_TRAIT_GROUPS", ",".join(f"G{i}" for i in range(11))),
    ("OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS", "00"), ("OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS", "8761"),
    ("OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS", "soon"), ("OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS", "2.5")])
def test_a_bad_trait_setting_is_named_without_its_value(name: str, value: str) -> None:
    with pytest.raises(RuntimeError, match=f"^{name} must") as err:
        trait_settings_from_env({name: value})
    assert value not in str(err.value)


def test_the_app_starts_with_trait_settings_and_refuses_a_bad_one(offline_start: dict[str, Any]) -> None:
    app = build_app_from_env({**_BRIDGE_ENV, **STRANDS_ENV, "OWLBANK_MEMORY_TRAIT_GROUPS": "Contact",
                              "OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS": "48"})
    cfg = app.state.bridge._cfg
    assert (cfg.memory_trait_groups, cfg.web_activity_max_age_hours) == (("Contact",), 48)
    memory = app.state.bridge._memory  # the client that filters gets the same settings
    assert (memory._allow, memory._max_age_hours) == (["Contact"], 48)
    with pytest.raises(RuntimeError, match="^OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS must"):
        build_app_from_env({**_BRIDGE_ENV, **STRANDS_ENV, "OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS": "0"})
