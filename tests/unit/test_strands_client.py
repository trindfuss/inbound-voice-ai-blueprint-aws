"""``StrandsAgentClient``, offline.

The model is a scripted ``FakeModel`` or a real ``BedrockModel`` over a faked boto client
(``tests/strands_fakes.py``); the Gateway is a fake provider. No test touches the network.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from strands.handlers.callback_handler import PrintingCallbackHandler, null_callback_handler
from strands.models import BedrockModel, CacheConfig
from strands.types.exceptions import (
    ContextWindowOverflowException,
    EventLoopException,
    MaxTokensReachedException,
    ModelThrottledException,
)

from owlbank.tools_contract import INLINE_TOOLS, inline_tool_specs
from tac_bridge.agent_client import (
    AgentError,
    AgentEvent,
    MessageEnd,
    TextBeforeTool,
    TextDelta,
    TextSuppressed,
    ToolCall,
    ToolResult,
    ToolStarted,
    TurnEnd,
)
from tac_bridge.agent_client_strands import (
    ALLOWED_GATEWAY_TOOLS,
    ENABLED_THINKING_PREFIX,
    MIN_THINKING_BUDGET,
    _GATEWAY_NAMES,
    _SENTENCE_END,
    _USAGE_KEYS,
    THINKING_PARAMS,
    McpGatewayTools,
    ModelEntry,
    StrandsAgentClient,
    default_model_factory,
    make_gateway_tool_provider,
    plan_invocations,
    system_blocks,
    text_release_conflict,
    thinking_request_fields,
    to_agent_error,
    _int_metric,
    _tool_calls,
    _Translator,
    _TurnState,
)
from tests.logutil import lines
from tests.strands_fakes import (
    FakeBedrockRuntime,
    FakeGatewayProvider,
    FakeModel,
    GatewayRuns,
    ScriptError,
    blocks_msg,
    client_error,
    collect,
    default_gateway,
    gateway_tool,
    history_shape,
    make_client,
    settings,
    text_msg,
    tool_msg,
    usage,
)

SID = "CA" + "5" * 32
ROOT = Path(__file__).resolve().parents[2]
LOGGER = "tac_bridge.agent_client_strands"


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def turn(client: StrandsAgentClient, *, text: str | None = None, results: Any = (),
         prompt: str | None = "BASE PROMPT", suffix: str | None = "Clock: 12:00") -> list[Any]:
    return run(collect(client.turn(SID, user_text=text, tool_results=list(results),
                                   system_prompts=None if prompt is None else {"haiku-5-5": prompt},
                                   system_suffix=suffix)))


def agent_of(client: StrandsAgentClient, sid: str = SID) -> Any:
    return client._calls[sid].build.result().agent


def kinds(events: list[Any]) -> list[str]:
    return [type(e).__name__ for e in events]


# --------------------------------------------------------------------------- settings and model


def _infra_constant(name: str) -> Any:
    """A constant from ``infra/owlbank_infra/config.py`` (read from the file's AST, so this project
    needn't import the CDK app)."""
    tree = ast.parse((ROOT / "infra" / "owlbank_infra" / "config.py").read_text())
    (node,) = [n for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
               and any(isinstance(t, ast.Name) and t.id == name
                       for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    return ast.literal_eval(node.value)


def test_thinking_mapping_equals_infra():
    """The per-entry thinking settings infra writes into ``OWLBANK_MODELS`` (
    ``ENTRY_THINKING_MODES`` plus ``enabled:<budget>``) are the ones the client maps."""
    assert set(_infra_constant("ENTRY_THINKING_MODES")) == set(THINKING_PARAMS)
    assert _infra_constant("ENABLED_THINKING_PREFIX") == ENABLED_THINKING_PREFIX
    assert _infra_constant("MIN_THINKING_BUDGET") == MIN_THINKING_BUDGET
    assert thinking_request_fields("default") is None
    assert thinking_request_fields("off") == {"thinking": {"type": "disabled"}}
    assert thinking_request_fields("adaptive")["thinking"]["type"] == "adaptive"
    assert thinking_request_fields("between_tools") == {"thinking": {"type": "between_tools"}}
    assert thinking_request_fields("enabled:2048") == {"thinking": {"type": "enabled", "budget_tokens": 2048}}


@pytest.mark.parametrize("bad", [{"thinking": "on"}, {"model_id": ""}, {"max_tokens": 0},
                                 {"max_iterations": 0}, {"turn_timeout_s": 0}, {"cache_max_calls": 0}])
def test_settings_validation(bad):
    with pytest.raises(ValueError):
        settings(**bad)


def test_settings_defaults():
    s = settings()
    assert (s.max_iterations, s.turn_timeout_s, s.window_messages) == (6, 30.0, 40)
    assert (s.cache_max_calls, s.cache_idle_s) == (500, 900.0)


@pytest.mark.parametrize("thinking", ["adaptive", "off", "default"])
@pytest.mark.parametrize("caching", [True, False])
def test_default_model_factory(thinking, caching):
    m = default_model_factory(settings(thinking=thinking, prompt_caching=caching, max_tokens=777))
    assert isinstance(m, BedrockModel)
    cfg = m.get_config()
    assert cfg["model_id"] == "us.anthropic.claude-haiku-5-5"
    assert cfg["max_tokens"] == 777
    assert cfg.get("additional_request_fields") == thinking_request_fields(thinking)
    if caching:
        assert cfg["cache_config"] == CacheConfig(strategy="anthropic", tools_ttl=True)
    else:
        assert cfg.get("cache_config") is None


@pytest.mark.parametrize("caching", [True, False])
def test_default_model_factory_with_a_one_hour_ttl(caching):
    """Agent prompt caching: ``1h`` sets the tools point's own TTL; ``cache_config.ttl``
    stays unset so the message point keeps the default 5 minutes."""
    cfg = default_model_factory(settings(prompt_caching=caching, prompt_cache_ttl="1h")).get_config()
    if caching:
        assert cfg["cache_config"] == CacheConfig(strategy="anthropic", tools_ttl="1h")
        assert cfg["cache_config"].ttl is None
    else:
        assert cfg.get("cache_config") is None


def test_the_prompt_cache_ttl_setting():
    assert settings().prompt_cache_ttl == "5m"
    assert settings(prompt_cache_ttl="1h").prompt_cache_ttl == "1h"
    for bad in ("1H", "5M", "10m", "", None, 3600, True):
        with pytest.raises(ValueError, match="^prompt_cache_ttl must be one of"):
            settings(prompt_cache_ttl=bad)


def test_system_blocks():
    assert system_blocks("B", "S", caching=False) == [{"text": "B"}, {"text": "S"}]
    assert system_blocks("B", "S", caching=True) == [{"text": "B"}, {"cachePoint": {"type": "default"}},
                                                     {"text": "S"}]
    assert system_blocks("B", None, caching=True) == [{"text": "B"}, {"cachePoint": {"type": "default"}}]
    assert system_blocks(None, "S", caching=True) == [{"text": "S"}]
    assert system_blocks("", "", caching=True) is None


def test_system_blocks_with_a_ttl():
    """``1h`` is written on the hand-placed point; ``5m`` (Bedrock's default) writes none."""
    one_hour = {"cachePoint": {"type": "default", "ttl": "1h"}}
    assert system_blocks("B", "S", caching=True, ttl="1h") == [{"text": "B"}, one_hour, {"text": "S"}]
    assert system_blocks("B", None, caching=True, ttl="1h") == [{"text": "B"}, one_hour]
    assert system_blocks("B", "S", caching=True, ttl="5m") == system_blocks("B", "S", caching=True)
    assert system_blocks("B", "S", caching=True, ttl=None) == system_blocks("B", "S", caching=True)
    assert system_blocks("B", "S", caching=False, ttl="1h") == [{"text": "B"}, {"text": "S"}]
    assert system_blocks(None, "S", caching=True, ttl="1h") == [{"text": "S"}]


@pytest.mark.parametrize("ttl", ["5m", "1h"])
def test_the_turn_places_the_settings_ttl_on_the_system_point(ttl):
    """``_use_entry`` passes ``StrandsSettings.prompt_cache_ttl`` to :func:`system_blocks`."""
    model = FakeModel([text_msg("ok")])
    client = make_client(model, prompt_caching=True, prompt_cache_ttl=ttl)
    turn(client, text="hi")
    assert model.calls[0].system == system_blocks("BASE PROMPT", "Clock: 12:00", caching=True, ttl=ttl)


def test_agent_settings():
    client = make_client(FakeModel([text_msg("hi")]))
    turn(client, text="hello")
    agent = agent_of(client)
    assert agent.callback_handler is null_callback_handler
    assert not isinstance(agent.callback_handler, PrintingCallbackHandler)
    cm = agent.conversation_manager
    assert (cm.window_size, cm.per_turn) == (40, True)


# --------------------------------------------------------------------------- tools


def _without_registry_defaults(schema: dict[str, Any]) -> dict[str, Any]:
    """Strands' tool registry gives every property without a description ``"Property <name>"``
    (``ToolRegistry.validate_tool_spec``, applied to every tool); undo only that."""
    out = json.loads(json.dumps(schema))
    for name, prop in out.get("properties", {}).items():
        if prop.get("description") == f"Property {name}":
            del prop["description"]
    return out


def test_offered_tool_specs_are_the_contract_specs():
    m = FakeModel([text_msg("hi")])
    turn(make_client(m), text="hello")
    offered = {t["name"]: t for t in m.calls[0].tool_specs}
    assert set(offered) == set(INLINE_TOOLS)
    for entry in inline_tool_specs():
        fn = entry["config"]["inlineFunction"]
        spec = offered[entry["name"]]
        assert spec["description"] == fn["description"]
        assert _without_registry_defaults(spec["inputSchema"]["json"]) == fn["inputSchema"]


def test_inline_tool_interrupts_and_resumes():
    m = FakeModel([tool_msg(("t1", "get_balance", {"account": "checking"})), text_msg("Five dollars.")])
    client = make_client(m)
    ev = turn(client, text="balance?")
    assert ev == [ToolCall("t1", "get_balance", {"account": "checking"}), TurnEnd("tool_use", ev[-1].usage)]
    ev = turn(client, results=[ToolResult("t1", {"balance": "5.00"})])
    assert ev[:-1] == [TextDelta("Five dollars.")]
    assert ev[-1].stop_reason == "end_turn"
    msgs = agent_of(client).messages
    assert history_shape(msgs) == [("user", ["text"]), ("assistant", ["toolUse"]),
                                   ("user", ["toolResult"]), ("assistant", ["text"])]
    result = msgs[2]["content"][0]["toolResult"]
    assert result == {"toolUseId": "t1", "status": "success",
                      "content": [{"text": json.dumps({"balance": "5.00"}, separators=(",", ":"))}]}


def test_two_inline_tools_in_one_message():
    m = FakeModel([tool_msg(("ta", "get_balance", {"account": "checking"}),
                            ("tb", "list_recent_transactions", {"account": "savings"})), text_msg("done")])
    client = make_client(m)
    ev = turn(client, text="both")
    calls = [e for e in ev if isinstance(e, ToolCall)]
    assert {c.tool_use_id for c in calls} == {"ta", "tb"}
    assert {c.name for c in calls} == {"get_balance", "list_recent_transactions"}
    # the bridge answers only one: the other gets a cancelled result, in the same user message
    ev = turn(client, results=[ToolResult("tb", {"items": []})])
    assert kinds(ev) == ["TextDelta", "TurnEnd"]
    msgs = agent_of(client).messages
    results = {c["toolResult"]["toolUseId"]: c["toolResult"] for c in msgs[2]["content"]}
    assert set(results) == {"ta", "tb"}
    assert results["ta"]["status"] == "error"
    assert json.loads(results["ta"]["content"][0]["text"]) == {"error": "cancelled"}
    assert results["tb"]["status"] == "success"


def test_inline_and_gateway_tool_in_one_message():
    runs = GatewayRuns()
    m = FakeModel([tool_msg(("tg", "level0___faq_lookup", {"q": "hours"}), ("ti", "get_balance", {"account": "savings"})),
                   text_msg("ok")])
    client = make_client(m, gateway=default_gateway(runs))
    ev = turn(client, text="x")
    assert [e for e in ev if isinstance(e, ToolCall)] == [ToolCall("ti", "get_balance", {"account": "savings"})]
    assert runs.calls == [("level0___faq_lookup", {"q": "hours"})]
    turn(client, results=[ToolResult("ti", {"balance": "1.00"})])
    assert len(runs.calls) == 1  # the Gateway tool's result was kept, not re-run
    msgs = agent_of(client).messages
    assert {c["toolResult"]["toolUseId"] for c in msgs[2]["content"]} == {"tg", "ti"}


def test_gateway_tool_runs_inside_the_loop():
    runs = GatewayRuns()
    m = FakeModel([tool_msg(("level0___get_branch_hours", {}), text="checking"), text_msg("Open at nine.")])
    ev = turn(make_client(m, gateway=default_gateway(runs), text_release="message"), text="hours?")
    assert ev[:-1] == [ToolStarted("level0___get_branch_hours", {}), TextSuppressed(len("checking")),
                       TextDelta("Open at nine.")]
    assert len(runs.calls) == 1 and len(m.calls) == 2


# --------------------------------------------------------------------------- ToolStarted


async def _logged(gen: Any, log: list[Any]) -> list[Any]:
    out = []
    async for ev in gen:
        out.append(ev)
        log.append(ev)
    return out


def test_gateway_tool_use_yields_one_tool_started_before_the_tool_runs():
    log: list[Any] = []
    runs = GatewayRuns(calls=log)  # the tool's run lands in the same log as the events
    m = FakeModel([tool_msg(("tg", "level0___faq_lookup", {"q": "fees"})), text_msg("No fee.")])
    client = make_client(m, gateway=default_gateway(runs))
    ev = run(_logged(client.turn(SID, user_text="fees?", tool_results=[], system_prompts={"haiku-5-5": "P"}), log))
    started = [e for e in ev if isinstance(e, ToolStarted)]
    assert started == [ToolStarted("level0___faq_lookup", {"q": "fees"})]
    assert not [e for e in ev if isinstance(e, ToolCall)]  # observation only, nothing to run
    assert log.index(started[0]) < log.index(("level0___faq_lookup", {"q": "fees"}))
    assert ev.index(started[0]) < ev.index(TextDelta("No fee."))
    assert isinstance(ev[-1], TurnEnd)


def test_inline_tool_use_yields_no_tool_started():
    m = FakeModel([tool_msg(("ti", "get_balance", {"account": "savings"}))])
    ev = turn(make_client(m, gateway=default_gateway()), text="balance?")
    assert not [e for e in ev if isinstance(e, ToolStarted)]
    assert [e for e in ev if isinstance(e, ToolCall)] == [ToolCall("ti", "get_balance", {"account": "savings"})]


def test_several_gateway_tools_in_one_message_yield_one_each():
    runs = GatewayRuns()
    m = FakeModel([tool_msg(("g1", "level0___faq_lookup", {"q": "a"}), ("g2", "level0___get_branch_hours", {}),
                            ("g3", "level0___faq_lookup", {"q": "b"}), ("ti", "get_balance", {"account": "checking"})),
                   text_msg("ok")])
    client = make_client(m, gateway=default_gateway(runs))
    ev = turn(client, text="x")
    assert [e for e in ev if isinstance(e, ToolStarted)] == [
        ToolStarted("level0___faq_lookup", {"q": "a"}), ToolStarted("level0___get_branch_hours", {}),
        ToolStarted("level0___faq_lookup", {"q": "b"})]
    assert [e.name for e in ev if isinstance(e, ToolCall)] == ["get_balance"]
    # resuming the inline call doesn't report the Gateway tools again
    ev = turn(client, results=[ToolResult("ti", {"balance": "1.00"})])
    assert not [e for e in ev if isinstance(e, ToolStarted)]


def test_unparseable_gateway_input_is_flagged():
    m = FakeModel([[{"messageStart": {"role": "assistant"}},
                    {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "g1", "name": "level0___faq_lookup"}}}},
                    {"contentBlockDelta": {"delta": {"toolUse": {"input": "[1, 2]"}}}},
                    {"contentBlockStop": {}}, {"messageStop": {"stopReason": "end_turn"}}, usage()]])
    ev = turn(make_client(m, gateway=default_gateway()), text="x")
    assert [e for e in ev if isinstance(e, ToolStarted)] == [ToolStarted("level0___faq_lookup", {"_unparseable": True})]


def test_gateway_provider_failure_omits_tools_and_warns(caplog):
    m = FakeModel([text_msg("hi")])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        ev = turn(make_client(m, gateway=FakeGatewayProvider(fail=True)), text="hello")
    assert kinds(ev) == ["TextDelta", "TurnEnd"]
    assert m.calls[0].tool_names == list(INLINE_TOOLS)
    warn = [f for f in lines(caplog.records, LOGGER) if f.get("decision") == "omit_gateway_tools"]
    assert warn and warn[0]["error_class"] == "ConnectionError"


def test_unapproved_gateway_tools_never_reach_the_model(caplog):
    extras = [gateway_tool(n) for n in ("level0___delete_all", "shell", "get_balance", "faq_lookup")]
    provider = FakeGatewayProvider([*extras, gateway_tool("level0___faq_lookup")])
    m = FakeModel([text_msg("hi")])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        turn(make_client(m, gateway=provider), text="hello")
    assert m.calls[0].tool_names == [*INLINE_TOOLS, "level0___faq_lookup"]
    spec = next(t for t in m.calls[0].tool_specs if t["name"] == "get_balance")
    assert "fake gateway" not in spec["description"]  # the inline tool, not the impostor
    assert any(f.get("event") == "gateway_tools_dropped" for f in lines(caplog.records, LOGGER))


def test_make_gateway_tool_provider_with_a_fake_mcp_client():
    made: dict[str, Any] = {}

    class FakeMcp:
        def __init__(self, transport, **kw):
            made.update(kw, transport=transport, started=False, stopped=False)

        def start(self):
            made["started"] = True

        def list_tools_sync(self):
            return [gateway_tool("level0___faq_lookup")]

        def stop(self, *a):
            made["stopped"] = True

    provider = make_gateway_tool_provider("https://gw.example/mcp", "us-east-1", mcp_client_factory=FakeMcp)
    assert made["tool_filters"] == {"allowed": list(ALLOWED_GATEWAY_TOOLS)}
    assert made["prefix"] is None and made["started"] and callable(made["transport"])
    assert [t.tool_name for t in provider.tools()] == ["level0___faq_lookup"]
    provider.close()
    assert made["stopped"]


def test_gateway_provider_start_failure_is_not_fatal(caplog):
    class Broken:
        def __init__(self, transport, **kw):
            pass

        def start(self):
            raise ConnectionError("no route")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        provider = make_gateway_tool_provider("https://gw.example/mcp", "us-east-1", mcp_client_factory=Broken)
    assert isinstance(provider, McpGatewayTools)
    with pytest.raises(RuntimeError):
        provider.tools()
    m = FakeModel([text_msg("hi")])
    turn(make_client(m, gateway=provider), text="hello")
    assert m.calls[0].tool_names == list(INLINE_TOOLS)


# --------------------------------------------------------------------------- events


def test_text_released_at_message_end_in_order():
    m = FakeModel([text_msg("Hello there, how can I help?", chunks=4, reasoning="thinking...")])
    ev = turn(make_client(m), text="hi")
    assert "".join(e.text for e in ev if isinstance(e, TextDelta)) == "Hello there, how can I help?"
    assert len([e for e in ev if isinstance(e, TextDelta)]) == 4
    assert "thinking" not in "".join(e.text for e in ev if isinstance(e, TextDelta))
    assert isinstance(ev[-1], TurnEnd)


def test_text_with_tool_use_is_suppressed():
    m = FakeModel([tool_msg(("get_balance", {"account": "checking"}), text="Let me check that.", chunks=3)])
    ev = turn(make_client(m, text_release="message"), text="balance")
    assert kinds(ev) == ["TextSuppressed", "ToolCall", "TurnEnd"]
    assert ev[0] == TextSuppressed(len("Let me check that."))


def test_text_is_held_until_message_stop():
    """Message mode: no TextDelta is yielded before the model message's stop is seen."""
    stop_seen = asyncio.Event()

    class Watching(FakeModel):
        async def stream(self, *a, **kw):
            async for chunk in super().stream(*a, **kw):
                if "messageStop" in chunk:
                    stop_seen.set()
                yield chunk

    async def go():
        client = make_client(Watching([text_msg("abcdef", chunks=3)]), text_release="message")
        async for ev in client.turn(SID, user_text="x", tool_results=(), system_prompts={"haiku-5-5": "p"}):
            if isinstance(ev, TextDelta):
                assert stop_seen.is_set()

    run(go())


def test_usage_and_cache_tokens_in_turn_end():
    m = FakeModel([tool_msg(("level0___faq_lookup", {"q": "a"}), use=usage(100, 10, cache_read=80, cache_write=20)),
                   text_msg("done", use=usage(50, 5, cache_read=40))])
    ev = turn(make_client(m, gateway=default_gateway()), text="x")
    assert ev[-1].usage == {"input_tokens": 150, "output_tokens": 15, "cache_read_tokens": 120,
                            "cache_write_tokens": 20}


def test_model_call_timing_logs_have_no_content(caplog):
    m = FakeModel([tool_msg(("level0___faq_lookup", {"q": "SECRET-QUERY"}), text="NARRATION"),
                   text_msg("REPLY-TEXT", use=usage(9, 3, cache_read=7, cache_write=2))])
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        turn(make_client(m, gateway=default_gateway()), text="CALLER-WORDS")
    calls = [f for f in lines(caplog.records, LOGGER) if f["event"] == "strands_model_call"]
    assert [c["call"] for c in calls] == [1, 2]
    for c in calls:
        assert c["model_ttfb_s"] is not None and c["model_latency_s"] >= c["model_ttfb_s"] >= 0
    assert (calls[1]["input_tokens"], calls[1]["output_tokens"]) == (9, 3)
    assert (calls[1]["cache_read_tokens"], calls[1]["cache_write_tokens"]) == (7, 2)
    blob = " ".join(json.dumps(f, default=str) + r.getMessage() for f, r in zip(lines(caplog.records, LOGGER), caplog.records, strict=False))
    for secret in ("SECRET-QUERY", "NARRATION", "REPLY-TEXT", "CALLER-WORDS"):
        assert secret not in blob


@pytest.mark.parametrize("release", ["stream", "message"])
@pytest.mark.parametrize("reply", ["First sentence here. And a second one.", "Only one sentence", "Costs $3.50 today"])
def test_model_call_times_first_text_and_first_sentence(caplog, reply, release):
    m = FakeModel([tool_msg(("level0___faq_lookup", {"q": "a"})), text_msg(reply, chunks=4)])
    with caplog.at_level(logging.INFO, logger=LOGGER):
        turn(make_client(m, gateway=default_gateway(), text_release=release), text="x")
    tool_call, answer = [f for f in lines(caplog.records, LOGGER) if f["event"] == "strands_model_call"]
    assert tool_call["model_first_text_s"] is None and tool_call["model_first_sentence_s"] is None
    assert 0 <= answer["model_first_text_s"] <= answer["model_first_sentence_s"] <= answer["model_latency_s"]


def test_sentence_end_needs_following_whitespace():
    from tac_bridge.agent_client_strands import _SENTENCE_END
    assert _SENTENCE_END.search("Sure. ") and _SENTENCE_END.search('He said "yes." Then')
    assert not _SENTENCE_END.search("Costs $3.50") and not _SENTENCE_END.search("Done.")


# --------------------------------------------------------------------------- input building


def test_plan_invocations():
    r = ToolResult("a", {"x": 1})
    assert plan_invocations({}, {}, "hi") == [("hi", None)]
    with pytest.raises(ValueError):
        plan_invocations({}, {}, None)
    (inp, lim), = plan_invocations({"a": "ia"}, {"a": r}, None)
    assert lim is None
    assert inp[0]["interruptResponse"]["interruptId"] == "ia"
    assert inp[0]["interruptResponse"]["response"]["status"] == "success"
    first, second = plan_invocations({"a": "ia", "b": "ib"}, {"a": r}, "and then")
    assert first[1] == {"turns": 1} and second == ("and then", None)
    responses = {x["interruptResponse"]["interruptId"]: x["interruptResponse"]["response"] for x in first[0]}
    assert responses["ib"]["status"] == "error" and responses["ia"]["status"] == "success"
    assert all(v is not None for v in responses.values())


def test_barge_in_text_cancels_pending_then_sends_text():
    m = FakeModel([tool_msg(("t1", "get_balance", {"account": "checking"})), text_msg("Sure, savings.")])
    client = make_client(m)
    turn(client, text="balance")
    ev = turn(client, text="actually savings")
    assert ev[:-1] == [TextDelta("Sure, savings.")]
    assert len(m.calls) == 2  # no model call between the cancelled result and the text
    msgs = agent_of(client).messages
    assert history_shape(msgs) == [("user", ["text"]), ("assistant", ["toolUse"]), ("user", ["toolResult"]),
                                   ("user", ["text"]), ("assistant", ["text"])]
    assert msgs[2]["content"][0]["toolResult"]["status"] == "error"


def test_results_and_text_in_one_turn():
    m = FakeModel([tool_msg(("t1", "get_balance", {"account": "checking"})), text_msg("Anything else?")])
    client = make_client(m)
    turn(client, text="balance")
    ev = turn(client, text="thanks", results=[ToolResult("t1", {"balance": "3.00"})])
    assert kinds(ev) == ["TextDelta", "TurnEnd"]
    msgs = agent_of(client).messages
    assert history_shape(msgs)[2:] == [("user", ["toolResult"]), ("user", ["text"]), ("assistant", ["text"])]
    assert msgs[2]["content"][0]["toolResult"]["status"] == "success"


def test_unknown_tool_use_id_is_rejected_without_logging_content(caplog):
    m = FakeModel([tool_msg(("t1", "get_balance", {"account": "checking"}))])
    client = make_client(m)
    turn(client, text="balance")
    before = list(agent_of(client).messages)
    with caplog.at_level(logging.DEBUG, logger=LOGGER), pytest.raises(AgentError) as ei:
        turn(client, results=[ToolResult("t-other", {"secret": "CONTENT-XYZ"})])
    assert (ei.value.code, ei.value.retryable) == ("unknown_tool_use", False)
    assert agent_of(client).messages == before
    assert all("CONTENT-XYZ" not in json.dumps(f, default=str) for f in lines(caplog.records, LOGGER))
    assert any(f.get("event") == "strands_unknown_tool_use" for f in lines(caplog.records, LOGGER))


def test_result_with_nothing_pending_is_unknown():
    client = make_client(FakeModel([text_msg("hi")]))
    with pytest.raises(AgentError) as ei:
        turn(client, results=[ToolResult("t9", {})])
    assert ei.value.code == "unknown_tool_use"


# --------------------------------------------------------------------------- limits


def test_timeout_is_retryable_and_history_stays_usable():
    m = FakeModel([text_msg("a long answer " * 20, chunks=40), text_msg("second")], delay=0.01)
    client = make_client(m, turn_timeout_s=0.1)
    with pytest.raises(AgentError) as ei:
        turn(client, text="hi")
    assert (ei.value.code, ei.value.retryable) == ("timeout_exceeded", True)
    m.delay = 0
    ev = turn(client, text="again")
    assert ev[:-1] == [TextDelta("second")]
    assert history_shape(agent_of(client).messages) == [("user", ["text"]), ("user", ["text"]), ("assistant", ["text"])]


def test_iteration_limit():
    m = FakeModel([tool_msg(("level0___faq_lookup", {"q": str(i)})) for i in range(10)])
    client = make_client(m, gateway=default_gateway())
    with pytest.raises(AgentError) as ei:
        turn(client, text="loop")
    assert (ei.value.code, ei.value.retryable) == ("max_iterations_exceeded", False)
    assert len(m.calls) == 6
    # the counter is per turn: the next turn starts again
    m.script = [text_msg("fine")]
    assert kinds(turn(client, text="next")) == ["TextDelta", "TurnEnd"]


def test_iteration_limit_counts_across_both_invocations():
    m = FakeModel([tool_msg(("t1", "get_balance", {"account": "checking"}))]
                  + [tool_msg(("level0___faq_lookup", {"q": str(i)})) for i in range(10)])
    client = make_client(m, gateway=default_gateway(), max_iterations=3)
    turn(client, text="balance")
    with pytest.raises(AgentError) as ei:
        turn(client, text="but also")
    assert ei.value.code == "max_iterations_exceeded"
    assert len(m.calls) == 1 + 3


# --------------------------------------------------------------------------- errors


def _bedrock_client(script: list[Any], **kw: Any) -> tuple[StrandsAgentClient, FakeBedrockRuntime]:
    rt = FakeBedrockRuntime(script)

    def factory(s, _sid):
        model = default_model_factory(s)
        model.client = rt
        return model

    return StrandsAgentClient(settings(**kw), model_factory=factory), rt


@pytest.mark.parametrize(("code", "retryable"), [
    ("ThrottlingException", True), ("ServiceUnavailableException", True), ("InternalServerException", True),
    ("ModelErrorException", True), ("ValidationException", False), ("AccessDeniedException", False),
    ("ResourceNotFoundException", False),
])
def test_bedrock_error_mapping(code, retryable):
    client, rt = _bedrock_client([client_error(code, "service diagnostic")])
    with pytest.raises(AgentError) as ei:
        turn(client, text="CALLER WORDS")
    assert (ei.value.code, ei.value.retryable) == (code, retryable)
    assert "CALLER WORDS" not in str(ei.value)
    assert len(rt.requests) == 1  # no retries inside the loop (the bridge retries)


def test_bedrock_error_after_a_tool_hop_is_mapped_too():
    client, _ = _bedrock_client([tool_msg(("level0___faq_lookup", {"q": "x"})), client_error("ValidationException")])
    client._gateway = default_gateway()
    with pytest.raises(AgentError) as ei:
        turn(client, text="x")
    assert (ei.value.code, ei.value.retryable) == ("ValidationException", False)


def test_context_overflow_maps_to_its_error_code():
    msg = "Input is too long for requested model"
    client, _ = _bedrock_client([client_error("ValidationException", msg) for _ in range(5)])
    with pytest.raises(AgentError) as ei:
        turn(client, text="x")
    assert (ei.value.code, ei.value.retryable) == ("model_context_window_exceeded", False)


def test_unexpected_model_exception_is_retryable_without_message():
    m = FakeModel([[{"messageStart": {"role": "assistant"}}, ScriptError("SECRET MODEL TEXT")]])
    with pytest.raises(AgentError) as ei:
        turn(make_client(m), text="hi")
    assert ei.value.retryable is True and ei.value.code is None
    assert "SECRET" not in str(ei.value)


@pytest.mark.parametrize(("exc", "code", "retryable"), [
    (ContextWindowOverflowException("x"), "model_context_window_exceeded", False),
    (MaxTokensReachedException("x"), "max_tokens", False),
    (ModelThrottledException("x"), "ThrottlingException", True),
    (EventLoopException(AgentError("m", retryable=False, code="max_iterations_exceeded")),
     "max_iterations_exceeded", False),
    (EventLoopException(client_error("AccessDeniedException")), "AccessDeniedException", False),
    (RuntimeError("boom"), None, True),
])
def test_to_agent_error(exc, code, retryable):
    err = to_agent_error(exc)
    assert (err.code, err.retryable) == (code, retryable)


# --------------------------------------------------------------------------- barge-in


def test_aclose_mid_text_leaves_history_usable():
    m = FakeModel([text_msg("one two three four", chunks=4), text_msg("next")])
    client = make_client(m)

    async def go():
        gen = client.turn(SID, user_text="hi", tool_results=(), system_prompts={"haiku-5-5": "p"})
        await gen.__anext__()  # the first TextDelta: the message has stopped
        await gen.aclose()

    run(go())
    assert kinds(turn(client, text="again")) == ["TextDelta", "TurnEnd"]


def test_aclose_at_tool_call_then_text():
    m = FakeModel([tool_msg(("t1", "get_balance", {"account": "checking"})), text_msg("ok")])
    client = make_client(m)

    async def go():
        gen = client.turn(SID, user_text="balance", tool_results=(), system_prompts={"haiku-5-5": "p"})
        assert isinstance(await gen.__anext__(), MessageEnd)  # the tool-use message ended
        ev = await gen.__anext__()
        assert isinstance(ev, ToolCall)
        await gen.aclose()

    run(go())
    assert kinds(turn(client, text="never mind")) == ["TextDelta", "TurnEnd"]
    shape = history_shape(agent_of(client).messages)
    assert shape[1:4] == [("assistant", ["toolUse"]), ("user", ["toolResult"]), ("user", ["text"])]


# --------------------------------------------------------------------------- cache and sessions


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _cached_client(clock: Clock | None = None, **kw: Any) -> StrandsAgentClient:
    return StrandsAgentClient(settings(**kw), model_factory=lambda _s, _sid: FakeModel(),
                              clock=clock or Clock())


def test_lru_bound():
    client = _cached_client(cache_max_calls=2)

    async def go():
        await client.begin_session("a")
        await client.begin_session("b")
        await client.begin_session("a")  # touch: b is now the oldest
        await client.begin_session("c")

    run(go())
    assert list(client._calls) == ["a", "c"]


def test_idle_eviction():
    clock = Clock()
    client = _cached_client(clock, cache_idle_s=100)

    async def go():
        await client.begin_session("a")
        clock.now = 50
        await client.begin_session("b")
        clock.now = 120  # a idle for 120 s, b for 70 s
        await client.begin_session("c")

    run(go())
    assert list(client._calls) == ["b", "c"]


def test_idle_session_records_go_and_a_touched_one_stays():
    """The per-session failover records are kept in last-used order: the sweep stops at the first one
    still in use, so a record used again moves behind an idle one (else the idle one would stay)."""
    clock = Clock()
    client = _cached_client(clock, cache_idle_s=100)
    client._session_failover("a")
    clock.now = 50
    client._session_failover("b")
    clock.now = 90
    client._session_failover("a")  # a used again: b is now the oldest
    clock.now = 160  # b idle for 110 s, a for 70 s
    client._evict()
    assert list(client._sessions) == ["a"]
    clock.now = 400
    client._evict()
    assert list(client._sessions) == []


def test_begin_session_then_turn_uses_the_same_agent():
    m = FakeModel([text_msg("hi")])
    client = make_client(m)

    async def go():
        await client.begin_session(SID)
        first = client._calls[SID].build.result().agent
        await collect(client.turn(SID, user_text="x", tool_results=(), system_prompts={"haiku-5-5": "p"}))
        assert client._calls[SID].build.result().agent is first
        await client.begin_session(SID, history="cached")
        assert client._calls[SID].build.result().agent is first
        await client.begin_session(SID)  # reload: no store, so the cached agent stays
        assert client._calls[SID].build.result().agent is first
        await client.begin_session(SID, history="new")
        assert client._calls[SID].build.result().agent is not first
        assert client._calls[SID].build.result().agent.messages == []
        with pytest.raises(ValueError):
            await client.begin_session(SID, history="bogus")  # type: ignore[arg-type]

    run(go())


def test_build_failure_is_an_agent_error_and_retried():
    attempts = []

    def factory(_s, _sid):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("model unavailable")
        return FakeModel([text_msg("hi")])

    client = StrandsAgentClient(settings(), model_factory=factory)
    with pytest.raises(AgentError):
        turn(client, text="x")
    assert SID not in client._calls
    assert kinds(turn(client, text="x")) == ["TextDelta", "TurnEnd"]


def test_aclose_closes_gateway_and_clears_cache():
    provider = default_gateway()
    client = make_client(FakeModel([text_msg("hi")]), gateway=provider)
    turn(client, text="x")
    run(client.aclose())
    assert provider.closed and not client._calls


def test_session_manager_factory_hook():
    seen = []
    client = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: FakeModel([text_msg("hi")]),
                                session_manager_factory=lambda sid, history: seen.append((sid, history)))
    turn(client, text="x")
    assert seen == [(SID, "reload")]


# --------------------------------------------------------------------------- record without answering


def test_plan_invocations_record_only():
    r = ToolResult("a", {"next": "await_caller"})
    (inp, lim), = plan_invocations({"a": "ia"}, {"a": r}, None, respond=False)
    assert lim == {"turns": 1} and inp[0]["interruptResponse"]["interruptId"] == "ia"
    # with words, recording is what already happens: one capped invocation, then the words
    assert plan_invocations({"a": "ia"}, {"a": r}, "hi", respond=False) == plan_invocations({"a": "ia"}, {"a": r}, "hi")


def _paused(model: FakeModel) -> StrandsAgentClient:
    client = make_client(model)
    turn(client, text="pay 120")
    return client


AWAIT = {"status": "completed", "tool": "make_payment", "next": "await_caller", "data": {"executed": True}}


def test_record_results_adds_the_result_with_no_model_call_and_the_words_follow():
    model = FakeModel([tool_msg(("tu-1", "make_payment", {"amount": "120.00"})), text_msg("Connecting you.")])
    client = _paused(model)
    assert len(model.calls) == 1
    out = run(collect(client.record_results(SID, [ToolResult("tu-1", AWAIT)], system_prompts={"haiku-5-5": "P"})))
    assert out == [TurnEnd("end_turn", None)]
    assert len(model.calls) == 1  # recorded, never answered
    agent = agent_of(client)
    assert history_shape(agent.messages)[-2:] == [("assistant", ["toolUse"]), ("user", ["toolResult"])]
    # a second record of the same results (a bridge retry) adds nothing and calls no model
    again = run(collect(client.record_results(SID, [ToolResult("tu-1", AWAIT)], system_prompts={"haiku-5-5": "P"})))
    assert again == [TurnEnd("end_turn", None)] and len(agent.messages) == 3
    # the caller's words: one model call, after the result
    assert kinds(turn(client, text="transfer me")) == ["TextDelta", "TurnEnd"]
    assert len(model.calls) == 2
    assert history_shape(model.calls[-1].messages)[-2:] == [("user", ["toolResult"]), ("user", ["text"])]


def test_results_with_words_after_an_earlier_record_send_only_the_words():
    """A bridge retry of the combined turn after the results were recorded but before the words."""
    model = FakeModel([tool_msg(("tu-1", "make_payment", {})), text_msg("Connecting you.")])
    client = _paused(model)
    run(collect(client.record_results(SID, [ToolResult("tu-1", AWAIT)], system_prompts={"haiku-5-5": "P"})))
    out = turn(client, text="transfer me", results=[ToolResult("tu-1", AWAIT)])
    assert kinds(out) == ["TextDelta", "TurnEnd"] and len(model.calls) == 2
    shape = history_shape(agent_of(client).messages)
    assert [s for s in shape if s == ("user", ["toolResult"])] == [("user", ["toolResult"])]


def test_a_recorded_result_survives_a_reload(monkeypatch):
    from tests.strands_fakes import FakeMemoryData, light_memory_client, make_memory_client

    light_memory_client(monkeypatch)
    data = FakeMemoryData()
    model = FakeModel([tool_msg(("tu-1", "make_payment", {})), text_msg("Sure.")])

    async def go() -> list[Any]:
        first = make_memory_client(model, data)
        await first.begin_session(SID, history="new")
        await collect(first.turn(SID, user_text="pay", tool_results=[], system_prompts={"haiku-5-5": "P"}))
        await collect(first.record_results(SID, [ToolResult("tu-1", AWAIT)], system_prompts={"haiku-5-5": "P"}))
        await first.aclose()
        second = make_memory_client(model, data)
        await second.begin_session(SID, history="reload")
        await collect(second.turn(SID, user_text="thanks", tool_results=[], system_prompts={"haiku-5-5": "P"}))
        return agent_of(second).messages

    messages = run(go())
    assert len(model.calls) == 2
    assert history_shape(messages) == [("user", ["text"]), ("assistant", ["toolUse"]), ("user", ["toolResult"]),
                                       ("user", ["text"]), ("assistant", ["text"])]

# --------------------------------------------------------------------------- text release
#
# The translator is driven directly with
# Strands stream events (``feed_all``), and the client end to end over ``FakeModel``.


class _ReferenceTranslator:
    """A frozen copy of ``_Translator`` as it was before stream mode:
    message mode must yield exactly what this yields. Do not
    edit it to follow the client; it is the behaviour message mode is held to.

    Strands stream events → protocol events.

    Text is held per model message (raw ``event`` chunks) and released at ``messageStop`` unless the
    message has a tool use, in which case its length is reported as :class:`TextSuppressed`. Text
    still held when the message ends without a stop is suppressed too (nothing proves it had no
    tool use). Reasoning is never spoken. A :class:`ToolCall` comes from the interrupt event, or,
    for a call the client defers (hand-offs), from the invocation's ``result``.

    A tool-use block naming an approved Gateway tool yields one
    :class:`ToolStarted` when the block completes in the stream (``contentBlockStop``, or the
    ``messageStop`` if the block was never closed). That is the earliest point its input is known,
    and Strands runs the tool only after the message ends, so it always precedes the tool's run.
    Once per tool-use ID; never for inline tools (they yield :class:`ToolCall`) or other names.
    """

    def __init__(self, state: _TurnState, defer: Callable[[str], bool]) -> None:
        self.state = state
        self.defer = defer
        self.held: list[str] = []
        self.tool_message = False
        self.deferred: list[ToolCall] = []
        self.open_tool: dict[str, str] | None = None  # the tool-use block being streamed
        self.started: set[str] = set()  # tool-use IDs already reported as ToolStarted

    def feed(self, ev: Mapping[str, Any]) -> Iterator[AgentEvent]:
        if "event" in ev:
            yield from self._chunk(ev["event"])
        elif "message" in ev:
            msg = ev["message"]
            if isinstance(msg, Mapping) and msg.get("role") == "assistant":
                yield from self._suppress_held()
        elif "tool_interrupt_event" in ev:
            for call in _tool_calls(ev["tool_interrupt_event"]):
                if self.defer(call.name):
                    self.deferred.append(call)
                else:
                    yield call
        elif "result" in ev:
            yield from self.deferred
            self.deferred = []

    def _chunk(self, chunk: Mapping[str, Any]) -> Iterator[AgentEvent]:
        st = self.state
        if "messageStart" in chunk:
            yield from self._suppress_held()
            self.tool_message = False
            self.open_tool = None
        elif "contentBlockStart" in chunk:
            self._first_byte()
            yield from self._close_tool()
            start = (chunk["contentBlockStart"] or {}).get("start") or {}
            if "toolUse" in start:
                self.tool_message = True
                use = start["toolUse"] or {}
                self.open_tool = {"id": str(use.get("toolUseId", "")), "name": str(use.get("name", "")),
                                  "input": ""}
        elif "contentBlockDelta" in chunk:
            self._first_byte()
            delta = (chunk["contentBlockDelta"] or {}).get("delta") or {}
            if "text" in delta:
                if delta["text"]:
                    self.held.append(delta["text"])
                    self._time_text()
            elif "toolUse" in delta:
                self.tool_message = True
                if self.open_tool is not None:
                    piece = (delta["toolUse"] or {}).get("input", "")
                    if isinstance(piece, str):
                        self.open_tool["input"] += piece
            # reasoningContent (and anything else) is never spoken
        elif "contentBlockStop" in chunk:
            yield from self._close_tool()
        elif "messageStop" in chunk:
            yield from self._close_tool()
            stop = (chunk["messageStop"] or {}).get("stopReason")
            st.call_stop = stop
            if self.held and st.first_sentence is None:
                st.first_sentence = time.monotonic()  # a one-sentence answer ends with the message
            if self.tool_message or stop == "tool_use":
                yield from self._suppress_held()
            else:
                for text in self.held:
                    yield TextDelta(text)
                self.held = []
            self.tool_message = False
        elif "metadata" in chunk:
            meta = chunk["metadata"] or {}
            raw = meta.get("usage")
            if isinstance(raw, Mapping):
                for src, dst in _USAGE_KEYS:
                    value = raw.get(src)
                    if isinstance(value, int) and not isinstance(value, bool):
                        st.call_usage[dst] = st.call_usage.get(dst, 0) + value
                        st.usage[dst] = st.usage.get(dst, 0) + value
            timings = meta.get("metrics")
            if isinstance(timings, Mapping):  # The provider's timings for strands_model_call
                for key in ("latencyMs", "timeToFirstByteMs"):
                    value = _int_metric(timings, key)
                    if value is not None:
                        st.call_metrics[key] = value

    def _close_tool(self) -> Iterator[AgentEvent]:
        """End the open tool-use block: a :class:`ToolStarted` if it names an approved Gateway tool
        not yet reported. The input is the model's JSON; anything else is reported as
        ``{"_unparseable": True}``, as for an inline tool."""
        block, self.open_tool = self.open_tool, None
        if block is None or block["name"] not in _GATEWAY_NAMES:
            return
        if block["id"]:
            if block["id"] in self.started:
                return
            self.started.add(block["id"])
        try:
            args = json.loads(block["input"] or "{}")
        except ValueError:
            args = None
        if not isinstance(args, dict):
            args = {"_unparseable": True}
        yield ToolStarted(block["name"], args)

    def _suppress_held(self) -> Iterator[AgentEvent]:
        if self.held:
            yield TextSuppressed(sum(map(len, self.held)))
        self.held = []

    def _first_byte(self) -> None:
        if self.state.first_byte is None:
            self.state.first_byte = time.monotonic()

    def _time_text(self) -> None:
        """When the call's first answer text, and the end of its first sentence, arrived."""
        st = self.state
        if st.first_text is None:
            st.first_text = time.monotonic()
        if st.first_sentence is None and _SENTENCE_END.search("".join(self.held)):
            st.first_sentence = time.monotonic()


def feed_all(chunks: Any, *, release: str, defer: Any = lambda _name: False, translator: Any = None,
             extra: Any = ({"message": {"role": "assistant"}},), message_ends: bool = False) -> list[Any]:
    """The events a translator yields for one model call's ``chunks`` (raw ``event`` chunks), then
    the ``extra`` Strands events (default: the ``message`` event Strands sends after each model
    message). ``translator``: the class (default ``_Translator``; ``_ReferenceTranslator`` takes no
    ``release``). :class:`MessageEnd` only with ``message_ends`` (checked by its own tests)."""
    cls = translator or _Translator
    tr = cls(_TurnState(), defer) if cls is _ReferenceTranslator else cls(_TurnState(), defer, release=release)
    out: list[Any] = []
    for c in chunks:
        out += list(tr.feed({"event": c}))
    for ev in extra:
        out += list(tr.feed(ev))
    return out if message_ends else [e for e in out if not isinstance(e, MessageEnd)]


def texts(events: list[Any]) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


def test_text_release_setting_default_and_values():
    assert settings().primary.text_release == "stream"
    assert settings(text_release="message").primary.text_release == "message"
    with pytest.raises(ValueError, match="text_release"):
        settings(text_release="sentence")


def test_stream_release_with_thinking_off_is_refused():
    # The same rule as infra's per-entry check, at the client.
    base = {"id": "m", "max_tokens": 10, "prompt_caching": False}
    with pytest.raises(ValueError, match="text_release=stream needs thinking adaptive or default"):
        ModelEntry(**base, thinking="off")  # stream is the default
    with pytest.raises(ValueError, match="set text_release=message"):
        ModelEntry(**base, thinking="off", text_release="stream")
    assert ModelEntry(**base, thinking="off", text_release="message").text_release == "message"


@pytest.mark.parametrize("release", ["stream", "message"])
@pytest.mark.parametrize("thinking", sorted(THINKING_PARAMS))
def test_text_release_conflict_is_the_one_rule(release, thinking):
    # One predicate and message for StrandsSettings, the bridge's environment and
    # the bench: only stream release with reasoning off is refused.
    conflict = text_release_conflict(release, thinking)
    assert (conflict is not None) == (release == "stream" and thinking == "off")
    base = {"id": "m", "max_tokens": 10, "prompt_caching": False}
    if conflict is None:
        assert ModelEntry(**base, thinking=thinking, text_release=release).text_release == release
    else:
        with pytest.raises(ValueError) as err:
            ModelEntry(**base, thinking=thinking, text_release=release)
        assert str(err.value) == conflict
    for thinking in ("adaptive", "default"):
        assert ModelEntry(**base, thinking=thinking).text_release == "stream"


@pytest.mark.parametrize("release", ["stream", "message"])
@pytest.mark.parametrize("parts", [["Hi there.", " More"], ['He said "ok.', '"', " then"], ["Wait?", ")", "]\n"]])
def test_a_sentence_end_split_across_chunks_is_timed_at_its_whitespace(release, parts):
    st = _TurnState()
    tr = _Translator(st, lambda _n: False, release=release)
    list(tr.feed({"event": {"messageStart": {"role": "assistant"}}}))
    list(tr.feed({"event": {"contentBlockStart": {"start": {}}}}))
    for i, p in enumerate(parts):
        list(tr.feed({"event": {"contentBlockDelta": {"delta": {"text": p}}}}))
        assert (st.first_sentence is not None) == (i == len(parts) - 1), (i, p)


def test_sentence_end_search_reads_only_a_short_tail():
    # Linear: the search window is the new chunk plus a short tail of the message's text.
    st = _TurnState()
    tr = _Translator(st, lambda _n: False, release="stream")
    list(tr.feed({"event": {"messageStart": {"role": "assistant"}}}))
    for _ in range(200):
        list(tr.feed({"event": {"contentBlockDelta": {"delta": {"text": "word "}}}}))
        assert len(tr.tail) <= 16
    assert st.first_sentence is None


@pytest.mark.parametrize("release", ["stream", "message"])
def test_plain_answer_same_text_in_both_modes(release):
    chunks = text_msg("Your savings balance is five dollars. Anything else?", chunks=5, reasoning="SECRET-THOUGHT")
    ev = feed_all(chunks, release=release)
    assert texts(ev) == "Your savings balance is five dollars. Anything else?"
    assert kinds(ev) == ["TextDelta"] * 5  # every chunk, nothing else (no suppression, no late tool)
    assert "SECRET-THOUGHT" not in texts(ev)


def test_stream_mode_yields_each_chunk_before_the_message_ends():
    """Stream mode: each chunk is yielded on its own delta, not at ``messageStop``."""
    tr = _Translator(_TurnState(), lambda _n: False, release="stream")
    seen: list[tuple[str, list[Any]]] = []
    for c in text_msg("abcdef", chunks=3):
        seen.append((next(iter(c)), list(tr.feed({"event": c}))))
    assert [ev for kind, ev in seen if kind == "contentBlockDelta"] == [[TextDelta("ab")], [TextDelta("cd")],
                                                                       [TextDelta("ef")]]
    assert all(not ev for kind, ev in seen if kind not in ("contentBlockDelta", "messageStop"))
    assert [ev for kind, ev in seen if kind == "messageStop"] == [[MessageEnd()]]


def test_stream_mode_client_yields_text_before_the_message_stop():
    stop_seen = asyncio.Event()
    early: list[bool] = []

    class Watching(FakeModel):
        async def stream(self, *a, **kw):
            async for chunk in super().stream(*a, **kw):
                if "messageStop" in chunk:
                    stop_seen.set()
                yield chunk

    async def go():
        client = make_client(Watching([text_msg("abcdef", chunks=3)]), text_release="stream")
        async for ev in client.turn(SID, user_text="x", tool_results=(), system_prompts={"haiku-5-5": "p"}):
            if isinstance(ev, TextDelta):
                early.append(not stop_seen.is_set())

    run(go())
    assert early == [True, True, True]


@pytest.mark.parametrize("release", ["stream", "message"])
def test_text_then_inline_tool(release):
    m = FakeModel([tool_msg(("t1", "list_recent_transactions", {"account": "checking"}),
                            text="Let me check that.", chunks=3), text_msg("Two items.")])
    client = make_client(m, text_release=release)
    ev = turn(client, text="recent?")
    if release == "stream":
        assert ev[:-1] == [TextDelta("Let me"), TextDelta(" check"), TextDelta(" that."),
                           TextBeforeTool(len("Let me check that."), "list_recent_transactions"),
                           ToolCall("t1", "list_recent_transactions", {"account": "checking"})]
    else:
        assert ev[:-1] == [TextSuppressed(len("Let me check that.")),
                           ToolCall("t1", "list_recent_transactions", {"account": "checking"})]
    ev = turn(client, results=[ToolResult("t1", {"items": []})])
    assert ev[:-1] == [TextDelta("Two items.")]
    # the released narration is in the history as the model wrote it: one assistant message
    msgs = agent_of(client).messages
    assert history_shape(msgs)[1] == ("assistant", ["text", "toolUse"])


@pytest.mark.parametrize("release", ["stream", "message"])
def test_text_then_gateway_tool(release):
    runs = GatewayRuns()
    m = FakeModel([tool_msg(("level0___get_branch_hours", {}), text="checking"), text_msg("Open at nine.")])
    ev = turn(make_client(m, gateway=default_gateway(runs), text_release=release), text="hours?")
    if release == "stream":
        assert ev[:-1] == [TextDelta("checking"), TextBeforeTool(8, "level0___get_branch_hours"),
                           ToolStarted("level0___get_branch_hours", {}), TextDelta("Open at nine.")]
    else:
        assert ev[:-1] == [ToolStarted("level0___get_branch_hours", {}), TextSuppressed(8),
                           TextDelta("Open at nine.")]
    assert len(runs.calls) == 1 and len(m.calls) == 2


@pytest.mark.parametrize("release", ["stream", "message"])
def test_text_then_hand_off(release):
    """A hand-off's ToolCall is deferred to the invocation's result; in stream mode the text
    before it was already released, and TextBeforeTool comes before the deferred call."""
    args = {"reason": "fraud_suspected", "summary": "Caller reports fraud."}
    m = FakeModel([tool_msg(("th", "escalate", args), text="One moment.")])
    ev = turn(make_client(m, text_release=release), text="fraud")
    if release == "stream":
        assert ev[:-1] == [TextDelta("One moment."), TextBeforeTool(len("One moment."), "escalate"),
                           ToolCall("th", "escalate", args)]
    else:
        assert ev[:-1] == [TextSuppressed(len("One moment.")), ToolCall("th", "escalate", args)]


@pytest.mark.parametrize("release", ["stream", "message"])
def test_tool_then_text_is_suppressed_in_both_modes(release):
    chunks = blocks_msg(("tool", "list_recent_transactions", {"account": "savings"}), ("text", "AFTER-TOOL", 2))
    assert feed_all(chunks, release=release) == [TextSuppressed(len("AFTER-TOOL"))]


def test_text_tool_text_releases_only_the_text_before_the_tool():
    chunks = blocks_msg(("text", "Before. ", 2), ("tool", "level0___faq_lookup", {"q": "a"}, "g1"),
                        ("text", "AFTER", 1), ("tool", "list_recent_transactions", {"account": "savings"}))
    ev = feed_all(chunks, release="stream")
    assert ev == [TextDelta("Befo"), TextDelta("re. "), TextBeforeTool(8, "level0___faq_lookup"),
                  ToolStarted("level0___faq_lookup", {"q": "a"}), TextSuppressed(len("AFTER"))]
    assert feed_all(chunks, release="message") == [ToolStarted("level0___faq_lookup", {"q": "a"}),
                                                   TextSuppressed(len("Before. AFTER"))]


def test_no_text_before_tool_means_no_text_before_tool_event():
    chunks = blocks_msg(("reasoning", "thinking"), ("tool", "list_recent_transactions", {"account": "savings"}))
    assert feed_all(chunks, release="stream") == []
    assert feed_all(chunks, release="message") == []


@pytest.mark.parametrize(("text", "n"), [("\n\n", 1), ("  ", 2), ("<silent/>", 3), ("\n<silent/> ", 4),
                                         ("<Silent />", 3), ("<thinking>x</thinking><br/>", 2)])
def test_no_text_before_tool_when_the_caller_hears_nothing(text, n):
    """Whitespace, ``<silent/>`` (any case or spacing) or markup before a tool: released, but nothing
    is heard, so no TextBeforeTool."""
    ev = feed_all(blocks_msg(("text", text, n), ("tool", "list_recent_transactions", {})), release="stream")
    assert texts(ev) == text
    assert not [e for e in ev if isinstance(e, TextBeforeTool)]


def test_text_before_tool_counts_only_what_the_caller_hears():
    ev = feed_all(blocks_msg(("text", "<silent/>One moment. ", 4), ("tool", "list_recent_transactions", {})),
                  release="stream")
    assert [e for e in ev if isinstance(e, TextBeforeTool)] == [
        TextBeforeTool(len("One moment. "), "list_recent_transactions")]


@pytest.mark.parametrize("release", ["stream", "message"])
def test_reasoning_is_never_text(release):
    chunks = blocks_msg(("reasoning", "SECRET-A"), ("text", "Hi.", 1), ("reasoning", "SECRET-B"), ("text", " Bye.", 1))
    ev = feed_all(chunks, release=release)
    assert ev == [TextDelta("Hi."), TextDelta(" Bye.")]


def test_message_without_stop():
    """The ``message`` event without a ``messageStop``: held text is suppressed as
    today; text already released stays released."""
    plain = [c for c in text_msg("Released text", chunks=2) if "messageStop" not in c]
    assert feed_all(plain, release="stream") == [TextDelta("Release"), TextDelta("d text")]
    assert feed_all(plain, release="message") == [TextSuppressed(len("Released text"))]
    late = [c for c in blocks_msg(("text", "Spoken", 1), ("tool", "list_recent_transactions", {}),
                                  ("text", "HELD", 1)) if "messageStop" not in c]
    assert feed_all(late, release="stream") == [TextDelta("Spoken"), TextBeforeTool(6, "list_recent_transactions"),
                                                TextSuppressed(4)]


def test_text_before_tool_once_per_message_and_reset_per_message():
    one = blocks_msg(("text", "A", 1), ("tool", "level0___faq_lookup", {"q": "x"}, "g1"),
                     ("tool", "level0___get_branch_hours", {}, "g2"))
    two = blocks_msg(("text", "BB", 1), ("tool", "list_recent_transactions", {}))
    ev = feed_all(one + [{"message": {"role": "assistant"}}] + two, release="stream")
    assert [e for e in ev if isinstance(e, TextBeforeTool)] == [TextBeforeTool(1, "level0___faq_lookup"),
                                                                 TextBeforeTool(2, "list_recent_transactions")]


def test_tool_use_stop_without_a_tool_block_still_reports_released_text():
    """Defensive: a ``tool_use`` stop with no tool-use block seen still reports the spoken text."""
    chunks = text_msg("Hmm", chunks=1)
    chunks = [{"messageStop": {"stopReason": "tool_use"}} if "messageStop" in c else c for c in chunks]
    assert feed_all(chunks, release="stream") == [TextDelta("Hmm"), TextBeforeTool(3, "")]
    assert feed_all(chunks, release="message") == [TextSuppressed(3)]


@pytest.mark.parametrize("chunks", [
    text_msg("Hello there, how can I help?", chunks=4, reasoning="thinking..."),
    blocks_msg(("text", "Before", 2), ("tool", "level0___faq_lookup", {"q": "a"}, "g9"), ("text", "after", 1)),
    blocks_msg(("tool", "escalate", {"reason": "complaint", "summary": "x"}, "e1"), ("text", "after", 1)),
    [c for c in text_msg("no stop", chunks=2) if "messageStop" not in c],
])
def test_message_mode_equals_the_reference_translator(chunks):
    """On fixed inputs (the property test covers generated ones)."""
    for defer in (lambda _n: False, lambda n: n == "escalate"):
        assert feed_all(chunks, release="message", defer=defer) == \
            feed_all(chunks, release="message", defer=defer, translator=_ReferenceTranslator)


@pytest.mark.parametrize("release", ["stream", "message"])
def test_translator_times_first_text_and_first_sentence_in_both_modes(release):
    for chunks, sentence in ((text_msg("First one. Then more.", chunks=6), True),
                             (text_msg("Only one sentence", chunks=3), True),  # ends with the message
                             (blocks_msg(("tool", "list_recent_transactions", {})), False)):
        st = _TurnState()
        tr = _Translator(st, lambda _n: False, release=release)
        for c in chunks:
            list(tr.feed({"event": c}))
        if sentence:
            assert st.first_text is not None and st.first_sentence is not None
            assert st.first_text <= st.first_sentence
        else:
            assert st.first_text is None and st.first_sentence is None
