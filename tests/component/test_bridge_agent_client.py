"""``tac_bridge.agent_client``: the ``AgentClient`` protocol's shared types and error mapping, offline."""

from __future__ import annotations

import json
from pathlib import Path

from owlbank import agent_prompt
from owlbank.catalog import PromptCatalog
from tac_bridge.agent_client import (
    SERVICE_MESSAGE_MAX,
    AgentClient,
    ToolResult,
    agent_error_from_aws,
    join_system,
)
from tac_bridge.agent_client_strands import StrandsAgentClient

ROOT = Path(__file__).resolve().parents[2]


def test_the_strands_client_is_an_agent_client() -> None:
    assert issubclass(StrandsAgentClient, AgentClient)


def test_join_system_puts_the_suffix_on_its_own_line() -> None:
    assert join_system("base", None) == "base"
    assert join_system("base", "Now: Monday.") == "base\nNow: Monday."
    assert join_system(None, "Now: Monday.") == "\nNow: Monday."


def test_the_bridges_suffix_joined_is_the_renderers_prompt() -> None:
    """Agent prompt caching: the bridge's suffix (``per_call + "\\n" + clock``) joined to a
    model's static part is what the renderer says the model reads, for the seed catalog."""
    catalog = PromptCatalog.from_dict(json.loads((ROOT / "data" / "prompt_catalog.json").read_text()))
    clock = "Now: Monday, 9 November 2026, 9:00 AM."
    facts = agent_prompt.CallerFacts("Ana", "name Ana Ruiz; accounts: checking", None)
    for lang in ("en-US", "es-US"):
        for caller in (None, facts):
            rp = agent_prompt.render(catalog, "haiku-5-5", lang, caller=caller, web_block=None)
            assert join_system(rp.static, f"{rp.per_call}\n{clock}") == rp.joined(clock)
            assert rp.joined(clock).splitlines()[-1] == clock


def test_tool_results_are_always_json_text_never_a_json_block() -> None:
    """Tool results go as JSON *text*, found at the first real step-up call."""
    block = ToolResult("tu_1", {"error": "AUTH_LEVEL_INSUFFICIENT", "tool": "make_payment"}, "error").to_block()
    content = block["toolResult"]["content"]
    assert len(content) == 1 and set(content[0]) == {"text"}
    assert json.loads(content[0]["text"]) == {"error": "AUTH_LEVEL_INSUFFICIENT", "tool": "make_payment"}
    assert block["toolResult"]["status"] == "error" and block["toolResult"]["toolUseId"] == "tu_1"
    plain = ToolResult("tu_2", "already text", "success").to_block()["toolResult"]["content"]
    assert plain == [{"text": "already text"}]


class _Err(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__("boom")
        self.response = {"Error": {"Code": code, "Message": message}}


def test_service_diagnostics_are_kept_only_for_selected_aws_error_codes() -> None:
    """runtimeClientError and AccessDenied could only be diagnosed by hand-replaying calls while their
    message was dropped. The message is now kept for infrastructure error codes only."""
    kept = agent_error_from_aws(_Err("ValidationException", "tools.2.custom.input_schema: JSON schema is invalid"
                                     + "x" * 900), operation="Bedrock ConverseStream")
    assert kept.code == "ValidationException" and kept.retryable is False
    assert kept.service_message is not None and kept.service_message.startswith("tools.2.custom.input_schema")
    assert len(kept.service_message) <= SERVICE_MESSAGE_MAX
    assert str(kept) == "Bedrock ConverseStream failed: _Err"

    other = agent_error_from_aws(_Err("SomethingElseEntirely", "the caller said hello"), operation="x")
    assert other.service_message is None and other.retryable is True


def test_throttling_is_retryable() -> None:
    err = agent_error_from_aws(_Err("ThrottlingException", "slow down"), operation="x")
    assert err.retryable is True and err.code == "ThrottlingException"
