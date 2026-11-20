"""Agent prompt caching, the bridge half.

The request is built by the real Strands ``BedrockModel.format_request`` (no network) from the
model :func:`default_model_factory` builds, the system blocks :func:`system_blocks` builds and the
real inline tool specs, over generated message histories (user text, assistant text, toolUse and
toolResult). Every accepted ``OWLBANK_PROMPT_CACHE_TTL`` parses to its setting,
the setting written back parses to itself, and anything else is refused without its value.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tac_bridge.agent_client_strands import (
    PROMPT_CACHE_TTLS,
    default_model_factory,
    inline_tools,
    system_blocks,
)
from tac_bridge.app import strands_settings_from_env
from tests.strands_fakes import settings as strands_settings

TOOL_SPECS = [t.tool_spec for t in inline_tools()]
TTL_ORDER = {"5m": 0, "1h": 1}  # a point with no ttl is Bedrock's default, 5m
ENV = {
    "OWLBANK_MODELS": json.dumps([{"id": "us.anthropic.claude-haiku-5-5", "thinking": "adaptive",
                                   "max_tokens": 2000, "text_release": "stream", "prompt_caching": True}]),
    "OWLBANK_GATEWAY_URL": "https://gw.example/mcp",
    "OWLBANK_MEMORY_ID": "mem-1",
    "AWS_REGION": "us-east-1",
}

_MODELS: dict[tuple[bool, str], Any] = {}


def model(caching: bool, ttl: str) -> Any:
    """The real Bedrock model for the settings (built once per setting; format_request is pure)."""
    key = (caching, ttl)
    if key not in _MODELS:
        _MODELS[key] = default_model_factory(strands_settings(prompt_caching=caching, prompt_cache_ttl=ttl))
    return _MODELS[key]


words = st.text(alphabet=st.characters(categories=("L", "N", "Zs", "P")), min_size=1, max_size=40).filter(
    lambda s: s.strip() != "")
tool_names = st.sampled_from([s["name"] for s in TOOL_SPECS])


@st.composite
def histories(draw: Any) -> list[dict[str, Any]]:
    """A history ending in a user message: user words or tool results answering the previous
    assistant message's toolUse blocks (with or without words), then the assistant's text and/or
    toolUse blocks, repeated."""
    messages: list[dict[str, Any]] = []
    pending: list[str] = []
    n = 0
    for _ in range(draw(st.integers(1, 6))):
        content: list[dict[str, Any]] = [
            {"toolResult": {"toolUseId": tid, "status": draw(st.sampled_from(["success", "error"])),
                            "content": [{"text": draw(words)}]}} for tid in pending]
        if not pending or draw(st.booleans()):
            content.append({"text": draw(words)})
        messages.append({"role": "user", "content": content})
        reply: list[dict[str, Any]] = []
        if draw(st.booleans()):
            reply.append({"text": draw(words)})
        pending = []
        for _ in range(draw(st.integers(0 if reply else 1, 2))):
            n += 1
            tid = f"tooluse_{n}"
            pending.append(tid)
            reply.append({"toolUse": {"toolUseId": tid, "name": draw(tool_names), "input": {"q": draw(words)}}})
        messages.append({"role": "assistant", "content": reply})
    if pending:
        messages.append({"role": "user", "content": [
            {"toolResult": {"toolUseId": tid, "status": "success", "content": [{"text": draw(words)}]}}
            for tid in pending]})
    else:
        messages.append({"role": "user", "content": [{"text": draw(words)}]})
    return messages


suffixes = st.one_of(st.none(), st.just(""), words)


def request(caching: bool, ttl: str, history: list[dict[str, Any]], static: str, suffix: str | None) -> dict[str, Any]:
    return model(caching, ttl).format_request(
        copy.deepcopy(history), tool_specs=TOOL_SPECS,
        system_prompt_content=system_blocks(static, suffix, caching=caching, ttl=ttl))


def points(req: dict[str, Any]) -> list[tuple[str, int, int, dict[str, Any]]]:
    """Every cache point as (section, message index or -1, block index, point)."""
    out = [("tools", -1, i, b["cachePoint"]) for i, b in enumerate(req["toolConfig"]["tools"]) if "cachePoint" in b]
    out += [("system", -1, i, b["cachePoint"]) for i, b in enumerate(req["system"]) if "cachePoint" in b]
    out += [("messages", m, i, b["cachePoint"]) for m, msg in enumerate(req["messages"])
            for i, b in enumerate(msg["content"]) if "cachePoint" in b]
    return out


def without_points(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [b for b in blocks if "cachePoint" not in b]


@settings(max_examples=150, deadline=None)
@given(ttl=st.sampled_from(PROMPT_CACHE_TTLS), history=histories(), static=words, suffix=suffixes)
def test_three_cache_points_with_non_increasing_ttls(ttl: str, history: list[dict[str, Any]],
                                                        static: str, suffix: str | None) -> None:
    req = request(True, ttl, history, static, suffix)
    found = points(req)
    last_user = max(i for i, m in enumerate(req["messages"]) if m["role"] == "user")
    assert [(sec, m) for sec, m, _, _ in found] == [("tools", -1), ("system", -1), ("messages", last_user)]
    (_, _, tools_at, tools_pt), (_, _, system_at, system_pt), (_, _, _, message_pt) = found
    assert tools_at == len(req["toolConfig"]["tools"]) - 1 and system_at == 1
    assert req["system"][0] == {"text": static}
    assert "ttl" not in message_pt
    if ttl == "1h":
        assert tools_pt == system_pt == {"type": "default", "ttl": "1h"}
    else:
        assert all("ttl" not in pt for *_, pt in found)
    ranks = [TTL_ORDER[pt.get("ttl", "5m")] for *_, pt in found]
    assert ranks == sorted(ranks, reverse=True)
    assert len(found) <= 4  # Bedrock's limit


@settings(max_examples=150, deadline=None)
@given(ttl=st.sampled_from(PROMPT_CACHE_TTLS), history=histories(), static=words, suffix=suffixes)
def test_caching_off_has_no_cache_point_and_the_same_text(ttl: str, history: list[dict[str, Any]],
                                                             static: str, suffix: str | None) -> None:
    off = request(False, ttl, history, static, suffix)
    on = request(True, ttl, history, static, suffix)
    assert "cachePoint" not in json.dumps(off)
    assert off["system"] == without_points(on["system"])
    assert off["toolConfig"]["tools"] == without_points(on["toolConfig"]["tools"])
    assert [{**m, "content": without_points(m["content"])} for m in on["messages"]] == off["messages"]


def case_variants(value: str) -> Any:
    return st.tuples(*[st.sampled_from(sorted({c.lower(), c.upper()})) for c in value]).map("".join)


blank = st.text(alphabet=" \t", max_size=3)


@settings(max_examples=200, deadline=None)
@given(data=st.data(), ttl=st.sampled_from(PROMPT_CACHE_TTLS))
def test_an_accepted_ttl_parses_back_to_the_same_setting(data: Any, ttl: str) -> None:
    raw = data.draw(blank) + data.draw(case_variants(ttl)) + data.draw(blank)
    parsed = strands_settings_from_env({**ENV, "OWLBANK_PROMPT_CACHE_TTL": raw})
    assert parsed.prompt_cache_ttl == ttl
    again = strands_settings_from_env({**ENV, "OWLBANK_PROMPT_CACHE_TTL": parsed.prompt_cache_ttl})
    assert again == parsed


@settings(max_examples=300, deadline=None)
@given(raw=st.text(max_size=12))
def test_any_other_ttl_is_refused_without_its_value(raw: str) -> None:
    env = {**ENV, "OWLBANK_PROMPT_CACHE_TTL": raw}
    if raw.strip().lower() in ("", *PROMPT_CACHE_TTLS):
        assert strands_settings_from_env(env).prompt_cache_ttl == (raw.strip().lower() or "5m")
        return
    with pytest.raises(RuntimeError) as err:
        strands_settings_from_env(env)
    assert str(err.value) == "OWLBANK_PROMPT_CACHE_TTL must be 5m or 1h"


@settings(max_examples=200, deadline=None)
@given(value=st.one_of(st.text(max_size=6), st.none(), st.integers(), st.booleans()))
def test_the_settings_accept_exactly_the_two_ttls(value: Any) -> None:
    if value in PROMPT_CACHE_TTLS:
        assert strands_settings(prompt_cache_ttl=value).prompt_cache_ttl == value
    else:
        with pytest.raises(ValueError, match="^prompt_cache_ttl must be one of"):
            strands_settings(prompt_cache_ttl=value)
