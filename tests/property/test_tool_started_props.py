"""The Strands client yields ``ToolStarted`` only for approved Gateway tools,
once per tool use.

Streams are scripted with ``FakeModel`` over arbitrary messages (text, inline, Gateway and
unapproved tool uses, errors).
"""

from __future__ import annotations

import asyncio
import string
from collections import Counter
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank.tools_contract import INLINE_TOOLS
from tac_bridge.agent_client import AgentError, ToolStarted
from tac_bridge.agent_client_strands import ALLOWED_GATEWAY_TOOLS
from tests.strands_fakes import FakeModel, ScriptError, default_gateway, make_client, text_msg, tool_msg

SID = "CA" + "3" * 32
PROPS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])

#: Names a model might call that are neither inline nor approved Gateway tools.
UNAPPROVED = ("level0___delete_all", "shell", "faq_lookup", "get_branch_hours", "level0___faq_lookupx")

words = st.text(alphabet=string.ascii_letters + " ,.", min_size=1, max_size=16)
use = st.one_of(
    st.tuples(st.just("inline"), st.sampled_from(INLINE_TOOLS)),
    st.tuples(st.just("gateway"), st.sampled_from(ALLOWED_GATEWAY_TOOLS)),
    st.tuples(st.just("other"), st.sampled_from(UNAPPROVED)),
)
inputs = st.dictionaries(st.sampled_from(["q", "account", "n"]), st.one_of(words, st.integers(0, 9)), max_size=2)
message = st.one_of(
    st.tuples(st.just("text"), words),
    st.tuples(st.just("tool"), st.lists(st.tuples(use, inputs), min_size=1, max_size=4), st.one_of(st.just(""), words)),
    st.tuples(st.just("error")),
)


def render(spec: tuple) -> Any:
    if spec[0] == "text":
        return text_msg(spec[1])
    if spec[0] == "error":
        return ScriptError("scripted failure")
    _, uses, text = spec
    return tool_msg(*[(name, inp) for (_, name), inp in uses], text=text or None)


def reached(script: list[tuple], max_iter: int = 6) -> tuple[list[tuple[str, dict]], bool]:
    """The Gateway tool uses of the model calls the turn reaches (in order), and whether it ends
    cleanly (no error, no iteration limit)."""
    out: list[tuple[str, dict]] = []
    for i in range(max_iter):
        spec = script[i] if i < len(script) else ("text", "ok")
        if spec[0] == "error":
            return out, False
        if spec[0] == "text":
            return out, True
        _, uses, _ = spec
        out += [(name, inp) for (kind, name), inp in uses if kind == "gateway"]
        if any(kind == "inline" for (kind, _), _ in uses):
            return out, True
    return out, False


@PROPS
@given(st.lists(message, max_size=7))
def test_tool_started_only_for_approved_gateway_tools(script):
    """ToolStarted names only approved Gateway tools (never inline or unapproved names), one per
    Gateway tool use the turn reaches, with that use's input, in stream order."""
    client = make_client(FakeModel([render(s) for s in script]), gateway=default_gateway())
    events: list[Any] = []

    async def go() -> None:
        async for ev in client.turn(SID, user_text="hello", tool_results=(), system_prompts={"haiku-5-5": "P"}):
            events.append(ev)

    clean = True
    try:
        asyncio.run(go())
    except AgentError:
        clean = False
    started = [e for e in events if isinstance(e, ToolStarted)]
    assert all(e.name in ALLOWED_GATEWAY_TOOLS for e in started)
    assert not any(e.name in INLINE_TOOLS or e.name in UNAPPROVED for e in started)
    expected, ends_cleanly = reached(script)
    if clean and ends_cleanly:
        assert [(e.name, e.input) for e in started] == expected
    else:  # cut short: a prefix of what the reached calls would report
        assert Counter(e.name for e in started) <= Counter(n for n, _ in expected)
