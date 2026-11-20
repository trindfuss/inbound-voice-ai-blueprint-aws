"""The caller hears only text from model messages without tool use.

Random turns: one or more hops (agent turns), each a sequence of model messages, each a random
list of text and toolUse blocks, shaped as a real agent loop runs them: every message of a hop but
the last uses a Gateway tool (the loop runs it and calls the model again), every hop but the last
ends in a message with an inline bank tool (the bridge runs it and resumes), and the turn's last
message uses no tool (it ends the loop). The model streams go through the real Strands client and
the real bridge turn loop.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.catalog import PromptCatalog
from owlbank.pii import FakePiiRedactor, PiiGuard
from owlbank.policy import Policy
from owlbank.segments import SegmentLog
from tac_bridge.agent_client_strands import StrandsAgentClient
from tac_bridge.bridge import Bridge
from tests.component.masking_fixtures import unmasked_policy_doc
from tests.component.test_bridge_fakes import (
    CALL_SID,
    NOW,
    SEALER,
    FakeBackend,
    _inline_io,
    h_msg,
    h_text,
    h_tool,
    load_json,
    seeded_store,
    write_call_state,
)
from tests.strands_fakes import FakeModel, default_gateway, usage
from tests.strands_fakes import settings as strands_settings

GATEWAY_TOOL = "level0___get_branch_hours"
INLINE_TOOL = "list_recent_transactions"

# No "<" or ">": the bridge drops markup and brackets and "<silent/>" markers, which is not
# what this property is about.
words = st.text(alphabet=st.characters(blacklist_characters="<>", blacklist_categories=("Cs",)),
                min_size=1, max_size=12)
block = st.one_of(st.tuples(st.just("text"), words), st.just(("tool", "")))
message = st.lists(block, max_size=4)


@st.composite
def turns(draw: st.DrawFn) -> list[list[list[tuple[str, str]]]]:
    """Hops of messages of blocks, shaped as the loop runs them (see the module docs): a message
    that isn't its hop's last gets a Gateway tool block if it has none; a non-final hop's last
    message gets one inline tool block; the turn's last message has no tool block."""
    hops = draw(st.lists(st.lists(message, min_size=1, max_size=3), min_size=1, max_size=4))
    for h, hop in enumerate(hops):
        for m, msg in enumerate(hop):
            last_of_hop, last_of_turn = m == len(hop) - 1, h == len(hops) - 1 and m == len(hop) - 1
            if last_of_turn:
                msg[:] = [b for b in msg if b[0] == "text"]
            elif not last_of_hop and not any(k == "tool" for k, _ in msg):
                msg.insert(draw(st.integers(0, len(msg))), ("tool", ""))
        if h < len(hops) - 1:
            last = hop[-1]
            last.insert(draw(st.integers(0, len(last))), ("inline", ""))
    return hops


def model_calls(hops: list[list[list[tuple[str, str]]]]) -> list[list[dict[str, Any]]]:
    """One model call per message, in order across the hops."""
    calls: list[list[dict[str, Any]]] = []
    for hop_index, hop in enumerate(hops):
        for m_index, msg in enumerate(hop):
            blocks = []
            for i, (kind, text) in enumerate(msg):
                if kind == "text":
                    blocks.append(h_text(text, i))
                else:
                    name = INLINE_TOOL if kind == "inline" else GATEWAY_TOOL
                    args = {"account": "checking", "limit": 3} if kind == "inline" else {"branch": "main"}
                    blocks.append(h_tool(name, f"tu_{hop_index}_{m_index}_{i}", i, **args))
            if not blocks:  # Strands needs a content block: an empty text one (nothing is spoken)
                blocks.append(h_text("", 0))
            calls.append([*h_msg(*blocks), usage()])
    return calls


def expected_speech(hops: list[list[list[tuple[str, str]]]]) -> str:
    return "".join(text for hop in hops for msg in hop if all(k == "text" for k, _ in msg)
                   for _, text in msg)


def make_bridge(model: FakeModel) -> Bridge:
    store = seeded_store()
    write_call_state(store)
    # This is about the agent's own text: masking phrases off (they have their own test).
    policy = Policy.from_dict(unmasked_policy_doc(load_json("policy.json")))
    backend = FakeBackend({INLINE_TOOL: lambda **_: {"account": "checking", "transactions": []}})
    # This is message mode's contract: stream mode has its own properties.
    agent = StrandsAgentClient(strands_settings(text_release="message"), model_factory=lambda _s, _sid: model, gateway_tools=default_gateway())
    return Bridge(store=store, agent=agent, backend=backend, memory=None,
                  catalog=PromptCatalog.from_dict(load_json("prompt_catalog.json")), policy=policy,
                  pii=PiiGuard.for_policy(FakePiiRedactor(), policy),
                  journal=SegmentLog(store, now=lambda: NOW), sealer=SEALER, now=lambda: NOW,
                  io=_inline_io)


async def speak(bridge: Bridge) -> str:
    context = SimpleNamespace(call_sid=CALL_SID, conversation_id="conv_cp12", metadata={},
                              profile_id=None, profile=None, pending_handoff_data=None)
    return "".join([chunk async for chunk in bridge.run_turn(context, "hello")])


@settings(max_examples=150, deadline=None)
@given(hops=turns())
def test_spoken_text_is_exactly_the_text_of_messages_without_tool_use(
        hops: list[list[list[tuple[str, str]]]]) -> None:
    """Spoken text == the in-order concatenation of the text blocks of the
    messages that contain no toolUse block, across all hops."""
    model = FakeModel(model_calls(hops))
    bridge = make_bridge(model)
    assert asyncio.run(speak(bridge)) == expected_speech(hops)
    assert model.script == []  # every message was streamed
