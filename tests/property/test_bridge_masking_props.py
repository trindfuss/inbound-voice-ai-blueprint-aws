"""A masking phrase never reaches the agent.

For random phrase texts (a catalog override of every ``mask.*`` entry) and random tool sequences,
the phrase text appears in no hop's ``user_text`` or tool results, and in no ``ai_turn`` journal
text, while the caller does hear it when the planner picks it.
"""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank.catalog import PromptCatalog
from owlbank.pii import FakePiiRedactor, PiiGuard
from owlbank.policy import MASK_GENERIC_KEYS, Policy
from owlbank.segments import SegmentLog
from tac_bridge.agent_client import TextDelta, ToolCall, TurnEnd
from tac_bridge.bridge import Bridge
from tests.component.masking_fixtures import masked_policy_doc
from tests.component.test_bridge_fakes import (
    CALL_SID,
    NOW,
    SEALER,
    FakeAgent,
    FakeBackend,
    _inline_io,
    load_json,
    seeded_store,
    write_call_state,
)

CATALOG = load_json("prompt_catalog.json")
MASK_TOOL_KEYS = ("mask.list_recent_transactions", "mask.change_address", "mask.get_balance")
TOOLS: dict[str, dict[str, Any]] = {
    "list_recent_transactions": {"account": "checking"},
    "change_address": {"street": "1 Main St", "city": "Springfield", "state": "CA", "postal_code": "94105"},
    "get_balance": {"account": "savings"},
    "no_such_tool": {},
}

# A distinctive phrase: no catalog placeholder braces, and a prefix no other text has.
phrases = st.text(alphabet="abcdefghijklmnopqrstuvwxyz ,.!", min_size=4, max_size=40).map(
    lambda t: "Qzx" + t.strip())

# One model message: text only, or tool calls only (as the agent client hands them over).
message = st.one_of(
    st.lists(st.sampled_from(["Sure.", " Here it is.", "<silent/>", " Done."]), min_size=1, max_size=3)
    .map(lambda parts: ("text", parts)),
    st.lists(st.sampled_from(sorted(TOOLS)), min_size=1, max_size=3).map(lambda names: ("tools", names)),
)


def catalog_with(phrase_by_key: dict[str, str]) -> PromptCatalog:
    doc = copy.deepcopy(CATALOG)
    for entry in doc["entries"]:
        if entry["key"] in phrase_by_key:
            entry["text"] = phrase_by_key[entry["key"]]
    return PromptCatalog.from_dict(doc)


def script_of(messages: list[tuple[str, list[str]]]) -> list[list[Any]]:
    script: list[list[Any]] = []
    for i, (kind, items) in enumerate(messages):
        if kind == "text":
            script.append([*(TextDelta(t) for t in items), TurnEnd("end_turn")])
        else:
            script.append([*(ToolCall(f"tu_{i}_{j}", name, dict(TOOLS[name])) for j, name in enumerate(items)),
                           TurnEnd("tool_use")])
    return script


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(texts=st.fixed_dictionaries(dict.fromkeys((*MASK_GENERIC_KEYS, *MASK_TOOL_KEYS), phrases)),
       messages=st.lists(message, min_size=1, max_size=4),
       level=st.sampled_from([1, 2]),
       slow=st.booleans())
def test_a_masking_phrase_never_reaches_the_agent(
        texts: dict[str, str], messages: list[tuple[str, list[str]]], level: int, slow: bool) -> None:
    store = seeded_store()
    write_call_state(store, level=level)
    # Timer on and short; a slow agent makes it fire. End immediately: no hand-off hold.
    # Tool phrases at once (tool_after_ms=0), so the "not vacuous" check below still holds.
    policy = Policy.from_dict(masked_policy_doc(after_ms=5, tool_after_ms=0, end_mode="immediate",
                                                handoff_wait_ms=0))
    journal = SegmentLog(store, now=lambda: NOW)
    agent = FakeAgent(script_of(messages), delay_s=0.01 if slow else 0.0)
    backend = FakeBackend({
        "list_recent_transactions": lambda **_: {"tool": "list_recent_transactions",
                                                 "data": {"transactions": []}},
        "change_address": lambda **_: {"tool": "change_address", "data": {"updated": True}},
        "get_balance": lambda **_: {"tool": "get_balance", "data": {"available": "1.00"}},
    })
    bridge = Bridge(store=store, agent=agent, backend=backend, memory=None, catalog=catalog_with(texts),
                    policy=policy, pii=PiiGuard.for_policy(FakePiiRedactor(), policy), journal=journal,
                    sealer=SEALER, now=lambda: NOW, io=_inline_io)
    context = SimpleNamespace(call_sid=CALL_SID, conversation_id="conv_cp15", metadata={},
                              profile_id=None, profile=None, pending_handoff_data=None)

    async def go() -> str:
        return "".join([c async for c in bridge.run_turn(context, "hello")])

    heard = asyncio.run(go())
    seen_by_agent = agent.all_inputs_text()
    ai_turns = " ".join(str(e.get("text", "")) for e in journal.read(CALL_SID) if e["type"] == "ai_turn")
    for text in texts.values():
        assert text not in seen_by_agent
        assert text not in ai_turns
    # Not vacuous: the first tool that will run (or hand off) is masked; on a fast first hop its
    # phrase comes first, on a slow one it may follow the acknowledgement.
    kind, items = messages[0]
    if kind == "tools":
        runnable = [n for n in items if n == "list_recent_transactions"
                    or (n == "change_address" and level >= 2) or (n == "get_balance" and len(items) == 1)]
        if runnable:
            phrase = texts[f"mask.{runnable[0]}"]
            assert heard.startswith(phrase) if not slow else phrase in heard
