"""The bridge's scrub never changes the agent's input.

For any caller text and any sensitive values sealed into call state (by the front door, here
``sensitive.remember`` and then a forget, as the bridge is another process), the text the bridge
hands to the agent is byte-identical with and without the values, and so is what the caller
hears; the journal copy of the AI turn (an echo of the caller's words) contains no registered
value (case, spaces and hyphens tolerated).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank import segments, sensitive
from owlbank.catalog import PromptCatalog
from owlbank.pii import FakePiiRedactor, PiiGuard
from owlbank.policy import Policy
from owlbank.segments import SegmentLog
from tac_bridge.bridge import Bridge
from tests.component.test_bridge_fakes import (
    CALL_SID,
    NOW,
    SEALER,
    FakeAgent,
    FakeBackend,
    _inline_io,
    load_json,
    say,
    seeded_store,
    write_call_state,
)

CATALOG = PromptCatalog.from_dict(load_json("prompt_catalog.json"))
POLICY = Policy.from_dict(load_json("policy.json"))

values = st.lists(st.text(alphabet="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJ", min_size=4,
                          max_size=9).filter(lambda v: any(c.isdigit() for c in v)),
                  min_size=1, max_size=3, unique_by=segments._norm_text)
# A digit in every value: a letters-only value could be part of a placeholder the scrub writes
# ("dact" in "[redacted]"), a limit of ``segments.redact_text`` that is not the bridge's.
filler = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ,.-?'ÁéñÜ", max_size=20)


@st.composite
def caller_texts(draw: st.DrawFn, vals: list[str]) -> str:
    """Words around the values, a value sometimes spaced, hyphenated or upper-cased."""
    parts = [draw(filler)]
    for v in draw(st.lists(st.sampled_from(vals), max_size=4)):
        style = draw(st.sampled_from(["plain", "spaced", "hyphen", "upper"]))
        parts.append({"plain": v, "spaced": " ".join(v), "hyphen": "-".join(v),
                      "upper": v.upper()}[style])
        parts.append(draw(filler))
    return " ".join(parts)


def _run(caller: str, sealed: list[str]) -> tuple[Any, str, list[dict[str, Any]]]:
    sensitive.forget(CALL_SID)
    store = seeded_store()
    write_call_state(store)
    for i, value in enumerate(sealed):
        sensitive.remember(store, SEALER, CALL_SID, value, slot=f"slot{i}", now=lambda: NOW)
    sensitive.forget(CALL_SID)  # the front door's process, not this one
    journal = SegmentLog(store, now=lambda: NOW)
    agent = FakeAgent([say(caller)])
    bridge = Bridge(store=store, agent=agent, backend=FakeBackend(), memory=None, catalog=CATALOG,
                    policy=POLICY, pii=PiiGuard.for_policy(FakePiiRedactor(), POLICY), journal=journal,
                    sealer=SEALER, now=lambda: NOW, io=_inline_io)
    context = SimpleNamespace(call_sid=CALL_SID, conversation_id="conv_cpm13b", metadata={},
                              profile_id=None, profile=None, pending_handoff_data=None)

    async def go() -> str:
        return "".join([c async for c in bridge.run_turn(context, caller)])

    try:
        heard = asyncio.run(go())
        return agent.calls, heard, journal.read(CALL_SID)
    finally:
        sensitive.forget(CALL_SID)


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data(), vals=values)
def test_cpm13b_agent_input_is_unchanged_and_the_journal_copy_is_scrubbed(
        data: st.DataObject, vals: list[str]) -> None:
    caller = data.draw(caller_texts(vals))
    plain_calls, plain_heard, _ = _run(caller, [])
    calls, heard, events = _run(caller, vals)
    assert [c["user_text"] for c in calls] == [c["user_text"] for c in plain_calls]
    assert calls[0]["user_text"] == caller
    assert heard == plain_heard
    journal_text = segments._norm_text(" ".join(str(e.get("text", "")) for e in events
                                                if e["type"] == "ai_turn"))
    for v in vals:
        assert segments._norm_text(v) not in journal_text
