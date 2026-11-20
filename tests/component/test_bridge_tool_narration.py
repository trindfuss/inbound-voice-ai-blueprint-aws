"""No narration of tool use, through the bridge over real TAC.

The caller hears only text from a model message that has no tool use. Text from a
message that also calls a tool is not spoken, not journaled as ``ai_turn``, and is
logged as ``ai_text_suppressed`` with its length only. The steps are raw
ConverseStream events of one model message, played through the real Strands client
(``tests.strands_parity``).
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from tests.component.masking_fixtures import unmasked_policy_doc
from tests.component.test_bridge_fakes import (
    FakeBackend,
    end_frame,
    h_msg,
    h_reasoning,
    h_text,
    h_tool,
    make_rig,
    run,
    spoken,
    write_call_state,
)
from tests.logutil import lines, text_of

BRIDGE_LOG = "tac_bridge.bridge"


def converse(rig: Any, prompt: str) -> list[dict[str, Any]]:
    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt(prompt)
        await sim.hangup()
        return frames

    return run(go())


def suppressed(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [f for f in lines(caplog.records, BRIDGE_LOG) if f["event"] == "ai_text_suppressed"]


def ai_turn_texts(rig: Any) -> list[str]:
    return [e.get("text", "") for e in rig.events() if e["type"] == "ai_turn"]


def test_text_then_tool_use_in_one_message_is_not_spoken(caplog: pytest.LogCaptureFixture) -> None:
    narration = "The caller now asks for a human. Escalating."
    rig = make_rig(script=[h_msg(h_text(narration),
                                 h_tool("escalate", "tu_esc", i=1, reason="caller_requested_human",
                                        summary="Caller asked for a person."))])
    write_call_state(rig.store)
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames = converse(rig, "Can I talk to a person?")
    assert spoken(frames) == ""
    assert end_frame(frames) is not None  # the escalate handoff still happens
    assert ai_turn_texts(rig) == []
    (line,) = suppressed(caplog)
    assert line["chars"] == len(narration)
    assert narration not in text_of(caplog.records)


def test_tool_hop_then_text_only_message_is_spoken(caplog: pytest.LogCaptureFixture) -> None:
    narration = "The tool result includes a note, so I'll act on that now."
    backend = FakeBackend({"list_recent_transactions": [{"account": "checking", "transactions": []}]})
    rig = make_rig(backend=backend, policy_doc=unmasked_policy_doc(), script=[  # masking phrases off
        h_msg(h_text(narration), h_tool("list_recent_transactions", "tu_tx", i=1, account="checking", limit=3)),
        h_msg(h_text("You have no recent transactions on checking.")),
    ])
    write_call_state(rig.store)
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames = converse(rig, "Any recent transactions?")
    assert spoken(frames) == "You have no recent transactions on checking."
    assert len(backend.calls) == 1
    assert ai_turn_texts(rig) == ["You have no recent transactions on checking."]
    assert [line["chars"] for line in suppressed(caplog)] == [len(narration)]
    assert narration not in text_of(caplog.records)


def test_text_only_message_is_spoken_once(caplog: pytest.LogCaptureFixture) -> None:
    rig = make_rig(script=[h_msg(h_text("We open at nine."))])
    write_call_state(rig.store)
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames = converse(rig, "When do you open?")
    assert spoken(frames) == "We open at nine."
    assert ai_turn_texts(rig) == ["We open at nine."]
    assert suppressed(caplog) == []


def test_several_text_blocks_are_spoken_in_order() -> None:
    rig = make_rig(script=[h_msg(h_text("First. ", 0), h_reasoning("hidden thought", 1),
                                 h_text("Second. ", 2), h_text("Third.", 3))])
    write_call_state(rig.store)
    frames = converse(rig, "Tell me three things.")
    assert spoken(frames) == "First. Second. Third."  # reasoning is never spoken
    assert ai_turn_texts(rig) == ["First. Second. Third."]
