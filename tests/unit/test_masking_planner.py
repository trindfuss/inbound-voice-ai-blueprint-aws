"""MaskPlanner unit cases.

The API: :meth:`MaskPlanner.on_tool_calls` takes the time the calls reached the bridge
and returns the phrase to *schedule* (a :class:`ScheduledPhrase` with its delay and deadline); the
phrase is spoken by :meth:`MaskPlanner.on_deadline` only if no answer text came first.
"""

from __future__ import annotations

import pytest

from owlbank.masking_config import ToolMask
from owlbank.policy import MASK_GENERIC_KEYS
from tac_bridge.masking import (
    SKIP_ANSWER_STARTED,
    SKIP_ONE_PER_TURN,
    SKIP_SUPERSEDED,
    SKIP_TOOL_OFF,
    EndMode,
    MaskPlanner,
    speech_end,
)

MASKS = {"list_recent_transactions": ToolMask("mask.list_recent_transactions"),
         "change_address": ToolMask("mask.change_address", after_ms=250),
         "get_balance": ToolMask("mask.get_balance", handoff_wait_ms=900)}
HANDOFF = {"get_balance", "make_payment", "authenticate", "escalate"}


def planner(after_ms: int = 2000, handoff_wait_ms: int = 2500, end_mode: EndMode = "wait",
            chars_per_s: float = 15.0, generic_key: str = MASK_GENERIC_KEYS[0],
            tool_after_ms: int = 600) -> MaskPlanner:
    return MaskPlanner(after_ms=after_ms, handoff_wait_ms=handoff_wait_ms, end_mode=end_mode,
                       chars_per_s=chars_per_s, generic_key=generic_key, tool_after_ms=tool_after_ms,
                       tool_mask=MASKS.get, is_handoff=HANDOFF.__contains__)


def test_default_delay_and_deadline() -> None:
    p = planner()
    (s,) = p.on_tool_calls(["list_recent_transactions"], now=10.0)
    assert (s.tool, s.phrase_key, s.after_ms, s.source) == (
        "list_recent_transactions", "mask.list_recent_transactions", 600, "default")
    assert s.deadline == pytest.approx(10.6) and s.delay_s == pytest.approx(0.6)
    assert p.on_deadline(10.59) is None  # not due yet
    assert p.on_deadline(10.6) == s
    assert p.spoken_tool and p.pending is None


def test_tool_own_after_ms() -> None:
    (s,) = planner().on_tool_calls(["change_address"], now=0.0)
    assert (s.after_ms, s.source, s.deadline) == (250, "tool", pytest.approx(0.25))


def test_handoff_tool_is_due_at_once_and_never_anchored() -> None:
    p = planner()
    assert p.on_timer(spoken_anything=False) == MASK_GENERIC_KEYS[0]
    p.note_ack("One moment.", 5.0)
    (s,) = p.on_tool_calls(["get_balance"], now=5.1)
    assert (s.after_ms, s.source, s.deadline) == (0, "handoff", 5.1)


def test_ack_end_anchors_an_inline_phrase() -> None:
    """The later of tool call + delay and the acknowledgement's estimated end."""
    p = planner(chars_per_s=10.0)
    assert p.on_timer(spoken_anything=False) == MASK_GENERIC_KEYS[0]
    p.note_ack("x" * 20, 1.0)  # ends at 1.0 + 0.3 + 2.0 = 3.3
    assert p.ack_end == pytest.approx(speech_end("x" * 20, 1.0, 10.0)) == pytest.approx(3.3)
    (s,) = p.on_tool_calls(["list_recent_transactions"], now=1.5)  # 1.5 + 0.6 = 2.1 < 3.3
    assert s.deadline == pytest.approx(3.3)
    q = planner(chars_per_s=30.0)
    q.on_timer(spoken_anything=False)
    q.note_ack("x" * 3, 1.0)  # ends at 1.4
    (t,) = q.on_tool_calls(["list_recent_transactions"], now=1.5)
    assert t.deadline == pytest.approx(2.1)


def test_answer_text_cancels_the_pending_phrase() -> None:
    p = planner()
    (s,) = p.on_tool_calls(["list_recent_transactions"], now=0.0)
    assert p.on_text() == [s]
    assert p.pending is None and p.on_deadline(5.0) is None and not p.spoken_tool


def test_timer_then_tool_speaks_both() -> None:
    """A timed acknowledgement does not block the per-tool phrase."""
    p = planner(after_ms=1500)
    assert p.timer_s() == 1.5
    assert p.on_timer(spoken_anything=False) == MASK_GENERIC_KEYS[0]
    (s,) = p.on_tool_calls(["list_recent_transactions"], now=0.0)
    assert p.on_deadline(s.deadline) == s
    # one of each kind per turn
    assert p.on_timer(spoken_anything=False) is None
    assert p.on_tool_calls(["get_balance"], now=1.0) == []
    assert p.last_skips == [("get_balance", SKIP_ONE_PER_TURN)]


def test_timer_then_answer_text_blocks_the_tool_phrase() -> None:
    p = planner()
    assert p.on_timer(spoken_anything=False) == MASK_GENERIC_KEYS[0]
    p.on_text()
    assert p.on_tool_calls(["list_recent_transactions"], now=0.0) == []
    assert p.last_skips == [("list_recent_transactions", SKIP_ANSWER_STARTED)]


def test_on_timer_returns_the_planners_generic_key() -> None:
    for key in MASK_GENERIC_KEYS:
        assert planner(generic_key=key).on_timer(spoken_anything=False) == key


def test_text_already_spoken_gives_no_phrase() -> None:
    p = planner()
    assert p.on_timer(spoken_anything=True) is None
    assert p.on_tool_calls(["get_balance"], now=0.0, spoken_anything=True) == []
    q = planner()
    q.on_text()
    assert q.on_tool_calls(["get_balance"], now=0.0) == []


def test_two_masked_tools_give_one_phrase_for_the_first() -> None:
    p = planner()
    (s,) = p.on_tool_calls(["authenticate", "change_address", "list_recent_transactions"], now=0.0)
    assert s.tool == "change_address"
    assert p.last_skips == [("authenticate", SKIP_TOOL_OFF),
                            ("list_recent_transactions", SKIP_ONE_PER_TURN)]
    assert p.on_tool_calls(["list_recent_transactions"], now=0.1) == []  # one pending already
    assert p.on_timer(spoken_anything=False) is None


def test_unmasked_tool_gives_nothing() -> None:
    p = planner()
    assert p.on_tool_calls(["escalate"], now=0.0) == []
    assert p.last_skips == [("escalate", SKIP_TOOL_OFF)]
    assert p.on_tool_calls([], now=0.0) == []
    # a later masked call in the same turn may still be masked
    (s,) = p.on_tool_calls(["get_balance"], now=0.0)
    assert s.tool == "get_balance"


def test_a_handoff_supersedes_a_pending_inline_phrase() -> None:
    p = planner()
    (s,) = p.on_tool_calls(["list_recent_transactions"], now=0.0)
    (h,) = p.on_tool_calls(["get_balance"], now=0.2)
    assert p.last_skips == [("list_recent_transactions", SKIP_SUPERSEDED)]
    assert p.pending == h and p.on_deadline(0.2) == h and s != h


def test_discard_frees_the_slot() -> None:
    p = planner()
    (s,) = p.on_tool_calls(["list_recent_transactions"], now=0.0)
    p.discard(s)
    (t,) = p.on_tool_calls(["change_address"], now=0.1)
    assert t.tool == "change_address"


def test_end_turn_drops_the_pending_phrase() -> None:
    p = planner()
    (s,) = p.on_tool_calls(["list_recent_transactions"], now=0.0)
    assert p.end_turn() == [s] and p.end_turn() == []


def test_timer_not_spoken_after_a_tool_call() -> None:
    p = planner()
    assert p.on_tool_calls(["escalate"], now=0.0) == []
    assert p.on_timer(spoken_anything=False) is None


def test_after_ms_zero_turns_timer_off() -> None:
    p = planner(after_ms=0)
    assert p.timer_s() is None
    assert p.on_timer(spoken_anything=False) is None


def test_reset_starts_a_fresh_turn() -> None:
    p = planner()
    assert p.on_timer(spoken_anything=False) == MASK_GENERIC_KEYS[0]
    (s,) = p.on_tool_calls(["get_balance"], now=0.0)
    assert p.on_deadline(0.0) == s
    p.on_text()
    p.reset()
    # both kinds are available again, and answer text from the last turn no longer blocks them
    assert p.on_timer(spoken_anything=False) == MASK_GENERIC_KEYS[0]
    assert [x.tool for x in p.on_tool_calls(["get_balance"], now=1.0)] == ["get_balance"]


def test_phrase_for_still_works_without_tool_masks() -> None:
    p = MaskPlanner(after_ms=0, handoff_wait_ms=2500, end_mode="wait", chars_per_s=15.0,
                    phrase_for={"get_balance": "mask.get_balance"}.get, tool_after_ms=0)
    (s,) = p.on_tool_calls(["get_balance"], now=0.0)
    assert (s.phrase_key, s.after_ms, s.source) == ("mask.get_balance", 0, "default")





def test_per_tool_handoff_wait_ms() -> None:
    """The tool's own handoff_wait_ms, else the policy-wide one."""
    p = planner(handoff_wait_ms=2500)
    assert p.handoff_wait_ms_for("get_balance") == 900
    assert p.handoff_wait_ms_for("make_payment") == 2500
    assert p.handoff_wait_ms_for(None) == 2500
