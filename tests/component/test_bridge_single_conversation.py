"""SingleConversationGuard keeps one ACTIVE conversation per call so TAC never polls in vain."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tac_bridge.bridge import SingleConversationGuard

SID = "CA" + "7" * 32


@dataclass
class Conv:
    id: str
    created_at: str | None


class FakeOrchestrator:
    def __init__(self, conversations: list[Conv], fail_close: bool = False) -> None:
        self.conversations = conversations
        self.closed: list[str] = []
        self.fail_close = fail_close
        self.other_calls = 0

    async def list_conversations(self, **kwargs: Any) -> list[Conv]:
        return [c for c in self.conversations if c.id not in self.closed]

    async def update_conversation(self, conversation_id: str, status: str) -> Conv:
        if self.fail_close:
            raise RuntimeError("twilio is down")
        assert status == "CLOSED"
        self.closed.append(conversation_id)
        return Conv(conversation_id, None)

    def get_configuration(self) -> str:
        self.other_calls += 1
        return "cfg"


def _list(guard: SingleConversationGuard, **kw: Any) -> list[Conv]:
    return asyncio.run(guard.list_conversations(**kw))


def test_more_than_one_active_keeps_the_newest_and_closes_the_rest(caplog):
    fake = FakeOrchestrator([Conv("conv_a", "2026-10-14T03:38:06Z"), Conv("conv_c", "2026-10-14T03:39:44Z"),
                             Conv("conv_b", "2026-10-14T03:38:34Z")])
    with caplog.at_level(logging.WARNING):
        out = _list(SingleConversationGuard(fake), channel_id=SID, status=["ACTIVE"])
    assert [c.id for c in out] == ["conv_c"]
    assert sorted(fake.closed) == ["conv_a", "conv_b"]
    assert any(getattr(r, "event", "") == "extra_conversations_closed" for r in caplog.records)


def test_one_or_none_passes_through_untouched():
    for convs in ([], [Conv("conv_a", "2026-10-14T03:38:06Z")]):
        fake = FakeOrchestrator(list(convs))
        assert _list(SingleConversationGuard(fake), channel_id=SID, status=["ACTIVE"]) == convs
        assert fake.closed == []


@pytest.mark.parametrize("kw", [{}, {"channel_id": SID}, {"status": ["ACTIVE"]},
                                {"channel_id": SID, "status": ["ACTIVE", "INACTIVE"]}])
def test_other_queries_are_not_deduplicated(kw):
    fake = FakeOrchestrator([Conv("conv_a", "1"), Conv("conv_b", "2")])
    assert len(_list(SingleConversationGuard(fake), **kw)) == 2 and fake.closed == []


def test_a_failed_close_still_returns_the_newest():
    fake = FakeOrchestrator([Conv("conv_a", "1"), Conv("conv_b", "2")], fail_close=True)
    assert [c.id for c in _list(SingleConversationGuard(fake), channel_id=SID, status=["ACTIVE"])] == ["conv_b"]


def test_same_second_ties_break_by_id():
    fake = FakeOrchestrator([Conv("conv_01m4w7v2", "2026-10-14T03:38:34Z"), Conv("conv_01m4w7x7", "2026-10-14T03:38:34Z")])
    assert [c.id for c in _list(SingleConversationGuard(fake), channel_id=SID, status=["ACTIVE"])] == ["conv_01m4w7x7"]


def test_other_attributes_delegate():
    fake = FakeOrchestrator([])
    assert SingleConversationGuard(fake).get_configuration() == "cfg" and fake.other_calls == 1


@settings(max_examples=100, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 10**6), st.text("abcdef0123456789", min_size=1, max_size=8)),
                min_size=0, max_size=6, unique_by=lambda t: t[1]))
def test_property_exactly_one_survives_and_it_is_the_newest(rows):
    """For any set of ACTIVE conversations, the guard returns at most one, it is the maximum by
    (created_at, id), and every other one was closed."""
    convs = [Conv(f"conv_{i}", f"{ts:08d}") for ts, i in rows]
    fake = FakeOrchestrator(list(convs))
    out = _list(SingleConversationGuard(fake), channel_id=SID, status=["ACTIVE"])
    assert len(out) == min(1, len(convs))
    if len(convs) >= 2:
        newest = max(convs, key=lambda c: (c.created_at, c.id))
        assert out == [newest]
        assert sorted(fake.closed) == sorted(c.id for c in convs if c is not newest)
    else:
        assert fake.closed == []
