"""Every ConversationRelay connect creates a new Orchestrator conversation for the CallSid, and TAC
insists on exactly one ACTIVE one ("Expected exactly 1 conversation ..., but found 3"; found at the first
real step-up call). The bridge closes the finished segment's conversation when its WebSocket ends."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from tac_bridge.bridge import BridgeConfig, Bridge, TACConversationCloser


class _OrchestratorClient:
    def __init__(self, active: list[str]) -> None:
        self.active = list(active)
        self.updated: list[tuple[str, str]] = []
        self.listed: list[dict[str, Any]] = []

    async def list_conversations(self, *, channel_id: str, status: list[str]) -> list[Any]:
        self.listed.append({"channel_id": channel_id, "status": status})
        return [SimpleNamespace(id=i) for i in self.active]

    async def update_conversation(self, conversation_id: str, status: str) -> None:
        self.updated.append((conversation_id, status))
        self.active.remove(conversation_id)


def test_closer_closes_every_active_conversation_for_the_call() -> None:
    client = _OrchestratorClient(["conv_a", "conv_b"])
    closer = TACConversationCloser(SimpleNamespace(conversation_orchestrator_client=client))
    assert asyncio.run(closer.close_active("CA" + "1" * 32)) == 2
    assert client.listed == [{"channel_id": "CA" + "1" * 32, "status": ["ACTIVE"]}]
    assert client.updated == [("conv_a", "CLOSED"), ("conv_b", "CLOSED")]


def test_closer_without_an_orchestrator_client_does_nothing() -> None:
    closer = TACConversationCloser(SimpleNamespace(conversation_orchestrator_client=None))
    assert asyncio.run(closer.close_active("CA" + "2" * 32)) == 0


class _Closer:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    async def close_active(self, call_sid: str) -> int:
        self.calls.append(call_sid)
        if self.fail:
            raise RuntimeError("twilio is down")
        return 1


def _bridge(closer: Any, *, reuse: bool = False) -> Bridge:
    bridge = Bridge.__new__(Bridge)  # only what session_end touches
    bridge._cfg = BridgeConfig(reuse_conversation=reuse)
    bridge._conversations = closer
    bridge._sessions = {}
    bridge._locks = {}
    bridge._tasks = set()
    bridge._session_starts = {}
    bridge._speech, bridge._scripted = {}, {}  # the silence watch: none in these tests
    bridge._first_session = {}
    bridge._keypad, bridge._keypad_wakes, bridge._keypad_tasks = {}, {}, {}  # AI keypad shortcuts: none here
    return bridge


def test_session_end_schedules_the_close_and_a_failure_never_escapes() -> None:
    async def run(fail: bool) -> _Closer:
        closer = _Closer(fail=fail)
        bridge = _bridge(closer)
        bridge.session_end("CA" + "3" * 32, "closed")
        await asyncio.gather(*bridge._tasks)
        return closer

    ok = asyncio.run(run(False))
    assert ok.calls == ["CA" + "3" * 32]
    failed = asyncio.run(run(True))  # must not raise
    assert failed.calls == ["CA" + "3" * 32]


def test_no_closer_means_no_task() -> None:
    bridge = _bridge(None)
    bridge.session_end("CA" + "4" * 32, "closed")
    assert not bridge._tasks


def test_reuse_conversation_keeps_the_conversation_open_at_segment_end() -> None:
    """With one conversation per call, a segment end must not close it (the next
    segment reconnects to it with ``conversationId``)."""
    closer = _Closer()
    bridge = _bridge(closer, reuse=True)
    bridge.session_end("CA" + "5" * 32, "closed")
    assert not bridge._tasks and closer.calls == []
