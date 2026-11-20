"""The resume on `setup` must wait for TAC to register the new WebSocket (found at the first real
step-up call: the send ran before the socket existed, TAC dropped it, and the caller heard silence)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from tac_bridge.bridge import Bridge


class _Voice:
    def __init__(self, ready_after: int | None) -> None:
        self.polls = 0
        self.ready_after = ready_after
        self.sent: list[str] = []
        self.context = SimpleNamespace(conversation_id="conv_1", call_sid="CA" + "0" * 32)

    def get_conversation_session_by_call_sid(self, _call_sid: str) -> Any:
        return self.context  # the stale session from the first connection exists at once

    def get_websocket(self, _conversation_id: str) -> Any:
        self.polls += 1
        if self.ready_after is not None and self.polls > self.ready_after:
            return object()
        return None

    async def send_response(self, conversation_id: str, _gen: Any) -> None:
        # What TAC does: no registered socket means the send is dropped.
        if self.get_websocket(conversation_id) is None:
            return
        self.sent.append(conversation_id)


def _bridge(voice: _Voice, wait_s: float = 2.0) -> Bridge:
    bridge = Bridge.__new__(Bridge)  # only the pieces _resume_on_setup touches
    bridge.voice = voice
    bridge._cfg = SimpleNamespace(session_wait_s=wait_s, session_poll_s=0.001)
    bridge.run_turn = lambda context, message: None  # never iterated by the fake
    bridge._send = lambda context, gen: voice.send_response(context.conversation_id, gen)
    return bridge


def test_resume_waits_for_the_new_socket_then_sends() -> None:
    voice = _Voice(ready_after=5)
    asyncio.run(_bridge(voice)._resume_on_setup(voice.context.call_sid))
    assert voice.sent == ["conv_1"]
    assert voice.polls > 5


def test_resume_defers_to_the_first_prompt_when_the_socket_never_appears(caplog) -> None:
    voice = _Voice(ready_after=None)
    asyncio.run(_bridge(voice, wait_s=0.05)._resume_on_setup(voice.context.call_sid))
    assert voice.sent == []
    assert any(getattr(r, "decision", None) == "defer_to_first_prompt" for r in caplog.records)
