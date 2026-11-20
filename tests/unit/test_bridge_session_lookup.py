"""The bridge's CallSid lookup prefers the TAC session with a registered WebSocket,
else the newest, instead of TAC's ``get_conversation_session_by_call_sid`` (the oldest match).

The stale case: a call's earlier session is still in TAC's dict (its ``CLOSED`` webhook hasn't won
the race yet) and has no socket; the newer session for the same CallSid has one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest
from tac.models.session import ConversationSession

from tac_bridge.bridge import Bridge, pick_session, tac_sessions_for_call, tac_stream_state
from tests.component.test_bridge_fakes import CALL_SID, make_rig

OTHER_SID = "CA" + "e" * 32


def _session(conv_id: str, call_sid: str = CALL_SID) -> ConversationSession:
    return ConversationSession(conversation_id=conv_id, call_sid=call_sid, channel="VOICE")


# --------------------------------------------------------------------------- TAC's shape (fails loudly)


def test_tac_voice_channel_still_keeps_sessions_in_an_ordered_conversations_dict() -> None:
    """``tac_sessions_for_call`` reads TAC's private ``_conversations``. If this fails, TAC changed its
    shape: update ``tac_sessions_for_call`` (the only reader) before upgrading."""
    voice = make_rig().voice
    conversations = getattr(voice, "_conversations", None)
    assert isinstance(conversations, dict), "TAC VoiceChannel._conversations is gone or not a dict"
    conversations["conv_old"] = _session("conv_old")
    conversations["conv_new"] = _session("conv_new")
    conversations["conv_other"] = _session("conv_other", OTHER_SID)
    # TAC's own lookup returns the oldest: the behaviour this task works around.
    assert voice.get_conversation_session_by_call_sid(CALL_SID).conversation_id == "conv_old"
    assert [s.conversation_id for s in tac_sessions_for_call(voice, CALL_SID)] == ["conv_old", "conv_new"]
    assert callable(getattr(voice, "get_websocket", None))
    assert voice.get_websocket("conv_new") is None


def test_tac_still_keeps_a_cancellable_stream_task_per_conversation() -> None:
    """``tac_stream_state`` reads ``VoiceChannel._provider.session_manager`` (AI keypad shortcuts:
    a keypad turn is the conversation's stream task, so a barge-in or a prompt cancels it). If this
    fails, TAC changed its shape: update ``tac_stream_state`` (the only reader) before upgrading."""
    voice = make_rig().voice
    manager = getattr(getattr(voice, "_provider", None), "session_manager", None)
    assert manager is not None, "TAC VoiceChannel._provider.session_manager is gone"
    assert tac_stream_state(voice, "conv_x") is None  # no state before the socket's setup
    created = manager.get_or_create_session("conv_x")
    state = tac_stream_state(voice, "conv_x")
    assert state is created and hasattr(state, "stream_task")

    async def go() -> bool:
        state.stream_task = asyncio.create_task(asyncio.sleep(5))
        await state.cancel_stream_task()
        return bool(state.stream_task.cancelled())

    assert asyncio.run(go())
    assert tac_stream_state(SimpleNamespace(), "conv_x") is None  # a fake channel: no state


def test_a_channel_without_the_dict_falls_back_to_the_public_lookup() -> None:
    one = SimpleNamespace(conversation_id="conv_1", call_sid=CALL_SID)
    voice = SimpleNamespace(get_conversation_session_by_call_sid=lambda sid: one if sid == CALL_SID else None)
    assert tac_sessions_for_call(voice, CALL_SID) == [one]
    assert tac_sessions_for_call(voice, OTHER_SID) == []


# --------------------------------------------------------------------------- the choice


def test_pick_prefers_the_session_with_a_socket_else_the_newest() -> None:
    old, new = _session("conv_old"), _session("conv_new")
    assert pick_session([old, new], lambda s: s is new) == (new, True)
    assert pick_session([old, new], lambda s: s is old) == (old, True)
    assert pick_session([old, new], lambda _s: False) == (new, False)
    assert pick_session([old], lambda _s: False) == (old, False)  # one session: unchanged
    assert pick_session([], lambda _s: True) == (None, False)


class _Voice:
    """A TAC-shaped channel: an ordered ``_conversations`` dict and a socket registry."""

    def __init__(self, sessions: list[Any], sockets: set[str]) -> None:
        self._conversations: Mapping[str, Any] = {s.conversation_id: s for s in sessions}
        self.sockets = sockets
        self.sent: list[str] = []

    def get_conversation_session_by_call_sid(self, call_sid: str) -> Any:
        return next((s for s in self._conversations.values() if s.call_sid == call_sid), None)

    def get_websocket(self, conversation_id: str) -> Any:
        return object() if conversation_id in self.sockets else None

    async def send_response(self, conversation_id: str, _gen: Any) -> None:
        if conversation_id in self.sockets:  # TAC drops a send with no registered socket
            self.sent.append(conversation_id)


def _bridge(voice: _Voice, wait_s: float = 1.0) -> Bridge:
    bridge = Bridge.__new__(Bridge)  # only the pieces _resume_on_setup touches
    bridge.voice = voice  # type: ignore[assignment]
    bridge._cfg = SimpleNamespace(session_wait_s=wait_s, session_poll_s=0.001)  # type: ignore[assignment]
    bridge._speech = {}  # _send tracks a session's sends here (none in this test)
    bridge.run_turn = lambda context, message: None  # type: ignore[assignment,method-assign,misc]
    return bridge


def test_resume_uses_the_live_session_not_the_stale_older_one() -> None:
    voice = _Voice([_session("conv_stale"), _session("conv_live")], sockets={"conv_live"})
    asyncio.run(_bridge(voice)._resume_on_setup(CALL_SID))
    assert voice.sent == ["conv_live"]


def test_resume_follows_a_newer_session_whose_socket_registers_while_it_waits() -> None:
    """The stale session exists at once; the new one's socket lands a few polls later."""
    voice = _Voice([_session("conv_stale"), _session("conv_live")], sockets=set())

    async def go() -> None:
        resume = asyncio.create_task(_bridge(voice)._resume_on_setup(CALL_SID))
        await asyncio.sleep(0.02)
        voice.sockets.add("conv_live")
        await resume

    asyncio.run(go())
    assert voice.sent == ["conv_live"]


def test_resume_with_one_session_is_unchanged() -> None:
    voice = _Voice([_session("conv_1")], sockets={"conv_1"})
    asyncio.run(_bridge(voice)._resume_on_setup(CALL_SID))
    assert voice.sent == ["conv_1"]


def test_resume_defers_when_no_session_gets_a_socket(caplog: pytest.LogCaptureFixture) -> None:
    voice = _Voice([_session("conv_stale"), _session("conv_new")], sockets=set())
    asyncio.run(_bridge(voice, wait_s=0.03)._resume_on_setup(CALL_SID))
    assert voice.sent == []
    assert any(getattr(r, "decision", None) == "defer_to_first_prompt" for r in caplog.records)


def test_session_for_call_never_returns_another_calls_session() -> None:
    voice = _Voice([_session("conv_other", OTHER_SID)], sockets={"conv_other"})
    assert _bridge(voice)._session_for_call(CALL_SID) == (None, False)


def test_silence_uses_the_live_session_not_the_stale_older_one() -> None:
    """End to end on real TAC: a stale session for the call sits first in TAC's dict. The greeting's
    silence repeat must reach the caller (sent on the live session, not dropped on the stale one)."""
    from tests.component.test_silence_watch import GREETING, rig_with, texts
    from tests.component.test_bridge_fakes import wait_for

    rig = rig_with()
    rig.voice._conversations["conv_stale"] = _session("conv_stale")  # inserted before the live one

    async def go() -> list[str]:
        sim = rig.sim()
        await sim.connect()
        await wait_for(lambda: len(texts(sim)) >= 1, 4)
        sent = texts(sim)
        await sim.hangup()
        return sent

    assert asyncio.run(go()) == [GREETING]
