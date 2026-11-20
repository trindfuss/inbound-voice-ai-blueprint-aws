"""For any TAC sessions for a CallSid (mixed with other calls' sessions, in any
order) and any subset holding a registered WebSocket, the bridge's lookup returns a session with a
socket if one exists, else the last inserted; never a session of another CallSid, and never a
socketless (closed) one while a session with a socket exists."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from tac_bridge.bridge import Bridge

CALL = "CA" + "a" * 32
OTHERS = ["CA" + "b" * 32, "CA" + "c" * 32]

# Each entry: (belongs to CALL or which other call, has a socket)
entries = st.lists(st.tuples(st.sampled_from([CALL, *OTHERS]), st.booleans()), max_size=12)


class _Voice:
    def __init__(self, spec: list[tuple[str, bool]]) -> None:
        self._conversations = {
            f"conv_{i}": SimpleNamespace(conversation_id=f"conv_{i}", call_sid=sid) for i, (sid, _) in enumerate(spec)}
        self.sockets = {f"conv_{i}" for i, (_, sock) in enumerate(spec) if sock}

    def get_conversation_session_by_call_sid(self, call_sid: str) -> Any:
        return next((s for s in self._conversations.values() if s.call_sid == call_sid), None)

    def get_websocket(self, conversation_id: str) -> Any:
        return object() if conversation_id in self.sockets else None


@given(entries)
def test_lookup_prefers_a_socket_else_the_newest_and_stays_on_the_call(spec: list[tuple[str, bool]]) -> None:
    voice = _Voice(spec)
    bridge = Bridge.__new__(Bridge)
    bridge.voice = voice  # type: ignore[assignment]
    chosen, has_socket = bridge._session_for_call(CALL)

    mine = [s for s in voice._conversations.values() if s.call_sid == CALL]
    live = [s for s in mine if s.conversation_id in voice.sockets]
    if not mine:
        assert (chosen, has_socket) == (None, False)
        return
    assert chosen.call_sid == CALL  # never another call's session
    assert has_socket == (chosen.conversation_id in voice.sockets)
    if live:
        assert has_socket  # never a socketless one while a live one exists
        assert chosen is live[-1]
    else:
        assert chosen is mine[-1]  # the last inserted
