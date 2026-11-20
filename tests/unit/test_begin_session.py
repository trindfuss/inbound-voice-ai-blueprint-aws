"""``Bridge.on_setup`` calls the agent's ``begin_session(call_sid, history=...)``
once, in the background (after the session-ownership decision). A slow hook never delays
setup (and so the greeting or resume it starts); a failing one is logged with ``log_caught`` and
never reaches the caller. A client without the hook is skipped."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

from tac_bridge.bridge import Bridge

CALL = "CA" + "3" * 32


class _Agent:
    def __init__(self, *, gate: asyncio.Event | None = None, fail: bool = False) -> None:
        self.sessions: list[str] = []
        self.histories: list[str] = []
        self.finished: list[str] = []
        self.gate, self.fail = gate, fail

    async def begin_session(self, session_id: str, *, history: str = "reload") -> None:
        self.sessions.append(session_id)
        self.histories.append(history)
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise RuntimeError("warm-up failed")
        self.finished.append(session_id)


def _bridge(agent: Any) -> Bridge:
    bridge = Bridge.__new__(Bridge)  # only the pieces on_setup touches
    bridge._agent = agent
    bridge._tasks = set()
    bridge._session_starts = {}
    bridge._speech, bridge._scripted = {}, {}  # the silence watch: none in these tests
    bridge.instance_id = "proc-1"
    bridge.setups = 0

    async def io(fn: Any, *a: Any) -> Any:
        return fn(*a)

    def load(_sid: str) -> None:
        bridge.setups += 1
        return  # no call state: nothing to resume

    def claim(_sid: str, instance_id: str) -> str:
        assert instance_id == "proc-1"
        return "new"  # no owner yet

    bridge._io = io
    bridge._calls = SimpleNamespace(load=load, claim_agent_owner=claim)
    return bridge


def test_begin_session_runs_once_per_setup() -> None:
    agent = _Agent()

    async def go() -> None:
        bridge = _bridge(agent)
        await bridge.on_setup(CALL, {"callSid": CALL})
        await bridge.wait_idle()
        assert bridge.setups == 1

    asyncio.run(go())
    assert agent.sessions == [CALL]
    assert agent.histories == ["new"]
    assert agent.finished == [CALL]


def test_no_call_sid_no_begin_session() -> None:
    agent = _Agent()
    asyncio.run(_bridge(agent).on_setup(None))
    assert agent.sessions == []


def test_slow_begin_session_does_not_delay_setup() -> None:
    async def go() -> tuple[_Agent, Bridge]:
        gate = asyncio.Event()  # never set until setup has returned
        agent = _Agent(gate=gate)
        bridge = _bridge(agent)
        await asyncio.wait_for(bridge.on_setup(CALL), timeout=1.0)
        assert bridge.setups == 1  # the setup path (state read, resume) ran
        assert agent.finished == []  # while begin_session is still pending
        assert len(bridge._tasks) == 1  # held, so it isn't garbage-collected
        gate.set()
        await bridge.wait_idle()
        return agent, bridge

    agent, bridge = asyncio.run(go())
    assert agent.finished == [CALL]
    assert not bridge._tasks


def test_failing_begin_session_is_logged_not_raised(caplog: Any) -> None:
    agent = _Agent(fail=True)
    caplog.set_level(logging.DEBUG, logger="tac_bridge.bridge")

    async def go() -> None:
        bridge = _bridge(agent)
        await bridge.on_setup(CALL)
        await bridge.wait_idle()

    asyncio.run(go())  # no exception
    assert agent.sessions == [CALL]
    rec = [r for r in caplog.records if getattr(r, "operation", None) == "begin_session"]
    assert rec and rec[0].decision == "ignored_best_effort"
    assert getattr(rec[0], "error_class", None) == "RuntimeError"


def test_client_without_the_hook_is_skipped() -> None:
    async def go() -> int:
        bridge = _bridge(SimpleNamespace())  # no begin_session attribute
        await bridge.on_setup(CALL)
        assert not bridge._tasks
        return bridge.setups

    assert asyncio.run(go()) == 1
