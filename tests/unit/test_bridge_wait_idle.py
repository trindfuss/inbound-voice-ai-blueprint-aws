"""``Bridge.wait_idle`` (a test helper) returns once background tasks finish, and raises ``TimeoutError``
instead of hanging when one never does."""

from __future__ import annotations

import asyncio

import pytest

from tac_bridge.bridge import Bridge


def _bridge() -> Bridge:
    bridge = Bridge.__new__(Bridge)
    bridge._tasks = set()
    return bridge


def _spawn(bridge: Bridge, coro) -> asyncio.Task:
    task = asyncio.ensure_future(coro)
    bridge._tasks.add(task)
    task.add_done_callback(bridge._tasks.discard)
    return task


def test_returns_when_tasks_finish() -> None:
    async def go() -> None:
        bridge = _bridge()
        _spawn(bridge, asyncio.sleep(0.01))
        await bridge.wait_idle(timeout=2)
        assert not bridge._tasks

    asyncio.run(go())


def test_a_task_that_never_finishes_raises_instead_of_hanging() -> None:
    async def go() -> None:
        bridge = _bridge()
        hung = _spawn(bridge, asyncio.Event().wait())
        with pytest.raises(TimeoutError):
            await bridge.wait_idle(timeout=0.05)
        assert not hung.done()  # not cancelled by the timeout
        hung.cancel()
        await asyncio.gather(hung, return_exceptions=True)

    asyncio.run(go())


def test_a_failed_task_does_not_raise() -> None:
    async def boom() -> None:
        raise RuntimeError("boom")

    async def go() -> None:
        bridge = _bridge()
        _spawn(bridge, boom())
        await bridge.wait_idle(timeout=2)

    asyncio.run(go())
