"""``_race_deadlines`` races an agent stream with named deadlines in one ``asyncio.wait``;
the stream is never cancelled or delayed, and closing the race (barge-in) settles cleanly.
``_race_timer`` is a thin wrapper and keeps its behaviour."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

from tac_bridge.agent_client import TextDelta, TurnEnd
from tac_bridge.bridge import _TIMER, _Deadlines, _Fired, _race_deadlines, _race_timer


class Stream:
    """``(delay_s, event)`` pairs; records whether it was closed and whether it ran to the end."""

    def __init__(self, items: list[tuple[float, Any]]) -> None:
        self.items = items
        self.closed = False
        self.finished = False

    async def gen(self) -> AsyncGenerator[Any, None]:
        try:
            for delay, ev in self.items:
                await asyncio.sleep(delay)
                yield ev
            self.finished = True
        finally:
            self.closed = True


def test_deadlines_fire_in_order_before_a_slow_event_and_the_stream_is_not_cancelled() -> None:
    async def go() -> list[tuple[Any, float]]:
        s = Stream([(0.15, TextDelta("late")), (0.0, TurnEnd("end_turn"))])
        d = _Deadlines()
        t0 = time.monotonic()
        d.arm("b", t0 + 0.08)
        d.arm("a", t0 + 0.03)
        out = []
        events = s.gen()
        async with aclosing(events), aclosing(_race_deadlines(events, d)) as race:
            async for ev in race:
                out.append((ev, time.monotonic() - t0))
        assert s.finished
        return out

    out = run(go())
    assert [e for e, _ in out] == [_Fired("a"), _Fired("b"), TextDelta("late"), TurnEnd("end_turn")]
    assert 0.03 <= out[0][1] < 0.08 <= out[1][1] < 0.15 <= out[2][1]


def test_a_deadline_armed_between_events_fires_and_a_cancelled_one_does_not() -> None:
    async def go() -> list[Any]:
        s = Stream([(0.0, TextDelta("a")), (0.12, TextDelta("b"))])
        d = _Deadlines()
        d.arm("never", time.monotonic() + 0.05)
        out: list[Any] = []
        events = s.gen()
        async with aclosing(events), aclosing(_race_deadlines(events, d)) as race:
            async for ev in race:
                out.append(ev)
                if ev == TextDelta("a"):
                    d.cancel("never")
                    d.arm("tool", time.monotonic() + 0.04)
        return out

    assert run(go()) == [TextDelta("a"), _Fired("tool"), TextDelta("b")]


def test_closing_the_race_while_waiting_settles_the_pending_read() -> None:
    async def go() -> tuple[list[Any], Stream]:
        s = Stream([(0.5, TextDelta("never read"))])
        d = _Deadlines()
        d.arm("ack", time.monotonic() + 0.02)
        events = s.gen()
        got: list[Any] = []
        async with aclosing(events), aclosing(_race_deadlines(events, d)) as race:
            got.append(await anext(race))  # the deadline; the read is still pending
        return got, s

    got, s = run(go())
    assert got == [_Fired("ack")] and s.closed and not s.finished


def test_no_deadline_passes_events_straight_through() -> None:
    async def go() -> list[Any]:
        s = Stream([(0.0, TextDelta("x")), (0.0, TurnEnd("end_turn"))])
        events = s.gen()
        async with aclosing(events), aclosing(_race_deadlines(events, _Deadlines())) as race:
            return [ev async for ev in race]

    assert run(go()) == [TextDelta("x"), TurnEnd("end_turn")]


def test_race_timer_alias_fires_once_and_is_disarmed_by_text() -> None:
    async def go(items: list[tuple[float, Any]], timeout: float | None) -> list[Any]:
        s = Stream(items)
        events = s.gen()
        async with aclosing(events), aclosing(_race_timer(events, timeout)) as race:
            return [ev async for ev in race]

    assert run(go([(0.08, TextDelta("x"))], 0.02)) == [_TIMER, TextDelta("x")]
    assert run(go([(0.0, TextDelta("x")), (0.08, TurnEnd("end_turn"))], 0.04)) == [
        TextDelta("x"), TurnEnd("end_turn")]
    assert run(go([(0.0, TextDelta("x"))], None)) == [TextDelta("x")]


def run(coro: Any) -> Any:
    return asyncio.run(coro)
