"""Clock once per call.

The agent's clock line sits in the system suffix, before the conversation history, so a line that
moved every minute broke the messages cache point. The bridge now sets it at the call's first turn
and rebuilds it only on a new bank-local day or a language change; a re-entry (a new session on the
same TAC context in this process) and the fallback prompt keep it; a later call on the same
context (a reused conversation) gets its own (the line is held with the call's SID). Accepted limit: the time of day drifts
during a long call.

The rig's clock is moved between turns (``bridge._now``); every time here is before the call
state's expiry (``NOW + 3600``). ``NOW`` is 2027-01-15 08:00 UTC, 03:00 in New York.
"""

from __future__ import annotations

import asyncio
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from owlbank import agent_prompt
from tac_bridge.bridge import META_KEY, CallView
from tests.component.test_bridge_fakes import (
    CALL_SID,
    NOW,
    Rig,
    clock_line_at,
    front_door_update,
    make_rig,
    rearm_nonce,
    say,
    spoken,
    write_call_state,
)

REPLY_S = 5.0
HOUR = 3600
#: 04:59 and 05:01 UTC: 23:59 on the 14th and 00:01 on the 15th in New York, one UTC date.
NY_BEFORE_MIDNIGHT, NY_AFTER_MIDNIGHT = NOW - 3 * HOUR - 60, NOW - 3 * HOUR + 60
#: 23:59 on the 14th and 00:01 on the 15th UTC: 18:59 and 19:01 in New York, one bank-local date.
UTC_BEFORE_MIDNIGHT, UTC_AFTER_MIDNIGHT = NOW - 8 * HOUR - 60, NOW - 8 * HOUR + 60


def rig_at(t: list[float], turns: int, **state: Any) -> Rig:
    rig = make_rig(script=[say(f"Answer {n}.") for n in range(turns)])
    rig.bridge._now = lambda: t[0]  # noqa: SLF001
    write_call_state(rig.store, **state)
    return rig


def clock_of(call: dict[str, Any]) -> str:
    return str(call["system_suffix"]).splitlines()[-1]


async def _turns(rig: Rig, t: list[float], times: list[float], between: Any = None) -> None:
    sim = rig.sim()
    await sim.connect()
    for n, at in enumerate(times):
        if between is not None:
            between(n)
        t[0] = at
        await sim.prompt(f"question {n}", timeout=REPLY_S)
    await sim.hangup()


def converse(times: list[float], between: Any = None, **state: Any) -> tuple[Rig, list[dict[str, Any]]]:
    t = [times[0]]
    rig = rig_at(t, len(times), **state)
    asyncio.run(_turns(rig, t, times, between))
    assert len(rig.agent.calls) == len(times)
    return rig, rig.agent.calls


@pytest.mark.parametrize("gap", [60, HOUR - 1], ids=["a_minute", "an_hour"])
def test_turns_on_one_bank_local_day_send_the_same_suffix(gap: int) -> None:
    start = NOW - HOUR  # 02:00 in New York
    rig, calls = converse([start, start + gap])
    assert calls[0]["system_suffix"] == calls[1]["system_suffix"]
    assert clock_of(calls[1]) == clock_line_at(rig.bridge, "en-US", start)  # the first turn's
    assert clock_line_at(rig.bridge, "en-US", start + gap) != clock_of(calls[1])  # not vacuous


def test_a_turn_after_bank_local_midnight_gets_the_new_dates_line() -> None:
    rig, calls = converse([NY_BEFORE_MIDNIGHT, NY_AFTER_MIDNIGHT])
    assert clock_of(calls[0]) == clock_line_at(rig.bridge, "en-US", NY_BEFORE_MIDNIGHT)
    assert "January 14" in clock_of(calls[0])
    assert clock_of(calls[1]) == clock_line_at(rig.bridge, "en-US", NY_AFTER_MIDNIGHT)
    assert "January 15" in clock_of(calls[1]) and "12:01 AM" in clock_of(calls[1])


def test_utc_midnight_is_not_a_new_bank_local_day() -> None:
    rig, calls = converse([UTC_BEFORE_MIDNIGHT, UTC_AFTER_MIDNIGHT])
    assert calls[0]["system_suffix"] == calls[1]["system_suffix"]
    assert clock_of(calls[1]) == clock_line_at(rig.bridge, "en-US", UTC_BEFORE_MIDNIGHT)


def test_a_language_change_mid_call_gives_that_languages_line() -> None:
    start = NOW - HOUR
    times = [start, start + 60, start + 120, start + 180]
    rig: list[Rig] = []

    def between(n: int) -> None:
        if n == 2:
            front_door_update(rig[0].store, language="es-US")

    t = [times[0]]
    rig.append(rig_at(t, len(times)))
    asyncio.run(_turns(rig[0], t, times, between))
    calls = rig[0].agent.calls
    assert clock_of(calls[0]) == clock_of(calls[1]) == clock_line_at(rig[0].bridge, "en-US", start)
    assert clock_of(calls[2]) == clock_of(calls[3]) == clock_line_at(rig[0].bridge, "es-US", start + 120)
    assert clock_of(calls[2]).startswith("Fecha y hora actuales del banco")


def test_a_reentry_keeps_the_calls_line() -> None:
    start = NOW - HOUR
    t = [start]
    rig = rig_at(t, 2)

    async def go() -> None:
        await _turns(rig, t, [start])
        rearm_nonce(rig.store)  # the front door's reconnect after a hand-off
        await _turns(rig, t, [start + 25 * 60])

    asyncio.run(go())
    first, second = rig.agent.calls
    assert len(rig.agent.sessions) == 2  # two sessions of the call, one TAC context
    assert first["system_suffix"] == second["system_suffix"]
    assert clock_of(second) == clock_line_at(rig.bridge, "en-US", start)
    assert clock_line_at(rig.bridge, "en-US", start + 25 * 60) != clock_of(second)  # not vacuous


@pytest.mark.parametrize("failing_turns", [{1}, {0, 1}, {0}], ids=["second", "both", "first"])
def test_the_fallback_prompt_uses_the_calls_line(failing_turns: set[int]) -> None:
    start = NOW - HOUR
    t = [start]
    rig = rig_at(t, 2)
    real = rig.bridge._system_prompt  # noqa: SLF001
    turn = [0]

    async def sometimes_failing(context: Any, view: Any) -> Any:
        n, turn[0] = turn[0], turn[0] + 1
        if n in failing_turns:
            raise RuntimeError("profile read failed")
        return await real(context, view)

    rig.bridge._system_prompt = sometimes_failing  # type: ignore[method-assign]  # noqa: SLF001
    asyncio.run(_turns(rig, t, [start, start + 600]))
    first, second = rig.agent.calls
    assert clock_of(first) == clock_of(second) == clock_line_at(rig.bridge, "en-US", start)


def test_a_later_call_on_the_same_context_gets_a_fresh_line() -> None:
    """A reused conversation (``OWLBANK_REUSE_CONVERSATION``): two calls share one TAC context, 20
    minutes apart on one bank-local day; the second call's line is its own, not the first's."""
    start = NOW - HOUR
    t = [start]
    rig = rig_at(t, 0)
    context = SimpleNamespace(metadata={}, profile_id=None, profile=None)
    other = "CA" + "9" * 32

    def line_for(call_sid: str) -> str:
        view = CallView(call_sid=call_sid, language="en-US", level=0)
        return asyncio.run(rig.bridge._system_prompt(context, view))[2]  # noqa: SLF001

    first = line_for(CALL_SID)
    t[0] = start + 20 * 60
    assert line_for(CALL_SID) == first  # the same call: kept
    second = line_for(other)
    assert second == clock_line_at(rig.bridge, "en-US", start + 20 * 60) != first
    assert context.metadata[META_KEY]["clock"]["call_sid"] == other


def test_a_later_call_on_the_same_context_renders_its_own_per_call_part(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prompt parts are held per call too: a reused conversation's next call, with the same
    caller, language and offer, renders its per-call part again (a fresh Memory read) rather than
    reusing the first call's."""
    rig = rig_at([NOW - HOUR], 0)
    context = SimpleNamespace(metadata={}, profile_id=None, profile=None)
    real, renders = agent_prompt.render_per_call, []

    def counting(*args: Any, **kwargs: Any) -> str:
        renders.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(agent_prompt, "render_per_call", counting)

    def parts_for(call_sid: str) -> None:
        view = CallView(call_sid=call_sid, language="en-US", level=0)
        asyncio.run(rig.bridge._system_prompt(context, view))  # noqa: SLF001

    parts_for(CALL_SID)
    parts_for(CALL_SID)
    assert len(renders) == 1  # the same call: cached
    parts_for("CA" + "9" * 32)
    assert len(renders) == 2


@pytest.mark.parametrize("held", [None, MappingProxyType({})], ids=["none", "read_only"])
def test_a_fallback_turn_on_metadata_that_cant_hold_the_line_still_answers(held: Any) -> None:
    start = NOW - HOUR
    t = [start]
    rig = rig_at(t, 2)

    async def failing(context: Any, _view: Any) -> Any:
        context.metadata[META_KEY] = held  # what made the build fail
        raise RuntimeError("metadata unusable")

    rig.bridge._system_prompt = failing  # type: ignore[method-assign]  # noqa: SLF001

    async def go() -> list[Any]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("question 0", timeout=REPLY_S)
        t[0] = start + 20 * 60
        frames += await sim.prompt("question 1", timeout=REPLY_S)
        await sim.hangup()
        return frames

    frames = asyncio.run(go())
    assert "Answer 0." in spoken(frames) and "Answer 1." in spoken(frames)
    first, second = rig.agent.calls
    assert clock_of(first) == clock_line_at(rig.bridge, "en-US", start)
    assert clock_of(second) == clock_line_at(rig.bridge, "en-US", start + 20 * 60)  # fresh each turn, not kept
