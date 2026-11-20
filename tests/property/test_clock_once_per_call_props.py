"""Clock once per call (the clock line is not rebuilt per turn).

* **Clock once per call**: for any increasing sequence of turn times in a call, the clock line sent
  at each turn is :func:`local_now_line` at the first turn of the current bank-local date run: it
  changes exactly when the bank-local date changes, never otherwise. Through
  :meth:`Bridge._clock_line` for any bank zone (UTC dates that differ from the bank's included), and
  through whole turns of the rig (the ``system_suffix`` the client is handed).
* The same with the language and the call changing between turns: a new line exactly when the
  bank-local date, the language (the line is per language) or the call's SID changes (a later call
  on the same TAC context, a reused conversation, never gets an earlier call's line).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tac_bridge.bridge import Bridge
from tac_bridge.clock import local_now_line, zone
from tests.component.test_bridge_fakes import NOW, front_door_update, make_rig, say, write_call_state

LANGS = ("en-US", "es-US")
ZONES = ("America/New_York", "America/Los_Angeles", "UTC", "Asia/Kolkata", "Pacific/Kiritimati",
         "Pacific/Pago_Pago", "", "Not/AZone")
DAY = 86_400
REPLY_S = 5.0
CALL = "CA-1"

#: Gaps between turns: seconds (one minute), a few hours (one day or the next), days.
gaps = st.one_of(st.floats(0, 120, allow_nan=False), st.integers(0, 6 * 3600), st.integers(0, 3 * DAY))


def local_date(t: float, tz: str) -> Any:
    return datetime.fromtimestamp(t, UTC).astimezone(zone(tz)).date()


def expected_lines(turns: list[tuple[float, str]], tz: str, calls: list[str] | None = None) -> list[str]:
    """The oracle: each turn gets the line of the first turn of its run of equal (bank-local date,
    language, call SID)."""
    out: list[str] = []
    run_key: Any = None
    line = ""
    for i, (t, lang) in enumerate(turns):
        key = (local_date(t, tz), lang, CALL if calls is None else calls[i])
        if key != run_key:
            run_key, line = key, local_now_line(datetime.fromtimestamp(t, UTC), zone(tz), lang)
        out.append(line)
    return out


def stub(tz: str, t: list[float]) -> Bridge:
    bridge = Bridge.__new__(Bridge)  # only the pieces the clock line touches
    bridge._now = lambda: t[0]  # noqa: SLF001
    bridge._cfg = SimpleNamespace(timezone=tz)  # type: ignore[assignment]  # noqa: SLF001
    return bridge


def times_from(start: float, steps: list[float]) -> list[float]:
    out, t = [], start
    for g in steps:
        t += g
        out.append(t)
    return out


@settings(max_examples=400, deadline=None)
@given(start=st.floats(0, 4e9, allow_nan=False), steps=st.lists(gaps, min_size=1, max_size=12),
       tz=st.sampled_from(ZONES), lang=st.sampled_from(LANGS))
def test_clock_once_per_call_the_line_changes_exactly_with_the_bank_local_date(
        start: float, steps: list[float], tz: str, lang: str) -> None:
    times = times_from(start, steps)
    t = [times[0]]
    bridge, context = stub(tz, t), SimpleNamespace(metadata={})
    got = []
    for at in times:
        t[0] = at
        got.append(bridge._clock_line(context, CALL, lang))  # noqa: SLF001
    assert got == expected_lines([(at, lang) for at in times], tz)
    for i in range(1, len(times)):  # a new line exactly when the bank-local date moved
        assert (got[i] != got[i - 1]) == (local_date(times[i], tz) != local_date(times[i - 1], tz))


@settings(max_examples=400, deadline=None)
@given(start=st.floats(0, 4e9, allow_nan=False),
       steps=st.lists(st.tuples(gaps, st.sampled_from(LANGS), st.sampled_from(["CA-1", "CA-2", "CA-3"])),
                      min_size=1, max_size=12),
       tz=st.sampled_from(ZONES))
def test_clock_once_per_call_a_language_or_call_change_starts_a_new_line(
        start: float, steps: list[tuple[float, str, str]], tz: str) -> None:
    """One TAC context throughout: with a reused conversation, several calls share it."""
    turns = list(zip(times_from(start, [g for g, _, _ in steps]), [lang for _, lang, _ in steps], strict=True))
    calls = [sid for _, _, sid in steps]
    t = [turns[0][0]]
    bridge, context = stub(tz, t), SimpleNamespace(metadata={})
    got = []
    for (at, lang), sid in zip(turns, calls, strict=True):
        t[0] = at
        got.append(bridge._clock_line(context, sid, lang))  # noqa: SLF001
    assert got == expected_lines(turns, tz, calls)


#: Rig turns stay before the call state's expiry (``NOW + 3600``): the sequence ends by then.
rig_gaps = st.one_of(st.integers(0, 120), st.integers(0, 6 * 3600))


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(steps=st.lists(st.tuples(rig_gaps, st.sampled_from(LANGS)), min_size=1, max_size=5),
       back=st.integers(0, 2 * DAY))
def test_clock_once_per_call_through_the_bridges_turns(steps: list[tuple[int, str]], back: int) -> None:
    total = sum(g for g, _ in steps)
    times = times_from(NOW + 3000 - total - back, [g for g, _ in steps])
    langs = [lang for _, lang in steps]
    rig = make_rig(script=[say(f"Answer {n}.") for n in range(len(steps))])
    t = [times[0]]
    rig.bridge._now = lambda: t[0]  # noqa: SLF001
    write_call_state(rig.store, language=langs[0])

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        for n, (at, lang) in enumerate(zip(times, langs, strict=True)):
            front_door_update(rig.store, language=lang)
            t[0] = at
            await sim.prompt(f"question {n}", timeout=REPLY_S)
        await sim.hangup()

    asyncio.run(go())
    sent = [str(c["system_suffix"]).splitlines()[-1] for c in rig.agent.calls]
    tz = rig.bridge._cfg.timezone  # noqa: SLF001
    assert sent == expected_lines(list(zip(times, langs, strict=True)), tz)
