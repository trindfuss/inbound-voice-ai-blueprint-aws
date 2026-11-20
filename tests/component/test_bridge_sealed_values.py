"""The bridge restores sealed values.

The front door seals each sensitive value into call state's ``sealed_sensitive``
(``sensitive.remember``). The bridge is another process: it learns the values only from that field.
On every call-state read (setup, each turn, an interrupt, the bank tool reload) it calls
``sensitive.restore``, so the exact-value scrub of AI-turn journal text, of the escalation summary
(and so of the task attributes built from it) and of every log line has the values. A value that
can't be unsealed fails closed: AI-turn text becomes ``[WITHHELD]`` and the summary the catalog's
``escalation.summary_withheld``. What the agent receives is never changed. At session end the
bridge forgets the call's values.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest

from owlbank import callstate as cs
from owlbank import obs, segments, sensitive
from owlbank.pii import WITHHELD_TURN
from owlbank.task_attributes import build_task_attributes
from owlbank.token_seal import FakeSealer
from tac_bridge.agent_client import TextDelta, TurnEnd
from tests import logutil
from tests.component.test_bridge_fakes import (
    CALL_SID,
    NOW,
    SEALER,
    FakeBackend,
    make_rig,
    run,
    say,
    spoken,
    tool,
    write_call_state,
)

SSN4 = "4821"  # 4 digits: neither the fake PII check nor the log pattern scrub removes it
WORD = "zephyrine"  # a letters-only value: no pattern scrub would ever touch it
REASON = "caller_requested_human"


def _norm(text: str) -> str:
    return segments._norm_text(text)


def _leaks(text: str, values: tuple[str, ...] = (SSN4, WORD)) -> list[str]:
    norm = _norm(text)
    return [v for v in values if _norm(v) in norm]


@pytest.fixture(autouse=True)
def _fresh_process() -> Iterator[None]:
    sensitive.forget(CALL_SID)
    yield
    sensitive.forget(CALL_SID)


@pytest.fixture
def scrubbed_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Captured records pass ``obs.ScrubFilter`` first, as the bridge's JSON handler does."""
    caplog.set_level(logging.DEBUG)
    scrub = obs.ScrubFilter()
    caplog.handler.addFilter(scrub)
    yield caplog
    caplog.handler.removeFilter(scrub)


def _seal_from_front_door(rig: Any, *values: tuple[str, str]) -> None:
    """What a front-door run leaves: the values sealed in call state. This process (the bridge)
    then forgets them, as a separate process never had them."""
    for slot, value in values:
        assert sensitive.remember(rig.store, SEALER, CALL_SID, value, slot=slot, now=lambda: NOW)
    sensitive.forget(CALL_SID)
    assert segments.sensitive_values(CALL_SID) == frozenset()


def _state(rig: Any) -> cs.CallState:
    state = cs.load(rig.store, CALL_SID)
    assert state is not None
    return state


def _task_summary(rig: Any) -> str:
    state = _state(rig)
    summary = (state.escalation or {}).get("summary")
    attrs = build_task_attributes(state, {}, REASON, summary, now=NOW)
    return str(attrs["ai_summary"])


def _escalate_script() -> list[Any]:
    echo = f"Your last four are {SSN4} and your word is {WORD}."
    return [say(echo),
            tool("escalate", reason=REASON, summary=f"Caller gave {SSN4} then {WORD.upper()}.")]


def test_restored_values_are_scrubbed_from_journal_summary_task_attributes_and_logs(
        scrubbed_logs: pytest.LogCaptureFixture) -> None:
    rig = make_rig(script=[*_escalate_script()[:1],
                           tool("list_recent_transactions", account="checking"),
                           say("Sorry."), _escalate_script()[1]],
                   backend=FakeBackend({"list_recent_transactions": RuntimeError(
                       f"upstream echoed {SSN4} and {WORD}")}))
    write_call_state(rig.store)
    _seal_from_front_door(rig, ("ssn4", SSN4), ("secret_word", WORD))
    caller = f"it's {SSN4}, my word is {WORD}"

    async def go() -> tuple[str, str, str]:
        sim = rig.sim()
        await sim.connect()
        first = spoken(await sim.prompt(caller))
        second = spoken(await sim.prompt(caller))
        third = spoken(await sim.prompt("a person please"))
        await sim.hangup()
        return first, second, third

    first, _, _ = run(go())
    # What the caller hears and what the agent receives are not changed.
    assert SSN4 in first and WORD in first
    assert [c["user_text"] for c in rig.agent.calls if c["user_text"]][:2] == [caller, caller]
    events = rig.events()
    ai_turns = [e for e in events if e["type"] == "ai_turn"]
    assert ai_turns and all(not _leaks(e["text"]) for e in ai_turns)
    assert any("[redacted]" in e["text"] for e in ai_turns)  # the scrub ran, not the withhold
    assert WITHHELD_TURN not in " ".join(e["text"] for e in ai_turns)
    escalation = [e for e in events if e["type"] == "escalation"]
    assert len(escalation) == 1 and not _leaks(escalation[0]["summary"])
    assert "Caller gave" in escalation[0]["summary"]
    stored = (_state(rig).escalation or {}).get("summary")
    assert isinstance(stored, str) and stored and not _leaks(stored)
    assert not _leaks(_task_summary(rig))
    assert not _leaks(" ".join(str(e) for e in events))
    # The backend error's message carried both values: the log scrub had them.
    text = logutil.text_of(scrubbed_logs.records)
    assert "bank_tools call failed" in text
    assert not _leaks(text)


def test_values_are_restored_at_setup_before_any_turn() -> None:
    rig = make_rig()
    write_call_state(rig.store)
    _seal_from_front_door(rig, ("ssn4", SSN4))

    async def go() -> frozenset[str]:
        sim = rig.sim()
        await sim.connect()
        await sim.idle()
        registered = segments.sensitive_values(CALL_SID)
        await sim.hangup()
        return registered

    assert run(go()) == frozenset({_norm(SSN4)})


def test_an_interrupted_turn_is_scrubbed() -> None:
    rig = make_rig(script=[[TextDelta(f"word{i} ") for i in range(200)] + [TurnEnd("end_turn")]])
    rig.agent.delay_s = 0.005
    write_call_state(rig.store)
    _seal_from_front_door(rig, ("ssn4", SSN4))

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt_nowait("tell me everything")
        await asyncio.sleep(0.05)
        await sim.interrupt(heard=f"your last four are {SSN4}")
        await sim.idle()
        await sim.hangup()

    run(go())
    turns = [e for e in rig.events() if e["type"] == "ai_turn" and e.get("interrupted")]
    assert len(turns) == 1 and not _leaks(turns[0]["text"])
    assert "your last four are" in turns[0]["text"]


def test_an_unseal_failure_withholds_both_texts_and_logs_no_value(
        scrubbed_logs: pytest.LogCaptureFixture) -> None:
    # The production level (INFO). At DEBUG, Strands' own lines carry the model's tool input, and
    # with no value unsealed nothing can scrub them (Strands isn't a pinned logger).
    scrubbed_logs.set_level(logging.INFO)
    rig = make_rig(script=_escalate_script())
    write_call_state(rig.store)
    blob = FakeSealer(key=b"another-key").seal(json.dumps({"v": 1, "slot": "ssn4", "value": SSN4}),
                                               call_sid=CALL_SID)
    cs.mutate(rig.store, CALL_SID, lambda _s: {cs.F_SEALED_SENSITIVE: (blob,)}, now=lambda: NOW)

    async def go() -> tuple[str, str]:
        sim = rig.sim()
        await sim.connect()
        first = spoken(await sim.prompt(f"it's {SSN4}"))
        second = spoken(await sim.prompt("a person please"))
        await sim.hangup()
        return first, second

    first, _ = run(go())
    assert SSN4 in first  # the call goes on; the caller still hears the agent
    assert rig.agent.calls[0]["user_text"] == f"it's {SSN4}"
    events = rig.events()
    ai_turns = [e for e in events if e["type"] == "ai_turn"]
    assert ai_turns and all(e["text"] == WITHHELD_TURN for e in ai_turns)
    withheld = rig.bridge._catalog.text("escalation.summary_withheld", "en-US").strip()
    escalation = [e for e in events if e["type"] == "escalation"]
    assert len(escalation) == 1 and escalation[0]["summary"] == withheld
    assert (_state(rig).escalation or {}).get("summary") == withheld
    assert not _leaks(" ".join(str(e) for e in events))
    records = scrubbed_logs.records
    assert any(logutil.fields(r).get("decision") == "withhold_texts" for r in records)
    assert not _leaks(logutil.text_of(records))


def test_no_sealed_values_behaves_as_today() -> None:
    rig = make_rig(script=[say("Your balance is fine."),
                           tool("escalate", reason=REASON, summary="Wants a person.")])
    write_call_state(rig.store)
    unseals: list[str] = []

    class Counting(FakeSealer):
        def unseal(self, sealed: str, *, call_sid: str) -> str:
            unseals.append(sealed)
            return super().unseal(sealed, call_sid=call_sid)

    rig.bridge._sealer = rig.bridge._calls._sealer = Counting()

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("hi")
        await sim.prompt("a person please")
        await sim.hangup()

    run(go())
    texts = [e["text"] for e in rig.events() if e["type"] == "ai_turn"]
    assert texts == ["Your balance is fine."]
    assert [e["summary"] for e in rig.events() if e["type"] == "escalation"] == ["Wants a person."]
    assert unseals == []  # no registry entry to unseal (the token is only unsealed for a bank tool)
    assert not sensitive.is_unverifiable(CALL_SID)


def test_session_end_forgets_the_call() -> None:
    rig = make_rig()
    write_call_state(rig.store)
    _seal_from_front_door(rig, ("ssn4", SSN4))
    blob = FakeSealer(key=b"another-key").seal("x", call_sid=CALL_SID)
    cs.mutate(rig.store, CALL_SID,
              lambda s: {cs.F_SEALED_SENSITIVE: (*s.sealed_sensitive, blob)}, now=lambda: NOW)

    async def go() -> tuple[frozenset[str], bool]:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("hi")
        during = segments.sensitive_values(CALL_SID), sensitive.is_unverifiable(CALL_SID)
        await sim.hangup()
        rig.bridge.session_end(CALL_SID, "closed")
        await sim.idle()
        return during

    registered, unverifiable = run(go())
    assert registered == frozenset({_norm(SSN4)}) and unverifiable
    assert segments.sensitive_values(CALL_SID) == frozenset()
    assert not sensitive.is_unverifiable(CALL_SID)


def test_session_end_during_a_turn_forgets_after_the_turn() -> None:
    """A turn still running when the socket closes journals its text with the values in place."""
    rig = make_rig()
    write_call_state(rig.store)
    _seal_from_front_door(rig, ("ssn4", SSN4))

    async def go() -> tuple[frozenset[str], frozenset[str]]:
        rig.bridge._calls.load(CALL_SID)  # a read restores the values
        lock = rig.bridge._locks.setdefault(CALL_SID, asyncio.Lock())
        await lock.acquire()  # the running turn
        rig.bridge.session_end(CALL_SID, "closed")
        await asyncio.sleep(0)
        during = segments.sensitive_values(CALL_SID)
        lock.release()
        await rig.bridge.wait_idle()
        return during, segments.sensitive_values(CALL_SID)

    during, after = run(go())
    assert during == frozenset({_norm(SSN4)})
    assert after == frozenset()
