"""AI keypad shortcuts: the adapter and the actions.

Real TAC through the ConversationRelay simulator and ``PeekingWebSocket``, the bridge's fakes. The
keypad window is shortened to 50 ms (``owlbank.keypad.WINDOW_S``) and the silence watch is kept
out of the way (a long timeout) unless a test is about it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from owlbank import keypad
from owlbank.envelope import ResultEnvelope
from owlbank.tools_contract import FALLBACK_REASON
from tac_bridge import silence
from tac_bridge.app import AI_KEYPAD_ENV, PeekingWebSocket, keypad_setting_from_env
from tac_bridge.bridge import BridgeConfig, tac_stream_state
from tests.component.masking_fixtures import unmasked_policy_doc
from tests.component.test_bridge_fakes import (
    CALL_SID,
    CRSimulator,
    FakeOrchestrator,
    SimSocket,
    end_frame,
    make_rig,
    run,
    say,
    tool,
    wait_for,
    write_call_state,
)
from tests.leakscan import value_pattern
from tests.logutil import fields, lines, text_of, with_context

BRIDGE_LOG = "tac_bridge.bridge"
WINDOW = 0.05
REPLY = "Your card is on its way."
HINT_EN = "Please say what you need, or press 0 for an agent."
HINT_ES = "Por favor, diga lo que necesita, o marque 0 para hablar con un agente."
SUMMARY = "The caller pressed 0 to speak with an agent."


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(keypad, "WINDOW_S", WINDOW)
    monkeypatch.setattr(silence, "TTS_START_S", 0.02)
    caplog.set_level(logging.INFO)


def policy(**masking: Any) -> dict[str, Any]:
    doc = unmasked_policy_doc()  # no masking phrases, no timed acknowledgement
    doc["masking"] = {**doc["masking"], "chars_per_s": 30, "silence_timeout_ms": 10000,
                      "silence_max_repeats": 2, "greeting_silence_timeout_ms": 10000, **masking}
    return doc


def rig_with(script: list[Any] | None = None, *, on: bool = True, language: str = "en-US",
             delay_s: float = 0.0, **masking: Any) -> Any:
    rig = make_rig(script=script or [], policy_doc=policy(**masking), config=BridgeConfig(ai_keypad=on))
    rig.agent.delay_s = delay_s
    write_call_state(rig.store, language=language)
    return rig


def texts(sim: CRSimulator) -> list[str]:
    return [f["token"] for f in sim.socket.sent if f.get("type") == "text" and f.get("token")]


def ended(sim: CRSimulator) -> bool:
    return any(f.get("type") == "end" for f in sim.socket.sent)


def user_texts(rig: Any) -> list[str | None]:
    return [c["user_text"] for c in rig.agent.calls]


def bridge_events(caplog: pytest.LogCaptureFixture, name: str) -> list[dict[str, Any]]:
    return [ln for ln in lines(caplog.records, BRIDGE_LOG) if ln["event"] == name]


def journal(rig: Any, etype: str) -> list[dict[str, Any]]:
    return [e for e in rig.events() if e["type"] == etype]


async def settle(sim: CRSimulator, seconds: float = 0.25) -> None:
    await asyncio.sleep(seconds)
    await sim.idle()


def assert_keypad_lines_clean(caplog: pytest.LogCaptureFixture) -> None:
    """A ``keypad_action`` line has the action and a key count only; no line has a digit field."""
    for ln in bridge_events(caplog, "keypad_action"):
        assert set(ln) <= {"event", "action", "keys", "call_sid", "component"}
        assert ln["action"] in keypad.ACTIONS and isinstance(ln["keys"], int)
    for record in caplog.records:
        assert "digit" not in fields(record)


# --------------------------------------------------------------------------- 1 and 2: a caller turn


@pytest.mark.parametrize(("language", "digit", "word"), [
    ("en-US", "1", "Yes"), ("es-US", "1", "Sí"), ("en-US", "2", "No"), ("es-US", "2", "No")])
def test_one_and_two_are_a_caller_turn_in_the_call_language(
        caplog: pytest.LogCaptureFixture, language: str, digit: str, word: str) -> None:
    rig = rig_with([say(REPLY)], language=language)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        with_context(caplog)
        await sim.dtmf(digit)
        await wait_for(lambda: REPLY in texts(sim), 3)
        await settle(sim)
        await sim.hangup()
        return sim

    sim = run(go())
    assert user_texts(rig) == [word]  # exactly one turn, the catalog word
    assert texts(sim) == [REPLY]
    (line,) = bridge_events(caplog, "keypad_action")
    assert line["action"] == ("yes" if digit == "1" else "no") and line["keys"] == 1
    assert line["call_sid"] == CALL_SID
    assert_keypad_lines_clean(caplog)


def test_a_key_during_a_turn_acts_after_it() -> None:
    rig = rig_with([say("One moment.", " Here it is."), say(REPLY)], delay_s=0.08)

    async def go() -> tuple[CRSimulator, int]:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt_nowait("what is my balance")
        await wait_for(lambda: len(rig.agent.calls) == 1, 3)
        await sim.dtmf("1")  # due after 50 ms, while the first turn is still speaking
        await asyncio.sleep(0.1)
        calls_mid_turn = len(rig.agent.calls)
        await wait_for(lambda: REPLY in texts(sim), 3)
        await settle(sim)
        await sim.hangup()
        return sim, calls_mid_turn

    sim, calls_mid_turn = run(go())
    assert calls_mid_turn == 1  # the key's turn waited for the lock
    assert user_texts(rig) == ["what is my balance", "Yes"]
    assert texts(sim) == ["One moment.", " Here it is.", REPLY]


PARTS = ("Part one.", " Part two.", " Part three.")
#: The tests that time a turn's chunks: the plain scripted agent streams each after ``delay_s``.
TIMED = pytest.mark.protocol_fake("times the keypad turn's chunks against a barge-in")


@TIMED
@pytest.mark.parametrize("after_prompt", [False, True])
def test_an_interrupt_during_a_keypad_turn_stops_it(after_prompt: bool) -> None:
    """Review findings 1 and 7: a barge-in (speech or a key, ``interruptible="any"``) cancels the
    keypad turn as it cancels a spoken turn, with one ``last: true`` acknowledgement. After the
    socket's first prompt TAC does it (the turn is its stream task); before, TAC ignores the
    interrupt and the bridge does it."""
    rig = rig_with(([say("Hello.")] if after_prompt else []) + [say(*PARTS)], delay_s=0.1)

    async def go() -> tuple[CRSimulator, list[dict[str, Any]]]:
        sim = rig.sim()
        await sim.connect()
        if after_prompt:
            await sim.prompt("hello")
        await sim.dtmf("1")
        await wait_for(lambda: "Part one." in texts(sim), 3)
        mark = len(sim.socket.sent)
        await sim.interrupt("Part one.")
        await settle(sim, 0.5)
        after = sim.socket.sent[mark:]
        await sim.hangup()
        return sim, after

    sim, after = run(go())
    assert user_texts(rig) == (["hello"] if after_prompt else []) + ["Yes"]
    assert texts(sim)[-1] == "Part one."  # no token after the barge-in
    assert after == [{"type": "text", "token": "", "last": True}]  # the interrupt's acknowledgement, once
    if after_prompt:  # TAC calls on_interrupt only once it has the conversation
        assert [e["text"] for e in journal(rig, "ai_turn") if e.get("interrupted")] == ["Part one."]


@TIMED
def test_a_spoken_prompt_during_a_keypad_turn_cancels_it_and_runs_at_once() -> None:
    rig = rig_with([say(*PARTS), say(REPLY)], delay_s=0.1)

    async def go() -> tuple[CRSimulator, float]:
        sim = rig.sim()
        await sim.connect()
        await sim.dtmf("1")
        await wait_for(lambda: "Part one." in texts(sim), 3)
        loop = asyncio.get_running_loop()
        start = loop.time()
        await sim.prompt_nowait("what is my balance")
        await wait_for(lambda: REPLY in texts(sim), 3)
        took = loop.time() - start
        await settle(sim)
        await sim.hangup()
        return sim, took

    sim, took = run(go())
    assert user_texts(rig) == ["Yes", "what is my balance"]
    assert " Part three." not in texts(sim) and texts(sim)[-1] == REPLY
    assert took < 0.3  # one agent step (0.1 s), not the rest of the keypad turn first


def test_pressing_one_and_saying_yes_is_one_turn(caplog: pytest.LogCaptureFixture) -> None:
    """A prompt drops the keys not yet acted on."""
    rig = rig_with([say(REPLY), say("Second answer.")])

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await sim.dtmf("1")
        await sim.prompt("yes")  # within the key's window
        await settle(sim, WINDOW * 4)
        await sim.hangup()
        return sim

    sim = run(go())
    assert user_texts(rig) == ["yes"]
    assert texts(sim) == [REPLY]
    assert bridge_events(caplog, "keypad_action") == []


# --------------------------------------------------------------------------- 0: a human


def test_zero_escalates_with_caller_requested_human_and_no_model_call(caplog: pytest.LogCaptureFixture) -> None:
    rig = rig_with()

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        await sim.dtmf("0")
        await wait_for(lambda: ended(sim), 3)
        await settle(sim)
        frames = list(sim.socket.sent)
        await sim.hangup()
        return frames

    frames = run(go())
    end = end_frame(frames)
    assert end is not None
    data = end["data"]
    assert data["reason"] == "escalate" and data["tool"] == "escalate"
    assert data["escalationReason"] == "caller_requested_human" and data["trigger"] == "keypad"
    assert data["journal"] and data["conversationId"]
    escalation = rig.call()["escalation"]
    assert escalation["reason"] == "caller_requested_human" and escalation["source"] == "keypad"
    assert escalation["summary"] == SUMMARY and escalation["journal_ref"] == data["journal"]
    assert [(e["reason"], e["summary"]) for e in journal(rig, "escalation")] == [("caller_requested_human", SUMMARY)]
    (handoff,) = journal(rig, "handoff")
    assert handoff["trigger"] == "keypad" and handoff["reason"] == "escalate"
    assert rig.agent.calls == []  # no model call
    assert [f for f in frames if f.get("type") == "text"] == [{"type": "text", "token": "", "last": True}]
    (req,) = bridge_events(caplog, "escalation_requested")
    assert req["reason"] == "caller_requested_human" and req["source"] == "keypad"
    assert_keypad_lines_clean(caplog)


def test_a_failed_keypad_escalation_falls_back_to_the_error_escalation(
        caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = rig_with()

    async def broken(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("journal down")

    monkeypatch.setattr(rig.bridge, "_keypad_escalate", broken)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        await sim.dtmf("0")
        await wait_for(lambda: ended(sim), 3)
        frames = list(sim.socket.sent)
        await sim.hangup()
        return frames

    end = end_frame(run(go()))
    assert end is not None and end["data"]["escalationReason"] == FALLBACK_REASON
    (failed,) = bridge_events(caplog, "keypad_action_failed")
    assert failed["action"] == "agent" and failed["error_class"] == "RuntimeError"


# --------------------------------------------------------------------------- 9: repeat


def test_nine_repeats_the_last_message_without_the_agent(caplog: pytest.LogCaptureFixture) -> None:
    rig = rig_with([say(REPLY)])

    async def go() -> tuple[CRSimulator, int]:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("where is my card")
        await sim.dtmf("9")
        await wait_for(lambda: texts(sim).count(REPLY) == 2, 3)
        await settle(sim)
        repeats = rig.bridge._speech[CALL_SID].watch.repeats
        await sim.hangup()
        return sim, repeats

    sim, repeats = run(go())
    assert texts(sim) == [REPLY, REPLY]
    assert user_texts(rig) == ["where is my card"]  # the agent's history is unchanged
    assert repeats == 0  # the silence repeat count is unchanged
    (rep,) = journal(rig, "prompt_repeat")
    assert rep["trigger"] == "keypad" and rep["what"] == "agent" and rep["text"] == REPLY
    (line,) = bridge_events(caplog, "prompt_repeat")
    assert line["trigger"] == "keypad" and line["chars"] == len(REPLY) and "repeat" not in line
    assert REPLY not in text_of([r for r in caplog.records if r.name.startswith("tac_bridge")])


def test_after_a_keypad_repeat_the_silence_repeat_still_has_its_full_count(caplog: pytest.LogCaptureFixture) -> None:
    rig = rig_with([say(REPLY)], silence_timeout_ms=150, silence_max_repeats=1, silence_action="escalate")

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("where is my card")
        await sim.dtmf("9")
        await wait_for(lambda: ended(sim), 5)
        await sim.hangup()
        return sim

    sim = run(go())
    # the reply, the keypad repeat, one silence repeat (max 1), then no_input
    assert texts(sim) == [REPLY, REPLY, REPLY]
    triggers = [e.get("trigger") for e in journal(rig, "prompt_repeat")]
    assert triggers == ["keypad", None]
    assert end_frame(sim.socket.sent)["data"]["escalationReason"] == "no_input"  # type: ignore[index]


# --------------------------------------------------------------------------- hints and multi-key entries


@pytest.mark.parametrize(("language", "digit", "hint"), [
    ("en-US", "5", HINT_EN), ("es-US", "#", HINT_ES), ("en-US", "*", HINT_EN), ("en-US", "x", HINT_EN)])
def test_another_key_speaks_the_hint(caplog: pytest.LogCaptureFixture, language: str, digit: str,
                                     hint: str) -> None:
    rig = rig_with(language=language)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await sim.dtmf(digit)
        await wait_for(lambda: hint in texts(sim), 3)
        await settle(sim)
        await sim.hangup()
        return sim

    sim = run(go())
    assert texts(sim) == [hint]
    assert rig.agent.calls == []
    (line,) = bridge_events(caplog, "keypad_action")
    assert line["action"] == "hint" and line["keys"] == 1
    assert_keypad_lines_clean(caplog)


def test_the_hint_rearms_the_silence_watch(caplog: pytest.LogCaptureFixture) -> None:
    rig = rig_with(silence_timeout_ms=150, silence_max_repeats=1)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await sim.dtmf("5")
        await wait_for(lambda: texts(sim).count(HINT_EN) == 2, 5)
        await sim.hangup()
        return sim

    run(go())
    (rep,) = journal(rig, "prompt_repeat")
    assert rep["what"] == "keypad.hint" and rep["prompt_key"] == "keypad.hint"


def test_nine_after_the_hint_repeats_the_agent_not_the_hint() -> None:
    """The hint never replaces the message 9 repeats."""
    rig = rig_with([say(REPLY)])

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("where is my card")
        await sim.dtmf("5")
        await wait_for(lambda: HINT_EN in texts(sim), 3)
        await sim.dtmf("9")
        await wait_for(lambda: texts(sim).count(REPLY) == 2, 3)
        await settle(sim)
        await sim.hangup()
        return sim

    sim = run(go())
    assert texts(sim) == [REPLY, HINT_EN, REPLY]
    (rep,) = journal(rig, "prompt_repeat")
    assert rep["trigger"] == "keypad" and rep["what"] == "agent" and rep["text"] == REPLY


def test_silence_after_the_hint_repeats_the_agent_after_the_hint_ends() -> None:
    """The silence repeat after a hint is the agent's last message, and the wait
    counts from the hint's estimated end (it never fires while the hint plays)."""
    cps, timeout_ms = 100, 150
    rig = rig_with([say(REPLY)], silence_timeout_ms=timeout_ms, silence_max_repeats=1, chars_per_s=cps)

    async def go() -> tuple[CRSimulator, float]:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("where is my card")
        await sim.dtmf("5")
        await wait_for(lambda: HINT_EN in texts(sim), 3)
        loop = asyncio.get_running_loop()
        hint_at = loop.time()
        await wait_for(lambda: texts(sim).count(REPLY) == 2, 5)
        waited = loop.time() - hint_at
        await sim.hangup()
        return sim, waited

    sim, waited = run(go())
    assert texts(sim)[:3] == [REPLY, HINT_EN, REPLY]
    assert waited >= len(HINT_EN) / cps + timeout_ms / 1000 - 0.03
    (rep,) = journal(rig, "prompt_repeat")
    assert rep["what"] == "agent" and rep.get("trigger") is None and rep["text"] == REPLY


def test_a_multi_digit_entry_gives_only_the_hint(caplog: pytest.LogCaptureFixture) -> None:
    rig = rig_with()
    entry = "4111"

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        for d in entry:
            await sim.dtmf(d)
        await wait_for(lambda: HINT_EN in texts(sim), 3)
        await settle(sim)
        await sim.hangup()
        return sim

    sim = run(go())
    assert texts(sim) == [HINT_EN]
    assert rig.agent.calls == []  # no "yes" for the 1s
    (line,) = bridge_events(caplog, "keypad_action")
    assert line == {**line, "action": "hint", "keys": 4}


# --------------------------------------------------------------------------- a pending hand-off, no session


def pending_handoff(rig: Any) -> Any:
    """A hand-off set with its ``end`` not yet sent (e.g. ``make_payment`` or ``escalate`` called,
    the turn holding for its last words)."""
    return rig.bridge._handoff_factory(json.dumps({"reason": "escalate", "tool": "escalate",
                                                   "escalationReason": "caller_requested_human"}))


def end_frames(sim: CRSimulator) -> list[dict[str, Any]]:
    return [f for f in sim.socket.sent if f.get("type") == "end"]


def test_a_key_while_a_hand_off_is_pending_does_nothing_but_send_its_end(caplog: pytest.LogCaptureFixture) -> None:
    """No key acts, but the pending hand-off's ``end`` (which a key's
    barge-in can keep from going out) is sent, once."""
    rig = rig_with()

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        sim.session.pending_handoff_data = pending_handoff(rig)
        for d in "1290":
            await sim.dtmf(d)
            await asyncio.sleep(WINDOW * 2)
        await settle(sim)
        await sim.hangup()
        return sim

    sim = run(go())
    assert texts(sim) == []
    (end,) = end_frames(sim)
    assert json.loads(end["handoffData"])["escalationReason"] == "caller_requested_human"
    assert rig.agent.calls == [] and journal(rig, "escalation") == []
    assert bridge_events(caplog, "keypad_action") == []
    assert {ln["cause"] for ln in bridge_events(caplog, "keypad_ignored")} == {"handoff_pending"}
    (resent,) = bridge_events(caplog, "handoff_end_resent")
    assert resent["cause"] == "keypad"


def test_an_interrupt_with_a_hand_off_pending_sends_its_end(caplog: pytest.LogCaptureFixture) -> None:
    """A barge-in (``interruptible="any"``) that cancelled the turn after its
    hand-off was set, before TAC sent the ``end``: the bridge sends it."""
    rig = rig_with()

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        sim.session.pending_handoff_data = pending_handoff(rig)
        await sim.interrupt("One moment while I")
        await settle(sim)
        await sim.hangup()
        return sim

    sim = run(go())
    assert len(end_frames(sim)) == 1
    assert sim.session is None or sim.session.pending_handoff_data is None
    (resent,) = bridge_events(caplog, "handoff_end_resent")
    assert resent["cause"] == "interrupt"


def test_with_shortcuts_off_an_interrupt_leaves_a_pending_hand_off_alone() -> None:
    rig = rig_with(on=False)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        sim.session.pending_handoff_data = pending_handoff(rig)
        await sim.interrupt("One moment while I")
        await settle(sim)
        await sim.hangup()
        return sim

    assert end_frames(run(go())) == []  # today's behaviour (off restores it)


def test_a_key_during_a_turn_that_hands_off_sends_no_second_end(caplog: pytest.LogCaptureFixture) -> None:
    escalate = tool("escalate", "tu_esc", reason="caller_requested_human", summary="Caller asked for a person.")
    rig = rig_with([escalate], delay_s=0.08)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt_nowait("a person please")
        await wait_for(lambda: len(rig.agent.calls) == 1, 3)
        await sim.dtmf("1")  # while the turn that sets the hand-off still runs: it sends the end
        await sim.dtmf("5")
        await wait_for(lambda: ended(sim), 3)
        await settle(sim)
        await sim.hangup()
        return sim

    sim = run(go())
    assert len(end_frames(sim)) == 1 and user_texts(rig) == ["a person please"]
    assert bridge_events(caplog, "handoff_end_resent") == []


def draining_socket(monkeypatch: pytest.MonkeyPatch, *, stall_first_end: bool = False) -> None:
    """The simulator's socket yields after each frame, as a real socket's drain does. With
    ``stall_first_end`` the first ``end`` frame is written and its drain never returns (the turn
    sending it stays suspended there until it is cancelled)."""
    real = SimSocket.send_text
    stalled: list[bool] = []

    async def send_text(self: SimSocket, data: str) -> None:
        await real(self, data)
        if stall_first_end and not stalled and json.loads(data).get("type") == "end":
            stalled.append(True)
            await asyncio.Event().wait()
        await asyncio.sleep(0)

    monkeypatch.setattr(SimSocket, "send_text", send_text)


@pytest.mark.parametrize("barge", ["key", "interrupt"])
def test_a_key_during_a_resume_turns_hand_off_hold_sends_one_end(
        caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, barge: str) -> None:
    """A resume turn started at ``setup`` is not TAC's
    stream task and releases the call's lock before TAC sends its ``last`` and ``end``. A key or a
    barge-in during its hand-off hold must not send a second ``end``."""
    interrupted = ResultEnvelope(status="interrupted", tool="get_balance", spoken=[], data={},
                                 next="handle_utterance", callerUtterance="get me a person").to_json()
    rig = make_rig(script=[say("<silent/>"), tool("escalate", "tu_esc", reason="caller_requested_human",
                                                  summary="Caller asked for a person.")],
                   policy_doc=policy(), config=BridgeConfig(ai_keypad=True))
    write_call_state(rig.store, pending_tool={"tool_use_id": "tu_bal", "name": "get_balance", "args": {}},
                     pending_result=interrupted, interrupted_utterance="get me a person")
    draining_socket(monkeypatch)
    bridge = rig.bridge
    real_start = bridge._start_handoff

    async def start_handoff(*args: Any, **kwargs: Any) -> Any:
        out = await real_start(*args, **kwargs)
        await asyncio.sleep(0.3)  # the hold for the last words, with the lock taken
        return out

    monkeypatch.setattr(bridge, "_start_handoff", start_handoff)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await wait_for(lambda: sim.session is not None and sim.session.pending_handoff_data is not None, 3)
        if barge == "key":
            await sim.dtmf("1")
        else:
            await sim.interrupt("One moment")
        await wait_for(lambda: ended(sim), 3)
        await settle(sim, 0.5)
        await sim.hangup()
        return sim

    sim = run(go())
    assert len(end_frames(sim)) == 1
    assert bridge_events(caplog, "handoff_end_resent") == []


def test_an_interrupt_cancelling_a_turn_inside_its_end_send_sends_one_end(
        caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """TAC's interrupt cancels the stream turn while
    it is suspended in the ``end`` frame's send, so ``pending_handoff_data`` is never cleared. The
    ``end`` went out: the bridge must not send another."""
    escalate = tool("escalate", "tu_esc", reason="caller_requested_human", summary="Caller asked for a person.")
    rig = rig_with([escalate])
    draining_socket(monkeypatch, stall_first_end=True)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt_nowait("a person please")
        await wait_for(lambda: ended(sim), 3)
        await sim.interrupt("")
        await settle(sim, 0.5)
        await sim.hangup()
        return sim

    sim = run(go())
    assert len(end_frames(sim)) == 1
    assert bridge_events(caplog, "handoff_end_resent") == []


def test_a_barge_in_before_the_first_prompt_that_cuts_a_keypad_hand_off_sends_one_logged_end(
        caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """Before the socket's first prompt TAC ignores an interrupt and
    the bridge acknowledges it (``_keypad_ack``). With the keypad turn's hand-off pending, that
    acknowledgement's ``end`` is the resend: once, marked and logged as ``handoff_end_resent``."""
    escalate = tool("escalate", "tu_esc", reason="caller_requested_human", summary="Caller asked for a person.")
    rig = rig_with([escalate])
    bridge = rig.bridge
    real_start = bridge._start_handoff

    async def start_handoff(*args: Any, **kwargs: Any) -> Any:
        out = await real_start(*args, **kwargs)
        await asyncio.sleep(0.3)  # the hold for the last words
        return out

    monkeypatch.setattr(bridge, "_start_handoff", start_handoff)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await sim.dtmf("1")
        await wait_for(lambda: sim.session is not None and sim.session.pending_handoff_data is not None, 3)
        await sim.interrupt("")
        await wait_for(lambda: ended(sim), 3)
        await settle(sim, 0.5)
        await sim.hangup()
        return sim

    sim = run(go())
    assert user_texts(rig) == ["Yes"]
    assert len(end_frames(sim)) == 1
    assert sim.socket.sent[-1]["type"] == "end"  # nothing after the end
    assert sim.socket.sent[-2] == {"type": "text", "token": "", "last": True}
    (resent,) = bridge_events(caplog, "handoff_end_resent")
    assert resent["cause"] == "interrupt"


def test_a_keypad_turn_cancelled_before_it_starts_cancels_the_turn_it_waited_for(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A barge-in or a prompt that
    cancels the keypad turn in the same loop iteration it became TAC's stream task also cancels
    the turn it was waiting for, and nothing more is spoken after the acknowledgement."""
    rig = rig_with([say("Hello."), say(REPLY)], delay_s=0.1)
    bridge = rig.bridge
    holder: list[CRSimulator] = []
    seen: dict[str, Any] = {}
    real_word, real_spawn = bridge._keypad_word, bridge._keypad_spawn

    def word(kind: str, lang: str) -> str:  # under the keypad action's lock: a spoken turn starts now
        session = holder[0].session
        state = tac_stream_state(bridge.voice, session.conversation_id)
        state.stream_task = asyncio.create_task(bridge._send(session, bridge.run_turn(session, "my balance")))
        return real_word(kind, lang)

    def spawn(call_sid: str, work: Any) -> asyncio.Task[Any]:
        task = real_spawn(call_sid, work)
        if getattr(work, "__qualname__", "").endswith("_keypad_turn_run"):
            session = holder[0].session
            seen["previous"] = tac_stream_state(bridge.voice, session.conversation_id).stream_task
            seen["mark"] = len(holder[0].socket.sent)
            task.cancel()  # before its first step, as TAC's cancel_stream_task or _keypad_barge can
        return task

    monkeypatch.setattr(bridge, "_keypad_word", word)
    monkeypatch.setattr(bridge, "_keypad_spawn", spawn)

    async def go() -> CRSimulator:
        sim = rig.sim()
        holder.append(sim)
        await sim.connect()
        await sim.prompt("hello")
        await sim.dtmf("1")
        await wait_for(lambda: "previous" in seen, 3)
        await sim.interrupt("")  # TAC's acknowledgement (the socket has had a prompt)
        await settle(sim, 0.5)
        await sim.hangup()
        return sim

    sim = run(go())
    assert seen["previous"] is not None and seen["previous"].cancelled()
    assert REPLY not in texts(sim)
    assert sim.socket.sent[seen["mark"]:] == [{"type": "text", "token": "", "last": True}]


def test_a_key_during_a_turn_that_hands_off_does_nothing() -> None:
    escalate = tool("escalate", "tu_esc", reason="caller_requested_human", summary="Caller asked for a person.")
    rig = rig_with([escalate], delay_s=0.08)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt_nowait("a person please")
        await wait_for(lambda: len(rig.agent.calls) == 1, 3)
        await sim.dtmf("1")  # due mid-turn; by the time it gets the lock the hand-off is pending
        await wait_for(lambda: ended(sim), 3)
        await settle(sim)
        await sim.hangup()
        return sim

    sim = run(go())
    assert user_texts(rig) == ["a person please"]
    data = end_frame(sim.socket.sent)["data"]  # type: ignore[index]
    assert "trigger" not in data  # the agent's escalation, not a second one


def test_a_keypad_turn_that_waited_for_a_hand_off_turn_calls_no_model(
        caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """The keypad turn is checked again once it has the lock. Here a spoken turn
    takes the lock between the keypad action's check and the keypad turn, and hands off."""
    escalate = tool("escalate", "tu_esc", reason="caller_requested_human", summary="Caller asked for a person.")
    rig = rig_with([escalate])
    bridge = rig.bridge
    holder: list[CRSimulator] = []
    spoken: list[asyncio.Task[Any]] = []
    real_word = bridge._keypad_word

    def word(kind: str, lang: str) -> str:  # under the keypad action's lock: a spoken turn queues now
        session = holder[0].session
        spoken.append(asyncio.create_task(bridge._send(session, bridge.run_turn(session, "a person please"))))
        return real_word(kind, lang)

    monkeypatch.setattr(bridge, "_keypad_word", word)

    async def go() -> CRSimulator:
        sim = rig.sim()
        holder.append(sim)
        await sim.connect()
        await sim.dtmf("1")
        await wait_for(lambda: ended(sim), 3)
        await asyncio.gather(*spoken)
        await settle(sim)
        await sim.hangup()
        return sim

    sim = run(go())
    assert user_texts(rig) == ["a person please"]  # no model call for "Yes"
    assert len(end_frames(sim)) == 1
    assert [ln["action"] for ln in bridge_events(caplog, "keypad_action")] == ["yes"]
    assert "handoff_pending" in {ln["cause"] for ln in bridge_events(caplog, "keypad_ignored")}


@TIMED
def test_session_end_cancels_a_keypad_turn_and_creates_no_lock() -> None:
    """Keypad work doesn't outlive its session (and never re-creates its lock)."""
    rig = rig_with([say("Part one.", " Part two.", " Part three.")], delay_s=0.1)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await sim.dtmf("1")
        await wait_for(lambda: "Part one." in texts(sim), 3)
        rig.bridge.session_end(CALL_SID, "closed")
        await settle(sim, 0.5)
        await sim.hangup()
        return sim

    sim = run(go())
    assert texts(sim) == ["Part one."]
    assert CALL_SID not in rig.bridge._locks and CALL_SID not in rig.bridge._keypad_tasks


def test_session_end_while_a_keypad_action_waits_for_the_lock() -> None:
    rig = rig_with([say(REPLY)])

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        lock = rig.bridge._locks[CALL_SID]
        await lock.acquire()  # a turn still running
        await sim.dtmf("1")
        await asyncio.sleep(WINDOW * 3)  # the key is due: its action waits for the lock
        assert rig.bridge._keypad_tasks.get(CALL_SID)
        rig.bridge.session_end(CALL_SID, "closed")
        lock.release()
        await settle(sim)
        await sim.hangup()

    run(go())
    assert rig.agent.calls == []
    assert CALL_SID not in rig.bridge._locks and CALL_SID not in rig.bridge._keypad_tasks


def test_buffers_are_dropped_at_session_end() -> None:
    rig = rig_with()

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.dtmf("5")
        await wait_for(lambda: CALL_SID in rig.bridge._keypad, 1)
        rig.bridge.session_end(CALL_SID, "closed")  # the app's relay session does this on close
        await sim.hangup()

    run(go())
    assert CALL_SID not in rig.bridge._keypad and CALL_SID not in rig.bridge._keypad_wakes


# --------------------------------------------------------------------------- the setting


def test_setting_off_passes_the_frame_on_and_nothing_acts(caplog: pytest.LogCaptureFixture) -> None:
    rig = rig_with([say(REPLY)], on=False)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        for d in "1095":
            await sim.dtmf(d)
            await asyncio.sleep(WINDOW * 2)
        await settle(sim)
        await sim.hangup()
        return sim

    sim = run(go())
    assert texts(sim) == [] and not ended(sim) and rig.agent.calls == []
    assert bridge_events(caplog, "keypad_action") == [] and rig.bridge._keypad == {}


class _Inner:
    def __init__(self, frames: list[Any]) -> None:
        self.frames = frames

    async def receive_json(self) -> Any:
        return self.frames.pop(0)


class _Bridge:
    def __init__(self, on: bool) -> None:
        self.keypad_enabled = on
        self.inbound: list[tuple[str, str | None]] = []
        self.keys: list[str] = []

    def on_inbound(self, kind: str, call_sid: str | None) -> None:
        self.inbound.append((kind, call_sid))

    def on_keypad(self, call_sid: str | None, digit: str) -> None:
        self.keys.append(digit)


@pytest.mark.parametrize("on", [True, False])
def test_the_adapter_swallows_dtmf_only_when_on(on: bool) -> None:
    dtmf = {"type": "dtmf", "digit": "7"}
    prompt = {"type": "prompt", "voicePrompt": "hello", "last": True}
    bridge = _Bridge(on)
    peek = PeekingWebSocket(_Inner([dtmf, prompt]), bridge)  # type: ignore[arg-type]
    peek.verified, peek.call_sid = True, CALL_SID

    async def go() -> list[Any]:
        return [await peek.receive_json() for _ in range(1 if on else 2)]

    got = run(go())
    if on:
        assert got == [prompt]  # TAC never sees the key
        assert bridge.keys == ["7"]
    else:
        assert got == [dtmf, prompt]  # today's behaviour
        assert bridge.keys == []
    assert bridge.inbound == [("dtmf", CALL_SID), ("prompt", CALL_SID)]  # the silence reset either way


def test_the_adapter_notes_an_end_frame_before_sending_it() -> None:
    """The hand-off ``end`` resend's guard: only a real ``end`` frame is noted, before it is sent."""
    events: list[Any] = []

    class Out:
        async def send_text(self, data: str) -> None:
            events.append(json.loads(data)["type"])

    class Noting:
        keypad_enabled = True

        def on_relay_end(self, call_sid: str | None) -> None:
            events.append(("noted", call_sid))

    peek = PeekingWebSocket(Out(), Noting())  # type: ignore[arg-type]
    peek.call_sid = CALL_SID

    async def go() -> None:
        await peek.send_text(json.dumps({"type": "text", "token": '{"type":"end"}', "last": False}))
        await peek.send_text('{"type":"end","handoffData":"{}"}')

    run(go())
    assert events == ["text", ("noted", CALL_SID), "end"]
    # A bridge without the hook (the stub above): frames pass through unchanged.
    run(PeekingWebSocket(Out(), _Bridge(True)).send_text('{"type":"end"}'))  # type: ignore[arg-type]


def test_the_adapter_ignores_dtmf_before_setup() -> None:
    dtmf = {"type": "dtmf", "digit": "1"}
    bridge = _Bridge(True)
    peek = PeekingWebSocket(_Inner([dtmf]), bridge)  # type: ignore[arg-type]
    assert run(peek.receive_json()) == dtmf
    assert bridge.keys == [] and bridge.inbound == []


@pytest.mark.parametrize(("raw", "want"), [(None, True), ("", True), (" ON ", True), ("off", False), ("Off", False)])
def test_the_env_setting(raw: str | None, want: bool) -> None:
    env = {} if raw is None else {AI_KEYPAD_ENV: raw}
    assert keypad_setting_from_env(env) is want


def test_a_bad_env_value_stops_start_up_naming_the_variable() -> None:
    with pytest.raises(RuntimeError) as info:
        keypad_setting_from_env({AI_KEYPAD_ENV: "maybe-41111"})
    assert AI_KEYPAD_ENV in str(info.value) and "41111" not in str(info.value)


# --------------------------------------------------------------------------- leak scan


def test_no_digit_reaches_logs_journal_or_the_agent(caplog: pytest.LogCaptureFixture) -> None:
    rig = rig_with([say(REPLY), say(REPLY)])
    entry = "4738562"  # a card-number-like run: one hint, nothing else

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("hello")
        for d in entry:
            await sim.dtmf(d)
        await wait_for(lambda: HINT_EN in texts(sim), 3)
        await sim.dtmf("1")
        await wait_for(lambda: len(rig.agent.calls) == 2, 3)
        await settle(sim)
        await sim.hangup()
        return sim

    sim = run(go())
    assert user_texts(rig) == ["hello", "Yes"]
    pattern = value_pattern(entry)
    surfaces = {
        "logs": text_of(caplog.records),
        "journal": json.dumps(rig.events(), default=str),
        "call state": json.dumps(rig.call(), default=str),
        "agent": rig.agent.all_inputs_text(),
        "relay": json.dumps(sim.socket.sent),
    }
    for name, text in surfaces.items():
        assert not pattern.search(text.casefold()), name
    assert_keypad_lines_clean(caplog)


def test_keys_act_the_same_on_agentcore_hosting(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.component import test_bridge_fakes as fakes

    monkeypatch.setattr(fakes, "SIM_HOSTING", "agentcore")
    rig = rig_with([say(REPLY)])

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await sim.dtmf("1")
        await wait_for(lambda: REPLY in texts(sim), 3)
        await settle(sim)
        await sim.hangup()
        return sim

    run(go())
    assert user_texts(rig) == ["Yes"]


# --------------------------------------------------------------------------- a key pressed while TAC's session is still being set up (hold, early wake-ups)
#
# Call CAac1177…: the welcome plays at once, but TAC's session only exists once its Orchestrator
# conversation lookup is done (seconds later). The first 1 was dropped as ``no_session``;
# the second 1, with a live session, did nothing and logged nothing: uvloop (uvicorn's loop in
# production) runs a ``call_later`` callback up to about a millisecond before ``time.monotonic()``
# reaches its due time, so the buffer's ``poll`` returned ``None`` and the key was lost silently.


class _SlowOrchestrator(FakeOrchestrator):
    """TAC's conversation lookup takes ``delay_s`` (the live call: about 5 s after the socket opened)."""

    def __init__(self, delay_s: float) -> None:
        super().__init__()
        self.delay_s = delay_s

    async def list_conversations(self, channel_id: str | None = None, status: Any = None, **kw: Any):
        await asyncio.sleep(self.delay_s)
        return await super().list_conversations(channel_id, status, **kw)


def slow_rig(script: list[Any] | None = None, *, delay_s: float, wait_s: float = 2.0) -> Any:
    rig = make_rig(script=script or [], policy_doc=policy(),
                   config=BridgeConfig(ai_keypad=True, session_wait_s=wait_s, session_poll_s=0.01))
    rig.tac.conversation_orchestrator_client = _SlowOrchestrator(delay_s)  # type: ignore[assignment]
    write_call_state(rig.store, language="en-US")
    return rig


async def connect_early(sim: CRSimulator) -> asyncio.Task[None]:
    """``setup`` sent, the bridge's session started, TAC's session not there yet."""
    connecting = asyncio.create_task(sim.connect())
    await wait_for(lambda: CALL_SID in sim.bridge._speech, 1)
    assert sim.voice.get_conversation_session_by_call_sid(CALL_SID) is None
    return connecting


def early_wakes(monkeypatch: pytest.MonkeyPatch, loop: asyncio.AbstractEventLoop, early_s: float) -> None:
    """The keypad wake-up runs ``early_s`` before its due time, as uvloop's millisecond timers can."""
    original = loop.call_later

    def call_later(delay: float, callback: Any, *args: Any, **kwargs: Any) -> asyncio.TimerHandle:
        if getattr(callback, "__name__", "") == "_keypad_wake":
            delay = max(0.0, delay - early_s)
        return original(delay, callback, *args, **kwargs)

    monkeypatch.setattr(loop, "call_later", call_later)


def test_a_key_before_the_session_is_ready_acts_once_it_is(caplog: pytest.LogCaptureFixture) -> None:
    """A key pressed while TAC's session is still being set up is held."""
    rig = slow_rig([say(REPLY)], delay_s=0.3)

    async def go() -> tuple[CRSimulator, float]:
        sim = rig.sim()
        connecting = await connect_early(sim)
        await sim.dtmf("1")  # right after the welcome, before TAC's session exists
        await connecting
        ready = asyncio.get_running_loop().time()
        await wait_for(lambda: REPLY in texts(sim), 3)
        took = asyncio.get_running_loop().time() - ready
        await settle(sim)
        await sim.hangup()
        return sim, took

    sim, took = run(go())
    assert user_texts(rig) == ["Yes"] and texts(sim) == [REPLY]
    assert [ln["action"] for ln in bridge_events(caplog, "keypad_action")] == ["yes"]
    assert bridge_events(caplog, "keypad_ignored") == []
    assert took < 0.5  # acted as soon as the session was there, not after another wait
    assert_keypad_lines_clean(caplog)


def test_a_key_held_for_a_session_that_never_comes_is_dropped_after_the_bound(
        caplog: pytest.LogCaptureFixture) -> None:
    rig = slow_rig([say(REPLY)], delay_s=0.6, wait_s=0.2)

    async def go() -> CRSimulator:
        sim = rig.sim()
        connecting = await connect_early(sim)
        await sim.dtmf("1")
        await asyncio.sleep(0.4)  # past the bound, the session still not there
        assert [ln["cause"] for ln in bridge_events(caplog, "keypad_ignored")] == ["session_not_ready"]
        await connecting
        await settle(sim, 0.2)
        await sim.dtmf("1")  # a key after the bound with a live session acts as usual
        await wait_for(lambda: REPLY in texts(sim), 3)
        await settle(sim)
        await sim.hangup()
        return sim

    run(go())
    assert user_texts(rig) == ["Yes"]  # the held key never acted late
    assert len(bridge_events(caplog, "keypad_ignored")) == 1
    assert_keypad_lines_clean(caplog)


def test_a_key_after_the_bound_with_no_session_is_dropped_at_once(caplog: pytest.LogCaptureFixture) -> None:
    rig = slow_rig([say(REPLY)], delay_s=0.4, wait_s=0.0)

    async def go() -> None:
        sim = rig.sim()
        connecting = await connect_early(sim)
        await sim.dtmf("1")
        await wait_for(lambda: bool(bridge_events(caplog, "keypad_ignored")), 1)
        await connecting
        await settle(sim)
        await sim.hangup()

    run(go())
    assert [ln["cause"] for ln in bridge_events(caplog, "keypad_ignored")] == ["no_session"]
    assert rig.agent.calls == [] and CALL_SID not in rig.bridge._keypad_wakes


def test_pressing_one_and_saying_yes_before_the_session_is_ready_is_one_turn(
        caplog: pytest.LogCaptureFixture) -> None:
    """TAC holds a spoken prompt until its conversation lookup is done; the key and the prompt are
    one turn, the spoken one, as when the session is ready."""
    rig = slow_rig([say(REPLY), say("Second answer.")], delay_s=0.3)

    async def go() -> CRSimulator:
        sim = rig.sim()
        connecting = await connect_early(sim)
        await sim.dtmf("1")
        await sim.prompt_nowait("yes")
        await connecting
        await wait_for(lambda: REPLY in texts(sim), 3)
        await settle(sim, WINDOW * 4)
        await sim.hangup()
        return sim

    sim = run(go())
    assert user_texts(rig) == ["yes"] and texts(sim) == [REPLY]
    assert bridge_events(caplog, "keypad_action") == []
    assert [ln["cause"] for ln in bridge_events(caplog, "keypad_ignored")] == ["prompt"]


def test_a_key_held_for_the_session_is_still_dropped_on_a_pending_hand_off(
        caplog: pytest.LogCaptureFixture) -> None:
    rig = slow_rig([say(REPLY)], delay_s=0.3)

    async def go() -> CRSimulator:
        sim = rig.sim()
        connecting = await connect_early(sim)
        await sim.dtmf("1")
        rig.bridge._speech[CALL_SID].handoff = True  # a hand-off noted (_note_handoff) while the key is held
        await connecting
        await settle(sim, WINDOW * 4)
        await sim.hangup()
        return sim

    run(go())
    assert rig.agent.calls == [] and bridge_events(caplog, "keypad_action") == []
    assert "handoff_pending" in {ln["cause"] for ln in bridge_events(caplog, "keypad_ignored")}


@pytest.mark.parametrize("early_s", [0.0005, 0.002, 0.004])
def test_a_wake_up_that_runs_before_the_due_time_still_acts(
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, early_s: float) -> None:
    """Problem B: the wake-up ran before ``poll`` saw the entry complete; the key must still act."""
    rig = rig_with([say(REPLY)])

    async def go() -> CRSimulator:
        early_wakes(monkeypatch, asyncio.get_running_loop(), early_s)
        sim = rig.sim()
        await sim.connect()
        await sim.dtmf("1")
        await wait_for(lambda: REPLY in texts(sim), 1)
        await settle(sim)
        await sim.hangup()
        return sim

    run(go())
    assert user_texts(rig) == ["Yes"]
    assert [ln["action"] for ln in bridge_events(caplog, "keypad_action")] == ["yes"]


def test_the_live_call_on_uvloop_with_the_real_window(monkeypatch: pytest.MonkeyPatch,
                                                      caplog: pytest.LogCaptureFixture) -> None:
    """The live call, on the production loop (uvloop) and the real 1.5 s window: the first 1 comes
    before TAC's session (held, acts once it is there), the second 1 with a live session acts about
    1.5 s after it is pressed. Wake-ups are also forced 1 ms early, as uvloop's timers can run."""
    uvloop = pytest.importorskip("uvloop")
    monkeypatch.setattr(keypad, "WINDOW_S", 1.5)
    rig = slow_rig([say(REPLY), say("Second answer.")], delay_s=0.3, wait_s=12.0)
    seen: dict[str, float] = {}

    async def go() -> CRSimulator:
        loop = asyncio.get_running_loop()
        assert type(loop).__module__.startswith("uvloop")
        early_wakes(monkeypatch, loop, 0.001)
        sim = rig.sim()
        connecting = await connect_early(sim)
        await sim.dtmf("1")
        await connecting
        await wait_for(lambda: REPLY in texts(sim), 3)
        await settle(sim, 0.1)
        pressed = loop.time()
        await sim.dtmf("1")
        await wait_for(lambda: len(rig.agent.calls) == 2, 3)
        seen["took"] = loop.time() - pressed
        await settle(sim)
        await sim.hangup()
        return sim

    uvloop.run(go())
    assert user_texts(rig) == ["Yes", "Yes"]
    assert 1.45 <= seen["took"] < 1.9
    assert [ln["action"] for ln in bridge_events(caplog, "keypad_action")] == ["yes", "yes"]
    assert bridge_events(caplog, "keypad_ignored") == []


def test_no_early_return_drops_a_pending_key_silently(caplog: pytest.LogCaptureFixture) -> None:
    """Every exit of the wake-up's work that drops a pending key logs why (never a digit)."""
    rig = rig_with([say(REPLY)])

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        with_context(caplog)
        bridge = rig.bridge
        # the session's lock gone (the session ended) while the key waited
        buffer = bridge._keypad.setdefault(CALL_SID, keypad.KeypadBuffer())
        buffer.on_digit("1", 0.0)
        lock = bridge._locks.pop(CALL_SID)
        await bridge._keypad_due(CALL_SID, buffer)
        assert not buffer.pending
        bridge._locks[CALL_SID] = lock
        # a buffer that is no longer the call's (the session restarted)
        stale = keypad.KeypadBuffer()
        stale.on_digit("1", 0.0)
        await bridge._keypad_due(CALL_SID, stale)
        assert not stale.pending
        stale.on_digit("1", 0.0)
        bridge._keypad_wake(CALL_SID, stale)
        assert not stale.pending
        # the session ends with a key not yet acted on
        bridge.on_keypad(CALL_SID, "2")
        bridge.session_end(CALL_SID, "closed")
        await settle(sim)
        await sim.hangup()

    run(go())
    causes = [ln["cause"] for ln in bridge_events(caplog, "keypad_ignored")]
    assert causes == ["no_session", "no_session", "no_session", "session_end"]
    assert all(ln["call_sid"] == CALL_SID for ln in bridge_events(caplog, "keypad_ignored"))
    assert rig.agent.calls == []
    assert_keypad_lines_clean(caplog)
