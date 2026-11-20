"""The silence watch (repeat, then escalate or hang up) and ``relay_interrupt`` logging
on both agent backends (parity rig).

Real TAC through the ConversationRelay simulator and ``PeekingWebSocket``. Timings are real and
short: ``silence_timeout_ms`` of 120 ms, ``chars_per_s`` 30, and the 0.3 s TTS start estimate
shortened to 20 ms for these tests, so a short message's silence expires in a few hundred ms.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import pytest

from owlbank import last_spoken
from tac_bridge import silence
from tests.component.masking_fixtures import unmasked_policy_doc
from tests.component.test_bridge_fakes import (
    CALL_SID,
    NOW,
    SEALER,
    CRSimulator,
    end_frame,
    make_rig,
    run,
    say,
    wait_for,
    write_call_state,
)
from tests.logutil import lines, text_of

GREETING = "Paid. Anything else?"
SECRET = "my secret words"
BRIDGE_LOG = "tac_bridge.bridge"
APP_LOG = "tac_bridge.app"


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(silence, "TTS_START_S", 0.02)
    caplog.set_level(logging.INFO)


def policy(**masking: Any) -> dict[str, Any]:
    doc = unmasked_policy_doc()  # no masking phrases, no timed acknowledgement
    doc["masking"] = {**doc["masking"], "chars_per_s": 30, "silence_timeout_ms": 120,
                      "silence_max_repeats": 1, "silence_action": "escalate",
                      "greeting_silence_timeout_ms": 120, **masking}
    return doc


def rig_with(script: list[Any] | None = None, *, greeting: str | None = GREETING,
             call_fields: dict[str, Any] | None = None, **masking: Any) -> Any:
    rig = make_rig(script=script or [], policy_doc=policy(**masking))
    sealed = last_spoken.seal(SEALER, greeting, call_sid=CALL_SID, now=NOW) if greeting else None
    write_call_state(rig.store, last_spoken=sealed, **(call_fields or {}))
    return rig


def texts(sim: CRSimulator) -> list[str]:
    """Every text token sent on the socket so far (non-empty ones)."""
    return [f["token"] for f in sim.socket.sent if f.get("type") == "text" and f.get("token")]


def ended(sim: CRSimulator) -> bool:
    return any(f.get("type") == "end" for f in sim.socket.sent)


def events(rig: Any, etype: str) -> list[dict[str, Any]]:
    return [e for e in rig.events() if e["type"] == etype]


def bridge_events(caplog: pytest.LogCaptureFixture, name: str) -> list[dict[str, Any]]:
    return [ln for ln in lines(caplog.records, BRIDGE_LOG) if ln["event"] == name]


# --------------------------------------------------------------------------- greeting, repeat, action


def test_greeting_then_silence_repeats_it_then_escalates_with_no_input(caplog: pytest.LogCaptureFixture) -> None:
    rig = rig_with()

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        await wait_for(lambda: ended(sim), 4)
        await asyncio.sleep(0.4)  # nothing more after the action
        frames = list(sim.socket.sent)
        await sim.hangup()
        return frames

    frames = run(go())
    assert [f.get("token") for f in frames if f.get("type") == "text" and f.get("token")] == [GREETING]
    assert [f for f in frames if f.get("type") == "text"][-1] == {"type": "text", "token": "", "last": True}
    end = end_frame(frames)
    assert end is not None
    assert end["data"]["reason"] == "escalate" and end["data"]["escalationReason"] == "no_input"
    assert end["data"]["tool"] == "escalate" and end["data"]["journal"]
    # the existing escalation path: the reason and journal ref stored for the front door
    assert rig.call()["escalation"]["reason"] == "no_input"
    assert rig.call()["escalation"]["journal_ref"] == end["data"]["journal"]
    assert [e["reason"] for e in events(rig, "escalation")] == ["no_input"]
    repeats = events(rig, "prompt_repeat")
    assert len(repeats) == 1 and repeats[0]["repeat"] == 1 and repeats[0]["what"] == "greeting"
    # a repeat is never sent to the agent and isn't a turn
    assert rig.agent.calls == []
    assert bridge_events(caplog, "turn") == []
    # logged without the text
    (rep,) = bridge_events(caplog, "prompt_repeat")
    assert rep["what"] == "greeting" and rep["chars"] == len(GREETING) and rep["repeat"] == 1
    (act,) = bridge_events(caplog, "silence_action")
    assert act["action"] == "escalate" and act["repeats"] == 1
    assert GREETING not in text_of([r for r in caplog.records if r.name.startswith("tac_bridge")])


def test_hangup_action_hands_off_with_no_input_hangup() -> None:
    rig = rig_with(silence_action="hangup")

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        await wait_for(lambda: ended(sim), 4)
        frames = list(sim.socket.sent)
        await sim.hangup()
        return frames

    frames = run(go())
    assert [f.get("token") for f in frames if f.get("type") == "text" and f.get("token")] == [GREETING]
    end = end_frame(frames)
    assert end is not None
    assert end["data"]["reason"] == "no_input_hangup" and "escalationReason" not in end["data"]
    assert events(rig, "escalation") == []
    assert [e["reason"] for e in events(rig, "handoff")] == ["no_input_hangup"]


def test_no_repeats_configured_goes_straight_to_the_action() -> None:
    rig = rig_with(silence_max_repeats=0)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        await wait_for(lambda: ended(sim), 4)
        frames = list(sim.socket.sent)
        await sim.hangup()
        return frames

    frames = run(go())
    assert [f.get("token") for f in frames if f.get("type") == "text" and f.get("token")] == []
    assert end_frame(frames)["data"]["escalationReason"] == "no_input"  # type: ignore[index]


def test_timeout_zero_never_fires() -> None:
    rig = rig_with([say("OK.")], silence_timeout_ms=0)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("hello")
        await asyncio.sleep(0.6)
        await sim.hangup()
        return sim

    sim = run(go())
    assert texts(sim) == ["OK."] and not ended(sim)
    assert events(rig, "prompt_repeat") == []


# --------------------------------------------------------------------------- the opening greeting


def _greeting_repeat_after(rig: Any) -> float:
    """Seconds from connect to the greeting's repeat being sent."""

    async def go() -> float:
        sim = rig.sim()
        start = time.monotonic()
        await sim.connect()
        await wait_for(lambda: texts(sim) == [GREETING], 5)
        took = time.monotonic() - start
        await sim.hangup()
        return took

    return run(go())


def test_first_session_greeting_waits_the_greeting_timeout(caplog: pytest.LogCaptureFixture) -> None:
    # Calls CA…74b01a, CA…0e8149: the repeat at 4 s met the caller starting to speak. The first AI
    # session's greeting waits greeting_silence_timeout_ms (here 900 ms) instead of 120 ms. The
    # session's own ai_sessions write races the setup read; the decision is
    # taken at the nonce check, before it.
    rig = rig_with(greeting_silence_timeout_ms=900)
    speech_end = 0.02 + len(GREETING) / 30
    took = _greeting_repeat_after(rig)
    assert took >= speech_end + 0.9 - 0.05
    (rep,) = bridge_events(caplog, "prompt_repeat")
    assert rep["what"] == "greeting" and rep["waited_ms"] == 900 and rep["timeout_ms"] == 120


def test_later_session_greeting_uses_the_silence_timeout(caplog: pytest.LogCaptureFixture) -> None:
    # an earlier AI session (ai_sessions written by it): "Anything else?" keeps silence_timeout_ms
    rig = rig_with(greeting_silence_timeout_ms=2000, call_fields={"ai_sessions": 1})
    took = _greeting_repeat_after(rig)
    assert took < 1.5  # 0.69 s of speech + 0.12 s, far below the 2 s greeting timeout
    (rep,) = bridge_events(caplog, "prompt_repeat")
    assert rep["what"] == "greeting" and rep["waited_ms"] == 120


def test_after_the_greeting_repeat_the_silence_timeout_applies(caplog: pytest.LogCaptureFixture) -> None:
    # the repeat re-arms with silence_timeout_ms, then the action follows
    rig = rig_with(greeting_silence_timeout_ms=600)

    async def go() -> tuple[float, float]:
        sim = rig.sim()
        await sim.connect()
        await wait_for(lambda: texts(sim) == [GREETING], 5)
        repeated = time.monotonic()
        await wait_for(lambda: ended(sim), 5)
        acted = time.monotonic()
        await sim.hangup()
        return repeated, acted

    repeated, acted = run(go())
    assert acted - repeated < 0.02 + len(GREETING) / 30 + 0.45  # 120 ms after the repeat, not 600
    (act,) = bridge_events(caplog, "silence_action")
    assert act["action"] == "escalate"


def test_greeting_timeout_zero_means_no_greeting_repeat() -> None:
    rig = rig_with([say("OK.")], greeting_silence_timeout_ms=0)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await asyncio.sleep(1.2)  # well past the greeting's end + 120 ms
        assert texts(sim) == [] and not ended(sim)
        await sim.prompt("hello")
        await wait_for(lambda: texts(sim) == ["OK.", "OK."], 3)  # later messages: 120 ms
        await sim.hangup()
        return sim

    run(go())


# --------------------------------------------------------------------------- input resets


def test_caller_speaking_after_the_repeat_resets_the_count() -> None:
    rig = rig_with([say("Sure, what else?")])

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await wait_for(lambda: texts(sim) == [GREETING], 4)  # the greeting's one repeat
        await sim.prompt("pay my bill")  # caller input: count back to 0
        await wait_for(lambda: ended(sim), 4)
        await sim.hangup()
        return sim

    sim = run(go())
    # the turn's reply is repeated once (the count reset), then the action
    assert texts(sim) == [GREETING, "Sure, what else?", "Sure, what else?"]
    reps = events(rig, "prompt_repeat")
    assert [(e["repeat"], e["what"]) for e in reps] == [(1, "greeting"), (1, "agent")]
    assert "text" in reps[1] and "rendered_text" not in reps[1]  # agent words: journaled like an ai_turn
    assert end_frame(sim.socket.sent)["data"]["escalationReason"] == "no_input"  # type: ignore[index]
    assert len(rig.agent.calls) == 1  # the repeat never reached the agent
    assert rig.call().get("last_spoken") is None  # cleared once a turn spoke


def test_dtmf_with_no_turn_resets_the_count_then_repeats_and_acts(caplog: pytest.LogCaptureFixture) -> None:
    # A key press during the AI segment starts no turn (TAC drops dtmf): the count resets, and
    # after the timeout counted from the key press the last message is repeated, then the action.
    rig = rig_with()

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await wait_for(lambda: texts(sim) == [GREETING], 4)  # repeat 1 of 1
        await sim.socket.inbound.put({"type": "dtmf", "digit": "7"})
        await wait_for(lambda: ended(sim), 4)
        await sim.hangup()
        return sim

    sim = run(go())
    assert texts(sim) == [GREETING, GREETING]  # repeated again: the key press reset the count
    assert [e["repeat"] for e in events(rig, "prompt_repeat")] == [1, 1]
    assert end_frame(sim.socket.sent)["data"]["escalationReason"] == "no_input"  # type: ignore[index]
    assert rig.agent.calls == []
    (reset,) = bridge_events(caplog, "silence_reset")
    assert reset["kind"] == "dtmf" and reset["repeats"] == 1
    assert "digit" not in text_of(caplog.records)


def test_dtmf_during_the_ai_segment_counts_the_timeout_from_the_key_press() -> None:
    rig = rig_with([say("Anything else?")], greeting=None, silence_timeout_ms=300)

    async def go() -> tuple[CRSimulator, float, float]:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("hi")
        await asyncio.sleep(0.2)  # the reply is still estimated to be playing
        pressed = time.monotonic()
        await sim.socket.inbound.put({"type": "dtmf", "digit": "5"})
        await wait_for(lambda: texts(sim) == ["Anything else?", "Anything else?"], 4)
        repeated = time.monotonic()
        await wait_for(lambda: ended(sim), 4)
        await sim.hangup()
        return sim, pressed, repeated

    sim, pressed, repeated = run(go())
    assert repeated - pressed >= 0.3  # inbound time + silence_timeout_ms
    assert [(e["repeat"], e["what"]) for e in events(rig, "prompt_repeat")] == [(1, "agent")]
    assert end_frame(sim.socket.sent)["data"]["escalationReason"] == "no_input"  # type: ignore[index]
    assert len(rig.agent.calls) == 1


# --------------------------------------------------------------------------- never mid-turn


def test_no_firing_while_a_turn_runs_and_the_turn_s_answer_is_what_repeats() -> None:
    # The acknowledgement plays mid-turn (a scripted phrase), the turn runs far longer than the
    # greeting's silence timeout, and the answer comes last: nothing fires during the turn, and the
    # repeat is the answer, not the phrase.
    rig = rig_with([say("Your balance is ready.")], after_ms=100)
    rig.agent.delay_s = 0.35

    async def go() -> tuple[CRSimulator, list[str]]:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("balance please", timeout=5)
        during = texts(sim)
        await wait_for(lambda: ended(sim), 5)
        await sim.hangup()
        return sim, during

    sim, during = run(go())
    assert not any(f.get("type") == "end" for f in sim.socket.sent[: len(during)])
    said = texts(sim)
    assert said[-2:] == ["Your balance is ready.", "Your balance is ready."]
    ack = said[0]
    assert ack.strip() and ack.strip() != GREETING and "Your balance" not in ack  # the scripted phrase
    reps = events(rig, "prompt_repeat")
    assert [(e["repeat"], e["what"]) for e in reps] == [(1, "agent")]


# --------------------------------------------------------------------------- last_spoken missing


def test_no_last_spoken_means_no_greeting_repeat_but_the_action_still_follows_a_turn() -> None:
    rig = rig_with([say("Anything else?")], greeting=None)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await asyncio.sleep(0.6)
        assert texts(sim) == [] and not ended(sim)  # nothing to repeat, nothing armed
        await sim.prompt("hi")
        await wait_for(lambda: ended(sim), 4)
        await sim.hangup()
        return sim

    sim = run(go())
    assert texts(sim) == ["Anything else?", "Anything else?"]
    assert end_frame(sim.socket.sent)["data"]["escalationReason"] == "no_input"  # type: ignore[index]


def test_an_expired_last_spoken_is_not_repeated() -> None:
    rig = make_rig(script=[], policy_doc=policy())
    sealed = last_spoken.seal(SEALER, GREETING, call_sid=CALL_SID, now=NOW - last_spoken.TTL_S - 1)
    write_call_state(rig.store, last_spoken=sealed)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await asyncio.sleep(0.6)
        await sim.hangup()
        return sim

    sim = run(go())
    assert texts(sim) == [] and not ended(sim)


# --------------------------------------------------------------------------- relay_interrupt


def test_relay_interrupt_is_logged_with_what_played_and_never_the_words(caplog: pytest.LogCaptureFixture) -> None:
    rig = rig_with([say("Here are your options.")])

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.interrupt(heard=f"Your payment {SECRET}", ms=420)  # during the greeting
        await sim.prompt("options")
        await sim.interrupt(heard=SECRET)  # during the agent's answer
        await sim.socket.inbound.put({"type": "interrupt"})  # no duration from ConversationRelay
        await sim.idle()
        await asyncio.sleep(0.05)
        await sim.hangup()

    run(go())
    got = [ln for ln in lines(caplog.records, APP_LOG) if ln["event"] == "relay_interrupt"]
    assert [(ln["playing"], ln["ms_played"]) for ln in got] == [("greeting", 420), ("agent", 300),
                                                                ("agent", None)]
    assert SECRET not in text_of(caplog.records)


def test_relay_interrupt_names_the_scripted_phrase_that_was_playing(caplog: pytest.LogCaptureFixture) -> None:
    rig = rig_with()

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await wait_for(lambda: texts(sim) == [GREETING], 4)
        await sim.interrupt(ms=100)  # during the repeat
        await sim.idle()
        await sim.hangup()

    run(go())
    got = [ln for ln in lines(caplog.records, APP_LOG) if ln["event"] == "relay_interrupt"]
    assert [ln["playing"] for ln in got] == ["greeting"]  # the repeat of the greeting
    assert [e["what"] for e in events(rig, "prompt_repeat")] == ["greeting"]
    (rep,) = events(rig, "prompt_repeat")
    assert rep["rendered_text"] == GREETING and "prompt_key" not in rep  # as the front door journaled it


# --------------------------------------------------------------------------- collided repeats


def test_a_repeat_interrupted_at_once_is_not_counted_and_is_logged_collided(
        caplog: pytest.LogCaptureFixture) -> None:
    # Call CA…74b01a: greeting, silence, the greeting's repeat, an interrupt 0 ms into it as the
    # caller starts speaking, then the caller's prompt and a turn.
    rig = rig_with([say("Sure, what else?")])

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await wait_for(lambda: texts(sim) == [GREETING], 4)  # the greeting's repeat
        await sim.interrupt(heard=SECRET, ms=0)  # at +0 ms: the caller was already speaking
        await sim.prompt("pay my bill")
        await wait_for(lambda: ended(sim), 4)
        await sim.hangup()
        return sim

    sim = run(go())
    # the turn's reply still gets its one repeat, then the action: the collided repeat didn't count
    assert texts(sim) == [GREETING, "Sure, what else?", "Sure, what else?"]
    assert [(e["repeat"], e["what"]) for e in events(rig, "prompt_repeat")] == [(1, "greeting"), (1, "agent")]
    assert end_frame(sim.socket.sent)["data"]["escalationReason"] == "no_input"  # type: ignore[index]
    assert len(rig.agent.calls) == 1
    (col,) = bridge_events(caplog, "prompt_repeat_collided")
    assert col["repeat"] == 1 and col["what"] == "greeting" and col["grace_ms"] == silence.REPEAT_GRACE_MS
    assert 0 <= col["ms_after_repeat"] <= silence.REPEAT_GRACE_MS
    assert bridge_events(caplog, "silence_reset") == []  # nothing counted was reset
    assert SECRET not in text_of(caplog.records) and GREETING not in text_of(
        [r for r in caplog.records if r.name.startswith("tac_bridge")])


def test_an_interrupt_after_the_grace_is_not_a_collision(caplog: pytest.LogCaptureFixture,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(silence, "REPEAT_GRACE_MS", 30)
    rig = rig_with([say("Sure, what else?")])

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await wait_for(lambda: texts(sim) == [GREETING], 4)
        await asyncio.sleep(0.08)  # past the 30 ms grace
        await sim.interrupt(ms=80)
        await sim.idle()
        await sim.hangup()

    run(go())
    assert bridge_events(caplog, "prompt_repeat_collided") == []
    (reset,) = bridge_events(caplog, "silence_reset")
    assert reset["kind"] == "interrupt" and reset["repeats"] == 1


# --------------------------------------------------------------------------- speaker events

ASK = "I am the OwlBank virtual agent. How can I help you today?"
BALANCE_ASK = "what's my checking balance"


async def speaker(sim: CRSimulator, name: str, value: str) -> None:
    """A ConversationRelay speaker event, in the shape assumed until the real shape is confirmed."""
    await sim.socket.inbound.put({"type": "info", "name": name, "value": value})
    await asyncio.sleep(0.01)


def app_events(caplog: pytest.LogCaptureFixture, name: str) -> list[dict[str, Any]]:
    return [ln for ln in lines(caplog.records, APP_LOG) if ln["event"] == name]


def test_replay_ca_d4f992_caller_speaking_past_the_old_deadline_is_not_cut_off(
        caplog: pytest.LogCaptureFixture) -> None:
    # Call CA…d4f992 (scaled: 4 s -> 400 ms, 7 s -> 600 ms; one repeat, as deployed then): the
    # greeting, silence, the greeting's repeat, the repeat plays out (agentSpeaking off), the
    # caller starts 300 ms later (3 s on the call) and keeps speaking well past the deadline the
    # repeat had (its estimated end + 400 ms), stops, and the prompt arrives. No escalation; the
    # agent's turn runs.
    rig = rig_with([say("Your checking balance is ready.")], greeting=ASK,
                   silence_timeout_ms=400, greeting_silence_timeout_ms=600, silence_max_repeats=1)
    repeat_estimate = 0.02 + len(ASK) / 30  # the repeat's estimated speaking time

    async def go() -> tuple[CRSimulator, bool, list[str]]:
        sim = rig.sim()
        await sim.connect()
        await speaker(sim, "agentSpeaking", "on")  # the welcome greeting
        await asyncio.sleep(0.1)
        await speaker(sim, "agentSpeaking", "off")
        await wait_for(lambda: texts(sim) == [ASK], 5)  # the greeting's repeat
        sent = time.monotonic()
        await speaker(sim, "agentSpeaking", "on")
        await asyncio.sleep(0.1)
        await speaker(sim, "agentSpeaking", "off")  # the repeat played out
        await asyncio.sleep(0.3)
        await speaker(sim, "clientSpeaking", "on")  # "what's my checking ..."
        old_deadline = sent + repeat_estimate + 0.4
        await asyncio.sleep(max(0.0, old_deadline - time.monotonic()) + 0.6)  # still speaking
        spoke_through = not ended(sim) and texts(sim) == [ASK]
        await speaker(sim, "clientSpeaking", "off")
        reply = await sim.prompt(BALANCE_ASK)
        said = [f["token"] for f in reply if f.get("type") == "text" and f.get("token")]
        await sim.hangup()
        sim.bridge.session_end(CALL_SID, "closed")  # what the app's /ws handler does (no second line)
        return sim, spoke_through, said

    sim, spoke_through, said = run(go())
    assert spoke_through, "nothing fired while the caller was speaking"
    assert said == ["Your checking balance is ready."]
    assert len(rig.agent.calls) == 1  # the turn ran on the caller's prompt
    assert end_frame(sim.socket.sent) is None
    assert bridge_events(caplog, "silence_action") == [] and events(rig, "escalation") == []
    assert [e["repeat"] for e in events(rig, "prompt_repeat")] == [1]
    (timing,) = bridge_events(caplog, "silence_timing")
    assert timing["mode"] == "speaker_events"
    got = app_events(caplog, "relay_speaker_event")
    assert [(ln["speaker_event"], ln["value"]) for ln in got] == [
        ("agentSpeaking", "on"), ("agentSpeaking", "off"), ("agentSpeaking", "on"), ("agentSpeaking", "off"),
        ("clientSpeaking", "on"), ("clientSpeaking", "off")]
    assert all(ln["recognised"] is True for ln in got)
    logged = text_of([r for r in caplog.records if r.name.startswith("tac_bridge")])
    assert ASK not in logged and BALANCE_ASK not in logged


def test_the_wait_counts_from_the_agent_speech_end_and_restarts_when_the_caller_stops() -> None:
    rig = rig_with(greeting=ASK, silence_timeout_ms=300, greeting_silence_timeout_ms=300,
                   silence_max_repeats=2)

    async def go() -> tuple[float, float, float, float]:
        sim = rig.sim()
        await sim.connect()
        await speaker(sim, "agentSpeaking", "on")
        await asyncio.sleep(1.2)  # the greeting plays far longer than its estimate (~0.85 s)
        await speaker(sim, "agentSpeaking", "off")
        agent_off = time.monotonic()
        await wait_for(lambda: texts(sim) == [ASK], 5)
        first = time.monotonic()
        await asyncio.sleep(0.55)  # past the collision grace
        await speaker(sim, "clientSpeaking", "on")  # a cough: no prompt follows
        await asyncio.sleep(0.8)
        assert texts(sim) == [ASK] and not ended(sim)
        await speaker(sim, "clientSpeaking", "off")
        caller_off = time.monotonic()
        await wait_for(lambda: texts(sim) == [ASK, ASK], 5)
        second = time.monotonic()
        await sim.hangup()
        return agent_off, first, caller_off, second

    agent_off, first, caller_off, second = run(go())
    assert first - agent_off >= 0.3 - 0.02  # from the reported end, not the estimate
    assert second - caller_off >= 0.3 - 0.02  # restarted from the caller's stop
    # the pause didn't change the count: the second repeat is number 2, then (later) the action
    assert [e["repeat"] for e in events(rig, "prompt_repeat")] == [1, 2]


def test_client_speaking_at_a_repeat_collides_like_an_interrupt(caplog: pytest.LogCaptureFixture) -> None:
    rig = rig_with(silence_max_repeats=1)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await wait_for(lambda: texts(sim) == [GREETING], 4)
        await sim.socket.inbound.put({"type": "info", "name": "clientSpeaking", "value": "on"})
        await asyncio.sleep(0.6)
        await speaker(sim, "clientSpeaking", "off")
        await wait_for(lambda: texts(sim) == [GREETING, GREETING], 4)  # the collided one didn't count
        await sim.hangup()
        return sim

    run(go())
    (col,) = bridge_events(caplog, "prompt_repeat_collided")
    assert col["cause"] == "client_speaking" and col["repeat"] == 1 and col["what"] == "greeting"
    assert 0 <= col["ms_after_repeat"] <= silence.REPEAT_GRACE_MS


def test_speaker_events_never_reach_tac_and_unrecognised_shapes_are_logged_without_values(
        caplog: pytest.LogCaptureFixture) -> None:
    rig = rig_with([say("OK.")], greeting=None)

    async def go() -> CRSimulator:
        sim = rig.sim()
        await sim.connect()
        await sim.socket.inbound.put({"type": "info", "name": "clientSpeaking", "value": SECRET})
        await sim.socket.inbound.put({"type": "info", "name": SECRET, "value": "on", "text": SECRET})
        await sim.socket.inbound.put({"type": "agentSpeaking", "value": "on"})  # the name as the type
        await sim.idle()
        await asyncio.sleep(0.05)
        await sim.hangup()
        return sim

    sim = run(go())
    assert rig.agent.calls == [] and texts(sim) == []  # nothing reached the agent as caller input
    got = app_events(caplog, "relay_speaker_event")
    assert [ln["recognised"] for ln in got] == [False, False, True]
    assert got[0]["type"] == "info" and got[0]["keys"] == ["name", "type", "value"]
    assert got[1]["keys"] == ["name", "text", "type", "value"]
    assert (got[2]["speaker_event"], got[2]["value"]) == ("agentSpeaking", "on")
    assert SECRET not in text_of(caplog.records)


def test_no_speaker_events_the_repeat_waits_from_its_estimated_end(caplog: pytest.LogCaptureFixture) -> None:
    # Fallback: the estimate, for the repeat too (from when it was sent, not before).
    rig = rig_with(greeting_silence_timeout_ms=300, silence_timeout_ms=300)
    estimate = 0.02 + len(GREETING) / 30

    async def go() -> tuple[float, float]:
        sim = rig.sim()
        await sim.connect()
        await wait_for(lambda: texts(sim) == [GREETING], 5)
        repeated = time.monotonic()
        await wait_for(lambda: ended(sim), 5)
        acted = time.monotonic()
        await sim.hangup()
        sim.bridge.session_end(CALL_SID, "closed")  # what the app's /ws handler does
        return repeated, acted

    repeated, acted = run(go())
    assert acted - repeated >= estimate + 0.3 - 0.03
    assert acted - repeated < estimate + 0.3 + 0.35
    (timing,) = bridge_events(caplog, "silence_timing")
    assert timing["mode"] == "estimate"
