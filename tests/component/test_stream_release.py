"""Stream the agent's text, through the bridge over real TAC.

The rigs here run a real :class:`~tac_bridge.agent_client_strands.StrandsAgentClient` in
``stream`` mode (the deployed default); the shared rigs (``tests.strands_parity``) stay in
``message`` mode, so the existing suite pins message mode unchanged. Steps are raw
ConverseStream events of one model message, with :class:`~tests.strands_parity._Pause` markers
where a test needs time between chunks. The simulator's socket records when each frame left.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import pytest

from tac_bridge import bridge as bridge_mod
from tac_bridge import silence
from tac_bridge.agent_client_strands import StrandsAgentClient
from tests.component.masking_fixtures import masked_policy_doc
from tests.component.test_bridge_fakes import (
    CRSimulator,
    FakeBackend,
    Rig,
    SimSocket,
    StrandsFakeAgent,
    end_frame,
    h_msg,
    h_text,
    h_tool,
    make_rig,
    run,
    say,
    spoken,
    wait_for,
    write_call_state,
)
from tests.component.test_silence_watch import policy as silence_policy
from tests.component.test_silence_watch import speaker
from tests.strands_fakes import FakeMemoryData, default_gateway, memory_factory, settings
from tests.strands_parity import ParityModel, _Pause, memory_for
from tests.logutil import lines, text_of

BRIDGE_LOG = "tac_bridge.bridge"
APOLOGY = "I'm sorry, something went wrong on our end. "


# --------------------------------------------------------------------------- the stream-mode rig


@dataclass
class StreamAgent(StrandsFakeAgent):
    """:class:`StrandsFakeAgent` with the real client in ``stream`` mode."""

    def _attach(self, memory: FakeMemoryData) -> None:
        self.model = ParityModel()
        self.memory_data = memory
        self.client = StrandsAgentClient(
            settings(memory_id="mem-1", text_release="stream", thinking="adaptive"),
            model_factory=lambda _s, _sid: self.model, gateway_tools=default_gateway(),
            session_manager_factory=memory_factory(memory))
        self.emitted: set[str] = set()


def stream_rig(script: list[Any], **kw: Any) -> Rig:
    rig = make_rig(**kw)
    agent = StreamAgent(list(script), memory=memory_for(rig.store))
    rig.agent = agent
    rig.bridge._agent = agent  # noqa: SLF001 - the rig's agent, swapped for the stream-mode one
    write_call_state(rig.store)
    return rig


class TimedSocket(SimSocket):
    """The simulator's socket, recording ``time.monotonic()`` of each frame sent."""

    def __init__(self) -> None:
        super().__init__()
        self.times: list[float] = []

    async def send_text(self, data: str) -> None:
        await super().send_text(data)
        self.times.append(time.monotonic())


def timed_sim(rig: Rig) -> tuple[CRSimulator, TimedSocket]:
    sim = rig.sim()
    sock = TimedSocket()
    sim.socket = sock
    return sim, sock


def chunks(*parts: str | float, i: int = 0) -> list[Any]:
    """One text block streamed as several deltas; a float is a pause (seconds) between them."""
    out: list[Any] = [{"contentBlockStart": {"contentBlockIndex": i, "start": {}}}]
    for p in parts:
        if isinstance(p, str):
            out.append({"contentBlockDelta": {"contentBlockIndex": i, "delta": {"text": p}}})
        else:
            out.append(_Pause(lambda s=p: s))
    out.append({"contentBlockStop": {"contentBlockIndex": i}})
    return out


def msg(*blocks: list[Any], stop: str = "end_turn") -> list[Any]:
    """One model message of text blocks that may hold pauses (``h_msg`` reads only dict events)."""
    return [{"messageStart": {"role": "assistant"}}, *(ev for b in blocks for ev in b),
            {"messageStop": {"stopReason": stop}}]


def token_time(sock: TimedSocket, token: str) -> float:
    (t,) = [at for f, at in zip(sock.sent, sock.times, strict=True)
            if f.get("type") == "text" and f.get("token") == token]
    return t


def end_time(sock: TimedSocket) -> float:
    (t,) = [at for f, at in zip(sock.sent, sock.times, strict=True) if f.get("type") == "end"]
    return t


def events(caplog: pytest.LogCaptureFixture, name: str) -> list[dict[str, Any]]:
    return [ln for ln in lines(caplog.records, BRIDGE_LOG) if ln["event"] == name]


def assert_held(held: float, estimate: float) -> None:
    """``end`` was held for about ``estimate`` seconds: wide enough for a loaded machine (a
    late wake-up only lengthens a hold), narrow enough to tell held from not held."""
    assert estimate * 0.8 <= held <= estimate + 1.0, (held, estimate)


def ai_turn_texts(rig: Rig) -> list[str]:
    return [e.get("text", "") for e in rig.events() if e["type"] == "ai_turn" and "text" in e]


def converse(rig: Rig, prompt: str) -> tuple[list[dict[str, Any]], TimedSocket]:
    async def go() -> tuple[list[dict[str, Any]], TimedSocket]:
        sim, sock = timed_sim(rig)
        await sim.connect()
        frames = await sim.prompt(prompt, timeout=5.0)
        await sim.hangup()
        return frames, sock

    return run(go())


# --------------------------------------------------------------------------- plain answers


def test_plain_answer_tokens_leave_before_the_message_ends(caplog: pytest.LogCaptureFixture) -> None:
    # the first chunk goes to ConversationRelay while the model is still writing the message
    # (message mode would send both chunks together at messageStop).
    rig = stream_rig([msg(chunks("Your savings ", 0.3, "balance is fine."))])
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, sock = converse(rig, "How is my savings?")
    assert spoken(frames) == "Your savings balance is fine."
    assert token_time(sock, "balance is fine.") - token_time(sock, "Your savings ") >= 0.3 * 0.8
    assert ai_turn_texts(rig) == ["Your savings balance is fine."]  # journaled whole
    (turn,) = events(caplog, "turn")
    assert turn["text_before_tool"] == 0
    assert events(caplog, "ai_text_spoken_before_tool") == []


def test_silent_marker_split_across_chunks_is_never_spoken() -> None:
    # the marker split over three chunks, and one ending a sentence.
    rig = stream_rig([h_msg(chunks("<", "silent", "/>")),
                      h_msg(chunks("Okay.", "<sil", "ent", "/>"))])

    async def go() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        sim, _ = timed_sim(rig)
        await sim.connect()
        first = await sim.prompt("hmm")
        second = await sim.prompt("okay then")
        await sim.hangup()
        return first, second

    first, second = run(go())
    assert spoken(first) == ""
    assert spoken(second) == "Okay."
    assert "<" not in spoken(first) + spoken(second)
    assert ai_turn_texts(rig) == ["Okay."]


# --------------------------------------------------------------------------- a late tool call (3.1-3.6)


def test_late_inline_tool_text_is_spoken_journaled_and_gets_no_tool_phrase(
        caplog: pytest.LogCaptureFixture) -> None:
    # 3.2-3.4, 3.6: list_recent_transactions is masked in data/policy.json (tool_after_ms 0), but
    # the turn is already answering, so no phrase; the tool runs; the spoken part is journaled.
    early = "Let me look at your checking account. "
    backend = FakeBackend({"list_recent_transactions": [{"account": "checking", "transactions": []}]})
    rig = stream_rig([
        h_msg(h_text(early), h_tool("list_recent_transactions", "tu_tx", i=1, account="checking", limit=3)),
        h_msg(h_text("You have no recent transactions on checking.")),
    ], backend=backend)
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "Any recent transactions?")
    assert spoken(frames) == early + "You have no recent transactions on checking."
    assert len(backend.calls) == 1
    assert ai_turn_texts(rig) == [early, "You have no recent transactions on checking."]
    kinds = [e["type"] for e in rig.events()]
    assert kinds.index("ai_turn") < kinds.index("tool_call")  # journaled as heard, before the tool
    (late,) = events(caplog, "ai_text_spoken_before_tool")
    assert (late["chars"], late["tool"], late["hop"], late["turn"]) == (
        len(early), "list_recent_transactions", 1, 1)
    assert [r.levelno for r in caplog.records
            if r.name == BRIDGE_LOG and r.getMessage() == "ai_text_spoken_before_tool"] == [logging.WARNING]
    assert events(caplog, "mask_spoken") == []
    assert [s.get("reason") for s in events(caplog, "mask_skipped")] == ["answer_started"]
    assert events(caplog, "ai_text_suppressed") == []
    (turn,) = events(caplog, "turn")
    assert turn["text_before_tool"] == 1
    assert early.strip() not in text_of(caplog.records)  # never the text in logs


def test_late_handoff_holds_end_for_the_released_text(caplog: pytest.LogCaptureFixture) -> None:
    # no hand-off phrase after released text, so end waits for the text's estimated end
    # (0.3 s + chars / chars_per_s), with handoff_wait_ms well above it.
    early = "Connecting you now."
    rig = stream_rig([h_msg(h_text(early), h_tool("escalate", "tu_esc", i=1, reason="caller_requested_human",
                                                  summary="Caller asked for a person."))],
                     policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=5000, chars_per_s=30, end_mode="wait"))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, sock = converse(rig, "Can I talk to a person?")
    assert spoken(frames) == early
    assert end_frame(frames) is not None
    estimate = 0.3 + len(early) / 30
    assert_held(end_time(sock) - token_time(sock, early), estimate)
    (hold,) = events(caplog, "handoff_hold")
    assert hold["source"] == "released_text" and hold["tool"] == "escalate"
    assert hold["prefetched"] is False and 0 < hold["wait_ms"] <= estimate * 1000
    (late,) = events(caplog, "ai_text_spoken_before_tool")
    assert late["tool"] == "escalate"
    assert ai_turn_texts(rig) == [early]


def test_handoff_after_an_inline_tool_holds_end_for_the_earlier_hops_text(
        caplog: pytest.LogCaptureFixture) -> None:
    # hop 1 speaks, then calls an inline tool; hop 2 is a hand-off alone. The planner gives it
    # no hand-off phrase (the answer started), so end waits for hop 1's text, counted from its
    # first chunk, not for nothing.
    early = "Let me look at your checking account first."
    backend = FakeBackend({"list_recent_transactions": [{"account": "checking", "transactions": []}]})
    rig = stream_rig([
        h_msg(h_text(early), h_tool("list_recent_transactions", "tu_tx", i=1, account="checking", limit=3)),
        h_msg(h_tool("escalate", "tu_esc", reason="caller_requested_human", summary="Caller asked for a person.")),
    ], backend=backend, policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=5000, chars_per_s=30, end_mode="wait"))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, sock = converse(rig, "Any recent transactions? Actually, get me a person.")
    assert spoken(frames) == early
    assert end_frame(frames) is not None
    estimate = 0.3 + len(early) / 30
    assert_held(end_time(sock) - token_time(sock, early), estimate)
    (hold,) = events(caplog, "handoff_hold")
    assert hold["source"] == "released_text" and hold["tool"] == "escalate"
    assert hold["basis"] == "watch"  # hop 2 released nothing: the session's estimate of hop 1's text
    assert 0 < hold["wait_ms"] <= estimate * 1000
    assert events(caplog, "mask_spoken") == []
    (late,) = events(caplog, "ai_text_spoken_before_tool")
    assert (late["tool"], late["hop"]) == ("list_recent_transactions", 1)


@dataclass
class SlowBackend(FakeBackend):
    """:class:`FakeBackend` whose every call takes ``delay_s``."""

    delay_s: float = 0.0

    async def call(self, op: str, **kw: Any) -> dict[str, Any]:  # type: ignore[override]
        await asyncio.sleep(self.delay_s)
        return await super().call(op, **kw)


def test_handoff_after_a_slow_inline_tool_holds_end_for_the_last_hops_text(
        caplog: pytest.LogCaptureFixture) -> None:
    # hop 1 speaks, an inline tool runs longer than handoff_wait_ms, then hop 2
    # speaks and hands off. The hold is for what is still to be spoken (hop 2's text), at most
    # handoff_wait_ms from the hand-off; timed from hop 1's first chunk it would be 0 and cut hop 2 off.
    early, late = "One moment.", "Connecting you now."
    backend = SlowBackend({"list_recent_transactions": [{"account": "checking", "transactions": []}]},
                          delay_s=1.2)
    rig = stream_rig([
        h_msg(h_text(early), h_tool("list_recent_transactions", "tu_tx", i=1, account="checking", limit=3)),
        h_msg(h_text(late), h_tool("escalate", "tu_esc", i=1, reason="caller_requested_human",
                                   summary="Caller asked for a person.")),
    ], backend=backend, policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=1000, chars_per_s=30, end_mode="wait"))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, sock = converse(rig, "Any recent transactions? Then get me a person.")
    assert spoken(frames) == early + late
    # hop 1's text (0.67 s estimated) has ended by then: the tool alone took longer
    assert token_time(sock, late) - token_time(sock, early) >= backend.delay_s - 0.05
    estimate = 0.3 + len(late) / 30  # under the 1 s cap
    assert_held(end_time(sock) - token_time(sock, late), estimate)
    (hold,) = events(caplog, "handoff_hold")
    assert hold["source"] == "released_text" and hold["tool"] == "escalate"
    assert estimate * 0.8 * 1000 <= hold["wait_ms"] <= 1000


def test_handoff_after_a_retry_holds_end_for_the_apology_and_the_retried_text(
        caplog: pytest.LogCaptureFixture) -> None:
    # the failed call's words, the apology and the retry's text all went out
    # (queued one after the other); the hand-off after the retry holds until all of it is spoken.
    late = "Connecting you now."
    failed = [{"messageStart": {"role": "assistant"}}, *chunks("Let me check ")[:-1],
              RuntimeError("stream broke")]
    rig = stream_rig([failed, h_msg(h_text(late), h_tool("escalate", "tu_esc", i=1,
                                                         reason="caller_requested_human",
                                                         summary="Caller asked for a person."))],
                     policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=5000, chars_per_s=30, end_mode="wait"))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, sock = converse(rig, "Can I talk to a person?")
    said = spoken(frames)
    assert said == "Let me check " + APOLOGY + late
    estimate = 0.3 + len(said) / 30  # all three, queued from the first chunk
    assert_held(end_time(sock) - token_time(sock, "Let me check "), estimate)
    (hold,) = events(caplog, "handoff_hold")
    assert hold["source"] == "released_text" and hold["tool"] == "escalate"
    assert hold["wait_ms"] > (0.3 + len(late) / 30) * 1000  # more than the retried text alone


def test_late_handoff_hold_is_capped_by_handoff_wait_ms(caplog: pytest.LogCaptureFixture) -> None:
    early = "I'll connect you with a colleague now."
    rig = stream_rig([h_msg(h_text(early), h_tool("escalate", "tu_esc", i=1, reason="caller_requested_human",
                                                  summary="Caller asked for a person."))],
                     policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=300, chars_per_s=8))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, sock = converse(rig, "Can I talk to a person?")
    # the 300 ms cap, far under the text's 5 s estimate (8 chars/s)
    assert end_time(sock) - token_time(sock, early) <= 0.3 + 1.0
    (hold,) = events(caplog, "handoff_hold")
    assert hold["source"] == "released_text" and hold["wait_ms"] <= 300


def test_late_handoff_with_end_mode_immediate_does_not_hold(caplog: pytest.LogCaptureFixture) -> None:
    early = "I'll connect you with a colleague now."
    rig = stream_rig([h_msg(h_text(early), h_tool("escalate", "tu_esc", i=1, reason="caller_requested_human",
                                                  summary="Caller asked for a person."))],
                     policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=2500, end_mode="immediate"))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, sock = converse(rig, "Can I talk to a person?")
    assert end_frame(frames) is not None
    assert end_time(sock) - token_time(sock, early) < 1.0  # the text's estimate is 2.8 s (15 chars/s)
    (hold,) = events(caplog, "handoff_hold")
    assert hold["source"] == "released_text" and hold["wait_ms"] == 0


def test_masked_handoff_hold_keeps_source_mask(caplog: pytest.LogCaptureFixture) -> None:
    # Today's hold (a hand-off phrase, no released text) is logged with source=mask.
    rig = stream_rig([h_msg(h_tool("get_balance", "tu_bal", account="checking"))],
                     backend=FakeBackend({"get_balance": [{"account": "checking", "available": "1.00"}]}),
                     policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=200, chars_per_s=30))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "my checking balance")
    assert end_frame(frames) is not None
    (hold,) = events(caplog, "handoff_hold")
    assert hold["source"] == "mask"
    assert hold["basis"] == "phrase"  # no other speech, so today's hold (the phrase)
    assert events(caplog, "ai_text_spoken_before_tool") == []


ESCALATE = {"reason": "caller_requested_human", "summary": "Caller asked for a person."}


def test_handoff_hold_covers_the_hops_text_when_a_speaker_report_clamps_the_watch(
        caplog: pytest.LogCaptureFixture) -> None:
    # an ``agentSpeaking off`` (for earlier audio) arrives just before the hop's
    # last chunk and clamps the watch's estimate to then; the hold still covers the hop's text,
    # counted from its first chunk.
    first, last = "I'll connect you with a colleague ", "now."
    rig = stream_rig([msg(chunks(first, 0.3, last), h_tool("escalate", "tu_esc", i=1, **ESCALATE),
                          stop="tool_use")],
                     policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=5000, chars_per_s=15, end_mode="wait"))

    async def go() -> TimedSocket:
        sim, sock = timed_sim(rig)
        await sim.connect()
        await sim.prompt_nowait("Can I talk to a person?")
        await wait_for(lambda: any(f.get("token") == first for f in sock.sent))
        await speaker(sim, "agentSpeaking", "off")
        await wait_for(lambda: any(f.get("type") == "end" for f in sock.sent), 6.0)
        await sim.hangup()
        return sock

    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        sock = run(go())
    estimate = 0.3 + len(first + last) / 15  # 2.8 s; the clamped watch alone: about 0.3 + 0.6 s
    assert_held(end_time(sock) - token_time(sock, first), estimate)
    (hold,) = events(caplog, "handoff_hold")
    assert (hold["source"], hold["basis"]) == ("released_text", "hop_text")


def test_handoff_hold_covers_the_hops_text_without_a_speech_session(
        caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    # no speech session (or a new one after a reconnect): the watch knows
    # nothing (-inf), and the hold comes from the hop's released text.
    early = "Connecting you now."
    rig = stream_rig([h_msg(h_text(early), h_tool("escalate", "tu_esc", i=1, **ESCALATE))],
                     policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=5000, chars_per_s=15, end_mode="wait"))
    monkeypatch.setattr(rig.bridge, "_speech_end", lambda _sid: float("-inf"))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, sock = converse(rig, "Can I talk to a person?")
    assert spoken(frames) == early
    assert_held(end_time(sock) - token_time(sock, early), 0.3 + len(early) / 15)
    (hold,) = events(caplog, "handoff_hold")
    assert (hold["source"], hold["basis"]) == ("released_text", "hop_text")


def test_masked_handoff_hold_covers_an_acknowledgement_still_playing(
        caplog: pytest.LogCaptureFixture) -> None:
    # the acknowledgement is still playing when the hand-off phrase is sent; the
    # phrase plays after it, and the hold covers both (the watch), not the phrase alone.
    cps = 10
    slow = [{"messageStart": {"role": "assistant"}}, _Pause(lambda: 0.15),
            *h_tool("get_balance", "tu_bal", account="checking"), {"messageStop": {"stopReason": "tool_use"}}]
    rig = stream_rig([slow], backend=FakeBackend({"get_balance": [{"account": "checking", "available": "1.00"}]}),
                     policy_doc=masked_policy_doc(after_ms=100, handoff_wait_ms=5000, chars_per_s=cps, end_mode="wait"))

    async def go() -> tuple[list[dict[str, Any]], TimedSocket]:
        sim, sock = timed_sim(rig)
        await sim.connect()
        frames = await sim.prompt("my checking balance", timeout=8.0)
        await sim.hangup()
        return frames, sock

    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, sock = run(go())
    ack, phrase = [f["token"] for f in frames if f.get("type") == "text" and f.get("token")]
    ack_at, phrase_at = token_time(sock, ack), token_time(sock, phrase)
    ack_end = ack_at + 0.3 + len(ack.rstrip()) / cps
    covers_both = max(ack_end, phrase_at + 0.3) + len(phrase.rstrip()) / cps - phrase_at
    phrase_alone = 0.3 + len(phrase.rstrip()) / cps
    assert covers_both > phrase_alone + 0.5  # the ack has time left when the phrase is sent
    held = end_time(sock) - phrase_at
    assert_held(held, covers_both)
    assert held > (phrase_alone + covers_both) / 2  # nearer both than the phrase alone (the old hold)
    (hold,) = events(caplog, "handoff_hold")
    assert (hold["source"], hold["basis"]) == ("mask", "watch")


def test_step_up_shortcut_speaks_its_phrase_and_holds_end_in_stream_mode(
        caplog: pytest.LogCaptureFixture) -> None:
    # a level-1 make_payment takes the step-up shortcut; the
    # caller hears the tool's hand-off phrase, end waits for it, and nothing is prefetched.
    phrase = "Let me set that up."
    backend = FakeBackend({})
    rig = stream_rig([h_msg(h_tool("make_payment", "tu_pay", from_account="checking", to="purple",
                                   amount="50.00"))],
                     backend=backend,
                     policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=5000, chars_per_s=30, end_mode="wait"))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, sock = converse(rig, "Pay fifty dollars on my card")
    assert spoken(frames) == phrase + " "
    end = end_frame(frames)
    assert end is not None and end["data"]["reason"] == "step-up"
    estimate = 0.3 + len(phrase) / 30
    assert_held(end_time(sock) - token_time(sock, phrase + " "), estimate)
    (hold,) = events(caplog, "handoff_hold")
    assert (hold["source"], hold["tool"], hold["prefetched"]) == ("mask", "make_payment", False)
    assert [s["tool"] for s in events(caplog, "step_up_shortcut")] == ["make_payment"]
    assert backend.calls == []


# --------------------------------------------------------------------------- failure and barge-in


def test_model_failure_mid_stream_speaks_the_apology_before_the_retried_text(
        caplog: pytest.LogCaptureFixture) -> None:
    # text of the failed call was already spoken; the apology comes next, then the retry's text.
    failed = [{"messageStart": {"role": "assistant"}}, *chunks("Let me check ", "<sil")[:-1],
              RuntimeError("stream broke")]
    rig = stream_rig([failed, h_msg(h_text("Your answer is ready."))])
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "What's up?")
    said = spoken(frames)
    assert said == "Let me check " + APOLOGY + "Your answer is ready."  # the held "<sil" is dropped
    assert said.index(APOLOGY) < said.index("Your answer is ready.")
    assert len(rig.agent.calls) == 2
    (turn,) = events(caplog, "turn")
    assert turn["outcome"] == "ok"
    # the failed call's spoken words are journaled too (not the dropped "<sil"), then the retry's
    assert ai_turn_texts(rig) == ["Let me check ", "Your answer is ready."]


def test_barge_in_mid_stream_releases_no_more_text() -> None:
    # the interrupt closes the agent stream; the message's later chunks are never sent.
    rig = stream_rig([msg(chunks("Your balance ", 0.6, "is one hundred dollars.")),
                      h_msg(h_text("Sure, savings."))])

    async def go() -> list[dict[str, Any]]:
        sim, sock = timed_sim(rig)
        await sim.connect()
        await sim.prompt_nowait("checking balance")
        await wait_for(lambda: any(f.get("token") == "Your balance " for f in sock.sent))
        await sim.interrupt("Your balance")
        await asyncio.sleep(0.8)  # past the pause: the second chunk would be out by now
        await sim.idle()
        await sim.prompt("savings please")
        await sim.hangup()
        return list(sock.sent)

    sent = run(go())
    tokens = [f.get("token") for f in sent if f.get("type") == "text"]
    assert "is one hundred dollars." not in tokens
    assert "Sure, savings." in tokens
    assert rig.agent.closed == 2


# --------------------------------------------------------------------------- speech start


def test_agent_speech_started_is_logged_once_per_turn(caplog: pytest.LogCaptureFixture) -> None:
    rig = stream_rig([h_msg(h_text("We open at nine.")), h_msg(h_text("And close at five."))])

    async def go() -> None:
        sim, _ = timed_sim(rig)
        await sim.connect()
        await speaker(sim, "agentSpeaking", "on")  # the greeting: no turn yet, nothing logged
        await speaker(sim, "agentSpeaking", "off")
        await sim.prompt("When do you open?")
        await asyncio.sleep(0.02)
        await speaker(sim, "agentSpeaking", "on")
        await speaker(sim, "agentSpeaking", "off")
        await speaker(sim, "agentSpeaking", "on")  # a second burst of the same turn: not again
        await speaker(sim, "agentSpeaking", "off")
        await sim.prompt("And close?")
        await speaker(sim, "agentSpeaking", "on")
        await sim.hangup()

    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        run(go())
    started = events(caplog, "agent_speech_started")
    assert [s["turn"] for s in started] == [1, 2]
    assert started[0]["ms_after_first_text"] >= 20
    assert all(isinstance(s["ms_after_first_text"], float) and s["ms_after_first_text"] >= 0 for s in started)
    assert "We open at nine." not in json.dumps(started)


def test_agent_speech_started_is_not_logged_past_the_cap(caplog: pytest.LogCaptureFixture,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    # speech starting long after the turn's first chunk is not that chunk's (here a 150 ms cap,
    # AGENT_SPEECH_START_MAX_MS): nothing logged for turn 1, and turn 2 still logs its own.
    monkeypatch.setattr(bridge_mod, "AGENT_SPEECH_START_MAX_MS", 150.0)
    rig = stream_rig([h_msg(h_text("We open at nine.")), h_msg(h_text("And close at five."))])

    async def go() -> None:
        sim, _ = timed_sim(rig)
        await sim.connect()
        await sim.prompt("When do you open?")
        await asyncio.sleep(0.3)
        await speaker(sim, "agentSpeaking", "on")
        await speaker(sim, "agentSpeaking", "off")
        await sim.prompt("And close?")
        await speaker(sim, "agentSpeaking", "on")
        await sim.hangup()

    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        run(go())
    assert [s["turn"] for s in events(caplog, "agent_speech_started")] == [2]


def test_agent_speech_started_is_not_logged_for_a_silence_repeat(caplog: pytest.LogCaptureFixture,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    # the turn's speech was never reported; the silence repeat goes out, and the agentSpeaking
    # on that follows is the repeat's, not the turn's: nothing logged.
    monkeypatch.setattr(silence, "TTS_START_S", 0.02)
    answer = "We open at nine."
    rig = make_rig(script=[say(answer)], policy_doc=silence_policy(silence_max_repeats=2))
    write_call_state(rig.store)

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("When do you open?")
        await wait_for(lambda: [f.get("token") for f in sim.socket.sent if f.get("type") == "text"
                                and f.get("token")] == [answer, answer], 4)
        await speaker(sim, "agentSpeaking", "on")
        await sim.hangup()

    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        run(go())
    assert [e["what"] for e in events(caplog, "prompt_repeat")] == ["agent"]
    assert events(caplog, "agent_speech_started") == []
