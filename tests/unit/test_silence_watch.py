"""``SilenceWatch`` examples; the state-machine properties are in
``tests/property/test_silence_watch_props.py``."""

from __future__ import annotations

import pytest

from tac_bridge.silence import Expiry, SilenceWatch


def test_the_deadline_is_the_estimated_speech_end_plus_the_timeout() -> None:
    w = SilenceWatch(timeout_ms=4000, max_repeats=1, chars_per_s=15)
    assert w.arm("x" * 30, 100.0)
    assert w.deadline == pytest.approx(100.0 + 0.3 + 2.0 + 4.0)


def test_repeat_then_action_once() -> None:
    w = SilenceWatch(timeout_ms=1000, max_repeats=1, action="hangup", chars_per_s=10)
    w.arm("Hello.", 0.0)  # ends at 0.9, expires at 1.9
    assert w.poll(1.8) is None
    assert w.poll(1.9) == Expiry("repeat", 1, text="Hello.")
    assert w.deadline == pytest.approx(1.9 + 0.9 + 1.0)
    assert w.poll(3.7) is None
    assert w.poll(3.8) == Expiry("action", 1, action="hangup")
    assert w.poll(100.0) is None and not w.arm("Hello.", 100.0)  # spent


def test_earlier_speech_still_playing_pushes_the_deadline() -> None:
    w = SilenceWatch(timeout_ms=1000, chars_per_s=10)
    w.note_speech("x" * 50, 0.0)  # plays until 5.3
    w.arm("Hi.", 1.0)
    assert w.deadline == pytest.approx(5.3 + 1.0)


def test_timeout_zero_and_an_unknown_action() -> None:
    w = SilenceWatch(timeout_ms=0, action="explode")
    assert w.action == "escalate"
    assert not w.arm("Hello.", 0.0) and w.poll(1e9) is None
    w.configure(timeout_ms=500, max_repeats=0, action="escalate", chars_per_s=15)
    assert w.arm("Hello.", 0.0)
    w.configure(timeout_ms=0, max_repeats=0, action="escalate", chars_per_s=15)
    assert w.deadline is None and w.poll(1e9) is None


def test_input_with_no_turn_after_it_re_arms_from_the_input() -> None:
    w = SilenceWatch(timeout_ms=1000, max_repeats=1, chars_per_s=10)
    w.arm("Hello.", 0.0)
    assert w.poll(1.9) == Expiry("repeat", 1, text="Hello.")
    w.on_inbound(2.0)  # a key press during the repeat: count reset, re-counted from the input
    assert w.repeats == 0 and w.deadline == pytest.approx(3.0)
    assert w.poll(2.9) is None
    assert w.poll(3.0) == Expiry("repeat", 1, text="Hello.")
    w.on_inbound(3.5)
    w.turn_started()  # a prompt that starts a turn: the turn re-arms at its end
    assert w.deadline is None and w.poll(1e9) is None


def test_input_before_anything_was_said_arms_nothing() -> None:
    w = SilenceWatch(timeout_ms=1000)
    w.on_inbound(5.0)
    assert w.deadline is None


# --------------------------------------------------------------------------- speaker events


def test_the_default_is_two_repeats() -> None:
    assert SilenceWatch().max_repeats == 2  # three chances


def test_a_repeat_counts_from_when_it_was_sent() -> None:
    # (no speaker events): the repeat's estimate counts from the send, not the poll.
    w = SilenceWatch(timeout_ms=1000, max_repeats=2, chars_per_s=10)
    w.arm("Hello.", 0.0)  # ends at 0.9, expires at 1.9
    assert w.poll(1.9) == Expiry("repeat", 1, text="Hello.")
    w.repeat_sent(2.4)  # the send took 0.5 s (the lock, the socket)
    assert w.deadline == pytest.approx(2.4 + 0.9 + 1.0)
    assert w.poll(4.2) is None and w.poll(4.3) == Expiry("repeat", 2, text="Hello.")


def test_agent_speech_end_replaces_the_estimate() -> None:
    w = SilenceWatch(timeout_ms=1000, chars_per_s=10)
    w.on_speaker("agent", True, 0.1)
    w.arm("Hello.", 0.0)
    assert w.deadline is None  # nothing while the agent speaks
    w.on_speaker("agent", False, 3.0)  # spoke far longer than the 0.9 s estimate
    assert w.deadline == pytest.approx(4.0)
    w.on_speaker("agent", True, 3.5)
    assert w.deadline is None and w.poll(100.0) is None


def test_a_message_armed_after_the_agent_already_stopped_counts_from_that_stop() -> None:
    w = SilenceWatch(timeout_ms=1000, chars_per_s=10)
    w.note_speech("Hello.", 0.0)
    w.on_speaker("agent", True, 0.3)
    w.on_speaker("agent", False, 0.7)  # before the 0.9 s estimate
    assert w.arm("Hello.", 0.0)  # the turn ended a little later
    assert w.deadline == pytest.approx(1.7)


def test_caller_speaking_pauses_and_stopping_restarts_with_the_count_unchanged() -> None:
    w = SilenceWatch(timeout_ms=1000, max_repeats=1, chars_per_s=10)
    w.arm("Hello.", 0.0)
    assert w.poll(1.9) == Expiry("repeat", 1, text="Hello.")
    w.on_speaker("caller", True, 3.0)  # after the grace: not a collision
    assert w.deadline is None and w.poll(3.8) is None and w.poll(50.0) is None
    w.on_speaker("caller", False, 60.0)
    assert w.repeats == 1 and w.deadline == pytest.approx(61.0)
    assert w.poll(61.0) == Expiry("action", 1, action="escalate")


def test_the_greeting_timeout_survives_a_pause() -> None:
    w = SilenceWatch(timeout_ms=1000, chars_per_s=10)
    w.arm("Hello.", 0.0, timeout_ms=7000)
    w.on_speaker("caller", True, 2.0)
    w.on_speaker("caller", False, 3.0)
    assert w.deadline == pytest.approx(10.0)


def test_client_speaking_within_the_grace_of_a_repeat_collides() -> None:
    w = SilenceWatch(timeout_ms=1000, max_repeats=1, chars_per_s=10)
    w.arm("Hello.", 0.0)
    assert w.poll(1.9) is not None
    assert w.on_speaker("caller", True, 2.0) is True
    assert w.repeats == 0 and w.collided == 1
    assert w.on_inbound(2.1, "interrupt") is False  # the same speech: not counted twice
    assert w.collided == 1


def test_a_repeat_not_sent_does_not_count() -> None:
    w = SilenceWatch(timeout_ms=1000, max_repeats=1, chars_per_s=10)
    w.arm("Hello.", 0.0)
    assert w.poll(1.9) is not None and w.repeats == 1
    w.unrepeat()
    assert w.repeats == 0 and w.repeat_at is None
    w.unrepeat()
    assert w.repeats == 0


def test_a_prompt_ends_the_caller_speech_and_an_interrupt_the_agent_speech() -> None:
    w = SilenceWatch(timeout_ms=1000, chars_per_s=10)
    w.arm("Hello.", 0.0)
    w.on_speaker("caller", True, 1.0)
    w.on_speaker("agent", True, 1.0)
    w.on_inbound(2.0, "interrupt")
    assert w.agent_speaking is False and w.caller_speaking is True and w.deadline is None
    w.on_inbound(3.0, "prompt")
    assert w.caller_speaking is False and w.deadline == pytest.approx(4.0)
