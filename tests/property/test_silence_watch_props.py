"""``SilenceWatch`` as a state machine over a fake clock.

Events: arm (with text, sent now or a little earlier), the greeting's arm with its own timeout,
caller input (``prompt``, ``dtmf``, ``interrupt``), speech going out (a phrase playing), the
bridge's repeat-sent and repeat-not-sent reports, turn start and end, ConversationRelay speaker
events (caller and agent, on and off, in any order) and clock ticks (each followed by a poll).
Properties:

* (a) nothing fires while the caller is reported speaking (``clientSpeaking on`` with no
  ``off`` or ``prompt`` since), nor while the agent is;
* (b) a pause and resume (caller on, caller off) never changes the repeat count, except that a
  ``clientSpeaking on`` within ``REPEAT_GRACE_MS`` of a repeat takes that repeat out (exactly one, reported as a collision); agent events never change it;
* (c) with speaker events, every expiry and every deadline is at least the agent's last
  reported speech end plus the applicable timeout, and the caller's last reported speech end
  plus it;
* (d) the action happens at most once, only after ``silence_max_repeats`` repeats; repeats since
  the caller's last input never exceed ``silence_max_repeats``;
* without speaker events (the estimate): the watch never fires before the timeout
  counted from the estimated end of all speech, a repeat's estimate counting from when the
  bridge sent it;
* nothing fires while a turn runs or while speech is estimated to be playing; any caller input
  resets the count; ``silence_timeout_ms == 0`` never fires;
* An ``interrupt`` within the grace of a repeat collides with it (reported, counted in
  ``collided``, the count is 0 after it as after any input); other input never collides;
* An arm with its own timeout never fires before it; 0 doesn't arm but keeps the text;
  the repeat after it and every later input count ``silence_timeout_ms`` again.
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

from tac_bridge.masking import TTS_START_S
from tac_bridge.silence import AGENT, CALLER, REPEAT_GRACE_MS, SilenceWatch

EPS = 1e-9
GRACE_S = REPEAT_GRACE_MS / 1000.0 * 0.98  # inside the grace, clear of float rounding at its edge
NEG = float("-inf")
texts = st.text(alphabet=st.sampled_from("abc .,?"), max_size=40)


class SilenceMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.now = 1000.0
        self.watch = SilenceWatch()

    @initialize(timeout_ms=st.one_of(st.just(0), st.integers(50, 5000)), max_repeats=st.integers(0, 3),
                action=st.sampled_from(["escalate", "hangup", "bogus"]),
                cps=st.floats(8, 30, allow_nan=False))
    def setup(self, timeout_ms: int, max_repeats: int, action: str, cps: float) -> None:
        self.watch.configure(timeout_ms=timeout_ms, max_repeats=max_repeats, action=action, chars_per_s=cps)
        self.timeout_s = timeout_ms / 1000.0
        self.max_repeats = max_repeats
        self.action = action if action in ("escalate", "hangup") else "escalate"
        self.cps = cps
        # the model
        self.armed = False  # counting a wait (possibly paused)
        self.text: str | None = None
        self.repeats = 0
        self.actions = 0
        self.turns = 0
        self.playing_until = NEG
        self.last_sent = NEG
        self.earliest = NEG  # estimate mode: the soonest the armed watch may fire
        self.repeat_at: float | None = None  # the last repeat, while no input or arm followed it
        self.collided = 0
        self.arm_timeout_s = self.timeout_s  # the current arm's timeout
        self.speaker_seen = False
        self.caller_speaking = False
        self.agent_speaking = False
        self.agent_end: float | None = None
        self.caller_end = NEG

    # ------------------------------------------------------------------ model helpers

    def _estimate(self, text: str, at: float) -> float:
        return at + TTS_START_S + len(text) / self.cps

    def _speak(self, text: str, at: float) -> None:
        if text:
            self.last_sent = max(self.last_sent, at)
            start = max(self.playing_until, at + TTS_START_S)
            self.playing_until = start + len(text) / self.cps

    def _ended_since(self, at: float) -> bool:
        return (not self.agent_speaking and self.agent_end is not None
                and self.agent_end >= at and self.agent_end >= self.last_sent)

    def _within_grace(self) -> bool:
        return (self.repeat_at is not None and self.repeats > 0
                and 0.0 <= self.now - self.repeat_at <= REPEAT_GRACE_MS / 1000.0)

    def _arm_model(self, text: str, sent_at: float, wait_s: float) -> None:
        self.text = text
        self.arm_timeout_s = wait_s
        if not self.speaker_seen:
            base = max(self._estimate(text, sent_at), self.playing_until)
            self.earliest = base + wait_s

    # ------------------------------------------------------------------ rules: speech and arming

    @rule(text=texts, back=st.floats(0, 2, allow_nan=False))
    def arm(self, text: str, back: float) -> None:
        sent_at = self.now - back
        ok = self.watch.arm(text, sent_at)
        self.repeat_at = None
        expect = self.timeout_s > 0 and self.actions == 0 and bool(text.strip())
        assert ok == expect
        self.armed = ok
        if ok:
            self._arm_model(text, sent_at, self.timeout_s)

    @rule(text=texts, back=st.floats(0, 2, allow_nan=False),
          greeting_ms=st.one_of(st.just(0), st.integers(50, 30000)))
    def arm_greeting(self, text: str, back: float, greeting_ms: int) -> None:
        # The first AI session's opening greeting waits its own timeout.
        sent_at = self.now - back
        ok = self.watch.arm(text, sent_at, timeout_ms=greeting_ms)
        self.repeat_at = None
        usable = self.timeout_s > 0 and self.actions == 0 and bool(text.strip())
        assert ok == (usable and greeting_ms > 0)
        self.armed = ok
        if usable:  # kept as the last message even when the greeting arm is off
            self._arm_model(text, sent_at, greeting_ms / 1000.0)

    @rule(text=texts.filter(bool))
    def phrase_playing(self, text: str) -> None:
        self.watch.note_speech(text, self.now)
        self._speak(text, self.now)
        if self.armed and not self.speaker_seen:
            self.earliest = max(self.earliest, self.playing_until + self.arm_timeout_s)

    @precondition(lambda self: getattr(self, "repeat_at", None) is not None)
    @rule()
    def repeat_sent(self) -> None:
        # The bridge sent the repeat (a little after the poll): its estimate counts from the send.
        assert self.repeat_at is not None
        at = self.now
        self.watch.repeat_sent(at)
        self.repeat_at = max(self.repeat_at, at)
        if self.armed and self.text and not self._ended_since(at):
            end = self._estimate(self.text, at)
            self.last_sent = max(self.last_sent, at)
            self.playing_until = max(self.playing_until, end)
            if not self.speaker_seen:
                self.earliest = max(self.earliest, end + self.arm_timeout_s)

    @rule()
    def repeat_not_sent(self) -> None:
        before = self.watch.repeats
        self.watch.unrepeat()
        if self.repeat_at is not None and self.repeats > 0:
            self.repeats -= 1
        self.repeat_at = None
        assert self.watch.repeats in (before, before - 1)

    # ------------------------------------------------------------------ rules: caller input

    def _input(self, kind: str) -> None:
        at = self.now
        collides = kind == "interrupt" and self._within_grace()
        assert self.watch.on_inbound(at, kind) == collides
        if collides:
            self.collided += 1
        self.repeat_at = None
        self.repeats = 0
        self.playing_until = min(self.playing_until, at)  # input cuts the playback estimate
        if kind == "prompt":
            self.caller_speaking = False
        if kind == "interrupt" and self.agent_speaking:
            self.agent_speaking = False
            self.agent_end = at
        self.armed = self.timeout_s > 0 and self.actions == 0 and bool(self.text and self.text.strip())
        self.arm_timeout_s = self.timeout_s
        if not self.speaker_seen:
            self.earliest = at + self.timeout_s

    @rule(kind=st.sampled_from(["prompt", "dtmf", "interrupt"]))
    def inbound(self, kind: str) -> None:
        self._input(kind)

    @precondition(lambda self: getattr(self, "repeat_at", None) is not None
                  and self.now - self.repeat_at <= GRACE_S)
    @rule(offset=st.floats(0, GRACE_S, allow_nan=False))
    def interrupt_within_grace_after_repeat(self, offset: float) -> None:
        # The caller was already speaking when the repeat went out.
        assert self.repeat_at is not None
        self.now = max(self.now, self.repeat_at + offset)
        before = self.collided
        self._input("interrupt")
        if self.collided == before:  # a repeat already taken out (e.g. by clientSpeaking)
            return
        assert self.watch.repeats == 0, "the collided repeat doesn't count"

    # ------------------------------------------------------------------ rules: speaker events

    def _caller(self, speaking: bool) -> None:
        before = self.watch.repeats
        collides = speaking and self._within_grace()
        assert self.watch.on_speaker(CALLER, speaking, self.now) == collides
        self.speaker_seen = True
        if speaking:
            if collides:
                self.collided += 1
                self.repeats -= 1
            self.repeat_at = None
            self.caller_speaking = True
        else:
            self.caller_speaking = False
            self.caller_end = self.now
        # (b) a pause or resume never changes the count, except a collided repeat
        assert self.watch.repeats == before - (1 if collides else 0), "pause/resume changed the count"

    @rule()
    def caller_on(self) -> None:
        self._caller(True)

    @rule()
    def caller_off(self) -> None:
        self._caller(False)

    @precondition(lambda self: getattr(self, "repeat_at", None) is not None
                  and self.now - self.repeat_at <= GRACE_S)
    @rule(offset=st.floats(0, GRACE_S, allow_nan=False))
    def caller_on_within_grace_after_repeat(self, offset: float) -> None:
        assert self.repeat_at is not None
        self.now = max(self.now, self.repeat_at + offset)
        before = self.collided
        self._caller(True)
        assert self.collided == before + 1, "clientSpeaking on within the grace after a repeat collides"

    def _agent(self, speaking: bool) -> None:
        before = self.watch.repeats
        assert self.watch.on_speaker(AGENT, speaking, self.now) is False
        self.speaker_seen = True
        if speaking:
            self.agent_speaking = True
        else:
            self.agent_speaking = False
            self.agent_end = self.now
            self.playing_until = min(self.playing_until, self.now)
        assert self.watch.repeats == before, "an agent event changed the count"

    @rule()
    def agent_on(self) -> None:
        self._agent(True)

    @rule()
    def agent_off(self) -> None:
        self._agent(False)

    # ------------------------------------------------------------------ rules: turns and time

    @rule()
    def turn_start(self) -> None:
        self.watch.turn_started()
        self.turns += 1
        self.armed = False
        self.repeat_at = None

    @precondition(lambda self: self.turns > 0)
    @rule()
    def turn_end(self) -> None:
        self.watch.turn_ended()
        self.turns -= 1

    @rule(dt=st.floats(0, 8, allow_nan=False))
    def tick(self, dt: float) -> None:
        self.now += dt
        exp = self.watch.poll(self.now)
        if exp is None:
            return
        assert self.timeout_s > 0, "timeout 0 never fires"
        assert self.armed, "fired while disarmed (a turn, no message known, or never armed)"
        assert self.turns == 0, "fired during a turn"
        assert not self.caller_speaking, "(a) fired while the caller is speaking"
        assert not self.agent_speaking, "fired while the agent is speaking"
        assert self.now + EPS >= self.playing_until, "fired while speech is estimated to be playing"
        self._check_bounds(self.now, "fired")
        if exp.kind == "repeat":
            self.repeats += 1
            assert exp.repeat == self.repeats <= self.max_repeats, "(d) too many repeats"
            assert exp.text == self.text
            self._speak(self.text or "", self.now)
            self.repeat_at = self.now
            self._arm_model(self.text or "", self.now, self.timeout_s)  # silence_timeout_ms again
        else:
            self.actions += 1
            assert self.actions == 1, "(d) the action is taken at most once"
            assert self.repeats == self.max_repeats, "(d) the action only after the last repeat"
            assert exp.action == self.action
            self.armed = False

    def _check_bounds(self, t: float, what: str) -> None:
        if self.speaker_seen:  # (c)
            if self.agent_end is not None:
                assert t + EPS >= self.agent_end + self.arm_timeout_s, \
                    f"(c) {what} before the agent's speech end + timeout"
            assert t + EPS >= self.caller_end + self.arm_timeout_s, \
                f"(c) {what} before the caller's speech end + timeout"
        else:
            assert t + EPS >= self.earliest, f"{what} before the timeout counted from the estimated end"

    @invariant()
    def counts(self) -> None:
        if not hasattr(self, "max_repeats"):
            return
        assert self.watch.repeats == self.repeats <= self.max_repeats
        assert self.watch.done == (self.actions == 1)
        assert self.watch.collided == self.collided
        assert self.watch.repeat_at == self.repeat_at
        assert self.watch.text == self.text
        assert self.watch.caller_speaking == self.caller_speaking
        assert self.watch.agent_speaking == self.agent_speaking
        assert self.watch.speaker_events == self.speaker_seen
        if self.armed:
            assert abs(self.watch.arm_timeout_ms / 1000.0 - self.arm_timeout_s) < EPS
        quiet = not self.caller_speaking and not self.agent_speaking
        if self.timeout_s == 0 or not self.armed or self.turns > 0 or not quiet:
            assert self.watch.deadline is None, "a deadline while paused, disarmed or in a turn"
        else:
            assert self.watch.deadline is not None
            self._check_bounds(self.watch.deadline, "deadline")


SilenceMachine.TestCase.settings = settings(max_examples=400, stateful_step_count=50, deadline=None,
                                            suppress_health_check=[HealthCheck.too_slow])
TestSilenceWatch = SilenceMachine.TestCase
