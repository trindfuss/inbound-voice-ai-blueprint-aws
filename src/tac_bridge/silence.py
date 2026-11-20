"""Caller-silence decisions for one ConversationRelay session: ``SilenceWatch``.

After the AI segment has finished speaking (an agent reply, a masking phrase that ends the turn's
speech, the session's welcome greeting, or a repeat), the caller has ``silence_timeout_ms`` to say
or key something. When the timeout passes:

* with repeats left (fewer than ``silence_max_repeats`` since the caller's last input), the same
  text is repeated and the watch re-arms for the repeat;
* otherwise the configured ``silence_action`` (``escalate`` or ``hangup``) is taken, once: the
  watch never fires again.

**Where the wait counts from.** The watch keeps one point in time, the start of the wait
(``wait_from``); the deadline is ``wait_from`` plus the current arm's timeout.

* *Estimate* (every session until a speaker event arrives, and for any message whose
  speech end hasn't been reported): ``sent_at + 0.3 s + len(text) / chars_per_s``, or
  later if earlier speech is still queued ahead of it (:meth:`SilenceWatch.note_speech`). A
  repeat counts from its own estimated end, re-based on when the bridge actually sent it
  (:meth:`SilenceWatch.repeat_sent`).
* *Speaker events* (ConversationRelay ``events="speaker-events"``,
  :meth:`SilenceWatch.on_speaker`): ``agentSpeaking on`` holds the wait (no deadline) and
  ``agentSpeaking off`` starts it at that moment, replacing the estimate for that message; a
  message armed after the agent's speech already ended counts from that end. ``clientSpeaking on``
  pauses the wait (no deadline, nothing fires) and ``clientSpeaking off`` restarts it from that
  moment with the current arm's timeout, the repeat count unchanged. :attr:`SilenceWatch.speaker_events`
  records that the session had one.

Any caller input (:meth:`SilenceWatch.on_inbound`: a ``prompt``, ``dtmf`` or ``interrupt`` frame)
resets the repeat count and cancels the pending expiry. Input that starts no turn (a key press, a
barge-in that no prompt follows) must not leave the watch off for good, so the watch re-arms from
the input itself: ``inbound time + silence_timeout_ms``, repeating the last message the caller
heard. A ``prompt`` also ends the caller's speech (the utterance is over); an ``interrupt`` ends
the agent's (a barge-in cuts the audio). A turn (:meth:`SilenceWatch.turn_started`) disarms, and
the turn re-arms the watch at its end. The watch never fires while a turn runs, while speech is
estimated to be playing, or while either side is reported speaking. ``silence_timeout_ms == 0``
turns the watch off (it never arms).

One arm may use its own timeout: :meth:`SilenceWatch.arm` takes ``timeout_ms``, which
the bridge passes as ``greeting_silence_timeout_ms`` for the opening greeting of the call's first
AI session. It applies to that arm only (:attr:`SilenceWatch.arm_timeout_ms`), including a
restart after the caller stops speaking; a repeat and the re-arm after caller input count
``silence_timeout_ms`` again. An override of 0 doesn't arm, but the text is still the last
message (input with no turn after it still repeats it). The override never turns a watch with
``silence_timeout_ms == 0`` on.

A repeat that meets the caller's speech doesn't count: an ``interrupt`` or a
``clientSpeaking on`` within :data:`REPEAT_GRACE_MS` of a repeat being sent means the caller was
already speaking, so that repeat is taken back out of the count (:attr:`SilenceWatch.collided`
counts these) and :meth:`SilenceWatch.on_inbound` / :meth:`SilenceWatch.on_speaker` report it,
for the bridge's ``prompt_repeat_collided`` line. After an ``interrupt`` the count resets and the
watch re-arms from the input, as for any input; after ``clientSpeaking on`` the watch pauses.

This module only decides; it sends nothing, reads no policy and does no I/O, so it is
property-tested directly. Time is passed in (``sent_at``, ``now``), on the ``time.monotonic()``
scale the bridge uses. The bridge schedules a wake-up at :attr:`SilenceWatch.wake_at` and calls
:meth:`SilenceWatch.poll` then; :attr:`SilenceWatch.epoch` changes on every event, so the bridge
can tell whether anything happened while it was awaiting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tac_bridge.masking import TTS_START_S

#: ``silence_action`` values; anything else is treated as ``escalate``.
SILENCE_ACTIONS: frozenset[str] = frozenset({"escalate", "hangup"})
DEFAULT_ACTION = "escalate"
#: An ``interrupt`` (or ``clientSpeaking on``) this soon after a repeat was sent means the
#: caller was already speaking; the repeat doesn't count toward ``silence_max_repeats``.
REPEAT_GRACE_MS = 500
#: Who a speaker event is about.
CALLER = "caller"
AGENT = "agent"


@dataclass(frozen=True, slots=True)
class Expiry:
    """What :meth:`SilenceWatch.poll` decided: repeat ``text`` (``kind == "repeat"``, ``repeat``
    is its number since the caller's last input) or take ``action`` (``kind == "action"``)."""

    kind: Literal["repeat", "action"]
    repeat: int
    text: str | None = None
    action: str | None = None


class SilenceWatch:
    """Per-session silence state (see the module docstring)."""

    def __init__(self, *, timeout_ms: int = 4000, max_repeats: int = 2, action: str = DEFAULT_ACTION,
                 chars_per_s: float = 15.0) -> None:
        self.timeout_ms = 0
        self.max_repeats = 0
        self.action = DEFAULT_ACTION
        self.chars_per_s = 15.0
        #: The watch is counting a wait (possibly paused while someone speaks).
        self.counting = False
        #: When the current wait started (meaningful while :attr:`counting`).
        self.wait_from = 0.0
        #: When the watch expires (``None``: disarmed, or paused while someone speaks).
        self.deadline: float | None = None
        #: The action was taken (the watch never fires again).
        self.done = False
        #: Turns running now (the watch never fires while one runs).
        self.turns = 0
        #: A speaker event arrived in this session; who is speaking now; when the agent's
        #: speech last ended (``agentSpeaking off`` or a barge-in), and when speech was last sent.
        self.speaker_events = False
        self.caller_speaking = False
        self.agent_speaking = False
        self.agent_end: float | None = None
        self.last_sent = float("-inf")
        #: When the caller last stopped speaking (``clientSpeaking off``).
        self.caller_end = float("-inf")
        #: The timeout of the current arm in ms: ``timeout_ms``, or an arm's override.
        self.arm_timeout_ms = 0
        self.configure(timeout_ms=timeout_ms, max_repeats=max_repeats, action=action,
                       chars_per_s=chars_per_s)
        self.arm_timeout_ms = self.timeout_ms
        #: The message a repeat speaks (the last one armed).
        self.text: str | None = None
        #: Repeats spoken since the caller's last input.
        self.repeats = 0
        #: When the speech sent so far is estimated to end.
        self.playing_until = float("-inf")
        #: Bumped on every event.
        self.epoch = 0
        #: Bumped on caller input and turn starts only (not on speech or speaker events), so the
        #: bridge's late greeting arm can tell whether the caller did anything since ``setup``.
        self.input_epoch = 0
        #: When the last repeat was sent, while it is the last thing the caller heard and no input
        #: has followed it (``None`` otherwise): an ``interrupt`` within the grace collides with it.
        self.repeat_at: float | None = None
        #: The grace in ms.
        self.repeat_grace_ms = REPEAT_GRACE_MS
        #: Repeats that collided with the caller's speech, for the whole session.
        self.collided = 0

    # ------------------------------------------------------------------ settings

    def configure(self, *, timeout_ms: int, max_repeats: int, action: str, chars_per_s: float) -> None:
        """The policy's current values (read per turn by the bridge). A timeout of 0 disarms."""
        self.timeout_ms = max(0, int(timeout_ms))
        self.max_repeats = max(0, int(max_repeats))
        self.action = action if action in SILENCE_ACTIONS else DEFAULT_ACTION
        self.chars_per_s = float(chars_per_s) if chars_per_s and chars_per_s > 0 else 15.0
        if self.timeout_ms == 0:
            self.counting = False
        self._update()

    @property
    def timeout_s(self) -> float:
        return self.timeout_ms / 1000.0

    def speech_end(self, text: str, sent_at: float) -> float:
        """Estimated end of ``text`` sent at ``sent_at``, if nothing else were playing."""
        return sent_at + TTS_START_S + len(text) / self.chars_per_s

    @property
    def paused(self) -> bool:
        """Someone is reported speaking, so nothing fires."""
        return self.caller_speaking or self.agent_speaking

    @property
    def wake_at(self) -> float | None:
        """When the bridge should poll next: the deadline, but never while speech is estimated
        to be playing (``None``: nothing to wait for)."""
        if self.deadline is None:
            return None
        return max(self.deadline, self.playing_until)

    def _update(self) -> None:
        """The deadline from the wait's start and the arm's timeout; ``None`` when not counting,
        paused, spent or off."""
        if (not self.counting or self.done or self.timeout_ms == 0 or self.turns > 0
                or self.arm_timeout_ms == 0 or self.paused):
            self.deadline = None
        else:
            self.deadline = self.wait_from + self.arm_timeout_ms / 1000.0

    def _speech_ended_since(self, sent_at: float) -> bool:
        """The agent's speech was reported over, at or after ``sent_at`` and after the last
        speech sent: the wait counts from that report, not the estimate."""
        return (not self.agent_speaking and self.agent_end is not None
                and self.agent_end >= sent_at and self.agent_end >= self.last_sent)

    # ------------------------------------------------------------------ events

    def note_speech(self, text: str, sent_at: float) -> None:
        """``text`` went to ConversationRelay at ``sent_at``. It plays after anything still queued;
        a counting wait moves so the timeout counts from the end of all speech."""
        if not text:
            return
        self.epoch += 1
        self.last_sent = max(self.last_sent, sent_at)
        start = max(self.playing_until, sent_at + TTS_START_S)
        self.playing_until = start + len(text) / self.chars_per_s
        if self.counting:
            self.wait_from = max(self.wait_from, self.playing_until)
        self._update()

    def arm(self, text: str | None, sent_at: float, *, timeout_ms: int | None = None) -> bool:
        """``text`` (sent at ``sent_at``) ends the AI's speech for now: count the caller's silence
        from its end: reported if the agent's speech already ended after it was sent,
        else estimated (or the end of anything still playing). No text, a timeout of 0, or an
        action already taken: not armed (``False``). The repeat count is kept (only caller input
        resets it). While someone is speaking the wait is armed but paused.

        ``timeout_ms`` replaces ``silence_timeout_ms`` for this arm only (the opening
        greeting); 0 doesn't arm but keeps ``text`` as the last message. It never turns on a
        watch whose ``silence_timeout_ms`` is 0."""
        self.epoch += 1
        self.repeat_at = None
        if self.timeout_ms == 0 or self.done or not text or not text.strip():
            self.counting = False
            self._update()
            return False
        self.text = text
        wait_ms = self.timeout_ms if timeout_ms is None else max(0, int(timeout_ms))
        self.arm_timeout_ms = wait_ms
        if wait_ms == 0:
            self.counting = False
            self._update()
            return False
        self.counting = True
        if self._speech_ended_since(sent_at):
            assert self.agent_end is not None  # noqa: S101 - implied by the check
            self.wait_from = self.agent_end
        else:
            self.wait_from = max(self.speech_end(text, sent_at), self.playing_until)
        self.wait_from = max(self.wait_from, self.caller_end)  # never before the caller stopped
        self._update()
        return True

    def disarm(self) -> None:
        self.epoch += 1
        self.counting = False
        self._update()

    def _within_grace(self, now: float) -> bool:
        return (self.repeat_at is not None and self.repeats > 0
                and 0.0 <= now - self.repeat_at <= self.repeat_grace_ms / 1000.0)

    def collides(self, kind: str, now: float) -> bool:
        """Whether caller input ``kind`` at ``now`` is an ``interrupt`` within the grace of the
        last repeat. Pure: nothing changes."""
        return kind == "interrupt" and self._within_grace(now)

    def repeat_sent(self, at: float) -> None:
        """The bridge sent the repeat :meth:`poll` returned at ``at`` (a little after the poll).
        The grace and, unless the agent's speech end is already reported, the estimate count
        from the send. Ignored once input has followed the repeat."""
        if self.repeat_at is None:
            return
        self.repeat_at = max(self.repeat_at, at)
        if self.counting and self.text and not self._speech_ended_since(at):
            self.last_sent = max(self.last_sent, at)
            end = self.speech_end(self.text, at)
            self.playing_until = max(self.playing_until, end)
            self.wait_from = max(self.wait_from, end)
            self._update()

    def unrepeat(self) -> None:
        """The bridge didn't send the repeat :meth:`poll` returned (something happened while it
        waited): it doesn't count. Ignored once input or a collision has taken it out already."""
        if self.repeat_at is not None and self.repeats > 0:
            self.repeats -= 1
        self.repeat_at = None

    def on_inbound(self, now: float, kind: str = "prompt") -> bool:
        """Caller input at ``now`` (speech, keypad, or a barge-in): the count resets and the pending
        expiry is cancelled. Input stops the playback estimate (a barge-in cuts the audio), and
        unless a turn starts first the watch expires at ``now + silence_timeout_ms``, repeating
        the last message the caller heard (none known, timeout 0, or the action taken: disarmed).
        A ``prompt`` ends the caller's speech; an ``interrupt`` ends the agent's.

        Returns ``True`` when the input is an ``interrupt`` within the grace of the last repeat
        That repeat doesn't count (it is in :attr:`collided`, not in the repeats the
        reset clears)."""
        collided = self.collides(kind, now)
        if collided:
            self.collided += 1
        self.epoch += 1
        self.input_epoch += 1
        self.repeat_at = None
        self.repeats = 0
        self.playing_until = min(self.playing_until, now)
        if kind == "prompt":
            self.caller_speaking = False
        if kind == "interrupt" and self.agent_speaking:
            self.agent_speaking = False
            self.agent_end = now
        self.arm_timeout_ms = self.timeout_ms
        self.counting = bool(self.timeout_ms > 0 and not self.done and self.text and self.text.strip())
        self.wait_from = now
        self._update()
        return collided

    def on_speaker(self, who: str, speaking: bool, now: float) -> bool:
        """ConversationRelay reported that ``who`` (:data:`CALLER` or :data:`AGENT`)
        started (``speaking``) or stopped speaking at ``now``.

        * caller on: the wait pauses (no deadline); the repeat count is unchanged, except that a
          repeat sent within the grace doesn't count (returns ``True``);
        * caller off: a counting wait restarts from ``now`` with the current arm's timeout;
        * agent on: the wait is held until the agent stops;
        * agent off: the agent's speech ended, so a counting wait starts from ``now``.

        Not input: the count isn't reset (a ``prompt`` or ``dtmf`` that follows does that)."""
        self.epoch += 1
        self.speaker_events = True
        collided = False
        if who == CALLER:
            if speaking:
                if self._within_grace(now):
                    collided = True
                    self.collided += 1
                    self.repeats -= 1
                self.repeat_at = None
                self.caller_speaking = True
            else:
                self.caller_speaking = False
                self.caller_end = now
                if self.counting:
                    self.wait_from = max(now, self.playing_until)
        elif who == AGENT:
            if speaking:
                self.agent_speaking = True
            else:
                self.agent_speaking = False
                self.agent_end = now
                self.playing_until = min(self.playing_until, now)
                if self.counting:
                    self.wait_from = now
        self._update()
        return collided

    def turn_started(self) -> None:
        self.input_epoch += 1
        self.turns += 1
        self.repeat_at = None
        self.disarm()

    def turn_ended(self) -> None:
        self.epoch += 1
        self.turns = max(0, self.turns - 1)
        self._update()

    # ------------------------------------------------------------------ expiry

    @property
    def armed(self) -> bool:
        return self.deadline is not None

    def poll(self, now: float) -> Expiry | None:
        """The deadline passed (``now``): a repeat (the watch re-arms from the repeat's estimated
        end, as if it was sent at ``now``) or the action (once). ``None`` when nothing is due: not
        armed, a turn running, someone speaking, speech still playing, or too early. Every
        condition is checked here again, so a wake-up already scheduled can't fire after a pause."""
        if (self.deadline is None or self.done or self.turns > 0 or self.timeout_ms == 0
                or self.paused or not self.counting
                or now < self.deadline or now < self.playing_until):
            return None
        if self.repeats < self.max_repeats and self.text:
            self.repeats += 1
            text = self.text
            self.note_speech(text, now)
            self.arm(text, now)
            self.repeat_at = now
            return Expiry("repeat", self.repeats, text=text)
        self.epoch += 1
        self.done = True
        self.counting = False
        self._update()
        return Expiry("action", self.repeats, action=self.action)

    def retract(self) -> None:
        """The bridge could not take the action :meth:`poll` returned (the caller spoke, or the
        session ended, while it waited to send): the watch is no longer spent."""
        self.done = False
