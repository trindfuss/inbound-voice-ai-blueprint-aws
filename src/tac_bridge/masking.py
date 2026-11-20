"""Latency-masking decisions for one bridge turn: ``MaskPlanner``.

Correctness properties.

The bridge hides model and tool latency with scripted catalog phrases, never model text. Two kinds exist:

* a **per-tool phrase** (the tool's pool, ``mask.<tool>`` and ``mask.<tool>.<n>``), for a tool
  that is masked in the policy, and
* a **timed acknowledgement** (``mask.generic.<n>``, a rotated pool picked by the bridge), spoken
  if the turn has said nothing and called no tool after ``after_ms``
  (``after_ms == 0`` turns it off). A per-tool phrase can follow it in the same turn.

**A tool phrase is scheduled, not sent**. :meth:`MaskPlanner.on_tool_calls`
returns the phrase to schedule, with its delay and absolute deadline:

* the delay is the tool's own ``after_ms`` (``source == "tool"``), else the policy-wide
  ``tool_after_ms`` (``"default"``); a tool handed to the front door has delay 0 and is spoken at
  once (``"handoff"``: there is no answer to wait for in this session);
* the deadline is ``tool call time + delay``, or, if the turn's acknowledgement was spoken, its
  estimated end (``ack sent + 0.3 s + len(ack) / chars_per_s``) when that is later;
  a hand-off phrase is never anchored (spoken at once);
* when the deadline passes the bridge asks :meth:`MaskPlanner.on_deadline`, which speaks the
  phrase only if the turn has sent no answer text; answer text (:meth:`MaskPlanner.on_text`)
  cancels a pending phrase.

This module only decides; it does no I/O, reads no policy and sends nothing, so it is
property-tested directly. Time is passed in (``now``), never read here. The bridge creates one
planner per turn and asks it at each event:

* :meth:`MaskPlanner.timer_s` when the hop starts, to arm the acknowledgement,
* :meth:`MaskPlanner.on_timer` when it fires, then :meth:`MaskPlanner.note_ack` once it is sent,
* :meth:`MaskPlanner.on_tool_calls` for the tool calls of one model message (or a Gateway tool
  start),
* :meth:`MaskPlanner.on_deadline` when a scheduled phrase's deadline passes,
* :meth:`MaskPlanner.on_text` whenever the turn sends answer text (not a masking phrase).

The hand-off hold is computed in the bridge (``_handoff_hold_s``) from :meth:`MaskPlanner.handoff_wait_ms_for`.

Rules: a turn speaks at most one acknowledgement and at most one tool
phrase; neither after answer text; the acknowledgement only before any tool call; with several
tool calls in one message only the first masked tool in call order is scheduled; while one tool
phrase is pending no other is scheduled, except that a hand-off replaces a pending inline phrase
(the session is about to end, so the inline phrase could never play).

Hand-off wait: with ``end_mode == "immediate"`` the bridge sends
``end`` straight away (0 s). With ``"wait"`` it holds the hand-off until the phrase's estimated
end, less the time already elapsed, clamped to ``[0, handoff_wait_ms / 1000]``, where
``handoff_wait_ms`` is the tool's own value when set, else the policy-wide one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from owlbank.masking_config import DEFAULT_TOOL_AFTER_MS, ToolMask

#: Estimated delay from sending text to ConversationRelay until TTS audio starts.
TTS_START_S = 0.3

EndMode = Literal["wait", "immediate"]
#: Where a tool phrase's delay came from (the ``mask_spoken`` / ``mask_skipped`` ``source``).
DelaySource = Literal["tool", "default", "handoff"]

#: ``mask_skipped`` reasons.
SKIP_ANSWER_STARTED = "answer_started"
SKIP_TOOL_OFF = "tool_off"
SKIP_NO_RENDERABLE = "no_renderable_phrase"
SKIP_ONE_PER_TURN = "turn_phrase_used"
SKIP_SUPERSEDED = "superseded_by_handoff"
SKIP_TURN_ENDED = "turn_ended"


def speech_end(text: str, sent_at: float, chars_per_s: float) -> float:
    """Estimated time ``text`` sent at ``sent_at`` finishes playing."""
    return speech_end_chars(len(text), sent_at, chars_per_s)


def speech_end_chars(chars: int, sent_at: float, chars_per_s: float) -> float:
    """:func:`speech_end` for ``chars`` characters of text."""
    rate = chars_per_s if chars_per_s > 0 else 1.0
    return sent_at + TTS_START_S + chars / rate


@dataclass(frozen=True, slots=True)
class ScheduledPhrase:
    """A tool phrase to speak at ``deadline`` (``time.monotonic()`` scale) unless answer text
    comes first. ``phrase_key`` is the tool's pool base key (the policy's ``phrase_key``)."""

    tool: str
    phrase_key: str
    after_ms: int
    source: DelaySource
    called_at: float
    deadline: float

    @property
    def delay_s(self) -> float:
        return self.after_ms / 1000.0


@dataclass
class MaskPlanner:
    """Per-turn masking decisions (pure; see the module docstring).

    ``tool_mask`` maps a tool name to its :class:`~owlbank.masking_config.ToolMask` or ``None``
    (unmasked). For callers that only know phrase keys, ``phrase_for`` (tool → key or ``None``)
    is used instead, with no per-tool delays. ``is_handoff`` tells hand-off tools (delay 0).
    ``generic_key`` is this turn's acknowledgement from the rotated pool.
    """

    after_ms: int
    handoff_wait_ms: int
    end_mode: EndMode
    chars_per_s: float
    phrase_for: Callable[[str], str | None] | None = None
    generic_key: str = "mask.generic.1"
    tool_after_ms: int = DEFAULT_TOOL_AFTER_MS
    tool_mask: Callable[[str], ToolMask | None] | None = None
    is_handoff: Callable[[str], bool] = field(default=lambda _name: False)
    spoken_timer: bool = field(default=False, init=False)
    spoken_tool: bool = field(default=False, init=False)
    spoken_text: bool = field(default=False, init=False)
    saw_tool_call: bool = field(default=False, init=False)
    #: When the acknowledgement spoken this turn is estimated to end (``None``: none spoken).
    ack_end: float | None = field(default=None, init=False)
    pending: ScheduledPhrase | None = field(default=None, init=False)
    #: ``(tool, reason)`` for each tool of the last :meth:`on_tool_calls` that got no phrase.
    last_skips: list[tuple[str, str]] = field(default_factory=list, init=False)

    @property
    def spoken_mask(self) -> bool:
        return self.spoken_timer or self.spoken_tool

    def reset(self) -> None:
        """Start a new turn: forget what this turn said, called and scheduled."""
        self.spoken_timer = self.spoken_tool = self.spoken_text = self.saw_tool_call = False
        self.ack_end = None
        self.pending = None
        self.last_skips = []

    # ------------------------------------------------------------------ settings

    def mask_of(self, tool: str) -> ToolMask | None:
        """The tool's masking settings, or ``None`` (unmasked)."""
        if self.tool_mask is not None:
            return self.tool_mask(tool)
        key = self.phrase_for(tool) if self.phrase_for is not None else None
        return ToolMask(phrase_key=key) if key else None

    def delay_for(self, tool: str, mask: ToolMask) -> tuple[int, DelaySource]:
        """``(after_ms, source)`` for a masked tool."""
        if self.is_handoff(tool):
            return 0, "handoff"
        if mask.after_ms is not None:
            return max(0, mask.after_ms), "tool"
        return max(0, self.tool_after_ms), "default"

    def handoff_wait_ms_for(self, tool: str | None) -> int:
        """The hold cap for a hand-off of ``tool``: its own ``handoff_wait_ms``, else the
        policy-wide value."""
        mask = self.mask_of(tool) if tool else None
        if mask is not None and mask.handoff_wait_ms is not None:
            return max(0, mask.handoff_wait_ms)
        return max(0, self.handoff_wait_ms)

    # ------------------------------------------------------------------ acknowledgement

    def timer_s(self) -> float | None:
        """Seconds after the hop starts at which to fire the acknowledgement; ``None`` if off."""
        if self.after_ms <= 0:
            return None
        return self.after_ms / 1000.0

    def on_timer(self, spoken_anything: bool) -> str | None:
        """The acknowledgement timer fired: this turn's acknowledgement key, or ``None`` (
        only before any tool call, any phrase or any answer text)."""
        if (self.timer_s() is None or self.saw_tool_call or self.spoken_timer or self.spoken_tool
                or self.spoken_text or spoken_anything):
            return None
        self.spoken_timer = True
        return self.generic_key

    def note_ack(self, text: str, sent_at: float) -> None:
        """The acknowledgement ``text`` went on the wire at ``sent_at`` (anchor)."""
        self.ack_end = speech_end(text, sent_at, self.chars_per_s)

    # ------------------------------------------------------------------ tool phrases

    def on_tool_calls(self, names: Sequence[str], now: float,
                      spoken_anything: bool = False) -> list[ScheduledPhrase]:
        """Tool calls of one model message (or one Gateway tool start) reaching the bridge at
        ``now``: the phrase to schedule (zero or one), see the module docstring. The reasons the
        other tools got none are in :attr:`last_skips`."""
        self.last_skips = []
        if not names:
            return []
        self.saw_tool_call = True
        masked: list[tuple[str, ToolMask]] = []
        for name in names:
            mask = self.mask_of(name)
            if mask is None:
                self.last_skips.append((name, SKIP_TOOL_OFF))
            else:
                masked.append((name, mask))
        if not masked:
            return []
        if self.spoken_text or spoken_anything:
            self.last_skips += [(n, SKIP_ANSWER_STARTED) for n, _ in masked]
            return []
        name, mask = masked[0]
        self.last_skips += [(n, SKIP_ONE_PER_TURN) for n, _ in masked[1:]]
        handoff = self.is_handoff(name)
        if self.spoken_tool or (self.pending is not None and not handoff):
            self.last_skips.append((name, SKIP_ONE_PER_TURN))
            return []
        after_ms, source = self.delay_for(name, mask)
        deadline = now + after_ms / 1000.0
        if not handoff and self.ack_end is not None:
            deadline = max(deadline, self.ack_end)
        if self.pending is not None:  # a hand-off replaces a pending inline phrase
            self.last_skips.append((self.pending.tool, SKIP_SUPERSEDED))
        self.pending = ScheduledPhrase(tool=name, phrase_key=mask.phrase_key, after_ms=after_ms,
                                       source=source, called_at=now, deadline=deadline)
        return [self.pending]

    def discard(self, phrase: ScheduledPhrase) -> None:
        """The bridge could not render ``phrase`` (or dropped it): forget it, so a later tool call
        of the turn may still be masked."""
        if self.pending == phrase:
            self.pending = None

    def on_deadline(self, now: float) -> ScheduledPhrase | None:
        """The pending phrase's deadline passed: the phrase to speak now, or ``None`` (answer text
        already sent, nothing pending, or not due yet)."""
        p = self.pending
        if p is None or now < p.deadline:
            return None
        self.pending = None
        if self.spoken_text or self.spoken_tool:
            return None
        self.spoken_tool = True
        return p

    def on_text(self) -> list[ScheduledPhrase]:
        """The turn sent answer text (or an apology). Cancels and returns the pending phrase."""
        self.spoken_text = True
        cancelled = [self.pending] if self.pending is not None else []
        self.pending = None
        return cancelled

    def note_spoken(self) -> list[ScheduledPhrase]:
        """Alias of :meth:`on_text` (kept for callers of the API)."""
        return self.on_text()

    def end_turn(self) -> list[ScheduledPhrase]:
        """The turn is over: a still-pending phrase is dropped and returned."""
        cancelled = [self.pending] if self.pending is not None else []
        self.pending = None
        return cancelled
