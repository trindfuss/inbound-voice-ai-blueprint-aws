"""Keypad shortcuts during the AI segment: ``KeypadBuffer`` decides what a key press means.

A single key acts once the caller has stopped pressing keys for :data:`WINDOW_S`:

* ``1`` → ``yes``, ``2`` → ``no``, ``0`` → ``agent``, ``9`` → ``repeat``;
* any other key (``3`` to ``8``, ``*``, ``#``, or any character outside ``0-9*#``) → ``hint``.

A second key within :data:`WINDOW_S` of the previous one makes a multi-key entry (a card number,
a date, a PIN typed by habit). The entry ends :data:`WINDOW_S` after its last key and yields one
``hint`` and nothing else, so a card number starting with 1 never answers "yes".

**The window edge.** "Within" is strict: a key less than :data:`WINDOW_S` after the previous key
joins its entry; a key exactly :data:`WINDOW_S` (or more) after it starts a new one. This matches
:meth:`KeypadBuffer.poll`, which completes an entry once ``now >= last_key + WINDOW_S``, so an
entry is complete at the same instant whether the bridge polls first or the next key arrives
first. An entry completed by a late key (the bridge's poll hasn't run yet) is held and returned
by the next :meth:`KeypadBuffer.poll`, oldest first; nothing is lost.

**Privacy.** The buffer never keeps a pressed character: it keeps the action the first key
maps to, the key count and the last key's time. :class:`Action` holds a kind and a count, never
the digits, and the buffer's ``repr`` shows neither digits nor counts.

The module is product-neutral: it knows nothing of TAC, ConversationRelay or WebSockets,
has no timers, does no I/O and logs nothing. Time is passed in on the ``time.monotonic()`` scale.
The bridge calls :meth:`KeypadBuffer.on_digit` for each key, schedules a wake-up at the due time
it returns and calls :meth:`KeypadBuffer.poll` then.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

#: Seconds of quiet after a key before it acts; a second key inside it makes a multi-key entry.
WINDOW_S = 1.5

YES = "yes"
NO = "no"
AGENT = "agent"
REPEAT = "repeat"
HINT = "hint"
#: Every action kind the buffer can return.
ACTIONS: frozenset[str] = frozenset({YES, NO, AGENT, REPEAT, HINT})
#: Keys with an action of their own when pressed alone; every other key is a ``hint``.
KEY_ACTIONS: dict[str, str] = {"1": YES, "2": NO, "0": AGENT, "9": REPEAT}


@dataclass(frozen=True, slots=True)
class Action:
    """What one keypad entry means: ``kind`` (one of :data:`ACTIONS`) and how many keys made it.

    ``keys`` is a count, never the pressed characters: 1 for a lone key, 2 or more for a
    multi-key entry (always a ``hint``).
    """

    kind: str
    keys: int


def kind_for(digit: str) -> str:
    """The action a key pressed alone maps to (any unknown character is a ``hint``)."""
    return KEY_ACTIONS.get(digit, HINT)


class KeypadBuffer:
    """One call's pending keypad entry: a pure state machine over a passed-in clock."""

    __slots__ = ("_first_kind", "_keys", "_last_at", "_ready")

    def __init__(self) -> None:
        self._first_kind: str = HINT
        self._keys: int = 0
        self._last_at: float = 0.0
        self._ready: deque[Action] = deque()

    def __repr__(self) -> str:
        # No digits and no counts: the repr may end up in a log line or a traceback.
        state = "pending" if self._keys or self._ready else "idle"
        return f"KeypadBuffer({state})"

    __str__ = __repr__

    @property
    def pending(self) -> bool:
        """True while an entry is open or a completed one waits for :meth:`poll`."""
        return bool(self._keys or self._ready)

    @property
    def due(self) -> float | None:
        """When the open entry completes (``None`` with no open entry). A wake-up that runs before
        this (an event loop's timer may run slightly early) polls ``None`` and must wake again."""
        return self._last_at + WINDOW_S if self._keys else None

    def on_digit(self, digit: str, now: float) -> float:
        """Record one key at ``now``; return the due time at which the caller should :meth:`poll`.

        Any string counts as one key; only the action it maps to is kept, never the string.
        """
        # The same comparison as poll's (not ``now - last >= WINDOW_S``), so float rounding can't
        # put the edge in a different place for the two.
        if self._keys and now >= self._last_at + WINDOW_S:
            # The open entry was already complete (its poll hasn't run yet): hold its action.
            self._ready.append(self._complete())
        if self._keys:
            self._keys += 1
        else:
            self._first_kind = kind_for(digit)
            self._keys = 1
        self._last_at = now
        return now + WINDOW_S

    def poll(self, now: float) -> Action | None:
        """The next completed entry's action, once; ``None`` while nothing is complete."""
        if self._ready:
            return self._ready.popleft()
        if self._keys and now >= self._last_at + WINDOW_S:
            return self._complete()
        return None

    def clear(self) -> None:
        """Drop any open or completed entry (a hand-off started, or the session ended)."""
        self._keys = 0
        self._first_kind = HINT
        self._ready.clear()

    def _complete(self) -> Action:
        action = Action(self._first_kind if self._keys == 1 else HINT, self._keys)
        self._keys = 0
        self._first_kind = HINT
        return action
