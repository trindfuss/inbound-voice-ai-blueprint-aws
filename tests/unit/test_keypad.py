"""owlbank.keypad: each key
alone, the multi-key rule, the window edge (strict: a key exactly ``WINDOW_S`` after the previous one
starts a new entry), ``clear``, ``poll`` timing and the no-digits rule for reprs. Property test is
in tests/property/test_keypad_props.py."""

from __future__ import annotations

import dataclasses

import pytest

from owlbank.keypad import ACTIONS, WINDOW_S, Action, KeypadBuffer

T0 = 100.0


def _press(buf: KeypadBuffer, keys: str, start: float = T0, gap: float = 0.2) -> float:
    due = start
    for i, key in enumerate(keys):
        due = buf.on_digit(key, start + i * gap)
    return due


@pytest.mark.parametrize(
    ("key", "kind"),
    [("1", "yes"), ("2", "no"), ("0", "agent"), ("9", "repeat")] + [(k, "hint") for k in "345678*#"],
)
def test_each_key_alone(key: str, kind: str) -> None:
    buf = KeypadBuffer()
    due = buf.on_digit(key, T0)
    assert due == T0 + WINDOW_S
    assert buf.poll(due) == Action(kind, 1)
    assert buf.poll(due + 10) is None


def test_window_is_one_and_a_half_seconds() -> None:
    assert WINDOW_S == 1.5
    assert {"yes", "no", "agent", "repeat", "hint"} == ACTIONS


@pytest.mark.parametrize("first", ["1", "2", "0", "9", "5"])
def test_second_key_inside_window_gives_one_hint(first: str) -> None:
    buf = KeypadBuffer()
    buf.on_digit(first, T0)
    assert buf.poll(T0 + WINDOW_S - 0.01) is None
    due = buf.on_digit("4", T0 + WINDOW_S - 0.01)
    assert due == T0 + WINDOW_S - 0.01 + WINDOW_S
    assert buf.poll(T0 + WINDOW_S) is None  # the first key's old due time: nothing
    assert buf.poll(due) == Action("hint", 2)
    assert buf.poll(due + 10) is None


def test_long_entry_starting_with_one_never_says_yes() -> None:
    buf = KeypadBuffer()
    due = _press(buf, "1234567812345678")
    assert buf.poll(due - 0.001) is None
    assert buf.poll(due) == Action("hint", 16)
    assert buf.poll(due) is None


def test_key_just_after_window_acts_alone() -> None:
    buf = KeypadBuffer()
    first_due = buf.on_digit("1", T0)
    assert buf.poll(first_due) == Action("yes", 1)
    second_due = buf.on_digit("2", T0 + WINDOW_S + 0.01)
    assert buf.poll(second_due) == Action("no", 1)


def test_edge_exactly_at_window_starts_a_new_entry() -> None:
    """Strict "within": a key at exactly last + WINDOW_S is a new entry, even before the poll ran."""
    buf = KeypadBuffer()
    buf.on_digit("1", T0)
    second_due = buf.on_digit("0", T0 + WINDOW_S)  # the first entry's poll hasn't run yet
    assert buf.poll(T0 + WINDOW_S) == Action("yes", 1)  # held, not lost
    assert buf.poll(T0 + WINDOW_S) is None
    assert buf.poll(second_due) == Action("agent", 1)


def test_late_poll_returns_held_entries_oldest_first() -> None:
    buf = KeypadBuffer()
    buf.on_digit("1", T0)
    buf.on_digit("2", T0 + 2)
    buf.on_digit("3", T0 + 2.1)
    last_due = buf.on_digit("9", T0 + 5)
    assert buf.poll(T0 + 5) == Action("yes", 1)
    assert buf.poll(T0 + 5) == Action("hint", 2)
    assert buf.poll(T0 + 5) is None
    assert buf.poll(last_due) == Action("repeat", 1)


@pytest.mark.parametrize("key", ["A", "x", "", "11", "é", " "])
def test_unknown_character_counts_as_a_key(key: str) -> None:
    buf = KeypadBuffer()
    due = buf.on_digit(key, T0)
    assert buf.poll(due) == Action("hint", 1)
    buf.on_digit("1", T0 + 10)
    due = buf.on_digit(key, T0 + 10.5)
    assert buf.poll(due) == Action("hint", 2)


def test_clear_drops_pending_and_held_entries() -> None:
    buf = KeypadBuffer()
    buf.on_digit("0", T0)
    buf.on_digit("1", T0 + 2)  # holds the 0's action
    assert buf.pending
    buf.clear()
    assert not buf.pending
    assert buf.poll(T0 + 100) is None
    due = buf.on_digit("2", T0 + 101)  # a fresh entry after clear
    assert buf.poll(due) == Action("no", 1)


def test_poll_before_due_returns_none() -> None:
    buf = KeypadBuffer()
    assert buf.poll(T0) is None  # nothing pressed
    due = buf.on_digit("1", T0)
    assert buf.poll(T0) is None
    assert buf.poll(due - 1e-6) is None
    assert buf.poll(due) == Action("yes", 1)


def test_poll_twice_returns_the_action_once() -> None:
    buf = KeypadBuffer()
    due = buf.on_digit("9", T0)
    assert buf.poll(due + 1) == Action("repeat", 1)
    assert buf.poll(due + 1) is None
    assert buf.poll(due + 2) is None


def test_action_holds_a_kind_and_a_count_only() -> None:
    assert [f.name for f in dataclasses.fields(Action)] == ["kind", "keys"]
    action = Action("hint", 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        action.kind = "yes"  # type: ignore[misc]
    assert repr(action) == "Action(kind='hint', keys=2)"


def test_reprs_contain_no_digits() -> None:
    buf = KeypadBuffer()
    assert repr(buf) == str(buf) == "KeypadBuffer(idle)"
    # Seven presses of 5 then 3: the buffer's repr has no digit at all; the action's repr has only
    # the count (8), never a pressed digit.
    due = _press(buf, "55555553")
    text = repr(buf) + str(buf)
    assert text == "KeypadBuffer(pending)" * 2
    assert not any(ch.isdigit() for ch in text)
    action = buf.poll(due)
    assert action == Action("hint", 8)
    assert "5" not in repr(action) and "3" not in repr(action)
    assert not any(ch.isdigit() for ch in repr(buf))
    assert not any(ch.isdigit() for ch in vars_of(buf))


def vars_of(buf: KeypadBuffer) -> str:
    """Every stored string attribute of the buffer (no pressed character may be kept)."""
    return "".join(getattr(buf, name) for name in buf.__slots__ if isinstance(getattr(buf, name), str))
