"""``KeypadBuffer`` over generated key sequences and timings.

A sequence is a list of (key, gap before it) pairs; gaps are drawn from a wide range and from values
right at the window edge (``WINDOW_S`` and its float neighbours), and runs are biased to start with
0, 1, 2 or 9 and to go on for many keys (a card number). The reference model splits the keys into
runs: a key less than ``WINDOW_S`` after the previous one joins its run (strict "within", as in the
module). The simulation polls the way the bridge does, at each due time ``on_digit`` returned (in time
order, interleaved with the keys), optionally late (to exercise held entries). Properties:

* a non-``hint`` action comes only from a key with no other key within ``WINDOW_S`` before or after it,
  and is that key's own action;
* no key yields more than one action (every action's ``keys`` count covers distinct presses, and the
  counts sum to the number of presses);
* every multi-key run yields exactly one ``hint`` with that run's key count;
* the number of actions equals the number of runs, in order;
* no output carries a pressed character: each action is exactly (kind in ``ACTIONS``, int count),
  its repr is built from those alone, and the buffer's repr has no digit.
"""

from __future__ import annotations

import heapq
import math

from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.keypad import ACTIONS, KEY_ACTIONS, WINDOW_S, Action, KeypadBuffer

T0 = 1000.0
KEYS = list("0123456789*#") + ["A", "x", ""]
EDGE = [WINDOW_S, math.nextafter(WINDOW_S, 0), math.nextafter(WINDOW_S, 10), WINDOW_S - 0.01, WINDOW_S + 0.01]

gaps = st.one_of(
    st.sampled_from(EDGE),
    st.floats(min_value=0.0, max_value=WINDOW_S * 3, allow_nan=False),
    st.floats(min_value=0.0, max_value=0.4, allow_nan=False),  # fast typing
)


@st.composite
def runs(draw: st.DrawFn) -> list[tuple[str, float]]:
    """One burst: a first key (biased to 0/1/2/9) then 0 to 20 fast keys."""
    first = draw(st.one_of(st.sampled_from("0129"), st.sampled_from(KEYS)))
    more = draw(st.lists(st.tuples(st.sampled_from(KEYS), gaps), max_size=20))
    return [(first, draw(gaps)), *more]


sequences = st.lists(runs(), min_size=1, max_size=6).map(lambda rs: [p for r in rs for p in r])


def reference_runs(times: list[float], keys: list[str]) -> list[list[int]]:
    """Indices of the presses, split into runs by the module's own edge expression."""
    out: list[list[int]] = []
    for i, t in enumerate(times):
        if out and not t >= times[i - 1] + WINDOW_S:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def simulate(
    presses: list[tuple[str, float]], late: list[float]
) -> tuple[list[Action], list[float], list[str]]:
    buf = KeypadBuffer()
    times: list[float] = []
    t = T0
    for _, gap in presses:
        t += gap
        times.append(t)
    keys = [k for k, _ in presses]
    actions: list[Action] = []
    due: list[float] = []  # min-heap of scheduled polls
    reprs: list[str] = []
    for i, (key, now) in enumerate(zip(keys, times, strict=True)):
        while due and due[0] <= now:  # polls scheduled before (or at) this key run first, in order
            got = buf.poll(heapq.heappop(due))
            if got is not None:
                actions.append(got)
        d = buf.on_digit(key, now)
        assert d == now + WINDOW_S
        heapq.heappush(due, d + late[i % len(late)])
        reprs.append(repr(buf))
    while due:
        got = buf.poll(heapq.heappop(due))
        if got is not None:
            actions.append(got)
    # A final poll far later must find nothing left.
    assert buf.poll(times[-1] + 100 * WINDOW_S) is None
    reprs.append(repr(buf))
    return actions, times, keys


@settings(max_examples=400, deadline=None)
@given(
    presses=sequences,
    late=st.lists(st.sampled_from([0.0, 0.0, 0.0, 0.05, WINDOW_S, 2 * WINDOW_S]), min_size=1, max_size=4),
)
def test_one_action_per_run_never_a_digit(presses: list[tuple[str, float]], late: list[float]) -> None:
    actions, times, keys = simulate(presses, late)
    expected = reference_runs(times, keys)

    # Total actions = runs, in order, with each run's count; counts cover every press exactly once.
    assert len(actions) == len(expected)
    assert sum(a.keys for a in actions) == len(keys)
    for action, run in zip(actions, expected, strict=True):
        assert action.keys == len(run)
        if len(run) == 1:
            assert action.kind == KEY_ACTIONS.get(keys[run[0]], "hint")
        else:
            assert action.kind == "hint"

    # A non-hint action only for a key with no other key within WINDOW_S before or after it.
    for action, run in zip(actions, expected, strict=True):
        if action.kind != "hint":
            (i,) = run
            if i > 0:
                assert times[i] >= times[i - 1] + WINDOW_S
            if i + 1 < len(times):
                assert times[i + 1] >= times[i] + WINDOW_S

    # No output carries a pressed character.
    for action in actions:
        assert action.kind in ACTIONS
        assert type(action.keys) is int
        assert repr(action) == f"Action(kind={action.kind!r}, keys={action.keys})"


@settings(max_examples=200, deadline=None)
@given(presses=sequences)
def test_buffer_repr_has_no_digit(presses: list[tuple[str, float]]) -> None:
    buf = KeypadBuffer()
    t = T0
    for key, gap in presses:
        t += gap
        buf.on_digit(key, t)
        assert not any(ch.isdigit() for ch in repr(buf) + str(buf))
        for name in buf.__slots__:
            value = getattr(buf, name)
            if isinstance(value, str):
                assert value in ACTIONS  # only a mapped kind is stored, never the key


@settings(max_examples=200, deadline=None)
@given(
    first=st.sampled_from("0129"),
    rest=st.lists(st.sampled_from(KEYS), min_size=1, max_size=40),
    gap=st.floats(min_value=0.0, max_value=WINDOW_S - 0.001, allow_nan=False),
)
def test_long_entry_is_one_hint(first: str, rest: list[str], gap: float) -> None:
    """A long run starting with an action key, every gap inside the window: one hint, nothing else."""
    presses = [(first, 0.0)] + [(k, gap) for k in rest]
    actions, _, _ = simulate(presses, [0.0])
    assert actions == [Action("hint", len(presses))]


@settings(max_examples=300, deadline=None)
@given(presses=sequences, early=st.one_of(st.just(0.0), st.floats(1e-6, 0.005)))
def test_due_is_when_the_open_entry_completes(presses: list[tuple[str, float]], early: float) -> None:
    """``due`` (the bridge's re-wake time after an early wake-up): with an entry open
    and nothing held, ``poll`` is ``None`` before ``due`` and acts at ``due``; idle, ``due`` is ``None``."""
    buf = KeypadBuffer()
    now = 0.0
    for key, gap in presses:
        now += gap
        buf.on_digit(key, now)
    while buf.poll(-math.inf) is not None:  # drain the entries a later key completed (held)
        pass
    due = buf.due
    assert due is not None and due == now + WINDOW_S
    if early > 0:
        assert buf.poll(due - early) is None and buf.due == due  # an early poll changes nothing
    assert buf.poll(due) is not None
    assert buf.due is None and not buf.pending
