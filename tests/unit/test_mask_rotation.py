"""The timed acknowledgement rotates through a pool within a call.

``next_generic_key`` picks the next pool key after the call's last acknowledgement; the bridge keeps
the last one per call SID, so one call never hears the same acknowledgement twice in a row, and
another call starts at the first.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from owlbank.policy import MASK_GENERIC_KEY, MASK_GENERIC_KEYS
from tac_bridge.bridge import next_generic_key
from tests.component.masking_fixtures import masked_policy_doc
from tests.component.test_bridge_fakes import CALL_SID, run, say, write_call_state
from tests.component.test_bridge_masking import collect, context, direct, en


def test_pool_shape() -> None:
    assert len(MASK_GENERIC_KEYS) >= 4
    assert len(set(MASK_GENERIC_KEYS)) == len(MASK_GENERIC_KEYS)
    assert MASK_GENERIC_KEYS[0] == "mask.generic.1" and MASK_GENERIC_KEYS[0] == MASK_GENERIC_KEY


def test_starts_at_the_first_key_for_none_or_unknown() -> None:
    assert next_generic_key(None) == MASK_GENERIC_KEYS[0]
    assert next_generic_key("mask.generic") == MASK_GENERIC_KEYS[0]  # the previous key
    assert next_generic_key("mask.get_balance") == MASK_GENERIC_KEYS[0]


def test_cycles_through_the_pool_in_order() -> None:
    seen: list[str] = []
    last: str | None = None
    for _ in range(2 * len(MASK_GENERIC_KEYS)):
        last = next_generic_key(last)
        seen.append(last)
    assert seen == [*MASK_GENERIC_KEYS, *MASK_GENERIC_KEYS]


@given(st.one_of(st.none(), st.sampled_from(MASK_GENERIC_KEYS), st.text(max_size=20)))
def test_never_returns_the_previous_key(last: str | None) -> None:
    got = next_generic_key(last)
    assert got in MASK_GENERIC_KEYS
    assert got != last


@given(st.lists(st.sampled_from(["a", "b", "c", "d"]), min_size=2, max_size=6, unique=True),
       st.integers(0, 20))
def test_any_pool_of_two_or_more_rotates_without_repeats(pool: list[str], steps: int) -> None:
    last: str | None = None
    for _ in range(steps):
        nxt = next_generic_key(last, pool)
        assert nxt in pool and nxt != last
        last = nxt


def test_consecutive_slow_turns_of_a_call_speak_different_acknowledgements() -> None:
    """Component: two slow turns of one call hear .1 then .2; another call starts at .1."""
    other_sid = "CA" + "f" * 32
    d = direct([say("First."), say("Second."), say("Other call.")], agent_delay_s=0.15,
               policy_doc=masked_policy_doc(after_ms=50))
    write_call_state(d.store, other_sid)
    first, _ = run(collect(d, "hello"))
    second, _ = run(collect(d, "and again"))
    other_ctx = context()
    other_ctx.call_sid = other_sid
    other, _ = run(collect(d, "hi", other_ctx))
    ack1, ack2 = en(MASK_GENERIC_KEYS[0]), en(MASK_GENERIC_KEYS[1])
    assert ack1 != ack2
    assert first == [ack1 + " ", "First."]
    assert second == [ack2 + " ", "Second."]
    assert other == [ack1 + " ", "Other call."]
    keys = [e["prompt_key"] for e in d.events() if e["type"] == "prompt"]
    assert keys == [MASK_GENERIC_KEYS[0], MASK_GENERIC_KEYS[1]]  # CALL_SID's journal only
    assert other_sid != CALL_SID


def test_rotation_memory_is_bounded() -> None:
    """The per-call memory never grows past MASK_ROTATION_MAX_CALLS (oldest calls dropped)."""
    import tac_bridge.bridge as b

    rot: dict[str, str] = {}
    for n in range(b.MASK_ROTATION_MAX_CALLS + 25):
        rot.pop(f"CA{n}", None)
        rot[f"CA{n}"] = "mask.generic.1"
        while len(rot) > b.MASK_ROTATION_MAX_CALLS:
            rot.pop(next(iter(rot)))
    assert len(rot) == b.MASK_ROTATION_MAX_CALLS and "CA0" not in rot
