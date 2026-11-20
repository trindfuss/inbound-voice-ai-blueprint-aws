"""Orchestrator webhook de-duplication (``IdempotencyCache``) over a fake clock.

* With room for every token, each distinct token is processed exactly once per expiry window:
  a token is processed iff it was never processed, or its last processing was at least
  ``ttl_s`` ago.
* Whatever the bound, the cache never holds more than ``max_entries``, and a token processed
  less than ``ttl_s`` ago that is still held is never processed again.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from tac_bridge.app import IdempotencyCache

TTL = 10.0
events = st.lists(st.tuples(st.sampled_from([f"t{i}" for i in range(12)]),
                            st.floats(0, 4, allow_nan=False)), max_size=200)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@settings(max_examples=300, deadline=None)
@given(events)
def test_each_unexpired_token_is_processed_exactly_once(seq: list[tuple[str, float]]) -> None:
    clock = Clock()
    cache = IdempotencyCache(ttl_s=TTL, max_entries=1000, clock=clock)
    first_seen: dict[str, float] = {}
    for token, dt in seq:
        clock.t += dt
        processed = not cache.check_and_add(token)
        expect = token not in first_seen or clock.t >= first_seen[token] + TTL
        assert processed == expect
        if processed:
            first_seen[token] = clock.t
        assert len(cache) <= len(first_seen)


@settings(max_examples=300, deadline=None)
@given(events, st.integers(1, 8))
def test_the_cache_never_exceeds_its_bound(seq: list[tuple[str, float]], bound: int) -> None:
    clock = Clock()
    cache = IdempotencyCache(ttl_s=TTL, max_entries=bound, clock=clock)
    for token, dt in seq:
        clock.t += dt
        cache.check_and_add(token)
        assert len(cache) <= bound
        # just recorded or a live duplicate: an immediate repeat is always a duplicate
        assert cache.check_and_add(token) is True
        assert len(cache) <= bound
