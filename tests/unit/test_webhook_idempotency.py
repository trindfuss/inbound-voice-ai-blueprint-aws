"""``IdempotencyCache``, the bridge's Orchestrator webhook de-duplication."""

from __future__ import annotations

import pytest

from tac_bridge.app import IdempotencyCache, token_ref


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_a_repeated_token_is_a_duplicate() -> None:
    cache = IdempotencyCache(clock=Clock())
    assert cache.check_and_add("tok-1") is False
    assert cache.check_and_add("tok-1") is True
    assert cache.check_and_add("tok-2") is False
    assert len(cache) == 2


def test_an_expired_token_is_processed_again_and_a_repeat_does_not_extend_it() -> None:
    clock = Clock()
    cache = IdempotencyCache(ttl_s=600, clock=clock)
    assert cache.check_and_add("tok") is False
    clock.t += 599
    assert cache.check_and_add("tok") is True  # still remembered, not extended
    clock.t += 1
    assert cache.check_and_add("tok") is False  # 600 s after first seen: expired
    assert cache.check_and_add("tok") is True


def test_the_cache_is_bounded_and_drops_the_oldest() -> None:
    clock = Clock()
    cache = IdempotencyCache(max_entries=3, clock=clock)
    for t in ("a", "b", "c", "d"):
        clock.t += 1
        assert cache.check_and_add(t) is False
    assert len(cache) == 3
    assert cache.check_and_add("d") is True
    assert cache.check_and_add("a") is False  # dropped as the oldest, so new again


def test_expired_entries_are_pruned() -> None:
    clock = Clock()
    cache = IdempotencyCache(ttl_s=10, clock=clock)
    for t in ("a", "b", "c"):
        cache.check_and_add(t)
    clock.t += 11
    cache.check_and_add("d")
    assert len(cache) == 1


def test_tokens_are_never_kept_and_the_log_reference_is_a_short_hash() -> None:
    cache = IdempotencyCache(clock=Clock())
    cache.check_and_add("secret-token-value")
    assert all(isinstance(k, bytes) and b"secret" not in k for k in cache._seen)  # noqa: SLF001
    ref = token_ref("secret-token-value")
    assert len(ref) == 8 and "secret" not in ref and ref == token_ref("secret-token-value")


@pytest.mark.parametrize(("ttl", "size"), [(0, 10), (10, 0), (-1, 10)])
def test_invalid_settings_are_rejected(ttl: float, size: int) -> None:
    with pytest.raises(ValueError):
        IdempotencyCache(ttl_s=ttl, max_entries=size)
