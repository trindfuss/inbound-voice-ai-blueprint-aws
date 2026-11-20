"""For any sequence of partial texts, ``partial_count`` equals the
number of partials and ``last_change_ms`` equals the time of the last length change.

Checked twice: on :func:`lambdas.frontdoor.gathers.count_partial` over both stores' contract
(``MemoryStore``; DynamoStore runs the same ``increment`` contract in ``tests/unit/test_store.py``),
and end to end through the front door's ``/voice/partial`` route with telemetry installed, where a
partial that completes the answer claims the Gather (the claim must keep the counters).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lambdas.frontdoor import gathers
from owlbank import sensitive
from owlbank.segments import clear_sensitive
from owlbank.store import MemoryStore
from owlbank.telemetry import frontdoor
from tests.component.test_frontdoor_early_digits import gather_of, gid_of, make, partial
from tests.component.test_frontdoor_support import CLOCK, THOMAS, sid
from tests.component.test_frontdoor_telemetry import install

CALL = "CA" + "0a" * 16
WORDS = ["nine", "four", "one", "oh", "five", "um", "zero", "two"]

texts = st.lists(st.sampled_from(WORDS), max_size=8).map(" ".join)
partials = st.lists(st.tuples(texts, st.integers(min_value=0, max_value=5_000)), min_size=1,
                    max_size=12)


def expected(seq: list[tuple[str, int]], times_ms: list[int]) -> tuple[int, int | None, int | None]:
    """The model: (count, last length, time of the last length change)."""
    last_len, last_change = None, None
    for (text, _gap), now in zip(seq, times_ms, strict=True):
        if len(text.strip()) != last_len:
            last_len, last_change = len(text.strip()), now
    return len(seq), last_len, last_change


@given(seq=partials)
def test_partial_count_and_last_change(seq: list[tuple[str, int]]) -> None:
    store = MemoryStore()
    now, times = 1_000_000, []
    for text, gap in seq:
        now += gap
        times.append(now)
        gathers.count_partial(store, CALL, "g" * 16, length=len(text.strip()), now_ms=now,
                              expires_at=99)
    item = gathers.load(store, CALL, "g" * 16)
    count, last_len, last_change = expected(seq, times)
    assert item is not None
    assert item["partial_count"] == count
    assert item["last_len"] == last_len and item["last_change_ms"] == last_change
    assert item["expires_at"] == 99 and gathers.state_of(item) == gathers.NONE


@pytest.fixture
def _traced(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(frontdoor, "_tracer", None)
    install(InMemorySpanExporter())
    yield
    monkeypatch.setattr(frontdoor, "_tracer", None)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seq=partials)
def test_through_the_partial_route(_traced: None, seq: list[tuple[str, int]]) -> None:
    h, _calls = make()
    now = [CLOCK]
    h.app.d.clock = lambda: now[0]
    call = sid(3)
    try:
        g = gather_of(h.start(call, THOMAS))
        times = []
        for text, gap in seq:
            now[0] += gap / 1000
            times.append(int(now[0] * 1000))   # the front door's own ms (int(clock() * 1000))
            partial(h, call, g, text)
        item = gathers.load(h.store, call, gid_of(g))
        count, last_len, last_change = expected(seq, times)
        assert item is not None
        assert item["partial_count"] == count
        assert item.get("last_len") == last_len
        assert item.get("last_change_ms") == last_change
    finally:
        clear_sensitive(call)
        sensitive.forget(call)
