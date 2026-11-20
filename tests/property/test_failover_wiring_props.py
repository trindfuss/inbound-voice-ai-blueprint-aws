"""Model failover, properties of the client's reporting (design, 4a.2,
4a.4).

For any sequence of model-call outcomes on any entries, the fleet reporter receives exactly the calls
that are samples (never a request error, never a call the turn's cancellation ended), each with its
outcome, an OK call with its TTFT and an abandoned one with the deadline it missed; the
``Unavailable`` metric counts each entry once, when this process first marks it.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.failover import UNAVAILABLE, FailoverState, ModelUnusable, Outcome
from tests.strands_fakes import client_error
from tests.unit.test_strands_failover import EntryModel, failover_settings, make

N = 3

#: (kind, exception or None, stop reason, ttft)
_KINDS = st.sampled_from(["ok", "abandoned", "throttled", "model_error", "unavailable", "request_error",
                          "cancelled"])


def _call(kind: str, ttft: float) -> tuple[BaseException | None, str | None, float | None]:
    if kind == "ok":
        return None, "end_turn", ttft
    if kind == "cancelled":
        return None, "cancelled", None
    if kind == "abandoned":
        return ModelUnusable(0, Outcome.ABANDONED), None, None
    if kind == "throttled":
        return client_error("ThrottlingException"), None, None
    if kind == "model_error":
        return client_error("ModelErrorException"), None, None
    if kind == "unavailable":
        return client_error("ResourceNotFoundException"), None, None
    return client_error("ValidationException", "The model returned the following errors: x"), None, None


EXPECTED = {"ok": Outcome.OK, "abandoned": Outcome.ABANDONED, "throttled": Outcome.THROTTLED,
            "model_error": Outcome.ERROR, "unavailable": UNAVAILABLE}


class Reporter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, Any, float | None]] = []

    def record(self, index: int, outcome: Any, ttft_s: float | None = None) -> None:
        self.calls.append((index, outcome, ttft_s))


class Events:
    def __init__(self) -> None:
        self.unavailable_count = 0
        self.failovers = 0

    def unavailable(self) -> None:
        self.unavailable_count += 1

    def failover(self) -> None:
        self.failovers += 1

    def active(self, rank: int) -> None:
        return None


@settings(max_examples=200, deadline=None)
@given(st.lists(st.tuples(st.integers(0, N - 1), _KINDS, st.floats(0, 10, allow_nan=False)), max_size=40))
def test_the_reporter_gets_exactly_the_samples(calls: list[tuple[int, str, float]]) -> None:
    s = failover_settings(N)
    client = make([EntryModel() for _ in range(N)], state=FailoverState(s))
    reporter, events = Reporter(), Events()
    client.use_fleet_reporter(reporter)
    client.use_failover_events(events)
    expected: list[tuple[int, Any, float | None]] = []
    marked: set[int] = set()
    for entry, kind, ttft in calls:
        exc, stop, t = _call(kind, ttft)
        client._on_model_call(entry, exc, stop, t)  # noqa: SLF001 - the one place every call is recorded
        if kind in ("cancelled", "request_error"):
            continue
        outcome = EXPECTED[kind]
        reported = ttft if kind == "ok" else s.entry(entry).deadline_s if kind == "abandoned" else None
        expected.append((entry, outcome, reported))
        if kind == "unavailable":
            marked.add(entry)
    assert reporter.calls == expected
    assert events.unavailable_count == len(marked)
    assert events.failovers == 0  # only a turn moves a call: recording never counts a failover
