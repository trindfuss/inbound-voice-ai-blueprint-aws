"""The close path's deadline and journal event.

Over a fake monotonic clock and the real Twilio clients (:mod:`tests.humanleg_close_fakes`):

* one deadline (:data:`~owlbank.humanleg_close.CLOSE_DEADLINE_S`) bounds every Twilio request of a
  close, TaskRouter's reservations read included; a close cut short releases its claim, logs
  ``human_leg_close_deadline`` and is retried by the next signal or the sweep;
* a successful close writes one ``conversation_closed`` journal event (allow-listed fields only), with
  ``closed_by`` from its signal;
* the sweep's 20 s stop rule still holds with the per-close deadline.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pytest

from lambdas.conversation_sweep import handler as sweep
from lambdas.frontdoor import runtime_sessions
from owlbank import callstate, humanleg, humanleg_client
from owlbank.humanleg import CallEnd
from owlbank.humanleg_client import TaskReservations
from owlbank.humanleg_close import CLOSE_DEADLINE_S, CLOSED_BY
from owlbank.segments import EVENT_TYPES, SegmentLog
from owlbank.store import MemoryStore
from tests.humanleg_close_fakes import (
    CONV,
    T0,
    TASK,
    FakeClock,
    SlowTwilio,
    closed_events,
    closer_over,
    escalated_call,
    fake_monotonic,
)

CALL = "CA" + "e" * 32
ROOT = Path(__file__).resolve().parents[2]


def _claim(store: MemoryStore, call_sid: str = CALL) -> Any:
    state = callstate.load(store, call_sid)
    assert state is not None
    return (state.escalation or {}).get(humanleg.K_CLOSE_CLAIMED_AT)


# --------------------------------------------------------------------------- the deadline


def test_the_deadline_fits_the_front_door_lambda_with_the_runtime_stop() -> None:
    """4 s of Twilio plus the AgentCore Runtime stop that may follow on ``/voice/status`` (connect +
    read) leaves room for DynamoDB inside the front door's 10 s Lambda timeout."""
    src = (ROOT / "infra" / "owlbank_infra" / "compute_stack.py").read_text()
    found = re.search(r'"frontdoor":\s*(\d+)', src[src.index("TIMEOUTS = {"):])
    assert found is not None
    frontdoor_timeout_s = int(found.group(1))
    runtime_stop_s = runtime_sessions.CONNECT_TIMEOUT_S + runtime_sessions.READ_TIMEOUT_S
    assert CLOSE_DEADLINE_S + runtime_stop_s <= frontdoor_timeout_s - 2


def test_a_slow_chain_never_exceeds_the_deadline_and_releases_the_claim(caplog) -> None:
    clock = FakeClock()
    store = MemoryStore()
    escalated_call(store, CALL)
    twilio = SlowTwilio(clock, default=1.5)  # every request 1.5 s: a full close would take 13.5 s
    closer = closer_over(store, twilio)
    start = clock.t
    with fake_monotonic(clock), caplog.at_level(logging.INFO):
        assert closer.end(CALL, humanleg.SWEEP) is False
    assert clock.t - start <= CLOSE_DEADLINE_S
    assert ("PUT", f"/v2/Conversations/{CONV}") not in twilio.sent  # never reached
    assert all(0 < t <= 2.0 for t in twilio.timeouts)
    assert _claim(store) is None  # released: the next signal or the sweep retries
    assert not humanleg.conversation_closed(callstate.load(store, CALL))
    assert closed_events(store, CALL) == []
    hit = [r for r in caplog.records if r.getMessage() == "human_leg_close_deadline"]
    assert len(hit) == 1 and hit[0].__dict__.get("decision") == "claim_released_retry_later"

    # The sweep's next run, with Twilio healthy again, closes it.
    twilio.default = 0.05
    with fake_monotonic(clock):
        assert closer.end(CALL, humanleg.SWEEP) is True
    assert [e["closed_by"] for e in closed_events(store, CALL)] == ["sweep"]


def test_a_timeout_on_the_last_request_is_cut_to_what_is_left() -> None:
    clock = FakeClock()
    store = MemoryStore()
    escalated_call(store, CALL)
    # resolve, reservations, stop, 2 x (GET, POST), GET: 3.2 s of 4 s; the PUT gets the last 0.8 s.
    twilio = SlowTwilio(clock, delays=[0.4] * 8 + [1.0])
    with fake_monotonic(clock):
        assert closer_over(store, twilio).end(CALL, humanleg.SWEEP) is False
    assert twilio.sent[-1] == ("PUT", f"/v2/Conversations/{CONV}")
    assert twilio.timeouts[-1] == pytest.approx(0.8)
    assert _claim(store) is None


def test_the_reservations_read_honours_the_budget() -> None:
    seen: list[float | None] = []

    class Source:
        def reservations(self, task_sid: str, *, timeout_s: float | None = None) -> list:
            seen.append(timeout_s)
            return []

    clock = FakeClock()
    with fake_monotonic(clock):
        reader = TaskReservations(Source(), timeout_s=2.0)
        assert reader.list(TASK) == []  # outside a budget: the source's own timeout
        with humanleg_client.budget(0.7):
            assert reader.list(TASK) == []
        with humanleg_client.budget(5.0):
            assert reader.list(TASK) == []
        with humanleg_client.budget(0.0):
            assert reader.list(TASK) is None  # spent: no read
    assert seen == [None, pytest.approx(0.7), 2.0]


def test_budget_left_is_none_outside_and_never_negative() -> None:
    clock = FakeClock()
    with fake_monotonic(clock):
        assert humanleg_client.budget_left() is None
        with humanleg_client.budget(1.0):
            assert humanleg_client.budget_left() == pytest.approx(1.0)
            clock.t += 3
            assert humanleg_client.budget_left() == 0.0


# --------------------------------------------------------------------------- the journal event


def test_conversation_closed_is_a_journal_event_type() -> None:
    assert "conversation_closed" in EVENT_TYPES
    assert set(CLOSED_BY.values()) == {"status", "transcription", "sweep"}


@pytest.mark.parametrize(("transcription", "event", "closed_by"), [
    (False, humanleg.CALL_ENDED, "status"),
    (True, humanleg.TRANSCRIPTION_STOPPED, "transcription"),
    (True, humanleg.SWEEP, "sweep"),
])
def test_a_successful_close_writes_one_event_with_allow_listed_fields(transcription, event, closed_by) -> None:
    clock = FakeClock()
    store = MemoryStore()
    escalated_call(store, CALL, transcription=transcription)
    closer = closer_over(store, SlowTwilio(clock))
    with fake_monotonic(clock):
        if event == humanleg.TRANSCRIPTION_STOPPED:
            assert closer.end(CALL, humanleg.CALL_ENDED) is False  # waits for the transcript's end
        assert closer.end(CALL, event) is True
        # Repeats (a late callback, a sweep) close nothing again and write no second event.
        for again in (humanleg.CALL_ENDED, humanleg.TRANSCRIPTION_STOPPED, humanleg.SWEEP):
            assert closer.end(CALL, again) is False
    events = closed_events(store, CALL)
    assert len(events) == 1
    ev = events[0]
    assert {k: ev[k] for k in ("type", "conversation_id", "closed_by")} == {
        "type": "conversation_closed", "conversation_id": CONV, "closed_by": closed_by}
    assert set(ev) == {"type", "conversation_id", "closed_by", "call_sid", "seq", "ts", "expires_at"}


def test_a_refused_close_writes_no_event() -> None:
    clock = FakeClock()
    store = MemoryStore()
    escalated_call(store, CALL, transcription=False)

    class Refusing(SlowTwilio):
        def _answer(self, method: str, path: str) -> tuple[int, Any]:
            if method == "PUT":
                return 409, {}
            return super()._answer(method, path)

    with fake_monotonic(clock):
        assert closer_over(store, Refusing(clock)).end(CALL, humanleg.CALL_ENDED) is False
    assert closed_events(store, CALL) == []
    assert _claim(store) is None


def test_a_journal_failure_does_not_undo_the_close(caplog) -> None:
    clock = FakeClock()
    store = MemoryStore()
    escalated_call(store, CALL, transcription=False)

    class Broken:
        def append(self, call_sid: str, event: Any) -> int:
            raise RuntimeError("segments table down")

    with fake_monotonic(clock), caplog.at_level(logging.WARNING):
        assert closer_over(store, SlowTwilio(clock), journal=Broken()).end(CALL, humanleg.CALL_ENDED) is True  # type: ignore[arg-type]
    assert humanleg.conversation_closed(callstate.load(store, CALL))
    assert [r for r in caplog.records if r.getMessage() == "human_leg_close_journal_failed"]


# --------------------------------------------------------------------------- the sweep


def test_the_sweep_stop_margin_covers_one_close_and_one_status_read() -> None:
    assert CLOSE_DEADLINE_S + sweep.REQUEST_TIMEOUT_S < sweep.STOP_MARGIN_S


def test_slow_sweep_closes_each_end_within_the_deadline_and_the_stop_rule_holds() -> None:
    """Five due conversations, Twilio so slow every close hits its deadline, 30 s of Lambda left:
    closes start only while 20 s or more remain, each ends within the deadline, so the run ends with
    more than the margin minus one deadline left; each cut-short close released its claim."""
    clock = FakeClock()
    store = MemoryStore()
    calls = tuple("CA" + c * 32 for c in "abcde")
    for sid in calls:
        escalated_call(store, sid)
    twilio = SlowTwilio(clock, default=1.9)
    closer = closer_over(store, twilio)
    lambda_end = clock.t + 30.0

    class Ended:
        def status(self, call_sid: str) -> CallEnd:
            return CallEnd("completed", T0 - 3600)

    deps = sweep.SweepDeps(due=sweep.store_due(store), states=sweep.store_states(store, clock=lambda: T0),
                           calls=Ended(), closer=closer, write=sweep.store_writer(store, clock=lambda: T0),
                           clock=lambda: T0)
    starts: list[float] = []
    real_end = closer.end

    def timed_end(call_sid: str, event: str, *, conversation_id: str | None = None) -> bool:
        starts.append(clock.t)
        before = clock.t
        out = real_end(call_sid, event, conversation_id=conversation_id)
        assert clock.t - before <= CLOSE_DEADLINE_S
        return out

    closer.end = timed_end  # type: ignore[method-assign]
    with fake_monotonic(clock):
        report = sweep.run_sweep(deps, time_left=lambda: lambda_end - clock.t)
    assert report.out_of_time and report.closed == 0 and 0 < len(starts) < len(calls)
    assert all(lambda_end - s >= sweep.STOP_MARGIN_S for s in starts)
    assert lambda_end - clock.t >= sweep.STOP_MARGIN_S - CLOSE_DEADLINE_S
    assert all(_claim(store, sid) is None for sid in calls)


def test_sweep_build_deps_journals_to_the_segments_table() -> None:
    deps = sweep.build_deps({"OWLBANK_SECRET_TWILIO_API_KEY": "arn:aws:secretsmanager:us-east-1:111122223333:"
                             "secret:twilio-api-key"}, store=MemoryStore())
    assert isinstance(deps.closer._journal, SegmentLog)  # type: ignore[attr-defined]
