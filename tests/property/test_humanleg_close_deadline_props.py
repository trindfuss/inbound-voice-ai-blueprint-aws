"""For any per-request delays, the close path's total time stays under its
deadline, and its outcome is consistent.

The real :class:`~owlbank.humanleg_close.HumanLegCloser` and Twilio clients over
:class:`tests.humanleg_close_fakes.SlowTwilio` and a fake monotonic clock: each request takes its
drawn delay, or times out after the timeout it was given. For any delays, any end signal and any
deadline:

* the Twilio time of one close never exceeds the deadline, and no request is sent with no time left
  or with more than the per-request timeout;
* the close either succeeded (``conversation_closed_at`` set, exactly one ``conversation_closed``
  journal event with the signal's ``closed_by``) or left nothing behind (claim released, no event,
  not marked closed), so the next signal or the sweep can retry;
* a close cut short and then retried with Twilio healthy closes once and writes one event in total.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank import callstate, humanleg
from owlbank.humanleg_close import CLOSE_DEADLINE_S, CLOSED_BY
from owlbank.segments import SegmentLog
from owlbank.store import MemoryStore
from tests.humanleg_close_fakes import (
    REQUEST_TIMEOUT_S,
    T0,
    FakeClock,
    SlowTwilio,
    closed_events,
    closer_over,
    escalated_call,
    fake_monotonic,
)

CALL = "CA" + "e" * 32
DELAYS = st.lists(st.floats(min_value=0.0, max_value=3 * REQUEST_TIMEOUT_S, allow_nan=False), max_size=12)
#: Signals that close at once (a transcription named: only the sweep closes while it may run).
SIGNALS = st.sampled_from([(False, humanleg.CALL_ENDED), (True, humanleg.SWEEP),
                           (False, humanleg.SWEEP), (False, humanleg.TRANSCRIPTION_ERROR)])
EPS = 1e-9


def _run(delays: list[float], signal: tuple[bool, str], deadline_s: float = CLOSE_DEADLINE_S,
         default: float = 0.05) -> tuple[bool, float, SlowTwilio, MemoryStore, FakeClock]:
    transcription, event = signal
    clock = FakeClock()
    store = MemoryStore()
    escalated_call(store, CALL, transcription=transcription)
    twilio = SlowTwilio(clock, delays=delays, default=default)
    closer = closer_over(store, twilio, journal=SegmentLog(store, now=lambda: T0))
    closer._deadline_s = deadline_s  # noqa: SLF001 - the property ranges over deadlines too
    start = clock.t
    with fake_monotonic(clock):
        closed = closer.end(CALL, event)
    return closed, clock.t - start, twilio, store, clock


@settings(max_examples=400, deadline=None)
@given(delays=DELAYS, signal=SIGNALS, default=st.floats(min_value=0.0, max_value=REQUEST_TIMEOUT_S + 1),
       deadline_s=st.floats(min_value=0.1, max_value=CLOSE_DEADLINE_S))
def test_total_close_time_stays_under_the_deadline(delays, signal, default, deadline_s) -> None:
    closed, elapsed, twilio, store, _ = _run(delays, signal, deadline_s, default)
    assert elapsed <= deadline_s + EPS
    assert all(0 < t <= REQUEST_TIMEOUT_S for t in twilio.timeouts)
    state = callstate.load(store, CALL)
    assert state is not None
    esc = state.escalation or {}
    events = closed_events(store, CALL)
    if closed:
        assert humanleg.conversation_closed(state)
        assert [(e["closed_by"], e["conversation_id"]) for e in events] == [
            (CLOSED_BY[signal[1]], esc[humanleg.K_CONVERSATION_ID])]
    else:
        assert not humanleg.conversation_closed(state)
        assert humanleg.K_CLOSE_CLAIMED_AT not in esc
        assert events == []


@settings(max_examples=200, deadline=None)
@given(delays=DELAYS, signal=SIGNALS)
def test_a_cut_short_close_is_retried_once_and_journalled_once(delays, signal) -> None:
    closed, _, twilio, store, clock = _run(delays, signal)
    retry = closer_over(store, SlowTwilio(clock), journal=SegmentLog(store, now=lambda: T0))
    with fake_monotonic(clock):
        again = retry.end(CALL, humanleg.SWEEP)
    assert closed != again  # exactly one of the two closed it
    assert humanleg.conversation_closed(callstate.load(store, CALL))
    assert len(closed_events(store, CALL)) == 1
