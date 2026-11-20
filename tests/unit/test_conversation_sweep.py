"""Unit tests for the stale-conversation sweep from our own records.

* :func:`owlbank.humanleg.sweep_step`, :func:`~owlbank.humanleg.sweep_due_at` and the pure writes:
  close only a call ended more than 5 minutes ago; never a live or unknown call; the due-time rule;
  nothing written once the conversation is closed;
* :class:`lambdas.conversation_sweep.handler.CallStatusReader`: the Calls API request and its
  answers, failures as ``None`` (unknown), no key in the logs;
* :func:`lambdas.conversation_sweep.handler.run_sweep` over the ``sweep-due`` index of a call-state
  store and the real close path (:class:`owlbank.humanleg_close.HumanLegCloser`): ended calls are closed
  once (most with no Calls API read), live ones rescheduled, stale index entries skipped, the run stops
  for time, a failed query fails the run.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

import pytest

from lambdas.conversation_sweep import handler as sweep
from owlbank import callstate, humanleg
from owlbank.callstate import CallState
from owlbank.humanleg import (
    K_CALL_ENDED_AT,
    K_CONVERSATION_CLOSED_AT,
    K_CONVERSATION_ID,
    K_ESCALATED_AT,
    K_SWEEP_NEXT_AT,
    STEP_CLOSE,
    STEP_LIVE,
    STEP_SKIP,
    STEP_UNKNOWN,
    STEP_WAIT,
    SWEEP_GRACE_S,
    CallEnd,
    SweepStep,
    sweep_due_at,
    sweep_step,
)
from owlbank.humanleg_close import HumanLegCloser
from owlbank.store import MemoryStore

ACCOUNT = "AC" + "4" * 32
KEY, SECRET = "SK" + "3" * 32, "s3cr3t-value-xyz"
CREDS = (ACCOUNT, KEY, SECRET)
NOW = 1_793_873_730.0
SLACK = humanleg.SWEEP_SLACK_S
CALLS = tuple("CA" + c * 32 for c in "abcdef")
CONVS = tuple(f"conv_{c * 26}" for c in "abcdef")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _esc(**extra: Any) -> dict[str, Any]:
    return {"reason": "fraud_suspected", "queue": "q", K_ESCALATED_AT: _iso(NOW - 3600), **extra}


def _ended(ago: float, status: str = "completed") -> CallEnd:
    return CallEnd(status, NOW - ago)


# --------------------------------------------------------------------------- sweep_step

def test_a_call_not_escalated_or_closed_is_skipped() -> None:
    assert sweep_step(None, _ended(600), NOW) == SweepStep(STEP_SKIP)
    assert sweep_step({}, _ended(600), NOW) == SweepStep(STEP_SKIP)
    assert sweep_step(_esc(**{K_CONVERSATION_CLOSED_AT: int(NOW)}), _ended(600), NOW) == SweepStep(STEP_SKIP)


def test_a_recorded_end_beyond_the_grace_period_closes_without_a_status() -> None:
    assert sweep_step(_esc(**{K_CALL_ENDED_AT: int(NOW) - 301}), None, NOW) == SweepStep(STEP_CLOSE)


def test_a_recorded_end_within_the_grace_period_waits_until_just_after_it() -> None:
    end = int(NOW) - 300
    assert sweep_step(_esc(**{K_CALL_ENDED_AT: end}), None, NOW) == SweepStep(STEP_WAIT, end + SWEEP_GRACE_S + 1)
    # and the call is not closed before end + grace has passed
    assert sweep_step(_esc(**{K_CALL_ENDED_AT: end}), None, end + SWEEP_GRACE_S).outcome == STEP_WAIT
    assert sweep_step(_esc(**{K_CALL_ENDED_AT: end}), None, end + SWEEP_GRACE_S + 1).outcome == STEP_CLOSE


@pytest.mark.parametrize("status", sorted(humanleg.ENDED_CALL_STATUSES))
def test_every_ended_status_with_an_end_time_is_closed_or_waits(status: str) -> None:
    assert sweep_step(_esc(), _ended(600, status), NOW).outcome == STEP_CLOSE
    assert sweep_step(_esc(), _ended(60, status), NOW) == SweepStep(STEP_WAIT, int(NOW) - 60 + SWEEP_GRACE_S + 1)
    assert sweep_step(_esc(), CallEnd(f" {status.upper()} ", NOW - 600), NOW).outcome == STEP_CLOSE


def test_an_ended_status_with_no_end_time_is_unknown() -> None:
    assert sweep_step(_esc(), CallEnd("completed", None), NOW) == SweepStep(STEP_UNKNOWN, int(NOW) + 600 - SLACK)


@pytest.mark.parametrize("age, back_off", [(0, 600), (600, 600), (1500, 750), (2400, 1200), (86_400, 1200)])
def test_a_live_call_backs_off_half_its_age_within_10_to_20_minutes(age: float, back_off: int) -> None:
    esc = _esc(**{K_ESCALATED_AT: _iso(NOW - age)})
    for status in sorted(humanleg.LIVE_CALL_STATUSES):
        assert sweep_step(esc, CallEnd(status, None), NOW) == SweepStep(STEP_LIVE, int(NOW) + back_off - SLACK)


def test_a_live_call_with_no_escalation_time_backs_off_the_minimum() -> None:
    esc = {"reason": "r"}
    assert sweep_step(esc, CallEnd("in-progress", None), NOW) == SweepStep(STEP_LIVE, int(NOW) + 600 - SLACK)


@pytest.mark.parametrize("answer", [None, CallEnd("initiated", None), CallEnd("", None), CallEnd("weird", NOW - 900)])
def test_no_answer_or_an_unrecognised_status_is_unknown_and_never_closed(answer: CallEnd | None) -> None:
    assert sweep_step(_esc(), answer, NOW) == SweepStep(STEP_UNKNOWN, int(NOW) + humanleg.SWEEP_RETRY_S - SLACK)


def test_a_recorded_end_wins_over_a_reported_live_status() -> None:
    esc = _esc(**{K_CALL_ENDED_AT: int(NOW) - 900})
    assert sweep_step(esc, CallEnd("in-progress", None), NOW).outcome == STEP_CLOSE


@pytest.mark.parametrize("now", [float("nan"), float("inf"), "now", None])
def test_a_bad_clock_does_nothing(now: Any) -> None:
    assert sweep_step(_esc(**{K_CALL_ENDED_AT: 1}), _ended(600), now).outcome == STEP_SKIP


def test_grace_is_five_minutes() -> None:
    assert SWEEP_GRACE_S == 300


# --------------------------------------------------------------------------- the due time

def test_due_time_rule() -> None:
    at = NOW - 3600
    assert sweep_due_at(_esc(**{K_SWEEP_NEXT_AT: 5, K_CALL_ENDED_AT: 7})) == 5
    assert sweep_due_at(_esc(**{K_CALL_ENDED_AT: 7})) == 7 + SWEEP_GRACE_S
    assert sweep_due_at(_esc()) == int(at) + humanleg.SWEEP_FIRST_LOOK_S
    assert sweep_due_at({"reason": "r"}) == 0
    assert sweep_due_at({"reason": "r", K_ESCALATED_AT: "yesterday"}) == 0
    assert sweep_due_at(None) == 0


def test_due_time_is_always_an_int_in_range() -> None:
    assert sweep_due_at({"r": 1, K_SWEEP_NEXT_AT: -50}) == 0
    assert sweep_due_at({"r": 1, K_SWEEP_NEXT_AT: 1e300}) == 253_402_300_799
    assert sweep_due_at({"r": 1, K_SWEEP_NEXT_AT: 10.2}) == 11
    assert sweep_due_at({"r": 1, K_SWEEP_NEXT_AT: float("nan"), K_CALL_ENDED_AT: True}) == 0


def test_writes_are_none_once_closed_or_unchanged() -> None:
    closed = _esc(**{K_CONVERSATION_CLOSED_AT: 1})
    assert humanleg.sweep_retry(closed, NOW) is None
    assert humanleg.sweep_retry({}, NOW) is None
    assert humanleg.sweep_reschedule(closed, CallEnd("in-progress", None), NOW) is None
    live = humanleg.sweep_reschedule(_esc(), CallEnd("in-progress", None), NOW)
    assert live is not None and live[K_SWEEP_NEXT_AT] == int(NOW) + 1200 - SLACK
    assert humanleg.K_SWEEP_RETRIES not in live  # a live reschedule is not a retry
    assert humanleg.sweep_reschedule(live, CallEnd("in-progress", None), NOW) is None  # unchanged
    assert humanleg.sweep_reschedule(_esc(), None, NOW) is None  # unknown: written by sweep_retry instead
    retried = humanleg.sweep_retry(_esc(), NOW)
    assert retried is not None and retried[K_SWEEP_NEXT_AT] == int(NOW) + humanleg.SWEEP_RETRY_S - SLACK
    assert retried[humanleg.K_SWEEP_RETRIES] == 1
    again = humanleg.sweep_retry(retried, NOW + 600)
    assert again is not None and again[humanleg.K_SWEEP_RETRIES] == 2
    # a useful answer resets the count of consecutive futile tries, even with the due time unchanged
    reset = humanleg.sweep_reschedule(again, CallEnd("in-progress", None), NOW)
    assert reset is not None and humanleg.K_SWEEP_RETRIES not in reset
    waited = humanleg.sweep_reschedule({**again, K_CALL_ENDED_AT: int(NOW) - 10}, None, NOW)
    assert waited is not None and humanleg.K_SWEEP_RETRIES not in waited
    assert humanleg.sweep_reschedule({**live, humanleg.K_SWEEP_RETRIES: 3}, CallEnd("queued", None), NOW) \
        == live


def test_recording_the_end_drops_the_sweeps_next_look() -> None:
    esc = _esc(**{K_SWEEP_NEXT_AT: int(NOW) + 1200})
    decision = humanleg.end_decision(esc, humanleg.CALL_ENDED, int(NOW))
    assert decision.escalation is not None
    assert K_SWEEP_NEXT_AT not in decision.escalation
    assert decision.escalation[K_CALL_ENDED_AT] == int(NOW)
    assert sweep_due_at(decision.escalation) == int(NOW) + SWEEP_GRACE_S
    # an end already recorded is kept, and so is a later sweep_next_at
    again = {**esc, K_CALL_ENDED_AT: int(NOW) - 10}
    later = humanleg.end_decision(again, humanleg.CALL_ENDED, int(NOW)).escalation
    assert later is None or later.get(K_SWEEP_NEXT_AT) == int(NOW) + 1200


# --------------------------------------------------------------------------- the index keys

def test_index_keys_are_derived_for_open_escalated_calls_only() -> None:
    base = CallState(call_sid=CALLS[0], expires_at=int(NOW) + 86_400, ai_sessions=1)
    assert callstate.SWEEP_OPEN_ATTR not in base.to_item()
    item = replace(base, escalation=_esc()).to_item()
    assert item[callstate.SWEEP_OPEN_ATTR] == callstate.SWEEP_OPEN
    assert item[callstate.SWEEP_DUE_ATTR] == int(NOW) - 3600 + 600
    closed = replace(base, escalation=_esc(**{K_CONVERSATION_CLOSED_AT: 1})).to_item()
    assert callstate.SWEEP_OPEN_ATTR not in closed and callstate.SWEEP_DUE_ATTR not in closed
    no_session = replace(base, ai_sessions=0, escalation=_esc()).to_item()
    assert callstate.SWEEP_OPEN_ATTR not in no_session  # no conversation possible
    with_id = replace(base, ai_sessions=0, escalation=_esc(**{K_CONVERSATION_ID: CONVS[0]})).to_item()
    assert with_id[callstate.SWEEP_OPEN_ATTR] == callstate.SWEEP_OPEN
    tried = replace(base, escalation=_esc(**{humanleg.K_SWEEP_RETRIES: 5})).to_item()
    assert tried[callstate.SWEEP_OPEN_ATTR] == callstate.SWEEP_OPEN
    spent = replace(base, escalation=_esc(**{humanleg.K_SWEEP_RETRIES: 6})).to_item()
    assert callstate.SWEEP_OPEN_ATTR not in spent and callstate.SWEEP_DUE_ATTR not in spent
    back = CallState.from_item(item)
    assert back == replace(base, escalation=_esc())
    assert callstate.SWEEP_OPEN_ATTR not in back.to_item().get("_extras", {})


def test_mutate_returning_none_writes_nothing() -> None:
    store = MemoryStore()
    callstate.create(store, CallState(call_sid=CALLS[0], expires_at=int(NOW) + 86_400, ai_sessions=1))
    before = store.get(callstate.TABLE, {"call_sid": CALLS[0]})
    got = callstate.mutate(store, CALLS[0], lambda s: None, now=lambda: NOW)
    assert got.version == 1
    assert store.get(callstate.TABLE, {"call_sid": CALLS[0]}) == before


# --------------------------------------------------------------------------- CallStatusReader

class _Resp:
    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self._raw = b"" if body is None else json.dumps(body).encode()

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._raw


class _Opener:
    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.requests: list[urllib.request.Request] = []

    def __call__(self, req: urllib.request.Request, timeout: float) -> Any:
        self.requests.append(req)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _reader(opener: _Opener, creds: tuple[str, str, str] = CREDS) -> sweep.CallStatusReader:
    return sweep.CallStatusReader(lambda: creds, opener=opener)


def test_status_request_and_answer() -> None:
    opener = _Opener(_Resp(200, {"sid": CALLS[0], "status": "completed",
                                 "end_time": "Thu, 05 Nov 2026 10:15:30 +0000"}))
    got = _reader(opener).status(CALLS[0])
    assert got == CallEnd("completed", NOW)
    req = opener.requests[0]
    assert req.get_method() == "GET"
    assert req.full_url == f"https://api.twilio.com/2010-04-01/Accounts/{ACCOUNT}/Calls/{CALLS[0]}.json"
    assert req.data is None
    assert req.get_header("Authorization", "").startswith("Basic ")


def test_live_call_has_no_end_time() -> None:
    opener = _Opener(_Resp(200, {"status": "in-progress", "end_time": None}))
    assert _reader(opener).status(CALLS[0]) == CallEnd("in-progress", None)


@pytest.mark.parametrize("raw", ["", "yesterday", "2026-11-05T10:15:30Z", "Thu, 05 Nov 2026 10:15:30", 12])
def test_unparseable_end_time_is_none(raw: Any) -> None:
    opener = _Opener(_Resp(200, {"status": "completed", "end_time": raw}))
    assert _reader(opener).status(CALLS[0]) == CallEnd("completed", None)


@pytest.mark.parametrize("answer", [
    urllib.error.HTTPError("u", 404, "not found", {}, None),  # type: ignore[arg-type]
    urllib.error.HTTPError("u", 500, "error", {}, None),  # type: ignore[arg-type]
    _Resp(200, {"no": "status"}),
    _Resp(200, None),
    TimeoutError("slow"),
    OSError("network"),
])
def test_failures_are_unknown_and_never_raise(answer: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    assert _reader(_Opener(answer)).status(CALLS[0]) is None
    assert SECRET not in caplog.text and KEY not in caplog.text


def test_bad_input_sends_nothing() -> None:
    opener = _Opener()
    assert _reader(opener).status("CA123") is None
    assert _reader(opener).status(None) is None  # type: ignore[arg-type]
    assert _reader(opener, ("not-an-account", KEY, SECRET)).status(CALLS[0]) is None
    assert opener.requests == []


def test_reader_refuses_a_non_https_base() -> None:
    with pytest.raises(ValueError, match="https"):
        sweep.CallStatusReader(lambda: CREDS, base_url="http://api.twilio.com")



# --------------------------------------------------------------------------- run_sweep over a store and the real close path

@dataclass
class World:
    """Orchestrator and the Calls API as the sweep sees them, with every request recorded."""

    statuses: dict[str, CallEnd | None] = field(default_factory=dict)
    close_results: list[bool] = field(default_factory=list)
    requests: list[tuple[str, ...]] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    closed: set[str] = field(default_factory=set)

    def resolve_conversation(self, call_sid: str) -> str | None:
        return None

    def add_human_participant(self, conversation_id: str, worker_sid: str, identity: str) -> bool:
        self.requests.append(("add", conversation_id, worker_sid))
        return True

    def mark_handoff(self, conversation_id: str, metadata: Any) -> bool:
        self.requests.append(("mark", conversation_id))
        return True

    def close(self, conversation_id: str) -> bool:
        self.requests.append(("close", conversation_id))
        ok = self.close_results.pop(0) if self.close_results else True
        if ok:
            self.closed.add(conversation_id)
        return ok

    def stop_transcription(self, call_sid: str, name: str) -> bool:
        self.requests.append(("stop", call_sid, name))
        return True

    def status(self, call_sid: str) -> CallEnd | None:
        self.asked.append(call_sid)
        return self.statuses.get(call_sid)


def _escalated(store: MemoryStore, sid: str, conv: str | None = None, *, ai_sessions: int = 1,
               **extra: Any) -> None:
    esc = _esc(**extra)
    if conv:
        esc[K_CONVERSATION_ID] = conv
    callstate.create(store, CallState(call_sid=sid, expires_at=int(NOW) + 86_400, ai_sessions=ai_sessions,
                                      escalation=esc))


def _deps(world: World, store: MemoryStore, now: float = NOW, recap: Any = None) -> sweep.SweepDeps:
    closer = HumanLegCloser(store, world, calls=world, clock=lambda: now, recap=recap)
    return sweep.SweepDeps(due=sweep.store_due(store), states=sweep.store_states(store, clock=lambda: now),
                           calls=world, closer=closer, write=sweep.store_writer(store, clock=lambda: now),
                           clock=lambda: now, recap=recap)


class _Recap:
    """A recording ``RecapTrigger``; ``fail`` raises after recording (a failed invoke)."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.fired: list[str] = []
        self.fail = fail

    def __call__(self, call_sid: str) -> None:
        self.fired.append(call_sid)
        if self.fail is not None:
            raise self.fail


def _esc_of(store: MemoryStore, sid: str) -> Mapping[str, Any]:
    state = callstate.load(store, sid)
    assert state is not None and state.escalation is not None
    return state.escalation


def _due_now(store: MemoryStore, now: float = NOW) -> list[str]:
    return list(sweep.store_due(store)(int(now)))


def test_a_recorded_end_is_closed_with_no_calls_api_read_and_leaves_the_index() -> None:
    world, store = World(), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0], **{K_CALL_ENDED_AT: int(NOW) - 600})
    report = sweep.run_sweep(_deps(world, store))
    assert report.as_dict() == {"due": 1, "skipped": 0, "statuses": 0, "closed": 1, "rescheduled": 0,
                                "unknown": 0, "out_of_time": False}
    assert world.asked == [] and world.requests == [("close", CONVS[0])]
    assert humanleg.conversation_closed(_esc_of(store, CALLS[0]))
    assert _due_now(store, NOW + 10**6) == []
    # running again sends nothing more
    assert sweep.run_sweep(_deps(world, store)).as_dict()["due"] == 0
    assert world.requests == [("close", CONVS[0])]


def test_an_ended_call_with_no_recorded_end_is_read_once_then_closed() -> None:
    world, store = World(statuses={CALLS[0]: _ended(600)}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0])
    report = sweep.run_sweep(_deps(world, store))
    assert report.statuses == 1 and report.closed == 1 and world.asked == [CALLS[0]]


def test_a_live_call_is_rescheduled_never_closed() -> None:
    world, store = World(statuses={CALLS[0]: CallEnd("in-progress", None)}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0])
    report = sweep.run_sweep(_deps(world, store))
    assert report.rescheduled == 1 and report.closed == 0 and world.requests == []
    assert _esc_of(store, CALLS[0])[K_SWEEP_NEXT_AT] == int(NOW) + 1200 - SLACK
    assert humanleg.K_SWEEP_RETRIES not in _esc_of(store, CALLS[0])
    assert _due_now(store) == []
    assert _due_now(store, NOW + 1200 - SLACK) == [CALLS[0]]
    # many hours later it is still live: still never closed
    later = NOW + 6 * 3600
    assert sweep.run_sweep(_deps(world, store, later)).rescheduled == 1
    assert world.requests == [] and _esc_of(store, CALLS[0])[K_SWEEP_NEXT_AT] == int(later) + 1200 - SLACK


def test_a_call_ended_within_the_grace_period_comes_back_just_after_it() -> None:
    world, store = World(statuses={CALLS[0]: _ended(60)}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0])
    assert sweep.run_sweep(_deps(world, store)).rescheduled == 1
    nxt = _esc_of(store, CALLS[0])[K_SWEEP_NEXT_AT]
    assert nxt == int(NOW) - 60 + SWEEP_GRACE_S + 1
    assert sweep.run_sweep(_deps(world, store, nxt)).closed == 1


@pytest.mark.parametrize("answer", [None, CallEnd("completed", None), CallEnd("initiated", None)])
def test_an_unknown_call_backs_off_ten_minutes_and_is_never_closed(answer: CallEnd | None) -> None:
    """A call that stays unknown (404, an ended status with no end time, ``initiated``) goes to the
    back of the queue each run instead of holding the front."""
    world, store = World(statuses={CALLS[0]: answer}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0])
    report = sweep.run_sweep(_deps(world, store))
    assert report.unknown == 1 and report.rescheduled == 0 and report.closed == 0 and world.requests == []
    assert _esc_of(store, CALLS[0])[K_SWEEP_NEXT_AT] == int(NOW) + 600 - SLACK
    assert _esc_of(store, CALLS[0])[humanleg.K_SWEEP_RETRIES] == 1
    assert _due_now(store) == []
    assert sweep.run_sweep(_deps(world, store)).due == 0
    later = NOW + 6 * 3600  # still unknown hours later: still never closed
    assert sweep.run_sweep(_deps(world, store, later)).unknown == 1
    assert world.requests == [] and _esc_of(store, CALLS[0])[K_SWEEP_NEXT_AT] == int(later) + 600 - SLACK


def test_a_failed_unknown_back_off_write_leaves_the_call_due(caplog: pytest.LogCaptureFixture) -> None:
    world, store = World(statuses={CALLS[0]: None}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0])

    def broken(call_sid: str, change: Any) -> bool:
        raise RuntimeError("conditional update failed")

    report = sweep.run_sweep(replace(_deps(world, store), write=broken))
    assert report.unknown == 1 and report.rescheduled == 0
    assert "conversation_sweep_reschedule_failed" in caplog.text
    assert _due_now(store) == [CALLS[0]]


def test_the_unknown_back_off_writes_nothing_once_closed() -> None:
    """The conversation closed between the batch read and the back-off write."""
    world, store = World(statuses={CALLS[0]: None}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0])
    deps = _deps(world, store)
    real_states = deps.states

    def then_close(call_sids: Sequence[str]) -> Mapping[str, CallState]:
        got = real_states(call_sids)
        callstate.mutate(store, CALLS[0], lambda s: {"escalation": {**(s.escalation or {}),
                                                                    K_CONVERSATION_CLOSED_AT: int(NOW)}})
        return got

    version = callstate.load(store, CALLS[0]).version  # type: ignore[union-attr]
    assert sweep.run_sweep(replace(deps, states=then_close)).unknown == 1
    state = callstate.load(store, CALLS[0])
    assert state is not None and state.version == version + 1
    assert K_SWEEP_NEXT_AT not in (state.escalation or {})


def test_a_failed_close_is_tried_again_ten_minutes_later() -> None:
    world, store = World(close_results=[False]), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0], **{K_CALL_ENDED_AT: int(NOW) - 600})
    report = sweep.run_sweep(_deps(world, store))
    assert report.closed == 0 and report.rescheduled == 1
    esc = _esc_of(store, CALLS[0])
    assert not humanleg.conversation_closed(esc) and esc[K_SWEEP_NEXT_AT] == int(NOW) + 600 - SLACK
    assert esc[humanleg.K_SWEEP_RETRIES] == 1
    assert sweep.run_sweep(_deps(world, store)).due == 0
    assert sweep.run_sweep(_deps(world, store, NOW + 600)).closed == 1
    assert world.requests == [("close", CONVS[0]), ("close", CONVS[0])]


def test_a_call_whose_close_always_fails_leaves_the_index_after_six_retries() -> None:
    """A futile call costs at most SWEEP_MAX_RETRIES tries, then leaves the index for good
    (Orchestrator's timeout is the backstop)."""
    world, store = World(close_results=[False] * 50), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0], **{K_CALL_ENDED_AT: int(NOW) - 600})
    for run in range(20):
        sweep.run_sweep(_deps(world, store, NOW + 600 * run))
    assert world.requests == [("close", CONVS[0])] * humanleg.SWEEP_MAX_RETRIES
    esc = _esc_of(store, CALLS[0])
    assert esc[humanleg.K_SWEEP_RETRIES] == humanleg.SWEEP_MAX_RETRIES and not humanleg.conversation_closed(esc)
    item = store.get(callstate.TABLE, {"call_sid": CALLS[0]})
    assert item is not None and callstate.SWEEP_OPEN_ATTR not in item and callstate.SWEEP_DUE_ATTR not in item
    assert _due_now(store, NOW + 10**6) == []


@pytest.mark.parametrize("ended", [False, True])
def test_giving_up_on_a_call_starts_the_recap_once_and_never_again(ended: bool,
                                                                   caplog: pytest.LogCaptureFixture) -> None:
    """The retry that takes the call out of the index (unknown
    status, or a close never done) starts the recap writer once, source ``sweep_gave_up``; later runs,
    and a second give-up after an end brought the call back, start nothing more."""
    caplog.set_level(logging.INFO)
    recap = _Recap()
    world, store = World(statuses={CALLS[0]: None}, close_results=[False] * 50), MemoryStore()
    extra = {K_CALL_ENDED_AT: int(NOW) - 600} if ended else {}
    _escalated(store, CALLS[0], CONVS[0], **extra)
    t = NOW
    for run in range(humanleg.SWEEP_MAX_RETRIES):
        assert recap.fired == []  # not before the last retry
        sweep.run_sweep(_deps(world, store, t, recap))
        t += 600
    assert recap.fired == [CALLS[0]]
    esc = _esc_of(store, CALLS[0])
    assert esc[humanleg.K_SWEEP_RETRIES] == humanleg.SWEEP_MAX_RETRIES
    assert esc[humanleg.K_SWEEP_GAVE_UP_AT] == int(t - 600)
    line = [r for r in caplog.records if r.getMessage() == "conversation_sweep_gave_up"][-1]
    assert getattr(line, "call_sid", None) == CALLS[0]
    for _ in range(5):
        sweep.run_sweep(_deps(world, store, t, recap))
        t += 600
    assert recap.fired == [CALLS[0]]
    # An end (not recorded before) brings it back with fresh retries; giving up again starts nothing.
    end = int(t)

    def ended_now(state: CallState) -> Mapping[str, Any]:
        new = {**(state.escalation or {}), K_CALL_ENDED_AT: end}
        new.pop(humanleg.K_SWEEP_RETRIES, None)
        new.pop(K_SWEEP_NEXT_AT, None)
        return {"escalation": new}
    callstate.mutate(store, CALLS[0], ended_now, now=lambda: end)
    assert _due_now(store, end + 10**6) == [CALLS[0]]
    t = end + 400
    for _ in range(humanleg.SWEEP_MAX_RETRIES + 2):
        sweep.run_sweep(_deps(world, store, t, recap))
        t += 600
    assert _due_now(store, t + 10**6) == []
    assert recap.fired == [CALLS[0]]


def test_a_failing_recap_on_giving_up_changes_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """For the give-up path: a failed or slow invoke is logged and the sweep's writes and counts are
    the same."""
    results = []
    for recap in (_Recap(), _Recap(fail=TimeoutError("read timeout"))):
        world, store = World(statuses={CALLS[0]: None}), MemoryStore()
        _escalated(store, CALLS[0], CONVS[0])
        reports = [sweep.run_sweep(_deps(world, store, NOW + 600 * run, recap))
                   for run in range(humanleg.SWEEP_MAX_RETRIES + 1)]
        results.append((reports, dict(_esc_of(store, CALLS[0])), recap.fired))
    assert results[0] == results[1]
    line = [r for r in caplog.records if r.getMessage() == "recap_trigger_failed"][-1]
    assert getattr(line, "source", None) == "sweep_gave_up"


def test_a_failed_give_up_write_starts_no_recap() -> None:
    recap = _Recap()
    world, store = World(statuses={CALLS[0]: None}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0], **{humanleg.K_SWEEP_RETRIES: humanleg.SWEEP_MAX_RETRIES - 1})

    def broken(call_sid: str, change: Any) -> bool:
        raise RuntimeError("conditional update failed")

    sweep.run_sweep(replace(_deps(world, store, recap=recap), write=broken))
    assert recap.fired == [] and _due_now(store) == [CALLS[0]]
    sweep.run_sweep(_deps(world, store, recap=recap))
    assert recap.fired == [CALLS[0]]


def test_sweep_gave_up_is_the_first_give_up_only() -> None:
    esc = {**_esc(), humanleg.K_SWEEP_RETRIES: humanleg.SWEEP_MAX_RETRIES - 2}
    once = humanleg.sweep_retry(esc, NOW)
    assert once is not None and humanleg.K_SWEEP_GAVE_UP_AT not in once
    assert not humanleg.sweep_gave_up(esc, once)
    last = humanleg.sweep_retry(once, NOW + 600)
    assert last is not None and last[humanleg.K_SWEEP_GAVE_UP_AT] == int(NOW + 600)
    assert humanleg.sweep_gave_up(once, last)
    again = humanleg.sweep_retry({**last, humanleg.K_SWEEP_RETRIES: humanleg.SWEEP_MAX_RETRIES - 1}, NOW + 9000)
    assert again is not None and again[humanleg.K_SWEEP_GAVE_UP_AT] == int(NOW + 600)
    assert not humanleg.sweep_gave_up(last, again) and not humanleg.sweep_gave_up(esc, None)


def test_a_call_that_stays_unknown_is_read_six_times_at_most() -> None:
    world, store = World(statuses={CALLS[0]: None}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0])
    for run in range(20):
        sweep.run_sweep(_deps(world, store, NOW + 600 * run))
    assert world.asked == [CALLS[0]] * humanleg.SWEEP_MAX_RETRIES and world.requests == []


@pytest.mark.parametrize("recorded_end", [True, False])
def test_a_call_escalated_before_any_ai_session_never_enters_the_index(recorded_end: bool) -> None:
    """``ai_sessions == 0`` and no recorded conversation id (VERIFICATION_RISK, a caller who asked
    for a human straight away): there is no conversation to close, so the sweep never sees it."""
    world, store = World(statuses={CALLS[0]: _ended(900)}), MemoryStore()
    extra = {K_CALL_ENDED_AT: int(NOW) - 900} if recorded_end else {}
    _escalated(store, CALLS[0], ai_sessions=0, **extra)
    item = store.get(callstate.TABLE, {"call_sid": CALLS[0]})
    assert item is not None and callstate.SWEEP_OPEN_ATTR not in item
    for run in range(5):
        report = sweep.run_sweep(_deps(world, store, NOW + 600 * run))
        assert report.due == 0
    assert world.asked == [] and world.requests == []


def test_a_call_with_no_ai_session_but_a_recorded_conversation_is_swept() -> None:
    world, store = World(), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0], ai_sessions=0, **{K_CALL_ENDED_AT: int(NOW) - 900})
    assert sweep.run_sweep(_deps(world, store)).closed == 1
    assert world.requests == [("close", CONVS[0])]


def test_a_live_call_is_due_for_a_run_that_starts_a_little_early() -> None:
    """Rescheduled by a run that started at t, the call is due for the run at t + 600 - 30
    (the schedule's jitter), not pushed to the run after."""
    world, store = World(statuses={CALLS[0]: CallEnd("queued", None)}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0], **{K_ESCALATED_AT: _iso(NOW), K_SWEEP_NEXT_AT: int(NOW)})  # min back-off
    assert sweep.run_sweep(_deps(world, store)).rescheduled == 1
    assert _due_now(store, NOW + 600 - 30) == [CALLS[0]]
    assert sweep.run_sweep(_deps(world, store, NOW + 600 - 30)).statuses == 1


def test_a_live_reschedule_resets_the_count() -> None:
    world, store = World(statuses={CALLS[0]: CallEnd("in-progress", None)}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0], **{humanleg.K_SWEEP_RETRIES: 4, K_SWEEP_NEXT_AT: int(NOW)})
    assert sweep.run_sweep(_deps(world, store)).rescheduled == 1
    assert humanleg.K_SWEEP_RETRIES not in _esc_of(store, CALLS[0])
    for run in range(1, 12):
        sweep.run_sweep(_deps(world, store, NOW + 1200 * run))
    assert len(world.asked) == 12 and humanleg.K_SWEEP_RETRIES not in _esc_of(store, CALLS[0])


def test_unknowns_around_a_live_answer_keep_the_call_in_the_index() -> None:
    """a call alternating between a flaky Calls API and live stays in the index while
    the caller waits. 5 unknowns, a live answer, 5 more unknowns: still in, still swept."""
    world, store = World(), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0], **{K_SWEEP_NEXT_AT: int(NOW)})
    answers = [None] * 5 + [CallEnd("in-progress", None)] + [None] * 5
    t = NOW
    for answer in answers:
        world.statuses[CALLS[0]] = answer
        assert sweep.run_sweep(_deps(world, store, t)).statuses == 1
        t = _esc_of(store, CALLS[0])[K_SWEEP_NEXT_AT]
    assert len(world.asked) == 11 and world.requests == []
    assert _esc_of(store, CALLS[0])[humanleg.K_SWEEP_RETRIES] == 5
    assert _due_now(store, t) == [CALLS[0]]
    world.statuses[CALLS[0]] = None  # a sixth unknown in a row: out for good
    sweep.run_sweep(_deps(world, store, t))
    assert _due_now(store, t + 10**6) == []


def test_an_end_after_running_out_of_retries_brings_the_call_back() -> None:
    """A live call that used up its retries leaves the index; its end
    (with the transcription-stopped callback lost) puts it back with fresh close attempts."""
    world, store = World(), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0], **{K_SWEEP_NEXT_AT: int(NOW)})
    t = NOW
    for _ in range(humanleg.SWEEP_MAX_RETRIES):
        sweep.run_sweep(_deps(world, store, t))
        t = _esc_of(store, CALLS[0]).get(K_SWEEP_NEXT_AT, t)
    assert _due_now(store, t + 10**6) == []
    end = int(t) + 60

    def ended(state: CallState) -> Mapping[str, Any]:
        return {"escalation": humanleg.end_decision(state.escalation, humanleg.CALL_ENDED, end).escalation}
    callstate.mutate(store, CALLS[0], ended, now=lambda: end)
    assert humanleg.K_SWEEP_RETRIES not in _esc_of(store, CALLS[0])
    assert _due_now(store, end + humanleg.SWEEP_GRACE_S) == [CALLS[0]]
    world.statuses[CALLS[0]] = None
    assert sweep.run_sweep(_deps(world, store, end + 400)).closed == 1
    assert world.requests == [("close", CONVS[0])]


def test_a_retry_after_an_end_landed_meanwhile_writes_nothing() -> None:
    """An end recorded between the sweep's read and its write is not a
    futile try; the call stays due and the next run closes it on time."""
    seen = _esc()
    ended = {**seen, K_CALL_ENDED_AT: int(NOW)}
    assert humanleg.sweep_retry(ended, NOW, seen_ended_at=None) is None
    assert humanleg.sweep_retry(seen, NOW, seen_ended_at=None) is not None
    retried = humanleg.sweep_retry(ended, NOW, seen_ended_at=int(NOW))  # a close not done on that end
    assert retried is not None and retried[humanleg.K_SWEEP_RETRIES] == 1


def test_a_running_transcription_is_stopped_before_the_close() -> None:
    world, store = World(statuses={CALLS[0]: _ended(600)}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0], **{humanleg.K_TRANSCRIPTIONS: {CALLS[0]: {
        "name": humanleg.TRANSCRIPTION_NAME, "role": humanleg.ROLE_CUSTOMER, "started_at": 1, "ended_at": None}}})
    assert sweep.run_sweep(_deps(world, store)).closed == 1
    assert world.requests == [("stop", CALLS[0], humanleg.TRANSCRIPTION_NAME), ("close", CONVS[0])]


def test_the_sweep_stops_every_running_leg_before_the_close() -> None:
    """An agent leg whose stop never arrived can't hold the conversation open beyond the
    sweep; legs already ended are not stopped again."""
    agent_a, agent_b = "CA" + "1" * 32, "CA" + "2" * 32
    world, store = World(statuses={CALLS[0]: _ended(600)}), MemoryStore()

    def entry(role: str, ended_at: int | None) -> dict[str, Any]:
        return {"name": humanleg.TRANSCRIPTION_NAME, "role": role, "started_at": 1, "ended_at": ended_at}

    _escalated(store, CALLS[0], CONVS[0], **{humanleg.K_TRANSCRIPTIONS: {
        CALLS[0]: entry(humanleg.ROLE_CUSTOMER, None), agent_a: entry(humanleg.ROLE_AGENT, 5),
        agent_b: entry(humanleg.ROLE_AGENT, None)}})
    assert sweep.run_sweep(_deps(world, store)).closed == 1
    assert world.requests == [("stop", CALLS[0], humanleg.TRANSCRIPTION_NAME),
                              ("stop", agent_b, humanleg.TRANSCRIPTION_NAME), ("close", CONVS[0])]


def test_ai_only_calls_are_not_in_the_index() -> None:
    """The bridge closes AI-only conversations; the sweep never even sees them."""
    world, store = World(), MemoryStore()
    callstate.create(store, CallState(call_sid=CALLS[0], expires_at=int(NOW) + 86_400, ai_sessions=1))
    assert sweep.run_sweep(_deps(world, store)).as_dict()["due"] == 0
    assert world.asked == []


def test_stale_index_entries_are_skipped_before_any_status_read() -> None:
    """The index is eventually consistent: an entry may point at a call closed, expired, gone or not
    due any more on the strongly read state; none costs a Calls API read or a write."""
    world, store = World(statuses={s: _ended(600) for s in CALLS}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0], **{K_CONVERSATION_CLOSED_AT: int(NOW) - 60})
    callstate.create(store, CallState(call_sid=CALLS[2], expires_at=int(NOW) - 10, ai_sessions=1,
                                      escalation=_esc()))
    _escalated(store, CALLS[3], CONVS[3], **{K_SWEEP_NEXT_AT: int(NOW) + 600})
    callstate.create(store, CallState(call_sid=CALLS[4], expires_at=int(NOW) + 86_400, ai_sessions=1))
    _escalated(store, CALLS[5], CONVS[5])
    before = {s: store.get(callstate.TABLE, {"call_sid": s}) for s in CALLS}
    deps = replace(_deps(world, store), due=lambda now: list(CALLS))
    report = sweep.run_sweep(deps)
    assert report.as_dict() == {"due": 6, "skipped": 5, "statuses": 1, "closed": 1, "rescheduled": 0,
                                "unknown": 0, "out_of_time": False}
    assert world.asked == [CALLS[5]] and world.requests == [("close", CONVS[5])]
    for s in CALLS[:5]:
        assert store.get(callstate.TABLE, {"call_sid": s}) == before[s]


def test_a_failed_query_fails_the_run(caplog: pytest.LogCaptureFixture) -> None:
    world, store = World(), MemoryStore()

    def boom(now: int) -> Sequence[str]:
        raise RuntimeError("dynamodb down")

    with caplog.at_level(logging.INFO), pytest.raises(sweep.SweepFailed):
        sweep.run_sweep(replace(_deps(world, store), due=boom))
    failed = [r for r in caplog.records if r.getMessage() == "conversation_sweep_query_failed"]
    assert failed and failed[0].levelno == logging.ERROR
    assert world.asked == [] and world.requests == []


def test_a_failed_state_read_leaves_its_chunk_for_the_next_run(caplog: pytest.LogCaptureFixture) -> None:
    world, store = World(statuses={CALLS[0]: _ended(600)}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0])

    def boom(call_sids: Any) -> Any:
        raise RuntimeError("dynamodb down")

    report = sweep.run_sweep(replace(_deps(world, store), states=boom))
    assert report.closed == 0 and report.unknown == 1 and world.requests == [] and world.asked == []
    assert "conversation_sweep_state_failed" in caplog.text
    assert _due_now(store) == [CALLS[0]]


def test_a_failed_reschedule_leaves_the_call_due(caplog: pytest.LogCaptureFixture) -> None:
    world, store = World(statuses={CALLS[0]: CallEnd("ringing", None)}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0])

    def broken(call_sid: str, change: Any) -> bool:
        raise RuntimeError("conditional update failed")

    report = sweep.run_sweep(replace(_deps(world, store), write=broken))
    assert report.unknown == 1 and report.rescheduled == 0
    assert "conversation_sweep_reschedule_failed" in caplog.text
    assert _due_now(store) == [CALLS[0]]


def test_the_write_re_decides_on_the_fresh_state_and_writes_nothing_once_closed() -> None:
    """Between the batch read and the write the conversation closed: nothing is written."""
    world, store = World(statuses={CALLS[0]: CallEnd("in-progress", None)}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0])
    deps = _deps(world, store)
    real_states = deps.states

    def then_close(call_sids: Sequence[str]) -> Mapping[str, CallState]:
        got = real_states(call_sids)
        callstate.mutate(store, CALLS[0], lambda s: {"escalation": {**(s.escalation or {}),
                                                                    K_CONVERSATION_CLOSED_AT: int(NOW)}})
        return got

    before_version = callstate.load(store, CALLS[0]).version  # type: ignore[union-attr]
    report = sweep.run_sweep(replace(deps, states=then_close))
    assert report.rescheduled == 0 and report.unknown == 1
    state = callstate.load(store, CALLS[0])
    assert state is not None and state.version == before_version + 1  # only the close, no sweep write
    assert K_SWEEP_NEXT_AT not in (state.escalation or {})


def test_stops_starting_work_when_time_runs_short_and_the_rest_keep_their_due_time() -> None:
    world, store = World(statuses={s: CallEnd("in-progress", None) for s in CALLS}), MemoryStore()
    for i, s in enumerate(CALLS):
        _escalated(store, s, CONVS[i], **{K_SWEEP_NEXT_AT: int(NOW) - 100 + i})
    budget = iter([100.0, 100.0, 100.0, 5.0])
    report = sweep.run_sweep(_deps(world, store), time_left=lambda: next(budget, 5.0))
    assert report.out_of_time and report.statuses == 2 and report.rescheduled == 2
    assert world.asked == [CALLS[0], CALLS[1]]
    assert _due_now(store) == list(CALLS[2:])  # oldest first, untouched
    assert [_esc_of(store, s)[K_SWEEP_NEXT_AT] for s in CALLS[2:]] == [int(NOW) - 100 + i for i in range(2, 6)]


def test_summary_line_has_counts_only(caplog: pytest.LogCaptureFixture) -> None:
    world, store = World(statuses={CALLS[1]: CallEnd("queued", None)}), MemoryStore()
    _escalated(store, CALLS[0], CONVS[0], **{K_CALL_ENDED_AT: int(NOW) - 600})
    _escalated(store, CALLS[1], CONVS[1])
    caplog.set_level(logging.DEBUG)
    sweep.run_sweep(_deps(world, store))
    summary = [r for r in caplog.records if r.getMessage() == "conversation_sweep"]
    assert len(summary) == 1
    fields = {k: getattr(summary[0], k) for k in ("decision", "due", "skipped", "statuses", "closed",
                                                     "rescheduled", "unknown", "out_of_time")}
    assert fields == {"decision": "done", "due": 2, "skipped": 0, "statuses": 1, "closed": 1,
                      "rescheduled": 1, "unknown": 0, "out_of_time": False}
    for r in caplog.records:
        if r.name.startswith("lambdas.conversation_sweep"):
            text = json.dumps({k: str(v) for k, v in vars(r).items()})
            assert "https://" not in text and SECRET not in text and KEY not in text


def test_scale_hundreds_live_and_many_ai_only_calls() -> None:
    """5,000 AI-only calls, 500 live escalated calls and 50 ended ones (end recorded by
    ``/voice/status``): only the 550 escalated calls are due; the 50 ended ones close with no Calls API
    read; the 500 live ones cost one read each and are rescheduled 10 to 20 minutes on."""
    store = MemoryStore()
    live = [f"CA{i:032x}" for i in range(500)]
    ended = [f"CA{i:032x}" for i in range(500, 550)]
    ai_only = [f"CA{i:032x}" for i in range(550, 5550)]
    world = World(statuses={s: CallEnd("in-progress", None) for s in live})
    for s in live:
        _escalated(store, s, "conv_" + s[2:28])
    for s in ended:
        _escalated(store, s, "conv_" + s[2:28], **{K_CALL_ENDED_AT: int(NOW) - 900})
    for s in ai_only:
        callstate.create(store, CallState(call_sid=s, expires_at=int(NOW) + 86_400, ai_sessions=1))
    report = sweep.run_sweep(_deps(world, store))
    assert report.as_dict() == {"due": 550, "skipped": 0, "statuses": 500, "closed": 50, "rescheduled": 500,
                                "unknown": 0, "out_of_time": False}
    assert sorted(world.asked) == sorted(live)
    assert sorted(c for (kind, c, *_) in world.requests if kind == "close") == sorted("conv_" + s[2:28] for s in ended)
    assert _due_now(store, NOW + 599) == []
    assert sorted(_due_now(store, NOW + 1200)) == sorted(live)


# --------------------------------------------------------------------------- wiring

def test_build_deps_needs_the_api_key_secret() -> None:
    with pytest.raises(ValueError, match=sweep.API_KEY_SECRET_ENV):
        sweep.build_deps({}, store=MemoryStore())


def test_build_deps_reads_no_secret_and_works_from_the_store() -> None:
    store = MemoryStore()
    deps = sweep.build_deps({sweep.API_KEY_SECRET_ENV: "arn:aws:secretsmanager:us-east-1:1:secret:x"}, store=store)
    assert isinstance(deps.closer, HumanLegCloser)
    assert isinstance(deps.calls, sweep.CallStatusReader)
    assert list(deps.due(int(NOW))) == []
    assert not hasattr(sweep, "CONFIGURATION_ENV")


def test_handler_returns_the_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep, "_deps", _deps(World(), MemoryStore()))
    assert sweep.handler({}, None) == {"due": 0, "skipped": 0, "statuses": 0, "closed": 0, "rescheduled": 0,
                                       "unknown": 0, "out_of_time": False}


def test_one_bad_call_state_item_skips_only_that_call(caplog: pytest.LogCaptureFixture) -> None:
    """An item that does not parse leaves only its own call unknown."""
    good = CallState(call_sid=CALLS[0], expires_at=int(NOW) + 86_400, ai_sessions=1,
                     escalation={"reason": "r"}).to_item()
    other = CallState(call_sid=CALLS[2], expires_at=int(NOW) + 86_400, ai_sessions=1).to_item()

    class Store:
        def batch_get(self, table: str, keys: Any) -> list[Any]:
            return [good, "junk", {"call_sid": CALLS[1], "expires_at": "soon"}, other]

    found = sweep.store_states(Store(), clock=lambda: NOW)([CALLS[0], CALLS[1], CALLS[2]])
    assert set(found) == {CALLS[0], CALLS[2]}
    unreadable = [r for r in caplog.records if r.getMessage() == "conversation_sweep_state_unreadable"]
    assert len(unreadable) == 2
    assert [getattr(r, "call_sid", None) for r in unreadable] == [None, CALLS[1]]
