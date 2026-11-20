"""Memory-writes: the front door and the human-leg close start the call recap writer. AI-only calls at call end; escalated calls with a conversation
after the close (the front door's or the sweep's), never at status; a failing trigger changes nothing in
the response."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lambdas.frontdoor.app import NoopConversationCloser
from owlbank import callstate, humanleg
from owlbank.humanleg_close import HumanLegCloser

from tests.component.test_frontdoor_humanleg import (  # noqa: F401  (fixtures)
    CONV,
    TASK,
    FakeCalls,
    FakeOrchestrator,
    FakeReservations,
    closes,
    customer_leg,
    escalate,
    fdh,
    hl,
    human_leg,
    ops,
    transcription,
)
from tests.component.test_frontdoor_support import CLOCK, fd, sid  # noqa: F401  (fd is a fixture)


class RecordingRecap:
    """A ``RecapTrigger``: records each CallSid; ``fail`` raises after recording (a failed invoke)."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.fired: list[str] = []
        self.fail = fail

    def __call__(self, call_sid: str) -> None:
        self.fired.append(call_sid)
        if self.fail is not None:
            raise self.fail


@pytest.fixture
def recap(fd) -> RecordingRecap:
    r = RecordingRecap()
    fd.app.d = replace(fd.app.d, recap=r)
    return r


def with_closer(h, ops: list, recap: RecordingRecap, **orchestrator) -> HumanLegCloser:
    closer = HumanLegCloser(h.store, FakeOrchestrator(ops, **orchestrator), calls=FakeCalls(ops),
                            reservations=FakeReservations(ops, []), clock=lambda: CLOCK,
                            journal=h.app.journal, recap=recap)
    h.app.d = replace(h.app.d, closer=closer)
    return closer


def ai_only(h, call_sid: str) -> None:
    h.login_thomas(call_sid)
    callstate.mutate(h.store, call_sid, lambda s: {"ai_sessions": 1})


@pytest.mark.parametrize("call_status", ["completed", "failed"])
def test_ai_only_call_end_triggers_once(fd, recap, call_status):
    ai_only(fd, sid(1))
    fd.post("status", sid(1), CallStatus=call_status)
    fd.post("status", sid(1), CallStatus=call_status)  # Twilio's retry
    assert recap.fired == [sid(1)]


def test_ai_only_call_with_the_real_closer_triggers_once_from_status(fdh, ops, recap):
    ai_only(fdh, sid(1))
    with_closer(fdh, ops, recap)
    fdh.post("status", sid(1), CallStatus="completed")
    assert recap.fired == [sid(1)] and ops == []


def test_non_final_or_unanswered_statuses_never_trigger(fd, recap):
    ai_only(fd, sid(1))
    for status in ("in-progress", "busy", "no-answer", "canceled"):
        fd.post("status", sid(1), CallStatus=status)
    assert recap.fired == []


def test_a_call_with_no_state_never_triggers(fd, recap):
    fd.post("status", sid(9), CallStatus="completed")
    assert recap.fired == []


def test_escalated_call_waits_for_the_close_then_triggers_once(fdh, ops, recap):
    escalate(fdh, sid(1))
    human_leg(fdh, sid(1), transcriptions=customer_leg(sid(1)), task_sid=TASK)
    with_closer(fdh, ops, recap)
    fdh.post("status", sid(1), CallStatus="completed")
    assert closes(ops) == [] and recap.fired == []  # the transcript is still running
    transcription(fdh, sid(1), "transcription-stopped")
    assert closes(ops) == [("close", CONV)] and recap.fired == [sid(1)]
    transcription(fdh, sid(1), "transcription-stopped")  # retries after the close
    fdh.post("status", sid(1), CallStatus="completed")
    assert recap.fired == [sid(1)]


def test_escalated_call_closed_at_status_triggers_once_from_the_close(fdh, ops, recap):
    escalate(fdh, sid(1))
    with_closer(fdh, ops, recap)
    fdh.post("status", sid(1), CallStatus="completed")
    assert closes(ops) == [("close", CONV)]
    assert recap.fired == [sid(1)]  # from the close, not a second one from status


def test_a_failed_close_triggers_nothing_and_the_sweep_close_triggers_once(fdh, ops, recap):
    escalate(fdh, sid(1))
    closer = with_closer(fdh, ops, recap, close_ok=False)
    fdh.post("status", sid(1), CallStatus="completed")
    assert recap.fired == []
    closer._client.close_ok = True  # noqa: SLF001
    assert closer.end(sid(1), humanleg.SWEEP) is True
    assert recap.fired == [sid(1)]
    assert closer.end(sid(1), humanleg.SWEEP) is False  # a repeat finds it closed
    assert recap.fired == [sid(1)]


def test_sweep_close_triggers_once(fdh, ops, recap):
    escalate(fdh, sid(1))
    human_leg(fdh, sid(1), transcriptions=customer_leg(sid(1)))
    closer = with_closer(fdh, ops, recap)
    fdh.post("status", sid(1), CallStatus="completed")  # waits for the transcript
    assert recap.fired == []
    assert closer.end(sid(1), humanleg.SWEEP) is True
    assert recap.fired == [sid(1)]


def test_escalated_without_a_conversation_triggers_at_status(fdh, ops, recap):
    escalate(fdh, sid(1), ai_sessions=0)
    st = fdh.state(sid(1))
    assert st.escalation and not st.escalation.get(humanleg.K_CONVERSATION_ID)
    assert fdh.session(sid(1)).conversation_id is None
    with_closer(fdh, ops, recap)
    fdh.post("status", sid(1), CallStatus="completed")
    assert recap.fired == [sid(1)]
    assert closes(ops) == []  # the closer found no conversation


def test_escalated_with_the_noop_closer_triggers_at_status(fdh, recap):
    escalate(fdh, sid(1))
    fdh.app.d = replace(fdh.app.d, closer=NoopConversationCloser())
    fdh.post("status", sid(1), CallStatus="completed")
    assert recap.fired == [sid(1)]


@pytest.mark.parametrize("failure", [RuntimeError("invoke refused"), TimeoutError("read timeout")])
def test_a_failing_trigger_never_changes_the_status_response(fd, caplog, failure):
    ai_only(fd, sid(1))
    ai_only(fd, sid(2))
    ok = fd.post("status", sid(1), CallStatus="completed")
    fd.app.d = replace(fd.app.d, recap=RecordingRecap(fail=failure))
    bad = fd.post("status", sid(2), CallStatus="completed")
    assert (bad["statusCode"], bad["body"]) == (ok["statusCode"], ok["body"])
    assert fd.session(sid(2)).ended
    line = [r for r in caplog.records if r.getMessage() == "recap_trigger_failed"][-1]
    assert getattr(line, "call_sid", None) == sid(2) and getattr(line, "source", None) == "status"


def test_a_failing_trigger_after_the_close_keeps_the_close(fdh, ops, caplog):
    escalate(fdh, sid(1))
    failing = RecordingRecap(fail=RuntimeError("invoke refused"))
    with_closer(fdh, ops, failing)
    resp = fdh.post("status", sid(1), CallStatus="completed")
    assert resp["statusCode"] == 200 and "<Response />" in resp["body"]
    assert failing.fired == [sid(1)]
    assert fdh.state(sid(1)).escalation[humanleg.K_CONVERSATION_CLOSED_AT] == int(CLOCK)
    assert [r for r in caplog.records if r.getMessage() == "recap_trigger_failed"]


def _record_fails(monkeypatch) -> None:
    """Make the close's record write (the one that sets ``conversation_closed_at``) fail; every other
    call-state write goes through."""
    real = callstate.mutate

    def mutate(store, call_sid, fn, **kw):
        state = callstate.load(store, call_sid)
        if state is not None:
            out = fn(state) or {}
            esc = out.get("escalation") or {}
            if humanleg.K_CONVERSATION_CLOSED_AT in esc \
                    and not humanleg.conversation_closed(state.escalation):
                raise RuntimeError("ProvisionedThroughputExceededException")
        return real(store, call_sid, fn, **kw)

    monkeypatch.setattr(callstate, "mutate", mutate)


def test_a_confirmed_close_whose_record_fails_still_triggers_once(fdh, ops, recap, monkeypatch, caplog):
    """Orchestrator confirmed the close, so the writer starts although recording
    ``conversation_closed_at`` failed; once for the close attempt, and the status path stays quiet."""
    escalate(fdh, sid(1))
    closer = with_closer(fdh, ops, recap)
    _record_fails(monkeypatch)
    resp = fdh.post("status", sid(1), CallStatus="completed")
    assert resp["statusCode"] == 200
    assert closes(ops) == [("close", CONV)]
    assert humanleg.K_CONVERSATION_CLOSED_AT not in (fdh.state(sid(1)).escalation or {})
    assert [r for r in caplog.records if r.getMessage() == "human_leg_close_record_failed"]
    assert recap.fired == [sid(1)]
    fdh.post("status", sid(1), CallStatus="completed")  # Twilio's retry, within the close's claim
    assert closer.end(sid(1), humanleg.SWEEP) is False
    assert recap.fired == [sid(1)] and closes(ops) == [("close", CONV)]


def test_a_failed_close_with_a_failing_record_triggers_nothing(fdh, ops, recap, monkeypatch):
    escalate(fdh, sid(1))
    with_closer(fdh, ops, recap, close_ok=False)
    _record_fails(monkeypatch)
    fdh.post("status", sid(1), CallStatus="completed")
    assert closes(ops) == [("close", CONV)] and recap.fired == []


def test_a_close_already_recorded_by_another_close_does_not_trigger_again(fdh, ops, recap):
    """The record write finds ``conversation_closed_at`` set by a concurrent close (which started the
    writer): this close starts nothing."""
    escalate(fdh, sid(1))
    closer = with_closer(fdh, ops, recap)
    real_close = closer._client.close  # noqa: SLF001

    def close_and_race(conversation_id: str) -> bool:
        ok = real_close(conversation_id)
        callstate.mutate(fdh.store, sid(1), lambda s: {"escalation": {
            **(s.escalation or {}), humanleg.K_CONVERSATION_CLOSED_AT: int(CLOCK)}})
        return ok

    closer._client.close = close_and_race  # type: ignore[method-assign]  # noqa: SLF001
    fdh.post("status", sid(1), CallStatus="completed")
    assert closes(ops) == [("close", CONV)] and recap.fired == []
