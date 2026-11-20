"""The recap trigger never affects the call, and at most one trigger per path, over random end-signal
interleavings.

The real front door (``/voice/status``, ``/voice/transcription``) and the real
:class:`~owlbank.humanleg_close.HumanLegCloser` over one call-state store, with a fake Orchestrator.
Calls are drawn as AI-only, escalated with a running transcript, escalated with no conversation,
escalated with no close path wired, or escalated with a close Orchestrator always refuses; signals as
final and non-final statuses, transcript ends, sweep closes and whole sweep runs
(:func:`lambdas.conversation_sweep.handler.run_sweep`, whose Calls API answer is always unknown, so
enough runs give up on the call), duplicated and reordered; and the close's record write may fail.
The status path, the close path and the sweep's give-up each get their own
recording trigger so each is counted on its own:

* at most one trigger per path, however the signals interleave (the writer's claim does the rest);
  the close path fires exactly for the close Orchestrator confirmed, recorded or not, and never for
  a refused one; the give-up path fires exactly when the sweep gave up on the call;
* exactly one trigger in total once the call has ended and, for an escalated call with a conversation,
  the close is recorded; none before;
* The same sequence run with a trigger that raises or times out gives byte-for-byte the same
  responses (sweep reports included) as with one that succeeds, and the same close.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any
from unittest import mock

from hypothesis import example, given, settings
from hypothesis import strategies as st

from lambdas.conversation_sweep import handler as sweep
from lambdas.frontdoor.app import NoopConversationCloser
from owlbank import callstate, humanleg
from owlbank.callstate import CallState
from owlbank.humanleg_close import HumanLegCloser
from tests.component.test_frontdoor_support import CLOCK, build

CALL = "CA" + "e" * 32
CONV = "conv_" + "a" * 26
T0 = int(CLOCK)

KINDS = ("ai_only", "escalated_transcript", "escalated_no_conversation", "escalated_no_closer",
         "escalated_closes_now", "escalated_close_refused")
#: ``sweep_run`` three times over: a give-up needs SWEEP_MAX_RETRIES futile runs.
SIGNALS = st.sampled_from(["completed", "failed", "busy", "in-progress", "stopped", "error", "sweep",
                           "sweep_run", "sweep_run", "sweep_run"])
OUTCOMES = st.sampled_from(["raise", "timeout"])


class Recap:
    def __init__(self, failure: str | None) -> None:
        self.fired: list[str] = []
        self.failure = failure

    def __call__(self, call_sid: str) -> None:
        self.fired.append(call_sid)
        if self.failure == "raise":
            raise RuntimeError("invoke refused")
        if self.failure == "timeout":
            raise TimeoutError("read timeout")


class Orchestrator:
    def __init__(self, close_ok: bool = True) -> None:
        self.closes = 0
        self.confirmed = 0
        self.close_ok = close_ok

    def resolve_conversation(self, call_sid: str) -> str | None:
        return CONV

    def add_human_participant(self, conversation_id: str, worker_sid: str, identity: str) -> bool:
        return True

    def close(self, conversation_id: str) -> bool:
        self.closes += 1
        self.confirmed += self.close_ok
        return self.close_ok

    def mark_handoff(self, conversation_id: str, metadata: Any) -> bool:
        return True


class Unknown:
    """The Calls API as the sweep sees it: never a usable answer."""

    def status(self, call_sid: str) -> None:
        return None


@contextmanager
def record_fails(on: bool) -> Iterator[None]:
    """With ``on``, every call-state write that would set ``conversation_closed_at`` fails (the close's
    record); the rest go through."""
    if not on:
        yield
        return
    real = callstate.mutate

    def mutate(store: Any, call_sid: str, fn: Any, **kw: Any) -> Any:
        state = callstate.load(store, call_sid)
        if state is not None:
            esc = (fn(state) or {}).get("escalation") or {}
            if humanleg.K_CONVERSATION_CLOSED_AT in esc and not humanleg.conversation_closed(state.escalation):
                raise RuntimeError("ProvisionedThroughputExceededException")
        return real(store, call_sid, fn, **kw)

    with mock.patch.object(callstate, "mutate", mutate):
        yield


def setup(kind: str, failure: str | None) -> tuple[Any, Recap, Recap, Recap, Orchestrator, HumanLegCloser]:
    h = build()
    escalation: dict[str, Any] | None = None
    ai_sessions = 1
    if kind != "ai_only":
        escalation = {"reason": "fraud_suspected", "queue": "q", "journal_ref": "jr"}
    if kind == "escalated_transcript":
        escalation = {**(escalation or {}), "transcriptions": {CALL: {
            "name": humanleg.TRANSCRIPTION_NAME, "role": humanleg.ROLE_CUSTOMER,
            "started_at": T0 - 60, "ended_at": None}}}
    if kind == "escalated_no_conversation":
        ai_sessions = 0
    callstate.create(h.store, CallState(call_sid=CALL, expires_at=T0 + 86_400, ai_sessions=ai_sessions,
                                        language="en-US", escalation=escalation))
    at_status, at_close, at_sweep = Recap(failure), Recap(failure), Recap(failure)
    orch = Orchestrator(close_ok=kind != "escalated_close_refused")
    closer = HumanLegCloser(h.store, orch, clock=lambda: CLOCK, recap=at_close)
    h.app.d = replace(h.app.d, recap=at_status,
                      closer=NoopConversationCloser() if kind == "escalated_no_closer" else closer)
    return h, at_status, at_close, at_sweep, orch, closer


def run(h: Any, closer: HumanLegCloser, at_sweep: Recap, signals: list[str],
        trace: list[tuple[str, bool, int, int]] | None = None) -> list[tuple[Any, ...]]:
    """The responses to ``signals``; ``trace`` (when given) gets, after each signal, the signal, whether
    ``sweep_gave_up_at`` is set, ``sweep_retries`` and the give-up triggers so far."""
    out: list[tuple[Any, ...]] = []
    runs = 0
    for s in signals:
        _step(h, closer, at_sweep, s, runs, out)
        runs += s == "sweep_run"
        if trace is not None:
            st_ = callstate.load(h.store, CALL)
            esc = (st_.escalation if st_ is not None else None) or {}
            trace.append((s, humanleg.K_SWEEP_GAVE_UP_AT in esc, humanleg.sweep_retries(esc),
                          len(at_sweep.fired)))
    return out


def _step(h: Any, closer: HumanLegCloser, at_sweep: Recap, s: str, runs: int,
          out: list[tuple[Any, ...]]) -> None:
    if s == "sweep":
        out.append(("sweep", closer.end(CALL, humanleg.SWEEP)))
    elif s == "sweep_run":  # past every due time the previous run wrote, and the end's grace
        now = T0 + 1_000 + 700 * runs
        deps = sweep.SweepDeps(due=sweep.store_due(h.store),
                               states=sweep.store_states(h.store, clock=lambda now=now: now),
                               calls=Unknown(), closer=closer,
                               write=sweep.store_writer(h.store, clock=lambda now=now: now),
                               clock=lambda now=now: now, recap=at_sweep)
        out.append(("sweep_run", sweep.run_sweep(deps).as_dict()))
    elif s in ("stopped", "error"):
        r = h.post("transcription", CALL, TranscriptionEvent=f"transcription-{s}",
                   TranscriptionSid="GT" + "e" * 32)
        out.append((s, r["statusCode"], r["body"]))
    else:
        r = h.post("status", CALL, CallStatus=s)
        out.append((s, r["statusCode"], r["body"]))


GIVE_UP = ["sweep_run"] * (humanleg.SWEEP_MAX_RETRIES + 2)


@settings(max_examples=120, deadline=None)
@given(kind=st.sampled_from(KINDS), signals=st.lists(SIGNALS, max_size=14), failure=OUTCOMES,
       fail_record=st.booleans())
@example(kind="escalated_transcript", signals=GIVE_UP, failure="raise", fail_record=False)
@example(kind="escalated_close_refused", signals=["completed", *GIVE_UP, "sweep"], failure="timeout",
         fail_record=False)
@example(kind="escalated_closes_now", signals=["completed", "sweep", *GIVE_UP], failure="raise",
         fail_record=True)
# An end after the give-up resets sweep_retries but keeps the marker (no second fire).
@example(kind="escalated_transcript", signals=[*GIVE_UP, "completed", *GIVE_UP], failure="raise",
         fail_record=False)
@example(kind="escalated_close_refused", signals=[*GIVE_UP, "completed", *GIVE_UP], failure="timeout",
         fail_record=False)  # and a second give-up after that end: still no second fire
@example(kind="escalated_transcript", signals=["completed", "stopped", "stopped", "sweep_run"],
         failure="timeout", fail_record=True)
def test_at_most_one_trigger_per_path_and_never_affects_the_call(kind: str, signals: list[str],
                                                                 failure: str, fail_record: bool) -> None:
    if kind == "escalated_no_closer":  # the sweep is deployed only with humanLegConversation=on
        signals = [s for s in signals if s not in ("sweep", "sweep_run")]
    h, at_status, at_close, at_sweep, orch, closer = setup(kind, None)
    trace: list[tuple[str, bool, int, int]] = []
    with record_fails(fail_record):
        responses = run(h, closer, at_sweep, signals, trace)
    # At most one per path.
    assert len(at_status.fired) <= 1 and len(at_close.fired) <= 1 and len(at_sweep.fired) <= 1
    assert set(at_status.fired + at_close.fired + at_sweep.fired) <= {CALL}
    state = callstate.load(h.store, CALL)
    assert state is not None
    closed = humanleg.conversation_closed(state.escalation)
    ended = any(s in ("completed", "failed") for s in signals)
    # The close path fires exactly for the close Orchestrator confirmed (recorded or not; the claim
    # keeps a repeat out), never for a refused one.
    assert len(at_close.fired) == orch.confirmed
    if not fail_record:
        assert len(at_close.fired) == (1 if closed else 0)
    # The give-up path fires exactly once, in the sweep run that first set sweep_gave_up_at (the run
    # whose retry reached SWEEP_MAX_RETRIES), and never at any other step. The marker is kept after
    # that, while sweep_retries may be reset later (an end brings the call back), so this is checked
    # step by step rather than on the final state.
    gave_up = humanleg.K_SWEEP_GAVE_UP_AT in (state.escalation or {})
    assert len(at_sweep.fired) == (1 if gave_up else 0)
    before_marker, before_fired = False, 0
    for signal, marker, retries, fired in trace:
        assert not (before_marker and not marker)  # once set, kept
        first = marker and not before_marker
        assert fired - before_fired == (1 if first else 0)
        if first:
            assert signal == "sweep_run" and retries == humanleg.SWEEP_MAX_RETRIES
        before_marker, before_fired = marker, fired
    if kind in ("ai_only", "escalated_no_conversation", "escalated_no_closer"):
        assert at_close.fired == [] and at_sweep.fired == []
        # The status path fires once the first final status was "completed"/"failed".
        first_end = next((s for s in signals if s in ("completed", "failed", "busy")), None)
        assert len(at_status.fired) == (1 if first_end in ("completed", "failed") else 0)
    else:
        assert at_status.fired == []  # a close will (or did) start the writer
        if ended and kind == "escalated_closes_now":
            assert len(at_close.fired) == 1 and (closed or fail_record)
        if kind == "escalated_close_refused":
            assert at_close.fired == [] and not closed
    # The same signals with a failing trigger: the same responses and the same close.
    h2, s2, c2, w2, orch2, closer2 = setup(kind, failure)
    with record_fails(fail_record):
        assert run(h2, closer2, w2, signals) == responses
    assert (len(s2.fired), len(c2.fired), len(w2.fired), orch2.closes) == (
        len(at_status.fired), len(at_close.fired), len(at_sweep.fired), orch.closes)
    state2 = callstate.load(h2.store, CALL)
    assert state2 is not None and state2.escalation == state.escalation
