"""Front-door Tier 1.

Signed fake webhooks into the real front door (``test_frontdoor_support.build``) with telemetry
installed against an in-memory exporter:

* speech timing on the ``gather.<slot>`` span: the partials counted on the Gather's item, the time
  from the last change in their length to the answering request, and whether the Gather finished
  early, for an early finish and a normal finish; a failing count is ignored; with telemetry off no
  write per partial and the same TwiML;
* ``call.end``: Twilio's call status and duration and the last step, for each end status, never a
  phone number;
* ``/voice/taskrouter``: a signed event gives ``handoff.queue`` under the task's call with the
  queue, the wait and the outcome and no worker or caller identity; a bad signature is a 403 with no
  span; an event without ``call_sid`` is dropped.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lambdas.frontdoor import gathers
from lambdas.frontdoor.app import END_STATUSES
from owlbank import sensitive
from owlbank.segments import clear_sensitive
from owlbank.telemetry import frontdoor
from tests.component.test_frontdoor_early_digits import (
    ZIP_WORDS, gather_of, gid_of, make, partial, redirect_route, route_of,
)
from tests.component.test_frontdoor_support import (
    CLOCK, LEES, MARIA, NOBODY, THOMAS, Harness, make_event, sid, twilio_post,
)
from tests.component.test_frontdoor_telemetry import deterministic, install, strings_of

PHONES = (THOMAS, LEES, MARIA, NOBODY, "+12025550100")


class Clock:
    """A settable clock for the front door (seconds since the epoch, as ``time.time``)."""

    def __init__(self) -> None:
        self.now = CLOCK

    def __call__(self) -> float:
        return self.now

    def at(self, offset_ms: int) -> None:
        self.now = CLOCK + offset_ms / 1000


@pytest.fixture(autouse=True)
def _untraced_after(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(frontdoor, "_tracer", None)
    yield
    for n in range(40):
        clear_sensitive(sid(n))
        sensitive.forget(sid(n))


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    exp = InMemorySpanExporter()
    install(exp)
    return exp


def early_harness() -> tuple[Harness, Any, Clock]:
    h, calls = make()
    clock = Clock()
    h.app.d.clock = clock
    return h, calls, clock


def answered(spans: list[ReadableSpan], slot: str = "zip") -> ReadableSpan:
    """The one webhook span that answered the Gather (the one with ``finished_early``). It took the
    ``gather.<slot>`` step; a verified factor's span is then named by its last step, ``verify``."""
    (span,) = [s for s in spans if "owlbank.finished_early" in (s.attributes or {})]
    assert span.name in (f"gather.{slot}", "verify"), span.name
    return span


def attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


# --------------------------------------------------------------------------- speech timing


def test_early_finish_gives_partials_end_of_speech_and_finished_early(exporter) -> None:
    h, calls, clock = early_harness()
    call = sid(1)
    g = gather_of(h.start(call, THOMAS))
    clock.at(500)
    partial(h, call, g, "nine four one")          # 13 characters
    clock.at(900)
    partial(h, call, g, "nine four one")          # the same length: no change
    clock.at(1500)
    partial(h, call, g, ZIP_WORDS)                # complete: the claim, the last change
    assert len(calls.updates) == 1
    clock.at(1800)
    first = h.post(redirect_route(calls.updates[0][1]), call)
    assert first["statusCode"] == 200
    span = answered(list(exporter.get_finished_spans()))
    a = attrs(span)
    assert a["owlbank.partials"] == 3
    assert a["owlbank.end_of_speech_ms"] == 300
    assert a["owlbank.finished_early"] is True
    # Twilio's own final webhook for the same Gather replays: no second answering span.
    clock.at(2600)
    final = h.post(route_of(g.get("action") or ""), call, SpeechResult=ZIP_WORDS, Confidence="0.9")
    assert final["body"] == first["body"]
    answered(list(exporter.get_finished_spans()))
    # Lengths and times only on the item: never the text.
    item = gathers.load(h.store, call, gid_of(g))
    assert item is not None and item["last_len"] == len(ZIP_WORDS)
    assert ZIP_WORDS not in json.dumps(item) and "nine four one" not in json.dumps(item)


def test_normal_finish_gives_partials_end_of_speech_and_not_early(exporter) -> None:
    h, calls, clock = early_harness()
    call = sid(2)
    g = gather_of(h.start(call, THOMAS))
    clock.at(400)
    partial(h, call, g, "nine")
    clock.at(1000)
    partial(h, call, g, "nine four one")
    assert calls.updates == []
    clock.at(2200)
    resp = h.post(route_of(g.get("action") or ""), call, SpeechResult=ZIP_WORDS, Confidence="0.9")
    assert resp["statusCode"] == 200
    a = attrs(answered(list(exporter.get_finished_spans())))
    assert (a["owlbank.partials"], a["owlbank.end_of_speech_ms"], a["owlbank.finished_early"]) == \
        (2, 1200, False)
    texts = strings_of(list(exporter.get_finished_spans()))
    assert not any("nine four" in t or ZIP_WORDS in t for t in texts)


def test_final_without_partials_has_finished_early_only(exporter) -> None:
    h, _, clock = early_harness()
    call = sid(3)
    g = gather_of(h.start(call, THOMAS))
    clock.at(3000)
    h.post(route_of(g.get("action") or ""), call, SpeechResult=ZIP_WORDS, Confidence="0.9")
    a = attrs(answered(list(exporter.get_finished_spans())))
    assert a["owlbank.finished_early"] is False
    assert "owlbank.partials" not in a and "owlbank.end_of_speech_ms" not in a


def test_a_failing_partial_count_is_ignored(exporter, caplog: pytest.LogCaptureFixture,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    caplog.set_level(logging.DEBUG)
    h, calls, clock = early_harness()
    call = sid(4)
    g = gather_of(h.start(call, THOMAS))

    def broken(*a: Any, **kw: Any) -> None:
        raise RuntimeError("dynamodb down")

    monkeypatch.setattr(h.store, "increment", broken)
    clock.at(500)
    partial(h, call, g, "nine four one")
    clock.at(1500)
    partial(h, call, g, ZIP_WORDS)
    assert len(calls.updates) == 1                    # the early finish still happens
    first = h.post(redirect_route(calls.updates[0][1]), call)
    assert first["statusCode"] == 200 and "<ConversationRelay" in first["body"]
    failed = [r for r in caplog.records if getattr(r, "event", None) == "partial_count_failed"]
    assert len(failed) == 2 and all(r.decision == "ignored" for r in failed)
    a = attrs(answered(list(exporter.get_finished_spans())))
    assert a["owlbank.finished_early"] is True
    assert "owlbank.partials" not in a and "owlbank.end_of_speech_ms" not in a


def _partials_journey(h: Harness, calls: Any, clock: Clock, call: str) -> list[str]:
    g = gather_of(h.start(call, THOMAS))
    for i, text in enumerate(("nine", "nine four one", ZIP_WORDS)):
        clock.at(400 * (i + 1))
        partial(h, call, g, text)
    h.post(redirect_route(calls.updates[0][1]), call)
    h.post(route_of(g.get("action") or ""), call, SpeechResult=ZIP_WORDS, Confidence="0.9")
    return h.bodies


def test_off_no_write_per_partial_and_the_same_twiml(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: dict[str, list[str]] = {}
    counted: dict[str, int] = {}
    for mode in ("off", "on"):
        monkeypatch.setattr(frontdoor, "_tracer", None)
        if mode == "on":
            install(InMemorySpanExporter())
        h, calls, clock = early_harness()
        real = h.store.increment
        counted[mode] = 0

        def counting(*a: Any, _real: Any = real, _mode: str = mode, **kw: Any) -> None:
            counted[_mode] += 1
            _real(*a, **kw)

        monkeypatch.setattr(h.store, "increment", counting)
        with deterministic():
            bodies[mode] = list(_partials_journey(h, calls, clock, sid(5)))
        if mode == "off":
            item = next(v for k, v in h.store._tables["calls"].items()  # noqa: SLF001
                        if "#gather#" in k[0])
            assert not set(gathers.PARTIAL_FIELDS) & set(item)
    assert counted == {"off": 0, "on": 3}
    assert bodies["off"] == bodies["on"]


# --------------------------------------------------------------------------- call end


@pytest.mark.parametrize("status", sorted(END_STATUSES))
def test_call_end_carries_status_duration_and_last_step(exporter, status: str) -> None:
    h, _, _ = early_harness()
    call = sid(6)
    h.start(call, THOMAS)                                   # waiting at the ZIP question
    resp = h.post("status", call, CallStatus=status, CallDuration="42", From=THOMAS,
                  To="+12025550100", Caller=THOMAS, Called="+12025550100")
    assert resp["statusCode"] == 200
    spans = list(exporter.get_finished_spans())
    (end,) = [s for s in spans if s.name == "call.end"]
    a = attrs(end)
    assert a["owlbank.call_status"] == status
    assert a["owlbank.call_duration_s"] == 42
    assert a["owlbank.last_step"] == "gather.zip"
    assert a["session.id"] == call
    assert not any(p in t for t in strings_of([end]) for p in PHONES)


def test_call_end_after_the_ai_has_last_step_ai_and_no_duration_when_absent(exporter) -> None:
    from owlbank import callstate

    h, _, _ = early_harness()
    call = sid(7)
    h.login_thomas(call)                                    # verified: ConversationRelay connected
    callstate.mutate(h.store, call, lambda s: {"ai_sessions": 1})
    h.post("status", call, CallStatus="completed")
    (end,) = [s for s in exporter.get_finished_spans() if s.name == "call.end"]
    a = attrs(end)
    assert a["owlbank.last_step"] == "ai" and "owlbank.call_duration_s" not in a


def test_status_that_is_not_an_end_has_the_status_only(exporter) -> None:
    h, _, _ = early_harness()
    call = sid(8)
    h.start(call, THOMAS)
    h.post("status", call, CallStatus="in-progress")
    (end,) = [s for s in exporter.get_finished_spans() if s.name == "call.end"]
    a = attrs(end)
    assert a["owlbank.call_status"] == "in-progress" and "owlbank.last_step" not in a


# --------------------------------------------------------------------------- queue wait

TASK_CALL = "CA" + "0f" * 16
IDENTITY = {"WorkerName": "Alice Agent", "WorkerSid": "WK" + "a" * 32,
            "WorkerAttributes": json.dumps({"contact_uri": "client:alice", "full_name": "Alice Agent"}),
            "WorkerActivityName": "Busy"}


def task_event(event_type: str, *, attributes: dict[str, Any] | None = None, **extra: str) -> dict[str, str]:
    attrs_ = {"call_sid": TASK_CALL, "escalation_reason": "caller_request", "language": "en-US",
              "from": THOMAS, "caller": THOMAS} if attributes is None else attributes
    return {"EventType": event_type, "AccountSid": "AC" + "b" * 32, "WorkspaceSid": "WS" + "c" * 32,
            "WorkspaceName": "Flex Task Assignment", "ResourceType": "task",
            "ResourceSid": "WT" + "d" * 32, "TaskSid": "WT" + "d" * 32,
            "TaskAttributes": json.dumps(attrs_), "TaskAge": "37", "TaskPriority": "0",
            "TaskQueueSid": "WQ" + "e" * 32, "TaskQueueName": "OwlBank Support EN",
            "Timestamp": "1760000037", "EventDescription": "Task WT... updated", **extra}


def post_event(h: Harness, params: dict[str, str]) -> dict[str, Any]:
    return twilio_post(h.app.handle, "taskrouter", params)


@pytest.mark.parametrize("event_type, extra, outcome", [
    ("reservation.accepted", {**IDENTITY, "ReservationSid": "WR" + "f" * 32}, "accepted"),
    ("task.canceled", {"TaskCanceledReason": "Task canceled on Workflow timeout"}, "cancelled"),
    ("task.canceled", {"TaskCanceledReason": "hangup"}, "hangup"),
    ("workflow.timeout", {}, "timeout"),
])
def test_taskrouter_event_gives_handoff_queue_without_identity(exporter, event_type: str,
                                                               extra: dict[str, str], outcome: str) -> None:
    h, _, _ = early_harness()
    resp = post_event(h, task_event(event_type, **extra))
    assert resp == {"statusCode": 204, "headers": {"Content-Type": "application/json"}, "body": ""}
    (span,) = exporter.get_finished_spans()
    a = attrs(span)
    assert span.name == "handoff.queue"
    assert a["session.id"] == a["call_sid"] == TASK_CALL
    assert a["owlbank.queue"] == "OwlBank Support EN"
    assert a["owlbank.wait_ms"] == 37_000
    assert a["owlbank.outcome"] == outcome
    assert {k for k in a if k.startswith("owlbank.")} == {
        "owlbank.route", "owlbank.step", "owlbank.queue", "owlbank.wait_ms", "owlbank.outcome"}
    texts = strings_of([span])
    for secret in ("Alice", "client:alice", IDENTITY["WorkerSid"], *PHONES, "Flex Task Assignment"):
        assert not any(secret in t for t in texts), secret


def test_queue_falls_back_to_the_queue_sid() -> None:
    exp = InMemorySpanExporter()
    install(exp)
    h, _, _ = early_harness()
    params = task_event("workflow.timeout")
    del params["TaskQueueName"]
    post_event(h, params)
    (span,) = exp.get_finished_spans()
    assert attrs(span)["owlbank.queue"] == "WQ" + "e" * 32


def test_taskrouter_with_a_bad_signature_is_rejected_with_no_span(exporter) -> None:
    h, _, _ = early_harness()
    resp = h.app.handle(make_event("taskrouter", task_event("reservation.accepted"), signature="bad"))
    assert resp["statusCode"] == 403
    wrong_basic = h.app.handle(make_event("taskrouter", task_event("reservation.accepted"),
                                          auth="Basic d3Jvbmc6d3Jvbmc="))
    assert wrong_basic["statusCode"] == 403
    assert exporter.get_finished_spans() == ()


@pytest.mark.parametrize("attributes", [
    {"escalation_reason": "caller_request"},                  # no call_sid
    {"call_sid": "not-a-call-sid"},
])
def test_taskrouter_event_without_a_call_sid_is_dropped(exporter, caplog: pytest.LogCaptureFixture,
                                                        attributes: dict[str, Any]) -> None:
    caplog.set_level(logging.INFO)
    h, _, _ = early_harness()
    resp = post_event(h, task_event("reservation.accepted", attributes=attributes))
    assert resp["statusCode"] == 204
    # Dropped: without a call the per-call sampler records no span at all.
    assert exporter.get_finished_spans() == ()
    dropped = [r for r in caplog.records if getattr(r, "event", None) == "taskrouter_event_dropped"]
    assert len(dropped) == 1 and dropped[0].reason == "no_call_sid"


def test_unparseable_attributes_are_dropped(exporter) -> None:
    h, _, _ = early_harness()
    params = task_event("workflow.timeout")
    params["TaskAttributes"] = "{not json"
    assert post_event(h, params)["statusCode"] == 204
    assert exporter.get_finished_spans() == ()


def test_other_event_types_are_a_no_op(exporter) -> None:
    h, _, _ = early_harness()
    resp = post_event(h, task_event("worker.activity.update", **IDENTITY))
    assert resp["statusCode"] == 204
    (span,) = exporter.get_finished_spans()
    assert span.name == "voice.taskrouter"
    assert {k for k in attrs(span) if k.startswith("owlbank.")} == {"owlbank.route"}
    assert not any("Alice" in t for t in strings_of([span]))


def test_taskrouter_with_telemetry_off_answers_204_and_touches_nothing() -> None:
    h, _, _ = early_harness()
    before = {t: dict(v) for t, v in h.store._tables.items()}  # noqa: SLF001
    resp = post_event(h, task_event("reservation.accepted"))
    assert resp["statusCode"] == 204
    assert {t: dict(v) for t, v in h.store._tables.items()} == before  # noqa: SLF001


def test_a_real_enqueue_names_its_call_in_the_task_attributes() -> None:
    """The ``call_sid`` the route reads is the one ``build_task_attributes`` puts in ``<Enqueue>``."""
    h, _, _ = early_harness()
    call = sid(10)
    h.login_thomas(call)
    root = h.handoff(call, {"reason": "escalate", "escalationReason": "complaint"})
    task = root.find("Enqueue/Task")
    assert task is not None, ET.tostring(root)
    assert json.loads(task.text or "{}")["call_sid"] == call
