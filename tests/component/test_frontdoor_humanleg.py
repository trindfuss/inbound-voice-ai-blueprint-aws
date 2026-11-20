"""Human-leg: the escalation's conversation id, ``handoff`` journal event and ``<Enqueue>``
``action`` (no ``waitUrl``), and the queue route.

The close after the human leg (``/voice/status`` and ``/voice/transcription``). Participants,
hand-off metadata and the transcription start from TaskRouter's ``reservation.accepted``. The
no-leak scan runs at teardown by the ``fd`` fixture.

Characterization of the Flex-configured ``_escalate`` (task attributes, order of side effects,
TwiML) so a later refactor must leave it unchanged. No property test: it pins fixed behaviour
of one path, not an invariant over an input space. The no-workflow guard is deliberately not
pinned here."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

from lambdas.frontdoor.app import NoopConversationCloser
from lambdas.frontdoor.handler import HUMAN_LEG_TIMEOUT_S, human_leg_client, human_leg_closer
from owlbank import callstate, humanleg
from owlbank.humanleg_client import OrchestratorHumanLeg, Reservation, TaskReservations
from owlbank.humanleg_close import HumanLegCloser
from owlbank.policy import Policy

from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    CLOCK, ENQUEUE_ACTION_URL, WORKFLOW, build, fd, make_event, says, sid,
)

DATA = Path(__file__).resolve().parents[2] / "data"
CONV = "conv_" + "a" * 26
GOODBYE_EN = "Thank you for calling OwlBank. Goodbye."


class FakeHumanLeg:
    """``resolve_conversation`` only (the escalation's one call): ``answer`` or ``raise``."""

    def __init__(self, answer: str | None = CONV, *, fail: bool = False) -> None:
        self.answer, self.fail = answer, fail
        self.resolved: list[str] = []

    def resolve_conversation(self, call_sid: str) -> str | None:
        self.resolved.append(call_sid)
        if self.fail:
            raise RuntimeError("orchestrator down")
        return self.answer


@pytest.fixture
def hl() -> FakeHumanLeg:
    return FakeHumanLeg()


@pytest.fixture
def fdh(fd, hl):
    """``fd`` with the human-leg client wired (as ``humanLegConversation=on``)."""
    fd.app.d = replace(fd.app.d, human_leg=hl)
    return fd


def escalate(h, call_sid: str, *, ai_sessions: int = 1, **handoff: str) -> ET.Element:
    h.login_thomas(call_sid)
    callstate.mutate(h.store, call_sid, lambda s: {"ai_sessions": ai_sessions})
    h.set_pending_tool(call_sid, "escalate", {"reason": "fraud_suspected", "summary": "Card lost."})
    return h.handoff(call_sid, {"reason": "escalate", "escalationReason": "fraud_suspected",
                                "journal": "jr-7f3a", **handoff})


def task_attrs(root: ET.Element) -> dict:
    return json.loads(root.find("Enqueue/Task").text)


def with_transcription(h, on: bool, *, agent_legs: bool = True) -> None:
    doc = json.loads((DATA / "policy.json").read_text())
    doc["human_leg"] = {"transcription": on, "agent_legs": agent_legs}
    h.app.d = replace(h.app.d, policy=Policy.from_dict(doc))


# --------------------------------------------------------------------------- the escalation TwiML


@pytest.mark.parametrize("switch", [False, True])
def test_enqueue_has_an_action_and_no_wait_url_with_the_switch_on_or_off(fdh, switch):
    with_transcription(fdh, switch)
    root = escalate(fdh, sid(1))
    assert [c.tag for c in root] == ["Say", "Enqueue"]  # no <Start><Transcription>
    assert root.find(".//Start") is None and root.find(".//Transcription") is None
    enq = root.find("Enqueue")
    assert enq.attrib == {"workflowSid": WORKFLOW, "action": ENQUEUE_ACTION_URL}  # no waitUrl
    assert root.find("Enqueue/Task").attrib == {"priority": "100"}  # no <Task> timeout


def test_known_conversation_from_the_handoff_is_used_without_a_lookup(fdh, hl):
    root = escalate(fdh, sid(1), conversationId=CONV)
    assert task_attrs(root)["conversation_id"] == CONV
    assert hl.resolved == []


def test_unknown_conversation_is_looked_up_by_call_sid(fdh, hl):
    root = escalate(fdh, sid(1))
    assert hl.resolved == [sid(1)]
    assert task_attrs(root)["conversation_id"] == CONV


def test_no_lookup_when_the_call_never_had_an_ai_session(fdh, hl):
    root = escalate(fdh, sid(1), ai_sessions=0)
    assert hl.resolved == []  # no Orchestrator request: there is no conversation to find
    assert task_attrs(root)["conversation_id"] is None
    assert [c.tag for c in root] == ["Say", "Enqueue"]


@pytest.mark.parametrize("hl", [FakeHumanLeg(None), FakeHumanLeg(fail=True)],
                         ids=["zero_or_two_active", "lookup_raises"])
def test_no_conversation_id_still_enqueues_the_same_task(fdh, hl):
    root = escalate(fdh, sid(1))
    healthy = escalate(build(human_leg=FakeHumanLeg()), sid(1))
    assert task_attrs(root)["conversation_id"] is None
    assert {k: v for k, v in task_attrs(root).items() if k != "conversation_id"} == \
        {k: v for k, v in task_attrs(healthy).items() if k != "conversation_id"}
    assert root.find("Enqueue").attrib == healthy.find("Enqueue").attrib
    assert says(root) == says(healthy)


def test_without_the_client_nothing_is_looked_up(fd):
    root = escalate(fd, sid(1))  # humanLegConversation=off: Deps.human_leg is None
    assert task_attrs(root)["conversation_id"] is None
    assert root.find("Enqueue").get("waitUrl") is None


def test_a_handoff_to_human_is_journaled_and_logged(fdh, caplog):
    escalate(fdh, sid(1))
    events = [e for e in fdh.journal(sid(1)) if e["type"] == "handoff" and e.get("to") == "human"]
    assert len(events) == 1 and events[0]["conversation_id"] == CONV
    lines = [r for r in caplog.records if r.getMessage() == "human_leg_handoff"]
    assert lines and getattr(lines[-1], "conversation_id", None) == CONV


def test_a_failed_handoff_journal_write_changes_nothing_in_the_twiml(fdh, monkeypatch):
    healthy = escalate(build(human_leg=FakeHumanLeg()), sid(1))
    append = fdh.app.journal.append

    def failing(call_sid, event):
        if event.get("type") == "handoff" and event.get("to") == "human":
            raise RuntimeError("journal down")
        return append(call_sid, event)

    monkeypatch.setattr(fdh.app.journal, "append", failing)
    root = escalate(fdh, sid(1))
    assert ET.tostring(root) == ET.tostring(healthy)


def test_non_escalating_calls_never_look_up_a_conversation(fdh, hl):
    root = fdh.login_thomas(sid(1))
    assert root.find(".//Enqueue") is None
    assert hl.resolved == []
    assert not [e for e in fdh.journal(sid(1)) if e["type"] == "handoff" and e.get("to") == "human"]


# --------------------------------------------------------------------------- action (/voice/enqueue-action)


def action(h, call_sid: str, result: str | None, **extra: str) -> ET.Element:
    params = {"QueueResult": result, **extra} if result is not None else extra
    return h.xml("enqueue-action", call_sid, follow=False, **params)


@pytest.mark.parametrize("result", ["leave", "bridged"])
def test_a_caller_who_left_the_queue_hears_goodbye(fdh, result):
    escalate(fdh, sid(1))
    root = action(fdh, sid(1), result, QueueTime="7200")
    assert says(root) == [GOODBYE_EN] and [c.tag for c in root] == ["Say", "Hangup"]


@pytest.mark.parametrize("result", ["queue-full", "error", "system-error"])
def test_a_failed_enqueue_gets_the_apology_then_goodbye(fdh, result):
    escalate(fdh, sid(1))
    root = action(fdh, sid(1), result)
    assert [c.tag for c in root] == ["Say", "Say", "Hangup"] and says(root)[-1] == GOODBYE_EN


@pytest.mark.parametrize("result", ["hangup", "redirected", "bridging-in-process",
                                    "redirected-from-bridged", "something-new", None])
def test_no_hangup_unless_the_caller_left(fdh, result):
    escalate(fdh, sid(1))
    root = action(fdh, sid(1), result, QueueTime="99999")
    assert list(root) == []  # never a <Hangup> for a caller who may still be connected


def test_the_action_never_escalates_on_an_unexpected_error(fdh, monkeypatch):
    escalate(fdh, sid(1))
    monkeypatch.setattr(fdh.app, "enqueue_action", lambda req: (_ for _ in ()).throw(RuntimeError("bug")))
    assert list(action(fdh, sid(1), "leave")) == []


# --------------------------------------------------------------------------- handler wiring


def test_handler_builds_the_client_only_when_on_and_a_secret_is_set():
    secret = "arn:aws:secretsmanager:us-east-1:111122223333:secret:twilio-api-key"
    client = human_leg_client({"OWLBANK_HUMAN_LEG_CONVERSATION": "on"}, secret)
    assert isinstance(client, OrchestratorHumanLeg) and client._timeout_s == HUMAN_LEG_TIMEOUT_S  # noqa: SLF001
    assert human_leg_client({"OWLBANK_HUMAN_LEG_CONVERSATION": "off"}, secret) is None
    assert human_leg_client({}, secret) is None
    assert human_leg_client({"OWLBANK_HUMAN_LEG_CONVERSATION": "on"}, None) is None


def test_handler_builds_the_closer_only_with_the_client():
    secret = "arn:aws:secretsmanager:us-east-1:111122223333:secret:twilio-api-key"
    client = human_leg_client({"OWLBANK_HUMAN_LEG_CONVERSATION": "on"}, secret)
    env = {"OWLBANK_HUMAN_LEG_CONVERSATION": "on", "OWLBANK_TASKROUTER_WORKSPACE_SID": "WS" + "f" * 32}
    closer = human_leg_closer(env, secret, object(), client)
    assert isinstance(closer, HumanLegCloser)
    assert isinstance(closer._reservations, TaskReservations) and closer._calls is not None  # noqa: SLF001
    no_workspace = human_leg_closer({"OWLBANK_HUMAN_LEG_CONVERSATION": "on"}, secret, object(), client)
    assert no_workspace is not None and no_workspace._reservations is None  # noqa: SLF001
    assert human_leg_closer(env, secret, object(), None) is None  # off: the no-op closer stays
    assert human_leg_closer(env, None, object(), client) is None


# --------------------------------------------------------------------------- the close

W_A, W_B, W_C = ("WK" + c * 32 for c in "abc")
TASK = "WT" + "d" * 32


class FakeOrchestrator:
    """Every request the close path makes, in order, into one shared ``ops`` list."""

    def __init__(self, ops: list, *, close_ok: bool = True, raise_on_close: bool = False) -> None:
        self.ops, self.close_ok, self.raise_on_close = ops, close_ok, raise_on_close

    def resolve_conversation(self, call_sid: str) -> str | None:
        self.ops.append(("resolve", call_sid))
        return CONV

    def add_human_participant(self, conversation_id: str, worker_sid: str, identity: str) -> bool:
        self.ops.append(("add", conversation_id, worker_sid))
        return True

    def close(self, conversation_id: str) -> bool:
        self.ops.append(("close", conversation_id))
        if self.raise_on_close:
            raise RuntimeError("orchestrator down")
        return self.close_ok

    def mark_handoff(self, conversation_id, metadata) -> bool:  # pragma: no cover - not on this path
        self.ops.append(("mark", conversation_id))
        return True


class FakeCalls:
    def __init__(self, ops: list) -> None:
        self.ops = ops

    def stop_transcription(self, call_sid: str, name: str) -> bool:
        self.ops.append(("stop", call_sid, name))
        return True


class FakeReservations:
    def __init__(self, ops: list, rows: list[Reservation] | None) -> None:
        self.ops, self.rows = ops, rows

    def list(self, task_sid: str) -> list[Reservation] | None:
        self.ops.append(("reservations", task_sid))
        return self.rows


@pytest.fixture
def ops() -> list:
    return []


def with_closer(h, ops: list, *, rows: list[Reservation] | None = None, **orchestrator) -> HumanLegCloser:
    closer = HumanLegCloser(h.store, FakeOrchestrator(ops, **orchestrator), calls=FakeCalls(ops),
                            reservations=FakeReservations(ops, rows or []), clock=lambda: CLOCK,
                            journal=h.app.journal)
    h.app.d = replace(h.app.d, closer=closer)
    return closer


def human_leg(h, call_sid: str, **escalation) -> None:
    """What an accept records (task, participants, the transcriptions)."""
    callstate.mutate(h.store, call_sid, lambda s: {"escalation": {**(s.escalation or {}), **escalation}})


def customer_leg(call_sid: str) -> dict:
    """The ``transcriptions`` map with the customer leg's transcription running."""
    return {call_sid: {"name": humanleg.TRANSCRIPTION_NAME, "role": humanleg.ROLE_CUSTOMER,
                       "started_at": int(CLOCK) - 60, "ended_at": None}}


def transcription(h, call_sid: str, event: str, **extra: str) -> dict:
    return h.post("transcription", call_sid, TranscriptionEvent=event,
                  TranscriptionSid="GT" + "e" * 32, **extra)


def closes(ops: list) -> list:
    return [o for o in ops if o[0] == "close"]


def test_switch_on_completed_alone_does_not_close_and_stopped_closes_once(fdh, ops):
    escalate(fdh, sid(1))
    human_leg(fdh, sid(1), transcriptions=customer_leg(sid(1)), task_sid=TASK)
    with_closer(fdh, ops)
    fdh.post("status", sid(1), CallStatus="completed")
    assert closes(ops) == []
    assert fdh.state(sid(1)).escalation["call_ended_at"] == int(CLOCK)
    resp = transcription(fdh, sid(1), "transcription-stopped")
    assert resp["statusCode"] == 204 and resp["body"] == ""
    assert closes(ops) == [("close", CONV)]
    assert ("stop", sid(1), "humanleg") not in ops  # Twilio already ended it
    esc = fdh.state(sid(1)).escalation
    assert esc["conversation_closed_at"] == int(CLOCK) and esc["conversation_id"] == CONV
    transcription(fdh, sid(1), "transcription-stopped")  # a retry
    fdh.post("status", sid(1), CallStatus="completed")
    assert closes(ops) == [("close", CONV)]


def test_stopped_before_the_status_callback_closes_and_the_status_adds_nothing(fdh, ops):
    escalate(fdh, sid(1))
    human_leg(fdh, sid(1), transcriptions=customer_leg(sid(1)))
    with_closer(fdh, ops)
    transcription(fdh, sid(1), "transcription-stopped")
    fdh.post("status", sid(1), CallStatus="completed")
    assert closes(ops) == [("close", CONV)]


@pytest.mark.parametrize("call_status", ["completed", "failed"])
def test_switch_off_completed_closes_once(fdh, ops, call_status):
    escalate(fdh, sid(1))
    with_closer(fdh, ops)
    fdh.post("status", sid(1), CallStatus=call_status)
    fdh.post("status", sid(1), CallStatus=call_status)
    assert closes(ops) == [("close", CONV)]
    assert not [o for o in ops if o[0] == "stop"]


def test_non_final_status_never_closes(fdh, ops):
    escalate(fdh, sid(1))
    with_closer(fdh, ops)
    for status in ("in-progress", "busy", "no-answer", "canceled"):
        fdh.post("status", sid(1), CallStatus=status)
    assert closes(ops) == []


def test_transcription_error_mid_call_then_completed_closes(fdh, ops):
    escalate(fdh, sid(1))
    human_leg(fdh, sid(1), transcriptions=customer_leg(sid(1)))
    with_closer(fdh, ops)
    transcription(fdh, sid(1), "transcription-error", TranscriptionError="x", TranscriptionErrorCode="32650")
    assert closes(ops) == []  # the call is still up
    fdh.post("status", sid(1), CallStatus="completed")
    assert closes(ops) == [("close", CONV)]


def test_transcription_error_after_call_end_closes(fdh, ops):
    escalate(fdh, sid(1))
    human_leg(fdh, sid(1), transcriptions=customer_leg(sid(1)))
    with_closer(fdh, ops)
    fdh.post("status", sid(1), CallStatus="completed")
    transcription(fdh, sid(1), "transcription-error")
    assert closes(ops) == [("close", CONV)]


def test_a_running_transcription_is_stopped_before_the_close(fdh, ops):
    """The sweep's path closes while Twilio never reported the transcript's end."""
    escalate(fdh, sid(1))
    human_leg(fdh, sid(1), transcriptions=customer_leg(sid(1)))
    closer = with_closer(fdh, ops)
    assert closer.end(sid(1), humanleg.SWEEP) is True
    kinds = [o[0] for o in ops]
    assert ("stop", sid(1), "humanleg") in ops and kinds.index("stop") < kinds.index("close")


def test_a_missed_accept_is_repaired_before_the_close(fdh, ops):
    escalate(fdh, sid(1))
    human_leg(fdh, sid(1), task_sid=TASK, participants=[{"worker_sid": W_A, "added_at": int(CLOCK)}])
    with_closer(fdh, ops, rows=[Reservation(W_A, "completed"), Reservation(W_B, "completed"),
                                Reservation(W_C, "rejected")])
    fdh.post("status", sid(1), CallStatus="completed")
    assert [o for o in ops if o[0] in ("add", "close")] == [("add", CONV, W_B), ("close", CONV)]
    assert ("reservations", TASK) in ops
    listed = [p["worker_sid"] for p in fdh.state(sid(1)).escalation["participants"]]
    assert listed == [W_A, W_B]


def test_events_after_close_write_nothing(fdh, ops, caplog):
    escalate(fdh, sid(1))
    human_leg(fdh, sid(1), transcriptions=customer_leg(sid(1)), task_sid=TASK)
    with_closer(fdh, ops, rows=[Reservation(W_B, "accepted")])
    transcription(fdh, sid(1), "transcription-stopped")
    done, escalation = len(ops), dict(fdh.state(sid(1)).escalation)
    caplog.clear()
    transcription(fdh, sid(1), "transcription-error")
    transcription(fdh, sid(1), "transcription-stopped")
    fdh.post("status", sid(1), CallStatus="completed")
    assert len(ops) == done  # no Orchestrator, TaskRouter or Calls request
    assert dict(fdh.state(sid(1)).escalation) == escalation  # nothing recorded after the close
    assert [r for r in caplog.records if r.getMessage() == "human_leg_after_close"]


def test_a_failed_close_releases_the_claim_so_a_later_signal_retries(fdh, ops, caplog):
    escalate(fdh, sid(1))
    closer = with_closer(fdh, ops, close_ok=False)
    resp = fdh.post("status", sid(1), CallStatus="completed")
    assert resp["statusCode"] == 200 and "<Response />" in resp["body"]
    esc = fdh.state(sid(1)).escalation
    assert "conversation_closed_at" not in esc and "close_claimed_at" not in esc
    failed = [r for r in caplog.records if r.getMessage() == "human_leg_close_failed"]
    assert failed and getattr(failed[-1], "conversation_id", None) == CONV
    closer._client.close_ok = True  # noqa: SLF001
    assert closer.end(sid(1), humanleg.SWEEP) is True
    assert closes(ops) == [("close", CONV), ("close", CONV)]


def test_a_raising_dependency_never_breaks_the_status_answer(fdh, ops, caplog):
    escalate(fdh, sid(1))
    with_closer(fdh, ops, raise_on_close=True)
    resp = fdh.post("status", sid(1), CallStatus="completed")
    assert resp["statusCode"] == 200 and "<Response />" in resp["body"]
    assert fdh.session(sid(1)).ended
    line = [r for r in caplog.records if r.getMessage() == "human_leg_close_failed"][-1]
    assert getattr(line, "decision", None) == "ignored_best_effort"
    assert getattr(line, "call_sid", None) == sid(1)
    assert "close_claimed_at" not in fdh.state(sid(1)).escalation


def test_ai_only_calls_are_unchanged(fdh, ops):
    fdh.login_thomas(sid(1))
    callstate.mutate(fdh.store, sid(1), lambda s: {"ai_sessions": 1})
    with_closer(fdh, ops)
    fdh.post("status", sid(1), CallStatus="completed")
    transcription(fdh, sid(1), "transcription-stopped")
    assert ops == []  # no Orchestrator request of any kind
    assert fdh.state(sid(1)).escalation is None  # nothing recorded for the human leg


def test_transcription_content_is_acknowledged_without_reaching_the_closer(fd):
    escalate(fd, sid(1))
    for event in ("transcription-started", "transcription-content", "something-new"):
        resp = transcription(fd, sid(1), event, TranscriptionData='{"transcript":"my card number"}')
        assert resp["statusCode"] == 204 and resp["body"] == ""
    assert fd.closer.transcriptions == []
    transcription(fd, sid(1), "transcription-stopped")
    assert fd.closer.transcriptions == [(sid(1), None, humanleg.TRANSCRIPTION_STOPPED)]


def test_transcription_route_with_a_bad_signature_is_403(fd):
    params = {"CallSid": sid(1), "TranscriptionEvent": "transcription-stopped"}
    resp = fd.app.handle(make_event("transcription", params, signature="bogus"))
    assert resp["statusCode"] == 403
    assert fd.closer.transcriptions == []


def test_transcription_route_failure_is_logged_and_still_204(caplog):
    from tests.component.test_frontdoor_support import RecordingCloser
    h = build(closer=RecordingCloser(fail=True))
    escalate(h, sid(1))
    resp = transcription(h, sid(1), "transcription-stopped")
    assert resp["statusCode"] == 204
    assert any(r.getMessage() == "conversation close failed" for r in caplog.records)


def test_the_noop_closer_takes_transcription_events():
    NoopConversationCloser().transcription_ended(call_sid=sid(1), conversation_id=None,
                                                 event=humanleg.TRANSCRIPTION_STOPPED)


# --------------------------------------------------------------------------- accepts

from lambdas.frontdoor import app as frontdoor_app  # noqa: E402
from lambdas.frontdoor.handler import human_leg_transcriptions  # noqa: E402
from owlbank.humanleg_client import TwilioCalls  # noqa: E402

from tests.component.test_frontdoor_support import BASIC, HOST, STAGE, twilio_post  # noqa: E402

#: The transcription's status callback: absolute, with the Basic credentials (http.authenticate).
TRANSCRIPTION_URL = f"https://{BASIC[0]}:{BASIC[1]}@{HOST}{STAGE}/voice/transcription"


class FakeAccept:
    """Orchestrator and the Calls API as the accept sees them; every request into ``ops``. A worker
    already present makes the add return ``False`` (what the client does on "listed" or 409)."""

    def __init__(self, ops: list, *, add_ok: bool = True, start_ok: bool = True,
                 conversation: str | None = CONV) -> None:
        self.ops, self.add_ok, self.start_ok, self.conversation = ops, add_ok, start_ok, conversation
        self.present: list[str] = []

    def resolve_conversation(self, call_sid: str) -> str | None:
        self.ops.append(("resolve", call_sid))
        return self.conversation

    def mark_handoff(self, conversation_id: str, metadata) -> bool:
        self.ops.append(("mark", conversation_id, dict(metadata)))
        return True

    def add_human_participant(self, conversation_id: str, worker_sid: str, identity: str) -> bool:
        self.ops.append(("add", conversation_id, worker_sid, identity))
        if not self.add_ok or worker_sid in self.present:
            return False
        self.present.append(worker_sid)
        return True

    def close(self, conversation_id: str) -> bool:
        self.ops.append(("close", conversation_id))
        return True

    def start_transcription(self, call_sid: str, params) -> bool:
        self.ops.append(("start", call_sid, dict(params)))
        return self.start_ok

    def stop_transcription(self, call_sid: str, name: str) -> bool:
        self.ops.append(("stop", call_sid, name))
        return True


def with_accept(h, ops: list, *, switch: bool = False, agent_legs: bool = True, **kw) -> FakeAccept:
    fake = FakeAccept(ops, **kw)
    with_transcription(h, switch, agent_legs=agent_legs)
    h.app.d = replace(h.app.d, human_leg=fake, transcriptions=fake)
    return fake


def accepted(attrs: dict, worker: str, n: int = 1, *, name: str = "Alice Agent", **extra: str) -> dict:
    return {"EventType": "reservation.accepted", "EventSid": "EA" + f"{n:032x}",
            "AccountSid": "AC" + "b" * 32, "WorkspaceSid": "WS" + "c" * 32, "TaskSid": TASK,
            "TaskAttributes": json.dumps(attrs), "WorkerSid": worker, "WorkerName": name,
            "ReservationSid": "WR" + f"{n:032x}", "TaskAge": "12", **extra}


def deliver(h, params: dict) -> dict:
    resp = twilio_post(h.app.handle, "taskrouter", params)
    assert resp == {"statusCode": 204, "headers": {"Content-Type": "application/json"}, "body": ""}
    return resp


def escalated(h, n: int = 1) -> dict:
    return task_attrs(escalate(h, sid(n)))


def named(ops: list, kind: str) -> list:
    return [o for o in ops if o[0] == kind]


@pytest.mark.parametrize("switch", [False, True])
def test_first_accept_marks_the_handoff_and_adds_one_participant(fdh, ops, switch):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=switch)
    deliver(fdh, accepted(attrs, W_A))
    assert ops[0] == ("mark", CONV, {"handoff_at": str(int(CLOCK)), "handoff_to": "human"})
    assert ops[1] == ("add", CONV, W_A, "Alice Agent")
    esc = fdh.state(sid(1)).escalation
    assert esc["participants"] == [{"worker_sid": W_A, "added_at": int(CLOCK)}]
    assert esc["task_sid"] == TASK and esc["conversation_id"] == CONV and esc["handoff_at"] == int(CLOCK)
    assert esc.get("participant_claims") == {}
    if switch:
        (start,) = named(ops, "start")
        assert ops.index(start) == 2  # after the hand-off metadata and the participant
        assert start[1] == sid(1)
        assert start[2]["StatusCallbackUrl"] == TRANSCRIPTION_URL
        assert start[2]["Name"] == humanleg.TRANSCRIPTION_NAME == esc["transcriptions"][sid(1)]["name"]
        assert start[2]["ConversationId"] == CONV and start[2]["Track"] == "inbound_track"
        assert start[2]["InboundTrackLabel"] == "customer" and "OutboundTrackLabel" not in start[2]
        assert (start[2]["LanguageCode"], start[2]["TranscriptionEngine"], start[2]["SpeechModel"]) == (
            "en-US", "deepgram", "nova-3-general")
    else:
        assert named(ops, "start") == [] and "transcriptions" not in esc


def test_a_transfer_adds_the_second_worker_and_keeps_the_first(fdh, ops):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True)
    deliver(fdh, accepted(attrs, W_A, 1))
    deliver(fdh, accepted(attrs, W_B, 2, name="Bob Manager"))
    assert named(ops, "add") == [("add", CONV, W_A, "Alice Agent"), ("add", CONV, W_B, "Bob Manager")]
    assert len(named(ops, "mark")) == 1 and len(named(ops, "start")) == 1  # first accept only
    assert [p["worker_sid"] for p in fdh.state(sid(1)).escalation["participants"]] == [W_A, W_B]


def test_a_replayed_event_and_a_second_accept_by_the_same_worker_send_nothing(fdh, ops):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True)
    deliver(fdh, accepted(attrs, W_A, 1))
    done = len(ops)
    deliver(fdh, accepted(attrs, W_A, 1))  # Twilio retry, same EventSid
    deliver(fdh, accepted(attrs, W_A, 3))  # the same worker again (transferred back)
    assert len(ops) == done
    assert len(fdh.state(sid(1)).escalation["participants"]) == 1


def test_an_accept_for_a_call_that_never_escalated_is_ignored(fdh, ops):
    fdh.login_thomas(sid(1))
    with_accept(fdh, ops, switch=True)
    deliver(fdh, accepted({"call_sid": sid(1), "conversation_id": CONV}, W_A))
    deliver(fdh, accepted({"conversation_id": CONV}, W_A, 2))  # no call SID at all
    assert ops == [] and fdh.state(sid(1)).escalation is None


def test_an_accept_after_close_writes_nothing(fdh, ops, caplog):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True)
    human_leg(fdh, sid(1), conversation_closed_at=int(CLOCK))
    before = dict(fdh.state(sid(1)).escalation)
    caplog.clear()
    deliver(fdh, accepted(attrs, W_A))
    assert ops == [] and dict(fdh.state(sid(1)).escalation) == before
    assert [r for r in caplog.records if r.getMessage() == "human_leg_after_close"]


def test_switch_off_ignores_accepts(fd, ops):
    attrs = escalated(fd)
    deliver(fd, accepted(attrs, W_A))  # no human-leg client wired
    esc = fd.state(sid(1)).escalation
    assert "participants" not in esc and "handoff_at" not in esc


def test_an_accept_with_a_bad_signature_is_rejected(fdh, ops):
    attrs = escalated(fdh)
    with_accept(fdh, ops)
    resp = fdh.app.handle(make_event("taskrouter", accepted(attrs, W_A), signature="bogus"))
    assert resp["statusCode"] == 403 and ops == []


def test_a_failed_add_answers_204_logs_and_is_left_for_the_repair(fdh, ops, caplog):
    attrs = escalated(fdh)
    with_accept(fdh, ops, add_ok=False)
    deliver(fdh, accepted(attrs, W_A))
    esc = fdh.state(sid(1)).escalation
    assert "participants" not in esc and esc["participant_claims"] == {}
    assert esc["task_sid"] == TASK  # the close repairs from the task's reservations
    line = [r for r in caplog.records if r.getMessage() == "human_leg_accept"][-1]
    assert getattr(line, "participant", None) == "not_added"
    assert all("Alice" not in r.getMessage() and "Alice" not in str(vars(r)) for r in caplog.records)


def test_a_failed_start_releases_the_leg_so_the_close_does_not_wait_and_a_later_event_retries(fdh, ops):
    attrs = escalated(fdh)
    fake = with_accept(fdh, ops, switch=True, start_ok=False)
    deliver(fdh, accepted(attrs, W_A))
    assert len(named(ops, "start")) == 1
    assert "transcriptions" not in fdh.state(sid(1)).escalation
    fake.start_ok = True
    deliver(fdh, accepted(attrs, W_B, 2))  # a released leg may start again
    assert [o[1] for o in named(ops, "start")] == [sid(1), sid(1)]
    assert humanleg.running_legs(fdh.state(sid(1)).escalation) == ((sid(1), humanleg.TRANSCRIPTION_NAME),)
    deliver(fdh, accepted(attrs, W_C, 3))
    assert len(named(ops, "start")) == 2  # running: never a second one on the same leg


def test_an_accept_after_the_call_ended_starts_no_transcription(fdh, ops):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True)
    human_leg(fdh, sid(1), call_ended_at=int(CLOCK))
    deliver(fdh, accepted(attrs, W_A))
    assert named(ops, "start") == [] and len(named(ops, "add")) == 1


def test_an_unknown_conversation_is_looked_up_once(fdh, ops):
    attrs = escalated(fdh)
    attrs["conversation_id"] = None
    fake = with_accept(fdh, ops)
    callstate.mutate(fdh.store, sid(1), lambda s: {"ai_sessions": 1})
    deliver(fdh, accepted(attrs, W_A))
    assert ops[0] == ("resolve", sid(1)) and ops[1][0] == "mark"
    assert fdh.state(sid(1)).escalation["conversation_id"] == CONV
    fake.conversation = None
    deliver(fdh, accepted(attrs, W_B, 2))
    assert named(ops, "resolve") == [("resolve", sid(1))]  # recorded: no second lookup


def test_a_failed_conversation_lookup_after_the_claim_still_records_the_accept(fdh, ops, caplog,
                                                                             monkeypatch):
    """An error in the lookup (a session read, say) after the claim is taken means no
    conversation; the accept is still recorded and the participant claim released."""
    attrs = escalated(fdh)
    attrs["conversation_id"] = None
    with_accept(fdh, ops, switch=True)

    detail = "-".join(["secret", "detail"])  # built at run time: a source line in the stack is not it

    def boom(*_a, **_k):
        raise RuntimeError(detail)

    monkeypatch.setattr(fdh.app, "_accept_conversation", boom)
    deliver(fdh, accepted(attrs, W_A))
    esc = fdh.state(sid(1)).escalation
    assert esc.get("participant_claims") == {}
    assert esc["taskrouter_events"] == ["EA" + f"{1:032x}"]
    assert ops == []  # no conversation: nothing sent
    failed = [r for r in caplog.records if r.getMessage() == "human_leg_accept_failed"]
    assert failed and getattr(failed[-1], "operation", None) == "accept_conversation"
    assert all(detail not in r.getMessage() and detail not in str(vars(r))
               for r in caplog.records)
    assert [r for r in caplog.records if r.getMessage() == "human_leg_accept"]
    deliver(fdh, accepted(attrs, W_A, 2))  # the claim was released: a retry by the worker can add
    assert fdh.state(sid(1)).escalation.get("participant_claims") == {}


def test_a_failed_open_check_after_the_claim_still_records_the_accept(fdh, ops, caplog, monkeypatch):
    """``_still_open`` re-reads call state and can raise; the accept is still recorded."""
    attrs = escalated(fdh)
    with_accept(fdh, ops)

    def boom(*_a, **_k):
        raise RuntimeError("store down")

    monkeypatch.setattr(fdh.app, "_still_open", boom)
    deliver(fdh, accepted(attrs, W_A))
    esc = fdh.state(sid(1)).escalation
    assert esc.get("participant_claims") == {} and "participants" not in esc
    assert esc["conversation_id"] == CONV
    assert [r for r in caplog.records if r.getMessage() == "human_leg_accept_failed"]


def test_no_conversation_found_sends_nothing(fdh, ops, caplog):
    attrs = escalated(fdh)
    attrs["conversation_id"] = None
    with_accept(fdh, ops, switch=True, conversation=None)
    deliver(fdh, accepted(attrs, W_A))
    assert [o[0] for o in ops] == ["resolve"]
    assert [r for r in caplog.records if r.getMessage() == "human_leg_no_conversation"]


def test_a_slow_twilio_still_answers_within_the_budget(fdh, monkeypatch):
    """The real clients over a Twilio that takes 0.3 s per request: the accept stops sending once its
    total budget is spent and still answers 204; each request's timeout fits what is left."""
    import time as _time
    import urllib.request

    attrs = escalated(fdh)
    monkeypatch.setattr(frontdoor_app, "HUMAN_LEG_ACCEPT_BUDGET_S", 0.5)
    seen: list[tuple[str, float]] = []

    class Slow:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def read(self):
            return json.dumps({"id": CONV, "status": "ACTIVE", "metadata": {}, "participants": []}).encode()

    def opener(req: urllib.request.Request, timeout: float):
        seen.append((req.get_method(), timeout))
        _time.sleep(0.3)
        return Slow()

    creds = lambda: ("AC" + "b" * 32, "SK" + "1" * 32, "secret")  # noqa: E731
    client = OrchestratorHumanLeg(creds, opener=opener, timeout_s=2.0)
    calls = TwilioCalls(creds, opener=opener, timeout_s=2.0)
    with_transcription(fdh, True)
    fdh.app.d = replace(fdh.app.d, human_leg=client, transcriptions=calls)
    started = _time.monotonic()
    deliver(fdh, accepted(attrs, W_A))
    elapsed = _time.monotonic() - started
    assert len(seen) == 2 and elapsed < 1.0  # GET + PUT of the hand-off; nothing after the budget
    assert all(t <= 0.5 for _, t in seen)


def test_handler_builds_the_transcription_starter_only_with_the_client():
    secret = "arn:aws:secretsmanager:us-east-1:111122223333:secret:twilio-api-key"
    client = human_leg_client({"OWLBANK_HUMAN_LEG_CONVERSATION": "on"}, secret)
    assert isinstance(human_leg_transcriptions(secret, client), TwilioCalls)
    assert human_leg_transcriptions(secret, None) is None
    assert human_leg_transcriptions(None, client) is None


def test_a_failed_close_still_lists_the_repaired_worker(fdh, ops):
    """The repair's add is recorded even when the close is refused, so a later accept or
    close does not add the same worker again."""
    escalate(fdh, sid(1))
    human_leg(fdh, sid(1), task_sid=TASK, conversation_id=CONV)
    with_closer(fdh, ops, rows=[Reservation(W_A, "accepted")], close_ok=False)
    fdh.post("status", sid(1), CallStatus="completed")
    esc = fdh.state(sid(1)).escalation
    assert esc["participants"] == [{"worker_sid": W_A, "added_at": int(CLOCK)}]
    assert "conversation_closed_at" not in esc


def test_an_accept_while_the_close_is_under_way_leaves_the_worker_to_its_repair(fdh, ops):
    attrs = escalated(fdh)
    with_accept(fdh, ops)
    human_leg(fdh, sid(1), call_ended_at=int(CLOCK), close_claimed_at=int(CLOCK))
    deliver(fdh, accepted(attrs, W_A))
    assert ops == []
    assert fdh.state(sid(1)).escalation["taskrouter_events"] == ["EA" + f"{1:032x}"]


# --------------------------------------------------------------------------- the close's journal event


def _closed_events(h, call_sid: str) -> list:
    return [e for e in h.app.journal.read(call_sid) if e["type"] == "conversation_closed"]


@pytest.mark.parametrize("transcribed", [False, True])
def test_a_close_journals_one_conversation_closed_shown_in_the_timeline(fdh, ops, transcribed):
    """The close writes one ``conversation_closed`` (allow-listed fields), and the Context API
    timeline shows it, with ``closed_by``, as its only new entry; repeats write nothing more."""
    from lambdas.context_api.handler import build_timeline

    escalate(fdh, sid(1))
    human_leg(fdh, sid(1), task_sid=TASK, conversation_id=CONV,
              **({"transcriptions": customer_leg(sid(1))} if transcribed else {}))
    with_closer(fdh, ops)
    before = build_timeline(fdh.app.journal.read(sid(1)))
    fdh.post("status", sid(1), CallStatus="completed")
    if transcribed:
        assert _closed_events(fdh, sid(1)) == []  # waits for the transcript's end
        transcription(fdh, sid(1), "transcription-stopped")
    middle = build_timeline(fdh.app.journal.read(sid(1)))
    fdh.post("status", sid(1), CallStatus="completed")
    transcription(fdh, sid(1), "transcription-stopped")
    events = _closed_events(fdh, sid(1))
    assert len(events) == 1 and closes(ops) == [("close", CONV)]
    assert {k: v for k, v in events[0].items() if k not in ("seq", "ts", "expires_at")} == {
        "type": "conversation_closed", "call_sid": sid(1), "conversation_id": CONV,
        "closed_by": "transcription" if transcribed else "status"}
    after = build_timeline(fdh.app.journal.read(sid(1)))
    assert after == middle
    new = [e for e in after if e not in before]
    assert [e["type"] for e in new] == ["call_end", "conversation_closed"]
    assert new[-1] == {"seq": events[0]["seq"], "ts": events[0]["ts"], "type": "conversation_closed",
                       "closed_by": "transcription" if transcribed else "status"}


def test_handler_closer_journals_to_the_given_log(fdh):
    secret = "arn:aws:secretsmanager:us-east-1:111122223333:secret:twilio-api-key"
    client = human_leg_client({"OWLBANK_HUMAN_LEG_CONVERSATION": "on"}, secret)
    closer = human_leg_closer({}, secret, fdh.store, client, journal=fdh.app.journal)
    assert closer is not None and closer._journal is fdh.app.journal


# --------------------------------------------------------------------------- the start fails or is slow

from owlbank.humanleg_client import budget_left  # noqa: E402
from owlbank.humanleg_close import CLOSE_DEADLINE_S  # noqa: E402


class BudgetAwareOrchestrator(FakeOrchestrator):
    """Records the budget left when each close request would be sent (as the real client's
    ``_timeout`` cuts it), so a test sees the deadline a close actually gets."""

    def __init__(self, ops: list, **kw) -> None:
        super().__init__(ops, **kw)
        self.close_budgets: list[float | None] = []

    def close(self, conversation_id: str) -> bool:
        self.close_budgets.append(budget_left())
        return super().close(conversation_id)


def _race(fdh, ops, monkeypatch, *, start: str) -> BudgetAwareOrchestrator:
    """The call ends (``/voice/status completed``) while the transcription start is in flight; the
    start then fails (``fails``: returns False after a delay longer than the accept's budget;
    ``raises``: raises)."""
    import time as _time

    attrs = escalated(fdh)
    monkeypatch.setattr(frontdoor_app, "HUMAN_LEG_ACCEPT_BUDGET_S", 0.2)
    fake = with_accept(fdh, ops, switch=True)
    orchestrator = BudgetAwareOrchestrator(ops)
    closer = HumanLegCloser(fdh.store, orchestrator, calls=FakeCalls(ops),
                            reservations=FakeReservations(ops, []), clock=lambda: CLOCK,
                            journal=fdh.app.journal)
    fdh.app.d = replace(fdh.app.d, closer=closer)

    def start_transcription(call_sid: str, params) -> bool:
        ops.append(("start", call_sid, dict(params)))
        fdh.post("status", call_sid, CallStatus="completed")  # the caller hangs up meanwhile
        assert closes(ops) == []  # that close waits for this transcript
        if start == "raises":
            raise RuntimeError("calls api down")
        _time.sleep(0.3)  # slower than the whole accept budget
        return False

    fake.start_transcription = start_transcription  # type: ignore[method-assign]
    deliver(fdh, accepted(attrs, W_A))
    return orchestrator


@pytest.mark.parametrize("start", ["fails", "raises"])
def test_call_ending_during_a_failed_start_closes_once_with_its_full_deadline(fdh, ops, monkeypatch,
                                                                              caplog, start):
    caplog.set_level("INFO")
    orchestrator = _race(fdh, ops, monkeypatch, start=start)
    assert closes(ops) == [("close", CONV)]
    # The close ran after the accept's budget: its own CLOSE_DEADLINE_S, not what was left of 0.2 s.
    (left,) = orchestrator.close_budgets
    assert left is not None and left > CLOSE_DEADLINE_S - 0.5
    esc = fdh.state(sid(1)).escalation
    assert esc["conversation_closed_at"] == int(CLOCK)
    assert "transcriptions" not in esc
    assert len(_closed_events(fdh, sid(1))) == 1
    line = next(r for r in caplog.records if r.getMessage() == "human_leg_close_after_start")
    assert getattr(line, "deadline_s", None) == CLOSE_DEADLINE_S
    # The accept is recorded all the same: the worker listed, its claim released.
    assert esc["participants"] == [{"worker_sid": W_A, "added_at": int(CLOCK)}]
    assert W_A not in (esc.get("participant_claims") or {})
    # Later signals send nothing more.
    fdh.post("status", sid(1), CallStatus="completed")
    transcription(fdh, sid(1), "transcription-stopped")
    assert closes(ops) == [("close", CONV)] and len(_closed_events(fdh, sid(1))) == 1


def test_a_raising_start_still_records_the_accept(fdh, ops, caplog):
    attrs = escalated(fdh)
    fake = with_accept(fdh, ops, switch=True)

    def boom(call_sid: str, params) -> bool:
        raise RuntimeError("calls api down")

    fake.start_transcription = boom  # type: ignore[method-assign]
    deliver(fdh, accepted(attrs, W_A))
    esc = fdh.state(sid(1)).escalation
    assert esc["participants"] == [{"worker_sid": W_A, "added_at": int(CLOCK)}]
    assert W_A not in (esc.get("participant_claims") or {})
    assert "transcriptions" not in esc
    assert [r for r in caplog.records if r.getMessage() == "human_leg_transcription_start_failed"]
    line = next(r for r in caplog.records if r.getMessage() == "human_leg_accept")
    assert getattr(line, "transcriptions_started", None) == 0


def test_a_raising_catalog_read_still_records_the_accept(fdh, ops, monkeypatch):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True)

    def boom(*a, **k):
        raise RuntimeError("catalog unreadable")

    monkeypatch.setattr(fdh.app.d.catalog, "settings", boom)
    deliver(fdh, accepted(attrs, W_A))
    esc = fdh.state(sid(1)).escalation
    assert esc["participants"] == [{"worker_sid": W_A, "added_at": int(CLOCK)}]
    assert named(ops, "start") == [] and "transcriptions" not in esc


def test_a_failed_release_records_the_transcript_end_so_the_close_does_not_wait(fdh, ops, monkeypatch,
                                                                                 caplog):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True, start_ok=False)
    with_closer(fdh, ops)

    def boom(*a, **k):
        raise RuntimeError("dynamodb down")

    monkeypatch.setattr(humanleg, "release_transcription", boom)
    deliver(fdh, accepted(attrs, W_A))
    esc = fdh.state(sid(1)).escalation
    assert esc["transcriptions"][sid(1)]["name"] == humanleg.TRANSCRIPTION_NAME
    assert esc["transcriptions"][sid(1)]["ended_at"] == int(CLOCK)
    assert not humanleg.transcription_running(esc)
    assert [r for r in caplog.records if r.getMessage() == "human_leg_transcription_release_failed"]
    fdh.post("status", sid(1), CallStatus="completed")  # closes now: no transcript to wait for
    assert closes(ops) == [("close", CONV)]
    assert named(ops, "stop") == []


# --------------------------------------------------------------------------- agent legs

AGENT_1, AGENT_2 = ("CA" + c * 32 for c in "78")


def with_legs(attrs: dict, *legs: str) -> dict:
    """The task attributes as Flex writes them once the conference formed."""
    participants = {"customer": attrs["call_sid"]}
    participants.update({("worker" if i == 0 else f"worker{i}"): leg for i, leg in enumerate(legs)})
    return {**attrs, "conference": {"sid": "CF" + "f" * 32, "participants": participants}}


def updated(attrs: dict, n: int = 1, **extra: str) -> dict:
    return {"EventType": "task.updated", "EventSid": "EU" + f"{n:032x}",
            "AccountSid": "AC" + "b" * 32, "WorkspaceSid": "WS" + "c" * 32, "TaskSid": TASK,
            "TaskAttributes": json.dumps(attrs), "TaskAge": "30", **extra}


def leg_stopped(h, leg: str, call: str, event: str = "transcription-stopped") -> dict:
    """An agent leg's status callback: ``CallSid`` is the leg, ``call`` the customer call."""
    params = {"CallSid": leg, "TranscriptionEvent": event, "TranscriptionSid": "GT" + "e" * 32}
    return h.app.handle(make_event(f"transcription?call={call}", params))


def starts_on(ops: list, leg: str) -> list:
    return [o for o in named(ops, "start") if o[1] == leg]


def test_escalation_with_one_agent_starts_the_customer_leg_and_the_agent_leg(fdh, ops):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True)
    deliver(fdh, accepted(attrs, W_A))
    assert [o[1] for o in named(ops, "start")] == [sid(1)]
    deliver(fdh, updated(with_legs(attrs, AGENT_1)))  # Flex writes the agent leg after the accept
    starts = named(ops, "start")
    assert [o[1] for o in starts] == [sid(1), AGENT_1]
    agent = starts[1][2]
    assert (agent["Track"], agent["InboundTrackLabel"], agent["Name"], agent["ConversationId"]) == (
        "inbound_track", "agent", humanleg.TRANSCRIPTION_NAME, CONV)
    assert agent["StatusCallbackUrl"] == f"{TRANSCRIPTION_URL}?call={sid(1)}"
    assert (agent["LanguageCode"], agent["TranscriptionEngine"], agent["SpeechModel"]) == (
        "en-US", "deepgram", "nova-3-general")
    legs = humanleg.transcriptions(fdh.state(sid(1)).escalation)
    assert {k: v["role"] for k, v in legs.items()} == {sid(1): "customer", AGENT_1: "agent"}


def test_an_agent_leg_already_in_the_accept_starts_with_the_customer_leg(fdh, ops):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True)
    deliver(fdh, accepted(with_legs(attrs, AGENT_1), W_A))
    assert [o[1] for o in named(ops, "start")] == [sid(1), AGENT_1]
    deliver(fdh, updated(with_legs(attrs, AGENT_1)))
    assert len(named(ops, "start")) == 2


def test_a_transfer_starts_the_second_agent_leg_and_the_first_agents_stop_does_not_close(fdh, ops):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True)
    with_closer(fdh, ops)
    deliver(fdh, accepted(attrs, W_A, 1))
    deliver(fdh, updated(with_legs(attrs, AGENT_1), 1))
    deliver(fdh, accepted(with_legs(attrs, AGENT_1), W_B, 2, name="Bob Manager"))
    deliver(fdh, updated(with_legs(attrs, AGENT_1, AGENT_2), 2))  # the manager's leg joins
    assert [o[1] for o in named(ops, "start")] == [sid(1), AGENT_1, AGENT_2]
    resp = leg_stopped(fdh, AGENT_1, sid(1))  # the first agent leaves (a transfer)
    assert resp["statusCode"] == 204
    assert closes(ops) == []
    esc = fdh.state(sid(1)).escalation
    assert esc["transcriptions"][AGENT_1]["ended_at"] == int(CLOCK)
    assert esc["transcriptions"][AGENT_2]["ended_at"] is None and "call_ended_at" not in esc
    deliver(fdh, updated(with_legs(attrs, AGENT_2), 3))  # Flex drops the first leg: nothing to start
    assert len(named(ops, "start")) == 3
    fdh.post("status", sid(1), CallStatus="completed")
    assert closes(ops) == []  # the customer and the manager legs still run
    transcription(fdh, sid(1), "transcription-stopped")
    assert closes(ops) == []  # the manager's leg still runs
    leg_stopped(fdh, AGENT_2, sid(1))
    assert closes(ops) == [("close", CONV)]
    assert named(ops, "stop") == []  # every leg reported its own end


def test_the_sweep_stops_every_running_leg_on_its_own_call_before_the_close(fdh, ops):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True)
    closer = with_closer(fdh, ops)
    deliver(fdh, accepted(with_legs(attrs, AGENT_1), W_A))
    assert closer.end(sid(1), humanleg.SWEEP) is True
    kinds = [o[0] for o in ops]
    assert ("stop", sid(1), "humanleg") in ops and ("stop", AGENT_1, "humanleg") in ops
    assert kinds.index("close") > max(i for i, k in enumerate(kinds) if k == "stop")


def test_a_duplicate_task_updated_starts_nothing_and_writes_nothing(fdh, ops):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True)
    deliver(fdh, accepted(attrs, W_A))
    deliver(fdh, updated(with_legs(attrs, AGENT_1), 1))
    before, done = fdh.state(sid(1)), len(ops)
    deliver(fdh, updated(with_legs(attrs, AGENT_1), 1))  # Twilio retry
    deliver(fdh, updated(with_legs(attrs, AGENT_1), 2))  # another update, same legs
    after = fdh.state(sid(1))
    assert len(ops) == done and after == before


@pytest.mark.parametrize("case", ["no_call_sid", "no_state", "not_escalated", "before_handoff",
                                  "call_ended", "closed"])
def test_unrelated_task_updates_are_ignored_without_a_state_write(fdh, ops, monkeypatch, case):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True)
    if case not in ("before_handoff", "not_escalated"):
        deliver(fdh, accepted(attrs, W_A))
    if case == "call_ended":
        human_leg(fdh, sid(1), call_ended_at=int(CLOCK))
    if case == "closed":
        human_leg(fdh, sid(1), conversation_closed_at=int(CLOCK))
    event = with_legs(attrs, AGENT_1)
    if case == "no_call_sid":
        event = {k: v for k, v in event.items() if k != "call_sid"}
    if case == "no_state":
        event = with_legs({**attrs, "call_sid": sid(9)}, AGENT_1)
    if case == "not_escalated":
        fdh.login_thomas(sid(2))
        event = with_legs({**attrs, "call_sid": sid(2)}, AGENT_1)
    done = len(ops)
    writes: list = []
    real = callstate.mutate
    monkeypatch.setattr(callstate, "mutate", lambda *a, **k: writes.append(a) or real(*a, **k))
    deliver(fdh, updated(event))
    assert writes == [] and len(ops) == done


def test_task_updates_with_the_agent_legs_switch_off_give_one_both_tracks_transcription(fdh, ops):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True, agent_legs=False)
    deliver(fdh, accepted(with_legs(attrs, AGENT_1), W_A))
    deliver(fdh, updated(with_legs(attrs, AGENT_1, AGENT_2)))
    (start,) = named(ops, "start")
    assert start[1] == sid(1) and start[2]["Track"] == "both_tracks"
    assert (start[2]["InboundTrackLabel"], start[2]["OutboundTrackLabel"]) == ("customer", "agent")
    assert list(humanleg.transcriptions(fdh.state(sid(1)).escalation)) == [sid(1)]


def test_task_updates_with_the_transcription_switch_off_start_nothing(fdh, ops):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=False)
    deliver(fdh, accepted(with_legs(attrs, AGENT_1), W_A))
    deliver(fdh, updated(with_legs(attrs, AGENT_1, AGENT_2)))
    assert named(ops, "start") == [] and "transcriptions" not in fdh.state(sid(1)).escalation


def test_a_failed_agent_leg_start_is_released_and_retried_by_the_next_update(fdh, ops):
    attrs = escalated(fdh)
    fake = with_accept(fdh, ops, switch=True)
    deliver(fdh, accepted(attrs, W_A))
    fake.start_ok = False
    deliver(fdh, updated(with_legs(attrs, AGENT_1), 1))
    assert AGENT_1 not in humanleg.transcriptions(fdh.state(sid(1)).escalation)
    fake.start_ok = True
    deliver(fdh, updated(with_legs(attrs, AGENT_1), 2))
    assert len(starts_on(ops, AGENT_1)) == 2
    assert AGENT_1 in dict(humanleg.running_legs(fdh.state(sid(1)).escalation))


def test_no_more_than_six_agent_legs_are_transcribed(fdh, ops):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True)
    deliver(fdh, accepted(attrs, W_A))
    legs = ["CA" + f"{i:032x}" for i in range(1, 9)]
    deliver(fdh, updated(with_legs(attrs, *legs[:4]), 1))
    deliver(fdh, updated(with_legs(attrs, *legs[4:]), 2))
    agent_starts = [o[1] for o in named(ops, "start") if o[1] != sid(1)]
    assert agent_starts == legs[:humanleg.MAX_AGENT_LEGS]


def test_a_task_update_with_a_bad_signature_is_rejected(fdh, ops):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True)
    deliver(fdh, accepted(attrs, W_A))
    done = len(ops)
    resp = fdh.app.handle(make_event("taskrouter", updated(with_legs(attrs, AGENT_1)), signature="bogus"))
    assert resp["statusCode"] == 403 and len(ops) == done


@pytest.mark.parametrize("call", ["forged", "other_call", "bad", "empty"])
def test_a_leg_callback_with_a_forged_or_unknown_call_changes_nothing(fdh, ops, call):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True)
    with_closer(fdh, ops)
    deliver(fdh, accepted(with_legs(attrs, AGENT_1), W_A))
    escalate(fdh, sid(2))  # another escalated call, which does not record AGENT_1
    fdh.post("status", sid(1), CallStatus="completed")
    transcription(fdh, sid(1), "transcription-stopped")
    before = (fdh.state(sid(1)), fdh.state(sid(2)))
    target = {"forged": sid(9), "other_call": sid(2), "bad": "not-a-call", "empty": ""}[call]
    resp = leg_stopped(fdh, AGENT_1, target)
    assert resp["statusCode"] == 204
    assert closes(ops) == [] and (fdh.state(sid(1)), fdh.state(sid(2))) == before
    leg_stopped(fdh, AGENT_1, sid(1))  # the real callback closes
    assert closes(ops) == [("close", CONV)]


def test_a_customer_leg_callback_names_no_call_and_ends_that_leg_only(fdh, ops):
    attrs = escalated(fdh)
    with_accept(fdh, ops, switch=True)
    with_closer(fdh, ops)
    deliver(fdh, accepted(with_legs(attrs, AGENT_1), W_A))
    transcription(fdh, sid(1), "transcription-error")
    legs = humanleg.transcriptions(fdh.state(sid(1)).escalation)
    assert legs[sid(1)]["ended_at"] == int(CLOCK) and legs[AGENT_1]["ended_at"] is None
    # An agent leg's callback without ``call`` names its own call SID: no state, nothing changes.
    before = fdh.state(sid(1))
    transcription(fdh, AGENT_1, "transcription-stopped")
    assert fdh.state(sid(1)) == before and closes(ops) == []


def test_the_route_passes_the_leg_and_the_customer_call_to_the_closer(fd):
    escalate(fd, sid(1))
    leg_stopped(fd, AGENT_1, sid(1))
    transcription(fd, sid(1), "transcription-stopped")
    assert fd.closer.transcriptions == [(sid(1), None, humanleg.TRANSCRIPTION_STOPPED)] * 2
    assert fd.closer.legs == [AGENT_1, sid(1)]



# --------------------------------------------------------------------------- characterization: the Flex escalation


def _spy_escalate(h, monkeypatch) -> list[str]:
    """Step names of what ``_escalate`` does, in order. Recording starts when ``_escalate``
    is entered, so the earlier hand-off and login steps don't appear."""
    from lambdas.frontdoor import app as app_mod
    from owlbank import tokens

    app = h.app
    steps: list[str] = []
    live = [False]

    def wrap(obj, name, label, *, only=None):
        orig = getattr(obj, name)

        def spy(*a, **kw):
            if live[0] and (only is None or only(a, kw)):
                steps.append(label)
            return orig(*a, **kw)
        monkeypatch.setattr(obj, name, spy)

    orig_escalate = app._escalate

    def escalate_spy(*a, **kw):
        live[0] = True
        try:
            return orig_escalate(*a, **kw)
        finally:
            live[0] = False
    monkeypatch.setattr(app, "_escalate", escalate_spy)
    wrap(app, "_handoff_conversation", "conversation_handoff")
    wrap(app_mod, "build_task_attributes", "build_task_attributes")
    wrap(app, "_update", "state_update", only=lambda a, kw: "escalation" in kw)
    wrap(tokens, "revoke_call", "revoke_call")
    wrap(app, "_prompts", "spoken_items")
    wrap(app, "_log_event", "journal_escalation",
         only=lambda a, kw: a[1].get("type") == "escalation")
    wrap(app, "_journal_handoff", "journal_handoff")
    return steps


def test_flex_escalation_side_effects_happen_in_this_order(fdh, monkeypatch):
    steps = _spy_escalate(fdh, monkeypatch)
    escalate(fdh, sid(1))
    assert steps == ["conversation_handoff", "build_task_attributes", "state_update",
                     "revoke_call", "spoken_items", "journal_escalation", "journal_handoff"]


def test_flex_escalation_task_attributes_are_pinned(fdh):
    root = escalate(fdh, sid(1), conversationId=CONV)
    assert task_attrs(root) == {
        "ai_summary": "Card lost.", "auth_level": 1, "call_sid": sid(1), "conversation_id": CONV,
        "customer_id": "C-1001", "escalated_at": "2025-10-09T08:53:20Z",
        "escalation_reason": "fraud_suspected", "factors_verified": ["ani_match", "zip"],
        "from": "+12025550101", "identity_source": "ani", "journal_ref": "jr-7f3a",
        "language": "en-US", "last_tool_attempted": "escalate", "name": "Thomas Doe",
        "profile_id": None, "test": False, "tools_run": [], "type": "inbound",
    }


def test_flex_escalation_twiml_is_pinned(fdh):
    root = escalate(fdh, sid(1), conversationId=CONV)
    assert root.tag == "Response" and [c.tag for c in root] == ["Say", "Enqueue"]
    assert says(root) == ["I'll connect you with our fraud team right away."]
    enq = root.find("Enqueue")
    assert enq.attrib == {"workflowSid": WORKFLOW, "action": ENQUEUE_ACTION_URL}
    assert [c.tag for c in enq] == ["Task"] and enq.find("Task").attrib == {"priority": "100"}


def test_flex_escalation_journals_one_escalation_then_the_human_handoff(fdh):
    escalate(fdh, sid(1), conversationId=CONV)
    events = [e for e in fdh.journal(sid(1)) if e["type"] in ("escalation", "handoff")
              and e.get("to", "human") == "human"]
    assert [e["type"] for e in events] == ["escalation", "handoff"]
    assert events[0]["reason"] == "fraud_suspected" and events[0]["journal_ref"] == "jr-7f3a"
    assert events[1]["conversation_id"] == CONV


# --------------------------------------------------------------------------- risk escalation


def _risk_inputs(h, call_sid: str):
    from lambdas.frontdoor import http
    h.login_thomas(call_sid)
    req = http.TwilioRequest("factor", {"CallSid": call_sid}, "", origin="https://h.example")
    return req, h.state(call_sid), h.session(call_sid)


def test_a_risk_escalation_resets_to_level_zero_before_the_handoff(fdh, monkeypatch):
    """The journal logs the pre-reset level; the state (and so the briefing and task) say 0."""
    req, st, sess = _risk_inputs(fdh, sid(1))
    assert st.level == 1
    steps: list[str] = []
    app = fdh.app
    for name, label, only in (
        ("_update", "reset_level_0", lambda a, kw: kw.get("level") == 0),
        ("_update", "escalation_update", lambda a, kw: "escalation" in kw),
        ("_journal_handoff", "journal_handoff", None),
    ):
        orig = getattr(app, name)

        def spy(*a, _o=orig, _l=label, _only=only, **kw):
            if _only is None or _only(a, kw):
                steps.append(_l)
            return _o(*a, **kw)
        monkeypatch.setattr(app, name, spy)
    root = ET.fromstring(app._risk_escalate(req, st, sess, "precheck").encode())
    assert steps == ["reset_level_0", "escalation_update", "journal_handoff"]
    assert fdh.state(sid(1)).level == 0
    assert task_attrs(root)["auth_level"] == 0 and task_attrs(root)["factors_verified"] == []
    assert task_attrs(root)["escalation_reason"] == "verification_risk"


def test_pre_items_are_spoken_in_order_before_the_route_phrase(fdh):
    req, st, sess = _risk_inputs(fdh, sid(1))
    app = fdh.app
    pre = [app._say("verify.failed", st.language), app._say("error.lookup_unavailable", st.language)]
    root = ET.fromstring(app._escalate(req, st, sess, "verification_risk", "", pre=pre).encode())
    spoken = says(root)
    first, second = (s.text for _, s in pre)
    assert [c.tag for c in root] == ["Say", "Say", "Say", "Enqueue"]
    assert spoken[:2] == [first, second] and len(spoken) == 3
    assert spoken[2] not in (first, second)
