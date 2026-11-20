"""scripts/rehearse_human_leg.py: safety flags, caps, TwiML, transcription start, long wait,
cleanup after a failure at every step, the report.

A stateful fake Twilio (:class:`FakeWorld`) stands in for the script's :class:`Api`; :class:`Failing` injects a
failure at one request (before or after its effect) or makes one operation always fail. The REST adapter is
checked against a recording opener. Nothing here reaches Twilio.
"""

from __future__ import annotations

import io
import itertools
import json
import urllib.error
import urllib.parse
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from owlbank import humanleg
from scripts import rehearse_human_leg as rh

ACCOUNT = "AC" + "a" * 32
WS = "WS" + "b" * 32
WW = "WW" + "c" * 32
CONFIG = "conv_configuration_test"
FROM, TO, AGENT, MANAGER = "+12065550100", "+12065550101", "+12065550102", "+12065550103"
STATUS_URL = "https://api.owlbank.example/voice/transcription"
CALLBACK_URL = "https://api.owlbank.example/voice/taskrouter"
OTHER_CALL = "CA" + "9" * 32
OTHER_TASK = "WT" + "9" * 32
STT = humanleg.SttSettings(engine="deepgram", speech_model="nova-3-general", language_code="en-US")


def sid(prefix: str, n: int) -> str:
    return f"{prefix}{n:032x}"


# --------------------------------------------------------------------------- fake Twilio


class FakeWorld:
    """Calls, tasks, conferences and conversations with Twilio's transitions, and a log of every request.

    Strict on purpose: ending one call ends no other, and a call that ends leaves its task open, so the
    script's cleanup must close everything itself. ``find_tasks`` ignores its filter (``leaky``) unless
    told otherwise, so the script's own check is what keeps the other task out.
    """

    def __init__(self, *, mode: str = "standalone", owned: tuple[str, ...] = (FROM, TO, AGENT, MANAGER),
                 account_status: str = "active", reservation_delay: int = 0, reservations: bool = True,
                 leaky: bool = True, alerts: list[dict[str, Any]] | None = None,
                 callback_url: str | None = CALLBACK_URL) -> None:
        self.mode = mode
        self.owned = set(owned)
        self.account_status = account_status
        self.reservation_delay = reservation_delay
        self.reservations = reservations
        self.leaky = leaky
        self.alert_rows = alerts or []
        self.callback = callback_url
        self.clock = 1_800_000_000.0
        self.ids = itertools.count(1)
        self.calls: dict[str, dict[str, Any]] = {OTHER_CALL: {"from": "+12065550199", "to": TO,
                                                              "direction": "inbound", "status": "in-progress",
                                                              "start": self.clock - 600}}
        self.tasks: dict[str, dict[str, Any]] = {OTHER_TASK: {"call_sid": OTHER_CALL, "status": "pending"}}
        self.conferences: dict[str, dict[str, Any]] = {}
        self.conversations: dict[str, dict[str, Any]] = {}
        self.transcriptions: list[tuple[str, dict[str, str]]] = []
        self.log: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.touched_tasks: set[str] = set()

    # time
    def now(self) -> float:
        return self.clock

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.clock += seconds

    def _new_call(self, frm: str, to: str, direction: str) -> str:
        call = sid("CA", next(self.ids))
        self.calls[call] = {"from": frm, "to": to, "direction": direction, "status": "in-progress",
                            "start": self.clock}
        return call

    def _new_task(self, call_sid: str) -> None:
        self.tasks[sid("WT", next(self.ids))] = {"call_sid": call_sid, "status": "pending", "polls": 0}

    # account and numbers
    def account(self) -> Mapping[str, Any]:
        return {"sid": ACCOUNT, "status": self.account_status}

    def owns_number(self, number: str) -> bool:
        return number in self.owned

    # calls
    def create_call(self, *, to: str, from_: str, twiml: str, time_limit_s: int) -> str:
        self.time_limit = time_limit_s
        self.twiml = twiml
        call = self._new_call(from_, to, "outbound-api")
        if self.mode == "standalone":
            assert "<Enqueue" in twiml
            self._new_task(call)
        else:  # the deployed front door answers, the AI creates the conversation, the caller escalates
            inbound = self._new_call(from_, to, "inbound")
            self.conversations[sid("conv_", next(self.ids))] = {"status": "ACTIVE", "name": None,
                                                                "call_sid": inbound}
            self._new_task(inbound)
        return call

    def find_calls(self, *, from_: str, to: str, since: float) -> list[Mapping[str, Any]]:
        return [{"sid": s, "status": c["status"], "direction": c["direction"]} for s, c in self.calls.items()
                if c["from"] == from_ and c["to"] == to and c["start"] >= since - 60]

    def call_status(self, call_sid: str) -> str:
        return self.calls[call_sid]["status"]

    def end_call(self, call_sid: str) -> None:
        if self.calls[call_sid]["status"] in rh.OPEN_CALL:
            self.calls[call_sid]["status"] = "completed"

    # TaskRouter
    def find_tasks(self, call_sid: str) -> list[Mapping[str, Any]]:
        return [{"sid": s, "status": t["status"], "call_sid": t["call_sid"]} for s, t in self.tasks.items()
                if self.leaky or t["call_sid"] == call_sid]

    def task_status(self, task_sid: str) -> str:
        self.touched_tasks.add(task_sid)
        return self.tasks[task_sid]["status"]

    def set_task_status(self, task_sid: str, status: str) -> None:
        self.touched_tasks.add(task_sid)
        task = self.tasks[task_sid]
        allowed = {"canceled": ("pending", "reserved"), "completed": ("assigned", "wrapping")}[status]
        if task["status"] not in allowed:
            raise rh.ApiError("tasks.update: HTTP 400")
        task["status"] = status

    def pending_reservation(self, task_sid: str) -> str | None:
        self.touched_tasks.add(task_sid)
        task = self.tasks[task_sid]
        task["polls"] = task.get("polls", 0) + 1
        if not self.reservations or task["polls"] <= self.reservation_delay \
                or task["status"] not in ("pending", "reserved"):
            return None
        task["status"] = "reserved"
        return "WR" + task_sid[2:]

    def accept_conference(self, task_sid: str, reservation_sid: str, *, from_: str, to: str) -> None:
        self.touched_tasks.add(task_sid)
        task = self.tasks[task_sid]
        assert task["status"] == "reserved" and reservation_sid == "WR" + task_sid[2:]
        task["status"] = "assigned"
        self.accepted_at = self.clock
        agent = self._new_call(from_, to, "outbound-api")
        self.conferences[sid("CF", next(self.ids))] = {"task": task_sid, "calls": [task["call_sid"], agent]}

    def conference_for_task(self, task_sid: str) -> str | None:
        return next((c for c, v in self.conferences.items() if v["task"] == task_sid), None)

    def conference_calls(self, conference_sid: str) -> list[str]:
        return list(self.conferences[conference_sid]["calls"])

    def add_participant(self, conference_sid: str, *, from_: str, to: str, time_limit_s: int) -> str:
        self.participant_time_limit = time_limit_s
        call = self._new_call(from_, to, "outbound-api")
        self.conferences[conference_sid]["calls"].append(call)
        return call

    def task_events(self, task_sid: str) -> list[Mapping[str, Any]]:
        return [{"event_type": "task.created", "event_date": "2026-11-05T10:00:00Z"},
                {"event_type": "reservation.accepted", "event_date": "2026-11-05T10:00:05Z"}]

    def callback_url(self) -> str | None:
        return self.callback

    def alerts(self, since: float, until: float) -> list[Mapping[str, Any]]:
        return list(self.alert_rows)

    # Orchestrator
    def create_conversation(self, configuration_id: str, name: str) -> str:
        cid = sid("conv_", next(self.ids))
        self.conversations[cid] = {"status": "ACTIVE", "name": name, "call_sid": None}
        return cid

    def find_conversation(self, configuration_id: str, name: str) -> str | None:
        return next((c for c, v in self.conversations.items() if v["name"] == name and v["status"] == "ACTIVE"),
                    None)

    def resolve_conversation(self, call_sid: str) -> str | None:
        found = [c for c, v in self.conversations.items() if v["call_sid"] == call_sid and v["status"] == "ACTIVE"]
        return found[0] if len(found) == 1 else None

    def conversation(self, conversation_id: str) -> Mapping[str, Any]:
        return {"id": conversation_id, "status": self.conversations[conversation_id]["status"]}

    def participants(self, conversation_id: str) -> list[Mapping[str, Any]]:
        return [{"type": "CUSTOMER", "addresses": [{"channel": "VOICE", "address": FROM}]}]

    def communications(self, conversation_id: str) -> list[Mapping[str, Any]]:
        return [{"author": {"address": "agent", "channel": "VOICE"}, "content": {"text": "Hello, how can I help?"}}]

    def close_conversation(self, conversation_id: str) -> bool:
        self.conversations[conversation_id]["status"] = "CLOSED"
        return True

    def start_transcription(self, call_sid: str, params: Mapping[str, str]) -> bool:
        self.transcriptions.append((call_sid, dict(params)))
        return True

    def stop_transcription(self, call_sid: str, name: str) -> bool:
        return True

    # checks
    def open_things(self) -> dict[str, list[str]]:
        return {"calls": [s for s, c in self.calls.items() if s != OTHER_CALL and c["status"] in rh.OPEN_CALL],
                "tasks": [s for s, t in self.tasks.items() if s != OTHER_TASK and t["status"] in rh.OPEN_TASK],
                "conversations": [c for c, v in self.conversations.items() if v["status"] == "ACTIVE"]}


#: The shared clients never raise: these answer a failure with a value (as the real ones do).
FAILURE_VALUE: dict[str, Any] = {"close_conversation": False, "start_transcription": False,
                                 "stop_transcription": False, "resolve_conversation": None}


class Failing:
    """Wraps a :class:`FakeWorld` as the script's :class:`Api`; logs every request in order.

    ``fail_at``: the index of the one request that fails; ``kind`` ``before`` (no effect) or ``after``
    (the effect happens, the answer is lost). ``always``: an operation that fails every time.
    """

    def __init__(self, world: FakeWorld, *, fail_at: int | None = None, kind: str = "before",
                 always: str | None = None) -> None:
        self.world = world
        self.fail_at = fail_at
        self.kind = kind
        self.always = always
        self.count = 0

    def __getattr__(self, name: str) -> Callable[..., Any]:
        method = getattr(self.world, name)

        def call(*args: Any, **kwargs: Any) -> Any:
            index = self.count
            self.count += 1
            self.world.log.append((name, args, kwargs))
            failing = index == self.fail_at or name == self.always
            if not failing:
                return method(*args, **kwargs)
            if self.kind == "after" and name != self.always:
                method(*args, **kwargs)
            if name in FAILURE_VALUE:
                return FAILURE_VALUE[name]
            raise rh.ApiError(f"{name}: HTTP 500")

        return call


def argv(mode: str = "standalone", *extra: str, drop: tuple[str, ...] = ()) -> list[str]:
    flags = {"--yes": None, "--expect-account-sid": ACCOUNT, "--account-sid": ACCOUNT, "--from": FROM,
             "--to": TO, "--agent": AGENT, "--manager": MANAGER, "--workspace-sid": WS}
    if mode == "standalone":
        flags |= {"--workflow-sid": WW, "--configuration-id": CONFIG, "--status-callback-url": STATUS_URL}
    out = [mode]
    for flag, value in flags.items():
        if flag in drop:
            continue
        out += [flag] if value is None else [flag, value]
    return out + list(extra)


def run(world: FakeWorld, args: list[str], **failing: Any) -> tuple[int, list[str], Failing]:
    api = Failing(world, **failing)
    lines: list[str] = []
    code = rh.main(args, api_factory=lambda cfg: api, environ={}, clock=world.now, sleep=world.sleep,
                   stt_loader=lambda lang: STT, out=lines.append, run_id="r1")
    return code, lines, api


def names(world: FakeWorld) -> list[str]:
    return [name for name, _a, _k in world.log]


# --------------------------------------------------------------------------- refusals


@pytest.mark.parametrize("flag", ["--yes", "--from", "--to", "--agent", "--manager", "--expect-account-sid"])
@pytest.mark.parametrize("mode", ["standalone", "frontdoor"])
def test_refuses_without_each_safety_flag(flag: str, mode: str) -> None:
    world = FakeWorld(mode=mode)
    made: list[Any] = []
    lines: list[str] = []
    code = rh.main(argv(mode, drop=(flag,)), api_factory=lambda cfg: made.append(cfg), environ={},
                   clock=world.now, sleep=world.sleep, stt_loader=lambda lang: STT, out=lines.append)
    assert code == 2
    assert made == [] and world.log == []
    assert flag in lines[0]


@pytest.mark.parametrize("extra, why", [
    (["--expect-account-sid", "AC" + "f" * 32], "does not match"),
    (["--agent", FROM], "all be different"),
    (["--to", "2065550101"], "E.164"),
    (["--long-wait", "10"], "--confirm-long-wait"),
    (["--long-wait", "10", "--confirm-long-wait", "11"], "--confirm-long-wait"),
    (["--long-wait", str(rh.MAX_LONG_WAIT_MINUTES + 1), "--confirm-long-wait",
      str(rh.MAX_LONG_WAIT_MINUTES + 1)], "--long-wait must be"),
    (["--confirm-long-wait", "10"], "without --long-wait"),
    (["--talk", "61"], "--talk"),
    (["--status-callback-url", "http://api.owlbank.example/voice/transcription"], "--status-callback-url"),
])
def test_refuses_bad_arguments(extra: list[str], why: str) -> None:
    world = FakeWorld()
    code, lines, _api = run(world, argv("standalone", *extra))
    assert code == 2 and world.log == []
    assert why in lines[0]


def test_account_in_use_comes_from_the_environment_when_not_given() -> None:
    world = FakeWorld()
    lines: list[str] = []
    args = argv("standalone", drop=("--account-sid",))
    code = rh.main(args, api_factory=lambda cfg: Failing(world), environ={"OWLBANK_TWILIO_ACCOUNT_SID": ACCOUNT},
                   clock=world.now, sleep=world.sleep, stt_loader=lambda lang: STT, out=lines.append)
    assert code == 0
    code = rh.main(args, api_factory=lambda cfg: Failing(world), environ={}, clock=world.now,
                   sleep=world.sleep, stt_loader=lambda lang: STT, out=lines.append)
    assert code == 2


@pytest.mark.parametrize("world, why", [
    (FakeWorld(account_status="suspended"), "expected active account"),
    (FakeWorld(owned=(FROM, TO, AGENT)), "--manager"),
    (FakeWorld(owned=(TO, AGENT, MANAGER)), "--from"),
])
def test_preflight_refuses_before_any_call(world: FakeWorld, why: str) -> None:
    code, lines, _api = run(world, argv())
    assert code == 2 and why in lines[-1]
    assert "create_call" not in names(world) and "create_conversation" not in names(world)


def test_frontdoor_mode_needs_the_called_number_to_be_ours() -> None:
    world = FakeWorld(mode="frontdoor", owned=(FROM, AGENT, MANAGER))
    code, lines, _api = run(world, argv("frontdoor"))
    assert code == 2 and "--to" in lines[-1] and "create_call" not in names(world)


def test_standalone_needs_engine_and_model() -> None:
    world = FakeWorld()
    lines: list[str] = []
    code = rh.main(argv(), api_factory=lambda cfg: Failing(world), environ={}, clock=world.now,
                   sleep=world.sleep, stt_loader=lambda lang: humanleg.SttSettings(None, None, "en-US"),
                   out=lines.append)
    assert code == 2 and "speech model" in lines[0]
    code, _lines, _api = run(world, argv("standalone", "--engine", "google", "--speech-model", "telephony"))
    assert code == 0 and world.transcriptions[0][1]["TranscriptionEngine"] == "google"


def test_the_estimate_is_printed_before_any_request() -> None:
    world = FakeWorld()
    seen: list[tuple[str, int]] = []
    rh.main(argv(), api_factory=lambda cfg: Failing(world), environ={}, clock=world.now, sleep=world.sleep,
            stt_loader=lambda lang: STT, out=lambda line: seen.append((line, len(world.log))))
    assert seen[0][0].startswith("Estimated billable minutes") and seen[0][1] == 0


def test_estimate_bounds() -> None:
    args = rh.build_parser().parse_args(argv("standalone", "--transfer"))
    cfg = rh.validate(args, environ={}, stt_loader=lambda lang: STT)
    assert rh.estimate_minutes(cfg) == {"voice": 2 * 3 + 2 * 3 * 2, "transcription": 3}


# --------------------------------------------------------------------------- the happy path


def test_standalone_run_end_to_end() -> None:
    world = FakeWorld()
    code, lines, _api = run(world, argv("standalone", "--transfer"))
    assert code == 0, lines
    assert world.open_things() == {"calls": [], "tasks": [], "conversations": []}
    order = names(world)
    # the conversation is created first, then the one call; transcription only after the accept
    assert order.index("create_conversation") < order.index("create_call") < order.index("accept_conference") \
        < order.index("start_transcription") < order.index("add_participant")
    assert order.count("create_call") == 1 and order.count("add_participant") == 1
    assert world.time_limit == rh.MAX_CALL_MINUTES * 60
    assert "cleanup: no call, task or conversation left open" in lines


def test_twiml_has_no_transcription_verb() -> None:
    world = FakeWorld()
    run(world, argv())
    sent = world.twiml
    assert "<Transcription" not in sent and "<Start" not in sent and "<Stream" not in sent
    assert f'workflowSid="{WW}"' in sent and "<Say" in sent
    task = json.loads(sent.split("<Task>")[1].split("</Task>")[0])
    assert task == {"escalation_reason": "rehearsal", "language": "en-US", "rehearsal_run": "r1"}
    world = FakeWorld(mode="frontdoor")
    run(world, argv("frontdoor"))
    assert "<Transcription" not in world.twiml and "<Start" not in world.twiml
    assert 'digits="0"' in world.twiml


def test_transcription_start_after_accept_carries_every_attribute() -> None:
    world = FakeWorld()
    run(world, argv())
    (call_sid, params), = world.transcriptions
    customer = next(s for s, c in world.calls.items() if c["direction"] == "outbound-api" and c["to"] == TO)
    assert call_sid == customer
    assert all(params.get(k) for k in rh.P10_KEYS)
    assert params == {"Name": "humanleg", "Track": "both_tracks", "InboundTrackLabel": "customer",
                      "OutboundTrackLabel": "agent", "TranscriptionEngine": "deepgram", "LanguageCode": "en-US",
                      "SpeechModel": "nova-3-general", "StatusCallbackUrl": STATUS_URL,
                      "ConversationId": next(iter(world.conversations))}
    assert world.clock >= world.accepted_at  # (and the log order is checked in the end-to-end test)


def test_frontdoor_mode_starts_no_transcription_and_creates_no_conversation() -> None:
    world = FakeWorld(mode="frontdoor")
    code, lines, _api = run(world, argv("frontdoor"))
    assert code == 0, lines
    assert "create_conversation" not in names(world) and "start_transcription" not in names(world)
    assert world.open_things() == {"calls": [], "tasks": [], "conversations": []}
    # the conversation the AI made for the inbound leg is the one reported and closed
    assert any("Orchestrator status: ACTIVE" in line for line in lines)


def test_other_tasks_in_the_queue_are_never_touched() -> None:
    for mode in ("standalone", "frontdoor"):
        world = FakeWorld(mode=mode)  # leaky: find_tasks returns the other task too
        run(world, argv(mode))
        assert OTHER_TASK not in world.touched_tasks
        assert world.tasks[OTHER_TASK]["status"] == "pending"
        assert world.calls[OTHER_CALL]["status"] == "in-progress"


# --------------------------------------------------------------------------- caps and the long wait


def test_long_wait_polls_every_five_minutes_and_stops_at_its_cap() -> None:
    world = FakeWorld()
    start = world.clock
    code, lines, _api = run(world, argv("standalone", "--long-wait", "12", "--confirm-long-wait", "12"))
    assert code == 0, lines
    assert [line for line in lines if line.startswith("long wait")] == [
        f"long wait: minute {m:>3}: conversation ACTIVE" for m in (0, 5, 10, 12)]
    assert world.accepted_at - start >= 12 * 60  # no agent leg before the wait is over
    assert world.time_limit == (12 + rh.LONG_WAIT_TAIL_MINUTES) * 60


def test_the_longest_wait_is_within_150_minutes() -> None:
    world = FakeWorld()
    m = str(rh.MAX_LONG_WAIT_MINUTES)
    code, _lines, _api = run(world, argv("standalone", "--long-wait", m, "--confirm-long-wait", m))
    assert code == 0 and world.time_limit == rh.MAX_LONG_WAIT_TOTAL_MINUTES * 60


def test_no_reservation_stops_at_the_call_cap_and_cleans_up() -> None:
    world = FakeWorld(reservations=False)
    start = world.clock
    code, lines, _api = run(world, argv())
    assert code == 1
    assert any("timed out waiting for a reservation" in line for line in lines)
    assert world.clock - start <= rh.MAX_CALL_MINUTES * 60 + 30  # + the settle-free cleanup retries
    assert world.open_things() == {"calls": [], "tasks": [], "conversations": []}


def test_caller_leaving_during_long_wait_aborts_and_cleans_up() -> None:
    world = FakeWorld()
    original = world.sleep

    def hang_up_after_first_poll(seconds: float) -> None:
        original(seconds)
        if seconds == rh.LONG_WAIT_POLL_S:
            for c in world.calls.values():
                if c["to"] == TO and c["direction"] == "outbound-api":
                    c["status"] = "completed"

    world.sleep = hang_up_after_first_poll  # type: ignore[method-assign]
    code, lines, _api = run(world, argv("standalone", "--long-wait", "20", "--confirm-long-wait", "20"))
    assert code == 1 and any("caller left the queue" in line for line in lines)
    assert world.open_things() == {"calls": [], "tasks": [], "conversations": []}


def session(world: FakeWorld, *extra: str) -> rh.Session:
    cfg = rh.validate(rh.build_parser().parse_args(argv("standalone", *extra)), environ={},
                      stt_loader=lambda lang: STT)
    return rh.Session(cfg, Failing(world), clock=world.now, sleep=world.sleep, out=lambda line: None, run_id="r1")


def test_one_call_and_two_legs_at_most() -> None:
    s = session(FakeWorld())
    s._place_call(to=TO, twiml_doc="<Response><Enqueue/></Response>")
    with pytest.raises(rh.Abort, match="at most 1 call"):
        s._place_call(to=TO, twiml_doc="<Response><Enqueue/></Response>")
    s._new_leg()
    s._new_leg()
    with pytest.raises(rh.Abort, match="at most 2 legs"):
        s._new_leg()


def test_manager_leg_time_limit_is_what_is_left_of_the_cap() -> None:
    world = FakeWorld()
    run(world, argv("standalone", "--transfer", "--talk", "30"))
    assert 0 < world.participant_time_limit <= rh.MAX_CALL_MINUTES * 60 - 30


# --------------------------------------------------------------------------- cleanup after a failure


def clean_run_length(mode: str, *extra: str) -> int:
    world = FakeWorld(mode=mode)
    _code, _lines, api = run(world, argv(mode, *extra))
    return api.count


@pytest.mark.parametrize("mode, extra", [("standalone", ("--transfer",)), ("frontdoor", ("--transfer",)),
                                         ("standalone", ("--long-wait", "6", "--confirm-long-wait", "6"))])
@pytest.mark.parametrize("kind", ["before", "after"])
def test_cleanup_after_a_failure_at_each_step(mode: str, extra: tuple[str, ...], kind: str) -> None:
    for index in range(clean_run_length(mode, *extra)):
        world = FakeWorld(mode=mode)
        code, lines, api = run(world, argv(mode, *extra), fail_at=index, kind=kind)
        failed = names(world)[index]
        assert world.open_things() == {"calls": [], "tasks": [], "conversations": []}, (index, failed, lines)
        assert code in (0, 1, 2), (index, failed, lines)  # 2: a preflight read failed, nothing placed
        assert OTHER_TASK not in world.touched_tasks


@pytest.mark.parametrize("always, what", [("end_call", "call"), ("set_task_status", "task"),
                                          ("close_conversation", "conversation")])
def test_what_cannot_be_cleaned_up_is_reported(always: str, what: str) -> None:
    world = FakeWorld()
    code, lines, _api = run(world, argv(), always=always)
    assert code == 3
    assert "NOT CLEANED UP (check the Console):" in lines
    left = world.open_things()
    for kind_ in ("calls", "tasks", "conversations"):
        for thing in left[kind_]:
            assert any(thing in line for line in lines), (thing, lines)
    assert any(f"- {what}" in line for line in lines)


def test_interrupt_still_cleans_up() -> None:
    world = FakeWorld()
    original = world.accept_conference

    def interrupted(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        raise KeyboardInterrupt

    world.accept_conference = interrupted  # type: ignore[method-assign]
    code, _lines, _api = run(world, argv())
    assert code == 130
    assert world.open_things() == {"calls": [], "tasks": [], "conversations": []}


# --------------------------------------------------------------------------- the report


def test_report_lists_callbacks_and_their_signature_status() -> None:
    alerts = [{"error_code": "11200", "request_url": CALLBACK_URL,
               "request_variables": "EventType=reservation.accepted&TaskSid=" + OTHER_TASK},
              {"error_code": "11200", "request_url": "https://elsewhere.example/hook", "request_variables": ""}]
    world = FakeWorld(alerts=alerts)
    code, lines, _api = run(world, argv())
    assert code == 0
    assert f"TaskRouter callbacks to {CALLBACK_URL}:" in lines
    assert "  task.created: accepted (no delivery alert)" in lines
    assert "  reservation.accepted: rejected (alert 11200)" in lines


def test_report_without_a_callback_url() -> None:
    world = FakeWorld(callback_url=None)
    _code, lines, _api = run(world, argv())
    assert "  task.created: not sent" in lines


def test_report_masks_phone_numbers_and_shows_the_transcript() -> None:
    world = FakeWorld()
    _code, lines, _api = run(world, argv())
    text = "\n".join(lines)
    assert FROM not in text and "+1******0100" in text
    assert "agent: Hello, how can I help?" in text
    assert "Orchestrator status: ACTIVE" in text and "conversation close: closed by the script" in text


def test_report_json(tmp_path: Any) -> None:
    world = FakeWorld()
    out = tmp_path / "report.json"
    run(world, argv("standalone", "--report-json", str(out)))
    doc = json.loads(out.read_text())
    assert doc["run_id"] == "r1" and doc["transcription_started"] is True and doc["leftovers"] == []


def test_callback_delivery_matches_host_and_path_only() -> None:
    got = rh.callback_delivery([{"event_type": "task.canceled"}],
                               [{"error_code": "11205", "request_url": CALLBACK_URL + "?x=1",
                                 "request_variables": "EventType=task.canceled"}], CALLBACK_URL)
    assert got["events"][0]["delivery"] == "rejected (alert 11205)"


# --------------------------------------------------------------------------- the REST adapter


class Resp:
    def __init__(self, doc: Any, status: int = 200) -> None:
        self.status = status
        self._raw = json.dumps(doc).encode()

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> Resp:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class Opener:
    """Records requests; answers by the first matching ``(method, url substring)``."""

    def __init__(self, answers: list[tuple[str, str, Any]]) -> None:
        self.answers = answers
        self.requests: list[Any] = []

    def __call__(self, req: Any, timeout: float) -> Resp:
        self.requests.append(req)
        for method, part, doc in self.answers:
            if req.get_method() == method and part in req.full_url:
                if isinstance(doc, int):
                    raise urllib.error.HTTPError(req.full_url, doc, "err", {}, io.BytesIO(b"secret body"))  # type: ignore[arg-type]
                return Resp(doc)
        raise AssertionError(f"unexpected {req.get_method()} {req.full_url}")


def form(req: Any) -> dict[str, str]:
    return dict(urllib.parse.parse_qsl(req.data.decode()))


def adapter(opener: Opener) -> rh.TwilioRehearsalApi:
    return rh.TwilioRehearsalApi(ACCOUNT, "SK" + "1" * 32, "s3cret-value", WS, opener=opener)


def test_adapter_start_transcription_posts_the_calls_api_form() -> None:
    call = "CA" + "2" * 32
    opener = Opener([("POST", f"/Calls/{call}/Transcriptions.json", {"sid": "GT1"})])
    params = humanleg.transcription_start("conv_1", enabled=True, language="en-US", status_url=STATUS_URL,
                                          engine="deepgram", speech_model="nova-3-general")
    assert adapter(opener).start_transcription(call, params) is True
    req, = opener.requests
    assert req.full_url == f"https://api.twilio.com/2010-04-01/Accounts/{ACCOUNT}/Calls/{call}/Transcriptions.json"
    assert form(req) == dict(params) and set(rh.P10_KEYS) <= set(form(req))


def test_adapter_finds_tasks_by_call_sid_only() -> None:
    call = "CA" + "2" * 32
    tasks = {"tasks": [{"sid": "WT" + "3" * 32, "assignment_status": "pending",
                        "attributes": json.dumps({"call_sid": call})}]}
    opener = Opener([("GET", "/Tasks?", tasks)])
    got = adapter(opener).find_tasks(call)
    assert got == [{"sid": "WT" + "3" * 32, "status": "pending", "call_sid": call}]
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(opener.requests[0].full_url).query))
    assert query["EvaluateTaskAttributes"] == f'call_sid == "{call}"'
    with pytest.raises(rh.ApiError):
        adapter(opener).find_tasks('CA" OR 1==1')


def test_adapter_accept_uses_the_conference_instruction() -> None:
    task, res = "WT" + "3" * 32, "WR" + "4" * 32
    opener = Opener([("POST", f"/Tasks/{task}/Reservations/{res}", {})])
    adapter(opener).accept_conference(task, res, from_=FROM, to=AGENT)
    assert form(opener.requests[0]) == {"Instruction": "conference", "From": FROM, "To": AGENT,
                                        "Timeout": str(rh.RING_TIMEOUT_S), "EndConferenceOnExit": "false",
                                        "EndConferenceOnCustomerExit": "true"}


@pytest.mark.parametrize("status, target", [("ringing", "canceled"), ("queued", "canceled"),
                                            ("in-progress", "completed"), ("completed", None)])
def test_adapter_end_call(status: str, target: str | None) -> None:
    call = "CA" + "2" * 32
    opener = Opener([("GET", f"/Calls/{call}.json", {"status": status}), ("POST", f"/Calls/{call}.json", {})])
    adapter(opener).end_call(call)
    posts = [r for r in opener.requests if r.get_method() == "POST"]
    assert [form(r)["Status"] for r in posts] == ([target] if target else [])


def test_adapter_create_call_sends_the_time_limit() -> None:
    opener = Opener([("POST", "/Calls.json", {"sid": "CA" + "5" * 32})])
    assert adapter(opener).create_call(to=TO, from_=FROM, twiml="<Response/>", time_limit_s=180) == "CA" + "5" * 32
    assert form(opener.requests[0]) == {"To": TO, "From": FROM, "Twiml": "<Response/>", "TimeLimit": "180"}


def test_adapter_errors_carry_no_secret_or_body() -> None:
    opener = Opener([("GET", ".json", 401)])
    with pytest.raises(rh.ApiError) as err:
        adapter(opener).account()
    assert str(err.value) == "account.fetch: HTTP 401"
    assert "s3cret" not in repr(err.value) and "secret body" not in repr(err.value)


def test_adapter_is_https_only() -> None:
    opener = Opener([])
    api = adapter(opener)
    with pytest.raises(rh.ApiError, match="non-https"):
        api._req("GET", "http://api.twilio.com/x", "probe")
    assert opener.requests == []


def test_adapter_conversation_create_body_and_find_by_name() -> None:
    opener = Opener([("POST", "/v2/Conversations", {"id": "conv_1"}),
                     ("GET", "/v2/Conversations?", {"conversations": [
                         {"id": "conv_0", "name": "other"}, {"id": "conv_1", "name": "humanleg-rehearsal-r1"}]})])
    api = adapter(opener)
    assert api.create_conversation(CONFIG, "humanleg-rehearsal-r1") == "conv_1"
    assert json.loads(opener.requests[0].data) == {"configurationId": CONFIG, "name": "humanleg-rehearsal-r1"}
    assert api.find_conversation(CONFIG, "humanleg-rehearsal-r1") == "conv_1"


def test_adapter_find_calls_keeps_recent_calls_only() -> None:
    calls = {"calls": [{"sid": "CA" + "1" * 32, "status": "in-progress", "direction": "inbound",
                        "start_time": "Fri, 15 Jan 2027 08:00:00 +0000"},
                       {"sid": "CA" + "2" * 32, "status": "completed", "direction": "inbound",
                        "start_time": "Fri, 15 Jan 2027 07:00:00 +0000"}]}
    opener = Opener([("GET", "/Calls.json", calls)])
    since = rh._epoch("Fri, 15 Jan 2027 07:59:30 +0000")
    assert since is not None
    got = adapter(opener).find_calls(from_=FROM, to=TO, since=since)
    assert [c["sid"] for c in got] == ["CA" + "1" * 32]
