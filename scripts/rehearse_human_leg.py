"""Scripted rehearsal call of the human leg, no Flex login.

Places **one real call** by API and drives the human leg without a browser, then reports what
Orchestrator holds. Costs money and rings real numbers: owner go-ahead and two spare Twilio numbers
(the "agent" and the "manager", each answered by TwiML that speaks a few lines) are needed. Usage::

    # standalone (before our code is deployed): the script creates the conversation, places a call to
    # --to whose inline TwiML says a line and enqueues to the workflow (no transcription verb), and
    # starts transcription through the Calls API after it accepts the reservation
    uv run python scripts/rehearse_human_leg.py standalone --yes \\
        --expect-account-sid AC... --account-sid AC... \\
        --from +1... --to +1... --agent +1... --manager +1... [--transfer] \\
        --workspace-sid WS... --workflow-sid WW... --configuration-id conv_configuration_... \\
        --status-callback-url https://<front door host>/voice/transcription

    # Wait in the queue first (hold music, no agent leg), then accept; status every 5 minutes
    ... standalone ... --long-wait 45 --confirm-long-wait 45

    # frontdoor (after the deploy): the call goes to the deployed number (--to); the caller side
    # presses 0 (the AI's "agent" shortcut) after a pause; the front door transcribes and closes
    uv run python scripts/rehearse_human_leg.py frontdoor --yes --expect-account-sid AC... --account-sid AC... \\
        --from +1... --to <front door number> --agent +1... --manager +1... --workspace-sid WS...

In both modes it waits for the call's task, accepts the reservation through the API with the
``conference`` instruction to ``--agent`` (no Flex login), optionally (``--transfer``) adds ``--manager``
to the conference and hangs up the agent (a call-level cold transfer), hangs up, waits ``--settle``
seconds, then reports: the conversation's status, participants and transcript communications, the
TaskRouter events of **this** task and, per event, whether the callback was accepted (see "Callbacks").

**Safety.** Refuses (exit 2) without ``--yes``; without each of ``--from/--to/--agent/--manager``
(E.164, all different); unless ``--expect-account-sid`` equals the account in use (``--account-sid`` or
``OWLBANK_TWILIO_ACCOUNT_SID``) and that account answers as active; unless ``--from``, ``--agent`` and
``--manager`` (and in ``frontdoor`` mode ``--to``) are numbers of that account. Hard caps: one call
(:data:`MAX_CALLS`), :data:`MAX_CALL_MINUTES` minutes (Twilio's ``TimeLimit`` on the call too, so a killed
script cannot exceed it), or up to :data:`MAX_LONG_WAIT_TOTAL_MINUTES` with ``--long-wait`` and a matching
``--confirm-long-wait``; at most :data:`MAX_LEGS` legs besides the caller. The estimated billable minutes
are printed before any request. Whatever happens (an error, a timeout, Ctrl-C) the cleanup stops the
transcription, ends every call it placed or can find from ``--from`` to the four numbers since the start,
cancels or completes the task of **its own call** and closes the test conversation, each with retries; what
it could not clean up is printed and the exit code is 3. Tasks are only ever looked up by this call's SID
(``call_sid`` attribute, checked again here): other tasks in the queue are never read or written.

**Preconditions not done by the script.** A worker in the queue the workflow picks must be Available (set
it in the Console's TaskRouter workers page; no Flex login), or no reservation is made and the run times
out and cleans up. ``--to`` in standalone mode, ``--agent`` and ``--manager`` must answer and stay on the
line (TwiML with ``<Say>`` lines and ``<Pause>``).

**Callbacks.** The script cannot see a callback's signature. It lists the task's events from the
TaskRouter Events API (what TaskRouter emitted), and the Monitor alerts in the run's window for the
workspace's callback URL: the front door rejects a bad signature with 403, which Twilio records as an
alert. An event without an alert is reported "accepted (no delivery alert)"; with one, "rejected".

**Unverified API shapes** (to confirm at the first live run, with the list in the live check): see :data:`LIVE_CHECKS`.

Credentials: the Twilio API key secret in Secrets Manager (``twilio_api_key``, found from
``--cdk-outputs`` or ``--secret``, as ``twilio_setup.py``); its account must be the account in use.
Never printed or logged. Requests are https only.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from owlbank import humanleg, twiml
from owlbank.humanleg_client import OrchestratorHumanLeg, TwilioCalls
from scripts._common import (
    DATA_DIR,
    PHONE_LIKE,
    ScriptError,
    add_secret_args,
    load_cdk_outputs,
    mask_phone,
    parse_secret_overrides,
    read_secret,
    resolve_region,
    script_error,
    script_start,
    script_step,
    secret_arn,
    start_logging,
)

slog = logging.getLogger("rehearse_human_leg")
SCRIPT = "rehearse_human_leg"

# --------------------------------------------------------------------------- caps and settings

MAX_CALLS = 1
MAX_LEGS = 2  # agent and manager
MAX_CALL_MINUTES = 3
MAX_LONG_WAIT_TOTAL_MINUTES = 150
#: Minutes kept after a long wait for the accept, the talk, the transfer and the hang-up.
LONG_WAIT_TAIL_MINUTES = MAX_CALL_MINUTES
MAX_LONG_WAIT_MINUTES = MAX_LONG_WAIT_TOTAL_MINUTES - LONG_WAIT_TAIL_MINUTES
LONG_WAIT_POLL_S = 300
POLL_S = 2.0
CLEANUP_ATTEMPTS = 3
CLEANUP_RETRY_S = 1.0
MAX_TALK_S = 60
MAX_SETTLE_S = 120
#: Seconds the agent's and the manager's phone may ring (conference instruction ``Timeout``).
RING_TIMEOUT_S = 20
NAME_PREFIX = "humanleg-rehearsal-"

#: Every transcription start carries these.
P10_KEYS = ("Name", "Track", "InboundTrackLabel", "OutboundTrackLabel", "TranscriptionEngine",
            "LanguageCode", "SpeechModel", "StatusCallbackUrl")

OPEN_CALL = frozenset({"queued", "initiated", "ringing", "in-progress"})
OPEN_TASK = frozenset({"pending", "reserved", "assigned", "wrapping"})

#: Shapes this script assumes and the first live run must confirm (joins the "verify" list).
LIVE_CHECKS = (
    "POST /Calls/{CallSid}/Transcriptions.json binds to the conversation with ConversationId "
    "(humanleg.transcription_start); utterances of both parties land in it",
    "TaskRouter conference instruction: the conference's FriendlyName is the task SID; the agent leg is "
    "From=--from To=--agent; EndConferenceOnCustomerExit=true ends the agent leg with the caller",
    "Tasks list filter EvaluateTaskAttributes=call_sid==\"CA...\" finds the <Enqueue> task (Enqueue sets "
    "call_sid in the task attributes)",
    "Reservations list filter ReservationStatus=pending; a reservation exists only with an Available worker",
    "Task cleanup: AssignmentStatus=canceled from pending/reserved, completed from assigned/wrapping",
    "Calls list filters From/To/StartTime>= and the direction field (inbound leg at the front door number)",
    "POST /v2/Conversations {configurationId, name} creates; GET /v2/Conversations/{id}/Communications "
    "answers {communications: [{author, content: {text}}]}",
    "TaskRouter Events API filter TaskSid; Monitor Alerts list carries request_url and the fetch "
    "carries request_variables (EventType) for a rejected callback",
    "S3 (partly): a call-level cold transfer (add manager, end agent) keeps the customer leg and its "
    "transcription; it makes no second reservation.accepted (a Flex transfer would)",
    "frontdoor mode: DTMF 0 after --caller-pause seconds reaches a human (the AI keypad shortcut)",
)

_E164 = re.compile(r"^\+[1-9]\d{6,14}$")
_ACCOUNT = re.compile(r"^AC[0-9a-fA-F]{32}$")
_WORKSPACE = re.compile(r"^WS[0-9a-fA-F]{32}$")
_WORKFLOW = re.compile(r"^WW[0-9a-fA-F]{32}$")
_CALL = re.compile(r"^CA[0-9a-fA-F]{32}$")
_TASK = re.compile(r"^WT[0-9a-fA-F]{32}$")
_RESERVATION = re.compile(r"^WR[0-9a-fA-F]{32}$")
_CONFERENCE = re.compile(r"^CF[0-9a-fA-F]{32}$")
_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_LANGUAGE = re.compile(r"^[a-z]{2,3}-[A-Z]{2}$")

API_BASE = "https://api.twilio.com"
TASKROUTER_BASE = "https://taskrouter.twilio.com"
MONITOR_BASE = "https://monitor.twilio.com"
ORCHESTRATOR_BASE = "https://conversations.twilio.com"


class Refusal(ScriptError):
    """The run is refused before any request (exit 2)."""


class Abort(ScriptError):
    """The run stopped part-way (a timeout, a cap, a caller who left); cleanup follows."""


class ApiError(ScriptError):
    """A Twilio request failed. The message names the operation and the HTTP status only."""


# --------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class Settings:
    mode: str
    account_sid: str
    from_: str
    to: str
    agent: str
    manager: str
    workspace_sid: str
    workflow_sid: str | None
    configuration_id: str | None
    status_callback_url: str | None
    language: str
    engine: str | None
    speech_model: str | None
    transfer: bool = False
    long_wait_min: int = 0
    talk_s: int = 30
    settle_s: int = 20
    caller_pause_s: int = 15
    caller_digits: str = "0"
    say: str = "This is an OwlBank rehearsal call. Please hold for an agent."
    report_json: Path | None = None

    @property
    def cap_s(self) -> int:
        minutes = self.long_wait_min + LONG_WAIT_TAIL_MINUTES if self.long_wait_min else MAX_CALL_MINUTES
        return minutes * 60

    @property
    def numbers(self) -> tuple[str, ...]:
        return (self.to, self.agent, self.manager)


def estimate_minutes(cfg: Settings) -> dict[str, int]:
    """Upper bounds of the billable minutes: every Twilio-to-Twilio leg bills at both ends."""
    cap = cfg.cap_s // 60
    tail = LONG_WAIT_TAIL_MINUTES if cfg.long_wait_min else MAX_CALL_MINUTES
    agent_legs = 2 if cfg.transfer else 1
    return {"voice": 2 * cap + 2 * tail * agent_legs,
            "transcription": tail if cfg.mode == "standalone" else 0}


def _https(url: str) -> bool:
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    return parts.scheme == "https" and bool(parts.hostname) and not parts.username and not parts.password


def validate(args: argparse.Namespace, *, environ: Mapping[str, str],
             stt_loader: Callable[[str], humanleg.SttSettings]) -> Settings:
    """Settings from the arguments, or :class:`Refusal` listing every problem. No request is made."""
    problems: list[str] = []
    if not args.yes:
        problems.append("--yes is required: this places a real, billed call")
    numbers = {"--from": args.from_, "--to": args.to, "--agent": args.agent, "--manager": args.manager}
    for flag, value in numbers.items():
        if not value:
            problems.append(f"{flag} is required")
        elif not _E164.match(value):
            problems.append(f"{flag} must be an E.164 number (+...)")
    given = [v for v in numbers.values() if v]
    if len(set(given)) != len(given):
        problems.append("--from, --to, --agent and --manager must all be different numbers")
    account = args.account_sid or environ.get("OWLBANK_TWILIO_ACCOUNT_SID") or ""
    if not args.expect_account_sid:
        problems.append("--expect-account-sid is required")
    elif not _ACCOUNT.match(args.expect_account_sid):
        problems.append("--expect-account-sid must be an account SID (AC...)")
    elif args.expect_account_sid != account:
        problems.append("--expect-account-sid does not match the account in use "
                        "(--account-sid or OWLBANK_TWILIO_ACCOUNT_SID)")
    if not _WORKSPACE.match(args.workspace_sid or ""):
        problems.append("--workspace-sid (WS...) is required")
    if not _LANGUAGE.match(args.language or ""):
        problems.append("--language must look like en-US")
    if args.mode == "standalone":
        if not _WORKFLOW.match(args.workflow_sid or ""):
            problems.append("standalone needs --workflow-sid (WW...)")
        if not _ID.match(args.configuration_id or ""):
            problems.append("standalone needs --configuration-id (the Conversation Configuration)")
        if not args.status_callback_url or not _https(args.status_callback_url):
            problems.append("standalone needs --status-callback-url, an https URL on the front door's host")
    if args.long_wait is not None:
        if not 1 <= args.long_wait <= MAX_LONG_WAIT_MINUTES:
            problems.append(f"--long-wait must be 1 to {MAX_LONG_WAIT_MINUTES} minutes "
                            f"(the run's cap is {MAX_LONG_WAIT_TOTAL_MINUTES} minutes)")
        if args.confirm_long_wait != args.long_wait:
            problems.append("--long-wait needs --confirm-long-wait with the same number of minutes")
    elif args.confirm_long_wait is not None:
        problems.append("--confirm-long-wait without --long-wait")
    if not 1 <= args.talk <= MAX_TALK_S:
        problems.append(f"--talk must be 1 to {MAX_TALK_S} seconds")
    if not 0 <= args.settle <= MAX_SETTLE_S:
        problems.append(f"--settle must be 0 to {MAX_SETTLE_S} seconds")
    if not re.fullmatch(r"[0-9*#w]{1,8}", args.caller_digits or ""):
        problems.append("--caller-digits must be 1 to 8 of 0-9 * # w")
    if not 1 <= args.caller_pause <= 60:
        problems.append("--caller-pause must be 1 to 60 seconds")

    engine, model = args.engine, args.speech_model
    if args.mode == "standalone" and (not engine or not model) and _LANGUAGE.match(args.language or ""):
        try:
            stt = stt_loader(args.language)
        except Exception as exc:  # noqa: BLE001 - reported as a refusal
            problems.append(f"could not read the catalog's STT settings ({type(exc).__name__}): "
                            "pass --engine and --speech-model")
        else:
            engine, model = engine or stt.engine, model or stt.speech_model
    if args.mode == "standalone" and (not engine or not model):
        problems.append("standalone needs a transcription engine and speech model: "
                        "--engine/--speech-model or the catalog's conversationRelay settings")
    if problems:
        raise Refusal("refusing to run:\n  - " + "\n  - ".join(problems))
    return Settings(mode=args.mode, account_sid=account, from_=args.from_, to=args.to, agent=args.agent,
                    manager=args.manager, workspace_sid=args.workspace_sid, workflow_sid=args.workflow_sid,
                    configuration_id=args.configuration_id, status_callback_url=args.status_callback_url,
                    language=args.language, engine=engine, speech_model=model, transfer=args.transfer,
                    long_wait_min=args.long_wait or 0, talk_s=args.talk, settle_s=args.settle,
                    caller_pause_s=args.caller_pause, caller_digits=args.caller_digits,
                    report_json=args.report_json)


def catalog_stt(language: str) -> humanleg.SttSettings:
    """The AI segment's STT settings for ``language`` from ``data/prompt_catalog.json``."""
    from owlbank.catalog import PromptCatalog

    doc = json.loads((DATA_DIR / "prompt_catalog.json").read_text(encoding="utf-8"))
    return humanleg.stt_settings(PromptCatalog.from_dict(doc).settings(language).get("conversationRelay"))


# --------------------------------------------------------------------------- TwiML (no transcription)


def standalone_twiml(cfg: Settings, run_id: str) -> str:
    """Says a line, then ``<Enqueue>`` to the workflow (default hold music while waiting). No
    ``<Start>``/``<Transcription>``: transcription starts through the Calls API at accept."""
    attributes = {"language": cfg.language, "escalation_reason": "rehearsal", "rehearsal_run": run_id}
    return twiml.render(twiml.Say(text=cfg.say, language=cfg.language),
                        twiml.Enqueue(workflow_sid=str(cfg.workflow_sid), task_attributes=attributes))


def caller_twiml(cfg: Settings) -> str:
    """The caller's side of a ``frontdoor`` call: wait, press ``--caller-digits``, then stay on the line."""
    from twilio.twiml.voice_response import VoiceResponse

    resp = VoiceResponse()
    resp.pause(length=cfg.caller_pause_s)
    resp.play(digits=cfg.caller_digits)
    for _ in range(max(1, cfg.cap_s // 600 + 1)):
        resp.pause(length=600)
    return str(resp.to_xml())


# --------------------------------------------------------------------------- the API the run uses


class Api(Protocol):
    """Every Twilio request the rehearsal makes (a fake replaces it in tests). Each raises
    :class:`ApiError` on failure, except the three ``bool`` ones (the shared clients' contract)."""

    def account(self) -> Mapping[str, Any]: ...
    def owns_number(self, number: str) -> bool: ...
    def create_call(self, *, to: str, from_: str, twiml: str, time_limit_s: int) -> str: ...
    def find_calls(self, *, from_: str, to: str, since: float) -> list[Mapping[str, Any]]: ...
    def call_status(self, call_sid: str) -> str: ...
    def end_call(self, call_sid: str) -> None: ...
    def find_tasks(self, call_sid: str) -> list[Mapping[str, Any]]: ...
    def task_status(self, task_sid: str) -> str: ...
    def set_task_status(self, task_sid: str, status: str) -> None: ...
    def pending_reservation(self, task_sid: str) -> str | None: ...
    def accept_conference(self, task_sid: str, reservation_sid: str, *, from_: str, to: str) -> None: ...
    def conference_for_task(self, task_sid: str) -> str | None: ...
    def conference_calls(self, conference_sid: str) -> list[str]: ...
    def add_participant(self, conference_sid: str, *, from_: str, to: str, time_limit_s: int) -> str: ...
    def task_events(self, task_sid: str) -> list[Mapping[str, Any]]: ...
    def callback_url(self) -> str | None: ...
    def alerts(self, since: float, until: float) -> list[Mapping[str, Any]]: ...
    def create_conversation(self, configuration_id: str, name: str) -> str: ...
    def find_conversation(self, configuration_id: str, name: str) -> str | None: ...
    def resolve_conversation(self, call_sid: str) -> str | None: ...
    def conversation(self, conversation_id: str) -> Mapping[str, Any]: ...
    def participants(self, conversation_id: str) -> list[Mapping[str, Any]]: ...
    def communications(self, conversation_id: str) -> list[Mapping[str, Any]]: ...
    def close_conversation(self, conversation_id: str) -> bool: ...
    def start_transcription(self, call_sid: str, params: Mapping[str, str]) -> bool: ...
    def stop_transcription(self, call_sid: str, name: str) -> bool: ...


Opener = Callable[[urllib.request.Request, float], Any]


def _open(req: urllib.request.Request, timeout: float) -> Any:
    if not _https(req.full_url):
        raise ApiError("refused a non-https URL")
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310  # nosec B310 (https checked above)


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _epoch(value: Any) -> float | None:
    """RFC 2822 (Calls API) or ISO 8601 (TaskRouter, Monitor) as epoch seconds."""
    if not isinstance(value, str) or not value:
        return None
    from datetime import datetime
    from email.utils import parsedate_to_datetime

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError):
        return None


class TwilioRehearsalApi:
    """:class:`Api` over Twilio's REST APIs (https, HTTP Basic with the API key). Transcription start and
    stop reuse :class:`owlbank.humanleg_client.TwilioCalls`; conversation lookup and close reuse
    :class:`owlbank.humanleg_client.OrchestratorHumanLeg`. Only the rehearsal's own conversation is created
    here (the shared client never creates one)."""

    def __init__(self, account_sid: str, key: str, secret: str, workspace_sid: str, *,
                 opener: Opener = _open, timeout_s: float = 10.0) -> None:
        if not _ACCOUNT.match(account_sid) or not _WORKSPACE.match(workspace_sid):
            raise ApiError("bad account or workspace SID")
        self._account = account_sid
        self._auth = (key, secret)
        self._ws = workspace_sid
        self._opener = opener
        self._timeout_s = timeout_s
        creds = (account_sid, key, secret)
        self._calls = TwilioCalls(lambda: creds, opener=opener, timeout_s=timeout_s)
        self._orchestrator = OrchestratorHumanLeg(lambda: creds, opener=opener, timeout_s=timeout_s)

    # ------------------------------------------------------------------ HTTP

    def _req(self, method: str, url: str, operation: str, *, query: Mapping[str, Any] | None = None,
             form: Mapping[str, Any] | None = None, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if query:
            url += "?" + urllib.parse.urlencode(query)
        if not _https(url):
            raise ApiError(f"{operation}: refused a non-https URL")
        basic = base64.b64encode(f"{self._auth[0]}:{self._auth[1]}".encode()).decode()
        headers = {"Accept": "application/json", "Authorization": f"Basic {basic}"}
        data = None
        if form is not None:
            data = urllib.parse.urlencode({k: v for k, v in form.items() if v is not None}).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)  # noqa: S310 (https checked)
        try:
            with self._opener(req, self._timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raise ApiError(f"{operation}: HTTP {exc.code}") from None
        except ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 - never chain: the exception text could carry a URL
            raise ApiError(f"{operation}: {type(exc).__name__}") from None
        try:
            parsed = json.loads(raw) if raw else {}
        except ValueError:
            raise ApiError(f"{operation}: response is not JSON") from None
        return parsed if isinstance(parsed, dict) else {}

    def _api(self, path: str) -> str:
        return f"{API_BASE}/2010-04-01/Accounts/{self._account}{path}"

    def _tr(self, path: str) -> str:
        return f"{TASKROUTER_BASE}/v1/Workspaces/{self._ws}{path}"

    # ------------------------------------------------------------------ account and numbers

    def account(self) -> Mapping[str, Any]:
        return self._req("GET", self._api(".json"), "account.fetch")

    def owns_number(self, number: str) -> bool:
        got = self._req("GET", self._api("/IncomingPhoneNumbers.json"), "numbers.list",
                        query={"PhoneNumber": number, "PageSize": 5})
        listed = got.get("incoming_phone_numbers") or []
        return any(isinstance(n, Mapping) and n.get("phone_number") == number for n in listed)

    # ------------------------------------------------------------------ calls

    def create_call(self, *, to: str, from_: str, twiml: str, time_limit_s: int) -> str:
        got = self._req("POST", self._api("/Calls.json"), "calls.create",
                        form={"To": to, "From": from_, "Twiml": twiml, "TimeLimit": time_limit_s})
        sid = got.get("sid")
        if not isinstance(sid, str) or not _CALL.match(sid):
            raise ApiError("calls.create: no call SID in the answer")
        return sid

    def find_calls(self, *, from_: str, to: str, since: float) -> list[Mapping[str, Any]]:
        got = self._req("GET", self._api("/Calls.json"), "calls.list",
                        query={"From": from_, "To": to, "StartTime>": _iso(since - 86400)[:10],
                               "PageSize": 50})
        out = []
        for c in got.get("calls") or []:
            if not isinstance(c, Mapping) or not _CALL.match(str(c.get("sid", ""))):
                continue
            started = _epoch(c.get("start_time")) or _epoch(c.get("date_created"))
            if started is not None and started < since - 60:
                continue
            out.append({"sid": c["sid"], "status": c.get("status"), "direction": c.get("direction")})
        return out

    def call_status(self, call_sid: str) -> str:
        return str(self._req("GET", self._api(f"/Calls/{call_sid}.json"), "calls.fetch").get("status", ""))

    def end_call(self, call_sid: str) -> None:
        status = self.call_status(call_sid)
        if status not in OPEN_CALL:
            return
        # A call not yet answered is cancelled; an answered one is completed (Calls API).
        target = "completed" if status == "in-progress" else "canceled"
        self._req("POST", self._api(f"/Calls/{call_sid}.json"), "calls.end", form={"Status": target})

    # ------------------------------------------------------------------ TaskRouter

    def find_tasks(self, call_sid: str) -> list[Mapping[str, Any]]:
        if not _CALL.match(call_sid):
            raise ApiError("tasks.list: bad call SID")
        got = self._req("GET", self._tr("/Tasks"), "tasks.list",
                        query={"EvaluateTaskAttributes": f'call_sid == "{call_sid}"', "PageSize": 20})
        out = []
        for t in got.get("tasks") or []:
            if not isinstance(t, Mapping) or not _TASK.match(str(t.get("sid", ""))):
                continue
            try:
                attrs = json.loads(t.get("attributes") or "{}")
            except ValueError:
                attrs = {}
            out.append({"sid": t["sid"], "status": t.get("assignment_status"),
                        "call_sid": attrs.get("call_sid") if isinstance(attrs, dict) else None})
        return out

    def task_status(self, task_sid: str) -> str:
        return str(self._req("GET", self._tr(f"/Tasks/{task_sid}"), "tasks.fetch").get("assignment_status", ""))

    def set_task_status(self, task_sid: str, status: str) -> None:
        self._req("POST", self._tr(f"/Tasks/{task_sid}"), "tasks.update",
                  form={"AssignmentStatus": status, "Reason": "humanleg rehearsal cleanup"})

    def pending_reservation(self, task_sid: str) -> str | None:
        got = self._req("GET", self._tr(f"/Tasks/{task_sid}/Reservations"), "reservations.list",
                        query={"ReservationStatus": "pending", "PageSize": 5})
        for r in got.get("reservations") or []:
            if isinstance(r, Mapping) and _RESERVATION.match(str(r.get("sid", ""))):
                return str(r["sid"])
        return None

    def accept_conference(self, task_sid: str, reservation_sid: str, *, from_: str, to: str) -> None:
        self._req("POST", self._tr(f"/Tasks/{task_sid}/Reservations/{reservation_sid}"), "reservations.accept",
                  form={"Instruction": "conference", "From": from_, "To": to, "Timeout": RING_TIMEOUT_S,
                        "EndConferenceOnExit": "false", "EndConferenceOnCustomerExit": "true"})

    def conference_for_task(self, task_sid: str) -> str | None:
        got = self._req("GET", self._api("/Conferences.json"), "conferences.list",
                        query={"FriendlyName": task_sid, "Status": "in-progress", "PageSize": 5})
        for c in got.get("conferences") or []:
            if isinstance(c, Mapping) and _CONFERENCE.match(str(c.get("sid", ""))):
                return str(c["sid"])
        return None

    def conference_calls(self, conference_sid: str) -> list[str]:
        got = self._req("GET", self._api(f"/Conferences/{conference_sid}/Participants.json"),
                        "participants.list", query={"PageSize": 20})
        return [str(p["call_sid"]) for p in got.get("participants") or []
                if isinstance(p, Mapping) and _CALL.match(str(p.get("call_sid", "")))]

    def add_participant(self, conference_sid: str, *, from_: str, to: str, time_limit_s: int) -> str:
        got = self._req("POST", self._api(f"/Conferences/{conference_sid}/Participants.json"),
                        "participants.create",
                        form={"From": from_, "To": to, "Label": "manager", "TimeLimit": time_limit_s,
                              "Timeout": RING_TIMEOUT_S, "EndConferenceOnExit": "false"})
        sid = got.get("call_sid")
        if not isinstance(sid, str) or not _CALL.match(sid):
            raise ApiError("participants.create: no call SID in the answer")
        return sid

    def task_events(self, task_sid: str) -> list[Mapping[str, Any]]:
        got = self._req("GET", self._tr("/Events"), "events.list", query={"TaskSid": task_sid, "PageSize": 100})
        return [{"event_type": e.get("event_type"), "event_date": e.get("event_date")}
                for e in got.get("events") or [] if isinstance(e, Mapping)]

    def callback_url(self) -> str | None:
        url = self._req("GET", self._tr(""), "workspace.fetch").get("event_callback_url")
        return url if isinstance(url, str) and url else None

    def alerts(self, since: float, until: float) -> list[Mapping[str, Any]]:
        got = self._req("GET", f"{MONITOR_BASE}/v1/Alerts", "alerts.list",
                        query={"StartDate": _iso(since - 60), "EndDate": _iso(until + 60), "PageSize": 50})
        out = []
        for a in got.get("alerts") or []:
            if not isinstance(a, Mapping) or not isinstance(a.get("sid"), str):
                continue
            row = {"error_code": a.get("error_code"), "request_url": a.get("request_url"),
                   "request_variables": None, "response_status": None}
            if len(out) < 20 and re.fullmatch(r"NO[0-9a-fA-F]{32}", a["sid"]):
                detail = self._req("GET", f"{MONITOR_BASE}/v1/Alerts/{a['sid']}", "alerts.fetch")
                row["request_variables"] = detail.get("request_variables")
                row["request_url"] = detail.get("request_url") or row["request_url"]
            out.append(row)
        return out

    # ------------------------------------------------------------------ Orchestrator

    def create_conversation(self, configuration_id: str, name: str) -> str:
        got = self._req("POST", f"{ORCHESTRATOR_BASE}/v2/Conversations", "conversations.create",
                        body={"configurationId": configuration_id, "name": name})
        cid = got.get("id")
        if not isinstance(cid, str) or not _ID.match(cid):
            raise ApiError("conversations.create: no id in the answer")
        return cid

    def find_conversation(self, configuration_id: str, name: str) -> str | None:
        got = self._req("GET", f"{ORCHESTRATOR_BASE}/v2/Conversations", "conversations.list",
                        query={"configurationId": configuration_id, "status": "ACTIVE", "pageSize": 50})
        for c in got.get("conversations") or []:
            if isinstance(c, Mapping) and c.get("name") == name and _ID.match(str(c.get("id", ""))):
                return str(c["id"])
        return None

    def resolve_conversation(self, call_sid: str) -> str | None:
        return self._orchestrator.resolve_conversation(call_sid)

    def conversation(self, conversation_id: str) -> Mapping[str, Any]:
        return self._req("GET", f"{ORCHESTRATOR_BASE}/v2/Conversations/{conversation_id}", "conversations.get")

    def participants(self, conversation_id: str) -> list[Mapping[str, Any]]:
        got = self._req("GET", f"{ORCHESTRATOR_BASE}/v2/Conversations/{conversation_id}/Participants",
                        "participants.list")
        return [p for p in got.get("participants") or [] if isinstance(p, Mapping)]

    def communications(self, conversation_id: str) -> list[Mapping[str, Any]]:
        got = self._req("GET", f"{ORCHESTRATOR_BASE}/v2/Conversations/{conversation_id}/Communications",
                        "communications.list", query={"pageSize": 100})
        return [c for c in got.get("communications") or [] if isinstance(c, Mapping)]

    def close_conversation(self, conversation_id: str) -> bool:
        return self._orchestrator.close(conversation_id)

    def start_transcription(self, call_sid: str, params: Mapping[str, str]) -> bool:
        return self._calls.start_transcription(call_sid, params)

    def stop_transcription(self, call_sid: str, name: str) -> bool:
        return self._calls.stop_transcription(call_sid, name)


# --------------------------------------------------------------------------- the run


@dataclass
class Report:
    mode: str
    run_id: str
    customer_call: str | None = None
    task_sid: str | None = None
    conversation_id: str | None = None
    transcription_params: dict[str, str] | None = None
    transcription_started: bool | None = None
    transfer: str | None = None
    long_wait: list[tuple[int, str]] = field(default_factory=list)
    orchestrator: dict[str, Any] = field(default_factory=dict)
    callbacks: dict[str, Any] = field(default_factory=dict)
    conversation_closed: str | None = None
    error: str | None = None
    leftovers: list[str] = field(default_factory=list)


def _mask(value: Any) -> Any:
    """Phone numbers in a report value become ``+1******1234``."""
    if isinstance(value, str):
        return PHONE_LIKE.sub(lambda m: mask_phone(re.sub(r"[^\d+]", "", m.group(0))), value)
    return value


def summarize_participants(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in rows:
        addresses = p.get("addresses") if isinstance(p.get("addresses"), list) else []
        out.append({"type": p.get("type"), "name": _mask(p.get("name")),
                    "addresses": [{"channel": a.get("channel"), "address": _mask(a.get("address"))}
                                  for a in addresses if isinstance(a, Mapping)]})
    return out


def summarize_communications(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """PROVISIONAL shape: author address and participant, and the text (cut to 160 characters)."""
    out = []
    for c in rows:
        author = c.get("author") if isinstance(c.get("author"), Mapping) else {}
        content = c.get("content") if isinstance(c.get("content"), Mapping) else {}
        text = content.get("text") if isinstance(content.get("text"), str) else c.get("text")
        out.append({"author": _mask(author.get("address")), "channel": author.get("channel"),
                    "participant": author.get("participantId") or author.get("type"),
                    "text": _mask(text[:160]) if isinstance(text, str) else None})
    return out


def callback_delivery(events: Iterable[Mapping[str, Any]], alerts: Iterable[Mapping[str, Any]],
                      callback_url: str | None) -> dict[str, Any]:
    """Per event of the task: "accepted (no delivery alert)" or "rejected (<code>)" from the Monitor
    alerts on the workspace's callback URL (the front door answers a bad signature with 403)."""
    if not callback_url:
        return {"url": None, "note": "the workspace has no event callback URL: no callbacks are sent",
                "events": [{"event": e.get("event_type"), "delivery": "not sent"} for e in events]}
    target = urllib.parse.urlsplit(callback_url)
    rejected: dict[str, str] = {}
    unmatched = 0
    for a in alerts:
        url = a.get("request_url")
        if not isinstance(url, str):
            continue
        parts = urllib.parse.urlsplit(url)
        if (parts.hostname, parts.path) != (target.hostname, target.path):
            continue
        variables = urllib.parse.parse_qs(a.get("request_variables") or "")
        event_type = (variables.get("EventType") or [None])[0]
        label = f"rejected (alert {a.get('error_code')})"
        if event_type:
            rejected[event_type] = label
        else:
            unmatched += 1
    rows = [{"event": e.get("event_type"), "at": e.get("event_date"),
             "delivery": rejected.get(str(e.get("event_type")), "accepted (no delivery alert)")}
            for e in events]
    return {"url": f"https://{target.hostname}{target.path}", "events": rows,
            "unmatched_alerts": unmatched,
            "note": "signature status is inferred: Twilio signs every callback and the front door "
                    "rejects a bad signature with 403, which Twilio records as an alert"}


class Session:
    """One rehearsal: :meth:`run` executes, always cleans up, prints the report and returns the exit code."""

    def __init__(self, cfg: Settings, api: Api, *, clock: Callable[[], float],
                 sleep: Callable[[float], None], out: Callable[[str], None], run_id: str) -> None:
        self.cfg = cfg
        self.api = api
        self.clock = clock
        self.sleep = sleep
        self.out = out
        self.report = Report(mode=cfg.mode, run_id=run_id)
        self.name = NAME_PREFIX + run_id
        self.started = clock()
        self.deadline = self.started + cfg.cap_s
        self.calls: dict[str, str] = {}  # call SID -> role
        self.tasks: set[str] = set()
        self.calls_placed = 0
        self.legs = 0

    # ------------------------------------------------------------------ caps

    def _place_call(self, *, to: str, twiml_doc: str) -> str:
        if self.calls_placed >= MAX_CALLS:
            raise Abort(f"cap: at most {MAX_CALLS} call per run")
        self.calls_placed += 1
        self.deadline = self.clock() + self.cfg.cap_s
        sid = self.api.create_call(to=to, from_=self.cfg.from_, twiml=twiml_doc, time_limit_s=self.cfg.cap_s)
        self.calls[sid] = "caller"
        return sid

    def _new_leg(self) -> None:
        if self.legs >= MAX_LEGS:
            raise Abort(f"cap: at most {MAX_LEGS} legs besides the caller")
        self.legs += 1

    def _remaining_s(self) -> int:
        return max(0, int(self.deadline - self.clock()))

    def _wait(self, what: str, probe: Callable[[], Any], interval: float = POLL_S) -> Any:
        while True:
            found = probe()
            if found:
                return found
            if self.clock() + interval > self.deadline:
                raise Abort(f"timed out waiting for {what} (call cap {self.cfg.cap_s // 60} min)")
            self.sleep(interval)

    def _pause(self, seconds: float) -> None:
        """Sleep, but never past the call cap."""
        self.sleep(max(0.0, min(seconds, self.deadline - self.clock())))

    # ------------------------------------------------------------------ steps

    def execute(self) -> None:
        cfg, api, rep = self.cfg, self.api, self.report
        if cfg.mode == "standalone":
            rep.conversation_id = api.create_conversation(str(cfg.configuration_id), self.name)
            script_step(slog, SCRIPT, "conversation_created", "changed")
            customer = self._place_call(to=cfg.to, twiml_doc=standalone_twiml(cfg, rep.run_id))
            self.calls[customer] = "customer"
        else:
            self._place_call(to=cfg.to, twiml_doc=caller_twiml(cfg))
            customer = self._wait("the call at the front door", self._inbound_leg)
            self.calls[customer] = "customer"
        rep.customer_call = customer
        script_step(slog, SCRIPT, "call_placed", "changed", call_sid=customer)

        rep.task_sid = self._wait("the call's task", lambda: self._own_task(customer))
        script_step(slog, SCRIPT, "task_found", "ok", task_sid=rep.task_sid)
        if cfg.mode == "frontdoor":
            rep.conversation_id = api.resolve_conversation(customer)

        if cfg.long_wait_min:
            self._long_wait(customer)

        reservation = self._wait("a reservation (is a worker Available in the queue?)",
                                 lambda: api.pending_reservation(str(rep.task_sid)))
        self._new_leg()
        api.accept_conference(str(rep.task_sid), reservation, from_=cfg.from_, to=cfg.agent)
        script_step(slog, SCRIPT, "reservation_accepted", "changed", task_sid=rep.task_sid)
        conference = self._wait("the conference", lambda: api.conference_for_task(str(rep.task_sid)))
        agent = self._wait("the agent leg", lambda: self._other_leg(conference, customer))
        self.calls[agent] = "agent"

        if cfg.mode == "standalone":
            self._start_transcription(customer)
        elif rep.conversation_id is None:
            rep.conversation_id = api.resolve_conversation(customer)

        self._pause(cfg.talk_s)
        if cfg.transfer:
            self._new_leg()
            manager = api.add_participant(conference, from_=cfg.from_, to=cfg.manager,
                                          time_limit_s=max(1, self._remaining_s()))
            self.calls[manager] = "manager"
            api.end_call(agent)
            rep.transfer = "cold: manager added to the conference, agent leg ended"
            script_step(slog, SCRIPT, "transferred", "changed")
            self._pause(cfg.talk_s)

        api.end_call(customer)
        script_step(slog, SCRIPT, "hung_up", "changed")
        self.sleep(cfg.settle_s)

    def _inbound_leg(self) -> str | None:
        for c in self.api.find_calls(from_=self.cfg.from_, to=self.cfg.to, since=self.started):
            if c.get("direction") == "inbound" and c.get("sid") not in self.calls:
                return str(c["sid"])
        return None

    def _own_task(self, call_sid: str) -> str | None:
        """The task of this call only: the server filter is checked again here, so another task is
        never returned (never touches the queue's other tasks)."""
        for t in self.api.find_tasks(call_sid):
            if t.get("call_sid") == call_sid and _TASK.match(str(t.get("sid", ""))):
                self.tasks.add(str(t["sid"]))
                return str(t["sid"])
        return None

    def _other_leg(self, conference: str, customer: str) -> str | None:
        for sid in self.api.conference_calls(conference):
            if sid != customer and sid not in self.calls:
                return sid
        return None

    def _long_wait(self, customer: str) -> None:
        """No agent leg; the caller waits with hold music. The conversation's status is recorded
        at the start and every :data:`LONG_WAIT_POLL_S` seconds until ``--long-wait`` minutes have passed."""
        start = self.clock()
        end = start + self.cfg.long_wait_min * 60
        while True:
            minute = int((self.clock() - start) // 60)
            self.report.long_wait.append((minute, self._observe_status(customer)))
            if self._observe_call(customer) not in OPEN_CALL:
                raise Abort("the caller left the queue during the long wait")
            remaining = end - self.clock()
            if remaining <= 0:
                return
            self.sleep(min(LONG_WAIT_POLL_S, remaining))

    def _observe_status(self, customer: str) -> str:
        try:
            if self.report.conversation_id is None and self.cfg.mode == "frontdoor":
                self.report.conversation_id = self.api.resolve_conversation(customer)
            if self.report.conversation_id is None:
                return "no conversation"
            return str(self.api.conversation(self.report.conversation_id).get("status") or "unknown")
        except ScriptError as exc:
            return f"unreadable ({exc})"

    def _observe_call(self, call_sid: str) -> str:
        try:
            return self.api.call_status(call_sid)
        except ScriptError:
            return "in-progress"  # unknown: keep waiting; the call's own TimeLimit still caps it

    def _start_transcription(self, customer: str) -> None:
        """After the accept, on the customer leg, bound to the conversation, with every transcription attribute."""
        cfg = self.cfg
        params = humanleg.transcription_start(
            self.report.conversation_id, enabled=True, language=cfg.language,
            status_url=cfg.status_callback_url, engine=cfg.engine, speech_model=cfg.speech_model)
        missing = [k for k in P10_KEYS if not (params or {}).get(k)]
        if params is None or missing:
            raise Abort(f"transcription start lacks the required attributes: {', '.join(missing) or 'all'}")
        self.report.transcription_params = dict(params)
        self.report.transcription_started = self.api.start_transcription(customer, params)
        script_step(slog, SCRIPT, "transcription_start",
                    "changed" if self.report.transcription_started else "error")

    # ------------------------------------------------------------------ cleanup

    def _retry(self, what: str, action: Callable[[], Any]) -> bool:
        for attempt in range(CLEANUP_ATTEMPTS):
            try:
                result = action()
            except Exception as exc:  # noqa: BLE001 - cleanup must go on
                script_error(slog, SCRIPT, exc, step=f"cleanup_{what}", decision="retry", attempt=attempt)
            else:
                if result is not False:
                    return True
            if attempt + 1 < CLEANUP_ATTEMPTS:
                self.sleep(CLEANUP_RETRY_S)
        return False

    def _discover_calls(self) -> list[Mapping[str, Any]]:
        found: list[Mapping[str, Any]] = []
        for number in self.cfg.numbers:
            rows: list[Mapping[str, Any]] = []

            def probe(n: str = number, sink: list[Mapping[str, Any]] = rows) -> None:
                sink[:] = self.api.find_calls(from_=self.cfg.from_, to=n, since=self.started)

            if self._retry("find_calls", probe):
                found.extend({**r, "to": number} for r in rows)
            else:
                self.report.leftovers.append(f"calls to {mask_phone(number)}: could not be listed")
        return found

    def cleanup(self) -> None:
        """Stop the transcription, end every call, close the task; never raises. The conversation is
        closed by :meth:`close_conversation` after the report has been read."""
        rep, api = self.report, self.api
        customer = rep.customer_call
        if self.calls_placed == 0:
            return  # nothing was dialled, so no call, task or transcription exists
        if rep.transcription_started and customer:
            self._retry("stop_transcription", lambda: api.stop_transcription(customer, humanleg.TRANSCRIPTION_NAME))
        if self.calls_placed:
            listed = self._discover_calls()
            for c in listed:
                sid = str(c.get("sid"))
                # The customer leg: our own call in standalone, the front door's inbound leg in frontdoor.
                leg = "outbound-api" if self.cfg.mode == "standalone" else "inbound"
                if not customer and c.get("direction") == leg and c.get("to") == self.cfg.to:
                    customer = rep.customer_call = sid
                if sid not in self.calls:
                    self.calls[sid] = "found"
            for sid, role in list(self.calls.items()):
                if not self._retry("end_call", lambda s=sid: api.end_call(s)):
                    rep.leftovers.append(f"call {sid} ({role}) may still be up (its TimeLimit ends it)")
            if customer:
                tasks: list[str] = []

                def find() -> None:
                    tasks[:] = [str(t["sid"]) for t in api.find_tasks(customer)
                                if t.get("call_sid") == customer and _TASK.match(str(t.get("sid", "")))]

                if self._retry("find_tasks", find):
                    self.tasks.update(tasks)
                elif not self.tasks:
                    rep.leftovers.append("the call's task could not be looked up")
            for task in sorted(self.tasks):
                if not self._retry("close_task", lambda t=task: self._close_task(t)):
                    rep.leftovers.append(f"task {task} may still be open")
            if rep.task_sid is None and self.tasks:
                rep.task_sid = sorted(self.tasks)[0]

    def _close_task(self, task_sid: str) -> None:
        status = self.api.task_status(task_sid)
        if status in ("pending", "reserved"):
            self.api.set_task_status(task_sid, "canceled")
        elif status in ("assigned", "wrapping"):
            self.api.set_task_status(task_sid, "completed")

    def close_conversation(self) -> None:
        rep, api, cfg = self.report, self.api, self.cfg
        cid = rep.conversation_id
        if cid is None:
            found: list[str | None] = [None]

            def find() -> bool:
                if cfg.mode == "standalone":
                    found[0] = api.find_conversation(str(cfg.configuration_id), self.name)
                    return True
                # The shared client answers None for "none" and for a failure alike: ask again.
                found[0] = api.resolve_conversation(str(rep.customer_call))
                return found[0] is not None

            if cfg.mode == "frontdoor" and not rep.customer_call:
                return  # no call reached the front door, so no conversation exists for it
            if not self._retry("find_conversation", find):
                if cfg.mode == "standalone":
                    rep.leftovers.append("the test conversation could not be looked up")
                else:
                    rep.conversation_closed = "no ACTIVE conversation found for the call"
                return
            cid = rep.conversation_id = found[0]
        if cid is None:
            return
        if rep.orchestrator.get("status") == "CLOSED":
            rep.conversation_closed = "already CLOSED before the script's close"
        if self._retry("close_conversation", lambda: api.close_conversation(cid)):
            rep.conversation_closed = rep.conversation_closed or "closed by the script"
        else:
            rep.leftovers.append(f"conversation {cid} may still be open")

    # ------------------------------------------------------------------ report

    def collect(self) -> None:
        """What Orchestrator holds and which callbacks arrived; best effort, never raises."""
        rep, api = self.report, self.api
        try:
            if rep.conversation_id is None and self.cfg.mode == "frontdoor" and rep.customer_call:
                rep.conversation_id = api.resolve_conversation(rep.customer_call)
            if rep.conversation_id:
                cid = rep.conversation_id
                rep.orchestrator = {"status": api.conversation(cid).get("status"),
                                    "participants": summarize_participants(api.participants(cid)),
                                    "communications": summarize_communications(api.communications(cid))}
        except Exception as exc:  # noqa: BLE001 - a report gap, not a failure
            rep.orchestrator["error"] = str(exc) if isinstance(exc, ScriptError) else type(exc).__name__
        try:
            if rep.task_sid:
                rep.callbacks = callback_delivery(api.task_events(rep.task_sid),
                                                  api.alerts(self.started, self.clock()), api.callback_url())
        except Exception as exc:  # noqa: BLE001
            rep.callbacks = {"error": str(exc) if isinstance(exc, ScriptError) else type(exc).__name__}

    def print_report(self) -> None:
        rep, out = self.report, self.out
        out(f"\n== Rehearsal {rep.run_id} ({rep.mode}) ==")
        out(f"customer call: {rep.customer_call}   task: {rep.task_sid}   conversation: {rep.conversation_id}")
        if rep.transcription_params is not None:
            out(f"transcription start (Calls API, after accept): started={rep.transcription_started} "
                f"params={json.dumps(rep.transcription_params, sort_keys=True)}")
        if rep.transfer:
            out(f"transfer: {rep.transfer}")
        for minute, status in rep.long_wait:
            out(f"long wait: minute {minute:>3}: conversation {status}")
        orch = rep.orchestrator
        out(f"Orchestrator status: {orch.get('status')}" + (f" (read error: {orch['error']})"
                                                          if orch.get("error") else ""))
        for p in orch.get("participants", []):
            out(f"  participant: {json.dumps(p, sort_keys=True)}")
        out(f"  communications: {len(orch.get('communications', []))}")
        for c in orch.get("communications", []):
            out(f"    {c.get('participant') or c.get('author')}: {c.get('text')}")
        cb = rep.callbacks
        out(f"TaskRouter callbacks to {cb.get('url')}:" + (f" (read error: {cb['error']})"
                                                           if cb.get("error") else ""))
        for e in cb.get("events", []):
            out(f"  {e.get('event')}: {e.get('delivery')}")
        if cb.get("note"):
            out(f"  ({cb['note']})")
        out(f"conversation close: {rep.conversation_closed}")
        if rep.error:
            out(f"run stopped: {rep.error}")
        if rep.leftovers:
            out("NOT CLEANED UP (check the Console):")
            for item in rep.leftovers:
                out(f"  - {item}")
        else:
            out("cleanup: no call, task or conversation left open")
        if self.cfg.report_json:
            self.cfg.report_json.write_text(json.dumps(rep.__dict__, indent=2, default=str) + "\n",
                                            encoding="utf-8")

    # ------------------------------------------------------------------ run

    def run(self) -> int:
        code = 0
        try:
            self.execute()
        except ScriptError as exc:
            self.report.error = str(exc)
            script_error(slog, SCRIPT, exc, step="execute", decision="cleanup")
            code = 1
        except KeyboardInterrupt:
            self.report.error = "interrupted"
            code = 130
        except Exception as exc:  # noqa: BLE001 - any failure still cleans up
            self.report.error = type(exc).__name__
            script_error(slog, SCRIPT, exc, step="execute", decision="cleanup")
            code = 1
        finally:
            self.cleanup()
            self.collect()
            self.close_conversation()
        self.print_report()
        return 3 if self.report.leftovers else code


# --------------------------------------------------------------------------- preflight and CLI


def preflight(cfg: Settings, api: Api) -> None:
    """The account answers as the expected, active account; the numbers we dial from or to are ours."""
    try:
        acct = api.account()
    except ApiError as exc:
        raise Refusal(f"refusing to run: the account could not be read ({exc})") from None
    if acct.get("sid") != cfg.account_sid or acct.get("status") != "active":
        raise Refusal("refusing to run: the credentials' account is not the expected active account")
    must_own = [("--from", cfg.from_), ("--agent", cfg.agent), ("--manager", cfg.manager)]
    if cfg.mode == "frontdoor":
        must_own.append(("--to", cfg.to))
    for flag, number in must_own:
        try:
            owned = api.owns_number(number)
        except ApiError as exc:
            raise Refusal(f"refusing to run: could not check {flag} ({exc})") from None
        if not owned:
            raise Refusal(f"refusing to run: {flag} {mask_phone(number)} is not a number of this account")


ApiFactory = Callable[[Settings], Api]


def secrets_api_factory(args: argparse.Namespace, sm_client: Any | None = None) -> ApiFactory:
    """The API from the ``twilio_api_key`` secret (``--cdk-outputs`` or ``--secret``). A missing ARN,
    an unreadable secret or a key of another account is a :class:`ScriptError` (exit 2)."""
    def make(cfg: Settings) -> Api:
        arn = secret_arn("twilio_api_key", load_cdk_outputs(args.cdk_outputs),
                         parse_secret_overrides(args.secret))
        client = sm_client
        if client is None:
            import boto3

            client = boto3.client("secretsmanager", region_name=resolve_region(args.region, arns=[arn]))
        creds = read_secret("twilio_api_key", arn, client)
        if creds["account_sid"] != cfg.account_sid:
            raise ScriptError("refusing to run: the twilio_api_key secret belongs to another account "
                              "than the account in use")
        return TwilioRehearsalApi(cfg.account_sid, creds["api_key"], creds["api_secret"], cfg.workspace_sid)

    return make


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rehearse the human leg with one real call, no Flex login. "
                                            "Places a billed call: see the module docstring.")
    p.add_argument("mode", choices=("standalone", "frontdoor"))
    p.add_argument("--yes", action="store_true", help="confirm: place one real, billed call")
    p.add_argument("--expect-account-sid", help="must equal the account in use")
    p.add_argument("--account-sid", help="the account in use (default: OWLBANK_TWILIO_ACCOUNT_SID)")
    p.add_argument("--from", dest="from_", help="caller ID: a number of the account (E.164)")
    p.add_argument("--to", help="standalone: the caller's end; frontdoor: the deployed front door number")
    p.add_argument("--agent", help="a spare number of the account, answered by TwiML (the agent)")
    p.add_argument("--manager", help="a spare number of the account, answered by TwiML (the manager)")
    p.add_argument("--transfer", action="store_true", help="cold-transfer from the agent to the manager")
    p.add_argument("--workspace-sid", help="the Flex TaskRouter workspace (WS...)")
    p.add_argument("--workflow-sid", help="standalone: the workflow <Enqueue> uses (WW...)")
    p.add_argument("--configuration-id", help="standalone: the Conversation Configuration id")
    p.add_argument("--status-callback-url", help="standalone: transcription status URL on the front door host")
    p.add_argument("--language", default="en-US", help="the call's language (default en-US)")
    p.add_argument("--engine", help="transcription engine (default: the catalog's, for --language)")
    p.add_argument("--speech-model", help="speech model (default: the catalog's, for --language)")
    p.add_argument("--long-wait", type=int, metavar="MINUTES",
                   help=f"wait in the queue this long first (1-{MAX_LONG_WAIT_MINUTES}); status every 5 min")
    p.add_argument("--confirm-long-wait", type=int, metavar="MINUTES", help="the same MINUTES again")
    p.add_argument("--talk", type=int, default=30, help="seconds with each agent leg (default 30)")
    p.add_argument("--settle", type=int, default=20, help="seconds after hang-up before the report")
    p.add_argument("--caller-pause", type=int, default=15, help="frontdoor: seconds before the digits")
    p.add_argument("--caller-digits", default="0", help="frontdoor: DTMF the caller sends (default 0)")
    p.add_argument("--report-json", type=Path, help="also write the report as JSON here")
    add_secret_args(p)
    return p


def main(argv: Sequence[str] | None = None, *, api_factory: ApiFactory | None = None,
         environ: Mapping[str, str] | None = None, clock: Callable[[], float] = time.time,
         sleep: Callable[[float], None] = time.sleep,
         stt_loader: Callable[[str], humanleg.SttSettings] = catalog_stt,
         out: Callable[[str], None] = print, run_id: str | None = None) -> int:
    args = build_parser().parse_args(argv)
    script_start(slog, SCRIPT, mode=args.mode)
    try:
        cfg = validate(args, environ=os.environ if environ is None else environ, stt_loader=stt_loader)
    except Refusal as exc:
        script_error(slog, SCRIPT, exc, step="validate", decision="exit_2")
        out(str(exc))
        return 2
    est = estimate_minutes(cfg)
    out(f"Estimated billable minutes (upper bound): voice {est['voice']}, transcription {est['transcription']}"
        f" (one call, capped at {cfg.cap_s // 60} min"
        + (", plus the AI segment's own minutes)" if cfg.mode == "frontdoor" else ")"))
    try:
        api = (api_factory or secrets_api_factory(args))(cfg)
        preflight(cfg, api)
    except ScriptError as exc:
        script_error(slog, SCRIPT, exc, step="preflight", decision="exit_2")
        out(str(exc))
        return 2
    session = Session(cfg, api, clock=clock, sleep=sleep, out=out, run_id=run_id or uuid.uuid4().hex[:8])
    return session.run()


if __name__ == "__main__":
    start_logging(SCRIPT)
    sys.exit(main())
