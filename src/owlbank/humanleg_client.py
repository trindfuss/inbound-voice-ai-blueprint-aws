"""Client for the human leg of a call in the Orchestrator conversation.

Operations on **existing** conversations of Twilio's Conversation Orchestrator (Conversations v2),
over HTTPS with the ``twilio_api_key`` secret (HTTP Basic, API key SID and secret):

* :meth:`~OrchestratorHumanLeg.resolve_conversation`: ``GET /v2/Conversations?channelId=<CallSid>&status=ACTIVE``;
  the id only when exactly one conversation matches (TAC's own rule), else ``None``.
* :meth:`~OrchestratorHumanLeg.latest_conversation` (the recap writer):
  ``GET /v2/Conversations?channelId=<CallSid>`` with no status filter; the newest by ``createdAt``.
* :meth:`~OrchestratorHumanLeg.add_human_participant`: ``POST /v2/Conversations/{id}/Participants`` with
  ``type`` ``HUMAN_AGENT``, once per worker (a second add is a no-op).
* :meth:`~OrchestratorHumanLeg.close`: ``PUT /v2/Conversations/{id}`` ``{"status": "CLOSED"}`` (the PUT the
  bridge's ``TACConversationCloser`` already uses; the docs show PUT, the local skill says PATCH). A
  conversation that is already CLOSED counts as closed, and is not sent a second request.
* :meth:`~OrchestratorHumanLeg.mark_handoff`: ``PUT /v2/Conversations/{id}`` with ``status``
  ``ACTIVE`` and the **full** metadata: the conversation's current metadata merged with the hand-off keys,
  because whether a PUT keeps omitted fields is undocumented. Limits are checked before any request
  (at most 8 keys, keys up to 128 and values up to 512 characters, all strings). Never sent to a
  conversation this client has seen CLOSED.

Two Twilio REST helpers for the accept and close paths:

* :class:`TwilioCalls`: start a named real-time transcription on a call when the first agent accepts
  (``POST /2010-04-01/Accounts/{AccountSid}/Calls/{CallSid}/Transcriptions.json``, the form parameters of
  :func:`owlbank.humanleg.transcription_start`), and stop it by the same name
  (``POST .../Calls/{CallSid}/Transcriptions/{name}.json``, ``Status=stopped``).
* :class:`TaskReservations`: a task's reservations (worker SID and status) for the participant repair at
  close, over :class:`owlbank.taskrouter.TwilioTaskRouterReader` (the Context API's reader).

Nothing here creates a conversation: there is no POST to ``/v2/Conversations`` in this module.

**Budget.** :func:`budget` puts a deadline over every request made inside it (each request's timeout is the
smaller of its own and what is left; none is sent once it has passed), so a sequence of calls on one webhook
stays within one total (the ``reservation.accepted`` handler's 2 s; the close path's
:data:`owlbank.humanleg_close.CLOSE_DEADLINE_S`). :class:`TaskReservations` honours it too.

**Contract.** No operation raises: a failure of any kind (credentials, network, timeout, HTTP error, bad
body) is logged with the call SID and conversation ID (never the credentials, the response body or the
exception message) and returned as ``None`` or ``False``. The caller is on a call's response path.

Provisional (not yet seen against the live API): the base URL and the paths above are the ones
TAC uses (checked in the installed ``tac`` package); the participant body (``_participant_body``) and the way
an existing participant is recognised (``_has_worker``) are guesses, isolated in those two functions. Also
unverified until: how a PUT treats omitted fields (``mark_handoff`` merges).
"""

from __future__ import annotations

import base64
import contextlib
import contextvars
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from owlbank import obs
from owlbank.obs import timed_call
from owlbank.taskrouter import TwilioTaskRouterReader

log = logging.getLogger(__name__)

#: TAC's Orchestrator client builds the same base (``https://conversations.twilio.com``).
DEFAULT_BASE_URL = "https://conversations.twilio.com"
DEFAULT_TIMEOUT_S = 3.0
#: More than one match is already "not exactly one"; a small page keeps the answer small.
LIST_PAGE_SIZE = 10
#: Page size and page cap of the communications and participants reads (a call's conversation is small).
READ_PAGE_SIZE = 100
READ_MAX_PAGES = 20
#: Conversation metadata limits (documented): at most 8 keys, values up to 512 characters.
METADATA_MAX_KEYS = 8
METADATA_MAX_KEY_LEN = 128
METADATA_MAX_VALUE_LEN = 512
#: Twilio's REST API (Calls, Transcriptions).
DEFAULT_API_BASE_URL = "https://api.twilio.com"

_CALL_SID_RE = re.compile(r"^CA[0-9a-fA-F]{32}$")
_WORKER_SID_RE = re.compile(r"^WK[0-9a-fA-F]{32}$")
#: Orchestrator ids look like ``comms_conversation_01...``; the check keeps anything else out of a URL path.
_CONVERSATION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_ACCOUNT_SID_RE = re.compile(r"^AC[0-9a-fA-F]{32}$")
#: A transcription name we set ourselves (``humanleg``); anything else stays out of the URL path.
_TRANSCRIPTION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

Opener = Callable[[urllib.request.Request, float], Any]


class HumanLegClient(Protocol):
    def resolve_conversation(self, call_sid: str) -> str | None:
        """The id of the one ACTIVE conversation for ``call_sid``; ``None`` for zero, several or any failure."""
        ...

    def add_human_participant(self, conversation_id: str, worker_sid: str, identity: str) -> bool:
        """``True`` if this call added the worker; ``False`` if already there, or on any failure."""
        ...

    def close(self, conversation_id: str) -> bool:
        """``True`` if the conversation is CLOSED afterwards (already closed included); ``False`` on failure."""
        ...

    def mark_handoff(self, conversation_id: str, metadata: Mapping[str, str]) -> bool:
        """``True`` if the conversation is ACTIVE with ``metadata`` merged in; ``False`` on bad input,
        a CLOSED conversation, or any failure."""
        ...


@dataclass(frozen=True)
class Reservation:
    """One TaskRouter reservation of a task: the worker and the reservation status (``accepted``...)."""

    worker_sid: str
    status: str


def metadata_problem(metadata: Any) -> str | None:
    """Why ``metadata`` breaks Orchestrator's limits (a short reason, no values), or ``None`` if it fits."""
    if not isinstance(metadata, Mapping):
        return "not_a_mapping"
    if len(metadata) > METADATA_MAX_KEYS:
        return "too_many_keys"
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            return "not_a_string"
        if not key or len(key) > METADATA_MAX_KEY_LEN:
            return "key_length"
        if len(value) > METADATA_MAX_VALUE_LEN:
            return "value_length"
    return None


def _created_at(conversation: Mapping[str, Any]) -> datetime | None:
    """A conversation's ``createdAt`` (ISO 8601, ``Z`` or an offset) as an aware time; ``None`` if absent,
    unparseable or without a zone."""
    raw = conversation.get("createdAt")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00") if raw.endswith("Z") else raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def newest_conversation(conversations: Sequence[Mapping[str, Any]]) -> str | None:
    """The id of the newest of ``conversations`` by ``createdAt``. Those without a
    usable ``createdAt`` count only when none has one: then one alone is taken and several give
    ``None``. Two sharing the newest time also give ``None`` (no guess)."""
    dated = [(t, str(c["id"])) for c in conversations if (t := _created_at(c)) is not None]
    if not dated:
        return str(conversations[0]["id"]) if len(conversations) == 1 else None
    newest = max(t for t, _ in dated)
    ids = {cid for t, cid in dated if t == newest}
    return ids.pop() if len(ids) == 1 else None


def _require_https(url: str) -> None:
    """Raise unless ``url`` is an ``https`` URL with a host (``urlopen`` also speaks ``file:`` and ``ftp:``).

    An explicit ``raise``, not an ``assert``, so ``python -O`` cannot remove it.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError("Orchestrator URL must be https")


def _default_opener(req: urllib.request.Request, timeout: float) -> Any:
    _require_https(req.full_url)
    # S310 is satisfied by the check above: https only, host from DEFAULT_BASE_URL (or the base URL
    # given at construction, which is checked too).
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310  # nosec B310


def _participant_body(worker_sid: str, identity: str) -> dict[str, Any]:
    """PROVISIONAL: the ``POST .../Participants`` body for a human agent.

    ``type`` is always set (``HUMAN_AGENT`` for a worker); ``type`` and ``name`` are fields TAC sends. The
    worker is a ``VOICE`` address ``client:<name>`` (the Flex
    agent's leg is a Voice SDK client, not a phone number) with the worker SID as ``channelId``, so a
    later add can recognise it (:func:`_has_worker`). Channel ``API`` got HTTP 400 (the
    v2 enum is VOICE, SMS, RCS, WHATSAPP, CHAT, VIDEO), and Orchestrator stores non-E.164 VOICE
    addresses (it recorded a withheld caller as ``Anonymous``). Change the body here only.
    """
    return {
        "type": "HUMAN_AGENT",
        "name": identity or worker_sid,
        "addresses": [{"channel": "VOICE", "address": f"client:{identity or worker_sid}",
                       "channelId": worker_sid}],
    }


def _has_worker(participant: Mapping[str, Any], worker_sid: str, identity: str = "") -> bool:
    """PROVISIONAL: is this listed participant the worker added by :func:`_participant_body`?"""
    own = f"client:{identity or worker_sid}"
    for address in participant.get("addresses") or []:
        if isinstance(address, Mapping) and worker_sid in (address.get("channelId"), address.get("address")):
            return True
        if isinstance(address, Mapping) and participant.get("type") == "HUMAN_AGENT" \
                and address.get("address") == own:
            return True
    return participant.get("name") == worker_sid


class _Reply:
    """HTTP status and parsed JSON object (``None`` when the body is empty or not an object)."""

    __slots__ = ("data", "status")

    def __init__(self, status: int, data: dict[str, Any] | None) -> None:
        self.status = status
        self.data = data

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


#: The monotonic deadline of the enclosing :func:`budget`, if any.
_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("owlbank_human_leg_deadline",
                                                                         default=None)


class BudgetSpent(TimeoutError):
    """The enclosing :func:`budget` has no time left: the request is not sent."""


@contextlib.contextmanager
def budget(seconds: float) -> Iterator[None]:
    """Every request of this module made inside the block shares one deadline ``seconds`` from now
    (nested budgets keep the earlier deadline). A request past it is not sent and fails like a timeout
    (the operations' never-raise contract turns it into ``False`` or ``None``)."""
    end = time.monotonic() + seconds
    outer = _deadline.get()
    token = _deadline.set(end if outer is None else min(outer, end))
    try:
        yield
    finally:
        _deadline.reset(token)


def budget_left() -> float | None:
    """Seconds left of the enclosing :func:`budget` (never below 0), or ``None`` outside one."""
    end = _deadline.get()
    if end is None:
        return None
    return max(0.0, end - time.monotonic())


def _timeout(timeout_s: float) -> float:
    """``timeout_s``, cut to what is left of the enclosing :func:`budget`; raises :class:`BudgetSpent`."""
    end = _deadline.get()
    if end is None:
        return timeout_s
    left = end - time.monotonic()
    if left <= 0:
        raise BudgetSpent("human-leg budget spent")
    return min(timeout_s, left)


def _send(opener: Opener, timeout_s: float, method: str, url: str, dependency: str, operation: str,
          auth: tuple[str, str], data: bytes | None, content_type: str) -> _Reply:
    """One HTTPS request with HTTP Basic ``auth``. An HTTP error status comes back as a :class:`_Reply`;
    anything else (network, timeout, a non-https URL, a spent :func:`budget`) raises. The credentials go
    in the header only."""
    _require_https(url)
    timeout_s = _timeout(timeout_s)
    basic = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
    headers = {"Accept": "application/json", "Authorization": f"Basic {basic}"}
    if data is not None:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, headers=headers, method=method)  # noqa: S310 (https checked above)
    try:
        with timed_call(dependency, operation), opener(req, timeout_s) as resp:
            status = int(getattr(resp, "status", 200))
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        return _Reply(exc.code, None)
    parsed: Any = None
    if raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
    return _Reply(status, parsed if isinstance(parsed, dict) else None)


class OrchestratorHumanLeg:
    """:class:`HumanLegClient` over Conversations v2.

    ``credentials`` returns ``(account_sid, api_key, api_secret)`` (see :func:`credentials_from_secret`);
    ``opener`` is injectable for tests.
    """

    def __init__(self, credentials: Callable[[], tuple[str, str, str]], *,
                 base_url: str = DEFAULT_BASE_URL, timeout_s: float = DEFAULT_TIMEOUT_S,
                 opener: Opener = _default_opener) -> None:
        _require_https(base_url)
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._opener = opener
        # Ids this process has seen CLOSED: a repeat sends nothing even if the status read fails.
        self._closed: dict[str, None] = {}

    # ------------------------------------------------------------------ operations

    def resolve_conversation(self, call_sid: str) -> str | None:
        if not _CALL_SID_RE.match(call_sid or ""):
            obs.emit(log, logging.WARNING, "human_leg_bad_call_sid", operation="resolve", decision="return_none")
            return None
        try:
            reply = self._request("GET", "/v2/Conversations", "conversations.list",
                                  query={"channelId": call_sid, "status": "ACTIVE", "pageSize": LIST_PAGE_SIZE})
            if not reply.ok or reply.data is None:
                self._log_http("human_leg_resolve_failed", reply, call_sid=call_sid)
                return None
            found = reply.data.get("conversations")
            if not isinstance(found, list):
                return None
            # The filter is asked for, but only ACTIVE ones count even if the server ignored it.
            active = [c for c in found if isinstance(c, Mapping) and c.get("status", "ACTIVE") == "ACTIVE"
                      and isinstance(c.get("id"), str) and _CONVERSATION_ID_RE.match(c["id"])]
            if len(active) != 1:
                obs.emit(log, logging.INFO, "human_leg_no_conversation", call_sid=call_sid,
                         matches=len(active), decision="return_none")
                return None
            return str(active[0]["id"])
        except Exception as exc:  # noqa: BLE001 - the contract: never raise
            obs.log_caught(log, "human_leg_resolve_failed", exc, call_sid=call_sid,
                           operation="resolve", decision="return_none")
            return None

    def latest_conversation(self, call_sid: str) -> str | None:
        """The recap writer's lookup: the newest conversation
        listed for ``call_sid`` by ``createdAt``, **whatever its status** (Orchestrator may already
        have made it INACTIVE or CLOSED by the time the writer runs). ``GET /v2/Conversations?
        channelId=<CallSid>`` with no status filter, one page. Of conversations without a usable
        ``createdAt``, one alone is taken; several are ``None`` (never a guess). ``None`` too for
        none listed or any failure. :meth:`resolve_conversation` (ACTIVE only) is unchanged."""
        if not _CALL_SID_RE.match(call_sid or ""):
            obs.emit(log, logging.WARNING, "human_leg_bad_call_sid", operation="latest",
                     decision="return_none")
            return None
        try:
            reply = self._request("GET", "/v2/Conversations", "conversations.list",
                                  query={"channelId": call_sid, "pageSize": LIST_PAGE_SIZE})
            if not reply.ok or reply.data is None:
                self._log_http("human_leg_latest_failed", reply, call_sid=call_sid)
                return None
            found = reply.data.get("conversations")
            listed = [c for c in found if isinstance(c, Mapping) and isinstance(c.get("id"), str)
                      and _CONVERSATION_ID_RE.match(c["id"])] if isinstance(found, list) else []
            picked = newest_conversation(listed)
            if picked is None:
                obs.emit(log, logging.INFO, "human_leg_no_conversation", call_sid=call_sid,
                         matches=len(listed), operation="latest", decision="return_none")
            return picked
        except Exception as exc:  # noqa: BLE001 - the contract: never raise
            obs.log_caught(log, "human_leg_latest_failed", exc, call_sid=call_sid,
                           operation="latest", decision="return_none")
            return None

    def add_human_participant(self, conversation_id: str, worker_sid: str, identity: str) -> bool:
        ids: dict[str, Any] = {"call_sid": obs.current_call_sid(), "conversation_id": conversation_id}
        if not _CONVERSATION_ID_RE.match(conversation_id or "") or not _WORKER_SID_RE.match(worker_sid or ""):
            obs.emit(log, logging.WARNING, "human_leg_bad_argument", operation="add_participant",
                     decision="return_false", **ids)
            return False
        if conversation_id in self._closed:  # Nothing after close
            obs.emit(log, logging.INFO, "human_leg_after_close", operation="add_participant",
                     decision="no_op", **ids)
            return False
        path = f"/v2/Conversations/{conversation_id}/Participants"
        try:
            listed = self._request("GET", path, "participants.list")
            if not listed.ok or listed.data is None:
                self._log_http("human_leg_add_failed", listed, **ids)
                return False
            present = listed.data.get("participants")
            if not isinstance(present, list):
                present = []
            if any(isinstance(p, Mapping) and _has_worker(p, worker_sid, identity) for p in present):
                obs.emit(log, logging.INFO, "human_leg_participant_present", decision="no_op", **ids)
                return False
            created = self._request("POST", path, "participants.create",
                                    body=_participant_body(worker_sid, identity))
            if created.status == 409:  # a concurrent add won the race
                obs.emit(log, logging.INFO, "human_leg_participant_present", decision="no_op", **ids)
                return False
            if not created.ok:
                self._log_http("human_leg_add_failed", created, **ids)
                return False
            return True
        except Exception as exc:  # noqa: BLE001 - the contract: never raise
            obs.log_caught(log, "human_leg_add_failed", exc, operation="add_participant",
                           decision="return_false", **ids)
            return False

    def close(self, conversation_id: str) -> bool:
        ids: dict[str, Any] = {"call_sid": obs.current_call_sid(), "conversation_id": conversation_id}
        if not _CONVERSATION_ID_RE.match(conversation_id or ""):
            obs.emit(log, logging.WARNING, "human_leg_bad_argument", operation="close",
                     decision="return_false", **ids)
            return False
        path = f"/v2/Conversations/{conversation_id}"
        if conversation_id in self._closed:
            obs.emit(log, logging.INFO, "human_leg_already_closed", decision="no_op", **ids)
            return True
        try:
            # Read first so a repeated close sends no second request, in this process or another.
            # A failed read is not fatal: the PUT below decides.
            current = self._request("GET", path, "conversations.get")
            if current.ok and (current.data or {}).get("status") == "CLOSED":
                obs.emit(log, logging.INFO, "human_leg_already_closed", decision="no_op", **ids)
                self._remember_closed(conversation_id)
                return True
            done = self._request("PUT", path, "conversations.close", body={"status": "CLOSED"})
            if done.ok:
                self._remember_closed(conversation_id)
                return True
            # A refused PUT can still mean "closed meanwhile" (CLOSED is final); look again.
            again = self._request("GET", path, "conversations.get")
            if again.ok and (again.data or {}).get("status") == "CLOSED":
                self._remember_closed(conversation_id)
                return True
            self._log_http("human_leg_close_failed", done, **ids)
            return False
        except Exception as exc:  # noqa: BLE001 - the contract: never raise
            obs.log_caught(log, "human_leg_close_failed", exc, operation="close",
                           decision="return_false", **ids)
            return False

    def mark_handoff(self, conversation_id: str, metadata: Mapping[str, str]) -> bool:
        ids: dict[str, Any] = {"call_sid": obs.current_call_sid(), "conversation_id": conversation_id}
        problem = metadata_problem(metadata)
        if not _CONVERSATION_ID_RE.match(conversation_id or "") or problem is not None:
            obs.emit(log, logging.WARNING, "human_leg_bad_argument", operation="mark_handoff",
                     reason=problem or "conversation_id", decision="return_false", **ids)
            return False
        if conversation_id in self._closed:  # Nothing after close
            obs.emit(log, logging.INFO, "human_leg_after_close", operation="mark_handoff",
                     decision="no_op", **ids)
            return False
        path = f"/v2/Conversations/{conversation_id}"
        try:
            # Read first: the PUT carries the full metadata (the bank's keys too), and a CLOSED
            # conversation is never written to again.
            current = self._request("GET", path, "conversations.get")
            if not current.ok or current.data is None:
                self._log_http("human_leg_handoff_failed", current, **ids)
                return False
            if current.data.get("status") == "CLOSED":
                self._remember_closed(conversation_id)
                obs.emit(log, logging.INFO, "human_leg_after_close", operation="mark_handoff",
                         decision="no_op", **ids)
                return False
            existing = current.data.get("metadata")
            if existing is None:
                existing = {}
            if metadata_problem(existing) not in (None, "too_many_keys"):
                # An unknown shape is not overwritten: a full PUT would lose what is there.
                obs.emit(log, logging.WARNING, "human_leg_handoff_failed", reason="metadata_unreadable",
                         decision="return_false", **ids)
                return False
            merged = {**existing, **metadata}
            problem = metadata_problem(merged)
            if problem is not None:
                obs.emit(log, logging.WARNING, "human_leg_handoff_failed", reason=problem,
                         decision="return_false", **ids)
                return False
            if merged == dict(existing) and current.data.get("status") == "ACTIVE":
                return True  # already marked: nothing to send
            done = self._request("PUT", path, "conversations.update",
                                 body={"status": "ACTIVE", "metadata": merged})
            if not done.ok:
                self._log_http("human_leg_handoff_failed", done, **ids)
                return False
            return True
        except Exception as exc:  # noqa: BLE001 - the contract: never raise
            obs.log_caught(log, "human_leg_handoff_failed", exc, operation="mark_handoff",
                           decision="return_false", **ids)
            return False

    def communications(self, conversation_id: str, *,
                       more: Callable[[], bool] | None = None) -> list[Mapping[str, Any]] | None:
        """All communications of the conversation (``GET .../Communications``), oldest page first;
        ``None`` on a bad id, any failure, more than :data:`READ_MAX_PAGES` pages, or ``more()``
        (asked before each page request; a caller with a deadline passes it) answering False (never
        a part)."""
        return self._read_all(conversation_id, "Communications", "communications", more)

    def participants(self, conversation_id: str, *,
                     more: Callable[[], bool] | None = None) -> list[Mapping[str, Any]] | None:
        """All participants of the conversation (``GET .../Participants``); ``None`` as for
        :meth:`communications`."""
        return self._read_all(conversation_id, "Participants", "participants", more)

    def _read_all(self, conversation_id: str, segment: str, key: str,
                  more: Callable[[], bool] | None = None) -> list[Mapping[str, Any]] | None:
        ids: dict[str, Any] = {"call_sid": obs.current_call_sid(), "conversation_id": conversation_id}
        operation = f"{key}.list"
        if not _CONVERSATION_ID_RE.match(conversation_id or ""):
            obs.emit(log, logging.WARNING, "human_leg_bad_argument", operation=operation,
                     decision="return_none", **ids)
            return None
        path = f"/v2/Conversations/{conversation_id}/{segment}"
        try:
            items: list[Mapping[str, Any]] = []
            token: str | None = None
            for _ in range(READ_MAX_PAGES):
                if more is not None and not more():
                    obs.emit(log, logging.WARNING, "human_leg_read_failed", operation=operation,
                             reason="deadline", decision="return_none", **ids)
                    return None
                query: dict[str, Any] = {"pageSize": READ_PAGE_SIZE}
                if token:
                    query["pageToken"] = token
                reply = self._request("GET", path, operation, query=query)
                page = reply.data.get(key) if reply.ok and reply.data is not None else None
                if not isinstance(page, list):
                    self._log_http("human_leg_read_failed", reply, operation=operation, **ids)
                    return None
                items.extend(i for i in page if isinstance(i, Mapping))
                meta = reply.data.get("meta") if reply.data else None
                nxt = meta.get("nextToken") if isinstance(meta, Mapping) else None
                if not isinstance(nxt, str) or not nxt:
                    return items
                token = nxt
            obs.emit(log, logging.WARNING, "human_leg_read_failed", operation=operation,
                     reason="page_cap", decision="return_none", **ids)
            return None
        except Exception as exc:  # noqa: BLE001 - the contract: never raise
            obs.log_caught(log, "human_leg_read_failed", exc, operation=operation,
                           decision="return_none", **ids)
            return None

    def _remember_closed(self, conversation_id: str) -> None:
        self._closed[conversation_id] = None
        while len(self._closed) > 1024:  # bounded: a warm Lambda sees many calls
            self._closed.pop(next(iter(self._closed)), None)

    # ------------------------------------------------------------------ HTTP

    def _request(self, method: str, path: str, operation: str, *,
                 query: Mapping[str, Any] | None = None, body: Mapping[str, Any] | None = None) -> _Reply:
        """One HTTPS call. An HTTP error status comes back as a :class:`_Reply`; anything else raises."""
        account_sid, username, password = self._credentials()  # (account_sid is unused: v2 is key-scoped)
        del account_sid
        url = self._base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = None if body is None else json.dumps(body).encode()
        return _send(self._opener, self._timeout_s, method, url, "twilio_orchestrator", operation,
                     (username, password), data, "application/json")

    @staticmethod
    def _log_http(event: str, reply: _Reply, /, **ids: Any) -> None:
        obs.emit(log, logging.WARNING, event, http_status=reply.status, decision="return_failure", **ids)


class TwilioCalls:
    """Twilio REST calls on a live call (Calls API). Same contract as :class:`OrchestratorHumanLeg`:
    nothing raises, a failure is logged with the call SID (never the key) and returned as ``False``.

    ``credentials`` returns ``(account_sid, api_key, api_secret)``; the account SID is in the path.
    """

    def __init__(self, credentials: Callable[[], tuple[str, str, str]], *,
                 base_url: str = DEFAULT_API_BASE_URL, timeout_s: float = DEFAULT_TIMEOUT_S,
                 opener: Opener = _default_opener) -> None:
        _require_https(base_url)
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._opener = opener

    def start_transcription(self, call_sid: str, params: Mapping[str, str]) -> bool:
        """Start a real-time transcription on ``call_sid`` with the Calls API form ``params``
        (:func:`owlbank.humanleg.transcription_start`: its ``Name`` is the one
        :meth:`stop_transcription` stops). ``True`` on 2xx; ``False`` on bad input or any failure,
        a duplicate name included (the hand-off goes on without a transcript: error table)."""
        ids: dict[str, Any] = {"call_sid": call_sid}
        name = params.get("Name") if isinstance(params, Mapping) else None
        if not _CALL_SID_RE.match(call_sid or "") or not isinstance(name, str) \
                or not _TRANSCRIPTION_NAME_RE.match(name) \
                or not all(isinstance(k, str) and isinstance(v, str) for k, v in params.items()):
            obs.emit(log, logging.WARNING, "human_leg_bad_argument", operation="start_transcription",
                     decision="return_false", call_sid=call_sid if _CALL_SID_RE.match(call_sid or "") else None)
            return False
        try:
            account_sid, key, secret = self._credentials()
            if not _ACCOUNT_SID_RE.match(account_sid or ""):
                obs.emit(log, logging.WARNING, "human_leg_start_failed", reason="account_sid",
                         decision="return_false", **ids)
                return False
            url = f"{self._base_url}/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}/Transcriptions.json"
            data = urllib.parse.urlencode(dict(params)).encode()
            reply = _send(self._opener, self._timeout_s, "POST", url, "twilio_calls", "transcriptions.start",
                          (key, secret), data, "application/x-www-form-urlencoded")
            if not reply.ok:
                obs.emit(log, logging.WARNING, "human_leg_start_failed", http_status=reply.status,
                         decision="return_failure", **ids)
                return False
            return True
        except Exception as exc:  # noqa: BLE001 - the contract: never raise
            obs.log_caught(log, "human_leg_start_failed", exc, operation="start_transcription",
                           decision="return_false", **ids)
            return False

    def stop_transcription(self, call_sid: str, name: str) -> bool:
        """Stop the transcription named ``name`` on ``call_sid`` (``Status=stopped``). ``True`` on 2xx;
        ``False`` on bad input or any failure (the caller closes anyway: error table)."""
        ids: dict[str, Any] = {"call_sid": call_sid}
        if not _CALL_SID_RE.match(call_sid or "") or not _TRANSCRIPTION_NAME_RE.match(name or ""):
            obs.emit(log, logging.WARNING, "human_leg_bad_argument", operation="stop_transcription",
                     decision="return_false", call_sid=call_sid if _CALL_SID_RE.match(call_sid or "") else None)
            return False
        try:
            account_sid, key, secret = self._credentials()
            if not _ACCOUNT_SID_RE.match(account_sid or ""):
                obs.emit(log, logging.WARNING, "human_leg_stop_failed", reason="account_sid",
                         decision="return_false", **ids)
                return False
            url = (f"{self._base_url}/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}"
                   f"/Transcriptions/{name}.json")
            data = urllib.parse.urlencode({"Status": "stopped"}).encode()
            reply = _send(self._opener, self._timeout_s, "POST", url, "twilio_calls", "transcriptions.stop",
                          (key, secret), data, "application/x-www-form-urlencoded")
            if not reply.ok:
                obs.emit(log, logging.WARNING, "human_leg_stop_failed", http_status=reply.status,
                         decision="return_failure", **ids)
                return False
            return True
        except Exception as exc:  # noqa: BLE001 - the contract: never raise
            obs.log_caught(log, "human_leg_stop_failed", exc, operation="stop_transcription",
                           decision="return_false", **ids)
            return False


class ReservationSource(Protocol):
    def reservations(self, task_sid: str, *, timeout_s: float | None = None) -> Sequence[Mapping[str, Any]]:
        """``[{"worker_sid", "status"}]`` for the task; raises when TaskRouter can't be asked.
        ``timeout_s``: this read's limit (``None``: the source's own)."""
        ...


class TaskReservations:
    """A task's reservations for the participant repair at close.

    Wraps :class:`owlbank.taskrouter.TwilioTaskRouterReader` (it fails closed on an incomplete page, and
    answers
    ``[]`` for an unknown task). :meth:`list` never raises: ``None`` on any failure.

    Inside a :func:`budget` the read's timeout is the smaller of ``timeout_s`` (the source's own when
    ``None``) and what is left, and no read is made once it is spent, as for the other clients here.
    """

    def __init__(self, source: ReservationSource, *, timeout_s: float | None = None) -> None:
        self._source = source
        self._timeout_s = timeout_s

    def list(self, task_sid: str) -> list[Reservation] | None:
        try:
            left = budget_left()
            if left is None:
                rows = self._source.reservations(task_sid)
            else:
                if left <= 0:
                    raise BudgetSpent("human-leg budget spent")
                cap = left if self._timeout_s is None else min(self._timeout_s, left)
                rows = self._source.reservations(task_sid, timeout_s=cap)
            return [Reservation(str(r["worker_sid"]), str(r["status"])) for r in rows
                    if isinstance(r, Mapping) and isinstance(r.get("worker_sid"), str)
                    and _WORKER_SID_RE.match(r["worker_sid"]) and isinstance(r.get("status"), str)]
        except Exception as exc:  # noqa: BLE001 - the contract: never raise
            obs.log_caught(log, "human_leg_reservations_failed", exc, call_sid=obs.current_call_sid(),
                           operation="reservations", decision="return_none")
            return None


def taskrouter_reservations(workspace_sid: str | None, credentials: Callable[[], tuple[str, str, str]], *,
                            timeout_s: float = DEFAULT_TIMEOUT_S,
                            opener: Opener | None = None) -> TaskReservations:
    """:class:`TaskReservations` over :class:`owlbank.taskrouter.TwilioTaskRouterReader`, with the same
    ``(account_sid, api_key, api_secret)`` credentials as the other clients here."""

    def key_and_secret() -> tuple[str, str]:
        _account, key, secret = credentials()
        return key, secret

    return TaskReservations(TwilioTaskRouterReader(workspace_sid, key_and_secret, timeout_s=timeout_s,
                                                   opener=opener), timeout_s=timeout_s)


def credentials_from_secret(arn: str | None) -> Callable[[], tuple[str, str, str]]:
    """Read ``(account_sid, api_key, api_secret)`` from the ``twilio_api_key`` secret (as the call updater does)."""

    def read() -> tuple[str, str, str]:
        from owlbank.secrets import get_secret

        value = get_secret(arn or "")
        return str(value["account_sid"]), str(value["api_key"]), str(value["api_secret"])

    return read
