"""TaskRouter reads over REST: a task's attributes and its reservations.

Shared by the Context API's per-task check (``lambdas.context_api``) and the front door's
participant repair at close (:func:`owlbank.humanleg_client.taskrouter_reservations`). HTTP
Basic with the ``twilio/api-key`` secret's API key and secret, https only. Any failure is
:class:`TaskRouterUnavailable` (fail closed); its text never carries the key or secret.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from owlbank import obs

TASKROUTER_URL = "https://taskrouter.twilio.com/v1/Workspaces/{workspace_sid}/Tasks/{task_sid}"
#: Seconds for one read when the caller gives none (passed to ``urlopen``, which applies it to each
#: socket operation). Callers with a budget over several reads pass ``timeout_s`` per read.
DEFAULT_TIMEOUT_S = 3.0
_RESERVATION_PAGE = 100

_TASK_SID_RE = re.compile(r"^WT[0-9a-fA-F]{32}$")
_WORKSPACE_SID_RE = re.compile(r"^WS[0-9a-fA-F]{32}$")


class TaskRouterUnavailable(Exception):
    """TaskRouter could not be asked (network, credentials, HTTP error, odd answer): no decision."""


class TaskRouterReader(Protocol):
    def task_attributes(self, task_sid: str, *, timeout_s: float | None = None) -> Mapping[str, Any] | None:
        """The task's parsed attributes, or ``None`` if the task doesn't exist. ``timeout_s``: this
        read's limit (``None``: the reader's own). Raises :class:`TaskRouterUnavailable` when no
        answer can be had."""
        ...

    def reservations(self, task_sid: str, *, timeout_s: float | None = None) -> Sequence[Mapping[str, Any]]:
        """The task's reservations as ``{"worker_sid", "status"}`` (as TaskRouter sent them). Raises
        :class:`TaskRouterUnavailable` when no answer can be had."""
        ...


def _require_https(url: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise TaskRouterUnavailable("TaskRouter URL must be https")


class TwilioTaskRouterReader:
    """:class:`TaskRouterReader` over TaskRouter's REST API with HTTP Basic (API key) credentials.

    ``credentials`` returns ``(api_key, api_secret)`` (the ``twilio/api-key`` secret), read on each
    use (``owlbank.secrets`` caches it). https only, short timeout; a missing workspace, bad SID,
    network error or non-2xx answer other than 404 on the task is :class:`TaskRouterUnavailable`.
    The error text never carries the key or secret."""

    def __init__(self, workspace_sid: str | None, credentials: Callable[[], tuple[str, str]], *,
                 timeout_s: float = DEFAULT_TIMEOUT_S,
                 opener: Callable[[urllib.request.Request, float], Any] | None = None) -> None:
        self._workspace = workspace_sid
        self._credentials = credentials
        self._timeout_s = timeout_s
        self._opener = opener or self._open

    @staticmethod
    def _open(req: urllib.request.Request, timeout: float) -> Any:
        _require_https(req.full_url)
        return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310  # nosec B310 (https checked)

    def _get(self, task_sid: str, suffix: str = "", query: str = "", *,
             timeout_s: float | None = None) -> tuple[int, Any]:
        if not isinstance(self._workspace, str) or not _WORKSPACE_SID_RE.match(self._workspace):
            raise TaskRouterUnavailable("TaskRouter workspace is not configured")
        if not _TASK_SID_RE.match(task_sid):
            raise TaskRouterUnavailable("bad task SID")
        timeout = self._timeout_s if timeout_s is None else timeout_s
        if not timeout > 0:
            raise TaskRouterUnavailable("no time left for the TaskRouter read")
        try:
            key, secret = self._credentials()
        except Exception as exc:
            raise TaskRouterUnavailable(f"credentials unavailable: {type(exc).__name__}") from None
        url = TASKROUTER_URL.format(workspace_sid=self._workspace, task_sid=task_sid) + suffix + query
        _require_https(url)
        basic = base64.b64encode(f"{key}:{secret}".encode()).decode()
        req = urllib.request.Request(  # noqa: S310 (https checked just above)
            url, headers={"Authorization": f"Basic {basic}", "Accept": "application/json"},
            method="GET")
        try:
            with obs.timed_call("twilio_taskrouter", "tasks.read", expected_codes=("404",)), \
                    self._opener(req, timeout) as resp:
                status, body = getattr(resp, "status", 200), resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return 404, None
            raise TaskRouterUnavailable(f"TaskRouter returned HTTP {exc.code}") from None
        except Exception as exc:
            raise TaskRouterUnavailable(f"TaskRouter call failed: {type(exc).__name__}") from None
        if status == 404:
            return 404, None
        if status != 200:
            raise TaskRouterUnavailable(f"TaskRouter returned HTTP {status}")
        try:
            data = json.loads(body)
        except ValueError:
            raise TaskRouterUnavailable("TaskRouter answer is not JSON") from None
        if not isinstance(data, Mapping):
            raise TaskRouterUnavailable("TaskRouter answer is not an object")
        return 200, data

    def task_attributes(self, task_sid: str, *, timeout_s: float | None = None) -> Mapping[str, Any] | None:
        status, data = self._get(task_sid, timeout_s=timeout_s)
        if status == 404 or data is None:
            return None
        raw = data.get("attributes")
        try:
            attrs = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            raise TaskRouterUnavailable("task attributes are not JSON") from None
        if not isinstance(attrs, Mapping):
            raise TaskRouterUnavailable("task attributes are not an object")
        return attrs

    def reservations(self, task_sid: str, *, timeout_s: float | None = None) -> Sequence[Mapping[str, Any]]:
        status, data = self._get(task_sid, "/Reservations", f"?PageSize={_RESERVATION_PAGE}",
                                 timeout_s=timeout_s)
        if status == 404 or data is None:
            return []
        items = data.get("reservations")
        meta = data.get("meta")
        if not isinstance(items, list) or (isinstance(meta, Mapping) and meta.get("next_page_url")):
            raise TaskRouterUnavailable("reservation list is incomplete")  # fail closed
        return [{"worker_sid": r.get("worker_sid"), "status": r.get("reservation_status")}
                for r in items if isinstance(r, Mapping)]
