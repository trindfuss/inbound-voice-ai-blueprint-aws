"""A fake Twilio for the close path's deadline tests, under a fake monotonic clock.

The real clients (:class:`OrchestratorHumanLeg`, :class:`TwilioCalls`, :class:`TaskReservations` over
:class:`TwilioTaskRouterReader`) send through :class:`SlowTwilio`, which answers every request a close
makes and advances :class:`FakeClock` by that request's delay. A request slower than the timeout it
was given takes the timeout and fails as a socket timeout does (``TimeoutError``).
"""

from __future__ import annotations

import contextlib
import json
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from typing import Any
from unittest import mock

from owlbank import callstate, humanleg_client
from owlbank.callstate import CallState
from owlbank.humanleg_client import OrchestratorHumanLeg, TwilioCalls, taskrouter_reservations
from owlbank.humanleg_close import HumanLegCloser
from owlbank.segments import SegmentLog
from owlbank.store import MemoryStore

ACCOUNT = "AC" + "4" * 32
CREDS = (ACCOUNT, "SK" + "3" * 32, "s3cr3t-value-xyz")
WORKSPACE = "WS" + "5" * 32
TASK = "WT" + "d" * 32
WORKERS = ("WK" + "a" * 32, "WK" + "b" * 32)
CONV = "conv_" + "a" * 26
T0 = 1_760_000_000
#: The front door's per-request timeout (``HUMAN_LEG_TIMEOUT_S``).
REQUEST_TIMEOUT_S = 2.0


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def monotonic(self) -> float:
        return self.t


class _Resp:
    def __init__(self, status: int, body: Any) -> None:
        self.status, self._body = status, json.dumps(body).encode()

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class SlowTwilio:
    """Orchestrator, Calls and TaskRouter in one opener. ``delays[i]`` is request ``i``'s time;
    past the list, ``default``. Records each request and the timeout it was sent with."""

    def __init__(self, clock: FakeClock, delays: Sequence[float] = (), default: float = 0.05) -> None:
        self.clock = clock
        self.delays = list(delays)
        self.default = default
        self.sent: list[tuple[str, str]] = []
        self.timeouts: list[float] = []
        self.closed = False

    def __call__(self, req: urllib.request.Request, timeout: float) -> _Resp:
        assert timeout > 0, "a request was sent with no time left"
        delay = self.delays.pop(0) if self.delays else self.default
        method, path = req.get_method(), urllib.parse.urlsplit(req.full_url).path
        self.sent.append((method, path))
        self.timeouts.append(timeout)
        if delay > timeout:
            self.clock.t += timeout
            raise TimeoutError("timed out")
        self.clock.t += delay
        return _Resp(*self._answer(method, path))

    def _answer(self, method: str, path: str) -> tuple[int, Any]:
        if path.endswith("/Reservations"):
            return 200, {"reservations": [{"worker_sid": w, "reservation_status": "accepted"}
                                          for w in WORKERS], "meta": {}}
        if "/Transcriptions/" in path:
            return 200, {"status": "stopped"}
        if path.endswith("/Participants"):
            return (200, {"participants": []}) if method == "GET" else (201, {})
        if path == "/v2/Conversations":
            return 200, {"conversations": [{"id": CONV, "status": "ACTIVE"}]}
        if method == "PUT":
            self.closed = True
            return 200, {"id": CONV, "status": "CLOSED"}
        return 200, {"id": CONV, "status": "CLOSED" if self.closed else "ACTIVE"}


@contextlib.contextmanager
def fake_monotonic(clock: FakeClock) -> Iterator[None]:
    """The budget's clock (``humanleg_client.time``) is ``clock`` inside the block."""
    with mock.patch.object(humanleg_client, "time", clock):
        yield


def escalated_call(store: MemoryStore, call_sid: str, *, transcription: bool = True) -> None:
    """An escalated call whose close makes every request: resolve (no id recorded), reservations,
    stop (transcription named), two missed workers (GET and POST each), GET and PUT."""
    esc: dict[str, Any] = {"reason": "fraud_suspected", "queue": "q", "journal_ref": "jr",
                           "task_sid": TASK}
    if transcription:
        esc["transcriptions"] = {call_sid: {"name": "humanleg", "role": "customer", "started_at": int(T0) - 60,
                                            "ended_at": None}}
    callstate.create(store, CallState(call_sid=call_sid, expires_at=T0 + 86_400, ai_sessions=1,
                                      escalation=esc))


def closer_over(store: MemoryStore, twilio: SlowTwilio, *, now: float = T0,
                journal: SegmentLog | None = None) -> HumanLegCloser:
    creds = lambda: CREDS  # noqa: E731
    return HumanLegCloser(
        store, OrchestratorHumanLeg(creds, opener=twilio, timeout_s=REQUEST_TIMEOUT_S),
        calls=TwilioCalls(creds, opener=twilio, timeout_s=REQUEST_TIMEOUT_S),
        reservations=taskrouter_reservations(WORKSPACE, creds, timeout_s=REQUEST_TIMEOUT_S, opener=twilio),
        clock=lambda: now, journal=journal if journal is not None else SegmentLog(store, now=lambda: now))


def closed_events(store: MemoryStore, call_sid: str) -> list[dict[str, Any]]:
    return [e for e in SegmentLog(store).read(call_sid) if e.get("type") == "conversation_closed"]
