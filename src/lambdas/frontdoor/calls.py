"""Twilio REST from the front door: replace a live call's TwiML.

One call per early finish: ``Calls(CallSid).update(twiml=...)`` ends the running
``<Gather>`` at once. The client is built from the Twilio API key secret (the one
``auth_service`` uses; JSON ``account_sid``, ``api_key``, ``api_secret``). Where Lambda starts
the environment ahead of calls (provisioned concurrency or SnapStart, :func:`build_at_init`),
it is built during INIT, so the first early finish doesn't pay the client's start-up
(about 1 s on test call CA…74b01a); if that fails it is logged and built on first use. INIT then
also makes one cheap authenticated read (a one-row Calls list, :meth:`TwilioCallUpdater.warm`)
to open the client's pooled HTTPS connection, so the first update doesn't pay the TLS
set-up either (the earlier account-resource read, which served the same purpose, measured
1.11 s first and 0.05 s after on call CA…0e8149; those figures predate the Calls-list read and
are not re-measured); a failure is logged and ignored. Otherwise
it is built on first use, so a request that never finishes early reads no secret. Short
timeouts: a slow update must not hold the ``partialResultCallback`` request, and a failed one
only means the Gather ends on its own.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from owlbank import coldstart
from owlbank.obs import emit, log_caught, timed_call

log = logging.getLogger(__name__)

#: Lambda initialization types that run INIT ahead of any request (``AWS_LAMBDA_INITIALIZATION_TYPE``).
AHEAD_OF_TIME_INIT = frozenset({coldstart.PROVISIONED, "snap-start"})


def build_at_init(env: Mapping[str, str] | None = None) -> bool:
    """True when the Twilio client should be built during INIT."""
    return coldstart.init_type(env) in AHEAD_OF_TIME_INIT

#: Seconds for the REST call (connect and read together, Twilio's HTTP client ``timeout``).
UPDATE_TIMEOUT_S = 3.0
#: Seconds for the INIT connection warm-up read: short, it only opens the connection.
WARM_TIMEOUT_S = 1.5
#: Seconds a warm-up or update keeps the connection counted as open: idle
#: keep-alive connections are closed by the far end, so an older one is warmed again.
WARM_FRESH_S = 30.0
#: The warm-up read: a one-row Calls list under the account (2010 API, the same host as
#: ``calls.update``); a Standard key can read it, unlike the account resource.
WARM_URL = "https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls.json?PageSize=1"


class CallUpdater(Protocol):
    def update_twiml(self, call_sid: str, twiml: str) -> None:
        """Replace the call's running TwiML; raise on any failure."""
        ...

    def ensure_warm(self) -> None:
        """Open the connection the update will use, if not already open."""
        ...


class TwilioCallUpdater:
    """:class:`CallUpdater` over ``twilio.rest.Client``, created lazily and reused."""

    def __init__(self, client_factory: Callable[[], Any],
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self._factory = client_factory
        self._client: Any = None
        self._lock = threading.Lock()
        self._monotonic = monotonic
        #: When a warm-up or update last succeeded (``None``: never in this environment).
        self._used_at: float | None = None

    def _get(self) -> Any:
        with self._lock:
            if self._client is None:
                self._client = self._factory()
            return self._client

    def prebuild(self) -> bool:
        """Build the client now (Lambda INIT). On failure, log the error class (never
        the message, which could name the secret) and leave it to be built on first use."""
        try:
            self._get()
        except Exception as exc:  # noqa: BLE001 - the first early finish retries the build
            log_caught(log, "twilio_client_init_failed", exc, decision="build_on_first_use")
            return False
        self.warm()
        return True

    def warm(self) -> bool:
        """One authenticated GET of a one-row Calls list, with a short timeout, to open
        the pooled HTTPS connection at INIT. Any failure (an exception or a non-2xx answer) is
        logged as ``twilio_connection_warm_failed`` (error class and status only) and ignored: the
        first update just opens the connection itself."""
        started = time.monotonic()
        try:
            client = self._get()
            resp = client.request("GET", WARM_URL.format(account_sid=client.account_sid),
                                  timeout=WARM_TIMEOUT_S)
            status = int(getattr(resp, "status_code", 0) or 0)
        except Exception as exc:  # noqa: BLE001 - a warm-up must never fail INIT
            log_caught(log, "twilio_connection_warm_failed", exc, decision="ignored")
            return False
        if not 200 <= status < 300:
            emit(log, logging.WARNING, "twilio_connection_warm_failed", error_class="HttpStatus",
                 http_status=status, decision="ignored")
            return False
        emit(log, logging.INFO, "twilio_connection_warmed",
             ms=round((time.monotonic() - started) * 1000, 1))
        self._used_at = self._monotonic()
        return True

    def ensure_warm(self) -> None:
        """:meth:`warm` unless a warm-up or an update opened the connection in
        the last :data:`WARM_FRESH_S` (the phone slot's settle wait calls it; an on-demand
        environment never warmed at INIT, and an idle connection may have been closed)."""
        if self._used_at is None or self._monotonic() - self._used_at > WARM_FRESH_S:
            self.warm()

    def update_twiml(self, call_sid: str, twiml: str) -> None:
        with timed_call("twilio", "calls.update"):
            self._get().calls(call_sid).update(twiml=twiml)
        self._used_at = self._monotonic()


def twilio_client_factory(secret_arn: str) -> Callable[[], Any]:
    """Build ``twilio.rest.Client`` from the API key secret, with a short HTTP timeout."""

    def build() -> Any:
        from twilio.http.http_client import TwilioHttpClient
        from twilio.rest import Client

        from owlbank.secrets import get_secret

        s = get_secret(secret_arn, logical="twilio_api_key")
        return Client(s["api_key"], s["api_secret"], s["account_sid"],
                      http_client=TwilioHttpClient(timeout=UPDATE_TIMEOUT_S))

    return build


class FakeCallUpdater:
    """Test double: records updates; ``fail`` makes every update raise."""

    def __init__(self, fail: bool = False) -> None:
        self.updates: list[tuple[str, str]] = []
        self.fail = fail
        self.warms = 0

    def ensure_warm(self) -> None:
        self.warms += 1

    def update_twiml(self, call_sid: str, twiml: str) -> None:
        if self.fail:
            raise RuntimeError("twilio unavailable")
        self.updates.append((call_sid, twiml))
