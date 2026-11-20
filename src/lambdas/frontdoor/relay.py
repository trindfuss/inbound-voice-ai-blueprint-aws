"""ConversationRelay WebSocket URLs, one per ``<Connect>``.

* :class:`StaticRelayUrl` (``fargate``): today's bridge ``wss://.../ws`` URL with the one-time
  nonce in its query (:func:`owlbank.ws_nonce.with_nonce`), byte for byte.
* :class:`RuntimeUrlSigner` (``agentcore``): a SigV4 query-signed ``wss://`` URL for
  ``InvokeAgentRuntimeWithWebSocketStream`` on our Runtime, signed locally with botocore's
  ``SigV4QueryAuth`` (no network call, no AgentCore SDK; the algorithm is the SDK's
  ``generate_presigned_url``). Session ID = the CallSid (``CALL_SID_RE`` guarantees the
  33-character minimum). The URL never carries the nonce: AgentCore forwards no query parameter
  to the container, so the front door sends it as ``<Parameter name="n">`` instead.

Signing credentials come from ``sts:AssumeRole`` on the presign role, cached and refreshed
when less than the URL's validity plus 60 s is left, so a URL never outlives them, and by
:meth:`RuntimeUrlSigner.refresh_if_due` at INIT and when less than 15 minutes is left, so an
``AssumeRole`` rarely falls on a call's webhook. A credential provider can be injected instead of
STS (``scripts/bench/handshake_timer.py`` signs with the operator's own session).

Nothing here logs a URL, a nonce or a credential: ``mint`` never logs its result, and the
DEBUG lines botocore's signer writes (canonical request, string to sign, signature) are dropped
while this module signs, whatever the log level. A signing failure raises :class:`PresignError`
carrying the error class only.
"""

from __future__ import annotations

import contextvars
import logging
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from urllib.parse import quote, urlencode

from botocore.auth import SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.credentials import ReadOnlyCredentials

from owlbank import ws_nonce
from owlbank.obs import log_caught
from owlbank.relay_param import RUNTIME_ARN_RE  # one pattern for the parameter and the signer
from owlbank.sids import CALL_SID_RE  # one pattern for the signer and the warmer

log = logging.getLogger(__name__)

#: A Twilio CallSid, which is the Runtime session ID: 34 characters, above AgentCore's
#: documented minimum of 33 (and far below its maximum of 256).

SERVICE = "bedrock-agentcore"
#: The AgentCore maximum for a pre-signed WebSocket URL's validity.
MAX_TTL_S = 300
#: Credentials must outlive a URL by at least this much.
CREDENTIAL_MARGIN_S = 60
#: ``refresh_if_due`` default: refresh when less than this is left (INIT and opportunistic).
REFRESH_MARGIN_S = 900
SESSION_QUERY_PARAM = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"
ROLE_SESSION_NAME = "owlbank-relay-presign"

_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z]+)+-\d{1,2}\Z")


class RelayUrls(Protocol):
    def mint(self, *, call_sid: str, nonce: str) -> str: ...  # never logs its result


class PresignError(Exception):
    """Signing credentials couldn't be obtained. Carries the error class only."""

    def __init__(self, error_class: str) -> None:
        super().__init__(error_class)
        self.error_class = error_class


@dataclass(frozen=True)
class SigningCredentials:
    """Credentials for signing, with their expiry as epoch seconds."""

    access_key: str
    secret_key: str
    token: str | None
    expires_at: float

    def __repr__(self) -> str:  # never show the key or token in a traceback or a debugger
        return f"SigningCredentials(expires_at={self.expires_at!r})"


@dataclass(frozen=True)
class StaticRelayUrl:
    """``fargate``: today's relay URL with the nonce in its query."""

    relay_ws_url: str

    def mint(self, *, call_sid: str, nonce: str) -> str:
        return ws_nonce.with_nonce(self.relay_ws_url, nonce)


# botocore.auth logs the canonical request (with the token and nonce), the string to sign and the
# signature at DEBUG. ``owlbank.obs`` pins botocore to WARNING; this drops them regardless.
_signing: contextvars.ContextVar[bool] = contextvars.ContextVar("owlbank_relay_signing", default=False)


class _DropWhileSigning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _signing.get()


_BOTOCORE_AUTH_FILTER = _DropWhileSigning()
logging.getLogger("botocore.auth").addFilter(_BOTOCORE_AUTH_FILTER)


@contextmanager
def _quiet_signer() -> Iterator[None]:
    token = _signing.set(True)
    try:
        yield
    finally:
        _signing.reset(token)


def _expiry_epoch(value: Any) -> float:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise TypeError("naive expiration")
        return value.timestamp()
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise TypeError("expiration")


def sts_credential_provider(sts: Any, role_arn: str) -> Callable[[], SigningCredentials]:
    """``AssumeRole`` on ``role_arn`` with ``sts`` (any boto3-like STS client)."""

    def provide() -> SigningCredentials:
        creds = sts.assume_role(RoleArn=role_arn, RoleSessionName=ROLE_SESSION_NAME)["Credentials"]
        return SigningCredentials(access_key=str(creds["AccessKeyId"]),
                                  secret_key=str(creds["SecretAccessKey"]),
                                  token=str(creds["SessionToken"]),
                                  expires_at=_expiry_epoch(creds["Expiration"]))

    return provide


class RuntimeUrlSigner:
    """``agentcore``: a fresh pre-signed ``wss://`` URL for each ``<Connect>``."""

    def __init__(self, *, runtime_arn: str, region: str, clock: Callable[[], float],
                 role_arn: str | None = None, sts: Any = None,
                 credential_provider: Callable[[], SigningCredentials] | None = None,
                 ttl_s: int = MAX_TTL_S) -> None:
        if not _REGION_RE.match(region):
            raise ValueError("region is not an AWS region name")
        arn = RUNTIME_ARN_RE.match(runtime_arn)
        if arn is None or arn.group("region") != region:
            raise ValueError("runtime_arn is not an AgentCore Runtime ARN in this region")
        if isinstance(ttl_s, bool) or not isinstance(ttl_s, int) or not 0 < ttl_s <= MAX_TTL_S:
            raise ValueError(f"ttl_s must be an integer in 1..{MAX_TTL_S}")
        if credential_provider is None:
            if sts is None or not role_arn:
                raise ValueError("give either sts and role_arn, or a credential_provider")
            credential_provider = sts_credential_provider(sts, role_arn)
        elif sts is not None or role_arn:
            raise ValueError("give either sts and role_arn, or a credential_provider, not both")
        self._region = region
        self._host = f"{SERVICE}.{region}.amazonaws.com"
        self._path = f"/runtimes/{quote(runtime_arn, safe='')}/ws"
        self._clock = clock
        self._provide = credential_provider
        self.ttl_s = ttl_s
        self._creds: SigningCredentials | None = None
        self._lock = threading.Lock()

    @property
    def host(self) -> str:
        return self._host

    @property
    def path(self) -> str:
        return self._path

    def _remaining(self) -> float:
        return -1.0 if self._creds is None else self._creds.expires_at - self._clock()

    def _refresh(self) -> None:
        try:
            creds = self._provide()
        except Exception as exc:  # STS errors, network, malformed responses: class only
            raise PresignError(type(exc).__name__) from None
        if not isinstance(creds, SigningCredentials) or not creds.access_key or not creds.secret_key:
            raise PresignError("InvalidCredentials")
        self._creds = creds

    def _credentials(self) -> SigningCredentials:
        needed = self.ttl_s + CREDENTIAL_MARGIN_S
        with self._lock:
            if self._remaining() < needed:
                self._refresh()
            creds = self._creds
            if creds is None or self._remaining() < needed:
                raise PresignError("CredentialsExpireTooSoon")
            return creds

    def refresh_if_due(self, *, margin_s: int = REFRESH_MARGIN_S) -> None:
        """Refresh when less than ``margin_s`` (and at least TTL + 60 s) is left. Never raises."""
        with self._lock:
            if self._remaining() >= max(margin_s, self.ttl_s + CREDENTIAL_MARGIN_S):
                return
            try:
                self._refresh()
            except PresignError as exc:
                log_caught(log, "relay_presign_refresh_failed", exc, decision="retry_on_mint")

    def mint(self, *, call_sid: str, nonce: str) -> str:
        """The pre-signed URL for ``call_sid``'s session. Raises :class:`PresignError` when
        credentials can't be obtained. Never logs its result.

        ``nonce`` is ignored (it is part of :class:`RelayUrls` for :class:`StaticRelayUrl`): the
        URL never carries it, the front door sends it as ``<Parameter name="n">``."""
        del nonce
        if not isinstance(call_sid, str) or not CALL_SID_RE.match(call_sid):
            raise ValueError("call_sid is not a CallSid")
        query = {SESSION_QUERY_PARAM: call_sid}
        creds = self._credentials()
        url = f"https://{self._host}{self._path}?{urlencode(query, quote_via=quote, safe='')}"
        request = AWSRequest(method="GET", url=url, headers={"host": self._host})
        frozen = ReadOnlyCredentials(creds.access_key, creds.secret_key, creds.token)
        with _quiet_signer():
            SigV4QueryAuth(frozen, SERVICE, self._region, expires=self.ttl_s).add_auth(request)
        signed: str = request.url
        return "wss://" + signed.removeprefix("https://")
