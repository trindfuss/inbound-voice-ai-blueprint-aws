"""API Gateway REST proxy events in, Twilio-checked requests out.

* The public URL Twilio signed is rebuilt from the forwarded headers and the
  request context path (which keeps the stage or custom-domain base path), with
  :func:`owlbank.twilio_sig.reconstruct_url`.
* HTTP Basic is a challenge-response: Twilio sends the first request of a
  password-protected URL with *no* ``Authorization`` header and repeats it with
  ``Authorization: Basic`` only after a ``401`` with ``WWW-Authenticate``. A request without
  the header therefore gets that 401 challenge (empty body), before anything else is
  looked at, and is logged as ``twilio_basic_challenge`` (INFO, not an auth failure).
* ``X-Twilio-Signature`` and the Basic credentials are both checked before anything else; a
  failure is a 403 with an empty body, logged as a security event with no header
  values.
* The signature is accepted over the public URL without credentials, or over the same URL with
  ``user:password@`` inserted (which one Twilio signs is unverified); the form that
  matched is logged by name.
* Request parameters are never logged (they carry ``Digits`` and ``SpeechResult``).
* ``/voice/orchestrator`` (``agentcore`` hosting only) is checked by
  :func:`authenticate_callback` instead: the API WAF has already answered a request without
  ``Authorization`` with the Basic challenge (API Gateway REST renames a Lambda's
  ``WWW-Authenticate``, so the front door can't challenge here), so missing or wrong Basic
  credentials are a ``401`` and ``twilio_auth_failed`` reason ``basic``; then the signature, over a
  JSON body when the URL carries ``bodySHA256`` (Orchestrator's form), else a form body: ``403``
  reason ``signature``.
* Every REST proxy event shape is accepted: a missing or ``null`` body is an empty form,
  a base64 body is decoded, header names in any case (normalized once, ``headers`` over
  ``multiValueHeaders``), ``null`` header and query maps. A body that can't be read as a UTF-8
  form (bad base64, bytes that aren't UTF-8, a bad percent-escape, too many fields) is a generic
  400 with an empty ``<Response/>``, after the Basic credentials have been checked; it is logged
  with the error class only, never the body.
"""

from __future__ import annotations

import base64
import binascii
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

from owlbank import basic_auth, twilio_sig
from owlbank.basic_auth import BasicResult, BasicSecret, Match
from owlbank.obs import emit, log_caught
from owlbank.security_log import auth_failed, basic_challenge, signature_form

log = logging.getLogger(__name__)

ROUTES: frozenset[str] = frozenset(
    {"incoming", "language", "factor", "connect-action", "tool-flow", "status", "partial",
     "orchestrator", "taskrouter", "enqueue-action", "transcription"}
)
#: The human leg's real-time transcription status callback: signed and
#: Basic-protected like the voice webhooks, so :func:`authenticate` checks it; answered 204.
TRANSCRIPTION_ROUTE = "transcription"
#: The ``<Enqueue>`` ``action``: a voice webhook like the others.
ENQUEUE_ACTION_ROUTE = "enqueue-action"
#: TaskRouter's workspace event callback (signed and Basic-protected
#: like the voice webhooks, so :func:`authenticate` checks it; answered with :func:`event_ack`.
TASKROUTER_ROUTE = "taskrouter"
#: Orchestrator's status callbacks (``agentcore`` only): checked by
#: :func:`authenticate_callback`, never by :func:`authenticate` (they are not voice webhooks).
ORCHESTRATOR_ROUTE = "orchestrator"
ROUTE_PREFIX = "/voice/"
#: The challenge for a request without credentials (the API WAF answers the same, infra/waf.py).
BASIC_REALM = "owlbank"
BASIC_CHALLENGE = f'Basic realm="{BASIC_REALM}"'
#: Upper bound on form fields; a Twilio voice webhook sends well under a hundred.
MAX_FORM_FIELDS = 1000


@dataclass(frozen=True, slots=True)
class TwilioRequest:
    route: str
    params: Mapping[str, str]
    base_path: str  # path prefix before /voice/ (stage or base path), for action URLs
    #: The signed URL's query parameters (the Gather ids ``g`` and ``early``).
    query: Mapping[str, str] = field(default_factory=dict)
    #: ``https://host[:port]`` of the URL Twilio called, for URLs that can't be relative.
    origin: str = ""

    def url(self, route: str, **query: str) -> str:
        """Path-absolute URL of another route. Relative references keep the webhook
        URL's authority, including its Basic credentials (RFC 3986 section 5.2)."""
        tail = f"?{urlencode(query)}" if query else ""
        return f"{self.base_path}{ROUTE_PREFIX}{route}{tail}"

    def absolute_url(self, route: str, *, credentials: tuple[str, str] | None = None,
                     **query: str) -> str:
        """Absolute URL of another route, for TwiML sent through the REST API, which has
        no document URL to resolve a relative one against. ``credentials`` go in as
        ``user:password@`` the way the webhook URLs carry them, so Twilio sends Basic auth."""
        scheme, _, host = self.origin.partition("://")
        if not scheme or not host:
            raise ValueError("request origin unknown")
        userinfo = (f"{quote(credentials[0], safe='')}:{quote(credentials[1], safe='')}@"
                    if credentials else "")
        return f"{scheme}://{userinfo}{host}{self.url(route, **query)}"

    @property
    def call_sid(self) -> str:
        return self.params.get("CallSid", "")


class Rejected(Exception):
    """The request failed signature or Basic auth (respond 403)."""


class Challenge(Exception):
    """The request carried no ``Authorization`` header (respond 401 with ``WWW-Authenticate``)."""


class Malformed(Exception):
    """The body can't be read as a UTF-8 form (respond with :func:`bad_request`)."""


def _mapping(value: Any) -> Mapping[Any, Any]:
    return value if isinstance(value, Mapping) else {}


def _headers(event: Mapping[str, Any]) -> dict[str, str]:
    """Header names lowercased; ``headers`` wins over ``multiValueHeaders`` (first value)."""
    out: dict[str, str] = {}
    for name, values in _mapping(event.get("multiValueHeaders")).items():
        if isinstance(values, str):
            values = [values]
        if isinstance(values, (list, tuple)) and values and values[0] is not None:
            out[str(name).lower()] = str(values[0])
    for name, value in _mapping(event.get("headers")).items():
        if value is not None:
            out[str(name).lower()] = str(value)
    return out


def _body(event: Mapping[str, Any]) -> str:
    """The body as text: ``""`` when missing or ``null``; :class:`Malformed` if unreadable."""
    raw = event.get("body")
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise Malformed("body type")
    if event.get("isBase64Encoded"):
        try:
            return base64.b64decode(raw, validate=True).decode("utf-8")
        except (binascii.Error, ValueError) as exc:  # UnicodeDecodeError is a ValueError
            raise Malformed(type(exc).__name__) from None
    return raw


def parse_form(body: str) -> dict[str, str]:
    """An ``application/x-www-form-urlencoded`` body as a dict; :class:`Malformed` for a
    percent-escape that isn't UTF-8 or more than :data:`MAX_FORM_FIELDS` fields."""
    try:
        return dict(parse_qsl(body, keep_blank_values=True, errors="strict",
                              max_num_fields=MAX_FORM_FIELDS))
    except ValueError as exc:  # UnicodeDecodeError included
        raise Malformed(type(exc).__name__) from None


def _query(event: Mapping[str, Any]) -> str | None:
    """The query string. REST proxy events only carry it parsed; re-encode it in order."""
    multi = _mapping(event.get("multiValueQueryStringParameters"))
    if multi:
        return urlencode([(k, v) for k, vs in multi.items()
                          for v in ([vs] if isinstance(vs, str) else vs or []) if v is not None])
    single = _mapping(event.get("queryStringParameters"))
    if single:
        return urlencode([(k, v) for k, v in single.items() if v is not None])
    return None


def _full_path(event: Mapping[str, Any]) -> str:
    ctx = _mapping(event.get("requestContext"))
    if isinstance(ctx.get("path"), str) and ctx["path"]:
        return ctx["path"]  # includes the stage, or the custom domain's base path
    stage = ctx.get("stage")
    path = event.get("path") or "/"
    return f"/{stage}{path}" if stage and stage != "$default" else path


def route_of(path: str) -> tuple[str, str] | None:
    """``(base_path, route)`` for ``.../voice/<route>``, or ``None``."""
    idx = path.rfind(ROUTE_PREFIX)
    if idx < 0:
        return None
    route = path[idx + len(ROUTE_PREFIX):].strip("/")
    if route not in ROUTES:
        return None
    return path[:idx], route


def route_name(event: Mapping[str, Any]) -> str | None:
    """The route of an API Gateway event (``incoming``, ``factor``, ...), for log lines; never raises."""
    try:
        found = route_of(_full_path(event))
    except Exception:  # noqa: BLE001 - a malformed event only loses the op name
        return None  # (logged by authenticate with the rejection if the route is bad)
    return found[1] if found else None


def public_url(event: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> str:
    headers = _headers(event) if headers is None else headers
    ctx = _mapping(event.get("requestContext"))
    host = headers.get("host") or ctx.get("domainName") or ""
    return twilio_sig.reconstruct_url(
        "https", host, _full_path(event), _query(event),
        forwarded_proto=headers.get("x-forwarded-proto"),
        forwarded_port=headers.get("x-forwarded-port"),
    )


#: Reads the Basic secret; the argument is ``force`` (skip the cache, best effort, rate limited).
BasicSecretReader = Callable[[bool], BasicSecret]


def _match_secret(header: str | None, secret: BasicSecret, now: float) -> BasicResult:
    return basic_auth.match(header, secret.user, secret.password, secret.previous_password,
                            secret.previous_until, now=now)


def _basic_match(header: str | None, secret: BasicSecret, read_secret: BasicSecretReader,
                 now: float, route: str) -> tuple[BasicSecret, BasicResult]:
    """Match ``header`` against ``secret``; on a miss, one best-effort forced re-read and one more
    match (a rotation may be newer than this instance's cache). A failed forced read keeps the
    cached secret, so the request is rejected as a plain ``auth`` failure, not ``unverifiable``."""
    result = _match_secret(header, secret, now)
    if result.match is not Match.NONE:
        return secret, result
    try:
        fresh = read_secret(True)
    except Exception as exc:  # best effort: keep the cached secret, the request is refused as before
        log_caught(log, "basic secret refresh failed", exc, route=route, decision="use_cached")
        return secret, result
    return fresh, _match_secret(header, fresh, now)


def _log_previous_accepted(result: BasicResult, route: str, params: Mapping[str, str] | None) -> None:
    """Call-keyed event when the previous password authenticated a request (never the password)."""
    if result.match is Match.PREVIOUS:
        emit(log, logging.INFO, "twilio_basic_previous_accepted", route=route,
             call_sid=(params or {}).get("CallSid"))


def authenticate(event: Mapping[str, Any], *, auth_token: Callable[[], str],
                 basic_secret: BasicSecretReader,
                 clock: Callable[[], float] = time.time) -> TwilioRequest:
    """Check the request and parse it, or raise :class:`Challenge` (no ``Authorization`` header:
    the first request of every Twilio webhook), :class:`Malformed` (right Basic credentials but
    a body that isn't a readable form) or :class:`Rejected` (anything else wrong)."""
    headers = _headers(event)
    if not headers.get("authorization"):
        basic_challenge("frontdoor")
        raise Challenge("basic")
    path = _full_path(event)
    found = route_of(path)
    if found is None or (event.get("httpMethod") or "POST").upper() != "POST":
        log.warning("security: rejected request", extra={"reason": "route"})
        auth_failed("frontdoor", "route")
        raise Rejected("route")
    base_path, route = found
    if route == ORCHESTRATOR_ROUTE:  # not a voice webhook: authenticate_callback checks it
        log.warning("security: rejected request", extra={"reason": "route"})
        auth_failed("frontdoor", "route")
        raise Rejected("route")
    try:
        url = public_url(event, headers)
        token = auth_token()
        secret = basic_secret(False)
    except Exception as exc:  # malformed host, or a secret we can't read: fail closed
        log_caught(log, "security: rejected request", exc, reason="unverifiable", route=route,
                   decision="reject_403")
        auth_failed("frontdoor", "unverifiable")
        raise Rejected("unverifiable") from None
    secret, result = _basic_match(headers.get("authorization"), secret, basic_secret, clock(), route)
    if result.match is Match.NONE or result.password is None:
        log.warning("security: rejected request", extra={"reason": "basic_auth", "route": route})
        auth_failed("frontdoor", "basic")
        raise Rejected("auth")
    try:
        params = parse_form(_body(event))
    except Malformed as exc:  # the error class only: never the body
        log.warning("frontdoor: malformed request",
                    extra={"reason": "malformed_body", "error_class": str(exc), "route": route,
                           "decision": "reject_400"})
        raise
    # The credentials are tried in the signed URL only once the header has proved them, and with the
    # password that matched: Twilio signed the URL it holds, which may carry the previous one.
    form = twilio_sig.signed_url_form(url, params, headers.get("x-twilio-signature", ""), token,
                                      credentials=(secret.user, result.password))
    if form is None:
        log.warning("security: rejected request", extra={"reason": "signature", "route": route})
        auth_failed("frontdoor", "signature")
        raise Rejected("auth")
    signature_form("frontdoor", form)
    _log_previous_accepted(result, route, params)
    parts = urlsplit(url)
    return TwilioRequest(route=route, params=params, base_path=base_path,
                         query=dict(parse_qsl(parts.query, keep_blank_values=True)),
                         origin=f"{parts.scheme}://{parts.netloc}")


class Unauthorized(Exception):
    """Missing or wrong Basic credentials on a callback route (respond :func:`unauthorized`)."""


@dataclass(frozen=True, slots=True)
class CallbackRequest:
    """An authenticated Orchestrator callback. ``body`` is never logged."""

    body: str
    is_json: bool


def authenticate_callback(event: Mapping[str, Any], *, auth_token: Callable[[], str],
                          basic_secret: BasicSecretReader,
                          clock: Callable[[], float] = time.time) -> CallbackRequest:
    """Check an Orchestrator status callback (``POST .../voice/orchestrator``) or raise.

    Order: route and method (:class:`Rejected`, reason ``route``); Basic credentials, missing or
    wrong (:class:`Unauthorized`, reason ``basic``: the WAF has already challenged); a secret or host
    that can't be read (:class:`Rejected`, ``unverifiable``); ``X-Twilio-Signature`` over a JSON body
    (URL with ``bodySHA256``) or a form body, with the URL with ``user:password@`` also accepted
    (:class:`Rejected`, ``signature``; an unreadable body counts as a bad signature). Each failure
    logs ``twilio_auth_failed`` with its reason and nothing from the request."""
    headers = _headers(event)
    found = route_of(_full_path(event))
    if (found is None or found[1] != ORCHESTRATOR_ROUTE
            or (event.get("httpMethod") or "POST").upper() != "POST"):
        auth_failed("frontdoor", "route")
        raise Rejected("route")
    try:
        url = public_url(event, headers)
        token = auth_token()
        secret = basic_secret(False)
    except Exception as exc:  # malformed host, or a secret we can't read: fail closed
        log_caught(log, "security: rejected request", exc, reason="unverifiable",
                   route=ORCHESTRATOR_ROUTE, decision="reject_403")
        auth_failed("frontdoor", "unverifiable")
        raise Rejected("unverifiable") from None
    secret, result = _basic_match(headers.get("authorization"), secret, basic_secret, clock(),
                                  ORCHESTRATOR_ROUTE)
    if result.match is Match.NONE or result.password is None:
        auth_failed("frontdoor", "basic")
        raise Unauthorized("basic")
    is_json = any(k == "bodySHA256" for k, _ in parse_qsl(urlsplit(url).query, keep_blank_values=True))
    try:
        body = _body(event)
        params = None if is_json else parse_form(body)
    except Malformed:  # can't be verified; the error class isn't needed, nothing from the body
        auth_failed("frontdoor", "signature")
        raise Rejected("signature") from None
    form = twilio_sig.signed_url_form(url, params, headers.get("x-twilio-signature", ""), token,
                                      body=body if is_json else None,
                                      credentials=(secret.user, result.password))
    if form is None:
        auth_failed("frontdoor", "signature")
        raise Rejected("signature")
    signature_form("frontdoor", form)
    _log_previous_accepted(result, ORCHESTRATOR_ROUTE, params)
    return CallbackRequest(body=body, is_json=is_json)


def ok_empty() -> dict[str, Any]:
    """200, empty: the acknowledgement of an Orchestrator callback."""
    return {"statusCode": 200, "headers": {}, "body": ""}


def unauthorized() -> dict[str, Any]:
    """401, empty, no ``WWW-Authenticate`` (the WAF sends the challenge; API Gateway would rename it)."""
    return {"statusCode": 401, "headers": {}, "body": ""}


def twiml_response(xml: str) -> dict[str, Any]:
    return {"statusCode": 200, "headers": {"Content-Type": "text/xml; charset=utf-8"},
            "body": xml}


def no_content() -> dict[str, Any]:
    """204, empty: the answer to a ``partialResultCallback`` (Twilio ignores the body)."""
    return {"statusCode": 204, "headers": {}, "body": ""}


def event_ack() -> dict[str, Any]:
    """204 with ``Content-Type: application/json``: what TaskRouter asks of an event callback."""
    return {"statusCode": 204, "headers": {"Content-Type": "application/json"}, "body": ""}


#: The generic 400: valid, empty TwiML (nothing about the request, no error detail).
BAD_REQUEST_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response/>'


def bad_request() -> dict[str, Any]:
    """400 for a request whose body can't be read; the same body every time."""
    return {"statusCode": 400, "headers": {"Content-Type": "text/xml; charset=utf-8"},
            "body": BAD_REQUEST_TWIML}


def forbidden() -> dict[str, Any]:
    return {"statusCode": 403, "headers": {}, "body": ""}


def challenge() -> dict[str, Any]:
    """401 with the Basic challenge and an empty body (what makes Twilio repeat with credentials)."""
    return {"statusCode": 401, "headers": {"WWW-Authenticate": BASIC_CHALLENGE}, "body": ""}
