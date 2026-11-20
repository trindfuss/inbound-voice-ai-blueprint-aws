"""Security log lines and log redaction shared by the front door and the bridge.

**Auth-failure line.** Every Twilio-auth failure (signature, Basic auth, relay
nonce, ...) is logged by :func:`auth_failed` on the ``owlbank.security`` logger at
WARNING. Through :mod:`owlbank.obs` it is one JSON line whose fields include::

    "event":"twilio_auth_failed","component":"<frontdoor|bridge>","reason":"<reason>"

(plus ``ts``, ``level`` and the correlation IDs). CloudWatch metric filters match the quoted
term ``twilio_auth_failed`` (a literal term, which also appears as the ``message`` under Lambda
JSON logging), and may split by ``component``/``reason`` with a JSON filter on ``$.event``.
The line never carries request data: no URL, header, parameter or nonce. (``call_sid`` is
there when the invocation or session has one: it is the key for tracing the call.)
:func:`auth_failed_line` still renders the same fields as the compact JSON text.

**Basic challenge line.** Twilio sends the first request of every password-protected
webhook without credentials and repeats it with ``Authorization: Basic`` only after a ``401``
challenge, so a request without ``Authorization`` is the expected first half of every webhook,
not an attack. The front door answers it with the challenge and logs, at INFO::

    {"event":"twilio_basic_challenge","component":"frontdoor","reason":"basic","challenge":true}

The event name must not contain the term ``twilio_auth_failed`` (the CloudWatch alarm's literal
filter), so the challenge never counts as an auth failure. A request *with* wrong credentials is
still ``twilio_auth_failed`` reason ``basic``.

**Untrusted-ANI code line.** When the auth service sends a one-time code on a
call whose caller ID is untrusted (the ``ani_trust`` decision), :func:`otp_sent_untrusted_ani`
logs, once per send, at WARNING::

    {"event":"otp_sent_untrusted_ani","component":"auth_service"}

A CloudWatch metric filter and alarm match the event name. It indicates a possibly spoofed
caller ID aimed at a real customer: a warning signal, not a block. No number, code or
customer ID is on the line (the audit ``otp_sent`` line carries the call and ``ani_trusted``).

**Keyed-in number code line.** A code sent on a call identified
by a keyed-in phone number (always untrusted, and sent only to a number on file after ZIP and
SSN-4) logs :func:`otp_sent_keyed_phone` instead, at WARNING so it is kept at every log level::

    {"event":"otp_sent_keyed_phone","component":"auth_service"}

It is counted (``OtpSentKeyedPhone``) but never alarmed on: it is the expected path for those
calls, and the untrusted-ANI alarm stays a signal of a spoofed caller ID.

**Query-string redaction.** The relay URL carries the one-time
nonce (``/ws?n=...``); on ``agentcore`` the pre-signed URL carries ``X-Amz-Signature``,
``X-Amz-Credential`` and ``X-Amz-Security-Token``. :class:`QueryStringFilter` rewrites any URL or
path query in a record's message, its arguments (also a query the template and an argument make
together) and its ``extra`` fields (nested ones included) to ``?[redacted]``, and a bare query (two
or more ``name=value`` joined by ``&``) to ``[redacted]``. :func:`install_log_redaction` attaches it
to the ``tac``, ``uvicorn``, ``starlette`` and ``fastapi`` loggers (their existing children, and
their own handlers, such as TAC's, which doesn't propagate) and installs a log-record factory that
applies the same rewrite to the message and arguments of every record created afterwards, wherever
it is logged, so a third-party line can't leak the nonce or the URL's credentials.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from owlbank.obs import emit

SECURITY_LOGGER = "owlbank.security"
AUTH_FAILED_EVENT = "twilio_auth_failed"
COMPONENTS = frozenset({"frontdoor", "bridge"})
REASONS = frozenset({"signature", "basic", "nonce", "binding", "route", "unverifiable"})
OTP_UNTRUSTED_EVENT = "otp_sent_untrusted_ani"
OTP_KEYED_PHONE_EVENT = "otp_sent_keyed_phone"
BASIC_CHALLENGE_EVENT = "twilio_basic_challenge"
SIGNATURE_FORM_EVENT = "twilio_signature_form"

_log = logging.getLogger(SECURITY_LOGGER)

# A query after a URL or an absolute path: "wss://h/ws?n=abc", "/ws?n=abc", "GET /x?y=1".
_QUERY = re.compile(r"((?:[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s?#'\"<>]*|(?<![\w/])/[^\s?#'\"<>]*)\?)[^\s'\"<>]*")
_REDACTED = r"\1[redacted]"
#: A query without its URL (two or more ``name=value`` joined by ``&``: a logged ``url.query``).
_BARE_QUERY = re.compile(r"(?<![^\s'\"<>(\[{,;:])[\w.~%+\-]+=[^\s&'\"<>]*(?:&[\w.~%+\-]+=[^\s&'\"<>]*)+")


def auth_failed_line(component: str, reason: str) -> str:
    if component not in COMPONENTS or reason not in REASONS:
        raise ValueError("unknown component or reason")
    return json.dumps({"event": AUTH_FAILED_EVENT, "component": component, "reason": reason},
                      separators=(",", ":"))


def auth_failed(component: str, reason: str) -> None:
    """Log the stable auth-failure line (see the module docstring)."""
    if component not in COMPONENTS or reason not in REASONS:
        raise ValueError("unknown component or reason")
    emit(_log, logging.WARNING, AUTH_FAILED_EVENT, component=component, reason=reason)


def basic_challenge(component: str) -> None:
    """Log the expected first request of a Twilio webhook (answered 401); INFO, never an alarm."""
    if component not in COMPONENTS:
        raise ValueError("unknown component")
    emit(_log, logging.INFO, BASIC_CHALLENGE_EVENT, component=component, reason="basic", challenge=True)


def signature_form(component: str, form: str) -> None:
    """Log which signed-URL form validated: the form name only, never a URL."""
    if component not in COMPONENTS or form not in ("plain", "credentials"):
        raise ValueError("unknown component or form")
    emit(_log, logging.INFO, SIGNATURE_FORM_EVENT, component=component, form=form)


def otp_sent_untrusted_ani_line() -> str:
    return json.dumps({"event": OTP_UNTRUSTED_EVENT, "component": "auth_service"},
                      separators=(",", ":"))


def otp_sent_untrusted_ani() -> None:
    """Log the stable untrusted-ANI code line (see the module docstring)."""
    emit(_log, logging.WARNING, OTP_UNTRUSTED_EVENT, component="auth_service")


def otp_sent_keyed_phone() -> None:
    """Log the stable keyed-in number code line (see the module docstring); counted, no alarm."""
    emit(_log, logging.WARNING, OTP_KEYED_PHONE_EVENT, component="auth_service")


def strip_queries(text: str) -> str:
    """Every URL or path query (``?[redacted]``) and every bare query (``[redacted]``)."""
    return _BARE_QUERY.sub("[redacted]", _QUERY.sub(_REDACTED, text))


def _has_query(text: str) -> bool:
    return bool(_QUERY.search(text) or _BARE_QUERY.search(text))


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return strip_queries(value)
    if isinstance(value, (bytes, bytearray)):
        return strip_queries(bytes(value).decode("utf-8", "replace"))
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    try:  # URL objects (starlette URL, ...): replaced by redacted text only if they show a query
        text = str(value)
    except Exception:
        return value
    return strip_queries(text) if _has_query(text) else value


#: A plain record's attributes: anything else on a record is an ``extra`` field.
_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime", "taskName"}


def _redact_field(value: Any, depth: int = 0) -> Any:
    """``value`` with every query in its strings stripped; containers are copied only if changed."""
    if isinstance(value, str):
        return strip_queries(value)
    if depth >= 6 or value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, dict):
        out = {k: _redact_field(v, depth + 1) for k, v in value.items()}
        return out if any(out[k] is not value[k] for k in value) else value
    if isinstance(value, (list, tuple)):
        items = [_redact_field(v, depth + 1) for v in value]
        return type(value)(items) if any(a is not b for a, b in zip(items, value, strict=True)) else value
    return _redact_value(value)


def _redact_extras(record: logging.LogRecord) -> None:
    """``extra`` fields too: a library handler (TAC's own, which doesn't propagate) writes them
    without our scrubber. Only a field that shows a query is replaced."""
    for key, value in list(record.__dict__.items()):
        if key in _RECORD_ATTRS or key.startswith("_"):
            continue
        redacted = _redact_field(value)
        if redacted is not value:
            record.__dict__[key] = redacted


def redact_record(record: logging.LogRecord) -> logging.LogRecord:
    """Strip queries from the message, its arguments and the ``extra`` fields. The record
    factory sees a record before its ``extra`` fields are set, so the logger filter (which runs
    after) does the fields again even on a record the factory already redacted."""
    _redact_extras(record)
    if getattr(record, "_owlbank_redacted", False):
        return record
    if isinstance(record.msg, str):
        record.msg = strip_queries(record.msg)
    elif record.msg is not None:
        record.msg = _redact_value(record.msg)
    if isinstance(record.args, tuple):
        record.args = tuple(_redact_value(a) for a in record.args)
    elif isinstance(record.args, dict):
        record.args = {k: _redact_value(v) for k, v in record.args.items()}
    if record.args:  # a query made by the template and an argument together ("/%s" and "ws?n=...")
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad template: the handler reports it as before
            message = None
        if message is not None and _has_query(message):
            record.msg, record.args = strip_queries(message), None
    record._owlbank_redacted = True
    return record


class QueryStringFilter(logging.Filter):
    """Rewrites URL and path query strings in a record's message and arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        redact_record(record)
        return True


_FILTER = QueryStringFilter()
_installed = False
REDACTED_LOGGER_PREFIXES = ("tac", "uvicorn", "starlette", "fastapi")


def install_log_redaction() -> None:
    """Idempotent: filter on the TAC/uvicorn loggers, plus a redacting record factory."""
    global _installed
    names = [n for n in logging.root.manager.loggerDict
             if n.split(".")[0] in REDACTED_LOGGER_PREFIXES]
    for name in {*REDACTED_LOGGER_PREFIXES, "uvicorn.error", "uvicorn.access", *names}:
        logger = logging.getLogger(name)
        if _FILTER not in logger.filters:
            logger.addFilter(_FILTER)
        for handler in logger.handlers:  # e.g. TAC's own handler: it sees its children's records too
            if _FILTER not in handler.filters:
                handler.addFilter(_FILTER)
    if _installed:
        return
    _installed = True
    previous = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        return redact_record(previous(*args, **kwargs))

    logging.setLogRecordFactory(factory)
