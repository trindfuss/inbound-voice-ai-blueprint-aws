"""Diagnosable logging for every OwlBank component.

One module, stdlib only, shared by the Lambdas, the TAC bridge and the scripts.

**What a log line looks like.** One JSON object per line with ``ts``, ``level``,
``component``, ``event`` (the event name: the log message, or the ``event`` key) and, when
known, ``call_sid``, ``request_id`` and ``trace_id``, then any ``extra`` fields. Code logs an
event with :func:`emit` (``emit(log, logging.INFO, "turn", turn=3, duration_ms=812.4)``) or with
plain ``log.info("message", extra={...})``; both end up in the same shape.

**Correlation.** ``call_sid``, ``request_id`` and ``trace_id`` live in :mod:`contextvars`;
:class:`ScrubFilter` (or :class:`ContextFilter` on its own, for a handler that doesn't scrub)
copies them onto every record at emit time, so any record written while they are bound carries
them, whichever logger made it. (A record factory would be simpler, but it makes
``extra={"call_sid": ...}`` raise ``KeyError``.) :func:`bind_lambda` binds them for one Lambda
invocation (``context.aws_request_id``, the ``_X_AMZN_TRACE_ID`` environment variable, and a
``call_sid`` found in the event). The bridge binds the call SID for each WebSocket session.
Searching one ``call_sid`` across all ``/owlbank/*`` log groups gives the whole call.

**Caught errors.** Code that catches an error and carries on logs it with :func:`log_caught`: error
class, error code, the ``decision`` taken, identifiers and ``stack``, which is :func:`safe_traceback`
(frames and class names; never the exception message, arguments or local variables, because the
message is where caller input shows up). ``log.exception`` and ``exc_info=`` are not used (a test
enforces it). The one place a message is logged is an unhandled error's ``invocation_error``
(``error_message``, scrubbed). Expected, high-volume outcomes are logged with :func:`emit`, no stack.

**Scrubber** (:func:`scrub`; a :class:`ScrubFilter` on the handler and again as a step in the
formatter). Pattern classes only, no phrase list: E.164 numbers (``+*******1234``), runs of six
or more digits, token-like strings (a run of 20 or more base64url characters that is not an
identifier such as ``AUTH_LEVEL_INSUFFICIENT`` or ``ProvisionedThroughputExceeded``),
Authorization and Basic/Bearer header values, and URL query strings. Twilio SIDs (two capitals
and 32 hex digits) and UUIDs are kept: they are identifiers, and they are what you search on.
It is deterministic and idempotent (scrubbing twice equals once). **Limit:** it is
pattern-based and cannot recognise free-form personal data (a name, an address, a spoken
sentence). That is why no message text is logged at all.

**Lambda.** With the function's log format set to JSON (``LoggingFormat.JSON`` in the CDK
stack), the Python runtime installs its own handler on the root logger. ``configure_logging``
keeps that handler (it never adds a second one, so no line is written twice) and relies on two
documented behaviours of it (docs.aws.amazon.com/lambda/latest/dg/python-logging.html): keys
passed through ``extra=`` appear as top-level keys of the JSON line next to the runtime's own
``timestamp``, ``level``, ``message``, ``logger`` and ``requestId``; and the runtime formats
exceptions itself (``errorType``, ``errorMessage``, ``stackTrace``), which is why
:class:`ScrubFilter` turns ``exc_info`` into scrubbed ``error_class``, ``error_message`` and
``stack_trace`` fields before the runtime sees the record. With the text format, the runtime
handler keeps its place and gets our JSON formatter. The Lambda log level is applied by the
runtime from ``application_log_level_v2``; ``OWLBANK_LOG_LEVEL`` carries the same value.

**Span timings.** :class:`Timings` breaks one request's time into named spans
(``timings_ms={"state_load": 12, "resolve": 3}``) for a single summary field on an existing
line, such as the front door's ``factor_answer``. Names and whole
milliseconds only. :class:`timed_call` stays the per-dependency-call line.

**Level.** ``OWLBANK_LOG_LEVEL`` (default INFO; an invalid value falls back to INFO and warns
once). ``botocore``, ``boto3``, ``urllib3``, ``s3transfer`` and ``hpack`` (and a few HTTP
client loggers) are pinned to WARNING at every level: their DEBUG output includes request and
response bodies, secret values among them.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Literal, TextIO

__all__ = [
    "LOG_LEVEL_ENV", "SLOW_CALL_MS", "PINNED_LOGGERS", "scrub", "scrub_value", "ScrubFilter",
    "JsonFormatter", "ContextFilter", "configure_logging", "reset_logging", "emit", "bind", "unbind", "bind_lambda",
    "bind_call_sid", "unbind_all", "call_context", "current_call_sid", "current_request_id", "extract_call_sid",
    "instrumented", "timed_call", "note", "degraded", "classify_response", "error_code_of",
    "resolve_level", "safe_traceback", "log_caught", "set_event_fields", "Timings",
]

LOG_LEVEL_ENV = "OWLBANK_LOG_LEVEL"
#: A dependency call slower than this logs at WARNING.
SLOW_CALL_MS = 500
#: Libraries whose DEBUG output carries request and response bodies. Pinned to WARNING.
PINNED_LOGGERS = ("botocore", "boto3", "urllib3", "s3transfer", "hpack",
                  "twilio", "httpx", "httpcore", "websockets",
                  # Strands' DEBUG lines print tool inputs (an escalation summary); slots.
                  "strands")

_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARN": logging.WARNING,
           "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}

_log = logging.getLogger("owlbank.obs")

# --------------------------------------------------------------------------- scrubber

_SID = r"\b[A-Z]{2}[0-9a-f]{32}\b"
_UUID = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
_AUTH = (r"(?P<ahead>(?i:\b(?:proxy-)?authorization)[\"']?\s*[:=]\s*[\"']?)"
         r"(?:(?i:basic|bearer|digest)\s+)?[^\s\"',;}]+")
_SCHEME = r"(?P<shead>\b(?:Basic|Bearer))\s+(?=[^\s]*[A-Z0-9=+/])[A-Za-z0-9+/=._~-]{6,}"
_URL = (r"(?P<uhead>(?:[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s?#'\"<>]*|(?<![\w/])/[^\s?#'\"<>]*)\?)"
        r"[^\s'\"<>]*")
_QS = r"\?[\w.~%-]+=[^\s'\"<>]*"
# Bounded runs (RFC 5321 lengths), so a long run with no "@" costs linear time, not quadratic.
# Whole runs only, at both ends: a match ending inside a word would leave a new word start
# behind it, and a second scrub would then mask that (not idempotent).
_EMAIL = r"(?<![\w.%+-])[\w.%+-]{1,64}@(?:[A-Za-z0-9-]{1,63}\.){1,8}[A-Za-z]{2,63}(?![A-Za-z0-9-])"
_NANP = r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"
_PHONE = r"\+\d{7,}"
_TOKEN = r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{20,}"  # noqa: S105 (a scrubbing regex, not a secret)
_DIGITS = r"\d{6,}"

_PATTERN = re.compile("|".join(f"(?P<{n}>{p})" for n, p in (
    ("sid", _SID), ("uuid", _UUID), ("auth", _AUTH), ("scheme", _SCHEME), ("url", _URL),
    ("qs", _QS), ("email", _EMAIL), ("nanp", _NANP), ("phone", _PHONE), ("tok", _TOKEN), ("digits", _DIGITS))))

#: A run that is a code name rather than a secret: snake or kebab case, CamelCase, CONSTANT_CASE.
#: At least two words, so a degenerate run (``AAAA...``, ``aaaa...``) is not mistaken for one.
_IDENTIFIER = re.compile(
    r"[a-z]+(?:[_-][a-z]+)+|[A-Z]+(?:[_-][A-Z]+)+|(?:[A-Z][a-z]+){2,}|[A-Z][a-z]+(?:[_-][A-Z]?[a-z]+)+")


def _mask_phone(text: str) -> str:
    return "+" + "*" * (len(text) - 5) + text[-4:]


def _replace(m: re.Match[str]) -> str:
    kind = m.lastgroup
    if kind in ("sid", "uuid"):
        return m.group(0)
    if kind == "auth":
        return m.group("ahead") + "[redacted]"
    if kind == "scheme":
        return m.group("shead") + " [redacted]"
    if kind == "url":
        return m.group("uhead") + "[redacted]"
    if kind == "qs":
        return "?[redacted]"
    if kind == "email":
        return "[email]"
    if kind == "nanp":
        return "[phone]"
    if kind == "phone":
        return _mask_phone(m.group(0))
    if kind == "tok":
        run = m.group(0)
        if _IDENTIFIER.fullmatch(run):
            return run
        return "[token]"
    return "[num]"


#: Upper bound on repeated scrub passes; each pass only replaces text with shorter markers, so a
#: handful always reaches a fixed point.
_SCRUB_PASSES = 4


def scrub(text: str) -> str:
    """Mask email addresses, phone numbers, digit runs, token-like strings, auth header values
    and URL queries.

    Deterministic and idempotent: ``scrub(scrub(x)) == scrub(x)``.
    """
    if len(text) < 4:
        return text
    # A mask can leave text that is itself maskable (``[email]`` followed by a long run of token
    # characters). Repeat until nothing changes, which is what makes the result idempotent.
    for _ in range(_SCRUB_PASSES):
        masked = _PATTERN.sub(_replace, text)
        if masked == text:
            break
        text = masked
    return text


# Model and inference-profile IDs are not customer data (``us.anthropic.claude-haiku-4-5-20251001-v1:0``
# would otherwise lose its date to the token pattern, and the preferred-model metric filter compares the
# logged ID with the configured one). The fields that carry them are exempt, but only for a value of
# that shape; anything else in those fields is scrubbed as usual.
MODEL_ID_FIELDS = frozenset({"model_id", "from_model", "to_model", "models", "preferred", "global_entries"})
# Letter-led provider and model segments, so a dotted number (a phone number, a card, an IP) never passes.
_MODEL_ID_OK = re.compile(r"(?:[a-z]{2,6}\.)?[a-z][a-z0-9-]*\.[a-z][a-z0-9.:-]*")


def scrub_field(key: str, value: Any, clean: Callable[[str], str] | None = None) -> Any:
    """:func:`scrub_value` for one ``extra`` field, leaving well-formed model IDs in
    :data:`MODEL_ID_FIELDS` as they are (element-wise for a list)."""
    clean = clean or scrub
    if key in MODEL_ID_FIELDS:
        if isinstance(value, str):
            return value if _MODEL_ID_OK.fullmatch(value) else clean(value)
        if isinstance(value, (list, tuple)):
            return [scrub_field(key, item, clean) for item in value]
    return _scrub_tree(value, clean)


def scrub_value(value: Any, _depth: int = 0) -> Any:
    """:func:`scrub` applied to every string inside ``value`` (maps, lists, bytes, objects)."""
    return _scrub_tree(value, scrub, _depth)


def _scrub_tree(value: Any, clean: Callable[[str], str], _depth: int = 0) -> Any:
    """``clean`` applied to every string inside ``value``; new containers, ``value`` is never changed."""
    if isinstance(value, str):
        return clean(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if _depth >= 6:
        return "[truncated]"
    if isinstance(value, (bytes, bytearray)):
        return clean(bytes(value).decode("utf-8", "replace"))
    if isinstance(value, Mapping):
        return {str(k): _scrub_tree(v, clean, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_scrub_tree(v, clean, _depth + 1) for v in value]
    try:
        return clean(str(value))
    except Exception:  # an object whose __str__ raises: never let logging fail  # noqa: BLE001
        return "[unprintable]"


def _exact_step() -> Callable[[str], str] | None:
    """The exact-value step for the bound call, or ``None`` when there is nothing to do.

    ``segments.redact_text(current_call_sid(), text)`` for each string, applied only when the text
    holds a registered value (``redact_text`` NFKC-normalizes, which would otherwise change clean
    text). ``segments`` imports this module, so it is looked up in ``sys.modules`` rather than
    imported: if it was never imported, nothing can have been registered. Cost with no call bound:
    one context-variable read; with a call bound and nothing registered: one dict lookup more.
    """
    call_sid = _call_sid_var.get()
    if not call_sid:
        return None
    segments = sys.modules.get("owlbank.segments")
    if segments is None or not segments.sensitive_values(call_sid):
        return None
    contains, redact = segments.contains_sensitive, segments.redact_text

    def exact(text: str) -> str:
        return redact(call_sid, text) if contains(call_sid, text) else text

    return exact


#: Upper bounds for :func:`safe_traceback`: output characters, exceptions followed along the chain, frames per exception.
_MAX_TRACE = 8192
_MAX_CHAIN = 5
_MAX_FRAMES = 40
_MAX_ERROR_MESSAGE = 500
#: Event names the metric filters (alarmed or counted) match as literal substrings of a whole log line (see
#: ``security_log``). A stack frame's source line could contain one (a caught error raised under a
#: call to ``otp_sent_untrusted_ani()``), which would count a metric for a line that is not that
#: event, so :func:`safe_traceback` breaks them up.
_ALARM_TERMS = ("twilio_auth_failed", "otp_sent_untrusted_ani", "otp_sent_keyed_phone")


def _class_name(exc: BaseException) -> str:
    cls = type(exc)
    module = getattr(cls, "__module__", "") or ""
    name = getattr(cls, "__qualname__", None) or cls.__name__
    return name if module in ("", "builtins", "__main__") else f"{module}.{name}"


def _chain_of(exc: BaseException) -> list[BaseException]:
    """``exc`` and what led to it (``__cause__``, else ``__context__``), at most :data:`_MAX_CHAIN`.

    ``raise X from None`` only hides the context when printing; here the class is still listed,
    because the root cause is what an investigation needs and a class name carries no data.
    """
    chain: list[BaseException] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen and len(chain) < _MAX_CHAIN:
        chain.append(cur)
        seen.add(id(cur))
        cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
    return chain


def safe_traceback(exc: BaseException) -> str:
    """The stack of a caught exception without anything the caller could have supplied.

    Per exception: ``Traceback (most recent call last):``, then each frame's file, line, function
    and source line (``traceback.format_tb``), then the module-qualified class name. For the
    ``__cause__``/``__context__`` chain the oldest exception comes first, joined by the
    standard wording. **Never** the exception message, ``args``, ``str(exc)``, ``repr(exc)`` or
    local variables (none of those is read), so a message that echoes a phone number, a code or a
    token cannot reach a log. Source lines can carry literals, so the result goes through
    :func:`scrub` anyway. Bounded: at most 5 exceptions, 40 frames each (the innermost), and
    8 KB (the end is kept, where the raise is). Never raises.
    """
    try:
        blocks: list[str] = []
        for item in _chain_of(exc):
            frames = traceback.extract_tb(item.__traceback__)[-_MAX_FRAMES:]
            lines = "".join(traceback.format_list(frames))
            blocks.append(f"Traceback (most recent call last):\n{lines}{_class_name(item)}")
        blocks.reverse()
        text = "\n\nThe above exception led to this one:\n\n".join(blocks)
        text = scrub(text)
        for term in _ALARM_TERMS:
            text = text.replace(term, term.replace("_", "-", 1))
        return text[-_MAX_TRACE:]
    except Exception:  # noqa: BLE001 - logging must never fail the caller
        return ""


def _exception_fields(exc_info: Any, exact: Callable[[str], str] | None = None) -> dict[str, Any]:
    """Fields for a record that carries ``exc_info`` (an unhandled error, or a library's own logging).

    ``error_message`` is the scrubbed message (logs it for unhandled errors only; code
    that catches an error uses :func:`log_caught`, which never logs the message). ``stack_trace`` is
    :func:`safe_traceback`, so the message appears once, scrubbed, and not again in the stack.
    ``exact`` (the exact-value step, :func:`_exact_step`) runs before the pattern scrub on the
    message, and on the stack (already pattern-scrubbed by :func:`safe_traceback`).
    """
    exc_type, exc, _tb = exc_info
    out: dict[str, Any] = {"error_class": getattr(exc_type, "__name__", "Exception")}
    try:
        message = str(exc)
    except Exception:  # noqa: BLE001
        message = ""
    if exact is not None:
        message = exact(message)
    out["error_message"] = scrub(message)[:_MAX_ERROR_MESSAGE]
    code = error_code_of(exc) if exc is not None else None
    if code is not None:
        out["error_code"] = code
    stack = safe_traceback(exc) if exc is not None else ""
    out["stack_trace"] = exact(stack) if exact is not None else stack
    return out


_STD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "taskName"}
#: Keys set by the formatter itself; an ``extra`` field with one of these names is renamed.
_RESERVED_KEYS = _STD_ATTRS | {"ts", "event"}
_FORMATTER_KEYS = _RESERVED_KEYS | {"component", "call_sid", "request_id", "trace_id"}
#: Keys the formatter writes itself; an ``emit`` field with one of these names gets a ``_`` suffix.
_OUTPUT_KEYS = frozenset({"ts", "level", "logger", "message"})
_CONTEXT_FIELDS = ("component", "call_sid", "request_id", "trace_id")


def _inject_context(record: logging.LogRecord) -> None:
    """Copy the bound correlation values onto ``record``; vet any explicit value (it is exempt from scrubbing)."""
    for name, var in _VARS.items():
        current = getattr(record, name, None)
        if current is None:
            value = var.get()
            if value:
                setattr(record, name, value)
        elif isinstance(current, str) and not _VALIDATORS[name].match(current):
            setattr(record, name, scrub(current))
        elif not isinstance(current, str):
            setattr(record, name, scrub_value(current))


class ContextFilter(logging.Filter):
    """Adds ``call_sid``, ``request_id``, ``trace_id`` and ``component`` to records (no scrubbing)."""

    def filter(self, record: logging.LogRecord) -> bool:
        _inject_context(record)
        return True


class ScrubFilter(logging.Filter):
    """Scrubs the message, arguments, ``extra`` fields and exception of every record.

    Two steps per string: first, while a call is bound, the exact-value step
    (``segments.redact_text`` with the values registered for that call), then the
    pattern scrub (:func:`scrub`). The bound identifiers (``call_sid``, ``request_id``,
    ``trace_id``, ``component``) are exempt from both. It only rebinds attributes of the record:
    the caller's message, arguments and ``extra`` objects are never changed.

    Attach it to a handler (it sees every record that handler writes, whichever logger made it).
    It also names the event, drops DEBUG and INFO records from the pinned libraries, and replaces
    ``exc_info`` by scrubbed ``error_class``/``error_message``/``stack_trace`` fields, so the
    Lambda runtime's own exception formatting (which would be unscrubbed) never runs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING and record.name.split(".", 1)[0] in PINNED_LOGGERS:
            return False
        _inject_context(record)
        if getattr(record, "_owlbank_scrubbed", False):
            return True
        exact = _exact_step()
        clean: Callable[[str], str] = scrub if exact is None else (lambda text: scrub(exact(text)))
        try:
            message = record.getMessage()
        except Exception:  # a bad format string or args: keep the raw template, scrubbed
            message = str(record.msg)
        message = clean(message)
        record.msg, record.args = message, None
        if not getattr(record, "component", None):
            record.component = _default_component()
        event = getattr(record, "event", None)
        record.__dict__["event"] = clean(event) if isinstance(event, str) and event else message
        for key, value in list(record.__dict__.items()):
            if key in _STD_ATTRS or key in _CONTEXT_FIELDS or key == "event" or key.startswith("_"):
                continue
            record.__dict__[key] = scrub_field(key, value, clean)
        if record.exc_info and record.exc_info[0] is not None:
            for key, value in _exception_fields(record.exc_info, exact).items():
                record.__dict__.setdefault(key, value)
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        record._owlbank_scrubbed = True
        return True


# --------------------------------------------------------------------------- formatter


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + f".{int(ts * 1000) % 1000:03d}Z"


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Scrubs again as its own step (idempotent)."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001
            message = str(record.msg)
        event = getattr(record, "event", None)
        if not isinstance(event, str) or not event:
            event = message
        out: dict[str, Any] = {"ts": _iso(record.created), "level": record.levelname,
                               "component": getattr(record, "component", None) or _default_component(),
                               "event": scrub(event)}
        for key in ("call_sid", "request_id", "trace_id"):
            value = getattr(record, key, None)
            if value:
                out[key] = value
        if scrub(message) != out["event"]:
            out["message"] = scrub(message)
        out["logger"] = record.name
        for key, value in record.__dict__.items():
            if key in _FORMATTER_KEYS or key.startswith("_"):
                continue
            out[key] = scrub_field(key, value)
        if record.exc_info and record.exc_info[0] is not None:
            for key, value in _exception_fields(record.exc_info).items():
                out.setdefault(key, value)
        return json.dumps(out, separators=(",", ":"), default=str, ensure_ascii=False)


# --------------------------------------------------------------------------- context

_call_sid_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("owlbank_call_sid", default=None)
_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("owlbank_request_id", default=None)
_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("owlbank_trace_id", default=None)
_component_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("owlbank_component", default=None)
_notes_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("owlbank_notes", default=None)
_VARS = {"call_sid": _call_sid_var, "request_id": _request_id_var, "trace_id": _trace_id_var,
         "component": _component_var}

# These values are exempt from scrubbing (they are what you search on), so only well-formed ones
# are ever bound: no free text, no log injection.
_CALL_SID_OK = re.compile(r"^[A-Za-z0-9_-]{4,64}$")
_REQUEST_ID_OK = re.compile(r"^[A-Za-z0-9-]{8,64}$")
_TRACE_ID_OK = re.compile(r"^1-[0-9a-f]{8}-[0-9a-f]{24}$")
_COMPONENT_OK = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
_VALIDATORS = {"call_sid": _CALL_SID_OK, "request_id": _REQUEST_ID_OK, "trace_id": _TRACE_ID_OK,
               "component": _COMPONENT_OK}

_default_component_name = "owlbank"
_Tokens = list[tuple[contextvars.ContextVar[Any], contextvars.Token[Any]]]


def _default_component() -> str:
    return _component_var.get() or _default_component_name


def current_call_sid() -> str | None:
    return _call_sid_var.get()


def current_request_id() -> str | None:
    return _request_id_var.get()


def bind(*, call_sid: str | None = None, request_id: str | None = None, trace_id: str | None = None,
         component: str | None = None) -> _Tokens:
    """Bind correlation values (malformed or empty ones are ignored). Returns tokens for :func:`unbind`."""
    tokens: _Tokens = []
    for name, value in (("call_sid", call_sid), ("request_id", request_id), ("trace_id", trace_id),
                        ("component", component)):
        if isinstance(value, str) and _VALIDATORS[name].match(value):
            var = _VARS[name]
            tokens.append((var, var.set(value)))
    return tokens


def unbind(tokens: _Tokens) -> None:
    for var, token in reversed(tokens):
        try:
            var.reset(token)
        except ValueError:  # reset in a different context (a task boundary): drop the value instead
            var.set(None)


def unbind_all() -> None:
    """Clear every bound correlation value in this context (tests; never needed in production)."""
    for var in _VARS.values():
        var.set(None)


def bind_call_sid(call_sid: Any) -> _Tokens:
    return bind(call_sid=call_sid if isinstance(call_sid, str) else None)


@contextlib.contextmanager
def call_context(call_sid: Any = None, *, component: str | None = None,
                 request_id: str | None = None) -> Iterator[None]:
    """Bind the call SID (and optionally component and request ID) for the ``with`` block."""
    tokens = bind(call_sid=call_sid if isinstance(call_sid, str) else None, component=component,
                  request_id=request_id)
    try:
        yield
    finally:
        unbind(tokens)


_BODY_CALL_SID = re.compile(r"(?:^|[&\s{,\"])(?:CallSid|call_sid)(?:=|\"?\s*:\s*\")([A-Za-z0-9_-]{4,64})")


def extract_call_sid(event: Any) -> str | None:
    """The call SID of a Lambda event, if it has one: direct key, query string, or form body."""
    if not isinstance(event, Mapping):
        return None
    for key in ("call_sid", "CallSid", "callSid"):
        value = event.get(key)
        if isinstance(value, str) and _CALL_SID_OK.match(value):
            return value
    query = event.get("queryStringParameters")
    if isinstance(query, Mapping):
        value = query.get("call_sid") or query.get("CallSid")
        if isinstance(value, str) and _CALL_SID_OK.match(value):
            return value
    body = event.get("body")
    if isinstance(body, str) and body and len(body) < 65536:
        if event.get("isBase64Encoded"):
            import base64  # noqa: PLC0415

            try:
                body = base64.b64decode(body, validate=False).decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                return None
        m = _BODY_CALL_SID.search(body)
        if m:
            return m.group(1)
    return None


def _trace_from_env() -> str | None:
    raw = os.environ.get("_X_AMZN_TRACE_ID", "")
    for part in raw.split(";"):
        key, _, value = part.strip().partition("=")
        if key == "Root" and _TRACE_ID_OK.match(value):
            return value
    return None


def bind_lambda(event: Any, context: Any = None, *, component: str | None = None) -> _Tokens:
    """Bind request ID, trace ID and call SID for one Lambda invocation (call :func:`unbind` after)."""
    request_id = getattr(context, "aws_request_id", None)
    return bind(call_sid=extract_call_sid(event), request_id=request_id if isinstance(request_id, str) else None,
                trace_id=_trace_from_env(), component=component)


# --------------------------------------------------------------------------- emit


#: Fields stamped on every :func:`emit` line of some events (:func:`set_event_fields`).
_event_fields: dict[str, dict[str, str]] = {}


def set_event_fields(events: Iterable[str], /, **fields: str) -> None:
    """Stamp ``fields`` on every :func:`emit` line whose event is in ``events``; a field the caller
    passes explicitly wins. Replaces any earlier stamp (no events clears it). For process-wide
    facts that a few lines must carry, logged by code that doesn't know them: the bridge's
    ``hosting`` on its ``ws_session_*`` lines. Names must be free (not a key the
    formatter writes) and values short identifiers (``[A-Za-z0-9_.-]{1,40}``), so a stamp can
    never carry free text; anything else raises ``ValueError``."""
    global _event_fields
    for name, value in fields.items():
        if name in _FORMATTER_KEYS or name in _OUTPUT_KEYS or not name.isidentifier():
            raise ValueError("a stamped field name must not be a reserved key")
        if not isinstance(value, str) or not _COMPONENT_OK.match(value):
            raise ValueError("a stamped field value must be a short identifier")
    _event_fields = {event: dict(fields) for event in events} if fields else {}


def emit(logger: logging.Logger, level: int, event: str, /, **fields: Any) -> None:
    """Log ``event`` with ``fields`` as top-level JSON keys (reserved key names are suffixed ``_``).
    Fields stamped for ``event`` (:func:`set_event_fields`) are added unless passed explicitly."""
    if not logger.isEnabledFor(level):
        return
    stamp = _event_fields.get(event)
    if stamp:
        fields = {**stamp, **fields}
    exc_info = fields.pop("exc_info", None)
    extra = {(k + "_" if k in _RESERVED_KEYS or k in _OUTPUT_KEYS else k): v for k, v in fields.items()}
    extra["event"] = event
    logger.log(level, event, extra=extra, exc_info=exc_info, stacklevel=2)


def log_caught(logger: logging.Logger, event: str, exc: BaseException, /, *, decision: str,
               level: int = logging.WARNING, **fields: Any) -> None:
    """Log an error the code caught and handled: what it was, where, and what was done about it.

    Fields: ``error_class``, ``error_code`` (as :func:`error_code_of`), ``http_status`` when the
    exception has one, ``decision`` (``retry``, ``fallback_*``, ``reject``...), the identifiers in
    ``fields`` and ``stack`` (:func:`safe_traceback`: frames and class names, never the message).
    The exception is not attached as ``exc_info``, so its message is not logged anywhere. Use it
    in place of ``log.exception`` and ``exc_info=True`` (a test enforces this). Not for expected,
    high-volume outcomes; log those with :func:`emit` and no stack.
    """
    if not logger.isEnabledFor(level):
        return
    out: dict[str, Any] = {"error_class": type(exc).__name__}
    code = error_code_of(exc)
    if code is not None:
        out["error_code"] = code
    status = _http_status_of(exc)
    if status is not None:
        out["http_status"] = status
    out.update(fields)
    out["decision"] = decision
    out["stack"] = safe_traceback(exc)
    emit(logger, level, event, **out)


# --------------------------------------------------------------------------- configuration

_lock = threading.Lock()
_warned_invalid = False
_OWN_HANDLER = "_owlbank_handler"


def _parse_level(raw: str | None) -> tuple[int, str | None]:
    """``(level, invalid_value)``: INFO when unset, and INFO plus the offending value when invalid."""
    value = os.environ.get(LOG_LEVEL_ENV) if raw is None else raw
    if value is None or not value.strip():
        return logging.INFO, None
    level = _LEVELS.get(value.strip().upper())
    if level is None:
        return logging.INFO, value.strip()[:20]
    return level, None


def _warn_invalid_once(invalid: str | None) -> None:
    global _warned_invalid
    if invalid is not None and not _warned_invalid:
        _warned_invalid = True
        emit(_log, logging.WARNING, "invalid_log_level", variable=LOG_LEVEL_ENV, value=invalid, using="INFO")


def resolve_level(raw: str | None = None) -> int:
    """The log level from ``OWLBANK_LOG_LEVEL`` (or ``raw``). Invalid values fall back to INFO, warning once."""
    level, invalid = _parse_level(raw)
    _warn_invalid_once(invalid)
    return level


def _pin_loggers() -> None:
    for name in PINNED_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in list(logging.root.manager.loggerDict):
        if name.split(".", 1)[0] in PINNED_LOGGERS and "." in name:
            child = logging.root.manager.loggerDict[name]
            if isinstance(child, logging.Logger):
                child.setLevel(logging.NOTSET)  # inherit the pinned parent


def _is_lambda_handler(handler: logging.Handler) -> bool:
    return type(handler).__name__.startswith("LambdaLoggerHandler")


def _is_json_runtime(handler: logging.Handler) -> bool:
    fmt = handler.formatter
    if fmt is not None and "json" in type(fmt).__name__.lower():
        return True
    return os.environ.get("AWS_LAMBDA_LOG_FORMAT", "").upper() == "JSON"


def _add_filter(handler: logging.Handler) -> None:
    if not any(isinstance(f, ScrubFilter) for f in handler.filters):
        handler.addFilter(ScrubFilter())


def route_to_root(logger: logging.Logger) -> None:
    """Remove ``logger``'s own handlers and let its records propagate to the root handler that
    :func:`configure_logging` set up (scrubber, JSON formatter, query-string filter). Idempotent.

    For a library that installs its own handler and stops propagation (TAC does both on the
    ``tac`` logger). The logger's level is kept, so the library's own level setting still applies.
    """
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = True


def configure_logging(component: str, *, level: int | str | None = None,
                      stream: TextIO | None = None) -> int:
    """Set up JSON logging for ``component`` (``frontdoor``, ``bridge``, ...). Idempotent.

    * Under the Lambda runtime: keep its handler, add the scrubber, and (text format only) give it
      the JSON formatter. No second handler is added.
    * Elsewhere: one stream handler on the root logger (stdout unless ``stream``), JSON formatter,
      scrubber; uvicorn's own handlers are removed so its lines flow through ours.
    * The level comes from ``OWLBANK_LOG_LEVEL`` unless ``level`` is given; pinned libraries stay
      at WARNING whatever it is.

    Returns the effective level.
    """
    global _default_component_name
    invalid: str | None = None
    with _lock:
        if isinstance(level, int):
            resolved = level
        else:
            resolved, invalid = _parse_level(level)
        if _COMPONENT_OK.match(component or ""):
            _default_component_name = component
        root = logging.getLogger()
        lambda_handlers = [h for h in root.handlers if _is_lambda_handler(h)]
        if lambda_handlers:
            for h in lambda_handlers:
                _add_filter(h)
                if not _is_json_runtime(h):
                    h.setFormatter(JsonFormatter())
        else:
            own = [h for h in root.handlers if getattr(h, _OWN_HANDLER, False)]
            if own:
                handler: logging.Handler = own[0]
                if stream is not None and isinstance(handler, logging.StreamHandler):
                    handler.setStream(stream)
            else:
                handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
                setattr(handler, _OWN_HANDLER, True)
                handler.setFormatter(JsonFormatter())
                root.addHandler(handler)
            _add_filter(handler)
            for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
                lg = logging.getLogger(name)
                if lg.handlers:
                    lg.handlers.clear()
                lg.propagate = True
        root.setLevel(resolved)
        _pin_loggers()
    _warn_invalid_once(invalid)  # after the handler is in place, so the warning is a JSON line too
    from owlbank import security_log  # noqa: PLC0415  (query-string stripping stays installed)

    security_log.install_log_redaction()
    return resolved


def reset_logging() -> None:
    """Test helper: remove our handler, reset the warn-once flag, the default component and the
    stamped event fields."""
    global _warned_invalid, _default_component_name, _event_fields
    with _lock:
        _event_fields = {}
        root = logging.getLogger()
        for h in list(root.handlers):
            if getattr(h, _OWN_HANDLER, False):
                root.removeHandler(h)
        for h in root.handlers:
            for f in [f for f in h.filters if isinstance(f, ScrubFilter)]:
                h.removeFilter(f)
        _warned_invalid = False
        _default_component_name = "owlbank"


# --------------------------------------------------------------------------- errors and responses


def error_code_of(exc: BaseException) -> str | None:
    """The AWS or Twilio error code of ``exc``, never its message.

    botocore ``ClientError``: ``response["Error"]["Code"]``. Twilio exceptions: ``code``
    (Twilio's error number), else ``status`` (HTTP). Anything else: ``None``.
    """
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        err = response.get("Error")
        if isinstance(err, Mapping) and err.get("Code"):
            return str(err["Code"])[:80]
    for attr in ("code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and value != "":
            return str(value)[:80]
    return None


def _http_status_of(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        meta = response.get("ResponseMetadata")
        if isinstance(meta, Mapping) and isinstance(meta.get("HTTPStatusCode"), int):
            return int(meta["HTTPStatusCode"])
    status = getattr(exc, "status", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def _aws_request_id_of(exc: BaseException) -> str | None:
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        meta = response.get("ResponseMetadata")
        if isinstance(meta, Mapping) and isinstance(meta.get("RequestId"), str):
            value = meta["RequestId"]
            return value if _REQUEST_ID_OK.match(value) else None
    return None


def note(**fields: Any) -> None:
    """Add fields to the current invocation's ``invocation_end`` line (no-op outside one)."""
    notes = _notes_var.get()
    if notes is not None:
        notes.update(fields)


def degraded(reason: str) -> None:
    """Mark the current invocation as degraded (it answered, but through a fallback)."""
    notes = _notes_var.get()
    if notes is not None:
        notes.setdefault("degraded_by", reason)


_DEGRADED_CODES = frozenset({"unavailable", "UNAVAILABLE", "internal", "bad_request", "UNKNOWN_TOOL",
                             "UNKNOWN_OPERATION"})


def classify_response(response: Any) -> tuple[str, str | None]:
    """``(outcome, result_code)`` for a handler's return value. Never inspects the body text."""
    if isinstance(response, Mapping):
        status = response.get("statusCode")
        if isinstance(status, int) and not isinstance(status, bool):
            return ("error" if status >= 500 else "ok"), str(status)
        policy = response.get("policyDocument")
        if isinstance(policy, Mapping):
            stmts = policy.get("Statement")
            effect = stmts[0].get("Effect") if isinstance(stmts, list) and stmts and isinstance(stmts[0], Mapping) else None
            return "ok", str(effect).lower() if effect else None
        for key in ("error", "result"):
            code = response.get(key)
            if isinstance(code, Mapping):
                code = code.get("code")
            if isinstance(code, str) and code:
                return ("degraded" if code in _DEGRADED_CODES else "ok"), code[:80]
    return "ok", None


def _size_of(event: Any) -> int:
    try:
        return len(json.dumps(event, default=str, separators=(",", ":")))
    except Exception:  # noqa: BLE001
        return -1


_inv_log = logging.getLogger("owlbank.invocation")


def instrumented(component: str, op: str | Callable[[Any, Any], str | None] | None = None) -> Callable[..., Any]:
    """Decorator for a Lambda handler ``(event, context=None)``: correlation, start, end and error lines.

    Logs ``invocation_start`` (op, event size in bytes, never the body) and ``invocation_end``
    (outcome ``ok|error|degraded``, ``duration_ms``, ``result_code``, plus anything added with
    :func:`note`). On an unhandled exception it logs ``invocation_error`` (error class, scrubbed
    message, :func:`safe_traceback` stack trace, the call SID and op) then the end line with outcome ``error``, and
    re-raises. The return value is passed through untouched. ``op`` is a fixed name, a function
    ``(event, context) -> name``, or ``None`` (use ``event["op"]`` when present).
    """
    configured = False

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(event: Any, context: Any = None, *args: Any, **kwargs: Any) -> Any:
            nonlocal configured
            if not configured and os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
                # On the first invocation, not at import: the runtime has installed its handler by then.
                configured = True
                configure_logging(component)
            tokens = bind_lambda(event, context, component=component)
            notes: dict[str, Any] = {}
            notes_token = _notes_var.set(notes)
            op_name = _resolve_op(op, event, context)
            started = time.perf_counter()
            fields: dict[str, Any] = {}
            if op_name:
                fields["op"] = op_name
            emit(_inv_log, logging.INFO, "invocation_start", event_bytes=_size_of(event), **fields)
            try:
                result = fn(event, context, *args, **kwargs)
            except Exception as exc:
                duration = round((time.perf_counter() - started) * 1000, 1)
                emit(_inv_log, logging.ERROR, "invocation_error", exc_info=sys.exc_info(), **fields)
                emit(_inv_log, logging.ERROR, "invocation_end", outcome="error", duration_ms=duration,
                     result_code=type(exc).__name__, **fields)
                raise
            else:
                duration = round((time.perf_counter() - started) * 1000, 1)
                outcome, code = classify_response(result)
                if outcome == "ok" and notes.get("degraded_by"):
                    outcome = "degraded"
                extra = {k: v for k, v in notes.items() if k not in ("outcome", "duration_ms", "result_code", "op")}
                level = logging.WARNING if outcome != "ok" else logging.INFO
                if code is not None:
                    extra["result_code"] = code
                emit(_inv_log, level, "invocation_end", outcome=outcome, duration_ms=duration, **fields, **extra)
                return result
            finally:
                _notes_var.reset(notes_token)
                unbind(tokens)

        return wrapper

    return decorate


def _resolve_op(op: Any, event: Any, context: Any) -> str | None:
    try:
        if callable(op):
            value = op(event, context)
        elif isinstance(op, str):
            value = op
        else:
            value = event.get("op") if isinstance(event, Mapping) else None
    except Exception:  # noqa: BLE001
        value = None
    return value[:64] if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", value) else None


# --------------------------------------------------------------------------- dependency calls

_dep_log = logging.getLogger("owlbank.dependency")


class timed_call:  # noqa: N801  (used like a function: a context manager and a decorator)
    """Time one call to a dependency and log ``dependency_call``.

    ``with timed_call("dynamodb", "get_item"): ...`` or ``@timed_call("kms", "decrypt")``. Fields:
    ``dependency``, ``operation``, ``outcome`` (``ok``, ``slow``, ``error``, ``condition_failed``),
    ``duration_ms`` and, on failure, ``error_class`` and ``error_code`` (from
    :func:`error_code_of`), ``http_status``, the AWS request ID when present and ``stack``
    (:func:`safe_traceback`); never the message. An expected failure (``condition_failed``) has no stack.

    Levels: failures and calls slower than :data:`SLOW_CALL_MS` log at WARNING; successes at
    ``ok_level`` (INFO by default; DEBUG for high-frequency calls such as store reads).
    The exception is always re-raised. ``expected_codes`` lists error codes that are normal
    control flow (a conditional-write miss): logged at DEBUG as ``condition_failed``.
    """

    def __init__(self, dependency: str, operation: str, *, ok_level: int = logging.INFO,
                 expected_codes: tuple[str, ...] = ()) -> None:
        self.dependency, self.operation = dependency, operation
        self.ok_level, self.expected_codes = ok_level, expected_codes
        self._started = 0.0

    def __enter__(self) -> timed_call:
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> Literal[False]:
        duration = round((time.perf_counter() - self._started) * 1000, 1)
        fields: dict[str, Any] = {"dependency": self.dependency, "operation": self.operation,
                                  "duration_ms": duration}
        if exc is None:
            slow = duration > SLOW_CALL_MS
            emit(_dep_log, logging.WARNING if slow else self.ok_level, "dependency_call",
                 outcome="slow" if slow else "ok", **fields)
            return False
        code = error_code_of(exc)
        fields.update(error_class=type(exc).__name__)
        if code is not None:
            fields["error_code"] = code
        status = _http_status_of(exc)
        if status is not None:
            fields["http_status"] = status
        request_id = _aws_request_id_of(exc)
        if request_id:
            fields["aws_request_id"] = request_id
        if code is not None and code in self.expected_codes:
            emit(_dep_log, logging.DEBUG, "dependency_call", outcome="condition_failed", **fields)
        else:
            emit(_dep_log, logging.WARNING, "dependency_call", outcome="error",
                 stack=safe_traceback(exc), **fields)
        return False

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def fresh() -> timed_call:
            return timed_call(self.dependency, self.operation, ok_level=self.ok_level,
                              expected_codes=self.expected_codes)

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                with fresh():
                    return await fn(*args, **kwargs)

            return awrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with fresh():
                return fn(*args, **kwargs)

        return wrapper


# --------------------------------------------------------------------------- span timings

#: A span name: a short lower-case identifier (``state_load``, ``resolve``), never a value.
_SPAN_NAME = re.compile(r"[a-z][a-z0-9_]{0,39}")


class Timings:
    """Named spans of one request, accumulated in milliseconds, for a log line's ``timings_ms``.

    ::

        t = Timings()
        with t.span("state_load"):
            sess = load()
        with t.span("resolve"):
            with t.span("catalog"):     # nesting: "resolve" includes the "catalog" time
                ...
        emit(log, logging.INFO, "factor_answer", ..., timings_ms=t.to_dict())

    * A name used again adds to its total (two ``catalog`` reads give one ``catalog`` entry).
    * Spans nest. Each span's total is its own wall time, so a parent includes its children; the
      entries are not meant to add up to the request's ``duration_ms``. A span re-entered while
      it is already open (the same name, nested) is counted once, by the outer span.
    * A span whose body raises still records its time, and the exception propagates.
    * :meth:`to_dict` gives ``{name: int}`` in the order the names were first opened, each total
      rounded to whole milliseconds once, at the end (so many short spans don't round to zero).
    * Names only, never values: a name must match ``[a-z][a-z0-9_]{0,39}``, or ``ValueError``.
      Nothing about the work timed is kept.

    ``clock`` returns seconds (default :func:`time.perf_counter`); tests pass a fake. One
    instance per request; not thread-safe.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._totals: dict[str, float] = {}
        self._open: set[str] = set()

    @contextlib.contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Time the ``with`` body under ``name`` (see the class docstring)."""
        if not isinstance(name, str) or _SPAN_NAME.fullmatch(name) is None:
            raise ValueError("a span name must be a short lower-case identifier")
        self._totals.setdefault(name, 0.0)
        if name in self._open:  # re-entered: the outer span counts this time
            yield
            return
        self._open.add(name)
        started = self._clock()
        try:
            yield
        finally:
            self._totals[name] += self._clock() - started
            self._open.discard(name)

    def to_dict(self) -> dict[str, int]:
        """``{name: milliseconds}``; a span still open counts what it has recorded so far."""
        return {name: round(total * 1000) for name, total in self._totals.items()}
