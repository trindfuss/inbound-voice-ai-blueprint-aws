"""Context API for the Flex screen pop: ``GET /context/{journal_ref}?call_sid=CA...``.

* **Auth:** the Flex token authorizer (``lambdas.flex_authorizer``) runs first and
  puts ``worker_sid`` and ``roles`` (comma-separated) in the request context. This handler
  refuses requests without ``worker_sid`` (defence in depth if the route is ever wired without
  the authorizer). **Per-task check**: a valid token is
  not enough. The plugin also sends ``task_sid`` (query), and :func:`authorize_task` looks the
  task up in TaskRouter: the task's attributes ``call_sid`` and ``journal_ref`` must equal the
  request's, and the caller must hold a reservation on it (pending, accepted or wrapping) or have
  the ``supervisor`` or ``admin`` role. Every denial (a missing or bad ``task_sid`` included) is
  the same 404 as an unknown call; if TaskRouter can't be reached within one 3 s deadline for
  both reads (:data:`TASKROUTER_TIMEOUT_S`) the answer is 503 with no data (never fail open). The
  reader is :class:`owlbank.taskrouter.TwilioTaskRouterReader`. Each read writes one
  ``context api: access`` log line: worker, journal ref, call SID and decision, never
  content, the token or the Twilio key.
* **Ref check:** the plugin sends the task's ``journal_ref`` (path) and
  ``call_sid`` (query), both from the task attributes. The handler loads the
  call from the ``calls`` table and requires the stored escalation
  ``journal_ref`` to equal the path ref (constant-time compare). An unknown call,
  an expired call, a call that never escalated and a mismatched ref all give the
  same 404, so the API can't be used to probe refs or calls.
* **Redaction:** the timeline is built from an allow-list of journal fields:
  ``seq``, ``ts``, ``type``, ``prompt_key``, ``rendered_text`` (prompt events
  only), ``outcome``, ``tool`` (name only, no args or results), ``factor`` (the
  factor *name*), ``level``, ``reason`` and ``queue``. Any event carrying a
  denied key (``owlbank.segments.is_denied_key``) is dropped whole, digit runs of
  four or more in rendered text are masked, and AI turns (free caller speech)
  are never returned. Factor values and tokens therefore never leave the API.
* **Transcript redaction** the kept prompts'
  ``rendered_text`` (digit runs already masked) are joined with newline separators, redacted with
  one :meth:`owlbank.pii.PiiGuard.text` call (Comprehend, chunked by the redactor), and split
  back by each text's own line count. The guard is built from the policy table's ``redaction``
  item (re-read after the cache TTL). ``enabled: false`` makes no Comprehend call. If the check is
  unavailable (``on_failure: withhold``), or the redacted text no longer splits into the same
  texts (a mask across a separator), every prompt text becomes the withheld text. Prompts never
  carry a sensitive value, so the API needs no token-key (KMS) grant.
* **CORS:** only ``https://flex.twilio.com`` (``ALLOWED_ORIGIN``) is echoed in
  ``Access-Control-Allow-Origin``; other origins get no CORS headers.

Environment: ``OWLBANK_TABLE_CALLS``, ``OWLBANK_TABLE_SEGMENTS``, ``OWLBANK_TABLE_POLICY``,
``OWLBANK_TABLE_CATALOG`` (see ``owlbank.config``), ``OWLBANK_CACHE_TTL_S``, ``AWS_REGION``;
``OWLBANK_TASKROUTER_WORKSPACE_SID`` (the Flex workspace, ``WS...``) and
``OWLBANK_SECRET_TWILIO_API_KEY`` (the key read for the TaskRouter lookup: its own Restricted-key
secret ``twilio/context-api-key`` with ``contextApiRestrictedKey=on``, the default, else the shared
``twilio/api-key``).

**Withheld text**: a withheld prompt shows the catalog's ``escalation.summary_withheld`` in
the call's language, read through :class:`owlbank.catalog.PromptCatalog` with its cache
(:func:`withheld`; the role reads the ``catalog`` table, same data CMK, no new KMS grant). Any
catalog error falls back to :data:`owlbank.pii.WITHHELD_TURN`. :func:`get_context` takes any
``withheld(language)`` supplier.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from owlbank import callstate, obs
from owlbank.catalog import PromptCatalog
from owlbank.pii import WITHHELD_TURN, PiiGuard, PiiRedactor, RedactionConfig, load_redaction_config
from owlbank.segments import SegmentLog, is_denied_key
from owlbank.store import Store
from owlbank.taskrouter import TaskRouterReader, TaskRouterUnavailable, TwilioTaskRouterReader

log = logging.getLogger(__name__)

ALLOWED_ORIGIN = "https://flex.twilio.com"
ALLOWED_HEADERS = "Content-Type,X-Flex-Token"

_CALL_SID_RE = re.compile(r"^CA[0-9a-fA-F]{32}$")
_REF_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_TASK_SID_RE = re.compile(r"^WT[0-9a-fA-F]{32}$")
_DIGIT_RUN_RE = re.compile(r"\d(?:[\s-]*\d){3,}")
_MAX_TEXT = 500

#: Event types returned in the timeline. ``ai_turn`` (free speech) is left out.
TIMELINE_TYPES = frozenset({
    "call_start", "lookup", "prompt", "factor_asked", "verification",
    "tool_call", "tool_result", "handoff", "escalation", "call_end",
    "confirm_asked", "confirm_answer",
    "conversation_closed",  # human leg: with ``closed_by`` only
})
#: Fields copied from an event, besides ``rendered_text`` (prompt events only) and ``closed_by``
#: (``conversation_closed`` events only).
TIMELINE_FIELDS = ("seq", "ts", "type", "prompt_key", "outcome", "tool", "factor", "level",
                   "reason", "queue", "answer")
#: ``conversation_closed.closed_by`` values shown; its ``conversation_id`` stays out (the agent's
#: desktop has no use for it).
CLOSED_BY_VALUES = frozenset({"status", "transcription", "sweep"})


# --------------------------------------------------------------------------- redaction


def _mask_digits(text: str) -> str:
    return _DIGIT_RUN_RE.sub("[redacted]", text)


def _scalar(value: Any) -> str | int | bool | None:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return _mask_digits(value[:_MAX_TEXT])
    return None  # nested structures are never returned


def _has_denied_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(is_denied_key(str(k)) or _has_denied_key(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_denied_key(v) for v in value)
    return False


def redact_event(event: Mapping[str, Any]) -> dict[str, Any] | None:
    """One journal event as a timeline entry, or ``None`` if it must not be shown."""
    etype = event.get("type")
    if etype not in TIMELINE_TYPES or _has_denied_key(event):
        return None
    out: dict[str, Any] = {}
    for key in TIMELINE_FIELDS:
        if key in event:
            value = _scalar(event[key])
            if value is not None:
                out[key] = value
    if etype == "prompt" and isinstance(event.get("rendered_text"), str):
        out["rendered_text"] = _mask_digits(event["rendered_text"][:_MAX_TEXT])
    if etype == "conversation_closed" and event.get("closed_by") in CLOSED_BY_VALUES:
        out["closed_by"] = event["closed_by"]
    return out


SURFACE_TIMELINE = "screen_pop_timeline"
WITHHELD_KEY = "escalation.summary_withheld"
_SEP = "\n"


def join_texts(texts: Sequence[str]) -> str:
    """The texts as one string, newline-separated (one Comprehend batch)."""
    return _SEP.join(texts)


def split_texts(joined: str, like: Sequence[str]) -> list[str] | None:
    """Undo :func:`join_texts` after redaction: text ``i`` takes as many lines as ``like[i]`` has.
    ``None`` when the line count changed (a mask swallowed or added a newline)."""
    if not like:
        return [] if joined == "" else None
    lines = joined.split(_SEP)
    sizes = [t.count(_SEP) + 1 for t in like]
    if len(lines) != sum(sizes):
        return None
    out, pos = [], 0
    for size in sizes:
        out.append(_SEP.join(lines[pos:pos + size]))
        pos += size
    return out


Withheld = str | Callable[[], str]


def _resolve(withheld: Withheld) -> str:
    return withheld if isinstance(withheld, str) else withheld()


def redact_texts(texts: Sequence[str], pii: PiiGuard, *, language: str,
                 withheld: Withheld) -> list[str]:
    """``texts`` with PII masked, in order, with one :meth:`PiiGuard.text` call. On ``withhold``
    (check unavailable), or if the result no longer splits into ``len(texts)`` texts, every text
    is ``withheld`` (fail closed). With redaction off, the texts unchanged and no call.
    ``withheld`` may be a supplier, called only when something is withheld (so the catalog is
    read only then)."""
    if not texts:
        return []
    marker = f"\x00withheld-{uuid.uuid4().hex}\x00"  # can't be a redaction result of real text
    joined = pii.text(SURFACE_TIMELINE, join_texts(texts), language=language, withheld=marker)
    if joined == marker:
        return [_resolve(withheld)] * len(texts)
    parts = split_texts(joined, texts)
    if parts is None:
        log.warning("timeline redaction changed the prompt boundaries; prompts withheld",
                    extra={"surface": SURFACE_TIMELINE, "decision": "withhold"})
        return [_resolve(withheld)] * len(texts)
    return parts


def build_timeline(events: Sequence[Mapping[str, Any]], *, pii: PiiGuard | None = None,
                   language: str | None = None,
                   withheld: Withheld = WITHHELD_TURN) -> list[dict[str, Any]]:
    """The allow-listed timeline. With ``pii``, the prompt texts are redacted in one batch;
    without it, only the digit-run mask applies."""
    timeline = [e for e in (redact_event(ev) for ev in events) if e is not None]
    if pii is None:
        return timeline
    prompts = [e for e in timeline if isinstance(e.get("rendered_text"), str)]
    redacted = redact_texts([e["rendered_text"] for e in prompts], pii,
                            language=language or "en-US", withheld=withheld)
    for entry, text in zip(prompts, redacted, strict=True):
        entry["rendered_text"] = text
    return timeline


class _RedactionItem:
    """The policy table's ``redaction`` item as a :class:`RedactionConfig`, re-read after
    ``ttl_s``. A read error raises (``PiiGuard`` then falls back to the enabled defaults)."""

    def __init__(self, store: Store, ttl_s: float, clock: Callable[[], float]) -> None:
        self._store, self._ttl, self._clock = store, ttl_s, clock
        self._lock = threading.Lock()
        self._cfg: RedactionConfig | None = None
        self._at = 0.0

    def __call__(self) -> RedactionConfig:
        with self._lock:
            now = self._clock()
            if self._cfg is None or now - self._at >= self._ttl:
                item = self._store.get("policy", {"pk": "redaction"})
                raw = None if item is None else {k: v for k, v in item.items()
                                                 if k not in ("pk", "rev")}
                self._cfg, self._at = load_redaction_config(raw), now
            return self._cfg


def redaction_guard(store: Store, redactor: PiiRedactor, *, ttl_s: float,
                    clock: Callable[[], float] = time.monotonic) -> PiiGuard:
    """A :class:`PiiGuard` on the policy's ``redaction`` item (the bridge and front door use the
    whole ``Policy``; this API needs only the one item, so a bad item elsewhere can't break it)."""
    return PiiGuard(redactor, _RedactionItem(store, ttl_s, clock))


class CatalogWithheld:
    """``withheld(language)``: the catalog's ``escalation.summary_withheld`` in ``language``.

    The :class:`PromptCatalog` is built on first use (its first read can fail; it is tried again
    on the next call) and then served from its cache, re-read after ``ttl_s``. Any error, or an
    empty text, gives :data:`WITHHELD_TURN`: logged with no text (fail closed, never a 500)."""

    def __init__(self, store: Store, *, ttl_s: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._store, self._ttl, self._clock = store, ttl_s, clock
        self._lock = threading.Lock()
        self._catalog: PromptCatalog | None = None

    def _get_catalog(self) -> PromptCatalog:
        with self._lock:
            if self._catalog is None:
                self._catalog = PromptCatalog.from_store(self._store, ttl_s=self._ttl,
                                                         clock=self._clock)
            return self._catalog

    def __call__(self, language: str) -> str:
        try:
            text = self._get_catalog().text(WITHHELD_KEY, language)
        except Exception as exc:
            obs.log_caught(log, "withheld text: catalog unavailable", exc,
                           operation="withheld_text", decision="fallback_builtin_text")
            return WITHHELD_TURN
        return text if isinstance(text, str) and text.strip() else WITHHELD_TURN


# --------------------------------------------------------------------------- per-task check

#: Roles that may read any escalated call's panel. Exact, case-sensitive match.
PRIVILEGED_ROLES = frozenset({"supervisor", "admin"})
#: Reservation statuses that count as "holds the task": offered, taken, or in wrap-up.
HELD_RESERVATION_STATUSES = frozenset({"pending", "accepted", "wrapping"})

#: One deadline, in seconds, for the per-task check's TaskRouter reads together (the task, then its
#: reservations): the second read gets only what the first left, and none left is "unavailable"
#: (503). The agent is waiting on a panel, and the Lambda's own timeout is 5 s. Each read's limit
#: goes to ``urlopen``, which applies it to every socket wait (it bounds a stall, not a trickle).
TASKROUTER_TIMEOUT_S = 3.0


def parse_roles(raw: Any) -> tuple[str, ...]:
    """The authorizer's comma-separated ``roles`` context value as a tuple (empty if absent)."""
    if not isinstance(raw, str):
        return ()
    return tuple(r.strip() for r in raw.split(",") if r.strip())


def authorize_task(worker_sid: str, roles: Sequence[str], task_sid: str, call_sid: str,
                   journal_ref: str, *, taskrouter: TaskRouterReader,
                   budget_s: float = TASKROUTER_TIMEOUT_S,
                   clock: Callable[[], float] = time.monotonic) -> bool:
    """True only if the task's attributes carry this ``call_sid`` and ``journal_ref`` AND the
    worker holds a reservation on the task or has a privileged role. Anything else is False
    (the caller answers the one 404). Raises :class:`TaskRouterUnavailable` when TaskRouter can't
    be asked: the caller answers 503 and returns no data. Any other error from the reader is
    treated the same way (fail closed).

    Both reads share one deadline, ``budget_s`` from the start: the reservations read gets the
    time the task read left, and if none is left that is :class:`TaskRouterUnavailable`."""
    deadline = clock() + budget_s
    try:
        attrs = taskrouter.task_attributes(task_sid, timeout_s=budget_s)
        if attrs is None or not _attr_equals(attrs.get("call_sid"), call_sid) \
                or not _attr_equals(attrs.get("journal_ref"), journal_ref):
            return False
        if any(r in PRIVILEGED_ROLES for r in roles):
            return True
        left = deadline - clock()
        if not left > 0:
            raise TaskRouterUnavailable("TaskRouter time budget spent")
        return any(
            res.get("worker_sid") == worker_sid and res.get("status") in HELD_RESERVATION_STATUSES
            for res in taskrouter.reservations(task_sid, timeout_s=left))
    except TaskRouterUnavailable:
        raise
    except Exception as exc:
        raise TaskRouterUnavailable(type(exc).__name__) from None


def _attr_equals(value: Any, expected: str) -> bool:
    return isinstance(value, str) and hmac.compare_digest(value.encode(), expected.encode())


# --------------------------------------------------------------------------- HTTP


def _header(event: Mapping[str, Any], name: str) -> str | None:
    for k, v in (event.get("headers") or {}).items():
        if isinstance(k, str) and k.lower() == name.lower() and isinstance(v, str):
            return v
    return None


def cors_headers(origin: str | None) -> dict[str, str]:
    headers = {"Vary": "Origin"}
    if origin == ALLOWED_ORIGIN:
        headers.update({
            "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
            "Access-Control-Allow-Methods": "GET,OPTIONS",
            "Access-Control-Allow-Headers": ALLOWED_HEADERS,
            "Access-Control-Max-Age": "600",
        })
    return headers


def _response(status: int, body: Mapping[str, Any] | None, origin: str | None) -> dict[str, Any]:
    headers = {
        **cors_headers(origin),
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    return {"statusCode": status, "headers": headers,
            "body": json.dumps(body) if body is not None else ""}


def _not_found(origin: str | None) -> dict[str, Any]:
    return _response(404, {"error": "not_found"}, origin)


def _withheld_text(withheld: Callable[[str], str] | None, language: str) -> str:
    if withheld is None:
        return WITHHELD_TURN
    try:
        text = withheld(language)
    except Exception as exc:
        obs.log_caught(log, "withheld text unavailable", exc, operation="withheld_text",
                       decision="fallback_builtin_text")
        return WITHHELD_TURN
    return text if isinstance(text, str) and text.strip() else WITHHELD_TURN


def _access(worker_sid: str, ref: str, call_sid: str, decision: str) -> None:
    """The access record: one line per read, identifiers and decision only."""
    log.info("context api: access", extra={"worker_sid": worker_sid, "journal_ref": ref,
                                           "call_sid": call_sid, "decision": decision})


def get_context(event: Mapping[str, Any], store: Store, *,
                now: Callable[[], float] = time.time, pii: PiiGuard | None = None,
                withheld: Callable[[str], str] | None = None,
                taskrouter: TaskRouterReader | None = None) -> dict[str, Any]:
    """Handle one API Gateway (REST, proxy) event. Never raises. ``pii``: the timeline's
    transcript redaction; ``withheld(language)``: the text for a withheld prompt.
    ``taskrouter``: the per-task check's reader. ``None`` means unconfigured and answers
    503 (fail closed), never data; :func:`handler` always passes a reader."""
    origin = _header(event, "Origin")
    try:
        method = event.get("httpMethod")
        if method == "OPTIONS":
            status = 204 if origin == ALLOWED_ORIGIN else 403
            return {"statusCode": status, "headers": cors_headers(origin), "body": ""}
        if method != "GET":
            return _response(405, {"error": "method_not_allowed"}, origin)

        authz = (event.get("requestContext") or {}).get("authorizer") or {}
        worker_sid = authz.get("worker_sid")
        if not isinstance(worker_sid, str) or not worker_sid:
            return _response(403, {"error": "forbidden"}, origin)

        ref = (event.get("pathParameters") or {}).get("journal_ref")
        call_sid = (event.get("queryStringParameters") or {}).get("call_sid")
        if not isinstance(ref, str) or not _REF_RE.match(ref):
            return _response(400, {"error": "bad_journal_ref"}, origin)
        if not isinstance(call_sid, str) or not _CALL_SID_RE.match(call_sid):
            return _response(400, {"error": "bad_call_sid"}, origin)

        state = callstate.load(store, call_sid, now=now())
        escalation = dict(state.escalation or {}) if state is not None else {}
        stored_ref = escalation.get("journal_ref")
        if state is None or not isinstance(stored_ref, str) or not hmac.compare_digest(
                stored_ref.encode(), ref.encode()):
            _access(worker_sid, ref, call_sid, "return_404_no_match")
            return _not_found(origin)

        if taskrouter is None:
            log.warning("context api: taskrouter not configured",
                        extra={"decision": "return_503"})
            _access(worker_sid, ref, call_sid, "return_503_taskrouter_unavailable")
            return _response(503, {"error": "unavailable"}, origin)
        task_sid = (event.get("queryStringParameters") or {}).get("task_sid")
        try:
            allowed = isinstance(task_sid, str) and bool(_TASK_SID_RE.match(task_sid)) \
                and authorize_task(worker_sid, parse_roles(authz.get("roles")), task_sid,
                                   call_sid, ref, taskrouter=taskrouter)
        except TaskRouterUnavailable as exc:
            log.warning("context api: taskrouter unavailable",
                        extra={"reason": str(exc)[:80], "decision": "return_503"})
            _access(worker_sid, ref, call_sid, "return_503_taskrouter_unavailable")
            return _response(503, {"error": "unavailable"}, origin)
        if not allowed:
            _access(worker_sid, ref, call_sid, "return_404_task_denied")
            return _not_found(origin)

        events = SegmentLog(store).read(call_sid)
        language = state.language
        timeline = build_timeline(events) if pii is None else build_timeline(
            events, pii=pii, language=language,
            withheld=lambda: _withheld_text(withheld, language or "en-US"))
        reason = escalation.get("reason")
        if not isinstance(reason, str):
            reason = next((e.get("reason") for e in reversed(timeline)
                           if e.get("type") == "escalation" and isinstance(e.get("reason"), str)),
                          None)
        _access(worker_sid, ref, call_sid, "served")
        return _response(200, {
            "journal_ref": ref,
            "call_sid": call_sid,
            "language": state.language,
            "verified_level": state.level,
            "escalation_reason": reason,
            "timeline": timeline,
        }, origin)
    except Exception as exc:  # never leak internals
        obs.log_caught(log, "context api: unexpected error", exc, level=logging.ERROR, decision="return_500")
        return _response(500, {"error": "internal"}, origin)


_store: Store | None = None
_pii: PiiGuard | None = None
_pii_store: Store | None = None
_withheld: CatalogWithheld | None = None
_withheld_store: Store | None = None


def _default_store() -> Store:
    global _store
    if _store is None:
        from owlbank.config import Settings
        from owlbank.store import DynamoStore

        _store = DynamoStore.from_settings(Settings.from_env())
    return _store


def _redactor() -> PiiRedactor:  # pragma: no cover - needs boto3 (the client is built on first use)
    from owlbank import coldstart
    from owlbank.pii import ComprehendPiiRedactor, make_comprehend_client

    return ComprehendPiiRedactor(coldstart.LazyClient(make_comprehend_client))


def _cache_ttl_s() -> int:
    import os

    ttl = (os.environ.get("OWLBANK_CACHE_TTL_S") or "60").strip()
    return int(ttl) if ttl.isdigit() else 60


def _default_withheld(store: Store) -> CatalogWithheld:
    """One catalog reader per process, on ``store`` (rebuilt if the store changes)."""
    global _withheld, _withheld_store
    if _withheld is None or _withheld_store is not store:
        _withheld, _withheld_store = CatalogWithheld(store, ttl_s=_cache_ttl_s()), store
    return _withheld


def withheld(language: str) -> str:
    """The withheld timeline text in ``language``: the catalog's ``escalation.summary_withheld``,
    or :data:`WITHHELD_TURN` on any catalog error."""
    try:
        reader = _default_withheld(_default_store())
    except Exception as exc:
        obs.log_caught(log, "withheld text: store unavailable", exc,
                       operation="withheld_text", decision="fallback_builtin_text")
        return WITHHELD_TURN
    return reader(language)


def _api_key_credentials() -> tuple[str, str]:
    from owlbank.config import Settings
    from owlbank.secrets import get_secret

    value = get_secret(Settings.from_env().secrets.twilio_api_key or "")
    return str(value["api_key"]), str(value["api_secret"])


def _default_taskrouter() -> TaskRouterReader:
    """The production reader. Missing configuration is not an error here: the reader raises
    :class:`TaskRouterUnavailable` on use, so the API answers 503 (fail closed)."""
    return TwilioTaskRouterReader(os.environ.get("OWLBANK_TASKROUTER_WORKSPACE_SID"),
                                  _api_key_credentials)


def _default_pii(store: Store) -> PiiGuard:
    """One guard per process, on the policy item of ``store`` (rebuilt if the store changes)."""
    global _pii, _pii_store
    if _pii is None or _pii_store is not store:
        _pii = redaction_guard(store, _redactor(), ttl_s=_cache_ttl_s())
        _pii_store = store
    return _pii


@obs.instrumented("context_api", op="get_context")
def handler(event: Mapping[str, Any], context: Any = None) -> dict[str, Any]:
    """Lambda entry point."""
    try:
        store = _default_store()
        pii = _default_pii(store)
    except Exception as exc:
        obs.log_caught(log, "context api: setup failed", exc, level=logging.ERROR,
                       operation="default_store", decision="return_500")
        return _response(500, {"error": "internal"}, _header(event, "Origin"))
    return get_context(event, store, pii=pii, withheld=_default_withheld(store),
                       taskrouter=_default_taskrouter())
