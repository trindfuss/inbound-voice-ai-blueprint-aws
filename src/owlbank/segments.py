"""The call journal: an append-only segment log per call, with a redaction guard.

The ``segments`` table (``call_sid`` PK, ``seq`` SK). Every
handler appends events here; none rewrites. The log is the source the bridge
uses to rebuild context on resume and the server-side facts for
escalation task attributes.

**Redaction guard.** :func:`assert_clean` rejects a payload if

* any key (at any depth) is on the deny-list (``digits``, ``speech_result``,
  ``factor_value``, ``ssn``, ``zip``, ``otp``, ``token``; matched case-insensitively,
  ignoring ``_``/``-``, and as a prefix, so ``ssn4``, ``zip_code``, ``token_hash``,
  ``SpeechResult`` and ``Digits`` are all denied), or
* any string (or integer) value contains a value registered for this call with
  :func:`register_sensitive`. Matching ignores case, whitespace and hyphens, so
  ``"1 2-3 4 5"`` matches a registered ``"12345"`` (NFKC-normalized first, so
  full-width digits match too). A value matches on word boundaries only: a letter or
  digit right before or after it means no match, so a registered ``"lang"`` doesn't
  match ``"language"`` and ``"1234"`` doesn't match ``"12345"``. Key names are never
  scanned for values; the deny-list above applies to
  keys.

Values in :data:`ID_KEYS` (server-generated identifiers, timestamps, ANI) are not
scanned: they never carry caller input, and random hex/phone digits would otherwise
collide with short factors. Registered values live only in this process's memory
and are never stored. :class:`RedactionError` messages name the path only, never
the value.
"""

from __future__ import annotations

import functools
import logging
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from owlbank.obs import emit
from owlbank.store import ConditionFailed, Store

log = logging.getLogger(__name__)

#: Event types the journal accepts.
EVENT_TYPES: frozenset[str] = frozenset({
    "call_start", "lookup", "prompt", "factor_asked", "verification", "ai_turn",
    "tool_call", "tool_result", "handoff", "escalation", "call_end",
    "confirm_asked", "confirm_answer",  # deterministic confirmation
    "slot_resolution",  # how one slot answer resolved
    "prompt_repeat",  # the bridge repeated the last message after caller silence
    "keyed_lookup",  # Outcome and count only, never the number
    # Human leg: the Orchestrator conversation was closed, once per call. Fields:
    # ``conversation_id`` and ``closed_by`` (``status``, ``transcription`` or ``sweep``); no content.
    "conversation_closed",
})

#: Key names that may never appear in a journal, call-state or task-attribute payload.
DENY_KEYS: tuple[str, ...] = ("digits", "speech_result", "factor_value", "ssn", "zip", "otp", "token")

#: Keys whose values are server-generated identifiers or numbers and are not scanned
#: for registered factor values (see module docstring). ``at`` is the server's own time on
#: call-state records (``escalation.at``, an ISO string from the front door; ``otp_send.at``,
#: ``welcome_prepared.at``, ``prefetched.at`` and the bridge's ``escalation.at``, epoch seconds).
#: Scanned, an SSN last four equal to the year or the month and day ("2026", "1102" in
#: "2026-11-02T...") blocked the escalation write and every later call-state write.
ID_KEYS: frozenset[str] = frozenset({
    "call_sid", "seq", "ts", "expires_at", "version", "tool_use_id",
    "conversation_id", "profile_id", "journal_ref", "customer_id", "verified_customer_id",
    "candidate_ids", "ani", "from", "escalated_at", "at", "level", "auth_level", "since_seq",
    "ws_nonce_hash",
})

#: Registered values shorter than this are refused: they would match almost anything.
MIN_SENSITIVE_LEN = 4

_RESERVED_FIELDS = frozenset({"call_sid", "seq", "ts", "expires_at"})
_SEPARATORS = re.compile(r"[\s\-]+")


class RedactionError(ValueError):
    """A payload contained a denied key or a registered factor value."""


class AppendConflict(RuntimeError):
    """The sequence number could not be allocated within the retry budget."""


# --------------------------------------------------------------------------- guard

_SENSITIVE: dict[str, set[str]] = {}


def _norm_key(key: str) -> str:
    return key.casefold().replace("_", "").replace("-", "")


_DENY_NORM = tuple(_norm_key(k) for k in DENY_KEYS)


def _norm_text(text: str) -> str:
    return _SEPARATORS.sub("", unicodedata.normalize("NFKC", text)).casefold()


def norm_text(text: str) -> str:
    """``text`` normalized as the registry stores a value (NFKC, whitespace and hyphens removed,
    case-folded): the public name of the rule :func:`register_sensitive` applies."""
    return _norm_text(text)


def register_sensitive(call_sid: str, values: Iterable[str]) -> None:
    """Remember factor values collected on ``call_sid`` (memory only, never stored)."""
    bucket = _SENSITIVE.setdefault(call_sid, set())
    for value in values:
        if not isinstance(value, str):
            raise TypeError("sensitive values must be strings")
        norm = _norm_text(value)
        if len(norm) < MIN_SENSITIVE_LEN:
            raise ValueError(f"sensitive values must have at least {MIN_SENSITIVE_LEN} characters")
        bucket.add(norm)


def clear_sensitive(call_sid: str) -> None:
    """Forget the registered values for a call (at call end)."""
    _SENSITIVE.pop(call_sid, None)


def registered_calls() -> tuple[str, ...]:
    """The call SIDs that currently have registered values (a snapshot; calls register and
    forget concurrently)."""
    return tuple(_SENSITIVE)


def sensitive_values(call_sid: str) -> frozenset[str]:
    """The normalized registered values for a call (for redaction helpers)."""
    return frozenset(_SENSITIVE.get(call_sid, ()))


def is_denied_key(key: str) -> bool:
    norm = _norm_key(key)
    return any(norm.startswith(d) for d in _DENY_NORM)


def _is_word_char(ch: str) -> bool:
    return ch.isalnum()


@functools.lru_cache(maxsize=4096)
def _value_pattern(norm: str) -> re.Pattern[str]:
    """The pattern of a normalized value (:func:`_norm_text`): its characters with any whitespace
    or hyphens between them, any case, on word boundaries. A boundary is required only on a
    side where the value's own character is a letter or digit (as ``\\b`` would), and ``_`` counts
    as a boundary. An all-digit value (an SSN last-4, a code, a ZIP, an account or phone number)
    needs only a non-digit on each side, so ``acct1234`` and ``ssn-1234`` still match ``1234`` but
    ``12345`` doesn't."""
    body = r"[\s\-]*".join(re.escape(ch) for ch in norm)
    if norm.isdigit():
        return re.compile(r"(?<!\d)" + body + r"(?!\d)", re.IGNORECASE)
    before = r"(?<![^\W_])" if _is_word_char(norm[0]) else ""
    after = r"(?![^\W_])" if _is_word_char(norm[-1]) else ""
    return re.compile(before + body + after, re.IGNORECASE)


def _contains_norm(norm: str, text: str) -> bool:
    """True if the normalized value ``norm`` occurs in ``text`` (NFKC, case-folded) on word
    boundaries."""
    return bool(norm) and _value_pattern(norm).search(
        unicodedata.normalize("NFKC", text).casefold()) is not None


def contains_value(text: str, value: str) -> bool:
    """True if ``value`` occurs in ``text`` the way the guard matches a registered value (case,
    whitespace and hyphens ignored, NFKC, word boundaries), with no registry and no minimum length
    (the sensitive placeholder's rule 4)."""
    return _contains_norm(_norm_text(value), text)


def value_spans(text: str, value: str) -> list[tuple[int, int]]:
    """``(start, end)`` of each match of ``value`` in ``text`` as given (not normalized), matched
    as :func:`contains_value` does apart from NFKC (callers that need it check the result with
    :func:`contains_value`)."""
    norm = _norm_text(value)
    if not norm:
        return []
    return [m.span() for m in _value_pattern(norm).finditer(text)]


def contains_sensitive(call_sid: str, text: str) -> bool:
    """True if ``text`` contains a value registered for ``call_sid`` (on word boundaries)."""
    values = _SENSITIVE.get(call_sid)
    if not values:
        return False
    return any(_contains_norm(v, text) for v in values)


def redact_text(call_sid: str, text: str, replacement: str = "[redacted]",
                withheld: str = "[redacted]") -> str:
    """Replace every registered value in ``text``, tolerating case, spaces and hyphens, on word
    boundaries.

    Text is NFKC-normalized first. If a match survives the substitution (for
    example through a case-folding expansion), the whole text becomes ``withheld``.
    """
    return redact_values(_SENSITIVE.get(call_sid, ()), text, replacement, withheld)


def redact_values(values: Iterable[str], text: str, replacement: str = "[redacted]",
                  withheld: str = "[redacted]") -> str:
    """:func:`redact_text` with the given normalized values (as :func:`sensitive_values` returns
    them) in place of a call's registered ones: for a holder that keeps a snapshot of a call's
    values after the call forgot them (the bridge's trace redaction)."""
    norms = sorted(set(values), key=len, reverse=True)
    text = unicodedata.normalize("NFKC", text)
    for value in norms:
        text = _value_pattern(value).sub(replacement, text)
    if any(_contains_norm(v, text) for v in norms):
        return withheld
    return text


def assert_clean(call_sid: str, payload: Any, *, deny_keys: bool = True, path: str = "$") -> None:
    """Raise :class:`RedactionError` if ``payload`` leaks a factor (see module docstring)."""

    def walk(value: Any, where: str, scan: bool) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                key = str(k)
                if deny_keys and is_denied_key(key):
                    raise RedactionError(f"{where}.{key}: denied key")
                # Key names aren't scanned for values: a registered "lang" mustn't block
                # every write holding the key "language". Keys are ours, not the caller's words.
                walk(v, f"{where}.{key}", scan and key not in ID_KEYS)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for i, v in enumerate(value):
                walk(v, f"{where}[{i}]", scan)
        elif isinstance(value, bool) or value is None:
            return
        elif isinstance(value, (str, int)) and scan and contains_sensitive(call_sid, str(value)):
            raise RedactionError(f"{where}: contains a factor value")

    walk(payload, path, True)


# --------------------------------------------------------------------------- log


class SegmentLog:
    """Append-only journal over the ``segments`` table.

    ``now`` returns epoch seconds (injectable for tests); records expire after
    ``ttl_s`` (``Settings.record_ttl_s``).
    """

    def __init__(self, store: Store, *, now: Callable[[], float] = time.time,
                 ttl_s: int = 86400, max_retries: int = 20) -> None:
        self._store = store
        self._now = now
        self._ttl_s = ttl_s
        self._max_retries = max_retries
        self._next_seq: dict[str, int] = {}

    register_sensitive = staticmethod(register_sensitive)
    clear_sensitive = staticmethod(clear_sensitive)

    def _last_seq(self, call_sid: str) -> int:
        items = self._store.query("segments", {"call_sid": call_sid})
        return max((int(i["seq"]) for i in items), default=0)

    def append(self, call_sid: str, event: Mapping[str, Any]) -> int:
        """Append ``event`` (must have a known ``type``); return its ``seq`` (1-based).

        The sequence is allocated with a ``not_exists`` conditional put and retried
        on conflict, so concurrent writers never overwrite each other.
        """
        if not call_sid:
            raise ValueError("call_sid is required")
        if not isinstance(event, Mapping):
            raise TypeError("event must be a mapping")
        etype = event.get("type")
        if etype not in EVENT_TYPES:
            raise ValueError(f"unknown event type {etype!r}")
        clash = _RESERVED_FIELDS & set(event)
        if clash:
            raise ValueError(f"event may not set {sorted(clash)}")
        assert_clean(call_sid, event)

        now = int(self._now())
        seq = self._next_seq.get(call_sid) or self._last_seq(call_sid) + 1
        for _ in range(self._max_retries):
            item = {**event, "call_sid": call_sid, "seq": seq, "ts": now,
                    "expires_at": now + self._ttl_s}
            try:
                self._store.put("segments", item, condition="not_exists")
            except ConditionFailed:
                emit(log, logging.DEBUG, "segment_seq_taken", operation="append", decision="retry")
                seq = max(seq + 1, self._last_seq(call_sid) + 1)
                continue
            self._next_seq[call_sid] = seq + 1
            return seq
        emit(log, logging.WARNING, "segment_append_exhausted", operation="append",
             retries=self._max_retries, decision="raise_append_conflict")
        raise AppendConflict(f"could not allocate a sequence number for {call_sid}")

    def read(self, call_sid: str, *, since_seq: int = 0) -> list[dict[str, Any]]:
        """All events for the call with ``seq > since_seq``, in order."""
        items = self._store.query("segments", {"call_sid": call_sid})
        return [i for i in items if int(i["seq"]) > since_seq]

    def summary_for_resume(self, call_sid: str, since_seq: int = 0) -> dict[str, Any]:
        """Structured summary of the events after ``since_seq`` (resume).

        Returns ``promptsPlayed`` (catalog keys, in order), ``factorsAsked`` (names),
        ``attempts`` (verification attempts), ``outcome`` (the last verification or
        handoff/tool outcome, or ``None``), ``levelReached``, ``toolsCalled``,
        ``lastTool``, ``escalation`` (last escalation event's reason/queue) and
        ``lastSeq``. Only server-recorded, already-redacted fields are copied.
        """
        events = self.read(call_sid, since_seq=since_seq)
        prompts: list[str] = []
        factors: list[str] = []
        tools: list[str] = []
        attempts = 0
        outcome: str | None = None
        level: int | None = None
        escalation: dict[str, Any] | None = None
        for ev in events:
            etype = ev.get("type")
            if etype == "prompt" and ev.get("prompt_key"):
                prompts.append(str(ev["prompt_key"]))
            elif etype == "factor_asked" and ev.get("factor"):
                factors.append(str(ev["factor"]))
            elif etype == "verification":
                attempts += 1
                outcome = ev.get("outcome", outcome)
                if isinstance(ev.get("level"), int):
                    level = ev["level"]
            elif etype == "tool_call" and ev.get("tool"):
                tools.append(str(ev["tool"]))
            elif etype in ("tool_result", "handoff") and ev.get("outcome") is not None:
                outcome = ev["outcome"]
            elif etype == "escalation":
                escalation = {k: ev[k] for k in ("reason", "queue") if k in ev}
        return {
            "promptsPlayed": prompts,
            "factorsAsked": factors,
            "attempts": attempts,
            "outcome": outcome,
            "levelReached": level,
            "toolsCalled": tools,
            "lastTool": tools[-1] if tools else None,
            "escalation": escalation,
            "lastSeq": max((int(e["seq"]) for e in events), default=since_seq),
        }
