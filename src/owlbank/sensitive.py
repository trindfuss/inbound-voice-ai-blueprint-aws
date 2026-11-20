"""The sealed registry of a call's sensitive values.

The exact-value scrub (:func:`owlbank.segments.redact_text`, the log scrub) needs every
sensitive value of the call in every process that handles it: any front-door instance and the
bridge. Registration alone is per-process memory, so :func:`remember` also seals the value with the
existing :class:`~owlbank.token_seal.Sealer` (the key, the ``CallSid`` as encryption context)
and appends it to call state's ``sealed_sensitive``. Every process that loads the call passes that
field to :func:`restore`, which unseals the entries it hasn't seen yet (cached by blob) and
registers them. What the AI receives is never changed by any of this.

**Short values.** A value under :data:`~owlbank.segments.MIN_SENSITIVE_LEN`
characters is still sealed (so a later process knows the slot was answered) but not registered for
the scrub (it would match almost anything). ``sensitive_scrub_skipped`` (INFO, ``slot``,
``reason=too_short``) is logged once per slot per call in this process.

**Fail closed.** An entry that can't be unsealed marks the call *unverifiable* in this process
(:func:`is_unverifiable`): its journal surfaces then withhold instead of passing text. It is logged
with ``log_caught(..., decision="withhold_texts")``, once per failed try. A transient failure is
retried on later loads, at most :data:`UNSEAL_TRIES` tries per blob per call; a non-transient one
never. A later successful unseal does not clear *unverifiable* (it stays until :func:`forget`).

**Yes/no and choice answers.** A ``yes_no`` or ``choice`` value ("yes", "checking") is sealed
but never registered, here or on restore: scrubbing the word from every later log line would mangle
them, and the answering turn is already protected by :func:`placeholder_text`.
``sensitive_scrub_skipped`` is logged with ``reason=slot_type`` (once per slot per call in this
process).

Each entry seals a small JSON record, ``{"v": 1, "slot": <slot name>, "value": <value>}``, plus
``"type": <slot type>`` when :func:`remember` was given one, so a process that restores it knows
the slot and whether to register it. No value, record or blob is ever logged.

**The answering turn.** :func:`placeholder_text` is the text every surface gets
for a sensitive slot's answering turn (AI, journal, summary): the words that carry the answer become
``[<slot name>]``, the rest of what the caller said is kept.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from owlbank import callstate
from owlbank.callstate import F_SEALED_SENSITIVE, MAX_SEALED_SENSITIVE, CallState
from owlbank.obs import emit, log_caught
from owlbank.segments import clear_sensitive, contains_value, register_sensitive, value_spans
from owlbank.slots.normalize import number_runs
from owlbank.store import Store
from owlbank.token_seal import Sealer, UnsealError

if TYPE_CHECKING:
    from owlbank.slot_policy import SlotSettings

__all__ = ["remember", "restore", "recall", "is_unverifiable", "forget", "placeholder_text",
           "UNREGISTERED_SLOT_TYPES"]

log = logging.getLogger(__name__)

_RECORD_VERSION = 1

#: Slot types whose values are sealed but never registered for the exact-value scrub.
UNREGISTERED_SLOT_TYPES: frozenset[str] = frozenset({"yes_no", "choice"})


@dataclass
class _ProcessState:
    """What this process knows about each call's sealed values (memory only, never stored)."""

    #: call -> blobs already handled (unsealed, failed, or sealed here).
    seen: dict[str, set[str]] = field(default_factory=dict)
    #: call -> (slot, value) entries known to be sealed in the call state.
    entries: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    #: call -> blob -> (slot, value), for the blobs sealed or unsealed here (:func:`recall`).
    by_blob: dict[str, dict[str, tuple[str, str]]] = field(default_factory=dict)
    #: call -> slots whose ``sensitive_scrub_skipped`` line was logged.
    skipped: dict[str, set[str]] = field(default_factory=dict)
    #: calls with an entry that couldn't be unsealed.
    unverifiable: set[str] = field(default_factory=set)
    #: call -> blob -> failed unseal tries so far (transient failures; :data:`UNSEAL_TRIES`).
    tries: dict[str, dict[str, int]] = field(default_factory=dict)


_state = _ProcessState()
_lock = threading.Lock()

#: Unseal tries per blob per call in this process (the first and 2 retries, on later loads): a
#: blob that keeps failing is not retried on every webhook (log flood, KMS timeouts past Twilio's
#: 15 s limit).
UNSEAL_TRIES = 3


class _Full(Exception):
    """``sealed_sensitive`` already holds :data:`MAX_SEALED_SENSITIVE` entries."""


def _record(slot: str, value: str, slot_type: str | None) -> str:
    doc: dict[str, Any] = {"v": _RECORD_VERSION, "slot": slot}
    if slot_type is not None:
        doc["type"] = slot_type
    doc["value"] = value
    return json.dumps(doc, separators=(",", ":"))


def _parse(text: str) -> tuple[str, str | None, str]:
    """``(slot, slot type, value)`` of a sealed record. ``ValueError`` (naming nothing) if it
    isn't one."""
    try:
        doc = json.loads(text)
    except ValueError:
        raise ValueError("sealed entry is not a registry record") from None
    if not isinstance(doc, Mapping) or doc.get("v") != _RECORD_VERSION:
        raise ValueError("sealed entry is not a registry record")
    slot, slot_type, value = doc.get("slot"), doc.get("type"), doc.get("value")
    if not isinstance(slot, str) or not slot or not isinstance(value, str) or not value \
            or not (slot_type is None or (isinstance(slot_type, str) and slot_type)):
        raise ValueError("sealed entry is not a registry record")
    return slot, slot_type, value


def _skip(call_sid: str, slot: str, reason: str) -> None:
    """Log ``sensitive_scrub_skipped`` once per (call, slot) in this process."""
    with _lock:
        slots = _state.skipped.setdefault(call_sid, set())
        first = slot not in slots
        slots.add(slot)
    if first:
        emit(log, logging.INFO, "sensitive_scrub_skipped", slot=slot, reason=reason,
             decision="sealed_not_registered")


def _register(call_sid: str, slot: str, slot_type: str | None, value: str) -> None:
    """Register ``value`` for the scrub, or log why not once per (call, slot): a ``yes_no`` or
    ``choice`` slot, or a value under ``MIN_SENSITIVE_LEN``."""
    if slot_type in UNREGISTERED_SLOT_TYPES:
        _skip(call_sid, slot, "slot_type")
        return
    try:
        register_sensitive(call_sid, [value])
    except ValueError:  # under MIN_SENSITIVE_LEN (the only ValueError for a string)
        _skip(call_sid, slot, "too_short")


def remember(store: Store, sealer: Sealer, call_sid: str, value: str, *, slot: str,
             slot_type: str | None = None, now: Callable[[], float]) -> bool:
    """Register ``value`` (a sensitive slot's accepted value) in this process, seal it with the
    ``call_sid`` context and append it to call state's ``sealed_sensitive`` (``callstate.mutate``).

    ``slot_type`` is the slot's type (``SlotSettings.decl.slot_type``): a ``yes_no`` or
    ``choice`` value is sealed but not registered, here or by :func:`restore`. Without it the
    value is registered (as for a ``digits`` slot).

    Returns ``True`` once the value is in the call state (also when this process already knew it
    was), ``False`` when the field is full: then an ERROR naming the field is logged, nothing is
    appended and the value stays registered in this process only. Errors of the sealer and the
    store (``KeyError`` for a call with no state, ``VersionConflict``) propagate; the value is
    registered here before either is called."""
    if not isinstance(value, str) or not value:
        raise TypeError("a sensitive value must be a non-empty string")
    if not isinstance(slot, str) or not slot:
        raise TypeError("slot must be a non-empty string")
    if slot_type is not None and (not isinstance(slot_type, str) or not slot_type):
        raise TypeError("slot_type must be a non-empty string or None")
    _register(call_sid, slot, slot_type, value)
    with _lock:
        if (slot, value) in _state.entries.get(call_sid, ()):
            return True
    blob = sealer.seal(_record(slot, value, slot_type), call_sid=call_sid)

    def append(state: CallState) -> Mapping[str, Any]:
        if len(state.sealed_sensitive) >= MAX_SEALED_SENSITIVE:
            raise _Full
        return {F_SEALED_SENSITIVE: (*state.sealed_sensitive, blob)}

    try:
        callstate.mutate(store, call_sid, append, now=now)
    except _Full:
        emit(log, logging.ERROR, "sensitive_seal_refused", field=F_SEALED_SENSITIVE, slot=slot,
             reason="limit", limit=MAX_SEALED_SENSITIVE, decision="registered_in_process_only")
        return False
    with _lock:
        _state.seen.setdefault(call_sid, set()).add(blob)
        _state.entries.setdefault(call_sid, set()).add((slot, value))
        _state.by_blob.setdefault(call_sid, {})[blob] = (slot, value)
    return True


def _transient(exc: Exception) -> bool:
    """Whether a retry of a failed unseal may succeed: an :class:`UnsealError` says so itself
    (``transient``: KMS throttled, timed out or unreachable); a record that isn't one
    (``ValueError`` from :func:`_parse`) never does; any other sealer exception may."""
    if isinstance(exc, UnsealError):
        return exc.transient
    return not isinstance(exc, ValueError)


def restore(call_sid: str, sealed: Sequence[str], sealer: Sealer) -> None:
    """Unseal the entries of ``sealed`` (call state's ``sealed_sensitive``) not seen yet in this
    process and register them. Never raises: an entry that doesn't unseal (or isn't a registry
    record) is logged with ``decision="withhold_texts"`` (one line per failed try) and marks the
    call unverifiable.

    A transient failure (a KMS blip) is retried on later calls, so it doesn't leave the value
    unknown, and :func:`recall` empty, for the rest of the call on this instance; but a blob is
    tried at most :data:`UNSEAL_TRIES` times per call in this process. A non-transient failure (bad
    ciphertext, another call's context, not a registry record) is never retried. Either way, once
    marked, the call stays unverifiable until :func:`forget`: a later successful unseal doesn't
    clear it (fail closed: surfaces written while the value was unknown can't be re-checked)."""
    for blob in sealed:
        with _lock:
            if blob in _state.seen.get(call_sid, ()):
                continue
        try:
            slot, slot_type, value = _parse(sealer.unseal(blob, call_sid=call_sid))
        except Exception as exc:  # UnsealError, KMS errors, a malformed record: fail closed
            transient = _transient(exc)
            with _lock:
                _state.unverifiable.add(call_sid)
                tries = _state.tries.setdefault(call_sid, {})
                tries[blob] = tries.get(blob, 0) + 1
                attempt = tries[blob]
                retry = transient and attempt < UNSEAL_TRIES
                if not retry:  # given up on: handled, never tried again for this call here
                    _state.seen.setdefault(call_sid, set()).add(blob)
                    tries.pop(blob, None)
            log_caught(log, "sensitive_unseal_failed", exc, operation="restore_sensitive",
                       decision="withhold_texts", attempt=attempt, transient=transient,
                       will_retry=retry)
            continue
        with _lock:
            _state.tries.get(call_sid, {}).pop(blob, None)
            _state.seen.setdefault(call_sid, set()).add(blob)
            _state.entries.setdefault(call_sid, set()).add((slot, value))
            _state.by_blob.setdefault(call_sid, {})[blob] = (slot, value)
        _register(call_sid, slot, slot_type, value)


def recall(call_sid: str, sealed: Sequence[str], *, slot: str) -> str | None:
    """The value sealed for ``slot`` among ``sealed`` (the call state's ``sealed_sensitive``, after
    :func:`restore` ran over it in this process), or ``None`` when there is none, or more than one
    value (fail closed). Only blobs of this call state count, never what this process remembers of
    them otherwise. Never unseals and never logs."""
    with _lock:
        known = _state.by_blob.get(call_sid, {})
        values = {known[b][1] for b in sealed if b in known and known[b][0] == slot}
    return next(iter(values)) if len(values) == 1 else None


def is_unverifiable(call_sid: str) -> bool:
    """True when an entry of the call's ``sealed_sensitive`` couldn't be unsealed in this process:
    its journal surfaces must withhold text. It stays true after a later retry unseals the
    entry (fail closed), until :func:`forget`."""
    with _lock:
        return call_sid in _state.unverifiable


def forget(call_sid: str) -> None:
    """Drop everything this process knows about the call's sensitive values (at call end),
    including the registered values (:func:`owlbank.segments.clear_sensitive`)."""
    with _lock:
        _state.seen.pop(call_sid, None)
        _state.entries.pop(call_sid, None)
        _state.by_blob.pop(call_sid, None)
        _state.skipped.pop(call_sid, None)
        _state.unverifiable.discard(call_sid)
        _state.tries.pop(call_sid, None)
    clear_sensitive(call_sid)


# --------------------------------------------------------------------------- placeholder

#: The placeholder rule reads number words of both languages, whatever the call's language.
_PLACEHOLDER_LANGS = ("en", "es")
#: What may stand between two tokens that collapse into one: whitespace or punctuation only.
_GAP_RE = re.compile(r"[\W_]*")
_SPAN_TYPES = frozenset({"yes_no", "choice"})


def placeholder_text(speech: str, slot: SlotSettings, value: str | None, *,
                     value_runs: Sequence[int] | None = None,
                     span: tuple[int, int] | None = None, every_run: bool = False) -> str:
    """The answering turn's text for a sensitive slot:
    ``speech`` with the words that carry the answer replaced by ``[<slot name>]`` (the name as
    configured, e.g. ``[ssn4]``). The caller applies it to a sensitive slot only; it doesn't look
    at ``slot.sensitive``.

    ``speech`` is NFC-normalized first and the result is NFC. ``value_runs`` are indices into
    ``number_runs(nfc_speech, ("en", "es"))``, and ``span`` is ``(start, end)`` in the NFC speech.

    1. A number run becomes the token if it has 2 or more number tokens (a digit token of k digits
       counts as k) or reads as 2 or more digits ("fifteen": no run of 2 or more digits survives), or its
       index is in ``value_runs`` (it supplied digits of the value), or its digits
       can't be read (fail closed).
    2. Fallback, when ``value`` is set but ``value_runs`` is ``None`` (a keypad value with speech):
       every one-token run whose digit occurs in ``value`` becomes the token too.
    3. ``yes_no`` and ``choice``: ``span`` (the matched phrase) becomes the token; with a value but
       no span (a resolver value) the whole utterance does.
    4. The value itself, wherever it occurs (case, spaces and hyphens ignored, on word boundaries;
       :func:`owlbank.segments.contains_value`). If it still matches after the replacements (an
       NFKC form), the whole utterance becomes the token.

    5. ``every_run``: every number run becomes the token, one-token runs too (a value not known yet,
       said as single digits split by filler: "one and then two and then three four").

    Adjacent tokens (only whitespace or punctuation between them) collapse to one. Runs out of
    range in ``value_runs`` are ignored; a ``span`` is clamped to the text.
    """
    text = unicodedata.normalize("NFC", speech)
    token = f"[{slot.name}]"
    slot_type = slot.decl.slot_type
    if slot_type in _SPAN_TYPES and value is not None and span is None:
        return token
    located = frozenset(value_runs) if value_runs is not None else None
    value_digits = frozenset(c for c in value if c.isdigit()) if value else frozenset()
    cut: list[tuple[int, int]] = []
    for i, run in enumerate(number_runs(text, _PLACEHOLDER_LANGS)):
        digits = run.digits
        if (every_run or run.tokens >= 2 or digits is None or len(digits) >= 2
                or (located is not None and i in located)
                or (located is None and value is not None and digits in value_digits)):
            cut.append((run.start, run.end))
    if span is not None and slot_type in _SPAN_TYPES:
        start, end = max(0, span[0]), min(len(text), span[1])
        if start < end:
            cut.append((start, end))
    if value:
        cut.extend(value_spans(text, value))
    out = _render(text, cut, token)
    if value and contains_value(out.replace(token, " "), value):  # the token isn't the caller's
        return token
    return out


def _render(text: str, cut: list[tuple[int, int]], token: str) -> str:
    """``text`` with each interval of ``cut`` (merged where they overlap or touch) replaced by
    ``token``, and tokens separated only by whitespace or punctuation collapsed to one."""
    merged: list[list[int]] = []
    for start, end in sorted(cut):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out: list[str] = []
    pos = 0
    for start, end in merged:
        gap = text[pos:start]
        if out and _GAP_RE.fullmatch(gap):
            pass  # adjacent to the previous token: collapse (the gap goes with it)
        else:
            out.append(gap)
            out.append(token)
        pos = end
    out.append(text[pos:])
    return "".join(out)
