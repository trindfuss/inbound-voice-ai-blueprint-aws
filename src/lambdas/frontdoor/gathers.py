"""One answer per Gather: the claim item next to the ``#session`` sidecar.

A ``<Gather>`` with early finish on (a digit factor's, or a choice question's) carries an opaque id (``g``) in its ``action``
and ``partialResultCallback`` URLs. Whatever reaches the answer first (an early-finish partial,
the redirect it causes, or Twilio's final ``action`` webhook) moves this item through its states
with conditional writes, so exactly one request verifies the answer and every other request for
the same Gather (later partials, the final webhook after an early finish, a late redirect,
Twilio's retries) gets the TwiML that one produced.

Item ``calls["<CallSid>#gather#<id>"]`` (``#`` keeps it apart from ``CallState`` and
``#session``; same TTL as the call):

* ``state``: ``early`` (a partial held a complete answer; it is sealed here and the REST update is
  in flight), ``released`` (that update failed: the Gather finishes as today),
  ``processing`` (one request owns the answer), ``done`` (its TwiML is stored).
* ``sealed_value``: the early answer (the digits, or a choice's option value), sealed with the
  call's ``Sealer`` (``CallSid`` context), readable for :data:`DIGITS_TTL_S` only (until
  ``digits_expires_at``); dropped when the answer is taken.
* ``sealed_twiml``: the owner's response, compressed and sealed in chunks (it can hold a
  ConversationRelay nonce, which is never stored in the clear).

Transitions (each a compare-and-set on ``state``; ``none`` = no item)::

    none -> early (partial)          early -> released (update failed)
    none|early|released -> processing (final webhook)
    early|released -> processing (early redirect)
    processing -> done

Never stored here: plaintext digits, option values or caller speech.

Call observability, only with front-door telemetry on: each partial adds to
``partial_count`` and, when the partial's length differs from ``last_len`` (or none is stored),
sets ``last_len`` and ``last_change_ms`` (:func:`count_partial`). The length only, never the
text or a hash of it. An item made by a partial has no ``state`` (it reads as ``none``) and the
call's ``expires_at``; the transitions above keep these fields (:func:`take` copies the item,
the early claim in ``app.py`` carries them over with :func:`partial_counters`).

Phone settle, the phone slot's Gathers only: a live partial with a valid
``SequenceNumber`` higher than the stored ``settle_seq`` sets ``settle_seq`` to it and
``settle_id`` to a fresh random id (:func:`mark_partial`; one conditional write, no read, never the
text); an older one, delivered late, writes nothing and never claims. A partial without a valid
``SequenceNumber`` only replaces ``settle_id`` (and counts ``settle_unseq``), so no partial before
it can claim, and never claims itself. A partial with a complete reading waits ``settle_ms`` and claims only while
``settle_id`` is still its own (:func:`cas` with ``settle_id``: one condition that also means the
state is ``none``, since every transition out of ``none`` drops the settle fields: :func:`take`
drops them and the early claim writes a fresh item).
"""

from __future__ import annotations

import base64
import zlib
from collections.abc import Collection, Mapping
from typing import Any

from owlbank.segments import assert_clean
from owlbank.store import ConditionFailed, Expect, Store, TxPut
from owlbank.token_seal import Sealer

TABLE = "calls"
INFIX = "#gather#"
NONE, EARLY, RELEASED, PROCESSING, DONE = "none", "early", "released", "processing", "done"
#: How long a sealed early answer may be read (60 s).
DIGITS_TTL_S = 60
#: The partial-result counters, kept across every transition.
PARTIAL_FIELDS = ("partial_count", "last_len", "last_change_ms")
#: The phone settle fields: dropped by every transition out of ``none``.
SETTLE_FIELDS = ("settle_seq", "settle_id", "settle_unseq")
#: Sealed chunk size (KMS ``Encrypt`` takes at most 4096 bytes of plaintext).
_CHUNK = 3000


def key(call_sid: str, gather_id: str) -> str:
    return f"{call_sid}{INFIX}{gather_id}"


def load(store: Store, call_sid: str, gather_id: str) -> dict[str, Any] | None:
    return store.get(TABLE, {"call_sid": key(call_sid, gather_id)})


def state_of(item: Mapping[str, Any] | None) -> str:
    return NONE if item is None else str(item.get("state") or NONE)


def cas(store: Store, call_sid: str, gather_id: str, expected: str,
        item: Mapping[str, Any], *, settle_id: str | None = None) -> bool:
    """Write ``item`` (whole) only if the stored state is still ``expected``. With ``settle_id``
    (``expected`` must be ``none``): only if the stored ``settle_id`` is still that one, which
    also means the state is still ``none`` (module docstring)."""
    full = {**item, "call_sid": key(call_sid, gather_id)}
    assert_clean(call_sid, full, deny_keys=False)
    if settle_id is not None and expected != NONE:
        raise ValueError("a settle claim is made from state none only")
    condition = (Expect("settle_id", settle_id, must_exist=True) if settle_id is not None
                 else Expect("state", expected, absent_as=NONE))
    try:
        store.transact_write([TxPut(TABLE, full, condition)])
    except ConditionFailed:
        return False
    return True


def take(store: Store, call_sid: str, gather_id: str, *, allowed: Collection[str],
         changes: Mapping[str, Any], drop: Collection[str] = (),
         defaults: Mapping[str, Any] | None = None) -> tuple[bool, dict[str, Any] | None]:
    """Move the item from any state in ``allowed`` with ``changes`` applied (and ``drop`` keys
    removed), retrying on a concurrent change. Returns ``(True, previous item)`` on success,
    ``(False, current item)`` when the state is (or became) one not in ``allowed``."""
    for _ in range(5):
        current = load(store, call_sid, gather_id)
        state = state_of(current)
        if state not in allowed:
            return False, current
        base = {k: v for k, v in (current or dict(defaults or {})).items()
                if k not in drop and k not in SETTLE_FIELDS}
        if cas(store, call_sid, gather_id, state, {**base, **changes}):
            return True, current
    return False, load(store, call_sid, gather_id)


def count_partial(store: Store, call_sid: str, gather_id: str, *, length: int, now_ms: int,
                  expires_at: int) -> None:
    """One partial result: ``partial_count + 1`` and, if ``length`` differs from the stored
    ``last_len``, ``last_len = length`` and ``last_change_ms = now_ms``. One write, no read; the
    item is created (with ``expires_at``) if this is the Gather's first."""
    store.increment(TABLE, {"call_sid": key(call_sid, gather_id)}, "partial_count",
                    defaults={"expires_at": expires_at},
                    changes={"last_len": length, "last_change_ms": now_ms},
                    unless_equal=("last_len", length))


def mark_partial(store: Store, call_sid: str, gather_id: str, *, settle_id: str,
                 seq: int | None, expires_at: int) -> bool:
    """Record a partial for this phone Gather. With its ``SequenceNumber`` (``seq``):
    ``settle_seq = seq`` and ``settle_id = settle_id`` only if ``seq`` is higher than the stored
    ``settle_seq`` (an older partial delivered late changes nothing); ``True`` when written, that
    is when this partial is the Gather's newest. Without one (``None``): ``settle_id`` is replaced
    (``settle_unseq`` + 1), so no earlier partial can claim; ``False`` (it can't claim either).
    One write, no read; the item is created (with ``expires_at``) if this is the Gather's first."""
    k = {"call_sid": key(call_sid, gather_id)}
    if seq is None:
        store.increment(TABLE, k, "settle_unseq", defaults={"expires_at": expires_at},
                        changes={"settle_id": settle_id})
        return False
    return store.set_if_greater(TABLE, k, "settle_seq", seq, defaults={"expires_at": expires_at},
                                changes={"settle_id": settle_id})


def settle_seq_of(item: Mapping[str, Any] | None) -> int | None:
    value = (item or {}).get("settle_seq")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def settle_id_of(item: Mapping[str, Any] | None) -> str | None:
    value = (item or {}).get("settle_id")
    return value if isinstance(value, str) else None


def partial_counters(item: Mapping[str, Any] | None) -> dict[str, int]:
    """The partial-result counters of a claim item (those present and integers)."""
    return {k: v for k in PARTIAL_FIELDS
            if isinstance(v := (item or {}).get(k), int) and not isinstance(v, bool)}


def seal_twiml(sealer: Sealer, xml: str, *, call_sid: str) -> list[str]:
    packed = base64.b64encode(zlib.compress(xml.encode("utf-8"))).decode("ascii")
    return [sealer.seal(packed[i:i + _CHUNK], call_sid=call_sid)
            for i in range(0, len(packed), _CHUNK)] or [sealer.seal("", call_sid=call_sid)]


def unseal_twiml(sealer: Sealer, chunks: list[str], *, call_sid: str) -> str:
    packed = "".join(sealer.unseal(c, call_sid=call_sid) for c in chunks)
    return zlib.decompress(base64.b64decode(packed)).decode("utf-8")
