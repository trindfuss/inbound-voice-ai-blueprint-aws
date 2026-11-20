"""Per-call state in the ``calls`` table.

Every per-call fact lives here, keyed by ``CallSid``, so any instance can pick
the call up. Writes use optimistic concurrency on ``version``.

**How the version check works.** The :class:`~owlbank.store.Store` protocol only
offers ``not_exists``/``exists`` conditions, so a writer moving the state from
version *n* to *n+1* first claims the transition by putting a marker item
``<call_sid>#v<n+1>`` with ``not_exists``. Exactly one writer can win a claim;
the loser gets :class:`VersionConflict` and should reload (see :func:`mutate`).
A claim left behind by a writer that crashed before writing the state is taken
over after ``claim_timeout_s``. Claim items carry the same TTL as the call.

Never stored here: factor values, digits, raw tokens. The call token is
held only as ``sealed_token``, encrypted by :mod:`owlbank.token_seal`; the relay
nonce only as ``ws_nonce_hash`` (:mod:`owlbank.ws_nonce`); a sensitive slot's value only sealed, in
``sealed_sensitive`` (:mod:`owlbank.sensitive`). :func:`create` and
:func:`update` run the redaction guard over the item (registered values only;
the fields are fixed by the dataclass, plus any preserved unknown attribute).

**Unknown attributes are preserved.** An item written by newer code can carry attributes this
code's :class:`CallState` has no field for. :meth:`CallState.from_item` keeps them (``_extras``)
and :meth:`CallState.to_item` writes them back, so a writer running older code never drops a newer
field: a whole-item ``put`` by an older front door once erased the bridge's ``agent_owner``. The redaction
guard scans them like every other attribute.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal, get_args

from owlbank import humanleg, last_spoken, welcome
from owlbank.obs import emit, log_caught
from owlbank.segments import assert_clean
from owlbank.store import ConditionFailed, Store
from owlbank.token_seal import Sealer, UnsealError

log = logging.getLogger(__name__)

TABLE = "calls"

#: Field name of the hand-off prefetch, for ``patch``/``mutate`` by name.
F_PREFETCHED = "prefetched"
#: Field name of the asynchronous one-time-code send result, written by the auth service.
F_OTP_SEND = "otp_send"
#: Field name of the bridge process that ran the call's latest ConversationRelay session (session
#: ownership decides whether the agent's history must be reloaded).
F_AGENT_OWNER = "agent_owner"
#: Field name of the count of the call's ConversationRelay sessions that recorded an owner:
#: set in the same write as ``agent_owner``, so a session is the call's first
#: only while it is 0 (with no other sign of an earlier AI session).
F_AI_SESSIONS = "ai_sessions"
#: Field name of the sealed text the caller hears at the start of the latest ConversationRelay
#: session; see :mod:`owlbank.last_spoken`.
F_LAST_SPOKEN = "last_spoken"
#: Field name of the KMS-sealed sensitive values of the call;
#: written by :func:`owlbank.sensitive.remember`, read by :func:`owlbank.sensitive.restore`.
F_SEALED_SENSITIVE = "sealed_sensitive"
#: Field name of the welcome the auth service prepared while the last factor was asked
#: See :func:`seal_welcome` and :func:`unseal_welcome`.
F_WELCOME_PREPARED = "welcome_prepared"
#: Field name of the front door's note to the bridge that the welcome made the web-activity offer:
#: :data:`WELCOME_OFFER_MADE` or ``None``, cleared with ``last_spoken``.
F_WELCOME_OFFER = "welcome_offer"
#: Field name of the recap writer's claim and outcome, written by
#: ``lambdas.recap_writer`` only.
F_RECAP = "recap"
#: Field name of the no-second-try flag: set by the front door
#: when the call's login ended as ``verify_failed``, ``locked`` or ``declined``; once set, no
#: further verification (``authenticate``, step-up) is started on the call.
F_LOGIN_CLOSED = "login_closed"
WELCOME_OFFER_MADE = "made"
#: How the caller's identity was claimed: ``ani`` (the calling
#: number) or ``keyed_phone`` (a number the caller keyed in after the calling number found no
#: customer). Only the source is recorded, never the keyed number.
IdentitySource = Literal["ani", "keyed_phone"]
IDENTITY_SOURCES: frozenset[str] = frozenset(get_args(IdentitySource))
#: The derived top-level attributes that put a call in the ``calls``
#: table's sparse ``sweep-due`` index (partition :data:`SWEEP_OPEN_ATTR`, always
#: :data:`SWEEP_OPEN`; sort :data:`SWEEP_DUE_ATTR`, epoch seconds). Written by
#: :meth:`CallState.to_item` exactly when ``escalation`` is set and its conversation is not closed,
#: never read back as state (:meth:`CallState.from_item` drops them).
SWEEP_INDEX = "sweep-due"
SWEEP_OPEN_ATTR = "sweep_open"
SWEEP_DUE_ATTR = "sweep_due_at"
SWEEP_OPEN = "open"
_DERIVED = (SWEEP_OPEN_ATTR, SWEEP_DUE_ATTR)
#: At most this many entries in ``sealed_sensitive``.
MAX_SEALED_SENSITIVE = 16
#: ``otp_send["status"]`` values.
OTP_SEND_STATUSES = frozenset({"sent", "escalate", "error"})


class VersionConflict(RuntimeError):
    """Another writer updated the call state first; reload and retry."""


@dataclass(frozen=True, slots=True)
class PendingTool:
    """The tool call the agent loop is waiting on while the call is in a deterministic flow."""

    tool_use_id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CallState:
    """Everything the app knows about one call. Construct with keywords."""

    call_sid: str
    expires_at: int
    language: str = "en-US"
    ani: str | None = None
    stir_verstat: str | None = None  # StirVerstat from the incoming webhook (None: absent)
    # stir_verstat is in the policy's accepted_verstat; cleared once the caller keys in a
    # phone number (option A). Read only by the recap target.
    ani_trusted: bool = False
    candidate_ids: tuple[str, ...] = ()
    verified_customer_id: str | None = None
    level: int = 0
    factors_verified: tuple[str, ...] = ()  # names only, never values
    pending_tool: PendingTool | None = None
    attempts: Mapping[str, int] = field(default_factory=dict)  # factor name -> count
    step_up_target: str | None = None  # the tool the caller is stepping up for
    escalation: Mapping[str, Any] | None = None  # {reason, at, journal_ref, ...}; summary stays in the journal
    sealed_token: str | None = None  # KMS-sealed call token (owlbank.token_seal), never the raw token
    pending_result: Mapping[str, Any] | None = None  # result envelope for pending_tool, held until the next prompt
    interrupted_utterance: str | None = None  # caller's intent-switch utterance for the bridge (handle_utterance)
    ws_nonce_hash: str | None = None  # SHA-256 of the one-time relay nonce (owlbank.ws_nonce)
    # Hand-off prefetch: {tool_use_id, tool, args, result, at}; the front door
    # uses it once, for the same pending tool call, then clears it.
    prefetched: Mapping[str, Any] | None = None
    # The auth service's answer to an asynchronous ``send_otp`` ({status, at}; status is
    # ``sent`` | ``escalate`` | ``error``, ``at`` epoch seconds). Never a code or a phone number.
    # The front door clears it before each asynchronous send and after reading it.
    otp_send: Mapping[str, Any] | None = None
    # The ``instance_id`` of the bridge process that ran the call's latest
    # ConversationRelay session (None: no AI session yet). An opaque random ID, no call data.
    agent_owner: str | None = None
    # How many ConversationRelay sessions recorded an owner (``agent_owner``
    # writes; a session that keeps the same owner doesn't count). 0: no AI session yet.
    ai_sessions: int = 0
    # The session's welcome greeting, KMS-sealed (key, CallSid context):
    # {sealed, expires_at}. Written by the front door when it connects ConversationRelay, read by
    # the bridge only to repeat it after caller silence. Never plaintext (owlbank.last_spoken).
    last_spoken: Mapping[str, Any] | None = None
    # AgentCore hosting only: epoch seconds of the last relay URL the front door
    # minted (for the 60-minute-cap duration rule and the warm/cold metric). Never the URL itself.
    relay_minted_at: int | None = None
    # The call's sensitive values, each sealed by ``owlbank.sensitive.remember``
    # (the key, CallSid context), so every process that loads the call can register them for
    # the exact-value scrub. Ciphertext only, at most MAX_SEALED_SENSITIVE entries; never printed.
    sealed_sensitive: tuple[str, ...] = field(default=(), repr=False)
    # The welcome prepared by the auth service before verification,
    # {sealed, customer_id, language, reason, at} (seal_welcome); used once by the front door after
    # verification, then cleared. Only an offer ID is sealed in it, never a trait value or text.
    welcome_prepared: Mapping[str, Any] | None = None
    # WELCOME_OFFER_MADE when the session's welcome made the offer, else None.
    welcome_offer: str | None = None
    # Keyed-in phone login: ``keyed_phone`` once candidates were found by a number
    # the caller keyed in; the number itself is only sealed in ``sealed_sensitive``, never here.
    identity_source: IdentitySource = "ani"
    # The recap writer's claim and outcome,
    # {claimed_at, writing_at?, outcome?, reason?, profile_id?, at?} (owlbank.recap). Never recap text.
    recap: Mapping[str, Any] | None = None
    # Keyed-in phone log: the login ended without verification (failed, locked or
    # declined), so the call gets no second try. An item without it loads as False.
    login_closed: bool = False
    version: int = 0
    # Attributes of the stored item this code has no field for (written by newer code): kept on
    # load and written back, so this writer never drops them. Not part of equality.
    _extras: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.call_sid or "#" in self.call_sid:
            raise ValueError("invalid call_sid")
        if not isinstance(self.level, int) or isinstance(self.level, bool) or self.level < 0:
            raise ValueError("level must be a non-negative int")
        if self.identity_source not in IDENTITY_SOURCES:
            raise ValueError(f"identity_source must be one of {sorted(IDENTITY_SOURCES)}")
        # Normalize mutable inputs so the dataclass stays value-like.
        object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids))
        object.__setattr__(self, "factors_verified", tuple(self.factors_verified))
        object.__setattr__(self, "attempts", dict(self.attempts))
        sealed = tuple(self.sealed_sensitive)
        if len(sealed) > MAX_SEALED_SENSITIVE or not all(isinstance(b, str) for b in sealed):
            raise ValueError(f"{F_SEALED_SENSITIVE} must hold at most {MAX_SEALED_SENSITIVE} strings")
        object.__setattr__(self, "sealed_sensitive", sealed)
        if isinstance(self.pending_tool, Mapping):
            object.__setattr__(self, "pending_tool", PendingTool(**self.pending_tool))
        extras = dict(self._extras)
        clash = set(extras) & _FIELD_NAMES
        if clash:
            raise ValueError(f"extra attributes collide with fields: {sorted(clash)}")
        object.__setattr__(self, "_extras", extras)

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def to_item(self) -> dict[str, Any]:
        item = asdict(self)
        del item["_extras"]
        item["candidate_ids"] = list(self.candidate_ids)
        item["factors_verified"] = list(self.factors_verified)
        item[F_SEALED_SENSITIVE] = list(self.sealed_sensitive)
        item["attempts"] = dict(self.attempts)
        item["escalation"] = dict(self.escalation) if self.escalation is not None else None
        item["pending_result"] = dict(self.pending_result) if self.pending_result is not None else None
        item["prefetched"] = dict(self.prefetched) if self.prefetched is not None else None
        item["otp_send"] = dict(self.otp_send) if self.otp_send is not None else None
        item["last_spoken"] = dict(self.last_spoken) if self.last_spoken is not None else None
        item[F_WELCOME_PREPARED] = (dict(self.welcome_prepared) if self.welcome_prepared is not None
                                    else None)
        item[F_RECAP] = dict(self.recap) if self.recap is not None else None
        # Preserved unknown attributes, written back unchanged (never over a field: __post_init__).
        item.update(copy.deepcopy(dict(self._extras)))
        # The sweep index keys, derived here for every writer at once (never carried over).
        for name in _DERIVED:
            item.pop(name, None)
        if humanleg.sweep_indexed(self.escalation, self.ai_sessions):  # Closable, < 6 retries
            item[SWEEP_OPEN_ATTR] = SWEEP_OPEN
            item[SWEEP_DUE_ATTR] = humanleg.sweep_due_at(self.escalation)
        return item

    @classmethod
    def from_item(cls, item: Mapping[str, Any]) -> CallState:
        known = {k: v for k, v in item.items() if k in _FIELD_NAMES}
        extras = {k: copy.deepcopy(v) for k, v in item.items()
                  if k not in _FIELD_NAMES and k not in _DERIVED}
        return cls(**known, _extras=extras)


#: The stored attributes :class:`CallState` has fields for (``_extras`` is not one).
_FIELD_NAMES = frozenset(f.name for f in fields(CallState) if f.name != "_extras")


def _claim_key(call_sid: str, version: int) -> str:
    return f"{call_sid}#v{version}"


#: Records whose ``sealed`` sub-field holds sealer output only (``owlbank.last_spoken.seal``,
#: :func:`seal_welcome`): that sub-field is not scanned, the rest of the record is.
_SEALED_RECORDS = (F_LAST_SPOKEN, F_WELCOME_PREPARED)


def _check(state: CallState) -> dict[str, Any]:
    item = state.to_item()
    # ``sealed_sensitive`` and the ``sealed`` sub-field of ``_SEALED_RECORDS`` hold sealer output
    # only (ciphertext, never plaintext), so they aren't scanned: a blob can hold a registered
    # value's characters by chance, which must not block every later write of the call.
    scanned = {k: v for k, v in item.items() if k != F_SEALED_SENSITIVE}
    for name in _SEALED_RECORDS:
        record = scanned.get(name)
        if isinstance(record, Mapping):
            scanned[name] = {k: v for k, v in record.items() if k != "sealed"}
    assert_clean(state.call_sid, scanned, deny_keys=False)
    return item


def load(store: Store, call_sid: str, *, now: float | None = None) -> CallState | None:
    """The call's state, or ``None`` if absent or expired at ``now`` (TTL is lazy)."""
    item = store.get(TABLE, {"call_sid": call_sid})
    if item is None:
        return None
    state = CallState.from_item(item)
    if now is not None and state.is_expired(now):
        return None
    return state


def create(store: Store, state: CallState) -> CallState:
    """Write a new call state at version 1; :class:`VersionConflict` if it exists."""
    new = replace(state, version=1)
    item = _check(new)
    try:
        store.put(TABLE, item, condition="not_exists")
    except ConditionFailed as exc:
        emit(log, logging.DEBUG, "call_state_exists", operation="create", decision="raise_version_conflict")
        raise VersionConflict(f"call state for {state.call_sid} already exists") from exc
    return new


def update(store: Store, state: CallState, *, now: Callable[[], float] = time.time,
           claim_timeout_s: int = 10, **changes: Any) -> CallState:
    """Write ``state`` with ``changes`` as version ``state.version + 1``.

    Raises :class:`VersionConflict` if the stored state has moved on since
    ``state`` was loaded.
    """
    if "call_sid" in changes or "version" in changes:
        raise ValueError("call_sid and version can't be changed")
    new = replace(state, **changes, version=state.version + 1)
    item = _check(new)
    claim = {"call_sid": _claim_key(state.call_sid, new.version), "kind": "version_claim",
             "claimed_at": int(now()), "expires_at": new.expires_at}
    try:
        store.put(TABLE, claim, condition="not_exists")
    except ConditionFailed:
        emit(log, logging.DEBUG, "call_state_claim_taken", operation="update", version=new.version,
             decision="check_stale_claim")
        if not _take_over_stale_claim(store, state, claim, now=now, timeout_s=claim_timeout_s):
            raise VersionConflict(f"call {state.call_sid} moved past version {state.version}") from None
    head = store.get(TABLE, {"call_sid": state.call_sid})
    if head is None or head.get("version") != state.version:
        raise VersionConflict(f"call {state.call_sid} is not at version {state.version}")
    store.put(TABLE, item, condition="exists")
    return new


def _take_over_stale_claim(store: Store, state: CallState, claim: dict[str, Any], *,
                           now: Callable[[], float], timeout_s: int) -> bool:
    existing = store.get(TABLE, {"call_sid": claim["call_sid"]})
    head = store.get(TABLE, {"call_sid": state.call_sid})
    if existing is None or head is None or head.get("version") != state.version:
        return False
    if int(now()) - int(existing.get("claimed_at", 0)) < timeout_s:
        return False
    # The claimer died before writing; replace its claim (still one winner per claim).
    store.delete(TABLE, {"call_sid": claim["call_sid"]})
    try:
        store.put(TABLE, claim, condition="not_exists")
    except ConditionFailed:
        emit(log, logging.DEBUG, "call_state_claim_lost", operation="take_over_claim",
             decision="ignored_because_another_writer_won")
        return False
    return True


def mutate(store: Store, call_sid: str, fn: Callable[[CallState], Mapping[str, Any] | None], *,
           retries: int = 5, now: Callable[[], float] = time.time) -> CallState:
    """Load, compute ``changes = fn(state)``, update; reload and retry on conflict. ``fn`` returning
    ``None`` writes nothing at all (the state is returned as loaded); ``{}`` still writes a new version."""
    for _ in range(retries):
        state = load(store, call_sid)
        if state is None:
            raise KeyError(call_sid)
        changes = fn(state)
        if changes is None:
            return state
        try:
            return update(store, state, now=now, **dict(changes))
        except VersionConflict:
            emit(log, logging.DEBUG, "call_state_conflict", operation="mutate", decision="retry")
            continue
    emit(log, logging.WARNING, "call_state_conflict_exhausted", operation="mutate", retries=retries,
         decision="raise_version_conflict")
    raise VersionConflict(f"gave up updating {call_sid} after {retries} attempts")


# --------------------------------------------------------------------------- prepared welcome


@dataclass(frozen=True, slots=True)
class PreparedWelcome:
    """An unsealed :data:`F_WELCOME_PREPARED` record. ``offer_id`` is set only when ``reason`` is
    ``matched`` (:data:`owlbank.welcome.MATCHED`); any other reason means the fallback welcome.
    ``at`` is when it was prepared."""

    offer_id: str | None
    customer_id: str
    language: str
    reason: str
    at: int


def _offer_id_ok(offer_id: Any, reason: Any) -> bool:
    """A known reason (:data:`owlbank.welcome.REASONS`), with an offer ID
    (:data:`owlbank.welcome.OFFER_ID_RE`) exactly when it is ``matched``."""
    if reason not in welcome.REASONS:
        return False
    if offer_id is None:
        return reason != welcome.MATCHED
    return (reason == welcome.MATCHED and isinstance(offer_id, str)
            and welcome.OFFER_ID_RE.fullmatch(offer_id) is not None)


def _welcome_payload(offer_id: str | None, customer_id: str, language: str, reason: str) -> str:
    return json.dumps({"offer_id": offer_id, "customer_id": customer_id, "language": language, "reason": reason},
                      sort_keys=True, separators=(",", ":"))


def seal_welcome(sealer: Sealer, offer_id: str | None, *, call_sid: str, customer_id: str, language: str,
                 reason: str, now: float) -> dict[str, Any] | None:
    """The :data:`F_WELCOME_PREPARED` record: ``{sealed, customer_id, language, reason, at}``.

    ``sealed`` seals the offer ID together with the customer, language and reason
    (``owlbank.last_spoken``'s sealer: the key with the ``CallSid`` as encryption context), so a
    record copied to another call, or with an edited customer, language or reason, doesn't unseal.
    A fallback record (no offer) is sealed too: it is call-bound like the rest. ``None`` when sealing
    fails (logged without the payload: the front door then finds no prepared welcome and speaks
    today's). ``offer_id`` (:data:`owlbank.welcome.OFFER_ID_RE`) must be given exactly when
    ``reason`` is ``matched``; ``reason`` is one of :data:`owlbank.welcome.REASONS`. The record never holds a
    trait value or welcome text."""
    if not _offer_id_ok(offer_id, reason):
        raise ValueError("a known reason, and an offer_id exactly for a matched welcome")
    if not customer_id or not language or not reason:
        raise ValueError("customer_id, language and reason are required")
    try:
        sealed = sealer.seal(_welcome_payload(offer_id, customer_id, language, reason), call_sid=call_sid)
    except Exception as exc:
        log_caught(log, "welcome_prepared could not be sealed", exc, level=logging.ERROR,
                   operation="seal_welcome", decision="store_none_fallback_welcome")
        return None
    return {"sealed": sealed, "customer_id": customer_id, "language": language, "reason": reason,
            "at": int(now)}


def unseal_welcome(sealer: Sealer, record: Any, *, call_sid: str, now: float) -> PreparedWelcome | None:
    """The :func:`seal_welcome` record for this call, or ``None`` if it is missing, malformed, older
    than ``owlbank.last_spoken.TTL_S``, doesn't unseal for ``call_sid`` (another call's record), or
    its clear fields differ from the sealed ones. Never raises; logs the reason, never the payload."""
    if record is None:
        return None
    try:
        return _unseal_welcome(sealer, record, call_sid=call_sid, now=now)
    except Exception as exc:
        log_caught(log, "welcome_prepared could not be unsealed", exc, operation="unseal_welcome",
                   decision="fallback_welcome")
        return None


def _unusable(reason: str, level: int = logging.WARNING) -> None:
    emit(log, level, "welcome_prepared_unusable", reason=reason, decision="fallback_welcome")


def _unseal_welcome(sealer: Sealer, record: Any, *, call_sid: str, now: float) -> PreparedWelcome | None:
    if not isinstance(record, Mapping):
        _unusable("bad_shape")
        return None
    sealed, at = record.get("sealed"), record.get("at")
    clear = (record.get("customer_id"), record.get("language"), record.get("reason"))
    if (not isinstance(sealed, str) or not sealed or isinstance(at, bool) or not isinstance(at, int | float)
            or not math.isfinite(at) or not all(isinstance(v, str) and v for v in clear)):
        _unusable("bad_shape")
        return None
    if now >= at + last_spoken.TTL_S:
        _unusable("expired", logging.INFO)
        return None
    try:
        payload = json.loads(sealer.unseal(sealed, call_sid=call_sid))
    except UnsealError:
        _unusable("unseal_failed")
        return None
    except ValueError:
        _unusable("bad_shape")
        return None
    offer_id = payload.get("offer_id") if isinstance(payload, dict) else None
    if (not isinstance(payload, dict)
            or (payload.get("customer_id"), payload.get("language"), payload.get("reason")) != clear
            or not _offer_id_ok(offer_id, clear[2])):
        _unusable("mismatch")
        return None
    return PreparedWelcome(offer_id=offer_id, customer_id=str(clear[0]), language=str(clear[1]),
                           reason=str(clear[2]), at=int(at))
