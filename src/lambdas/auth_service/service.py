"""Auth service logic.

Invoked only by the front door (Lambda direct invoke through IAM; no API). Two
operations, both pure functions over injected dependencies (:class:`AuthDeps`):

``verify(call_sid, ani, factors, target_level, stir_verstat, identity_source)``
    Checks the factors for ``target_level`` against the candidate customer(s),
    then issues the call-bound token (initial verification) or upgrades it
    (step-up). Returns ``{"result": ..., "level": ...}`` where ``result`` is one of
    :data:`RESULTS`.

``send_otp(call_sid, ani, language, stir_verstat, identity_source)``
    Starts a Twilio Verify code (see :func:`otp_destination`), before the front door
    gathers the ``otp`` factor. Needs an active token. The code goes only to the calling
    ANI, or on a keyed-in number's call (``identity_source`` ``keyed_phone``) to that number
    after ZIP and SSN-4, and only if it resolves to the verified customer (the same
    compliance review applies); the call's
    ``language`` becomes the Verify ``Locale``. With no such destination, or when Verify
    can't deliver, the result is ``{"result": "escalate", "reason": "otp_unavailable"}``
    and the front door transfers the call to a human.

``precheck(call_sid, ani, target_level, stir_verstat, identity_source)``
    The untrusted-caller gate on its own, called by the front door before
    it asks the first factor of a login or step-up: ``{"result": "ok"}`` or
    ``{"result": "escalate", "reason": "verification_risk"}``. ``verify`` applies the
    same gate again before it checks any factor.

``prepare_welcome(call_sid, customer_id, language)``
    The web-activity welcome, started by the front door
    as an asynchronous ``Event`` invoke when the last factor is asked. Reads the customer's
    ``first_name`` and ``memory_email``, looks the profile up in Twilio Memory and reads its
    ``WebActivity`` traits (1 s timeout each), bounds them (``bound_profile_traits``), matches
    the product against the policy's ``web_offers`` (:func:`owlbank.welcome.prepare`) and writes
    ``welcome_prepared`` to the call state, sealed (:func:`owlbank.callstate.seal_welcome`):
    only an offer ID, the customer, the language and a reason. Never raises. Logs one
    ``welcome_built`` line (reason, ``duration_ms``, trait count), never a trait value.

**Initial verification vs step-up.** The service reads the call's active tokens
(``tokens`` GSI by ``call_sid``). With none, it is an initial verification: the
candidates are the customers whose phone equals the ANI (``ani_match``; ANI alias
records with ``alias_of`` resolve to their fixture), and every
collected factor of ``target_level`` is required. With an active token, it is a
step-up: the only candidate is the token's customer (a step-up can't switch
customer), only the factors still missing are required, and on success
the old token(s) are revoked and a new one is issued at ``target_level``.

**Disambiguation.** Every supplied factor must match one candidate's
stored salted hash. Extra factors beyond the level's are allowed and checked, so
the front door can add a disambiguating factor. If more than one candidate still
matches, the result is ``not_verified`` with ``reason: "ambiguous"`` and the
attempt is not counted as a failure.

**Rate limits.** Counter items in the ``calls`` table, each a "slot"
claimed with ``put(condition="not_exists")``, so concurrent attempts can't exceed
a limit:

* per call: ``ratelimit#call#<CallSid>#verify#<n>`` (initial verification,
  ``limits.verify_attempts_per_call``) and ``...#stepup#<n>``
  (``limits.stepup_attempts_per_call``). A slot is claimed before the check and
  released on success or on an ambiguous result, so the count is *failed*
  attempts. TTL: the record TTL.
* per customer: ``ratelimit#customer#<id>#w<window>#<n>``, a fixed window of
  ``limits.verify_customer_window_s`` (default 3600 s) allowing
  ``limits.verify_attempts_per_customer`` (default 5) failures across calls.
  Failures are recorded for every candidate of the ANI; a locked candidate is
  excluded, and if none is left the result is ``rate_limited``.
* OTP sends per call: ``ratelimit#call#<CallSid>#otpsend#<n>``
  (``limits.otp_sends_per_call``, default 3).

**Caller-ID trust.** ``stir_verstat`` is the call's ``StirVerstat`` as
the front door received it (signed webhook; absent on many calls). It is trusted only
if the policy's ``ani_trust.accepted_verstat`` lists it; missing is untrusted, never an
error. This service, not the front door, applies the gate:

* With ``on_untrusted: no_ani_match`` and an untrusted ANI, ``ani_match`` is not a
  factor: the factors required are :meth:`owlbank.policy.Policy.effective_factors`
  (the next level's collected factors), and ``factors_verified`` leaves it out.
* With an untrusted ANI (either mode), a failure counts toward the per-call limit but
  is **not** recorded against any customer's lockout window, so a spoofed caller ID
  can't lock the real customer out. An existing lockout still applies.
* ``identity_source`` says how the front door identified the caller:
  ``"ani"`` (the default when absent: the rules above) or ``"keyed_phone"`` (a number the
  caller keyed in, sent as ``ani``). A keyed call is untrusted and ``ani_match`` never counts,
  whatever the verstat and ``on_untrusted`` (also under ``allow``); every ``ani_trusted`` it
  logs is false. Any other value, or a non-string, is ``bad_request`` (fail closed).

**Untrusted-attempt cap.** Instead, each failure from an untrusted
ANI is counted for every candidate customer it failed against, in the ``risk`` table:
slot items ``untrusted#<customer_id>#<n>`` (``n`` 1..``ani_trust.untrusted_failure_limit``)
claimed with ``put(condition="not_exists")``. Every slot of one window carries the same
``expires_at``: ``ani_trust.untrusted_window_s`` after the window's first failure (the
DynamoDB TTL is only a backstop; expiry is checked on read). The items hold the customer
ID, the claim time and the expiry, nothing else (no ANI, no factor values). With an
untrusted ANI, before any factor is checked:

* if the factors required include ``otp`` and the call has no token yet, the result is
  ``escalate`` / ``verification_risk``: no code is ever sent before login;
* if any candidate customer has all its slots live (the count is at the limit), the
  result is ``escalate`` / ``verification_risk`` and no factor is evaluated.

A trusted caller never reads or writes the ``risk`` table.

**Audit.** One JSON line per event on the ``owlbank.audit`` logger:
``token_issued``, ``step_up``, ``verification_failed``, ``verification_ambiguous``,
``rate_limited``, ``invalid_level``, ``otp_sent`` (with ``ani_trusted``), ``otp_unavailable``,
``verification_risk``, ``bad_request``, ``unavailable``.
Fields are call SID, customer ID, levels, factor *names* and a reason. Factor
values, OTP codes, phone numbers, tokens and hashes are never logged or returned except
the new token itself, which goes back to the front door in the response.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeGuard

from lambdas.auth_service.verify_client import VerifyClient, VerifyUnavailable
from owlbank import callstate, memory_http, obs, security_log, tokens, web_activity, welcome
from owlbank.policy import ESTABLISHED_FACTORS, Policy
from owlbank.store import ConditionFailed, Store
from owlbank.token_seal import Sealer
from scripts.hash_tool import verify as verify_hash

log = logging.getLogger(__name__)
audit_log = logging.getLogger("owlbank.audit")

VERIFIED = "verified"
NOT_VERIFIED = "not_verified"
RATE_LIMITED = "rate_limited"
INVALID_LEVEL = "invalid_level"
UNAVAILABLE = "unavailable"
BAD_REQUEST = "bad_request"
SENT = "sent"
ESCALATE = "escalate"
OK = "ok"
VERIFICATION_RISK = "verification_risk"
OTP_UNAVAILABLE = "otp_unavailable"

#: ``result`` values of :func:`verify`.
RESULTS: frozenset[str] = frozenset({VERIFIED, NOT_VERIFIED, RATE_LIMITED, INVALID_LEVEL,
                                     UNAVAILABLE, BAD_REQUEST, ESCALATE})

OTP_FACTOR = "otp"
CALLS = "calls"
CUSTOMERS = "customers"
RISK = "risk"
PHONE_INDEX = "phone-index"

#: ``identity_source`` values: how the front door identified the caller. ``ani``
#: (the default) is the call's own caller ID; ``keyed_phone`` is a number the caller keyed in,
#: which is never trusted and never counts ``ani_match``, whatever ``ani_trust.on_untrusted`` says.
IDENTITY_ANI = "ani"
IDENTITY_KEYED_PHONE = "keyed_phone"
IDENTITY_SOURCES: frozenset[str] = frozenset({IDENTITY_ANI, IDENTITY_KEYED_PHONE})

#: Defaults for limits the policy table may omit.
DEFAULT_LIMITS: Mapping[str, int] = {
    "verify_attempts_per_call": 3,
    "stepup_attempts_per_call": 2,
    "verify_attempts_per_customer": 5,
    "verify_customer_window_s": 3600,
    "otp_sends_per_call": 3,
}

#: Verify channel for a factor's policy ``channel`` value.
_VERIFY_CHANNELS: Mapping[str, str] = {"verify_sms": "sms", "verify_call": "call",
                                       "verify_whatsapp": "whatsapp"}

_E164 = re.compile(r"\+[1-9]\d{1,14}")
_LANGUAGE = re.compile(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8}){0,2}")

_MAX_ID_LEN = 64
_MAX_FACTOR_LEN = 32
_MAX_VERSTAT_LEN = 64


def clean_verstat(value: Any) -> str | None:
    """A ``StirVerstat`` safe to compare and log; anything else is ``None`` (untrusted)."""
    if not isinstance(value, str) or not 0 < len(value) <= _MAX_VERSTAT_LEN:
        return None
    return value if all(c.isalnum() or c in "-_." for c in value) else None


@dataclass
class AuthDeps:
    """Everything the auth service needs; the handler builds the production set."""

    store: Store
    policy: Policy
    verify_client: VerifyClient | None = None
    clock: Callable[[], int] = field(default=lambda: int(time.time()))
    token_max_ttl_s: int = 1800
    record_ttl_s: int = 86400
    # prepare_welcome: ``sealer`` seals welcome_prepared;
    # ``memory_client`` returns a twilio.rest.Client (the twilio_api_key secret), called lazily.
    sealer: Sealer | None = None
    memory_client: Callable[[], Any] | None = None
    memory_store_id: str | None = None
    memory_trait_groups: tuple[str, ...] = web_activity.DEFAULT_TRAIT_GROUPS
    web_activity_max_age_hours: int = web_activity.DEFAULT_MAX_AGE_HOURS
    memory_timeout_s: float = 1.0


# --------------------------------------------------------------------------- helpers


#: Audit events that mean something went wrong or was refused: WARNING, so they show at any level.
_WARN_EVENTS = frozenset({"unavailable", "bad_request", "verification_risk", "rate_limited",
                          "otp_unavailable", "invalid_level"})


def audit(event: str, **fields: Any) -> None:
    """One structured audit line. Callers pass names and IDs only, never values."""
    if "level" in fields:  # "level" is the log level in the JSON line
        fields["auth_level"] = fields.pop("level")
    level = logging.WARNING if event in _WARN_EVENTS else logging.INFO
    obs.emit(audit_log, level, event, audit=event, component="auth_service", **fields)


def _limit(policy: Policy, name: str) -> int:
    try:
        return policy.limit(name)
    except KeyError:
        obs.emit(log, logging.DEBUG, "policy_limit_missing", limit=name, decision="fallback_default_limit")
        return DEFAULT_LIMITS[name]


def _is_int(v: object) -> TypeGuard[int]:
    return isinstance(v, int) and not isinstance(v, bool)


def _valid_call_sid(call_sid: object) -> TypeGuard[str]:
    return (isinstance(call_sid, str) and 0 < len(call_sid) <= _MAX_ID_LEN
            and "#" not in call_sid)


class _Slots:
    """Bounded attempt counters as claimable slot items in the ``calls`` table."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def reserve(self, prefix: str, limit: int, *, now: int, ttl_s: int) -> str | None:
        """Claim the first free slot ``prefix#1..limit``; ``None`` if all are taken."""
        for n in range(1, limit + 1):
            key = f"{prefix}#{n}"
            item = {"call_sid": key, "kind": "rate_limit_slot", "claimed_at": now,
                    "expires_at": now + ttl_s}
            try:
                self._store.put(CALLS, item, condition="not_exists")
                return key
            except ConditionFailed:
                obs.emit(log, logging.DEBUG, "rate_slot_taken", decision="retry_next_slot")
                continue
        return None

    def full(self, prefix: str, limit: int, *, now: int) -> bool:
        """True if every slot is taken (slots without release fill in order)."""
        item = self._store.get(CALLS, {"call_sid": f"{prefix}#{limit}"})
        return item is not None and int(item.get("expires_at", 0)) > now

    def release(self, key: str) -> None:
        self._store.delete(CALLS, {"call_sid": key})


class UntrustedCounter:
    """Failures from untrusted ANIs per customer, as slot items in the ``risk`` table.

    ``untrusted#<customer_id>#<n>`` for ``n`` in ``1..limit``. A slot is live while
    ``now < expires_at``. Claiming uses ``put(condition="not_exists")``, so concurrent
    failures each get a different slot and the count never goes past ``limit``. All slots
    of a window share one ``expires_at`` (``window_s`` after its first failure), so they
    lapse together and counting restarts. An expired slot is reclaimed by delete and a
    conditional put; two attempts racing on the same expired slot can lose one count at
    a window boundary (never add one).
    """

    def __init__(self, store: Store, *, limit: int, window_s: int) -> None:
        self._store, self.limit, self.window_s = store, limit, window_s

    @staticmethod
    def key(customer_id: str, n: int) -> str:
        return f"untrusted#{customer_id}#{n}"

    def _slots(self, customer_id: str) -> list[dict[str, Any] | None]:
        return [self._store.get(RISK, {"risk_key": self.key(customer_id, n)})
                for n in range(1, self.limit + 1)]

    @staticmethod
    def _live(item: Mapping[str, Any] | None, now: int) -> bool:
        return item is not None and _is_int(item.get("expires_at")) and now < item["expires_at"]

    def count(self, customer_id: str, now: int) -> int:
        return sum(1 for it in self._slots(customer_id) if self._live(it, now))

    def at_limit(self, customer_id: str, now: int) -> bool:
        return self.count(customer_id, now) >= self.limit

    def record(self, customer_id: str, now: int) -> bool:
        """Count one failure; ``False`` if every slot is already live."""
        slots = self._slots(customer_id)
        live = [it["expires_at"] for it in slots if it is not None and self._live(it, now)]
        expires_at = min(live) if live else now + self.window_s
        for n, item in enumerate(slots, start=1):
            if self._live(item, now):
                continue
            key = {"risk_key": self.key(customer_id, n)}
            if item is not None:  # expired: reclaim, unless someone just did
                if self._live(self._store.get(RISK, key), now):
                    continue
                self._store.delete(RISK, key)
            try:
                self._store.put(RISK, {**key, "kind": "untrusted_failure",
                                       "customer_id": customer_id, "claimed_at": now,
                                       "expires_at": expires_at}, condition="not_exists")
                return True
            except ConditionFailed:
                obs.emit(log, logging.DEBUG, "untrusted_slot_taken", decision="retry_next_slot")
                continue
        return False


def _untrusted_counter(deps: AuthDeps) -> UntrustedCounter:
    cfg = deps.policy.ani_trust
    return UntrustedCounter(deps.store, limit=cfg.untrusted_failure_limit,
                            window_s=cfg.untrusted_window_s)


def _risk_gate(deps: AuthDeps, *, call_sid: str, candidates: list[dict[str, Any]],
               step_up: bool, required: list[str], now: int, current_level: int,
               op: str) -> dict[str, Any] | None:
    """The untrusted-caller gate. Call only for an untrusted ANI.

    ``escalate`` / ``verification_risk`` if a one-time code would be needed before a
    token exists, or if any candidate's untrusted-failure count is at the limit."""
    cause = None
    if not step_up and OTP_FACTOR in required:
        cause = "otp_before_login"
    else:
        counter = _untrusted_counter(deps)
        hit = [c["customer_id"] for c in candidates if counter.at_limit(c["customer_id"], now)]
        if hit:
            cause = "untrusted_failure_limit"
    if cause is None:
        return None
    audit("verification_risk", call_sid=call_sid, op=op, cause=cause,
          kind="stepup" if step_up else "verify", ani_trusted=False,
          customer_ids=[c["customer_id"] for c in candidates])
    return {"result": ESCALATE, "reason": VERIFICATION_RISK, "level": current_level}


def _customer_prefix(deps: AuthDeps, customer_id: str, now: int) -> str:
    window_s = _limit(deps.policy, "verify_customer_window_s")
    return f"ratelimit#customer#{customer_id}#w{now // window_s}"


def _active_tokens(deps: AuthDeps, call_sid: str, now: int) -> list[dict[str, Any]]:
    items = deps.store.query(tokens.TABLE, {"call_sid": call_sid}, index=tokens.CALL_INDEX)
    return [i for i in items if i.get("revoked") is False and _is_int(i.get("expires_at"))
            and now < i["expires_at"] and i.get("call_sid") == call_sid]


def _candidates_for_ani(deps: AuthDeps, ani: Any) -> list[dict[str, Any]]:
    if not isinstance(ani, str) or not ani.startswith("+") or len(ani) > 20:
        return []  # withheld ("anonymous", ""), malformed, or not E.164
    items = deps.store.query(CUSTOMERS, {"phone": ani}, index=PHONE_INDEX)
    ids: set[str] = set()
    for item in items:  # ANI alias records (scripts/seed.py) stand for their fixture
        target = item.get("alias_of", item.get("customer_id"))
        if isinstance(target, str) and target:
            ids.add(target)
    out = []
    for cid in sorted(ids):
        c = deps.store.get(CUSTOMERS, {"customer_id": cid})
        if c is not None and "alias_of" not in c:
            out.append(c)
    return out


def otp_destination(deps: AuthDeps, customer: Mapping[str, Any], ani: Any) -> str | None:
    """Where the OTP goes: ``ani``, only if it is E.164 and resolves
    to ``customer`` (the customer's own phone, or an ``alias_of`` record for them, e.g. a
    registered developer or test number). In every other case ``None``: a withheld or
    malformed ANI, another customer's number, a number with no record. There is no
    fallback to the phone on file when it differs from ``ani``, and no other number is ever
    used.

    ``ani`` is the calling ANI (the call's ``From`` as the front door received it, signed
    webhook), or on a keyed call the keyed number (keyed-in phone login, sent with
    ``stir_verstat=None``): a number on file for the customer either way, so the code stays a
    second factor. ``verify`` and ``send_otp`` use the same rule, so Verify checks the number it
    sent to.
    """
    cid = customer.get("customer_id")
    if not isinstance(cid, str) or not isinstance(ani, str) or not _E164.fullmatch(ani):
        return None
    for item in deps.store.query(CUSTOMERS, {"phone": ani}, index=PHONE_INDEX):
        if item.get("alias_of", item.get("customer_id")) == cid:
            return ani
    return None


def _clean_language(language: Any) -> str | None:
    """A BCP-47 tag safe to pass on (``en-US``); anything else is ``None``."""
    return language if isinstance(language, str) and _LANGUAGE.fullmatch(language) else None


def _matches(customer: Mapping[str, Any], hashed: Mapping[str, str]) -> bool:
    stored = customer.get("factors") or {}
    ok = True
    for name, value in hashed.items():  # check all (no early exit on the first mismatch)
        h = stored.get(name)
        ok = (isinstance(h, str) and verify_hash(value, h)) and ok
    return ok


def _bad(call_sid: Any, reason: str, level: int = 0, **extra: Any) -> dict[str, Any]:
    audit("bad_request", call_sid=call_sid if _valid_call_sid(call_sid) else None, reason=reason)
    return {"result": BAD_REQUEST, "level": level, "reason": reason, **extra}


def _valid_identity_source(value: object) -> TypeGuard[str]:
    """One of :data:`IDENTITY_SOURCES`; anything else (another string, ``None``, a non-string) is
    a ``bad_request`` (fail closed, never read as ``ani``)."""
    return isinstance(value, str) and value in IDENTITY_SOURCES


def _trust(policy: Policy, verstat: str | None, identity_source: str) -> tuple[bool, bool]:
    """``(ani_trusted, ani_match_counts)`` for a call. A keyed-in
    number is neither, whatever the verstat and ``ani_trust.on_untrusted``."""
    if identity_source == IDENTITY_KEYED_PHONE:
        return False, False
    return policy.ani_trusted(verstat), policy.ani_match_counts(verstat)


# --------------------------------------------------------------------------- verify


def verify(deps: AuthDeps, *, call_sid: Any, ani: str | None, factors: Mapping[str, str],
           target_level: Any, stir_verstat: str | None = None,
           identity_source: Any = IDENTITY_ANI) -> dict[str, Any]:
    """Verify ``factors`` for ``target_level`` on ``call_sid``. See the module docstring.

    ``identity_source``: ``"ani"`` (default) or ``"keyed_phone"``, which makes
    the call untrusted with no ``ani_match`` whatever ``ani_trust.on_untrusted`` says; any other
    value is ``bad_request``.

    Response keys: ``result``, ``level`` (the call's level after this attempt), and

    * on ``verified``: ``customer_id``, ``token`` (the new opaque token; the old one is
      revoked on step-up), ``factors_verified`` (names), ``step_up`` (bool);
    * on ``not_verified``: ``reason`` (``mismatch`` | ``ambiguous`` | ``no_candidates``)
      and ``attempts_remaining`` (per-call failures left);
    * on ``escalate``: ``reason`` (``verification_risk``);
    * on ``bad_request``: ``reason``, and ``missing`` (factor names) when relevant.
    """
    policy = deps.policy
    if not _valid_call_sid(call_sid):
        return _bad(call_sid, "call_sid")
    if not _valid_identity_source(identity_source):
        return _bad(call_sid, "identity_source")
    if not isinstance(factors, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) and 0 < len(v) <= _MAX_FACTOR_LEN
        for k, v in factors.items()
    ):
        return _bad(call_sid, "factors")
    if not _is_int(target_level) or not 1 <= target_level <= policy.max_level:
        audit("invalid_level", call_sid=call_sid,
              target_level=target_level if _is_int(target_level) else None)
        return {"result": INVALID_LEVEL, "level": 0}
    names = set(factors)
    if names & ESTABLISHED_FACTORS:
        return _bad(call_sid, "established_factor_supplied")
    if any(policy.factor(n) is None for n in names):
        return _bad(call_sid, "unknown_factor")

    try:
        return _verify(deps, call_sid=call_sid, ani=ani, factors=factors,
                       target_level=target_level, stir_verstat=clean_verstat(stir_verstat),
                       identity_source=identity_source)
    except Exception as exc:
        obs.log_caught(log, "auth_service_operation_failed", exc, level=logging.ERROR, operation="verify",
                       call_sid=call_sid if _valid_call_sid(call_sid) else None,
                       decision="return_unavailable")
        audit("unavailable", call_sid=call_sid, op="verify")
        return {"result": UNAVAILABLE, "level": 0}


def _verify(deps: AuthDeps, *, call_sid: str, ani: str | None, factors: Mapping[str, str],
            target_level: int, stir_verstat: str | None,
            identity_source: str = IDENTITY_ANI) -> dict[str, Any]:
    policy, store = deps.policy, deps.store
    now = deps.clock()
    trusted, ani_counts = _trust(policy, stir_verstat, identity_source)
    active = _active_tokens(deps, call_sid, now)
    customer_ids = {t["customer_id"] for t in active}
    if len(customer_ids) > 1:  # should never happen; fail closed
        audit("unavailable", call_sid=call_sid, reason="tokens_for_several_customers")
        return {"result": UNAVAILABLE, "level": 0}
    current_level = max((t["level"] for t in active), default=0)
    step_up = bool(active)
    if target_level <= current_level:
        audit("invalid_level", call_sid=call_sid, target_level=target_level,
              current_level=current_level, reason="not_above_current")
        return {"result": INVALID_LEVEL, "level": current_level}

    have = set(policy.effective_factors(current_level, ani_counts=ani_counts)) \
        | set(ESTABLISHED_FACTORS)
    required = [f for f in policy.effective_factors(target_level, ani_counts=ani_counts)
                if f not in have]
    if step_up:
        cust = store.get(CUSTOMERS, {"customer_id": next(iter(customer_ids))})
        candidates = [cust] if cust is not None and "alias_of" not in cust else []
    else:
        candidates = _candidates_for_ani(deps, ani)
    if not trusted:  # before any factor is checked
        gate = _risk_gate(deps, call_sid=call_sid, candidates=candidates, step_up=step_up,
                          required=required, now=now, current_level=current_level, op="verify")
        if gate is not None:
            return gate
    missing = [f for f in required if f not in factors]
    if missing:
        return _bad(call_sid, "missing_factors", current_level, missing=missing)

    kind = "stepup" if step_up else "verify"
    call_limit = _limit(policy, f"{kind}_attempts_per_call")
    slots = _Slots(store)
    slot = slots.reserve(f"ratelimit#call#{call_sid}#{kind}", call_limit, now=now,
                         ttl_s=deps.record_ttl_s)
    if slot is None:
        audit("rate_limited", call_sid=call_sid, scope="call", kind=kind,
              target_level=target_level)
        return {"result": RATE_LIMITED, "level": current_level}

    def remaining() -> int:
        used = sum(1 for n in range(1, call_limit + 1)
                   if store.get(CALLS, {"call_sid": f"ratelimit#call#{call_sid}#{kind}#{n}"}))
        return max(call_limit - used, 0)

    cust_limit = _limit(policy, "verify_attempts_per_customer")
    open_candidates = [c for c in candidates
                       if not slots.full(_customer_prefix(deps, c["customer_id"], now),
                                         cust_limit, now=now)]
    if candidates and not open_candidates:
        audit("rate_limited", call_sid=call_sid, scope="customer", kind=kind,
              customer_ids=[c["customer_id"] for c in candidates], target_level=target_level)
        return {"result": RATE_LIMITED, "level": current_level}

    def fail(reason: str, failed: list[dict[str, Any]]) -> dict[str, Any]:
        window_ttl = _limit(policy, "verify_customer_window_s")
        if trusted:  # an untrusted (possibly spoofed) ANI never locks the customer out
            for c in failed:
                slots.reserve(_customer_prefix(deps, c["customer_id"], now), cust_limit, now=now,
                              ttl_s=window_ttl)
        else:  # ... but counts toward the untrusted-attempt cap
            counter = _untrusted_counter(deps)
            for c in failed:
                counter.record(c["customer_id"], now)
        audit("verification_failed", call_sid=call_sid, kind=kind, reason=reason,
              target_level=target_level, factor_names=sorted(factors),
              customer_ids=[c["customer_id"] for c in failed], ani_trusted=trusted,
              customer_lockout_counted=trusted and bool(failed),
              untrusted_failure_counted=not trusted and bool(failed))
        return {"result": NOT_VERIFIED, "level": current_level, "reason": reason,
                "attempts_remaining": remaining()}

    if not open_candidates:
        return fail("no_candidates", [])

    hashed = {k: v for k, v in factors.items() if k != OTP_FACTOR}
    matched = [c for c in open_candidates if _matches(c, hashed)]
    if not matched:
        return fail("mismatch", open_candidates)
    if len(matched) > 1:
        slots.release(slot)
        audit("verification_ambiguous", call_sid=call_sid, kind=kind,
              target_level=target_level, factor_names=sorted(factors),
              candidate_count=len(matched))
        return {"result": NOT_VERIFIED, "level": current_level, "reason": "ambiguous",
                "attempts_remaining": remaining()}
    customer = matched[0]

    if OTP_FACTOR in factors:
        to = otp_destination(deps, customer, ani)
        if deps.verify_client is None or to is None:
            slots.release(slot)
            audit("unavailable", call_sid=call_sid, reason="verify_not_configured")
            return {"result": UNAVAILABLE, "level": current_level}
        try:
            ok = deps.verify_client.check(to, factors[OTP_FACTOR])
        except VerifyUnavailable:
            slots.release(slot)
            audit("unavailable", call_sid=call_sid, reason="verify_check_failed",
                  decision="return_unavailable")
            return {"result": UNAVAILABLE, "level": current_level}
        if not ok:
            return fail("mismatch", [customer])

    customer_id = customer["customer_id"]
    new_token = tokens.issue(store, call_sid=call_sid, customer_id=customer_id,
                             level=target_level, now=now, max_ttl_s=deps.token_max_ttl_s)
    for old in active:
        try:
            store.update(tokens.TABLE, {"token_hash": old["token_hash"]}, {"revoked": True})
        except ConditionFailed:
            obs.emit(log, logging.DEBUG, "token_revoke_skipped",
                     decision="ignored_because_token_already_gone")
    slots.release(slot)
    audit("step_up" if step_up else "token_issued", call_sid=call_sid, customer_id=customer_id,
          level=target_level, previous_level=current_level, factor_names=sorted(factors),
          ani_trusted=trusted, ani_match_counted=ani_counts)
    return {"result": VERIFIED, "level": target_level, "customer_id": customer_id,
            "token": new_token,
            "factors_verified": policy.effective_factors(target_level, ani_counts=ani_counts),
            "step_up": step_up}


# --------------------------------------------------------------------------- precheck


def precheck(deps: AuthDeps, *, call_sid: Any, ani: str | None, target_level: Any,
             stir_verstat: str | None = None,
             identity_source: Any = IDENTITY_ANI) -> dict[str, Any]:
    """The untrusted-caller gate before the front door asks any factor.

    ``{"result": "ok"}`` (always, without touching the ``risk`` table, for a trusted
    caller), ``{"result": "escalate", "reason": "verification_risk"}``, or
    ``bad_request`` / ``unavailable``. ``verify`` re-applies the gate. A ``keyed_phone``
    ``identity_source`` is never trusted."""
    if not _valid_call_sid(call_sid):
        return _bad(call_sid, "call_sid")
    if not _valid_identity_source(identity_source):
        return _bad(call_sid, "identity_source")
    if not _is_int(target_level) or not 1 <= target_level <= deps.policy.max_level:
        return _bad(call_sid, "target_level")
    try:
        policy = deps.policy
        trusted, ani_counts = _trust(policy, clean_verstat(stir_verstat), identity_source)
        if trusted:
            return {"result": OK}
        now = deps.clock()
        active = _active_tokens(deps, call_sid, now)
        customer_ids = {t["customer_id"] for t in active}
        if len(customer_ids) > 1:
            audit("unavailable", call_sid=call_sid, reason="tokens_for_several_customers")
            return {"result": UNAVAILABLE}
        current_level = max((t["level"] for t in active), default=0)
        if target_level <= current_level:
            return {"result": OK}
        have = set(policy.effective_factors(current_level, ani_counts=ani_counts)) \
            | set(ESTABLISHED_FACTORS)
        required = [f for f in policy.effective_factors(target_level, ani_counts=ani_counts)
                    if f not in have]
        if active:
            cust = deps.store.get(CUSTOMERS, {"customer_id": next(iter(customer_ids))})
            candidates = [cust] if cust is not None and "alias_of" not in cust else []
        else:
            candidates = _candidates_for_ani(deps, ani)
        gate = _risk_gate(deps, call_sid=call_sid, candidates=candidates, step_up=bool(active),
                          required=required, now=now, current_level=current_level,
                          op="precheck")
        return gate or {"result": OK}
    except Exception as exc:
        obs.log_caught(log, "auth_service_operation_failed", exc, level=logging.ERROR, operation="precheck",
                       call_sid=call_sid if _valid_call_sid(call_sid) else None,
                       decision="return_unavailable")
        audit("unavailable", call_sid=call_sid, op="precheck")
        return {"result": UNAVAILABLE}


# --------------------------------------------------------------------------- send_otp


def send_otp(deps: AuthDeps, *, call_sid: Any, ani: str | None = None,
             language: str | None = None, stir_verstat: str | None = None,
             record: bool = False, identity_source: Any = IDENTITY_ANI) -> dict[str, Any]:
    """Send a Verify code to the call's verified customer (needs an active token).

    ``identity_source``: ``"ani"`` (default) or ``"keyed_phone"`` (reported as
    ``ani_trusted: false`` whatever the verstat); any other value is ``bad_request`` (recorded as
    ``error`` with ``record``). The destination rule (:func:`otp_destination`) is the same.

    ``record=True`` (the front door's asynchronous ``Event`` invoke, whose response nobody
    reads): once the outcome is decided, it is also written to the call state as
    ``otp_send = {"status": "sent" | "escalate" | "error", "at": <epoch s>}``, where the front
    door polls for it. Only the status and the time: never the code or a phone number. Without
    ``record`` nothing is written. The result is returned either way.

    Returns ``{"result": "sent" | "escalate" | "not_verified" | "rate_limited" |
    "unavailable" | "bad_request"}``. ``escalate`` carries ``reason: "otp_unavailable"``:
    the ANI is withheld, malformed or not on file for the customer (so no code is sent), or
    Verify could not deliver. The destination is never returned or logged.
    ``language`` (``en-US``, ``es-US``) selects the Verify ``Locale``.

    A code is sent whether or not the caller ID is trusted (by design: the caller has
    passed the factor questions). When ``stir_verstat`` doesn't make the ANI trusted, a
    stable ``otp_sent_untrusted_ani`` line goes to the ``owlbank.security`` logger, which
    the deployment alarms on (a spoofed caller ID aimed at a real customer): a warning, not
    a block. A send on a keyed-in number's call logs ``otp_sent_keyed_phone`` instead, which is
    counted but not alarmed on (its caller ID is never trusted, so the alarm would fire on each).
    """
    if not _valid_call_sid(call_sid):
        return _bad(call_sid, "call_sid")  # no call to record against
    if not _valid_identity_source(identity_source):
        out = _bad(call_sid, "identity_source")
    else:
        out = _send_otp(deps, call_sid=call_sid, ani=ani, language=language,
                        stir_verstat=stir_verstat, identity_source=identity_source)
    if record:
        _record_otp_send(deps, call_sid, out)
    return out


def otp_send_status(result: Mapping[str, Any]) -> str:
    """The ``otp_send`` status for a :func:`send_otp` result, as the front door's
    ``parse_send_otp`` reads the synchronous answer: ``sent``, ``escalate`` (otp_unavailable)
    or ``error`` (anything else)."""
    if result.get("result") == SENT:
        return SENT
    if result.get("result") == ESCALATE and result.get("reason") == OTP_UNAVAILABLE:
        return ESCALATE
    return "error"


def _record_otp_send(deps: AuthDeps, call_sid: str, result: Mapping[str, Any]) -> None:
    """Write the outcome to the call state for the front door. Never raises: a failed
    write means the front door times out and takes its error path."""
    status = otp_send_status(result)
    try:
        at = int(deps.clock())
        callstate.mutate(deps.store, call_sid, lambda _s: {callstate.F_OTP_SEND: {"status": status, "at": at}},
                         now=deps.clock)
    except Exception as exc:  # no call state, version conflicts exhausted, store down
        obs.log_caught(log, "otp_send_not_recorded", exc, level=logging.ERROR, operation="record_otp_send",
                       call_sid=call_sid, status=status, decision="frontdoor_times_out")
        return
    obs.emit(log, logging.INFO, "otp_send_recorded", call_sid=call_sid, status=status)


def _send_otp(deps: AuthDeps, *, call_sid: str, ani: str | None, language: str | None,
              stir_verstat: str | None, identity_source: str = IDENTITY_ANI) -> dict[str, Any]:
    try:
        now = deps.clock()
        active = _active_tokens(deps, call_sid, now)
        if not active:
            audit("verification_failed", call_sid=call_sid, kind="otp_send",
                  reason="no_active_token")
            return {"result": NOT_VERIFIED}
        customer_id = active[0]["customer_id"]
        customer = deps.store.get(CUSTOMERS, {"customer_id": customer_id})
        factor = deps.policy.factor(OTP_FACTOR) or {}
        channel = _VERIFY_CHANNELS.get(str(factor.get("channel", "")), "sms")
        to = otp_destination(deps, customer, ani) if customer is not None else None
        if to is None:
            return _otp_unavailable(call_sid, customer_id, "no_destination")
        if deps.verify_client is None:
            audit("unavailable", call_sid=call_sid, reason="verify_not_configured")
            return {"result": UNAVAILABLE}
        slot = _Slots(deps.store).reserve(f"ratelimit#call#{call_sid}#otpsend",
                                          _limit(deps.policy, "otp_sends_per_call"),
                                          now=now, ttl_s=deps.record_ttl_s)
        if slot is None:
            audit("rate_limited", call_sid=call_sid, scope="call", kind="otp_send")
            return {"result": RATE_LIMITED}
        try:
            deps.verify_client.send(to, channel, _clean_language(language))
        except VerifyUnavailable:
            return _otp_unavailable(call_sid, customer_id, "verify_send_failed")
        trusted, _ = _trust(deps.policy, clean_verstat(stir_verstat), identity_source)
        audit("otp_sent", call_sid=call_sid, customer_id=customer_id, channel=channel,
              ani_trusted=trusted)
        if identity_source == IDENTITY_KEYED_PHONE:  # counted, never the spoofed-ANI alarm
            security_log.otp_sent_keyed_phone()
        elif not trusted:
            security_log.otp_sent_untrusted_ani()
        return {"result": SENT}
    except Exception as exc:
        obs.log_caught(log, "auth_service_operation_failed", exc, level=logging.ERROR, operation="send_otp",
                       call_sid=call_sid if _valid_call_sid(call_sid) else None,
                       decision="return_unavailable")
        audit("unavailable", call_sid=call_sid, op="send_otp")
        return {"result": UNAVAILABLE}


def _otp_unavailable(call_sid: str, customer_id: str, reason: str) -> dict[str, Any]:
    """No code was (or could be) sent: the front door transfers the call to a human."""
    audit("otp_unavailable", call_sid=call_sid, customer_id=customer_id, reason=reason,
          decision="escalate")
    return {"result": ESCALATE, "reason": OTP_UNAVAILABLE}


# --------------------------------------------------------------------------- prepare_welcome


def _valid_customer_id(customer_id: object) -> TypeGuard[str]:
    return (isinstance(customer_id, str) and 0 < len(customer_id) <= _MAX_ID_LEN
            and "#" not in customer_id and customer_id.strip() == customer_id)


def _is_timeout(exc: BaseException) -> bool:
    """A socket, connect or read timeout (``requests`` or the standard library), by class."""
    return any(c.__name__ in {"Timeout", "TimeoutError", "ReadTimeout", "ConnectTimeout", "timeout"}
               for c in type(exc).__mro__)


class _MemoryOutcome(Exception):
    """Internal: a Memory step ended the preparation with ``reason``."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _memory_step(what: str, call: Callable[[], Any], *, call_sid: str, not_found: str) -> Any:
    """Run one Memory request; a failure becomes :class:`_MemoryOutcome`: ``memory_timeout``, a 404
    as ``not_found``, anything else ``memory_error``. Logs status and code only (never a body)."""
    try:
        return call()
    except Exception as exc:
        if _is_timeout(exc):
            reason = welcome.MEMORY_TIMEOUT
        elif isinstance(exc, memory_http.MemoryApiError) and exc.status == 404:
            reason = not_found
        else:
            reason = welcome.MEMORY_ERROR
        obs.log_caught(log, "welcome_memory_failed", exc, operation=what, call_sid=call_sid,
                       reason=reason, decision="fallback_welcome")
        raise _MemoryOutcome(reason) from None


def prepare_welcome(deps: AuthDeps, *, call_sid: Any, customer_id: Any, language: Any) -> dict[str, Any]:
    """Prepare the web-activity welcome for ``customer_id`` and write it to the call state.

    The front door's asynchronous ``Event`` invoke; nobody reads the
    answer, which is ``{"result": "ok" | "unavailable", "reason": <welcome reason>}`` (``unavailable``:
    nothing could be written, so the front door finds no prepared welcome and speaks today's), or
    ``bad_request``. Never raises.

    Reasons (:data:`owlbank.welcome.REASONS`): :func:`owlbank.welcome.prepare`'s ``matched``,
    ``no_web_activity`` (also when ``WebActivity`` isn't an allowed group: no Memory call),
    ``no_match``, ``no_first_name``; here ``no_profile`` (no such customer, no ``memory_email``, no
    Memory store configured, no profile or more than one for the email, or the profile read 404s:
    without a store no Memory call is made), ``memory_timeout`` and ``memory_error`` (any other
    failure, including an unexpected one).

    The record holds only the offer ID: the trait value never leaves this function, and no
    log line carries it."""
    if not _valid_call_sid(call_sid):
        return _bad(call_sid, "call_sid")
    if not _valid_customer_id(customer_id):
        return _bad(call_sid, "customer_id")
    lang = _clean_language(language)
    if lang is None:
        return _bad(call_sid, "language")
    started = time.perf_counter()
    trait_count = 0
    try:
        result, trait_count = _prepare_welcome(deps, call_sid=call_sid, customer_id=customer_id)
    except _MemoryOutcome as outcome:
        result = welcome.WelcomeResult(None, outcome.reason)
    except Exception as exc:
        obs.log_caught(log, "auth_service_operation_failed", exc, level=logging.ERROR,
                       operation="prepare_welcome", call_sid=call_sid, decision="record_memory_error")
        result = welcome.WelcomeResult(None, welcome.MEMORY_ERROR)
    duration_ms = int((time.perf_counter() - started) * 1000)
    recorded = _record_welcome(deps, call_sid=call_sid, customer_id=customer_id, language=lang,
                               result=result)
    with contextlib.suppress(Exception):  # logging must not turn into a raise from an Event invoke
        obs.emit(log, logging.INFO, "welcome_built", call_sid=call_sid, customer_id=customer_id,
                 language=lang, reason=result.reason, duration_ms=duration_ms, trait_count=trait_count,
                 recorded=recorded)
    return {"result": OK if recorded else UNAVAILABLE, "reason": result.reason}


def _prepare_welcome(deps: AuthDeps, *, call_sid: str,
                     customer_id: str) -> tuple[welcome.WelcomeResult, int]:
    """The result and the number of bounded traits it was made from."""
    if web_activity.GROUP not in deps.memory_trait_groups:
        return welcome.WelcomeResult(None, welcome.NO_WEB_ACTIVITY), 0
    customer = deps.store.get(CUSTOMERS, {"customer_id": customer_id})
    email = customer.get("memory_email") if customer is not None and "alias_of" not in customer else None
    if (customer is None or not isinstance(email, str) or not email.strip()
            or not deps.memory_store_id or deps.memory_client is None):
        return welcome.WelcomeResult(None, welcome.NO_PROFILE), 0
    store_id, timeout_s = deps.memory_store_id, deps.memory_timeout_s
    client = _memory_step("memory_client", deps.memory_client, call_sid=call_sid,
                          not_found=welcome.MEMORY_ERROR)
    ids = _memory_step("profile_lookup",
                       lambda: memory_http.lookup_profile(client, store_id, email, timeout_s=timeout_s),
                       call_sid=call_sid, not_found=welcome.MEMORY_ERROR)  # a 404 here: no such store
    if len(ids) != 1:  # none, or two profiles share the email: never guess
        obs.emit(log, logging.INFO, "welcome_profile_not_found", call_sid=call_sid, profile_count=len(ids),
                 decision="fallback_welcome")
        return welcome.WelcomeResult(None, welcome.NO_PROFILE), 0
    raw = _memory_step("profile_read",
                       lambda: memory_http.get_profile_traits(client, store_id, ids[0], [web_activity.GROUP],
                                                              timeout_s=timeout_s),
                       call_sid=call_sid, not_found=welcome.NO_PROFILE)
    bounded = web_activity.bound_profile_traits(
        raw, allow=deps.memory_trait_groups, now=datetime.fromtimestamp(deps.clock(), UTC),
        max_age_hours=deps.web_activity_max_age_hours)
    trait_count = sum(len(group) for group in bounded.values())
    return welcome.prepare(bounded, deps.policy.web_offers, customer.get("first_name")), trait_count


def _record_welcome(deps: AuthDeps, *, call_sid: str, customer_id: str, language: str,
                    result: welcome.WelcomeResult) -> bool:
    """Write ``welcome_prepared`` (the sealed offer ID or none, with the reason). Never raises: a
    failure means the front door finds nothing and speaks today's welcome."""
    try:
        if deps.sealer is None:
            obs.emit(log, logging.ERROR, "welcome_not_recorded", call_sid=call_sid, cause="no_sealer",
                     decision="fallback_welcome")
            return False
        record = callstate.seal_welcome(deps.sealer, result.offer_id, call_sid=call_sid,
                                        customer_id=customer_id, language=language, reason=result.reason,
                                        now=deps.clock())
        if record is None:  # sealing failed (logged by seal_welcome)
            return False
        callstate.mutate(deps.store, call_sid, lambda _s: {callstate.F_WELCOME_PREPARED: record},
                         now=deps.clock)
    except Exception as exc:  # no call state, version conflicts exhausted, store down
        with contextlib.suppress(Exception):  # never raise out of the Event invoke
            obs.log_caught(log, "welcome_not_recorded", exc, level=logging.ERROR,
                           operation="record_welcome", call_sid=call_sid, decision="fallback_welcome")
        return False
    return True


# --------------------------------------------------------------------------- event


def handle_event(deps: AuthDeps, event: Any) -> dict[str, Any]:
    """Dispatch a direct-invoke event (see ``handler.py`` for the contract)."""
    if isinstance(event, (str, bytes)):
        try:
            event = json.loads(event)
        except ValueError:
            return _bad(None, "event_json")
    if not isinstance(event, Mapping):
        return _bad(None, "event")
    op = event.get("op", "verify")
    call_sid = event.get("call_sid")
    if op == "verify":
        allowed = {"op", "call_sid", "ani", "factors", "target_level", "stir_verstat",
                   "identity_source"}
        if set(event) - allowed:
            return _bad(call_sid, "unexpected_fields")
        return verify(deps, call_sid=call_sid, ani=event.get("ani"),
                      factors=event.get("factors") or {}, target_level=event.get("target_level"),
                      stir_verstat=event.get("stir_verstat"),
                      identity_source=event.get("identity_source", IDENTITY_ANI))
    if op == "precheck":
        if set(event) - {"op", "call_sid", "ani", "target_level", "stir_verstat",
                         "identity_source"}:
            return _bad(call_sid, "unexpected_fields")
        return precheck(deps, call_sid=call_sid, ani=event.get("ani"),
                        target_level=event.get("target_level"),
                        stir_verstat=event.get("stir_verstat"),
                        identity_source=event.get("identity_source", IDENTITY_ANI))
    if op == "send_otp":
        if set(event) - {"op", "call_sid", "ani", "language", "stir_verstat", "record",
                         "identity_source"}:
            return _bad(call_sid, "unexpected_fields")
        return send_otp(deps, call_sid=call_sid, ani=event.get("ani"),
                        language=event.get("language"), stir_verstat=event.get("stir_verstat"),
                        record=event.get("record") is True,
                        identity_source=event.get("identity_source", IDENTITY_ANI))
    if op == "prepare_welcome":
        if set(event) - {"op", "call_sid", "customer_id", "language"}:
            return _bad(call_sid, "unexpected_fields")
        return prepare_welcome(deps, call_sid=call_sid, customer_id=event.get("customer_id"),
                               language=event.get("language"))
    return _bad(call_sid, "unknown_op")
