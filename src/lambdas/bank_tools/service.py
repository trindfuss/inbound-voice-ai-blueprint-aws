"""Mock bank API logic (handover "Mock bank API").

One entry point, :func:`handle_event`, for the event
``{"op": ..., "token": ..., "call_sid": ..., "args": {...}}``.

Operations:

* ``ani_lookup`` (``args.ani``): candidate customer IDs for a caller ID. No token,
  no names. Used by the front door only (IAM). ANI alias records (``alias_of``,
  written by ``scripts/seed.py`` for real dev/test numbers) resolve to the fixture's
  ID, deduplicated; the alias record itself is never returned.
* Customer-data operations (token required): ``get_balance``,
  ``list_recent_transactions``, ``change_address``, ``make_payment``,
  ``transfer_funds``. Arguments follow :mod:`owlbank.tools_contract`.

Rules for every customer-data operation:

1. **No IDs from the caller.** Any ``customer_id``/``account_id``-like key, at any
   depth in ``args`` or at the top level of the event, is rejected
   (``INVALID_ARGUMENT``) and audited. Accounts are chosen by type only.
2. **Arguments are schema-checked** (``tools_contract.validate_args``).
3. **The token is validated on every call** (``tokens.validate``) against the
   policy's ``min_level`` for the operation (unknown operation: max level, fail
   closed). Any failure (missing, unknown, wrong call, revoked, expired, low
   level) returns exactly ``tools_contract.auth_level_insufficient(op)``: no data,
   no levels, no hint of which check failed. The specific code is audited.
4. **The customer comes from the token claims only.**

Responses: ``{"tool": op, "data": {...}}`` on success; ``{"error": CODE, "tool": op}``
otherwise, with CODE in :data:`ERROR_CODES`. Amounts are decimal strings.

Writes (``change_address``, ``make_payment``, ``transfer_funds``) are conditional:
the customer item carries a ``version``; a writer claims ``v+1`` with a
``not_exists`` marker item in the ``calls`` table (TTL), re-checks the head, then
writes with ``exists``. Funds are checked against the item read inside that
transition, so concurrent payments can't overdraw. Each payment or transfer
records a mock transaction with an 8-digit confirmation number.

**Idempotency.** ``make_payment`` and ``transfer_funds`` need a top-level
``idempotency_key`` (the model's ``tool_use_id``; ``[A-Za-z0-9_.:-]{1,128}``); without
one the answer is ``INVALID_ARGUMENT`` (bad request). After the token check, the
operation claims ``<CallSid>#idem#<key>`` in the ``calls`` table with a ``not_exists``
put (state ``pending``, the call's TTL), runs, and records the result (``done``). A
repeat of a ``done`` key returns the stored result without moving money again; a
repeat while the first is still ``pending`` gets ``UNAVAILABLE`` (it is never taken
over, so money can't move twice). A key reused for another operation or customer is
``INVALID_ARGUMENT``. If the operation fails before writing (``WriteConflict``, an
error) the claim is released so a retry can run.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from owlbank import obs, tokens
from owlbank.policy import Policy
from owlbank.store import ConditionFailed, Store
from owlbank.tools_contract import IDEMPOTENT_OPS, auth_level_insufficient, card_values, validate_args
from scripts.hash_tool import hash_factor

log = logging.getLogger(__name__)
audit_log = logging.getLogger("owlbank.audit")

NOT_FOUND = "NOT_FOUND"
UNAVAILABLE = "UNAVAILABLE"
INVALID_ARGUMENT = "INVALID_ARGUMENT"
INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
UNKNOWN_OPERATION = "UNKNOWN_OPERATION"
AUTH_LEVEL_INSUFFICIENT = "AUTH_LEVEL_INSUFFICIENT"
ERROR_CODES: frozenset[str] = frozenset({AUTH_LEVEL_INSUFFICIENT, NOT_FOUND, UNAVAILABLE,
                                         INVALID_ARGUMENT, INSUFFICIENT_FUNDS,
                                         UNKNOWN_OPERATION})

CUSTOMERS = "customers"
CALLS = "calls"
PHONE_INDEX = "phone-index"
CREDIT_CARD = "credit_card"
DEFAULT_TRANSACTIONS = 5
_EVENT_KEYS = frozenset({"op", "token", "call_sid", "args", "idempotency_key"})
_IDEM_KEY_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")
_IDEM_TTL_S = 86400  # fallback when the call has no state (the call's own TTL otherwise)
_CENT = Decimal("0.01")
_CLAIM_TTL_S = 3600
_CLAIM_TIMEOUT_S = 10
_WRITE_RETRIES = 5

#: Normalized key names (casefolded, ``_``/``-`` removed) that may never be inputs.
_FORBIDDEN_EXACT = frozenset({"accountid", "accountnumber", "accountno", "acctid", "acctno",
                              "acctnumber", "userid", "memoryuserid", "profileid", "cif"})
_FORBIDDEN_PREFIX = ("customer",)
_FORBIDDEN_SUFFIX = ("accountid", "accountnumber", "acctid", "customerid")


class WriteConflict(RuntimeError):
    """The customer record kept changing; give up (reported as ``UNAVAILABLE``)."""


class _Refused(Exception):
    """A business-rule refusal raised inside a write transition."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class BankDeps:
    store: Store
    policy: Policy
    clock: Callable[[], int] = field(default=lambda: int(time.time()))
    confirmation: Callable[[], str] = field(
        default=lambda: f"{secrets.randbelow(10**8):08d}")


def audit(event: str, **fields: Any) -> None:
    if "level" in fields:  # "level" is the log level in the JSON line
        fields["auth_level"] = fields.pop("level")
    obs.emit(audit_log, logging.INFO, event, audit=event, component="bank_tools", **fields)


def _err(code: str, op: Any) -> dict[str, Any]:
    return {"error": code, "tool": op if isinstance(op, str) and len(op) <= 64 else None}


def _ok(op: str, data: Mapping[str, Any]) -> dict[str, Any]:
    return {"tool": op, "data": dict(data)}


# --------------------------------------------------------------------------- ID injection


def is_forbidden_key(key: str) -> bool:
    norm = key.casefold().replace("_", "").replace("-", "")
    return (norm in _FORBIDDEN_EXACT or norm.startswith(_FORBIDDEN_PREFIX)
            or norm.endswith(_FORBIDDEN_SUFFIX))


def _find_forbidden(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            if not isinstance(k, str) or is_forbidden_key(k):
                return f"{path}.{k}"
            hit = _find_forbidden(v, f"{path}.{k}")
            if hit:
                return hit
    elif isinstance(value, list):
        for i, v in enumerate(value):
            hit = _find_forbidden(v, f"{path}[{i}]")
            if hit:
                return hit
    return None


# --------------------------------------------------------------------------- money


def _money(value: Any) -> Decimal:
    return Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP)


def _fmt(d: Decimal) -> str:
    d = d.quantize(_CENT, rounding=ROUND_HALF_UP)
    return f"{d + 0:.2f}" if d != 0 else "0.00"


def _account(customer: Mapping[str, Any], account_type: str) -> dict[str, Any] | None:
    """The deposit account of ``account_type`` (``checking``/``savings``); never a card."""
    for acct in customer.get("accounts") or []:
        if acct.get("type") == account_type and account_type != CREDIT_CARD:
            return acct
    return None


def _card(customer: Mapping[str, Any], product: str) -> dict[str, Any] | None:
    """The customer's card whose ``product`` is ``product`` (``make_payment.to``)."""
    for acct in customer.get("accounts") or []:
        if acct.get("type") == CREDIT_CARD and acct.get("product") == product:
            return acct
    return None


# --------------------------------------------------------------------------- ani_lookup


def is_withheld(ani: Any) -> bool:
    """True for a withheld or unusable caller ID (``anonymous``, ``restricted``, empty...)."""
    return not isinstance(ani, str) or not ani.startswith("+") or not ani[1:].isdigit() \
        or not 8 <= len(ani) <= 16


def ani_lookup(deps: BankDeps, ani: Any) -> dict[str, Any]:
    """Candidate customer IDs (sorted) for a caller ID. Never names or other data."""
    if is_withheld(ani):
        return _ok("ani_lookup", {"candidates": [], "withheld": True})
    items = deps.store.query(CUSTOMERS, {"phone": ani}, index=PHONE_INDEX)
    ids = sorted({cid for i in items if (cid := resolve_alias(i)) is not None})
    return _ok("ani_lookup", {"candidates": ids, "withheld": False})


def resolve_alias(item: Mapping[str, Any]) -> str | None:
    """The customer ID a phone-index hit stands for: ``alias_of`` for an ANI alias
    record (``scripts/seed.py``), else the record's own ID. Alias IDs are never returned."""
    target = item.get("alias_of")
    if target is not None:
        return target if isinstance(target, str) and target else None
    cid = item.get("customer_id")
    return cid if isinstance(cid, str) and cid else None


# --------------------------------------------------------------------------- reads


def get_balance(deps: BankDeps, customer: Mapping[str, Any], args: Mapping[str, Any]) -> dict:
    acct = _account(customer, args["account"])
    if acct is None:
        return _err(NOT_FOUND, "get_balance")
    return _ok("get_balance", {
        "accountType": acct["type"], "last4": acct.get("last4"),
        "available": acct.get("available"), "current": acct.get("current"),
        "pending": acct.get("pending", "0.00"), "currency": acct.get("currency", "USD"),
        "asOf": acct.get("asOf"),
    })


def list_recent_transactions(deps: BankDeps, customer: Mapping[str, Any],
                             args: Mapping[str, Any]) -> dict:
    acct = _account(customer, args["account"])
    if acct is None:
        return _err(NOT_FOUND, "list_recent_transactions")
    limit = args.get("limit", DEFAULT_TRANSACTIONS)
    txns = [{k: t.get(k) for k in ("date", "description", "amount", "status")}
            for t in (acct.get("transactions") or [])[:limit]]
    return _ok("list_recent_transactions",
               {"accountType": acct["type"], "last4": acct.get("last4"), "transactions": txns})


# --------------------------------------------------------------------------- writes


def _claim_key(customer_id: str, version: int) -> str:
    return f"bankclaim#{customer_id}#v{version}"


def _write_customer(deps: BankDeps, customer_id: str,
                    fn: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """Apply ``fn`` to the customer item as one versioned transition; return the new item.

    ``fn`` gets a copy of the current item, may raise :class:`_Refused`, and returns
    the new item. Retries on conflict; raises :class:`WriteConflict` after
    ``_WRITE_RETRIES`` attempts.
    """
    store = deps.store
    for _ in range(_WRITE_RETRIES):
        current = store.get(CUSTOMERS, {"customer_id": customer_id})
        if current is None:
            raise _Refused(NOT_FOUND)
        version = current.get("version", 0)
        new = fn(copy.deepcopy(current))
        new["version"] = version + 1
        now = deps.clock()
        claim = {"call_sid": _claim_key(customer_id, version + 1), "kind": "bank_write_claim",
                 "claimed_at": now, "expires_at": now + _CLAIM_TTL_S}
        try:
            store.put(CALLS, claim, condition="not_exists")
        except ConditionFailed:
            obs.emit(log, logging.DEBUG, "bank_write_claim_taken", decision="retry")
            existing = store.get(CALLS, {"call_sid": claim["call_sid"]})
            stale = existing is not None and now - int(existing.get("claimed_at", 0)) >= _CLAIM_TIMEOUT_S
            head = store.get(CUSTOMERS, {"customer_id": customer_id})
            if not (stale and head is not None and head.get("version", 0) == version):
                continue
            store.delete(CALLS, {"call_sid": claim["call_sid"]})  # the claimer crashed
            try:
                store.put(CALLS, claim, condition="not_exists")
            except ConditionFailed:
                obs.emit(log, logging.DEBUG, "bank_write_claim_taken", decision="retry")
                continue
        head = store.get(CUSTOMERS, {"customer_id": customer_id})
        if head is None or head.get("version", 0) != version:
            continue
        store.put(CUSTOMERS, new, condition="exists")
        return new
    raise WriteConflict(customer_id)


def _today(deps: BankDeps) -> str:
    return datetime.fromtimestamp(deps.clock(), UTC).date().isoformat()


def _move(deps: BankDeps, customer: Mapping[str, Any], op: str, src: str, dst: str,
          amount_s: str) -> dict:
    """Move ``amount`` from account ``src`` to ``dst``: a card's product for ``make_payment``
    (pays down its balance), a deposit account type for ``transfer_funds``."""
    amount = _money(amount_s)
    conf = deps.confirmation()
    date = _today(deps)
    find_dst = _card if op == "make_payment" else _account

    def apply(item: dict[str, Any]) -> dict[str, Any]:
        a, b = _account(item, src), find_dst(item, dst)
        if a is None or b is None:
            raise _Refused(NOT_FOUND)
        if _money(a.get("available", "0")) < amount:
            raise _Refused(INSUFFICIENT_FUNDS)
        a["available"] = _fmt(_money(a["available"]) - amount)
        a["current"] = _fmt(_money(a.get("current", a["available"])) - amount)
        if b["type"] == CREDIT_CARD:  # paying the card lowers what is owed and frees credit
            b["current"] = _fmt(_money(b.get("current", "0")) - amount)
            b["available"] = _fmt(_money(b.get("available", "0")) + amount)
            desc_a = f"Payment to credit card ending {b.get('last4')}"
            desc_b = "Payment received, thank you"
        else:
            b["available"] = _fmt(_money(b.get("available", "0")) + amount)
            b["current"] = _fmt(_money(b.get("current", "0")) + amount)
            desc_a = f"Transfer to {dst}"
            desc_b = f"Transfer from {src}"
        for acct, desc, signed in ((a, desc_a, -amount), (b, desc_b, amount)):
            acct["transactions"] = [{"date": date, "description": desc, "amount": _fmt(signed),
                                     "status": "posted", "confirmation": conf}] + list(
                acct.get("transactions") or [])
        return item

    new = _write_customer(deps, customer["customer_id"], apply)
    a, b = _account(new, src), find_dst(new, dst)
    if a is None or b is None:  # data invariant: both accounts exist in the record we just wrote
        raise RuntimeError("transfer accounts missing after write")
    to_account: dict[str, Any]
    if b["type"] == CREDIT_CARD:
        to_account = {"type": CREDIT_CARD, "product": dst, "last4": b.get("last4"),
                      "balance": b["current"]}
    else:
        to_account = {"type": dst, "last4": b.get("last4"), "available": b["available"]}
    return _ok(op, {
        "confirmation": conf, "amount": _fmt(amount),
        "fromAccount": {"type": src, "last4": a.get("last4"), "available": a["available"]},
        "toAccount": to_account,
    })


def make_payment(deps: BankDeps, customer: Mapping[str, Any], args: Mapping[str, Any]) -> dict:
    """Pay the card named by ``to``, the card the caller confirmed."""
    return _move(deps, customer, "make_payment", args["from_account"], args["to"], args["amount"])


def transfer_funds(deps: BankDeps, customer: Mapping[str, Any], args: Mapping[str, Any]) -> dict:
    return _move(deps, customer, "transfer_funds", args["from_account"], args["to_account"],
                 args["amount"])


def change_address(deps: BankDeps, customer: Mapping[str, Any], args: Mapping[str, Any]) -> dict:
    """Update the mailing address. The ZIP is a verification factor, so it is stored
    only as a new salted hash (``factors.zip``), never in the address or the response."""
    zip5 = args["postal_code"][:5]

    def apply(item: dict[str, Any]) -> dict[str, Any]:
        item["address"] = {"line1": args["street"], "city": args["city"], "state": args["state"]}
        item["factors"] = {**(item.get("factors") or {}), "zip": hash_factor(zip5)}
        return item

    new = _write_customer(deps, customer["customer_id"], apply)
    return _ok("change_address", {"status": "updated", "address": dict(new["address"])})


#: Customer-data operations (tool name -> implementation).
OPERATIONS: Mapping[str, Callable[[BankDeps, Mapping[str, Any], Mapping[str, Any]], dict]] = {
    "get_balance": get_balance,
    "list_recent_transactions": list_recent_transactions,
    "change_address": change_address,
    "make_payment": make_payment,
    "transfer_funds": transfer_funds,
}


# --------------------------------------------------------------------------- idempotency


def idem_record_key(call_sid: str, key: str) -> str:
    return f"{call_sid}#idem#{key}"


def _run_once(deps: BankDeps, call_sid: str, key: str, op: str, customer_id: str,
              run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run ``run`` at most once per ``(call_sid, key)``; replay the stored result after."""
    store = deps.store
    rec_key = {"call_sid": idem_record_key(call_sid, key)}

    def replay(rec: Mapping[str, Any]) -> dict[str, Any]:
        if rec.get("op") != op or rec.get("customer_id") != customer_id:
            audit("idempotency_key_reused", call_sid=call_sid, op=op)
            return _err(INVALID_ARGUMENT, op)
        if rec.get("state") == "done" and isinstance(rec.get("result"), Mapping):
            audit("idempotent_replay", call_sid=call_sid, op=op)
            return copy.deepcopy(dict(rec["result"]))
        return _err(UNAVAILABLE, op)  # the first attempt is still running (or died mid-way)

    existing = store.get(CALLS, rec_key)
    if existing is not None:
        return replay(existing)
    now = deps.clock()
    state = store.get(CALLS, {"call_sid": call_sid})
    expires = state.get("expires_at") if state else None
    expires_at = expires if isinstance(expires, int) and not isinstance(expires, bool) \
        else now + _IDEM_TTL_S
    claim = {**rec_key, "kind": "idempotency", "op": op, "customer_id": customer_id,
             "state": "pending", "claimed_at": now, "expires_at": expires_at}
    try:
        store.put(CALLS, claim, condition="not_exists")
    except ConditionFailed:
        obs.emit(log, logging.DEBUG, "idempotency_claim_exists", op=op, decision="replay")
        current = store.get(CALLS, rec_key)
        return replay(current) if current is not None else _err(UNAVAILABLE, op)
    try:
        result = run()
    except _Refused as r:
        obs.emit(log, logging.DEBUG, "business_refusal", op=op, code=r.code,
                 decision="ignored_because_final_business_answer")
        result = _err(r.code, op)  # a business refusal is a final answer for this key
    except BaseException as exc:
        obs.log_caught(log, "idempotent_operation_failed", exc, op=op, decision="release_claim_and_raise")
        store.delete(CALLS, rec_key)  # nothing was written: let a retry run
        raise
    store.put(CALLS, {**claim, "state": "done", "result": copy.deepcopy(result)})
    return result


# --------------------------------------------------------------------------- dispatch


def handle_event(deps: BankDeps, event: Any) -> dict[str, Any]:
    """Run one operation. Never raises; see the module docstring for the contract."""
    if isinstance(event, (str, bytes)):
        try:
            event = json.loads(event)
        except ValueError:
            obs.emit(log, logging.WARNING, "bad_event", reason="not_json", decision="return_invalid_argument")
            return _err(INVALID_ARGUMENT, None)
    if not isinstance(event, Mapping):
        return _err(INVALID_ARGUMENT, None)
    op = event.get("op")
    call_sid = event.get("call_sid")
    call_ref = call_sid if isinstance(call_sid, str) and len(call_sid) <= 64 else None
    args = event.get("args", {})
    if args is None:
        args = {}
    hit = _find_forbidden({k: v for k, v in event.items() if k != "args"}) or (
        _find_forbidden(args, "$.args") if isinstance(args, Mapping) else None)
    if hit:
        audit("id_injection_rejected", call_sid=call_ref, op=op if isinstance(op, str) else None,
              path=hit[:128])
        return _err(INVALID_ARGUMENT, op)
    if set(event) - _EVENT_KEYS or not isinstance(args, Mapping):
        return _err(INVALID_ARGUMENT, op)

    try:
        if op == "ani_lookup":
            if set(args) - {"ani"}:
                return _err(INVALID_ARGUMENT, op)
            return ani_lookup(deps, args.get("ani"))
        impl = OPERATIONS.get(op) if isinstance(op, str) else None
        if not isinstance(op, str) or impl is None:
            audit("unknown_operation", call_sid=call_ref)
            return _err(UNKNOWN_OPERATION, op)
        if validate_args(op, args, cards=card_values(deps.policy)):
            return _err(INVALID_ARGUMENT, op)
        if op == "make_payment" and "to" not in args:
            # Optional for the model (the front door asks the card); the bank needs it.
            return _err(INVALID_ARGUMENT, op)
        idem_key = event.get("idempotency_key")
        if op in IDEMPOTENT_OPS:
            if not isinstance(idem_key, str) or not _IDEM_KEY_RE.match(idem_key) or not call_ref:
                audit("idempotency_key_missing", call_sid=call_ref, op=op)
                return _err(INVALID_ARGUMENT, op)
        elif idem_key is not None:
            return _err(INVALID_ARGUMENT, op)
        claims = tokens.validate(deps.store, event.get("token"), call_sid=call_ref or "",
                                 min_level=deps.policy.min_level(op), now=deps.clock())
        if isinstance(claims, tokens.AuthError):
            audit("token_rejected", call_sid=call_ref, op=op, code=str(claims.code))
            return auth_level_insufficient(op)
        customer = deps.store.get(CUSTOMERS, {"customer_id": claims.customer_id})
        if customer is None or "alias_of" in customer:
            return _err(NOT_FOUND, op)
        if op in IDEMPOTENT_OPS:
            result = _run_once(deps, call_ref or "", str(idem_key), op, claims.customer_id,
                               lambda: impl(deps, customer, args))
        else:
            try:
                result = impl(deps, customer, args)
            except _Refused as r:
                obs.emit(log, logging.DEBUG, "business_refusal", op=op, code=r.code,
                         decision="ignored_because_final_business_answer")
                return _err(r.code, op)
        if "error" not in result and op in ("change_address", "make_payment", "transfer_funds"):
            audit("customer_write", call_sid=call_ref, op=op, customer_id=claims.customer_id,
                  level=claims.level)
        return result
    except Exception as exc:
        obs.log_caught(log, "bank_tools_operation_failed", exc, level=logging.ERROR,
                       op=op if isinstance(op, str) else None, call_sid=call_ref,
                       decision="return_unavailable")
        return _err(UNAVAILABLE, op)
