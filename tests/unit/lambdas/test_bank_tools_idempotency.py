"""Money movement is idempotent per ``(CallSid, idempotency_key)``."""

from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.bank_tools import service
from owlbank import callstate
from owlbank.callstate import CallState
from owlbank.store import MemoryStore

from tests.unit.lambdas.test_bank_tools import (
    FIXTURES,
    NOW,
    THOMAS,
    acct,
    call,
    customer,
    make_deps,
    token_for,
)

PAY = {"from_account": "checking", "to": "red", "amount": "10.00"}
XFER = {"from_account": "checking", "to_account": "savings", "amount": "5.00"}


def balances(deps: service.BankDeps) -> tuple[str, ...]:
    item = customer(deps, THOMAS)
    return tuple(acct(item, k)["available"] for k in ("checking", "savings", "credit_card"))


def txn_count(deps: service.BankDeps) -> int:
    return len(acct(customer(deps, THOMAS), "checking").get("transactions") or [])


@pytest.mark.parametrize("op,args", [("make_payment", PAY), ("transfer_funds", XFER)])
@pytest.mark.parametrize("key", [None, "", "has#hash", "x" * 129, 12])
def test_missing_or_bad_key_is_a_bad_request(op: str, args: dict, key: Any) -> None:
    deps = make_deps()
    before = balances(deps)
    # key=None: the helper sends no key at all
    out = call(deps, op, token_for(deps, THOMAS, 3), args, idempotency_key=key)
    assert out == {"error": "INVALID_ARGUMENT", "tool": op}
    assert balances(deps) == before


def test_key_on_a_read_is_rejected() -> None:
    deps = make_deps()
    out = call(deps, "get_balance", token_for(deps, THOMAS, 3), {"account": "checking"},
               idempotency_key="tu_1")
    assert out == {"error": "INVALID_ARGUMENT", "tool": "get_balance"}


def test_repeat_returns_the_stored_result_and_moves_money_once() -> None:
    deps = make_deps()
    tok = token_for(deps, THOMAS, 3)
    first = call(deps, "make_payment", tok, PAY, idempotency_key="tu_pay")
    after = balances(deps)
    again = call(deps, "make_payment", tok, PAY, idempotency_key="tu_pay")
    assert again == first and "data" in first
    assert balances(deps) == after
    rec = deps.store.get("calls", {"call_sid": "CA1#idem#tu_pay"})
    assert rec["state"] == "done" and rec["kind"] == "idempotency"
    # A new key is a new payment.
    assert call(deps, "make_payment", tok, PAY, idempotency_key="tu_pay2") != first
    assert balances(deps) != after


def test_record_uses_the_call_ttl() -> None:
    deps = make_deps()
    callstate.create(deps.store, CallState(call_sid="CA1", expires_at=NOW + 1234))
    call(deps, "make_payment", token_for(deps, THOMAS, 3), PAY, idempotency_key="tu_ttl")
    assert deps.store.get("calls", {"call_sid": "CA1#idem#tu_ttl"})["expires_at"] == NOW + 1234


def test_key_is_scoped_to_the_call_and_the_operation() -> None:
    deps = make_deps()
    tok = token_for(deps, THOMAS, 3)
    call(deps, "make_payment", tok, PAY, idempotency_key="tu_same")
    assert call(deps, "transfer_funds", tok, XFER, idempotency_key="tu_same") == \
        {"error": "INVALID_ARGUMENT", "tool": "transfer_funds"}
    tok2 = token_for(deps, THOMAS, 3, "CA2")
    out = call(deps, "make_payment", tok2, PAY, call_sid="CA2", idempotency_key="tu_same")
    assert "data" in out  # another call: an independent key space


def test_refusal_is_final_but_a_crash_releases_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = make_deps()
    tok = token_for(deps, THOMAS, 3)
    big = {**PAY, "amount": "999999.00"}
    assert call(deps, "make_payment", tok, big, idempotency_key="tu_big")["error"] == "INSUFFICIENT_FUNDS"
    assert call(deps, "make_payment", tok, big, idempotency_key="tu_big")["error"] == "INSUFFICIENT_FUNDS"

    real = service._write_customer
    monkeypatch.setattr(service, "_write_customer",
                        lambda *a, **k: (_ for _ in ()).throw(service.WriteConflict("x")))
    before = balances(deps)
    assert call(deps, "make_payment", tok, PAY, idempotency_key="tu_retry")["error"] == "UNAVAILABLE"
    assert balances(deps) == before
    assert deps.store.get("calls", {"call_sid": "CA1#idem#tu_retry"}) is None  # released
    monkeypatch.setattr(service, "_write_customer", real)
    assert "data" in call(deps, "make_payment", tok, PAY, idempotency_key="tu_retry")


def test_double_submit_inside_the_first_attempt_is_refused() -> None:
    """The second submit lands while the first holds the claim: UNAVAILABLE, one debit."""
    store = MemoryStore()
    for c in FIXTURES["customers"]:
        store.put("customers", copy.deepcopy(c))
    deps = make_deps(store)
    tok = token_for(deps, THOMAS, 3)
    real_put = store.put
    seen: dict[str, Any] = {}

    def racing_put(table: str, item: Any, *, condition: Any = None) -> None:
        real_put(table, item, condition=condition)
        if table == "calls" and item.get("kind") == "idempotency" and item["state"] == "pending" \
                and "second" not in seen:
            seen["second"] = call(deps, "make_payment", tok, PAY, idempotency_key="tu_dup")

    store.put = racing_put  # type: ignore[method-assign]
    before = Decimal(acct(customer(deps, THOMAS), "checking")["available"])
    first = call(deps, "make_payment", tok, PAY, idempotency_key="tu_dup")
    assert "data" in first and seen["second"] == {"error": "UNAVAILABLE", "tool": "make_payment"}
    after = Decimal(acct(customer(deps, THOMAS), "checking")["available"])
    assert before - after == Decimal("10.00")
    assert call(deps, "make_payment", tok, PAY, idempotency_key="tu_dup") == first


class LockedStore(MemoryStore):
    """MemoryStore with one lock per operation, standing in for DynamoDB's atomic writes."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()

    def get(self, *a: Any, **k: Any) -> Any:
        with self._lock:
            return super().get(*a, **k)

    def put(self, *a: Any, **k: Any) -> Any:
        with self._lock:
            return super().put(*a, **k)

    def delete(self, *a: Any, **k: Any) -> Any:
        with self._lock:
            return super().delete(*a, **k)

    def query(self, *a: Any, **k: Any) -> Any:
        with self._lock:
            return super().query(*a, **k)


def test_concurrent_double_submit_gives_exactly_one_debit() -> None:
    store = LockedStore()
    for c in FIXTURES["customers"]:
        store.put("customers", copy.deepcopy(c))
    deps = make_deps(store)
    tok = token_for(deps, THOMAS, 3)
    before = Decimal(acct(customer(deps, THOMAS), "checking")["available"])
    txns = txn_count(deps)
    barrier = threading.Barrier(8)

    def submit(_: int) -> dict[str, Any]:
        barrier.wait()
        return call(deps, "make_payment", tok, PAY, idempotency_key="tu_conc")

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(submit, range(8)))
    ok = [r for r in results if "data" in r]
    assert ok and all(r == ok[0] for r in ok)
    assert all(r == {"error": "UNAVAILABLE", "tool": "make_payment"} for r in results if "data" not in r)
    after = Decimal(acct(customer(deps, THOMAS), "checking")["available"])
    assert before - after == Decimal("10.00")
    assert txn_count(deps) == txns + 1


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(retries=st.integers(min_value=1, max_value=12),
       amount=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("100"), places=2),
       op=st.sampled_from(["make_payment", "transfer_funds"]))
def test_property_any_number_of_retries_changes_balances_once(retries: int, amount: Decimal,
                                                              op: str) -> None:
    """PBT: N submits of one key == one submit."""
    deps, ref = make_deps(), make_deps()
    args = {**(PAY if op == "make_payment" else XFER), "amount": f"{amount:.2f}"}
    tok = token_for(deps, THOMAS, 3)
    results = [call(deps, op, tok, args, idempotency_key="tu_prop") for _ in range(retries)]
    once = call(ref, op, token_for(ref, THOMAS, 3), args, idempotency_key="tu_prop")
    assert all(r == results[0] for r in results)
    assert results[0] == once
    assert balances(deps) == balances(ref)
    assert txn_count(deps) == txn_count(ref)
