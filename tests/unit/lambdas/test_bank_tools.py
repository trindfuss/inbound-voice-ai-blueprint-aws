"""Unit and security tests for the mock bank API Lambda (handover "Done when")."""

from __future__ import annotations

from tests.logutil import lines as lines_of

import copy
import itertools
import json
import logging
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.bank_tools import handler as bank_handler
from lambdas.bank_tools import service
from lambdas.bank_tools.service import BankDeps
from owlbank import tokens
from owlbank.config import Settings
from owlbank.policy import Policy
from owlbank.store import SCHEMAS, DynamoStore, MemoryStore, Store, create_table_kwargs
from owlbank.tools_contract import IDEMPOTENT_OPS, auth_level_insufficient
from scripts.hash_tool import verify as verify_hash

DATA = Path(__file__).resolve().parents[3] / "data"
FIXTURES = json.loads((DATA / "fixtures.json").read_text(encoding="utf-8"))
POLICY_DOC = json.loads((DATA / "policy.json").read_text(encoding="utf-8"))
NOW = 1_700_000_000
THOMAS, SAM, JORDAN, MARIA = "C-1001", "C-1002", "C-1003", "C-1004"
NAMES = [c[k] for c in FIXTURES["customers"] for k in ("first_name", "last_name")]


class Clock:
    def __init__(self, t: int = NOW) -> None:
        self.t = t

    def __call__(self) -> int:
        return self.t


def make_deps(store: Store | None = None) -> BankDeps:
    if store is None:
        store = MemoryStore()
        for c in FIXTURES["customers"]:
            store.put("customers", copy.deepcopy(c))
    counter = iter(range(10_000_000, 99_999_999))
    return BankDeps(store=store, policy=Policy.from_dict(POLICY_DOC), clock=Clock(),
                    confirmation=lambda: str(next(counter)))


@pytest.fixture
def deps() -> BankDeps:
    return make_deps()


def token_for(deps: BankDeps, cid: str, level: int, call_sid: str = "CA1",
              ttl: int = 1800) -> str:
    return tokens.issue(deps.store, call_sid=call_sid, customer_id=cid, level=level,
                        now=deps.clock(), max_ttl_s=ttl)


_KEYS = itertools.count(1)


def call(deps: BankDeps, op: str, token: str | None, args: dict[str, Any] | None = None,
         call_sid: str = "CA1", **extra: Any) -> dict[str, Any]:
    """One bank_tools event. Money ops get a fresh ``idempotency_key`` unless one is given
    (``idempotency_key=None`` sends none)."""
    event: dict[str, Any] = {"op": op, "token": token, "call_sid": call_sid, "args": args or {}}
    if op in IDEMPOTENT_OPS and "idempotency_key" not in extra:
        event["idempotency_key"] = f"tu_{next(_KEYS)}"
    event.update({k: v for k, v in extra.items() if v is not None or k != "idempotency_key"})
    return service.handle_event(deps, event)


def customer(deps: BankDeps, cid: str) -> dict[str, Any]:
    item = deps.store.get("customers", {"customer_id": cid})
    assert item is not None
    return item


def acct(item: dict[str, Any], kind: str) -> dict[str, Any]:
    return next(a for a in item["accounts"] if a["type"] == kind)


# --------------------------------------------------------------------------- ani_lookup


@pytest.mark.parametrize(("ani", "ids"), [
    ("+12025550101", [THOMAS]),
    ("+12025550102", [SAM, JORDAN]),
    ("+12025550103", [MARIA]),
    ("+12025550199", []),
])
def test_ani_lookup(deps: BankDeps, ani: str, ids: list[str]) -> None:
    out = call(deps, "ani_lookup", None, {"ani": ani})
    assert out == {"tool": "ani_lookup", "data": {"candidates": ids, "withheld": False}}


@pytest.mark.parametrize("ani", ["anonymous", "", None, "restricted", "+1abc", "2025550101"])
def test_ani_lookup_withheld(deps: BankDeps, ani: Any) -> None:
    assert call(deps, "ani_lookup", None, {"ani": ani})["data"] == {"candidates": [],
                                                                    "withheld": True}


def test_ani_lookup_returns_no_names(deps: BankDeps) -> None:
    text = json.dumps(call(deps, "ani_lookup", None, {"ani": "+12025550102"}))
    assert not any(n in text for n in NAMES)


# --------------------------------------------------------------------------- happy paths


@pytest.mark.parametrize(("cid", "available", "pending"), [
    (THOMAS, "1234.56", "75.50"), (SAM, "-42.10", "0.00"), (JORDAN, "2950.75", "0.00"),
    (MARIA, "310.00", "0.00"),
])
def test_get_balance_each_fixture(deps: BankDeps, cid: str, available: str, pending: str) -> None:
    out = call(deps, "get_balance", token_for(deps, cid, 1), {"account": "checking"})
    data = out["data"]
    assert out["tool"] == "get_balance"
    assert data["available"] == available and data["pending"] == pending
    assert set(data) == {"accountType", "last4", "available", "current", "pending", "currency",
                         "asOf"}
    assert all(isinstance(v, str) for v in data.values())


def test_thomas_balance_matches_handover_example(deps: BankDeps) -> None:
    out = call(deps, "get_balance", token_for(deps, THOMAS, 1), {"account": "checking"})
    assert out["data"] == {"accountType": "checking", "last4": "4821", "available": "1234.56",
                           "current": "1310.06", "pending": "75.50", "currency": "USD",
                           "asOf": "2026-10-10T14:58:00Z"}


def test_maria_has_no_savings(deps: BankDeps) -> None:
    out = call(deps, "get_balance", token_for(deps, MARIA, 1), {"account": "savings"})
    assert out == {"error": "NOT_FOUND", "tool": "get_balance"}


def test_list_recent_transactions(deps: BankDeps) -> None:
    tok = token_for(deps, THOMAS, 1)
    out = call(deps, "list_recent_transactions", tok, {"account": "checking", "limit": 2})
    assert out["data"]["last4"] == "4821" and len(out["data"]["transactions"]) == 2
    assert out["data"]["transactions"][0]["description"] == "Coffee Example Co"
    jordan = call(deps, "list_recent_transactions", token_for(deps, JORDAN, 1, "CA2"),
                  {"account": "savings"}, call_sid="CA2")
    assert jordan["data"]["transactions"] == []


def test_change_address_needs_level_2_and_hashes_zip(deps: BankDeps) -> None:
    args = {"street": "1 New Street", "city": "Shelbyville", "state": "OR",
            "postal_code": "97035-1234"}
    assert call(deps, "change_address", token_for(deps, THOMAS, 1), args) \
        == auth_level_insufficient("change_address")
    out = call(deps, "change_address", token_for(deps, THOMAS, 2), args)
    assert out["data"] == {"status": "updated", "address": {"line1": "1 New Street",
                                                           "city": "Shelbyville", "state": "OR"}}
    item = customer(deps, THOMAS)
    assert "97035" not in json.dumps(item)
    assert verify_hash("97035", item["factors"]["zip"]) and item["version"] == 1


def test_make_payment_updates_balances_and_records_transaction(deps: BankDeps) -> None:
    tok = token_for(deps, THOMAS, 3)
    out = call(deps, "make_payment", tok,
               {"from_account": "checking", "to": "red", "amount": "250.00"})
    assert out["data"] == {
        "confirmation": "10000000", "amount": "250.00",
        "fromAccount": {"type": "checking", "last4": "4821", "available": "984.56"},
        "toAccount": {"type": "credit_card", "product": "red", "last4": "3309", "balance": "362.40"},
    }
    item = customer(deps, THOMAS)
    chk, card = acct(item, "checking"), acct(item, "credit_card")
    assert (chk["available"], chk["current"]) == ("984.56", "1060.06")
    assert (card["current"], card["available"]) == ("362.40", "4637.60")
    assert chk["transactions"][0] == {"date": "2023-11-14", "description":
                                      "Payment to credit card ending 3309", "amount": "-250.00",
                                      "status": "posted", "confirmation": "10000000"}
    assert card["transactions"][0]["amount"] == "250.00"


def test_transfer_funds(deps: BankDeps) -> None:
    tok = token_for(deps, JORDAN, 3)
    out = call(deps, "transfer_funds", tok,
               {"from_account": "savings", "to_account": "checking", "amount": "1000"})
    assert out["data"]["amount"] == "1000.00"
    item = customer(deps, JORDAN)
    assert acct(item, "savings")["available"] == "11000.00"
    assert acct(item, "checking")["available"] == "3950.75"


def test_insufficient_funds_changes_nothing(deps: BankDeps) -> None:
    before = customer(deps, SAM)
    out = call(deps, "make_payment", token_for(deps, SAM, 3),
               {"from_account": "checking", "to": "rewards", "amount": "10.00"})
    assert out == {"error": "INSUFFICIENT_FUNDS", "tool": "make_payment"}
    out = call(deps, "transfer_funds", token_for(deps, SAM, 3, "CA2"),
               {"from_account": "savings", "to_account": "checking", "amount": "15.01"},
               call_sid="CA2")
    assert out == {"error": "INSUFFICIENT_FUNDS", "tool": "transfer_funds"}
    assert customer(deps, SAM) == before


def test_payment_without_card_is_not_found(deps: BankDeps) -> None:
    out = call(deps, "make_payment", token_for(deps, MARIA, 3),
               {"from_account": "checking", "to": "red", "amount": "1.00"})
    assert out == {"error": "NOT_FOUND", "tool": "make_payment"}


# --------------------------------------------------------------------------- security suite


def test_no_token(deps: BankDeps) -> None:
    for tok in (None, ""):
        assert call(deps, "get_balance", tok, {"account": "checking"}) \
            == auth_level_insufficient("get_balance")


def test_fabricated_token(deps: BankDeps) -> None:
    assert call(deps, "get_balance", "made-up-token", {"account": "checking"}) \
        == auth_level_insufficient("get_balance")


def test_wrong_call_token(deps: BankDeps) -> None:
    tok = token_for(deps, THOMAS, 3, call_sid="CA-OTHER")
    assert call(deps, "get_balance", tok, {"account": "checking"}, call_sid="CA1") \
        == auth_level_insufficient("get_balance")


def test_expired_token(deps: BankDeps) -> None:
    tok = token_for(deps, THOMAS, 3, ttl=60)
    assert "data" in call(deps, "get_balance", tok, {"account": "checking"})
    deps.clock.t += 60  # type: ignore[attr-defined]
    assert call(deps, "get_balance", tok, {"account": "checking"}) \
        == auth_level_insufficient("get_balance")


def test_revoked_token(deps: BankDeps) -> None:
    tok = token_for(deps, THOMAS, 3)
    tokens.revoke_call(deps.store, "CA1")
    assert call(deps, "get_balance", tok, {"account": "checking"}) \
        == auth_level_insufficient("get_balance")


@pytest.mark.parametrize(("op", "args"), [
    ("make_payment", {"from_account": "checking", "to": "red", "amount": "1.00"}),
    ("transfer_funds", {"from_account": "checking", "to_account": "savings", "amount": "1.00"}),
    ("change_address", {"street": "x", "city": "y", "state": "CA", "postal_code": "12345"}),
])
def test_low_level_token(deps: BankDeps, op: str, args: dict[str, Any]) -> None:
    before = customer(deps, THOMAS)
    out = call(deps, op, token_for(deps, THOMAS, 1), args)
    assert out == {"error": "AUTH_LEVEL_INSUFFICIENT", "tool": op}
    assert customer(deps, THOMAS) == before


@pytest.mark.parametrize("extra", [
    {"args": {"account": "checking", "customer_id": SAM}},
    {"args": {"account": "checking", "customerId": SAM}},
    {"args": {"account": "checking", "account_id": "1957"}},
    {"args": {"account": "checking", "accountNumber": "1957"}},
    {"args": {"account": "checking", "nested": {"from_account_id": "1957"}}},
    {"customer_id": SAM},
])
def test_injected_customer_or_account_id(deps: BankDeps, extra: dict[str, Any],
                                         caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="owlbank.audit")
    event = {"op": "get_balance", "token": token_for(deps, THOMAS, 3), "call_sid": "CA1",
             "args": {"account": "checking"}, **extra}
    out = service.handle_event(deps, event)
    assert out == {"error": "INVALID_ARGUMENT", "tool": "get_balance"}
    assert any(r.getMessage() == "id_injection_rejected" for r in caplog.records)


def test_customer_comes_from_token_only(deps: BankDeps) -> None:
    """Sam and Jordan share a phone; each token sees only its own customer."""
    sam = call(deps, "get_balance", token_for(deps, SAM, 1), {"account": "checking"})
    jordan = call(deps, "get_balance", token_for(deps, JORDAN, 1, "CA2"),
                  {"account": "checking"}, call_sid="CA2")
    assert (sam["data"]["last4"], jordan["data"]["last4"]) == ("1957", "8813")


def test_audit_logs_rejections_without_tokens(deps: BankDeps,
                                              caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="owlbank.audit")
    tok = token_for(deps, THOMAS, 1, call_sid="CA-OTHER")
    call(deps, "get_balance", tok, {"account": "checking"})
    call(deps, "make_payment", token_for(deps, THOMAS, 3),
         {"from_account": "checking", "to": "red", "amount": "5.00"})
    msgs = lines_of(caplog.records, "owlbank.audit")
    assert [m["audit"] for m in msgs] == ["token_rejected", "customer_write"]
    assert msgs[0]["code"] == "WRONG_CALL"
    text = json.dumps(msgs, default=str)
    assert tok not in text and tokens.hash_token(tok) not in text


# --------------------------------------------------------------------------- bad input / errors


@pytest.mark.parametrize(("op", "args"), [
    ("get_balance", {"account": "brokerage"}),
    ("get_balance", {}),
    ("make_payment", {"from_account": "checking", "to": "red", "amount": 25.0}),
    ("make_payment", {"from_account": "checking", "to": "red", "amount": "0"}),
    ("make_payment", {"from_account": "checking", "to": "red", "amount": "-5"}),
    ("transfer_funds", {"from_account": "checking", "to_account": "checking", "amount": "1"}),
    ("list_recent_transactions", {"account": "checking", "limit": 50}),
])
def test_invalid_arguments(deps: BankDeps, op: str, args: dict[str, Any]) -> None:
    assert call(deps, op, token_for(deps, THOMAS, 3), args) \
        == {"error": "INVALID_ARGUMENT", "tool": op}


def test_unknown_operation_and_bad_events(deps: BankDeps) -> None:
    assert call(deps, "delete_everything", None)["error"] == "UNKNOWN_OPERATION"
    assert service.handle_event(deps, "not json")["error"] == "INVALID_ARGUMENT"
    assert service.handle_event(deps, ["op"])["error"] == "INVALID_ARGUMENT"
    assert service.handle_event(deps, {"op": "get_balance", "args": "x"})["error"] \
        == "INVALID_ARGUMENT"
    assert call(deps, "get_balance", None, {"account": "checking"}, extra_field=1)["error"] \
        == "INVALID_ARGUMENT"


def test_store_failure_is_unavailable(deps: BankDeps) -> None:
    class Broken(MemoryStore):
        def get(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("boom")

    deps.store = Broken()
    assert call(deps, "get_balance", "tok", {"account": "checking"}) \
        == {"error": "UNAVAILABLE", "tool": "get_balance"}


def test_missing_customer_is_not_found(deps: BankDeps) -> None:
    tok = token_for(deps, "C-9999", 1)
    assert call(deps, "get_balance", tok, {"account": "checking"})["error"] == "NOT_FOUND"


def test_write_conflict_retries_then_gives_up(deps: BankDeps) -> None:
    # A live (fresh) claim for v1 blocks the transition; after retries -> UNAVAILABLE.
    deps.store.put("calls", {"call_sid": "bankclaim#C-1001#v1", "claimed_at": NOW,
                             "expires_at": NOW + 3600})
    tok = token_for(deps, THOMAS, 3)
    args = {"from_account": "checking", "to": "red", "amount": "1.00"}
    assert call(deps, "make_payment", tok, args)["error"] == "UNAVAILABLE"
    deps.clock.t += 11  # type: ignore[attr-defined]  # the claim is now stale: taken over
    assert "data" in call(deps, "make_payment", tok, args)


def test_concurrent_writer_detected() -> None:
    """A write that lands between read and claim forces a re-read (no lost update)."""
    store = MemoryStore()
    for c in FIXTURES["customers"]:
        store.put("customers", copy.deepcopy(c))
    deps = make_deps(store)
    tok = token_for(deps, THOMAS, 3)
    real_put = store.put
    state = {"done": False}

    def racing_put(table: str, item: Any, *, condition: Any = None) -> None:
        if table == "calls" and str(item["call_sid"]).startswith("bankclaim") and not state["done"]:
            state["done"] = True  # another writer pays 1000.00 first
            other = store.get("customers", {"customer_id": THOMAS})
            acct(other, "checking")["available"] = "234.56"
            other["version"] = other.get("version", 0) + 1
            real_put("customers", other)
        real_put(table, item, condition=condition)

    store.put = racing_put  # type: ignore[method-assign]
    out = call(deps, "make_payment", tok,
               {"from_account": "checking", "to": "red", "amount": "300.00"})
    assert out == {"error": "INSUFFICIENT_FUNDS", "tool": "make_payment"}


# --------------------------------------------------------------------------- property


@settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(st.tuples(st.sampled_from(["checking", "savings"]),
                          st.decimals(min_value=Decimal("0.01"), max_value=Decimal("5000"),
                                      places=2)), max_size=8))
def test_transfers_conserve_money_and_never_overdraw(moves: list[tuple[str, Decimal]]) -> None:
    deps = make_deps()
    tok = token_for(deps, THOMAS, 3)
    start = customer(deps, THOMAS)
    total = Decimal(acct(start, "checking")["available"]) + Decimal(acct(start, "savings")["available"])
    for src, amount in moves:
        dst = "savings" if src == "checking" else "checking"
        before = Decimal(acct(customer(deps, THOMAS), src)["available"])
        out = call(deps, "transfer_funds", tok,
                   {"from_account": src, "to_account": dst, "amount": f"{amount:.2f}"})
        assert ("data" in out) == (amount <= before)
        item = customer(deps, THOMAS)
        assert Decimal(acct(item, src)["available"]) >= 0
        assert Decimal(acct(item, "checking")["available"]) \
            + Decimal(acct(item, "savings")["available"]) == total


# --------------------------------------------------------------------------- handler / DynamoDB


@pytest.fixture
def dynamo(monkeypatch: pytest.MonkeyPatch) -> Iterator[DynamoStore]:
    for var, val in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
                     "AWS_SESSION_TOKEN": "testing", "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(var, val)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    import boto3
    from moto import mock_aws

    cfg = Settings(aws_region="us-east-1")
    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-east-1")
        for logical in SCHEMAS:
            client.create_table(**create_table_kwargs(logical, cfg.table(logical)))
        store = DynamoStore.from_settings(cfg)
        for c in FIXTURES["customers"]:
            store.put("customers", copy.deepcopy(c))
        for item in Policy.to_items(POLICY_DOC):
            store.put("policy", item)
        yield store


def test_handler_on_dynamodb(dynamo: DynamoStore, monkeypatch: pytest.MonkeyPatch) -> None:
    deps = bank_handler.build_deps({"AWS_REGION": "us-east-1"})
    monkeypatch.setattr(bank_handler, "_deps", deps)
    import time as _time

    tok = tokens.issue(dynamo, call_sid="CA1", customer_id=THOMAS, level=3,
                       now=int(_time.time()), max_ttl_s=600)
    assert bank_handler.handler({"op": "ani_lookup", "args": {"ani": "+12025550102"}})["data"] \
        == {"candidates": [SAM, JORDAN], "withheld": False}
    out = bank_handler.handler({"op": "make_payment", "token": tok, "call_sid": "CA1",
                                "idempotency_key": "tu_dynamo_1",
                                "args": {"from_account": "checking", "to": "red",
                                         "amount": "12.34"}})
    assert out["data"]["fromAccount"]["available"] == "1222.22"
    bal = bank_handler.handler({"op": "get_balance", "token": tok, "call_sid": "CA1",
                                "args": {"account": "checking"}})
    assert bal["data"]["available"] == "1222.22"


# --------------------------------------------------------------------------- ANI aliases


def _alias(label: str, fid: str, phone: str) -> dict[str, Any]:
    """The alias record shape written by scripts/seed.py."""
    return {"customer_id": f"ani#{label}#{fid}", "phone": phone, "alias_of": fid,
            "kind": "ani_alias", "label": label}


def test_ani_lookup_resolves_aliases(deps: BankDeps) -> None:
    deps.store.put("customers", _alias("A", THOMAS, "+15005550006"))
    deps.store.put("customers", _alias("dev", THOMAS, "+15005550006"))  # same number twice
    deps.store.put("customers", _alias("B", SAM, "+15005550007"))
    deps.store.put("customers", _alias("B", JORDAN, "+15005550007"))
    one = call(deps, "ani_lookup", None, {"ani": "+15005550006"})
    assert one["data"] == {"candidates": [THOMAS], "withheld": False}
    lee = call(deps, "ani_lookup", None, {"ani": "+15005550007"})
    assert lee["data"]["candidates"] == [SAM, JORDAN]
    assert "ani#" not in json.dumps([one, lee])
    # the fictional fixture phone still works alongside its alias
    assert call(deps, "ani_lookup", None, {"ani": "+12025550101"})["data"]["candidates"] == [THOMAS]


def test_alias_record_is_never_served_as_a_customer(deps: BankDeps) -> None:
    deps.store.put("customers", _alias("A", THOMAS, "+15005550006"))
    tok = token_for(deps, "ani#A#C-1001", 3)
    assert call(deps, "get_balance", tok, {"account": "checking"})["error"] == "NOT_FOUND"
