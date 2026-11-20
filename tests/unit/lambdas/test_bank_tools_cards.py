"""``make_payment`` pays the customer's card whose ``product`` is ``to``."""

from __future__ import annotations

import pytest

from lambdas.bank_tools.service import BankDeps

from tests.unit.lambdas.test_bank_tools import (
    FIXTURES,
    JORDAN,
    MARIA,
    SAM,
    THOMAS,
    call,
    customer,
    make_deps,
    token_for,
)


@pytest.fixture
def deps() -> BankDeps:
    return make_deps()


def cards(item: dict) -> dict[str, dict]:
    return {a["product"]: a for a in item["accounts"] if a["type"] == "credit_card"}


def test_fixture_cards() -> None:
    by_id = {c["customer_id"]: c for c in FIXTURES["customers"]}
    thomas = cards(by_id[THOMAS])
    assert set(thomas) == {"red", "purple"}
    assert thomas["red"]["nickname"] == "OwlBank Red Card" and thomas["red"]["last4"] == "3309"
    assert thomas["purple"]["nickname"] == "OwlBank Purple Card"
    assert thomas["purple"]["last4"] not in {"3309", "4821", "7730"}
    sam = cards(by_id[SAM])
    assert set(sam) == {"rewards"} and sam["rewards"]["nickname"] == "OwlBank Rewards Card"
    assert cards(by_id[JORDAN]) == {} and cards(by_id[MARIA]) == {}
    last4s = [a["last4"] for c in FIXTURES["customers"] for a in c["accounts"]]
    assert len(last4s) == len(set(last4s))  # every fictional last4 is distinct


def test_payment_to_purple_moves_money_to_the_purple_card(deps: BankDeps) -> None:
    before = cards(customer(deps, THOMAS))
    out = call(deps, "make_payment", token_for(deps, THOMAS, 3),
               {"from_account": "checking", "to": "purple", "amount": "25.00"})
    purple = before["purple"]
    assert out["data"]["toAccount"] == {
        "type": "credit_card", "product": "purple", "last4": purple["last4"],
        "balance": f"{float(purple['current']) - 25:.2f}"}
    after = cards(customer(deps, THOMAS))
    assert after["red"] == before["red"]  # the other card is untouched
    assert after["purple"]["current"] == out["data"]["toAccount"]["balance"]
    assert after["purple"]["transactions"][0]["amount"] == "25.00"
    checking = next(a for a in customer(deps, THOMAS)["accounts"] if a["type"] == "checking")
    assert checking["transactions"][0]["description"] == f"Payment to credit card ending {purple['last4']}"


def test_payment_to_red_moves_money_to_the_red_card(deps: BankDeps) -> None:
    before = cards(customer(deps, THOMAS))
    out = call(deps, "make_payment", token_for(deps, THOMAS, 3),
               {"from_account": "checking", "to": "red", "amount": "10.00"})
    assert out["data"]["toAccount"]["last4"] == "3309"
    after = cards(customer(deps, THOMAS))
    assert after["purple"] == before["purple"]
    assert after["red"]["current"] == "602.40"


def test_rewards_for_thomas_is_an_error_and_moves_nothing(deps: BankDeps) -> None:
    before = customer(deps, THOMAS)
    out = call(deps, "make_payment", token_for(deps, THOMAS, 3),
               {"from_account": "checking", "to": "rewards", "amount": "1.00"})
    assert out == {"error": "NOT_FOUND", "tool": "make_payment"}
    assert customer(deps, THOMAS) == before


@pytest.mark.parametrize("to", ["credit_card", "checking", "savings", "blue", ""])
def test_unknown_card_is_invalid_and_moves_nothing(deps: BankDeps, to: str) -> None:
    before = customer(deps, THOMAS)
    out = call(deps, "make_payment", token_for(deps, THOMAS, 3),
               {"from_account": "savings", "to": to, "amount": "1.00"})
    assert out == {"error": "INVALID_ARGUMENT", "tool": "make_payment"}
    assert customer(deps, THOMAS) == before


def test_payment_without_to_is_invalid_and_moves_nothing(deps: BankDeps) -> None:
    before = customer(deps, THOMAS)
    out = call(deps, "make_payment", token_for(deps, THOMAS, 3),
               {"from_account": "checking", "amount": "1.00"})
    assert out == {"error": "INVALID_ARGUMENT", "tool": "make_payment"}
    assert customer(deps, THOMAS) == before


def test_the_bank_checks_to_against_the_policy_cards(deps: BankDeps) -> None:
    """A card option the policy doesn't list is refused even if the record has such a card."""
    item = customer(deps, THOMAS)
    for a in item["accounts"]:
        if a.get("product") == "purple":
            a["product"] = "blue"
    deps.store.put("customers", item)
    out = call(deps, "make_payment", token_for(deps, THOMAS, 3),
               {"from_account": "checking", "to": "blue", "amount": "1.00"})
    assert out == {"error": "INVALID_ARGUMENT", "tool": "make_payment"}


def test_transfers_never_reach_a_card(deps: BankDeps) -> None:
    out = call(deps, "transfer_funds", token_for(deps, JORDAN, 3),
               {"from_account": "savings", "to_account": "checking", "amount": "1.00"})
    assert out["data"]["toAccount"] == {"type": "checking", "last4": "8813", "available": "2951.75"}
