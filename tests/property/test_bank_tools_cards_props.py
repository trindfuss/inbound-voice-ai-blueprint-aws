"""Bank-tool half: ``make_payment`` never pays a card that isn't the
caller's card named by ``to``.

For any fixture customer, any ``to`` (any string, the card values weighted in), any source
account and any amount, ``make_payment`` either credits exactly the customer's card whose
``product == to`` (and debits only the source account, by the amount) or fails with no change
to the customer record.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.unit.lambdas.test_bank_tools import FIXTURES, call, customer, make_deps, token_for

CUSTOMER_IDS = [c["customer_id"] for c in FIXTURES["customers"]]
CARD_VALUES = ["red", "purple", "rewards"]

to_values = st.one_of(st.sampled_from([*CARD_VALUES, "credit_card", "checking", "savings"]),
                      st.text(max_size=12))
amounts = st.one_of(st.sampled_from(["0.01", "1.00", "25.00", "1000", "99999.99"]),
                    st.decimals(min_value=Decimal("0.01"), max_value=Decimal("20000"), places=2)
                    .map(lambda d: f"{d:.2f}"))


def _key(a: dict[str, Any]) -> tuple[str, str | None]:
    return a["type"], a.get("product")


def _accounts(item: dict[str, Any]) -> dict[tuple[str, str | None], dict[str, Any]]:
    return {_key(a): a for a in item["accounts"]}


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(cid=st.sampled_from(CUSTOMER_IDS), to=to_values,
       src=st.sampled_from(["checking", "savings"]), amount=amounts)
def test_make_payment_credits_exactly_the_named_card_or_changes_nothing(
        cid: str, to: str, src: str, amount: str) -> None:
    deps = make_deps()
    before = customer(deps, cid)
    out = call(deps, "make_payment", token_for(deps, cid, 3),
               {"from_account": src, "to": to, "amount": amount})
    after = customer(deps, cid)
    if "error" in out:
        assert after == before
        return
    card_key = ("credit_card", to)
    b, a = _accounts(before), _accounts(after)
    assert card_key in b, "paid a card the customer doesn't have"
    assert out["data"]["toAccount"]["product"] == to
    assert out["data"]["toAccount"]["last4"] == b[card_key]["last4"]
    amt = Decimal(amount)
    assert Decimal(a[card_key]["current"]) == Decimal(b[card_key]["current"]) - amt
    assert Decimal(a[(src, None)]["available"]) == Decimal(b[(src, None)]["available"]) - amt
    for k in b:  # every other account is untouched
        if k not in (card_key, (src, None)):
            assert a[k] == b[k], k
