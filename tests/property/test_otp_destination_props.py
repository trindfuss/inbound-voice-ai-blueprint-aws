"""A one-time code goes only to the calling number, and only to a number
on file for the verified customer.

For any ANI (well-formed or not, anyone's number) and any customer data:

* ``otp_destination`` is ``None`` or exactly the ANI, and the ANI is E.164 and is the
  customer's own phone or the phone of an ``alias_of`` record for that customer;
* it is never the customer's phone on file when that differs from the ANI;
* ``send_otp`` sends to nothing but the ANI, and when ``otp_destination`` is ``None`` it sends
  nothing and answers ``escalate`` / ``otp_unavailable``.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from lambdas.auth_service import service
from lambdas.auth_service.service import AuthDeps
from lambdas.auth_service.verify_client import VerifyUnavailable
from owlbank import tokens
from owlbank.policy import Policy
from owlbank.store import MemoryStore

DATA = Path(__file__).resolve().parents[2] / "data"
CUSTOMERS = json.loads((DATA / "fixtures.json").read_text())["customers"]
DOC = json.loads((DATA / "policy.json").read_text())
NOW = 1_760_000_000
E164 = re.compile(r"\+[1-9]\d{1,14}")


class Recorder:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, to: str, channel: str, locale: str | None = None) -> None:
        self.sent.append((to, channel))

    def check(self, to: str, code: str) -> bool:
        return False


PHONES = st.sampled_from(sorted({c["phone"] for c in CUSTOMERS}))
E164S = st.from_regex(r"\+[1-9][0-9]{1,14}", fullmatch=True)
JUNK = st.one_of(st.none(), st.integers(), st.text(max_size=24),
                 st.sampled_from(["", "anonymous", "restricted", "+", "+0", "+1 202 555 0101",
                                  "+12025550101\n", "tel:+12025550101"]))
ANIS = st.one_of(PHONES, E164S, JUNK)
ALIASES = st.lists(st.tuples(st.sampled_from([c["customer_id"] for c in CUSTOMERS]),
                             st.one_of(E164S, PHONES)), max_size=4)


def build(aliases: list[tuple[str, str]]) -> MemoryStore:
    store = MemoryStore()
    for c in CUSTOMERS:
        store.put("customers", copy.deepcopy(c))
    for i, (cid, phone) in enumerate(aliases):
        store.put("customers", {"customer_id": f"ani#X{i}#{cid}", "phone": phone,
                                "alias_of": cid, "kind": "ani_alias"})
    return store


def deps_for(store: MemoryStore, verify: Any) -> AuthDeps:
    return AuthDeps(store=store, policy=Policy.from_dict(DOC), verify_client=verify,
                    clock=lambda: NOW, token_max_ttl_s=1800, record_ttl_s=86400)


@settings(max_examples=300, deadline=None)
@given(aliases=ALIASES, ani=ANIS, customer=st.sampled_from(CUSTOMERS))
def test_destination_is_none_or_an_ani_on_file_for_the_customer(aliases, ani, customer) -> None:
    store = build(aliases)
    deps = deps_for(store, Recorder())
    to = service.otp_destination(deps, customer, ani)
    if to is None:
        return
    assert to == ani and isinstance(ani, str) and E164.fullmatch(ani)
    assert ani == customer["phone"] or any(
        cid == customer["customer_id"] and phone == ani for cid, phone in aliases)


@settings(max_examples=300, deadline=None)
@given(aliases=ALIASES, ani=ANIS, customer=st.sampled_from(CUSTOMERS))
def test_destination_is_never_the_phone_on_file_when_it_differs(aliases, ani, customer) -> None:
    deps = deps_for(build(aliases), Recorder())
    to = service.otp_destination(deps, customer, ani)
    if customer["phone"] != ani:
        assert to != customer["phone"]


@settings(max_examples=300, deadline=None)
@given(aliases=ALIASES, ani=ANIS, customer=st.sampled_from(CUSTOMERS), fail=st.booleans())
def test_send_otp_sends_only_to_the_ani_or_escalates(aliases, ani, customer, fail) -> None:
    store = build(aliases)
    rec = Recorder()
    if fail:
        def boom(to: str, channel: str, locale: str | None = None) -> None:
            raise VerifyUnavailable("down")
        rec.send = boom  # type: ignore[method-assign]
    deps = deps_for(store, rec)
    tokens.issue(store, call_sid="CAp1", customer_id=customer["customer_id"], level=2,
                 now=NOW, max_ttl_s=1800)
    expected = service.otp_destination(deps, customer, ani)
    out = service.send_otp(deps, call_sid="CAp1", ani=ani, language="en-US")  # type: ignore[arg-type]
    if expected is None or fail:
        assert out == {"result": "escalate", "reason": "otp_unavailable"}
    else:
        assert out == {"result": "sent"}
    assert all(to == ani and to == expected for to, _ in rec.sent)
    if expected is None:
        assert rec.sent == []
