"""The prepared welcome in call state: ``welcome_prepared`` and ``welcome_offer``
round-trip through the store; ``seal_welcome``/``unseal_welcome`` keep the offer ID sealed (an ID, never text or a trait value), expire with
``owlbank.last_spoken``'s TTL, and refuse another call's or an edited record. The seal round-trip over
generated values is in tests/property/test_callstate_welcome_props.py."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from owlbank import callstate as cs
from owlbank import last_spoken
from owlbank import segments as sg
from owlbank import welcome as wl
from owlbank.store import MemoryStore
from owlbank.token_seal import FakeSealer
from tests.logutil import lines, text_of

SID = "CA" + "w" * 32
OTHER = "CA" + "x" * 32
OFFER = "heloc"
NOW = 1_800_000_000.0
SEALER = FakeSealer()


@pytest.fixture(autouse=True)
def _clear() -> None:
    sg.clear_sensitive(SID)
    yield
    sg.clear_sensitive(SID)


def made(**kw: Any) -> dict[str, Any]:
    record = cs.seal_welcome(SEALER, OFFER, call_sid=SID, customer_id="C-1", language="en-US",
                             reason="matched", now=NOW, **kw)
    assert record is not None
    return record


def test_record_shape_has_no_plaintext() -> None:
    record = made()
    assert set(record) == {"sealed", "customer_id", "language", "reason", "at"}
    assert record["customer_id"] == "C-1" and record["language"] == "en-US" and record["reason"] == "matched"
    assert record["at"] == int(NOW) and OFFER not in record["sealed"]


def test_round_trip() -> None:
    assert cs.unseal_welcome(SEALER, made(), call_sid=SID, now=NOW + 1) == cs.PreparedWelcome(
        offer_id=OFFER, customer_id="C-1", language="en-US", reason="matched", at=int(NOW))


@pytest.mark.parametrize("reason", sorted(wl.REASONS - {wl.MATCHED}))
def test_fallback_record_round_trips_without_an_offer(reason: str) -> None:
    record = cs.seal_welcome(SEALER, None, call_sid=SID, customer_id="C-1", language="es-US",
                             reason=reason, now=NOW)
    got = cs.unseal_welcome(SEALER, record, call_sid=SID, now=NOW)
    assert got == cs.PreparedWelcome(offer_id=None, customer_id="C-1", language="es-US", reason=reason,
                                     at=int(NOW))


def test_another_calls_record_is_refused(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert cs.unseal_welcome(SEALER, made(), call_sid=OTHER, now=NOW) is None
    (line,) = lines(caplog.records, "owlbank.callstate")
    assert line["event"] == "welcome_prepared_unusable" and line["reason"] == "unseal_failed"
    assert OFFER not in text_of(caplog.records)


@pytest.mark.parametrize(("field", "value"), [("customer_id", "C-2"), ("language", "es-US"),
                                              ("reason", "no_match")])
def test_edited_clear_fields_are_refused(field: str, value: str) -> None:
    assert cs.unseal_welcome(SEALER, {**made(), field: value}, call_sid=SID, now=NOW) is None


def test_expires_with_the_last_spoken_ttl() -> None:
    record = made()
    assert cs.unseal_welcome(SEALER, record, call_sid=SID, now=NOW + last_spoken.TTL_S - 1) is not None
    assert cs.unseal_welcome(SEALER, record, call_sid=SID, now=NOW + last_spoken.TTL_S) is None


@pytest.mark.parametrize("record", [None, "x", [], {}, {"sealed": ""}, {"sealed": "abc"},
                                    {"sealed": "abc", "customer_id": "C-1", "language": "en-US", "reason": "matched",
                                     "at": True},
                                    {"sealed": "abc", "customer_id": "C-1", "language": "en-US", "reason": "matched",
                                     "at": float("nan")},
                                    {"sealed": "!!", "customer_id": "C-1", "language": "en-US", "reason": "matched",
                                     "at": NOW}])
def test_bad_records_give_none(record: Any) -> None:
    assert cs.unseal_welcome(SEALER, record, call_sid=SID, now=NOW) is None


def test_a_sealed_non_json_payload_is_refused() -> None:
    record = {**made(), "sealed": SEALER.seal("not json", call_sid=SID)}
    assert cs.unseal_welcome(SEALER, record, call_sid=SID, now=NOW) is None


@pytest.mark.parametrize(("offer_id", "reason"), [
    (None, "matched"), ("", "matched"), ("  ", "matched"), ("HELOC", "matched"), ("a" * 33, "matched"),
    ("offer.heloc", "matched"), (7, "matched"), (OFFER, "no_match"), (OFFER, "memory_error"),
    (None, "made"), (None, "timeout"),  # reasons are gone
])
def test_a_known_reason_and_an_offer_id_exactly_for_matched(offer_id: Any, reason: str) -> None:
    with pytest.raises(ValueError):
        cs.seal_welcome(SEALER, offer_id, call_sid=SID, customer_id="C-1", language="en-US", reason=reason,
                        now=NOW)


@pytest.mark.parametrize("payload", [
    {"offer_id": None, "reason": "matched"},          # matched without an offer
    {"offer_id": "heloc", "reason": "no_match"},      # an offer with another reason
    {"offer_id": "Bad ID", "reason": "matched"},      # not an offer ID
    {"text": "Thanks, Thomas. Want it?", "reason": "made"},  # an record
])
def test_a_sealed_payload_breaking_the_offer_rule_is_refused(payload: dict[str, Any]) -> None:
    body = {"customer_id": "C-1", "language": "en-US", **payload}
    record = {"sealed": SEALER.seal(json.dumps(body), call_sid=SID), "customer_id": "C-1", "language": "en-US",
              "reason": payload["reason"], "at": NOW}
    assert cs.unseal_welcome(SEALER, record, call_sid=SID, now=NOW) is None


def test_a_failed_seal_stores_none_and_logs_no_text(caplog: pytest.LogCaptureFixture) -> None:
    class Broken:
        def seal(self, token: str, *, call_sid: str) -> str:
            raise RuntimeError(f"kms down {token}")

        def unseal(self, sealed: str, *, call_sid: str) -> str:
            raise AssertionError

    with caplog.at_level(logging.INFO):
        assert cs.seal_welcome(Broken(), OFFER, call_sid=SID, customer_id="C-1", language="en-US",
                               reason="matched", now=NOW) is None
    assert caplog.records and OFFER not in text_of(caplog.records)


def test_fields_round_trip_through_the_store() -> None:
    store = MemoryStore()
    state = cs.create(store, cs.CallState(call_sid=SID, expires_at=2_000_000_000))
    assert state.welcome_prepared is None and state.welcome_offer is None
    record = made()
    updated = cs.update(store, state, welcome_prepared=record, welcome_offer=cs.WELCOME_OFFER_MADE)
    loaded = cs.load(store, SID)
    assert loaded == updated and loaded is not None
    assert loaded.welcome_prepared == record and loaded.welcome_offer == "made"
    assert loaded.to_item()[cs.F_WELCOME_PREPARED] == record and loaded.to_item()[cs.F_WELCOME_OFFER] == "made"
    got = cs.unseal_welcome(SEALER, loaded.welcome_prepared, call_sid=SID, now=NOW)
    assert got is not None and got.offer_id == OFFER
