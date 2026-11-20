"""The prepared-welcome seal over generated values:
``unseal_welcome`` gives back exactly what ``seal_welcome`` sealed for the same call, and nothing for any
other call. The record carries an offer ID exactly for ``matched``, and never in plaintext."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from owlbank import callstate as cs
from owlbank import welcome as wl
from owlbank.token_seal import FakeSealer

PROPS = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
SEALER = FakeSealer()
NOW = 1_800_000_000.0

sids = st.text(alphabet="0123456789abcdef", min_size=32, max_size=32).map(lambda s: "CA" + s)
ids = st.text(min_size=1, max_size=40)
reasons = st.sampled_from(sorted(wl.REASONS))
offer_ids = st.from_regex(r"[a-z0-9_]{1,32}", fullmatch=True)


@PROPS
@given(sids, offer_ids, ids, ids, reasons, st.floats(0, 599))
def test_seal_round_trips_for_the_same_call(sid: str, offer_id: str, customer_id: str, language: str, reason: str,
                                            later: float) -> None:
    body = offer_id if reason == wl.MATCHED else None
    record = cs.seal_welcome(SEALER, body, call_sid=sid, customer_id=customer_id, language=language,
                             reason=reason, now=NOW)
    assert record is not None
    assert set(record) == {"sealed", "customer_id", "language", "reason", "at"}
    if len(offer_id) >= 8:
        assert offer_id not in str(record["sealed"])
    got = cs.unseal_welcome(SEALER, record, call_sid=sid, now=NOW + later)
    assert got == cs.PreparedWelcome(offer_id=body, customer_id=customer_id, language=language, reason=reason,
                                     at=int(NOW))


@PROPS
@given(sids, sids, offer_ids, ids, reasons)
def test_another_calls_record_never_unseals(sid: str, other: str, offer_id: str, customer_id: str,
                                            reason: str) -> None:
    assume(sid != other)
    record = cs.seal_welcome(SEALER, offer_id if reason == wl.MATCHED else None, call_sid=sid,
                             customer_id=customer_id, language="en-US", reason=reason, now=NOW)
    assert cs.unseal_welcome(SEALER, record, call_sid=other, now=NOW) is None


@PROPS
@given(st.none() | offer_ids | st.text(max_size=40) | st.integers(), reasons | st.text(min_size=1, max_size=20))
def test_an_offer_id_is_sealed_exactly_for_matched(offer_id: Any, reason: str) -> None:
    ok = reason in wl.REASONS and (offer_id is None) == (reason != wl.MATCHED) and (
        offer_id is None or (isinstance(offer_id, str) and wl.OFFER_ID_RE.fullmatch(offer_id) is not None))
    if ok:
        assert cs.seal_welcome(SEALER, offer_id, call_sid="CA" + "a" * 32, customer_id="C-1", language="en-US",
                               reason=reason, now=NOW) is not None
    else:
        with pytest.raises(ValueError):
            cs.seal_welcome(SEALER, offer_id, call_sid="CA" + "a" * 32, customer_id="C-1", language="en-US",
                            reason=reason, now=NOW)
