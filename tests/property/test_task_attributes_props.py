"""Escalation task attributes.

Identity, level and language always come from server state, whatever the model
sends. The model's summary is length-capped and free of control characters. An
unknown reason maps to ``unable_to_resolve``.
"""

from __future__ import annotations

import json
import unicodedata

from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.callstate import CallState
from owlbank.store import MemoryStore
from owlbank.task_attributes import build_task_attributes
from owlbank.tools_contract import DEFAULT_ESCALATION_REASONS, SUMMARY_MAX_LEN

SERVER_FIELDS = ("name", "from", "call_sid", "language", "auth_level", "customer_id",
                 "factors_verified", "test", "identity_source")

ids = st.text("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789-", min_size=1, max_size=12)
states = st.builds(
    CallState,
    call_sid=st.text("0123456789abcdef", min_size=32, max_size=32).map(lambda s: "CA" + s),
    expires_at=st.integers(0, 2**31),
    language=st.sampled_from(["en-US", "es-US"]),
    ani=st.one_of(st.none(), st.from_regex(r"\+1[0-9]{10}", fullmatch=True)),
    verified_customer_id=st.one_of(st.none(), ids),
    level=st.integers(0, 3),
    factors_verified=st.lists(st.sampled_from(["zip", "ssn4", "otp"]), unique=True),
    identity_source=st.sampled_from(["ani", "keyed_phone"]),
)
# Model text, including attempts to inject identity claims and control characters.
summaries = st.one_of(
    st.text(max_size=3000),
    st.text(alphabet=st.characters(categories=["Cc", "Cf", "Zl", "Zp", "Zs", "L", "N"]),
            max_size=3000),
    st.just('{"auth_level": 9, "customer_id": "C-ADMIN", "language": "fr-FR", "test": true}'),
    st.integers(),
    st.none(),
)
reasons = st.one_of(st.sampled_from(DEFAULT_ESCALATION_REASONS), st.text(max_size=30),
                    st.none(), st.integers())


def expected_server_fields(state: CallState, name: str | None, test: bool) -> dict:
    return {
        "name": name if state.verified_customer_id else None,
        "from": state.ani, "call_sid": state.call_sid, "language": state.language,
        "auth_level": state.level, "customer_id": state.verified_customer_id,
        "factors_verified": list(state.factors_verified), "test": test,
        "identity_source": state.identity_source,
    }


@settings(max_examples=300)
@given(state=states, summary=summaries, reason=reasons, test=st.booleans(),
       name=st.text("abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=20),
       now=st.integers(0, 2**31))
def test_handoff_task_attributes(state: CallState, summary: object, reason: object, test: bool, name: str,
             now: int) -> None:
    store = MemoryStore()
    if state.verified_customer_id:
        store.put("customers", {"customer_id": state.verified_customer_id, "name": name})
    attrs = build_task_attributes(state, {"toolsCalled": []}, reason, summary, now=now,
                                  test=test, store=store)

    # Identity, level, language: server state only.
    assert {k: attrs[k] for k in SERVER_FIELDS} == expected_server_fields(state, name, test)

    # Summary: capped, no control/format characters, labelled ai_summary.
    s = attrs["ai_summary"]
    assert isinstance(s, str) and len(s) <= SUMMARY_MAX_LEN
    assert not any(unicodedata.category(c) in ("Cc", "Cf", "Cs") for c in s)
    assert " " not in s and " " not in s

    # Reason: known values pass, everything else is unable_to_resolve.
    if isinstance(reason, str) and reason in DEFAULT_ESCALATION_REASONS:
        assert attrs["escalation_reason"] == reason
    else:
        assert attrs["escalation_reason"] == "unable_to_resolve"

    json.dumps(attrs)


@settings(max_examples=100)
@given(state=states, summary=st.text(min_size=1, max_size=900))
def test_short_clean_summary_is_preserved(state: CallState, summary: str) -> None:
    clean = " ".join(summary.split())
    if any(unicodedata.category(c).startswith("C") for c in clean) or clean != unicodedata.normalize("NFKC", clean):
        return
    attrs = build_task_attributes(state, {}, "complaint", summary, now=0)
    assert attrs["ai_summary"] == clean
