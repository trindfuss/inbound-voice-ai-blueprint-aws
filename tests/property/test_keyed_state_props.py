"""Keyed-in phone login: call state and session round-trips with the new fields.

``CallState.identity_source`` and ``Session.keyed_tries`` survive create/update/load and the
whole-item round-trip with unknown attributes kept (``_extras``); items written before the fields
existed load with the defaults (``ani``, 0). There was no session round-trip property, so one is
added here over the session's stored fields.
"""

from __future__ import annotations

import string
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lambdas.frontdoor import state as sessions
from owlbank import callstate as cs
from owlbank import segments as sg
from owlbank.store import MemoryStore

SID = "CA" + "9" * 32
PROPS = settings(max_examples=80, deadline=None)

sources = st.sampled_from(["ani", "keyed_phone"])
extra_names = st.text(alphabet=string.ascii_letters + "_", min_size=1, max_size=12).filter(
    lambda k: k[0].isalpha() and k not in cs._FIELD_NAMES)  # noqa: SLF001
extras = st.dictionaries(extra_names, st.one_of(st.none(), st.integers(-99, 99),
                                                st.text(string.ascii_letters, max_size=8)), max_size=3)


@pytest.fixture(autouse=True)
def _clear() -> Any:
    sg.clear_sensitive(SID)
    yield
    sg.clear_sensitive(SID)


@PROPS
@given(first=sources, later=st.lists(sources, max_size=4), extra=extras,
       level=st.integers(0, 3), drop_field=st.booleans())
def test_call_state_identity_source_round_trips(first: str, later: list[str], extra: dict[str, Any],
                                                level: int, drop_field: bool) -> None:
    store = MemoryStore()
    state = cs.CallState(call_sid=SID, expires_at=10_000, level=level, identity_source=first,
                         _extras=extra)
    item = state.to_item()
    assert cs.CallState.from_item(item) == state and cs.CallState.from_item(item).to_item() == item
    if drop_field:  # written before the field existed: loads as the default
        legacy = {k: v for k, v in item.items() if k != "identity_source"}
        assert cs.CallState.from_item(legacy).identity_source == "ani"
    cur = cs.create(store, state)
    for source in later:
        cur = cs.update(store, cur, now=lambda: 0, identity_source=source)
        loaded = cs.load(store, SID)
        assert loaded == cur and loaded is not None and loaded.identity_source == source
        raw = store.get(cs.TABLE, {"call_sid": SID})
        assert raw is not None and {k: raw[k] for k in extra} == extra
    final = cs.load(store, SID)
    assert final is not None and final.identity_source == (later[-1] if later else first)
    assert final.level == level


@PROPS
@given(keyed_tries=st.integers(0, 2), stepups=st.integers(0, 5), lookup=st.sampled_from(
    ["pending", "ok", "error", "withheld"]), conversation_id=st.one_of(st.none(), st.text(
        string.ascii_letters + string.digits, min_size=1, max_size=10)),
    ended=st.booleans(), welcome=st.booleans(), drop_field=st.booleans())
def test_session_round_trips_with_keyed_tries(keyed_tries: int, stepups: int, lookup: str,
                                              conversation_id: str | None, ended: bool,
                                              welcome: bool, drop_field: bool) -> None:
    store = MemoryStore()
    sess = sessions.Session(call_sid=SID, expires_at=10_000, lookup=lookup, stepups=stepups,
                            conversation_id=conversation_id, ended=ended,
                            welcome_requested=welcome, keyed_tries=keyed_tries)
    sessions.save(store, sess)
    assert sessions.load(store, SID) == sess
    if drop_field:
        item = store.get(sessions.TABLE, {"call_sid": sess.key()})
        assert item is not None
        del item["keyed_tries"]
        store.put(sessions.TABLE, item)
        loaded = sessions.load(store, SID)
        assert loaded is not None and loaded.keyed_tries == 0
        assert loaded == sessions.Session(**{**_fields(sess), "keyed_tries": 0})


def _fields(sess: sessions.Session) -> dict[str, Any]:
    return {f: getattr(sess, f) for f in sessions.Session.__dataclass_fields__}
