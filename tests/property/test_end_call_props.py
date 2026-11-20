"""End-call tool properties: the offered tool list and the call-state flag.

* For the ``end_call`` switch on or off, any
  verified level and any escalation reasons and cards, ``end_call`` is in the inline specs exactly
  when the switch is on, when the specs are built the way the agent client builds them
  (``exclude= if on else {END_CALL}``), and its spec is the same at every level. The other
  tools are never affected by the switch.
* ``login_closed`` round-trip: any call state, with the flag either way, comes
  back equal through ``to_item``/``from_item`` and through the store (``create``, ``update``), and
  an item without the attribute loads as ``False``.
"""

from __future__ import annotations

import copy
import json
import string
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank import callstate as cs
from owlbank import segments as sg
from owlbank import tools_contract as tc
from owlbank.policy import END_CALL_ITEM, Policy
from owlbank.store import MemoryStore

PROPS = settings(max_examples=60, deadline=None)
SID = "CA" + "d" * 32
SEED: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[2] / "data" / "policy.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _clear() -> Any:
    sg.clear_sensitive(SID)
    yield
    sg.clear_sensitive(SID)




def _offered(policy: Policy, reasons: list[str], cards: list[str]) -> list[dict[str, Any]]:
    """The inline specs as the agent client will build them."""
    exclude = () if policy.end_call_enabled() else {tc.END_CALL}
    return tc.inline_tool_specs(reasons, cards, exclude=exclude)


reason_names = st.lists(st.text(string.ascii_lowercase + "_", min_size=1, max_size=12),
                        min_size=1, max_size=4, unique=True)
card_names = st.lists(st.sampled_from(["red", "purple", "rewards", "gold"]), min_size=1, max_size=4,
                      unique=True)
switch = st.one_of(st.booleans().map(lambda b: {"enabled": b}),
                   st.sampled_from([None, {}, {"enabled": "false"}, {"enabled": 0}, []]))


@PROPS
@given(switch, st.integers(0, SEED["max_level"]), reason_names, card_names)
def test_end_call_is_offered_iff_the_switch_is_on(raw: Any, level: int, reasons: list[str],
                                                     cards: list[str]) -> None:
    doc = copy.deepcopy(SEED)
    if raw is None:
        del doc[END_CALL_ITEM]
    else:
        doc[END_CALL_ITEM] = raw
    store = MemoryStore()
    for item in Policy.to_items(doc):
        store.put("policy", item)
    on = not (isinstance(raw, dict) and raw.get("enabled") is False)  # malformed / missing: on
    for policy in (Policy.from_dict(doc), Policy.from_store(store)):
        assert policy.end_call_enabled() is on
        assert policy.min_level(tc.END_CALL) <= level  # the tool needs no level
        specs = _offered(policy, reasons, cards)
        names = [s["name"] for s in specs]
        assert (tc.END_CALL in names) is on
        assert [n for n in names if n != tc.END_CALL] == [n for n in tc.INLINE_TOOLS if n != tc.END_CALL]
        if on:  # the same spec whatever the level, reasons or cards
            spec = next(s for s in specs if s["name"] == tc.END_CALL)
            assert spec == next(s for s in tc.inline_tool_specs() if s["name"] == tc.END_CALL)


# --------------------------------------------------------------------------- login_closed round-trip


states = st.builds(
    lambda closed, level, lang, src, verified, attempts, sessions: cs.CallState(
        call_sid=SID, expires_at=10_000, login_closed=closed, level=level, language=lang,
        identity_source=src, factors_verified=verified, attempts=attempts, ai_sessions=sessions),
    st.booleans(), st.integers(0, 3), st.sampled_from(["en-US", "es-US"]),
    st.sampled_from(["ani", "keyed_phone"]), st.lists(st.sampled_from(["zip", "ssn4", "otp"]), unique=True),
    st.dictionaries(st.sampled_from(["zip", "ssn4", "otp"]), st.integers(0, 3)), st.integers(0, 3))


@PROPS
@given(states, st.booleans())
def test_login_closed_round_trips(state: cs.CallState, later: bool) -> None:
    item = state.to_item()
    assert item[cs.F_LOGIN_CLOSED] is state.login_closed
    assert cs.CallState.from_item(item) == state
    legacy = {k: v for k, v in item.items() if k != cs.F_LOGIN_CLOSED}
    assert cs.CallState.from_item(legacy).login_closed is False

    store = MemoryStore()
    created = cs.create(store, state)
    assert cs.load(store, SID) == created and created.login_closed is state.login_closed
    updated = cs.update(store, created, now=lambda: 100, **{cs.F_LOGIN_CLOSED: later})
    loaded = cs.load(store, SID)
    assert loaded == updated and loaded.login_closed is later
