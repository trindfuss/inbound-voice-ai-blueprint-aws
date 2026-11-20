"""The front door uses the bridge's prefetched balance if and only if the
tool, tool use ID and arguments match the pending tool and the record is at most 30 s old; the
record is always cleared after the front door's ``get_balance`` pass."""

from __future__ import annotations

import json
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from lambdas.frontdoor.app import PREFETCH_MAX_AGE_S, prefetched_balance
from owlbank import callstate
from owlbank.callstate import PendingTool
from owlbank.segments import clear_sensitive
from tests.component.test_frontdoor_support import CLOCK, build, sid

BALANCE = {"accountType": "checking", "last4": "4821", "available": "1234.56",
           "current": "1310.06", "pending": "75.50", "currency": "USD",
           "asOf": "2026-10-10T14:58:00Z"}
NOW = int(CLOCK)

tool_use_ids = st.one_of(st.sampled_from(["tooluse_1", "tooluse_2"]),
                         st.text(min_size=1, max_size=12))
tools = st.sampled_from(["get_balance", "list_recent_transactions", "make_payment"])
accounts = st.sampled_from(["checking", "savings"])
ages = st.integers(min_value=0, max_value=120)


def _record(tool_use_id: str, tool: str, account: str, age: int) -> dict[str, Any]:
    return {"tool_use_id": tool_use_id, "tool": tool, "args": {"account": account},
            "result": {"status": "ok", "data": dict(BALANCE)}, "at": NOW - age}


@settings(max_examples=400)
@given(pre_id=tool_use_ids, pre_tool=tools, pre_account=accounts, age=ages,
       pend_id=tool_use_ids, pend_account=accounts)
def test_used_iff_tool_id_args_match_and_fresh(pre_id, pre_tool, pre_account, age, pend_id,
                                               pend_account):
    res, reason, _age = prefetched_balance(
        _record(pre_id, pre_tool, pre_account, age), tool="get_balance", tool_use_id=pend_id,
        args={"account": pend_account}, now=NOW)
    match = (pre_tool == "get_balance" and pre_id == pend_id and pre_account == pend_account
             and age <= PREFETCH_MAX_AGE_S)
    assert (res is not None) == match
    assert (reason is None) == match
    if match:
        assert res is not None and res.status == "ok" and res.data["available"] == "1234.56"


json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=True) | st.text(max_size=8),
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=8), inner,
                                                                max_size=3),
    max_leaves=10)


@settings(max_examples=300)
@given(record=st.one_of(st.none(), st.dictionaries(
    st.sampled_from(["tool_use_id", "tool", "args", "result", "at"]), json_values)))
def test_any_record_is_decided_without_raising(record):
    res, reason, _ = prefetched_balance(record, tool="get_balance", tool_use_id="tooluse_1",
                                        args={"account": "checking"}, now=NOW)
    assert (res is None) != (reason is None)
    assert reason in (None, "missing", "other_call", "other_args", "stale", "bad_shape")


@settings(max_examples=40, deadline=None)
@given(pre_id=st.sampled_from(["tooluse_1", "tooluse_2"]), pre_tool=tools,
       pre_account=accounts, age=ages, pend_id=st.sampled_from(["tooluse_1", "tooluse_2"]),
       pend_account=accounts)
def test_front_door_uses_iff_match_and_always_clears(pre_id, pre_tool, pre_account, age,
                                                     pend_id, pend_account):
    h = build()
    call_sid = sid(1)
    try:
        h.login_thomas(call_sid)
        record = _record(pre_id, pre_tool, pre_account, age)
        callstate.mutate(h.store, call_sid, lambda _s: {
            callstate.F_PREFETCHED: record,
            "pending_tool": PendingTool(tool_use_id=pend_id, name="get_balance",
                                        args={"account": pend_account})})
        h.lam.calls.clear()
        h.handoff(call_sid, {"reason": "tool", "tool": "get_balance",
                             "args": {"account": pend_account}})
        match = (pre_tool == "get_balance" and pre_id == pend_id
                 and pre_account == pend_account and age <= PREFETCH_MAX_AGE_S)
        assert ("get_balance" not in h.lam.ops()) == match
        assert h.state(call_sid).prefetched is None
        assert json.dumps(h.state(call_sid).to_item())  # still a plain storable item
    finally:
        clear_sensitive(call_sid)
