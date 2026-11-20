"""The front door reads out the bridge's prefetched balance when it
is for the same pending tool call, arguments and tool, at most 30 s old; otherwise it fetches.
The record is cleared on every pass. The no-leak scan runs at teardown."""

from __future__ import annotations

from typing import Any

import pytest

from owlbank import callstate
from owlbank.slots import ResolverOutput
from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    _STD_LOG_ATTRS, CLOCK, THOMAS, fd, greeting, says, sid,
)

#: Thomas's checking account, as the bank returns it (data/fixtures.json).
CHECKING = {"accountType": "checking", "last4": "4821", "available": "1234.56",
            "current": "1310.06", "pending": "75.50", "currency": "USD",
            "asOf": "2026-10-10T14:58:00Z"}


def _prefetch(fd, call_sid, *, tool_use_id="tooluse_1", tool="get_balance",
              args: dict[str, Any] | None = None, age=5, result: Any = None) -> None:
    record = {"tool_use_id": tool_use_id, "tool": tool,
              "args": {"account": "checking"} if args is None else args,
              "result": {"status": "ok", "data": dict(CHECKING)} if result is None else result,
              "at": int(CLOCK) - age}
    callstate.mutate(fd.store, call_sid, lambda _s: {callstate.F_PREFETCHED: record})


def _handoff(fd, call_sid, account="checking"):
    fd.set_pending_tool(call_sid, "get_balance", {"account": account})  # tool_use_id "tooluse_1"
    return fd.handoff(call_sid, {"reason": "tool", "tool": "get_balance",
                                 "args": {"account": account}, "journal": "jr-1"})


def _events(caplog, name):
    return [r for r in caplog.records if getattr(r, "event", None) == name]


def _fetched_greeting(fd) -> tuple[str, dict[str, Any]]:
    """The read-out and envelope data of the ordinary fetched path, on another call."""
    fd.login_thomas(sid(9))
    root = _handoff(fd, sid(9))
    return greeting(root), dict(fd.pending_result(sid(9)).data)


@pytest.mark.parametrize("result", [
    {"status": "ok", "data": dict(CHECKING)},
    {"tool": "get_balance", "data": dict(CHECKING)},  # the raw bank_tools payload
])
def test_matching_prefetch_is_used_without_a_bank_call(fd, caplog, result):
    expected_text, expected_data = _fetched_greeting(fd)
    fd.lam.calls.clear()
    fd.login_thomas(sid(1))
    _prefetch(fd, sid(1), result=result)
    caplog.clear()
    root = _handoff(fd, sid(1))
    assert "get_balance" not in fd.lam.ops()
    assert greeting(root) == expected_text and says(root) == []
    env = fd.pending_result(sid(1))
    assert env.status == "completed" and env.next == "await_caller"
    assert dict(env.data) == expected_data
    assert fd.state(sid(1)).prefetched is None
    used = _events(caplog, "prefetch_used")
    assert len(used) == 1 and used[0].age_s == 5.0
    assert not _events(caplog, "prefetch_ignored")


def test_prefetch_at_exactly_30_seconds_is_used(fd):
    fd.login_thomas(sid(1))
    _prefetch(fd, sid(1), age=30)
    fd.lam.calls.clear()
    _handoff(fd, sid(1))
    assert "get_balance" not in fd.lam.ops()


@pytest.mark.parametrize(("kw", "reason"), [
    ({"age": 31}, "stale"),
    ({"tool_use_id": "tooluse_other"}, "other_call"),
    ({"tool": "list_recent_transactions"}, "other_call"),
    ({"args": {"account": "savings"}}, "other_args"),
    ({"result": {"status": "ok", "data": {"available": "1.00"}}}, "bad_shape"),
    ({"result": {"error": "NOT_FOUND"}}, "bad_shape"),
    ({"result": {"status": "ok", "data": {**CHECKING, "available": "lots"}}}, "bad_shape"),
])
def test_non_matching_prefetch_is_ignored_and_the_bank_is_called(fd, caplog, kw, reason):
    fd.login_thomas(sid(1))
    _prefetch(fd, sid(1), **kw)
    fd.lam.calls.clear()
    caplog.clear()
    root = _handoff(fd, sid(1))
    assert fd.lam.ops().count("get_balance") == 1
    assert "four, eight, two, one" in greeting(root)
    assert fd.state(sid(1)).prefetched is None
    ignored = _events(caplog, "prefetch_ignored")
    assert [r.reason for r in ignored] == [reason]
    assert not _events(caplog, "prefetch_used")


def test_stale_prefetch_with_other_figures_is_not_read_out(fd):
    fd.login_thomas(sid(1))
    _prefetch(fd, sid(1), age=31,
              result={"status": "ok", "data": {**CHECKING, "available": "1.00", "current": "1.00",
                                                "pending": "0.00"}})
    root = _handoff(fd, sid(1))
    assert "one thousand two hundred thirty-four dollars" in greeting(root)
    assert fd.pending_result(sid(1)).data["available"] == "1234.56"


def test_missing_prefetch_fetches_as_today(fd, caplog):
    fd.login_thomas(sid(1))
    fd.lam.calls.clear()
    caplog.clear()
    _handoff(fd, sid(1))
    assert fd.lam.ops().count("get_balance") == 1
    assert [r.reason for r in _events(caplog, "prefetch_ignored")] == ["missing"]


def test_prefetch_is_used_once_only(fd):
    fd.login_thomas(sid(1))
    _prefetch(fd, sid(1))
    fd.lam.calls.clear()
    _handoff(fd, sid(1))
    assert "get_balance" not in fd.lam.ops()
    _handoff(fd, sid(1))  # same tool_use_id again: the record is gone, so the bank is asked
    assert fd.lam.ops().count("get_balance") == 1


def test_failed_token_check_steps_up_and_never_uses_the_prefetch(fd, caplog):
    fd.start(sid(1), THOMAS)
    # Unverified, no token, login not closed: the caller asks something else at the ZIP
    # question (a cancel there is a decline and closes the login).
    fd.resolver.default = ResolverOutput(kind="intent_switch")
    fd.xml("factor", sid(1), SpeechResult="what are your hours", Confidence="0.9")
    fd.resolver.default = ResolverOutput(kind="none")
    _prefetch(fd, sid(1))
    caplog.clear()
    root = _handoff(fd, sid(1))
    assert says(root)[0].startswith("I can help you check your balance.")
    assert "get_balance" not in fd.lam.ops()
    assert "four, eight, two, one" not in " ".join(says(root)) + greeting_or_empty(root)
    assert fd.state(sid(1)).prefetched is None
    assert not _events(caplog, "prefetch_used")


def test_prefetch_logs_carry_no_figures(fd, caplog):
    fd.login_thomas(sid(1))
    _prefetch(fd, sid(1))
    caplog.clear()
    _handoff(fd, sid(1))
    _prefetch(fd, sid(1), age=99)
    _handoff(fd, sid(1))
    for r in _events(caplog, "prefetch_used") + _events(caplog, "prefetch_ignored"):
        extra = {k: v for k, v in vars(r).items() if k not in _STD_LOG_ATTRS and not k.startswith("_")}
        assert set(extra) <= {"event", "call_sid", "tool", "age_s", "reason", "component",
                              "request_id", "trace_id"}, extra
        text = r.getMessage() + " ".join(str(v) for v in extra.values())
        for figure in ("1234.56", "1310.06", "75.50", "4821"):
            assert figure not in text


def greeting_or_empty(root) -> str:
    found = root.find("./Connect/ConversationRelay")
    return (found.get("welcomeGreeting") or "") if found is not None else ""
