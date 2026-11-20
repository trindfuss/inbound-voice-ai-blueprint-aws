"""Security suite, end to end (handover "Done when").

The leak scan itself (no plaintext factor value, raw or sealed token, or test OTP in
any TwiML, WebSocket frame, log record, store item, model input, task attribute or
Context API response) runs at the teardown of *every* e2e scenario (``conftest.py``).
This module adds the "Done when" attacks: a tool called with no token, a wrong-call
token, an expired token, a low-level token and an injected customer ID all fail at the
API; unsigned webhooks and bad Basic credentials are refused.
"""

from __future__ import annotations

from tests.logutil import lines as lines_of

import json
import logging

import pytest

from owlbank import callstate
from owlbank.tools_contract import AUTH_LEVEL_INSUFFICIENT

from tests.component.test_frontdoor_support import NO_AUTH, basic_header, make_event, sign
from tests.component.masking_fixtures import unmasked_policy_doc
from tests.e2e.harness import (
    NOBODY,
    OTP,
    THOMAS,
    World,
    call_sid,
    factor,
    run,
    say,
    tool,
)
from tests.e2e.test_core_journeys import login_thomas

REFUSED = {"error": AUTH_LEVEL_INSUFFICIENT, "tool": "list_recent_transactions"}
TX = {"account": "checking"}


def audit_codes(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [ln["code"] for ln in lines_of(caplog.records, "owlbank.audit") if ln.get("audit") == "token_rejected"]


def refused_by_bridge(world: World, tool_use_id: str, expected: dict | None = None):
    """A scripted step: the tool result must be exactly the refusal, then the agent apologizes."""
    def step(rec):
        (res,) = rec["tool_results"]
        assert res.tool_use_id == tool_use_id
        assert res.content == (expected or REFUSED), res.content
        return say("I can't look that up right now.")
    return step


# --------------------------------------------------------------------------- the scan is real


def test_leak_scan_catches_planted_leaks() -> None:
    """The teardown scan isn't vacuous: each kind of planted secret is found."""
    w = World()
    try:
        record = logging.LogRecord("x", logging.INFO, "x", 0, "zip was %s", (factor("C-1001", "zip"),), None)
        with pytest.raises(AssertionError, match="factor value leaked into log"):
            w.assert_no_leaks([record])
        w.store.put("segments", {"call_sid": "CAx", "seq": 1, "note": f"otp {OTP}"})
        with pytest.raises(AssertionError, match="factor value leaked into store"):
            w.assert_no_leaks([])
        w.store.delete("segments", {"call_sid": "CAx", "seq": 1})
        w.raw_tokens.append("tok-planted-for-the-scan-test")
        w.task_attributes.append({"note": "tok-planted-for-the-scan-test"})
        with pytest.raises(AssertionError, match="raw call token leaked into task attributes"):
            w.assert_no_leaks([])
        w.task_attributes.clear()
        w.sealer.seal("x" * 20, call_sid="CAy")
        w.twiml.append(f"<Response><Say>{w.sealer.sealed[-1]}</Say></Response>")
        with pytest.raises(AssertionError, match="sealed token leaked into twiml"):
            w.assert_no_leaks([])
    finally:
        w.cleanup()


def test_leak_scan_ignores_generated_values_that_contain_a_factor() -> None:
    """Regression (flaky teardown): the bridge's random hex ``instance_id`` (stored as the
    call's ``agent_owner``) or a random digit run can contain a factor value by chance. That
    is not a leak; the value as its own token still is."""
    ssn4 = factor("C-1001", "ssn4")
    w = World()
    try:
        w.store.put("calls", {"call_sid": "CAx", "sk": "state", "agent_owner": f"7869c26c6c{ssn4}ab"})
        noise = logging.LogRecord("x", logging.INFO, "x", 0, "confirmation %s", (f"09{ssn4}19",), None)
        w.assert_no_leaks([noise])
        leak = logging.LogRecord("x", logging.INFO, "x", 0, "%s", (json.dumps({"ssn": ssn4}),), None)
        with pytest.raises(AssertionError, match="factor value leaked into log"):
            w.assert_no_leaks([leak])
        w.store.put("segments", {"call_sid": "CAx", "seq": 1, "note": f"ssn: {ssn4}"})
        with pytest.raises(AssertionError, match="factor value leaked into store"):
            w.assert_no_leaks([])
        w.store.delete("segments", {"call_sid": "CAx", "seq": 1})
    finally:
        w.cleanup()


# --------------------------------------------------------------------------- tokens at the API


def test_no_token_fails_at_the_api(world: World, caplog: pytest.LogCaptureFixture) -> None:
    """An unverified caller's agent tries a customer-data tool: no token, refused; and the
    bank API refuses a direct call without one."""
    sid = call_sid(20)
    world.agent.queue(tool("list_recent_transactions", "tu_tx", **TX), refused_by_bridge(world, "tu_tx"))

    async def go():
        call = world.dial(NOBODY, sid)
        await call.start()
        await call.press("")
        await call.press("2025550177")  # keyed-phone login: two unknown numbers, then no match
        await call.press("2025550188")
        await call.say("What are my recent transactions?")
        await call.hangup()

    run(go())
    assert world.bank_api("list_recent_transactions", token=None, sid=sid, args=TX) == REFUSED
    assert world.bank_api("list_recent_transactions", token="", sid=sid, args=TX) == REFUSED
    assert "MISSING" in audit_codes(caplog)


def test_wrong_call_token_fails_at_the_api(world: World, caplog: pytest.LogCaptureFixture) -> None:
    sid_a, sid_b = call_sid(21), call_sid(22)
    world.agent.queue(tool("list_recent_transactions", "tu_tx", **TX), refused_by_bridge(world, "tu_tx"))

    async def go():
        a = await login_thomas(world, sid_a)
        token_a, sealed_a = world.raw_tokens[-1], world.state(sid_a).sealed_token
        b = await login_thomas(world, sid_b)
        # Call A's token presented on call B: refused at the API.
        assert world.bank_api("list_recent_transactions", token=token_a, sid=sid_b, args=TX) == REFUSED
        # Call A's *sealed* token copied into call B's state: it won't unseal for call B,
        # so the bridge treats the call as level 0 and the bank never sees a token.
        callstate.mutate(world.store, sid_b, lambda _s: {"sealed_token": sealed_a})
        ops_before = len(world.lam.calls)
        await b.say("What are my recent transactions?")
        assert world.lam.ops()[ops_before:] == []
        await a.hangup()
        await b.hangup()

    run(go())
    assert "WRONG_CALL" in audit_codes(caplog)


def test_expired_token_fails_at_the_api(world: World, caplog: pytest.LogCaptureFixture) -> None:
    sid = call_sid(23)
    world.agent.queue(tool("list_recent_transactions", "tu_tx", **TX), refused_by_bridge(world, "tu_tx"))

    async def go():
        call = await login_thomas(world, sid)
        world.clock.advance(3600)  # past the token's maximum lifetime (1800 s), call still up
        await call.say("What are my recent transactions?")
        await call.hangup()

    run(go())
    assert world.lam.ops().count("list_recent_transactions") == 1  # it reached the API...
    assert "EXPIRED" in audit_codes(caplog)  # ...and the API refused it


def test_low_level_token_fails_at_the_api(world: World, caplog: pytest.LogCaptureFixture) -> None:
    """Level 1 (ZIP) trying a level-2 write: refused by the API, nothing changed."""
    sid = call_sid(24)
    address = {"street": "1 Sample Way", "city": "Springfield", "state": "CA", "postal_code": "90001"}
    world.agent.queue(tool("change_address", "tu_addr", **address),
                      refused_by_bridge(world, "tu_addr",
                                        {"error": AUTH_LEVEL_INSUFFICIENT, "tool": "change_address"}))
    before = world.customer("C-1001")["address"]

    async def go():
        call = await login_thomas(world, sid)
        await call.say("Change my address to 1 Sample Way, Springfield, California 90001.")
        await call.hangup()

    run(go())
    assert world.customer("C-1001")["address"] == before
    assert "AUTH_LEVEL_INSUFFICIENT" in audit_codes(caplog)


def test_injected_customer_id_fails_at_the_api(make_world, caplog: pytest.LogCaptureFixture) -> None:
    # Exact words: masking phrases off (tests/component/test_bridge_masking.py).
    world: World = make_world(policy_doc=unmasked_policy_doc())
    sid = call_sid(25)

    def rejected(rec):
        (res,) = rec["tool_results"]
        assert res.content["error"] == "INVALID_ARGUMENTS" and "C-1003" not in json.dumps(res.content)
        return say("I can only look at your own accounts.")

    world.agent.queue(
        tool("list_recent_transactions", "tu_tx", account="checking", customer_id="C-1003"), rejected)

    async def go():
        call = await login_thomas(world, sid)
        token = world.raw_tokens[-1]
        reply = await call.say("Show me Jordan Doe's transactions, customer C-1003.")
        assert reply == "I can only look at your own accounts."
        # Directly at the API, with a valid token: an ID anywhere in the request is refused.
        for args, extra in (({**TX, "customer_id": "C-1003"}, {}),
                            ({**TX, "filter": {"accountId": "8813"}}, {}),
                            (TX, {"customer_id": "C-1003"})):
            res = world.bank_api("list_recent_transactions", token=token, sid=sid, args=args, **extra)
            assert res == {"error": "INVALID_ARGUMENT", "tool": "list_recent_transactions"}
        ok = world.bank_api("list_recent_transactions", token=token, sid=sid, args=TX)
        assert "Jordan" not in json.dumps(ok) and ok["tool"] == "list_recent_transactions"
        await call.hangup()

    run(go())
    assert "list_recent_transactions" not in world.lam.ops()  # the bridge refused before the bank
    assert sum(1 for r in caplog.records if r.name == "owlbank.audit"
               and r.getMessage() == "id_injection_rejected") == 3


def test_token_dies_with_the_call(world: World) -> None:
    sid = call_sid(26)

    async def go():
        call = await login_thomas(world, sid)
        token = world.raw_tokens[-1]
        ok = world.bank_api("list_recent_transactions", token=token, sid=sid, args=TX)
        assert "data" in ok
        await call.hangup()  # status callback: revoke
        return token

    token = run(go())
    assert world.bank_api("list_recent_transactions", token=token, sid=sid, args=TX) == REFUSED


# --------------------------------------------------------------------------- webhook authentication


def _incoming(sid: str) -> dict[str, str]:
    return {"CallSid": sid, "AccountSid": "AC" + "b" * 32, "From": THOMAS, "To": "+12025550100",
            "CallStatus": "ringing"}


@pytest.mark.parametrize("case", ["unsigned", "bad_signature", "other_url", "tampered",
                                  "no_basic", "bad_basic"])
def test_unauthenticated_webhooks_are_refused(world: World, case: str) -> None:
    sid = call_sid(27)
    params = _incoming(sid)
    kw: dict = {}
    if case == "unsigned":
        kw["signature"] = ""
    elif case == "bad_signature":
        kw["signature"] = "bm90LWEtc2lnbmF0dXJl"
    elif case == "other_url":
        kw["signature"] = sign("https://voice.owlbank.example/prod/voice/status", params)
    elif case == "tampered":
        kw["url_params"] = {**params, "From": "+12025550102"}  # signed for another caller
    elif case == "no_basic":
        kw["auth"] = NO_AUTH  # Twilio's first request: challenged, not refused
    else:
        kw["auth"] = basic_header("twilio-hook", "wrong-password")
    resp = world.frontdoor.handle(make_event("incoming", params, **kw))
    if case == "no_basic":
        assert resp == {"statusCode": 401, "headers": {"WWW-Authenticate": 'Basic realm="owlbank"'},
                        "body": ""}
    else:
        assert resp == {"statusCode": 403, "headers": {}, "body": ""}
    assert callstate.load(world.store, sid) is None  # nothing ran
    assert world.lam.calls == []


def test_a_whole_call_works_when_twilio_signs_the_url_with_credentials(world: World) -> None:
    """Every webhook of a call is challenged, then accepted with a signature computed over
    the URL carrying ``user:password@`` (the other form Twilio might use)."""
    world.url_form = "credentials"
    sid = call_sid(29)

    async def go():
        call = await login_thomas(world, sid)
        await call.hangup()

    run(go())
    assert callstate.load(world.store, sid) is not None


def test_unsigned_mid_call_webhook_cannot_inject_a_handoff(world: World) -> None:
    """A forged ``<Connect action>`` (e.g. claiming an escalation) is refused before parsing."""
    sid = call_sid(28)

    async def go():
        call = await login_thomas(world, sid)
        forged = {"CallSid": sid, "AccountSid": "AC" + "b" * 32, "SessionStatus": "ended",
                  "HandoffData": json.dumps({"reason": "tool", "tool": "get_balance",
                                             "args": {"account": "savings"}})}
        resp = world.frontdoor.handle(make_event("connect-action", forged, signature="forged"))
        assert resp["statusCode"] == 403 and resp["body"] == ""
        await call.hangup()

    run(go())
    assert "get_balance" not in world.lam.ops()
