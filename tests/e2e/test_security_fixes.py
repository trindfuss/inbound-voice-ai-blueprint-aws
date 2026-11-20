"""Security fixes end to end: PII masking, the one-time relay nonce and idempotent
payments, through the real front door, bridge, auth service and bank tools.
The leak scan runs at teardown (conftest)."""

from __future__ import annotations

import copy
import json
import re
from decimal import Decimal
from typing import Any

from owlbank.pii import luhn_ok

from tests.e2e.harness import (
    CONFIRMATION,
    DATA,
    OTP,
    THOMAS,
    RelaySession,
    World,
    call_sid,
    factor,
    result_then_words,
    run,
    say,
    tool,
)
from tests.e2e.test_core_journeys import login_thomas


def _card() -> str:
    body = "453914880343646"
    return body + next(d for d in "0123456789" if luhn_ok(body + d))


CARD = _card()                                     # a Luhn-valid 16-digit card number
CARD_SPOKEN = " ".join(CARD[i:i + 4] for i in range(0, 16, 4))
SSN = "387-29-6075"
SSN_DIGITS = SSN.replace("-", "")
WITHHELD_EN = "Summary withheld: PII check unavailable."


def _digits(text: str) -> str:
    return re.sub(r"[\s\-]", "", text)


def _assert_absent(world: World, sid: str, attrs: dict[str, Any], context_body: str) -> None:
    surfaces = {
        "task attributes": json.dumps(attrs),
        "call state": json.dumps(world.store.get("calls", {"call_sid": sid})),
        "segments": json.dumps(world.journal(sid)),
        "context api": context_body,
        "all stored": " ".join(world.stored_strings(skip=frozenset())),
    }
    for where, text in surfaces.items():
        flat = _digits(text)
        assert CARD not in flat, f"card number in {where}"
        assert SSN_DIGITS not in flat, f"SSN in {where}"


def _pii_escalation(world: World, sid: str) -> tuple[Any, dict[str, Any]]:
    """Caller reads a full card number and SSN to the AI; the AI repeats them, then escalates
    with a summary containing both."""
    world.agent.queue(
        say(f"Thanks. I have card {CARD_SPOKEN} and Social Security number {SSN}."),
        tool("escalate", "tu_esc_pii", reason="caller_requested_human",
             summary=f"Caller read card {CARD_SPOKEN} and SSN {SSN}; wants a person."),
    )

    async def go():
        call = await login_thomas(world, sid)
        await call.say(f"My card is {CARD_SPOKEN} and my social is {SSN}.")
        assert call.state == "relay"
        await call.say("Let me talk to a person.")
        assert call.state == "queued"  # the escalation enqueued
        await call.hangup()
        return call

    call = run(go())
    resp = world.context(call.task["journal_ref"], sid)
    assert resp["statusCode"] == 200
    return call, resp


def test_pii_never_reaches_task_attributes_state_segments_or_context(world: World) -> None:
    sid = call_sid(40)
    call, resp = _pii_escalation(world, sid)
    attrs = call.task
    assert attrs["escalation_reason"] == "caller_requested_human"
    assert attrs["ai_summary"].startswith("Caller read card [CREDIT_DEBIT_NUMBER] and SSN "
                                          "[SSN]")
    turns = [e for e in world.journal(sid) if e["type"] == "ai_turn"]
    assert turns and all("[CREDIT_DEBIT_NUMBER]" in e["text"] for e in turns if "card" in e["text"])
    # Defence in depth: the front door checked the (already masked) summary again.
    assert world.pii.calls >= 3  # bridge turn + bridge summary + front door summary
    _assert_absent(world, sid, attrs, resp["body"])


def test_comprehend_down_withholds_and_still_enqueues(world: World) -> None:
    sid = call_sid(41)
    world.pii.fail = True
    call, resp = _pii_escalation(world, sid)
    assert call.state == "ended" and call.enqueue is not None  # queued, then the caller hung up
    assert call.task["ai_summary"] == WITHHELD_EN
    assert world.state(sid).escalation["reason"] == "caller_requested_human"
    assert all(e["text"] == "[WITHHELD]" for e in world.journal(sid) if e["type"] == "ai_turn")
    _assert_absent(world, sid, call.task, resp["body"])


def _policy_with(**redaction: Any) -> dict[str, Any]:
    doc = json.loads((DATA / "policy.json").read_text())
    doc["redaction"] = {**copy.deepcopy(doc["redaction"]), **redaction}
    return doc


def _one_ai_turn(world: World, sid: str, text: str) -> str:
    world.agent.queue(say(text))

    async def go():
        call = await login_thomas(world, sid)
        await call.say("hello")
        await call.hangup()

    run(go())
    (turn,) = [e for e in world.journal(sid) if e["type"] == "ai_turn"]
    return turn["text"]


AI_TEXT = "Your PIN is 7302 and we'll mail it to 77 Oak Street."


def test_default_config_masks_the_pin_but_not_the_address(make_world) -> None:
    world = make_world()
    out = _one_ai_turn(world, call_sid(42), AI_TEXT)
    assert out == "Your PIN is [PIN] and we'll mail it to 77 Oak Street."


def test_custom_config_adds_address_and_drops_pin(make_world) -> None:
    types = [t for t in json.loads((DATA / "policy.json").read_text())["redaction"]["entity_types"]
             if t != "PIN"] + ["ADDRESS"]
    world = make_world(policy_doc=_policy_with(entity_types=types, mask="<{type}>"))
    out = _one_ai_turn(world, call_sid(43), AI_TEXT)
    assert out == "Your PIN is 7302 and we'll mail it to <ADDRESS>."
    assert world.store.get("policy", {"pk": "redaction"})["entity_types"] == types  # seeded


def test_redaction_disabled_makes_no_pii_call(make_world) -> None:
    world = make_world(policy_doc=_policy_with(enabled=False))
    out = _one_ai_turn(world, call_sid(44), AI_TEXT)
    assert out == AI_TEXT and world.pii.calls == 0


# --------------------------------------------------------------------------- idempotent payments


def test_payment_timeout_and_retry_moves_money_once(world: World) -> None:
    sid = call_sid(45)
    pay = {"from_account": "checking", "to": "red", "amount": "50.00"}
    real = world.lam.functions["owlbank-bank"]
    state = {"dropped": 0}

    def flaky_bank(event: dict[str, Any]) -> Any:
        result = real(event)
        if event.get("op") == "make_payment" and "data" in result and not state["dropped"]:
            state["dropped"] = 1
            raise TimeoutError("reply lost after the bank moved the money")
        return result

    world.lam.functions = {**world.lam.functions, "owlbank-bank": flaky_bank}

    def confirm(env):  # the front door ran it after the step-up and the caller's yes
        assert env["status"] == "completed" and env["data"]["confirmation"] == CONFIRMATION

    world.agent.queue(tool("make_payment", "tu_pay1", **pay),
                      *result_then_words(world.agent, "Thanks.", say("You're welcome."),
                                         on_result=confirm))

    async def go():
        call = await login_thomas(world, sid)
        await call.say("Pay fifty dollars toward my credit card from checking.")
        await call.press(factor("C-1001", "ssn4"))
        await call.press(OTP)
        assert call.state == "gather"  # the read-back: yes or no
        await call.press("1")
        assert call.state == "relay"
        await call.say("Thanks.")
        await call.hangup()

    run(go())
    assert state["dropped"] == 1
    payloads = [p for p in world.lam.payloads if p.get("op") == "make_payment"]
    # tu_pay1 was held for the step-up by the bridge's level check; once confirmed it is
    # retried once by the front door with the same key after the lost reply.
    assert [p.get("idempotency_key") for p in payloads] == ["tu_pay1", "tu_pay1"]
    cust = world.customer("C-1001")
    checking = next(a for a in cust["accounts"] if a["type"] == "checking")
    card = next(a for a in cust["accounts"] if a["type"] == "credit_card")
    assert Decimal(checking["available"]) == Decimal("1184.56")  # debited once
    assert [t.get("confirmation") for t in card["transactions"]].count(CONFIRMATION) == 1
    rec = world.store.get("calls", {"call_sid": f"{sid}#idem#tu_pay1"})
    assert rec["state"] == "done"


# --------------------------------------------------------------------------- relay nonce


def test_replayed_relay_nonces_are_refused_after_a_real_call(world: World) -> None:
    sid, other = call_sid(46), call_sid(47)
    world.agent.queue(say("Hello Thomas."), say("Hello again."))

    async def go():
        call = await login_thomas(world, sid)
        await call.say("hi")
        first_nonce = world.relay_nonces[-1]
        second = await login_thomas(world, other)
        await second.say("hi")
        turns = len(world.agent.calls)
        for target, nonce in ((sid, first_nonce), (sid, world.relay_nonces[-1]), (other, None)):
            replay = RelaySession(world, target, THOMAS, {"callSid": target}, nonce)
            replay.task = __import__("asyncio").create_task(world.voice.handle_websocket(replay.peek))
            await replay.socket.inbound.put({"type": "setup", "callSid": target, "from": THOMAS,
                                             "customParameters": {"callSid": target}})
            await replay.socket.inbound.put({"type": "prompt", "voicePrompt": "pay 900 dollars",
                                             "lang": "en-US", "last": True})
            await __import__("asyncio").wait_for(replay.task, 5)
            assert replay.socket.closed and not replay.peek.verified
        assert len(world.agent.calls) == turns  # nothing ran
        await call.hangup()
        await second.hangup()

    run(go())
    assert all(world.state(s).ws_nonce_hash is None for s in (sid, other))
