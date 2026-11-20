"""Offline end to end: a one-time code goes only to the calling number, and only when it
is on file for the verified customer. from the number on file still
works (see ``test_core_journeys``); here the same step-up from a phone that is not on file ends
in a transfer. The security (leak) scan runs at teardown (conftest)."""

from __future__ import annotations


from tests.e2e.harness import THOMAS, World, call_sid, factor, run, tool

ALIAS = "+15005550006"  # a registered test number for Thomas, not his phone on file
ALIAS_KEY = {"customer_id": "ani#A#C-1001"}
PHRASE = "For your security, I'll connect you with a specialist who can help finish this."


def alias_record() -> dict:
    return {**ALIAS_KEY, "phone": ALIAS, "alias_of": "C-1001", "kind": "ani_alias", "label": "A"}


def queue_payment(world: World) -> None:
    # Refused for the level only, so the bridge hands off for step-up at once.
    world.agent.queue(
        tool("make_payment", "tu_pay", from_account="checking", to="red", amount="50.00"))


def test_from_an_alias_number_sends_the_code_to_that_number(world: World) -> None:
    """The ANI is a registered alias for Thomas: the code goes to it, not to his phone on file."""
    sid = call_sid(50)
    world.store.put("customers", alias_record())
    queue_payment(world)

    async def go():
        call = world.dial(ALIAS, sid)
        await call.start()
        await call.press("")
        await call.press(factor("C-1001", "zip"))
        assert call.state == "relay"
        await call.say("Pay fifty dollars toward my credit card from checking.")
        await call.press(factor("C-1001", "ssn4"))
        assert call.state == "gather"
        assert call.texts("say")[-1].startswith("We just sent a six-digit code")
        await call.hangup()

    run(go())
    assert world.verify.sent == [(ALIAS, "sms")]
    assert THOMAS not in [to for to, _ in world.verify.sent]


def test_from_another_phone_ends_in_otp_unavailable_and_no_code_is_sent(world: World) -> None:
    """Thomas's own number is on file but the call's number (ANI) is a different one that no
    longer resolves to him: no code goes anywhere, the caller hears the neutral phrase and
    is queued with reason ``otp_unavailable``. Before the code went to his phone on file."""
    sid = call_sid(51)
    world.store.put("customers", alias_record())
    queue_payment(world)

    async def go():
        call = world.dial(ALIAS, sid)
        await call.start()
        await call.press("")
        await call.press(factor("C-1001", "zip"))
        assert call.state == "relay"
        await call.say("Pay fifty dollars toward my credit card from checking.")
        assert call.state == "gather"  # step-up: the SSN-4 question
        world.store.delete("customers", ALIAS_KEY)  # the number is no longer on file for him
        await call.press(factor("C-1001", "ssn4"))
        assert call.state == "queued"
        assert call.texts("say")[-1] == PHRASE
        assert not any("six-digit code" in t for t in call.texts("say"))
        task = call.task
        assert task is not None
        assert task["escalation_reason"] == "otp_unavailable"
        assert task["customer_id"] == "C-1001" and task["auth_level"] == 2  # as far as he got
        assert call.enqueue is not None and call.enqueue.get("workflowSid")
        await call.hangup()

    run(go())
    assert world.verify.sent == []  # the fake Verify client never received a send
    st = world.state(sid)
    assert st.escalation and st.escalation["reason"] == "otp_unavailable"
    assert st.escalation["queue"] == "OwlBank Support EN"  # the default human queue, not fraud
    sends = [p for p in world.lam.payloads if p.get("op") == "send_otp"]
    assert len(sends) == 1 and sends[0]["language"] == "en-US"
    events = [e for e in world.journal(sid) if e["type"] == "verification"]
    assert events[-1]["outcome"] == "otp_unavailable"
