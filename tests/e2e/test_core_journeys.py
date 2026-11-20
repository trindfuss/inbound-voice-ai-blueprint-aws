"""Core suite, offline: balance, step-up then payment, escalation.

Each test drives one whole call through the real front door, auth service, bank
tools and bridge (TAC v2.5.0), with a scripted agent. Pass criteria are read from
system state (segment log prompt keys, back-end state, task attributes), not from
matching words in the transcript. The security scan runs at teardown (conftest).
"""

from __future__ import annotations

import json
from decimal import Decimal

from owlbank import tokens
from owlbank.tools_contract import AUTH_LEVEL_INSUFFICIENT

from tests.component.masking_fixtures import unmasked_policy_doc
from tests.e2e.harness import (
    CONFIRMATION,
    OTP,
    THOMAS,
    World,
    call_sid,
    factor,
    result_then_words,
    run,
    say,
    tool,
)


async def login_thomas(world: World, sid: str):
    call = world.dial(THOMAS, sid)
    await call.start()
    assert call.state == "gather"  # "Welcome to OwlBank." "Para español, oprima dos."
    await call.press("")  # no input: en-US
    assert call.state == "gather"
    await call.press(factor("C-1001", "zip"))
    assert call.state == "relay"
    return call




def test_balance_expected_conversation(make_world) -> None:
    """Handover "Expected conversation": ask the account, read it, follow up, compare."""
    # The exact words of the handover script: masking phrases off (tests/component/test_bridge_masking.py).
    world: World = make_world(policy_doc=unmasked_policy_doc())
    sid = call_sid(1)
    seen: dict[str, dict] = {}

    def checking(rec):
        assert rec["user_text"] == "Checking."
        return tool("get_balance", "tu_bal1", account="checking")

    def checking_read(env):
        seen["checking"] = env
        assert env["status"] == "completed" and env["tool"] == "get_balance"
        assert [s["template"] for s in env["spoken"]] == [
            "balance.read", "balance.read.pending", "common.anything_else"]

    def savings_read(env):
        seen["savings"] = env

    def compare(rec):
        diff = Decimal(seen["savings"]["data"]["available"]) - Decimal(seen["checking"]["data"]["available"])
        return say(f"You have {diff} dollars more in savings than in checking.")

    # Each read-out's result goes back first (the agent stays silent), then the caller's
    # words as caller text.
    world.agent.queue(
        say("Which account would you like the balance for, checking or savings?"),
        checking,
        *result_then_words(world.agent, "What about my savings?",
                           tool("get_balance", "tu_bal2", account="savings"),  # no re-asking
                           on_result=checking_read),
        *result_then_words(world.agent, "So how much more is in savings than checking?", compare,
                           on_result=savings_read),
    )

    async def go():
        call = await login_thomas(world, sid)
        assert call.texts("welcome") == [
            "Hi Thomas. I am the OwlBank virtual agent. How can I help you today?"]
        assert "Thomas" not in " ".join(call.texts("say"))  # no name before verification

        reply = await call.say("What's my balance?")
        assert reply == "Which account would you like the balance for, checking or savings?"

        mark = call.mark()
        await call.say("Checking.")
        assert call.state == "relay" and call.relay_opens == 2  # handed off, read, reconnected
        # The read-out and the follow-up are one interruptible welcomeGreeting on the
        # reconnect, in order, joined by single spaces; nothing is <Say>'d before it.
        assert call.since(mark, "say") == []
        assert call.since(mark) == call.since(mark, "welcome") == [" ".join([
            "Your checking account ending in four, eight, two, one has an available balance of "
            "one thousand two hundred thirty-four dollars and fifty-six cents. Your current "
            "balance is one thousand three hundred ten dollars and six cents.",
            "That includes seventy-five dollars and fifty cents in pending transactions.",
            "Is there anything else I can help you with?",
        ])]
        assert call.since(mark, "ai") == []  # await_caller: the agent said nothing
        turns_before = len(world.agent.calls)
        assert world.envelope(sid).next == "await_caller"  # held until the caller speaks

        mark = call.mark()
        await call.say("What about my savings?")
        assert len(world.agent.calls) == turns_before + 2  # the result, then the words: one tool call
        assert call.since(mark, "say") == []
        assert any("seven, seven, three, zero" in t for t in call.since(mark, "welcome"))

        reply = await call.say("So how much more is in savings than checking?")
        assert reply == "You have 7185.44 dollars more in savings than in checking."
        await call.hangup()
        return call

    call = run(go())
    assert call.handoffs[0] == {**call.handoffs[0], "reason": "tool", "tool": "get_balance",
                                "args": {"account": "checking"}}
    assert [h["args"]["account"] for h in call.handoffs] == ["checking", "savings"]
    # Segment log: verified at level 1, then balance.read with the seeded values.
    journal = world.journal(sid)
    verified = [e for e in journal if e["type"] == "verification" and e.get("outcome") == "verified"]
    assert verified and verified[0]["level"] == 1
    reads = [e for e in journal if e["type"] == "prompt" and e.get("prompt_key") == "balance.read"]
    assert len(reads) == 2
    assert "one thousand two hundred thirty-four dollars and fifty-six cents" in reads[0]["rendered_text"]
    assert "eight thousand four hundred twenty dollars" in reads[1]["rendered_text"]
    assert seen["checking"]["data"]["available"] == "1234.56"  # what the agent got = seeded data
    # Call end: token revoked, call marked ended, Orchestrator conversation closed.
    assert tokens.validate(world.store, world.raw_tokens[-1], call_sid=sid, min_level=1,
                           now=int(world.clock())) == tokens.AuthError(tokens.AuthErrorCode.REVOKED)
    assert world.state(sid).sealed_token is None
    assert world.session(sid).ended
    assert world.closer.closed == [(sid, call.handoffs[0]["conversationId"])]




def test_step_up_then_payment(world: World) -> None:
    """A level-1 make_payment hands off for step-up at once (one model hop, no
    AUTH_LEVEL_INSUFFICIENT round trip) -> only the missing factors -> level 3 -> straight to the
    deterministic read-back (no AI hop in between), caller says yes -> the front door runs
    it once -> the result answers the model's ORIGINAL make_payment call -> the mock bank shows
    the payment."""
    sid = call_sid(2)
    pay = {"from_account": "checking", "to": "red", "amount": "50.00"}
    seen: dict[str, dict] = {}

    def confirmed(env):
        seen["env"] = env

    world.agent.queue(
        tool("make_payment", "tu_pay1", **pay),
        *result_then_words(world.agent, "No, that's everything.", say("Thanks for calling OwlBank."),
                           on_result=confirmed))

    async def go():
        call = await login_thomas(world, sid)
        assert world.state(sid).level == 1
        mark = call.mark()
        turns = len(world.agent.calls)
        await call.say("Pay fifty dollars toward my credit card from checking.")
        assert len(world.agent.calls) == turns + 1  # one model hop: the bridge started the step-up
        # Handed off: make_payment's hand-off phrase on the relay first (no silence before the
        # step-up), then the connecting
        # phrase from the policy's phrase key, then only the missing factors.
        assert call.state == "gather"
        assert call.since(mark) == [
            "Let me set that up. ",
            "I can help you make that payment. First I need to ask you a few more "
            "questions to verify your identity.",
            "Please say or enter the last four digits of your Social Security number. If you can't, press pound.",
        ]
        await call.press(factor("C-1001", "ssn4"))
        assert call.state == "gather"
        assert call.texts("say")[-1].startswith("We just sent a six-digit code")
        assert world.verify.sent == [(THOMAS, "sms")]
        assert world.verify.locales == ["en-US"]  # the call's language goes to Verify as the Locale
        mark = call.mark()
        turns = len(world.agent.calls)
        relays = call.relay_opens
        await call.press(OTP)
        # Verified: the success line and the read-back in the same response, no AI hop and no
        # ConversationRelay connect in between.
        assert call.state == "gather"
        assert call.relay_opens == relays and len(world.agent.calls) == turns
        assert call.since(mark) == [
            "Thanks, you're verified.",
            "To confirm, you'd like to pay fifty dollars toward your OwlBank Red Card from your "
            "checking account. Is that right? Say yes or no, or press 1 for yes and 2 for no."]
        assert [op for fn, op in world.lam.calls if op == "make_payment"] == []  # not yet
        mark = call.mark()
        await call.speak_to_gather("Yes.")
        assert call.state == "relay"
        # The outcome is a plain <Say> (it can't be cut off); only the follow-up
        # is the interruptible welcome greeting.
        assert call.since(mark) == [
            "Done. Your payment of fifty dollars to your OwlBank Red Card from your checking "
            "account is complete. Your confirmation number is eight, zero, seven, zero, six, zero, five, zero.",
            "Is there anything else I can help you with?"]
        assert call.since(mark, "welcome") == ["Is there anything else I can help you with?"]
        reply = await call.say("No, that's everything.")
        assert reply == "Thanks for calling OwlBank."
        await call.hangup()
        return call

    call = run(go())
    # The result answers the model's original call, with everything the caller heard.
    (res,) = world.agent.calls[-2]["tool_results"]
    assert res.tool_use_id == "tu_pay1"
    env = seen["env"]
    assert env["status"] == "completed" and env["tool"] == "make_payment"
    assert env["data"]["confirmation"] == CONFIRMATION and env["data"]["executed"] is True
    assert [s["template"] for s in env["spoken"]] == [
        "stepup.intro", "factor.ssn4", "factor.ack", "factor.otp", "stepup.success",
        "confirm.make_payment", "confirm.make_payment.done", "common.anything_else"]
    results = [r.content for c in world.agent.calls for r in c["tool_results"]]
    assert AUTH_LEVEL_INSUFFICIENT not in json.dumps(results)  # never refused to the model
    (handoff,) = call.handoffs
    assert handoff["reason"] == "step-up" and handoff["tool"] == "make_payment"
    assert "50" not in json.dumps(handoff)
    # Only the missing factors were asked: zip at login, then ssn4 and otp (never zip again).
    asked = [e["factor"] for e in world.journal(sid) if e["type"] == "factor_asked"]
    assert asked == ["zip", "ssn4", "otp"]
    assert world.state(sid).level == 3
    # The mock bank shows the payment.
    cust = world.customer("C-1001")
    checking = next(a for a in cust["accounts"] if a["type"] == "checking")
    card = next(a for a in cust["accounts"] if a["type"] == "credit_card")
    assert checking["available"] == "1184.56" and card["current"] == "562.40"
    assert card["transactions"][0]["confirmation"] == CONFIRMATION
    # The confirmed payment ran exactly once, from the front door, keyed by the model's
    # original tool_use_id.
    pays = [p for p in world.lam.payloads if p.get("op") == "make_payment"]
    assert [p["idempotency_key"] for p in pays] == ["tu_pay1"]
    answers = [e["answer"] for e in world.journal(sid) if e["type"] == "confirm_answer"]
    assert answers == ["yes"]




def test_escalation_to_a_human(world: World) -> None:
    """escalate -> "please hold" (escalate.default) -> <Enqueue> with the task attributes.
    The test ends there (no human needed), then fetches the Flex Context API timeline."""
    sid = call_sid(3)
    summary = "Caller asked to speak to a person about his account. Verified at level 1."
    world.agent.queue(tool("escalate", "tu_esc", reason="caller_requested_human", summary=summary))

    async def go():
        call = await login_thomas(world, sid)
        mark = call.mark()
        await call.say("I want to speak to a person.")
        assert call.state == "queued"
        assert call.since(mark) == [
            "I'll connect you with a specialist who can help with that. Please hold for a moment."]
        await call.hangup()  # the runner hangs up and cancels the task
        return call

    call = run(go())
    enq = call.enqueue
    assert enq is not None and enq.get("workflowSid") == "WW" + "a" * 32
    assert enq.get("waitUrl") is None and enq.get("action") == "/prod/voice/enqueue-action"
    attrs = call.task
    assert attrs["escalation_reason"] == "caller_requested_human"
    assert attrs["language"] == "en-US"
    assert attrs["ai_summary"] == summary  # model text, cleaned; kept server-side, not in HandoffData
    assert attrs["conversation_id"] == call.handoffs[-1]["conversationId"] and attrs["conversation_id"]
    assert attrs["profile_id"] == "mem_profile_verified_c1001"  # the verified Memory profile
    assert attrs["name"] == "Thomas Doe" and attrs["customer_id"] == "C-1001"
    assert attrs["auth_level"] == 1 and attrs["call_sid"] == sid and attrs["test"] is True
    assert "escalate.default" in world.prompt_keys(sid)
    assert "summary" not in call.handoffs[-1] and summary not in str(call.handoffs[-1])
    # The Flex screen pop: the task's journal_ref opens the Context API timeline.
    ref = attrs["journal_ref"]
    assert ref == call.handoffs[-1]["journal"] == world.state(sid).escalation["journal_ref"]
    assert len(ref) >= 20 and sid not in ref and not ref.startswith(("seg:", "call:"))  # opaque
    resp = world.context(ref, attrs["call_sid"])
    assert resp["statusCode"] == 200, resp
    body = json.loads(resp["body"])
    assert body["escalation_reason"] == "caller_requested_human" and body["verified_level"] == 1
    types = [e["type"] for e in body["timeline"]]
    assert "escalation" in types and "ai_turn" not in types
    # A wrong ref gets the same 404 as an unknown call.
    assert world.context(ref + "x", sid)["statusCode"] == 404
    # The AI session is over: the call's token no longer works.
    assert tokens.validate(world.store, world.raw_tokens[-1], call_sid=sid, min_level=1,
                           now=int(world.clock())) == tokens.AuthError(tokens.AuthErrorCode.REVOKED)
