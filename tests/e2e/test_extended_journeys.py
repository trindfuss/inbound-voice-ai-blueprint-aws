"""Extended suite, offline: the handover's step-up intent switch,
``await_caller`` on return, shared-household disambiguation, the WebSocket-drop
fallback and escalation routing. The security scan runs at teardown (conftest)."""

from __future__ import annotations

import json

import pytest

from owlbank.slots import FakeResolver, ResolverOutput
from owlbank.tools_contract import AUTH_LEVEL_INSUFFICIENT

from tests.component.masking_fixtures import unmasked_policy_doc
from tests.e2e.harness import (
    LEES,
    MARIA,
    NOBODY,
    THOMAS,
    World,
    call_sid,
    factor,
    result_then_words,
    run,
    say,
    tool,
)
from tests.e2e.test_core_journeys import login_thomas

INTENT_SWITCH = ResolverOutput(kind="intent_switch")


def bank_ops(world: World) -> list[str]:
    return [op for fn, op in world.lam.calls if fn == "owlbank-bank"]




def test_failed_verification_caps_retries_then_falls_back(make_world) -> None:
    """Thomas with a wrong ZIP: retries capped, the fallback ran, no customer data spoken. The
    bank chose ``escalate`` for ``verify_failed`` (``OWLBANK_FALLBACKS``; the default is the AI)."""
    world: World = make_world(config={"fallbacks": {"verify_failed": "escalate"}})
    sid = call_sid(10)

    async def go():
        call = world.dial(THOMAS, sid)
        await call.start()
        await call.press("")
        for wrong in ("94106", "94107"):
            await call.press(wrong)
            assert call.state == "gather"
            assert call.texts("say")[-2:] == ["Sorry, that didn't match our records. Let's try again.",
                                             "Please say or enter your five-digit ZIP code. If you can't, press pound."]
        await call.press("94108")  # third failure: verify_attempts_per_call = 3
        assert call.state == "queued"  # the bank's chosen verify_failed fallback: a human
        await call.hangup()
        return call

    call = run(go())
    assert call.texts("say")[-2:] == [
        "I'm sorry, I wasn't able to verify your identity.",
        "I wasn't able to verify you, so I'll connect you with a specialist who can help."]
    # No customer data in the flow's speech, and the AI never ran.
    heard = " ".join(call.texts())
    for leak in ("Thomas", "Doe", "4821", "four, eight, two, one", "thousand"):
        assert leak not in heard
    assert world.agent.calls == [] and call.relay_opens == 0
    attempts = [e["outcome"] for e in world.journal(sid) if e["type"] == "verification"
                and e.get("outcome") in ("not_verified", "locked")]
    assert attempts == ["not_verified", "not_verified", "locked"]  # the auth service's cap
    st = world.state(sid)
    assert st.level == 0 and st.verified_customer_id is None and st.sealed_token is None
    attrs = call.task
    assert attrs["escalation_reason"] == "step_up_failed" and attrs["auth_level"] == 0
    assert attrs["name"] is None and attrs["customer_id"] is None
    assert world.raw_tokens == []  # no token was ever issued
    # The front-door-only escalation's ref also opens the Context API (item 4a).
    assert world.context(attrs["journal_ref"], sid)["statusCode"] == 200




def test_intent_switch_at_the_zip_prompt(make_world) -> None:
    """At the login ZIP prompt Thomas asks "are you open tomorrow?": the flow ends
    ``interrupted``, the agent answers from the FAQ tool, and the call continues."""
    world: World = make_world(resolver=FakeResolver([INTENT_SWITCH]),
                              # Speech at a factor prompt needs spoken factors on.
                              config={"accept_spoken_factors": True})
    sid = call_sid(11)
    utterance = "Actually, are you open tomorrow?"

    def answer(rec):
        assert rec["user_text"] == utterance and rec["tool_results"] == []
        assert "<caller_context>\nVerified: no." in rec["system_prompt"]  # the unverified frame
        hours = world.gateway.call("get_branch_hours", {"lang": "en-US"})
        return say(f"Yes. Our Downtown branch is open {hours['branches'][0]['hours']}")

    # Still unverified: the bridge's level check hands the get_balance call off for
    # step-up at once, and after the step-up the front door reads the balance itself.
    world.agent.queue(answer, tool("get_balance", "tu_bal1", account="checking"))

    async def go():
        call = world.dial(THOMAS, sid)
        await call.start()
        await call.press("")
        mark = call.mark()
        await call.speak_to_gather(utterance)
        # The bridge answers at setup, without waiting for another prompt.
        assert call.state == "relay"
        assert call.since(mark, "welcome") == ["One moment while I get that for you."]
        assert call.since(mark, "ai")[0].startswith("Yes. Our Downtown branch is open Monday to Friday")
        # The call continues: the caller can still verify and get a balance.
        await call.say("Thanks. And what's my checking balance?")
        assert call.state == "gather"
        assert call.texts("say")[-2:] == [
            "I can help you check your balance. First I need to ask you a few more "
            "questions to verify your identity.", "Please say or enter your five-digit ZIP code. If you can't, press pound."]
        mark = call.mark()
        turns = len(world.agent.calls)
        await call.press(factor("C-1001", "zip"))
        # Verified: the pending get_balance runs at once (no model hop) and is read out.
        assert len(world.agent.calls) == turns
        # Success line and read-out are one (interruptible) welcomeGreeting, not a <Say>.
        assert call.since(mark, "say") == []
        (welcome,) = call.since(mark, "welcome")
        assert welcome.startswith("Thanks, you're verified. Your checking account ending in four, eight")
        assert welcome.endswith(" Is there anything else I can help you with?")
        await call.hangup()
        return call

    run(go())
    journal = world.journal(sid)
    assert any(e["type"] == "verification" and e.get("outcome") == "off_script_intent_switch"
               for e in journal)
    assert any(e["type"] == "handoff" and e.get("to") == "ai" and e.get("outcome") == "interrupted"
               for e in journal)
    assert world.gateway.calls == [("get_branch_hours", {"lang": "en-US"})]
    # The interrupted envelope was delivered once; what's held now is the balance read's.
    assert world.state(sid).interrupted_utterance is None
    env = world.envelope(sid)
    assert env.tool == "get_balance" and env.next == "await_caller"
    assert [s.template for s in env.spoken][:3] == ["stepup.intro", "factor.zip", "stepup.success"]
    assert world.state(sid).pending_tool.tool_use_id == "tu_bal1"  # the model's original call




def test_unknown_caller_gets_only_level_0_tools(world: World) -> None:
    sid = call_sid(12)

    def hours(rec):
        # The unverified frame (memory gating): no name, profile or Memory.
        assert "<caller_context>\nVerified: no." in rec["system_prompt"]
        assert "Recent interactions from other channels:" not in rec["system_prompt"]
        result = world.gateway.call("faq_lookup", {"question": "How do I report a lost card?"})
        return say(result["answer"])

    def refused_transactions(rec):
        (res,) = rec["tool_results"]
        assert res.content == {"error": AUTH_LEVEL_INSUFFICIENT, "tool": "list_recent_transactions"}
        return tool("authenticate", "tu_auth", tool="list_recent_transactions")

    def not_verified(env):
        assert env["status"] == "not_verified" and env["tool"] == "authenticate"

    world.agent.queue(hours, tool("list_recent_transactions", "tu_tx", account="checking"),
                      refused_transactions,
                      *result_then_words(world.agent, "Okay.", say(
                          "I can't open your account on this call, but I can help with general questions."),
                          on_result=not_verified))

    async def go():
        call = world.dial(NOBODY, sid)
        await call.start()
        await call.press("")
        # No ANI match: asked for the account's phone (keyed-phone login); two unknown numbers
        # lead to the unauthenticated AI.
        await call.press("2025550177")
        await call.press("2025550188")
        assert call.state == "relay"
        assert call.texts() == ["Welcome to OwlBank. This call may be recorded and transcribed for quality and training.", "Para español, oprima dos.",
                                "What is the phone number associated with your account? You can say or enter it. If you don't have an account with us, press pound.",
                                "Please say or enter the 10-digit phone number on your account. If you don't have an account with us, press pound.",
                                "I am the OwlBank virtual agent. How can I help you today?"]
        reply = await call.say("What do I do if I lose my card?")
        assert "card" in reply.lower()
        mark = call.mark()
        await call.say("Can you read me my recent transactions?")
        # No candidate record: the no-match fallback (limited AI), with no step-up intro.
        assert call.state == "relay"
        assert call.since(mark) == [
            "I'm sorry, I wasn't able to verify your identity, so I can't do that right now. Is there anything else I can help you with?"]
        reply = await call.say("Okay.")
        assert reply.startswith("I can't open your account on this call")
        await call.hangup()
        return call

    run(go())
    assert world.state(sid).candidate_ids == () and world.state(sid).level == 0
    # The ANI, then the two keyed numbers; no customer-data operation ever reached the bank.
    assert bank_ops(world) == ["ani_lookup"] * 3
    assert world.raw_tokens == []
    assert "Thomas" not in world.agent.inputs_text()  # no customer data in any model input


# --------------------------------------------------------------------------- intent switch in step-up


def test_intent_switch_during_step_up_reads_the_balance_instead(make_world) -> None:
    """Handover "Done when": asked for verification during step-up, the caller says
    "Actually, can you give me my checking balance instead?" The flow stops, the agent
    gets ``interrupted`` with the utterance, abandons the payment and reads the balance.
    Digits spoken during the SSN question reach the resolver unredacted but never the
    agent or the journal: the resolver gave no value, so it is an intent switch."""
    world: World = make_world(resolver=FakeResolver([INTENT_SWITCH]),
                              # Speech at a factor prompt needs spoken factors on.
                              config={"accept_spoken_factors": True})
    sid = call_sid(13)
    pay = {"from_account": "checking", "to": "red", "amount": "50.00"}
    ssn4 = factor("C-1001", "ssn4")
    utterance = f"Actually forget it, {ssn4}, can you give me my checking balance instead?"

    def switched(rec):
        # The interrupted result first (it answers the original make_payment call)...
        env = world.agent.envelope(rec)
        assert rec["tool_results"][0].tool_use_id == "tu_pay"
        assert env["status"] == "interrupted" and env["tool"] == "make_payment"
        assert env["next"] == "await_caller" and "callerUtterance" not in env
        assert env.get("completed") is False
        return say("<silent/>")

    def words(rec):
        # ...then the caller's (redacted) words as their own turn.
        assert rec["tool_results"] == []
        assert ssn4 not in rec["user_text"] and "checking balance" in rec["user_text"]
        return tool("get_balance", "tu_bal", account="checking")

    world.agent.queue(tool("make_payment", "tu_pay", **pay), switched, words)

    async def go():
        call = await login_thomas(world, sid)
        await call.say("I'd like to pay fifty dollars on my card.")
        assert call.texts("say")[-1] == "Please say or enter the last four digits of your Social Security number. If you can't, press pound."
        mark = call.mark()
        await call.speak_to_gather(utterance)
        # Resumed at setup; the agent's reply was itself a deterministic tool: read, then reconnect.
        assert call.state == "relay"
        # Read-out and follow-up are one interruptible welcomeGreeting, read-out first.
        assert call.since(mark, "say") == []
        assert call.texts("welcome")[-1].startswith("Your checking account ending in four, eight")
        assert call.texts("welcome")[-1].endswith(" Is there anything else I can help you with?")
        await call.hangup()
        return call

    run(go())
    assert world.resolver.calls[0]["req"].utterance == utterance  # Unredacted
    # The payment was abandoned: the level-1 call was held for the step-up by the bridge's level
    # check (deterministic confirm) and never reached the bank.
    assert bank_ops(world).count("make_payment") == 0
    assert world.state(sid).level == 1  # nothing was verified during the interrupted flow


# --------------------------------------------------------------------------- await_caller


def test_await_caller_return_is_silent_then_the_next_prompt_carries_the_envelope(make_world) -> None:
    """María (es-US) asks for savings she doesn't have: ``not_found`` + ``await_caller``. On
    reconnect only ``common.anything_else`` plays; the agent isn't invoked until she speaks,
    and then gets the envelope, then her words as caller text."""
    # Exact words and timing of the hand-offs: masking phrases off
    # (tests/component/test_bridge_masking.py).
    world: World = make_world(policy_doc=unmasked_policy_doc())
    sid = call_sid(14)

    def not_found(env):
        assert env["status"] == "not_found" and env["tool"] == "get_balance"
        assert [s["template"] for s in env["spoken"]] == ["balance.no_account", "common.anything_else"]

    world.agent.queue(tool("get_balance", "tu_bal1", account="savings"),
                      *result_then_words(world.agent, "¿Y mi cuenta de cheques?",
                                         tool("get_balance", "tu_bal2", account="checking"),
                                         on_result=not_found))

    async def go():
        call = world.dial(MARIA, sid)
        await call.start()
        await call.press("2")
        assert call.texts("say")[-2:] == ["Bienvenido a OwlBank. Esta llamada puede ser grabada y transcrita con fines de calidad y capacitación.",
                                         "Por favor, diga o ingrese su código postal de cinco dígitos. Si no puede, presione numeral."]
        await call.press(factor("C-1004", "zip"))
        assert call.texts("welcome")[-1].startswith("Hola, María.")
        mark = call.mark()
        await call.say("¿Cuál es el saldo de mi cuenta de ahorros?", "es-US")
        turns = len(world.agent.calls)
        assert call.since(mark) == call.since(mark, "welcome") == [  # one greeting
            "No veo una cuenta de ahorros en su perfil. ¿Hay algo más en lo que pueda ayudarle?"]
        assert call.since(mark, "ai") == []  # the agent said nothing on return
        assert world.envelope(sid).next == "await_caller"
        await call.say("¿Y mi cuenta de cheques?", "es-US")
        assert len(world.agent.calls) == turns + 2  # the result, then her words
        assert call.texts("welcome")[-1].startswith("Su cuenta de cheques")
        await call.hangup()
        return call

    run(go())
    assert world.state(sid).language == "es-US"
    assert "balance.no_account" in world.prompt_keys(sid)




def _share_zip(world: World) -> None:
    """Make the Doe household share one ZIP, so the ZIP alone can't tell them apart."""
    sam, jordan = world.customer("C-1002"), world.customer("C-1003")
    jordan["factors"]["zip"] = sam["factors"]["zip"]
    world.store.put("customers", jordan)


def test_shared_household_disambiguation_deletes_the_sealed_factor(world: World) -> None:
    sid = call_sid(15)
    _share_zip(world)

    async def go():
        call = world.dial(LEES, sid)
        await call.start()
        await call.press("")
        await call.press(factor("C-1002", "zip"))  # matches both Sam and Jordan
        assert call.state == "gather"
        assert call.texts("say")[-1] == "Please say or enter the last four digits of your Social Security number. If you can't, press pound."
        held = world.session(sid).flow["held"]
        assert set(held) == {"zip"}
        sealed = held["zip"]
        assert sealed in world.sealer.sealed and factor("C-1002", "zip") not in json.dumps(held)
        await call.press(factor("C-1003", "ssn4"))
        assert call.state == "relay"
        assert call.texts("welcome")[-1].startswith("Hi Jordan.")
        # Gone as soon as the verify returned.
        assert "held" not in (world.session(sid).flow or {})
        assert sealed not in world.stored_strings(skip=frozenset())
        await call.hangup()
        return call, sealed

    _, sealed = run(go())
    st = world.state(sid)
    assert st.verified_customer_id == "C-1003" and st.level == 2
    verifies = [e for e in world.lam.payloads if e.get("op") == "verify"]
    assert sorted(verifies[-1]["factors"]) == ["ssn4", "zip"]  # the held factor was re-sent
    assert sealed not in world.stored_strings(skip=frozenset())


def test_hang_up_during_disambiguation_deletes_the_sealed_factor(world: World) -> None:
    sid = call_sid(16)
    _share_zip(world)

    async def go():
        call = world.dial(LEES, sid)
        await call.start()
        await call.press("")
        await call.press(factor("C-1002", "zip"))
        sealed = world.session(sid).flow["held"]["zip"]
        await call.hangup()  # status callback: call end
        return sealed

    sealed = run(go())
    assert world.session(sid).flow is None and world.session(sid).ended
    assert sealed not in world.stored_strings(skip=frozenset())


def test_shared_phone_distinct_zip_verifies_the_right_customer(world: World) -> None:
    """No disambiguating factor needed when the ZIP alone tells them apart."""
    sid = call_sid(17)

    async def go():
        call = world.dial(LEES, sid)
        await call.start()
        await call.press("")
        await call.press(factor("C-1002", "zip"))
        assert call.texts("welcome")[-1].startswith("Hi Sam.")
        await call.hangup()

    run(go())
    assert world.state(sid).verified_customer_id == "C-1002"


# --------------------------------------------------------------------------- failure paths


def test_websocket_drop_reconnects_once_then_escalates(world: World) -> None:
    """Handover "Done when": the WebSocket-drop fallback."""
    sid = call_sid(18)

    async def go():
        call = await login_thomas(world, sid)
        for attempt in (1, 2):
            await call.relay.close()  # the socket drops; Twilio calls the <Connect action>
            call.relay = None
            body = call._post(call.connect_action or "", {  # noqa: SLF001 - drive Twilio's side
                "CallStatus": "in-progress", "SessionStatus": "failed", "ErrorCode": "64107"})
            await call._follow(body)  # noqa: SLF001
            if attempt == 1:
                assert call.state == "relay"
                assert call.texts()[-2:] == ["I'm sorry, something went wrong on our end.",
                                             "Is there anything else I can help you with?"]
        assert call.state == "queued"
        assert call.task["escalation_reason"] == "unable_to_resolve"
        await call.hangup()
        return call

    call = run(go())
    assert call.relay_opens == 2
    assert world.context(call.task["journal_ref"], sid)["statusCode"] == 200


@pytest.mark.parametrize("reason,phrase,priority", [
    ("fraud_suspected", "I'll connect you with our fraud team right away.", "100"),
    ("vulnerable_customer",
     "I'll connect you with a specialist who can help with that. Please hold for a moment.", "50"),
])
def test_escalation_routing_by_reason(world: World, reason: str, phrase: str, priority: str) -> None:
    sid = call_sid(19)
    world.agent.queue(tool("escalate", "tu_esc", reason=reason, summary="Caller needs a specialist."))

    async def go():
        call = await login_thomas(world, sid)
        await call.say("Someone used my card and I'm really upset.")
        await call.hangup()
        return call

    call = run(go())
    assert call.texts()[-1] == phrase
    assert call.enqueue.find("Task").get("priority") == priority
    assert call.task["escalation_reason"] == reason
    assert world.state(sid).escalation["queue"] == ("OwlBank Fraud" if reason == "fraud_suspected"
                                                    else "OwlBank Support EN")


def test_rebuilt_handoff_journal_ref_opens_the_context_api(world: World) -> None:
    """Item 4a: when the ``end`` was lost (barge-in), the bridge rebuilds the handoff with an
    opaque ref (stored in ``escalation.journal_ref``); the task attributes carry the same ref,
    and the Context API serves it."""
    from owlbank import callstate
    from owlbank.callstate import PendingTool

    sid = call_sid(20)

    async def go():
        call = await login_thomas(world, sid)
        # The agent's escalate was stored, but the session never ended (the stream was cut).
        callstate.mutate(world.store, sid, lambda _s: {
            "pending_tool": PendingTool("tu_esc", "escalate", {"reason": "complaint"}),
            "escalation": {"reason": "complaint", "summary": "Caller wants to complain about a fee.",
                           "source": "agent"}})
        await call.say("Hello? Are you still there?")
        assert call.state == "queued"
        await call.hangup()
        return call

    call = run(go())
    ref = call.handoffs[-1]["journal"]
    assert len(ref) >= 20 and sid not in ref and not ref.startswith(("seg:", "call:"))  # opaque
    attrs = call.task
    assert attrs["journal_ref"] == ref == world.state(sid).escalation["journal_ref"]
    assert attrs["escalation_reason"] == "complaint"
    assert attrs["ai_summary"] == "Caller wants to complain about a fee."
    resp = world.context(attrs["journal_ref"], attrs["call_sid"])
    assert resp["statusCode"] == 200
    assert json.loads(resp["body"])["escalation_reason"] == "complaint"


def test_no_candidate_step_up_follows_the_no_match_fallback_escalate(make_world) -> None:
    """With the bank's ``no_match`` fallback set to a human: a caller who reached the
    limited AI without a candidate record (here: the lookup failed) and then needs a
    step-up goes straight to the queue, with no step-up intro."""
    world: World = make_world(config={"fallbacks": {"no_match": "escalate"}})
    world.lam.fail_ops.add("ani_lookup")  # back-end lookup error: limited AI
    sid = call_sid(21)
    world.agent.queue(tool("authenticate", "tu_auth", tool="list_recent_transactions"))

    async def go():
        call = world.dial(THOMAS, sid)
        await call.start()
        await call.press("")
        assert call.state == "relay" and world.state(sid).candidate_ids == ()
        mark = call.mark()
        await call.say("Read me my recent transactions.")
        await call.hangup()
        return call, mark

    call, mark = run(go())
    assert call.state == "ended" and call.task["escalation_reason"] == "unable_to_resolve"
    assert call.since(mark) == [
        "I'll connect you with a specialist who can help with that. Please hold for a moment."]
