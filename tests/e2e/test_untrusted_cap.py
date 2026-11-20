"""Offline end to end: the untrusted-attempt cap and no one-time code before login. The security (leak) scan runs at teardown (conftest)."""

from __future__ import annotations


from owlbank.slots import ResolverOutput

from tests.e2e.harness import THOMAS, World, call_sid, run, say, tool
from tests.e2e.test_core_journeys import login_thomas

LIMIT = 3  # data/policy.json ani_trust.untrusted_failure_limit
RISK_PHRASE = "Let me connect you with a specialist who can help verify your account. Please hold for a moment."


def risk_keys(world: World) -> list[str]:
    return sorted(str(k[0]) for k in world.store._tables["risk"])  # noqa: SLF001


def test_repeated_spoofed_calls_end_in_verification_risk_and_the_real_customer_still_logs_in(
        world: World) -> None:
    async def spoofed_failure(n: int) -> None:
        """A spoofed caller ID (no StirVerstat) with wrong answers, hanging up after one try."""
        call = world.dial(THOMAS, call_sid(80 + n))
        await call.start(stir_verstat=None)
        await call.press("")
        assert call.texts("say")[-1] == "Please say or enter your five-digit ZIP code. If you can't, press pound."
        await call.press("00000")
        await call.press("0000")
        assert call.state == "gather"  # one failure: asked again
        assert call.texts("say")[-2:][0].startswith("Sorry")
        await call.hangup()

    async def go():
        for n in range(LIMIT):
            await spoofed_failure(n)
        assert len(risk_keys(world)) == LIMIT

        # Call N+1: transferred before any factor is asked.
        call = world.dial(THOMAS, call_sid(80 + LIMIT))
        await call.start(stir_verstat=None)
        await call.press("")
        assert call.state == "queued"
        assert call.texts("say")[-1] == RISK_PHRASE
        assert not any("ZIP" in t for t in call.texts("say"))
        task = call.task
        assert task is not None
        assert task["escalation_reason"] == "verification_risk"
        assert task["auth_level"] == 0 and task["customer_id"] is None
        assert task["factors_verified"] == []
        assert call.enqueue is not None and call.enqueue.get("workflowSid")
        await call.hangup()

        # The real customer, trusted ANI, afterwards: logs in normally.
        world.agent.queue(say("Hello Thomas."))
        real = await login_thomas(world, call_sid(89))
        assert real.state == "relay"
        await real.say("Hi.")
        await real.hangup()

    run(go())
    risky = call_sid(80 + LIMIT)
    st = world.state(risky)
    assert st.escalation and st.escalation["reason"] == "verification_risk"
    assert st.escalation["queue"] == "OwlBank Fraud"
    assert st.level == 0
    verifies = [p for p in world.lam.payloads if p.get("op") == "verify" and p["call_sid"] == risky]
    assert verifies == []  # no factor was collected or checked
    prechecks = [p for p in world.lam.payloads if p.get("op") == "precheck"]
    assert {p["call_sid"] for p in prechecks} == {call_sid(80 + n) for n in range(LIMIT + 1)}
    assert all("factors" not in p for p in prechecks)
    events = [e for e in world.journal(risky) if e["type"] == "verification"]
    assert events[-1]["outcome"] == "verification_risk"
    assert world.state(call_sid(89)).level == 1
    # The trusted call never touched the counter.
    assert len(risk_keys(world)) == LIMIT


def test_untrusted_step_up_that_needs_an_otp_before_login_goes_to_a_human(make_world) -> None:
    """Untrusted caller ID, login left for another question (limited AI, no token), then a
    payment: level 3 would need the one-time code before any token exists, so no code is sent.
    (A cancel at the ZIP prompt would be a decline and close the login.)"""
    # The caller speaks at the ZIP prompt: that needs spoken factors on.
    world: World = make_world(config={"accept_spoken_factors": True})
    sid = call_sid(90)

    # Refused for the level only, so the bridge hands off for step-up at once.
    world.agent.queue(
        tool("make_payment", "tu_pay", from_account="checking", to="red", amount="50.00"))

    async def go():
        call = world.dial(THOMAS, sid)
        await call.start(stir_verstat=None)
        await call.press("")
        assert call.texts("say")[-1] == "Please say or enter your five-digit ZIP code. If you can't, press pound."
        world.resolver.default = ResolverOutput(kind="intent_switch")
        await call.speak_to_gather("Pay fifty dollars toward my credit card from checking.")
        world.resolver.default = ResolverOutput(kind="none")
        # The interrupted login hands the caller's words to the AI (limited, no token), which
        # asks for the payment at once.
        assert call.state == "queued"
        assert call.texts("say")[-1] == RISK_PHRASE
        assert call.task is not None and call.task["escalation_reason"] == "verification_risk"
        assert call.task["auth_level"] == 0
        await call.hangup()

    run(go())
    ops = [p.get("op") for p in world.lam.payloads]
    assert "send_otp" not in ops and "verify" not in ops
    events = [e for e in world.journal(sid) if e["type"] == "verification"]
    assert events[-1]["outcome"] == "verification_risk"
    assert risk_keys(world) == []  # nothing was checked, so nothing was counted
