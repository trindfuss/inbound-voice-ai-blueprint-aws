"""Model-assisted slots: the scenarios.

Each test drives whole calls through the real front door and the real bridge (TAC, the agent
through the real Strands client) over one store, with the front door and the bridge as separate
processes (``tests.component.slot_world.SplitWorld``): the bridge knows a sensitive value only from
call state's ``sealed_sensitive``. Every log line is captured through ``obs.ScrubFilter``, and the
leak scans cover every surface of the call (``slot_world.surfaces``).

No prompt or TwiML contains a sensitive value in these runs (there is no
read-back), which every scan below checks on the ``twiml (prompts)`` and ``journal`` surfaces.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from owlbank import obs, segments, sensitive
from owlbank.slots import FakeResolver, ResolverOutput
from tests.admin_api_fakes import call as admin_call
from tests.admin_api_fakes import claims
from tests.component.slot_world import (
    SplitWorld,
    answering_turns,
    capture_logs,
    echo_agent,
    gather_attrs,
    leaked,
    policy_doc,
    surfaces,
    verb_tags,
    without_slot_items,
    words_for,
)
from tests.e2e.harness import OTP, THOMAS, Call, call_sid, run

ZIP, SSN4 = "94105", "1111"  # Thomas's fixture factors (data/fixtures.test-factors.json)
INTENT_SWITCH = ResolverOutput(kind="intent_switch")
USE_KEYPAD = "For your security, please use your keypad."
SSN4_PROMPT = "Please say or enter the last four digits of your Social Security number. If you can't, press pound."
SSN4_KEYPAD = "Please enter the last four digits of your Social Security number. If you can't, press pound."


@pytest.fixture
def make() -> Iterator[Callable[..., SplitWorld]]:
    made: list[SplitWorld] = []

    def build(*, switch: bool = False, spoken: bool = True,
              resolver: FakeResolver | None = None, **kw: Any) -> SplitWorld:
        config = {"accept_spoken_factors": spoken, **kw.pop("config", {})}
        if resolver is None and switch:
            resolver = FakeResolver(default=INTENT_SWITCH)
        sw = SplitWorld(resolver=resolver, config=config, **kw)
        sw.respond(echo_agent(escalate_on="a person"))
        made.append(sw)
        return sw

    yield build
    for sw in made:
        sw.close()


def _norm(v: str) -> str:
    return segments._norm_text(v)


async def _login(sw: SplitWorld, sid: str, *, zip_speech: str | None = None) -> Call:
    call = sw.world.dial(THOMAS, sid)
    await call.start()
    await call.press("")  # English
    if zip_speech is None:
        await call.press(ZIP)
    else:
        await call.speak_to_gather(zip_speech)
    return call


async def _to_ssn4(sw: SplitWorld, sid: str, prompt: str = SSN4_PROMPT) -> Call:
    """Logged in, then a level-2 tool: the bridge hands off for step-up and the SSN last four is asked."""
    call = await _login(sw, sid)
    assert call.state == "relay"
    await call.say("I need to change my address")
    assert call.state == "gather" and call.texts("say")[-1] == prompt
    return call


def _assert_clean(sw: SplitWorld, logs: Any, sid: str, values: set[str]) -> None:
    where = leaked(values, surfaces(sw, logs, [sid]))
    assert where == [], f"a sensitive value reached {where}"


# --------------------------------------------------------------------------- SSN last four


def test_ssn4_spoken_cleanly_is_accepted_and_the_answering_turn_reads_the_placeholder(
        make: Callable[..., SplitWorld], monkeypatch: pytest.MonkeyPatch) -> None:
    sw = make()
    sid = call_sid(1)
    turns = answering_turns(monkeypatch)
    with capture_logs() as logs:
        async def go() -> Call:
            call = await _to_ssn4(sw, sid)
            await call.speak_to_gather(words_for(SSN4))
            assert call.state == "relay"  # verified, back to the agent
            await call.say("thanks, that's all")
            # The bridge learned the value from call state alone (its registry is its own).
            registered.update(fd=sw.fd.values(sid), bridge=sw.bridge.values(sid))
            await call.hangup()
            return call

        registered: dict[str, frozenset[str]] = {}
        run(go())
    st = sw.world.state(sid)
    assert st.level == 2 and "ssn4" in st.factors_verified
    ssn = [r for r in turns if r.value == SSN4]
    assert len(ssn) == 1 and ssn[0].match == "exact" and ssn[0].source == "speech"
    assert ssn[0].utterance_for_ai == "[ssn4]"
    assert len(st.sealed_sensitive) == 1  # sealed once; the ZIP (not sensitive) never
    assert registered == {"fd": frozenset({_norm(SSN4)}), "bridge": frozenset({_norm(SSN4)})}
    # Forgotten at the end of the call in both processes.
    assert sw.fd.values(sid) == sw.bridge.values(sid) == frozenset()
    _assert_clean(sw, logs, sid, {SSN4, OTP})
    # the journal never carries the caller's words for a digit slot.
    assert words_for(SSN4) not in json.dumps(sw.world.journal(sid))


def test_ssn4_by_keypad_is_sealed_and_scrubbed_everywhere(
        make: Callable[..., SplitWorld], monkeypatch: pytest.MonkeyPatch) -> None:
    sw = make()
    sid = call_sid(2)
    turns = answering_turns(monkeypatch)
    with capture_logs() as logs:
        async def go() -> None:
            call = await _to_ssn4(sw, sid)
            await call.press(SSN4)
            assert call.state == "relay"
            await call.say("thanks")
            assert sw.bridge.values(sid) == frozenset({_norm(SSN4)})
            await call.hangup()

        run(go())
    (ssn,) = [r for r in turns if r.value == SSN4]
    assert ssn.source == "dtmf" and ssn.utterance_for_ai is None  # no speech, no AI text
    assert sw.world.state(sid).level == 2
    _assert_clean(sw, logs, sid, {SSN4, OTP})


# --------------------------------------------------------------------------- intent switch


def _intent_switch_on_ssn4(sw: SplitWorld, sid: str, utterance: str) -> list[dict[str, Any]]:
    async def go() -> Call:
        call = await _to_ssn4(sw, sid)
        await call.speak_to_gather(utterance)
        assert call.state == "relay", (
            f"no intent switch: the front door answered {call.texts('say')[-2:]} "
            f"(journal: {[e.get('match') for e in sw.world.journal(sid) if e['type'] == 'slot_resolution']})")
        await call.hangup()
        return call

    run(go())
    return [c for c in sw.world.agent.calls if c["user_text"]]


def test_a_request_with_the_value_at_the_ssn4_prompt_is_taken_as_the_answer(
        make: Callable[..., SplitWorld], monkeypatch: pytest.MonkeyPatch) -> None:
    """At the SSN last-4 prompt, "I want my balance
    and my last four are 1 2 3 4" is an answer, not an intent switch. The value is accepted,
    sealed and verified; the AI text for that turn is the placeholder text, and no value leaks."""
    sw = make()
    sid = call_sid(3)
    turns = answering_turns(monkeypatch)
    utterance = f"I want my balance and my last four are {words_for(SSN4)}"
    with capture_logs() as logs:
        async def go() -> None:
            call = await _to_ssn4(sw, sid)
            await call.speak_to_gather(utterance)
            assert call.state == "relay"  # verified, back to the agent
            await call.hangup()

        run(go())
    st = sw.world.state(sid)
    assert st.level == 2 and "ssn4" in st.factors_verified
    (ssn,) = [r for r in turns if r.value == SSN4]
    assert ssn.match == "exact" and ssn.source == "speech"
    assert "[ssn4]" in ssn.utterance_for_ai and SSN4 not in ssn.utterance_for_ai
    assert len(st.sealed_sensitive) == 1
    _assert_clean(sw, logs, sid, {SSN4, OTP})


def test_a_new_request_at_the_ssn4_prompt_is_an_intent_switch_to_the_agent(
        make: Callable[..., SplitWorld]) -> None:
    """No slot value can be extracted from "actually, can I get my recent
    transactions", so it is an intent switch; the agent gets the caller's words unchanged and
    calls list_recent_transactions."""
    sw = make(switch=True)
    sid = call_sid(30)
    utterance = "actually, can I get my recent transactions"
    with capture_logs() as logs:
        texts = _intent_switch_on_ssn4(sw, sid, utterance)
    assert texts[-1]["user_text"] == utterance
    assert [op for _, op in sw.world.lam.calls].count("list_recent_transactions") == 1
    assert sw.world.state(sid).sealed_sensitive == ()
    _assert_clean(sw, logs, sid, {SSN4})


def test_a_request_with_a_value_only_the_resolver_finds_is_still_the_answer(
        make: Callable[..., SplitWorld], monkeypatch: pytest.MonkeyPatch) -> None:
    """"yes my last four are 1111, and
    what's my balance?" is refused by stage 2b (words follow the value), so the resolver decides.
    It gets the words unredacted and, answer first, gives the value even with an
    intent-switch kind: the value grounds, so it is the answer. SSN last-4 is ``confirm: never``,
    so it is accepted and verified; the turn's AI text is the placeholder text and the request
    isn't carried on. Nothing leaks."""
    resolver = FakeResolver([ResolverOutput(kind="intent_switch", value=SSN4)])
    sw = make(resolver=resolver)
    sid = call_sid(4)
    turns = answering_turns(monkeypatch)
    utterance = f"yes my last four are {SSN4}, and what's my balance?"
    with capture_logs() as logs:
        async def go() -> None:
            call = await _to_ssn4(sw, sid)
            await call.speak_to_gather(utterance)
            assert call.state == "relay"  # verified, back to the agent
            await call.hangup()

        run(go())
    assert resolver.calls[-1]["req"].utterance == utterance  # unredacted
    st = sw.world.state(sid)
    assert st.level == 2 and "ssn4" in st.factors_verified
    (ssn,) = [r for r in turns if r.value == SSN4]
    assert (ssn.match, ssn.source, ssn.model) == ("inferred", "model", "ok")
    assert ssn.utterance_for_ai == "yes my last four are [ssn4], and what's my balance?"
    assert [op for _, op in sw.world.lam.calls].count("get_balance") == 0  # not carried on
    assert len(st.sealed_sensitive) == 1
    _assert_clean(sw, logs, sid, {SSN4, OTP})


# --------------------------------------------------------------------------- ZIP, not sensitive


def test_zip_spoken_and_not_sensitive_is_accepted_and_never_sealed(
        make: Callable[..., SplitWorld], monkeypatch: pytest.MonkeyPatch) -> None:
    sw = make()
    sid = call_sid(5)
    turns = answering_turns(monkeypatch)

    async def go() -> None:
        call = await _login(sw, sid, zip_speech="nine four one oh five")
        assert call.state == "relay"
        await call.hangup()

    run(go())
    st = sw.world.state(sid)
    assert st.level == 1 and st.sealed_sensitive == ()
    (zip_turn,) = turns
    assert zip_turn.value == ZIP and zip_turn.utterance_for_ai == "nine four one oh five"
    assert sw.fd.values(sid) == sw.bridge.values(sid) == frozenset()


def test_zip_intent_switch_reaches_the_agent_and_the_journal_unchanged(
        make: Callable[..., SplitWorld]) -> None:
    sw = make(switch=True)
    sid = call_sid(6)
    utterance = f"Actually my zip is {ZIP} but what are your branch hours?"

    async def go() -> None:
        call = await _login(sw, sid, zip_speech=utterance)
        assert call.state == "relay"
        await call.hangup()

    run(go())
    texts = [c["user_text"] for c in sw.world.agent.calls if c["user_text"]]
    assert texts == [utterance]  # the caller's words, unchanged
    assert sw.world.resolver.calls[0]["req"].utterance == utterance  # unredacted
    ai_turns = [e["text"] for e in sw.world.journal(sid) if e["type"] == "ai_turn"]
    assert ai_turns == [f"You said: {utterance}"]  # not registered: the journal keeps the words
    assert sw.world.state(sid).sealed_sensitive == ()


# --------------------------------------------------------------------------- keypad only


def test_keypad_only_slot_takes_the_keypad_only(make: Callable[..., SplitWorld]) -> None:
    sw = make(policy_doc=policy_doc(slots__ssn4__keypad_only=True))
    sid = call_sid(7)

    async def go() -> None:
        call = await _to_ssn4(sw, sid, SSN4_KEYPAD)
        ssn_gather = gather_attrs(sw.world.twiml[-1])[-1]
        assert ssn_gather["input"] == "dtmf"
        assert not {"speechTimeout", "speechModel", "hints", "partialResultCallback"} & set(ssn_gather)
        assert "g=" not in ssn_gather["action"]  # no Gather id: no early finish
        # Twilio sends only digits for a keypad-only Gather; a speech result anyway is refused.
        await call._follow(call._post(ssn_gather["action"], {
            "CallStatus": "in-progress", "SpeechResult": words_for(SSN4), "Confidence": "0.9"}))
        assert call.texts("say")[-2:] == [USE_KEYPAD, SSN4_KEYPAD]
        assert sw.world.state(sid).level == 1
        await call.press(SSN4)
        assert call.state == "relay" and sw.world.state(sid).level == 2
        await call.hangup()

    run(go())
    # The ZIP (keypad_only false, spoken factors on) still takes speech.
    login_gathers = [g for x in sw.world.twiml for g in gather_attrs(x) if g.get("numDigits") == "5"]
    assert login_gathers and all(g["input"] == "dtmf speech" for g in login_gathers)


def test_spoken_factors_off_makes_every_factor_keypad_only_but_not_yes_no(
        make: Callable[..., SplitWorld]) -> None:
    sw = make(spoken=False)
    sid = call_sid(8)

    async def go() -> None:
        call = await _login(sw, sid)
        assert call.state == "relay"
        await call.say("please pay my card")
        await call.press(SSN4)
        await call.press(OTP)
        assert call.state == "gather"  # the read-back
        await call.speak_to_gather("Yes.")
        await call.hangup()

    run(go())
    gathers = [g for x in sw.world.twiml for g in gather_attrs(x) if "/voice/factor" in g["action"]]
    factor = [g for g in gathers if g.get("numDigits") in ("4", "5", "6")]
    assert len(factor) == 3 and all(g["input"] == "dtmf" for g in factor)
    assert all("partialResultCallback" not in g for g in factor)
    confirm = [g for g in gathers if "numDigits" not in g]
    assert confirm and all(g["input"] == "dtmf speech" for g in confirm)  # yes_no isn't a factor
    assert [op for _, op in sw.world.lam.calls].count("make_payment") == 1


# --------------------------------------------------------------------------- barge-in off (path B)


def test_barge_in_off_plays_the_prompt_before_a_gather_with_nothing_nested(
        make: Callable[..., SplitWorld]) -> None:
    on, off = make(), make(policy_doc=policy_doc(slots__ssn4__barge_in=False))
    shapes: dict[str, tuple[list[str], dict[str, str], list[str]]] = {}
    for name, sw in (("on", on), ("off", off)):
        sid = call_sid(9)

        async def go(sw: SplitWorld = sw, sid: str = sid) -> None:
            call = await _to_ssn4(sw, sid)
            await call.press(SSN4)
            assert call.state == "relay" and sw.world.state(sid).level == 2
            await call.hangup()

        run(go())
        body = next(x for x in sw.world.twiml if SSN4_PROMPT in x)
        gather = ET.fromstring(body.encode()).find("Gather")
        assert gather is not None
        shapes[name] = (verb_tags(body), dict(gather.attrib), [c.tag for c in gather])
    tags_on, attrs_on, nested_on = shapes["on"]
    tags_off, attrs_off, nested_off = shapes["off"]
    assert tags_on == ["Gather"] and nested_on == ["Say", "Say"]  # intro and prompt, cut-able
    assert tags_off == ["Say", "Say", "Gather"] and nested_off == []  # path B
    assert attrs_off == attrs_on  # same timeouts, input and action
    assert "bargeIn" not in attrs_off




def test_a_factor_with_no_slot_item_is_sensitive(make: Callable[..., SplitWorld]) -> None:
    """A table seeded before the slot items: ``slot#zip`` absent. The ZIP is a factor, so it loads
    sensitive (fail closed): sealed, registered in both processes, and the placeholder on an
    intent switch."""
    sw = make(switch=True)
    without_slot_items(sw, "zip")
    sw.world.clock.advance(61)  # the components' policy cache
    sid, other = call_sid(10), call_sid(11)
    with capture_logs() as logs:
        async def go() -> None:
            call = await _login(sw, sid)
            assert call.state == "relay"
            await call.say("hello")
            assert sw.bridge.values(sid) == frozenset({_norm(ZIP)})
            await call.say("a person please")  # escalate: the task attributes are built
            await call.hangup()
            switch = await _login(sw, other, zip_speech=f"my zip is {ZIP} but when are you open?")
            assert switch.state == "relay"
            await switch.hangup()

        run(go())
    assert sw.world.frontdoor.d.policy.slot("zip").sensitive is True
    assert len(sw.world.state(sid).sealed_sensitive) == 1
    texts = [c["user_text"] for c in sw.world.agent.calls if c["user_text"]]
    assert texts[-1] == "my zip is [zip] but when are you open?"
    _assert_clean(sw, logs, sid, {ZIP})
    _assert_clean(sw, logs, other, {ZIP})




EMAIL = "ana.doe@mail.example"


def test_a_sensitive_email_is_scrubbed_on_word_boundaries(make: Callable[..., SplitWorld]) -> None:
    """A sensitive text value the front door sealed reaches the bridge through call state; the
    bridge scrubs it from the agent's journal turn and from log lines as a whole word, never from
    inside a longer word, and never from what the agent receives."""
    sw = make()
    sid = call_sid(12)
    longer = EMAIL + "s"  # the value with a word character after it: another word
    with capture_logs() as logs:
        async def go() -> None:
            call = await _login(sw, sid)
            with sw.fd.active():  # a front-door instance accepted a sensitive text answer
                assert sensitive.remember(sw.world.store, sw.world.sealer, sid, EMAIL, slot="email",
                                          slot_type="text", now=sw.world.clock)
            assert sw.bridge.values(sid) == frozenset()
            reply = await call.say(f"my email is {EMAIL}, not {longer}")
            assert EMAIL in reply  # what the caller hears is never changed
            assert sw.bridge.values(sid) == frozenset({_norm(EMAIL)})  # restored on the turn
            tokens = obs.bind(call_sid=sid)
            try:
                logging.getLogger("tac_bridge.bridge").warning(
                    "upstream echoed %s. then %s", EMAIL.upper(), longer)
            finally:
                obs.unbind(tokens)
            await call.hangup()

        run(go())
    user = [c["user_text"] for c in sw.world.agent.calls if c["user_text"]]
    assert user[-1] == f"my email is {EMAIL}, not {longer}"  # the agent's input as said
    (turn,) = [e["text"] for e in sw.world.journal(sid) if e["type"] == "ai_turn"]
    assert turn == f"You said: my email is [redacted], not {longer}"
    (line,) = [x for x in logs.lines if "upstream echoed" in x]
    # The exact-value step leaves the longer word alone (as in the journal turn above); the
    # pattern step then masks it as an email address.
    assert "upstream echoed [redacted]. then [email]" in line and longer not in line


def test_a_sensitive_yes_no_answer_is_sealed_but_never_registered(
        make: Callable[..., SplitWorld]) -> None:
    sw = make(policy_doc=policy_doc(slots__yes_no__sensitive=True))
    sid = call_sid(13)
    with capture_logs() as logs:
        async def go() -> None:
            call = await _login(sw, sid)
            await call.say("please pay my card")
            await call.press(SSN4)
            await call.press(OTP)
            sealed = len(sw.world.state(sid).sealed_sensitive)
            await call.speak_to_gather("yes")
            assert call.state == "relay"
            assert len(sw.world.state(sid).sealed_sensitive) == sealed + 1
            await call.say("yes, that is all")
            for proc in (sw.fd, sw.bridge):
                assert "yes" not in proc.values(sid)
                assert _norm(SSN4) in proc.values(sid)
            await call.hangup()

        run(go())
    assert [op for _, op in sw.world.lam.calls].count("make_payment") == 1
    skipped = [r for r in logs.events("sensitive_scrub_skipped") if r.get("slot") == "yes_no"]
    assert skipped and all(r["reason"] == "slot_type" for r in skipped)
    turns = [e["text"] for e in sw.world.journal(sid) if e["type"] == "ai_turn"]
    assert "You said: yes, that is all" in turns  # "yes" isn't scrubbed from later text


# --------------------------------------------------------------------------- console to call


def test_console_put_changes_the_next_calls_gather_after_the_policy_refresh(
        make: Callable[..., SplitWorld]) -> None:
    sw = make()
    api = sw.admin()
    admin = claims(["admin"])
    status, body = admin_call(api, "GET", "/slots", who=admin)
    assert status == 200
    ssn = next(s for s in body["slots"] if s["name"] == "ssn4")
    assert ssn["keypad_only"] is False and ssn["effective_keypad_only"] is False

    def ssn_gather(sid: str, prompt: str = SSN4_PROMPT) -> dict[str, str]:
        async def go() -> None:
            call = await _to_ssn4(sw, sid, prompt)
            await call.press(SSN4)
            await call.hangup()

        run(go())
        body = next(x for x in reversed(sw.world.twiml) if prompt in x)
        return gather_attrs(body)[-1]

    assert ssn_gather(call_sid(14))["input"] == "dtmf speech"
    status, put = admin_call(api, "PUT", "/slots/ssn4", {"keypad_only": True, "rev": ssn["rev"]},
                             who=admin)
    assert status == 200, put
    item = sw.world.store.get("policy", {"pk": "slot#ssn4"})
    assert item is not None and item["keypad_only"] is True
    # Within the policy cache's TTL the running components still use the old settings.
    assert ssn_gather(call_sid(15))["input"] == "dtmf speech"
    sw.world.clock.advance(61)
    after = ssn_gather(call_sid(16), SSN4_KEYPAD)
    assert after["input"] == "dtmf" and "partialResultCallback" not in after
    audit = sw.world.store.query("admin_audit", {"pk": "policy#slot#ssn4"})
    assert len(audit) == 1 and audit[0]["security_change"] is True
