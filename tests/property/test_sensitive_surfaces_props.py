"""Model-assisted slots: the cross-component properties at the front-door level and
end to end.

**No leak.** For any sensitive values (the SSN last four, the code, and the ZIP when it is flagged)
and any event sequence through the front door (login, step-up, confirm, with ``FakeResolver``, the
fake sealer and the real back end) and the bridge (a fake agent that echoes the caller's text, also
into its hand-off summary), no log line (every logger through ``ScrubFilter``), journal event,
prompt, TwiML, call-state text, task attribute, Context API timeline entry
(``context_api.build_timeline`` over the journal) or agent input contains any of them (case, spaces
and hyphens tolerated, ``tests.leakscan``), and neither does any answering turn's AI-bound text
(``Resolution.utterance_for_ai``). Metric dimensions are logged fields, so the log scan covers
them. It holds with today's redaction setting, with redaction off (``enabled: false``) and with the
fake PII check failing. The front door and the bridge are separate processes
(``tests.component.slot_world.SplitWorld``).

**Front-door level.** For a sensitive slot, ``callerUtterance``, ``interrupted_utterance``
and the hand-off summary an agent builds from that turn are one string with no
exact-value match; outside the ``[<slot name>]`` tokens every number run is one token and one
digit; the accepted value never survives). For a non-sensitive slot they equal the caller's words.

**Short values, end to end.** A sensitive value under 4 characters (a 3-digit ZIP, flagged) is never
registered in either process, is sealed, and is in no log line or journal field: it occurs there
no more often than in a control run of the same call in which it was never said (server text of
its own, such as times, may hold the same digits).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
import re
from dataclasses import replace
from typing import Any
from unittest import mock

from hypothesis import HealthCheck, assume, event, example, given, settings
from hypothesis import strategies as st

import lambdas.frontdoor.slotflow as fd_slotflow
from lambdas.context_api.handler import build_timeline
from lambdas.frontdoor.app import FrontDoor
from owlbank import segments
from owlbank.pii import PiiGuard
from owlbank.policy import Policy
from owlbank.segments import contains_value
from owlbank.slots import FakeResolver, ResolverOutput, number_runs
from tests import strands_parity
from tests.component.slot_world import (
    SplitWorld,
    capture_logs,
    collides,
    echo_agent,
    leak_report,
    leaked,
    policy_doc,
    surfaces,
    words_for,
)
from tests.component.test_frontdoor_corrections import ssn_first, untrusted_start
from tests.component.test_frontdoor_support import OTP, TEST_FACTORS, THOMAS, build, sid
from tests.e2e.harness import Call, call_sid, run, tool
from tests.leakscan import value_pattern
from tests.property.test_ai_text_props import resolver_outputs, utterance

INTENT_SWITCH = ResolverOutput(kind="intent_switch")
S_VALUE = TEST_FACTORS["C-1001"]["ssn4"]
WHOLE_CALLS = settings(max_examples=30, deadline=None,
                       suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
FRONT_DOOR = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@contextmanager
def _plain_agent() -> Iterator[None]:
    """The plain scripted agent (the agent's internals aren't under test here)."""
    strands_parity.reset(plain=True)
    try:
        yield
    finally:
        strands_parity.reset()


@contextmanager
def _answering_turns() -> Iterator[list[Any]]:
    results: list[Any] = []
    real = fd_slotflow.resolve

    def spy(*a: Any, **k: Any) -> Any:
        out = real(*a, **k)
        results.append((a[0], out))
        return out

    with mock.patch.object(fd_slotflow, "resolve", spy):
        yield results


def _digits(n: int) -> st.SearchStrategy[str]:
    return st.text("0123456789", min_size=n, max_size=n).filter(lambda v: not collides(v))




@st.composite
def call_plans(draw: st.DrawFn) -> dict[str, Any]:
    ssn4 = draw(_digits(4))
    zip_value = draw(_digits(5).filter(lambda z: ssn4 not in z))
    wrong = draw(st.none() | _digits(4).filter(lambda w: w != ssn4 and w not in zip_value))
    return {
        "ssn4": ssn4, "zip": zip_value, "wrong": wrong,
        "zip_sensitive": draw(st.booleans()),
        "redaction": draw(st.sampled_from(["today", "disabled", "pii_down"])),
        "zip_form": draw(st.sampled_from(["keypad", "words"])),
        "ssn_form": draw(st.sampled_from(["keypad", "digits", "words", "spaced"])),
        "switch_first": draw(st.booleans()),
        "path": draw(st.sampled_from(["address", "pay"])),
        "confirm": draw(st.sampled_from(["Yes.", "No.", "what's my balance?"])),
        "after": draw(st.lists(st.sampled_from(["echo", "escalate"]), max_size=2)),
        "blurt": draw(st.booleans()),
    }


def _ssn_speech(form: str, value: str) -> str | None:
    return {"keypad": None, "digits": value, "words": words_for(value),
            "spaced": " ".join(value)}[form]


async def _whole_call(sw: SplitWorld, sid_: str, plan: dict[str, Any], registered: list[str]) -> None:
    call: Call = sw.world.dial(THOMAS, sid_)
    await call.start()
    await call.press("")
    if plan["zip_form"] == "keypad":
        await call.press(plan["zip"])
    else:
        await call.speak_to_gather(words_for(plan["zip"]))
    assert call.state == "relay"
    await call.say("hello there")
    trigger = "I need to change my address" if plan["path"] == "address" else "please pay my card"
    if plan["switch_first"]:  # an intent switch on the SSN answer, with the value in it
        await call.say(trigger)
        assert call.state == "gather"
        await call.speak_to_gather(f"yes my last four are {plan['ssn4']}, and what's my balance?")
        assert call.state == "relay"
        await call.say("thanks")
    await call.say(trigger)
    assert call.state == "gather"
    if plan["wrong"]:
        await call.press(plan["wrong"])
        assert call.state == "gather"
    speech = _ssn_speech(plan["ssn_form"], plan["ssn4"])
    if speech is None:
        await call.press(plan["ssn4"])
    else:
        await call.speak_to_gather(speech)
    plan["_given"]["said"] = True
    if plan["path"] == "pay" and call.state == "gather":
        await call.press(OTP)
        if call.state == "gather":  # the read-back
            await call.speak_to_gather(plan["confirm"])
    registered.extend(sorted(sw.fd.values(sid_) | sw.bridge.values(sid_)))
    for step in plan["after"]:
        if call.state != "relay":
            break
        await call.say("can you repeat that" if step == "echo" else "a person please")
    await call.hangup()


@WHOLE_CALLS
@given(plan=call_plans())
def test_no_sensitive_value_on_any_surface_of_a_whole_call(plan: dict[str, Any]) -> None:
    changes: dict[str, Any] = {"slots__zip__sensitive": plan["zip_sensitive"]}
    if plan["redaction"] == "disabled":
        changes["redaction__enabled"] = False
    with _plain_agent():
        sw = SplitWorld(resolver=FakeResolver(default=INTENT_SWITCH),
                        config={"accept_spoken_factors": True}, policy_doc=policy_doc(**changes))
        try:
            sw.world.pii.fail = plan["redaction"] == "pii_down"
            # A misbehaving agent repeats the SSN last four once the caller has given it (as if a
            # tool had shown it): what it says is heard, but no journal, log or task surface keeps it.
            given_ssn = {"said": False}
            sw.respond(echo_agent(escalate_on="a person", blurt=lambda: plan["ssn4"] if (
                plan["blurt"] and given_ssn["said"]) else None))
            plan["_given"] = given_ssn
            sw.set_factor("C-1001", "ssn4", plan["ssn4"])
            sw.set_factor("C-1001", "zip", plan["zip"])
            sid_ = call_sid(1)
            registered: list[str] = []
            with capture_logs() as logs, _answering_turns() as turns:
                run(_whole_call(sw, sid_, plan, registered))
            values = {plan["ssn4"]} | ({plan["wrong"]} if plan["wrong"] else set())
            if plan["path"] == "pay":
                values.add(OTP)
            if plan["zip_sensitive"]:
                values.add(plan["zip"])
            # Every accepted sensitive answer was registered (in one process or both).
            assert set(registered) <= {segments._norm_text(v) for v in values}
            assert segments._norm_text(plan["ssn4"]) in registered
            found = surfaces(sw, logs, [sid_])
            if plan["blurt"]:
                found.pop("relay frames")  # what the agent says is spoken as it said it
            assert leaked(values, found) == [], (plan, leak_report(values, found))
            for slot, result in turns:
                if slot.sensitive and result.utterance_for_ai is not None:
                    assert leaked(values, {"ai text": [result.utterance_for_ai]}) == [], plan
        finally:
            sw.close()


# --------------------------------------------------------------------------- (front door)


def _outside(text: str, name: str) -> str:
    return text.replace(f"[{name}]", " | ")


def _cpm13(text: str, name: str, value: str | None, generated: str | None) -> None:
    rest = _outside(text, name)
    for r in number_runs(rest, ("en", "es")):
        assert r.tokens == 1 and r.digits is not None and len(r.digits) == 1, text
    for v in {value, generated} - {None}:
        assert not contains_value(rest, v), text  # type: ignore[arg-type]


@FRONT_DOOR
@given(data=st.data(), out=st.one_of(st.just(INTENT_SWITCH), resolver_outputs),
       spanish=st.booleans())
def test_answering_turn_of_a_sensitive_slot_on_every_front_door_surface(
        data: st.DataObject, out: ResolverOutput, spanish: bool) -> None:
    h = build(resolver=FakeResolver([out]), config_kw={"accept_spoken_factors": True})
    slot = h.app.d.policy.slot("ssn4")
    speech, generated, _ = data.draw(utterance(slot, False))
    call = sid(1)
    try:
        h.start(call, THOMAS, lang_digit="2" if spanish else "")
        h.factor(call, "94105")
        h.set_pending_tool(call, "authenticate", {"tool": "change_address"})
        h.handoff(call, {"reason": "step-up", "tool": "change_address"})
        with _answering_turns() as turns:
            root = h.xml("factor", call, SpeechResult=speech, Confidence="0.9")
        (_, result), = turns
        if result.utterance_for_ai is not None:
            _cpm13(result.utterance_for_ai, "ssn4", result.value, generated)
        st_ = h.state(call)
        if root.find("./Connect/ConversationRelay") is None or st_.interrupted_utterance is None:
            return
        env = h.pending_result(call)
        text = env.callerUtterance
        assert text is not None and text == st_.interrupted_utterance == result.utterance_for_ai
        _cpm13(text, "ssn4", None, generated)
        # An agent that hands the turn back as its summary: the screen pop gets the same string.
        other = replace(h, app=FrontDoor(h.app.d, h.app.cfg), bodies=[])
        other.set_pending_tool(call, "escalate", {"reason": "complaint", "summary": text})
        done = other.handoff(call, {"reason": "escalate", "escalationReason": "complaint"})
        task = done.find("Enqueue/Task")
        assert task is not None
        assert json.loads(task.text or "{}")["ai_summary"] == text
    finally:
        segments.clear_sensitive(call)


@FRONT_DOOR
@given(data=st.data(), out=st.one_of(st.just(INTENT_SWITCH), resolver_outputs))
def test_answering_turn_of_a_non_sensitive_slot_is_the_callers_words(
        data: st.DataObject, out: ResolverOutput) -> None:
    h = build(resolver=FakeResolver([out]), config_kw={"accept_spoken_factors": True})
    slot = h.app.d.policy.slot("zip")
    assert not slot.sensitive  # the seed
    speech, _, _ = data.draw(utterance(slot, False))
    call = sid(2)
    try:
        h.start(call, THOMAS)
        with _answering_turns() as turns:
            h.xml("factor", call, SpeechResult=speech, Confidence="0.9")
        (_, result), = turns
        assert result.utterance_for_ai == speech
        st_ = h.state(call)
        if st_.interrupted_utterance is not None:
            assert h.pending_result(call).callerUtterance == st_.interrupted_utterance == speech
        assert st_.sealed_sensitive == ()
    finally:
        segments.clear_sensitive(call)




SHORT_ZIP = policy_doc(slots__zip__sensitive=True, factors__zip__constraints={"length": 3})


_SIZE_FIELD = re.compile(r'"event_bytes": \d+')


def _short_call(value: str, form: str, scan: str) -> tuple[dict[str, int], dict[str, Any]]:
    """One call whose (flagged, 3-digit) ZIP is ``value``, given by keypad or spoken. Returns the
    occurrences of ``scan`` (as a whole value) per log and journal surface, and what the processes
    registered and sealed."""
    sw = SplitWorld(config={"accept_spoken_factors": True}, policy_doc=SHORT_ZIP)
    try:
        sw.respond(echo_agent(escalate_on="a person"))
        sw.set_factor("C-1001", "zip", value)
        sid_ = call_sid(2)
        seen: dict[str, Any] = {}

        async def go() -> None:
            call = sw.world.dial(THOMAS, sid_)
            await call.start()
            await call.press("")
            if form == "keypad":
                await call.press(value)
            else:
                await call.speak_to_gather(words_for(value))
            if call.state == "relay":
                await call.say("hello")
            seen["fd"], seen["bridge"] = sw.fd.values(sid_), sw.bridge.values(sid_)
            seen["sealed"] = len(sw.world.state(sid_).sealed_sensitive)
            if call.state == "relay":
                await call.say("a person please")
            await call.hangup()

        with capture_logs() as logs:
            run(go())
        found = surfaces(sw, logs, [sid_])
        seen["skipped"] = [r for r in logs.events("sensitive_scrub_skipped") if r.get("slot") == "zip"]
        pattern = value_pattern(scan)
        # A size field is a count, not caller data: the spoken words change the request size, so
        # ``event_bytes`` can equal a 3-digit value by chance ("795", found by Hypothesis).
        counts = {where: sum(len(pattern.findall(_SIZE_FIELD.sub("", t.casefold())))
                             for t in found[where])
                  for where in ("log", "journal", "timeline", "task attributes")}
        return counts, seen
    finally:
        sw.close()


@WHOLE_CALLS
@given(value=_digits(3), other=_digits(3), form=st.sampled_from(["keypad", "words"]))
def test_cpm13c_a_short_sensitive_value_is_in_no_log_or_journal_field(
        value: str, other: str, form: str) -> None:
    assume(value != other)
    with _plain_agent():
        counts, seen = _short_call(value, form, scan=value)
        control, _ = _short_call(other, form, scan=value)  # the same call; ``value`` never said
    assert seen["fd"] == seen["bridge"] == frozenset()  # never registered (MIN_SENSITIVE_LEN)
    assert seen["sealed"] == 1  # still sealed
    assert seen["skipped"] and {r["reason"] for r in seen["skipped"]} == {"too_short"}
    for where, n in counts.items():
        assert n <= control[where], (where, value)


# =========================================================================== phase 2
#
# The no-leak property again, over arbitrary and adversarial resolver output in every mode, with corrections,
# the card questions and generated text, and with redaction ``enabled: true`` and ``false`` and the
# fake PII check up and failing. The AI-text and short-value properties again over resolver output that
# carries the sensitive value, and over a correction turn of a sensitive slot.

MODES = ("deterministic", "assisted", "generative")


def adversarial(values: list[str]) -> st.SearchStrategy[ResolverOutput]:
    """Resolver outputs that carry the call's sensitive values in every field the model fills: a
    value, a correction of a sensitive slot, candidates, and generated text with the digits or
    their words."""
    v = st.sampled_from(values)
    d = st.sampled_from([x for x in values if x.isdigit()] or ["0"])
    return st.one_of(
        st.builds(ResolverOutput, kind=st.sampled_from(["exact", "inferred", "intent_switch",
                                                        "escalate"]), value=v),
        st.builds(ResolverOutput, kind=st.just("correction"),
                  slot=st.sampled_from(["ssn4", "otp", "zip", "card"]), value=v),
        st.builds(ResolverOutput, kind=st.just("ambiguous"),
                  candidates=st.lists(v, min_size=2, max_size=3).map(tuple)),
        st.builds(ResolverOutput, kind=st.sampled_from(["inferred", "ambiguous", "exact"]),
                  value=st.sampled_from(["red", "purple", "rewards"]),
                  candidates=st.just(("red", "purple")),
                  say=st.one_of(v.map(lambda x: f"The {{option:red}}, ending {x}?"),
                                d.map(lambda x: f"Your {{option:purple}} {words_for(x)}?"),
                                st.just("The {option:red} or the {option:purple}?")),
                  say_lang=st.sampled_from(["en-US", "es-US"])),
        st.builds(ResolverOutput, kind=st.just("none"), reply=st.sampled_from(["yes", "no"]),
                  value=v),
    )


@st.composite
def phase2_plans(draw: st.DrawFn) -> dict[str, Any]:
    ssn4 = draw(_digits(4))
    zip_value = draw(_digits(5).filter(lambda z: ssn4 not in z))
    wrong = f"{(int(ssn4) + draw(st.integers(1, 9998))) % 10000:04d}"
    assume(not collides(wrong) and wrong not in zip_value)
    values = [ssn4, wrong, OTP, zip_value, words_for(ssn4)]
    plausible = st.sampled_from([
        ResolverOutput(kind="inferred", value=ssn4), ResolverOutput(kind="inferred", value=zip_value),
        ResolverOutput(kind="correction", slot="zip", value=zip_value),
        ResolverOutput(kind="correction", slot="ssn4", value=ssn4),
        ResolverOutput(kind="correction", slot="zip", value=None),
        ResolverOutput(kind="correction", slot="card", value="red"),
        ResolverOutput(kind="inferred", value="purple",
                       say="You'd like the {option:purple}, right?", say_lang="en-US"),
        ResolverOutput(kind="ambiguous", candidates=("red", "purple"),
                       say="The {option:red} or the {option:purple}?", say_lang="en-US"),
        ResolverOutput(kind="ambiguous", candidates=("red", "purple"),
                       say=f"The {{option:red}} or {ssn4}?", say_lang="en-US"),
        ResolverOutput(kind="intent_switch", value=ssn4), INTENT_SWITCH])
    outs = draw(st.lists(st.one_of(plausible, plausible, adversarial(values), resolver_outputs),
                         max_size=12))
    return {
        "ssn4": ssn4, "zip": zip_value, "wrong": wrong, "outs": outs,
        "default": draw(st.one_of(st.just(INTENT_SWITCH), adversarial(values), resolver_outputs)),
        "mode": draw(st.sampled_from(("generative", "assisted", "deterministic"))),
        "zip_sensitive": draw(st.booleans()),
        "enabled": draw(st.booleans()),
        "pii_fail": draw(st.booleans()),
        "to": draw(st.sampled_from(["red", "purple", "rewards", None])),
        "trigger": draw(st.sampled_from(["please pay my card", "I need to change my address"])),
        # 0 is the expected answer at each prompt (the call moves on); anything else is drawn
        # from that prompt's other answers.
        "moves": draw(st.lists(st.one_of(st.just(0), st.integers(1, 40)), min_size=6,
                               max_size=16)),
        "blurt": draw(st.booleans()),
        "trusted": draw(st.booleans()),
    }


def _pay_agent(plan: dict[str, Any], given: dict[str, bool]) -> Any:
    """``echo_agent``, but ``make_payment`` names ``plan["to"]`` (or no card), so the card
    question and its follow-ups run."""
    echo = echo_agent(escalate_on="a person", blurt=lambda: plan["ssn4"] if (
        plan["blurt"] and given["said"]) else None)

    def respond(rec: dict[str, Any]) -> list[Any]:
        text = rec.get("user_text") or ""
        if "pay" in text:
            args = {"from_account": "checking", "amount": "25.00"}
            if plan["to"] is not None:
                args["to"] = plan["to"]
            return tool("make_payment", "tu_pay", **args)
        return echo(rec)

    return respond


def _answers(plan: dict[str, Any], factor: str | None) -> list[tuple[str, str]]:
    """What a caller may answer at the Gather asking ``factor``: ``("press", keys)`` or
    ``("speak", words)``. A sensitive value is only ever given at its own (sensitive) prompt: the
    words of a non-sensitive answer reach the AI unchanged by design."""
    ssn, z = plan["ssn4"], plan["zip"]
    common = [("speak", "what's my balance?"), ("speak", "hmm"), ("speak", "say that again"),
              ("press", "1"), ("press", "2"), ("speak", "yes"), ("speak", "no")]
    if factor == "zip":
        return [("press", z), ("speak", words_for(z)), ("speak", f"um, {z}, I think, yes"),
                ("speak", f"23456, oh, no actually it is {z}"), *common]
    if factor == "ssn4":
        return [("press", ssn), ("press", plan["wrong"]), ("speak", words_for(ssn)),
                ("speak", f"eh, {words_for(ssn[:1])}, hold on... {words_for(ssn[1:])}"),
                ("speak", f"yes my last four are {ssn}, and what's my balance?"),
                ("speak", f"actually the ZIP should be {z}"),
                ("speak", f"the ZIP code is {z} and the last four are {ssn}"), *common]
    if factor == "otp":
        return [("press", OTP), ("speak", " ".join(OTP)),
                ("speak", f"sorry, my last four are actually {words_for(ssn)}"),
                ("speak", f"actually the ZIP should be {z}"), *common]
    return [("speak", "the purple card"), ("speak", "the one that's purple"),
            ("speak", "the OwlBank card"), ("speak", "no, I meant the red one"),
            ("speak", "no, the red one"), ("speak", "the violet one"), ("press", "1"),
            ("press", "2"), *common]


async def _phase2_call(sw: SplitWorld, sid_: str, plan: dict[str, Any]) -> None:
    call: Call = sw.world.dial(THOMAS, sid_)
    # An untrusted caller ID holds the ZIP, sealed, while the SSN last four is asked: the ZIP is
    # correctable there.
    await call.start() if plan["trusted"] else await call.start(stir_verstat=None)
    await call.press("")
    if plan.get("script"):
        await _scripted(sw, call, sid_, plan)
        await call.hangup()
        return
    await call.press(plan["zip"])
    for move in plan["moves"]:
        if call.state == "relay":
            if move % 4 == 0 and call.relay_opens > 1:
                await call.say("a person please")
            else:
                await call.say(plan["trigger"] if move % 2 else "hello there")
            continue
        if call.state != "gather":
            break
        sess = sw.world.session(sid_)
        factor = (sess.flow or {}).get("factor") if sess is not None else None
        options = _answers(plan, factor if isinstance(factor, str) else None)
        how, what = options[move % len(options)]
        if how == "speak" and "speech" not in (call.gather.get("input") or "") \
                if call.gather is not None else False:
            how, what = "press", plan["ssn4"] if factor == "ssn4" else "1"
        if how == "press":
            await call.press(what)
        else:
            await call.speak_to_gather(what)
        # The misbehaving agent repeats the SSN last four once the system holds it (accepted and
        # registered); a value it never accepted is one it can't know to scrub, or to leak.
        registered = sw.fd.values(sid_) | sw.bridge.values(sid_)
        plan["_given"]["said"] = segments._norm_text(plan["ssn4"]) in registered
    await call.hangup()


async def _scripted(sw: SplitWorld, call: Call, sid_: str, plan: dict[str, Any]) -> None:
    """``plan["script"]``: ``(at, how, text)`` steps in order, ``{ssn4}``, ``{zip}``, ``{wrong}``,
    ``{otp}`` and ``{ssn4_words}`` filled in. ``at`` is the factor the Gather must be asking
    (``relay`` for the AI); a step for anything else is skipped."""
    fill = {"ssn4": plan["ssn4"], "zip": plan["zip"], "wrong": plan["wrong"], "otp": OTP,
            "ssn4_words": words_for(plan["ssn4"]),
            "first_zip": plan["zip"] if plan["trusted"] else "60607"}
    for at, how, text in plan["script"]:
        if call.state not in ("relay", "gather"):
            break
        sess = sw.world.session(sid_)
        asking = (sess.flow or {}).get("factor") if sess is not None else None
        here = "relay" if call.state == "relay" else asking
        if at != here:
            continue
        what = text.format(**fill)
        if how == "say":
            await call.say(what)
        elif how == "press":
            await call.press(what)
        else:
            await call.speak_to_gather(what)
        registered = sw.fd.values(sid_) | sw.bridge.values(sid_)
        plan["_given"]["said"] = segments._norm_text(plan["ssn4"]) in registered


def _tour(mode: str, *, enabled: bool = True, pii_fail: bool = False, zip_sensitive: bool = False,
          trusted: bool = True, to: str | None = None, blurt: bool = False,
          outs: list[ResolverOutput], script: list[tuple[str, str, str]]) -> dict[str, Any]:
    """An explicit plan that reaches what random plans rarely do: corrections, the card question
    and its follow-ups, the read-back, and generated text (accepted and refused)."""
    return {"ssn4": "7391", "zip": "60606", "wrong": "7392", "outs": outs,
            "default": INTENT_SWITCH, "mode": mode, "zip_sensitive": zip_sensitive,
            "enabled": enabled, "pii_fail": pii_fail, "to": to, "trigger": "please pay my card",
            "moves": [], "blurt": blurt, "trusted": trusted, "script": script}


class WordsResolver(FakeResolver):
    """Answers by the caller's words: the output of the first ``(words, output)`` pair whose words
    are in the utterance, else ``default`` (a tour's turns may be skipped, so order can't key)."""

    def __init__(self, by_words: list[tuple[str, ResolverOutput]], default: ResolverOutput) -> None:
        super().__init__(default=default)
        self.by_words = by_words

    def resolve(self, req: Any, *, budget_ms: int, generative: bool,
                flow: str | None = None) -> ResolverOutput:
        super().resolve(req, budget_ms=budget_ms, generative=generative, flow=flow)
        return next((o for w, o in self.by_words if w in req.utterance), self.default)


#: A tour: an untrusted login (the ZIP held while the SSN is asked) with a wrong ZIP corrected at
#: the SSN prompt; or a trusted one, whose step-up takes a messy SSN (inferred) and a correction
#: of the held SSN at the code prompt. Then the payment with no card: the card question, the
#: OwlBank card (ambiguous; in ``generative`` wording with the digits refused, or plain wording
#: spoken), Red, a card correction at the read-back, yes, and a hand-off to a person.
def _tour_words(say: str) -> list[tuple[str, ResolverOutput]]:
    return [
        ("ZIP should be", ResolverOutput(kind="correction", slot="zip", value="60606")),
        ("the first is", ResolverOutput(kind="inferred", value="7391")),
        ("actually seven", ResolverOutput(kind="correction", slot="ssn4", value="7391")),
        ("OwlBank card", ResolverOutput(kind="ambiguous", candidates=("red", "purple"), say=say,
                                        say_lang="en-US")),
        ("no, the red", ResolverOutput(kind="correction", slot="card", value="red")),
    ]


_SAY_DIGITS = "The {option:red}, ending 7391, or the {option:purple}?"
_SAY_OK = "The {option:red} or the {option:purple}?"
_TOUR_SCRIPT = [
    ("zip", "press", "{first_zip}"),
    ("ssn4", "speak", "actually the ZIP should be {zip}"),
    ("ssn4", "press", "{ssn4}"),
    ("otp", "press", "{otp}"),
    ("relay", "say", "please pay my card"),
    ("ssn4", "speak", "eh, the first is, hold on, {ssn4_words}, I believe, yes"),
    ("ssn4", "press", "{ssn4}"),
    ("otp", "speak", "sorry, my last four are actually {ssn4_words}"),
    ("otp", "press", "{otp}"),
    ("card", "speak", "the OwlBank card"),
    ("card", "speak", "the red one"),
    ("yes_no", "speak", "no, the red one"),
    ("yes_no", "press", "1"),
    ("relay", "say", "a person please"),
]


def _tour(mode: str, *, enabled: bool, pii_fail: bool, zip_sensitive: bool, trusted: bool,
          say: str, blurt: bool) -> dict[str, Any]:
    return {"ssn4": "7391", "zip": "60606", "wrong": "7392", "outs": [],
            "by_words": _tour_words(say), "default": INTENT_SWITCH, "mode": mode,
            "zip_sensitive": zip_sensitive, "enabled": enabled, "pii_fail": pii_fail, "to": None,
            "trigger": "please pay my card", "moves": [], "blurt": blurt, "trusted": trusted,
            "script": _TOUR_SCRIPT}


_TOURS = [
    _tour(mode, enabled=enabled, pii_fail=pii_fail, zip_sensitive=zs, trusted=trusted, say=say,
          blurt=blurt)
    for mode in MODES
    for enabled, pii_fail, zs, say, blurt, trusted in (
        (True, False, False, _SAY_DIGITS, False, False), (False, False, True, _SAY_OK, True, False),
        (True, True, True, _SAY_DIGITS, False, True), (False, True, False, _SAY_OK, False, True))
]


def _with_examples(fn: Any) -> Any:
    for plan in _TOURS:
        fn = example(plan=plan)(fn)
    return fn


@settings(max_examples=80, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(plan=phase2_plans())
@_with_examples
def test_phase2_no_sensitive_value_on_any_surface_whatever_the_resolver_returns(
        plan: dict[str, Any]) -> None:
    changes: dict[str, Any] = {"slots__zip__sensitive": plan["zip_sensitive"],
                               "redaction__enabled": plan["enabled"]}
    config: dict[str, Any] = {"accept_spoken_factors": True}
    if plan["mode"] == "deterministic":
        config["slot_mode_ceiling"] = "deterministic"
    elif plan["mode"] == "generative":
        config["slot_mode_ceiling"] = "generative"
        changes.update(slot_resolution__generative_enabled=True,
                       slot_resolution__default_mode="generative")
    resolver = (WordsResolver(plan["by_words"], plan["default"]) if plan.get("by_words")
                else FakeResolver(list(plan["outs"]), default=plan["default"]))
    with _plain_agent():
        sw = SplitWorld(resolver=resolver, config=config, policy_doc=policy_doc(**changes))
        try:
            sw.world.pii.fail = plan["pii_fail"]
            given_ssn = {"said": False}
            plan["_given"] = given_ssn
            sw.respond(_pay_agent(plan, given_ssn))
            sw.set_factor("C-1001", "ssn4", plan["ssn4"])
            sw.set_factor("C-1001", "zip", plan["zip"])
            sid_ = call_sid(3)
            with capture_logs() as logs, _answering_turns() as turns:
                run(_phase2_call(sw, sid_, plan))
            values = {plan["ssn4"], plan["wrong"], OTP}
            if plan["zip_sensitive"]:
                values.add(plan["zip"])
            found = surfaces(sw, logs, [sid_])
            # A size field is a count, not caller data: "event_bytes": 1141 matched a drawn SSN 1141.
            found = {where: [_SIZE_FIELD.sub("", t) for t in texts] for where, texts in found.items()}
            policy = Policy.from_store(sw.world.store, ttl_s=60, clock=sw.world.clock)
            found["timeline (redacted)"] = [
                json.dumps(e, ensure_ascii=False) for e in build_timeline(
                    sw.world.journal(sid_), pii=PiiGuard.for_policy(sw.world.pii, policy))]
            if plan["blurt"]:
                found.pop("relay frames")  # what the agent says is spoken as it said it
            assert leaked(values, found) == [], (plan, leak_report(values, found))
            for slot, result in turns:
                if slot.sensitive and result.utterance_for_ai is not None:
                    assert leaked(values, {"ai text": [result.utterance_for_ai]}) == [], plan
            if plan["mode"] == "deterministic":
                assert resolver.calls == []                                 # Deterministic mode never calls the resolver
            # What this run reached (``--hypothesis-show-statistics``).
            keys = set(sw.world.prompt_keys(sid_))
            event(f"mode {plan['mode']}")
            for key in ("prompt.card", "confirm.make_payment", "slot.corrected",
                        "slot.corrected_sensitive", "slot.confirm_digits", "slot.disambiguate",
                        "slot.confirm_option", "factor.otp"):
                if key in keys:
                    event(f"reached {key}")
            if any(e.get("generated") for e in sw.world.journal(sid_)):
                event("generated text spoken")
            if resolver.calls:
                event("resolver called")
        finally:
            sw.close()


# --------------------------------------------------------------------- Over resolver output


@FRONT_DOOR
@given(data=st.data(), spanish=st.booleans())
def test_phase2_resolver_output_carrying_the_value_never_reaches_ai_text(
        data: st.DataObject, spanish: bool) -> None:
    """The model's output repeats the caller's sensitive value (as the value, a correction, the
    candidates or generated text): the answering turn's AI-bound text is still clean."""
    h = build(config_kw={"accept_spoken_factors": True})
    slot = h.app.d.policy.slot("ssn4")
    speech, generated, _ = data.draw(utterance(slot, False))
    carried = [v for v in (generated, S_VALUE) if v] or [S_VALUE]
    out = data.draw(st.one_of(adversarial(carried), resolver_outputs))
    h.resolver = FakeResolver([out])
    h.app.d.resolver = h.resolver
    call = sid(5)
    try:
        h.start(call, THOMAS, lang_digit="2" if spanish else "")
        h.factor(call, "94105")
        h.set_pending_tool(call, "authenticate", {"tool": "change_address"})
        h.handoff(call, {"reason": "step-up", "tool": "change_address"})
        with _answering_turns() as turns:
            root = h.xml("factor", call, SpeechResult=speech, Confidence="0.9")
        (_, result), = turns
        if result.utterance_for_ai is not None:
            _cpm13(result.utterance_for_ai, "ssn4", result.value, generated)
        st_ = h.state(call)
        if root.find("./Connect/ConversationRelay") is not None and st_.interrupted_utterance:
            _cpm13(h.pending_result(call).callerUtterance or "", "ssn4", None, generated)
    finally:
        segments.clear_sensitive(call)


@FRONT_DOOR
@given(data=st.data())
def test_phase2_a_correction_of_a_sensitive_slot_gets_its_placeholder(
        data: st.DataObject) -> None:
    """At the ZIP prompt (not sensitive), with the SSN last four held, a correction of ``ssn4``:
    whatever value the model gives, the turn's AI-bound text is clean for ``ssn4``, and nothing is sent to the auth service on that turn. (Words the parse reads as a ZIP
    are the ZIP answer, and verified as such.)"""
    h = ssn_first(build(config_kw={"accept_spoken_factors": True}))
    slot = h.app.d.policy.slot("ssn4")
    speech, generated, _ = data.draw(utterance(slot, False))
    value = data.draw(st.one_of(st.none(), st.sampled_from([v for v in (generated, "1111") if v]),
                                st.from_regex(r"[0-9]{4}", fullmatch=True)))
    h.resolver = FakeResolver([ResolverOutput(kind="correction", slot="ssn4", value=value)])
    h.app.d.resolver = h.resolver
    call = sid(6)
    try:
        untrusted_start(h, call)
        h.factor(call, "9999")                                 # the SSN last four, held
        with _answering_turns() as turns:
            h.xml("factor", call, SpeechResult=f"sorry, my last four are {speech}",
                  Confidence="0.9")
        (_, result), = turns
        if result.match == "correction" and result.target == "ssn4":
            assert result.utterance_for_ai is not None
            _cpm13(result.utterance_for_ai, "ssn4", result.value, generated)
            assert not [p for p in h.lam.payloads if p.get("op") == "verify"]   # 6.6
    finally:
        segments.clear_sensitive(call)


# --------------------------------------------------------------------- Over resolver output


def _short_outs(value: str) -> list[ResolverOutput]:
    """Resolver outputs that repeat the (3-digit) value said in the call, as each field allows."""
    return [ResolverOutput(kind="inferred", value=value),
            ResolverOutput(kind="correction", slot="zip", value=value),
            ResolverOutput(kind="intent_switch", value=value)]


def _short_call_resolved(value: str, outs: list[ResolverOutput],
                         scan: str) -> tuple[dict[str, int], dict[str, Any]]:
    """:func:`_short_call`'s call with the value said messily (so the resolver runs) and the
    resolver answering ``outs``."""
    sw = SplitWorld(resolver=FakeResolver(outs, default=INTENT_SWITCH),
                    config={"accept_spoken_factors": True}, policy_doc=SHORT_ZIP)
    try:
        sw.respond(echo_agent(escalate_on="a person"))
        sw.set_factor("C-1001", "zip", value)
        sid_ = call_sid(4)
        seen: dict[str, Any] = {}

        async def go() -> None:
            call = sw.world.dial(THOMAS, sid_)
            await call.start()
            await call.press("")
            await call.speak_to_gather(f"well it's {words_for(value)} and I need my balance too")
            if call.state == "gather":
                await call.press(value)
            if call.state == "relay":
                await call.say("hello")
            seen["fd"], seen["bridge"] = sw.fd.values(sid_), sw.bridge.values(sid_)
            if call.state == "relay":
                await call.say("a person please")
            await call.hangup()

        with capture_logs() as logs:
            run(go())
        found = surfaces(sw, logs, [sid_])
        seen["calls"] = len(sw.world.resolver.calls)
        pattern = value_pattern(scan)
        counts = {where: sum(len(pattern.findall(_SIZE_FIELD.sub("", t.casefold())))
                             for t in found[where])
                  for where in ("log", "journal", "timeline", "task attributes")}
        return counts, seen
    finally:
        sw.close()


@WHOLE_CALLS
@given(value=_digits(3), other=_digits(3), pick=st.integers(0, 2))
def test_cpm13c_phase2_a_short_value_the_model_repeats_is_in_no_log_or_journal_field(
        value: str, other: str, pick: int) -> None:
    assume(value != other)
    with _plain_agent():
        counts, seen = _short_call_resolved(value, [_short_outs(value)[pick]], scan=value)
        control, _ = _short_call_resolved(other, [_short_outs(other)[pick]], scan=value)
    assert seen["calls"] >= 1                                # the resolver ran on that turn
    assert seen["fd"] == seen["bridge"] == frozenset()       # never registered (MIN_SENSITIVE_LEN)
    for where, n in counts.items():
        assert n <= control[where], (where, value)
