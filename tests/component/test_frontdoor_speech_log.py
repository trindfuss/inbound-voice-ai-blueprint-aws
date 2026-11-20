"""The speech log of Gather turns.

* (a) ``speechLog`` on (the default): ``factor_answer`` carries ``speech``, the Gather's
  ``SpeechResult`` for a non-sensitive slot, and for a sensitive one the text the agent would get
  (the value's words as ``[<slot name>]``). A login factor's value is a placeholder too, sensitive
  or not, and its line has no ``value``. Keypad input is not speech. ``off``: no ``speech``.
* (b) ``speechLogRedaction`` on (the default): the logged text through the transcript redaction
  (the policy's types, threshold and mask); a failure logs ``[withheld]``, whatever ``on_failure``
  says.
* (c) The speech is never on ``resolver_call``, the journal or any other line.
* ``OWLBANK_SPEECH_LOG`` / ``OWLBANK_SPEECH_LOG_REDACTION``: ``on`` | ``off``; anything else fails
  INIT without echoing the value.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import pytest

from lambdas.frontdoor.app import FrontDoorConfig
from lambdas.frontdoor.slotflow import SPEECH_WITHHELD
from owlbank import segments
from owlbank.pii import Entity, FakePiiRedactor, PiiGuard
from owlbank.policy import Policy
from owlbank.slots import BedrockResolver, FakeResolver, ResolverOutput
from tests.component.slot_world import capture_logs
from tests.component.test_frontdoor_support import (
    DATA,
    OTP,
    THOMAS,
    Harness,
    all_fixture_factor_values,
    assert_no_leak,
    build,
    sid,
)
from tests.leakscan import compile_values, leaks
from tests.logutil import fields
from tests.unit.test_resolver import StubClient, response, tool_input

Z, S = "94105", "1111"   # C-1001's ZIP (not sensitive) and SSN last four (sensitive)
SSN_WORDS = "the first two are one one and then the last two after that are one one"
ENV = {"OWLBANK_RELAY_WS_URL": "wss://bridge.owlbank.example/ws"}


@dataclass
class NameRedactor(FakePiiRedactor):
    """The fake PII check, which also finds the name "Thomas" (as Comprehend's ``NAME``)."""

    def detect(self, text: str) -> list[Entity]:
        found = FakePiiRedactor.detect(self, text)
        at = text.find("Thomas")
        if at >= 0:
            found.append(Entity(at, at + len("Thomas"), "NAME"))
        return found


def _policy(**redaction: Any) -> Policy:
    raw = json.loads((DATA / "policy.json").read_text())
    raw["redaction"].update(redaction)
    return Policy.from_dict(raw)


def _with_policy(h: Harness, policy: Policy, redactor: FakePiiRedactor) -> Harness:
    h.app.d.policy = policy
    h.app.d.pii = PiiGuard.for_policy(redactor, policy)
    return h


def _answers(records: list[logging.LogRecord]) -> list[dict[str, Any]]:
    return [fields(r) for r in records if getattr(r, "event", None) == "factor_answer"]


def _speak(h: Harness, call: str, text: str) -> Any:
    return h.xml("factor", call, SpeechResult=text, Confidence="0.9")


def _to_ssn_prompt(h: Harness, call: str) -> None:
    h.login_thomas(call)
    h.set_pending_tool(call, "authenticate", {"tool": "change_address"})
    h.handoff(call, {"reason": "step-up", "tool": "change_address"})


# --------------------------------------------------------------------------- (a)


@pytest.mark.parametrize("redaction", [False, True])
def test_a_login_factor_turn_is_logged_with_its_placeholder(caplog, redaction: bool) -> None:
    """The ZIP isn't sensitive in the seed, but it is a login factor: its words are never logged."""
    caplog.set_level(logging.INFO)
    h = build(config_kw={"speech_log_redaction": redaction})
    call = sid(1)
    h.start(call, THOMAS)
    _speak(h, call, "um, my ZIP is nine four one zero five, I think")
    (line,) = _answers(caplog.records)
    assert line["factor"] == "zip" and line["speech"] == "um, my ZIP is [zip], I think"
    assert "value" not in line
    assert ("speech_redact" in line["timings_ms"]) is redaction  # off: no Comprehend call


def test_a_sensitive_turn_is_logged_with_its_placeholder_and_no_value_anywhere(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    h = build(resolver=FakeResolver([ResolverOutput(kind="inferred", value=S)]),
              config_kw={"accept_spoken_factors": True})
    call = sid(2)
    try:
        with capture_logs() as scrubbed:
            _to_ssn_prompt(h, call)
            _speak(h, call, SSN_WORDS)
        assert h.state(call).level == 2                   # verified by the model's value
        (line,) = [a for a in _answers(caplog.records) if a["factor"] == "ssn4"]
        speech = line["speech"]
        # The engine's placeholder text, fail closed: "first two" reads as a number run too.
        assert speech == "the [ssn4] are [ssn4] and then the last two after that are [ssn4]"
        assert "one one" not in speech and S not in speech
        # The raw records (before the log handler's scrub) and the written lines: no value.
        assert_no_leak(h, caplog.records, all_fixture_factor_values() | h.sent_factors | {S})
        patterns = compile_values([S, OTP])
        assert not any(leaks(patterns, t) for t in scrubbed.lines)
    finally:
        segments.clear_sensitive(call)


def test_a_model_found_sensitive_value_in_words_is_a_placeholder_too(caplog) -> None:
    """The value the model found, said as digits inside free text (no deterministic parse)."""
    caplog.set_level(logging.INFO)
    h = build(resolver=FakeResolver([ResolverOutput(kind="inferred", value=S)]),
              config_kw={"accept_spoken_factors": True})
    call = sid(3)
    try:
        _to_ssn_prompt(h, call)
        _speak(h, call, "hmm it's 1 1, and, the rest, 1 1 I think")
        (line,) = [a for a in _answers(caplog.records) if a["factor"] == "ssn4"]
        assert "[ssn4]" in line["speech"] and not leaks(compile_values([S]), line["speech"])
    finally:
        segments.clear_sensitive(call)


def test_keypad_input_is_not_speech(caplog) -> None:
    caplog.set_level(logging.INFO)
    h = build()
    call = sid(4)
    h.login_thomas(call)
    (line,) = _answers(caplog.records)
    assert line["factor"] == "zip" and "speech" not in line and "value" not in line


def test_speech_log_off_logs_no_speech(caplog) -> None:
    caplog.set_level(logging.INFO)
    h = build(config_kw={"speech_log": False})
    call = sid(5)
    h.start(call, THOMAS)
    _speak(h, call, "my ZIP is nine four one zero five")
    (line,) = _answers(caplog.records)
    assert "speech" not in line


def test_the_interrupted_sensitive_slots_digits_are_hidden_in_a_correction_reask(caplog) -> None:
    """An untrusted caller ID logs in with the ZIP and
    then the SSN last four. At ssn4 the model calls the turn a ZIP correction whose value fails,
    so ZIP (not sensitive) is re-asked; the caller then says the SSN's digits, which the model
    calls a ZIP correction again. That turn answers ZIP, but ssn4 is still being asked and its
    value isn't registered yet: its digits are hidden all the same (9.2(a), 18.19(a))."""
    caplog.set_level(logging.DEBUG)
    zip_fix = ResolverOutput(kind="correction", slot="zip", value="0001")
    h = build(resolver=FakeResolver([], default=zip_fix),
              config_kw={"accept_spoken_factors": True})
    call = sid(13)
    said = f"yes my last four are {S}, and what's my balance?"
    try:
        with capture_logs() as scrubbed:
            h.xml("incoming", call, From=THOMAS, To="+12025550100", CallStatus="ringing")
            h.xml("language", call, Digits="")
            h.factor(call, Z)                                     # untrusted: ssn4 is asked next
            _speak(h, call, "eh, zero, hold on... zero zero one")  # a failed ZIP correction
            _speak(h, call, said)                                 # in the ZIP's re-ask
        lines = [a for a in _answers(caplog.records) if "speech" in a]
        assert [a["factor"] for a in lines] == ["ssn4", "zip"]   # the re-ask answers the ZIP
        assert lines[-1]["speech"] == "yes my last [ssn4] are [ssn4], and what's my balance?"  # every run, "four" too (fail closed)
        assert_no_leak(h, caplog.records, all_fixture_factor_values() | h.sent_factors | {S})
        assert not any(leaks(compile_values([S]), t) for t in scrubbed.lines)
    finally:
        segments.clear_sensitive(call)


def test_the_interrupted_sensitive_slots_digits_are_hidden_from_the_agent_too(caplog) -> None:
    """The same turn as an intent switch from the ZIP's re-ask: the text the agent gets (the
    envelope's ``callerUtterance``, ``interrupted_utterance``) hides ssn4's digits (9.2(a))."""
    caplog.set_level(logging.INFO)
    h = build(resolver=FakeResolver([ResolverOutput(kind="correction", slot="zip", value="0001")],
                                    default=ResolverOutput(kind="intent_switch")),
              config_kw={"accept_spoken_factors": True})
    call = sid(14)
    try:
        h.xml("incoming", call, From=THOMAS, To="+12025550100", CallStatus="ringing")
        h.xml("language", call, Digits="")
        h.factor(call, Z)
        _speak(h, call, "eh, zero, hold on... zero zero one")
        _speak(h, call, f"yes my last four are {S}, and what's my balance?")
        assert h.state(call).interrupted_utterance == \
            "yes my last [ssn4] are [ssn4], and what's my balance?"  # every run, "four" too (fail closed)
    finally:
        segments.clear_sensitive(call)


# --------------------------------------------------------------------------- (b)


def test_redaction_on_masks_with_the_policys_mask(caplog) -> None:
    caplog.set_level(logging.INFO)
    redactor = NameRedactor()
    h = _with_policy(build(config_kw={"speech_log_redaction": True}),
                     _policy(entity_types=["SSN", "NAME"]), redactor)
    call = sid(6)
    h.start(call, THOMAS)
    _speak(h, call, "actually my name is Thomas")
    (line,) = _answers(caplog.records)
    assert line["speech"] == "actually my name is [NAME]"
    assert "speech_redact" in line["timings_ms"]
    assert redactor.languages == ["en"]


def test_redaction_on_in_spanish_uses_the_calls_language(caplog) -> None:
    caplog.set_level(logging.INFO)
    redactor = NameRedactor()
    h = _with_policy(build(config_kw={"speech_log_redaction": True}),
                     _policy(entity_types=["NAME"]), redactor)
    call = sid(7)
    h.start(call, THOMAS, lang_digit="2")
    _speak(h, call, "me llamo Thomas")
    (line,) = _answers(caplog.records)
    assert line["speech"] == "me llamo [NAME]" and redactor.languages == ["es"]


def test_redaction_disabled_in_the_policy_masks_nothing_more(caplog) -> None:
    caplog.set_level(logging.INFO)
    redactor = NameRedactor()
    h = _with_policy(build(config_kw={"speech_log_redaction": True}),
                     _policy(enabled=False, entity_types=["NAME"]), redactor)
    call = sid(8)
    h.start(call, THOMAS)
    _speak(h, call, "actually my name is Thomas")
    (line,) = _answers(caplog.records)
    assert line["speech"] == "actually my name is Thomas" and redactor.calls == 0


@pytest.mark.parametrize("on_failure", ["withhold", "allow"])
def test_a_redaction_failure_logs_withheld(caplog, on_failure: str) -> None:
    caplog.set_level(logging.INFO)
    h = _with_policy(build(config_kw={"speech_log_redaction": True}),
                     _policy(entity_types=["NAME"], on_failure=on_failure),
                     NameRedactor(fail=True))
    call = sid(9)
    h.start(call, THOMAS)
    _speak(h, call, "actually my name is Thomas")
    (line,) = _answers(caplog.records)
    assert line["speech"] == SPEECH_WITHHELD == "[withheld]"
    assert "Thomas" not in "\n".join(f"{r.getMessage()} {fields(r)}" for r in caplog.records)


def test_a_redactor_bug_logs_withheld(caplog) -> None:
    caplog.set_level(logging.INFO)

    class Broken(PiiGuard):
        def text(self, *a: Any, **k: Any) -> str:
            raise RuntimeError("bug")

    h = build(config_kw={"speech_log_redaction": True})
    h.app.d.pii = Broken(FakePiiRedactor())
    call = sid(10)
    h.start(call, THOMAS)
    _speak(h, call, "actually my name is Thomas")
    (line,) = _answers(caplog.records)
    assert line["speech"] == SPEECH_WITHHELD


def test_a_sensitive_turn_is_redacted_after_its_placeholder(caplog) -> None:
    caplog.set_level(logging.INFO)
    redactor = NameRedactor()
    h = _with_policy(build(resolver=FakeResolver([ResolverOutput(kind="inferred", value=S)]),
                           config_kw={"accept_spoken_factors": True,
                                      "speech_log_redaction": True}),
                     _policy(entity_types=["NAME"]), redactor)
    call = sid(11)
    try:
        _to_ssn_prompt(h, call)
        _speak(h, call, f"Thomas here, {SSN_WORDS}")
        (line,) = [a for a in _answers(caplog.records) if a["factor"] == "ssn4"]
        assert line["speech"] == ("[NAME] here, the [ssn4] are [ssn4] and then the last two "
                                  "after that are [ssn4]")
    finally:
        segments.clear_sensitive(call)


# --------------------------------------------------------------------------- (c)


def test_speech_is_never_on_resolver_call_or_the_journal(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    said = "nine four, uh, one, hold on, zero five"
    client = StubClient(response(tool_input(kind="inferred", value=Z)))
    h = build(resolver=BedrockResolver(lambda _budget: client))   # the real one logs resolver_call
    call = sid(12)
    h.start(call, THOMAS)
    _speak(h, call, said)
    assert client.calls, "the resolver ran on this turn"
    assert [a["speech"] for a in _answers(caplog.records)] == ["[zip], uh, [zip], hold on, [zip]"]
    assert [r for r in caplog.records if getattr(r, "event", None) == "resolver_call"]
    others = [r for r in caplog.records if getattr(r, "event", None) != "factor_answer"]
    assert all("speech" not in fields(r) for r in others)
    assert all(said not in f"{r.getMessage()} {fields(r)}" for r in others)
    events = h.journal(call)
    assert all("speech" not in e for e in events)
    assert said not in json.dumps(events, ensure_ascii=False)


# --------------------------------------------------------------------------- config


def test_config_defaults_to_on_and_on() -> None:
    cfg = FrontDoorConfig.from_env(ENV)
    assert (cfg.speech_log, cfg.speech_log_redaction) == (True, True)
    bare = FrontDoorConfig(relay_ws_url="wss://x.example/ws")     # as the component tests build it
    assert (bare.speech_log, bare.speech_log_redaction) == (True, True)


@pytest.mark.parametrize(("log", "red"), [("on", "on"), ("off", "off"), ("OFF", " On "), ("", "")])
def test_config_reads_both(log: str, red: str) -> None:
    cfg = FrontDoorConfig.from_env({**ENV, "OWLBANK_SPEECH_LOG": log,
                                    "OWLBANK_SPEECH_LOG_REDACTION": red})
    assert cfg.speech_log is (log.strip().lower() != "off")
    assert cfg.speech_log_redaction is (red.strip().lower() != "off")


@pytest.mark.parametrize("name", ["OWLBANK_SPEECH_LOG", "OWLBANK_SPEECH_LOG_REDACTION"])
def test_an_invalid_value_fails_init_without_echoing_it(name: str) -> None:
    with pytest.raises(ValueError, match=name) as exc:
        FrontDoorConfig.from_env({**ENV, name: "s3cr3t-ish"})
    assert "s3cr3t-ish" not in str(exc.value)
