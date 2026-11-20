"""The front door's sealed values and the
one answering-turn text.

After an accepted value of a sensitive slot the front door calls ``sensitive.remember`` (sealed into
call state's ``sealed_sensitive``, registered in this process), and every call-state load calls
``sensitive.restore``. Two front-door instances are simulated in one test process by giving each
its own registry (its own ``segments`` registered values and ``sensitive`` process state) over
one shared store, as Lambda instances share only the calls table. A value that can't be unsealed
withholds the journal surfaces the front door writes (the screen-pop summary). The answering turn's
text is built once (``SlotResult.utterance_for_ai``) and that one string is the envelope's
``callerUtterance``, call state's ``interrupted_utterance`` and, when an agent hands it back as
the summary, the screen pop's ``ai_summary``.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest

from lambdas.frontdoor.app import FrontDoor
from owlbank import callstate as cs
from owlbank import obs, segments, sensitive
from owlbank.policy import Policy
from owlbank.slots import FakeResolver, ResolverOutput
from owlbank.token_seal import FakeSealer
from tests import logutil
from tests.component.test_frontdoor_support import (
    DATA, TEST_FACTORS, THOMAS, Harness, all_fixture_factor_values, assert_no_leak, build, relay,
    sid,
)

Z, S = TEST_FACTORS["C-1001"]["zip"], TEST_FACTORS["C-1001"]["ssn4"]
INTENT_SWITCH = ResolverOutput(kind="intent_switch")
WITHHELD_EN = "Summary withheld: PII check unavailable."


class Instance:
    """One front-door Lambda instance: its own app object and its own process registry, over the
    harness's shared store, sealer and backend."""

    def __init__(self, h: Harness, *, fresh_app: bool = True) -> None:
        self.h = h if not fresh_app else replace(h, app=FrontDoor(h.app.d, h.app.cfg), bodies=[])
        self.registered: dict[str, set[str]] = {}
        self.state = sensitive._ProcessState()

    @contextlib.contextmanager
    def active(self) -> Iterator[Harness]:
        saved = segments._SENSITIVE, sensitive._state
        segments._SENSITIVE, sensitive._state = self.registered, self.state  # type: ignore[assignment]
        try:
            yield self.h
        finally:
            segments._SENSITIVE, sensitive._state = saved  # type: ignore[assignment]


@pytest.fixture
def scrubbed(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Captured records pass ``obs.ScrubFilter`` first, as the front door's JSON handler does."""
    caplog.set_level(logging.DEBUG)
    scrub = obs.ScrubFilter()
    caplog.handler.addFilter(scrub)
    yield caplog
    caplog.handler.removeFilter(scrub)


def _norm(value: str) -> str:
    return segments._norm_text(value)


def _events(records: list[logging.LogRecord], name: str) -> list[logging.LogRecord]:
    return [r for r in records if getattr(r, "event", None) == name]


def _step_up_to_ssn4(h: Harness, call: str) -> None:
    """Thomas logs in with his ZIP; the agent asks for a level-2 tool: the SSN last four is asked."""
    h.login_thomas(call)
    h.set_pending_tool(call, "authenticate", {"tool": "change_address"})
    root = h.handoff(call, {"reason": "step-up", "tool": "change_address", "journal": "jr-1"})
    assert root.find("Gather") is not None


def _policy(**changes: Any) -> Policy:
    raw = json.loads((DATA / "policy.json").read_text())
    for path, value in changes.items():
        *parents, leaf = path.split("__")
        node = raw
        for p in parents:
            node = node[p]
        node[leaf] = value
    return Policy.from_dict(raw)


# --------------------------------------------------------------------------- across instances


def test_value_remembered_on_one_instance_is_scrubbed_from_the_other_after_it_loads(
        scrubbed: pytest.LogCaptureFixture) -> None:
    h = build()
    a, b = Instance(h, fresh_app=False), Instance(h)
    call = sid(1)
    with a.active() as ha:
        _step_up_to_ssn4(ha, call)
        ha.factor(call, S)
        assert _norm(S) in segments.sensitive_values(call)
    st = h.state(call)
    assert st.level == 2 and len(st.sealed_sensitive) == 1  # sealed once, ZIP never
    assert S not in json.dumps(list(st.sealed_sensitive))
    with b.active() as hb:
        assert segments.sensitive_values(call) == frozenset()  # B never saw the answer
        hb.set_pending_tool(call, "authenticate", {"tool": "list_recent_transactions"})
        hb.handoff(call, {"reason": "step-up", "tool": "list_recent_transactions"})  # B loads it
        assert segments.sensitive_values(call) == frozenset({_norm(S)})
        tokens = obs.bind(call_sid=call)
        try:
            logging.getLogger("lambdas.frontdoor.app").warning("upstream echoed %s", S)
        finally:
            obs.unbind(tokens)
    line = [r for r in scrubbed.records if "upstream echoed" in r.getMessage()]
    assert len(line) == 1 and S not in line[0].getMessage()
    assert S not in logutil.text_of(scrubbed.records)
    # B restored the existing entry: it never sealed a second copy.
    assert len(h.state(call).sealed_sensitive) == 1


def test_non_sensitive_zip_is_neither_sealed_nor_registered(
        scrubbed: pytest.LogCaptureFixture) -> None:
    h = build()
    a = Instance(h, fresh_app=False)
    call = sid(2)
    with a.active() as ha:
        ha.login_thomas(call)
        assert segments.sensitive_values(call) == frozenset()
    assert h.state(call).sealed_sensitive == ()
    assert h.state(call).level == 1
    assert not _events(scrubbed.records, "sensitive_scrub_skipped")


def test_short_sensitive_value_logs_skip_once_and_is_still_sealed(
        scrubbed: pytest.LogCaptureFixture) -> None:
    h = build()
    # A 3-digit sensitive slot (under MIN_SENSITIVE_LEN): ZIP flagged, length 3.
    h.app.d.policy = _policy(slots__zip__sensitive=True, factors__zip__constraints={"length": 3})
    a, b = Instance(h, fresh_app=False), Instance(h)
    call = sid(3)
    with a.active() as ha:
        ha.start(call, THOMAS)
        ha.xml("factor", call, Digits="941")  # not Thomas's ZIP: asked again
        ha.xml("factor", call, Digits="941")
        assert segments.sensitive_values(call) == frozenset()
    skipped = _events(scrubbed.records, "sensitive_scrub_skipped")
    assert len(skipped) == 1
    assert skipped[0].slot == "zip" and skipped[0].reason == "too_short"
    assert len(h.state(call).sealed_sensitive) == 1  # sealed, once
    with b.active() as hb:  # another instance restores it: knows it, never registers it
        hb.app._load(call)
        assert segments.sensitive_values(call) == frozenset()
        assert not sensitive.is_unverifiable(call)
    assert len(_events(scrubbed.records, "sensitive_scrub_skipped")) == 2  # once per process


def test_sensitive_yes_no_answer_is_sealed_but_never_registered(
        scrubbed: pytest.LogCaptureFixture) -> None:
    from tests.component.test_frontdoor_caller_id_trust import confirm_handoff, to_level3

    h = build()
    h.app.d.policy = _policy(slots__yes_no__sensitive=True)
    a = Instance(h, fresh_app=False)
    call = sid(4)
    with a.active() as ha:
        to_level3(ha, call)
        before = len(h.state(call).sealed_sensitive)  # ssn4 and the OTP
        assert before == 2
        confirm_handoff(ha, call)
        ha.xml("factor", call, SpeechResult="yes", Confidence="0.9")
        assert "yes" not in segments.sensitive_values(call)
    assert len(h.state(call).sealed_sensitive) == before + 1
    skipped = _events(scrubbed.records, "sensitive_scrub_skipped")
    assert [(r.slot, r.reason) for r in skipped] == [("yes_no", "slot_type")]


def test_early_finished_sensitive_digits_are_sealed(scrubbed: pytest.LogCaptureFixture) -> None:
    from lambdas.frontdoor.calls import FakeCallUpdater
    from tests.component.test_frontdoor_early_digits import gather_of, partial

    h = build(config_kw={"accept_spoken_factors": True}, calls=FakeCallUpdater())
    a = Instance(h, fresh_app=False)
    call = sid(5)
    words = " ".join(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                      "nine"][int(c)] for c in S)
    with a.active() as ha:
        ha.login_thomas(call)
        ha.set_pending_tool(call, "authenticate", {"tool": "change_address"})
        g = gather_of(ha.handoff(call, {"reason": "step-up", "tool": "change_address"}))
        partial(ha, call, g, words)
        assert _norm(S) in segments.sensitive_values(call)
    assert len(h.state(call).sealed_sensitive) == 1
    assert not any(words in r.getMessage() for r in scrubbed.records)


# --------------------------------------------------------------------------- fail closed


def test_unseal_failure_withholds_the_screen_pop_summary(
        scrubbed: pytest.LogCaptureFixture) -> None:
    h = build()
    b = Instance(h)
    call = sid(6)
    a = Instance(h, fresh_app=False)
    with a.active() as ha:
        ha.login_thomas(call)
    # An entry this deployment's key can't open (a corrupted or foreign blob).
    blob = FakeSealer(b"another-key").seal('{"v":1,"slot":"ssn4","value":"' + S + '"}',
                                           call_sid=call)
    cs.mutate(h.store, call, lambda s: {cs.F_SEALED_SENSITIVE: (*s.sealed_sensitive, blob)})
    summary = "Caller wants to dispute a charge."
    with b.active() as hb:
        hb.set_pending_tool(call, "escalate", {"reason": "complaint", "summary": summary})
        root = hb.handoff(call, {"reason": "escalate", "escalationReason": "complaint",
                                 "journal": "jr-9"})
        assert sensitive.is_unverifiable(call)
    attrs = json.loads(root.find("Enqueue/Task").text or "{}")
    assert attrs["ai_summary"] == WITHHELD_EN
    failed = _events(scrubbed.records, "sensitive_unseal_failed")
    assert failed and all(getattr(r, "decision", None) == "withhold_texts" for r in failed)
    assert S not in logutil.text_of(scrubbed.records)
    # The instance that can read every entry still passes the summary.
    with a.active():
        assert not sensitive.is_unverifiable(call)


# --------------------------------------------------------------------------- one answering turn


def test_answering_turn_is_one_string_on_every_surface(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    h = build(resolver=FakeResolver([INTENT_SWITCH]))
    a, b = Instance(h, fresh_app=False), Instance(h)
    call = sid(7)
    utterance = "what's my balance? I think it ends in one two"
    expected = "what's my balance? I think it ends in [ssn4]"
    with a.active() as ha:
        _step_up_to_ssn4(ha, call)
        root = ha.xml("factor", call, SpeechResult=utterance, Confidence="0.9")
    assert relay(root) is not None
    env = h.pending_result(call)
    assert env.status == "interrupted" and env.next == "handle_utterance"
    assert env.callerUtterance == expected
    assert h.state(call).interrupted_utterance == expected
    # An agent that echoes the turn as its hand-off summary: the screen pop gets the same string.
    with b.active() as hb:
        hb.set_pending_tool(call, "escalate", {"reason": "complaint", "summary": expected})
        done = hb.handoff(call, {"reason": "escalate", "escalationReason": "complaint"})
    attrs = json.loads(done.find("Enqueue/Task").text or "{}")
    assert attrs["ai_summary"] == expected
    # No journal event carries the caller's words for a digit or sensitive slot.
    journal = json.dumps(h.journal(call))
    assert "ends in" not in journal and "one two" not in journal
    assert_no_leak(h, caplog.records, all_fixture_factor_values())


def test_non_sensitive_answering_turn_is_the_callers_words(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    h = build(resolver=FakeResolver([INTENT_SWITCH]))
    a = Instance(h, fresh_app=False)
    call = sid(8)
    utterance = "Actually what are your branch hours on Saturday?"
    with a.active() as ha:
        ha.start(call, THOMAS)
        ha.xml("factor", call, SpeechResult=utterance, Confidence="0.9")
    env = h.pending_result(call)
    assert env.callerUtterance == utterance == h.state(call).interrupted_utterance
    assert h.state(call).sealed_sensitive == ()
