"""Model-assisted slots: properties of ``factor_answer.speech``.

**Sensitive slot.** For any utterance at the SSN last-four prompt and any resolver output (also one
that carries the value the caller said, or the registered value), the ``speech`` the front door
logs, read from the raw record before the log handler's exact-value scrub, holds no exact-value
match of the value said or the value accepted, and outside the ``[ssn4]`` tokens every number run
is one token and one digit. It holds with ``speechLogRedaction`` off, on, and on with the
PII check failing (then it is ``[withheld]``). It is the same text the agent would get.

**Login factor.** At the ZIP prompt (a login factor the seed doesn't mark sensitive), for any
utterance and resolver output and with ``speechLogRedaction`` off, on, or on with the PII check
failing, no log record of the turn, from any logger and before any filter, holds the value said or
the value accepted: the speech carries ``[zip]`` instead and the line carries no ``value``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.frontdoor.slotflow import SPEECH_WITHHELD
from owlbank import segments
from owlbank.pii import FakePiiRedactor, PiiGuard
from owlbank.segments import contains_value
from owlbank.slots import FakeResolver, ResolverOutput, number_runs
from tests.component.test_frontdoor_support import TEST_FACTORS, THOMAS, build, sid
from tests.leakscan import compile_values, leaks
from tests.property.test_ai_text_props import resolver_outputs, utterance
from tests.property.test_sensitive_surfaces_props import (
    INTENT_SWITCH,
    _answering_turns,
    adversarial,
)

S_VALUE = TEST_FACTORS["C-1001"]["ssn4"]
Z_VALUE = TEST_FACTORS["C-1001"]["zip"]
PROPS = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])


class _Raw(logging.Handler):
    """The records as made, before any filter (no ``ScrubFilter``): the strongest check."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _raw_answers() -> Iterator[list[dict[str, Any]]]:
    logger = logging.getLogger("lambdas.frontdoor.slotflow")
    handler, old = _Raw(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    out: list[dict[str, Any]] = []
    try:
        yield out
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old)
        out.extend(r.__dict__ for r in handler.records if getattr(r, "event", None) == "factor_answer")


def _shape(text: str, name: str) -> None:
    rest = text.replace(f"[{name}]", " | ")
    for r in number_runs(rest, ("en", "es")):
        assert r.tokens == 1 and r.digits is not None and len(r.digits) == 1, text


@PROPS
@given(data=st.data(), spanish=st.booleans(),
       redaction=st.sampled_from(["off", "on", "pii_down"]))
def test_the_logged_speech_of_a_sensitive_slot_holds_no_value(
        data: st.DataObject, spanish: bool, redaction: str) -> None:
    h = build(config_kw={"accept_spoken_factors": True,
                         "speech_log_redaction": redaction != "off"})
    h.app.d.pii = PiiGuard.for_policy(FakePiiRedactor(fail=redaction == "pii_down"), h.app.d.policy)
    slot = h.app.d.policy.slot("ssn4")
    assert slot.sensitive
    speech, generated, _ = data.draw(utterance(slot, False))
    carried = [v for v in (generated, S_VALUE) if v]
    out = data.draw(st.one_of(st.just(INTENT_SWITCH), adversarial(carried), resolver_outputs,
                              st.builds(ResolverOutput, kind=st.just("inferred"),
                                        value=st.sampled_from(carried))))
    h.resolver = FakeResolver([out])
    h.app.d.resolver = h.resolver
    call = sid(1)
    try:
        h.start(call, THOMAS, lang_digit="2" if spanish else "")
        h.factor(call, "94105")
        h.set_pending_tool(call, "authenticate", {"tool": "change_address"})
        h.handoff(call, {"reason": "step-up", "tool": "change_address"})
        with _raw_answers() as answers, _answering_turns() as turns:
            h.xml("factor", call, SpeechResult=speech, Confidence="0.9")
        (_, result), = turns
        (line,) = [a for a in answers if a["factor"] == "ssn4"]
        logged = line["speech"]
        if redaction == "pii_down":
            assert logged == SPEECH_WITHHELD
            return
        if redaction == "off":
            assert logged == result.utterance_for_ai     # the text the agent would get
        _shape(logged, "ssn4")
        values = {v for v in (generated, result.value) if v}
        for v in values:
            assert not contains_value(logged.replace("[ssn4]", " | "), v), (speech, logged)
        assert not leaks(compile_values(v for v in values if v.isdigit()), logged), (speech, logged)
    finally:
        segments.clear_sensitive(call)


@contextlib.contextmanager
def _every_record() -> Iterator[list[logging.LogRecord]]:
    """Every record any logger makes, as made (root handler, no filter)."""
    root, handler = logging.getLogger(), _Raw()
    old = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        root.removeHandler(handler)
        root.setLevel(old)


def _record_text(record: logging.LogRecord) -> str:
    return " ".join([record.getMessage(), *(repr(v) for k, v in record.__dict__.items()
                                            if k not in ("msg", "args", "created", "msecs",
                                                         "relativeCreated"))])


@PROPS
@given(data=st.data(), redaction=st.sampled_from(["off", "on", "pii_down"]))
def test_no_log_record_of_a_factor_slot_holds_the_spoken_value(
        data: st.DataObject, redaction: str) -> None:
    h = build(config_kw={"accept_spoken_factors": True,
                         "speech_log_redaction": redaction != "off"})
    h.app.d.pii = PiiGuard.for_policy(FakePiiRedactor(fail=redaction == "pii_down"), h.app.d.policy)
    slot = h.app.d.policy.slot("zip")
    assert slot.is_factor and not slot.sensitive  # the seed
    speech, generated, _ = data.draw(utterance(slot, False))
    carried = [v for v in (generated, Z_VALUE) if v]
    out = data.draw(st.one_of(st.just(INTENT_SWITCH), adversarial(carried), resolver_outputs,
                              st.builds(ResolverOutput, kind=st.just("inferred"),
                                        value=st.sampled_from(carried))))
    h.resolver = FakeResolver([out])
    h.app.d.resolver = h.resolver
    call = sid(2)
    try:
        h.start(call, THOMAS)
        with _every_record() as records, _answering_turns() as turns:
            h.xml("factor", call, SpeechResult=speech, Confidence="0.9")
        (_, result), = turns
        (line,) = [r.__dict__ for r in records if getattr(r, "event", None) == "factor_answer"]
        assert line["factor"] == "zip" and "value" not in line
        if redaction == "pii_down":
            assert line["speech"] == SPEECH_WITHHELD
        values = {v for v in (generated, result.value) if v}
        patterns = compile_values(v for v in values if v.isdigit())
        for r in records:
            text = _record_text(r)
            for v in values:
                assert not contains_value(text.replace("[zip]", " | "), v), (speech, text)
            assert not leaks(patterns, text), (speech, text)
        if redaction == "off":
            _shape(line["speech"], "zip")
    finally:
        segments.clear_sensitive(call)
