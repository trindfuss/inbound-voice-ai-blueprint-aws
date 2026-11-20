"""The escalation briefing's summary and transcript (the summary and transcript the model is given): what reaches the model, how a
failure degrades (transcript, then summary, then facts alone), and that no text is logged."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from owlbank.briefing import BriefingFacts, facts_payload

from tests.component.test_frontdoor_briefing import (  # noqa: F401  (briefed is a fixture)
    CLAUSE, escalate, spoken, briefed,
)
from tests.component.test_frontdoor_support import clear_sensitive, sid  # noqa: F401

PARTS = [{"id": "p1", "type": "CUSTOMER"}, {"id": "p2", "type": "AI_AGENT"}]
COMMS = [
    {"content": {"text": "I am interested in a home equity line of credit"}, "author": {"participantId": "p1"}},
    {"content": {"text": "Let me find someone who can help"}, "author": {"participantId": "p2"}},
]


class CapturingModel:
    def __init__(self) -> None:
        self.facts: list[BriefingFacts] = []

    def summarise(self, facts: BriefingFacts, *, deadline: float) -> str | None:
        self.facts.append(facts)
        return CLAUSE


class Reader:
    def __init__(self, *, comms: Any = COMMS, found: str | None = "conv_9") -> None:
        self.comms, self.found = comms, found
        self.reads: list[str] = []

    def resolve_conversation(self, call_sid: str) -> str | None:
        return self.found

    def communications(self, conversation_id: str, *, more: Any = None):
        self.reads.append(conversation_id)
        return self.comms

    def participants(self, conversation_id: str, *, more: Any = None):
        return PARTS


def test_the_summary_and_the_transcript_reach_the_model(briefed) -> None:
    model = CapturingModel()
    h = briefed(model)
    h.app.d.transcripts = Reader()
    root = escalate(h, sid(40))
    (facts,) = model.facts
    assert facts.summary == "Card lost."
    assert facts.transcript == ("Customer: I am interested in a home equity line of credit\n"
                                "AI agent: Let me find someone who can help")
    payload = facts_payload(facts)
    assert payload["agent_summary"] == "Card lost." and "home equity" in payload["conversation"]
    assert CLAUSE in spoken(root)


def test_the_handoffs_conversation_id_is_read(briefed) -> None:
    h = briefed(CapturingModel())
    reader = h.app.d.transcripts = Reader(found=None)
    escalate(h, sid(41), conversationId="conv_handoff")
    assert reader.reads == ["conv_handoff"]


def test_no_conversation_means_no_transcript_but_the_summary_stays(briefed) -> None:
    model = CapturingModel()
    h = briefed(model)
    h.app.d.transcripts = Reader(found=None)
    escalate(h, sid(42))
    (facts,) = model.facts
    assert facts.transcript is None and facts.summary == "Card lost."


def test_a_failed_read_degrades_to_the_summary(briefed) -> None:
    model = CapturingModel()
    h = briefed(model)
    h.app.d.transcripts = Reader(comms=None)
    root = escalate(h, sid(43))
    (facts,) = model.facts
    assert facts.transcript is None and facts.summary == "Card lost."
    assert CLAUSE in spoken(root)


def test_no_reader_and_no_summary_is_facts_alone(briefed) -> None:
    model = CapturingModel()
    h = briefed(model)
    h.login_thomas(sid(44))
    from owlbank import callstate
    callstate.mutate(h.store, sid(44), lambda s: {"ai_sessions": 1})
    h.set_pending_tool(sid(44), "escalate", {"reason": "caller_requested_human", "summary": ""})
    h.handoff(sid(44), {"reason": "escalate", "escalationReason": "caller_requested_human",
                        "journal": "jr-7f3a"})
    (facts,) = model.facts
    assert facts.summary is None and facts.transcript is None


def test_a_call_with_no_ai_session_reads_no_transcript(briefed) -> None:
    model = CapturingModel()
    h = briefed(model)
    reader = h.app.d.transcripts = Reader()
    h.login_thomas(sid(45))
    h.set_pending_tool(sid(45), "escalate", {"reason": "caller_requested_human", "summary": "x"})
    h.handoff(sid(45), {"reason": "escalate", "escalationReason": "caller_requested_human",
                        "journal": "jr-7f3a"})
    assert reader.reads == [] and model.facts[0].transcript is None


def test_a_repeated_request_keeps_the_summary(briefed) -> None:
    model = CapturingModel()
    h = briefed(model)
    call = sid(46)
    escalate(h, call)
    assert h.state(call).escalation["summary"] == "Card lost."
    h.set_pending_tool(call, "escalate", {"reason": "caller_requested_human", "summary": ""})
    h.handoff(call, {"reason": "escalate", "escalationReason": "caller_requested_human",
                     "journal": "jr-7f3a"})
    assert [f.summary for f in model.facts] == ["Card lost.", "Card lost."]


def test_the_texts_are_not_logged(briefed, caplog: pytest.LogCaptureFixture) -> None:
    h = briefed(CapturingModel())
    h.app.d.transcripts = Reader()
    caplog.set_level(logging.DEBUG)
    escalate(h, sid(47))
    logged = "\n".join(r.getMessage() + repr(getattr(r, "__dict__", {})) for r in caplog.records)
    assert "home equity" not in logged and "Card lost" not in logged
    event = next(r for r in caplog.records if r.getMessage() == "briefing_generated")
    assert event.has_summary is True and event.has_transcript is True


def test_a_slow_pii_check_costs_the_model_budget_not_the_deadline(briefed, monkeypatch) -> None:
    """Review finding: the summary check and the transcript read share one deadline, and the model
    gets what is left of the briefing budget."""
    import time

    from owlbank import briefing, pii as pii_mod
    model = CapturingModel()
    h = briefed(model, config_kw={"briefing_budget_ms": 2000})
    h.app.d.transcripts = Reader()
    budgets: list[int] = []
    real = briefing.generate
    monkeypatch.setattr(briefing, "generate",
                        lambda m, f, **kw: (budgets.append(kw["budget_ms"]), real(m, f, **kw))[1])
    slow = pii_mod.FakePiiRedactor.redact

    def slower(self, text, **kw):
        time.sleep(3)
        return slow(self, text, **kw)

    monkeypatch.setattr(pii_mod.FakePiiRedactor, "redact", slower)
    started = time.monotonic()
    root = escalate(h, sid(48))
    assert time.monotonic() - started < 2.5          # not the sum of two 3 s PII calls
    (facts,) = model.facts
    assert facts.summary is None and facts.transcript is None   # both late: dropped
    assert budgets and budgets[0] < 2000                  # the wait came off the model's budget
    assert CLAUSE in spoken(root)
