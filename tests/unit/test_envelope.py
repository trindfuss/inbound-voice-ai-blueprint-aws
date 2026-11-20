"""Unit tests for owlbank.envelope."""

from __future__ import annotations

import json

import pytest

from owlbank.envelope import EnvelopeError, ResultEnvelope, Spoken

BALANCE = {
    "status": "completed",
    "tool": "get_balance",
    "spoken": [
        {"template": "balance.read", "text": "Your checking account ending in four, eight, two, one..."},
        {"template": "balance.read.pending", "text": "That includes seventy-five dollars..."},
        {"template": "common.anything_else", "text": "Is there anything else I can help you with?"},
    ],
    "data": {"account": "checking", "last4": "4821", "available": "1234.56", "current": "1310.06",
             "pending": "75.50", "currency": "USD", "asOf": "2026-10-10T14:58:00Z"},
    "next": "await_caller",
}

INTERRUPTED = {
    "status": "interrupted",
    "tool": "authenticate",
    "forTool": "make_payment",
    "spoken": [{"template": "factor.zip", "text": "Please enter your five-digit ZIP code."}],
    "callerUtterance": "Actually, can you give me my checking balance instead?",
    "completed": False,
    "next": "handle_utterance",
}


@pytest.mark.parametrize("doc", [BALANCE, INTERRUPTED], ids=["balance", "intent-switch"])
def test_handover_examples_round_trip(doc: dict) -> None:
    env = ResultEnvelope.from_json(json.dumps(doc))
    assert ResultEnvelope.from_json(env.to_json()) == env
    out = json.loads(env.to_json())
    expected = {**doc, "data": doc.get("data", {})}
    assert out == expected


def test_await_caller_needs_spoken() -> None:
    with pytest.raises(EnvelopeError, match="non-empty spoken"):
        ResultEnvelope(status="completed", tool="get_balance", next="await_caller")


def test_await_caller_needs_follow_up_last() -> None:
    with pytest.raises(EnvelopeError, match="follow-up"):
        ResultEnvelope(status="completed", tool="get_balance", next="await_caller",
                       spoken=[Spoken("common.anything_else", "?"), Spoken("balance.read", "x")])


def test_handle_utterance_needs_caller_utterance() -> None:
    with pytest.raises(EnvelopeError, match="callerUtterance"):
        ResultEnvelope(status="interrupted", tool="authenticate", next="handle_utterance")
    with pytest.raises(EnvelopeError):
        ResultEnvelope(status="interrupted", tool="authenticate", next="handle_utterance",
                       callerUtterance="")


@pytest.mark.parametrize(
    "kw",
    [
        {"status": "done"},
        {"next": "say_something"},
        {"tool": ""},
        {"data": {"amount": float("nan")}},
        {"data": {"when": object()}},
        {"data": "not a dict"},
        {"completed": "yes"},
        {"forTool": ""},
    ],
)
def test_invalid_fields_rejected(kw: dict) -> None:
    base = {"status": "failed", "tool": "get_balance", "next": "offer_alternative"}
    with pytest.raises(EnvelopeError):
        ResultEnvelope(**{**base, **kw})


def test_retry_original_tool_after_verify() -> None:
    env = ResultEnvelope(status="verified", tool="authenticate", forTool="make_payment",
                         spoken=[Spoken("stepup.success", "Thanks, you're verified.")],
                         data={"levelReached": 3}, next="retry_original_tool")
    assert json.loads(env.to_json())["forTool"] == "make_payment"
    assert "callerUtterance" not in json.loads(env.to_json())


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        json.dumps([1, 2]),
        json.dumps({**BALANCE, "extra": 1}),
        json.dumps({k: v for k, v in BALANCE.items() if k != "tool"}),
        json.dumps({**BALANCE, "spoken": [{"template": "common.anything_else"}]}),
        json.dumps({**BALANCE, "spoken": "x"}),
    ],
)
def test_from_json_rejects_bad_input(text: str) -> None:
    with pytest.raises(EnvelopeError):
        ResultEnvelope.from_json(text)


def test_non_ascii_preserved() -> None:
    env = ResultEnvelope(status="completed", tool="get_balance", next="await_caller",
                         spoken=[Spoken("common.anything_else", "¿Hay algo más en lo que pueda ayudarle?")])
    assert "¿Hay algo más" in env.to_json()
    assert ResultEnvelope.from_json(env.to_json()) == env
