"""``tool_outcome``: how one inline or Gateway tool call ended, for the bridge's ``tool_result``
line and the tool error-rate alarm.

Each code family and each cancellation form; the property tests over generated inputs are in
``tests/property/test_tool_outcome_props.py``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from owlbank.tools_contract import (
    BRIDGE_REJECTION_CODES,
    BUSINESS_REFUSAL_CODES,
    KNOWN_ERROR_CODES,
    TOOL_OUTCOMES,
    auth_level_insufficient,
    tool_error_code,
    tool_outcome,
)
from tac_bridge.agent_client_strands import CANCELLED


# --------------------------------------------------------------------------- inline


def test_inline_success_is_a_result_without_an_error() -> None:
    assert tool_outcome("inline", {"tool": "list_recent_transactions", "data": {"transactions": []}}) == "success"
    assert tool_outcome("inline", {}) == "success"


@pytest.mark.parametrize("code", ["UNAVAILABLE", "INVALID_ARGUMENT", "UNKNOWN_OPERATION"])
def test_inline_failures_are_errors(code: str) -> None:
    """The bank call raised or timed out (UNAVAILABLE), or our request was malformed."""
    assert tool_outcome("inline", {"error": code, "tool": "make_payment"}) == "error"


@pytest.mark.parametrize("code", ["SOMETHING_NEW", "unavailable", "", "WRONG_CALL"])
def test_an_unknown_bank_code_is_an_error(code: str) -> None:
    """Fails towards counting: a code nobody listed is a failure until it is listed."""
    assert tool_outcome("inline", {"error": code}) == "error"


@pytest.mark.parametrize("content", [{"error": None}, {"error": {"code": "X"}}, "text", None, 3, ["x"]])
def test_an_inline_result_of_no_known_shape_is_an_error(content: Any) -> None:
    assert tool_outcome("inline", content) == "error"


@pytest.mark.parametrize("code", ["NOT_FOUND", "INSUFFICIENT_FUNDS", "AUTH_LEVEL_INSUFFICIENT"])
def test_business_answers_are_refused(code: str) -> None:
    assert tool_outcome("inline", {"error": code, "tool": "transfer_funds"}) == "refused"


def test_the_level_refusal_shape_is_refused() -> None:
    assert tool_outcome("inline", auth_level_insufficient("change_address")) == "refused"


@pytest.mark.parametrize("code", ["INVALID_ARGUMENTS", "UNKNOWN_TOOL", "CALL_ALONE", "ONE_HANDOFF_AT_A_TIME"])
def test_bridge_rejections_are_rejected(code: str) -> None:
    assert tool_outcome("inline", {"error": code, "tool": "x", "details": ["$.a: required"]}) == "rejected"


def test_the_code_sets_are_disjoint_and_the_outcomes_fixed() -> None:
    assert not BUSINESS_REFUSAL_CODES & BRIDGE_REJECTION_CODES
    assert "UNAVAILABLE" not in BUSINESS_REFUSAL_CODES | BRIDGE_REJECTION_CODES
    assert TOOL_OUTCOMES == ("success", "error", "refused", "rejected", "cancelled")


def test_an_exception_other_than_cancellation_is_an_error() -> None:
    assert tool_outcome("inline", None, exception=RuntimeError("boom")) == "error"
    assert tool_outcome("inline", None, exception=TimeoutError()) == "error"


# --------------------------------------------------------------------------- cancellation


def test_the_synthesized_cancelled_resume_answer_is_cancelled() -> None:
    """``agent_client_strands.CANCELLED``: the answer to an inline call the bridge never answered."""
    assert tool_outcome("inline", dict(CANCELLED)) == "cancelled"


@pytest.mark.parametrize("kind", ["inline", "gateway"])
@pytest.mark.parametrize("exc", [asyncio.CancelledError(), GeneratorExit()], ids=["CancelledError", "GeneratorExit"])
def test_a_cancelled_task_is_cancelled(kind: str, exc: BaseException) -> None:
    """A hang-up or barge-in cancels the task running the call: never an error, whatever the result."""
    assert tool_outcome(kind, {"error": "UNAVAILABLE"}, exception=exc) == "cancelled"
    assert tool_outcome(kind, None, exception=exc) == "cancelled"


def test_strands_cancelled_gateway_tool_is_cancelled() -> None:
    """Strands' ``cancel_tool`` result (status error, a message) and an MCP call cancelled locally."""
    cancel_result = {"toolUseId": "t1", "status": "error", "content": [{"text": "tool cancelled by user"}]}
    assert tool_outcome("gateway", cancel_result, cancel_message="tool cancelled by user") == "cancelled"
    mcp = {"toolUseId": "t1", "status": "error", "cancelled": True,
           "content": [{"text": "Tool execution cancelled locally; remote execution may have continued"}]}
    assert tool_outcome("gateway", mcp) == "cancelled"


# --------------------------------------------------------------------------- gateway


def test_gateway_success_and_error_follow_the_status() -> None:
    assert tool_outcome("gateway", {"toolUseId": "t", "status": "success", "content": [{"text": "ok"}]}) == "success"
    assert tool_outcome("gateway", {"toolUseId": "t", "status": "error", "content": [{"text": "x"}]}) == "error"


@pytest.mark.parametrize("result", [None, {}, {"status": "weird"}, "text"])
def test_a_gateway_result_of_no_known_shape_is_an_error(result: Any) -> None:
    assert tool_outcome("gateway", result) == "error"


def test_a_gateway_exception_is_an_error() -> None:
    assert tool_outcome("gateway", {"status": "error"}, exception=ConnectionError("down")) == "error"


# --------------------------------------------------------------------------- Gateway answers


def mcp(answer: Any, *, structured: bool = False, status: str = "success") -> dict[str, Any]:
    """A Strands MCP result as the Gateway returns the Level-0 Lambda's answer: JSON text, and
    optionally ``structuredContent``."""
    out: dict[str, Any] = {"toolUseId": "t1", "status": status,
                           "content": [{"text": answer if isinstance(answer, str) else json.dumps(answer)}]}
    if structured:
        out["structuredContent"] = answer
    return out


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("code", ["UNAVAILABLE", "INVALID_ARGUMENT", "UNKNOWN_TOOL", "UNKNOWN_OPERATION",
                                  "SOMETHING_NEW", "INVALID_ARGUMENTS", "cancelled"])
def test_a_gateway_answer_with_an_error_code_is_an_error(code: str, structured: bool) -> None:
    """The Level-0 Lambda never raises; it answers ``{"error": CODE}`` in a successful MCP result.
    ``UNKNOWN_TOOL`` there means our Gateway wiring is broken: an error, not a bridge rejection."""
    assert tool_outcome("gateway", mcp({"error": code}, structured=structured)) == "error"


@pytest.mark.parametrize("code", ["NOT_FOUND", "INSUFFICIENT_FUNDS", "AUTH_LEVEL_INSUFFICIENT"])
def test_a_gateway_business_answer_is_refused(code: str) -> None:
    assert tool_outcome("gateway", mcp({"error": code})) == "refused"


def test_a_gateway_error_code_wrapped_in_a_body_is_found() -> None:
    assert tool_outcome("gateway", mcp({"statusCode": 200, "body": json.dumps({"error": "UNAVAILABLE"})})) == "error"
    assert tool_outcome("gateway", mcp({"body": {"error": "UNAVAILABLE"}})) == "error"


def test_a_later_json_text_item_with_a_code_counts() -> None:
    r = {"toolUseId": "t", "status": "success",
         "content": [{"text": "Here you go."}, {"text": json.dumps({"error": "UNAVAILABLE"})}]}
    assert tool_outcome("gateway", r) == "error"


@pytest.mark.parametrize("text", ["Branches open 9 to 5.", "{not json", json.dumps({"answer": "x"}),
                                  json.dumps(["error"]), json.dumps("UNAVAILABLE"), ""])
def test_gateway_text_without_an_error_code_is_a_success(text: str) -> None:
    assert tool_outcome("gateway", mcp(text)) == "success"


def test_a_gateway_answer_with_a_non_string_code_is_an_error() -> None:
    assert tool_outcome("gateway", mcp({"error": {"code": "X"}})) == "error"
    assert tool_outcome("gateway", mcp({"error": None})) == "error"


# --------------------------------------------------------------------------- error_code


def test_error_code_is_a_known_code_or_other() -> None:
    assert tool_error_code("inline", {"error": "NOT_FOUND", "tool": "x"}) == "NOT_FOUND"
    assert tool_error_code("inline", {"error": "cancelled"}) == "cancelled"
    for code in KNOWN_ERROR_CODES:
        assert tool_error_code("inline", {"error": code}) == code
    assert tool_error_code("inline", {"error": "X" * 100}) == "other"
    assert tool_error_code("inline", {"error": "caller said 4111 1111 1111 1111"}) == "other"
    assert tool_error_code("inline", {"error": {"nested": "x"}}) == "other"
    assert tool_error_code("inline", {"error": None}) == "other"
    assert tool_error_code("inline", {"tool": "x", "data": {}}) is None
    assert tool_error_code("inline", None) is None


def test_error_code_of_a_gateway_answer() -> None:
    assert tool_error_code("gateway", mcp({"error": "UNAVAILABLE"})) == "UNAVAILABLE"
    assert tool_error_code("gateway", mcp({"error": "made up"})) == "other"
    assert tool_error_code("gateway", mcp("plain text")) is None


def test_known_error_codes() -> None:
    assert (BUSINESS_REFUSAL_CODES | BRIDGE_REJECTION_CODES
                                 | {"UNAVAILABLE", "INVALID_ARGUMENT", "UNKNOWN_OPERATION", "cancelled"}) == KNOWN_ERROR_CODES
