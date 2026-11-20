"""Only actual failures are tool errors.

For any tool result or exception, ``tool_outcome`` is ``error`` only for a failure of the tool or a
dependency; a cancellation (hang-up, barge-in) is never ``error`` and never counted in the total;
the alarm's error rate (``IF(calls > 0, 100 * errors / calls, 0)`` over ``ToolCalls`` = outcome
not ``cancelled`` and ``ToolErrors`` = outcome ``error``) is 0 for any sequence of calls without
an ``error`` outcome.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from owlbank.tools_contract import (
    BRIDGE_REJECTION_CODES,
    BUSINESS_REFUSAL_CODES,
    CANCELLED_CODE,
    TOOL_OUTCOMES,
    tool_outcome,
)

FAILURE_CODES = ("UNAVAILABLE", "INVALID_ARGUMENT", "UNKNOWN_OPERATION")
NON_FAILURE_CODES = BUSINESS_REFUSAL_CODES | BRIDGE_REJECTION_CODES | {CANCELLED_CODE}

kinds = st.sampled_from(["inline", "gateway"])
json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=8),
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=6), inner, max_size=3),
    max_leaves=8)
codes = st.sampled_from(sorted(NON_FAILURE_CODES) + list(FAILURE_CODES)) | st.text(max_size=12) | json_values
inline_results = st.builds(lambda code, extra: {**extra, "error": code},
                           codes, st.dictionaries(st.text(max_size=6), json_values, max_size=3)) \
    | st.dictionaries(st.text(max_size=6).filter(lambda k: k not in ("error", "cancelled")), json_values,
                      max_size=3)
#: What the Level-0 Lambda answers (never raises): data, or ``{"error": CODE}``, optionally in a
#: Lambda-proxy ``body``. The Gateway hands it back as JSON text in a successful MCP result.
lambda_answers = st.builds(lambda code, extra: {**extra, "error": code},
                           codes, st.dictionaries(st.sampled_from(["tool", "detail"]), json_values, max_size=2)) \
    | st.dictionaries(st.sampled_from(["answer", "hours", "tool"]), json_values, max_size=2)
wrapped_answers = lambda_answers | lambda_answers.map(lambda a: {"statusCode": 200, "body": json.dumps(a)})
text_items = st.text(max_size=10) | wrapped_answers.map(json.dumps)
gateway_results = st.fixed_dictionaries(
    {"toolUseId": st.text(max_size=6), "status": st.sampled_from(["success", "error"]) | st.text(max_size=6),
     "content": st.lists(st.fixed_dictionaries({"text": text_items}), max_size=2)},
    optional={"cancelled": st.booleans(), "structuredContent": wrapped_answers})
any_results = inline_results | gateway_results | json_values
cancellations = st.sampled_from([asyncio.CancelledError, GeneratorExit]).map(lambda c: c())
failures = st.sampled_from([RuntimeError, TimeoutError, ConnectionError, ValueError, KeyError]).map(lambda c: c())


def _code_in(answer: Any) -> tuple[bool, Any]:
    """(has a code, the code) for one Lambda answer, as JSON text or an object, maybe in ``body``."""
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except ValueError:
            return False, None
    if isinstance(answer, dict) and "error" in answer:
        return True, answer["error"]
    if isinstance(answer, dict) and "body" in answer:
        return _code_in(answer["body"])
    return False, None


def _is_failure(kind: str, result: Any) -> bool:
    """The spec's failures, written independently of the implementation."""
    if not isinstance(result, dict):
        return True
    if result.get("cancelled") is True:
        return False
    if kind == "gateway":
        if result.get("status") != "success":
            return True
        answers = [result["structuredContent"]] if "structuredContent" in result else []
        answers += [c["text"] for c in result.get("content") or []]
        for answer in answers:
            found, code = _code_in(answer)
            if found:  # the first answer with a code decides: only a business answer is no failure
                return not (isinstance(code, str) and code in BUSINESS_REFUSAL_CODES)
        return False
    if "error" not in result:
        return False
    code = result["error"]
    return not isinstance(code, str) or code not in NON_FAILURE_CODES


def _rate(outcomes: list[str]) -> float:
    calls = sum(1 for o in outcomes if o != "cancelled")
    errors = sum(1 for o in outcomes if o == "error")
    return 100 * errors / calls if calls > 0 else 0


@given(kinds, any_results)
def test_error_only_for_failures(kind: str, result: Any) -> None:
    outcome = tool_outcome(kind, result)
    assert outcome in TOOL_OUTCOMES
    assert (outcome == "error") == _is_failure(kind, result)


@given(gateway_results)
def test_gateway_error_only_for_failures(result: dict[str, Any]) -> None:
    """Gateway results on their own, so answers carrying a code inside a successful MCP result are
    generated often (review of b0e25a3: the Level-0 Lambda never raises)."""
    assert (tool_outcome("gateway", result) == "error") == _is_failure("gateway", result)


@given(kinds, any_results, st.one_of(st.none(), st.text(max_size=10)))
def test_a_cancellation_is_never_an_error(kind: str, result: Any, message: str | None) -> None:
    for exc in (asyncio.CancelledError(), GeneratorExit()):
        assert tool_outcome(kind, result, exception=exc, cancel_message=message) == "cancelled"
    if message is not None:
        assert tool_outcome(kind, result, cancel_message=message) == "cancelled"


@given(kinds, any_results)
def test_the_synthesized_cancelled_answer_and_mcp_cancel_are_cancelled(kind: str, result: Any) -> None:
    assert tool_outcome("inline", {"error": CANCELLED_CODE}) == "cancelled"
    base = result if isinstance(result, dict) else {}
    assert tool_outcome(kind, {**base, "cancelled": True}) == "cancelled"


@given(kinds, any_results, failures)
def test_any_other_exception_is_an_error(kind: str, result: Any, exc: BaseException) -> None:
    assert tool_outcome(kind, result, exception=exc) == "error"


calls = st.tuples(kinds, any_results, st.one_of(st.none(), cancellations, failures))


@given(st.lists(calls, max_size=30))
def test_rate_is_zero_without_errors_and_cancelled_is_never_in_the_total(seq: list[Any]) -> None:
    outcomes = [tool_outcome(k, r, exception=e) for k, r, e in seq]
    rate = _rate(outcomes)
    if "error" not in outcomes:
        assert rate == 0
    # Dropping every cancelled call changes neither count, so not the rate either.
    assert _rate([o for o in outcomes if o != "cancelled"]) == rate
    # Adding cancelled calls never changes it.
    assert _rate(outcomes + ["cancelled"] * 5) == rate
    assert 0 <= rate <= 100
