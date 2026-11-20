""":func:`classify` over every row, and :class:`ModelUnusable`.

* ``ValidationException`` with the exact message ``The provided model identifier is invalid.`` is
  unavailable; ``ResourceNotFoundException`` and ``AccessDeniedException`` are unavailable whatever
  the message (4a.1: one is enough).
* ``ValidationException`` with a message starting ``The model returned the following errors:`` is a
  request error, and so is any other ``ValidationException``: only the exact model-identifier
  message makes a validation error the model's fault.
* Throttles, timeouts, model errors and connection errors are hard errors.
* Event-stream error codes arrive in lower camel case (``throttlingException``); they classify the same.
* :class:`ModelUnusable` is an ``Exception`` (so Strands fires ``AfterModelCallEvent``), carries the
  entry, cause and error code, and never the error message.
"""

from __future__ import annotations

import pytest

from owlbank.failover.classify import (
    INVALID_MODEL_MESSAGE,
    REQUEST_ERROR_PREFIX,
    UNAVAILABLE,
    ModelUnusable,
    Outcome,
    classify,
)


def test_message_constants_are_the_live_texts() -> None:
    assert INVALID_MODEL_MESSAGE == "The provided model identifier is invalid."
    assert REQUEST_ERROR_PREFIX == "The model returned the following errors:"


# --------------------------------------------------------------------------- unavailable


def test_invalid_model_identifier_is_unavailable() -> None:
    assert classify("ValidationException", "The provided model identifier is invalid.") == UNAVAILABLE


@pytest.mark.parametrize("code", ["ResourceNotFoundException", "AccessDeniedException"])
@pytest.mark.parametrize("message", [None, "", "Model not found", "You don't have access to the model",
                                     "The model returned the following errors: x"])
def test_not_found_and_access_denied_are_unavailable(code: str, message: str | None) -> None:
    assert classify(code, message) == UNAVAILABLE


def test_invalid_identifier_message_only_counts_when_exact() -> None:
    assert classify("ValidationException", "The provided model identifier is invalid. Try again") is (
        Outcome.REQUEST_ERROR)
    assert classify("ValidationException", "the provided model identifier is invalid.") is Outcome.REQUEST_ERROR
    # Surrounding whitespace is not free text.
    assert classify("ValidationException", "  The provided model identifier is invalid.\n") == UNAVAILABLE


def test_invalid_identifier_message_with_another_code_is_not_unavailable() -> None:
    assert classify("ThrottlingException", INVALID_MODEL_MESSAGE) is Outcome.THROTTLED


# --------------------------------------------------------------------------- request errors


@pytest.mark.parametrize("message", [
    "The model returned the following errors: messages.1.content.0: Invalid signature in thinking block",
    "The model returned the following errors: adaptive thinking is not supported on this model",
    "The model returned the following errors:",
])
def test_model_returned_errors_prefix_is_a_request_error(message: str) -> None:
    assert classify("ValidationException", message) is Outcome.REQUEST_ERROR


@pytest.mark.parametrize("message", [None, "", "Malformed input request", "A conversation must start with a user message."])
def test_other_validation_errors_are_request_errors(message: str | None) -> None:
    assert classify("ValidationException", message) is Outcome.REQUEST_ERROR


def test_context_window_overflow_is_a_request_error() -> None:
    assert classify("ContextWindowOverflowException", "too long") is Outcome.REQUEST_ERROR


# --------------------------------------------------------------------------- hard errors


@pytest.mark.parametrize(("code", "outcome"), [
    ("ThrottlingException", Outcome.THROTTLED),
    ("ModelThrottledException", Outcome.THROTTLED),  # Strands' wrapper of a throttle
    ("ServiceUnavailableException", Outcome.ERROR),
    ("ModelTimeoutException", Outcome.TIMEOUT),
    ("ModelErrorException", Outcome.ERROR),
    ("ModelStreamErrorException", Outcome.ERROR),
    ("ModelNotReadyException", Outcome.ERROR),
    ("InternalServerException", Outcome.ERROR),
    ("EndpointConnectionError", Outcome.ERROR),
    ("ConnectionClosedError", Outcome.ERROR),
    ("ConnectionError", Outcome.ERROR),
    ("ProxyConnectionError", Outcome.ERROR),
    ("ReadTimeoutError", Outcome.TIMEOUT),
    ("ConnectTimeoutError", Outcome.TIMEOUT),
])
def test_hard_errors(code: str, outcome: Outcome) -> None:
    assert classify(code, "anything at all") is outcome
    assert classify(code, None) is outcome


@pytest.mark.parametrize(("code", "expected"), [
    ("throttlingException", Outcome.THROTTLED),
    ("modelStreamErrorException", Outcome.ERROR),
    ("serviceUnavailableException", Outcome.ERROR),
    ("validationException", Outcome.REQUEST_ERROR),
    ("resourceNotFoundException", UNAVAILABLE),
])
def test_event_stream_codes_in_lower_camel_case(code: str, expected: object) -> None:
    assert classify(code, "x") == expected


@pytest.mark.parametrize("code", [None, "", "SomethingNew", "TypeError"])
def test_unknown_codes_are_hard_errors(code: str | None) -> None:
    assert classify(code, "x") is Outcome.ERROR


def test_classify_never_returns_ok_or_abandoned() -> None:
    codes = [None, "ValidationException", "ThrottlingException", "Other", "AccessDeniedException"]
    seen = {classify(c, m) for c in codes for m in (None, "", INVALID_MODEL_MESSAGE, REQUEST_ERROR_PREFIX)}
    assert Outcome.OK not in seen
    assert Outcome.ABANDONED not in seen


# --------------------------------------------------------------------------- ModelUnusable


def test_model_unusable_is_a_plain_exception() -> None:
    err = ModelUnusable(1, Outcome.THROTTLED, error_code="ThrottlingException")
    assert isinstance(err, Exception)
    assert (err.entry, err.cause, err.error_code) == (1, Outcome.THROTTLED, "ThrottlingException")
    assert not err.is_deadline


def test_model_unusable_for_a_missed_deadline() -> None:
    err = ModelUnusable(0, Outcome.ABANDONED)
    assert err.is_deadline
    assert err.error_code is None
    assert "deadline" in str(err)


def test_model_unusable_for_an_unavailable_model() -> None:
    err = ModelUnusable(2, UNAVAILABLE, error_code="ResourceNotFoundException")
    assert err.cause == UNAVAILABLE
    assert "2" in str(err)


@pytest.mark.parametrize("cause", [Outcome.OK, Outcome.REQUEST_ERROR, "slow", ""])
def test_model_unusable_refuses_a_cause_that_is_no_reason_to_fail_over(cause: object) -> None:
    with pytest.raises(ValueError, match="cause"):
        ModelUnusable(0, cause)  # type: ignore[arg-type]


def test_model_unusable_refuses_a_negative_entry() -> None:
    with pytest.raises(ValueError, match="entry"):
        ModelUnusable(-1, Outcome.ERROR)


def test_model_unusable_text_names_no_message() -> None:
    err = ModelUnusable(0, Outcome.ERROR, error_code="ModelErrorException")
    assert str(err) == "model entry 0 unusable: error (ModelErrorException)"
