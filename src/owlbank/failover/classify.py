"""What a model call's outcome means for its model.

:func:`classify` maps an error's code (the boto ``Error.Code``, an event-stream ``:exception-type`` or
an exception class name) and its message to an :class:`Outcome` or :data:`UNAVAILABLE`. It reads the
code and, for ``ValidationException`` only, an exact message or a fixed prefix (both live texts); no other
part of a message is looked at,
and the message is never kept.

:class:`ModelUnusable` is what the guarded model raises before any output (a missed first-token
deadline, a hard error, an unavailable model) so the client moves the turn to the next entry.
"""

from __future__ import annotations

import enum
from typing import Final, Literal


class Outcome(enum.StrEnum):
    """One model call's outcome."""

    OK = "ok"
    THROTTLED = "throttled"
    TIMEOUT = "timeout"
    ERROR = "error"
    ABANDONED = "abandoned"  # no first token within the deadline
    REQUEST_ERROR = "request_error"  # the request's fault, never the model's


#: The model itself can't be used; one such error is enough.
UNAVAILABLE: Final[Literal["unavailable"]] = "unavailable"

type Classification = Outcome | Literal["unavailable"]

#: ``ValidationException`` with exactly this message: an unknown or malformed model ID.
INVALID_MODEL_MESSAGE: Final = "The provided model identifier is invalid."
#: ``ValidationException`` messages starting so are about the request.
REQUEST_ERROR_PREFIX: Final = "The model returned the following errors:"

_UNAVAILABLE_CODES: Final = frozenset({"ResourceNotFoundException", "AccessDeniedException"})
#: Request errors other than ``ValidationException``: Strands' wrapper of a too-long input.
_REQUEST_CODES: Final = frozenset({"ContextWindowOverflowException"})
_HARD: Final[dict[str, Outcome]] = {
    "ThrottlingException": Outcome.THROTTLED,
    "ModelThrottledException": Outcome.THROTTLED,  # Strands' wrapper of a throttle
    "ServiceUnavailableException": Outcome.ERROR,
    "ModelTimeoutException": Outcome.TIMEOUT,
    "ModelErrorException": Outcome.ERROR,
    "ModelStreamErrorException": Outcome.ERROR,
    "ModelNotReadyException": Outcome.ERROR,
    "InternalServerException": Outcome.ERROR,
    # botocore connection errors (no ``Error.Code``; the class name is the code)
    "EndpointConnectionError": Outcome.ERROR,
    "ConnectionClosedError": Outcome.ERROR,
    "ConnectionError": Outcome.ERROR,
    "ProxyConnectionError": Outcome.ERROR,
    "ReadTimeoutError": Outcome.TIMEOUT,
    "ConnectTimeoutError": Outcome.TIMEOUT,
}


def _normalise(code: str | None) -> str:
    """Event-stream errors name their type in lower camel case (``throttlingException``)."""
    code = (code or "").strip()
    return code[:1].upper() + code[1:]


def classify(error_code: str | None, message: str | None) -> Classification:
    """The class of one failed model call. Never raises.

    * ``ResourceNotFoundException``, ``AccessDeniedException``, or ``ValidationException`` with
      exactly :data:`INVALID_MODEL_MESSAGE`: :data:`UNAVAILABLE`.
    * Any other ``ValidationException`` (those starting :data:`REQUEST_ERROR_PREFIX` among them) and
      a context-window overflow: :attr:`Outcome.REQUEST_ERROR`.
    * Throttles, timeouts, model and connection errors: the hard-error outcome.
    * Anything else (no code, an unknown code): :attr:`Outcome.ERROR`, a hard error, so a turn that
      can't be answered before any output still moves on.
    """
    code = _normalise(error_code)
    if code in _UNAVAILABLE_CODES:
        return UNAVAILABLE
    if code == "ValidationException":
        text = (message or "").strip()
        if text == INVALID_MODEL_MESSAGE:
            return UNAVAILABLE
        # Prefix or not, a validation error other than the unknown model is the request's.
        return Outcome.REQUEST_ERROR
    if code in _REQUEST_CODES:
        return Outcome.REQUEST_ERROR
    return _HARD.get(code, Outcome.ERROR)


#: The causes that move a turn to the next entry.
_FAILOVER_CAUSES: Final[frozenset[str]] = frozenset(
    {Outcome.ABANDONED, Outcome.THROTTLED, Outcome.TIMEOUT, Outcome.ERROR, UNAVAILABLE})


class ModelUnusable(Exception):
    """A model call that produced no output and must go to the next entry.

    ``entry`` is the entry's index in the list; ``cause`` is :attr:`Outcome.ABANDONED` for a missed
    first-token deadline, a hard-error outcome, or :data:`UNAVAILABLE`; ``error_code`` is the error's
    code when there was one. The error's message is never kept (it may echo the request).
    """

    def __init__(self, entry: int, cause: Classification, *, error_code: str | None = None) -> None:
        if not isinstance(entry, int) or isinstance(entry, bool) or entry < 0:
            raise ValueError("entry must be a list index")
        if cause not in _FAILOVER_CAUSES:
            raise ValueError("cause is no reason to fail over")
        text = "deadline" if cause == Outcome.ABANDONED else str(cause)
        super().__init__(f"model entry {entry} unusable: {text}" + (f" ({error_code})" if error_code else ""))
        self.entry = entry
        self.cause: Classification = cause
        self.error_code = error_code

    @property
    def is_deadline(self) -> bool:
        return self.cause == Outcome.ABANDONED
