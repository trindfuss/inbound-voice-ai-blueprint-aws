"""The guarded model: a first-token deadline and a history filter around one entry's model.

Model failover. One :class:`GuardedModel` wraps
one list entry's Strands model (a ``BedrockModel``) and does two things in :meth:`GuardedModel.stream`:

* **The first-token deadline.** Until the first output (a text delta or a tool use, see
  :func:`is_output`) the wrapper holds the stream's events back. If no output comes within the entry's
  ``deadline_s``, or the model fails before any output with a throttle, timeout, model or connection
  error, or the model is unavailable, the call is abandoned and :class:`~owlbank.failover.ModelUnusable`
  is raised so the client moves the turn to the next entry. Nothing has reached Strands by then, so
  nothing is spoken and nothing is added to history. Once output has started the stream passes through
  untouched: no deadline, and errors go the existing way (no mid-stream failover). The last entry has
  no deadline. A request error is raised unchanged (it is never a failover), and a turn the
  agent is cancelling (barge-in, hang-up) is never a failover either.
* **The history filter.** ``invocation_state["reasoning_producers"]`` (set by the client before
  every call) maps a reasoning block's key (:func:`reasoning_key`, a digest of its signature or
  redacted content) to the entry whose model produced it. A ``reasoningContent`` block is sent only
  to the entry that produced it: one produced by another entry, or of unknown origin (a history
  rebuilt from Memory, a block with neither signature nor redacted content), is left out, on every
  call, not only the first after a switch, since its signature belongs to another model. Without the
  key every reasoning block is left out (the safe side). When nothing is left out the
  request is exactly the one Strands built, so the healthy path and its prompt cache are unchanged. The stored
  history is never touched. (The record is keyed by the block, not the message:
  Strands sends the model a deep copy of the history without the messages' ``tracking_id``.)

The per-call cancel signal: the inner model gets its own ``threading.Event`` per call,
which also reads as set when the agent's signal is set. A missed deadline sets only the per-call one,
so the Bedrock reader thread stops at its next chunk; the agent's signal is never set (that would end
the call as "Cancelled by user").

Timing never relies on a timer firing at or after its due time (uvloop's can fire early): every wait
is followed by an explicit comparison against the wrapper's clock, and a wait that wakes early waits
again for the rest.

**Test delay.** ``OWLBANK_FAILOVER_TEST`` (seconds, set by infra only when on, at most 30) holds back
the first output of the preferred entry (index 0) by that long from the call's start, as a slow model
would, still under the deadline. Synth is the real guard (it never sets it at prod); the bridge also refuses
it at ``stage=prod`` (:func:`failover_test_delay`), but its stage is known only when ``OWLBANK_STAGE`` is set
(call observability on).

Nothing here logs; no caller text, message or digit is kept on an exception.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import math
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from typing import Any, Final

from strands.models.model import Model
from strands.types.content import ContentBlock, Message, Messages
from strands.types.exceptions import ModelThrottledException
from strands.types.streaming import StreamEvent

from owlbank.failover.classify import ModelUnusable, Outcome, classify

#: The ``invocation_state`` key with the entry that produced each reasoning block, by
#: :func:`reasoning_key`: a reasoning block is sent only to the entry that produced it.
PRODUCERS_KEY: Final = "reasoning_producers"
#: The error code of a Strands ``ModelThrottledException`` (Strands' wrapper of Bedrock's throttle),
#: so a throttle keeps the code it has without failover (``to_agent_error``).
THROTTLING_CODE: Final = "ThrottlingException"
#: The test-only first-token delay of the preferred entry, in seconds.
FAILOVER_TEST_ENV: Final = "OWLBANK_FAILOVER_TEST"
MAX_TEST_DELAY_S: Final = 30.0
#: The stage at which the test delay is refused.
PROD_STAGE: Final = "prod"


class TestDelayError(ValueError):
    """The test delay is unusable or not allowed here; names the setting, never its value."""

    __test__ = False  # not a pytest test class

    def __init__(self, reason: str) -> None:
        super().__init__(f"{FAILOVER_TEST_ENV}: {reason}")


def check_test_delay(delay_s: float | None, *, stage: str) -> float | None:
    """``delay_s`` if it is a usable test delay at ``stage``; ``None`` stays ``None``."""
    if delay_s is None:
        return None
    if stage == PROD_STAGE:
        raise TestDelayError("refused at stage=prod")
    if (isinstance(delay_s, bool) or not isinstance(delay_s, int | float) or not math.isfinite(delay_s)
            or not 0 < delay_s <= MAX_TEST_DELAY_S):
        raise TestDelayError(f"must be seconds above 0 and at most {MAX_TEST_DELAY_S:g}")
    return float(delay_s)


def failover_test_delay(env: Mapping[str, str], *, stage: str) -> float | None:
    """The test delay from the environment: ``None`` when unset or empty; refused at ``stage=prod``."""
    raw = (env.get(FAILOVER_TEST_ENV) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise TestDelayError("must be a number of seconds") from None
    return check_test_delay(value, stage=stage)


def is_output(event: Mapping[str, Any]) -> bool:
    """Whether a stream event is output in the sense of 3.3: a text delta or a tool use (its start or
    its input). Reasoning, message start and stop, block stop and metadata are not."""
    delta = event.get("contentBlockDelta")
    if isinstance(delta, Mapping):
        inner = delta.get("delta")
        return isinstance(inner, Mapping) and ("text" in inner or "toolUse" in inner)
    start = event.get("contentBlockStart")
    if isinstance(start, Mapping):
        inner = start.get("start")
        return isinstance(inner, Mapping) and "toolUse" in inner
    return False


def strip_reasoning(messages: Messages) -> Messages:
    """``messages`` without any ``reasoningContent`` block, every other block kept in order.

    New message dicts and content lists are built; the input and its blocks are never changed. A
    message left empty is kept (``BedrockModel`` leaves empty messages out of the request itself), so
    roles and their order are unchanged.
    """
    out: Messages = []
    for message in messages:
        copied = dict(message)
        copied["content"] = [block for block in message["content"] if "reasoningContent" not in block]
        out.append(copied)  # type: ignore[arg-type]
    return out


def reasoning_key(block: ContentBlock | Mapping[str, Any]) -> str | None:
    """The key of a ``reasoningContent`` block (a SHA-256 digest of its signature, else of its redacted
    content), or ``None`` for any other block or a reasoning block with neither. Only a digest is
    kept, never the reasoning itself."""
    reasoning = block.get("reasoningContent")
    if not isinstance(reasoning, Mapping):
        return None
    text = reasoning.get("reasoningText")
    signature = text.get("signature") if isinstance(text, Mapping) else None
    if isinstance(signature, str) and signature:
        return "sig:" + hashlib.sha256(signature.encode()).hexdigest()
    redacted = reasoning.get("redactedContent")
    if isinstance(redacted, bytes | bytearray) and redacted:
        return "red:" + hashlib.sha256(redacted).hexdigest()
    return None


def _foreign(block: ContentBlock, producers: Mapping[str, int] | None, entry: int) -> bool:
    """A reasoning block ``entry`` did not produce (or whose producer is unknown)."""
    if "reasoningContent" not in block:
        return False
    key = reasoning_key(block)
    return producers is None or key is None or producers.get(key) != entry


def foreign_reasoning_stripped(messages: Messages, producers: Mapping[str, int] | None, entry: int) -> Messages:
    """``messages`` without the ``reasoningContent`` blocks ``entry`` did not produce: a reasoning
    block is kept only if ``producers`` names ``entry`` for its :func:`reasoning_key`.

    ``messages`` itself when nothing is left out (the request Strands built, prompt cache intact);
    otherwise a new list in which each changed message is a new dict with a new content list, and
    every other message is the original. The input is never changed; a message left empty is kept
    (see :func:`strip_reasoning`).
    """
    if not any(_foreign(b, producers, entry) for m in messages for b in m["content"]):
        return messages
    out: Messages = []
    for message in messages:
        if any(_foreign(b, producers, entry) for b in message["content"]):
            copied = dict(message)
            copied["content"] = [b for b in message["content"] if not _foreign(b, producers, entry)]
            out.append(copied)  # type: ignore[arg-type]
        else:
            out.append(message)
    return out


def record_reasoning(producers: dict[str, int], message: Message | Mapping[str, Any], entry: int) -> None:
    """Record ``entry`` as the producer of every keyed reasoning block of ``message``."""
    for block in message.get("content") or ():
        key = reasoning_key(block) if isinstance(block, Mapping) else None
        if key is not None:
            producers[key] = entry


def _error_code(exc: BaseException) -> tuple[str, str | None]:
    """The error's code (a boto ``Error.Code``; Strands' throttle wrapper as :data:`THROTTLING_CODE`;
    else the exception's class name) and message."""
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error")
        if isinstance(error, Mapping) and error.get("Code"):
            message = error.get("Message")
            return str(error["Code"]), message if isinstance(message, str) else None
    if isinstance(exc, ModelThrottledException):
        return THROTTLING_CODE, None
    return type(exc).__name__, None


def _consume(future: asyncio.Future[Any]) -> None:
    """Retrieve an abandoned read's outcome so the loop never reports it as unretrieved."""
    with contextlib.suppress(BaseException):
        future.result()


class _CallSignal(threading.Event):
    """A per-call cancel signal that also reads as set when the agent's own signal is set."""

    def __init__(self, parent: threading.Event | None) -> None:
        super().__init__()
        self._parent = parent

    def is_set(self) -> bool:
        return super().is_set() or (self._parent is not None and self._parent.is_set())


class GuardedModel(Model):
    """One list entry's model with the first-token deadline and the history filter.

    ``entry`` is the entry's index in the list (carried on :class:`ModelUnusable`); ``deadline_s`` is
    its first-token deadline, ``None`` for the last entry; ``test_delay_s`` applies to entry 0
    only and is refused at ``stage=prod``. ``clock`` is a monotonic clock in seconds. One instance can
    serve any number of calls at once: it keeps no per-call state.
    """

    def __init__(self, inner: Model, *, entry: int, deadline_s: float | None, stage: str,
                 test_delay_s: float | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        if isinstance(entry, bool) or not isinstance(entry, int) or entry < 0:
            raise ValueError("entry must be a list index")
        if deadline_s is not None and (isinstance(deadline_s, bool) or not isinstance(deadline_s, int | float)
                                       or not math.isfinite(deadline_s) or deadline_s <= 0):
            raise ValueError("deadline_s must be positive seconds or None")
        delay = check_test_delay(test_delay_s, stage=stage)
        self._inner = inner
        self._entry = entry
        self._deadline_s = None if deadline_s is None else float(deadline_s)
        self._test_delay_s = delay if entry == 0 else None
        self._clock = clock

    # ------------------------------------------------------------------------------- delegation

    @property
    def inner(self) -> Model:
        return self._inner

    @property
    def entry(self) -> int:
        return self._entry

    @property
    def deadline_s(self) -> float | None:
        return self._deadline_s

    @property
    def test_delay_s(self) -> float | None:
        return self._test_delay_s

    @property
    def stateful(self) -> bool:
        return self._inner.stateful

    @property
    def context_window_limit(self) -> int | None:
        return self._inner.context_window_limit

    def __getattr__(self, name: str) -> Any:
        # Anything else Strands reads off a model (``config`` for the model ID in traces) is the
        # inner model's. Only called for attributes not found on the wrapper.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def update_config(self, **model_config: Any) -> None:
        self._inner.update_config(**model_config)

    def get_config(self) -> Any:
        return self._inner.get_config()

    def structured_output(self, output_model: Any, prompt: Messages, system_prompt: str | None = None,
                          **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        return self._inner.structured_output(output_model, prompt, system_prompt, **kwargs)

    async def count_tokens(self, messages: Messages, tool_specs: Any = None, system_prompt: str | None = None,
                           system_prompt_content: Any = None) -> int:
        return await self._inner.count_tokens(messages, tool_specs, system_prompt, system_prompt_content)

    # ----------------------------------------------------------------------------------- stream

    async def stream(
        self,
        messages: Messages,
        tool_specs: Any = None,
        system_prompt: str | None = None,
        *,
        tool_choice: Any = None,
        system_prompt_content: Any = None,
        invocation_state: dict[str, Any] | None = None,
        cancel_signal: threading.Event | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        producers = invocation_state.get(PRODUCERS_KEY) if invocation_state else None
        request = foreign_reasoning_stripped(messages, producers if isinstance(producers, Mapping) else None,
                                             self._entry)
        signal = _CallSignal(cancel_signal)
        start = self._clock()
        due = None if self._deadline_s is None else start + self._deadline_s
        release_at = None if self._test_delay_s is None else start + self._test_delay_s

        stream = self._inner.stream(request, tool_specs, system_prompt, tool_choice=tool_choice,
                                    system_prompt_content=system_prompt_content,
                                    invocation_state=invocation_state, cancel_signal=signal, **kwargs)
        it: AsyncIterator[StreamEvent] = stream.__aiter__()
        pending: asyncio.Future[StreamEvent] | None = None
        finished = False
        held: list[StreamEvent] = []
        failure: ModelUnusable | None = None
        try:
            # Before the first output: hold events back, watch the deadline and pre-output errors.
            while True:
                if pending is None:
                    pending = asyncio.ensure_future(it.__anext__())
                if not pending.done():
                    remaining = None if due is None else due - self._clock()
                    if remaining is not None and remaining <= 0:
                        if cancel_signal is not None and cancel_signal.is_set():
                            return  # a cancelled turn, not a failover: Strands reports it
                        failure = ModelUnusable(self._entry, Outcome.ABANDONED)
                        break
                    # Woken early or on time: the loop compares against the clock again.
                    await asyncio.wait({pending}, timeout=remaining)
                    continue
                done, pending = pending, None
                try:
                    event = done.result()
                except StopAsyncIteration:
                    finished = True  # ended without output (or cancelled): pass on what came
                    for held_event in held:
                        yield held_event
                    return
                except Exception as exc:  # noqa: BLE001 - classified below, never swallowed silently
                    failure = self._pre_output_failure(exc, cancel_signal)
                    if failure is None:
                        raise
                    break
                held.append(event)
                if is_output(event):
                    break

            if failure is None and release_at is not None:
                # The test delay: the first output is not released before ``release_at``, still
                # under the deadline, as if the model were that slow.
                while (now := self._clock()) < release_at:
                    if due is not None and now >= due:
                        if cancel_signal is not None and cancel_signal.is_set():
                            return
                        failure = ModelUnusable(self._entry, Outcome.ABANDONED)
                        break
                    await asyncio.sleep(min(release_at, due if due is not None else release_at) - now)

            if failure is not None:
                signal.set()
            else:
                # Output has started: everything passes through, no deadline, errors unchanged.
                for held_event in held:
                    yield held_event
                held.clear()
                async for event in it:
                    yield event
                finished = True
        finally:
            if not finished:
                signal.set()  # the reader thread stops at its next chunk
            try:
                if pending is not None and not pending.done():
                    pending.cancel()
                    pending.add_done_callback(_consume)
                    # Only the pending read's own outcome is ignored: a cancellation of this task
                    # (barge-in, the turn's timeout) raised by the wait propagates, and the missed
                    # deadline below is then never raised, so the turn is not reissued.
                    await asyncio.wait({pending})
            finally:
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    with contextlib.suppress(Exception):
                        await aclose()
        if failure is not None:
            # Raised outside any ``except`` so no provider message is chained onto it.
            raise failure

    def _pre_output_failure(self, exc: Exception, cancel_signal: threading.Event | None) -> ModelUnusable | None:
        """The :class:`ModelUnusable` for an error before any output, or ``None`` to raise it unchanged
        (a request error, or a turn being cancelled)."""
        if cancel_signal is not None and cancel_signal.is_set():
            return None
        code, message = _error_code(exc)
        outcome = classify(code, message)
        if outcome in (Outcome.REQUEST_ERROR, Outcome.OK):
            return None
        return ModelUnusable(self._entry, outcome, error_code=code)
