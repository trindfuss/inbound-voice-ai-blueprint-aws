"""Per-call sampling from the CallSid.

:func:`bucket` maps a CallSid to a number in [0, 1) from its SHA-256, so the front door and every
bridge session reach the same decision for a call without passing it between them, and re-entry
on another task agrees too. :class:`CallSampler` samples a root span when the ``call_sid`` in its
parent context's baggage falls below the rate; a span with no ``call_sid`` in baggage is dropped.
:func:`make_sampler` puts it under ``ParentBased``, so children follow their root.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from opentelemetry import baggage
from opentelemetry.context import Context
from opentelemetry.sdk.trace.sampling import Decision, ParentBased, Sampler, SamplingResult
from opentelemetry.trace import Link, SpanKind, get_current_span
from opentelemetry.trace.span import TraceState
from opentelemetry.util.types import Attributes

CALL_SID_KEY = "call_sid"
_SCALE = float(1 << 64)


def bucket(call_sid: str) -> float:
    """A stable number in [0, 1) for ``call_sid``: the first 8 bytes of its SHA-256."""
    digest = hashlib.sha256(call_sid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / _SCALE


class CallSampler(Sampler):
    """Samples by the baggage ``call_sid`` alone: ``bucket(call_sid) < rate``; none means drop."""

    def __init__(self, rate: float) -> None:
        self._rate = rate

    def should_sample(self, parent_context: Context | None, trace_id: int, name: str,
                      kind: SpanKind | None = None, attributes: Attributes = None,
                      links: Sequence[Link] | None = None,
                      trace_state: TraceState | None = None) -> SamplingResult:
        parent_state = get_current_span(parent_context).get_span_context().trace_state
        call_sid = baggage.get_baggage(CALL_SID_KEY, parent_context)
        if not isinstance(call_sid, str) or not call_sid or bucket(call_sid) >= self._rate:
            return SamplingResult(Decision.DROP, None, parent_state)
        return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes, parent_state)

    def get_description(self) -> str:
        return f"CallSampler{{{self._rate}}}"


def make_sampler(rate: float) -> Sampler:
    """The sampler both processes use: children follow the parent, roots go by the CallSid."""
    return ParentBased(root=CallSampler(rate))
