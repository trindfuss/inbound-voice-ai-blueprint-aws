"""The span shape AgentCore Evaluations reads.

Testing found two gaps in multi-step calls:

* a turn that resumes from a tool result (the front door's pending result) has an
  ``invoke_agent`` span with no ``gen_ai.user.message`` event, so Evaluations can't parse the
  user query. :class:`EvalShapeProcessor` adds one **at span start**, while the span is still
  mutable, from :data:`HOP`, which the bridge sets per hop: the caller's utterance when the front
  door's result has one, else ``[call continues after <step>]``;
* an inline tool pauses the turn with a Strands interrupt, so its ``execute_tool`` span ends with
  no result and no status. :func:`fill_interrupted_tool`, called by the content filter **at
  export**, returns a copy with a placeholder ``gen_ai.choice`` event, ``gen_ai.tool.status`` and
  ``owlbank.result_next_turn=true``; the real result is on the next turn's ``chat`` span.

Both events go through the content filter afterwards like any other (with ``off`` only their names
and IDs leave). Neither function raises: an error leaves the span as it was.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass

from opentelemetry.context import Context
from opentelemetry.sdk.trace import Event, ReadableSpan, Span, SpanProcessor

INVOKE_AGENT = "invoke_agent"
EXECUTE_TOOL = "execute_tool"
USER_MESSAGE = "gen_ai.user.message"
CHOICE = "gen_ai.choice"
RESULT_NEXT_TURN = "owlbank.result_next_turn"


@dataclass(frozen=True)
class HopInfo:
    """What the bridge knows about the current hop (one agent invocation)."""

    caller_words: bool  # the hop starts from the caller's words, not from a tool result
    caller_utterance: str | None = None  # the front door's ``callerUtterance``, when it has one
    step: str | None = None  # the step the call resumes after, for the placeholder


HOP: ContextVar[HopInfo | None] = ContextVar("owlbank_telemetry_hop", default=None)


def _text(text: str) -> str:
    """Strands' message encoding: a JSON list of content blocks."""
    return json.dumps([{"text": text}])


class EvalShapeProcessor(SpanProcessor):
    """Adds the user message to an ``invoke_agent`` span of a hop with no caller words."""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        try:
            hop = HOP.get()
            if hop is None or hop.caller_words or not _is_invoke_agent(span):
                return
            utterance = hop.caller_utterance or f"[call continues after {hop.step or 'the previous step'}]"
            span.add_event(USER_MESSAGE, {"content": _text(utterance)})
        except Exception:  # noqa: S110 - a failed enrichment leaves the span as Strands made it
            pass

    def on_end(self, span: ReadableSpan) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _is_invoke_agent(span: Span) -> bool:
    if (span.name or "").startswith(INVOKE_AGENT):
        return True
    return (span.attributes or {}).get("gen_ai.operation.name") == INVOKE_AGENT


def fill_interrupted_tool(span: ReadableSpan) -> ReadableSpan:
    """A copy of an ``execute_tool`` span that has no ``gen_ai.choice``, with a placeholder result.

    Any other span is returned as it is; so is this one if copying fails."""
    try:
        attributes = dict(span.attributes or {})
        if attributes.get("gen_ai.operation.name") != EXECUTE_TOOL:
            return span
        if any(event.name == CHOICE for event in span.events):
            return span
        tool = attributes.get("gen_ai.tool.name") or "the tool"
        choice: dict[str, str] = {"message": _text(f"[result from the front door: {tool}]")}
        call_id = attributes.get("gen_ai.tool.call.id")
        if isinstance(call_id, str) and call_id:
            choice["id"] = call_id
        attributes.setdefault("gen_ai.tool.status", "success")
        attributes[RESULT_NEXT_TURN] = True
        events = (*span.events, Event(CHOICE, choice, timestamp=span.end_time))
        return ReadableSpan(
            name=span.name, context=span.context, parent=span.parent, resource=span.resource,
            attributes=attributes, events=events, links=span.links, kind=span.kind,
            status=span.status, start_time=span.start_time, end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )
    except Exception:  # noqa: BLE001 - the span is exported without the placeholder
        return span
