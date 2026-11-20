"""The span keys that may leave the process with ``traceContent=off``.

:data:`ATTRIBUTES` is the one list: a key, or a prefix written ``prefix.*``, mapped to the reason
it holds no conversation text. The content filter (:mod:`owlbank.telemetry.filtering`) applies it
to span attributes, event attributes and link attributes alike; anything not listed is dropped.
Keys come from what Strands 1.61 and botocore emit and from the span catalogue.

:data:`TEXT_KEYS` are the keys known to hold text (prompts, messages, tool arguments and
results, exception messages). None of them is allowed; with ``redacted`` they go through the
redaction instead, and with ``full`` they pass unchanged.

``aws.auth.*`` (credentials and access key IDs on botocore spans) is never allowed, in any
mode.

Stdlib only.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

ATTRIBUTES: Mapping[str, str] = MappingProxyType({
    # Correlation.
    "session.id": "the CallSid, which groups a call's traces into one session",
    "call_sid": "the CallSid, which joins spans to the call's log lines",
    # Strands' GenAI convention keys that Evaluations and the GenAI views read.
    "gen_ai.operation.name": "span kind: invoke_agent, execute_event_loop_cycle, chat, execute_tool",
    "gen_ai.system": "the model provider's name (aws.bedrock)",
    "gen_ai.provider.name": "the model provider's name (newer convention)",
    "gen_ai.agent.name": "our fixed agent name",
    "gen_ai.agent.tools": "the agent's tool names (no descriptions or schemas), for the GenAI views",
    "gen_ai.request.model": "the model ID",
    "gen_ai.usage.*": "token counts (input, output, cache read and write)",
    "gen_ai.server.*": "server timings: time to first token, request duration",
    "gen_ai.tool.name": "the tool's name",
    "gen_ai.tool.call.id": "the model's tool-use ID, opaque",
    "gen_ai.tool.status": "the tool call's outcome (success, error)",
    "gen_ai.event.start_time": "a timestamp",
    "gen_ai.event.end_time": "a timestamp",
    "event_loop.*": "Strands' event-loop cycle IDs, opaque",
    "finish_reason": "the model's stop reason (span attribute or gen_ai.choice event attribute)",
    # Event attributes allowed with off: the event names stay, the text attributes go.
    "id": "the tool-use ID on a tool message or choice event, opaque",
    "role": "the message role (user, assistant, tool)",
    # Ours, on the turn, session and webhook spans; never a value or speech.
    "owlbank.*": "our own identifiers, counts, timings, outcomes and error classes",
    # Errors: the class only, never the message or stack.
    "exception.type": "the exception's class name",
    # Botocore CLIENT spans.
    "rpc.*": "the AWS service and operation called",
    "aws.region": "the Region called",
    "aws.request_id": "AWS's request ID, for support cases",
    "aws.remote.*": "the remote service and operation, for the service map",
    "aws.local.*": "our own service name, for the service map",
    "aws.span.kind": "X-Ray's span kind",
    "aws.xray.*": "X-Ray trace IDs, to join Lambda segments",
    # Botocore CLIENT span metadata that can show a token: no text.
    "gen_ai.request.max_tokens": "the output token limit we set",
    "gen_ai.response.finish_reasons": "the model's stop reasons (end_turn, tool_use, ...)",
    "http.status_code": "the AWS call's HTTP status",
    "http.response.status_code": "the AWS call's HTTP status (newer convention)",
    "server.address": "the AWS endpoint host",
    "server.port": "the AWS endpoint port",
    "retry_attempts": "botocore's retry count",
    "cloud.region": "the Region called",
})

TEXT_KEYS: frozenset[str] = frozenset({
    "content",  # gen_ai.user/assistant/system/tool message events
    "message",  # gen_ai.choice event: the model's answer, or a tool's result
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "tool.result",  # gen_ai.choice event on execute_event_loop_cycle
    "exception.message",
    "exception.stacktrace",
    "owlbank.reasoning_text",  # bridge.turn: the reasoning the model wrote as text, never spoken
})

NEVER_PREFIX = "aws.auth."

_EXACT = frozenset(k for k in ATTRIBUTES if not k.endswith(".*"))
_PREFIXES = tuple(k[:-1] for k in ATTRIBUTES if k.endswith(".*"))  # "gen_ai.usage.*" -> "gen_ai.usage."


def never(key: str) -> bool:
    """Whether ``key`` is dropped in every mode (``aws.auth.*``)."""
    return key.startswith(NEVER_PREFIX)


def allowed(key: str) -> bool:
    """Whether ``key`` may be exported with ``traceContent=off``."""
    if never(key) or key in TEXT_KEYS:
        return False
    return key in _EXACT or key.startswith(_PREFIXES)
