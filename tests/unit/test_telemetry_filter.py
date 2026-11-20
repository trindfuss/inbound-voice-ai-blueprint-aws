"""Call observability: the content filter, the allow-list and the evaluation shape.

Spans are made with the OpenTelemetry SDK (a real ``TracerProvider``) in the shapes recorded
for Strands 1.61 and botocore, and exported through
:class:`ContentFilterExporter` to an ``InMemorySpanExporter``.

* ``off``: allow-listed attributes only; events keep their names, text attributes go; exception
  events keep ``exception.type``; status descriptions go.
* ``redacted``: text keys through ``redact``, a key dropped when it returns ``None`` or raises.
* ``full``: unchanged. ``aws.auth.*`` goes in every mode.
* :func:`fill_interrupted_tool` and :class:`EvalShapeProcessor`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from owlbank.telemetry import allowlist
from owlbank.telemetry.evalshape import HOP, RESULT_NEXT_TURN, EvalShapeProcessor, HopInfo, fill_interrupted_tool
from owlbank.telemetry.filtering import ContentFilterExporter

SYSTEM_PROMPT = "You are OwlBank's assistant. The caller is Thomas."
CALLER = "what's my balance on the card ending 4242"
ANSWER = "Your balance is 512 dollars."
TOOL_ARGS = '{"account": "checking"}'
TOOL_RESULT = '{"balance": 512}'
ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
MARKERS = (SYSTEM_PROMPT, CALLER, ANSWER, TOOL_ARGS, TOOL_RESULT, ACCESS_KEY, "secret boom")


@pytest.fixture
def made() -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    raw = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(EvalShapeProcessor())
    provider.add_span_processor(SimpleSpanProcessor(raw))
    yield provider, raw
    provider.shutdown()


def _strands_like(provider: TracerProvider) -> None:
    """An invoke_agent > chat + execute_tool tree with Strands' text events, a botocore CLIENT
    span with an access key, and an error with an exception event."""
    tracer = provider.get_tracer("strands.telemetry.tracer")
    common = {"session.id": "CA1", "call_sid": "CA1"}
    with tracer.start_as_current_span("invoke_agent owlbank", attributes={
            **common, "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "owlbank",
            "gen_ai.request.model": "us.anthropic.claude", "gen_ai.usage.input_tokens": 120}) as agent:
        agent.add_event("gen_ai.system.message", {"content": SYSTEM_PROMPT})
        agent.add_event("gen_ai.user.message", {"content": json.dumps([{"text": CALLER}])})
        with tracer.start_as_current_span("chat", attributes={
                **common, "gen_ai.operation.name": "chat", "gen_ai.server.time_to_first_token": 410}) as chat:
            chat.add_event("gen_ai.choice", {"message": ANSWER, "finish_reason": "end_turn"})
        with tracer.start_as_current_span("execute_tool get_balance", attributes={
                **common, "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "get_balance",
                "gen_ai.tool.call.id": "tooluse_1", "gen_ai.tool.status": "success",
                "gen_ai.tool.description": "Reads the balance"}) as tool:
            tool.add_event("gen_ai.tool.message", {"content": TOOL_ARGS, "id": "tooluse_1", "role": "tool"})
            tool.add_event("gen_ai.choice", {"message": TOOL_RESULT, "id": "tooluse_1"})
        with tracer.start_as_current_span("chat us.anthropic.claude", attributes={
                **common, "rpc.service": "BedrockRuntime", "rpc.method": "ConverseStream",
                "aws.region": "us-east-1", "aws.auth.account.access_key": ACCESS_KEY,
                "aws.auth.region": "us-east-1", "gen_ai.system": "aws.bedrock"}):
            pass
        try:
            with tracer.start_as_current_span("execute_tool broken", attributes={
                    **common, "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "broken"}):
                raise RuntimeError("secret boom")
        except RuntimeError:
            pass


def _export(raw: InMemorySpanExporter, mode: str, redact: Any = None) -> dict[str, ReadableSpan]:
    out = InMemorySpanExporter()
    result = ContentFilterExporter(out, mode=mode, redact=redact).export(raw.get_finished_spans())  # type: ignore[arg-type]
    assert result is SpanExportResult.SUCCESS
    return {s.name: s for s in out.get_finished_spans()}


def _all_text(spans: dict[str, ReadableSpan]) -> str:
    parts: list[str] = []
    for s in spans.values():
        parts += [str(dict(s.attributes or {})), str(s.status.description)]
        parts += [f"{e.name} {dict(e.attributes or {})}" for e in s.events]
    return "\n".join(parts)


def _keys(s: ReadableSpan) -> set[str]:
    return set(s.attributes or {})


def _events(s: ReadableSpan) -> list[tuple[str, dict[str, Any]]]:
    return [(e.name, dict(e.attributes or {})) for e in s.events]


# -- allow-list ---------------------------------------------------------------------------------


def test_no_text_key_is_allowed_and_every_entry_has_a_reason() -> None:
    assert not [k for k in allowlist.TEXT_KEYS if allowlist.allowed(k)]
    assert all(reason.strip() for reason in allowlist.ATTRIBUTES.values())


@pytest.mark.parametrize("key", ["session.id", "call_sid", "gen_ai.usage.input_tokens",
                                 "gen_ai.server.time_to_first_token", "event_loop.cycle_id", "owlbank.turn",
                                 "rpc.method", "aws.remote.service", "aws.xray.lambda_trace_id", "finish_reason",
                                 "id", "role", "exception.type", "gen_ai.tool.status",
                                 "http.status_code", "http.response.status_code", "gen_ai.response.finish_reasons",
                                 "gen_ai.request.max_tokens", "server.address", "server.port", "retry_attempts",
                                 "cloud.region"])
def test_allowed_keys(key: str) -> None:
    assert allowlist.allowed(key)


@pytest.mark.parametrize("key", ["content", "message", "gen_ai.tool.call.arguments", "exception.message",
                                 "exception.stacktrace", "aws.auth.account.access_key", "aws.auth.region",
                                 "gen_ai.tool.description", "gen_ai.usage", "owlbank", "gen_ai.usagex",
                                 "http.url", "anything",
                                 # text-carrying siblings of the botocore keys: a widening to a prefix must fail
                                 "url.full", "http.request.body", "http.response.body", "server.request.body",
                                 "gen_ai.request.messages", "gen_ai.response.text", "gen_ai.prompt",
                                 "gen_ai.completion"])
def test_refused_keys(key: str) -> None:
    assert not allowlist.allowed(key)


# -- modes --------------------------------------------------------------------------------------


def test_off_keeps_allowed_keys_and_event_names_only(made: Any) -> None:
    provider, raw = made
    _strands_like(provider)
    spans = _export(raw, "off")
    text = _all_text(spans)
    for marker in MARKERS:
        assert marker not in text
    for s in spans.values():
        assert all(allowlist.allowed(k) for k in _keys(s))
        for _, attrs in _events(s):
            assert all(allowlist.allowed(k) for k in attrs)
    agent = spans["invoke_agent owlbank"]
    assert [name for name, _ in _events(agent)] == ["gen_ai.system.message", "gen_ai.user.message"]
    assert _keys(agent) >= {"session.id", "call_sid", "gen_ai.operation.name", "gen_ai.usage.input_tokens"}
    assert _events(spans["chat"]) == [("gen_ai.choice", {"finish_reason": "end_turn"})]
    tool = spans["execute_tool get_balance"]
    assert "gen_ai.tool.description" not in _keys(tool)
    assert _events(tool) == [("gen_ai.tool.message", {"id": "tooluse_1", "role": "tool"}),
                             ("gen_ai.choice", {"id": "tooluse_1"})]


def test_exception_events_keep_type_only_and_status_loses_its_message(made: Any) -> None:
    provider, raw = made
    _strands_like(provider)
    for mode in ("off", "redacted"):
        broken = _export(raw, mode, redact=lambda t: t)["execute_tool broken"]
        exc = [attrs for name, attrs in _events(broken) if name == "exception"]
        assert exc == [{"exception.type": "RuntimeError"}]
        assert broken.status.status_code is StatusCode.ERROR
        assert not broken.status.description


@pytest.mark.parametrize("mode", ["off", "redacted", "full"])
def test_aws_auth_dropped_in_every_mode(made: Any, mode: str) -> None:
    provider, raw = made
    _strands_like(provider)
    boto = _export(raw, mode, redact=lambda t: t)["chat us.anthropic.claude"]
    assert not [k for k in _keys(boto) if k.startswith("aws.auth.")]
    assert ACCESS_KEY not in _all_text({"b": boto})
    assert _keys(boto) >= {"rpc.service", "rpc.method", "aws.region", "gen_ai.system"}


def test_redacted_passes_text_keys_through_redact(made: Any) -> None:
    provider, raw = made
    _strands_like(provider)
    seen: list[str] = []

    def redact(text: str) -> str:
        seen.append(text)
        return text.replace("Thomas", "[name]").replace("512", "[amount]")

    spans = _export(raw, "redacted", redact)
    agent = dict(_events(spans["invoke_agent owlbank"]))
    assert agent["gen_ai.system.message"] == {"content": SYSTEM_PROMPT.replace("Thomas", "[name]")}
    assert _events(spans["chat"]) == [("gen_ai.choice", {"message": "Your balance is [amount] dollars.",
                                                         "finish_reason": "end_turn"})]
    assert SYSTEM_PROMPT in seen
    # Non-text, non-allowed keys still go; exception messages never reach redact.
    assert "gen_ai.tool.description" not in _keys(spans["execute_tool get_balance"])
    assert "secret boom" not in seen


@pytest.mark.parametrize("redact", [lambda t: None, lambda t: 1 / 0, None])
def test_redacted_drops_a_key_whose_redaction_fails(made: Any, redact: Any) -> None:
    provider, raw = made
    _strands_like(provider)
    spans = _export(raw, "redacted", redact)
    text = _all_text(spans)
    for marker in MARKERS:
        assert marker not in text
    assert _events(spans["chat"]) == [("gen_ai.choice", {"finish_reason": "end_turn"})]


def test_full_is_unchanged_except_aws_auth(made: Any) -> None:
    provider, raw = made
    _strands_like(provider)
    before = {s.name: s for s in raw.get_finished_spans()}
    after = _export(raw, "full")
    for name, s in before.items():
        expected = {k: v for k, v in (s.attributes or {}).items() if not k.startswith("aws.auth.")}
        if name == "execute_tool broken":  # filled with a placeholder choice
            expected = {**expected, "gen_ai.tool.status": "success", RESULT_NEXT_TURN: True}
        assert dict(after[name].attributes or {}) == expected
        assert after[name].status.description == s.status.description
        assert after[name].context == s.context and after[name].parent == s.parent
    chat = after["chat"]
    assert _events(chat) == [("gen_ai.choice", {"message": ANSWER, "finish_reason": "end_turn"})]
    exc = [a for n, a in _events(after["execute_tool broken"]) if n == "exception"][0]
    assert exc["exception.message"] == "secret boom"


def _broken_filter(monkeypatch: pytest.MonkeyPatch, out: InMemorySpanExporter, fail: Any) -> ContentFilterExporter:
    exporter = ContentFilterExporter(out, mode="off")
    real = exporter._copy

    def copy(span: ReadableSpan) -> ReadableSpan:
        if fail(span):
            raise RuntimeError(f"secret {CALLER}")
        return real(span)

    monkeypatch.setattr(exporter, "_copy", copy)
    return exporter


def test_a_batch_where_every_span_fails_filtering_is_a_failure(
        made: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, raw = made
    _strands_like(provider)
    out = InMemorySpanExporter()
    exporter = _broken_filter(monkeypatch, out, lambda span: True)
    spans = raw.get_finished_spans()
    assert exporter.export(spans) is SpanExportResult.FAILURE
    assert exporter.dropped == len(spans) and out.get_finished_spans() == ()
    assert exporter.export([]) is SpanExportResult.SUCCESS  # nothing to export is not a failure


def test_a_batch_with_some_spans_surviving_is_a_success(made: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, raw = made
    _strands_like(provider)
    out = InMemorySpanExporter()
    exporter = _broken_filter(monkeypatch, out, lambda span: span.name == "chat")
    spans = raw.get_finished_spans()
    assert exporter.export(spans) is SpanExportResult.SUCCESS
    assert exporter.dropped == 1 and len(out.get_finished_spans()) == len(spans) - 1


def test_a_dropped_span_is_a_rate_limited_warning_with_the_error_class_only(
        made: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    provider, raw = made
    _strands_like(provider)
    exporter = _broken_filter(monkeypatch, InMemorySpanExporter(), lambda span: True)
    with caplog.at_level(logging.WARNING, logger="owlbank.telemetry.filtering"):
        exporter.export(raw.get_finished_spans())
        exporter.export(raw.get_finished_spans())
    (record,) = [r for r in caplog.records if r.getMessage() == "trace_filter_failed"]
    assert record.levelno == logging.WARNING and record.error_class == "RuntimeError"  # type: ignore[attr-defined]
    assert CALLER not in str(record.__dict__) and CALLER not in caplog.text


def test_filter_delegates_shutdown_and_flush() -> None:
    class Inner(InMemorySpanExporter):
        flushed = False

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            Inner.flushed = True
            return True

    inner = Inner()
    f = ContentFilterExporter(inner, mode="off")
    assert f.force_flush() and Inner.flushed
    f.shutdown()
    assert inner._stopped  # noqa: SLF001


def test_mode_is_validated() -> None:
    with pytest.raises(ValueError, match="mode"):
        ContentFilterExporter(InMemorySpanExporter(), mode="some")  # type: ignore[arg-type]


# -- evaluation shape ---------------------------------------------------------------------------


def _tool_span(provider: TracerProvider, raw: InMemorySpanExporter, *, with_choice: bool,
               status: str | None = None) -> ReadableSpan:
    attrs: dict[str, Any] = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "make_payment",
                             "gen_ai.tool.call.id": "tooluse_9"}
    if status:
        attrs["gen_ai.tool.status"] = status
    with provider.get_tracer("t").start_as_current_span("execute_tool make_payment", attributes=attrs) as span:
        span.add_event("gen_ai.tool.message", {"content": TOOL_ARGS, "id": "tooluse_9"})
        if with_choice:
            span.add_event("gen_ai.choice", {"message": TOOL_RESULT, "id": "tooluse_9"})
    return raw.get_finished_spans()[-1]


def test_fill_interrupted_tool_adds_choice_status_and_flag(made: Any) -> None:
    provider, raw = made
    span = _tool_span(provider, raw, with_choice=False)
    filled = fill_interrupted_tool(span)
    assert filled is not span
    attrs = dict(filled.attributes or {})
    assert attrs["gen_ai.tool.status"] == "success" and attrs[RESULT_NEXT_TURN] is True
    choice = [dict(e.attributes or {}) for e in filled.events if e.name == "gen_ai.choice"]
    assert choice == [{"message": json.dumps([{"text": "[result from the front door: make_payment]"}]),
                       "id": "tooluse_9"}]
    assert filled.events[-1].timestamp == span.end_time
    assert (filled.context, filled.parent, filled.start_time, filled.end_time) == (
        span.context, span.parent, span.start_time, span.end_time)
    # The original is untouched.
    assert RESULT_NEXT_TURN not in (span.attributes or {})


def test_fill_interrupted_tool_keeps_an_existing_status(made: Any) -> None:
    provider, raw = made
    filled = fill_interrupted_tool(_tool_span(provider, raw, with_choice=False, status="error"))
    assert (filled.attributes or {})["gen_ai.tool.status"] == "error"


def test_fill_interrupted_tool_leaves_other_spans(made: Any) -> None:
    provider, raw = made
    done = _tool_span(provider, raw, with_choice=True)
    assert fill_interrupted_tool(done) is done
    with provider.get_tracer("t").start_as_current_span("chat", attributes={"gen_ai.operation.name": "chat"}):
        pass
    chat = raw.get_finished_spans()[-1]
    assert fill_interrupted_tool(chat) is chat


def test_fill_interrupted_tool_never_raises() -> None:
    class Broken:
        @property
        def attributes(self) -> Any:
            raise RuntimeError("no")

    broken: Any = Broken()
    assert fill_interrupted_tool(broken) is broken


def test_interrupted_tool_placeholder_survives_off_as_name_and_id(made: Any) -> None:
    provider, raw = made
    _tool_span(provider, raw, with_choice=False)
    tool = _export(raw, "off")["execute_tool make_payment"]
    assert _events(tool)[-1] == ("gen_ai.choice", {"id": "tooluse_9"})
    assert (tool.attributes or {})["gen_ai.tool.status"] == "success"
    assert (tool.attributes or {})[RESULT_NEXT_TURN] is True


def _agent_events(provider: TracerProvider, raw: InMemorySpanExporter, hop: HopInfo | None,
                  name: str = "invoke_agent owlbank", attrs: dict[str, Any] | None = None) -> list[Any]:
    token = HOP.set(hop)
    try:
        with provider.get_tracer("t").start_as_current_span(name, attributes=attrs or {}):
            pass
    finally:
        HOP.reset(token)
    return _events(raw.get_finished_spans()[-1])


def test_user_message_added_only_when_the_hop_has_no_caller_words(made: Any) -> None:
    provider, raw = made
    assert _agent_events(provider, raw, None) == []
    assert _agent_events(provider, raw, HopInfo(caller_words=True, caller_utterance="hi", step="verify")) == []
    assert _agent_events(provider, raw, HopInfo(caller_words=False, caller_utterance="yes, pay it", step="x")) == [
        ("gen_ai.user.message", {"content": json.dumps([{"text": "yes, pay it"}])})]
    assert _agent_events(provider, raw, HopInfo(caller_words=False, step="confirm.readback")) == [
        ("gen_ai.user.message", {"content": json.dumps([{"text": "[call continues after confirm.readback]"}])})]
    assert _agent_events(provider, raw, HopInfo(caller_words=False)) == [
        ("gen_ai.user.message", {"content": json.dumps([{"text": "[call continues after the previous step]"}])})]


def test_user_message_by_operation_name_and_not_on_other_spans(made: Any) -> None:
    provider, raw = made
    hop = HopInfo(caller_words=False, step="verify")
    assert [n for n, _ in _agent_events(provider, raw, hop, "agent", {"gen_ai.operation.name": "invoke_agent"})] == [
        "gen_ai.user.message"]
    assert _agent_events(provider, raw, hop, "chat", {"gen_ai.operation.name": "chat"}) == []


def test_eval_shape_processor_never_raises() -> None:
    class Broken:
        name = "invoke_agent x"

        def add_event(self, *_: Any) -> None:
            raise RuntimeError("no")

    token = HOP.set(HopInfo(caller_words=False))
    try:
        EvalShapeProcessor().on_start(Broken())  # type: ignore[arg-type]
    finally:
        HOP.reset(token)


def test_status_without_description_passes(made: Any) -> None:
    provider, raw = made
    with provider.get_tracer("t").start_as_current_span("x") as s:
        s.set_status(Status(StatusCode.OK))
    assert _export(raw, "off")["x"].status.status_code is StatusCode.OK
