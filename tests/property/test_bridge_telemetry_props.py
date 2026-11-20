"""Call observability properties for the bridge.

Each example is a generated call run end to end: the CR simulator and real TAC, the bridge, and the
real Strands client on a scripted ``FakeModel``, with telemetry installed as in production
(``tests.strands_fakes.traced``: baggage, evaluation-shape and batch processors over the content
filter, an in-memory exporter). A call is one to three ConversationRelay sessions; each session has
zero to three caller turns (a plain answer, an inline bank tool, or a Gateway tool, each answered),
and every session but the last ends on a step-up hand-off (``authenticate``, an inline tool that
ends its turn on an interrupt), after which the next session resumes from the front door's result
with no caller words.

* Every exported span of the call carries ``session.id == call_sid`` (the
  generated CallSid); a span started outside any call is not exported, one started in the call's
  baggage outside a turn is, with the session.
* (8.0, 8.1): every exported ``invoke_agent`` span has a ``gen_ai.user.message`` event and
  every ``execute_tool`` span a ``gen_ai.choice`` event and a ``gen_ai.tool.status``.
* With an exception injected into the exporter, the content filter, our span
  processors, the client's hooks or the bridge's own spans (including the start of a
  ``memory.reload`` child span, run on the fake Memory), the call's turn outcomes (outcome, stop
  reason, hops, tools) and everything sent to ConversationRelay equal the untraced run's.
* With ``redacted`` content, for any order of span starts and ends,
  session-end forgets, re-registrations and exports within ``RETAIN_TTL_S`` of the forget, no value
  registered for call C when one of C's spans was written appears in that span as exported, nor
  any call's value in a span with no call. (Spans and values only, no call run: the order is what
  varies.) The snapshot's TTL and cap are unit-tested in ``tests/unit/test_telemetry_bridge_wiring.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank.envelope import ResultEnvelope, Spoken
from owlbank.telemetry import bridge as tb
from owlbank.telemetry import evalshape, filtering
from tac_bridge import agent_client_strands as acs
from tac_bridge.agent_client_strands import StrandsAgentClient
from tests.component.masking_fixtures import unmasked_policy_doc
from tests.component.test_bridge_fakes import (
    FakeBackend,
    front_door_update,
    make_rig,
    rearm_nonce,
    write_call_state,
)
from tests.logutil import lines
from tests.strands_fakes import (
    FakeMemoryData,
    FakeModel,
    Traced,
    default_gateway,
    light_memory_client,
    memory_factory,
    text_msg,
    tool_msg,
    traced,
)
from tests.strands_fakes import settings as client_settings

PROPS = settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
BRIDGE_LOG = "tac_bridge.bridge"
TXNS = {"account": "checking", "transactions": [{"date": "2026-10-09", "description": "Coffee", "amount": "-4.50"}]}
PROMPTS = {"say": "hello there", "inline": "what did I spend?", "gateway": "any fees?"}
KINDS = st.sampled_from(sorted(PROMPTS))


@dataclass(frozen=True)
class Plan:
    call_sid: str
    sessions: tuple[tuple[str, ...], ...]  # caller turn kinds per session


plans = st.builds(
    Plan,
    call_sid=st.text("0123456789abcdef", min_size=32, max_size=32).map(lambda h: "CA" + h),
    sessions=st.lists(st.lists(KINDS, max_size=3).map(tuple), min_size=1, max_size=3).map(tuple),
)


def _verified() -> dict[str, Any]:
    return json.loads(ResultEnvelope(status="verified", tool="authenticate",
                                     spoken=[Spoken("stepup.success", "Thanks, you're verified.")],
                                     data={"forTool": "make_payment"}, next="retry_original_tool",
                                     forTool="make_payment").to_json())


def model_script(plan: Plan) -> list[Any]:
    """The model's answers in call order: per turn kind, then the step-up and the resume."""
    out: list[Any] = []
    n = 0
    for i, kinds in enumerate(plan.sessions):
        if i > 0:
            out.append(text_msg("You're all set."))  # the resumed turn after the step-up
        for kind in kinds:
            n += 1
            if kind == "inline":
                out += [tool_msg((f"tu_tx{n}", "list_recent_transactions", {"account": "checking"})),
                        text_msg("Your latest was four fifty.")]
            elif kind == "gateway":
                out += [tool_msg((f"tu_gw{n}", "level0___faq_lookup", {"q": "fees"})), text_msg("No fee.")]
            else:
                out.append(text_msg("Hello, how can I help?"))
        if i < len(plan.sessions) - 1:
            out.append(tool_msg((f"tu_auth{i}", "authenticate", {"tool": "make_payment"})))
    return out


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def _bridge_records() -> Iterator[_Records]:
    logger = logging.getLogger(BRIDGE_LOG)
    handler, level = _Records(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


@dataclass
class Outcome:
    spoken: list[list[str]]  # text tokens per session
    ends: list[tuple[Any, Any]]  # (reason, tool) of each session's end frame
    turns: list[tuple[Any, ...]]


def run_call(plan: Plan, t: Traced | None, *, tracer: Any = None,
             memory: pytest.MonkeyPatch | None = None) -> Outcome:
    """Run ``plan``; with ``t`` the bridge traces through it (``tracer`` overrides its tracer).
    With ``memory`` the client keeps its history in the fake Memory and the call's owner is another
    process, so each session start reloads (a ``memory.reload`` span)."""
    sid = plan.call_sid
    rig = make_rig(backend=FakeBackend({"list_recent_transactions": lambda **_: dict(TXNS)}),
                   policy_doc=unmasked_policy_doc())  # no timed phrases: runs compare exactly
    model = FakeModel(model_script(plan))
    if memory is None:
        rig.bridge._agent = StrandsAgentClient(client_settings(), model_factory=lambda _s, _sid: model,
                                               gateway_tools=default_gateway())
    else:
        light_memory_client(memory)
        rig.bridge._agent = StrandsAgentClient(client_settings(memory_id="mem-1"),
                                               model_factory=lambda _s, _sid: model,
                                               gateway_tools=default_gateway(),
                                               session_manager_factory=memory_factory(FakeMemoryData()))
    if t is not None:
        provider = t.handle.tracer_provider
        rig.bridge.enable_tracing(tracer if tracer is not None else t.tracer,
                                  flush=lambda: provider.force_flush(3000))
    write_call_state(rig.store, sid, **({} if memory is None else {"agent_owner": "another-process"}))
    spoken: list[list[str]] = []
    ends: list[tuple[Any, Any]] = []

    async def go() -> None:
        for i, kinds in enumerate(plan.sessions):
            if i > 0:
                front_door_update(rig.store, sid, pending_result=_verified(), level=3)
                rearm_nonce(rig.store, sid)
            sim = rig.sim()
            await sim.connect(sid)
            frames: list[Any] = []
            if i > 0:
                frames += await sim.wait_reply()
            for kind in kinds:
                frames += await sim.prompt(PROMPTS[kind])
            if i < len(plan.sessions) - 1:
                frames += await sim.prompt("pay 25 dollars on my card")
            await sim.hangup()
            rig.bridge.session_end(sid, "closed")
            await rig.bridge.wait_idle()
            spoken.append([f.get("token", "") for f in frames if f.get("type") == "text"])
            end = [json.loads(f["handoffData"]) for f in frames if f.get("type") == "end"]
            ends.append((end[0].get("reason"), end[0].get("tool")) if end else (None, None))

    with _bridge_records() as rec:
        asyncio.run(go())
    turns = [(ln["outcome"], ln["stop_reason"], ln["hops"], tuple(ln["tools"]), ln["resumed"])
             for ln in lines(rec.records, BRIDGE_LOG) if ln["event"] == "turn"]
    assert not model.script, "every scripted model call ran"
    return Outcome(spoken, ends, turns)


@pytest.fixture(scope="module")
def telemetry() -> Iterator[Traced]:
    with traced() as t:
        yield t


def operation(span: Any) -> str | None:
    return (span.attributes or {}).get("gen_ai.operation.name")




@PROPS
@given(plan=plans)
def test_every_exported_span_of_a_call_carries_its_session(telemetry: Traced, plan: Plan) -> None:
    from opentelemetry import baggage
    from opentelemetry.context import Context

    telemetry.clear()
    run_call(plan, telemetry)
    tracer = telemetry.tracer
    tracer.start_span("outside.any.call").end()  # no call in context: dropped
    with tracer.start_as_current_span("outside.under.a.parent"):
        tracer.start_span("child.of.no.call").end()
    in_call = baggage.set_baggage("call_sid", plan.call_sid, context=Context())
    tracer.start_span("memory.write.after.turn", context=in_call).end()  # known call: kept
    spans = telemetry.spans()
    assert spans
    assert {s.attributes.get("session.id") for s in spans} == {plan.call_sid}
    assert {s.attributes.get("call_sid") for s in spans} == {plan.call_sid}
    names = {s.name for s in spans}
    assert not names & {"outside.any.call", "outside.under.a.parent", "child.of.no.call"}
    assert "memory.write.after.turn" in names
    assert len([s for s in spans if s.name == "relay.session"]) == len(plan.sessions)
    turns = sum(len(k) for k in plan.sessions) + 2 * (len(plan.sessions) - 1)
    assert len([s for s in spans if s.name == "bridge.turn"]) == turns




@PROPS
@given(plan=plans)
def test_every_invoke_agent_has_a_user_message_and_every_tool_a_choice_and_status(
        telemetry: Traced, plan: Plan) -> None:
    telemetry.clear()
    run_call(plan, telemetry)
    spans = telemetry.spans()
    invokes = [s for s in spans if operation(s) == "invoke_agent"]
    tools = [s for s in spans if operation(s) == "execute_tool"]
    assert bool(invokes) == (any(plan.sessions[0]) or len(plan.sessions) > 1)
    for s in invokes:
        assert any(e.name == "gen_ai.user.message" for e in s.events), s.name
    for s in tools:
        assert any(e.name == "gen_ai.choice" for e in s.events), s.attributes.get("gen_ai.tool.name")
        assert s.attributes.get("gen_ai.tool.status") in ("success", "error"), s.attributes.get("gen_ai.tool.name")
    expected_tools = sum(k != "say" for kinds in plan.sessions for k in kinds) + len(plan.sessions) - 1
    assert len({s.attributes.get("gen_ai.tool.call.id") for s in tools}) == expected_tools




class _Boom(RuntimeError):
    pass


def _raise(*_a: Any, **_k: Any) -> Any:
    raise _Boom("injected")


class _RaisingExporter:
    def export(self, spans: Any) -> Any:
        raise _Boom("export")

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class _RaisingBaggage:
    """Stands in for ``opentelemetry.baggage`` inside the baggage processor."""

    get_baggage = staticmethod(_raise)


class _RaisingHop:
    get = staticmethod(_raise)


class _RaisingSpan:
    def __getattr__(self, name: str) -> Any:
        return _raise


class _RaisingTracer:
    """The bridge's tracer, broken: start fails, or the spans it gives fail on every method."""

    def __init__(self, inner: Any, *, at_start: bool) -> None:
        self._inner, self._at_start = inner, at_start

    def start_span(self, *a: Any, **k: Any) -> Any:
        if self._at_start:
            raise _Boom("start_span")
        return _RaisingSpan()


class _ChildRaisingTracer:
    """The bridge's tracer with only ``_Tracing.child``'s start broken (the ``memory.reload`` span)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def start_span(self, name: str, *a: Any, **k: Any) -> Any:
        if name == acs.SPAN_MEMORY_RELOAD:
            raise _Boom("start_child")
        return self._inner.start_span(name, *a, **k)


#: Where the fault goes: a context manager patching it in, plus the tracer the bridge gets.
FAULTS = ("exporter", "filter", "baggage_processor", "evalshape_processor", "model_hook", "tool_hook",
          "bridge_start_span", "bridge_span_methods", "bridge_child_start")
#: Faults whose component runs only with the client on Memory (both runs then use it).
MEMORY_FAULTS = ("bridge_child_start",)


@contextmanager
def _fault(name: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Traced, Any]]:
    with ExitStack() as stack:
        mp = stack.enter_context(monkeypatch.context())
        if name == "exporter":
            t = stack.enter_context(traced(exporter=_RaisingExporter()))
        else:
            t = stack.enter_context(traced())
        tracer: Any = None
        if name == "filter":
            mp.setattr(filtering, "fill_interrupted_tool", _raise)
        elif name == "baggage_processor":
            mp.setattr(tb, "baggage", _RaisingBaggage)
        elif name == "evalshape_processor":
            mp.setattr(evalshape, "HOP", _RaisingHop())
        elif name == "model_hook":
            mp.setattr(acs, "_log_model_call", _raise)
        elif name == "tool_hook":
            mp.setattr(acs, "tool_kind", _raise)
        elif name == "bridge_start_span":
            tracer = _RaisingTracer(t.tracer, at_start=True)
        elif name == "bridge_span_methods":
            tracer = _RaisingTracer(t.tracer, at_start=False)
        elif name == "bridge_child_start":
            tracer = _ChildRaisingTracer(t.tracer)
        yield t, tracer


@settings(max_examples=16, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(plan=plans, fault=st.sampled_from(FAULTS))
def test_telemetry_faults_never_change_the_call(plan: Plan, fault: str, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = monkeypatch if fault in MEMORY_FAULTS else None
    untraced = run_call(plan, None, memory=memory)
    with _fault(fault, monkeypatch) as (t, tracer):
        faulty = run_call(plan, t, tracer=tracer, memory=memory)
        t.handle.tracer_provider.force_flush(5000)  # the export (and the filter) ran on what was queued
    assert faulty.turns == untraced.turns
    assert faulty.spoken == untraced.spoken
    assert faulty.ends == untraced.ends


def test_fault_list_covers_each_component() -> None:
    """Every component the property names has a fault (exporter, filter, processors, hooks, ours)."""
    covered: dict[str, Callable[[str], bool]] = {
        "exporter": lambda f: f == "exporter", "filter": lambda f: f == "filter",
        "processors": lambda f: f.endswith("_processor"), "hooks": lambda f: f.endswith("_hook"),
        "bridge spans": lambda f: f.startswith("bridge_")}
    for component, matches in covered.items():
        assert any(matches(f) for f in FAULTS), component




class _PassGuard:
    """``PiiGuard.text`` that changes nothing: only the exact-value scrub can remove a value."""

    def text(self, surface: str, text: str, **_k: Any) -> str:
        return text


_CALLS = ("CA" + "1" * 32, "CA" + "2" * 32, "CA" + "3" * 32)
#: Per call, distinct digit values (all-digit values match on non-digit boundaries only).
_VALUES = {call: (f"{i}0{i}0{i}", f"7{i}7{i}7{i}7") for i, call in enumerate(_CALLS, start=1)}

#: ``span`` starts a span of a call with text holding the call's registered values (left open),
#: ``nocall`` writes a span with no call holding every call's registered values (it reaches the
#: filter with the next export: the sampler never keeps one, so it stands for a span that lost its
#: call), ``end`` ends the oldest open span, ``forget`` is the bridge's session-end forget
#: (``retain``, then ``sensitive.forget``), ``restore`` registers the call's values again (a new
#: session), ``flush`` exports what has ended.
_OPS = st.one_of(
    st.tuples(st.just("span"), st.integers(0, 2)),
    st.tuples(st.just("nocall"), st.just(0)),
    st.tuples(st.just("end"), st.just(0)),
    st.tuples(st.just("forget"), st.integers(0, 2)),
    st.tuples(st.just("restore"), st.integers(0, 2)),
    st.tuples(st.just("flush"), st.just(0)),
)


def _span_texts(span: Any) -> list[str]:
    texts = [str(v) for v in (span.attributes or {}).values()]
    return texts + [str(v) for e in span.events for v in (e.attributes or {}).values()]


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ops=st.lists(_OPS, max_size=30))
def test_redacted_spans_never_carry_their_calls_values_whatever_the_export_order(
        ops: list[tuple[str, int]]) -> None:
    """For any order of span starts and ends, forgets, re-registrations and exports (all within
    the TTL: the redactor's clock stands still), no value registered for call C when one of C's
    spans was written appears in that span as exported, nor any call's value in a span with no call."""
    from opentelemetry import baggage
    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from owlbank import segments, sensitive

    redactor = tb.TranscriptRedactor(_PassGuard(), clock=lambda: 0.0)
    for call in _CALLS:
        sensitive.forget(call)
    written: dict[str, set[str]] = {}  # span name -> the values its text holds
    no_call_sink = InMemorySpanExporter()
    no_call_filter = tb.CallScopedContentFilter(no_call_sink, redactor=redactor)
    no_call_tracer = TracerProvider().get_tracer("p12")  # no processors: ended spans are exported by hand
    try:
        with traced("redacted", redact=redactor) as t:
            tracer = t.tracer
            opened: list[Any] = []
            no_call_ended: list[Any] = []

            def flush() -> None:
                t.handle.tracer_provider.force_flush(5000)
                no_call_filter.export(no_call_ended)
                no_call_ended.clear()

            for call in _CALLS:
                segments.register_sensitive(call, _VALUES[call])
            for n, (op, i) in enumerate(ops):
                call = _CALLS[i]
                if op in ("span", "nocall"):
                    values = sorted(segments.sensitive_values(call) if op == "span"
                                    else {v for c in _CALLS for v in segments.sensitive_values(c)})
                    text = "caller said " + " and ".join(values) if values else "nothing sensitive"
                    if op == "span":
                        ctx = baggage.set_baggage("call_sid", call, context=Context())
                        span = tracer.start_span(f"s{n}", context=ctx, attributes={"gen_ai.input.messages": text})
                    else:
                        span = no_call_tracer.start_span(f"s{n}", attributes={"gen_ai.input.messages": text})
                    span.add_event("gen_ai.user.message", {"content": text})
                    written[f"s{n}"] = set(values)
                    opened.append(span)
                elif op == "end" and opened:
                    span = opened.pop(0)
                    span.end()
                    if tb.span_call(span) is None:
                        no_call_ended.append(span)
                elif op == "forget":
                    redactor.retain(call)
                    sensitive.forget(call)
                elif op == "restore":
                    segments.register_sensitive(call, _VALUES[call])
                elif op == "flush":
                    flush()
            for span in opened:
                span.end()
                if tb.span_call(span) is None:
                    no_call_ended.append(span)
            flush()
            spans = [*t.spans(), *no_call_sink.get_finished_spans()]
    finally:
        for call in _CALLS:
            sensitive.forget(call)
    assert {s.name for s in spans} == set(written)
    for s in spans:
        texts = _span_texts(s)
        for value in written[s.name]:
            assert not any(segments.contains_value(text, value) for text in texts), (s.name, s.attributes)
