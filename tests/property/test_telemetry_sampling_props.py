"""Call observability: properties of per-call sampling.

* The decision for a ``call_sid`` is the same whatever the span's name, kind, attributes and
  trace ID, on repeated calls, and through ``make_sampler`` (``ParentBased``) for a root span;
  it equals ``bucket(call_sid) < rate``, a pure function (so every process agrees).
* Over many CallSids the sampled share is within tolerance of the rate; 0 samples nothing and 1
  samples everything.
* No ``call_sid`` in baggage: dropped.
* A child follows its parent whatever its own baggage says.
"""

from __future__ import annotations

import random

from hypothesis import given, settings
from hypothesis import strategies as st
from opentelemetry import baggage, trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace.sampling import Decision
from opentelemetry.trace import NonRecordingSpan, SpanContext, SpanKind, TraceFlags

from owlbank.telemetry.sampling import CallSampler, bucket, make_sampler

SETTINGS = settings(max_examples=200, deadline=None)

HEX = "0123456789abcdef"
call_sids = st.text(alphabet=HEX, min_size=32, max_size=32).map(lambda h: "CA" + h)
rates = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
names = st.text(max_size=40)
kinds = st.sampled_from(list(SpanKind))
attrs = st.dictionaries(st.text(max_size=20), st.one_of(st.text(max_size=20), st.integers(), st.booleans()),
                        max_size=6)
trace_ids = st.integers(min_value=1, max_value=2**128 - 1)


def _ctx(call_sid: str | None) -> Context:
    return baggage.set_baggage("call_sid", call_sid, Context()) if call_sid is not None else Context()


def _sampled(sampler: object, call_sid: str | None, name: str, kind: SpanKind, attributes: dict,
             trace_id: int, parent: Context | None = None) -> bool:
    ctx = parent if parent is not None else _ctx(call_sid)
    result = sampler.should_sample(ctx, trace_id, name, kind, attributes)  # type: ignore[attr-defined]
    return result.decision is Decision.RECORD_AND_SAMPLE


@SETTINGS
@given(call_sids, rates, st.lists(st.tuples(names, kinds, attrs, trace_ids), min_size=2, max_size=6))
def test_decision_depends_on_the_call_sid_alone(call_sid: str, rate: float,
                                                   spans: list[tuple[str, SpanKind, dict, int]]) -> None:
    expected = bucket(call_sid) < rate
    for sampler in (CallSampler(rate), make_sampler(rate)):
        for name, kind, attributes, trace_id in spans + spans[::-1]:
            assert _sampled(sampler, call_sid, name, kind, attributes, trace_id) is expected
    # A fresh sampler (another process) agrees.
    assert _sampled(CallSampler(rate), call_sid, "x", SpanKind.SERVER, {}, 1) is expected


@SETTINGS
@given(call_sids)
def test_bucket_is_stable_and_in_range(call_sid: str) -> None:
    b = bucket(call_sid)
    assert 0.0 <= b < 1.0
    assert bucket(str(call_sid)) == b
    assert _sampled(CallSampler(0.0), call_sid, "n", SpanKind.INTERNAL, {}, 1) is False
    assert _sampled(CallSampler(1.0), call_sid, "n", SpanKind.INTERNAL, {}, 1) is True


@settings(max_examples=50, deadline=None)
@given(rates, st.integers(min_value=0, max_value=2**32))
def test_sampled_share_approaches_the_rate(rate: float, seed: int) -> None:
    rng = random.Random(seed)
    n = 4000
    sampler = CallSampler(rate)
    hits = sum(_sampled(sampler, "CA" + "".join(rng.choice(HEX) for _ in range(32)), "t", SpanKind.SERVER,
                        {}, 1) for _ in range(n))
    # Binomial standard deviation is at most 0.008 at n=4000; 0.04 is five of them.
    assert abs(hits / n - rate) <= 0.04


@SETTINGS
@given(rates, names, kinds, attrs, trace_ids, st.sampled_from([None, ""]))
def test_no_call_sid_is_dropped(rate: float, name: str, kind: SpanKind, attributes: dict, trace_id: int,
                                call_sid: str | None) -> None:
    for sampler in (CallSampler(rate), make_sampler(rate)):
        assert _sampled(sampler, call_sid, name, kind, attributes, trace_id) is False


@SETTINGS
@given(call_sids, rates, st.booleans(), names)
def test_children_follow_the_parent(call_sid: str, rate: float, parent_sampled: bool, name: str) -> None:
    flags = TraceFlags(TraceFlags.SAMPLED if parent_sampled else TraceFlags.DEFAULT)
    parent = NonRecordingSpan(SpanContext(trace_id=7, span_id=9, is_remote=False, trace_flags=flags))
    ctx = trace.set_span_in_context(parent, _ctx(call_sid))
    assert _sampled(make_sampler(rate), call_sid, name, SpanKind.INTERNAL, {}, 7, parent=ctx) is parent_sampled


@SETTINGS
@given(call_sids, rates)
def test_in_a_provider_a_call_is_all_or_nothing(call_sid: str, rate: float) -> None:
    from opentelemetry import context as otel_context
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    out = InMemorySpanExporter()
    provider = TracerProvider(sampler=make_sampler(rate))
    provider.add_span_processor(SimpleSpanProcessor(out))
    tracer = provider.get_tracer("t")
    token = otel_context.attach(baggage.set_baggage("call_sid", call_sid))
    try:
        for root in ("bridge.turn", "relay.session", "gather.card"):
            with tracer.start_as_current_span(root), tracer.start_as_current_span("chat"):
                pass
    finally:
        otel_context.detach(token)
    assert len(out.get_finished_spans()) == (6 if bucket(call_sid) < rate else 0)
