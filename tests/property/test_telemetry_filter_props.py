"""Call observability: properties of the content filter.

* For any span (arbitrary attribute keys and values, events and links),
  ``ContentFilterExporter(mode="off")`` exports only keys:func:`allowlist.allowed` accepts, in
  span, event and link attributes, never ``aws.auth.*``, and no status description; the
  allow-listed span attributes it was given come out unchanged.
* When filtering a span raises, that span never reaches the inner exporter and
  every span that does reach it is filtered; a raising ``redact`` or inner exporter changes
  nothing about that.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Link, SpanContext, Status, StatusCode, TraceFlags

from owlbank.telemetry import allowlist
from owlbank.telemetry.filtering import ContentFilterExporter
from owlbank.telemetry.health import ExportHealth

import logging

SETTINGS = settings(max_examples=300, deadline=None)

KNOWN_KEYS = sorted({*(k.replace(".*", ".x") for k in allowlist.ATTRIBUTES), *allowlist.TEXT_KEYS,
                     "aws.auth.account.access_key", "aws.auth.region", "aws.auth", "gen_ai.tool.description",
                     "gen_ai.agent.tools", "http.url", "exception.escaped", "gen_ai.operation.name"})
keys = st.one_of(st.sampled_from(KNOWN_KEYS), st.text(max_size=30),
                 st.builds(lambda a, b: f"{a}.{b}", st.sampled_from(["aws", "aws.auth", "gen_ai", "owlbank",
                                                                     "rpc", "exception", "event_loop"]),
                           st.text(max_size=12)))
scalars = st.one_of(st.text(max_size=40), st.integers(), st.floats(allow_nan=False), st.booleans())
values = st.one_of(scalars, st.lists(st.text(max_size=20), max_size=4).map(tuple),
                   st.sampled_from(["invoke_agent", "execute_tool", "chat"]))
attr_maps = st.dictionaries(keys, values, max_size=12)
event_names = st.one_of(st.sampled_from(["gen_ai.user.message", "gen_ai.choice", "gen_ai.tool.message",
                                         "gen_ai.system.message", "exception"]), st.text(max_size=20))
events = st.lists(st.builds(lambda n, a: Event(n, a, timestamp=1), event_names, attr_maps), max_size=5)
_ids = st.integers(min_value=1, max_value=2**63)


def _ctx(span_id: int) -> SpanContext:
    return SpanContext(trace_id=span_id, span_id=span_id, is_remote=False, trace_flags=TraceFlags(1))


links = st.lists(st.builds(lambda i, a: Link(_ctx(i), a), _ids, attr_maps), max_size=2)


@st.composite
def spans(draw: Any, span_id: int | None = None) -> ReadableSpan:
    sid = span_id if span_id is not None else draw(_ids)
    status = draw(st.sampled_from([Status(StatusCode.UNSET), Status(StatusCode.OK),
                                   Status(StatusCode.ERROR, draw(st.text(max_size=30)))]))
    return ReadableSpan(name=draw(st.text(max_size=30)), context=_ctx(sid), attributes=draw(attr_maps),
                        events=draw(events), links=draw(links), status=status, start_time=1, end_time=2)


class Capture(SpanExporter):
    def __init__(self, raises: bool = False) -> None:
        self.got: list[ReadableSpan] = []
        self.raises = raises

    def export(self, batch: Sequence[ReadableSpan]) -> SpanExportResult:
        self.got.extend(batch)
        if self.raises:
            raise ConnectionError("down")
        return SpanExportResult.SUCCESS


def _assert_filtered(span: ReadableSpan) -> None:
    for key in span.attributes or {}:
        assert allowlist.allowed(key), key
        assert not key.startswith("aws.auth.")
    for event in span.events:
        for key in event.attributes or {}:
            assert allowlist.allowed(key), (event.name, key)
            if event.name == "exception":
                assert key == "exception.type"
    for link in span.links:
        assert all(allowlist.allowed(k) for k in link.attributes or {})
    assert not span.status.description


@SETTINGS
@given(st.lists(spans(), min_size=1, max_size=4))
def test_off_exports_only_allowed_keys(batch: list[ReadableSpan]) -> None:
    inner = Capture()
    assert ContentFilterExporter(inner, mode="off").export(batch) is SpanExportResult.SUCCESS
    assert len(inner.got) == len(batch)
    for before, after in zip(batch, inner.got, strict=True):
        _assert_filtered(after)
        assert after.context == before.context and after.name == before.name
        assert [e.name for e in after.events][: len(before.events)] == [e.name for e in before.events]
        kept = {k: v for k, v in (before.attributes or {}).items() if allowlist.allowed(k)}
        assert {k: v for k, v in (after.attributes or {}).items() if k in kept} == kept


@SETTINGS
@given(st.lists(spans(), min_size=1, max_size=4), st.sampled_from(["off", "redacted", "full"]))
def test_aws_auth_never_leaves(batch: list[ReadableSpan], mode: str) -> None:
    inner = Capture()
    ContentFilterExporter(inner, mode=mode, redact=lambda t: t).export(batch)  # type: ignore[arg-type]
    for span in inner.got:
        assert not [k for k in span.attributes or {} if k.startswith("aws.auth.")]
        for event in span.events:
            assert not [k for k in event.attributes or {} if k.startswith("aws.auth.")]


POISONABLE = ("attributes", "events", "links", "status", "name", "context", "resource", "parent")


def _poisoned(span: ReadableSpan, prop: str) -> ReadableSpan:
    def boom(self: Any) -> Any:
        raise RuntimeError("filter fault")

    cls = type("Poisoned", (ReadableSpan,), {prop: property(boom)})
    bad = cls.__new__(cls)
    bad.__dict__.update(span.__dict__)
    return bad


@st.composite
def mixed_batches(draw: Any) -> tuple[list[ReadableSpan], set[int]]:
    n = draw(st.integers(min_value=1, max_value=5))
    batch: list[ReadableSpan] = []
    poisoned: set[int] = set()
    for i in range(1, n + 1):
        span = draw(spans(span_id=i))
        if draw(st.booleans()):
            span = _poisoned(span, draw(st.sampled_from(POISONABLE)))
            poisoned.add(i)
        batch.append(span)
    return batch, poisoned


def _boom(_: str) -> str:
    raise RuntimeError("redaction fault")


@SETTINGS
@given(mixed_batches(), st.sampled_from(["off", "redacted"]), st.booleans(), st.booleans())
def test_filter_fails_closed(mixed: tuple[list[ReadableSpan], set[int]], mode: str,
                                redact_raises: bool, inner_raises: bool) -> None:
    batch, poisoned = mixed
    inner = Capture(raises=inner_raises)
    flt = ContentFilterExporter(inner, mode=mode, redact=_boom if redact_raises else (lambda t: t))  # type: ignore[arg-type]
    health = ExportHealth(flt, logging.getLogger("test.p3"))
    result = health.export(batch)
    got_ids = {s.context.span_id for s in inner.got}
    assert not got_ids & poisoned
    assert got_ids == set(range(1, len(batch) + 1)) - poisoned
    assert flt.dropped == len(poisoned)
    for span in inner.got:
        assert type(span) is ReadableSpan
        if mode == "off":
            _assert_filtered(span)
            continue
        assert not span.status.description
        for event in span.events:
            if event.name == "exception":
                assert set(event.attributes or {}) <= {"exception.type"}
        maps = [span.attributes or {}, *(e.attributes or {} for e in span.events)]
        for attrs in maps:
            for key, value in attrs.items():
                assert allowlist.allowed(key) or key in allowlist.TEXT_KEYS, key
                if key in allowlist.TEXT_KEYS and redact_raises:
                    # Only a value redact was never called on survives: an empty list of texts.
                    assert value == (), (key, value)
    # A non-empty batch whose every span was dropped by the filter is a failure too.
    failed = inner_raises if inner.got else bool(batch)
    expected = SpanExportResult.FAILURE if failed else SpanExportResult.SUCCESS
    assert result is expected
