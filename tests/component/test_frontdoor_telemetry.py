"""Front-door spans.

Signed fake webhooks into the real front door (``test_frontdoor_support.build``) with telemetry
installed against an in-memory exporter behind the real content filter and ``ExportHealth``
(``install_frontdoor(settings, exporter=...)``), or against the real UDP exporter on a fake socket:

* one root span per authenticated webhook (none for the 401 challenge), named by its last step across a
  verify -> card -> read-back -> yes journey, every span carrying ``session.id`` and ``call_sid``;
* the ``factor_answer`` fields and ``timings_ms`` as ``owlbank.*`` attributes, never ``value`` or
  ``speech``, and no factor value anywhere in the exported spans (:mod:`tests.leakscan`);
* ``lambda.invoke <op>`` children with function, invocation type and outcome, and the
  ``X-Amzn-Trace-Id`` header ``FakeLambdaClient`` saw pointing at them;
* the Lambda's own trace ID as ``aws.xray.lambda_trace_id``;
* an exporter or a span that raises, and a UDP socket that fails, leave every response identical.
"""

from __future__ import annotations

import base64
import random
import secrets
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from owlbank import sensitive
from owlbank.segments import clear_sensitive
from owlbank.telemetry import frontdoor
from owlbank.telemetry.exporters import UDP_HEADER, UDP_SIGNAL_PREFIX, UdpSpanExporter
from owlbank.telemetry.settings import TelemetrySettings
from tests.component.test_frontdoor_card import ASSISTED, build_h, propose, speak, to_level3
from tests.component.test_frontdoor_support import Harness, all_fixture_factor_values, sid
from tests.leakscan import compile_values, leaks

ON = {"OWLBANK_CALL_OBSERVABILITY": "on", "OWLBANK_STAGE": "dev"}
LAMBDA_ROOT = "1-5759e988-bd862e3fe1be46a994272793"


def settings() -> TelemetrySettings:
    found = TelemetrySettings.from_env(ON, service="frontdoor")
    assert found is not None
    return found


@pytest.fixture(autouse=True)
def _untraced_after(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test starts and ends with telemetry not installed (``_tracer`` restored to ``None``)."""
    monkeypatch.setattr(frontdoor, "_tracer", None)
    yield
    for n in range(8):
        clear_sensitive(sid(n))
        sensitive.forget(sid(n))


def install(exporter: SpanExporter) -> None:
    assert frontdoor.install_frontdoor(settings(), exporter=exporter)


@contextmanager
def deterministic() -> Iterator[None]:
    """Seeded nonces, tokens, Gather ids and confirmation numbers, so two runs of one journey give
    the same TwiML."""
    rng = random.Random(7)
    real = secrets.token_urlsafe, secrets.choice, secrets.randbelow
    secrets.token_urlsafe = lambda n=32: base64.urlsafe_b64encode(rng.randbytes(n)).rstrip(b"=").decode()
    secrets.choice = rng.choice
    secrets.randbelow = rng.randrange
    try:
        yield
    finally:
        secrets.token_urlsafe, secrets.choice, secrets.randbelow = real


def journey(h: Harness, call: str) -> None:
    """Verify (ZIP, then step-up SSN-4 and code), the AI proposes a payment with no card, an unclear
    card answer, the card named (read back), a keypad yes (the payment runs), call end."""
    to_level3(h, call)
    propose(h, call, absent=True)
    speak(h, call, "hmm")
    speak(h, call, "the OwlBank Red Card")
    h.xml("factor", call, Digits="1")
    h.post("status", call, CallStatus="completed", CallDuration="42")


def roots(spans: list[ReadableSpan]) -> list[ReadableSpan]:
    return [s for s in spans if s.parent is None]


def children_of(spans: list[ReadableSpan], parent: ReadableSpan) -> list[ReadableSpan]:
    return [s for s in spans if s.parent is not None and s.parent.span_id == parent.context.span_id]


def strings_of(spans: list[ReadableSpan]) -> list[str]:
    out: list[str] = []
    for s in spans:
        out.append(s.name)
        out.extend(f"{k}={v}" for k, v in (s.attributes or {}).items())
        for e in s.events:
            out.append(e.name)
            out.extend(f"{k}={v}" for k, v in (e.attributes or {}).items())
    return out


# --------------------------------------------------------------------------- one span per webhook


def test_one_root_span_per_webhook_named_by_step_across_a_verify_card_readback_journey() -> None:
    exporter = InMemorySpanExporter()
    install(exporter)
    h = build_h(ASSISTED)
    call = sid(1)
    journey(h, call)
    spans = list(exporter.get_finished_spans())
    top = roots(spans)
    # One span per authenticated webhook; Twilio's 401 challenge before each one gets none.
    assert len(top) == len(h.bodies)
    named = [s.name for s in top]
    assert named[:2] == ["voice.incoming", "voice.language"]
    assert named.count("verify") == 3               # ZIP, SSN-4, code
    assert "handoff.stepup" in named and "handoff.confirm" in named
    tail = named[named.index("handoff.confirm"):]
    assert tail == ["handoff.confirm", "gather.card", "confirm.readback", "gather.yes_no", "call.end"]
    for s in top:
        attrs = s.attributes or {}
        assert attrs["session.id"] == attrs["call_sid"] == call
        assert attrs["owlbank.route"] == s.name.removeprefix("voice.") or attrs.get("owlbank.step") == s.name
    assert len({s.context.trace_id for s in top}) == len(top)   # each webhook its own trace
    assert all(s.resource.attributes["service.name"] == "owlbank-dev-frontdoor" for s in spans)


def test_factor_answer_fields_and_timings_are_on_the_span_never_a_value_or_speech() -> None:
    exporter = InMemorySpanExporter()
    install(exporter)
    h = build_h(ASSISTED)
    call = sid(2)
    journey(h, call)
    spans = list(exporter.get_finished_spans())
    card = next(s for s in spans if s.name == "confirm.readback")
    attrs = dict(card.attributes or {})
    assert attrs["owlbank.factor"] == "card" and attrs["owlbank.flow"] == "confirm"
    assert attrs["owlbank.match"] == "exact" and attrs["owlbank.result_kind"] == "value"
    assert attrs["owlbank.speech_present"] is True and attrs["owlbank.tries"] == 1
    for name in ("policy", "catalog", "state_load", "resolve", "render", "state_save"):
        assert isinstance(attrs[f"owlbank.timing.{name}_ms"], int)
    zip_span = next(s for s in spans if s.name == "verify")
    assert zip_span.attributes["owlbank.factor"] == "zip"
    assert zip_span.attributes["owlbank.digit_count"] == 5
    keys = {k for s in spans for k in (s.attributes or {})}
    assert not {k for k in keys if k.rsplit(".", 1)[-1] in ("value", "speech")}
    assert "owlbank.call_sid" not in keys
    patterns = compile_values(v for v in all_fixture_factor_values() | h.sent_factors if len(v) >= 4)
    for text in strings_of(spans):
        assert not leaks(patterns, text), text
    assert not any("Red Card" in t or "hmm" in t for t in strings_of(spans))   # no speech text


def test_add_fields_drops_value_and_speech_whatever_the_caller_passes() -> None:
    exporter = InMemorySpanExporter()
    install(exporter)
    with frontdoor.webhook_span("factor", "CA" + "f" * 32):
        frontdoor.add_fields(value="94105", speech="nine four", factor="zip", tries=None,
                             nested={"a": 1})
    (span,) = exporter.get_finished_spans()
    assert {k for k in span.attributes if k.startswith("owlbank.")} == {"owlbank.route", "owlbank.factor"}


# --------------------------------------------------------------------------- downstream Lambdas


def test_lambda_invokes_are_children_and_carry_the_trace_header() -> None:
    exporter = InMemorySpanExporter()
    install(exporter)
    h = build_h(ASSISTED)
    call = sid(3)
    journey(h, call)
    spans = list(exporter.get_finished_spans())
    invokes = [s for s in spans if s.name.startswith("lambda.invoke ")]
    assert [s.name.removeprefix("lambda.invoke ") for s in invokes] == h.lam.ops()
    assert len(h.lam.trace_headers) == len(invokes)
    by_id = {s.context.span_id: s for s in spans}
    for span, header, kind in zip(invokes, h.lam.trace_headers, h.lam.invocation_types, strict=True):
        parent = by_id[span.parent.span_id]
        assert parent.parent is None and span.context.trace_id == parent.context.trace_id
        attrs = span.attributes or {}
        assert attrs["owlbank.function"] in ("owlbank-auth", "owlbank-bank")
        assert attrs["owlbank.invocation_type"] == kind
        assert attrs["owlbank.outcome"] in ("ok", "accepted")
        assert attrs["session.id"] == attrs["call_sid"] == call
        assert set(attrs) == {"owlbank.function", "owlbank.invocation_type", "owlbank.outcome",
                              "session.id", "call_sid"}          # no payload, token or result
        trace_hex = format(span.context.trace_id, "032x")
        assert header == (f"Root=1-{trace_hex[:8]}-{trace_hex[8:]};"
                          f"Parent={format(span.context.span_id, '016x')};Sampled=1")
    pay = next(s for s in invokes if s.name == "lambda.invoke make_payment")
    assert by_id[pay.parent.span_id].name == "gather.yes_no"
    assert children_of(spans, by_id[pay.parent.span_id]) == [pay]


def test_failed_invokes_record_their_outcome() -> None:
    exporter = InMemorySpanExporter()
    install(exporter)
    h = build_h(ASSISTED)
    h.lam.fail_ops.add("ani_lookup")
    call = sid(4)
    h.start(call, "+12025550101")
    lookups = [s for s in exporter.get_finished_spans() if s.name == "lambda.invoke ani_lookup"]
    assert lookups and {s.attributes["owlbank.outcome"] for s in lookups} == {"error"}


def test_no_trace_header_and_no_spans_when_not_installed() -> None:
    h = build_h(ASSISTED)
    journey(h, sid(5))
    assert h.lam.trace_headers and set(h.lam.trace_headers) == {None}
    assert frontdoor.lambda_trace_header() is None


def test_the_lambda_trace_id_is_an_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("_X_AMZN_TRACE_ID", f"Root={LAMBDA_ROOT};Parent=53995c3f42cd8ad8;Sampled=1")
    exporter = InMemorySpanExporter()
    install(exporter)
    h = build_h(ASSISTED)
    h.start(sid(6), "+12025550101")
    top = roots(list(exporter.get_finished_spans()))
    assert top and all(s.attributes["aws.xray.lambda_trace_id"] == LAMBDA_ROOT for s in top)


# --------------------------------------------------------------------------- UDP and failures


class _FakeSocket:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[tuple[bytes, tuple[str, int]]] = []

    def sendto(self, data: bytes, addr: tuple[str, int]) -> int:
        if self.fail:
            raise OSError("simulated sendto failure")
        self.sent.append((data, addr))
        return len(data)

    def close(self) -> None:
        pass


def test_spans_go_out_as_udp_datagrams_to_the_daemon() -> None:
    sock = _FakeSocket()
    install(UdpSpanExporter("127.0.0.1:2000", sock=sock))
    h = build_h(ASSISTED)
    call = sid(1)
    h.start(call, "+12025550101")
    assert sock.sent and {addr for _, addr in sock.sent} == {("127.0.0.1", 2000)}
    names = []
    for data, _ in sock.sent:
        prefix = UDP_HEADER + UDP_SIGNAL_PREFIX
        assert data.startswith(prefix)
        req = ExportTraceServiceRequest.FromString(base64.b64decode(data[len(prefix):]))
        names += [s.name for rs in req.resource_spans for ss in rs.scope_spans for s in ss.spans]
    assert "voice.incoming" in names and "lambda.invoke ani_lookup" in names


class _RaisingExporter(SpanExporter):
    def export(self, spans: Any) -> SpanExportResult:
        raise RuntimeError("exporter down")


class _RaisingSpan:
    """A span whose every method raises (the ``set_step`` / ``add_fields`` paths fail)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def get_span_context(self) -> Any:
        return self._inner.get_span_context()

    def is_recording(self) -> bool:
        return True

    def end(self, *a: Any, **k: Any) -> None:
        self._inner.end()
        raise RuntimeError("end failed")

    def __getattr__(self, name: str) -> Callable[..., Any]:
        def boom(*a: Any, **k: Any) -> Any:
            raise RuntimeError(f"{name} failed")
        return boom


class _RaisingTracer:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def start_span(self, *a: Any, **k: Any) -> Any:
        return _RaisingSpan(self._inner.start_span(*a, **k))


def _bodies(setup: Callable[[], None]) -> list[str]:
    with deterministic():
        setup()
        h = build_h(ASSISTED)
        journey(h, sid(7))
    clear_sensitive(sid(7))
    sensitive.forget(sid(7))
    return h.bodies


@pytest.mark.parametrize("failure", ["exporter", "span", "udp", "tracer"])
def test_a_telemetry_failure_leaves_every_response_identical(
        failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    untraced = _bodies(lambda: None)

    def setup() -> None:
        if failure == "exporter":
            install(_RaisingExporter())
        elif failure == "udp":
            install(UdpSpanExporter("127.0.0.1:2000", sock=_FakeSocket(fail=True)))
        else:
            install(InMemorySpanExporter())
            if failure == "span":
                monkeypatch.setattr(frontdoor, "_tracer", _RaisingTracer(frontdoor._tracer))
            else:
                monkeypatch.setattr(frontdoor, "_tracer", object())   # start_span itself fails

    assert _bodies(setup) == untraced
