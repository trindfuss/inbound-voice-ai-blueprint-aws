"""Front door: telemetry never changes a
webhook's outcome.

For generated slot journeys (the card-confirmation journeys of ``test_frontdoor_card_props``: any
customer, mode, proposed card and sequence of caller turns and resolver outputs, then call end),
every response with telemetry installed and a failure injected (an exporter that raises, or spans
whose every method raises, so ``set_step``, ``add_fields`` and ``child_span`` fail), or with healthy
telemetry, is identical to the untraced run's. Nonces, tokens, Gather ids and confirmation numbers
are seeded so the two runs are comparable.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from owlbank.slots import ResolverOutput
from owlbank.telemetry import frontdoor
from tests.component.test_frontdoor_telemetry import (
    _RaisingExporter,
    _RaisingTracer,
    deterministic,
    settings as telemetry_settings,
)
from tests.property.test_frontdoor_card_props import (
    CARDS,
    PROPS,
    TURNS,
    _answer,
    _call,
    _confirming,
    _harness,
    _level3,
    _propose,
    proposed,
    turns,
)


def _install(failure: str) -> None:
    if failure == "exporter":
        assert frontdoor.install_frontdoor(telemetry_settings(), exporter=_RaisingExporter())
        return
    assert frontdoor.install_frontdoor(telemetry_settings(), exporter=InMemorySpanExporter())
    if failure == "span":
        frontdoor._tracer = _RaisingTracer(frontdoor._tracer)  # noqa: SLF001


def _run(who: str, mode: str, to: str | None, script: list[tuple[str, ResolverOutput]],
         failure: str | None) -> list[str]:
    saved: Any = frontdoor._tracer  # noqa: SLF001
    try:
        with deterministic():
            if failure is not None:
                _install(failure)
            h = _harness(mode)
            with _call(42) as call:
                _level3(h, call, who)
                _propose(h, call, to)
                for name, out in script:
                    if not _confirming(h, call):
                        break
                    _answer(h, call, TURNS[name][0], out)
                h.post("status", call, CallStatus="completed", CallDuration="9")
        return h.bodies
    finally:
        frontdoor._tracer = saved  # noqa: SLF001


@settings(PROPS, max_examples=60)
@given(who=st.sampled_from(sorted(CARDS)), mode=st.sampled_from(["assisted", "deterministic"]),
       to=proposed, script=turns, failure=st.sampled_from(["exporter", "span", "healthy"]))
def test_a_telemetry_failure_leaves_every_response_identical(
        who: str, mode: str, to: str | None, script: list[tuple[str, ResolverOutput]],
        failure: str) -> None:
    assert frontdoor._tracer is None  # noqa: SLF001
    untraced = _run(who, mode, to, script, None)
    traced = _run(who, mode, to, script, failure)
    assert traced == untraced


# ---------------------------------------------------------------- the X-Ray header merge

LINEAGE_VALUE = st.text(alphabet=st.characters(blacklist_characters=";", blacklist_categories=("Cc", "Cs", "Zs")),
                        min_size=1, max_size=40)


@given(value=LINEAGE_VALUE, before=st.sampled_from(["", "Root=1-5759e988-bd862e3fe1be46a994272793;"]),
       sampled=st.sampled_from(["0", "1"]))
@settings(max_examples=50, deadline=None)
def test_the_merged_trace_header_keeps_an_existing_lineage_exactly_once(
        value: str, before: str, sampled: str) -> None:
    """Any ``Lineage=`` value on botocore's header is kept exactly once, verbatim, in the merged
    header of a sampled span; the span's own ``Sampled=1`` is the only sampled field."""
    existing = f"{before}Parent=53995c3f42cd8ad8;Sampled={sampled};Lineage={value}"
    real = frontdoor._tracer
    try:
        assert frontdoor.install_frontdoor(telemetry_settings(), exporter=InMemorySpanExporter())
        with frontdoor.webhook_span("factor", "CA" + "e" * 32):
            merged = frontdoor.lambda_trace_header(existing)
    finally:
        frontdoor._tracer = real
    assert merged is not None
    assert merged.count("Lineage=") == 1 and merged.endswith(f";Lineage={value}")
    assert merged.count("Sampled=") == 1 and ";Sampled=1;" in merged
