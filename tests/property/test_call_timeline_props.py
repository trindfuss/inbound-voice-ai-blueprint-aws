"""scripts/call_timeline.py: properties of the gap and summary logic, over any list of spans."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from scripts import call_timeline as ct


@st.composite
def spans(draw: st.DrawFn) -> ct.Span:
    start = draw(st.integers(0, 10_000_000))
    length = draw(st.integers(0, 600_000))
    return ct.Span(name=draw(st.sampled_from(["relay.session", "bridge.turn", "invoke_agent", "tool"])),
                   service="s", start_ms=float(start), end_ms=float(start + length), duration_ms=float(length))


SPANS = st.lists(spans(), min_size=1, max_size=40)


def _union(intervals):
    """Brute force on a 1 ms grid (spans are whole milliseconds here)."""
    covered = set()
    for start, end in intervals:
        covered.update(range(int(start), int(end)))
    return float(len(covered))


@given(SPANS)
def test_our_time_is_the_union_of_non_session_spans_and_twilio_side_the_rest(items):
    """Time in our code = the length of the union of the non-session spans (nested spans counted
    once); Twilio-side = the rest of the call's wall time; both within it."""
    lines, totals = ct.build_timeline(items)
    assert totals.spans == len(items) == len(lines)
    work = [(s.start_ms, s.end_ms) for s in items if s.name not in ct.CONTAINER_SPANS]
    assert totals.our_code_ms == _union(work)
    assert totals.wall_ms == max(s.end_ms for s in items) - min(s.start_ms for s in items)
    assert totals.twilio_side_ms == totals.wall_ms - totals.our_code_ms
    assert 0 <= totals.our_code_ms <= totals.wall_ms


@given(SPANS)
def test_offsets_are_non_decreasing_start_at_zero_and_gaps_follow_the_latest_end(items):
    """Each gap is the start minus the latest end of the spans before it (a session span counts
    only by its start), and the label follows the sign."""
    lines, _ = ct.build_timeline(items)
    offsets = [ln.offset_ms for ln in lines]
    assert offsets == sorted(offsets) and offsets[0] == 0
    assert lines[0].gap_ms is None and lines[0].label == ""
    for i, ln in enumerate(lines[1:], start=1):
        latest = max(p.span.start_ms if p.span.name in ct.CONTAINER_SPANS else p.span.end_ms
                     for p in lines[:i])
        assert ln.gap_ms == ln.span.start_ms - latest
        assert ln.label == (ct.TWILIO_SIDE if ln.gap_ms > 0 else ct.NESTED if ln.gap_ms < 0 else "")


@given(SPANS)
def test_the_order_of_the_input_does_not_matter(items):
    assert ct.build_timeline(items) == ct.build_timeline(list(reversed(items)))
