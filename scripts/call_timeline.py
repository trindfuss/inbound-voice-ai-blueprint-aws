"""scripts/call_timeline.py: every span of one call in time order, with the gaps between them.

Looks up the saved Logs Insights query "OwlBank/Call timeline" (the stack saves it over the traces log
group and ``aws/spans``, call observability), puts the CallSid in place of its ``{{session_id}}``
parameter, runs it and prints one line per span in start order: the offset from the first span (ms), the
span name, its service, its duration, and the gap before it. The gap is the span's start minus the
**latest end of every span before it** (in start order), computed here from each row's ``start_ms``
and ``end_ms``, not from the query's own ``prev_end_ms`` (Logs Insights can't compute on
``autoregress`` output). Measuring from the latest end, not the previous span's, keeps nested
spans right: a turn's gap is measured from the end of the previous turn, not of its last child.

* A **positive** gap is time outside our code: Twilio's speech recognition, playback, or waiting for
  the caller. It is labelled ``twilio-side``.
* A **negative** (or zero) gap means a span before it was still open: it is nested in or overlaps it
  (a child span, a parallel tool call). Labelled ``nested``.

``relay.session`` frames the whole WebSocket session (speech recognition and playback included), so
only its start counts as the end of our work (``CONTAINER_SPANS``). A positive gap is then time when
none of our other spans was open. The totals are the span count, the call's wall time (first start to
last end), time in our code (the length of the union of the non-session spans) and Twilio-side time
(the rest of the wall time, the session's tail after the last turn included).

Output is span names, services, timings and IDs only. Span attributes are never selected for display,
so no conversation text or caller data can appear (the query returns none either).

Usage (repository root; the SSO profile of the deployed stack)::

    uv run python scripts/call_timeline.py --profile <profile> --call-sid CA0123...
    uv run python scripts/call_timeline.py --call-sid bench-1a2b... --hours 2 --json
    uv run python scripts/call_timeline.py --call-sid CA0123... --since 2026-10-23T14:00:00Z
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):  # run as a file: make `scripts` importable
    sys.path.insert(0, str(ROOT))

from scripts._common import DEFAULT_REGION, ScriptError  # noqa: E402

#: The saved query's name (``infra/owlbank_infra/call_observability.py``: ``CALL_TIMELINE_QUERY_NAME``).
QUERY_NAME = "OwlBank/Call timeline"
PLACEHOLDER = "{{session_id}}"
#: A CallSid (``CA`` and 32 hex) or a bench session ID (``bench-...``): letters, digits, ``-``, ``_``, ``.``
#: only, so the value can't break out of the query's double quotes.
SESSION_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,80}")
DEFAULT_HOURS = 24.0
QUERY_LIMIT = 10_000
POLL_INTERVAL_S = 1.0
QUERY_TIMEOUT_S = 120.0

#: Spans that only frame a session (open for the whole WebSocket session, Twilio's speech recognition
#: and playback included): they get a line, but only their start counts as the end of our work, so
#: the gaps between turns stay visible.
CONTAINER_SPANS = frozenset({"relay.session"})
TWILIO_SIDE = "twilio-side"
NESTED = "nested"
_PRINTABLE = re.compile(r"[^\x20-\x7e]")


# --------------------------------------------------------------------------- the pure part


@dataclass(frozen=True)
class Span:
    """One row of the query: a span's name, service, timings (ms since the epoch) and IDs."""

    name: str
    service: str
    start_ms: float
    end_ms: float
    duration_ms: float
    span_id: str = ""
    trace_id: str = ""


@dataclass(frozen=True)
class Line:
    """A span placed on the timeline."""

    span: Span
    offset_ms: float  # span start minus the first span's start
    gap_ms: float | None  # start minus the previous span's end; ``None`` for the first span
    label: str  # ``twilio-side`` (gap > 0), ``nested`` (gap < 0) or ``""``


@dataclass(frozen=True)
class Totals:
    spans: int
    wall_ms: float  # first start to the last end
    twilio_side_ms: float  # wall time minus time in our code
    our_code_ms: float  # the length of the union of the non-session spans


def label_for(gap_ms: float | None) -> str:
    if gap_ms is None or gap_ms == 0:
        return ""
    return TWILIO_SIDE if gap_ms > 0 else NESTED


def build_timeline(spans: Iterable[Span]) -> tuple[list[Line], Totals]:
    """Sort by start (a longer span first when starts are equal, so a parent precedes its child; then name and span ID, so the order is
    deterministic), then
    give each span its offset, its gap from the latest end of the spans before it and its label."""
    ordered = sorted(spans, key=lambda s: (s.start_ms, -s.end_ms, s.name, s.span_id))
    if not ordered:
        return [], Totals(0, 0.0, 0.0, 0.0)
    first = ordered[0].start_ms
    lines: list[Line] = []
    previous_end: float | None = None
    for s in ordered:
        gap = None if previous_end is None else s.start_ms - previous_end
        lines.append(Line(s, s.start_ms - first, gap, label_for(gap)))
        end = s.start_ms if s.name in CONTAINER_SPANS else s.end_ms
        previous_end = end if previous_end is None else max(previous_end, end)
    wall = max(s.end_ms for s in ordered) - first
    ours = _union_ms([(s.start_ms, s.end_ms) for s in ordered if s.name not in CONTAINER_SPANS])
    return lines, Totals(len(ordered), wall, wall - ours, ours)


def _union_ms(intervals: list[tuple[float, float]]) -> float:
    """The length of the union of ``intervals``: time when at least one of our spans was open."""
    total, open_start, open_end = 0.0, None, None
    for start, end in sorted(intervals):
        if open_end is None or start > open_end:
            if open_end is not None:
                total += open_end - open_start
            open_start, open_end = start, end
        else:
            open_end = max(open_end, end)
    if open_end is not None:
        total += open_end - open_start
    return total


def _number(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None  # not NaN or infinite


def parse_rows(rows: Iterable[Sequence[Mapping[str, str]]]) -> tuple[list[Span], int]:
    """Spans from a ``GetQueryResults`` ``results`` list (each row a list of ``{field, value}``).

    Returns ``(spans, skipped)``. A row without a usable ``start_ms`` and ``end_ms`` is skipped; a span
    seen twice (same trace and span ID, as when both log groups hold it) is kept once. A missing
    ``duration_ms`` is ``end_ms - start_ms``."""
    spans: list[Span] = []
    seen: set[tuple[str, str]] = set()
    skipped = 0
    for row in rows:
        f = {c.get("field", ""): c.get("value", "") for c in row}
        start, end = _number(f.get("start_ms")), _number(f.get("end_ms"))
        if start is None or end is None or end < start:
            skipped += 1
            continue
        key = (f.get("traceId", ""), f.get("spanId", ""))
        if key[0] and key[1]:
            if key in seen:
                continue
            seen.add(key)
        duration = _number(f.get("duration_ms"))
        spans.append(Span(name=_clean(f.get("name")) or "(unnamed)", service=_clean(f.get("service")) or "-",
                          start_ms=start, end_ms=end, duration_ms=end - start if duration is None else duration,
                          span_id=_clean(f.get("spanId")), trace_id=_clean(f.get("traceId"))))
    return spans, skipped


def _clean(value: str | None) -> str:
    """A field value safe to print: printable ASCII only (a span name can't carry a terminal escape)."""
    return _PRINTABLE.sub("?", value or "")


def substitute(query: str, session_id: str) -> str:
    """The saved query with ``{{session_id}}`` replaced by the quoted session ID."""
    if not SESSION_ID_RE.fullmatch(session_id):
        raise ScriptError("--call-sid may hold only letters, digits, '.', '_' and '-' (80 characters at most)")
    if PLACEHOLDER not in query:
        raise ScriptError(f"the saved query {QUERY_NAME!r} has no {PLACEHOLDER} parameter")
    return query.replace(PLACEHOLDER, f'"{session_id}"')


# --------------------------------------------------------------------------- output


def _ms(x: float | None, width: int = 10, *, signed: bool = False) -> str:
    if x is None:
        return "-".rjust(width)
    return f"{x:+,.1f}".rjust(width) if signed else f"{x:,.1f}".rjust(width)


def format_lines(lines: Sequence[Line], totals: Totals) -> list[str]:
    name_w = max([len("span")] + [len(ln.span.name) for ln in lines])
    svc_w = max([len("service")] + [len(ln.span.service) for ln in lines])
    out = [f"{'offset_ms':>10}  {'span':<{name_w}}  {'service':<{svc_w}}  {'duration_ms':>11}  {'gap_ms':>10}  label"]
    for ln in lines:
        out.append(f"{_ms(ln.offset_ms)}  {ln.span.name:<{name_w}}  {ln.span.service:<{svc_w}}  "
                   f"{_ms(ln.span.duration_ms, 11)}  {_ms(ln.gap_ms, signed=True)}  {ln.label}".rstrip())
    out += ["",
            f"spans            {totals.spans}",
            f"call wall time   {totals.wall_ms:,.1f} ms",
            f"Twilio-side time {totals.twilio_side_ms:,.1f} ms (no span of ours open: speech recognition, "
            "playback, waiting for the caller)",
            f"time in our code {totals.our_code_ms:,.1f} ms (at least one non-session span open)"]
    return out


def as_json(session_id: str, lines: Sequence[Line], totals: Totals, *, skipped: int = 0) -> dict[str, Any]:
    return {
        "call_sid": session_id,
        "spans": [{"offset_ms": ln.offset_ms, "name": ln.span.name, "service": ln.span.service,
                   "start_ms": ln.span.start_ms, "end_ms": ln.span.end_ms, "duration_ms": ln.span.duration_ms,
                   "gap_ms": ln.gap_ms, "label": ln.label, "span_id": ln.span.span_id,
                   "trace_id": ln.span.trace_id} for ln in lines],
        "totals": asdict(totals),
        "skipped_rows": skipped,
    }


# --------------------------------------------------------------------------- Logs Insights


def find_query_definition(logs: Any, name: str = QUERY_NAME) -> Mapping[str, Any]:
    """The saved query named exactly ``name`` (looked up by name prefix, then matched exactly)."""
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"queryDefinitionNamePrefix": name}
        if token:
            kwargs["nextToken"] = token
        resp = logs.describe_query_definitions(**kwargs)
        for qd in resp.get("queryDefinitions", []):
            if qd.get("name") == name:
                return qd
        token = resp.get("nextToken")
        if not token:
            raise ScriptError(f"no saved Logs Insights query named {name!r} in this Region and account; "
                              "is the stack deployed with callObservability=on, and is --region right?")


def run_query(logs: Any, *, query: str, log_groups: Sequence[str], start: datetime, end: datetime,
              sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
              poll_s: float = POLL_INTERVAL_S, timeout_s: float = QUERY_TIMEOUT_S) -> list[list[dict[str, str]]]:
    """StartQuery, then GetQueryResults until it completes; the result rows."""
    started = logs.start_query(logGroupNames=list(log_groups), startTime=int(start.timestamp()),
                               endTime=int(end.timestamp()) + 1, queryString=query, limit=QUERY_LIMIT)
    query_id = started["queryId"]
    deadline = clock() + timeout_s
    while True:
        resp = logs.get_query_results(queryId=query_id)
        status = resp.get("status")
        if status == "Complete":
            return resp.get("results", [])
        if status in ("Failed", "Cancelled", "Timeout", "Unknown"):
            raise ScriptError(f"the Logs Insights query ended with status {status}")
        if clock() >= deadline:
            with contextlib.suppress(Exception):
                logs.stop_query(queryId=query_id)
            raise ScriptError(f"the Logs Insights query did not finish within {timeout_s:.0f} s")
        sleep(poll_s)


def fetch_spans(logs: Any, session_id: str, *, start: datetime, end: datetime,
                sleep: Callable[[float], None] = time.sleep,
                clock: Callable[[], float] = time.monotonic) -> tuple[list[Span], int]:
    definition = find_query_definition(logs)
    groups = definition.get("logGroupNames") or []
    if not groups:
        raise ScriptError(f"the saved query {QUERY_NAME!r} names no log groups")
    rows = run_query(logs, query=substitute(str(definition.get("queryString", "")), session_id),
                     log_groups=groups, start=start, end=end, sleep=sleep, clock=clock)
    return parse_rows(rows)


# --------------------------------------------------------------------------- CLI


def window(since: str | None, hours: float | None, *, now: datetime | None = None) -> tuple[datetime, datetime]:
    """``(start, end)``: ``--since`` (an ISO 8601 time, UTC if it has no offset) until now, or the last
    ``--hours`` (default 24)."""
    end = now or datetime.now(UTC)
    if since:
        try:
            start = datetime.fromisoformat(since.strip().replace("Z", "+00:00"))
        except ValueError:
            raise ScriptError(f"--since {since!r} is not an ISO 8601 time, e.g. 2026-10-23T14:00:00Z") from None
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
    else:
        start = end - timedelta(hours=DEFAULT_HOURS if hours is None else hours)
    if start >= end:
        raise ScriptError("the time window is empty: the start is not before now")
    return start, end


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--call-sid", required=True, metavar="CA...",
                    help="the call's SID (its session.id); a bench session ID (bench-...) works too")
    ap.add_argument("--profile", default=None, help="AWS profile (default: the environment's)")
    ap.add_argument("--region", default=DEFAULT_REGION, help=f"deployment Region (default {DEFAULT_REGION})")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--since", default=None, metavar="TIME", help="search from this ISO 8601 time until now")
    g.add_argument("--hours", type=float, default=None,
                   help=f"search the last N hours (default {DEFAULT_HOURS:g})")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    return ap


def _client(profile: str | None, region: str) -> Any:
    import boto3  # noqa: PLC0415

    return boto3.Session(profile_name=profile, region_name=region).client("logs")


def main(argv: list[str] | None = None, *, logs: Any = None, now: datetime | None = None,
         sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.hours is not None and args.hours <= 0:
        ap.error("--hours must be positive")
    try:
        start, end = window(args.since, args.hours, now=now)
        spans, skipped = fetch_spans(logs or _client(args.profile, args.region), args.call_sid,
                                     start=start, end=end, sleep=sleep, clock=clock)
    except ScriptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - an AWS error: its code and class, not its message
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        print(f"error: AWS call failed: {code or type(exc).__name__}"
              + ("; if it is an expired SSO token, refresh your AWS credentials" if code else ""), file=sys.stderr)
        return 2
    lines, totals = build_timeline(spans)
    if args.json:
        print(json.dumps(as_json(args.call_sid, lines, totals, skipped=skipped), indent=1))
    elif not lines:
        print(f"no spans for {args.call_sid} between {start:%Y-%m-%d %H:%M}Z and {end:%Y-%m-%d %H:%M}Z "
              "(spans reach the log groups within a minute or two; check --hours and --region)")
    else:
        print("\n".join(format_lines(lines, totals)))
    if skipped:
        print(f"note: {skipped} row(s) without usable start and end times were skipped", file=sys.stderr)
    return 0 if lines else 1


if __name__ == "__main__":
    raise SystemExit(main())
