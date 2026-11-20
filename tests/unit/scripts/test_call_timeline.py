"""scripts/call_timeline.py: the gap, label and summary logic on fixed rows, and the Logs
Insights flow against a fake client. No AWS calls."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from scripts import call_timeline as ct
from scripts._common import ScriptError

SID = "CA" + "0123456789abcdef" * 2


def row(name: str, start: float, end: float, *, service: str = "owlbank-dev-bridge", span_id: str = "",
        trace_id: str = "t1", **extra: str) -> list[dict[str, str]]:
    fields = {"name": name, "service": service, "start_ms": str(start), "end_ms": str(end),
              "duration_ms": str(end - start), "spanId": span_id, "traceId": trace_id, **extra}
    return [{"field": k, "value": v} for k, v in fields.items()] + [{"field": "@ptr", "value": "xyz"}]


# One call: a session span around two turns, the caller thinking between them.
ROWS = [
    row("bridge.turn", 5_000, 6_000, span_id="b"),
    row("relay.session", 1_000, 20_000, span_id="a"),
    row("bridge.turn", 9_000, 9_500, span_id="d"),
    row("invoke_agent", 5_100, 5_900, span_id="c", service="owlbank-dev-bridge"),
]


def test_timeline_orders_spans_and_computes_offsets_gaps_and_labels():
    spans, skipped = ct.parse_rows(ROWS)
    lines, totals = ct.build_timeline(spans)
    assert skipped == 0
    assert [(ln.span.name, ln.offset_ms, ln.gap_ms, ln.label) for ln in lines] == [
        ("relay.session", 0, None, ""),
        ("bridge.turn", 4_000, 5_000 - 1_000, ct.TWILIO_SIDE),   # from the session's start
        ("invoke_agent", 4_100, 5_100 - 6_000, ct.NESTED),
        ("bridge.turn", 8_000, 9_000 - 6_000, ct.TWILIO_SIDE),  # from the previous turn's end
    ]
    # Ours: the two turns (1,000 + 500 ms; invoke_agent is inside the first). The rest is Twilio-side.
    assert totals == ct.Totals(spans=4, wall_ms=19_000, twilio_side_ms=17_500, our_code_ms=1_500)


def test_a_turn_gap_is_measured_from_the_previous_turn_not_its_last_child():
    spans = [ct.Span("turn", "s", 0, 1_000, 1_000), ct.Span("model", "s", 10, 20, 10),
             ct.Span("turn", "s", 1_500, 2_000, 500)]
    lines, totals = ct.build_timeline(spans)
    assert [(ln.gap_ms, ln.label) for ln in lines] == [(None, ""), (10 - 1_000, ct.NESTED),
                                                        (500, ct.TWILIO_SIDE)]
    assert (totals.twilio_side_ms, totals.our_code_ms) == (500, 1_500)


def test_a_gap_is_start_minus_the_previous_spans_end_and_zero_has_no_label():
    spans = [ct.Span("a", "s", 0, 100, 100), ct.Span("b", "s", 100, 150, 50), ct.Span("c", "s", 400, 410, 10)]
    lines, totals = ct.build_timeline(spans)
    assert [(ln.gap_ms, ln.label) for ln in lines] == [(None, ""), (0, ""), (250, ct.TWILIO_SIDE)]
    assert (totals.wall_ms, totals.twilio_side_ms, totals.our_code_ms) == (410, 250, 160)


def test_equal_starts_put_the_longer_span_first():
    lines, _ = ct.build_timeline([ct.Span("child", "s", 10, 20, 10), ct.Span("parent", "s", 10, 90, 80)])
    assert [ln.span.name for ln in lines] == ["parent", "child"]


def test_no_spans():
    assert ct.build_timeline([]) == ([], ct.Totals(0, 0.0, 0.0, 0.0))


def test_rows_without_usable_times_are_skipped_and_duplicates_kept_once():
    bad = [{"field": "name", "value": "x"}, {"field": "start_ms", "value": "nan"}, {"field": "end_ms", "value": "5"}]
    backwards = row("y", 10, 5)
    dup = row("bridge.turn", 5_000, 6_000, span_id="b")  # also in the other log group
    spans, skipped = ct.parse_rows([*ROWS, bad, backwards, dup])
    assert skipped == 2 and len(spans) == 4


def test_scientific_notation_and_missing_duration():
    r = [{"field": "name", "value": "n"}, {"field": "start_ms", "value": "1.7612E12"},
         {"field": "end_ms", "value": "1761200000250"}]
    (span,), _ = ct.parse_rows([r])
    assert span.start_ms == 1.7612e12 and span.duration_ms == 250


def test_names_are_printable_ascii_only():
    (span,), _ = ct.parse_rows([row("evil\x1b[31m\nname", 0, 1)])
    assert "\x1b" not in span.name and "\n" not in span.name


def test_format_lines_one_line_per_span_and_totals():
    lines, totals = ct.build_timeline(ct.parse_rows(ROWS)[0])
    out = ct.format_lines(lines, totals)
    body = out[1:5]
    assert out[0].split()[:3] == ["offset_ms", "span", "service"]
    assert [b.split()[1] for b in body] == ["relay.session", "bridge.turn", "invoke_agent", "bridge.turn"]
    assert body[3].endswith("twilio-side") and body[2].endswith("nested")
    text = "\n".join(out)
    assert "spans            4" in text and "19,000.0 ms" in text and "17,500.0 ms" in text and "1,500.0 ms" in text


def test_json_shape_has_no_attributes():
    lines, totals = ct.build_timeline(ct.parse_rows(ROWS)[0])
    data = ct.as_json(SID, lines, totals)
    assert data["call_sid"] == SID and data["totals"]["spans"] == 4
    assert set(data["spans"][0]) == {"offset_ms", "name", "service", "start_ms", "end_ms", "duration_ms",
                                     "gap_ms", "label", "span_id", "trace_id"}
    json.dumps(data)


# ----------------------------------------------------------------------------- the query


def test_substitute_quotes_the_session_id_and_refuses_anything_that_could_break_out():
    q = 'filter `attributes.session.id` = {{session_id}}\n| limit 5'
    assert ct.substitute(q, SID) == f'filter `attributes.session.id` = "{SID}"\n| limit 5'
    assert ct.substitute(q, "bench-abc.1_2")
    for bad in ('CA" or 1=1 | "', "CA 1", "", "x" * 81, "CA\n1"):
        with pytest.raises(ScriptError):
            ct.substitute(q, bad)
    with pytest.raises(ScriptError, match="no {{session_id}}"):
        ct.substitute("fields @timestamp", SID)


class FakeLogs:
    def __init__(self, *, definitions: list[dict[str, Any]] | None = None, results: list[dict[str, Any]] | None = None,
                 pages: bool = False) -> None:
        self.definitions = definitions if definitions is not None else [{
            "name": ct.QUERY_NAME, "queryString": "filter x = {{session_id}}",
            "logGroupNames": ["/owlbank/traces/bridge", "aws/spans"]}]
        self.results = results if results is not None else [{"status": "Complete", "results": ROWS}]
        self.pages = pages
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def describe_query_definitions(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("describe", kw))
        if self.pages and "nextToken" not in kw:  # a near-match first, the exact one on page two
            return {"queryDefinitions": [{"name": ct.QUERY_NAME + " (old)"}], "nextToken": "p2"}
        return {"queryDefinitions": self.definitions}

    def start_query(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("start", kw))
        return {"queryId": "q1"}

    def get_query_results(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("get", kw))
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]

    def stop_query(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("stop", kw))
        return {}


NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


def run(argv: list[str], logs: FakeLogs, capsys) -> tuple[int, str, str]:
    code = ct.main(argv, logs=logs, now=NOW, sleep=lambda _s: None)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_main_runs_the_saved_query_for_the_call_and_prints_the_timeline(capsys):
    logs = FakeLogs(results=[{"status": "Running"}, {"status": "Complete", "results": ROWS}], pages=True)
    code, out, _ = run(["--call-sid", SID], logs, capsys)
    assert code == 0 and "relay.session" in out and "twilio-side" in out
    start = next(kw for name, kw in logs.calls if name == "start")
    assert start["queryString"] == f'filter x = "{SID}"'
    assert start["logGroupNames"] == ["/owlbank/traces/bridge", "aws/spans"]
    assert start["endTime"] - start["startTime"] == 24 * 3600 + 1  # the default window: 24 h
    assert [name for name, _ in logs.calls].count("get") == 2
    assert [kw for name, kw in logs.calls if name == "describe"][0]["queryDefinitionNamePrefix"] == ct.QUERY_NAME


def test_main_json_and_hours_and_since(capsys):
    logs = FakeLogs()
    code, out, _ = run(["--call-sid", SID, "--json", "--hours", "2"], logs, capsys)
    data = json.loads(out)
    assert code == 0 and data["totals"]["spans"] == 4
    start = next(kw for name, kw in logs.calls if name == "start")
    assert start["endTime"] - start["startTime"] == 2 * 3600 + 1
    logs = FakeLogs()
    run(["--call-sid", SID, "--since", "2026-10-23T11:00:00Z"], logs, capsys)
    start = next(kw for name, kw in logs.calls if name == "start")
    assert start["startTime"] == int(datetime(2026, 10, 23, 11, tzinfo=UTC).timestamp())


def test_main_no_spans_exits_1(capsys):
    code, out, _ = run(["--call-sid", SID], FakeLogs(results=[{"status": "Complete", "results": []}]), capsys)
    assert code == 1 and "no spans" in out


@pytest.mark.parametrize(("logs", "message"), [
    (FakeLogs(definitions=[]), "no saved Logs Insights query"),
    (FakeLogs(results=[{"status": "Failed"}]), "status Failed"),
    (FakeLogs(definitions=[{"name": ct.QUERY_NAME, "queryString": "x = {{session_id}}"}]), "names no log groups"),
])
def test_main_errors_exit_2_with_a_message(logs, message, capsys):
    code, out, err = run(["--call-sid", SID], logs, capsys)
    assert code == 2 and message in err and out == ""


def test_main_refuses_a_bad_call_sid_before_querying(capsys):
    logs = FakeLogs()
    code, _, err = run(["--call-sid", 'CA"x'], logs, capsys)
    assert code == 2 and "--call-sid" in err and not any(n == "start" for n, _ in logs.calls)


def test_a_query_that_never_finishes_times_out_and_is_stopped():
    logs = FakeLogs(results=[{"status": "Running"}])
    ticks = iter(range(0, 10_000, 40))
    with pytest.raises(ScriptError, match="did not finish"):
        ct.run_query(logs, query="q", log_groups=["g"], start=NOW, end=NOW, sleep=lambda _s: None,
                     clock=lambda: float(next(ticks)))
    assert ("stop", {"queryId": "q1"}) in logs.calls


def test_window_errors():
    with pytest.raises(ScriptError, match="ISO 8601"):
        ct.window("yesterday", None, now=NOW)
    with pytest.raises(ScriptError, match="empty"):
        ct.window("2026-10-24T00:00:00Z", None, now=NOW)
    start, end = ct.window("2026-10-23T09:00:00", None, now=NOW)  # no offset: UTC
    assert start.tzinfo is not None and end == NOW


def test_since_and_hours_are_exclusive():
    with pytest.raises(SystemExit):
        ct.build_parser().parse_args(["--call-sid", SID, "--since", "2026-10-23T00:00:00Z", "--hours", "1"])


def test_aws_errors_print_the_code_and_the_sso_hint_but_never_the_message(capsys):
    class Expired(Exception):
        response = {"Error": {"Code": "ExpiredTokenException"}}

    class Logs:
        def get_paginator(self, _name):
            raise Expired("token for user 94105 expired")

        def describe_query_definitions(self, **_k):
            raise Expired("token for user 94105 expired")

    code = ct.main(["--call-sid", SID], logs=Logs())
    err = capsys.readouterr().err
    assert code == 2 and "ExpiredTokenException" in err and "refresh your AWS credentials" in err and "94105" not in err
