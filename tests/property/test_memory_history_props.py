"""Segment-traits property: over generated profiles (real and demo items already on them, lists
paged, summaries lagging), ``--apply`` then ``--clear --apply`` leaves no ``owlbank-demo`` item and
every other item untouched, and no output or log record carries a credential."""

from __future__ import annotations

import contextlib
import io
import logging
import tempfile
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from scripts import memory_history as mh
from tests.unit.scripts.test_memory_history_script import (
    API_SECRET,
    API_SID,
    ARN,
    Clock,
    HistoryStub,
    SecretsStub,
    setup_file,
)

EMAIL = "thomas.doe@owlbank-demo.example"
SOURCES = st.sampled_from(["owlbank-demo", "whatsapp-bot", "crm", "owlbank-demo-2", "OWLBANK-DEMO", ""])
ITEMS = st.lists(st.tuples(st.text(max_size=30), SOURCES), max_size=6)


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(f"{record.getMessage()} {record.__dict__}")


def _run(path: Path, stub: HistoryStub, argv: list[str], sink: list[str]) -> int:
    out, err, records, clock = io.StringIO(), io.StringIO(), _Records(), Clock()
    root = logging.getLogger()
    root.addHandler(records)
    old = root.level
    root.setLevel(logging.DEBUG)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = mh.main(["--setup-output", str(path), "--secret", f"twilio_api_key={ARN}", *argv],
                         sm_client=SecretsStub(),
                         client_factory=lambda *a: stub, sleep=clock.sleep, clock=clock.now)
    finally:
        root.removeHandler(records)
        root.setLevel(old)
    sink.append(out.getvalue() + err.getvalue() + "\n".join(records.lines))
    return rc


@settings(max_examples=60, deadline=None)
@given(summaries=ITEMS, observations=ITEMS, lag=st.integers(0, 3), page=st.integers(1, 4),
       preset=st.booleans())
def test_apply_then_clear_leaves_no_demo_item_and_the_rest_untouched(
        summaries: list[tuple[str, str]], observations: list[tuple[str, str]], lag: int, page: int,
        preset: bool) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = setup_file(Path(tmp))
        stub, sink = HistoryStub(lag=lag, page=page), []
        pid = stub.add_profile(EMAIL)
        stub.summaries[pid] = [stub.item(c, s) for c, s in summaries]
        stub.observations[pid] = [stub.item(c, s) for c, s in observations]
        others: dict[str, list[dict[str, Any]]] = {
            "s": [i for i in stub.summaries[pid] if i["source"] != mh.SOURCE],
            "o": [i for i in stub.observations[pid] if i["source"] != mh.SOURCE]}
        argv = [] if preset else ["--summary", "Generated summary.", "--observation", "Generated observation."]

        assert _run(path, stub, [*argv, "--apply"], sink) == 0
        assert _run(path, stub, ["--clear", "--apply"], sink) == 0

        stub.pending.clear()
        assert [i for i in stub.summaries[pid] + stub.observations[pid] if i["source"] == mh.SOURCE] == []
        assert [i for i in stub.summaries[pid] if i["source"] != mh.SOURCE] == others["s"]
        assert [i for i in stub.observations[pid] if i["source"] != mh.SOURCE] == others["o"]
        for secret in (API_SECRET, API_SID):
            assert secret not in "\n".join(sink)


@given(customer=st.text(min_size=1, max_size=20), channel=st.sampled_from(["whatsapp", "sms", "rcs", "chat"]))
def test_conversation_id_always_matches_memorys_pattern(customer: str, channel: str) -> None:
    """Memory refuses any other shape (HTTP 400, Twilio error 20001; seen live)."""
    import re

    from scripts import memory_history

    assert re.fullmatch(r"conv_conversation_[0-7][0-9a-z]{25}", memory_history.conversation_id(customer, channel))
