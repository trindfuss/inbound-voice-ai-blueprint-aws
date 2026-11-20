"""Over generated option combinations against the
stateful Memory stub, ``--apply`` twice leaves the same profile as once (and the second run writes
nothing), ``--clear`` after ``--apply`` leaves no ``WebActivity`` value, and no output or log record
carries a credential. Found here: a trait in the group that the script doesn't write (one a feed
added) made the read-back never match and ``--clear`` leave the group non-empty."""

from __future__ import annotations

import contextlib
import io
import json
import logging
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank.web_activity import GROUP, TRAITS
from scripts import web_activity as wa
from scripts._common import DATA_DIR
from tests.unit.scripts.test_web_activity_script import (
    ACCOUNT,
    API_SECRET,
    API_SID,
    ARN,
    NOW,
    NOW_TEXT,
    STORE,
    Clock,
    ProfileStub,
    SecretsStub,
)

_FIXTURES = json.loads((DATA_DIR / "fixtures.json").read_text(encoding="utf-8"))
CUSTOMERS = {c["customer_id"]: c["memory_email"] for c in _FIXTURES["customers"]}


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(f"{record.getMessage()} {record.__dict__}")


@st.composite
def options(draw: st.DrawFn) -> tuple[list[str], dict[str, Any]]:
    """``(argv, the WebActivity values --apply should write)``."""
    argv = ["--stage", draw(st.sampled_from(wa.STAGES))]
    want: dict[str, Any] = {"lastViewedProduct": wa.DEFAULT_PRODUCT, "lastViewedAt": NOW_TEXT,
                            "viewCount": wa.DEFAULT_VIEWS}
    if draw(st.booleans()):
        argv += ["--customer", draw(st.sampled_from(sorted(CUSTOMERS)))]
    if draw(st.booleans()):
        product = draw(st.text(min_size=1, max_size=40).filter(lambda s: s.strip()))
        argv.append(f"--product={product}")  # "=" so a value starting with "-" stays a value
        want["lastViewedProduct"] = product.strip()
    if draw(st.booleans()):
        want["viewCount"] = draw(st.integers(0, 10**6))
        argv += ["--views", str(want["viewCount"])]
    if draw(st.booleans()):
        at = NOW + timedelta(minutes=draw(st.integers(-60 * 24 * 400, 60 * 24 * 30)))
        offset = draw(st.sampled_from([None, timedelta(0), timedelta(hours=-5), timedelta(hours=5, minutes=30)]))
        text = (at.replace(tzinfo=None).isoformat() if offset is None
                else at.astimezone(timezone(offset)).isoformat())  # no offset: read as UTC
        argv += ["--viewed-at", text]
        want["lastViewedAt"] = at.strftime("%Y-%m-%dT%H:%M:%SZ")
    return argv, want


#: What is on the customer's profile before the run: nothing, a profile without web activity, or one
#: with other web-activity values (and maybe an extra trait of the group the script doesn't own).
prior = st.sampled_from(["none", "bare", "other", "other_extra"])


def _stub(state: str, email: str, lag: int, drop_empty: bool) -> ProfileStub:
    stub = ProfileStub(lag=lag, drop_empty=drop_empty)
    if state == "bare":
        stub.add(email)
    elif state.startswith("other"):
        values: dict[str, Any] = {"lastViewedProduct": "savings", "lastViewedAt": "2026-01-01T00:00:00Z",
                                  "viewCount": 9}
        if state == "other_extra":
            values["referrer"] = "newsletter"
        stub.add(email, {GROUP: values})
    return stub


def _run(tmp: Path, stub: ProfileStub, argv: list[str], sink: list[str]) -> int:
    path = tmp / "twilio.context.json"
    path.write_text(json.dumps({"memoryStoreId": STORE, "accountSid": ACCOUNT}), encoding="utf-8")
    out, err, records = io.StringIO(), io.StringIO(), _Records()
    root = logging.getLogger()
    root.addHandler(records)
    old_level = root.level
    root.setLevel(logging.DEBUG)
    clock = Clock()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = wa.main(["--setup-output", str(path), "--secret", f"twilio_api_key={ARN}", *argv],
                         sm_client=SecretsStub(),
                         client_factory=lambda *a: stub, sleep=clock.sleep, clock=clock.now, now=lambda: NOW)
    finally:
        root.removeHandler(records)
        root.setLevel(old_level)
    text = out.getvalue()
    assert text.startswith(wa.BANNER + "\n")
    sink.append(text + err.getvalue() + "\n".join(records.lines))
    return rc


def _email(argv: list[str]) -> str:
    return CUSTOMERS[argv[argv.index("--customer") + 1]] if "--customer" in argv else CUSTOMERS["C-1001"]


def _settled_web_activity(stub: ProfileStub, email: str) -> dict[str, Any]:
    (pid,) = stub.emails[email]
    return {k: v for k, v in stub.settled(pid).get(GROUP, {}).items() if v is not None}


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(opts=options(), state=prior, lag=st.integers(0, 3), drop_empty=st.booleans())
def test_apply_twice_equals_once_and_clear_leaves_no_web_activity(
        tmp_path: Path, opts: tuple[list[str], dict[str, Any]], state: str, lag: int, drop_empty: bool) -> None:
    (argv, want), outputs = opts, []
    email = _email(argv)

    once = _stub(state, email, lag, drop_empty)
    assert _run(tmp_path, once, [*argv, "--apply"], outputs) == 0

    twice = _stub(state, email, lag, drop_empty)
    assert _run(tmp_path, twice, [*argv, "--apply"], outputs) == 0
    writes_before = len(twice.writes())
    assert _run(tmp_path, twice, [*argv, "--apply"], outputs) == 0
    assert len(twice.writes()) == writes_before  # the second run wrote nothing
    assert twice.emails[email] == once.emails[email]
    assert twice.settled(twice.emails[email][0]) == once.settled(once.emails[email][0])
    # The three traits hold what was asked for; a trait the script doesn't own is left alone.
    written = _settled_web_activity(once, email)
    assert {k: written.get(k) for k in TRAITS} == want
    assert ("referrer" in written) == (state == "other_extra")

    assert _run(tmp_path, once, [*argv, "--clear", "--apply"], outputs) == 0
    assert _settled_web_activity(once, email) == {}

    for text in outputs:
        assert API_SECRET not in text and API_SID not in text


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(opts=options(), state=prior, apply_first=st.booleans(),
       mode=st.sampled_from([[], ["--check"], ["--clear"], ["--clear", "--apply"]]))
def test_no_credential_in_any_output(tmp_path: Path, opts: tuple[list[str], dict[str, Any]], state: str,
                                        apply_first: bool, mode: list[str]) -> None:
    argv = opts[0]
    stub, outputs = _stub(state, _email(argv), 0, True), []
    if apply_first:
        _run(tmp_path, stub, [*argv, "--apply"], outputs)
    _run(tmp_path, stub, [*argv, *mode], outputs)
    for text in outputs:
        assert API_SECRET not in text and API_SID not in text
