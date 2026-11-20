"""Portable secrets, properties (ARN resolution order) and (no value leak), for
``scripts/_common.secret_arn`` and ``read_secret``.

For any outputs map and ``--secret`` map, ``secret_arn`` returns the explicit ARN when given,
else the output of the first matching candidate in ``SECRET_OUTPUTS`` order, else raises naming the
secret and every candidate, before any AWS call.

For any secret values (well formed, malformed, the placeholder, not JSON, or carried in the
client's own exception message), no substring of 4 or more characters of any value appears in the
error text or the captured log records. Fixed text can't produce a false alarm: an example is only
kept when none of its values' 4-grams occurs in the source of the two modules that write the
messages, the ARN or the stub's error codes.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from owlbank.secret_schema import SCHEMAS
from scripts._common import SECRET_OUTPUTS, ScriptError, read_secret, secret_arn

ROOT = Path(__file__).resolve().parents[2]
ARN = "arn:aws:secretsmanager:us-east-1:111122223333:secret:owlbank/twilio/x-AbC"
LOGICAL = sorted(SECRET_OUTPUTS)
CODES = ("AccessDeniedException", "AccessDenied", "ResourceNotFoundException", "DecryptionFailure",
         "InternalServiceError")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


ALL_CANDIDATES = [_norm(c) for cands in SECRET_OUTPUTS.values() for c in cands]


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def get_secret_value(self, **kw: Any) -> Any:  # pragma: no cover - must not be reached
        self.calls.append(kw)
        raise AssertionError("no AWS call expected")


def resolve_and_read(logical: str, outputs: dict[str, str], explicit: dict[str, str], client: Any) -> Any:
    """The order every reworked script follows: resolve the ARN, then read."""
    return read_secret(logical, secret_arn(logical, outputs, explicit), client)



arns = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789:/-", min_size=1, max_size=30)
#: Output keys that match no candidate of any secret, exactly or as a prefix.
noise_keys = st.text(min_size=0, max_size=25).filter(
    lambda k: not any(_norm(k).startswith(c) for c in ALL_CANDIDATES))


@given(logical=st.sampled_from(LOGICAL), outputs=st.dictionaries(st.text(max_size=30), arns, max_size=8),
       explicit=st.dictionaries(st.sampled_from(LOGICAL), arns, max_size=3), given_arn=arns)
@settings(max_examples=200)
def test_an_explicit_arn_always_wins(logical, outputs, explicit, given_arn):
    explicit = {**explicit, logical: given_arn}
    assert secret_arn(logical, outputs, explicit) == given_arn


@given(logical=st.sampled_from(LOGICAL), data=st.data(),
       noise=st.dictionaries(noise_keys, arns, max_size=6),
       other_explicit=st.dictionaries(st.sampled_from(LOGICAL), arns, max_size=3))
@settings(max_examples=200)
def test_else_the_first_candidate_in_order(logical, data, noise, other_explicit):
    candidates = SECRET_OUTPUTS[logical]
    present = data.draw(st.lists(st.sampled_from(candidates), min_size=1, unique=True))
    values = data.draw(st.lists(arns, min_size=len(present), max_size=len(present), unique=True))
    outputs = {**noise, **dict(zip(present, values, strict=True))}
    explicit = {k: v for k, v in other_explicit.items() if k != logical}
    first = min(present, key=candidates.index)
    assert secret_arn(logical, outputs, explicit) == outputs[first]


@given(logical=st.sampled_from(LOGICAL), noise=st.dictionaries(noise_keys, arns, max_size=6),
       other_explicit=st.dictionaries(st.sampled_from(LOGICAL), arns, max_size=3))
@settings(max_examples=200)
def test_else_raises_naming_secret_and_candidates_without_an_aws_call(logical, noise, other_explicit):
    explicit = {k: v for k, v in other_explicit.items() if k != logical}
    client = RecordingClient()
    with pytest.raises(ScriptError) as ei:
        resolve_and_read(logical, noise, explicit, client)
    msg = str(ei.value)
    assert logical in msg
    assert all(c in msg for c in SECRET_OUTPUTS[logical])
    assert client.calls == []


@given(logical=st.sampled_from(LOGICAL), outputs=st.dictionaries(st.text(max_size=30), arns, max_size=8))
@settings(max_examples=100)
def test_an_empty_explicit_arn_is_not_an_arn(logical, outputs):
    """An empty ``--secret`` value can't reach here (``parse_pairs`` refuses it); if it did, it falls
    through to the outputs rather than being used as an ARN."""
    try:
        got = secret_arn(logical, outputs, {logical: ""})
    except ScriptError:
        return
    assert got != ""



_SOURCES = "\n".join((ROOT / p).read_text(encoding="utf-8")
                     for p in ("scripts/_common.py", "src/owlbank/secret_schema.py"))
CORPUS = "\n".join([_SOURCES, ARN, *CODES]).casefold()


def grams(value: str, n: int = 4) -> set[str]:
    folded = value.casefold()
    return {folded[i:i + n] for i in range(len(folded) - n + 1)}


def leaked(values: list[str], text: str) -> list[str]:
    """Values with a substring of 4+ characters in ``text``; checking every 4-gram covers longer ones."""
    folded = text.casefold()
    return [v for v in values if any(g in folded for g in grams(v))]


class ValueClient:
    def __init__(self, *, raw: str | None = None, error: Exception | None = None) -> None:
        self.raw, self.error = raw, error

    def get_secret_value(self, **kw: Any) -> Any:
        if self.error is not None:
            raise self.error
        return {"ARN": kw["SecretId"], "SecretString": self.raw}


class FakeClientError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.response = {"Error": {"Code": code, "Message": message}}


class Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        # Only this thread's records: ``read_secret`` logs synchronously, while threads left by
        # earlier tests in a full run log unrelated text whose digits a value's 4-grams can match.
        if record.thread == threading.get_ident():
            self.records.append(record)

    def text(self) -> str:
        # Message, args and ``extra`` fields only: the standard attributes (timestamps, thread and
        # process ids) are digits that a hex value's 4-grams can match by chance.
        # Keys beginning with an underscore are added by the logging setup (``_owlbank_redacted`` on
        # every record once any earlier test installs the redacting record factory), not by the code
        # under test, and "redacted" holds the hex 4-gram "edac".
        return "\n".join(
            f"{r.getMessage()} {r.args!r} "
            f"{ {k: v for k, v in r.__dict__.items() if k not in STANDARD_RECORD_ATTRS and not k.startswith('_')}!r}"
            for r in self.records
        )


STANDARD_RECORD_ATTRS = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


HEX = "0123456789abcdef"
ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#: Well-formed field values, so the success path is covered too.
WELL_FORMED = {
    "account_sid": st.text(HEX, min_size=32, max_size=32).map(lambda b: "AC" + b),
    "api_key": st.text(HEX, min_size=32, max_size=32).map(lambda b: "SK" + b),
    "api_secret": st.text(ALNUM, min_size=32, max_size=32),
    "auth_token": st.text(HEX, min_size=32, max_size=32),
    "user": st.text(ALNUM + "._-", min_size=4, max_size=64),
    "password": st.text(HEX, min_size=32, max_size=72),
}
anything = st.text(min_size=4, max_size=60)


@st.composite
def scenario(draw: st.DrawFn) -> tuple[str, Any, list[str]]:
    """``(logical, client, values)``: the client returns or raises something built from ``values``."""
    logical = draw(st.sampled_from(LOGICAL))
    fields = list(SCHEMAS[logical])
    kind = draw(st.sampled_from(["valid", "malformed", "placeholder", "not_json", "not_object",
                                 "client_error"]))
    if kind == "valid":
        value: dict[str, Any] = {f: draw(WELL_FORMED[f]) for f in fields}
        values = list(value.values())
        return logical, ValueClient(raw=json.dumps(value)), values
    if kind == "malformed":
        value = {f: draw(st.one_of(WELL_FORMED[f], anything)) for f in fields}
        bad = draw(st.sampled_from(fields))
        value[bad] = draw(anything)
        dropped = draw(st.sets(st.sampled_from(fields), max_size=len(fields) - 1))
        for f in dropped - {bad}:
            del value[f]
        value[draw(st.sampled_from(["extra", "note"]))] = draw(anything)
        values = [v for v in value.values() if isinstance(v, str)]
        return logical, ValueClient(raw=json.dumps(value)), values
    if kind == "placeholder":
        unset = draw(anything)
        return logical, ValueClient(raw=json.dumps({"placeholder": True, "unset": unset})), [unset]
    if kind == "not_json":
        raw = draw(anything).replace("{", "(")
        assume(not _is_json(raw))
        return logical, ValueClient(raw="{" + raw), [raw]
    if kind == "not_object":
        items = draw(st.lists(anything, min_size=1, max_size=3))
        return logical, ValueClient(raw=json.dumps(items)), items
    secret = draw(anything)
    code = draw(st.sampled_from(CODES))
    return logical, ValueClient(error=FakeClientError(code, f"denied: {secret}")), [secret]


def _is_json(raw: str) -> bool:
    try:
        json.loads("{" + raw)
    except ValueError:
        return False
    return True


@given(case=scenario())
@settings(max_examples=400)
def test_no_value_in_error_text_or_logs(case):
    logical, client, values = case
    # Fixed text (module source, ARN, codes) can't stand in for a leak.
    assume(not any(g in CORPUS for v in values for g in grams(v)))
    handler = Capture()
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        try:
            read_secret(logical, ARN, client)
            out = ""
        except ScriptError as exc:
            assert exc.__cause__ is None
            out = f"{exc}\n{exc!r}\n{exc.args!r}"
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    assert leaked(values, out) == []
    assert leaked(values, handler.text()) == []


def test_capture_text_ignores_keys_the_logging_setup_adds():
    """The redacting record factory marks every record; that marker must not count as logged text."""
    record = logging.makeLogRecord({"msg": "secret read", "_owlbank_redacted": True, "secret_name": "x"})
    handler = Capture()
    handler.emit(record)
    assert "_owlbank_redacted" not in handler.text()
    assert "secret_name" in handler.text()
