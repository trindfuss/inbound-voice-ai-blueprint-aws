"""scripts/check_secrets.py with stubbed Secrets Manager and KMS clients.

Examples: all valid, placeholder, malformed, AWS-managed key, usage errors. Properties: (no value
leak in stdout, stderr or log records) and (exit 0 iff every value validates). No AWS calls.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from owlbank.secret_schema import SCHEMAS, validate
from scripts import check_secrets as cs

ROOT = Path(__file__).resolve().parents[3]
CMK = "arn:aws:kms:us-east-1:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab"
ARNS = {
    "twilio_api_key": "arn:aws:secretsmanager:us-east-1:111122223333:secret:twilio/api-key-AbC",
    "twilio_auth_token": "arn:aws:secretsmanager:us-east-1:111122223333:secret:twilio/auth-token-DeF",
    "basic_auth": "arn:aws:secretsmanager:us-east-1:111122223333:secret:twilio/basic-auth-GhI",
    "context_api_key": "arn:aws:secretsmanager:us-east-1:111122223333:secret:twilio/context-api-key-JkL",
}
NOW = 1_800_000_000.0
PREV = "3c" * 24
ARGS = [f"--secret={n}={a}" for n, a in ARNS.items()]
BASE_ARGS = [a for a in ARGS if "context_api_key" not in a]  # the three always-deployed secrets
GOOD = {
    "twilio_api_key": {"account_sid": "AC" + "0123456789abcdef" * 2, "api_key": "SK" + "a1" * 16,
                       "api_secret": "Ab3" * 10 + "Zz"},
    "twilio_auth_token": {"auth_token": "fedcba9876543210" * 2},
    "basic_auth": {"user": "owlbank", "password": "5a" * 24},
    "context_api_key": {"account_sid": "AC" + "0123456789abcdef" * 2, "api_key": "SK" + "b2" * 16,
                        "api_secret": "Cd4" * 10 + "Yy"},
}


class StubSM:
    def __init__(self, values: dict[str, Any], keys: dict[str, str | None] | None = None) -> None:
        self.by_arn = {ARNS[n]: v for n, v in values.items()}
        self.keys = dict.fromkeys(ARNS.values(), CMK) if keys is None else {ARNS[n]: k for n, k in keys.items()}
        self.calls: list[str] = []

    def describe_secret(self, **kw: Any) -> dict[str, Any]:
        self.calls.append("describe_secret")
        out: dict[str, Any] = {"ARN": kw["SecretId"]}
        if self.keys.get(kw["SecretId"]):
            out["KmsKeyId"] = self.keys[kw["SecretId"]]
        return out

    def get_secret_value(self, **kw: Any) -> dict[str, Any]:
        self.calls.append("get_secret_value")
        v = self.by_arn[kw["SecretId"]]
        return {"ARN": kw["SecretId"], "SecretString": v if isinstance(v, str) else json.dumps(v)}

    def put_secret_value(self, **kw: Any) -> None:  # pragma: no cover - must never be called
        raise AssertionError("check_secrets is read-only")


class StubKMS:
    def __init__(self, manager: str = "CUSTOMER") -> None:
        self.manager = manager

    def describe_key(self, **kw: Any) -> dict[str, Any]:
        return {"KeyMetadata": {"KeyManager": self.manager}}


def run(values: dict[str, Any], argv: list[str] | None = None, **kw: Any) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    sm = kw.pop("sm", None) or StubSM(values, kw.pop("keys", None))
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cs.main(ARGS if argv is None else argv, sm_client=sm, kms_client=kw.pop("kms", StubKMS()),
                       clock=kw.pop("clock", lambda: NOW))
    return code, out.getvalue(), err.getvalue()


def test_all_valid_exits_0_and_lists_every_field():
    code, out, err = run(GOOD)
    assert code == 0
    assert err == ""
    for name in GOOD:
        for f in SCHEMAS["twilio_api_key" if name.endswith("_key") else name]:
            assert any(name in line and f in line and "ok" in line for line in out.splitlines())
    for v in GOOD.values():
        assert not any(s in out for s in v.values())


def test_placeholder_exits_1_naming_secret_and_field():
    code, out, err = run({**GOOD, "basic_auth": {"placeholder": True, "unset": "zzzz-unset-1234"}})
    assert code == 1
    assert "basic_auth" in out and "field placeholder: placeholder" in out
    assert "basic_auth" in err
    assert "twilio_api_key" not in err


def test_malformed_exits_1_naming_field():
    bad = {**GOOD["twilio_api_key"], "api_key": "not-an-sk"}
    code, out, _ = run({**GOOD, "twilio_api_key": bad})
    assert code == 1
    assert "twilio_api_key: field api_key: format" in out
    assert "not-an-sk" not in out


def test_missing_field_and_not_json_are_problems():
    code, out, _ = run({**GOOD, "twilio_auth_token": {}, "basic_auth": "not json{"})
    assert code == 1
    assert "twilio_auth_token: field auth_token: missing" in out
    assert "basic_auth" in out and "not JSON" in out


def test_aws_managed_key_exits_1_naming_secret():
    keys = dict.fromkeys(ARNS, CMK) | {"twilio_auth_token": "alias/aws/secretsmanager"}
    code, out, _ = run(GOOD, keys=keys)
    assert code == 1
    line = next(ln for ln in out.splitlines() if "AWS-managed" in ln)
    assert "twilio_auth_token" in line
    assert "twilio_api_key" not in line


def test_no_key_id_means_aws_managed():
    keys = dict.fromkeys(ARNS, CMK) | {"basic_auth": None}
    assert run(GOOD, keys=keys)[0] == 1


def test_kms_key_not_customer_managed_exits_1():
    code, out, _ = run(GOOD, kms=StubKMS("AWS"))
    assert code == 1 and "not CUSTOMER" in out


def test_read_only_and_checks_every_secret_after_a_failure():
    sm = StubSM({**GOOD, "twilio_api_key": {"placeholder": True}})
    code, out, _ = run(GOOD, sm=sm)
    assert code == 1
    assert sm.calls.count("get_secret_value") == 4  # one read per secret
    assert "basic_auth" in out and "twilio_auth_token" in out


def test_access_denied_is_a_problem_naming_the_action():
    class Denied(StubSM):
        def get_secret_value(self, **kw: Any) -> dict[str, Any]:
            err = Exception("boom secret-ish")
            err.response = {"Error": {"Code": "AccessDeniedException"}}  # type: ignore[attr-defined]
            raise err

    code, out, _ = run(GOOD, sm=Denied(GOOD))
    assert code == 1 and "secretsmanager:GetSecretValue" in out and "secret-ish" not in out


@pytest.mark.parametrize("argv", [
    ["--bogus"],
    ["--secret", "nope=arn"],
    ["--secret", "no-equals"],
    ["--secret", f"twilio_api_key={ARNS['twilio_api_key']}"],  # the other two required have no ARN
    ["--secret", f"context_api_key={ARNS['context_api_key']}"],  # an optional one doesn't stand in for the rest
    ["--cdk-outputs", "/nonexistent/outputs.json"],
])
def test_usage_errors_exit_2_without_aws_calls(argv):
    sm = StubSM(GOOD)
    code, out, _ = run(GOOD, argv, sm=sm)
    assert code == 2
    assert out == "" and sm.calls == []


def test_arns_from_cdk_outputs_file(tmp_path):
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"OwlBankData": {f"Secret{n}": ARNS[k] for n, k in
                                             (("TwilioApiKey", "twilio_api_key"),
                                              ("TwilioAuthToken", "twilio_auth_token"),
                                              ("BasicAuth", "basic_auth"),
                                              ("ContextApiKey", "context_api_key"))}}))
    assert run(GOOD, ["--cdk-outputs", str(p)])[0] == 0


PLACEHOLDER_KEY = {"placeholder": True, "unset": "zzzz-unset-1234"}


def test_placeholder_in_context_api_key_is_pending_not_a_failure():
    """Runbook step 6 runs before twilio_setup writes this key, so it must exit 0 there."""
    code, out, err = run({**GOOD, "context_api_key": PLACEHOLDER_KEY})
    assert code == 0
    assert "context_api_key" in out and "PENDING" in out and "twilio_setup.py --phase pre" in out
    assert "NOT READY" not in err and "still to write: context_api_key" in out


def test_strict_reports_the_placeholder_in_context_api_key():
    code, out, err = run({**GOOD, "context_api_key": PLACEHOLDER_KEY}, ["--strict", *ARGS])
    assert code == 1
    assert "context_api_key: field placeholder: placeholder" in out
    assert "context_api_key" in err and "basic_auth" not in err


@pytest.mark.parametrize("name", ["twilio_api_key", "twilio_auth_token", "basic_auth"])
def test_placeholder_in_an_always_written_secret_still_fails_without_strict(name):
    code, _, err = run({**GOOD, name: PLACEHOLDER_KEY})
    assert code == 1 and name in err


def test_malformed_context_api_key_is_checked_like_a_twilio_key():
    code, out, _ = run({**GOOD, "context_api_key": {**GOOD["context_api_key"], "api_key": "not-an-sk"}})
    assert code == 1
    assert "context_api_key: field api_key: format" in out and "not-an-sk" not in out


def test_context_api_key_absent_is_not_checked_and_exit_unchanged():
    sm = StubSM(GOOD)
    code, out, err = run(GOOD, BASE_ARGS, sm=sm)
    assert code == 0 and err == ""
    assert "context_api_key" not in out
    assert sm.calls.count("get_secret_value") == 3  # one read per secret


# --------------------------------------------------------------------------- properties

HEX = "0123456789abcdef"
ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
WELL_FORMED = {
    "account_sid": st.text(HEX, min_size=32, max_size=32).map(lambda b: "AC" + b),
    "api_key": st.text(HEX, min_size=32, max_size=32).map(lambda b: "SK" + b),
    "api_secret": st.text(ALNUM, min_size=32, max_size=32),
    "auth_token": st.text(HEX, min_size=32, max_size=32),
    "user": st.text(ALNUM + "._-", min_size=4, max_size=64),
    "password": st.text(HEX, min_size=32, max_size=72),
}
anything = st.text(min_size=4, max_size=40)


@st.composite
def all_values(draw: st.DrawFn) -> dict[str, Any]:
    """Each secret's value: well formed, or one field mangled, or the placeholder, or not an object."""
    out: dict[str, Any] = {}
    for name in GOOD:
        fields = SCHEMAS["twilio_api_key" if name == "context_api_key" else name]
        value: dict[str, Any] = {f: draw(WELL_FORMED[f]) for f in fields}
        kind = draw(st.sampled_from(["valid", "valid", "bad_field", "drop", "placeholder", "list"]))
        if kind == "bad_field":
            value[draw(st.sampled_from(list(fields)))] = draw(anything)
        elif kind == "drop":
            del value[draw(st.sampled_from(list(fields)))]
        elif kind == "placeholder":
            value = {"placeholder": True, "unset": draw(anything)}
        elif kind == "list":
            out[name] = [draw(anything)]
            continue
        out[name] = value
    return out


def _strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


class Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def text(self) -> str:
        # message, args and the caller's ``extra`` fields; not timestamps or ids (digits would collide)
        std = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}
        return "\n".join(f"{r.getMessage()} {r.args!r} "
                         f"{ {k: v for k, v in r.__dict__.items() if k not in std}!r}"
                         for r in self.records)


def _run_logged(values: dict[str, Any], argv: list[str] | None = None) -> tuple[int, str]:
    handler, root = Capture(), logging.getLogger()
    old = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        code, out, err = run(values, argv)
    finally:
        root.removeHandler(handler)
        root.setLevel(old)
    return code, f"{out}\n{err}\n{handler.text()}"


def _problems(name: str, value: Any) -> list[Any]:
    return validate("twilio_api_key" if name == "context_api_key" else name, value)


@given(values=all_values())
@settings(max_examples=200, deadline=None)
def test_strict_exit_0_iff_every_value_validates(values):
    code, _ = _run_logged(values, ["--strict", *ARGS])
    assert (code == 0) == all(not _problems(n, v) for n, v in values.items())


@given(values=all_values())
@settings(max_examples=200, deadline=None)
def test_default_exit_0_iff_every_value_validates_or_only_the_restricted_key_is_pending(values):
    """Without --strict the one exception is the restricted key that is still the placeholder."""
    code, _ = _run_logged(values)

    def fine(name: str, value: Any) -> bool:
        problems = _problems(name, value)
        return not problems or (name == "context_api_key" and any(p.kind == "placeholder" for p in problems))

    assert (code == 0) == all(fine(n, v) for n, v in values.items())


_CORPUS = "\n".join([(ROOT / p).read_text(encoding="utf-8") for p in
                     # obs.py writes fixed field names (``event``) into every record
                     ("scripts/check_secrets.py", "scripts/_common.py", "src/owlbank/secret_schema.py",
                      "src/owlbank/obs.py")]
                    + list(ARNS.values()) + [CMK]
                    # the shared redaction marker: a log filter another test leaves on the root logger
                    # writes it into records, so "edac" in a value isn't a leak
                    + ["[redacted]"]).casefold()


def _grams(value: str, n: int = 4) -> set[str]:
    folded = value.casefold()
    return {folded[i:i + n] for i in range(len(folded) - n + 1)}


@given(values=all_values())
@settings(max_examples=200, deadline=None)
def test_no_value_substring_in_output_or_logs(values):
    strings = _strings(values)
    # Fixed text (sources, ARNs, key) can't stand in for a leak: skip examples that collide with it.
    assume(not any(g in _CORPUS for s in strings for g in _grams(s)))
    _, text = _run_logged(values)
    folded = text.casefold()
    assert [s for s in strings if any(g in folded for g in _grams(s))] == []


# --------------------------------------------------------------------------- previous Basic password


def basic_with(**extra: Any) -> dict[str, Any]:
    return {**GOOD, "basic_auth": {**GOOD["basic_auth"], **extra}}


def test_no_previous_password_is_reported_as_none():
    code, out, err = run(GOOD)
    assert code == 0 and err == ""
    assert any("previous_password" in ln and "no previous password" in ln for ln in out.splitlines())


def test_active_previous_password_shows_its_deadline_not_its_value():
    until = int(NOW) + 3600
    code, out, err = run(basic_with(previous_password=PREV, previous_until=until))
    assert code == 0 and err == ""
    deadline = cs.datetime.fromtimestamp(until, cs.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    assert f"previous: active until {deadline}" in out
    assert PREV not in out and GOOD["basic_auth"]["password"] not in out


def test_expired_previous_password_warns_to_clear_it_and_exits_0():
    until = int(NOW) - 1
    code, out, err = run(basic_with(previous_password=PREV, previous_until=until))
    assert code == 0
    assert "expired" in out and "clear it" in out
    assert "warning" in err and "basic_auth" in err
    assert PREV not in out + err


@pytest.mark.parametrize("extra", [{"previous_password": PREV},
                                   {"previous_until": 1_900_000_000},
                                   {"previous_password": PREV, "previous_until": 1_900_000_000_000},
                                   {"previous_password": "nope", "previous_until": 1_900_000_000},
                                   {"previous_password": PREV, "previous_until": "1900000000"}])
def test_malformed_previous_pair_is_a_problem_without_values(extra):
    code, out, err = run(basic_with(**extra))
    assert code == 1
    assert "basic_auth" in out and "NOT READY" in err
    assert PREV not in out + err and "nope" not in out + err


@settings(max_examples=40, deadline=None)
@given(prev=st.text("0123456789abcdef", min_size=32, max_size=48), delta=st.integers(-10**6, 10**6))
def test_previous_report_never_prints_either_password(prev, delta):
    assume(prev not in cs.__doc__ and prev != GOOD["basic_auth"]["password"])
    values = basic_with(previous_password=prev, previous_until=int(NOW) + delta)
    code, out, err = run(values)
    assert code == 0
    assert prev not in out + err
    assert GOOD["basic_auth"]["password"] not in out + err
