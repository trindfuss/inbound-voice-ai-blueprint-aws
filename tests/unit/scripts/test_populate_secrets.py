"""scripts/populate_secrets.py as an optional helper.

Values come from an injected prompt or a ``--from-file`` JSON; Secrets Manager and KMS are stubs.
No test calls AWS or reads the Keychain.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from scripts import populate_secrets as ps
from tests.logutil import text_of

ROOT = Path(__file__).resolve().parents[3]
ACCOUNT_SID = "AC" + "0123456789abcdef" * 2
API_KEY = "SK" + "a1" * 16
API_SECRET = "Zq7Rw2Lx9Pt4Vn6Bm8Hc3Jd5Kf1Gs0Yy"   # 32 alnum
AUTH_TOKEN = "fedcba9876543210" * 2
BASIC_USER = "twilio-hooks"
OLD_PASSWORD = "9e" * 24
NEW_PASSWORD = "c4" * 24
NOW = 1_800_000_000.0
VALUES = (API_SECRET, AUTH_TOKEN, OLD_PASSWORD, NEW_PASSWORD)
TWILIO = {
    "twilio_api_key": {"account_sid": ACCOUNT_SID, "api_key": API_KEY, "api_secret": API_SECRET},
    "twilio_auth_token": {"auth_token": AUTH_TOKEN},
}
ARNS = {
    "twilio_api_key": "arn:aws:secretsmanager:us-east-1:111122223333:secret:owlbank/twilio/api-key-AbC",
    "twilio_auth_token": "arn:aws:secretsmanager:us-east-1:111122223333:secret:owlbank/twilio/auth-token-DeF",
    "basic_auth": "arn:aws:secretsmanager:us-east-1:111122223333:secret:owlbank/twilio/basic-auth-GhI",
}
SECRET_ARGS = [a for name, arn in ARNS.items() for a in ("--secret", f"{name}={arn}")]
CMK = "arn:aws:kms:us-east-1:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab"


class StubSecretsManager:
    def __init__(self, kms_key: str | None = CMK, *, basic: dict[str, Any] | None = None,
                 random_password: str = NEW_PASSWORD) -> None:
        self.kms_key = kms_key
        self.basic = basic if basic is not None else {"user": BASIC_USER, "password": OLD_PASSWORD}
        self.random_password = random_password
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def describe_secret(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("describe_secret", kw))
        out: dict[str, Any] = {"ARN": kw["SecretId"], "Name": kw["SecretId"].split(":")[-1]}
        if self.kms_key:
            out["KmsKeyId"] = self.kms_key
        return out

    def get_secret_value(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("get_secret_value", kw))
        assert kw["SecretId"] == ARNS["basic_auth"]
        return {"ARN": kw["SecretId"], "SecretString": json.dumps(self.basic)}

    def get_random_password(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("get_random_password", kw))
        return {"RandomPassword": self.random_password}

    def put_secret_value(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("put_secret_value", kw))
        return {"ARN": kw["SecretId"], "VersionId": "v-1"}

    def ops(self) -> list[str]:
        return [op for op, _ in self.calls]

    def puts(self) -> dict[str, Any]:
        return {kw["SecretId"]: json.loads(kw["SecretString"]) for op, kw in self.calls if op == "put_secret_value"}


class StubKms:
    def __init__(self, manager: str = "CUSTOMER") -> None:
        self.manager = manager
        self.calls: list[dict[str, Any]] = []

    def describe_key(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(kw)
        return {"KeyMetadata": {"KeyId": kw["KeyId"], "KeyManager": self.manager}}


class Forbidden:
    """A client that fails the test on any call."""

    def __call__(self, *a: Any, **kw: Any) -> Any:
        raise AssertionError("AWS client not expected")

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"AWS call {name} not expected")


class FakePrompt:
    """Answers each ``"<secret> <field>: "`` prompt from a ``{secret: {field: value}}`` map."""

    def __init__(self, values: dict[str, dict[str, str]] | None = None) -> None:
        self.values = values if values is not None else TWILIO
        self.asked: list[str] = []

    def __call__(self, text: str) -> str:
        self.asked.append(text)
        m = re.fullmatch(r"(\w+) (\w+): ", text)
        assert m, text
        return self.values[m[1]][m[2]] + "\n"


def no_prompt(text: str) -> str:
    raise AssertionError(f"unexpected prompt {text!r}")


def write_file(tmp_path: Path, data: Any, mode: int = 0o600, name: str = "twilio.json") -> Path:
    p = tmp_path / name
    p.write_text(data if isinstance(data, str) else json.dumps(data))
    os.chmod(p, mode)
    return p


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    p = tmp_path / "cdk-outputs.json"
    p.write_text(json.dumps({"OwlBankData": {
        "SecretTwilioApiKey": ARNS["twilio_api_key"],
        "TwilioAuthTokenSecretArn": ARNS["twilio_auth_token"],
        "SecretBasicAuthE5F6A7B8": ARNS["basic_auth"],   # CDK hash suffix
    }}))
    return p


def no_values(*texts: str) -> None:
    for text in texts:
        for v in VALUES:
            assert v not in text


# --------------------------------------------------------------------------- dry run (prompts)


def test_dry_run_is_the_default_prompts_validates_and_makes_no_aws_call(
        outputs: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    prompt = FakePrompt()
    rc = ps.main(["--cdk-outputs", str(outputs)], sm_client=Forbidden(), kms_client=Forbidden(), prompt=prompt)
    assert rc == 0
    assert prompt.asked == ["twilio_api_key account_sid: ", "twilio_api_key api_key: ",
                            "twilio_api_key api_secret: ", "twilio_auth_token auth_token: "]
    cap = capsys.readouterr()
    assert cap.out.startswith("DRY RUN")
    assert ARNS["twilio_api_key"] in cap.out and ARNS["twilio_auth_token"] in cap.out
    assert ARNS["basic_auth"] not in cap.out
    assert f"api_secret ({len(API_SECRET)} chars)" in cap.out
    assert f"auth_token ({len(AUTH_TOKEN)} chars)" in cap.out
    no_values(cap.out, cap.err, caplog.text)
    assert ACCOUNT_SID not in cap.out and API_KEY not in cap.out


def test_dry_run_builds_no_boto3_client(monkeypatch: pytest.MonkeyPatch, outputs: Path) -> None:
    import boto3

    monkeypatch.setattr(boto3, "client", Forbidden())
    assert ps.main(["--cdk-outputs", str(outputs)], prompt=FakePrompt()) == 0
    assert ps.main(["--cdk-outputs", str(outputs), "--new-basic-password"], prompt=no_prompt) == 0


def test_only_prompts_for_the_named_secret() -> None:
    sm = StubSecretsManager()
    prompt = FakePrompt()
    rc = ps.main(["--secret", f"twilio_auth_token={ARNS['twilio_auth_token']}", "--only", "twilio_auth_token",
                  "--apply"], sm_client=sm, kms_client=StubKms(), prompt=prompt)
    assert rc == 0
    assert prompt.asked == ["twilio_auth_token auth_token: "]
    assert sm.puts() == {ARNS["twilio_auth_token"]: {"auth_token": AUTH_TOKEN}}


def test_only_refuses_basic_auth() -> None:
    with pytest.raises(SystemExit):
        ps.main(["--only", "basic_auth"], prompt=no_prompt)


# --------------------------------------------------------------------------- apply


def test_apply_checks_every_key_then_writes_the_validated_values(
        outputs: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    sm, kms = StubSecretsManager(), StubKms()
    rc = ps.main(["--cdk-outputs", str(outputs), "--apply"], sm_client=sm, kms_client=kms, prompt=FakePrompt())
    assert rc == 0
    assert sm.ops() == ["describe_secret"] * 2 + ["put_secret_value"] * 2   # check all before writing
    assert sm.puts() == {ARNS[n]: v for n, v in TWILIO.items()}
    assert len(kms.calls) == 2
    cap = capsys.readouterr()
    no_values(cap.out, cap.err, caplog.text)


@pytest.mark.parametrize("bad", [
    {"twilio_api_key": {**TWILIO["twilio_api_key"], "account_sid": "AC" + "0" * 31}},
    {"twilio_api_key": {**TWILIO["twilio_api_key"], "api_key": "AC" + "0" * 32}},
    {"twilio_api_key": {**TWILIO["twilio_api_key"], "api_secret": ""}},
    {"twilio_auth_token": {"auth_token": AUTH_TOKEN.upper()}},
])
def test_an_invalid_prompted_value_is_refused_before_any_aws_call(
        bad: dict[str, dict[str, str]], capsys: pytest.CaptureFixture[str]) -> None:
    values = {**TWILIO, **bad}
    rc = ps.main([*SECRET_ARGS, "--apply"], sm_client=Forbidden(), kms_client=Forbidden(),
                 prompt=FakePrompt(values))
    assert rc == 2
    err = capsys.readouterr().err
    (name,) = bad
    (field,) = [f for f in values[name] if values[name][f] != TWILIO[name][f]]
    assert f"refusing to write {name}" in err and f"field {field}: format" in err
    no_values(err)


def test_an_invalid_file_value_is_refused_before_writing(tmp_path: Path,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    f = write_file(tmp_path, {"twilio_api_key": TWILIO["twilio_api_key"],
                              "twilio_auth_token": {"token": AUTH_TOKEN}})
    rc = ps.main([*SECRET_ARGS, "--from-file", str(f), "--apply"], sm_client=Forbidden(),
                 kms_client=Forbidden(), prompt=no_prompt)
    assert rc == 2
    err = capsys.readouterr().err
    assert "twilio_auth_token: field auth_token: missing" in err
    no_values(err)


@pytest.mark.parametrize("sm, kms", [
    (StubSecretsManager(kms_key=None), StubKms()),                       # default aws/secretsmanager
    (StubSecretsManager(kms_key="alias/aws/secretsmanager"), StubKms()),
    (StubSecretsManager(), StubKms(manager="AWS")),
])
def test_refuses_without_customer_managed_key(outputs: Path, sm: StubSecretsManager, kms: StubKms,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    rc = ps.main(["--cdk-outputs", str(outputs), "--apply"], sm_client=sm, kms_client=kms, prompt=FakePrompt())
    assert rc == 3
    assert "put_secret_value" not in sm.ops()
    err = capsys.readouterr().err
    assert "refusing" in err
    no_values(err)


def test_missing_arn_is_an_error_before_prompting(capsys: pytest.CaptureFixture[str]) -> None:
    assert ps.main([], sm_client=Forbidden(), kms_client=Forbidden(), prompt=no_prompt) == 2
    err = capsys.readouterr().err
    assert "no ARN for secret twilio_api_key" in err and "SecretTwilioApiKey" in err


def test_unknown_secret_override_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert ps.main(["--secret", "nope=arn:x"], prompt=no_prompt) == 2
    assert "unknown secret name" in capsys.readouterr().err


def test_put_failure_continues_and_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    class Failing(StubSecretsManager):
        def put_secret_value(self, **kw: Any) -> dict[str, Any]:
            if kw["SecretId"] == ARNS["twilio_api_key"]:
                raise RuntimeError(kw["SecretString"])
            return super().put_secret_value(**kw)

    sm = Failing()
    assert ps.main([*SECRET_ARGS, "--apply"], sm_client=sm, kms_client=StubKms(), prompt=FakePrompt()) == 1
    assert list(sm.puts()) == [ARNS["twilio_auth_token"]]
    cap = capsys.readouterr()
    assert "twilio_api_key: put_secret_value failed: RuntimeError" in cap.err
    no_values(cap.out, cap.err)


# --------------------------------------------------------------------------- --from-file


def test_from_file_writes_its_entries_and_skips_a_missing_one(tmp_path: Path) -> None:
    f = write_file(tmp_path, {"twilio_auth_token": {"auth_token": AUTH_TOKEN, "note": "ignored"}})
    sm = StubSecretsManager()
    rc = ps.main([*SECRET_ARGS, "--from-file", str(f), "--apply"], sm_client=sm, kms_client=StubKms(),
                 prompt=no_prompt)
    assert rc == 0
    assert sm.puts() == {ARNS["twilio_auth_token"]: {"auth_token": AUTH_TOKEN}}   # schema fields only


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o666])
def test_a_group_or_world_readable_file_is_refused_before_any_aws_call(
        tmp_path: Path, mode: int, capsys: pytest.CaptureFixture[str]) -> None:
    f = write_file(tmp_path, TWILIO, mode=mode)
    rc = ps.main([*SECRET_ARGS, "--from-file", str(f), "--apply"], sm_client=Forbidden(),
                 kms_client=Forbidden(), prompt=no_prompt)
    assert rc == 2
    err = capsys.readouterr().err
    assert "readable by group or others" in err and f"{mode:04o}" in err and "chmod 600" in err
    no_values(err)


def test_a_0644_file_is_refused_in_both_modes_and_builds_no_client(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3

    monkeypatch.setattr(boto3, "client", Forbidden())
    f = write_file(tmp_path, TWILIO, mode=0o644)
    assert ps.main([*SECRET_ARGS, "--from-file", str(f)], prompt=no_prompt) == 2
    assert ps.main([*SECRET_ARGS, "--from-file", str(f), "--apply"], prompt=no_prompt) == 2


@pytest.mark.parametrize("data, needle", [
    ({**TWILIO, "basic_auth": {"user": BASIC_USER, "password": OLD_PASSWORD}}, "--new-basic-password"),
    ({**TWILIO, API_SECRET: {}}, "1 unknown entry"),
    ([TWILIO], "expected a JSON object"),
    ({}, "nothing to write"),
    ('{"twilio_auth_token": ' + AUTH_TOKEN, "not valid JSON"),
])
def test_bad_files_are_refused_without_echoing_content(tmp_path: Path, data: Any, needle: str,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    f = write_file(tmp_path, data)
    rc = ps.main([*SECRET_ARGS, "--from-file", str(f), "--apply"], sm_client=Forbidden(),
                 kms_client=Forbidden(), prompt=no_prompt)
    assert rc == 2
    err = capsys.readouterr().err
    assert needle in err
    no_values(err)


def test_missing_file_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert ps.main([*SECRET_ARGS, "--from-file", str(tmp_path / "absent.json")], prompt=no_prompt) == 2
    assert "cannot open" in capsys.readouterr().err


def test_only_is_for_prompts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = write_file(tmp_path, TWILIO)
    assert ps.main([*SECRET_ARGS, "--from-file", str(f), "--only", "twilio_api_key"], prompt=no_prompt) == 2
    assert "--only" in capsys.readouterr().err


def test_from_file_and_new_basic_password_are_exclusive(tmp_path: Path) -> None:
    f = write_file(tmp_path, TWILIO)
    with pytest.raises(SystemExit):
        ps.main([*SECRET_ARGS, "--from-file", str(f), "--new-basic-password"], prompt=no_prompt)


# --------------------------------------------------------------------------- --new-basic-password


def test_new_basic_password_sends_the_pd2_rules_and_writes_only_basic_auth(
        outputs: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    sm, kms = StubSecretsManager(), StubKms()
    rc = ps.main(["--cdk-outputs", str(outputs), "--new-basic-password", "--apply"], sm_client=sm,
                 kms_client=kms, prompt=no_prompt, clock=lambda: NOW)
    assert rc == 0
    assert sm.ops() == ["describe_secret", "get_secret_value", "get_random_password", "put_secret_value"]
    (rules,) = [kw for op, kw in sm.calls if op == "get_random_password"]
    assert rules == {"PasswordLength": 48, "ExcludePunctuation": True, "ExcludeUppercase": True,
                     "IncludeSpace": False, "ExcludeCharacters": "ghijklmnopqrstuvwxyz"}
    assert sm.puts() == {ARNS["basic_auth"]: {
        "user": BASIC_USER, "password": NEW_PASSWORD, "previous_password": OLD_PASSWORD,
        "previous_until": int(NOW) + 8 * 3600}}
    assert [kw["SecretId"] for op, kw in sm.calls if op == "describe_secret"] == [ARNS["basic_auth"]]
    cap = capsys.readouterr()
    assert "--phase post --apply" in cap.out
    no_values(cap.out, cap.err, caplog.text)


def test_pd2_rules_leave_exactly_lower_case_hex() -> None:
    """The character set GetRandomPassword draws from under BASIC_PASSWORD_RULES."""
    import string

    rules = ps.BASIC_PASSWORD_RULES
    pool = set(string.ascii_lowercase + string.digits) - set(rules["ExcludeCharacters"])
    assert rules["ExcludeUppercase"] and rules["ExcludePunctuation"] and not rules["IncludeSpace"]
    assert pool == set("0123456789abcdef") and rules["PasswordLength"] >= 32


def test_new_basic_password_dry_run_makes_no_aws_call(outputs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = ps.main(["--cdk-outputs", str(outputs), "--new-basic-password"], sm_client=Forbidden(),
                 kms_client=Forbidden(), prompt=no_prompt)
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("DRY RUN") and ARNS["basic_auth"] in out and "48 lower-case hex" in out
    assert ARNS["twilio_api_key"] not in out


def test_new_basic_password_refuses_a_bad_generated_value(outputs: Path,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    sm = StubSecretsManager(random_password="ABC")
    rc = ps.main(["--cdk-outputs", str(outputs), "--new-basic-password", "--apply"], sm_client=sm,
                 kms_client=StubKms(), prompt=no_prompt)
    assert rc == 1
    assert "put_secret_value" not in sm.ops()
    assert "field password: format" in capsys.readouterr().err


def test_new_basic_password_stops_on_a_placeholder_secret(outputs: Path,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    sm = StubSecretsManager(basic={"placeholder": True, "unset": OLD_PASSWORD})
    rc = ps.main(["--cdk-outputs", str(outputs), "--new-basic-password", "--apply"], sm_client=sm,
                 kms_client=StubKms(), prompt=no_prompt)
    assert rc == 1
    assert "put_secret_value" not in sm.ops() and "get_random_password" not in sm.ops()
    err = capsys.readouterr().err
    assert "placeholder" in err
    no_values(err)


@pytest.mark.parametrize("old", [{"user": BASIC_USER, "password": "not-hex!"}, {"user": BASIC_USER}])
def test_new_basic_password_reads_only_the_user(outputs: Path, old: dict[str, Any],
                                                capsys: pytest.CaptureFixture[str]) -> None:
    """A malformed or missing old password is what a rotation replaces; it must not block it (and is
    not kept as the previous one)."""
    sm = StubSecretsManager(basic=old)
    rc = ps.main(["--cdk-outputs", str(outputs), "--new-basic-password", "--apply"], sm_client=sm,
                 kms_client=StubKms(), prompt=no_prompt, clock=lambda: NOW)
    assert rc == 0
    assert sm.puts() == {ARNS["basic_auth"]: {"user": BASIC_USER, "password": NEW_PASSWORD}}
    no_values(*capsys.readouterr())


@pytest.mark.parametrize("old", [{"password": OLD_PASSWORD}, {"user": "bad user", "password": OLD_PASSWORD}])
def test_new_basic_password_stops_without_a_valid_user(outputs: Path, old: dict[str, Any],
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    sm = StubSecretsManager(basic=old)
    rc = ps.main(["--cdk-outputs", str(outputs), "--new-basic-password", "--apply"], sm_client=sm,
                 kms_client=StubKms(), prompt=no_prompt)
    assert rc == 1
    assert "put_secret_value" not in sm.ops()
    err = capsys.readouterr().err
    assert "field user" in err
    no_values(err)


def test_new_basic_password_refuses_without_customer_managed_key(outputs: Path) -> None:
    sm = StubSecretsManager(kms_key="alias/aws/secretsmanager")
    assert ps.main(["--cdk-outputs", str(outputs), "--new-basic-password", "--apply"], sm_client=sm,
                   kms_client=StubKms(), prompt=no_prompt) == 3
    assert sm.ops() == ["describe_secret"]


def test_no_keychain_in_the_script() -> None:
    src = (ROOT / "scripts" / "populate_secrets.py").read_text(encoding="utf-8").lower()
    assert "keychain" not in src and "find-generic-password" not in src and "subprocess" not in src


# --------------------------------------------------------------------------- no value leak
#
# For any values, in prompt and file modes, dry run and apply, accepted or refused, no substring of
# 4+ characters of any value appears in stdout, stderr or the log records. Fixed text can't stand in
# for a leak: an example is kept only when none of its values' 4-grams occurs in the source of the
# modules that write the output, the ARNs, the KMS key, the stub's version id, the file path or the checkout's path.

_SOURCES = "\n".join((ROOT / p).read_text(encoding="utf-8") for p in (
    "scripts/populate_secrets.py", "scripts/_common.py", "src/owlbank/secret_schema.py", "src/owlbank/obs.py"))
CORPUS = "\n".join([_SOURCES, *ARNS.values(), CMK, "v-1", "OwlBankData", str(ROOT)]).casefold()

HEX = "0123456789abcdef"
ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
WELL_FORMED = {
    "account_sid": st.text(HEX, min_size=32, max_size=32).map(lambda b: "AC" + b),
    "api_key": st.text(HEX, min_size=32, max_size=32).map(lambda b: "SK" + b),
    "api_secret": st.text(ALNUM, min_size=32, max_size=32),
    "auth_token": st.text(HEX, min_size=32, max_size=32),
}
#: Printable, no whitespace (a prompt answer is stripped).
anything = st.text(st.characters(min_codepoint=33, max_codepoint=0x2FF), min_size=4, max_size=48)


def grams(value: str, n: int = 4) -> set[str]:
    folded = value.casefold()
    return {folded[i:i + n] for i in range(len(folded) - n + 1)}


def leaked(values: list[str], text: str) -> list[str]:
    folded = text.casefold()
    return [v for v in values if any(g in folded for g in grams(v))]


@st.composite
def twilio_values(draw: st.DrawFn) -> dict[str, dict[str, str]]:
    """Both Twilio secrets, each field well formed or anything (so both accept and refuse paths run)."""
    valid = draw(st.booleans())
    return {name: {f: draw(WELL_FORMED[f] if valid else st.one_of(WELL_FORMED[f], anything))
                   for f in ps.SCHEMAS[name]}
            for name in ps.WRITABLE}


@pytest.fixture
def capture_all(caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]) -> Iterator[Any]:
    caplog.set_level(logging.DEBUG)

    def collect() -> str:
        cap = capsys.readouterr()
        text = f"{cap.out}\n{cap.err}\n{text_of(caplog.records)}"
        caplog.clear()
        return text

    yield collect


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture,
                                                                 HealthCheck.filter_too_much])
@given(values=twilio_values(), file_mode=st.booleans(), apply=st.booleans())
def test_no_value_in_output_or_logs(tmp_path: Path, capture_all: Any, values: dict[str, dict[str, str]],
                                        file_mode: bool, apply: bool) -> None:
    flat = [v for fields in values.values() for v in fields.values()]
    corpus = CORPUS + str(tmp_path).casefold()
    assume(not any(g in corpus for v in flat for g in grams(v)))
    capture_all()
    if file_mode:
        f = write_file(tmp_path, values)
        argv, prompt = [*SECRET_ARGS, "--from-file", str(f)], no_prompt
    else:
        argv, prompt = list(SECRET_ARGS), FakePrompt(values)
    sm = StubSecretsManager()
    rc = ps.main([*argv, *(["--apply"] if apply else [])], sm_client=sm, kms_client=StubKms(), prompt=prompt)
    assert rc in (0, 2)
    if apply and rc == 0:
        assert len(sm.puts()) == 2
    assert leaked(flat, capture_all()) == []


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture,
                                                                 HealthCheck.filter_too_much])
@given(old=st.text(HEX, min_size=32, max_size=64), new=st.one_of(st.text(HEX, min_size=48, max_size=48), anything),
       user=st.text(ALNUM + "._-", min_size=4, max_size=64))
def test_new_basic_password_never_shows_the_old_or_new_value(capture_all: Any, old: str, new: str,
                                                                 user: str) -> None:
    flat = [old, new, user]
    assume(not any(g in CORPUS for v in flat for g in grams(v)))
    capture_all()
    sm = StubSecretsManager(basic={"user": user, "password": old}, random_password=new)
    rc = ps.main([*SECRET_ARGS, "--new-basic-password", "--apply"], sm_client=sm, kms_client=StubKms(),
                 prompt=no_prompt)
    assert rc in (0, 1)
    assert leaked(flat, capture_all()) == []


@pytest.mark.parametrize("pair", [{"previous_password": "0f" * 16},
                                  {"previous_password": "0f" * 16, "previous_until": 1_900_000_000_000}])
def test_new_basic_password_ignores_a_malformed_optional_pair(outputs: Path, pair: dict[str, Any]) -> None:
    sm = StubSecretsManager(basic={"user": BASIC_USER, "password": OLD_PASSWORD, **pair})
    rc = ps.main(["--cdk-outputs", str(outputs), "--new-basic-password", "--apply"], sm_client=sm,
                 kms_client=StubKms(), prompt=no_prompt, clock=lambda: NOW)
    assert rc == 0
    assert sm.puts() == {ARNS["basic_auth"]: {
        "user": BASIC_USER, "password": NEW_PASSWORD, "previous_password": OLD_PASSWORD,
        "previous_until": int(NOW) + 8 * 3600}}


# --------------------------------------------------------------------------- rotation with an overlap

from owlbank.secret_schema import MAX_EPOCH_S, validate  # noqa: E402

hex_password = st.text(HEX, min_size=32, max_size=64)


def passwords_in(value: dict[str, Any]) -> list[str]:
    return [value[k] for k in ("password", "previous_password") if k in value]


@settings(max_examples=150, deadline=None)
@given(start=hex_password,
       steps=st.lists(st.one_of(
           st.tuples(st.just("rotate"), hex_password, st.floats(0.01, 500, allow_nan=False),
                     st.integers(0, 86_400)),
           st.tuples(st.just("clear"), st.just(""), st.just(1.0), st.just(0))), max_size=12))
def test_rotation_keeps_at_most_two_passwords_and_previous_is_the_then_current(
        start: str, steps: list[tuple[str, str, float, int]]) -> None:
    value: dict[str, Any] = {"user": BASIC_USER, "password": start}
    now = float(NOW)
    for kind, new, hours, advance in steps:
        now += advance
        before = dict(value)
        if kind == "rotate":
            value = ps.rotate_basic_value(before, new, now=now, overlap_hours=hours)
            assert value["password"] == new
            assert value["previous_password"] == before["password"]  # whatever was previous is dropped
            until = value["previous_until"]
            assert type(until) is int and now < until <= MAX_EPOCH_S
            assert until == int(now + hours * 3600)
        else:
            value = ps.clear_previous_value(before)
            assert "previous_password" not in value and "previous_until" not in value
            assert value["password"] == before["password"]
        assert validate("basic_auth", value) == []
        assert len(passwords_in(value)) <= 2 and set(value) <= {"user", "password", "previous_password",
                                                                 "previous_until"}


def test_second_rotation_drops_the_oldest_password() -> None:
    first = ps.rotate_basic_value({"user": BASIC_USER, "password": "a1" * 20}, "b2" * 20, now=NOW)
    second = ps.rotate_basic_value(first, "c3" * 20, now=NOW + 60)
    assert (second["password"], second["previous_password"]) == ("c3" * 20, "b2" * 20)
    assert "a1" * 20 not in json.dumps(second)


@pytest.mark.parametrize("hours", [0, -1, float("nan"), float("inf"), 1e9, True])
def test_a_bad_overlap_is_refused(hours: Any) -> None:
    with pytest.raises(ps.ScriptError):
        ps.rotate_basic_value({"user": BASIC_USER, "password": OLD_PASSWORD}, NEW_PASSWORD, now=NOW,
                              overlap_hours=hours)


@pytest.mark.parametrize("hours", ["0", "-2", "nan", "inf", "1e9"])
def test_a_bad_overlap_on_the_command_line_is_a_usage_error(outputs: Path, hours: str,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    rc = ps.main(["--cdk-outputs", str(outputs), "--new-basic-password", "--overlap-hours", hours, "--apply"],
                 sm_client=Forbidden(), kms_client=Forbidden(), prompt=no_prompt, clock=lambda: NOW)
    assert rc == 2
    assert "--overlap-hours" in capsys.readouterr().err


def test_overlap_hours_only_goes_with_a_rotation(outputs: Path) -> None:
    rc = ps.main(["--cdk-outputs", str(outputs), "--clear-previous-basic-password", "--overlap-hours", "2"],
                 sm_client=Forbidden(), kms_client=Forbidden(), prompt=no_prompt)
    assert rc == 2


def test_clear_and_new_password_are_exclusive() -> None:
    with pytest.raises(SystemExit):
        ps.main([*SECRET_ARGS, "--new-basic-password", "--clear-previous-basic-password"], prompt=no_prompt)


class RecordingSm(StubSecretsManager):
    """Keeps what is written, so a second command reads the first one's result."""

    def put_secret_value(self, **kw: Any) -> dict[str, Any]:
        self.basic = json.loads(kw["SecretString"])
        return super().put_secret_value(**kw)


def run_basic(outputs: Path, sm: StubSecretsManager, *flags: str, now: float = NOW) -> int:
    return ps.main(["--cdk-outputs", str(outputs), *flags, "--apply"], sm_client=sm, kms_client=StubKms(),
                   prompt=no_prompt, clock=lambda: now)


def test_two_rotations_then_a_clear_end_to_end(outputs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sm = RecordingSm()
    assert run_basic(outputs, sm, "--new-basic-password", "--overlap-hours", "2") == 0
    assert sm.basic == {"user": BASIC_USER, "password": NEW_PASSWORD, "previous_password": OLD_PASSWORD,
                        "previous_until": int(NOW) + 2 * 3600}
    sm.random_password = "d5" * 24
    assert run_basic(outputs, sm, "--new-basic-password", now=NOW + 600) == 0
    assert sm.basic == {"user": BASIC_USER, "password": "d5" * 24, "previous_password": NEW_PASSWORD,
                        "previous_until": int(NOW + 600) + 8 * 3600}
    assert OLD_PASSWORD not in json.dumps(sm.basic)
    capsys.readouterr()
    assert run_basic(outputs, sm, "--clear-previous-basic-password") == 0
    assert sm.basic == {"user": BASIC_USER, "password": "d5" * 24}
    out = capsys.readouterr().out
    assert "60 s" in out and "d5" * 24 not in out


def test_rotation_prints_the_deadline_the_wait_and_the_twilio_step(
        outputs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_basic(outputs, RecordingSm(), "--new-basic-password") == 0
    out = capsys.readouterr().out
    assert f"apply by {ps.deadline_text(int(NOW) + 8 * 3600)}" in out
    assert "wait about 60 s" in out and "twilio_setup.py --phase post --apply" in out
    assert "drops the oldest password" in out


def test_rotation_dry_run_makes_no_aws_call_and_names_the_overlap(
        outputs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = ps.main(["--cdk-outputs", str(outputs), "--new-basic-password", "--overlap-hours", "3"],
                 sm_client=Forbidden(), kms_client=Forbidden(), prompt=no_prompt)
    assert rc == 0 and "for 3 h" in capsys.readouterr().out


def test_clear_dry_run_makes_no_aws_call(outputs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = ps.main(["--cdk-outputs", str(outputs), "--clear-previous-basic-password"],
                 sm_client=Forbidden(), kms_client=Forbidden(), prompt=no_prompt)
    assert rc == 0 and capsys.readouterr().out.startswith("DRY RUN")


def test_clear_refuses_a_secret_whose_password_is_not_well_formed(outputs: Path) -> None:
    sm = StubSecretsManager(basic={"user": BASIC_USER, "password": "bad!", "previous_password": OLD_PASSWORD,
                                   "previous_until": int(NOW)})
    assert run_basic(outputs, sm, "--clear-previous-basic-password") == 1
    assert "put_secret_value" not in sm.ops()


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture,
                                                                 HealthCheck.filter_too_much])
@given(old=hex_password, prev=hex_password, new=hex_password, user=st.text(ALNUM + "._-", min_size=4, max_size=64),
       clear=st.booleans(), apply=st.booleans())
def test_rotation_and_clear_never_show_any_password(outputs_dir: Path, capture_all: Any, old: str, prev: str,
                                                   new: str, user: str, clear: bool, apply: bool) -> None:
    flat = [old, prev, new, user]
    assume(not any(g in CORPUS + str(outputs_dir).casefold() for v in flat for g in grams(v)))
    assume(len({old, prev, new}) == 3)
    capture_all()
    sm = StubSecretsManager(basic={"user": user, "password": old, "previous_password": prev,
                                   "previous_until": int(NOW) + 100}, random_password=new)
    flag = "--clear-previous-basic-password" if clear else "--new-basic-password"
    rc = ps.main(["--cdk-outputs", str(outputs_dir / "cdk-outputs.json"), flag, *(["--apply"] if apply else [])],
                 sm_client=sm, kms_client=StubKms(), prompt=no_prompt, clock=lambda: NOW)
    assert rc == 0
    if apply:
        assert sm.puts()  # the write happened, with passwords inside the secret only
    assert leaked(flat, capture_all()) == []


@pytest.fixture(scope="module")
def outputs_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("basic-outputs")
    (d / "cdk-outputs.json").write_text(json.dumps({"OwlBankData": {
        "SecretTwilioApiKey": ARNS["twilio_api_key"],
        "TwilioAuthTokenSecretArn": ARNS["twilio_auth_token"],
        "SecretBasicAuthE5F6A7B8": ARNS["basic_auth"]}}))
    return d


# --------------------------------------------------------------------------- rotation warnings

SECOND_ROTATION = "a second rotation drops the oldest password"


def test_a_rotation_over_an_active_previous_password_warns_before_writing_and_in_the_next_steps(
        outputs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    seen_at_write: list[str] = []

    class Spy(RecordingSm):
        def put_secret_value(self, **kw: Any) -> dict[str, Any]:
            seen_at_write.append(capsys.readouterr().out)
            return super().put_secret_value(**kw)

    sm = Spy(basic={"user": BASIC_USER, "password": NEW_PASSWORD, "previous_password": OLD_PASSWORD,
                    "previous_until": int(NOW) + 3600}, random_password="d5" * 24)
    assert run_basic(outputs, sm, "--new-basic-password") == 0
    assert SECOND_ROTATION in seen_at_write[0]  # printed before the write
    steps = capsys.readouterr().out
    assert SECOND_ROTATION in steps and "twilio_setup.py --phase post --apply" in steps
    assert not {OLD_PASSWORD, NEW_PASSWORD, "d5" * 24} & set(re.findall(r"[0-9a-f]{48}", seen_at_write[0] + steps))


@pytest.mark.parametrize("basic", [
    {"user": BASIC_USER, "password": OLD_PASSWORD},
    {"user": BASIC_USER, "password": OLD_PASSWORD, "previous_password": NEW_PASSWORD,
     "previous_until": int(NOW) - 1},
])
def test_a_rotation_without_an_active_previous_password_has_no_second_rotation_warning(
        outputs: Path, capsys: pytest.CaptureFixture[str], basic: dict[str, Any]) -> None:
    assert run_basic(outputs, RecordingSm(basic=basic), "--new-basic-password") == 0
    assert SECOND_ROTATION not in capsys.readouterr().out


def test_the_dry_run_shows_the_second_rotation_warning(outputs: Path,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    rc = ps.main(["--cdk-outputs", str(outputs), "--new-basic-password"], sm_client=Forbidden(),
                 kms_client=Forbidden(), prompt=no_prompt, clock=lambda: NOW)
    assert rc == 0 and SECOND_ROTATION in capsys.readouterr().out


@pytest.mark.parametrize(("hours", "warned"), [("1", True), ("5.9", True), ("6", False), ("8", False)])
@pytest.mark.parametrize("apply", [False, True])
def test_a_short_overlap_is_warned_about_but_not_refused(outputs: Path, capsys: pytest.CaptureFixture[str],
                                                        hours: str, warned: bool, apply: bool) -> None:
    argv = ["--cdk-outputs", str(outputs), "--new-basic-password", "--overlap-hours", hours]
    rc = ps.main([*argv, *(["--apply"] if apply else [])], sm_client=RecordingSm() if apply else Forbidden(),
                 kms_client=StubKms() if apply else Forbidden(), prompt=no_prompt, clock=lambda: NOW)
    out = capsys.readouterr().out
    assert rc == 0
    assert ("is shorter than 6" in out) is warned
