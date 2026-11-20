"""The deploy scripts log a start line, one line per step with its outcome, and errors
with context, and never a secret value (the scripts still never print one either)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from owlbank.store import MemoryStore
from scripts import populate_secrets as ps
from tests.logutil import fields, text_of
from tests.unit.scripts.test_populate_secrets import ARNS, FakePrompt, Forbidden, VALUES
from tests.unit.scripts.test_seed import TABLE_ARGS, run
from tests.unit.scripts.test_twilio_setup import API_SECRET as TW_SECRET, PASSWORD, FakeTwilio, run_apply


def script_lines(caplog: pytest.LogCaptureFixture, script: str) -> list[dict]:
    return [fields(r) | {"_level": r.levelname} for r in caplog.records
            if getattr(r, "event", "").startswith("script_") and fields(r).get("script") == script]


def test_seed_apply_logs_start_a_line_per_step_and_the_counts(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    store = MemoryStore()
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    lines = script_lines(caplog, "seed")
    assert lines[0]["event"] == "script_start" and lines[0]["mode"] == "apply" and lines[0]["stage"] == "prod"
    steps = {ln["step"]: ln for ln in lines if ln["event"] == "script_step"}
    assert {"plan", "aliases_delete", "customers", "aliases", "policy", "catalog"} <= set(steps)
    assert all(ln["outcome"] == "ok" for ln in steps.values()) and steps["customers"]["written"] > 0


def test_seed_dry_run_and_errors(caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]) -> None:
    caplog.set_level(logging.DEBUG)
    assert run(["--stage", "dev", *TABLE_ARGS], MemoryStore()) == 0
    assert any(ln.get("outcome") == "dry_run" for ln in script_lines(caplog, "seed"))
    caplog.clear()
    assert run(["--stage", "dev", "--apply"], MemoryStore()) == 2  # no table names
    (err,) = [ln for ln in script_lines(caplog, "seed") if ln["event"] == "script_error"]
    assert err["decision"] == "exit_2" and err["error_class"] == "ScriptError" and "no table name" in err["error_message"]
    assert err["_level"] == "ERROR"


def test_seed_apply_failure_is_logged_with_context_and_reraised(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    class Failing(MemoryStore):
        def put(self, table, item, *, condition=None):  # type: ignore[override]
            raise RuntimeError("write failed for +12025550123")

    with pytest.raises(RuntimeError):
        run(["--stage", "prod", *TABLE_ARGS, "--apply"], Failing())
    lines = script_lines(caplog, "seed")
    err = next(ln for ln in lines if ln["event"] == "script_error")
    assert err["step"] == "customers" and err["decision"] == "reraise" and err["error_class"] == "RuntimeError"
    assert any(ln["event"] == "script_step" and ln["outcome"] == "error" and ln["step"] == "customers" for ln in lines)
    assert "+12025550123" not in str([f for f in lines if "error_message" in f])  # a non-script error has no message


def _outputs(tmp_path: Path) -> Path:
    p = tmp_path / "cdk-outputs.json"
    p.write_text(json.dumps({"S": {"SecretTwilioApiKey": ARNS["twilio_api_key"],
                                   "TwilioAuthTokenSecretArn": ARNS["twilio_auth_token"],
                                   "SecretBasicAuthE5F6A7B8": ARNS["basic_auth"]}}))
    return p


def test_populate_secrets_logs_steps_and_never_a_value(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    rc = ps.main(["--cdk-outputs", str(_outputs(tmp_path))], sm_client=Forbidden(), kms_client=Forbidden(),
                 prompt=FakePrompt())
    assert rc == 0
    lines = script_lines(caplog, "populate_secrets")
    assert lines[0]["event"] == "script_start" and lines[0]["mode"] == "dry_run"
    assert lines[0]["source"] == "prompt"
    assert [ln["step"] for ln in lines if ln["event"] == "script_step"] == ["read_values", "apply"]
    flat = text_of(caplog.records)
    for v in VALUES:
        assert v not in flat


def test_populate_secrets_put_failure_is_an_error_line_with_the_aws_code(tmp_path: Path,
                                                                         caplog: pytest.LogCaptureFixture) -> None:
    from botocore.exceptions import ClientError

    caplog.set_level(logging.DEBUG)
    cmk = "arn:aws:kms:us-east-1:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab"

    class SM:
        def describe_secret(self, SecretId: str) -> dict:
            return {"KmsKeyId": cmk}

        def put_secret_value(self, **kw):
            raise ClientError({"Error": {"Code": "AccessDeniedException", "Message": json.dumps(kw)}}, "Put")

    class KMS:
        def describe_key(self, KeyId: str) -> dict:
            return {"KeyMetadata": {"KeyManager": "CUSTOMER"}}

    rc = ps.main(["--cdk-outputs", str(_outputs(tmp_path)), "--apply"], sm_client=SM(), kms_client=KMS(),
                 prompt=FakePrompt())
    assert rc == 1
    errs = [ln for ln in script_lines(caplog, "populate_secrets") if ln["event"] == "script_error"]
    assert errs and all(e["error_code"] == "AccessDeniedException" and e["decision"] == "continue_with_next_secret" for e in errs)
    flat = text_of(caplog.records)
    for v in VALUES:
        assert v not in flat


def test_populate_secrets_refused_file_is_an_exit_2_error_line(tmp_path: Path,
                                                               caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    f = tmp_path / "twilio.json"
    f.write_text(json.dumps({"twilio_auth_token": {"auth_token": VALUES[1]}}))
    f.chmod(0o644)
    rc = ps.main(["--cdk-outputs", str(_outputs(tmp_path)), "--from-file", str(f), "--apply"],
                 sm_client=Forbidden(), kms_client=Forbidden(), prompt=FakePrompt())
    assert rc == 2
    (err,) = [ln for ln in script_lines(caplog, "populate_secrets") if ln["event"] == "script_error"]
    assert err["step"] == "read_values" and err["decision"] == "exit_2" and "readable by group" in err["error_message"]
    flat = text_of(caplog.records)
    for v in VALUES:
        assert v not in flat


def test_twilio_setup_logs_each_step_and_no_credential(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    assert run_apply(tmp_path, FakeTwilio()) == 0
    lines = script_lines(caplog, "twilio_setup")
    assert lines[0]["event"] == "script_start"
    steps = [ln["step"] for ln in lines if ln["event"] == "script_step" and ln["outcome"] == "ok"]
    assert steps[:3] == ["plan", "check_account", "configure_number"] and steps[-1] == "write_output"
    assert {"ensure_queue", "ensure_workflow", "ensure_verify_service"} <= set(steps)
    flat = text_of(caplog.records)
    assert PASSWORD not in flat and TW_SECRET not in flat and "+12065550007" not in flat
    assert any(r.name == "owlbank.dependency" and fields(r)["dependency"] == "twilio" for r in caplog.records)
