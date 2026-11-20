"""scripts/twilio_setup.py: the Restricted API keys for the Context API and the Flex authorizer.

The setup script creates each key with Twilio's Key resource (``POST https://iam.twilio.com/v1/Keys``,
``KeyType=restricted``, a ``Policy`` allow list) using the account's Auth Token, and writes the answer
straight to the key's own Secrets Manager secret. The key's secret is never printed or logged.
Fakes stand in for Twilio, Secrets Manager and KMS: no network.

No PBT: flag handling and one create-or-skip step around two fixed requests (glue).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import twilio_setup as ts
from tests.logutil import text_of
from tests.unit.scripts.test_twilio_setup import (
    API_KEY_ARN, API_SECRET, API_SID, AUTH_TOKEN, BASE, BASIC_ARN, SECRET_VALUES, SUB, TOKEN_ARN, FakeTwilio, NoSM,
)
from tests.unit.scripts.test_twilio_setup import (  # noqa: F401 (autouse: no real Secrets Manager client)
    _no_real_secrets_manager,
)

ARN = "arn:aws:secretsmanager:us-east-1:123456789012:secret:owlbank/twilio/"
CTX_ARN = ARN + "context-api-key-StUvWx"
NEW_SID = "SK" + "c3" * 16
NEW_SECRET = "RestrictedSecretValue0123456789Z"  # 32 letters and digits
OLD_SID = "SK" + "d4" * 16
OTHER_ACCOUNT = "AC" + "9" * 32
PLACEHOLDER = {"placeholder": True, "unset": "x"}
CMK = "arn:aws:kms:us-east-1:123456789012:key/11111111-2222-3333-4444-555555555555"
CONTEXT_PERMISSIONS = ["/twilio/taskrouter/workspaces.tasks/read",
                       "/twilio/taskrouter/workspaces.tasks.reservations/list"]


class FakeSM:
    """``get_secret_value``, ``describe_secret`` and ``put_secret_value`` over values by ARN."""

    def __init__(self, values: dict[str, Any], *, kms_key: str = CMK, put_fails: bool = False) -> None:
        self.values = {**SECRET_VALUES, **values}
        self.kms_key, self.put_fails = kms_key, put_fails
        self.reads: list[str] = []
        self.puts: list[tuple[str, dict[str, str]]] = []

    def get_secret_value(self, *, SecretId: str) -> dict[str, str]:  # noqa: N803 (boto3's name)
        self.reads.append(SecretId)
        return {"ARN": SecretId, "SecretString": json.dumps(self.values[SecretId])}

    def describe_secret(self, *, SecretId: str) -> dict[str, str]:  # noqa: N803
        return {"ARN": SecretId, "KmsKeyId": self.kms_key}

    def put_secret_value(self, *, SecretId: str, SecretString: str) -> dict[str, str]:  # noqa: N803
        if self.put_fails:
            raise RuntimeError(f"boom {SecretString}")  # a client error must never be echoed
        self.puts.append((SecretId, json.loads(SecretString)))
        return {"VersionId": "v2"}


class FakeKMS:
    def describe_key(self, *, KeyId: str) -> dict[str, Any]:  # noqa: N803
        return {"KeyMetadata": {"KeyId": KeyId, "KeyManager": "CUSTOMER"}}


class FakeIam:
    """The Auth-Token client's ``request``: answers each key create with :data:`NEW_SID`."""

    def __init__(self, *, status: int = 201, body: dict[str, Any] | None = None) -> None:
        self.status = status
        self.body = {"sid": NEW_SID, "secret": NEW_SECRET, "friendly_name": "x"} if body is None else body
        self.requests: list[tuple[str, str, dict[str, str]]] = []

    def request(self, method: str, uri: str, data: dict[str, str] | None = None, **_kw: Any) -> Any:
        self.requests.append((method, uri, dict(data or {})))
        return SimpleNamespace(status_code=self.status, text=json.dumps(self.body))


class AuthClient:
    """The Auth-Token client: ``request`` for the key create (:class:`FakeIam`), Accounts as the fake's."""

    def __init__(self, iam: FakeIam, twilio: FakeTwilio) -> None:
        self.request, self.api = iam.request, twilio.factory(SUB, AUTH_TOKEN, SUB).api


class Clients:
    """``client_factory``: the Standard-key client for setup, the Auth-Token client for the keys and
    the account check. The Standard-key client refuses the Accounts resource (70004), as Twilio does."""

    def __init__(self, iam: FakeIam | None = None) -> None:
        self.twilio, self.iam = FakeTwilio(), iam or FakeIam()
        self.made: list[tuple[str, str, str]] = []

    def __call__(self, user: str, password: str, account: str) -> Any:
        self.made.append((user, password, account))
        if (user, password, account) == (API_SID, API_SECRET, SUB):
            return self.twilio.factory(user, password, account)
        assert (user, password, account) == (SUB, AUTH_TOKEN, SUB), "unexpected credentials"
        return AuthClient(self.iam, self.twilio)


def apply(tmp_path: Path, sm: FakeSM, clients: Clients, extra: list[str] | None = None,
          *, phase: str = "pre") -> int:
    args = [*BASE, "--phase", phase, "--account-sid", SUB, "--apply", "--output", str(tmp_path / "tw.json"),
            "--secret", f"twilio_api_key={API_KEY_ARN}", "--secret", f"basic_auth={BASIC_ARN}",
            "--secret", f"twilio_auth_token={TOKEN_ARN}",
            "--restricted-key-secret", f"context_api_key={CTX_ARN}", *(extra or [])]
    return ts.main(args, sm_client=sm, kms_client=FakeKMS(), client_factory=clients)


def no_client(*_a: Any) -> Any:
    raise AssertionError("Twilio client built in a dry run")


def _key(account: str = SUB, sid: str = OLD_SID) -> dict[str, str]:
    return {"account_sid": account, "api_key": sid, "api_secret": "OldSecretValue0123456789abcdefgh"}


# --------------------------------------------------------------------------- dry run


def test_dry_run_lists_the_context_api_key_and_its_two_permissions(capsys: pytest.CaptureFixture[str]) -> None:
    assert ts.main(["--phase", "pre"], sm_client=NoSM(), client_factory=no_client) == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "ensure_restricted_key" in ln]
    assert len(lines) == 1 and "context_api_key" in lines[0]
    for text in (*CONTEXT_PERMISSIONS, "https://iam.twilio.com/v1/Keys", "KeyType=restricted",
                 "owlbank/twilio/context-api-key", "contextApiRestrictedKey=on", "Auth Token"):
        assert text in lines[0], text


def test_keys_come_before_the_trait_group_which_stays_last() -> None:
    cfg = ts.SetupConfig(phase="pre", restricted_keys=("context_api_key",),
                         queue_names={k: k for k, *_ in ts.QUEUES})
    kinds = [s.kind for s in ts.build_plan(cfg)]
    assert kinds[-2:] == ["ensure_restricted_key", "ensure_trait_group"]


def test_skip_leaves_the_context_api_key_out(capsys: pytest.CaptureFixture[str]) -> None:
    assert ts.main(["--phase", "pre", "--skip-context-api-key"], sm_client=NoSM(), client_factory=no_client) == 0
    assert "ensure_restricted_key" not in capsys.readouterr().out


def test_post_makes_no_key_and_refuses_rotation(capsys: pytest.CaptureFixture[str]) -> None:
    assert ts.main([*BASE, "--phase", "post"], sm_client=NoSM(), client_factory=no_client) == 0
    assert "ensure_restricted_key" not in capsys.readouterr().out
    assert ts.main([*BASE, "--phase", "post", "--rotate-restricted-keys"], sm_client=NoSM()) == 2


def test_restricted_keys_for() -> None:
    assert ts.restricted_keys_for("pre", skip_context_api_key=False) == ("context_api_key",)
    assert ts.restricted_keys_for("all", skip_context_api_key=True) == ()
    assert ts.restricted_keys_for("post", skip_context_api_key=False) == ()


# --------------------------------------------------------------------------- apply


def test_apply_creates_the_key_and_writes_it_only_to_its_secret(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    sm, clients = FakeSM({CTX_ARN: PLACEHOLDER}), Clients()
    assert apply(tmp_path, sm, clients) == 0
    ((method, uri, data),) = clients.iam.requests
    assert (method, uri) == ("POST", "https://iam.twilio.com/v1/Keys")
    assert data["KeyType"] == "restricted" and data["AccountSid"] == SUB and data["FriendlyName"]
    assert json.loads(data["Policy"]) == {"allow": CONTEXT_PERMISSIONS}
    assert sm.puts == [(CTX_ARN, {"account_sid": SUB, "api_key": NEW_SID, "api_secret": NEW_SECRET})]
    # The Auth Token is read for the account check and the key create; the shared key secret is never written.
    assert TOKEN_ARN in sm.reads and API_KEY_ARN not in [arn for arn, _ in sm.puts]
    captured = capsys.readouterr()
    assert NEW_SID in captured.out
    for leaked in (NEW_SECRET, AUTH_TOKEN):
        assert leaked not in captured.out + captured.err + text_of(caplog.records)
        assert leaked not in (tmp_path / "tw.json").read_text()


def test_a_secret_that_already_holds_a_key_is_left_alone(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sm, clients = FakeSM({CTX_ARN: _key()}), Clients()
    assert apply(tmp_path, sm, clients) == 0
    assert clients.iam.requests == [] and sm.puts == []
    assert clients.twilio.refused == []
    assert f"({OLD_SID}) already in" in capsys.readouterr().out


def test_rotate_creates_a_new_key_and_names_the_old_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sm, clients = FakeSM({CTX_ARN: _key()}), Clients()
    assert apply(tmp_path, sm, clients, ["--rotate-restricted-keys"]) == 0
    assert len(clients.iam.requests) == 1
    assert sm.puts == [(CTX_ARN, {"account_sid": SUB, "api_key": NEW_SID, "api_secret": NEW_SECRET})]
    out = capsys.readouterr().out
    assert f"delete the old key {OLD_SID}" in out and NEW_SECRET not in out


def test_a_key_for_another_account_stops_the_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sm, clients = FakeSM({CTX_ARN: _key(OTHER_ACCOUNT)}), Clients()
    assert apply(tmp_path, sm, clients) == 3
    assert clients.iam.requests == [] and sm.puts == []
    assert "--rotate-restricted-keys" in capsys.readouterr().err


def test_the_arns_come_from_the_data_stack_outputs(tmp_path: Path) -> None:
    outputs = tmp_path / "cdk-outputs.json"
    outputs.write_text(json.dumps({"OwlBankData": {
        "SecretTwilioApiKey": API_KEY_ARN, "SecretTwilioAuthToken": TOKEN_ARN, "SecretBasicAuth": BASIC_ARN,
        "SecretContextApiKey": CTX_ARN}}))
    sm, clients = FakeSM({CTX_ARN: PLACEHOLDER}), Clients()
    args = ["--phase", "pre", "--account-sid", SUB, "--apply", "--output", str(tmp_path / "tw.json"),
            "--cdk-outputs", str(outputs)]
    assert ts.main(args, sm_client=sm, kms_client=FakeKMS(), client_factory=clients) == 0
    assert [arn for arn, _ in sm.puts] == [CTX_ARN]


def test_a_missing_key_secret_arn_stops_before_any_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--phase", "pre", "--account-sid", SUB, "--apply", "--output", str(tmp_path / "tw.json"),
            "--secret", f"twilio_api_key={API_KEY_ARN}", "--secret", f"twilio_auth_token={TOKEN_ARN}"]
    assert ts.main(args, sm_client=NoSM(), client_factory=no_client) == 2
    err = capsys.readouterr().err
    assert "SecretContextApiKey" in err and "contextApiRestrictedKey=on" in err and "--skip-context-api-key" in err


def test_an_unknown_key_secret_name_is_refused(tmp_path: Path) -> None:
    args = ["--phase", "pre", "--account-sid", SUB, "--apply", "--output", str(tmp_path / "tw.json"),
            "--secret", f"twilio_api_key={API_KEY_ARN}", "--restricted-key-secret", f"other_key={CTX_ARN}"]
    assert ts.main(args, sm_client=NoSM(), client_factory=no_client) == 2


def test_twilio_refusal_names_status_and_code_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    iam = FakeIam(status=403, body={"code": 20003, "message": "denied", "more_info": NEW_SECRET})
    sm = FakeSM({CTX_ARN: PLACEHOLDER})
    assert apply(tmp_path, sm, Clients(iam)) == 3
    err = capsys.readouterr().err
    assert "HTTP 403" in err and "20003" in err and NEW_SECRET not in err
    assert sm.puts == []


def test_an_answer_without_a_usable_secret_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sm = FakeSM({CTX_ARN: PLACEHOLDER})
    assert apply(tmp_path, sm, Clients(FakeIam(body={"sid": NEW_SID}))) == 3
    assert sm.puts == [] and f"delete key {NEW_SID}" in capsys.readouterr().err


def test_a_failed_write_names_the_new_key_but_never_its_secret(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    sm = FakeSM({CTX_ARN: PLACEHOLDER}, put_fails=True)
    assert apply(tmp_path, sm, Clients()) == 3
    captured = capsys.readouterr()
    assert f"delete key {NEW_SID}" in captured.err
    assert NEW_SECRET not in captured.out + captured.err + text_of(caplog.records)


def test_an_aws_managed_kms_key_is_refused_before_the_create(tmp_path: Path) -> None:
    sm, clients = FakeSM({CTX_ARN: PLACEHOLDER}, kms_key="alias/aws/secretsmanager"), Clients()
    assert apply(tmp_path, sm, clients) == 3
    assert clients.iam.requests == [] and sm.puts == []


# --------------------------------------------------------------------------- --check


def check(sm: FakeSM, extra: list[str] | None = None) -> int:
    args = [*BASE, "--phase", "pre", "--account-sid", SUB, "--check",
            "--secret", f"twilio_api_key={API_KEY_ARN}", "--secret", f"basic_auth={BASIC_ARN}",
            "--secret", f"twilio_auth_token={TOKEN_ARN}",
            "--restricted-key-secret", f"context_api_key={CTX_ARN}", *(extra or [])]
    return ts.main(args, sm_client=sm, kms_client=FakeKMS(), client_factory=Clients())


def test_check_reports_the_key_the_secret_holds_like_apply(capsys: pytest.CaptureFixture[str]) -> None:
    assert check(FakeSM({CTX_ARN: _key()})) == 0
    out = capsys.readouterr().out
    assert OLD_SID in out and "apply keeps it" in out and "api_secret" not in out
    assert "OldSecretValue" not in out


def test_check_reports_a_placeholder_secret_as_to_create(capsys: pytest.CaptureFixture[str]) -> None:
    assert check(FakeSM({CTX_ARN: PLACEHOLDER})) == 0
    assert "apply will create one" in capsys.readouterr().out


def test_check_reports_rotation_and_another_account(capsys: pytest.CaptureFixture[str]) -> None:
    assert check(FakeSM({CTX_ARN: _key()}), ["--rotate-restricted-keys"]) == 0
    assert "apply will create a new key and replace it" in capsys.readouterr().out
    assert check(FakeSM({CTX_ARN: _key(OTHER_ACCOUNT)})) == 3
    out = capsys.readouterr().out
    assert "another account" in out and "apply will stop" in out
