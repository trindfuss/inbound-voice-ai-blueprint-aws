"""AWS Region for the scripts: ``scripts._common.resolve_region`` and its use in
``populate_secrets.py`` and ``seed.py``.

The SSO profile has no default Region and boto3 ignores ``AWS_REGION``, so a script that built a
client without a Region raised ``NoRegionError``. Precedence: ``--region``, the Region in the ARN
of the secret or table, ``AWS_DEFAULT_REGION`` or ``AWS_REGION``, then us-east-1 with an INFO line.
The environment is injected, so no test depends on the machine's.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts import populate_secrets as ps
from scripts import seed
from scripts._common import DEFAULT_REGION, ScriptError, region_from_arn, resolve_region
from tests.unit.scripts.test_populate_secrets import (
    CMK,
    FakePrompt,
    StubKms,
    StubSecretsManager,
)

EAST = "arn:aws:secretsmanager:us-east-1:111122223333:secret:twilio/api-key-AbC"
WEST = "arn:aws:secretsmanager:us-west-2:111122223333:secret:twilio/api-key-AbC"
REGIONS = st.sampled_from(["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-2", "eu-central-1"])


# --------------------------------------------------------------------------- region_from_arn


@pytest.mark.parametrize("arn, region", [
    (EAST, "us-east-1"),
    (WEST, "us-west-2"),
    ("arn:aws:dynamodb:eu-west-1:111122223333:table/owlbank-customers", "eu-west-1"),
    ("arn:aws-us-gov:kms:us-gov-west-1:111122223333:key/abc", "us-gov-west-1"),
    ("arn:aws:iam::111122223333:role/x", None),      # global: no Region
    ("arn:aws:s3:::bucket", None),
    ("owlbank-customers", None),                     # a plain name
    ("", None),
])
def test_region_from_arn(arn: str, region: str | None) -> None:
    assert region_from_arn(arn) == region


# --------------------------------------------------------------------------- precedence, one source at a time


def test_explicit_region_wins_over_everything() -> None:
    env = {"AWS_DEFAULT_REGION": "eu-west-1", "AWS_REGION": "eu-west-2"}
    assert resolve_region("ap-southeast-2", arns=[EAST], env=env) == "ap-southeast-2"


def test_arn_region_beats_the_environment() -> None:
    env = {"AWS_DEFAULT_REGION": "eu-west-1", "AWS_REGION": "eu-west-2"}
    assert resolve_region(None, arns=[WEST], env=env) == "us-west-2"
    assert resolve_region("", arns=["not-an-arn", WEST], env=env) == "us-west-2"


def test_aws_default_region_beats_aws_region() -> None:
    assert resolve_region(None, env={"AWS_DEFAULT_REGION": "eu-west-1", "AWS_REGION": "eu-west-2"}) == "eu-west-1"


def test_aws_region_is_read_when_boto3s_variable_is_unset() -> None:
    assert resolve_region(None, env={"AWS_REGION": "eu-west-2"}) == "eu-west-2"
    assert resolve_region(None, env={"AWS_DEFAULT_REGION": "", "AWS_REGION": "eu-west-2"}) == "eu-west-2"


def test_default_is_us_east_1_with_an_info_line(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    assert resolve_region(None, arns=["a-table-name"], env={}) == DEFAULT_REGION == "us-east-1"
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1 and "us-east-1" in infos[0].getMessage()


def test_no_default_line_when_a_source_was_found(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    resolve_region("eu-west-1", env={})
    resolve_region(None, arns=[EAST], env={})
    resolve_region(None, env={"AWS_REGION": "eu-west-1"})
    assert not [r for r in caplog.records if "using us-east-1" in r.getMessage()]


def test_arns_in_two_regions_are_an_error_unless_a_region_is_given() -> None:
    with pytest.raises(ScriptError, match="--region"):
        resolve_region(None, arns=[EAST, WEST], env={})
    assert resolve_region("us-east-1", arns=[EAST, WEST], env={}) == "us-east-1"
    assert resolve_region(None, arns=[EAST, EAST], env={}) == "us-east-1"


def test_the_process_environment_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setenv("AWS_REGION", "eu-west-3")
    assert resolve_region() == "eu-west-3"


@given(explicit=st.none() | REGIONS, arn_region=st.none() | REGIONS, default_env=st.none() | REGIONS,
       region_env=st.none() | REGIONS)
def test_precedence_property(explicit: str | None, arn_region: str | None, default_env: str | None,
                             region_env: str | None) -> None:
    """Whichever sources are present, the highest-precedence one is the result; with none, us-east-1."""
    env = {k: v for k, v in (("AWS_DEFAULT_REGION", default_env), ("AWS_REGION", region_env)) if v}
    arns = [f"arn:aws:secretsmanager:{arn_region}:111122223333:secret:s-AbC"] if arn_region else []
    assert resolve_region(explicit, arns=arns, env=env) == (
        explicit or arn_region or default_env or region_env or "us-east-1")


# --------------------------------------------------------------------------- populate_secrets.py


@pytest.fixture
def boto_clients(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str | None]]:
    """Replace ``boto3.client`` with stubs; returns the (service, region_name) of each call."""
    import boto3

    made: list[tuple[str, str | None]] = []

    def fake_client(service: str, **kw: Any) -> Any:
        made.append((service, kw.get("region_name")))
        return StubSecretsManager(CMK) if service == "secretsmanager" else StubKms()

    monkeypatch.setattr(boto3, "client", fake_client)
    for var in ("AWS_DEFAULT_REGION", "AWS_REGION"):
        monkeypatch.delenv(var, raising=False)
    return made


def _populate(*extra: str, arn: str = "twilio/api-key") -> int:
    return ps.main(["--secret", f"twilio_api_key={arn}", "--only", "twilio_api_key",
                    *extra, "--apply"], prompt=FakePrompt())


def test_populate_uses_the_explicit_region(boto_clients: list[tuple[str, str | None]]) -> None:
    assert _populate("--region", "eu-west-1", arn=EAST) == 0
    assert boto_clients == [("secretsmanager", "eu-west-1"), ("kms", "eu-west-1")]


def test_populate_uses_the_secret_arns_region(boto_clients: list[tuple[str, str | None]],
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    assert _populate(arn=WEST) == 0
    assert boto_clients == [("secretsmanager", "us-west-2"), ("kms", "us-west-2")]


def test_populate_reads_aws_default_region_then_aws_region(boto_clients: list[tuple[str, str | None]],
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "eu-west-2")
    assert _populate() == 0
    assert {r for _, r in boto_clients} == {"eu-west-2"}
    boto_clients.clear()
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    assert _populate() == 0
    assert {r for _, r in boto_clients} == {"eu-west-1"}


def test_populate_falls_back_to_us_east_1_and_logs_it(boto_clients: list[tuple[str, str | None]],
                                                      caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    assert _populate() == 0
    assert boto_clients == [("secretsmanager", "us-east-1"), ("kms", "us-east-1")]
    assert any("using us-east-1" in r.getMessage() for r in caplog.records)


def test_populate_with_arns_in_two_regions_exits_2(boto_clients: list[tuple[str, str | None]],
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    rc = ps.main(["--secret", f"twilio_api_key={EAST}",
                  "--secret", f"twilio_auth_token={WEST}", "--only", "twilio_api_key",
                  "--only", "twilio_auth_token", "--apply"], prompt=FakePrompt())
    assert rc == 2 and boto_clients == [] and "--region" in capsys.readouterr().err


def test_populate_dry_run_needs_no_region(boto_clients: list[tuple[str, str | None]]) -> None:
    assert ps.main(["--secret", "twilio_api_key=twilio/api-key", "--only", "twilio_api_key"],
                   prompt=FakePrompt()) == 0
    assert boto_clients == []


# --------------------------------------------------------------------------- seed.py


class RecordingStore:
    regions: list[str | None] = []

    def __init__(self, tables: Any, *, region_name: str | None = None, **_: Any) -> None:
        RecordingStore.regions.append(region_name)

    def get(self, table: str, key: Any) -> None:  # the seed reads the live masking values first
        return None

    def scan(self, table: str) -> list[Any]:  # ...and the live catalog
        return []


@pytest.fixture
def seed_region(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    RecordingStore.regions = []
    monkeypatch.setattr(seed, "DynamoStore", RecordingStore)
    monkeypatch.setattr(seed, "apply_plan", lambda plan, store: {"customers_written": 0})
    for var in ("AWS_DEFAULT_REGION", "AWS_REGION"):
        monkeypatch.delenv(var, raising=False)
    return RecordingStore.regions


def _seed(*extra: str, table: str = "t-{}") -> int:
    tables = [a for name in ("customers", "policy", "catalog", "calls", "segments", "tokens", "risk")
              for a in ("--table", f"{name}={table.format(name)}")]
    return seed.main(["--stage", "dev", *tables, *extra, "--apply"])


def test_seed_uses_the_explicit_region(seed_region: list[str | None]) -> None:
    arn = "arn:aws:dynamodb:us-west-2:111122223333:table/{}"
    assert _seed("--region", "eu-west-1", table=arn) == 0
    assert seed_region == ["eu-west-1"]


def test_seed_uses_the_table_arns_region(seed_region: list[str | None],
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    assert _seed(table="arn:aws:dynamodb:us-west-2:111122223333:table/{}") == 0
    assert seed_region == ["us-west-2"]


def test_seed_reads_the_environment(seed_region: list[str | None], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "eu-west-2")
    assert _seed() == 0
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    assert _seed() == 0
    assert seed_region == ["eu-west-2", "eu-west-1"]


def test_seed_falls_back_to_us_east_1_and_logs_it(seed_region: list[str | None],
                                                  caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    assert _seed() == 0
    assert seed_region == ["us-east-1"]
    assert any("using us-east-1" in r.getMessage() for r in caplog.records)


def test_twilio_setup_resolves_its_region() -> None:
    """twilio_setup.py reads its secrets from Secrets Manager, so it resolves a Region."""
    src = (Path(__file__).resolve().parents[3] / "scripts" / "twilio_setup.py").read_text(encoding="utf-8")
    assert "resolve_region(" in src
