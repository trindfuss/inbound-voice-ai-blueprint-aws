"""Unit tests for owlbank.secrets, against moto (fake credentials, no real AWS)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest

from owlbank import secrets as sm

VALUE = "super-secret-auth-token-9f8e7d"


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    sm.clear_cache()
    yield
    sm.clear_cache()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    for var, val in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
    }.items():
        monkeypatch.setenv(var, val)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    import boto3
    from moto import mock_aws

    with mock_aws():
        yield boto3.client("secretsmanager", region_name="us-east-1")


def make(client: Any, name: str, secret_string: str) -> str:
    return client.create_secret(Name=name, SecretString=secret_string)["ARN"]


class Counting:
    """Wraps a client and counts reads."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.calls = inner, 0

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.inner, name)
        if not callable(attr):
            return attr

        def wrapped(*a: Any, **kw: Any) -> Any:
            self.calls += 1
            return attr(*a, **kw)

        return wrapped


def test_reads_and_parses_json(client: Any) -> None:
    arn = make(client, "twilio", json.dumps({"auth_token": VALUE}))
    assert sm.get_secret(arn, client=client) == {"auth_token": VALUE}


def test_cached_within_ttl_then_refreshed(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    arn = make(client, "twilio", json.dumps({"auth_token": "v1"}))
    now = [1000.0]
    monkeypatch.setattr(sm, "clock", lambda: now[0])
    counting = Counting(client)
    assert sm.get_secret(arn, client=counting, ttl_s=300)["auth_token"] == "v1"
    client.put_secret_value(SecretId=arn, SecretString=json.dumps({"auth_token": "v2"}))
    now[0] += 299
    assert sm.get_secret(arn, client=counting, ttl_s=300)["auth_token"] == "v1"
    assert counting.calls == 1
    now[0] += 1
    assert sm.get_secret(arn, client=counting, ttl_s=300)["auth_token"] == "v2"
    assert counting.calls == 2


def test_returned_dict_is_a_copy(client: Any) -> None:
    arn = make(client, "basic", json.dumps({"user": "u", "password": "p"}))
    sm.get_secret(arn, client=client)["user"] = "mutated"
    assert sm.get_secret(arn, client=client)["user"] == "u"


def test_clear_cache_forces_reread(client: Any) -> None:
    arn = make(client, "x", json.dumps({"k": "v"}))
    counting = Counting(client)
    sm.get_secret(arn, client=counting)
    sm.clear_cache()
    sm.get_secret(arn, client=counting)
    assert counting.calls == 2


@pytest.mark.parametrize("raw", [VALUE, json.dumps([VALUE]), json.dumps(VALUE), "{" + VALUE])
def test_non_object_value_raises_without_leaking(
    client: Any, raw: str, caplog: pytest.LogCaptureFixture
) -> None:
    arn = make(client, "bad", raw)
    caplog.set_level(logging.DEBUG, logger="owlbank")
    with pytest.raises(sm.SecretError) as info:
        sm.get_secret(arn, client=client)
    assert VALUE not in str(info.value)
    assert info.value.__cause__ is None
    assert VALUE not in repr(info.value.__context__)
    assert VALUE not in caplog.text


def test_missing_secret_raises(client: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    with pytest.raises(sm.SecretError, match="ResourceNotFoundException"):
        sm.get_secret("arn:aws:secretsmanager:us-east-1:123456789012:secret:nope", client=client)


def test_empty_arn_raises() -> None:
    with pytest.raises(sm.SecretError):
        sm.get_secret("")


def test_value_never_logged(client: Any, caplog: pytest.LogCaptureFixture) -> None:
    """Our logs at DEBUG, and every logger at INFO (the Lambda default), never carry the value.

    botocore's own DEBUG logging prints response bodies, secret values included;
    it must never be enabled in a deployed function.
    """
    arn = make(client, "twilio", json.dumps({"auth_token": VALUE}))
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.DEBUG, logger="owlbank")
    sm.get_secret(arn, client=client)  # miss
    sm.get_secret(arn, client=client)  # hit
    assert caplog.records, "expected debug logging of hit/miss"
    assert VALUE not in caplog.text
    for record in caplog.records:
        assert VALUE not in repr(record.__dict__)


# --------------------------------------------------------------------------- logical=

TOKEN = "0123456789abcdef" * 2  # a well-formed auth token (fictional)
PLACEHOLDER_UNSET = "Rk2mQ9xZpL7vT4nW8yB3cF6hJ1dS5gA0"


def test_logical_valid_value_is_returned(client: Any) -> None:
    arn = make(client, "twilio", json.dumps({"auth_token": TOKEN}))
    assert sm.get_secret(arn, client=client, logical="twilio_auth_token") == {"auth_token": TOKEN}


def test_logical_placeholder_fails_closed_naming_secret_and_field(
    client: Any, caplog: pytest.LogCaptureFixture
) -> None:
    arn = make(client, "twilio", json.dumps({"placeholder": True, "unset": PLACEHOLDER_UNSET}))
    caplog.set_level(logging.INFO)  # botocore's DEBUG prints response bodies (module docstring)
    caplog.set_level(logging.DEBUG, logger="owlbank")
    with pytest.raises(sm.SecretError) as info:
        sm.get_secret(arn, client=client, logical="twilio_auth_token")
    msg = str(info.value)
    assert "twilio_auth_token" in msg and "placeholder" in msg and arn in msg
    assert info.value.problem is not None and info.value.problem.kind == "placeholder"
    assert info.value.__cause__ is None
    assert PLACEHOLDER_UNSET not in msg and PLACEHOLDER_UNSET not in caplog.text
    [rec] = [r for r in caplog.records if getattr(r, "event", None) == "secret_invalid"]
    assert rec.levelno == logging.WARNING
    assert (rec.secret, rec.field, rec.problem, rec.secret_arn, rec.decision) == (  # type: ignore[attr-defined]
        "twilio_auth_token", "placeholder", "placeholder", arn, "raise_secret_error")
    for record in caplog.records:
        assert PLACEHOLDER_UNSET not in repr(record.__dict__)


@pytest.mark.parametrize(("doc", "field", "kind"), [
    ({"user": "owlbank"}, "password", "missing"),
    ({"user": "owlbank", "password": VALUE}, "password", "format"),  # not lower-case hex
    ({"user": "owl bank", "password": "ab" * 24}, "user", "format"),
])
def test_logical_malformed_value_names_the_field_never_the_value(
    client: Any, caplog: pytest.LogCaptureFixture, doc: dict[str, str], field: str, kind: str
) -> None:
    arn = make(client, "basic", json.dumps(doc))
    caplog.set_level(logging.INFO)  # botocore's DEBUG prints response bodies (module docstring)
    caplog.set_level(logging.DEBUG, logger="owlbank")
    with pytest.raises(sm.SecretError) as info:
        sm.get_secret(arn, client=client, logical="basic_auth")
    assert f"basic_auth: field {field}: {kind}" in str(info.value)
    for v in doc.values():
        if len(v) >= 4 and v != "owlbank":
            assert v not in str(info.value) and v not in caplog.text


def test_logical_invalid_value_is_not_cached(client: Any) -> None:
    """A corrected value is picked up on the next read, not after the TTL."""
    arn = make(client, "twilio", json.dumps({"placeholder": True, "unset": PLACEHOLDER_UNSET}))
    counting = Counting(client)
    with pytest.raises(sm.SecretError):
        sm.get_secret(arn, client=counting, logical="twilio_auth_token")
    client.put_secret_value(SecretId=arn, SecretString=json.dumps({"auth_token": TOKEN}))
    assert sm.get_secret(arn, client=counting, logical="twilio_auth_token")["auth_token"] == TOKEN
    assert counting.calls == 2
    sm.get_secret(arn, client=counting, logical="twilio_auth_token")
    assert counting.calls == 2  # the valid value is cached


def test_logical_checks_a_value_cached_by_an_unchecked_read(client: Any) -> None:
    arn = make(client, "twilio", json.dumps({"placeholder": True, "unset": PLACEHOLDER_UNSET}))
    counting = Counting(client)
    assert sm.get_secret(arn, client=counting)["placeholder"] is True  # no logical: no check, cached
    with pytest.raises(sm.SecretError, match="placeholder"):
        sm.get_secret(arn, client=counting, logical="twilio_auth_token")
    assert counting.calls == 1


def test_logical_empty_arn_names_the_secret() -> None:
    with pytest.raises(sm.SecretError, match="^secret basic_auth ARN is not configured$"):
        sm.get_secret("", logical="basic_auth")


def test_logical_non_object_still_raises_without_leaking(client: Any) -> None:
    arn = make(client, "bad", json.dumps([VALUE]))
    with pytest.raises(sm.SecretError, match="not a JSON object") as info:
        sm.get_secret(arn, client=client, logical="twilio_auth_token")
    assert VALUE not in str(info.value)


# --- per-secret TTL, forced re-read (rotation) ---

import copy as _copy  # noqa: E402

BASIC_V1 = {"user": "owlbank", "password": "a" * 32}
BASIC_V2 = {"user": "owlbank", "password": "b" * 32, "previous_password": "a" * 32,
            "previous_until": 1_900_000_000}


class Scripted:
    """A fake Secrets Manager client: serves ``values`` in turn (the last repeats); ``fail`` raises."""

    def __init__(self, *values: dict[str, Any]) -> None:
        self.values = list(values)
        self.calls = 0
        self.fail = False

    def get_secret_value(self, SecretId: str) -> dict[str, Any]:  # noqa: N803
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        v = self.values[min(self.calls - 1, len(self.values) - 1)]
        return {"SecretString": json.dumps(_copy.deepcopy(v))}


@pytest.fixture
def now(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    t = [1000.0]
    monkeypatch.setattr(sm, "clock", lambda: t[0])
    return t


def test_basic_secret_ttl_is_60_seconds_others_300(now: list[float]) -> None:
    basic, other = Scripted(BASIC_V1), Scripted({"k": "v"})
    sm.get_secret("arn:b", client=basic, logical="basic_auth")
    sm.get_secret("arn:o", client=other)
    now[0] += 59
    sm.get_secret("arn:b", client=basic, logical="basic_auth")
    sm.get_secret("arn:o", client=other)
    assert (basic.calls, other.calls) == (1, 1)
    now[0] += 2  # 61 s: Basic expired, the default 300 s not
    sm.get_secret("arn:b", client=basic, logical="basic_auth")
    sm.get_secret("arn:o", client=other)
    assert (basic.calls, other.calls) == (2, 1)
    now[0] += 250
    sm.get_secret("arn:o", client=other)
    assert other.calls == 2


def test_explicit_ttl_wins_over_the_default(now: list[float]) -> None:
    c = Scripted(BASIC_V1)
    sm.get_secret("arn:b", client=c, logical="basic_auth", ttl_s=1000)
    now[0] += 500
    sm.get_secret("arn:b", client=c, logical="basic_auth", ttl_s=1000)
    assert c.calls == 1


def test_force_rereads_and_updates_the_cache(now: list[float]) -> None:
    c = Scripted(BASIC_V1, BASIC_V2)
    assert sm.get_secret("arn:b", client=c, logical="basic_auth") == BASIC_V1
    assert sm.get_secret("arn:b", client=c, logical="basic_auth", force=True) == BASIC_V2
    assert c.calls == 2
    assert sm.get_secret("arn:b", client=c, logical="basic_auth") == BASIC_V2  # from cache
    assert c.calls == 2


def test_forced_reads_are_limited_to_one_per_arn_per_10_seconds(now: list[float]) -> None:
    c = Scripted(BASIC_V1, BASIC_V2)
    sm.get_secret("arn:b", client=c, logical="basic_auth")
    sm.get_secret("arn:b", client=c, logical="basic_auth", force=True)
    for _ in range(5):
        now[0] += 1.5
        assert sm.get_secret("arn:b", client=c, logical="basic_auth", force=True) == BASIC_V2
    assert c.calls == 2  # all rate limited, served from cache
    other = Scripted(BASIC_V1)
    sm.get_secret("arn:x", client=other, logical="basic_auth")
    sm.get_secret("arn:x", client=other, logical="basic_auth", force=True)
    assert other.calls == 2  # another ARN has its own allowance
    now[0] += 10  # 10 s after the first forced read (+7.5 already): allowed again
    sm.get_secret("arn:b", client=c, logical="basic_auth", force=True)
    assert c.calls == 3


def test_a_failed_forced_read_uses_the_cached_value(now: list[float],
                                                   caplog: pytest.LogCaptureFixture) -> None:
    c = Scripted(BASIC_V1)
    sm.get_secret("arn:b", client=c, logical="basic_auth")
    c.fail = True
    with caplog.at_level(logging.DEBUG):
        got = sm.get_secret("arn:b", client=c, logical="basic_auth", force=True)
    assert got == BASIC_V1
    assert "a" * 32 not in caplog.text


def test_a_failed_forced_read_with_an_expired_cache_still_uses_it(now: list[float]) -> None:
    c = Scripted(BASIC_V1)
    sm.get_secret("arn:b", client=c, logical="basic_auth")
    now[0] += 500
    c.fail = True
    assert sm.get_secret("arn:b", client=c, logical="basic_auth", force=True) == BASIC_V1


def test_a_failed_forced_read_with_nothing_cached_raises(now: list[float]) -> None:
    c = Scripted(BASIC_V1)
    c.fail = True
    with pytest.raises(sm.SecretError):
        sm.get_secret("arn:b", client=c, logical="basic_auth", force=True)


def test_a_malformed_previous_pair_does_not_make_the_secret_unusable(now: list[float]) -> None:
    lone = {**BASIC_V1, "previous_password": "a" * 32}
    got = sm.get_secret("arn:b", client=Scripted(lone), logical="basic_auth")
    assert got == lone  # readers ignore the pair (previous_pair returns None); no SecretError


def test_clear_cache_resets_the_force_allowance(now: list[float]) -> None:
    c = Scripted(BASIC_V1, BASIC_V2)
    sm.get_secret("arn:b", client=c, logical="basic_auth", force=True)
    sm.clear_cache()
    sm.get_secret("arn:b", client=c, logical="basic_auth", force=True)
    assert c.calls == 2


@pytest.mark.parametrize("bad", [{"user": "owlbank"}, "not json {", ["a list"], {"user": "owlbank", "password": "short"}],
                         ids=["schema_invalid", "unparsable", "not_object", "bad_format"])
def test_a_forced_read_with_unusable_content_falls_back_to_the_cached_value(
        now: list[float], bad: Any, caplog: pytest.LogCaptureFixture) -> None:
    class Bad:
        def get_secret_value(self, SecretId: str) -> dict[str, Any]:  # noqa: N803
            return {"SecretString": bad if isinstance(bad, str) else json.dumps(bad)}

    good = Scripted(BASIC_V1)
    assert sm.get_secret("arn:b", client=good, logical="basic_auth") == BASIC_V1
    caplog.set_level(logging.DEBUG)
    assert sm.get_secret("arn:b", client=Bad(), logical="basic_auth", force=True) == BASIC_V1
    assert "short" not in caplog.text and "not json" not in caplog.text


def test_a_forced_read_with_unusable_content_and_nothing_cached_raises(now: list[float]) -> None:
    class Bad:
        def get_secret_value(self, SecretId: str) -> dict[str, Any]:  # noqa: N803
            return {"SecretString": "not json"}

    with pytest.raises(sm.SecretError):
        sm.get_secret("arn:b", client=Bad(), logical="basic_auth", force=True)
