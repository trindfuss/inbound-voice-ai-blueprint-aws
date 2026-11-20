"""Unit tests for owlbank.tokens, on MemoryStore and DynamoStore (moto)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import pytest

from owlbank import tokens
from owlbank.config import Settings
from owlbank.store import SCHEMAS, DynamoStore, MemoryStore, Store, create_table_kwargs
from owlbank.tokens import AuthError, TokenClaims
from owlbank.tokens import AuthErrorCode as C

NOW = 1_700_000_000
TTL = 1800


@pytest.fixture(params=["memory", "dynamo"])
def store(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[Store]:
    if request.param == "memory":
        yield MemoryStore()
        return
    for var, val in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
                     "AWS_SESSION_TOKEN": "testing", "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(var, val)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    import boto3
    from moto import mock_aws

    settings = Settings(aws_region="us-east-1")
    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-east-1")
        for logical in SCHEMAS:
            client.create_table(**create_table_kwargs(logical, settings.table(logical)))
        yield DynamoStore.from_settings(settings)


def issue(store: Store, call_sid: str = "CA1", level: int = 1, now: int = NOW) -> str:
    return tokens.issue(store, call_sid=call_sid, customer_id="cust-1", level=level, now=now,
                        max_ttl_s=TTL)


def check(store: Store, token: str | None, call_sid: str = "CA1", min_level: int = 1,
          now: int = NOW + 1) -> TokenClaims | AuthError:
    return tokens.validate(store, token, call_sid=call_sid, min_level=min_level, now=now)


def test_hash_is_sha256_hex() -> None:
    assert tokens.hash_token("abc") == hashlib.sha256(b"abc").hexdigest()


def test_issue_stores_hash_only(store: Store) -> None:
    token = issue(store, level=2)
    assert len(token) >= 43
    item = store.get("tokens", {"token_hash": tokens.hash_token(token)})
    assert item == {
        "token_hash": tokens.hash_token(token), "call_sid": "CA1", "customer_id": "cust-1",
        "level": 2, "issued_at": NOW, "expires_at": NOW + TTL, "revoked": False,
    }
    assert token not in repr(item)


def test_valid_token_returns_claims(store: Store) -> None:
    token = issue(store, level=2)
    assert check(store, token, min_level=2) == TokenClaims("CA1", "cust-1", 2, NOW, NOW + TTL)


@pytest.mark.parametrize("value", [None, ""])
def test_missing(store: Store, value: str | None) -> None:
    assert check(store, value) == AuthError(C.MISSING)


def test_unknown(store: Store) -> None:
    issue(store)
    assert check(store, "made-up-token") == AuthError(C.UNKNOWN)


def test_wrong_call(store: Store) -> None:
    token = issue(store)
    assert check(store, token, call_sid="CA2") == AuthError(C.WRONG_CALL)
    assert check(store, token, call_sid="") == AuthError(C.WRONG_CALL)


def test_expiry_boundary(store: Store) -> None:
    token = issue(store)
    assert isinstance(check(store, token, now=NOW + TTL - 1), TokenClaims)
    assert check(store, token, now=NOW + TTL) == AuthError(C.EXPIRED)


def test_level_insufficient(store: Store) -> None:
    token = issue(store, level=1)
    assert check(store, token, min_level=2) == AuthError(C.AUTH_LEVEL_INSUFFICIENT)
    assert isinstance(check(store, token, min_level=0), TokenClaims)


def test_revoke_call_revokes_all_and_only_that_call(store: Store) -> None:
    a1, a2, b = issue(store), issue(store, level=2), issue(store, call_sid="CA2")
    tokens.revoke_call(store, "CA1")
    tokens.revoke_call(store, "CA1")  # idempotent
    assert check(store, a1) == AuthError(C.REVOKED)
    assert check(store, a2) == AuthError(C.REVOKED)
    assert isinstance(check(store, b, call_sid="CA2"), TokenClaims)
    tokens.revoke_call(store, "CA-none")
    tokens.revoke_call(store, "")


def test_error_precedence(store: Store) -> None:
    token = issue(store, level=0)
    tokens.revoke_call(store, "CA1")
    # revoked + expired + insufficient, presented on another call -> WRONG_CALL
    assert check(store, token, call_sid="CA2", min_level=3, now=NOW + TTL) == AuthError(
        C.WRONG_CALL
    )
    assert check(store, token, min_level=3, now=NOW + TTL) == AuthError(C.REVOKED)


def test_upgrade_raises_level_and_revokes_old(store: Store) -> None:
    old = issue(store, level=1)
    new = tokens.upgrade(store, old, call_sid="CA1", new_level=3, now=NOW + 10, max_ttl_s=TTL)
    assert isinstance(new, str) and new != old
    assert check(store, old) == AuthError(C.REVOKED)
    claims = check(store, new, min_level=3, now=NOW + 11)
    assert claims == TokenClaims("CA1", "cust-1", 3, NOW + 10, NOW + 10 + TTL)


def test_upgrade_same_level_allowed(store: Store) -> None:
    old = issue(store, level=2)
    new = tokens.upgrade(store, old, call_sid="CA1", new_level=2, now=NOW, max_ttl_s=TTL)
    assert isinstance(new, str)
    assert isinstance(check(store, new, min_level=2), TokenClaims)


def test_upgrade_never_decreases(store: Store) -> None:
    old = issue(store, level=2)
    with pytest.raises(ValueError):
        tokens.upgrade(store, old, call_sid="CA1", new_level=1, now=NOW, max_ttl_s=TTL)
    assert isinstance(check(store, old, min_level=2), TokenClaims)  # old untouched


def test_upgrade_requires_valid_token_for_this_call(store: Store) -> None:
    old = issue(store)

    def up(t: str | None, sid: str = "CA1", now: int = NOW) -> str | AuthError:
        return tokens.upgrade(store, t, call_sid=sid, new_level=2, now=now, max_ttl_s=TTL)

    assert up(None) == AuthError(C.MISSING)
    assert up("bogus") == AuthError(C.UNKNOWN)
    assert up(old, sid="CA2") == AuthError(C.WRONG_CALL)
    assert up(old, now=NOW + TTL) == AuthError(C.EXPIRED)
    tokens.revoke_call(store, "CA1")
    assert up(old) == AuthError(C.REVOKED)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"call_sid": ""}, {"customer_id": ""}, {"level": -1}, {"level": True},
        {"now": -1}, {"max_ttl_s": 0}, {"level": 1.0},
    ],
)
def test_issue_rejects_bad_input(kwargs: dict) -> None:
    args = {"call_sid": "CA1", "customer_id": "c", "level": 1, "now": NOW, "max_ttl_s": TTL}
    with pytest.raises(ValueError):
        tokens.issue(MemoryStore(), **{**args, **kwargs})


def test_corrupt_record_fails_closed() -> None:
    store = MemoryStore()
    token = issue(store)
    key = {"token_hash": tokens.hash_token(token)}
    store.update("tokens", key, {"revoked": None})
    assert check(store, token) == AuthError(C.REVOKED)
    store.update("tokens", key, {"revoked": False, "expires_at": "never"})
    assert check(store, token) == AuthError(C.EXPIRED)
    store.update("tokens", key, {"expires_at": NOW + TTL, "level": "9"})
    assert check(store, token, min_level=0) == AuthError(C.AUTH_LEVEL_INSUFFICIENT)


def test_ttl_deleted_token_is_unknown(store: Store) -> None:
    token = issue(store)
    store.delete("tokens", {"token_hash": tokens.hash_token(token)})
    assert check(store, token) == AuthError(C.UNKNOWN)
