"""The start-up audit entry for ``acceptSpokenFactors``, with an in-memory store. No network."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from lambdas.admin_api import deploy_settings as ds
from lambdas.admin_api import handler as handler_mod
from lambdas.admin_api.app import Deps
from lambdas.admin_api.audit import AuditClock
from owlbank.store import ConditionFailed, Item, MemoryStore, TxPut
from tests.admin_api_fakes import POOL_ID, Clock, FakeCognito, call, claims, seeded_store

VIEWER = claims(["admin-viewer"])
EDITOR = claims(["admin-editor"])


#: One clock for every start in this module, so entries' sort keys never collide (as in real time).
_CLOCK = Clock()


def clock() -> AuditClock:
    return AuditClock(now=_CLOCK, suffix=_CLOCK.suffix)


def entries(store: MemoryStore) -> list[dict[str, Any]]:
    return [e for e in store.query("admin_audit", {"pk": ds.DEPLOY_PK}) if e["sk"] != ds.MARKER_SK]


def marker(store: MemoryStore) -> dict[str, Any] | None:
    return store.get("admin_audit", ds.marker_key())


def record(store: Any, value: bool, c: AuditClock | None = None) -> str:
    return ds.record_accept_spoken_factors(store, value, clock=c or clock(), retention_days=90)


class RacingStore:
    """A store where, right after this start reads the marker, ``other`` runs to completion, so
    both cold starts read the same state (two Lambdas starting together)."""

    def __init__(self, inner: MemoryStore, other: Callable[[], object]) -> None:
        self.inner, self.other, self.fired = inner, other, False

    def get(self, table: str, key: Mapping[str, Any]) -> Item | None:
        got = self.inner.get(table, key)
        if not self.fired:
            self.fired = True
            self.other()
        return got

    def transact_write(self, puts: Sequence[TxPut]) -> None:
        self.inner.transact_write(puts)


# --------------------------------------------------------------------------- the entry


def test_first_start_true_writes_marker_and_one_security_entry() -> None:
    store = MemoryStore()
    assert record(store, True) == "recorded"
    [e] = entries(store)
    assert e["action"] == "deploy_setting"
    assert e["actor"] == {"sub": "deploy"}
    assert e["target"] == {"kind": "deploy_setting", "id": "acceptSpokenFactors"}
    assert e["changes"] == [{"field": "acceptSpokenFactors", "old": None, "new": True}]
    assert e["security_change"] is True
    assert isinstance(e["expires_at"], int)  # entries expire like the others
    m = marker(store)
    assert m == {"pk": ds.DEPLOY_PK, "sk": ds.MARKER_SK, "value": True, "rev": 1}
    assert "expires_at" not in m  # the marker has no TTL


def test_first_start_false_is_not_a_security_change() -> None:
    store = MemoryStore()
    assert record(store, False) == "recorded"
    [e] = entries(store)
    assert e["changes"] == [{"field": "acceptSpokenFactors", "old": None, "new": False}]
    assert e["security_change"] is False
    assert marker(store)["value"] is False  # type: ignore[index]


def test_no_change_writes_nothing() -> None:
    store = MemoryStore()
    record(store, True)
    assert record(store, True) == "unchanged"
    assert len(entries(store)) == 1
    assert marker(store)["rev"] == 1  # type: ignore[index]


def test_change_writes_one_entry_each_way() -> None:
    store = MemoryStore()
    record(store, True)
    assert record(store, False) == "recorded"
    off = entries(store)[-1]
    assert off["changes"] == [{"field": "acceptSpokenFactors", "old": True, "new": False}]
    assert off["security_change"] is False
    assert record(store, True) == "recorded"
    on = entries(store)[-1]
    assert on["changes"] == [{"field": "acceptSpokenFactors", "old": False, "new": True}]
    assert on["security_change"] is True
    assert len(entries(store)) == 3
    assert marker(store) == {"pk": ds.DEPLOY_PK, "sk": ds.MARKER_SK, "value": True, "rev": 3}


def test_malformed_marker_is_recorded_over_with_before_null() -> None:
    store = MemoryStore()
    store.put("admin_audit", {**ds.marker_key(), "value": "yes", "rev": 4})
    assert record(store, False) == "recorded"
    [e] = entries(store)
    assert e["changes"][0]["old"] is None and e["security_change"] is False
    assert marker(store)["rev"] == 5  # type: ignore[index]


@pytest.mark.parametrize("first, second", [(True, True), (False, False), (True, False)])
def test_two_concurrent_cold_starts_write_one_entry(first: bool, second: bool) -> None:
    inner = MemoryStore()
    outcomes: list[str] = []
    racing = RacingStore(inner, lambda: outcomes.append(record(inner, second)))
    outcomes.append(record(racing, first))
    assert sorted(outcomes) == ["lost_race", "recorded"]
    [e] = entries(inner)
    assert marker(inner)["value"] is second  # type: ignore[index]  # the one that wrote first
    assert e["changes"][0]["new"] is second


def test_concurrent_starts_on_a_recorded_marker() -> None:
    inner = MemoryStore()
    record(inner, False)
    outcomes: list[str] = []
    racing = RacingStore(inner, lambda: outcomes.append(record(inner, True)))
    outcomes.append(record(racing, True))
    assert sorted(outcomes) == ["lost_race", "recorded"]
    assert len(entries(inner)) == 2
    assert marker(inner)["value"] is True  # type: ignore[index]


class BrokenStore:
    def get(self, table: str, key: Mapping[str, Any]) -> Item | None:
        raise RuntimeError("secret detail 1234")

    def transact_write(self, puts: Sequence[TxPut]) -> None:  # pragma: no cover - never reached
        raise AssertionError


class FailingWrite(MemoryStore):
    def transact_write(self, puts: Sequence[TxPut]) -> None:
        raise ConditionFailed("not a transaction cancel: secret detail 1234")


@pytest.mark.parametrize("store", [BrokenStore(), FailingWrite()])
def test_errors_are_logged_and_never_raise(store: Any, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger=ds.__name__):
        assert record(store, True) == "failed"
    [rec] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert rec.__dict__["decision"] == "continue"
    assert rec.__dict__["setting"] == "acceptSpokenFactors"
    assert "secret detail" not in caplog.text


def test_logs_carry_names_and_booleans_only(caplog: pytest.LogCaptureFixture) -> None:
    store = MemoryStore()
    with caplog.at_level(logging.DEBUG, logger=ds.__name__):
        record(store, True)
        record(store, True)
        record(store, False)
    base = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
    allowed = {"setting", "before", "after", "security_change", "outcome", "event"}
    seen = 0
    for rec in caplog.records:
        fields = {k: v for k, v in rec.__dict__.items() if k not in base}
        assert set(fields) <= allowed, fields
        assert all(isinstance(fields[k], str) for k in ("setting", "event", "outcome"))
        assert all(v is None or isinstance(v, bool) for k, v in fields.items()
                   if k in ("before", "after", "security_change"))
        seen += 1
    assert seen == 3


# --------------------------------------------------------------------------- start-up and the API


def deps(store: MemoryStore, accept: bool) -> Deps:
    return Deps(store=store, cognito=FakeCognito(), user_pool_id=POOL_ID, retention_days=90,
                clock=_CLOCK, suffix=_CLOCK.suffix, accept_spoken_factors=accept)


def test_start_api_records_once_then_serves_the_setting() -> None:
    store = seeded_store()
    api = handler_mod.start_api(deps(store, False))
    assert len(entries(store)) == 1
    status, body = call(api, "GET", "/slots", who=VIEWER)
    assert status == 200 and body["acceptSpokenFactors"] is False
    handler_mod.start_api(deps(store, False))  # another cold start, same deploy
    assert len(entries(store)) == 1


def test_start_api_survives_a_failing_store() -> None:
    store = FailingWrite()
    api = handler_mod.start_api(deps(store, True))
    assert api is not None


def test_entry_is_listed_by_get_audit_without_the_marker() -> None:
    store = seeded_store()
    handler_mod.start_api(deps(store, True))
    handler_mod.start_api(deps(store, False))
    api = handler_mod.start_api(deps(store, False))
    status, body = call(api, "GET", "/audit", who=VIEWER, query={"item": "deploy#accept_spoken_factors"})
    assert status == 200, body
    assert body["item"] == ds.DEPLOY_PK
    got = body["entries"]
    assert [e["changes"][0]["new"] for e in got] == [False, True]  # newest first
    assert all(e["action"] == "deploy_setting" and e["sk"] != ds.MARKER_SK for e in got)
    assert ds.MARKER_SK not in json.dumps([e["sk"] for e in got])


def test_deploy_entries_are_not_restorable() -> None:
    store = seeded_store()
    api = handler_mod.start_api(deps(store, True))
    [e] = entries(store)
    for sk in (e["sk"], ds.MARKER_SK):
        status, body = call(api, "POST", "/restore", who=EDITOR,
                            body={"audit_pk": ds.DEPLOY_PK, "audit_sk": sk, "rev": 0})
        assert status == 400 and body["error"] == "not_restorable", body
    assert len(entries(store)) == 1


def test_build_api_passes_the_env_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """``build_api`` hands ``OWLBANK_ACCEPT_SPOKEN_FACTORS`` to ``start_api`` (boto3 stubbed)."""
    import sys
    import types

    seen: list[Deps] = []
    fake_boto3 = types.SimpleNamespace(client=lambda *a, **k: FakeCognito())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    import owlbank.store as store_mod
    monkeypatch.setattr(store_mod, "DynamoStore", lambda *a, **k: MemoryStore())
    monkeypatch.setattr(handler_mod, "start_api", lambda d: seen.append(d) or "api")
    env = {"OWLBANK_ADMIN_USER_POOL_ID": POOL_ID, "OWLBANK_ACCEPT_SPOKEN_FACTORS": "false"}
    assert handler_mod.build_api(env) == "api"
    assert seen[0].accept_spoken_factors is False
    assert handler_mod.build_api({"OWLBANK_ADMIN_USER_POOL_ID": POOL_ID}) == "api"
    assert seen[1].accept_spoken_factors is True
