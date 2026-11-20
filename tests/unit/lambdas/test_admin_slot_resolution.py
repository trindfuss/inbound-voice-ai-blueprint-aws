"""Admin API ``PUT /slots/resolution`` and ``POST /slots/preview``, with an in-memory store. No
network."""

from __future__ import annotations

import copy
import logging
from typing import Any

import pytest

from lambdas.admin_api import audit
from lambdas.admin_api.app import ADMIN, USER_GROUPS, AdminApi, Deps
from owlbank.policy import Policy
from owlbank.store import MemoryStore
from tests.admin_api_fakes import POOL_ID, Clock, FakeCognito, call, claims, seeded_store

VIEWER = claims(["admin-viewer"])
EDITOR = claims(["admin-editor"], sub="sub-editor", email="editor@example.com")
ADMIN_USER = claims(["admin"], sub="sub-admin", email="admin@example.com")


def make(store: MemoryStore | None = None, *, ceiling: str = "assisted") -> tuple[AdminApi, MemoryStore]:
    store = store if store is not None else seeded_store()
    clock = Clock()
    api = AdminApi(Deps(store=store, cognito=FakeCognito(), user_pool_id=POOL_ID, retention_days=90,
                        clock=clock, suffix=clock.suffix,
                        slot_mode_ceiling=ceiling))  # type: ignore[arg-type]
    return api, store


def res_item(store: MemoryStore) -> dict[str, Any] | None:
    return store.get("policy", {"pk": "slot_resolution"})


def res_audits(store: MemoryStore) -> list[dict[str, Any]]:
    return sorted(store.query("admin_audit", {"pk": "policy#slot_resolution"}), key=lambda e: e["sk"])


def all_audits(store: MemoryStore) -> list[dict[str, Any]]:
    return list(store._tables["admin_audit"].values())


def codes(warnings: list[dict[str, Any]]) -> list[str]:
    return [w["code"] for w in warnings]


def test_admin_group_is_the_user_groups_set() -> None:
    assert ADMIN is USER_GROUPS and frozenset({"admin"}) == ADMIN


# --------------------------------------------------------------------------- PUT /slots/resolution


def test_put_resolution_writes_item_and_one_audit_entry() -> None:
    api, store = make()
    status, body = call(api, "PUT", "/slots/resolution", {"rev": 0, "resolver_budget_ms": 2000}, who=EDITOR)
    assert status == 200, body
    assert body["changed"] is True and body["security_change"] is False and body["warnings"] == []
    assert body["slot_resolution"]["resolver_budget_ms"] == 2000 and body["slot_resolution"]["rev"] == 1
    item = res_item(store)
    assert item is not None and item["resolver_budget_ms"] == 2000 and item["rev"] == 1
    (entry,) = res_audits(store)
    assert entry["target"] == {"kind": "slot_resolution", "id": "slot_resolution"}
    assert entry["changes"] == [{"field": "resolver_budget_ms", "old": 2500, "new": 2000}]
    assert entry["security_change"] is False and entry["rev"] == 1
    assert {"pk": entry["pk"], "sk": entry["sk"]} == body["audit"]
    assert Policy.from_store(store).slot_resolution.resolver_budget_ms == 2000


def test_put_default_mode_is_a_security_change() -> None:
    api, store = make()
    status, body = call(api, "PUT", "/slots/resolution", {"rev": 0, "default_mode": "deterministic"},
                        who=EDITOR)
    assert status == 200 and body["security_change"] is True and body["warnings"] == [], body
    status, body = call(api, "PUT", "/slots/resolution", {"rev": 1, "default_mode": "assisted"}, who=EDITOR)
    assert status == 200 and body["security_change"] is True
    assert codes(body["warnings"]) == ["away_from_deterministic"]
    assert [e["security_change"] for e in res_audits(store)] == [True, True]


def test_put_flows_is_a_security_change_and_warns() -> None:
    api, store = make()
    flows = {"login": {"mode": "deterministic"}, "stepup": {"mode": "deterministic"}}
    status, body = call(api, "PUT", "/slots/resolution", {"rev": 0, "flows": flows}, who=EDITOR)
    assert status == 200 and body["security_change"] is True and body["warnings"] == [], body
    assert body["slot_resolution"]["flows"] == {"login": "deterministic", "stepup": "deterministic"}
    status, body = call(api, "PUT", "/slots/resolution", {"rev": 1, "flows": {"login": {"mode": "deterministic"}}},
                        who=EDITOR)
    assert status == 200 and codes(body["warnings"]) == ["away_from_deterministic"]  # stepup now assisted
    assert Policy.from_store(store).slot_resolution.flows == {"login": "deterministic"}


def test_put_resolution_no_change_writes_nothing() -> None:
    api, store = make()
    status, body = call(api, "PUT", "/slots/resolution", {"rev": 0, "default_mode": "assisted"}, who=EDITOR)
    assert status == 200 and body["changed"] is False and not all_audits(store)
    assert res_item(store).get("rev", 0) == 0  # type: ignore[union-attr]


@pytest.mark.parametrize("body,field", [
    ({"rev": 0, "default_mode": "fancy"}, "slot_resolution.default_mode"),
    ({"rev": 0, "default_mode": None}, "slot_resolution.default_mode"),
    ({"rev": 0, "flows": {"billing": {"mode": "assisted"}}}, "slot_resolution.flows"),
    ({"rev": 0, "flows": {"login": "assisted"}}, "slot_resolution.flows.login.mode"),
    ({"rev": 0, "flows": {"login": {"mode": "x"}}}, "slot_resolution.flows.login.mode"),
    ({"rev": 0, "flows": []}, "slot_resolution.flows"),
    ({"rev": 0, "resolver_budget_ms": 100}, "slot_resolution.resolver_budget_ms"),
    ({"rev": 0, "resolver_budget_ms": 4001}, "slot_resolution.resolver_budget_ms"),
    ({"rev": 0, "resolver_budget_ms": "1500"}, "slot_resolution.resolver_budget_ms"),
    ({"rev": 0, "resolver_budget_ms": True}, "slot_resolution.resolver_budget_ms"),
    ({"rev": 0, "bogus": 1}, "bogus"),
    ({"rev": "0", "default_mode": "assisted"}, "rev"),
    ({"default_mode": "assisted"}, "rev"),
])
def test_put_resolution_validation_names_the_field(body, field) -> None:
    api, store = make()
    before = copy.deepcopy(res_item(store))
    status, out = call(api, "PUT", "/slots/resolution", body, who=ADMIN_USER)
    assert status == 400 and out["field"] == field, out
    assert res_item(store) == before and not all_audits(store)


def test_put_resolution_generative_enabled_must_be_boolean() -> None:
    api, store = make()
    status, out = call(api, "PUT", "/slots/resolution", {"rev": 0, "generative_enabled": "yes"}, who=ADMIN_USER)
    assert status == 400 and out["field"] == "slot_resolution.generative_enabled" and not all_audits(store)


def test_put_resolution_nothing_to_update_is_400() -> None:
    api, _ = make()
    assert call(api, "PUT", "/slots/resolution", {"rev": 0}, who=EDITOR)[0] == 400


def test_put_resolution_stale_rev_is_409_with_current() -> None:
    api, store = make()
    assert call(api, "PUT", "/slots/resolution", {"rev": 0, "resolver_budget_ms": 2000}, who=EDITOR)[0] == 200
    before = copy.deepcopy(res_item(store))
    status, body = call(api, "PUT", "/slots/resolution", {"rev": 0, "default_mode": "deterministic"}, who=EDITOR)
    assert status == 409 and body["error"] == "conflict"
    assert body["current"]["rev"] == 1 and body["current"]["resolver_budget_ms"] == 2000
    assert res_item(store) == before and len(res_audits(store)) == 1


def test_put_resolution_lost_race_is_409() -> None:
    api, store = make()
    real = store.transact_write

    def racing(puts):
        item = res_item(store)
        store.put("policy", {**item, "rev": 4})
        return real(puts)

    store.transact_write = racing  # type: ignore[method-assign]
    status, body = call(api, "PUT", "/slots/resolution", {"rev": 0, "resolver_budget_ms": 900}, who=EDITOR)
    assert status == 409 and body["current"]["rev"] == 4 and not all_audits(store)
    assert res_item(store)["resolver_budget_ms"] == 2500  # type: ignore[index]


def test_put_resolution_when_the_item_is_missing() -> None:
    """A table seeded before phase 2 has no item: it loads ``deterministic``; a write that
    doesn't send ``default_mode`` keeps it so."""
    api, store = make()
    store.delete("policy", {"pk": "slot_resolution"})
    view = call(api, "GET", "/slots", who=VIEWER)[1]["slot_resolution"]
    assert view["default_mode"] == "deterministic" and view["problems"] and view["rev"] == 0
    status, body = call(api, "PUT", "/slots/resolution", {"rev": 0, "resolver_budget_ms": 1000}, who=EDITOR)
    assert status == 200, body
    item = res_item(store)
    assert item is not None and item["default_mode"] == "deterministic" and item["rev"] == 1
    (entry,) = res_audits(store)
    assert entry["changes"] == [{"field": "resolver_budget_ms", "old": 2500, "new": 1000}]


def test_put_resolution_not_seeded_is_503() -> None:
    api, _ = make(MemoryStore())
    assert call(api, "PUT", "/slots/resolution", {"rev": 0, "default_mode": "assisted"}, who=EDITOR)[0] == 503


# --------------------------------------------------------------------------- generative_enabled


@pytest.mark.parametrize("value", [True, False])
def test_generative_enabled_from_an_editor_is_403_and_writes_nothing(value, caplog) -> None:
    api, store = make()
    before = copy.deepcopy(res_item(store))
    with caplog.at_level(logging.WARNING):
        status, body = call(api, "PUT", "/slots/resolution",
                            {"rev": 0, "generative_enabled": value, "resolver_budget_ms": 2000}, who=EDITOR)
    assert status == 403 and body["error"] == "forbidden" and body["field"] == "generative_enabled"
    assert res_item(store) == before and not all_audits(store)
    assert any(getattr(r, "event", None) == "admin_access_denied" or "admin_access_denied" in r.getMessage()
               for r in caplog.records)


def test_generative_enabled_from_a_viewer_is_403() -> None:
    api, store = make()
    status, _ = call(api, "PUT", "/slots/resolution", {"rev": 0, "generative_enabled": True}, who=VIEWER)
    assert status == 403 and not all_audits(store)


def test_generative_enabled_by_an_admin_is_recorded_with_who_and_when() -> None:
    api, store = make(ceiling="generative")
    status, body = call(api, "PUT", "/slots/resolution", {"rev": 0, "generative_enabled": True}, who=ADMIN_USER)
    assert status == 200, body
    assert body["security_change"] is True
    assert codes(body["warnings"]) == ["generative_unvalidated_wording"]
    (entry,) = res_audits(store)
    assert entry["changes"] == [{"field": "generative_enabled", "old": False, "new": True}]
    assert entry["actor"] == {"sub": "sub-admin", "email": "admin@example.com"}
    assert entry["security_change"] is True and entry["ts"]
    # The view shows who set it and when, from the audit log.
    view = call(api, "GET", "/slots", who=VIEWER)[1]["slot_resolution"]
    assert view["generative_enabled"] is True
    assert view["generative_enabled_set"] == {"value": True, "by": entry["actor"], "at": entry["ts"]}
    assert body["slot_resolution"]["generative_enabled_set"] == view["generative_enabled_set"]
    # An editor may still change other fields; the record stays the admin's.
    status, body = call(api, "PUT", "/slots/resolution", {"rev": 1, "resolver_budget_ms": 2500}, who=EDITOR)
    assert status == 200, body
    assert body["slot_resolution"]["generative_enabled_set"]["by"]["sub"] == "sub-admin"
    assert Policy.from_store(store).slot_resolution.generative_enabled is True


def test_generative_enabled_without_generative_ceiling_warns_no_effect() -> None:
    api, _ = make(ceiling="assisted")
    status, body = call(api, "PUT", "/slots/resolution", {"rev": 0, "generative_enabled": True}, who=ADMIN_USER)
    assert status == 200
    assert codes(body["warnings"]) == ["generative_unvalidated_wording", "generative_no_effect"]
    assert codes(body["slot_resolution"]["warnings"]) == ["generative_no_effect"]
    view = call(api, "GET", "/slots", who=VIEWER)[1]["slot_resolution"]
    assert codes(view["warnings"]) == ["generative_no_effect"]


def test_generative_default_mode_by_an_editor_warns() -> None:
    api, _ = make(ceiling="generative")
    status, body = call(api, "PUT", "/slots/resolution", {"rev": 0, "default_mode": "generative"}, who=EDITOR)
    assert status == 200 and codes(body["warnings"]) == ["generative_unvalidated_wording"]


# --------------------------------------------------------------------------- POST /slots/preview


def snapshot(store: MemoryStore) -> Any:
    return copy.deepcopy(store._tables)


def test_preview_makes_no_write_and_shows_who_calls_a_model() -> None:
    api, store = make()
    before = snapshot(store)
    status, body = call(api, "POST", "/slots/preview",
                        {"slot_resolution": {"default_mode": "deterministic"},
                         "slots": {"card": {"mode": "assisted"}}}, who=VIEWER)
    assert status == 200, body
    assert snapshot(store) == before
    assert body["ceiling"] == "assisted"
    assert body["calls_model"] == [{"slot": "card", "flow": "confirm", "effective": "assisted"}]
    assert body["current"]["zip"]["login"]["effective"] == "assisted"
    assert body["proposed"]["zip"]["login"]["effective"] == "deterministic"
    assert {"slot": "zip", "flow": "login", "from": "assisted", "to": "deterministic"} in body["changes"]
    assert not any(c["slot"] == "card" for c in body["changes"])
    assert body["warnings"] == []


def test_preview_capped_by_the_ceiling() -> None:
    api, store = make(ceiling="deterministic")
    status, body = call(api, "POST", "/slots/preview", {"slot_resolution": {"default_mode": "generative"}},
                        who=EDITOR)
    assert status == 200, body
    assert body["calls_model"] == []
    for by_flow in body["proposed"].values():
        for d in by_flow.values():
            assert d["effective"] == "deterministic" and d["capped"] == "ceiling"
    assert codes(body["warnings"]) == ["generative_no_effect"]


def test_preview_away_from_deterministic_warns() -> None:
    api, store = make()
    assert call(api, "PUT", "/slots/resolution", {"rev": 0, "default_mode": "deterministic"}, who=EDITOR)[0] == 200
    status, body = call(api, "POST", "/slots/preview", {"slots": {"zip": {"mode": "assisted"}}}, who=VIEWER)
    assert status == 200 and codes(body["warnings"]) == ["away_from_deterministic"]
    assert list(body["calls_model"]) == [{"slot": "zip", "flow": "login", "effective": "assisted"},
                                                {"slot": "zip", "flow": "stepup", "effective": "assisted"}]


def test_preview_empty_body_is_the_current_table() -> None:
    api, _ = make()
    status, body = call(api, "POST", "/slots/preview", {}, who=VIEWER)
    assert status == 200 and body["current"] == body["proposed"] and body["changes"] == []


@pytest.mark.parametrize("body,field", [
    ({"slot_resolution": {"default_mode": "x"}}, "slot_resolution.default_mode"),
    ({"slot_resolution": {"bogus": 1}}, "slot_resolution.bogus"),
    ({"slot_resolution": []}, "slot_resolution"),
    ({"slots": {"nope": {"mode": "assisted"}}}, "slots.nope"),
    ({"slots": {"zip": {"mode": "x"}}}, "slot#zip.mode"),
    ({"slots": {"zip": {"slotType": "x"}}}, "slots.zip.slotType"),
    ({"slots": {"zip": "assisted"}}, "slots.zip"),
    ({"slots": []}, "slots"),
    ({"other": 1}, "other"),
])
def test_preview_validation(body, field) -> None:
    api, store = make()
    before = snapshot(store)
    status, out = call(api, "POST", "/slots/preview", body, who=VIEWER)
    assert status == 400 and out["field"] == field, out
    assert snapshot(store) == before


def test_preview_and_resolution_routes_need_claims_and_groups() -> None:
    api, store = make()
    assert call(api, "POST", "/slots/preview", {}, no_claims=True)[0] == 401
    assert call(api, "PUT", "/slots/resolution", {"rev": 0, "default_mode": "assisted"}, no_claims=True)[0] == 401
    assert call(api, "POST", "/slots/preview", {}, who=claims(None))[0] == 403
    assert call(api, "PUT", "/slots/resolution", {"rev": 0, "default_mode": "deterministic"}, who=VIEWER)[0] == 403
    assert call(api, "GET", "/slots/resolution", who=VIEWER)[0] == 405
    assert call(api, "GET", "/slots/preview", who=VIEWER)[0] == 405
    assert call(api, "PUT", "/api/slots/resolution", {"rev": 0, "default_mode": "deterministic"},
                who=EDITOR)[0] == 200
    assert not [e for e in all_audits(store) if e["pk"] != audit.RESOLUTION_PK]
