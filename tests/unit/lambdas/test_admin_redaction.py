"""Admin API ``GET /redaction`` and ``PUT /redaction``, with an in-memory store. No network."""

from __future__ import annotations

import copy
import logging
from typing import Any

import pytest

from lambdas.admin_api import audit
from lambdas.admin_api.app import ACCESS, ADMIN, ROUTES, AdminApi, Deps
from owlbank.pii import COMPREHEND_PII_ENTITY_TYPES, DEFAULT_CONFIG, DEFAULT_ENTITY_TYPES, RedactionConfig
from owlbank.policy import Policy
from owlbank.store import MemoryStore, TransactionCancelled
from tests.admin_api_fakes import POOL_ID, Clock, FakeCognito, call, claims, seeded_store

VIEWER = claims(["admin-viewer"], sub="sub-viewer", email="viewer@example.com")
EDITOR = claims(["admin-editor"], sub="sub-editor", email="editor@example.com")
ADMIN_USER = claims(["admin"], sub="sub-admin", email="admin@example.com")


def make(store: MemoryStore | None = None) -> tuple[AdminApi, MemoryStore]:
    store = store if store is not None else seeded_store()
    clock = Clock()
    api = AdminApi(Deps(store=store, cognito=FakeCognito(), user_pool_id=POOL_ID, retention_days=90,
                        clock=clock, suffix=clock.suffix))
    return api, store


def item(store: MemoryStore) -> dict[str, Any] | None:
    return store.get("policy", {"pk": "redaction"})


def audits(store: MemoryStore) -> list[dict[str, Any]]:
    return sorted(store.query("admin_audit", {"pk": "policy#redaction"}), key=lambda e: e["sk"])


def all_audits(store: MemoryStore) -> list[dict[str, Any]]:
    return list(store._tables["admin_audit"].values())


def codes(warnings: list[dict[str, Any]]) -> list[str]:
    return [w["code"] for w in warnings]


# --------------------------------------------------------------------------- routes and access


def test_routes_and_access() -> None:
    by_name = {r.name: r for r in ROUTES}
    assert (by_name["get_redaction"].method, by_name["get_redaction"].template) == ("GET", "/redaction")
    assert by_name["get_redaction"].access == "read"
    assert (by_name["put_redaction"].method, by_name["put_redaction"].template) == ("PUT", "/redaction")
    assert ACCESS[by_name["put_redaction"].access] == ADMIN == frozenset({"admin"})


def test_security_fields_hold_every_redaction_field_but_not_a_tools_mask() -> None:
    assert {f"redaction.{f}" for f in ("enabled", "entity_types", "min_confidence", "mask", "on_failure")} \
        <= audit.SECURITY_FIELDS
    # A tool's ``mask`` (the masking phrase) stays a non-security change.
    assert "mask" not in audit.SECURITY_FIELDS


# --------------------------------------------------------------------------- GET /redaction


@pytest.mark.parametrize("who", [VIEWER, EDITOR, ADMIN_USER])
def test_any_console_group_can_get(who) -> None:
    api, _ = make()
    status, body = call(api, "GET", "/redaction", who=who)
    assert status == 200, body
    assert body["effective"] == DEFAULT_CONFIG.to_doc() and body["defaults"] == DEFAULT_CONFIG.to_doc()
    assert body["redaction"]["enabled"] is True and body["rev"] == 0
    assert body["problems"] == [] and body["warnings"] == []
    assert body["entity_types_available"] == sorted(COMPREHEND_PII_ENTITY_TYPES)
    assert body["takes_effect_s"] == 60


def test_get_with_the_api_prefix() -> None:
    api, _ = make()
    assert call(api, "GET", "/api/redaction", who=VIEWER)[0] == 200


def test_get_while_disabled_warns() -> None:
    store = seeded_store()
    store.put("policy", {**item(store), "enabled": False})  # type: ignore[dict-item]
    api, _ = make(store)
    status, body = call(api, "GET", "/redaction", who=VIEWER)
    assert status == 200 and body["effective"]["enabled"] is False
    assert codes(body["warnings"]) == ["redaction_off"]


def test_get_invalid_item_shows_the_defaults_and_the_field() -> None:
    store = seeded_store()
    store.put("policy", {**item(store), "on_failure": "shrug"})  # type: ignore[dict-item]
    api, _ = make(store)
    status, body = call(api, "GET", "/redaction", who=VIEWER)
    assert status == 200 and body["effective"] == DEFAULT_CONFIG.to_doc()
    assert body["redaction"]["on_failure"] == "shrug"
    assert [p["field"] for p in body["problems"]] == ["redaction.on_failure"]


def test_get_missing_item() -> None:
    store = seeded_store()
    store.delete("policy", {"pk": "redaction"})
    api, _ = make(store)
    status, body = call(api, "GET", "/redaction", who=VIEWER)
    assert status == 200 and body["redaction"] == {} and body["rev"] == 0
    assert body["effective"] == DEFAULT_CONFIG.to_doc()
    assert [p["field"] for p in body["problems"]] == ["redaction"]


# --------------------------------------------------------------------------- PUT /redaction


def test_put_writes_item_and_one_security_audit_entry() -> None:
    api, store = make()
    status, body = call(api, "PUT", "/redaction", {"rev": 0, "min_confidence": 0.75}, who=ADMIN_USER)
    assert status == 200, body
    assert body["changed"] is True and body["security_change"] is True and body["warnings"] == []
    assert body["rev"] == 1 and body["effective"]["min_confidence"] == "0.75"
    stored = item(store)
    assert stored is not None and stored["rev"] == 1 and stored["min_confidence"] == "0.75"
    (entry,) = audits(store)
    assert entry["target"] == {"kind": "redaction", "id": "redaction"}
    assert entry["changes"] == [{"field": "redaction.min_confidence", "old": "0.5", "new": "0.75"}]
    assert entry["security_change"] is True and entry["rev"] == 1
    assert entry["actor"] == {"sub": "sub-admin", "email": "admin@example.com"}
    assert {"pk": entry["pk"], "sk": entry["sk"]} == body["audit"]
    assert Policy.from_store(store).redaction.min_confidence == 0.75
    status, listed = call(api, "GET", "/audit", query={"item": "policy#redaction"}, who=VIEWER)
    assert status == 200 and len(listed["entries"]) == 1


def test_put_keeps_the_fields_it_was_not_sent() -> None:
    api, store = make()
    assert call(api, "PUT", "/redaction", {"rev": 0, "mask": "***"}, who=ADMIN_USER)[0] == 200
    status, body = call(api, "PUT", "/redaction", {"rev": 1, "on_failure": "allow"}, who=ADMIN_USER)
    assert status == 200, body
    cfg = Policy.from_store(store).redaction
    assert cfg == RedactionConfig(mask="***", on_failure="allow")


def test_switching_off_warns() -> None:
    api, store = make()
    status, body = call(api, "PUT", "/redaction", {"rev": 0, "enabled": False}, who=ADMIN_USER)
    assert status == 200 and body["security_change"] is True, body
    assert codes(body["warnings"]) == ["redaction_off"]
    cfg = Policy.from_store(store).redaction
    assert cfg.enabled is False and cfg.entity_types == DEFAULT_ENTITY_TYPES  # kept for switching back
    (entry,) = audits(store)
    assert entry["changes"] == [{"field": "redaction.enabled", "old": True, "new": False}]


def test_no_types_selected_is_saved_as_disabled() -> None:
    """No type selected means nothing is redacted; the console saves it as
    ``enabled: false`` with an empty list, which the server's strict parse accepts."""
    api, store = make()
    status, body = call(api, "PUT", "/redaction", {"rev": 0, "enabled": False, "entity_types": []},
                        who=ADMIN_USER)
    assert status == 200, body
    assert codes(body["warnings"]) == ["redaction_off", "entity_type_removed"]
    removed = next(w for w in body["warnings"] if w["code"] == "entity_type_removed")
    assert removed["entity_types"] == sorted(DEFAULT_ENTITY_TYPES)
    cfg = Policy.from_store(store).redaction
    assert cfg.enabled is False and cfg.entity_types == ()


def test_enabled_with_no_types_is_refused() -> None:
    api, store = make()
    before = copy.deepcopy(item(store))
    status, body = call(api, "PUT", "/redaction", {"rev": 0, "entity_types": []}, who=ADMIN_USER)
    assert status == 400 and body["field"] == "redaction.entity_types", body
    assert item(store) == before and not all_audits(store)


def test_removing_a_type_warns_with_the_removed_types() -> None:
    api, _ = make()
    types = [t for t in DEFAULT_ENTITY_TYPES if t not in ("PIN", "SSN")] + ["EMAIL"]
    status, body = call(api, "PUT", "/redaction", {"rev": 0, "entity_types": types}, who=ADMIN_USER)
    assert status == 200, body
    assert codes(body["warnings"]) == ["entity_type_removed"]
    assert body["warnings"][0]["entity_types"] == ["PIN", "SSN"]


def test_adding_a_type_does_not_warn() -> None:
    api, _ = make()
    status, body = call(api, "PUT", "/redaction",
                        {"rev": 0, "entity_types": [*DEFAULT_ENTITY_TYPES, "EMAIL"]}, who=ADMIN_USER)
    assert status == 200 and body["warnings"] == [], body


def test_switching_back_on_does_not_warn() -> None:
    api, _ = make()
    assert call(api, "PUT", "/redaction", {"rev": 0, "enabled": False}, who=ADMIN_USER)[0] == 200
    status, body = call(api, "PUT", "/redaction", {"rev": 1, "enabled": True}, who=ADMIN_USER)
    assert status == 200 and body["warnings"] == [] and body["security_change"] is True, body


def test_no_change_writes_nothing() -> None:
    api, store = make()
    status, body = call(api, "PUT", "/redaction", {"rev": 0, "min_confidence": "0.5", "enabled": True},
                        who=ADMIN_USER)
    assert status == 200 and body["changed"] is False, body
    assert not all_audits(store) and item(store).get("rev", 0) == 0  # type: ignore[union-attr]


@pytest.mark.parametrize("body,field", [
    ({"rev": 0, "enabled": "false"}, "redaction.enabled"),
    ({"rev": 0, "enabled": None}, "redaction.enabled"),
    ({"rev": 0, "entity_types": "SSN"}, "redaction.entity_types"),
    ({"rev": 0, "entity_types": ["SSN", "NOT_A_TYPE"]}, "redaction.entity_types"),
    ({"rev": 0, "entity_types": [1]}, "redaction.entity_types"),
    ({"rev": 0, "min_confidence": 1.5}, "redaction.min_confidence"),
    ({"rev": 0, "min_confidence": -0.1}, "redaction.min_confidence"),
    ({"rev": 0, "min_confidence": "high"}, "redaction.min_confidence"),
    ({"rev": 0, "min_confidence": True}, "redaction.min_confidence"),
    ({"rev": 0, "mask": ""}, "redaction.mask"),
    ({"rev": 0, "mask": "[{name}]"}, "redaction.mask"),
    ({"rev": 0, "mask": 7}, "redaction.mask"),
    ({"rev": 0, "on_failure": "shrug"}, "redaction.on_failure"),
    ({"rev": 0, "on_failure": ["allow"]}, "redaction.on_failure"),
    ({"rev": 0, "apply_to": ["journal"]}, "apply_to"),
    ({"rev": 0, "bogus": 1}, "bogus"),
    ({"rev": 0}, None),
    ({"rev": "0", "enabled": False}, "rev"),
    ({"rev": -1, "enabled": False}, "rev"),
    ({"enabled": False}, "rev"),
])
def test_put_validation_names_the_field(body, field) -> None:
    api, store = make()
    before = copy.deepcopy(item(store))
    status, out = call(api, "PUT", "/redaction", body, who=ADMIN_USER)
    assert status == 400 and out["error"] == "invalid", out
    assert out.get("field") == field
    assert item(store) == before and not all_audits(store)


def test_put_body_must_be_an_object() -> None:
    api, store = make()
    status, _ = call(api, "PUT", "/redaction", raw_body="[1]", who=ADMIN_USER)
    assert status == 400 and not all_audits(store)


@pytest.mark.parametrize("who", [VIEWER, EDITOR, claims(["Admin"]), claims([])])
def test_put_is_admin_only(who, caplog: pytest.LogCaptureFixture) -> None:
    api, store = make()
    before = copy.deepcopy(item(store))
    with caplog.at_level(logging.WARNING, logger="lambdas.admin_api.app"):
        status, body = call(api, "PUT", "/redaction", {"rev": 0, "enabled": False}, who=who)
    assert status == 403 and body["error"] == "forbidden", body
    assert item(store) == before and not all_audits(store)
    assert any(getattr(r, "event", None) == "admin_access_denied" or "admin_access_denied" in r.getMessage()
               for r in caplog.records)


def test_put_without_a_token_is_401() -> None:
    api, store = make()
    status, _ = call(api, "PUT", "/redaction", {"rev": 0, "enabled": False}, no_claims=True)
    assert status == 401 and not all_audits(store)


def test_other_methods_are_405() -> None:
    api, _ = make()
    status, body = call(api, "POST", "/redaction", {"rev": 0}, who=ADMIN_USER)
    assert status == 405 and "GET" in body["message"] and "PUT" in body["message"]


def test_stale_rev_is_409_and_changes_nothing() -> None:
    api, store = make()
    assert call(api, "PUT", "/redaction", {"rev": 0, "mask": "***"}, who=ADMIN_USER)[0] == 200
    before = copy.deepcopy(item(store))
    status, body = call(api, "PUT", "/redaction", {"rev": 0, "enabled": False}, who=ADMIN_USER)
    assert status == 409 and body["error"] == "conflict", body
    assert body["current"]["rev"] == 1 and body["current"]["effective"]["mask"] == "***"
    assert item(store) == before and len(audits(store)) == 1


def test_rev_ahead_is_409() -> None:
    api, store = make()
    status, _ = call(api, "PUT", "/redaction", {"rev": 5, "enabled": False}, who=ADMIN_USER)
    assert status == 409 and not all_audits(store)


def test_lost_race_in_the_transaction_is_409(monkeypatch: pytest.MonkeyPatch) -> None:
    api, store = make()
    before = copy.deepcopy(item(store))

    def cancelled(puts: Any) -> None:
        raise TransactionCancelled("transaction cancelled: condition failed", [0])

    monkeypatch.setattr(store, "transact_write", cancelled)
    status, body = call(api, "PUT", "/redaction", {"rev": 0, "enabled": False}, who=ADMIN_USER)
    assert status == 409 and "current" in body, body
    assert item(store) == before and not all_audits(store)


def test_put_writes_one_transaction_of_item_and_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    api, store = make()
    seen: list[Any] = []
    real = store.transact_write

    def spy(puts: Any) -> None:
        seen.append(list(puts))
        real(puts)

    monkeypatch.setattr(store, "transact_write", spy)
    assert call(api, "PUT", "/redaction", {"rev": 0, "enabled": False}, who=ADMIN_USER)[0] == 200
    (puts,) = seen
    assert [p.table for p in puts] == ["policy", audit.AUDIT_TABLE]
    assert puts[0].item["pk"] == "redaction" and puts[0].item["rev"] == 1


def test_missing_item_is_written_whole() -> None:
    store = seeded_store()
    store.delete("policy", {"pk": "redaction"})
    api, _ = make(store)
    status, body = call(api, "PUT", "/redaction", {"rev": 0, "on_failure": "withhold"}, who=ADMIN_USER)
    assert status == 200 and body["changed"] is True, body
    stored = item(store)
    assert stored is not None and {k: v for k, v in stored.items() if k not in ("pk", "rev")} \
        == DEFAULT_CONFIG.to_doc()
    (entry,) = audits(store)
    assert {c["field"] for c in entry["changes"]} == {
        f"redaction.{f}" for f in ("enabled", "entity_types", "min_confidence", "mask", "on_failure")}
    assert all(c["old"] is None for c in entry["changes"])


def test_a_legacy_apply_to_is_dropped_on_write() -> None:
    store = seeded_store()
    store.put("policy", {**item(store), "apply_to": ["journal"]})  # type: ignore[dict-item]
    api, _ = make(store)
    status, body = call(api, "PUT", "/redaction", {"rev": 0, "mask": "***"}, who=ADMIN_USER)
    assert status == 200, body
    assert "apply_to" not in item(store)  # type: ignore[operator]
    assert Policy.from_store(store).redaction.mask == "***"


def test_an_invalid_stored_field_can_be_fixed() -> None:
    store = seeded_store()
    store.put("policy", {**item(store), "on_failure": "shrug"})  # type: ignore[dict-item]
    api, _ = make(store)
    status, body = call(api, "PUT", "/redaction", {"rev": 0, "mask": "***"}, who=ADMIN_USER)
    assert status == 400 and body["field"] == "redaction.on_failure"
    status, body = call(api, "PUT", "/redaction", {"rev": 0, "on_failure": "allow"}, who=ADMIN_USER)
    assert status == 200 and body["problems"] == [], body


def test_a_redaction_change_cannot_be_restored() -> None:
    """``POST /restore`` is open to editors; a redaction change must not be restorable through it
    (``PUT /redaction`` is admin only)."""
    api, store = make()
    assert call(api, "PUT", "/redaction", {"rev": 0, "enabled": False}, who=ADMIN_USER)[0] == 200
    (entry,) = audits(store)
    before = copy.deepcopy(item(store))
    for who in (EDITOR, ADMIN_USER):
        status, body = call(api, "POST", "/restore",
                            {"audit_pk": entry["pk"], "audit_sk": entry["sk"], "rev": 1}, who=who)
        assert status == 400 and body["error"] == "not_restorable", body
    assert item(store) == before


def test_responses_and_logs_carry_only_setting_names(caplog: pytest.LogCaptureFixture) -> None:
    api, _ = make()
    with caplog.at_level(logging.DEBUG):
        status, _ = call(api, "PUT", "/redaction", {"rev": 0, "mask": "[HIDDEN {type} sentinel-x]"},
                         who=ADMIN_USER)
    assert status == 200
    assert not any("sentinel-x" in r.getMessage() or "sentinel-x" in str(r.__dict__) for r in caplog.records)
