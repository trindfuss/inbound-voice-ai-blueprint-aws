"""Admin API slot routes: ``GET /slots`` and
``PUT /slots/{name}``, with an in-memory store. No network. ``PUT /slots/resolution`` and
``POST /slots/preview`` are in ``test_admin_slot_resolution.py``."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from lambdas.admin_api import audit
from lambdas.admin_api.app import AdminApi, Deps
from lambdas.admin_api.config import AdminConfig
from owlbank.policy import Policy
from owlbank.slot_policy import gather_input
from owlbank.slots import modes
from owlbank.store import MemoryStore
from tests.admin_api_fakes import POOL_ID, Clock, FakeCognito, call, claims, seeded_store

VIEWER = claims(["admin-viewer"])
EDITOR = claims(["admin-editor"])
ADMIN = claims(["admin"])


def make(store: MemoryStore | None = None, *, accept: bool = True,
         ceiling: str = "assisted") -> tuple[AdminApi, MemoryStore]:
    store = store if store is not None else seeded_store()
    clock = Clock()
    api = AdminApi(Deps(store=store, cognito=FakeCognito(), user_pool_id=POOL_ID, retention_days=90,
                        clock=clock, suffix=clock.suffix, accept_spoken_factors=accept,
                        slot_mode_ceiling=ceiling))  # type: ignore[arg-type]
    return api, store


def audits(store: MemoryStore, name: str) -> list[dict[str, Any]]:
    return store.query("admin_audit", {"pk": f"policy#slot#{name}"})


def all_audits(store: MemoryStore) -> list[dict[str, Any]]:
    return list(store._tables["admin_audit"].values())


def slots_by_name(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["name"]: s for s in body["slots"]}


def codes(warnings: list[dict[str, Any]]) -> list[str]:
    return [w["code"] for w in warnings]


# --------------------------------------------------------------------------- config


def test_config_accept_spoken_factors_default_true_and_env() -> None:
    base = {"OWLBANK_ADMIN_USER_POOL_ID": POOL_ID}
    assert AdminConfig.from_env(base).accept_spoken_factors is True
    assert AdminConfig.from_env({**base, "OWLBANK_ACCEPT_SPOKEN_FACTORS": ""}).accept_spoken_factors is True
    assert AdminConfig.from_env({**base, "OWLBANK_ACCEPT_SPOKEN_FACTORS": "true"}).accept_spoken_factors is True
    assert AdminConfig.from_env({**base, "OWLBANK_ACCEPT_SPOKEN_FACTORS": "TRUE"}).accept_spoken_factors is True
    assert AdminConfig.from_env({**base, "OWLBANK_ACCEPT_SPOKEN_FACTORS": "false"}).accept_spoken_factors is False


def test_security_fields_cover_the_slot_settings() -> None:
    assert {"sensitive", "keypad_only"} <= audit.SECURITY_FIELDS
    assert "barge_in" not in audit.SECURITY_FIELDS
    assert {"min_level", "confirm"} <= audit.SECURITY_FIELDS


# --------------------------------------------------------------------------- GET /slots


def test_get_slots_lists_every_slot_with_its_settings() -> None:
    api, _ = make()
    status, body = call(api, "GET", "/slots", who=VIEWER)
    assert status == 200, body
    assert body["acceptSpokenFactors"] is True
    assert [s["name"] for s in body["slots"]] == ["zip", "ssn4", "otp", "phone", "yes_no", "card"]
    s = slots_by_name(body)
    assert s["zip"] == {
        "name": "zip", "type": "digits", "sensitive": False, "keypad_only": False, "barge_in": True,
        "maxRetries": 2, "is_factor": True, "effective_keypad_only": False, "flow": None, "rev": 0,
        "loaded": True, "problems": [], "mode": None, "confirm": "when_inferred", "options": None,
        "modes": {f: {"configured": "assisted", "source": "default", "ceiling": "assisted",
                      "effective": "assisted", "capped": None} for f in ("login", "stepup")},
        "warnings": [{"code": "verification_not_sensitive", "slot": "zip",
                      "message": s["zip"]["warnings"][0]["message"]}],
    }
    assert s["ssn4"]["sensitive"] is True and s["ssn4"]["warnings"] == []
    assert s["yes_no"]["is_factor"] is False and s["yes_no"]["type"] == "yes_no"
    assert s["yes_no"]["flow"] == "confirm" and s["yes_no"]["warnings"] == []
    assert body["takes_effect_s"] == 60


def test_get_slots_verification_warning_only_for_factors_not_sensitive() -> None:
    api, store = make()
    for name in ("zip", "ssn4", "otp", "yes_no"):
        item = store.get("policy", {"pk": f"slot#{name}"})
        store.put("policy", {**item, "sensitive": False})
    s = slots_by_name(call(api, "GET", "/slots", who=VIEWER)[1])
    for name in ("zip", "ssn4", "otp"):
        assert codes(s[name]["warnings"]) == ["verification_not_sensitive"]
        assert "production" in s[name]["warnings"][0]["message"].lower()
    assert s["yes_no"]["warnings"] == []


def test_get_slots_effective_keypad_only_when_spoken_factors_off() -> None:
    api, _ = make(accept=False)
    body = call(api, "GET", "/slots", who=VIEWER)[1]
    assert body["acceptSpokenFactors"] is False
    s = slots_by_name(body)
    for name in ("zip", "ssn4", "otp"):
        assert s[name]["keypad_only"] is False and s[name]["effective_keypad_only"] is True
    assert s["yes_no"]["effective_keypad_only"] is False


def test_get_slots_matches_the_front_doors_gather_input() -> None:
    api, store = make(accept=False)
    item = store.get("policy", {"pk": "slot#yes_no"})
    store.put("policy", {**item, "keypad_only": True, "barge_in": False})
    s = slots_by_name(call(api, "GET", "/slots", who=VIEWER)[1])
    policy = Policy.from_store(store)
    for name in policy.slots():
        g = gather_input(policy.slot(name), False)
        assert (s[name]["effective_keypad_only"], s[name]["barge_in"]) == (g.keypad_only, g.barge_in)


def test_get_slots_loads_fail_closed_like_the_front_door() -> None:
    api, store = make()
    item = store.get("policy", {"pk": "slot#zip"})
    store.put("policy", {**item, "keypad_only": "TYPO", "rev": 3})
    store.delete("policy", {"pk": "slot#ssn4"})
    s = slots_by_name(call(api, "GET", "/slots", who=VIEWER)[1])
    assert s["zip"]["keypad_only"] is True and s["zip"]["effective_keypad_only"] is True
    assert s["zip"]["rev"] == 3
    assert s["zip"]["problems"] == ["slot#zip.keypad_only: must be a boolean; loaded as true (fail closed)"]
    assert s["ssn4"]["sensitive"] is True and s["ssn4"]["rev"] == 0
    assert s["ssn4"]["problems"] and "TYPO" not in str(s)


def test_get_slots_unloadable_non_factor_slot() -> None:
    api, store = make()
    store.put("policy", {"pk": "slot#yes_no", "slotType": "bogus", "rev": 2})
    s = slots_by_name(call(api, "GET", "/slots", who=VIEWER)[1])
    assert s["yes_no"]["loaded"] is False and s["yes_no"]["rev"] == 2
    assert s["yes_no"]["problems"] and s["yes_no"]["sensitive"] is None


def test_get_slots_not_seeded_is_503() -> None:
    api, _ = make(MemoryStore())
    assert call(api, "GET", "/slots", who=VIEWER)[0] == 503


# --------------------------------------------------------------------------- access


@pytest.mark.parametrize("who,method,want", [
    (VIEWER, "GET", 200), (EDITOR, "GET", 200), (ADMIN, "GET", 200),
    (VIEWER, "PUT", 403), (EDITOR, "PUT", 200), (ADMIN, "PUT", 200),
    (claims(None), "GET", 403), (claims(None), "PUT", 403),
])
def test_slot_route_groups(who, method, want) -> None:
    api, store = make()
    path = "/slots" if method == "GET" else "/slots/zip"
    status, _ = call(api, method, path, {"rev": 0, "keypad_only": True}, who=who)
    assert status == want
    if want == 403:
        assert not all_audits(store)
        assert store.get("policy", {"pk": "slot#zip"})["keypad_only"] is False


@pytest.mark.parametrize("method,path", [("GET", "/slots"), ("PUT", "/slots/zip")])
def test_slot_routes_need_claims(method, path) -> None:
    api, store = make()
    status, body = call(api, method, path, {"rev": 0, "keypad_only": True}, no_claims=True)
    assert status == 401 and body["error"] == "unauthorized" and not all_audits(store)


def test_slot_routes_wrong_method_is_405() -> None:
    api, _ = make()
    assert call(api, "DELETE", "/slots/zip", who=ADMIN)[0] == 405
    assert call(api, "POST", "/slots", who=ADMIN)[0] == 405


def test_slot_routes_with_api_prefix() -> None:
    api, _ = make()
    assert call(api, "GET", "/api/slots", who=VIEWER)[0] == 200


# --------------------------------------------------------------------------- PUT /slots/{name}


def test_put_keypad_only_on_writes_item_and_one_security_audit_entry() -> None:
    api, store = make()
    status, body = call(api, "PUT", "/slots/zip", {"rev": 0, "keypad_only": True}, who=EDITOR)
    assert status == 200, body
    assert body["changed"] is True and body["security_change"] is True
    assert body["slot"]["keypad_only"] is True and body["slot"]["rev"] == 1
    item = store.get("policy", {"pk": "slot#zip"})
    assert item["keypad_only"] is True and item["rev"] == 1
    assert item["sensitive"] is False and item["barge_in"] is True
    (entry,) = audits(store, "zip")
    assert entry["action"] == "update" and entry["target"] == {"kind": "slot", "id": "zip"}
    assert entry["changes"] == [{"field": "keypad_only", "old": False, "new": True}]
    assert entry["security_change"] is True and entry["rev"] == 1
    assert entry["actor"] == {"sub": "sub-actor", "email": "actor@example.com"}
    assert {"pk": entry["pk"], "sk": entry["sk"]} == body["audit"]
    assert codes(body["warnings"]) == ["verification_not_sensitive"]  # zip still isn't sensitive
    assert Policy.from_store(store).slot("zip").keypad_only is True


def test_put_sensitive_off_warns_and_is_a_security_change() -> None:
    api, store = make()
    status, body = call(api, "PUT", "/slots/ssn4", {"rev": 0, "sensitive": False}, who=EDITOR)
    assert status == 200, body
    assert body["security_change"] is True
    assert codes(body["warnings"]) == ["sensitive_off", "verification_not_sensitive"]
    w = body["warnings"][0]
    assert w["slot"] == "ssn4" and "security change" in w["message"]
    (entry,) = audits(store, "ssn4")
    assert entry["changes"] == [{"field": "sensitive", "old": True, "new": False}]
    assert entry["security_change"] is True


def test_put_keypad_only_off_warns_and_is_a_security_change() -> None:
    api, store = make()
    assert call(api, "PUT", "/slots/yes_no", {"rev": 0, "keypad_only": True}, who=EDITOR)[0] == 200
    status, body = call(api, "PUT", "/slots/yes_no", {"rev": 1, "keypad_only": False}, who=EDITOR)
    assert status == 200, body
    assert codes(body["warnings"]) == ["keypad_only_off"]
    assert "security change" in body["warnings"][0]["message"]
    entries = sorted(audits(store, "yes_no"), key=lambda e: e["sk"])
    assert [e["security_change"] for e in entries] == [True, True]
    assert entries[1]["changes"] == [{"field": "keypad_only", "old": True, "new": False}]


def test_put_barge_in_is_not_a_security_change() -> None:
    api, store = make()
    status, body = call(api, "PUT", "/slots/ssn4", {"rev": 0, "barge_in": False}, who=EDITOR)
    assert status == 200 and body["security_change"] is False and body["warnings"] == []
    (entry,) = audits(store, "ssn4")
    assert entry["security_change"] is False
    assert entry["changes"] == [{"field": "barge_in", "old": True, "new": False}]


def test_put_sensitive_on_clears_the_verification_warning() -> None:
    api, _ = make()
    status, body = call(api, "PUT", "/slots/zip", {"rev": 0, "sensitive": True}, who=EDITOR)
    assert status == 200 and body["warnings"] == [] and body["security_change"] is True


def test_put_no_change_writes_nothing() -> None:
    api, store = make()
    status, body = call(api, "PUT", "/slots/zip", {"rev": 0, "keypad_only": False}, who=EDITOR)
    assert status == 200 and body["changed"] is False
    assert codes(body["warnings"]) == ["verification_not_sensitive"]
    assert store.get("policy", {"pk": "slot#zip"}).get("rev", 0) == 0 and not all_audits(store)


def test_put_stale_rev_is_409_with_the_current_item() -> None:
    api, store = make()
    assert call(api, "PUT", "/slots/zip", {"rev": 0, "keypad_only": True}, who=EDITOR)[0] == 200
    before = store.get("policy", {"pk": "slot#zip"})
    status, body = call(api, "PUT", "/slots/zip", {"rev": 0, "sensitive": True}, who=EDITOR)
    assert status == 409 and body["error"] == "conflict"
    assert body["current"]["rev"] == 1 and body["current"]["keypad_only"] is True
    assert store.get("policy", {"pk": "slot#zip"}) == before and len(audits(store, "zip")) == 1


def test_put_lost_race_is_409_and_writes_nothing() -> None:
    api, store = make()
    real = store.transact_write

    def racing(puts):  # another writer lands between the read and the transaction
        item = store.get("policy", {"pk": "slot#zip"})
        store.put("policy", {**item, "rev": 5})
        return real(puts)

    store.transact_write = racing  # type: ignore[method-assign]
    status, body = call(api, "PUT", "/slots/zip", {"rev": 0, "keypad_only": True}, who=EDITOR)
    assert status == 409 and body["current"]["rev"] == 5
    assert store.get("policy", {"pk": "slot#zip"})["keypad_only"] is False and not all_audits(store)


@pytest.mark.parametrize("body,field", [
    ({"rev": 0, "sensitive": "yes"}, "slot#zip.sensitive"),
    ({"rev": 0, "keypad_only": 1}, "slot#zip.keypad_only"),
    ({"rev": 0, "barge_in": None}, "slot#zip.barge_in"),
    ({"rev": 0, "mode": "fancy"}, "slot#zip.mode"),
    ({"rev": 0, "mode": "generative"}, "slot#zip.mode"),  # Not a choice slot
    ({"rev": 0, "confirm": "sometimes"}, "slot#zip.confirm"),
    ({"rev": 0, "confirm": None}, "slot#zip.confirm"),
    ({"rev": 0, "options": []}, "slot#zip.options"),  # a factor carries no options
    ({"rev": 0, "slotType": "digits"}, "slotType"),
    ({"rev": "0", "sensitive": True}, "rev"),
    ({"rev": -1, "sensitive": True}, "rev"),
    ({"rev": True, "sensitive": True}, "rev"),
    ({"sensitive": True}, "rev"),
])
def test_put_validation_names_the_field(body, field) -> None:
    api, store = make()
    status, out = call(api, "PUT", "/slots/zip", body, who=EDITOR)
    assert status == 400 and out["field"] == field, out
    assert store.get("policy", {"pk": "slot#zip"})["keypad_only"] is False and not all_audits(store)
    assert "yes" not in out.get("message", "")


def test_put_nothing_to_update_is_400() -> None:
    api, _ = make()
    status, body = call(api, "PUT", "/slots/zip", {"rev": 0}, who=EDITOR)
    assert status == 400 and body["error"] == "invalid"


def test_put_existing_invalid_field_must_be_fixed() -> None:
    api, store = make()
    item = store.get("policy", {"pk": "slot#zip"})
    store.put("policy", {**item, "barge_in": "no"})
    status, body = call(api, "PUT", "/slots/zip", {"rev": 0, "keypad_only": True}, who=EDITOR)
    assert status == 400 and body["field"] == "slot#zip.barge_in"
    status, body = call(api, "PUT", "/slots/zip", {"rev": 0, "barge_in": False}, who=EDITOR)
    assert status == 200, body
    (entry,) = audits(store, "zip")
    assert entry["changes"] == [{"field": "barge_in", "old": "no", "new": False}]


@pytest.mark.parametrize("name", ["nope", "get_balance", "Zip", "a" * 70, "meta"])
def test_put_unknown_slot_is_404(name) -> None:
    api, store = make()
    status, body = call(api, "PUT", f"/slots/{name}", {"rev": 0, "sensitive": True}, who=EDITOR)
    assert status == 404 and body["error"] == "not_found" and not all_audits(store)


def test_put_missing_factor_item_keeps_it_sensitive() -> None:
    """A factor with no ``slot#`` item loads sensitive; a write that doesn't send
    ``sensitive`` keeps it so, and records the item fully."""
    api, store = make()
    store.delete("policy", {"pk": "slot#otp"})
    status, body = call(api, "PUT", "/slots/otp", {"rev": 0, "keypad_only": True}, who=EDITOR)
    assert status == 200, body
    item = store.get("policy", {"pk": "slot#otp"})
    assert item["sensitive"] is True and item["keypad_only"] is True and item["rev"] == 1
    assert Policy.from_store(store).slot("otp").sensitive is True
    (entry,) = audits(store, "otp")
    assert entry["changes"] == [{"field": "keypad_only", "old": False, "new": True}]


def test_put_factor_not_listed_in_meta_is_503() -> None:
    api, store = make()
    meta = store.get("policy", {"pk": "meta"})
    store.put("policy", {**meta, "slots": ["ssn4", "otp", "yes_no"]})
    status, body = call(api, "PUT", "/slots/zip", {"rev": 0, "sensitive": True}, who=EDITOR)
    assert status == 503 and body["error"] == "not_seeded" and not all_audits(store)


def test_slot_audit_is_listed_by_get_audit() -> None:
    api, _ = make()
    assert call(api, "PUT", "/slots/zip", {"rev": 0, "sensitive": True}, who=EDITOR)[0] == 200
    status, body = call(api, "GET", "/audit", query={"item": "policy#slot#zip"}, who=VIEWER)
    assert status == 200 and len(body["entries"]) == 1


def test_put_logs_no_values(caplog) -> None:
    api, store = make()
    item = store.get("policy", {"pk": "slot#zip"})
    store.put("policy", {**item, "barge_in": "SECRETVALUE"})
    with caplog.at_level(logging.DEBUG):
        call(api, "PUT", "/slots/zip", {"rev": 0, "barge_in": True}, who=EDITOR)
        call(api, "PUT", "/slots/zip", {"rev": 0, "sensitive": "SECRETVALUE"}, who=EDITOR)
        call(api, "GET", "/slots", who=VIEWER)
    text = "\n".join(r.getMessage() + " " + str(r.__dict__) for r in caplog.records)
    assert "SECRETVALUE" not in text


# =========================================================================== phase 2


def test_config_slot_mode_ceiling_default_valid_and_invalid() -> None:
    from lambdas.admin_api.config import slot_mode_ceiling_from_env

    base = {"OWLBANK_ADMIN_USER_POOL_ID": POOL_ID}
    assert AdminConfig.from_env(base).slot_mode_ceiling == "assisted"
    assert AdminConfig.from_env({**base, "OWLBANK_SLOT_MODE_CEILING": ""}).slot_mode_ceiling == "assisted"
    for m in ("deterministic", "assisted", "generative"):
        assert AdminConfig.from_env({**base, "OWLBANK_SLOT_MODE_CEILING": m}).slot_mode_ceiling == m
    for bad in ("Generative", "none", "true"):
        with pytest.raises(ValueError, match="OWLBANK_SLOT_MODE_CEILING"):
            AdminConfig.from_env({**base, "OWLBANK_SLOT_MODE_CEILING": bad})
        with pytest.raises(ValueError):
            slot_mode_ceiling_from_env({"OWLBANK_SLOT_MODE_CEILING": bad})
    with pytest.raises(ValueError):
        AdminConfig(user_pool_id=POOL_ID, slot_mode_ceiling="bogus")  # type: ignore[arg-type]


def test_handler_import_fails_on_an_invalid_ceiling(monkeypatch) -> None:
    import importlib
    import sys

    monkeypatch.setenv("OWLBANK_SLOT_MODE_CEILING", "bogus")
    sys.modules.pop("lambdas.admin_api.handler", None)
    with pytest.raises(ValueError):
        importlib.import_module("lambdas.admin_api.handler")
    monkeypatch.setenv("OWLBANK_SLOT_MODE_CEILING", "deterministic")
    sys.modules.pop("lambdas.admin_api.handler", None)
    assert importlib.import_module("lambdas.admin_api.handler").SLOT_MODE_CEILING == "deterministic"


def test_security_fields_phase2() -> None:
    assert {"mode", "default_mode", "flows", "confirm", "generative_enabled"} <= audit.SECURITY_FIELDS
    assert not {"options", "resolver_budget_ms", "barge_in"} & audit.SECURITY_FIELDS


def test_get_slots_phase2_fields() -> None:
    api, _ = make()
    status, body = call(api, "GET", "/slots", who=VIEWER)
    assert status == 200, body
    assert body["ceiling"] == "assisted" and body["languages"] == ["en-US", "es-US"]
    r = body["slot_resolution"]
    assert {k: r[k] for k in ("rev", "default_mode", "flows", "generative_enabled", "resolver_budget_ms",
                              "problems", "generative_enabled_set", "warnings")} == {
        "rev": 0, "default_mode": "assisted", "flows": {}, "generative_enabled": False,
        "resolver_budget_ms": 2500, "problems": [], "generative_enabled_set": None, "warnings": []}
    s = slots_by_name(body)
    assert s["ssn4"]["confirm"] == "never" and s["zip"]["confirm"] == "when_inferred"
    assert s["yes_no"]["options"] is None and set(s["yes_no"]["modes"]) == {"confirm"}
    card = s["card"]
    assert card["type"] == "choice" and card["mode"] is None and set(card["modes"]) == {"confirm"}
    assert card["options"][0] == {
        "value": "red", "dtmf": None,
        "spoken_names": {"en-US": "OwlBank Red Card", "es-US": "Tarjeta OwlBank Roja"},
        "aliases": {"en-US": ["red"], "es-US": ["roja"]}}
    assert [o["value"] for o in card["options"]] == ["red", "purple", "rewards"]


@pytest.mark.parametrize("ceiling", ["deterministic", "assisted", "generative"])
def test_get_slots_modes_equal_the_front_doors_freeze(ceiling) -> None:
    api, store = make(ceiling=ceiling)
    res = store.get("policy", {"pk": "slot_resolution"})
    store.put("policy", {**res, "generative_enabled": True, "flows": {"login": {"mode": "deterministic"},
                                                                       "confirm": {"mode": "generative"}}})
    card = store.get("policy", {"pk": "slot#card"})
    store.put("policy", {**card, "mode": "generative"})
    zip_item = store.get("policy", {"pk": "slot#zip"})
    store.put("policy", {**zip_item, "mode": "bogus"})
    body = call(api, "GET", "/slots", who=VIEWER)[1]
    got = {s["name"]: s["modes"] for s in body["slots"]}
    policy = Policy.from_store(store)
    for name, by_flow in modes.mode_table(policy, ceiling).items():
        assert got[name] == {f: d.to_doc() for f, d in by_flow.items()}
    assert got["zip"]["login"]["source"] == "invalid" and got["zip"]["login"]["effective"] == "deterministic"
    want_card = "generative" if ceiling == "generative" else ceiling
    assert got["card"]["confirm"]["effective"] == want_card
    s = slots_by_name(body)
    no_effect = ceiling != "generative"
    assert ("generative_no_effect" in codes(s["card"]["warnings"])) is no_effect
    assert ("generative_no_effect" in codes(body["slot_resolution"]["warnings"])) is no_effect


def test_put_mode_is_a_security_change_and_warns_away_from_deterministic() -> None:
    api, store = make()
    status, body = call(api, "PUT", "/slots/ssn4", {"rev": 0, "mode": "deterministic"}, who=EDITOR)
    assert status == 200, body
    assert body["security_change"] is True and body["warnings"] == []
    assert body["slot"]["modes"]["login"]["configured"] == "deterministic"
    assert body["slot"]["modes"]["login"]["source"] == "slot"
    (entry,) = audits(store, "ssn4")
    assert entry["changes"] == [{"field": "mode", "old": None, "new": "deterministic"}]
    status, body = call(api, "PUT", "/slots/ssn4", {"rev": 1, "mode": None}, who=EDITOR)
    assert status == 200, body
    assert codes(body["warnings"]) == ["away_from_deterministic"]
    assert "mode" not in store.get("policy", {"pk": "slot#ssn4"})
    assert body["slot"]["modes"]["login"]["source"] == "default"
    assert Policy.from_store(store).slot("ssn4").mode is None


def test_put_confirm_is_a_security_change() -> None:
    api, store = make()
    status, body = call(api, "PUT", "/slots/zip", {"rev": 0, "confirm": "always"}, who=EDITOR)
    assert status == 200 and body["security_change"] is True, body
    assert body["slot"]["confirm"] == "always"
    assert Policy.from_store(store).slot("zip").confirm == "always"


def test_put_generative_on_choice_warns_unvalidated_wording_and_no_effect() -> None:
    api, _ = make()
    status, body = call(api, "PUT", "/slots/card", {"rev": 0, "mode": "generative"}, who=EDITOR)
    assert status == 200, body
    assert codes(body["warnings"]) == ["generative_unvalidated_wording", "generative_no_effect"]
    d = body["slot"]["modes"]["confirm"]
    assert (d["configured"], d["effective"], d["capped"]) == ("generative", "assisted", "ceiling")
    api, _ = make(ceiling="generative")
    status, body = call(api, "PUT", "/slots/card", {"rev": 0, "mode": "generative"}, who=EDITOR)
    assert codes(body["warnings"]) == ["generative_unvalidated_wording"]
    d = body["slot"]["modes"]["confirm"]
    assert (d["effective"], d["capped"]) == ("assisted", "generative_disabled")


def test_put_generative_on_sensitive_choice_is_400() -> None:
    api, store = make()
    status, body = call(api, "PUT", "/slots/card", {"rev": 0, "sensitive": True, "mode": "generative"},
                        who=EDITOR)
    assert status == 400 and body["field"] == "slot#card.mode" and not all_audits(store)


def _card_options(**extra: Any) -> list[dict[str, Any]]:
    return [{"value": "red", "aliases": {"en-US": ["red"], "es-US": ["roja"]}},
            {"value": "purple", "aliases": {"en-US": ["purple"], "es-US": ["morada"]}, **extra}]


def test_put_options_writes_and_is_not_a_security_change() -> None:
    api, store = make()
    status, body = call(api, "PUT", "/slots/card", {"rev": 0, "options": _card_options(dtmf="2")},
                        who=EDITOR)
    assert status == 200, body
    assert body["security_change"] is False
    assert [o["value"] for o in body["slot"]["options"]] == ["red", "purple"]
    assert body["slot"]["options"][1]["dtmf"] == "2"
    (entry,) = audits(store, "card")
    assert [c["field"] for c in entry["changes"]] == ["options"]
    assert [o.value for o in Policy.from_store(store).slot("card").options] == ["red", "purple"]


@pytest.mark.parametrize("options,field", [
    ([{"value": "red"}, {"value": "red"}], "slot#card.options[1].value"),
    ([{"value": "Red"}], "slot#card.options[0].value"),
    ([{"value": "red", "aliases": {"fr-FR": ["rouge"]}}], "slot#card.options[0].aliases"),
    ([{"value": "red", "aliases": {"en-US": ["card"]}}, {"value": "purple", "aliases": {"en-US": ["card"]}}],
     "slot#card.options.aliases.en-US"),
    ([{"value": "red", "dtmf": "12"}], "slot#card.options[0].dtmf"),
    ([{"value": "red", "colour": "x"}], "slot#card.options[0].colour"),
    ("red", "slot#card.options"),
])
def test_put_bad_options_name_the_field(options, field) -> None:
    api, store = make()
    status, body = call(api, "PUT", "/slots/card", {"rev": 0, "options": options}, who=EDITOR)
    assert status == 400 and body["field"] == field, body
    assert not all_audits(store) and store.get("policy", {"pk": "slot#card"}).get("rev", 0) == 0


def test_put_alias_equal_to_another_options_spoken_name_is_400() -> None:
    """An alias of one option that is another option's spoken name (normalized) is refused."""
    api, store = make()
    options = [{"value": "red", "aliases": {"en-US": ["owlbank purple card"]}}, {"value": "purple"}]
    status, body = call(api, "PUT", "/slots/card", {"rev": 0, "options": options}, who=EDITOR)
    assert status == 400 and body["field"] == "slot#card.options.aliases.en-US", body
    assert "purple" in body["message"] and "red" in body["message"] and not all_audits(store)


def test_put_option_without_catalog_text_in_every_language_is_400() -> None:
    api, store = make()
    status, body = call(api, "PUT", "/slots/card", {"rev": 0, "options": [{"value": "gold"}]}, who=EDITOR)
    assert status == 400 and body["field"] == "slot#card.options[0].value", body
    assert "option.card.gold" in body["message"] and "en-US" in body["message"]
    # One language only: still refused, naming the missing one.
    assert call(api, "PUT", "/prompts/option.card.gold",
                {"rev": 0, "texts": {"en-US": "OwlBank Gold Card", "es-US": "Tarjeta OwlBank Dorada"}},
                who=EDITOR)[0] == 200
    item = store.get("catalog", {"key": "option.card.gold", "lang": "es-US"})
    store.delete("catalog", {"key": "option.card.gold", "lang": "es-US"})
    status, body = call(api, "PUT", "/slots/card", {"rev": 0, "options": [{"value": "gold"}]}, who=EDITOR)
    assert status == 400 and "es-US" in body["message"], body
    store.put("catalog", item)
    status, body = call(api, "PUT", "/slots/card", {"rev": 0, "options": [{"value": "gold"}]}, who=EDITOR)
    assert status == 200, body
    assert body["slot"]["options"][0]["spoken_names"] == {"en-US": "OwlBank Gold Card",
                                                          "es-US": "Tarjeta OwlBank Dorada"}


def test_put_options_on_factor_or_yes_no_is_400() -> None:
    api, store = make()
    for name in ("zip", "yes_no"):
        status, body = call(api, "PUT", f"/slots/{name}", {"rev": 0, "options": [{"value": "red"}]},
                            who=EDITOR)
        assert status == 400 and body["field"] == f"slot#{name}.options", body
    assert not all_audits(store)


def test_put_mode_stale_rev_is_409() -> None:
    api, store = make()
    assert call(api, "PUT", "/slots/card", {"rev": 0, "confirm": "always"}, who=EDITOR)[0] == 200
    status, body = call(api, "PUT", "/slots/card", {"rev": 0, "mode": "deterministic"}, who=EDITOR)
    assert status == 409 and body["current"]["confirm"] == "always"
    assert "mode" not in store.get("policy", {"pk": "slot#card"}) and len(audits(store, "card")) == 1


# --------------------------------------------------------------------------- option.* prompts


def test_option_prompts_are_listed_and_editable() -> None:
    api, store = make()
    status, body = call(api, "GET", "/prompts", query={"prefix": "option.card."}, who=VIEWER)
    assert status == 200 and {p["key"] for p in body["prompts"]} == {
        "option.card.red", "option.card.purple", "option.card.rewards"}
    texts = {"en-US": "OwlBank Scarlet Card", "es-US": "Tarjeta OwlBank Escarlata"}
    status, body = call(api, "PUT", "/prompts/option.card.red", {"rev": 0, "texts": texts}, who=EDITOR)
    assert status == 200 and body["changed"] is True, body
    view = slots_by_name(call(api, "GET", "/slots", who=VIEWER)[1])["card"]["options"][0]
    assert view["spoken_names"] == texts


@pytest.mark.parametrize("key,texts,field", [
    ("option.card.red", {"en-US": " ", "es-US": "Roja"}, "texts[en-US]"),
    ("option.card.red", {"en-US": "Red {x}", "es-US": "Roja"}, "texts[en-US]"),
    ("option.card.red", {"en-US": "Red"}, "texts[es-US]"),
    ("option.zip.red", {"en-US": "Red", "es-US": "Roja"}, "key"),       # not a choice slot
    ("option.nope.red", {"en-US": "Red", "es-US": "Roja"}, "key"),      # no such slot
    ("option.card.Red", {"en-US": "Red", "es-US": "Roja"}, "key"),      # not an option value
    ("option.card", {"en-US": "Red", "es-US": "Roja"}, "key"),
    # Purple's spoken name would equal red's alias
    ("option.card.purple", {"en-US": "Red", "es-US": "Morada"}, "texts[en-US]"),
])
def test_option_prompt_validation(key, texts, field) -> None:
    api, store = make()
    before = store.query("catalog", {"key": key})
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts}, who=EDITOR)
    assert status == 400 and body["field"] == field, body
    assert store.query("catalog", {"key": key}) == before and not all_audits(store)
