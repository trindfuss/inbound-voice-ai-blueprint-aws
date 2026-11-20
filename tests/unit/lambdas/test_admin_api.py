"""Admin API: every route and status code, with an
in-memory store and a fake Cognito pool. No network."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from lambdas.admin_api import config_routes
from lambdas.admin_api import handler as handler_mod
from lambdas.admin_api.app import ROUTES, AdminApi
from lambdas.admin_api.config import AdminConfig
from lambdas.admin_api.http import ApiError, parse_groups
from owlbank import masking_config as mc
from owlbank.catalog import PromptCatalog
from owlbank.policy import Policy
from owlbank.store import MemoryStore
from tests.admin_api_fakes import POOL_ID, FakeCognito, call, claims, event, make_api

VIEWER = claims(["admin-viewer"])
EDITOR = claims(["admin-editor"])
ADMIN = claims(["admin"])


@pytest.fixture
def rig() -> tuple[AdminApi, MemoryStore, FakeCognito]:
    api, store, cognito, _ = make_api()
    return api, store, cognito


def audits(store: MemoryStore, pk: str) -> list[dict[str, Any]]:
    return store.query("admin_audit", {"pk": pk})


def all_audits(store: MemoryStore) -> list[dict[str, Any]]:
    return [it for table in [store._tables["admin_audit"]] for it in table.values()]


def _concrete(template: str) -> str:
    return (template.replace("{name}", "get_balance").replace("{key}", "mask.get_balance")
            .replace("{tool}", "get_balance").replace("{username}", "u-1"))


# --------------------------------------------------------------------------- auth and routing


@pytest.mark.parametrize("route", ROUTES, ids=lambda r: f"{r.method} {r.template}")
def test_every_route_needs_claims(rig, route) -> None:
    api, store, cognito = rig
    status, body = call(api, route.method, _concrete(route.template), {"rev": 0}, no_claims=True)
    assert status == 401 and body["error"] == "unauthorized"
    assert not cognito.calls and not all_audits(store)


def test_unknown_route_without_claims_is_401_and_with_claims_404(rig) -> None:
    api, _, _ = rig
    assert call(api, "GET", "/customers", no_claims=True)[0] == 401
    assert call(api, "GET", "/customers")[0] == 404


def test_claims_without_sub_are_401(rig) -> None:
    api, _, _ = rig
    who = claims(["admin"])
    del who["sub"]
    assert call(api, "GET", "/tools", who=who)[0] == 401


def test_wrong_method_is_405(rig) -> None:
    api, _, _ = rig
    status, body = call(api, "DELETE", "/tools/get_balance")
    assert status == 405 and "GET" in body["message"]


@pytest.mark.parametrize("who,path,method,want", [
    (VIEWER, "/tools", "GET", 200),
    (VIEWER, "/tools/get_balance", "PUT", 403),
    (VIEWER, "/masking", "PUT", 403),
    (VIEWER, "/restore", "POST", 403),
    (VIEWER, "/users", "GET", 403),
    (EDITOR, "/users", "GET", 403),
    (EDITOR, "/users/u-1", "DELETE", 403),
    (claims(None), "/tools", "GET", 403),
    (claims(["Admin"]), "/tools", "GET", 403),
    (claims("[admin-viewer admin-editor]"), "/masking", "GET", 200),
])
def test_group_checks(rig, who, path, method, want) -> None:
    api, store, cognito = rig
    status, _ = call(api, method, path, {"rev": 0, "min_level": 1}, who=who)
    assert status == want
    if want == 403:
        assert not cognito.calls and not all_audits(store)


def test_parse_groups_formats() -> None:
    assert parse_groups("admin") == {"admin"}
    assert parse_groups("admin,admin-editor") == {"admin", "admin-editor"}
    assert parse_groups("[admin admin-viewer]") == {"admin", "admin-viewer"}
    assert parse_groups(["admin"]) == {"admin"}
    assert parse_groups(None) == frozenset() and parse_groups(3) == frozenset()


def test_api_prefix_and_resource_templates(rig) -> None:
    api, _, _ = rig
    assert call(api, "GET", "/api/tools/get_balance")[0] == 200
    ev = event("GET", "/api/tools/get_balance")
    ev["resource"], ev["pathParameters"] = "/tools/{name}", {"name": "get_balance"}
    resp = api.handle(ev)
    assert resp["statusCode"] == 200 and json.loads(resp["body"])["tool"]["name"] == "get_balance"
    ev["resource"] = "/{proxy+}"
    assert api.handle(ev)["statusCode"] == 200


def test_responses_are_json_and_not_cached(rig) -> None:
    api, _, _ = rig
    resp = api.handle(event("GET", "/tools"))
    assert resp["headers"]["Content-Type"] == "application/json"
    assert resp["headers"]["Cache-Control"] == "no-store"


def test_unexpected_errors_are_500_without_internals(rig, caplog) -> None:
    api, store, _ = rig

    def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    store.get = boom  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR):
        status, body = call(api, "GET", "/tools")
    assert (status, body) == (500, {"error": "internal"})
    assert "secret internal detail" not in caplog.text


@pytest.mark.parametrize("raw,status,error", [
    ("not json", 400, "bad_json"),
    ("[1, 2]", 400, "bad_request"),
    ('{"rev": NaN}', 400, "bad_json"),
    ("x" * (64 * 1024 + 1), 413, "too_large"),
])
def test_bad_bodies(rig, raw, status, error) -> None:
    api, _, _ = rig
    got, body = call(api, "PUT", "/masking", raw_body=raw)
    assert (got, body["error"]) == (status, error)


def test_missing_body_and_base64_body(rig) -> None:
    api, _, _ = rig
    assert call(api, "PUT", "/masking")[0] == 400
    ev = event("PUT", "/masking", raw_body=base64.b64encode(
        json.dumps({"rev": 0, "masking": {"after_ms": 1500}}).encode()).decode())
    ev["isBase64Encoded"] = True
    assert api.handle(ev)["statusCode"] == 200


def test_not_seeded_is_503() -> None:
    api, _, _, _ = make_api(store=MemoryStore())
    assert call(api, "GET", "/tools")[0] == 503


# --------------------------------------------------------------------------- tools


def test_list_tools_shows_every_policy_tool(rig) -> None:
    api, store, _ = rig
    status, body = call(api, "GET", "/tools", who=VIEWER)
    assert status == 200
    names = [t["name"] for t in body["tools"]]
    assert names == store.get("policy", {"pk": "meta"})["tools"]
    bal = next(t for t in body["tools"] if t["name"] == "get_balance")
    seeded = store.get("policy", {"pk": "masking"})["tool_after_ms"]  # the seed default
    assert bal["mask"]["after_ms"] == {"value": seeded, "source": "default"}
    assert bal["mask"]["pool"] and bal["problems"] == []
    assert body["takes_effect_s"] == 60


def test_get_tool_and_unknown_tool(rig) -> None:
    api, _, _ = rig
    status, body = call(api, "GET", "/tools/get_balance", who=VIEWER)
    assert status == 200
    assert body["tool"]["row"]["template_key"] == "balance.read"
    assert body["tool"]["placeholders"][0]["name"] == "account"
    assert call(api, "GET", "/tools/nope")[0] == 404
    assert call(api, "GET", "/tools/Bad%20Name")[0] == 404


def test_put_tool_min_level_is_a_security_change(rig) -> None:
    api, store, _ = rig
    status, body = call(api, "PUT", "/tools/get_balance", {"rev": 0, "min_level": 2}, who=EDITOR)
    assert status == 200 and body["changed"] and body["security_change"]
    row = store.get("policy", {"pk": "tool#get_balance"})
    assert row["min_level"] == 2 and row["rev"] == 1
    [entry] = audits(store, "policy#tool#get_balance")
    assert entry["changes"] == [{"field": "min_level", "old": 1, "new": 2}]
    assert entry["security_change"] is True and entry["rev"] == 1
    assert entry["actor"] == {"sub": "sub-actor", "email": "actor@example.com"}
    assert Policy.from_store(store).min_level("get_balance") == 2


def test_audit_ttl_is_the_retention(rig) -> None:
    api, store, _, clock = make_api()
    call(api, "PUT", "/tools/get_balance", {"rev": 0, "min_level": 2})
    [entry] = audits(store, "policy#tool#get_balance")
    assert abs(entry["expires_at"] - (clock.t + 90 * 86400)) < 2


def test_put_tool_confirm_change_and_removal(rig) -> None:
    api, store, _ = rig
    status, body = call(api, "PUT", "/tools/list_recent_transactions",
                        {"rev": 0, "confirm": "deterministic"})
    assert status == 200 and body["security_change"]
    status, body = call(api, "PUT", "/tools/list_recent_transactions", {"rev": 1, "confirm": None})
    assert status == 200
    assert "confirm" not in store.get("policy", {"pk": "tool#list_recent_transactions"})
    assert body["tool"]["confirm_source"] == "default"


def test_put_tool_mask_delay_is_not_a_security_change(rig) -> None:
    api, store, _ = rig
    mask = {"phrase_key": "mask.get_balance", "after_ms": 900}
    status, body = call(api, "PUT", "/tools/get_balance", {"rev": 0, "mask": mask})
    assert status == 200 and not body["security_change"]
    assert body["tool"]["mask"]["after_ms"] == {"value": 900, "source": "tool"}
    [entry] = audits(store, "policy#tool#get_balance")
    assert entry["security_change"] is False
    assert Policy.from_store(store).tool_mask("get_balance").after_ms == 900


def test_put_tool_mask_off(rig) -> None:
    api, store, _ = rig
    status, body = call(api, "PUT", "/tools/get_balance", {"rev": 0, "mask": None})
    assert status == 200 and body["tool"]["mask"] is None
    assert "mask" not in store.get("policy", {"pk": "tool#get_balance"})


def test_put_tool_no_change_writes_nothing(rig) -> None:
    api, store, _ = rig
    status, body = call(api, "PUT", "/tools/get_balance", {"rev": 0, "min_level": 1})
    assert status == 200 and body["changed"] is False
    assert not audits(store, "policy#tool#get_balance")
    assert "rev" not in store.get("policy", {"pk": "tool#get_balance"})


def test_put_tool_stale_rev_is_409_with_the_current_item(rig) -> None:
    api, store, _ = rig
    assert call(api, "PUT", "/tools/get_balance", {"rev": 0, "min_level": 2})[0] == 200
    status, body = call(api, "PUT", "/tools/get_balance", {"rev": 0, "min_level": 3})
    assert status == 409 and body["error"] == "conflict"
    assert body["current"]["min_level"] == 2 and body["current"]["rev"] == 1
    assert store.get("policy", {"pk": "tool#get_balance"})["min_level"] == 2
    assert len(audits(store, "policy#tool#get_balance")) == 1


@pytest.mark.parametrize("body,field", [
    ({"rev": 0, "min_level": 9}, "tools.get_balance.min_level"),
    ({"rev": 0, "min_level": "1"}, "tools.get_balance.min_level"),
    ({"rev": 0, "min_level": True}, "tools.get_balance.min_level"),
    ({"rev": 0, "min_level": None}, "tools.get_balance.min_level"),
    ({"rev": 0, "confirm": "deterministic"}, "tools.get_balance.confirm"),  # kind deterministic
    ({"rev": 0, "confirm": "maybe"}, "tools.get_balance.confirm"),
    ({"rev": 0, "mask": {"phrase_key": "mask.get_balance", "after_ms": 10001}},
     "tools.get_balance.mask.after_ms"),
    ({"rev": 0, "mask": {"phrase_key": "tool.get_balance"}}, "tools.get_balance.mask.phrase_key"),
    ({"rev": 0, "mask": {"phrase_key": "mask.get_balance", "x": 1}}, "tools.get_balance.mask.x"),
    ({"rev": 0, "mask": "mask.get_balance"}, "tools.get_balance.mask"),
    ({"rev": 0, "mask": {"phrase_key": "mask.nothing_here"}}, "tools.get_balance.mask.phrase_key"),
    ({"rev": 0, "kind": "handoff"}, "kind"),
    ({"min_level": 1}, "rev"),
    ({"rev": -1, "min_level": 1}, "rev"),
    ({"rev": "0", "min_level": 1}, "rev"),
])
def test_put_tool_validation(rig, body, field) -> None:
    api, store, _ = rig
    before = store.get("policy", {"pk": "tool#get_balance"})
    status, out = call(api, "PUT", "/tools/get_balance", body)
    assert status == 400 and out["field"] == field, out
    assert store.get("policy", {"pk": "tool#get_balance"}) == before
    assert not audits(store, "policy#tool#get_balance")


def test_put_tool_confirm_on_a_gateway_tool_is_refused(rig) -> None:
    api, _, _ = rig
    status, body = call(api, "PUT", "/tools/faq_lookup", {"rev": 0, "confirm": "none"})
    assert status == 400 and body["field"] == "tools.faq_lookup.confirm"


def test_put_tool_nothing_to_update_and_unknown_tool(rig) -> None:
    api, _, _ = rig
    assert call(api, "PUT", "/tools/get_balance", {"rev": 0})[0] == 400
    assert call(api, "PUT", "/tools/nope", {"rev": 0, "min_level": 1})[0] == 404


# --------------------------------------------------------------------------- masking


def test_get_masking(rig) -> None:
    api, store, _ = rig
    status, body = call(api, "GET", "/masking", who=VIEWER)
    assert status == 200 and body["rev"] == 0 and body["problems"] == []
    seeded = store.get("policy", {"pk": "masking"})["after_ms"]  # the seed default
    assert body["effective"]["after_ms"] == seeded and len(body["ack_pool"]) == 5


def test_put_masking_normalizes_and_the_bridge_reads_it_without_errors(rig, caplog) -> None:
    api, store, _ = rig
    new = {"after_ms": 1500, "tool_after_ms": 300, "handoff_wait_ms": 2500, "end_mode": "wait",
           "chars_per_s": 15.5, "silence_timeout_ms": 0}
    status, body = call(api, "PUT", "/masking", {"rev": 0, "masking": new}, who=EDITOR)
    assert status == 200 and body["changed"] and body["rev"] == 1
    item = store.get("policy", {"pk": "masking"})
    assert item["chars_per_s"] == "15.5" and item["rev"] == 1
    assert "silence_action" not in item  # omitted = the default
    [entry] = audits(store, "policy#masking")
    assert {c["field"] for c in entry["changes"]} >= {"after_ms", "end_mode", "silence_action"}
    with caplog.at_level(logging.ERROR):
        cfg = Policy.from_store(store).masking
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert (cfg.after_ms, cfg.chars_per_s, cfg.end_mode) == (1500, 15.5, "wait")


@pytest.mark.parametrize("masking,field", [
    ({"after_ms": -1}, "masking.after_ms"),
    ({"after_ms": 1.0}, "masking.after_ms"),
    ({"end_mode": "later"}, "masking.end_mode"),
    ({"chars_per_s": "nan"}, "masking.chars_per_s"),
    ({"bogus": 1}, "masking.bogus"),
    ("fast", "masking"),
])
def test_put_masking_validation(rig, masking, field) -> None:
    api, store, _ = rig
    before = store.get("policy", {"pk": "masking"})
    status, body = call(api, "PUT", "/masking", {"rev": 0, "masking": masking})
    assert status == 400 and body["field"] == field
    assert store.get("policy", {"pk": "masking"}) == before and not audits(store, "policy#masking")


def test_put_masking_conflict_and_missing_field(rig) -> None:
    api, store, _ = rig
    assert call(api, "PUT", "/masking", {"rev": 0})[1]["field"] == "masking"
    assert call(api, "PUT", "/masking", {"rev": 0, "masking": {"after_ms": 1000}})[0] == 200
    status, body = call(api, "PUT", "/masking", {"rev": 0, "masking": {"after_ms": 1200}})
    assert status == 409 and body["current"]["masking"] == {"after_ms": 1000}
    assert store.get("policy", {"pk": "masking"})["after_ms"] == 1000


# --------------------------------------------------------------------------- prompts


def test_list_prompts(rig) -> None:
    api, _, _ = rig
    status, body = call(api, "GET", "/prompts", query={"prefix": "mask."}, who=VIEWER)
    assert status == 200 and body["languages"] == ["en-US", "es-US"]
    keys = {p["key"]: p for p in body["prompts"]}
    assert "mask.get_balance.1" in keys and keys["mask.get_balance.1"]["owner_tool"] == "get_balance"
    assert all(k.startswith("mask.") for k in keys)
    status, body = call(api, "GET", "/prompts", query={"prefix": "arg."})
    assert status == 200 and {p["key"] for p in body["prompts"]} >= {"arg.account.checking"}
    for bad in ({}, {"prefix": "t"}, {"prefix": "Tool."}, {"prefix": "_meta"}, {"prefix": "mask.{"},
                {"prefix": "-agent."}, {"prefix": "a" * 129}):
        assert call(api, "GET", "/prompts", query=bad or None)[0] == 400


def test_list_prompts_gives_each_keys_languages_and_a_single_language_key_saves(rig) -> None:
    """``language.select`` is written in es-US only; the list says so, and a
    save with exactly the listed languages is accepted."""
    api, store, _ = rig
    _, body = call(api, "GET", "/prompts", query={"prefix": "language."}, who=VIEWER)
    views = {p["key"]: p for p in body["prompts"]}
    assert views["language.select"]["languages"] == ["es-US"]
    _, body = call(api, "GET", "/prompts", query={"prefix": "greeting."}, who=VIEWER)
    assert all(p["languages"] == ["en-US", "es-US"] for p in body["prompts"])
    sel = views["language.select"]
    texts = dict.fromkeys(sel["languages"], "Para español, oprima el dos.")
    status, out = call(api, "PUT", "/prompts/language.select", {"rev": sel["rev"], "texts": texts},
                       who=EDITOR)
    assert status == 200 and out["changed"], out


def test_list_prompts_shows_editable_and_the_protected_reason(rig) -> None:
    api, _, _ = rig
    status, body = call(api, "GET", "/prompts", query={"prefix": "greeting."}, who=VIEWER)
    assert status == 200 and body["prompts"]
    assert all(p["editable"] is True and "reason" not in p for p in body["prompts"])
    _, body = call(api, "GET", "/prompts", query={"prefix": "confirm."})
    confirm = {p["key"]: p for p in body["prompts"]}
    assert {"confirm.make_payment", "confirm.transfer_funds.done", "confirm.cancelled"} <= set(confirm)
    assert all(p["editable"] is False and p["reason"] == "protected read-back" for p in confirm.values())
    _, body = call(api, "GET", "/prompts", query={"prefix": "clarify."})
    clarify = {p["key"]: p for p in body["prompts"]}
    assert clarify["clarify.yes_no"]["editable"] is False
    assert clarify["clarify.yes_no"]["reason"] == "protected read-back"
    assert clarify["clarify.digits"]["editable"] is True
    _, body = call(api, "GET", "/prompts", query={"prefix": "agent."})
    assert body["prompts"] and all(p["editable"] is True and "reason" not in p for p in body["prompts"])
    _, body = call(api, "GET", "/prompts", query={"prefix": "mask."})
    assert all(p["editable"] is True for p in body["prompts"])


@pytest.mark.parametrize("key", [
    "confirm.make_payment", "confirm.make_payment.done", "confirm.transfer_funds",
    "confirm.transfer_funds.done", "confirm.cancelled", "confirm.failed",
    "confirm.insufficient_funds", "clarify.yes_no", "confirm.not_yet_a_key",
])
def test_put_protected_prompt_is_403_with_no_write_and_no_audit(rig, key) -> None:
    api, store, _ = rig
    before = {lang: store.get("catalog", {"key": key, "lang": lang}) for lang in ("en-US", "es-US")}
    cmeta = store.get("catalog", {"key": "_meta", "lang": "_"})
    texts = {"en-US": "Okay.", "es-US": "Bueno."}
    for who in (EDITOR, ADMIN):
        status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts}, who=who)
        assert status == 403 and body["error"] == "protected" and body["reason"], body
    assert {lang: store.get("catalog", {"key": key, "lang": lang}) for lang in before} == before
    assert store.get("catalog", {"key": "_meta", "lang": "_"}) == cmeta
    assert not all_audits(store)


def test_restore_of_a_protected_prompt_is_403(rig) -> None:
    api, store, _ = rig
    store.put("admin_audit", {"pk": "catalog#confirm.make_payment", "sk": "2026-01-01T00:00:00.000000Z#x",
                              "target": {"kind": "prompt", "id": "confirm.make_payment"},
                              "changes": [{"field": "text[en-US]", "old": "Pay it?", "new": "x"}]})
    before = store.get("catalog", {"key": "confirm.make_payment", "lang": "en-US"})
    status, body = call(api, "POST", "/restore", {"audit_pk": "catalog#confirm.make_payment",
                                                  "audit_sk": "2026-01-01T00:00:00.000000Z#x", "rev": 0})
    assert status == 403 and body["error"] == "protected"
    assert store.get("catalog", {"key": "confirm.make_payment", "lang": "en-US"}) == before
    assert len(all_audits(store)) == 1


def test_put_a_caller_facing_prompt(rig) -> None:
    api, store, _ = rig
    texts = {"en-US": "Hi {firstName}, what can I do for you?", "es-US": "Hola {firstName}, ¿en qué le ayudo?"}
    status, body = call(api, "PUT", "/prompts/greeting.personal", {"rev": 0, "texts": texts}, who=EDITOR)
    assert status == 200 and body["changed"] and body["prompt"]["texts"] == texts
    [entry] = audits(store, "catalog#greeting.personal")
    assert entry["action"] == "update" and entry["rev"] == 1
    assert PromptCatalog.from_store(store).template("greeting.personal", "es-US") == texts["es-US"]
    status, body = call(api, "PUT", "/prompts/greeting.personal", {"rev": 0, "texts": texts})
    assert status == 409 and body["current"]["rev"] == 1
    assert len(audits(store, "catalog#greeting.personal")) == 1
    old = body["current"]["texts"]
    status, body = call(api, "POST", "/restore", {"audit_pk": entry["pk"], "audit_sk": entry["sk"], "rev": 1})
    assert status == 200 and body["prompt"]["texts"] != old and body["prompt"]["rev"] == 2
    assert _last(store, "catalog#greeting.personal")["action"] == "restore"


def test_put_a_single_language_prompt(rig) -> None:
    api, store, _ = rig
    status, body = call(api, "PUT", "/prompts/language.select",
                        {"rev": 0, "texts": {"es-US": "Para español, marque dos."}})
    assert status == 200 and body["prompt"]["texts"] == {"es-US": "Para español, marque dos."}
    assert store.get("catalog", {"key": "language.select", "lang": "en-US"}) is None
    status, body = call(api, "PUT", "/prompts/language.select",
                        {"rev": 1, "texts": {"es-US": "Dos.", "en-US": "Two."}})
    assert status == 400 and body["field"] == "texts[en-US]"


@pytest.mark.parametrize("key,texts,error,field", [
    ("greeting.generic", {"en-US": "  ", "es-US": "Hola."}, "invalid", "texts[en-US]"),
    ("greeting.generic", {"en-US": "", "es-US": "Hola."}, "invalid", "texts[en-US]"),
    ("greeting.generic", {"en-US": "Hi.", "es-US": "x" * 601}, "invalid", "texts[es-US]"),
    ("greeting.generic", {"en-US": "<speak>Hi.</speak>", "es-US": "Hola."}, "invalid", "texts[en-US]"),
    ("greeting.generic", {"en-US": "Hi.", "es-US": "Hola <break time='1s'/>"}, "invalid", "texts[es-US]"),
    ("greeting.generic", {"en-US": "Hi > there.", "es-US": "Hola."}, "invalid", "texts[en-US]"),
    ("greeting.generic", {"en-US": "Hi {firstName}.", "es-US": "Hola."}, "placeholders", "texts[en-US]"),
    ("greeting.generic", {"en-US": "Hi {.", "es-US": "Hola."}, "placeholders", "texts[en-US]"),
    ("greeting.personal", {"en-US": "Thanks. How can I help?", "es-US": "Gracias, {firstName}."},
     "placeholders", "texts[en-US]"),
    ("greeting.personal", {"en-US": "Thanks, {firstName}.", "es-US": "Gracias, {first_name}."},
     "placeholders", "texts[es-US]"),
    ("greeting.personal", {"en-US": "Thanks, {firstName} {lastName}.", "es-US": "Gracias, {firstName}."},
     "placeholders", "texts[en-US]"),
    ("balance.read", {"en-US": "Your balance.", "es-US": "Su saldo."}, "placeholders", "texts[en-US]"),
    ("greeting.nope", {"en-US": "a", "es-US": "b"}, "invalid", "key"),
    ("_meta", {"en-US": "a", "es-US": "b"}, "invalid", "key"),
    ("Greeting.generic", {"en-US": "a", "es-US": "b"}, "invalid", "key"),
])
def test_put_prompt_generic_rules(rig, key, texts, error, field) -> None:
    api, store, _ = rig
    before = {lang: store.get("catalog", {"key": key, "lang": lang}) for lang in ("en-US", "es-US")}
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts})
    assert status == 400 and body["error"] == error and body["field"] == field, body
    assert {lang: store.get("catalog", {"key": key, "lang": lang}) for lang in before} == before
    assert not all_audits(store)


def _agent_texts(store: MemoryStore, key: str) -> dict[str, str]:
    items = {lang: store.get("catalog", {"key": key, "lang": lang}) for lang in ("en-US", "es-US")}
    return {lang: it["text"] for lang, it in items.items() if it}


def _agent_keys(store: MemoryStore) -> list[str]:
    cmeta = store.get("catalog", {"key": "_meta", "lang": "_"}) or {}
    return sorted(k for k in cmeta.get("keys", []) if k.startswith(config_routes.AGENT_PREFIX))


def test_put_a_model_instruction_prompt_is_audited_as_a_security_change(rig) -> None:
    """agent.* is editable under its own rules; each edit (and its restore) is
    audited with security_change true, and the audit list shows it."""
    api, store, _ = rig
    key = "agent.call_started"
    texts = {lang: t + " <be brief>" for lang, t in _agent_texts(store, key).items()}
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts}, who=EDITOR)
    assert status == 200 and body["changed"], body
    [entry] = audits(store, f"catalog#{key}")
    assert entry["action"] == "update" and entry["rev"] == 1
    assert entry["security_change"] is True
    _, listed = call(api, "GET", "/audit", query={"item": f"catalog#{key}"})
    assert listed["entries"][0]["security_change"] is True
    status, body = call(api, "POST", "/restore", {"audit_pk": entry["pk"], "audit_sk": entry["sk"], "rev": 1})
    assert status == 200, body
    assert _last(store, f"catalog#{key}")["action"] == "restore"
    assert _last(store, f"catalog#{key}")["security_change"] is True


def test_a_spoken_prompt_edit_is_not_a_security_change(rig) -> None:
    api, store, _ = rig
    texts = {"en-US": "Hi there.", "es-US": "Hola."}
    status, body = call(api, "PUT", "/prompts/greeting.generic", {"rev": 0, "texts": texts})
    assert status == 200 and body["changed"], body
    [entry] = audits(store, "catalog#greeting.generic")
    assert entry["security_change"] is False


def test_every_shipped_agent_prompt_passes_as_is(rig) -> None:
    """The shipped system prompts (about 4,000-4,800 characters, with markup) can be
    written back unchanged: a no-op resubmit is accepted, writes nothing and audits nothing."""
    api, store, _ = rig
    keys = _agent_keys(store)
    assert any(k.startswith("agent.system.") for k in keys)
    assert any(len(t) > config_routes.MAX_PROMPT_CHARS and "<" in t
               for k in keys for t in _agent_texts(store, k).values())
    for key in keys:
        texts = _agent_texts(store, key)
        status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts}, who=EDITOR)
        assert status == 200 and body["changed"] is False, (key, body)
    assert not all_audits(store)


@pytest.mark.parametrize("pad_to,ok", [(16_000, True), (16_001, False)])
def test_agent_prompt_length_limit(rig, pad_to, ok) -> None:
    api, store, _ = rig
    key = "agent.system.haiku-5-5"
    texts = {lang: t + "a" * (pad_to - len(t)) for lang, t in _agent_texts(store, key).items()}
    assert config_routes.MAX_AGENT_PROMPT_CHARS == 16_000
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts}, who=EDITOR)
    if ok:
        assert status == 200 and body["changed"], body
        assert len(audits(store, f"catalog#{key}")) == 1
    else:
        assert status == 400 and body["error"] == "invalid" and "16000" in body["message"], body
        assert not all_audits(store)


@pytest.mark.parametrize("key,edit,error", [
    # A placeholder dropped: the caller-context frame has them since agent prompt caching.
    ("agent.caller_context", lambda t: t.replace("{memory}", "memory"), "placeholders"),
    ("agent.system.haiku-5-5", lambda t: t + " {lastName}", "placeholders"),
    ("agent.system.haiku-5-5", lambda t: t + " {", "placeholders"),
    ("agent.system.haiku-5-5", lambda t: " \n ", "invalid"),
], ids=["dropped", "added", "stray-brace", "blank"])
def test_agent_prompt_keeps_the_other_rules(rig, key, edit, error) -> None:
    api, store, _ = rig
    texts = {lang: edit(t) for lang, t in _agent_texts(store, key).items()}
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts}, who=EDITOR)
    assert status == 400 and body["error"] == error and body["field"].startswith("texts["), body
    assert not all_audits(store)


_NO_PLACEHOLDERS = "must have no placeholders"
_CALLER_SET = "must have exactly {firstName}, {memory}, {profile}"


@pytest.mark.parametrize("key,edit,message", [
    ("agent.system.haiku-5-5", lambda t: t + "\nCaller: {firstName}.", _NO_PLACEHOLDERS),
    ("agent.caller_context", lambda t: t.replace("{profile}", "the profile"), _CALLER_SET),
    ("agent.caller_context", lambda t: t + " {lastName}", _CALLER_SET),
    ("agent.caller_context_unverified", lambda t: t + " {memory}", _NO_PLACEHOLDERS),
    ("agent.web_activity", lambda t: t + " {firstName}", _NO_PLACEHOLDERS),
], ids=["system-added", "caller-dropped", "caller-added", "unverified-added", "web-added"])
def test_agent_prompt_placeholder_allow_list(rig, key, edit, message) -> None:
    """Agent prompt caching: an agent.* text has exactly its key's allowed set,
    and the 400 names that set; nothing is written or audited."""
    api, store, _ = rig
    before = _agent_texts(store, key)
    texts = {lang: edit(t) for lang, t in before.items()}
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts}, who=EDITOR)
    assert status == 400 and body["error"] == "placeholders" and body["field"].startswith("texts["), body
    assert body["message"] == message, body
    assert _agent_texts(store, key) == before
    assert not all_audits(store)


def test_a_new_agent_key_with_a_placeholder_is_refused(rig) -> None:
    """For a new key: agent.system.opus-5-5 with {firstName} is refused and nothing is
    written, the catalog index included. The API adds no agent.* key at all (only mask.*, arg.*
    and option.* may be added), and the text check refuses it on its own as well."""
    api, store, _ = rig
    key = "agent.system.opus-5-5"
    meta_before = store.get("catalog", {"key": "_meta", "lang": "_"})
    texts = {"en-US": "Rules.\nCaller: {firstName}.", "es-US": "Reglas.\nCliente: {firstName}."}
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts}, who=EDITOR)
    assert status == 400, body
    assert _agent_texts(store, key) == {}
    assert store.get("catalog", {"key": "_meta", "lang": "_"}) == meta_before
    assert not all_audits(store)
    with pytest.raises(ApiError) as err:
        config_routes._check_prompt_text(key, texts["en-US"], "texts[en-US]", None)
    assert err.value.status == 400 and err.value.error == "placeholders"
    assert err.value.message == _NO_PLACEHOLDERS


def test_restore_of_a_pre_feature_agent_revision_is_refused(rig) -> None:
    """A catalog still on the pre-feature text (caller line in agent.system.*) can be edited
    to a text with no placeholders, but restoring that old revision is refused with a clear
    message; the stored text stays the new one and only the edit is audited."""
    api, store, _ = rig
    key = "agent.system.haiku-5-5"
    shipped = _agent_texts(store, key)
    old_line = ("\nCaller: {firstName}. Verified profile: {profile}. "
                "Recent interactions from other channels: {memory}.")
    for lang, text in shipped.items():  # the catalog as it was before the reseed
        item = store.get("catalog", {"key": key, "lang": lang})
        store.put("catalog", {**item, "text": text + old_line})
    # Resubmitting the pre-feature text unchanged is refused (replaces the same-set rule).
    stale = {lang: t + old_line for lang, t in shipped.items()}
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": stale}, who=EDITOR)
    assert status == 400 and body["error"] == "placeholders", body
    assert body["message"] == _NO_PLACEHOLDERS, body
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": shipped}, who=EDITOR)
    assert status == 200 and body["changed"], body
    [entry] = audits(store, f"catalog#{key}")
    assert entry["security_change"] is True
    status, body = call(api, "POST", "/restore", {"audit_pk": entry["pk"], "audit_sk": entry["sk"], "rev": 1})
    assert status == 400 and body["error"] == "placeholders", body
    assert body["message"] == _NO_PLACEHOLDERS, body
    assert _agent_texts(store, key) == shipped
    assert len(audits(store, f"catalog#{key}")) == 1


@pytest.mark.parametrize("text", ["Hello <there>.", "x" * 601])
def test_spoken_prompt_keeps_its_limits(rig, text) -> None:
    api, store, _ = rig
    status, body = call(api, "PUT", "/prompts/greeting.generic",
                        {"rev": 0, "texts": {"en-US": text, "es-US": "Hola."}}, who=EDITOR)
    assert status == 400 and body["error"] == "invalid" and body["field"] == "texts[en-US]", body
    assert not all_audits(store)


_RULES = json.loads((Path(__file__).parents[2] / "fixtures" / "prompt_text_rules.json")
                    .read_text(encoding="utf-8"))["cases"]


def _fixture_text(case: dict[str, Any]) -> str:
    text: str = case["text"]
    return text + case.get("pad_char", "a") * (case.get("pad_to", len(text)) - len(text))


@pytest.mark.parametrize("case", _RULES, ids=lambda c: c["name"])
def test_prompt_text_rules_match_the_shared_table(case) -> None:
    """Parity with the console: src/admin_console/tests/validate.test.js asserts the same table."""
    text = _fixture_text(case)
    if "pad_to" in case:
        assert len(text) == case["pad_to"]
    cur = case["current"]
    expected = None if cur is None else frozenset(mc.placeholder_names(cur))
    try:
        config_routes._check_prompt_text(case["key"], text, "texts[en-US]", expected)
        accepted = True
    except ApiError:
        accepted = False
    assert accepted is case["accept"], case["name"]


def test_put_mask_phrase_may_gain_or_lose_an_allowed_placeholder(rig) -> None:
    """mask.* is exempt from the same-placeholder rule; validate_phrase decides."""
    api, store, _ = rig
    gain = {"en-US": "Getting your {account} balance.", "es-US": "Un momento."}
    status, body = call(api, "PUT", "/prompts/mask.get_balance", {"rev": 0, "texts": gain}, who=EDITOR)
    assert status == 200 and body["prompt"]["texts"] == gain, body
    lose = {"en-US": "Getting your balance.", "es-US": "Un momento."}
    status, body = call(api, "PUT", "/prompts/mask.get_balance", {"rev": 1, "texts": lose}, who=EDITOR)
    assert status == 200 and body["prompt"]["texts"] == lose, body
    assert len(audits(store, "catalog#mask.get_balance")) == 2
    for texts in ({"en-US": "Your {amount} balance.", "es-US": "Un momento."},
                  {"en-US": "Your <b>balance</b>.", "es-US": "Un momento."},
                  {"en-US": "x" * 601, "es-US": "Un momento."}):
        status, body = call(api, "PUT", "/prompts/mask.get_balance", {"rev": 2, "texts": texts})
        assert status == 400 and body["field"] == "texts[en-US]", body
    assert len(audits(store, "catalog#mask.get_balance")) == 2


def test_put_prompt_at_the_length_limit(rig) -> None:
    api, _, _ = rig
    texts = {"en-US": "x" * 600, "es-US": "Hola."}
    assert call(api, "PUT", "/prompts/greeting.generic", {"rev": 0, "texts": texts})[0] == 200


def test_put_prompt_with_a_placeholder(rig) -> None:
    api, store, _ = rig
    texts = {"en-US": "Checking your {account} balance.", "es-US": "Revisando su saldo de {account}."}
    status, body = call(api, "PUT", "/prompts/mask.get_balance.1", {"rev": 0, "texts": texts}, who=EDITOR)
    assert status == 200 and body["prompt"]["rev"] == 1 and body["prompt"]["texts"] == texts
    for lang in texts:
        assert store.get("catalog", {"key": "mask.get_balance.1", "lang": lang})["rev"] == 1
    [entry] = audits(store, "catalog#mask.get_balance.1")
    assert {c["field"] for c in entry["changes"]} == {"text[en-US]", "text[es-US]"}
    cat = PromptCatalog.from_store(store)
    assert cat.template("mask.get_balance.1", "en-US") == texts["en-US"]


def test_put_new_pool_key_joins_the_catalog_index(rig) -> None:
    api, store, _ = rig
    texts = {"en-US": "Looking up your balance.", "es-US": "Buscando su saldo."}
    status, _ = call(api, "PUT", "/prompts/mask.get_balance.9", {"rev": 0, "texts": texts})
    assert status == 200
    meta = store.get("catalog", {"key": "_meta", "lang": "_"})
    assert "mask.get_balance.9" in meta["keys"] and meta["rev"] == 1
    assert "mask.get_balance.9" in PromptCatalog.from_store(store).keys
    assert len(audits(store, "catalog#mask.get_balance.9")) == 1


@pytest.mark.parametrize("key,texts,field", [
    ("mask.get_balance.1", {"en-US": "Your {customer} balance.", "es-US": "Su saldo."}, "texts[en-US]"),
    ("mask.get_balance.1", {"en-US": "Your balance }.", "es-US": "Su saldo."}, "texts[en-US]"),
    ("mask.get_balance.1", {"en-US": "Your balance."}, "texts[es-US]"),
    ("mask.get_balance.1", {"en-US": "a", "es-US": "b", "fr-FR": "c"}, "texts[fr-FR]"),
    ("mask.get_balance.1", {"en-US": "   ", "es-US": "b"}, "texts[en-US]"),
    ("mask.get_balance.1", {"en-US": 5, "es-US": "b"}, "texts[en-US]"),
    ("mask.generic.1", {"en-US": "One {account} moment.", "es-US": "Un momento."}, "texts[en-US]"),
    ("mask.unowned", {"en-US": "a", "es-US": "b"}, "key"),
    ("arg.account.checking", {"en-US": "{checking}", "es-US": "cheques"}, "texts[en-US]"),
    ("arg.account.checking", {"en-US": "", "es-US": "cheques"}, "texts[en-US]"),
    ("arg.colour.blue", {"en-US": "blue", "es-US": "azul"}, "key"),
    ("tool.unknown", {"en-US": "a", "es-US": "b"}, "key"),
    ("mask.{x}", {"en-US": "a", "es-US": "b"}, "key"),
])
def test_put_prompt_validation(rig, key, texts, field) -> None:
    api, store, _ = rig
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts})
    assert status == 400 and body["field"] == field, body
    assert not audits(store, f"catalog#{key}")


def test_put_prompt_arg_word_and_conflict(rig) -> None:
    api, store, _ = rig
    texts = {"en-US": "everyday", "es-US": "cheques"}
    assert call(api, "PUT", "/prompts/arg.account.checking", {"rev": 0, "texts": texts})[0] == 200
    status, body = call(api, "PUT", "/prompts/arg.account.checking", {"rev": 0, "texts": texts})
    assert status == 409 and body["current"]["rev"] == 1
    assert len(audits(store, "catalog#arg.account.checking")) == 1


def test_put_prompt_drops_a_stale_recording(rig) -> None:
    api, store, _ = rig
    item = store.get("catalog", {"key": "mask.get_balance", "lang": "en-US"})
    store.put("catalog", {**item, "audio_url": "https://example.com/a.mp3"})
    texts = {"en-US": "Getting your balance.", "es-US": "Consultando su saldo ahora."}
    assert call(api, "PUT", "/prompts/mask.get_balance", {"rev": 0, "texts": texts})[0] == 200
    assert store.get("catalog", {"key": "mask.get_balance", "lang": "en-US"})["audio_url"] is None
    [entry] = audits(store, "catalog#mask.get_balance")
    assert {"field": "audio_url[en-US]", "old": "https://example.com/a.mp3", "new": None} in entry["changes"]


# --------------------------------------------------------------------------- placeholders


def test_placeholders(rig) -> None:
    api, store, _ = rig
    status, body = call(api, "GET", "/placeholders/get_balance", who=VIEWER)
    assert status == 200
    [account] = body["placeholders"]
    assert account["placeholder"] == "{account}" and account["missing"] == []
    assert {v["value"] for v in account["values"]} == {"checking", "savings"}
    store.delete("catalog", {"key": "arg.account.savings", "lang": "es-US"})
    _, body = call(api, "GET", "/placeholders/get_balance")
    assert body["placeholders"][0]["missing"] == ["arg.account.savings[es-US]"]
    assert call(api, "GET", "/placeholders/faq_lookup")[1]["placeholders"] == []
    assert call(api, "GET", "/placeholders/nope")[0] == 404


# --------------------------------------------------------------------------- audit and restore


def test_audit_listing(rig) -> None:
    api, _, _ = rig
    for rev, level in enumerate((2, 3, 1)):
        assert call(api, "PUT", "/tools/get_balance", {"rev": rev, "min_level": level})[0] == 200
    status, body = call(api, "GET", "/audit", query={"item": "policy#tool#get_balance"}, who=VIEWER)
    assert status == 200 and [e["rev"] for e in body["entries"]] == [3, 2, 1]
    _, body = call(api, "GET", "/audit", query={"item": "policy#tool#get_balance", "limit": "1"})
    assert [e["rev"] for e in body["entries"]] == [3]
    for q in ({}, {"item": "customers#x"}, {"item": "policy#tool#get_balance", "limit": "0"},
              {"item": "policy#tool#get_balance", "limit": "abc"}):
        assert call(api, "GET", "/audit", query=q or None)[0] == 400


def _last(store: MemoryStore, pk: str) -> dict[str, Any]:
    return max(audits(store, pk), key=lambda e: e["sk"])


def test_restore_a_tool_change(rig) -> None:
    api, store, _ = rig
    call(api, "PUT", "/tools/get_balance", {"rev": 0, "min_level": 2})
    entry = _last(store, "policy#tool#get_balance")
    status, body = call(api, "POST", "/restore",
                        {"audit_pk": entry["pk"], "audit_sk": entry["sk"], "rev": 1}, who=EDITOR)
    assert status == 200 and body["tool"]["min_level"] == 1 and body["security_change"]
    restore = _last(store, "policy#tool#get_balance")
    assert restore["action"] == "restore" and restore["rev"] == 2
    assert restore["restored_from"] == {"pk": entry["pk"], "sk": entry["sk"]}
    # A stale rev: refused, nothing changes.
    status, _ = call(api, "POST", "/restore", {"audit_pk": entry["pk"], "audit_sk": entry["sk"], "rev": 1})
    assert status == 409 and len(audits(store, "policy#tool#get_balance")) == 2


def test_restore_a_mask_that_was_absent(rig) -> None:
    api, store, _ = rig
    call(api, "PUT", "/tools/get_balance", {"rev": 0, "mask": None})
    entry = _last(store, "policy#tool#get_balance")
    status, body = call(api, "POST", "/restore", {"audit_pk": entry["pk"], "audit_sk": entry["sk"], "rev": 1})
    assert status == 200 and body["tool"]["mask"]["phrase_key"] == "mask.get_balance"


def test_restore_masking_and_prompt(rig) -> None:
    api, store, _ = rig
    seeded = store.get("policy", {"pk": "masking"})["after_ms"]
    changed = seeded + 500  # any value other than the seed
    call(api, "PUT", "/masking", {"rev": 0, "masking": {"after_ms": changed}})
    m = _last(store, "policy#masking")
    status, body = call(api, "POST", "/restore", {"audit_pk": m["pk"], "audit_sk": m["sk"], "rev": 1})
    assert status == 200 and body["masking"]["after_ms"] == seeded != changed
    old = PromptCatalog.from_store(store).template("mask.get_balance", "en-US")
    call(api, "PUT", "/prompts/mask.get_balance",
         {"rev": 0, "texts": {"en-US": "New text.", "es-US": "Texto nuevo."}})
    p = _last(store, "catalog#mask.get_balance")
    status, body = call(api, "POST", "/restore", {"audit_pk": p["pk"], "audit_sk": p["sk"], "rev": 1})
    assert status == 200 and body["prompt"]["texts"]["en-US"] == old


def test_list_and_restore_an_agent_prompt_with_a_dash_in_its_key(rig) -> None:
    """The console's Restore lists ``prefix=<the key>`` for the current rev;
    ``agent.system.haiku-5-5`` has a ``-``, which the prefix rule refused."""
    api, store, _ = rig
    key = "agent.system.haiku-5-5"
    status, body = call(api, "GET", "/prompts", query={"prefix": key}, who=VIEWER)
    assert status == 200 and [p["key"] for p in body["prompts"]] == [key]
    old = _agent_texts(store, key)
    call(api, "PUT", f"/prompts/{key}",
         {"rev": 0, "texts": {lang: t + " Be brief." for lang, t in old.items()}}, who=EDITOR)
    rev = call(api, "GET", "/prompts", query={"prefix": key})[1]["prompts"][0]["rev"]
    p = _last(store, f"catalog#{key}")
    status, body = call(api, "POST", "/restore", {"audit_pk": p["pk"], "audit_sk": p["sk"], "rev": rev})
    assert status == 200 and body["prompt"]["texts"] == old, body


def test_restore_refusals(rig) -> None:
    api, store, cognito = rig
    call(api, "PUT", "/prompts/mask.get_balance.9", {"rev": 0, "texts": {"en-US": "a", "es-US": "b"}})
    new = _last(store, "catalog#mask.get_balance.9")
    assert call(api, "POST", "/restore", {"audit_pk": new["pk"], "audit_sk": new["sk"], "rev": 1})[0] == 409
    cognito.add("x@example.com", "admin")
    call(api, "POST", "/users", {"email": "y@example.com", "group": "admin-viewer"})
    user = all_audits(store)[-1]
    assert user["target"]["kind"] == "user"
    assert call(api, "POST", "/restore", {"audit_pk": user["pk"], "audit_sk": user["sk"], "rev": 0})[0] == 400
    assert call(api, "POST", "/restore", {"audit_pk": "policy#masking", "audit_sk": "nope", "rev": 0})[0] == 404
    assert call(api, "POST", "/restore", {"audit_sk": "x", "rev": 0})[1]["field"] == "audit_pk"


def test_restore_goes_through_validation(rig) -> None:
    api, store, _ = rig
    store.put("admin_audit", {"pk": "policy#tool#get_balance", "sk": "2026-01-01T00:00:00.000000Z#x",
                              "target": {"kind": "tool", "id": "get_balance"},
                              "changes": [{"field": "min_level", "old": 99, "new": 1}]})
    status, body = call(api, "POST", "/restore", {"audit_pk": "policy#tool#get_balance",
                                                  "audit_sk": "2026-01-01T00:00:00.000000Z#x", "rev": 0})
    assert status == 400 and body["field"] == "tools.get_balance.min_level"


# --------------------------------------------------------------------------- users


def test_list_users_with_groups_and_pages(rig) -> None:
    api, _, cognito = rig
    for i, g in enumerate(["admin", "admin-editor", "admin-viewer", None, "admin-viewer"]):
        cognito.add(f"user{i}@example.com", g)
    status, body = call(api, "GET", "/users")
    assert status == 200 and len(body["users"]) == 5
    by_email = {u["email"]: u for u in body["users"]}
    assert by_email["user0@example.com"]["group"] == "admin"
    assert by_email["user3@example.com"]["group"] is None
    assert body["groups"] == ["admin", "admin-editor", "admin-viewer"]


def test_create_user(rig, caplog) -> None:
    api, store, cognito = rig
    with caplog.at_level(logging.DEBUG):
        status, body = call(api, "POST", "/users", {"email": "new@example.com", "group": "admin-editor"})
    assert status == 201 and body["audited"] is True
    username = body["user"]["username"]
    assert cognito.groups["admin-editor"] == {username}
    create = next(kw for op, kw in cognito.calls if op == "admin_create_user")
    assert create["DesiredDeliveryMediums"] == ["EMAIL"] and create["Username"] == "new@example.com"
    assert "MessageAction" not in create and "TemporaryPassword" not in create
    [entry] = audits(store, f"user#{username}")
    assert entry["security_change"] is True and entry["action"] == "user_create"
    assert {"field": "group", "old": None, "new": "admin-editor"} in entry["changes"]
    assert "password" not in json.dumps(entry).lower()
    assert "new@example.com" not in caplog.text  # no PII in logs


@pytest.mark.parametrize("body,status,field", [
    ({"email": "not-an-email", "group": "admin"}, 400, "email"),
    ({"email": "a@example.com", "group": "root"}, 400, "group"),
    ({"email": "a@example.com"}, 400, "group"),
    ({"email": "a@example.com", "group": "admin", "password": "x"}, 400, "password"),
])
def test_create_user_validation(rig, body, status, field) -> None:
    api, _, cognito = rig
    got, out = call(api, "POST", "/users", body)
    assert (got, out["field"]) == (status, field)
    assert [op for op, _ in cognito.calls] == ["admin_get_user", "admin_list_groups_for_user"]  # the caller


def test_create_duplicate_user_is_409(rig) -> None:
    api, _, cognito = rig
    cognito.add("dup@example.com", "admin")
    status, body = call(api, "POST", "/users", {"email": "dup@example.com", "group": "admin"})
    assert status == 409 and body["error"] == "user_exists"
    assert "secret" not in json.dumps(body)


def test_create_user_rolls_back_when_the_group_fails(rig) -> None:
    api, _, cognito = rig
    cognito.fail["admin_add_user_to_group"] = "InternalErrorException"
    status, body = call(api, "POST", "/users", {"email": "z@example.com", "group": "admin"})
    assert status == 502 and "removed" in body["message"]
    assert not cognito.users


def test_change_group_disable_enable(rig) -> None:
    api, store, cognito = rig
    cognito.add("a@example.com", "admin")
    b = cognito.add("b@example.com", "admin-viewer")
    status, body = call(api, "PATCH", f"/users/{b}", {"group": "admin-editor"})
    assert status == 200 and body["user"]["group"] == "admin-editor"
    assert cognito.groups["admin-viewer"] == set() and b in cognito.groups["admin-editor"]
    assert call(api, "PATCH", f"/users/{b}", {"enabled": False})[0] == 200
    assert cognito.users[b]["Enabled"] is False
    assert call(api, "PATCH", f"/users/{b}", {"enabled": True})[0] == 200
    entries = audits(store, f"user#{b}")
    assert len(entries) == 3 and all(e["security_change"] for e in entries)
    assert call(api, "PATCH", f"/users/{b}", {"enabled": True})[1]["changed"] is False


def test_last_admin_is_protected(rig) -> None:
    api, _, cognito = rig
    a = cognito.add("a@example.com", "admin")
    for body in ({"group": "admin-editor"}, {"enabled": False}):
        status, out = call(api, "PATCH", f"/users/{a}", body)
        assert status == 409 and out["error"] == "last_admin"
    assert call(api, "DELETE", f"/users/{a}")[1]["error"] == "last_admin"
    assert a in cognito.enabled_admins()
    b = cognito.add("b@example.com", "admin")
    assert call(api, "PATCH", f"/users/{a}", {"group": "admin-editor"})[0] == 200
    assert cognito.enabled_admins() == {b}


def test_a_disabled_admin_can_be_demoted(rig) -> None:
    api, _, cognito = rig
    cognito.add("a@example.com", "admin")
    d = cognito.add("d@example.com", "admin", enabled=False)
    assert call(api, "PATCH", f"/users/{d}", {"group": "admin-viewer"})[0] == 200


def test_update_user_validation(rig) -> None:
    api, _, cognito = rig
    b = cognito.add("b@example.com", "admin-viewer")
    assert call(api, "PATCH", f"/users/{b}", {})[0] == 400
    assert call(api, "PATCH", f"/users/{b}", {"group": "root"})[1]["field"] == "group"
    assert call(api, "PATCH", f"/users/{b}", {"enabled": "no"})[1]["field"] == "enabled"
    assert call(api, "PATCH", "/users/u-404", {"enabled": False})[0] == 404


def test_resend_only_while_pending(rig) -> None:
    api, store, cognito = rig
    p = cognito.add("p@example.com", "admin-viewer", status="FORCE_CHANGE_PASSWORD")
    c = cognito.add("c@example.com", "admin-viewer")
    status, body = call(api, "POST", f"/users/{p}/resend")
    assert status == 200 and body["resent"] and cognito.resent == [p]
    resend = [kw for op, kw in cognito.calls if op == "admin_create_user"][-1]
    assert resend["MessageAction"] == "RESEND" and resend["DesiredDeliveryMediums"] == ["EMAIL"]
    assert audits(store, f"user#{p}")[0]["action"] == "user_resend"
    assert call(api, "POST", f"/users/{c}/resend")[1]["error"] == "not_pending"


def test_delete_user_and_no_self_delete(rig) -> None:
    api, store, cognito = rig
    me = cognito.add("me@example.com", "admin")
    other = cognito.add("o@example.com", "admin-editor")
    who = claims(["admin"], sub=me)
    status, body = call(api, "DELETE", f"/users/{me}", who=who)
    assert status == 403 and body["error"] == "self_delete" and me in cognito.users
    by_name = claims(["admin"], sub="different-sub", username=me)
    assert call(api, "DELETE", f"/users/{me}", who=by_name)[0] == 403
    status, body = call(api, "DELETE", f"/users/{other}", who=who)
    assert status == 200 and body["deleted"] == other and other not in cognito.users
    [entry] = audits(store, f"user#{other}")
    assert entry["action"] == "user_delete" and entry["actor"]["sub"] == me
    assert call(api, "DELETE", f"/users/{other}", who=who)[0] == 404


def test_user_change_stands_when_the_audit_write_fails(rig, caplog) -> None:
    api, store, cognito = rig
    b = cognito.add("b@example.com", "admin-viewer")
    real_put = store.put

    def put(table, item, **kw):
        if table == "admin_audit":
            raise RuntimeError("audit table down")
        return real_put(table, item, **kw)

    store.put = put  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR):
        status, body = call(api, "PATCH", f"/users/{b}", {"enabled": False})
    assert status == 200 and body["audited"] is False and cognito.users[b]["Enabled"] is False
    assert "admin_audit_write_failed" in caplog.text and "audit table down" not in caplog.text


def test_cognito_failure_is_502_without_its_message(rig) -> None:
    api, _, cognito = rig
    cognito.fail["list_users"] = "InternalErrorException"
    status, body = call(api, "GET", "/users")
    assert status == 502 and body == {"error": "cognito_error"}


# --------------------------------------------------------------------------- sign-out and live groups


def _sign_outs(cognito: FakeCognito) -> list[str]:
    return [kw["Username"] for op, kw in cognito.calls if op == "admin_user_global_sign_out"]


def test_disable_signs_the_user_out_before_disabling(rig) -> None:
    api, _, cognito = rig
    cognito.add("a@example.com", "admin")
    b = cognito.add("b@example.com", "admin-editor")
    status, body = call(api, "PATCH", f"/users/{b}", {"enabled": False})
    assert status == 200 and body["signed_out"] is True and _sign_outs(cognito) == [b]
    ops = [op for op, _ in cognito.calls]
    assert ops.index("admin_user_global_sign_out") < ops.index("admin_disable_user")
    cognito.calls.clear()
    status, body = call(api, "PATCH", f"/users/{b}", {"enabled": True})  # enabling: no sign-out
    assert status == 200 and "signed_out" not in body and not _sign_outs(cognito)


def test_group_change_signs_the_user_out_once(rig) -> None:
    api, _, cognito = rig
    cognito.add("a@example.com", "admin")
    b = cognito.add("b@example.com", "admin")
    status, body = call(api, "PATCH", f"/users/{b}", {"group": "admin-viewer", "enabled": False})
    assert status == 200 and body["signed_out"] is True and _sign_outs(cognito) == [b]
    ops = [op for op, _ in cognito.calls]
    assert ops.index("admin_remove_user_from_group") < ops.index("admin_user_global_sign_out") \
        < ops.index("admin_disable_user")
    cognito.calls.clear()
    assert call(api, "PATCH", f"/users/{b}", {"group": "admin-viewer"})[1]["changed"] is False
    assert not _sign_outs(cognito)


def test_delete_signs_the_user_out_before_deleting(rig) -> None:
    api, _, cognito = rig
    b = cognito.add("b@example.com", "admin-viewer")
    status, body = call(api, "DELETE", f"/users/{b}")
    assert status == 200 and body["signed_out"] is True and _sign_outs(cognito) == [b]
    ops = [op for op, _ in cognito.calls]
    assert ops.index("admin_user_global_sign_out") < ops.index("admin_delete_user")


def test_a_failed_sign_out_never_blocks_the_change(rig, caplog) -> None:
    api, _, cognito = rig
    b = cognito.add("b@example.com", "admin-viewer")
    cognito.fail["admin_user_global_sign_out"] = "InternalErrorException"
    with caplog.at_level(logging.ERROR):
        status, body = call(api, "DELETE", f"/users/{b}")
    assert status == 200 and body["signed_out"] is False and b not in cognito.users
    assert "admin_user_sign_out_failed" in caplog.text and "secret detail" not in caplog.text


def test_a_demoted_admin_is_refused_once_the_cached_groups_expire() -> None:
    api, _, cognito, clock = make_api()
    me = cognito.add("me@example.com", "admin")
    cognito.add("other@example.com", "admin")
    token = claims(["admin"], sub=me)  # the claim never changes for the token's lifetime
    assert call(api, "GET", "/users", who=token)[0] == 200
    cognito.groups["admin"].discard(me)  # demoted elsewhere (another Lambda instance)
    cognito.groups["admin-viewer"].add(me)
    clock.t += 59.0
    assert call(api, "GET", "/users", who=token)[0] == 200  # cached for under 60 s
    clock.t += 1.0
    status, body = call(api, "GET", "/users", who=token)
    assert status == 403 and body["error"] == "forbidden"
    assert call(api, "PUT", "/redaction", {"rev": 0}, who=token)[0] == 403
    assert call(api, "PUT", "/slots/resolution", {"rev": 0, "generative_enabled": True},
                who=token)[0] == 403
    assert call(api, "GET", "/tools", who=token)[0] == 200  # reads still pass on the token


def test_a_change_through_this_instance_takes_effect_at_once() -> None:
    api, _, cognito, _ = make_api()
    me = cognito.add("me@example.com", "admin")
    other = cognito.add("other@example.com", "admin")
    mine, theirs = claims(["admin"], sub=me), claims(["admin"], sub=other)
    assert call(api, "GET", "/users", who=theirs)[0] == 200  # cached: admin
    assert call(api, "PATCH", f"/users/{other}", {"group": "admin-editor"}, who=mine)[0] == 200
    assert call(api, "GET", "/users", who=theirs)[0] == 403  # no wait for the cache


def test_a_fresh_entry_reflects_a_promotion_and_a_deletion() -> None:
    api, _, cognito, clock = make_api()
    cognito.add("a@example.com", "admin")
    v = cognito.add("v@example.com", "admin-viewer")
    token = claims(["admin"], sub=v)  # a token with the claim, but Cognito says viewer
    assert call(api, "GET", "/users", who=token)[0] == 403
    cognito.groups["admin-viewer"].discard(v)
    cognito.groups["admin"].add(v)
    clock.t += 60.0
    assert call(api, "GET", "/users", who=token)[0] == 200
    del cognito.users[v]
    cognito.groups["admin"].discard(v)
    clock.t += 60.0
    assert call(api, "GET", "/users", who=token)[0] == 403  # deleted: no groups, not 404


def test_live_group_lookup_uses_the_username_then_the_sub() -> None:
    api, _, cognito, _ = make_api()
    me = cognito.add("me@example.com", "admin")
    assert call(api, "GET", "/users", who=claims(["admin"], sub="other-sub", username=me))[0] == 200
    looked_up = [kw["Username"] for op, kw in cognito.calls if op == "admin_list_groups_for_user"]
    assert looked_up == [me]
    no_username = claims(["admin"], sub=me)
    del no_username["cognito:username"]
    assert call(api, "GET", "/users", who=no_username)[0] == 200


def test_a_failed_live_group_lookup_refuses_and_is_not_cached(rig) -> None:
    api, _, cognito = rig
    cognito.fail["admin_list_groups_for_user"] = "InternalErrorException"
    assert call(api, "GET", "/users")[0] == 502
    assert call(api, "GET", "/users")[0] == 200


def test_a_disabled_admin_is_refused_once_the_cached_entry_expires(caplog) -> None:
    api, _, cognito, clock = make_api()
    me = cognito.add("me@example.com", "admin")
    cognito.add("other@example.com", "admin")
    token = claims(["admin"], sub=me)
    assert call(api, "GET", "/users", who=token)[0] == 200
    cognito.users[me]["Enabled"] = False  # disabled elsewhere, still in admin
    clock.t += 59.0
    assert call(api, "GET", "/users", who=token)[0] == 200  # cached for under 60 s
    clock.t += 1.0
    with caplog.at_level(logging.WARNING):
        status, body = call(api, "GET", "/users", who=token)
    assert status == 403 and body["error"] == "forbidden"
    assert [getattr(r, "reason", None) for r in caplog.records
            if getattr(r, "event", None) == "admin_access_denied"] == ["cognito_disabled"]
    assert call(api, "PUT", "/redaction", {"rev": 0}, who=token)[0] == 403
    assert call(api, "PUT", "/slots/resolution", {"rev": 0, "generative_enabled": True},
                who=token)[0] == 403
    assert call(api, "GET", "/tools", who=token)[0] == 200  # reads still pass on the token


def test_a_disable_through_this_instance_takes_effect_at_once() -> None:
    api, _, cognito, _ = make_api()
    me = cognito.add("me@example.com", "admin")
    other = cognito.add("other@example.com", "admin")
    mine, theirs = claims(["admin"], sub=me), claims(["admin"], sub=other)
    assert call(api, "GET", "/users", who=theirs)[0] == 200  # cached: enabled admin
    assert call(api, "PATCH", f"/users/{other}", {"enabled": False}, who=mine)[0] == 200
    assert other in cognito.groups["admin"]
    assert call(api, "GET", "/users", who=theirs)[0] == 403  # no wait for the cache


def test_a_fresh_entry_reflects_a_re_enable() -> None:
    api, _, cognito, clock = make_api()
    cognito.add("a@example.com", "admin")
    me = cognito.add("me@example.com", "admin", enabled=False)
    token = claims(["admin"], sub=me)
    assert call(api, "GET", "/users", who=token)[0] == 403
    cognito.users[me]["Enabled"] = True
    clock.t += 59.0
    assert call(api, "GET", "/users", who=token)[0] == 403  # the refusal is cached too
    clock.t += 1.0
    assert call(api, "GET", "/users", who=token)[0] == 200


def test_a_disabled_actor_is_refused(rig) -> None:
    api, _, cognito = rig
    cognito.disabled_actors.add("sub-actor")
    assert call(api, "GET", "/users")[0] == 403
    assert not [op for op, _ in cognito.calls if op == "admin_list_groups_for_user"]


def test_a_failed_live_enabled_lookup_refuses_and_is_not_cached(rig) -> None:
    api, _, cognito = rig
    cognito.fail["admin_get_user"] = "InternalErrorException"
    assert call(api, "GET", "/users")[0] == 502
    assert call(api, "GET", "/users")[0] == 200


# --------------------------------------------------------------------------- handler and config


def test_handler_entry_point(monkeypatch) -> None:
    api, _, _, _ = make_api()
    monkeypatch.setattr(handler_mod, "_api", api)
    resp = handler_mod.handler(event("GET", "/tools"))
    assert resp["statusCode"] == 200


def test_handler_setup_failure_is_500(monkeypatch) -> None:
    monkeypatch.setattr(handler_mod, "_api", None)

    def fail(env=None):
        raise RuntimeError("no pool id")

    monkeypatch.setattr(handler_mod, "build_api", fail)
    resp = handler_mod.handler(event("GET", "/tools"))
    assert resp["statusCode"] == 500 and json.loads(resp["body"]) == {"error": "internal"}


def test_config_from_env() -> None:
    cfg = AdminConfig.from_env({"OWLBANK_ADMIN_USER_POOL_ID": POOL_ID})
    assert cfg.audit_retention_days == 90 and cfg.table_names == {
        "policy": "owlbank-policy", "catalog": "owlbank-catalog", "admin_audit": "owlbank-admin-audit"}
    prod = AdminConfig.from_env({"OWLBANK_ADMIN_USER_POOL_ID": POOL_ID, "OWLBANK_STAGE": "prod",
                                 "OWLBANK_TABLE_ADMIN_AUDIT": "audit-x"})
    assert prod.audit_retention_days == 400 and prod.audit_table == "audit-x"
    assert AdminConfig.from_env({"OWLBANK_ADMIN_USER_POOL_ID": POOL_ID,
                                 "OWLBANK_ADMIN_AUDIT_RETENTION_DAYS": "30"}).audit_retention_days == 30
    for env in ({}, {"OWLBANK_ADMIN_USER_POOL_ID": POOL_ID, "OWLBANK_STAGE": "qa"}):
        with pytest.raises(ValueError):
            AdminConfig.from_env(env)
