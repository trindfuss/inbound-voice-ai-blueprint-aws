"""Admin API slot route properties:

* For any sequence of ``PUT /slots/{name}`` with random bodies and revs,
  each successful change writes exactly one audit entry, ``security_change`` iff ``sensitive`` or
  ``keypad_only`` changed; a stale ``rev`` (or a refused body) changes neither the item nor the
  audit table.
* Every effective ``keypad_only`` true -> false is a ``security_change``
  entry (and carries the ``keypad_only_off`` warning); likewise ``sensitive``.
* For any slot items and ``acceptSpokenFactors``, ``GET /slots``'s
  ``effective_keypad_only`` and ``barge_in`` equal the front door's ``gather_input`` over the
  policy the front door loads from the same table.

Phase 2:

* For any sequence of ``PUT``s to ``slot#*`` (``mode``, ``confirm``,
  ``options`` and the settings) and ``slot_resolution`` with random bodies, callers and revs,
  each success writes exactly one audit entry, ``security_change`` iff a changed field is in
  ``SECURITY_FIELDS``; a stale rev, a refused body and an editor's ``generative_enabled`` change
  nothing.
* For any policy (slot modes, sensitivity, ``slot_resolution``) and ceiling,
  ``GET /slots``'s effective-mode table equals:func:`owlbank.slots.modes.freeze` over the policy
  the front door loads, for every flow.
* A ``PUT /slots/card`` of a valid option set is refused iff
:func:`~owlbank.slot_policy.alias_conflicts` (with the catalog's spoken names) is non-empty.
"""

from __future__ import annotations

import copy
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.admin_api.app import AdminApi, Deps
from owlbank.policy import Policy
from lambdas.admin_api import audit
from owlbank.slot_policy import FLOWS, MODES, SETTING_FIELDS, OptionDef, alias_conflicts, gather_input
from owlbank.slots import modes as modes_mod
from owlbank.store import MemoryStore
from tests.admin_api_fakes import POOL_ID, Clock, FakeCognito, call, claims, seeded_store

_SEED = seeded_store()
EDITOR = claims(["admin-editor"])
VIEWER = claims(["admin-viewer"])
SLOTS = ("zip", "ssn4", "otp", "yes_no")
SECURITY = {"sensitive", "keypad_only"}


def fresh_store() -> MemoryStore:
    store = MemoryStore()
    store._tables = copy.deepcopy(_SEED._tables)
    return store


def api_for(store: MemoryStore, accept: bool = True, ceiling: str = "assisted") -> AdminApi:
    clock = Clock()
    return AdminApi(Deps(store=store, cognito=FakeCognito(), user_pool_id=POOL_ID, retention_days=90,
                         clock=clock, suffix=clock.suffix, accept_spoken_factors=accept,
                         slot_mode_ceiling=ceiling))  # type: ignore[arg-type]


_json = st.one_of(st.booleans(), st.booleans(), st.none(), st.integers(-1, 2), st.text(max_size=3),
                  st.lists(st.booleans(), max_size=1))
#: A slot item's setting fields: each absent, a boolean, or any JSON value.
_fields = st.fixed_dictionaries({}, optional=dict.fromkeys(SETTING_FIELDS, _json))


def _apply_items(store: MemoryStore, items: dict[str, dict[str, Any] | None]) -> None:
    """Replace the seed's setting fields of each slot item (``None`` = delete the item)."""
    for name, fields in items.items():
        pk = {"pk": f"slot#{name}"}
        if fields is None:
            store.delete("policy", pk)
            continue
        base = {k: v for k, v in (store.get("policy", pk) or {"pk": pk["pk"]}).items()
                if k not in SETTING_FIELDS}
        store.put("policy", {**base, **fields})


_items = st.dictionaries(st.sampled_from(SLOTS), st.one_of(st.none(), _fields), max_size=4)




@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_items, st.booleans())
def test_get_slots_equals_front_door_gather_input(items: dict[str, Any], accept: bool) -> None:
    store = fresh_store()
    _apply_items(store, items)
    status, body = call(api_for(store, accept), "GET", "/slots", who=VIEWER)
    assert status == 200, body
    assert body["acceptSpokenFactors"] is accept
    got = {s["name"]: s for s in body["slots"] if s["loaded"]}
    policy = Policy.from_store(store)  # what the front door loads
    assert set(got) == set(policy.slots())
    for name in policy.slots():
        slot = policy.slot(name)
        g = gather_input(slot, accept)
        assert got[name]["effective_keypad_only"] is g.keypad_only
        assert got[name]["barge_in"] is g.barge_in
        assert got[name]["sensitive"] is slot.sensitive and got[name]["keypad_only"] is slot.keypad_only
        warned = "verification_not_sensitive" in [w["code"] for w in got[name]["warnings"]]
        assert warned == (slot.is_factor and not slot.sensitive)  # 9.8




def _effective(store: MemoryStore, name: str) -> dict[str, bool] | None:
    policy = Policy.from_store(store)
    if name not in policy.slots():  # a non-factor slot that can't be loaded
        return None
    s = policy.slot(name)
    return {f: getattr(s, f) for f in SETTING_FIELDS}


def _snapshot(store: MemoryStore, name: str) -> Any:
    return copy.deepcopy(store.get("policy", {"pk": f"slot#{name}"}))


_body = st.fixed_dictionaries({}, optional={**dict.fromkeys(SETTING_FIELDS, _json),
                                            "bogus": st.booleans()})
_rev_choice = st.sampled_from(["current", "current", "current", "stale", "ahead"])
_ops = st.lists(st.tuples(st.sampled_from(SLOTS), _body, _rev_choice), min_size=1, max_size=12)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_items, _ops)
def test_audit_per_put(items: dict[str, Any], ops: list[tuple[str, dict[str, Any], str]]) -> None:
    store = fresh_store()
    _apply_items(store, items)
    api = api_for(store)
    for name, fields, rev_choice in ops:
        before_item = _snapshot(store, name)
        before_audits = len(store._tables["admin_audit"])
        before_eff = _effective(store, name)
        rev = (before_item or {}).get("rev", 0)
        if not isinstance(rev, int):
            rev = 0
        sent_rev = {"current": rev, "stale": rev - 1, "ahead": rev + 1}[rev_choice]
        if sent_rev < 0:
            sent_rev = rev + 2
        status, out = call(api, "PUT", f"/slots/{name}", {**fields, "rev": sent_rev}, who=EDITOR)
        after_audits = len(store._tables["admin_audit"])
        if status != 200 or not out["changed"]:
            assert status in (200, 400, 409) or (status == 404 and before_eff is None), (status, out)
            if rev_choice != "current":
                assert status in (400, 404, 409)
            assert _snapshot(store, name) == before_item and after_audits == before_audits
            continue
        assert rev_choice == "current"
        assert after_audits == before_audits + 1
        after_item = store.get("policy", {"pk": f"slot#{name}"})
        assert after_item["rev"] == rev + 1
        (entry,) = [e for e in store.query("admin_audit", {"pk": f"policy#slot#{name}"})
                    if e["sk"] == out["audit"]["sk"]]
        changed = {c["field"] for c in entry["changes"]}
        assert changed and changed <= set(SETTING_FIELDS)
        for c in entry["changes"]:
            assert after_item[c["field"]] == c["new"]
        assert entry["security_change"] is bool(changed & SECURITY) is out["security_change"]
        after_eff = _effective(store, name)
        assert before_eff is not None and after_eff is not None
        warnings = [w["code"] for w in out["warnings"]]
        for f in SECURITY:
            if before_eff[f] != after_eff[f]:
                assert entry["security_change"] is True
        # Turning either setting off is a security change with its warning.
        assert ("keypad_only_off" in warnings) == (before_eff["keypad_only"] and not after_eff["keypad_only"])
        assert ("sensitive_off" in warnings) == (before_eff["sensitive"] and not after_eff["sensitive"])
    # rev = number of audited writes, per slot item that was written (the seed's items have no rev)
    for name in SLOTS:
        n = len(store.query("admin_audit", {"pk": f"policy#slot#{name}"}))
        if n:
            assert store.get("policy", {"pk": f"slot#{name}"})["rev"] == n


# =========================================================================== phase 2

ADMIN = claims(["admin"], sub="sub-admin")
ALL_SLOTS = (*SLOTS, "card")
_mode_json = st.one_of(st.sampled_from(MODES), st.none(), st.just("bogus"), st.integers(0, 1))
_flows_json = st.one_of(
    st.dictionaries(st.sampled_from([*FLOWS, "billing"]),
                    st.one_of(st.fixed_dictionaries({"mode": _mode_json}), st.sampled_from(MODES)),
                    max_size=3),
    st.none(), st.just([]))




_slot_modes = st.dictionaries(
    st.sampled_from(ALL_SLOTS),
    st.fixed_dictionaries({}, optional={"mode": _mode_json, "sensitive": _json}),
    max_size=5)
_resolution_item = st.one_of(
    st.none(),
    st.fixed_dictionaries({}, optional={"default_mode": _mode_json, "flows": _flows_json,
                                        "generative_enabled": st.one_of(st.booleans(), st.just("yes"))}))


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_slot_modes, _resolution_item, st.sampled_from(MODES))
def test_get_slots_modes_equal_freeze(slot_fields: dict[str, Any], res: dict[str, Any] | None,
                                           ceiling: str) -> None:
    store = fresh_store()
    for name, fields in slot_fields.items():
        pk = {"pk": f"slot#{name}"}
        base = {k: v for k, v in (store.get("policy", pk) or {}).items() if k not in fields}
        store.put("policy", {**base, **pk, **{k: v for k, v in fields.items() if v is not None}})
    if res is None:
        store.delete("policy", {"pk": "slot_resolution"})
    else:
        store.put("policy", {"pk": "slot_resolution", **res})
    status, body = call(api_for(store, ceiling=ceiling), "GET", "/slots", who=VIEWER)
    assert status == 200, body
    assert body["ceiling"] == ceiling
    got = {s["name"]: s["modes"] for s in body["slots"] if s["loaded"]}
    policy = Policy.from_store(store)
    assert set(got) == set(policy.slots())
    for flow in FLOWS:
        names = [n for n in policy.slots() if flow in modes_mod.slot_flows(policy.slot(n))]
        frozen = modes_mod.freeze(policy, flow, names, ceiling)  # type: ignore[arg-type]
        for name in policy.slots():
            if name in frozen:
                assert got[name][flow] == frozen[name].to_doc(), (name, flow)
            else:
                assert flow not in got[name]
    r = policy.slot_resolution
    assert body["slot_resolution"]["generative_enabled"] is r.generative_enabled
    assert body["slot_resolution"]["default_mode"] == r.default_mode




_PHRASES = ("red", "roja", "purple", "morada", "rewards", "card", "the card", "OwlBank Red Card",
            "tarjeta owlbank morada", "gold")
_aliases = st.fixed_dictionaries({}, optional={
    "en-US": st.lists(st.sampled_from(_PHRASES), max_size=2, unique=True),
    "es-US": st.lists(st.sampled_from(_PHRASES), max_size=2, unique=True)})
_option_sets = st.lists(st.tuples(st.sampled_from(("red", "purple", "rewards")), _aliases),
                        min_size=1, max_size=3, unique_by=lambda t: t[0])
_SPOKEN = {"red": {"en-US": "OwlBank Red Card", "es-US": "Tarjeta OwlBank Roja"},
           "purple": {"en-US": "OwlBank Purple Card", "es-US": "Tarjeta OwlBank Morada"},
           "rewards": {"en-US": "OwlBank Rewards Card", "es-US": "Tarjeta OwlBank de Recompensas"}}


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_option_sets)
def test_put_refused_iff_alias_conflicts(option_set: list[tuple[str, dict[str, list[str]]]]) -> None:
    store = fresh_store()
    body_options = [{"value": v, **({"aliases": a} if a else {})} for v, a in option_set]
    defs = [OptionDef(value=v, aliases={lang: tuple(p) for lang, p in a.items()}) for v, a in option_set]
    conflicts = alias_conflicts(defs, _SPOKEN, ("en-US", "es-US"))
    before = _snapshot(store, "card")
    status, out = call(api_for(store), "PUT", "/slots/card", {"rev": 0, "options": body_options}, who=EDITOR)
    if conflicts:
        assert status == 400, out
        assert out["field"].startswith("slot#card.options.aliases.")
        assert _snapshot(store, "card") == before and not store._tables.get("admin_audit")
    else:
        assert status == 200, out
        assert [o.value for o in Policy.from_store(store).slot("card").options] == [v for v, _ in option_set]


# --------------------------------------------------------------------------- Phase 2 fields


_options_json = st.one_of(
    st.none(),
    st.lists(st.fixed_dictionaries({"value": st.sampled_from(("red", "purple", "rewards", "gold", "Bad"))},
                                   optional={"aliases": _aliases, "dtmf": st.sampled_from(("1", "2", "x"))}),
             max_size=3),
    st.just("red"))
_slot_body = st.fixed_dictionaries({}, optional={
    **dict.fromkeys(SETTING_FIELDS, st.one_of(st.booleans(), st.just("x"))),
    "mode": _mode_json, "confirm": st.sampled_from(("when_inferred", "always", "never", "sometimes")),
    "options": _options_json})
_valid_flows = st.dictionaries(st.sampled_from(FLOWS), st.fixed_dictionaries({"mode": st.sampled_from(MODES)}),
                               max_size=3)
#: Mostly valid bodies, so that most writes succeed; some invalid values in every field.
_res_body = st.fixed_dictionaries({}, optional={
    "default_mode": st.one_of(st.sampled_from(MODES), st.sampled_from(MODES), _mode_json),
    "flows": st.one_of(_valid_flows, _valid_flows, _flows_json),
    "generative_enabled": st.one_of(st.booleans(), st.booleans(), st.just(1)),
    "resolver_budget_ms": st.one_of(st.integers(500, 4000), st.integers(400, 4100), st.just("1500"))})
_target = st.one_of(st.tuples(st.sampled_from(ALL_SLOTS), st.one_of(_slot_body.filter(bool), _slot_body)),
                    st.tuples(st.just("resolution"), st.one_of(_res_body.filter(bool), _res_body)))
_ops2 = st.lists(st.tuples(_target, _rev_choice, st.sampled_from(("editor", "admin"))), min_size=1, max_size=10)


def _target_item(store: MemoryStore, target: str) -> Any:
    pk = "slot_resolution" if target == "resolution" else f"slot#{target}"
    return copy.deepcopy(store.get("policy", {"pk": pk}))


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_ops2, st.sampled_from(MODES))
def test_phase2_audit_per_put(ops: list[tuple[tuple[str, dict[str, Any]], str, str]], ceiling: str) -> None:
    store = fresh_store()
    api = api_for(store, ceiling=ceiling)
    for (target, fields), rev_choice, who in ops:
        before_item = _target_item(store, target)
        before_audits = len(store._tables.get("admin_audit", {}))
        rev = (before_item or {}).get("rev", 0)
        sent_rev = {"current": rev, "stale": rev - 1, "ahead": rev + 1}[rev_choice]
        if sent_rev < 0:
            sent_rev = rev + 2
        caller = EDITOR if who == "editor" else ADMIN
        status, out = call(api, "PUT", f"/slots/{target}", {**fields, "rev": sent_rev}, who=caller)
        after_audits = len(store._tables.get("admin_audit", {}))
        if target == "resolution" and who == "editor" and "generative_enabled" in fields:
            assert status == 403, out  # 14.4: admin only, whatever else the body holds
        if status != 200 or not out["changed"]:
            assert status in (200, 400, 403, 409), (status, out)
            if rev_choice != "current":
                assert status in (400, 403, 409)
            assert _target_item(store, target) == before_item and after_audits == before_audits
            continue
        assert rev_choice == "current"
        assert after_audits == before_audits + 1
        after_item = _target_item(store, target)
        assert after_item["rev"] == rev + 1
        pk = audit.RESOLUTION_PK if target == "resolution" else audit.slot_pk(target)
        (entry,) = [e for e in store.query("admin_audit", {"pk": pk}) if e["sk"] == out["audit"]["sk"]]
        changed = {c["field"] for c in entry["changes"]}
        assert changed
        for c in entry["changes"]:
            assert after_item.get(c["field"]) == c["new"]
        assert entry["security_change"] is bool(changed & audit.SECURITY_FIELDS) is out["security_change"]
        if "generative_enabled" in changed:
            assert entry["actor"]["sub"] == "sub-admin" and entry["ts"]
    # rev = number of audited writes, per item written
    for target in (*ALL_SLOTS, "resolution"):
        pk = audit.RESOLUTION_PK if target == "resolution" else audit.slot_pk(target)
        n = len(store.query("admin_audit", {"pk": pk}))
        if n:
            assert _target_item(store, target)["rev"] == n
    Policy.from_store(store)  # every accepted write leaves a table the front door loads
