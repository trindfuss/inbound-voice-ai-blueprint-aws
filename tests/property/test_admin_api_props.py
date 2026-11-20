"""Admin API properties:

* The API accepts a masking or tool-mask value exactly when the bridge's lenient loader
  would use it without a problem, and what it stores loads back unchanged with no problem.
* A stale ``rev`` never changes an item; every successful write leaves exactly one audit
  entry, written atomically with it (an item's ``rev`` = its number of audit entries, even
  when the audit write itself fails).
* No write route succeeds without ``admin-editor`` or ``admin``, no user route without
  ``admin``, no route without claims.
* No sequence of user operations by any admin leaves zero enabled admins, and no admin
  deletes themselves.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, initialize, invariant, rule

from lambdas.admin_api.app import ACCESS, ROUTES, AdminApi, Deps
from owlbank import masking_config as mc
from owlbank.policy import Policy
from owlbank.store import MemoryStore
from tests.admin_api_fakes import POOL_ID, Clock, FakeCognito, call, claims, seeded_store

_SEED = seeded_store()


def fresh_store() -> MemoryStore:
    store = MemoryStore()
    store._tables = copy.deepcopy(_SEED._tables)
    return store


def api_for(store: MemoryStore, cognito: FakeCognito | None = None, clock: Clock | None = None) -> AdminApi:
    clock = clock or Clock()
    return AdminApi(Deps(store=store, cognito=cognito or FakeCognito(), user_pool_id=POOL_ID,
                         retention_days=90, clock=clock, suffix=clock.suffix))


EDITOR = claims(["admin-editor"])


_json_scalars = st.one_of(
    st.none(), st.booleans(), st.integers(-10**6, 10**6), st.integers(0, 10001), st.integers(-1, 6),
    st.floats(allow_nan=False, allow_infinity=False), st.floats(0, 40), st.text(max_size=10),
    st.sampled_from(["wait", "immediate", "escalate", "hangup", "15", "8.0", "nan", "inf", "1e999",
                     "-0", "30.0001", "mask.get_balance", "mask.", "mask.x{y}", "tool.x"]),
)
_values = st.one_of(_json_scalars, st.lists(_json_scalars, max_size=2),
                    st.dictionaries(st.text(max_size=4), _json_scalars, max_size=2))
_masking_raw = st.one_of(
    _values,
    st.dictionaries(st.sampled_from([*mc.MASKING_FIELDS, "rev", "bogus", "after"]), _values, max_size=6),
    st.fixed_dictionaries({}, optional=dict.fromkeys(mc.MASKING_FIELDS, _json_scalars)),
)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_masking_raw)
def test_masking_accepted_exactly_when_lenient_has_no_problem(raw: Any) -> None:
    store = fresh_store()
    api = api_for(store)
    status, body = call(api, "PUT", "/masking", {"rev": 0, "masking": raw}, who=EDITOR)
    settings_, problems = mc.masking_parts(raw)
    if problems:
        assert status == 400 and body["field"] == problems[0].field
        assert store.get("policy", {"pk": "masking"}) == _SEED.get("policy", {"pk": "masking"})
        return
    assert status == 200, body
    item = store.get("policy", {"pk": "masking"})
    stored = {k: v for k, v in item.items() if k not in ("pk", "rev")}
    again, again_problems = mc.masking_parts(stored)
    assert again_problems == [] and again == settings_
    assert Policy.from_store(store).masking == settings_


_mask_raw = st.one_of(
    _values,
    st.fixed_dictionaries(
        {}, optional={"phrase_key": st.one_of(_json_scalars, st.text(max_size=20).map(lambda s: "mask." + s)),
                      "after_ms": _json_scalars, "handoff_wait_ms": _json_scalars,
                      "extra": _json_scalars}),
    st.builds(lambda k, a, h: {"phrase_key": k, "after_ms": a, "handoff_wait_ms": h},
              st.sampled_from(["mask.get_balance", "mask.tuned"]), st.integers(-5, 10005),
              st.one_of(st.none(), st.integers(-5, 10005))),
)


def _seed_phrase(store: MemoryStore, key: Any) -> None:
    """Give a plausible phrase key a plain phrase in every language, so tests the policy
    value alone (the catalog pool check is separate)."""
    if not isinstance(key, str) or not key or key in store.get("catalog", {"key": "_meta", "lang": "_"})["keys"]:
        return
    meta = store.get("catalog", {"key": "_meta", "lang": "_"})
    store.put("catalog", {**meta, "keys": [*meta["keys"], key]})
    for lang in meta["languages"]:
        store.put("catalog", {"key": key, "lang": lang, "text": "One moment.", "version": 1,
                              "status": "published", "audio_url": None})


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_mask_raw)
def test_tool_mask_accepted_exactly_when_lenient_has_no_problem(raw: Any) -> None:
    store = fresh_store()
    if isinstance(raw, dict):
        _seed_phrase(store, raw.get("phrase_key"))
    api = api_for(store)
    status, body = call(api, "PUT", "/tools/get_balance", {"rev": 0, "mask": raw}, who=EDITOR)
    mask, problems = mc.tool_mask_parts("get_balance", raw)
    if problems:
        assert status == 400 and body["field"] == problems[0].field, body
        assert store.get("policy", {"pk": "tool#get_balance"}) == _SEED.get("policy", {"pk": "tool#get_balance"})
        return
    assert status == 200, body
    row = store.get("policy", {"pk": "tool#get_balance"})
    assert mc.tool_mask_parts("get_balance", row.get("mask")) == (mask, [])
    assert Policy.from_store(store).tool_mask("get_balance") == mask




class _FreezableClock(Clock):
    """``frozen``: the next entry reuses the previous time and suffix, so its audit put collides
    with an existing entry and the transaction must fail as a whole."""

    def __init__(self) -> None:
        super().__init__()
        self.frozen = False
        self._last_suffix = "0"

    def __call__(self) -> float:
        if not self.frozen:
            self.t += 0.001
        return self.t

    def suffix(self) -> str:
        if not self.frozen:
            self._last_suffix = super().suffix()
        return self._last_suffix


_ITEMS = {
    "tool": ("PUT", "/tools/get_balance", "policy#tool#get_balance"),
    "masking": ("PUT", "/masking", "policy#masking"),
    "prompt": ("PUT", "/prompts/mask.get_balance", "catalog#mask.get_balance"),
}


def _current_rev(store: MemoryStore, kind: str) -> int:
    if kind == "tool":
        return store.get("policy", {"pk": "tool#get_balance"}).get("rev", 0)
    if kind == "masking":
        return (store.get("policy", {"pk": "masking"}) or {}).get("rev", 0)
    return max(it.get("rev", 0) for it in store.query("catalog", {"key": "mask.get_balance"}))


def _snapshot(store: MemoryStore, kind: str) -> Any:
    if kind == "tool":
        return store.get("policy", {"pk": "tool#get_balance"})
    if kind == "masking":
        return store.get("policy", {"pk": "masking"})
    return store.query("catalog", {"key": "mask.get_balance"})


def _body(kind: str, rev: int, value: int) -> dict[str, Any]:
    if kind == "tool":
        return {"rev": rev, "min_level": value % 4}
    if kind == "masking":
        return {"rev": rev, "masking": {"after_ms": value * 100}}
    return {"rev": rev, "texts": {"en-US": f"One moment {value}.", "es-US": f"Un momento {value}."}}


_op = st.tuples(st.sampled_from(sorted(_ITEMS)), st.integers(-1, 2), st.integers(0, 7), st.booleans())


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(_op, min_size=1, max_size=12))
def test_stale_rev_never_writes_and_each_write_has_one_audit(ops: list[tuple[str, int, int, bool]]) -> None:
    store = fresh_store()
    clock = _FreezableClock()
    api = api_for(store, clock=clock)
    last_pk: str | None = None  # the item of the newest audit entry (whose sk a frozen clock repeats)
    for kind, skew, value, collide in ops:
        method, path, pk = _ITEMS[kind]
        before, before_rev = _snapshot(store, kind), _current_rev(store, kind)
        before_audits = len(store.query("admin_audit", {"pk": pk}))
        rev = max(0, before_rev + skew)
        clock.frozen = collide and last_pk == pk
        status, body = call(api, method, path, _body(kind, rev, value), who=EDITOR)
        after_audits = len(store.query("admin_audit", {"pk": pk}))
        if rev != before_rev or (clock.frozen and not (status == 200 and body["changed"] is False)):
            # A stale rev, or an audit put that collides: the whole write is refused.
            assert status == 409, body
            assert _snapshot(store, kind) == before and after_audits == before_audits
        else:
            assert status == 200, body
            if body["changed"]:
                last_pk = pk
                assert after_audits == before_audits + 1
                assert _current_rev(store, kind) == before_rev + 1
            else:
                assert _snapshot(store, kind) == before and after_audits == before_audits
        clock.frozen = False
        for k, (_m, _p, item_pk) in _ITEMS.items():  # rev = number of audited writes, per item
            assert _current_rev(store, k) == len(store.query("admin_audit", {"pk": item_pk}))



_GROUPS = st.sets(st.sampled_from(["admin", "admin-editor", "admin-viewer", "Admin", "editor", ""]))


def _request(route_name: str, users: dict[str, str]) -> tuple[str, dict[str, Any] | None, dict[str, str] | None]:
    """A request that succeeds for a caller with the right group."""
    other = users["other"]
    return {
        "list_tools": ("/tools", None, None),
        "get_tool": ("/tools/get_balance", None, None),
        "put_tool": ("/tools/get_balance", {"rev": 0, "min_level": 2}, None),
        "get_masking": ("/masking", None, None),
        "put_masking": ("/masking", {"rev": 0, "masking": {"after_ms": 1500}}, None),
        "list_prompts": ("/prompts", None, {"prefix": "mask."}),
        "put_prompt": ("/prompts/mask.get_balance",
                       {"rev": 0, "texts": {"en-US": "Hold on.", "es-US": "Espere."}}, None),
        "get_placeholders": ("/placeholders/get_balance", None, None),
        "list_audit": ("/audit", None, {"item": "policy#tool#get_balance"}),
        "restore": ("/restore", {"audit_pk": "policy#tool#get_balance", "audit_sk": users["audit_sk"],
                                 "rev": 1}, None),
        "list_slots": ("/slots", None, None),
        "put_slot": ("/slots/zip", {"rev": 0, "keypad_only": True}, None),
        "put_resolution": ("/slots/resolution", {"rev": 0, "resolver_budget_ms": 2000}, None),
        "preview_slots": ("/slots/preview", {"slot_resolution": {"default_mode": "deterministic"}}, None),
        "get_redaction": ("/redaction", None, None),
        "put_redaction": ("/redaction", {"rev": 0, "min_confidence": 0.75}, None),
        "list_users": ("/users", None, None),
        "create_user": ("/users", {"email": "new@example.com", "group": "admin-viewer"}, None),
        "update_user": (f"/users/{other}", {"group": "admin-editor"}, None),
        "resend_invitation": (f"/users/{users['pending']}/resend", None, None),
        "delete_user": (f"/users/{other}", None, None),
    }[route_name]


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(ROUTES), _GROUPS, st.booleans(), st.booleans())
def test_access_by_group_and_claims(route, groups: set[str], has_claims: bool, groups_as_list: bool) -> None:
    store = fresh_store()
    cognito = FakeCognito()
    users = {"admin": cognito.add("a@example.com", "admin"), "other": cognito.add("o@example.com", "admin-viewer"),
             "pending": cognito.add("p@example.com", "admin-viewer", status="FORCE_CHANGE_PASSWORD")}
    api = api_for(store, cognito)
    # One earlier change, so the restore request has an entry to restore (rev is now 1).
    assert call(api, "PUT", "/tools/get_balance", {"rev": 0, "min_level": 3}, who=EDITOR)[0] == 200
    users["audit_sk"] = store.query("admin_audit", {"pk": "policy#tool#get_balance"})[0]["sk"]
    audits_before = len(store._tables["admin_audit"])
    path, body, query = _request(route.name, users)
    if route.name == "put_tool":
        body = {**body, "rev": 1}
    pool_before = copy.deepcopy((cognito.users, cognito.groups))
    cognito.calls.clear()
    who = claims(sorted(groups) if groups_as_list else ",".join(sorted(groups)), sub="sub-caller")
    cognito.actors["sub-caller"] = set(groups)  # Cognito agrees with the token
    if groups_as_list:
        who["cognito:groups"] = sorted(groups)
    status, out = call(api, route.method, path, body, query=query, who=who, no_claims=not has_claims)
    allowed = has_claims and bool(groups & ACCESS[route.access])
    if not has_claims:
        assert status == 401
    elif not allowed:
        assert status == 403
    else:
        assert 200 <= status < 300, (route.name, status, out)
    if not allowed:
        assert len(store._tables["admin_audit"]) == audits_before
        assert not cognito.calls and (cognito.users, cognito.groups) == pool_before
    if route.access == "write" and 200 <= status < 300:
        assert groups & {"admin", "admin-editor"}
    if route.access in ("users", "admin") and 200 <= status < 300:
        assert "admin" in groups




class UserAdminMachine(RuleBasedStateMachine):
    """Random admins create, regroup, disable, enable and delete users."""

    users = Bundle("users")

    @initialize(target=users)
    def first_admin(self) -> str:
        self.store = MemoryStore()
        self.cognito = FakeCognito()
        self.api = api_for(self.store, self.cognito)
        self.first = self.cognito.add("first@example.com", "admin")
        self.emails = 0
        return self.first

    def _actor(self, pick: int) -> dict[str, Any]:
        """An enabled admin now (the API checks Cognito, not only the token's groups)."""
        names = sorted(self.cognito.enabled_admins()) or [self.first]
        return claims(["admin"], sub=names[pick % len(names)])

    @rule(target=users, group=st.sampled_from(["admin", "admin-editor", "admin-viewer"]), pick=st.integers(0, 50))
    def create(self, group: str, pick: int) -> str:
        self.emails += 1
        status, body = call(self.api, "POST", "/users", {"email": f"user{self.emails}@example.com", "group": group},
                            who=self._actor(pick))
        assert status == 201, body
        return body["user"]["username"]

    def _admin_would_vanish(self, target: str) -> bool:
        return self.cognito.enabled_admins() == {target}

    @rule(user=users, group=st.sampled_from(["admin", "admin-editor", "admin-viewer"]), pick=st.integers(0, 50))
    def regroup(self, user: str, group: str, pick: int) -> None:
        if user not in self.cognito.users:
            assert call(self.api, "PATCH", f"/users/{user}", {"group": group}, who=self._actor(pick))[0] == 404
            return
        vanish = group != "admin" and self._admin_would_vanish(user)
        status, body = call(self.api, "PATCH", f"/users/{user}", {"group": group}, who=self._actor(pick))
        assert status == (409 if vanish else 200), body
        if vanish:
            assert body["error"] == "last_admin"

    @rule(user=users, enabled=st.booleans(), pick=st.integers(0, 50))
    def set_enabled(self, user: str, enabled: bool, pick: int) -> None:
        if user not in self.cognito.users:
            return
        vanish = not enabled and self._admin_would_vanish(user)
        status, body = call(self.api, "PATCH", f"/users/{user}", {"enabled": enabled}, who=self._actor(pick))
        assert status == (409 if vanish else 200), body

    @rule(user=users, pick=st.integers(0, 50), self_delete=st.booleans())
    def delete(self, user: str, pick: int, self_delete: bool) -> None:
        if user not in self.cognito.users:
            return
        actor = claims(["admin"], sub=user) if self_delete else self._actor(pick)
        is_self = actor["sub"] == user
        vanish = self._admin_would_vanish(user)
        status, body = call(self.api, "DELETE", f"/users/{user}", who=actor)
        if is_self and user not in self.cognito.enabled_admins():  # a stale admin token
            assert status == 403 and body["error"] == "forbidden" and user in self.cognito.users
        elif is_self:
            assert status == 403 and body["error"] == "self_delete" and user in self.cognito.users
        elif vanish:
            assert status == 409 and user in self.cognito.users
        else:
            assert status == 200 and user not in self.cognito.users

    @invariant()
    def never_zero_enabled_admins(self) -> None:
        if hasattr(self, "cognito"):
            assert len(self.cognito.enabled_admins()) >= 1

    @invariant()
    def every_change_is_audited(self) -> None:
        if hasattr(self, "store"):
            for item in self.store._tables["admin_audit"].values():
                assert item["security_change"] is True and item["target"]["kind"] == "user"
                assert "password" not in json.dumps(item).lower()


UserAdminMachine.TestCase.settings = settings(max_examples=60, stateful_step_count=25, deadline=None,
                                              suppress_health_check=[HealthCheck.too_slow])
TestUserAdmin = UserAdminMachine.TestCase

