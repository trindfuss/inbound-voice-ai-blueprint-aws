"""Fakes and fixtures for the admin API tests: an in-memory Cognito user pool, seeded stores
and API Gateway events. No network."""

from __future__ import annotations

import copy
import itertools
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from lambdas.admin_api.app import AdminApi, Deps
from owlbank.catalog import PromptCatalog
from owlbank.policy import Policy
from owlbank.store import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
POOL_ID = "us-east-1_TESTPOOL"
GROUPS = ("admin", "admin-editor", "admin-viewer")


def _err(code: str, op: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "fake message with secret detail"},
                        "ResponseMetadata": {"HTTPStatusCode": 400}}, op)


class FakeCognito:
    """The ``cognito-idp`` calls the admin API makes, over an in-memory pool.

    Usernames are generated (``u-<n>``, also the ``sub``), as in a pool that signs in by email.
    ``calls`` records each call; ``fail`` maps an operation name to an error code to raise once.
    Small page sizes exercise pagination.

    ``actors`` gives the groups of callers who aren't users of the pool under test, for the
    API's own check (``AdminGetUser`` and ``AdminListGroupsForUser``) only, so they don't change
    user lists or admin counts; an actor in ``disabled_actors`` is disabled. The default actor of
    :func:`claims` (``sub-actor``) and ``sub-admin`` are ``admin``; ``sub-editor`` and
    ``sub-viewer`` are in their groups."""

    PAGE = 2

    def __init__(self, pool_id: str = POOL_ID) -> None:
        self.pool_id = pool_id
        self.users: dict[str, dict[str, Any]] = {}
        self.groups: dict[str, set[str]] = {g: set() for g in GROUPS}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail: dict[str, str] = {}
        self.resent: list[str] = []
        self.signed_out: list[str] = []
        self.actors: dict[str, set[str]] = {"sub-actor": {"admin"}, "sub-admin": {"admin"},
                                            "sub-editor": {"admin-editor"},
                                            "sub-viewer": {"admin-viewer"}}
        self.disabled_actors: set[str] = set()
        self._ids = itertools.count(1)

    # -- helpers

    def add(self, email: str, group: str | None, *, enabled: bool = True,
            status: str = "CONFIRMED") -> str:
        name = f"u-{next(self._ids)}"
        self.users[name] = {"Username": name, "Enabled": enabled, "UserStatus": status,
                            "UserCreateDate": datetime(2026, 10, 1, tzinfo=UTC),
                            "Attributes": [{"Name": "sub", "Value": name},
                                           {"Name": "email", "Value": email}]}
        if group:
            self.groups[group].add(name)
        return name

    def enabled_admins(self) -> set[str]:
        return {u for u in self.groups["admin"] if u in self.users and self.users[u]["Enabled"]}

    def _enter(self, op: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((op, copy.deepcopy(kwargs)))
        assert kwargs.get("UserPoolId") == self.pool_id
        code = self.fail.pop(op, None)
        if code:
            raise _err(code, op)

    def _user(self, op: str, username: str) -> dict[str, Any]:
        user = self.users.get(username)
        if user is None:
            raise _err("UserNotFoundException", op)
        return user

    @staticmethod
    def _page(items: list[Any], token: str | None, size: int) -> tuple[list[Any], str | None]:
        start = int(token or 0)
        page = items[start:start + size]
        nxt = str(start + size) if start + size < len(items) else None
        return page, nxt

    # -- the API

    def list_users(self, **kw: Any) -> dict[str, Any]:
        self._enter("list_users", kw)
        page, nxt = self._page(sorted(self.users), kw.get("PaginationToken"), self.PAGE)
        out: dict[str, Any] = {"Users": [copy.deepcopy(self.users[u]) for u in page]}
        if nxt:
            out["PaginationToken"] = nxt
        return out

    def list_users_in_group(self, **kw: Any) -> dict[str, Any]:
        self._enter("list_users_in_group", kw)
        group = kw["GroupName"]
        if group not in self.groups:
            raise _err("ResourceNotFoundException", "list_users_in_group")
        page, nxt = self._page(sorted(self.groups[group]), kw.get("NextToken"), self.PAGE)
        out: dict[str, Any] = {"Users": [copy.deepcopy(self.users[u]) for u in page]}
        if nxt:
            out["NextToken"] = nxt
        return out

    def admin_get_user(self, **kw: Any) -> dict[str, Any]:
        self._enter("admin_get_user", kw)
        name = kw["Username"]
        if name not in self.users and name in self.actors:
            return {"Username": name, "UserAttributes": [{"Name": "sub", "Value": name}],
                    "Enabled": name not in self.disabled_actors, "UserStatus": "CONFIRMED",
                    "UserCreateDate": datetime(2026, 10, 1, tzinfo=UTC)}
        u = self._user("admin_get_user", name)
        return {"Username": u["Username"], "UserAttributes": copy.deepcopy(u["Attributes"]),
                "Enabled": u["Enabled"], "UserStatus": u["UserStatus"],
                "UserCreateDate": u["UserCreateDate"]}

    def admin_list_groups_for_user(self, **kw: Any) -> dict[str, Any]:
        self._enter("admin_list_groups_for_user", kw)
        if kw["Username"] not in self.users and kw["Username"] in self.actors:
            return {"Groups": [{"GroupName": g} for g in GROUPS if g in self.actors[kw["Username"]]]}
        self._user("admin_list_groups_for_user", kw["Username"])
        return {"Groups": [{"GroupName": g} for g in GROUPS if kw["Username"] in self.groups[g]]}

    def admin_create_user(self, **kw: Any) -> dict[str, Any]:
        self._enter("admin_create_user", kw)
        assert "TemporaryPassword" not in kw
        if kw.get("MessageAction") == "RESEND":
            u = self._user("admin_create_user", kw["Username"])
            if u["UserStatus"] != "FORCE_CHANGE_PASSWORD":
                raise _err("UnsupportedUserStateException", "admin_create_user")
            self.resent.append(u["Username"])
            return {"User": copy.deepcopy(u)}
        email = kw["Username"]
        if any(_email(u) == email for u in self.users.values()):
            raise _err("UsernameExistsException", "admin_create_user")
        name = self.add(email, None, status="FORCE_CHANGE_PASSWORD")
        return {"User": copy.deepcopy(self.users[name])}

    def admin_add_user_to_group(self, **kw: Any) -> dict[str, Any]:
        self._enter("admin_add_user_to_group", kw)
        self._user("admin_add_user_to_group", kw["Username"])
        self.groups[kw["GroupName"]].add(kw["Username"])
        return {}

    def admin_remove_user_from_group(self, **kw: Any) -> dict[str, Any]:
        self._enter("admin_remove_user_from_group", kw)
        self._user("admin_remove_user_from_group", kw["Username"])
        self.groups[kw["GroupName"]].discard(kw["Username"])
        return {}

    def admin_disable_user(self, **kw: Any) -> dict[str, Any]:
        self._enter("admin_disable_user", kw)
        self._user("admin_disable_user", kw["Username"])["Enabled"] = False
        return {}

    def admin_enable_user(self, **kw: Any) -> dict[str, Any]:
        self._enter("admin_enable_user", kw)
        self._user("admin_enable_user", kw["Username"])["Enabled"] = True
        return {}

    def admin_delete_user(self, **kw: Any) -> dict[str, Any]:
        self._enter("admin_delete_user", kw)
        self._user("admin_delete_user", kw["Username"])
        del self.users[kw["Username"]]
        for g in self.groups.values():
            g.discard(kw["Username"])
        return {}

    def admin_user_global_sign_out(self, **kw: Any) -> dict[str, Any]:
        self._enter("admin_user_global_sign_out", kw)
        self._user("admin_user_global_sign_out", kw["Username"])
        self.signed_out.append(kw["Username"])
        return {}


def _email(user: Mapping[str, Any]) -> str | None:
    return next((a["Value"] for a in user["Attributes"] if a["Name"] == "email"), None)


# --------------------------------------------------------------------------- stores


def seed_docs() -> tuple[dict[str, Any], dict[str, Any]]:
    policy = json.loads((ROOT / "data" / "policy.json").read_text())
    catalog = json.loads((ROOT / "data" / "prompt_catalog.json").read_text())
    return policy, catalog


def seeded_store() -> MemoryStore:
    store = MemoryStore()
    policy, catalog = seed_docs()
    for item in Policy.to_items(policy):
        store.put("policy", item)
    for item in PromptCatalog.to_items(catalog):
        store.put("catalog", item)
    return store


class Clock:
    def __init__(self, t: float = 1_792_000_000.0) -> None:
        self.t = t
        self._n = itertools.count()

    def __call__(self) -> float:
        self.t += 0.001
        return self.t

    def suffix(self) -> str:
        return f"{next(self._n):012x}"


def make_api(store: MemoryStore | None = None, cognito: FakeCognito | None = None,
             clock: Clock | None = None) -> tuple[AdminApi, MemoryStore, FakeCognito, Clock]:
    store = store if store is not None else seeded_store()
    cognito = cognito if cognito is not None else FakeCognito()
    clock = clock if clock is not None else Clock()
    api = AdminApi(Deps(store=store, cognito=cognito, user_pool_id=POOL_ID, retention_days=90,
                        clock=clock, suffix=clock.suffix))
    return api, store, cognito, clock


# --------------------------------------------------------------------------- events


def claims(groups: Iterable[str] | str | None = ("admin",), *, sub: str = "sub-actor",
           email: str = "actor@example.com", username: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"sub": sub, "email": email, "cognito:username": username or sub}
    if groups is not None:
        out["cognito:groups"] = groups if isinstance(groups, str) else ",".join(groups)
    return out


def event(method: str, path: str, body: Any = None, *, query: Mapping[str, str] | None = None,
          who: Mapping[str, Any] | None = None, raw_body: str | None = None,
          no_claims: bool = False) -> dict[str, Any]:
    ev: dict[str, Any] = {"httpMethod": method, "path": path, "resource": path,
                          "queryStringParameters": dict(query) if query else None,
                          "body": raw_body if raw_body is not None else
                          (json.dumps(body) if body is not None else None),
                          "requestContext": {}}
    if not no_claims:
        ev["requestContext"]["authorizer"] = {"claims": dict(who if who is not None else claims())}
    return ev


def call(api: AdminApi, method: str, path: str, body: Any = None, **kw: Any) -> tuple[int, Any]:
    resp = api.handle(event(method, path, body, **kw))
    return resp["statusCode"], (json.loads(resp["body"]) if resp["body"] else None)
