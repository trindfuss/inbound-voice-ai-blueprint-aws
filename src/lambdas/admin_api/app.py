"""The admin API's routing and access checks (.5a, .15, .16).

**Auth.** API Gateway's Cognito authorizer verifies the token and passes its claims in
``requestContext.authorizer.claims``; the API reads ``sub``, ``email``, ``cognito:username``
and ``cognito:groups``. Without claims (or without ``sub``) every route is 401, whatever the
path. Reads need any console group; config writes need ``admin-editor`` or ``admin``; user
routes need ``admin``, and so do sending ``generative_enabled`` and ``PUT /redaction``
 (:data:`ADMIN`). A
signed-in user without the group gets 403. The checks are here, not
only in the console.

The token's groups are what they were at sign-in, so the ``users`` and ``admin`` routes (and
``generative_enabled``) also need the caller to be enabled and in ``admin`` now, as Cognito
says (:mod:`.membership`, cached up to 60 s per user). A caller missing from either, or
disabled, is refused (403).

**Routes** (paths may carry the ``/api`` prefix of the console's CloudFront behaviour):

======  ===============================  ======  =========================================
Method  Path                             Access  Body / query
======  ===============================  ======  =========================================
GET     /tools                           read
GET     /tools/{name}                    read
PUT     /tools/{name}                    write   ``{rev, min_level?, confirm?, mask?}``
GET     /masking                         read
PUT     /masking                         write   ``{rev, masking: {...}}``
GET     /prompts                         read    ``?prefix=mask.`` or ``arg.`` (or longer)
PUT     /prompts/{key}                   write   ``{rev, texts: {lang: text}}``
GET     /placeholders/{tool}             read
GET     /audit                           read    ``?item=policy#tool#x&limit=50``
POST    /restore                         write   ``{audit_pk, audit_sk, rev}``
GET     /slots                           read
PUT     /slots/resolution                write   ``{rev, default_mode?, flows?, resolver_budget_ms?,
                                                 generative_enabled?}``; ``generative_enabled``
                                                 is admin only (403 otherwise)
POST    /slots/preview                   read    ``{slot_resolution?: {...}, slots?: {name: {...}}}``;
                                                 no write
PUT     /slots/{name}                    write   ``{rev, sensitive?, keypad_only?, barge_in?,
                                                 mode?, confirm?, options?}``
GET     /redaction                       read
PUT     /redaction                       admin   ``{rev, enabled?, entity_types?, min_confidence?,
                                                 mask?, on_failure?}``
GET     /users                           users
POST    /users                           users   ``{email, group}``
PATCH   /users/{username}                users   ``{group?, enabled?}``
POST    /users/{username}/resend         users
DELETE  /users/{username}                users
======  ===============================  ======  =========================================

Responses are JSON with ``Cache-Control: no-store``. Errors are ``{"error", "message"?,
"field"?}`` (409 adds ``current``); an unexpected exception is a 500 ``{"error": "internal"}``
and is logged with :func:`owlbank.obs.log_caught` (class and stack, never the message).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from owlbank import obs
from owlbank.slot_policy import Mode
from owlbank.store import Store

from .audit import AuditClock
from .config_routes import ConfigRoutes
from .http import (
    ApiError,
    Caller,
    Route,
    caller_of,
    error_response,
    json_body,
    path_params,
    query_params,
    request_path,
    response,
)
from .membership import GroupCache, Membership
from .redaction_routes import RedactionRoutes
from .slot_routes import SlotRoutes
from .users import CognitoClient, UserRoutes

log = logging.getLogger(__name__)

READ_GROUPS: frozenset[str] = frozenset({"admin", "admin-editor", "admin-viewer"})
WRITE_GROUPS: frozenset[str] = frozenset({"admin", "admin-editor"})
USER_GROUPS: frozenset[str] = frozenset({"admin"})
#: Admin-only settings (``generative_enabled``; ``PUT /redaction``): the user routes'
#: group, not a new one.
ADMIN: frozenset[str] = USER_GROUPS
ACCESS: Mapping[str, frozenset[str]] = {"read": READ_GROUPS, "write": WRITE_GROUPS, "users": USER_GROUPS,
                                        "admin": ADMIN}
#: Access levels checked against the caller's groups in Cognito too, not only the token's.
SERVER_CHECKED: frozenset[str] = frozenset({"users", "admin"})

ROUTES: tuple[Route, ...] = (
    Route("GET", "/tools", "list_tools", "read"),
    Route("GET", "/tools/{name}", "get_tool", "read"),
    Route("PUT", "/tools/{name}", "put_tool", "write"),
    Route("GET", "/masking", "get_masking", "read"),
    Route("PUT", "/masking", "put_masking", "write"),
    Route("GET", "/prompts", "list_prompts", "read"),
    Route("PUT", "/prompts/{key}", "put_prompt", "write"),
    Route("GET", "/placeholders/{tool}", "get_placeholders", "read"),
    Route("GET", "/audit", "list_audit", "read"),
    Route("POST", "/restore", "restore", "write"),
    Route("GET", "/slots", "list_slots", "read"),
    # Before ``/slots/{name}``: the first route whose path and method match wins.
    Route("PUT", "/slots/resolution", "put_resolution", "write"),
    Route("POST", "/slots/preview", "preview_slots", "read"),
    Route("PUT", "/slots/{name}", "put_slot", "write"),
    Route("GET", "/redaction", "get_redaction", "read"),
    Route("PUT", "/redaction", "put_redaction", "admin"),
    Route("GET", "/users", "list_users", "users"),
    Route("POST", "/users", "create_user", "users"),
    Route("PATCH", "/users/{username}", "update_user", "users"),
    Route("POST", "/users/{username}/resend", "resend_invitation", "users"),
    Route("DELETE", "/users/{username}", "delete_user", "users"),
)


@dataclass(frozen=True, slots=True)
class Deps:
    """Everything the API touches. ``store`` holds the policy, catalog and audit tables."""

    store: Store
    cognito: CognitoClient
    user_pool_id: str
    retention_days: int = 90
    clock: Callable[[], float] = time.time
    suffix: Callable[[], str] | None = None
    #: The deployment's ``acceptSpokenFactors`` (``AdminConfig.accept_spoken_factors``), shown by
    #: ``GET /slots``.
    accept_spoken_factors: bool = True
    #: The deployment's slot mode ceiling (``AdminConfig.slot_mode_ceiling``).
    slot_mode_ceiling: Mode = "assisted"


class AdminApi:
    def __init__(self, deps: Deps) -> None:
        clock = AuditClock(now=deps.clock) if deps.suffix is None \
            else AuditClock(now=deps.clock, suffix=deps.suffix)
        self.config = ConfigRoutes(deps.store, clock=clock, retention_days=deps.retention_days)
        self.slots = SlotRoutes(deps.store, clock=clock, retention_days=deps.retention_days,
                                accept_spoken_factors=deps.accept_spoken_factors,
                                slot_mode_ceiling=deps.slot_mode_ceiling)
        self.redaction = RedactionRoutes(deps.store, clock=clock, retention_days=deps.retention_days)
        self.groups = GroupCache(self._fetch_membership, clock=deps.clock)
        self.users = UserRoutes(deps.cognito, deps.store, user_pool_id=deps.user_pool_id,
                                clock=clock, retention_days=deps.retention_days,
                                on_change=self.groups.forget)

    def _fetch_membership(self, username: str) -> Membership:
        return self.users.current_membership(username)

    def _refusal(self, caller: Caller, groups: frozenset[str]) -> str | None:
        """``None`` if the caller's token *and* Cognito, now, put them in one of ``groups``;
        otherwise why not (``token_groups``, ``cognito_disabled`` or ``cognito_groups``)."""
        if not caller.groups & groups:
            return "token_groups"
        now = self.groups.groups(caller.username or caller.sub)
        if not now.enabled:
            return "cognito_disabled"
        return None if now.groups & groups else "cognito_groups"

    @staticmethod
    def resolve(event: Mapping[str, Any]) -> tuple[Route, dict[str, str]] | None:
        """The route and path parameters for an event, ignoring the method; ``None`` if no
        path matches. Raises a 405 :class:`ApiError` for a known path with another method."""
        method = event.get("httpMethod")
        path = request_path(event)
        allowed: list[str] = []
        for route in ROUTES:
            params = path_params(event, route, path)
            if params is None:
                continue
            if route.method == method:
                return route, params
            allowed.append(route.method)
        if allowed:
            raise ApiError(405, "method_not_allowed", f"allowed: {', '.join(sorted(set(allowed)))}")
        return None

    def handle(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Handle one API Gateway REST proxy event. Never raises."""
        route_name = "unknown"
        try:
            caller = caller_of(event)
            if caller is None:  # before routing: no route, real or not, answers without a token
                raise ApiError(401, "unauthorized", "sign in first")
            found = self.resolve(event)
            if found is None:
                raise ApiError(404, "not_found", "no such route")
            route, params = found
            route_name = route.name
            if not caller.groups & ACCESS[route.access]:
                obs.emit(log, logging.WARNING, "admin_access_denied", route=route.name,
                         access=route.access, actor_sub=caller.sub)
                raise ApiError(403, "forbidden", "your group can't do this")
            refusal = self._refusal(caller, ACCESS[route.access]) \
                if route.access in SERVER_CHECKED else None
            if refusal is not None:
                obs.emit(log, logging.WARNING, "admin_access_denied", route=route.name,
                         access=route.access, actor_sub=caller.sub, reason=refusal)
                raise ApiError(403, "forbidden", "your group can't do this")
            status, body = self._dispatch(route, params, event, caller)
            return response(status, body)
        except ApiError as err:
            if err.status >= 500:
                obs.emit(log, logging.ERROR, "admin_api_error", route=route_name, status=err.status,
                         error=err.error)
            return error_response(err)
        except Exception as exc:  # never echo internals
            obs.log_caught(log, "admin api: unexpected error", exc, level=logging.ERROR,
                           route=route_name, decision="return_500")
            return response(500, {"error": "internal"})

    def _dispatch(self, route: Route, params: Mapping[str, str], event: Mapping[str, Any],
                  caller: Caller) -> tuple[int, dict[str, Any]]:
        c, u, q = self.config, self.users, query_params(event)
        name = route.name
        if name == "list_tools":
            return 200, c.list_tools()
        if name == "get_tool":
            return 200, c.get_tool(params["name"])
        if name == "put_tool":
            return 200, c.put_tool(caller, params["name"], json_body(event))
        if name == "get_masking":
            return 200, c.get_masking()
        if name == "put_masking":
            return 200, c.put_masking(caller, json_body(event))
        if name == "list_prompts":
            return 200, c.list_prompts(q.get("prefix"))
        if name == "put_prompt":
            return 200, c.put_prompt(caller, params["key"], json_body(event))
        if name == "get_placeholders":
            return 200, c.get_placeholders(params["tool"])
        if name == "list_audit":
            return 200, c.list_audit(q.get("item"), q.get("limit"))
        if name == "restore":
            return 200, c.restore(caller, json_body(event))
        if name == "list_slots":
            return 200, self.slots.list_slots()
        if name == "put_resolution":
            body = json_body(event)
            refusal = self._refusal(caller, ADMIN) if "generative_enabled" in body else None
            if refusal is not None:
                obs.emit(log, logging.WARNING, "admin_access_denied", route=route.name,
                         access="admin", actor_sub=caller.sub, field="generative_enabled",
                         reason=refusal)
                raise ApiError(403, "forbidden", "only an admin can change generative_enabled",
                               field="generative_enabled")
            return 200, self.slots.put_resolution(caller, body)
        if name == "preview_slots":
            return 200, self.slots.preview(json_body(event))
        if name == "put_slot":
            return 200, self.slots.put_slot(caller, params["name"], json_body(event))
        if name == "get_redaction":
            return 200, self.redaction.get_redaction()
        if name == "put_redaction":
            return 200, self.redaction.put_redaction(caller, json_body(event))
        if name == "list_users":
            return 200, u.list_users()
        if name == "create_user":
            return 201, u.create_user(caller, json_body(event))
        if name == "update_user":
            return 200, u.update_user(caller, params["username"], json_body(event))
        if name == "resend_invitation":
            return 200, u.resend(caller, params["username"])
        if name == "delete_user":
            return 200, u.delete_user(caller, params["username"])
        raise ApiError(404, "not_found", "no such route")  # pragma: no cover - ROUTES and this agree
