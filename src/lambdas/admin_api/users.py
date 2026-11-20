"""User management for admins.

Routes (the caller must be in ``admin``; :mod:`.app` checks): list, create (Cognito emails a
temporary password: ``AdminCreateUser`` with ``DesiredDeliveryMediums=EMAIL``), change group,
disable and enable, resend the invitation (``MessageAction=RESEND``, only while the user is
``FORCE_CHANGE_PASSWORD``) and delete. A user is in exactly one of :data:`GROUPS`.

**The last admin**. Before a demotion, a disable or a delete of an enabled admin, the
enabled members of ``admin`` are counted (``ListUsersInGroup``); if the target is the only
one, the change is refused (409 ``last_admin``). An admin can't delete themselves (403
``self_delete``; matched by the token's ``sub`` or ``cognito:username``).

**Audit.** Each change writes an audit entry (``security_change`` true, the target's email,
old and new group and status; never a password). Cognito isn't transactional with DynamoDB,
so the entry is written right after the Cognito call succeeds; if that write fails the change
stands, the response says ``audited: false`` and an ERROR (``admin_audit_write_failed``) is
logged for an operator to reconcile. The last-admin count and the change are two calls, so
two admins demoting each other at the same moment could both pass the check; with a handful
of operators this is accepted (documented limitation).

**Sign-out.** Disabling a user, changing their group and deleting them also call
``AdminUserGlobalSignOut`` (before the disable and the delete, after the group change), which
revokes the user's refresh tokens so the console can't get a new token with the old groups.
An ID token already issued stays valid until it expires; :mod:`.membership` closes that gap for
the admin-only routes. A failed sign-out is logged (ERROR ``admin_user_sign_out_failed``) and
never undoes or blocks the change; the response says ``signed_out: false``.

Cognito actions used: ``ListUsers``, ``ListUsersInGroup``, ``AdminGetUser``,
``AdminListGroupsForUser``, ``AdminCreateUser``, ``AdminAddUserToGroup``,
``AdminRemoveUserFromGroup``, ``AdminDisableUser``, ``AdminEnableUser``, ``AdminDeleteUser``,
``AdminUserGlobalSignOut``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any, Protocol

from owlbank import obs
from owlbank.store import Store

from . import audit
from .audit import AuditClock
from .http import ApiError, Caller
from .membership import Membership

log = logging.getLogger(__name__)

ADMIN = "admin"
GROUPS: tuple[str, ...] = ("admin", "admin-editor", "admin-viewer")
PENDING_STATUS = "FORCE_CHANGE_PASSWORD"
_EMAIL = re.compile(r"[^@\s{}<>]{1,64}@[A-Za-z0-9.-]{1,189}\.[A-Za-z]{2,63}")
_USERNAME = re.compile(r"[\w.@+-]{1,128}")
_LIST_LIMIT = 60


class CognitoClient(Protocol):
    """The ``cognito-idp`` calls the API makes (a boto3 client satisfies it)."""

    def list_users(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def list_users_in_group(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def admin_get_user(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def admin_list_groups_for_user(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def admin_create_user(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def admin_add_user_to_group(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def admin_remove_user_from_group(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def admin_disable_user(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def admin_enable_user(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def admin_delete_user(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def admin_user_global_sign_out(self, **kwargs: Any) -> Mapping[str, Any]: ...


#: Cognito error codes and the response each becomes (anything else: 502).
_COGNITO_ERRORS: Mapping[str, tuple[int, str]] = {
    "UserNotFoundException": (404, "not_found"),
    "UsernameExistsException": (409, "user_exists"),
    "AliasExistsException": (409, "user_exists"),
    "InvalidParameterException": (400, "invalid"),
    "TooManyRequestsException": (429, "throttled"),
    "LimitExceededException": (429, "throttled"),
    "UnsupportedUserStateException": (409, "unsupported_user_state"),
}


def _attrs(raw: Any) -> dict[str, str]:
    return {a["Name"]: a.get("Value", "") for a in raw or [] if isinstance(a, Mapping) and "Name" in a}


def _when(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else (value if isinstance(value, str) else None)


class UserRoutes:
    def __init__(self, cognito: CognitoClient, store: Store, *, user_pool_id: str,
                 clock: AuditClock, retention_days: int,
                 on_change: Callable[[str], None] | None = None) -> None:
        self.cognito, self.store, self.pool = cognito, store, user_pool_id
        self.clock, self.retention_days = clock, retention_days
        #: Called with the username and the ``sub`` of each user whose groups or status change
        #: (the API drops their cached groups).
        self.on_change = on_change

    # ------------------------------------------------------------------ Cognito calls

    def _call(self, op: str, **kwargs: Any) -> Mapping[str, Any]:
        fn = getattr(self.cognito, op)
        try:
            with obs.timed_call("cognito-idp", op):
                return fn(UserPoolId=self.pool, **kwargs)
        except Exception as exc:
            code = obs.error_code_of(exc)
            status, error = _COGNITO_ERRORS.get(code or "", (502, "cognito_error"))
            if status == 502:
                obs.log_caught(log, "admin api: cognito call failed", exc, level=logging.ERROR,
                               operation=op, decision="return_502")
            raise ApiError(status, error, None if status == 502 else {
                "not_found": "no such user", "user_exists": "a user with that email exists",
                "invalid": "Cognito refused the request's parameters",
                "throttled": "too many requests; try again shortly",
                "unsupported_user_state": "the user's state doesn't allow this",
            }[error]) from None

    def _group_members(self, group: str) -> list[Mapping[str, Any]]:
        users: list[Mapping[str, Any]] = []
        token: str | None = None
        while True:
            resp = self._call("list_users_in_group", GroupName=group, Limit=_LIST_LIMIT,
                              **({"NextToken": token} if token else {}))
            users += list(resp.get("Users") or [])
            token = resp.get("NextToken")
            if not token:
                return users

    def _all_users(self) -> list[Mapping[str, Any]]:
        users: list[Mapping[str, Any]] = []
        token: str | None = None
        while True:
            resp = self._call("list_users", Limit=_LIST_LIMIT,
                              **({"PaginationToken": token} if token else {}))
            users += list(resp.get("Users") or [])
            token = resp.get("PaginationToken")
            if not token:
                return users

    def _groups_of(self, username: str) -> list[str]:
        resp = self._call("admin_list_groups_for_user", Username=username, Limit=_LIST_LIMIT)
        return [g["GroupName"] for g in resp.get("Groups") or [] if isinstance(g, Mapping)
                and g.get("GroupName") in GROUPS]

    def current_membership(self, username: str) -> Membership:
        """Whether ``username`` (a username or ``sub``) is enabled now and the console groups
        they are in; no groups for a user that doesn't exist (deleted), and none looked up for
        a disabled one."""
        if not _USERNAME.fullmatch(username):
            return Membership(frozenset())
        try:
            if not self._call("admin_get_user", Username=username).get("Enabled", True):
                return Membership(frozenset(), enabled=False)
            return Membership(frozenset(self._groups_of(username)))
        except ApiError as err:
            if err.error == "not_found":
                return Membership(frozenset())
            raise

    def _sign_out(self, target: Mapping[str, Any]) -> bool:
        """Revoke the user's refresh tokens; ``False`` (and an ERROR) if Cognito refused."""
        try:
            self._call("admin_user_global_sign_out", Username=target["username"])
        except ApiError as err:
            obs.emit(log, logging.ERROR, "admin_user_sign_out_failed", target_id=target["username"],
                     error=err.error, decision="change_stands")
            return False
        return True

    def _changed(self, target: Mapping[str, Any]) -> None:
        if self.on_change is not None:
            for key in {target.get("username"), target.get("sub")}:
                if isinstance(key, str) and key:
                    self.on_change(key)

    def _get(self, username: str) -> dict[str, Any]:
        if not _USERNAME.fullmatch(username):
            raise ApiError(404, "not_found", "no such user")
        resp = self._call("admin_get_user", Username=username)
        attrs = _attrs(resp.get("UserAttributes"))
        groups = self._groups_of(username)
        return {"username": resp.get("Username", username), "email": attrs.get("email"),
                "sub": attrs.get("sub"), "status": resp.get("UserStatus"),
                "enabled": bool(resp.get("Enabled", True)), "groups": groups,
                "group": groups[0] if len(groups) == 1 else (ADMIN if ADMIN in groups else
                                                             (groups[0] if groups else None)),
                "created": _when(resp.get("UserCreateDate"))}

    @staticmethod
    def _view(u: Mapping[str, Any]) -> dict[str, Any]:
        return {k: u.get(k) for k in ("username", "email", "status", "enabled", "group", "groups", "created")}

    def _enabled_admins(self) -> set[str]:
        return {u["Username"] for u in self._group_members(ADMIN)
                if u.get("Enabled", True) and isinstance(u.get("Username"), str)}

    def _refuse_last_admin(self, target: Mapping[str, Any]) -> None:
        if ADMIN not in target["groups"] or not target["enabled"]:
            return
        admins = self._enabled_admins()
        if target["username"] in admins and len(admins) <= 1:
            raise ApiError(409, "last_admin", "this is the only enabled admin; make another user an "
                                              "admin first")

    @staticmethod
    def _is_self(caller: Caller, target: Mapping[str, Any]) -> bool:
        return (target.get("sub") is not None and target.get("sub") == caller.sub) \
            or target.get("username") == caller.sub \
            or (caller.username is not None and target.get("username") == caller.username)

    # ------------------------------------------------------------------ audit

    def _audit(self, caller: Caller, *, username: str, action: str,
               changes: list[dict[str, Any]]) -> bool:
        item = audit.entry(pk=audit.user_pk(username), action=action, kind="user", target_id=username,
                           actor=caller, changes=changes, security_change=True,
                           now=self.clock.now(), suffix=self.clock.suffix(),
                           retention_days=self.retention_days)
        try:
            self.store.put(audit.AUDIT_TABLE, item, condition="not_exists")
        except Exception as exc:
            obs.log_caught(log, "admin_audit_write_failed", exc, level=logging.ERROR,
                           operation="audit_user_change", action=action, target_id=username,
                           actor_sub=caller.sub, decision="change_stands_unaudited")
            return False
        obs.emit(log, logging.INFO, "admin_user_change", action=action, target_id=username,
                 actor_sub=caller.sub, fields=[c["field"] for c in changes])
        return True

    # ------------------------------------------------------------------ routes

    def list_users(self) -> dict[str, Any]:
        members = {g: {u.get("Username") for u in self._group_members(g)} for g in GROUPS}
        out = []
        for u in self._all_users():
            name = u.get("Username")
            attrs = _attrs(u.get("Attributes"))
            groups = [g for g in GROUPS if name in members[g]]
            out.append({"username": name, "email": attrs.get("email"), "status": u.get("UserStatus"),
                        "enabled": bool(u.get("Enabled", True)), "groups": groups,
                        "group": groups[0] if groups else None,
                        "created": _when(u.get("UserCreateDate"))})
        out.sort(key=lambda v: (str(v.get("email") or ""), str(v.get("username") or "")))
        return {"users": out, "groups": list(GROUPS)}

    def create_user(self, caller: Caller, body: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(str(k) for k in body if k not in ("email", "group"))
        if unknown:
            raise ApiError(400, "invalid", "unknown field", field=unknown[0])
        email, group = body.get("email"), body.get("group")
        if not isinstance(email, str) or len(email) > 254 or not _EMAIL.fullmatch(email.strip()):
            raise ApiError(400, "invalid", "must be an email address", field="email")
        if group not in GROUPS:
            raise ApiError(400, "invalid", f"must be one of {list(GROUPS)}", field="group")
        email = email.strip()
        resp = self._call("admin_create_user", Username=email,
                          UserAttributes=[{"Name": "email", "Value": email},
                                          {"Name": "email_verified", "Value": "true"}],
                          DesiredDeliveryMediums=["EMAIL"])
        user = resp.get("User") or {}
        username = user.get("Username") or email
        try:
            self._call("admin_add_user_to_group", Username=username, GroupName=group)
        except ApiError:
            try:  # don't leave a user with no group
                self._call("admin_delete_user", Username=username)
                rolled_back = True
            except ApiError:
                rolled_back = False
            self._audit(caller, username=username, action="user_create_failed",
                        changes=[{"field": "email", "old": None, "new": email},
                                 {"field": "deleted", "old": None, "new": rolled_back}])
            raise ApiError(502, "cognito_error", "the user was created but couldn't be put in the "
                           "group" + ("; it was removed again" if rolled_back else "")) from None
        audited = self._audit(caller, username=username, action="user_create", changes=[
            {"field": "email", "old": None, "new": email},
            {"field": "group", "old": None, "new": group},
            {"field": "enabled", "old": None, "new": True}])
        view = {"username": username, "email": email, "status": user.get("UserStatus"),
                "enabled": bool(user.get("Enabled", True)), "group": group, "groups": [group],
                "created": _when(user.get("UserCreateDate"))}
        return {"user": view, "audited": audited}

    def update_user(self, caller: Caller, username: str, body: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(str(k) for k in body if k not in ("group", "enabled"))
        if unknown:
            raise ApiError(400, "invalid", "unknown field", field=unknown[0])
        if "group" not in body and "enabled" not in body:
            raise ApiError(400, "invalid", "send group and/or enabled")
        group, enabled = body.get("group"), body.get("enabled")
        if "group" in body and group not in GROUPS:
            raise ApiError(400, "invalid", f"must be one of {list(GROUPS)}", field="group")
        if "enabled" in body and not isinstance(enabled, bool):
            raise ApiError(400, "invalid", "must be true or false", field="enabled")
        target = self._get(username)
        username = target["username"]
        demote = "group" in body and group != ADMIN and ADMIN in target["groups"]
        disable = enabled is False and target["enabled"]
        if demote or disable:
            self._refuse_last_admin(target)
        changes: list[dict[str, Any]] = []
        signed_out: bool | None = None
        if "group" in body and target["groups"] != [group]:
            try:
                if group not in target["groups"]:
                    self._call("admin_add_user_to_group", Username=username, GroupName=group)
                for g in target["groups"]:
                    if g != group:
                        self._call("admin_remove_user_from_group", Username=username, GroupName=g)
            finally:
                self._changed(target)
            changes.append({"field": "group", "old": target["group"], "new": group})
            signed_out = self._sign_out(target)
        if "enabled" in body and enabled != target["enabled"]:
            if not enabled and signed_out is None:
                signed_out = self._sign_out(target)
            try:
                self._call("admin_enable_user" if enabled else "admin_disable_user", Username=username)
            finally:
                self._changed(target)
            changes.append({"field": "enabled", "old": target["enabled"], "new": enabled})
        if not changes:
            return {"changed": False, "user": self._view(target)}
        changes.append({"field": "email", "old": target["email"], "new": target["email"]})
        audited = self._audit(caller, username=username, action="user_update", changes=changes)
        new = {**target}
        if "group" in body:
            new["group"], new["groups"] = group, [group]
        if "enabled" in body:
            new["enabled"] = enabled
        out = {"changed": True, "user": self._view(new), "audited": audited}
        if signed_out is not None:
            out["signed_out"] = signed_out
        return out

    def resend(self, caller: Caller, username: str) -> dict[str, Any]:
        target = self._get(username)
        if target["status"] != PENDING_STATUS:
            raise ApiError(409, "not_pending", "the user has already signed in; there is no "
                                               "invitation to resend")
        self._call("admin_create_user", Username=target["username"], MessageAction="RESEND",
                   DesiredDeliveryMediums=["EMAIL"])
        audited = self._audit(caller, username=target["username"], action="user_resend",
                              changes=[{"field": "invitation", "old": None, "new": "resent"},
                                       {"field": "email", "old": target["email"], "new": target["email"]}])
        return {"resent": True, "user": self._view(target), "audited": audited}

    def delete_user(self, caller: Caller, username: str) -> dict[str, Any]:
        target = self._get(username)
        if self._is_self(caller, target):
            raise ApiError(403, "self_delete", "you can't delete your own user")
        self._refuse_last_admin(target)
        signed_out = self._sign_out(target)  # before the delete: afterwards there is no user
        try:
            self._call("admin_delete_user", Username=target["username"])
        finally:
            self._changed(target)
        audited = self._audit(caller, username=target["username"], action="user_delete", changes=[
            {"field": "email", "old": target["email"], "new": None},
            {"field": "group", "old": target["group"], "new": None},
            {"field": "enabled", "old": target["enabled"], "new": None}])
        return {"deleted": target["username"], "audited": audited, "signed_out": signed_out}
