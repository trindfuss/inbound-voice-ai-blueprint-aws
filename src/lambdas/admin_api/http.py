"""Requests and JSON responses for the admin API (API Gateway REST proxy events)."""

from __future__ import annotations

import base64
import binascii
import json
import re
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

#: Request bodies are small JSON objects; anything bigger is refused.
MAX_BODY_BYTES = 64 * 1024

_HEADERS = {
    "Content-Type": "application/json",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
}


class ApiError(Exception):
    """An expected refusal: becomes a JSON error response. ``message`` is ours, never an
    exception's text; ``field`` names the invalid field."""

    def __init__(self, status: int, error: str, message: str | None = None, *,
                 field: str | None = None, extra: Mapping[str, Any] | None = None) -> None:
        super().__init__(error)
        self.status, self.error, self.message, self.field = status, error, message, field
        self.extra = dict(extra or {})

    def body(self) -> dict[str, Any]:
        out: dict[str, Any] = {"error": self.error}
        if self.message:
            out["message"] = self.message
        if self.field:
            out["field"] = self.field
        out.update(self.extra)
        return out


def response(status: int, body: Mapping[str, Any] | None) -> dict[str, Any]:
    return {"statusCode": status, "headers": dict(_HEADERS),
            "body": json.dumps(body, default=str, sort_keys=True) if body is not None else ""}


def error_response(err: ApiError) -> dict[str, Any]:
    return response(err.status, err.body())


@dataclass(frozen=True, slots=True)
class Caller:
    """The signed-in user, from the Cognito authorizer's claims."""

    sub: str
    email: str | None
    username: str | None
    groups: frozenset[str] = field(default_factory=frozenset)


def parse_groups(raw: Any) -> frozenset[str]:
    """``cognito:groups`` as API Gateway passes it: a list, or a string such as ``admin``,
    ``admin,admin-editor`` or ``[admin admin-editor]``."""
    if isinstance(raw, list | tuple):
        return frozenset(g for g in raw if isinstance(g, str) and g)
    if isinstance(raw, str):
        return frozenset(g for g in re.split(r"[\s,]+", raw.strip().strip("[]")) if g)
    return frozenset()


def caller_of(event: Mapping[str, Any]) -> Caller | None:
    """The caller, or ``None`` without the authorizer's claims (or without ``sub``)."""
    ctx = event.get("requestContext")
    authz = ctx.get("authorizer") if isinstance(ctx, Mapping) else None
    claims = authz.get("claims") if isinstance(authz, Mapping) else None
    if not isinstance(claims, Mapping):
        return None
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub.strip():
        return None
    email = claims.get("email")
    username = claims.get("cognito:username", claims.get("username"))
    return Caller(sub=sub, email=email if isinstance(email, str) and email else None,
                  username=username if isinstance(username, str) and username else None,
                  groups=parse_groups(claims.get("cognito:groups")))


def json_body(event: Mapping[str, Any]) -> dict[str, Any]:
    """The request's JSON object body (400 if missing, malformed, too large or not an object)."""
    raw = event.get("body")
    if raw is None or raw == "":
        raise ApiError(400, "bad_request", "a JSON object body is required")
    if not isinstance(raw, str):
        raise ApiError(400, "bad_request", "the body must be JSON text")
    if event.get("isBase64Encoded"):
        try:
            raw = base64.b64decode(raw, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            raise ApiError(400, "bad_request", "the body is not valid base64 UTF-8") from None
    if len(raw.encode("utf-8")) > MAX_BODY_BYTES:
        raise ApiError(413, "too_large", f"the body is larger than {MAX_BODY_BYTES} bytes")
    try:
        doc = json.loads(raw, parse_constant=_no_constant)
    except (ValueError, RecursionError):
        raise ApiError(400, "bad_json", "the body is not valid JSON") from None
    if not isinstance(doc, dict):
        raise ApiError(400, "bad_request", "the body must be a JSON object")
    return doc


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def query_params(event: Mapping[str, Any]) -> dict[str, str]:
    raw = event.get("queryStringParameters")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)} \
        if isinstance(raw, Mapping) else {}


# --------------------------------------------------------------------------- routing


@dataclass(frozen=True, slots=True)
class Route:
    method: str
    template: str  # e.g. "/tools/{name}"
    name: str
    access: str  # "read" | "write" | "users" | "admin"

    def match(self, path: str) -> dict[str, str] | None:
        want, got = self.template.strip("/").split("/"), path.strip("/").split("/")
        if len(want) != len(got):
            return None
        params: dict[str, str] = {}
        for w, g in zip(want, got, strict=True):
            if w.startswith("{") and w.endswith("}"):
                if not g:
                    return None
                params[w[1:-1]] = urllib.parse.unquote(g)
            elif w != g:
                return None
        return params


#: Prefixes the console's CloudFront behaviour (``/api/*``) or the stage may add.
_PREFIXES = ("/api", "/v1")


def request_path(event: Mapping[str, Any]) -> str:
    """The route path: the API resource template when API Gateway gives one (its path
    parameters come separately), else the request path; without ``/api`` or ``/v1``."""
    resource = event.get("resource")
    path = resource if isinstance(resource, str) and resource and "{proxy+}" not in resource \
        else event.get("path")
    if not isinstance(path, str) or not path:
        return "/"
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if path == p or path.startswith(p + "/"):
                path, changed = path[len(p):] or "/", True
    return path


def path_params(event: Mapping[str, Any], route: Route, path: str) -> dict[str, str] | None:
    """Path parameters: API Gateway's (already decoded) when the path is the template."""
    params = route.match(path)
    if params is None:
        return None
    given = event.get("pathParameters")
    if isinstance(given, Mapping) and path == route.template:
        params = {k: v for k, v in given.items() if isinstance(k, str) and isinstance(v, str)}
        if set(params) != set(route.match(route.template) or {}):
            return None
    return params
