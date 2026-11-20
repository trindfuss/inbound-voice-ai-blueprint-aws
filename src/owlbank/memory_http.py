"""Twilio Memory profile reads over raw HTTP, shared by the auth service and the scripts.

Plain requests through the Twilio client's HTTP layer (Basic auth with the API key), not the SDK
resources: ``Client.request`` drops every ``None`` from the body, and a PATCH needs ``null`` to
remove a trait. ``scripts/_memory_traits.py`` builds its trait-group and write helpers on
:func:`request` and re-exports :func:`lookup_profile` and :func:`get_profile_traits`.

Errors carry the HTTP status and Twilio error code only, never a response body or a credential.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

MEMORY_HOST = "memory.twilio.com"
MEMORY_URL = f"https://{MEMORY_HOST}"
REQUEST_TIMEOUT_S = 30
_JSON = {"Content-Type": "application/json", "Accept": "application/json"}


class MemoryApiError(Exception):
    """A Memory request failed; ``status`` is the HTTP status, ``code`` the Twilio error code."""

    def __init__(self, message: str, *, status: int, code: int | None = None) -> None:
        super().__init__(message)
        self.status, self.code = status, code


def _doc(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


def request(client: Any, method: str, url: str, body: Mapping[str, Any] | None = None, *,
            what: str, allow_404: bool = False, params: Mapping[str, str] | None = None,
            timeout_s: float = REQUEST_TIMEOUT_S) -> tuple[int, Any]:
    """One request; returns (status, parsed JSON or None). Raises :class:`MemoryApiError` on a
    4xx/5xx or a redirect, except a 404 when the caller asks to see it (``allow_404``: the group read
    branches on it)."""
    resp = client.http_client.request(
        method, url, params=dict(params) if params else None, data=dict(body) if body is not None else None,
        headers=dict(_JSON), auth=(client.username, client.password), timeout=timeout_s,
        allow_redirects=False)
    status = int(resp.status_code)
    doc = _doc(resp.text)
    if status == 404 and allow_404:
        return status, doc
    if status >= 400 or status in (301, 302, 307, 308):
        code = doc.get("code") if isinstance(doc, dict) else None
        code = code if isinstance(code, int) and not isinstance(code, bool) else None
        raise MemoryApiError(f"Memory {what} failed (HTTP {status}, Twilio error {code})",
                             status=status, code=code)
    return status, doc


def store_url(store_id: str) -> str:
    return f"{MEMORY_URL}/v1/Stores/{quote(store_id, safe='')}"


def profile_url(store_id: str, profile_id: str) -> str:
    return f"{store_url(store_id)}/Profiles/{quote(profile_id, safe='')}"


def lookup_profile_by(client: Any, store_id: str, id_type: str, value: str, *,
                      timeout_s: float = REQUEST_TIMEOUT_S) -> list[str]:
    """Profile IDs for an identifier of ``id_type`` (``email``, ``phone``...; ``Profiles/Lookup``); empty
    if there is none. More than one means two profiles share the identifier: the caller reports that, it
    never guesses."""
    _, doc = request(client, "POST", f"{store_url(store_id)}/Profiles/Lookup",
                     {"idType": id_type, "value": value}, what="profile lookup", timeout_s=timeout_s)
    profiles = doc.get("profiles") if isinstance(doc, dict) else None
    return [p for p in profiles if isinstance(p, str)] if isinstance(profiles, list) else []


def lookup_profile(client: Any, store_id: str, email: str, *,
                   timeout_s: float = REQUEST_TIMEOUT_S) -> list[str]:
    """Profile IDs for an email: :func:`lookup_profile_by` with ``idType`` ``email``."""
    return lookup_profile_by(client, store_id, "email", email, timeout_s=timeout_s)


def get_profile_traits(client: Any, store_id: str, profile_id: str, groups: Sequence[str] | None = None, *,
                       timeout_s: float = REQUEST_TIMEOUT_S) -> dict[str, Any]:
    """The profile's traits by group. ``groups`` names the trait groups to fetch (``traitGroups``);
    ``None`` fetches every group, as the bridge's TAC fetch and the scripts do; an empty
    sequence is a ``ValueError`` (a programming error, not "every group"). A 308
    (merged profile) raises :class:`MemoryApiError` with ``status`` 308; a 404 raises too (a profile
    the lookup named is gone)."""
    if isinstance(groups, str):
        raise TypeError("groups must be a sequence of group names, not a str")
    if groups is not None and not groups:
        raise ValueError("groups must name at least one group, or be None for every group")
    params = {"traitGroups": ",".join(groups)} if groups else None
    _, doc = request(client, "GET", profile_url(store_id, profile_id), what="profile read", params=params,
                     timeout_s=timeout_s)
    traits = doc.get("traits") if isinstance(doc, dict) else None
    return dict(traits) if isinstance(traits, dict) else {}


# --------------------------------------------------------------------------- observations and summaries
# Both collections are written in batches (a 202 for summaries: accepted, shown later) and read by
# token pagination: items under the collection key, the next page's token in ``meta.nextToken``.

LIST_PAGE_SIZE = 100
_MAX_PAGES = 50  # a bound on a runaway token chain


def _items_url(store_id: str, profile_id: str, collection: str) -> str:
    return f"{profile_url(store_id, profile_id)}/{collection}"


def _create(client: Any, url: str, key: str, items: Sequence[Mapping[str, Any]], *, what: str,
            timeout_s: float) -> Any:
    _, doc = request(client, "POST", url, {key: [dict(i) for i in items]}, what=what, timeout_s=timeout_s)
    return doc


def _list(client: Any, url: str, key: str, *, what: str, timeout_s: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    token: str | None = None
    for _ in range(_MAX_PAGES):
        params = {"pageSize": str(LIST_PAGE_SIZE), **({"pageToken": token} if token else {})}
        _, doc = request(client, "GET", url, what=what, params=params, timeout_s=timeout_s)
        items = doc.get(key) if isinstance(doc, dict) else None
        if isinstance(items, list):
            out.extend(i for i in items if isinstance(i, dict))
        meta = doc.get("meta") if isinstance(doc, dict) else None
        nxt = meta.get("nextToken") if isinstance(meta, dict) else None
        if not isinstance(nxt, str) or not nxt:
            break
        token = nxt
    return out


def create_observations(client: Any, store_id: str, profile_id: str,
                        observations: Sequence[Mapping[str, Any]], *,
                        timeout_s: float = REQUEST_TIMEOUT_S) -> Any:
    """``POST .../Observations`` with ``{"observations": [{content, source, occurredAt,
    conversationIds?}]}``; returns the parsed response."""
    return _create(client, _items_url(store_id, profile_id, "Observations"), "observations", observations,
                   what="observation create", timeout_s=timeout_s)


def create_summaries(client: Any, store_id: str, profile_id: str,
                     summaries: Sequence[Mapping[str, Any]], *,
                     timeout_s: float = REQUEST_TIMEOUT_S) -> Any:
    """``POST .../ConversationSummaries`` with ``{"summaries": [{content, conversationId, occurredAt,
    source}]}``. Memory accepts these asynchronously: list until they show."""
    return _create(client, _items_url(store_id, profile_id, "ConversationSummaries"), "summaries", summaries,
                   what="summary create", timeout_s=timeout_s)


def list_observations(client: Any, store_id: str, profile_id: str, *,
                      timeout_s: float = REQUEST_TIMEOUT_S) -> list[dict[str, Any]]:
    """Every observation of the profile (all pages)."""
    return _list(client, _items_url(store_id, profile_id, "Observations"), "observations",
                 what="observation list", timeout_s=timeout_s)


def list_summaries(client: Any, store_id: str, profile_id: str, *,
                   timeout_s: float = REQUEST_TIMEOUT_S) -> list[dict[str, Any]]:
    """Every conversation summary of the profile (all pages)."""
    return _list(client, _items_url(store_id, profile_id, "ConversationSummaries"), "summaries",
                 what="summary list", timeout_s=timeout_s)


def delete_observation(client: Any, store_id: str, profile_id: str, observation_id: str, *,
                       timeout_s: float = REQUEST_TIMEOUT_S) -> None:
    request(client, "DELETE", f"{_items_url(store_id, profile_id, 'Observations')}/{quote(observation_id, safe='')}",
            what="observation delete", timeout_s=timeout_s)


def delete_summary(client: Any, store_id: str, profile_id: str, summary_id: str, *,
                   timeout_s: float = REQUEST_TIMEOUT_S) -> None:
    request(client, "DELETE",
            f"{_items_url(store_id, profile_id, 'ConversationSummaries')}/{quote(summary_id, safe='')}",
            what="summary delete", timeout_s=timeout_s)


__all__ = ["LIST_PAGE_SIZE", "MEMORY_HOST", "MEMORY_URL", "REQUEST_TIMEOUT_S", "MemoryApiError",
           "create_observations", "create_summaries", "delete_observation", "delete_summary",
           "get_profile_traits", "list_observations", "list_summaries", "lookup_profile", "profile_url",
           "request", "store_url"]
