"""Twilio Memory trait helpers shared by ``twilio_setup.py`` and the demo trait writer.

Plain requests through the Twilio client's HTTP layer (Basic auth with the API key, as
``twilio_setup.py`` does for Flex), not the SDK resources: ``Client.request`` drops every ``None``
from the body, and a PATCH needs ``null`` to remove a trait. The request itself, the profile lookup
and the profile read live in :mod:`owlbank.memory_http` (the auth service reads profiles too);
here they raise :class:`MemoryApiError`, a :class:`ScriptError`, so the scripts print
its message.

Errors carry the HTTP status and Twilio error code only, never a response body or a credential.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote, urlsplit

from owlbank import memory_http
from owlbank.memory_http import MEMORY_HOST, MEMORY_URL
from owlbank.web_activity import GROUP, TRAITS
from scripts._common import ScriptError

#: Creating a group answers 202 with an operation; live it was readable within 2 s.
GROUP_TIMEOUT_S = 60
GROUP_POLL_S = 2
#: OperationStatus in twilio-oai twilio_memory_v1.json: PENDING, RUNNING, CANCELLED, COMPLETED, FAILED.
_OP_DONE = frozenset({"completed"})
_OP_FAILED = frozenset({"failed", "cancelled"})


class MemoryApiError(ScriptError, memory_http.MemoryApiError):
    """A Memory request failed; ``status`` is the HTTP status, ``code`` the Twilio error code."""

    def __init__(self, message: str, *, status: int, code: int | None = None) -> None:
        memory_http.MemoryApiError.__init__(self, message, status=status, code=code)


def _script_errors[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """``fn`` raising :class:`MemoryApiError` (a :class:`ScriptError`) for the shared module's error."""
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except MemoryApiError:
            raise
        except memory_http.MemoryApiError as exc:
            raise MemoryApiError(str(exc), status=exc.status, code=exc.code) from None
    return wrapper


_call = _script_errors(memory_http.request)
lookup_profile = _script_errors(memory_http.lookup_profile)
get_profile_traits = _script_errors(memory_http.get_profile_traits)
_profile_url = memory_http.profile_url


def _group_url(store_id: str, group: str) -> str:
    return f"{MEMORY_URL}/v1/ControlPlane/Stores/{quote(store_id, safe='')}/TraitGroups/{quote(group, safe='')}"


def _operation_url(status_url: str) -> str:
    """The operation's absolute URL; credentials go only to Memory's own host."""
    url = status_url if urlsplit(status_url).scheme else MEMORY_URL + "/" + status_url.lstrip("/")
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname != MEMORY_HOST:
        raise ScriptError("Memory answered with an operation URL outside memory.twilio.com; not followed")
    return url


def ensure_trait_group(client: Any, store_id: str, *, log: Callable[[str], None] = print,
                       sleep: Callable[[float], None] = time.sleep,
                       clock: Callable[[], float] = time.monotonic,
                       timeout_s: float = GROUP_TIMEOUT_S, poll_s: float = GROUP_POLL_S) -> bool:
    """Make sure the ``WebActivity`` group exists on the store; True if this call created it.

    Reads the group first and creates it (with exactly ``TRAITS``) only on a 404. Memory refuses
    traits for a group that doesn't exist (error 520047). A group that is present is left alone even
    when its trait list reads empty: right after creation the read lagged the write."""
    url = _group_url(store_id, GROUP)
    status, _ = _call(client, "GET", url, what="trait group read", allow_404=True)
    if status != 404:
        log(f"ok      trait group {GROUP}")
        return False
    body = {"displayName": GROUP, "description": "Recent web activity (OwlBank demo and Segment-style feeds)",
            "traits": {name: {"dataType": data_type} for name, data_type in TRAITS.items()}}
    _, created = _call(client, "POST", f"{MEMORY_URL}/v1/ControlPlane/Stores/{quote(store_id, safe='')}/TraitGroups",
                       body, what="trait group create")
    status_url = created.get("statusUrl") if isinstance(created, dict) else None
    log(f"create  trait group {GROUP}")
    deadline = clock() + timeout_s
    while True:
        if isinstance(status_url, str) and status_url:
            _, op = _call(client, "GET", _operation_url(status_url), what="trait group operation read")
            state = str(op.get("status", "")).lower() if isinstance(op, dict) else ""
            if state in _OP_FAILED or (isinstance(op, dict) and op.get("error")):
                raise ScriptError(f"Memory could not create trait group {GROUP} (operation status "
                                  f"{state[:40]!r}); check the store in the Console")
            if state in _OP_DONE:
                return True
        status, _ = _call(client, "GET", url, what="trait group read", allow_404=True)
        if status != 404:  # readable: settled, whatever the operation says
            return True
        if clock() >= deadline:
            raise ScriptError(f"trait group {GROUP} is not readable after {int(timeout_s)} s; wait a few "
                              "minutes, then re-run the same command (it creates the group only if it is "
                              "still missing)")
        sleep(poll_s)


def patch_traits(client: Any, store_id: str, profile_id: str,
                 traits: Mapping[str, Mapping[str, Any]]) -> None:
    """Partial update of a profile's traits: ``{"WebActivity": {"viewCount": 3}}`` changes only what
    is named, and ``None`` removes a trait. Memory answers 202 (accepted for processing); a 308
    (merged profile) raises :class:`MemoryApiError` with ``status`` 308 so the caller can re-look it up."""
    _call(client, "PATCH", _profile_url(store_id, profile_id),
          {"traits": {g: dict(t) for g, t in traits.items()}}, what="profile update")


def upsert_profile(client: Any, store_id: str, traits: Mapping[str, Mapping[str, Any]]) -> str:
    """Create a profile by identity resolution (``POST Profiles``, the call the bridge's
    ``create_profile`` makes through TAC): with an identifier trait in ``traits`` an existing profile
    is returned, not a second one. Memory answers 202 with the canonical ID; the traits are queued."""
    _, doc = _call(client, "POST", f"{MEMORY_URL}/v1/Stores/{quote(store_id, safe='')}/Profiles",
                   {"traits": {g: dict(t) for g, t in traits.items()}}, what="profile upsert")
    profile_id = doc.get("id") if isinstance(doc, dict) else None
    if not isinstance(profile_id, str) or not profile_id:
        raise ScriptError("Memory profile upsert answered without a profile ID")
    return profile_id
