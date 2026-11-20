"""Server-side group membership for the admin-only routes.

API Gateway's Cognito authorizer checks only a token's signature and expiry, so the
``cognito:groups`` claim of an ID token stays what it was at sign-in for the token's whole
lifetime (60 minutes). The user and admin-only routes therefore also ask Cognito whether the
caller is enabled (``AdminGetUser``) and which groups they are in now
(``AdminListGroupsForUser``). The combined answer is cached per user for at most :data:`TTL_S`
seconds, so a disabled, demoted or deleted admin loses those routes within that time on every
warm Lambda instance; the instance that made the change forgets the entry at once.

An entry is stamped with the clock read *before* the Cognito calls, so its age is never
understated. An entry is served only while ``0 <= now - stamp < TTL_S``: a clock that went
backwards counts as stale too.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

#: The longest an answer from Cognito is served from the cache.
TTL_S = 60.0
#: Entries kept before expired ones are dropped (a console has a handful of users).
MAX_ENTRIES = 256


class Membership(NamedTuple):
    """What Cognito says about a user now: their console groups and whether they are enabled.
    A user Cognito can't find (deleted) has no groups."""

    groups: frozenset[str]
    enabled: bool = True


class GroupCache:
    """``groups(username)``: the user's membership now, from ``fetch`` or a fresh enough entry."""

    def __init__(self, fetch: Callable[[str], Membership], *, clock: Callable[[], float],
                 ttl_s: float = TTL_S, max_entries: int = MAX_ENTRIES) -> None:
        if not 0 < ttl_s <= TTL_S:
            raise ValueError(f"ttl_s must be in (0, {TTL_S}]")
        self._fetch, self._clock, self.ttl_s, self._max = fetch, clock, ttl_s, max_entries
        self._entries: dict[str, tuple[float, Membership]] = {}

    def _fresh(self, stamp: float, now: float) -> bool:
        return 0 <= now - stamp < self.ttl_s

    def groups(self, username: str) -> Membership:
        now = self._clock()
        hit = self._entries.get(username)
        if hit is not None and self._fresh(hit[0], now):
            return hit[1]
        value = self._fetch(username)  # an error is not cached
        if len(self._entries) >= self._max:
            self._entries = {k: v for k, v in self._entries.items() if self._fresh(v[0], now)}
            if len(self._entries) >= self._max:
                self._entries.clear()
        self._entries[username] = (now, value)
        return value

    def forget(self, username: str) -> None:
        self._entries.pop(username, None)
