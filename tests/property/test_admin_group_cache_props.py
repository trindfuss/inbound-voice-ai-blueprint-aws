"""Admin API live group check properties:

* The group cache never serves an answer older than 60 s: over generated clock sequences
  (steps forward, standing still, jumps past the limit and backwards), mixed users and
  forgotten entries, every value it returns came from a Cognito read made at most 60 s
  earlier (``0 <= now - read_at < 60``), and a missing or stale entry always reads again.
* Through the API: once Cognito no longer has a caller in ``admin``, or has disabled them, no
  user route succeeds for them 60 s or more later, whatever their token's groups say.
"""

from __future__ import annotations

import itertools

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.admin_api.membership import TTL_S, GroupCache, Membership
from tests.admin_api_fakes import call, claims, make_api

USERS = ("u-a", "u-b", "u-c")

_deltas = st.one_of(st.just(0.0), st.floats(0.0, 5.0), st.floats(55.0, 65.0), st.floats(0.0, 200.0),
                    st.floats(-30.0, 0.0), st.sampled_from([TTL_S, TTL_S - 1e-9, TTL_S + 1e-9]))
_steps = st.lists(st.tuples(_deltas, st.sampled_from(USERS), st.booleans()), min_size=1, max_size=60)


@settings(max_examples=300, deadline=None)
@given(_steps)
def test_the_cache_never_serves_an_entry_older_than_60_s(steps) -> None:
    now = [1_792_000_000.0]
    read_at: dict[Membership, float] = {}
    serial = itertools.count()

    def fetch(username: str) -> Membership:
        value = Membership(frozenset({f"{username}#{next(serial)}"}))  # each read is distinguishable
        read_at[value] = now[0]
        return value

    cache = GroupCache(fetch, clock=lambda: now[0])
    for delta, user, forget in steps:
        now[0] += delta
        if forget:
            cache.forget(user)
        reads = len(read_at)
        value = cache.groups(user)
        age = now[0] - read_at[value]
        assert 0 <= age < TTL_S, (age, steps)
        assert next(iter(value.groups)).startswith(f"{user}#")
        if forget:
            assert len(read_at) == reads + 1  # a forgotten entry is read again


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.tuples(st.floats(0.0, 90.0), st.booleans()), min_size=1, max_size=15),
       st.sampled_from(["demote", "disable"]))
def test_an_admin_is_refused_60_s_after_cognito_demotes_or_disables_them(steps, change) -> None:
    api, _, cognito, clock = make_api()
    me = cognito.add("me@example.com", "admin")
    cognito.add("other@example.com", "admin")
    token = claims(["admin"], sub=me)
    demoted_at: float | None = None
    for delta, demote in steps:
        clock.t += delta
        if demote and demoted_at is None:  # changed outside this API instance
            if change == "demote":
                cognito.groups["admin"].discard(me)
                cognito.groups["admin-viewer"].add(me)
            else:
                cognito.users[me]["Enabled"] = False  # still in admin
            demoted_at = clock.t
        status, _ = call(api, "GET", "/users", who=token)
        if demoted_at is not None and clock.t - demoted_at >= TTL_S:
            assert status == 403
        elif demoted_at is None:
            assert status == 200
