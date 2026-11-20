"""Property tests for owlbank.tokens: validation and hash-only storage."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank import tokens
from owlbank.store import MemoryStore
from owlbank.tokens import AuthError, AuthErrorCode, TokenClaims

call_sids = st.text(alphabet="0123456789abcdef", min_size=1, max_size=8).map(lambda s: "CA" + s)
customer_ids = st.text(min_size=1, max_size=20)
levels = st.integers(min_value=0, max_value=4)
times = st.integers(min_value=0, max_value=2_000_000_000)
ttls = st.integers(min_value=1, max_value=7200)




def expected(
    *, presented: str, token_known: bool, same_call: bool, revoked: bool, expired: bool,
    level: int, min_level: int,
) -> AuthErrorCode | None:
    """Oracle: the documented precedence MISSING > UNKNOWN > WRONG_CALL > REVOKED > EXPIRED > LEVEL."""
    if presented == "missing":
        return AuthErrorCode.MISSING
    if not token_known:
        return AuthErrorCode.UNKNOWN
    if not same_call:
        return AuthErrorCode.WRONG_CALL
    if revoked:
        return AuthErrorCode.REVOKED
    if expired:
        return AuthErrorCode.EXPIRED
    if level < min_level:
        return AuthErrorCode.AUTH_LEVEL_INSUFFICIENT
    return None


@settings(max_examples=300)
@given(
    call_sid=call_sids,
    other_sid=call_sids,
    customer_id=customer_ids,
    level=levels,
    min_level=levels,
    issued_at=times,
    ttl=ttls,
    offset=st.integers(min_value=-100, max_value=8000),
    revoked=st.booleans(),
    same_call=st.booleans(),
    presented=st.sampled_from(["real", "missing", "garbled", "other_token"]),
    garbage=st.text(min_size=1, max_size=60),
)
def test_validation_matrix(
    call_sid: str, other_sid: str, customer_id: str, level: int, min_level: int,
    issued_at: int, ttl: int, offset: int, revoked: bool, same_call: bool, presented: str,
    garbage: str,
) -> None:
    store = MemoryStore()
    token = tokens.issue(store, call_sid=call_sid, customer_id=customer_id, level=level,
                         now=issued_at, max_ttl_s=ttl)
    # A different, valid token on another call: must never stand in for this one.
    other = tokens.issue(store, call_sid=call_sid + "x", customer_id="someone-else", level=4,
                         now=issued_at, max_ttl_s=ttl)
    if revoked:
        tokens.revoke_call(store, call_sid)
    now = max(0, issued_at + offset)
    sid = call_sid if same_call or other_sid == call_sid else other_sid
    value = {"real": token, "missing": None, "garbled": garbage, "other_token": other}[presented]
    known = presented == "real" or (presented == "garbled" and garbage == token)

    result = tokens.validate(store, value, call_sid=sid, min_level=min_level, now=now)

    if presented == "other_token":
        # bound to call_sid+"x", never equal to sid (sid is call_sid or another "CA..." id
        # that can't end with an extra "x")
        assert result == AuthError(AuthErrorCode.WRONG_CALL)
        return
    want = expected(
        presented=presented, token_known=known, same_call=sid == call_sid, revoked=revoked,
        expired=now >= issued_at + ttl, level=level, min_level=min_level,
    )
    if want is None:
        assert result == TokenClaims(call_sid, customer_id, level, issued_at, issued_at + ttl)
    else:
        assert result == AuthError(want)


@settings(max_examples=150)
@given(
    call_sid=call_sids,
    steps=st.lists(st.tuples(levels, st.integers(min_value=0, max_value=100)), min_size=1,
                   max_size=6),
    start_level=levels,
)
def test_upgrade_levels_never_decrease_and_only_latest_is_valid(
    call_sid: str, steps: list[tuple[int, int]], start_level: int
) -> None:
    store = MemoryStore()
    now, ttl = 1_000_000, 10_000
    current = tokens.issue(store, call_sid=call_sid, customer_id="c", level=start_level, now=now,
                           max_ttl_s=ttl)
    level, history = start_level, [current]
    for new_level, dt in steps:
        now += dt
        if new_level < level:
            with pytest.raises(ValueError):
                tokens.upgrade(store, current, call_sid=call_sid, new_level=new_level, now=now,
                               max_ttl_s=ttl)
            continue
        result = tokens.upgrade(store, current, call_sid=call_sid, new_level=new_level, now=now,
                                max_ttl_s=ttl)
        assert isinstance(result, str)
        current, level = result, new_level
        history.append(current)
    claims = tokens.validate(store, current, call_sid=call_sid, min_level=0, now=now)
    assert isinstance(claims, TokenClaims) and claims.level == level
    assert claims.level >= start_level and claims.customer_id == "c"
    for old in history[:-1]:
        assert tokens.validate(store, old, call_sid=call_sid, min_level=0, now=now) == AuthError(
            AuthErrorCode.REVOKED
        )




class RecordingStore(MemoryStore):
    """MemoryStore that records every payload written."""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[str] = []

    def put(self, table: str, item: Mapping[str, Any], *, condition: Any = None) -> None:
        self.writes.append(repr((table, dict(item))))
        super().put(table, item, condition=condition)

    def update(self, table: str, key: Mapping[str, Any], updates: Mapping[str, Any], *,
               remove: Iterable[str] = ()) -> dict[str, Any]:
        self.writes.append(repr((table, dict(key), dict(updates))))
        return super().update(table, key, updates, remove=remove)

    def dump(self) -> str:
        return repr(self._tables)


@settings(max_examples=100)
@given(
    specs=st.lists(st.tuples(call_sids, customer_ids, levels), min_size=1, max_size=15),
    upgrade_to=levels,
)
def test_store_never_contains_raw_token(
    specs: list[tuple[str, str, int]], upgrade_to: int
) -> None:
    store = RecordingStore()
    issued = [
        (tokens.issue(store, call_sid=sid, customer_id=cid, level=lvl, now=100, max_ttl_s=60), sid,
         lvl)
        for sid, cid, lvl in specs
    ]
    tok, sid, lvl = issued[0]
    up = tokens.upgrade(store, tok, call_sid=sid, new_level=max(lvl, upgrade_to), now=101,
                        max_ttl_s=60)
    assert isinstance(up, str)
    all_tokens = [t for t, _, _ in issued] + [up]
    tokens.revoke_call(store, specs[-1][0])
    contents = store.dump()
    written = "\n".join(store.writes)
    for t in all_tokens:
        assert t not in contents
        assert t not in written
        assert tokens.hash_token(t) in contents
    # no collisions, of tokens or of hashes
    assert len(set(all_tokens)) == len(all_tokens)
    assert len({tokens.hash_token(t) for t in all_tokens}) == len(all_tokens)


@given(st.text())
def test_hash_is_deterministic_sha256(token: str) -> None:
    h = tokens.hash_token(token)
    assert h == tokens.hash_token(token)
    assert h == hashlib.sha256(token.encode("utf-8")).hexdigest()
    assert len(h) == 64 and h == h.lower()
