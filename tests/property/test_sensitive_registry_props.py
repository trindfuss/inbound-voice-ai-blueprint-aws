"""Properties of the sealed registry.

Processes are simulated by giving each its own registry: its own ``segments`` registered values
and its own ``sensitive`` process state, over one shared store.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank import callstate as cs
from owlbank import segments as sg
from owlbank import sensitive
from owlbank.store import MemoryStore
from owlbank.token_seal import FakeSealer
from tests import logutil

SIDS = ("CA" + "7" * 32, "CA" + "8" * 32)
SEALER = FakeSealer()

# Values a sensitive slot holds today (ssn4, otp, zip): digits, 1–12 characters (short ones
# included). Letters are left out here: a registered word can equal part of a call-state attribute
# name ("lang" in "language"), which the redaction guard then refuses on the next write; that is
# the guard's behaviour, not the registry's (reported separately).
values = st.text(alphabet="0123456789", min_size=1, max_size=12)
slots = st.sampled_from(["ssn4", "otp", "zip", "pin"])


class _Process:
    """One simulated process: its own registered values and sealed-registry state."""

    def __init__(self) -> None:
        self.registered: dict[str, set[str]] = {}
        self.state = sensitive._ProcessState()

    @contextlib.contextmanager
    def active(self) -> Iterator[None]:
        saved = sg._SENSITIVE, sensitive._state
        sg._SENSITIVE, sensitive._state = self.registered, self.state  # type: ignore[assignment]
        try:
            yield
        finally:
            sg._SENSITIVE, sensitive._state = saved  # type: ignore[assignment]


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _captured() -> Iterator[_Capture]:
    handler = _Capture()
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(level)


def _store() -> MemoryStore:
    store = MemoryStore()
    for sid in SIDS:
        cs.create(store, cs.CallState(call_sid=sid, expires_at=10_000))
    return store


def _sealed(store: MemoryStore, sid: str) -> tuple[str, ...]:
    state = cs.load(store, sid)
    assert state is not None
    return state.sealed_sensitive


def _long(v: str) -> bool:
    return len(v) >= sg.MIN_SENSITIVE_LEN


# One operation: ("remember", process, call, slot, value) or ("restore", process, call).
remember_op = st.tuples(st.just("remember"), st.integers(0, 2), st.integers(0, 1), slots, values)
restore_op = st.tuples(st.just("restore"), st.integers(0, 2), st.integers(0, 1))
ops = st.lists(st.one_of(remember_op, restore_op), min_size=1, max_size=24)


@settings(max_examples=150, deadline=None)
@given(n_procs=st.integers(1, 3), script=ops)
def test_registries_converge_on_restore_and_restore_is_idempotent(n_procs: int, script: list[Any]) -> None:
    """Convergence: after a process restores a call, its registry holds exactly the remembered
    values of 4 or more characters; restoring again changes nothing."""
    store, procs = _store(), [_Process() for _ in range(n_procs)]
    remembered: dict[str, list[str]] = {sid: [] for sid in SIDS}
    for op in script:
        proc, sid = procs[op[1] % n_procs], SIDS[op[2]]
        with proc.active():
            if op[0] == "remember":
                if len(remembered[sid]) >= cs.MAX_SEALED_SENSITIVE:
                    continue  # the limit has its own example test
                assert sensitive.remember(store, SEALER, sid, op[4], slot=op[3], now=lambda: 100.0)
                remembered[sid].append(op[4])
            else:
                sensitive.restore(sid, _sealed(store, sid), SEALER)
                expected = frozenset(v for v in remembered[sid] if _long(v))
                assert sg.sensitive_values(sid) == expected
                sensitive.restore(sid, _sealed(store, sid), SEALER)
                assert sg.sensitive_values(sid) == expected
                assert not sensitive.is_unverifiable(sid)
    # Finally every process restores every call and they all agree.
    for proc in procs:
        with proc.active():
            for sid in SIDS:
                sensitive.restore(sid, _sealed(store, sid), SEALER)
                assert sg.sensitive_values(sid) == frozenset(v for v in remembered[sid] if _long(v))


short_values = st.text(alphabet="0123456789abcdefghijklmnopqrstuvwxyz", min_size=1,
                       max_size=sg.MIN_SENSITIVE_LEN - 1)
# Slot names no generated value can equal (an underscore), so "no field carries the value" is exact.
short_slots = st.sampled_from(["ssn_last4", "otp_code", "pin_code"])
_SENSITIVE_LOGGER = "owlbank.sensitive"


def _opened(store: MemoryStore, sid: str) -> set[str]:
    """The values sealed in the call's state (the test reads the sealed record directly)."""
    return {json.loads(SEALER.unseal(b, call_sid=sid))["value"] for b in _sealed(store, sid)}


@settings(max_examples=150, deadline=None)
@given(n_procs=st.integers(1, 3),
       script=st.lists(st.tuples(st.integers(0, 2), st.integers(0, 1), short_slots, short_values,
                                 st.booleans()), min_size=1, max_size=12))
def test_short_values_are_sealed_never_registered_and_skipped_once(n_procs: int, script: list[Any]) -> None:
    """A value under 4 characters is never registered, is still sealed,
    yields exactly one ``sensitive_scrub_skipped`` per (call, slot) per process, and appears in no
    captured log record field."""
    store, procs = _store(), [_Process() for _ in range(n_procs)]
    remembered: dict[str, set[str]] = {sid: set() for sid in SIDS}
    slots_of: dict[str, set[str]] = {sid: set() for sid in SIDS}
    with _captured() as cap:
        for p, c, slot, value, then_restore in script:
            sid = SIDS[c]
            with procs[p % n_procs].active():
                assert sensitive.remember(store, SEALER, sid, value, slot=slot, now=lambda: 100.0)
                remembered[sid].add(value)
                slots_of[sid].add(slot)
                if then_restore:
                    sensitive.restore(sid, _sealed(store, sid), SEALER)
                assert sg.sensitive_values(sid) == frozenset()
        # Every process restores every call, so each meets every slot of every call.
        for proc in procs:
            with proc.active():
                for sid in SIDS:
                    sensitive.restore(sid, _sealed(store, sid), SEALER)
                    sensitive.restore(sid, _sealed(store, sid), SEALER)
                    assert sg.sensitive_values(sid) == frozenset()
                    assert not sensitive.is_unverifiable(sid)
    for sid in SIDS:
        assert _opened(store, sid) == remembered[sid]  # still sealed
    skipped = [logutil.fields(r) | {"_level": r.levelno} for r in cap.records
               if logutil.fields(r).get("event") == "sensitive_scrub_skipped"]
    assert all(f["_level"] == logging.INFO and f["reason"] == "too_short" for f in skipped)
    assert len(skipped) == n_procs * sum(len(s) for s in slots_of.values())
    all_values = set().union(*remembered.values())
    for r in cap.records:
        f = logutil.fields(r)
        assert r.getMessage() not in all_values
        assert not any(isinstance(v, str) and v in all_values for v in f.values())
        if r.name == _SENSITIVE_LOGGER:  # structural: only fixed keys, no free text
            assert set(f) <= {"event", "slot", "reason", "decision"}
