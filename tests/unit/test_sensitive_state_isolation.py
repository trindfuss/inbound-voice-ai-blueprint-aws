"""``owlbank.sensitive`` process state does not leak from one test to the
next (``tests/conftest.py``'s autouse fixture). Tests reuse CallSids, and a leaked entry made a later
test's ``remember`` skip sealing (it already "knew" the value), so the new call state lacked it.

The two tests run in file order: the first leaves state behind on purpose, the second must not
see it."""

from __future__ import annotations

from owlbank import callstate as cs
from owlbank import sensitive
from owlbank.store import MemoryStore
from owlbank.token_seal import FakeSealer

SID = "CA" + "e" * 30 + "ab"
VALUE = "94105"


def _store() -> MemoryStore:
    store = MemoryStore()
    cs.create(store, cs.CallState(call_sid=SID, expires_at=10_000))
    return store


def test_1_leaves_sensitive_state_behind() -> None:
    sealer = FakeSealer()
    assert sensitive.remember(_store(), sealer, SID, VALUE, slot="zip", now=lambda: 100.0)
    sensitive.restore(SID, ("garbage",), sealer)
    assert sensitive.is_unverifiable(SID)  # no forget: the fixture must clear it


def test_2_a_later_test_on_the_same_call_sid_starts_fresh() -> None:
    assert not sensitive.is_unverifiable(SID)
    store = _store()
    assert sensitive.remember(store, FakeSealer(), SID, VALUE, slot="zip", now=lambda: 100.0)
    state = cs.load(store, SID)
    assert state is not None and len(state.sealed_sensitive) == 1  # sealed into this call state
    sensitive.forget(SID)
