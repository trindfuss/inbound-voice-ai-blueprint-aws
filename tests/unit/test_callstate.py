"""Unit tests for owlbank.callstate."""

from __future__ import annotations

import pytest

from owlbank import callstate as cs
from owlbank import segments as sg
from owlbank.store import MemoryStore

SID = "CA" + "c" * 32


@pytest.fixture(autouse=True)
def _clear() -> None:
    sg.clear_sensitive(SID)
    yield
    sg.clear_sensitive(SID)


def new_state(**kw) -> cs.CallState:
    return cs.CallState(call_sid=SID, expires_at=10_000, language="es-US", ani="+15551234567", **kw)


def test_create_load_round_trip() -> None:
    store = MemoryStore()
    state = new_state(candidate_ids=["C-1", "C-2"], attempts={"zip": 1},
                      pending_tool={"tool_use_id": "tu1", "name": "make_payment",
                                    "args": {"amount": "25.00"}})
    created = cs.create(store, state)
    assert created.version == 1
    loaded = cs.load(store, SID)
    assert loaded == created
    assert loaded.pending_tool == cs.PendingTool("tu1", "make_payment", {"amount": "25.00"})
    assert loaded.candidate_ids == ("C-1", "C-2")


def test_identity_source_defaults_to_ani_and_round_trips() -> None:
    """Keyed-in phone login: only the source is stored, never a keyed number."""
    store = MemoryStore()
    created = cs.create(store, new_state())
    assert created.identity_source == "ani" and cs.load(store, SID).identity_source == "ani"
    keyed = cs.update(store, created, now=lambda: 100, identity_source="keyed_phone")
    assert cs.load(store, SID) == keyed and keyed.identity_source == "keyed_phone"
    assert store.get(cs.TABLE, {"call_sid": SID})["identity_source"] == "keyed_phone"


def test_an_item_without_identity_source_loads_as_ani() -> None:
    """An item written before the field existed."""
    item = new_state().to_item()
    del item["identity_source"]
    assert cs.CallState.from_item(item).identity_source == "ani"


def test_login_closed_defaults_to_false_and_round_trips() -> None:
    """Keyed-in phone login: the no-second-try flag."""
    store = MemoryStore()
    created = cs.create(store, new_state())
    assert created.login_closed is False and cs.load(store, SID).login_closed is False
    closed = cs.update(store, created, now=lambda: 100, **{cs.F_LOGIN_CLOSED: True})
    assert cs.load(store, SID) == closed and closed.login_closed is True
    assert store.get(cs.TABLE, {"call_sid": SID})[cs.F_LOGIN_CLOSED] is True


def test_an_item_without_login_closed_loads_as_false() -> None:
    """An item written before the field existed."""
    item = new_state().to_item()
    del item[cs.F_LOGIN_CLOSED]
    state = cs.CallState.from_item(item)
    assert state.login_closed is False and cs.F_LOGIN_CLOSED not in state._extras  # noqa: SLF001


@pytest.mark.parametrize("bad", ["keyed", "", None, "KEYED_PHONE"])
def test_identity_source_rejects_unknown_values(bad: object) -> None:
    with pytest.raises(ValueError, match="identity_source"):
        new_state(identity_source=bad)


def test_load_missing_and_expired() -> None:
    store = MemoryStore()
    assert cs.load(store, SID) is None
    cs.create(store, new_state())
    assert cs.load(store, SID, now=9_999) is not None
    assert cs.load(store, SID, now=10_000) is None


def test_create_twice_conflicts() -> None:
    store = MemoryStore()
    cs.create(store, new_state())
    with pytest.raises(cs.VersionConflict):
        cs.create(store, new_state())


def test_update_bumps_version_and_stale_writer_conflicts() -> None:
    store = MemoryStore()
    v1 = cs.create(store, new_state())
    v2 = cs.update(store, v1, now=lambda: 100, level=1, factors_verified=("zip",),
                   verified_customer_id="C-1")
    assert v2.version == 2 and cs.load(store, SID) == v2
    with pytest.raises(cs.VersionConflict):
        cs.update(store, v1, now=lambda: 100, level=3)  # stale copy
    assert cs.load(store, SID).level == 1
    v3 = cs.update(store, v2, now=lambda: 100, level=2)
    assert v3.version == 3


def test_stale_claim_is_taken_over_after_timeout() -> None:
    store = MemoryStore()
    v1 = cs.create(store, new_state())
    # A writer claimed v2 at t=100 and died before writing the state.
    store.put("calls", {"call_sid": f"{SID}#v2", "kind": "version_claim", "claimed_at": 100,
                        "expires_at": 10_000})
    with pytest.raises(cs.VersionConflict):
        cs.update(store, v1, now=lambda: 105, level=1)
    v2 = cs.update(store, v1, now=lambda: 200, level=1)
    assert cs.load(store, SID) == v2


def test_mutate_retries_after_conflict() -> None:
    store = MemoryStore()
    cs.create(store, new_state())
    calls = []

    def fn(state: cs.CallState) -> dict:
        calls.append(state.version)
        if len(calls) == 1:  # someone else writes in between
            cs.update(store, state, now=lambda: 1, step_up_target="make_payment")
        return {"level": state.level + 1}

    final = cs.mutate(store, SID, fn, now=lambda: 1)
    assert calls == [1, 2]
    assert final.version == 3 and final.level == 1 and final.step_up_target == "make_payment"


def test_update_rejects_key_or_version_changes() -> None:
    store = MemoryStore()
    v1 = cs.create(store, new_state())
    with pytest.raises(ValueError):
        cs.update(store, v1, version=7)


def test_factor_values_rejected_in_state() -> None:
    store = MemoryStore()
    v1 = cs.create(store, new_state())
    sg.register_sensitive(SID, ["94110"])
    with pytest.raises(sg.RedactionError):
        cs.update(store, v1, pending_tool=cs.PendingTool("t", "escalate", {"summary": "zip 94110"}))
    assert cs.load(store, SID).version == 1


@pytest.mark.parametrize("bad", [{"call_sid": ""}, {"call_sid": "CA#1"}, {"level": -1}])
def test_invalid_state(bad: dict) -> None:
    kw = {"call_sid": SID, "expires_at": 1, **bad}
    with pytest.raises(ValueError):
        cs.CallState(**kw)


def test_prefetched_round_trips_and_clears() -> None:
    """The hand-off prefetch is stored, loaded and cleared like ``pending_result``."""
    store = MemoryStore()
    created = cs.create(store, new_state())
    assert created.prefetched is None and created.to_item()["prefetched"] is None
    prefetched = {"tool_use_id": "tu1", "tool": "get_balance", "args": {"account": "checking"},
                  "result": {"status": "ok", "data": {"available": "8420.00"}}, "at": 1_700_000_000}
    after = cs.update(store, created, prefetched=prefetched)
    loaded = cs.load(store, SID)
    assert loaded == after and loaded is not None
    assert loaded.prefetched == prefetched
    assert cs.CallState.from_item(loaded.to_item()) == loaded
    cleared = cs.mutate(store, SID, lambda _s: {cs.F_PREFETCHED: None})
    assert cleared.prefetched is None
    assert cs.load(store, SID).prefetched is None  # type: ignore[union-attr]


def test_otp_send_round_trips_and_clears() -> None:
    """The asynchronous send's outcome is stored, loaded and cleared like ``prefetched``."""
    store = MemoryStore()
    created = cs.create(store, new_state())
    assert created.otp_send is None and created.to_item()["otp_send"] is None
    rec = {"status": "sent", "at": 1_700_000_000}
    after = cs.mutate(store, SID, lambda _s: {cs.F_OTP_SEND: rec})
    loaded = cs.load(store, SID)
    assert loaded == after and loaded is not None and loaded.otp_send == rec
    assert cs.CallState.from_item(loaded.to_item()) == loaded
    assert {"sent", "escalate", "error"} == cs.OTP_SEND_STATUSES
    cleared = cs.mutate(store, SID, lambda _s: {cs.F_OTP_SEND: None})
    assert cleared.otp_send is None
    assert cs.load(store, SID).otp_send is None  # type: ignore[union-attr]


def test_item_from_before_the_harness_removal_still_loads_and_keeps_its_field() -> None:
    """``harness_session`` is no longer a field; an older item keeps it as an extra."""
    store = MemoryStore()
    v1 = cs.create(store, new_state())
    store.put("calls", {**v1.to_item(), "harness_session": "h1"})
    old = cs.load(store, SID)
    assert old is not None and not hasattr(old, "harness_session")
    v2 = cs.update(store, old, now=lambda: 1, level=1)
    assert v2.level == 1
    assert cs.load(store, SID).to_item()["harness_session"] == "h1"  # type: ignore[union-attr]


@pytest.mark.parametrize("field", [cs.F_WELCOME_PREPARED, cs.F_LAST_SPOKEN])
def test_sealed_blob_of_a_record_is_not_scanned(field: str) -> None:
    """A KMS blob can hold a registered value's characters by chance (``abc1234def``); the
    ``sealed`` sub-field of ``welcome_prepared`` and ``last_spoken`` is sealer output only, so it
    must not block the write. Every other field of those records is still scanned."""
    store = MemoryStore()
    v1 = cs.create(store, new_state())
    sg.register_sensitive(SID, ["1234"])
    v2 = cs.update(store, v1, **{field: {"sealed": "QUJDabc1234defXYZ==", "at": 1}})
    assert cs.load(store, SID).version == v2.version == 2
    with pytest.raises(sg.RedactionError):
        cs.update(store, v2, **{field: {"sealed": "QUJD", "language": "1234"}})
    with pytest.raises(sg.RedactionError):  # a non-sealed field elsewhere still raises
        cs.update(store, v2, pending_tool=cs.PendingTool("t", "escalate", {"note": "abc1234def"}))
    assert cs.load(store, SID).version == 2
