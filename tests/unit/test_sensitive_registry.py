"""The sealed registry.

``sensitive.remember`` seals a sensitive slot's value with the CallSid context, appends it to call
state's ``sealed_sensitive`` and registers it for the exact-value scrub in this process;
``sensitive.restore`` registers what other processes sealed. No value ever reaches a log line.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from owlbank import callstate as cs
from owlbank import segments as sg
from owlbank import sensitive
from owlbank.store import MemoryStore
from owlbank.token_seal import FakeSealer, UnsealError
from tests import logutil

SID = "CA" + "5" * 32
OTHER = "CA" + "6" * 32
VALUE = "1111"  # the fixture customer's SSN last four


@pytest.fixture(autouse=True)
def _fresh_process():
    """Every test starts as a fresh process: no registered values, no seen blobs."""
    for sid in (SID, OTHER):
        sensitive.forget(sid)
    yield
    for sid in (SID, OTHER):
        sensitive.forget(sid)


def _store(sid: str = SID) -> MemoryStore:
    store = MemoryStore()
    cs.create(store, cs.CallState(call_sid=sid, expires_at=10_000))
    return store


def _now() -> float:
    return 100.0


def _sealed(store: MemoryStore, sid: str = SID) -> tuple[str, ...]:
    state = cs.load(store, sid)
    assert state is not None
    return state.sealed_sensitive


class _BrokenSealer(FakeSealer):
    def unseal(self, sealed: str, *, call_sid: str) -> str:
        raise UnsealError("token could not be unsealed for this call")


# --------------------------------------------------------------------------- call state field

def test_call_state_field_defaults_round_trips_and_is_never_printed() -> None:
    state = cs.CallState(call_sid=SID, expires_at=10_000)
    assert state.sealed_sensitive == ()
    sealed = FakeSealer().seal("secret-blob-value", call_sid=SID)
    state = cs.CallState(call_sid=SID, expires_at=10_000, sealed_sensitive=[sealed])
    assert state.sealed_sensitive == (sealed,)  # normalized to a tuple
    assert sealed not in repr(state)
    assert "sealed_sensitive" not in repr(state)
    item = state.to_item()
    assert item[cs.F_SEALED_SENSITIVE] == [sealed]
    assert cs.CallState.from_item(item) == state


def test_call_state_refuses_more_than_the_limit() -> None:
    assert cs.MAX_SEALED_SENSITIVE == 16
    with pytest.raises(ValueError, match="sealed_sensitive"):
        cs.CallState(call_sid=SID, expires_at=10_000, sealed_sensitive=["x"] * 17)
    with pytest.raises(ValueError, match="sealed_sensitive"):
        cs.CallState(call_sid=SID, expires_at=10_000, sealed_sensitive=[1])  # type: ignore[list-item]


def test_sealed_blobs_are_not_scanned_by_the_redaction_guard() -> None:
    """A ciphertext can contain a registered value's characters by chance; that must never stop
    the call state from being written (the field only ever holds sealer output)."""
    store = _store()
    sg.register_sensitive(SID, ["abcd"])
    state = cs.load(store, SID)
    assert state is not None
    cs.update(store, state, now=_now, sealed_sensitive=("xxABCDxx",))
    assert _sealed(store) == ("xxABCDxx",)
    # Other fields are still scanned.
    state = cs.load(store, SID)
    assert state is not None
    with pytest.raises(sg.RedactionError):
        cs.update(store, state, now=_now, interrupted_utterance="it is a b c d")


# --------------------------------------------------------------------------- remember

def test_remember_seals_appends_and_registers(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    store, sealer = _store(), FakeSealer()
    assert sensitive.remember(store, sealer, SID, VALUE, slot="ssn4", now=_now) is True
    blobs = _sealed(store)
    assert len(blobs) == 1
    assert VALUE not in blobs[0]
    assert sg.sensitive_values(SID) == frozenset({VALUE})
    assert sg.redact_text(SID, "it's 1 1-1 1") == "it's [redacted]"
    # Sealed with the call-SID context: it opens for this call only.
    with pytest.raises(UnsealError):
        sealer.unseal(blobs[0], call_sid=OTHER)
    assert VALUE not in logutil.text_of(caplog.records)


def test_remember_the_same_answer_twice_appends_once() -> None:
    store, sealer = _store(), FakeSealer()
    sensitive.remember(store, sealer, SID, VALUE, slot="ssn4", now=_now)
    sensitive.remember(store, sealer, SID, VALUE, slot="ssn4", now=_now)
    assert len(_sealed(store)) == 1
    sensitive.remember(store, sealer, SID, "94105", slot="zip", now=_now)
    assert len(_sealed(store)) == 2


def test_a_seventeenth_entry_is_refused_with_an_error_naming_the_field(
        caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    store, sealer = _store(), FakeSealer()
    values = [f"{n:04d}{n:04d}" for n in range(1000, 1017)]
    for v in values[:16]:
        assert sensitive.remember(store, sealer, SID, v, slot="otp", now=_now) is True
    assert len(_sealed(store)) == 16
    assert sensitive.remember(store, sealer, SID, values[16], slot="otp", now=_now) is False
    assert len(_sealed(store)) == 16
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert logutil.fields(errors[0])["field"] == "sealed_sensitive"
    text = logutil.text_of(caplog.records)
    assert not any(v in text for v in values)
    # Still protected in this process's logs.
    assert sg.contains_sensitive(SID, values[16])


def test_a_short_value_is_sealed_not_registered_and_logged_once_per_slot(
        caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    store, sealer = _store(), FakeSealer()
    sensitive.remember(store, sealer, SID, "123", slot="pin", now=_now)
    sensitive.remember(store, sealer, SID, "456", slot="pin", now=_now)
    sensitive.remember(store, sealer, SID, "78", slot="code", now=_now)
    assert len(_sealed(store)) == 3  # still sealed
    assert sg.sensitive_values(SID) == frozenset()  # never registered
    skipped = [logutil.fields(r) for r in caplog.records
               if logutil.fields(r).get("event") == "sensitive_scrub_skipped"]
    assert [(f["slot"], f["reason"]) for f in skipped] == [("pin", "too_short"), ("code", "too_short")]
    assert all(r.levelno == logging.INFO for r in caplog.records
               if logutil.fields(r).get("event") == "sensitive_scrub_skipped")
    text = logutil.text_of(caplog.records)
    assert "123" not in text and "456" not in text and " 78" not in text


def test_remember_needs_the_call_state() -> None:
    with pytest.raises(KeyError):
        sensitive.remember(MemoryStore(), FakeSealer(), SID, VALUE, slot="ssn4", now=_now)
    # The value is still registered in this process before the write fails.
    assert sg.contains_sensitive(SID, VALUE)


# --------------------------------------------------------------------------- restore

def test_restore_registers_what_another_process_sealed() -> None:
    store, sealer = _store(), FakeSealer()
    sensitive.remember(store, sealer, SID, VALUE, slot="ssn4", now=_now)
    sensitive.remember(store, sealer, SID, "94105", slot="zip", now=_now)
    sensitive.forget(SID)  # now a different process
    assert sg.sensitive_values(SID) == frozenset()
    sensitive.restore(SID, _sealed(store), sealer)
    assert sg.sensitive_values(SID) == frozenset({VALUE, "94105"})
    assert not sensitive.is_unverifiable(SID)


def test_restore_unseals_each_blob_once() -> None:
    store = _store()
    calls: list[str] = []

    class Counting(FakeSealer):
        def unseal(self, sealed: str, *, call_sid: str) -> str:
            calls.append(sealed)
            return super().unseal(sealed, call_sid=call_sid)

    sealer = Counting()
    sensitive.remember(store, sealer, SID, VALUE, slot="ssn4", now=_now)
    sensitive.restore(SID, _sealed(store), sealer)
    assert calls == []  # this process sealed it: already known
    sensitive.forget(SID)
    sensitive.restore(SID, _sealed(store), sealer)
    sensitive.restore(SID, _sealed(store), sealer)
    assert len(calls) == 1
    sensitive.remember(store, sealer, SID, "94105", slot="zip", now=_now)
    sensitive.restore(SID, _sealed(store), sealer)
    assert len(calls) == 1


def test_restore_of_a_short_value_logs_the_skip_once(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    store, sealer = _store(), FakeSealer()
    sensitive.remember(store, sealer, SID, "12", slot="pin", now=_now)
    sensitive.forget(SID)
    caplog.clear()
    sensitive.restore(SID, _sealed(store), sealer)
    sensitive.restore(SID, _sealed(store), sealer)
    skipped = [logutil.fields(r) for r in caplog.records
               if logutil.fields(r).get("event") == "sensitive_scrub_skipped"]
    assert [(f["slot"], f["reason"]) for f in skipped] == [("pin", "too_short")]
    assert sg.sensitive_values(SID) == frozenset()


def test_an_unseal_failure_marks_the_call_unverifiable_and_logs_no_value(
        caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    store = _store()
    sensitive.remember(store, FakeSealer(), SID, VALUE, slot="ssn4", now=_now)
    sensitive.forget(SID)
    caplog.clear()
    sensitive.restore(SID, _sealed(store), _BrokenSealer())  # never raises
    assert sensitive.is_unverifiable(SID)
    assert not sensitive.is_unverifiable(OTHER)
    caught = [logutil.fields(r) for r in caplog.records if logutil.fields(r).get("decision") == "withhold_texts"]
    assert len(caught) == 1 and caught[0]["error_class"] == "UnsealError"
    text = logutil.text_of(caplog.records)
    assert VALUE not in text
    assert _sealed(store)[0] not in text  # nor the blob


def test_a_failed_unseal_is_retried_on_the_next_restore() -> None:
    """A KMS error on one webhook must not leave the value unknown for the
    rest of the call on this instance; the next restore unseals it."""
    store = _store()
    calls: list[str] = []

    class FlakyOnce(FakeSealer):
        def unseal(self, sealed: str, *, call_sid: str) -> str:
            calls.append(sealed)
            if len(calls) == 1:
                raise UnsealError("KMS throttled", transient=True)
            return super().unseal(sealed, call_sid=call_sid)

    sealer = FlakyOnce()
    sensitive.remember(store, sealer, SID, VALUE, slot="ssn4", now=_now)
    sensitive.forget(SID)  # another process
    sensitive.restore(SID, _sealed(store), sealer)
    assert sg.sensitive_values(SID) == frozenset()
    assert sensitive.recall(SID, _sealed(store), slot="ssn4") is None
    sensitive.restore(SID, _sealed(store), sealer)
    assert len(calls) == 2
    assert sg.sensitive_values(SID) == frozenset({VALUE})
    assert sensitive.recall(SID, _sealed(store), slot="ssn4") == VALUE
    sensitive.restore(SID, _sealed(store), sealer)
    assert len(calls) == 2  # unsealed once it worked


def test_a_blob_from_another_call_is_an_unseal_failure() -> None:
    store, sealer = _store(OTHER), FakeSealer()
    sensitive.remember(store, sealer, OTHER, VALUE, slot="ssn4", now=_now)
    sensitive.restore(SID, _sealed(store, OTHER), sealer)
    assert sensitive.is_unverifiable(SID)
    assert sg.sensitive_values(SID) == frozenset()


def test_a_sealed_entry_that_is_not_a_registry_record_fails_closed() -> None:
    sealer = FakeSealer()
    sensitive.restore(SID, (sealer.seal("not a record", call_sid=SID),), sealer)
    assert sensitive.is_unverifiable(SID)


def test_any_unseal_exception_fails_closed(caplog: pytest.LogCaptureFixture) -> None:
    class Boom(FakeSealer):
        def unseal(self, sealed: str, *, call_sid: str) -> str:
            raise RuntimeError("kms down " + VALUE)

    caplog.set_level(logging.DEBUG)
    sensitive.restore(SID, (FakeSealer().seal("x", call_sid=SID),), Boom())
    assert sensitive.is_unverifiable(SID)
    assert VALUE not in logutil.text_of(caplog.records)


def test_restore_with_nothing_sealed_changes_nothing() -> None:
    sensitive.restore(SID, (), FakeSealer())
    assert not sensitive.is_unverifiable(SID)
    assert sg.sensitive_values(SID) == frozenset()


def test_forget_clears_the_process_state() -> None:
    store, sealer = _store(), FakeSealer()
    sensitive.remember(store, sealer, SID, VALUE, slot="ssn4", now=_now)
    sensitive.restore(SID, ("garbage",), sealer)
    assert sensitive.is_unverifiable(SID)
    sensitive.forget(SID)
    assert not sensitive.is_unverifiable(SID)
    assert sg.sensitive_values(SID) == frozenset()


# --------------------------------------------------------------------------- yes_no and choice

@pytest.mark.parametrize("slot_type,slot,value", [
    ("yes_no", "confirm", "yes"), ("yes_no", "confirm_pay", "nope"), ("choice", "account", "checking")])
def test_yes_no_and_choice_values_are_sealed_but_never_registered(
        caplog: pytest.LogCaptureFixture, slot_type: str, slot: str, value: str) -> None:
    # Only the placeholder protects them; registering "checking" would scrub the word everywhere.
    caplog.set_level(logging.DEBUG)
    store, sealer = _store(), FakeSealer()
    assert sensitive.remember(store, sealer, SID, value, slot=slot, slot_type=slot_type, now=_now)
    assert len(_sealed(store)) == 1
    assert sg.sensitive_values(SID) == frozenset()
    sensitive.forget(SID)
    sensitive.restore(SID, _sealed(store), sealer)  # another process: still not registered
    assert sg.sensitive_values(SID) == frozenset()
    assert not sensitive.is_unverifiable(SID)
    skipped = [logutil.fields(r) for r in caplog.records
               if logutil.fields(r).get("event") == "sensitive_scrub_skipped"]
    assert [(f["slot"], f["reason"]) for f in skipped] == [(slot, "slot_type"), (slot, "slot_type")]
    assert value not in logutil.text_of(caplog.records)


@pytest.mark.parametrize("slot_type", ["digits", "amount", "date", None])
def test_other_slot_types_are_registered(slot_type: str | None) -> None:
    store, sealer = _store(), FakeSealer()
    sensitive.remember(store, sealer, SID, VALUE, slot="ssn4", slot_type=slot_type, now=_now)
    assert sg.sensitive_values(SID) == frozenset({VALUE})
    sensitive.forget(SID)
    sensitive.restore(SID, _sealed(store), sealer)
    assert sg.sensitive_values(SID) == frozenset({VALUE})


def test_the_sealed_record_carries_the_slot_type_only_when_given() -> None:
    store, sealer = _store(), FakeSealer()
    sensitive.remember(store, sealer, SID, VALUE, slot="ssn4", now=_now)
    sensitive.remember(store, sealer, SID, "savings", slot="account", slot_type="choice", now=_now)
    records = [json.loads(sealer.unseal(b, call_sid=SID)) for b in _sealed(store)]
    assert records == [{"v": 1, "slot": "ssn4", "value": VALUE},
                       {"v": 1, "slot": "account", "type": "choice", "value": "savings"}]


def test_a_record_with_a_bad_type_fails_closed() -> None:
    sealer = FakeSealer()
    blob = sealer.seal(json.dumps({"v": 1, "slot": "s", "type": 3, "value": "1234"}), call_sid=SID)
    sensitive.restore(SID, (blob,), sealer)
    assert sensitive.is_unverifiable(SID)
    assert sg.sensitive_values(SID) == frozenset()


# --------------------------------------------------------------------------- recall (keyed login)

def test_recall_returns_the_value_of_a_slot_only_from_this_call_states_blobs() -> None:
    store, sealer = _store(), FakeSealer()
    sensitive.remember(store, sealer, SID, "+12025550101", slot="keyed_phone", now=_now)
    sensitive.remember(store, sealer, SID, VALUE, slot="ssn4", now=_now)
    blobs = _sealed(store)
    assert sensitive.recall(SID, blobs, slot="keyed_phone") == "+12025550101"
    assert sensitive.recall(SID, blobs, slot="ssn4") == VALUE
    assert sensitive.recall(SID, blobs, slot="zip") is None
    assert sensitive.recall(SID, (), slot="keyed_phone") is None   # not in this call state
    assert sensitive.recall(OTHER, blobs, slot="keyed_phone") is None


def test_recall_after_restore_in_a_fresh_process_and_none_when_ambiguous() -> None:
    store, sealer = _store(), FakeSealer()
    sensitive.remember(store, sealer, SID, "+12025550101", slot="keyed_phone", now=_now)
    sensitive.remember(store, sealer, SID, "+12025550102", slot="keyed_phone", now=_now)
    blobs = _sealed(store)
    assert sensitive.recall(SID, blobs, slot="keyed_phone") is None   # two values: fail closed
    sensitive.forget(SID)
    assert sensitive.recall(SID, blobs[:1], slot="keyed_phone") is None   # not restored yet
    sensitive.restore(SID, blobs[:1], sealer)
    assert sensitive.recall(SID, blobs[:1], slot="keyed_phone") == "+12025550101"


# --------------------------------------------------------------------------- bounded retries


class _Counting(FakeSealer):
    """Counts unseal calls; fails each with ``error`` (an exception factory) while ``failing``."""

    def __init__(self, error: Any) -> None:
        super().__init__()
        self.calls = 0
        self.error = error
        self.failing = True

    def unseal(self, sealed: str, *, call_sid: str) -> str:
        self.calls += 1
        if self.failing:
            raise self.error()
        return super().unseal(sealed, call_sid=call_sid)


def _failed_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [logutil.fields(r) for r in caplog.records
            if logutil.fields(r).get("event") == "sensitive_unseal_failed"]


@pytest.mark.parametrize("error", [lambda: UnsealError("KMS throttled", transient=True),
                                   lambda: RuntimeError("connect timeout")])
def test_a_transiently_failing_blob_is_tried_three_times_per_call_then_left(
        error: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    store = _store()
    sensitive.remember(store, FakeSealer(), SID, VALUE, slot="ssn4", now=_now)
    sensitive.forget(SID)
    caplog.clear()
    sealer = _Counting(error)
    for _ in range(10):  # ten webhooks of one call
        sensitive.restore(SID, _sealed(store), sealer)
    assert sealer.calls == sensitive.UNSEAL_TRIES == 3
    lines = _failed_lines(caplog)
    assert [(f["attempt"], f["will_retry"]) for f in lines] == [(1, True), (2, True), (3, False)]
    assert sensitive.is_unverifiable(SID)
    sensitive.forget(SID)  # a new call (or the end of this one) starts over
    sensitive.restore(SID, _sealed(store), sealer)
    assert sealer.calls == 4


@pytest.mark.parametrize("error", [lambda: UnsealError(),  # bad ciphertext, wrong context
                                   lambda: UnsealError("token could not be unsealed", transient=False)])
def test_a_non_transient_unseal_failure_is_never_retried(
        error: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    store = _store()
    sensitive.remember(store, FakeSealer(), SID, VALUE, slot="ssn4", now=_now)
    sensitive.forget(SID)
    caplog.clear()
    sealer = _Counting(error)
    for _ in range(5):
        sensitive.restore(SID, _sealed(store), sealer)
    assert sealer.calls == 1
    [line] = _failed_lines(caplog)
    assert line["transient"] is False and line["will_retry"] is False


def test_a_record_that_does_not_parse_is_never_retried(caplog: pytest.LogCaptureFixture) -> None:
    sealer = _Counting(UnsealError)
    sealer.failing = False
    blob = FakeSealer().seal("not a record", call_sid=SID)
    for _ in range(5):
        sensitive.restore(SID, (blob,), sealer)
    assert sealer.calls == 1 and sensitive.is_unverifiable(SID)


def test_a_wrong_context_blob_from_the_fake_sealer_is_not_retried() -> None:
    store, sealer = _store(OTHER), FakeSealer()
    sensitive.remember(store, sealer, OTHER, VALUE, slot="ssn4", now=_now)
    counting = _Counting(UnsealError)
    counting.failing = False  # the real FakeSealer check: the HMAC fails under SID
    for _ in range(4):
        sensitive.restore(SID, _sealed(store, OTHER), counting)
    assert counting.calls == 1 and sensitive.is_unverifiable(SID)


def test_a_successful_retry_does_not_clear_unverifiable() -> None:
    store = _store()
    sensitive.remember(store, FakeSealer(), SID, VALUE, slot="ssn4", now=_now)
    sensitive.forget(SID)
    sealer = _Counting(lambda: UnsealError("KMS throttled", transient=True))
    sensitive.restore(SID, _sealed(store), sealer)
    assert sensitive.is_unverifiable(SID)
    sealer.failing = False
    sensitive.restore(SID, _sealed(store), sealer)
    assert sealer.calls == 2 and sg.sensitive_values(SID) == frozenset({VALUE})
    assert sensitive.is_unverifiable(SID)  # fails closed: stays set until forget
    sensitive.forget(SID)
    assert not sensitive.is_unverifiable(SID)
