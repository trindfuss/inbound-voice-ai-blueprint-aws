"""``send_otp(record=True)`` writes its outcome to the call state for the front door."""

from __future__ import annotations

import logging

import pytest

from lambdas.auth_service import service
from lambdas.auth_service.service import AuthDeps
from owlbank import callstate
from owlbank.callstate import CallState

from tests.unit.lambdas.test_auth_service import ANI, NOW, THOMAS, FakeVerify, make_deps, verify

SID = "CA1"


def with_call(deps: AuthDeps, *, level: int = 2) -> AuthDeps:
    callstate.create(deps.store, CallState(call_sid=SID, expires_at=NOW + 86400))
    if level:
        assert verify(deps, THOMAS, level=level)["result"] == "verified"
    return deps


def otp_send(deps: AuthDeps) -> object:
    st = callstate.load(deps.store, SID)
    assert st is not None
    return st.otp_send


def assert_only_status(deps: AuthDeps, status: str) -> None:
    rec = otp_send(deps)
    assert rec == {"status": status, "at": NOW}
    assert isinstance(rec["at"], int)  # type: ignore[index]
    item = deps.store.get("calls", {"call_sid": SID})
    assert item is not None
    text = repr(item["otp_send"])
    assert ANI[THOMAS] not in text and "123456" not in text  # never a phone or a code


def test_sent_is_recorded() -> None:
    deps = with_call(make_deps())
    out = service.send_otp(deps, call_sid=SID, ani=ANI[THOMAS], record=True)
    assert out == {"result": "sent"}
    assert_only_status(deps, "sent")


def test_escalate_is_recorded_for_an_unknown_number() -> None:
    deps = with_call(make_deps())
    out = service.send_otp(deps, call_sid=SID, ani="+12025550177", record=True)
    assert out["result"] == "escalate"
    assert_only_status(deps, "escalate")


def test_escalate_is_recorded_when_verify_cannot_deliver() -> None:
    deps = with_call(make_deps(verify=FakeVerify(fail=True)))
    assert service.send_otp(deps, call_sid=SID, ani=ANI[THOMAS], record=True)["result"] == "escalate"
    assert_only_status(deps, "escalate")


@pytest.mark.parametrize("why", ["no_token", "unavailable", "rate_limited"])
def test_anything_else_is_recorded_as_error(why: str) -> None:
    deps = with_call(make_deps(), level=0 if why == "no_token" else 2)
    if why == "unavailable":
        deps.verify_client = None
    if why == "rate_limited":
        for _ in range(service._limit(deps.policy, "otp_sends_per_call")):  # noqa: SLF001
            service.send_otp(deps, call_sid=SID, ani=ANI[THOMAS])
    out = service.send_otp(deps, call_sid=SID, ani=ANI[THOMAS], record=True)
    assert out["result"] in {"not_verified", "unavailable", "rate_limited"}
    assert_only_status(deps, "error")


def test_without_record_nothing_is_written() -> None:
    deps = with_call(make_deps())
    before = callstate.load(deps.store, SID)
    assert service.send_otp(deps, call_sid=SID, ani=ANI[THOMAS]) == {"result": "sent"}
    after = callstate.load(deps.store, SID)
    assert after is not None and before is not None
    assert after.otp_send is None and after.version == before.version


def test_the_event_field_turns_recording_on_only_when_true() -> None:
    deps = with_call(make_deps())
    base = {"op": "send_otp", "call_sid": SID, "ani": ANI[THOMAS], "language": "en-US"}
    assert service.handle_event(deps, {**base, "record": "yes"}) == {"result": "sent"}
    assert otp_send(deps) is None
    assert service.handle_event(deps, {**base, "record": True}) == {"result": "sent"}
    assert_only_status(deps, "sent")


def test_a_missing_call_state_is_logged_and_the_result_still_returned(
        caplog: pytest.LogCaptureFixture) -> None:
    deps = make_deps()
    verify(deps, THOMAS, level=2)
    caplog.set_level(logging.INFO)
    assert service.send_otp(deps, call_sid=SID, ani=ANI[THOMAS], record=True) == {"result": "sent"}
    (rec,) = [r for r in caplog.records if r.getMessage() == "otp_send_not_recorded"]
    assert rec.levelno == logging.ERROR and rec.decision == "frontdoor_times_out"
    assert deps.store.get("calls", {"call_sid": SID}) is None


def test_an_invalid_call_sid_writes_nothing() -> None:
    deps = make_deps()
    out = service.send_otp(deps, call_sid="not-a-sid#x", ani=ANI[THOMAS], record=True)
    assert out["result"] == "bad_request"
