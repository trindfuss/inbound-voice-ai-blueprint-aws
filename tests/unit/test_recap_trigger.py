"""Unit tests for owlbank.recap_trigger (section 5): the Event invoke,
the no-op, :func:`fire` never raising, :func:`at_status`, and the wiring in the front door and the sweep."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from lambdas.conversation_sweep import handler as sweep
from lambdas.frontdoor.handler import human_leg_client, human_leg_closer
from owlbank import humanleg
from owlbank.recap_trigger import (
    RECAP_FUNCTION_ENV,
    EventRecap,
    NoopRecap,
    at_status,
    client_config,
    fire,
    recap_trigger,
)
from owlbank.store import MemoryStore

CALL = "CA" + "1" * 32
FUNCTION = "arn:aws:lambda:us-east-1:111122223333:function:owlbank-recap-writer"
SECRET = "arn:aws:secretsmanager:us-east-1:111122223333:secret:twilio-api-key"


class FakeLambda:
    def __init__(self, status: int = 202, error: Exception | None = None) -> None:
        self.status, self.error = status, error
        self.calls: list[dict[str, Any]] = []

    def invoke(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(kw)
        if self.error is not None:
            raise self.error
        return {"StatusCode": self.status}


def test_event_recap_invokes_async_with_the_call_sid_only() -> None:
    lam = FakeLambda()
    EventRecap(lam, FUNCTION)(CALL)
    assert len(lam.calls) == 1
    call = lam.calls[0]
    assert call["FunctionName"] == FUNCTION and call["InvocationType"] == "Event"
    assert json.loads(call["Payload"]) == {"call_sid": CALL}


@pytest.mark.parametrize("lam", [FakeLambda(status=200), FakeLambda(status=500),
                                 FakeLambda(error=TimeoutError("read timeout"))])
def test_event_recap_raises_when_not_accepted(lam: FakeLambda) -> None:
    with pytest.raises((RuntimeError, TimeoutError)):
        EventRecap(lam, FUNCTION)(CALL)


def test_event_recap_needs_a_function() -> None:
    with pytest.raises(ValueError):
        EventRecap(FakeLambda(), "")


def test_recap_trigger_is_noop_without_the_function_and_builds_no_client() -> None:
    built: list[int] = []
    for env in ({}, {RECAP_FUNCTION_ENV: ""}, {RECAP_FUNCTION_ENV: "  "}):
        assert isinstance(recap_trigger(env, lambda: built.append(1)), NoopRecap)
    assert built == []
    lam = FakeLambda()
    trigger = recap_trigger({RECAP_FUNCTION_ENV: FUNCTION}, lambda: lam)
    assert isinstance(trigger, EventRecap)
    trigger(CALL)
    assert lam.calls[0]["FunctionName"] == FUNCTION


def test_fire_never_raises_and_logs_the_failure(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    lam = FakeLambda(error=ConnectionError("down"))
    assert fire(EventRecap(lam, FUNCTION), CALL, source="status") is False
    line = [r for r in caplog.records if r.getMessage() == "recap_trigger_failed"][-1]
    assert getattr(line, "call_sid", None) == CALL and getattr(line, "source", None) == "status"
    assert getattr(line, "decision", None) == "ignored_no_recap"
    assert fire(EventRecap(FakeLambda(), FUNCTION), CALL, source="close") is True
    assert fire(NoopRecap(), CALL, source="close") is True
    assert fire(None, CALL, source="close") is False


def test_fire_sends_nothing_for_a_malformed_call_sid() -> None:
    lam = FakeLambda()
    assert fire(EventRecap(lam, FUNCTION), "not-a-sid", source="status") is False
    assert lam.calls == []


ESC = {"reason": "fraud_suspected", "queue": "q", "journal_ref": "jr"}


@pytest.mark.parametrize(("escalation", "ai_sessions", "session_conv", "wired", "expected"), [
    (None, 1, None, True, True),                                     # not escalated
    ({}, 1, "conv_x", True, True),                                    # not escalated
    (ESC, 1, None, False, True),                                      # no close path wired
    (ESC, 0, None, True, True),                                       # escalated, no conversation
    (ESC, 1, None, True, False),                                      # the closer looks it up
    (ESC, 0, "conv_x", True, False),                                  # known to the session
    ({**ESC, humanleg.K_CONVERSATION_ID: "conv_x"}, 0, None, True, False),  # recorded
    ({**ESC, humanleg.K_CONVERSATION_ID: "conv_x", humanleg.K_CONVERSATION_CLOSED_AT: 1}, 1, None, True,
     False),                                                          # already closed: that close fired
])
def test_at_status(escalation: Any, ai_sessions: int, session_conv: str | None, wired: bool,
                   expected: bool) -> None:
    assert at_status(escalation, ai_sessions=ai_sessions, session_conversation=session_conv,
                     closer_wired=wired) is expected


def test_front_door_closer_gets_the_trigger() -> None:
    client = human_leg_client({"OWLBANK_HUMAN_LEG_CONVERSATION": "on"}, SECRET)
    trigger = NoopRecap()
    closer = human_leg_closer({"OWLBANK_HUMAN_LEG_CONVERSATION": "on"}, SECRET, object(), client, recap=trigger)
    assert closer is not None and closer._recap is trigger  # noqa: SLF001
    plain = human_leg_closer({"OWLBANK_HUMAN_LEG_CONVERSATION": "on"}, SECRET, object(), client)
    assert plain is not None and plain._recap is None  # noqa: SLF001


def test_sweep_closer_gets_the_trigger() -> None:
    env = {sweep.API_KEY_SECRET_ENV: SECRET}
    unset = sweep.build_deps(env, store=MemoryStore())
    assert isinstance(unset.closer._recap, NoopRecap)  # type: ignore[attr-defined]  # noqa: SLF001
    trigger = NoopRecap()
    given = sweep.build_deps(env, store=MemoryStore(), recap=trigger)
    assert given.closer._recap is trigger  # type: ignore[attr-defined]  # noqa: SLF001


def test_sweep_builds_an_event_recap_from_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    lam = FakeLambda()
    monkeypatch.setattr(sweep, "_lambda_client", lambda: lam)
    deps = sweep.build_deps({sweep.API_KEY_SECRET_ENV: SECRET, RECAP_FUNCTION_ENV: FUNCTION},
                            store=MemoryStore())
    trigger = deps.closer._recap  # type: ignore[attr-defined]  # noqa: SLF001
    assert isinstance(trigger, EventRecap)
    trigger(CALL)
    assert lam.calls[0]["FunctionName"] == FUNCTION


def test_the_invoke_client_config_is_one_short_attempt() -> None:
    """0.3 s to connect, 0.7 s to read, one attempt in total."""
    config = client_config()
    assert (config.connect_timeout, config.read_timeout) == (0.3, 0.7)
    assert config.retries == {"mode": "standard", "total_max_attempts": 1}


def test_the_sweeps_lambda_client_uses_the_recap_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3

    built: list[dict[str, Any]] = []
    monkeypatch.setattr(boto3, "client", lambda service, **kw: built.append({"service": service, **kw})
                        or FakeLambda())
    client = sweep._lambda_client()  # noqa: SLF001
    assert built == []  # lazy: nothing built until the first invoke
    client.invoke(FunctionName=FUNCTION, InvocationType="Event", Payload=b"{}")
    assert len(built) == 1 and built[0]["service"] == "lambda"
    config = built[0]["config"]
    assert (config.connect_timeout, config.read_timeout) == (0.3, 0.7)
    assert config.retries == {"mode": "standard", "total_max_attempts": 1}


def test_fire_logs_the_sweep_gave_up_source(caplog: pytest.LogCaptureFixture) -> None:
    lam = FakeLambda(error=TimeoutError("read timeout"))
    assert fire(EventRecap(lam, FUNCTION), CALL, source="sweep_gave_up") is False
    line = [r for r in caplog.records if r.getMessage() == "recap_trigger_failed"][-1]
    assert getattr(line, "source", None) == "sweep_gave_up"


def test_sweep_deps_keep_the_trigger_for_giving_up() -> None:
    trigger = NoopRecap()
    deps = sweep.build_deps({sweep.API_KEY_SECRET_ENV: SECRET}, store=MemoryStore(), recap=trigger)
    assert deps.recap is trigger
