"""A caught error logs its stack and the decision, and never the exception message.

The error messages below carry everything the leak scan hunts for: a factor value, the one-time code, a
payment confirmation, a raw call token, a full phone number and the caller's own words. The ``world``
fixture scans every record of the run (DEBUG included) at teardown; these tests also check the shape of
the caught-error records and the real JSON pipeline's output.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from owlbank import obs
from tac_bridge.agent_client import AgentError
from tests.e2e.harness import CONFIRMATION, THOMAS, World, call_sid, factor, run
from tests.e2e.test_core_journeys import login_thomas
from tests.logutil import fields

WORDS = "my daughter's school fund is at the Maple Street branch"
TOKEN = "tok_" + "Zq9secretNonceValue_abcdefgh123456"  # noqa: S105 (a fake token for the leak scan)


def _leaky(world: World) -> str:
    zip_code = factor("C-1001", "zip")
    world.said.append(WORDS)
    world.raw_tokens.append(TOKEN)
    return f"failed for {THOMAS}: zip {zip_code}, code {CONFIRMATION}, token {TOKEN}, said '{WORDS}'"


def _records(log_capture, event: str):
    return [r for r in log_capture.records if getattr(r, "event", None) == event or r.getMessage() == event]


def _assert_stack_without_message(world: World, record: logging.LogRecord, leaky: str) -> None:
    stack = record.__dict__["stack"]
    assert stack.startswith("Traceback (most recent call last):") and "File " in stack
    assert "AgentError" in stack or "RuntimeError" in stack  # the class names are there
    for piece in (leaky, WORDS, TOKEN, THOMAS, CONFIRMATION, factor("C-1001", "zip"), *world.factor_values()):
        assert piece not in stack
    assert len(stack) <= 8192


def test_a_caught_agent_error_logs_its_stack_and_decision_but_not_its_message(world: World, log_capture) -> None:
    sid = call_sid(71)
    leaky = _leaky(world)
    world.agent.queue(AgentError(leaky, retryable=False, code="boom"))

    async def go():
        call = await login_thomas(world, sid)
        await call.say("Checking, please.")
        await call.hangup()

    run(go())
    (rec,) = _records(log_capture, "agent turn failed")
    f = fields(rec)
    assert rec.levelno == logging.WARNING and f["decision"] == "escalate" and f["error_class"] == "AgentError"
    assert f["error_code"] == "boom" and f["operation"] == "agent_turn" and f["attempt"] == 1
    _assert_stack_without_message(world, rec, leaky)
    assert "error_message" not in f and "exc_info" not in f
    assert rec.call_sid == sid  # still correlated


@pytest.mark.protocol_fake("injects a raw RuntimeError from the agent; the Strands client maps every model "
                           "failure to AgentError, so it would be a retried agent error instead")
def test_an_unexpected_bridge_error_logs_its_stack_at_error_level(world: World, log_capture) -> None:
    sid = call_sid(72)
    leaky = _leaky(world)
    world.agent.queue(RuntimeError(leaky))

    async def go():
        call = await login_thomas(world, sid)
        await call.say("Checking, please.")
        await call.hangup()

    run(go())
    (rec,) = _records(log_capture, "bridge turn failed; escalating")
    assert rec.levelno == logging.ERROR and fields(rec)["decision"] == "escalate"
    _assert_stack_without_message(world, rec, leaky)
    assert "error_message" not in fields(rec)


def test_the_json_output_of_a_caught_error_has_the_stack_field_and_no_message(world: World, log_capture) -> None:
    stream = io.StringIO()
    obs.configure_logging("e2e", level=logging.DEBUG, stream=stream)
    sid = call_sid(73)
    leaky = _leaky(world)
    world.agent.queue(AgentError(leaky, retryable=False, code="boom"))

    async def go():
        call = await login_thomas(world, sid)
        await call.say("Checking, please.")
        await call.hangup()

    run(go())
    lines = [json.loads(ln) for ln in stream.getvalue().splitlines() if ln.strip()]
    (line,) = [ln for ln in lines if ln["event"] == "agent turn failed"]
    assert line["decision"] == "escalate" and line["error_class"] == "AgentError"
    assert isinstance(line["stack"], str) and "Traceback" in line["stack"]
    # a non-sensitive slot's value is allowed in its factor_answer line's ``value``.
    open_slots = world.nonsensitive_slots()
    kept = [{k: v for k, v in ln.items()
             if not (k == "value" and ln.get("event") == "factor_answer" and ln.get("factor") in open_slots)}
            for ln in lines]
    raw = "\n".join(json.dumps(ln, ensure_ascii=False) for ln in kept)
    for piece in (WORDS, TOKEN, CONFIRMATION, THOMAS, factor("C-1001", "zip")):
        assert piece not in raw
