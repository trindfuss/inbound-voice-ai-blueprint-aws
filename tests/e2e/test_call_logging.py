"""One call is traceable across the front door, auth service, bank tools and bridge.

The ``world`` fixture already scans every log record of every journey at DEBUG for factor values,
tokens, full phone numbers, confirmation codes and message text, and checks that each call's SID
is on the records of the components that handled it. These tests add the strict version for one
full journey, and run the real JSON pipeline (formatter + scrubber) over the whole run.
"""

from __future__ import annotations

import io
import json
import logging

from owlbank import obs
from tac_bridge.agent_client import TextDelta, TurnEnd
from tests.e2e.harness import CONFIRMATION, THOMAS, call_sid, run, say, tool
from tests.e2e.test_core_journeys import login_thomas
from tests.logutil import fields


def _components(records, sid):
    return {r.component for r in records if getattr(r, "call_sid", None) == sid and getattr(r, "component", None)}


def test_a_balance_call_is_correlated_across_every_component(world, log_capture) -> None:
    sid = call_sid(61)
    world.agent.queue(tool("get_balance", "tu_r6", account="checking"), say("<silent/>"), say("Anything else?"))

    async def go():
        call = await login_thomas(world, sid)
        await call.say("What is my checking balance?")  # handed off to the front door, read, reconnected
        await call.say("No, that is all.")
        await call.hangup()

    run(go())
    assert {"frontdoor", "auth_service", "bridge"} <= _components(log_capture.records, sid)
    events = {(r.component, getattr(r, "event", r.getMessage())) for r in log_capture.records
              if getattr(r, "call_sid", None) == sid}
    assert ("frontdoor", "invocation_start") in events and ("frontdoor", "invocation_end") in events
    assert ("auth_service", "invocation_end") in events
    assert ("bridge", "ws_session_start") in events and ("bridge", "ws_session_end") in events
    assert ("bridge", "turn") in events


def test_the_bridge_logs_one_turn_line_per_prompt_with_counts_and_no_text(world, log_capture) -> None:
    sid = call_sid(62)
    world.agent.queue(tool("get_balance", "tu_r6b", account="savings"), say("<silent/>"), say("Anything else?"))

    async def go():
        call = await login_thomas(world, sid)
        await call.say("Savings please.")
        await call.say("No, that is all.")
        await call.hangup()

    run(go())
    turns = [fields(r) | {"_level": r.levelname} for r in log_capture.records
             if r.name == "tac_bridge.bridge" and getattr(r, "event", "") == "turn"]
    assert turns, "no turn line"
    for t in turns:
        assert {"turn", "duration_ms", "stop_reason", "tools", "outcome"} <= set(t)
        assert isinstance(t["tools"], list)
    assert any("get_balance" in t["tools"] for t in turns)
    ends = [fields(r) for r in log_capture.records if getattr(r, "event", "") == "ws_session_end"
            and getattr(r, "call_sid", None) == sid]
    starts = [r for r in log_capture.records if getattr(r, "event", "") == "ws_session_start"
              and getattr(r, "call_sid", None) == sid]
    assert len(starts) == len(ends) == 2  # the relay reopens after the handoff
    assert all(e["reason"] == "closed" and e["duration_ms"] >= 0 for e in ends)
    assert sum(e["prompts"] for e in ends) >= 2 and sum(e["tool_calls"] for e in ends) >= 1


def test_the_json_pipeline_output_is_clean_and_carries_the_call_sid(world, log_capture, caplog) -> None:
    """The real formatter and scrubber, over the whole journey: JSON lines, the call SID, no secrets."""
    stream = io.StringIO()
    obs.configure_logging("e2e", level=logging.DEBUG, stream=stream)
    sid = call_sid(63)
    world.agent.queue(tool("get_balance", "tu_r6c", account="checking"), say("<silent/>"), say("Done."))

    async def go():
        call = await login_thomas(world, sid)
        await call.say("Checking, please.")
        await call.say("No, that is all.")
        await call.hangup()

    run(go())
    lines = [json.loads(ln) for ln in stream.getvalue().splitlines() if ln.strip()]  # every line is JSON
    assert lines and all({"ts", "level", "component", "event"} <= set(ln) for ln in lines)
    mine = [ln for ln in lines if ln.get("call_sid") == sid]
    assert {"frontdoor", "auth_service", "bridge"} <= {ln["component"] for ln in mine}
    # Every string value (timestamps and durations are numbers or time text that can hold any digits).
    # a non-sensitive slot's value is allowed in its factor_answer line's ``value``.
    open_slots = world.nonsensitive_slots()

    def sanctioned(ln: dict, k: str) -> bool:
        return k == "value" and ln.get("event") == "factor_answer" and ln.get("factor") in open_slots

    text = "\n".join(str(v) for ln in lines for k, v in ln.items()
                     if k not in ("ts", "duration_ms") and isinstance(v, str) and not sanctioned(ln, k))
    for secret in (CONFIRMATION, *world.raw_tokens, *world.factor_values(), *world.sealed_values()):
        assert secret not in text
    assert THOMAS not in text and "+1202555" not in text.replace("+*******", "")


def test_botocore_and_friends_stay_off_at_debug(log_capture) -> None:
    obs.configure_logging("e2e", level=logging.DEBUG, stream=io.StringIO())
    for name in ("botocore", "boto3", "urllib3", "s3transfer", "hpack", "botocore.hooks", "urllib3.connectionpool"):
        assert not logging.getLogger(name).isEnabledFor(logging.DEBUG), name


def test_an_escalation_logs_its_reason_and_source(world, log_capture) -> None:
    sid = call_sid(64)
    world.agent.queue(tool("escalate", "tu_r6esc", reason="caller_requested_human",
                           summary="Caller asked to speak to a person."))

    async def go():
        call = await login_thomas(world, sid)
        await call.say("I want to speak to a person.")
        await call.hangup()

    run(go())
    esc = [(fields(r), r.component) for r in log_capture.records
           if getattr(r, "event", "") == "escalation" and getattr(r, "call_sid", None) == sid]
    assert [(f["reason"], f["source"], comp) for f, comp in esc] == [("caller_requested_human", "frontdoor", "frontdoor")]
    requested = [fields(r) for r in log_capture.records
                 if getattr(r, "event", "") == "escalation_requested" and getattr(r, "call_sid", None) == sid]
    assert [(f["reason"], f["source"]) for f in requested] == [("caller_requested_human", "agent")]


def test_the_turn_line_carries_token_counts_when_the_stream_has_them(world, log_capture) -> None:
    sid = call_sid(65)
    world.agent.queue([TextDelta("How can I help?"), TurnEnd("end_turn", {"input_tokens": 120, "output_tokens": 15})])

    async def go():
        call = await login_thomas(world, sid)
        await call.say("Hello there.")
        await call.hangup()

    run(go())
    (turn,) = [fields(r) for r in log_capture.records if getattr(r, "event", "") == "turn"
               and getattr(r, "call_sid", None) == sid and not fields(r)["resumed"]]
    assert turn["input_tokens"] == 120 and turn["output_tokens"] == 15 and turn["stop_reason"] == "end_turn"
    assert turn["tools"] == [] and turn["outcome"] == "ok"
