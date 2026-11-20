"""The bridge's journal text and the task attributes, with values sealed by a front-door
run.

The front door accepts Thomas's SSN last four and seals it into call state. The bridge is another
process (``tests.component.slot_world.SplitWorld``): it learns the value only by restoring
``sealed_sensitive`` on its call-state reads. An agent that repeats the value (in a spoken turn and
in its hand-off summary) is still heard by the caller as it said it, and the agent's input is never
changed; but the journal's AI turn (``_turn_text``), the escalation summary (``_summary_text``), the
task attributes built from it, the Context API's timeline and every log line carry no value. An
entry that can't be unsealed withholds both texts instead (fail closed).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from owlbank import callstate as cs
from owlbank import segments
from owlbank.pii import WITHHELD_TURN
from owlbank.token_seal import FakeSealer
from tests.component.slot_world import SplitWorld, capture_logs, leaked, surfaces
from tests.e2e.harness import THOMAS, Call, call_sid, run, say, tool

ZIP, SSN4 = "94105", "1111"
REASON = "caller_requested_human"
WITHHELD_SUMMARY = "Summary withheld: PII check unavailable."


def _parrot(rec: dict[str, Any]) -> list[Any]:
    """An agent that repeats the SSN last four it must never repeat."""
    text = rec.get("user_text") or ""
    for res in rec.get("tool_results") or ():
        if isinstance(res.content, dict) and res.content.get("error") == "AUTH_LEVEL_INSUFFICIENT":
            return tool("authenticate", "tu_auth", tool=res.content["tool"])
    if not text:
        return say("<silent/>")
    if "address" in text:
        return tool("change_address", "tu_addr", street="12 Example Lane", city="Springfield",
                    state="CA", postal_code="60601")
    if "person" in text:
        return tool("escalate", "tu_esc", reason=REASON,
                    summary=f"Caller verified with {SSN4} and wants a person.")
    return say(f"Thanks. Your last four are {SSN4}, all set.")


@pytest.fixture
def sw() -> Iterator[SplitWorld]:
    world = SplitWorld()
    world.respond(_parrot)
    yield world
    world.close()


def _call(sw: SplitWorld, sid: str, seen: dict[str, Any], *,
          between: Callable[[], None] | None = None) -> Call:
    async def go() -> Call:
        call = sw.world.dial(THOMAS, sid)
        await call.start()
        await call.press("")
        await call.press(ZIP)
        await call.say("I need to change my address")
        assert call.state == "gather"
        await call.press(SSN4)  # the front door accepts, verifies and seals it
        assert call.state == "relay" and sw.world.state(sid).level == 2
        if between is not None:
            between()
        seen["heard"] = await call.say("is that everything?")
        seen["fd"], seen["bridge"] = sw.fd.values(sid), sw.bridge.values(sid)
        await call.say("a person please")
        assert call.state == "queued"
        await call.hangup()
        return call

    return run(go())


def test_values_sealed_by_the_front_door_are_scrubbed_from_the_bridge_journal_and_task_attributes(
        sw: SplitWorld) -> None:
    sid = call_sid(1)
    seen: dict[str, Any] = {}
    with capture_logs() as logs:
        call = _call(sw, sid, seen)
    assert SSN4 in seen["heard"]  # the caller hears the agent as it spoke
    st = sw.world.state(sid)
    assert len(st.sealed_sensitive) == 1
    # Two registries: the bridge's value came from call state (restore), not the front door's memory.
    assert seen["fd"] == seen["bridge"] == frozenset({segments._norm_text(SSN4)})
    assert sw.fd.registered is not sw.bridge.registered
    # The agent never received the value from the caller's side.
    assert not any(SSN4 in (c["user_text"] or "") for c in sw.world.agent.calls)

    events = sw.world.journal(sid)
    ai_turns = [e["text"] for e in events if e["type"] == "ai_turn"]
    assert "Thanks. Your last four are [redacted], all set." in ai_turns
    summaries = [e["summary"] for e in events if e["type"] == "escalation" and "summary" in e]
    assert summaries == ["Caller verified with [redacted] and wants a person."]
    assert call.task is not None and call.task["ai_summary"] == summaries[0]
    ref = call.task["journal_ref"]
    resp = sw.world.context(ref, sid)
    assert resp["statusCode"] == 200 and SSN4 not in (resp["body"] or "")

    found = surfaces(sw, logs, [sid])
    # The agent said it and the caller heard it: what is spoken (the relay frames) and the agent's
    # own history (its earlier replies) are not changed by the scrub. Everything else is clean.
    found.pop("relay frames")
    found.pop("agent input")
    found["context api"] = [resp["body"] or ""]
    assert leaked({SSN4}, found) == []


def test_an_entry_the_bridge_cannot_unseal_withholds_both_texts(sw: SplitWorld) -> None:
    sid = call_sid(2)

    def foreign_blob() -> None:
        blob = FakeSealer(b"another-key").seal(
            json.dumps({"v": 1, "slot": "ssn4", "value": SSN4}), call_sid=sid)
        cs.mutate(sw.world.store, sid,
                  lambda s: {cs.F_SEALED_SENSITIVE: (*s.sealed_sensitive, blob)},
                  now=sw.world.clock)

    with capture_logs() as logs:
        call = _call(sw, sid, {}, between=foreign_blob)
    events = sw.world.journal(sid)
    ai_turns = [e["text"] for e in events if e["type"] == "ai_turn"]
    assert ai_turns and ai_turns[-1] == WITHHELD_TURN
    summaries = [e["summary"] for e in events if e["type"] == "escalation" and "summary" in e]
    assert summaries == [WITHHELD_SUMMARY]
    assert call.task is not None and call.task["ai_summary"] == WITHHELD_SUMMARY
    decisions = [r.get("decision") for r in logs.records if r.get("event") == "sensitive_unseal_failed"]
    assert decisions and set(decisions) == {"withhold_texts"}
    found = surfaces(sw, logs, [sid])
    found.pop("relay frames")
    found.pop("agent input")
    assert leaked({SSN4}, found) == []
    assert segments.sensitive_values(sid) == frozenset()
