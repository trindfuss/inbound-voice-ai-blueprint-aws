"""Call observability: the bridge's spans over the ConversationRelay rig.

The bridge runs the real Strands client (:class:`StrandsAgentClient` on a scripted ``FakeModel``)
under TAC and the CR simulator, with the telemetry installed as ``install_bridge`` builds it
(baggage, evaluation-shape and batch processors over the content filter) and an in-memory exporter
(``tests.strands_fakes.traced``). Checked here:

* the span tree of a greeting, an inline tool, a Gateway tool, a step-up with a resumed turn, and a
  hand-off: ``relay.session`` and each ``bridge.turn`` are roots of their own traces,
  Strands' spans nest under the turn, and every span carries ``session.id == call_sid``;
* masking events carry the catalog key, never the phrase text;
* interrupt attributes and relay ``info``/``error`` events;
* the ``strands_model_call`` line for a failed model call, with telemetry on and off, and the
  provider's timings on a successful one (from the message or the stream's ``metadata`` chunk);
  a failing hook is logged at most once a minute and the turn goes on;
* a ``memory.reload`` span for each history reload from Memory, at session start (under the
  session span) and in a turn (under the turn span), with the call and its AWS calls under it;
* the ``turn`` line's ``silence_ms`` and ``wait_ms`` with telemetry off too, and the
  ``ws_session_end`` token totals.

No property-based tests here: these are fixed scenarios (the properties are in
``tests/property/test_bridge_telemetry_props.py``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from owlbank.envelope import ResultEnvelope, Spoken
from tac_bridge.agent_client_strands import StrandsAgentClient
from tac_bridge.app import _close_code
from tests.component.test_bridge_fakes import (
    CALL_SID,
    FakeBackend,
    front_door_update,
    make_rig,
    rearm_nonce,
    run,
    spoken,
    write_call_state,
)
from tests.logutil import lines
from tests.strands_fakes import (
    FakeModel,
    ScriptError,
    Traced,
    default_gateway,
    descendants,
    settings,
    text_msg,
    tool_msg,
    traced,
)

BRIDGE_LOG = "tac_bridge.bridge"
CLIENT_LOG = "tac_bridge.agent_client_strands"
TXNS = {"account": "checking", "transactions": [
    {"date": "2026-10-09", "description": "Coffee Example Co", "amount": "-4.50"}]}
STRANDS_OPS = {"invoke_agent", "execute_event_loop_cycle", "chat", "execute_tool"}


# --------------------------------------------------------------------------- helpers


def raw_rig(model_calls: list[Any], *, delay: float = 0.0, **kw: Any) -> Any:
    """The CR rig with the real Strands client on a scripted model (one entry per model call)."""
    rig = make_rig(**kw)
    model = FakeModel(model_calls, delay=delay)
    rig.bridge._agent = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: model,
                                           gateway_tools=default_gateway())
    rig.model = model
    return rig


async def one_session(rig: Any, *prompts: str, end: str = "disconnect", resume: bool = False) -> list[Any]:
    """Connect, send ``prompts`` (or wait for the resume at setup), hang up, and end the session as
    the app's ``/ws`` handler does. Returns every frame sent."""
    sim = rig.sim()
    await sim.connect()
    frames: list[Any] = []
    if resume:
        frames += await sim.wait_reply()
    for p in prompts:
        frames += await sim.prompt(p)
    await sim.hangup()
    rig.bridge.session_end(CALL_SID, end, close_code=1000)
    await rig.bridge.wait_idle()
    return frames


def by_name(spans: list[Any], name: str) -> list[Any]:
    return [s for s in spans if s.name == name or s.name.startswith(name + " ")]


def op(span: Any) -> str | None:
    return (span.attributes or {}).get("gen_ai.operation.name")


def events_of(span: Any, name: str) -> list[Any]:
    return [e for e in span.events if e.name == name]


def assert_session_everywhere(spans: list[Any]) -> None:
    """Every exported span carries the call."""
    assert spans
    for s in spans:
        assert s.attributes.get("session.id") == CALL_SID, s.name
        assert s.attributes.get("call_sid") == CALL_SID, s.name


def assert_roots(spans: list[Any]) -> tuple[list[Any], list[Any]]:
    """``relay.session`` and ``bridge.turn`` spans are roots, each in its own trace, and
    every Strands span sits under a turn."""
    sessions, turns = by_name(spans, "relay.session"), by_name(spans, "bridge.turn")
    for s in sessions + turns:
        assert s.parent is None, s.name
    assert len({s.context.trace_id for s in sessions + turns}) == len(sessions + turns)
    under_turns = {d.context.span_id for t in turns for d in descendants(spans, t)}
    for s in spans:
        if op(s) in STRANDS_OPS:
            assert s.context.span_id in under_turns, s.name
    return sessions, turns


def turn_tree(spans: list[Any], turn: Any) -> list[Any]:
    return descendants(spans, turn)


# --------------------------------------------------------------------------- span trees


def test_greeting_session_and_turn_are_separate_roots_with_strands_spans_under_the_turn() -> None:
    rig = raw_rig([text_msg("Hello there.")])
    write_call_state(rig.store)
    with traced() as t:
        t.enable(rig.bridge)
        frames = run(one_session(rig, "hi"))
        spans = t.spans()
    assert spoken(frames) == "Hello there."
    assert_session_everywhere(spans)
    (session,), (turn,) = assert_roots(spans)
    tree = turn_tree(spans, turn)
    assert sorted(op(s) for s in tree) == ["chat", "execute_event_loop_cycle", "invoke_agent"]
    a = turn.attributes
    assert (a["owlbank.turn"], a["owlbank.hops"], a["owlbank.outcome"], a["owlbank.resumed"]) == (1, 1, "ok", False)
    assert a["owlbank.stop_reason"] == "end_turn" and a["owlbank.tools"] == ()
    assert (a["owlbank.input_tokens"], a["owlbank.output_tokens"]) == (10, 5)
    assert a["owlbank.silence_ms"] >= 0 and a["owlbank.wait_ms"] == a["owlbank.silence_ms"]
    s = session.attributes
    assert (s["owlbank.end_reason"], s["owlbank.turns"], s["owlbank.close_code"]) == ("disconnect", 1, 1000)
    assert (s["owlbank.input_tokens"], s["owlbank.output_tokens"]) == (10, 5)
    assert "owlbank.handoff_tool" not in s
    # Strands' own attributes on its spans: trace_attributes and the GenAI keys.
    (invoke,) = [x for x in tree if op(x) == "invoke_agent"]
    assert invoke.attributes["gen_ai.request.model"] == "fake"


def test_inline_tool_turn_has_two_hops_a_tool_span_and_mask_events_by_key() -> None:
    backend = FakeBackend({"list_recent_transactions": [TXNS]})
    rig = raw_rig([tool_msg(("tu_tx", "list_recent_transactions", {"account": "checking", "limit": 3})),
                   text_msg("Your latest was four fifty.")], backend=backend)
    write_call_state(rig.store)
    with traced() as t:
        t.enable(rig.bridge)
        frames = run(one_session(rig, "What did I spend recently?"))
        spans = t.spans()
    said = spoken(frames)
    assert said.endswith("Your latest was four fifty.")
    assert_session_everywhere(spans)
    _, (turn,) = assert_roots(spans)
    assert turn.attributes["owlbank.hops"] == 2
    assert turn.attributes["owlbank.tools"] == ("list_recent_transactions",)
    tree = turn_tree(spans, turn)
    assert len([s for s in tree if op(s) == "invoke_agent"]) == 2  # one per hop
    tools = [s for s in tree if op(s) == "execute_tool"]
    assert tools and {s.attributes["gen_ai.tool.name"] for s in tools} == {"list_recent_transactions"}
    assert {s.attributes.get("owlbank.tool_kind") for s in tools} == {"inline"}
    # the masking phrase is an event with its catalog key, never its text.
    masks = events_of(turn, "mask_spoken")
    phrase = said[: -len("Your latest was four fifty.")].strip()
    assert masks and phrase  # the default policy masks this tool
    for e in masks:
        assert e.attributes["owlbank.key"].startswith("mask.")
        assert e.attributes["owlbank.kind"] in ("tool", "timer")
        assert phrase not in json.dumps(dict(e.attributes))
    # the caller heard the phrase before the answer
    assert turn.attributes["owlbank.silence_ms"] <= turn.attributes["owlbank.wait_ms"]


def test_gateway_tool_span_has_its_kind_and_outcome() -> None:
    rig = raw_rig([tool_msg(("tg", "level0___faq_lookup", {"q": "fees"})), text_msg("No fee.")])
    write_call_state(rig.store)
    with traced() as t:
        t.enable(rig.bridge)
        run(one_session(rig, "any fees?"))
        spans = t.spans()
    assert_session_everywhere(spans)
    _, (turn,) = assert_roots(spans)
    assert turn.attributes["owlbank.hops"] == 1 and turn.attributes["owlbank.tools"] == ()
    (gw,) = [s for s in turn_tree(spans, turn) if op(s) == "execute_tool"]
    a = gw.attributes
    assert a["gen_ai.tool.name"] == "level0___faq_lookup"
    assert (a["owlbank.tool_kind"], a["owlbank.outcome"]) == ("gateway", "success")
    assert a["gen_ai.tool.status"] == "success"
    assert "owlbank.result_next_turn" not in a  # it had its own result


def test_step_up_then_resumed_turn_and_confirm_handoff() -> None:
    verified = ResultEnvelope(status="verified", tool="authenticate",
                              spoken=[Spoken("stepup.success", "Thanks, you're verified.")],
                              data={"forTool": "make_payment"}, next="retry_original_tool",
                              forTool="make_payment").to_json()
    rig = raw_rig([tool_msg(("tu_auth", "authenticate", {"tool": "make_payment"})),
                   tool_msg(("tu_pay", "make_payment",
                             {"from_account": "checking", "to": "red", "amount": "250.00"}))])
    write_call_state(rig.store)
    with traced() as t:
        t.enable(rig.bridge)
        run(one_session(rig, "Pay 250 dollars on my card from checking", end="closed"))
        front_door_update(rig.store, pending_result=json.loads(verified), level=3)
        rearm_nonce(rig.store)
        run(one_session(rig, resume=True, end="closed"))
        spans = t.spans()
    assert_session_everywhere(spans)
    sessions, turns = assert_roots(spans)
    assert len(sessions) == 2 and len(turns) == 2
    first, second = sorted(sessions, key=lambda s: s.start_time)
    assert first.attributes["owlbank.handoff_tool"] == "authenticate"
    assert second.attributes["owlbank.handoff_tool"] == "make_payment"
    assert second.attributes.get("owlbank.resumed") is True  # a re-entry after an AI session
    t1, t2 = sorted(turns, key=lambda s: s.start_time)
    assert (t1.attributes["owlbank.resumed"], t2.attributes["owlbank.resumed"]) == (False, True)
    assert "owlbank.wait_ms" not in t2.attributes  # a hand-off: no answer text
    # the resumed turn's invoke_agent has a user message (added at span start), and the
    # inline tool that ended on an interrupt has a choice and a status (filled at export).
    (resumed,) = [s for s in turn_tree(spans, t2) if op(s) == "invoke_agent"]
    assert events_of(resumed, "gen_ai.user.message")
    interrupted = [s for s in turn_tree(spans, t1) if op(s) == "execute_tool"]
    assert [s.attributes["gen_ai.tool.name"] for s in interrupted] == ["authenticate"]
    (auth,) = interrupted
    assert events_of(auth, "gen_ai.choice") and auth.attributes["gen_ai.tool.status"]
    assert auth.attributes["owlbank.result_next_turn"] is True
    assert auth.attributes["owlbank.tool_kind"] == "handoff"


def test_escalation_handoff_names_the_tool_on_the_session() -> None:
    rig = raw_rig([tool_msg(("tu_esc", "escalate", {"reason": "caller_requested_human",
                                                    "summary": "wants a person"}))])
    write_call_state(rig.store)
    with traced() as t:
        t.enable(rig.bridge)
        run(one_session(rig, "a person please", end="closed"))
        spans = t.spans()
    assert_session_everywhere(spans)
    (session,), (turn,) = assert_roots(spans)
    assert session.attributes["owlbank.handoff_tool"] == "escalate"
    assert turn.attributes["owlbank.tools"] == ("escalate",)
    assert "owlbank.wait_ms" not in turn.attributes  # no answer text: the hand-off only
    (esc,) = [s for s in turn_tree(spans, turn) if op(s) == "execute_tool"]
    assert esc.attributes["owlbank.tool_kind"] == "handoff"
    assert "wants a person" not in json.dumps([dict(s.attributes) for s in spans])


# --------------------------------------------------------------------------- relay signals


def test_interrupt_sets_attributes_on_session_and_turn_never_the_text() -> None:
    heard = "Hello th"
    rig = raw_rig([text_msg("Hello there, this is a long answer.", chunks=12)], delay=0.02)
    write_call_state(rig.store)
    with traced() as t:
        t.enable(rig.bridge)

        async def go() -> None:
            sim = rig.sim()
            await sim.connect()
            await sim.prompt_nowait("hi")
            await asyncio.sleep(0.08)  # mid-turn
            await sim.interrupt(heard=heard, ms=420)
            await asyncio.sleep(0.05)
            await sim.idle()
            await sim.hangup()
            rig.bridge.session_end(CALL_SID, "disconnect")
            await rig.bridge.wait_idle()

        run(go())
        spans = t.spans()
    assert_session_everywhere(spans)
    (session,) = by_name(spans, "relay.session")
    (event,) = events_of(session, "relay.interrupt")
    assert dict(event.attributes) == {"owlbank.interrupt_ms": 420, "owlbank.chars_played": len(heard)}
    (turn,) = by_name(spans, "bridge.turn")
    assert (turn.attributes["owlbank.interrupt_ms"], turn.attributes["owlbank.chars_played"]) == (420, len(heard))
    everything = json.dumps([[dict(s.attributes), [dict(e.attributes) for e in s.events]] for s in spans])
    assert heard not in everything


def test_info_and_error_frames_are_session_events_with_type_and_code_only() -> None:
    rig = raw_rig([text_msg("Hello.")])
    write_call_state(rig.store)
    with traced() as t:
        t.enable(rig.bridge)

        async def go() -> None:
            sim = rig.sim()
            await sim.connect()
            await sim.socket.inbound.put({"type": "info", "name": "roundTripDelayMs", "value": 12})
            await sim.socket.inbound.put({"type": "error", "description": "Invalid message: caller said 4111"})
            await sim.socket.inbound.put({"type": "error", "code": 64107, "description": "secret detail"})
            await sim.prompt("hi")
            await sim.hangup()
            rig.bridge.session_end(CALL_SID, "disconnect")
            await rig.bridge.wait_idle()

        run(go())
        spans = t.spans()
    (session,) = by_name(spans, "relay.session")
    info = [dict(e.attributes) for e in events_of(session, "relay.info")]
    errors = [dict(e.attributes) for e in events_of(session, "relay.error")]
    assert info == [{"owlbank.type": "info", "owlbank.code": "roundTripDelayMs"}]
    assert errors == [{"owlbank.type": "error"}, {"owlbank.type": "error", "owlbank.code": 64107}]
    assert "4111" not in json.dumps([dict(e.attributes) for e in session.events])


def test_close_code_comes_from_the_disconnect_or_its_cause() -> None:
    class Disc(Exception):
        def __init__(self, code: int) -> None:
            self.code = code

    assert _close_code(Disc(1001)) == 1001
    wrapped = RuntimeError("wrapped")
    wrapped.__cause__ = Disc(1006)
    assert _close_code(wrapped) == 1006
    assert _close_code(RuntimeError("no code")) is None


# --------------------------------------------------------------------------- log lines


@pytest.mark.parametrize("on", [False, True])
def test_a_failed_model_call_gets_a_strands_model_call_line(on: bool, caplog: pytest.LogCaptureFixture) -> None:
    rig = raw_rig([ScriptError("model down"), ScriptError("model down again")])
    write_call_state(rig.store)

    def go(t: Traced | None) -> None:
        if t is not None:
            t.enable(rig.bridge)
        with caplog.at_level(logging.INFO):
            run(one_session(rig, "hi"))

    if on:
        with traced() as t:
            go(t)
    else:
        go(None)
    calls = [ln for ln in lines(caplog.records, CLIENT_LOG) if ln["event"] == "strands_model_call"]
    assert calls and all(ln["outcome"] == "error" and ln["error_class"] == "ScriptError" for ln in calls)
    for ln in calls:
        assert ln["model_latency_s"] >= 0 and ln["model_ttfb_s"] is None
        assert "model down" not in json.dumps(ln)


def test_a_successful_model_call_line_keeps_the_bridge_and_provider_timings(caplog: pytest.LogCaptureFixture) -> None:
    rig = raw_rig([text_msg("Hello.")], delay=0.01)  # a measurable first byte
    write_call_state(rig.store)
    with caplog.at_level(logging.INFO):
        run(one_session(rig, "hi"))
    (ln,) = [ln for ln in lines(caplog.records, CLIENT_LOG) if ln["event"] == "strands_model_call"]
    assert ln["outcome"] == "ok" and "error_class" not in ln
    assert ln["call"] == 1 and ln["stop_reason"] == "end_turn"
    assert ln["model_ttfb_s"] is not None and ln["model_latency_s"] >= ln["model_ttfb_s"]
    assert ln["bedrock_latency_ms"] == 1  # the fake's metadata latencyMs
    assert ln["strands_ttfb_ms"] >= 10  # Strands' own measurement of the stream
    assert (ln["input_tokens"], ln["output_tokens"]) == (10, 5)


def test_provider_timings_come_from_the_stream_metadata_when_the_message_has_none(
        caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """``bedrock_latency_ms`` and ``strands_ttfb_ms`` are read from the stream's ``metadata``
    chunk too, so the line keeps them if Strands stops attaching metrics to the message."""
    from strands.hooks import AfterModelCallEvent

    real = AfterModelCallEvent.ModelStopResponse

    def bare(*, stop_reason: Any, message: Any) -> Any:  # the message without its "metadata"
        return real(stop_reason=stop_reason, message={k: v for k, v in message.items() if k != "metadata"})

    monkeypatch.setattr(AfterModelCallEvent, "ModelStopResponse", bare)
    chunks = text_msg("Hello.")
    chunks[-1] = {"metadata": {**chunks[-1]["metadata"],
                               "metrics": {"latencyMs": 42, "timeToFirstByteMs": 7}}}
    rig = raw_rig([chunks])
    write_call_state(rig.store)
    with caplog.at_level(logging.INFO):
        run(one_session(rig, "hi"))
    (ln,) = [ln for ln in lines(caplog.records, CLIENT_LOG) if ln["event"] == "strands_model_call"]
    assert (ln["bedrock_latency_ms"], ln["strands_ttfb_ms"]) == (42, 7)


def test_a_failing_telemetry_hook_is_logged_rate_limited_and_the_turn_goes_on(
        caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    from tac_bridge import agent_client_strands as acs

    words = "caller " + "words"

    def broken(*_a: Any, **_k: Any) -> None:
        raise RuntimeError(words)  # the message is never logged (the stack shows source lines only)

    monkeypatch.setattr(acs, "_log_model_call", broken)
    monkeypatch.setattr(acs, "_hook_failure_logged_at", None)
    rig = raw_rig([text_msg("Hello."), text_msg("Again.")])
    write_call_state(rig.store)
    with caplog.at_level(logging.INFO):
        frames = run(one_session(rig, "hi", "and again"))
    assert "Hello." in spoken(frames) and "Again." in spoken(frames)
    failed = [r for r in caplog.records if r.name == CLIENT_LOG and r.getMessage() == "telemetry_hook_failed"]
    assert len(failed) == 1 and failed[0].levelno == logging.WARNING  # once a minute, not per call
    (ln,) = lines(failed, CLIENT_LOG)
    assert ln["error_class"] == "RuntimeError" and ln["operation"] == "model_call_after"
    assert words not in json.dumps(ln, default=str)


def test_turn_line_has_silence_and_wait_with_telemetry_off(caplog: pytest.LogCaptureFixture) -> None:
    backend = FakeBackend({"list_recent_transactions": [TXNS]})
    rig = raw_rig([text_msg("Hello."),
                   tool_msg(("tu_tx", "list_recent_transactions", {"account": "checking"})),
                   text_msg("Your latest was four fifty."),
                   tool_msg(("tu_esc", "escalate", {"reason": "caller_requested_human", "summary": "x"}))],
                  backend=backend)
    write_call_state(rig.store)
    assert rig.bridge.tracing is None
    with caplog.at_level(logging.INFO):
        run(one_session(rig, "hi", "what did I spend?", "a person", end="closed"))
    turns = [ln for ln in lines(caplog.records, BRIDGE_LOG) if ln["event"] == "turn"]
    assert len(turns) == 3
    greet, tool_turn, handoff = turns
    assert greet["silence_ms"] is not None and greet["wait_ms"] == greet["silence_ms"]
    assert tool_turn["silence_ms"] <= tool_turn["wait_ms"]  # the masking phrase came first
    assert handoff["wait_ms"] is None  # no answer text
    (end,) = [ln for ln in lines(caplog.records, BRIDGE_LOG) if ln["event"] == "ws_session_end"]
    assert end["input_tokens"] == sum(t.get("input_tokens", 0) for t in turns)
    assert end["output_tokens"] == sum(t.get("output_tokens", 0) for t in turns)
    assert end["input_tokens"] > 0


# --------------------------------------------------------------------------- memory.reload


def _memory_rig(monkeypatch: pytest.MonkeyPatch, t: Traced | None) -> tuple[Any, Any, Any]:
    """The rig with the Strands client on the fake Memory; the call's owner is another process, so
    the session start reloads. With ``t``, each ``list_events`` on the fake data plane runs
    in a span, as botocore's instrumentation would make one."""
    from tests.strands_fakes import FakeMemoryData, light_memory_client, memory_factory

    light_memory_client(monkeypatch)
    rig = make_rig()
    data = FakeMemoryData()
    model = FakeModel([text_msg("Hello."), text_msg("Again.")])
    client = StrandsAgentClient(settings(memory_id="mem-1"), model_factory=lambda _s, _sid: model,
                                gateway_tools=default_gateway(), session_manager_factory=memory_factory(data))
    rig.bridge._agent = client
    write_call_state(rig.store, agent_owner="another-process")
    if t is not None:
        real = data.list_events

        def list_events(**kw: Any) -> Any:
            with t.tracer.start_as_current_span("BedrockAgentCore.ListEvents"):
                return real(**kw)

        data.list_events = list_events  # type: ignore[method-assign]
        t.enable(rig.bridge)
    return rig, client, data


async def _reload_at_start_and_in_turn(rig: Any, client: Any) -> None:
    from tests.component.test_bridge_fakes import wait_for

    sim = rig.sim()
    await sim.connect()  # owner changed: the session start reloads from Memory
    await wait_for(lambda: all(task.done() for task in rig.bridge._session_starts.values()))
    await sim.prompt("hi")
    client._calls.clear()  # the cached agent is evicted: the next turn reloads (cache_miss)
    await sim.prompt("and again")
    await sim.hangup()
    rig.bridge.session_end(CALL_SID, "disconnect", close_code=1000)
    await rig.bridge.wait_idle()


def test_memory_reloads_have_spans_with_the_call_and_their_aws_calls_under_them(
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    with traced() as t:
        rig, client, _data = _memory_rig(monkeypatch, t)
        with caplog.at_level(logging.INFO):
            run(_reload_at_start_and_in_turn(rig, client))
        spans = t.spans()
    assert_session_everywhere(spans)
    reloads = sorted(by_name(spans, "memory.reload"), key=lambda s: s.start_time)
    assert [(s.attributes["owlbank.history"], s.attributes["owlbank.reason"]) for s in reloads] == [
        ("reload", "owner_changed"), ("cached", "cache_miss")]
    assert all(s.attributes["owlbank.outcome"] == "ok" for s in reloads)
    logged = [ln["reason"] for ln in lines(caplog.records, CLIENT_LOG) if ln["event"] == "agent_history_reload"]
    assert logged == ["owner_changed", "cache_miss"]  # one span per reload
    (session,) = by_name(spans, "relay.session")
    at_start, in_turn = reloads
    assert at_start.parent is not None and at_start.parent.span_id == session.context.span_id
    turns = sorted(by_name(spans, "bridge.turn"), key=lambda s: s.start_time)
    assert in_turn.context.span_id in {d.context.span_id for d in descendants(spans, turns[-1])}
    for reload in reloads:
        aws = [d for d in descendants(spans, reload) if d.name == "BedrockAgentCore.ListEvents"]
        assert aws, "the reload's Memory reads nest under its span"


def test_memory_reloads_without_tracing_are_unchanged(monkeypatch: pytest.MonkeyPatch,
                                                      caplog: pytest.LogCaptureFixture) -> None:
    rig, client, _data = _memory_rig(monkeypatch, None)
    assert rig.bridge.tracing is None
    with caplog.at_level(logging.INFO):
        run(_reload_at_start_and_in_turn(rig, client))
    assert client._tracing is None
    logged = [ln["reason"] for ln in lines(caplog.records, CLIENT_LOG) if ln["event"] == "agent_history_reload"]
    assert logged == ["owner_changed", "cache_miss"]
    turns = [ln for ln in lines(caplog.records, BRIDGE_LOG) if ln["event"] == "turn"]
    assert [ln["outcome"] for ln in turns] == ["ok", "ok"]


def _failing_reload_run(monkeypatch: pytest.MonkeyPatch, t: Traced | None,
                        caplog: pytest.LogCaptureFixture) -> list[tuple[Any, ...]]:
    """Both reloads with Memory's ``list_events`` failing; the bridge's ``turn`` lines."""
    rig, client, data = _memory_rig(monkeypatch, t)
    data.fail["list_events"] = RuntimeError("memory down")
    caplog.clear()
    with caplog.at_level(logging.INFO):
        run(_reload_at_start_and_in_turn(rig, client))
    return [(ln["outcome"], ln.get("stop_reason"), ln.get("hops"), tuple(ln.get("tools") or ()))
            for ln in lines(caplog.records, BRIDGE_LOG) if ln["event"] == "turn"]


def test_a_failed_memory_reload_ends_its_span_as_an_error_and_the_turn_as_untraced(
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    untraced = _failing_reload_run(monkeypatch, None, caplog)
    with traced() as t:
        turns = _failing_reload_run(monkeypatch, t, caplog)
        spans = t.spans()
    assert turns and turns == untraced  # the failed reload's outcome, traced or not
    reloads = by_name(spans, "memory.reload")
    assert reloads, "each reload attempt has a span"
    for reload in reloads:
        assert reload.attributes["owlbank.outcome"] == "error"
        assert reload.attributes["owlbank.error_class"]
