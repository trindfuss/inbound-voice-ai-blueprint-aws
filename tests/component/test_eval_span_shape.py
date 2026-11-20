"""Call observability: the span shape AgentCore Evaluations reads (runs in CI, no AWS access).

One scripted call runs the bridge with the real
Strands client on a real ``BedrockModel`` whose ``bedrock-runtime`` client is the stubbed
ConverseStream (:class:`tests.strands_fakes.FakeBedrockRuntime`), exported to an in-memory exporter
through the production processors and content filter. The call has a greeting, an inline tool, a
Gateway tool, an inline hand-off (``authenticate``) that ends its turn on an interrupt, and a
resumed turn after the step-up that starts from the front door's result (no caller words).

The test fails when a Strands upgrade, or a change to the content filter or the evaluation-shape
processor, drops a key or event Evaluations needs:

* every span carries ``session.id``;
* each Strands span has ``gen_ai.operation.name`` in the four operations, each tool span
  ``gen_ai.tool.name``;
* with ``full``, the ``gen_ai.user.message``, ``gen_ai.choice`` and ``gen_ai.tool.message`` events;
* every ``invoke_agent`` has a ``gen_ai.user.message`` event and every ``execute_tool`` a
  ``gen_ai.choice`` event and a ``gen_ai.tool.status``, the resumed turn and the interrupted inline
  tool included; with ``off`` the names and statuses stay and the text goes.

No property-based test here (a property over generated hop sequences is in
``tests/property/test_bridge_telemetry_props.py``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from owlbank.envelope import ResultEnvelope, Spoken
from tac_bridge.agent_client_strands import StrandsAgentClient, default_model_factory
from tests.component.test_bridge_fakes import (
    CALL_SID,
    FakeBackend,
    front_door_update,
    make_rig,
    rearm_nonce,
    run,
    write_call_state,
)
from tests.strands_fakes import FakeBedrockRuntime, default_gateway, settings, text_msg, tool_msg, traced

OPERATIONS = {"invoke_agent", "execute_event_loop_cycle", "chat", "execute_tool"}
STRANDS_SCOPE = "strands.telemetry.tracer"
TXNS = {"account": "checking", "transactions": [{"date": "2026-10-09", "description": "Coffee", "amount": "-4.50"}]}
VERIFIED = ResultEnvelope(status="verified", tool="authenticate",
                          spoken=[Spoken("stepup.success", "Thanks, you're verified.")],
                          data={"forTool": "make_payment"}, next="retry_original_tool",
                          forTool="make_payment").to_json()


def scripted_call(content: str) -> list[Any]:
    """The call above; returns the exported spans."""
    runtime = FakeBedrockRuntime([
        text_msg("Hello, how can I help?"),
        tool_msg(("tu_tx", "list_recent_transactions", {"account": "checking"})),
        text_msg("Your latest was four fifty."),
        tool_msg(("tu_faq", "level0___faq_lookup", {"q": "fees"})),
        text_msg("There is no fee."),
        tool_msg(("tu_auth", "authenticate", {"tool": "make_payment"})),
        tool_msg(("tu_pay", "make_payment", {"from_account": "checking", "to": "red", "amount": "25.00"})),
    ])

    def factory(s: Any, _sid: str) -> Any:
        model = default_model_factory(s)
        model.client = runtime
        return model

    rig = make_rig(backend=FakeBackend({"list_recent_transactions": [TXNS]}))
    rig.bridge._agent = StrandsAgentClient(settings(), model_factory=factory, gateway_tools=default_gateway())
    write_call_state(rig.store)

    async def session(*prompts: str, resume: bool = False) -> None:
        sim = rig.sim()
        await sim.connect()
        if resume:
            await sim.wait_reply()
        for p in prompts:
            await sim.prompt(p)
        await sim.hangup()
        rig.bridge.session_end(CALL_SID, "closed")
        await rig.bridge.wait_idle()

    with traced(content) as t:
        t.enable(rig.bridge)
        run(session("hi", "what did I spend?", "any fees?", "pay 25 on my card"))
        front_door_update(rig.store, pending_result=json.loads(VERIFIED), level=3)
        rearm_nonce(rig.store)
        run(session(resume=True))
        spans = t.spans()
    assert not runtime.script, "every scripted model call ran"
    return spans


def strands(spans: list[Any]) -> list[Any]:
    return [s for s in spans if s.instrumentation_scope is not None and s.instrumentation_scope.name == STRANDS_SCOPE]


def operation(span: Any) -> str | None:
    return span.attributes.get("gen_ai.operation.name")


def names(span: Any) -> set[str]:
    return {e.name for e in span.events}


@pytest.fixture(scope="module", params=["off", "full"])
def call(request: pytest.FixtureRequest) -> tuple[str, list[Any]]:
    return request.param, scripted_call(request.param)


def test_every_span_carries_the_session(call: tuple[str, list[Any]]) -> None:
    _, spans = call
    assert spans
    assert {s.attributes.get("session.id") for s in spans} == {CALL_SID}


def test_strands_spans_carry_the_genai_operation_and_tool_names(call: tuple[str, list[Any]]) -> None:
    _, spans = call
    ours = strands(spans)
    assert {operation(s) for s in ours} == OPERATIONS
    for s in ours:
        assert operation(s) in OPERATIONS, s.name
        if operation(s) == "execute_tool":
            assert s.attributes.get("gen_ai.tool.name"), s.name
    tools = {s.attributes["gen_ai.tool.name"] for s in ours if operation(s) == "execute_tool"}
    assert tools >= {"list_recent_transactions", "level0___faq_lookup", "authenticate"}


def test_every_invoke_agent_has_a_user_message_and_every_tool_a_choice_and_status(
        call: tuple[str, list[Any]]) -> None:
    _, spans = call
    invokes = [s for s in spans if operation(s) == "invoke_agent"]
    tools = [s for s in spans if operation(s) == "execute_tool"]
    assert len(invokes) >= 6 and len(tools) >= 4
    for s in invokes:
        assert "gen_ai.user.message" in names(s), s.name
    for s in tools:
        assert "gen_ai.choice" in names(s), s.attributes.get("gen_ai.tool.name")
        assert s.attributes.get("gen_ai.tool.status") in ("success", "error"), s.attributes.get("gen_ai.tool.name")


def test_the_resumed_turn_and_the_interrupted_inline_tool_are_filled(call: tuple[str, list[Any]]) -> None:
    content, spans = call
    turns = sorted((s for s in spans if s.name == "bridge.turn"), key=lambda s: s.start_time)
    resumed = turns[-1]
    assert resumed.attributes["owlbank.resumed"] is True
    (invoke,) = [s for s in spans if operation(s) == "invoke_agent" and s.context.trace_id == resumed.context.trace_id]
    (user,) = [e for e in invoke.events if e.name == "gen_ai.user.message"]
    interrupted = [s for s in spans if operation(s) == "execute_tool" and s.attributes.get("owlbank.result_next_turn")]
    assert {s.attributes["gen_ai.tool.name"] for s in interrupted} >= {"authenticate", "list_recent_transactions"}
    auth = next(s for s in interrupted if s.attributes["gen_ai.tool.name"] == "authenticate")
    (choice,) = [e for e in auth.events if e.name == "gen_ai.choice"]
    if content == "full":
        assert "[call continues after authenticate]" in user.attributes["content"]
        assert "[result from the front door: authenticate]" in choice.attributes["message"]
        assert choice.attributes.get("id") == "tu_auth"
    else:  # 8.0 with off: the event names and statuses stay, the text goes
        assert "content" not in user.attributes
        assert "message" not in choice.attributes


def test_full_content_has_the_message_events_evaluations_parse(call: tuple[str, list[Any]]) -> None:
    content, spans = call
    events = {e.name for s in spans for e in s.events}
    assert {"gen_ai.user.message", "gen_ai.choice", "gen_ai.tool.message"} <= events
    tool_messages = [e for s in spans for e in s.events if e.name == "gen_ai.tool.message"]
    if content == "full":
        assert all("content" in e.attributes for e in tool_messages)
        assert any("Your latest was four fifty." in json.dumps(dict(e.attributes)) for s in spans for e in s.events)
    else:
        assert not any("Your latest was four fifty." in json.dumps(dict(e.attributes))
                       for s in spans for e in s.events)
