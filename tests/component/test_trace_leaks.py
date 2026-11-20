"""Call observability: marker values never come out of the exported spans.

A scripted call runs the bridge with the real Strands
client on a real ``BedrockModel`` whose ``bedrock-runtime`` client is a real botocore client with a
``Stubber`` (``converse``, so botocore's instrumentation runs and makes its Bedrock CLIENT spans),
with telemetry installed as in production (baggage, evaluation-shape and batch processors, the
content filter) and **botocore instrumentation on**. Marker values sit in every place text can
reach a span: the system prompt (the verified customer's name and the Memory observation), the
caller's words, the model's answer, tool arguments, tool results (bank and Gateway), an error's
message and the step placeholder's neighbours.

* ``off``: no marker appears anywhere in the export (span names, attributes, events, links,
  status), and no ``aws.auth.*`` key;
* ``redacted``: the markers the transcripts' redaction covers (a card number, a registered
  sensitive value) are absent; the others may appear (that is the setting's point).

Markers are matched as whole values (``tests/leakscan.py``). No property-based test here (the
filter over arbitrary spans, is ``tests/property/test_telemetry_filter_props.py``).
"""

from __future__ import annotations

import json
from typing import Any

import boto3
import pytest
from botocore.stub import Stubber

from owlbank import segments
from owlbank.pii import FakePiiRedactor, PiiGuard
from owlbank.policy import Policy
from owlbank.telemetry.bridge import TranscriptRedactor
from tac_bridge.agent_client_strands import StrandsAgentClient
from tests.component.test_bridge_fakes import (
    CALL_SID,
    FakeBackend,
    FakeMemory,
    load_json,
    make_rig,
    run,
    spoken,
    write_call_state,
)
from tests.leakscan import compile_values, leaks
from tests.strands_fakes import default_gateway, settings, traced
from strands.models import BedrockModel

CARD = "4111 1111 1111 1111"  # Luhn-valid: the fake Comprehend covers it
ZIP = "94107"  # registered as the call's sensitive value: the exact-value scrub covers it
USER_MARKER = "zebraquartz"
ANSWER_MARKER = "pelicantopaz"
ARG_MARKER = "walrusgarnet"
GATEWAY_MARKER = "otteramethyst"
TOOL_RESULT_MARKER = "Marmotjasper Bakery"
ERROR_MARKER = "lynxobsidian"
MEMORY_MARKER = "heronbasalt"
NAME_MARKER = "Doe"  # C-1001's surname, in the system prompt (the prompt holds the name)
UNCOVERED = [USER_MARKER, ANSWER_MARKER, ARG_MARKER, GATEWAY_MARKER, TOOL_RESULT_MARKER, ERROR_MARKER,
             MEMORY_MARKER, NAME_MARKER]
COVERED = [CARD, ZIP]


def _converse(*content: dict[str, Any], stop: str = "end_turn") -> dict[str, Any]:
    return {"output": {"message": {"role": "assistant", "content": list(content)}}, "stopReason": stop,
            "usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18}, "metrics": {"latencyMs": 3}}


def _tool_use(tid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"toolUse": {"toolUseId": tid, "name": name, "input": args}}


def _bedrock_model() -> tuple[BedrockModel, Stubber]:
    model = BedrockModel(model_id="us.anthropic.claude-haiku-5-5", region_name="us-east-1", streaming=False,
                         max_tokens=1000)
    model.client = boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id="AKIDTESTONLY",
                                aws_secret_access_key="test-secret-not-real")  # noqa: S106 - offline stub
    stub = Stubber(model.client)
    stub.add_response("converse", _converse({"text": f"Hello, the code {USER_MARKER} is noted, card {CARD}."}))
    stub.add_client_error("converse", service_error_code="ThrottlingException",
                          service_message=f"throttled near {ERROR_MARKER}", http_status_code=400)
    stub.add_response("converse", _converse(_tool_use("tu_tx", "list_recent_transactions",
                                                      {"account": "checking"}), stop="tool_use"))
    stub.add_response("converse", _converse({"text": f"Your latest was at {TOOL_RESULT_MARKER}, "
                                                     f"{ANSWER_MARKER}, zip {ZIP}."}))
    stub.add_response("converse", _converse(_tool_use("tu_gw", "level0___faq_lookup", {"q": ARG_MARKER}),
                                            stop="tool_use"))
    stub.add_response("converse", _converse({"text": "There is no fee."}))
    stub.activate()
    return model, stub


def scripted_call(content: str) -> list[Any]:
    model, stub = _bedrock_model()
    backend = FakeBackend({"list_recent_transactions": [{"account": "checking", "transactions": [
        {"date": "2026-10-09", "description": TOOL_RESULT_MARKER, "amount": "-4.50",
         "memo": f"card {CARD} zip {ZIP}"}]}]})
    memory = FakeMemory()
    memory.prompts = dict.fromkeys(memory.prompts, f"## Key Observations\n- Mentioned {MEMORY_MARKER} last week")
    rig = make_rig(backend=backend, memory=memory)
    rig.bridge._agent = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: model,
                                           gateway_tools=default_gateway(answer=f"Fees: {GATEWAY_MARKER}"))
    write_call_state(rig.store)
    segments.register_sensitive(CALL_SID, [ZIP])
    policy = Policy.from_dict(load_json("policy.json"))
    redact = TranscriptRedactor(PiiGuard.for_policy(FakePiiRedactor(), policy)) if content == "redacted" else None

    async def session() -> list[Any]:
        sim = rig.sim()
        await sim.connect()
        frames = []
        for words in (f"hi, my code word is {USER_MARKER} and my card is {CARD}",
                      f"what did I spend? zip {ZIP}", "any fees?"):
            frames += await sim.prompt(words)
        await sim.hangup()
        rig.bridge.session_end(CALL_SID, "closed")
        await rig.bridge.wait_idle()
        return frames

    try:
        with traced(content, redact=redact, botocore=True) as t:
            t.enable(rig.bridge)
            frames = run(session())
            # Still holds: the session end forgot the values, after its spans were exported.
            assert not segments.sensitive_values(CALL_SID)
            spans = t.spans()
    finally:
        segments.clear_sensitive(CALL_SID)
    stub.assert_no_pending_responses()
    assert ANSWER_MARKER in spoken(frames)  # the markers did flow through the call
    return spans


def dump(spans: list[Any]) -> str:
    """Everything an exported span carries that could hold text."""
    out = []
    for s in spans:
        out.append({"name": s.name, "attributes": dict(s.attributes or {}),
                    "events": [{"name": e.name, "attributes": dict(e.attributes or {})} for e in s.events],
                    "links": [dict(link.attributes or {}) for link in s.links],
                    "status": s.status.description})
    return json.dumps(out, default=str)


@pytest.fixture(scope="module", params=["off", "redacted"])
def exported(request: pytest.FixtureRequest) -> tuple[str, list[Any]]:
    return request.param, scripted_call(request.param)


def test_the_call_made_botocore_and_strands_spans(exported: tuple[str, list[Any]]) -> None:
    _, spans = exported
    scopes = {s.instrumentation_scope.name for s in spans if s.instrumentation_scope is not None}
    assert "strands.telemetry.tracer" in scopes
    clients = [s for s in spans if s.attributes.get("rpc.service") == "Bedrock Runtime"
               or (s.attributes.get("rpc.system") == "aws-api" and "Converse" in s.name)]
    assert len(clients) >= 5  # one per converse, the throttled one included
    assert {s.attributes.get("session.id") for s in spans} == {CALL_SID}


def test_no_aws_auth_key_in_any_mode(exported: tuple[str, list[Any]]) -> None:
    _, spans = exported
    for s in spans:
        assert not [k for k in s.attributes if k.startswith("aws.auth.")]
    assert "AKIDTESTONLY" not in dump(spans)


def test_markers_never_leave_with_off_and_covered_ones_never_with_redacted(
        exported: tuple[str, list[Any]]) -> None:
    content, spans = exported
    text = dump(spans)
    covered = compile_values(COVERED)
    assert not leaks(covered, text), content
    if content == "off":
        for marker in UNCOVERED:
            assert not leaks(compile_values([marker]), text), marker
