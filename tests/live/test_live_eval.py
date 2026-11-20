"""Call observability: the live evaluation test (on demand, dev only; ``-m live_eval``).

Skipped by the default ``pytest`` run (``addopts``
in ``pyproject.toml``); run it before a release and after any Strands or ADOT upgrade::

    AWS_PROFILE=<dev profile> uv run pytest -m live_eval -rA tests/live/test_live_eval.py

A bench call: the bridge from the component rig (``tests/component/test_bridge_fakes.py``: the
simulated ConversationRelay socket, in-memory call state, a fake ``bank_tools``), with the real
Strands client on real Bedrock, the production telemetry (``install_bridge``: content filter,
evaluation-shape processor) and the real SigV4 exporter into the deployed traces log group
``/owlbank/traces/bridge``. The caller is synthetic and every account value is a fixture. No phone
call and no deploy: the content mode is this process's, so ``full`` and ``redacted`` need no
change to the deployed bridge.

The call: recent transactions (an inline tool), then a payment at level 1, which the bridge steps
up (its step-up shortcut hands off on ``make_payment`` itself, ending the turn on an interrupt); a resumed turn after the step-up
(the payment's confirmation hand-off); a resumed turn after the payment, with the caller's words.

Then, after the spans arrive in the log group (bounded wait), on-demand ``Evaluate`` (it creates
no configuration, so nothing is left behind) with ``Builtin.GoalSuccessRate`` (session level) and
``Builtin.Helpfulness`` (trace level, one trace per agent turn). With ``full`` the test fails on a
mapping error, a missing score, or a trace-level result count other than one per agent turn (up to
the API's 10). With ``redacted`` the same results are reported (a warning), never failed on.

Before the call the test checks that Bedrock model invocation logging doesn't deliver text in the
Region (preflight's ``check_invocation_logging``): ``full`` puts the synthetic conversation in
spans, and logging would also store the prompts elsewhere.

No property-based test (a live service).
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from owlbank.envelope import ResultEnvelope, Spoken
from scripts.call_timeline import run_query
from tac_bridge.agent_client_strands import StrandsAgentClient
from tests.component.test_bridge_fakes import (
    FakeBackend,
    end_frame,
    front_door_update,
    load_json,
    make_rig,
    rearm_nonce,
    run,
    spoken,
    write_call_state,
)
from tests.strands_fakes import default_gateway, settings, traced

pytestmark = pytest.mark.live_eval

PREFLIGHT = Path(__file__).resolve().parents[2] / "infra" / "scripts" / "preflight.py"
REGION = os.environ.get("AWS_REGION", "us-east-1")
#: ``infra/owlbank_infra/call_observability.py``: ``TRACES_LOG_GROUP_NAME``.
TRACES_LOG_GROUP = "/owlbank/traces/bridge"
SESSION_EVALUATOR = "Builtin.GoalSuccessRate"
TRACE_EVALUATOR = "Builtin.Helpfulness"
#: ``Evaluate`` returns at most this many trace-level results (the last ones).
MAX_RESULTS = 10
#: Spans reach the log group within seconds to a minute or two; then the test fails.
ARRIVAL_TIMEOUT_S = 300.0
ARRIVAL_POLL_S = 15.0
#: Model turns take a few seconds each; the simulator's default (3 s) is for the fakes.
TURN_TIMEOUT_S = 90.0

TXNS = {"account": "checking", "transactions": [
    {"date": "2026-10-09", "description": "Grocery Sample Mart", "amount": "-71.00", "status": "pending"},
    {"date": "2026-10-08", "description": "Payroll Example Corp", "amount": "2100.00", "status": "posted"}]}
VERIFIED = ResultEnvelope(status="verified", tool="authenticate",
                          spoken=[Spoken("stepup.success", "Thanks, you're verified.")],
                          data={"forTool": "make_payment"}, next="retry_original_tool",
                          forTool="make_payment").to_json()
PAID = ResultEnvelope(status="completed", tool="make_payment",
                      spoken=[Spoken("confirm.make_payment.done",
                                     "Done. Your payment of twenty-five dollars from your checking account "
                                     "is complete.")],
                      data={"amount": "25.00", "confirmation": "85027343", "executed": True},
                      next="handle_utterance", callerUtterance="Thanks, that's all for today.").to_json()


# --------------------------------------------------------------------------- the bench call


class TeeExporter:
    """Exports to the real exporter and keeps a copy, so the test knows which spans to wait for."""

    def __init__(self, real: Any) -> None:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        self.real = real
        self.copy = InMemorySpanExporter()

    def export(self, spans: Sequence[Any]) -> Any:
        self.copy.export(spans)
        return self.real.export(spans)

    def get_finished_spans(self) -> list[Any]:
        return list(self.copy.get_finished_spans())

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return bool(self.real.force_flush(timeout_millis))

    def shutdown(self) -> None:
        self.real.shutdown()
        self.copy.shutdown()


@dataclass
class Call:
    content: str
    call_sid: str
    started: datetime
    span_ids: set[str]
    #: Trace IDs (hex) that hold an ``invoke_agent`` span: one per agent turn.
    agent_traces: set[str]
    tools: set[str]
    ended: datetime = field(default_factory=lambda: datetime.now(UTC))


def redactor_for(content: str) -> Any:
    """``redacted``: the bridge's ``TranscriptRedactor`` with the real Comprehend guard, as
    ``tac_bridge.app`` attaches it; ``None`` otherwise."""
    if content != "redacted":
        return None
    from owlbank.pii import ComprehendPiiRedactor, PiiGuard, make_comprehend_client
    from owlbank.policy import Policy
    from owlbank.telemetry.bridge import TranscriptRedactor

    redactor = TranscriptRedactor()
    redactor.attach(PiiGuard.for_policy(ComprehendPiiRedactor(make_comprehend_client(REGION)),
                                        Policy.from_dict(load_json("policy.json"))))
    return redactor


def bench_call(content: str) -> Call:
    """The call above, traced with ``content`` into the traces log group."""
    from owlbank.telemetry.exporters import SigV4OtlpSpanExporter

    call_sid = "CA" + uuid.uuid4().hex
    started = datetime.now(UTC)
    rig = make_rig(backend=FakeBackend({"list_recent_transactions": [TXNS]}))
    rig.bridge._agent = StrandsAgentClient(settings(region=REGION), gateway_tools=default_gateway())
    write_call_state(rig.store, call_sid)

    async def session(*prompts: str, resume: bool = False, handoff: bool = False) -> None:
        sim = rig.sim()
        await sim.connect(call_sid)
        if resume:
            await sim.wait_reply(TURN_TIMEOUT_S)
        for p in prompts:
            await sim.prompt(p, timeout=TURN_TIMEOUT_S)
        if handoff and end_frame(sim.socket.sent) is None:
            await sim.hangup()
            pytest.fail(f"no hand-off after {prompts[-1]!r}; the agent said {spoken(sim.socket.sent)[-200:]!r} "
                        "(rerun; the model chooses its tools)")
        await sim.hangup()
        rig.bridge.session_end(call_sid, "closed")
        await rig.bridge.wait_idle()

    tee = TeeExporter(SigV4OtlpSpanExporter(REGION, TRACES_LOG_GROUP))
    with traced(content, redact=redactor_for(content), exporter=tee) as t:
        t.enable(rig.bridge)
        run(session("What are my recent transactions on checking?",
                    "Please pay 25 dollars from my checking account to my Red Card.", handoff=True))
        front_door_update(rig.store, call_sid, pending_result=json.loads(VERIFIED), level=3)
        rearm_nonce(rig.store, call_sid)
        run(session(resume=True))
        front_door_update(rig.store, call_sid, pending_result=json.loads(PAID))
        rearm_nonce(rig.store, call_sid)
        run(session(resume=True))
        spans = t.spans()
    assert {r["op"] for r in rig.backend.calls} >= {"list_recent_transactions"}, \
        "the model didn't call the inline tool (rerun; the model chooses its tools)"
    agent_traces = {format(s.context.trace_id, "032x") for s in spans
                    if s.attributes.get("gen_ai.operation.name") == "invoke_agent"}
    # The first session (two prompts) and the two resumed turns: fewer means a turn didn't run, or a
    # Strands or ADOT upgrade renamed the operation, and the count check below would pass at 0 == 0.
    assert len(agent_traces) >= 3, f"only {len(agent_traces)} agent turn(s) were traced"
    tools = {s.attributes.get("gen_ai.tool.name") for s in spans
             if s.attributes.get("gen_ai.operation.name") == "execute_tool"}
    return Call(content=content, call_sid=call_sid, started=started,
                span_ids={format(s.context.span_id, "016x") for s in spans},
                agent_traces=agent_traces,
                tools=tools)


# --------------------------------------------------------------------------- spans and Evaluate


def stored_spans(logs: Any, call: Call) -> list[dict[str, Any]]:
    """The call's span records in the traces log group (Logs Insights on ``attributes.session.id``)."""
    rows = run_query(logs, query=f'fields @message | filter attributes.session.id = "{call.call_sid}" '
                                 "| limit 10000",
                     log_groups=[TRACES_LOG_GROUP], start=call.started - timedelta(minutes=1),
                     end=datetime.now(UTC) + timedelta(minutes=1))
    out = []
    for row in rows:
        message = next((f["value"] for f in row if f.get("field") == "@message"), None)
        if message:
            out.append(json.loads(message))
    return out


def wait_for_spans(logs: Any, call: Call) -> list[dict[str, Any]]:
    """Polls until every exported span is stored; fails after :data:`ARRIVAL_TIMEOUT_S`."""
    deadline = time.monotonic() + ARRIVAL_TIMEOUT_S
    while True:
        records = stored_spans(logs, call)
        missing = call.span_ids - {r.get("spanId") for r in records}
        if not missing:
            return records
        if time.monotonic() >= deadline:
            pytest.fail(f"{len(missing)} of {len(call.span_ids)} spans of {call.call_sid} not in "
                        f"{TRACES_LOG_GROUP} after {ARRIVAL_TIMEOUT_S:.0f} s")
        time.sleep(ARRIVAL_POLL_S)


@dataclass
class Scores:
    evaluator: str
    results: list[dict[str, Any]]
    #: A request-level failure (for example ``AgentSpanMappingException``), else ``None``.
    error: str | None = None

    def problems(self) -> list[str]:
        if self.error:
            return [self.error]
        out = [f"{r.get('errorCode')}: {r.get('errorMessage')}" for r in self.results
               if r.get("errorCode") or r.get("errorMessage")]
        out += [f"no score in {r.get('context')}" for r in self.results
                if r.get("value") is None and not r.get("errorCode")]
        return out

    def summary(self) -> str:
        values = [f"{r['value']:.2f} {r.get('label', '')}".strip() for r in self.results if r.get("value") is not None]
        return f"{self.evaluator}: {len(self.results)} result(s) [{', '.join(values)}]" + \
            (f"; problems: {self.problems()}" if self.problems() else "")


def evaluate(client: Any, evaluator: str, records: list[dict[str, Any]]) -> Scores:
    from botocore.exceptions import ClientError

    try:
        resp = client.evaluate(evaluatorId=evaluator, evaluationInput={"sessionSpans": records})
    except ClientError as exc:
        err = exc.response.get("Error", {})
        return Scores(evaluator, [], f"{err.get('Code')}: {err.get('Message')}")
    return Scores(evaluator, list(resp.get("evaluationResults", [])))


def check_invocation_logging() -> None:
    import importlib.util

    import boto3

    # preflight.py is a standalone script (no package); load it by path, as its unit tests do.
    spec = importlib.util.spec_from_file_location("preflight_live_eval", PREFLIGHT)
    assert spec is not None and spec.loader is not None
    preflight = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = preflight  # dataclasses need the module registered
    spec.loader.exec_module(preflight)
    # A non-deterministic ceiling, so text delivery FAILs rather than WARNs.
    (result,) = preflight.check_invocation_logging(lambda r: boto3.client("bedrock", region_name=r),
                                                   "assisted", (REGION,))
    if not result.ok or result.warn:
        pytest.fail(f"{result.name}: {result.detail}. Fix: {result.fix}")


# --------------------------------------------------------------------------- tests


@pytest.fixture(scope="module")
def aws() -> dict[str, Any]:
    import boto3

    check_invocation_logging()
    return {"logs": boto3.client("logs", region_name=REGION),
            "agentcore": boto3.client("bedrock-agentcore", region_name=REGION)}


def evaluated(aws: dict[str, Any], content: str) -> tuple[Call, Scores, Scores]:
    call = bench_call(content)
    records = wait_for_spans(aws["logs"], call)
    print(f"\n{content}: {call.call_sid}, {len(records)} spans, {len(call.agent_traces)} agent turns, "
          f"tools {sorted(call.tools)}")
    session = evaluate(aws["agentcore"], SESSION_EVALUATOR, records)
    trace = evaluate(aws["agentcore"], TRACE_EVALUATOR, records)
    print(f"  {session.summary()}\n  {trace.summary()}")
    return call, session, trace


def test_full_content_scores_without_mapping_errors(aws: dict[str, Any]) -> None:
    call, session, trace = evaluated(aws, "full")
    assert not session.problems(), session.summary()
    assert session.results, session.summary()
    assert not trace.problems(), trace.summary()
    assert len(trace.results) == min(len(call.agent_traces), MAX_RESULTS), \
        f"{trace.summary()}; expected one per agent turn ({len(call.agent_traces)})"


def test_redacted_content_is_reported(aws: dict[str, Any]) -> None:
    """Repeated with ``redacted``; reports whether scores come back, never fails on them."""
    call, session, trace = evaluated(aws, "redacted")
    scored = not session.problems() and session.results and not trace.problems() \
        and len(trace.results) == min(len(call.agent_traces), MAX_RESULTS)
    warnings.warn(f"traceContent=redacted ({call.call_sid}): {'scores came back' if scored else 'NOT scored'}; "
                  f"{session.summary()}; {trace.summary()}", stacklevel=1)
