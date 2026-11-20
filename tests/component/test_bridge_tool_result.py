"""The bridge's ``tool_result`` line.

One line per inline or Gateway tool call that reaches an outcome, with ``tool``, ``tool_kind``,
``outcome``, ``error_code`` when present and the call SID, never arguments or result text; the
tool error-rate alarm counts these. Checked here, with telemetry off (the line is a plain log line):

* inline: a bank success, ``UNAVAILABLE`` (error), ``NOT_FOUND`` (refused), an unlisted code (error,
  logged as ``other``), a bridge rejection, a refused hand-off (``CALL_ALONE``: rejected), an unknown
  tool (rejected, name cut to 64 characters);
* a hang-up or barge-in during a bank call is never an error: the cancelled turn logs ``cancelled``;
* a hand-off (and its prefetch) logs nothing;
* Gateway: success, a raising tool (error), an answer carrying an error code inside a successful
  MCP result (error), and a barge-in mid-tool (nothing, never an error).

A property test over generated results is in ``tests/property/test_tool_outcome_props.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest
from strands import tool as strands_tool

from tac_bridge.agent_client import ToolCall, TurnEnd
from tac_bridge.agent_client_strands import StrandsAgentClient
from tests.component.masking_fixtures import masked_policy_doc, unmasked_policy_doc
from tests.component.test_bridge_fakes import (
    CALL_SID,
    FakeBackend,
    make_rig,
    run,
    say,
    tool,
    wait_for,
    write_call_state,
)
from tests.component.test_bridge_masking import balance_backend, collect, context, direct
from tests.logutil import lines, with_context
from tests.strands_fakes import (
    FakeGatewayProvider,
    FakeModel,
    GatewayRuns,
    default_gateway,
    settings,
    text_msg,
    tool_msg,
)

BRIDGE_LOG = "tac_bridge.bridge"
CLIENT_LOG = "tac_bridge.agent_client_strands"
TXNS = {"tool": "list_recent_transactions",
        "data": {"accountType": "checking", "last4": "4821",
                 "transactions": [{"date": "2026-10-09", "description": "Coffee Example Co", "amount": "-4.50"}]}}
FIELDS = {"event", "tool", "tool_kind", "outcome", "error_code", "call_sid", "component"}


def tool_results(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    out = [ln for name in (BRIDGE_LOG, CLIENT_LOG) for ln in lines(caplog.records, name)
           if ln["event"] == "tool_result"]
    for ln in out:
        assert set(ln) <= FIELDS, ln  # never arguments or result text
    return out


def one_prompt(rig: Any, text: str) -> None:
    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt(text)
        await sim.hangup()

    run(go())


def inline_rig(backend: FakeBackend, *steps: Any) -> Any:
    rig = make_rig(backend=backend, policy_doc=unmasked_policy_doc(), script=list(steps))
    write_call_state(rig.store)
    assert rig.bridge.tracing is None  # telemetry off: the line is logged all the same
    return rig


# --------------------------------------------------------------------------- inline


def test_an_inline_bank_success_logs_success(caplog: pytest.LogCaptureFixture) -> None:
    rig = inline_rig(FakeBackend({"list_recent_transactions": [TXNS]}),
                     tool("list_recent_transactions", "tu_tx", account="checking", limit=3), say("Four fifty."))
    with_context(caplog)
    with caplog.at_level(logging.INFO):
        one_prompt(rig, "What did I spend?")
    (ln,) = tool_results(caplog)
    assert ln == {"event": "tool_result", "tool": "list_recent_transactions", "tool_kind": "inline",
                  "outcome": "success", "call_sid": CALL_SID, "component": ln.get("component")}
    assert "Coffee" not in json.dumps(ln) and "checking" not in json.dumps(ln)


def test_a_bank_failure_logs_an_error(caplog: pytest.LogCaptureFixture) -> None:
    rig = inline_rig(FakeBackend({"list_recent_transactions": [RuntimeError("lambda down")]}),
                     tool("list_recent_transactions", account="checking"), say("Sorry."))
    with caplog.at_level(logging.INFO):
        one_prompt(rig, "transactions")
    (ln,) = tool_results(caplog)
    assert (ln["outcome"], ln["error_code"], ln["tool_kind"]) == ("error", "UNAVAILABLE", "inline")


def test_not_found_is_a_business_answer(caplog: pytest.LogCaptureFixture) -> None:
    rig = inline_rig(FakeBackend(), tool("list_recent_transactions", account="savings"), say("None."))
    with caplog.at_level(logging.INFO):
        one_prompt(rig, "savings transactions")
    (ln,) = tool_results(caplog)
    assert (ln["outcome"], ln["error_code"]) == ("refused", "NOT_FOUND")


def test_invalid_arguments_are_rejected_by_the_bridge(caplog: pytest.LogCaptureFixture) -> None:
    backend = FakeBackend()
    rig = inline_rig(backend, tool("list_recent_transactions", account="brokerage"), say("Which account?"))
    with caplog.at_level(logging.INFO):
        one_prompt(rig, "transactions")
    assert backend.calls == []
    (ln,) = tool_results(caplog)
    assert (ln["outcome"], ln["error_code"]) == ("rejected", "INVALID_ARGUMENTS")
    assert "brokerage" not in json.dumps(ln)


def test_an_unlisted_bank_code_is_an_error_logged_as_other(caplog: pytest.LogCaptureFixture) -> None:
    """Fails towards counting; the code itself isn't logged (only known codes are)."""
    rig = inline_rig(FakeBackend({"list_recent_transactions": [{"error": "SOMETHING_NEW 4111", "tool": "x"}]}),
                     tool("list_recent_transactions", account="checking"), say("Sorry."))
    with caplog.at_level(logging.INFO):
        one_prompt(rig, "transactions")
    (ln,) = tool_results(caplog)
    assert (ln["outcome"], ln["error_code"]) == ("error", "other")
    assert "4111" not in json.dumps(ln)


# --------------------------------------------------------------------------- hang-up, barge-in


class HeldBackend:
    """A bank call that waits until ``release`` is set, then answers ``answer``."""

    def __init__(self, answer: dict[str, Any]) -> None:
        self.answer = answer
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.calls: list[str] = []

    async def call(self, op: str, *, token: str | None, call_sid: str, args: Any,
                   idempotency_key: str | None = None) -> dict[str, Any]:
        self.calls.append(op)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return dict(self.answer)


def test_a_turn_cancelled_mid_bank_call_logs_cancelled_never_an_error(caplog: pytest.LogCaptureFixture) -> None:
    """A barge-in or hang-up cancels the running turn while the bank call is in flight: no result,
    ``cancelled`` (in neither count of the alarm)."""
    backend = HeldBackend(TXNS)
    d = direct([tool("list_recent_transactions", "tu_tx", account="checking"), say("Four fifty.")],
               backend=backend, policy_doc=unmasked_policy_doc())

    async def go() -> None:
        turn = asyncio.create_task(collect(d, "what did I spend?", context()))
        await asyncio.wait_for(backend.started.wait(), 3)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

    with caplog.at_level(logging.INFO):
        run(go())
    assert backend.cancelled
    (ln,) = tool_results(caplog)
    assert (ln["tool"], ln["outcome"]) == ("list_recent_transactions", "cancelled")
    assert "error_code" not in ln


def test_a_hang_up_during_a_bank_call_is_never_an_error(caplog: pytest.LogCaptureFixture) -> None:
    """The caller hangs up while the bank call runs: the socket closes and the turn is cancelled, so
    the call logs ``cancelled``. (Had the bank call finished first, the revoked token would have
    answered ``AUTH_LEVEL_INSUFFICIENT``: ``refused``, still not an error.)"""
    backend = HeldBackend({"error": "AUTH_LEVEL_INSUFFICIENT", "tool": "list_recent_transactions"})
    rig = make_rig(backend=backend, policy_doc=unmasked_policy_doc(),
                   script=[tool("list_recent_transactions", account="checking"), say("Sorry.")])
    write_call_state(rig.store)

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt_nowait("what did I spend?")
        await asyncio.wait_for(backend.started.wait(), 3)
        hang = asyncio.create_task(sim.hangup())
        rig.bridge.session_end(CALL_SID, "disconnect", close_code=1000)
        await asyncio.sleep(0.02)
        backend.release.set()
        await hang
        await rig.bridge.wait_idle()

    with caplog.at_level(logging.INFO):
        run(go())
    assert [ln["outcome"] for ln in tool_results(caplog)] == ["cancelled"]
    assert backend.cancelled


def test_a_barge_in_during_a_bank_call_is_never_an_error(caplog: pytest.LogCaptureFixture) -> None:
    backend = HeldBackend(TXNS)
    rig = make_rig(backend=backend, policy_doc=unmasked_policy_doc(),
                   script=[tool("list_recent_transactions", account="checking"), say("Four fifty."),
                           say("Sure.")])
    write_call_state(rig.store)

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt_nowait("what did I spend?")
        await asyncio.wait_for(backend.started.wait(), 3)
        await sim.interrupt("")
        await sim.idle()
        backend.release.set()
        await sim.idle()
        await sim.hangup()

    with caplog.at_level(logging.INFO):
        run(go())
    assert [ln["outcome"] for ln in tool_results(caplog)] == ["cancelled"]
    assert backend.cancelled


# --------------------------------------------------------------------------- hand-offs log nothing


def test_a_hand_off_logs_no_tool_result(caplog: pytest.LogCaptureFixture) -> None:
    rig = inline_rig(FakeBackend(), tool("escalate", reason="caller_requested_human", summary="wants a person"))
    with caplog.at_level(logging.INFO):
        one_prompt(rig, "a person please")
    assert tool_results(caplog) == []


def test_a_masked_get_balance_hand_off_and_its_prefetch_log_nothing(caplog: pytest.LogCaptureFixture) -> None:
    backend = balance_backend()
    d = direct([tool("get_balance", "tu_bal", account="checking")], backend=backend,
               policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=200))
    with caplog.at_level(logging.INFO):
        run(collect(d, "my checking balance", context()))
    assert backend.calls  # the prefetch ran
    assert tool_results(caplog) == []


def test_a_refused_hand_off_logs_rejected_never_error(caplog: pytest.LogCaptureFixture) -> None:
    """A hand-off tool called with another tool is refused before any hand-off (CALL_ALONE): it is
    logged ``rejected``; the other tool runs and logs its own outcome."""
    d = direct([[ToolCall("tu_b", "get_balance", {"account": "checking"}),
                 ToolCall("tu_a", "list_recent_transactions", {"account": "checking"}),
                 TurnEnd("tool_use")],
                say("OK.")])
    with caplog.at_level(logging.INFO):
        run(collect(d, "hi", context()))
    got = {ln["tool"]: (ln["outcome"], ln.get("error_code")) for ln in tool_results(caplog)}
    assert got == {"get_balance": ("rejected", "CALL_ALONE"), "list_recent_transactions": ("success", None)}


@pytest.mark.protocol_fake("the Strands client offers the model only the known tools; an unknown name "
                           "reaches the bridge only through the protocol")
def test_an_unknown_tool_is_rejected_and_its_name_cut_to_64(caplog: pytest.LogCaptureFixture) -> None:
    name = "x" * 200
    d = direct([[ToolCall("tu_x", name, {}), TurnEnd("tool_use")], say("OK.")])
    with caplog.at_level(logging.INFO):
        run(collect(d, "hi", context()))
    (ln,) = tool_results(caplog)
    assert (ln["tool"], ln["outcome"], ln["error_code"]) == ("x" * 64, "rejected", "UNKNOWN_TOOL")


# --------------------------------------------------------------------------- Gateway


def gateway_rig(model_calls: list[Any], gateway: Any) -> Any:
    rig = make_rig()
    model = FakeModel(model_calls)
    rig.bridge._agent = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: model,
                                           gateway_tools=gateway)
    write_call_state(rig.store)
    return rig


def test_a_gateway_success_logs_success_with_the_call(caplog: pytest.LogCaptureFixture) -> None:
    rig = gateway_rig([tool_msg(("tg", "level0___faq_lookup", {"q": "fees"})), text_msg("No fee.")],
                      default_gateway())
    with_context(caplog)
    with caplog.at_level(logging.INFO):
        one_prompt(rig, "any fees?")
    (ln,) = tool_results(caplog)
    assert (ln["tool"], ln["tool_kind"], ln["outcome"]) == ("level0___faq_lookup", "gateway", "success")
    assert ln["call_sid"] == CALL_SID and "error_code" not in ln


def test_a_raising_gateway_tool_logs_an_error(caplog: pytest.LogCaptureFixture) -> None:
    @strands_tool(name="level0___faq_lookup", description="fails",
                  inputSchema={"json": {"type": "object", "properties": {"q": {"type": "string"}}}})
    def _broken(q: str = "") -> str:
        raise ConnectionError("gateway down")

    rig = gateway_rig([tool_msg(("tg", "level0___faq_lookup", {"q": "fees"})), text_msg("Sorry.")],
                      FakeGatewayProvider([_broken]))
    with caplog.at_level(logging.INFO):
        one_prompt(rig, "any fees?")
    (ln,) = tool_results(caplog)
    assert (ln["tool_kind"], ln["outcome"]) == ("gateway", "error")


@pytest.mark.parametrize("answer, outcome, code", [
    ({"error": "UNAVAILABLE"}, "error", "UNAVAILABLE"),
    ({"error": "UNKNOWN_TOOL"}, "error", "UNKNOWN_TOOL"),  # our Gateway wiring is broken
    ({"error": "INVALID_ARGUMENT"}, "error", "INVALID_ARGUMENT"),
    ({"answer": "Open 9 to 5."}, "success", None),
])
def test_a_gateway_answer_with_an_error_code_is_classified_by_it(
        answer: dict[str, Any], outcome: str, code: str | None, caplog: pytest.LogCaptureFixture) -> None:
    """The Level-0 Lambda never raises: it answers ``{"error": CODE}`` inside a successful MCP
    result (JSON text), which the bridge must count as a failure."""
    @strands_tool(name="level0___faq_lookup", description="answers",
                  inputSchema={"json": {"type": "object", "properties": {"q": {"type": "string"}}}})
    def _answers(q: str = "") -> str:
        return json.dumps(answer)

    rig = gateway_rig([tool_msg(("tg", "level0___faq_lookup", {"q": "fees"})), text_msg("OK.")],
                      FakeGatewayProvider([_answers]))
    with caplog.at_level(logging.INFO):
        one_prompt(rig, "any fees?")
    (ln,) = tool_results(caplog)
    assert (ln["tool_kind"], ln["outcome"], ln.get("error_code")) == ("gateway", outcome, code)


def test_a_barge_in_mid_gateway_tool_is_never_an_error(caplog: pytest.LogCaptureFixture) -> None:
    runs = GatewayRuns()
    rig = gateway_rig([tool_msg(("tg", "level0___faq_lookup", {"q": "fees"})), text_msg("No fee."),
                       text_msg("Sure.")], default_gateway(runs, delay=2.0))

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt_nowait("any fees?")
        await wait_for(lambda: bool(runs.calls), 3)
        await sim.interrupt("")
        await sim.idle()
        await sim.hangup()

    with caplog.at_level(logging.INFO):
        run(go())
    assert runs.calls
    # The cancellation reaches the tool as ``CancelledError`` through Strands: no after-hook, no line.
    assert tool_results(caplog) == []
