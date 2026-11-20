"""Bridge flows over real TAC v2.5.0 with a ConversationRelay simulator.

Covers: a plain answer; an inline customer tool with the call token;
AUTH_LEVEL_INSUFFICIENT then ``authenticate`` (handoff data); return with
``pending_result`` (``await_caller`` -> ``callerUtterance``); an intent switch
(``handle_utterance`` resumed at ``setup``); ``escalate``; agent error -> retry ->
escalate; Memory gating; and that the token never reaches the model.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from tests.component.masking_fixtures import unmasked_policy_doc
from tests.component.test_bridge_fakes import (
    ANI,
    ANI_PROFILE,
    CALL_SID,
    CALL_TOKEN,
    SEALER,
    VERIFIED_EMAIL,
    VERIFIED_PROFILE,
    FakeBackend,
    FakeMemory,
    end_frame,
    front_door_update,
    h_msg,
    h_text,
    h_tool,
    make_rig,
    make_tac,
    run,
    say,
    spoken,
    tool,
    write_call_state,
)

from owlbank import agent_prompt
from owlbank import callstate as cs
from owlbank.envelope import ResultEnvelope, Spoken
from owlbank.tools_contract import AUTH_LEVEL_INSUFFICIENT
from owlbank.untrusted import wrap_result
from tac_bridge.agent_client import AgentError, TextDelta, ToolCall, TurnEnd
from tac_bridge.bridge import BridgeConfig


def assert_token_never_sent_to_model(rig: Any) -> None:
    text = rig.agent.all_inputs_text()
    assert CALL_TOKEN not in text
    sealed = rig.call().get("sealed_token")
    assert not sealed or sealed not in text


def balance_envelope(next_: str = "await_caller", **kw: Any) -> str:
    return ResultEnvelope(
        status="completed", tool="get_balance",
        spoken=[Spoken("balance.read", "Your checking account ending in 4821 has an available balance of "
                                       "$1,234.56."),
                Spoken("common.anything_else", "Is there anything else I can help you with?")],
        data={"account": "checking", "last4": "4821", "available": "1234.56"}, next=next_, **kw).to_json()


# --------------------------------------------------------------------------- 1. plain answer


def test_plain_answer_streams_tokens_with_verified_context() -> None:
    rig = make_rig(script=[say("Our branches ", "open at nine.")])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("When do you open?")
        await sim.hangup()
        return frames

    frames = run(go())
    assert spoken(frames) == "Our branches open at nine."
    assert frames[-1] == {"type": "text", "token": "", "last": True}
    assert end_frame(frames) is None

    (call,) = rig.agent.calls
    assert call["session_id"] == CALL_SID  # runtimeSessionId = actorId = CallSid
    assert call["user_text"] == "When do you open?"
    assert call["tool_results"] == []
    sp = call["system_prompt"]
    assert "OwlBank's voice assistant" in sp and "Speak English only" in sp
    assert "Thomas" in sp and "checking" in sp
    assert "card replacement" in sp  # verified Memory
    assert "Call language: en-US." in sp
    assert "{" not in sp.split("Rules:")[0]  # every placeholder filled
    types = [e["type"] for e in rig.events()]
    assert types == ["ai_turn"]
    assert rig.events()[0]["text"] == "Our branches open at nine."
    assert_token_never_sent_to_model(rig)


def test_spanish_call_uses_spanish_system_prompt() -> None:
    rig = make_rig(script=[say("Claro.")])
    write_call_state(rig.store, language="es-US")

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("Hola")
        await sim.hangup()

    run(go())
    sp = rig.agent.calls[0]["system_prompt"]
    assert "Hable solo en español" in sp and "Call language: es-US." in sp


@pytest.mark.parametrize("language", ["en-US", "es-US"])
def test_fallback_system_prompt_ends_with_the_call_language_line(language: str) -> None:
    """The fallback prompt (the build failed) is the static part (placeholder rule, then the
    call-language line) and the fallback per-call part, then the clock line, with no caller data
    (a frame that claims no verification either way)."""
    rig = make_rig(script=[say("OK.")])

    async def failing(*a: Any) -> Any:
        raise RuntimeError("policy table unavailable")

    rig.bridge._system_prompt = failing  # type: ignore[method-assign]  # noqa: SLF001
    write_call_state(rig.store, language=language)

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("Hi")
        await sim.hangup()

    run(go())
    sp = rig.agent.calls[0]["system_prompt"]
    per_call = agent_prompt.render_per_call(rig.bridge._catalog, language, caller=None,  # noqa: SLF001
                                            web_block=None, fallback=True)
    assert sp.endswith(f"\nCall language: {language}.\n{per_call}\n{sp.splitlines()[-1]}")  # then the clock
    assert per_call.splitlines()[0] == "<caller_context>" and per_call.splitlines()[-1] == "</caller_context>"
    assert per_call == agent_prompt.FALLBACK_CONTEXT[language]
    for key in agent_prompt.CALLER_CONTEXT_KEYS:  # neither frame's verification line
        assert rig.bridge._catalog.template(key, language).splitlines()[1].split(".")[0] not in sp  # noqa: SLF001
    assert "Thomas" not in sp


def test_silent_marker_is_never_spoken() -> None:
    rig = make_rig(script=[say("<sil", "ent/>")])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("ok")
        await sim.hangup()
        return frames

    frames = run(go())
    assert spoken(frames) == ""
    assert frames == [{"type": "text", "token": "", "last": True}]


# --------------------------------------------------------------------------- 2. inline customer tool


def test_inline_customer_tool_runs_with_the_call_token() -> None:
    txns = {"account": "checking", "transactions": [
        {"date": "2026-10-09", "description": "Coffee Example Co", "amount": "-4.50"}]}
    backend = FakeBackend({"list_recent_transactions": [txns]})
    rig = make_rig(backend=backend, policy_doc=unmasked_policy_doc(), script=[  # exact words (no masking phrase)
        # the text in the tool-calling message is never spoken (it used to be).
        h_msg(h_text("Let me check. "),
              h_tool("list_recent_transactions", "tu_tx", i=1, account="checking", limit=3)),
        say("Your latest was four fifty at Coffee Example Co."),
    ])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("What did I spend recently?")
        await sim.hangup()
        return frames

    frames = run(go())
    assert spoken(frames) == "Your latest was four fifty at Coffee Example Co."
    assert end_frame(frames) is None
    (bc,) = backend.calls
    assert bc == {"op": "list_recent_transactions", "token": CALL_TOKEN, "call_sid": CALL_SID,
                  "args": {"account": "checking", "limit": 3}}
    first, second = rig.agent.calls
    assert second["user_text"] is None
    (res,) = second["tool_results"]
    # the description reaches the model only inside an untrusted block.
    assert res.tool_use_id == "tu_tx" and res.status == "success" and res.content == wrap_result(txns)
    assert res.content["transactions"][0]["description"] == (
        '<untrusted_data source="transaction_description">Coffee Example Co</untrusted_data>')
    assert second["system_prompt"] == first["system_prompt"]
    types = [e["type"] for e in rig.events()]
    assert types == ["tool_call", "tool_result", "ai_turn"]
    tc = next(e for e in rig.events() if e["type"] == "tool_call")
    assert tc["tool"] == "list_recent_transactions" and tc["tool_use_id"] == "tu_tx"
    tr = next(e for e in rig.events() if e["type"] == "tool_result")
    assert tr["outcome"] == "success"
    assert_token_never_sent_to_model(rig)
    assert CALL_TOKEN not in json.dumps(rig.events())


def test_invalid_tool_args_are_returned_without_calling_the_back_end() -> None:
    backend = FakeBackend()
    rig = make_rig(backend=backend, script=[tool("list_recent_transactions", account="brokerage"),
                                            say("Which account?")])
    write_call_state(rig.store)

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("transactions")
        await sim.hangup()

    run(go())
    assert backend.calls == []
    res = rig.agent.calls[1]["tool_results"][0]
    assert res.status == "error" and res.content["error"] == "INVALID_ARGUMENTS"
    assert "brokerage" not in json.dumps(res.content)


def test_back_end_result_echoing_the_token_is_withheld() -> None:
    backend = FakeBackend({"list_recent_transactions": [lambda token, **_: {"debug": f"used {token}"}]})
    rig = make_rig(backend=backend, script=[tool("list_recent_transactions", account="checking"),
                                            say("Sorry.")])
    write_call_state(rig.store)

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("transactions")
        await sim.hangup()

    run(go())
    res = rig.agent.calls[1]["tool_results"][0]
    assert res.content == {"error": "UNAVAILABLE", "tool": "list_recent_transactions"}
    assert_token_never_sent_to_model(rig)


def test_back_end_failure_becomes_unavailable() -> None:
    backend = FakeBackend({"list_recent_transactions": [RuntimeError("lambda down")]})
    rig = make_rig(backend=backend, policy_doc=unmasked_policy_doc(),  # exact words (no masking phrase)
                   script=[tool("list_recent_transactions", account="checking"),
                           say("I can't reach that right now.")])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("transactions")
        await sim.hangup()
        return frames

    frames = run(go())
    assert rig.agent.calls[1]["tool_results"][0].content["error"] == "UNAVAILABLE"
    assert spoken(frames) == "I can't reach that right now."


# --------------------------------------------------------------------------- 3. step-up


def test_model_called_authenticate_hands_off_for_step_up() -> None:
    """The model's own ``authenticate`` call (e.g. after an inline tool's AUTH_LEVEL_INSUFFICIENT)
    still hands off as before. A refused front-door tool takes the step-up shortcut instead
    (tests/component/test_stepup_shortcut.py)."""
    backend = FakeBackend({})
    rig = make_rig(backend=backend, script=[
        tool("authenticate", "tu_auth", tool="make_payment"),
    ])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("Pay 250 dollars on my card from checking")
        await sim.hangup()
        return frames

    frames = run(go())
    assert len(rig.agent.calls) == 1
    # no AI speech; then TAC's last:true, then end with handoffData
    assert spoken(frames) == ""
    assert [f["type"] for f in frames] == ["text", "end"]
    end = end_frame(frames)
    assert end is not None
    data = end["data"]
    assert data["reason"] == "step-up" and data["tool"] == "make_payment"
    assert len(data["journal"]) >= 20 and CALL_SID not in data["journal"]  # opaque, not seg:/call:
    assert set(data) <= {"reason", "tool", "args", "journal", "conversationId", "profileId"}
    assert data["conversationId"].startswith("conv_")  # TAC's Orchestrator conversation
    assert data["profileId"] == VERIFIED_PROFILE  # the verified profile, never the ANI one
    raw = end["frame"]["handoffData"]
    for secret in (CALL_TOKEN, "C-1001", ANI, "Thomas"):
        assert secret not in raw
    call = rig.call()
    assert call["pending_tool"] == {"tool_use_id": "tu_auth", "name": "authenticate",
                                    "args": {"tool": "make_payment"}}
    assert call["step_up_target"] == "make_payment"
    assert set(call) <= set(cs.CallState.__dataclass_fields__)  # typed fields only
    assert SEALER.unseal(call["sealed_token"], call_sid=CALL_SID) == CALL_TOKEN  # untouched
    assert CALL_TOKEN not in json.dumps(call)
    assert call["version"] > 1
    handoff = [e for e in rig.events() if e["type"] == "handoff"]
    assert len(handoff) == 1 and handoff[0]["tool"] == "authenticate"
    assert_token_never_sent_to_model(rig)


def test_get_balance_below_level_steps_up_without_a_model_hop() -> None:
    """Refused only for the level, so the bridge hands off for step-up at once."""
    rig = make_rig(script=[tool("get_balance", "tu_b", account="checking")])
    write_call_state(rig.store, verified=False, token=None)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("What's my balance?")
        await sim.hangup()
        return frames

    frames = run(go())
    assert len(rig.agent.calls) == 1  # no AUTH_LEVEL_INSUFFICIENT round trip
    end = end_frame(frames)
    assert end is not None and end["data"]["reason"] == "step-up" and end["data"]["tool"] == "get_balance"
    assert rig.call()["pending_tool"] == {"tool_use_id": "tu_b", "name": "get_balance",
                                          "args": {"account": "checking"}}


def test_get_balance_hands_off_with_validated_args() -> None:
    rig = make_rig(script=[tool("get_balance", "tu_bal", account="savings")])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("savings balance please")
        await sim.hangup()
        return frames

    end = end_frame(run(go()))
    assert end is not None
    assert end["data"]["reason"] == "tool" and end["data"]["tool"] == "get_balance"
    assert end["data"]["args"] == {"account": "savings"}
    assert rig.call()["pending_tool"]["tool_use_id"] == "tu_bal"


# --------------------------------------------------------------------------- 4. back from a deterministic flow


def test_pending_result_is_sent_on_first_prompt_with_caller_utterance() -> None:
    rig = make_rig(script=[say("<silent/>"),
                           say("Your savings balance needs another lookup. Shall I check it?"),
                           say("Sure.")])
    write_call_state(rig.store, pending_tool={"tool_use_id": "tu_bal", "name": "get_balance",
                                              "args": {"account": "checking"}},
                     pending_result=balance_envelope())

    async def go() -> tuple[list[dict[str, Any]], int]:
        sim = rig.sim()
        await sim.connect()
        await sim.idle()
        calls_before_prompt = len(rig.agent.calls)  # nothing sent at reconnect (await_caller)
        await sim.prompt("And what about savings?")
        await sim.prompt("Yes please")
        await sim.hangup()
        return sim.socket.sent, calls_before_prompt

    _, before = run(go())
    assert before == 0
    first, words, second = rig.agent.calls
    assert first["user_text"] is None
    (res,) = first["tool_results"]
    assert res.tool_use_id == "tu_bal" and res.status == "success"
    env = res.content
    # the result first (the agent stays silent), then the words as a real caller turn
    assert "callerUtterance" not in env
    assert env["next"] == "await_caller"
    assert words["user_text"] == "And what about savings?" and words["tool_results"] == []
    assert [s["template"] for s in env["spoken"]] == ["balance.read", "common.anything_else"]
    assert env["data"]["available"] == "1234.56"
    ResultEnvelope.from_dict(env)  # still a valid envelope
    # cleared exactly once
    call = rig.call()
    for k in ("pending_result", "pending_tool", "interrupted_utterance"):
        assert call[k] is None
    assert second["user_text"] == "Yes please" and second["tool_results"] == []
    tr = [e for e in rig.events() if e["type"] == "tool_result"]
    assert tr[0]["tool_use_id"] == "tu_bal" and tr[0]["outcome"] == "completed"
    assert_token_never_sent_to_model(rig)


def test_step_up_result_then_retry_of_the_original_tool() -> None:
    """After step-up the model retries make_payment; with confirm deterministic (the default) the retry is a confirm handoff, never a bank call from the bridge."""
    backend = FakeBackend({"make_payment": [{"status": "completed", "confirmation": "PMT-1"}]})
    verified = ResultEnvelope(status="verified", tool="authenticate",
                              spoken=[Spoken("stepup.success", "Thanks, you're verified.")],
                              data={"forTool": "make_payment"}, next="retry_original_tool",
                              forTool="make_payment").to_json()
    rig = make_rig(backend=backend, script=[
        tool("make_payment", "tu_pay2", from_account="checking", to="red", amount="250.00"),
    ])
    write_call_state(rig.store, level=3, pending_tool={"tool_use_id": "tu_auth", "name": "authenticate",
                                                       "args": {"tool": "make_payment"}},
                     pending_result=verified)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.wait_reply()  # no caller prompt: the retry starts at setup (no dead air)
        await sim.hangup()
        return frames

    frames = run(go())
    first = rig.agent.calls[0]
    assert first["user_text"] is None
    assert first["tool_results"][0].tool_use_id == "tu_auth"
    assert first["tool_results"][0].content["next"] == "retry_original_tool"
    assert "callerUtterance" not in first["tool_results"][0].content
    assert backend.calls == []  # the bridge never runs a deterministic-confirm tool
    end = end_frame(frames)
    assert end is not None and end["data"]["reason"] == "confirm" and end["data"]["tool"] == "make_payment"
    assert "args" not in end["data"]  # amounts and accounts stay in call state
    assert rig.call()["pending_tool"] == {"tool_use_id": "tu_pay2", "name": "make_payment", "args": {
        "from_account": "checking", "to": "red", "amount": "250.00"}}


# --------------------------------------------------------------------------- 5. intent switch


def test_handle_utterance_resumes_at_setup_without_waiting_for_a_prompt() -> None:
    interrupted = ResultEnvelope(status="interrupted", tool="get_balance",
                                 spoken=[Spoken("balance.which", "Checking or savings?")],
                                 data={}, next="handle_utterance",
                                 callerUtterance="actually I lost my card").to_json()
    rig = make_rig(script=[say("<silent/>"), say("I'm sorry to hear that. ", "I can help you lock it.")])
    write_call_state(rig.store, pending_tool={"tool_use_id": "tu_bal", "name": "get_balance",
                                              "args": {}},
                     pending_result=interrupted, interrupted_utterance="actually I lost my card")

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.wait_reply()  # no prompt sent
        await sim.hangup()
        return frames

    frames = run(go())
    assert spoken(frames) == "I'm sorry to hear that. I can help you lock it."
    call, words = rig.agent.calls
    res = call["tool_results"][0]
    assert res.tool_use_id == "tu_bal"
    assert res.content["next"] == "await_caller"  # the words follow as caller text
    assert "callerUtterance" not in res.content
    assert words["user_text"] == "actually I lost my card"
    assert rig.call()["pending_result"] is None and rig.call()["interrupted_utterance"] is None


def test_handle_utterance_reply_can_itself_hand_off() -> None:
    interrupted = ResultEnvelope(status="interrupted", tool="get_balance", spoken=[], data={},
                                 next="handle_utterance", callerUtterance="get me a person").to_json()
    rig = make_rig(script=[say("<silent/>"),
                           tool("escalate", "tu_esc", reason="caller_requested_human",
                                summary="Caller asked for a person during a balance check.")])
    write_call_state(rig.store, pending_tool={"tool_use_id": "tu_bal", "name": "get_balance", "args": {}},
                     pending_result=interrupted, interrupted_utterance="get me a person")

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.wait_reply()
        await sim.hangup()
        return frames

    end = end_frame(run(go()))
    assert end is not None and end["data"]["escalationReason"] == "caller_requested_human"
    assert rig.agent.calls[1]["user_text"] == "get me a person"  # the words' own hop


# --------------------------------------------------------------------------- 6. escalate


def test_escalate_stores_reason_and_summary_server_side_only() -> None:
    summary = "Caller reports an unknown charge at Coffee Example Co.\x07 Wants it investigated."
    rig = make_rig(script=[tool("escalate", "tu_esc", reason="fraud_suspected", summary=summary)])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("There's a charge I didn't make")
        await sim.hangup()
        return frames

    end = end_frame(run(go()))
    assert end is not None
    assert end["data"] == {"reason": "escalate", "tool": "escalate", "escalationReason": "fraud_suspected",
                           "journal": end["data"]["journal"],
                           "conversationId": end["data"]["conversationId"],
                           "profileId": VERIFIED_PROFILE}
    assert end["data"]["conversationId"].startswith("conv_")
    assert "Coffee" not in end["frame"]["handoffData"]
    call = rig.call()
    assert call["escalation"]["reason"] == "fraud_suspected"
    assert call["escalation"]["summary"].startswith("Caller reports an unknown charge")
    assert "\x07" not in call["escalation"]["summary"]
    assert call["pending_tool"]["name"] == "escalate"
    assert call["pending_tool"]["args"] == {"reason": "fraud_suspected"}
    esc = [e for e in rig.events() if e["type"] == "escalation"]
    assert esc and esc[0]["reason"] == "fraud_suspected"


def test_unknown_escalation_reason_maps_to_unable_to_resolve() -> None:
    rig = make_rig(script=[tool("escalate", "tu_esc", reason="make_me_a_sandwich", summary="x")])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("help")
        await sim.hangup()
        return frames

    end = end_frame(run(go()))
    assert end is not None and end["data"]["escalationReason"] == "unable_to_resolve"


# --------------------------------------------------------------------------- 7. errors


def test_agent_error_apology_retry_then_escalate() -> None:
    rig = make_rig(script=[AgentError("throttled"), AgentError("throttled again")])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("Hello?")
        await sim.hangup()
        return frames

    frames = run(go())
    assert len(rig.agent.calls) == 2  # one retry
    assert rig.agent.calls[0]["user_text"] == rig.agent.calls[1]["user_text"] == "Hello?"
    assert spoken(frames) == "I'm sorry, something went wrong on our end. "  # apology once
    end = end_frame(frames)
    assert end is not None
    assert end["data"]["reason"] == "escalate" and end["data"]["escalationReason"] == "unable_to_resolve"
    assert rig.call()["escalation"]["source"] == "bridge"
    assert rig.call().get("pending_tool") is None


def test_agent_error_then_successful_retry() -> None:
    rig = make_rig(script=[AgentError("blip"), say("Here I am.")])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("Hello?")
        await sim.hangup()
        return frames

    frames = run(go())
    assert spoken(frames) == "I'm sorry, something went wrong on our end. Here I am."
    assert end_frame(frames) is None


def test_agent_error_on_a_result_hop_then_successful_retry() -> None:
    """The bridge retries a failed hop once with the same input. A result hop whose
    model call failed is retried with the same results, and the retry answers (on the Strands
    backend the results are already in the history then, so the client must accept them again)."""
    txns = {"account": "checking", "transactions": []}
    backend = FakeBackend({"list_recent_transactions": [txns]})
    rig = make_rig(backend=backend, policy_doc=unmasked_policy_doc(), script=[
        tool("list_recent_transactions", "tu_tx", account="checking"),
        AgentError("blip"),
        say("Nothing recent on checking."),
    ])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("Anything recent?")
        await sim.hangup()
        return frames

    frames = run(go())
    assert spoken(frames) == "I'm sorry, something went wrong on our end. Nothing recent on checking."
    assert end_frame(frames) is None
    assert len(backend.calls) == 1  # the tool ran once; only the agent hop was retried
    first, failed, retried = rig.agent.calls
    assert failed["tool_results"] == retried["tool_results"] and retried["user_text"] is None
    if hasattr(rig.agent, "model"):  # Strands: the model saw the result once, then answered
        last = rig.agent.model.calls[-1]
        results = [b for m in last.messages for b in m["content"] if "toolResult" in b]
        assert [b["toolResult"]["toolUseId"] for b in results] == ["tu_tx"]


def test_agent_error_retry_does_not_repeat_the_callers_words() -> None:
    """A retried text hop gives the model the caller's words once."""
    rig = make_rig(script=[AgentError("blip"), say("Here I am.")])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("Hello?")
        await sim.hangup()
        return frames

    frames = run(go())
    assert spoken(frames) == "I'm sorry, something went wrong on our end. Here I am."
    if hasattr(rig.agent, "model"):  # Strands: what the model got on the retry
        texts = [b["text"] for m in rig.agent.model.calls[-1].messages for b in m["content"] if "text" in b]
        assert texts == ["Hello?"]


def test_mid_stream_error_and_non_retryable_error_escalate() -> None:
    rig = make_rig(script=[[TextDelta("Let me "), AgentError("stream", retryable=False)]])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("Hi")
        await sim.hangup()
        return frames

    frames = run(go())
    assert len(rig.agent.calls) == 1  # not retryable
    assert end_frame(frames)["data"]["escalationReason"] == "unable_to_resolve"


def test_no_call_state_at_setup_closes_the_socket() -> None:
    """With no call state there is no nonce to match, so setup is refused."""
    rig = make_rig(script=[])  # no call state written

    async def go() -> None:
        sim = rig.sim()
        with pytest.raises(AssertionError, match="timed out"):
            await sim.connect()
        await asyncio.wait_for(sim.task, 3)
        assert sim.socket.closed and not sim.peek.verified

    run(go())
    assert rig.agent.calls == [] and rig.backend.calls == []


def test_missing_call_state_apologizes_and_escalates() -> None:
    rig = make_rig(script=[say("never")])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        rig.store.delete("calls", {"call_sid": CALL_SID})  # gone (expired) after setup
        frames = await sim.prompt("Hi")
        await sim.hangup()
        return frames

    frames = run(go())
    assert rig.agent.calls == []
    assert end_frame(frames)["data"]["reason"] == "escalate"


def test_barge_in_during_handoff_resends_end_on_next_prompt() -> None:
    rig = make_rig(script=[[TextDelta("One moment."), ToolCall("tu_bal", "get_balance", {"account": "checking"}),
                            TurnEnd("tool_use")]])
    # Simulate a lost `end`: pending_tool was stored, but TAC never sent `end`.
    write_call_state(rig.store, pending_tool={"tool_use_id": "tu_bal", "name": "get_balance",
                                              "args": {"account": "checking"}})

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("hello?")
        await sim.hangup()
        return frames

    frames = run(go())
    assert rig.agent.calls == []  # no model turn: the pending call is still open
    end = end_frame(frames)
    assert end is not None and end["data"]["tool"] == "get_balance"
    assert end["data"]["args"] == {"account": "checking"}


def test_interrupt_cancels_the_stream_and_is_journaled() -> None:
    rig = make_rig(script=[[TextDelta(f"word{i} ") for i in range(200)] + [TurnEnd("end_turn")],
                           say("Sure, savings.")])
    rig.agent.delay_s = 0.005
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt_nowait("tell me everything")
        await __import__("asyncio").sleep(0.05)
        await sim.interrupt("word0 word1")
        await sim.idle()
        frames = await sim.prompt("savings please")
        await sim.hangup()
        return frames

    frames = run(go())
    assert len(rig.agent.calls) == 2 and rig.agent.closed == 2
    assert spoken(frames) == "Sure, savings."
    first_turn = [e for e in rig.events() if e["type"] == "ai_turn" and not e.get("interrupted")]
    assert all("word199" not in e.get("text", "") for e in first_turn)  # cancelled before the end
    assert any(e.get("interrupted") for e in rig.events())


# --------------------------------------------------------------------------- 8. Memory gating


def test_memory_is_never_loaded_before_verification() -> None:
    rig = make_rig(script=[say("I can help with general questions.")])
    write_call_state(rig.store, verified=False, token=None)

    async def go() -> Any:
        sim = rig.sim()
        await sim.connect()
        assert sim.session.profile_id == ANI_PROFILE  # what Orchestrator bound from the ANI
        await sim.prompt("Hi")
        session = sim.session
        await sim.hangup()
        return session

    session = run(go())
    assert rig.memory.lookups == [] and rig.memory.retrievals == []
    assert session.profile_id is None and session.profile is None
    sp = rig.agent.calls[0]["system_prompt"]
    # The unverified frame (memory gating): no name, profile or Memory.
    assert "<caller_context>\nVerified: no." in sp and "Recent interactions from other channels:" not in sp
    assert "Thomas" not in sp


def test_memory_uses_the_verified_email_through_tacs_client() -> None:
    rig = make_rig(memory="tac", script=[say("Hi Thomas.")])
    write_call_state(rig.store)
    twilio_memory = rig.tac.conversation_memory_client

    async def go() -> Any:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("Hi")
        session = sim.session
        await sim.hangup()
        return session

    session = run(go())
    assert twilio_memory.lookups == [("email", VERIFIED_EMAIL)]  # a valid type, never the ANI
    assert twilio_memory.get_profile_ids == [VERIFIED_PROFILE]  # context.profile was cleared
    assert twilio_memory.recall_ids == [VERIFIED_PROFILE]
    assert session.profile_id == VERIFIED_PROFILE
    assert session.profile.traits["Contact"]["name"] == "Thomas Doe"
    sp = rig.agent.calls[0]["system_prompt"]
    assert "Verified customer asked about card replacement" in sp
    assert "ANI HOUSEHOLD" not in sp and "Household (ANI)" not in sp


def test_memory_is_loaded_once_per_call_and_reloaded_on_verification_change() -> None:
    rig = make_rig(script=[say("a"), say("b")])
    write_call_state(rig.store)

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("one")
        await sim.prompt("two")
        await sim.hangup()

    run(go())
    assert rig.memory.lookups == [VERIFIED_EMAIL]
    assert rig.memory.retrievals == [{"profile_id": VERIFIED_PROFILE, "profile": None}]
    assert rig.agent.calls[0]["system_prompt"] == rig.agent.calls[1]["system_prompt"]


def test_a_miss_creates_the_verified_profile_with_the_email_trait_only() -> None:
    rig = make_rig(memory="tac", script=[say("Hi.")])
    write_call_state(rig.store)
    rig.tac.conversation_memory_client.by_email = {}

    async def go() -> Any:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("Hi")
        s = sim.session
        await sim.hangup()
        return s

    session = run(go())
    mem = rig.tac.conversation_memory_client
    assert mem.lookups == [("email", VERIFIED_EMAIL)]
    assert mem.recall_ids == [] and mem.get_profile_ids == []  # nothing to recall for a new profile
    assert session.profile_id.startswith("mem_profile_created_")  # created, so the next call finds it
    assert mem.creates == [{"Contact": {"email": VERIFIED_EMAIL}}]  # the verified trait only: no phone
    assert "Recent interactions from other channels: none" in rig.agent.calls[0]["system_prompt"]
    assert "ANI HOUSEHOLD" not in rig.agent.calls[0]["system_prompt"]


def _one_call(rig: Any) -> Any:
    async def go() -> Any:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("Hi")
        s = sim.session
        await sim.hangup()
        return s

    return run(go())


def test_the_second_call_finds_the_profile_the_first_call_created() -> None:
    rig = make_rig(memory="tac", script=[say("Hi."), say("Hi again.")])
    write_call_state(rig.store, call_sid=CALL_SID)
    rig.tac.conversation_memory_client.by_email = {}
    first = _one_call(rig)
    mem = rig.tac.conversation_memory_client
    assert len(mem.creates) == 1 and first.profile_id == mem.by_email[VERIFIED_EMAIL][0]
    rig2 = make_rig(memory="tac", script=[say("Hi again.")])
    write_call_state(rig2.store)
    rig2.tac.conversation_memory_client.by_email = dict(mem.by_email)
    second = _one_call(rig2)
    assert second.profile_id == first.profile_id
    assert rig2.tac.conversation_memory_client.creates == []  # found, not created again


def test_a_failed_lookup_continues_without_memory_and_creates_nothing() -> None:
    rig = make_rig(memory="tac", script=[say("Hi.")])
    write_call_state(rig.store)
    mem = rig.tac.conversation_memory_client
    mem.lookup_error = httpx.ConnectError("memory unreachable")
    session = _one_call(rig)
    assert mem.creates == [] and mem.recall_ids == []
    assert session.profile_id is None and session.profile is None
    assert len(rig.agent.calls) == 1  # the call went on


def test_a_failed_create_continues_without_memory() -> None:
    rig = make_rig(memory="tac", script=[say("Hi.")])
    write_call_state(rig.store)
    mem = rig.tac.conversation_memory_client
    mem.by_email = {}
    mem.create_error = httpx.ReadTimeout("slow")
    session = _one_call(rig)
    assert len(mem.creates) == 1
    assert session.profile_id is None and session.profile is None  # no ANI fallback
    assert len(rig.agent.calls) == 1
    assert "Recent interactions from other channels: none" in rig.agent.calls[0]["system_prompt"]


def test_a_hanging_create_is_bounded_by_the_memory_timeout() -> None:
    class Hanging(FakeMemory):
        async def create_profile(self, identifier: str) -> str:
            await asyncio.sleep(30)
            return "never"

    rig = make_rig(memory=Hanging(profiles={}), script=[say("Hi.")],
                   config=BridgeConfig(memory_timeout_s=0.05))
    write_call_state(rig.store)
    session = _one_call(rig)
    assert session.profile_id is None and len(rig.agent.calls) == 1


def test_the_ani_is_never_the_memory_identifier() -> None:
    rig = make_rig(memory="tac", script=[say("Hi.")])
    write_call_state(rig.store)
    rig.tac.conversation_memory_client.by_email = {}
    _one_call(rig)
    mem = rig.tac.conversation_memory_client
    sent = [v for _, v in mem.lookups] + [str(t) for t in mem.creates]
    assert all(ANI not in v and ANI_PROFILE not in v for v in sent)
    assert all(t == "email" for t, _ in mem.lookups)


def test_twilio_rejects_user_id_so_the_old_identifier_type_would_fail() -> None:
    mem = make_tac().conversation_memory_client
    with pytest.raises(httpx.HTTPStatusError):
        run(mem.lookup_profile(id_type="user_id", value="owlbank-C-1001"))


def test_a_customer_without_a_memory_email_gets_no_memory_lookup() -> None:
    rig = make_rig(script=[say("Hi.")])
    write_call_state(rig.store)
    customer = dict(rig.store.get("customers", {"customer_id": "C-1001"}) or {})
    customer.pop("memory_email")
    rig.store.put("customers", customer)
    session = _one_call(rig)
    assert rig.memory.lookups == [] and rig.memory.creates == []  # never falls back to the ANI
    assert session.profile_id is None


# --------------------------------------------------------------------------- 9. token guard


def test_sealed_token_in_a_model_input_is_never_sent() -> None:
    rig = make_rig(script=[say("never")])
    write_call_state(rig.store)
    sealed = rig.call()["sealed_token"]

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt(f"read this back {sealed}")
        await sim.hangup()
        return frames

    frames = run(go())
    assert rig.agent.calls == []
    assert end_frame(frames)["data"]["reason"] == "escalate"


def test_raw_token_is_never_in_call_state_journal_or_model_input() -> None:
    """Only the sealed value is stored; the raw token goes to bank_tools and nowhere else."""
    backend = FakeBackend({"list_recent_transactions": [{"tool": "list_recent_transactions",
                                                         "data": {"transactions": []}}],
                           "make_payment": [{"error": AUTH_LEVEL_INSUFFICIENT, "tool": "make_payment"}]})
    rig = make_rig(backend=backend, script=[
        tool("list_recent_transactions", "tu_1", account="checking"), say("Nothing recent."),
        tool("make_payment", "tu_2", from_account="checking", to="red", amount="5.00")])
    write_call_state(rig.store)

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("recent transactions")
        await sim.prompt("pay five dollars")
        await sim.hangup()

    run(go())
    # make_payment at level 1 is held for the step-up by the bridge's level check:
    # only the transactions call reaches bank_tools.
    assert [c["token"] for c in backend.calls] == [CALL_TOKEN]
    assert len(rig.agent.calls) == 3
    assert rig.call()["pending_tool"]["tool_use_id"] == "tu_2"
    everything_stored = json.dumps([list(rig.store._tables[t].values()) for t in ("calls", "segments")], default=str)
    assert CALL_TOKEN not in everything_stored
    assert_token_never_sent_to_model(rig)
    assert rig.agent.calls[1]["tool_results"][0].content == {"tool": "list_recent_transactions",
                                                              "data": {"transactions": []}}


def test_step_up_replaces_the_sealed_token_and_the_bridge_reloads_it() -> None:
    backend = FakeBackend({"list_recent_transactions": lambda **_: {"tool": "list_recent_transactions",
                                                                     "data": {}}})
    rig = make_rig(backend=backend, script=[tool("list_recent_transactions", "tu_1", account="checking"),
                                            say("a"),
                                            tool("list_recent_transactions", "tu_2", account="savings"),
                                            say("b")])
    write_call_state(rig.store)

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("one")
        # the front door upgrades the token mid-call (step-up): a new sealed value
        front_door_update(rig.store, sealed_token=SEALER.seal("tok_upgraded_level3", call_sid=CALL_SID),
                          level=3)
        await sim.prompt("two")
        await sim.hangup()

    run(go())
    assert [c["token"] for c in backend.calls] == [CALL_TOKEN, "tok_upgraded_level3"]


@pytest.mark.parametrize("sealed", ["not-a-sealed-value", "MISSING",
                                    SEALER.seal(CALL_TOKEN, call_sid="CA" + "9" * 32)])
def test_unsealable_or_missing_token_is_level_0(sealed: str) -> None:
    backend = FakeBackend()
    rig = make_rig(backend=backend, script=[tool("list_recent_transactions", "tu_1", account="checking"),
                                            tool("authenticate", "tu_2", tool="list_recent_transactions")])
    write_call_state(rig.store)
    if sealed == "MISSING":
        front_door_update(rig.store, sealed_token=None)
    else:  # corrupt, or sealed for another call (wrong encryption context)
        front_door_update(rig.store, sealed_token=sealed)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("transactions")
        await sim.hangup()
        return frames

    frames = run(go())
    assert backend.calls == []
    assert rig.agent.calls[1]["tool_results"][0].content == {"error": AUTH_LEVEL_INSUFFICIENT,
                                                              "tool": "list_recent_transactions"}
    assert end_frame(frames)["data"]["reason"] == "step-up"


@pytest.mark.parametrize("hops", [2])
def test_tool_loop_is_bounded(hops: int) -> None:
    backend = FakeBackend({"list_recent_transactions": lambda **_: {"transactions": []}})
    script = [tool("list_recent_transactions", f"tu_{i}", account="checking") for i in range(10)]
    rig = make_rig(backend=backend, script=script, config=BridgeConfig(max_hops=hops))
    write_call_state(rig.store)

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("loop")
        await sim.hangup()

    run(go())
    assert len(rig.agent.calls) == hops


def test_handoff_alongside_other_inline_calls_must_be_called_alone() -> None:
    backend = FakeBackend({"list_recent_transactions": [{"tool": "list_recent_transactions", "data": {}}]})
    rig = make_rig(backend=backend, script=[
        [ToolCall("tu_a", "list_recent_transactions", {"account": "checking"}),
         ToolCall("tu_b", "get_balance", {"account": "checking"}), TurnEnd("tool_use")],
        tool("get_balance", "tu_c", account="checking")])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("transactions and balance")
        await sim.hangup()
        return frames

    end = end_frame(run(go()))
    results = rig.agent.calls[1]["tool_results"]
    assert [r.tool_use_id for r in results] == ["tu_a", "tu_b"]
    assert results[1].content == {"error": "CALL_ALONE", "tool": "get_balance"}
    assert end is not None and rig.call()["pending_tool"]["tool_use_id"] == "tu_c"
