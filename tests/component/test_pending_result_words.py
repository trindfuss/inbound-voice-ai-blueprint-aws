"""Words spoken with a pending result reach the agent as a real caller turn.

The result goes without ``callerUtterance`` and with ``next`` = ``await_caller``, in one turn
with the caller's words as ``user_text``. An ``await_caller`` result is recorded with no
model call, never answered (the scripts' result step is never asked for, ``tests.strands_parity``).
The timed acknowledgement may play in that turn (it is the caller's words). A pending result
without caller words behaves as before.
"""

from __future__ import annotations

from typing import Any


from owlbank.envelope import ResultEnvelope, Spoken
from tests.component.masking_fixtures import masked_policy_doc
from tests.component.test_bridge_fakes import (
    end_frame,
    make_rig,
    run,
    say,
    spoken,
    tool,
    write_call_state,
)
from tests.component.test_bridge_masking import GENERIC, collect, direct

BAL_TOOL = {"tool_use_id": "tu_bal", "name": "get_balance", "args": {"account": "checking"}}


def balance_env(next_: str = "await_caller", **kw: Any) -> dict[str, Any]:
    return ResultEnvelope(
        status="completed", tool="get_balance",
        spoken=[Spoken("balance.read", "Your checking balance is twelve dollars."),
                Spoken("common.anything_else", "Is there anything else I can help you with?")],
        data={"account": "checking", "available": "12.00"}, next=next_, **kw).to_dict()


def prompt_once(rig: Any, text: str) -> list[dict[str, Any]]:
    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        await sim.idle()
        frames = await sim.prompt(text)
        await sim.hangup()
        return frames

    return run(go())


def test_result_first_then_the_words_as_caller_text() -> None:
    rig = make_rig(script=[say("<silent/>"), say("Your savings balance is next.")])
    write_call_state(rig.store, pending_tool=BAL_TOOL, pending_result=balance_env())
    frames = prompt_once(rig, "What about savings?")
    first, second = rig.agent.calls
    (res,) = first["tool_results"]
    assert first["user_text"] is None
    assert res.tool_use_id == "tu_bal"
    assert res.content["next"] == "await_caller" and "callerUtterance" not in res.content
    ResultEnvelope.from_dict(res.content)  # still a valid envelope
    assert second["user_text"] == "What about savings?" and second["tool_results"] == []
    # The <silent/> reply is never spoken; the answer to the words is.
    assert spoken(frames) == "Your savings balance is next."
    assert [e for e in rig.events() if e["type"] == "ai_turn"][-1]["text"] == \
        "Your savings balance is next."


def test_an_interrupted_result_with_words_resumes_at_setup_in_two_hops() -> None:
    env = ResultEnvelope(status="interrupted", tool="make_payment", spoken=[], data={},
                         next="handle_utterance", callerUtterance="what are your hours",
                         completed=False).to_dict()
    rig = make_rig(script=[say("<silent/>"), say("We open at nine.")])
    write_call_state(rig.store, pending_tool={"tool_use_id": "tu_pay", "name": "make_payment",
                                              "args": {}},
                     pending_result=env, interrupted_utterance="what are your hours")

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.wait_reply()  # no prompt: resumed at setup
        await sim.hangup()
        return frames

    frames = run(go())
    first, second = rig.agent.calls
    assert first["tool_results"][0].content["next"] == "await_caller"
    assert first["tool_results"][0].content["status"] == "interrupted"
    assert "callerUtterance" not in first["tool_results"][0].content
    assert second["user_text"] == "what are your hours"
    assert spoken(frames) == "We open at nine."


def test_the_answer_to_the_words_may_itself_be_a_tool_call() -> None:
    rig = make_rig(script=[say("<silent/>"), tool("get_balance", "tu_sav", account="savings")])
    write_call_state(rig.store, pending_tool=BAL_TOOL, pending_result=balance_env())
    end = end_frame(prompt_once(rig, "And savings?"))
    assert end is not None and end["data"]["args"] == {"account": "savings"}
    assert rig.call()["pending_tool"]["tool_use_id"] == "tu_sav"


def test_text_on_the_result_hop_is_never_spoken_then_the_words_follow() -> None:
    """The front door already said it; the model's text for it is dropped."""
    rig = make_rig(script=[say("Anything else? "), say("Savings is next.")])
    write_call_state(rig.store, pending_tool=BAL_TOOL, pending_result=balance_env())
    frames = prompt_once(rig, "And savings?")
    assert spoken(frames) == "Savings is next."
    assert rig.agent.calls[1]["user_text"] == "And savings?"


def test_a_pending_result_without_caller_words_behaves_as_before() -> None:
    env = ResultEnvelope(status="verified", tool="authenticate", spoken=[], data={"level": 2},
                         next="retry_original_tool", forTool="list_recent_transactions").to_dict()
    rig = make_rig(script=[say("Here are your transactions.")])
    write_call_state(rig.store, pending_tool={"tool_use_id": "tu_auth", "name": "authenticate",
                                              "args": {"tool": "list_recent_transactions"}},
                     pending_result=env)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.wait_reply()
        await sim.hangup()
        return frames

    frames = run(go())
    (only,) = rig.agent.calls
    assert only["tool_results"][0].content["next"] == "retry_original_tool"
    assert "callerUtterance" not in only["tool_results"][0].content
    assert spoken(frames) == "Here are your transactions."


# --------------------------------------------------------------------------- the timer


def test_the_timer_phrase_can_play_while_the_result_hop_runs() -> None:
    d = direct([say("<silent/>"), say("Savings is next.")], agent_delay_s=0.15,
               policy_doc=masked_policy_doc(after_ms=50), pending_tool=BAL_TOOL,
               pending_result=balance_env())
    chunks, _ = run(collect(d, "What about savings?"))
    assert chunks == [GENERIC + " ", "Savings is next."]


def test_no_timer_without_caller_words() -> None:
    d = direct([say("<silent/>")], agent_delay_s=0.15, policy_doc=masked_policy_doc(after_ms=50),
               pending_tool=BAL_TOOL, pending_result=balance_env())
    chunks, _ = run(collect(d, ""))
    assert "".join(chunks).strip() == ""
    assert len(d.agent.calls) == 1
