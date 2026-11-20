"""An ``await_caller`` result (the front door already spoke the outcome) is recorded in
the agent's history and never answered.

Replays call CA1982…841d: the front door confirmed a payment (``next`` = ``await_caller``,
``executed`` true) and the caller's words came with the return ("transfer me to an agent"). The
model spoke the confirmation again before escalating. Now one ``turn`` carries the result and the
words: the result is recorded with no model call, and the model is called once, for the words.

The scripts have one step per agent input (the result, then the words); the result's step is never
asked for (``tests.strands_parity``).
"""

from __future__ import annotations

import logging
import string
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank.envelope import ResultEnvelope, Spoken
from tac_bridge.agent_client import AgentError
from tests.component.masking_fixtures import masked_policy_doc
from tests.component.test_bridge_fakes import (
    CALL_SID,
    end_frame,
    make_rig,
    run,
    say,
    spoken,
    tool,
    write_call_state,
)
from tests.component.test_bridge_masking import collect, direct

PAY_TOOL = {"tool_use_id": "tu_pay", "name": "make_payment",
            "args": {"from_account": "checking", "payee": "electric", "amount": "120.00"}}
CONFIRMATION = "Your payment of one hundred twenty dollars to Electric is done. Confirmation one two three."
#: What the Strands model said again in the incident (here: scripted for the result hop).
REPEATED = "Your payment of $120 to Electric is confirmed, confirmation 123."
WORDS = "transfer me to an agent"


def payment_env() -> dict[str, Any]:
    return ResultEnvelope(
        status="completed", tool="make_payment",
        spoken=[Spoken("confirm.make_payment.done", CONFIRMATION),
                Spoken("common.anything_else", "Is there anything else I can help you with?")],
        data={"executed": True, "confirmation": "123"}, next="await_caller").to_dict()


ESCALATE = tool("escalate", "tu_esc", reason="caller_requested_human", summary="Caller asked for a person.")


def prompt_once(rig: Any, text: str) -> list[dict[str, Any]]:
    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        await sim.idle()
        frames = await sim.prompt(text)
        await sim.hangup()
        return frames

    return run(go())


def strands_history(agent: Any) -> list[dict[str, Any]]:
    built = run(agent.client._built(CALL_SID, "cached"))  # noqa: SLF001 - test seam
    return list(built.agent.messages)


def result_calls(agent: Any) -> list[Any]:
    """The Strands model calls that saw the payment's toolResult."""
    return [c for c in agent.model.calls
            if any("toolResult" in b and b["toolResult"].get("toolUseId") == "tu_pay"
                   for m in c.messages for b in m.get("content") or ())]


def test_incident_replay_the_payment_is_not_spoken_again_and_the_escalation_follows(
        caplog: pytest.LogCaptureFixture) -> None:
    # Step 1 (the result, never asked for): the confirmation the model repeated in the incident.
    # Step 2 (the words): it escalates.
    rig = make_rig(script=[say(REPEATED), ESCALATE])
    write_call_state(rig.store, pending_tool=PAY_TOOL, pending_result=payment_env())
    with caplog.at_level(logging.INFO, logger="tac_bridge.bridge"):
        frames = prompt_once(rig, WORDS)

    heard = spoken(frames)
    assert REPEATED not in heard and "payment" not in heard.lower() and "123" not in heard
    end = end_frame(frames)
    assert end is not None and end["data"]["reason"] == "escalate"
    # Nothing of the result hop is journaled as agent speech.
    assert not [e for e in rig.events() if e["type"] == "ai_turn" and e.get("text")]
    # The result went with next = await_caller; the words went as the caller's words.
    result_rec = rig.agent.calls[0]
    (res,) = result_rec["tool_results"]
    assert res.tool_use_id == "tu_pay" and res.content["next"] == "await_caller"
    assert res.content["data"]["executed"] is True
    assert rig.agent.calls[1]["user_text"] == WORDS
    # One model call for the whole turn, with the result already in history before the words.
    (call,) = result_calls(rig.agent)
    last_two = call.messages[-2:]
    assert last_two[0]["role"] == "user" and "toolResult" in last_two[0]["content"][0]
    assert last_two[1] == {"role": "user", "content": [{"text": WORDS}]}
    assert REPEATED not in caplog.text


def test_blank_words_with_an_await_caller_result_only_record_it() -> None:
    """Blank words are no words: the result is recorded, the model is never asked about it."""
    d = direct([say(REPEATED)], pending_tool=PAY_TOOL, pending_result=payment_env())
    chunks, _ = run(collect(d, "   "))
    assert "".join(chunks).strip() == ""
    assert result_calls(d.agent) == []
    assert not [e for e in d.events() if e["type"] == "ai_turn" and e.get("text")]


def test_await_caller_without_words_speaks_nothing_and_records_the_result(
        caplog: pytest.LogCaptureFixture) -> None:
    d = direct([say(REPEATED)], policy_doc=masked_policy_doc(after_ms=50),
               pending_tool=PAY_TOOL, pending_result=payment_env())
    with caplog.at_level(logging.INFO, logger="tac_bridge.bridge"):
        chunks, _ = run(collect(d, ""))
    assert "".join(chunks).strip() == ""
    assert len(d.agent.calls) == 1 and d.agent.calls[0]["tool_results"][0].tool_use_id == "tu_pay"
    assert not [e for e in d.events() if e["type"] == "ai_turn" and e.get("text")]
    assert REPEATED not in caplog.text
    assert result_calls(d.agent) == []  # no model call for the result
    history = strands_history(d.agent)
    assert history[-1]["role"] == "user"
    assert any("toolResult" in b and b["toolResult"]["toolUseId"] == "tu_pay"
               for b in history[-1]["content"])
    from tac_bridge.agent_client_strands import pending_interrupts

    built = run(d.agent.client._built(CALL_SID, "cached"))  # noqa: SLF001
    assert pending_interrupts(built.agent) == {}


def test_the_words_with_an_await_caller_result_are_answered() -> None:
    rig = make_rig(script=[say("<silent/>"), say("Sure.")])
    write_call_state(rig.store, pending_tool=PAY_TOOL, pending_result=payment_env())
    frames = prompt_once(rig, "thanks")
    assert spoken(frames) == "Sure."


def test_a_retry_after_a_failed_words_call_does_not_resend_the_recorded_result() -> None:
    """Strands: the result and the words were recorded, then the model call failed; the bridge's
    retry sends the same turn again and the client continues from history (no unknown tool use)."""
    down = AgentError("model down", retryable=True)
    # Each try of the combined turn skips a result step (never asked), then answers the words: the
    # result, the failed words call; the result again, the retried words call.
    rig = make_rig(script=[say(REPEATED), down, say(REPEATED), say("Connecting you.")])
    write_call_state(rig.store, pending_tool=PAY_TOOL, pending_result=payment_env())
    frames = prompt_once(rig, WORDS)
    heard = spoken(frames)
    assert heard.endswith("Connecting you.") and REPEATED not in heard
    assert end_frame(frames) is None  # retried and answered, not escalated
    history = strands_history(rig.agent)
    words = [m for m in history if m["content"] == [{"text": WORDS}]]
    assert len(words) == 1  # the words were not added twice
    results = [b for m in history for b in m["content"]
               if "toolResult" in b and b["toolResult"]["toolUseId"] == "tu_pay"]
    assert len(results) == 1


# --------------------------------------------------------------------------- property


def _envelope(next_: str, utterance: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(pending_tool, pending_result)`` for each ``next`` the front door sends."""
    if next_ == "await_caller":
        return PAY_TOOL, payment_env()
    auth = {"tool_use_id": "tu_auth", "name": "authenticate", "args": {"tool": "list_recent_transactions"}}
    if next_ == "retry_original_tool":
        return auth, ResultEnvelope(status="verified", tool="authenticate", spoken=[], data={"level": 2},
                                    next="retry_original_tool", forTool="list_recent_transactions").to_dict()
    if next_ == "offer_alternative":
        return auth, ResultEnvelope(status="not_verified", tool="authenticate",
                                    spoken=[Spoken("stepup.failed", "I couldn't verify you.")], data={},
                                    next="offer_alternative").to_dict()
    return PAY_TOOL, ResultEnvelope(status="interrupted", tool="make_payment", spoken=[], data={},
                                    next="handle_utterance", callerUtterance=utterance,
                                    completed=False).to_dict()


words_st = st.text(alphabet=string.ascii_letters + " ", min_size=1, max_size=16).filter(str.strip)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(next_=st.sampled_from(["await_caller", "retry_original_tool", "offer_alternative", "handle_utterance"]),
       words=st.one_of(st.none(), words_st), reply=words_st, answer=words_st)
def test_text_for_an_await_caller_result_is_never_spoken(next_: str, words: str | None, reply: str,
                                                          answer: str) -> None:
    """The model's text for a result sent with ``next`` = ``await_caller``
    (the front door's own, or the one recorded when caller words came with it) never reaches the caller,
    and the words get the answer. Any other result is answered and spoken exactly as before."""
    result_text, answer_text = f"RESULT {reply}", f"ANSWER {answer}"
    pending_tool, env = _envelope(next_, words or "what are your hours")
    d = direct([say(result_text), say(answer_text)], pending_tool=pending_tool, pending_result=env)
    chunks, _ = run(collect(d, words or ""))
    heard = "".join(chunks)
    sent = d.agent.calls[0]["tool_results"][0].content
    caller_words = bool(words) or next_ == "handle_utterance"
    assert (sent["next"] == "await_caller") == (next_ == "await_caller" or caller_words)
    journaled = " ".join(e.get("text") or "" for e in d.events() if e["type"] == "ai_turn")
    if sent["next"] == "await_caller":
        assert result_text not in heard and result_text not in journaled
        assert heard == (answer_text if caller_words else "")
    else:  # unchanged: one hop, its text spoken and journaled
        assert sent["next"] == next_ and len(d.agent.calls) == 1
        assert heard == result_text and result_text in journaled
