"""Bridge security: PII masking of AI turns and the summary, relay nonce
reconnects, idempotency keys on money movement."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any


from tests.component.test_bridge_fakes import (
    FakeBackend,
    end_frame,
    inline_money_policy,
    make_rig,
    rearm_nonce,
    run,
    say,
    tool,
    write_call_state,
)

CARD = "4111 1111 1111 1111"
SSN = "123-45-6789"


def _one_prompt(rig, text: str) -> list[dict[str, Any]]:
    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt(text)
        await sim.hangup()
        return frames

    return run(go())


def test_ai_turn_text_is_pii_masked_in_the_journal() -> None:
    rig = make_rig(script=[say(f"I heard card {CARD} and SSN {SSN}. Let me help.")])
    write_call_state(rig.store, language="es-US")
    _one_prompt(rig, "hola")
    (turn,) = [e for e in rig.events() if e["type"] == "ai_turn"]
    assert "4111" not in turn["text"] and "6789" not in turn["text"]
    assert "[CREDIT_DEBIT_NUMBER]" in turn["text"] and "[SSN]" in turn["text"]
    assert rig.pii.languages == ["es"]  # the call's language, from call state


def test_ai_turn_text_is_withheld_when_the_check_is_down() -> None:
    rig = make_rig(script=[say(f"Your card {CARD} is noted.")])
    rig.pii.fail = True
    write_call_state(rig.store)
    frames = _one_prompt(rig, "hi")
    assert "Your card" in "".join(f.get("token", "") for f in frames)  # the caller still hears it
    (turn,) = [e for e in rig.events() if e["type"] == "ai_turn"]
    assert turn["text"] == "[WITHHELD]"


def test_escalation_summary_is_masked_before_call_state_and_journal() -> None:
    summary = f"Caller gave card {CARD} and SSN {SSN}; wants a human."
    rig = make_rig(script=[tool("escalate", "tu_esc", reason="caller_requested_human", summary=summary)])
    write_call_state(rig.store)
    end = end_frame(_one_prompt(rig, "agent please"))
    assert end is not None and end["data"]["escalationReason"] == "caller_requested_human"
    stored = rig.call()["escalation"]["summary"]
    assert "4111" not in stored and SSN not in stored and stored.startswith("Caller gave card [CREDIT_DEBIT_NUMBER]")
    blob = json.dumps(rig.events())
    assert "4111" not in blob and SSN not in blob


def test_summary_withheld_when_the_check_is_down_and_escalation_proceeds() -> None:
    rig = make_rig(script=[tool("escalate", "tu_esc", reason="caller_requested_human",
                                summary=f"card {CARD}")])
    rig.pii.fail = True
    write_call_state(rig.store, language="es-US")
    end = end_frame(_one_prompt(rig, "agente"))
    assert end is not None and end["data"]["reason"] == "escalate"
    assert rig.call()["escalation"]["summary"] == (
        "Resumen retenido: la verificación de datos personales no está disponible.")
    assert rig.call()["pending_tool"]["name"] == "escalate"


def test_money_tools_carry_the_tool_use_id_as_idempotency_key() -> None:
    ok = {"tool": "make_payment", "data": {"confirmation": "11112222"}}
    backend = FakeBackend({"make_payment": [ok], "list_recent_transactions": [{"tool": "x", "data": {}}]})
    rig = make_rig(backend=backend, policy_doc=inline_money_policy(), script=[
        tool("make_payment", "tu_pay_42", from_account="checking", to="red", amount="5.00"),
        tool("list_recent_transactions", "tu_tx", account="checking"),
        say("Done."),
    ])
    write_call_state(rig.store, level=3)
    _one_prompt(rig, "pay five dollars")
    pay, txns = backend.calls
    assert pay["idempotency_key"] == "tu_pay_42" and "idempotency_key" not in pay["args"]
    assert "idempotency_key" not in txns  # reads carry none


def test_failed_money_call_is_retried_once_with_the_same_key() -> None:
    ok = {"tool": "make_payment", "data": {"confirmation": "11112222"}}
    backend = FakeBackend({"make_payment": [TimeoutError("lost reply"), ok]})
    rig = make_rig(backend=backend, policy_doc=inline_money_policy(), script=[
        tool("make_payment", "tu_pay_7", from_account="checking", to="red", amount="5.00"),
        say("Done."),
    ])
    write_call_state(rig.store, level=3)
    _one_prompt(rig, "pay")
    assert [c["idempotency_key"] for c in backend.calls] == ["tu_pay_7", "tu_pay_7"]
    (call,) = [c for c in rig.agent.calls if c["tool_results"]]
    assert call["tool_results"][0].content == ok


def test_each_socket_needs_a_fresh_nonce() -> None:
    rig = make_rig(script=[say("One."), say("Two.")])
    write_call_state(rig.store)

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("first")
        await sim.hangup()
        replay = rig.sim()  # the same URL again, after the call moved on
        with contextlib.suppress(AssertionError):  # no new session appears
            await replay.connect()
        await asyncio.wait_for(replay.task, 3)
        assert replay.socket.closed and not replay.peek.verified
        rearm_nonce(rig.store)  # the front door's reconnect TwiML
        again = rig.sim()
        await again.connect()
        await again.prompt("second")
        await again.hangup()

    run(go())
    assert [c["user_text"] for c in rig.agent.calls] == ["first", "second"]
    assert rig.call()["ws_nonce_hash"] is None
