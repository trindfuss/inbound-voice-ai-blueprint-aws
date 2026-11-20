"""Segment-traits: the web-activity blocks, the recall and the welcome in history.

Real TAC over ``FakeTwilioMemory``, the ConversationRelay simulator, and the rigs' agent: the real
``StrandsAgentClient`` playing the scripts (``tests.strands_parity``), so the history checked is
what the model is sent.

* after verification ``WebActivity`` stays in the recall whether or not the welcome made the
  offer; the block is ``agent.web_activity`` exactly when ``welcome_offer == "made"``,
  ``agent.web_activity_context`` when unset and ``WebActivity`` is kept, and absent otherwise (the
  prompt then byte-identical to the one without an offer); before verification neither.
* Observations and summaries in the recall reach the prompt unchanged after verification,
  never before.
* the session's welcome reaches the history once, before the caller's reply: the synthetic
  pair (``agent.call_started``, welcome) when no tool result is pending, else a ``welcome`` field in
  that result's envelope. No welcome, nothing added.

The property tests over generated sequences are in ``tests/property/test_welcome_history_props.py``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from tac.models.memory import MemoryRetrievalResponse, ObservationInfo, SummaryInfo

from owlbank import last_spoken
from owlbank.callstate import WELCOME_OFFER_MADE
from owlbank.catalog import PromptCatalog
from owlbank.envelope import ResultEnvelope, Spoken
from owlbank.web_activity import GROUP
from tests.component.test_bridge_fakes import (
    ANI_PROFILE,
    CALL_SID,
    NOW,
    SEALER,
    VERIFIED_PROFILE,
    load_json,
    make_rig,
    run,
    say,
    write_call_state,
)
from tests.logutil import lines, text_of

BRIDGE_LOG = "tac_bridge.bridge"
CATALOG = PromptCatalog.from_dict(load_json("prompt_catalog.json"))
FRESH = (datetime.fromtimestamp(NOW, UTC) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
WELCOME = "Hi Thomas, you're verified. Are you calling about the HELOC you looked at on our website?"
PAY_TOOL = {"tool_use_id": "tu_pay", "name": "make_payment",
            "args": {"from_account": "checking", "to": "red", "amount": "250.00"}}
SUMMARY = "WhatsApp chat: the customer asked about a duplicate charge; the chat said a team would follow up."
OBSERVATION = "Customer has an open question about a duplicate charge."
ANI_SUMMARY = "ANI HOUSEHOLD SUMMARY"


def offered(lang: str = "en-US") -> str:
    return CATALOG.text("agent.web_activity", lang)


def context_block(lang: str = "en-US") -> str:
    return CATALOG.text("agent.web_activity_context", lang)


def activity(product: str = "HELOC") -> dict[str, Any]:
    return {GROUP: {"lastViewedProduct": product, "lastViewedAt": FRESH, "viewCount": 3}}


def seal(text: str) -> Any:
    return last_spoken.seal(SEALER, text, call_sid=CALL_SID, now=NOW)


def recall(summary: str, observation: str) -> MemoryRetrievalResponse:
    return MemoryRetrievalResponse(
        observations=[ObservationInfo(content=observation, source="owlbank-demo", id="obs_9",
                                      createdAt="2026-10-28T00:00:00Z", updatedAt="2026-10-28T00:00:00Z")],
        summaries=[SummaryInfo(content=summary, conversationId="conv_demo", id="sum_9",
                               createdAt="2026-10-28T00:00:00Z", updatedAt="2026-10-28T00:00:00Z",
                               source="owlbank-demo")])


def connect_and_say(rig: Any, *prompts: str) -> list[Any]:
    async def go() -> list[Any]:
        sim = rig.sim()
        await sim.connect()
        await sim.idle()
        frames = []
        for p in prompts:
            frames += await sim.prompt(p)
        await sim.hangup()
        return frames

    return run(go())


def call(*, groups: dict[str, Any] | None = None, verified: bool = True, language: str = "en-US",
         welcome_offer: str | None = None, welcome: str | None = None, rig: Any = None,
         **fields: Any) -> tuple[Any, str]:
    rig = rig or make_rig(memory="tac", script=[say("Hello.")])
    if groups is not None:
        rig.tac.conversation_memory_client.trait_groups[VERIFIED_PROFILE] = groups
        rig.tac.conversation_memory_client.trait_groups[ANI_PROFILE] = groups
    write_call_state(rig.store, verified=verified, language=language, welcome_offer=welcome_offer,
                     last_spoken=seal(welcome) if welcome else None,
                     **({} if verified else {"token": None}), **fields)
    connect_and_say(rig, "Hi")
    return rig, rig.agent.calls[-1]["system_prompt"]


def memory_section(prompt: str) -> str:
    start = prompt.index('<untrusted_data source="memory">')
    return prompt[start:prompt.index("</untrusted_data>", start)]


def history(rig: Any) -> list[dict[str, Any]]:
    built = run(rig.agent.client._built(CALL_SID, "cached"))  # noqa: SLF001 - test seam
    return [{"role": m["role"], "content": m["content"]} for m in built.agent.messages]


def texts(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """``(role, text)`` per text block, and each toolResult's text as ``(role, "result:" + text)``."""
    out = []
    for m in messages:
        for b in m["content"]:
            if "text" in b:
                out.append((m["role"], b["text"]))
            elif "toolResult" in b:
                out.append((m["role"], "result:" + b["toolResult"]["content"][0]["text"]))
    return out




@pytest.mark.parametrize("lang", ["en-US", "es-US"])
def test_an_offer_gives_the_offered_block_and_keeps_web_activity_in_the_recall(lang: str) -> None:
    _, sp = call(groups=activity(), welcome_offer=WELCOME_OFFER_MADE, language=lang)
    assert f"- {GROUP}: " in memory_section(sp) and "'lastViewedProduct': 'HELOC'" in memory_section(sp)
    assert offered(lang) in sp and context_block(lang) not in sp
    assert sp.index(offered(lang)) > sp.index("</untrusted_data>")


@pytest.mark.parametrize("lang", ["en-US", "es-US"])
def test_no_offer_gives_the_context_block_and_keeps_web_activity_in_the_recall(lang: str) -> None:
    _, sp = call(groups=activity(), language=lang)
    assert f"- {GROUP}: " in memory_section(sp) and "'lastViewedProduct': 'HELOC'" in memory_section(sp)
    assert context_block(lang) in sp and offered(lang) not in sp


def test_an_offer_without_kept_web_activity_gives_no_block_and_the_same_prompt() -> None:
    """It holds with the offer flag set: no ``WebActivity`` kept, the prompt is byte-identical."""
    _, with_flag = call(groups={}, welcome_offer=WELCOME_OFFER_MADE)
    _, without = call(groups={})
    assert with_flag == without
    assert offered() not in with_flag and context_block() not in with_flag and GROUP not in with_flag


@pytest.mark.parametrize("welcome_offer", [None, WELCOME_OFFER_MADE])
def test_before_verification_neither_the_recall_nor_a_block(welcome_offer: str | None) -> None:
    rig, sp = call(groups=activity(), verified=False, welcome_offer=welcome_offer)
    assert "HELOC" not in sp and GROUP not in sp
    assert offered() not in sp and context_block() not in sp
    assert rig.tac.conversation_memory_client.recall_ids == []


def test_the_offer_flag_is_read_per_session() -> None:
    """A later session whose welcome made no offer (a re-entry) gets the context block."""
    rig = make_rig(memory="tac", script=[say("Hello."), say("Hello again.")])
    rig.tac.conversation_memory_client.trait_groups[VERIFIED_PROFILE] = activity()
    write_call_state(rig.store, welcome_offer=WELCOME_OFFER_MADE)
    connect_and_say(rig, "Hi")
    from tests.component.test_bridge_fakes import front_door_update, rearm_nonce

    front_door_update(rig.store, welcome_offer=None)
    rearm_nonce(rig.store)
    connect_and_say(rig, "Hi again")
    first, second = rig.agent.calls[0]["system_prompt"], rig.agent.calls[1]["system_prompt"]
    assert offered() in first and context_block() not in first
    assert context_block() in second and offered() not in second


# --------------------------------------------------------------------------- the recall


def test_observations_and_summaries_reach_the_prompt_unchanged_after_verification() -> None:
    rig = make_rig(memory="tac", script=[say("Hello.")])
    memory = rig.tac.conversation_memory_client
    memory.recalls[VERIFIED_PROFILE] = recall(SUMMARY, OBSERVATION)
    _, sp = call(rig=rig, groups=activity())
    section = memory_section(sp)
    assert "## Key Observations" in section and OBSERVATION in section
    assert "## Past Conversation Summaries" in section and SUMMARY in section
    assert "owlbank-demo" not in section and "conv_demo" not in section  # content only


@pytest.mark.parametrize("welcome_offer", [None, WELCOME_OFFER_MADE])
def test_observations_and_summaries_never_reach_the_prompt_before_verification(welcome_offer: str | None) -> None:
    rig = make_rig(memory="tac", script=[say("Hello.")])
    memory = rig.tac.conversation_memory_client
    memory.recalls[VERIFIED_PROFILE] = recall(SUMMARY, OBSERVATION)
    memory.recalls[ANI_PROFILE] = recall(ANI_SUMMARY, "ANI OBSERVATION")
    rig, sp = call(rig=rig, verified=False, welcome_offer=welcome_offer)
    for value in (SUMMARY, OBSERVATION, ANI_SUMMARY, "ANI OBSERVATION"):
        assert value not in sp and value not in rig.agent.model_inputs_text()
    assert memory.recall_ids == []




@pytest.mark.parametrize("lang", ["en-US", "es-US"])
def test_the_first_turn_records_the_synthetic_pair_before_the_callers_words(
        lang: str, caplog: pytest.LogCaptureFixture) -> None:
    rig = make_rig(script=[say("Great, let's talk about it."), say("Sure.")])
    write_call_state(rig.store, language=lang, last_spoken=seal(WELCOME), welcome_offer=WELCOME_OFFER_MADE)
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        connect_and_say(rig, "Yes", "And the rate?")
    started = CATALOG.text("agent.call_started", lang)
    assert rig.agent.exchanges == [(CALL_SID, started, WELCOME)]  # once, though two turns ran
    first_request = texts(rig.agent.model.calls[0].messages)
    assert first_request == [("user", started), ("assistant", WELCOME), ("user", "Yes")]
    h = texts(history(rig))
    assert [t for _, t in h].count(WELCOME) == 1 and h[0] == ("user", started)
    (rec,) = [e for e in lines(caplog.records, BRIDGE_LOG) if e["event"] == "welcome_recorded"]
    assert rec["path"] == "exchange" and rec["chars"] == len(WELCOME)
    assert WELCOME not in text_of(caplog.records) and "HELOC" not in text_of(caplog.records)


def test_no_welcome_adds_nothing() -> None:
    rig = make_rig(script=[say("Hello.")])
    write_call_state(rig.store)
    connect_and_say(rig, "Hi")
    assert rig.agent.exchanges == []
    assert texts(history(rig)) == [("user", "Hi"), ("assistant", "Hello.")]


def test_an_unsealable_welcome_adds_nothing() -> None:
    rig = make_rig(script=[say("Hello.")])
    other = last_spoken.seal(SEALER, WELCOME, call_sid="CA" + "9" * 32, now=NOW)  # another call's record
    write_call_state(rig.store, last_spoken=other)
    connect_and_say(rig, "Hi")
    assert rig.agent.exchanges == [] and WELCOME not in json.dumps(history(rig))


def test_each_connect_records_its_own_welcome() -> None:
    from tests.component.test_bridge_fakes import front_door_update, rearm_nonce

    rig = make_rig(script=[say("One."), say("Two.")])
    write_call_state(rig.store, last_spoken=seal("Welcome one."))
    connect_and_say(rig, "first")
    front_door_update(rig.store, last_spoken=seal("Welcome two."))
    rearm_nonce(rig.store)
    connect_and_say(rig, "second")
    started = CATALOG.text("agent.call_started", "en-US")
    assert texts(history(rig)) == [("user", started), ("assistant", "Welcome one."), ("user", "first"),
                                   ("assistant", "One."), ("user", started), ("assistant", "Welcome two."),
                                   ("user", "second"), ("assistant", "Two.")]


def payment_env() -> dict[str, Any]:
    return ResultEnvelope(
        status="completed", tool="make_payment",
        spoken=[Spoken("confirm.make_payment.done", "Your payment is done."),
                Spoken("common.anything_else", "Is there anything else I can help you with?")],
        data={"executed": True, "confirmation": "123"}, next="await_caller").to_dict()


def test_a_pending_result_carries_the_welcome_in_its_envelope(caplog: pytest.LogCaptureFixture) -> None:
    rig = make_rig(script=[say("unused: the result is recorded"), say("Anything else?")])
    reentry = "Thanks for waiting. Your payment is done."
    write_call_state(rig.store, pending_tool=PAY_TOOL, pending_result=payment_env(), last_spoken=seal(reentry))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        connect_and_say(rig, "thanks")
    assert rig.agent.exchanges == []
    (result,) = [r for c in rig.agent.calls for r in c["tool_results"]]
    assert result.content["welcome"] == reentry
    h = history(rig)
    flat = texts(h)
    assert sum(reentry in t for _, t in flat) == 1
    use_at = next(i for i, m in enumerate(h) if any("toolUse" in b for b in m["content"]))
    assert any("toolResult" in b for b in h[use_at + 1]["content"])  # toolUse, then its toolResult
    words_at = next(i for i, (r, t) in enumerate(flat) if t == "thanks")
    assert next(i for i, (_, t) in enumerate(flat) if reentry in t) < words_at
    (rec,) = [e for e in lines(caplog.records, BRIDGE_LOG) if e["event"] == "welcome_recorded"]
    assert rec["path"] == "envelope"


def test_a_catalog_without_call_started_records_nothing_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    doc = load_json("prompt_catalog.json")
    doc = {**doc, "entries": [e for e in doc["entries"] if e.get("key") != "agent.call_started"]}
    rig = make_rig(script=[say("Hello.")])
    rig.bridge._catalog = PromptCatalog.from_dict(doc)
    write_call_state(rig.store, last_spoken=seal(WELCOME))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        connect_and_say(rig, "Hi")
    assert rig.agent.exchanges == []
    assert "call_started_missing" in text_of(caplog.records)
    assert texts(history(rig)) == [("user", "Hi"), ("assistant", "Hello.")]


@pytest.mark.protocol_fake
def test_a_failed_record_is_retried_once_with_the_hop() -> None:
    """A failed ``record_exchange`` fails the hop retryably: the retry records it, then the turn runs."""
    from tac_bridge.agent_client import AgentError

    rig = make_rig(script=[say("Hello.")])
    real = rig.agent.record_exchange
    attempts: list[int] = []

    async def flaky(*a: Any) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise AgentError("history write failed", retryable=True, code="history_write_failed")
        await real(*a)

    rig.agent.record_exchange = flaky  # type: ignore[method-assign]
    write_call_state(rig.store, last_spoken=seal(WELCOME))
    connect_and_say(rig, "Hi")
    assert len(attempts) == 2 and len(rig.agent.exchanges) == 1
    assert [c["user_text"] for c in rig.agent.calls] == ["Hi"]


def _started() -> str:
    return CATALOG.text("agent.call_started", "en-US")


def test_a_turn_that_fails_before_the_record_leaves_the_welcome_for_the_next_turn() -> None:
    """The first turn fails while its system prompt is built (the fallback too), so its agent
    hop never runs; the welcome is not lost: the next turn records it, once."""
    rig = make_rig(script=[say("Hello.")])
    real, fallback = rig.bridge._system_prompt, rig.bridge._fallback_system_prompt  # noqa: SLF001
    builds: list[int] = []

    async def failing(*a: Any) -> Any:
        builds.append(1)
        if len(builds) == 1:
            raise RuntimeError("policy table unavailable")
        return await real(*a)

    def failing_fallback(*a: Any) -> Any:
        if len(builds) == 1:
            raise RuntimeError("catalog unavailable")
        return fallback(*a)

    rig.bridge._system_prompt = failing  # type: ignore[method-assign]  # noqa: SLF001
    rig.bridge._fallback_system_prompt = failing_fallback  # type: ignore[method-assign]  # noqa: SLF001
    write_call_state(rig.store, last_spoken=seal(WELCOME))

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.idle()
        await sim.prompt_nowait("Hi")
        await sim.idle()
        assert rig.agent.exchanges == [] and rig.agent.calls == []
        await sim.prompt("Hello?")
        await sim.hangup()

    run(go())
    assert rig.agent.exchanges == [(CALL_SID, _started(), WELCOME)]
    assert [c["user_text"] for c in rig.agent.calls] == ["Hello?"]
    assert [t for _, t in texts(history(rig))].count(WELCOME) == 1


@pytest.mark.protocol_fake
def test_a_record_that_exhausts_its_retries_leaves_the_welcome_for_the_next_turn() -> None:
    """``record_exchange`` fails on the hop and on its retry, so the first turn fails; the
    welcome stays the session's: the next turn records it, once."""
    from tac_bridge.agent_client import AgentError

    rig = make_rig(script=[say("Hello.")])
    real = rig.agent.record_exchange
    attempts: list[int] = []

    async def failing(*a: Any) -> None:
        attempts.append(1)
        if len(attempts) <= 2:
            raise AgentError("history write failed", retryable=True, code="history_write_failed")
        await real(*a)

    rig.agent.record_exchange = failing  # type: ignore[method-assign]
    write_call_state(rig.store, last_spoken=seal(WELCOME))

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.idle()
        await sim.prompt("Hi")
        assert len(attempts) == 2 and rig.agent.exchanges == [] and rig.agent.calls == []
        await sim.prompt("Hello?")
        await sim.hangup()

    run(go())
    assert len(attempts) == 3
    assert rig.agent.exchanges == [(CALL_SID, _started(), WELCOME)]  # once (the fake agent's record)
    assert [c["user_text"] for c in rig.agent.calls] == ["Hello?"]
