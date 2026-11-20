"""Segment-traits properties over the real bridge, real TAC,
``FakeTwilioMemory`` and the real ``StrandsAgentClient`` playing scripted model replies.

* Over generated sequences of connects (with or without a welcome), turns and hand-offs (a
  ``make_payment`` hand-off, then a re-entry with its result): every request the model gets is
  valid for Converse (starts with a user message, never two assistant messages in a row, every
  ``toolUse`` followed directly by its ``toolResult``), and each connect's welcome is in the
  history exactly once, before the caller's first words of that connect: as the assistant message
  of the synthetic pair when no result was pending, else in the result's envelope. No welcome,
  nothing added. (Two user messages in a row, a ``toolResult`` then the caller's words, is the
  existing shape, which Converse accepts.)
* Over generated ``welcome_offer`` values, trait states, verification and language: the block
  is ``agent.web_activity`` exactly when the offer was made and ``WebActivity`` kept,
  ``agent.web_activity_context`` when not made and kept, none otherwise; ``WebActivity`` is in the
  recall exactly when verified and kept; before verification neither.
"""

from __future__ import annotations

import string
from datetime import UTC, datetime, timedelta
from typing import Any

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

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
    front_door_update,
    load_json,
    make_rig,
    rearm_nonce,
    run,
    say,
    tool,
    write_call_state,
)

CATALOG = PromptCatalog.from_dict(load_json("prompt_catalog.json"))
STARTED = CATALOG.text("agent.call_started", "en-US")
#: Generous: a cold first example (the first agent build in the process) can take seconds.
PROMPT_TIMEOUT_S = 20.0
PAY_ARGS = {"from_account": "checking", "to": "red", "amount": "250.00"}


def seal(text: str | None) -> Any:
    return last_spoken.seal(SEALER, text, call_sid=CALL_SID, now=NOW) if text else None


def result_env() -> dict[str, Any]:
    return ResultEnvelope(
        status="completed", tool="make_payment",
        spoken=[Spoken("confirm.make_payment.done", "Your payment is done."),
                Spoken("common.anything_else", "Is there anything else I can help you with?")],
        data={"executed": True, "confirmation": "123"}, next="await_caller").to_dict()


def blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    return list(message.get("content") or ())


def flat_text(message: dict[str, Any]) -> str:
    out = []
    for b in blocks(message):
        if "text" in b:
            out.append(b["text"])
        elif "toolResult" in b:
            out.extend(c.get("text", "") for c in b["toolResult"].get("content") or ())
    return "\n".join(out)


def assert_valid(messages: list[dict[str, Any]], *, pending_tail: bool = False) -> None:
    """Converse's rules (see the module docs). ``pending_tail``: the stored history may end with a
    ``toolUse`` still waiting for its result (a hand-off in progress); a model request never does."""
    assert messages and messages[0]["role"] == "user"
    for i, m in enumerate(messages):
        if i and m["role"] == "assistant":
            assert messages[i - 1]["role"] == "user", "two assistant messages in a row"
        uses = [b["toolUse"]["toolUseId"] for b in blocks(m) if "toolUse" in b]
        if uses and pending_tail and i == len(messages) - 1:
            continue
        if uses:
            assert m["role"] == "assistant" and i + 1 < len(messages), "toolUse without its toolResult"
            answered = {b["toolResult"]["toolUseId"] for b in blocks(messages[i + 1]) if "toolResult" in b}
            assert set(uses) <= answered, "toolUse not followed directly by its toolResult"


SEGMENT = st.fixed_dictionaries({
    "welcome": st.booleans(),
    "turns": st.integers(min_value=1, max_value=3),
    "handoff": st.booleans(),
})


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(segments=st.lists(SEGMENT, min_size=1, max_size=3))
@example(segments=[{"welcome": True, "turns": 2, "handoff": True},  # a re-entry with a welcome,
                   {"welcome": True, "turns": 2, "handoff": True},  # twice in a row,
                   {"welcome": False, "turns": 1, "handoff": False}])  # then one without
@example(segments=[{"welcome": False, "turns": 1, "handoff": True}, {"welcome": True, "turns": 1, "handoff": False}])
def test_each_welcome_once_before_the_reply_and_a_valid_history(segments: list[dict[str, Any]]) -> None:
    rig = make_rig()
    agent = rig.agent
    plan: list[tuple[str | None, bool, list[str]]] = []  # (welcome, pending result, the caller's words)
    pending = False
    for i, seg in enumerate(segments):
        welcome = f"Welcome number {i}, how can I help?" if seg["welcome"] else None
        words = [f"caller words {i}.{t}" for t in range(seg["turns"])]
        if i == 0:
            write_call_state(rig.store, last_spoken=seal(welcome))
        else:
            changes: dict[str, Any] = {"last_spoken": seal(welcome)}
            if pending:
                changes["pending_result"] = result_env()
            front_door_update(rig.store, **changes)
            rearm_nonce(rig.store)
        for t in range(seg["turns"]):
            if pending and t == 0:
                agent.script.append(say("(recorded, never asked)"))  # The result's own step
            if seg["handoff"] and t == seg["turns"] - 1:
                agent.script.append(tool("make_payment", f"tu_pay_{i}", **PAY_ARGS))
            else:
                agent.script.append(say(f"Agent reply {i}.{t}."))
        plan.append((welcome, pending, words))
        hands_off = seg["handoff"]

        async def segment(words: list[str] = words) -> None:
            sim = rig.sim()
            await sim.connect()
            await sim.idle()
            for w in words:
                await sim.prompt(w, timeout=PROMPT_TIMEOUT_S)
            await sim.hangup()

        run(segment())
        pending = hands_off

    requests = [c.messages for c in agent.model.calls]
    for messages in requests:
        assert_valid(messages)
    built = run(agent.client._built(CALL_SID, "cached"))  # noqa: SLF001 - test seam
    final = list(built.agent.messages)
    assert_valid(final, pending_tail=pending)
    texts = [flat_text(m) for m in final]
    exchanges = 0
    for welcome, was_pending, words in plan:
        first_words = next(i for i, t in enumerate(texts) if words[0] in t)
        if welcome is None:
            continue
        where = [i for i, t in enumerate(texts) if welcome in t]
        assert len(where) == 1, (welcome, where)  # exactly once
        assert where[0] <= first_words  # before the caller's reply (or in the same result message)
        if was_pending:
            assert final[where[0]]["role"] == "user" and any("toolResult" in b for b in blocks(final[where[0]]))
        else:
            exchanges += 1
            assert final[where[0]]["role"] == "assistant" and texts[where[0]] == welcome
            assert texts[where[0] - 1] == STARTED and final[where[0] - 1]["role"] == "user"
        # the first model request of the connect saw the welcome before the words
        (req,) = [r for r in requests if flat_text(r[-1]).endswith(words[0])]
        req_texts = [flat_text(m) for m in req]
        assert sum(welcome in t for t in req_texts) == 1
    assert sum(t == STARTED for t in texts) == exchanges  # no welcome, nothing added
    assert len(agent.exchanges) == exchanges




FRESH = (datetime.fromtimestamp(NOW, UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
STALE = (datetime.fromtimestamp(NOW, UTC) - timedelta(hours=72)).isoformat().replace("+00:00", "Z")
PRODUCT = st.text(alphabet=string.ascii_letters, min_size=6, max_size=20).map(lambda s: "Pq" + s)
TRAITS = st.sampled_from(["kept", "stale", "absent", "other_group"])


def _groups(state: str, product: str) -> dict[str, Any]:
    entry = {"lastViewedProduct": product, "lastViewedAt": FRESH, "viewCount": 2}
    if state == "kept":
        return {GROUP: entry}
    if state == "stale":
        return {GROUP: {**entry, "lastViewedAt": STALE}}
    if state == "other_group":
        return {"Marketing": entry}
    return {}


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(welcome_offer=st.sampled_from([None, WELCOME_OFFER_MADE, "unexpected"]), traits=TRAITS,
       verified=st.booleans(), lang=st.sampled_from(["en-US", "es-US"]), product=PRODUCT)
def test_the_block_and_the_recall_follow_the_offer_and_the_traits(
        welcome_offer: str | None, traits: str, verified: bool, lang: str, product: str) -> None:
    rig = make_rig(memory="tac", script=[say("Hello.")])
    memory = rig.tac.conversation_memory_client
    memory.trait_groups[VERIFIED_PROFILE] = _groups(traits, product)
    memory.trait_groups[ANI_PROFILE] = _groups(traits, product)
    write_call_state(rig.store, verified=verified, language=lang, welcome_offer=welcome_offer,
                     **({} if verified else {"token": None}))

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("Hi", timeout=PROMPT_TIMEOUT_S)
        await sim.hangup()

    run(go())
    sp = rig.agent.calls[0]["system_prompt"]
    kept = verified and traits == "kept"
    made, context = CATALOG.text("agent.web_activity", lang), CATALOG.text("agent.web_activity_context", lang)
    assert (product in sp) is kept  # WebActivity in the recall whether or not an offer was made
    assert (made in sp) is (kept and welcome_offer == WELCOME_OFFER_MADE)
    assert (context in sp) is (kept and welcome_offer != WELCOME_OFFER_MADE)
    if not verified:
        assert memory.recall_ids == [] and GROUP not in sp
