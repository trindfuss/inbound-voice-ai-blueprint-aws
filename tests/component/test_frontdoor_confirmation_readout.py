"""A scripted read-out on the return to the AI is ConversationRelay's
interruptible ``welcomeGreeting``, not ``<Say>`` verbs before ``<Connect>``.

Signed fake webhooks into the real front door (``test_frontdoor_support.build``). The bridge
side (a caller prompt arriving first on an ``await_caller`` session is the caller's turn and the
agent gets the envelope) is ``tests/component/test_bridge_flows.py::
test_pending_result_is_sent_on_first_prompt_with_caller_utterance``; the barge-in variant (an
``interrupt`` of the greeting before that prompt) is covered here.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from decimal import Decimal
from typing import Any

import pytest

from lambdas.frontdoor.app import BARGE_IN_MODES, FrontDoorConfig
from lambdas.frontdoor.speech import digits_words, money_words
from owlbank.catalog import PromptCatalog

from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    DATA,
    TEST_FACTORS,
    THOMAS,
    Harness,
    assert_outcome,
    assert_scripted,
    build,
    fd,
    greeting,
    relay,
    says,
    sid,
)
from tests.component.test_frontdoor_caller_id_trust import READBACK, confirm_handoff, to_level3

WS = "wss://bridge.owlbank.example/ws"
AUDIO = "https://assets.example/balance-read-en.mp3"


def balance_handoff(h: Harness, call_sid: str, account: str = "checking") -> ET.Element:
    h.set_pending_tool(call_sid, "get_balance", {"account": account})
    return h.handoff(call_sid, {"reason": "tool", "tool": "get_balance",
                                "args": {"account": account}, "journal": "jr-1"})


def expected_checking_readout(h: Harness, lang: str = "en-US") -> list[str]:
    """Thomas's checking read-out, rendered from the catalog (fixture balances)."""
    cat = h.app.d.catalog
    return [
        cat.text("balance.read", lang, accountType="checking", last4=digits_words("4821", lang),
                 available=money_words(Decimal("1234.56"), lang),
                 current=money_words(Decimal("1310.06"), lang)),
        cat.text("balance.read.pending", lang, pending=money_words(Decimal("75.50"), lang)),
        cat.text("common.anything_else", lang),
    ]


# --------------------------------------------------------------------------- balance read-out


def test_balance_readout_is_the_interruptible_greeting(fd: Harness) -> None:
    fd.login_thomas(sid(1))
    root = balance_handoff(fd, sid(1))
    texts = expected_checking_readout(fd)
    assert_scripted(root, *texts, interruptible="any")  # no <Say> before <Connect>
    # The envelope and the journal still list every template spoken, with the same words.
    env = fd.pending_result(sid(1))
    assert [s.template for s in env.spoken] == ["balance.read", "balance.read.pending",
                                                "common.anything_else"]
    assert [s.text for s in env.spoken] == texts
    assert env.next == "await_caller"
    keys = [e.get("prompt_key") for e in fd.journal(sid(1)) if e["type"] == "prompt"]
    assert keys[-3:] == ["balance.read", "balance.read.pending", "common.anything_else"]


@pytest.mark.parametrize("mode", ["speech", "dtmf", "none"])
def test_interruptible_attribute_follows_the_setting(mode: str) -> None:
    h = build(config_kw={"scripted_barge_in": mode})
    h.login_thomas(sid(2))
    root = balance_handoff(h, sid(2))
    assert_scripted(root, *expected_checking_readout(h), interruptible=mode)
    # The AI's own turns keep the ordinary barge-in setting.
    assert relay(root).get("interruptible") == "speech"


def test_a_plain_reconnect_has_no_greeting_interruptible_attribute(fd: Harness) -> None:
    """Nothing scripted (say_now empty): the old path, welcome only, no attribute."""
    root = fd.login_thomas(sid(3))
    cr = relay(root)
    assert says(root) == [] and cr.get("welcomeGreeting")
    assert cr.get("welcomeGreetingInterruptible") is None


# --------------------------------------------------------------------------- config


def test_barge_in_modes_are_the_four_conversationrelay_values() -> None:
    assert set(BARGE_IN_MODES) == {"any", "speech", "dtmf", "none"}


@pytest.mark.parametrize("bad", ["", "all", "ANY", "speech,dtmf", "true"])
def test_invalid_setting_raises(bad: str) -> None:
    with pytest.raises(ValueError, match="scripted_barge_in"):
        FrontDoorConfig(relay_ws_url=WS, scripted_barge_in=bad)


def test_from_env_default_is_any() -> None:
    assert FrontDoorConfig.from_env({"OWLBANK_RELAY_WS_URL": WS}).scripted_barge_in == "any"
    assert FrontDoorConfig.from_env({"OWLBANK_RELAY_WS_URL": WS,
                                     "OWLBANK_SCRIPTED_BARGE_IN": ""}).scripted_barge_in == "any"


@pytest.mark.parametrize("value, expected", [("speech", "speech"), ("dtmf", "dtmf"), ("none", "none"),
                                             ("DTMF", "dtmf")])
def test_from_env_reads_the_setting(value: str, expected: str) -> None:
    cfg = FrontDoorConfig.from_env({"OWLBANK_RELAY_WS_URL": WS, "OWLBANK_SCRIPTED_BARGE_IN": value})
    assert cfg.scripted_barge_in == expected


def test_from_env_refuses_junk() -> None:
    with pytest.raises(ValueError, match="scripted_barge_in"):
        FrontDoorConfig.from_env({"OWLBANK_RELAY_WS_URL": WS, "OWLBANK_SCRIPTED_BARGE_IN": "junk"})


# --------------------------------------------------------------------------- pre-recorded audio


def _catalog_with_audio(key: str, lang: str, url: str) -> PromptCatalog:
    doc: dict[str, Any] = json.loads((DATA / "prompt_catalog.json").read_text())
    hits = [e for e in doc["entries"] if e["key"] == key and e["lang"] == lang]
    assert len(hits) == 1
    hits[0]["audioUrl"] = url
    return PromptCatalog.from_dict(doc)


def test_a_play_item_keeps_the_verbs_then_relay_path() -> None:
    h = build()
    h.app.d.catalog = _catalog_with_audio("balance.read", "en-US", AUDIO)
    h.login_thomas(sid(4))
    root = balance_handoff(h, sid(4))
    texts = expected_checking_readout(h)
    assert [c.tag for c in root] == ["Play", "Say", "Connect"]
    assert root.find("Play").text == AUDIO
    assert says(root) == texts[1:2]  # the pending sentence, still <Say>
    cr = relay(root)
    assert cr.get("welcomeGreeting") == texts[2]  # only the welcome
    assert cr.get("welcomeGreetingInterruptible") is None
    # The envelope still carries the words of the recorded prompt.
    assert [s.text for s in h.pending_result(sid(4)).spoken] == texts


# --------------------------------------------------------------------------- Gathers unchanged


def _assert_gather_prompt(root: ET.Element, *texts: str) -> None:
    gather = root.find("Gather")
    assert gather is not None, ET.tostring(root)
    assert [e.text for e in gather.iter("Say")] == list(texts)
    assert root.find(".//ConversationRelay") is None


def test_factor_prompts_are_still_gathers(fd: Harness) -> None:
    root = fd.start(sid(5), THOMAS)
    _assert_gather_prompt(root, "Please say or enter your five-digit ZIP code. If you can't, press pound.")
    fd.login_thomas(sid(6))
    fd.set_pending_tool(sid(6), "authenticate", {"tool": "make_payment"})
    root = fd.handoff(sid(6), {"reason": "step-up", "tool": "make_payment", "journal": "jr-2"})
    assert "Please say or enter the last four digits of your Social Security number. If you can't, press pound." in \
        [e.text for e in root.find("Gather").iter("Say")]
    assert root.find(".//ConversationRelay") is None
    root = fd.factor(sid(6), TEST_FACTORS["C-1001"]["ssn4"], follow=False)
    # the ack is spoken on its own while the code is sent, then a redirect to the prompt.
    assert [c.tag for c in root] == ["Say", "Redirect"] and root.find("Say").text == "Got it."
    root = fd.xml("factor", sid(6))  # the redirect, as Twilio follows it
    _assert_gather_prompt(root, "We just sent a six-digit code to your phone by text message. "
                                "Please say or enter it now, or press pound if you didn't get it.")


def test_confirmation_readback_is_still_a_gather(fd: Harness) -> None:
    to_level3(fd, sid(7))
    root = confirm_handoff(fd, sid(7))
    _assert_gather_prompt(root, READBACK)
    root = fd.xml("factor", sid(7))  # silence: reprompt, still a Gather
    _assert_gather_prompt(root, "Please say yes or no.", READBACK)


def test_confirmation_result_is_said_and_the_follow_up_is_the_greeting(fd: Harness) -> None:
    """For outcome read-outs the outcome can't be cut off, so it
    is a plain <Say>; only the follow-up is the interruptible greeting."""
    to_level3(fd, sid(8))
    confirm_handoff(fd, sid(8))
    root = fd.xml("factor", sid(8), Digits="2")
    assert_outcome(root, "Okay, I haven't done that.", follow="Is there anything else I can help you with?")
    outcome, follow = fd.pending_result(sid(8)).spoken[-2:]
    assert says(root) == [outcome.text] and greeting(root) == follow.text


# --------------------------------------------------------------------------- bridge: barge-in


def test_bridge_caller_barging_in_on_the_greeting_gets_the_envelope() -> None:
    """ConversationRelay sends ``interrupt`` (the greeting was cut off), then the caller's
    ``prompt``. That prompt is the caller's turn: the agent's first request carries the
    completed get_balance envelope (the spoken texts) with ``next`` = ``await_caller``, then the caller's words as a real caller turn."""
    from tests.component.test_bridge_fakes import make_rig, run, say, spoken, write_call_state
    from tests.component.test_bridge_flows import balance_envelope

    rig = make_rig(script=[say("<silent/>"), say("Your savings balance is next. Shall I check it?")])
    write_call_state(rig.store, pending_tool={"tool_use_id": "tu_bal", "name": "get_balance",
                                              "args": {"account": "checking"}},
                     pending_result=balance_envelope())

    async def go() -> tuple[list[dict[str, Any]], int]:
        sim = rig.sim()
        await sim.connect()
        await sim.idle()
        before = len(rig.agent.calls)  # await_caller: nothing sent at reconnect
        await sim.interrupt("Your checking account ending in", ms=900)
        await sim.idle()
        frames = await sim.prompt("What about savings?")
        await sim.hangup()
        return frames, before

    frames, before = run(go())
    assert before == 0
    assert spoken(frames) == "Your savings balance is next. Shall I check it?"
    first, words = rig.agent.calls
    assert first["user_text"] is None
    (res,) = first["tool_results"]
    assert res.tool_use_id == "tu_bal" and res.status == "success"
    env = res.content
    assert "callerUtterance" not in env and env["next"] == "await_caller"
    assert words["user_text"] == "What about savings?"
    assert [s["text"] for s in env["spoken"]] == [
        "Your checking account ending in 4821 has an available balance of $1,234.56.",
        "Is there anything else I can help you with?"]
    assert rig.call()["pending_result"] is None
