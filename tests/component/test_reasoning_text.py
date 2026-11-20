"""Model failover: reasoning written as text is never spoken.

Haiku 4.5 with reasoning off sometimes writes its reasoning as text in ``<thinking>...</thinking>``
before its answer. The bridge removes every such block, and the markup of any other tag,
from each model call's text before the ``<silent/>`` filter, in ``stream`` and ``message`` modes:
the caller, the journal (``ai_turn``) and the agent's spoken text see only what is left. The turn
logs ``reasoning_text_removed`` with counts (never the text) and puts the removed reasoning on its
``bridge.turn`` span as ``owlbank.reasoning_text``, a text key that follows ``traceContent``;
Strands' own model-call span keeps the model's raw output (with ``full``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from owlbank.catalog import PromptCatalog
from owlbank.telemetry.allowlist import TEXT_KEYS, allowed
from tac_bridge.agent_client_strands import StrandsAgentClient, default_model_factory
from tests.component.masking_fixtures import masked_policy_doc, unmasked_policy_doc
from tests.component.test_bridge_fakes import (
    CALL_SID,
    FakeBackend,
    Rig,
    h_msg,
    h_text,
    h_tool,
    load_json,
    make_rig,
    run,
    say,
    spoken,
    wait_for,
    write_call_state,
)
from tests.component.test_stream_release import (
    ESCALATE,
    ai_turn_texts,
    chunks,
    converse,
    events,
    msg,
    stream_rig,
    timed_sim,
)
from tests.logutil import text_of
from tests.strands_fakes import (
    FakeBedrockRuntime,
    FakeModel,
    default_gateway,
    settings,
    text_msg,
    tool_msg,
    traced,
)

BRIDGE_LOG = "tac_bridge.bridge"
REASONING = "<thinking>\nEl cliente pregunta por su saldo.\n</thinking>"
ANSWER = "Setenta y cinco dólares."
LIVE = REASONING + "\n\n" + ANSWER  # as Haiku 4.5 wrote it


def message_rig(*script: Any) -> Rig:
    rig = make_rig(script=list(script))
    write_call_state(rig.store)
    return rig


def removed(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return events(caplog, "reasoning_text_removed")


def removed_at(caplog: pytest.LogCaptureFixture) -> list[int]:
    """The level of each ``reasoning_text_removed`` record."""
    return [r.levelno for r in caplog.records if r.name == BRIDGE_LOG and "reasoning_text_removed" in r.getMessage()]


def timer_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [ln for ln in events(caplog, "mask_spoken") if ln["kind"] == "timer"]


# --------------------------------------------------------------------------- stream mode


def test_stream_a_block_split_across_chunks_is_never_spoken(caplog: pytest.LogCaptureFixture) -> None:
    rig = stream_rig([msg(chunks("<thin", "king>\nEl cliente", " pregunta por su saldo.\n</thi",
                                 "nking>\n\n", ANSWER))])
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "¿Cuál es mi saldo?")
    assert spoken(frames) == ANSWER  # no "<", and no blank lines left in front of the answer
    assert ai_turn_texts(rig) == [ANSWER]
    (line,) = removed(caplog)
    assert (line["blocks"], line["chars"], line["tags"], line["turn"]) == (1, len(REASONING), 0, 1)
    assert "El cliente" not in text_of(caplog.records)  # counts only, never the text


def test_stream_an_unclosed_block_is_dropped_to_the_end(caplog: pytest.LogCaptureFixture) -> None:
    rig = stream_rig([msg(chunks("Sure. ", "<thinking>the caller", " wants a transfer"))])
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "Transfer money")
    assert spoken(frames) == "Sure. "
    assert ai_turn_texts(rig) == ["Sure. "]
    (line,) = removed(caplog)
    assert (line["blocks"], line["chars"]) == (1, len("<thinking>the caller wants a transfer"))


def test_stream_other_markup_goes_and_its_text_stays(caplog: pytest.LogCaptureFixture) -> None:
    rig = stream_rig([msg(chunks("<mess", "age>Your balance is ", "5 dollars.</message>"))])
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "Balance?")
    assert spoken(frames) == "Your balance is 5 dollars."
    assert ai_turn_texts(rig) == ["Your balance is 5 dollars."]
    (line,) = removed(caplog)
    assert (line["blocks"], line["tags"]) == (0, 2)


def test_stream_the_silent_marker_still_reaches_the_silent_filter() -> None:
    rig = stream_rig([msg(chunks("<thinking>hand over</thinking>", "<sil", "ent/>"))])
    frames, _ = converse(rig, "hmm")
    assert spoken(frames) == ""
    assert ai_turn_texts(rig) == []


def test_stream_an_unclosed_block_ends_with_its_model_message(caplog: pytest.LogCaptureFixture) -> None:
    # A Gateway tool runs inside the agent loop, so two model messages share one hop: the first
    # message's unclosed block is dropped when its tool use ends it, and the next one is spoken.
    model = FakeModel([tool_msg(("tu_gw", "level0___faq_lookup", {"q": "fees"}), text="<thinking>check the fees"),
                       text_msg("Hello<thinking>x</thinking>there, no fee.", chunks=4)])
    rig = make_rig()
    rig.bridge._agent = StrandsAgentClient(settings(text_release="stream", thinking="adaptive"),  # noqa: SLF001
                                           model_factory=lambda _s, _sid: model, gateway_tools=default_gateway())
    write_call_state(rig.store)
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "Any fees?")
    assert not model.script and len(model.calls) == 2  # one hop, two model messages
    assert spoken(frames) == "Hello there, no fee."  # one space where the block sat between words
    assert ai_turn_texts(rig) == ["Hello there, no fee."]
    (line,) = removed(caplog)
    assert line["blocks"] == 2


def test_stream_a_reasoning_only_slow_prefix_lets_the_acknowledgement_fire(caplog: pytest.LogCaptureFixture) -> None:
    # Removed reasoning is no speech, so it doesn't cancel the timed
    # acknowledgement. The model writes only reasoning for longer than the timer: the ack is
    # spoken once, then the answer follows.
    rig = stream_rig([msg(chunks("<thinking type=\"x\">", "checking the balance", 0.4, "</thinking >\n",
                                 "Your balance is fine."))],
                     policy_doc=masked_policy_doc(after_ms=100))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "How is my balance?")
    tokens = [f["token"] for f in frames if f.get("type") == "text" and f.get("token")]
    (timer,) = timer_lines(caplog)
    assert len(tokens) == 2 and tokens[1] == "Your balance is fine."  # the ack, then the answer
    assert spoken(frames) == tokens[0] + "Your balance is fine."
    assert timer["ms_since_turn_start"] >= 100
    assert ai_turn_texts(rig) == ["Your balance is fine."]
    assert removed_at(caplog) == [logging.INFO]  # the turn spoke: INFO


def test_stream_real_text_first_cancels_the_acknowledgement(caplog: pytest.LogCaptureFixture) -> None:
    rig = stream_rig([msg(chunks("Sure. ", "<thinking>", "checking", 0.4, "</thinking>", "It is fine."))],
                     policy_doc=masked_policy_doc(after_ms=100))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "How is my balance?")
    assert spoken(frames) == "Sure. It is fine."
    assert timer_lines(caplog) == []
    assert ai_turn_texts(rig) == ["Sure. It is fine."]


@pytest.mark.parametrize("script", [
    [msg(chunks("<thinking>The answer is", " 75 dollars.</thinkng>", "Seventy-five dollars."))],  # misspelt close
    [msg(chunks("<thinking>Nothing", " to say.</thinking>", "\n"))],
])
def test_a_turn_that_removed_reasoning_and_spoke_nothing_warns(caplog: pytest.LogCaptureFixture,
                                                                script: list[Any]) -> None:
    # Fail-closed silence (an unclosed or misspelled close tag swallows
    # the whole answer) is visible as a WARNING, counts only.
    rig = stream_rig(script)
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "Balance?")
    assert spoken(frames) == ""
    assert removed_at(caplog) == [logging.WARNING]
    (line,) = removed(caplog)
    assert line["blocks"] == 1
    assert "75 dollars" not in text_of(caplog.records) and "Nothing" not in text_of(caplog.records)


@pytest.mark.parametrize("mode", ["stream", "message"])
def test_an_earlier_tool_call_does_not_hide_a_swallowed_final_answer(caplog: pytest.LogCaptureFixture,
                                                                      mode: str) -> None:
    # The WARNING is decided on the turn's last hop. Hop 1 calls a tool, hop 2
    # writes reasoning with a misspelt close: the answer is swallowed, so WARNING.
    backend = FakeBackend({"list_recent_transactions": [{"account": "checking", "transactions": []}]})
    tool = h_msg(h_tool("list_recent_transactions", "tu_tx", i=0, account="checking", limit=3))
    if mode == "stream":
        rig = stream_rig([tool, msg(chunks("<thinking>The balance", " is 5.</thinkng>", " Your balance is $5"))],
                         backend=backend, policy_doc=unmasked_policy_doc())
    else:
        rig = make_rig(backend=backend, policy_doc=unmasked_policy_doc(),
                       script=[tool, say("<thinking>The balance is 5.</thinkng> Your balance is $5")])
        write_call_state(rig.store)
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "Recent transactions?")
    assert len(backend.calls) == 1
    assert spoken(frames) == ""
    assert removed_at(caplog) == [logging.WARNING]


def gateway_rig(second: str, text_release: str) -> tuple[Rig, FakeModel]:
    """One hop of two model messages: a Gateway tool (``faq_lookup``) runs inside the agent loop."""
    model = FakeModel([tool_msg(("tu_gw", "level0___faq_lookup", {"q": "fees"}), text="Let me check."),
                       text_msg(second, chunks=4)])
    rig = make_rig(policy_doc=unmasked_policy_doc())
    rig.bridge._agent = StrandsAgentClient(settings(text_release=text_release, thinking="adaptive"),  # noqa: SLF001
                                           model_factory=lambda _s, _sid: model, gateway_tools=default_gateway())
    write_call_state(rig.store)
    return rig, model


@pytest.mark.parametrize("text_release", ["stream", "message"])
@pytest.mark.parametrize(("second", "level"), [
    ("<thinking>The balance is 5.</thinkng> Your balance is $5", logging.WARNING),  # swallowed
    ("<thinking>The balance is 5.</thinking> Your balance is $5", logging.INFO),  # spoken
])
def test_a_gateway_tool_in_the_same_hop_does_not_hide_a_swallowed_answer(
        caplog: pytest.LogCaptureFixture, text_release: str, second: str, level: int) -> None:
    # The WARNING is decided per model message. The first message's Gateway tool
    # call is a tool call of the same hop, yet the second message's answer was swallowed: WARNING.
    rig, model = gateway_rig(second, text_release)
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "Any fees?")
    assert not model.script and len(model.calls) == 2  # one hop, two model messages
    assert spoken(frames).endswith("Your balance is $5") == (level == logging.INFO)
    assert "The balance is 5" not in spoken(frames)
    assert removed_at(caplog) == [level]


def test_stream_words_held_before_a_gateway_tool_are_spoken_before_its_masking_phrase() -> None:
    # A possible tag held at the end of a message's words is released when its
    # tool-use block starts (TextBeforeTool / ToolStarted), not at the message's end, so it is
    # never spoken after the tool's masking phrase. Here the held fragment is the only text: it
    # is the answer's start, so the phrase is skipped (answer_started) rather than spoken first.
    model = FakeModel([tool_msg(("tu_gw", "level0___faq_lookup", {"q": "fees"}), text="<Savings"),
                       text_msg(" has no fee.", chunks=2)])
    rig = make_rig(policy_doc=masked_policy_doc(tool_after_ms=0, masks={"faq_lookup": "mask.get_balance"}))
    rig.bridge._agent = StrandsAgentClient(settings(text_release="stream", thinking="adaptive"),  # noqa: SLF001
                                           model_factory=lambda _s, _sid: model, gateway_tools=default_gateway())
    write_call_state(rig.store)
    frames, _ = converse(rig, "Any fees?")
    phrase = PromptCatalog.from_dict(load_json("prompt_catalog.json")).text("mask.get_balance", "en-US")
    assert spoken(frames) == "Savings has no fee."
    assert phrase not in spoken(frames)


def unknown_tool_rig(first: str, second: str, text_release: str) -> tuple[Rig, FakeModel]:
    """One hop of two model messages: the first calls a tool name the model made up (neither inline
    nor a Gateway tool: no ToolStarted, no ToolCall); Strands answers it as unknown and loops."""
    model = FakeModel([tool_msg(("tu_x", "lookup_balance", {"account": "checking"}), text=first),
                       text_msg(second, chunks=4)])
    rig = make_rig(policy_doc=unmasked_policy_doc())
    rig.bridge._agent = StrandsAgentClient(settings(text_release=text_release, thinking="adaptive"),  # noqa: SLF001
                                           model_factory=lambda _s, _sid: model, gateway_tools=default_gateway())
    write_call_state(rig.store)
    return rig, model


@pytest.mark.parametrize("text_release", ["stream", "message"])
def test_an_unclosed_block_before_an_unknown_tool_ends_with_its_message(
        caplog: pytest.LogCaptureFixture, text_release: str) -> None:
    # The client's MessageEnd ends the first message even though the bridge sees
    # no tool use for it, so its unclosed block doesn't swallow the second message's answer.
    rig, model = unknown_tool_rig("<thinking>I'll look it up", "Your balance is $5", text_release)
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "Balance?")
    assert not model.script and len(model.calls) == 2  # one hop, two model messages
    assert spoken(frames) == "Your balance is $5"
    assert ai_turn_texts(rig) == ["Your balance is $5"]
    # message mode: the first message's text went with its tool use, never filtered
    assert removed_at(caplog) == ([logging.INFO] if text_release == "stream" else [])


@pytest.mark.parametrize("text_release", ["stream", "message"])
def test_silent_before_an_unknown_tool_does_not_hide_a_swallowed_answer(
        caplog: pytest.LogCaptureFixture, text_release: str) -> None:
    # The first message's <silent/> is that message's own: the second message's swallowed answer
    # (a misspelt close) is judged on its own, so WARNING.
    rig, model = unknown_tool_rig("<silent/>", "<thinking>The balance is 5.</thinkng> Your balance is $5",
                                  text_release)
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "Balance?")
    assert not model.script and len(model.calls) == 2
    assert spoken(frames) == ""
    assert removed_at(caplog) == [logging.WARNING]


@pytest.mark.parametrize("mode", ["stream", "message"])
def test_an_over_long_tag_dropped_to_the_end_of_the_answer_warns(caplog: pytest.LogCaptureFixture,
                                                                  mode: str) -> None:
    # A tag past MAX_TAG is dropped up to its ">" (or a "<", or the end), never
    # spoken; when that runs to the end of the last message and nothing was spoken, WARNING.
    text = '<answer note="' + "y" * 200 + ' Your balance is $5'
    rig = stream_rig([msg(chunks(text[:60], text[60:]))]) if mode == "stream" else message_rig(say(text))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "Balance?")
    assert spoken(frames) == ""
    assert removed_at(caplog) == [logging.WARNING]
    (line,) = removed(caplog)
    assert (line["blocks"], line["tags"]) == (0, 1)


def test_an_over_long_tag_ended_by_its_bracket_stays_info(caplog: pytest.LogCaptureFixture) -> None:
    rig = message_rig(say('<answer note="' + "y" * 200 + '">Your balance is $5'))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "Balance?")
    assert spoken(frames) == "Your balance is $5"
    assert removed_at(caplog) == [logging.INFO]


def test_reasoning_before_a_tool_then_a_spoken_answer_stays_info(caplog: pytest.LogCaptureFixture) -> None:
    # Reasoning removed in hop 1 (with its tool call), hop 2 speaks: INFO.
    backend = FakeBackend({"list_recent_transactions": [{"account": "checking", "transactions": []}]})
    rig = stream_rig([h_msg(h_text("<thinking>look it up</thinking>"),
                            h_tool("list_recent_transactions", "tu_tx", i=1, account="checking", limit=3)),
                      msg(chunks("No recent transactions."))],
                     backend=backend, policy_doc=unmasked_policy_doc())
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "Recent transactions?")
    assert spoken(frames) == "No recent transactions."
    assert removed_at(caplog) == [logging.INFO]


@pytest.mark.parametrize("mark", ["<sil|ent/>", "<silent |/>", "<Sil|ent/>", "<silent/| >", "< sil|ent / >"])
def test_reasoning_then_silent_stays_info(caplog: pytest.LogCaptureFixture, mark: str) -> None:
    # A deliberate <silent/> (any case or spacing) after reasoning: the caller hears nothing on
    # purpose, and the silent filter sees the marker.
    rig = stream_rig([msg(chunks("<thinking>hand over</thinking>", *mark.split("|")))])
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "hmm")
    assert spoken(frames) == ""
    assert ai_turn_texts(rig) == []
    assert removed_at(caplog) == [logging.INFO]
    (line,) = removed(caplog)
    assert (line["blocks"], line["tags"]) == (1, 0)


def test_a_barge_in_after_reasoning_stays_info(caplog: pytest.LogCaptureFixture) -> None:
    # The caller spoke again while the model was still reasoning: the turn ended early, not
    # silently, so no WARNING.
    rig = stream_rig([msg(chunks("<thinking>The caller wants", 0.6, " the balance.</thinking>", "It is fine.")),
                      h_msg(h_text("Sure, savings."))])

    async def go() -> list[dict[str, Any]]:
        sim, sock = timed_sim(rig)
        await sim.connect()
        await sim.prompt_nowait("checking balance")
        await wait_for(lambda: len(rig.agent.calls) == 1)
        await asyncio.sleep(0.1)
        await sim.interrupt("")
        await asyncio.sleep(0.8)
        await sim.idle()
        await sim.prompt("savings please")
        await sim.hangup()
        return list(sock.sent)

    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        sent = run(go())
    tokens = [f.get("token") for f in sent if f.get("type") == "text"]
    assert "It is fine." not in tokens and "Sure, savings." in tokens
    assert removed_at(caplog) == [logging.INFO]


def test_an_error_escalation_after_reasoning_stays_info(caplog: pytest.LogCaptureFixture) -> None:
    # Both model calls fail after writing only reasoning: the apology, then the escalation. The
    # turn did not end silently by itself, so no WARNING.
    def failed() -> list[Any]:
        return [{"messageStart": {"role": "assistant"}}, *chunks("<thinking>checking</thinking>")[:-1],
                RuntimeError("stream broke")]

    rig = stream_rig([failed(), failed()])
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        converse(rig, "Balance?")
    (turn,) = events(caplog, "turn")
    assert turn["outcome"] == "escalated"
    assert removed_at(caplog) == [logging.INFO]
    (line,) = removed(caplog)
    assert line["blocks"] == 2


def test_reasoning_then_a_handoff_stays_info(caplog: pytest.LogCaptureFixture) -> None:
    # Reasoning, then a hand-off with no words of the agent's own: a tool call, so INFO.
    rig = stream_rig([h_msg(h_text("<thinking>The caller wants a person.</thinking>"),
                            h_tool("escalate", "tu_esc", i=1, **ESCALATE))])
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        converse(rig, "Can I talk to a person?")
    assert events(caplog, "handoff_hold") or events(caplog, "escalation_requested")
    assert removed_at(caplog) == [logging.INFO]


def test_markup_only_removed_and_spoken_stays_info(caplog: pytest.LogCaptureFixture) -> None:
    rig = message_rig(say("<message>Hi.</message>"))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "Hello")
    assert spoken(frames) == "Hi."
    assert removed_at(caplog) == [logging.INFO]


def test_text_in_angle_brackets_loses_only_the_brackets(caplog: pytest.LogCaptureFixture) -> None:
    # No "<" or ">" reaches text-to-speech; the words are spoken.
    rig = stream_rig([msg(chunks("Your nickname is <Vaca", "tion fund>. And a <b and", " c > d."))])
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "My nickname?")
    assert spoken(frames) == "Your nickname is Vacation fund. And a b and c d."
    assert ai_turn_texts(rig) == ["Your nickname is Vacation fund. And a b and c d."]
    assert removed(caplog) == []


def turn_shape(rig: Rig, frames: list[dict[str, Any]]) -> tuple[list[Any], list[str]]:
    return ([f.get("type") for f in frames], [e["type"] for e in rig.events()])


def test_stream_a_reply_of_only_reasoning_is_an_empty_reply(caplog: pytest.LogCaptureFixture) -> None:
    rig = stream_rig([msg(chunks("<thinking>Nothing", " to say.</thinking>", "\n"))])
    empty = stream_rig([msg(chunks(""))])
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "hmm")
    frames_empty, _ = converse(empty, "hmm")
    assert spoken(frames) == ""
    assert ai_turn_texts(rig) == []
    assert turn_shape(rig, frames) == turn_shape(empty, frames_empty)
    (line,) = removed(caplog)
    assert line["blocks"] == 1


# --------------------------------------------------------------------------- message mode


def test_message_a_split_block_and_markup_are_removed(caplog: pytest.LogCaptureFixture) -> None:
    rig = message_rig(say("<thin", "king>\nEl cliente pregunta por su saldo.\n</think", "ing>\n\n<answer>",
                          ANSWER, "</answer>"))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "¿Cuál es mi saldo?")
    assert spoken(frames) == ANSWER
    assert ai_turn_texts(rig) == [ANSWER]
    (line,) = removed(caplog)
    assert (line["blocks"], line["chars"], line["tags"]) == (1, len(REASONING), 2)
    assert "El cliente" not in text_of(caplog.records)


def test_message_an_unclosed_block_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    rig = message_rig(say("One moment. <thinking>", "the caller wants"))
    frames, _ = converse(rig, "Help")
    assert spoken(frames) == "One moment. "
    assert ai_turn_texts(rig) == ["One moment. "]


def test_message_a_reply_of_only_reasoning_is_an_empty_reply(caplog: pytest.LogCaptureFixture) -> None:
    rig = message_rig(say("<thinking>Nothing to say.</thinking>"))
    empty = message_rig(say(""))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "hmm")
    frames_empty, _ = converse(empty, "hmm")
    assert spoken(frames) == ""
    assert turn_shape(rig, frames) == turn_shape(empty, frames_empty)
    (line,) = removed(caplog)
    assert line["blocks"] == 1


def test_nothing_removed_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    rig = message_rig(say("Less than < 5 dollars."))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames, _ = converse(rig, "Fees?")
    assert spoken(frames) == "Less than 5 dollars."  # a bracket is no markup: dropped, not counted
    assert removed(caplog) == []


# --------------------------------------------------------------------------- the turn span and Strands' span


def test_reasoning_text_is_a_text_key_and_the_counts_are_allowed() -> None:
    assert "owlbank.reasoning_text" in TEXT_KEYS and not allowed("owlbank.reasoning_text")
    for key in ("owlbank.reasoning_blocks", "owlbank.reasoning_chars", "owlbank.markup_tags"):
        assert allowed(key) and key not in TEXT_KEYS


def traced_call(content: str, redact: Any = None) -> tuple[list[Any], list[dict[str, Any]]]:
    """One turn on the real Strands client and ``BedrockModel`` over a stubbed ConverseStream,
    exported through the production processors and content filter; the spans and the frames."""
    runtime = FakeBedrockRuntime([text_msg(LIVE, chunks=4)])

    def factory(s: Any, _sid: str) -> Any:
        model = default_model_factory(s)
        model.client = runtime
        return model

    rig = make_rig()
    rig.bridge._agent = StrandsAgentClient(settings(text_release="message", thinking="off"),  # noqa: SLF001
                                           model_factory=factory, gateway_tools=default_gateway())
    write_call_state(rig.store)

    async def session() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("¿Cuál es mi saldo?")
        await sim.hangup()
        rig.bridge.session_end(CALL_SID, "closed")
        await rig.bridge.wait_idle()
        return frames

    with traced(content, redact=redact) as t:
        t.enable(rig.bridge)
        frames = run(session())
        spans = t.spans()
    assert not runtime.script
    return spans, frames


def turn_span(spans: list[Any]) -> Any:
    (turn,) = [s for s in spans if s.name == "bridge.turn"]
    return turn


def model_output(spans: list[Any]) -> str:
    """Everything Strands' spans carry (attributes and events), as text."""
    strands = [s for s in spans if s.instrumentation_scope is not None
               and s.instrumentation_scope.name == "strands.telemetry.tracer"]
    assert strands
    return json.dumps([[dict(s.attributes or {}), [dict(e.attributes or {}) for e in s.events]] for s in strands],
                      ensure_ascii=False)


def test_full_content_keeps_the_reasoning_on_the_turn_span_and_strands_raw_output() -> None:
    spans, frames = traced_call("full")
    assert spoken(frames) == ANSWER
    turn = turn_span(spans)
    assert turn.attributes["owlbank.reasoning_text"] == REASONING
    assert (turn.attributes["owlbank.reasoning_blocks"], turn.attributes["owlbank.reasoning_chars"],
            turn.attributes["owlbank.markup_tags"]) == (1, len(REASONING), 0)
    raw = model_output(spans)
    # The filter is on the speech path only: the model's own output, as Evaluations reads it.
    assert json.dumps(LIVE, ensure_ascii=False)[1:-1] in raw


def test_off_keeps_the_counts_and_drops_the_text() -> None:
    spans, frames = traced_call("off")
    assert spoken(frames) == ANSWER
    turn = turn_span(spans)
    assert "owlbank.reasoning_text" not in turn.attributes
    assert turn.attributes["owlbank.reasoning_blocks"] == 1
    assert turn.attributes["owlbank.markup_tags"] == 0
    assert "El cliente" not in json.dumps([dict(s.attributes or {}) for s in spans], ensure_ascii=False)


def test_redacted_passes_the_text_through_the_redaction() -> None:
    seen: list[str] = []

    def redact(text: str) -> str:
        seen.append(text)
        return "[redacted]"

    spans, _ = traced_call("redacted", redact=redact)
    assert turn_span(spans).attributes["owlbank.reasoning_text"] == "[redacted]"
    assert REASONING in seen
