"""Stream-agent-text: the translator's text release.

``_Translator.feed`` is driven directly with generated Strands stream events: model messages made of
text blocks (random text split into random chunks, empty deltas included), reasoning blocks and
tool-use blocks at random positions, framed by ``messageStart`` and ``messageStop`` (or ending
without the stop, the ``message`` event path), with inline interrupts and the invocation's
``result`` for the message-mode check.
"""

from __future__ import annotations

import string
from types import SimpleNamespace
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank.tools_contract import HANDOFF_TOOLS, INLINE_TOOLS
from tac_bridge.agent_client import (
    SILENT_MARK,
    MessageEnd,
    TextBeforeTool,
    ToolStarted,
    TextDelta,
    TextSuppressed,
    ToolCall,
    strip_reasoning,
    strip_silent,
)
from tac_bridge.agent_client_strands import ALLOWED_GATEWAY_TOOLS, INTERRUPT_PREFIX, _Translator, _TurnState
from tests.unit.test_strands_client import _ReferenceTranslator

PROPS = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])

#: Reasoning is written in this alphabet only, answer text never uses it: any of it in the released
#: text would be reasoning spoken.
REASONING_CHARS = "§¶"
#: Answer text, with whitespace-only pieces and ``<silent/>`` markers (whole or in pieces): text the
#: caller doesn't hear.
text_piece = st.one_of(
    st.text(alphabet=string.ascii_letters + " \n,.?!'$0123456789", min_size=0, max_size=12),
    st.sampled_from([" ", "\n\n", SILENT_MARK, SILENT_MARK[:4], SILENT_MARK[4:], "<"]))
chunked_text = st.lists(text_piece, min_size=1, max_size=6)  # one text block's deltas ("" allowed)
reasoning = st.lists(st.text(alphabet=REASONING_CHARS, min_size=1, max_size=6), min_size=0, max_size=3)
tool_name = st.sampled_from((*INLINE_TOOLS, *ALLOWED_GATEWAY_TOOLS, "level0___unapproved"))
tool_input = st.lists(st.sampled_from(['{"q": ', '"a"', "}", "[1]", ""]), max_size=3)

text_block = st.tuples(st.just("text"), chunked_text)
reasoning_block = st.tuples(st.just("reasoning"), reasoning)
tool_block = st.tuples(st.just("tool"), tool_name, tool_input)
any_block = st.one_of(text_block, reasoning_block, tool_block)


@st.composite
def message(draw: Any, *, tools: bool | None = None, stop: bool | None = None) -> dict[str, Any]:
    """A model message spec. ``tools``: ``False`` no tool block, ``True`` at least one, ``None`` either.
    ``stop``: whether it ends with ``messageStop`` (``None``: either)."""
    pool = st.one_of(text_block, reasoning_block) if tools is False else any_block
    blocks = draw(st.lists(pool, min_size=0, max_size=6))
    if tools and not any(b[0] == "tool" for b in blocks):
        blocks.insert(draw(st.integers(0, len(blocks))), draw(tool_block))
    has_stop = draw(st.booleans()) if stop is None else stop
    return {"blocks": blocks, "stop": has_stop}


def render(msg: dict[str, Any], ids: Any) -> list[dict[str, Any]]:
    """The Strands events for one model message: raw chunks as ``{"event": chunk}``, then the
    ``message`` event Strands sends when the model message is complete."""
    out: list[dict[str, Any]] = [{"messageStart": {"role": "assistant"}}]
    for b in msg["blocks"]:
        if b[0] == "text":
            out.append({"contentBlockStart": {"start": {}}})
            out += [{"contentBlockDelta": {"delta": {"text": t}}} for t in b[1]]
        elif b[0] == "reasoning":
            out.append({"contentBlockStart": {"start": {}}})
            out += [{"contentBlockDelta": {"delta": {"reasoningContent": {"text": t}}}} for t in b[1]]
            out.append({"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "sig"}}}})
        else:
            out.append({"contentBlockStart": {"start": {"toolUse": {"toolUseId": next(ids), "name": b[1]}}}})
            out += [{"contentBlockDelta": {"delta": {"toolUse": {"input": p}}}} for p in b[2]]
        out.append({"contentBlockStop": {}})
    if msg["stop"]:
        has_tool = any(b[0] == "tool" for b in msg["blocks"])
        out.append({"messageStop": {"stopReason": "tool_use" if has_tool else "end_turn"}})
        out.append({"metadata": {"usage": {"inputTokens": 3, "outputTokens": 2}, "metrics": {"latencyMs": 1}}})
    return [{"event": c} for c in out] + [{"message": {"role": "assistant", "content": []}}]


def interrupt_events(msg: dict[str, Any], ids: Any) -> list[dict[str, Any]]:
    """Strands' interrupt event for each inline tool block of a message (inputs of the message-mode check)."""
    out = []
    for b in msg["blocks"]:
        if b[0] == "tool" and b[1] in INLINE_TOOLS:
            tid = next(ids)
            out.append({"tool_interrupt_event": {
                "tool_use": {"toolUseId": tid, "input": {}},
                "interrupts": [SimpleNamespace(name=INTERRUPT_PREFIX + b[1],
                                               reason={"toolUseId": tid, "input": {"n": 1}})]}})
    return out


def _ids() -> Any:
    return (f"tu-{i}" for i in range(10_000))


def run_messages(msgs: list[dict[str, Any]], release: str, *, translator: Any = None,
                 interrupts: bool = False, message_ends: bool = False) -> tuple[list[list[Any]], _TurnState]:
    """Feed the messages through one translator; the events yielded per message (an event yielded
    while feeding a message's events belongs to it), and the turn state. :class:`MessageEnd` only
    with ``message_ends`` (the reference translator predates it)."""
    state = _TurnState()
    defer = HANDOFF_TOOLS.__contains__
    tr = _ReferenceTranslator(state, defer) if translator is _ReferenceTranslator \
        else _Translator(state, defer, release=release)  # type: ignore[arg-type]
    ids, iids = _ids(), _ids()
    per: list[list[Any]] = []
    for msg in msgs:
        evs = render(msg, ids) + (interrupt_events(msg, iids) if interrupts else [])
        out: list[Any] = []
        for ev in evs:
            out += list(tr.feed(ev))
        per.append(out)
    if interrupts:
        per[-1] += list(tr.feed({"result": object()}))
    if not message_ends:
        per = [[e for e in evs if not isinstance(e, MessageEnd)] for evs in per]
    return per, state


def text_of(blocks: list[tuple[Any, ...]]) -> str:
    return "".join("".join(b[1]) for b in blocks if b[0] == "text")


def released(events: list[Any]) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


@PROPS
@given(st.lists(message(tools=False, stop=True), min_size=1, max_size=3))
def test_no_tool_use_same_text_in_both_modes(msgs):
    """No tool-use block: the TextDeltas' concatenation is the same in stream
    and message mode and equals the message's text; reasoning never appears."""
    stream, _ = run_messages(msgs, "stream")
    message_, _ = run_messages(msgs, "message")
    for msg, s_ev, m_ev in zip(msgs, stream, message_, strict=True):
        assert released(s_ev) == released(m_ev) == text_of(msg["blocks"])
        assert not set(REASONING_CHARS) & set(released(s_ev))
        assert all(isinstance(e, TextDelta) for e in s_ev + m_ev)  # nothing suppressed, no late tool


@PROPS
@given(message(tools=True, stop=True))
def test_tool_use_stream_releases_exactly_the_text_before_the_first_tool(msg):
    """A message with a tool-use block: stream mode yields as TextDelta exactly the
    text before the first tool-use block, then one TextBeforeTool with the length of what the caller
    hears of it (``<silent/>`` removed; only if that is not blank), and suppresses the text after it;
    message mode yields no TextDelta and one TextSuppressed (of all the message's text, when it has
    any)."""
    first = next(i for i, b in enumerate(msg["blocks"]) if b[0] == "tool")
    before, after = text_of(msg["blocks"][:first]), text_of(msg["blocks"][first:])
    (s_ev,), _ = run_messages([msg], "stream")
    assert released(s_ev) == before
    late = [e for e in s_ev if isinstance(e, TextBeforeTool)]
    heard = strip_silent(strip_reasoning(before))  # what the bridge speaks of it
    assert late == ([TextBeforeTool(len(heard), msg["blocks"][first][1])] if heard.strip() else [])
    assert [e for e in s_ev if isinstance(e, TextSuppressed)] == ([TextSuppressed(len(after))] if after else [])
    (m_ev,), _ = run_messages([msg], "message")
    assert not [e for e in m_ev if isinstance(e, (TextDelta, TextBeforeTool))]
    total = before + after
    assert [e for e in m_ev if isinstance(e, TextSuppressed)] == ([TextSuppressed(len(total))] if total else [])


@PROPS
@given(st.lists(message(), min_size=1, max_size=4), st.booleans(), st.sampled_from(["stream", "message"]))
def test_each_model_message_ends_with_one_message_end(msgs, interrupts, release):
    """Every model message, with or without ``messageStop`` or a tool use the bridge
    sees, yields exactly one :class:`MessageEnd`, after all of its text and tool starts (only an
    inline tool's ToolCall, from the interrupt that follows the message, may come after it)."""
    per, _ = run_messages(msgs, release, interrupts=interrupts, message_ends=True)
    for evs in per:
        ends = [i for i, e in enumerate(evs) if isinstance(e, MessageEnd)]
        assert len(ends) == 1
        assert all(isinstance(e, ToolCall) for e in evs[ends[0] + 1:])
        assert not any(isinstance(e, (TextDelta, TextBeforeTool, TextSuppressed, ToolStarted))
                       for e in evs[ends[0] + 1:])


@PROPS
@given(st.lists(message(), min_size=1, max_size=4), st.booleans())
def test_message_mode_equals_the_reference_translator(msgs, interrupts):
    """Message mode's event sequence equals today's translator's for every input; the
    turn state it records (usage, provider timings, stop reason, which timings were taken) too."""
    got, gs = run_messages(msgs, "message", interrupts=interrupts)
    want, ws = run_messages(msgs, "message", translator=_ReferenceTranslator, interrupts=interrupts)
    assert got == want
    assert (gs.call_usage, gs.usage, gs.call_metrics, gs.call_stop) == \
        (ws.call_usage, ws.usage, ws.call_metrics, ws.call_stop)
    for name in ("first_byte", "first_text", "first_sentence"):
        assert (getattr(gs, name) is None) == (getattr(ws, name) is None), name
    if interrupts:  # the inputs reach the interrupt path: every inline tool block is a ToolCall
        calls = sum(1 for m in msgs for b in m["blocks"] if b[0] == "tool" and b[1] in INLINE_TOOLS)
        assert sum(isinstance(e, ToolCall) for m in got for e in m) == calls


@PROPS
@given(st.lists(message(), min_size=1, max_size=4))
def test_at_most_one_text_before_tool_per_message_after_its_last_text(msgs):
    """Per message, at most one TextBeforeTool, never before that message's last
    TextDelta (and only after text the caller hears was released in it)."""
    stream, _ = run_messages(msgs, "stream")
    for s_ev in stream:
        marks = [i for i, e in enumerate(s_ev) if isinstance(e, TextBeforeTool)]
        deltas = [i for i, e in enumerate(s_ev) if isinstance(e, TextDelta)]
        assert len(marks) <= 1
        if marks:
            assert deltas and max(deltas) < marks[0]
            heard = strip_silent(strip_reasoning("".join(s_ev[i].text for i in deltas)))
            assert heard.strip() and s_ev[marks[0]].chars == len(heard)
