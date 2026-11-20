"""AI keypad shortcuts through the bridge.

The bridge runs with its real catalog, policy, journal and call state, a scripted agent and a fake
voice channel holding one TAC-like session. Keys go into the call's ``KeypadBuffer`` at generated
times in the past, then the bridge's own ``_keypad_due`` runs every completed entry's action, as the
scheduled wake-up does. Properties:

* For generated key sequences and timings, no log line has a digit field, a
  ``keypad_action`` line has only the action and a key count, every agent input is a catalog word for
  1 or 2 (one per ``yes``/``no`` action, in order, in the call's language), and a long digit entry
  appears in no log, journal event, call state, agent input, text sent to the caller or exported
  span (the bridge's tracing on, into an in-memory exporter: names, attributes and events).
* A lone 1 (2) gives exactly one agent turn, its text ``keypad.yes`` (``keypad.no``)
  in the call's language.
* (1.2, the bridge's part): a lone 0 gives exactly one escalation (``caller_requested_human``,
  ``source`` and ``trigger`` ``keypad``) and no model call.
* A lone 9 sends exactly the last spoken text again; the agent's history and the
  silence repeat count are unchanged.
* With a hand-off pending, no key does anything, except that the pending hand-off's ``end``
  goes out (an empty send, TAC then sends the ``end``) exactly once, however many keys come.
* **Hold and early wake-ups**, on a real event loop through
  ``on_keypad`` and the scheduled wake-up: a lone key acts exactly once however early (up to 5 ms)
  the wake-up runs; a key pressed before TAC's session exists acts exactly once when the session
  comes within ``session_wait_s`` of the socket's session start, and is dropped exactly once,
  logged ``session_not_ready``, when it doesn't (it never acts late).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from owlbank import keypad
from owlbank.catalog import PromptCatalog
from owlbank.pii import FakePiiRedactor, PiiGuard
from owlbank.policy import Policy
from owlbank.segments import SegmentLog
from tac_bridge.bridge import Bridge, BridgeConfig
from tests.component.masking_fixtures import unmasked_policy_doc
from tests.component.test_bridge_fakes import (
    CALL_SID,
    NOW,
    SEALER,
    FakeAgent,
    FakeBackend,
    _inline_io,
    load_json,
    seeded_store,
    write_call_state,
)
from tests.leakscan import value_pattern
from tests.logutil import fields, text_of

CATALOG = PromptCatalog.from_dict(load_json("prompt_catalog.json"))
LANGUAGES = ("en-US", "es-US")
WORDS = {(kind, lang): CATALOG.text(f"keypad.{kind}", lang) for kind in ("yes", "no") for lang in LANGUAGES}
KEYS = list("0123456789*#")
GAPS = [0.0, 0.2, keypad.WINDOW_S - 0.01, keypad.WINDOW_S, keypad.WINDOW_S + 0.01, 3.0]

languages = st.sampled_from(LANGUAGES)
spoken = st.text(alphabet="abcdefghijklmnopqrstuvwxyz ,.?!'ABCDEFGáé", min_size=1, max_size=60).filter(
    lambda t: bool(t.strip()))
sequences = st.lists(st.tuples(st.sampled_from(KEYS), st.sampled_from(GAPS)), min_size=1, max_size=14)

#: The bridge's spans, recorded as exported (one provider for the module; cleared per example).
SPANS = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(SPANS))
TRACER = _PROVIDER.get_tracer("test_bridge_keypad_props")


def spans_text() -> str:
    """Every exported span's name, attributes and events (names and attributes), as text."""
    return json.dumps([{"name": s.name, "attributes": dict(s.attributes or {}),
                        "events": [{"name": e.name, "attributes": dict(e.attributes or {})} for e in s.events]}
                       for s in SPANS.get_finished_spans()], default=str)


class _Voice:
    """One TAC-like session with a registered socket; every send recorded as the text it carried."""

    def __init__(self, context: Any) -> None:
        self._conversations = {context.conversation_id: context}
        self.sent: list[str] = []

    def get_conversation_session_by_call_sid(self, call_sid: str) -> Any:
        return next((s for s in self._conversations.values() if s.call_sid == call_sid), None)

    def get_websocket(self, conversation_id: str) -> Any:
        return object()

    async def send_response(self, conversation_id: str, gen: Any) -> None:
        if isinstance(gen, str):
            self.sent.append(gen)
            return
        self.sent.append("".join([chunk async for chunk in gen]))


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def captured() -> Iterator[_Records]:
    logger = logging.getLogger("tac_bridge")
    handler, level = _Records(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def build(language: str, last_text: str | None, *, pending: bool = False, repeats: int = 0,
          traced: bool = False, config: BridgeConfig | None = None,
          voice_cls: type[_Voice] | None = None) -> tuple[Bridge, FakeAgent, _Voice, Any, SegmentLog, Any]:
    store = seeded_store()
    write_call_state(store, language=language)
    policy = Policy.from_dict(unmasked_policy_doc())
    journal = SegmentLog(store, now=lambda: NOW)
    agent = FakeAgent([])
    bridge = Bridge(store=store, agent=agent, backend=FakeBackend(), memory=None, catalog=CATALOG,
                    policy=policy, pii=PiiGuard.for_policy(FakePiiRedactor(), policy), journal=journal,
                    sealer=SEALER, config=config or BridgeConfig(ai_keypad=True), now=lambda: NOW, io=_inline_io)
    context = SimpleNamespace(call_sid=CALL_SID, conversation_id="conv_keypad", metadata={}, profile_id=None,
                              profile=None, pending_handoff_data=object() if pending else None)
    voice = (voice_cls or _Voice)(context)
    bridge.voice = voice  # type: ignore[assignment]
    if traced:  # the session and turn spans, and their keypad_action events, into SPANS
        SPANS.clear()
        bridge.enable_tracing(TRACER)
    bridge.session_start(CALL_SID)
    speech = bridge._speech[CALL_SID]
    speech.language, speech.text, speech.label = language, last_text, "agent"
    speech.watch.repeats = repeats
    return bridge, agent, voice, context, journal, store


async def press(bridge: Bridge, keys: list[tuple[str, float]]) -> None:
    """The keys at generated times (all in the past), then the wake-up's work."""
    buffer = bridge._keypad.setdefault(CALL_SID, keypad.KeypadBuffer())
    at = time.monotonic() - 1000.0
    for key, gap in keys:
        at += gap
        buffer.on_digit(key, at)
    await bridge._keypad_due(CALL_SID, buffer)


def keypad_lines(handler: _Records) -> list[dict[str, Any]]:
    return [fields(r) for r in handler.records if fields(r).get("event") == "keypad_action"]




@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(keys=sequences, language=languages, last=spoken)
def test_no_digit_reaches_any_output_and_the_agent_gets_only_catalog_words(
        keys: list[tuple[str, float]], language: str, last: str) -> None:
    bridge, agent, voice, _ctx, journal, store = build(language, last, traced=True)

    async def go() -> None:
        await press(bridge, keys)
        bridge.session_end(CALL_SID, "closed")  # ends the session span, so it is exported

    with captured() as handler:
        asyncio.run(go())
    actions = keypad_lines(handler)
    spans = SPANS.get_finished_spans()
    assert any(s.name == "relay.session" for s in spans)
    events = [e for s in spans for e in s.events if e.name == "keypad_action"]
    assert [(e.attributes or {}).get("owlbank.action") for e in events] == [a["action"] for a in actions]
    for event in events:
        assert set(event.attributes or {}) <= {"owlbank.action", "owlbank.keys"}
    for span in spans:
        keys_seen = set(span.attributes or {}) | {k for e in span.events for k in (e.attributes or {})}
        assert not any("digit" in k for k in keys_seen), span.name
    for line in actions:
        assert set(line) <= {"event", "action", "keys", "call_sid", "component"}
        assert line["action"] in keypad.ACTIONS and isinstance(line["keys"], int)
    for record in handler.records:
        assert "digit" not in fields(record)
    # the agent only ever gets the catalog words, one per yes/no action, in order
    expected = [WORDS[(a["action"], language)] for a in actions if a["action"] in ("yes", "no")]
    assert [c["user_text"] for c in agent.calls] == expected
    # nothing acts after a 0 (the hand-off is pending)
    kinds = [a["action"] for a in actions]
    if "agent" in kinds:
        assert kinds.index("agent") == len(kinds) - 1
    entry = "".join(k for k, _ in keys)
    if len(entry) >= 6 and entry.isdigit():
        pattern = value_pattern(entry)
        for name, text in {"logs": text_of(handler.records),
                           "journal": json.dumps(journal.read(CALL_SID), default=str),
                           "call state": json.dumps(store.get("calls", {"call_sid": CALL_SID}), default=str),
                           "agent": agent.all_inputs_text(), "relay": json.dumps(voice.sent),
                           "spans": spans_text()}.items():
            assert not pattern.search(text.casefold()), name




@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(language=languages, digit=st.sampled_from(["1", "2"]), last=st.one_of(st.none(), spoken),
       repeats=st.integers(0, 3))
def test_a_lone_one_or_two_is_one_turn_with_the_catalog_word(
        language: str, digit: str, last: str | None, repeats: int) -> None:
    bridge, agent, _voice, _ctx, _journal, _store = build(language, last, repeats=repeats)
    asyncio.run(press(bridge, [(digit, 0.0)]))
    kind = "yes" if digit == "1" else "no"
    assert [c["user_text"] for c in agent.calls] == [WORDS[(kind, language)]]


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(language=languages, last=st.one_of(st.none(), spoken))
def test_a_lone_zero_is_one_escalation_and_no_model_call(language: str, last: str | None) -> None:
    bridge, agent, voice, context, journal, store = build(language, last)
    asyncio.run(press(bridge, [("0", 0.0)]))
    assert agent.calls == []
    assert voice.sent == [""]  # TAC sends last: true, then the end
    events = journal.read(CALL_SID)
    assert [e["reason"] for e in events if e["type"] == "escalation"] == ["caller_requested_human"]
    data = json.loads(context.pending_handoff_data.handoff_data)
    assert data["escalationReason"] == "caller_requested_human" and data["trigger"] == "keypad"
    escalation = store.get("calls", {"call_sid": CALL_SID})["escalation"]
    assert escalation["source"] == "keypad" and escalation["journal_ref"] == data["journal"]


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(language=languages, last=spoken, repeats=st.integers(0, 3))
def test_a_lone_nine_resends_the_last_text_only(language: str, last: str, repeats: int) -> None:
    bridge, agent, voice, _ctx, journal, _store = build(language, last, repeats=repeats)
    asyncio.run(press(bridge, [("9", 0.0)]))
    assert voice.sent == [last]
    assert agent.calls == []  # never in the agent's history
    assert bridge._speech[CALL_SID].watch.repeats == repeats  # the silence count is unchanged
    (event,) = [e for e in journal.read(CALL_SID) if e["type"] == "prompt_repeat"]
    assert event["trigger"] == "keypad"




@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(keys=sequences, language=languages, last=spoken)
def test_keys_do_nothing_while_a_hand_off_is_pending(
        keys: list[tuple[str, float]], language: str, last: str) -> None:
    bridge, agent, voice, _ctx, journal, _store = build(language, last, pending=True)

    async def go() -> None:
        for key, _gap in keys:
            bridge.on_keypad(CALL_SID, key)  # dropped at once
        await press(bridge, keys)  # and a wake-up that got there anyway does nothing
        while bridge._keypad_tasks.get(CALL_SID):  # the pending hand-off's end, re-sent
            await asyncio.gather(*bridge._keypad_tasks[CALL_SID])

    with captured() as handler:
        asyncio.run(go())
    assert voice.sent == [""]  # only the hand-off's end (TAC: last: true, then the end), once
    assert agent.calls == []
    assert [e for e in journal.read(CALL_SID) if e["type"] in ("escalation", "prompt_repeat", "handoff")] == []
    assert keypad_lines(handler) == []
    assert not bridge._keypad[CALL_SID].pending


# --------------------------------------------------------------------------- Hold, early wake-ups


SHORT = 0.02  # the keypad window for the real-loop properties


@contextmanager
def short_window() -> Iterator[None]:
    saved = keypad.WINDOW_S
    keypad.WINDOW_S = SHORT
    try:
        yield
    finally:
        keypad.WINDOW_S = saved


class _LateVoice(_Voice):
    """TAC's session for the call appears only when :meth:`ready` is called (its conversation lookup)."""

    def __init__(self, context: Any) -> None:
        super().__init__(context)
        self._late = context
        self._conversations = {}

    def ready(self) -> None:
        self._conversations = {self._late.conversation_id: self._late}


def early_wakes(loop: asyncio.AbstractEventLoop, early_s: float) -> None:
    original = loop.call_later

    def call_later(delay: float, callback: Any, *args: Any, **kwargs: Any) -> asyncio.TimerHandle:
        if getattr(callback, "__name__", "") == "_keypad_wake":
            delay = max(0.0, delay - early_s)
        return original(delay, callback, *args, **kwargs)

    loop.call_later = call_later  # type: ignore[method-assign]


async def drained(bridge: Bridge, seconds: float) -> None:
    await asyncio.sleep(seconds)
    while bridge._keypad_tasks.get(CALL_SID):
        await asyncio.gather(*bridge._keypad_tasks[CALL_SID], return_exceptions=True)


def ignored_lines(handler: _Records) -> list[dict[str, Any]]:
    return [fields(r) for r in handler.records if fields(r).get("event") == "keypad_ignored"]


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(key=st.sampled_from(KEYS), early_s=st.floats(0.0, 0.005), language=languages, last=spoken)
def test_a_lone_key_acts_once_however_early_its_wake_up_runs(
        key: str, early_s: float, language: str, last: str) -> None:
    bridge, _agent, _voice, _ctx, _journal, _store = build(language, last)

    async def go() -> None:
        early_wakes(asyncio.get_running_loop(), early_s)
        bridge.on_keypad(CALL_SID, key)
        await drained(bridge, SHORT + 0.03)

    with short_window(), captured() as handler:
        asyncio.run(go())
    assert [a["action"] for a in keypad_lines(handler)] == [keypad.kind_for(key)]
    assert ignored_lines(handler) == []


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(digit=st.sampled_from(["1", "2"]), ready_s=st.floats(0.0, 0.12), bound_s=st.floats(0.02, 0.2),
       language=languages)
def test_a_key_before_the_session_acts_once_it_comes_or_is_dropped_once(
        digit: str, ready_s: float, bound_s: float, language: str) -> None:
    assume(abs(ready_s - bound_s) >= 0.03)  # clear of the bound's edge (real clock)
    config = BridgeConfig(ai_keypad=True, session_wait_s=bound_s, session_poll_s=0.005)
    bridge, agent, voice, _ctx, _journal, _store = build(language, "Hello.", config=config, voice_cls=_LateVoice)
    assert isinstance(voice, _LateVoice)

    async def go() -> None:
        asyncio.get_running_loop().call_later(ready_s, voice.ready)
        bridge.on_keypad(CALL_SID, digit)  # the socket's session has just started: TAC's isn't there
        await drained(bridge, max(ready_s, bound_s, SHORT) + 0.05)

    with short_window(), captured() as handler:
        asyncio.run(go())
    kind = "yes" if digit == "1" else "no"
    if ready_s < bound_s:
        assert [a["action"] for a in keypad_lines(handler)] == [kind]
        assert [c["user_text"] for c in agent.calls] == [WORDS[(kind, language)]]
        assert ignored_lines(handler) == []
    else:
        assert keypad_lines(handler) == [] and agent.calls == []
        assert [ln["cause"] for ln in ignored_lines(handler)] == ["session_not_ready"]
