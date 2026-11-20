"""The bridge's log lines are the same on both hostings.

The same scripted sessions run through the real app (``build_app``, Starlette's test client, real
TAC and the bridge) on ``fargate`` (a signed ``/ws?n=`` upgrade) and on ``agentcore`` (the
Runtime session-ID header, the call binding and the ``customParameters["n"]`` nonce). The scenarios
are the parity rig: masking, hand-off, re-entry and silence. For each, the frames Twilio receives
are equal, and so is the sequence of ``(event, sorted field names)`` of every JSON line, except
``hosting``'s value and ``agentcore``'s ``ws_signature_check`` start-up line. Every line from
``ws_session_start`` on carries the call SID. ``hosting`` is on the start-up and ``ws_session_*``
lines; ``startup_complete`` is logged once per app, before any socket.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from twilio.request_validator import RequestValidator

from owlbank import last_spoken, obs
from tac_bridge import silence
from tac_bridge.app import STARTUP_COMPLETE_EVENT, build_app
from tac_bridge.hosting import SESSION_ID_HEADER
from tests.component.test_bridge_fakes import (
    AUTH_TOKEN,
    CALL_SID,
    NOW,
    PUBLIC_DOMAIN,
    SEALER,
    FakeBackend,
    Rig,
    make_rig,
    rearm_nonce,
    relay_nonce,
    say,
    tool,
    write_call_state,
)
from tests.component.masking_fixtures import masked_policy_doc, unmasked_policy_doc

HOSTINGS = ("fargate", "agentcore")
VALIDATOR = RequestValidator(AUTH_TOKEN)
WS_URL = f"wss://{PUBLIC_DOMAIN}/ws"
#: Start-up lines only one hosting has.
HOSTING_ONLY_EVENTS = frozenset({"ws_signature_check"})
#: The provider's own timings on ``model_call`` lines, logged only when the call reported them (Strands
#: sets ``timeToFirstByteMs`` only when it is non-zero, ``strands/event_loop/streaming.py``), so a fake
#: model that answers within 1 ms drops them on one hosting and not the other: not a hosting difference.
PROVIDER_TIMING_FIELDS = frozenset({"bedrock_latency_ms", "strands_ttfb_ms"})
#: The background session start's lines (``Bridge._begin_session``): it runs alongside the setup
#: and the first turn, so where they fall among the turn's lines (``system_prompt_rendered``) is
#: scheduling, not a hosting difference. They are compared as their own sequence.
SESSION_START_EVENTS = frozenset({"agent_session_start", "agent_history_new", "agent_history_reload"})
HOSTING_LINES = frozenset({STARTUP_COMPLETE_EVENT, "ws_session_start", "ws_session_end"})
OUR_LOGGERS = frozenset({"tac_bridge", "owlbank"})
GREETING = "Paid. Anything else?"
TXNS = {"tool": "list_recent_transactions",
        "data": {"accountType": "checking", "last4": "4821", "transactions": []}}


# --------------------------------------------------------------------------- the harness


@contextmanager
def captured() -> Iterator[io.StringIO]:
    """Every log line as production writes it: the bridge's JSON handler on the root logger."""
    buf = io.StringIO()
    obs.configure_logging("bridge", level=logging.INFO, stream=buf)
    yield buf


def json_lines(buf: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]


class Session:
    """One ConversationRelay socket on ``hosting``: the upgrade, ``setup``, then prompts."""

    def __init__(self, client: TestClient, hosting: str, call_sid: str = CALL_SID) -> None:
        self.client, self.hosting, self.call_sid = client, hosting, call_sid
        self.frames: list[dict[str, Any]] = []

    @contextmanager
    def open(self) -> Iterator[Session]:
        nonce = relay_nonce(self.call_sid)
        params: dict[str, Any] = {"callSid": self.call_sid}
        if self.hosting == "fargate":
            query = urlencode({"n": nonce})
            path = f"/ws?{query}"
            headers = {"X-Twilio-Signature": VALIDATOR.compute_signature(f"{WS_URL}?{query}", {})}
        else:
            path, headers = "/ws", {SESSION_ID_HEADER: self.call_sid}
            params["n"] = nonce
        with self.client.websocket_connect(path, headers=headers) as ws:
            self.ws = ws
            ws.send_json({"type": "setup", "callSid": self.call_sid, "from": "+12025550101",
                          "to": "+12025550199", "customParameters": params})
            yield self

    def receive_until(self, done: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        while True:
            frame = json.loads(self.ws.receive_text())
            out.append(frame)
            self.frames.append(frame)
            if done(frame):
                return out

    def prompt(self, text: str) -> list[dict[str, Any]]:
        self.ws.send_json({"type": "prompt", "voicePrompt": text, "lang": "en-US", "last": True})
        return self.receive_until(lambda f: f.get("type") == "end" or (f.get("type") == "text" and f.get("last")))

    def until_end(self) -> list[dict[str, Any]]:
        return self.receive_until(lambda f: f.get("type") == "end")


Scenario = Callable[[TestClient, str, Rig], list[dict[str, Any]]]


def run_on(hosting: str, rig_factory: Callable[[], Rig], scenario: Scenario) -> tuple[list[dict[str, Any]],
                                                                                         list[dict[str, Any]]]:
    """The scenario on ``hosting``: the frames Twilio received and every JSON log line."""
    rig = rig_factory()
    with captured() as buf:
        app = build_app(rig.tac, rig.voice, rig.bridge, hosting=hosting)  # type: ignore[arg-type]
        with TestClient(app) as client:
            frames = scenario(client, hosting, rig)
    return frames, json_lines(buf)


def shape(lines: Sequence[dict[str, Any]]) -> tuple[list[tuple[str, tuple[str, ...]]], ...]:
    """``(event, sorted field names)`` of the bridge's lines, without the hosting-only start-up lines and
    the provider-timing fields (:data:`PROVIDER_TIMING_FIELDS`): the session start's lines
    (:data:`SESSION_START_EVENTS`), then all the others, each in order."""
    kept = [(ln["event"], tuple(sorted(set(ln) - PROVIDER_TIMING_FIELDS)))
            for ln in ours(lines) if ln["event"] not in HOSTING_ONLY_EVENTS]
    return ([s for s in kept if s[0] in SESSION_START_EVENTS], [s for s in kept if s[0] not in SESSION_START_EVENTS])


def ours(lines: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The bridge's own lines (``tac_bridge.*``, ``owlbank.*``). Library lines (TAC, Strands, the
    Memory session manager) are not part of the contract, and some are once per process."""
    return [ln for ln in lines if str(ln.get("logger", "")).split(".")[0] in OUR_LOGGERS]


def assert_parity(rig_factory: Callable[[], Rig], scenario: Scenario) -> dict[str, list[dict[str, Any]]]:
    results = {h: run_on(h, rig_factory, scenario) for h in HOSTINGS}
    (f_frames, f_lines), (a_frames, a_lines) = results["fargate"], results["agentcore"]
    assert _comparable(f_frames) == _comparable(a_frames)
    assert shape(f_lines) == shape(a_lines)
    for hosting, (_frames, lines) in results.items():
        assert_hosting_fields(lines, hosting)
        assert_call_sid_after_setup(lines)
    assert [ln["event"] for ln in a_lines if ln["event"] in HOSTING_ONLY_EVENTS] == ["ws_signature_check"]
    assert not [ln for ln in f_lines if ln["event"] in HOSTING_ONLY_EVENTS]
    return {h: lines for h, (_f, lines) in results.items()}


def _comparable(frames: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Frames with the hand-off journal reference (random per run) blanked."""
    out = []
    for f in frames:
        if f.get("type") == "end" and "handoffData" in f:
            data = json.loads(f["handoffData"])
            data["journal"] = "<ref>" if data.get("journal") else data.get("journal")
            data.pop("conversationId", None)
            f = {**f, "handoffData": data}
        out.append(f)
    return out


def assert_hosting_fields(lines: Sequence[dict[str, Any]], hosting: str) -> None:
    """``hosting`` is on the start-up and ``ws_session_*`` lines, with this hosting's value."""
    for ln in lines:
        if ln["event"] in HOSTING_LINES:
            assert ln.get("hosting") == hosting, ln["event"]
    assert [ln["event"] for ln in lines].count(STARTUP_COMPLETE_EVENT) == 1
    events = [ln["event"] for ln in lines]
    assert events.index(STARTUP_COMPLETE_EVENT) < events.index("ws_session_start")


def assert_call_sid_after_setup(lines: Sequence[dict[str, Any]]) -> None:
    events = [ln["event"] for ln in lines]
    first = events.index("ws_session_start")
    missing = [ln["event"] for ln in lines[first:] if not ln.get("call_sid")]
    assert missing == [], missing


# --------------------------------------------------------------------------- scenarios


def plain_call(*replies: str) -> Scenario:
    def scenario(client: TestClient, hosting: str, rig: Rig) -> list[dict[str, Any]]:
        s = Session(client, hosting)
        with s.open():
            for i, _ in enumerate(replies):
                s.prompt(f"question {i}")
        return s.frames
    return scenario


def test_a_plain_call_logs_the_same_lines_on_both_hostings() -> None:
    def rig() -> Rig:
        r = make_rig(script=[say("Hello ", "from the bridge.")])
        write_call_state(r.store)
        return r

    lines = assert_parity(rig, plain_call("Hello from the bridge."))
    for hosting, got in lines.items():
        session = [ln for ln in got if ln["event"].startswith("ws_session_")]
        assert [ln["event"] for ln in session] == ["ws_session_start", "ws_session_end"]
        assert {ln["hosting"] for ln in session} == {hosting}


def test_masking_parity() -> None:
    """A tool read-out the bridge masks (the timed acknowledgement and the read-out) on both."""
    def rig() -> Rig:
        backend = FakeBackend({"list_recent_transactions": [dict(TXNS)]})
        r = make_rig(backend=backend, policy_doc=masked_policy_doc(after_ms=0, tool_after_ms=0), script=[
            tool("list_recent_transactions", "tu_tx", account="checking"),
            say("You have no recent", " transactions.")])
        write_call_state(r.store)
        return r

    def scenario(client: TestClient, hosting: str, r: Rig) -> list[dict[str, Any]]:
        s = Session(client, hosting)
        with s.open():
            s.prompt("What did I spend recently?")
        return s.frames

    lines = assert_parity(rig, scenario)
    for got in lines.values():
        assert [ln["key"] for ln in got if ln["event"] == "mask_spoken"] == ["mask.list_recent_transactions"]


def test_handoff_then_reentry_parity() -> None:
    """A hand-off for step-up (``end`` with handoffData), then the call comes back on a new socket."""
    def rig() -> Rig:
        r = make_rig(script=[tool("get_balance", "tu_bal", account="savings"), say("Welcome back.")])
        write_call_state(r.store)
        return r

    def scenario(client: TestClient, hosting: str, r: Rig) -> list[dict[str, Any]]:
        first = Session(client, hosting)
        with first.open():
            first.prompt("savings balance please")
        rearm_nonce(r.store)  # the front door's reconnect: a fresh nonce for the next socket
        second = Session(client, hosting)
        with second.open():
            second.prompt("I'm back")
        return first.frames + second.frames

    lines = assert_parity(rig, scenario)
    for got in lines.values():
        assert [ln["event"] for ln in got].count("ws_session_start") == 2


def test_silence_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The greeting, one silence repeat, then the configured action (escalate, no input)."""
    monkeypatch.setattr(silence, "TTS_START_S", 0.02)

    def rig() -> Rig:
        doc = unmasked_policy_doc()
        doc["masking"] = {**doc["masking"], "chars_per_s": 30, "silence_timeout_ms": 120,
                          "silence_max_repeats": 1, "silence_action": "escalate",
                          "greeting_silence_timeout_ms": 120}
        r = make_rig(policy_doc=doc)
        write_call_state(r.store, last_spoken=last_spoken.seal(SEALER, GREETING, call_sid=CALL_SID, now=NOW))
        return r

    def scenario(client: TestClient, hosting: str, r: Rig) -> list[dict[str, Any]]:
        s = Session(client, hosting)
        with s.open():
            s.until_end()
        return s.frames

    assert_parity(rig, scenario)


# --------------------------------------------------------------------------- the property

words = st.text("abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)
replies = st.lists(st.lists(words, min_size=1, max_size=3).map(" ".join), min_size=1, max_size=3)


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(answers=replies, handoff=st.booleans(), reentry=st.booleans())
def test_same_scripted_session_same_lines(answers: list[str], handoff: bool, reentry: bool) -> None:
    """For any scripted session (turns, an optional hand-off, an optional re-entry), both
    hostings log equal ``(event, sorted field names)`` sequences, apart from ``hosting``'s value and
    ``ws_signature_check``, and every line from ``ws_session_start`` on carries the call SID."""
    script: list[Any] = [say(a) for a in answers]
    if handoff:
        script.append(tool("get_balance", "tu_bal", account="savings"))
    if reentry:
        script.append(say("Welcome back."))

    def rig() -> Rig:
        r = make_rig(script=list(script))
        write_call_state(r.store)
        return r

    def scenario(client: TestClient, hosting: str, r: Rig) -> list[dict[str, Any]]:
        s = Session(client, hosting)
        with s.open():
            for i in range(len(answers) + (1 if handoff else 0)):
                s.prompt(f"turn {i}")
        frames = list(s.frames)
        if reentry:
            rearm_nonce(r.store)
            back = Session(client, hosting)
            with back.open():
                back.prompt("again")
            frames += back.frames
        return frames

    assert_parity(rig, scenario)


# --------------------------------------------------------------------------- the stamp itself


def test_event_fields_stamp_only_their_events_and_an_explicit_field_wins(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    log = logging.getLogger("tac_bridge.bridge")
    obs.set_event_fields(("ws_session_start",), hosting="agentcore")
    obs.emit(log, logging.INFO, "ws_session_start")
    obs.emit(log, logging.INFO, "ws_session_start", hosting="fargate")
    obs.emit(log, logging.INFO, "turn")
    obs.set_event_fields(())  # cleared
    obs.emit(log, logging.INFO, "ws_session_start")
    got = [getattr(r, "hosting", None) for r in caplog.records if r.name == "tac_bridge.bridge"]
    assert got == ["agentcore", "fargate", None, None]


@pytest.mark.parametrize("fields", [{"hosting": "two words"}, {"hosting": "x" * 41}, {"event": "x"},
                                    {"call_sid": "x"}, {"ts": "x"}, {"hosting": 3}])
def test_event_fields_take_short_identifiers_under_free_names(fields: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        obs.set_event_fields(("ws_session_start",), **fields)


def test_build_app_stamps_the_session_lines_with_its_hosting(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    log = logging.getLogger("tac_bridge.bridge")
    for hosting in HOSTINGS:
        r = make_rig()
        build_app(r.tac, r.voice, r.bridge, hosting=hosting)  # type: ignore[arg-type]
        obs.emit(log, logging.INFO, "ws_session_end", reason="closed")
    got = [r.hosting for r in caplog.records if r.name == "tac_bridge.bridge"]
    assert got == list(HOSTINGS)


# --------------------------------------------------------------------------- call_latency reads the lines


@pytest.mark.parametrize("hosting", HOSTINGS)
def test_call_latency_classifies_cold_connects_by_startup_complete(hosting: str) -> None:
    """``scripts/bench/call_latency.py`` reads the real lines: ``hosting`` from ``ws_session_start``,
    and cold when the process's ``startup_complete`` falls between the mint and the session start
    (``agent_settings`` stays its fallback; it is checked in the bench tests)."""
    from scripts.bench import call_latency as cl

    def rig() -> Rig:
        r = make_rig(script=[say("Hello.")])
        write_call_state(r.store)
        return r

    _frames, lines = run_on(hosting, rig, plain_call("Hello."))
    startup = next(ln for ln in lines if ln["event"] == cl.STARTUP_EVENTS[0])
    start = next(ln for ln in lines if ln["event"] == "ws_session_start")
    assert startup["event"] == STARTUP_COMPLETE_EVENT and "call_sid" not in startup
    stream = "process-1"
    sessions = cl.parse_sessions([(100.0, start, stream)])
    assert sessions[0]["hosting"] == hosting
    minted = [(90.0, {"event": "relay_minted", "call_sid": CALL_SID, "hosting": hosting})]
    cold = cl.pair_connects(minted, sessions, {stream: [95.0]})  # the process started after the mint
    warm = cl.pair_connects(minted, sessions, {stream: [10.0]})  # it was already up
    assert (cold[0]["start"], warm[0]["start"]) == ("cold", "warm")
    assert cold[0]["hosting"] == hosting
