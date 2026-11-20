"""Call observability, ``traceContent=redacted``: the exact-value scrub doesn't depend on when spans
are exported.

The bridge forgets a call's sensitive values at session end (or when the turn still running at
hang-up ends). Before it does, the trace redactor (``TranscriptRedactor.retain``) keeps a snapshot
of them for ``RETAIN_TTL_S``, and scrubs each span with its own call's values. So
a span exported after the forget (a turn in flight at hang-up, a flush that raised or timed out)
still has the values scrubbed, and ``owlbank.segments`` forgets them promptly either way. The guard
here passes text through unchanged, so any value that reaches the exporter was missed by the scrub.

Fixed scenarios (the property over export orders is in
``tests/property/test_bridge_telemetry_props.py``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from owlbank import segments, sensitive
from owlbank.telemetry.bridge import TranscriptRedactor
from tac_bridge import bridge as bridge_mod
from tac_bridge.agent_client_strands import StrandsAgentClient
from tests.component.test_bridge_fakes import (
    CALL_SID,
    NOW,
    SEALER,
    make_rig,
    rearm_nonce,
    run,
    write_call_state,
)
from tests.logutil import lines
from tests.strands_fakes import FakeModel, Traced, default_gateway, settings, text_msg, traced

SSN4 = "4821"
WORD = "zephyrine"
VALUES = (SSN4, WORD)
BRIDGE_LOG = "tac_bridge.bridge"


class PassGuard:
    """``PiiGuard.text`` that changes nothing: only the exact-value scrub can remove a value."""

    def text(self, surface: str, text: str, **_k: Any) -> str:
        return text


@pytest.fixture(autouse=True)
def _fresh_process() -> Iterator[None]:
    sensitive.forget(CALL_SID)
    yield
    sensitive.forget(CALL_SID)


def _rig(model: FakeModel) -> Any:
    rig = make_rig()
    rig.bridge._agent = StrandsAgentClient(settings(), model_factory=lambda _s, _sid: model,
                                           gateway_tools=default_gateway())
    write_call_state(rig.store)
    for slot, value in (("ssn4", SSN4), ("word", WORD)):  # what the front door leaves (sealed)
        assert sensitive.remember(rig.store, SEALER, CALL_SID, value, slot=slot, now=lambda: NOW)
    sensitive.forget(CALL_SID)  # the bridge is another process: it restores them on its reads
    return rig


def _texts(spans: list[Any]) -> list[str]:
    out: list[str] = []
    for s in spans:
        out += [json.dumps(v, default=str) for v in (s.attributes or {}).values()]
        for e in s.events:
            out += [json.dumps(v, default=str) for v in (e.attributes or {}).values()]
    return out


def _assert_scrubbed(spans: list[Any]) -> None:
    texts = _texts(spans)
    leaks = [v for v in VALUES for t in texts if segments.contains_value(t, v)]
    assert leaks == []
    assert any("[redacted]" in t for t in texts), "the scrub ran on the call's spans"


@pytest.fixture
def redacted() -> Iterator[tuple[Traced, TranscriptRedactor]]:
    redactor = TranscriptRedactor(PassGuard())
    with traced("redacted", redact=redactor) as t:
        yield t, redactor


async def _say_values(rig: Any) -> Any:
    sim = rig.sim()
    await sim.connect()
    await sim.prompt(f"my last four are {SSN4} and my word is {WORD}")
    return sim


# --------------------------------------------------------------------------- (a) hang-up mid-turn


def test_a_turn_running_at_hang_up_exports_its_spans_scrubbed_after_the_forget(
        redacted: tuple[Traced, TranscriptRedactor]) -> None:
    t, redactor = redacted
    model = FakeModel([text_msg("Noted."), text_msg(f"Thanks, I have {WORD} and {SSN4} on file.", chunks=4)],
                      delay=0.0)
    rig = _rig(model)
    t.enable(rig.bridge)

    async def go() -> None:
        sim = await _say_values(rig)
        model.delay = 0.05  # the second turn streams slowly: the session ends while it runs
        await sim.prompt_nowait(f"repeat {WORD} and {SSN4} back")
        await asyncio.wait_for(_until(lambda: len(model.calls) == 2), 3)
        rig.bridge.session_end(CALL_SID, "disconnect")  # the hang-up, mid-turn
        assert segments.sensitive_values(CALL_SID), "the running turn keeps the values"
        await sim.wait_reply()  # the turn finishes after the session ended
        await rig.bridge.wait_idle()
        await sim.hangup()

    run(go())
    assert not segments.sensitive_values(CALL_SID)  # forgotten once the turn ended
    spans = t.spans()  # the in-flight turn's spans are exported only now
    assert CALL_SID in redactor.retained_calls()  # kept for its TTL, export or not (fails closed)
    turn_spans = [s for s in spans if s.name == "bridge.turn"]
    assert len(turn_spans) == 2
    _assert_scrubbed(spans)


async def _until(pred: Any) -> None:
    while not pred():
        await asyncio.sleep(0.002)


# --------------------------------------------------------------------------- (b) the flush fails


def _boom() -> bool:
    raise RuntimeError("exporter down")


def _slow() -> bool:
    time.sleep(0.3)
    return True


@pytest.mark.parametrize("flush", ["raises", "times_out", "returns_false"])
def test_a_failed_flush_still_forgets_and_late_spans_are_scrubbed(
        flush: str, redacted: tuple[Traced, TranscriptRedactor], monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture) -> None:
    t, redactor = redacted
    monkeypatch.setattr(bridge_mod, "TRACE_FLUSH_TIMEOUT_S", 0.05)
    rig = _rig(FakeModel([text_msg(f"Got {WORD}.")]))
    fn = {"raises": _boom, "times_out": _slow, "returns_false": lambda: False}[flush]
    rig.bridge.enable_tracing(t.tracer, flush=fn, retain=redactor.retain)

    async def go() -> None:
        sim = await _say_values(rig)
        await sim.hangup()
        rig.bridge.session_end(CALL_SID, "disconnect")
        assert not segments.sensitive_values(CALL_SID)  # forgotten at once, not after the flush
        await rig.bridge.wait_idle()

    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        run(go())
    assert not segments.sensitive_values(CALL_SID)
    assert CALL_SID in redactor.retained_calls()  # nothing exported yet: the snapshot is held
    _assert_scrubbed(t.spans())  # the late export
    assert CALL_SID in redactor.retained_calls()  # held for its TTL after the export too
    (warn,) = [r for r in caplog.records
               if r.name == BRIDGE_LOG and lines([r], BRIDGE_LOG)[0]["event"] == "trace_flush_incomplete"]
    assert warn.levelno == logging.WARNING
    assert rig.bridge.tracing is not None and rig.bridge.tracing.failures >= 1
    assert not [v for v in VALUES for r in caplog.records if v in r.getMessage()]


def test_a_retain_that_raises_still_forgets_and_warns_once_without_values(
        caplog: pytest.LogCaptureFixture) -> None:
    calls = []

    def retain(call_sid: str) -> None:
        calls.append(call_sid)
        raise RuntimeError(f"snapshot {SSN4} failed")  # a message holding a value: never logged

    with traced("redacted", redact=TranscriptRedactor(PassGuard())) as t:
        rig = _rig(FakeModel([text_msg("Noted.")]))
        rig.bridge.enable_tracing(t.tracer, flush=None, retain=retain)

        async def go() -> None:
            sim = await _say_values(rig)
            await sim.hangup()
            rig.bridge.session_end(CALL_SID, "disconnect")
            assert not segments.sensitive_values(CALL_SID)  # forgotten all the same
            await rig.bridge.wait_idle()
            rig.bridge._forget_values(CALL_SID)  # a second failure within the minute: not logged again

        with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
            run(go())
    assert calls == [CALL_SID, CALL_SID]
    warns = [r for r in caplog.records
             if r.name == BRIDGE_LOG and lines([r], BRIDGE_LOG)[0]["event"] == "trace_retain_failed"]
    assert len(warns) == 1 and warns[0].levelno == logging.WARNING
    assert rig.bridge.tracing is not None and rig.bridge.tracing.failures >= 2
    assert not [v for v in VALUES for r in caplog.records if v in r.getMessage()]


# --------------------------------------------------------------------------- (c) re-entry during the flush


def test_a_new_session_started_during_the_flush_keeps_its_values(
        redacted: tuple[Traced, TranscriptRedactor]) -> None:
    t, redactor = redacted
    release = threading.Event()
    provider = t.handle.tracer_provider

    def gated_flush() -> bool:
        release.wait(3)
        return provider.force_flush(3000)

    rig = _rig(FakeModel([text_msg("Noted."), text_msg(f"Still have {WORD}.")]))
    rig.bridge.enable_tracing(t.tracer, flush=gated_flush, retain=redactor.retain)

    async def go() -> None:
        sim = await _say_values(rig)
        await sim.hangup()
        rig.bridge.session_end(CALL_SID, "disconnect")  # the flush now waits on ``release``
        assert not segments.sensitive_values(CALL_SID)
        rearm_nonce(rig.store, CALL_SID)
        sim = rig.sim()
        await sim.connect()  # session 2, while session 1's flush is still running: restores them
        await sim.prompt(f"it's {WORD}")
        assert segments.sensitive_values(CALL_SID)
        release.set()
        await rig.bridge.wait_idle()  # session 1's flush finishes
        assert segments.sensitive_values(CALL_SID), "the old session's flush forgot nothing"
        await sim.hangup()
        rig.bridge.session_end(CALL_SID, "disconnect")
        await rig.bridge.wait_idle()

    run(go())
    assert not segments.sensitive_values(CALL_SID)
    _assert_scrubbed(t.spans())
