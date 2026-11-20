"""The ``world`` fixture: every e2e scenario ends with the security scan and
a scan of every log record of the run at DEBUG (no sensitive value, and every call correlated)."""

from __future__ import annotations

import logging

import pytest

from owlbank import obs
from tests.e2e.harness import World


class Capture(logging.Handler):
    """Keeps every record, with the call SID and component the JSON handler's filter would add."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []
        self.addFilter(obs.ContextFilter())

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def log_capture(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)  # the whole run is scanned at DEBUG, the noisiest level
    cap = Capture()
    logging.getLogger().addHandler(cap)
    yield cap
    logging.getLogger().removeHandler(cap)


@pytest.fixture
def make_world(caplog: pytest.LogCaptureFixture, log_capture: Capture):
    """Build a :class:`World` (optionally with options); scanned for leaks at teardown."""
    worlds: list[World] = []

    def make(**kw) -> World:
        w = World(**kw)
        worlds.append(w)
        return w

    yield make
    try:
        for w in worlds:
            w.assert_no_leaks(caplog.records)
            w.assert_logs_clean(log_capture.records)
            w.assert_correlated(log_capture.records)
            assert w.agent.unexpected == 0, "the agent got a turn the scenario didn't script"
            assert not w.agent.script, "scripted agent turns were never used"
    finally:
        for w in worlds:
            w.cleanup()


@pytest.fixture
def world(make_world) -> World:
    return make_world()
