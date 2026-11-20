"""TAC's records go through our root handler: scrubbed, one JSON line each.

Constructing ``TAC`` replaces the ``tac`` logger's handlers with TAC's own stream handler and stops
propagation, so its records would skip the root handler and its scrubber. A failed Memory lookup
logs the request body at ERROR, and for a verified caller that body holds the customer's email.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from tac.core.logging import get_logger

from tests.component.test_bridge_fakes import make_rig, make_tac, say

from owlbank import obs
from tac_bridge.app import build_app

EMAIL = "jane.doe+bank@mail.example.co.uk"
PHONE = "+14155550123"
SECRET = "Zq9secretNonceValue_abcdefgh123456"


@pytest.fixture
def root_stream() -> Iterator[io.StringIO]:
    """What our root handler writes, with the ``tac`` logger restored afterwards."""
    tac_logger = logging.getLogger("tac")
    saved = (list(tac_logger.handlers), tac_logger.level, tac_logger.propagate)
    out = io.StringIO()
    obs.configure_logging("bridge", stream=out)
    yield out
    tac_logger.handlers[:] = saved[0]
    tac_logger.setLevel(saved[1])
    tac_logger.propagate = saved[2]


def _memory_failure() -> None:
    """The line TAC's Memory client logs when a profile lookup fails."""
    payload = {"idType": "email", "value": EMAIL}
    get_logger("context.memory").error(
        f"Failed to lookup profile from Conversation Memory: 429 Too Many Requests\n"
        f"URL: https://memory.twilio.com/v1/Stores/mem_store_test/Profiles/Lookup\n"
        f"Request body: {payload}\n"
        f"Response: caller {PHONE} token {SECRET}")


def _tac_lines(out: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.getvalue().splitlines()
            if line.strip() and "Conversation Memory" in line]


def _assert_one_masked_json_line(out: io.StringIO) -> None:
    (line,) = _tac_lines(out)  # one line per record: not also written by a TAC handler
    assert line["level"] == "ERROR" and line["component"] == "bridge"
    text = json.dumps(line)
    for value in (EMAIL, PHONE, SECRET, "jane.doe", PHONE[2:]):
        assert value not in text, text
    assert "[email]" in line["event"] and "+*******0123" in line["event"] and "[token]" in line["event"]


def test_route_to_root_sends_tac_records_through_the_root_handler(root_stream: io.StringIO) -> None:
    make_tac()  # TAC's setup_logging: its own handler, propagation off
    tac_logger = logging.getLogger("tac")
    assert tac_logger.propagate is False and tac_logger.handlers
    obs.route_to_root(tac_logger)
    obs.route_to_root(tac_logger)  # idempotent
    assert tac_logger.handlers == [] and tac_logger.propagate is True
    _memory_failure()
    _assert_one_masked_json_line(root_stream)


def test_tac_log_level_is_still_honoured(root_stream: io.StringIO) -> None:
    make_tac()  # log_level="WARNING"
    tac_logger = logging.getLogger("tac")
    obs.route_to_root(tac_logger)
    assert tac_logger.level == logging.WARNING
    get_logger("context.memory").info("Conversation Memory lookup started")
    assert _tac_lines(root_stream) == []
    _memory_failure()
    assert len(_tac_lines(root_stream)) == 1


def test_build_app_routes_tac_to_root(root_stream: io.StringIO) -> None:
    rig = make_rig(script=[say("Hello.")])  # constructs TAC: its own handler again
    assert logging.getLogger("tac").propagate is False
    build_app(rig.tac, rig.voice, rig.bridge)
    assert logging.getLogger("tac").handlers == [] and logging.getLogger("tac").propagate is True
    _memory_failure()
    _assert_one_masked_json_line(root_stream)
