"""owlbank.security_log: the stable auth-failure line and query-string redaction."""

from __future__ import annotations

from tests.logutil import lines as lines_of

import json
import logging

import pytest

from owlbank import security_log as sl

NONCE = "Zq9secretNonceValue_abcdefgh123456"


def test_auth_failed_line_is_stable_json(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=sl.SECURITY_LOGGER)
    sl.auth_failed("bridge", "nonce")
    (rec,) = [r for r in caplog.records if r.name == sl.SECURITY_LOGGER]
    assert rec.getMessage() == "twilio_auth_failed"
    assert lines_of([rec], sl.SECURITY_LOGGER) == [{"event": "twilio_auth_failed", "component": "bridge",
                                                    "reason": "nonce"}]
    # the stable text form is still available
    assert json.loads(sl.auth_failed_line("bridge", "nonce")) == {
        "event": "twilio_auth_failed", "component": "bridge", "reason": "nonce"}
    assert rec.levelno == logging.WARNING
    with pytest.raises(ValueError):
        sl.auth_failed_line("bridge", "free text")


@pytest.mark.parametrize("text", [
    f"WebSocket signature validation failed. URL used: wss://relay.example/ws?n={NONCE}",
    f'127.0.0.1:5000 - "WebSocket /ws?n={NONCE}" 403',
    f"GET https://relay.example/webhook?bodySHA256=abc&n={NONCE} 403",
])
def test_strip_queries(text: str) -> None:
    out = sl.strip_queries(text)
    assert NONCE not in out and "?[redacted]" in out


def test_installed_redaction_covers_tac_uvicorn_and_any_logger(caplog: pytest.LogCaptureFixture) -> None:
    sl.install_log_redaction()
    sl.install_log_redaction()  # idempotent
    caplog.set_level(logging.DEBUG)
    logging.getLogger("tac.server.signature_validation").warning(
        "WebSocket signature validation failed. URL used: %s", f"wss://relay.example/ws?n={NONCE}")
    logging.getLogger("uvicorn.error").info('%s - "WebSocket %s" %s', "10.0.0.1:1", f"/ws?n={NONCE}", 403)
    logging.getLogger("somewhere.else").warning(f"url wss://relay.example/ws?n={NONCE}")
    logging.getLogger("x").info("%d calls, %s", 3, "no query here")
    texts = [r.getMessage() for r in caplog.records]
    assert len(texts) == 4 and all(NONCE not in t for t in texts)
    assert texts[-1] == "3 calls, no query here"
    assert all(NONCE not in str(r.args) for r in caplog.records)


def test_basic_challenge_is_a_separate_info_event_the_alarm_filter_cannot_match(
        caplog: pytest.LogCaptureFixture) -> None:
    """Twilio's first request of every webhook has no Authorization header. It is logged as
    its own INFO event; the auth-failure alarm's literal term must not be in it."""
    caplog.set_level(logging.DEBUG, logger=sl.SECURITY_LOGGER)
    sl.basic_challenge("frontdoor")
    (rec,) = [r for r in caplog.records if r.name == sl.SECURITY_LOGGER]
    assert rec.levelno == logging.INFO and rec.getMessage() == "twilio_basic_challenge"
    assert lines_of([rec], sl.SECURITY_LOGGER) == [{
        "event": "twilio_basic_challenge", "component": "frontdoor", "reason": "basic", "challenge": True}]
    assert sl.AUTH_FAILED_EVENT not in sl.BASIC_CHALLENGE_EVENT
    assert sl.OTP_UNTRUSTED_EVENT not in sl.BASIC_CHALLENGE_EVENT
    with pytest.raises(ValueError):
        sl.basic_challenge("somewhere")


def test_signature_form_line_has_the_form_name_only(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=sl.SECURITY_LOGGER)
    sl.signature_form("frontdoor", "credentials")
    assert lines_of(caplog.records, sl.SECURITY_LOGGER) == [{
        "event": "twilio_signature_form", "component": "frontdoor", "form": "credentials"}]
    assert sl.AUTH_FAILED_EVENT not in sl.SIGNATURE_FORM_EVENT
    with pytest.raises(ValueError):
        sl.signature_form("frontdoor", "https://user:pass@host/")
