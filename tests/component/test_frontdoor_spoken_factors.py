"""Spoken verification factors, keypad-only prompts, two retries per factor and the
``factor_answer`` log line. The no-leak scan runs at teardown.

The journal event is ``slot_resolution`` in place of ``factor_answer``, which
stays a log line with the same shape fields plus the modes, ``match``, ``source``, ``model`` and,
for a non-sensitive slot (ZIP in the seed), the value. ``flow["tries"]`` is per slot."""

from __future__ import annotations

import logging

import pytest

from lambdas.frontdoor.app import MAX_KEYPAD_REMINDERS
from owlbank.segments import clear_sensitive
from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    FAILED_AI, MARIA, TEST_FACTORS, THOMAS, Harness, all_fixture_factor_values, assert_no_leak, build, fd,
    relay, says, sid,
)

THOMAS_ZIP = TEST_FACTORS["C-1001"]["zip"]
MARIA_ZIP = TEST_FACTORS["C-1004"]["zip"]
EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ES = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
REMINDER = "For your security, please use your keypad."
ZIP_PROMPT = "Please enter your five-digit ZIP code. If you can't, press pound."   # keypad-only twin
CLARIFY = "Please enter the 5 digits of your ZIP code on your keypad."
GIVE_UP = "I'm sorry, I wasn't able to verify your identity."
FACTOR_ANSWER_KEYS = {"factor", "input_source", "digit_count", "speech_present", "result_kind",
                      "tries"}
#: Without ``value`` (non-sensitive slots only).
SLOT_RESOLUTION_KEYS = {"flow", "slot", "configured_mode", "mode_source", "ceiling", "effective_mode",
                        "capped", "match", "source", "model", "asked", "answer", "correction",
                        "correction_of", "tries"}


def words(digits: str, table: list[str]) -> str:
    return " ".join(table[int(c)] for c in digits)


@pytest.fixture
def spoken(caplog: pytest.LogCaptureFixture):
    """A front door with ``accept_spoken_factors`` on (the default in every stage)."""
    caplog.set_level(logging.DEBUG)
    h = build(config_kw={"accept_spoken_factors": True})
    yield h
    assert_no_leak(h, caplog.records, all_fixture_factor_values() | h.sent_factors)
    for call in {k[0].split("#")[0] for k in h.store._tables["calls"]}:  # noqa: SLF001
        clear_sensitive(call)


@pytest.fixture
def keypad(caplog: pytest.LogCaptureFixture):
    """Spoken factors off (``accept_spoken_factors=False``, the default is on):
    every factor is keypad-only. The no-leak scan runs at teardown, as for ``fd``."""
    caplog.set_level(logging.DEBUG)
    h = build(config_kw={"accept_spoken_factors": False})
    yield h
    assert_no_leak(h, caplog.records, all_fixture_factor_values() | h.sent_factors)
    for call in {k[0].split("#")[0] for k in h.store._tables["calls"]}:  # noqa: SLF001
        clear_sensitive(call)


def speak(h: Harness, call: str, text: str, value: str | None = None):
    if value:
        h.sent_factors.add(value)
    return h.xml("factor", call, SpeechResult=text, Confidence="0.9")


def answers(caplog: pytest.LogCaptureFixture) -> list[dict]:
    """The ``factor_answer`` log lines, as their shape fields."""
    return [{k: getattr(r, k) for k in FACTOR_ANSWER_KEYS} for r in caplog.records
            if getattr(r, "event", None) == "factor_answer"]


def resolutions(h: Harness, call: str) -> list[dict]:
    return [e for e in h.journal(call) if e["type"] == "slot_resolution"]


def zip_tries(h: Harness, call: str) -> int:
    return h.session(call).flow["tries"].get("zip", 0)


# --------------------------------------------------------------------------- spoken factors on


def test_spoken_on_gather_takes_dtmf_and_speech(spoken):
    g = spoken.start(sid(1), THOMAS).find("Gather")
    assert g.get("input") == "dtmf speech" and g.get("numDigits") == "5"
    assert g.get("language") == "en-US" and g.get("finishOnKey") == "#"


def test_spoken_on_english_digits_are_verified(spoken, caplog):
    spoken.start(sid(1), THOMAS)
    root = speak(spoken, sid(1), words(THOMAS_ZIP, EN), THOMAS_ZIP)
    assert "verify" in spoken.lam.ops()
    assert relay(root) is not None and spoken.state(sid(1)).level == 1
    (ans,) = answers(caplog)
    assert ans["input_source"] == "speech" and ans["speech_present"] is True
    assert ans["digit_count"] == 0 and ans["result_kind"] == "value"


def test_spoken_on_spanish_digits_are_verified(spoken):
    root = spoken.start(sid(1), MARIA, lang_digit="2")
    assert root.find("Gather").get("language") == "es-US"
    root = speak(spoken, sid(1), words(MARIA_ZIP, ES), MARIA_ZIP)
    assert "verify" in spoken.lam.ops() and spoken.state(sid(1)).level == 1


# --------------------------------------------------------------------------- spoken factors off


def test_spoken_off_gather_is_keypad_only(keypad):
    fd = keypad
    g = fd.start(sid(1), THOMAS).find("Gather")
    assert g.get("input") == "dtmf" and g.get("numDigits") == "5"
    assert g.get("finishOnKey") == "#" and g.get("actionOnEmptyResult") == "true"
    for attr in ("language", "speechModel", "hints"):
        assert g.get(attr) is None, attr


def test_spoken_off_reminder_does_not_use_a_try(keypad):
    fd = keypad
    fd.start(sid(1), THOMAS)
    root = speak(fd, sid(1), words(THOMAS_ZIP, EN), THOMAS_ZIP)
    assert says(root) == [REMINDER, ZIP_PROMPT]
    assert root.find("Gather").get("input") == "dtmf"
    assert "verify" not in fd.lam.ops()
    assert zip_tries(fd, sid(1)) == 0
    # Both retries are still there: two unclear answers re-prompt, the third gives up.
    assert says(fd.factor(sid(1), "12")) == [CLARIFY]
    assert says(fd.factor(sid(1), "")) == [CLARIFY]
    root = fd.factor(sid(1), "")
    assert relay(root).get("welcomeGreeting") == FAILED_AI and "verify" not in fd.lam.ops()


def test_spoken_off_keypad_after_reminder_is_verified(keypad):
    fd = keypad
    fd.start(sid(1), THOMAS)
    speak(fd, sid(1), words(THOMAS_ZIP, EN), THOMAS_ZIP)
    root = fd.factor(sid(1), THOMAS_ZIP)
    assert "verify" in fd.lam.ops() and relay(root) is not None


def test_spoken_off_reminders_are_capped_then_count_as_tries(keypad):
    fd = keypad
    fd.start(sid(1), THOMAS)
    for _ in range(MAX_KEYPAD_REMINDERS):
        root = speak(fd, sid(1), words(THOMAS_ZIP, EN), THOMAS_ZIP)
        assert says(root) == [REMINDER, ZIP_PROMPT]
    assert zip_tries(fd, sid(1)) == 0
    # Past the cap each reminder uses a try: two more re-prompt, the next one gives up.
    for tries in (1, 2):
        root = speak(fd, sid(1), words(THOMAS_ZIP, EN))
        assert says(root) == [REMINDER, ZIP_PROMPT]
        assert zip_tries(fd, sid(1)) == tries
    root = speak(fd, sid(1), words(THOMAS_ZIP, EN))
    assert relay(root).get("welcomeGreeting") == FAILED_AI and "verify" not in fd.lam.ops()


def test_valid_keypad_wins_over_speech(fd, caplog):
    fd.start(sid(1), THOMAS)
    root = fd.factor(sid(1), THOMAS_ZIP, SpeechResult="one two three", Confidence="0.9")
    assert "verify" in fd.lam.ops() and relay(root) is not None
    (ans,) = answers(caplog)
    assert ans["input_source"] == "dtmf" and ans["speech_present"] is True


# --------------------------------------------------------------------------- retries


def test_two_retries_then_give_up(fd, caplog):
    fd.start(sid(1), THOMAS)
    assert says(fd.factor(sid(1), "12")) == [CLARIFY]
    assert says(fd.factor(sid(1), "123")) == [CLARIFY]
    root = fd.factor(sid(1), "")
    assert relay(root).get("welcomeGreeting") == FAILED_AI  # verify_failed -> limited_ai
    assert "verify" not in fd.lam.ops()
    assert [a["tries"] for a in answers(caplog)] == [0, 1, 2]
    assert [r["tries"] for r in resolutions(fd, sid(1))] == [0, 1, 2]


# --------------------------------------------------------------------------- factor_answer log


def test_factor_answer_is_logged_as_shape_only(keypad, caplog):
    fd = keypad
    fd.start(sid(1), THOMAS)
    speak(fd, sid(1), words(THOMAS_ZIP, EN), THOMAS_ZIP)
    fd.factor(sid(1), "123")
    fd.factor(sid(1), THOMAS_ZIP)
    logged = answers(caplog)
    assert [(a["factor"], a["input_source"], a["digit_count"], a["speech_present"],
             a["result_kind"], a["tries"]) for a in logged] == [
        ("zip", "speech", 0, True, "value", 0),
        ("zip", "dtmf", 3, False, "unclear", 0),
        ("zip", "dtmf", 5, False, "value", 1),
    ]
    journal = resolutions(fd, sid(1))
    assert [(e["slot"], e["match"], e["source"], e["tries"], e.get("value")) for e in journal] == [
        ("zip", "exact", "speech", 0, THOMAS_ZIP),   # ZIP isn't sensitive in the seed
        ("zip", "none", "dtmf", 0, None),
        ("zip", "exact", "dtmf", 1, THOMAS_ZIP),
    ]
    for e in journal:
        assert set(e) - {"type", "call_sid", "seq", "ts", "expires_at", "value"} == SLOT_RESOLUTION_KEYS
        assert "nine" not in str(e)   # never the caller's words
    recs = [r for r in caplog.records if getattr(r, "event", None) == "factor_answer"]
    assert len(recs) == 3
    for r in recs:
        assert r.factor == "zip" and isinstance(r.digit_count, int)
        fields = {k: getattr(r, k) for k in FACTOR_ANSWER_KEYS}
        assert all(isinstance(v, (str, int, bool)) for v in fields.values())
        text = r.getMessage() + " " + " ".join(str(v) for v in fields.values())
        assert THOMAS_ZIP not in text and "nine" not in text and "123" not in text


def test_factor_answer_with_no_input(fd, caplog):
    fd.start(sid(1), THOMAS)
    fd.factor(sid(1), "")
    (ans,) = answers(caplog)
    assert ans["input_source"] == "none" and ans["digit_count"] == 0
    assert ans["speech_present"] is False and ans["result_kind"] == "unclear"
