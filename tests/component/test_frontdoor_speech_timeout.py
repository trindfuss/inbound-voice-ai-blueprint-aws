"""Spoken factor answers end after 2 s of silence (``speechTimeout``), a verified factor is
acknowledged ("Got it.") before the next factor's prompt (before the OTP, on its own with a
redirect while the code is sent), and the language menu line is spoken in
the catalog's Spanish menu voice (``say.menuVoice``).

Signed fake webhooks into the real front door (``test_frontdoor_support.build``). Expected words
come from the prompt catalog."""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from typing import Any

import pytest

from lambdas.frontdoor.app import FrontDoorConfig
from owlbank.catalog import PromptCatalog
from owlbank.segments import clear_sensitive

from tests.component.test_frontdoor_caller_id_trust import confirm_handoff, to_level3
from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    DATA,
    TEST_FACTORS,
    OTP,
    THOMAS,
    Harness,
    all_fixture_factor_values,
    assert_no_leak,
    build,
    fd,
    says,
    sid,
)

EN, ES = "en-US", "es-US"
SPOKEN = {"accept_spoken_factors": True}


@pytest.fixture
def keypad(caplog: pytest.LogCaptureFixture):
    """Spoken factors off (``accept_spoken_factors=False``; the default is on). The no-leak scan runs
    at teardown, as for ``fd``."""
    caplog.set_level(logging.DEBUG)
    h = build(config_kw={"accept_spoken_factors": False})
    yield h
    assert_no_leak(h, caplog.records, all_fixture_factor_values() | h.sent_factors)
    for call in {k[0].split("#")[0] for k in h.store._tables["calls"]}:  # noqa: SLF001
        clear_sensitive(call)


def _cat(h: Harness) -> PromptCatalog:
    return h.app.d.catalog


def _gather(root: ET.Element) -> ET.Element:
    gather = root.find("Gather")
    assert gather is not None, ET.tostring(root)
    return gather


def _gather_says(root: ET.Element) -> list[str]:
    return [e.text or "" for e in _gather(root).iter("Say")]


def _step_up(h: Harness, call_sid: str, *, lang_digit: str = "") -> ET.Element:
    """Log Thomas in (ZIP), then the AI hands off for a level-3 step-up (SSN4, then OTP)."""
    h.start(call_sid, THOMAS, lang_digit=lang_digit)
    h.factor(call_sid, TEST_FACTORS["C-1001"]["zip"])
    h.set_pending_tool(call_sid, "authenticate", {"tool": "make_payment"})
    return h.handoff(call_sid, {"reason": "step-up", "tool": "make_payment", "journal": "jr-2"})


def _prompt_keys(h: Harness, call_sid: str) -> list[str]:
    return [e["prompt_key"] for e in h.journal(call_sid) if e["type"] == "prompt"]


def _catalog_with(mutate: Any) -> PromptCatalog:
    doc: dict[str, Any] = json.loads((DATA / "prompt_catalog.json").read_text())
    mutate(doc)
    return PromptCatalog.from_dict(doc)


# --------------------------------------------------------------------------- speechTimeout


def test_speech_factor_gathers_use_three_seconds_with_the_deepgram_model() -> None:
    """Twilio requires an integer speechTimeout with a speechModel, so the default
    is 3 s (early finish ends sooner) and every spoken factor Gather keeps the catalog's Deepgram model."""
    h = build(config_kw=SPOKEN)
    assert h.app.cfg.speech_timeout_s == 3
    root = h.start(sid(1), THOMAS)  # the ZIP prompt
    gather = _gather(root)
    assert gather.get("input") == "dtmf speech"
    assert gather.get("speechTimeout") == "3"
    assert gather.get("speechModel") == "deepgram_nova-3"
    h.factor(sid(1), TEST_FACTORS["C-1001"]["zip"])
    h.set_pending_tool(sid(1), "authenticate", {"tool": "make_payment"})
    root = h.handoff(sid(1), {"reason": "step-up", "tool": "make_payment", "journal": "jr-2"})
    g = _gather(root)  # SSN4
    assert (g.get("speechTimeout"), g.get("speechModel")) == ("3", "deepgram_nova-3")
    root = h.factor(sid(1), TEST_FACTORS["C-1001"]["ssn4"])
    g = _gather(root)  # OTP
    assert (g.get("speechTimeout"), g.get("speechModel")) == ("3", "deepgram_nova-3")


def test_auto_speech_timeout_drops_the_speech_model() -> None:
    """"auto" is never sent together with a speechModel (the docs rule it out)."""
    h = build(config_kw={**SPOKEN, "speech_timeout_s": "auto"})
    gather = _gather(h.start(sid(3), THOMAS))
    assert gather.get("speechTimeout") == "auto"
    assert "speechModel" not in gather.attrib


@pytest.mark.parametrize("raw, expected", [(None, 3), ("", 3), ("auto", "auto"), (" AUTO ", "auto"),
                                           ("2", 2), ("10", 10)])
def test_speech_timeout_from_env(raw: str | None, expected: int | str) -> None:
    env = {"OWLBANK_RELAY_WS_URL": "wss://relay.example/ws"}
    if raw is not None:
        env["OWLBANK_SPEECH_TIMEOUT"] = raw
    assert FrontDoorConfig.from_env(env).speech_timeout_s == expected


@pytest.mark.parametrize("raw", ["0", "11", "1.5", "fast"])
def test_bad_speech_timeout_from_env_is_refused(raw: str) -> None:
    with pytest.raises(ValueError):
        FrontDoorConfig.from_env({"OWLBANK_RELAY_WS_URL": "wss://relay.example/ws", "OWLBANK_SPEECH_TIMEOUT": raw})


def test_spoken_factors_are_accepted_by_default(fd: Harness) -> None:
    """``accept_spoken_factors`` is on by default, in the config and in the
    environment fallback; only an explicit ``false`` turns it off."""
    assert fd.app.cfg.accept_spoken_factors is True
    base = {"OWLBANK_RELAY_WS_URL": "wss://relay.example/ws"}
    assert FrontDoorConfig.from_env(base).accept_spoken_factors is True
    assert FrontDoorConfig.from_env(
        {**base, "OWLBANK_ACCEPT_SPOKEN_FACTORS": "false"}).accept_spoken_factors is False


def test_keypad_only_factor_gathers_have_no_speech_timeout(keypad: Harness) -> None:
    fd = keypad
    root = fd.start(sid(2), THOMAS)
    gather = _gather(root)
    assert gather.get("input") == "dtmf"
    assert "speechTimeout" not in gather.attrib
    root = fd.factor(sid(2), TEST_FACTORS["C-1001"]["zip"])
    fd.set_pending_tool(sid(2), "authenticate", {"tool": "make_payment"})
    root = fd.handoff(sid(2), {"reason": "step-up", "tool": "make_payment", "journal": "jr-2"})
    gather = _gather(root)
    assert gather.get("input") == "dtmf" and "speechTimeout" not in gather.attrib


def test_yes_no_confirmation_still_listens_for_speech_with_the_timeout(keypad: Harness) -> None:
    fd = keypad
    # Not a verification factor: "dtmf speech" even with spoken factors off, so it carries a
    # timeout: its own (default 2 s), not the digit factors' 3 s.
    to_level3(fd, sid(3))
    gather = _gather(confirm_handoff(fd, sid(3)))
    assert gather.get("input") == "dtmf speech" and gather.get("speechTimeout") == "2"


# --------------------------------------------------------------------------- "Got it."


def test_got_it_precedes_the_second_factor_of_a_step_up(fd: Harness) -> None:
    cat = _cat(fd)
    ack = cat.text("factor.ack", EN)
    assert ack == "Got it."
    _step_up(fd, sid(4))
    first = fd.factor(sid(4), TEST_FACTORS["C-1001"]["ssn4"], follow=False)
    assert fd.state(sid(4)).level == 2
    # the ack on its own while the code is sent, then a redirect to the OTP prompt.
    assert [c.tag for c in first] == ["Say", "Redirect"] and says(first) == [ack]
    root = fd.xml("factor", sid(4))  # the redirect
    assert fd.verify.sent == [(THOMAS, "sms")]  # the code was sent before the prompt
    assert _gather_says(root) == [cat.text("factor.otp", EN)]
    assert says(root) == _gather_says(root)
    # Recorded like other prompts: the journal and the flow's spoken list.
    keys = _prompt_keys(fd, sid(4))
    assert keys[-2:] == ["factor.ack", "factor.otp"]
    spoken = [s["template"] for s in fd.session(sid(4)).flow["spoken"]]
    assert spoken[-2:] == ["factor.ack", "factor.otp"]
    fd.factor(sid(4), OTP)
    env = fd.pending_result(sid(4))
    assert [(s.template, s.text) for s in env.spoken][1:4] == [
        ("factor.ssn4", cat.text("factor.ssn4", EN)), ("factor.ack", ack),
        ("factor.otp", cat.text("factor.otp", EN))]


def test_spanish_step_up_acknowledges_with_entendido(fd: Harness) -> None:
    cat = _cat(fd)
    ack = cat.text("factor.ack", ES)
    assert ack == "Entendido."
    root = _step_up(fd, sid(5), lang_digit="2")
    assert fd.state(sid(5)).language == ES
    assert ack not in says(root)
    root = fd.factor(sid(5), TEST_FACTORS["C-1001"]["ssn4"])  # follows the redirect
    assert says(root) == [ack, cat.text("factor.otp", ES)]
    assert _gather_says(root) == [cat.text("factor.otp", ES)]
    say = root.find("Say")
    assert say is not None and say.get("language") == ES
    assert say.get("voice") == cat.settings(ES)["say"]["voice"]


def test_no_ack_before_the_first_factor_of_a_flow(fd: Harness) -> None:
    en_ack, es_ack = _cat(fd).text("factor.ack", EN), _cat(fd).text("factor.ack", ES)
    # Login: the first factor (ZIP) follows the opening, not a verified factor.
    root = fd.start(sid(6), THOMAS)
    assert says(root) == [_cat(fd).text("factor.zip", EN)]
    # Step-up: the intro then the first factor, no ack.
    root = _step_up(fd, sid(7))
    assert en_ack not in says(root) and es_ack not in says(root)
    assert says(root)[-1] == _cat(fd).text("factor.ssn4", EN)
    assert "factor.ack" not in _prompt_keys(fd, sid(7))


def test_no_ack_when_the_last_factor_completes_the_flow(fd: Harness) -> None:
    _step_up(fd, sid(8))
    fd.factor(sid(8), TEST_FACTORS["C-1001"]["ssn4"])
    root = fd.factor(sid(8), OTP)  # level 3: back to the AI
    assert fd.state(sid(8)).level == 3
    assert _cat(fd).text("factor.ack", EN) not in ET.tostring(root, encoding="unicode")
    assert _prompt_keys(fd, sid(8)).count("factor.ack") == 1  # only before the OTP


# --------------------------------------------------------------------------- menu voice


def _menu(root: ET.Element) -> ET.Element:
    welcome, menu = list(_gather(root))
    assert welcome.get("language") == EN and menu.get("language") == ES
    return menu


def test_the_menu_line_uses_the_catalogs_menu_voice() -> None:
    h = build()

    def other_voice(doc: dict[str, Any]) -> None:
        doc["languages"][ES]["say"]["menuVoice"] = "Polly.Miguel"

    h.app.d.catalog = _catalog_with(other_voice)
    root = h.xml("incoming", sid(9), From=THOMAS)
    welcome = _gather(root)[0]
    assert welcome.get("voice") == h.app.d.catalog.settings(EN)["say"]["voice"]
    assert _menu(root).get("voice") == "Polly.Miguel"


def test_the_menu_line_falls_back_to_the_default_languages_voice() -> None:
    h = build()

    def no_menu_voice(doc: dict[str, Any]) -> None:
        del doc["languages"][ES]["say"]["menuVoice"]

    h.app.d.catalog = _catalog_with(no_menu_voice)
    root = h.xml("incoming", sid(10), From=THOMAS)
    en_voice = h.app.d.catalog.settings(EN)["say"]["voice"]
    assert _menu(root).get("voice") == en_voice
    assert _gather(root)[0].get("voice") == en_voice
