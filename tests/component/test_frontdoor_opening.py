"""The opening. ``/voice/incoming`` welcomes the caller in the default
language, in that language's voice, then offers Spanish in the catalog's Spanish menu voice
(``say.menuVoice``), in one one-digit Gather.
An English caller is not welcomed twice; a Spanish caller hears the welcome again in Spanish.
Expected words come from the prompt catalog."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from owlbank.catalog import PromptCatalog

from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    MARIA, TEST_FACTORS, THOMAS, Harness, fd, relay, sid,
)

EN, ES = "en-US", "es-US"


def _cat(h: Harness) -> PromptCatalog:
    return h.app.d.catalog


def _voice(h: Harness, lang: str) -> str:
    return _cat(h).settings(lang)["say"]["voice"]


def _says(root: ET.Element) -> list[ET.Element]:
    return list(root.iter("Say"))


def test_incoming_gather_nests_the_welcome_then_the_menu_in_the_menu_voice(fd: Harness) -> None:
    cat = _cat(fd)
    assert cat.default_language == EN
    root = fd.xml("incoming", sid(1), From=THOMAS)
    gather = root.find("Gather")
    assert gather is not None and gather.get("numDigits") == "1"
    nested = list(gather)
    assert [e.tag for e in nested] == ["Say", "Say"]
    welcome, menu = nested
    assert welcome.text == cat.text("greeting.generic", EN) == "Welcome to OwlBank. This call may be recorded and transcribed for quality and training."
    assert menu.text == cat.text("language.select", ES) == "Para español, oprima dos."
    en_voice = _voice(fd, EN)
    assert en_voice == "ElevenLabs.UgBBYS2sOqTuMpoF3BR0"
    assert welcome.get("voice") == en_voice
    menu_voice = _cat(fd).settings(ES)["say"]["menuVoice"]
    assert menu_voice == "Polly.Pedro-Neural"
    assert menu.get("voice") == menu_voice
    assert welcome.get("language") == EN
    assert menu.get("language") == ES  # pronunciation stays Spanish
    # Nothing is spoken outside the Gather.
    assert len(_says(root)) == 2


def test_english_caller_is_not_welcomed_twice_before_the_zip_prompt(fd: Harness) -> None:
    cat = _cat(fd)
    root = fd.start(sid(1), THOMAS)  # no digit: the default language
    assert fd.state(sid(1)).language == EN
    texts = [e.text for e in _says(root)]
    assert texts == [cat.text("factor.zip", EN)]
    assert cat.text("greeting.generic", EN) not in texts


def test_spanish_caller_hears_the_welcome_in_spanish_then_the_zip_prompt(fd: Harness) -> None:
    cat = _cat(fd)
    root = fd.start(sid(1), MARIA, lang_digit="2")
    assert fd.state(sid(1)).language == ES
    says = _says(root)
    assert [e.text for e in says] == [cat.text("greeting.generic", ES), cat.text("factor.zip", ES)]
    assert says[0].text == "Bienvenido a OwlBank. Esta llamada puede ser grabada y transcrita con fines de calidad y capacitación."
    es_voice = _voice(fd, ES)
    assert es_voice != _voice(fd, EN)
    for e in says:
        assert e.get("voice") == es_voice and e.get("language") == ES


def test_personal_greeting_after_verification(fd: Harness) -> None:
    cat = _cat(fd)
    root = fd.login_thomas(sid(1))
    expected = cat.text("greeting.personal", EN, firstName="Thomas")
    assert expected == "Hi Thomas. I am the OwlBank virtual agent. How can I help you today?"
    assert relay(root).get("welcomeGreeting") == expected


def test_spanish_personal_greeting_after_verification(fd: Harness) -> None:
    cat = _cat(fd)
    fd.start(sid(1), MARIA, lang_digit="2")
    root = fd.factor(sid(1), TEST_FACTORS["C-1004"]["zip"])
    assert relay(root).get("welcomeGreeting") == cat.text("greeting.personal", ES, firstName="María")


def test_journal_records_both_opening_prompts_at_incoming(fd: Harness) -> None:
    cat = _cat(fd)
    fd.xml("incoming", sid(1), From=THOMAS)
    prompts = [(e["prompt_key"], e["rendered_text"]) for e in fd.journal(sid(1))
               if e["type"] == "prompt"]
    assert prompts == [("greeting.generic", cat.text("greeting.generic", EN)),
                       ("language.select", cat.text("language.select", ES))]
