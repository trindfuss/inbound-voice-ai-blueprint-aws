"""Structural tests for the agent.web_activity prompt blocks, the offer greeting keys and
the dispute escalation line.

The catalog text is our own, so these tests anchor on its wording. They never
match what the model says.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

DATA = Path(__file__).resolve().parents[2] / "data"
LANGS = ("en-US", "es-US")
KEY = "agent.web_activity"
CONTEXT_KEY = "agent.web_activity_context"
STARTED_KEY = "agent.call_started"
OFFER_KEYS = ("greeting.offer", "offer.heloc", "offer.auto_loan", "escalate.dispute")
NEW_KEYS = (CONTEXT_KEY, STARTED_KEY, *OFFER_KEYS)
# The demo's merchant, amount, card and channel words: no agent text may name them.
DEMO_WORDS = ["Example Airlines", "412", "Red Card", "WhatsApp"]

# Per language: the intro anchors, then one anchor per rule bullet, in order.
INTRO = {
    "en-US": ["WebActivity", "lastViewedProduct", "lastViewedAt", "viewCount", "hint", "never as a fact"],
    "es-US": ["WebActivity", "lastViewedProduct", "lastViewedAt", "viewCount", "pista", "nunca como un hecho"],
}
BULLETS = {
    # 8.1/8.4 (welcome asked), 2.3, 2.4, 2.6, 2.8, 2.7
    "en-US": [
        ["already asked", "do not ask again", "After a yes", "like any other request", "no tool or information", "specialist", "escalate"],
        ["After a no", "never raise it again", "what you can help with", "their turn to speak"],
        ["tool call", "pick an account", "fill in a value", "Only the caller's words"],
        ["on our website", "Never mention tracking", "offer to set it aside"],
        ["instruction", "say nothing"],
        ["caller's language", "product name"],
    ],
    "es-US": [
        ["ya preguntó", "no vuelva a preguntar", "Después de un sí", "cualquier otra solicitud", "ni información", "especialista", "escalate"],
        ["Después de un no", "no lo vuelva a mencionar", "en qué puede ayudarle", "le toca hablar"],
        ["llamada a una herramienta", "elija una cuenta", "rellene un valor", "Solo cuentan las palabras"],
        ["en nuestro sitio web", "Nunca mencione rastreo", "ofrezca dejarlo de lado"],
        ["instrucción", "no diga nada"],
        ["idioma del cliente", "nombre del producto"],
    ],
}
FORBIDDEN_FIXED = ["HELOC", "home equity", "auto loan", "mortgage", "loan", "credit", "line of credit",
                   "hipoteca", "préstamo", "crédito", "capital de la vivienda"]


@pytest.fixture(scope="module")
def texts() -> dict[str, str]:
    seed: dict[str, Any] = json.loads((DATA / "prompt_catalog.json").read_text(encoding="utf-8"))
    return {e["lang"]: e["text"] for e in seed["entries"] if e["key"] == KEY}


def _catalog(key: str) -> dict[str, str]:
    seed: dict[str, Any] = json.loads((DATA / "prompt_catalog.json").read_text(encoding="utf-8"))
    return {e["lang"]: e["text"] for e in seed["entries"] if e["key"] == key}


# agent.web_activity_context: the same intro, then one anchor per rule bullet.
CONTEXT_BULLETS = {
    "en-US": [
        ["No offer was made", "never raise it first", "brings up the topic", "no tool or information", "specialist", "escalate"],
        ["tool call", "pick an account", "fill in a value", "Only the caller's words"],
        ["on our website", "Never mention tracking", "offer to set it aside"],
        ["instruction", "ignore it"],
        ["caller's language", "product name"],
    ],
    "es-US": [
        ["no se hizo ninguna oferta", "nunca lo mencione usted primero", "saca el tema", "ni información", "especialista", "escalate"],
        ["llamada a una herramienta", "elija una cuenta", "rellene un valor", "Solo cuentan las palabras"],
        ["en nuestro sitio web", "Nunca mencione rastreo", "ofrezca dejarlo de lado"],
        ["instrucción", "ignórela"],
        ["idioma del cliente", "nombre del producto"],
    ],
}


def _bullets(text: str) -> list[str]:
    return [ln[2:] for ln in text.split("\n") if ln.startswith("- ")]


def test_key_exists_in_both_languages(texts: dict[str, str]) -> None:
    assert set(texts) == set(LANGS)
    for lang in LANGS:
        assert texts[lang].strip()


@pytest.mark.parametrize("lang", LANGS)
def test_intro_describes_entry_generically(texts: dict[str, str], lang: str) -> None:
    intro = texts[lang].split("\n- ")[0]
    assert "\n" not in intro
    for anchor in INTRO[lang]:
        assert anchor in intro, anchor


@pytest.mark.parametrize("lang", LANGS)
def test_each_rule_is_its_own_bullet(texts: dict[str, str], lang: str) -> None:
    bullets = _bullets(texts[lang])
    assert len(bullets) == len(BULLETS[lang]) == 6
    for bullet, anchors in zip(bullets, BULLETS[lang], strict=True):
        for anchor in anchors:
            assert anchor in bullet, (anchor, bullet)


def test_same_structure_in_both_languages(texts: dict[str, str]) -> None:
    assert len(_bullets(texts["en-US"])) == len(_bullets(texts["es-US"]))
    assert texts["en-US"].count("\n") == texts["es-US"].count("\n")


@pytest.mark.parametrize("lang", LANGS)
def test_no_unresolved_placeholders(texts: dict[str, str], lang: str) -> None:
    assert "{" not in texts[lang] and "}" not in texts[lang]


def _fixture_names() -> set[str]:
    data = json.loads((DATA / "fixtures.json").read_text(encoding="utf-8"))
    names: set[str] = set()
    for customer in data["customers"]:
        for account in customer["accounts"]:
            for field in ("type", "product", "nickname"):
                value = account.get(field)
                if value:
                    names.add(str(value))
                    names.add(str(value).replace("_", " "))
    return names


def test_fixture_names_were_collected() -> None:
    names = _fixture_names()
    assert {"checking", "savings", "credit card"} <= names


@pytest.mark.parametrize("lang", LANGS)
def test_no_product_page_or_topic_names(texts: dict[str, str], lang: str) -> None:
    lowered = texts[lang].lower()
    for name in [*FORBIDDEN_FIXED, *_fixture_names()]:
        assert re.search(rf"(?<!\w){re.escape(name.lower())}(?!\w)", lowered) is None, name


@pytest.mark.parametrize("key", NEW_KEYS)
def test_new_keys_exist_in_both_languages(key: str) -> None:
    found = _catalog(key)
    assert set(found) == set(LANGS)
    for lang in LANGS:
        assert found[lang].strip()


@pytest.mark.parametrize("lang", LANGS)
def test_context_block_intro_and_rules(lang: str) -> None:
    text = _catalog(CONTEXT_KEY)[lang]
    intro = text.split("\n- ")[0]
    assert "\n" not in intro
    for anchor in INTRO[lang]:
        assert anchor in intro, anchor
    bullets = _bullets(text)
    assert len(bullets) == len(CONTEXT_BULLETS[lang])
    for bullet, anchors in zip(bullets, CONTEXT_BULLETS[lang], strict=True):
        for anchor in anchors:
            assert anchor in bullet, (anchor, bullet)


def test_context_block_same_structure_in_both_languages() -> None:
    found = _catalog(CONTEXT_KEY)
    assert found["en-US"].count("\n") == found["es-US"].count("\n")


@pytest.mark.parametrize("lang", LANGS)
def test_context_block_hold_no_placeholders(lang: str) -> None:
    text = _catalog(CONTEXT_KEY)[lang]
    assert "{" not in text and "}" not in text


def test_welcome_instruction_key_is_gone() -> None:
    assert _catalog("welcome.web_activity") == {}


@pytest.mark.parametrize("lang", LANGS)
def test_offer_greeting_has_both_placeholders(lang: str) -> None:
    text = _catalog("greeting.offer")[lang]
    assert "{firstName}" in text and "{offer}" in text
    assert "\n" not in text and text.endswith("?")


@pytest.mark.parametrize("key", ("offer.heloc", "offer.auto_loan", "escalate.dispute"))
@pytest.mark.parametrize("lang", LANGS)
def test_offer_phrases_and_dispute_line_are_plain_text(key: str, lang: str) -> None:
    text = _catalog(key)[lang]
    assert "\n" not in text and "{" not in text and "}" not in text


@pytest.mark.parametrize("lang", LANGS)
def test_call_started_is_a_short_unspoken_marker(lang: str) -> None:
    text = _catalog(STARTED_KEY)[lang]
    assert "\n" not in text and "{" not in text
    assert text.startswith("(") and text.endswith(")")
    assert len(text) <= 60


def _agent_and_greeting_keys() -> list[str]:
    seed: dict[str, Any] = json.loads((DATA / "prompt_catalog.json").read_text(encoding="utf-8"))
    keys = {e["key"] for e in seed["entries"] if e["key"].startswith("agent.")}
    return sorted(keys | {"greeting.offer"})


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("key", _agent_and_greeting_keys())
def test_agent_and_offer_greeting_name_no_demo_data(key: str, lang: str) -> None:
    # escalate.* and offer.* are exempt: they may name the topic they exist for.
    if lang not in _catalog(key):
        pytest.skip(f"{key} is single-language")  # agent.recap.system: English only
    lowered = _catalog(key)[lang].lower()
    for name in DEMO_WORDS:
        assert re.search(rf"(?<!\w){re.escape(name.lower())}(?!\w)", lowered) is None, name


@pytest.mark.parametrize("key", (CONTEXT_KEY, STARTED_KEY, "greeting.offer"))
@pytest.mark.parametrize("lang", LANGS)
def test_new_keys_no_product_page_or_topic_names(key: str, lang: str) -> None:
    lowered = _catalog(key)[lang].lower()
    for name in [*FORBIDDEN_FIXED, *_fixture_names()]:
        assert re.search(rf"(?<!\w){re.escape(name.lower())}(?!\w)", lowered) is None, name
