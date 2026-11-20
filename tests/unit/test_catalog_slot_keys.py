"""The model-assisted slot keys in the prompt catalog, in every language.

Structural checks only: each key exists in en-US and es-US with exactly the placeholders the
front door fills, and the seed catalog still loads and renders them. The wording itself is
owned by the catalog file (and, for Spanish, a native speaker on the bank's side).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from owlbank.catalog import PLACEHOLDER, PromptCatalog

DATA = Path(__file__).resolve().parents[2] / "data"
LANGS = ("en-US", "es-US")

# Key -> the exact placeholder set every language's text must use.
SLOT_KEYS: dict[str, frozenset[str]] = {
    # Spoken names of the card options, the card slot's prompt and label.
    "option.card.red": frozenset(),
    "option.card.purple": frozenset(),
    "option.card.rewards": frozenset(),
    "prompt.card": frozenset({"options"}),   # The caller's cards
    "label.card": frozenset(),
    # Follow-up questions.
    "slot.confirm_option": frozenset({"option"}),
    "slot.confirm_digits": frozenset({"digits"}),
    "slot.disambiguate": frozenset({"options"}),
    "slot.not_available_one": frozenset({"named", "option"}),
    "slot.not_available_many": frozenset({"named", "options"}),
    # Correction confirmations.
    "slot.corrected": frozenset({"value"}),
    "slot.corrected_sensitive": frozenset({"label"}),
    # Keypad menus for a keypad-only slot's follow-ups: a yes/no menu, and a choice menu
    # built from one item per option.
    "slot.keypad_yes_no": frozenset(),
    "slot.keypad_choice": frozenset({"menu"}),
    "slot.keypad_option": frozenset({"digit", "option"}),
    # Factor and login questions: "say or enter" wording and the ``.keypad`` twin of each.
    **{k + sfx: frozenset() for k in ("factor.zip", "factor.ssn4", "factor.otp", "login.phone_number",
                                      "login.phone_number_retry") for sfx in ("", ".keypad")},
}

# Placeholders that would carry a slot's value into speech.
VALUE_PLACEHOLDERS = frozenset({"value", "digits", "option", "options", "named", "menu", "digit"})


@pytest.fixture(scope="module")
def doc() -> dict[str, Any]:
    return json.loads((DATA / "prompt_catalog.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def texts(doc: dict[str, Any]) -> dict[tuple[str, str], str]:
    return {(e["key"], e["lang"]): e["text"] for e in doc["entries"]
            if e["status"] == "published"}


@pytest.mark.parametrize("key", sorted(SLOT_KEYS))
@pytest.mark.parametrize("lang", LANGS)
def test_key_present_in_every_language(texts: dict[tuple[str, str], str], key: str,
                                       lang: str) -> None:
    assert texts.get((key, lang), "").strip(), f"{key} missing or empty in {lang}"


@pytest.mark.parametrize("key", sorted(SLOT_KEYS))
def test_placeholders_are_exactly_the_expected_set(texts: dict[tuple[str, str], str],
                                                   key: str) -> None:
    for lang in LANGS:
        found = frozenset(PLACEHOLDER.findall(texts[(key, lang)]))
        assert found == SLOT_KEYS[key], (key, lang, sorted(found))


def test_slot_keys_are_not_single_language(doc: dict[str, Any]) -> None:
    assert not set(SLOT_KEYS) & set(doc["singleLanguageKeys"])


@pytest.mark.parametrize("lang", LANGS)
def test_corrected_sensitive_never_carries_a_value(texts: dict[tuple[str, str], str],
                                                   lang: str) -> None:
    found = set(PLACEHOLDER.findall(texts[("slot.corrected_sensitive", lang)]))
    assert not found & VALUE_PLACEHOLDERS
    assert not any(ch.isdigit() for ch in texts[("slot.corrected_sensitive", lang)])


@pytest.mark.parametrize("lang", LANGS)
def test_card_spoken_names_are_distinct(texts: dict[tuple[str, str], str], lang: str) -> None:
    names = [texts[(f"option.card.{v}", lang)].casefold() for v in ("red", "purple", "rewards")]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("lang", LANGS)
def test_seed_catalog_loads_and_renders_every_slot_key(doc: dict[str, Any], lang: str) -> None:
    cat = PromptCatalog.from_dict(doc)
    assert set(SLOT_KEYS) <= cat.keys
    for key, names in SLOT_KEYS.items():
        out = cat.text(key, lang, **{n: f"<{n}>" for n in names})
        assert PLACEHOLDER.search(out) is None, (key, lang)
        for n in names:
            assert f"<{n}>" in out, (key, lang, n)


@pytest.mark.parametrize("lang", LANGS)
def test_keypad_choice_menu_composes(doc: dict[str, Any], lang: str) -> None:
    """A choice menu is ``slot.keypad_choice`` filled with ``slot.keypad_option`` items."""
    cat = PromptCatalog.from_dict(doc)
    items = [cat.text("slot.keypad_option", lang, digit=str(i),
                      option=cat.text(f"option.card.{v}", lang))
             for i, v in enumerate(("red", "purple"), start=1)]
    out = cat.text("slot.keypad_choice", lang, menu=", ".join(items))
    assert PLACEHOLDER.search(out) is None
    for i, v in enumerate(("red", "purple"), start=1):
        assert str(i) in out and cat.text(f"option.card.{v}", lang) in out
