"""Keypad shortcut catalog entries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

DATA = Path(__file__).resolve().parents[2] / "data"
KEYS = ("keypad.yes", "keypad.no", "keypad.hint", "keypad.agent_summary")
LANGS = ("en-US", "es-US")


@pytest.fixture(scope="module")
def entries() -> dict[tuple[str, str], dict[str, Any]]:
    seed = json.loads((DATA / "prompt_catalog.json").read_text(encoding="utf-8"))
    return {(e["key"], e["lang"]): e for e in seed["entries"]}


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("key", KEYS)
def test_keypad_keys_exist_without_placeholders(
    entries: dict[tuple[str, str], dict[str, Any]], key: str, lang: str
) -> None:
    entry = entries[(key, lang)]
    assert entry["text"].strip()
    assert not re.search(r"[{}]", entry["text"])
    assert entry["status"] == "published"


def test_agent_summary_identical_in_both_languages(
    entries: dict[tuple[str, str], dict[str, Any]],
) -> None:
    assert entries[("keypad.agent_summary", "en-US")]["text"] == (
        entries[("keypad.agent_summary", "es-US")]["text"]
    )


@pytest.mark.parametrize("lang", LANGS)
def test_every_keypad_twin_has_a_say_or_enter_base_and_never_invites_speech(
    entries: dict[tuple[str, str], dict[str, Any]], lang: str
) -> None:
    """A ``<base>.keypad`` twin has a published base in the same
    language; the base says "say or enter" (diga/dígalo/decirlo), the twin does not."""
    twins = sorted(k for k, lg in entries if lg == lang and k.endswith(".keypad")
                   and not k.startswith("keypad."))
    assert twins
    for twin in twins:
        base = twin.removesuffix(".keypad")
        assert (base, lang) in entries, twin
        assert re.search(r"\b(say|diga|dígalo|decirlo)\b", entries[(base, lang)]["text"], re.I), base
        assert not re.search(r"\b(say|diga|dígalo|decirlo)\b", entries[(twin, lang)]["text"], re.I), twin
