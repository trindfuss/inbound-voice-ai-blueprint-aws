"""Structural checks over the masking phrase pools in
data/prompt_catalog.json. Structural only: no test asserts particular words.

The allowed placeholders come straight from ``owlbank.tools_contract`` (each tool's ``enum``
arguments), so the data can't drift from the tool schemas. A tool's pool is its policy
``phrase_key`` itself (if present in the catalog) plus every ``phrase_key.<n>``.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

import pytest

from owlbank import tools_contract

DATA = Path(__file__).resolve().parents[2] / "data"
LANGS = ("en-US", "es-US")
MAX_CHARS = 80
PLACEHOLDER = re.compile(r"\{([^{}]*)\}")


def _load(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


CATALOG = _load("prompt_catalog.json")
POLICY = _load("policy.json")
TEXTS: dict[tuple[str, str], str] = {(e["key"], e["lang"]): e["text"] for e in CATALOG["entries"]}
KEYS = {k for k, _ in TEXTS}


def enums_for(tool: str) -> dict[str, list[str]]:
    """The tool's enumerated arguments and their values, from its input schema."""
    props = tools_contract.tool_schemas()[tool]["properties"]
    return {name: list(p["enum"]) for name, p in props.items() if "enum" in p}


def pool(phrase_key: str) -> list[str]:
    numbered = re.compile(re.escape(phrase_key) + r"\.\d+$")
    keys = [phrase_key] if phrase_key in KEYS else []
    return keys + sorted(k for k in KEYS if numbered.match(k))


def tool_phrase_keys() -> list[tuple[str, str]]:
    """(tool, key) for every mask phrase of a tool: the policy pools and any catalog
    ``mask.<tool>`` / ``mask.<tool>.<n>`` key for a tool in the contract."""
    out: set[tuple[str, str]] = set()
    for tool, row in POLICY["tools"].items():
        if row.get("mask"):
            out |= {(tool, k) for k in pool(row["mask"]["phrase_key"])}
    for tool in tools_contract.tool_schemas():
        out |= {(tool, k) for k in pool(f"mask.{tool}")}
    return sorted(out)


PHRASES = tool_phrase_keys()


def render(text: str, tool: str, values: dict[str, str], lang: str) -> str:
    return PLACEHOLDER.sub(lambda m: TEXTS[(f"arg.{m.group(1)}.{values[m.group(1)]}", lang)], text)


def test_there_are_tool_phrases() -> None:
    assert PHRASES


@pytest.mark.parametrize("tool,key", PHRASES)
def test_phrase_exists_in_every_language(tool: str, key: str) -> None:
    for lang in LANGS:
        assert TEXTS.get((key, lang), "").strip(), (key, lang)


@pytest.mark.parametrize("tool,key", PHRASES)
def test_placeholders_are_enum_arguments_of_the_tool(tool: str, key: str) -> None:
    allowed = set(enums_for(tool))
    for lang in LANGS:
        used = set(PLACEHOLDER.findall(TEXTS[(key, lang)]))
        assert used <= allowed, (key, lang, sorted(used - allowed))


@pytest.mark.parametrize("tool,key", PHRASES)
def test_every_value_of_a_used_placeholder_has_a_word_in_every_language(tool: str, key: str) -> None:
    enums = enums_for(tool)
    for lang in LANGS:
        for arg in set(PLACEHOLDER.findall(TEXTS[(key, lang)])):
            for value in enums[arg]:
                for word_lang in LANGS:
                    word = TEXTS.get((f"arg.{arg}.{value}", word_lang), "")
                    assert word.strip(), (key, f"arg.{arg}.{value}", word_lang)


@pytest.mark.parametrize("tool,key", PHRASES)
def test_every_rendering_is_brace_free_and_short(tool: str, key: str) -> None:
    enums = enums_for(tool)
    for lang in LANGS:
        text = TEXTS[(key, lang)]
        used = sorted(set(PLACEHOLDER.findall(text)))
        for combo in itertools.product(*(enums[a] for a in used)):
            out = render(text, tool, dict(zip(used, combo, strict=True)), lang)
            assert "{" not in out and "}" not in out, (key, lang, out)
            assert len(out) <= MAX_CHARS, (key, lang, len(out))
        assert len(text) <= MAX_CHARS, (key, lang)


@pytest.mark.parametrize("tool", [t for t, r in POLICY["tools"].items() if r.get("mask")])
def test_every_masked_tool_has_a_phrase_without_placeholders(tool: str) -> None:
    """The pool always holds a phrase that renders whatever the arguments."""
    keys = pool(POLICY["tools"][tool]["mask"]["phrase_key"])
    assert keys, tool
    for lang in LANGS:
        assert any(not PLACEHOLDER.search(TEXTS[(k, lang)]) for k in keys), (tool, lang)


@pytest.mark.parametrize("tool", ["get_balance", "list_recent_transactions"])
def test_account_tools_have_a_pool_with_templates(tool: str) -> None:
    """2 or more phrases each, some using an enum placeholder, in both languages."""
    keys = pool(POLICY["tools"][tool]["mask"]["phrase_key"])
    assert len(keys) >= 2, tool
    for lang in LANGS:
        templated = [k for k in keys if PLACEHOLDER.search(TEXTS[(k, lang)])]
        assert len(templated) >= 2, (tool, lang)


def test_arg_words_name_real_enum_values() -> None:
    """Every ``arg.<argument>.<value>`` entry names an enum value some tool can carry, and
    exists in every language."""
    known = {(a, v) for t in tools_contract.tool_schemas() for a, vs in enums_for(t).items() for v in vs}
    # ``escalate``'s ``reason`` values at runtime are the policy's reasons (the bridge renders with
    # ``policy.reasons``), which include system reasons the model can't choose.
    policy = json.loads((Path(__file__).resolve().parents[2] / "data" / "policy.json").read_text())
    known |= {("reason", r) for r in policy["escalation"]["reasons"]}
    arg_keys = {k for k in KEYS if k.startswith("arg.")}
    assert arg_keys
    for key in arg_keys:
        _, arg, value = key.split(".", 2)
        assert (arg, value) in known, key
        for lang in LANGS:
            assert TEXTS.get((key, lang), "").strip(), (key, lang)


def test_no_input_goodbye_exists_in_every_language() -> None:
    for lang in LANGS:
        text = TEXTS.get(("common.goodbye_no_input", lang), "")
        assert text.strip() and not PLACEHOLDER.search(text), lang
