"""``owlbank.masking_config``: settings, tool masks, placeholders, pools,
phrase validation and rendering."""

from __future__ import annotations

from typing import Any

import pytest

from owlbank.catalog import PromptCatalog
from owlbank.masking_config import (
    DEFAULT_SETTINGS,
    MaskingSettings,
    Problem,
    ToolMask,
    parse_masking,
    parse_tool_mask,
    placeholders_for,
    pool_keys,
    render,
    validate_catalog_for_tool,
    validate_phrase,
    words_for,
)
from owlbank.policy import PolicySectionError
from owlbank.tools_contract import (
    DEFAULT_CARDS,
    DEFAULT_ESCALATION_REASONS,
    GATEWAY_LEVEL0_TOOLS,
    INLINE_TOOLS,
)

LANGS = ("en-US", "es-US")


def catalog(entries: dict[tuple[str, str], str]) -> PromptCatalog:
    return PromptCatalog.from_dict({
        "defaultLanguage": "en-US", "languages": {lang: {} for lang in LANGS},
        "entries": [{"key": k, "lang": lang, "text": t} for (k, lang), t in entries.items()]})


WORDS = {("arg.account.checking", "en-US"): "checking", ("arg.account.checking", "es-US"): "corriente",
         ("arg.account.savings", "en-US"): "savings", ("arg.account.savings", "es-US"): "ahorros"}


# --------------------------------------------------------------------------- settings


def test_defaults_match_the_design() -> None:
    s = DEFAULT_SETTINGS
    assert (s.after_ms, s.tool_after_ms, s.handoff_wait_ms, s.end_mode, s.chars_per_s,
            s.silence_timeout_ms, s.silence_max_repeats, s.silence_action) == \
        (1000, 0, 2500, "wait", 15.0, 4000, 2, "escalate")  # two repeats
    assert parse_masking(None, strict=True) == s
    assert parse_masking({}, strict=False) == (s, [])


def test_todays_masking_item_still_parses() -> None:
    raw = {"after_ms": 1500, "handoff_wait_ms": 2500, "end_mode": "wait", "chars_per_s": 15}
    assert parse_masking(raw, strict=True) == MaskingSettings(1500, 2500, "wait", 15.0)


def test_full_item_round_trips() -> None:
    s = MaskingSettings(0, 10000, "immediate", 12.5, 0, 0, 5, "hangup")
    assert parse_masking(s.to_doc(), strict=True) == s
    assert s.to_doc()["chars_per_s"] == "12.5"


@pytest.mark.parametrize("field,value", [
    ("tool_after_ms", 10001), ("silence_timeout_ms", -1), ("silence_max_repeats", 6),
    ("silence_action", "transfer"), ("after_ms", "2000"),
])
def test_bad_field_strict_names_it_lenient_defaults_it(field: str, value: Any) -> None:
    raw = {"after_ms": 100, field: value}
    with pytest.raises(PolicySectionError) as exc:
        parse_masking(raw, strict=True)
    assert exc.value.field == f"masking.{field}"
    got, problems = parse_masking(raw, strict=False)
    assert getattr(got, field) == getattr(DEFAULT_SETTINGS, field)
    assert [p.field for p in problems] == [f"masking.{field}"]


def test_invalid_silence_action_falls_back_to_escalate() -> None:
    got, problems = parse_masking({"silence_action": "hangup!"}, strict=False)
    assert got.silence_action == "escalate" and problems


# --------------------------------------------------------------------------- tool masks


def test_tool_mask_shapes() -> None:
    assert parse_tool_mask("t", None, strict=True) is None
    assert parse_tool_mask("t", {"phrase_key": "mask.t"}, strict=True) == ToolMask("mask.t")
    assert parse_tool_mask("t", {"phrase_key": "mask.t", "after_ms": 0, "handoff_wait_ms": 10000},
                           strict=True) == ToolMask("mask.t", 0, 10000)
    # null delays are "use the policy-wide value"
    assert parse_tool_mask("t", {"phrase_key": "mask.t", "after_ms": None}, strict=True) == \
        ToolMask("mask.t")
    assert ToolMask("mask.t", 5).to_doc() == {"phrase_key": "mask.t", "after_ms": 5}


@pytest.mark.parametrize("raw,field,expected", [
    ("mask.t", "tools.t.mask", None),
    ({}, "tools.t.mask.phrase_key", None),
    ({"phrase_key": "tool.t"}, "tools.t.mask.phrase_key", None),
    ({"phrase_key": "mask.{x}"}, "tools.t.mask.phrase_key", None),
    ({"phrase_key": "mask.t", "after_ms": -1}, "tools.t.mask.after_ms", ToolMask("mask.t")),
    ({"phrase_key": "mask.t", "handoff_wait_ms": True}, "tools.t.mask.handoff_wait_ms",
     ToolMask("mask.t")),
    ({"phrase_key": "mask.t", "x": 1}, "tools.t.mask.x", ToolMask("mask.t")),
])
def test_tool_mask_problems(raw: Any, field: str, expected: ToolMask | None) -> None:
    with pytest.raises(PolicySectionError) as exc:
        parse_tool_mask("t", raw, strict=True)
    assert exc.value.field == field
    got, problems = parse_tool_mask("t", raw, strict=False)
    assert got == expected and problems[0].field == field


# --------------------------------------------------------------------------- placeholders, pools


def test_placeholders_for_every_contract_tool() -> None:
    assert placeholders_for("get_balance") == {"account": ["checking", "savings"]}
    assert placeholders_for("list_recent_transactions") == {"account": ["checking", "savings"]}
    assert placeholders_for("make_payment") == {"from_account": ["checking", "savings"],
                                                "to": list(DEFAULT_CARDS)}
    assert placeholders_for("transfer_funds") == {"from_account": ["checking", "savings"],
                                                  "to_account": ["checking", "savings"]}
    assert placeholders_for("change_address") == {}
    assert placeholders_for("escalate") == {"reason": list(DEFAULT_ESCALATION_REASONS)}
    assert placeholders_for("escalate", reasons=["no_input"]) == {"reason": ["no_input"]}
    assert "tool" in placeholders_for("authenticate")
    for t in GATEWAY_LEVEL0_TOOLS:
        assert placeholders_for(t) == {}
    assert placeholders_for("no_such_tool") == {}
    for t in INLINE_TOOLS:  # never a free-text, numeric or amount argument
        assert not {"amount", "summary", "street", "limit"} & set(placeholders_for(t))


def test_pool_keys() -> None:
    keys = ["mask.get_balance.10", "mask.get_balance.2", "mask.get_balance", "mask.get_balance.x",
            "mask.get_balance.01", "mask.get_balance.1", "mask.get_balancer.1", "mask.other.1"]
    assert pool_keys("mask.get_balance", keys) == [
        "mask.get_balance", "mask.get_balance.1", "mask.get_balance.2", "mask.get_balance.10"]
    assert pool_keys("mask.t", ["mask.t.2"]) == ["mask.t.2"]
    assert pool_keys("mask.t", []) == []


# --------------------------------------------------------------------------- validation


def test_valid_template_phrase() -> None:
    cat = catalog(WORDS)
    assert validate_phrase("get_balance", "Checking your {account} balance now.", cat, LANGS) == []
    assert validate_phrase("change_address", "Updating your address now.", cat, LANGS) == []


@pytest.mark.parametrize("text,needle", [
    ("Checking your {amount} now.", "{amount}"),
    ("Checking your {to} now.", "{to}"),
    ("Checking {account balance.", "'{'"),
    ("Checking } now.", "'{'"),
    ("Checking {{account}} now.", "'{'"),
    ("", "non-empty"),
])
def test_invalid_phrases(text: str, needle: str) -> None:
    problems = validate_phrase("get_balance", text, catalog(WORDS), LANGS)
    assert problems and any(needle in p.message for p in problems)


def test_missing_word_in_one_language_is_a_problem() -> None:
    words = {k: v for k, v in WORDS.items() if k != ("arg.account.savings", "es-US")}
    problems = validate_phrase("get_balance", "Your {account} balance.", catalog(words), LANGS)
    assert problems == [Problem("phrase", "no catalog word arg.account.savings in es-US")]


def test_validate_catalog_for_tool_covers_the_pool_in_every_language() -> None:
    entries = {**WORDS, ("mask.get_balance.1", "en-US"): "Checking your {account} balance.",
               ("mask.get_balance.1", "es-US"): "Reviso su cuenta de {account}.",
               ("mask.get_balance.2", "en-US"): "One moment for your {acct}.",
               ("mask.get_balance.2", "es-US"): "Un momento."}
    problems = validate_catalog_for_tool("get_balance", ToolMask("mask.get_balance"),
                                         catalog(entries), LANGS)
    assert [p.field for p in problems] == ["catalog.mask.get_balance.2[en-US]"]
    entries.pop(("mask.get_balance.2", "es-US"))
    problems = validate_catalog_for_tool("get_balance", ToolMask("mask.get_balance"),
                                         catalog(entries), LANGS)
    assert Problem("catalog.mask.get_balance.2[es-US]", "missing") in problems
    assert validate_catalog_for_tool("get_balance", ToolMask("mask.none"), catalog(entries),
                                     LANGS)[0].field == "tools.get_balance.mask.phrase_key"


# --------------------------------------------------------------------------- rendering


def test_render() -> None:
    cat = catalog(WORDS)
    en, es = words_for("get_balance", cat, "en-US"), words_for("get_balance", cat, "es-US")
    t = "Checking your {account} balance now."
    assert render("get_balance", t, {"account": "savings"}, en) == "Checking your savings balance now."
    assert render("get_balance", "Su cuenta de {account}.", {"account": "savings"}, es) == \
        "Su cuenta de ahorros."
    assert render("get_balance", "One moment.", {}, en) == "One moment."
    assert render("get_balance", "One moment.", None, en) == "One moment."


@pytest.mark.parametrize("tool,text,args,words", [
    ("get_balance", "Your {account}.", {"account": "brokerage"}, {"arg.account.brokerage": "b"}),
    ("get_balance", "Your {account}.", {}, {"arg.account.savings": "savings"}),
    ("get_balance", "Your {account}.", {"account": "savings"}, {}),
    ("get_balance", "Your {account}.", {"account": "savings"}, {"arg.account.savings": " "}),
    ("get_balance", "Your {account}.", {"account": "savings"}, {"arg.account.savings": "{x}"}),
    ("get_balance", "Your {limit}.", {"limit": "3"}, {"arg.limit.3": "three"}),
    ("list_recent_transactions", "Your {limit}.", {"limit": 3}, {"arg.limit.3": "three"}),
    ("faq_lookup", "On {question}.", {"question": "x"}, {"arg.question.x": "x"}),
    ("get_balance", "Your {account} {.", {"account": "savings"}, {"arg.account.savings": "s"}),
    ("get_balance", "Your {account}.", ["savings"], {"arg.account.savings": "s"}),
])
def test_render_returns_none_when_it_cannot_fill_everything(
        tool: str, text: str, args: Any, words: dict[str, str]) -> None:
    assert render(tool, text, args, words) is None
