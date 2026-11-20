"""scripts/seed.py's briefing-label cross-check: every policy level and escalation reason has its
spoken ``escalate.briefing.*`` label in every catalog language, none holds a digit, else exit 2."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from owlbank.catalog import PromptCatalog
from owlbank.tools_contract import DEFAULT_ESCALATION_REASONS, FALLBACK_REASON
from scripts import seed
from scripts._common import DATA_DIR, ScriptError

POLICY = json.loads((DATA_DIR / "policy.json").read_text(encoding="utf-8"))
CATALOG = json.loads((DATA_DIR / "prompt_catalog.json").read_text(encoding="utf-8"))
LANGS = ("en-US", "es-US")
LEVELS = range(POLICY["max_level"] + 1)
REASONS = (*POLICY["escalation"]["reasons"], "default")
LABEL_KEYS = ([f"escalate.briefing.level.{n}" for n in LEVELS]
              + [f"escalate.briefing.reason.{r}" for r in REASONS])


def _without(key: str, lang: str) -> dict[str, Any]:
    catalog = copy.deepcopy(CATALOG)
    catalog["entries"] = [e for e in catalog["entries"] if (e["key"], e["lang"]) != (key, lang)]
    return catalog


def test_the_seed_files_pass_the_cross_check() -> None:
    seed.check_briefing_labels(POLICY, CATALOG)


def test_the_defaults_and_the_fallback_reason_have_labels() -> None:
    assert {*DEFAULT_ESCALATION_REASONS, FALLBACK_REASON} <= set(POLICY["escalation"]["reasons"])


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("key", LABEL_KEYS)
def test_a_missing_label_is_refused(key: str, lang: str) -> None:
    with pytest.raises(ScriptError, match=rf"{key} \({lang}\)"):
        seed.check_briefing_labels(POLICY, _without(key, lang))


def test_a_new_level_and_a_new_reason_need_labels() -> None:
    policy = copy.deepcopy(POLICY)
    policy["max_level"] = 4
    policy["levels"]["4"] = policy["levels"]["3"]
    policy["escalation"]["reasons"]["lost_card"] = dict(policy["escalation"]["reasons"]["complaint"])
    with pytest.raises(ScriptError) as err:
        seed.check_briefing_labels(policy, CATALOG)
    for needed in ("level.4", "reason.lost_card"):
        for lang in LANGS:
            assert f"escalate.briefing.{needed} ({lang})" in str(err.value)


def test_a_label_with_a_digit_is_refused() -> None:
    catalog = copy.deepcopy(CATALOG)
    for e in catalog["entries"]:
        if (e["key"], e["lang"]) == ("escalate.briefing.level.2", "es-US"):
            e["text"] = "2"
    with pytest.raises(ScriptError, match=r"level\.2 \(es-US\)"):
        seed.check_briefing_labels(POLICY, catalog)


def test_a_policy_without_levels_or_reasons_needs_nothing() -> None:
    seed.check_briefing_labels({}, {"languages": {"en-US": {}}, "entries": []})


@pytest.mark.parametrize("lang", LANGS)
def test_no_spoken_briefing_text_holds_a_digit(lang: str) -> None:
    """Every spoken briefing text in the catalog, labels and templates, written with words only."""
    catalog = PromptCatalog.from_dict(CATALOG)
    keys = sorted(k for k in catalog.keys if k.startswith("escalate.briefing."))
    assert {*LABEL_KEYS, "escalate.briefing.lead", "escalate.briefing.fallback",
            "escalate.briefing.clause"} <= set(keys)
    for key in keys:
        text = catalog.template(key, lang)
        assert text.strip(), key
        assert not any(c.isdigit() for c in text), (key, lang)


def test_the_briefing_prompt_is_english_only_and_asks_for_no_digits() -> None:
    catalog = PromptCatalog.from_dict(CATALOG)
    assert catalog.has("agent.briefing.system", "en-US")
    assert not catalog.has("agent.briefing.system", "es-US")
    assert "no digits" in catalog.template("agent.briefing.system", "en-US")


def test_a_missing_label_at_runtime_falls_back_to_the_default_language(
        caplog: pytest.LogCaptureFixture) -> None:
    catalog = PromptCatalog.from_dict(_without("escalate.briefing.level.1", "es-US"))
    with caplog.at_level("WARNING", logger="owlbank.catalog"):
        assert catalog.text("escalate.briefing.level.1", "es-US") == "one"
    assert "falling back" in caplog.text
