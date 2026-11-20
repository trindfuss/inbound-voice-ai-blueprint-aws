"""scripts/seed.py's offer cross-check: every offer in
``policy.json``'s ``web_offers`` has ``offer.<id>`` in every catalog language, else exit 2."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from scripts import seed
from scripts._common import DATA_DIR, ScriptError


def load(name: str) -> Any:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


POLICY = load("policy.json")
FIXTURES = load("fixtures.json")
CATALOG = load("prompt_catalog.json")

LANGS = {"en-US": {}, "es-US": {}}


def policy_doc(*ids: str) -> dict[str, Any]:
    return {"web_offers": {"offers": [{"id": i, "values": [i.upper()]} for i in ids]}}


def catalog_doc(*entries: tuple[str, str], langs: dict[str, Any] = LANGS) -> dict[str, Any]:
    return {"languages": langs,
            "entries": [{"key": k, "lang": lang, "text": f"{k} in {lang}"} for k, lang in entries]}


def test_every_offer_in_every_language_passes() -> None:
    seed.check_offers(policy_doc("heloc", "auto_loan"), catalog_doc(
        ("offer.heloc", "en-US"), ("offer.heloc", "es-US"),
        ("offer.auto_loan", "en-US"), ("offer.auto_loan", "es-US")))


def test_an_extra_phrase_without_an_offer_is_allowed() -> None:
    seed.check_offers(policy_doc("heloc"), catalog_doc(
        ("offer.heloc", "en-US"), ("offer.heloc", "es-US"),
        ("offer.retired", "en-US"), ("offer.retired", "es-US")))


def test_no_section_or_no_offers_passes() -> None:
    seed.check_offers({}, catalog_doc())
    seed.check_offers(policy_doc(), catalog_doc())


def test_a_missing_phrase_is_refused() -> None:
    with pytest.raises(ScriptError, match=r"offer\.auto_loan \(en-US\), offer\.auto_loan \(es-US\)"):
        seed.check_offers(policy_doc("heloc", "auto_loan"), catalog_doc(
            ("offer.heloc", "en-US"), ("offer.heloc", "es-US")))


def test_a_missing_language_is_refused() -> None:
    with pytest.raises(ScriptError) as err:
        seed.check_offers(policy_doc("heloc"), catalog_doc(("offer.heloc", "en-US")))
    assert "offer.heloc (es-US)" in str(err.value) and "en-US" not in str(err.value)


def test_a_blank_phrase_counts_as_missing() -> None:
    catalog = catalog_doc(("offer.heloc", "en-US"), ("offer.heloc", "es-US"))
    catalog["entries"][1]["text"] = "  "
    with pytest.raises(ScriptError, match=r"offer\.heloc \(es-US\)"):
        seed.check_offers(policy_doc("heloc"), catalog)


def test_the_seed_files_pass_the_offer_cross_check() -> None:
    seed.check_offers(POLICY, CATALOG)


def test_build_plan_refuses_a_missing_offer_phrase() -> None:
    catalog = copy.deepcopy(CATALOG)
    catalog["entries"] = [e for e in catalog["entries"] if (e["key"], e["lang"]) != ("offer.heloc", "es-US")]
    with pytest.raises(ScriptError, match=r"offer\.heloc \(es-US\)"):
        seed.build_plan(FIXTURES, POLICY, catalog, stage="prod")


def test_build_plan_refuses_an_invalid_offer_list() -> None:
    policy = copy.deepcopy(POLICY)
    policy["web_offers"]["offers"].append({"id": "dup", "values": ["heloc"]})
    with pytest.raises(Exception, match=r"web_offers\.offers\[2\]\.values\[0\]"):
        seed.build_plan(FIXTURES, policy, CATALOG, stage="prod")


def test_seed_main_exits_2_on_a_missing_offer_phrase(tmp_path: Any) -> None:
    catalog = copy.deepcopy(CATALOG)
    catalog["entries"] = [e for e in catalog["entries"] if not e["key"].startswith("offer.auto_loan")]
    for name, doc in (("fixtures.json", FIXTURES), ("policy.json", POLICY), ("prompt_catalog.json", catalog)):
        (tmp_path / name).write_text(json.dumps(doc), encoding="utf-8")
    assert seed.main(["--stage", "prod", "--data-dir", str(tmp_path)]) == 2


def test_seed_main_exits_2_on_an_invalid_offer_list(tmp_path: Any) -> None:
    policy = copy.deepcopy(POLICY)
    policy["web_offers"]["offers"].append({"id": "dup", "values": ["heloc"]})
    for name, doc in (("fixtures.json", FIXTURES), ("policy.json", policy), ("prompt_catalog.json", CATALOG)):
        (tmp_path / name).write_text(json.dumps(doc), encoding="utf-8")
    assert seed.main(["--stage", "prod", "--data-dir", str(tmp_path)]) == 2
