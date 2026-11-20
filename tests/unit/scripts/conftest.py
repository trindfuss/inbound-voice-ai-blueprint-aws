"""Shared fixtures for the script tests.

``scripts/seed.py`` refuses unvalidated ``agent.system.*`` texts (model failover), and the
repository's records are empty until the prompts are validated. The seed tests here are about
other rules, so every seed run reads records that validate the catalog it seeds; only the inputs are
replaced, the gate itself runs. The gate has its own tests (``tests/unit/test_seed_prompt_gate.py``,
``tests/property/test_seed_prompt_gate_props.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from owlbank.agent_prompt import SYSTEM_KEY_PREFIX, prompt_hash
from owlbank.catalog import PromptCatalog
from owlbank.prompt_validation import CaseTally, Record, Suite, ValidationFile
from scripts import seed

#: A case file with no cases: any record with its hash has full runs.
SUITE = Suite(runs=5, cases=(), hash="5" * 64)


def validating_records(catalog_doc: dict) -> ValidationFile:
    """A record with the current prompt hash for every published ``agent.system.*`` text."""
    catalog = PromptCatalog.from_dict(catalog_doc)
    prefix = SYSTEM_KEY_PREFIX + "."
    keys = {(e["key"][len(prefix):], e["lang"]) for e in catalog_doc["entries"]
            if e["key"].startswith(prefix) and e.get("status", "published") == "published"}
    return ValidationFile(header={}, records=tuple(
        Record(model_key=mk, lang=lang, prompt_hash=prompt_hash(catalog, mk, lang), suite_hash=SUITE.hash,
               params_hash="6" * 64, model_id=f"us.anthropic.claude-{mk}", date="2026-11-02", cases={"C1": CaseTally(5, 5)})
        for mk, lang in sorted(keys)))


@pytest.fixture(autouse=True)
def validated_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    def load(data_dir: Path) -> tuple[ValidationFile, Suite]:
        doc = json.loads((data_dir / "prompt_catalog.json").read_text(encoding="utf-8"))
        return validating_records(doc), SUITE

    monkeypatch.setattr(seed, "load_prompt_validation", load)
