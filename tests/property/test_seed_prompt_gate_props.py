"""The seed refuses exactly the unvalidated
``agent.system.*`` texts, and ``--allow-unvalidated-prompt`` works only at stage dev.

The oracle: each published ``agent.system.<model key>`` entry's ``(key, lang)`` for which
``ValidationFile.is_validated`` is false (current prompt and suite hashes and full runs; no case
file: every one).
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from owlbank.agent_prompt import SYSTEM_KEY_PREFIX
from owlbank.catalog import PromptCatalog
from owlbank.prompt_validation import Suite
from scripts import seed
from scripts._common import ScriptError
from tests.prompt_gate_strategies import SUITE, catalogs, records_for

PREFIX = SYSTEM_KEY_PREFIX + "."


@st.composite
def cases(draw: Any) -> tuple[dict[str, Any], Any, Suite | None, str, bool]:
    doc = draw(catalogs())
    return (doc, draw(records_for(doc)), draw(st.sampled_from([SUITE, None])),
            draw(st.sampled_from(["dev", "test", "prod", "DEV", "staging"])), draw(st.booleans()))


def expected_unvalidated(doc: dict[str, Any], records: Any, suite: Suite | None) -> set[tuple[str, str]]:
    catalog = PromptCatalog.from_dict(doc)
    published = {(e["key"], e["lang"]) for e in doc["entries"]
                 if e["key"].startswith(PREFIX) and e["status"] == "published"}
    return {(key, lang) for key, lang in published
            if suite is None or not records.is_validated(catalog, key[len(PREFIX):], lang, suite)}


@given(cases())
def test_the_seed_refuses_exactly_the_unvalidated_texts_and_the_override_works_only_in_dev(case: Any) -> None:
    doc, records, suite, stage, allow = case
    want = expected_unvalidated(doc, records, suite)
    if allow and stage != "dev":
        with pytest.raises(ScriptError, match="only with --stage dev"):
            seed.check_prompt_gate(doc, records, suite, stage=stage, allow_unvalidated=allow)
        return
    if want and not allow:
        with pytest.raises(ScriptError) as exc:
            seed.check_prompt_gate(doc, records, suite, stage=stage, allow_unvalidated=allow)
        named = (str(exc.value).split("validation record: ", 1)[1].split(". Run", 1)[0]
                 .removesuffix(f" (no {seed.PROMPT_SUITE_FILE} yet)"))
        assert {tuple(n.removesuffix(")").split(" (")) for n in named.split(", ")} == want
        return
    gate = seed.check_prompt_gate(doc, records, suite, stage=stage, allow_unvalidated=allow)
    assert set(gate.unvalidated) == want
    assert len(gate.unvalidated) == len(want)  # each text once
    if want:
        report = seed.render_prompt_report(gate)
        assert all(f"{key} ({lang})" in report for key, lang in want)
