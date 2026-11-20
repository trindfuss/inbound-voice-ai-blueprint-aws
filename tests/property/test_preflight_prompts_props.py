"""Preflight's ``check_prompts``
FAILs exactly when some entry of the list and some catalog language lacks a record with the current
prompt, suite and parameter hashes and full runs (every case of the language at least the case
file's ``runs``).

The oracle is ``owlbank.prompt_validation.ValidationFile.is_validated`` on the bridge's own
``prompt_hash``, the parsed case file and ``owlbank.prompt_validation.params_hash`` of the entry's
parameters (``preflight.entry_params``: the resolver's defaults and the ``--model-params``
overrides, a ``prompt_caching`` override among them and not hashed; pinned against synth's ``modelParams`` by
``tests/unit/test_failover_parity.py``). Preflight reads the same records through its standalone
copies, from the file text ``ValidationFile.dumps`` writes. Also the resolver's ``has_prompt``:
validated in every language.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from hypothesis import event, given, settings
from hypothesis import strategies as st

from owlbank.agent_prompt import model_key
from owlbank.catalog import PromptCatalog
from owlbank.prompt_validation import params_hash, parse_suite
from tests.prompt_gate_strategies import MODEL_KEYS, SUITE, SUITE_DOC, catalogs, records_for

_PATH = Path(__file__).resolve().parents[2] / "infra" / "scripts" / "preflight.py"
_spec = importlib.util.spec_from_file_location("_preflight_for_p93", _PATH)
assert _spec is not None and _spec.loader is not None
preflight = importlib.util.module_from_spec(_spec)
sys.modules["_preflight_for_p93"] = preflight  # dataclasses look the module up
_spec.loader.exec_module(preflight)

_IDS = st.sampled_from([f"{pre}.anthropic.claude-{mk}" for mk in MODEL_KEYS for pre in ("us", "global")]
                       + ["us.anthropic.claude-haiku-4-5-20251001-v1:0", "us.anthropic.claude-opus-4-1"])
_OVERRIDE = st.fixed_dictionaries({}, optional={
    "thinking": st.sampled_from(["adaptive", "off", "enabled:2048", " Adaptive "]),
    "text_release": st.sampled_from(["stream", "message"]),
    "max_tokens": st.sampled_from([400, 1500, 2000]),
    "prompt_caching": st.sampled_from([True, False, "on", "off"]),
})


@st.composite
def cases(draw: Any) -> tuple[Any, ...]:
    doc = draw(catalogs())
    ids = draw(st.lists(_IDS, min_size=1, max_size=3, unique=True))
    overrides = {mid: draw(_OVERRIDE) for mid in ids if draw(st.booleans())}
    # A "current" record carries the parameters of one entry of its model key (or the defaults), so
    # entries of one series with other overrides see it stale.
    params = {}
    for mk in MODEL_KEYS:
        mid = next((i for i in ids if model_key(i) == mk), f"us.anthropic.claude-{mk}")
        params[mk] = params_hash(preflight.entry_params(mid, overrides))
    records = draw(records_for(doc, SUITE, params))
    suite_doc = draw(st.sampled_from([SUITE_DOC, {**SUITE_DOC, "runs": 3}, {**SUITE_DOC, "runs": 6}, None]))
    return doc, records, suite_doc, ids, overrides


@settings(max_examples=300, deadline=None)
@given(cases())
def test_check_prompts_fails_exactly_for_the_entries_and_languages_without_a_current_record(case: Any) -> None:
    doc, records, suite_doc, ids, overrides = case
    catalog = PromptCatalog.from_dict(doc)
    prompts = preflight.parse_prompt_records(doc, json.loads(records.dumps()), suite_doc)
    suite = None if suite_doc is None else parse_suite(suite_doc)

    def params(mid: str) -> dict[str, Any]:
        return preflight.entry_params(mid, overrides)

    def validated(mid: str, lang: str) -> bool:
        return suite is not None and records.is_validated(catalog, model_key(mid), lang, suite,
                                                          params_hash(params(mid)))

    results = preflight.check_prompts(prompts, ids, overrides)
    want_fail = {f"Model prompt {mid} ({lang})" for mid in ids for lang in catalog.languages
                 if not validated(mid, lang)}
    assert [r.name for r in results] == [f"Model prompt {mid} ({lang})" for mid in ids for lang in catalog.languages]
    assert {r.name for r in results if not r.ok} == want_fail
    assert all(r.ok for r in results) == (not want_fail)
    for r in results:
        event("PASS" if r.ok else r.detail.split(" (")[0])  # coverage of each reason (--hypothesis-show-statistics)
        if not r.ok:
            assert r.fix and "prompt_suite" in r.fix
    for mid in ids:
        assert prompts.has_prompt(mid, params(mid)) == all(validated(mid, lang) for lang in catalog.languages)
