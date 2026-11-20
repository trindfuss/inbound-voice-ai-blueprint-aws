"""Model failover: ``is_validated`` and the record file.

* ``is_validated(catalog, model_key, lang, suite, params_hash)`` is true exactly when the file holds
  a record for that model and language carrying the current ``prompt_hash`` (from the catalog), the
  suite's hash, full runs of every case of the language and, when given, ``params_hash``: over
  random catalogs, record sets (each hash current or not, runs full or short) and queries.
* Serialization is deterministic: any insertion order gives the same text, and the text parses back
  to the same file.
"""

from __future__ import annotations

import json
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.agent_prompt import prompt_hash, system_key
from owlbank.catalog import CatalogKeyError, PromptCatalog
from owlbank.prompt_validation import Case, CaseTally, Check, Record, Suite, ValidationFile, parse_records

LANGS = ("en-US", "es-US")
MODEL_KEYS = ("haiku-5-5", "haiku-4-5", "sonnet-5-5")

text = st.text(max_size=30)
hex64 = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)
tallies = st.dictionaries(st.from_regex(r"C[0-9]{1,2}", fullmatch=True),
                          st.integers(1, 9).map(lambda n: CaseTally(n, n)), min_size=1, max_size=4)


@st.composite
def catalogs(draw: st.DrawFn) -> PromptCatalog:
    """Model texts for some (model, language) pairs (via ``system_key``, so the key name is the
    module's), optional web-activity blocks."""
    entries = [{"key": system_key(mk), "lang": lang, "text": draw(text)}
               for mk in MODEL_KEYS for lang in LANGS if draw(st.booleans())]
    entries += [{"key": key, "lang": lang, "text": draw(text)}
                for key in ("agent.web_activity", "agent.web_activity_context") for lang in LANGS
                if draw(st.booleans())]
    entries.append({"key": "other.key", "lang": "en-US", "text": "x"})
    # system_key maps every model to one key: keep one text per (key, lang).
    unique = {(e["key"], e["lang"]): e for e in entries}
    return PromptCatalog.from_dict({"defaultLanguage": "en-US", "languages": {lang: {} for lang in LANGS},
                                    "entries": list(unique.values())})


def _current(catalog: PromptCatalog, mk: str, lang: str) -> str | None:
    try:
        return prompt_hash(catalog, mk, lang)
    except CatalogKeyError:  # no model text: nothing can match
        return None


@st.composite
def suites(draw: st.DrawFn) -> Suite:
    """A case file: ``runs`` and some of the cases in each language (the content is not read)."""
    cases = tuple(Case(id=cid, lang=lang, history=(), caller="hi", tool_results={}, expect=(Check("speaks"),))
                  for lang in LANGS for cid in ("C1", "C2", "C3") if draw(st.booleans()))
    return Suite(runs=draw(st.integers(1, 6)), cases=cases, hash=draw(hex64))


@st.composite
def scenarios(draw: st.DrawFn) -> tuple[PromptCatalog, ValidationFile, Suite, str | None, str, str]:
    catalog = draw(catalogs())
    suite = draw(suites())
    params = draw(st.none() | hex64)
    vf = ValidationFile.empty()
    for _ in range(draw(st.integers(0, 6))):
        mk, lang = draw(st.sampled_from(MODEL_KEYS)), draw(st.sampled_from(LANGS))
        current = _current(catalog, mk, lang)
        ph = draw(st.sampled_from([current]) if current and draw(st.booleans()) else hex64)
        sh = draw(st.just(suite.hash) if draw(st.booleans()) else hex64)
        pa = draw(st.just(params) if params and draw(st.booleans()) else hex64)
        # each case of the language some number of runs around the case file's, or absent
        cases = {c.id: CaseTally(n, n) for c in suite.for_lang(lang)
                 for n in [draw(st.integers(max(1, suite.runs - 1), suite.runs + 1))] if draw(st.integers(0, 5))}
        cases = cases or draw(tallies)
        vf = vf.add(Record(model_key=mk, lang=lang, prompt_hash=ph, suite_hash=sh, params_hash=pa, model_id=f"us.{mk}",
                           date="2026-11-02", cases=cases))
    return catalog, vf, suite, params, draw(st.sampled_from(MODEL_KEYS)), draw(st.sampled_from(LANGS))


@settings(max_examples=400, deadline=None)
@given(s=scenarios())
def test_is_validated_exactly_when_a_record_carries_the_current_hashes_and_full_runs(
        s: tuple[PromptCatalog, ValidationFile, Suite, str | None, str, str]) -> None:
    catalog, vf, suite, params, mk, lang = s
    current = _current(catalog, mk, lang)
    expected = current is not None and any(
        r.model_key == mk and r.lang == lang and r.prompt_hash == current and r.suite_hash == suite.hash
        and (params is None or r.params_hash == params)
        and all(c.id in r.cases and r.cases[c.id].runs >= suite.runs for c in suite.for_lang(lang))
        for r in vf.records)
    assert vf.is_validated(catalog, mk, lang, suite, params) is expected


@st.composite
def records(draw: st.DrawFn) -> list[Record]:
    keys = draw(st.lists(st.tuples(st.sampled_from(MODEL_KEYS), st.sampled_from(LANGS)), unique=True, max_size=6))
    return [Record(model_key=mk, lang=lang, prompt_hash=draw(hex64), suite_hash=draw(hex64), params_hash=draw(hex64),
                   model_id=draw(st.text(min_size=1, max_size=20)), date="2026-11-02", cases=draw(tallies))
            for mk, lang in keys]


header = st.dictionaries(st.text(max_size=8), st.none() | st.integers() | text, max_size=3)


@settings(max_examples=200, deadline=None)
@given(rs=records(), h=header, data=st.data())
def test_serialization_is_order_independent_and_round_trips(rs: list[Record], h: dict[str, Any],
                                                            data: st.DataObject) -> None:
    def build(order: list[Record]) -> ValidationFile:
        vf = ValidationFile.empty()
        for r in order:
            vf = vf.add(r)
        return ValidationFile(header=h, records=vf.records)

    one = build(rs)
    other = build(data.draw(st.permutations(rs)))
    assert one.dumps() == other.dumps()
    assert parse_records(json.loads(one.dumps())) == one
