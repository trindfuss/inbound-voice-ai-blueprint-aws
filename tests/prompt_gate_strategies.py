"""Hypothesis strategies for the per-model prompt gates (parity).

:func:`catalogs` draws seed-file catalogs exercising every rule the prompt hash depends on: the
``agent.system.<model key>`` texts in some languages only (so the default-language and
single-language fallbacks apply), draft entries and lower versions (not served), the two shared
web-activity blocks and the two caller-context frames present or missing, and non-ASCII text. :func:`records_for` draws a records
file over a catalog and a case file (:data:`SUITE`: in every language, in ``en-US`` only):
per model and language no record, a current one (every case run at least the case file's ``runs``),
one whose prompt, suite or parameter hash is stale, or one with a case run too few times or missing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hypothesis import strategies as st

from owlbank.agent_prompt import SYSTEM_KEY_PREFIX, prompt_hash
from owlbank.catalog import CatalogKeyError, PromptCatalog
from owlbank.prompt_validation import CaseTally, Record, Suite, ValidationFile, parse_suite

LANGS = ("en-US", "es-US", "fr-CA")
MODEL_KEYS = ("haiku-5-5", "haiku-4-5", "sonnet-5-5")
WEB_KEYS = ("agent.web_activity", "agent.web_activity_context")
CONTEXT_KEYS = ("agent.caller_context", "agent.caller_context_unverified")
SUITE_DOC: dict[str, Any] = {"version": 1, "runs": 5, "cases": [
    {"id": cid, "lang": lang, "history": [], "caller": "hello", "tool_results": {}, "expect": [{"check": "speaks"}]}
    for cid, lang in [("C1", la) for la in LANGS] + [("C3", "en-US")]]}
SUITE: Suite = parse_suite(SUITE_DOC)
#: The parameter hash a "current" record carries for a model key not in ``records_for``'s ``params``.
PARAMS = "c" * 64
OTHER = "b" * 64

_TEXT = st.text(st.characters(codec="utf-8", exclude_categories=("Cs",)), max_size=12)


@st.composite
def catalogs(draw: Any) -> dict[str, Any]:
    langs = draw(st.lists(st.sampled_from(LANGS), min_size=1, max_size=3, unique=True))
    default = draw(st.sampled_from(langs))
    keys = [f"{SYSTEM_KEY_PREFIX}.{mk}" for mk in MODEL_KEYS] + list(WEB_KEYS) + list(CONTEXT_KEYS)
    entries: list[dict[str, Any]] = []
    for key in keys:
        for lang in langs:
            for _ in range(draw(st.integers(0, 2))):
                entries.append({"key": key, "lang": lang, "text": draw(_TEXT),
                                "version": draw(st.integers(0, 3)),
                                "status": draw(st.sampled_from(["published", "published", "draft"]))})
    single = draw(st.dictionaries(st.sampled_from(keys), st.sampled_from(langs), max_size=2))
    return {"defaultLanguage": default, "languages": {lang: {} for lang in langs},
            "singleLanguageKeys": single, "entries": draw(st.permutations(entries))}


def current_hash(catalog: PromptCatalog, model_key: str, lang: str) -> str | None:
    try:
        return prompt_hash(catalog, model_key, lang)
    except CatalogKeyError:
        return None


def _tallies(draw: Any, suite: Suite, lang: str, short: bool) -> dict[str, CaseTally]:
    """Every case of ``lang`` at least ``suite.runs`` times; ``short``: then one case below it, or
    left out (a record always names at least one case)."""
    ids = [c.id for c in suite.for_lang(lang)] or ["C9"]
    cases = {cid: CaseTally(n, n) for cid in ids for n in [draw(st.integers(suite.runs, suite.runs + 2))]}
    if short and suite.for_lang(lang):
        cid = draw(st.sampled_from(ids))
        if len(cases) > 1 and draw(st.booleans()):
            del cases[cid]
        else:
            n = draw(st.integers(1, suite.runs - 1))
            cases[cid] = CaseTally(n, n)
    return cases


@st.composite
def records_for(draw: Any, doc: dict[str, Any], suite: Suite = SUITE,
                params: Mapping[str, str] | None = None) -> ValidationFile:
    """Per model key and catalog language (and one language outside it): none, current (for the
    case file ``suite`` and the model key's ``params`` hash, default :data:`PARAMS`), stale prompt,
    suite or parameter hash, or short runs."""
    catalog = PromptCatalog.from_dict(doc)
    records = []
    for mk in MODEL_KEYS:
        for lang in [*doc["languages"], "xx-XX"]:
            kind = draw(st.sampled_from(["none", "current", "current", "stale_prompt", "stale_suite",
                                         "stale_params", "short"]))
            if kind == "none":
                continue
            h = current_hash(catalog, mk, lang) or OTHER
            records.append(Record(model_key=mk, lang=lang,
                                  prompt_hash=OTHER if kind == "stale_prompt" else h,
                                  suite_hash=OTHER if kind == "stale_suite" else suite.hash,
                                  params_hash=OTHER if kind == "stale_params" else (params or {}).get(mk, PARAMS),
                                  model_id=f"us.anthropic.claude-{mk}", date="2026-11-02",
                                  cases=_tallies(draw, suite, lang, kind == "short")))
    return ValidationFile(header={}, records=tuple(records))
