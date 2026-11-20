"""Level-0 tools: ``get_branch_hours`` and ``faq_lookup``.

Pure functions over the content of ``data/faq.json``. They read no customer data
and take no token. ``faq_lookup`` is simple keyword matching: the question is
normalized (case, accents, punctuation), stop words are dropped, light plural
stemming is applied, and each FAQ is scored by how many of its question's
keywords (in the caller's language, plus the default language) the question
contains. The best FAQ wins if it shares at least :data:`MIN_SCORE` keywords.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_LANG = "en-US"
SUPPORTED_LANGS = ("en-US", "es-US")
MIN_SCORE = 1
MAX_QUESTION_LEN = 500
DEFAULT_FAQ_PATH = Path(__file__).resolve().parents[3] / "data" / "faq.json"

_STOP = frozenset("""
a an the i me my we our you your to of in on for with and or is are be do does did can could
should would will what when where how who which if it its this that there at by from about
please want need know tell get have has
el la los las un una unos unas y o de del en con por para que qué como cómo cuál cual cuando
cuándo donde dónde mi mis su sus se me lo le es son hay puedo puede quiero necesito saber
""".split())  # noqa: SIM905 (a readable word list; a literal list would be one long line)

_WORD = re.compile(r"[a-z0-9]+")


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("es") and not word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def keywords(text: str) -> set[str]:
    """Normalized, stemmed keywords of ``text`` without stop words."""
    return {_stem(w) for w in _WORD.findall(_fold(text)) if w not in _STOP and len(w) > 1}


def lang_or_default(lang: Any) -> str:
    return lang if lang in SUPPORTED_LANGS else DEFAULT_LANG


def _text(entry: Mapping[str, str], lang: str) -> str:
    return entry.get(lang) or entry.get(DEFAULT_LANG) or ""


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_content(path: str | Path | None = None) -> dict[str, Any]:
    """The FAQ/branch content (cached per path)."""
    return _load(str(path or DEFAULT_FAQ_PATH))


def get_branch_hours(content: Mapping[str, Any], *, branch: str | None = None,
                     lang: str | None = None) -> dict[str, Any]:
    """Hours for one branch (matched by id or name keywords), or for every branch."""
    lang = lang_or_default(lang)
    branches = content.get("branches", [])
    chosen = branches
    if isinstance(branch, str) and branch.strip():
        want = keywords(branch[:MAX_QUESTION_LEN]) - {"owlbank", "branch", "sucursal"}
        chosen = [b for b in branches
                  if b.get("id") in want
                  or want & set().union(*(keywords(n) for n in b.get("name", {}).values()))]
        if not chosen:
            return {"found": False, "branches": [_branch(b, lang) for b in branches],
                    "phoneSupportHours": _text(content.get("phoneSupportHours", {}), lang)}
    return {"found": True, "branches": [_branch(b, lang) for b in chosen],
            "phoneSupportHours": _text(content.get("phoneSupportHours", {}), lang)}


def _branch(b: Mapping[str, Any], lang: str) -> dict[str, Any]:
    return {"id": b.get("id"), "name": _text(b.get("name", {}), lang),
            "address": b.get("address"), "hours": _text(b.get("hours", {}), lang),
            "timezone": b.get("timezone")}


def faq_lookup(content: Mapping[str, Any], *, question: str, lang: str | None = None) -> dict[str, Any]:
    """The best-matching FAQ answer in ``lang`` (default-language fallback), or ``found: false``."""
    lang = lang_or_default(lang)
    if not isinstance(question, str) or not question.strip():
        return {"found": False}
    asked = keywords(question[:MAX_QUESTION_LEN])
    best: tuple[int, Mapping[str, Any]] | None = None
    for faq in content.get("faqs", []):
        q = faq.get("question", {})
        kws = keywords(_text(q, lang)) | keywords(q.get(DEFAULT_LANG, ""))
        score = len(asked & kws)
        if score >= MIN_SCORE and (best is None or score > best[0]):
            best = (score, faq)
    if best is None:
        return {"found": False}
    faq = best[1]
    return {"found": True, "id": faq.get("id"), "question": _text(faq.get("question", {}), lang),
            "answer": _text(faq.get("answer", {}), lang), "lang": lang}


TOOLS = ("get_branch_hours", "faq_lookup")


def dispatch(content: Mapping[str, Any], tool: str, args: Mapping[str, Any]) -> dict[str, Any]:
    """Run ``tool`` with Gateway ``args``. Unknown tools and bad args return ``error``."""
    if not isinstance(args, Mapping):
        return {"error": "INVALID_ARGUMENT"}
    if tool == "get_branch_hours":
        branch, lang = args.get("branch"), args.get("lang")
        if branch is not None and not isinstance(branch, str):
            return {"error": "INVALID_ARGUMENT"}
        return get_branch_hours(content, branch=branch, lang=lang)
    if tool == "faq_lookup":
        q = args.get("question")
        if not isinstance(q, str):
            return {"error": "INVALID_ARGUMENT"}
        return faq_lookup(content, question=q, lang=args.get("lang"))
    return {"error": "UNKNOWN_TOOL"}
