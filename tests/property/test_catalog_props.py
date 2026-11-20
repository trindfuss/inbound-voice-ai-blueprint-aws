"""Catalog fallback and placeholder properties."""

from __future__ import annotations

from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from owlbank.catalog import PLACEHOLDER, PromptCatalog, render

ident = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,8}", fullmatch=True)
literal = st.text(alphabet=st.characters(blacklist_characters="{}", blacklist_categories=("Cs",)),
                  max_size=12)
# Values may contain anything, including placeholder and format syntax.
value = st.text(max_size=20)


@st.composite
def templates(draw: st.DrawFn) -> list[tuple[str, str]]:
    """A template as pieces: ("lit", text) or ("ph", name)."""
    return draw(st.lists(st.one_of(st.tuples(st.just("lit"), literal),
                                   st.tuples(st.just("ph"), ident)), max_size=8))


def _join(pieces: list[tuple[str, str]]) -> str:
    return "".join(t if kind == "lit" else "{" + t + "}" for kind, t in pieces)


@given(pieces=templates(), data=st.data())
def test_known_placeholders_filled_from_values_only(pieces: list[tuple[str, str]],
                                                    data: st.DataObject) -> None:
    names = {t for kind, t in pieces if kind == "ph"}
    values = {n: data.draw(value) for n in names}
    expected = "".join(t if kind == "lit" else values[t] for kind, t in pieces)
    assert render(_join(pieces), values) == expected


@given(pieces=templates(), data=st.data())
def test_unknown_placeholders_left_out(pieces: list[tuple[str, str]], data: st.DataObject) -> None:
    names = sorted({t for kind, t in pieces if kind == "ph"})
    known = set(data.draw(st.lists(st.sampled_from(names), unique=True))) if names else set()
    # Known values are brace-free here so any "{x}" in the output could only be echoed.
    values = {n: data.draw(literal) for n in known}
    out = render(_join(pieces), values)
    expected = "".join(t if kind == "lit" else values.get(t, "") for kind, t in pieces)
    # Only spaces next to a removed placeholder may differ.
    assert out.replace(" ", "") == expected.replace(" ", "")
    assert PLACEHOLDER.search(out) is None


@given(pieces=templates(), injected=value)
def test_values_never_expanded(pieces: list[tuple[str, str]], injected: str) -> None:
    names = {t for kind, t in pieces if kind == "ph"}
    values = dict.fromkeys(names, injected)
    out = render(_join(pieces), values)
    lits = "".join(t for kind, t in pieces if kind == "lit")
    assert len(out) == len(lits) + len(injected) * sum(kind == "ph" for kind, _ in pieces)


@st.composite
def catalogs(draw: st.DrawFn) -> dict[str, Any]:
    keys = draw(st.lists(st.from_regex(r"[a-z]{1,6}\.[a-z]{1,6}", fullmatch=True),
                         min_size=1, max_size=8, unique=True))
    entries = []
    for k in keys:
        entries.append({"key": k, "lang": "en-US", "text": draw(literal), "version": 1,
                        "status": "published", "audioUrl": None})
        if draw(st.booleans()):
            entries.append({"key": k, "lang": "es-US", "text": draw(literal), "version": 1,
                            "status": "published", "audioUrl": None})
    return {"defaultLanguage": "en-US", "languages": {"en-US": {}, "es-US": {}},
            "entries": entries}


@given(doc=catalogs(), lang=st.sampled_from(["en-US", "es-US", "fr-FR", ""]))
def test_missing_language_returns_default_text(doc: dict[str, Any], lang: str) -> None:
    cat = PromptCatalog.from_dict(doc)
    have = {(e["key"], e["lang"]): e["text"] for e in doc["entries"]}
    for key in cat.keys:
        got = cat.text(key, lang)
        assert got == have.get((key, lang), have[(key, "en-US")])
    missing = cat.missing_keys("es-US")
    assert missing == sorted(k for k in cat.keys if (k, "es-US") not in have)
