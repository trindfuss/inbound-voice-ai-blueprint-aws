"""Properties for ordinal choices and the early-choice check.

* For any partial text, any caller options and any listed order,
  :func:`early_choice` is ``complete`` only when the engine's final deterministic parse of the same
  text is ``exact`` for that same single option (one the caller holds). A text with a yes/no word,
  or naming two options, is never ``complete``. (The one-answer-per-Gather half, with choice
  Gathers, is the front door's.)
* For any listed order of length n and any ordinal phrase, the selected option is
  ``listed[k]`` for the phrase's position k < n, else none; with ``listed`` empty an ordinal never
  selects. In every mode, with no model call.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from owlbank.policy import Policy
from owlbank.slots import OptionView, Pending, commands_from_catalog, make_context, normalize, resolve
from owlbank.slots.validators import (
    CARRIERS,
    NO_WORDS,
    ORDINAL_WORDS,
    YES_WORDS,
    early_choice,
    ordinal_where,
)

DATA = Path(__file__).resolve().parents[2] / "data"
CATALOG = json.loads((DATA / "prompt_catalog.json").read_text())
POLICY = Policy.from_dict(json.loads((DATA / "policy.json").read_text()))
LANGS = ("en-US", "es-US")
TODAY = date(2026, 10, 20)
TEXTS = {lang: {e["key"]: e["text"] for e in CATALOG["entries"] if e["lang"] == lang} for lang in LANGS}
COMMANDS = {lang: commands_from_catalog(CATALOG["entries"], lang) for lang in LANGS}
PROPS = settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
MODES = ("deterministic", "assisted", "generative")
CARD = POLICY.slot("card")
VALUES = tuple(o.value for o in CARD.options)
NON_YES_NO_PENDING = ("disambiguate", "not_available_many", "correction_reask")


def _fam(lang: str) -> str:
    return "es" if lang.startswith("es") else "en"


def _cards(lang: str) -> tuple[OptionView, ...]:
    return tuple(OptionView(o.value, TEXTS[lang][f"option.card.{o.value}"],
                            tuple(o.aliases.get(lang, ())))
                 for o in CARD.options)


class Refuses:
    """A resolver that must never be called."""

    def resolve(self, *a, **k):  # pragma: no cover - the assertion is that it isn't
        raise AssertionError("the resolver was called")


def _final(text: str, lang: str, *, held: tuple[str, ...], listed: tuple[str, ...], kind: str,
           mode: str = "deterministic"):
    pending = Pending(kind, CARD.name, candidates=listed, listed=listed)  # type: ignore[arg-type]
    ctx = make_context(question_text="question", labels=TEXTS[lang], commands=COMMANDS[lang],
                       known_options=_cards(lang), caller_options=held, pending=pending)
    return resolve(CARD, digits=None, speech=text, confidence=None, lang=lang, mode=mode,
                   ctx=ctx, resolver=Refuses(), today=TODAY)


def _vocab(lang: str) -> list[str]:
    fam = _fam(lang)
    words: list[str] = [TEXTS[lang][f"option.card.{v}"] for v in VALUES]
    words += [a for o in CARD.options for a in o.aliases.get(lang, ())]
    words += sorted(ORDINAL_WORDS[fam]) + ["última", "último"] * (fam == "es")
    words += sorted(CARRIERS[fam]) + ["one", TEXTS[lang]["label.card"], "OwlBank"]
    words += sorted(YES_WORDS[fam])[:8] + sorted(NO_WORDS[fam])[:8]
    words += ["or", "and", "o", "y", "um", "zip code", "to", "I'll take", "uh"]
    return words


@st.composite
def partials(draw: st.DrawFn, lang: str) -> str:
    """Words a partial result may hold: names, aliases, ordinals, carriers, yes/no phrases, joiners,
    punctuation, and arbitrary text."""
    parts = draw(st.lists(st.one_of(st.sampled_from(_vocab(lang)), st.text(max_size=6)),
                          max_size=6))
    seps = draw(st.lists(st.sampled_from((" ", ", ", ". ", " - ", "? ")), min_size=len(parts),
                         max_size=len(parts)))
    text = "".join(p + s for p, s in zip(parts, seps, strict=True)).strip()
    return draw(st.sampled_from((text, text.upper(), text.capitalize())))


@st.composite
def situations(draw: st.DrawFn):
    lang = draw(st.sampled_from(LANGS))
    held = tuple(draw(st.lists(st.sampled_from(VALUES), unique=True, min_size=1)))
    listed = tuple(draw(st.lists(st.sampled_from(VALUES), unique=True)))
    kind = draw(st.sampled_from(NON_YES_NO_PENDING))
    other = tuple(draw(st.lists(st.sampled_from(("zip code", "código postal", "last four")),
                                max_size=2)))
    return lang, held, listed, kind, other, draw(partials(lang))


def _has_phrase(text: str, lang: str, phrases: frozenset[str]) -> bool:
    toks = normalize(text, lang).split()
    for p in phrases:
        want = p.split()
        if any(toks[i:i + len(want)] == want for i in range(len(toks) - len(want) + 1)):
            return True
    return False




@PROPS
@given(situations())
def test_complete_only_when_the_final_parse_is_exact(s):
    lang, held, listed, kind, other, text = s
    known = _cards(lang)
    options = tuple(o for o in known if o.value in held)
    outcome, value = early_choice(CARD.decl, text, lang, options=options, listed=listed,
                                  labels=TEXTS[lang], other_labels=other, known=known)
    event(outcome)
    assert outcome in ("complete", "incomplete")
    if outcome == "incomplete":
        assert value is None
        return
    assert value in held
    fam = _fam(lang)
    assert not _has_phrase(text, lang, YES_WORDS[fam] | NO_WORDS[fam])
    for mode in MODES:
        final = _final(text, lang, held=held, listed=listed, kind=kind, mode=mode)
        assert (final.match, final.value, final.model) == ("exact", value, "not_called"), (text, final)
        assert final.source in ("speech", "alias", "ordinal")


@PROPS
@given(st.sampled_from(LANGS), st.data())
def test_two_names_or_a_yes_no_word_never_complete(lang, data):
    known = _cards(lang)
    fam = _fam(lang)
    a, b = data.draw(st.lists(st.sampled_from(known), unique=True, min_size=2, max_size=2))
    name = lambda o: data.draw(st.sampled_from((o.name, *o.aliases)))  # noqa: E731
    yn = data.draw(st.sampled_from(sorted(YES_WORDS[fam] | NO_WORDS[fam])))
    joiner = data.draw(st.sampled_from((" ", " or ", " and ", ", ", " o ", " y ")))
    texts = [f"{name(a)}{joiner}{name(b)}", f"{yn} {name(a)}", f"{name(a)} {yn}",
             f"{yn}, the {name(a)} one"]
    listed = tuple(o.value for o in known)
    for text in texts:
        assert early_choice(CARD.decl, text, lang, options=known, listed=listed,
                            labels=TEXTS[lang]) == ("incomplete", None), text




@st.composite
def ordinal_phrases(draw: st.DrawFn, lang: str) -> tuple[str, int]:
    """An ordinal phrase and its position (-1: last)."""
    fam = _fam(lang)
    word = draw(st.sampled_from(sorted(ORDINAL_WORDS[fam])))
    pos = ORDINAL_WORDS[fam][word]
    if fam == "es" and word.startswith("ultim"):
        word = draw(st.sampled_from((word, word.replace("u", "ú", 1))))
    pre = draw(st.sampled_from(("", "the", "my", "it's the") if fam == "en" else ("", "la", "el", "mi")))
    post = draw(st.sampled_from(("", "one", "please", TEXTS[lang]["label.card"]) if fam == "en"
                                else ("", "por favor", TEXTS[lang]["label.card"])))
    text = " ".join(p for p in (pre, word, post) if p)
    text = draw(st.sampled_from((text, text.upper(), text.capitalize(), text + ".", text + "?")))
    return text, pos


@PROPS
@given(st.sampled_from(LANGS), st.data())
def test_ordinal_selects_listed_k(lang, data):
    text, pos = data.draw(ordinal_phrases(lang))
    listed = tuple(data.draw(st.lists(st.text(min_size=1, max_size=4), unique=True, max_size=5)))
    n = len(listed)
    k = n - 1 if pos < 0 else pos
    hit = ordinal_where(text, lang, n, label=TEXTS[lang]["label.card"])
    if n and k < n:
        assert hit is not None and hit[0] == k
        span = hit[1]
        assert span is not None and normalize(text[span[0]:span[1]], lang) in ORDINAL_WORDS[_fam(lang)]
    else:
        assert hit is None


@PROPS
@given(st.sampled_from(LANGS), st.sampled_from(MODES), st.sampled_from(NON_YES_NO_PENDING),
       st.data())
def test_engine_selects_listed_k_in_every_mode(lang, mode, kind, data):
    text, pos = data.draw(ordinal_phrases(lang))
    listed = tuple(data.draw(st.lists(st.sampled_from(VALUES), unique=True)))
    n = len(listed)
    k = n - 1 if pos < 0 else pos
    r = _final(text, lang, held=VALUES, listed=listed, kind=kind, mode=mode)
    if n and k < n:
        assert (r.match, r.source, r.value, r.model) == ("exact", "ordinal", listed[k], "not_called")
    else:
        # not selected; outside deterministic mode the phrase goes on to the resolver (here one
        # that refuses, so the engine records a resolver error and still selects nothing)
        assert r.source != "ordinal" and r.value is None
        assert mode != "deterministic" or r.model == "not_called"


@PROPS
@given(st.sampled_from(LANGS), st.integers(min_value=-3, max_value=6), st.data())
def test_any_text_selects_in_range_or_nothing(lang, n, data):
    text = data.draw(st.one_of(partials(lang), st.text(max_size=20)))
    hit = ordinal_where(text, lang, n)
    if n < 1:
        assert hit is None
    if hit is not None:
        assert 0 <= hit[0] < n
        toks = [t for t in normalize(text, lang).split() if t not in CARRIERS[_fam(lang)] | {"one"}]
        assert len(toks) == 1 and toks[0] in ORDINAL_WORDS[_fam(lang)]
