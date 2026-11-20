"""Ordinal choices and the early-choice check.

Examples for :func:`ordinal_where` over two and three listed options, the engine's ordinal stage
(``match: exact``, source ``ordinal``, every mode, no model call, the placeholder span) and
:func:`early_choice` on the partials the design names.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from owlbank.policy import Policy
from owlbank.slots import OptionView, Pending, commands_from_catalog, make_context, resolve
from owlbank.slots.dialog import DETERMINISTIC_SOURCES
from owlbank.slots.types import SOURCES
from owlbank.slots.validators import ORDINAL_WORDS, early_choice, ordinal_where

DATA = Path(__file__).resolve().parents[2] / "data"
CATALOG = json.loads((DATA / "prompt_catalog.json").read_text())
POLICY = Policy.from_dict(json.loads((DATA / "policy.json").read_text()))
TODAY = date(2026, 10, 20)
CARD = POLICY.slot("card")
MODES = ("deterministic", "assisted", "generative")


def texts(lang: str) -> dict[str, str]:
    return {e["key"]: e["text"] for e in CATALOG["entries"] if e["lang"] == lang}


def cards(lang: str = "en-US", only: tuple[str, ...] | None = None) -> tuple[OptionView, ...]:
    t = texts(lang)
    return tuple(OptionView(value=o.value, name=t[f"option.card.{o.value}"],
                            aliases=tuple(o.aliases.get(lang, ())))
                 for o in CARD.options if only is None or o.value in only)


class Refuses:
    """A resolver that must never be called."""

    def resolve(self, *a, **k):  # pragma: no cover - the assertion is that it isn't
        raise AssertionError("the resolver was called")


def run(speech, *, listed=(), lang="en-US", mode="assisted", kind="disambiguate", slot=CARD):
    pending = Pending(kind, slot.name, candidates=tuple(listed), listed=tuple(listed))
    ctx = make_context(question_text="question", labels=texts(lang),
                       commands=commands_from_catalog(CATALOG["entries"], lang),
                       known_options=cards(lang), pending=pending)
    return resolve(slot, digits=None, speech=speech, confidence=0.9, lang=lang, mode=mode, ctx=ctx,
                   resolver=Refuses(), today=TODAY)


# ------------------------------------------------------------------------------- ordinal_where


@pytest.mark.parametrize(("text", "lang", "n", "index"), [
    ("the second one", "en-US", 2, 1),
    ("the second one", "en-US", 3, 1),
    ("second", "en-US", 2, 1),
    ("The first one, please.", "en-US", 3, 0),
    ("the third card", "en-US", 3, 2),
    ("the last one", "en-US", 2, 1),
    ("the last one", "en-US", 3, 2),
    ("la segunda", "es-US", 2, 1),
    ("la segunda", "es-US", 3, 1),
    ("el primero", "es-US", 3, 0),
    ("la tercera tarjeta", "es-US", 3, 2),
    ("la última", "es-US", 3, 2),
    ("el último", "es-US", 2, 1),
])
def test_ordinal_picks_the_position(text, lang, n, index):
    label = texts(lang)["label.card"]
    hit = ordinal_where(text, lang, n, label=label)
    assert hit is not None and hit[0] == index


@pytest.mark.parametrize(("text", "lang", "n"), [
    ("the third one", "en-US", 2),           # beyond n
    ("la tercera", "es-US", 2),
    ("the first one", "en-US", 0),           # nothing listed
    ("the first or the second", "en-US", 3), # two ordinals
    ("first second", "en-US", 3),
    ("la primera o la segunda", "es-US", 3),
    ("yes the second one", "en-US", 3),      # another word
    ("the second to last", "en-US", 3),
    ("I'll take the second one", "en-US", 3),
    ("the one", "en-US", 3),                 # no ordinal
    ("", "en-US", 3),
    ("segunda", "en-US", 3),                 # the other language's word
])
def test_ordinal_none(text, lang, n):
    assert ordinal_where(text, lang, n, label=texts(lang)["label.card"]) is None


def test_ordinal_span_is_the_ordinal_word():
    text = "The second one"
    idx, span = ordinal_where(text, "en-US", 3)
    assert idx == 1 and span is not None and text[span[0]:span[1]] == "second"
    text = "la Última"
    idx, span = ordinal_where(text, "es-US", 3)
    assert idx == 2 and span is not None and text[span[0]:span[1]] == "Última"


def test_ordinal_words_are_single_normalized_tokens():
    for words in ORDINAL_WORDS.values():
        assert all(w.isascii() and w.isalpha() and w.islower() for w in words)


# ------------------------------------------------------------------------------- Pending.listed


def test_pending_listed_default_and_validation():
    assert Pending("disambiguate", "card").listed == ()
    assert Pending("disambiguate", "card", listed=("red", "purple")).listed == ("red", "purple")
    for bad in (["red"], ("red", "red"), ("red", ""), ("red", 3)):
        with pytest.raises(ValueError):
            Pending("disambiguate", "card", listed=bad)  # type: ignore[arg-type]


def test_ordinal_is_a_deterministic_source():
    assert "ordinal" in SOURCES and "ordinal" in DETERMINISTIC_SOURCES


# ------------------------------------------------------------------------------- engine stage


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("speech", "lang", "listed", "value"), [
    ("the second one", "en-US", ("red", "purple"), "purple"),
    ("the second one", "en-US", ("purple", "red", "rewards"), "red"),
    ("the last one", "en-US", ("red", "purple"), "purple"),
    ("the last one", "en-US", ("red", "purple", "rewards"), "rewards"),
    ("la segunda", "es-US", ("red", "purple"), "purple"),
    ("la última", "es-US", ("rewards", "purple", "red"), "red"),
])
def test_engine_ordinal_exact_every_mode(mode, speech, lang, listed, value):
    r = run(speech, listed=listed, lang=lang, mode=mode)
    assert (r.match, r.source, r.value, r.model) == ("exact", "ordinal", value, "not_called")


def test_engine_names_win_over_ordinals():
    r = run("purple", listed=("red", "purple"))
    assert (r.match, r.source, r.value) == ("exact", "alias", "purple")


@pytest.mark.parametrize(("speech", "listed"), [
    ("the third one", ("red", "purple")),          # beyond n
    ("the first or the second", ("red", "purple")),
    ("the second one", ()),                        # the question listed nothing
])
def test_engine_ordinal_none(speech, listed):
    r = run(speech, listed=listed, mode="deterministic")
    assert r.source != "ordinal" and r.value is None


def test_engine_ordinal_span_feeds_the_placeholder():
    sensitive = replace(CARD, sensitive=True)
    r = run("the second one", listed=("red", "purple"), slot=sensitive, mode="deterministic")
    assert (r.source, r.value) == ("ordinal", "purple")
    assert r.utterance_for_ai is not None and "second" not in r.utterance_for_ai
    assert "[card]" in r.utterance_for_ai


# ------------------------------------------------------------------------------- early_choice


def early(text, *, lang="en-US", held=("red", "purple"), listed=(), other_labels=(), known=None):
    return early_choice(CARD.decl, text, lang, options=cards(lang, held), listed=listed,
                        labels=texts(lang), other_labels=other_labels, known=known)


@pytest.mark.parametrize(("text", "value"), [
    ("red", "red"),
    ("the red one", "red"),
    ("OwlBank Purple Card", "purple"),
    ("my purple card", "purple"),
])
def test_early_choice_complete(text, value):
    assert early(text) == ("complete", value)


@pytest.mark.parametrize("text", [
    "red no", "yes red", "the OwlBank card", "red or purple", "yes", "no", "", "the", "uh",
    "the rewards card",            # known but not held
    "red please, that's right",    # a yes word anywhere
])
def test_early_choice_incomplete(text):
    assert early(text) == ("incomplete", None)


def test_early_choice_spanish():
    assert early("la roja", lang="es-US") == ("complete", "red")
    assert early("roja no", lang="es-US") == ("incomplete", None)
    assert early("sí, la morada", lang="es-US") == ("incomplete", None)


def test_early_choice_ordinals():
    assert early("the second one", listed=("red", "purple")) == ("complete", "purple")
    assert early("the last one", listed=("purple", "red")) == ("complete", "red")
    assert early("the second one") == ("incomplete", None)                  # nothing listed
    assert early("the third one", listed=("red", "purple")) == ("incomplete", None)
    assert early("la segunda", lang="es-US", listed=("red", "purple")) == ("complete", "purple")


def test_early_choice_label_guard():
    assert early("red zip code", other_labels=("zip code",)) == ("incomplete", None)
    assert early("red", other_labels=("zip code",)) == ("complete", "red")


def test_early_choice_name_shared_with_an_option_not_held():
    shared = (OptionView("red", "OwlBank Red Card", ("owlbank card",)),
              OptionView("purple", "OwlBank Purple Card", ("owlbank card",)))
    held = shared[:1]
    assert early_choice(CARD.decl, "the owlbank card", "en-US", options=held, listed=(),
                        labels=texts("en-US")) == ("complete", "red")
    assert early_choice(CARD.decl, "the owlbank card", "en-US", options=held, listed=(),
                        labels=texts("en-US"), known=shared) == ("incomplete", None)


def test_early_choice_not_a_choice_slot():
    assert early_choice(POLICY.slot("zip").decl, "red", "en-US", options=cards(), listed=(),
                        labels=texts("en-US")) == ("incomplete", None)
