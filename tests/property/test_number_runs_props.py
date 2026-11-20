"""Property tests for ``number_runs``.

* Run shape: runs are ordered, disjoint and inside the text; each has ``tokens >= 1`` and
  ``digits`` ``None`` or a digit string; a one-language run lies inside a two-language run.
"""

from __future__ import annotations

import re
import unicodedata

from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.slots import number_runs
from owlbank.slots.normalize import FILLERS, NUMBER_WORDS

LANGS = ("en-US", "es-US")
LANG_SETS = (("en-US",), ("es-US",), ("en-US", "es-US"), ("es-US", "en-US"))

_WORDS = sorted(NUMBER_WORDS["en"] | NUMBER_WORDS["es"])
_JOINERS = ["and", "y", "con", "And", "Y", "CON"]
_FILLER_WORDS = sorted(FILLERS["en"] | FILLERS["es"])
_ORDINARY = ["yes", "my", "last", "four", "are", "no", "quiero", "la", "otra", "it's", "o'clock",
             "dólares", "tarjeta", "a", "is", "the", "sí", "número", "ó", "Ó", "uno", "Una"]
_SEPARATORS = [" ", "  ", ",", ", ", ".", ". ", "-", "/", " - ", "\t", "\n", ""]
_PUNCT = ["!", "?", ";", ":", "'", "’", "(", ")", "$", "_", "%", "#", "...", " ", "́"]


def _cased(word: str) -> st.SearchStrategy[str]:
    return st.sampled_from([word, word.upper(), word.capitalize()])


def _accented(word: str) -> str:
    """Put an acute accent on the first vowel (composed or decomposed)."""
    for i, c in enumerate(word):
        if c in "aeiou":
            return word[:i] + unicodedata.normalize("NFC", c + "́") + word[i + 1:]
    return word


number_word = st.sampled_from(_WORDS).flatmap(
    lambda w: st.one_of(_cased(w), st.just(_accented(w)),
                        st.just(unicodedata.normalize("NFD", _accented(w)))))
digit_token = st.text(alphabet="0123456789", min_size=1, max_size=6)
unicode_numeric = st.sampled_from(["٣", "²", "½", "Ⅷ", "７", "1st", "4th", "2nd"])
piece = st.one_of(
    number_word,
    digit_token,
    unicode_numeric,
    st.sampled_from(_JOINERS),
    st.sampled_from(_FILLER_WORDS),
    st.sampled_from(_ORDINARY),
    st.sampled_from(_PUNCT),
    st.text(max_size=4),
)
separator = st.one_of(st.sampled_from(_SEPARATORS + _PUNCT), st.text(max_size=2))


@st.composite
def utterances(draw) -> str:
    pieces = draw(st.lists(piece, max_size=12))
    out = []
    for p in pieces:
        out.append(p)
        out.append(draw(separator))
    return "".join(out)


texts = st.one_of(utterances(), st.text(max_size=40))


_DIGITS_RE = re.compile(r"[0-9]+")


@settings(max_examples=800, deadline=None)
@given(texts, st.sampled_from(LANG_SETS))
def test_run_shape(text, langs):
    runs = number_runs(text, langs)
    prev_end = -1
    for r in runs:
        assert 0 <= r.start < r.end <= len(text)
        assert r.start >= prev_end, "runs must be ordered and disjoint"
        prev_end = r.end
        assert r.tokens >= 1
        assert r.digits is None or _DIGITS_RE.fullmatch(r.digits)


@settings(max_examples=800, deadline=None)
@given(texts, st.sampled_from(LANGS))
def test_one_language_run_lies_inside_a_two_language_run(text, lang):
    both = number_runs(text, LANGS)
    for r in number_runs(text, (lang,)):
        assert any(b.start <= r.start and r.end <= b.end for b in both), (r, both)
