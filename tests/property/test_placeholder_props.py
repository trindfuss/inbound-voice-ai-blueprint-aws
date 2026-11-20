"""Property tests for ``sensitive.placeholder_text``.

Generated answering turns for each seed digit slot (``zip`` 5, ``ssn4`` 4, ``otp`` 6 digits): the
value said as one run or split across runs, each run in digits or English or Spanish number words,
among fillers, lone number words that supplied nothing, option names and yes/no phrases. The runs
that supplied the value are located (``value_runs``) or not (``None``: a keypad value with speech,
the fallback). The properties:

(i) no exact-value match (``redact_text``'s normalization);
(ii) every run that supplied a value digit is replaced;
(iii) every surviving number run (``number_runs``, both languages) is one token with one digit;
(iv) with the fallback, no surviving run's digit occurs in the value;
and words outside replaced runs and spans are kept in order, a lone word that supplied nothing
being kept (located) unless its digit is in the value (fallback).

The generator knows what each piece is, so (ii) and the kept words are checked against the exact
expected text; (i), (iii) and (iv) are also checked directly on the output, and again over
arbitrary text. A last property covers the ``yes_no`` and ``choice`` span rule.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from owlbank import segments
from owlbank.sensitive import placeholder_text
from owlbank.slot_policy import SlotSettings
from owlbank.slots import number_runs
from owlbank.slots.normalize import _JOINERS, NUMBER_WORDS, UNITS
from owlbank.slots.types import SlotDecl

LANGS = ("en", "es")
SEED_SLOTS = {"zip": 5, "ssn4": 4, "otp": 6}

#: Single-digit words per language and digit (never "o": it is also a Spanish word; "oh" is fine).
_UNIT_WORDS: dict[str, dict[str, list[str]]] = {
    fam: {str(d): sorted(w for w, v in UNITS[fam].items() if v == d and w != "o") for d in range(10)}
    for fam in LANGS
}
_NUMERIC_WORDS = NUMBER_WORDS["en"] | NUMBER_WORDS["es"] | _JOINERS["en"] | _JOINERS["es"]
_FILLERS = ["uh", "um", "eh", "no", "I mean", "este", "pues", "my", "last", "are", "is", "the",
            "yes", "sí", "claro", "that's right", "checking", "savings", "ahorros", "cuenta",
            "please", "por favor", "wait", "Okay"]
assert not {w.lower() for f in _FILLERS for w in f.split()} & _NUMERIC_WORDS
#: Lone number words that supplied nothing and always go: 2 or more digits, or unreadable.
_LONE_OTHER = ["fifteen", "twenty", "doce", "hundred", "first", "mil"]
_JOIN = [" ", ", ", " ... ", "; ", " - "]
_IN_RUN = [" ", "-", " ", ", "]


@dataclass(frozen=True)
class Piece:
    kind: str  # "value" (part of the value), "lone" (one digit), "other" (always goes), "words"
    text: str
    digits: str = ""


def _slot(name: str) -> SlotSettings:
    return SlotSettings(name=name, decl=SlotDecl("digits", {"length": SEED_SLOTS[name]}),
                        is_factor=True, sensitive=True)


@st.composite
def say_digits(draw: st.DrawFn, digits: str) -> str:
    """``digits`` as one run: a digit string, or one word per digit in English or Spanish."""
    form = draw(st.sampled_from(["digits", "en", "es"]))
    if form == "digits":
        return digits if draw(st.booleans()) else draw(st.sampled_from(_IN_RUN)).join(digits)
    words = [draw(st.sampled_from(_UNIT_WORDS[form][d])) for d in digits]
    if draw(st.booleans()):
        words = [w.capitalize() for w in words]
    return " ".join(words)


@st.composite
def utterances(draw: st.DrawFn) -> tuple[str, str, list[Piece]]:
    name = draw(st.sampled_from(sorted(SEED_SLOTS)))
    value = draw(st.text("0123456789", min_size=SEED_SLOTS[name], max_size=SEED_SLOTS[name]))
    cuts = sorted(draw(st.sets(st.integers(1, len(value) - 1), max_size=len(value) - 1)))
    parts = [value[a:b] for a, b in zip([0, *cuts], [*cuts, len(value)], strict=True)]
    pieces = [Piece("value", draw(say_digits(p)), p) for p in parts]
    for _ in range(draw(st.integers(0, 3))):
        kind = draw(st.sampled_from(["lone", "other", "words"]))
        if kind == "lone":
            d = draw(st.sampled_from("0123456789"))
            fam = draw(st.sampled_from(["digit", *LANGS]))
            word = d if fam == "digit" else draw(st.sampled_from(_UNIT_WORDS[fam][d]))
            piece = Piece("lone", word, d)
        elif kind == "other":
            piece = Piece("other", draw(st.sampled_from(_LONE_OTHER)))
        else:
            piece = Piece("words", draw(st.sampled_from(_FILLERS)))
        pieces.insert(draw(st.integers(0, len(pieces))), piece)
    # A filler word between any two number pieces keeps them apart as separate runs.
    out: list[Piece] = []
    for p in pieces:
        if out and out[-1].kind != "words" and p.kind != "words":
            out.append(Piece("words", draw(st.sampled_from(_FILLERS))))
        out.append(p)
    return name, value, out


def _render(pieces: list[Piece], joins: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text, spans = "", []
    for i, p in enumerate(pieces):
        if i:
            text += joins[i - 1]
        spans.append((len(text), len(text) + len(p.text)))
        text += p.text
    return text, spans


def _collapse(text: str, token: str) -> str:
    """Adjacent tokens (only whitespace or punctuation between) as one, the rule under test."""
    pattern = re.compile(re.escape(token) + r"(?:[\W_]*?" + re.escape(token) + ")+")
    return pattern.sub(token, text)


def _surviving_runs(out: str, token: str) -> list[tuple[str, int, str | None]]:
    """The number runs of ``out`` outside the placeholder tokens: (text, tokens, digits)."""
    bare = out.replace(token, " | ")
    return [(bare[r.start:r.end], r.tokens, r.digits) for r in number_runs(bare, LANGS)]


def _check_output(out: str, value: str, token: str, *, fallback: bool) -> None:
    # (i) no exact-value match, by the scrub's matching and as a plain substring.
    assert not segments.contains_value(out.replace(token, " "), value)
    assert segments._norm_text(value) not in segments._norm_text(out.replace(token, "|"))
    for text, tokens, digits in _surviving_runs(out, token):
        # (iii) one token, one digit.
        assert tokens == 1 and digits is not None and len(digits) == 1, (text, out)
        # (iv) with the fallback, no surviving digit occurs in the value.
        if fallback:
            assert digits not in value, (text, out)


@settings(max_examples=400, deadline=None)
@given(data=st.data(), generated=utterances(), located=st.booleans())
def test_cpm13a_placeholder(data: st.DataObject, generated: tuple[str, str, list[Piece]],
                            located: bool) -> None:
    name, value, pieces = generated
    joins = [data.draw(st.sampled_from(_JOIN)) for _ in range(len(pieces) - 1)]
    text, spans = _render(pieces, joins)
    runs = number_runs(unicodedata.normalize("NFC", text), LANGS)
    by_span = {(r.start, r.end): i for i, r in enumerate(runs)}
    # The generator's number pieces are exactly the runs (fillers keep them apart).
    numeric = [s for p, s in zip(pieces, spans, strict=True) if p.kind != "words"]
    assert sorted(numeric) == [(r.start, r.end) for r in runs], (text, runs)
    value_runs = [by_span[s] for p, s in zip(pieces, spans, strict=True) if p.kind == "value"]
    token = f"[{name}]"

    out = placeholder_text(text, _slot(name), value, value_runs=value_runs if located else None)

    def kept(p: Piece) -> bool:
        if p.kind == "words":
            return True
        if p.kind == "lone":
            return located or p.digits not in value
        return False  # (ii) the value's runs, and the runs that always go

    expected = "".join((j if i else "") + (p.text if kept(p) else token)
                       for i, (p, j) in enumerate(zip(pieces, ["", *joins], strict=True)))
    assert out == _collapse(expected, token), (text, value_runs if located else None)
    _check_output(out, value, token, fallback=not located)


_ANY_PIECE = st.one_of(
    st.text(max_size=6),
    st.sampled_from(sorted(NUMBER_WORDS["en"] | NUMBER_WORDS["es"])),
    st.text("0123456789", min_size=1, max_size=6),
    st.sampled_from(["and", "y", "con", " ", ", ", "-", "/", ".", "!", "_", "١", "٣", "½", "７",
                     "²", "1st", "o", "una", "Ó", "́"]),
)


@settings(max_examples=400, deadline=None)
@given(name=st.sampled_from(sorted(SEED_SLOTS)), parts=st.lists(_ANY_PIECE, max_size=12),
       data=st.data(), located=st.booleans())
def test_cpm13a_over_arbitrary_text(name: str, parts: list[str], data: st.DataObject,
                                    located: bool) -> None:
    """(i), (iii), (iv) for any text: values said anywhere, any runs located."""
    value = data.draw(st.text("0123456789", min_size=SEED_SLOTS[name], max_size=SEED_SLOTS[name]))
    if data.draw(st.booleans()):
        parts.insert(data.draw(st.integers(0, len(parts))), data.draw(say_digits(value)))
    text = "".join(parts)
    token = f"[{name}]"
    assume(token not in unicodedata.normalize("NFC", text))
    runs = number_runs(unicodedata.normalize("NFC", text), LANGS)
    value_runs = None
    if located:
        value_runs = sorted(data.draw(st.sets(st.integers(0, max(len(runs) - 1, 0)))))
    out = placeholder_text(text, _slot(name), value, value_runs=value_runs)
    assert not segments.contains_value(out.replace(token, " "), value)
    for run_text, tokens, digits in _surviving_runs(out, token):
        assert tokens == 1 and digits is not None and len(digits) == 1, (run_text, out)
        if value_runs is None:
            assert digits not in value, (run_text, out)


_SPAN_SLOTS = {
    "yes_no": (SlotDecl("yes_no"), ["yes", "no", "sí", "yeah", "that's right", "correct"]),
    "account": (SlotDecl("choice", {"options": [{"value": "savings", "labelKey": "l.s"},
                                                {"value": "checking", "labelKey": "l.c"}]}),
                ["savings", "checking", "ahorros", "the savings one"]),
}
_PLAIN = st.lists(st.sampled_from(["please", "the", "I", "think", "account", "um", "okay",
                                   "por", "favor", "well"]), max_size=4).map(" ".join)


@settings(max_examples=300, deadline=None)
@given(name=st.sampled_from(sorted(_SPAN_SLOTS)), data=st.data(), pre=_PLAIN, post=_PLAIN,
       has_span=st.booleans())
def test_yes_no_and_choice_span(name: str, data: st.DataObject, pre: str, post: str,
                                has_span: bool) -> None:
    """Rule 3: the matched span becomes the token and every other word is kept in order; with a
    value but no located span, the whole utterance becomes the token."""
    decl, phrases = _SPAN_SLOTS[name]
    phrase = data.draw(st.sampled_from(phrases))
    value = data.draw(st.sampled_from(["yes", "no"] if name == "yes_no" else ["savings", "checking"]))
    assume(not segments.contains_value(f"{pre} {post}", value))
    head = f"{pre} " if pre else ""
    text = f"{head}{phrase}{' ' + post if post else ''}"
    slot = SlotSettings(name=name, decl=decl, is_factor=False, sensitive=True, flow="confirm")
    span = (len(head), len(head) + len(phrase)) if has_span else None
    out = placeholder_text(text, slot, value, span=span)
    token = f"[{name}]"
    if not has_span:
        assert out == token
    else:
        assert out == f"{head}{token}{' ' + post if post else ''}"
    assert not segments.contains_value(out.replace(token, " "), value)
