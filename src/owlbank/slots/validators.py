"""Generic slot types: one validator and one DTMF/speech parser per type.

* :func:`validate` is the single deterministic gate. It takes a candidate value
  (from DTMF, the speech parser or the resolver) and returns its canonical form,
  or ``None`` when the value doesn't satisfy the slot. Canonical forms:
  ``digits`` -> the digit string; ``yes_no`` -> ``"yes"``/``"no"``;
  ``choice`` -> the option value; ``amount`` -> a decimal string with two places
  (``"250.00"``); ``date`` -> an ISO date (``"2026-10-15"``).
* :func:`parse_dtmf` and :func:`parse_speech` turn raw input into a candidate;
  their output is always passed through :func:`validate`.

``validate(slot, v)`` is idempotent: ``validate(slot, validate(slot, v)) ==
validate(slot, v)`` (relies on this).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal, InvalidOperation

from . import normalize as nz
from .types import OptionView, SlotDecl, SpeechParse

# -------------------------------------------------------------------- validate


def _digits_ok(slot: SlotDecl, value: str) -> bool:
    if not (value.isascii() and value.isdigit()):
        return False
    c = slot.constraints
    if "length" in c:
        return len(value) == int(c["length"])
    lo = int(c.get("minLength", 1))
    hi = int(c.get("maxLength", 64))
    return lo <= len(value) <= hi


def _to_decimal(value: object) -> Decimal | None:
    try:
        d = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return d if d.is_finite() else None


def _to_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return None


def validate(slot: SlotDecl, value: object, *, today: date | None = None) -> str | None:
    """Return the canonical value if ``value`` satisfies ``slot``, else ``None``."""
    if value is None:
        return None
    if isinstance(value, bool):
        value = "yes" if value else "no"
    s = str(value).strip()
    t = slot.slot_type
    if t == "digits":
        return s if _digits_ok(slot, s) else None
    if t == "yes_no":
        low = s.lower()
        return low if low in ("yes", "no") else None
    if t == "choice":
        values = {o["value"] for o in slot.options()}
        return s if s in values else None
    if t == "amount":
        if not re.fullmatch(r"-?\d+(\.\d{1,2})?", s):
            return None
        d = _to_decimal(s)
        if d is None or d < 0:
            return None
        c = slot.constraints
        lo = _to_decimal(c["min"]) if "min" in c else None
        hi = _to_decimal(c["max"]) if "max" in c else None
        if (lo is not None and d < lo) or (hi is not None and d > hi):
            return None
        return f"{d.quantize(Decimal('0.01'))}"
    if t == "date":
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return None
        day = _to_date(s)
        if day is None:
            return None
        first = _to_date(slot.constraints["min"]) if "min" in slot.constraints else None
        last = _to_date(slot.constraints["max"]) if "max" in slot.constraints else None
        if (first and day < first) or (last and day > last):
            return None
        return day.isoformat()
    return None  # pragma: no cover - SlotDecl rejects unknown types


# ------------------------------------------------------------------------ DTMF

YES_NO_DTMF = {"1": "yes", "2": "no"}


def parse_dtmf(slot: SlotDecl, digits: str) -> str | None:
    """Map keypad input to a candidate value (validated by the caller).

    * ``digits``: as typed (a trailing ``#`` is dropped).
    * ``yes_no``: 1 = yes, 2 = no.
    * ``choice``: each option's ``dtmf`` key, default its 1-based position.
    * ``amount``: whole dollars, ``*`` as the decimal point (``25*50`` = 25.50).
    * ``date``: ``MMDDYYYY``.
    """
    d = (digits or "").strip().rstrip("#")
    if not d:
        return None
    t = slot.slot_type
    if t == "digits":
        return d
    if t == "yes_no":
        return YES_NO_DTMF.get(d)
    if t == "choice":
        for opt in slot.options():
            if opt["dtmf"] == d:
                return opt["value"]
        return None
    if t == "amount":
        if not re.fullmatch(r"\d+(\*\d{1,2})?", d):
            return None
        return d.replace("*", ".")
    if t == "date":
        if not re.fullmatch(r"\d{8}", d):
            return None
        return f"{d[4:]}-{d[:2]}-{d[2:4]}"
    return None


# ---------------------------------------------------------------------- speech

#: Words that may surround a value without changing it ("my savings account",
#: "it's nine four one oh five"). Anything else sends the utterance to the resolver.
CARRIERS: dict[str, frozenset[str]] = {
    "en": frozenset({
        "its", "it", "is", "my", "the", "number", "code", "that", "thats", "would", "be",
        "please", "account", "digits", "ok", "okay",
    }),
    "es": frozenset({
        "es", "mi", "el", "la", "numero", "codigo", "son", "los", "digitos", "seria",
        "cuenta", "de", "por", "favor",
    }),
}

#: Affirmation and denial phrases per language, over normalized text (lowercase, no
#: accents or punctuation, apostrophes joined, fillers such as "um", "uh", "mm" dropped). An answer
#: made only of phrases from one list (and :data:`YES_NO_NEUTRAL` words) resolves without the
#: resolver: see :func:`yes_no_speech`. "uh huh" / "mm hmm" / "uh uh" can't be listed: the
#: normalizer drops "uh" and "mm", leaving "huh" (which also means "what?") or nothing; the
#: one-word spellings ("uhhuh", "mhm", "mmhmm") are listed.
YES_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset({
        "yes", "yeah", "yep", "yup", "yea", "ya", "yes please", "correct", "right", "thats right",
        "thats correct", "that is right", "that is correct", "sure", "ok", "okay", "yes it is",
        "it is", "affirmative", "all right", "alright", "fine", "thats fine", "that is fine",
        "exactly", "absolutely", "definitely", "of course", "sure thing", "you bet", "perfect",
        "sounds good", "yes sir", "yes maam", "uhhuh", "mhm", "mhmm", "mmhm", "mmhmm",
    }),
    "es": frozenset({
        "si", "si por favor", "claro", "claro que si", "correcto", "es correcto", "esta correcto",
        "si es correcto", "eso es correcto", "asi es", "de acuerdo", "esta bien", "ok", "okey",
        "vale", "eso es", "exacto", "exactamente", "por supuesto", "afirmativo", "perfecto",
        "cierto", "es cierto", "aja", "si senor", "si senora", "sale",
    }),
}
NO_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset({
        "no", "nope", "nah", "no thanks", "no thank you", "incorrect", "thats incorrect",
        "that is incorrect", "wrong", "thats wrong", "that is wrong", "not right", "thats not right",
        "that is not right", "not correct", "thats not correct", "that is not correct",
        "negative", "no it isnt", "no its not", "it isnt", "its not", "not really", "not at all",
        "no way", "absolutely not", "definitely not", "of course not", "no sir", "no maam",
    }),
    "es": frozenset({
        "no", "no gracias", "incorrecto", "es incorrecto", "negativo", "no es correcto",
        "eso no es correcto", "no esta correcto", "no es asi", "no esta bien", "no es cierto",
        "esta mal", "para nada", "de ninguna manera", "claro que no", "por supuesto que no",
        "no senor", "no senora",
    }),
}
#: Words that may stand next to an affirmation or denial without changing it ("oh yes", "well,
#: no"). Anything not in these lists sends the utterance to the resolver.
YES_NO_NEUTRAL: dict[str, frozenset[str]] = {
    "en": frozenset({"oh", "well"}),
    "es": frozenset({"ah", "oh", "bueno"}),
}

#: Ordinal references to the options the question just listed, over normalized text, kept
#: with the yes/no lists. The value is the 0-based position; ``-1`` is the last one.
#: "the second one", "la segunda": the carrier words around the ordinal are trimmed as the choice
#: parse trims them (:func:`ordinal_where`).
ORDINAL_WORDS: dict[str, dict[str, int]] = {
    "en": {"first": 0, "second": 1, "third": 2, "last": -1},
    "es": {"primera": 0, "primero": 0, "segunda": 1, "segundo": 1, "tercera": 2, "tercero": 2,
           "ultima": -1, "ultimo": -1},
}


def _phrase_table(fam: str) -> dict[tuple[str, ...], str | None]:
    table: dict[tuple[str, ...], str | None] = {(w,): None for w in YES_NO_NEUTRAL[fam]}
    table.update({tuple(p.split()): "yes" for p in YES_WORDS[fam]})
    table.update({tuple(p.split()): "no" for p in NO_WORDS[fam]})
    return table


_PHRASES: dict[str, dict[tuple[str, ...], str | None]] = {f: _phrase_table(f) for f in YES_WORDS}
_MAX_PHRASE = {f: max(len(k) for k in t) for f, t in _PHRASES.items()}


def yes_no_speech(normalized: str, lang: str) -> str | None:
    """``"yes"`` / ``"no"`` when the normalized utterance splits wholly into known
    affirmation (or denial) phrases and neutral words, else ``None`` (the resolver decides).

    The split with the fewest phrases is used (so "no es correcto" is one denial, not "no" plus
    "es correcto"); if a fewest-phrase split mixes yes and no, or two such splits disagree, or a
    word is covered by no phrase, or there are only neutral words, the answer is ``None``. Pure;
    catalog-independent; a dynamic program over token positions."""
    toks = normalized.split()
    if not toks:
        return None
    fam = nz.lang_family(lang)
    table, longest = _PHRASES[fam], _MAX_PHRASE[fam]
    # best[i]: (fewest phrases covering toks[:i], the polarity sets of all such splits)
    best: list[tuple[int, set[frozenset[str]]] | None] = [None] * (len(toks) + 1)
    best[0] = (0, {frozenset()})
    for i in range(1, len(toks) + 1):
        for j in range(max(0, i - longest), i):
            prev = best[j]
            key = tuple(toks[j:i])
            if prev is None or key not in table:
                continue
            pol = table[key]
            count = prev[0] + 1
            readings = {r | {pol} if pol else r for r in prev[1]}
            cur = best[i]
            if cur is None or count < cur[0]:
                best[i] = (count, readings)
            elif count == cur[0]:
                cur[1].update(readings)
    final = best[len(toks)]
    if final is None or len(final[1]) != 1:
        return None
    (reading,) = final[1]
    return next(iter(reading)) if len(reading) == 1 else None


def _yes_no_where(normalized: str, lang: str) -> tuple[str | None, tuple[int, int] | None]:
    """:func:`yes_no_speech` and the ``[lo, hi)`` normalized tokens of the phrase (the neutral
    words at its edges left out)."""
    value = yes_no_speech(normalized, lang)
    if value is None:
        return None, None
    neutral = YES_NO_NEUTRAL[nz.lang_family(lang)]
    at = [i for i, t in enumerate(normalized.split()) if t not in neutral]
    return value, (at[0], at[-1] + 1)


def _label_words(text: str | None, lang: str) -> set[str]:
    return set(nz.normalize(text, lang).split()) if text else set()


def _parse_digits_speech(slot: SlotDecl, norm: str, lang: str, label: str | None,
                         other_labels: Sequence[str] = (), *, phone: bool = False) -> str | None:
    return _digits_where(slot, norm, lang, label, other_labels, phone=phone)[0]


def names_label(toks: Sequence[str], labels: Sequence[str], lang: str) -> bool:
    """Some label of ``labels`` (catalog text) occurs in ``toks`` (normalized tokens) as a
    contiguous run of whole tokens. The 2b label guard."""
    for text in labels:
        want = nz.normalize(text, lang).split()
        n = len(want)
        if n and any(list(toks[i:i + n]) == want for i in range(len(toks) - n + 1)):
            return True
    return False


def _digits_where(slot: SlotDecl, norm: str, lang: str, label: str | None,
                  other_labels: Sequence[str] = (), *, phone: bool = False
                  ) -> tuple[str | None, tuple[tuple[int, int], ...]]:
    """The digits parse, and the ``[lo, hi)`` ranges of normalized tokens it read them from.

    ``other_labels`` are the labels of the other slots already collected on the call: stage 2b (a
    value inside a longer sentence) refuses a sentence that names one of them ("the zip code is
    1234" at the SSN prompt may be a correction; the resolver decides).

    ``phone`` (keyed-phone, the phone slot only): a reading is a North American number
    (:func:`_as_phone`): eleven digits with a leading 1 are read as the ten after it, and a
    ten-digit reading starting with 0 or 1 is never a value. A number run after the one that
    gives the value ("212 555 1234 no 1235") may be a correction: no value, the resolver decides."""
    value, ranges = _digits_where_raw(slot, norm, lang, label, other_labels, phone=phone)
    if phone and value is not None:
        value = _as_phone(value)
        if _length_ok(slot, value) and not _fits(slot, value, phone=True):
            return None, ()
    return value, ranges


def _digits_where_raw(slot: SlotDecl, norm: str, lang: str, label: str | None,
                      other_labels: Sequence[str], *, phone: bool
                      ) -> tuple[str | None, tuple[tuple[int, int], ...]]:
    fam = nz.lang_family(lang)
    allowed = CARRIERS[fam] | _label_words(label, lang)
    toks = norm.split()
    lo, hi = 0, len(toks)
    # Drop carrier words only from the edges, so "one" inside a number still counts.
    while lo < hi and toks[lo] in allowed and nz.spoken_digits(" ".join(toks[lo:hi]), lang) is None:
        lo += 1
    while lo < hi and toks[hi - 1] in allowed and nz.spoken_digits(" ".join(toks[lo:hi]), lang) is None:
        hi -= 1
    whole = nz.spoken_digits(" ".join(toks[lo:hi]), lang)
    if whole is not None and _fits(slot, _as_phone(whole) if phone else whole, phone=phone):
        return whole, ((lo, hi),)
    # The digits inside a longer sentence ("yeah, the last four are 1111", "my zip is
    # 94105"). Each separate run of number words is a candidate, whole or with a leading count
    # dropped ("the last *four* 1111"; :func:`_run_candidates_where`). Never cut at the end: a
    # value is never the caller's digits with the last ones missing. Accepted only
    # if exactly one distinct value has the slot's length; otherwise the caller is asked again.
    # Only when the caller stopped after the number (at most MAX_WORDS_AFTER words follow it):
    # someone who keeps talking ("…nine four one oh, hold on, what's my balance") may be changing
    # the subject, which the resolver decides. A count, not phrase matching.
    if other_labels and names_label(toks, other_labels, lang):
        return whole, ((lo, hi),) if whole is not None else ()
    sized = [c for c in _run_candidates_where(toks, lang, phone=phone) if _fits(slot, c[0], phone=phone)]
    if len({c[0] for c in sized}) == 1 and min(c[1] for c in sized) <= MAX_WORDS_AFTER:
        end = max(c[3] for c in sized)
        if phone and any(a >= end for a, _ in _number_runs(toks, lang)):
            # Another number follows the value ("212 555 1234 no 1235"): the resolver decides.
            return None, ()
        # exactly one value of the right length anywhere in the sentence: every place it was read
        return sized[0][0], tuple(dict.fromkeys((c[2], c[3]) for c in sized))
    return whole, ((lo, hi),) if whole is not None else ()


def _as_phone(digits: str) -> str:
    """A North American reading: eleven digits ``1`` + a ten-digit number whose first digit
    is 2-9 (the country code said or keyed in front) read as the ten; anything else unchanged."""
    if len(digits) == 11 and digits[0] == "1" and digits[1] in _NANP_FIRST:
        return digits[1:]
    return digits


#: The first digit of a US area code (none starts with 0 or 1).
_NANP_FIRST = frozenset("23456789")


def _fits(slot: SlotDecl, digits: str, *, phone: bool) -> bool:
    """The slot's length, and for the phone slot a first digit of 2-9."""
    return _length_ok(slot, digits) and (not phone or digits[:1] in _NANP_FIRST)


#: Words allowed after a number found inside a sentence ("um 1111 I think").
MAX_WORDS_AFTER = 2


def _length_ok(slot: SlotDecl, digits: str) -> bool:
    c = slot.constraints
    if isinstance(c.get("length"), int):
        return len(digits) == c["length"]
    lo, hi = c.get("minLength"), c.get("maxLength")
    return (not isinstance(lo, int) or len(digits) >= lo) and (not isinstance(hi, int) or len(digits) <= hi)


def short_digit_answer(slot: SlotDecl, norm: str, lang: str, label: str | None = None) -> bool:
    """True when a fixed-length digit answer was cut short: every word is a number word (or a
    carrier word such as "my" or the slot's own label) and they spell fewer digits than the slot
    needs ("one", "nine four one"). The caller is asked again at once, without the resolver,
    which can't recover digits it never heard. Pure; no phrase
    matching beyond the number and carrier word lists."""
    length = slot.constraints.get("length")
    if not isinstance(length, int) or not norm.strip():
        return False
    fam = nz.lang_family(lang)
    allowed = CARRIERS[fam] | _label_words(label, lang)
    toks = norm.split()
    numbers = [t for t in toks if _is_number_token(t, lang)]
    if not numbers or any(t not in allowed and not _is_number_token(t, lang) for t in toks):
        return False
    digits = nz.spoken_digits(" ".join(numbers), lang)
    return digits is not None and 0 < len(digits) < length


def _is_number_token(tok: str, lang: str) -> bool:
    fam = nz.lang_family(lang)
    return (nz._is_digit_token(tok) or tok in nz.UNITS[fam] or tok in nz.TEENS[fam]
            or tok in nz.TENS[fam] or tok in nz.REPEATERS[fam]
            or (nz.TENS_JOINER[fam] is not None and tok == nz.TENS_JOINER[fam]))


#: Number words that are also interjections ("23456, oh, no ..."): only these fall off a run's end.
_INTERJECTION_NUMBER_WORDS = frozenset({"oh", "o"})


def _run_candidates(toks: list[str], lang: str, *, phone: bool = False) -> list[tuple[str, int]]:
    """``(digits, words after)`` from each maximal run of number tokens, whole, and with its first
    token dropped when that token is a count of the digits after it: a single digit of 2 or more
    (a word or a numeral) followed by a digit token, and the rest of the run has exactly that many
    digits ("the last four 1111", "the last 4 1111"), or an "oh" before a digit token ("Oh, 94105":
    an interjection); and with its last token dropped when that is
    "oh" after a digit token ("23456, oh, no actually it is 12345": an interjection, so both values
    are seen). Nothing else is dropped: trimming more would read the caller's number with digits
    missing ("one four oh seven ... five" read as its first ten digits; "one 2 3 4 5" read
    as 2345 for the last four of an SSN). With ``phone`` each value is read as a North
    American number (:func:`_as_phone`: a leading 1 before ten digits is the country code, the
    phone slot only). "Words after" counts the tokens after that part of the run. Pure; no
    phrase matching: only number words."""
    return [(d, after) for d, after, _, _ in _run_candidates_where(toks, lang, phone=phone)]


def _number_runs(toks: Sequence[str], lang: str) -> list[tuple[int, int]]:
    """The ``[start, end)`` token indexes of each maximal run of number tokens."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, tok in enumerate(toks):
        if _is_number_token(tok, lang):
            start = i if start is None else start
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(toks)))
    return runs


def _leading_count(toks: Sequence[str], a: int, b: int, lang: str) -> bool:
    """``toks[a]`` is a count of the digits after it in the run ``[a, b)`` (see
    :func:`_run_candidates`). Never a 1, so a leading one is never dropped here."""
    if b - a < 2 or not nz._is_digit_token(toks[a + 1]):
        return False
    count = nz.spoken_digits(toks[a], lang)
    if count is None or len(count) != 1 or count in ("0", "1"):
        return False
    if nz._is_digit_token(toks[a]) and len(toks[a]) != 1:
        return False
    rest = nz.spoken_digits(" ".join(toks[a + 1:b]), lang)
    return rest is not None and len(rest) == int(count)


def _run_candidates_where(toks: list[str], lang: str, *, phone: bool = False
                          ) -> list[tuple[str, int, int, int]]:
    """:func:`_run_candidates` with each candidate's ``[lo, hi)`` token range."""
    out: list[tuple[str, int, int, int]] = []
    for a, b in _number_runs(toks, lang):
        spans = [(a, b)]
        if _leading_count(toks, a, b, lang) or (
                b - a > 1 and toks[a] in _INTERJECTION_NUMBER_WORDS and nz._is_digit_token(toks[a + 1])):
            spans.append((a + 1, b))
        if b - a > 1 and toks[b - 1] in _INTERJECTION_NUMBER_WORDS and nz._is_digit_token(toks[b - 2]):
            spans.append((a, b - 1))
        for lo, hi in spans:
            d = nz.spoken_digits(" ".join(toks[lo:hi]), lang)
            if d:
                out.append((_as_phone(d) if phone else d, len(toks) - hi, lo, hi))
    return out


def _parse_choice_speech(slot: SlotDecl, norm: str, lang: str,
                         labels: Mapping[str, str]) -> str | None:
    return _choice_where(slot, norm, lang, labels)[0]


def _choice_where(slot: SlotDecl, norm: str, lang: str, labels: Mapping[str, str]
                  ) -> tuple[str | None, tuple[int, int] | None]:
    """The choice parse and the ``[lo, hi)`` normalized tokens that matched the label."""
    fam = nz.lang_family(lang)

    # "the savings one", "my checking account", "cuenta de ahorros"
    def kept(t: str) -> bool:
        return t not in CARRIERS[fam] and t != "one"

    def strip(text: str) -> str:
        return " ".join(t for t in text.split() if kept(t))

    stripped = strip(norm)
    matches = []
    whole = False
    for opt in slot.options():
        label = labels.get(opt["label_key"])
        if not label:
            continue
        label_n = nz.normalize(label, lang)
        if norm == label_n or (stripped and stripped == (strip(label_n) or label_n)):
            matches.append(opt["value"])
            whole = whole or norm == label_n
    if len(matches) != 1:
        return None, None
    toks = norm.split()
    if whole:
        return matches[0], (0, len(toks))
    at = [i for i, t in enumerate(toks) if kept(t)]
    return matches[0], (at[0], at[-1] + 1)


def option_where(options: Sequence[OptionView], speech: str | None, lang: str, *,
                 label: str | None = None) -> tuple[OptionView | None, str | None, SpeechParse]:
    """The known option whose spoken name or alias the caller said, deterministically.

    Matches when the normalized utterance, with the carrier words (:data:`CARRIERS`, "one" and the
    slot's own label words, as the digits and choice parses trim them) left out, equals one
    option's spoken name or alias (the phrase trimmed the same way). Exactly one option must
    match. Returns the option, ``"speech"`` (its spoken name) or ``"alias"``, and a
    :class:`SpeechParse` whose ``span`` locates the words in the NFC speech; ``(None, None,
    SpeechParse(None))`` otherwise. No phrase list beyond the options' own names and aliases."""
    none: tuple[OptionView | None, str | None, SpeechParse] = (None, None, SpeechParse(None))
    if not speech or not options:
        return none
    norm = nz.normalize(speech, lang)
    if not norm:
        return none
    drop = CARRIERS[nz.lang_family(lang)] | _label_words(label, lang) | {"one"}

    def strip(text: str) -> str:
        return " ".join(t for t in text.split() if t not in drop)

    stripped = strip(norm)
    found: dict[str, tuple[OptionView, str]] = {}
    for opt in options:
        for source, phrase in [("speech", opt.name), *(("alias", a) for a in opt.aliases)]:
            phrase_n = nz.normalize(phrase, lang)
            if not phrase_n:
                continue
            if norm == phrase_n or (stripped and stripped == (strip(phrase_n) or phrase_n)):
                prev = found.get(opt.value)
                if prev is None or (prev[1] == "alias" and source == "speech"):
                    found[opt.value] = (opt, source)
    if len(found) != 1:
        return none
    ((opt, source),) = found.values()
    text = unicodedata.normalize("NFC", speech)
    located = nz.normalize_located(text, lang)
    toks = norm.split()
    at = [i for i, t in enumerate(toks) if t not in drop] or list(range(len(toks)))
    if located is None or len(located) != len(toks):
        return opt, source, SpeechParse(opt.value)
    span = (located[at[0]].start, located[at[-1]].end)
    return opt, source, SpeechParse(opt.value, value_runs=(), span=span)


def ordinal_where(text: str | None, lang: str, n: int, *, label: str | None = None
                  ) -> tuple[int, tuple[int, int] | None] | None:
    """The position an ordinal answer picks from a list of ``n`` options just read out,
    deterministically. ``(index, span)`` when the normalized utterance, with the carrier words
    left out (:data:`CARRIERS`, "one" and the slot's own ``label`` words, as :func:`option_where`
    trims them), is exactly one word of :data:`ORDINAL_WORDS`; "last" is ``n - 1``. ``span`` is
    the ordinal word in the NFC text (``None`` when it can't be located). ``None`` for no ordinal,
    more than one, any other word, an ordinal beyond ``n``, or ``n < 1``. Pure."""
    if not text or n < 1:
        return None
    fam = nz.lang_family(lang)
    norm = nz.normalize(text, lang)
    toks = norm.split()
    drop = CARRIERS[fam] | _label_words(label, lang) | {"one"}
    kept = [i for i, t in enumerate(toks) if t not in drop]
    if len(kept) != 1 or toks[kept[0]] not in ORDINAL_WORDS[fam]:
        return None
    pos = ORDINAL_WORDS[fam][toks[kept[0]]]
    index = n - 1 if pos < 0 else pos
    if index >= n:
        return None
    located = nz.normalize_located(unicodedata.normalize("NFC", text), lang)
    if located is None or len(located) != len(toks):
        return index, None
    tok = located[kept[0]]
    return index, (tok.start, tok.end)


def _resolve_year(month: int, day: int, year: int | None, slot: SlotDecl,
                  today: date) -> str | None:
    if year is not None:
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None
    candidates = []
    for y in (today.year - 1, today.year, today.year + 1):
        try:
            candidates.append(date(y, month, day))
        except ValueError:
            continue
    valid = [d for d in candidates if validate(slot, d.isoformat(), today=today)]
    pool = valid or candidates
    if not pool:
        return None
    return min(pool, key=lambda d: abs(d.toordinal() - today.toordinal())).isoformat()


def parse_speech(slot: SlotDecl, normalized: str, lang: str, *,
                 labels: Mapping[str, str] | None = None,
                 today: date | None = None, other_labels: Sequence[str] = (),
                 phone: bool = False) -> str | None:
    """Deterministically parse a normalized utterance into a candidate value. ``other_labels``:
    the 2b label guard; ``phone``: the phone slot's North American reading (both
    :func:`_digits_where`)."""
    labels = labels or {}
    if not normalized:
        return None
    t = slot.slot_type
    if t == "digits":
        return _parse_digits_speech(slot, normalized, lang, labels.get(slot.label_key or ""),
                                    other_labels, phone=phone)
    if t == "yes_no":
        return yes_no_speech(normalized, lang)
    if t == "choice":
        return _parse_choice_speech(slot, normalized, lang, labels)
    if t == "amount":
        currency = str(slot.constraints.get("currency", "USD")).upper()
        if currency != "USD":
            return None
        d = nz.parse_amount(normalized, lang)
        return None if d is None else f"{d}"
    if t == "date":
        parts = nz.parse_date_parts(normalized, lang, today or date.today())
        if parts is None:
            return None
        return _resolve_year(*parts, slot=slot, today=today or date.today())
    return None


#: :func:`parse_speech_located` reports number runs over both languages, as the placeholder reads
#: them.
_RUN_LANGS = ("en", "es")


def parse_speech_located(slot: SlotDecl, speech: str | None, lang: str, *,
                         labels: Mapping[str, str] | None = None,
                         today: date | None = None,
                         other_labels: Sequence[str] = (),
                         phone: bool = False) -> SpeechParse:
    """:func:`parse_speech` of ``normalize(speech, lang)``, plus where in the caller's words the
    value was read. The value is always the one
    :func:`parse_speech` gives.

    ``digits``: ``value_runs``, the indices into ``number_runs(NFC(speech), ("en", "es"))`` of
    the runs the parse read the value from. ``yes_no`` and ``choice``: ``span``, the matched phrase
    in the NFC speech (``value_runs`` empty). ``amount`` and ``date``, or words that can't be
    located: neither (``None``)."""
    labels = labels or {}
    norm = nz.normalize(speech, lang)
    value = parse_speech(slot, norm, lang, labels=labels, today=today, other_labels=other_labels,
                         phone=phone)
    if value is None or not speech:
        return SpeechParse(value)
    t = slot.slot_type
    ranges: tuple[tuple[int, int], ...]
    if t == "digits":
        got, ranges = _digits_where(slot, norm, lang, labels.get(slot.label_key or ""), other_labels,
                                    phone=phone)
    elif t == "yes_no":
        got, rng = _yes_no_where(norm, lang)
        ranges = () if rng is None else (rng,)
    elif t == "choice":
        got, rng = _choice_where(slot, norm, lang, labels)
        ranges = () if rng is None else (rng,)
    else:
        return SpeechParse(value)
    text = unicodedata.normalize("NFC", speech)
    located = nz.normalize_located(text, lang)
    if got != value or not ranges or located is None:
        return SpeechParse(value)
    spans = [(located[lo].start, located[hi - 1].end) for lo, hi in ranges if hi > lo]
    if not spans:
        return SpeechParse(value)
    if t == "digits":
        runs = nz.number_runs(text, _RUN_LANGS)
        picked = tuple(i for i, r in enumerate(runs)
                       if any(r.start < end and start < r.end for start, end in spans))
        return SpeechParse(value, value_runs=picked)
    return SpeechParse(value, value_runs=(), span=(min(s for s, _ in spans), max(e for _, e in spans)))


# -------------------------------------------------------------------- early finish

#: Outcomes of :func:`early_digits`.
EARLY_OUTCOMES = ("incomplete", "ambiguous", "complete")
#: Number words that a later word can still extend ("twenty" -> "twenty five", "double" -> "double
#: five"), so a partial result ending in one is never complete. Joiners ("y") too.
_EXTENDABLE_WORDS = frozenset({"hundred", "thousand", "million", "point", "cien", "ciento", "mil",
                               "millon", "millones", "punto"})


def _extendable(tok: str, lang: str) -> bool:
    fam = nz.lang_family(lang)
    if tok in nz.TENS[fam] or tok in nz.REPEATERS[fam] or tok in _EXTENDABLE_WORDS:
        return True
    if nz.TENS_JOINER[fam] is not None and tok == nz.TENS_JOINER[fam]:
        return True
    # A recognizer that writes numerals may show "20" before it hears "twenty five" ("25").
    return nz._is_digit_token(tok) and len(tok) == 2 and tok[1] == "0" and tok[0] != "0"


def early_digits(slot: SlotDecl, speech: str | None, lang: str, *,
                 label: str | None = None,
                 other_labels: Sequence[str] = (),
                 phone: bool = False) -> tuple[str, str | None]:
    """Is a *partial* speech result already a complete answer for a fixed-length
    digits slot? ``("complete", value)`` only when the deterministic speech parser (steps
    2-2b, the same one :func:`parse_speech` runs; no resolver) gives a valid value of exactly
    the slot's ``length``, no other distinct value of that length occurs in the text, and the last
    word can't still grow into another number ("twenty" may become "twenty five").
    ``("ambiguous", None)`` when two or more distinct values of the right length occur;
    otherwise ``("incomplete", None)``. ``other_labels``: the 2b label guard, as for the final
    answer (a partial that names another collected slot never finishes early). ``phone``:
    the phone slot's reading, as for the final answer (:func:`_digits_where`). Pure; the
    value is never logged by callers."""
    length = slot.constraints.get("length")
    if slot.slot_type != "digits" or not isinstance(length, int) or isinstance(length, bool):
        return "incomplete", None
    norm = nz.normalize(speech, lang)
    toks = norm.split()
    if not toks or _extendable(toks[-1], lang):
        return "incomplete", None
    sized = {c for c, _ in _run_candidates(toks, lang, phone=phone) if _fits(slot, c, phone=phone)}
    if len(sized) > 1:
        return "ambiguous", None
    value = validate(slot, _parse_digits_speech(slot, norm, lang, label, other_labels, phone=phone))
    if value is None or len(value) != length or (sized and value not in sized):
        return "incomplete", None
    return "complete", value


def early_phone(slot: SlotDecl, speech: str | None, lang: str, *,
                label: str | None = None,
                other_labels: Sequence[str] = ()) -> tuple[str, str | None]:
    """Early finish: :func:`early_digits` for the ten-digit US phone slot, with the same
    reading as its final answer (``phone``, :func:`_digits_where`). A reading is complete only as
    ten digits whose first is 2-9 (no US area code starts with 0 or 1), or as eleven with a
    leading 1, read as the ten after it. A ten-digit reading starting with 0 or 1 (for example a
    spoken "one" and nine digits) is never complete, nor is one followed by another number
    ("212 555 1234 no 1235"). Same outcomes as :func:`early_digits`; pure; callers never log the
    value."""
    return early_digits(slot, speech, lang, label=label, other_labels=other_labels, phone=True)


def _says_yes_no(toks: Sequence[str], lang: str) -> bool:
    """Some affirmation or denial phrase (lists) occurs in ``toks`` as a contiguous run
    of whole tokens."""
    fam = nz.lang_family(lang)
    for phrase in YES_WORDS[fam] | NO_WORDS[fam]:
        want = phrase.split()
        k = len(want)
        if any(list(toks[i:i + k]) == want for i in range(len(toks) - k + 1)):
            return True
    return False


def early_choice(decl: SlotDecl, text: str | None, lang: str, *,
                 options: Sequence[OptionView], listed: Sequence[str],
                 labels: Mapping[str, str], other_labels: Sequence[str] = (),
                 known: Sequence[OptionView] | None = None) -> tuple[str, str | None]:
    """Is a *partial* speech result already a complete answer for a choice
    slot? ``("complete", value)`` only when the deterministic choice parse the engine runs on a
    final answer gives ``exact`` for exactly one of the caller's ``options``: a spoken name or
    alias (:func:`option_where`) or, when ``listed`` (the order the question read the options
    in) is non-empty, an ordinal (:func:`ordinal_where`). Otherwise ``("incomplete", None)``:
    no match, more than one, any yes/no word in the text (lists), a sentence naming
    another collected slot's label (``other_labels``, the 2b guard), or anything only the
    model could resolve.

    ``known`` are the options the final parse matches names against (the engine's known options;
    default ``options``): a name shared with an option the caller doesn't hold is ambiguous there,
    so it never finishes early. ``labels`` maps catalog keys to text (the slot's own label words
    are trimmed like carriers). Pure; no model call; callers never log the value."""
    incomplete: tuple[str, str | None] = ("incomplete", None)
    if decl.slot_type != "choice" or not text or not options:
        return incomplete
    toks = nz.normalize(text, lang).split()
    if not toks or _says_yes_no(toks, lang):
        return incomplete
    if other_labels and names_label(toks, other_labels, lang):
        return incomplete
    held = {o.value for o in options}
    label = labels.get(decl.label_key or "")
    opt, _, _ = option_where(tuple(options if known is None else known), text, lang, label=label)
    if opt is not None:
        return ("complete", opt.value) if opt.value in held else incomplete
    if listed:
        hit = ordinal_where(text, lang, len(listed), label=label)
        if hit is not None and listed[hit[0]] in held:
            return "complete", listed[hit[0]]
    return incomplete
