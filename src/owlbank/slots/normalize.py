"""Per-language speech normalization: text cleanup, spoken digits, amounts, dates.

Supported languages: ``en-US`` and ``es-US`` (anything starting ``es`` uses the
Spanish tables, everything else English). All matching is case- and
accent-insensitive: :func:`normalize` lowercases and strips diacritics, so
"Dólares", "dolares" and "DÓLARES" are the same token.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

# --------------------------------------------------------------------------- text


def lang_family(lang: str) -> str:
    return "es" if (lang or "").lower().startswith("es") else "en"


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


FILLERS: dict[str, frozenset[str]] = {
    "en": frozenset({"um", "umm", "uh", "uhh", "uhm", "erm", "er", "ah", "hmm", "hm", "mm", "mmm"}),
    "es": frozenset({"eh", "em", "ehm", "emm", "mmm", "mm", "pues"}),
}

_DOLLAR_RE = re.compile(r"\$\s*(\d[\d,]*(?:\.\d+)?)")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_ORDINAL_SUFFIX_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th|ro|do|to|vo|no|o|a)\b")
_NON_WORD_RE = re.compile(r"[^\w\s.]|_")
_STRAY_DOT_RE = re.compile(r"(?<!\d)\.|\.(?!\d)")


def normalize(text: str | None, lang: str) -> str:
    """Lowercase, strip accents and punctuation, drop filler words.

    Keeps a decimal point between digits ("250.50"), drops thousands commas,
    turns "$250" into "250 dollars" and "15th" into "15", and joins apostrophes
    ("it's" -> "its").
    """
    if not text:
        return ""
    fam = lang_family(lang)
    s = strip_accents(text).lower()
    s = _DOLLAR_RE.sub(lambda m: f" {m.group(1)} dollars ", s)
    s = _THOUSANDS_RE.sub("", s)
    s = _ORDINAL_SUFFIX_RE.sub(r"\1", s)
    s = s.replace("'", "").replace("’", "")
    s = _NON_WORD_RE.sub(" ", s)
    s = _STRAY_DOT_RE.sub(" ", s)  # keep only decimal points between digits
    tokens = [t for t in s.split() if t not in FILLERS[fam]]
    return " ".join(tokens)


@dataclass(frozen=True, slots=True)
class LocatedToken:
    """One token of :func:`normalize`'s output and where it came from in the input text."""

    text: str
    start: int
    end: int


def _sub_located(chars: list[str], origins: list[int], pattern: re.Pattern[str],
                 repl: str) -> tuple[list[str], list[int]]:
    """``pattern.sub`` over ``chars`` that also carries each character's origin. ``repl`` is a
    fixed string (every character takes the match start's origin), ``"group1"`` (the match is
    replaced by its first group, which keeps its own origins) or ``"dollars"`` (``$250`` becomes
    `` 250 dollars ``, as in :func:`normalize`)."""
    s = "".join(chars)
    out_c: list[str] = []
    out_o: list[int] = []
    pos = 0
    for m in pattern.finditer(s):
        out_c.extend(chars[pos:m.start()])
        out_o.extend(origins[pos:m.start()])
        if repl == "group1":
            out_c.extend(chars[m.start(1):m.end(1)])
            out_o.extend(origins[m.start(1):m.end(1)])
        elif repl == "dollars":
            at = origins[m.start()]
            out_c.append(" ")
            out_o.append(at)
            out_c.extend(chars[m.start(1):m.end(1)])
            out_o.extend(origins[m.start(1):m.end(1)])
            out_c.extend(" dollars ")
            out_o.extend([at] * len(" dollars "))
        else:
            out_c.extend(repl)
            out_o.extend([origins[m.start()]] * len(repl))
        pos = m.end()
    out_c.extend(chars[pos:])
    out_o.extend(origins[pos:])
    return out_c, out_o


_APOSTROPHE_RE = re.compile(r"['’]")
_NOT_SPACE_RE = re.compile(r"\S+")


def normalize_located(text: str | None, lang: str) -> list[LocatedToken] | None:
    """:func:`normalize`'s tokens with the ``[start, end)`` they came from in ``text`` (model-assisted
    slots: the deterministic parse reports which number runs or which span of the
    caller's words it read).

    The same steps as :func:`normalize`, character by character. ``None`` when the tokens don't
    join to exactly ``normalize(text, lang)`` (a case-folding corner such as a final sigma): then
    the caller has no location and falls back (fail closed)."""
    if not text:
        return []
    fam = lang_family(lang)
    chars: list[str] = []
    origins: list[int] = []
    for i, c in enumerate(text):
        for d in strip_accents(c).lower():
            chars.append(d)
            origins.append(i)
    chars, origins = _sub_located(chars, origins, _DOLLAR_RE, "dollars")
    chars, origins = _sub_located(chars, origins, _THOUSANDS_RE, "")
    chars, origins = _sub_located(chars, origins, _ORDINAL_SUFFIX_RE, "group1")
    chars, origins = _sub_located(chars, origins, _APOSTROPHE_RE, "")
    chars, origins = _sub_located(chars, origins, _NON_WORD_RE, " ")
    chars, origins = _sub_located(chars, origins, _STRAY_DOT_RE, " ")
    s = "".join(chars)
    out = [LocatedToken(m.group(), origins[m.start()], origins[m.end() - 1] + 1)
           for m in _NOT_SPACE_RE.finditer(s) if m.group() not in FILLERS[fam]]
    if " ".join(t.text for t in out) != normalize(text, lang):
        return None
    return out


# ------------------------------------------------------------------ number words

_EN_UNITS = {
    "zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_EN_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_EN_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_ES_UNITS = {
    "cero": 0, "uno": 1, "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
}
_ES_TEENS = {
    "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
    "veintiuno": 21, "veintiun": 21, "veintiuna": 21, "veintidos": 22, "veintitres": 23,
    "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26, "veintisiete": 27,
    "veintiocho": 28, "veintinueve": 29,
}
_ES_TENS = {
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
    "ochenta": 80, "noventa": 90,
}
_ES_HUNDREDS = {
    "cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200, "trescientos": 300,
    "trescientas": 300, "cuatrocientos": 400, "cuatrocientas": 400, "quinientos": 500,
    "quinientas": 500, "seiscientos": 600, "seiscientas": 600, "setecientos": 700,
    "setecientas": 700, "ochocientos": 800, "ochocientas": 800, "novecientos": 900,
    "novecientas": 900,
}

UNITS = {"en": _EN_UNITS, "es": _ES_UNITS}
TEENS = {"en": _EN_TEENS, "es": _ES_TEENS}
TENS = {"en": _EN_TENS, "es": _ES_TENS}
REPEATERS = {"en": {"double": 2, "triple": 3}, "es": {"doble": 2, "triple": 3}}
TENS_JOINER = {"en": None, "es": "y"}

_EN_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}
_ES_ORDINALS = {"primero": 1, "primer": 1}
ORDINALS = {"en": _EN_ORDINALS, "es": _ES_ORDINALS}

#: Every word that can carry part of a number, per language (used for redaction).
NUMBER_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        list(_EN_UNITS) + list(_EN_TEENS) + list(_EN_TENS) + list(_EN_ORDINALS)
        + ["double", "triple", "hundred", "thousand", "million", "point"]
    ),
    "es": frozenset(
        list(_ES_UNITS) + list(_ES_TEENS) + list(_ES_TENS) + list(_ES_HUNDREDS)
        + list(_ES_ORDINALS) + ["doble", "triple", "mil", "millon", "millones", "punto"]
    ),
}


def _is_digit_token(tok: str) -> bool:
    return tok.isascii() and tok.isdigit()


def spoken_digits(normalized: str, lang: str) -> str | None:
    """Read a whole normalized utterance as a digit string, or ``None``.

    Every token must be part of the number: digit tokens ("94105", "9"), unit words
    (including "oh"/"zero"/"cero"), teens, tens with an optional unit ("twenty one",
    "treinta y uno"), and repeaters ("double five", "doble cinco"). Used for ZIP,
    PIN, OTP and last-four answers, where digits are read one by one.
    """
    fam = lang_family(lang)
    toks = normalized.split()
    if not toks:
        return None
    units, teens, tens, rep = UNITS[fam], TEENS[fam], TENS[fam], REPEATERS[fam]
    joiner = TENS_JOINER[fam]
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if _is_digit_token(t):
            out.append(t)
            i += 1
        elif t in rep:
            if i + 1 >= len(toks):
                return None
            nxt = toks[i + 1]
            if nxt in units:
                d = str(units[nxt])
            elif _is_digit_token(nxt) and len(nxt) == 1:
                d = nxt
            else:
                return None
            out.append(d * rep[t])
            i += 2
        elif t in units:
            out.append(str(units[t]))
            i += 1
        elif t in teens:
            out.append(str(teens[t]))
            i += 1
        elif t in tens:
            value = tens[t]
            j = i + 1
            if joiner and j + 1 < len(toks) and toks[j] == joiner and toks[j + 1] in units \
                    and units[toks[j + 1]] > 0:
                value += units[toks[j + 1]]
                j += 2
            elif joiner is None and j < len(toks) and toks[j] in units and units[toks[j]] > 0 \
                    and toks[j] not in ("oh", "o"):
                value += units[toks[j]]
                j += 1
            out.append(str(value))
            i = j
        else:
            return None
    return "".join(out)


# --------------------------------------------------------------- cardinal numbers


def _parse_below_1000_en(toks: list[str], i: int) -> tuple[int, int] | None:
    """Parse an English number < 1000 at ``toks[i]``; return (value, next index)."""
    units = {k: v for k, v in _EN_UNITS.items() if k not in ("oh", "o")}
    start = i
    value = 0
    if i < len(toks) and (toks[i] in units or toks[i] == "a") and i + 1 < len(toks) \
            and toks[i + 1] == "hundred":
        value = (1 if toks[i] == "a" else units[toks[i]]) * 100
        i += 2
        if i + 1 < len(toks) and toks[i] == "and" and (
            toks[i + 1] in units or toks[i + 1] in _EN_TEENS or toks[i + 1] in _EN_TENS
        ):
            i += 1
    if i < len(toks) and toks[i] in _EN_TENS:
        value += _EN_TENS[toks[i]]
        i += 1
        if i < len(toks) and toks[i] in units and units[toks[i]] > 0:
            value += units[toks[i]]
            i += 1
    elif i < len(toks) and toks[i] in _EN_TEENS:
        value += _EN_TEENS[toks[i]]
        i += 1
    elif i < len(toks) and toks[i] in units:
        value += units[toks[i]]
        i += 1
    if i == start:
        return None
    return value, i


def _parse_below_1000_es(toks: list[str], i: int) -> tuple[int, int] | None:
    start = i
    value = 0
    if i < len(toks) and toks[i] in _ES_HUNDREDS:
        value = _ES_HUNDREDS[toks[i]]
        i += 1
        if toks[i - 1] == "cien":
            return value, i
    if i < len(toks) and toks[i] in _ES_TENS:
        value += _ES_TENS[toks[i]]
        i += 1
        if i + 1 < len(toks) and toks[i] == "y" and toks[i + 1] in _ES_UNITS \
                and _ES_UNITS[toks[i + 1]] > 0:
            value += _ES_UNITS[toks[i + 1]]
            i += 2
    elif i < len(toks) and toks[i] in _ES_TEENS:
        value += _ES_TEENS[toks[i]]
        i += 1
    elif i < len(toks) and toks[i] in _ES_UNITS:
        value += _ES_UNITS[toks[i]]
        i += 1
    if i == start:
        return None
    return value, i


def parse_cardinal(toks: list[str], i: int, lang: str) -> tuple[int, int] | None:
    """Parse a cardinal number (< 1,000,000) starting at ``toks[i]``.

    Accepts a single digit token ("250") or number words ("two hundred fifty",
    "one thousand two hundred", "doscientos cincuenta", "mil quinientos").
    Returns ``(value, next index)`` or ``None``.
    """
    if i >= len(toks):
        return None
    if _is_digit_token(toks[i]):
        return int(toks[i]), i + 1
    fam = lang_family(lang)
    below = _parse_below_1000_es if fam == "es" else _parse_below_1000_en
    thousand_word = "mil" if fam == "es" else "thousand"
    total = 0
    first = below(toks, i)
    if first is None:
        if fam == "es" and toks[i] == "mil":  # "mil" alone is 1000
            first = (1, i)
        else:
            return None
    value, j = first
    if j < len(toks) and toks[j] == thousand_word:
        total = max(value, 1) * 1000
        j += 1
        if fam == "en" and j + 1 < len(toks) and toks[j] == "and" and below(toks, j + 1) is not None:
            j += 1
        rest = below(toks, j)
        if rest is not None:
            total += rest[0]
            j = rest[1]
        return total, j
    return value, j


# --------------------------------------------------------------------- amounts

CURRENCY_WORDS = {
    "en": {"major": {"dollars", "dollar", "bucks", "usd"}, "minor": {"cents", "cent"},
           "joiners": {"and"}},
    "es": {"major": {"dolares", "dolar", "usd"}, "minor": {"centavos", "centavo"},
           "joiners": {"con", "y"}},
}
_AMOUNT_CARRIERS = {
    "en": {"its", "it", "is", "about", "exactly", "just", "pay", "make", "please"},
    "es": {"son", "es", "unos", "exactamente", "por", "favor", "pagar"},
}


def _number_token(toks: list[str], i: int, lang: str) -> tuple[Decimal, int] | None:
    if i < len(toks) and re.fullmatch(r"\d+\.\d+", toks[i]):
        return Decimal(toks[i]), i + 1
    card = parse_cardinal(toks, i, lang)
    if card is None:
        return None
    return Decimal(card[0]), card[1]


def parse_amount(normalized: str, lang: str) -> Decimal | None:
    """Parse a spoken or written amount from a normalized utterance.

    "two hundred fifty dollars", "250 dollars" (from "$250"), "250.50",
    "fifty dollars and twenty five cents", "doscientos cincuenta",
    "cien dolares con cincuenta centavos". The whole utterance must be the amount
    (plus a few carrier words such as "it's"); otherwise ``None``.
    """
    fam = lang_family(lang)
    words = CURRENCY_WORDS[fam]
    toks = [t for t in normalized.split() if t not in _AMOUNT_CARRIERS[fam]]
    if not toks:
        return None
    first = _number_token(toks, 0, lang)
    if first is None:
        return None
    value, i = first
    if i < len(toks) and toks[i] in words["minor"]:  # "fifty cents"
        if i + 1 != len(toks) or value != value.to_integral_value() or value >= 100:
            return None
        return value / 100
    has_major = i < len(toks) and toks[i] in words["major"]
    if has_major:
        i += 1
    if i == len(toks):
        return value
    # Cents only after a currency word: "two dollars fifty" yes, "two fifty" no.
    if not has_major or value != value.to_integral_value():
        return None
    if toks[i] in words["joiners"]:
        i += 1
    cents = parse_cardinal(toks, i, lang)
    if cents is None:
        return None
    c, i = cents
    if i < len(toks) and toks[i] in words["minor"]:
        i += 1
    if i != len(toks) or c >= 100:
        return None
    return value + Decimal(c) / 100


# ------------------------------------------------------------------------ dates

MONTHS = {
    "en": {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
        "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
        "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    },
    "es": {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
        "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11,
        "diciembre": 12,
    },
}
_DATE_STOP = {
    "en": {"the", "of", "on", "its", "it", "is", "please"},
    "es": {"el", "de", "del", "es", "por", "favor"},
}
_RELATIVE = {"en": {"today": 0, "tomorrow": 1}, "es": {"hoy": 0, "manana": 1}}


def _parse_day(toks: list[str], i: int, lang: str) -> tuple[int, int] | None:
    fam = lang_family(lang)
    if i >= len(toks):
        return None
    t = toks[i]
    if t in ORDINALS[fam]:
        return ORDINALS[fam][t], i + 1
    if fam == "en" and t in _EN_TENS and i + 1 < len(toks) and toks[i + 1] in _EN_ORDINALS \
            and _EN_ORDINALS[toks[i + 1]] < 10:
        return _EN_TENS[t] + _EN_ORDINALS[toks[i + 1]], i + 2
    card = parse_cardinal(toks, i, lang)
    if card is None:
        return None
    return card


def _parse_year(toks: list[str], i: int, lang: str) -> tuple[int, int] | None:
    if i >= len(toks):
        return None
    card = parse_cardinal(toks, i, lang)
    if card is not None and 1900 <= card[0] <= 2100:
        return card
    # "twenty twenty six"
    fam = lang_family(lang)
    if fam == "en":
        a = _parse_below_1000_en(toks, i)
        if a and 19 <= a[0] <= 21:
            b = _parse_below_1000_en(toks, a[1])
            if b and b[0] < 100:
                return a[0] * 100 + b[0], b[1]
    return None


def parse_date_parts(normalized: str, lang: str, today: date) -> tuple[int, int, int | None] | None:
    """Return ``(month, day, year or None)``, or ``None`` when it isn't a date."""
    fam = lang_family(lang)
    toks = [t for t in normalized.split() if t not in _DATE_STOP[fam]]
    if len(toks) == 1 and toks[0] in _RELATIVE[fam]:
        d = date.fromordinal(today.toordinal() + _RELATIVE[fam][toks[0]])
        return d.month, d.day, d.year
    months = MONTHS[fam]
    if not toks:
        return None
    month = day = None
    i = 0
    if toks[0] in months:  # "october fifteenth"
        month = months[toks[0]]
        parsed = _parse_day(toks, 1, lang)
        if parsed is None:
            return None
        day, i = parsed
    else:  # "fifteenth of october", "quince de octubre"
        parsed = _parse_day(toks, 0, lang)
        if parsed is None or parsed[1] >= len(toks) or toks[parsed[1]] not in months:
            return None
        day, i = parsed
        month = months[toks[i]]
        i += 1
    year = None
    if i < len(toks):
        y = _parse_year(toks, i, lang)
        if y is None or y[1] != len(toks):
            return None
        year = y[0]
    return month, day, year


# ------------------------------------------------------------------ number runs

_TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*")
_RUN_SEP_RE = re.compile(r"[\s,.\-/]*")
_JOINERS = {"en": frozenset({"and"}), "es": frozenset({"y", "con"})}


def _token_key(tok: str) -> str:
    """The token as :func:`normalize` would see it (accents, case, apostrophes)."""
    return strip_accents(tok).lower().replace("'", "").replace("\u2019", "")


def _is_numeric_token(tok: str, words: frozenset[str]) -> bool:
    if any(c.isnumeric() for c in tok) or any(c.isnumeric() for c in strip_accents(tok)):
        return True
    return _token_key(tok) in words


def _token_count(tok: str) -> int:
    """Number tokens in one word token: a token of k numeric characters counts as k."""
    k = max(sum(c.isnumeric() for c in tok), sum(c.isnumeric() for c in strip_accents(tok)))
    return max(k, 1)


@dataclass(frozen=True, slots=True)
class NumberRun:
    """A run of digits or number words in a text.

    ``start``/``end`` index the text passed to :func:`number_runs`. ``tokens`` counts its number
    tokens, joiners excluded, a digit token of k digits counted as k; a run made only of a joiner
    (one punctuation can isolate, "five; and; six") counts as 1. ``digits`` is the run read as a
    digit string by :func:`spoken_digits` in the first language that reads it, else ``None``.
    """

    start: int
    end: int
    tokens: int
    digits: str | None


def number_runs(text: str, langs: Sequence[str]) -> list[NumberRun]:
    """Every run of digits or number words in ``text``, over the union of ``langs``' word lists.

    A token is numeric when it holds a numeric character or is a number word of any of the
    languages (compared the way :func:`normalize` reads it: accent-, case- and
    apostrophe-insensitive). A joiner ("and", "y", "con") between two numeric tokens joins the
    run, and a run continues across separators (whitespace, ``,``, ``.``, ``-``, ``/``).
    Offsets index ``text`` as given: callers that want composed and decomposed accents read
    alike pass NFC text, as :func:`owlbank.sensitive.placeholder_text` does.
    """
    fams = list(dict.fromkeys(lang_family(lang) for lang in langs))
    words = frozenset().union(*(NUMBER_WORDS[f] for f in fams))
    joiners = frozenset().union(*(_JOINERS[f] for f in fams))
    toks = list(_TOKEN_RE.finditer(text))
    is_number = [_is_numeric_token(m.group(), words) for m in toks]
    numeric = list(is_number)
    # A joiner ("and", "y", "con") between two numeric tokens is part of the run.
    for k in range(1, len(toks) - 1):
        if not numeric[k] and _token_key(toks[k].group()) in joiners \
                and numeric[k - 1] and numeric[k + 1]:
            numeric[k] = True
    runs: list[NumberRun] = []
    k = 0
    while k < len(toks):
        if not numeric[k]:
            k += 1
            continue
        start = toks[k].start()
        end = toks[k].end()
        j = k + 1
        while j < len(toks) and numeric[j] and _RUN_SEP_RE.fullmatch(text[end:toks[j].start()]):
            end = toks[j].end()
            j += 1
        count = sum(_token_count(toks[i].group()) for i in range(k, j) if is_number[i])
        runs.append(NumberRun(start, end, max(count, 1), _read_digits(text[start:end], langs)))
        k = j
    return runs


def _read_digits(run_text: str, langs: Sequence[str]) -> str | None:
    for lang in langs:
        d = spoken_digits(normalize(run_text, lang), lang)
        if d:
            return d
    return None
