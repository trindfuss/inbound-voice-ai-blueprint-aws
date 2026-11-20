"""Property tests for the generative check.

* For any text (built from placeholders for allowed, disallowed and
  uncatalogued options, stray braces, digits in several scripts, en/es number words and ordinals,
  the Spanish articles, safe words, characters healing removes and arbitrary Unicode, of any length)
  and any ``say_lang``, ``check`` accepts iff every rule holds on the healed text, as computed by an
  independent oracle below whose number words are the call language's only; ``fallback_reason``
  names the first rule broken.
* An accepted text's ``render`` is the healed text with each placeholder
  replaced by that option's catalog spoken name and nothing else changed, and it holds no numeric
  character.
* ``heal`` output holds no Cc/Cf character, surrogate or non-character, nor any
  whitespace other than single inner spaces; it is NFC and idempotent. Any text that passes
  ``check`` renders TwiML that parses as XML with exactly the one ``<Say>`` and no child element,
  whose text is the rendered text: markup-like text never adds an element.
"""

from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET

from hypothesis import event, given, settings
from hypothesis import strategies as st

from owlbank import twiml
from owlbank.catalog import PromptCatalog
from owlbank.slots.generative import check, fallback_reason, heal, render
from owlbank.slots.normalize import NUMBER_WORDS as ALL_NUMBER_WORDS

EN, ES = "en-US", "es-US"
LANGS = (EN, ES)
CATALOGUED = {EN: ("red", "purple", "rewards", "travel_2x"), ES: ("red", "purple", "travel_2x")}
SPOKEN = {"red": ("the red card", "la tarjeta roja"), "purple": ("the purple card", "la morada"),
          "rewards": ("the rewards card", "la de recompensas"),
          "travel_2x": ("the travel card", "la tarjeta de viaje")}
CATALOG = PromptCatalog.from_dict({
    "defaultLanguage": EN,
    "languages": {EN: {}, ES: {}},
    "entries": [
        {"key": f"option.card.{v}", "lang": lang, "text": SPOKEN[v][LANGS.index(lang)]}
        for lang in LANGS for v in CATALOGUED[lang]
    ],
})
VALUES = ["red", "purple", "rewards", "travel_2x", "gold", "x"]

#: The oracle's number words, written out independently of the module's: the call language's
#: digit words, with Spanish's articles "un" and "una" dropped and Spanish ordinals added.
_ES_ORDINALS = {"primero", "primera", "primer", "segundo", "segunda", "tercero", "tercera",
                "tercer", "cuarto", "cuarta", "quinto", "quinta", "sexto", "sexta", "septimo",
                "septima", "octavo", "octava", "noveno", "novena", "decimo", "decima"}
ORACLE_WORDS = {EN: ALL_NUMBER_WORDS["en"],
                ES: (ALL_NUMBER_WORDS["es"] - {"un", "una"}) | _ES_ORDINALS}

_NUMBER_WORDS = sorted(ORACLE_WORDS[EN] | ORACLE_WORDS[ES])
_CASED_NUMBER_WORDS = _NUMBER_WORDS + ["One", "DOS", "Úno", "veintidós", "Cero", "Segunda",
                                       "décimo", "un", "una", "Una", "second", "once", "o"]
_SAFE = ["do", "you", "mean", "or", "the", "card", "¿quiere", "usar", "la", "tarjeta", "roja",
         "which", "would", "like", "?", "¿", ",", ".", " ", "  ", "it's", "señor", "\n"]
_DIGITS = ["0", "7", "42", "٣", "७", "①", "½", "²", "Ⅳ", "１"]
_BRACES = ["{", "}", "{}", "{option:}", "{option:Red}", "{option: red}", "{card:red}", "{{", "}}",
           "{option:red", "option:red}"]
#: Characters healing removes or turns into a space, and markup-like text.
_HEALED = ["\x1b", "\x00", "\x7f", "\x85", "\t", "\r", "\n", "\u200b", "\u200d", "\ufeff", "\u2066",
           "\u00ad", "\ufffe", "\uffff", "\ufdd0", "\U0010ffff", "\ud800", "\udfff", "\u2028",
           "\u00a0", "\u0301", "e\u0301", "\u212b"]
_MARKUP = ["<break/>", "<speak>", "</speak>", "</Say>", "<Hangup/>", '<break time="1s"/>', "&",
           "&amp;", "&#0;", "]]>", "<![CDATA[", "<!--", "-->", "<?xml?>", '"', "'", "<", ">"]

_surrogate = st.integers(0xD800, 0xDFFF).map(chr)
_noncharacter = st.one_of(st.integers(0xFDD0, 0xFDEF),
                          st.integers(0, 16).map(lambda p: p * 0x10000 + 0xFFFE),
                          st.integers(0, 16).map(lambda p: p * 0x10000 + 0xFFFF)).map(chr)
_any_char = st.one_of(st.characters(), st.characters(max_codepoint=0x7F), _surrogate, _noncharacter,
                      st.characters(categories=["Cc", "Cf", "Zs", "Zl", "Zp", "Mn", "Co", "Cn"]))

_segment = st.one_of(
    st.sampled_from(VALUES).map(lambda v: f"{{option:{v}}}"),
    st.sampled_from(_CASED_NUMBER_WORDS),
    st.sampled_from(_SAFE),
    st.sampled_from(_DIGITS),
    st.sampled_from(_BRACES),
    st.sampled_from(_HEALED),
    st.sampled_from(_MARKUP),
    st.text(_any_char, max_size=8),
)
_text = st.one_of(
    st.none(),
    st.lists(_segment, max_size=30).map(lambda parts: " ".join(parts)),
    st.lists(_segment, max_size=30).map("".join),
    st.lists(st.sampled_from(_SAFE + ["{option:red}"]), min_size=30, max_size=70).map(" ".join),
    st.text(),
    st.text(_any_char),
)
_say_lang = st.one_of(st.sampled_from([EN, ES, "en", "es", "en-us"]), st.none(), st.text(max_size=6))
_allowed = st.lists(st.sampled_from(VALUES), max_size=4, unique=True)

# ------------------------------------------------------------------ independent oracle

_PH = re.compile(r"\{option:([a-z][a-z0-9_]{0,31})\}")


def _numeric(text: str) -> bool:
    for form in (text, unicodedata.normalize("NFKD", text)):
        if any(unicodedata.numeric(c, None) is not None for c in form):
            return True
    return False


def _words(text: str) -> list[str]:
    folded = "".join(c for c in unicodedata.normalize("NFKD", text)
                     if unicodedata.category(c) != "Mn").lower()
    return [w for w in re.split(r"[\W_]+", folded) if w]


def oracle(text: str | None, lang: str, say_lang: str | None, allowed: list[str]) -> str | None:
    if text is not None:
        text = heal(text)  # heal itself is checked by its own property
    if text is None or text.strip() == "":
        return "missing"
    if say_lang != lang:
        return "lang"
    rest: list[str] = []
    i = 0
    while i < len(text):
        if text[i] in "{}":
            m = _PH.match(text, i)
            if m is None:
                return "placeholder"
            v = m.group(1)
            if v not in allowed or v not in CATALOGUED[lang]:
                return "placeholder"
            rest.append(" ")
            i = m.end()
            continue
        rest.append(text[i])
        i += 1
    body = "".join(rest)
    if _numeric(body) or any(w in ORACLE_WORDS[lang] for w in _words(body)):
        return "number"
    if len(text) > 200:
        return "length"
    return None


# ------------------------------------------------------------------ properties


@settings(max_examples=600)
@given(text=_text, lang=st.sampled_from(LANGS), other=_say_lang, same=st.integers(0, 4),
       allowed=_allowed)
def test_check_accepts_iff_every_rule_holds(text: str | None, lang: str, other: str | None,
                                                  same: int, allowed: list[str]) -> None:
    say_lang = other if same == 0 else lang  # mostly the call's language, so later rules are reached
    expected = oracle(text, lang, say_lang, allowed)
    event(f"reason={expected}")
    got_reason = fallback_reason(text, lang, say_lang, slot="card", allowed=allowed,
                                 catalog_has=CATALOG.has)
    got = check(text, lang, say_lang, slot="card", allowed=allowed, catalog_has=CATALOG.has)
    assert got_reason == expected
    assert (got is None) == (expected is not None)
    if got is not None:
        assert text is not None and got == heal(text)


@settings(max_examples=600)
@given(text=_text, lang=st.sampled_from(LANGS), allowed=_allowed)
def test_render_fills_only_spoken_names_and_no_digit(text: str | None, lang: str,
                                                           allowed: list[str]) -> None:
    got = check(text, lang, lang, slot="card", allowed=allowed, catalog_has=CATALOG.has)
    if got is None:
        return
    assert text is not None and got == heal(text)
    pieces = _PH.split(got)  # literal, value, literal, value, ..., literal
    spoken = SPOKEN_BY_LANG[lang]
    expected = "".join(p if i % 2 == 0 else spoken[p] for i, p in enumerate(pieces))
    out = render(got, "card", lang, CATALOG)
    assert out == expected
    assert render(text, "card", lang, CATALOG) == expected  # render heals the raw text itself
    assert not _numeric(out)
    assert _PH.search(out) is None


SPOKEN_BY_LANG = {lang: {v: SPOKEN[v][LANGS.index(lang)] for v in CATALOGUED[lang]} for lang in LANGS}


@given(lang=st.sampled_from(LANGS), allowed=st.lists(st.sampled_from(["red", "purple"]), min_size=1,
                                                      unique=True),
       safe=st.lists(st.sampled_from(_SAFE), max_size=8))
def test_well_formed_texts_are_accepted(lang: str, allowed: list[str], safe: list[str]) -> None:
    """The generator reaches the accepting side: safe words and allowed placeholders always pass."""
    text = " ".join(["do", *safe, *(f"{{option:{v}}}" for v in allowed)])
    assert check(text, lang, lang, slot="card", allowed=allowed, catalog_has=CATALOG.has) == heal(text)


@given(lang=st.sampled_from(LANGS), words=st.lists(st.sampled_from(sorted(ORACLE_WORDS[EN]
                                                                         ^ ORACLE_WORDS[ES])),
                                                    min_size=1, max_size=3))
def test_number_words_count_only_in_their_language(lang: str, words: list[str]) -> None:
    """18.4: a word that is a number word in one language only fails exactly in that language."""
    text = "do you mean " + " ".join(words)
    want = "number" if any(w in ORACLE_WORDS[lang] for w in words) else None
    assert fallback_reason(text, lang, lang, slot="card", allowed=[], catalog_has=CATALOG.has) == want


@given(lang=st.sampled_from(LANGS), article=st.sampled_from(["un", "una", "Un", "UNA"]),
       safe=st.lists(st.sampled_from(_SAFE), max_size=6))
def test_spanish_articles_pass(lang: str, article: str, safe: list[str]) -> None:
    text = " ".join(["¿quiere", article, *safe, "{option:red}?"])
    assert check(text, ES, ES, slot="card", allowed=["red"], catalog_has=CATALOG.has) == heal(text)


# ------------------------------------------------------------------ Healing and TwiML


def _forbidden(c: str) -> bool:
    cp = ord(c)
    return (unicodedata.category(c) in ("Cc", "Cf", "Cs")
            or 0xFDD0 <= cp <= 0xFDEF or (cp & 0xFFFE) == 0xFFFE)


@settings(max_examples=800)
@given(text=st.one_of(st.text(_any_char), st.lists(_segment, max_size=20).map("".join)))
def test_heal_output_is_clean_nfc_and_idempotent(text: str) -> None:
    out = heal(text)
    assert not any(_forbidden(c) for c in out)
    assert unicodedata.is_normalized("NFC", out)
    assert heal(out) == out
    assert out == out.strip() and "  " not in out
    assert all(c == " " or not c.isspace() for c in out)
    # Nothing but what healing removes is lost: every other character survives, in order, up to NFC.
    kept = "".join(" " if c in "\t\n\r" else c for c in text if c in "\t\n\r" or not _forbidden(c))
    assert out == " ".join(unicodedata.normalize("NFC", kept).split())


def _parse_say(text: str) -> ET.Element:
    root = ET.fromstring(twiml.render(twiml.Say(text, voice="Polly.Lupe", language=ES)))
    assert root.tag == "Response" and len(root) == 1
    assert root[0].tag == "Say" and len(root[0]) == 0 and not root[0].tail
    return root[0]


@settings(max_examples=800)
@given(text=_text, lang=st.sampled_from(LANGS), allowed=_allowed)
def test_checked_text_renders_parseable_twiml_without_new_elements(
        text: str | None, lang: str, allowed: list[str]) -> None:
    got = check(text, lang, lang, slot="card", allowed=allowed, catalog_has=CATALOG.has)
    if got is None:
        return
    event("accepted")
    if any(m in got for m in ("<", ">", "&")):
        event("accepted markup-like text")
    spoken = render(got, "card", lang, CATALOG)
    assert (_parse_say(spoken).text or "") == spoken


@settings(max_examples=600)
@given(parts=st.lists(st.one_of(st.sampled_from(_MARKUP + _SAFE + _HEALED), st.text(_any_char,
                                                                                    max_size=6)),
                      max_size=25))
def test_any_healed_text_is_escaped_never_markup(parts: list[str]) -> None:
    """Stronger than the check's side: every healed text, accepted or not, renders as one plain
    ``<Say>``."""
    healed = heal("".join(parts))
    assert (_parse_say(healed).text or "") == healed
