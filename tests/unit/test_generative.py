"""Examples for the generative check."""

from __future__ import annotations

import typing
import xml.etree.ElementTree as ET

import pytest

from owlbank import twiml
from owlbank.catalog import PromptCatalog
from owlbank.slots import generative
from owlbank.slots.generative import (MAX_LEN, NUMBER_WORDS, PLACEHOLDER, Reason, check,
                                      fallback_reason, heal, render)

EN, ES = "en-US", "es-US"
NAMES = {
    EN: {"red": "the red card", "purple": "the purple card", "rewards": "the rewards card"},
    ES: {"red": "la tarjeta roja", "purple": "la tarjeta morada"},  # no Spanish "rewards"
}
CATALOG = PromptCatalog.from_dict({
    "defaultLanguage": EN,
    "languages": {EN: {}, ES: {}},
    "entries": [
        {"key": f"option.card.{v}", "lang": lang, "text": name}
        for lang, names in NAMES.items() for v, name in names.items()
    ],
})
ALL = ("red", "purple", "rewards")


def reason(text: str | None, lang: str = EN, say_lang: str | None = EN,
           allowed: tuple[str, ...] = ALL) -> str | None:
    return fallback_reason(text, lang, say_lang, slot="card", allowed=allowed,
                           catalog_has=CATALOG.has)


def accepts(text: str | None, lang: str = EN, say_lang: str | None = EN,
            allowed: tuple[str, ...] = ALL) -> bool:
    got = check(text, lang, say_lang, slot="card", allowed=allowed, catalog_has=CATALOG.has)
    assert (got is None) == (reason(text, lang, say_lang, allowed) is not None)
    if got is not None:
        assert text is not None and got == heal(text)
    return got is not None


# ---------------------------------------------------------------- each reason


@pytest.mark.parametrize("text", [None, "", "   ", "\n\t", "\u200b", "\x1b\r\n", "\ufeff \ud800"])
def test_missing(text: str | None) -> None:
    assert reason(text) == "missing"
    assert not accepts(text)


@pytest.mark.parametrize(("lang", "say_lang"), [(EN, ES), (ES, EN), (EN, None), (EN, "en"),
                                                (EN, "en-us"), (ES, "es")])
def test_lang_must_match_exactly(lang: str, say_lang: str | None) -> None:
    assert reason("Did you mean {option:red}?", lang, say_lang) == "lang"


@pytest.mark.parametrize("text", [
    "Did you mean {option:gold}?",  # not a known option
    "Did you mean {card:red}?",  # braces that aren't a placeholder
    "Did you mean {red}?",
    "Did you mean {option:Red}?",  # uppercase isn't a placeholder value
    "Did you mean {option: red}?",
    "Did you mean {option:red?",
    "Did you mean option:red}?",
    "Braces {} alone",
    "Nested {{option:red}}",
])
def test_placeholder(text: str) -> None:
    assert reason(text) == "placeholder"


def test_placeholder_for_a_non_candidate_falls_back() -> None:
    text = "Do you mean {option:red} or {option:rewards}?"
    assert reason(text, allowed=("red", "purple")) == "placeholder"
    assert accepts(text, allowed=("red", "rewards"))


def test_placeholder_missing_from_the_catalog_in_the_call_language() -> None:
    text = "¿Se refiere a {option:rewards}?"
    assert reason(text, ES, ES) == "placeholder"
    assert accepts("Do you mean {option:rewards}?")


@pytest.mark.parametrize(("text", "lang"), [
    ("Do you mean card 2?", EN),
    ("Your card ending ٣٤", EN),  # Arabic-Indic digits
    ("Option ① please", EN),  # circled digit, numeric after NFKD
    ("Half ½ of them", EN),
    ("Do you mean the first one?", EN),
    ("Do you mean {option:red}, the second option?", EN),
    ("ONE of them?", EN),
    ("¿Quiere la tarjeta número dos?", ES),
    ("¿Úno de ellos?", ES),  # accents don't hide a Spanish number word
    ("¿La tarjeta ٣?", ES),  # digits fail in every language
])
def test_number(text: str, lang: str) -> None:
    assert reason(text, lang, lang) == "number"


def test_the_red_one_falls_back_q7() -> None:
    assert reason("Do you mean the red one?") == "number"
    assert reason("Do you mean {option:red} one?") == "number"


def test_number_words_are_checked_only_in_the_call_language() -> None:
    assert accepts("Did you mean {option:red}, dos?")  # Spanish word in English: not a number
    assert accepts("¿Se refiere a {option:red}? Okay, two.", ES, ES)
    assert accepts("Do you want to pay once?")  # Spanish "once" (11) in English text
    assert reason("¿Paga once?", ES, ES) == "number"


def test_spanish_articles_un_and_una_pass() -> None:
    assert accepts("¿Se refiere a una Tarjeta OwlBank Morada?", ES, ES)
    assert accepts("¿Quiere usar un {option:red}?", ES, ES)
    assert accepts("¿Una o {option:purple}?", ES, ES)  # "o" is "or" in Spanish
    assert "un" not in NUMBER_WORDS["es"] and "una" not in NUMBER_WORDS["es"]


@pytest.mark.parametrize("text", [
    "¿Se refiere a uno de ellos?",
    "¿Se refiere a la segunda?",
    "¿La tercera tarjeta?",
    "¿El primero?",
    "¿La tarjeta 3?",
    "¿Dos tarjetas?",
    "¿Quiere la décima?",
])
def test_spanish_number_words_and_ordinals_still_fail(text: str) -> None:
    assert reason(text, ES, ES) == "number"


@pytest.mark.parametrize("text", ["Do you mean the second one?", "The third card?", "Card 3?",
                                  "Is it one of them?"])
def test_english_numbers_and_ordinals_still_fail(text: str) -> None:
    assert reason(text) == "number"


def test_digits_inside_placeholders_do_not_count() -> None:
    names = {"card2": "the travel card"}
    cat = PromptCatalog.from_dict({
        "defaultLanguage": EN, "languages": {EN: {}},
        "entries": [{"key": "option.card.card2", "lang": EN, "text": names["card2"]}],
    })
    got = check("Do you mean {option:card2}?", EN, EN, slot="card", allowed=["card2"],
                catalog_has=cat.has)
    assert got is not None
    assert render(got, "card", EN, cat) == "Do you mean the travel card?"


def test_length() -> None:
    base = "Do you mean {option:red} or {option:purple}"
    ok = base + "?" * (MAX_LEN - len(base))
    assert len(ok) == MAX_LEN
    assert accepts(ok)
    assert reason(ok + "?") == "length"


def test_reasons_follow_the_documented_order() -> None:
    long_bad = "{option:gold} 2 " + "x" * MAX_LEN
    assert reason(long_bad, EN, ES) == "lang"
    assert reason(long_bad) == "placeholder"
    assert reason("2 " + "x" * MAX_LEN) == "number"


def test_reason_values() -> None:
    assert set(typing.get_args(Reason)) == {"missing", "lang", "placeholder", "number", "length",
                                            "twiml_invalid", "budget"}


# ---------------------------------------------------------------- healing


@pytest.mark.parametrize(("raw", "healed"), [
    ("Do you mean\x1b {option:red}?", "Do you mean {option:red}?"),  # ESC
    ("Do you\u200b mean {option:red}?", "Do you mean {option:red}?"),  # zero-width space (Cf)
    ("Do you mean \ud800{option:red}?", "Do you mean {option:red}?"),  # lone surrogate
    ("Do you mean {option:red}?\ufffe", "Do you mean {option:red}?"),  # non-character
    ("A\uffffB\ufdd0C\U0010fffeD\U0001ffffE", "ABCDE"),  # more non-characters
    ("\ufeffWhich card?\u2066", "Which card?"),  # BOM and a bidi isolate (Cf)
    ("Do you\tmean\r\n{option:red}?", "Do you mean {option:red}?"),  # tab/CR/LF become spaces
    ("  Do   you \u2028 mean\u00a0 it?  ", "Do you mean it?"),  # whitespace runs collapse
    ("Cafe\u0301", "Café"),  # NFC
    ("Cafe\u200b\u0301", "Café"),  # a format character removed before NFC
    ("Line\x00one\x7fx\x85y", "Lineonexy"),  # other Cc removed, not spaced
    ("", ""),
])
def test_heal(raw: str, healed: str) -> None:
    assert heal(raw) == healed
    assert heal(healed) == healed


def test_check_and_render_use_the_healed_text() -> None:
    raw = "Do you\u200b mean\n{option:red}?\x1b"
    assert reason(raw) is None
    got = check(raw, EN, EN, slot="card", allowed=ALL, catalog_has=CATALOG.has)
    assert got == "Do you mean {option:red}?"
    assert render(raw, "card", EN, CATALOG) == "Do you mean the red card?"


def test_healing_can_reveal_a_number_word() -> None:
    assert reason("Do you mean t\u200bwo of them?") == "number"
    assert reason("Do you mean {opt\u200bion:gold}?") == "placeholder"


def test_length_is_measured_on_the_healed_text() -> None:
    base = "Do you mean {option:red}?"
    padded = base + " " * (MAX_LEN + 50) + "\u200b" * 10
    assert len(padded) > MAX_LEN
    assert accepts(padded)
    assert reason(base + "?" * (MAX_LEN - len(base) + 1)) == "length"


def _said(text: str) -> ET.Element:
    """The rendered Say element of a one-verb TwiML response, parsed as XML."""
    root = ET.fromstring(twiml.render(twiml.Say(text, voice="Polly.Joanna", language=EN)))
    assert root.tag == "Response" and len(root) == 1
    say = root[0]
    assert say.tag == "Say" and len(say) == 0  # no child element
    return say


@pytest.mark.parametrize("text", [
    "Do you mean {option:red}? <break/>",
    "<speak>Do you mean {option:red}?</speak>",
    '<break strength="strong"/> Do you mean {option:purple}?',
    "</Say><Hangup/> Do you mean {option:red}?",
    "Do you mean {option:red} &amp; <![CDATA[x]]> <!-- c --> <?pi?>",
])
def test_markup_from_the_model_never_becomes_an_element(text: str) -> None:
    got = check(text, EN, EN, slot="card", allowed=ALL, catalog_has=CATALOG.has)
    assert got is not None
    spoken = render(got, "card", EN, CATALOG)
    assert _said(spoken).text == spoken  # spoken as plain text, escaped once by the SDK


# ---------------------------------------------------------------- accepted and rendered


def test_valid_disambiguation_renders_both_spoken_names() -> None:
    text = "Do you mean {option:red} or {option:purple}?"
    got = check(text, EN, EN, slot="card", allowed=["red", "purple"], catalog_has=CATALOG.has)
    assert got == text
    assert render(got, "card", EN, CATALOG) == "Do you mean the red card or the purple card?"


def test_valid_spanish_confirmation_renders() -> None:
    text = "¿Quiere usar {option:red}?"
    got = check(text, ES, ES, slot="card", allowed=["red"], catalog_has=CATALOG.has)
    assert got is not None
    assert render(got, "card", ES, CATALOG) == "¿Quiere usar la tarjeta roja?"


def test_text_without_placeholders_is_accepted() -> None:
    assert accepts("Which card would you like to use?")


def test_render_does_not_rescan_spoken_names() -> None:
    cat = PromptCatalog.from_dict({
        "defaultLanguage": EN, "languages": {EN: {}},
        "entries": [{"key": "option.card.red", "lang": EN, "text": "the {option:red} card"}],
    })
    spoken = cat.text("option.card.red", EN)
    assert render("{option:red}", "card", EN, cat) == spoken  # one pass: not filled again
    assert PLACEHOLDER.pattern == r"\{option:([a-z][a-z0-9_]{0,31})\}"


def test_option_key() -> None:
    assert generative.option_key("card", "red") == "option.card.red"
