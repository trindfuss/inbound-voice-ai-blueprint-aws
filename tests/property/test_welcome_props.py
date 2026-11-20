"""Segment-traits properties of owlbank.welcome.

- ``match_offer`` is invariant under case, surrounding and repeated whitespace and
  NFKC-equivalent forms of a listed value; a value that differs from every listed value after
  normalisation never matches (in particular no listed value plus extra text). Over generated lists.
- At module level: ``prepare`` then ``render_offer`` gives the filled ``greeting.offer`` of the
  one matching entry, or ``None`` (the caller then speaks ``greeting.personal``); never anything else.
- ``check_welcome`` never raises, accepts only its structure, and rejects a valid welcome once it gets
  markup or a brace, a second question, a line break, or loses its lead.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from owlbank import welcome as wl
from owlbank.web_activity import GROUP, MAX_VALUE_CHARS

PROPS = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
CONTROL = {"Cc", "Cf", "Cs", "Zl", "Zp"}
LANGS = ("en-US", "es-US")
TEMPLATES = {"en-US": "Thanks, {firstName}. Would you like help with {offer}?",
             "es-US": "Gracias, {firstName}. ¿Le puedo ayudar con {offer}?"}

any_text = st.text(alphabet=st.characters(exclude_categories=()), max_size=300)
names = st.text(alphabet=st.characters(categories=("Lu", "Ll")), min_size=1, max_size=12)
leads = names.map(lambda n: f"Thanks, {n}.") | st.sampled_from(["Gracias, José.", "Thanks, Ana-Marie."])
safe_words = st.text(alphabet=st.characters(categories=("Lu", "Ll", "Nd"), include_characters=" ,.'$"),
                     min_size=1, max_size=120).filter(lambda s: any(c.isalpha() for c in s))
offer_ids = st.from_regex(r"[a-z0-9_]{1,32}", fullmatch=True)
#: Latin-1 letters and digits, where ``upper``/``lower`` keep the casefolded form (unlike, say, the
#: Turkish dotless i, a different letter); plus the separators banks put in product names.
latin = st.characters(codec="latin-1", categories=("Lu", "Ll", "Nd"))
case_values = st.text(alphabet=latin | st.sampled_from(" -/"), min_size=1, max_size=30).filter(
    lambda s: wl.normalise(s) != "")
#: Any listed value (any script), for the NFKC and whitespace properties.
listed_values = (case_values | st.text(min_size=1, max_size=30)).filter(lambda s: wl.normalise(s) != "")
#: Whitespace ``str.split`` collapses, some of which NFKC maps to a space first.
SPACES = [" ", "\t", "\n", " ", " ", "　", "\r\n"]


@st.composite
def offer_lists(draw: st.DrawFn, values: st.SearchStrategy[str] = listed_values,
                min_size: int = 1) -> tuple[wl.WebOffer, ...]:
    """A valid list: unique IDs, values unique across all offers after normalisation."""
    ids = draw(st.lists(offer_ids, min_size=min_size, max_size=6, unique=True))
    seen: set[str] = set()
    out = []
    for offer_id in ids:
        vals = [v for v in draw(st.lists(values, min_size=1, max_size=4)) if wl.normalise(v) not in seen]
        seen.update(wl.normalise(v) for v in vals)
        if vals:
            out.append(wl.WebOffer(offer_id, tuple(vals)))
    assume(len(out) >= min_size)
    return tuple(out)


def _listed(offers: tuple[wl.WebOffer, ...]) -> dict[str, str]:
    """Normalised listed value -> offer ID (the list is valid, so each value has one offer)."""
    return {wl.normalise(v): o.id for o in offers for v in o.values}


@st.composite
def case_variant(draw: st.DrawFn, value: str) -> str:
    return "".join(c.upper() if draw(st.booleans()) else c.lower() for c in value)


@st.composite
def nfkc_variant(draw: st.DrawFn, value: str) -> str:
    """An NFKC-equivalent spelling: a normal form, or ASCII letters as their full-width forms."""
    form = draw(st.sampled_from(["NFC", "NFD", "NFKC", "NFKD", "fullwidth"]))
    if form == "fullwidth":
        return "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" and draw(st.booleans()) else c for c in value)
    return unicodedata.normalize(form, value)  # type: ignore[arg-type]


@st.composite
def space_variant(draw: st.DrawFn, value: str) -> str:
    """Surrounding whitespace added and each internal whitespace run replaced by another run."""
    run = st.lists(st.sampled_from(SPACES), min_size=1, max_size=3).map("".join)
    inner = draw(run).join(value.split())
    pad = st.lists(st.sampled_from(SPACES), max_size=3).map("".join)
    return draw(pad) + inner + draw(pad)




@PROPS
@given(st.data(), offer_lists(case_values))
def test_case_variants_of_a_listed_value_match(data: st.DataObject, offers: tuple[wl.WebOffer, ...]) -> None:
    offer = data.draw(st.sampled_from(offers))
    value = data.draw(st.sampled_from(offer.values))
    assert wl.match_offer(data.draw(case_variant(value)), offers) == offer.id


@PROPS
@given(st.data(), offer_lists())
def test_nfkc_and_whitespace_variants_of_a_listed_value_match(data: st.DataObject,
                                                                  offers: tuple[wl.WebOffer, ...]) -> None:
    offer = data.draw(st.sampled_from(offers))
    value = data.draw(st.sampled_from(offer.values))
    variant = data.draw(space_variant(data.draw(nfkc_variant(value))))
    assert wl.match_offer(variant, offers) == offer.id


@PROPS
@given(st.data(), offer_lists(), st.text(min_size=1, max_size=40))
def test_a_listed_value_plus_extra_text_never_matches(data: st.DataObject, offers: tuple[wl.WebOffer, ...],
                                                          extra: str) -> None:
    value = data.draw(st.sampled_from([v for o in offers for v in o.values]))
    joined = data.draw(st.sampled_from([f"{value} {extra}", f"{value}{extra}", f"{extra} {value}",
                                        f"{value}. {extra}"]))
    assume(wl.normalise(joined) not in _listed(offers))  # e.g. extra text that makes another listed value
    assert wl.match_offer(joined, offers) is None


@PROPS
@given(offer_lists(min_size=0), listed_values | any_text)
def test_matches_exactly_the_normalised_listed_values(offers: tuple[wl.WebOffer, ...], value: str) -> None:
    listed = _listed(offers)
    assert wl.match_offer(value, offers) == listed.get(wl.normalise(value))


@PROPS
@given(any_text)
def test_normalise_is_idempotent_and_tidy(value: str) -> None:
    once = wl.normalise(value)
    assert wl.normalise(once) == once
    assert once == once.strip() and "  " not in once and not any(c.isspace() and c != " " for c in once)


# --------------------------------------------------------------------------- (module level)


class StubCatalog:
    def __init__(self, entries: dict[tuple[str, str], str]) -> None:
        self.entries = entries

    def has(self, key: str, lang: str) -> bool:
        return (key, lang) in self.entries

    def template(self, key: str, lang: str) -> str:
        return self.entries[(key, lang)]


phrases = safe_words | st.sampled_from(["our <b>HELOC</b>", "a loan? or two", "x" * 240, "our {product}",
                                        "line\nbreak", ""])
first_names = names | st.sampled_from(["", "  ", "{offer}", "Ana<b>", "José", " Thomas "]) | st.none()


@st.composite
def catalogs(draw: st.DrawFn, offers: tuple[wl.WebOffer, ...]) -> StubCatalog:
    """Some languages lack ``greeting.offer``; some offers lack a phrase in some languages."""
    entries: dict[tuple[str, str], str] = {}
    for lang in LANGS:
        if draw(st.integers(0, 9)):
            entries[("greeting.offer", lang)] = TEMPLATES[lang]
        for offer in offers:
            if draw(st.integers(0, 3)):
                entries[(wl.offer_key(offer.id), lang)] = draw(phrases)
    return StubCatalog(entries)


def _fill(template: str, first_name: str, phrase: str) -> str:
    """The expected welcome, filled independently of the module (one pass over the template)."""
    return re.sub(r"\{(firstName|offer)\}", lambda m: first_name if m[1] == "firstName" else phrase, template)


@PROPS
@given(st.data(), offer_lists(), first_names, st.sampled_from(LANGS))
def test_the_welcome_is_the_one_matching_entrys_template_or_none(
        data: st.DataObject, offers: tuple[wl.WebOffer, ...], first_name: Any, lang: str) -> None:
    listed = [v for o in offers for v in o.values]
    product = data.draw(st.sampled_from(listed) | st.sampled_from(listed).map(lambda v: f"  {v.upper()} ")
                        | any_text | st.none())
    catalog = data.draw(catalogs(offers))
    result = wl.prepare({GROUP: {"lastViewedProduct": product}}, offers, first_name)
    assert result.reason in wl.REASONS and (result.offer_id is None) == (result.reason != wl.MATCHED)
    text = wl.render_offer(catalog, lang, first_name, result.offer_id) if result.reason == wl.MATCHED else None

    matching = [o for o in offers
                if isinstance(product, str) and len(product) <= MAX_VALUE_CHARS and wl.normalise(product)
                and wl.normalise(product) in {wl.normalise(v) for v in o.values}]
    assert len(matching) <= 1  # a valid list: at most one entry lists any value
    name_ok = isinstance(first_name, str) and bool(first_name.strip())
    if not matching or not name_ok:
        assert result.offer_id is None and text is None
        return
    (entry,) = matching
    assert result == wl.WelcomeResult(entry.id, wl.MATCHED)
    key = wl.offer_key(entry.id)
    if not (catalog.has("greeting.offer", lang) and catalog.has(key, lang)):
        assert text is None
        return
    template, phrase = catalog.template("greeting.offer", lang), catalog.template(key, lang)
    expected = _fill(template, first_name.strip(), phrase.strip())
    lead = _fill(template.partition("{offer}")[0], first_name.strip(), phrase.strip()).strip()
    if "{" in phrase or not wl.check_welcome(expected, lead):
        assert text is None
    else:
        assert text == expected


# --------------------------------------------------------------------------- check_welcome


@PROPS
@given(any_text, any_text)
def test_check_never_raises(text: str, lead: str) -> None:
    assert wl.check_welcome(text, lead) in (True, False)


@PROPS
@given(any_text, any_text)
def test_check_accepts_only_its_structure(text: str, lead: str) -> None:
    if wl.check_welcome(text, lead):
        assert lead.strip() and text.startswith(lead) and text.endswith("?") and text.count("?") == 1
        assert len(text) <= wl.MAX_WELCOME_CHARS
        assert not any(c in wl.FORBIDDEN_CHARS or unicodedata.category(c) in CONTROL for c in text)


@st.composite
def valid_welcomes(draw: st.DrawFn) -> tuple[str, str]:
    lead = draw(leads)
    text = f"{lead} {draw(safe_words)}?"
    assume(len(text) <= wl.MAX_WELCOME_CHARS)
    return text, lead


@PROPS
@given(valid_welcomes())
def test_check_accepts_generated_valid_welcomes(case: tuple[str, str]) -> None:
    assert wl.check_welcome(*case)


@PROPS
@given(valid_welcomes(), st.data())
def test_check_rejects_markup_braces_and_breaks(case: tuple[str, str], data: st.DataObject) -> None:
    text, lead = case
    bad = data.draw(st.sampled_from(sorted(wl.FORBIDDEN_CHARS) + ["\n", "\r", "\t", " ", "​", "\x00"]))
    at = data.draw(st.integers(len(lead), len(text) - 1))  # inside the question, before its "?"
    assert not wl.check_welcome(text[:at] + bad + text[at:], lead)


@PROPS
@given(valid_welcomes(), st.data())
def test_check_rejects_more_than_one_question(case: tuple[str, str], data: st.DataObject) -> None:
    text, lead = case
    at = data.draw(st.integers(len(lead), len(text) - 1))
    assert not wl.check_welcome(text[:at] + "?" + text[at:], lead)


@PROPS
@given(valid_welcomes(), st.data())
def test_check_rejects_a_missing_lead(case: tuple[str, str], data: st.DataObject) -> None:
    text, lead = case
    cut = data.draw(st.integers(1, len(lead)))  # drop the start of the lead
    assert not wl.check_welcome(text[cut:], lead)
    assert not wl.check_welcome(text, lead + "x")  # another lead doesn't match either
