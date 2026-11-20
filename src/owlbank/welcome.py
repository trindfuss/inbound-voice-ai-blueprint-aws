"""The web-activity offer as the call's welcome.

While the caller answers the last verification factor, the auth service reads the customer's
``WebActivity`` traits and matches the product they last looked at against the bank's list of
offers (policy ``web_offers``). No model is involved: :func:`prepare` is pure and gives only the
**offer ID** and a reason; the trait value never leaves the auth service. After verification the
front door renders the welcome with :func:`render_offer` from the catalog (``greeting.offer`` with
the phrase ``offer.<id>`` in the call's language). Anything other than a usable ``matched`` result
gives today's welcome (``greeting.personal``).

- :func:`normalise` and :func:`match_offer`: Unicode NFKC, ``casefold()``, trimmed, internal
  whitespace collapsed to one space; exact equality with a listed value after the same
  normalisation. Never a substring or fuzzy match.
- :func:`check_welcome` is structural only: it never matches the welcome's words.

:class:`WebOffer` is the shape of one list entry; :mod:`owlbank.policy` parses ``web_offers`` into
it. Nothing here logs a trait value or a welcome text.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol

from owlbank.catalog import PLACEHOLDER, render
from owlbank.web_activity import GROUP, MAX_VALUE_CHARS

__all__ = [
    "FORBIDDEN_CHARS",
    "GREETING_KEY",
    "MATCHED",
    "MAX_WELCOME_CHARS",
    "MEMORY_ERROR",
    "MEMORY_TIMEOUT",
    "NO_FIRST_NAME",
    "NO_MATCH",
    "NO_PROFILE",
    "NO_WEB_ACTIVITY",
    "OFFER_ID_RE",
    "OfferCatalog",
    "OFFER_KEY_PREFIX",
    "PRODUCT_TRAIT",
    "REASONS",
    "WebOffer",
    "WelcomeResult",
    "check_welcome",
    "match_offer",
    "normalise",
    "offer_key",
    "prepare",
    "render_offer",
]

#: The longest welcome that is spoken.
MAX_WELCOME_CHARS: Final = 240
#: The trait matched against the list; without it there is nothing to offer.
PRODUCT_TRAIT: Final = "lastViewedProduct"
#: An offer ID (it becomes part of a catalog key, ``offer.<id>``).
OFFER_ID_RE: Final = re.compile(r"[a-z0-9_]{1,32}")
#: The catalog template of the offer welcome; ``{firstName}`` and ``{offer}`` are filled.
GREETING_KEY: Final = "greeting.offer"
OFFER_KEY_PREFIX: Final = "offer."
#: Placeholders, tags and markdown never belong in speech.
FORBIDDEN_CHARS: Final = frozenset("{}<>[]*#`\\|~^_")
#: Line breaks, controls and invisible format characters (one line, nothing hidden).
_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})

#:  (``welcome_built``). :func:`prepare` gives the first four; the auth service adds the
#: Memory ones.
MATCHED, NO_WEB_ACTIVITY, NO_MATCH, NO_FIRST_NAME = "matched", "no_web_activity", "no_match", "no_first_name"
MEMORY_ERROR, MEMORY_TIMEOUT, NO_PROFILE = "memory_error", "memory_timeout", "no_profile"
REASONS: Final = frozenset({MATCHED, NO_WEB_ACTIVITY, NO_MATCH, NO_FIRST_NAME, MEMORY_ERROR, MEMORY_TIMEOUT,
                            NO_PROFILE})


@dataclass(frozen=True, slots=True)
class WebOffer:
    """One entry of the bank's list (policy ``web_offers``): an ID (:data:`OFFER_ID_RE`) and
    the trait values it answers to. Validation is :mod:`owlbank.policy`'s job."""

    id: str
    values: tuple[str, ...]

    def to_doc(self) -> dict[str, Any]:
        return {"id": self.id, "values": list(self.values)}


@dataclass(frozen=True, slots=True)
class WelcomeResult:
    """``offer_id`` is set exactly when ``reason`` is :data:`MATCHED`."""

    offer_id: str | None
    reason: str


def offer_key(offer_id: str) -> str:
    """The catalog key of an offer's phrase."""
    return f"{OFFER_KEY_PREFIX}{offer_id}"


def normalise(value: str) -> str:
    """NFKC, ``casefold()``, NFKC again (casefolding can leave a non-NFKC string, so the result
    is a fixed point), trimmed, with every run of whitespace collapsed to one space."""
    folded = unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", value).casefold())
    return " ".join(folded.split())


def match_offer(value: Any, offers: Iterable[WebOffer]) -> str | None:
    """The ID of the first offer that lists ``value`` (exact equality after :func:`normalise`), or
    ``None``. A blank value never matches; entries that aren't well-formed are skipped. Never a
    substring match: a listed value plus any other text doesn't match."""
    if not isinstance(value, str):
        return None
    wanted = normalise(value)
    if not wanted:
        return None
    for offer in offers:
        offer_id, values = getattr(offer, "id", None), getattr(offer, "values", None)
        if (not isinstance(offer_id, str) or not OFFER_ID_RE.fullmatch(offer_id)
                or not isinstance(values, Iterable) or isinstance(values, str | bytes)):
            continue
        if any(isinstance(v, str) and normalise(v) == wanted for v in values):
            return offer_id
    return None


def _product(traits: Any) -> str | None:
    """The ``lastViewedProduct`` string of the (bounded) ``WebActivity`` group, or ``None``."""
    group = traits.get(GROUP) if isinstance(traits, Mapping) else None
    value = group.get(PRODUCT_TRAIT) if isinstance(group, Mapping) else None
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_VALUE_CHARS:
        return None
    return value


def prepare(traits: Any, offers: Iterable[WebOffer], first_name: Any) -> WelcomeResult:
    """The offer to make, or why there is none. Pure.

    ``traits`` is :func:`owlbank.web_activity.bound_profile_traits` output; only
    ``WebActivity.lastViewedProduct`` is used. Reasons, in this order: ``no_web_activity`` (no
    product), ``no_match`` (not on the list), ``no_first_name`` (the welcome would read
    "Thanks," with no name), else ``matched`` with the offer ID."""
    product = _product(traits)
    if product is None:
        return WelcomeResult(None, NO_WEB_ACTIVITY)
    offer_id = match_offer(product, offers)
    if offer_id is None:
        return WelcomeResult(None, NO_MATCH)
    if not isinstance(first_name, str) or not first_name.strip():
        return WelcomeResult(None, NO_FIRST_NAME)
    return WelcomeResult(offer_id, MATCHED)


class OfferCatalog(Protocol):
    """What :func:`render_offer` needs from :class:`owlbank.catalog.PromptCatalog`."""

    def has(self, key: str, lang: str) -> bool: ...
    def template(self, key: str, lang: str) -> str: ...


def render_offer(catalog: OfferCatalog, language: str, first_name: Any, offer_id: Any) -> str | None:
    """``greeting.offer`` in ``language`` with ``{firstName}`` and ``{offer}`` (the phrase
    ``offer.<id>`` in ``language``). ``None`` when either entry is missing in exactly that language
    (no fallback to another: a welcome in the wrong language is worse than today's), the first name
    is blank, the ID is malformed, or the result fails :func:`check_welcome` against its lead (the
    template's text before ``{offer}``, filled). Never raises."""
    try:
        if (not isinstance(offer_id, str) or not OFFER_ID_RE.fullmatch(offer_id)
                or not isinstance(first_name, str) or not first_name.strip()):
            return None
        key = offer_key(offer_id)
        if not (catalog.has(GREETING_KEY, language) and catalog.has(key, language)):
            return None
        template, phrase = catalog.template(GREETING_KEY, language), catalog.template(key, language)
        if not isinstance(template, str) or not isinstance(phrase, str) or PLACEHOLDER.search(phrase):
            return None
        head, sep, _ = template.partition("{offer}")
        if not sep:
            return None
        values = {"firstName": first_name.strip(), "offer": phrase.strip()}
        text = render(template, values)
        lead = render(head, values).strip()
        return text if check_welcome(text, lead) else None
    except Exception:  # never raises: the fallback welcome always works
        return None


def check_welcome(text: Any, lead: Any) -> bool:
    """One line of at most :data:`MAX_WELCOME_CHARS` characters that starts with ``lead`` and
    goes on to a question: exactly one ``?``, at its end (at most one ``¿``), and no
    :data:`FORBIDDEN_CHARS`, control or format characters. ``lead`` must be non-empty. Never raises;
    anything that isn't a ``str`` fails."""
    try:
        if not isinstance(text, str) or not isinstance(lead, str) or not lead.strip():
            return False
        if len(text) > MAX_WELCOME_CHARS or not text.startswith(lead) or not text.endswith("?"):
            return False
        if text.count("?") != 1 or text.count("¿") > 1:
            return False
        if any(c in FORBIDDEN_CHARS or unicodedata.category(c) in _FORBIDDEN_CATEGORIES for c in text):
            return False
        return any(c.isalpha() for c in text[len(lead):])  # the question has words, not only "?"
    except Exception:  # never raises: a check that can't run fails
        return False
