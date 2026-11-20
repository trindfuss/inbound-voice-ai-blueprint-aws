"""Structural check for model-generated wording.

The model refers to options only through placeholders (``{option:red}``); :func:`render` fills them
from the catalog's spoken names. :func:`check` accepts a text only when every rule below holds, and
otherwise the caller speaks the catalog template. The check is structural: no phrase lists.

The text is first healed (:func:`heal`), so minor problems don't discard it: characters XML
can't carry and invisible format characters are removed, whitespace runs become one space, and the
text is NFC. :func:`check` and :func:`fallback_reason` judge the healed text, :func:`check` returns
it, and :func:`render` heals before filling (healing is idempotent). The rendered text stays plain
text: :class:`owlbank.twiml.Say` XML-escapes it, so markup the model writes (``<break/>``,
``<speak>``) is spoken as text and never becomes an element. Escaping here as well would make
Twilio read "&lt;" aloud.

Rules, checked in this order on the healed text (the first one broken is the fallback reason):

* ``missing``: the text is absent or blank;
* ``lang``: ``say_lang`` is not exactly the call's language;
* ``placeholder``: a ``{`` or ``}`` that isn't part of an ``{option:<v>}`` placeholder, or a
  placeholder whose ``v`` is not in ``allowed`` or has no ``option.<slot>.<v>`` entry in ``lang``;
* ``number``: with the placeholders removed, any numeric character (any script, including after
  compatibility decomposition) or any word token in :data:`NUMBER_WORDS` of the call's language
  only ("the red one" falls back to the template). Spanish's articles
  "un" and "una" aren't number words; "uno", ordinals ("segunda", English "second") are;
* ``length``: more than :data:`MAX_LEN` characters.

One more reason, ``twiml_invalid``, is not a rule of this module: the front door logs it when the
rendered response doesn't parse as XML, a rendering defect rather than model content.

Pure: no I/O and no logging. The caller logs ``generative_fallback`` with the reason, never the text.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from typing import Literal

from owlbank.catalog import PromptCatalog
from owlbank.slots.normalize import NUMBER_WORDS as _DIGIT_WORDS
from owlbank.slots.normalize import lang_family

PLACEHOLDER = re.compile(r"\{option:([a-z][a-z0-9_]{0,31})\}")

#: Longest generated text accepted, in characters, measured on the healed text before
#: rendering.
MAX_LEN = 200

#: Why generated text wasn't spoken. ``check`` returns the first five; the front door adds
#: ``twiml_invalid`` (a rendering defect) and ``budget`` (the resolver timed out in
#: ``generative`` mode).
Reason = Literal["missing", "lang", "placeholder", "number", "length", "twiml_invalid", "budget"]

#: Spanish ordinals, accent-folded (the check folds tokens before the lookup). ``normalize`` only
#: reads "primero"/"primer" as a number; a generated question may name none of them.
_ES_ORDINALS = frozenset({
    "primero", "primera", "primer", "segundo", "segunda", "tercero", "tercera", "tercer", "cuarto",
    "cuarta", "quinto", "quinta", "sexto", "sexta", "septimo", "septima", "octavo", "octava",
    "noveno", "novena", "decimo", "decima",
})

#: The words that make a generated text fall back as ``number``, per language family, checked only
#: for the call's language: the digit-reading words with their ordinals, minus Spanish's
#: articles "un" and "una". Lowercase and accent-folded.
NUMBER_WORDS: dict[str, frozenset[str]] = {
    "en": _DIGIT_WORDS["en"],
    "es": (_DIGIT_WORDS["es"] - {"un", "una"}) | _ES_ORDINALS,
}

_WORD_RE = re.compile(r"[^\W_]+")
_KEEP_AS_SPACE = frozenset("\t\n\r")


def _removed(c: str) -> bool:
    """Healing removes it: Cc, Cf, a lone surrogate (Cs) or a non-character (U+FDD0-U+FDEF and the
    last two code points of every plane)."""
    cp = ord(c)
    return (unicodedata.category(c) in ("Cc", "Cf", "Cs")
            or 0xFDD0 <= cp <= 0xFDEF or (cp & 0xFFFE) == 0xFFFE)


def heal(text: str) -> str:
    """Generated text made safe to judge and speak.

    Removes Cc and Cf characters, lone surrogates and non-characters, except tab, line feed and
    carriage return, which become spaces; then NFC-normalizes and collapses each whitespace run to
    one space, trimming the ends. Idempotent; the output is valid XML 1.0 character data.
    """
    kept = "".join(" " if c in _KEEP_AS_SPACE else c for c in text if c in _KEEP_AS_SPACE
                   or not _removed(c))
    return " ".join(unicodedata.normalize("NFC", kept).split())


def option_key(slot: str, value: str) -> str:
    """The catalog key holding an option's spoken name."""
    return f"option.{slot}.{value}"


def _fold(text: str) -> str:
    """NFKD, combining marks removed, lowercased: "Úno" -> "uno", "①" -> "1"."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    ).lower()


def _has_number(text: str, lang: str) -> bool:
    if any(c.isnumeric() for c in text):
        return True
    folded = _fold(text)
    if any(c.isnumeric() for c in folded):
        return True
    words = NUMBER_WORDS[lang_family(lang)]
    return any(tok in words for tok in _WORD_RE.findall(folded))


def fallback_reason(text: str | None, lang: str, say_lang: str | None, *, slot: str,
                    allowed: Sequence[str], catalog_has: Callable[[str, str], bool]) -> Reason | None:
    """Why ``text`` must fall back to the template, or ``None`` when its healed form passes every
    rule. Never ``twiml_invalid`` (the front door's, after rendering)."""
    if text is not None:
        text = heal(text)
    if not text:
        return "missing"
    if say_lang != lang:
        return "lang"
    allowed_set = frozenset(allowed)
    for m in PLACEHOLDER.finditer(text):
        value = m.group(1)
        if value not in allowed_set or not catalog_has(option_key(slot, value), lang):
            return "placeholder"
    rest = PLACEHOLDER.sub(" ", text)
    if "{" in rest or "}" in rest:
        return "placeholder"
    if _has_number(rest, lang):
        return "number"
    if len(text) > MAX_LEN:
        return "length"
    return None


def check(text: str | None, lang: str, say_lang: str | None, *, slot: str, allowed: Sequence[str],
          catalog_has: Callable[[str, str], bool]) -> str | None:
    """``heal(text)`` if it passes :func:`fallback_reason`, else ``None`` (fall back).

    ``allowed``: a confirmation's option; a disambiguation's candidates; a not-available question's
    named option and the caller's options. ``catalog_has(key, lang)`` is ``PromptCatalog.has``.
    """
    if fallback_reason(text, lang, say_lang, slot=slot, allowed=allowed,
                       catalog_has=catalog_has) is not None or text is None:
        return None
    return heal(text)


def render(text: str, slot: str, lang: str, catalog: PromptCatalog) -> str:
    """``heal(text)`` with each ``{option:<v>}`` filled with the catalog's spoken name for
    ``option.<slot>.<v>``.

    One pass: a spoken name is never re-scanned. Call only on a text :func:`check` accepted. The
    result is plain text, not markup: the TwiML ``Say`` escapes it.
    """
    return PLACEHOLDER.sub(lambda m: catalog.text(option_key(slot, m.group(1)), lang), heal(text))
