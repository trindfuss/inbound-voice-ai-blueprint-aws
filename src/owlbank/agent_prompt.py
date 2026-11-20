"""The agent's system prompt, rendered from the catalog and split into a static and a per-call
part for prompt caching.

One pure renderer for every place that builds the prompt: the bridge (``Bridge._system_prompt``),
the prompt suite and ``scripts/bench/agent_bench.py``, so the bench and the suite send what callers
get. The bridge keeps the data fetching (customer, Memory, tracing, the per-call cache); this module
only turns the catalog and those values into text. No I/O beyond the catalog reads.

The prompt is two parts, in this order:

* the **static part** (:func:`render_static`), the same for every caller of a (model key,
  language): the model's catalog text rendered with no values at all, the placeholder rule
  (:data:`PLACEHOLDER_SENTENCE`) as one more rule line, and the call-language line
  (:data:`CALL_LANGUAGE_LINE`). The client puts the system cache point after it.
* the **per-call part** (:func:`render_per_call`), which takes no model key: optionally one
  web-activity block (``agent.web_activity`` when the welcome made the offer,
  ``agent.web_activity_context`` when not), then the caller context: ``agent.caller_context``
  filled for a verified caller, or ``agent.caller_context_unverified`` as is (memory gating).
  The bridge's fallback prompt uses :data:`FALLBACK_CONTEXT` instead, which claims no verification
  either way.

The bridge's clock line for the call follows both (:meth:`RenderedPrompt.joined`); it is not
a part of :func:`prompt_parts`.

:func:`prompt_parts` is exactly what the renderer reads for a model and language, and
:func:`prompt_hash` is SHA-256 over it: a validation record is stale when any part changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import astuple, dataclass
from types import MappingProxyType
from typing import Final

from owlbank import untrusted
from owlbank.catalog import PLACEHOLDER, CatalogKeyError, PromptCatalog
from owlbank.catalog import render as fill
from owlbank.failover.settings import series_key
from owlbank.obs import emit

log = logging.getLogger(__name__)

#: The prefix of every model-instruction key.
AGENT_PREFIX: Final = "agent."
#: The prefix of each model's own catalog key, ``agent.system.<model key>``.
SYSTEM_KEY_PREFIX: Final = "agent.system"
#: The model key of a caller that knows no model list (a bare bridge config, the bench's default):
#: the default list's first model.
DEFAULT_MODEL_KEY: Final = "haiku-5-5"
WEB_ACTIVITY_KEY: Final = "agent.web_activity"
WEB_ACTIVITY_CONTEXT_KEY: Final = "agent.web_activity_context"
WEB_BLOCK_KEYS: Final = (WEB_ACTIVITY_KEY, WEB_ACTIVITY_CONTEXT_KEY)
#: The caller-context frames: the verified one has the caller placeholders, the unverified one
#: none, so nothing per caller can reach an unverified prompt (memory gating).
CALLER_CONTEXT_KEY: Final = "agent.caller_context"
CALLER_CONTEXT_UNVERIFIED_KEY: Final = "agent.caller_context_unverified"
CALLER_CONTEXT_KEYS: Final = (CALLER_CONTEXT_KEY, CALLER_CONTEXT_UNVERIFIED_KEY)
CALLER_PLACEHOLDERS: Final = frozenset({"firstName", "profile", "memory"})

#: The exact placeholder set of an ``agent.*`` text, by key. An ``agent.*`` key not listed
#: (every ``agent.system.*`` text, both web-activity blocks, the unverified frame, any new key)
#: allows none. Seed, preflight, the admin API and the console check against it.
ALLOWED_PLACEHOLDERS: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
    CALLER_CONTEXT_KEY: CALLER_PLACEHOLDERS,
})

#: One rule appended to the model's catalog text on both system-prompt
#: paths. The front door puts ``[<slot name>]`` on the answering turn in place of a sensitive slot's
#: words. A constant here, not a catalog key, so a bridge deploy needs no reseed. Fixed per language,
#: so the static part stays stable (the client's prompt cache point, reasoning signatures on
#: tool-result hops).
PLACEHOLDER_SENTENCE: Final[Mapping[str, str]] = MappingProxyType({
    "en-US": "A slot name in square brackets, such as [ssn4], stands for an answer the caller "
             "gave that is withheld from you: never ask the caller for it again or repeat it, "
             "and learn the result of verification from your tools.",
    "es-US": "Un nombre entre corchetes, como [ssn4], representa una respuesta que el cliente "
             "dio y que se le oculta a usted: nunca se la vuelva a pedir al cliente ni la repita, "
             "y conozca el resultado de la verificación por sus herramientas.",
})
DEFAULT_PLACEHOLDER_LANG: Final = "en-US"

#: The empty values of the verified caller context, per language; another
#: language gets the English one. An unknown first name, an empty profile or memory.
UNKNOWN_NAME: Final[Mapping[str, str]] = MappingProxyType({"en-US": "the caller", "es-US": "el cliente"})
EMPTY_VALUE: Final[Mapping[str, str]] = MappingProxyType({"en-US": "none", "es-US": "ninguna"})
#: The fallback prompt's whole per-call part, per language (another gets
#: the English one). The catalog frames' tags, no caller data, no web block, and no
#: verification claim either way: the verified frame said "Verified: yes" for unverified callers.
#: A constant, not a catalog key, so the fallback needs no catalog read for it.
FALLBACK_CONTEXT: Final[Mapping[str, str]] = MappingProxyType({
    "en-US": "<caller_context>\nVerified: unknown. The caller's details are unavailable right now: you "
             "don't know who the caller is or whether they are verified. Your tools check "
             "verification.\n</caller_context>",
    "es-US": "<caller_context>\nVerificado: desconocido. Los datos del cliente no están disponibles "
             "ahora: no sabe quién es ni si está verificado. Sus herramientas comprueban la "
             "verificación.\n</caller_context>",
})

#: The static part's last line; ``{lang}`` is the call's language (``str.format``).
CALL_LANGUAGE_LINE: Final = "Call language: {lang}."

#: A block exactly as :func:`owlbank.untrusted.wrap` writes it: its content has no ``<`` or ``>``.
_WRAPPED = re.compile(r'<untrusted_data source="[a-z][a-z0-9_]{0,31}">[^<>]*</untrusted_data>')

#: (event, key, language) already logged by this process: each degraded state logs once.
_LOGGED: set[tuple[str, str, str]] = set()


def system_key(model_key: str) -> str:
    """The catalog key of the model's own prompt text, ``agent.system.<model key>``, for a
    model key such as ``haiku-5-5`` (:func:`model_key`). :func:`render_static`, :func:`prompt_parts`,
    :func:`prompt_hash` and the bridge's fallback prompt all read the key from here."""
    return f"{SYSTEM_KEY_PREFIX}.{model_key}"


def model_key(model_id: str) -> str:
    """The prompt's model key for a model ID: its series key (``haiku-5-5`` for
    ``us.anthropic.claude-haiku-5-5`` and ``global.`` alike), or the ID itself for one that names
    no Claude series (no catalog entry, so the fallback applies). The key of the bridge's
    ``{model key: prompt}`` map, which the agent client reads per entry."""
    return series_key(model_id) or model_id


def _per_lang(table: Mapping[str, str], lang: str) -> str:
    return table.get(lang) or table[DEFAULT_PLACEHOLDER_LANG]


def placeholder_sentence(lang: str) -> str:
    """The placeholder rule for ``lang`` (another language gets the English sentence)."""
    return _per_lang(PLACEHOLDER_SENTENCE, lang)


def allowed_placeholders(key: str) -> frozenset[str] | None:
    """The exact set an ``agent.*`` text must have; None for a key outside ``agent.``."""
    if not key.startswith(AGENT_PREFIX):
        return None
    return ALLOWED_PLACEHOLDERS.get(key, frozenset())


def placeholder_problems(key: str, text: str) -> str | None:
    """Why ``text`` breaks for ``key`` (stray brace or wrong set), else None. Structural only:
    a ``{`` or ``}`` that isn't part of a ``{name}`` placeholder, or a placeholder set other than
    :func:`allowed_placeholders`. Always None for a key outside ``agent.``."""
    allowed = allowed_placeholders(key)
    if allowed is None:
        return None
    rest = PLACEHOLDER.sub("", text)
    if "{" in rest or "}" in rest:
        return "has a '{' or '}' that isn't a {placeholder}"
    if frozenset(PLACEHOLDER.findall(text)) != allowed:
        if not allowed:
            return "must have no placeholders"
        return "must have exactly " + ", ".join(f"{{{n}}}" for n in sorted(allowed))
    return None


@dataclass(frozen=True, slots=True)
class CallerFacts:
    """What the per-call part says about a verified caller (raw values; the renderer neutralizes
    names). ``profile`` is the server-built profile line, account nicknames already wrapped
    ``memory`` is the already wrapped recall or None."""

    first_name: str
    profile: str
    memory: str | None


@dataclass(frozen=True, slots=True)
class PromptParts:
    """What the prompt for one model and language is rendered from (extended by agent prompt
    caching). The templates are the served ones (after the catalog's language fallback); a
    missing block or frame is ``None``. Then the per-language empty values and the fallback
    per-call part."""

    model_text: str
    web_activity: str | None
    web_activity_context: str | None
    placeholder_rule: str
    language_line: str
    caller_context: str | None
    caller_context_unverified: str | None
    unknown_name: str
    empty_value: str
    fallback_context: str

    def digest(self) -> str:
        """SHA-256 (hex) over the parts, in field order, as a compact ASCII JSON array (``null``
        for a missing block), so distinct parts never hash alike."""
        blob = json.dumps(list(astuple(self)), ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """The two parts of one model's system prompt: ``static`` goes before the system cache
    point, ``per_call`` after it."""

    static: str
    per_call: str

    def joined(self, clock: str | None = None) -> str:
        """What the model reads: the static part, the per-call part and the clock line, one
        per line; the same text as ``join_system(static, per_call + "\\n" + clock)``."""
        suffix = self.per_call if clock is None else f"{self.per_call}\n{clock}"
        return f"{self.static}\n{suffix}" if suffix else self.static


def _block(catalog: PromptCatalog, key: str, lang: str) -> str | None:
    try:
        return catalog.template(key, lang)
    except CatalogKeyError:
        return None


def prompt_parts(catalog: PromptCatalog, model_key: str, lang: str) -> PromptParts:
    """The parts the renderer reads for ``model_key`` in ``lang``. ``CatalogKeyError`` when the
    model's own text is missing."""
    return PromptParts(
        model_text=catalog.template(system_key(model_key), lang),
        web_activity=_block(catalog, WEB_ACTIVITY_KEY, lang),
        web_activity_context=_block(catalog, WEB_ACTIVITY_CONTEXT_KEY, lang),
        placeholder_rule=placeholder_sentence(lang),
        language_line=CALL_LANGUAGE_LINE,
        caller_context=_block(catalog, CALLER_CONTEXT_KEY, lang),
        caller_context_unverified=_block(catalog, CALLER_CONTEXT_UNVERIFIED_KEY, lang),
        unknown_name=_per_lang(UNKNOWN_NAME, lang),
        empty_value=_per_lang(EMPTY_VALUE, lang),
        fallback_context=_per_lang(FALLBACK_CONTEXT, lang),
    )


def prompt_hash(catalog: PromptCatalog, model_key: str, lang: str) -> str:
    """SHA-256 over :func:`prompt_parts`: changes exactly when one of its parts changes."""
    return prompt_parts(catalog, model_key, lang).digest()


def _once(level: int, event: str, key: str, lang: str, **fields: object) -> None:
    if (event, key, lang) in _LOGGED:
        return
    _LOGGED.add((event, key, lang))
    emit(log, level, event, key=key, language=lang, **fields)


def render_static(catalog: PromptCatalog, model_key: str, lang: str) -> str:
    """The static part for ``model_key`` in ``lang``: the model's text rendered with no values,
    ``"\\n- "`` and the placeholder rule, then the call-language line. A placeholder left in the
    model's text is removed, never filled, and logged once per process per (key, language) as
    ``agent_prompt_placeholder`` (ERROR). A missing model text raises ``CatalogKeyError`` (the
    bridge then uses its fallback or its fallback prompt)."""
    key = system_key(model_key)
    text = catalog.template(key, lang)
    names = sorted(set(PLACEHOLDER.findall(text)))
    if names:
        _once(logging.ERROR, "agent_prompt_placeholder", key, lang, placeholders=names,
              decision="removed")
    return f"{fill(text, {})}\n- {placeholder_sentence(lang)}\n{CALL_LANGUAGE_LINE.format(lang=lang)}"


def _neutralize_outside_blocks(text: str) -> str:
    """``text`` with every ``<``, ``>`` and ``&`` escaped except inside ``untrusted_data``
    blocks as :func:`owlbank.untrusted.wrap` writes them, whose content is escaped already. So the
    profile's name fields can't close the frame, and its wrapped account nicknames stay wrapped."""
    out: list[str] = []
    pos = 0
    for m in _WRAPPED.finditer(text):
        out.append(untrusted.neutralize(text[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(untrusted.neutralize(text[pos:]))
    return "".join(out)


def _or_empty(value: str | None, empty: str) -> str:
    return value if value is not None and value.strip() else empty


def render_per_call(catalog: PromptCatalog, lang: str, *, caller: CallerFacts | None,
                    web_block: str | None, fallback: bool = False) -> str:
    """The per-call part in ``lang``: ``[web block "\\n"]`` and the caller context. Takes no model
    key.

    * ``caller`` ``None``: ``agent.caller_context_unverified`` as is (no values, memory gating).
    * ``caller`` given: ``agent.caller_context`` filled in one pass. An empty first name reads
      :data:`UNKNOWN_NAME`, an empty profile or memory :data:`EMPTY_VALUE`. The first name
      and the profile outside its wrapped blocks are escaped, as is memory outside its block.
    * ``fallback``: :data:`FALLBACK_CONTEXT` alone, no catalog frame and no web block, for the
      bridge's fallback prompt (no verification claim); ``caller`` and ``web_block`` must be
      ``None``.

    ``web_block`` is one of :data:`WEB_BLOCK_KEYS` or ``None``; a block missing from the catalog is
    left out with ``web_activity_block_missing`` (WARNING). A missing frame is left out with
    ``caller_context_missing`` (WARNING, once per process per key and language): no caller data.
    """
    if web_block is not None and web_block not in WEB_BLOCK_KEYS:
        raise ValueError("web_block must be a web-activity block key or None")
    if fallback and (caller is not None or web_block is not None):
        raise ValueError("the fallback per-call part takes no caller and no web block")
    if fallback:
        return _per_lang(FALLBACK_CONTEXT, lang)
    out: list[str] = []
    if web_block is not None:
        block = _block(catalog, web_block, lang)
        if block is None:
            emit(log, logging.WARNING, "web_activity_block_missing", language=lang, key=web_block)
        else:
            out.append(fill(block, {}))
    values: dict[str, str] = {}
    if caller is None:
        key = CALLER_CONTEXT_UNVERIFIED_KEY
    else:
        key = CALLER_CONTEXT_KEY
        empty = _per_lang(EMPTY_VALUE, lang)
        values = {"firstName": _or_empty(untrusted.neutralize(caller.first_name), _per_lang(UNKNOWN_NAME, lang)),
                  "profile": _or_empty(_neutralize_outside_blocks(caller.profile), empty),
                  "memory": _or_empty(None if caller.memory is None else _neutralize_outside_blocks(caller.memory),
                                      empty)}
    frame = _block(catalog, key, lang)
    if frame is None:
        _once(logging.WARNING, "caller_context_missing", key, lang, decision="leave_out")
    else:
        out.append(fill(frame, values))
    return "\n".join(out)


def render(catalog: PromptCatalog, model_key: str, lang: str, *, caller: CallerFacts | None,
           web_block: str | None) -> RenderedPrompt:
    """The system prompt for ``model_key`` in ``lang`` (without the clock line), as its
    static part (:func:`render_static`) and per-call part (:func:`render_per_call`)."""
    static = render_static(catalog, model_key, lang)  # a missing model text raises first
    return RenderedPrompt(static=static, per_call=render_per_call(catalog, lang, caller=caller, web_block=web_block))
