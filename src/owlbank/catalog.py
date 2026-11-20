"""The prompt catalog: every caller-facing phrase, by key and language.

Two sources share one validated shape:

* :meth:`PromptCatalog.from_dict` takes the seed file (``data/prompt_catalog.json``).
* :meth:`PromptCatalog.from_store` reads the ``catalog`` table and re-reads it after
  ``ttl_s`` seconds (default 60). A failed refresh keeps the last good catalog.

Only ``status == "published"`` entries are served. If a source has several published
entries for one key and language, the highest ``version`` wins.

Store layout (``catalog`` table, pk ``key``, sk ``lang``), written by :meth:`PromptCatalog.to_items`:

* ``key="_meta", lang="_"``: ``defaultLanguage``, ``languages`` (per-language say,
  gather and conversationRelay settings), ``singleLanguageKeys``, and ``keys`` (the
  index of entry keys: a reload scans the table and serves only the keys listed here).
* one item per entry: ``key``, ``lang``, ``text``, ``version``, ``status``, ``audio_url``.

Rendering rules:

* A key missing in the requested language falls back to the default language, with
  a logged warning. A key in ``singleLanguageKeys`` is served in its one language.
* ``{name}`` placeholders are filled only from the keyword values passed in. Unknown
  placeholders are removed. Substitution is a single pass over the *template*, so a
  value that itself contains ``{x}`` is never expanded, and ``str.format`` is never used.
"""

from __future__ import annotations

import copy
import logging
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from owlbank.obs import log_caught
from owlbank.store import Store, timed_reload

log = logging.getLogger(__name__)

META_KEY, META_LANG = "_meta", "_"
DEFAULT_TTL_S = 60
COMMAND_PREFIX = "command."
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class CatalogError(ValueError):
    """The catalog data is invalid."""


class CatalogKeyError(KeyError):
    """No published entry for the key in the language or its fallbacks."""


@dataclass(frozen=True, slots=True)
class Entry:
    key: str
    lang: str
    text: str
    version: int
    audio_url: str | None


@dataclass(frozen=True, slots=True)
class _CatalogData:
    default_language: str
    languages: Mapping[str, Mapping[str, Any]]
    single_language_keys: Mapping[str, str]
    entries: Mapping[tuple[str, str], Entry]
    keys: frozenset[str]


# --------------------------------------------------------------------------- parsing


def _parse(doc: Mapping[str, Any]) -> _CatalogData:
    if not isinstance(doc, Mapping):
        raise CatalogError("catalog must be a mapping")
    default = doc.get("defaultLanguage")
    languages = doc.get("languages")
    if not isinstance(languages, Mapping) or not languages:
        raise CatalogError("catalog needs a non-empty 'languages' mapping")
    if not isinstance(default, str) or default not in languages:
        raise CatalogError(f"defaultLanguage {default!r} is not a configured language")
    single = doc.get("singleLanguageKeys", {}) or {}
    if not isinstance(single, Mapping) or any(v not in languages for v in single.values()):
        raise CatalogError("singleLanguageKeys must map keys to configured languages")
    raw_entries = doc.get("entries", [])
    if not isinstance(raw_entries, list):
        raise CatalogError("entries must be a list")
    entries: dict[tuple[str, str], Entry] = {}
    for i, e in enumerate(raw_entries):
        if not isinstance(e, Mapping):
            raise CatalogError(f"entry {i} is not a mapping")
        key, lang, text = e.get("key"), e.get("lang"), e.get("text")
        if not isinstance(key, str) or not key or not isinstance(text, str):
            raise CatalogError(f"entry {i} needs a string key and text")
        if not isinstance(lang, str) or lang not in languages:
            raise CatalogError(f"entry {key!r} has unconfigured language {lang!r}")
        if e.get("status", "published") != "published":
            continue
        version = e.get("version", 0)
        if not isinstance(version, int) or isinstance(version, bool):
            raise CatalogError(f"entry {key!r}/{lang} version must be an int")
        audio = e.get("audioUrl", e.get("audio_url"))
        if audio is not None and not isinstance(audio, str):
            raise CatalogError(f"entry {key!r}/{lang} audio URL must be a string or null")
        prev = entries.get((key, lang))
        if prev is None or version > prev.version:
            entries[(key, lang)] = Entry(key, lang, text, version, audio or None)
    return _CatalogData(
        default_language=default,
        languages=copy.deepcopy(dict(languages)),
        single_language_keys=dict(single),
        entries=entries,
        keys=frozenset(k for k, _ in entries),
    )


def _doc_from_store(store: Store) -> dict[str, Any]:
    """Rebuild the seed-file-shaped dict from the ``catalog`` table.

    One paginated ``Scan`` (the table has no index to Query), then the
    entries the meta item's ``keys`` name, in that order and each key's languages in sort-key
    order: the same list the per-key Queries gave. Items no key names are ignored."""
    with timed_reload("catalog") as call:
        items = store.scan("catalog")
        call.items = len(items)
    meta: dict[str, Any] | None = None
    by_key: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if item["key"] == META_KEY and item["lang"] == META_LANG:
            meta = item
        by_key.setdefault(item["key"], []).append(item)
    if meta is None:
        raise CatalogError("catalog table has no meta item")
    entries: list[dict[str, Any]] = []
    for key in meta.get("keys", []):
        # A stored key is always a string; a listed non-string matches nothing (as a Query did).
        hits = by_key.get(key, []) if isinstance(key, str) else []
        for item in sorted(hits, key=lambda it: it["lang"]):
            entries.append({"key": item["key"], "lang": item["lang"], "text": item.get("text"),
                            "version": item.get("version", 0),
                            "status": item.get("status", "published"),
                            "audioUrl": item.get("audio_url")})
    return {
        "defaultLanguage": meta.get("defaultLanguage"),
        "languages": meta.get("languages", {}),
        "singleLanguageKeys": meta.get("singleLanguageKeys", {}),
        "entries": entries,
    }


def render(template: str, values: Mapping[str, Any]) -> str:
    """Fill ``{name}`` placeholders in ``template`` from ``values`` only.

    One pass over the template: replacement text is never re-scanned. Unknown
    placeholders are removed, along with one adjacent space so no double space or
    space-before-punctuation is left behind.
    """
    out: list[str] = []
    pos = 0
    for m in PLACEHOLDER.finditer(template):
        out.append(template[pos:m.start()])
        pos = m.end()
        name = m.group(1)
        if name in values and values[name] is not None:
            out.append(str(values[name]))
            continue
        nxt = template[pos:pos + 1]
        if out and out[-1].endswith(" ") and (nxt == "" or nxt == " " or nxt in ".,;:!?¿¡)"):
            out[-1] = out[-1][:-1]
    out.append(template[pos:])
    return "".join(out)


# --------------------------------------------------------------------------- PromptCatalog


class PromptCatalog:
    """Read-only prompt catalog, with an optional refresh cache."""

    def __init__(
        self,
        loader: Callable[[], Mapping[str, Any]],
        *,
        ttl_s: float | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._loader = loader
        self._ttl = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._data = _parse(loader())
        self._loaded_at = clock()

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> PromptCatalog:
        frozen = copy.deepcopy(dict(doc))
        return cls(lambda: frozen, ttl_s=None)

    @classmethod
    def from_store(
        cls, store: Store, *, ttl_s: float = DEFAULT_TTL_S, clock: Callable[[], float] = time.monotonic
    ) -> PromptCatalog:
        return cls(lambda: _doc_from_store(store), ttl_s=ttl_s, clock=clock)

    @staticmethod
    def to_items(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Store items for the ``catalog`` table from a seed-file-shaped dict (validated).

        Every entry is written, including non-published ones, so change control
        (draft, approved) lives in the table; only published entries are served.
        """
        data = _parse(doc)
        items: list[dict[str, Any]] = []
        keys: list[str] = []
        for e in doc.get("entries", []):
            if e["key"] not in keys:
                keys.append(e["key"])
            items.append({"key": e["key"], "lang": e["lang"], "text": e["text"],
                          "version": e.get("version", 0), "status": e.get("status", "published"),
                          "audio_url": e.get("audioUrl", e.get("audio_url"))})
        meta = {"key": META_KEY, "lang": META_LANG, "defaultLanguage": data.default_language,
                "languages": copy.deepcopy(dict(data.languages)),
                "singleLanguageKeys": dict(data.single_language_keys), "keys": keys}
        return [meta, *items]

    # -- cache

    def _current(self) -> _CatalogData:
        if self._ttl is None or self._clock() - self._loaded_at < self._ttl:
            return self._data
        with self._lock:
            if self._clock() - self._loaded_at >= self._ttl:
                try:
                    self._data = _parse(self._loader())
                except Exception as exc:
                    log_caught(log, "catalog refresh failed; keeping the last good catalog", exc,
                               level=logging.ERROR, operation="refresh_catalog", decision="keep_last_good")
                self._loaded_at = self._clock()
        return self._data

    def refresh(self) -> None:
        data = _parse(self._loader())
        with self._lock:
            self._data, self._loaded_at = data, self._clock()

    # -- reads

    @property
    def default_language(self) -> str:
        return self._current().default_language

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(self._current().languages)

    @property
    def keys(self) -> frozenset[str]:
        return self._current().keys

    def has(self, key: str, lang: str) -> bool:
        """Whether ``key`` has a published entry in exactly ``lang`` (no fallback)."""
        return (key, lang) in self._current().entries

    def entry(self, key: str, lang: str) -> Entry:
        """The entry served for ``key`` in ``lang``, after fallback. ``CatalogKeyError`` if none."""
        data = self._current()
        hit = data.entries.get((key, lang))
        if hit is not None:
            return hit
        single = data.single_language_keys.get(key)
        if single is not None and (key, single) in data.entries:
            return data.entries[(key, single)]
        hit = data.entries.get((key, data.default_language))
        if hit is None:
            raise CatalogKeyError(key)
        log.warning("catalog: no %r entry for %r; falling back to %r",
                    key, str(lang)[:16], data.default_language)
        return hit

    def text(self, key: str, lang: str, **values: Any) -> str:
        """Rendered text for ``key`` in ``lang``; placeholders only from ``values``."""
        return render(self.entry(key, lang).text, values)

    def template(self, key: str, lang: str) -> str:
        """The unrendered template (for validation and preview tools)."""
        return self.entry(key, lang).text

    def audio_url(self, key: str, lang: str) -> str | None:
        """The recorded-audio URL for the served entry, if any."""
        return self.entry(key, lang).audio_url

    def settings(self, lang: str) -> dict[str, Any]:
        """Per-language ``say``, ``gather`` and ``conversationRelay`` settings (a copy)."""
        data = self._current()
        if lang not in data.languages:
            log.warning("catalog: no settings for %r; using %r", str(lang)[:16], data.default_language)
            lang = data.default_language
        return copy.deepcopy(dict(data.languages[lang]))

    def commands(self, lang: str) -> dict[str, list[str]]:
        """Universal commands: ``{"repeat": [...phrases], "agent": [...], "cancel": [...]}``."""
        out: dict[str, list[str]] = {}
        for key in sorted(k for k in self._current().keys if k.startswith(COMMAND_PREFIX)):
            phrases = [p.strip() for p in self.entry(key, lang).text.split("|")]
            out[key[len(COMMAND_PREFIX):]] = [p for p in phrases if p]
        return out

    def missing_keys(self, lang: str) -> list[str]:
        """Keys that ``lang`` lacks (excluding single-language keys of other languages)."""
        data = self._current()
        return sorted(
            k for k in data.keys
            if (k, lang) not in data.entries and data.single_language_keys.get(k, lang) == lang
        )
