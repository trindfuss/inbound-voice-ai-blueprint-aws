"""The batched policy and catalog reloads load exactly what the per-item reads they
replace loaded, for any store contents (any items, any JSON values, missing keys).

The oracles below are the pre-18.14 loaders (one ``GetItem`` per policy item, one ``Query`` per
catalog key), kept verbatim as test-only references. Each example compares, for the same store:
the rebuilt document or the error raised (type and message), what ``_parse`` makes of it, the
records logged on the way, and what a cached object serves after a refresh (the last good copy on
any failure)."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank import catalog as catalog_mod
from owlbank import policy as policy_mod
from owlbank.catalog import META_KEY, META_LANG, CatalogError, PromptCatalog
from owlbank.policy import (
    _RESOLUTION_PROBLEMS_KEY,
    _SLOT_PROBLEMS_KEY,
    _STORE_ONLY_ATTRS,
    META_PK,
    Policy,
    PolicyError,
    _is_int,
)
from owlbank.slot_policy import RESOLUTION_ITEM, SlotPolicyError, load_slot, load_slot_resolution
from owlbank.store import MemoryStore, Store

DATA = Path(__file__).resolve().parents[2] / "data"
POLICY_SEED = json.loads((DATA / "policy.json").read_text(encoding="utf-8"))
CATALOG_SEED = json.loads((DATA / "prompt_catalog.json").read_text(encoding="utf-8"))
POLICY_ITEMS = Policy.to_items(POLICY_SEED)
CATALOG_ITEMS = PromptCatalog.to_items(CATALOG_SEED)

_policy_log = logging.getLogger("owlbank.policy")


# --------------------------------------------------------------------------- oracles (pre-18.14)


def oracle_policy_doc(store: Store) -> dict[str, Any]:
    """The per-item policy loader as it was before 18.14."""
    meta = store.get("policy", {"pk": META_PK})
    if meta is None:
        raise PolicyError("policy table has no meta item")
    max_level = meta.get("max_level")
    if not _is_int(max_level) or max_level < 1:
        raise PolicyError("policy meta max_level must be a positive int")
    levels: dict[str, Any] = {}
    for n in range(1, max_level + 1):
        item = store.get("policy", {"pk": f"level#{n}"})
        if item is None:
            raise PolicyError(f"policy table has no level#{n}")
        levels[str(n)] = item.get("factors")
    tools: dict[str, Any] = {}
    for name in meta.get("tools", []):
        item = store.get("policy", {"pk": f"tool#{name}"})
        if item is None:
            raise PolicyError(f"policy meta lists tool {name!r} but has no row")
        tools[name] = {k: v for k, v in item.items() if k not in _STORE_ONLY_ATTRS}
    reasons: dict[str, Any] = {}
    for name in meta.get("reasons", []):
        item = store.get("policy", {"pk": f"reason#{name}"})
        if item is None:
            raise PolicyError(f"policy meta lists reason {name!r} but has no row")
        reasons[name] = {k: v for k, v in item.items() if k not in _STORE_ONLY_ATTRS}
    factors = meta.get("factors", {})
    slots, problems = _oracle_slots(store, meta.get("slots"),
                                    factors if isinstance(factors, Mapping) else {})
    resolution, resolution_problems = load_slot_resolution(_oracle_section(store, RESOLUTION_ITEM))
    for p in resolution_problems:
        _policy_log.error("policy slot_resolution field %s is invalid (%s)", p.field, p.message,
                          extra={"decision": "slot_resolution_fail_closed", "field": p.field})
    return {
        "version": meta.get("version", 0),
        "max_level": max_level,
        "levels": levels,
        "factors": factors,
        "slots": slots,
        _SLOT_PROBLEMS_KEY: problems,
        RESOLUTION_ITEM: resolution.to_doc(),
        _RESOLUTION_PROBLEMS_KEY: resolution_problems,
        "tools": tools,
        "escalation": {
            "default_reason": meta.get("default_reason"),
            "queues": meta.get("queues", {}),
            "reasons": reasons,
        },
        "limits": meta.get("limits", {}),
        "redaction": _oracle_section(store, "redaction"),
        "ani_trust": _oracle_section(store, "ani_trust"),
        "masking": _oracle_section(store, "masking"),
        "web_offers": _oracle_section(store, "web_offers"),
        "model_failover": _oracle_section(store, "model_failover"),
        "human_leg": _oracle_section(store, "human_leg"),
        "memory": _oracle_section(store, "memory"),
        "end_call": _oracle_section(store, "end_call"),
    }


def _oracle_slots(store: Store, listed: Any, factor_defs: Mapping[str, Any]
                  ) -> tuple[dict[str, Any], list[SlotPolicyError]]:
    problems: list[SlotPolicyError] = []
    if listed is None:
        listed = []
    if not isinstance(listed, list) or not all(isinstance(n, str) and n for n in listed):
        problems.append(SlotPolicyError("meta.slots", "must be a list of names; ignored"))
        listed = []
    problems.extend(SlotPolicyError(f"factors.{name}.sensitive", f"moved to slot#{name}; ignored")
                    for name, fd in factor_defs.items() if isinstance(fd, Mapping) and "sensitive" in fd)
    names = [*factor_defs, *(n for n in dict.fromkeys(listed) if n not in factor_defs)]
    out: dict[str, Any] = {}
    for name in names:
        item = _oracle_section(store, f"slot#{name}") if name in listed else None
        settings_, errors = load_slot(name, item, factor_def=factor_defs.get(name))
        problems.extend(errors)
        if settings_ is not None:
            out[name] = settings_.to_doc()
    for p in problems:
        _policy_log.error("policy slot field %s is invalid (%s)", p.field, p.message,
                          extra={"decision": "slot_fail_closed", "field": p.field})
    return out, problems


def _oracle_section(store: Store, pk: str) -> Any:
    item = store.get("policy", {"pk": pk})
    return None if item is None else {k: v for k, v in item.items() if k not in _STORE_ONLY_ATTRS}


def oracle_catalog_doc(store: Store) -> dict[str, Any]:
    """The per-key catalog loader as it was before 18.14."""
    meta = store.get("catalog", {"key": META_KEY, "lang": META_LANG})
    if meta is None:
        raise CatalogError("catalog table has no meta item")
    entries: list[dict[str, Any]] = []
    for key in meta.get("keys", []):
        for item in store.query("catalog", {"key": key}):
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


# --------------------------------------------------------------------------- comparison


def norm(x: Any) -> Any:
    """A comparable form: exceptions by type, message and fields; dataclasses by their fields."""
    if isinstance(x, SlotPolicyError):
        return ("SlotPolicyError", x.field, x.message, x.mode_invalid)
    if isinstance(x, BaseException):
        return (type(x).__name__, str(x))
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return (type(x).__name__, tuple((f.name, norm(getattr(x, f.name))) for f in dataclasses.fields(x)))
    if isinstance(x, Mapping):
        return ("map", tuple((norm(k), norm(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, tuple(norm(v) for v in x))
    if isinstance(x, (set, frozenset)):
        return ("set", tuple(sorted(repr(norm(v)) for v in x)))
    return x


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[tuple[str, str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name != "owlbank.dependency":  # the store's own call timing differs by design
            self.lines.append((record.name, record.levelname, record.getMessage()))


@contextmanager
def recorded() -> Iterator[_Records]:
    handler = _Records()
    root = logging.getLogger("owlbank")
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)


def outcome(load: Callable[[Store], dict[str, Any]], parse: Callable[[Any], Any],
            store: Store) -> tuple[Any, Any, list[tuple[str, str, str]]]:
    """``(doc or error, parsed or error, log lines)``."""
    with recorded() as rec:
        try:
            doc: Any = load(store)
        except Exception as exc:  # noqa: BLE001 - the error is the outcome being compared
            return norm(exc), None, rec.lines
        try:
            parsed: Any = norm(parse(doc))
        except Exception as exc:  # noqa: BLE001
            parsed = norm(exc)
    return norm(doc), parsed, rec.lines


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


# --------------------------------------------------------------------------- strategies

#: Any value the store holds: no floats (store contract), maps with string keys.
json_values = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(-3, 300), st.text(max_size=8)),
    lambda inner: st.one_of(st.lists(inner, max_size=4),
                            st.dictionaries(st.text(max_size=6), inner, max_size=4)),
    max_leaves=10)

POLICY_NAMES = sorted({*POLICY_SEED["tools"], *POLICY_SEED["escalation"]["reasons"],
                       *POLICY_SEED.get("slots", {}), *POLICY_SEED["factors"], "x", ""})
policy_names = st.one_of(st.sampled_from(POLICY_NAMES), st.text(max_size=6))
policy_pks = st.one_of(
    st.sampled_from([META_PK, "redaction", "ani_trust", "masking", "web_offers", "model_failover", "human_leg", "memory", "end_call",
                     RESOLUTION_ITEM]),
    st.builds(lambda n: f"level#{n}", st.integers(0, 5)),
    st.builds(lambda p, n: f"{p}#{n}", st.sampled_from(["tool", "reason", "slot"]), policy_names),
).filter(bool)
META_FIELDS = ["max_level", "tools", "reasons", "slots", "factors", "default_reason", "queues",
               "limits", "version"]
# Name lists for the meta fields that index other rows, as well as arbitrary JSON.
name_lists = st.lists(policy_names, max_size=6)
meta_values = st.one_of(json_values, name_lists, st.integers(0, 5))


def _field_names(item: Mapping[str, Any]) -> list[str]:
    return sorted(k for k in item if k != "pk") or ["x"]


@st.composite
def policy_tables(draw: st.DrawFn) -> list[dict[str, Any]]:
    """The seed's items, then deletions, field changes and extra items."""
    seeded = draw(st.integers(0, 9)) > 0  # mostly the seed, so most examples get past the meta
    items = {it["pk"]: dict(it) for it in (POLICY_ITEMS if seeded else [])}
    for _ in range(draw(st.integers(0, 6))):
        op = draw(st.sampled_from(["delete", "set", "set", "drop", "add", "add", "meta"]))
        pks = sorted(items)
        if op == "add" or not pks:
            pk = draw(policy_pks)
            items[pk] = {"pk": pk, **draw(st.dictionaries(st.text(min_size=1, max_size=6).filter(
                lambda k: k != "pk"), json_values, max_size=4))}
            continue
        if op == "meta":
            meta = items.setdefault(META_PK, {"pk": META_PK})
            meta[draw(st.sampled_from(META_FIELDS))] = draw(meta_values)
            continue
        pk = draw(st.sampled_from(pks))
        if op == "delete":
            del items[pk]
        elif op == "set":
            items[pk][draw(st.sampled_from([*_field_names(items[pk]), "rev", "sensitive", "mode"]))] = \
                draw(json_values)
        else:
            items[pk].pop(draw(st.sampled_from(_field_names(items[pk]))), None)
    return list(items.values())


CATALOG_KEYS = sorted({e["key"] for e in CATALOG_SEED["entries"]})
SMALL_CATALOG = {
    "defaultLanguage": "en-US",
    "languages": {"en-US": {}, "es-US": {}},
    "singleLanguageKeys": {"lang.es": "es-US"},
    "entries": [
        {"key": "a", "lang": "en-US", "text": "A", "version": 1, "status": "published"},
        {"key": "a", "lang": "es-US", "text": "Á", "version": 1, "status": "published"},
        {"key": "b", "lang": "en-US", "text": "B", "version": 2, "status": "draft"},
        {"key": "lang.es", "lang": "es-US", "text": "Español", "version": 0},
    ],
}
SMALL_ITEMS = PromptCatalog.to_items(SMALL_CATALOG)
catalog_keys = st.one_of(st.sampled_from([*CATALOG_KEYS[:5], "a", "b", "lang.es", META_KEY, "zz"]),
                         st.text(min_size=1, max_size=5))
catalog_langs = st.one_of(st.sampled_from(["en-US", "es-US", META_LANG, "fr-FR"]),
                          st.text(min_size=1, max_size=5))
CATALOG_META_FIELDS = ["keys", "defaultLanguage", "languages", "singleLanguageKeys"]
ENTRY_FIELDS = ["text", "version", "status", "audio_url"]


@st.composite
def catalog_tables(draw: st.DrawFn) -> list[dict[str, Any]]:
    base = draw(st.sampled_from(["seed", "seed", "small", "small", "empty"]))
    source = {"seed": CATALOG_ITEMS, "small": SMALL_ITEMS, "empty": []}[base]
    items = {(it["key"], it["lang"]): dict(it) for it in source}
    for _ in range(draw(st.integers(0, 6))):
        op = draw(st.sampled_from(["delete", "set", "set", "drop", "add", "add", "meta"]))
        keys = sorted(items)
        if op == "add" or not keys:
            k = (draw(catalog_keys), draw(catalog_langs))
            items[k] = {"key": k[0], "lang": k[1], **draw(st.fixed_dictionaries(
                {}, optional=dict.fromkeys(ENTRY_FIELDS, json_values)))}
            continue
        if op == "meta":
            meta = items.setdefault((META_KEY, META_LANG), {"key": META_KEY, "lang": META_LANG})
            field = draw(st.sampled_from(CATALOG_META_FIELDS))
            meta[field] = draw(st.one_of(json_values, st.lists(catalog_keys, max_size=6))
                               if field == "keys" else json_values)
            continue
        k = draw(st.sampled_from(keys))
        if op == "delete":
            del items[k]
        elif op == "set":
            items[k][draw(st.sampled_from(ENTRY_FIELDS))] = draw(json_values)
        else:
            items[k].pop(draw(st.sampled_from(ENTRY_FIELDS)), None)
    return list(items.values())


def store_of(table: str, items: list[dict[str, Any]]) -> MemoryStore:
    store = MemoryStore()
    for item in items:
        store.put(table, item)
    return store


# --------------------------------------------------------------------------- properties

SETTINGS = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@SETTINGS
@given(items=policy_tables())
def test_policy_reload_matches_per_item_reads(items: list[dict[str, Any]]) -> None:
    store = store_of("policy", items)
    assert outcome(policy_mod._doc_from_store, policy_mod._parse, store) == \
        outcome(oracle_policy_doc, policy_mod._parse, store)


@SETTINGS
@given(max_level=st.integers(95, 260), missing=st.sets(st.integers(1, 260), max_size=3))
def test_policy_reload_matches_beyond_one_batch_of_levels(max_level: int, missing: set[int]) -> None:
    """Levels past the first chunk are fetched as needed; the first missing level is reported."""
    items = [dict(it) for it in POLICY_ITEMS if not it["pk"].startswith("level#")]
    items[0] = {**items[0], "max_level": max_level}
    items += [{"pk": f"level#{n}", "factors": ["ani_match"]} for n in range(1, max_level + 1)
              if n not in missing]
    store = store_of("policy", items)
    assert outcome(policy_mod._doc_from_store, policy_mod._parse, store) == \
        outcome(oracle_policy_doc, policy_mod._parse, store)


@SETTINGS
@given(items=catalog_tables())
def test_catalog_reload_matches_per_key_reads(items: list[dict[str, Any]]) -> None:
    store = store_of("catalog", items)
    assert outcome(catalog_mod._doc_from_store, catalog_mod._parse, store) == \
        outcome(oracle_catalog_doc, catalog_mod._parse, store)


@SETTINGS
@given(policy_items=policy_tables(), catalog_items=catalog_tables())
def test_cached_objects_serve_the_same_after_a_refresh(policy_items: list[dict[str, Any]],
                                                      catalog_items: list[dict[str, Any]]) -> None:
    """From the seed, the table is replaced and the TTL passes: both loaders leave the cache in the
    same state (the new copy, or the last good one on any failure)."""
    seeded = store_of("policy", POLICY_ITEMS)
    for item in CATALOG_ITEMS:
        seeded.put("catalog", item)
    clock = Clock()
    new_policy = Policy.from_store(seeded, ttl_s=60, clock=clock)
    old_policy = Policy(lambda: oracle_policy_doc(seeded), ttl_s=60, clock=clock)
    new_cat = PromptCatalog.from_store(seeded, ttl_s=60, clock=clock)
    old_cat = PromptCatalog(lambda: oracle_catalog_doc(seeded), ttl_s=60, clock=clock)
    for pk in [it["pk"] for it in seeded.scan("policy")]:
        seeded.delete("policy", {"pk": pk})
    for it in seeded.scan("catalog"):
        seeded.delete("catalog", {"key": it["key"], "lang": it["lang"]})
    for item in policy_items:
        seeded.put("policy", item)
    for item in catalog_items:
        seeded.put("catalog", item)
    clock.t = 61
    assert norm(new_policy._current()) == norm(old_policy._current())
    assert norm(new_cat._current()) == norm(old_cat._current())
