"""keyed-phone-login: a seed keeps prompt texts edited in the console unless
``--overwrite-prompts``. For any set of live edits and seed-file changes, a seed without the flag
never changes an edited text and adds every new key and language; with the flag the served catalog
equals the seed file. Either way the meta ``keys`` index names only keys with stored items.

Part 2: an item whose text still hashes to its ``seed_hash`` is not a console edit, so it always
ends at the latest seed text, over any sequence of seed-file changes, console edits and flags; an
item without a hash (seeded before part 2) is kept like an edit until one ``--overwrite-prompts``."""

from __future__ import annotations

import copy
import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.agent_prompt import CALLER_CONTEXT_KEYS, placeholder_problems
from owlbank.catalog import META_KEY, META_LANG, PromptCatalog, _doc_from_store
from owlbank.store import MemoryStore
from scripts import seed
from scripts._common import ScriptError

LANGS = ("en-US", "es-US")
KEYS = ("greeting.a", "greeting.b", "menu.c", "agent.system.x", "option.card.d")
CONSOLE_KEYS = ("mask.tool.1", "arg.kind.x")
TEXTS = st.sampled_from(["Hi.", "Hello.", "Hola.", "Bye {name}.", "Adiós."])

Entry = tuple[str, str]


def doc(entries: dict[Entry, tuple[str, int]]) -> dict[str, Any]:
    return {"defaultLanguage": "en-US", "languages": {lang: {} for lang in LANGS},
            "entries": [{"key": k, "lang": lang, "text": t, "version": v}
                        for (k, lang), (t, v) in sorted(entries.items())]}


def entries_strategy() -> st.SearchStrategy[dict[Entry, tuple[str, int]]]:
    pairs = [(k, lang) for k in KEYS for lang in LANGS]
    return st.dictionaries(st.sampled_from(pairs), st.tuples(TEXTS, st.integers(0, 2)), min_size=1)


def run_seed(store: MemoryStore, catalog: dict[str, Any], *, overwrite: bool) -> seed.PromptsMerge:
    items = PromptCatalog.to_items(catalog)
    merge = seed.merge_catalog_prompts(items, seed.read_live_catalog(store), overwrite=overwrite)
    for item in merge.items:
        store.put("catalog", item)
    return merge


def stored(store: MemoryStore) -> dict[Entry, dict[str, Any]]:
    return {(i["key"], i["lang"]): i for i in store.scan("catalog")
            if (i["key"], i["lang"]) != (META_KEY, META_LANG)}


def view_rev(items: dict[Entry, dict[str, Any]], key: str) -> int:
    """The console's view rev of ``key`` (the admin API's: the highest rev over its languages)."""
    return max((i.get("rev", 0) for (k, _), i in items.items() if k == key), default=0)


def visible(item: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(item.get(f) for f in ("text", "version", "status", "audio_url"))


def index(store: MemoryStore) -> list[str]:
    meta = store.get("catalog", {"key": META_KEY, "lang": META_LANG})
    assert meta is not None
    return list(meta["keys"])


@settings(max_examples=300, deadline=None)
@given(first=entries_strategy(), second=entries_strategy(),
       edits=st.dictionaries(st.sampled_from([(k, lang) for k in KEYS for lang in LANGS]), TEXTS),
       added=st.lists(st.sampled_from(CONSOLE_KEYS), unique=True), overwrite=st.booleans())
def test_property_keep_vs_overwrite(first: dict[Entry, tuple[str, int]],
                                    second: dict[Entry, tuple[str, int]], edits: dict[Entry, str],
                                    added: list[str], overwrite: bool) -> None:
    # A seed never deletes a stored item: a language dropped from a key the seed file keeps stays
    # served (as before). So the seed changes here keep each kept key's languages.
    second = {**{e: v for e, v in first.items() if e[0] in {k for k, _ in second}}, **second}
    store = MemoryStore()
    run_seed(store, doc(first), overwrite=False)
    # Console edits (as put_prompt writes them: text and a bumped rev) and console-added keys.
    for (key, lang), text in edits.items():
        if store.get("catalog", {"key": key, "lang": lang}) is not None:
            store.update("catalog", {"key": key, "lang": lang}, {"text": text, "rev": 3})
    for key in added:
        for lang in LANGS:
            store.put("catalog", {"key": key, "lang": lang, "text": "One moment.", "version": 0,
                                  "status": "published", "audio_url": None, "rev": 1})
    meta = store.get("catalog", {"key": META_KEY, "lang": META_LANG})
    assert meta is not None
    store.put("catalog", {**meta, "keys": [*meta["keys"], *added]})
    live = copy.deepcopy(stored(store))

    merge = run_seed(store, doc(second), overwrite=overwrite)
    after = stored(store)
    served = _doc_from_store(store)
    served_entries = {(e["key"], e["lang"]): (e["text"], e["version"]) for e in served["entries"]}

    if overwrite:
        # the served catalog equals the seed file
        assert served_entries == second
        assert set(index(store)) == {k for k, _ in second}
    else:
        for e, item in live.items():
            if e not in second or item["text"] == second[e][0]:
                continue
            if item["text"] != first[e][0]:  # edited in the console: no longer the seeded text
                assert after[e] == item  # an edited text (and its rev) is never changed
                assert e in merge.kept and e not in merge.legacy
            else:  # unedited: the seed file's new text is written, rev = the key's view rev + 1
                assert after[e]["text"] == second[e][0] and e in merge.updated
                assert after[e]["rev"] == view_rev(live, e[0]) + 1
        for e, (text, version) in second.items():
            if e not in live:  # every new key and language is added
                assert after[e]["text"] == text and after[e]["version"] == version
                assert e in merge.new
            assert e in served_entries
        # kept in the index: keys the console added (a console prefix, a console-written rev and no
        # seed_hash); a seeded key dropped from the seed file leaves it, edited in the console or not
        assert set(index(store)) == {k for k, _ in second} | set(added)
        assert set(merge.live_only) == set(added)
    # the index names only keys with stored items, each once, and the catalog loads
    keys = index(store)
    assert len(keys) == len(set(keys)) and all(any(k == ek for ek, _ in after) for k in keys)
    PromptCatalog.from_store(store)
    # no prompt text in the report
    report = seed.render_prompts_report(merge)
    assert not any(t in report for t in ("Hi.", "Hello.", "Hola.", "Bye {name}.", "Adiós.", "One moment."))
    # a second seed without the flag writes nothing but the meta item (idempotent)
    again = run_seed(store, doc(second), overwrite=False)
    assert [(i["key"], i["lang"]) for i in again.items] == [(META_KEY, META_LANG)]
    assert again.new == [] and again.replaced == []


PAIRS = [(k, lang) for k in KEYS for lang in LANGS]


@settings(max_examples=300, deadline=None)
@given(first=entries_strategy(), legacy=st.sets(st.sampled_from(PAIRS)),
       steps=st.lists(st.tuples(entries_strategy(), st.dictionaries(st.sampled_from(PAIRS), TEXTS),
                                st.booleans()), min_size=1, max_size=4))
def test_property_seed_file_changes_between_runs(
        first: dict[Entry, tuple[str, int]], legacy: set[Entry],
        steps: list[tuple[dict[Entry, tuple[str, int]], dict[Entry, str], bool]]) -> None:
    """Over seed-file changes between runs, interleaved with console edits and the flag: an
    unedited item (its text still the one the seed wrote) always ends at the latest seed text; an
    edited item, or one without a hash, never changes without the flag; with it every seeded entry
    is the seed file's, with its hash. ``seeded`` models the text each item's hash is of."""
    store = MemoryStore()
    run_seed(store, doc(first), overwrite=False)
    seeded: dict[Entry, str | None] = {e: t for e, (t, _) in first.items()}
    for e in legacy & set(first):  # seeded before part 2: no hash
        item = store.get("catalog", {"key": e[0], "lang": e[1]})
        assert item is not None
        store.put("catalog", {k: v for k, v in item.items() if k != seed.SEED_HASH})
        seeded[e] = None
    for entries, edits, overwrite in steps:
        for (key, lang), text in edits.items():  # as put_prompt: the old item, new text, rev + 1
            item = store.get("catalog", {"key": key, "lang": lang})
            if item is not None:
                store.put("catalog", {**item, "text": text, "rev": seed._next_rev(item)})
        before = copy.deepcopy(stored(store))
        merge = run_seed(store, doc(entries), overwrite=overwrite)
        after = stored(store)
        for e, item in before.items():
            if e not in entries:
                assert after[e] == item  # not in this seed file: untouched
        for e, (text, _) in entries.items():
            old = before.get(e)
            unedited = old is not None and seeded.get(e) is not None and old["text"] == seeded[e]
            if old is None or overwrite or unedited or old["text"] == text:
                assert after[e]["text"] == text  # the latest seed text
                assert after[e][seed.SEED_HASH] == seed.seed_hash(text)
                seeded[e] = text
            else:  # edited, or legacy: never changes without the flag
                assert after[e] == old
                assert e in merge.kept and (e in merge.legacy) == (seeded.get(e) is None)
            if old is not None:
                assert after[e].get("rev", 0) >= old.get("rev", 0)
        report = seed.render_prompts_report(merge)
        assert not any(t in report for t in ("Hi.", "Hello.", "Hola.", "Bye {name}.", "Adiós."))
    PromptCatalog.from_store(store)


@settings(max_examples=300, deadline=None)
@given(first=entries_strategy(),
       steps=st.lists(st.tuples(entries_strategy(), st.dictionaries(st.sampled_from(PAIRS), TEXTS),
                                st.lists(st.sampled_from(CONSOLE_KEYS), unique=True),
                                st.booleans()), min_size=1, max_size=4))
def test_property_view_rev_rises_and_dropped_seed_keys_leave_the_index(
        first: dict[Entry, tuple[str, int]],
        steps: list[tuple[dict[Entry, tuple[str, int]], dict[Entry, str], list[str], bool]]) -> None:
    """Over any sequence of seed files, console edits (as
    put_prompt: every language of the key at view rev + 1) and console-added keys: every text the
    seed changes raises its key's view rev, so an editor holding the old view conflicts; a key
    dropped from the seed file leaves the index unless the console created it."""
    store = MemoryStore()
    run_seed(store, doc(first), overwrite=False)
    console_keys: set[str] = set()
    for entries, edits, added, overwrite in steps:
        for (key, lang), text in edits.items():
            item = store.get("catalog", {"key": key, "lang": lang})
            if item is not None:
                rev = view_rev(stored(store), key) + 1
                store.put("catalog", {**item, "text": text, "rev": rev})
        for key in added:
            rev = view_rev(stored(store), key) + 1
            for lang in LANGS:
                old = store.get("catalog", {"key": key, "lang": lang}) or {}
                store.put("catalog", {"key": key, "lang": lang, "text": "One moment.", "version": 0,
                                      "status": "published", "audio_url": None, **old, "rev": rev})
            meta = store.get("catalog", {"key": META_KEY, "lang": META_LANG})
            assert meta is not None
            if key not in meta["keys"]:
                store.put("catalog", {**meta, "keys": [*meta["keys"], key]})
        console_keys |= set(added)
        before = copy.deepcopy(stored(store))
        run_seed(store, doc(entries), overwrite=overwrite)
        after = stored(store)
        for key in {k for k, _ in after}:
            entries_of = [e for e in after if e[0] == key]
            if not any(e in before for e in entries_of):
                continue  # a new key: no view an editor could hold
            if any(after[e].get("text") != (before.get(e) or {}).get("text") for e in entries_of):
                assert view_rev(after, key) > view_rev(before, key), key
            elif all(visible(after[e]) == visible(before[e]) for e in entries_of):
                assert view_rev(after, key) == view_rev(before, key), key  # hash-only: rev kept
        seed_keys = {k for k, _ in entries}
        expected = seed_keys if overwrite else seed_keys | console_keys
        assert set(index(store)) == expected
        if overwrite:
            console_keys = set()  # dropped from the index: no longer served, not re-added
    PromptCatalog.from_store(store)


# --------------------------------------------------------------------------- agent prompt caching: placeholder check (seed, preflight)

PREFLIGHT = Path(__file__).resolve().parents[2] / "infra" / "scripts" / "preflight.py"
D5_LANGS = ("en-US", "es-US", "fr-CA")
D5_KEYS = ("agent.system.haiku-5-5", "agent.system.opus-5-5", "agent.web_activity", "agent.web_activity_context",
           *CALLER_CONTEXT_KEYS, "agent.recap.system", "agent", "agentx.system", "greeting.welcome")
#: Fragments that make the allowed sets, wrong sets and stray braces likely.
_FRAGMENTS = st.sampled_from(["Hi. ", "{firstName}", "{profile}", "{memory}", "{name}", "{", "}", "{{", "{ x }",
                              "{9a}", "\n", "<caller_context>", "é"])
_D5_TEXT = st.one_of(
    st.lists(_FRAGMENTS, max_size=6).map("".join),
    st.just("<caller_context>\n{firstName} {profile} {memory}\n</caller_context>"),
    st.text(max_size=10),
)


@pytest.fixture(scope="module")
def preflight() -> Iterator[ModuleType]:
    name = "_preflight_for_seed_prompt_props"
    spec = importlib.util.spec_from_file_location(name, PREFLIGHT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up while building the class
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@st.composite
def d5_catalogs(draw: Any) -> dict[str, Any]:
    langs = draw(st.lists(st.sampled_from(D5_LANGS), min_size=1, max_size=3, unique=True))
    entries = [{"key": key, "lang": lang, "text": draw(_D5_TEXT), "version": draw(st.integers(0, 2)),
                "status": draw(st.sampled_from(["published", "published", "draft"]))}
               for key in D5_KEYS for lang in (*langs, "xx-XX") for _ in range(draw(st.integers(0, 2)))]
    return {"defaultLanguage": langs[0], "languages": draw(st.sampled_from([{la: {} for la in langs}, list(langs)])),
            "entries": draw(st.permutations(entries))}


def expected_problems(doc: dict[str, Any]) -> list[str]:
    """By definition: every published entry's :func:`placeholder_problems`, in catalog order,
    then each caller-context frame with no published entry in a catalog language."""
    out = [f"{e['key']} ({e['lang']}): {why}" for e in doc["entries"]
           if e.get("status", "published") == "published"
           for why in [placeholder_problems(e["key"], e["text"])] if why is not None]
    published = {(e["key"], e["lang"]) for e in doc["entries"] if e.get("status", "published") == "published"}
    return out + [f"{key} ({lang}): missing" for lang in doc["languages"] for key in CALLER_CONTEXT_KEYS
                  if (key, lang) not in published]


@settings(max_examples=400, deadline=None)
@given(doc=d5_catalogs())
def test_property_the_seeds_verdict_is_placeholder_problems_per_entry(doc: dict[str, Any]) -> None:
    """Seed half: the seed names exactly the entries ``placeholder_problems`` refuses and the
    missing frames, and refuses exactly when there is one."""
    expected = expected_problems(doc)
    assert seed.agent_text_problems(doc) == expected
    if expected:
        try:
            seed.check_agent_texts(doc)
        except ScriptError as exc:
            assert all(p in str(exc) for p in expected)
        else:
            raise AssertionError("check_agent_texts accepted a catalog with problems")
    else:
        seed.check_agent_texts(doc)


@settings(max_examples=400, deadline=None)
@given(doc=d5_catalogs())
def test_property_preflights_verdict_equals_the_seeds(preflight: ModuleType, doc: dict[str, Any]) -> None:
    """Preflight half: preflight's standalone copy names the same problems, and its line FAILs
    exactly when the seed refuses."""
    expected = expected_problems(doc)
    copy_ = preflight.PromptCatalogCopy(doc)
    assert preflight.agent_text_problems(copy_) == expected
    (line,) = preflight.check_agent_texts(preflight.PromptRecords(catalog=copy_))
    assert line.ok == (not expected) and all(p in line.detail for p in expected)
