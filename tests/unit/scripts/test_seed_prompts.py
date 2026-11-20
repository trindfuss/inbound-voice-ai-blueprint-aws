"""scripts/seed.py keeps prompt texts edited in the console unless --overwrite-prompts."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from owlbank.catalog import PromptCatalog
from owlbank.store import MemoryStore
from scripts import seed
from tests.admin_api_fakes import call, claims, make_api
from tests.unit.scripts.test_seed import TABLE_ARGS, load, run

EDITED = ("greeting.generic", "es-US")
SECRET_TEXT = "Texto editado en la consola"
EDITOR = claims(["admin-editor"])


def seeded_store() -> MemoryStore:
    store = MemoryStore()
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    return store


def entry(store: MemoryStore, key: str, lang: str) -> dict[str, Any] | None:
    return store.get("catalog", {"key": key, "lang": lang})


def index(store: MemoryStore) -> list[str]:
    meta = store.get("catalog", {"key": "_meta", "lang": "_"})
    assert meta is not None
    return list(meta["keys"])


def edit(store: MemoryStore, key: str, lang: str, text: str, rev: int = 4) -> None:
    store.update("catalog", {"key": key, "lang": lang}, {"text": text, "rev": rev})


def overwrite_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING
            and getattr(r, "step", None) == "prompts" and getattr(r, "outcome", None) == "overwrite"]


def test_a_reseed_keeps_an_edited_prompt_and_reports_key_and_language_only(
        capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
    store = seeded_store()
    edit(store, *EDITED, SECRET_TEXT)
    capsys.readouterr()
    with caplog.at_level(logging.INFO, logger="owlbank.seed"):
        assert run(["--stage", "prod", *TABLE_ARGS, "--reseed", "--apply"], store) == 0
    item = entry(store, *EDITED)
    assert item is not None and item["text"] == SECRET_TEXT and item["rev"] == 4  # untouched
    out = capsys.readouterr().out
    assert "KEPT" in out and "greeting.generic (es-US)" in out and "--overwrite-prompts" in out
    assert SECRET_TEXT not in out and SECRET_TEXT not in caplog.text
    assert not overwrite_warnings(caplog)
    assert PromptCatalog.from_store(store).template(*EDITED) == SECRET_TEXT


def test_a_dry_run_reports_the_kept_prompt_and_writes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded_store()
    edit(store, *EDITED, SECRET_TEXT)
    capsys.readouterr()
    assert run(["--stage", "prod", *TABLE_ARGS, "--overwrite-prompts"], store) == 0
    assert entry(store, *EDITED)["text"] == SECRET_TEXT  # type: ignore[index]
    out = capsys.readouterr().out
    assert "REPLACED" in out and "greeting.generic (es-US)" in out
    # the plan shows the seed file's line (its text), never the live one
    assert SECRET_TEXT not in out


def test_overwrite_prompts_replaces_the_edit_reports_it_and_bumps_rev(
        capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
    store = seeded_store()
    seed_text = entry(store, *EDITED)["text"]  # type: ignore[index]
    edit(store, *EDITED, SECRET_TEXT)
    capsys.readouterr()
    with caplog.at_level(logging.INFO, logger="owlbank.seed"):
        assert run(["--stage", "prod", *TABLE_ARGS, "--overwrite-prompts", "--apply"], store) == 0
    item = entry(store, *EDITED)
    assert item is not None and item["text"] == seed_text and item["rev"] == 5
    out = capsys.readouterr().out
    assert "REPLACED" in out and "greeting.generic (es-US)" in out and SECRET_TEXT not in out
    [warn] = overwrite_warnings(caplog)
    assert warn.__dict__["replaced"] == ["greeting.generic (es-US)"]
    assert SECRET_TEXT not in caplog.text


@pytest.mark.parametrize("flag", [[], ["--overwrite-prompts"]])
def test_a_key_or_language_new_in_the_seed_is_added_either_way(flag: list[str]) -> None:
    store = seeded_store()
    store.delete("catalog", {"key": "greeting.personal", "lang": "en-US"})
    store.delete("catalog", {"key": "greeting.personal", "lang": "es-US"})
    store.delete("catalog", {"key": "balance.read", "lang": "es-US"})
    meta = store.get("catalog", {"key": "_meta", "lang": "_"})
    assert meta is not None
    store.put("catalog", {**meta, "keys": [k for k in meta["keys"] if k != "greeting.personal"]})
    assert run(["--stage", "prod", *TABLE_ARGS, *flag, "--apply"], store) == 0
    for key, lang in (("greeting.personal", "en-US"), ("greeting.personal", "es-US"), ("balance.read", "es-US")):
        assert entry(store, key, lang) is not None
    assert "greeting.personal" in index(store)
    assert PromptCatalog.from_store(store).has("balance.read", "es-US")


def test_a_key_added_in_the_console_stays_in_the_index_unless_overwrite() -> None:
    store = seeded_store()
    added = "mask.get_balance.9"
    for lang in ("en-US", "es-US"):
        store.put("catalog", {"key": added, "lang": lang, "text": "One moment.", "version": 0,
                              "status": "published", "audio_url": None, "rev": 1})
    meta = store.get("catalog", {"key": "_meta", "lang": "_"})
    assert meta is not None
    store.put("catalog", {**meta, "keys": [*meta["keys"], added, "ghost.key"], "rev": 3})
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    keys = index(store)
    assert added in keys and "ghost.key" not in keys  # only keys with stored items
    assert store.get("catalog", {"key": "_meta", "lang": "_"})["rev"] == 4  # type: ignore[index]
    assert PromptCatalog.from_store(store).has(added, "es-US")
    assert run(["--stage", "prod", *TABLE_ARGS, "--overwrite-prompts", "--apply"], store) == 0
    assert added not in index(store)


def test_a_kept_prompt_keeps_its_rev_so_an_open_editor_gets_no_false_conflict() -> None:
    """A console editor reads, a reseed runs, the editor saves: accepted."""
    store = seeded_store()
    api, store, _, _ = make_api(store)
    key = "greeting.generic"
    texts = {"en-US": "Hello, how can I help?", "es-US": "Hola, ¿en qué le ayudo?"}
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts}, who=EDITOR)
    assert status == 200 and body["prompt"]["rev"] == 1
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    texts2 = {"en-US": "Hello again.", "es-US": "Hola otra vez."}
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 1, "texts": texts2}, who=EDITOR)
    assert status == 200, body
    # after an overwrite the editor holding the old rev is refused (the text did change)
    assert run(["--stage", "prod", *TABLE_ARGS, "--overwrite-prompts", "--apply"], store) == 0
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 2, "texts": texts}, who=EDITOR)
    assert status == 409


def test_a_text_equal_to_the_seed_with_another_field_changed_is_written_with_rev_bumped() -> None:
    items = seed.PromptCatalog.to_items({
        "defaultLanguage": "en-US", "languages": {"en-US": {}},
        "entries": [{"key": "a.b", "lang": "en-US", "text": "Hi.", "version": 2}]})
    live = {("a.b", "en-US"): {"key": "a.b", "lang": "en-US", "text": "Hi.", "version": 1,
                               "status": "published", "audio_url": None, "rev": 7}}
    merge = seed.merge_catalog_prompts(items, live, overwrite=False)
    [written] = [i for i in merge.items if i["key"] == "a.b"]
    assert written["version"] == 2 and written["rev"] == 8 and merge.kept == []


# --------------------------------------------------------------------------- seed_hash (part 2)

OLD_SEED_TEXT = "An older seed text."


def as_if_seeded_with(store: MemoryStore, key: str, lang: str, text: str, rev: int = 2) -> None:
    """The item as an older seed file wrote it: its text and that text's seed_hash."""
    store.update("catalog", {"key": key, "lang": lang},
                 {"text": text, seed.SEED_HASH: seed.seed_hash(text), "rev": rev})


def test_every_item_the_seed_writes_carries_the_hash_of_its_text() -> None:
    store = seeded_store()
    items = [i for i in store.scan("catalog") if i["key"] != "_meta"]
    assert items and all(i[seed.SEED_HASH] == seed.seed_hash(i["text"]) for i in items)
    assert seed.SEED_HASH not in entry(store, "_meta", "_")  # type: ignore[operator]
    assert seed.seed_hash("Hi.").startswith("sha256:") and len(seed.seed_hash("Hi.")) == 7 + 64


def test_a_changed_seed_text_replaces_an_unedited_item_on_a_plain_reseed(
        capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
    store = seeded_store()
    seed_text = entry(store, *EDITED)["text"]  # type: ignore[index]
    as_if_seeded_with(store, *EDITED, OLD_SEED_TEXT)
    capsys.readouterr()
    with caplog.at_level(logging.INFO, logger="owlbank.seed"):
        assert run(["--stage", "prod", *TABLE_ARGS, "--reseed", "--apply"], store) == 0
    item = entry(store, *EDITED)
    assert item is not None and item["text"] == seed_text and item["rev"] == 3
    assert item[seed.SEED_HASH] == seed.seed_hash(seed_text)
    out = capsys.readouterr().out
    assert "not edited in the console: 1, written" in out and "greeting.generic (es-US)" in out
    assert "KEPT" not in out and OLD_SEED_TEXT not in out
    assert not overwrite_warnings(caplog)  # no console edit was overwritten


def test_an_item_edited_in_the_console_is_kept_though_the_seed_text_changed(
        capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded_store()
    as_if_seeded_with(store, *EDITED, OLD_SEED_TEXT)
    edit(store, *EDITED, SECRET_TEXT)  # the console keeps seed_hash: it no longer matches the text
    capsys.readouterr()
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    item = entry(store, *EDITED)
    assert item is not None and item["text"] == SECRET_TEXT and item["rev"] == 4
    out = capsys.readouterr().out
    assert "edited in the console" in out and "KEPT" in out and "legacy" not in out
    assert SECRET_TEXT not in out


def test_a_legacy_item_without_a_hash_is_kept_and_reported_until_the_flag_writes_hashes(
        capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded_store()
    seed_text = entry(store, *EDITED)["text"]  # type: ignore[index]
    legacy = {k: v for k, v in entry(store, *EDITED).items() if k != seed.SEED_HASH}  # type: ignore[union-attr]
    store.put("catalog", {**legacy, "text": OLD_SEED_TEXT, "rev": 2})
    capsys.readouterr()
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    assert entry(store, *EDITED)["text"] == OLD_SEED_TEXT  # type: ignore[index]
    out = capsys.readouterr().out
    assert "legacy, no hash" in out and "greeting.generic (es-US)" in out and OLD_SEED_TEXT not in out
    assert run(["--stage", "prod", *TABLE_ARGS, "--overwrite-prompts", "--apply"], store) == 0
    item = entry(store, *EDITED)
    assert item is not None and item["text"] == seed_text and item["rev"] == 3
    assert item[seed.SEED_HASH] == seed.seed_hash(seed_text)
    # from now on a changed seed text reaches it on a plain reseed
    as_if_seeded_with(store, *EDITED, OLD_SEED_TEXT, rev=3)
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    assert entry(store, *EDITED)["text"] == seed_text  # type: ignore[index]


@pytest.mark.parametrize("stored_hash", [None, "md5:abc", "sha256:" + "0" * 64, 42])
def test_a_legacy_or_stale_hash_on_the_seed_text_is_rewritten_with_rev_kept(stored_hash: Any) -> None:
    store = seeded_store()
    item = {k: v for k, v in entry(store, *EDITED).items() if k != seed.SEED_HASH}  # type: ignore[union-attr]
    store.put("catalog", {**item, **({seed.SEED_HASH: stored_hash} if stored_hash is not None else {}),
                          "rev": 5})
    items = PromptCatalog.to_items(load("prompt_catalog.json"))
    merge = seed.merge_catalog_prompts(items, seed.read_live_catalog(store), overwrite=False)
    assert merge.hashed == [EDITED] and merge.kept == [] and merge.updated == []
    [written] = [i for i in merge.items if (i["key"], i["lang"]) == EDITED]
    assert written["rev"] == 5 and written[seed.SEED_HASH] == seed.seed_hash(item["text"])
    assert "seed hash" in seed.render_prompts_report(merge)


def test_an_unknown_hash_tag_counts_as_legacy() -> None:
    items = PromptCatalog.to_items({"defaultLanguage": "en-US", "languages": {"en-US": {}},
                                    "entries": [{"key": "a.b", "lang": "en-US", "text": "New."}]})
    live = {("a.b", "en-US"): {"key": "a.b", "lang": "en-US", "text": "Old.", "version": 0,
                               "status": "published", "audio_url": None, "rev": 1,
                               seed.SEED_HASH: "sha3:" + "0" * 64}}
    merge = seed.merge_catalog_prompts(items, live, overwrite=False)
    assert merge.kept == [("a.b", "en-US")] and merge.legacy == [("a.b", "en-US")]


def test_the_admin_api_keeps_the_hash_on_an_edit_and_a_restore_and_never_shows_it() -> None:
    store = seeded_store()
    api, store, _, _ = make_api(store)
    key = "greeting.generic"
    seeded = {lang: entry(store, key, lang) for lang in ("en-US", "es-US")}
    texts = {"en-US": "Hello, how can I help?", "es-US": SECRET_TEXT}
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts}, who=EDITOR)
    assert status == 200 and seed.SEED_HASH not in str(body)
    for lang, old in seeded.items():
        assert entry(store, key, lang)[seed.SEED_HASH] == old[seed.SEED_HASH]  # type: ignore[index]
    status, body = call(api, "GET", "/prompts", query={"prefix": "greeting."}, who=EDITOR)
    assert status == 200 and seed.SEED_HASH not in str(body)
    status, body = call(api, "GET", "/audit", query={"item": f"catalog#{key}"}, who=EDITOR)
    assert status == 200 and seed.SEED_HASH not in str(body)
    [audit_entry] = [i for i in store.scan("admin_audit") if i["pk"] == f"catalog#{key}"]
    assert {c["field"].split("[")[0] for c in audit_entry["changes"]} == {"text"}
    # a reseed sees the console edit and keeps it
    merge = seed.merge_catalog_prompts(
        PromptCatalog.to_items(load("prompt_catalog.json")),
        seed.read_live_catalog(store), overwrite=False)
    assert set(merge.kept) == {(key, "en-US"), (key, "es-US")} and merge.legacy == []
    # a restore (back to the seeded text) keeps the hash too: the item is unedited again
    status, body = call(api, "POST", "/restore",
                        {"audit_pk": audit_entry["pk"], "audit_sk": audit_entry["sk"], "rev": 1}, who=EDITOR)
    assert status == 200 and seed.SEED_HASH not in str(body)
    for lang, old in seeded.items():
        now = entry(store, key, lang)
        assert now is not None and now["text"] == old["text"]  # type: ignore[index]
        assert now[seed.SEED_HASH] == old[seed.SEED_HASH] == seed.seed_hash(now["text"])  # type: ignore[index]


# --------------------------------------------------------------------------- fixes


def test_a_seed_text_change_raises_the_view_rev_so_an_open_editor_conflicts() -> None:
    """En-US at revision 1 (the view revision), es-US seeded at revision 0 with an older text. The
    reseed writes es-US at the key's view revision + 1, so an editor holding view revision 1 gets a 409."""
    store = seeded_store()
    api, store, _, _ = make_api(store)
    key = "greeting.generic"
    store.update("catalog", {"key": key, "lang": "en-US"}, {"rev": 1})
    as_if_seeded_with(store, key, "es-US", OLD_SEED_TEXT, rev=0)
    status, body = call(api, "GET", "/prompts", query={"prefix": "greeting."}, who=EDITOR)
    assert status == 200 and {p["key"]: p for p in body["prompts"]}[key]["rev"] == 1, body
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    assert entry(store, key, "es-US")["rev"] == 2  # type: ignore[index]
    texts = {"en-US": "Hello again.", "es-US": "Hola otra vez."}
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 1, "texts": texts}, who=EDITOR)
    assert status == 409, body
    assert entry(store, key, "es-US")["text"] != texts["es-US"]  # type: ignore[index]


def test_a_new_language_of_a_live_key_raises_its_view_rev() -> None:
    items = seed.PromptCatalog.to_items({
        "defaultLanguage": "en-US", "languages": {"en-US": {}, "es-US": {}},
        "entries": [{"key": "a.b", "lang": "en-US", "text": "Hi.", "version": 0},
                    {"key": "a.b", "lang": "es-US", "text": "Hola.", "version": 0}]})
    en = {"key": "a.b", "lang": "en-US", "text": "Hi.", "version": 0, "status": "published",
          "audio_url": None, "rev": 5, seed.SEED_HASH: seed.seed_hash("Hi.")}
    merge = seed.merge_catalog_prompts(items, {("a.b", "en-US"): en}, overwrite=False)
    [es] = [i for i in merge.items if (i["key"], i["lang"]) == ("a.b", "es-US")]
    assert es["rev"] == 6 and merge.new == [("a.b", "es-US")]


def test_a_console_prefixed_key_the_seed_updated_then_dropped_leaves_the_index() -> None:
    """The seed's update gives the item rev >= 1, so rev alone can't tell it from a
    console-added key; its seed_hash does."""
    def catalog(text: str | None) -> list[dict[str, Any]]:
        entries = [{"key": "greeting.a", "lang": "en-US", "text": "Hi.", "version": 0}]
        if text is not None:
            entries.append({"key": "option.card.d", "lang": "en-US", "text": text, "version": 0})
        return seed.PromptCatalog.to_items({"defaultLanguage": "en-US",
                                            "languages": {"en-US": {}}, "entries": entries})

    store = MemoryStore()
    for text in ("Debit.", "Debit card.", None):  # seed, update, drop
        merge = seed.merge_catalog_prompts(catalog(text), seed.read_live_catalog(store),
                                           overwrite=False)
        for item in merge.items:
            store.put("catalog", item)
    assert entry(store, "option.card.d", "en-US")["rev"] >= 1  # type: ignore[index]
    assert index(store) == ["greeting.a"] and merge.live_only == []
    # a console-created key (no seed_hash) under the same prefix stays
    store.put("catalog", {"key": "option.card.e", "lang": "en-US", "text": "Credit.", "version": 0,
                          "status": "published", "audio_url": None, "rev": 1})
    meta = store.get("catalog", {"key": "_meta", "lang": "_"})
    assert meta is not None
    store.put("catalog", {**meta, "keys": [*meta["keys"], "option.card.e"]})
    merge = seed.merge_catalog_prompts(catalog(None), seed.read_live_catalog(store), overwrite=False)
    assert merge.live_only == ["option.card.e"]
