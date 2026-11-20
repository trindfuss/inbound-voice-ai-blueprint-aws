"""Unit tests for owlbank.catalog."""

from __future__ import annotations

import collections
import copy
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from owlbank.catalog import CatalogError, CatalogKeyError, PromptCatalog, render
from owlbank.policy import Policy
from owlbank import store as store_mod
from owlbank.store import SCHEMAS, DynamoStore, MemoryStore, StoreError, create_table_kwargs
from tests.logutil import lines

DATA = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def seed() -> dict[str, Any]:
    return json.loads((DATA / "prompt_catalog.json").read_text(encoding="utf-8"))


@pytest.fixture()
def catalog(seed: dict[str, Any]) -> PromptCatalog:
    return PromptCatalog.from_dict(seed)


@pytest.fixture(scope="module")
def policy() -> Policy:
    return Policy.from_dict(json.loads((DATA / "policy.json").read_text(encoding="utf-8")))


def _entry(key: str, lang: str, text: str, **kw: Any) -> dict[str, Any]:
    return {"key": key, "lang": lang, "text": text, "version": 1, "status": "published",
            "audioUrl": None, **kw}


# --------------------------------------------------------------------------- handover examples


def test_stepup_intro_make_payment_en(catalog: PromptCatalog, policy: Policy) -> None:
    lang = "en-US"
    phrase = catalog.text(policy.phrase_key("make_payment"), lang)
    assert catalog.text("stepup.intro", lang, phrase=phrase) == (
        "I can help you make that payment. First I need to ask you a few more "
        "questions to verify your identity."
    )


def test_stepup_intro_make_payment_es(catalog: PromptCatalog, policy: Policy) -> None:
    lang = "es-US"
    phrase = catalog.text(policy.phrase_key("make_payment"), lang)
    assert catalog.text("stepup.intro", lang, phrase=phrase) == (
        "Puedo ayudarle a hacer ese pago. Primero necesito hacerle unas preguntas "
        "más para verificar su identidad."
    )


def test_greeting_personal(catalog: PromptCatalog) -> None:
    assert catalog.text("greeting.personal", "es-US", firstName="Lucía") == (
        "Hola, Lucía. Soy el agente virtual de OwlBank. ¿En qué puedo ayudarle hoy?"
    )


def test_language_select_single_language(catalog: PromptCatalog,
                                         caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="owlbank.catalog"):
        assert catalog.text("language.select", "en-US") == "Para español, oprima dos."
    assert caplog.text == ""


#: A system prompt per model.
AGENT_SYSTEM_KEYS = ("agent.system.haiku-5-5", "agent.system.haiku-4-5", "agent.system.sonnet-5-5")


@pytest.mark.parametrize("key", AGENT_SYSTEM_KEYS)
def test_agent_system_has_one_entry_per_language(catalog: PromptCatalog, key: str) -> None:
    """Agent prompt caching: the texts have no placeholders; the caller's data is in
    ``agent.caller_context``."""
    en, es = catalog.template(key, "en-US"), catalog.template(key, "es-US")
    assert en != es  # structure only: no phrase-matching on prompt text
    assert "{" not in en and "{" not in es


def test_the_caller_context_keeps_its_placeholders(catalog: PromptCatalog) -> None:
    for lang in ("en-US", "es-US"):
        text = catalog.text("agent.caller_context", lang, firstName="FN_X", profile="PR_X", memory="ME_X")
        assert all(v in text for v in ("FN_X", "PR_X", "ME_X")), lang


# --------------------------------------------------------------------------- placeholders


def test_unknown_placeholder_removed_cleanly() -> None:
    assert render("Sure, I can help you {phrase}. Next.", {}) == "Sure, I can help you. Next."
    assert render("Hi {a} there", {}) == "Hi there"
    assert render("{a}", {}) == ""


def test_values_are_not_reexpanded() -> None:
    assert render("Hello {name}.", {"name": "{secret} {0} {name.__class__}"}) == (
        "Hello {secret} {0} {name.__class__}."
    )


def test_format_syntax_is_not_interpreted() -> None:
    # Attribute access, indexing and format specs are not placeholders.
    tmpl = "x {a.b} {a[0]} {a!r} {a:>10} {0}"
    assert render(tmpl, {"a": "Z"}) == tmpl


def test_extra_values_ignored(catalog: PromptCatalog) -> None:
    assert catalog.text("common.goodbye", "en-US", firstName="X") == (
        "Thank you for calling OwlBank. Goodbye."
    )


def test_none_value_treated_as_missing() -> None:
    assert render("Hi {name}.", {"name": None}) == "Hi."


# --------------------------------------------------------------------------- fallback


def test_missing_language_falls_back_with_warning(seed: dict[str, Any],
                                                  caplog: pytest.LogCaptureFixture) -> None:
    doc = copy.deepcopy(seed)
    doc["entries"] = [e for e in doc["entries"]
                      if not (e["key"] == "common.goodbye" and e["lang"] == "es-US")]
    cat = PromptCatalog.from_dict(doc)
    with caplog.at_level(logging.WARNING, logger="owlbank.catalog"):
        assert cat.text("common.goodbye", "es-US") == "Thank you for calling OwlBank. Goodbye."
    assert "falling back" in caplog.text
    assert cat.missing_keys("es-US") == ["common.goodbye"]
    assert cat.missing_keys("en-US") == []


def test_unknown_language_falls_back(catalog: PromptCatalog) -> None:
    assert catalog.text("greeting.generic", "fr-FR") == "Welcome to OwlBank. This call may be recorded and transcribed for quality and training."
    assert catalog.settings("fr-FR") == catalog.settings("en-US")


def test_unknown_key_raises(catalog: PromptCatalog) -> None:
    with pytest.raises(CatalogKeyError):
        catalog.text("no.such.key", "en-US")


def test_seed_has_no_missing_keys(catalog: PromptCatalog) -> None:
    for lang in catalog.languages:
        assert catalog.missing_keys(lang) == []


# --------------------------------------------------------------------------- settings, commands, audio


def test_settings_per_language(catalog: PromptCatalog) -> None:
    es = catalog.settings("es-US")
    assert es["say"]["voice"] == "ElevenLabs.CaJslL1xziwefCeTNzHv"
    assert es["conversationRelay"]["transcriptionLanguage"] == "es"
    es["say"]["voice"] = "mutated"
    assert catalog.settings("es-US")["say"]["voice"] != "mutated"


def test_commands(catalog: PromptCatalog) -> None:
    assert catalog.commands("en-US") == {
        "agent": ["agent", "representative", "a person"],
        "cancel": ["cancel", "never mind", "stop"],
        "repeat": ["repeat", "say that again", "what"],
    }
    assert catalog.commands("es-US")["repeat"] == ["repetir", "repita", "otra vez", "cómo"]


def test_audio_url_and_status_filter() -> None:
    doc = {
        "defaultLanguage": "en-US",
        "languages": {"en-US": {}},
        "entries": [
            _entry("d.legal", "en-US", "v1", audioUrl="https://example.invalid/a.mp3"),
            _entry("d.legal", "en-US", "v3 draft", version=3, status="draft"),
            _entry("d.legal", "en-US", "v2", version=2),
        ],
    }
    cat = PromptCatalog.from_dict(doc)
    assert cat.text("d.legal", "en-US") == "v2"
    assert cat.audio_url("d.legal", "en-US") is None


@pytest.mark.parametrize(
    "doc",
    [
        {"defaultLanguage": "de-DE", "languages": {"en-US": {}}, "entries": []},
        {"defaultLanguage": "en-US", "languages": {}, "entries": []},
        {"defaultLanguage": "en-US", "languages": {"en-US": {}},
         "entries": [_entry("a", "xx-XX", "t")]},
        {"defaultLanguage": "en-US", "languages": {"en-US": {}}, "entries": [{"key": "a"}]},
    ],
)
def test_invalid_catalog_rejected(doc: dict[str, Any]) -> None:
    with pytest.raises(CatalogError):
        PromptCatalog.from_dict(doc)


# --------------------------------------------------------------------------- store and cache


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _seed_store(store: MemoryStore, doc: dict[str, Any]) -> None:
    for item in PromptCatalog.to_items(doc):
        store.put("catalog", item)


def test_store_round_trip(seed: dict[str, Any]) -> None:
    store = MemoryStore()
    _seed_store(store, seed)
    a, b = PromptCatalog.from_dict(seed), PromptCatalog.from_store(store)
    assert a.keys == b.keys and a.languages == b.languages
    for key in a.keys:
        for lang in a.languages:
            assert a.template(key, lang) == b.template(key, lang)
    assert b.settings("es-US") == a.settings("es-US")
    assert "audio_url" in store.get("catalog", {"key": "greeting.generic", "lang": "en-US"})


def test_store_cache_refresh(seed: dict[str, Any]) -> None:
    store, clock = MemoryStore(), Clock()
    _seed_store(store, seed)
    cat = PromptCatalog.from_store(store, ttl_s=60, clock=clock)
    store.update("catalog", {"key": "greeting.generic", "lang": "en-US"},
                 {"text": "Hello from OwlBank.", "version": 2})
    clock.t = 30
    assert cat.text("greeting.generic", "en-US") == "Welcome to OwlBank. This call may be recorded and transcribed for quality and training."
    clock.t = 60
    assert cat.text("greeting.generic", "en-US") == "Hello from OwlBank."


def test_store_unpublished_entry_not_served(seed: dict[str, Any]) -> None:
    store, clock = MemoryStore(), Clock()
    _seed_store(store, seed)
    cat = PromptCatalog.from_store(store, ttl_s=60, clock=clock)
    store.update("catalog", {"key": "greeting.generic", "lang": "es-US"}, {"status": "draft"})
    clock.t = 61
    assert cat.text("greeting.generic", "es-US") == "Welcome to OwlBank. This call may be recorded and transcribed for quality and training."
    assert cat.missing_keys("es-US") == ["greeting.generic"]


# --------------------------------------------------------------------------- reload in a few calls


@pytest.fixture
def dynamo(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Any, collections.Counter[str]]]:
    """A moto DynamoStore and a counter of the DynamoDB API calls it makes."""
    import boto3
    from moto import mock_aws

    for var, val in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
                     "AWS_SESSION_TOKEN": "testing", "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(var, val)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    calls: collections.Counter[str] = collections.Counter()
    with mock_aws():
        resource = boto3.resource("dynamodb", region_name="us-east-1")
        names = {t: f"owlbank-{t}" for t in SCHEMAS}
        for logical, physical in names.items():
            resource.meta.client.create_table(**create_table_kwargs(logical, physical))
        resource.meta.client.meta.events.register(
            "before-call.dynamodb", lambda model, **_: calls.update([model.name]))
        yield DynamoStore(names, resource=resource), calls


def test_reload_is_one_scan(seed: dict[str, Any], dynamo: Any) -> None:
    store, calls = dynamo
    _seed_store(store, seed)
    calls.clear()
    cat = PromptCatalog.from_store(store)
    assert calls == {"Scan": 1}
    a = PromptCatalog.from_dict(seed)
    assert cat.keys == a.keys
    assert all(cat.template(k, lang) == a.template(k, lang) for k in a.keys for lang in a.languages)


def test_reload_follows_scan_pages(seed: dict[str, Any], dynamo: Any,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    store, calls = dynamo
    _seed_store(store, seed)
    table = store._table("catalog")
    real = table.scan

    def small_pages(**kw: Any) -> Any:
        return real(Limit=50, **kw)

    monkeypatch.setattr(table, "scan", small_pages)
    calls.clear()
    cat = PromptCatalog.from_store(store)
    assert calls["Scan"] == -(-len(PromptCatalog.to_items(seed)) // 50) and set(calls) == {"Scan"}
    assert cat.keys == PromptCatalog.from_dict(seed).keys


class _BrokenScan(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.broken = False

    def scan(self, table: str) -> list[dict[str, Any]]:
        if self.broken:
            raise StoreError("throttled")
        return super().scan(table)


def test_a_store_error_keeps_the_last_good_catalog(seed: dict[str, Any],
                                                   caplog: pytest.LogCaptureFixture) -> None:
    store, clock = _BrokenScan(), Clock()
    _seed_store(store, seed)
    cat = PromptCatalog.from_store(store, ttl_s=60, clock=clock)
    store.update("catalog", {"key": "greeting.generic", "lang": "en-US"}, {"text": "Hi.", "version": 9})
    store.broken = True
    clock.t = 61
    with caplog.at_level(logging.ERROR, logger="owlbank.catalog"):
        assert cat.text("greeting.generic", "en-US") == "Welcome to OwlBank. This call may be recorded and transcribed for quality and training."
    assert "keeping the last good catalog" in caplog.text
    store.broken = False
    clock.t = 122
    assert cat.text("greeting.generic", "en-US") == "Hi."


def test_items_no_key_names_are_ignored(seed: dict[str, Any]) -> None:
    store = MemoryStore()
    _seed_store(store, seed)
    store.put("catalog", {"key": "orphan", "lang": "en-US", "text": "x", "status": "published"})
    assert "orphan" not in PromptCatalog.from_store(store).keys


def test_reload_logs_one_dependency_call(seed: dict[str, Any], caplog: pytest.LogCaptureFixture,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    store = MemoryStore()
    _seed_store(store, seed)
    caplog.set_level(logging.DEBUG, logger="owlbank.dependency")
    PromptCatalog.from_store(store)
    (line,) = [f for f in lines(caplog.records, "owlbank.dependency") if f["operation"] == "reload:catalog"]
    assert line["event"] == "dependency_call" and line["dependency"] == "dynamodb"
    assert line["outcome"] == "ok" and line["items"] == len(PromptCatalog.to_items(seed))
    assert [r.levelno for r in caplog.records if getattr(r, "operation", "") == "reload:catalog"] == [logging.INFO]
    caplog.clear()
    monkeypatch.setattr(store_mod, "SLOW_CALL_MS", -1)
    PromptCatalog.from_store(store)
    (slow,) = [r for r in caplog.records if getattr(r, "operation", "") == "reload:catalog"]
    assert slow.levelno == logging.WARNING and lines([slow], "owlbank.dependency")[0]["outcome"] == "slow"


def test_a_failed_reload_logs_its_error(caplog: pytest.LogCaptureFixture) -> None:
    store = _BrokenScan()
    store.broken = True
    caplog.set_level(logging.DEBUG, logger="owlbank.dependency")
    with pytest.raises(StoreError):
        PromptCatalog.from_store(store)
    (line,) = [f for f in lines(caplog.records, "owlbank.dependency") if f["operation"] == "reload:catalog"]
    assert line["outcome"] == "error" and line["error_class"] == "StoreError" and "duration_ms" in line
