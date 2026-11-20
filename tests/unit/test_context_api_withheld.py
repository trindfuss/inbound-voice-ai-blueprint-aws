"""Model-assisted slots: a withheld screen-pop timeline prompt shows the catalog's
``escalation.summary_withheld`` in the call's language, read through ``PromptCatalog`` with its
cache. Any catalog error gives ``[WITHHELD]`` (``pii.WITHHELD_TURN``), logged with no text.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from lambdas.context_api import handler as ctx
from lambdas.context_api.handler import ALLOWED_ORIGIN, CatalogWithheld, get_context
from owlbank import callstate
from owlbank.callstate import CallState
from owlbank.catalog import PromptCatalog
from owlbank.pii import WITHHELD_TURN, FakePiiRedactor, PiiGuard, RedactionConfig
from owlbank.segments import SegmentLog
from owlbank.store import MemoryStore

CATALOG = json.loads((Path(__file__).resolve().parents[2] / "data" / "prompt_catalog.json")
                     .read_text(encoding="utf-8"))
EN = "Summary withheld: PII check unavailable."
ES = "Resumen retenido: la verificación de datos personales no está disponible."
SID = "CA" + "b" * 32
REF = "jr_f5"
NOW = 1_760_108_645
FAR = 4_000_000_000
ON = RedactionConfig(entity_types=("NAME",))


def _seed_catalog(store: MemoryStore, doc: dict[str, Any] | None = None) -> None:
    for item in PromptCatalog.to_items(doc or CATALOG):
        store.put("catalog", item)


class CountingStore(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.catalog_reads = 0
        self.fail = False

    def _catalog_read(self, table: str) -> None:
        if table == "catalog":
            self.catalog_reads += 1
            if self.fail:
                raise RuntimeError("catalog read failed")

    def get(self, table: str, key: Any) -> Any:
        self._catalog_read(table)
        return super().get(table, key)

    def scan(self, table: str) -> Any:
        # The catalog reload is one Scan.
        self._catalog_read(table)
        return super().scan(table)


@pytest.fixture
def store() -> CountingStore:
    s = CountingStore()
    _seed_catalog(s)
    return s


def test_seed_texts_are_what_the_tests_expect() -> None:
    cat = PromptCatalog.from_dict(CATALOG)
    assert cat.text("escalation.summary_withheld", "en-US") == EN
    assert cat.text("escalation.summary_withheld", "es-US") == ES


@pytest.mark.parametrize(("lang", "text"), [("en-US", EN), ("es-US", ES)])
def test_withheld_is_the_catalog_text_in_the_language(store: CountingStore, lang: str,
                                                      text: str) -> None:
    assert CatalogWithheld(store, ttl_s=60)(lang) == text


def test_catalog_error_gives_withheld_turn_and_logs_no_text(
        store: CountingStore, caplog: pytest.LogCaptureFixture) -> None:
    store.fail = True
    with caplog.at_level(logging.DEBUG):
        assert CatalogWithheld(store, ttl_s=60)("es-US") == WITHHELD_TURN
    assert any(getattr(r, "operation", None) == "withheld_text" for r in caplog.records)
    assert ES not in caplog.text and EN not in caplog.text


def test_missing_catalog_gives_withheld_turn() -> None:
    assert CatalogWithheld(MemoryStore(), ttl_s=60)("en-US") == WITHHELD_TURN


def test_missing_key_gives_withheld_turn() -> None:
    s = MemoryStore()
    doc = dict(CATALOG, entries=[e for e in CATALOG["entries"]
                                 if e["key"] != "escalation.summary_withheld"])
    _seed_catalog(s, doc)
    assert CatalogWithheld(s, ttl_s=60)("en-US") == WITHHELD_TURN


def test_a_failed_first_read_is_retried(store: CountingStore) -> None:
    reader = CatalogWithheld(store, ttl_s=60)
    store.fail = True
    assert reader("en-US") == WITHHELD_TURN
    store.fail = False
    assert reader("en-US") == EN


def test_the_catalog_is_cached_within_the_ttl(store: CountingStore) -> None:
    t = [0.0]
    reader = CatalogWithheld(store, ttl_s=60, clock=lambda: t[0])
    assert reader("en-US") == EN
    gets = store.catalog_reads
    t[0] = 30.0
    assert reader("es-US") == ES
    assert store.catalog_reads == gets
    t[0] = 61.0
    assert reader("en-US") == EN
    assert store.catalog_reads > gets


# ------------------------------------------------------------------------------- wiring


def _call(store: MemoryStore, language: str) -> None:
    callstate.create(store, CallState(call_sid=SID, expires_at=FAR, language=language,
                                      escalation={"reason": "complaint", "journal_ref": REF}))
    SegmentLog(store, now=lambda: NOW).append(
        SID, {"type": "prompt", "prompt_key": "greeting", "rendered_text": "Hello."})


def _request() -> dict:
    return {"httpMethod": "GET", "headers": {"Origin": ALLOWED_ORIGIN},
            "pathParameters": {"journal_ref": REF}, "queryStringParameters": {"call_sid": SID, "task_sid": "WT" + "4" * 32},
            "requestContext": {"authorizer": {"worker_sid": "WK" + "3" * 32, "roles": "supervisor"}}}


class _TaskRouterOk:
    """TaskRouter says the task is this call's (the per-task check has its own tests)."""

    def task_attributes(self, task_sid: str, *, timeout_s=None) -> dict:
        return {"call_sid": SID, "journal_ref": REF}

    def reservations(self, task_sid: str, *, timeout_s=None) -> list:
        return []


def _texts(resp: dict) -> list[str]:
    return [e["rendered_text"] for e in json.loads(resp["body"])["timeline"]
            if "rendered_text" in e]


@pytest.mark.parametrize(("lang", "text"), [("en-US", EN), ("es-US", ES)])
def test_handler_withholds_with_the_catalog_text(store: CountingStore, lang: str, text: str,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    _call(store, lang)
    monkeypatch.setattr(ctx, "_store", store)
    monkeypatch.setattr(ctx, "_pii", None)
    monkeypatch.setattr(ctx, "_withheld", None)
    monkeypatch.setattr(ctx, "_redactor", lambda: FakePiiRedactor(fail=True))
    monkeypatch.setattr(ctx, "_default_taskrouter", _TaskRouterOk)
    resp = ctx.handler(_request(), None)
    assert resp["statusCode"] == 200
    assert _texts(resp) == [text]
    assert ctx.withheld(lang) == text


def test_handler_catalog_error_gives_withheld_turn(store: CountingStore,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    _call(store, "es-US")
    store.fail = True
    monkeypatch.setattr(ctx, "_store", store)
    monkeypatch.setattr(ctx, "_pii", None)
    monkeypatch.setattr(ctx, "_withheld", None)
    monkeypatch.setattr(ctx, "_redactor", lambda: FakePiiRedactor(fail=True))
    monkeypatch.setattr(ctx, "_default_taskrouter", _TaskRouterOk)
    resp = ctx.handler(_request(), None)
    assert resp["statusCode"] == 200
    assert _texts(resp) == [WITHHELD_TURN]
    assert ctx.withheld("es-US") == WITHHELD_TURN


def test_withheld_store_setup_error_gives_withheld_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> MemoryStore:
        raise RuntimeError("no settings")

    monkeypatch.setattr(ctx, "_default_store", broken)
    assert ctx.withheld("en-US") == WITHHELD_TURN


def test_nothing_withheld_reads_no_catalog(store: CountingStore) -> None:
    _call(store, "en-US")
    reader = CatalogWithheld(store, ttl_s=60)
    resp = get_context(_request(), store, now=lambda: NOW, pii=PiiGuard(FakePiiRedactor(), ON),
                       withheld=reader, taskrouter=_TaskRouterOk())
    assert _texts(resp) == ["Hello."]
    assert store.catalog_reads == 0
