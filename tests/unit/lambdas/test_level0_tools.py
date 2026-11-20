"""Unit tests for the level-0 tools Lambda (AgentCore Gateway target)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lambdas.level0_tools import handler as l0_handler
from lambdas.level0_tools import service

DATA = Path(__file__).resolve().parents[3] / "data"
FIXTURES = json.loads((DATA / "fixtures.json").read_text(encoding="utf-8"))


def ctx(tool: str | None) -> Any:
    custom = {} if tool is None else {"bedrockAgentCoreToolName": tool}
    return SimpleNamespace(client_context=SimpleNamespace(custom=custom))


@pytest.fixture
def content() -> dict[str, Any]:
    return service.load_content()


@pytest.mark.parametrize(("question", "lang", "faq_id"), [
    ("I lost my credit card, what do I do?", "en-US", "lost_card"),
    ("How do I deposit a check with the app?", "en-US", "mobile_deposit"),
    ("I'd like to open an account", "en-US", "open_account"),
    ("wire transfer cutoff", "en-US", "wire_cutoff"),
    ("Perdí mi tarjeta de crédito", "es-US", "lost_card"),
    ("¿Cómo deposito un cheque?", "es-US", "mobile_deposit"),
    ("quiero abrir una cuenta nueva", "es-US", "open_account"),
])
def test_faq_lookup_matches(content: dict[str, Any], question: str, lang: str, faq_id: str) -> None:
    out = service.faq_lookup(content, question=question, lang=lang)
    assert out["found"] is True and out["id"] == faq_id and out["lang"] == lang
    faq = next(f for f in content["faqs"] if f["id"] == faq_id)
    assert out["answer"] == faq["answer"][lang]


@pytest.mark.parametrize("question", ["", "   ", "the a of", "tell me a joke about penguins"])
def test_faq_lookup_no_match(content: dict[str, Any], question: str) -> None:
    assert service.faq_lookup(content, question=question, lang="en-US") == {"found": False}


def test_faq_lookup_unknown_lang_falls_back(content: dict[str, Any]) -> None:
    out = service.faq_lookup(content, question="lost card", lang="fr-FR")
    assert out["lang"] == "en-US" and out["id"] == "lost_card"


def test_faq_spanish_question_english_keywords(content: dict[str, Any]) -> None:
    """Default-language keywords also count, so a mixed-language question still matches."""
    out = service.faq_lookup(content, question="wire transfer", lang="es-US")
    assert out["id"] == "wire_cutoff" and out["answer"].startswith("Las transferencias")


def test_branch_hours_all_and_one(content: dict[str, Any]) -> None:
    every = service.get_branch_hours(content, lang="en-US")
    assert every["found"] is True and [b["id"] for b in every["branches"]] == ["downtown", "riverside"]
    assert every["phoneSupportHours"].startswith("Our phone specialists")
    one = service.get_branch_hours(content, branch="Riverside branch", lang="es-US")
    assert [b["id"] for b in one["branches"]] == ["riverside"]
    assert one["branches"][0]["hours"].startswith("De lunes a viernes, de 10")
    assert service.get_branch_hours(content, branch="Centro", lang="es-US")["branches"][0]["id"] \
        == "downtown"
    missing = service.get_branch_hours(content, branch="Mars", lang="en-US")
    assert missing["found"] is False and len(missing["branches"]) == 2


@pytest.mark.parametrize(("tool", "event", "check"), [
    ("level0___faq_lookup", {"question": "lost my card", "lang": "en-US"},
     lambda o: o["id"] == "lost_card"),
    ("faq_lookup", {"question": "lost my card"}, lambda o: o["id"] == "lost_card"),
    ("level0___get_branch_hours", {"branch": "downtown"},
     lambda o: o["branches"][0]["name"] == "OwlBank Downtown"),
    ("level0___get_branch_hours", {}, lambda o: len(o["branches"]) == 2),
    ("level0___get_balance", {"account": "checking"}, lambda o: o == {"error": "UNKNOWN_TOOL"}),
    ("level0___faq_lookup", {"question": 5}, lambda o: o == {"error": "INVALID_ARGUMENT"}),
    ("level0___get_branch_hours", {"branch": ["x"]}, lambda o: o == {"error": "INVALID_ARGUMENT"}),
])
def test_gateway_handler(tool: str, event: dict[str, Any], check: Any) -> None:
    assert check(l0_handler.handler(event, ctx(tool)))


def test_handler_without_tool_name() -> None:
    assert l0_handler.handler({}, ctx(None)) == {"error": "UNKNOWN_TOOL"}
    assert l0_handler.handler({}, None) == {"error": "UNKNOWN_TOOL"}
    assert l0_handler.handler({"question": ""}, SimpleNamespace(client_context={"custom": {
        "bedrockAgentCoreToolName": "t___faq_lookup"}}))["found"] is False


def test_handler_bad_content_path_is_unavailable(monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path) -> None:
    monkeypatch.setenv("OWLBANK_FAQ_PATH", str(tmp_path / "missing.json"))
    assert l0_handler.handler({"question": "x"}, ctx("t___faq_lookup")) == {"error": "UNAVAILABLE"}


def test_no_customer_data_in_any_output(content: dict[str, Any]) -> None:
    outputs = [service.get_branch_hours(content, lang=lang) for lang in service.SUPPORTED_LANGS]
    outputs += [service.faq_lookup(content, question=f["question"][lang], lang=lang)
                for f in content["faqs"] for lang in service.SUPPORTED_LANGS]
    text = json.dumps(outputs, ensure_ascii=False)
    for c in FIXTURES["customers"]:
        for value in (c["customer_id"], c["phone"], c["last_name"]):
            assert value not in text
        for a in c["accounts"]:
            assert a["last4"] not in text
