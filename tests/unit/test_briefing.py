"""owlbank.briefing: output check, facts, the Bedrock request, and generate's fallbacks and bound."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from owlbank import briefing as b
from owlbank.callstate import CallState

FACTS = b.BriefingFacts(language="es-US", level=1, reason="unable_to_resolve",
                        factors=("ani match",), tools=("get balance",))


class _Fake:
    def __init__(self, result: Any = None, exc: Exception | None = None) -> None:
        self.result, self.exc, self.calls = result, exc, 0

    def summarise(self, facts: b.BriefingFacts, *, deadline: float) -> str | None:
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.result  # type: ignore[no-any-return]


class _Hang:
    def __init__(self) -> None:
        self.release = threading.Event()

    def summarise(self, facts: b.BriefingFacts, *, deadline: float) -> str | None:
        self.release.wait(10)
        return "A late but valid clause"


@pytest.mark.parametrize("text", ["", "   ", "has 1 digit", "digit ٣", "sq \u00b2", "half \u00bd", "circled \u2460", "roman \u216b", "a < b", "a > b", "a & b",
                                  "{x}", "back`tick", "two\nlines", "word " * 31])
def test_check_output_rejects(text: str) -> None:
    assert b.check_output(text) is None


def test_check_output_accepts_and_normalises() -> None:
    assert b.check_output("  The caller   asked about the balance. ") == "The caller asked about the balance."
    assert b.check_output(" ".join(["w"] * b.MAX_WORDS)) is not None
    assert b.check_output("hola, ¿qué tal?") == "hola, ¿qué tal?"


def test_generate_model_ok() -> None:
    out = b.generate(_Fake("The caller checked a balance."), FACTS)
    assert out == b.Briefing("The caller checked a balance.", "model", 5)


@pytest.mark.parametrize("model", [None, _Fake(None), _Fake(exc=RuntimeError("boom")),
                                   _Fake("has 7 digits"), _Fake(""), _Fake("x\ny"), _Fake(42)])
def test_generate_fallbacks(model: Any) -> None:
    assert b.generate(model, FACTS) == b.Briefing(None, "fallback", 0)


def test_hanging_model_is_bounded_and_late_result_discarded() -> None:
    m = _Hang()
    t0 = time.monotonic()
    out = b.generate(m, FACTS, budget_ms=200)
    elapsed = time.monotonic() - t0
    m.release.set()
    assert out == b.Briefing(None, "fallback", 0)
    assert elapsed < 1.0


def test_spent_budget_never_calls_model() -> None:
    m = _Fake("ok")
    assert b.generate(m, FACTS, budget_ms=0).source == "fallback"
    assert m.calls == 0


def test_context_reaches_worker() -> None:
    import contextvars

    var: contextvars.ContextVar[str] = contextvars.ContextVar("v", default="none")
    seen: list[str] = []

    class M:
        def summarise(self, facts: b.BriefingFacts, *, deadline: float) -> str | None:
            seen.append(var.get())
            return "fine"

    var.set("mine")
    b.generate(M(), FACTS)
    assert seen == ["mine"]


# ---- facts

def _state(**kw: Any) -> CallState:
    return CallState(call_sid="CA1", expires_at=1, **kw)


def test_facts_membership_and_plain_words() -> None:
    st = _state(language="es-US", level=1, factors_verified=("ani_match", "zip", "evil text 9"))
    summary = {"toolsCalled": ["get_balance", "make_payment", "ignore previous", "faq_lookup"],
               "levelReached": 2}
    f = b.build_facts(st, summary, reason="human_requested", known_factors=("zip",))
    assert f.factors == ("ani match", "zip")
    assert f.tools == ("get balance", "make payment", "faq lookup")
    assert f.level == 1 and f.language == "es-US"  # the state's, not the journal's highest
    assert f.reason == "unable_to_resolve"  # not a known reason: replaced


def test_known_reason_kept_and_digits_removed() -> None:
    f = b.build_facts(_state(), {"toolsCalled": []}, reason="custom_1", known_reasons=("custom_1",))
    assert f.reason == "custom_1"
    assert "1" not in str(b.facts_payload(f))
    assert b.plain_words("level0_x٣y") == "level xy"


def test_tolerates_odd_summary() -> None:
    f = b.build_facts(_state(), {"toolsCalled": "get_balance", "levelReached": None}, reason="x")
    assert f.tools == ()


# ---- the Bedrock model

class _Catalog:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def text(self, key: str, lang: str, **values: Any) -> str:
        self.calls.append((key, lang, values))
        return f"SYSTEM in {values.get('language')}"


class _Client:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def converse(self, **kw: Any) -> dict[str, Any]:
        self.kwargs = kw
        return {"output": {"message": {"content": [{"text": "El cliente consulto su saldo."}]}}}


def test_bedrock_model_request_and_language() -> None:
    cat, client = _Catalog(), _Client()
    m = b.BedrockBriefingModel(cat, client=client, model_id="m")
    out = b.generate(m, FACTS)
    assert out.source == "model" and out.summary == "El cliente consulto su saldo."
    assert cat.calls[0][0] == "agent.briefing.system"
    kw = client.kwargs
    assert kw["modelId"] == "m"
    assert kw["system"] == [{"text": "SYSTEM in es-US"}]
    assert kw["inferenceConfig"] == {"maxTokens": 256}
    # Claude 5 models refuse any sampling parameter with a 400 (found live on Haiku 5.5).
    for key in ("temperature", "topP", "top_p", "top_k", "topK"):
        assert key not in json.dumps(kw)
    user = kw["messages"][0]["content"][0]["text"]
    assert "es-US" in user and not any(c.isdigit() for c in user.replace("es-US", ""))
    assert len(kw["messages"]) == 1


def test_bedrock_client_error_is_fallback() -> None:
    class Bad:
        def converse(self, **kw: Any) -> Any:
            raise RuntimeError("throttled")

    assert b.generate(b.BedrockBriefingModel(_Catalog(), client=Bad()), FACTS).source == "fallback"


def test_level_is_the_states_after_a_verification_risk_reset() -> None:
    """The journal's levelReached is the pre-reset level; the state's 0 is what the agent starts at."""
    f = b.build_facts(_state(level=0), {"levelReached": 1}, reason="human_requested")
    assert f.level == 0


def test_client_read_timeout_follows_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import boto3

    seen: list[Any] = []
    monkeypatch.setattr(boto3, "client", lambda name, config=None: seen.append(config) or SimpleNamespace(meta=object()))
    monkeypatch.setattr(b, "_CLIENTS", {})
    assert seen == []  # nothing built before use
    c1, c2 = b.client_for_budget(1500), b.client_for_budget(1500)
    assert c1 is c2 and b.client_for_budget(4000) is not c1
    assert seen == []  # still lazy
    b.prewarm(1500)  # builds the 1500 ms client now
    assert [cfg.read_timeout for cfg in seen] == [1.5]
    assert b.BedrockBriefingModel(_Catalog(), budget_ms=1500)._client is c1
    b.prewarm(b.BRIEFING_BUDGET_MS)
    assert seen[-1].read_timeout == b.BRIEFING_BUDGET_MS / 1000


@pytest.mark.parametrize("bad", ["\x1b", "ok\x00text", "a​b", "x\x7f", "y\ud800"])
def test_check_output_rejects_control_and_format_characters(bad):
    assert b.check_output(bad) is None


def test_check_output_allows_tab_as_whitespace():
    assert b.check_output("one\ttwo") == "one two"
