"""scripts/bench/slot_model_bench.py: the slot model bench, offline.

No AWS: the resolver adapter runs over a fake ``bedrock-runtime`` client, and the logging check reads
canned ``GetModelInvocationLoggingConfiguration`` responses. The classifier adapter and its tests went
with the classifier; adds the resolver's, per mode.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import pytest

from owlbank.slots.types import SLOT_TYPES, SlotDecl
from scripts.bench import slot_model_bench as smb

# ----------------------------------------------------------------------------- fakes


class FakeAdapter:
    """Scripted adapter: each call returns the next (latency_ms, in, out, kind) or raises."""

    name = "fake"

    def __init__(self, script: list[tuple[float, int, int, str] | Exception]) -> None:
        self.script = list(script)
        self.seen: list[str] = []

    def describe(self) -> dict[str, Any]:
        return {"adapter": self.name, "model_id": "fake-model"}

    def call(self, case: smb.Case) -> smb.Sample:
        self.seen.append(case.id)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            return smb.Sample.failed(case, latency_ms=1.0, error=type(step).__name__)
        lat, tin, tout, kind = step
        return smb.Sample.of(case, latency_ms=lat, input_tokens=tin, output_tokens=tout,
                             server_latency_ms=None, kind=kind, value=case.expect_value)


PRICE = smb.Price(model_id="fake-model", input_per_mtok=1.0, output_per_mtok=5.0, currency="USD",
                  source="test", effective="2026-10-01", looked_up="2026-10-19")


def case(**kw: Any) -> smb.Case:
    base: dict[str, Any] = {"id": "zip-en-clean", "slot_type": "digits", "lang": "en-US", "category": "clean",
                            "question": "What is your ZIP code?", "utterance": "it's nine four one oh five",
                            "constraints": {"length": 5}, "label": "ZIP code", "expect_kind": "value",
                            "expect_value": "94105"}
    base.update(kw)
    return smb.Case(**base)


# ----------------------------------------------------------------------------- the utterance sets


def test_cases_cover_every_slot_type_language_and_category():
    seen = {(c.slot_type, c.lang) for c in smb.CASES}
    assert seen == {(t, lang) for t in SLOT_TYPES for lang in smb.LANGS}
    for lang in smb.LANGS:
        assert {c.category for c in smb.CASES if c.lang == lang} == set(smb.CATEGORIES)
    assert all(c.category == "choice" for c in smb.CASES if c.slot_type == "choice")


def test_case_ids_are_unique_and_case_fields_valid():
    ids = [c.id for c in smb.CASES]
    assert len(ids) == len(set(ids))
    for c in smb.CASES:
        SlotDecl(slot_type=c.slot_type, constraints=dict(c.constraints))  # the declaration builds
        assert c.lang in smb.LANGS and c.category in smb.CATEGORIES
        assert c.expect_kind in (None, *smb.KINDS)
        assert c.expect_value is None or c.expect_kind == "value"
        if c.slot_type == "choice":
            assert c.options and c.expect_value in (None, *(v for v, _ in c.options))


def test_cases_hold_only_fictional_values():
    """Fictional fixtures only: no long digit runs (phone or account numbers), no e-mail address.
    Phase 2: a sensitive slot's words reach the resolver unredacted, so its cases hold
    fictional digits as words, and are scored like any other (the ``[digits]`` placeholder is gone)."""
    for c in smb.CASES:
        for text in (c.utterance, c.question):
            assert not re.search(r"\d{6,}", text), c.id
            assert "@" not in text, c.id
        assert "[digits]" not in c.utterance, c.id
        assert c.expect_kind is not None, c.id
        if c.sensitive:
            assert not re.search(r"\d", c.utterance) and c.expect_value, c.id


def test_select_filters_cases():
    picked = smb.select(smb.CASES, slot_types=["yes_no"], langs=["es-US"], categories=None)
    assert picked and all(c.slot_type == "yes_no" and c.lang == "es-US" for c in picked)
    with pytest.raises(ValueError, match="no case"):
        smb.select(smb.CASES, slot_types=["digits"], langs=["en-US"], categories=["choice"])


# ----------------------------------------------------------------------------- statistics and cost


def test_pct_nearest_rank():
    xs = [float(x) for x in range(10, 0, -1)]
    assert smb.pct(xs, 0.5) == 5.0
    assert smb.pct(xs, 0.9) == 9.0
    assert smb.pct(xs, 1.0) == 10.0
    assert smb.pct(xs, 0.0) == 1.0
    assert smb.pct([3.0], 0.9) == 3.0
    assert smb.pct([], 0.5) is None


def test_price_per_call_and_per_thousand():
    assert PRICE.cost(1000, 200) == pytest.approx(0.001 + 0.001)
    assert smb.cost_per_1000([(1000, 200), (3000, 600)], PRICE) == pytest.approx(1000 * (0.002 + 0.006) / 2)
    assert smb.cost_per_1000([], PRICE) is None


def test_price_table_records_its_source():
    p = smb.PRICES[smb.DEFAULT_MODEL_ID]
    assert p.input_per_mtok > 0 and p.output_per_mtok > 0
    rec = p.record()
    assert rec["source"] and rec["looked_up"] and rec["effective"] and rec["currency"] == "USD"
    with pytest.raises(KeyError):
        smb.price_for("no-such-model")




# ----------------------------------------------------------------------------- running and the report


def test_run_and_summary_shape():
    cases = [case(id="a"), case(id="b", category="messy"), case(id="c", lang="es-US")]
    script: list[Any] = [(100.0, 1000, 200, "value"), (300.0, 2000, 100, "unclear"), RuntimeError("x"),
                         (200.0, 1000, 200, "value"), (500.0, 1000, 200, "value"), (400.0, 1000, 200, "value")]
    fake = FakeAdapter(script)
    samples = smb.run(fake, cases, repeats=2)
    assert fake.seen == ["a", "b", "c", "a", "b", "c"]
    assert len(samples) == 6 and sum(1 for s in samples if s.error) == 1
    summary = smb.summarize(samples, PRICE)
    o = summary["overall"]
    assert o["answers"] == 6 and o["model_calls"] == 6 and o["errors"] == 1
    assert o["latency_p50_ms"] == 300.0 and o["latency_p90_ms"] == 500.0  # nearest rank over 5 samples
    assert o["input_tokens_mean"] == 1200.0 and o["output_tokens_mean"] == 180.0
    assert o["cost_per_1000_usd"] == pytest.approx(1000 * (1200 * 1e-6 + 180 * 5e-6))
    assert o["kinds"] == {"value": 4, "unclear": 1}
    assert o["correct_rate"] == pytest.approx(4 / 5)  # case() expects "value"; the error isn't scored
    assert set(summary) == {"overall", "by_slot_type", "by_lang", "by_category"}
    assert summary["by_lang"]["es-US"]["answers"] == 2 and summary["by_category"]["messy"]["answers"] == 2


def test_summary_of_errors_only_has_no_numbers():
    s = smb.summarize([smb.Sample.failed(case(), latency_ms=5.0, error="Boom")], PRICE)["overall"]
    assert s["errors"] == 1 and s["latency_p50_ms"] is None and s["cost_per_1000_usd"] is None
    assert s["correct_rate"] is None


def test_answers_without_a_model_call_have_no_latency_or_cost():
    """Deterministic answers (0 calls) count as answers, not in the latency, tokens or cost."""
    det = smb.Sample.of(case(), latency_ms=1.0, input_tokens=None, output_tokens=None,
                        server_latency_ms=None, kind="value", value="94105", model_calls=0,
                        model="not_called", mode="deterministic")
    o = smb.summarize([det, det], PRICE)["overall"]
    assert o["answers"] == 2 and o["model_calls"] == 0 and o["correct_rate"] == 1.0
    assert o["latency_p90_ms"] is None and o["cost_per_1000_usd"] is None and o["model_outcomes"] == {}


def test_a_timed_out_call_counts_in_the_latency_but_not_the_tokens():
    to = smb.Sample.of(case(), latency_ms=1500.0, input_tokens=None, output_tokens=None,
                       server_latency_ms=None, kind="unclear", value=None, model="timeout")
    ok = smb.Sample.of(case(), latency_ms=300.0, input_tokens=1000, output_tokens=100,
                       server_latency_ms=250, kind="value", value="94105", model="ok")
    o = smb.summarize([to, ok], PRICE)["overall"]
    assert o["latency_p90_ms"] == 1500.0 and o["input_tokens_mean"] == 1000.0
    assert o["model_outcomes"] == {"timeout": 1, "ok": 1}


# ----------------------------------------------------------------------------- the resolver adapter


def tool_response(name: str, **inp: Any) -> dict[str, Any]:
    body = {"kind": "none", "value": None, "candidates": [], "slot": None, "reply": None}
    if name.endswith("_generative"):
        body.update(say=None, say_lang=None)
    body.update(inp)
    return {"stopReason": "tool_use",
            "output": {"message": {"content": [{"toolUse": {"toolUseId": "t1", "name": name, "input": body}}]}},
            "usage": {"inputTokens": 812, "outputTokens": 47}, "metrics": {"latencyMs": 380}}


class FakeClient:
    """A ``bedrock-runtime`` stand-in: ``answer(kwargs)`` gives the tool input (or raises)."""

    def __init__(self, answer: Any = None) -> None:
        self.answer = answer or (lambda kw: {})
        self.requests: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        name = kwargs["toolConfig"]["tools"][0]["toolSpec"]["name"]
        return tool_response(name, **self.answer(kwargs))


def adapter(mode: str, client: FakeClient | None = None, **kw: Any) -> smb.ResolverAdapter:
    budgets: list[int] = []

    def client_for(budget_ms: int) -> Any:
        if client is None:
            raise AssertionError("no client may be made")
        budgets.append(budget_ms)
        return client

    a = smb.ResolverAdapter(client_for, model_id=smb.DEFAULT_MODEL_ID, mode=mode, **kw)
    a.budgets = budgets  # type: ignore[attr-defined]
    return a


def user_text(kwargs: dict[str, Any]) -> str:
    return kwargs["messages"][0]["content"][0]["text"]


def test_deterministic_mode_makes_no_call_and_reports_zero():
    a = adapter("deterministic")  # client_for raises if called
    samples = smb.run(a, list(smb.CASES), repeats=1)
    o = smb.summarize(samples, PRICE)["overall"]
    assert o["model_calls"] == 0 and o["answers"] == len(smb.CASES) and o["errors"] == 0
    assert all(s.model == "not_called" and s.effective_mode == "deterministic" for s in samples)
    assert a.describe() == {"adapter": "resolver", "mode": "deterministic", "model_id": smb.DEFAULT_MODEL_ID,
                            "budget_ms": 2500, "max_tokens": None}


def test_assisted_resolves_through_the_model_with_tokens_and_grounding():
    client = FakeClient(lambda kw: {"kind": "inferred", "value": "red"})
    a = adapter("assisted", client)
    s = a.call(smb.CASE_BY_ID["card-messy-en"])
    assert s.model_calls == 1 and s.model == "ok" and s.kind == "value" and s.correct is True
    assert (s.input_tokens, s.output_tokens, s.server_latency_ms) == (812, 47, 380)
    assert s.effective_mode == "assisted" and s.fallback is None
    (req,) = client.requests
    assert req["toolConfig"]["tools"][0]["toolSpec"]["name"] == "resolve_answer"
    assert a.budgets == [2500]  # type: ignore[attr-defined]


def test_an_ungrounded_model_value_is_not_accepted():
    """The engine grounds what the model proposes: a ZIP the caller didn't say is ``none``."""
    a = adapter("assisted", FakeClient(lambda kw: {"kind": "exact", "value": "12345"}))
    s = a.call(smb.CASE_BY_ID["zip-corrected-en"])
    assert s.model_calls == 1 and s.model == "ungrounded" and s.kind == "unclear" and s.correct is False


def test_a_sensitive_slot_reaches_the_model_unredacted():
    """The resolver sees the caller's words for a sensitive digit slot too."""
    client = FakeClient(lambda kw: {"kind": "exact", "value": "4821"})
    c = smb.CASE_BY_ID["ssn4-corrected-en"]
    s = adapter("assisted", client).call(c)
    (req,) = client.requests
    assert c.utterance in user_text(req) and "[digits]" not in user_text(req)
    assert s.kind == "value" and s.correct is True


def test_generative_runs_only_on_a_non_sensitive_choice_slot():
    client = FakeClient(lambda kw: {"kind": "repeat"})
    a = adapter("generative", client)
    yn = a.call(smb.CASE_BY_ID["yesno-offscript-en"])
    card = a.call(smb.CASE_BY_ID["card-offscript-en"])
    assert yn.effective_mode == "assisted" and card.effective_mode == "generative"
    names = [r["toolConfig"]["tools"][0]["toolSpec"]["name"] for r in client.requests]
    assert names == ["resolve_answer", "resolve_answer_generative"]


@pytest.mark.parametrize(("say", "say_lang", "verdict"), [
    ("Did you mean the {option:red}?", "en-US", "accepted"),
    ("Did you mean card 2?", "en-US", "number"),
    ("¿Quiso decir la {option:red}?", "es-US", "lang"),
    ("Did you mean the {option:rewards}?", "en-US", "placeholder"),
    (None, None, "missing"),
])
def test_generative_fallback_by_reason(say, say_lang, verdict):
    client = FakeClient(lambda kw: {"kind": "inferred", "value": "red", "say": say, "say_lang": say_lang})
    s = adapter("generative", client).call(smb.CASE_BY_ID["card-messy-en"])
    assert s.kind == "value" and s.fallback == verdict


def test_generative_wording_not_needed_is_not_checked():
    """An answer that needs no follow-up question (an intent switch) has no fallback verdict."""
    client = FakeClient(lambda kw: {"kind": "intent_switch", "say": "anything 7", "say_lang": "en-US"})
    s = adapter("generative", client).call(smb.CASE_BY_ID["card-offscript-en"])
    assert s.kind == "intent_switch" and s.fallback is None


def test_fallback_rate_counts_every_reason_but_accepted():
    ss = [smb.Sample.of(case(), latency_ms=1.0, input_tokens=1, output_tokens=1, server_latency_ms=None,
                        kind="value", value=None, fallback=f) for f in ("accepted", "number", "number", "lang")]
    o = smb.summarize(ss, PRICE)["overall"]
    assert o["fallback"] == {"accepted": 1, "number": 2, "lang": 1} and o["fallback_rate"] == 0.75


def test_a_failing_client_is_a_model_error_not_a_bench_error():
    def boom(kw: Any) -> Any:
        raise ConnectionError("down")

    s = adapter("assisted", FakeClient(boom)).call(smb.CASE_BY_ID["card-messy-en"])
    assert s.error is None and s.model_calls == 1 and s.model == "error" and s.input_tokens is None
    assert s.kind == "unclear"


def test_a_slow_client_is_a_timeout_within_the_budget():
    def slow(kw: Any) -> Any:
        time.sleep(0.3)
        return {}

    s = adapter("assisted", FakeClient(slow), budget_ms=50).call(smb.CASE_BY_ID["card-messy-en"])
    assert s.model == "timeout" and s.input_tokens is None and s.latency_ms < 300


def test_warm_stops_after_n_model_calls():
    client = FakeClient()
    a = adapter("assisted", client)
    assert smb.warm(a, list(smb.CASES), 2) == 2 and len(client.requests) == 2
    assert smb.warm(adapter("deterministic"), list(smb.CASES), 1) == 0


# ----------------------------------------------------------------------------- the result file


SAY_MARK = "Zebra-Quartz wording"  # appears only in the model's wording; must never reach the file


def full_run(tmp_path: Any, latency: float | None = None) -> tuple[Any, dict[str, Any]]:
    client = FakeClient(lambda kw: {"kind": "inferred", "value": "red", "say": SAY_MARK, "say_lang": "en-US"})
    clock = None
    if latency is not None:
        t = [0.0]

        def tick() -> float:
            t[0] += latency / 1000  # one tick per clock read: start, end
            return t[0]

        clock = tick
    adapters: dict[str, Any] = {}
    runs = {}
    for m in smb.MODES:
        kw = {"clock": clock} if clock else {}
        adapters[m] = adapter(m, client if m != "deterministic" else None, **kw)
        runs[m] = smb.run(adapters[m], list(smb.CASES), repeats=1)
    path = smb.write_result(tmp_path, label="slot-resolver", adapters=adapters, price=PRICE, runs=runs,
                            cases=list(smb.CASES), repeats=1, warmup=0, logging_check={"us-east-1": "off"})
    return path, json.loads(path.read_text())


def test_result_has_no_utterance_text_and_a_clear_name(tmp_path):
    path, data = full_run(tmp_path)
    assert re.fullmatch(r"slot-resolver-\d{8}T\d{6}Z\.json", path.name)
    text = path.read_text()
    for c in smb.CASES:  # Tokens and latencies only, never the words
        assert c.utterance not in text and c.question not in text
        if c.label:
            assert c.label not in text
        for _, lab in c.options:
            assert lab not in text
    assert SAY_MARK not in text
    assert "94105" not in text and "4821" not in text  # no expected or extracted value
    assert set(data["modes"]) == set(smb.MODES)
    for m, v in data["modes"].items():
        assert all("value" not in s and "say" not in s for s in v["samples"])
        assert set(v["samples"][0]) == {"case_id", "slot_type", "lang", "category", "latency_ms",
                                        "input_tokens", "output_tokens", "server_latency_ms", "kind",
                                        "correct", "error", "model_calls", "model", "fallback", "mode",
                                        "effective_mode"}
        assert v["adapter"]["mode"] == m
    assert data["modes"]["deterministic"]["summary"]["overall"]["model_calls"] == 0
    assert data["modes"]["assisted"]["summary"]["overall"]["model_calls"] > 0
    assert data["modes"]["generative"]["summary"]["overall"]["fallback"]
    assert data["price"]["source"] == "test" and data["logging_check"] == {"us-east-1": "off"}
    assert "baseline" not in data
    assert data["latency_stop"]["stop"] is False


def test_stop_when_p90_over_one_second(tmp_path):
    _, data = full_run(tmp_path, latency=1200.0)
    v = data["latency_stop"]
    assert v["stop"] is True and v["over"] == ["assisted", "generative"]
    assert v["message"].startswith("STOP:")
    assert smb.latency_stop({"assisted": {"overall": {"latency_p90_ms": 1000.0}}})["stop"] is False


# ----------------------------------------------------------------------------- the logging check


@pytest.mark.parametrize(("response", "verdict"), [
    ({}, "off"),
    ({"loggingConfig": {}}, "on_text"),  # text delivery not stated: treated as on
    ({"loggingConfig": {"cloudWatchConfig": {"logGroupName": "g"}, "textDataDeliveryEnabled": False}},
     "on_no_text"),
    ({"loggingConfig": {"s3Config": {"bucketName": "b"}, "textDataDeliveryEnabled": True}}, "on_text"),
])
def test_logging_verdict(response, verdict):
    assert smb.logging_verdict(response) == verdict


def test_logging_check_blocks_when_any_region_delivers_text():
    responses = {"us-east-1": {"loggingConfig": {"textDataDeliveryEnabled": True}},
                 "us-east-2": {}, "us-west-2": {}}
    verdicts = smb.logging_check(lambda region: responses[region], ("us-east-1", "us-east-2", "us-west-2"))
    assert verdicts == {"us-east-1": "on_text", "us-east-2": "off", "us-west-2": "off"}
    assert smb.logging_blocks(verdicts)
    assert not smb.logging_blocks({"us-east-1": "off", "us-west-2": "on_no_text"})


def no_client(*a: Any, **k: Any) -> Any:
    raise AssertionError("no Bedrock runtime client may be made")


def test_main_refuses_to_run_when_logging_delivers_text(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(smb, "read_logging_config",
                        lambda profile, region: {"loggingConfig": {"textDataDeliveryEnabled": True}})
    monkeypatch.setattr(smb, "make_client_for", no_client)
    with pytest.raises(SystemExit) as exc:
        smb.main(["--out", str(tmp_path)])
    assert exc.value.code == 2
    assert "invocation logging" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_main_runs_every_mode_end_to_end(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(smb, "read_logging_config", lambda profile, region: {})
    client = FakeClient(lambda kw: {"kind": "inferred", "value": "red"})
    monkeypatch.setattr(smb, "make_client_for", lambda profile, region: lambda budget_ms: client)
    smb.main(["--out", str(tmp_path), "--slot-types", "choice", "--langs", "en-US", "--warmup", "1",
              "--repeats", "1"])
    (path,) = tmp_path.iterdir()
    data = json.loads(path.read_text())
    n = len(smb.select(smb.CASES, slot_types=["choice"], langs=["en-US"], categories=None))
    assert set(data["modes"]) == set(smb.MODES)
    assert all(v["summary"]["overall"]["answers"] == n for v in data["modes"].values())
    assert path.name.startswith("slot-resolver-") and data["warmup"] == 1
    out = capsys.readouterr().out
    assert "no stop" in out


def test_main_stops_when_p90_is_over_one_second(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(smb, "read_logging_config", lambda profile, region: {})
    monkeypatch.setattr(smb, "make_client_for", lambda profile, region: lambda budget_ms: FakeClient())
    real = smb.ResolverAdapter
    t = [0.0]

    def tick() -> float:
        t[0] += 1.2  # two reads per engine call: each takes 1.2 s
        return t[0]

    monkeypatch.setitem(smb.ADAPTERS, "resolver", lambda cf, **kw: real(cf, clock=tick, **kw))
    with pytest.raises(SystemExit) as exc:
        smb.main(["--out", str(tmp_path), "--modes", "assisted", "--slot-types", "choice", "--warmup", "0",
                  "--repeats", "1"])
    assert exc.value.code == 3
    captured = capsys.readouterr()
    assert "STOP:" in captured.out and "STOP:" in captured.err
    (path,) = tmp_path.iterdir()
    assert json.loads(path.read_text())["latency_stop"]["stop"] is True


def test_main_refuses_an_unknown_adapter_or_mode_after_the_logging_check(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(smb, "read_logging_config", lambda profile, region: {})
    monkeypatch.setattr(smb, "make_client_for", no_client)
    with pytest.raises(SystemExit) as exc:
        smb.main(["--out", str(tmp_path), "--adapter", "classifier"])
    assert exc.value.code == 2 and "--adapter" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        smb.main(["--out", str(tmp_path), "--modes", "assisted,fancy"])
    assert exc.value.code == 2 and "--modes" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_main_reads_only_the_source_region(monkeypatch, tmp_path):
    seen: list[str] = []

    def read(profile: Any, region: str) -> dict:
        seen.append(region)
        return {"loggingConfig": {"textDataDeliveryEnabled": True}}

    monkeypatch.setattr(smb, "read_logging_config", read)
    monkeypatch.setattr(smb, "make_client_for", no_client)
    with pytest.raises(SystemExit):
        smb.main(["--out", str(tmp_path), "--region", "us-east-1"])
    assert seen == ["us-east-1"]


def test_accept_text_logging_passes_the_guard_with_a_warning(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(smb, "read_logging_config",
                        lambda profile, region: {"loggingConfig": {"textDataDeliveryEnabled": True}})
    monkeypatch.setattr(smb, "make_client_for", no_client)
    with pytest.raises(AssertionError, match="no Bedrock runtime client"):  # got past the guard
        smb.main(["--out", str(tmp_path), "--accept-text-logging"])
    assert "running anyway" in capsys.readouterr().err
