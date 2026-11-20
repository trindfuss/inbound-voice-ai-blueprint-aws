"""scripts/check_resolver.py: the live resolver check (b), offline.

A fake Converse client and a fake clock; the logging guard reads canned
``GetModelInvocationLoggingConfiguration`` responses. No AWS calls. No PBT (an I/O check).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from botocore.exceptions import ClientError

from owlbank.slots.resolver import GENERATIVE_TOOL_NAME, TOOL_NAME, BedrockResolver
from scripts import check_resolver as cr

# ----------------------------------------------------------------------------- fakes


def tool_input(generative: bool, **over: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"kind": "exact", "value": "7777", "candidates": [], "slot": None, "reply": None}
    if generative:
        data.update(say=None, say_lang=None)
    data.update(over)
    return data


def reply(name: str, data: Any, *, latency: int = 420, stop: str = "tool_use") -> dict[str, Any]:
    return {"stopReason": stop,
            "output": {"message": {"role": "assistant",
                                   "content": [{"toolUse": {"toolUseId": "t1", "name": name, "input": data}}]}},
            "usage": {"inputTokens": 900, "outputTokens": 60},
            "metrics": {"latencyMs": latency}}


def good(kwargs: dict[str, Any]) -> dict[str, Any]:
    name = kwargs["toolConfig"]["toolChoice"]["tool"]["name"]
    return reply(name, tool_input(name == GENERATIVE_TOOL_NAME))


class FakeClient:
    """Answers each Converse call with ``respond(kwargs)`` (a response, or an exception to raise)."""

    def __init__(self, respond: Any = good) -> None:
        self.respond = respond
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        out = self.respond(kwargs)
        if isinstance(out, Exception):
            raise out
        return out


def ticking(step_s: float) -> Any:
    """A clock that advances ``step_s`` on every read."""
    def gen() -> Iterator[float]:
        t = 100.0
        while True:
            yield t
            t += step_s
    it = gen()
    return lambda: next(it)


def validation_error() -> ClientError:
    return ClientError({"Error": {"Code": "ValidationException",
                                  "Message": "Enum value 'yes' does not match declared type ['string', 'null']"}},
                       "Converse")


RESOLVER = BedrockResolver(lambda budget_ms: None, model_id="test-model")


def check(client: FakeClient, *, generative: bool = False, step_s: float = 0.4, budget_ms: int = 1500) -> cr.Result:
    return cr.check_tool(client, RESOLVER, generative=generative, budget_ms=budget_ms, clock=ticking(step_s))


# ----------------------------------------------------------------------------- check_tool


@pytest.mark.parametrize("generative", [False, True])
def test_pass_uses_the_resolvers_own_request(generative):
    client = FakeClient()
    r = check(client, generative=generative)
    assert r.passed and r.reason is None
    assert r.tool == (GENERATIVE_TOOL_NAME if generative else TOOL_NAME)
    assert (r.latency_ms, r.server_latency_ms, r.input_tokens, r.output_tokens) == (400, 420, 900, 60)
    assert client.calls == [RESOLVER.request(cr.REQUEST, generative=generative)]
    assert "PASS" in r.line() and "latency_ms=400" in r.line() and "server_latency_ms=420" in r.line()


def test_the_fixed_request_is_synthetic_ssn4_digits():
    req = cr.REQUEST
    assert (req.language, req.slot.name, req.slot.slot_type) == ("en-US", "ssn4", "digits")
    assert dict(req.slot.constraints) == {"length": 4}
    assert req.utterance == "seven seven and seven seven"
    assert not req.collected and not req.caller_options and req.pending is None


def test_bedrock_validation_exception_fails_with_its_code_and_message():
    r = check(FakeClient(lambda kw: validation_error()))
    assert not r.passed
    assert r.reason is not None and r.reason.startswith("bedrock refused: ValidationException")
    assert "does not match declared type" in r.reason
    assert r.server_latency_ms is None and r.input_tokens is None


def test_an_echoed_utterance_is_masked_and_the_message_cut():
    msg = f"bad input: {cr.UTTERANCE} " + "x" * 1000
    err = ClientError({"Error": {"Code": "ValidationException", "Message": msg}}, "Converse")
    r = check(FakeClient(lambda kw: err))
    assert r.reason is not None and cr.UTTERANCE not in r.reason and "[utterance]" in r.reason
    assert len(r.reason) <= len("bedrock refused: ValidationException: ") + cr.MAX_MESSAGE


def test_a_non_bedrock_error_shows_only_its_class():
    r = check(FakeClient(lambda kw: TimeoutError("read timed out with secret words")))
    assert r.reason == "error TimeoutError"


@pytest.mark.parametrize("respond, why", [
    (lambda kw: reply(TOOL_NAME, {"kind": "exact"}), "outcome malformed"),              # off the schema
    (lambda kw: reply(TOOL_NAME, "not json {"), "outcome malformed"),                   # input isn't JSON
    (lambda kw: reply("other_tool", tool_input(False)), "outcome malformed"),           # wrong tool
    (lambda kw: {**reply(TOOL_NAME, tool_input(False)), "stopReason": "end_turn"}, "outcome refusal"),
])
def test_a_malformed_tool_reply_fails(respond, why):
    r = check(FakeClient(respond))
    assert not r.passed and r.reason is not None and r.reason.startswith(why)


def test_a_malformed_reply_never_prints_model_text():
    text = "the model rambles about 1234"
    resp = {"stopReason": "end_turn", "output": {"message": {"content": [{"text": text}]}}}
    r = check(FakeClient(lambda kw: resp))
    assert r.reason == "outcome refusal (stop_reason end_turn)"
    assert text not in r.line()


def test_a_wrong_value_fails():
    r = check(FakeClient(lambda kw: reply(TOOL_NAME, tool_input(False, value="777"))))
    assert not r.passed and r.reason == "kind exact value '777', expected '7777'"


def test_a_none_kind_fails():
    r = check(FakeClient(lambda kw: reply(TOOL_NAME, tool_input(False, kind="none", value=None))))
    assert not r.passed and r.reason is not None and "value None" in r.reason


def test_over_budget_fails_and_at_budget_passes():
    over = check(FakeClient(), step_s=1.501)
    assert not over.passed and over.reason == "over budget (1501 ms > 1500 ms)"
    assert check(FakeClient(), step_s=1.5).passed


# ----------------------------------------------------------------------------- main and the guard


def _wire(monkeypatch, *, logging: dict[str, Any], client: FakeClient) -> dict[str, Any]:
    seen: dict[str, Any] = {"made": [], "budgets": []}

    def make_client_for(profile, region):
        seen["made"].append((profile, region))

        def client_for(budget_ms):
            seen["budgets"].append(budget_ms)
            return client

        return client_for

    monkeypatch.setattr(cr, "read_logging_config", lambda profile, region: logging)
    monkeypatch.setattr(cr, "make_client_for", make_client_for)
    return seen


def test_main_passes_both_tools(monkeypatch, capsys):
    client = FakeClient()
    seen = _wire(monkeypatch, logging={}, client=client)
    assert cr.main(["--profile", "p", "--model-id", "m", "--budget-ms", "2000"], clock=ticking(0.3)) == 0
    out = capsys.readouterr().out
    assert "invocation logging: us-east-1 off" in out
    lines = [ln for ln in out.splitlines() if ln.startswith("resolve_answer")]
    assert len(lines) == 2 and all(ln.endswith("PASS") for ln in lines)
    assert seen == {"made": [("p", "us-east-1")], "budgets": [2000]}
    assert [c["modelId"] for c in client.calls] == ["m", "m", "m"]  # one warm-up, then the two timed calls
    assert [c["toolConfig"]["toolChoice"]["tool"]["name"] for c in client.calls] == [
        TOOL_NAME, TOOL_NAME, GENERATIVE_TOOL_NAME]
    assert cr.UTTERANCE not in out


def test_main_defaults_to_the_resolvers_model(monkeypatch):
    client = FakeClient()
    _wire(monkeypatch, logging={}, client=client)
    monkeypatch.setenv("OWLBANK_RESOLVER_MODEL_ID", "env-model")
    assert cr.main([], clock=ticking(0.1)) == 0
    assert {c["modelId"] for c in client.calls} == {"env-model"}
    monkeypatch.delenv("OWLBANK_RESOLVER_MODEL_ID")
    client.calls.clear()
    assert cr.main([], clock=ticking(0.1)) == 0
    assert {c["modelId"] for c in client.calls} == {"us.anthropic.claude-haiku-5-5"}


def test_main_exits_1_when_any_tool_fails(monkeypatch, capsys):
    def respond(kw):
        name = kw["toolConfig"]["toolChoice"]["tool"]["name"]
        return validation_error() if name == GENERATIVE_TOOL_NAME else good(kw)

    _wire(monkeypatch, logging={}, client=FakeClient(respond))
    assert cr.main([], clock=ticking(0.1)) == 1
    out = capsys.readouterr().out
    assert "PASS" in out and "FAIL bedrock refused: ValidationException" in out


def test_main_refuses_when_logging_delivers_text(monkeypatch, capsys):
    client = FakeClient()
    seen = _wire(monkeypatch, logging={"loggingConfig": {"textDataDeliveryEnabled": True}}, client=client)
    assert cr.main([]) == 2
    assert "refusing to run" in capsys.readouterr().err
    assert seen["made"] == [] and client.calls == []


def test_the_refusal_names_only_the_accept_flag_and_never_advises_changing_the_account(monkeypatch, capsys):
    _wire(monkeypatch, logging={"loggingConfig": {"textDataDeliveryEnabled": True}}, client=FakeClient())
    assert cr.main([]) == 2
    err = capsys.readouterr().err
    assert "--accept-text-logging" in err and "whoever enabled it" in err
    assert "--slot-mode-ceiling" not in err and "deterministic" not in err
    for word in ("delete", "put-model-invocation", "text delivery off"):
        assert word not in err


def test_warm_up_is_untimed_and_first_and_its_failure_is_not_a_result(monkeypatch, capsys):
    calls = {"n": 0}

    def respond(kw):
        calls["n"] += 1
        return TimeoutError("cold") if calls["n"] == 1 else good(kw)

    client = FakeClient(respond)
    _wire(monkeypatch, logging={}, client=client)
    reads = {"n": 0}

    def clock():
        reads["n"] += 1
        return reads["n"] * 0.1

    assert cr.main([], clock=clock) == 0
    assert len(client.calls) == 3 and reads["n"] == 4   # two reads per timed call, none for the warm-up
    assert len([ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("resolve_answer")]) == 2


def test_main_phone_makes_no_warm_up_call(monkeypatch):
    client = FakeClient(phone_model)
    _wire(monkeypatch, logging={}, client=client)
    cr.main(["--scenarios", "phone"], clock=ticking(0.1))
    assert len(client.calls) == sum(sc.resolver_called for sc in cr.PHONE_SCENARIOS)


def test_main_refuses_when_text_delivery_isnt_stated(monkeypatch):
    client = FakeClient()
    _wire(monkeypatch, logging={"loggingConfig": {"s3Config": {"bucketName": "b"}}}, client=client)
    assert cr.main([]) == 2 and client.calls == []


def test_accept_text_logging_warns_and_runs(monkeypatch, capsys):
    client = FakeClient()
    _wire(monkeypatch, logging={"loggingConfig": {"textDataDeliveryEnabled": True}}, client=client)
    assert cr.main(["--accept-text-logging"], clock=ticking(0.1)) == 0
    assert "WARNING" in capsys.readouterr().err and len(client.calls) == 3


def test_logging_without_text_runs(monkeypatch):
    client = FakeClient()
    _wire(monkeypatch, logging={"loggingConfig": {"textDataDeliveryEnabled": False}}, client=client)
    assert cr.main([], clock=ticking(0.1)) == 0


def test_the_guard_is_the_benchs():
    from scripts.bench import slot_model_bench as smb

    assert cr.logging_check is smb.logging_check and cr.logging_blocks is smb.logging_blocks
    assert cr.read_logging_config is smb.read_logging_config


def test_request_serializes():
    json.dumps(RESOLVER.request(cr.REQUEST, generative=True))


# ----------------------------------------------------------------------------- phone scenarios


@pytest.mark.parametrize("sc", cr.PHONE_SCENARIOS, ids=lambda sc: sc.name)
def test_phone_scenarios_run_offline_against_the_resolver_fake(sc):
    from owlbank.slots import FakeResolver, ResolverOutput

    outputs = [ResolverOutput(kind="inferred", value=sc.model_value)] if sc.model_value else []
    fake = FakeResolver(outputs)
    r, why = cr.run_scenario(sc, fake)
    assert why is None, why
    assert len(fake.calls) == (1 if sc.resolver_called else 0)
    if sc.resolver_called:
        assert fake.calls[0]["req"].slot.name == "phone" and fake.calls[0]["req"].utterance == sc.utterance


def test_phone_scenarios_cover_the_required_cases():
    names = {sc.name for sc in cr.PHONE_SCENARIOS}
    assert {"groups_with_filler", "self_correction", "leading_one_digits", "leading_one_words",
            "deterministic_re_asks"} <= names
    owner = next(sc for sc in cr.PHONE_SCENARIOS if sc.name == "groups_with_filler")
    assert (owner.utterance, owner.value) == ("Yeah the number is 407, eh 544, what was it 5555", "4075445555")
    assert all(sc.mode != "deterministic" or not sc.resolver_called for sc in cr.PHONE_SCENARIOS)


def test_an_ungrounded_model_value_fails_the_scenario():
    from owlbank.slots import FakeResolver, ResolverOutput

    sc = cr.PHONE_SCENARIOS[0]
    _, why = cr.run_scenario(sc, FakeResolver([ResolverOutput(kind="exact", value="2125550000")]))
    assert why is not None and "expected exact|inferred" in why


# The model's answers in a live run (Haiku 5.5), replayed offline: each fails its scenario; the
# fixed prompt's answers (exact, as live after the fix, or inferred) pass and are grounded.
T8_ANSWERS = {
    "groups_with_filler": {"kind": "none", "value": None},
    "self_correction": {"kind": "correction", "value": "5445555", "slot": "phone"},
}


@pytest.mark.parametrize("name", sorted(T8_ANSWERS))
def test_the_live_answers_fail_and_the_fixed_answers_pass(name):
    from owlbank.slots import FakeResolver, ResolverOutput

    sc = next(s for s in cr.PHONE_SCENARIOS if s.name == name)
    _, why = cr.run_scenario(sc, FakeResolver([ResolverOutput(**T8_ANSWERS[name])]))
    assert why is not None
    for kind in ("exact", "inferred"):
        fake = FakeResolver([ResolverOutput(kind=kind, value="4075445555")])
        r, why = cr.run_scenario(sc, fake)
        assert why is None and (r.match, r.value) == (kind, "4075445555"), why
        assert len(fake.calls) == 1


def test_leading_one_words_never_reaches_the_model():
    # the model answered 1407544555 (leading one kept, last digit dropped) and it
    # grounded as a subsequence. The parser now reads the words itself.
    from owlbank.slots import FakeResolver, ResolverOutput

    sc = next(s for s in cr.PHONE_SCENARIOS if s.name == "leading_one_words")
    fake = FakeResolver([ResolverOutput(kind="exact", value="1407544555")])
    r, why = cr.run_scenario(sc, fake)
    assert why is None and (r.match, r.value) == ("exact", "4075445555") and not fake.calls


# ----------------------------------------------------------------------------- --scenarios


def phone_model(kwargs: dict[str, Any]) -> dict[str, Any]:
    """A fake Bedrock that answers a phone scenario's request with its ``model_value`` (and the
    default check's request as ``good`` does)."""
    name = kwargs["toolConfig"]["toolChoice"]["tool"]["name"]
    text = json.dumps(kwargs)
    sc = next((s for s in cr.PHONE_SCENARIOS if s.model_value and json.dumps(s.utterance)[1:-1] in text), None)
    if sc is None:
        return good(kwargs)
    return reply(name, tool_input(name == GENERATIVE_TOOL_NAME, kind="inferred", value=sc.model_value))


def test_scenarios_flag_parses_and_defaults_to_default():
    assert cr.build_parser().parse_args([]).scenarios == "default"
    for choice in ("default", "phone", "all"):
        assert cr.build_parser().parse_args(["--scenarios", choice]).scenarios == choice
    with pytest.raises(SystemExit):
        cr.build_parser().parse_args(["--scenarios", "card"])


def test_main_phone_runs_only_the_phone_scenarios_through_the_model(monkeypatch, capsys):
    client = FakeClient(phone_model)
    _wire(monkeypatch, logging={}, client=client)
    assert cr.main(["--scenarios", "phone", "--budget-ms", "2000"], clock=ticking(0.1)) == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("phone:")]
    assert [ln.split()[0] for ln in lines] == [f"phone:{sc.name}" for sc in cr.PHONE_SCENARIOS]
    assert all(ln.endswith("PASS") for ln in lines), lines
    assert not [ln for ln in out.splitlines() if ln.startswith("resolve_answer")]
    assert len(client.calls) == sum(sc.resolver_called for sc in cr.PHONE_SCENARIOS)
    assert all(sc.utterance not in out for sc in cr.PHONE_SCENARIOS)


def test_main_all_runs_both(monkeypatch, capsys):
    client = FakeClient(phone_model)
    _wire(monkeypatch, logging={}, client=client)
    assert cr.main(["--scenarios", "all"], clock=ticking(0.1)) == 0
    out = capsys.readouterr().out.splitlines()
    assert len([ln for ln in out if ln.startswith("resolve_answer")]) == 2
    assert len([ln for ln in out if ln.startswith("phone:")]) == len(cr.PHONE_SCENARIOS)


def test_main_phone_exits_1_when_a_scenario_fails(monkeypatch, capsys):
    _wire(monkeypatch, logging={}, client=FakeClient())  # answers 7777 to everything
    assert cr.main(["--scenarios", "phone"], clock=ticking(0.1)) == 1
    out = capsys.readouterr().out
    assert "phone:groups_with_filler" in out and "FAIL" in out


def test_main_phone_is_refused_by_the_logging_guard(monkeypatch):
    client = FakeClient(phone_model)
    _wire(monkeypatch, logging={"loggingConfig": {"textDataDeliveryEnabled": True}}, client=client)
    assert cr.main(["--scenarios", "phone"]) == 2 and client.calls == []
