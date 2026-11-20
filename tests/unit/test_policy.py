"""Unit tests for owlbank.policy."""

from __future__ import annotations

import collections
import copy
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from owlbank.catalog import PromptCatalog
from owlbank.policy import EscalationRoute, Policy, PolicyError
from owlbank.store import SCHEMAS, DynamoStore, MemoryStore, StoreError, create_table_kwargs
from tests.logutil import lines

DATA = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def seed() -> dict[str, Any]:
    return json.loads((DATA / "policy.json").read_text(encoding="utf-8"))


@pytest.fixture()
def policy(seed: dict[str, Any]) -> Policy:
    return Policy.from_dict(seed)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


# --------------------------------------------------------------------------- tool rows


@pytest.mark.parametrize(
    ("tool", "level"),
    [("get_branch_hours", 0), ("faq_lookup", 0), ("get_balance", 1),
     ("list_recent_transactions", 1), ("change_address", 2), ("transfer_funds", 3),
     ("make_payment", 3)],
)
def test_min_level_matches_handover_table(policy: Policy, tool: str, level: int) -> None:
    assert policy.min_level(tool) == level


def test_unknown_tool_fails_closed(policy: Policy, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="owlbank.policy"):
        assert policy.min_level("wire_everything") == policy.max_level == 3
    assert "wire_everything" in caplog.text
    assert policy.phrase_key("wire_everything") is None
    assert policy.kind("wire_everything") is None
    assert not policy.has_tool("wire_everything")


def test_phrase_key_and_kind(policy: Policy) -> None:
    assert policy.phrase_key("make_payment") == "tool.make_payment"
    assert policy.phrase_key("faq_lookup") is None
    assert policy.kind("get_balance") == "deterministic"
    assert policy.kind("make_payment") == "agentic_inline"
    assert policy.kind("authenticate") == "handoff"
    assert policy.tool_row("get_balance")["template_key"] == "balance.read"


# --------------------------------------------------------------------------- levels and factors


def test_factors_cumulative_in_policy_order(policy: Policy) -> None:
    assert policy.factors(0) == []
    assert policy.factors(1) == ["ani_match", "zip"]
    assert policy.factors(2) == ["ani_match", "zip", "ssn4"]
    assert policy.factors(3) == ["ani_match", "zip", "ssn4", "otp"]


def test_missing_factors(policy: Policy) -> None:
    assert policy.missing_factors({"ani_match", "zip"}, 3) == ["ssn4", "otp"]
    assert policy.missing_factors({"otp", "ani_match"}, 3) == ["zip", "ssn4"]
    assert policy.missing_factors(set(), 1) == ["ani_match", "zip"]
    assert policy.missing_factors({"ani_match", "zip", "ssn4"}, 2) == []
    assert policy.missing_factors(set(), 0) == []


@pytest.mark.parametrize("level", [-1, 4, 99])
def test_bad_level_rejected(policy: Policy, level: int) -> None:
    with pytest.raises(PolicyError):
        policy.factors(level)
    with pytest.raises(PolicyError):
        policy.missing_factors(set(), level)


def test_factor_definition(policy: Policy) -> None:
    assert policy.factor("ssn4")["promptKey"] == "factor.ssn4"
    assert policy.factor("ani_match") is None


def test_limits(policy: Policy) -> None:
    assert policy.limit("stepup_attempts_per_call") == 2
    assert policy.limit("summary_max_chars") == 1000


# --------------------------------------------------------------------------- escalation


@pytest.mark.parametrize(
    ("reason", "queue", "priority", "phrase"),
    [("caller_requested_human", "support", 0, "escalate.default"),
     ("unable_to_resolve", "support", 0, "escalate.default"),
     ("step_up_failed", "support", 0, "escalate.verify"),
     ("fraud_suspected", "fraud", 100, "escalate.fraud"),
     ("complaint", "support", 0, "escalate.default"),
     ("vulnerable_customer", "support", 50, "escalate.default")],
)
def test_escalation_reason_table(policy: Policy, reason: str, queue: str, priority: int,
                                 phrase: str) -> None:
    assert policy.escalation_route(reason) == EscalationRoute(queue, priority, phrase, reason)


def test_unknown_reason_defaults(policy: Policy, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="owlbank.policy"):
        route = policy.escalation_route("give_me_a_manager_now")
    assert route == policy.escalation_route("unable_to_resolve")
    assert route.reason == "unable_to_resolve"
    assert "unknown escalation reason" in caplog.text


def test_queue_names(policy: Policy) -> None:
    assert policy.queue_name("support", "es-US") == "OwlBank Support ES"
    assert policy.queue_name("support", "fr-FR") == "OwlBank Support EN"


def test_referenced_prompt_keys_exist_in_catalog(policy: Policy) -> None:
    catalog = PromptCatalog.from_dict(
        json.loads((DATA / "prompt_catalog.json").read_text(encoding="utf-8")))
    for lang in catalog.languages:
        missing = {k for k in policy.referenced_prompt_keys() if not catalog.has(k, lang)}
        assert not missing, (lang, missing)


# --------------------------------------------------------------------------- validation


def _broken(seed: dict[str, Any], mutate) -> dict[str, Any]:
    doc = copy.deepcopy(seed)
    mutate(doc)
    return doc


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d["levels"].pop("2"), id="gap-in-levels"),
        pytest.param(lambda d: d.__setitem__("max_level", 4), id="max-level-mismatch"),
        pytest.param(lambda d: d["levels"].__setitem__("3", ["ani_match", "otp"]), id="not-cumulative"),
        pytest.param(lambda d: d["levels"].__setitem__("0", []), id="level-0-row"),
        pytest.param(lambda d: d["levels"].__setitem__("1", ["ani_match", "pin"]), id="undefined-factor"),
        pytest.param(lambda d: d["tools"]["make_payment"].pop("phrase_key"), id="no-phrase-key-field"),
        pytest.param(lambda d: d["tools"]["make_payment"].__setitem__("phrase_key", None),
                     id="null-phrase-key-on-gated-tool"),
        pytest.param(lambda d: d["tools"]["make_payment"].__setitem__("min_level", 7), id="level-too-high"),
        pytest.param(lambda d: d["tools"]["make_payment"].__setitem__("kind", "magic"), id="bad-kind"),
        pytest.param(lambda d: d["escalation"].__setitem__("default_reason", "nope"), id="bad-default"),
        pytest.param(lambda d: d["escalation"]["reasons"]["complaint"].__setitem__("queue", "x"),
                     id="unknown-queue"),
    ],
)
def test_invalid_policy_rejected(seed: dict[str, Any], mutate) -> None:
    with pytest.raises(PolicyError):
        Policy.from_dict(_broken(seed, mutate))


# --------------------------------------------------------------------------- store and cache


def _seed_store(store: MemoryStore, doc: dict[str, Any]) -> None:
    for item in Policy.to_items(doc):
        store.put("policy", item)


def test_store_round_trip_matches_dict(seed: dict[str, Any]) -> None:
    store = MemoryStore()
    _seed_store(store, seed)
    a, b = Policy.from_dict(seed), Policy.from_store(store)
    assert a.tools == b.tools and a.reasons == b.reasons and a.max_level == b.max_level
    for tool in a.tools:
        assert a.tool_row(tool) == b.tool_row(tool)
    for n in range(a.max_level + 1):
        assert a.factors(n) == b.factors(n)
    for r in a.reasons:
        assert a.escalation_route(r) == b.escalation_route(r)
    assert store.get("policy", {"pk": "tool#make_payment"})["min_level"] == 3
    assert store.get("policy", {"pk": "level#2"})["factors"] == ["ani_match", "zip", "ssn4"]


def test_store_cache_refreshes_after_ttl(seed: dict[str, Any]) -> None:
    store, clock = MemoryStore(), Clock()
    _seed_store(store, seed)
    policy = Policy.from_store(store, ttl_s=60, clock=clock)
    store.update("policy", {"pk": "tool#make_payment"}, {"min_level": 2})
    clock.t += 59
    assert policy.min_level("make_payment") == 3  # cached
    clock.t += 1
    assert policy.min_level("make_payment") == 2  # refreshed


def test_failed_refresh_keeps_last_good(seed: dict[str, Any],
                                        caplog: pytest.LogCaptureFixture) -> None:
    store, clock = MemoryStore(), Clock()
    _seed_store(store, seed)
    policy = Policy.from_store(store, ttl_s=60, clock=clock)
    store.delete("policy", {"pk": "level#2"})
    clock.t += 61
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        assert policy.factors(2) == ["ani_match", "zip", "ssn4"]
    assert "keeping the last good policy" in caplog.text


def test_empty_store_raises() -> None:
    with pytest.raises(PolicyError):
        Policy.from_store(MemoryStore())


def test_from_dict_is_isolated_from_caller_mutation(seed: dict[str, Any]) -> None:
    doc = copy.deepcopy(seed)
    policy = Policy.from_dict(doc)
    doc["tools"]["make_payment"]["min_level"] = 0
    assert policy.min_level("make_payment") == 3


# --------------------------------------------------------------------------- redaction config


def test_seed_redaction_section_is_the_safe_default(policy: Policy) -> None:
    from owlbank.pii import DEFAULT_CONFIG

    assert policy.redaction == DEFAULT_CONFIG
    assert policy.redaction.entity_types == ("SSN", "CREDIT_DEBIT_NUMBER", "CREDIT_DEBIT_CVV",
                                             "CREDIT_DEBIT_EXPIRY", "BANK_ACCOUNT_NUMBER",
                                             "BANK_ROUTING", "PIN", "PASSWORD")


def test_redaction_round_trips_through_the_store(seed: dict[str, Any]) -> None:
    doc = copy.deepcopy(seed)
    doc["redaction"] = {"enabled": False, "entity_types": ["SSN", "ADDRESS"], "min_confidence": 0.75,
                        "mask": "<{type}>", "on_failure": "allow"}
    store = MemoryStore()
    for item in Policy.to_items(doc):  # the store rejects floats: min_confidence is a string there
        store.put("policy", item)
    assert store.get("policy", {"pk": "redaction"})["min_confidence"] == "0.75"
    cfg = Policy.from_store(store).redaction
    assert cfg.entity_types == ("SSN", "ADDRESS") and cfg.min_confidence == 0.75
    assert cfg.mask == "<{type}>" and cfg.on_failure == "allow" and cfg.enabled is False


@pytest.mark.parametrize("bad,field", [
    ({"entity_types": ["SSN", "FAVOURITE_COLOUR"]}, "entity_types"),
    ({"min_confidence": 2}, "min_confidence"),
    ({"mask": "{type}{value}"}, "mask"),
    ({"on_failure": "maybe"}, "on_failure"),
    ({"enabled": "no"}, "enabled"),
    ({"entity_types": []}, "entity_types"),  # Empty while enabled is invalid, not "off"
    ({"apply_to": ["ai_turn"]}, "apply_to"),  # removed: an unknown field
])
def test_invalid_redaction_falls_back_to_defaults_with_an_error(
        seed: dict[str, Any], bad: dict[str, Any], field: str, caplog: pytest.LogCaptureFixture) -> None:
    from owlbank.pii import DEFAULT_CONFIG

    caplog.set_level(logging.ERROR)
    doc = copy.deepcopy(seed)
    doc["redaction"] = {**doc["redaction"], **bad}
    p = Policy.from_dict(doc)  # the rest of the policy still loads
    assert p.redaction == DEFAULT_CONFIG and p.min_level("make_payment") == 3
    assert any(field in r.getMessage() and r.levelno == logging.ERROR for r in caplog.records)
    with pytest.raises(PolicyError, match=field):  # seeding refuses it outright
        Policy.to_items(doc)


def test_missing_redaction_section_uses_defaults_and_logs(seed: dict[str, Any],
                                                          caplog: pytest.LogCaptureFixture) -> None:
    from owlbank.pii import DEFAULT_CONFIG

    caplog.set_level(logging.ERROR)
    doc = {k: v for k, v in seed.items() if k != "redaction"}
    assert Policy.from_dict(doc).redaction == DEFAULT_CONFIG
    assert any("redaction" in r.getMessage() for r in caplog.records)
    store = MemoryStore()
    for item in Policy.to_items(doc):
        store.put("policy", item)
    assert store.get("policy", {"pk": "redaction"}) is None
    assert Policy.from_store(store).redaction == DEFAULT_CONFIG


def test_redaction_edit_in_the_table_takes_effect_after_the_ttl(seed: dict[str, Any]) -> None:
    store = MemoryStore()
    for item in Policy.to_items(seed):
        store.put("policy", item)
    clock = Clock()
    p = Policy.from_store(store, ttl_s=60, clock=clock)
    item = store.get("policy", {"pk": "redaction"})
    store.put("policy", {**item, "entity_types": [*item["entity_types"], "ADDRESS"]})
    assert "ADDRESS" not in p.redaction.entity_types
    clock.t += 61
    assert "ADDRESS" in p.redaction.entity_types


def test_redaction_off_is_enabled_false(seed: dict[str, Any]) -> None:
    doc = copy.deepcopy(seed)
    doc["redaction"]["enabled"] = False
    assert Policy.from_dict(doc).redaction.enabled is False
    assert any(i["pk"] == "redaction" and i["enabled"] is False for i in Policy.to_items(doc))
    doc["redaction"]["entity_types"] = []  # allowed while off
    assert Policy.from_dict(doc).redaction.entity_types == ()
    assert any(i["pk"] == "redaction" and i["entity_types"] == [] for i in Policy.to_items(doc))


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
        yield DynamoStore(names, resource=resource, sleep=lambda s: None), calls


def test_reload_is_one_get_and_one_batch_get(seed: dict[str, Any], dynamo: Any) -> None:
    store, calls = dynamo
    _seed_store(store, seed)
    calls.clear()
    policy = Policy.from_store(store)
    assert calls == {"GetItem": 1, "BatchGetItem": 1}
    assert policy.tools == Policy.from_dict(seed).tools


def test_reload_retries_unprocessed_keys(seed: dict[str, Any], dynamo: Any,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    store, calls = dynamo
    _seed_store(store, seed)
    resource = store._dynamodb()
    real = resource.batch_get_item
    held: list[bool] = []

    def flaky(**kw: Any) -> dict[str, Any]:  # the first response leaves two keys unprocessed
        if held:
            return real(**kw)
        held.append(True)
        ((physical, spec),) = kw["RequestItems"].items()
        out = real(RequestItems={physical: {**spec, "Keys": spec["Keys"][2:]}})
        out["UnprocessedKeys"] = {physical: {**spec, "Keys": spec["Keys"][:2]}}
        return out

    monkeypatch.setattr(resource, "batch_get_item", flaky)
    calls.clear()
    policy = Policy.from_store(store)
    assert calls == {"GetItem": 1, "BatchGetItem": 2}
    assert policy.factors(2) == Policy.from_dict(seed).factors(2)


class _BrokenBatch(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.broken = False

    def batch_get(self, table: str, keys: Any) -> list[dict[str, Any]]:
        if self.broken:
            raise StoreError("throttled")
        return super().batch_get(table, keys)


def test_a_store_error_keeps_the_last_good_policy(seed: dict[str, Any],
                                                  caplog: pytest.LogCaptureFixture) -> None:
    store, clock = _BrokenBatch(), Clock()
    _seed_store(store, seed)
    policy = Policy.from_store(store, ttl_s=60, clock=clock)
    store.update("policy", {"pk": "tool#make_payment"}, {"min_level": 2})
    store.broken = True
    clock.t += 61
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        assert policy.min_level("make_payment") == 3
    assert "keeping the last good policy" in caplog.text
    store.broken = False
    clock.t += 61
    assert policy.min_level("make_payment") == 2


def test_keys_still_unprocessed_keep_the_last_good_policy(seed: dict[str, Any], dynamo: Any,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    store, _calls = dynamo
    _seed_store(store, seed)
    clock = Clock()
    policy = Policy.from_store(store, ttl_s=60, clock=clock)
    resource = store._dynamodb()

    def never(**kw: Any) -> dict[str, Any]:
        return {"Responses": {}, "UnprocessedKeys": kw["RequestItems"]}

    monkeypatch.setattr(resource, "batch_get_item", never)
    store.update("policy", {"pk": "tool#make_payment"}, {"min_level": 2})
    clock.t += 61
    assert policy.min_level("make_payment") == 3
    with pytest.raises(StoreError):
        policy.refresh()


def test_reload_logs_one_dependency_call(seed: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    store = MemoryStore()
    _seed_store(store, seed)
    caplog.set_level(logging.DEBUG, logger="owlbank.dependency")
    Policy.from_store(store)
    (line,) = [f for f in lines(caplog.records, "owlbank.dependency") if f["operation"] == "reload:policy"]
    assert line["event"] == "dependency_call" and line["dependency"] == "dynamodb"
    assert line["outcome"] == "ok" and line["items"] == len(Policy.to_items(seed))
    assert isinstance(line["duration_ms"], float)
    (record,) = [r for r in caplog.records if getattr(r, "operation", None) == "reload:policy"]
    assert record.levelno == logging.INFO
