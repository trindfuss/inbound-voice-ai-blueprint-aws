"""``make_payment.to`` is an optional enum of the ``card`` slot's known options."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from owlbank import tools_contract as tc
from owlbank.policy import Policy

DATA = Path(__file__).resolve().parents[2] / "data"
POLICY_DOC = json.loads((DATA / "policy.json").read_text(encoding="utf-8"))
CATALOG = json.loads((DATA / "prompt_catalog.json").read_text(encoding="utf-8"))
POLICY = Policy.from_dict(POLICY_DOC)
SEED_CARDS = [o["value"] for o in POLICY_DOC["slots"]["card"]["options"]]


def _to(schema_or_spec: dict[str, Any]) -> dict[str, Any]:
    return schema_or_spec["properties"]["to"]


def _make_payment_spec(specs: list[dict[str, Any]]) -> dict[str, Any]:
    spec = next(t for t in specs if t["name"] == "make_payment")
    return spec["config"]["inlineFunction"]


def test_default_cards_are_the_seed_options_with_their_english_spoken_names() -> None:
    assert list(tc.DEFAULT_CARDS) == SEED_CARDS == ["red", "purple", "rewards"]
    names = {e["key"]: e["text"] for e in CATALOG["entries"] if e["lang"] == "en-US"}
    for value, name in tc.DEFAULT_CARDS.items():
        assert names[f"option.card.{value}"] == name


def test_card_values_are_the_policy_card_options() -> None:
    assert tc.card_values(POLICY) == tuple(SEED_CARDS)


def test_card_values_without_a_card_slot_is_empty() -> None:
    class NoCard:
        def slot(self, name: str) -> Any:
            raise KeyError(name)

    assert tc.card_values(NoCard()) == ()


def test_schema_enum_equals_the_policy_options_and_to_is_optional() -> None:
    schema = tc.tool_schemas(cards=tc.card_values(POLICY))["make_payment"]
    assert _to(schema)["enum"] == SEED_CARDS
    assert _to(schema)["type"] == "string"
    assert schema["required"] == ["from_account", "amount"]
    assert "to" not in schema["required"]


def test_default_schema_carries_the_default_cards() -> None:
    assert _to(tc.tool_schemas()["make_payment"])["enum"] == list(tc.DEFAULT_CARDS)


def test_description_lists_value_equals_spoken_name_and_a_separate_confirmation() -> None:
    desc = _to(tc.tool_schemas()["make_payment"])["description"]
    for value, name in tc.DEFAULT_CARDS.items():
        assert f"{value} = {name}" in desc
    assert "confirm" in desc.lower()
    # A plain collection (no spoken names) still lists each value.
    desc = _to(tc.tool_schemas(cards=["blue"])["make_payment"])["description"]
    assert "blue" in desc


@pytest.mark.parametrize("card", SEED_CARDS)
def test_each_card_is_valid(card: str) -> None:
    args = {"from_account": "checking", "to": card, "amount": "25.00"}
    assert tc.validate_args("make_payment", args, cards=tc.card_values(POLICY)) == []


def test_to_may_be_left_out() -> None:
    assert tc.validate_args("make_payment", {"from_account": "checking", "amount": "25.00"}) == []


@pytest.mark.parametrize("to", ["credit_card", "checking", "Red", "", "blue"])
def test_other_destinations_are_refused(to: str) -> None:
    errors = tc.validate_args("make_payment", {"from_account": "checking", "to": to, "amount": "1"})
    assert errors == ["$.to: not one of the allowed values"]


def test_custom_cards_replace_the_defaults() -> None:
    args = {"from_account": "checking", "to": "blue", "amount": "1"}
    assert tc.validate_args("make_payment", args, cards=["blue"]) == []
    assert tc.validate_args("make_payment", {**args, "to": "red"}, cards=["blue"])
    assert tc.validate_args("make_payment", {**args, "to": "red"}, cards=[])  # no cards: fail closed


def test_inline_specs_carry_the_enum_and_stay_sanitized() -> None:
    fn = _make_payment_spec(tc.inline_tool_specs(cards={"blue": "OwlBank Blue Card"}))
    assert _to(fn["inputSchema"])["enum"] == ["blue"]
    assert "blue = OwlBank Blue Card" in _to(fn["inputSchema"])["description"]
    assert fn["inputSchema"]["required"] == ["from_account", "amount"]
    Draft202012Validator.check_schema(fn["inputSchema"])
    assert "confirm" in fn["description"].lower()


def test_bridge_agent_tool_specs_carry_the_enum() -> None:
    from tac_bridge import agent_client_strands as acs

    by_name = {t.tool_spec["name"]: t.tool_spec for t in acs.inline_tools()}
    assert _to(by_name["make_payment"]["inputSchema"]["json"])["enum"] == list(tc.DEFAULT_CARDS)
    by_name = {t.tool_spec["name"]: t.tool_spec
               for t in acs.inline_tools(cards={"blue": "OwlBank Blue Card"})}
    assert _to(by_name["make_payment"]["inputSchema"]["json"])["enum"] == ["blue"]


def test_strands_client_offers_the_cards_it_is_given() -> None:
    from tac_bridge import agent_client_strands as acs

    settings = acs.StrandsSettings(models=(acs.ModelEntry(id="m", max_tokens=10, thinking=next(iter(acs.THINKING_PARAMS)),
                                                          prompt_caching=False),), region="us-east-1")
    client = acs.StrandsAgentClient(settings, cards={"blue": "OwlBank Blue Card"})
    spec = next(t.tool_spec for t in client._inline if t.tool_spec["name"] == "make_payment")
    assert _to(spec["inputSchema"]["json"])["enum"] == ["blue"]
