"""Unit tests for owlbank.tools_contract."""

from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema import Draft202012Validator

from owlbank import tools_contract as tc

ID_WORDS = ("customer", "account_id", "account_number", "id", "token", "level", "ssn", "zip", "otp")


def test_inline_tool_spec_shape() -> None:
    tools = tc.inline_tool_specs()
    names = [t["name"] for t in tools]
    assert names == list(tc.INLINE_TOOLS)
    assert set(names) == {"get_balance", "list_recent_transactions", "change_address",
                          "make_payment", "transfer_funds", "authenticate", "escalate", "end_call"}
    for t in tools:
        assert t["type"] == "inline_function"
        fn = t["config"]["inlineFunction"]
        assert fn["description"]
        assert fn["inputSchema"]["type"] == "object"
        # The model-facing schema carries no booleans; the full schema still does.
        assert "additionalProperties" not in fn["inputSchema"]
        assert tc.tool_schemas()[t["name"]]["additionalProperties"] is False
    json.dumps(tools)


def test_no_id_token_or_level_inputs_anywhere() -> None:
    for name, schema in tc.tool_schemas().items():
        for prop in schema["properties"]:
            low = prop.lower()
            assert low not in ID_WORDS and not low.endswith("_id"), (name, prop)
            assert "customer" not in low and "token" not in low and "level" not in low


def test_gateway_tools_not_inline() -> None:
    assert tc.GATEWAY_LEVEL0_TOOLS == ("get_branch_hours", "faq_lookup")
    assert not set(tc.GATEWAY_LEVEL0_TOOLS) & set(tc.INLINE_TOOLS)


def test_get_balance_schema_matches_handover() -> None:
    s = tc.tool_schemas()["get_balance"]
    assert s["properties"]["account"]["enum"] == ["checking", "savings"]
    assert s["required"] == ["account"]


def test_auth_level_insufficient_is_minimal() -> None:
    assert tc.auth_level_insufficient("make_payment") == {
        "error": "AUTH_LEVEL_INSUFFICIENT", "tool": "make_payment"}


@pytest.mark.parametrize("tool,args", [
    ("get_balance", {"account": "checking"}),
    ("list_recent_transactions", {"account": "savings"}),
    ("list_recent_transactions", {"account": "savings", "limit": 10}),
    ("change_address", {"street": "1 Main St", "city": "Austin", "state": "TX",
                        "postal_code": "78701-1234"}),
    ("make_payment", {"from_account": "checking", "to": "red", "amount": "250.00"}),
    ("make_payment", {"from_account": "checking", "amount": "250.00"}),
    ("transfer_funds", {"from_account": "checking", "to_account": "savings", "amount": "5"}),
    ("authenticate", {"tool": "make_payment"}),
    ("escalate", {"reason": "fraud_suspected", "summary": "Caller reports fraud."}),
    ("end_call", {}),
])
def test_valid_args(tool: str, args: dict) -> None:
    assert tc.validate_args(tool, args) == []


@pytest.mark.parametrize("tool,args", [
    ("get_balance", {}),
    ("get_balance", {"account": "brokerage"}),
    ("get_balance", {"account": "checking", "customer_id": "C-1"}),
    ("get_balance", "checking"),
    ("list_recent_transactions", {"account": "checking", "limit": 11}),
    ("list_recent_transactions", {"account": "checking", "limit": 0}),
    ("list_recent_transactions", {"account": "checking", "limit": True}),
    ("change_address", {"street": "", "city": "A", "state": "TX", "postal_code": "78701"}),
    ("change_address", {"street": "x", "city": "A", "state": "Texas", "postal_code": "78701"}),
    ("change_address", {"street": "x", "city": "A", "state": "TX", "postal_code": "7870"}),
    ("make_payment", {"from_account": "checking", "to": "savings", "amount": "1.00"}),
    ("make_payment", {"from_account": "checking", "to": "credit_card", "amount": "1.00"}),
    ("make_payment", {"from_account": "checking", "to": "red", "amount": 1.0}),
    ("make_payment", {"from_account": "checking", "to": "red", "amount": "1.001"}),
    ("make_payment", {"from_account": "checking", "to": "red", "amount": "0"}),
    ("make_payment", {"from_account": "checking", "to": "red", "amount": "-5"}),
    ("make_payment", {"from_account": "checking", "to": "red", "amount": "1e3"}),
    ("transfer_funds", {"from_account": "checking", "to_account": "checking", "amount": "5"}),
    ("authenticate", {"tool": "make_payment", "level": 3}),
    ("authenticate", {"tool": "escalate"}),
    ("escalate", {"reason": "bogus", "summary": "x"}),
    ("escalate", {"reason": "complaint", "summary": "x" * 1001}),
    ("escalate", {"reason": "complaint", "summary": "   "}),
    ("end_call", {"reason": "caller_done"}),
    ("end_call", "bye"),
    ("nope", {}),
])
def test_invalid_args(tool: str, args: object) -> None:
    errors = tc.validate_args(tool, args)
    assert errors
    assert "C-1" not in " ".join(errors)


def test_custom_reasons_list() -> None:
    assert tc.validate_args("escalate", {"reason": "billing", "summary": "x"},
                            reasons=["billing"]) == []
    tool = next(t for t in tc.inline_tool_specs(["billing"]) if t["name"] == "escalate")
    assert tool["config"]["inlineFunction"]["inputSchema"]["properties"]["reason"]["enum"] == ["billing"]


# The model-facing schemas hold only strings, objects and arrays of strings (they were first
# bound into a CloudFormation Json property, which turns every scalar into a string).


def _scalars(node: object, path: str = "$") -> list[tuple[str, object]]:
    """Every non-string leaf in a JSON tree, with its path."""
    if isinstance(node, dict):
        return [x for k, v in node.items() for x in _scalars(v, f"{path}.{k}")]
    if isinstance(node, list):
        return [x for i, v in enumerate(node) for x in _scalars(v, f"{path}[{i}]")]
    return [] if isinstance(node, str) else [(path, node)]


REASON_SETS = [tc.DEFAULT_ESCALATION_REASONS, ["billing"], ["a", "b", "c"]]


@pytest.mark.parametrize("reasons", REASON_SETS)
def test_inline_tool_specs_have_no_bool_or_number_anywhere(reasons) -> None:
    for tool in tc.inline_tool_specs(reasons):
        assert _scalars(tool) == [], tool["name"]


@pytest.mark.parametrize("reasons", REASON_SETS)
def test_inline_tool_schemas_are_valid_draft_2020_12(reasons) -> None:
    for tool in tc.inline_tool_specs(reasons):
        schema = tool["config"]["inlineFunction"]["inputSchema"]
        Draft202012Validator.check_schema(schema)
    for schema in tc.tool_schemas(reasons).values():  # the full schemas too
        Draft202012Validator.check_schema(schema)


def test_inline_tool_schemas_keep_the_model_facing_keywords() -> None:
    for tool in tc.inline_tool_specs():
        sanitized = tool["config"]["inlineFunction"]["inputSchema"]
        full = tc.tool_schemas()[tool["name"]]
        assert sanitized["type"] == "object"
        assert sanitized["required"] == full["required"]
        assert set(sanitized["properties"]) == set(full["properties"])
        for key, prop in full["properties"].items():
            kept = sanitized["properties"][key]
            for kw in ("type", "enum", "pattern", "description"):
                assert kept.get(kw) == prop.get(kw), (tool["name"], key, kw)


def test_limits_are_stated_in_the_property_descriptions() -> None:
    """Dropping the numeric keywords must not hide the limits from the model."""
    for tool in tc.inline_tool_specs():
        full = tc.tool_schemas()[tool["name"]]
        props = tool["config"]["inlineFunction"]["inputSchema"]["properties"]
        for key, prop in full["properties"].items():
            for kw in ("minimum", "maximum", "minLength", "maxLength"):
                if kw in prop:
                    assert str(prop[kw]) in props[key]["description"], (tool["name"], key, kw)


def test_full_schemas_keep_their_limits_for_validate_args() -> None:
    schemas = tc.tool_schemas()
    assert all(s["additionalProperties"] is False for s in schemas.values())
    lim = schemas["list_recent_transactions"]["properties"]["limit"]
    assert (lim["minimum"], lim["maximum"]) == (1, 10)
    assert schemas["escalate"]["properties"]["summary"]["maxLength"] == tc.SUMMARY_MAX_LEN
    # The sanitizer works on a copy: the source of truth is untouched.
    tc.inline_tool_specs()
    assert tc.tool_schemas()["list_recent_transactions"]["properties"]["limit"]["maximum"] == 10


def test_server_side_validation_still_enforces_what_the_model_view_drops() -> None:
    assert tc.validate_args("get_balance", {"account": "checking", "extra": "x"})
    assert tc.validate_args("list_recent_transactions", {"account": "checking", "limit": 0})
    assert tc.validate_args("list_recent_transactions", {"account": "checking", "limit": 11})
    assert tc.validate_args("change_address", {"street": "x" * 101, "city": "Austin", "state": "TX",
                                               "postal_code": "78701"})
    assert tc.validate_args("change_address", {"street": "1 Main", "city": "Austin", "state": "tx",
                                               "postal_code": "78701"})
    assert tc.validate_args("escalate", {"reason": "complaint", "summary": "x" * (tc.SUMMARY_MAX_LEN + 1)})


_JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=8),
    lambda c: st.lists(c, max_size=4) | st.dictionaries(st.text(max_size=6), c, max_size=4),
    max_leaves=25,
)


@given(_JSON)
def test_sanitize_schema_removes_every_bool_and_number(node) -> None:
    assert _scalars(tc.sanitize_schema({"root": node})) == []


@given(_JSON)
def test_sanitize_schema_is_idempotent_and_does_not_mutate(node) -> None:
    before = json.dumps(node, sort_keys=True)
    once = tc.sanitize_schema(node)
    assert json.dumps(node, sort_keys=True) == before
    assert tc.sanitize_schema(once) == once


# --------------------------------------------------------------------------- Gateway tool names


def test_gateway_tool_full_name_is_target_triple_underscore_tool() -> None:
    assert [tc.gateway_tool_full_name(t) for t in tc.GATEWAY_LEVEL0_TOOLS] == [
        "level0___get_branch_hours", "level0___faq_lookup"]


# --------------------------------------------------------------------------- reason descriptions


def _reason(schema_or_spec: dict) -> dict:
    return schema_or_spec["properties"]["reason"]


def test_card_dispute_is_a_default_reason() -> None:
    assert "card_dispute" in tc.DEFAULT_ESCALATION_REASONS
    assert "card_dispute" in _reason(tc.tool_schemas()["escalate"])["enum"]
    assert tc.validate_args("escalate", {"reason": "card_dispute", "summary": "x"}) == []


def test_escalate_reason_carries_the_descriptions() -> None:
    full = _reason(tc.tool_schemas()["escalate"])
    spec = next(t for t in tc.inline_tool_specs() if t["name"] == "escalate")
    sanitized = _reason(spec["config"]["inlineFunction"]["inputSchema"])
    assert sanitized == full  # strings and a list of strings: CloudFormation-safe as they are
    for reason, text in tc.DEFAULT_REASON_DESCRIPTIONS.items():
        assert f"{reason}: {text}" in full["description"]
    assert isinstance(full["description"], str)


def test_reason_descriptions_come_from_the_caller() -> None:
    schema = tc.tool_schemas(["billing", "other"], reason_descriptions={
        "billing": "The caller asks about a bill.", "unlisted": "Never shown."})
    reason = _reason(schema["escalate"])
    assert reason["enum"] == ["billing", "other"]
    assert "billing: The caller asks about a bill." in reason["description"]
    assert "unlisted" not in reason["description"] and "other:" not in reason["description"]


def test_no_descriptions_means_no_reason_description() -> None:
    reason = _reason(tc.tool_schemas(reason_descriptions={})["escalate"])
    assert "description" not in reason
    assert reason["enum"] == list(tc.DEFAULT_ESCALATION_REASONS)


def test_policy_descriptions_build_the_same_schema() -> None:
    from pathlib import Path

    from owlbank.policy import Policy

    seed = Path(__file__).resolve().parents[2] / "data" / "policy.json"
    policy = Policy.from_dict(json.loads(seed.read_text(encoding="utf-8")))
    assert tc.tool_schemas(reason_descriptions=policy.reason_descriptions) == tc.tool_schemas()


# --------------------------------------------------------------------------- end_call


def test_end_call_is_a_handoff_tool_with_an_empty_schema() -> None:
    assert tc.END_CALL == "end_call" and tc.CALLER_DONE == "caller_done"
    assert tc.END_CALL in tc.HANDOFF_TOOLS and tc.END_CALL in tc.INLINE_TOOLS
    assert tc.END_CALL not in tc.BANK_TOOLS and tc.END_CALL not in tc.PROTECTED_TOOLS
    assert tc.tool_schemas()[tc.END_CALL] == {"type": "object", "properties": {}, "required": [],
                                              "additionalProperties": False}
    spec = next(t for t in tc.inline_tool_specs() if t["name"] == tc.END_CALL)
    desc = spec["config"]["inlineFunction"]["description"]
    assert desc.startswith("End the call after the caller has said they are done")
    assert "without speaking first" in desc and "in the middle of a task" in desc
    assert spec["config"]["inlineFunction"]["inputSchema"] == {"type": "object", "properties": {},
                                                               "required": []}


def test_end_call_is_not_an_authenticate_target() -> None:
    assert tc.validate_args("authenticate", {"tool": tc.END_CALL})


def test_error_codes_are_upper_snake_strings() -> None:
    assert (tc.CALL_IN_PROGRESS, tc.TOOL_DISABLED, tc.VERIFICATION_CLOSED) == (
        "CALL_IN_PROGRESS", "TOOL_DISABLED", "VERIFICATION_CLOSED")


def test_exclude_drops_end_call_and_keeps_the_rest_in_order() -> None:
    full = tc.inline_tool_specs()
    without = tc.inline_tool_specs(exclude={tc.END_CALL})
    assert [t["name"] for t in without] == [n for n in tc.INLINE_TOOLS if n != tc.END_CALL]
    assert without == [t for t in full if t["name"] != tc.END_CALL]
    assert tc.inline_tool_specs(exclude=()) == full
    assert tc.inline_tool_specs(exclude={"no_such_tool"}) == full
