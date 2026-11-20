"""The bridge's deterministic step-up shortcut.

For any tool in {get_balance, make_payment, transfer_funds, change_address,
list_recent_transactions}, caller level 0..3 and valid or invalid arguments, the bridge hands
off for step-up without a model round trip exactly when the tool is a front-door tool
(``get_balance`` or a deterministic-confirm tool) and the level is below the tool's
``min_level`` and the arguments are valid. Otherwise behaviour is unchanged: invalid arguments
and inline-tool refusals go back to the model, and a front-door tool at its level hands off as
``tool`` or ``confirm``.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank.catalog import PromptCatalog
from owlbank.pii import FakePiiRedactor, PiiGuard
from owlbank.policy import Policy
from owlbank.segments import SegmentLog
from owlbank.tools_contract import AUTH_LEVEL_INSUFFICIENT
from tac_bridge.agent_client import TextDelta, ToolCall, TurnEnd
from tac_bridge.bridge import Bridge
from tests.component.masking_fixtures import masked_policy_doc, unmasked_policy_doc
from tests.component.test_bridge_fakes import (
    CALL_SID,
    NOW,
    SEALER,
    FakeAgent,
    FakeBackend,
    _inline_io,
    load_json,
    seeded_store,
    write_call_state,
)

POLICY = Policy.from_dict(unmasked_policy_doc())
# data/policy.json's phrases, no timer, end at once (no hold to sleep through).
MASKED = Policy.from_dict(masked_policy_doc(after_ms=0, end_mode="immediate"))
CATALOG = PromptCatalog.from_dict(load_json("prompt_catalog.json"))
FRONT_DOOR = frozenset({"get_balance", "make_payment", "transfer_funds"})
VALID: dict[str, dict[str, Any]] = {
    "get_balance": {"account": "checking"},
    "make_payment": {"from_account": "checking", "to": "red", "amount": "50.00"},
    "transfer_funds": {"from_account": "checking", "to_account": "savings", "amount": "20.00"},
    "change_address": {"street": "1 Main St", "city": "Springfield", "state": "IL",
                       "postal_code": "62701"},
    "list_recent_transactions": {"account": "checking"},
}


def invalid(args: dict[str, Any]) -> dict[str, Any]:
    """The arguments with a required one dropped."""
    out = dict(args)
    out.pop(next(iter(out)))
    return out


async def one_turn(name: str, args: dict[str, Any], level: int,
                   policy: Policy = POLICY) -> dict[str, Any]:
    store = seeded_store()
    write_call_state(store, verified=level > 0, level=max(level, 1))
    journal = SegmentLog(store, now=lambda: NOW)
    agent = FakeAgent([[ToolCall("tu_x", name, dict(args)), TurnEnd("tool_use")],
                       [TextDelta("Done."), TurnEnd("end_turn")]])
    min_level = policy.min_level(name)
    refusal = {"error": AUTH_LEVEL_INSUFFICIENT, "tool": name}
    backend = FakeBackend({name: (lambda **_: dict(refusal)) if level < min_level
                           else (lambda **_: {"tool": name, "data": {}})})
    bridge = Bridge(store=store, agent=agent, backend=backend, memory=None, catalog=CATALOG,
                    policy=policy, pii=PiiGuard.for_policy(FakePiiRedactor(), policy),
                    journal=journal, sealer=SEALER, now=lambda: NOW, io=_inline_io)
    ctx = SimpleNamespace(call_sid=CALL_SID, conversation_id="conv_p", metadata={},
                          profile_id=None, profile=None, pending_handoff_data=None)
    chunks = [chunk async for chunk in bridge.run_turn(ctx, "please")]
    data = json.loads(ctx.pending_handoff_data.handoff_data) if ctx.pending_handoff_data else None
    return {"agent": agent.calls, "handoff": data, "spoken": "".join(chunks),
            "pending": (store.get("calls", {"call_sid": CALL_SID}) or {}).get("pending_tool"),
            "backend": backend.calls}


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(name=st.sampled_from(sorted(VALID)), level=st.integers(min_value=0, max_value=3),
       valid=st.booleans())
def test_step_up_shortcut_exactly_for_front_door_tools_below_their_level(
        name: str, level: int, valid: bool) -> None:
    args = VALID[name] if valid else invalid(VALID[name])
    out = asyncio.run(one_turn(name, args, level))
    min_level = POLICY.min_level(name)
    shortcut = name in FRONT_DOOR and level < min_level and valid
    handoff = out["handoff"]
    is_step_up = handoff is not None and handoff["reason"] == "step-up"
    assert is_step_up == shortcut
    if shortcut:
        # No model round trip; the model's own call is the pending tool.
        assert len(out["agent"]) == 1
        assert handoff["tool"] == name and handoff["args"] == {"tool": name}
        assert out["pending"]["tool_use_id"] == "tu_x" and out["pending"]["name"] == name
        assert out["pending"]["args"] == args
        assert out["backend"] == []
        return
    # Otherwise unchanged.
    if not valid:
        assert handoff is None and len(out["agent"]) == 2
        (res,) = out["agent"][1]["tool_results"]
        assert res.content["error"] == "INVALID_ARGUMENTS"
    elif name in FRONT_DOOR:  # at its level: the usual hand-off
        assert len(out["agent"]) == 1
        assert handoff is not None
        assert handoff["reason"] == ("tool" if name == "get_balance" else "confirm")
        assert out["pending"]["tool_use_id"] == "tu_x"
    else:  # an inline bank tool: run by the bridge, the result (or refusal) back to the model
        assert handoff is None and len(out["agent"]) == 2
        (res,) = out["agent"][1]["tool_results"]
        if level < min_level:
            assert res.content == {"error": AUTH_LEVEL_INSUFFICIENT, "tool": name}


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(name=st.sampled_from(sorted(FRONT_DOOR)), level=st.integers(min_value=0, max_value=3),
       valid=st.booleans())
def test_a_front_door_tool_gets_its_phrase_exactly_when_it_hands_off(
        name: str, level: int, valid: bool) -> None:
    """The tool's hand-off phrase is spoken exactly when the call
    hands off at its level, or for step-up (the shortcut) if it is a deterministic-confirm
    tool; never for a balance step-up (its phrases announce the balance) and never for a refusal
    that goes back to the model (invalid arguments)."""
    args = VALID[name] if valid else invalid(VALID[name])
    out = asyncio.run(one_turn(name, args, level, MASKED))
    phrase = CATALOG.text(f"mask.{name}", "en-US").strip()
    step_up = level < MASKED.min_level(name)
    assert (out["handoff"] is not None) == valid
    assert out["spoken"].startswith(phrase) == (valid and not (step_up and name == "get_balance"))
    if valid:
        assert (out["handoff"]["reason"] == "step-up") == step_up
        if step_up:
            assert out["backend"] == []  # no prefetch on a step-up
