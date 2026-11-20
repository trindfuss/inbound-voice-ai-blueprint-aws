"""Policy fail-closed and missing-factor properties."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from owlbank.policy import Policy
from owlbank.store import MemoryStore

DATA = Path(__file__).resolve().parents[2] / "data"
SEED = json.loads((DATA / "policy.json").read_text(encoding="utf-8"))
SEED_POLICY = Policy.from_dict(SEED)

names = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_0123456789", min_size=1, max_size=12)


@st.composite
def policies(draw: st.DrawFn) -> dict[str, Any]:
    """Random valid policies: cumulative levels, tools at any level, one default reason."""
    factor_names = draw(st.lists(names, min_size=1, max_size=8, unique=True))
    max_level = draw(st.integers(1, len(factor_names)))
    # Split the ordered factor list into max_level non-empty cumulative prefixes.
    cuts = sorted(draw(st.lists(st.integers(1, len(factor_names) - 1), unique=True,
                                min_size=max_level - 1, max_size=max_level - 1))
                  if max_level > 1 else [])
    ends = [*cuts, len(factor_names)]
    levels = {str(i + 1): factor_names[:end] for i, end in enumerate(ends)}
    tool_names = draw(st.lists(names, max_size=6, unique=True))
    tools = {t: {"min_level": draw(st.integers(0, max_level)), "phrase_key": f"tool.{t}",
                 "kind": "agentic_inline"} for t in tool_names}
    return {
        "max_level": max_level,
        "levels": levels,
        "factors": {f: {"slotType": "digits"} for f in factor_names},
        "tools": tools,
        "escalation": {"default_reason": "unable_to_resolve",
                       "queues": {"support": {"en-US": "Support"}},
                       "reasons": {"unable_to_resolve": {"queue": "support", "priority": 0,
                                                         "phrase_key": "escalate.default"}}},
    }


@given(doc=policies(), tool=names)
def test_unknown_tool_requires_max_level(doc: dict[str, Any], tool: str) -> None:
    assume(tool not in doc["tools"])
    policy = Policy.from_dict(doc)
    assert policy.min_level(tool) == policy.max_level == doc["max_level"]


@given(tool=st.text(max_size=40))
def test_seed_unknown_tool_requires_max_level(tool: str) -> None:
    assume(tool not in SEED["tools"])
    assert SEED_POLICY.min_level(tool) == SEED_POLICY.max_level


@given(doc=policies(), data=st.data())
def test_missing_factors_subset_and_order(doc: dict[str, Any], data: st.DataObject) -> None:
    policy = Policy.from_dict(doc)
    target = data.draw(st.integers(0, policy.max_level))
    universe = list(doc["factors"]) + ["ani_match", "unrelated"]
    have = set(data.draw(st.lists(st.sampled_from(universe), max_size=10)))
    need = policy.factors(target)
    missing = policy.missing_factors(have, target)
    assert set(missing) <= set(need)
    assert set(missing) == set(need) - have
    # In policy order: missing is a subsequence of factors(target).
    it = iter(need)
    assert all(f in it for f in missing)
    if have >= set(need):
        assert missing == []


@given(doc=policies())
def test_levels_cumulative(doc: dict[str, Any]) -> None:
    policy = Policy.from_dict(doc)
    for n in range(1, policy.max_level + 1):
        assert set(policy.factors(n - 1)) <= set(policy.factors(n))


@given(reason=st.text(max_size=40))
def test_unknown_reason_routes_to_default(reason: str) -> None:
    route = SEED_POLICY.escalation_route(reason)
    if reason in SEED["escalation"]["reasons"]:
        assert route.reason == reason
    else:
        assert route == SEED_POLICY.escalation_route("unable_to_resolve")


@settings(max_examples=30)
@given(doc=policies(), tool=names)
def test_store_policy_fails_closed_too(doc: dict[str, Any], tool: str) -> None:
    assume(tool not in doc["tools"])
    store = MemoryStore()
    for item in Policy.to_items(doc):
        store.put("policy", item)
    policy = Policy.from_store(store)
    assert policy.min_level(tool) == doc["max_level"]
    for t, row in doc["tools"].items():
        assert policy.min_level(t) == row["min_level"]
