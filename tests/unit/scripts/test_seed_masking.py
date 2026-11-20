"""scripts/seed.py keeps tuned masking values unless --overwrite-masking."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank.masking_config import MASKING_FIELDS
from owlbank.store import MemoryStore
from scripts import seed
from scripts._common import DATA_DIR
from tests.unit.scripts.test_seed import TABLE_ARGS, dynamo, run  # noqa: F401 (fixture)

MISSING = object()


def load(name: str) -> Any:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


POLICY = load("policy.json")
FIXTURES = load("fixtures.json")
CATALOG = load("prompt_catalog.json")
TOOLS = list(POLICY["tools"])
#: The seed file's masking values (set; read here so the tests don't pin them).
SEED_AFTER_MS = POLICY["masking"]["after_ms"]
SEED_TOOL_AFTER_MS = POLICY["masking"]["tool_after_ms"]


def plan_for(policy_doc: Any) -> seed.SeedPlan:
    return seed.build_plan(FIXTURES, policy_doc, CATALOG, stage="prod")


def reseed_policy(store: MemoryStore, policy_doc: Any, *, overwrite: bool) -> seed.MaskingMerge:
    """What ``main`` does for the policy table: read live, merge, write."""
    plan = plan_for(policy_doc)
    merge = seed.merge_masking(plan.policy, seed.read_live_masking(store, plan.policy),
                               overwrite=overwrite)
    plan.policy = merge.items
    seed.apply_plan(plan, store)
    return merge


def policy_items(store: MemoryStore, doc: Any) -> dict[str, dict[str, Any] | None]:
    return {i["pk"]: store.get("policy", {"pk": i["pk"]}) for i in plan_for(doc).policy}


# --------------------------------------------------------------------------- strategies

delays = st.integers(0, 10000)
valid_masking = st.fixed_dictionaries({}, optional={
    "after_ms": delays, "tool_after_ms": delays, "handoff_wait_ms": delays,
    "end_mode": st.sampled_from(["wait", "immediate"]), "chars_per_s": st.integers(8, 30),
    "silence_timeout_ms": delays, "silence_max_repeats": st.integers(0, 5),
    "silence_action": st.sampled_from(["escalate", "hangup"])})
valid_mask = st.none() | st.fixed_dictionaries(
    {"phrase_key": st.sampled_from(["mask.get_balance", "mask.change_address", "mask.x.y"])},
    optional={"after_ms": delays, "handoff_wait_ms": delays})


@st.composite
def seed_docs(draw: st.DrawFn) -> dict[str, Any]:
    """policy.json with arbitrary valid masking values, tool masks and version."""
    doc = copy.deepcopy(POLICY)
    doc["version"] = draw(st.integers(0, 10**6))
    doc["masking"] = draw(valid_masking)
    for name in TOOLS:
        mask = draw(valid_mask)
        if mask is None:
            doc["tools"][name].pop("mask", None)
        else:
            doc["tools"][name]["mask"] = mask
    return doc


# Live values may be anything the store holds: wrong types, out of range, junk keys.
junk = st.recursive(st.none() | st.booleans() | st.integers(-10**6, 10**6) | st.text(max_size=8),
                    lambda c: st.lists(c, max_size=3) | st.dictionaries(st.text(max_size=6), c, max_size=3),
                    max_leaves=6)
live_masking = st.dictionaries(st.sampled_from(MASKING_FIELDS), junk | delays, max_size=len(MASKING_FIELDS))
live_mask = st.just(MISSING) | junk | valid_mask
revs = st.just(MISSING) | st.integers(0, 1000)


@st.composite
def live_tables(draw: st.DrawFn) -> dict[str, dict[str, Any] | None]:
    live: dict[str, dict[str, Any] | None] = {}
    if draw(st.booleans()):
        item: dict[str, Any] = {"pk": "masking", **draw(live_masking)}
        if (rev := draw(revs)) is not MISSING:
            item["rev"] = rev
        live["masking"] = item
    for name in TOOLS:
        if draw(st.integers(0, 4)) == 0:
            continue  # this tool isn't in the table yet
        tool: dict[str, Any] = {"pk": f"tool#{name}", "min_level": draw(st.integers(0, 3)),
                                "junk": draw(junk)}
        if (mask := draw(live_mask)) is not MISSING:
            tool["mask"] = mask
        if (rev := draw(revs)) is not MISSING:
            tool["rev"] = rev
        live[f"tool#{name}"] = tool
    return live


def _masking_view(item: dict[str, Any] | None, pk: str) -> dict[str, Any]:
    if item is None:
        return {}
    names = MASKING_FIELDS if pk == "masking" else ("mask",)
    return {n: item[n] for n in names if n in item}


def _rest(item: dict[str, Any], pk: str) -> dict[str, Any]:
    names = (*MASKING_FIELDS, "rev") if pk == "masking" else ("mask", "rev")
    return {k: v for k, v in item.items() if k not in names}




@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(live=live_tables(), doc=seed_docs(), overwrite=st.booleans())
def test_property_reseed_keeps_masking_fields(live: dict[str, Any], doc: dict[str, Any],
                                                   overwrite: bool) -> None:
    store = MemoryStore()
    for item in live.values():
        store.put("policy", item)
    merge = reseed_policy(store, doc, overwrite=overwrite)
    after = policy_items(store, doc)
    for seeded in plan_for(doc).policy:
        pk = seeded["pk"]
        got = after[pk]
        assert got is not None
        before = live.get(pk)
        if before is None or not (pk == "masking" or pk.startswith("tool#")):
            assert got == seeded  # new items (and everything not masking) are the seed file's
            continue
        # every non-masking field equals the seed file's
        assert _rest(got, pk) == _rest(seeded, pk)
        if overwrite:
            assert _masking_view(got, pk) == _masking_view(seeded, pk)
        else:
            assert _masking_view(got, pk) == _masking_view(before, pk)
        if isinstance(before.get("rev"), int) and not isinstance(before.get("rev"), bool):
            assert got["rev"] == before["rev"] + 1
        else:
            assert "rev" not in got
    # every masking value that differs is reported
    if not overwrite:
        reported = {f for f, _, _ in merge.differences}
        for pk, before in live.items():
            seeded_item = next(i for i in plan_for(doc).policy if i["pk"] == pk)
            differs = (_masking_view(before, pk) != _masking_view(seeded_item, pk)) if pk == "masking" \
                else before.get("mask") != seeded_item.get("mask")  # mask null == absent (unmasked)
            if differs:
                prefix = "masking." if pk == "masking" else f"tools.{pk[5:]}.mask"
                assert any(f.startswith(prefix) for f in reported)


# --------------------------------------------------------------------------- examples


def tuned_store() -> MemoryStore:
    store = MemoryStore()
    reseed_policy(store, POLICY, overwrite=True)
    store.update("policy", {"pk": "masking"}, {"after_ms": 1500, "end_mode": "nonsense"})
    store.update("policy", {"pk": "tool#get_balance"},
                 {"mask": {"phrase_key": "mask.get_balance", "after_ms": 900}})
    store.update("policy", {"pk": "tool#faq_lookup"}, {"mask": {"phrase_key": "mask.faq"}})
    return store


def test_plain_seed_keeps_tuned_values_and_reports_them(capsys: pytest.CaptureFixture[str]) -> None:
    store = tuned_store()
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    assert store.get("policy", {"pk": "masking"})["after_ms"] == 1500
    assert store.get("policy", {"pk": "tool#get_balance"})["mask"]["after_ms"] == 900
    assert store.get("policy", {"pk": "tool#faq_lookup"})["mask"] == {"phrase_key": "mask.faq"}
    out = capsys.readouterr().out
    assert "KEPT" in out and "field | live | seed file" in out
    assert f"masking.after_ms | 1500 | {SEED_AFTER_MS}" in out
    assert "tools.get_balance.mask.after_ms | 900 | (not set)" in out
    assert 'tools.faq_lookup.mask | {"phrase_key": "mask.faq"} | (not set)' in out
    # an invalid live value is kept, with a warning saying how to fix it
    assert store.get("policy", {"pk": "masking"})["end_mode"] == "nonsense"
    assert "WARNING: live masking value invalid: masking.end_mode" in out and "--overwrite-masking" in out


def test_overwrite_masking_writes_the_seed_files_values(capsys: pytest.CaptureFixture[str]) -> None:
    store = tuned_store()
    assert run(["--stage", "prod", *TABLE_ARGS, "--overwrite-masking", "--apply"], store) == 0
    assert store.get("policy", {"pk": "masking"})["after_ms"] == SEED_AFTER_MS
    assert store.get("policy", {"pk": "masking"})["end_mode"] == POLICY["masking"]["end_mode"]
    assert store.get("policy", {"pk": "tool#get_balance"})["mask"] == {"phrase_key": "mask.get_balance"}
    assert "mask" not in store.get("policy", {"pk": "tool#faq_lookup"})
    assert "REPLACED" in capsys.readouterr().out


def test_dry_run_reports_differences_and_writes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    store = tuned_store()
    before = policy_items(store, POLICY)
    assert run(["--stage", "prod", *TABLE_ARGS], store) == 0
    assert policy_items(store, POLICY) == before
    out = capsys.readouterr().out
    assert out.startswith("DRY RUN") and f"masking.after_ms | 1500 | {SEED_AFTER_MS}" in out
    # the printed plan shows what an apply would write: the live value
    policy_line = next(ln for ln in out.splitlines() if ln.startswith('{"after_ms"'))
    assert json.loads(policy_line)["after_ms"] == 1500


def test_first_seed_and_new_tools_get_the_seed_files_values(capsys: pytest.CaptureFixture[str]) -> None:
    store = MemoryStore()
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    assert store.get("policy", {"pk": "masking"})["after_ms"] == POLICY["masking"]["after_ms"]
    assert store.get("policy", {"pk": "tool#get_balance"})["mask"] == POLICY["tools"]["get_balance"]["mask"]
    assert "the table matches policy.json" in capsys.readouterr().out


def test_a_live_tool_not_in_the_seed_file_is_left_alone() -> None:
    store = tuned_store()
    store.put("policy", {"pk": "tool#retired", "mask": {"phrase_key": "mask.retired"}})
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    assert store.get("policy", {"pk": "tool#retired"}) == {"pk": "tool#retired",
                                                          "mask": {"phrase_key": "mask.retired"}}


def test_dry_run_without_a_policy_table_name_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--stage", "prod"], MemoryStore()) == 0
    assert "wasn't read (no policy table name)" in capsys.readouterr().out


def test_kept_values_round_trip_through_dynamodb(dynamo: Any) -> None:  # noqa: F811
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], dynamo) == 0
    dynamo.update("policy", {"pk": "masking"}, {"chars_per_s": "15.5", "after_ms": 1800, "rev": 4})
    assert run(["--stage", "prod", *TABLE_ARGS, "--reseed", "--apply"], dynamo) == 0
    item = dynamo.get("policy", {"pk": "masking"})
    assert (item["chars_per_s"], item["after_ms"], item["rev"]) == ("15.5", 1800, 5)
