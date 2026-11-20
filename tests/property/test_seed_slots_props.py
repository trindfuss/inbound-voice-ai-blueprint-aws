"""The seed's per-field merge of the ``slot_resolution`` and
``slot#<name>`` items and of the catalog's ``option.*`` entries. The ``redaction`` half is in
``test_seed_redaction_props.py``."""

from __future__ import annotations

import copy
import json
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank.policy import Policy
from owlbank.catalog import PromptCatalog
from owlbank.slot_policy import (
    CONFIRM_POLICIES,
    FLOWS,
    RESOLUTION_FIELDS,
    RESOLUTION_ITEM,
    SETTING_FIELDS,
    SHAPE_FIELDS,
)
from owlbank.store import MemoryStore
from scripts import seed
from scripts._common import DATA_DIR

POLICY = json.loads((DATA_DIR / "policy.json").read_text(encoding="utf-8"))
_FIXTURES = json.loads((DATA_DIR / "fixtures.json").read_text(encoding="utf-8"))
_CATALOG = json.loads((DATA_DIR / "prompt_catalog.json").read_text(encoding="utf-8"))
SLOT_NAMES = list(POLICY["slots"])
FACTORS = set(POLICY["factors"])
FIELDS = (*SETTING_FIELDS, *SHAPE_FIELDS, "flow", "mode", "confirm", "options", "junk")
RES_FIELDS = (*RESOLUTION_FIELDS, "junk")
MERGED = (RESOLUTION_ITEM, *(f"slot#{n}" for n in SLOT_NAMES))
OPTION_KEYS = [(e["key"], e["lang"]) for e in _CATALOG["entries"] if e["key"].startswith("option.")]
MISSING = object()
STORE_ONLY = ("pk", "rev")

# Any JSON value the table could hold (no floats: DynamoDB has none, the store takes no float).
json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-10**6, 10**6) | st.text(max_size=6),
    lambda c: st.lists(c, max_size=3) | st.dictionaries(st.text(max_size=5), c, max_size=3),
    max_leaves=5)
field_values = st.booleans() | json_values
revs = st.just(MISSING) | st.integers(0, 1000) | json_values


@st.composite
def seed_docs(draw: st.DrawFn) -> dict[str, Any]:
    """policy.json with arbitrary valid slot settings and ``slot_resolution``."""
    doc = copy.deepcopy(POLICY)
    for name in SLOT_NAMES:
        entry = doc["slots"][name]
        for f in SETTING_FIELDS:
            if draw(st.booleans()):
                entry[f] = draw(st.booleans())
            else:
                entry.pop(f, None)
        if name not in FACTORS:
            entry["maxRetries"] = draw(st.integers(0, 3))
        entry["confirm"] = draw(st.sampled_from(CONFIRM_POLICIES))
        modes = ["deterministic", "assisted"]
        if entry.get("slotType") == "choice" and not entry.get("sensitive", False):
            modes.append("generative")
        if (mode := draw(st.sampled_from([None, *modes]))) is None:
            entry.pop("mode", None)
        else:
            entry["mode"] = mode
    doc["slot_resolution"] = {
        "default_mode": draw(st.sampled_from(["deterministic", "assisted", "generative"])),
        "flows": {f: {"mode": m} for f, m in draw(st.dictionaries(
            st.sampled_from(FLOWS), st.sampled_from(["deterministic", "assisted", "generative"]),
            max_size=3)).items()},
        "generative_enabled": draw(st.booleans()),
        "resolver_budget_ms": draw(st.integers(500, 4000)),
    }
    return doc


@st.composite
def live_tables(draw: st.DrawFn) -> dict[str, dict[str, Any] | None]:
    """Each slot item present or absent; each field present, absent or any JSON value."""
    live: dict[str, dict[str, Any] | None] = {}
    for pk in MERGED:
        if not draw(st.booleans()):
            live[pk] = None
            continue
        item: dict[str, Any] = {"pk": pk}
        for f in RES_FIELDS if pk == RESOLUTION_ITEM else FIELDS:
            if draw(st.booleans()):
                item[f] = draw(field_values)
        if (rev := draw(revs)) is not MISSING:
            item["rev"] = rev
        live[pk] = item
    return live


def body(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if k not in STORE_ONLY}


def by_pk(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {i["pk"]: i for i in items}


def expected_rev(before: dict[str, Any] | None) -> int:
    rev = None if before is None else before.get("rev")
    return rev + 1 if isinstance(rev, int) and not isinstance(rev, bool) else 1


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(live=live_tables(), doc=seed_docs(), overwrite=st.booleans())
def test_property_slot_merge(live: dict[str, Any], doc: dict[str, Any], overwrite: bool) -> None:
    seeded = Policy.to_items(doc)
    merge = seed.merge_slots(seeded, live, overwrite=overwrite)
    got = by_pk(merge.items)
    assert [i["pk"] for i in merge.items] == [i["pk"] for i in seeded]  # same items, same order
    for s in seeded:
        pk = s["pk"]
        if pk not in MERGED:
            assert got[pk] == s  # not merged: the seed file's, untouched
            continue
        before = live.get(pk)
        out = got[pk]
        assert out["rev"] == expected_rev(before)
        if overwrite or before is None:
            assert body(out) == body(s)
            continue
        for f, v in body(before).items():
            assert out[f] == v  # every live field kept
        for f, v in body(s).items():
            if f not in before:
                assert out[f] == v  # every absent field takes the seed value
        assert set(body(out)) == set(body(before)) | set(body(s))
        # every kept value that differs from the seed file is reported
        where = RESOLUTION_ITEM if pk == RESOLUTION_ITEM else f"slots.{pk[5:]}"
        for f, v in body(before).items():
            if body(s).get(f, seed.NOT_SET) != v:
                assert (f"{where}.{f}", v, body(s).get(f, seed.NOT_SET)) in merge.differences
    # problems name a field, never carry a value
    for problem in merge.invalid:
        assert problem.startswith(("slot#", f"{RESOLUTION_ITEM}."))


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(live=live_tables(), doc=seed_docs(), overwrite=st.booleans())
def test_property_idempotent_apart_from_rev(live: dict[str, Any], doc: dict[str, Any],
                                                  overwrite: bool) -> None:
    seeded = Policy.to_items(doc)
    first = seed.merge_slots(seeded, live, overwrite=overwrite)
    again_live = {pk: by_pk(first.items)[pk] for pk in live}
    second = seed.merge_slots(seeded, again_live, overwrite=overwrite)
    a, b = by_pk(first.items), by_pk(second.items)
    for pk in a:
        if pk in MERGED:
            assert body(a[pk]) == body(b[pk])
            assert b[pk]["rev"] == a[pk]["rev"] + 1
        else:
            assert a[pk] == b[pk]
    assert second.added == []  # nothing is left to fill


def live_entry(key: tuple[str, str]) -> st.SearchStrategy[dict[str, Any] | None]:
    """The table's entry for ``key``: absent, or present with any text (a console edit)."""
    return st.none() | st.fixed_dictionaries({
        "key": st.just(key[0]), "lang": st.just(key[1]), "text": st.text(max_size=20),
        "version": st.integers(0, 9)})


@settings(max_examples=200, deadline=None)
@given(live=st.fixed_dictionaries({k: live_entry(k) for k in OPTION_KEYS}), overwrite=st.booleans())
def test_property_option_entries(live: dict[tuple[str, str], Any], overwrite: bool) -> None:
    """An ``option.*`` entry is written where the table lacks that key and language; one whose live
    text differs is written only with ``--overwrite-slots``. Entries the table lacks entirely
    (every non-option entry here) are written exactly as before, in order."""
    items = PromptCatalog.to_items(_CATALOG)
    present = {k: v for k, v in live.items() if v is not None}
    merge = seed.merge_catalog_prompts(items, present, overwrite=False, overwrite_options=overwrite)
    seed_text = {(i["key"], i["lang"]): i.get("text") for i in items}
    edited = [k for k in OPTION_KEYS if k in present and present[k]["text"] != seed_text[k]]
    written = {(i["key"], i["lang"]) for i in merge.items}
    assert all(k not in written for k in edited) or overwrite
    assert sorted(merge.replaced if overwrite else merge.kept) == sorted(edited)
    assert sorted(k for k in merge.new if k[0].startswith("option.")) == sorted(
        k for k in OPTION_KEYS if live[k] is None)
    # (each with the hash of its text, by which a later seed tells a console edit; a
    # language new to a key with live languages also gets the key's view rev + 1)
    live_keys = {k for k, _ in present}

    def expected(i: dict[str, Any]) -> dict[str, Any]:
        if "text" not in i:
            return i
        out = {**i, seed.SEED_HASH: seed.seed_hash(i["text"])}
        if i["key"] in live_keys:
            out["rev"] = max(v.get("rev", 0) for (k, _), v in present.items() if k == i["key"]) + 1
        return out

    assert [i for i in merge.items if (i["key"], i["lang"]) not in present] == [
        expected(i) for i in items if (i["key"], i["lang"]) not in present]
    # a second run against the table the first one wrote writes no option entry (idempotent)
    after = {**{(i["key"], i["lang"]): i for i in items}, **present,
             **{(i["key"], i["lang"]): i for i in merge.items}}
    again = seed.merge_catalog_prompts(items, after, overwrite=False)
    assert again.new == [] and not any(i["key"].startswith("option.") for i in again.items)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(tuned=st.fixed_dictionaries({n: st.fixed_dictionaries({}, optional={
    **{f: st.booleans() for f in SETTING_FIELDS},
    "mode": st.sampled_from(["deterministic", "assisted"]),
    "confirm": st.sampled_from(CONFIRM_POLICIES)}) for n in SLOT_NAMES}),
    res=st.fixed_dictionaries({}, optional={
        "default_mode": st.sampled_from(["deterministic", "assisted", "generative"]),
        "resolver_budget_ms": st.integers(500, 4000)}),
    names=st.fixed_dictionaries({}, optional={k: st.sampled_from(["Red", "Roja", "Card X"])
                                              for k in OPTION_KEYS}))
def test_property_reseeding_twice_through_the_store_keeps_tuned_values(
        tuned: dict[str, dict[str, Any]], res: dict[str, Any], names: dict[tuple[str, str], str]) -> None:
    """End to end: seed, tune in the table (as the console does), reseed twice. The tuned values
    survive both reseeds, the second reseed changes nothing but ``rev``, and the table loads."""
    store = MemoryStore()

    def reseed() -> None:
        plan = seed.build_plan(_FIXTURES, POLICY, _CATALOG, stage="prod")
        merge = seed.merge_slots(plan.policy, seed.read_live_slots(store, plan.policy), overwrite=False)
        plan.policy = merge.items
        plan.catalog = seed.merge_catalog_prompts(
            plan.catalog, seed.read_live_catalog(store), overwrite=False).items
        seed.apply_plan(plan, store)

    reseed()
    for name, fields in tuned.items():
        if fields:
            store.update("policy", {"pk": f"slot#{name}"}, fields)
    if res:
        store.update("policy", {"pk": RESOLUTION_ITEM}, res)
    for (key, lang), text in names.items():
        store.update("catalog", {"key": key, "lang": lang}, {"text": text})
    reseed()
    pks = [*(f"slot#{n}" for n in SLOT_NAMES), RESOLUTION_ITEM]
    once = {pk: store.get("policy", {"pk": pk}) for pk in pks}
    reseed()
    twice = {pk: store.get("policy", {"pk": pk}) for pk in pks}
    policy = Policy.from_store(store)
    for pk in pks:
        a, b = once[pk], twice[pk]
        assert a is not None and b is not None
        assert body(a) == body(b) and b["rev"] == a["rev"] + 1
    for name in SLOT_NAMES:
        for f, v in tuned[name].items():
            assert twice[f"slot#{name}"][f] == v  # type: ignore[index]
            assert getattr(policy.slot(name), f) == v
    for f, v in res.items():
        assert twice[RESOLUTION_ITEM][f] == v  # type: ignore[index]
        assert getattr(policy.slot_resolution, f) == v
    for (key, lang), text in names.items():
        assert store.get("catalog", {"key": key, "lang": lang})["text"] == text  # type: ignore[index]
    assert policy.slot_problems == ()
