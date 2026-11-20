"""The seed's per-field merge of the transcript redaction setting."""

from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank.pii import COMPREHEND_PII_ENTITY_TYPES, RedactionConfig, parse_redaction_config
from owlbank.policy import Policy
from owlbank.store import MemoryStore
from scripts import seed
from scripts._common import DATA_DIR

POLICY = json.loads((DATA_DIR / "policy.json").read_text(encoding="utf-8"))
_FIXTURES = json.loads((DATA_DIR / "fixtures.json").read_text(encoding="utf-8"))
_CATALOG = json.loads((DATA_DIR / "prompt_catalog.json").read_text(encoding="utf-8"))
SCHEMA = tuple(f.name for f in dataclasses.fields(RedactionConfig))
#: Fields the schema no longer has (``apply_to``) or never had.
REMOVED = ("apply_to", "junk")
MISSING = object()
STORE_ONLY = ("pk", "rev")
PK = "redaction"

json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-10**6, 10**6) | st.text(max_size=6),
    lambda c: st.lists(c, max_size=3) | st.dictionaries(st.text(max_size=5), c, max_size=3),
    max_leaves=5)
types = st.lists(st.sampled_from(sorted(COMPREHEND_PII_ENTITY_TYPES)), max_size=4, unique=True)
#: Values a live field may hold: any JSON, plus the shapes the console writes (an empty list too).
field_values: st.SearchStrategy[Any] = json_values | st.booleans() | types | st.just([])
revs = st.just(MISSING) | st.integers(0, 1000) | json_values


@st.composite
def seed_docs(draw: st.DrawFn) -> dict[str, Any]:
    """policy.json with an arbitrary valid ``redaction`` section (or none)."""
    doc = copy.deepcopy(POLICY)
    if draw(st.integers(0, 9)) == 0:
        doc.pop("redaction")
        return doc
    enabled = draw(st.booleans())
    chosen = draw(types.filter(lambda t: bool(t) or not enabled))
    doc["redaction"] = {"enabled": enabled, "entity_types": chosen,
                        "min_confidence": draw(st.sampled_from([0, 0.25, 0.5, 1])),
                        "mask": draw(st.sampled_from(["[REDACTED:{type}]", "***", "<{type}>"])),
                        "on_failure": draw(st.sampled_from(["withhold", "allow"]))}
    return doc


@st.composite
def live_items(draw: st.DrawFn) -> dict[str, Any] | None:
    """The live item absent or present; each field (schema, removed, unknown) present, absent or
    any JSON value."""
    if not draw(st.booleans()):
        return None
    item: dict[str, Any] = {"pk": PK}
    for f in (*SCHEMA, *REMOVED):
        if draw(st.booleans()):
            item[f] = draw(field_values)
    if (rev := draw(revs)) is not MISSING:
        item["rev"] = rev
    return item


def body(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if k not in STORE_ONLY}


def expected_rev(before: dict[str, Any] | None) -> int:
    rev = None if before is None else before.get("rev")
    return rev + 1 if isinstance(rev, int) and not isinstance(rev, bool) else 1


def off_by(live: dict[str, Any], seeded: dict[str, Any]) -> list[str]:
    """The fields that make a live item phase 1's "off": a live ``apply_to: []``, and
    an empty ``entity_types`` not paired with ``enabled: false``, judged on the live schema fields
    over the seed file's."""
    merged = {**body(seeded), **{f: v for f, v in live.items() if f in SCHEMA}}
    return ((["apply_to"] if live.get("apply_to") == [] else [])
            + (["entity_types"] if merged.get("entity_types") == [] and merged.get("enabled") is not False
               else []))


def phase1_off(live: dict[str, Any], seeded: dict[str, Any]) -> bool:
    return bool(off_by(live, seeded))


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(live=live_items(), doc=seed_docs(), overwrite=st.booleans())
def test_property_redaction_merge(live: dict[str, Any] | None, doc: dict[str, Any],
                                        overwrite: bool) -> None:
    seeded = Policy.to_items(doc)
    merge = seed.merge_redaction(seeded, {PK: live}, overwrite=overwrite)
    assert [i["pk"] for i in merge.items] == [i["pk"] for i in seeded]  # same items, same order
    got = {i["pk"]: i for i in merge.items}
    for s in seeded:
        if s["pk"] != PK:
            assert got[s["pk"]] == s  # not the redaction item: the seed file's, untouched
    if "redaction" not in doc:
        assert merge.items == seeded and merge.dropped == []
        return
    s = next(i for i in seeded if i["pk"] == PK)
    out = got[PK]
    assert out["rev"] == expected_rev(live)
    assert not set(body(out)) & set(REMOVED)  # removed fields never survive
    assert set(body(out)) <= set(SCHEMA)
    if live is None:
        assert body(out) == body(s) and merge.new
        return
    assert sorted(f for f, _ in merge.dropped) == sorted(f"redaction.{f}" for f in body(live)
                                                         if f not in SCHEMA)
    if overwrite:
        assert body(out) == body(s) and not merge.switched_off and merge.switched_off_by == []
        return
    off = phase1_off(live, s)
    assert merge.switched_off == off
    assert merge.switched_off_by == off_by(live, s)
    for f in SCHEMA:
        if off and f == "enabled":
            assert out[f] is False
        elif off and f == "entity_types":
            assert out[f] == s["entity_types"]
        elif f in live:
            assert out[f] == live[f]  # every live schema field kept
        else:
            assert out[f] == s[f]  # every absent field takes the seed value
    # every kept value that differs from the seed file is reported
    for f in SCHEMA:
        if f in live and live[f] != s[f] and not (off and f in ("enabled", "entity_types")):
            assert (f"redaction.{f}", live[f], s[f]) in merge.differences
    # a kept value is reported invalid iff the merged setting fails strict validation
    try:
        parse_redaction_config(body(out))
    except ValueError:
        assert merge.invalid and all(p.startswith("redaction.") for p in merge.invalid)
    else:
        assert merge.invalid == []


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(live=live_items(), doc=seed_docs(), overwrite=st.booleans())
def test_property_redaction_idempotent_apart_from_rev(
        live: dict[str, Any] | None, doc: dict[str, Any], overwrite: bool) -> None:
    seeded = Policy.to_items(doc)
    first = seed.merge_redaction(seeded, {PK: live}, overwrite=overwrite)
    a = {i["pk"]: i for i in first.items}
    second = seed.merge_redaction(seeded, {PK: a.get(PK)}, overwrite=overwrite)
    b = {i["pk"]: i for i in second.items}
    for pk in a:
        if pk == PK:
            assert body(a[pk]) == body(b[pk]) and b[pk]["rev"] == a[pk]["rev"] + 1
        else:
            assert a[pk] == b[pk]
    assert second.added == [] and second.dropped == [] and not second.switched_off
    assert second.switched_off_by == []


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(live=live_items())
def test_property_a_valid_phase1_item_loads_as_its_operator_left_it(live: dict[str, Any] | None) -> None:
    """End to end through the store: a phase 1 item whose own fields were valid reseeds into a
    setting that loads strictly, redacts nothing iff it redacted nothing before, and keeps
    its tuned fields. A phase 1 ``apply_to: []`` redacted nothing too."""
    if live is None:
        return
    phase1 = {k: v for k, v in live.items() if k != "enabled"}
    store = MemoryStore()
    plan = seed.build_plan(_FIXTURES, POLICY, _CATALOG, stage="prod")
    seed.apply_plan(plan, store)
    store.put("policy", phase1)
    plan = seed.build_plan(_FIXTURES, POLICY, _CATALOG, stage="prod")
    plan.policy = seed.merge_redaction(plan.policy, seed.read_live_redaction(store), overwrite=False).items
    seed.apply_plan(plan, store)
    stored = store.get("policy", {"pk": PK})
    assert stored is not None and "apply_to" not in stored and "junk" not in stored
    own = {k: v for k, v in phase1.items() if k in SCHEMA}
    try:
        parse_redaction_config({**own, "enabled": False})
    except ValueError:
        return  # an invalid live value is kept and reported (another test)
    cfg = parse_redaction_config(body(stored))
    assert cfg.enabled is (phase1.get("entity_types") != [] and phase1.get("apply_to") != [])
    assert Policy.from_store(store).redaction == cfg
