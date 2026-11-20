"""owlbank.policy: the ``model_failover`` section, the seed refusal and the shipped data file."""

from __future__ import annotations

import copy
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.failover.settings import Settings
from owlbank.policy import (
    MODEL_FAILOVER_ITEM,
    ModelFailoverConfig,
    Policy,
    PolicyError,
    PolicySectionError,
    parse_model_failover,
)
from owlbank.store import MemoryStore
from scripts import seed

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SEED: dict[str, Any] = json.loads((DATA / "policy.json").read_text(encoding="utf-8"))

HAIKU55 = "us.anthropic.claude-haiku-5-5-20260301-v1:0"
HAIKU45 = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET55 = "us.anthropic.claude-sonnet-5-5-20260301-v1:0"
IDS = (HAIKU55, HAIKU45, SONNET55)


def with_section(section: Any) -> dict[str, Any]:
    doc = copy.deepcopy(SEED)
    doc[MODEL_FAILOVER_ITEM] = section
    return doc


def without_section() -> dict[str, Any]:
    doc = copy.deepcopy(SEED)
    doc.pop(MODEL_FAILOVER_ITEM, None)
    return doc


def seeded(doc: dict[str, Any]) -> MemoryStore:
    store = MemoryStore()
    for item in Policy.to_items(doc):
        store.put("policy", item)
    return store


# --------------------------------------------------------------------------- defaults


def test_the_shipped_section_equals_the_built_in_defaults() -> None:
    policy = Policy.from_dict(SEED)
    assert policy.model_failover.settings_for(IDS) == Settings.for_models(IDS)


def test_an_absent_section_gives_the_defaults() -> None:
    policy = Policy.from_dict(without_section())
    assert policy.model_failover == ModelFailoverConfig({}, {})
    assert policy.model_failover.settings_for(IDS) == Settings.for_models(IDS)
    # ...also through the store
    assert Policy.from_store(seeded(without_section())).model_failover.settings_for(IDS) == Settings.for_models(IDS)


def test_the_default_numbers() -> None:
    s = Policy.from_dict(SEED).model_failover.settings_for(IDS)
    assert (s.window_calls, s.window_age_s, s.min_samples, s.miss_run) == (20, 120, 20, 3)
    assert (s.pings_per_cycle, s.recover_min_pings, s.recover_period_s) == (3, 10, 120)
    assert (s.dwell_s, s.idle_reset_s, s.unavailable_recheck_s) == (180, 600, 900)


# --------------------------------------------------------------------------- refusals

REFUSALS = [
    ("not an object", [], "model_failover"),
    ("unknown field", {"nope": 1}, "model_failover.nope"),
    ("pings 0", {"pings_per_cycle": 0}, "model_failover.pings_per_cycle"),
    ("pings 6", {"pings_per_cycle": 6}, "model_failover.pings_per_cycle"),
    ("pings float", {"pings_per_cycle": 3.0}, "model_failover.pings_per_cycle"),
    ("pings bool", {"pings_per_cycle": True}, "model_failover.pings_per_cycle"),
    ("window 0", {"window_calls": 0}, "model_failover.window_calls"),
    ("min_samples 0", {"min_samples": 0}, "model_failover.min_samples"),
    ("min_samples above window", {"window_calls": 10, "min_samples": 11}, "model_failover.min_samples"),
    ("miss_run 0", {"miss_run": 0}, "model_failover.miss_run"),
    ("recover_min_pings 0", {"recover_min_pings": 0}, "model_failover.recover_min_pings"),
    ("recover_min_pings negative", {"recover_min_pings": -1}, "model_failover.recover_min_pings"),
    ("window_age 0", {"window_age_s": 0}, "model_failover.window_age_s"),
    ("recover_period negative", {"recover_period_s": -1}, "model_failover.recover_period_s"),
    ("dwell negative", {"dwell_s": -1}, "model_failover.dwell_s"),
    ("idle_reset 0", {"idle_reset_s": 0}, "model_failover.idle_reset_s"),
    ("recheck 0", {"unavailable_recheck_s": 0}, "model_failover.unavailable_recheck_s"),
    ("text number", {"dwell_s": "soon"}, "model_failover.dwell_s"),
    ("nan", {"dwell_s": "nan"}, "model_failover.dwell_s"),
    ("null", {"dwell_s": None}, "model_failover.dwell_s"),
    ("entries not an object", {"entries": []}, "model_failover.entries"),
    ("entry not an object", {"entries": {HAIKU55: 3}}, f"model_failover.entries.{HAIKU55}"),
    ("entry unknown field", {"entries": {HAIKU55: {"x": 1}}}, f"model_failover.entries.{HAIKU55}.x"),
    ("trip equals recover", {"entries": {HAIKU55: {"trip_ttft_s": 2.5}}},
     f"model_failover.entries.{HAIKU55}.recover_ttft_s"),
    ("recover above trip", {"entries": {HAIKU55: {"recover_ttft_s": 4.0}}},
     f"model_failover.entries.{HAIKU55}.recover_ttft_s"),
    ("trip 0", {"entries": {HAIKU55: {"trip_ttft_s": 0}}}, f"model_failover.entries.{HAIKU55}.trip_ttft_s"),
    ("deadline 0", {"entries": {HAIKU55: {"deadline_s": 0}}}, f"model_failover.entries.{HAIKU55}.deadline_s"),
    ("deadline null", {"entries": {HAIKU55: {"deadline_s": None}}},
     f"model_failover.entries.{HAIKU55}.deadline_s"),
    ("error rate above 1", {"entries": {HAIKU55: {"error_rate_limit": 1.5}}},
     f"model_failover.entries.{HAIKU55}.error_rate_limit"),
    ("empty model ID", {"entries": {"": {"trip_ttft_s": 3}}}, "model_failover.entries"),
]


@pytest.mark.parametrize(("section", "field"), [(s, f) for _, s, f in REFUSALS], ids=[r[0] for r in REFUSALS])
def test_each_refusal(section: Any, field: str, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(PolicySectionError) as err:
        parse_model_failover(section)
    assert err.value.field == field
    doc = with_section(section)
    with pytest.raises(PolicyError):  # seeding refuses it
        Policy.to_items(doc)
    # At run time the defaults apply, with an ERROR naming the field and never a value.
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        assert Policy.from_dict(doc).model_failover == ModelFailoverConfig({}, {})
    assert field in caplog.text
    assert "soon" not in caplog.text


def test_a_bad_stored_section_gives_the_defaults(caplog: pytest.LogCaptureFixture) -> None:
    store = seeded(SEED)
    store.put("policy", {"pk": MODEL_FAILOVER_ITEM, "pings_per_cycle": 9, "dwell_s": 10})
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        cfg = Policy.from_store(store).model_failover
    assert cfg.settings_for(IDS) == Settings.for_models(IDS)  # not even the valid dwell is kept
    assert "model_failover.pings_per_cycle" in caplog.text


def test_the_boundaries_are_accepted() -> None:
    cfg = parse_model_failover({"pings_per_cycle": 1, "recover_min_pings": 1, "min_samples": 1})
    assert cfg.settings_for(IDS).pings_per_cycle == 1
    assert parse_model_failover({"pings_per_cycle": 5}).settings["pings_per_cycle"] == 5
    # no bound between the recovery period and the dwell: both apply separately
    s = parse_model_failover({"recover_period_s": 600, "dwell_s": 30}).settings_for(IDS)
    assert (s.recover_period_s, s.dwell_s) == (600, 30)
    parse_model_failover({"recover_period_s": 0, "dwell_s": 0})


# --------------------------------------------------------------------------- overrides


def test_per_entry_overrides_are_keyed_by_model_id() -> None:
    cfg = parse_model_failover({"dwell_s": 60, "entries": {
        HAIKU55: {"trip_ttft_s": 5, "deadline_s": 6.5},
        HAIKU45: {"error_rate_limit": 0.5},
        SONNET55: {"trip_ttft_s": 4.5, "deadline_s": 9},
        "us.anthropic.claude-gone-1-0": {"trip_ttft_s": 1, "recover_ttft_s": 0.5}}})
    s = cfg.settings_for(IDS)
    assert s.dwell_s == 60
    e0, e1, e2 = s.entries
    assert (e0.trip_ttft_s, e0.recover_ttft_s, e0.deadline_s) == (5, 2.5, 6.5)
    assert (e1.trip_ttft_s, e1.error_rate_limit, e1.deadline_s) == (1.6, 0.5, 3.0)
    assert e2.trip_ttft_s == 4.5 and e2.deadline_s is None  # the last entry has no deadline
    # the same override moves with the model, not with the position
    s2 = cfg.settings_for((SONNET55, HAIKU55))
    assert s2.entries[0].deadline_s == 9 and s2.entries[1].deadline_s is None
    assert s2.entries[1].trip_ttft_s == 5
    # a list of one has no deadline
    assert cfg.settings_for((HAIKU55,)).entries[0].deadline_s is None


# --------------------------------------------------------------------------- store


def test_store_round_trip() -> None:
    section = {"window_calls": 30, "window_age_s": 90.5, "min_samples": 12, "miss_run": 2, "pings_per_cycle": 5,
               "recover_min_pings": 7, "recover_period_s": 60, "dwell_s": 0.25, "idle_reset_s": 300,
               "unavailable_recheck_s": 600,
               "entries": {HAIKU55: {"trip_ttft_s": 3.75, "recover_ttft_s": 2.25, "deadline_s": 5, "error_rate_limit": 0.1}}}
    doc = with_section(section)
    items = {i["pk"]: i for i in Policy.to_items(doc)}
    stored = items[MODEL_FAILOVER_ITEM]
    assert stored["window_age_s"] == "90.5" and stored["pings_per_cycle"] == 5  # no floats in the store
    store = MemoryStore()
    for item in items.values():
        store.put("policy", item)
    assert Policy.from_store(store).model_failover == Policy.from_dict(doc).model_failover
    assert Policy.from_dict(doc).model_failover == parse_model_failover(section)


def test_the_item_is_written_even_without_the_section() -> None:
    items = {i["pk"]: i for i in Policy.to_items(without_section())}
    assert items[MODEL_FAILOVER_ITEM] == {"pk": MODEL_FAILOVER_ITEM, "entries": {}}


def test_a_reseed_without_the_section_clears_an_override() -> None:
    store = seeded(with_section({"dwell_s": 10}))
    assert Policy.from_store(store).model_failover.settings["dwell_s"] == 10
    for item in Policy.to_items(without_section()):
        store.put("policy", item)
    assert Policy.from_store(store).model_failover == ModelFailoverConfig({}, {})


# --------------------------------------------------------------------------- seed


def test_seed_exits_2_on_a_bad_section(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    doc = json.loads((data / "policy.json").read_text(encoding="utf-8"))
    doc[MODEL_FAILOVER_ITEM]["pings_per_cycle"] = 6
    (data / "policy.json").write_text(json.dumps(doc), encoding="utf-8")
    assert seed.main(["--stage", "dev", "--data-dir", str(data)]) == 2
    assert "model_failover.pings_per_cycle" in capsys.readouterr().err


def test_seed_plan_carries_the_item() -> None:
    fixtures = json.loads((DATA / "fixtures.json").read_text(encoding="utf-8"))
    catalog = json.loads((DATA / "prompt_catalog.json").read_text(encoding="utf-8"))
    plan = seed.build_plan(fixtures, SEED, catalog, stage="dev")
    assert [i for i in plan.policy if i["pk"] == MODEL_FAILOVER_ITEM]


# --------------------------------------------------------------------------- properties

counts = st.integers(1, 200)
secs = st.one_of(st.integers(1, 5000), st.decimals(min_value="0.01", max_value="5000", places=2).map(float))
secs0 = st.one_of(st.just(0), secs)


@st.composite
def valid_sections(draw: st.DrawFn) -> dict[str, Any]:
    window = draw(counts)
    section: dict[str, Any] = {}
    if draw(st.booleans()):
        section["window_calls"] = window
    if "window_calls" in section and (window < 20 or draw(st.booleans())):  # min_samples defaults to 20
        section["min_samples"] = draw(st.integers(1, window))
    elif "window_calls" not in section and draw(st.booleans()):
        section["min_samples"] = draw(st.integers(1, 20))
    for name in ("miss_run", "recover_min_pings"):
        if draw(st.booleans()):
            section[name] = draw(counts)
    if draw(st.booleans()):
        section["pings_per_cycle"] = draw(st.integers(1, 5))
    for name in ("window_age_s", "idle_reset_s", "unavailable_recheck_s"):
        if draw(st.booleans()):
            section[name] = draw(secs)
    for name in ("recover_period_s", "dwell_s"):  # independent: no bound between them
        if draw(st.booleans()):
            section[name] = draw(secs0)
    entries: dict[str, Any] = {}
    for model_id in draw(st.lists(st.sampled_from([*IDS, "x.model-a", "x.model-b"]), unique=True, max_size=4)):
        ov: dict[str, Any] = {}
        if draw(st.booleans()):  # trip above recover, jointly drawn so the pair is always valid
            recover = draw(st.integers(1, 50)) / 10
            ov["recover_ttft_s"] = recover
            ov["trip_ttft_s"] = recover + draw(st.integers(1, 50)) / 10
        if draw(st.booleans()):
            ov["deadline_s"] = draw(st.integers(1, 100)) / 10
        if draw(st.booleans()):
            ov["error_rate_limit"] = draw(st.integers(1, 100)) / 100
        entries[model_id] = ov
    with_entries = draw(st.booleans())
    if entries or with_entries:
        section["entries"] = entries
    return section


def _stored(doc: dict[str, Any]) -> Policy:
    return Policy.from_store(seeded(doc))


@settings(max_examples=100, deadline=None)
@given(valid_sections())
def test_valid_sections_parse_and_round_trip(section: dict[str, Any]) -> None:
    cfg = parse_model_failover(section)
    assert parse_model_failover(cfg.to_doc()) == cfg  # to_doc / parse are inverse
    doc = with_section(section)
    assert Policy.from_dict(doc).model_failover == cfg
    assert _stored(doc).model_failover == cfg  # through the store
    for ids in (IDS, (HAIKU55,), (SONNET55, HAIKU55)):
        assert isinstance(cfg.settings_for(ids), Settings)  # always builds


BAD = st.one_of(
    st.tuples(st.just("pings_per_cycle"), st.one_of(st.integers(max_value=0), st.integers(min_value=6))),
    st.tuples(st.sampled_from(["window_calls", "min_samples", "miss_run", "recover_min_pings"]),
              st.integers(max_value=0)),
    st.tuples(st.sampled_from(["window_age_s", "idle_reset_s", "unavailable_recheck_s"]),
              st.integers(max_value=0)),
    st.tuples(st.sampled_from(["recover_period_s", "dwell_s"]), st.integers(max_value=-1)),
)


@settings(max_examples=100, deadline=None)
@given(valid_sections(), BAD)
def test_any_bound_violation_is_refused_at_seed(section: dict[str, Any], bad: tuple[str, int]) -> None:
    section = {**section, bad[0]: bad[1]}
    with pytest.raises(PolicyError):
        Policy.to_items(with_section(section))
    assert Policy.from_dict(with_section(section)).model_failover == ModelFailoverConfig({}, {})


@settings(max_examples=100, deadline=None)
@given(st.sampled_from(IDS), st.integers(1, 100), st.integers(0, 100))
def test_a_trip_not_above_recover_is_refused(model_id: str, recover10: int, gap10: int) -> None:
    recover = recover10 / 10
    section = {"entries": {model_id: {"recover_ttft_s": recover, "trip_ttft_s": recover - gap10 / 1000}}}
    with pytest.raises(PolicySectionError):
        parse_model_failover(section)
    with pytest.raises(PolicyError):
        Policy.to_items(with_section(section))
