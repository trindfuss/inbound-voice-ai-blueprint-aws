"""The ``slots`` section, the ``slot#`` items and ``slot_resolution`` in the policy loader."""

from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any

import pytest

from owlbank.policy import Policy, PolicyError
from owlbank.slot_policy import OptionDef, ResolutionSettings
from owlbank.slots.types import SlotDecl
from owlbank.store import MemoryStore
from tests.logutil import text_of

DATA = Path(__file__).resolve().parents[2] / "data"
SEED: dict[str, Any] = json.loads((DATA / "policy.json").read_text(encoding="utf-8"))


def _store(doc: dict[str, Any]) -> MemoryStore:
    store = MemoryStore()
    for item in Policy.to_items(doc):
        store.put("policy", item)
    return store


def _drop(store: MemoryStore, pk: str) -> None:
    store.delete("policy", {"pk": pk})


def _put(store: MemoryStore, item: dict[str, Any]) -> None:
    store.put("policy", item)


# --------------------------------------------------------------------------- the seed file


def test_seed_loads_strictly_with_the_six_slots() -> None:
    p = Policy.from_dict(SEED)
    assert set(p.slots()) == {"zip", "ssn4", "otp", "phone", "yes_no", "card"}
    assert p.slot_problems == ()
    Policy.to_items(SEED)  # strict


@pytest.mark.parametrize("name, sensitive", [("zip", False), ("ssn4", True), ("otp", True)])
def test_seed_factor_slots(name: str, sensitive: bool) -> None:
    s = Policy.from_dict(SEED).slot(name)
    assert s.is_factor and s.sensitive is sensitive
    assert s.keypad_only is False and s.barge_in is True and s.flow is None
    fd = SEED["factors"][name]
    assert s.decl.slot_type == fd["slotType"] and dict(s.decl.constraints) == fd["constraints"]
    assert s.decl.max_retries == fd["maxRetries"]


def test_seed_yes_no_slot() -> None:
    s = Policy.from_dict(SEED).slot("yes_no")
    assert not s.is_factor and s.flow == "confirm"
    assert (s.sensitive, s.keypad_only, s.barge_in) == (False, False, True)
    assert s.decl.slot_type == "yes_no" and s.decl.max_retries == 1


def test_is_factor_exactly_for_factors_entries() -> None:
    p = Policy.from_dict(SEED)
    assert {n for n in p.slots() if p.slot(n).is_factor} == set(SEED["factors"])


def test_factors_entries_carry_no_sensitive() -> None:
    # ``sensitive`` moved from ``factors`` to the slot's settings.
    assert all("sensitive" not in SEED["factors"][f] for f in ("zip", "ssn4", "otp"))


@pytest.mark.parametrize("value", [True, False])
def test_seed_refuses_sensitive_in_a_factors_entry(value: bool) -> None:
    doc = copy.deepcopy(SEED)
    doc["factors"]["zip"]["sensitive"] = value
    with pytest.raises(PolicyError, match=r"factors\.zip\.sensitive"):
        Policy.from_dict(doc)


def test_store_ignores_a_sensitive_left_in_meta_factors(caplog: pytest.LogCaptureFixture) -> None:
    # A table seeded: the slot item decides; the leftover is logged, not read.
    store = _store(SEED)
    meta = store.get("policy", {"pk": "meta"})
    assert meta is not None
    meta["factors"]["zip"]["sensitive"] = True
    _put(store, meta)
    with caplog.at_level(logging.ERROR):
        p = Policy.from_store(store)
    assert p.slot("zip").sensitive is False
    assert [e.field for e in p.slot_problems] == ["factors.zip.sensitive"]
    assert "factors.zip.sensitive" in text_of(caplog.records)


def test_unknown_slot_raises_key_error() -> None:
    with pytest.raises(KeyError):
        Policy.from_dict(SEED).slot("nope")


# --------------------------------------------------------------------------- strict refusals


@pytest.mark.parametrize("field", ["sensitive", "keypad_only", "barge_in"])
@pytest.mark.parametrize("value", ["true", 1, None])
def test_strict_refuses_a_non_boolean(field: str, value: Any) -> None:
    doc = copy.deepcopy(SEED)
    doc["slots"]["ssn4"][field] = value
    with pytest.raises(PolicyError, match=rf"slot#ssn4\.{field}"):
        Policy.from_dict(doc)
    with pytest.raises(PolicyError, match=rf"slot#ssn4\.{field}"):
        Policy.to_items(doc)


def test_strict_refuses_a_non_object_section() -> None:
    doc = copy.deepcopy(SEED)
    doc["slots"] = ["zip"]
    with pytest.raises(PolicyError):
        Policy.from_dict(doc)


def test_seed_file_without_a_factor_entry_is_sensitive() -> None:  # seed side
    doc = copy.deepcopy(SEED)
    del doc["slots"]["ssn4"]
    assert Policy.from_dict(doc).slot("ssn4").sensitive is True


def test_seed_file_without_slots_section_still_loads() -> None:
    doc = copy.deepcopy(SEED)
    del doc["slots"]
    p = Policy.from_dict(doc)
    assert set(p.slots()) == set(SEED["factors"])
    assert all(p.slot(f).sensitive for f in SEED["factors"])


@pytest.mark.parametrize("slot_type", ["yes_no", "choice"])
def test_sensitive_on_yes_no_and_choice(slot_type: str) -> None:
    doc = copy.deepcopy(SEED)
    doc["slots"]["extra"] = {"slotType": slot_type, "sensitive": True, "flow": "confirm"}
    assert Policy.from_dict(doc).slot("extra").sensitive is True


# --------------------------------------------------------------------------- items and the store


def test_items_carry_meta_slots_and_one_item_per_slot() -> None:
    items = {i["pk"]: i for i in Policy.to_items(SEED)}
    assert items["meta"]["slots"] == list(Policy.from_dict(SEED).slots())
    assert items["slot#zip"] == {"pk": "slot#zip", "sensitive": False, "keypad_only": False, "barge_in": True,
                                 "confirm": "when_inferred", "declinable": True}
    assert items["slot#ssn4"]["sensitive"] is True
    assert items["slot#yes_no"] == {"pk": "slot#yes_no", "slotType": "yes_no", "maxRetries": 1,
                                    "sensitive": False, "keypad_only": False, "barge_in": True,
                                    "flow": "confirm", "confirm": "when_inferred"}


def test_store_round_trip() -> None:
    a, b = Policy.from_dict(SEED), Policy.from_store(_store(SEED))
    assert a.slots() == b.slots()
    for n in a.slots():
        assert a.slot(n) == b.slot(n)
    assert b.slot_problems == ()


def test_store_ignores_rev() -> None:
    store = _store(SEED)
    _put(store, {"pk": "slot#zip", "sensitive": False, "keypad_only": True, "barge_in": True, "rev": 4})
    p = Policy.from_store(store)
    assert p.slot("zip").keypad_only is True and p.slot_problems == ()


@pytest.mark.parametrize("field", ["sensitive", "keypad_only", "barge_in"])
def test_store_non_boolean_fails_closed_and_logs(field: str, caplog: pytest.LogCaptureFixture) -> None:
    store = _store(SEED)
    item = store.get("policy", {"pk": "slot#zip"})
    assert item is not None
    item[field] = "SECRET-94105"
    _put(store, item)
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        p = Policy.from_store(store)
    assert getattr(p.slot("zip"), field) is True
    assert [e.field for e in p.slot_problems] == [f"slot#zip.{field}"]
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and r.name == "owlbank.policy"]
    assert any(f"slot#zip.{field}" in r.getMessage() for r in errors)
    assert "SECRET" not in text_of(caplog.records) and "94105" not in text_of(caplog.records)


def test_store_listed_non_factor_without_item_is_left_out_with_a_problem() -> None:
    store = _store(SEED)
    _drop(store, "slot#yes_no")
    p = Policy.from_store(store)
    assert "yes_no" not in p.slots()
    assert [e.field for e in p.slot_problems] == ["slot#yes_no"]


@pytest.mark.parametrize("name", ["zip", "ssn4", "otp"])
def test_store_factor_without_item_is_sensitive(name: str) -> None:
    store = _store(SEED)
    _drop(store, f"slot#{name}")
    p = Policy.from_store(store)
    s = p.slot(name)
    assert s.sensitive is True and s.keypad_only is False and s.barge_in is True
    assert [e.field for e in p.slot_problems] == [f"slot#{name}"]


def test_store_item_without_sensitive_is_not_sensitive() -> None:
    store = _store(SEED)
    _put(store, {"pk": "slot#ssn4", "keypad_only": False, "barge_in": True})
    p = Policy.from_store(store)
    assert p.slot("ssn4").sensitive is False and p.slot_problems == ()


def test_table_from_before_the_reseed_fails_closed() -> None:
    """A table seeded before this feature: no ``meta.slots``, no ``slot#`` items."""
    store = _store(SEED)
    meta = store.get("policy", {"pk": "meta"})
    assert meta is not None
    del meta["slots"]
    _put(store, meta)
    for n in Policy.from_dict(SEED).slots():
        _drop(store, f"slot#{n}")
    p = Policy.from_store(store)
    assert set(p.slots()) == set(SEED["factors"])
    assert all(p.slot(f).sensitive for f in SEED["factors"])
    assert sorted(e.field for e in p.slot_problems) == sorted(f"slot#{f}" for f in SEED["factors"])


def test_store_refresh_picks_up_a_slot_edit() -> None:
    class Clock:
        t = 0.0

        def __call__(self) -> float:
            return self.t

    store, clock = _store(SEED), Clock()
    p = Policy.from_store(store, ttl_s=60, clock=clock)
    assert p.slot("zip").keypad_only is False
    _put(store, {"pk": "slot#zip", "sensitive": False, "keypad_only": True, "barge_in": True})
    clock.t = 61
    assert p.slot("zip").keypad_only is True


def test_slot_decl_of_a_factor_equals_the_factor_shape() -> None:
    s = Policy.from_dict(SEED).slot("otp")
    fd = SEED["factors"]["otp"]
    assert s.decl == SlotDecl(slot_type=fd["slotType"], constraints=fd["constraints"],
                              label_key=fd["labelKey"], prompt_key=fd["promptKey"],
                              max_retries=fd["maxRetries"])


# --------------------------------------------------------------------------- phase 2


def test_seed_card_slot_equals_the_documented_slot() -> None:
    s = Policy.from_dict(SEED).slot("card")
    assert not s.is_factor and s.flow == "confirm"
    assert s.decl == SlotDecl(slot_type="choice", prompt_key="prompt.card", label_key="label.card", max_retries=2)
    assert (s.sensitive, s.keypad_only, s.barge_in, s.confirm, s.mode) == (False, False, True, "when_inferred", None)
    assert s.options == (
        OptionDef("red", {"en-US": ("red",), "es-US": ("roja",)}),
        OptionDef("purple", {"en-US": ("purple",), "es-US": ("morada",)}),
        OptionDef("rewards", {"en-US": ("rewards",), "es-US": ("recompensas",)}),
    )


@pytest.mark.parametrize("name, confirm", [("zip", "when_inferred"), ("ssn4", "never"), ("otp", "never"),
                                           ("yes_no", "when_inferred")])
def test_seed_confirm(name: str, confirm: str) -> None:
    assert Policy.from_dict(SEED).slot(name).confirm == confirm


def test_seed_slot_resolution_equals_the_documented_values() -> None:
    p = Policy.from_dict(SEED)
    assert p.slot_resolution == ResolutionSettings(default_mode="assisted", flows={}, generative_enabled=False,
                                                   resolver_budget_ms=2500)
    assert p.slot_resolution_problems == () and p.invalid_mode_fields == frozenset()


def test_seed_meta_slots_lists_card_and_items_carry_slot_resolution() -> None:
    items = {i["pk"]: i for i in Policy.to_items(SEED)}
    assert items["meta"]["slots"] == ["zip", "ssn4", "otp", "phone", "yes_no", "card"]
    assert items["slot_resolution"] == {"pk": "slot_resolution", "default_mode": "assisted", "flows": {},
                                        "generative_enabled": False, "resolver_budget_ms": 2500}
    assert items["slot#card"]["options"][1] == {"value": "purple", "aliases": {"en-US": ["purple"],
                                                                                "es-US": ["morada"]}}


def test_seed_without_slot_resolution_gets_the_defaults() -> None:
    doc = copy.deepcopy(SEED)
    del doc["slot_resolution"]
    assert Policy.from_dict(doc).slot_resolution == ResolutionSettings()


@pytest.mark.parametrize("edit, field", [
    (lambda d: d["slots"]["ssn4"].update(mode="generative"), "slot#ssn4.mode"),
    (lambda d: d["slots"]["zip"].update(mode="fast"), "slot#zip.mode"),
    (lambda d: d["slots"]["otp"].update(confirm="maybe"), "slot#otp.confirm"),
    (lambda d: d["slots"]["card"]["options"][1]["aliases"]["en-US"].append("RED"), "slot#card.options.aliases.en-US"),
    (lambda d: d["slot_resolution"].update(default_mode="x"), "slot_resolution.default_mode"),
    (lambda d: d["slot_resolution"].update(flows={"lunch": {"mode": "assisted"}}), "slot_resolution.flows"),
])
def test_seed_refuses_bad_phase2_values(edit: Any, field: str) -> None:
    doc = copy.deepcopy(SEED)
    edit(doc)
    with pytest.raises(PolicyError, match=re.escape(field)):
        Policy.from_dict(doc)
    with pytest.raises(PolicyError, match=re.escape(field)):
        Policy.to_items(doc)


def test_store_round_trip_phase2() -> None:
    a, b = Policy.from_dict(SEED), Policy.from_store(_store(SEED))
    assert b.slot("card") == a.slot("card")
    assert b.slot_resolution == a.slot_resolution
    assert b.slot_resolution_problems == () and b.invalid_mode_fields == frozenset()


def test_store_bad_mode_loads_deterministic_and_is_named(caplog: pytest.LogCaptureFixture) -> None:
    store = _store(SEED)
    item = store.get("policy", {"pk": "slot#zip"})
    assert item is not None
    _put(store, {**item, "mode": "SECRET-94105"})
    res = store.get("policy", {"pk": "slot_resolution"})
    assert res is not None
    _put(store, {**res, "flows": {"stepup": {"mode": 7}}})
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        p = Policy.from_store(store)
    assert p.slot("zip").mode == "deterministic"
    assert p.slot_resolution.flows == {"stepup": "deterministic"}
    assert p.invalid_mode_fields == {"slot#zip.mode", "slot_resolution.flows.stepup.mode"}
    logged = text_of(caplog.records)
    assert "slot#zip.mode" in logged and "slot_resolution.flows.stepup.mode" in logged
    assert "SECRET" not in logged and "94105" not in logged


def test_store_without_slot_resolution_fails_closed() -> None:  # Missing default_mode
    store = _store(SEED)
    _drop(store, "slot_resolution")
    p = Policy.from_store(store)
    assert p.slot_resolution.default_mode == "deterministic"
    assert p.invalid_mode_fields == {"slot_resolution.default_mode"}
    assert p.slot_problems == ()  # the slot items are fine


def test_store_generative_on_a_sensitive_slot_loads_assisted() -> None:
    store = _store(SEED)
    item = store.get("policy", {"pk": "slot#ssn4"})
    assert item is not None
    _put(store, {**item, "mode": "generative"})
    p = Policy.from_store(store)
    assert p.slot("ssn4").mode == "assisted"
    assert [e.field for e in p.slot_problems] == ["slot#ssn4.mode"] and p.invalid_mode_fields == frozenset()
