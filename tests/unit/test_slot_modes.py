"""Slot modes."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from owlbank.policy import Policy
from owlbank.slot_policy import ResolutionSettings, SlotSettings, parse_slot
from owlbank.slots import modes
from owlbank.slots.modes import INVALID, ModeDecision, freeze, log_mode_table, mode_decision, restrict
from owlbank.store import MemoryStore
from tests.logutil import text_of

DATA = Path(__file__).resolve().parents[2] / "data"
SEED: dict[str, Any] = json.loads((DATA / "policy.json").read_text(encoding="utf-8"))
ZIP_DEF = SEED["factors"]["zip"]
CARD = parse_slot("card", SEED["slots"]["card"], factor_def=None)
ZIP = parse_slot("zip", SEED["slots"]["zip"], factor_def=ZIP_DEF)
SENSITIVE_CARD = parse_slot("card", {**SEED["slots"]["card"], "sensitive": True}, factor_def=None)
ON = ResolutionSettings(generative_enabled=True)
OFF = ResolutionSettings()


@pytest.fixture(autouse=True)
def _fresh_reports() -> Iterator[None]:
    modes.reset_invalid_reports()
    yield
    modes.reset_invalid_reports()


def _d(s: SlotSettings, ceiling: Any = "generative", r: ResolutionSettings = ON, *, slot: object = None,
       flow: object = None, default: object = "assisted") -> ModeDecision:
    return mode_decision(s, r, "confirm", ceiling, raw_mode=slot, raw_flow_mode=flow, raw_default=default)


def _store(doc: dict[str, Any]) -> MemoryStore:
    store = MemoryStore()
    for item in Policy.to_items(doc):
        store.put("policy", item)
    return store


def _events(caplog: pytest.LogCaptureFixture, event: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "event", None) == event]


# --------------------------------------------------------------------------- precedence


def test_slot_then_flow_then_default() -> None:
    assert (_d(CARD, slot="deterministic", flow="generative").configured, _d(CARD, slot="deterministic").source) \
        == ("deterministic", "slot")
    d = _d(CARD, flow="deterministic", default="generative")
    assert (d.configured, d.source) == ("deterministic", "flow")
    d = _d(CARD, default="deterministic")
    assert (d.configured, d.source, d.effective, d.capped) == ("deterministic", "default", "deterministic", None)


@pytest.mark.parametrize("bad", ["fast", "", 0, True, [], {"mode": "assisted"}, INVALID])
def test_a_present_invalid_value_gives_deterministic(bad: object) -> None:
    for kw in ({"slot": bad, "flow": "generative"}, {"flow": bad}, {"default": bad}):
        d = _d(CARD, **kw)  # type: ignore[arg-type]
        assert (d.configured, d.source, d.effective, d.capped) == ("deterministic", "invalid", "deterministic", None)


def test_a_missing_default_gives_deterministic() -> None:
    d = _d(CARD, default=None)
    assert (d.configured, d.source, d.effective) == ("deterministic", "invalid", "deterministic")


# --------------------------------------------------------------------------- ceiling and generative


@pytest.mark.parametrize("ceiling, configured, effective", [
    ("deterministic", "assisted", "deterministic"),
    ("deterministic", "generative", "deterministic"),
    ("assisted", "generative", "assisted"),
    ("assisted", "deterministic", "deterministic"),
    ("generative", "assisted", "assisted"),
])
def test_effective_is_the_more_restrictive(ceiling: str, configured: str, effective: str) -> None:
    d = _d(CARD, ceiling, slot=configured)
    assert d.effective == effective and d.ceiling == ceiling
    assert d.capped == ("ceiling" if configured != effective else None)


def test_generative_needs_the_setting_and_a_non_sensitive_choice() -> None:
    assert _d(CARD, slot="generative").effective == "generative"
    d = _d(CARD, r=OFF, slot="generative")
    assert (d.effective, d.capped) == ("assisted", "generative_disabled")
    d = _d(ZIP, default="generative")
    assert (d.effective, d.capped) == ("assisted", "type")
    d = _d(SENSITIVE_CARD, default="generative")
    assert (d.effective, d.capped) == ("assisted", "sensitive")


def test_bad_ceiling_is_a_configuration_error() -> None:
    with pytest.raises(ValueError):
        _d(CARD, "max")
    with pytest.raises(ValueError):
        restrict("assisted", "max")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- restrict


@pytest.mark.parametrize("frozen, ceiling, out", [
    ("generative", "assisted", "assisted"), ("assisted", "generative", "assisted"),
    ("deterministic", "generative", "deterministic"), ("assisted", "deterministic", "deterministic"),
])
def test_restrict(frozen: Any, ceiling: Any, out: str) -> None:
    assert restrict(frozen, ceiling) == out


def test_decision_to_doc() -> None:
    assert _d(CARD, "assisted", slot="generative").to_doc() == {
        "configured": "generative", "source": "slot", "ceiling": "assisted", "effective": "assisted",
        "capped": "ceiling"}


# --------------------------------------------------------------------------- freeze and slot_mode_invalid


def _bad_store() -> MemoryStore:
    store = _store(SEED)
    item = store.get("policy", {"pk": "slot#zip"})
    assert item is not None
    store.put("policy", {**item, "mode": "SECRET-94105"})
    return store


def test_freeze_seed() -> None:
    out = freeze(Policy.from_dict(SEED), "login", ["zip", "ssn4"], "assisted")
    assert {n: (d.configured, d.source, d.effective) for n, d in out.items()} == {
        "zip": ("assisted", "default", "assisted"), "ssn4": ("assisted", "default", "assisted")}


def test_freeze_invalid_emits_once_per_item_field_and_version(caplog: pytest.LogCaptureFixture) -> None:
    store = _bad_store()
    p = Policy.from_store(store)
    with caplog.at_level(logging.INFO, logger="owlbank.slots.modes"):
        for _ in range(3):
            d = freeze(p, "login", ["zip", "ssn4"], "generative")
            assert (d["zip"].effective, d["zip"].source) == ("deterministic", "invalid")
            assert d["ssn4"].effective == "assisted"
    recs = _events(caplog, "slot_mode_invalid")
    assert len(recs) == 1 and recs[0].levelno == logging.ERROR
    assert (recs[0].item, recs[0].field) == ("slot#zip", "mode")  # type: ignore[attr-defined]
    assert "SECRET" not in text_of(caplog.records) and "94105" not in text_of(caplog.records)
    # A new policy version is reported again.
    meta = store.get("policy", {"pk": "meta"})
    assert meta is not None
    store.put("policy", {**meta, "version": meta["version"] + 1})
    with caplog.at_level(logging.INFO, logger="owlbank.slots.modes"):
        freeze(Policy.from_store(store), "login", ["zip"], "generative")
    assert len(_events(caplog, "slot_mode_invalid")) == 2


def test_freeze_unknown_slot_raises_key_error() -> None:
    with pytest.raises(KeyError):
        freeze(Policy.from_dict(SEED), "login", ["nope"], "assisted")


# --------------------------------------------------------------------------- log_mode_table


def test_slot_modes_table(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="owlbank.slots.modes"):
        table = log_mode_table(Policy.from_dict(SEED), "assisted")
    (rec,) = _events(caplog, "slot_modes")
    assert rec.levelno == logging.INFO
    assert (rec.ceiling, rec.default_mode, rec.flows) == ("assisted", "assisted", {})  # type: ignore[attr-defined]
    slots = rec.slots  # type: ignore[attr-defined]
    assert set(slots) == {"zip", "ssn4", "otp", "phone", "yes_no", "card"}
    assert set(slots["zip"]) == {"login", "stepup"} and set(slots["card"]) == {"confirm"}
    assert slots["card"]["confirm"] == {"configured": "assisted", "source": "default", "effective": "assisted"}
    assert table["card"]["confirm"].effective == "assisted"
    assert _events(caplog, "slot_mode_capped") == [] and _events(caplog, "slot_mode_invalid") == []


def test_capped_events_name_where_the_mode_was_configured(caplog: pytest.LogCaptureFixture) -> None:
    doc = copy.deepcopy(SEED)
    doc["slot_resolution"].update(flows={"confirm": {"mode": "generative"}})
    doc["slots"]["zip"]["mode"] = "assisted"
    doc["slots"]["card"]["mode"] = "generative"
    with caplog.at_level(logging.INFO, logger="owlbank.slots.modes"):
        log_mode_table(Policy.from_dict(doc), "deterministic")
    got = {(r.scope, r.target, r.reason) for r in _events(caplog, "slot_mode_capped")}  # type: ignore[attr-defined]
    assert got == {("default", "default_mode", "ceiling"), ("flow", "confirm", "ceiling"),
                   ("slot", "card", "ceiling"), ("slot", "zip", "ceiling")}
    assert all(r.levelno == logging.INFO for r in _events(caplog, "slot_mode_capped"))


def test_capped_by_type_or_setting(caplog: pytest.LogCaptureFixture) -> None:
    doc = copy.deepcopy(SEED)
    doc["slot_resolution"]["default_mode"] = "generative"
    with caplog.at_level(logging.INFO, logger="owlbank.slots.modes"):
        table = log_mode_table(Policy.from_dict(doc), "generative")
    got = {(r.scope, r.target, r.reason) for r in _events(caplog, "slot_mode_capped")}  # type: ignore[attr-defined]
    # generative_enabled is off: the default is capped once; the digit slots are capped by type.
    assert ("default", "default_mode", "generative_disabled") in got
    assert {("slot", n, "type") for n in ("zip", "ssn4", "otp", "yes_no")} <= got
    assert table["card"]["confirm"].effective == "assisted"


def test_log_mode_table_reports_every_invalid_field(caplog: pytest.LogCaptureFixture) -> None:
    store = _bad_store()
    store.delete("policy", {"pk": "slot_resolution"})
    with caplog.at_level(logging.INFO, logger="owlbank.slots.modes"):
        log_mode_table(Policy.from_store(store), "assisted")
        log_mode_table(Policy.from_store(store), "assisted")
    got = sorted((r.item, r.field) for r in _events(caplog, "slot_mode_invalid"))  # type: ignore[attr-defined]
    assert got == [("slot#zip", "mode"), ("slot_resolution", "default_mode")]
    (first, _) = _events(caplog, "slot_modes")
    assert first.default_mode == "invalid"  # type: ignore[attr-defined]
    assert first.slots["ssn4"]["login"]["effective"] == "deterministic"  # type: ignore[attr-defined]
