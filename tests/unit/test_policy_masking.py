"""Policy fields: per-tool ``mask`` and the ``masking`` section.
Same fail-closed pattern as ``ani_trust`` and ``confirm``, per field."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from owlbank.policy import (
    DEFAULT_MASKING,
    MASK_GENERIC_KEY,
    MASK_GENERIC_KEYS,
    MaskingConfig,
    Policy,
    PolicyError,
    PolicySectionError,
    parse_mask,
    parse_masking,
)
from owlbank.masking_config import ToolMask
from owlbank.store import MemoryStore

DATA = Path(__file__).resolve().parents[2] / "data"
DOC: dict[str, Any] = json.loads((DATA / "policy.json").read_text())


def doc(masking: Any = None, **tool_masks: Any) -> dict[str, Any]:
    d = copy.deepcopy(DOC)
    if masking is None:
        d.pop("masking", None)
    else:
        d["masking"] = masking
    for tool, mask in tool_masks.items():
        d["tools"][tool]["mask"] = mask
    return d


# --------------------------------------------------------------------------- masking section


def test_defaults() -> None:
    # After_ms moved to 2000; tool_after_ms and the silence fields are new.
    assert MaskingConfig(1000, 2500, "wait", 15.0, 0, 4000, 2, "escalate") == DEFAULT_MASKING
    assert parse_masking({}) == DEFAULT_MASKING
    assert Policy.from_dict(doc(None)).masking == DEFAULT_MASKING
    assert Policy.from_dict(DOC).masking == parse_masking(DOC["masking"])


def test_valid_config_parses() -> None:
    cfg = {"after_ms": 0, "handoff_wait_ms": 5000, "end_mode": "immediate", "chars_per_s": 8}
    got = Policy.from_dict(doc(cfg)).masking
    assert got == MaskingConfig(0, 5000, "immediate", 8.0)
    assert parse_masking({"chars_per_s": "12.5"}).chars_per_s == 12.5
    assert parse_masking({"after_ms": 10000, "handoff_wait_ms": 0, "chars_per_s": 30.0}) == \
        MaskingConfig(10000, 0, "wait", 30.0)


def test_missing_section_is_defaults_without_error(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        p = Policy.from_dict(doc(None))
    assert p.masking == DEFAULT_MASKING
    assert not [r for r in caplog.records if "masking" in r.getMessage()]


@pytest.mark.parametrize("field,value", [
    ("after_ms", -1), ("after_ms", 10001), ("after_ms", "1500"), ("after_ms", 1500.0),
    ("after_ms", True), ("after_ms", None),
    ("handoff_wait_ms", -1), ("handoff_wait_ms", 10001), ("handoff_wait_ms", False),
    ("handoff_wait_ms", [1]),
    ("end_mode", "later"), ("end_mode", None), ("end_mode", 1), ("end_mode", "WAIT"),
    ("end_mode", ["wait"]),
    ("chars_per_s", 7.99), ("chars_per_s", 31), ("chars_per_s", "fast"), ("chars_per_s", True),
    ("chars_per_s", float("nan")), ("chars_per_s", float("inf")), ("chars_per_s", None),
    ("chars_per_s", {}),
    ("tool_after_ms", -1), ("tool_after_ms", 10001), ("tool_after_ms", "600"),
    ("silence_timeout_ms", 10001), ("silence_timeout_ms", None),
    ("silence_max_repeats", 6), ("silence_max_repeats", -1), ("silence_max_repeats", 1.0),
    ("silence_action", "transfer"), ("silence_action", None), ("silence_action", "ESCALATE"),
])
def test_invalid_field_gives_that_fields_default_and_an_error(
        field: str, value: Any, caplog: pytest.LogCaptureFixture) -> None:
    other = {"after_ms": 700, "handoff_wait_ms": 900, "end_mode": "immediate", "chars_per_s": 20,
             "tool_after_ms": 300, "silence_timeout_ms": 0, "silence_max_repeats": 5,
             "silence_action": "hangup"}
    raw = {**other, field: value}
    with pytest.raises(PolicySectionError) as exc:
        parse_masking(raw)
    assert exc.value.field == f"masking.{field}"
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        got = Policy.from_dict(doc(raw)).masking
    # The bad field falls back; the good ones are kept.
    assert getattr(got, field) == getattr(DEFAULT_MASKING, field)
    for name, v in other.items():
        if name != field:
            assert getattr(got, name) == v
    assert any(f"masking.{field}" in r.getMessage() and r.levelno == logging.ERROR
               for r in caplog.records)
    with pytest.raises(PolicyError):
        Policy.to_items(doc(raw))


def test_unknown_key_is_an_error_and_known_fields_still_apply(
        caplog: pytest.LogCaptureFixture) -> None:
    raw = {"after_ms": 800, "bogus": 1}
    with pytest.raises(PolicySectionError) as exc:
        parse_masking(raw)
    assert exc.value.field == "masking.bogus"
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        got = Policy.from_dict(doc(raw)).masking
    assert got == MaskingConfig(after_ms=800)
    assert any("masking.bogus" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("raw", [[], "wait", 3, True])
def test_non_object_section_gives_defaults_and_an_error(
        raw: Any, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        assert Policy.from_dict(doc(raw)).masking == DEFAULT_MASKING
    assert any("masking" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("cfg", [
    {"after_ms": 0, "handoff_wait_ms": 100, "end_mode": "immediate", "chars_per_s": 9},
    {"after_ms": 1200, "handoff_wait_ms": 2500, "end_mode": "wait", "chars_per_s": 17.5},
])
def test_masking_round_trips_through_the_store(cfg: dict[str, Any]) -> None:
    store = MemoryStore()
    for item in Policy.to_items(doc(cfg)):
        store.put("policy", item)  # MemoryStore refuses floats, like DynamoDB here
    item = store.get("policy", {"pk": "masking"})
    assert item is not None and item["pk"] == "masking"
    got = Policy.from_store(store).masking
    assert got == parse_masking(cfg) == Policy.from_dict(doc(cfg)).masking


def test_seed_without_masking_writes_no_item_and_reads_defaults() -> None:
    store = MemoryStore()
    for item in Policy.to_items(doc(None)):
        store.put("policy", item)
    assert store.get("policy", {"pk": "masking"}) is None
    assert Policy.from_store(store).masking == DEFAULT_MASKING


# --------------------------------------------------------------------------- per-tool mask


def test_mask_phrase_key_for_masked_unmasked_and_unknown_tools() -> None:
    p = Policy.from_dict(DOC)
    assert p.mask_phrase_key("list_recent_transactions") == "mask.list_recent_transactions"
    assert p.mask_phrase_key("change_address") == "mask.change_address"
    assert p.mask_phrase_key("get_balance") == "mask.get_balance"
    assert p.mask_phrase_key("make_payment") == "mask.make_payment"
    assert p.mask_phrase_key("transfer_funds") == "mask.transfer_funds"
    for tool in ("authenticate", "escalate", "faq_lookup"):
        assert p.mask_phrase_key(tool) is None
    assert p.mask_phrase_key("no_such_tool") is None


def test_explicit_null_mask_is_no_phrase() -> None:
    assert Policy.from_dict(doc(None, get_balance=None)).mask_phrase_key("get_balance") is None
    assert parse_mask("get_balance", {"mask": None}) is None


def test_mask_allowed_on_any_tool_kind() -> None:
    p = Policy.from_dict(doc(None, faq_lookup={"phrase_key": "mask.faq_lookup"},
                             escalate={"phrase_key": "mask.escalate"}))
    assert p.mask_phrase_key("faq_lookup") == "mask.faq_lookup"
    assert p.mask_phrase_key("escalate") == "mask.escalate"


@pytest.mark.parametrize("mask", [
    "mask.get_balance", {"phrase_key": "tool.get_balance"}, {"phrase_key": "mask."},
    {"phrase_key": ""}, {"phrase_key": None}, {"phrase_key": 3}, {},
    ["mask.get_balance"],
    {"phrase_key": "mask." + "x" * 200},
])
def test_invalid_mask_row_is_no_phrase_and_an_error(
        mask: Any, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(PolicySectionError) as exc:
        parse_mask("get_balance", {"mask": mask})
    assert exc.value.field.startswith("tools.get_balance.mask")
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        p = Policy.from_dict(doc(None, get_balance=mask))
    assert p.mask_phrase_key("get_balance") is None
    assert "mask" not in p.tool_row("get_balance")  # type: ignore[operator]
    assert any("tools.get_balance.mask" in r.getMessage() and r.levelno == logging.ERROR
               for r in caplog.records)
    with pytest.raises(PolicyError):
        Policy.to_items(doc(None, get_balance=mask))


def test_unknown_mask_field_is_an_error_and_the_phrase_is_kept(
        caplog: pytest.LogCaptureFixture) -> None:
    mask = {"phrase_key": "mask.get_balance", "extra": 1}
    with pytest.raises(PolicySectionError) as exc:
        parse_mask("get_balance", {"mask": mask})
    assert exc.value.field == "tools.get_balance.mask.extra"
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        p = Policy.from_dict(doc(None, get_balance=mask))
    assert p.mask_phrase_key("get_balance") == "mask.get_balance"
    assert any("tools.get_balance.mask.extra" in r.getMessage() for r in caplog.records)
    with pytest.raises(PolicyError):
        Policy.to_items(doc(None, get_balance=mask))


def test_per_tool_delays_parse_and_round_trip_through_the_store() -> None:
    mask = {"phrase_key": "mask.get_balance", "after_ms": 0, "handoff_wait_ms": 1200}
    d = doc(None, get_balance=mask, list_recent_transactions={
        "phrase_key": "mask.list_recent_transactions", "after_ms": 900})
    store = MemoryStore()
    for item in Policy.to_items(d):
        store.put("policy", item)
    assert store.get("policy", {"pk": "tool#get_balance"})["mask"] == mask  # type: ignore[index]
    for p in (Policy.from_dict(d), Policy.from_store(store)):
        assert p.tool_mask("get_balance") == ToolMask("mask.get_balance", 0, 1200)
        assert p.tool_mask("list_recent_transactions") == ToolMask(
            "mask.list_recent_transactions", 900, None)
        assert p.tool_mask("authenticate") is None and p.tool_mask("nope") is None
        assert p.tool_row("get_balance")["mask"] == mask  # type: ignore[index]


def test_bad_per_tool_delay_keeps_the_phrase_and_uses_the_default(
        caplog: pytest.LogCaptureFixture) -> None:
    mask = {"phrase_key": "mask.get_balance", "after_ms": 10001, "handoff_wait_ms": 100}
    with pytest.raises(PolicySectionError) as exc:
        parse_mask("get_balance", {"mask": mask})
    assert exc.value.field == "tools.get_balance.mask.after_ms"
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        p = Policy.from_dict(doc(None, get_balance=mask))
    assert p.tool_mask("get_balance") == ToolMask("mask.get_balance", None, 100)
    assert any("tools.get_balance.mask.after_ms" in r.getMessage() and r.levelno == logging.ERROR
               for r in caplog.records)


def test_masks_round_trip_through_the_store() -> None:
    store = MemoryStore()
    for item in Policy.to_items(DOC):
        store.put("policy", item)
    assert store.get("policy", {"pk": "tool#get_balance"})["mask"] == {  # type: ignore[index]
        "phrase_key": "mask.get_balance"}
    assert "mask" not in store.get("policy", {"pk": "tool#authenticate"})  # type: ignore[operator]
    a, b = Policy.from_dict(DOC), Policy.from_store(store)
    for tool in a.tools:
        assert a.mask_phrase_key(tool) == b.mask_phrase_key(tool)
        assert a.tool_row(tool) == b.tool_row(tool)
    assert "mask" not in (a.tool_row("get_balance") or {}).get("extra", {})


def test_referenced_prompt_keys_include_masks() -> None:
    keys = Policy.from_dict(DOC).referenced_prompt_keys()
    assert {"mask.get_balance", "mask.change_address", "mask.list_recent_transactions",
            *MASK_GENERIC_KEYS} <= keys
    assert MASK_GENERIC_KEYS[0] == MASK_GENERIC_KEY  # the alias is the pool's first phrase
    off = Policy.from_dict(doc({"after_ms": 0})).referenced_prompt_keys()
    assert not set(MASK_GENERIC_KEYS) & off
