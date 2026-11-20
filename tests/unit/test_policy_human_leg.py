"""owlbank.policy: the ``human_leg`` switch, and
the ``memory`` and ``end_call`` switches."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from owlbank.policy import END_CALL_ITEM, HUMAN_LEG_ITEM, MEMORY_ITEM, Policy, load_end_call
from owlbank.store import MemoryStore

SEED: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[2] / "data" / "policy.json").read_text(encoding="utf-8"))


def seeded(doc: dict[str, Any]) -> MemoryStore:
    store = MemoryStore()
    for item in Policy.to_items(doc):
        store.put("policy", item)
    return store


def test_the_shipped_seed_has_it_off() -> None:
    assert SEED[HUMAN_LEG_ITEM] == {"transcription": False, "agent_legs": True}
    assert Policy.from_dict(SEED).human_leg_transcription is False
    assert Policy.from_store(seeded(SEED)).human_leg_transcription is False


def test_a_missing_item_reads_as_off() -> None:
    doc = copy.deepcopy(SEED)
    del doc[HUMAN_LEG_ITEM]
    assert Policy.from_dict(doc).human_leg_transcription is False
    assert [i for i in Policy.to_items(doc) if i["pk"] == HUMAN_LEG_ITEM] == [
        {"pk": HUMAN_LEG_ITEM, "transcription": False, "agent_legs": True}]
    store = MemoryStore()
    for item in Policy.to_items(doc):
        if item["pk"] != HUMAN_LEG_ITEM:
            store.put("policy", item)
    assert Policy.from_store(store).human_leg_transcription is False


@pytest.mark.parametrize("raw", [None, [], "on", {}, {"transcription": "true"}, {"transcription": 1},
                                 {"transcription": None}, {"other": True}])
def test_malformed_reads_as_off(raw: Any) -> None:
    doc = copy.deepcopy(SEED)
    doc[HUMAN_LEG_ITEM] = raw
    assert Policy.from_dict(doc).human_leg_transcription is False
    store = seeded(SEED)
    store.put("policy", {"pk": HUMAN_LEG_ITEM, "transcription": raw})
    assert Policy.from_store(store).human_leg_transcription is False


def test_true_is_read_and_a_change_shows_after_the_ttl() -> None:
    store = seeded(SEED)
    now = [0.0]
    policy = Policy.from_store(store, ttl_s=60, clock=lambda: now[0])
    assert policy.human_leg_transcription is False
    store.put("policy", {"pk": HUMAN_LEG_ITEM, "transcription": True})
    now[0] = 59
    assert policy.human_leg_transcription is False
    now[0] = 61
    assert policy.human_leg_transcription is True


def test_seeding_writes_the_switch_value() -> None:
    doc = copy.deepcopy(SEED)
    doc[HUMAN_LEG_ITEM] = {"transcription": True}
    assert {"pk": HUMAN_LEG_ITEM, "transcription": True, "agent_legs": True} in Policy.to_items(doc)
    assert Policy.from_store(seeded(doc)).human_leg_transcription is True


# --- human_leg.agent_legs and memory.write: both default true ---


def test_the_shipped_seed_has_both_on() -> None:
    assert SEED[MEMORY_ITEM] == {"write": True}
    for policy in (Policy.from_dict(SEED), Policy.from_store(seeded(SEED))):
        assert policy.human_leg_agent_legs is True
        assert policy.memory_write is True


def test_missing_items_read_as_on() -> None:
    doc = copy.deepcopy(SEED)
    del doc[HUMAN_LEG_ITEM]
    del doc[MEMORY_ITEM]
    assert Policy.from_dict(doc).human_leg_agent_legs is True
    assert Policy.from_dict(doc).memory_write is True
    items = Policy.to_items(doc)
    assert {"pk": MEMORY_ITEM, "write": True} in items
    assert {"pk": HUMAN_LEG_ITEM, "transcription": False, "agent_legs": True} in items
    store = MemoryStore()  # no human_leg or memory item at all
    for item in items:
        if item["pk"] not in (HUMAN_LEG_ITEM, MEMORY_ITEM):
            store.put("policy", item)
    policy = Policy.from_store(store)
    assert policy.human_leg_agent_legs is True
    assert policy.memory_write is True


@pytest.mark.parametrize("raw", [None, [], "off", {}, {"agent_legs": "false"}, {"agent_legs": 0},
                                 {"agent_legs": None}, {"other": False}])
def test_malformed_agent_legs_reads_as_on(raw: Any) -> None:
    doc = copy.deepcopy(SEED)
    doc[HUMAN_LEG_ITEM] = raw
    assert Policy.from_dict(doc).human_leg_agent_legs is True
    store = seeded(SEED)
    store.put("policy", {"pk": HUMAN_LEG_ITEM, "agent_legs": raw})
    assert Policy.from_store(store).human_leg_agent_legs is True


@pytest.mark.parametrize("raw", [None, [], "off", {}, {"write": "false"}, {"write": 0},
                                 {"write": None}, {"other": False}])
def test_malformed_memory_write_reads_as_on(raw: Any) -> None:
    doc = copy.deepcopy(SEED)
    doc[MEMORY_ITEM] = raw
    assert Policy.from_dict(doc).memory_write is True
    store = seeded(SEED)
    store.put("policy", {"pk": MEMORY_ITEM, "write": raw})
    assert Policy.from_store(store).memory_write is True


def test_false_is_read_and_a_change_shows_after_the_ttl() -> None:
    store = seeded(SEED)
    now = [0.0]
    policy = Policy.from_store(store, ttl_s=60, clock=lambda: now[0])
    assert (policy.human_leg_agent_legs, policy.memory_write) == (True, True)
    store.put("policy", {"pk": HUMAN_LEG_ITEM, "transcription": False, "agent_legs": False})
    store.put("policy", {"pk": MEMORY_ITEM, "write": False})
    now[0] = 59
    assert (policy.human_leg_agent_legs, policy.memory_write) == (True, True)
    now[0] = 61
    assert (policy.human_leg_agent_legs, policy.memory_write) == (False, False)


def test_seeding_round_trips_false() -> None:
    doc = copy.deepcopy(SEED)
    doc[HUMAN_LEG_ITEM] = {"transcription": True, "agent_legs": False}
    doc[MEMORY_ITEM] = {"write": False}
    items = Policy.to_items(doc)
    assert {"pk": HUMAN_LEG_ITEM, "transcription": True, "agent_legs": False} in items
    assert {"pk": MEMORY_ITEM, "write": False} in items
    policy = Policy.from_store(seeded(doc))
    assert (policy.human_leg_transcription, policy.human_leg_agent_legs, policy.memory_write) == (True, False, False)


# --- end_call.enabled: default true; the tool's row is a level-0 hand-off ---


def test_the_shipped_seed_has_end_call_on_with_a_handoff_row() -> None:
    assert SEED[END_CALL_ITEM] == {"enabled": True}
    assert SEED["tools"]["end_call"] == {"min_level": 0, "phrase_key": None, "kind": "handoff",
                                         "confirm": "none"}
    for policy in (Policy.from_dict(SEED), Policy.from_store(seeded(SEED))):
        assert policy.end_call_enabled() is True
        assert policy.min_level("end_call") == 0


def test_a_missing_end_call_item_reads_as_on(caplog: pytest.LogCaptureFixture) -> None:
    doc = copy.deepcopy(SEED)
    del doc[END_CALL_ITEM]
    assert Policy.from_dict(doc).end_call_enabled() is True
    items = Policy.to_items(doc)
    assert [i for i in items if i["pk"] == END_CALL_ITEM] == [{"pk": END_CALL_ITEM, "enabled": True}]
    store = MemoryStore()
    for item in items:
        if item["pk"] != END_CALL_ITEM:
            store.put("policy", item)
    assert Policy.from_store(store).end_call_enabled() is True
    assert "end_call.enabled" not in caplog.text  # absent is the pre-switch seed, not an error


@pytest.mark.parametrize("raw", [[], "off", {}, {"enabled": "false"}, {"enabled": 0},
                                 {"enabled": None}, {"other": False}])
def test_malformed_end_call_reads_as_on_with_a_warning(raw: Any, caplog: pytest.LogCaptureFixture) -> None:
    doc = copy.deepcopy(SEED)
    doc[END_CALL_ITEM] = raw
    assert Policy.from_dict(doc).end_call_enabled() is True
    store = seeded(SEED)
    store.put("policy", {"pk": END_CALL_ITEM, "enabled": raw})
    assert Policy.from_store(store).end_call_enabled() is True
    assert "end_call.enabled must be true or false" in caplog.text


def test_end_call_false_is_read_and_round_trips() -> None:
    assert load_end_call({"enabled": False}) is False
    doc = copy.deepcopy(SEED)
    doc[END_CALL_ITEM] = {"enabled": False}
    assert {"pk": END_CALL_ITEM, "enabled": False} in Policy.to_items(doc)
    assert Policy.from_dict(doc).end_call_enabled() is False
    assert Policy.from_store(seeded(doc)).end_call_enabled() is False


def test_an_end_call_change_shows_after_the_ttl() -> None:
    store = seeded(SEED)
    now = [0.0]
    policy = Policy.from_store(store, ttl_s=60, clock=lambda: now[0])
    assert policy.end_call_enabled() is True
    store.put("policy", {"pk": END_CALL_ITEM, "enabled": False})
    now[0] = 59
    assert policy.end_call_enabled() is True
    now[0] = 61
    assert policy.end_call_enabled() is False
