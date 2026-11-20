"""scripts/seed.py keeps the stored transcript redaction setting unless --overwrite-redaction."""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any

import pytest

from owlbank.pii import parse_redaction_config
from owlbank.policy import Policy
from owlbank.store import MemoryStore
from scripts import seed
from tests.unit.scripts.test_seed import TABLE_ARGS, load, run

POLICY = load("policy.json")
SEED_REDACTION = Policy.to_items(POLICY)
SEED_BODY = next({k: v for k, v in i.items() if k != "pk"} for i in SEED_REDACTION if i["pk"] == "redaction")
#: The phase 1 shape: no ``enabled``, an ``apply_to`` list.
PHASE1 = {"pk": "redaction", "entity_types": ["SSN", "PIN"], "min_confidence": "0.8",
          "mask": "[X:{type}]", "on_failure": "allow", "apply_to": ["escalation_summary", "ai_turn"]}


def merged(live: dict[str, Any] | None, *, overwrite: bool = False) -> tuple[seed.RedactionMerge, dict[str, Any]]:
    merge = seed.merge_redaction(SEED_REDACTION, {"redaction": live}, overwrite=overwrite)
    return merge, next(i for i in merge.items if i["pk"] == "redaction")


def redaction_item(store: MemoryStore) -> dict[str, Any] | None:
    return store.get("policy", {"pk": "redaction"})


def seeded_store() -> MemoryStore:
    store = MemoryStore()
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    return store


# --------------------------------------------------------------------------- pure merge


def test_seed_file_redaction_is_enabled_without_apply_to() -> None:
    assert SEED_BODY["enabled"] is True and "apply_to" not in SEED_BODY


def test_new_item_takes_the_seed_file_with_rev_1() -> None:
    merge, item = merged(None)
    assert item == {"pk": "redaction", **SEED_BODY, "rev": 1}
    assert merge.new and merge.differences == [] and merge.dropped == [] and not merge.switched_off
    others = [i for i in merge.items if i["pk"] != "redaction"]
    assert others == [i for i in SEED_REDACTION if i["pk"] != "redaction"]  # untouched


def test_phase1_item_keeps_its_values_drops_apply_to_and_gains_enabled() -> None:
    merge, item = merged({**PHASE1, "rev": 4})
    assert item == {"pk": "redaction", "enabled": True, "entity_types": ["SSN", "PIN"],
                    "min_confidence": "0.8", "mask": "[X:{type}]", "on_failure": "allow", "rev": 5}
    assert merge.dropped == [("redaction.apply_to", ["escalation_summary", "ai_turn"])]
    assert ("redaction.enabled", True) in merge.added
    assert ("redaction.min_confidence", "0.8", SEED_BODY["min_confidence"]) in merge.differences
    assert merge.invalid == [] and not merge.switched_off
    parse_redaction_config({k: v for k, v in item.items() if k not in ("pk", "rev")})


def test_phase1_off_becomes_disabled_with_the_seed_types() -> None:
    """A live empty ``entity_types`` (phase 1 "off") stays off."""
    merge, item = merged({**PHASE1, "entity_types": []})
    assert item["enabled"] is False and item["entity_types"] == SEED_BODY["entity_types"]
    assert "apply_to" not in item and merge.switched_off and merge.invalid == []
    assert merge.switched_off_by == ["entity_types"]
    assert all(f != "redaction.entity_types" for f, _, _ in merge.differences)  # not "KEPT"
    report = seed.render_redaction_report(merge)
    assert "entity_types was empty" in report and "enabled: false" in report
    assert "redaction.apply_to" in report
    cfg = parse_redaction_config({k: v for k, v in item.items() if k not in ("pk", "rev")})
    assert cfg.enabled is False


def test_phase1_empty_apply_to_becomes_disabled_with_the_seed_types() -> None:
    """A live ``apply_to: []`` (phase 1: applied to nothing) is off, like an empty
    ``entity_types``: ``enabled: false`` with the seed file's types, reported."""
    merge, item = merged({**PHASE1, "apply_to": [], "rev": 6})
    assert item["enabled"] is False and item["entity_types"] == SEED_BODY["entity_types"]
    assert item["min_confidence"] == "0.8" and item["on_failure"] == "allow" and item["rev"] == 7
    assert "apply_to" not in item and merge.switched_off and merge.switched_off_by == ["apply_to"]
    assert merge.dropped == [("redaction.apply_to", [])] and merge.invalid == []
    assert all(f not in ("redaction.enabled", "redaction.entity_types") for f, _, _ in merge.differences)
    report = seed.render_redaction_report(merge)
    assert "apply_to was empty" in report and "enabled: false" in report
    assert "redaction.apply_to" in report
    cfg = parse_redaction_config({k: v for k, v in item.items() if k not in ("pk", "rev")})
    assert cfg.enabled is False


def test_empty_apply_to_and_empty_entity_types_report_both() -> None:
    merge, item = merged({**PHASE1, "apply_to": [], "entity_types": []})
    assert item["enabled"] is False and item["entity_types"] == SEED_BODY["entity_types"]
    assert merge.switched_off_by == ["apply_to", "entity_types"]
    assert "apply_to and entity_types was empty" not in seed.render_redaction_report(merge)
    assert "apply_to and entity_types were empty" in seed.render_redaction_report(merge)


def test_a_non_empty_apply_to_is_only_dropped() -> None:
    merge, item = merged({**PHASE1, "apply_to": ["ai_turn"], "enabled": True})
    assert item["enabled"] is True and item["entity_types"] == ["SSN", "PIN"]
    assert not merge.switched_off and merge.switched_off_by == []
    assert merge.dropped == [("redaction.apply_to", ["ai_turn"])]


def test_overwrite_with_an_empty_apply_to_writes_the_seed_values() -> None:
    merge, item = merged({**PHASE1, "apply_to": []}, overwrite=True)
    assert item == {"pk": "redaction", **SEED_BODY, "rev": 1}
    assert not merge.switched_off and merge.switched_off_by == []


def test_enabled_true_with_no_types_is_switched_off_too() -> None:
    merge, item = merged({"pk": "redaction", "enabled": True, "entity_types": []})
    assert merge.switched_off and item["enabled"] is False
    assert item["entity_types"] == SEED_BODY["entity_types"]


def test_disabled_with_no_types_is_a_valid_setting_and_kept_exactly() -> None:
    merge, item = merged({"pk": "redaction", "enabled": False, "entity_types": [], "rev": 2})
    assert item["enabled"] is False and item["entity_types"] == [] and not merge.switched_off
    assert item["rev"] == 3 and merge.invalid == []


def test_tuned_values_survive() -> None:
    live = {"pk": "redaction", **SEED_BODY, "enabled": False, "min_confidence": "0.9", "rev": 7}
    merge, item = merged(live)
    assert item["enabled"] is False and item["min_confidence"] == "0.9" and item["rev"] == 8
    assert ("redaction.enabled", False, True) in merge.differences
    assert merge.added == [] and merge.dropped == []


def test_overwrite_writes_the_seed_values() -> None:
    merge, item = merged({**PHASE1, "entity_types": [], "junk": 1, "rev": 2}, overwrite=True)
    assert item == {"pk": "redaction", **SEED_BODY, "rev": 3}
    assert not merge.switched_off
    assert ("redaction.apply_to", ["escalation_summary", "ai_turn"]) in merge.dropped
    assert ("redaction.junk", 1) in merge.dropped
    assert "REPLACED" in seed.render_redaction_report(merge)


def test_an_unknown_field_is_dropped_and_reported() -> None:
    merge, item = merged({"pk": "redaction", **SEED_BODY, "surfaces": ["x"]})
    assert "surfaces" not in item and ("redaction.surfaces", ["x"]) in merge.dropped


def test_invalid_live_value_is_kept_with_a_warning() -> None:
    merge, item = merged({"pk": "redaction", **SEED_BODY, "mask": ""})
    assert item["mask"] == ""
    assert merge.invalid and merge.invalid[0].startswith("redaction.mask")
    report = seed.render_redaction_report(merge)
    assert "WARNING: live redaction setting invalid: redaction.mask" in report
    assert "--overwrite-redaction" in report


def test_dynamodb_decimals_are_storable() -> None:
    _, item = merged({"pk": "redaction", **SEED_BODY, "min_confidence": Decimal("0.7"), "rev": Decimal(3)})
    assert item["min_confidence"] == "0.7" and item["rev"] == 4
    parse_redaction_config({k: v for k, v in item.items() if k not in ("pk", "rev")})


def test_no_redaction_in_the_seed_file_leaves_the_items_alone() -> None:
    items = [i for i in SEED_REDACTION if i["pk"] != "redaction"]
    merge = seed.merge_redaction(items, {"redaction": PHASE1}, overwrite=False)
    assert merge.items == items and merge.dropped == []


# --------------------------------------------------------------------------- through main


def phase1_store() -> MemoryStore:
    store = seeded_store()
    store.put("policy", PHASE1)
    return store


def test_fresh_table_gets_the_seed_setting() -> None:
    store = seeded_store()
    item = redaction_item(store)
    assert item is not None and item["rev"] == 1
    assert {k: v for k, v in item.items() if k not in ("pk", "rev")} == SEED_BODY
    assert Policy.from_store(store).redaction.enabled is True


def test_phase1_table_reseed_drops_apply_to_and_reports_it(capsys: pytest.CaptureFixture[str]) -> None:
    store = phase1_store()
    capsys.readouterr()
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    item = redaction_item(store)
    assert item is not None and "apply_to" not in item and item["enabled"] is True
    assert item["entity_types"] == ["SSN", "PIN"] and item["on_failure"] == "allow"
    out = capsys.readouterr().out
    assert "redaction.apply_to" in out and "dropped" in out
    cfg = Policy.from_store(store).redaction
    assert cfg.entity_types == ("SSN", "PIN") and cfg.min_confidence == 0.8


def test_phase1_off_stays_off_after_a_reseed(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded_store()
    store.put("policy", {**PHASE1, "entity_types": []})
    capsys.readouterr()
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    cfg = Policy.from_store(store).redaction
    assert cfg.enabled is False and cfg.entity_types == tuple(SEED_BODY["entity_types"])
    assert "entity_types was empty" in capsys.readouterr().out


def test_phase1_empty_apply_to_stays_off_after_a_reseed(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded_store()
    store.put("policy", {**PHASE1, "apply_to": []})
    capsys.readouterr()
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    item = redaction_item(store)
    assert item is not None and "apply_to" not in item
    cfg = Policy.from_store(store).redaction
    assert cfg.enabled is False and cfg.entity_types == tuple(SEED_BODY["entity_types"])
    assert "apply_to was empty" in capsys.readouterr().out


def test_dry_run_reports_the_n14_changes_and_writes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded_store()
    store.put("policy", {**PHASE1, "entity_types": []})
    capsys.readouterr()
    assert run(["--stage", "prod", *TABLE_ARGS], store) == 0
    assert redaction_item(store) == {**PHASE1, "entity_types": []}
    out = capsys.readouterr().out
    assert out.startswith("DRY RUN") and "redaction.apply_to" in out and "entity_types was empty" in out
    line = next(ln for ln in out.splitlines() if ln.startswith("{") and '"pk": "redaction"' in ln)
    shown = json.loads(line)
    assert shown["enabled"] is False and "apply_to" not in shown


def test_overwrite_redaction_restores_the_seed_value_with_a_warning(
        capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
    store = seeded_store()
    store.update("policy", {"pk": "redaction"}, {"enabled": False, "rev": 4})
    with caplog.at_level(logging.INFO, logger="owlbank.seed"):
        assert run(["--stage", "prod", *TABLE_ARGS, "--overwrite-redaction", "--apply"], store) == 0
    item = redaction_item(store)
    assert item is not None and item["enabled"] is True and item["rev"] == 5
    assert "REPLACED" in capsys.readouterr().out
    warn = [r for r in caplog.records if r.levelno == logging.WARNING
            and getattr(r, "event", None) == "script_step" and getattr(r, "outcome", None) == "overwrite"]
    assert [getattr(r, "step", None) for r in warn] == ["redaction"]


def test_tuned_setting_survives_without_the_flag(caplog: pytest.LogCaptureFixture) -> None:
    store = seeded_store()
    store.update("policy", {"pk": "redaction"}, {"enabled": False, "rev": 4})
    with caplog.at_level(logging.INFO, logger="owlbank.seed"):
        assert run(["--stage", "prod", *TABLE_ARGS, "--reseed", "--apply"], store) == 0
    item = redaction_item(store)
    assert item is not None and item["enabled"] is False and item["rev"] == 5
    assert not [r for r in caplog.records if getattr(r, "outcome", None) == "overwrite"]


def test_matching_table_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded_store()
    capsys.readouterr()
    assert run(["--stage", "prod", *TABLE_ARGS], store) == 0
    assert "redaction setting: the table matches policy.json" in capsys.readouterr().out


def test_dry_run_without_a_policy_table_name_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--stage", "prod"], MemoryStore()) == 0
    assert "redaction setting: the live table wasn't read" in capsys.readouterr().out
