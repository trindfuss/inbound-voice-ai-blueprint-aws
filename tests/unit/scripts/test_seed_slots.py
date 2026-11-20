"""scripts/seed.py keeps console-tuned slot settings unless --overwrite-slots."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

import pytest

from owlbank.policy import Policy
from owlbank.slot_policy import RESOLUTION_ITEM
from owlbank.store import MemoryStore
from scripts import seed
from scripts._common import DATA_DIR, ScriptError
from scripts.hash_tool import verify
from tests.unit.scripts.test_seed import DEV_PHONE, TABLE_ARGS, dynamo, run  # noqa: F401 (fixture)


def load(name: str) -> Any:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


POLICY = load("policy.json")
FIXTURES = load("fixtures.json")
CATALOG = load("prompt_catalog.json")
SLOT_NAMES = list(POLICY["slots"])
SLOT_PKS = [f"slot#{n}" for n in SLOT_NAMES]
MERGED_PKS = [*SLOT_PKS, RESOLUTION_ITEM]  # the order Policy.to_items gives
OPTION_KEYS = sorted((e["key"], e["lang"]) for e in CATALOG["entries"] if e["key"].startswith("option."))


def slot_item(store: MemoryStore, name: str) -> dict[str, Any] | None:
    return store.get("policy", {"pk": f"slot#{name}"})


def seeded_store() -> MemoryStore:
    store = MemoryStore()
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    return store


# --------------------------------------------------------------------------- pure merge


def test_slot_keys_are_the_seeded_slot_items() -> None:
    items = Policy.to_items(POLICY)
    assert seed.slot_keys(items) == SLOT_PKS


def test_merged_keys_are_slot_resolution_and_the_slot_items() -> None:
    items = Policy.to_items(POLICY)
    assert seed.slot_merge_keys(items) == MERGED_PKS


def test_new_items_take_the_seed_file_with_rev_1() -> None:
    items = Policy.to_items(POLICY)
    merge = seed.merge_slots(items, dict.fromkeys(MERGED_PKS), overwrite=False)
    by_pk = {i["pk"]: i for i in merge.items}
    for seeded in items:
        if seeded["pk"] in MERGED_PKS:
            assert by_pk[seeded["pk"]] == {**seeded, "rev": 1}
        else:
            assert by_pk[seeded["pk"]] == seeded  # everything else untouched
    assert merge.differences == [] and merge.invalid == []
    assert merge.new == MERGED_PKS


def test_live_field_kept_absent_field_filled_rev_bumped() -> None:
    items = Policy.to_items(POLICY)
    live = {"slot#zip": {"pk": "slot#zip", "keypad_only": True, "rev": 6}}
    merge = seed.merge_slots(items, live, overwrite=False)
    zip_item = next(i for i in merge.items if i["pk"] == "slot#zip")
    assert zip_item == {"pk": "slot#zip", "sensitive": False, "keypad_only": True,
                        "barge_in": True, "confirm": "when_inferred", "declinable": True, "rev": 7}
    assert ("slots.zip.keypad_only", True, False) in merge.differences
    assert {f for f, _ in merge.added} == {"slots.zip.sensitive", "slots.zip.barge_in",
                                          "slots.zip.confirm", "slots.zip.declinable"}


def test_overwrite_writes_the_seed_values_and_drops_unknown_fields() -> None:
    items = Policy.to_items(POLICY)
    live = {"slot#ssn4": {"pk": "slot#ssn4", "sensitive": False, "junk": 1, "rev": 2}}
    merge = seed.merge_slots(items, live, overwrite=True)
    got = next(i for i in merge.items if i["pk"] == "slot#ssn4")
    assert got == {"pk": "slot#ssn4", "sensitive": True, "keypad_only": False, "barge_in": True,
                   "confirm": "never", "declinable": True, "rev": 3}
    assert ("slots.ssn4.sensitive", False, True) in merge.differences
    assert ("slots.ssn4.junk", 1, seed.NOT_SET) in merge.differences


def test_invalid_live_value_is_kept_with_a_warning() -> None:
    items = Policy.to_items(POLICY)
    live = {"slot#otp": {"pk": "slot#otp", "sensitive": "no", "rev": 1}}
    merge = seed.merge_slots(items, live, overwrite=False)
    got = next(i for i in merge.items if i["pk"] == "slot#otp")
    assert got["sensitive"] == "no"
    assert any("slot#otp.sensitive" in p for p in merge.invalid)
    report = seed.render_slots_report(merge)
    assert "WARNING: live slot setting invalid: slot#otp.sensitive" in report
    assert "--overwrite-slots" in report
    assert '"no"' not in "".join(merge.invalid)  # problems name the field, never the value


def test_a_live_slot_not_in_the_seed_file_is_not_touched_by_the_merge() -> None:
    items = Policy.to_items(POLICY)
    live = {"slot#retired": {"pk": "slot#retired", "sensitive": True}}
    merge = seed.merge_slots(items, live, overwrite=False)
    assert "slot#retired" not in {i["pk"] for i in merge.items}


def test_dynamodb_decimals_are_storable() -> None:
    from decimal import Decimal

    items = Policy.to_items(POLICY)
    live = {"slot#yes_no": {"pk": "slot#yes_no", "maxRetries": Decimal(3), "rev": Decimal(4)}}
    merge = seed.merge_slots(items, live, overwrite=False)
    got = next(i for i in merge.items if i["pk"] == "slot#yes_no")
    assert got["maxRetries"] == 3 and type(got["maxRetries"]) is int and got["rev"] == 5


def test_tuned_slot_resolution_kept_absent_fields_filled() -> None:
    items = Policy.to_items(POLICY)
    live = {RESOLUTION_ITEM: {"pk": RESOLUTION_ITEM, "default_mode": "deterministic",
                              "flows": {"login": {"mode": "assisted"}}, "rev": 3}}
    merge = seed.merge_slots(items, live, overwrite=False)
    got = next(i for i in merge.items if i["pk"] == RESOLUTION_ITEM)
    assert got == {"pk": RESOLUTION_ITEM, "default_mode": "deterministic",
                   "flows": {"login": {"mode": "assisted"}}, "generative_enabled": False,
                   "resolver_budget_ms": 2500, "rev": 4}
    assert ("slot_resolution.default_mode", "deterministic", "assisted") in merge.differences
    assert {f for f, _ in merge.added} == {"slot_resolution.generative_enabled",
                                          "slot_resolution.resolver_budget_ms"}
    over = seed.merge_slots(items, live, overwrite=True)
    assert next(i for i in over.items if i["pk"] == RESOLUTION_ITEM)["default_mode"] == "assisted"


def test_invalid_live_slot_resolution_is_kept_with_a_warning() -> None:
    items = Policy.to_items(POLICY)
    live = {RESOLUTION_ITEM: {"pk": RESOLUTION_ITEM, "default_mode": "fast"}}
    merge = seed.merge_slots(items, live, overwrite=False)
    assert next(i for i in merge.items if i["pk"] == RESOLUTION_ITEM)["default_mode"] == "fast"
    assert any(p.startswith("slot_resolution.default_mode") for p in merge.invalid)
    assert "fast" not in "".join(merge.invalid)


def test_phase1_slot_item_gains_the_phase2_fields_and_keeps_tuned_ones() -> None:
    """The new seed fields are written into a live phase 1 item; tuned values are kept."""
    items = Policy.to_items(POLICY)
    live = {"slot#ssn4": {"pk": "slot#ssn4", "sensitive": True, "keypad_only": True,
                          "barge_in": True, "mode": "deterministic", "rev": 2}}
    merge = seed.merge_slots(items, live, overwrite=False)
    got = next(i for i in merge.items if i["pk"] == "slot#ssn4")
    assert got["confirm"] == "never" and got["keypad_only"] is True and got["mode"] == "deterministic"
    assert ("slots.ssn4.confirm", "never") in merge.added


# --------------------------------------------------------------------------- catalog option names


def test_option_entries_absent_from_the_table_are_written() -> None:
    items = seed.PromptCatalog.to_items(CATALOG)
    merge = seed.merge_catalog_prompts(items, {}, overwrite=False)
    assert merge.items == [{**i, seed.SEED_HASH: seed.seed_hash(i["text"])} if "text" in i else i
                           for i in items] and merge.kept == []
    assert sorted(k for k in merge.new if k[0].startswith("option.")) == OPTION_KEYS


def _live(items: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(i["key"], i["lang"]): copy.deepcopy(i) for i in items}


def test_option_entry_edited_in_the_table_is_not_written() -> None:
    items = seed.PromptCatalog.to_items(CATALOG)
    live = _live(items)
    live[("option.card.red", "es-US")]["text"] = "La Roja"
    merge = seed.merge_catalog_prompts(items, live, overwrite=False)
    keys = {(i["key"], i["lang"]) for i in merge.items}
    assert ("option.card.red", "es-US") not in keys
    assert merge.kept == [("option.card.red", "es-US")]
    assert "option.card.red (es-US)" in seed.render_prompts_report(merge)


def test_overwrite_slots_writes_the_option_entries_too() -> None:
    items = seed.PromptCatalog.to_items(CATALOG)
    live = _live(items)
    live[("option.card.red", "es-US")]["text"] = "La Roja"
    merge = seed.merge_catalog_prompts(items, live, overwrite=False, overwrite_options=True)
    written = {(i["key"], i["lang"]): i for i in merge.items}
    assert written[("option.card.red", "es-US")]["text"] != "La Roja"
    assert merge.replaced == [("option.card.red", "es-US")] and merge.kept == []
    assert "REPLACED" in seed.render_prompts_report(merge)


# --------------------------------------------------------------------------- seed-time cross-check


def test_the_seed_files_pass_the_option_cross_check() -> None:
    seed.check_options(POLICY, CATALOG)  # no error


def test_an_option_without_its_catalog_entry_in_one_language_is_refused() -> None:
    catalog = copy.deepcopy(CATALOG)
    catalog["entries"] = [e for e in catalog["entries"]
                          if (e["key"], e["lang"]) != ("option.card.purple", "es-US")]
    with pytest.raises(ScriptError, match=r"option\.card\.purple \(es-US\)"):
        seed.build_plan(FIXTURES, POLICY, catalog, stage="prod")


def test_an_alias_shared_by_two_options_is_refused() -> None:
    policy = copy.deepcopy(POLICY)
    policy["slots"]["card"]["options"][1]["aliases"]["en-US"] = ["purple", "OwlBank Red Card"]
    with pytest.raises(ScriptError, match="alias"):
        seed.build_plan(FIXTURES, policy, CATALOG, stage="prod")


def test_the_cross_check_exits_2_through_main(tmp_path: Any, capsys: pytest.CaptureFixture[str]) -> None:
    import shutil

    for name in ("fixtures.json", "policy.json", "prompt_catalog.json"):
        shutil.copy(DATA_DIR / name, tmp_path / name)
    catalog = copy.deepcopy(CATALOG)
    catalog["entries"] = [e for e in catalog["entries"] if e["key"] != "option.card.rewards"]
    (tmp_path / "prompt_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    assert run(["--stage", "prod", "--data-dir", str(tmp_path)], MemoryStore()) == 2
    assert "option.card.rewards" in capsys.readouterr().err


# --------------------------------------------------------------------------- through main


def test_fresh_table_gets_every_slot_item(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded_store()
    for name in SLOT_NAMES:
        item = slot_item(store, name)
        assert item is not None and item["rev"] == 1
        assert {k: v for k, v in item.items() if k not in ("pk", "rev")} == POLICY["slots"][name]
    assert sorted(SLOT_NAMES) == ["card", "otp", "phone", "ssn4", "yes_no", "zip"]
    res = store.get("policy", {"pk": RESOLUTION_ITEM})
    assert res is not None and res["rev"] == 1 and res["default_mode"] == "assisted"
    assert store.get("policy", {"pk": "meta"})["slots"] == SLOT_NAMES
    out = capsys.readouterr().out
    assert "not in the table yet" in out
    # the policy loads from the table with no slot problems
    assert Policy.from_store(store).slot_problems == ()


def test_tuned_keypad_only_survives_a_reseed_and_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded_store()
    store.update("policy", {"pk": "slot#zip"}, {"keypad_only": True, "rev": 2})
    capsys.readouterr()
    assert run(["--stage", "prod", *TABLE_ARGS, "--reseed", "--apply"], store) == 0
    item = slot_item(store, "zip")
    assert item is not None and item["keypad_only"] is True and item["rev"] == 3
    out = capsys.readouterr().out
    assert "KEPT" in out and "field | live | seed file" in out
    assert "slots.zip.keypad_only | true | false" in out
    assert Policy.from_store(store).slot("zip").keypad_only is True


def test_overwrite_slots_restores_the_seed_value_with_a_warning(
        capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
    store = seeded_store()
    store.update("policy", {"pk": "slot#zip"}, {"keypad_only": True, "rev": 2})
    with caplog.at_level(logging.INFO, logger="owlbank.seed"):
        assert run(["--stage", "prod", *TABLE_ARGS, "--overwrite-slots", "--apply"], store) == 0
    item = slot_item(store, "zip")
    assert item is not None and item["keypad_only"] is False and item["rev"] == 3
    assert "REPLACED" in capsys.readouterr().out
    warn = [r for r in caplog.records if r.levelno == logging.WARNING
            and getattr(r, "event", None) == "script_step" and getattr(r, "outcome", None) == "overwrite"]
    assert len(warn) == 1 and getattr(warn[0], "step", None) == "slots"


def test_no_overwrite_warning_without_the_flag(caplog: pytest.LogCaptureFixture) -> None:
    store = seeded_store()
    with caplog.at_level(logging.INFO, logger="owlbank.seed"):
        assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    assert not [r for r in caplog.records if getattr(r, "outcome", None) == "overwrite"]


def test_dry_run_prints_the_differences_and_writes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded_store()
    store.update("policy", {"pk": "slot#ssn4"}, {"barge_in": False, "rev": 5})
    before = {pk: store.get("policy", {"pk": pk}) for pk in SLOT_PKS}
    capsys.readouterr()
    assert run(["--stage", "prod", *TABLE_ARGS], store) == 0
    assert {pk: store.get("policy", {"pk": pk}) for pk in SLOT_PKS} == before
    out = capsys.readouterr().out
    assert out.startswith("DRY RUN") and "slots.ssn4.barge_in | false | true" in out
    # the printed plan shows what an apply would write: the live value, rev bumped
    line = next(ln for ln in out.splitlines() if '"pk": "slot#ssn4"' in ln)
    assert json.loads(line)["barge_in"] is False and json.loads(line)["rev"] == 6


def test_dry_run_with_overwrite_says_replaced_and_writes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded_store()
    store.update("policy", {"pk": "slot#ssn4"}, {"barge_in": False})
    capsys.readouterr()
    assert run(["--stage", "prod", *TABLE_ARGS, "--overwrite-slots"], store) == 0
    assert slot_item(store, "ssn4")["barge_in"] is False  # type: ignore[index]
    assert "REPLACED" in capsys.readouterr().out


def test_dry_run_without_a_policy_table_name_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--stage", "prod"], MemoryStore()) == 0
    assert "slot settings: the live table wasn't read" in capsys.readouterr().out


def test_matching_table_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded_store()
    capsys.readouterr()
    assert run(["--stage", "prod", *TABLE_ARGS], store) == 0
    assert "slot settings: the table matches policy.json" in capsys.readouterr().out


def test_developer_factor_is_unchanged_by_the_slot_merge() -> None:
    store = MemoryStore()
    assert run(["--stage", "dev", *TABLE_ARGS, "--developer-factor", "zip=55555", "--apply"], store) == 0
    assert slot_item(store, "zip") is not None


def test_kept_values_round_trip_through_dynamodb(dynamo: Any) -> None:  # noqa: F811
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], dynamo) == 0
    dynamo.update("policy", {"pk": "slot#yes_no"}, {"keypad_only": True, "maxRetries": 2})
    assert run(["--stage", "prod", *TABLE_ARGS, "--reseed", "--apply"], dynamo) == 0
    item = dynamo.get("policy", {"pk": "slot#yes_no"})
    assert (item["keypad_only"], item["maxRetries"], item["rev"]) == (True, 2, 2)


# --------------------------------------------------------------------------- a phase 1 table


def phase1_store() -> MemoryStore:
    """A table as phase 1 left it: no ``slot_resolution``, no ``slot#card``, no ``confirm`` (or
    ``mode``/``options``) on the slot items, no ``option.*`` catalog entries."""
    store = seeded_store()
    store.delete("policy", {"pk": RESOLUTION_ITEM})
    store.delete("policy", {"pk": "slot#card"})
    for name in ("zip", "ssn4", "otp", "yes_no"):
        item = slot_item(store, name)
        assert item is not None
        store.put("policy", {k: v for k, v in item.items() if k not in ("confirm", "mode", "options")})
    for key, lang in OPTION_KEYS:
        store.delete("catalog", {"key": key, "lang": lang})
    return store


def option_entry(store: MemoryStore, key: str, lang: str) -> dict[str, Any] | None:
    return store.get("catalog", {"key": key, "lang": lang})


def test_phase1_table_gets_the_phase2_items_and_fields(capsys: pytest.CaptureFixture[str]) -> None:
    store = phase1_store()
    capsys.readouterr()
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    res = store.get("policy", {"pk": RESOLUTION_ITEM})
    assert res is not None and res["rev"] == 1
    card = slot_item(store, "card")
    assert card is not None and card["rev"] == 1
    assert {k: v for k, v in card.items() if k not in ("pk", "rev")} == POLICY["slots"]["card"]
    for name in ("zip", "ssn4", "otp"):
        item = slot_item(store, name)
        assert item is not None and item["confirm"] == POLICY["slots"][name]["confirm"]
    for key, lang in OPTION_KEYS:
        assert option_entry(store, key, lang) is not None
    out = capsys.readouterr().out
    assert "slots.ssn4.confirm" in out and "slot_resolution" in out and "slot#card" in out
    policy = Policy.from_store(store)
    assert policy.slot_problems == () and policy.slot("ssn4").confirm == "never"
    assert [o.value for o in policy.slot("card").options] == ["red", "purple", "rewards"]


def test_phase1_dry_run_shows_the_phase2_fields_and_writes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    store = phase1_store()
    capsys.readouterr()
    assert run(["--stage", "prod", *TABLE_ARGS], store) == 0
    assert store.get("policy", {"pk": RESOLUTION_ITEM}) is None and slot_item(store, "card") is None
    assert option_entry(store, "option.card.red", "en-US") is None
    out = capsys.readouterr().out
    assert out.startswith("DRY RUN") and "slots.otp.confirm" in out
    assert "option.card.red (en-US)" in out


def test_tuned_keypad_only_and_mode_survive_a_phase2_reseed() -> None:
    store = phase1_store()
    store.update("policy", {"pk": "slot#zip"}, {"keypad_only": True, "mode": "deterministic"})
    assert run(["--stage", "prod", *TABLE_ARGS, "--reseed", "--apply"], store) == 0
    item = slot_item(store, "zip")
    assert item is not None and item["keypad_only"] is True and item["mode"] == "deterministic"
    assert item["confirm"] == "when_inferred"
    zip_slot = Policy.from_store(store).slot("zip")
    assert zip_slot.keypad_only is True and zip_slot.mode == "deterministic"


def test_tuned_option_name_survives_and_overwrite_slots_restores_it() -> None:
    store = seeded_store()
    store.update("catalog", {"key": "option.card.red", "lang": "en-US"}, {"text": "The Red One"})
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    assert option_entry(store, "option.card.red", "en-US")["text"] == "The Red One"  # type: ignore[index]
    assert run(["--stage", "prod", *TABLE_ARGS, "--overwrite-slots", "--apply"], store) == 0
    assert option_entry(store, "option.card.red", "en-US")["text"] == "OwlBank Red Card"  # type: ignore[index]


def test_developer_factor_handling_is_unchanged_on_a_phase1_table(capsys: pytest.CaptureFixture[str]) -> None:
    store = MemoryStore()
    assert run(["--stage", "dev", *TABLE_ARGS, "--developer-phone", DEV_PHONE, "--apply"], store) == 0
    for name in ("zip", "ssn4", "otp", "yes_no"):  # back to the phase 1 shape
        item = slot_item(store, name)
        assert item is not None
        store.put("policy", {k: v for k, v in item.items() if k != "confirm"})
    capsys.readouterr()
    assert run(["--stage", "dev", *TABLE_ARGS, "--developer-phone", DEV_PHONE,
                "--developer-factor", "zip=55555", "--reseed", "--apply"], store) == 0
    out = capsys.readouterr().out
    assert "55555" not in out
    fixture = FIXTURES["realNumberPlaceholders"]["developerPhone"]["fixture"]
    customer = store.get("customers", {"customer_id": fixture})
    assert customer is not None and verify("55555", customer["factors"]["zip"])
    assert slot_item(store, "zip")["confirm"] == "when_inferred"  # type: ignore[index]
    with pytest.raises(ScriptError, match="dev and test"):
        seed.build_plan(FIXTURES, POLICY, CATALOG, stage="prod", developer_factors={"zip": "55555"})
