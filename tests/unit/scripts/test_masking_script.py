"""scripts/masking.py: show and set masking values in the policy table."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from owlbank.store import DynamoStore, MemoryStore
from scripts import masking, seed
from tests.logutil import fields
from tests.unit.scripts.test_seed import TABLE_ARGS, dynamo  # noqa: F401 (fixture)
from tests.unit.scripts.test_seed_masking import CATALOG, FIXTURES, POLICY, SEED_AFTER_MS, SEED_TOOL_AFTER_MS


def seeded(store: Any = None) -> Any:
    store = store if store is not None else MemoryStore()
    seed.apply_plan(seed.build_plan(FIXTURES, POLICY, CATALOG, stage="prod"), store)
    return store


def run(*argv: str, store: Any, writer: Any = None) -> int:
    cmd, *rest = argv
    return masking.main([cmd, "--stage", "dev", *TABLE_ARGS, *rest], store=store, writer=writer)


def tool(store: Any, name: str) -> dict[str, Any]:
    return store.get("policy", {"pk": f"tool#{name}"})


def add_phrase(store: Any, key: str, langs: tuple[str, ...], text: str = "One moment.") -> None:
    for lang in langs:
        store.put("catalog", {"key": key, "lang": lang, "text": text, "version": 1, "status": "published",
                              "audio_url": None})
    meta = store.get("catalog", {"key": "_meta", "lang": "_"})
    meta["keys"] = [*meta["keys"], key]
    store.put("catalog", meta)


# --------------------------------------------------------------------------- show


def test_show_marks_own_values_and_defaults(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded()
    store.update("policy", {"pk": "tool#get_balance"},
                 {"mask": {"phrase_key": "mask.get_balance", "after_ms": 900}})
    store.update("policy", {"pk": "masking"}, {"end_mode": "bogus"}, remove=["silence_action"])
    assert run("show", store=store) == 0
    out = capsys.readouterr().out
    assert f"after_ms             {SEED_AFTER_MS:<10} (set in the table)" in out
    assert "silence_action       escalate   (default; not set in the table)" in out
    assert 'end_mode             wait       (default; the table\'s "bogus" is INVALID' in out
    assert "get_balance: phrase_key mask.get_balance, after_ms 900 (tool's own), " \
           "handoff_wait_ms 2500 (default: masking.handoff_wait_ms)" in out
    assert "list_recent_transactions: phrase_key mask.list_recent_transactions, " \
           f"after_ms {SEED_TOOL_AFTER_MS} (default: masking.tool_after_ms)" in out
    assert "faq_lookup: unmasked" in out


def test_show_one_tool_and_unknown_tool(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded()
    assert run("show", "--tool", "change_address", store=store) == 0
    out = capsys.readouterr().out
    assert "change_address: phrase_key" in out and "get_balance" not in out
    assert run("show", "--tool", "nope", store=store) == 2
    assert "no tool 'nope'" in capsys.readouterr().err


def test_show_needs_a_policy_table_name(capsys: pytest.CaptureFixture[str]) -> None:
    assert masking.main(["show", "--stage", "dev"], store=seeded()) == 2
    assert "no table name" in capsys.readouterr().err


# --------------------------------------------------------------------------- set a tool


def test_dry_run_prints_the_change_and_writes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded()
    before = tool(store, "get_balance")
    assert run("set", "--tool", "get_balance", "--after-ms", "900", store=store) == 0
    assert tool(store, "get_balance") == before
    out = capsys.readouterr().out
    assert "DRY RUN" in out and 'tools.get_balance.mask.after_ms: "(not set)" -> 900' in out


def test_set_tool_changes_only_the_mask(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded()
    before = tool(store, "get_balance")
    assert run("set", "--tool", "get_balance", "--after-ms", "900", "--handoff-wait-ms", "1000",
               "--apply", store=store) == 0
    after = tool(store, "get_balance")
    assert after["mask"] == {"phrase_key": "mask.get_balance", "after_ms": 900, "handoff_wait_ms": 1000}
    assert {k: v for k, v in after.items() if k != "mask"} == {k: v for k, v in before.items() if k != "mask"}
    assert "rev" not in after  # no rev added to an item that had none
    assert store.get("policy", {"pk": "masking"}) == seeded().get("policy", {"pk": "masking"})


def test_set_default_removes_the_tools_own_value() -> None:
    store = seeded()
    assert run("set", "--tool", "get_balance", "--after-ms", "900", "--apply", store=store) == 0
    assert run("set", "--tool", "get_balance", "--after-ms", "default", "--apply", store=store) == 0
    assert tool(store, "get_balance")["mask"] == {"phrase_key": "mask.get_balance"}


def test_phrase_key_off_unmasks(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded()
    assert run("set", "--tool", "change_address", "--phrase-key", "off", "--apply", store=store) == 0
    assert "mask" not in tool(store, "change_address")
    assert run("set", "--tool", "change_address", "--after-ms", "5", store=store) == 2
    assert "unmasked" in capsys.readouterr().err


def test_set_bumps_rev_when_the_item_has_one() -> None:
    store = seeded()
    store.update("policy", {"pk": "tool#get_balance"}, {"rev": 7})
    assert run("set", "--tool", "get_balance", "--after-ms", "300", "--apply", store=store) == 0
    assert tool(store, "get_balance")["rev"] == 8


@pytest.mark.parametrize("args, msg", [
    (("--after-ms", "20000"), "tools.get_balance.mask.after_ms: must be an integer from 0 to 10000"),
    (("--after-ms", "-1"), "tools.get_balance.mask.after_ms"),
    (("--after-ms", "fast"), "--after-ms: expected an integer"),
    (("--handoff-wait-ms", "99999"), "tools.get_balance.mask.handoff_wait_ms"),
    (("--phrase-key", "tool.get_balance"), "tools.get_balance.mask.phrase_key: must be a catalog key"),
    (("--phrase-key", "off", "--after-ms", "1"), "don't combine"),
    ((), "give --after-ms"),
])
def test_invalid_tool_values_are_refused(args: tuple[str, ...], msg: str,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded()
    before = tool(store, "get_balance")
    assert run("set", "--tool", "get_balance", *args, "--apply", store=store) == 2
    assert msg in capsys.readouterr().err
    assert tool(store, "get_balance") == before


def test_unknown_tool_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("set", "--tool", "nope", "--after-ms", "1", "--apply", store=seeded()) == 2
    assert "no tool 'nope'" in capsys.readouterr().err


def test_unknown_phrase_key_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded()
    assert run("set", "--tool", "get_balance", "--phrase-key", "mask.nothing_here", "--apply",
               store=store) == 2
    assert "no catalog phrase mask.nothing_here" in capsys.readouterr().err
    assert tool(store, "get_balance")["mask"]["phrase_key"] == "mask.get_balance"


def test_phrase_key_missing_a_language_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded()
    add_phrase(store, "mask.only_english", ("en-US",))
    assert run("set", "--tool", "get_balance", "--phrase-key", "mask.only_english", "--apply",
               store=store) == 2
    assert "catalog.mask.only_english[es-US]: missing" in capsys.readouterr().err


def test_phrase_with_a_bad_placeholder_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded()
    add_phrase(store, "mask.bad_placeholder", ("en-US", "es-US"), text="Checking {amount} now.")
    assert run("set", "--tool", "get_balance", "--phrase-key", "mask.bad_placeholder", "--apply",
               store=store) == 2
    assert "placeholder {amount}" in capsys.readouterr().err


def test_valid_new_phrase_key_in_every_language_is_written() -> None:
    store = seeded()
    add_phrase(store, "mask.new_phrase", ("en-US", "es-US"))
    assert run("set", "--tool", "get_balance", "--phrase-key", "mask.new_phrase", "--apply",
               store=store) == 0
    assert tool(store, "get_balance")["mask"]["phrase_key"] == "mask.new_phrase"


# --------------------------------------------------------------------------- set policy-wide values


def test_set_masking_fields_changes_only_those(capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded()
    before = store.get("policy", {"pk": "masking"})
    assert run("set", "--masking", "tool_after_ms=450", "--masking", "chars_per_s=15.5",
               "--masking", "silence_action=hangup", "--apply", store=store) == 0
    after = store.get("policy", {"pk": "masking"})
    assert after == {**before, "tool_after_ms": 450, "chars_per_s": "15.5", "silence_action": "hangup"}
    out = capsys.readouterr().out
    assert f"masking.tool_after_ms: {SEED_TOOL_AFTER_MS} -> 450" in out


def test_set_masking_default_removes_a_field() -> None:
    store = seeded()
    assert run("set", "--masking", "after_ms=default", "--apply", store=store) == 0
    assert "after_ms" not in store.get("policy", {"pk": "masking"})


def test_set_masking_creates_the_item_if_absent() -> None:
    store = seeded()
    store.delete("policy", {"pk": "masking"})
    assert run("set", "--masking", "after_ms=1000", "--apply", store=store) == 0
    assert store.get("policy", {"pk": "masking"}) == {"pk": "masking", "after_ms": 1000}


@pytest.mark.parametrize("pair, msg", [
    ("after_ms=10001", "masking.after_ms: must be an integer from 0 to 10000"),
    ("silence_max_repeats=9", "masking.silence_max_repeats"),
    ("end_mode=sometimes", "masking.end_mode: must be one of"),
    ("chars_per_s=99", "masking.chars_per_s"),
    ("chars_per_s=fast", "masking.chars_per_s: must be a number"),
    ("after_ms=soon", "masking.after_ms: must be an integer"),
    ("loudness=3", "masking.loudness: unknown field"),
])
def test_invalid_masking_values_are_refused(pair: str, msg: str, capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded()
    before = store.get("policy", {"pk": "masking"})
    assert run("set", "--masking", pair, "--apply", store=store) == 2
    assert msg in capsys.readouterr().err
    assert store.get("policy", {"pk": "masking"}) == before


def test_no_change_writes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    class NoWrite:
        def put_if_unchanged(self, *a: Any) -> None:
            raise AssertionError("written")

    store = seeded()
    assert run("set", "--tool", "get_balance", "--after-ms", "default", "--apply", store=store,
               writer=NoWrite()) == 0
    assert run("set", "--masking", f"after_ms={SEED_AFTER_MS}", "--apply", store=store, writer=NoWrite()) == 0
    assert "no change" in capsys.readouterr().out


# --------------------------------------------------------------------------- concurrency


class ConcurrentStore(MemoryStore):
    """Someone else edits ``pk`` right after this script first reads it."""

    def __init__(self, pk: str, change: dict[str, Any]) -> None:
        super().__init__()
        self.pk, self.change, self.armed = pk, change, False

    def get(self, table: str, key: Any) -> Any:
        item = super().get(table, key)
        if self.armed and table == "policy" and key == {"pk": self.pk}:
            self.armed = False
            super().update("policy", {"pk": self.pk}, self.change)
        return item


@pytest.mark.parametrize("rev", [None, 3])
def test_a_concurrent_change_is_not_clobbered(rev: int | None, capsys: pytest.CaptureFixture[str]) -> None:
    store = seeded(ConcurrentStore("tool#get_balance", {"mask": {"phrase_key": "mask.get_balance",
                                                                   "after_ms": 111}}))
    if rev is not None:
        store.update("policy", {"pk": "tool#get_balance"}, {"rev": rev})
        store.change["rev"] = rev + 1
    store.armed = True
    assert run("set", "--tool", "get_balance", "--after-ms", "900", "--apply", store=store) == 3
    assert "changed since it was read" in capsys.readouterr().err
    assert tool(store, "get_balance")["mask"]["after_ms"] == 111


def test_dynamo_writer_conditions_on_content_or_rev(dynamo: DynamoStore) -> None:  # noqa: F811
    import boto3

    seeded(dynamo)
    table = boto3.resource("dynamodb", region_name="us-east-1").Table("t-policy")
    writer = masking.DynamoPolicyWriter(table)
    read = dynamo.get("policy", {"pk": "tool#get_balance"})
    # someone else changes the mask: a write conditioned on the old content fails
    dynamo.update("policy", {"pk": "tool#get_balance"}, {"mask": {"phrase_key": "mask.get_balance",
                                                                  "after_ms": 111}})
    with pytest.raises(masking.Conflict):
        writer.put_if_unchanged({**read, "mask": {"phrase_key": "mask.get_balance", "after_ms": 900}}, read)
    assert dynamo.get("policy", {"pk": "tool#get_balance"})["mask"]["after_ms"] == 111
    # with the current content it succeeds
    now = dynamo.get("policy", {"pk": "tool#get_balance"})
    writer.put_if_unchanged({**now, "mask": {"phrase_key": "mask.get_balance", "after_ms": 900}}, now)
    assert dynamo.get("policy", {"pk": "tool#get_balance"})["mask"]["after_ms"] == 900
    # rev: only rev is compared, and it must match
    dynamo.update("policy", {"pk": "masking"}, {"rev": 2})
    m = dynamo.get("policy", {"pk": "masking"})
    writer.put_if_unchanged(masking.with_rev({**m, "after_ms": 1}, m), m)
    assert dynamo.get("policy", {"pk": "masking"})["rev"] == 3
    with pytest.raises(masking.Conflict):
        writer.put_if_unchanged(masking.with_rev({**m, "after_ms": 2}, m), m)  # stale
    # absent item: must still be absent
    with pytest.raises(masking.Conflict):
        writer.put_if_unchanged({"pk": "masking", "after_ms": 5}, None)


def test_main_uses_the_dynamo_writer_end_to_end(dynamo: DynamoStore) -> None:  # noqa: F811
    seeded(dynamo)
    assert masking.main(["set", "--stage", "dev", *TABLE_ARGS, "--region", "us-east-1",
                         "--tool", "get_balance", "--after-ms", "750", "--apply"]) == 0
    assert dynamo.get("policy", {"pk": "tool#get_balance"})["mask"]["after_ms"] == 750


# --------------------------------------------------------------------------- logging


def test_logs_start_steps_and_refusals(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    store = seeded()
    assert run("set", "--tool", "get_balance", "--after-ms", "900", "--apply", store=store) == 0
    assert run("set", "--masking", "after_ms=99999", store=store) == 2
    lines = [fields(r) for r in caplog.records if fields(r).get("script") == "masking"]
    assert lines[0]["event"] == "script_start" and lines[0]["command"] == "set"
    assert any(ln.get("step") == "write" and ln.get("outcome") == "changed" for ln in lines)
    err = next(ln for ln in lines if ln["event"] == "script_error")
    assert err["decision"] == "exit_2" and "masking.after_ms" in err["error_message"]
