"""scripts/seed.py's per-model prompt gate (model failover) through ``main``: the
repository's records, a copied data directory with records, the dev-only override and its log line.
The properties are in ``tests/property/test_seed_prompt_gate_props.py``."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pytest

from owlbank.agent_prompt import prompt_hash
from owlbank.catalog import PromptCatalog
from owlbank.config import TABLE_NAMES
from owlbank.prompt_validation import CaseTally, Record, ValidationFile, save_records, suite_hash
from owlbank.store import MemoryStore
from scripts import seed
from scripts._common import DATA_DIR
from tests.logutil import lines

TABLE_ARGS = [a for name in TABLE_NAMES for a in ("--table", f"{name}=t-{name}")]
SUITE_DOC = {"version": 1, "runs": 5, "cases": [
    {"id": "C1", "lang": lang, "history": [], "caller": "hello", "tool_results": {}, "expect": [{"check": "speaks"}]}
    for lang in ("en-US", "es-US")]}
TEXTS = [f"agent.system.{mk} ({lang})" for mk in ("haiku-5-5", "haiku-4-5", "sonnet-5-5")
         for lang in ("en-US", "es-US")]


def _catalog(d: Path) -> dict[str, Any]:
    return json.loads((d / "prompt_catalog.json").read_text(encoding="utf-8"))


def _data(tmp_path: Path, *, validate: tuple[str, ...] = (), suite: bool = True, runs: int = 5) -> Path:
    """A copy of the repository's data with records for the ``validate`` texts (``key (lang)``), each
    case run ``runs`` times."""
    d = tmp_path / "data"
    shutil.copytree(DATA_DIR, d)
    catalog = PromptCatalog.from_dict(_catalog(d))
    records = []
    for text in validate:
        key, lang = text.removesuffix(")").split(" (")
        mk = key.removeprefix("agent.system.")
        records.append(Record(model_key=mk, lang=lang, prompt_hash=prompt_hash(catalog, mk, lang),
                              suite_hash=suite_hash(SUITE_DOC), params_hash="6" * 64,
                              model_id=f"us.anthropic.claude-{mk}", date="2026-11-02",
                              cases={"C1": CaseTally(runs, runs)}))
    save_records(d / seed.PROMPT_VALIDATION_FILE, ValidationFile(header={}, records=tuple(records)))
    if suite:
        (d / seed.PROMPT_SUITE_FILE).write_text(json.dumps(SUITE_DOC), encoding="utf-8")
    else:
        (d / seed.PROMPT_SUITE_FILE).unlink(missing_ok=True)  # the repository's own case file
    return d


def test_the_repositorys_prompts_are_validated(capsys: pytest.CaptureFixture[str]) -> None:
    """Every committed agent.system.* text has a current record, so a prod seed passes the gate."""
    assert seed.main(["--stage", "prod", *TABLE_ARGS]) == 0
    assert "every agent.system.* text has a current validation record" in capsys.readouterr().out


def test_unvalidated_prompts_are_refused_naming_each(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = _data(tmp_path)
    store = MemoryStore()
    assert seed.main(["--stage", "prod", "--data-dir", str(d), *TABLE_ARGS, "--apply"], store=store) == 2
    err = capsys.readouterr().err
    assert all(t in err for t in TEXTS) and "--allow-unvalidated-prompt" in err
    assert store.get("catalog", {"key": "_meta", "lang": "_"}) is None  # nothing written


def test_every_text_validated_seeds_at_prod(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = _data(tmp_path, validate=tuple(TEXTS))
    store = MemoryStore()
    assert seed.main(["--stage", "prod", "--data-dir", str(d), *TABLE_ARGS, "--apply"], store=store) == 0
    assert "every agent.system.* text has a current validation record" in capsys.readouterr().out
    assert store.get("catalog", {"key": "agent.system.haiku-4-5", "lang": "es-US"}) is not None


def test_only_the_unvalidated_texts_are_named(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = _data(tmp_path, validate=tuple(t for t in TEXTS if "haiku-4-5" not in t))
    assert seed.main(["--stage", "test", "--data-dir", str(d)]) == 2
    err = capsys.readouterr().err
    assert "agent.system.haiku-4-5 (en-US), agent.system.haiku-4-5 (es-US)." in err
    assert "haiku-5-5" not in err and "sonnet-5-5" not in err


def test_a_changed_shared_block_makes_every_record_stale(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = _data(tmp_path, validate=tuple(TEXTS))
    doc = _catalog(d)
    for e in doc["entries"]:
        if e["key"] == "agent.web_activity" and e["lang"] == "es-US":
            e["text"] += " "
    (d / "prompt_catalog.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    assert seed.main(["--stage", "prod", "--data-dir", str(d)]) == 2
    err = capsys.readouterr().err
    assert all(t in err for t in TEXTS if "es-US" in t) and not any(t in err for t in TEXTS if "en-US" in t)


def test_a_record_of_fewer_runs_than_the_case_files_is_refused(tmp_path: Path,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    """A ``--runs 1`` record is no validation (every one of the case file's runs is needed)."""
    d = _data(tmp_path, validate=tuple(TEXTS), runs=1)
    assert seed.main(["--stage", "prod", "--data-dir", str(d)]) == 2
    err = capsys.readouterr().err
    assert all(t in err for t in TEXTS)


def test_an_unreadable_case_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = _data(tmp_path, validate=tuple(TEXTS))
    (d / seed.PROMPT_SUITE_FILE).write_text('{"version": 1, "cases": [{"id": "C1"}]}', encoding="utf-8")
    assert seed.main(["--stage", "dev", "--data-dir", str(d), "--allow-unvalidated-prompt"]) == 2
    assert "can't read prompt_suite.json" in capsys.readouterr().err


def test_without_a_case_file_no_record_is_current(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = _data(tmp_path, validate=tuple(TEXTS), suite=False)
    assert seed.main(["--stage", "prod", "--data-dir", str(d)]) == 2
    assert "no prompt_suite.json yet" in capsys.readouterr().err


def test_an_unreadable_records_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = _data(tmp_path)
    (d / seed.PROMPT_VALIDATION_FILE).write_text('{"version": 2}', encoding="utf-8")
    assert seed.main(["--stage", "dev", "--data-dir", str(d), "--allow-unvalidated-prompt"]) == 2
    assert "can't read prompt_validation.json" in capsys.readouterr().err


def test_draft_texts_are_not_gated(tmp_path: Path) -> None:
    d = _data(tmp_path, validate=tuple(TEXTS))
    doc = _catalog(d)
    doc["entries"].append({"key": "agent.system.opus-4-1", "lang": "en-US", "text": "draft", "version": 1,
                           "status": "draft"})
    (d / "prompt_catalog.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    assert seed.main(["--stage", "prod", "--data-dir", str(d)]) == 0


@pytest.mark.parametrize("stage", ["test", "prod", "DEV", "staging"])
def test_the_override_is_refused_outside_dev(stage: str, capsys: pytest.CaptureFixture[str]) -> None:
    store = MemoryStore()
    assert seed.main(["--stage", stage, *TABLE_ARGS, "--allow-unvalidated-prompt", "--apply"], store=store) == 2
    assert "only with --stage dev" in capsys.readouterr().err
    assert store.get("catalog", {"key": "_meta", "lang": "_"}) is None


def test_the_override_in_dev_lists_each_text_in_the_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = _data(tmp_path)
    assert seed.main(["--stage", "dev", "--data-dir", str(d), "--allow-unvalidated-prompt"]) == 0
    out = capsys.readouterr().out
    assert "WARNING: agent prompts seeded WITHOUT a current validation record" in out
    assert all(t in out for t in TEXTS)


def test_the_override_in_dev_writes_and_logs_the_texts_at_warning(tmp_path: Path,
                                                                  caplog: pytest.LogCaptureFixture) -> None:
    d = _data(tmp_path)
    store = MemoryStore()
    with caplog.at_level(logging.INFO, logger="owlbank.seed"):
        assert seed.main(["--stage", "dev", "--data-dir", str(d), *TABLE_ARGS, "--allow-unvalidated-prompt",
                          "--apply"], store=store) == 0
    assert store.get("catalog", {"key": "agent.system.sonnet-5-5", "lang": "en-US"}) is not None
    (line,) = [f for f in lines(caplog.records, "owlbank.seed") if f.get("step") == "agent_prompts"]
    assert line["outcome"] == "unvalidated" and line["texts"] == [
        f"agent.system.{mk} ({lang})" for lang in ("en-US", "es-US") for mk in ("haiku-5-5", "haiku-4-5", "sonnet-5-5")]
    assert [r.levelno for r in caplog.records if getattr(r, "step", None) == "agent_prompts"] == [logging.WARNING]


# --------------------------------------------------------------------------- agent prompt caching: catalog check

def _edit(d: Path, edit: Any) -> None:
    doc = _catalog(d)
    edit(doc)
    (d / "prompt_catalog.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def _set_text(key: str, lang: str, change: Any) -> Any:
    def edit(doc: dict[str, Any]) -> None:
        (e,) = [e for e in doc["entries"] if (e["key"], e["lang"]) == (key, lang)]
        e["text"] = change(e["text"])
    return edit


def _drop(key: str, lang: str) -> Any:
    def edit(doc: dict[str, Any]) -> None:
        doc["entries"] = [e for e in doc["entries"] if (e["key"], e["lang"]) != (key, lang)]
    return edit


def test_the_repositorys_catalog_passes_the_placeholder_check() -> None:
    """The shipped agent.* texts keep and both caller-context frames are in every language."""
    seed.check_agent_texts(json.loads((DATA_DIR / "prompt_catalog.json").read_text(encoding="utf-8")))
    assert seed.agent_text_problems(json.loads((DATA_DIR / "prompt_catalog.json").read_text(encoding="utf-8"))) == []


@pytest.mark.parametrize("edit, named, why", [
    (_set_text("agent.system.haiku-5-5", "es-US", lambda t: t + " Caller: {firstName}."),
     "agent.system.haiku-5-5 (es-US)", "must have no placeholders"),
    (_set_text("agent.caller_context", "en-US", lambda t: t.replace("{profile}", "the profile")),
     "agent.caller_context (en-US)", "must have exactly {firstName}, {memory}, {profile}"),
    (_set_text("agent.caller_context_unverified", "en-US", lambda t: t + " {memory}"),
     "agent.caller_context_unverified (en-US)", "must have no placeholders"),
    (_set_text("agent.web_activity", "en-US", lambda t: t + " {"),
     "agent.web_activity (en-US)", "has a '{' or '}' that isn't a {placeholder}"),
    (_drop("agent.caller_context_unverified", "es-US"), "agent.caller_context_unverified (es-US)", "missing"),
    (_drop("agent.caller_context", "en-US"), "agent.caller_context (en-US)", "missing"),
])
def test_a_text_breaking_a_rule_or_a_missing_frame_is_refused_with_no_override(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], edit: Any, named: str, why: str) -> None:
    """Refused naming the key and language, even with the dev-only gate override
    (the check has no escape), and nothing is written."""
    d = _data(tmp_path)
    _edit(d, edit)
    store = MemoryStore()
    assert seed.main(["--stage", "dev", "--data-dir", str(d), *TABLE_ARGS, "--allow-unvalidated-prompt",
                      "--apply"], store=store) == 2
    err = capsys.readouterr().err
    assert f"{named}: {why}" in err
    assert store.get("catalog", {"key": "_meta", "lang": "_"}) is None


def test_the_placeholder_check_runs_before_the_validation_gate(tmp_path: Path,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    """Unvalidated texts and a break: the refusal is the one reported."""
    d = _data(tmp_path)
    _edit(d, _set_text("agent.system.sonnet-5-5", "en-US", lambda t: t + " {memory}"))
    assert seed.main(["--stage", "prod", "--data-dir", str(d)]) == 2
    err = capsys.readouterr().err
    assert "agent.system.sonnet-5-5 (en-US): must have no placeholders" in err
    assert "without a current validation record" not in err


def test_every_problem_is_named_in_catalog_order(tmp_path: Path) -> None:
    d = _data(tmp_path)
    _edit(d, _set_text("agent.system.haiku-4-5", "en-US", lambda t: t + " {firstName}"))
    _edit(d, _drop("agent.caller_context_unverified", "es-US"))
    with pytest.raises(seed.ScriptError) as exc:
        seed.check_agent_texts(_catalog(d))
    assert str(exc.value).startswith("prompt_catalog.json: agent prompt text(s) break the placeholder rule")
    assert seed.agent_text_problems(_catalog(d)) == [
        "agent.system.haiku-4-5 (en-US): must have no placeholders",
        "agent.caller_context_unverified (es-US): missing"]


def test_draft_and_non_agent_texts_are_not_checked(tmp_path: Path) -> None:
    """Only published ``agent.*`` texts; a draft frame doesn't count as present."""
    d = _data(tmp_path, validate=tuple(TEXTS))

    def edit(doc: dict[str, Any]) -> None:
        doc["entries"] += [
            {"key": "agent.system.opus-4-1", "lang": "en-US", "text": "Hi {firstName}", "version": 1,
             "status": "draft"},
            {"key": "greeting.extra", "lang": "en-US", "text": "Hi {firstName}", "version": 1}]
    _edit(d, edit)
    assert seed.agent_text_problems(_catalog(d)) == []

    def draft_frame(doc: dict[str, Any]) -> None:
        for e in doc["entries"]:
            if (e["key"], e["lang"]) == ("agent.caller_context", "es-US"):
                e["status"] = "draft"
    _edit(d, draft_frame)
    assert seed.agent_text_problems(_catalog(d)) == ["agent.caller_context (es-US): missing"]
