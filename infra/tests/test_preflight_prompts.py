"""scripts/preflight.py's per-model prompt check: reading the
repository's files, the FAIL reasons (including too few runs and changed model parameters),
and which list and parameters ``run`` checks. Reads local files only, no AWS.

The FAIL-exactly-when-some-entry-and-language-lacks-a-current-record property and the parity of the copies
with ``owlbank.agent_prompt`` and ``owlbank.prompt_validation`` are root tests
(``tests/property/test_preflight_prompts_props.py``, ``tests/unit/test_failover_parity.py``): they
need ``src/``, which this project can't import."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"
spec = importlib.util.spec_from_file_location("preflight_for_prompts", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
sys.modules["preflight_for_prompts"] = preflight  # dataclasses need the module registered
spec.loader.exec_module(preflight)

ACCOUNT = "111111111111"
H55, H45, S55 = ("us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                 "us.anthropic.claude-sonnet-5-5")
LANGS = ("en-US", "es-US")
SUITE_DOC = {"version": 1, "runs": 5, "cases": [
    {"id": cid, "lang": lang, "history": [], "caller": "hello", "tool_results": {}, "expect": [{"check": "speaks"}]}
    for cid in ("C1", "C3") for lang in LANGS]}


def _params_hash(mk, **overrides):
    mid = f"us.anthropic.claude-{mk}"
    return preflight.params_hash(preflight.entry_params(mid, {mid: overrides} if overrides else None))


def _data(tmp_path: Path, validate=(), suite=True, stale_suite=(), stale_prompt=(), stale_params=(),
          short=()) -> Path:
    """A copy of the repository's catalog with records for ``validate`` (model key, lang) pairs, run
    with the entry's default parameters; ``stale_*`` pairs get a record with that hash wrong, ``short``
    pairs one with a case run fewer times than the case file's ``runs``."""
    d = tmp_path / "data"
    d.mkdir()
    shutil.copy(preflight.DATA_DIR / preflight.PROMPT_CATALOG_FILE, d)
    catalog = preflight.PromptCatalogCopy(json.loads((d / preflight.PROMPT_CATALOG_FILE).read_text("utf-8")))
    records = []
    for mk, lang in (*validate, *stale_suite, *stale_prompt, *stale_params, *short):
        records.append({"model_key": mk, "lang": lang, "model_id": f"us.anthropic.claude-{mk}",
                        "date": "2026-11-02",
                        "cases": {"C1": {"runs": 5, "passes": 5},
                                  "C3": {"runs": 1, "passes": 1} if (mk, lang) in short else {"runs": 5, "passes": 5}},
                        "prompt_hash": "0" * 64 if (mk, lang) in stale_prompt
                        else preflight.prompt_hash(catalog, mk, lang),
                        "suite_hash": "0" * 64 if (mk, lang) in stale_suite else preflight.suite_hash(SUITE_DOC),
                        "params_hash": "0" * 64 if (mk, lang) in stale_params else _params_hash(mk)})
    (d / preflight.PROMPT_VALIDATION_FILE).write_text(
        json.dumps({"version": 1, "header": {}, "records": records}), "utf-8")
    if suite:
        (d / preflight.PROMPT_SUITE_FILE).write_text(json.dumps(SUITE_DOC), "utf-8")
    return d


def _status(results):
    return {r.name: ("PASS" if r.ok else "FAIL") for r in results}


def test_a_current_record_in_every_language_passes(tmp_path):
    prompts = preflight.load_prompt_records(_data(tmp_path, validate=[("haiku-5-5", lang) for lang in LANGS]))
    results = preflight.check_prompts(prompts, [H55])
    assert _status(results) == {f"Model prompt {H55} ({lang})": "PASS" for lang in LANGS}
    params = preflight.entry_params(H55)
    assert prompts.has_prompt(H55, params) and prompts.has_prompt("global.anthropic.claude-haiku-5-5", params)


@pytest.mark.parametrize("kwargs, reason", [
    ({}, "no validation record for haiku-4-5 (es-US)"),
    ({"stale_suite": [("haiku-4-5", "es-US")]}, "cases are stale"),
    ({"stale_prompt": [("haiku-4-5", "es-US")]}, "prompt hash is stale"),
    ({"stale_params": [("haiku-4-5", "es-US")]}, "model parameters changed since validation"),
    ({"short": [("haiku-4-5", "es-US")]}, "not a full run"),
])
def test_each_missing_or_stale_record_fails_naming_the_entry_language_and_why(tmp_path, kwargs, reason):
    d = _data(tmp_path, validate=[("haiku-4-5", "en-US")], **kwargs)
    results = preflight.check_prompts(preflight.load_prompt_records(d), [H45])
    (fail,) = [r for r in results if not r.ok]
    assert fail.name == f"Model prompt {H45} (es-US)" and reason in fail.detail
    assert H45 in fail.fix and "es-US" in fail.fix and "scripts/prompt_suite.py" in fail.fix
    assert not preflight.load_prompt_records(d).has_prompt(H45, preflight.entry_params(H45))


@pytest.mark.parametrize("overrides", [
    {H55: {"max_tokens": 1500}},
    {H55: {"thinking": "enabled:2048"}},
    {H55: {"text_release": "message"}},
])
def test_changed_model_parameters_fail_with_their_reason(tmp_path, overrides):
    """A record validated with the default parameters is stale for an entry deployed with others."""
    prompts = preflight.load_prompt_records(_data(tmp_path, validate=[("haiku-5-5", lang) for lang in LANGS]))
    results = preflight.check_prompts(prompts, [H55], overrides)
    assert [r.ok for r in results] == [False, False]
    assert all("model parameters changed since validation" in r.detail for r in results)
    assert all("--models" in r.fix for r in results)


@pytest.mark.parametrize("caching", ["on", "off", True, False])
def test_prompt_caching_is_not_a_validated_parameter(tmp_path, caching):
    """Caching changes cost and latency, not the answer, so a record
    validated without it passes for an entry deployed with it on or off."""
    prompts = preflight.load_prompt_records(_data(tmp_path, validate=[("haiku-5-5", lang) for lang in LANGS]))
    assert all(r.ok for r in preflight.check_prompts(prompts, [H55], {H55: {"prompt_caching": caching}}))
    assert "prompt_caching" not in preflight.entry_params(H55, {H55: {"prompt_caching": caching}})


def test_a_record_validated_with_the_deployed_parameters_passes(tmp_path):
    overrides = {H55: {"max_tokens": 1500, "prompt_caching": True, "thinking": " Adaptive "}}
    d = _data(tmp_path)
    catalog = preflight.PromptCatalogCopy(json.loads((d / preflight.PROMPT_CATALOG_FILE).read_text("utf-8")))
    records = [{"model_key": "haiku-5-5", "lang": lang, "model_id": H55, "date": "2026-11-02",
                "cases": {"C1": {"runs": 7, "passes": 7}, "C3": {"runs": 5, "passes": 5}},
                "prompt_hash": preflight.prompt_hash(catalog, "haiku-5-5", lang),
                "suite_hash": preflight.suite_hash(SUITE_DOC),
                "params_hash": _params_hash("haiku-5-5", max_tokens=1500)} for lang in LANGS]
    (d / preflight.PROMPT_VALIDATION_FILE).write_text(json.dumps({"version": 1, "header": {}, "records": records}))
    prompts = preflight.load_prompt_records(d)
    assert all(r.ok for r in preflight.check_prompts(prompts, [H55], overrides))
    assert not any(r.ok for r in preflight.check_prompts(prompts, [H55]))


def test_entry_params_are_the_resolvers_defaults_with_overrides():
    assert preflight.entry_params(H55) == {"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000}
    assert preflight.entry_params(H45) == {"thinking": "off", "text_release": "message", "max_tokens": 1000}
    assert preflight.entry_params(H45, {H45: {"prompt_caching": "off", "max_tokens": 300},
                                        H55: {"max_tokens": 9}}) == {"thinking": "off", "text_release": "message",
                                                                     "max_tokens": 300}
    # synth's max_tokens follows the entry's thinking when not given (1000 with reasoning off)
    assert preflight.entry_params(H55, {H55: {"thinking": "off", "text_release": "message"}})["max_tokens"] == 1000


def test_a_model_without_a_catalog_text_fails(tmp_path):
    results = preflight.check_prompts(preflight.load_prompt_records(_data(tmp_path)), ["us.anthropic.claude-opus-4-1"])
    assert [r.ok for r in results] == [False, False]
    assert "no agent.system.opus-4-1 text in the catalog" in results[0].detail


def test_without_a_case_file_no_record_is_current(tmp_path):
    d = _data(tmp_path, validate=[("haiku-5-5", lang) for lang in LANGS], suite=False)
    results = preflight.check_prompts(preflight.load_prompt_records(d), [H55])
    assert [r.ok for r in results] == [False, False] and "prompt_suite.json" in results[0].detail


@pytest.mark.parametrize("damage", ["missing", "bad_json", "wrong_version", "no_records_key"])
def test_an_unreadable_file_is_one_fail_not_a_traceback(tmp_path, damage):
    d = _data(tmp_path)
    path = d / preflight.PROMPT_VALIDATION_FILE
    if damage == "missing":
        path.unlink()
    else:
        path.write_text({"bad_json": "{", "wrong_version": '{"version": 2, "records": []}',
                         "no_records_key": '{"version": 1}'}[damage], "utf-8")
    prompts = preflight.load_prompt_records(d)
    (result,) = preflight.check_prompts(prompts, [H55])
    assert not result.ok and "can't read the prompt files" in result.detail
    assert not prompts.has_prompt(H55, preflight.entry_params(H55))


def test_the_repositorys_files_are_read_by_default():
    prompts = preflight.load_prompt_records(preflight.DATA_DIR)
    assert prompts.error is None and prompts.languages == LANGS


def test_the_default_list_copy_is_infras():
    from owlbank_infra import config

    assert preflight.DEFAULT_MODEL_BASES == config.DEFAULT_MODEL_BASES
    assert preflight.default_list_ids("us-east-1") == list(config.default_model_ids("us-east-1"))
    assert preflight.default_list_ids("sa-east-1") == []


# --------------------------------------------------------------------------- run: which list


class _Sts:
    def get_caller_identity(self):
        return {"Account": ACCOUNT}


@pytest.fixture
def quiet(monkeypatch):
    """``run`` with the account checks stubbed out; returns the lists ``check_prompts`` was given."""
    monkeypatch.setattr(preflight, "check_s3_public_access_block", lambda *a: preflight.Result("bpa", True, ""))
    monkeypatch.setattr(preflight, "check_inference_profile", lambda *a: [])
    monkeypatch.setattr(preflight, "check_invocation_logging", lambda *a, **k: [])
    monkeypatch.setattr(preflight, "check_management_trail", lambda *a: preflight.Result("trail", True, ""))
    checked = []
    real = preflight.check_prompts
    monkeypatch.setattr(preflight, "check_prompts", lambda p, ids, *a: checked.append(list(ids)) or real(p, ids, *a))
    return checked


def _run(**kw):
    return preflight.run(sts=_Sts(), s3control=None, bedrock="bedrock", analyzer=None, stage="dev", **kw)


def test_run_checks_the_resolved_list_and_passes_has_prompt_to_the_resolver(quiet, monkeypatch, tmp_path):
    prompts = preflight.load_prompt_records(_data(tmp_path, validate=[("haiku-5-5", lang) for lang in LANGS]))
    seen = {}

    def resolve_list(bedrock, runtime, region, model_ids, params, geo=None, has_prompt=None):
        seen.update(model_ids=model_ids, has_prompt=has_prompt)
        return SimpleNamespace(findings=[], model_ids=[H55, S55])

    monkeypatch.setattr(preflight, "resolve_list", resolve_list)
    results = _run(bedrock_runtime="runtime", model_ids=[H55, H45], prompts=prompts)
    assert quiet == [[H55, S55]]
    assert seen["has_prompt"](H55) and not seen["has_prompt"](S55) and seen["model_ids"] == [H55, H45]
    assert {n: s for n, s in _status(results).items() if n.startswith("Model prompt")} == {
        f"Model prompt {H55} (en-US)": "PASS", f"Model prompt {H55} (es-US)": "PASS",
        f"Model prompt {S55} (en-US)": "FAIL", f"Model prompt {S55} (es-US)": "FAIL"}


def test_run_checks_each_entry_with_its_deployed_parameters(quiet, monkeypatch, tmp_path):
    """The resolver's ``has_prompt`` and ``check_prompts`` both see ``model_params``."""
    prompts = preflight.load_prompt_records(_data(tmp_path, validate=[("haiku-5-5", lang) for lang in LANGS]))
    seen = {}

    def resolve_list(bedrock, runtime, region, model_ids, params, geo=None, has_prompt=None):
        seen.update(has_prompt=has_prompt)
        return SimpleNamespace(findings=[], model_ids=[H55])

    monkeypatch.setattr(preflight, "resolve_list", resolve_list)
    results = _run(bedrock_runtime="runtime", model_ids=[H55], prompts=prompts)
    assert seen["has_prompt"](H55)
    assert [s for n, s in _status(results).items() if n.startswith("Model prompt")] == ["PASS", "PASS"]
    results = _run(bedrock_runtime="runtime", model_ids=[H55], prompts=prompts,
                   model_params={H55: {"max_tokens": 1999}})
    assert not seen["has_prompt"](H55)
    assert [s for n, s in _status(results).items() if n.startswith("Model prompt")] == ["FAIL", "FAIL"]


def test_run_skips_the_prompt_check_when_no_entry_resolved(quiet, monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "resolve_list",
                        lambda *a, **k: SimpleNamespace(findings=[], model_ids=[]))
    _run(bedrock_runtime="runtime", prompts=preflight.load_prompt_records(_data(tmp_path)))
    assert quiet == []


@pytest.mark.parametrize("model_ids, expected", [([H45], [H45]), (None, [H55, H45, S55])])
def test_without_resolution_run_checks_the_given_or_the_default_list(quiet, tmp_path, model_ids, expected):
    _run(model_ids=model_ids, prompts=preflight.load_prompt_records(_data(tmp_path)))
    assert quiet == [expected]


def test_run_without_prompts_runs_no_prompt_check(quiet):
    _run(model_ids=[H55])
    assert quiet == []


def test_cli_reads_the_prompt_files_from_data_dir(monkeypatch, tmp_path):
    d = _data(tmp_path)
    seen = {}

    class Session:
        def __init__(self, **kw):
            pass

        def client(self, name, **kw):
            return name

    monkeypatch.setitem(sys.modules, "boto3", type(sys)("boto3"))
    sys.modules["boto3"].Session = Session
    monkeypatch.setattr(preflight, "run", lambda **kw: seen.update(kw) or [])
    assert preflight.main(["--no-prefix-list", "--data-dir", str(d)]) == 0
    assert seen["prompts"].error is None and seen["prompts"].languages == LANGS
    assert "prompt_caching" not in seen
    with pytest.raises(SystemExit):  # --prompt-caching was removed with caching from the hash
        preflight.main(["--no-prefix-list", "--data-dir", str(d), "--prompt-caching", "on"])
    assert preflight.main(["--no-prefix-list", "--data-dir", str(tmp_path / "nowhere")]) == 0
    assert "can't read" in seen["prompts"].error


# --------------------------------------------------------------------------- agent prompt caching: catalog check

def _edited(tmp_path: Path, edit=None) -> Path:
    d = _data(tmp_path)
    if edit is not None:
        path = d / preflight.PROMPT_CATALOG_FILE
        doc = json.loads(path.read_text("utf-8"))
        edit(doc)
        path.write_text(json.dumps(doc, ensure_ascii=False), "utf-8")
    return d


def _set_text(key, lang, change):
    def edit(doc):
        (e,) = [e for e in doc["entries"] if (e["key"], e["lang"]) == (key, lang)]
        e["text"] = change(e["text"])
    return edit


def _drop(key, lang):
    def edit(doc):
        doc["entries"] = [e for e in doc["entries"] if (e["key"], e["lang"]) != (key, lang)]
    return edit


def test_the_repositorys_catalog_passes_the_placeholder_check():
    (r,) = preflight.check_agent_texts(preflight.load_prompt_records(preflight.DATA_DIR))
    assert (r.name, r.ok, r.warn) == ("Agent prompt texts", True, False)


@pytest.mark.parametrize("edit, named, why", [
    (_set_text("agent.system.haiku-5-5", "es-US", lambda t: t + " Caller: {firstName}."),
     "agent.system.haiku-5-5 (es-US)", "must have no placeholders"),
    (_set_text("agent.caller_context", "en-US", lambda t: t.replace("{profile}", "the profile")),
     "agent.caller_context (en-US)", "must have exactly {firstName}, {memory}, {profile}"),
    (_set_text("agent.web_activity", "en-US", lambda t: t + " }"),
     "agent.web_activity (en-US)", "has a '{' or '}' that isn't a {placeholder}"),
    (_drop("agent.caller_context_unverified", "es-US"), "agent.caller_context_unverified (es-US)", "missing"),
])
def test_a_text_breaking_or_a_missing_frame_fails_naming_key_and_language(tmp_path, edit, named, why):
    (r,) = preflight.check_agent_texts(preflight.load_prompt_records(_edited(tmp_path, edit)))
    assert r.name == "Agent prompt texts" and not r.ok and not r.warn
    assert f"{named}: {why}" in r.detail and "reseed" in r.fix


def test_a_new_agent_key_with_a_placeholder_fails(tmp_path):
    def edit(doc):
        doc["entries"].append({"key": "agent.system.opus-5-5", "lang": "en-US", "text": "Hi {firstName}.",
                               "version": 1})
    (r,) = preflight.check_agent_texts(preflight.load_prompt_records(_edited(tmp_path, edit)))
    assert not r.ok and "agent.system.opus-5-5 (en-US): must have no placeholders" in r.detail


def test_unreadable_prompt_files_fail_the_placeholder_check(tmp_path):
    (r,) = preflight.check_agent_texts(preflight.load_prompt_records(tmp_path / "nowhere"))
    assert not r.ok and "can't read" in r.detail


def test_run_adds_the_placeholder_check_with_prompts_even_when_no_entry_resolved(quiet, monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "resolve_list", lambda *a, **k: SimpleNamespace(findings=[], model_ids=[]))
    d = _edited(tmp_path, _set_text("agent.system.sonnet-5-5", "en-US", lambda t: t + " {memory}"))
    results = _run(bedrock_runtime="runtime", prompts=preflight.load_prompt_records(d))
    assert _status(results)["Agent prompt texts"] == "FAIL"
    assert "Agent prompt texts" not in _status(_run(model_ids=[H55]))
