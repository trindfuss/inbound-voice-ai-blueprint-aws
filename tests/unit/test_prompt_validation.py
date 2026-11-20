"""The prompt suite's checks, case file and validation records.

* Each structural check on synthetic outcomes, passing and failing (pure functions of
  an outcome). The response, ``<silent/>`` after an inline ``list_recent_transactions`` result,
  fails on ``speaks`` and only on it.
* The case-file schema (``data/prompt_suite.json`` is its own file) and ``suite_hash``.
* The record file: round trip, deterministic serialization, ``add`` replacing a model and
  language's record, ``is_validated`` against a fixture catalog (not the seed file, so neither a
  prompt edit nor a key rename touches these tests), full runs and ``params_hash``. The
  PBT of ``is_validated`` is in ``tests/property/test_prompt_validation_props.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from owlbank.agent_prompt import DEFAULT_MODEL_KEY, prompt_hash, system_key
from owlbank.agent_prompt import model_key as series_key
from owlbank.catalog import PromptCatalog
from owlbank.prompt_validation import (
    CHECK_KINDS,
    Case,
    CaseTally,
    Check,
    Outcome,
    Record,
    SchemaError,
    Suite,
    ToolCall,
    ValidationFile,
    failed,
    load_records,
    params_hash,
    parse_records,
    parse_suite,
    passes,
    save_records,
    suite_hash,
)

ROOT = Path(__file__).resolve().parents[2]
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_P = "d" * 64

SILENT = Check("silent")
SPEAKS = Check("speaks")
NO_TOOL = Check("no_tool")
NO_TEXT_BEFORE_TOOL = Check("no_text_before_tool")
NO_MARKDOWN = Check("no_markdown")
NO_MARKUP = Check("no_markup")


def tool(name: str, **args: Any) -> Check:
    return Check("tool", name=name, args=args or None)


def outcome(text: str = "", *, silent: bool = False, calls: tuple[ToolCall, ...] = (),
            text_before_tool: bool = False) -> Outcome:
    return Outcome(text=text, silent=silent, tool_calls=calls, text_before_tool=text_before_tool)


BALANCE = ToolCall("get_balance", {"accountType": "checking"})
TRANSACTIONS = ToolCall("list_recent_transactions", {"accountType": "checking", "limit": 5})


# --- the checks --------------------------------------------------------------------

def test_silent() -> None:
    assert passes(SILENT, outcome(silent=True))
    assert not passes(SILENT, outcome("Your balance is 1,200 dollars."))
    assert not passes(SILENT, outcome("Words", silent=True))  # text released before the marker
    assert passes(SILENT, outcome("  \n", silent=True))


def test_speaks() -> None:
    assert passes(SPEAKS, outcome("Here are your last five transactions."))
    assert not passes(SPEAKS, outcome(""))
    assert not passes(SPEAKS, outcome(" \n\t"))
    assert not passes(SPEAKS, outcome(silent=True))


def test_tool_by_name() -> None:
    assert passes(tool("get_balance"), outcome(calls=(BALANCE,)))
    assert passes(tool("list_recent_transactions"), outcome(calls=(BALANCE, TRANSACTIONS)))
    assert not passes(tool("transfer_funds"), outcome(calls=(BALANCE,)))
    assert not passes(tool("get_balance"), outcome("Your balance is 1,200 dollars."))


def test_tool_with_an_argument_subset() -> None:
    o = outcome(calls=(TRANSACTIONS,))
    assert passes(tool("list_recent_transactions", accountType="checking"), o)
    assert passes(tool("list_recent_transactions", accountType="checking", limit=5), o)
    assert not passes(tool("list_recent_transactions", accountType="savings"), o)
    assert not passes(tool("list_recent_transactions", fromDate="2026-10-01"), o)  # key absent
    assert not passes(tool("get_balance", accountType="checking"), o)  # args of another tool


def test_tool_matches_any_call_with_that_name() -> None:
    o = outcome(calls=(ToolCall("get_balance", {"accountType": "savings"}), BALANCE))
    assert passes(tool("get_balance", accountType="checking"), o)


def test_no_tool() -> None:
    assert passes(NO_TOOL, outcome("The pending amount is 40 dollars."))
    assert passes(NO_TOOL, outcome(silent=True))
    assert not passes(NO_TOOL, outcome(calls=(BALANCE,)))


def test_no_tool_with_a_name_fails_only_on_that_tool() -> None:
    no_transfer = Check("no_tool", "transfer_funds")
    assert passes(no_transfer, outcome("Nothing called."))
    assert passes(no_transfer, outcome(calls=(BALANCE,)))
    assert not passes(no_transfer, outcome(calls=(BALANCE, ToolCall("transfer_funds", {"amount": 500}))))


def test_no_text_before_tool() -> None:
    assert passes(NO_TEXT_BEFORE_TOOL, outcome("1,200 dollars.", calls=(BALANCE,)))
    assert not passes(NO_TEXT_BEFORE_TOOL, outcome("Let me check.", calls=(BALANCE,), text_before_tool=True))
    assert passes(NO_TEXT_BEFORE_TOOL, outcome("No tool at all."))


@pytest.mark.parametrize("text", [
    "Here they are:\n- coffee, 4 dollars",
    "Here they are:\n* coffee, 4 dollars",
    "# Transactions",
    "1. coffee, 4 dollars",
    "Here they are:\n  1. coffee",          # indented list item
    "You spent **40 dollars** on coffee.",
    "Your code is `1234`.",
    "  - indented bullet",
    "1.",
])
def test_no_markdown_fails(text: str) -> None:
    assert not passes(NO_MARKDOWN, outcome(text))


@pytest.mark.parametrize("text", [
    "",
    "Your last three transactions were coffee for 4 dollars, groceries for 62 dollars, and gas for 40 dollars.",
    "1.50 dollars went to the parking meter.",  # a number, not a list item
    "The rate is 4.5 percent - well, about that.",  # a dash inside a line
    "Two stars * between words, and one # sign.",
    "10. That was the tenth one.",  # only "1." starts a numbered list
])
def test_no_markdown_passes(text: str) -> None:
    assert passes(NO_MARKDOWN, outcome(text))


@pytest.mark.parametrize("text", [
    "<thinking>The caller asks.</thinking> Seventy-five dollars.",
    "<message>Your balance is 5 dollars.</message>",
    "Done.<br/>",
    '<answer lang="en">Yes.',
    "Yes.</response>",
    "<silent/>",  # the outcome's text is after the silent filter: a marker left is markup too
])
def test_no_markup_fails(text: str) -> None:
    assert not passes(NO_MARKUP, outcome(text))


@pytest.mark.parametrize("text", [
    "",
    "Seventy-five dollars.",
    "Less than <5 dollars, and 3 > 2.",
    "a < b and c > d",  # no letter right after "<"
    "<>",
])
def test_no_markup_passes(text: str) -> None:
    assert passes(NO_MARKUP, outcome(text))


def test_outcome_reasoning_removed_is_informational() -> None:
    o = Outcome(text="Yes.", silent=False, reasoning_removed=2)
    assert o.reasoning_removed == 2 and Outcome(text="", silent=False).reasoning_removed == 0
    assert failed((SPEAKS, NO_MARKUP), o) == []
    assert "no_markup" in CHECK_KINDS and Check.from_json({"check": "no_markup"}).to_json() == {"check": "no_markup"}


C1_EXPECT = (tool("list_recent_transactions"), SPEAKS, NO_MARKDOWN)


def test_response_fails_c1_on_speaks_only() -> None:
    """A model that called ``list_recent_transactions``, got the inline result and answered
    ``<silent/>``: after the filter, no text and the silent flag."""
    t13 = outcome(silent=True, calls=(TRANSACTIONS,))
    assert failed(C1_EXPECT, t13) == [SPEAKS]


def test_a_good_c1_response_passes() -> None:
    good = outcome("Your last two were coffee for 4 dollars and groceries for 62 dollars.", calls=(TRANSACTIONS,))
    assert failed(C1_EXPECT, good) == []


def test_checks_are_pure() -> None:
    o = outcome("- one", calls=(TRANSACTIONS,), text_before_tool=True)
    first = [passes(c, o) for c in (*C1_EXPECT, SILENT, NO_TOOL, NO_TEXT_BEFORE_TOOL)]
    second = [passes(c, o) for c in (*C1_EXPECT, SILENT, NO_TOOL, NO_TEXT_BEFORE_TOOL)]
    assert first == second
    assert o == outcome("- one", calls=(TRANSACTIONS,), text_before_tool=True)


def test_outcome_rejects_text_before_a_tool_with_no_tool() -> None:
    with pytest.raises(ValueError, match="text_before_tool"):
        outcome("Let me check.", text_before_tool=True)


def test_outcome_tool_args_are_frozen() -> None:
    args = {"accountType": "checking"}
    call = ToolCall("get_balance", args)
    args["accountType"] = "savings"
    assert call.args == {"accountType": "checking"}
    with pytest.raises(TypeError):
        call.args["x"] = 1  # type: ignore[index]


# --- check schema ---------------------------------------------------------------------------------

@pytest.mark.parametrize("doc", [
    {"check": "silent"}, {"check": "speaks"}, {"check": "no_tool"}, {"check": "no_text_before_tool"},
    {"check": "no_markdown"}, {"check": "tool", "name": "get_balance"},
    {"check": "tool", "name": "transfer_funds", "args": {"amount": 50, "toAccount": "savings"}},
    {"check": "no_tool", "name": "transfer_funds"},
])
def test_check_round_trip(doc: dict[str, Any]) -> None:
    assert Check.from_json(doc).to_json() == doc


@pytest.mark.parametrize("doc", [
    {"check": "language"},                                  # no language check
    {"check": "says", "text": "balance"},                   # no wording check
    {"check": "tool"},                                      # tool needs a name
    {"check": "tool", "name": ""},
    {"check": "tool", "name": "get_balance", "args": []},
    {"check": "speaks", "name": "get_balance"},             # only tool and no_tool take a name
    {"check": "no_tool", "name": ""},
    {"check": "no_tool", "name": "transfer_funds", "args": {}},  # no_tool takes no args
    {"check": "silent", "args": {}},
    {"check": "tool", "name": "get_balance", "extra": 1},
    "speaks",
])
def test_bad_checks_are_refused(doc: Any) -> None:
    with pytest.raises(SchemaError):
        Check.from_json(doc)


# --- the case file and suite_hash -------------------------------------------------------------------

def suite_doc() -> dict[str, Any]:
    history = [
        {"role": "user", "content": [{"text": "what's my balance?"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "get_balance",
                                                       "input": {"accountType": "checking"}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [
            {"json": {"spoken": "Your checking balance is 1,200 dollars.", "next": "await_caller"}}]}}]},
    ]
    return {
        "version": 1,
        "runs": 5,
        "cases": [
            {"id": "C1", "lang": "en-US", "history": history, "caller": "recent transactions",
             "tool_results": {"list_recent_transactions": {"transactions": []}},
             "expect": [{"check": "tool", "name": "list_recent_transactions"}, {"check": "speaks"},
                        {"check": "no_markdown"}]},
            {"id": "C2", "lang": "es-US", "history": history, "caller": None, "tool_results": {},
             "expect": [{"check": "silent"}]},
        ],
    }


def test_parse_suite() -> None:
    suite = parse_suite(suite_doc())
    assert suite.runs == 5
    assert [(c.id, c.lang) for c in suite.cases] == [("C1", "en-US"), ("C2", "es-US")]
    c1 = suite.cases[0]
    assert isinstance(c1, Case)
    assert c1.caller == "recent transactions"
    assert c1.expect == C1_EXPECT
    assert suite.cases[1].caller is None
    assert suite.hash == suite_hash(suite_doc())
    assert suite.for_lang("es-US") == (suite.cases[1],)


def _broken(edit: Any) -> dict[str, Any]:
    doc = suite_doc()
    edit(doc)
    return doc


@pytest.mark.parametrize("doc", [
    _broken(lambda d: d.update(version=2)),
    _broken(lambda d: d.update(runs=0)),
    _broken(lambda d: d.update(runs=True)),
    _broken(lambda d: d.update(cases=[])),
    _broken(lambda d: d.update(extra=1)),
    _broken(lambda d: d["cases"].append(dict(d["cases"][0]))),           # duplicate (id, lang)
    _broken(lambda d: d["cases"][0].update(expect=[])),
    _broken(lambda d: d["cases"][0].update(lang="")),
    _broken(lambda d: d["cases"][0].update(id="")),
    _broken(lambda d: d["cases"][0].update(caller=3)),
    _broken(lambda d: d["cases"][0].update(history={})),
    _broken(lambda d: d["cases"][0].update(history=[{"role": "system", "content": []}])),
    _broken(lambda d: d["cases"][0].update(history=[{"role": "user"}])),
    _broken(lambda d: d["cases"][0].update(tool_results=[])),
    _broken(lambda d: d["cases"][0].pop("expect")),
    _broken(lambda d: d["cases"][0].update(expect=[{"check": "language"}])),
])
def test_bad_suites_are_refused(doc: dict[str, Any]) -> None:
    with pytest.raises(SchemaError):
        parse_suite(doc)


def _with_context(context: Any) -> dict[str, Any]:
    return _broken(lambda d: d["cases"][0].update(context=context))


@pytest.mark.parametrize("context", [
    {"verified": False},
    {"verified": True},
    {"memory": "Asked about a card."},
    {"web_block": "offer"},
    {"web_block": "context", "memory": "m"},
    {"verified": True, "memory": "m", "web_block": "offer"},
    {},
])
def test_a_case_context_is_parsed(context: dict[str, Any]) -> None:
    case = parse_suite(_with_context(context)).cases[0]
    assert dict(case.context or {}) == context
    assert parse_suite(suite_doc()).cases[0].context is None


@pytest.mark.parametrize("context", [
    "verified", [], None, {"verified": "no"}, {"verified": 0}, {"memory": ""}, {"memory": 3},
    {"web_block": "agent.web_activity"}, {"web_block": None}, {"extra": 1},
    {"verified": False, "memory": "m"}, {"verified": False, "web_block": "offer"},
])
def test_a_bad_case_context_is_refused(context: Any) -> None:
    with pytest.raises(SchemaError):
        parse_suite(_with_context(context))


def test_suite_hash_ignores_formatting_and_key_order() -> None:
    doc = suite_doc()
    reordered = json.loads(json.dumps(doc, indent=4, sort_keys=True))
    reordered["cases"][0] = dict(reversed(list(reordered["cases"][0].items())))
    assert suite_hash(reordered) == suite_hash(doc)
    assert len(suite_hash(doc)) == 64


@pytest.mark.parametrize("edit", [
    lambda d: d.update(runs=3),
    lambda d: d["cases"][0].update(caller="recent transactions please"),
    lambda d: d["cases"][0]["expect"].append({"check": "no_text_before_tool"}),
    lambda d: d["cases"].reverse(),                                  # case order is part of the suite
    lambda d: d["cases"][1]["history"][0]["content"][0].update(text="¿cuál es mi saldo?"),
])
def test_suite_hash_changes_with_any_case_edit(edit: Any) -> None:
    doc = suite_doc()
    before = suite_hash(doc)
    edited = json.loads(json.dumps(doc))
    edit(edited)
    assert suite_hash(edited) != before


# --- the records ---------------------------------------------------------------------------

def record(model_key: str = "haiku-5-5", lang: str = "en-US", *, ph: str = HASH_A, sh: str = HASH_B,
           pa: str = HASH_P, cases: dict[str, CaseTally] | None = None) -> Record:
    return Record(model_key=model_key, lang=lang, prompt_hash=ph, suite_hash=sh, params_hash=pa,
                  model_id="us.anthropic.claude-haiku-5-5-v1:0", date="2026-11-02",
                  cases=cases or {"C1": CaseTally(5, 5), "C2": CaseTally(5, 5)})


def test_the_committed_records_are_canonical_and_current() -> None:
    """The committed file is canonical, and every default model has a current record in both
    languages for the committed catalog and case file (a prompt or case edit needs a new run)."""
    path = ROOT / "data" / "prompt_validation.json"
    loaded = load_records(path)
    assert path.read_text(encoding="utf-8") == loaded.dumps()
    catalog = PromptCatalog.from_dict(json.loads((ROOT / "data" / "prompt_catalog.json").read_text()))
    suite = parse_suite(json.loads((ROOT / "data" / "prompt_suite.json").read_text()))
    defaults = _default_entry_params_hashes()
    assert set(defaults) == {"haiku-5-5", "haiku-4-5", "sonnet-5-5"}
    for model_key, want in defaults.items():
        for lang in ("en-US", "es-US"):
            record = loaded.record_for(model_key, lang)
            assert record is not None, (model_key, lang)
            assert record.params_hash == want, (model_key, lang)
            assert loaded.is_validated(catalog, model_key, lang, suite, want), (model_key, lang)


def _default_entry_params_hashes() -> dict[str, str]:
    """``params_hash`` of each DEFAULT list entry's parameters, built as ``test_failover_parity``
    builds them: synth's ``OWLBANK_MODELS`` entries (``infra/owlbank_infra/config.py``, loaded by
    file path), read by the runner (``parse_models``) and hashed with its ``entry_params``. With
    ``promptCaching`` off and on alike (caching is not a validated parameter)."""
    from scripts import prompt_suite as ps

    spec = importlib.util.spec_from_file_location("_infra_config_for_prompt_validation",
                                                  ROOT / "infra" / "owlbank_infra" / "config.py")
    assert spec is not None and spec.loader is not None
    config = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = config  # dataclasses look the module up while building the class
    try:
        spec.loader.exec_module(config)
        by_caching = []
        for caching in (False, True):
            entries = config._models({}, region="us-east-1", prompt_caching=caching)  # noqa: SLF001
            env = ps.parse_models(json.dumps([e.as_env() for e in entries]))
            by_caching.append({series_key(e.id): params_hash(ps.entry_params(e)) for e in env})
    finally:
        sys.modules.pop(spec.name, None)
    assert by_caching[0] == by_caching[1]
    return by_caching[0]


def test_record_round_trip(tmp_path: Path) -> None:
    vf = ValidationFile.empty().add(record()).add(record("sonnet-5-5", "es-US"))
    vf = ValidationFile(header={"measured": {"inputTokens": 2400000, "note": "first run"}}, records=vf.records)
    assert parse_records(json.loads(vf.dumps())) == vf
    path = tmp_path / "prompt_validation.json"
    save_records(path, vf)
    assert load_records(path) == vf
    assert path.read_text(encoding="utf-8") == vf.dumps()


def test_serialization_is_deterministic() -> None:
    a, b, c = record("haiku-5-5", "es-US"), record("haiku-4-5", "en-US"), record("haiku-5-5", "en-US")
    one = ValidationFile.empty().add(a).add(b).add(c)
    other = ValidationFile.empty().add(c).add(a).add(b)
    assert one.dumps() == other.dumps()
    assert [(r.model_key, r.lang) for r in one.records] == [
        ("haiku-4-5", "en-US"), ("haiku-5-5", "en-US"), ("haiku-5-5", "es-US")]
    text = one.dumps()
    assert text.endswith("}\n") and not text.endswith("\n\n")
    assert json.loads(text)["records"][0]["cases"] == {"C1": {"passes": 5, "runs": 5},
                                                       "C2": {"passes": 5, "runs": 5}}
    # cases are written sorted whatever order they came in
    shuffled = record(cases={"C2": CaseTally(5, 5), "C10": CaseTally(5, 5), "C1": CaseTally(5, 5)})
    assert list(json.loads(ValidationFile.empty().add(shuffled).dumps())["records"][0]["cases"]) == [
        "C1", "C10", "C2"]


def test_add_replaces_the_record_of_a_model_and_language() -> None:
    old = record(ph=HASH_A)
    new = record(ph="c" * 64)
    vf = ValidationFile.empty().add(old).add(record(lang="es-US")).add(new)
    assert vf.record_for("haiku-5-5", "en-US") == new
    assert len(vf.records) == 2


@pytest.mark.parametrize("kwargs", [
    {"prompt_hash": "A" * 64}, {"prompt_hash": "a" * 63}, {"suite_hash": "xyz"}, {"params_hash": ""},
    {"model_key": ""}, {"lang": ""}, {"model_id": ""}, {"date": "02/11/2026"}, {"date": "2026-13-01"},
    {"cases": {}},
    {"cases": {"C1": CaseTally(5, 4)}},          # a record is a full passing run
    {"cases": {"C1": CaseTally(0, 0)}},
    {"cases": {"": CaseTally(5, 5)}},
])
def test_bad_records_are_refused(kwargs: dict[str, Any]) -> None:
    base = {"model_key": "haiku-5-5", "lang": "en-US", "prompt_hash": HASH_A, "suite_hash": HASH_B,
            "params_hash": HASH_P, "model_id": "us.anthropic.claude-haiku-5-5-v1:0", "date": "2026-11-02",
            "cases": {"C1": CaseTally(5, 5)}}
    with pytest.raises(SchemaError):
        Record(**{**base, **kwargs})


@pytest.mark.parametrize("doc", [
    {"version": 1, "header": {}},
    {"version": 2, "header": {}, "records": []},
    {"version": 1, "header": [], "records": []},
    {"version": 1, "header": {}, "records": {}},
    {"version": 1, "header": {}, "records": [{"model_key": "haiku-5-5"}]},
    {"version": 1, "header": {}, "records": [], "extra": 1},
    {"version": 1, "header": {}, "records": [record().to_json(), record().to_json()]},  # duplicate key
    {"version": 1, "header": {}, "records": [{k: v for k, v in record().to_json().items() if k != "params_hash"}]},
])
def test_bad_record_files_are_refused(doc: dict[str, Any]) -> None:
    with pytest.raises(SchemaError):
        parse_records(doc)


def fixture_catalog(model_text: str = "You are OwlBank's agent. Caller: {firstName}.") -> PromptCatalog:
    key = system_key(DEFAULT_MODEL_KEY)
    return PromptCatalog.from_dict({
        "defaultLanguage": "en-US", "languages": {"en-US": {}, "es-US": {}},
        "entries": [
            {"key": key, "lang": "en-US", "text": model_text},
            {"key": key, "lang": "es-US", "text": "Usted es el agente de OwlBank."},
            {"key": "agent.web_activity", "lang": "en-US", "text": "Web activity: asked."},
        ],
    })


def suite_of(runs: int = 5, sh: str = HASH_B, ids: tuple[str, ...] = ("C1", "C2"), lang: str = "en-US") -> Suite:
    cases = tuple(Case(id=cid, lang=lang, history=(), caller="hi", tool_results={}, expect=(SPEAKS,)) for cid in ids)
    return Suite(runs=runs, cases=cases, hash=sh)


def test_is_validated_against_the_current_catalog() -> None:
    catalog = fixture_catalog()
    current = prompt_hash(catalog, DEFAULT_MODEL_KEY, "en-US")
    vf = ValidationFile.empty().add(record(DEFAULT_MODEL_KEY, "en-US", ph=current, sh=HASH_B))
    assert vf.is_validated(catalog, DEFAULT_MODEL_KEY, "en-US", suite_of())
    assert not vf.is_validated(catalog, DEFAULT_MODEL_KEY, "en-US", suite_of(sh="c" * 64))    # cases changed
    assert not vf.is_validated(catalog, DEFAULT_MODEL_KEY, "es-US", suite_of())               # other language
    assert not vf.is_validated(fixture_catalog("Edited prompt."), DEFAULT_MODEL_KEY, "en-US", suite_of())
    assert not ValidationFile.empty().is_validated(catalog, DEFAULT_MODEL_KEY, "en-US", suite_of())


def test_is_validated_needs_full_runs_of_every_case_of_the_language() -> None:
    """every case of the record's language ran at least the case file's ``runs``."""
    catalog = fixture_catalog()
    current = prompt_hash(catalog, DEFAULT_MODEL_KEY, "en-US")
    vf = ValidationFile.empty().add(record(DEFAULT_MODEL_KEY, ph=current, cases={"C1": CaseTally(5, 5),
                                                                                "C2": CaseTally(1, 1)}))
    assert not vf.is_validated(catalog, DEFAULT_MODEL_KEY, "en-US", suite_of(runs=5))
    assert vf.is_validated(catalog, DEFAULT_MODEL_KEY, "en-US", suite_of(runs=1))
    assert vf.is_validated(catalog, DEFAULT_MODEL_KEY, "en-US", suite_of(runs=5, ids=("C1",)))
    assert not vf.is_validated(catalog, DEFAULT_MODEL_KEY, "en-US", suite_of(runs=1, ids=("C1", "C3")))  # missing
    assert vf.is_validated(catalog, DEFAULT_MODEL_KEY, "en-US", suite_of(runs=5, ids=("C1", "C3"), lang="es-US"))


def test_is_validated_compares_the_params_hash_only_when_given() -> None:
    catalog = fixture_catalog()
    current = prompt_hash(catalog, DEFAULT_MODEL_KEY, "en-US")
    vf = ValidationFile.empty().add(record(DEFAULT_MODEL_KEY, ph=current, pa=HASH_P))
    assert vf.is_validated(catalog, DEFAULT_MODEL_KEY, "en-US", suite_of(), HASH_P)
    assert not vf.is_validated(catalog, DEFAULT_MODEL_KEY, "en-US", suite_of(), HASH_A)
    assert vf.is_validated(catalog, DEFAULT_MODEL_KEY, "en-US", suite_of())  # the seed: no list, no parameters


def test_params_hash_covers_the_three_parameters_only() -> None:
    """Thinking, text_release and max_tokens; not ``id`` nor ``prompt_caching``."""
    base = {"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000}
    h = params_hash(base)
    assert len(h) == 64 and h == params_hash(dict(reversed(list(base.items()))))
    assert params_hash({**base, "id": "us.anthropic.claude-haiku-5-5"}) == h
    assert params_hash({**base, "prompt_caching": True}) == params_hash({**base, "prompt_caching": False}) == h
    for k, v in (("thinking", "off"), ("text_release", "message"), ("max_tokens", 1999)):
        assert params_hash({**base, k: v}) != h, k
    for missing in base:
        with pytest.raises(KeyError):
            params_hash({k: v for k, v in base.items() if k != missing})


def test_is_validated_is_false_when_the_model_text_is_missing() -> None:
    empty = PromptCatalog.from_dict({"defaultLanguage": "en-US", "languages": {"en-US": {}},
                                     "entries": [{"key": "other.key", "lang": "en-US", "text": "x"}]})
    vf = ValidationFile.empty().add(record(DEFAULT_MODEL_KEY))
    assert not vf.is_validated(empty, DEFAULT_MODEL_KEY, "en-US", suite_of())
