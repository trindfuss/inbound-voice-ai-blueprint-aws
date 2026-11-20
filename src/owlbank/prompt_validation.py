"""Prompt suite checks, case file and validation records.

* :class:`Outcome` is what one model run of a case produced: the text spoken (after the
  ``<silent/>`` filter), whether the turn was silent, the tool calls in order and whether text came
  before the first tool call.
* :class:`Check` and :func:`passes` are the structural checks: ``silent``, ``speaks``,
  ``tool`` (name, optional argument subset), ``no_tool`` (optional name), ``no_text_before_tool``,
  ``no_markdown`` and ``no_markup`` (9.10: no ``<...>`` tag). Pure functions of an outcome. No check matches
  wording and there is no language check.
* :func:`parse_suite` reads the case file (``data/prompt_suite.json``, written) and
  :func:`suite_hash` hashes it.
* :class:`ValidationFile` is ``data/prompt_validation.json``: one :class:`Record` per model and
  language, written after a full passing run. :meth:`ValidationFile.is_validated` holds when the
  record carries the catalog's current :func:`owlbank.agent_prompt.prompt_hash`, the case file's
  hash and every case's full runs (:meth:`Record.full_runs`), and, when given, the entry's
  :func:`params_hash`. The file is committed and reviewed in PRs, so :meth:`ValidationFile.dumps` is
  deterministic: records sorted by model and language, keys sorted, two-space indent, one newline.

No I/O beyond :func:`load_records` and :func:`save_records`.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from owlbank.agent_prompt import prompt_hash
from owlbank.catalog import CatalogKeyError, PromptCatalog

SUITE_VERSION: Final = 1
RECORDS_VERSION: Final = 1
DEFAULT_RUNS: Final = 5
#: The entry parameters a run is validated with (:func:`params_hash`): the ``OWLBANK_MODELS``
#: entry fields but ``id`` and ``prompt_caching`` (caching changes cost and latency, not the answer).
PARAM_KEYS: Final = ("thinking", "text_release", "max_tokens")

_HEX64 = re.compile(r"[0-9a-f]{64}")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
#: A line (after leading whitespace) starting with ``-``, ``*``, ``#`` or ``1.``.
#: ``1.`` counts only as a list marker (followed by whitespace or the line's end), so "1.50 dollars"
#: is not markdown.
_MARKDOWN_LINE = re.compile(r"[-*#]|1\.(\s|$)")
#: ``no_markup``: an HTML/XML-like tag, of any length (stricter than the bridge's filter).
_MARKUP = re.compile(r"</?[A-Za-z][^<>]*>")
_HISTORY_ROLES: Final = frozenset({"user", "assistant"})


class SchemaError(ValueError):
    """A case file, check or record file that doesn't match its schema."""


def _canonical(doc: Any) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _freeze(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


def _need(cond: bool, what: str) -> None:
    if not cond:
        raise SchemaError(what)


def _keys(doc: Mapping[str, Any], required: set[str], optional: set[str], what: str) -> None:
    keys = set(doc)
    _need(required <= keys, f"{what}: missing {sorted(required - keys)}")
    _need(keys <= required | optional, f"{what}: unknown {sorted(keys - required - optional)}")


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


# --- outcome -----------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool call the model made: its name and arguments (a frozen copy)."""

    name: str
    args: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", _freeze(self.args))


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one model run of a case produced.

    ``text`` is what reaches the caller after the reasoning filter and the ``<silent/>``
    filter; ``silent`` is whether the filter found the marker; ``tool_calls`` are in call order;
    ``text_before_tool`` is whether any text was released before the first tool call (so it needs a
    tool call). ``stop_reason`` is the model's last stop reason, for the record and for diagnosing a
    failed run, and ``reasoning_removed`` the reasoning blocks the filter removed; neither is
    a check.
    """

    text: str
    silent: bool
    tool_calls: tuple[ToolCall, ...] = ()
    text_before_tool: bool = False
    stop_reason: str | None = None
    reasoning_removed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if self.text_before_tool and not self.tool_calls:
            raise ValueError("text_before_tool needs a tool call")


# --- checks -----------------------------------------------------------------------------

def _silent(_: Check, o: Outcome) -> bool:
    return o.silent and not o.text.strip()


def _speaks(_: Check, o: Outcome) -> bool:
    return bool(o.text.strip())


def _tool(c: Check, o: Outcome) -> bool:
    want = c.args or {}
    return any(call.name == c.name and all(k in call.args and call.args[k] == v for k, v in want.items())
               for call in o.tool_calls)


def _no_tool(c: Check, o: Outcome) -> bool:
    return not any(c.name is None or call.name == c.name for call in o.tool_calls)


def _no_text_before_tool(_: Check, o: Outcome) -> bool:
    return not o.text_before_tool


def _no_markdown(_: Check, o: Outcome) -> bool:
    if "**" in o.text or "`" in o.text:
        return False
    return not any(_MARKDOWN_LINE.match(line.lstrip()) for line in o.text.splitlines())


def _no_markup(_: Check, o: Outcome) -> bool:
    return not _MARKUP.search(o.text)


_CHECKS: Final[Mapping[str, Callable[[Check, Outcome], bool]]] = MappingProxyType({
    "silent": _silent,
    "speaks": _speaks,
    "tool": _tool,
    "no_tool": _no_tool,
    "no_text_before_tool": _no_text_before_tool,
    "no_markdown": _no_markdown,
    "no_markup": _no_markup,
})
CHECK_KINDS: Final = tuple(_CHECKS)


@dataclass(frozen=True, slots=True)
class Check:
    """One entry of a case's ``expect``. ``tool`` takes ``name`` (required) and ``args`` (an
    optional subset the call's arguments must contain, compared by equality per key). ``no_tool``
    takes an optional ``name``: with it, no call of that tool; without it, no tool call at all.

    JSON: ``{"check": "<kind>"}``, ``{"check": "tool", "name": "...", "args": {...}}`` or
    ``{"check": "no_tool", "name": "..."}``.
    """

    kind: str
    name: str | None = None
    args: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _need(isinstance(self.kind, str) and self.kind in _CHECKS, f"check: unknown kind {self.kind!r}")
        if self.kind == "tool":
            _need(_nonempty_str(self.name), "check tool: needs a name")
            _need(self.args is None or isinstance(self.args, Mapping), "check tool: args must be an object")
            if self.args is not None:
                object.__setattr__(self, "args", _freeze(self.args))
        elif self.kind == "no_tool":
            _need(self.name is None or _nonempty_str(self.name), "check no_tool: name must be a non-empty string")
            _need(self.args is None, "check no_tool: takes no args")
        else:
            _need(self.name is None and self.args is None, f"check {self.kind}: takes no name or args")

    @classmethod
    def from_json(cls, doc: Any) -> Check:
        _need(isinstance(doc, Mapping), "check: must be an object")
        _keys(doc, {"check"}, {"name", "args"}, "check")
        return cls(doc["check"], doc.get("name"), doc.get("args"))

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"check": self.kind}
        if self.name is not None:
            out["name"] = self.name
        if self.args is not None:
            out["args"] = dict(self.args)
        return out


def passes(check: Check, outcome: Outcome) -> bool:
    """Whether ``outcome`` passes ``check`` (pure)."""
    return _CHECKS[check.kind](check, outcome)


def failed(expect: Sequence[Check], outcome: Outcome) -> list[Check]:
    """The checks of ``expect`` that ``outcome`` fails, in order (empty: the run passed)."""
    return [c for c in expect if not passes(c, outcome)]


# --- the case file -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Case:
    """One case in one language ("Initial cases").

    ``history`` is the conversation injected into the agent before the turn, as Bedrock Converse
    messages (``{"role": "user" | "assistant", "content": [...]}``; the content blocks are the
    runner's). ``caller`` is what the caller says on the turn, or ``None`` when the turn runs
    on the history alone (its last message is a tool result). ``tool_results`` are the scripted
    results by tool name for calls the model makes during the turn. ``expect`` is the checks.
    ``context`` is who the caller is, when not the verified fixture customer with no memory and no
    web block: ``{"verified": false}`` (unverified), or any of ``"memory"`` (recap text, wrapped
    as the bridge wraps it) and ``"web_block"`` (``"offer"`` or ``"context"``); ``None`` is the default.
    """

    id: str
    lang: str
    history: tuple[Mapping[str, Any], ...]
    caller: str | None
    tool_results: Mapping[str, Any]
    expect: tuple[Check, ...]
    context: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Suite:
    """The parsed case file: ``runs`` per case, the cases in file order, and :func:`suite_hash`."""

    runs: int
    cases: tuple[Case, ...]
    hash: str

    def for_lang(self, lang: str) -> tuple[Case, ...]:
        return tuple(c for c in self.cases if c.lang == lang)


def suite_hash(doc: Mapping[str, Any]) -> str:
    """SHA-256 (hex) over the case file's content: canonical JSON, so formatting and key
    order don't count, and any change to a value or to the case order does."""
    return hashlib.sha256(_canonical(doc).encode("ascii")).hexdigest()


def params_hash(params: Mapping[str, Any]) -> str:
    """SHA-256 (hex) over an entry's model parameters: :data:`PARAM_KEYS` only, as canonical
    JSON, so ``id``, ``prompt_caching`` and key order don't count. ``KeyError`` for a missing parameter."""
    return hashlib.sha256(_canonical({k: params[k] for k in PARAM_KEYS}).encode("ascii")).hexdigest()


def _parse_context(ctx: Any, what: str) -> Mapping[str, Any]:
    _need(isinstance(ctx, Mapping), f"{what}: context must be an object")
    _keys(ctx, set(), {"verified", "memory", "web_block"}, f"{what}: context")
    _need(type(ctx.get("verified", True)) is bool, f"{what}: context.verified must be a boolean")
    _need("memory" not in ctx or _nonempty_str(ctx["memory"]), f"{what}: context.memory must be a non-empty string")
    _need(ctx.get("web_block", "offer") in ("offer", "context"), f"{what}: context.web_block is offer or context")
    _need(ctx.get("verified", True) or set(ctx) == {"verified"},
          f"{what}: an unverified context has no memory and no web_block")
    return _freeze(ctx)


def _parse_case(doc: Any, i: int) -> Case:
    what = f"case {i}"
    _need(isinstance(doc, Mapping), f"{what}: must be an object")
    _keys(doc, {"id", "lang", "history", "caller", "tool_results", "expect"}, {"context"}, what)
    _need(_nonempty_str(doc["id"]), f"{what}: id must be a non-empty string")
    _need(_nonempty_str(doc["lang"]), f"{what}: lang must be a non-empty string")
    _need(doc["caller"] is None or isinstance(doc["caller"], str), f"{what}: caller must be a string or null")
    history = doc["history"]
    _need(isinstance(history, list), f"{what}: history must be a list")
    for m in history:
        _need(isinstance(m, Mapping) and set(m) == {"role", "content"} and m["role"] in _HISTORY_ROLES
              and isinstance(m["content"], list), f"{what}: history messages are {{role, content: [...]}}")
    _need(isinstance(doc["tool_results"], Mapping), f"{what}: tool_results must be an object")
    expect = doc["expect"]
    _need(isinstance(expect, list) and bool(expect), f"{what}: expect must be a non-empty list")
    return Case(id=doc["id"], lang=doc["lang"], history=tuple(history), caller=doc["caller"],
                tool_results=_freeze(doc["tool_results"]), expect=tuple(Check.from_json(c) for c in expect),
                context=_parse_context(doc["context"], what) if "context" in doc else None)


def parse_suite(doc: Mapping[str, Any]) -> Suite:
    """The case file: ``{"version": 1, "runs": <int >= 1>, "cases": [<case>, ...]}``; each case
    ``{"id", "lang", "history", "caller", "tool_results", "expect"[, "context"]}`` (see :class:`Case`), unique
    by ``(id, lang)``."""
    _need(isinstance(doc, Mapping), "suite: must be an object")
    _keys(doc, {"version", "cases"}, {"runs"}, "suite")
    _need(doc["version"] == SUITE_VERSION and type(doc["version"]) is int, "suite: unknown version")
    runs = doc.get("runs", DEFAULT_RUNS)
    _need(type(runs) is int and runs >= 1, "suite: runs must be an integer >= 1")
    raw = doc["cases"]
    _need(isinstance(raw, list) and bool(raw), "suite: cases must be a non-empty list")
    cases = tuple(_parse_case(c, i) for i, c in enumerate(raw))
    ids = [(c.id, c.lang) for c in cases]
    _need(len(set(ids)) == len(ids), "suite: duplicate (id, lang)")
    return Suite(runs=runs, cases=cases, hash=suite_hash(doc))


def load_suite(path: Path) -> Suite:
    return parse_suite(json.loads(path.read_text(encoding="utf-8")))


# --- the records ------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CaseTally:
    """Runs and passes of one case in a validating run."""

    runs: int
    passes: int

    def to_json(self) -> dict[str, int]:
        return {"passes": self.passes, "runs": self.runs}


@dataclass(frozen=True, slots=True)
class Record:
    """A full passing run of the suite for one model and language.

    ``model_key`` is the series key (``haiku-5-5``); ``model_id`` the profile ID the run used;
    ``params_hash`` its parameters' :func:`params_hash`; ``date`` ISO ``YYYY-MM-DD``; ``cases`` the
    runs and passes per case ID, every case passing every run (a record is only written for a full
    pass).
    """

    model_key: str
    lang: str
    prompt_hash: str
    suite_hash: str
    params_hash: str
    model_id: str
    date: str
    cases: Mapping[str, CaseTally]

    def __post_init__(self) -> None:
        for name in ("model_key", "lang", "model_id"):
            _need(_nonempty_str(getattr(self, name)), f"record: {name} must be a non-empty string")
        for name in ("prompt_hash", "suite_hash", "params_hash"):
            value = getattr(self, name)
            _need(isinstance(value, str) and bool(_HEX64.fullmatch(value)),
                  f"record: {name} must be 64 lowercase hex digits")
        _need(isinstance(self.date, str) and bool(_DATE.fullmatch(self.date)), "record: date must be YYYY-MM-DD")
        try:
            datetime.date.fromisoformat(self.date)
        except ValueError as e:
            raise SchemaError("record: date must be YYYY-MM-DD") from e
        _need(isinstance(self.cases, Mapping) and bool(self.cases), "record: cases must be a non-empty object")
        for case_id, t in self.cases.items():
            _need(_nonempty_str(case_id), "record: case IDs must be non-empty strings")
            _need(isinstance(t, CaseTally) and type(t.runs) is int and type(t.passes) is int
                  and t.runs >= 1 and t.passes == t.runs, f"record: case {case_id!r} must pass every run")
        object.__setattr__(self, "cases", MappingProxyType(dict(sorted(self.cases.items()))))

    @property
    def key(self) -> tuple[str, str]:
        return self.model_key, self.lang

    def full_runs(self, suite: Suite) -> bool:
        """Every case of ``suite`` in the record's language ran at least the case file's ``runs``
        times (9.4: a ``--runs 1`` pass is no validation)."""
        return all(c.id in self.cases and self.cases[c.id].runs >= suite.runs for c in suite.for_lang(self.lang))

    @classmethod
    def from_json(cls, doc: Any) -> Record:
        _need(isinstance(doc, Mapping), "record: must be an object")
        _keys(doc, {"model_key", "lang", "prompt_hash", "suite_hash", "params_hash", "model_id", "date", "cases"},
              set(), "record")
        cases = doc["cases"]
        _need(isinstance(cases, Mapping), "record: cases must be an object")
        tallies: dict[str, CaseTally] = {}
        for case_id, t in cases.items():
            _need(isinstance(t, Mapping), f"record: case {case_id!r} must be an object")
            _keys(t, {"runs", "passes"}, set(), f"record case {case_id!r}")
            tallies[case_id] = CaseTally(runs=t["runs"], passes=t["passes"])
        return cls(model_key=doc["model_key"], lang=doc["lang"], prompt_hash=doc["prompt_hash"],
                   suite_hash=doc["suite_hash"], params_hash=doc["params_hash"], model_id=doc["model_id"],
                   date=doc["date"], cases=tallies)

    def to_json(self) -> dict[str, Any]:
        return {"model_key": self.model_key, "lang": self.lang, "prompt_hash": self.prompt_hash,
                "suite_hash": self.suite_hash, "params_hash": self.params_hash, "model_id": self.model_id,
                "date": self.date,
                "cases": {k: t.to_json() for k, t in self.cases.items()}}


@dataclass(frozen=True, slots=True)
class ValidationFile:
    """``data/prompt_validation.json``: ``{"version": 1, "header": {...}, "records": [...]}``.

    ``header`` is free-form JSON (the first run's measured tokens and cost); records
    are unique by ``(model_key, lang)`` and kept sorted by it.
    """

    header: Mapping[str, Any]
    records: tuple[Record, ...]

    def __post_init__(self) -> None:
        records = tuple(sorted(self.records, key=lambda r: r.key))
        keys = [r.key for r in records]
        _need(len(set(keys)) == len(keys), "records: duplicate (model_key, lang)")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "header", _freeze(self.header))

    @classmethod
    def empty(cls) -> ValidationFile:
        return cls(header={}, records=())

    def add(self, record: Record) -> ValidationFile:
        """A copy with ``record`` in place of any record for its model and language."""
        return ValidationFile(header=self.header,
                              records=(*(r for r in self.records if r.key != record.key), record))

    def record_for(self, model_key: str, lang: str) -> Record | None:
        return next((r for r in self.records if r.key == (model_key, lang)), None)

    def is_validated(self, catalog: PromptCatalog, model_key: str, lang: str, suite: Suite,
                     params_hash: str | None = None) -> bool:
        """True when the record for ``model_key`` and ``lang`` carries the catalog's current
        ``prompt_hash`` and ``suite``'s hash, with full runs of every case (:meth:`Record.full_runs`),
        and ``params_hash`` when given (preflight's, for an entry of the list to deploy; the seed
        doesn't know the list). False when the model has no text in the catalog."""
        record = self.record_for(model_key, lang)
        if record is None or record.suite_hash != suite.hash or not record.full_runs(suite):
            return False
        if params_hash is not None and record.params_hash != params_hash:
            return False
        try:
            return record.prompt_hash == prompt_hash(catalog, model_key, lang)
        except CatalogKeyError:
            return False

    def to_json(self) -> dict[str, Any]:
        return {"version": RECORDS_VERSION, "header": dict(self.header),
                "records": [r.to_json() for r in self.records]}

    def dumps(self) -> str:
        """The file's text: deterministic for equal files (sorted keys and records, two-space indent,
        non-ASCII kept, one trailing newline)."""
        return json.dumps(self.to_json(), sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def parse_records(doc: Any) -> ValidationFile:
    _need(isinstance(doc, Mapping), "records file: must be an object")
    _keys(doc, {"version", "header", "records"}, set(), "records file")
    _need(doc["version"] == RECORDS_VERSION and type(doc["version"]) is int, "records file: unknown version")
    _need(isinstance(doc["header"], Mapping), "records file: header must be an object")
    _need(isinstance(doc["records"], list), "records file: records must be a list")
    return ValidationFile(header=doc["header"], records=tuple(Record.from_json(r) for r in doc["records"]))


def load_records(path: Path) -> ValidationFile:
    return parse_records(json.loads(path.read_text(encoding="utf-8")))


def save_records(path: Path, records: ValidationFile) -> None:
    path.write_text(records.dumps(), encoding="utf-8")
