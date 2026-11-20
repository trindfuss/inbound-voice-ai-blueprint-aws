"""Latency-masking settings, phrase validation, placeholders and rendering.

The one validator for masking data, shared by the seed, ``scripts/masking.py``, the bridge's
policy loader and the admin console API, so the writers and the reader can't disagree.

**Settings** (the policy table). A tool's item may carry ``mask``: ``null`` (unmasked) or
``{"phrase_key": "mask.<tool>", "after_ms"?: int, "handoff_wait_ms"?: int}``
(:class:`ToolMask`). The ``masking`` item holds the policy-wide :class:`MaskingSettings`.
Delays are integers 0..10000 ms.

Two modes, one rule set:

* ``strict=True`` (seed, script, console): returns the value or raises
  :class:`owlbank.policy.PolicySectionError` naming the first bad field.
* ``strict=False`` (bridge): never raises; returns ``(value, problems)``. Each bad field
  takes that field's default (a bad ``phrase_key`` or a non-object ``mask`` makes the tool
  unmasked) and adds a :class:`Problem` the caller logs as an ERROR.

Strict accepts exactly the inputs for which lenient reports no problem, and then returns the
same value.

**Phrases** (the catalog). A tool's phrase pool is the catalog key ``phrase_key`` itself, if
present, plus every ``phrase_key.<n>`` (``n`` a positive integer, ordered by ``n``;
:func:`pool_keys`). A phrase may contain ``{argument}`` placeholders, but only for the tool's
own enumerated arguments (:func:`placeholders_for`, from :mod:`owlbank.tools_contract`), and
each value is spoken through the catalog word ``arg.<argument>.<value>`` in the call's
language. :func:`render` is all-or-nothing: it returns ``None`` rather than speak a raw
placeholder, a value outside the enum or a stray brace.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, fields
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

from owlbank.tools_contract import DEFAULT_ESCALATION_REASONS, GATEWAY_LEVEL0_TOOLS, _schemas

if TYPE_CHECKING:  # the error class lives in owlbank.policy, which imports this module
    from owlbank.policy import PolicySectionError

# --------------------------------------------------------------------------- constants

MASK_PREFIX = "mask."
MAX_MASK_KEY_LEN = 128
ARG_PREFIX = "arg."

END_MODES: frozenset[str] = frozenset({"wait", "immediate"})
SILENCE_ACTIONS: frozenset[str] = frozenset({"escalate", "hangup"})

DELAY_MS_RANGE = (0, 10000)
MASK_AFTER_MS_RANGE = DELAY_MS_RANGE
TOOL_AFTER_MS_RANGE = DELAY_MS_RANGE
HANDOFF_WAIT_MS_RANGE = DELAY_MS_RANGE
SILENCE_TIMEOUT_MS_RANGE = DELAY_MS_RANGE
GREETING_SILENCE_TIMEOUT_MS_RANGE = (0, 30000)
SILENCE_MAX_REPEATS_RANGE = (0, 5)
CHARS_PER_S_RANGE = (8.0, 30.0)

DEFAULT_MASK_AFTER_MS = 1000
DEFAULT_TOOL_AFTER_MS = 0
DEFAULT_HANDOFF_WAIT_MS = 2500
DEFAULT_END_MODE = "wait"
DEFAULT_CHARS_PER_S = 15.0
DEFAULT_SILENCE_TIMEOUT_MS = 4000
DEFAULT_GREETING_SILENCE_TIMEOUT_MS = 7000
DEFAULT_SILENCE_MAX_REPEATS = 2  # three chances before the silence action
DEFAULT_SILENCE_ACTION = "escalate"

#: ``{name}``, the same placeholder syntax as :data:`owlbank.catalog.PLACEHOLDER`.
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_POOL_SUFFIX = re.compile(r"[1-9][0-9]{0,5}")


@dataclass(frozen=True, slots=True)
class Problem:
    """One invalid field or phrase: ``field`` names it (``masking.after_ms``,
    ``tools.get_balance.mask.phrase_key``, ``catalog.mask.get_balance.1[es-US]``)."""

    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"


def _section_error(problem: Problem) -> PolicySectionError:
    from owlbank.policy import PolicySectionError  # noqa: PLC0415 (import cycle)
    return PolicySectionError(problem.field, problem.message)


# --------------------------------------------------------------------------- data


@dataclass(frozen=True, slots=True)
class ToolMask:
    """A tool's masking settings. ``None`` delays mean "use the policy-wide value"."""

    phrase_key: str
    after_ms: int | None = None
    handoff_wait_ms: int | None = None

    def to_doc(self) -> dict[str, Any]:
        """The policy-item shape (unset delays omitted)."""
        doc: dict[str, Any] = {"phrase_key": self.phrase_key}
        if self.after_ms is not None:
            doc["after_ms"] = self.after_ms
        if self.handoff_wait_ms is not None:
            doc["handoff_wait_ms"] = self.handoff_wait_ms
        return doc


@dataclass(frozen=True, slots=True)
class MaskingSettings:
    """Policy-wide masking values.

    Field order keeps the original four first, so ``MaskingSettings(after_ms, handoff_wait_ms,
    end_mode, chars_per_s)`` still works. ``after_ms == 0`` turns the timed acknowledgement
    off; ``silence_timeout_ms == 0`` turns the silence watch off.
    ``greeting_silence_timeout_ms`` replaces ``silence_timeout_ms`` after the opening
    greeting of the call's first AI session only; 0 = no silence repeat after that greeting.
    """

    after_ms: int = DEFAULT_MASK_AFTER_MS
    handoff_wait_ms: int = DEFAULT_HANDOFF_WAIT_MS
    end_mode: str = DEFAULT_END_MODE
    chars_per_s: float = DEFAULT_CHARS_PER_S
    tool_after_ms: int = DEFAULT_TOOL_AFTER_MS
    silence_timeout_ms: int = DEFAULT_SILENCE_TIMEOUT_MS
    silence_max_repeats: int = DEFAULT_SILENCE_MAX_REPEATS
    silence_action: str = DEFAULT_SILENCE_ACTION
    greeting_silence_timeout_ms: int = DEFAULT_GREETING_SILENCE_TIMEOUT_MS

    def to_doc(self) -> dict[str, Any]:
        """The seed-file / policy-item shape. A non-integral ``chars_per_s`` as a decimal
        string (the store holds no floats)."""
        cps: int | str = int(self.chars_per_s) if self.chars_per_s.is_integer() \
            else repr(self.chars_per_s)
        return {"after_ms": self.after_ms, "tool_after_ms": self.tool_after_ms,
                "handoff_wait_ms": self.handoff_wait_ms, "end_mode": self.end_mode,
                "chars_per_s": cps, "silence_timeout_ms": self.silence_timeout_ms,
                "silence_max_repeats": self.silence_max_repeats,
                "silence_action": self.silence_action,
                "greeting_silence_timeout_ms": self.greeting_silence_timeout_ms}


DEFAULT_SETTINGS = MaskingSettings()
MASKING_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(MaskingSettings))
TOOL_MASK_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(ToolMask))


# --------------------------------------------------------------------------- field checks


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _check_int(value: Any, field_name: str, bounds: tuple[int, int]) -> int | Problem:
    lo, hi = bounds
    if not _is_int(value) or not lo <= value <= hi:
        return Problem(field_name, f"must be an integer from {lo} to {hi}")
    return int(value)


def _check_number(value: Any, field_name: str, bounds: tuple[float, float]) -> float | Problem:
    lo, hi = bounds
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal | str):
        return Problem(field_name, "must be a number")
    try:
        num = float(value)
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        return Problem(field_name, "must be a number")
    if not math.isfinite(num) or not lo <= num <= hi:
        return Problem(field_name, f"must be a number from {lo:g} to {hi:g}")
    return num


def _check_choice(value: Any, field_name: str, choices: frozenset[str]) -> str | Problem:
    if not isinstance(value, str) or value not in choices:
        return Problem(field_name, f"must be one of {sorted(choices)}")
    return value


def _masking_field(raw: Mapping[str, Any], name: str) -> Any:
    value, f = raw[name], f"masking.{name}"
    if name == "end_mode":
        return _check_choice(value, f, END_MODES)
    if name == "silence_action":
        return _check_choice(value, f, SILENCE_ACTIONS)
    if name == "chars_per_s":
        return _check_number(value, f, CHARS_PER_S_RANGE)
    if name == "silence_max_repeats":
        return _check_int(value, f, SILENCE_MAX_REPEATS_RANGE)
    if name == "greeting_silence_timeout_ms":
        return _check_int(value, f, GREETING_SILENCE_TIMEOUT_MS_RANGE)
    return _check_int(value, f, DELAY_MS_RANGE)  # after_ms, tool_after_ms, handoff_wait_ms, silence_timeout_ms


def _key_problem(key: Any, field_name: str) -> Problem | None:
    if not isinstance(key, str) or not key.startswith(MASK_PREFIX) or len(key) == len(MASK_PREFIX) \
            or len(key) > MAX_MASK_KEY_LEN or "{" in key or "}" in key:
        return Problem(field_name, f"must be a catalog key starting with {MASK_PREFIX!r}")
    return None


# --------------------------------------------------------------------------- parsing


def masking_parts(raw: Any) -> tuple[MaskingSettings, list[Problem]]:
    """Lenient parse of the ``masking`` item: each valid field, the default for each
    invalid one, and the problems in field order (an unknown key first). ``None`` (no item)
    is the defaults with no problem."""
    if raw is None:
        return DEFAULT_SETTINGS, []
    if not isinstance(raw, Mapping):
        return DEFAULT_SETTINGS, [Problem("masking", "must be an object")]
    problems: list[Problem] = []
    unknown = sorted(str(k) for k in raw if k not in MASKING_FIELDS)
    if unknown:
        problems.append(Problem(f"masking.{unknown[0]}", "unknown field"))
    values: dict[str, Any] = {}
    for name in MASKING_FIELDS:
        if name not in raw:
            continue
        got = _masking_field(raw, name)
        if isinstance(got, Problem):
            problems.append(got)
        else:
            values[name] = got
    return MaskingSettings(**values), problems


def tool_mask_parts(name: str, raw: Any) -> tuple[ToolMask | None, list[Problem]]:
    """Lenient parse of a tool's ``mask``. ``None`` is unmasked with no problem; a non-object
    or a bad ``phrase_key`` is unmasked with a problem; a bad delay is unset (the policy-wide
    value) with a problem; an unknown key is ignored with a problem."""
    if raw is None:
        return None, []
    base = f"tools.{name}.mask"
    if not isinstance(raw, Mapping):
        return None, [Problem(base, 'must be null or {"phrase_key": "mask.<name>", '
                                    '"after_ms"?: int, "handoff_wait_ms"?: int}')]
    problems: list[Problem] = []
    unknown = sorted(str(k) for k in raw if k not in TOOL_MASK_FIELDS)
    if unknown:
        problems.append(Problem(f"{base}.{unknown[0]}", "unknown field"))
    key = raw.get("phrase_key")
    key_problem = _key_problem(key, f"{base}.phrase_key")
    if key_problem is not None:
        problems.append(key_problem)
    delays: dict[str, int] = {}
    for f in ("after_ms", "handoff_wait_ms"):
        value = raw.get(f)
        if value is None:  # absent or null: the policy-wide value
            continue
        got = _check_int(value, f"{base}.{f}", DELAY_MS_RANGE)
        if isinstance(got, Problem):
            problems.append(got)
        else:
            delays[f] = got
    if key_problem is not None or not isinstance(key, str):
        return None, problems
    return ToolMask(phrase_key=key, **delays), problems


@overload
def parse_masking(raw: Any, *, strict: Literal[True]) -> MaskingSettings: ...
@overload
def parse_masking(raw: Any, *, strict: Literal[False]) -> tuple[MaskingSettings, list[Problem]]: ...
def parse_masking(raw: Any, *, strict: bool) -> Any:
    """The ``masking`` item. Strict: the settings, or :class:`~owlbank.policy.PolicySectionError`
    naming the first bad field. Lenient: ``(settings, problems)``, never raises."""
    value, problems = masking_parts(raw)
    if not strict:
        return value, problems
    if problems:
        raise _section_error(problems[0])
    return value


@overload
def parse_tool_mask(name: str, raw: Any, *, strict: Literal[True]) -> ToolMask | None: ...
@overload
def parse_tool_mask(name: str, raw: Any, *,
                    strict: Literal[False]) -> tuple[ToolMask | None, list[Problem]]: ...
def parse_tool_mask(name: str, raw: Any, *, strict: bool) -> Any:
    """A tool's ``mask`` value (``None`` = unmasked). Strict raises on the first problem;
    lenient returns ``(mask, problems)``, never raises."""
    value, problems = tool_mask_parts(name, raw)
    if not strict:
        return value, problems
    if problems:
        raise _section_error(problems[0])
    return value


# --------------------------------------------------------------------------- placeholders


def placeholders_for(tool: str, *,
                     reasons: Collection[str] = DEFAULT_ESCALATION_REASONS) -> dict[str, list[str]]:
    """The placeholders a phrase for ``tool`` may use: ``{argument: [allowed values]}`` for
    each argument whose input schema has an ``enum``. Gateway tools and unknown
    tools have none. ``reasons`` is the policy's escalation reasons (the ``escalate`` enum)."""
    if tool in GATEWAY_LEVEL0_TOOLS:
        return {}  # their Gateway schemas (infra contract) have no enumerated arguments
    entry = _schemas(reasons).get(tool)
    if entry is None:
        return {}
    props = entry[1].get("properties", {})
    out: dict[str, list[str]] = {}
    for arg, schema in props.items():
        enum = schema.get("enum") if isinstance(schema, Mapping) else None
        if isinstance(enum, list | tuple) and enum and all(isinstance(v, str) for v in enum):
            out[arg] = list(enum)
    return out


def word_key(arg: str, value: str) -> str:
    """The catalog key of the spoken word for an enum value: ``arg.<argument>.<value>``."""
    return f"{ARG_PREFIX}{arg}.{value}"


def placeholder_names(text: str) -> list[str]:
    """The ``{name}`` placeholders in ``text``, in order of first use."""
    return list(dict.fromkeys(PLACEHOLDER.findall(text)))


def has_stray_brace(text: str) -> bool:
    """A ``{`` or ``}`` that is not part of a ``{name}`` placeholder."""
    rest = PLACEHOLDER.sub("", text)
    return "{" in rest or "}" in rest


# --------------------------------------------------------------------------- pools


def pool_keys(phrase_key: str, catalog_keys: Iterable[str]) -> list[str]:
    """The tool's phrase pool: ``phrase_key`` if it's a catalog key, then each
    ``phrase_key.<n>`` (``n`` = 1..999999, no leading zero) ordered by ``n``."""
    keys = set(catalog_keys)
    prefix = phrase_key + "."
    numbered = sorted((int(k[len(prefix):]), k) for k in keys
                      if k.startswith(prefix) and _POOL_SUFFIX.fullmatch(k[len(prefix):]))
    return ([phrase_key] if phrase_key in keys else []) + [k for _, k in numbered]


# --------------------------------------------------------------------------- validation


class CatalogLike(Protocol):
    """What validation needs from :class:`owlbank.catalog.PromptCatalog`."""

    @property
    def keys(self) -> frozenset[str]: ...
    def has(self, key: str, lang: str) -> bool: ...
    def template(self, key: str, lang: str) -> str: ...


def validate_phrase(tool: str, text: Any, catalog: CatalogLike, languages: Iterable[str], *,
                    field: str = "phrase",
                    reasons: Collection[str] = DEFAULT_ESCALATION_REASONS) -> list[Problem]:
    """Problems with one phrase for ``tool`` (empty = valid).

    Every placeholder must be one of :func:`placeholders_for` ``(tool)``; for each one used,
    every allowed value needs the catalog word ``arg.<name>.<value>`` in every language (with
    no braces in it); no stray ``{`` or ``}``."""
    if not isinstance(text, str) or not text.strip():
        return [Problem(field, "must be non-empty text")]
    problems: list[Problem] = []
    if has_stray_brace(text):
        problems.append(Problem(field, "has a '{' or '}' that isn't a {placeholder}"))
    allowed = placeholders_for(tool, reasons=reasons)
    langs = list(languages)
    for name in placeholder_names(text):
        if name not in allowed:
            problems.append(Problem(field, f"placeholder {{{name}}} is not an enumerated argument "
                                           f"of {tool} (allowed: {sorted(allowed)})"))
            continue
        for value in allowed[name]:
            key = word_key(name, value)
            for lang in langs:
                if not catalog.has(key, lang):
                    problems.append(Problem(field, f"no catalog word {key} in {lang}"))
                    continue
                word = catalog.template(key, lang)
                if not word.strip() or "{" in word or "}" in word:
                    problems.append(Problem(field, f"catalog word {key} in {lang} must be "
                                                   "non-empty text without braces"))
    return problems


def validate_catalog_for_tool(tool: str, mask: ToolMask, catalog: CatalogLike,
                              languages: Iterable[str], *,
                              reasons: Collection[str] = DEFAULT_ESCALATION_REASONS) -> list[Problem]:
    """Problems with a masked tool's whole pool: at least one phrase, every phrase in every
    language, each valid by :func:`validate_phrase`."""
    langs = list(languages)
    pool = pool_keys(mask.phrase_key, catalog.keys)
    if not pool:
        return [Problem(f"tools.{tool}.mask.phrase_key",
                        f"no catalog phrase {mask.phrase_key} or {mask.phrase_key}.<n>")]
    problems: list[Problem] = []
    for key in pool:
        for lang in langs:
            field = f"catalog.{key}[{lang}]"
            if not catalog.has(key, lang):
                problems.append(Problem(field, "missing"))
                continue
            problems.extend(validate_phrase(tool, catalog.template(key, lang), catalog, [lang],
                                            field=field, reasons=reasons))
    return problems


# --------------------------------------------------------------------------- rendering


def words_for(tool: str, catalog: CatalogLike, lang: str, *,
              reasons: Collection[str] = DEFAULT_ESCALATION_REASONS) -> dict[str, str]:
    """``{arg.<name>.<value>: word}`` for ``tool``'s enum values that have a word in exactly
    ``lang`` (no fallback to another language), ready for :func:`render`."""
    out: dict[str, str] = {}
    for name, values in placeholders_for(tool, reasons=reasons).items():
        for value in values:
            key = word_key(name, value)
            if catalog.has(key, lang):
                out[key] = catalog.template(key, lang)
    return out


def render(tool: str, text: str, args: Any, words: Mapping[str, str], *,
           reasons: Collection[str] = DEFAULT_ESCALATION_REASONS) -> str | None:
    """``text`` with each ``{name}`` replaced by the word for the call's argument value, or
    ``None`` if it can't be rendered in full: a placeholder that isn't an enum argument
    of ``tool``, an argument missing or outside the enum, no (or an empty or braced) word, or
    a stray brace. Substitution is one pass, so a word is never expanded."""
    if not isinstance(text, str) or has_stray_brace(text):
        return None
    names = placeholder_names(text)
    if not names:
        return text
    if not isinstance(args, Mapping):
        return None
    allowed = placeholders_for(tool, reasons=reasons)
    fill: dict[str, str] = {}
    for name in names:
        value = args.get(name)
        if name not in allowed or not isinstance(value, str) or value not in allowed[name]:
            return None
        word = words.get(word_key(name, value))
        if not isinstance(word, str) or not word.strip() or "{" in word or "}" in word:
            return None
        fill[name] = word
    out = PLACEHOLDER.sub(lambda m: fill[m.group(1)], text)
    return None if "{" in out or "}" in out else out


__all__ = [
    "ARG_PREFIX", "CHARS_PER_S_RANGE", "DEFAULT_SETTINGS", "DELAY_MS_RANGE", "END_MODES",
    "GREETING_SILENCE_TIMEOUT_MS_RANGE",
    "HANDOFF_WAIT_MS_RANGE", "MASKING_FIELDS", "MASK_AFTER_MS_RANGE", "MASK_PREFIX",
    "SILENCE_ACTIONS", "SILENCE_MAX_REPEATS_RANGE", "SILENCE_TIMEOUT_MS_RANGE",
    "TOOL_AFTER_MS_RANGE", "CatalogLike", "MaskingSettings", "Problem", "ToolMask",
    "masking_parts", "parse_masking", "parse_tool_mask", "placeholder_names", "placeholders_for",
    "pool_keys", "render", "tool_mask_parts", "validate_catalog_for_tool", "validate_phrase",
    "word_key", "words_for", "has_stray_brace",
]
