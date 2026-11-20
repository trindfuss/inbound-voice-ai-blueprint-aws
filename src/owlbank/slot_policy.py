"""Per-slot settings from the policy table (model-assisted-slots, phase 1).

Every slot the front door asks for has settings. A ``factors`` entry (``zip``, ``ssn4``, ``otp``)
takes its shape (slot type, constraints, keys, retries) from its factor definition; any other
slot (``yes_no``) carries its shape in its own item. Phase 1 fields:

* ``sensitive`` (default ``false``): what the AI, the journal and the logs see.
  Allowed on every slot type.
* ``keypad_only`` (default ``false``) and ``barge_in`` (default ``true``): the
  Gather's shape.
* ``flow``: the owning flow of a non-factor slot (one of :data:`FLOWS`).
* ``declinable`` (default ``false``): the caller may decline the slot
  (a resolver ``decline`` or ``#`` alone on the keypad). Allowed on every slot. A
  non-boolean one loads as ``false`` (the slot keeps its behaviour).

Phase 2 fields:

* ``mode`` (absent = inherit from the flow, then ``slot_resolution.default_mode``): one of
  :data:`MODES`. ``generative`` only on a non-sensitive choice slot.
* ``confirm`` (default ``when_inferred``): ``when_inferred`` | ``always`` | ``never``.
* ``options`` (choice slots only): the known options, each a ``value``
  (``[a-z][a-z0-9_]{0,31}``, unique), ``aliases`` per language and an optional keypad ``dtmf``.
  No normalized alias may belong to two options in one language (:func:`alias_conflicts`
  also checks the catalog's spoken names, which the seed and the admin API pass it).

The ``slot_resolution`` item (:class:`ResolutionSettings`) holds ``default_mode``, the per-flow
overrides, ``generative_enabled`` and ``resolver_budget_ms``; :func:`parse_slot_resolution` is
strict and :func:`load_slot_resolution` lenient.

Two validation modes share one rule set:

* :func:`parse_slot` (strict: seed file, console) raises :class:`SlotPolicyError` naming the
  item and field.
* :func:`load_slot` (lenient: the policy table at run time) never raises. A non-boolean
  ``sensitive``, ``keypad_only`` or ``barge_in`` loads as ``true`` (fail closed); a
  ``factors`` entry with no item at all loads ``sensitive: true``. A bad ``mode`` loads as
  ``deterministic`` (the error has ``mode_invalid`` set, so :mod:`owlbank.slots.modes`
  reports ``slot_mode_invalid``); ``generative`` where it is refused loads as ``assisted``; a
  bad ``confirm`` loads as ``always`` (the strictest); a bad option or a conflicting alias is
  dropped. Each problem is returned as a :class:`SlotPolicyError`; the caller logs it.

No error message ever carries a value from the item.

:func:`gather_input` decides the Gather's input for the front door and the admin API
alike.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeGuard

from owlbank.slots.normalize import normalize
from owlbank.slots.types import SLOT_TYPES, SlotDecl

#: The flows a non-factor slot may belong to (and the keys of ``slot_resolution.flows``).
FLOWS: tuple[str, ...] = ("login", "stepup", "confirm")

Mode = Literal["deterministic", "assisted", "generative"]
#: The modes, which is also their order: ``deterministic < assisted < generative``.
MODES: tuple[Mode, ...] = ("deterministic", "assisted", "generative")
ConfirmPolicy = Literal["when_inferred", "always", "never"]
CONFIRM_POLICIES: tuple[ConfirmPolicy, ...] = ("when_inferred", "always", "never")
ModeSource = Literal["slot", "flow", "default", "invalid"]
CapReason = Literal["ceiling", "generative_disabled", "type", "sensitive"]

#: ``slot_resolution`` defaults and the budget's range.
DEFAULT_MODE: Mode = "assisted"
DEFAULT_RESOLVER_BUDGET_MS = 2500
RESOLVER_BUDGET_MS_RANGE = (500, 4000)
RESOLUTION_FIELDS: tuple[str, ...] = ("default_mode", "flows", "generative_enabled", "resolver_budget_ms")

#: Option limits (validation).
OPTION_VALUE_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
OPTION_FIELDS: tuple[str, ...] = ("value", "aliases", "dtmf")
DTMF_KEYS: frozenset[str] = frozenset("0123456789")
MAX_OPTIONS = 20
MAX_ALIASES = 20
MAX_ALIAS_LEN = 64
MAX_LANG_LEN = 16

#: The three boolean settings, their defaults when absent, and the value an invalid one loads as.
SETTING_FIELDS: tuple[str, ...] = ("sensitive", "keypad_only", "barge_in")
SETTING_DEFAULTS: Mapping[str, bool] = {"sensitive": False, "keypad_only": False, "barge_in": True}
FAIL_CLOSED: Mapping[str, bool] = {"sensitive": True, "keypad_only": True, "barge_in": True}
#: May the caller decline the slot. Absent or invalid = ``false`` (unchanged behaviour).
DECLINABLE_FIELD = "declinable"
_MSG_DECLINABLE_LOADED = "must be a boolean; loaded as false"

#: A non-factor slot's shape fields (a factor's live in its ``factors`` entry).
SHAPE_FIELDS: tuple[str, ...] = ("slotType", "constraints", "promptKey", "labelKey", "maxRetries")

#: Phase 2 fields: ``mode`` and ``confirm`` on every slot item, ``options`` on a choice slot.
PHASE2_FIELDS: tuple[str, ...] = ("mode", "confirm", "options")
#: What a ``confirm`` that isn't valid loads as at run time: the strictest (every value read back).
CONFIRM_FAIL_CLOSED: ConfirmPolicy = "always"

_MSG_BOOL = "must be a boolean"
_MSG_BOOL_LOADED = "must be a boolean; loaded as true (fail closed)"


class SlotPolicyError(ValueError):
    """A slot item or field is invalid. ``field`` names the item and field (``slot#zip.sensitive``),
    never a value."""

    def __init__(self, field_name: str, message: str, *, mode_invalid: bool = False) -> None:
        super().__init__(f"{field_name}: {message}")
        self.field = field_name
        self.message = message
        #: A mode value that is missing where required, unknown or of the wrong type, loaded as
        #: ``deterministic``: :mod:`owlbank.slots.modes` reports ``slot_mode_invalid``.
        self.mode_invalid = mode_invalid


def is_mode(v: object) -> TypeGuard[Mode]:
    """``v`` is one of :data:`MODES` (a string; never a bool or a list that compares equal)."""
    return isinstance(v, str) and v in MODES


def mode_rank(m: Mode) -> int:
    """The position of ``m`` in :data:`MODES` (``deterministic`` = 0)."""
    return MODES.index(m)


@dataclass(frozen=True, slots=True)
class OptionDef:
    """One known option of a choice slot. The spoken name is catalog
    ``option.<slot>.<value>``; ``aliases`` maps a language to extra phrases; ``dtmf`` is the keypad
    key (default: the option's position among the caller's options)."""

    value: str
    aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    dtmf: str | None = None

    def to_doc(self) -> dict[str, Any]:
        doc: dict[str, Any] = {"value": self.value}
        if self.aliases:
            doc["aliases"] = {lang: list(a) for lang, a in self.aliases.items()}
        if self.dtmf is not None:
            doc["dtmf"] = self.dtmf
        return doc


@dataclass(frozen=True, slots=True)
class ResolutionSettings:
    """The ``slot_resolution`` item."""

    default_mode: Mode = DEFAULT_MODE
    flows: Mapping[str, Mode] = field(default_factory=dict)
    generative_enabled: bool = False
    resolver_budget_ms: int = DEFAULT_RESOLVER_BUDGET_MS

    def to_doc(self) -> dict[str, Any]:
        """The item body (without ``pk``); :func:`parse_slot_resolution` reads it back."""
        return {"default_mode": self.default_mode,
                "flows": {f: {"mode": m} for f, m in self.flows.items()},
                "generative_enabled": self.generative_enabled,
                "resolver_budget_ms": self.resolver_budget_ms}


@dataclass(frozen=True, slots=True)
class SlotSettings:
    """One slot's settings."""

    name: str
    decl: SlotDecl                 # shape: type, constraints, keys, max_retries
    is_factor: bool                # a ``factors`` entry (acceptSpokenFactors ceiling)
    sensitive: bool                # Absent = False; any slot type
    keypad_only: bool = False
    barge_in: bool = True
    flow: str | None = None        # owning flow of a non-factor slot
    # Slot modes
    mode: Mode | None = None       # None = inherit (flow, then default_mode)
    confirm: ConfirmPolicy = "when_inferred"
    options: tuple[OptionDef, ...] = ()  # choice only
    declinable: bool = False       # A resolver ``decline`` or ``*`` ends the flow

    def to_doc(self) -> dict[str, Any]:
        """The ``slot#<name>`` item body (without ``pk``); :func:`parse_slot` reads it back."""
        doc: dict[str, Any] = {}
        if not self.is_factor:
            d = self.decl
            doc["slotType"] = d.slot_type
            if d.constraints:
                doc["constraints"] = copy.deepcopy(dict(d.constraints))
            if d.prompt_key is not None:
                doc["promptKey"] = d.prompt_key
            if d.label_key is not None:
                doc["labelKey"] = d.label_key
            doc["maxRetries"] = d.max_retries
        doc.update(sensitive=self.sensitive, keypad_only=self.keypad_only, barge_in=self.barge_in)
        if self.flow is not None:
            doc["flow"] = self.flow
        if self.mode is not None:
            doc["mode"] = self.mode
        doc["confirm"] = self.confirm
        if self.options:
            doc["options"] = [o.to_doc() for o in self.options]
        if self.declinable:
            doc[DECLINABLE_FIELD] = True
        return doc


@dataclass(frozen=True, slots=True)
class GatherInput:
    """What the slot's Gather accepts."""

    keypad_only: bool
    barge_in: bool


def gather_input(slot: SlotSettings, accept_spoken_factors: bool) -> GatherInput:
    """Effective keypad-only = ``keypad_only or (is_factor and not accept_spoken_factors)``. Shared by the
    front door and the admin API."""
    return GatherInput(keypad_only=slot.keypad_only or (slot.is_factor and not accept_spoken_factors),
                       barge_in=slot.barge_in)


# --------------------------------------------------------------------------- validation


def _item(name: str) -> str:
    return f"slot#{name}"


def _decl(name: str, shape: Mapping[str, Any], *, where: str) -> SlotDecl:
    """A :class:`SlotDecl` from a shape (a factor definition or a non-factor item), strict.
    ``where`` prefixes the field names in errors."""
    slot_type = shape.get("slotType")
    if slot_type not in SLOT_TYPES:
        raise SlotPolicyError(f"{where}.slotType", f"must be one of {list(SLOT_TYPES)}")
    constraints = shape.get("constraints", {})
    if constraints is None:
        constraints = {}
    if not isinstance(constraints, Mapping):
        raise SlotPolicyError(f"{where}.constraints", "must be an object")
    keys: dict[str, str | None] = {}
    for f in ("promptKey", "labelKey"):
        v = shape.get(f)
        if v is not None and (not isinstance(v, str) or not v):
            raise SlotPolicyError(f"{where}.{f}", "must be a non-empty string")
        keys[f] = v
    retries = shape.get("maxRetries", 1)
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise SlotPolicyError(f"{where}.maxRetries", "must be an integer >= 0")
    return SlotDecl(slot_type=slot_type, constraints=copy.deepcopy(dict(constraints)),
                    label_key=keys["labelKey"], prompt_key=keys["promptKey"],
                    max_retries=retries)


def _allowed(is_factor: bool) -> frozenset[str]:
    if is_factor:
        return frozenset((*SETTING_FIELDS, DECLINABLE_FIELD, "mode", "confirm"))
    return frozenset((*SETTING_FIELDS, DECLINABLE_FIELD, *SHAPE_FIELDS, "flow", *PHASE2_FIELDS))


def _check_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise SlotPolicyError("slots", "slot names must be non-empty strings")


_LANG_RE = re.compile(r"[a-z]{2,3}(?:-[A-Z]{2})?")
_MSG_MODE = f"must be one of {list(MODES)}"
_MSG_GENERATIVE = "generative needs a non-sensitive choice slot"
_MSG_CONFIRM = f"must be one of {list(CONFIRM_POLICIES)}"
_MSG_OPTIONS_TYPE = "only on a choice slot"


@dataclass(frozen=True, slots=True)
class AliasConflict:
    """A normalized phrase (a spoken name or an alias) that belongs to two or more options in one
    language. ``options`` is sorted."""

    lang: str
    phrase: str
    options: tuple[str, ...]


def _phrases(o: OptionDef, lang: str, spoken_names: Mapping[str, Mapping[str, str]]) -> Iterable[str]:
    names = spoken_names.get(o.value)
    if isinstance(names, Mapping):
        name = names.get(lang)
        if isinstance(name, str):
            yield name
    yield from o.aliases.get(lang, ())


def alias_conflicts(options: Sequence[OptionDef], spoken_names: Mapping[str, Mapping[str, str]],
                    langs: Sequence[str]) -> list[AliasConflict]:
    """Every normalized spoken name or alias that maps to two options in one of ``langs``. Pure.
    ``spoken_names`` maps an option value to its catalog ``option.<slot>.<value>``
    text per language; the seed and the admin API pass the catalog's. A phrase that normalizes to
    nothing is not a conflict (validation refuses such an alias on its own)."""
    out: list[AliasConflict] = []
    for lang in dict.fromkeys(langs):
        owners: dict[str, set[str]] = {}
        for o in options:
            for text in _phrases(o, lang, spoken_names):
                key = normalize(text, lang)
                if key:
                    owners.setdefault(key, set()).add(o.value)
        out.extend(AliasConflict(lang, k, tuple(sorted(v))) for k, v in sorted(owners.items()) if len(v) > 1)
    return out


def _mode_field(item: str, raw: Mapping[str, Any], *, choice: bool, sensitive: bool
                ) -> tuple[Mode | None, list[SlotPolicyError]]:
    """``mode``: absent (or null) = inherit. Lenient result and errors; strict raises the first."""
    v = raw.get("mode")
    if v is None:
        return None, []
    if not is_mode(v):
        return "deterministic", [SlotPolicyError(f"{item}.mode", _MSG_MODE, mode_invalid=True)]
    if v == "generative" and (sensitive or not choice):
        return "assisted", [SlotPolicyError(f"{item}.mode", _MSG_GENERATIVE)]
    return v, []


def _confirm_field(item: str, raw: Mapping[str, Any]) -> tuple[ConfirmPolicy, list[SlotPolicyError]]:
    v = raw.get("confirm", "when_inferred")
    if isinstance(v, str) and v in CONFIRM_POLICIES:
        return v, []
    return CONFIRM_FAIL_CLOSED, [SlotPolicyError(f"{item}.confirm", _MSG_CONFIRM)]


def _aliases(where: str, raw: Any, langs: Sequence[str] | None
             ) -> tuple[dict[str, tuple[str, ...]], list[SlotPolicyError]]:
    if raw is None:
        return {}, []
    if not isinstance(raw, Mapping):
        return {}, [SlotPolicyError(where, "must be an object of language -> list of aliases")]
    out: dict[str, tuple[str, ...]] = {}
    errors: list[SlotPolicyError] = []
    for lang, phrases in raw.items():
        if not isinstance(lang, str) or not _LANG_RE.fullmatch(lang) or (langs is not None and lang not in langs):
            errors.append(SlotPolicyError(where, "has a key that isn't a supported language"))
            continue
        here = f"{where}.{lang}"
        if not isinstance(phrases, list):
            errors.append(SlotPolicyError(here, "must be a list of strings"))
            continue
        if len(phrases) > MAX_ALIASES:
            errors.append(SlotPolicyError(here, f"at most {MAX_ALIASES} aliases"))
            phrases = phrases[:MAX_ALIASES]
        kept: list[str] = []
        for j, a in enumerate(phrases):
            if not isinstance(a, str) or not 0 < len(a) <= MAX_ALIAS_LEN or not normalize(a, lang):
                errors.append(SlotPolicyError(f"{here}[{j}]", f"must be a phrase of 1..{MAX_ALIAS_LEN} "
                                              "characters with at least one word"))
                continue
            kept.append(a)
        out[lang] = tuple(kept)
    return out, errors


def _options_field(item: str, raw: Any, *, langs: Sequence[str] | None
                   ) -> tuple[tuple[OptionDef, ...], list[SlotPolicyError]]:
    """``options``: the lenient result (bad entries and conflicting aliases dropped) and every
    error in field order; strict raises the first."""
    if raw is None:
        return (), []
    where = f"{item}.options"
    if not isinstance(raw, list):
        return (), [SlotPolicyError(where, "must be a list of options")]
    errors: list[SlotPolicyError] = []
    if len(raw) > MAX_OPTIONS:
        errors.append(SlotPolicyError(where, f"at most {MAX_OPTIONS} options"))
        raw = raw[:MAX_OPTIONS]
    out: list[OptionDef] = []
    seen_values: set[str] = set()
    seen_dtmf: set[str] = set()
    for i, entry in enumerate(raw):
        here = f"{where}[{i}]"
        if not isinstance(entry, Mapping):
            errors.append(SlotPolicyError(here, "must be an object"))
            continue
        errors.extend(SlotPolicyError(f"{here}.{k}" if isinstance(k, str) and k.isidentifier() else here,
                                      "unknown field") for k in entry if k not in OPTION_FIELDS)
        value = entry.get("value")
        if not isinstance(value, str) or not OPTION_VALUE_RE.fullmatch(value):
            errors.append(SlotPolicyError(f"{here}.value", "must match [a-z][a-z0-9_]{0,31}"))
            continue
        if value in seen_values:
            errors.append(SlotPolicyError(f"{here}.value", "is the value of an earlier option"))
            continue
        seen_values.add(value)
        aliases, alias_errors = _aliases(f"{here}.aliases", entry.get("aliases"), langs)
        errors.extend(alias_errors)
        dtmf = entry.get("dtmf")
        if dtmf is not None and (not isinstance(dtmf, str) or dtmf not in DTMF_KEYS):
            errors.append(SlotPolicyError(f"{here}.dtmf", "must be one keypad digit"))
            dtmf = None
        elif dtmf is not None and dtmf in seen_dtmf:
            errors.append(SlotPolicyError(f"{here}.dtmf", "is the key of an earlier option"))
            dtmf = None
        if dtmf is not None:
            seen_dtmf.add(dtmf)
        out.append(OptionDef(value=value, aliases=aliases, dtmf=dtmf))
    check = list(langs) if langs is not None else sorted({lang for o in out for lang in o.aliases})
    conflicts = alias_conflicts(out, {}, check)
    for c in conflicts:
        errors.append(SlotPolicyError(f"{where}.aliases.{c.lang}",
                                      f"an alias belongs to more than one option ({', '.join(c.options)})"))
    if conflicts:  # lenient: a conflicting alias matches no option (it can't select a wrong one)
        drop = {(c.lang, c.phrase) for c in conflicts}
        out = [OptionDef(value=o.value, dtmf=o.dtmf,
                         aliases={lang: tuple(a for a in phrases if (lang, normalize(a, lang)) not in drop)
                                  for lang, phrases in o.aliases.items()})
               for o in out]
    return tuple(out), errors


@dataclass(frozen=True, slots=True)
class _Phase2:
    mode: Mode | None
    confirm: ConfirmPolicy
    options: tuple[OptionDef, ...]


def _settings(name: str, decl: SlotDecl, is_factor: bool, flow: str | None, values: Mapping[str, bool],
              extra: _Phase2, declinable: bool) -> SlotSettings:
    return SlotSettings(name=name, decl=decl, is_factor=is_factor, flow=flow,
                        sensitive=values["sensitive"], keypad_only=values["keypad_only"],
                        barge_in=values["barge_in"], mode=extra.mode, confirm=extra.confirm,
                        options=extra.options, declinable=declinable)


def _phase2(item: str, raw: Mapping[str, Any], decl: SlotDecl, *, sensitive: bool,
            langs: Sequence[str] | None) -> tuple[_Phase2, list[SlotPolicyError]]:
    """``mode``, ``confirm`` and ``options``: the lenient values and the errors, in field order."""
    choice = decl.slot_type == "choice"
    mode, errors = _mode_field(item, raw, choice=choice, sensitive=sensitive)
    confirm, confirm_errors = _confirm_field(item, raw)
    errors.extend(confirm_errors)
    options: tuple[OptionDef, ...] = ()
    if "options" in raw:
        if choice:
            options, option_errors = _options_field(item, raw["options"], langs=langs)
            errors.extend(option_errors)
        else:
            errors.append(SlotPolicyError(f"{item}.options", _MSG_OPTIONS_TYPE))
    return _Phase2(mode, confirm, options), errors


def parse_slot(name: str, raw: Mapping[str, Any], *,
               factor_def: Mapping[str, Any] | None,
               langs: Sequence[str] | None = None) -> SlotSettings:
    """Strict (seed file, console): :class:`SlotPolicyError` naming the first bad field.

    ``factor_def`` is the slot's ``factors`` entry, or ``None`` for a non-factor slot. A factor's
    item carries only the settings, ``mode`` and ``confirm``; shape fields, ``flow`` and
    ``options`` are refused on it. Unknown fields are refused. ``langs``, when given, are the
    languages an alias may be in (the catalog's); aliases are checked for conflicts in each of
    them (else in every language they use). Spoken names are checked by the caller with
    :func:`alias_conflicts`."""
    _check_name(name)
    item = _item(name)
    if not isinstance(raw, Mapping):
        raise SlotPolicyError(item, "must be an object")
    is_factor = factor_def is not None
    allowed = _allowed(is_factor)
    for k in raw:
        if k not in allowed:
            if is_factor and k in (*SHAPE_FIELDS, "flow"):
                why = "set in the factor's entry, not the slot item"
            elif k == "options":
                why = _MSG_OPTIONS_TYPE
            else:
                why = "unknown field"
            raise SlotPolicyError(f"{item}.{k}", why)
    values: dict[str, bool] = {}
    for f in SETTING_FIELDS:
        v = raw.get(f, SETTING_DEFAULTS[f])
        if not isinstance(v, bool):
            raise SlotPolicyError(f"{item}.{f}", _MSG_BOOL)
        values[f] = v
    declinable = raw.get(DECLINABLE_FIELD, False)
    if not isinstance(declinable, bool):
        raise SlotPolicyError(f"{item}.{DECLINABLE_FIELD}", _MSG_BOOL)
    flow = raw.get("flow")
    if flow is not None and flow not in FLOWS:
        raise SlotPolicyError(f"{item}.flow", f"must be one of {list(FLOWS)}")
    if factor_def is not None:
        if not isinstance(factor_def, Mapping):
            raise SlotPolicyError(f"factors.{name}", "must be an object")
        decl = _decl(name, factor_def, where=f"factors.{name}")
    else:
        decl = _decl(name, raw, where=item)
    extra, errors = _phase2(item, raw, decl, sensitive=values["sensitive"], langs=langs)
    if errors:
        raise errors[0]
    return _settings(name, decl, is_factor, flow, values, extra, declinable)


def load_slot(name: str, raw: Mapping[str, Any] | None, *,
              factor_def: Mapping[str, Any] | None,
              langs: Sequence[str] | None = None,
              ) -> tuple[SlotSettings | None, list[SlotPolicyError]]:
    """Lenient (the policy table at run time): never raises.

    * A non-boolean ``sensitive``, ``keypad_only`` or ``barge_in`` loads as ``true`` with
      one error naming the field. An absent one takes its default (``sensitive`` false).
    * A ``factors`` entry with no item (``raw is None``) loads ``sensitive: true`` and the
      other defaults, with one error naming the item. A non-object item loads all three
      fail-closed.
    * A non-boolean ``declinable`` loads as ``false`` with an error naming the field.
    * An unknown or misplaced field is ignored with an error; an invalid ``flow`` loads as none.
    * A bad ``mode`` loads as ``deterministic`` with an error whose ``mode_invalid``
      is set; ``generative`` on a sensitive or non-choice slot loads as ``assisted``; a bad
      ``confirm`` loads as ``always``; a bad option, a bad alias and every alias that two options
      share are dropped; ``options`` on a non-choice slot is ignored. Each with an error.
    * The settings are ``None`` only when no shape can be built: a non-factor slot with no item
      or an invalid shape, or an invalid factor definition. The caller leaves that slot out."""
    errors: list[SlotPolicyError] = []
    item = _item(name)
    is_factor = factor_def is not None
    if raw is None:
        errors.append(SlotPolicyError(item, "no item; loaded fail closed (sensitive)" if is_factor
                                      else "no item; the slot can't be loaded"))
        raw, values = {}, {**SETTING_DEFAULTS, "sensitive": True}
    elif not isinstance(raw, Mapping):
        errors.append(SlotPolicyError(item, "must be an object; loaded fail closed"))
        raw, values = {}, dict(FAIL_CLOSED)
    else:
        values = {}
        for f in SETTING_FIELDS:
            v = raw.get(f, SETTING_DEFAULTS[f])
            if isinstance(v, bool):
                values[f] = v
            else:
                values[f] = FAIL_CLOSED[f]
                errors.append(SlotPolicyError(f"{item}.{f}", _MSG_BOOL_LOADED))
    declinable = raw.get(DECLINABLE_FIELD, False)
    if not isinstance(declinable, bool):
        errors.append(SlotPolicyError(f"{item}.{DECLINABLE_FIELD}", _MSG_DECLINABLE_LOADED))
        declinable = False
    allowed = _allowed(is_factor)
    errors.extend(SlotPolicyError(f"{item}.{k}", "unknown or misplaced field; ignored")
                  for k in raw if k not in allowed)
    flow = raw.get("flow") if not is_factor else None
    if flow is not None and flow not in FLOWS:
        errors.append(SlotPolicyError(f"{item}.flow", "invalid; loaded as none"))
        flow = None
    if not is_factor and not raw:
        return None, errors  # a non-factor slot with no item has no shape
    try:
        if factor_def is not None:
            if not isinstance(factor_def, Mapping):
                raise SlotPolicyError(f"factors.{name}", "must be an object")
            decl = _decl(name, factor_def, where=f"factors.{name}")
        else:
            decl = _decl(name, raw, where=item)
    except SlotPolicyError as exc:
        errors.append(SlotPolicyError(exc.field, f"{exc.message}; the slot can't be loaded"))
        return None, errors
    phase2_raw = {k: v for k, v in raw.items() if k in allowed}
    extra, phase2_errors = _phase2(item, phase2_raw, decl, sensitive=values["sensitive"], langs=langs)
    errors.extend(SlotPolicyError(e.field, f"{e.message}; {_loaded_as(e)}", mode_invalid=e.mode_invalid)
                  for e in phase2_errors)
    return _settings(name, decl, is_factor, flow, values, extra, declinable), errors


def _loaded_as(e: SlotPolicyError) -> str:
    """How :func:`load_slot` words what an invalid phase 2 field loaded as."""
    if e.mode_invalid:
        return "loaded as deterministic"
    if e.message == _MSG_GENERATIVE:
        return "loaded as assisted"
    if e.field.endswith(".confirm"):
        return f"loaded as {CONFIRM_FAIL_CLOSED}"
    return "ignored"


# --------------------------------------------------------------------------- slot_resolution

RESOLUTION_ITEM = "slot_resolution"


def _resolution(raw: Any, *, strict: bool = False) -> tuple[ResolutionSettings, list[SlotPolicyError]]:
    """The lenient ``slot_resolution`` and every error in field order (one rule set for both
    modes). A missing item or ``default_mode`` loads ``deterministic``. The messages say what
    a field loaded as only when ``strict`` is false: a strict parse refuses the item instead."""
    item = RESOLUTION_ITEM

    def loaded(msg: str, value: object) -> str:
        return msg if strict else f"{msg}; loaded as {value}"

    if raw is None:
        return (ResolutionSettings(default_mode="deterministic"),
                [SlotPolicyError(f"{item}.default_mode", loaded("no item", "deterministic"), mode_invalid=True)])
    if not isinstance(raw, Mapping):
        return (ResolutionSettings(default_mode="deterministic"),
                [SlotPolicyError(f"{item}.default_mode", loaded("the item must be an object", "deterministic"),
                                 mode_invalid=True)])
    errors = [SlotPolicyError(f"{item}.{k}" if isinstance(k, str) and k.isidentifier() else item,
                              "unknown field") for k in raw if k not in RESOLUTION_FIELDS]
    raw_default = raw.get("default_mode")
    default: Mode = "deterministic"
    if is_mode(raw_default):
        default = raw_default
    else:
        errors.append(SlotPolicyError(f"{item}.default_mode", _MSG_MODE, mode_invalid=True))
    flows: dict[str, Mode] = {}
    raw_flows = raw.get("flows", {})
    if raw_flows is None:
        raw_flows = {}
    if not isinstance(raw_flows, Mapping):
        errors.append(SlotPolicyError(f"{item}.flows", "must be an object of flow -> {mode}"))
        raw_flows = {}
    for f, entry in raw_flows.items():
        if f not in FLOWS:
            errors.append(SlotPolicyError(f"{item}.flows", f"keys must be among {list(FLOWS)}"))
            continue
        here = f"{item}.flows.{f}"
        if not isinstance(entry, Mapping) or set(entry) != {"mode"}:
            errors.append(SlotPolicyError(f"{here}.mode", loaded('must be {"mode": <mode>}', "deterministic"),
                                          mode_invalid=True))
            flows[f] = "deterministic"
            continue
        m = entry["mode"]
        if is_mode(m):
            flows[f] = m
        else:
            errors.append(SlotPolicyError(f"{here}.mode", _MSG_MODE, mode_invalid=True))
            flows[f] = "deterministic"
    gen = raw.get("generative_enabled", False)
    if not isinstance(gen, bool):
        errors.append(SlotPolicyError(f"{item}.generative_enabled", loaded("must be a boolean", "false")))
        gen = False
    budget = raw.get("resolver_budget_ms", DEFAULT_RESOLVER_BUDGET_MS)
    lo, hi = RESOLVER_BUDGET_MS_RANGE
    if not isinstance(budget, int) or isinstance(budget, bool) or not lo <= budget <= hi:
        errors.append(SlotPolicyError(f"{item}.resolver_budget_ms",
                                      loaded(f"must be an integer in {lo}..{hi}", DEFAULT_RESOLVER_BUDGET_MS)))
        budget = DEFAULT_RESOLVER_BUDGET_MS
    return ResolutionSettings(default_mode=default, flows={f: flows[f] for f in FLOWS if f in flows},
                              generative_enabled=gen, resolver_budget_ms=budget), errors


def parse_slot_resolution(raw: Any) -> ResolutionSettings:
    """Strict (seed file, console): :class:`SlotPolicyError` naming the first bad field. An absent
    field takes its default (``default_mode`` ``assisted``, no flow overrides,
    ``generative_enabled`` false, ``resolver_budget_ms`` 2500); ``flows`` keys must be known
    flows, each ``{"mode": <mode>}``."""
    if not isinstance(raw, Mapping):
        raise SlotPolicyError(RESOLUTION_ITEM, "must be an object")
    if "default_mode" not in raw:
        raw = {**raw, "default_mode": DEFAULT_MODE}
    settings, errors = _resolution(raw, strict=True)
    if errors:
        raise errors[0]
    return settings


def load_slot_resolution(raw: Any) -> tuple[ResolutionSettings, list[SlotPolicyError]]:
    """Lenient (the policy table at run time): never raises. A missing item, or a missing or
    invalid ``default_mode`` or flow mode, loads ``deterministic`` with an error whose
    ``mode_invalid`` is set; an unknown flow key is dropped; a bad ``generative_enabled`` loads
    ``false``; a bad budget loads 2500. Each problem is one error naming the field, never a value."""
    return _resolution(raw)
