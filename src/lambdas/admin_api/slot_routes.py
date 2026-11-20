"""Slot routes: ``GET /slots``, ``PUT /slots/{name}``, ``PUT /slots/resolution`` and
``POST /slots/preview`` (phases 1 and 2).

**Same rules as the front door**. ``GET`` loads each slot with
:func:`owlbank.slot_policy.load_slot` over the same items the front door reads (the ``factors``
entries, then ``meta.slots``; a factor with no ``slot#`` item loads sensitive; a non-boolean
setting loads as ``true``, fail closed), and ``slot_resolution`` with
:func:`~owlbank.slot_policy.load_slot_resolution`. It reports the effective keypad-only from
:func:`~owlbank.slot_policy.gather_input` with the deployment's ``acceptSpokenFactors``, and each
slot's configured and effective mode per flow from :func:`owlbank.slots.modes.mode_decision` with
the deployment's ceiling (``OWLBANK_SLOT_MODE_CEILING``), the front door's own function. A ``PUT``
is checked by :func:`~owlbank.slot_policy.parse_slot` or
:func:`~owlbank.slot_policy.parse_slot_resolution` (strict) over the whole new item, plus, for a
choice slot, :func:`~owlbank.slot_policy.alias_conflicts` against the catalog's spoken names and
the rule that every ``option.<slot>.<value>`` key has a text in every catalog language.

**Writes** are one transaction of the item (``rev + 1``, conditional on the ``rev`` read) and its
audit entry, as the tool routes do (:mod:`.config_routes`). ``PUT /slots/{name}`` accepts
``sensitive``, ``keypad_only``, ``barge_in``, ``mode`` (``null`` = inherit), ``confirm`` and
``options`` (``null`` = none); a setting absent from the stored item is written with the value it
loaded as, so a write never changes a setting it wasn't sent. ``PUT /slots/resolution`` accepts
``default_mode``, ``flows``, ``resolver_budget_ms`` and ``generative_enabled``; the API refuses
``generative_enabled`` from anyone but an ``admin`` (:mod:`.app`), and its audit entry
records who set it and when, which ``GET`` shows. :data:`.audit.SECURITY_FIELDS` says which
changes are security changes.

``POST /slots/preview`` takes a proposed ``slot_resolution`` and per-slot changes and answers
which flows and slots would then call a model, given the ceiling. It writes nothing.

**Warnings** (each ``{"code", "slot", "message"}``; ``slot`` is the slot name or
``slot_resolution``): ``sensitive_off`` and ``keypad_only_off`` when a write turns
the setting from true to false; ``verification_not_sensitive`` for a ``factors`` entry that
isn't sensitive, on every view of it; ``away_from_deterministic`` when a write or preview
moves a configured mode off ``deterministic``; ``generative_unvalidated_wording`` when a
write turns ``generative_enabled`` on or sets a mode to ``generative``; ``generative_no_effect``
 on a view where ``generative`` is enabled or configured but the ceiling isn't
``generative``. Neither a response nor a log line carries anything but setting names, modes,
booleans, option values, configured catalog texts and slot names.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from owlbank import catalog as catalog_mod
from owlbank import obs
from owlbank import policy as policy_mod
from owlbank.slot_policy import (
    FLOWS,
    PHASE2_FIELDS,
    RESOLUTION_FIELDS,
    RESOLUTION_ITEM,
    SETTING_FIELDS,
    Mode,
    ResolutionSettings,
    SlotPolicyError,
    SlotSettings,
    alias_conflicts,
    gather_input,
    load_slot,
    load_slot_resolution,
    parse_slot,
    parse_slot_resolution,
)
from owlbank.slots.modes import INVALID, ModeDecision, mode_decision, slot_flows
from owlbank.store import Expect, Store, TransactionCancelled, TxPut

from . import audit
from .audit import AuditClock
from .config import TAKES_EFFECT_S
from .config_routes import OPTION_PREFIX, _only_keys, _rev, _rev_of_body, spoken_names
from .http import ApiError, Caller

log = logging.getLogger(__name__)

REV = policy_mod.REV_ATTR
_SLOT_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
#: Every field ``PUT /slots/{name}`` may send (besides ``rev``) and the audit diffs.
SLOT_FIELDS: tuple[str, ...] = (*SETTING_FIELDS, *PHASE2_FIELDS)
#: Fields whose ``null`` removes them from the item (inherit the mode; no options).
_NULLABLE: tuple[str, ...] = ("mode", "options")

#: The warning texts.
WARNINGS: Mapping[str, str] = {
    "sensitive_off": (
        "Turning sensitive off is a security change: this slot's value may then be written to "
        "logs and the call journal, read back to the caller and passed to the AI agent."),
    "keypad_only_off": (
        "Turning keypad_only off is a security change: callers may then speak this answer, so "
        "people nearby can hear it and the words pass through Twilio speech recognition."),
    "verification_not_sensitive": (
        "This slot is a verification factor and isn't sensitive: its value may be logged, "
        "journaled, read back and passed to the AI agent. Production deployments should flag "
        "every verification factor as sensitive; the seed leaves ZIP unflagged for demos only."),
    "away_from_deterministic": (
        "Moving away from deterministic is a security change: callers' answers in the affected "
        "flows may then be sent to a model on Amazon Bedrock, and the answer it proposes is "
        "checked by code before it is used. Check the preview for which slots this affects."),
    "generative_unvalidated_wording": (
        "Generative mode lets the model word some follow-up questions (confirmations, "
        "disambiguations, not-available questions) on non-sensitive choice slots. That wording is "
        "checked structurally but is no longer validated catalog text."),
    "generative_no_effect": (
        "Generative mode has no effect in this deployment: the deployment's mode ceiling isn't "
        "generative, so these slots run assisted at most."),
}


def warning(code: str, slot: str) -> dict[str, str]:
    return {"code": code, "slot": slot, "message": WARNINGS[code]}


def _raw(item: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return None if item is None else {k: v for k, v in item.items() if k not in ("pk", REV)}


def _bad_modes(problems: Iterable[SlotPolicyError]) -> frozenset[str]:
    return frozenset(e.field for e in problems if e.mode_invalid)


@dataclass(frozen=True, slots=True)
class _Resolution:
    """``slot_resolution`` as the front door loads it, with its item and problems."""

    item: dict[str, Any] | None
    settings: ResolutionSettings
    problems: tuple[SlotPolicyError, ...]

    @property
    def bad(self) -> frozenset[str]:
        return _bad_modes(self.problems)


def decisions(s: SlotSettings, slot_problems: Iterable[SlotPolicyError], res: ResolutionSettings,
              res_problems: Iterable[SlotPolicyError], ceiling: Mode) -> dict[str, ModeDecision]:
    """The slot's decision in each flow it can be asked in, exactly as
    :func:`owlbank.slots.modes.freeze` computes them over the policy the front door loads:
    a mode field the lenient loader found invalid is passed as :data:`~owlbank.slots.modes.INVALID`."""
    bad = _bad_modes(slot_problems) | _bad_modes(res_problems)
    raw_mode: object = INVALID if f"slot#{s.name}.mode" in bad else s.mode
    raw_default: object = INVALID if f"{RESOLUTION_ITEM}.default_mode" in bad else res.default_mode
    out: dict[str, ModeDecision] = {}
    for flow in slot_flows(s):
        raw_flow: object = INVALID if f"{RESOLUTION_ITEM}.flows.{flow}.mode" in bad else res.flows.get(flow)
        out[flow] = mode_decision(s, res, flow, ceiling, raw_mode=raw_mode, raw_flow_mode=raw_flow,
                                  raw_default=raw_default)
    return out


def _flow_configured(r: ResolutionSettings) -> dict[str, Mode]:
    """Each flow's configured mode before slot overrides: its override, else the default."""
    return {f: r.flows.get(f, r.default_mode) for f in FLOWS}


def _resolution_state_warnings(r: ResolutionSettings, ceiling: Mode) -> list[dict[str, str]]:
    configured = (r.default_mode, *r.flows.values())
    if ceiling != "generative" and (r.generative_enabled or "generative" in configured):
        return [warning("generative_no_effect", RESOLUTION_ITEM)]
    return []


class SlotRoutes:
    def __init__(self, store: Store, *, clock: AuditClock, retention_days: int,
                 accept_spoken_factors: bool, slot_mode_ceiling: Mode = "assisted") -> None:
        self.store, self.clock, self.retention_days = store, clock, retention_days
        self.accept_spoken_factors = accept_spoken_factors
        self.ceiling: Mode = slot_mode_ceiling

    # ------------------------------------------------------------------ loading

    def _meta(self) -> dict[str, Any]:
        meta = self.store.get("policy", {"pk": policy_mod.META_PK})
        if meta is None:
            raise ApiError(503, "not_seeded", "the policy table has no meta item")
        return meta

    @staticmethod
    def _factors(meta: Mapping[str, Any]) -> Mapping[str, Any]:
        factors = meta.get("factors")
        return factors if isinstance(factors, Mapping) else {}

    @staticmethod
    def _listed(meta: Mapping[str, Any]) -> list[str]:
        """``meta.slots`` as the front door reads it (a malformed list counts as empty)."""
        listed = meta.get("slots")
        if not isinstance(listed, list) or not all(isinstance(n, str) and n for n in listed):
            return []
        return list(dict.fromkeys(listed))

    def _names(self, meta: Mapping[str, Any]) -> list[str]:
        """Every slot the front door has: the ``factors`` entries, then the other listed slots."""
        factors = self._factors(meta)
        return [*factors, *(n for n in self._listed(meta) if n not in factors)]

    def _item(self, name: str, meta: Mapping[str, Any]) -> dict[str, Any] | None:
        """The ``slot#<name>`` item the front door reads (none unless ``meta.slots`` lists it)."""
        if name not in self._listed(meta):
            return None
        return self.store.get("policy", {"pk": f"slot#{name}"})

    def _load(self, name: str, item: Mapping[str, Any] | None, meta: Mapping[str, Any]
              ) -> tuple[SlotSettings | None, list[SlotPolicyError]]:
        return load_slot(name, _raw(item), factor_def=self._factors(meta).get(name))

    def _resolution(self) -> _Resolution:
        item = self.store.get("policy", {"pk": RESOLUTION_ITEM})
        settings, problems = load_slot_resolution(_raw(item))
        return _Resolution(item, settings, tuple(problems))

    def _catalog_meta(self) -> dict[str, Any] | None:
        return self.store.get("catalog", {"key": catalog_mod.META_KEY, "lang": catalog_mod.META_LANG})

    @staticmethod
    def _langs(cmeta: Mapping[str, Any] | None) -> tuple[str, ...]:
        langs = (cmeta or {}).get("languages")
        return tuple(k for k in langs if isinstance(k, str)) if isinstance(langs, Mapping) else ()

    def spoken_names(self, slot: str, values: Iterable[str], cmeta: Mapping[str, Any] | None,
                     override: Mapping[tuple[str, str], str] | None = None) -> dict[str, dict[str, str]]:
        return spoken_names(self.store, slot, values, cmeta, override)

    # ------------------------------------------------------------------ views

    def _options_view(self, settings: SlotSettings, cmeta: Mapping[str, Any] | None
                      ) -> list[dict[str, Any]] | None:
        if settings.decl.slot_type != "choice":
            return None
        langs = self._langs(cmeta)
        names = self.spoken_names(settings.name, [o.value for o in settings.options], cmeta)
        return [{"value": o.value, "dtmf": o.dtmf,
                 "spoken_names": {lang: names[o.value].get(lang) for lang in langs},
                 "aliases": {lang: list(a) for lang, a in o.aliases.items()}}
                for o in settings.options]

    def _view(self, name: str, settings: SlotSettings | None, problems: list[SlotPolicyError],
              rev: int, res: _Resolution, cmeta: Mapping[str, Any] | None) -> dict[str, Any]:
        view: dict[str, Any] = {
            "name": name, "rev": rev, "loaded": settings is not None,
            "problems": [str(p) for p in problems],
            "type": None, "sensitive": None, "keypad_only": None, "barge_in": None,
            "maxRetries": None, "is_factor": None, "effective_keypad_only": None, "flow": None,
            "mode": None, "confirm": None, "options": None, "modes": {},
            "warnings": [],
        }
        if settings is None:
            return view
        g = gather_input(settings, self.accept_spoken_factors)
        modes = decisions(settings, problems, res.settings, res.problems, self.ceiling)
        view.update(type=settings.decl.slot_type, sensitive=settings.sensitive,
                    keypad_only=settings.keypad_only, barge_in=g.barge_in,
                    maxRetries=settings.decl.max_retries, is_factor=settings.is_factor,
                    effective_keypad_only=g.keypad_only, flow=settings.flow,
                    mode=settings.mode, confirm=settings.confirm,
                    options=self._options_view(settings, cmeta),
                    modes={f: d.to_doc() for f, d in modes.items()},
                    warnings=self._state_warnings(settings))
        return view

    def _state_warnings(self, settings: SlotSettings) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        if settings.is_factor and not settings.sensitive:
            out.append(warning("verification_not_sensitive", settings.name))
        if settings.mode == "generative" and self.ceiling != "generative":
            out.append(warning("generative_no_effect", settings.name))
        return out

    def _current_view(self, name: str, meta: Mapping[str, Any]) -> dict[str, Any]:
        item = self._item(name, meta)
        settings, problems = self._load(name, item, meta)
        return self._view(name, settings, problems, _rev(item), self._resolution(), self._catalog_meta())

    def _last_generative_set(self) -> dict[str, Any] | None:
        """Who last changed ``generative_enabled`` and when, from the audit log."""
        entries = sorted(self.store.query(audit.AUDIT_TABLE, {"pk": audit.RESOLUTION_PK}),
                         key=lambda e: e.get("sk", ""), reverse=True)
        for e in entries:
            for c in e.get("changes") or []:
                if isinstance(c, Mapping) and c.get("field") == "generative_enabled":
                    return {"value": c.get("new"), "by": e.get("actor"), "at": e.get("ts")}
        return None

    def _resolution_view(self, res: _Resolution, rev: int | None = None) -> dict[str, Any]:
        r = res.settings
        return {"rev": _rev(res.item) if rev is None else rev, "default_mode": r.default_mode,
                "flows": dict(r.flows), "generative_enabled": r.generative_enabled,
                "resolver_budget_ms": r.resolver_budget_ms,
                "problems": [str(p) for p in res.problems],
                "generative_enabled_set": self._last_generative_set(),
                "warnings": _resolution_state_warnings(r, self.ceiling)}

    def list_slots(self) -> dict[str, Any]:
        meta = self._meta()
        res, cmeta = self._resolution(), self._catalog_meta()
        slots = []
        for name in self._names(meta):
            item = self._item(name, meta)
            settings, problems = self._load(name, item, meta)
            slots.append(self._view(name, settings, problems, _rev(item), res, cmeta))
        return {"acceptSpokenFactors": self.accept_spoken_factors, "ceiling": self.ceiling,
                "languages": list(self._langs(cmeta)),
                "slot_resolution": self._resolution_view(res),
                "slots": slots, "takes_effect_s": TAKES_EFFECT_S}

    # ------------------------------------------------------------------ validation

    def _check_options(self, name: str, new: SlotSettings, cmeta: Mapping[str, Any] | None,
                       override: Mapping[tuple[str, str], str] | None = None) -> None:
        """Every option has a spoken name in every catalog language, and no spoken name or alias
        belongs to two options in one language."""
        if new.decl.slot_type != "choice" or not new.options:
            return
        if cmeta is None:
            raise ApiError(503, "not_seeded", "the catalog table has no meta item")
        langs = self._langs(cmeta)
        names = self.spoken_names(name, [o.value for o in new.options], cmeta, override)
        for i, o in enumerate(new.options):
            for lang in langs:
                if lang not in names[o.value]:
                    raise ApiError(400, "invalid",
                                   f"the catalog has no {OPTION_PREFIX}{name}.{o.value} text in "
                                   f"{lang}; add it first (PUT /prompts/{OPTION_PREFIX}{name}.{o.value})",
                                   field=f"slot#{name}.options[{i}].value")
        conflicts = alias_conflicts(new.options, names, langs)
        if conflicts:
            c = conflicts[0]
            raise ApiError(400, "invalid", "a spoken name or alias belongs to more than one option "
                                           f"({', '.join(c.options)})",
                           field=f"slot#{name}.options.aliases.{c.lang}")

    def _proposed_slot(self, name: str, meta: Mapping[str, Any], item: Mapping[str, Any] | None,
                       updates: Mapping[str, Any], langs: Sequence[str] | None
                       ) -> tuple[SlotSettings | None, list[SlotPolicyError], dict[str, Any],
                                  dict[str, Any], SlotSettings]:
        """``(loaded, problems, old_cmp, new_raw, new)``: the slot as loaded and its problems, the
        old item with absent settings filled with the values they loaded as, the new item, and the
        new item parsed strictly."""
        factor_def = self._factors(meta).get(name)
        old_raw = _raw(item) or {}
        loaded, problems = self._load(name, item, meta)
        # A setting the item doesn't hold is written as the value it loaded as (sensitive
        # for a factor with no item), so the write changes only what was sent.
        filled = {f: getattr(loaded, f) for f in SETTING_FIELDS if f not in old_raw} if loaded else {}
        old_cmp = {**filled, **old_raw}
        new_raw = {**old_cmp, **updates}
        for f in _NULLABLE:
            if f in new_raw and new_raw[f] is None:
                del new_raw[f]
        try:
            new = parse_slot(name, new_raw, factor_def=factor_def, langs=langs or None)
        except SlotPolicyError as exc:
            raise ApiError(400, "invalid", exc.message, field=exc.field) from None
        return loaded, problems, old_cmp, new_raw, new

    def _slot_target(self, name: str, meta: Mapping[str, Any]) -> dict[str, Any] | None:
        """The stored item of a slot that can be written (404 / 503 otherwise)."""
        if not _SLOT_NAME.fullmatch(name) or name not in self._names(meta):
            raise ApiError(404, "not_found", "no such slot")
        factor_def = self._factors(meta).get(name)
        if name not in self._listed(meta):  # the front door wouldn't read the item
            raise ApiError(503, "not_seeded", "meta.slots doesn't list this slot; reseed the policy table")
        item = self.store.get("policy", {"pk": f"slot#{name}"})
        if item is None and factor_def is None:
            raise ApiError(404, "not_found", "no such slot")
        return item

    @staticmethod
    def _proposed_resolution(item: Mapping[str, Any] | None, updates: Mapping[str, Any]
                             ) -> tuple[ResolutionSettings, dict[str, Any], ResolutionSettings]:
        """``(loaded, old_cmp, new)``; absent fields compare as the values they loaded as (a
        missing item or ``default_mode`` loads ``deterministic``)."""
        old_raw = _raw(item) or {}
        loaded, _ = load_slot_resolution(_raw(item))
        filled = {f: v for f, v in loaded.to_doc().items() if f not in old_raw}
        old_cmp = {**filled, **old_raw}
        try:
            new = parse_slot_resolution({**old_cmp, **updates})
        except SlotPolicyError as exc:
            raise ApiError(400, "invalid", exc.message, field=exc.field) from None
        return loaded, old_cmp, new

    # ------------------------------------------------------------------ warnings on a change

    @staticmethod
    def _away(before: Mapping[str, ModeDecision], after: Mapping[str, ModeDecision]) -> bool:
        return any(d.configured == "deterministic" and after[f].configured != "deterministic"
                   for f, d in before.items() if f in after)

    # ------------------------------------------------------------------ PUT /slots/{name}

    def put_slot(self, caller: Caller, name: str, body: Mapping[str, Any]) -> dict[str, Any]:
        _only_keys(body, {"rev", *SLOT_FIELDS})
        rev = _rev_of_body(body)
        updates = {f: body[f] for f in SLOT_FIELDS if f in body}
        if not updates:
            raise ApiError(400, "invalid", f"nothing to update: send any of {list(SLOT_FIELDS)}")
        meta = self._meta()
        item = self._slot_target(name, meta)
        if _rev(item) != rev:
            raise self._conflict(name, meta)
        cmeta = self._catalog_meta()
        loaded, loaded_problems, old_cmp, new_raw, new = self._proposed_slot(
            name, meta, item, updates, self._langs(cmeta))
        self._check_options(name, new, cmeta)
        res = self._resolution()

        changes = audit.diff(old_cmp, new_raw, SLOT_FIELDS)
        if not changes:
            return {"changed": False, "slot": self._view(name, new, [], rev, res, cmeta),
                    "warnings": self._state_warnings(new)}
        security_change = any(c["field"] in audit.SECURITY_FIELDS for c in changes)
        new_rev = rev + 1
        new_item = {"pk": f"slot#{name}", **new_raw, REV: new_rev}
        entry = audit.entry(pk=audit.slot_pk(name), action="update", kind="slot", target_id=name,
                            actor=caller, changes=changes, security_change=security_change,
                            now=self.clock.now(), suffix=self.clock.suffix(),
                            retention_days=self.retention_days, rev=new_rev)
        try:
            self.store.transact_write([TxPut("policy", new_item, Expect(REV, rev, absent_as=0)),
                                       TxPut(audit.AUDIT_TABLE, entry, "not_exists")])
        except TransactionCancelled:
            raise self._conflict(name, meta) from None
        obs.emit(log, logging.INFO, "admin_config_write", action="update", target_kind="slot",
                 target_id=name, actor_sub=caller.sub, rev=new_rev,
                 security_change=security_change, fields=[c["field"] for c in changes])

        warnings: list[dict[str, str]] = []
        if loaded is not None:
            for f, code in (("sensitive", "sensitive_off"), ("keypad_only", "keypad_only_off")):
                if getattr(loaded, f) and not getattr(new, f):
                    warnings.append(warning(code, name))
            before = decisions(loaded, loaded_problems, res.settings, res.problems, self.ceiling)
            after = decisions(new, [], res.settings, res.problems, self.ceiling)
            if self._away(before, after):
                warnings.append(warning("away_from_deterministic", name))
            if new.mode == "generative" and loaded.mode != "generative":
                warnings.append(warning("generative_unvalidated_wording", name))
        warnings += self._state_warnings(new)
        return {"changed": True, "audit": {"pk": entry["pk"], "sk": entry["sk"]},
                "takes_effect_s": TAKES_EFFECT_S, "security_change": security_change,
                "slot": self._view(name, new, [], new_rev, res, cmeta), "warnings": warnings}

    def _conflict(self, name: str, meta: Mapping[str, Any]) -> ApiError:
        return ApiError(409, "conflict", "the slot changed since it was read",
                        extra={"current": self._current_view(name, meta)})

    # ------------------------------------------------------------------ PUT /slots/resolution

    def put_resolution(self, caller: Caller, body: Mapping[str, Any]) -> dict[str, Any]:
        """``generative_enabled`` is admin only; :mod:`.app` refuses it from anyone else first."""
        _only_keys(body, {"rev", *RESOLUTION_FIELDS})
        rev = _rev_of_body(body)
        updates = {f: body[f] for f in RESOLUTION_FIELDS if f in body}
        if not updates:
            raise ApiError(400, "invalid", f"nothing to update: send any of {list(RESOLUTION_FIELDS)}")
        self._meta()
        res = self._resolution()
        if _rev(res.item) != rev:
            raise self._resolution_conflict()
        loaded, old_cmp, new = self._proposed_resolution(res.item, updates)
        new_doc = new.to_doc()
        changes = audit.diff(old_cmp, new_doc, RESOLUTION_FIELDS)
        new_res = _Resolution(new_doc, new, ())
        if not changes:
            return {"changed": False, "slot_resolution": self._resolution_view(new_res, rev),
                    "warnings": _resolution_state_warnings(new, self.ceiling)}
        security_change = any(c["field"] in audit.SECURITY_FIELDS for c in changes)
        new_rev = rev + 1
        new_item = {"pk": RESOLUTION_ITEM, **new_doc, REV: new_rev}
        entry = audit.entry(pk=audit.RESOLUTION_PK, action="update", kind="slot_resolution",
                            target_id=RESOLUTION_ITEM, actor=caller, changes=changes,
                            security_change=security_change, now=self.clock.now(),
                            suffix=self.clock.suffix(), retention_days=self.retention_days,
                            rev=new_rev)
        try:
            self.store.transact_write([TxPut("policy", new_item, Expect(REV, rev, absent_as=0)),
                                       TxPut(audit.AUDIT_TABLE, entry, "not_exists")])
        except TransactionCancelled:
            raise self._resolution_conflict() from None
        obs.emit(log, logging.INFO, "admin_config_write", action="update", target_kind="slot_resolution",
                 target_id=RESOLUTION_ITEM, actor_sub=caller.sub, rev=new_rev,
                 security_change=security_change, fields=[c["field"] for c in changes])

        warnings: list[dict[str, str]] = []
        before, after = _flow_configured(loaded), _flow_configured(new)
        if any(before[f] == "deterministic" and after[f] != "deterministic" for f in FLOWS):
            warnings.append(warning("away_from_deterministic", RESOLUTION_ITEM))
        turned_on = new.generative_enabled and not loaded.generative_enabled
        if turned_on or any(after[f] == "generative" and before[f] != "generative" for f in FLOWS):
            warnings.append(warning("generative_unvalidated_wording", RESOLUTION_ITEM))
        warnings += _resolution_state_warnings(new, self.ceiling)
        return {"changed": True, "audit": {"pk": entry["pk"], "sk": entry["sk"]},
                "takes_effect_s": TAKES_EFFECT_S, "security_change": security_change,
                "slot_resolution": self._resolution_view(_Resolution(new_item, new, ()), new_rev),
                "warnings": warnings}

    def _resolution_conflict(self) -> ApiError:
        return ApiError(409, "conflict", "slot_resolution changed since it was read",
                        extra={"current": self._resolution_view(self._resolution())})

    # ------------------------------------------------------------------ POST /slots/preview

    def preview(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """For a proposed ``slot_resolution`` (any of its fields) and per-slot changes (any of
        :data:`SLOT_FIELDS`), each slot's decisions per flow now and after, and which slots and
        flows would call a model (effective mode not ``deterministic``). Validated like the
        writes' parse; writes nothing."""
        _only_keys(body, {"slot_resolution", "slots"})
        raw_res = body.get("slot_resolution", {})
        raw_slots = body.get("slots", {})
        if not isinstance(raw_res, Mapping):
            raise ApiError(400, "invalid", "must be an object of slot_resolution fields",
                           field="slot_resolution")
        if not isinstance(raw_slots, Mapping):
            raise ApiError(400, "invalid", "must map slot names to their proposed fields", field="slots")
        unknown = sorted(str(k) for k in raw_res if k not in RESOLUTION_FIELDS)
        if unknown:
            raise ApiError(400, "invalid", "unknown field", field=f"slot_resolution.{unknown[0]}")
        meta = self._meta()
        res = self._resolution()
        cmeta = self._catalog_meta()
        langs = self._langs(cmeta)
        new_res, new_res_problems = res.settings, res.problems
        if raw_res:
            new_res, new_res_problems = self._proposed_resolution(res.item, raw_res)[2], ()
        names = self._names(meta)
        for n in raw_slots:
            if not isinstance(n, str) or n not in names:
                raise ApiError(400, "invalid", "no such slot", field=f"slots.{n}")
            fields = raw_slots[n]
            if not isinstance(fields, Mapping):
                raise ApiError(400, "invalid", "must be an object of slot fields", field=f"slots.{n}")
            bad = sorted(str(k) for k in fields if k not in SLOT_FIELDS)
            if bad:
                raise ApiError(400, "invalid", "unknown field", field=f"slots.{n}.{bad[0]}")

        current: dict[str, dict[str, Any]] = {}
        proposed: dict[str, dict[str, Any]] = {}
        calls_model: list[dict[str, str]] = []
        changes: list[dict[str, str]] = []
        away = False
        for name in names:
            item = self._item(name, meta)
            settings, problems = self._load(name, item, meta)
            if settings is None:
                continue
            now = decisions(settings, problems, res.settings, res.problems, self.ceiling)
            if name in raw_slots:
                new = self._proposed_slot(name, meta, item, raw_slots[name], langs)[-1]
                after = decisions(new, [], new_res, new_res_problems, self.ceiling)
            else:
                after = decisions(settings, problems, new_res, new_res_problems, self.ceiling)
            current[name] = {f: d.to_doc() for f, d in now.items()}
            proposed[name] = {f: d.to_doc() for f, d in after.items()}
            away = away or self._away(now, after)
            for f, d in after.items():
                if d.effective != "deterministic":
                    calls_model.append({"slot": name, "flow": f, "effective": d.effective})
                if d.effective != now[f].effective:
                    changes.append({"slot": name, "flow": f, "from": now[f].effective, "to": d.effective})
        warnings: list[dict[str, str]] = []
        if away:
            warnings.append(warning("away_from_deterministic", RESOLUTION_ITEM))
        warnings += _resolution_state_warnings(new_res, self.ceiling)
        return {"ceiling": self.ceiling, "current": current, "proposed": proposed,
                "calls_model": calls_model, "changes": changes, "warnings": warnings}
