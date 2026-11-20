"""Slot modes: which mode each slot runs in. Pure, apart from the log events.

* **Configured mode**: the first *present* value in the order slot ``mode``, then
  ``slot_resolution.flows[flow].mode``, then ``slot_resolution.default_mode``. A present value
  that isn't a mode, and a missing ``default_mode``, make the slot ``deterministic`` with source
  ``invalid`` (fail closed: no model call), never an exception.
* **Effective mode**: ``min(ceiling, configured)`` in :data:`MODES` order, and
  ``generative`` only with ``generative_enabled`` on a non-sensitive choice slot; otherwise
  ``assisted``. ``capped`` says why the effective mode is below the configured one.
* :func:`freeze` computes the decisions once at flow start; :func:`restrict` applies
  the running code's ceiling at each later use, so a frozen mode only gets stricter.

Log events (names and modes only):

* ``slot_modes`` (INFO) and one ``slot_mode_capped`` (INFO: ``scope``, ``target``, ``reason``,
  the modes) per capped default, flow or slot, from :func:`log_mode_table` once per cold start. The reason a
  generative mode is capped is checked in the order type, sensitive,
  ``generative_enabled``.
* ``slot_mode_invalid`` (ERROR, alarmed): ``item`` and ``field``, once per process per (item,
  field, policy version), from :func:`freeze` and :func:`log_mode_table`.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from owlbank import obs
from owlbank.slot_policy import (
    FLOWS,
    MODES,
    RESOLUTION_ITEM,
    CapReason,
    Mode,
    ModeSource,
    ResolutionSettings,
    SlotSettings,
    is_mode,
    mode_rank,
)

if TYPE_CHECKING:
    from owlbank.policy import Policy

log = logging.getLogger(__name__)

class _Invalid:
    __slots__ = ()

    def __repr__(self) -> str:
        return "INVALID"


#: A raw mode value that the lenient loader already found invalid (it loaded as ``deterministic``).
INVALID: Final = _Invalid()

#: The flows a ``factors`` slot is collected in (a non-factor slot names its own ``flow``).
FACTOR_FLOWS: tuple[str, ...] = ("login", "stepup")


@dataclass(frozen=True, slots=True)
class ModeDecision:
    """One slot's mode in one flow: what was configured, where it came from, the ceiling, the
    effective mode and why it is lower, if it is."""

    slot: str
    flow: str
    configured: Mode
    source: ModeSource
    ceiling: Mode
    effective: Mode
    capped: CapReason | None

    def to_doc(self) -> dict[str, Any]:
        """The session sidecar's ``flow["modes"][slot]`` and the journal's fields."""
        return {"configured": self.configured, "source": self.source, "ceiling": self.ceiling,
                "effective": self.effective, "capped": self.capped}


def _check_ceiling(ceiling: object) -> Mode:
    if not is_mode(ceiling):
        raise ValueError(f"the ceiling must be one of {list(MODES)}")  # config, validated at INIT
    return ceiling


def restrict(frozen: Mode, ceiling: Mode) -> Mode:
    """The mode to use now: the more restrictive of the frozen mode and the running ceiling."""
    if not is_mode(frozen):
        raise ValueError(f"the frozen mode must be one of {list(MODES)}")
    c = _check_ceiling(ceiling)
    return frozen if mode_rank(frozen) <= mode_rank(c) else c


def _configured(raw_mode: object, raw_flow_mode: object, raw_default: object
                ) -> tuple[Mode, ModeSource, str | None]:
    """The configured mode, its source and, when invalid, which field decided it (``slot``,
    ``flow`` or ``default``)."""
    candidates: tuple[tuple[object, ModeSource], ...] = ((raw_mode, "slot"), (raw_flow_mode, "flow"))
    for raw, source in candidates:
        if raw is None:
            continue
        if is_mode(raw):
            return raw, source, None
        return "deterministic", "invalid", source
    if is_mode(raw_default):
        return raw_default, "default", None
    return "deterministic", "invalid", "default"


def _decide(s: SlotSettings, r: ResolutionSettings, flow: str, ceiling: Mode, *, raw_mode: object,
            raw_flow_mode: object, raw_default: object) -> tuple[ModeDecision, str | None]:
    c = _check_ceiling(ceiling)
    configured, source, bad = _configured(raw_mode, raw_flow_mode, raw_default)
    effective: Mode = configured
    capped: CapReason | None = None
    if mode_rank(configured) > mode_rank(c):
        effective, capped = c, "ceiling"
    if effective == "generative":
        reason: CapReason | None = None
        if s.decl.slot_type != "choice":
            reason = "type"
        elif s.sensitive:
            reason = "sensitive"
        elif not r.generative_enabled:
            reason = "generative_disabled"
        if reason is not None:
            effective, capped = "assisted", reason
    decision = ModeDecision(slot=s.name, flow=flow, configured=configured, source=source,
                            ceiling=c, effective=effective, capped=capped)
    return decision, bad


def mode_decision(s: SlotSettings, r: ResolutionSettings, flow: str, ceiling: Mode,
                  *, raw_mode: object, raw_flow_mode: object, raw_default: object) -> ModeDecision:
    """The decision for slot ``s`` in ``flow``. ``raw_*`` are the values as
    stored: ``None`` is absent (inherit; for the default, invalid), :data:`INVALID` or any other
    non-mode value is invalid. ``r`` supplies ``generative_enabled``. Never raises for any raw
    value; a ceiling that isn't a mode is a configuration error (``ValueError``)."""
    return _decide(s, r, flow, ceiling, raw_mode=raw_mode, raw_flow_mode=raw_flow_mode,
                   raw_default=raw_default)[0]


# --------------------------------------------------------------------------- slot_mode_invalid

_reported: set[tuple[str, str, int]] = set()
_reported_lock = threading.Lock()


def reset_invalid_reports() -> None:
    """Forget which ``slot_mode_invalid`` events were emitted (tests; a new process does this)."""
    with _reported_lock:
        _reported.clear()


def _report_invalid(item: str, field: str, version: int) -> None:
    key = (item, field, version)
    with _reported_lock:
        if key in _reported:
            return
        _reported.add(key)
    obs.emit(log, logging.ERROR, "slot_mode_invalid", item=item, field=field, policy_version=version,
             decision="deterministic")


def _split(full_field: str) -> tuple[str, str]:
    item, _, field = full_field.partition(".")
    return item, field


def _raw_values(policy: Policy, s: SlotSettings, flow: str) -> tuple[object, object, object]:
    bad = policy.invalid_mode_fields
    r = policy.slot_resolution
    raw_mode = INVALID if f"slot#{s.name}.mode" in bad else s.mode
    raw_flow = INVALID if f"{RESOLUTION_ITEM}.flows.{flow}.mode" in bad else r.flows.get(flow)
    raw_default = INVALID if f"{RESOLUTION_ITEM}.default_mode" in bad else r.default_mode
    return raw_mode, raw_flow, raw_default


def freeze(policy: Policy, flow: str, slots: Sequence[str], ceiling: Mode) -> dict[str, ModeDecision]:
    """The decisions for ``slots`` at the start of ``flow``, to keep in the session sidecar.
    Emits ``slot_mode_invalid`` for each invalid value that decided a slot's mode (once per
    process per item, field and policy version). ``KeyError`` for a name that isn't a slot."""
    r, version = policy.slot_resolution, policy.version
    out: dict[str, ModeDecision] = {}
    for name in slots:
        s = policy.slot(name)
        raw_mode, raw_flow, raw_default = _raw_values(policy, s, flow)
        decision, bad = _decide(s, r, flow, ceiling, raw_mode=raw_mode, raw_flow_mode=raw_flow,
                                raw_default=raw_default)
        if bad == "slot":
            _report_invalid(f"slot#{name}", "mode", version)
        elif bad == "flow":
            _report_invalid(RESOLUTION_ITEM, f"flows.{flow}.mode", version)
        elif bad == "default":
            _report_invalid(RESOLUTION_ITEM, "default_mode", version)
        out[name] = decision
    return out


# --------------------------------------------------------------------------- the cold-start table


def slot_flows(s: SlotSettings) -> tuple[str, ...]:
    """The flows a slot can be asked in: a factor's are login and step-up; another slot's is its
    ``flow``, or every flow when it names none."""
    if s.is_factor:
        return FACTOR_FLOWS
    return (s.flow,) if s.flow is not None else FLOWS


def mode_table(policy: Policy, ceiling: Mode) -> dict[str, dict[str, ModeDecision]]:
    """Every slot's decision in each of its flows: ``{slot: {flow: decision}}``."""
    table: dict[str, dict[str, ModeDecision]] = {}
    for flow in FLOWS:
        names = [n for n in policy.slots() if flow in slot_flows(policy.slot(n))]
        for name, d in freeze(policy, flow, names, ceiling).items():
            table.setdefault(name, {})[flow] = d
    return table


def _capped_events(policy: Policy, ceiling: Mode, table: dict[str, dict[str, ModeDecision]]
                   ) -> Iterable[dict[str, Any]]:
    """One event per capped default, flow override or slot. A cap by the ceiling or by
    ``generative_enabled`` is reported where the mode was configured; a cap by the slot's type or
    sensitivity is the slot's."""
    r, bad = policy.slot_resolution, policy.invalid_mode_fields
    events: dict[tuple[str, str, str], dict[str, Any]] = {}
    for name, by_flow in table.items():
        for flow, d in by_flow.items():
            if d.capped is None:
                continue
            if d.source == "slot" or d.capped in ("type", "sensitive"):
                key = ("slot", name, d.capped)
            elif d.source == "flow":
                key = ("flow", flow, d.capped)
            else:
                key = ("default", "default_mode", d.capped)
            ev = events.setdefault(key, {"scope": key[0], "target": key[1], "configured": d.configured,
                                         "effective": d.effective, "ceiling": ceiling,
                                         "reason": d.capped, "flows": []})
            ev["flows"].append(flow)
    # A flow override or default above the ceiling that no listed slot inherits is still logged.
    scopes = [("default", "default_mode", r.default_mode,
               f"{RESOLUTION_ITEM}.default_mode" not in bad)]
    scopes += [("flow", f, m, f"{RESOLUTION_ITEM}.flows.{f}.mode" not in bad) for f, m in r.flows.items()]
    for scope, name, m, valid in scopes:
        if valid and mode_rank(m) > mode_rank(ceiling):
            events.setdefault((scope, name, "ceiling"),
                              {"scope": scope, "target": name, "configured": m, "effective": ceiling,
                               "ceiling": ceiling, "reason": "ceiling", "flows": []})
    return events.values()


def log_mode_table(policy: Policy, ceiling: Mode) -> dict[str, dict[str, ModeDecision]]:
    """Once per cold start: every invalid mode value as ``slot_mode_invalid``, then
    ``slot_modes`` (the ceiling, ``default_mode``, each flow override and each slot's configured
    and effective mode per flow) and one ``slot_mode_capped`` per capped default, flow or slot.
    Names and modes only. Returns the table."""
    c = _check_ceiling(ceiling)
    version = policy.version
    for full in sorted(policy.invalid_mode_fields):
        _report_invalid(*_split(full), version)
    table = mode_table(policy, c)
    r, bad = policy.slot_resolution, policy.invalid_mode_fields
    default = "invalid" if f"{RESOLUTION_ITEM}.default_mode" in bad else r.default_mode
    flows = {f: ("invalid" if f"{RESOLUTION_ITEM}.flows.{f}.mode" in bad else m) for f, m in r.flows.items()}
    slots = {name: {flow: {"configured": d.configured, "source": d.source, "effective": d.effective}
                    for flow, d in by_flow.items()}
             for name, by_flow in table.items()}
    obs.emit(log, logging.INFO, "slot_modes", ceiling=c, default_mode=default, flows=flows,
             generative_enabled=r.generative_enabled, slots=slots)
    for ev in _capped_events(policy, c, table):
        obs.emit(log, logging.INFO, "slot_mode_capped", **ev)
    return table
