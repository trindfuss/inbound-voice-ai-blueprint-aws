"""Redaction routes: ``GET /redaction`` and ``PUT /redaction``.

**Same rules as the components**. A ``PUT`` is checked by
:func:`owlbank.pii.parse_redaction_config` (strict), the parser the bridge, the front door and the
context API load the ``redaction`` item with, over the whole new section: the stored item's fields
(a field the schema no longer has, such as ``apply_to``, is dropped) with the body's fields over
them. The body is accepted iff that section parses, so ``enabled: true`` with no entity types is a
400. "No type selected" (nothing is redacted) is saved by the console as
``enabled: false``, which the parser accepts with any list, the empty one included. The item is
stored as the parsed config's :meth:`~owlbank.pii.RedactionConfig.to_doc`, so it round-trips.

``GET`` reports the stored fields, the config the components run with (a missing or invalid item
runs the built-in defaults, with the field named in ``problems``), the defaults and the
entity types Comprehend offers. It is open to every console group; ``PUT`` is ``admin`` only
(:data:`.app.ADMIN`).

**Writes** are one transaction of the item (``rev + 1``, conditional on the ``rev`` read) and its
audit entry, as the other config routes do. Every field is a security change: the
audit entry names each changed field as ``redaction.<field>``, all in
:data:`.audit.SECURITY_FIELDS` (prefixed so a tool's ``mask`` stays a non-security change).

**Warnings** (each ``{"code", "message"}``): ``redaction_off`` on any view where the config is
``enabled: false``; ``entity_type_removed`` (with ``entity_types``, the removed type names) on a
write that drops a type the config listed before. Neither a response nor a log line carries
anything but setting names, entity type names and the configured values.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from owlbank import obs
from owlbank import policy as policy_mod
from owlbank.pii import (
    COMPREHEND_PII_ENTITY_TYPES,
    DEFAULT_CONFIG,
    ON_FAILURE,
    RedactionConfig,
    RedactionConfigError,
    parse_redaction_config,
)
from owlbank.store import Expect, Store, TransactionCancelled, TxPut

from . import audit
from .audit import AuditClock
from .config import TAKES_EFFECT_S
from .config_routes import _only_keys, _rev, _rev_of_body
from .http import ApiError, Caller

log = logging.getLogger(__name__)

REV = policy_mod.REV_ATTR
#: The policy item's key.
REDACTION_ITEM = "redaction"
#: The fields ``PUT /redaction`` may send (besides ``rev``), in the parser's order.
FIELDS: tuple[str, ...] = ("enabled", "entity_types", "min_confidence", "mask", "on_failure")
#: The audit names of :data:`FIELDS` (all in :data:`.audit.SECURITY_FIELDS`).
AUDIT_FIELDS: Mapping[str, str] = {f: f"redaction.{f}" for f in FIELDS}

WARNINGS: Mapping[str, str] = {
    "redaction_off": (
        "Transcript redaction is off: the AI turns in the call log, the escalation summary and "
        "the screen-pop timeline are then stored and shown without PII masking. A sensitive "
        "slot's value is still masked."),
    "entity_type_removed": (
        "Removing an entity type is a security change: text of that type is then no longer "
        "masked in the call log, the escalation summary or the screen-pop timeline."),
}


def warning(code: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "message": WARNINGS[code], **extra}


def _fields(item: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The stored section: the item without its key and ``rev``; ``None`` if there is none."""
    return None if item is None else {k: v for k, v in item.items() if k not in ("pk", REV)}


def _state(raw: Mapping[str, Any] | None) -> tuple[RedactionConfig, list[dict[str, str]]]:
    """What the components run with (:func:`owlbank.pii.load_redaction_config`'s result, without
    its ERROR) and the problem that made them fall back, if any."""
    if raw is None:
        return DEFAULT_CONFIG, [{"field": REDACTION_ITEM, "message": "the item is missing"}]
    try:
        return parse_redaction_config(raw), []
    except RedactionConfigError as exc:
        return DEFAULT_CONFIG, [{"field": f"redaction.{exc.field}", "message": str(exc)}]


def _state_warnings(cfg: RedactionConfig) -> list[dict[str, Any]]:
    return [] if cfg.enabled else [warning("redaction_off")]


class RedactionRoutes:
    def __init__(self, store: Store, *, clock: AuditClock, retention_days: int) -> None:
        self.store, self.clock, self.retention_days = store, clock, retention_days

    def _item(self) -> dict[str, Any] | None:
        return self.store.get("policy", {"pk": REDACTION_ITEM})

    def _view(self, item: Mapping[str, Any] | None) -> dict[str, Any]:
        raw = _fields(item)
        effective, problems = _state(raw)
        return {
            "redaction": raw or {}, "effective": effective.to_doc(),
            "defaults": DEFAULT_CONFIG.to_doc(), "problems": problems, "rev": _rev(item),
            "entity_types_available": sorted(COMPREHEND_PII_ENTITY_TYPES),
            "on_failure_values": sorted(ON_FAILURE), "takes_effect_s": TAKES_EFFECT_S,
            "warnings": _state_warnings(effective),
        }

    def get_redaction(self) -> dict[str, Any]:
        return self._view(self._item())

    def _conflict(self) -> ApiError:
        return ApiError(409, "conflict", "the redaction settings changed since they were read",
                        extra={"current": self._view(self._item())})

    def put_redaction(self, caller: Caller, body: Mapping[str, Any]) -> dict[str, Any]:
        """Admin only; :mod:`.app` refuses anyone else first."""
        _only_keys(body, {"rev", *FIELDS})
        rev = _rev_of_body(body)
        updates = {f: body[f] for f in FIELDS if f in body}
        if not updates:
            raise ApiError(400, "invalid", f"nothing to update: send any of {list(FIELDS)}")
        item = self._item()
        if _rev(item) != rev:
            raise self._conflict()
        stored = _fields(item)
        base = {f: stored[f] for f in FIELDS if f in stored} if stored is not None else {}
        try:
            new = parse_redaction_config({**base, **updates})
        except RedactionConfigError as exc:
            raise ApiError(400, "invalid", str(exc), field=f"redaction.{exc.field}") from None
        loaded, _ = _state(stored)
        new_doc = new.to_doc()
        # Compare as the components read it (a missing item: every field is new).
        if stored is None:
            old_doc: dict[str, Any] = {}
        else:
            try:
                old_doc = parse_redaction_config(base).to_doc()
            except RedactionConfigError:
                old_doc = base
        changes = [{**c, "field": AUDIT_FIELDS[c["field"]]} for c in audit.diff(old_doc, new_doc, FIELDS)]
        if not changes:
            return {"changed": False, **self._view(item)}
        security_change = any(c["field"] in audit.SECURITY_FIELDS for c in changes)
        new_rev = rev + 1
        new_item = {"pk": REDACTION_ITEM, **new_doc, REV: new_rev}
        entry = audit.entry(pk=audit.REDACTION_PK, action="update", kind="redaction",
                            target_id=REDACTION_ITEM, actor=caller, changes=changes,
                            security_change=security_change, now=self.clock.now(),
                            suffix=self.clock.suffix(), retention_days=self.retention_days,
                            rev=new_rev)
        try:
            self.store.transact_write([TxPut("policy", new_item, Expect(REV, rev, absent_as=0)),
                                       TxPut(audit.AUDIT_TABLE, entry, "not_exists")])
        except TransactionCancelled:
            raise self._conflict() from None
        obs.emit(log, logging.INFO, "admin_config_write", action="update", target_kind="redaction",
                 target_id=REDACTION_ITEM, actor_sub=caller.sub, rev=new_rev,
                 security_change=security_change, fields=[c["field"] for c in changes])

        warnings = _state_warnings(new)
        removed = sorted(set(loaded.entity_types) - set(new.entity_types))
        if removed:
            warnings.append(warning("entity_type_removed", entity_types=removed))
        return {**self._view(new_item), "changed": True, "audit": {"pk": entry["pk"], "sk": entry["sk"]},
                "security_change": security_change, "warnings": warnings}
