"""Deploy-time settings recorded in the audit log at start-up.

``acceptSpokenFactors`` is set by the deploy (``-c acceptSpokenFactors=...``, passed to this Lambda
as ``OWLBANK_ACCEPT_SPOKEN_FACTORS``); the console only shows it. So that a change still shows up in
the audit log, the admin API compares it, at cold start, with the last recorded value:

* The **marker** is the ``admin_audit`` item ``pk = deploy#accept_spoken_factors``,
  ``sk = marker``: ``{"value": bool, "rev": int}``. It has no ``expires_at`` (no TTL), so the last
  recorded value is never forgotten. :func:`.config_routes.ConfigRoutes.list_audit` leaves it out.
* If the marker is absent (the first start) or holds another value, one transaction writes the
  marker (``rev + 1``, conditional on the ``rev`` read, so on the old value) and one audit entry
  under the same ``pk``: ``action: deploy_setting``, actor ``{"sub": "deploy"}``, ``changes``
  ``[{"field": "acceptSpokenFactors", "old": <before or None>, "new": <after>}]`` and
  ``security_change: true`` when the setting turned on (including a first start with ``true``).
* Two cold starts racing: the one whose condition fails writes nothing more (the other's entry
  stands). Any error is logged with :func:`owlbank.obs.log_caught` and never fails the cold start.

Log lines carry the setting's name and booleans only.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from owlbank import obs
from owlbank.store import Expect, Store, TransactionCancelled, TxPut

from . import audit
from .audit import AuditClock
from .http import Caller

log = logging.getLogger(__name__)

SETTING = "acceptSpokenFactors"
DEPLOY_PK = "deploy#accept_spoken_factors"
#: The marker's sort key. Entries' sort keys are ISO times, so they never collide with it.
MARKER_SK = "marker"
ACTION = "deploy_setting"
DEPLOY_ACTOR = Caller(sub="deploy", email=None, username=None)

Outcome = Literal["recorded", "unchanged", "lost_race", "failed"]


def marker_key() -> dict[str, str]:
    return {"pk": DEPLOY_PK, "sk": MARKER_SK}


def _rev(marker: dict[str, Any] | None) -> int:
    rev = (marker or {}).get("rev", 0)
    return rev if isinstance(rev, int) and not isinstance(rev, bool) else 0


def _value(marker: dict[str, Any] | None) -> bool | None:
    """The recorded value; ``None`` when there is no marker or it doesn't hold a boolean."""
    value = (marker or {}).get("value")
    return value if isinstance(value, bool) else None


def record_accept_spoken_factors(store: Store, value: bool, *, clock: AuditClock,
                                 retention_days: int = 90) -> Outcome:
    """Record ``acceptSpokenFactors`` if it changed since the marker. Never raises."""
    value = bool(value)
    try:
        marker = store.get(audit.AUDIT_TABLE, marker_key())
        before = _value(marker)
        if marker is not None and before is value:
            obs.emit(log, logging.DEBUG, "admin_deploy_setting", setting=SETTING, after=value,
                     outcome="unchanged")
            return "unchanged"
        rev = _rev(marker)
        security_change = value and before is not True
        entry = audit.entry(pk=DEPLOY_PK, action=ACTION, kind="deploy_setting", target_id=SETTING,
                            actor=DEPLOY_ACTOR,
                            changes=[{"field": SETTING, "old": before, "new": value}],
                            security_change=security_change, now=clock.now(),
                            suffix=clock.suffix(), retention_days=retention_days)
        entry["actor"] = {"sub": DEPLOY_ACTOR.sub}
        new_marker = {**marker_key(), "value": value, "rev": rev + 1}
        try:
            store.transact_write([
                TxPut(audit.AUDIT_TABLE, new_marker, Expect("rev", rev, absent_as=0)),
                TxPut(audit.AUDIT_TABLE, entry, "not_exists"),
            ])
        except TransactionCancelled:
            obs.emit(log, logging.INFO, "admin_deploy_setting", setting=SETTING, after=value,
                     outcome="lost_race")
            return "lost_race"
        obs.emit(log, logging.INFO, "admin_deploy_setting", setting=SETTING, before=before,
                 after=value, security_change=security_change, outcome="recorded")
        return "recorded"
    except Exception as exc:  # noqa: BLE001 - never fail the cold start over the audit entry
        obs.log_caught(log, "admin api: deploy setting not recorded", exc, level=logging.ERROR,
                       setting=SETTING, decision="continue")
        return "failed"
