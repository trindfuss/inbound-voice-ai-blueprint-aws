"""Lambda entry point for the admin API (``lambdas.admin_api.handler.handler``).

Thin: it builds :class:`.app.AdminApi` once, lazily, from the environment, and hands
each event to it. Importing this module creates no AWS client. Environment (see
:mod:`.config`): ``OWLBANK_ADMIN_USER_POOL_ID`` (required), ``OWLBANK_TABLE_POLICY``,
``OWLBANK_TABLE_CATALOG``, ``OWLBANK_TABLE_ADMIN_AUDIT``, ``OWLBANK_STAGE``,
``OWLBANK_ADMIN_AUDIT_RETENTION_DAYS`` (optional), ``OWLBANK_ACCEPT_SPOKEN_FACTORS``,
``OWLBANK_SLOT_MODE_CEILING`` (checked at import: an invalid value fails INIT), ``AWS_REGION``.

At cold start (:func:`start_api`) it records ``acceptSpokenFactors`` in the audit log if the deploy
changed it (:mod:`.deploy_settings`); that never fails the start.

The role needs ``dynamodb:GetItem``, ``Query`` and
``PutItem`` (also through ``TransactWriteItems``) on the policy and catalog tables,
``PutItem``/``Query``/``GetItem`` on the audit table, the project CMK, and the
``cognito-idp`` actions listed in :mod:`.users` on the one user pool.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from owlbank import obs

from .app import AdminApi, Deps
from .audit import AuditClock
from .config import slot_mode_ceiling_from_env
from .deploy_settings import record_accept_spoken_factors
from .http import response

log = logging.getLogger(__name__)

#: Fails INIT on a ceiling that isn't a mode, before any request.
SLOT_MODE_CEILING = slot_mode_ceiling_from_env()

_api: AdminApi | None = None


def build_api(env: Mapping[str, str] | None = None) -> AdminApi:  # pragma: no cover - needs AWS
    import boto3

    from owlbank.store import DynamoStore

    from .config import AdminConfig

    cfg = AdminConfig.from_env(env)
    names = cfg.table_names
    store = DynamoStore(names, region_name=cfg.aws_region, required=names)
    cognito = boto3.client("cognito-idp", region_name=cfg.aws_region)
    return start_api(Deps(store=store, cognito=cognito, user_pool_id=cfg.user_pool_id,
                          retention_days=cfg.audit_retention_days,
                          accept_spoken_factors=cfg.accept_spoken_factors,
                          slot_mode_ceiling=cfg.slot_mode_ceiling))


def start_api(deps: Deps) -> AdminApi:
    """The API for ``deps``, after recording the deploy's ``acceptSpokenFactors`` (once per cold
    start). Never raises on the recording."""
    clock = AuditClock(now=deps.clock) if deps.suffix is None \
        else AuditClock(now=deps.clock, suffix=deps.suffix)
    record_accept_spoken_factors(deps.store, deps.accept_spoken_factors, clock=clock,
                                 retention_days=deps.retention_days)
    return AdminApi(deps)


def _route_op(event: Any, context: Any) -> str | None:
    try:
        found = AdminApi.resolve(event) if isinstance(event, Mapping) else None
    except Exception:  # noqa: BLE001 - a 405 has no op name
        return None
    return found[0].name if found else None


@obs.instrumented("admin_api", op=_route_op)
def handler(event: Mapping[str, Any], context: Any = None) -> dict[str, Any]:
    """Lambda entry point."""
    global _api
    try:
        if _api is None:
            _api = build_api()
    except Exception as exc:
        obs.log_caught(log, "admin api: setup failed", exc, level=logging.ERROR,
                       operation="build_api", decision="return_500")
        return response(500, {"error": "internal"})
    return _api.handle(event)
