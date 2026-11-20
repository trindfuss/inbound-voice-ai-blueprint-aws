"""The admin API's settings, read once from the environment.

==========================================  ==============================================  ==========
Variable                                    Meaning                                         Default
==========================================  ==============================================  ==========
``OWLBANK_TABLE_POLICY``                    Physical name of the policy table               ``owlbank-policy``
``OWLBANK_TABLE_CATALOG``                   Physical name of the catalog table              ``owlbank-catalog``
``OWLBANK_TABLE_ADMIN_AUDIT`` Physical name of the audit table ``owlbank-admin-audit``
``OWLBANK_ADMIN_USER_POOL_ID`` The console's Cognito user pool ID required
``OWLBANK_STAGE``                           ``dev``, ``test`` or ``prod``                   ``dev``
``OWLBANK_ADMIN_AUDIT_RETENTION_DAYS``      Audit entry lifetime (TTL ``expires_at``)       90; 400 at ``prod``
``OWLBANK_ACCEPT_SPOKEN_FACTORS``           The front door's spoken-factors ceiling, shown  ``true``
                                            read-only
``OWLBANK_SLOT_MODE_CEILING``               The deployment's most permissive slot mode       ``assisted``
                                            (``deterministic``, ``assisted`` or
                                            ``generative``); any other value fails INIT
``AWS_REGION``                              Region for the DynamoDB and Cognito clients     unset
==========================================  ==============================================  ==========
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from owlbank.slot_policy import MODES, Mode, is_mode

#: Audit retention per stage: dev and test 90 days, prod 400 days.
DEFAULT_RETENTION_DAYS = {"dev": 90, "test": 90, "prod": 400}
STAGES = frozenset(DEFAULT_RETENTION_DAYS)
#: The bridge's policy and catalog cache: when a saved change takes effect.
TAKES_EFFECT_S = 60
#: The CDK context default of ``slotModeCeiling``, used when the variable is unset.
DEFAULT_SLOT_MODE_CEILING: Mode = "assisted"


def slot_mode_ceiling_from_env(env: Mapping[str, str] | None = None) -> Mode:
    """``OWLBANK_SLOT_MODE_CEILING``: unset or empty = ``assisted``; any other value that isn't a
    mode raises ``ValueError`` (the handler calls this at import, so a bad value fails INIT)."""
    env = os.environ if env is None else env
    raw = (env.get("OWLBANK_SLOT_MODE_CEILING") or "").strip()
    if not raw:
        return DEFAULT_SLOT_MODE_CEILING
    if not is_mode(raw):
        raise ValueError(f"OWLBANK_SLOT_MODE_CEILING must be one of {list(MODES)}")
    return raw


@dataclass(frozen=True, slots=True)
class AdminConfig:
    user_pool_id: str
    policy_table: str = "owlbank-policy"
    catalog_table: str = "owlbank-catalog"
    audit_table: str = "owlbank-admin-audit"
    stage: str = "dev"
    audit_retention_days: int = 90
    aws_region: str | None = None
    #: The deployment's ``acceptSpokenFactors`` (on by default). Shown read-only by
    #: ``GET /slots``; ``true``/``false`` read as the front door reads them, unset = ``true``.
    accept_spoken_factors: bool = True
    #: The deployment's slot mode ceiling, shown by ``GET /slots`` and used for the
    #: effective modes there and in the preview.
    slot_mode_ceiling: Mode = DEFAULT_SLOT_MODE_CEILING

    def __post_init__(self) -> None:
        if not self.user_pool_id:
            raise ValueError("OWLBANK_ADMIN_USER_POOL_ID is required")
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {sorted(STAGES)}")
        if not 1 <= self.audit_retention_days <= 3650:
            raise ValueError("audit retention must be 1..3650 days")
        if not is_mode(self.slot_mode_ceiling):
            raise ValueError(f"the slot mode ceiling must be one of {list(MODES)}")

    @property
    def table_names(self) -> dict[str, str]:
        """Logical to physical names, for :class:`owlbank.store.DynamoStore`."""
        return {"policy": self.policy_table, "catalog": self.catalog_table,
                "admin_audit": self.audit_table}

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AdminConfig:
        env = os.environ if env is None else env
        stage = env.get("OWLBANK_STAGE", "dev") or "dev"
        raw_days = env.get("OWLBANK_ADMIN_AUDIT_RETENTION_DAYS")
        days = int(raw_days) if raw_days else DEFAULT_RETENTION_DAYS.get(stage, 90)
        raw_spoken = (env.get("OWLBANK_ACCEPT_SPOKEN_FACTORS") or "").strip()
        return cls(
            user_pool_id=env.get("OWLBANK_ADMIN_USER_POOL_ID", ""),
            policy_table=env.get("OWLBANK_TABLE_POLICY", "owlbank-policy"),
            catalog_table=env.get("OWLBANK_TABLE_CATALOG", "owlbank-catalog"),
            audit_table=env.get("OWLBANK_TABLE_ADMIN_AUDIT", "owlbank-admin-audit"),
            stage=stage,
            audit_retention_days=days,
            aws_region=env.get("AWS_REGION") or None,
            accept_spoken_factors=raw_spoken.lower() == "true" if raw_spoken else True,
            slot_mode_ceiling=slot_mode_ceiling_from_env(env),
        )
