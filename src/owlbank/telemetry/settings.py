"""Telemetry settings from the environment.

:meth:`TelemetrySettings.from_env` returns ``None`` unless ``OWLBANK_CALL_OBSERVABILITY`` is
``on``, so a process with the switch off (the default) never goes further. The CDK app
validates the same values at synth; this is the process's own check. An invalid value logs
``telemetry_disabled`` with the setting's name, never its value, and returns ``None``: the
process runs untraced instead of failing.

This module must not import OpenTelemetry (a test checks it).
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from owlbank.obs import emit

log = logging.getLogger(__name__)

Content = Literal["off", "redacted", "full"]
Hosting = Literal["fargate", "agentcore"]

SWITCH_ENV = "OWLBANK_CALL_OBSERVABILITY"
SAMPLE_RATE_ENV = "OWLBANK_TRACE_SAMPLE_RATE"
CONTENT_ENV = "OWLBANK_TRACE_CONTENT"
TRACES_LOG_GROUP_ENV = "OWLBANK_TRACES_LOG_GROUP"
#: Neither the bridge nor the front door had a stage variable before this feature; the admin API
#: already reads ``OWLBANK_STAGE`` (``lambdas/admin_api/config.py``), so the stack sets the same.
STAGE_ENV = "OWLBANK_STAGE"
#: Same variable and values as :mod:`tac_bridge.hosting` (not imported: ``owlbank`` doesn't
#: depend on the bridge package).
HOSTING_ENV = "OWLBANK_HOSTING"
REGION_ENV = "AWS_REGION"

CONTENTS: tuple[Content, ...] = ("off", "redacted", "full")
HOSTINGS: tuple[Hosting, ...] = ("fargate", "agentcore")
DEFAULT_SAMPLE_RATE = 1.0
DEFAULT_STAGE = "dev"
#: A stage goes into ``service.name`` (``owlbank-<stage>-bridge``), so it is kept to a name.
_STAGE_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")


class _Invalid(Exception):
    """A setting's value is unusable; carries the setting's name only."""

    def __init__(self, setting: str) -> None:
        super().__init__(setting)
        self.setting = setting


@dataclass(frozen=True)
class TelemetrySettings:
    """What a process needs to start telemetry. Its existence means the switch is on."""

    stage: str
    service_name: str  # owlbank-<stage>-<service>
    hosting: Hosting
    sample_rate: float  # 0.0-1.0, per call
    content: Content  # traceContent
    traces_log_group: str | None  # the bridge's traces log group; the front door has none
    region: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, service: str) -> TelemetrySettings | None:
        """Settings for ``service`` (``bridge`` or ``frontdoor``), or ``None`` when off or invalid."""
        env = os.environ if env is None else env
        switch = (env.get(SWITCH_ENV) or "").strip()
        if switch != "on":
            if switch not in ("", "off"):
                emit(log, logging.WARNING, "telemetry_disabled", setting=SWITCH_ENV)
            return None
        try:
            stage = _stage(env)
            return cls(
                stage=stage,
                service_name=f"owlbank-{stage}-{service}",
                hosting=_choice(env, HOSTING_ENV, HOSTINGS, "fargate"),
                sample_rate=_rate(env),
                content=_choice(env, CONTENT_ENV, CONTENTS, "off"),
                traces_log_group=(env.get(TRACES_LOG_GROUP_ENV) or "").strip() or None,
                region=(env.get(REGION_ENV) or "").strip() or None,
            )
        except _Invalid as bad:
            emit(log, logging.WARNING, "telemetry_disabled", setting=bad.setting)
            return None


def _stage(env: Mapping[str, str]) -> str:
    raw = (env.get(STAGE_ENV) or "").strip() or DEFAULT_STAGE
    if not _STAGE_RE.fullmatch(raw):
        raise _Invalid(STAGE_ENV)
    return raw


def _choice[T: str](env: Mapping[str, str], name: str, allowed: tuple[T, ...], default: T) -> T:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    for value in allowed:
        if raw == value:
            return value
    raise _Invalid(name)


def _rate(env: Mapping[str, str]) -> float:
    raw = (env.get(SAMPLE_RATE_ENV) or "").strip()
    if not raw:
        return DEFAULT_SAMPLE_RATE
    try:
        rate = float(raw)
    except ValueError:
        raise _Invalid(SAMPLE_RATE_ENV) from None
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise _Invalid(SAMPLE_RATE_ENV)
    return rate
