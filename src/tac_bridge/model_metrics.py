"""The bridge's model-failover metrics.

Three CloudWatch EMF metrics, one JSON line each on stdout, namespace ``OwlBank/<stage>/Model``,
dimensions ``Service`` (``owlbank-<stage>-bridge``) and ``Environment`` (the stage), the names and
the dimension set ``infra/owlbank_infra/model_failover.py`` alarms on (a parity test keeps them equal):

* ``ActiveRank``: the rank of the entry this process would give a turn now (0 = the preferred
  entry), every :data:`ACTIVE_RANK_INTERVAL_S` and at once when a turn starts on another rank
  (:class:`ActiveRankMonitor`); the infra alarm fires when it stays above 0 for 15 minutes.
* ``Failover``: 1 per ``model_failover`` (a turn moved off its entry, or started below the
  preferred one because of the verdict).
* ``Unavailable``: 1 per entry this process found unavailable (``model_unavailable``).

**The stage.** The bridge has no stage setting of its own when call observability is off (
the fleet table's partitions are fixed values, so the bridge needs none). :func:`metric_stage`
takes it from the fleet table's name, ``owlbank-<stage>-modelstats`` (``InfraConfig
.failover_table_name``, the one variable every failover-on bridge has), and otherwise from
``OWLBANK_STAGE``; with neither, no metric is written (``model_metrics_off``).

Lines go straight to stdout (both hostings ship stdout to the log group, where CloudWatch extracts
EMF), under a lock, one ``write`` per line. Nothing here takes text: a line holds the fixed names,
the stage and a number. Writing never raises.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Final, TextIO

from owlbank.obs import emit, log_caught
from tac_bridge.fleet import TABLE_ENV, Periodic

log = logging.getLogger(__name__)

NAMESPACE_FORMAT: Final = "OwlBank/{stage}/Model"
SERVICE_FORMAT: Final = "owlbank-{stage}-bridge"
DIMENSIONS: Final = ("Service", "Environment")
ACTIVE_RANK: Final = "ActiveRank"
FAILOVER: Final = "Failover"
UNAVAILABLE: Final = "Unavailable"
#: ``ActiveRank`` every 60 s (and on change).
ACTIVE_RANK_INTERVAL_S: Final = 60.0
STAGE_ENV: Final = "OWLBANK_STAGE"
#: ``InfraConfig.failover_table_name``: ``owlbank-<stage>-modelstats``.
_TABLE_NAME = re.compile(r"owlbank-(?P<stage>[a-z0-9]+(?:-[a-z0-9]+)*)-modelstats")
_STAGE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def metric_stage(env: Mapping[str, str]) -> str | None:
    """The stage for the metric names: from the fleet table's name, else ``OWLBANK_STAGE``; ``None``
    when neither gives a well-formed one."""
    m = _TABLE_NAME.fullmatch((env.get(TABLE_ENV) or "").strip())
    if m is not None:
        return m["stage"]
    stage = (env.get(STAGE_ENV) or "").strip()
    return stage if _STAGE.fullmatch(stage) else None


class ModelMetrics:
    """Writes the three metrics as EMF lines (``stream``: ``sys.stdout`` at write time)."""

    def __init__(self, stage: str, *, stream: TextIO | None = None, wall: Callable[[], float] = time.time) -> None:
        if not isinstance(stage, str) or not _STAGE.fullmatch(stage):
            raise ValueError("stage must be a stage name")
        self.stage = stage
        self.namespace = NAMESPACE_FORMAT.format(stage=stage)
        self.service = SERVICE_FORMAT.format(stage=stage)
        self._stream, self._wall = stream, wall
        self._lock = threading.Lock()

    def record(self, name: str, value: int) -> dict[str, Any]:
        """The EMF object for one value."""
        return {
            "_aws": {"Timestamp": round(self._wall() * 1000), "CloudWatchMetrics": [{
                "Namespace": self.namespace, "Dimensions": [list(DIMENSIONS)],
                "Metrics": [{"Name": name, "Unit": "Count" if name != ACTIVE_RANK else "None"}]}]},
            "Service": self.service,
            "Environment": self.stage,
            name: value,
        }

    def put(self, name: str, value: int) -> None:
        """One line; never raises (a failure is a debug line)."""
        try:
            if name not in (ACTIVE_RANK, FAILOVER, UNAVAILABLE) or isinstance(value, bool) \
                    or not isinstance(value, int):
                raise ValueError("not a model metric")
            line = json.dumps(self.record(name, value), separators=(",", ":"), allow_nan=False) + "\n"
            with self._lock:
                out = self._stream if self._stream is not None else sys.stdout
                out.write(line)
                out.flush()
        except Exception as exc:  # noqa: BLE001 - a metric never fails a call
            log_caught(log, "model_metric_dropped", exc, decision="drop", level=logging.DEBUG)

    # The client's events (``agent_client_strands.FailoverEvents``).

    def failover(self) -> None:
        self.put(FAILOVER, 1)

    def unavailable(self) -> None:
        self.put(UNAVAILABLE, 1)


class ActiveRankMonitor:
    """``ActiveRank`` every :data:`ACTIVE_RANK_INTERVAL_S` from ``rank()`` and at once on a change
    (:meth:`observe`, called at each turn's start). The loop keeps its due time on the monotonic clock
    and sleeps again when woken early, so it never writes more often than its interval."""

    def __init__(self, metrics: ModelMetrics, rank: Callable[[], int], *,
                 interval_s: float = ACTIVE_RANK_INTERVAL_S, clock: Callable[[], float] = time.monotonic) -> None:
        self._metrics, self._rank = metrics, rank
        self._last: int | None = None
        self._loop = Periodic(self.tick, interval_s, clock, first_at_once=True)

    @property
    def last(self) -> int | None:
        return self._last

    def observe(self, rank: int) -> None:
        """A turn started on ``rank``: write it now if it differs from the last written."""
        if rank != self._last:
            self._write(rank)

    def _write(self, rank: int) -> None:
        self._last = rank
        self._metrics.put(ACTIVE_RANK, rank)

    async def tick(self) -> None:
        try:
            self._write(int(self._rank()))
        except Exception as exc:  # noqa: BLE001 - the next tick tries again
            log_caught(log, "model_metric_dropped", exc, decision="drop", level=logging.DEBUG)

    def start(self) -> None:
        self._loop.start()

    async def stop(self) -> None:
        await self._loop.stop()


def metrics_from_env(env: Mapping[str, str], *, stream: TextIO | None = None) -> ModelMetrics | None:
    """The metrics writer, or ``None`` (``model_metrics_off``) when no stage is known."""
    stage = metric_stage(env)
    if stage is None:
        emit(log, logging.WARNING, "model_metrics_off", reason="no_stage")
        return None
    return ModelMetrics(stage, stream=stream)


__all__ = ["ACTIVE_RANK", "ACTIVE_RANK_INTERVAL_S", "DIMENSIONS", "FAILOVER", "NAMESPACE_FORMAT", "SERVICE_FORMAT",
           "UNAVAILABLE", "ActiveRankMonitor", "ModelMetrics", "metric_stage", "metrics_from_env"]
