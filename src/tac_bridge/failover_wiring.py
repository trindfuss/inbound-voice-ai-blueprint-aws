"""Model failover's wiring in the bridge.

:meth:`FailoverRuntime.from_env` builds, for a client with two or more entries and the fleet table's
name in ``OWLBANK_TABLE_MODELSTATS``, the :class:`~tac_bridge.fleet.FleetReader` (the cached verdict,
:meth:`StrandsAgentClient.use_fleet_view`), the :class:`~tac_bridge.fleet.FleetReporter` (every model
call's outcome, :meth:`StrandsAgentClient.use_fleet_reporter`; it samples with the reader's
``report_every``, and re-reports the entries :meth:`FailoverState.unconfirmed
<owlbank.failover.core.FailoverState.unconfirmed>` names against the reader's view on every flush) and
the metrics (``ActiveRank``, ``Failover``, ``Unavailable``,
:mod:`tac_bridge.model_metrics`, :meth:`StrandsAgentClient.use_failover_events`). Without the table
variable (``modelFailover=off``: infra keeps one entry and no table env) or with a list of one it
returns ``None``: no reporter, reader or metric is built, and the client runs today's single-model
path. With the table but no stage for the metric names the fleet picture still runs, without
the metrics (``model_metrics_off``). A local unavailable mark that the fleet clears (its recheck) or
the backstop drops (the fleet never confirmed it) is logged as ``model_available`` (``scope=process``,
4a.5).

The background loops start in the app's lifespan (:meth:`FailoverRuntime.start`, on the running
loop) and stop at shutdown (:meth:`FailoverRuntime.stop`, with the reporter's final flush). Under
AgentCore a process is one call, so :meth:`FailoverRuntime.call_ended` flushes the reporter when the
relay socket closes (one update per call per 30 s and one at call end); under Fargate the 10 s
flush covers every call of the task.

:func:`log_models_resolved` writes the start-up line ``models_resolved``: the validated list
(model IDs only), the entries that use a ``global.`` profile (WARNING then: they may be processed in
any commercial Region) and whether the fleet picture is on.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, TextIO

from owlbank.failover.core import Transition
from owlbank.obs import emit, log_caught
from tac_bridge.fleet import TABLE, TABLE_ENV, FleetReader, FleetReporter
from tac_bridge.hosting import Hosting
from tac_bridge.model_metrics import ActiveRankMonitor, ModelMetrics, metrics_from_env

log = logging.getLogger(__name__)

GLOBAL_PREFIX = "global."


class _Events:
    """The client's :class:`~tac_bridge.agent_client_strands.FailoverEvents` over the metrics."""

    def __init__(self, metrics: ModelMetrics, monitor: ActiveRankMonitor) -> None:
        self._metrics, self._monitor = metrics, monitor

    def failover(self) -> None:
        self._metrics.failover()

    def unavailable(self) -> None:
        self._metrics.unavailable()

    def active(self, rank: int) -> None:
        self._monitor.observe(rank)


def _log_cleared(model_ids: tuple[str, ...]) -> Callable[[Transition], None]:
    """The decision state's ``on_cleared`` listener: ``model_available`` for a cleared local mark
    (WARNING when the backstop dropped it: the fleet never took this process's reports). Never raises."""

    def listener(change: Transition) -> None:
        try:
            emit(log, logging.WARNING if change.cause == "unconfirmed" else logging.INFO, "model_available",
                 entry=change.entry, model_id=model_ids[change.entry], cause=change.cause, scope="process")
        except Exception as exc:  # noqa: BLE001 - a log line never fails a turn
            log_caught(log, "model_available_log_failed", exc, decision="drop", level=logging.DEBUG)

    return listener


class FailoverRuntime:
    """The fleet picture and the metrics of one bridge process."""

    def __init__(self, *, reader: FleetReader, reporter: FleetReporter, hosting: Hosting,
                 metrics: ModelMetrics | None = None, monitor: ActiveRankMonitor | None = None) -> None:
        self.reader, self.reporter, self.hosting = reader, reporter, hosting
        self.metrics, self.monitor = metrics, monitor

    @classmethod
    def from_env(cls, env: Mapping[str, str], client: Any, *, hosting: Hosting, store: Any = None,
                 metrics_stream: TextIO | None = None) -> FailoverRuntime | None:
        """Build and attach to ``client`` (a ``StrandsAgentClient``), or ``None`` (today's path)."""
        n = len(client.settings.models)
        table = (env.get(TABLE_ENV) or "").strip()
        if n < 2 or not table:
            emit(log, logging.INFO, "model_fleet", state="off", reason="single_entry" if n < 2 else "no_table")
            return None
        if store is None:
            from owlbank.store import DynamoStore  # noqa: PLC0415 - boto3 only when on

            store = DynamoStore({TABLE: table}, region_name=env.get("AWS_REGION") or None, required=(TABLE,))
        model_ids = tuple(client.settings.model_ids)
        reader = FleetReader(store, model_ids=model_ids)
        state = client.failover
        reporter = FleetReporter(store, model_ids=model_ids, hosting=hosting, report_every=reader.report_every,
                                 unconfirmed=lambda: state.unconfirmed(fleet=reader.view()))
        state.on_cleared(_log_cleared(model_ids))
        client.use_fleet_view(reader.view)
        client.use_fleet_reporter(reporter)
        metrics = metrics_from_env(env, stream=metrics_stream)
        monitor = None
        if metrics is not None:
            monitor = ActiveRankMonitor(metrics, client.active_rank)
            client.use_failover_events(_Events(metrics, monitor))
        emit(log, logging.INFO, "model_fleet", state="on", hosting=hosting, entries=n,
             metrics="on" if metrics is not None else "off")
        return cls(reader=reader, reporter=reporter, hosting=hosting, metrics=metrics, monitor=monitor)

    async def start(self) -> None:
        """Start the read, flush and ``ActiveRank`` loops on the running loop (the app's lifespan)."""
        self.reader.start()
        self.reporter.start()
        if self.monitor is not None:
            self.monitor.start()

    async def call_ended(self) -> None:
        """AgentCore: flush this call's counters now. Fargate: nothing (the 10 s flush). Never
        raises."""
        if self.hosting == "agentcore":
            await self.reporter.flush()

    async def stop(self) -> None:
        """Stop the loops; the reporter flushes what is pending first. Never raises."""
        if self.monitor is not None:
            await self.monitor.stop()
        await self.reporter.stop(final_flush=True)
        await self.reader.stop()


def log_models_resolved(model_ids: tuple[str, ...], *, fleet: bool) -> None:
    """``models_resolved`` at start-up: the list (IDs only), its ``global.`` entries, the fleet."""
    global_ids = [m for m in model_ids if m.startswith(GLOBAL_PREFIX)]
    emit(log, logging.WARNING if global_ids else logging.INFO, "models_resolved", models=list(model_ids),
         entries=len(model_ids), preferred=model_ids[0], global_entries=global_ids,
         failover="on" if len(model_ids) > 1 else "off", fleet="on" if fleet else "off")


__all__ = ["FailoverRuntime", "log_models_resolved"]
