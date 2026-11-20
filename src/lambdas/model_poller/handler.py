"""Lambda entry point for the model-failover poller (thin: wiring only; logic in ``service``).

Invoked by an EventBridge schedule
once a minute (no API, URL or other trigger); the event is ignored. Each invocation runs two decision
cycles 30 s apart (:func:`lambdas.model_poller.service.run_invocation`) and returns
``{"cycles": <run>, "written": <state writes>}``. A cycle that would not finish within the function's
remaining time is skipped.

Environment:

* ``OWLBANK_STAGE`` (required): the stage, for the EMF metric names and the test delay's ``prod``
  check. Not a key: the table's partitions are the fixed ``samples`` and ``state`` (each stage has its
  own table).
* ``OWLBANK_MODELS`` (required): the model list, as the bridge reads it.
* ``OWLBANK_FAILOVER_TEST`` (optional, seconds): the test-only first-output delay of the preferred
  entry's pings; refused at ``stage=prod``.
* ``OWLBANK_TABLE_MODELSTATS``: the ``modelstats`` table (default ``owlbank-modelstats``): ``GetItem`` and
  ``PutItem`` (as a single-item ``TransactWriteItems``) on the state record, ``Query`` on the samples.
* ``OWLBANK_TABLE_POLICY``: the ``policy`` table (default ``owlbank-policy``), read for the
  ``model_failover`` section (``GetItem``, ``BatchGetItem``), cached 60 s; unreadable means the last
  settings, else the defaults.
* ``OWLBANK_LOG_LEVEL``: the log level (read by ``obs.configure_logging`` on the first invocation).
* ``AWS_REGION``: set by Lambda.

Both tables must have a name (the store is built with ``required=("modelstats", "policy")``).
Metrics: EMF lines on stdout, namespace ``OwlBank/<stage>/Model``, dimensions
``Service=owlbank-<stage>-model-poller`` and ``Environment=<stage>``: ``StateHeartbeat`` and
``UsableEntries`` (``service.emit_metric``). The cycle-skip check uses the context's remaining time.

Cold start: outside Lambda, importing this module makes no AWS call. Inside Lambda, the
dependencies are built once during INIT (``coldstart.build_at_init``); the Bedrock client has one
attempt and short timeouts (an abandoned ping's thread ends within the read timeout).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping
from typing import Any

from lambdas.model_poller import service
from owlbank import coldstart, obs
from owlbank.failover.settings import Settings
from owlbank.policy import ModelFailoverConfig, Policy
from owlbank.store import DynamoStore

#: Policy re-read interval (the bridges' cache).
POLICY_TTL_S = 60.0
BEDROCK_CONNECT_TIMEOUT_S = 2
BEDROCK_READ_TIMEOUT_S = 15

_deps: service.PollerDeps | None = None
_memory = service.PollerMemory()


def _bedrock_client(region: str | None) -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client("bedrock-runtime", region_name=region, config=Config(
        connect_timeout=BEDROCK_CONNECT_TIMEOUT_S, read_timeout=BEDROCK_READ_TIMEOUT_S,
        retries={"max_attempts": 1, "mode": "standard"}))


def build_deps(env: Mapping[str, str] | None = None, *, bedrock: Any | None = None,
               store: Any | None = None) -> service.PollerDeps:
    env = dict(os.environ) if env is None else env
    stage = (env.get(service.STAGE_ENV) or "").strip()
    if not stage:
        raise service.PollerConfigError(service.STAGE_ENV, "is required")
    models = service.parse_models(env.get(service.MODELS_ENV))
    delay = service.test_delay(env, stage=stage)
    region = env.get("AWS_REGION") or None
    if store is None:
        names = {"modelstats": env.get("OWLBANK_TABLE_MODELSTATS") or "owlbank-modelstats",
                 "policy": env.get("OWLBANK_TABLE_POLICY") or "owlbank-policy"}
        store = DynamoStore(names, region_name=region, required=tuple(names))
    ids = [m.id for m in models]
    policy: list[Policy] = []  # read on first use, so an unreadable policy never fails INIT

    def read() -> Settings:
        if not policy:
            policy.append(Policy.from_store(store, ttl_s=POLICY_TTL_S))
        return policy[0].model_failover.settings_for(ids)

    settings = service.settings_source(read, ModelFailoverConfig({}, {}).settings_for(ids))
    return service.PollerDeps(store=store, bedrock=bedrock if bedrock is not None else _bedrock_client(region),
                              stage=stage, models=models, settings=settings, fixture=service.load_fixture(),
                              test_delay_s=delay)


def _remaining(context: Any) -> Callable[[], float] | None:
    get = getattr(context, "get_remaining_time_in_millis", None)
    if not callable(get):
        return None
    return lambda: get() / 1000


@obs.instrumented("model_poller", op="poll")
def handler(event: Any, context: Any = None) -> dict[str, Any]:
    global _deps
    if _deps is None:
        _deps = build_deps()
    reports = asyncio.run(service.run_invocation(_deps, _memory, remaining_s=_remaining(context)))
    return {"cycles": len(reports), "written": sum(1 for r in reports if r.written)}


# Inside Lambda only (a no-op in tests and scripts): wire up now, in the INIT phase.
_deps = coldstart.build_at_init(build_deps)
