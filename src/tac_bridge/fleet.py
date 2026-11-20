"""The bridge's side of the fleet picture: :class:`FleetReporter` and :class:`FleetReader` (model
failover).

* :class:`FleetReporter` keeps one :class:`~owlbank.failover.fleet.Counters` per model and flushes them
  in the background, one ``Store.add_counters`` per model per flush (one write unit, the item well
  under 1 KB): every :data:`FLUSH_INTERVAL_S` (Fargate: per task, 10 s; AgentCore, where a process is
  one call: 30 s, plus :meth:`FleetReporter.stop` at call end). An OK call's latency is reported with
  the probability ``1 / report_every`` that the poller publishes in the state record, with the weight
  ``report_every``, so the totals stay unbiased; hard errors, deadline misses and unavailable detections
  are always counted (weight 1) and reach the store in the next flush, so an outage burst is still one
  write per model per flush. Request errors are never counted. A flush's counters go to the
  bucket of the newest call they hold (a later bucket, never an earlier one, so an unavailable detection
  is never dated before the verdict it follows). Items name the model by its tag
  (:func:`~owlbank.failover.fleet.model_tag`), not its place in the list. **Re-reports**: every
  flush also counts one unavailable detection for each entry this process holds unavailable that the
  fleet view hasn't shown unavailable yet (``unconfirmed``, :meth:`FailoverState.unconfirmed
  <owlbank.failover.core.FailoverState.unconfirmed>`), so a lost report (a dropped write, a poller
  outage longer than the window) is made good by the next flush.
* :class:`FleetReader` reads the state record (one eventually consistent ``GetItem``) every
  :data:`READ_INTERVAL_S` in the background and hands the bridge the cached verdict as a
  :class:`~owlbank.failover.core.FleetView` in this process's list order, matched by model tag, or
  ``None`` when the record is missing, malformed, has no verdict for one of this process's models (the
  list changed in a deploy and the poller hasn't caught up), or is older than :data:`STALE_AFTER_S`
  The bridge then uses its in-process picture. A read that fails keeps the last good record until it goes
  stale.
  :meth:`FleetReader.report_every` is the weight the reporter samples with (the last one read; 1 until
  a record has been read).

**Off the call path**. :meth:`FleetReporter.record` and :meth:`FleetReader.view` are
synchronous, do no I/O and never raise. Store calls (boto3, blocking) run on each object's own
one-thread executor, never the loop's default one (Strands streams Bedrock through it), under a timeout
(:data:`WRITE_TIMEOUT_S`, :data:`READ_TIMEOUT_S`); a timed-out call is left to finish in its thread and
the next flush or read is skipped while it does, so a stuck store never piles up threads. A failed or
timed-out write is dropped (a debug line, never retried): only that flush's counters are lost, and calls
recorded meanwhile wait for the next flush.

**Timers** (uvloop timers can fire early). The periodic loops keep their due time on the monotonic
clock and sleep again when woken before it, so a flush or read never comes sooner than its interval.

**No caller data**: a report holds a model index, an outcome and a time to first token, and
the items hold only the counter attributes; nothing here takes or logs text.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import random
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from owlbank.failover.classify import UNAVAILABLE, Classification, Outcome
from owlbank.failover.core import FleetView
from owlbank.failover.fleet import (
    MAX_REPORT_EVERY,
    Counters,
    State,
    bucket_start,
    model_tag,
    sample_expires_at,
    sample_key,
    state_key,
)
from owlbank.failover.settings import MAX_ENTRIES
from owlbank.obs import emit, log_caught
from owlbank.store import Store
from tac_bridge.hosting import Hosting

log = logging.getLogger(__name__)

TABLE: Final = "modelstats"
#: The table's name variable (infra ``model_failover.TABLE_ENV``): set only with ``modelFailover=on``
#: and two or more entries, so its absence means no reporter or reader.
TABLE_ENV: Final = f"OWLBANK_TABLE_{TABLE.upper()}"
#: Fargate flushes per task every 10 s; AgentCore (one call per process) every 30 s and at call end.
FLUSH_INTERVAL_S: Final[dict[Hosting, float]] = {"fargate": 10.0, "agentcore": 30.0}
#: The verdict is read (and cached) every 30 s.
READ_INTERVAL_S: Final = 30.0
#: A record older than this is not used.
STALE_AFTER_S: Final = 180.0
WRITE_TIMEOUT_S: Final = 2.0
READ_TIMEOUT_S: Final = 1.0
#: The shortest re-sleep when a timer fires before its due time (so an early timer never spins).
_MIN_SLEEP_S: Final = 0.001


@dataclass
class _Pending:
    counters: Counters = field(default_factory=Counters)
    newest_s: float = 0.0  # wall clock of the newest call recorded


@dataclass(frozen=True)
class Write:
    """One sample update: the key, the counter increments and the TTL default."""

    key: dict[str, str]
    counters: dict[str, int]
    expires_at: int


class _Offloaded:
    """Runs blocking store calls on its own one-thread executor, one at a time, under a timeout."""

    def __init__(self, name: str) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._inflight: concurrent.futures.Future[Any] | None = None

    @property
    def busy(self) -> bool:
        return self._inflight is not None and not self._inflight.done()

    async def run(self, fn: Callable[[], Any], timeout_s: float) -> Any:
        """``fn()`` in the thread; :class:`TimeoutError` after ``timeout_s`` (the thread carries on)."""
        future = self._executor.submit(fn)
        self._inflight = future
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout_s)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class _Periodic:
    """Calls ``step`` every ``interval_s`` (monotonic), never before its due time."""

    def __init__(self, step: Callable[[], Awaitable[Any]], interval_s: float, clock: Callable[[], float],
                 *, first_at_once: bool) -> None:
        self._step, self._interval, self._clock, self._first = step, interval_s, clock, first_at_once
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if not self.running:
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _loop(self) -> None:
        due = self._clock() if self._first else self._clock() + self._interval
        while True:
            now = self._clock()
            if now < due:  # first wait, or a timer that fired early: sleep again, never return early
                await asyncio.sleep(max(due - now, _MIN_SLEEP_S))
                continue
            await self._step()  # never raises (each step catches its own errors)
            due = max(due + self._interval, self._clock())


#: The periodic loop, for the bridge's other background timers (``model_metrics``).
Periodic = _Periodic


def _tags(model_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(model_ids, str) or not 1 <= len(model_ids) <= MAX_ENTRIES:
        raise ValueError("model_ids must list 1 to 3 models")
    tags = tuple(model_tag(m) for m in model_ids)
    if len(set(tags)) != len(tags):
        raise ValueError("model_ids must differ")
    return tags


def _clamp_every(k: Any) -> int:
    if not isinstance(k, int) or isinstance(k, bool):
        return 1
    return min(MAX_REPORT_EVERY, max(1, k))


class FleetReporter:
    """This process's model-call counters and their background flush."""

    def __init__(self, store: Store, *, model_ids: Sequence[str], hosting: Hosting = "fargate",
                 report_every: Callable[[], int] | None = None,
                 unconfirmed: Callable[[], Iterable[int]] | None = None,
                 rng: Callable[[], float] = random.random,  # noqa: S311 (sampling, not security)
                 wall: Callable[[], float] = time.time, clock: Callable[[], float] = time.monotonic,
                 interval_s: float | None = None, write_timeout_s: float = WRITE_TIMEOUT_S) -> None:
        self._tags = _tags(model_ids)
        self._store, self._entries = store, len(self._tags)
        self._report_every = report_every or (lambda: 1)
        self._unconfirmed = unconfirmed
        self._rng, self._wall = rng, wall
        self._timeout = write_timeout_s
        self._pending: dict[int, _Pending] = {}
        self._io = _Offloaded("fleet-report")
        self._loop = _Periodic(self.flush, FLUSH_INTERVAL_S[hosting] if interval_s is None else interval_s,
                               clock, first_at_once=False)

    # ------------------------------------------------------------------ the call path

    def record(self, index: int, outcome: Classification, ttft_s: float | None = None) -> None:
        """One model call on entry ``index``: its outcome (as :func:`~owlbank.failover.classify.classify`
        or the guarded model gives it) and, for an OK or abandoned call, its time to first token (for an
        abandoned call, the deadline it missed). Synchronous, no I/O, never raises."""
        try:
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < self._entries:
                raise ValueError("index")
            if outcome is Outcome.REQUEST_ERROR:
                return  # never a sample
            if outcome == UNAVAILABLE:
                self._slot(index).counters.record_unavailable()
                return
            if not isinstance(outcome, Outcome):
                raise ValueError("outcome")
            weight = 1
            if outcome is Outcome.OK:
                k = _clamp_every(self._report_every())
                if k > 1 and not self._rng() < 1.0 / k:
                    return  # not sampled this time
                weight = k
            pending = self._pending.get(index)
            counters = pending.counters if pending is not None else Counters()
            counters.record(outcome, ttft_s, weight=weight)  # validates before changing anything
            self._slot(index).counters = counters
        except Exception as exc:  # the call must never fail because of a report
            log_caught(log, "fleet_record_dropped", exc, decision="drop", level=logging.DEBUG)

    def _slot(self, index: int) -> _Pending:
        slot = self._pending.setdefault(index, _Pending())
        slot.newest_s = max(slot.newest_s, self._wall())
        return slot

    # ------------------------------------------------------------------ the flush

    def _re_report(self) -> None:
        """One unavailable detection for each unconfirmed entry not already counted in this flush."""
        if self._unconfirmed is None:
            return
        try:
            indexes = sorted(set(self._unconfirmed()))
            if not all(isinstance(i, int) and 0 <= i < self._entries for i in indexes):
                raise ValueError("index")
        except Exception as exc:  # a flush never fails because of it
            log_caught(log, "fleet_rereport_failed", exc, decision="skip", level=logging.DEBUG)
            return
        for index in indexes:
            slot = self._slot(index)  # dated now: never before the verdict it follows
            if not slot.counters.unavailable:
                slot.counters.record_unavailable()

    def drain(self) -> list[Write]:
        """Take every pending counter as one write per model (the pending set starts empty again), with
        the re-reports of unconfirmed unavailable entries."""
        self._re_report()
        pending, self._pending = self._pending, {}
        writes = []
        for index, slot in sorted(pending.items()):
            if slot.counters.empty:
                continue
            bucket = bucket_start(slot.newest_s)
            writes.append(Write(key=sample_key(bucket, self._tags[index]),
                                counters=slot.counters.increments(), expires_at=sample_expires_at(bucket)))
        return writes

    def _write_all(self, writes: list[Write]) -> int:
        """In the executor's thread: each write on its own, a failed one dropped. Returns the count done."""
        done = 0
        for w in writes:
            try:
                self._store.add_counters(TABLE, w.key, w.counters, defaults={"expires_at": w.expires_at})
                done += 1
            except Exception as exc:
                log_caught(log, "fleet_report_dropped", exc, decision="drop", level=logging.DEBUG)
        return done

    async def flush(self) -> int:
        """Write the pending counters now, off the loop. Returns the writes made; never raises.

        While an earlier write is still running (a slow store), nothing is taken: the counters wait for
        the next flush."""
        try:
            if self._io.busy:
                emit(log, logging.DEBUG, "fleet_report_deferred", cause="busy")
                return 0
            writes = self.drain()
            if not writes:
                return 0
            return int(await self._io.run(lambda: self._write_all(writes), self._timeout))
        except TimeoutError:
            emit(log, logging.DEBUG, "fleet_report_dropped", cause="timeout")
        except Exception as exc:
            log_caught(log, "fleet_report_dropped", exc, decision="drop", level=logging.DEBUG)
        return 0

    def start(self) -> None:
        """Start the background flush on the running loop (idempotent)."""
        self._loop.start()

    async def stop(self, *, final_flush: bool = True) -> None:
        """Stop the background flush and, by default, flush what is pending (AgentCore: at call end).
        Bounded by the write timeout; never raises."""
        try:
            await self._loop.stop()
            if final_flush:
                await self.flush()
        except Exception as exc:
            log_caught(log, "fleet_report_dropped", exc, decision="drop", level=logging.DEBUG)
        finally:
            self._io.close()


class FleetReader:
    """The cached fleet verdict with the staleness fallback."""

    def __init__(self, store: Store, *, model_ids: Sequence[str],
                 wall: Callable[[], float] = time.time, clock: Callable[[], float] = time.monotonic,
                 interval_s: float = READ_INTERVAL_S, stale_after_s: float = STALE_AFTER_S,
                 read_timeout_s: float = READ_TIMEOUT_S) -> None:
        self._tags = _tags(model_ids)
        self._store = store
        self._wall, self._stale, self._timeout = wall, stale_after_s, read_timeout_s
        self._state: State | None = None
        self._every = 1
        self._io = _Offloaded("fleet-read")
        self._loop = _Periodic(self.refresh, interval_s, clock, first_at_once=True)

    def view(self) -> FleetView | None:
        """The verdict to follow in this process's list order, or ``None`` (missing, malformed, without
        one of this list's models, or stale): use the in-process picture. Synchronous, no I/O, never
        raises."""
        try:
            state = self._state
            if state is None or abs(self._wall() - state.written_at_s) > self._stale:
                return None
            by_model = {m.model: m for m in state.models}
            ordered = [by_model[tag] for tag in self._tags]  # refresh kept only records naming every one
            return FleetView(states=tuple(m.state for m in ordered), p90_s=tuple(m.p90_s for m in ordered))
        except Exception as exc:
            log_caught(log, "fleet_view_failed", exc, decision="fallback_local", level=logging.DEBUG)
            return None

    def report_every(self) -> int:
        """The reporter's sampling weight: the last one read (1 before any record). Kept when the record
        goes stale, so a poller outage doesn't raise the fleet's write rate."""
        return self._every

    def _read(self) -> Any:
        return self._store.get(TABLE, state_key(), consistent=False)

    async def refresh(self) -> None:
        """One read of the state record, off the loop; never raises. Skipped while an earlier read is
        still running."""
        if self._io.busy:
            emit(log, logging.DEBUG, "fleet_read_deferred", cause="busy")
            return
        try:
            item = await self._io.run(self._read, self._timeout)
        except TimeoutError:
            emit(log, logging.DEBUG, "fleet_read_failed", cause="timeout")
            return
        except Exception as exc:  # keep the last good record until it goes stale
            log_caught(log, "fleet_read_failed", exc, decision="keep_cached", level=logging.DEBUG)
            return
        if item is None:
            self._state = None
            emit(log, logging.DEBUG, "fleet_read_failed", cause="missing")
            return
        try:
            state = State.from_item(item)
        except (ValueError, TypeError) as exc:
            self._state = None
            log_caught(log, "fleet_read_failed", exc, decision="fallback_local", level=logging.DEBUG)
            return
        if not set(self._tags) <= {m.model for m in state.models}:
            self._state = None  # the list changed in a deploy; the poller hasn't caught up
            emit(log, logging.DEBUG, "fleet_read_failed", cause="models")
            return
        self._state = state
        self._every = _clamp_every(state.report_every)

    def start(self) -> None:
        """Read now, then every interval, on the running loop (idempotent)."""
        self._loop.start()

    async def stop(self) -> None:
        with contextlib.suppress(Exception):
            await self._loop.stop()
        self._io.close()


__all__ = ["FLUSH_INTERVAL_S", "READ_INTERVAL_S", "STALE_AFTER_S", "TABLE", "TABLE_ENV", "FleetReader",
           "FleetReporter", "Periodic", "Write"]
