"""The fleet poller: one verdict per model for the whole fleet.

One invocation (a one-minute schedule) runs :data:`CYCLES` decision cycles :data:`CYCLE_GAP_S` apart.
Each cycle, in order:

1. **Reads the state record** (one ``GetItem`` of :func:`~owlbank.failover.fleet.state_key`). It is the
   previous verdict, so a cold start continues from it, matched to the list by model tag
   (:func:`~owlbank.failover.fleet.align`: a model new to the list starts healthy, one that moved keeps
   its state). Missing: every entry healthy. Malformed: the same, logged, and overwritten.
2. **Sends the recovery pings** while the previous verdict has a degraded entry:
   ``pings_per_cycle`` ConverseStream requests, :func:`ping_spacing` apart, to the highest-ranked
   degraded entry (:func:`~owlbank.failover.fleet.ping_target`, never an unavailable one), built from
   ``ping_fixture.json`` (:func:`ping_request`: the entry's thinking and max-tokens parameters, no cache
   point). Each is timed to its first output (:func:`is_output`, as the bridge's guarded model) and
   abandoned at the entry's deadline (:func:`ping_deadline`; the last entry: its trip threshold x 2),
   which counts as a miss. The ``OWLBANK_FAILOVER_TEST`` delay holds back the preferred entry's first
   output as in the bridge.
3. **Rechecks unavailable entries** every ``unavailable_recheck_s``: the synthetic one-token
   request (:func:`recheck_request`); success marks the entry available.
4. **Reads the window** with one ``Query`` (:func:`~owlbank.failover.fleet.window_range`) and folds the
   cycle into the previous verdict with the pure :func:`~owlbank.failover.fleet.fleet_verdict`.
5. **Writes the state record every cycle** (the heartbeat: ``written_at`` moves on even when nothing
   changed; ``version`` grows only on a change). The write is conditional on the stored record being
   the one read in step 1 (its ``version``, or its absence), so an overlapping or repeated invocation
   never overwrites a newer verdict with an older one; it is a single-item ``TransactWriteItems`` (IAM:
   ``dynamodb:PutItem`` on the state record).

Failures: a table that can't be read means no write this cycle (no heartbeat, so the staleness
alarm fires) and the stored record stays as it was; a failed or conflicting write leaves the stored record
as it was too. Ping samples and recheck results of a cycle whose write failed are kept in memory and
used by the next cycle, so they are not lost.

Timing never relies on a timer firing at or after its due time (uvloop): every wait is followed by an
explicit comparison against the injected monotonic clock, and an early wake waits again for the rest.
Bedrock calls run in a thread; the waiting coroutine judges a ping by the clock, never by a timeout firing.

Logs carry model IDs, entry indexes, states, counts, milliseconds and error codes only: no caller text, no
trait value, nothing from a model's answer or an error message. The fixture carries no caller data.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import math
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal, TextIO

from owlbank.failover.classify import UNAVAILABLE, Classification, Outcome, classify
from owlbank.failover.fleet import (
    SAMPLES_PK,
    Bucket,
    Pings,
    State,
    align,
    fleet_verdict,
    initial_state,
    judged_pings,
    model_tag,
    model_tags,
    ping_target,
    state_key,
    window_range,
)
from owlbank.failover.rules import EntryState, Sample, Stats
from owlbank.failover.settings import MAX_ENTRIES, Settings
from owlbank.obs import emit, log_caught
from owlbank.store import ConditionFailed, Expect, Store, TxPut

log = logging.getLogger(__name__)

#: The logical table (``owlbank.store.FAILOVER_SCHEMAS``).
TABLE: Final = "modelstats"
#: Decision cycles per invocation and the time between their starts (EventBridge runs at most
#: once a minute, the verdict needs a cycle every 30 s).
CYCLES: Final = 2
CYCLE_GAP_S: Final = 30.0
#: Pings are spread over at most this much of a cycle, at most :data:`MAX_PING_SPACING_S` apart (about
#: 8 s for the default 3), so a cycle's pings end before the next cycle starts.
PING_SPREAD_S: Final = 24.0
MAX_PING_SPACING_S: Final = 8.0
#: The last entry has no deadline; its pings are abandoned at its trip threshold times this.
LAST_ENTRY_DEADLINE_FACTOR: Final = 2.0
#: A recheck (one-token Converse) that hasn't answered by then failed.
RECHECK_TIMEOUT_S: Final = 10.0
#: Slack for the table reads and the write in :func:`cycle_budget_s`.
TABLE_SLACK_S: Final = 5.0

FIXTURE_PATH: Final = Path(__file__).with_name("ping_fixture.json")

#: Environment (set on the function; the bridge reads the same names).
MODELS_ENV: Final = "OWLBANK_MODELS"
FAILOVER_TEST_ENV: Final = "OWLBANK_FAILOVER_TEST"
STAGE_ENV: Final = "OWLBANK_STAGE"
MAX_TEST_DELAY_S: Final = 30.0
PROD_STAGE: Final = "prod"

#: The model request's reasoning fields per thinking setting: the bridge's
#: ``agent_client_strands.THINKING_PARAMS`` (a parity test keeps them equal; the Lambda bundle has no
#: Strands).
THINKING_BETA: Final = "thinking-binding-controls-2026-08-01"
THINKING_PARAMS: Final[Mapping[str, Mapping[str, Any] | None]] = {
    "adaptive": {"additionalModelRequestFields": {
        "anthropic_beta": [THINKING_BETA],
        "thinking": {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}}},
    "off": {"additionalModelRequestFields": {"thinking": {"type": "disabled"}}},
    "default": None,
    # Sonnet 5.5 refuses ``disabled`` and names ``between_tools`` instead.
    "between_tools": {"additionalModelRequestFields": {"thinking": {"type": "between_tools"}}},
}
#: ``enabled:<budget>``: extended thinking with a token budget, at least :data:`MIN_THINKING_BUDGET`
#: and below the entry's ``max_tokens`` (the bridge's ``agent_client_strands.thinking_error``; a
#: parity test keeps the rules equal).
ENABLED_THINKING_PREFIX: Final = "enabled:"
MIN_THINKING_BUDGET: Final = 1024

_HARD_OUTCOMES: Final = frozenset({Outcome.THROTTLED, Outcome.TIMEOUT, Outcome.ERROR})


class PollerConfigError(ValueError):
    """The poller's configuration is unusable; names the setting, never its value."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"{name}: {reason}")
        self.name = name


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class ModelSpec:
    """One ``OWLBANK_MODELS`` entry, the fields a ping needs (``text_release`` and ``prompt_caching``
    are the bridge's; a ping never caches)."""

    id: str
    thinking: str
    max_tokens: int


def thinking_budget(thinking: str) -> int | None:
    """The budget of ``enabled:<budget>``; ``None`` for any other setting or a malformed budget."""
    if not thinking.startswith(ENABLED_THINKING_PREFIX):
        return None
    raw = thinking[len(ENABLED_THINKING_PREFIX):]
    return int(raw) if raw.isascii() and raw.isdigit() else None


def thinking_error(thinking: Any, max_tokens: int) -> str | None:
    """Why ``thinking`` is not a usable setting with ``max_tokens``, or ``None``: a
    :data:`THINKING_PARAMS` key, or ``enabled:<budget>`` with the budget at least
    :data:`MIN_THINKING_BUDGET` and below ``max_tokens``. Never echoes the value."""
    if not isinstance(thinking, str):
        return "thinking must be a string"
    if thinking in THINKING_PARAMS:
        return None
    if not thinking.startswith(ENABLED_THINKING_PREFIX):
        return "unknown thinking setting"
    budget = thinking_budget(thinking)
    if budget is None or budget < MIN_THINKING_BUDGET:
        return f"the thinking budget must be an integer of at least {MIN_THINKING_BUDGET}"
    if budget >= max_tokens:
        return "the thinking budget must be below max_tokens"
    return None


def thinking_fields(thinking: str) -> Mapping[str, Any] | None:
    """The request's reasoning fields for a thinking setting (``None``: send nothing), as the bridge's
    ``thinking_request_fields`` builds them (wrapped in ``additionalModelRequestFields``)."""
    budget = thinking_budget(thinking)
    if budget is not None:
        return {"additionalModelRequestFields": {"thinking": {"type": "enabled", "budget_tokens": budget}}}
    return THINKING_PARAMS[thinking]


def parse_models(raw: str | None) -> tuple[ModelSpec, ...]:
    """``OWLBANK_MODELS`` (a JSON list of ``{id, thinking, max_tokens, ...}``, one to three)."""
    try:
        entries = json.loads(raw or "")
    except ValueError:
        raise PollerConfigError(MODELS_ENV, "must be a JSON list") from None
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_ENTRIES:
        raise PollerConfigError(MODELS_ENV, f"must list one to {MAX_ENTRIES} entries")
    out: list[ModelSpec] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise PollerConfigError(MODELS_ENV, "entries must be objects")
        model_id, thinking, max_tokens = entry.get("id"), entry.get("thinking"), entry.get("max_tokens")
        if not isinstance(model_id, str) or not model_id:
            raise PollerConfigError(f"{MODELS_ENV}.id", "must be a model ID")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise PollerConfigError(f"{MODELS_ENV}.max_tokens", "must be a positive integer")
        error = thinking_error(thinking, max_tokens)
        if error is not None:
            raise PollerConfigError(f"{MODELS_ENV}.thinking", error)
        out.append(ModelSpec(model_id, str(thinking), max_tokens))
    if len({m.id for m in out}) != len(out):
        raise PollerConfigError(MODELS_ENV, "model IDs must differ")
    return tuple(out)


def test_delay(env: Mapping[str, str], *, stage: str) -> float | None:
    """The test-only first-output delay of the preferred entry (``OWLBANK_FAILOVER_TEST``, seconds):
    ``None`` when unset; refused at ``stage=prod`` and outside ``(0, 30]``, as the bridge's
    ``guarded_model.failover_test_delay`` (a parity test keeps them equal)."""
    raw = (env.get(FAILOVER_TEST_ENV) or "").strip()
    if not raw:
        return None
    if stage == PROD_STAGE:
        raise PollerConfigError(FAILOVER_TEST_ENV, "refused at stage=prod")
    try:
        value = float(raw)
    except ValueError:
        raise PollerConfigError(FAILOVER_TEST_ENV, "must be a number of seconds") from None
    if not math.isfinite(value) or not 0 < value <= MAX_TEST_DELAY_S:
        raise PollerConfigError(FAILOVER_TEST_ENV, f"must be seconds above 0 and at most {MAX_TEST_DELAY_S:g}")
    return value


test_delay.__test__ = False  # type: ignore[attr-defined]  # not a pytest test


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    """The ping fixture: ``{system: str, tools: [{toolSpec}], messages: [one user turn]}``."""
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(fixture, dict) or set(fixture) != {"system", "tools", "messages"}
            or not isinstance(fixture["system"], str) or not isinstance(fixture["tools"], list)
            or not isinstance(fixture["messages"], list) or len(fixture["messages"]) != 1):
        raise PollerConfigError("ping_fixture", "unexpected shape")
    return fixture


# --------------------------------------------------------------------------- requests


def ping_spacing(settings: Settings) -> float:
    """The time between a cycle's ping starts."""
    return min(MAX_PING_SPACING_S, PING_SPREAD_S / settings.pings_per_cycle)


def ping_deadline(settings: Settings, index: int) -> float:
    """When a ping to entry ``index`` is abandoned: its first-token deadline; the last entry, which has
    none, its trip threshold x :data:`LAST_ENTRY_DEADLINE_FACTOR`."""
    entry = settings.entry(index)
    return entry.deadline_s if entry.deadline_s is not None else entry.trip_ttft_s * LAST_ENTRY_DEADLINE_FACTOR


def cycle_budget_s(settings: Settings) -> float:
    """An upper bound of one cycle's duration: its pings, one recheck round and the table calls."""
    longest = max(ping_deadline(settings, i) for i in range(len(settings)))
    return (settings.pings_per_cycle - 1) * ping_spacing(settings) + longest + RECHECK_TIMEOUT_S + TABLE_SLACK_S


def ping_request(spec: ModelSpec, fixture: Mapping[str, Any]) -> dict[str, Any]:
    """The ConverseStream request of one ping: the fixture's system prompt, tools and turn, the entry's
    thinking and max-tokens parameters, and no cache point (prompt caching off)."""
    request: dict[str, Any] = {
        "modelId": spec.id,
        "system": [{"text": fixture["system"]}],
        "messages": json.loads(json.dumps(fixture["messages"])),
        "toolConfig": {"tools": json.loads(json.dumps(fixture["tools"]))},
        "inferenceConfig": {"maxTokens": spec.max_tokens},
    }
    extra = thinking_fields(spec.thinking)
    if extra:
        request.update(json.loads(json.dumps(extra)))
    return request


def recheck_request(spec: ModelSpec) -> dict[str, Any]:
    """The synthetic request of an unavailable entry's recheck: one token, no caller data, the
    same as preflight's."""
    return {"modelId": spec.id, "messages": [{"role": "user", "content": [{"text": "ok"}]}],
            "inferenceConfig": {"maxTokens": 1}}


def is_output(event: Mapping[str, Any]) -> bool:
    """A text delta or a tool use (its start or its input): the bridge's first output; reasoning,
    message and block boundaries and metadata are not. The same as ``guarded_model.is_output``."""
    delta = event.get("contentBlockDelta")
    if isinstance(delta, Mapping):
        inner = delta.get("delta")
        return isinstance(inner, Mapping) and ("text" in inner or "toolUse" in inner)
    start = event.get("contentBlockStart")
    if isinstance(start, Mapping):
        inner = start.get("start")
        return isinstance(inner, Mapping) and "toolUse" in inner
    return False


def _error_parts(exc: BaseException) -> tuple[str, str | None]:
    """The error's code (a boto ``Error.Code``, else the class name) and message (for
    :func:`classify` only; never kept or logged)."""
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error")
        if isinstance(error, Mapping) and error.get("Code"):
            message = error.get("Message")
            return str(error["Code"]), message if isinstance(message, str) else None
    return type(exc).__name__, None


@dataclass(frozen=True)
class CallResult:
    """What a Bedrock call in its thread saw: when the first output came (monotonic), or what the
    failure means (:func:`classify`) and its code. Never a message or any model text."""

    first_at: float | None = None
    cause: Classification | None = None
    code: str | None = None


def first_output(client: Any, request: Mapping[str, Any], *, cancel: threading.Event,
                 clock: Callable[[], float]) -> CallResult:
    """Send one ping and read its stream up to the first output (run in a thread). Stops between
    events once ``cancel`` is set and always closes the stream. Never raises."""
    try:
        response = client.converse_stream(**request)
    except Exception as exc:  # noqa: BLE001 - classified, never raised
        code, message = _error_parts(exc)
        return CallResult(cause=classify(code, message), code=code)
    stream = response.get("stream") if isinstance(response, Mapping) else None
    try:
        for event in stream or ():
            if cancel.is_set():
                return CallResult()
            if isinstance(event, Mapping) and is_output(event):
                return CallResult(first_at=clock())
        return CallResult(cause=Outcome.ERROR, code="NoOutput")  # the stream ended without output
    except Exception as exc:  # noqa: BLE001 - an event-stream error, classified
        code, message = _error_parts(exc)
        return CallResult(cause=classify(code, message), code=code)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):  # closing an abandoned stream is best effort
                close()


def recheck_call(client: Any, request: Mapping[str, Any]) -> CallResult:
    """Send the synthetic request (run in a thread): ``CallResult()`` on success. Never raises."""
    try:
        client.converse(**request)
    except Exception as exc:  # noqa: BLE001 - classified, never raised
        code, message = _error_parts(exc)
        return CallResult(cause=classify(code, message), code=code)
    return CallResult()


# --------------------------------------------------------------------------- dependencies


@dataclass
class PollerDeps:
    """Everything the poller touches, injected. ``settings`` gives the current failover settings for
    ``models`` (the policy section, read with a cache, never raising); ``wall`` is epoch seconds (the
    state record's times), ``mono`` measures durations, ``sleep`` waits (it may wake early)."""

    store: Store
    bedrock: Any
    stage: str
    models: tuple[ModelSpec, ...]
    settings: Callable[[], Settings]
    fixture: Mapping[str, Any]
    test_delay_s: float | None = None
    wall: Callable[[], float] = time.time
    mono: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    executor: Executor | None = None
    metrics_out: TextIO | None = None  # EMF lines; ``sys.stdout`` at write time


@dataclass
class PollerMemory:
    """What a warm Lambda environment keeps between cycles and invocations: when each entry was last
    rechecked, and the ping samples and recheck successes of a cycle whose write did not happen."""

    last_recheck: dict[int, float] = field(default_factory=dict)
    pending_pings: Pings | None = None
    pending_available: frozenset[int] = frozenset()


@dataclass(frozen=True)
class CycleReport:
    """One cycle: whether the table was read and the record written, the state written, and what was
    sent (for tests and the handler's summary)."""

    read: bool
    written: bool
    now_s: float | None = None
    previous: State | None = None
    state: State | None = None
    target: int | None = None
    pings: tuple[Sample, ...] = ()
    sent: int = 0
    available: frozenset[int] = frozenset()
    rechecked: tuple[int, ...] = ()
    buckets: tuple[Bucket, ...] = ()


_default_executor: ThreadPoolExecutor | None = None


def _executor(deps: PollerDeps) -> Executor:
    """The threads Bedrock calls run in. Not the loop's default executor: an abandoned ping's thread may
    outlive the event loop (``asyncio.run`` would wait for it)."""
    global _default_executor
    if deps.executor is not None:
        return deps.executor
    if _default_executor is None:
        _default_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="model-poller")
    return _default_executor


async def wait_until(deps: PollerDeps, due: float) -> None:
    """Sleep until ``mono() >= due``; a wake before it sleeps again for the rest."""
    while (remaining := due - deps.mono()) > 0:
        await deps.sleep(remaining)


async def _result_by(fut: asyncio.Future[CallResult], due: float, deps: PollerDeps) -> CallResult | None:
    """``fut``'s result if it is done by ``due`` on the injected clock, else ``None``. The wait may
    wake early; the clock decides."""
    while not fut.done():
        remaining = due - deps.mono()
        if remaining <= 0:
            break
        await asyncio.wait({fut}, timeout=remaining)
    return fut.result() if fut.done() else None


# --------------------------------------------------------------------------- pings and rechecks


async def send_ping(deps: PollerDeps, settings: Settings, index: int) -> Sample:
    """One recovery ping to entry ``index``: a latency sample (OK with its TTFT, or ABANDONED at the
    deadline), a hard-error sample, or a request error (never a sample, 4a.2)."""
    deadline = ping_deadline(settings, index)
    delay = deps.test_delay_s if index == 0 else None
    cancel = threading.Event()
    at_s = deps.wall()
    started = deps.mono()
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(_executor(deps), functools.partial(
        first_output, deps.bedrock, ping_request(deps.models[index], deps.fixture), cancel=cancel,
        clock=deps.mono))
    result = await _result_by(fut, started + deadline, deps)
    if result is None:
        cancel.set()  # the thread stops at its next event
        return Sample(at_s, Outcome.ABANDONED, deadline)
    if result.cause is not None:
        if result.cause == Outcome.REQUEST_ERROR:
            return Sample(at_s, Outcome.REQUEST_ERROR)
        # A hard error, or the model turned unavailable: either way not a fast first token (the
        # bridges' reports mark it unavailable, 4a.1).
        outcome = result.cause if result.cause in _HARD_OUTCOMES else Outcome.ERROR
        return Sample(at_s, outcome)  # type: ignore[arg-type]
    if result.first_at is None:  # cancelled: can't happen before the deadline
        return Sample(at_s, Outcome.ABANDONED, deadline)
    ttft = max(0.0, result.first_at - started)
    if delay is not None:
        ttft = max(ttft, delay)  # the test delay holds the first output back from the start
    if ttft > deadline:
        return Sample(at_s, Outcome.ABANDONED, deadline)
    return Sample(at_s, Outcome.OK, ttft)


async def send_pings(deps: PollerDeps, settings: Settings, index: int) -> tuple[Sample, ...]:
    """``pings_per_cycle`` pings to entry ``index``, :func:`ping_spacing` apart (one at a time)."""
    spacing = ping_spacing(settings)
    start = deps.mono()
    samples: list[Sample] = []
    for k in range(settings.pings_per_cycle):
        await wait_until(deps, start + k * spacing)
        samples.append(await send_ping(deps, settings, index))
    _log_pings(deps, index, samples)
    return tuple(samples)


async def recheck(deps: PollerDeps, index: int) -> bool:
    """The synthetic request to an unavailable entry: ``True`` when it answered."""
    loop = asyncio.get_running_loop()
    started = deps.mono()
    fut = loop.run_in_executor(_executor(deps), functools.partial(
        recheck_call, deps.bedrock, recheck_request(deps.models[index])))
    result = await _result_by(fut, started + RECHECK_TIMEOUT_S, deps)
    ok = result is not None and result.cause is None
    if not ok:
        emit(log, logging.INFO, "model_recheck_failed", entry=index, model_id=deps.models[index].id,
             cause="timeout" if result is None else str(result.cause), error_code=None if result is None else result.code)
    return ok


def rechecks_due(previous: State | None, memory: PollerMemory, settings: Settings, now_s: float) -> list[int]:
    """The unavailable entries whose last recheck (or, never rechecked here, whose unavailable mark) is
    at least ``unavailable_recheck_s`` old (an explicit clock comparison)."""
    if previous is None:
        return []
    due = []
    for m in previous.models:
        if m.state is not EntryState.UNAVAILABLE:
            continue
        last = memory.last_recheck.get(m.index, m.since_s if m.since_s is not None else -math.inf)
        if now_s - last >= settings.unavailable_recheck_s:
            due.append(m.index)
    return due


# --------------------------------------------------------------------------- the cycle


def _parse_buckets(items: Iterable[Mapping[str, Any]], settings: Settings) -> tuple[Bucket, ...]:
    """The window's sample items of the listed models (another model's, e.g. from bridges still on an
    older list, are left out)."""
    listed = set(model_tags(settings))
    out: list[Bucket] = []
    bad = 0
    for item in items:
        try:
            bucket = Bucket.from_item(item)
        except ValueError:
            bad += 1
            continue
        if bucket.model in listed:
            out.append(bucket)
    if bad:
        emit(log, logging.WARNING, "model_samples_malformed", items=bad)
    return tuple(out)


def _read_previous(deps: PollerDeps) -> tuple[bool, Mapping[str, Any] | None, State | None]:
    """``(readable, stored item, previous state)``; the previous state (as stored, not yet aligned to
    the list) is ``None`` when the record is missing or malformed."""
    try:
        item = deps.store.get(TABLE, state_key(), consistent=True)
    except Exception as exc:  # noqa: BLE001 - the table is unreadable: no write this cycle
        log_caught(log, "model_state_unreadable", exc, decision="keep_last_state")
        return False, None, None
    if item is None:
        return True, None, None
    try:
        state = State.from_item(item)
    except ValueError:
        emit(log, logging.WARNING, "model_state_malformed", decision="start_healthy")
        return True, item, None
    return True, item, state


def _condition(item: Mapping[str, Any] | None) -> Expect | Literal["not_exists"] | None:
    """The write's condition: the stored record is still the one read (its version), or still absent.
    A malformed record (no integer version) is overwritten."""
    if item is None:
        return "not_exists"
    version = item.get("version")
    if isinstance(version, int) and not isinstance(version, bool):
        return Expect("version", version, must_exist=True)
    return None


async def run_cycle(deps: PollerDeps, memory: PollerMemory) -> CycleReport:
    """One decision cycle (see the module docstring)."""
    settings = deps.settings()
    readable, item, stored = _read_previous(deps)
    if not readable:
        return CycleReport(read=False, written=False)
    previous = None
    if stored is not None:  # this list's indexes from here on
        previous = align(stored, settings)
        was = [m.model for m in stored.models]
        if was != [m.model for m in previous.models]:  # the list changed in a deploy
            emit(log, logging.WARNING, "model_state_realigned", entries=len(settings),
                 kept=sum(1 for m in previous.models if m.model in was))

    target = ping_target(previous) if previous is not None else None
    carried: tuple[Sample, ...] = ()
    if memory.pending_pings is not None and memory.pending_pings.index == target:
        carried = memory.pending_pings.samples
    sent = await send_pings(deps, settings, target) if target is not None else ()
    pings = (*carried, *sent)

    due = rechecks_due(previous, memory, settings, deps.wall())
    succeeded = {i for i in due if await recheck(deps, i)}
    available = frozenset({*succeeded, *(i for i in memory.pending_available
                                         if previous is not None
                                         and previous.models[i].state is EntryState.UNAVAILABLE)})

    def keep_for_next() -> None:
        memory.pending_pings = Pings(target, pings) if target is not None and pings else None
        memory.pending_available = available

    now = deps.wall()
    low, high = window_range(now, settings)
    try:
        items = deps.store.query_range(TABLE, {"s": SAMPLES_PK}, low=low, high=high)
    except Exception as exc:  # noqa: BLE001 - the table is unreadable: no write this cycle
        log_caught(log, "model_samples_unreadable", exc, decision="keep_last_state")
        keep_for_next()
        return CycleReport(read=False, written=False, previous=previous, target=target, pings=pings,
                           sent=len(sent), available=available, rechecked=tuple(due))
    buckets = _parse_buckets(items, settings)

    state = fleet_verdict(buckets, settings, now, stored, pings=pings, available=available)
    if previous is None and item is not None:
        stored = item.get("version")
        if isinstance(stored, int) and not isinstance(stored, bool) and state.version <= stored:
            state = replace(state, version=stored + 1)  # a reset is a change: versions never go back
    emit_metric(deps, USABLE_ENTRIES_METRIC, usable_entries(state), now_s=now)
    report = CycleReport(read=True, written=False, now_s=now, previous=previous, state=state, target=target,
                         pings=pings, sent=len(sent), available=available, rechecked=tuple(due),
                         buckets=buckets)
    try:
        deps.store.transact_write([TxPut(TABLE, state.to_item(), condition=_condition(item))])
    except ConditionFailed:  # another invocation wrote since the read: its record stands
        emit(log, logging.WARNING, "model_state_conflict", decision="keep_stored_state")
        keep_for_next()
        return report
    except Exception as exc:  # noqa: BLE001 - a failed write keeps the stored record
        log_caught(log, "model_state_write_failed", exc, decision="keep_last_state")
        keep_for_next()
        return report
    memory.pending_pings, memory.pending_available = None, frozenset()
    for i in due:
        memory.last_recheck[i] = now
    old = previous or initial_state(settings, now)
    judged = judged_pings(previous, settings, now, pings)  # the pruned set the verdict judged
    _log_transitions(deps, old, state, buckets, ping_count=len(judged))
    emit(log, logging.INFO, "model_state_written", version=state.version,
         changed=previous is None or state.version != previous.version,
         report_every=state.report_every, pinged=target is not None)
    emit_metric(deps, HEARTBEAT_METRIC, 1, now_s=now)
    return replace(report, written=True)


async def run_invocation(deps: PollerDeps, memory: PollerMemory, *, cycles: int = CYCLES,
                         gap_s: float = CYCLE_GAP_S,
                         remaining_s: Callable[[], float] | None = None) -> list[CycleReport]:
    """``cycles`` decision cycles starting ``gap_s`` apart (by the injected clock). A cycle that would
    not finish in the invocation's remaining time is skipped (the next invocation runs it)."""
    start = deps.mono()
    reports: list[CycleReport] = []
    for c in range(cycles):
        if c:
            await wait_until(deps, start + c * gap_s)
        if remaining_s is not None and remaining_s() < cycle_budget_s(deps.settings()):
            emit(log, logging.WARNING, "model_poller_cycle_skipped", cycle=c, reason="time")
            break
        reports.append(await run_cycle(deps, memory))
    return reports


# --------------------------------------------------------------------------- metrics (EMF)

#: The poller's metrics (alarms watch them): ``StateHeartbeat`` 1 per cycle whose state write
#: succeeded (nothing when it failed or the table was unreadable: the staleness alarm treats missing
#: data as breaching); ``UsableEntries``, the entries neither degraded nor unavailable in the verdict
#: the cycle computed (every cycle that read the table, whether or not its write succeeded).
HEARTBEAT_METRIC: Final = "StateHeartbeat"
USABLE_ENTRIES_METRIC: Final = "UsableEntries"
METRIC_DIMENSIONS: Final = ("Service", "Environment")


def metric_namespace(stage: str) -> str:
    return f"OwlBank/{stage}/Model"


def metric_service(stage: str) -> str:
    return f"owlbank-{stage}-model-poller"


def usable_entries(state: State) -> int:
    return sum(1 for m in state.models if m.state is EntryState.HEALTHY)


def emf_record(stage: str, name: str, value: int | float, *, now_s: float, unit: str = "Count") -> dict[str, Any]:
    """One CloudWatch EMF object: the shape of ``owlbank.telemetry.exporters.EmfStdoutMetricExporter``
    (``Service`` and ``Environment`` dimensions), without its OpenTelemetry dependency."""
    return {
        "_aws": {"Timestamp": round(now_s * 1000), "CloudWatchMetrics": [{
            "Namespace": metric_namespace(stage), "Dimensions": [list(METRIC_DIMENSIONS)],
            "Metrics": [{"Name": name, "Unit": unit}]}]},
        "Service": metric_service(stage),
        "Environment": stage,
        name: value,
    }


def emit_metric(deps: PollerDeps, name: str, value: int | float, *, now_s: float) -> None:
    """Write one EMF line straight to stdout, not through ``logging``: with the Lambda JSON log format a
    logged line is wrapped in the runtime's own JSON and CloudWatch no longer extracts it, while a raw
    stdout line that is a JSON object is (AWS Lambda logging docs). Never raises."""
    try:
        out = deps.metrics_out if deps.metrics_out is not None else sys.stdout
        out.write(json.dumps(emf_record(deps.stage, name, value, now_s=now_s), separators=(",", ":"),
                             allow_nan=False) + "\n")
        out.flush()
    except Exception as exc:  # noqa: BLE001 - a metric must never fail a cycle
        log_caught(log, "metric_export_failed", exc, decision="drop_metric")


# --------------------------------------------------------------------------- settings and logs


def settings_source(read: Callable[[], Settings], fallback: Settings) -> Callable[[], Settings]:
    """A ``settings`` callable that never raises: the policy section's settings, else the last good
    ones, else ``fallback`` (the defaults for the deployed list)."""
    last: list[Settings] = [fallback]

    def current() -> Settings:
        try:
            settings = read()
        except Exception as exc:  # noqa: BLE001 - the policy is unreadable: keep deciding
            log_caught(log, "model_failover_settings_unreadable", exc, decision="last_settings")
            return last[0]
        if len(settings) != len(fallback):  # pragma: no cover - settings_for keeps the list's length
            return last[0]
        last[0] = settings
        return settings

    return current


def _ms(seconds: float | None) -> int | None:
    if seconds is None or math.isinf(seconds):
        return None
    return round(seconds * 1000)


def _log_pings(deps: PollerDeps, index: int, samples: Sequence[Sample]) -> None:
    stats = Stats.of(samples)
    emit(log, logging.INFO, "model_pings", entry=index, model_id=deps.models[index].id, sent=len(samples),
         ok=sum(1 for s in samples if s.outcome is Outcome.OK),
         missed=sum(1 for s in samples if s.outcome is Outcome.ABANDONED),
         errors=stats.errors, p90_ms=_ms(stats.p90_s))


def _log_transitions(deps: PollerDeps, old: State, new: State, buckets: Collection[Bucket], *,
                     ping_count: int) -> None:
    """``model_degraded``, ``model_recovered``, ``model_unavailable``, ``model_available``:
    model IDs, numbers and causes only."""
    for before, after in zip(old.models, new.models, strict=True):
        if before.state is after.state:
            continue
        i = after.index
        common: dict[str, Any] = {"entry": i, "model_id": deps.models[i].id, "rank": after.rank,
                                  "source": "poller"}
        if after.state is EntryState.UNAVAILABLE:
            emit(log, logging.ERROR, "model_unavailable", preferred=i == 0, **common)
        elif before.state is EntryState.UNAVAILABLE:
            emit(log, logging.INFO, "model_available", **common)
        elif after.state is EntryState.DEGRADED:
            emit(log, logging.WARNING, "model_degraded", p50_ms=_ms(after.p50_s), p90_ms=_ms(after.p90_s),
                 error_permille=None if after.error_rate is None else round(after.error_rate * 1000),
                 samples=sum(b.raw for b in buckets if b.model == model_tag(deps.models[i].id)), **common)
        else:
            emit(log, logging.INFO, "model_recovered", p50_ms=_ms(after.p50_s), p90_ms=_ms(after.p90_s),
                 error_permille=None if after.error_rate is None else round(after.error_rate * 1000),
                 pings=ping_count,
                 **common)


__all__ = [
    "CYCLES",
    "CYCLE_GAP_S",
    "TABLE",
    "UNAVAILABLE",
    "CallResult",
    "CycleReport",
    "ModelSpec",
    "PollerConfigError",
    "PollerDeps",
    "PollerMemory",
    "cycle_budget_s",
    "first_output",
    "is_output",
    "load_fixture",
    "parse_models",
    "ping_deadline",
    "ping_request",
    "ping_spacing",
    "recheck_request",
    "run_cycle",
    "run_invocation",
    "settings_source",
    "test_delay",
]
