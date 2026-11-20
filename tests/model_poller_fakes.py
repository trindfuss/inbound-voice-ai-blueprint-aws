"""Fakes for the model-failover poller's tests: a fake clock and sleeper (which can wake early, as
uvloop's timers do), an inline executor, a scripted Bedrock client and a store that can fail. No
network."""

from __future__ import annotations

import asyncio
import io
import json
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from typing import Any

from lambdas.model_poller import service
from owlbank.failover.classify import Outcome
from owlbank.failover.fleet import Counters, bucket_start, model_tag, sample_expires_at, sample_key, state_key
from owlbank.failover.rules import EntryState
from owlbank.failover.settings import Settings
from owlbank.store import MemoryStore, StoreError

STAGE = "dev"
MODEL_IDS = ("us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5-20251001-v1:0",
             "us.anthropic.claude-sonnet-5-5")
START_WALL = 1_800_000_000.0


class FakeTime:
    """One time base for ``wall`` (epoch seconds) and ``mono``. ``sleep`` advances it, waking
    ``early_s`` before the due time when the wait is longer than that."""

    def __init__(self, early_s: float = 0.0) -> None:
        self.t = 0.0
        self.early_s = early_s
        self.sleeps: list[float] = []

    def wall(self) -> float:
        return START_WALL + self.t

    def mono(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds - self.early_s if seconds > self.early_s else seconds
        await asyncio.sleep(0)


class InlineExecutor(Executor):
    """Runs the submitted function at once, on the caller's thread."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        fut: Future[Any] = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - handed to the future
            fut.set_exception(exc)
        return fut


class FakeClientError(Exception):
    """Shaped like botocore's ClientError (``response["Error"]``)."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": message}}


@dataclass(frozen=True)
class Ping:
    """One scripted ping answer: the first output after ``ttft_s`` (fake time), or an error."""

    ttft_s: float = 0.5
    error: str | None = None
    message: str = ""


class FakeStream:
    def __init__(self, events: Iterable[Mapping[str, Any]]) -> None:
        self._events = iter(events)
        self.closed = False

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return self._events

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeBedrock:
    """``converse_stream`` plays the next scripted :class:`Ping` of the model (default: fast), advancing
    the fake time by its TTFT before the first output; ``converse`` (the recheck) succeeds unless the
    model is listed in ``recheck_fails``."""

    time: FakeTime
    script: dict[str, deque[Ping]] = field(default_factory=lambda: defaultdict(deque))
    default: dict[str, Ping] = field(default_factory=dict)
    recheck_fails: set[str] = field(default_factory=set)
    pings: list[tuple[str, float, dict[str, Any]]] = field(default_factory=list)  # (model, wall, request)
    played: list[tuple[str, float, Ping]] = field(default_factory=list)  # (model, wall, scripted answer)
    rechecks: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    streams: list[FakeStream] = field(default_factory=list)

    def converse_stream(self, **request: Any) -> dict[str, Any]:
        model = request["modelId"]
        self.pings.append((model, self.time.wall(), request))
        queue = self.script.get(model)
        ping = queue.popleft() if queue else self.default.get(model, Ping())
        self.played.append((model, self.time.wall(), ping))
        if ping.error is not None:
            raise FakeClientError(ping.error, ping.message)
        ttft = ping.ttft_s

        def events() -> Iterator[Mapping[str, Any]]:
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockDelta": {"delta": {"reasoningContent": {"text": ""}}}}
            self.time.advance(ttft)
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t", "name": "get_balance"}}}}
            yield {"messageStop": {"stopReason": "tool_use"}}

        stream = FakeStream(events())
        self.streams.append(stream)
        return {"stream": stream}

    def converse(self, **request: Any) -> dict[str, Any]:
        self.rechecks.append((request["modelId"], request))
        if request["modelId"] in self.recheck_fails:
            raise FakeClientError("ValidationException", "The provided model identifier is invalid.")
        return {"output": {"message": {"role": "assistant", "content": [{"text": "o"}]}}}

    def pinged_models(self) -> list[str]:
        return [m for m, _, _ in self.pings]


class FlakyStore(MemoryStore):
    """A :class:`MemoryStore` whose reads or writes can be made to fail (``StoreError``)."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_get = False
        self.fail_query = False
        self.fail_write = False
        self.writes = 0

    def get(self, table: str, key: Mapping[str, Any], *, consistent: bool = True) -> Any:
        if self.fail_get:
            raise StoreError("unreadable")
        return super().get(table, key, consistent=consistent)

    def query_range(self, table: str, partition: Mapping[str, Any], *, low: str | int, high: str | int,
                    consistent: bool = False, index: str | None = None) -> Any:
        if self.fail_query:
            raise StoreError("unreadable")
        return super().query_range(table, partition, low=low, high=high, consistent=consistent, index=index)

    def transact_write(self, puts: Sequence[Any]) -> None:
        if self.fail_write:
            raise StoreError("unwritable")
        super().transact_write(puts)
        self.writes += 1


def models_env(ids: Sequence[str] = MODEL_IDS, thinking: str = "adaptive", max_tokens: int = 1024) -> str:
    return json.dumps([{"id": m, "thinking": thinking, "max_tokens": max_tokens, "text_release": "stream",
                        "prompt_caching": True} for m in ids])


def make_deps(*, ids: Sequence[str] = MODEL_IDS, store: FlakyStore | None = None, early_s: float = 0.0,
              settings: Settings | None = None, test_delay_s: float | None = None,
              **overrides: Any) -> tuple[service.PollerDeps, FakeTime, FakeBedrock, FlakyStore]:
    time = FakeTime(early_s)
    bedrock = FakeBedrock(time)
    store = store if store is not None else FlakyStore()
    s = settings if settings is not None else Settings.for_models(list(ids), **overrides)
    deps = service.PollerDeps(store=store, bedrock=bedrock, stage=STAGE,
                              models=service.parse_models(models_env(ids)), settings=lambda: s,
                              fixture=service.load_fixture(), test_delay_s=test_delay_s,
                              wall=time.wall, mono=time.mono, sleep=time.sleep, executor=InlineExecutor(),
                              metrics_out=io.StringIO())
    return deps, time, bedrock, store


def add_samples(store: MemoryStore, at_wall: float, index: int, ttfts: Iterable[float], *,
                weight: int = 1, errors: int = 0, unavailable: int = 0, misses: Sequence[float] = (),
                ids: Sequence[str] = MODEL_IDS) -> None:
    """Report latency samples (and errors, misses, unavailable detections) for model ``ids[index]`` at
    ``at_wall``, as a bridge's flush would."""
    counters = Counters()
    for t in ttfts:
        counters.record(Outcome.OK, t, weight=weight)
    for t in misses:
        counters.record(Outcome.ABANDONED, t)
    for _ in range(errors):
        counters.record(Outcome.ERROR, None)
    for _ in range(unavailable):
        counters.record_unavailable()
    if counters.empty:
        return
    b = bucket_start(at_wall)
    store.add_counters(service.TABLE, sample_key(b, model_tag(ids[index])), counters.increments(),
                       defaults={"expires_at": sample_expires_at(b)})


def stored_state(store: MemoryStore) -> Any:
    from owlbank.failover.fleet import State

    item = store.get(service.TABLE, state_key())
    return None if item is None else State.from_item(item)


def states(store: MemoryStore) -> list[EntryState] | None:
    s = stored_state(store)
    return None if s is None else [m.state for m in s.models]


def metrics(deps: service.PollerDeps) -> list[dict[str, Any]]:
    """The EMF objects the poller wrote (one JSON object per line)."""
    out = deps.metrics_out
    assert isinstance(out, io.StringIO)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def ev(record: Any, name: str = "event") -> Any:
    """A structured log field of a record (``obs.emit`` puts them on the record)."""
    return record.__dict__.get(name)


def run(coro_fn: Callable[[], Any], loop_name: str = "asyncio") -> Any:
    if loop_name == "uvloop":
        import uvloop

        factory: Callable[[], asyncio.AbstractEventLoop] = uvloop.new_event_loop
    else:
        factory = asyncio.new_event_loop
    with asyncio.Runner(loop_factory=factory) as runner:
        return runner.run(coro_fn())
