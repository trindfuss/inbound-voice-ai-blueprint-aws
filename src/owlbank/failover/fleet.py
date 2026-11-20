"""The fleet picture's pure core.

* **Table keys**: samples live in the fixed partition ``s = samples`` under the sort key
  ``<bucket_start>#<model_tag>``, the bucket start being epoch seconds floored to :data:`BUCKET_S`
  and zero-padded to ten digits, so sort keys order by time and :func:`window_range` gives the one
  ``BETWEEN`` condition that reads the window and leaves older (expired but undeleted) buckets out.
  The verdict is the single item ``{s: state, sk: state}``. The partitions are fixed values, not the
  stage: each stage has its own table (``owlbank-<stage>-modelstats``).
* **Model tags:** samples and verdicts name their model by :func:`model_tag` (eight hex digits of the
  model ID's SHA-256), never by its place in the list, so a reordered or changed list (a deploy, or
  Compute and Bridge stacks on different lists for a while) never hands one model's samples or state
  to another. The tag, not the ID, keeps the state record within one write unit whatever the IDs'
  length. The poller matches the stored verdicts to its list by tag (:func:`align`): a model
  that is new to the list starts healthy, one that moved keeps its state.
* **Sample counters**: :class:`Counters` is what one process adds to one sample item: weighted
  calls and TTFT bins for sampled latency samples (weight ``1 / report_prob``, an integer), and errors,
  deadline misses and unavailable detections always, with weight 1, and ``raw``, the unweighted
  number of latency samples reported. Request errors are never counted. Every counter is a
  non-negative integer. A process writes them with one ``Store.add_counters`` (``sample_key``,
  ``expires_at`` from :func:`sample_expires_at` as a default); the poller reads the window with one
  ``Store.query_range`` (:func:`window_range`) and bridges read :func:`state_key` with an eventually
  consistent ``Store.get``.
* **Window statistics:** :func:`window_stats` turns the window's buckets of one model into a
  :class:`~owlbank.failover.rules.Stats`, with percentiles from the merged histogram, so the trip rule is
  :func:`~owlbank.failover.rules.is_degraded`, the same as in a process.
* **The verdict**: :func:`fleet_verdict` folds one decision cycle into the previous
  :class:`State`: unavailable at once on a report and back only through the poller's recheck
  A healthy entry trips by the rules of 3.1, the p90 rule only when the window holds at least
  ``min_samples`` reported samples as well as ``min_samples`` weighted ones (so a few heavily
  weighted reports can't trip it); a degraded entry recovers only from the poller's
  ping samples (kept in the state, bounded to the recovery period), only the highest-ranked degraded
  entry is pinged, so recovery is one rank at a time; no change before the dwell (except to and
  from unavailable). ``report_prob`` is set from the window's traffic against the write budget.

Everything is pure: time is passed in (wall-clock epoch seconds, the poller's clock), nothing does I/O.
Times, percentiles and rates in the state item are integers (milliseconds, permille): the store holds
no floats.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from owlbank.failover.classify import Outcome
from owlbank.failover.histogram import Histogram, merge
from owlbank.failover.rules import EntryState, Sample, Stats, dwell_elapsed, is_degraded, is_recovered
from owlbank.failover.settings import MAX_ENTRIES, Settings

#: Sample bucket width (a Fargate task flushes every 10 s).
BUCKET_S: Final = 10
#: A sample item expires this long after its bucket starts (about 15 minutes).
SAMPLE_TTL_S: Final = 900
#: The two fixed partition key values; the IAM leading-key conditions name exactly these.
SAMPLES_PK: Final = "samples"
STATE_PK: Final = "state"
STATE_SK: Final = "state"
#: The fleet's sample-write budget.
WRITE_BUDGET_PER_S: Final = 5.0
#: The largest weight (``report_prob`` 1 in 10,000), so a counter stays far from any limit.
MAX_REPORT_EVERY: Final = 10_000
#: Ping samples kept in the state record: one entry's recovery period (5 per cycle, 2 cycles a minute,
#: 120 s, one at its edge), capped so the item stays one write unit.
MAX_PING_SAMPLES: Final = 24
#: One write unit.
ITEM_LIMIT_BYTES: Final = 1024

#: :func:`model_tag`'s length in hex digits.
MODEL_TAG_LEN: Final = 8

_TAG = re.compile(rf"[0-9a-f]{{{MODEL_TAG_LEN}}}")
_SK = re.compile(rf"(?P<bucket>\d{{10}})#(?P<model>[0-9a-f]{{{MODEL_TAG_LEN}}})")
_SAMPLE_COUNTERS: Final = ("n", "raw", "err", "miss", "unav")
_HARD: Final = frozenset({Outcome.THROTTLED, Outcome.TIMEOUT, Outcome.ERROR})


# --------------------------------------------------------------------------- keys


def bucket_start(epoch_s: float) -> int:
    """The start of the 10 s bucket holding ``epoch_s``."""
    if not isinstance(epoch_s, int | float) or not math.isfinite(epoch_s) or epoch_s < 0:
        raise ValueError("epoch_s must be a finite, non-negative number")
    return int(epoch_s // BUCKET_S) * BUCKET_S


def model_tag(model_id: str) -> str:
    """The model's name in the table: the first :data:`MODEL_TAG_LEN` hex digits of the SHA-256 of
    its ID (fixed length, no caller data)."""
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a model ID")
    return hashlib.sha256(model_id.encode()).hexdigest()[:MODEL_TAG_LEN]


def model_tags(settings: Settings) -> tuple[str, ...]:
    """The list's tags in list order; :class:`ValueError` if two collide (one chance in billions)."""
    tags = tuple(model_tag(e.model_id) for e in settings.entries)
    if len(set(tags)) != len(tags):
        raise ValueError("two model IDs have the same tag")
    return tags


def _tag(name: str, value: Any) -> str:
    if not isinstance(value, str) or _TAG.fullmatch(value) is None:
        raise ValueError(f"{name} must be a model tag")
    return value


def sample_sort_key(bucket: int, model: str) -> str:
    """``<bucket_start, ten digits>#<model_tag>``: lexical order is time order, then model."""
    if not isinstance(bucket, int) or bucket < 0 or bucket % BUCKET_S or bucket >= 10**10:
        raise ValueError("bucket must be a bucket start")
    return f"{bucket:010d}#{_tag('model', model)}"


def parse_sample_sort_key(sk: str) -> tuple[int, str]:
    """``(bucket_start, model_tag)`` of a sample sort key; :class:`ValueError` for anything else."""
    m = _SK.fullmatch(sk) if isinstance(sk, str) else None
    if m is None:
        raise ValueError("not a sample sort key")
    bucket, model = int(m["bucket"]), m["model"]
    sample_sort_key(bucket, model)  # the same checks as the builder
    return bucket, model


def sample_key(bucket: int, model: str) -> dict[str, str]:
    """A sample item's key: ``model`` is the :func:`model_tag`."""
    return {"s": SAMPLES_PK, "sk": sample_sort_key(bucket, model)}


def state_key() -> dict[str, str]:
    """The verdict's key, in its own partition."""
    return {"s": STATE_PK, "sk": STATE_SK}


def sample_expires_at(bucket: int) -> int:
    return bucket + SAMPLE_TTL_S


def window_buckets(now_s: float, settings: Settings) -> tuple[int, int]:
    """The first and last bucket of the window: every bucket starting within ``window_age_s`` of
    ``now_s``, up to the one holding ``now_s``."""
    last = bucket_start(now_s)
    oldest = max(0.0, now_s - settings.window_age_s)
    first = math.ceil(oldest / BUCKET_S) * BUCKET_S
    return min(first, last), last


def window_range(now_s: float, settings: Settings) -> tuple[str, str]:
    """``(low, high)`` for ``sk BETWEEN :low AND :high``: every model's sample of the window, nothing
    older. ``low`` is the first bucket's bare prefix (it sorts before ``<prefix>#<tag>``) and ``high``
    ends in ``~``, which sorts after every model tag."""
    first, last = window_buckets(now_s, settings)
    return f"{first:010d}", f"{last:010d}#~"


# --------------------------------------------------------------------------- sample counters


def _count(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass
class Counters:
    """One process's additions to one sample item (one model, one bucket), as integers.

    ``calls`` counts every counted call: sampled OK calls with their weight, and every error and miss
    with weight 1 (always reported). The histogram holds sampled OK TTFTs (weighted) and misses at their
    deadline (weight 1), as :meth:`Stats.of` counts an abandoned call's TTFT as the deadline."""

    calls: int = 0
    raw: int = 0  # latency samples reported, unweighted (min_samples applies to it too)
    errors: int = 0
    misses: int = 0
    unavailable: int = 0
    hist: Histogram = field(default_factory=Histogram.empty)

    def record(self, outcome: Outcome, ttft_s: float | None, *, weight: int = 1) -> None:
        """Add one model call. ``weight`` (``1 / report_prob``) applies to an OK call only."""
        if not isinstance(weight, int) or isinstance(weight, bool) or weight < 1:
            raise ValueError("weight must be a positive integer")
        if outcome is Outcome.REQUEST_ERROR:  # never a sample
            return
        if outcome in _HARD:
            self.calls += 1
            self.errors += 1
            return
        if ttft_s is None:
            raise ValueError("an OK or abandoned call has a TTFT")
        w = weight if outcome is Outcome.OK else 1
        self.hist = self.hist.add(ttft_s, weight=w)
        self.calls += w
        self.raw += 1
        if outcome is Outcome.ABANDONED:
            self.misses += 1

    def record_unavailable(self) -> None:
        """An unavailable detection; not a call sample."""
        self.unavailable += 1

    @property
    def empty(self) -> bool:
        return not (self.calls or self.unavailable)

    def increments(self) -> dict[str, int]:
        """The non-zero counters by attribute name, for one ``ADD`` update of the sample item."""
        values = {"n": self.calls, "raw": self.raw, "err": self.errors, "miss": self.misses, "unav": self.unavailable}
        return {k: v for k, v in values.items() if v} | self.hist.to_attrs()


@dataclass(frozen=True)
class Bucket:
    """One sample item as read back: one model's merged counters for one 10 s bucket."""

    start: int
    model: str  # the model_tag
    calls: int
    raw: int
    errors: int
    misses: int
    unavailable: int
    hist: Histogram

    @classmethod
    def from_item(cls, item: Mapping[str, Any]) -> Bucket:
        """Parse a sample item; a malformed one raises :class:`ValueError` (a missing counter is 0)."""
        start, model = parse_sample_sort_key(item.get("sk"))  # type: ignore[arg-type]
        n, raw, err, miss, unav = (_count(k, item.get(k, 0)) for k in _SAMPLE_COUNTERS)
        return cls(start=start, model=model, calls=n, raw=raw, errors=err, misses=miss, unavailable=unav,
                   hist=Histogram.from_attrs(item))


def window_stats(buckets: Iterable[Bucket]) -> Stats:
    """One model's window as :class:`Stats`, with percentiles from the merged histogram.

    The trailing run of misses counts the misses of the newest buckets whose every call was a miss; a
    bucket with any other call ends the run without its own misses (their order within it is unknown),
    so the fleet's run is never longer than the true one."""
    bs = sorted(buckets, key=lambda b: b.start)
    hist = merge(b.hist for b in bs)
    run = 0
    for b in reversed(bs):
        if b.calls == 0:
            continue
        if b.calls != b.misses:
            break
        run += b.misses
    return Stats(calls=sum(b.calls for b in bs), latency=hist.total, errors=sum(b.errors for b in bs),
                 p50_s=hist.percentile(50), p90_s=hist.percentile(90), miss_run=run,
                 span_s=float(bs[-1].start - bs[0].start) if bs else 0.0)


def report_every(window: Iterable[Bucket], settings: Settings, *,
                 budget_per_s: float = WRITE_BUDGET_PER_S) -> int:
    """The weight ``k`` (``report_prob = 1 / k``) that keeps sampled latency reports near the budget.

    The window's weighted OK calls (every model) estimate the fleet's true OK-call rate whatever the
    weights in force were; each sample write carries at least one sampled call, so at ``rate / k`` sampled
    calls a second the writes stay at or under the budget (errors and misses are bounded per process
    and flush). ``k`` is 1 at or below the budget and never falls as the rate rises."""
    if not math.isfinite(budget_per_s) or budget_per_s <= 0:
        raise ValueError("budget_per_s must be positive")
    ok = sum(b.calls - b.errors - b.misses for b in window)
    rate = max(ok, 0) / settings.window_age_s
    return min(MAX_REPORT_EVERY, max(1, math.ceil(rate / budget_per_s)))


# --------------------------------------------------------------------------- the state record


@dataclass(frozen=True)
class ModelVerdict:
    """One entry in the state record. ``rank`` is its place in the bridges' choice order: healthy
    entries in list order, then degraded ones by p90, then unavailable ones. ``since_s`` is the last
    state change (``None``: never). ``model`` is the entry's :func:`model_tag`: what the verdict is
    about (the index is only its place in the poller's list)."""

    index: int
    model: str
    state: EntryState
    rank: int
    since_s: float | None
    p50_s: float | None
    p90_s: float | None
    error_rate: float | None


@dataclass(frozen=True)
class Pings:
    """The recovery ping samples of the entry being pinged, oldest first."""

    index: int
    samples: tuple[Sample, ...]


@dataclass(frozen=True)
class State:
    """The verdict record. ``version`` grows on every change; ``written_at_s`` on every cycle."""

    version: int
    written_at_s: float
    models: tuple[ModelVerdict, ...]
    pings: Pings | None
    report_every: int

    @property
    def report_prob(self) -> float:
        return 1.0 / self.report_every

    def to_item(self) -> dict[str, Any]:
        """The state item: integers (ms, permille), strings, lists, maps and nulls only."""
        return {
            **state_key(),
            "version": self.version,
            "written_at": _ms(self.written_at_s),
            "models": [{"index": m.index, "m": m.model, "state": m.state.value, "rank": m.rank,
                        "since": _opt_ms(m.since_s), "p50": _opt_ms(m.p50_s), "p90": _opt_ms(m.p90_s),
                        "err": None if m.error_rate is None else round(m.error_rate * 1000)}
                       for m in self.models],
            "pings": None if self.pings is None else {
                "index": self.pings.index,
                # each sample's time as milliseconds before ``written_at``: 4 bytes fewer a sample
                "samples": [[_ms(self.written_at_s) - _ms(p.at_s), _opt_ms(p.ttft_s), p.outcome.value]
                            for p in self.pings.samples]},
            "report_every": self.report_every,
        }

    @classmethod
    def from_item(cls, item: Mapping[str, Any]) -> State:
        """Parse a state item; a malformed one raises :class:`ValueError`."""
        try:
            models = item["models"]
            if not isinstance(models, list) or not 1 <= len(models) <= MAX_ENTRIES:
                raise ValueError("models")
            verdicts = tuple(ModelVerdict(
                index=_count("index", m["index"]), model=_tag("m", m["m"]), state=EntryState(m["state"]), rank=_count("rank", m["rank"]),
                since_s=_opt_s(m["since"]), p50_s=_opt_s(m["p50"]), p90_s=_opt_s(m["p90"]),
                error_rate=None if m["err"] is None else _count("err", m["err"]) / 1000) for m in models)
            if ([v.index for v in verdicts] != list(range(len(verdicts)))
                    or len({v.model for v in verdicts}) != len(verdicts)):
                raise ValueError("models")
            written_ms = _count("written_at", item["written_at"])
            pings = item["pings"]
            parsed_pings = None
            if pings is not None:
                samples = []
                for raw in pings["samples"]:
                    if not isinstance(raw, list) or len(raw) != 3:
                        raise ValueError("pings")
                    age, ttft, outcome = raw
                    if not isinstance(age, int) or isinstance(age, bool):
                        raise ValueError("pings")
                    samples.append(Sample((written_ms - age) / 1000, Outcome(outcome), _opt_s(ttft)))
                parsed_pings = Pings(_count("index", pings["index"]), tuple(samples))
            every = _count("report_every", item["report_every"])
            if every < 1:
                raise ValueError("report_every")
            return cls(version=_count("version", item["version"]),
                       written_at_s=written_ms / 1000,
                       models=verdicts, pings=parsed_pings, report_every=every)
        except (KeyError, TypeError) as exc:
            raise ValueError("malformed state record") from exc


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def _opt_ms(seconds: float | None) -> int | None:
    """Milliseconds; ``inf`` (the histogram's open bin) is stored as -1."""
    if seconds is None:
        return None
    return -1 if math.isinf(seconds) else _ms(seconds)


def _opt_s(ms: Any) -> float | None:
    if ms is None:
        return None
    if ms == -1 and isinstance(ms, int) and not isinstance(ms, bool):
        return math.inf
    return _count("a time", ms) / 1000


def initial_state(settings: Settings, now_s: float) -> State:
    """Every entry healthy and never changed, in list order; ``report_prob`` 1; no pings."""
    return State(version=0, written_at_s=now_s, pings=None, report_every=1,
                 models=tuple(_initial_verdict(i, tag) for i, tag in enumerate(model_tags(settings))))


def _initial_verdict(index: int, model: str) -> ModelVerdict:
    return ModelVerdict(index=index, model=model, state=EntryState.HEALTHY, rank=index, since_s=None,
                        p50_s=None, p90_s=None, error_rate=None)


def align(previous: State, settings: Settings) -> State:
    """``previous`` for the list in ``settings``, matched by model tag, not by place: each listed
    model keeps its stored verdict (wherever it was in the stored list), a model the record doesn't
    name starts healthy, a stored model no longer listed is dropped; the pings stay only with
    their model. Ranks are recomputed; ``version`` and the other fields are kept. Idempotent."""
    stored = {m.model: m for m in previous.models}
    models = []
    for i, tag in enumerate(model_tags(settings)):
        m = stored.get(tag)
        models.append(_initial_verdict(i, tag) if m is None else dataclasses.replace(m, index=i))
    ranks = _ranks(models)
    aligned = tuple(dataclasses.replace(m, rank=ranks[m.index]) for m in models)
    pings = None
    if previous.pings is not None and previous.pings.index < len(previous.models):
        pinged = previous.models[previous.pings.index].model
        moved = next((m.index for m in aligned if m.model == pinged), None)
        if moved is not None:
            pings = Pings(moved, previous.pings.samples)
    return dataclasses.replace(previous, models=aligned, pings=pings)


def ping_target(state: State) -> int | None:
    """The entry the poller pings this cycle: the highest-ranked (first-listed) degraded entry,
    never an unavailable one; ``None`` while no entry is degraded."""
    return next((m.index for m in state.models if m.state is EntryState.DEGRADED), None)


def _prune_pings(samples: Iterable[Sample], now_s: float, settings: Settings) -> tuple[Sample, ...]:
    """Keep the samples newer than the recovery period and the newest one at or before its edge (so the
    kept samples can span the whole period), at most :data:`MAX_PING_SAMPLES` of the newest."""
    ordered = sorted(samples, key=lambda s: s.at_s)
    cutoff = now_s - settings.recover_period_s
    older = [s for s in ordered if s.at_s <= cutoff]
    kept = older[-1:] + [s for s in ordered if s.at_s > cutoff]
    return tuple(kept[-MAX_PING_SAMPLES:])


def judged_pings(previous: State | None, settings: Settings, now_s: float,
                 pings: Sequence[Sample] = ()) -> tuple[Sample, ...]:
    """The ping samples :func:`fleet_verdict` judges this cycle, with the same arguments: the stored
    pings of ``ping_target(previous)`` and this cycle's, pruned (:func:`_prune_pings`); none while no
    entry is degraded. The poller logs their count (``model_recovered``)."""
    prev = align(previous, settings) if previous is not None else initial_state(settings, now_s)
    return _judged_pings(prev, ping_target(prev), settings, now_s, pings)


def _judged_pings(prev: State, target: int | None, settings: Settings, now_s: float,
                  pings: Sequence[Sample]) -> tuple[Sample, ...]:
    if target is None:
        return ()
    carried = prev.pings.samples if prev.pings is not None and prev.pings.index == target else ()
    return _prune_pings((*carried, *pings), now_s, settings)


def _ranks(models: Sequence[ModelVerdict]) -> list[int]:
    def order(m: ModelVerdict) -> tuple[int, float, int]:
        if m.state is EntryState.HEALTHY:
            return 0, 0.0, m.index
        if m.state is EntryState.DEGRADED:
            return 1, math.inf if m.p90_s is None else m.p90_s, m.index
        return 2, 0.0, m.index

    ranks = [0] * len(models)
    for rank, m in enumerate(sorted(models, key=order)):
        ranks[m.index] = rank
    return ranks


def fleet_verdict(window: Iterable[Bucket], settings: Settings, now_s: float, previous: State | None, *,
                  pings: Sequence[Sample] = (), available: Collection[int] = (),
                  budget_per_s: float = WRITE_BUDGET_PER_S) -> State:
    """One decision cycle: the new state from the window's buckets, the previous state, the ping
    samples this cycle sent to ``ping_target(previous)`` and the unavailable entries whose recheck
    succeeded (``available``).

    ``previous`` is matched to the list by model tag (:func:`align`: the list changed in a deploy).
    Buckets outside :func:`window_buckets`, or of a model not in the list, are ignored, as the Query's
    range would leave the former out."""
    n = len(settings)
    prev = align(previous, settings) if previous is not None else initial_state(settings, now_s)
    ref = previous if previous is not None else prev  # a change of the stored record is a change
    first, last = window_buckets(now_s, settings)
    index_of = {tag: i for i, tag in enumerate(model_tags(settings))}
    by_index: dict[int, list[Bucket]] = {i: [] for i in range(n)}
    for b in window:
        if first <= b.start <= last and b.model in index_of:
            by_index[index_of[b.model]].append(b)

    target = ping_target(prev)
    ping_samples = _judged_pings(prev, target, settings, now_s, pings)
    ping_stats = Stats.of(ping_samples, good_under_s=None if target is None
                          else settings.entries[target].recover_ttft_s)

    updated: list[ModelVerdict] = []
    for i, entry in enumerate(settings.entries):
        pv = prev.models[i]
        stats = window_stats(by_index[i])
        state, since = pv.state, pv.since_s
        judged = stats
        if sum(b.raw for b in by_index[i]) < settings.min_samples:
            # a few heavily weighted reports never make a window: the p90 rule needs
            # min_samples reported samples as well as min_samples weighted ones
            judged = dataclasses.replace(stats, p90_s=None)
        if state is EntryState.UNAVAILABLE and i in available:
            state, since = EntryState.HEALTHY, now_s  # the recheck's request succeeded
        elif state is not EntryState.UNAVAILABLE and any(
                b.unavailable and (since is None or b.start >= since) for b in by_index[i]):
            state, since = EntryState.UNAVAILABLE, now_s  # one is enough, whatever the dwell
        elif dwell_elapsed(since_s=since, now_s=now_s, settings=settings):
            if state is EntryState.HEALTHY and is_degraded(judged, entry, settings):
                state, since = EntryState.DEGRADED, now_s
            elif (state is EntryState.DEGRADED and i == target
                  and is_recovered(ping_stats, entry, settings)):
                state, since = EntryState.HEALTHY, now_s
        seen = stats if stats.latency else ping_stats if i == target and ping_stats.latency else None
        updated.append(ModelVerdict(
            index=i, model=pv.model, state=state, rank=pv.rank, since_s=since,
            p50_s=seen.p50_s if seen else pv.p50_s, p90_s=seen.p90_s if seen else pv.p90_s,
            error_rate=seen.error_rate if seen else pv.error_rate))

    ranks = _ranks(updated)
    models = tuple(dataclasses.replace(m, rank=ranks[m.index]) for m in updated)
    draft = State(version=prev.version, written_at_s=now_s, models=models, pings=None,
                  report_every=report_every((b for bs in by_index.values() for b in bs), settings,
                                            budget_per_s=budget_per_s))
    next_target = ping_target(draft)
    kept = ping_samples if next_target is not None and next_target == target else ()
    new_pings = Pings(next_target, kept) if next_target is not None and kept else None
    changed = (models, new_pings, draft.report_every) != (ref.models, ref.pings, ref.report_every)
    return State(version=prev.version + 1 if changed else prev.version, written_at_s=now_s,
                 models=models, pings=new_pings, report_every=draft.report_every)


# --------------------------------------------------------------------------- item size


def item_size(item: Mapping[str, Any]) -> int:
    """An upper estimate of DynamoDB's item size: attribute names plus values (a number takes about
    one byte per two significant digits plus one; lists and maps 3 bytes plus 1 per element)."""
    return sum(len(k.encode()) + _value_size(v) for k, v in item.items())


def _value_size(v: Any) -> int:
    if v is None or isinstance(v, bool):
        return 1
    if isinstance(v, int):
        return (len(str(abs(v))) + 1) // 2 + 2
    if isinstance(v, str):
        return len(v.encode())
    if isinstance(v, list):
        return 3 + sum(1 + _value_size(x) for x in v)
    if isinstance(v, dict):
        return 3 + sum(1 + len(k.encode()) + _value_size(x) for k, x in v.items())
    raise ValueError(f"unsupported value type {type(v).__name__}")
