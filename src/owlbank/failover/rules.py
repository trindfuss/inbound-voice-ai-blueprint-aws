"""The pure trip and recovery rules.

:class:`Stats` summarises a window of samples: the in-process state builds it from raw
:class:`Sample` s with :meth:`Stats.of`, the poller from weighted histograms, and both judge it with
:func:`is_degraded` and :func:`is_recovered`, so the fleet and a single process apply the same rules.

* **Trip**: p90 TTFT strictly over the entry's trip threshold with at least ``min_samples``
  latency samples; or a trailing run of ``miss_run`` deadline misses, however few the samples; or a
  hard-error rate strictly over the limit with at least ``min_samples`` calls.
* **Recover**: at least ``recover_min_pings`` latency samples, the good ones (OK, TTFT strictly
  under the recovery threshold) spanning at least ``recover_period_s`` (from the oldest good sample to
  the newest sample: :attr:`Stats.good_span_s`), p90 strictly under the recovery threshold, the error rate strictly under the
  limit, and no trailing run of misses that would trip. Recovery implies not tripped (hysteresis).
* Request errors are not samples: :meth:`Stats.of` skips them.
* Dwell is an explicit clock comparison, never a timer's accuracy.

The window itself (the most recent ``window_calls`` samples, none older than ``window_age_s``) is the
caller's: these functions judge what they are given. Percentiles are nearest-rank, the definition a
histogram reproduces to within one bin.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from owlbank.failover.classify import Outcome
from owlbank.failover.settings import EntrySettings, Settings

_HARD = frozenset({Outcome.THROTTLED, Outcome.TIMEOUT, Outcome.ERROR})
_LATENCY = frozenset({Outcome.OK, Outcome.ABANDONED})


class EntryState(enum.StrEnum):
    """An entry's state, in order of severity; shared by the in-process state and the
    fleet verdict."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Sample:
    """One model call: when (monotonic seconds), its outcome, and its TTFT when it had one.

    An abandoned call's TTFT is the deadline."""

    at_s: float
    outcome: Outcome
    ttft_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, Outcome):
            raise ValueError("outcome must be an Outcome")
        if self.ttft_s is not None and (not math.isfinite(self.ttft_s) or self.ttft_s < 0):
            raise ValueError("ttft_s must be a finite, non-negative number")


def percentile(values: Sequence[float], p: int) -> float | None:
    """The nearest-rank ``p``-th percentile (1 to 100): the smallest value with at least ``p``% of the
    values at or below it. ``None`` for no values."""
    if not 1 <= p <= 100:
        raise ValueError("percentile must be 1 to 100")
    if not values:
        return None
    ordered = sorted(values)
    rank = (p * len(ordered) + 99) // 100  # ceil(p * n / 100), in integers
    return ordered[rank - 1]


@dataclass(frozen=True)
class Stats:
    """A window's numbers. ``calls`` counts every sample (request errors excluded), ``latency`` those
    with a TTFT, ``errors`` the hard errors; ``miss_run`` is the trailing run of deadline misses;
    ``span_s`` the time from the first sample to the last; ``good_span_s`` the time from the oldest OK
    sample with a TTFT strictly under :meth:`of`'s ``good_under_s`` to the newest sample (0 without
    one, or without a threshold: the recovery rule then never passes); ``good`` counts those samples."""

    calls: int
    latency: int
    errors: int
    p50_s: float | None
    p90_s: float | None
    miss_run: int
    span_s: float
    good_span_s: float = 0.0
    good: int = 0

    @property
    def error_rate(self) -> float:
        return self.errors / self.calls if self.calls else 0.0

    @classmethod
    def of(cls, samples: Iterable[Sample], *, good_under_s: float | None = None) -> Stats:
        """The numbers of raw samples, in time order; request errors are skipped. ``good_under_s``
        (the entry's recovery threshold) sets :attr:`good_span_s`; abandoned samples and hard errors are
        never good, though they still count in ``calls``, ``latency`` and the percentiles."""
        counted = sorted((s for s in samples if s.outcome is not Outcome.REQUEST_ERROR), key=lambda s: s.at_s)
        ttfts = [s.ttft_s for s in counted if s.outcome in _LATENCY and s.ttft_s is not None]
        run = 0
        for s in reversed(counted):
            if s.outcome is not Outcome.ABANDONED:
                break
            run += 1
        goods = [s for s in counted if good_under_s is not None and s.outcome is Outcome.OK
                 and s.ttft_s is not None and s.ttft_s < good_under_s]
        good = goods[0] if goods else None
        return cls(calls=len(counted), latency=len(ttfts),
                   errors=sum(1 for s in counted if s.outcome in _HARD),
                   p50_s=percentile(ttfts, 50), p90_s=percentile(ttfts, 90), miss_run=run,
                   span_s=counted[-1].at_s - counted[0].at_s if counted else 0.0,
                   good_span_s=counted[-1].at_s - good.at_s if good is not None else 0.0,
                   good=len(goods))


def is_degraded(stats: Stats, entry: EntrySettings, settings: Settings) -> bool:
    """The trip rule of 3.1 for one entry."""
    if stats.miss_run >= settings.miss_run:
        return True
    if (stats.latency >= settings.min_samples and stats.p90_s is not None
            and stats.p90_s > entry.trip_ttft_s):
        return True
    return stats.calls >= settings.min_samples and stats.error_rate > entry.error_rate_limit


def is_recovered(stats: Stats, entry: EntrySettings, settings: Settings) -> bool:
    """The recovery rule of 4.3 over a degraded entry's ping samples (the poller's pings) (dwell is
    :func:`dwell_elapsed`).

    The span is :attr:`Stats.good_span_s`, so ``stats`` must come from :meth:`Stats.of` with
    ``good_under_s=entry.recover_ttft_s`` (the samples under the recovery threshold span the period)."""
    return (stats.latency >= settings.recover_min_pings
            and stats.good >= 1  # a zero period still needs one good ping
            and stats.good_span_s >= settings.recover_period_s
            and stats.p90_s is not None and stats.p90_s < entry.recover_ttft_s
            and stats.error_rate < entry.error_rate_limit
            and stats.miss_run < settings.miss_run)


def dwell_elapsed(*, since_s: float | None, now_s: float, settings: Settings) -> bool:
    """Whether a state change is allowed: ``dwell_s`` has passed since the last one (``None``: never)."""
    return since_s is None or now_s - since_s >= settings.dwell_s
