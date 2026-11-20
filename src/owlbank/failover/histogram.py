"""A fixed-bin, weighted TTFT histogram for the fleet picture.

Bins are 0.25 s wide from 0 to :data:`RANGE_S`, plus one open bin for anything slower. Counts are
non-negative integers: a sampled latency is added with the integer weight ``1 / report_prob``, so
merged histograms stay unbiased and every counter stays a whole number (the store holds no floats).

:meth:`Histogram.percentile` is nearest-rank like :func:`owlbank.failover.rules.percentile` and returns
the upper edge of the bin holding that rank, so it is never below the raw-sample percentile and at most
one bin above it (``math.inf`` for the open bin). A trip (``p90 > trip``) can therefore come at most one
bin early and a recovery (``p90 < recover``) never early; :func:`covers` says whether a threshold is
inside the binned range, where that bound holds.

The range is 8 s, not the 3 s of the 12 bins (``h0..h11``): the default trip thresholds
(3.5 s) and deadlines (4.0 s) of would otherwise fall in the open bin.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

BIN_S: Final = 0.25
RANGE_S: Final = 8.0
#: 32 bins of 0.25 s and one open bin (``h32``: 8 s or more).
NUM_BINS: Final = int(RANGE_S / BIN_S) + 1
#: The sample item's attribute per bin.
BIN_ATTRS: Final[tuple[str, ...]] = tuple(f"h{i}" for i in range(NUM_BINS))


def _check_ttft(ttft_s: float) -> None:
    if not isinstance(ttft_s, int | float) or not math.isfinite(ttft_s) or ttft_s < 0:
        raise ValueError("ttft_s must be a finite, non-negative number")


def _check_count(name: str, value: Any, *, positive: bool = False) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < (1 if positive else 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'non-negative'} integer")


def bin_index(ttft_s: float) -> int:
    """The bin holding ``ttft_s``: ``floor(ttft / 0.25)``, the open bin from :data:`RANGE_S`."""
    _check_ttft(ttft_s)
    if ttft_s >= RANGE_S:
        return NUM_BINS - 1
    # Dividing by 0.25 (a power of two) is exact in binary floating point, so the edges are exact.
    return min(math.floor(ttft_s / BIN_S), NUM_BINS - 2)


def covers(threshold_s: float) -> bool:
    """Whether a threshold is inside the binned range (the one-bin bound of the module docstring)."""
    return 0 <= threshold_s < RANGE_S


@dataclass(frozen=True)
class Histogram:
    """Counts per bin; immutable, so a merge or an add returns a new one."""

    counts: tuple[int, ...] = (0,) * NUM_BINS

    def __post_init__(self) -> None:
        if not isinstance(self.counts, tuple) or len(self.counts) != NUM_BINS:
            raise ValueError(f"a histogram has {NUM_BINS} bins")
        for c in self.counts:
            _check_count("a bin count", c)

    @classmethod
    def empty(cls) -> Histogram:
        return cls()

    @classmethod
    def of(cls, ttfts: Iterable[float], *, weight: int = 1) -> Histogram:
        h = cls.empty()
        for t in ttfts:
            h = h.add(t, weight=weight)
        return h

    def add(self, ttft_s: float, *, weight: int = 1) -> Histogram:
        """This histogram with ``weight`` more in the bin of ``ttft_s`` (a positive integer)."""
        _check_count("weight", weight, positive=True)
        i = bin_index(ttft_s)
        counts = list(self.counts)
        counts[i] += weight
        return Histogram(tuple(counts))

    def __add__(self, other: Histogram) -> Histogram:
        return Histogram(tuple(a + b for a, b in zip(self.counts, other.counts, strict=True)))

    @property
    def total(self) -> int:
        return sum(self.counts)

    def percentile(self, p: int) -> float | None:
        """The nearest-rank ``p``-th percentile (1 to 100) as its bin's upper edge; ``None`` if empty."""
        if not 1 <= p <= 100:
            raise ValueError("percentile must be 1 to 100")
        total = self.total
        if total == 0:
            return None
        rank = (p * total + 99) // 100  # ceil(p * n / 100), as rules.percentile
        seen = 0
        for i, c in enumerate(self.counts):
            seen += c
            if seen >= rank:
                return math.inf if i == NUM_BINS - 1 else (i + 1) * BIN_S
        raise AssertionError("unreachable: rank <= total")  # pragma: no cover

    def to_attrs(self) -> dict[str, int]:
        """The non-zero bins as item attributes (``h<i>``): the increments of one sample update."""
        return {BIN_ATTRS[i]: c for i, c in enumerate(self.counts) if c}

    @classmethod
    def from_attrs(cls, item: Mapping[str, Any]) -> Histogram:
        """The bins of a sample item; a missing bin is 0, a bad counter raises :class:`ValueError`."""
        counts = []
        for name in BIN_ATTRS:
            value = item.get(name, 0)
            _check_count(name, value)
            counts.append(value)
        return cls(tuple(counts))


def merge(histograms: Iterable[Histogram]) -> Histogram:
    """The bin-wise sum."""
    out = Histogram.empty()
    for h in histograms:
        out = out + h
    return out
