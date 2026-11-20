"""Failover settings: per entry and global.

:func:`entry_defaults` gives the defaults for the three default models, matched by family and
version whatever the profile prefix (``us.``, ``global.``) and version tail, and the baseline-derived
defaults for any other model. :meth:`Settings.for_models` builds the settings for an ordered list; the
last entry has no first-token deadline, so a list of one behaves as today.

Every value is checked when built; a bad one raises :class:`SettingsError` naming the field, never
its value. The ``model_failover`` policy section builds on these.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

#: The baseline p90 TTFT assumed for a model without measured defaults.
DEFAULT_BASELINE_P90_S: Final = 1.5
DEFAULT_ERROR_RATE_LIMIT: Final = 0.2
MAX_ENTRIES: Final = 3
MAX_PINGS_PER_CYCLE: Final = 5

#: ``[<geo>.]anthropic.claude-<family>-<major>-<minor>[-<yyyymmdd>][-v<n>[:<k>]]``.
_KNOWN_ID = re.compile(r"(?:[a-z]+\.)?anthropic\.claude-(?P<family>[a-z]+)-(?P<major>\d+)-(?P<minor>\d{1,2})"
                       r"(?:-\d{8})?(?:-v\d+(?::\d+)?)?")


class SettingsError(ValueError):
    """A failover setting is unusable; carries the setting's name only."""

    def __init__(self, field_name: str) -> None:
        super().__init__(f"invalid failover setting: {field_name}")
        self.field = field_name


def _positive(name: str, value: float) -> None:
    if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise SettingsError(name)


def _non_negative(name: str, value: float) -> None:
    if not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise SettingsError(name)


def _count(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise SettingsError(name)


@dataclass(frozen=True)
class EntrySettings:
    """One entry's thresholds. ``deadline_s`` is ``None`` only on the last entry."""

    model_id: str
    trip_ttft_s: float  # p90 TTFT strictly above this trips the window
    recover_ttft_s: float  # ping p90 strictly below this recovers; below the trip
    deadline_s: float | None  # first-token deadline
    error_rate_limit: float = DEFAULT_ERROR_RATE_LIMIT  # hard-error share

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise SettingsError("model_id")
        _positive("trip_ttft_s", self.trip_ttft_s)
        _positive("recover_ttft_s", self.recover_ttft_s)
        if self.recover_ttft_s >= self.trip_ttft_s:  # hysteresis
            raise SettingsError("recover_ttft_s")
        if self.deadline_s is not None:
            _positive("deadline_s", self.deadline_s)
        _positive("error_rate_limit", self.error_rate_limit)
        if self.error_rate_limit > 1:
            raise SettingsError("error_rate_limit")


#: Keyed by (family, major, minor): trip, recover, deadline.
_KNOWN: Final[dict[tuple[str, int, int], tuple[float, float, float]]] = {
    ("haiku", 5, 5): (3.5, 2.5, 4.0),
    ("haiku", 4, 5): (1.6, 1.1, 3.0),
    # No deadline is listed because Sonnet 5.5 is the default last entry; placed before the
    # last it needs one: the same as the other adaptive model's.
    ("sonnet", 5, 5): (3.5, 2.5, 4.0),
}


def series_parts(model_id: str) -> tuple[str, int, int] | None:
    """``(family, major, minor)`` of a Claude model ID, whatever its profile prefix (``us.``,
    ``global.``) and version tail, or ``None`` for another ID. The key of :data:`_KNOWN`."""
    m = _KNOWN_ID.fullmatch(model_id)
    if m is None:
        return None
    return m["family"], int(m["major"]), int(m["minor"])


def series_key(model_id: str) -> str | None:
    """``<family>-<major>-<minor>`` (``haiku-5-5``) of a Claude model ID, or ``None`` for another
    ID: the per-model prompt's key, so ``us.`` and ``global.`` profiles share one."""
    parts = series_parts(model_id)
    return None if parts is None else "{}-{}-{}".format(*parts)


def entry_defaults(model_id: str, *, baseline_p90_s: float = DEFAULT_BASELINE_P90_S,
                   last: bool = False) -> EntrySettings:
    """The defaults for ``model_id``; ``last`` drops the deadline.

    A model without measured defaults gets trip ``2.3 x baseline``, recover ``1.5 x baseline`` and a
    deadline of ``max(2.7 x baseline)``.
    """
    _positive("baseline_p90_s", baseline_p90_s)
    series = series_parts(model_id)
    known = _KNOWN.get(series) if series is not None else None
    if known is not None:
        trip, recover, deadline = known
    else:
        trip, recover = 2.3 * baseline_p90_s, 1.5 * baseline_p90_s
        deadline = max(4.0, 2.7 * baseline_p90_s)
    return EntrySettings(model_id=model_id, trip_ttft_s=trip, recover_ttft_s=recover,
                         deadline_s=None if last else deadline)


@dataclass(frozen=True)
class Settings:
    """The ordered entries (one to three) and the global window, ping and recovery settings."""

    entries: tuple[EntrySettings, ...]
    window_calls: int = 20  # count window
    window_age_s: float = 120.0  # no sample older than this
    min_samples: int = 20  # for a window verdict; 20 so p90 tolerates two slow calls
    miss_run: int = 3  # deadline misses in a row that trip
    pings_per_cycle: int = 3  # poller pings per degraded entry per cycle, 1 to 5
    recover_min_pings: int = 10
    recover_period_s: float = 120.0  # pings must span this
    dwell_s: float = 180.0  # minimum time between state changes
    idle_reset_s: float = 600.0  # no samples this long: healthy again
    unavailable_recheck_s: float = 900.0

    def __post_init__(self) -> None:
        entries = self.entries
        if (not isinstance(entries, tuple) or not 1 <= len(entries) <= MAX_ENTRIES
                or not all(isinstance(e, EntrySettings) for e in entries)
                or len({e.model_id for e in entries}) != len(entries)):
            raise SettingsError("entries")
        if entries[-1].deadline_s is not None:
            raise SettingsError("deadline_s")
        if any(e.deadline_s is None for e in entries[:-1]):  # 3.3
            raise SettingsError("deadline_s")
        _count("window_calls", self.window_calls)
        _positive("window_age_s", self.window_age_s)
        _count("min_samples", self.min_samples)
        if self.min_samples > self.window_calls:  # a window could never be judged
            raise SettingsError("min_samples")
        _count("miss_run", self.miss_run)
        _count("pings_per_cycle", self.pings_per_cycle)
        if self.pings_per_cycle > MAX_PINGS_PER_CYCLE:
            raise SettingsError("pings_per_cycle")
        _count("recover_min_pings", self.recover_min_pings)
        _non_negative("recover_period_s", self.recover_period_s)
        _non_negative("dwell_s", self.dwell_s)
        _positive("idle_reset_s", self.idle_reset_s)
        _positive("unavailable_recheck_s", self.unavailable_recheck_s)

    @classmethod
    def for_models(cls, model_ids: Sequence[str], *, baseline_p90_s: float = DEFAULT_BASELINE_P90_S,
                   **global_settings: float) -> Settings:
        """Defaults for an ordered list of model IDs; ``global_settings`` override the values."""
        ids = list(model_ids)
        entries = tuple(entry_defaults(m, baseline_p90_s=baseline_p90_s, last=i == len(ids) - 1)
                        for i, m in enumerate(ids))
        return cls(entries=entries, **global_settings)  # type: ignore[arg-type]

    def entry(self, index: int) -> EntrySettings:
        return self.entries[index]

    def __len__(self) -> int:
        return len(self.entries)
