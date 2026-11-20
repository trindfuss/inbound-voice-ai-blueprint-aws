"""Latency-aware model failover: the pure decision module.

* :mod:`~owlbank.failover.settings`: per-entry thresholds and the global window, ping and
  recovery settings, with the measured defaults per known model;
* :mod:`~owlbank.failover.classify`: what a failed model call means for its model, and
  :class:`ModelUnusable`, raised before any output so a turn moves to the next entry;
* :mod:`~owlbank.failover.rules`: the trip and recovery rules over a window's numbers, shared by
  the in-process state and the fleet poller;
* :mod:`~owlbank.failover.core`: the in-process decision state, :class:`FailoverState` (which entry
  for a turn, which next before any output).

Everything here is pure: time is passed in, nothing does I/O, and nothing logs.
"""

from owlbank.failover.classify import UNAVAILABLE, ModelUnusable, Outcome, classify
from owlbank.failover.core import EntryHealth, EntryState, FailoverState, FleetView, Transition
from owlbank.failover.fleet import Bucket, Counters, fleet_verdict, ping_target
from owlbank.failover.histogram import Histogram
from owlbank.failover.rules import Sample, Stats, dwell_elapsed, is_degraded, is_recovered
from owlbank.failover.settings import EntrySettings, Settings, SettingsError, entry_defaults

__all__ = [
    "UNAVAILABLE",
    "Bucket",
    "Counters",
    "EntryHealth",
    "EntrySettings",
    "EntryState",
    "FailoverState",
    "FleetView",
    "Histogram",
    "ModelUnusable",
    "Outcome",
    "Sample",
    "Settings",
    "SettingsError",
    "Stats",
    "Transition",
    "classify",
    "dwell_elapsed",
    "entry_defaults",
    "fleet_verdict",
    "is_degraded",
    "is_recovered",
    "ping_target",
]
