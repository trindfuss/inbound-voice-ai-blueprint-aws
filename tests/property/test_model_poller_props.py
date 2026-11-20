"""Property tests of the model-failover poller.

Over generated sequences of decision cycles (what the bridges reported, how the pinged model answered,
which table calls failed):

* Each cycle that read the state sends exactly ``pings_per_cycle`` pings while the
  previous verdict has a degraded entry and none otherwise, only to the highest-ranked degraded entry,
  never to an unavailable one.
* A degraded entry recovers only after at least ``recover_min_pings`` pings spanning at least
  ``recover_period_s``; no state change (other than to or from unavailable) within ``dwell_s`` of the
  last one; at most one recovery per cycle.
* **Fold:** the state written equals the pure ``fleet_verdict`` folded over the same inputs: the window
  as stored, the previous record, and ping samples derived independently from the scripted answers
  (plus the samples of a cycle whose write failed).
* **Failures:** a cycle that didn't write leaves the stored record exactly as it was; versions never go
  back.

Each runs on asyncio and on uvloop (the production loop).
"""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from lambdas.model_poller import service
from owlbank.failover.classify import Outcome
from owlbank.failover.fleet import SAMPLES_PK, Bucket, State, fleet_verdict, ping_target, state_key, window_range
from owlbank.failover.rules import EntryState, Sample
from owlbank.failover.settings import Settings
from tests.model_poller_fakes import MODEL_IDS, Ping, add_samples, make_deps, run

D, U, H = EntryState.DEGRADED, EntryState.UNAVAILABLE, EntryState.HEALTHY
SETTINGS = Settings.for_models(list(MODEL_IDS), unavailable_recheck_s=60.0)
PPC = SETTINGS.pings_per_cycle

WINDOW = st.sampled_from(["none", "none", "none", "fast", "fast", "slow", "unav", "errors"])
# Mostly fast answers, so recoveries happen within a sequence.
PING = st.sampled_from([Ping(0.5), Ping(0.5), Ping(1.0), Ping(1.0), Ping(2.8), Ping(5.0), Ping(8.0),
                        Ping(error="ThrottlingException")])
FAILURE = st.sampled_from(["", "", "", "", "", "", "", "get", "query", "write"])
CYCLE = st.fixed_dictionaries({
    "window": st.lists(WINDOW, min_size=3, max_size=3),
    # often a cycle of good answers, so sequences reach a recovery
    "pings": st.one_of(st.just([Ping(0.5)] * PPC), st.just([Ping(1.0), Ping(0.5), Ping(2.0)]),
                       st.lists(PING, min_size=PPC, max_size=PPC)),
    "failure": FAILURE,
    "recheck_fails": st.sets(st.sampled_from(MODEL_IDS)),
})


def _report(store: Any, wall: float, window: list[str]) -> None:
    for i, kind in enumerate(window):
        if kind == "fast":
            add_samples(store, wall - 5, i, [0.5] * 20)
        elif kind == "slow":
            add_samples(store, wall - 5, i, [5.0] * 20)
        elif kind == "unav":
            add_samples(store, wall - 5, i, [], unavailable=1)
        elif kind == "errors":
            add_samples(store, wall - 5, i, [0.5] * 10, errors=10)


def _expected_sample(index: int, at: float, ping: Ping) -> Sample:
    """What a ping answering so must count as (independently of the poller's code)."""
    deadline = service.ping_deadline(SETTINGS, index)
    if ping.error is not None:
        return Sample(at, Outcome.THROTTLED)
    if ping.ttft_s > deadline:
        return Sample(at, Outcome.ABANDONED, deadline)
    return Sample(at, Outcome.OK, ping.ttft_s)


def _parsed(item: Any) -> State | None:
    return None if item is None else State.from_item(item)


@pytest.mark.parametrize("loop", ["asyncio", "uvloop"])
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(CYCLE, min_size=1, max_size=20), st.sampled_from(["", "degraded0", "degraded1", "unavailable0"]))
def test_cycles(loop: str, cycles: list[dict[str, Any]], start: str) -> None:
    deps, t, bedrock, store = make_deps(settings=SETTINGS)
    memory = service.PollerMemory()
    if start:
        states = [H, H, H]
        states[int(start[-1])] = D if start.startswith("degraded") else U
        from owlbank.failover.fleet import ModelVerdict, model_tag

        seeded = State(version=1, written_at_s=t.wall(), pings=None, report_every=1, models=tuple(
            ModelVerdict(index=i, model=model_tag(MODEL_IDS[i]), state=s, rank=i, since_s=t.wall() - 400,
                         p50_s=None, p90_s=None, error_rate=None)
            for i, s in enumerate(states)))
        store.put(service.TABLE, seeded.to_item())

    pending: tuple[int, tuple[Sample, ...]] | None = None
    sent_to: dict[int, list[float]] = {0: [], 1: [], 2: []}
    last_version = -1
    for c in cycles:
        t.advance(30)
        _report(store, t.wall(), c["window"])
        before_item = store.get(service.TABLE, state_key())
        previous = _parsed(before_item)
        for m in MODEL_IDS:
            bedrock.script[m] = deque(c["pings"])
        bedrock.recheck_fails = set(c["recheck_fails"])
        played_before = len(bedrock.played)
        rechecks_before = len(bedrock.rechecks)
        store.fail_get = c["failure"] == "get"
        store.fail_query = c["failure"] == "query"
        store.fail_write = c["failure"] == "write"

        report = run(lambda: service.run_cycle(deps, memory), loop)

        store.fail_get = store.fail_query = store.fail_write = False
        played = bedrock.played[played_before:]
        after_item = store.get(service.TABLE, state_key())

        # The ping half
        if c["failure"] == "get":
            assert played == []
            assert after_item == before_item
            continue
        target = ping_target(previous) if previous is not None else None
        if target is None:
            assert played == []
        else:
            assert len(played) == PPC
            event("pinged")
            assert {m for m, _, _ in played} == {MODEL_IDS[target]}
            assert previous is not None and previous.models[target].state is D
            assert all(m.state is not U for m in previous.models if m.index == target)
        sent = tuple(_expected_sample(target, at, p) for _, at, p in played) if target is not None else ()
        for _, at, _ in played:
            sent_to[target].append(at)  # type: ignore[index]
        carried = pending[1] if pending is not None and pending[0] == target else ()
        pings = (*carried, *sent)
        assert len(report.pings) == len(pings)
        for got, want in zip(report.pings, pings, strict=True):  # the fake clock's float sums: within 1 us
            assert (got.at_s, got.outcome) == (want.at_s, want.outcome)
            assert (got.ttft_s is None) == (want.ttft_s is None)
            assert got.ttft_s is None or abs(got.ttft_s - want.ttft_s) < 1e-6

        # rechecks: only unavailable entries, success only when the request answered
        rechecked = [m for m, _ in bedrock.rechecks[rechecks_before:]]
        assert all(previous is not None and previous.models[MODEL_IDS.index(m)].state is U for m in rechecked)
        for i in report.available:
            assert previous is not None and previous.models[i].state is U

        if not report.written:
            event("a cycle did not write")
            assert after_item == before_item  # a failed cycle never loses the stored record
            pending = (target, pings) if target is not None and pings else None
            continue
        pending = None

        # the fold: the same pure verdict over the same inputs
        low, high = window_range(report.now_s, SETTINGS)
        window = [Bucket.from_item(i) for i in store.query_range(service.TABLE, {"s": SAMPLES_PK}, low=low, high=high)]
        expected = fleet_verdict(window, SETTINGS, report.now_s, previous, pings=pings,
                                 available=report.available)
        new = _parsed(after_item)
        assert new == State.from_item(expected.to_item())
        assert new is not None and new.version >= last_version
        last_version = new.version

        # Dwell, recovery evidence, one rank at a time
        if previous is not None:
            recoveries = 0
            for old, cur in zip(previous.models, new.models, strict=True):
                if old.state is cur.state or U in (old.state, cur.state):
                    continue
                if old.since_s is not None:
                    assert cur.since_s - old.since_s >= SETTINGS.dwell_s - 1e-3  # type: ignore[operator]
                if old.state is D and cur.state is H:
                    recoveries += 1
                    event("a degraded entry recovered")
                    evidence = [a for a in sent_to[cur.index] if a >= old.since_s - 1e-3]  # type: ignore[operator]
                    assert len(evidence) >= SETTINGS.recover_min_pings
                    assert evidence[-1] - evidence[0] >= SETTINGS.recover_period_s - 1e-3
            assert recoveries <= 1


@pytest.mark.parametrize("loop", ["asyncio", "uvloop"])
@settings(max_examples=40, deadline=None)
@given(st.lists(st.booleans(), min_size=2, max_size=10))
def test_a_failed_write_never_loses_the_previous_state(loop: str, failures: list[bool]) -> None:
    """Writes fail at random: the stored record is always the last written one, unchanged by failures,
    and the next good write builds on it (its version is the stored one or the next)."""
    deps, t, _, store = make_deps(settings=SETTINGS)
    memory = service.PollerMemory()
    last: Any = None
    for i, fail in enumerate(failures):
        t.advance(30)
        if i % 3 == 0:
            _report(store, t.wall(), ["slow", "none", "none"] if i % 2 else ["fast", "slow", "none"])
        store.fail_write = fail
        report = run(lambda: service.run_cycle(deps, memory), loop)
        store.fail_write = False
        item = store.get(service.TABLE, state_key())
        if fail:
            assert not report.written and item == last
        else:
            assert report.written
            assert item is not None
            new = State.from_item(item)
            if last is not None:
                old = State.from_item(last)
                assert new.version in (old.version, old.version + 1)
                assert report.previous == old
            last = item
