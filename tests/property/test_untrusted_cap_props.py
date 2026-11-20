"""The untrusted-attempt cap.

For any interleaving of trusted and untrusted attempts (right or wrong factors) with
time moving forward:

* a trusted attempt never reads or changes the ``risk`` table;
* an untrusted attempt escalates (``verification_risk``) exactly when the customer has
  ``limit`` untrusted failures in the current window, and never before;
* the window is ``untrusted_window_s`` from its first failure; after it, counting restarts.

The model also tracks the trusted per-customer lockout (fixed windows of
``verify_customer_window_s``), which applies after the gate.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from lambdas.auth_service import service
from owlbank.policy import Policy
from owlbank.store import MemoryStore

DATA = Path(__file__).resolve().parents[2] / "data"
FIXTURES = json.loads((DATA / "fixtures.json").read_text())["customers"]
FACTORS = json.loads((DATA / "fixtures.test-factors.json").read_text())["customers"]
DOC = json.loads((DATA / "policy.json").read_text())
THOMAS, ANI, TRUSTED = "C-1001", "+12025550101", "TN-Validation-Passed-A"
T0 = 1_760_000_000
LOCK_LIMIT, LOCK_WINDOW = 5, 3600  # auth_service DEFAULT_LIMITS (not in policy.json)


class RecordingStore(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.touched: list[str] = []

    def get(self, table, key):  # type: ignore[override]
        self.touched.append(table)
        return super().get(table, key)

    def put(self, table, item, *, condition=None):  # type: ignore[override]
        self.touched.append(table)
        return super().put(table, item, condition=condition)

    def delete(self, table, key):  # type: ignore[override]
        self.touched.append(table)
        return super().delete(table, key)

    def update(self, table, key, updates, *, remove=()):  # type: ignore[override]
        self.touched.append(table)
        return super().update(table, key, updates, remove=remove)


STEP = st.tuples(st.booleans(),                      # trusted?
                 st.booleans(),                      # right factors?
                 st.integers(min_value=0, max_value=2500))  # seconds since the last step


@settings(max_examples=200, deadline=None)
@given(limit=st.integers(min_value=1, max_value=4),
       window=st.integers(min_value=60, max_value=5000),
       steps=st.lists(STEP, min_size=1, max_size=25))
def test_untrusted_failure_counter(limit: int, window: int, steps: list[tuple[bool, bool, int]]) -> None:
    doc = copy.deepcopy(DOC)
    doc["ani_trust"].update(untrusted_failure_limit=limit, untrusted_window_s=window)
    store = RecordingStore()
    for c in FIXTURES:
        store.put("customers", copy.deepcopy(c))
    now = T0
    d = service.AuthDeps(store=store, policy=Policy.from_dict(doc), clock=lambda: now)

    count, window_end = 0, None          # model: untrusted failures in the current window
    lock: dict[int, int] = {}             # model: trusted failures per fixed lockout window
    for i, (trusted, ok, dt) in enumerate(steps):
        now += dt
        if window_end is not None and now >= window_end:
            count, window_end = 0, None   # the window lapsed: counting restarts
        locked = lock.get(now // LOCK_WINDOW, 0) >= LOCK_LIMIT
        if trusted:
            factors = {"zip": FACTORS[THOMAS]["zip"] if ok else "00000"}
        else:
            factors = ({"zip": FACTORS[THOMAS]["zip"], "ssn4": FACTORS[THOMAS]["ssn4"]} if ok
                       else {"zip": "00000", "ssn4": "0000"})
        before = copy.deepcopy(store._tables["risk"])  # noqa: SLF001
        store.touched.clear()
        out = service.verify(d, call_sid=f"CAprop{i:04d}", ani=ANI, factors=factors,
                             target_level=1, stir_verstat=TRUSTED if trusted else None)
        result = out["result"]

        if trusted:
            assert "risk" not in store.touched
            assert store._tables["risk"] == before  # noqa: SLF001
            assert result != "escalate"
            if locked:
                assert result == "rate_limited"
            else:
                assert result == ("verified" if ok else "not_verified")
                if not ok:
                    lock[now // LOCK_WINDOW] = lock.get(now // LOCK_WINDOW, 0) + 1
            continue

        if count >= limit:
            assert out == {"result": "escalate", "reason": "verification_risk", "level": 0}
            continue
        assert result != "escalate", (i, count, limit)
        if locked:
            assert result == "rate_limited"  # an existing lockout still applies; not counted
            continue
        assert result == ("verified" if ok else "not_verified")
        if not ok:
            if count == 0:
                window_end = now + window
            count += 1
        live = [it for it in store._tables["risk"].values()  # noqa: SLF001
                if it["expires_at"] > now]
        assert len(live) == count
        assert all(it["expires_at"] == window_end for it in live)


@settings(max_examples=100, deadline=None)
@given(st.lists(st.booleans(), min_size=1, max_size=12))
def test_cp_precheck_agrees_with_verify(trusted_seq: list[bool]) -> None:
    """``precheck`` escalates exactly when ``verify`` would, and never for a trusted caller."""
    store = MemoryStore()
    for c in FIXTURES:
        store.put("customers", copy.deepcopy(c))
    d = service.AuthDeps(store=store, policy=Policy.from_dict(DOC), clock=lambda: T0)
    for i, trusted in enumerate(trusted_seq):
        verstat = TRUSTED if trusted else None
        pre = service.precheck(d, call_sid=f"CApc{i}", ani=ANI, target_level=1,
                               stir_verstat=verstat)
        out = service.verify(d, call_sid=f"CApc{i}", ani=ANI,
                             factors={"zip": "00000", "ssn4": "0000"}, target_level=1,
                             stir_verstat=verstat)
        assert (pre["result"] == "escalate") == (out["result"] == "escalate")
        if trusted:
            assert pre == {"result": "ok"}
