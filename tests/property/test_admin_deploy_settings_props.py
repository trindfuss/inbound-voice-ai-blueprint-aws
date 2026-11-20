"""The start-up entry's properties for ``acceptSpokenFactors``.

For any sequence of cold starts, each alone or two at once (both read the marker before either
writes), with any ``acceptSpokenFactors`` values:

* the number of audit entries equals the number of recorded changes, counting the first record;
  a start that loses the race to the other start of its pair writes nothing;
* each entry has the audit shape (``action``, actor, one ``acceptSpokenFactors`` change) and is a
  ``security_change`` iff the new value is ``true`` and the old one wasn't;
* consecutive entries chain (each ``old`` is the previous ``new``, the first ``old`` is ``None``);
* the marker always equals the last recorded value (the newest entry's ``new``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from lambdas.admin_api import deploy_settings as ds
from lambdas.admin_api.audit import AuditClock
from owlbank.store import Item, MemoryStore, TxPut
from tests.admin_api_fakes import Clock


class RacingStore:
    """Right after this start reads the marker, ``other`` runs to completion."""

    def __init__(self, inner: MemoryStore, other: Callable[[], object]) -> None:
        self.inner, self.other, self.fired = inner, other, False

    def get(self, table: str, key: Mapping[str, Any]) -> Item | None:
        got = self.inner.get(table, key)
        if not self.fired:
            self.fired = True
            self.other()
        return got

    def transact_write(self, puts: Sequence[TxPut]) -> None:
        self.inner.transact_write(puts)


step = st.one_of(st.booleans().map(lambda v: (v,)), st.tuples(st.booleans(), st.booleans()))


@settings(max_examples=300, deadline=None)
@given(steps=st.lists(step, max_size=12), seeded=st.sampled_from([None, True, False]))
def test_entries_match_value_changes(steps: list[tuple[bool, ...]], seeded: bool | None) -> None:
    store = MemoryStore()
    c = Clock()
    clock = AuditClock(now=c, suffix=c.suffix)
    expected: list[tuple[bool | None, bool]] = []  # (old, new) of each entry, oldest first
    recorded: bool | None = None
    has_marker = False
    if seeded is not None:  # a marker left by an earlier deploy
        store.put("admin_audit", {**ds.marker_key(), "value": seeded, "rev": 7})
        recorded, has_marker = seeded, True

    def run(s: Any, v: bool) -> str:
        return ds.record_accept_spoken_factors(s, v, clock=clock, retention_days=90)

    for st_ in steps:
        if len(st_) == 1:
            (v,) = st_
            out = run(store, v)
            if not has_marker or v != recorded:
                assert out == "recorded"
                expected.append((recorded, v))
                recorded, has_marker = v, True
            else:
                assert out == "unchanged"
            continue
        a, b = st_
        before, had = recorded, has_marker
        outs: dict[str, str] = {}
        racing = RacingStore(store, lambda b=b, outs=outs: outs.__setitem__("b", run(store, b)))
        outs["a"] = run(racing, a)
        # b read and wrote first; a read the same state as b but writes after.
        b_writes = not had or b != before
        a_wants = not had or a != before
        if b_writes:
            assert outs["b"] == "recorded"
            expected.append((before, b))
            recorded, has_marker = b, True
        else:
            assert outs["b"] == "unchanged"
        if a_wants:
            if b_writes:
                assert outs["a"] == "lost_race"
            else:
                assert outs["a"] == "recorded"
                expected.append((before, a))
                recorded, has_marker = a, True
        else:
            assert outs["a"] == "unchanged"

    got = [e for e in store.query("admin_audit", {"pk": ds.DEPLOY_PK}) if e["sk"] != ds.MARKER_SK]
    assert len(got) == len(expected)
    for e, (old, new) in zip(got, expected, strict=True):  # sorted by sk = time order
        assert e["action"] == "deploy_setting" and e["actor"] == {"sub": "deploy"}
        assert e["target"] == {"kind": "deploy_setting", "id": "acceptSpokenFactors"}
        assert e["changes"] == [{"field": "acceptSpokenFactors", "old": old, "new": new}]
        assert e["security_change"] is (new is True and old is not True)
    for prev, nxt in zip(got, got[1:], strict=False):
        assert nxt["changes"][0]["old"] == prev["changes"][0]["new"]
    if seeded is None and got:
        assert got[0]["changes"][0]["old"] is None
    marker = store.get("admin_audit", ds.marker_key())
    if has_marker:
        assert marker is not None and marker["value"] is recorded and "expires_at" not in marker
        if got:
            assert marker["value"] is got[-1]["changes"][0]["new"]
    else:
        assert marker is None
