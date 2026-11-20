"""Front-door session sidecar (``lambdas.frontdoor.state``): ``keyed_tries`` (keyed-in phone login) round-trips through the store and defaults to 0 for an item written before it existed."""

from __future__ import annotations

from lambdas.frontdoor import state as sessions
from owlbank.store import MemoryStore

SID = "CA" + "f" * 32


def test_keyed_tries_defaults_to_zero_and_round_trips() -> None:
    store = MemoryStore()
    sess = sessions.Session(call_sid=SID, expires_at=10_000)
    assert sess.keyed_tries == 0
    sess.keyed_tries = 2
    sessions.save(store, sess)
    loaded = sessions.load(store, SID)
    assert loaded == sess and loaded is not None and loaded.keyed_tries == 2
    assert store.get(sessions.TABLE, {"call_sid": sess.key()})["keyed_tries"] == 2


def test_an_item_without_keyed_tries_loads_as_zero() -> None:
    store = MemoryStore()
    sessions.save(store, sessions.Session(call_sid=SID, expires_at=10_000))
    item = store.get(sessions.TABLE, {"call_sid": f"{SID}{sessions.SUFFIX}"})
    assert item is not None
    del item["keyed_tries"]
    store.put(sessions.TABLE, item)
    loaded = sessions.load(store, SID)
    assert loaded is not None and loaded.keyed_tries == 0
