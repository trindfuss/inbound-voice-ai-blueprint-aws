"""Call state keeps attributes it has no field for.

A writer running older code loaded the call state, dropped the attributes its ``CallState`` didn't
know (``agent_owner``) and wrote the whole item back, so the bridge's next session started an empty
agent on a call with history (CA900e…9439). ``CallState.from_item`` now keeps unknown attributes and
``to_item`` writes them back, through every writer path (``create``, ``update``, ``mutate``).
"""

from __future__ import annotations

import string
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank import callstate as cs
from owlbank import segments as sg
from owlbank.store import MemoryStore

SID = "CA" + "e" * 32
PROPS = settings(max_examples=60, deadline=None)


@pytest.fixture(autouse=True)
def _clear() -> Any:
    sg.clear_sensitive(SID)
    yield
    sg.clear_sensitive(SID)


# DynamoDB-ish attribute names that are not CallState fields.
extra_names = st.text(alphabet=string.ascii_letters + string.digits + "_", min_size=1, max_size=20).filter(
    lambda k: k[0].isalpha() and k not in cs._FIELD_NAMES)  # noqa: SLF001
json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-10**9, 10**9) | st.text(string.ascii_letters + " -", max_size=12),
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(
        st.text(string.ascii_lowercase, min_size=1, max_size=6), inner, max_size=3),
    max_leaves=8)
extras = st.dictionaries(extra_names, json_values, min_size=1, max_size=4)
changes = st.one_of(
    st.builds(lambda n: {"level": n}, st.integers(0, 3)),
    st.sampled_from([{"language": "es-US"}, {"agent_owner": "p2"}, {"agent_owner": None},
                     {"pending_tool": {"tool_use_id": "tu1", "name": "get_balance", "args": {}}},
                     {"pending_tool": None, "pending_result": None},
                     {"escalation": {"reason": "caller_request", "at": 1}},
                     {"ws_nonce_hash": "ab" * 32}, {"ai_sessions": 2},
                     {"identity_source": "keyed_phone"}, {"identity_source": "ani"}]),
)


def _seed(store: MemoryStore, extra: dict[str, Any]) -> None:
    """An item written by newer code: the current fields plus attributes this code doesn't know."""
    item = cs.CallState(call_sid=SID, expires_at=10_000, agent_owner="p1", version=1).to_item()
    store.put(cs.TABLE, {**item, **extra})


@PROPS
@given(extras, st.lists(changes, min_size=1, max_size=4))
def test_update_keeps_unknown_attributes(extra: dict[str, Any], steps: list[dict[str, Any]]) -> None:
    store = MemoryStore()
    _seed(store, extra)
    for change in steps:
        state = cs.load(store, SID)
        assert state is not None
        cs.update(store, state, **change)
        raw = store.get(cs.TABLE, {"call_sid": SID})
        assert raw is not None
        assert {k: raw[k] for k in extra} == extra  # unchanged, whatever the update changed
        loaded = cs.load(store, SID)
        assert loaded is not None
        for k, v in change.items():
            got = getattr(loaded, k)
            assert (got if not isinstance(got, cs.PendingTool) else
                    {"tool_use_id": got.tool_use_id, "name": got.name, "args": got.args}) == v


@PROPS
@given(extras, changes)
def test_mutate_and_create_keep_unknown_attributes(extra: dict[str, Any], change: dict[str, Any]) -> None:
    store = MemoryStore()
    cs.create(store, cs.CallState(call_sid=SID, expires_at=10_000, _extras=extra))
    cs.mutate(store, SID, lambda _s: change)
    raw = store.get(cs.TABLE, {"call_sid": SID})
    assert raw is not None and {k: raw[k] for k in extra} == extra
    loaded = cs.load(store, SID)
    assert loaded is not None and cs.CallState.from_item(loaded.to_item()).to_item() == raw


def test_an_older_schema_without_agent_owner_keeps_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The incident: a writer whose CallState has no ``agent_owner`` (nor ``ai_sessions``) rewrites
    the item. Before the fix ``from_item`` dropped both and the whole-item put erased them."""
    store = MemoryStore()
    cs.create(store, cs.CallState(call_sid=SID, expires_at=10_000))
    cs.mutate(store, SID, lambda _s: {cs.F_AGENT_OWNER: "bridge-1", cs.F_AI_SESSIONS: 1})
    older = cs._FIELD_NAMES - {cs.F_AGENT_OWNER, cs.F_AI_SESSIONS}  # noqa: SLF001
    with monkeypatch.context() as m:
        m.setattr(cs, "_FIELD_NAMES", older)  # the older code's schema
        state = cs.load(store, SID)
        assert state is not None and state.agent_owner is None  # it doesn't know the field
        cs.update(store, state, level=1, ws_nonce_hash="cd" * 32)  # what the old front door wrote
    after = cs.load(store, SID)
    assert after is not None
    assert (after.agent_owner, after.ai_sessions, after.level) == ("bridge-1", 1, 1)


def test_extras_cannot_shadow_a_field() -> None:
    with pytest.raises(ValueError, match="collide"):
        cs.CallState(call_sid=SID, expires_at=10_000, _extras={"level": 3})


def test_extras_are_not_part_of_equality_and_are_copied() -> None:
    nested = {"a": [1]}
    a = cs.CallState(call_sid=SID, expires_at=10_000, _extras={"future": nested})
    assert a == cs.CallState(call_sid=SID, expires_at=10_000)
    item = a.to_item()
    item["future"]["a"].append(2)
    assert a.to_item()["future"] == {"a": [1]}


def test_the_redaction_guard_scans_unknown_attributes() -> None:
    store = MemoryStore()
    _seed(store, {"future_note": "zip 94110"})
    sg.register_sensitive(SID, ["94110"])
    state = cs.load(store, SID)
    assert state is not None
    with pytest.raises(sg.RedactionError):
        cs.update(store, state, level=1)
