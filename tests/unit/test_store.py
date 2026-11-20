"""One Store contract suite, run against MemoryStore and DynamoStore (moto)."""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import pytest

from owlbank.config import Settings
from owlbank.store import (
    ALL_SCHEMAS,
    BATCH_GET_RETRY_DELAYS_S,
    FAILOVER_SCHEMAS,
    MAX_BATCH_GET_KEYS,
    SCHEMAS,
    ConditionFailed,
    DynamoStore,
    MemoryStore,
    Store,
    StoreError,
    create_table_kwargs,
)


@pytest.fixture
def aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake credentials so nothing can reach a real AWS account."""
    for var, val in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_REGION": "us-east-1",
    }.items():
        monkeypatch.setenv(var, val)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture(params=["memory", "dynamo"])
def store(request: pytest.FixtureRequest, aws_env: None) -> Iterator[Store]:
    if request.param == "memory":
        yield MemoryStore()
        return
    import boto3
    from moto import mock_aws

    settings = Settings(aws_region="us-east-1")
    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-east-1")
        for logical in SCHEMAS:
            client.create_table(**create_table_kwargs(logical, settings.table(logical)))
        yield DynamoStore.from_settings(settings)


def test_implements_protocol(store: Store) -> None:
    assert isinstance(store, Store)


def test_get_missing_returns_none(store: Store) -> None:
    assert store.get("calls", {"call_sid": "CA-none"}) is None


def test_put_get_round_trip_preserves_types(store: Store) -> None:
    item = {
        "call_sid": "CA1",
        "language": "es-US",
        "level": 2,
        "verified": True,
        "nothing": None,
        "factors_verified": ["zip", "ssn4"],
        "pending_tool": {"name": "get_balance", "args": {"account": "checking"}, "n": 3},
    }
    store.put("calls", item)
    got = store.get("calls", {"call_sid": "CA1"})
    assert got == item
    assert type(got["level"]) is int and type(got["pending_tool"]["n"]) is int


def test_returned_items_are_copies(store: Store) -> None:
    store.put("calls", {"call_sid": "CA1", "attempts": [1]})
    got = store.get("calls", {"call_sid": "CA1"})
    got["attempts"].append(2)
    assert store.get("calls", {"call_sid": "CA1"})["attempts"] == [1]


def test_put_replaces_whole_item(store: Store) -> None:
    store.put("calls", {"call_sid": "CA1", "a": "1", "b": "2"})
    store.put("calls", {"call_sid": "CA1", "a": "3"})
    assert store.get("calls", {"call_sid": "CA1"}) == {"call_sid": "CA1", "a": "3"}


def test_put_condition_not_exists(store: Store) -> None:
    store.put("tokens", {"token_hash": "h1", "call_sid": "CA1"}, condition="not_exists")
    with pytest.raises(ConditionFailed):
        store.put("tokens", {"token_hash": "h1", "call_sid": "CA2"}, condition="not_exists")
    assert store.get("tokens", {"token_hash": "h1"})["call_sid"] == "CA1"


def test_put_condition_exists(store: Store) -> None:
    with pytest.raises(ConditionFailed):
        store.put("calls", {"call_sid": "CA1"}, condition="exists")
    store.put("calls", {"call_sid": "CA1"})
    store.put("calls", {"call_sid": "CA1", "x": "y"}, condition="exists")
    assert store.get("calls", {"call_sid": "CA1"})["x"] == "y"


def test_composite_key_get_and_query_order(store: Store) -> None:
    for seq in (3, 1, 2):
        store.put("segments", {"call_sid": "CA1", "seq": seq, "type": f"t{seq}"})
    store.put("segments", {"call_sid": "CA2", "seq": 1, "type": "other"})
    assert store.get("segments", {"call_sid": "CA1", "seq": 2})["type"] == "t2"
    rows = store.query("segments", {"call_sid": "CA1"})
    assert [r["seq"] for r in rows] == [1, 2, 3]
    assert store.query("segments", {"call_sid": "CA1", "seq": 3})[0]["type"] == "t3"
    assert store.query("segments", {"call_sid": "CA-none"}) == []


def test_catalog_key_lang(store: Store) -> None:
    store.put("catalog", {"key": "greeting.generic", "lang": "en-US", "text": "Welcome"})
    store.put("catalog", {"key": "greeting.generic", "lang": "es-US", "text": "Bienvenido"})
    rows = store.query("catalog", {"key": "greeting.generic"})
    assert [r["lang"] for r in rows] == ["en-US", "es-US"]


def test_query_by_index_finds_shared_phone(store: Store) -> None:
    store.put("customers", {"customer_id": "C-1", "phone": "+12025550102", "first_name": "Sam"})
    store.put("customers", {"customer_id": "C-2", "phone": "+12025550102", "first_name": "Jordan"})
    store.put("customers", {"customer_id": "C-3", "phone": "+12025550101"})
    store.put("customers", {"customer_id": "C-4"})  # no phone: not in the index
    rows = store.query("customers", {"phone": "+12025550102"}, index="phone-index")
    assert sorted(r["customer_id"] for r in rows) == ["C-1", "C-2"]
    assert store.query("customers", {"phone": "+12025550199"}, index="phone-index") == []


def test_tokens_by_call_sid_index(store: Store) -> None:
    store.put("tokens", {"token_hash": "h1", "call_sid": "CA1", "level": 1})
    store.put("tokens", {"token_hash": "h2", "call_sid": "CA1", "level": 3})
    store.put("tokens", {"token_hash": "h3", "call_sid": "CA2", "level": 1})
    rows = store.query("tokens", {"call_sid": "CA1"}, index="call_sid-index")
    assert sorted(r["token_hash"] for r in rows) == ["h1", "h2"]


def test_sweep_due_index_is_sparse_keys_only_and_ordered(store: Store) -> None:
    """A range query on the ``calls`` table's ``sweep-due`` index returns only the items
    carrying both index keys, oldest due first, inclusive bounds, with the key attributes only."""
    store.put("calls", {"call_sid": "CA3", "sweep_open": "open", "sweep_due_at": 30, "escalation": {"r": "x"}})
    store.put("calls", {"call_sid": "CA1", "sweep_open": "open", "sweep_due_at": 10, "escalation": {"r": "x"}})
    store.put("calls", {"call_sid": "CA2", "sweep_open": "open", "sweep_due_at": 20})
    store.put("calls", {"call_sid": "CA4", "ai_sessions": 1})  # AI-only: not in the index
    store.put("calls", {"call_sid": "CA5", "sweep_open": "open", "sweep_due_at": 99})  # not due yet
    got = store.query_range("calls", {"sweep_open": "open"}, low=0, high=30, index="sweep-due")
    assert [i["call_sid"] for i in got] == ["CA1", "CA2", "CA3"]
    assert got[0] == {"call_sid": "CA1", "sweep_open": "open", "sweep_due_at": 10}
    assert store.query_range("calls", {"sweep_open": "open"}, low=0, high=9, index="sweep-due") == []


def test_index_range_query_refuses_a_consistent_read_and_bad_bounds(store: Store) -> None:
    with pytest.raises(StoreError, match="consistently"):
        store.query_range("calls", {"sweep_open": "open"}, low=0, high=1, index="sweep-due", consistent=True)
    with pytest.raises(StoreError):
        store.query_range("calls", {"sweep_open": "open"}, low="0", high="1", index="sweep-due")
    with pytest.raises(StoreError):
        store.query_range("calls", {"call_sid": "CA1"}, low=0, high=1, index="sweep-due")


def test_update_sets_removes_and_returns_new_item(store: Store) -> None:
    store.put("calls", {"call_sid": "CA1", "level": 1, "pending_tool": {"name": "x"}})
    new = store.update("calls", {"call_sid": "CA1"}, {"level": 3, "language": "en-US"},
                       remove=["pending_tool"])
    assert new == {"call_sid": "CA1", "level": 3, "language": "en-US"}
    assert store.get("calls", {"call_sid": "CA1"}) == new


def test_update_missing_item_fails(store: Store) -> None:
    with pytest.raises(ConditionFailed):
        store.update("calls", {"call_sid": "CA-none"}, {"level": 1})
    assert store.get("calls", {"call_sid": "CA-none"}) is None


def test_increment_creates_the_item_with_defaults_and_changes(store: Store) -> None:
    key = {"call_sid": "CA1#gather#g1"}
    store.increment("calls", key, "n", defaults={"expires_at": 100},
                    changes={"len": 3, "at": 10}, unless_equal=("len", 3))
    assert store.get("calls", key) == {**key, "n": 1, "expires_at": 100, "len": 3, "at": 10}


def test_increment_skips_changes_when_unchanged_and_keeps_defaults_and_other_fields(store: Store) -> None:
    key = {"call_sid": "CA1#gather#g1"}
    store.put("calls", {**key, "state": "early", "expires_at": 50, "len": 3, "at": 10, "n": 4})
    store.increment("calls", key, "n", defaults={"expires_at": 100},
                    changes={"len": 3, "at": 20}, unless_equal=("len", 3))
    assert store.get("calls", key) == {**key, "state": "early", "expires_at": 50, "len": 3,
                                       "at": 10, "n": 5}
    store.increment("calls", key, "n", changes={"len": 5, "at": 30}, unless_equal=("len", 5))
    assert store.get("calls", key) == {**key, "state": "early", "expires_at": 50, "len": 5,
                                       "at": 30, "n": 6}


def test_increment_without_condition_always_sets_changes(store: Store) -> None:
    key = {"call_sid": "CA2"}
    store.increment("calls", key, "n", by=2, changes={"at": 1})
    store.increment("calls", key, "n", by=3, changes={"at": 2})
    assert store.get("calls", key) == {**key, "n": 5, "at": 2}


def test_set_if_greater_creates_advances_and_refuses_older(store: Store) -> None:
    # Keyed-phone: a partial's SequenceNumber only ever moves forward.
    key = {"call_sid": "CA4#gather#g1"}
    assert store.set_if_greater("calls", key, "seq", 3, defaults={"expires_at": 100},
                                changes={"id": "a"})
    assert store.get("calls", key) == {**key, "seq": 3, "expires_at": 100, "id": "a"}
    store.add_counters("calls", key, {"partial_count": 1})
    assert not store.set_if_greater("calls", key, "seq", 2, changes={"id": "b"})   # older
    assert not store.set_if_greater("calls", key, "seq", 3, changes={"id": "c"})   # same
    assert store.get("calls", key) == {**key, "seq": 3, "expires_at": 100, "id": "a",
                                       "partial_count": 1}
    assert store.set_if_greater("calls", key, "seq", 7, defaults={"expires_at": 1},
                                changes={"id": "d"})
    assert store.get("calls", key) == {**key, "seq": 7, "expires_at": 100, "id": "d",
                                       "partial_count": 1}
    # An item without the attribute (made by another write) takes any value.
    other = {"call_sid": "CA4#gather#g2"}
    store.add_counters("calls", other, {"partial_count": 1})
    assert store.set_if_greater("calls", other, "seq", 0, changes={"id": "e"})
    assert store.get("calls", other) == {**other, "partial_count": 1, "seq": 0, "id": "e"}
    assert not store.set_if_greater("calls", {"call_sid": "CA4#gather#g3"}, "seq", 5) or True


def test_set_if_greater_leaves_no_item_when_refused(store: Store) -> None:
    key = {"call_sid": "CA5"}
    store.put("calls", {**key, "seq": 9})
    assert not store.set_if_greater("calls", key, "seq", 1, defaults={"x": 1})
    assert store.get("calls", key) == {**key, "seq": 9}


def test_set_if_greater_rejects_keys_floats_bools_and_duplicates(store: Store) -> None:
    key = {"call_sid": "CA6"}
    with pytest.raises(StoreError):
        store.set_if_greater("calls", key, "call_sid", 1)
    with pytest.raises(StoreError):
        store.set_if_greater("calls", key, "seq", 1, changes={"seq": 2})
    for bad in (1.5, True, "3"):
        with pytest.raises(TypeError):
            store.set_if_greater("calls", key, "seq", bad)  # type: ignore[arg-type]
    assert store.get("calls", key) is None


def test_increment_rejects_keys_floats_and_duplicates(store: Store) -> None:
    key = {"call_sid": "CA3"}
    with pytest.raises(StoreError):
        store.increment("calls", key, "call_sid")
    with pytest.raises(StoreError):
        store.increment("calls", key, "n", changes={"n": 1})
    with pytest.raises(TypeError):
        store.increment("calls", key, "n", changes={"x": 1.5})
    with pytest.raises(TypeError):
        store.increment("calls", key, "n", by=True)
    assert store.get("calls", key) is None


def test_delete(store: Store) -> None:
    store.put("tokens", {"token_hash": "h1", "call_sid": "CA1"})
    store.delete("tokens", {"token_hash": "h1"})
    assert store.get("tokens", {"token_hash": "h1"}) is None
    store.delete("tokens", {"token_hash": "h1"})  # absent: no error


@pytest.mark.parametrize("bad", [1.5, Decimal("1.5"), {1: "x"}, {"s": {1, 2}}, b"bytes"])
def test_rejects_unsupported_values(store: Store, bad: object) -> None:
    with pytest.raises(TypeError):
        store.put("calls", {"call_sid": "CA1", "v": bad})


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.get("calls", {"call_sid": "CA1", "extra": "x"}),
        lambda s: s.get("segments", {"call_sid": "CA1"}),
        lambda s: s.get("nope", {"id": "x"}),
        lambda s: s.put("calls", {"language": "en-US"}),
        lambda s: s.put("segments", {"call_sid": "CA1", "seq": "1"}),
        lambda s: s.query("segments", {"seq": 1}),
        lambda s: s.query("customers", {"phone": "+1"}, index="missing-index"),
        lambda s: s.update("calls", {"call_sid": "CA1"}, {"call_sid": "CA2"}),
        lambda s: s.update("calls", {"call_sid": "CA1"}, {}),
        lambda s: s.batch_get("segments", [{"call_sid": "CA1"}]),
        lambda s: s.batch_get("nope", [{"id": "x"}]),
        lambda s: s.scan("nope"),
    ],
)
def test_rejects_bad_keys_and_tables(store: Store, call) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(StoreError):
        call(store)


def test_dynamo_store_construction_makes_no_aws_call() -> None:
    # No moto, no credentials: construction must still succeed (lazy client).
    DynamoStore(Settings().tables)
    with pytest.raises(StoreError):
        DynamoStore({"calls": "x"})


# --------------------------------------------------------------------------- batch_get and scan


def test_batch_get_returns_found_items_once(store: Store) -> None:
    for pk in ("meta", "tool#a", "tool#b"):
        store.put("policy", {"pk": pk, "v": pk})
    got = store.batch_get("policy", [{"pk": "tool#b"}, {"pk": "missing"}, {"pk": "meta"},
                                     {"pk": "tool#b"}])
    assert sorted(i["pk"] for i in got) == ["meta", "tool#b"]
    assert store.batch_get("policy", []) == []


def test_batch_get_composite_keys_and_copies(store: Store) -> None:
    store.put("segments", {"call_sid": "CA1", "seq": 2, "parts": [1]})
    (got,) = store.batch_get("segments", [{"call_sid": "CA1", "seq": 2}, {"call_sid": "CA1", "seq": 3}])
    assert got == {"call_sid": "CA1", "seq": 2, "parts": [1]} and type(got["seq"]) is int
    got["parts"].append(2)
    assert store.get("segments", {"call_sid": "CA1", "seq": 2})["parts"] == [1]


def test_batch_get_more_than_one_request_of_keys(store: Store) -> None:
    n = MAX_BATCH_GET_KEYS * 2 + 5
    for i in range(n):
        store.put("policy", {"pk": f"k{i}"})
    got = store.batch_get("policy", [{"pk": f"k{i}"} for i in range(n + 3)])
    assert sorted(i["pk"] for i in got) == sorted(f"k{i}" for i in range(n))


def test_scan_returns_every_item(store: Store) -> None:
    assert store.scan("catalog") == []
    store.put("catalog", {"key": "a", "lang": "en-US", "text": "A", "version": 2})
    store.put("catalog", {"key": "a", "lang": "es-US", "text": "Á"})
    store.put("catalog", {"key": "_meta", "lang": "_", "keys": ["a"]})
    got = store.scan("catalog")
    assert sorted((i["key"], i["lang"]) for i in got) == [("_meta", "_"), ("a", "en-US"), ("a", "es-US")]
    assert type(next(i for i in got if i["lang"] == "en-US")["version"]) is int
    got[0]["text"] = "changed"
    assert "changed" not in [i.get("text") for i in store.scan("catalog")]


class FakeBatchResource:
    """A boto3 resource stub: ``batch_get_item`` answers from ``rows``, leaving the first ``hold``
    keys of each of the first ``rounds`` responses unprocessed."""

    def __init__(self, rows: dict[str, dict[str, Any]], *, rounds: int, hold: int = 1) -> None:
        self.rows, self.rounds, self.hold = rows, rounds, hold
        self.requests: list[list[str]] = []

    def batch_get_item(self, *, RequestItems: dict[str, Any]) -> dict[str, Any]:  # noqa: N803
        ((physical, spec),) = RequestItems.items()
        assert spec["ConsistentRead"] is True and len(spec["Keys"]) <= MAX_BATCH_GET_KEYS
        keys = [k["pk"] for k in spec["Keys"]]
        self.requests.append(keys)
        held = keys[:self.hold] if len(self.requests) <= self.rounds else []
        done = [self.rows[k] for k in keys if k not in held and k in self.rows]
        out: dict[str, Any] = {"Responses": {physical: done}}
        if held:
            out["UnprocessedKeys"] = {physical: {"Keys": [{"pk": k} for k in held], "ConsistentRead": True}}
        return out

    def Table(self, name: str) -> Any:  # noqa: N802 (boto3 resource API)
        return None


def _dynamo(resource: Any, sleeps: list[float]) -> DynamoStore:
    return DynamoStore({t: f"owlbank-{t}" for t in SCHEMAS}, resource=resource, sleep=sleeps.append)


def test_batch_get_retries_unprocessed_keys_with_backoff() -> None:
    rows = {pk: {"pk": pk, "n": Decimal(i)} for i, pk in enumerate(["a", "b", "c"])}
    sleeps: list[float] = []
    resource = FakeBatchResource(rows, rounds=2)
    got = _dynamo(resource, sleeps).batch_get("policy", [{"pk": "a"}, {"pk": "b"}, {"pk": "c"}])
    assert sorted(i["pk"] for i in got) == ["a", "b", "c"] and all(type(i["n"]) is int for i in got)
    assert resource.requests == [["a", "b", "c"], ["a"], ["a"]]
    assert len(sleeps) == 2
    for wait, bound in zip(sleeps, BATCH_GET_RETRY_DELAYS_S, strict=False):
        assert bound / 2 <= wait <= bound


def test_batch_get_gives_up_on_keys_still_unprocessed() -> None:
    sleeps: list[float] = []
    resource = FakeBatchResource({"a": {"pk": "a"}}, rounds=99)
    with pytest.raises(StoreError, match="unprocessed"):
        _dynamo(resource, sleeps).batch_get("policy", [{"pk": "a"}])
    assert len(resource.requests) == len(BATCH_GET_RETRY_DELAYS_S) + 1
    assert len(sleeps) == len(BATCH_GET_RETRY_DELAYS_S) and sum(sleeps) <= 0.3


def test_batch_get_sends_chunks_of_at_most_100_keys() -> None:
    n = MAX_BATCH_GET_KEYS + 1
    resource = FakeBatchResource({f"k{i}": {"pk": f"k{i}"} for i in range(n)}, rounds=0)
    got = _dynamo(resource, []).batch_get("policy", [{"pk": f"k{i}"} for i in range(n)])
    assert len(got) == n and [len(r) for r in resource.requests] == [MAX_BATCH_GET_KEYS, 1]


def test_scan_follows_every_page() -> None:
    pages: list[dict[str, Any]] = [{"Items": [{"key": "a", "lang": "x"}], "LastEvaluatedKey": {"key": "a", "lang": "x"}},
             {"Items": [{"key": "b", "lang": "x", "v": Decimal(3)}]}]
    calls: list[dict[str, Any]] = []

    class Table:
        def scan(self, **kw: Any) -> dict[str, Any]:
            calls.append(kw)
            return pages[len(calls) - 1]

    class Resource:
        def Table(self, name: str) -> Table:  # noqa: N802
            assert name == "owlbank-catalog"
            return Table()

    got = _dynamo(Resource(), []).scan("catalog")
    assert got == [{"key": "a", "lang": "x"}, {"key": "b", "lang": "x", "v": 3}]
    assert calls == [{"ConsistentRead": True},
                     {"ConsistentRead": True, "ExclusiveStartKey": {"key": "a", "lang": "x"}}]


# --------------------------------------------------------------------------- model failover table
# (FAILOVER_SCHEMAS, a sort-key range Query, several counters in one write, an eventually
# consistent GetItem)


@pytest.fixture(params=["memory", "dynamo"])
def fstore(request: pytest.FixtureRequest, aws_env: None) -> Iterator[Store]:
    if request.param == "memory":
        yield MemoryStore()
        return
    import boto3
    from moto import mock_aws

    with mock_aws():
        boto3.client("dynamodb", region_name="us-east-1").create_table(
            **create_table_kwargs("modelstats", "owlbank-modelstats"))
        yield DynamoStore({"modelstats": "owlbank-modelstats"}, required=["modelstats"],
                          region_name="us-east-1")


def test_failover_table_is_separate_from_the_data_stack_tables() -> None:
    assert "modelstats" not in SCHEMAS and "modelstats" in ALL_SCHEMAS
    assert FAILOVER_SCHEMAS["modelstats"].key_attrs == ("s", "sk")
    # no existing component needs its physical name
    DynamoStore({t: f"owlbank-{t}" for t in SCHEMAS})
    assert "modelstats" not in Settings().tables


def test_add_counters_creates_then_adds_and_keeps_defaults(fstore: Store) -> None:
    key = {"s": "samples", "sk": "1800000000#0"}
    fstore.add_counters("modelstats", key, {"n": 3, "h2": 3}, defaults={"expires_at": 1_800_000_900})
    fstore.add_counters("modelstats", key, {"n": 1, "miss": 1}, defaults={"expires_at": 1})
    got = fstore.get("modelstats", key)
    assert got == {**key, "n": 4, "h2": 3, "miss": 1, "expires_at": 1_800_000_900}
    assert all(type(got[a]) is int for a in ("n", "h2", "miss", "expires_at"))


def test_add_counters_refuses_bad_input(fstore: Store) -> None:
    key = {"s": "samples", "sk": "1800000000#0"}
    with pytest.raises(StoreError):
        fstore.add_counters("modelstats", key, {})
    with pytest.raises(StoreError):
        fstore.add_counters("modelstats", key, {"sk": 1})
    with pytest.raises(StoreError):
        fstore.add_counters("modelstats", key, {"n": 1}, defaults={"n": 0})
    with pytest.raises(TypeError):
        fstore.add_counters("modelstats", key, {"n": 1.5})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        fstore.add_counters("modelstats", key, {"n": True})
    with pytest.raises(StoreError):
        fstore.add_counters("modelstats", {"s": "samples"}, {"n": 1})


def test_query_range_is_inclusive_ordered_and_one_partition(fstore: Store) -> None:
    for b in (980, 990, 1000, 1010):
        for i in (0, 1):
            fstore.add_counters("modelstats", {"s": "samples", "sk": f"{b:010d}#{i}"}, {"n": 1})
    fstore.put("modelstats", {"s": "state", "sk": "state", "version": 1})
    fstore.put("modelstats", {"s": "other", "sk": f"{990:010d}#0", "n": 9})
    got = fstore.query_range("modelstats", {"s": "samples"}, low=f"{990:010d}", high=f"{1000:010d}#~")
    assert [i["sk"] for i in got] == [f"{b:010d}#{i}" for b in (990, 1000) for i in (0, 1)]
    assert fstore.query_range("modelstats", {"s": "samples"}, low="z", high="zz") == []
    got[0]["n"] = 99
    assert fstore.query_range("modelstats", {"s": "samples"}, low=got[0]["sk"], high=got[0]["sk"])[0]["n"] == 1


def test_query_range_refuses_bad_input(fstore: Store) -> None:
    with pytest.raises(StoreError):
        fstore.query_range("modelstats", {"s": "samples", "sk": "x"}, low="a", high="b")
    with pytest.raises(StoreError):
        fstore.query_range("modelstats", {"s": "samples"}, low=1, high=2)
    with pytest.raises(StoreError):
        fstore.query_range("calls", {"call_sid": "CA1"}, low="a", high="b")


def test_eventually_consistent_get_reads_the_item(fstore: Store) -> None:
    fstore.put("modelstats", {"s": "state", "sk": "state", "version": 2})
    assert fstore.get("modelstats", {"s": "state", "sk": "state"}, consistent=False) == \
        {"s": "state", "sk": "state", "version": 2}


class RecordingTable:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_item(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("get_item", kw))
        return {}

    def query(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("query", kw))
        if "ExclusiveStartKey" in kw:
            return {"Items": [{"s": "samples", "sk": "2", "n": Decimal(2)}]}
        return {"Items": [{"s": "samples", "sk": "1"}], "LastEvaluatedKey": {"s": "samples", "sk": "1"}}

    def update_item(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("update_item", kw))
        return {}


def _recording() -> tuple[DynamoStore, RecordingTable]:
    table = RecordingTable()

    class Resource:
        def Table(self, name: str) -> Any:  # noqa: N802 (boto3 resource API)
            return table

    return DynamoStore({"modelstats": "m"}, required=["modelstats"], resource=Resource()), table


def test_dynamo_reads_are_eventually_consistent_only_when_asked() -> None:
    store, table = _recording()
    store.get("modelstats", {"s": "a", "sk": "state"}, consistent=False)
    store.get("modelstats", {"s": "a", "sk": "state"})
    got = store.query_range("modelstats", {"s": "samples"}, low="0", high="9")
    assert [i["sk"] for i in got] == ["1", "2"] and type(got[1]["n"]) is int
    flags = [kw["ConsistentRead"] for _, kw in table.calls]
    assert flags == [False, True, False, False]


def test_dynamo_add_counters_is_one_update_item() -> None:
    store, table = _recording()
    store.add_counters("modelstats", {"s": "samples", "sk": "x"}, {"n": 2, "h1": 2}, defaults={"expires_at": 5})
    ((op, kw),) = table.calls
    assert op == "update_item"
    assert kw["UpdateExpression"] == "ADD #c0 :c0, #c1 :c1 SET #d0 = if_not_exists(#d0, :d0)"
    assert kw["ExpressionAttributeNames"] == {"#c0": "n", "#c1": "h1", "#d0": "expires_at"}
    assert kw["ExpressionAttributeValues"] == {":c0": 2, ":c1": 2, ":d0": 5}
