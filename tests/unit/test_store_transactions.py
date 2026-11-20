"""``Store.transact_write``: one contract, run against MemoryStore and DynamoStore (moto)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.config import Settings
from owlbank.store import (
    ADMIN_SCHEMAS,
    ALL_SCHEMAS,
    SCHEMAS,
    ConditionFailed,
    DynamoStore,
    Expect,
    MemoryStore,
    Store,
    StoreError,
    TransactionCancelled,
    TxPut,
    create_table_kwargs,
)


@pytest.fixture
def aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var, val in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
                     "AWS_SESSION_TOKEN": "testing", "AWS_DEFAULT_REGION": "us-east-1",
                     "AWS_REGION": "us-east-1"}.items():
        monkeypatch.setenv(var, val)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture(params=["memory", "dynamo"])
def store(request: pytest.FixtureRequest, aws_env: None) -> Iterator[Store]:
    if request.param == "memory":
        yield MemoryStore()
        return
    import boto3
    from moto import mock_aws

    names = {t: f"t-{t}" for t in ALL_SCHEMAS}
    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-east-1")
        for logical in ALL_SCHEMAS:
            client.create_table(**create_table_kwargs(logical, names[logical]))
        yield DynamoStore(names, region_name="us-east-1")


def _audit(sk: str = "2026-10-15T00:00:00.000000Z#a") -> dict:
    return {"pk": "policy#masking", "sk": sk, "actor_sub": "u1"}


def test_all_puts_are_written(store: Store) -> None:
    store.put("policy", {"pk": "masking", "rev": 1, "after_ms": 2000})
    store.transact_write([
        TxPut("policy", {"pk": "masking", "rev": 2, "after_ms": 1500},
              Expect("rev", 1, absent_as=0, must_exist=True)),
        TxPut("admin_audit", _audit(), "not_exists"),
    ])
    assert store.get("policy", {"pk": "masking"}) == {"pk": "masking", "rev": 2, "after_ms": 1500}
    assert store.query("admin_audit", {"pk": "policy#masking"}) == [_audit()]


def test_a_failed_condition_writes_nothing(store: Store) -> None:
    store.put("policy", {"pk": "masking", "rev": 3, "after_ms": 2000})
    with pytest.raises(TransactionCancelled) as err:
        store.transact_write([
            TxPut("admin_audit", _audit(), "not_exists"),
            TxPut("policy", {"pk": "masking", "rev": 2, "after_ms": 1}, Expect("rev", 1)),
        ])
    assert err.value.failed == (1,)
    assert isinstance(err.value, ConditionFailed)
    assert store.get("policy", {"pk": "masking"})["after_ms"] == 2000
    assert store.query("admin_audit", {"pk": "policy#masking"}) == []


def test_missing_rev_counts_as_absent_as(store: Store) -> None:
    store.put("policy", {"pk": "tool#x", "min_level": 1})
    store.transact_write([TxPut("policy", {"pk": "tool#x", "min_level": 2, "rev": 1},
                                Expect("rev", 0, absent_as=0, must_exist=True))])
    assert store.get("policy", {"pk": "tool#x"})["rev"] == 1
    with pytest.raises(TransactionCancelled):  # rev is now 1, not absent
        store.transact_write([TxPut("policy", {"pk": "tool#x", "rev": 1},
                                    Expect("rev", 0, absent_as=0))])


def test_must_exist_and_absent_item(store: Store) -> None:
    with pytest.raises(TransactionCancelled):
        store.transact_write([TxPut("policy", {"pk": "tool#new", "rev": 1},
                                    Expect("rev", 0, absent_as=0, must_exist=True))])
    assert store.get("policy", {"pk": "tool#new"}) is None
    # Without must_exist, a missing item has a missing rev: it matches absent_as.
    store.transact_write([TxPut("catalog", {"key": "mask.x", "lang": "en-US", "text": "Hi", "rev": 1},
                                Expect("rev", 0, absent_as=0))])
    assert store.get("catalog", {"key": "mask.x", "lang": "en-US"})["rev"] == 1


def test_not_exists_and_exists_conditions(store: Store) -> None:
    store.transact_write([TxPut("admin_audit", _audit(), "not_exists")])
    with pytest.raises(TransactionCancelled) as err:
        store.transact_write([TxPut("admin_audit", _audit(), "not_exists")])
    assert err.value.failed == (0,)
    with pytest.raises(TransactionCancelled):
        store.transact_write([TxPut("policy", {"pk": "nope"}, "exists")])


def test_value_condition_is_type_strict(store: Store) -> None:
    store.put("policy", {"pk": "masking", "rev": "1"})
    with pytest.raises(TransactionCancelled):
        store.transact_write([TxPut("policy", {"pk": "masking", "rev": 2}, Expect("rev", 1))])


@pytest.mark.parametrize("puts", [
    [],
    [TxPut("nope", {"pk": "x"})],
    [TxPut("policy", {"pk": "a"}), TxPut("policy", {"pk": "a"})],
    [TxPut("policy", {"pk": "a"}, Expect("pk", "a"))],
    [TxPut("policy", {"pk": "a"}, "sometimes")],  # type: ignore[arg-type]
    [TxPut("policy", {"pk": str(i)}) for i in range(101)],
])
def test_rejects_bad_transactions(store: Store, puts: list[TxPut]) -> None:
    with pytest.raises(StoreError) as err:
        store.transact_write(puts)
    assert not isinstance(err.value, TransactionCancelled)


def test_rejects_floats_like_put(store: Store) -> None:
    with pytest.raises(TypeError):
        store.transact_write([TxPut("policy", {"pk": "a", "f": 1.5})])


def test_admin_tables_are_separate_from_the_data_stack() -> None:
    assert set(ADMIN_SCHEMAS) == {"admin_audit"} and not set(ADMIN_SCHEMAS) & set(SCHEMAS)
    assert ADMIN_SCHEMAS["admin_audit"].key_attrs == ("pk", "sk")


def test_dynamo_store_with_only_the_tables_it_needs() -> None:
    names = {"policy": "p", "catalog": "c", "admin_audit": "a"}
    store = DynamoStore(names, required=names)
    with pytest.raises(StoreError):
        store.get("calls", {"call_sid": "CA1"})
    with pytest.raises(StoreError):
        store.transact_write([TxPut("calls", {"call_sid": "CA1"})])
    with pytest.raises(StoreError):
        DynamoStore({"policy": "p"}, required=["policy", "admin_audit"])
    with pytest.raises(StoreError):
        DynamoStore({"bogus": "p"}, required=[])
    DynamoStore(Settings().tables)  # the default still needs every Data-stack table


# --------------------------------------------------------------------------- property: all or nothing

_keys = st.sampled_from(["a", "b", "c"])
_put = st.builds(
    lambda k, rev, cond, expect_rev, absent, must: TxPut(
        "policy", {"pk": k, "rev": rev},
        {"none": None, "ne": "not_exists", "ex": "exists",
         "expect": Expect("rev", expect_rev, absent_as=0 if absent else None, must_exist=must)}[cond]),
    _keys, st.integers(0, 3), st.sampled_from(["none", "ne", "ex", "expect"]), st.integers(0, 3),
    st.booleans(), st.booleans())


@settings(max_examples=200, deadline=None)
@given(st.lists(st.lists(_put, min_size=1, max_size=3, unique_by=lambda p: p.item["pk"]), max_size=8))
def test_memory_transactions_are_all_or_nothing(batches: list[list[TxPut]]) -> None:
    store = MemoryStore()
    model: dict[str, dict] = {}
    for puts in batches:
        expected_ok = all(
            p.condition is None
            or (p.condition == "not_exists" and p.item["pk"] not in model)
            or (p.condition == "exists" and p.item["pk"] in model)
            or (isinstance(p.condition, Expect) and p.condition.holds(model.get(p.item["pk"])))
            for p in puts)
        try:
            store.transact_write(puts)
            assert expected_ok
            for p in puts:
                model[p.item["pk"]] = dict(p.item)
        except TransactionCancelled:
            assert not expected_ok
        for k in ("a", "b", "c"):
            assert store.get("policy", {"pk": k}) == model.get(k)
