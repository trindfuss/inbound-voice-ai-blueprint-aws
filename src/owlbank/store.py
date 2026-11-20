"""Storage behind one small protocol.

Callers use *logical* table names (``customers``, ``tokens``, ``calls``,
``segments``, ``policy``, ``catalog``, ``risk``). Each has a fixed key schema in
:data:`SCHEMAS`; the CDK app creates the DynamoDB tables from the same schema.

Two implementations share one contract (``tests/unit/test_store.py``):

* :class:`MemoryStore` for unit, property and component tests.
* :class:`DynamoStore` over boto3, mapping logical names to physical tables.

Contract, in brief:

* **Values:** ``str``, ``int``, ``bool``, ``None``, ``list`` and ``dict`` (string
  keys), nested freely. ``float`` and ``Decimal`` are rejected (amounts are
  decimal strings). Integers come back as ``int``.
* **Keys:** ``get``/``update``/``delete`` take exactly the table's key attributes.
* **Copies:** returned items are independent copies; mutating them changes nothing.
* **Conditions:** ``put(condition="not_exists")`` fails with :class:`ConditionFailed`
  if an item with that key exists; ``"exists"`` fails if it doesn't. ``update``
  always requires the item to exist.
* **Query:** equality on the partition key of the table or index, optionally
  also on its sort key. Results are ordered by sort key when the table/index has
  one; otherwise order is unspecified.
* **Batch get**: :meth:`Store.batch_get` returns the items found for a list of
  keys (missing keys are left out; a key listed twice is read once), in no particular
  order. ``DynamoStore`` sends ``BatchGetItem`` in chunks of :data:`MAX_BATCH_GET_KEYS`,
  consistent reads, and retries ``UnprocessedKeys`` with a short jittered backoff
  (:data:`BATCH_GET_RETRY_DELAYS_S`); keys still unprocessed after that raise
  :class:`StoreError`, so a reader fails closed rather than seeing a partial result.
* **Scan**: :meth:`Store.scan` returns every item of a table (all pages,
  consistent reads), in no particular order. Only for the small config tables
  (``catalog``) read whole by a cache reload.
* **TTL:** expiry (``expires_at``) is a DynamoDB backstop and is lazy; readers
  must check ``expires_at`` themselves. ``MemoryStore`` never expires items.
* **Transactions**: :meth:`Store.transact_write` writes several whole items
  (:class:`TxPut`) all or nothing. Each put may carry a condition: ``"not_exists"``,
  ``"exists"`` or an :class:`Expect` (an attribute equals a value, a missing attribute
  counting as ``absent_as``, optionally requiring the item to exist). If any condition
  fails, nothing is written and :class:`TransactionCancelled` (a :class:`ConditionFailed`)
  names the failed puts. At most :data:`MAX_TRANSACTION_ITEMS` puts, each item at most once.
* **Counters** (call observability): :meth:`Store.increment` adds to a numeric attribute in one
  write that creates the item when missing (``defaults`` set only where absent), and sets
  ``changes`` too unless the stored ``unless_equal`` attribute already holds that value. Exact
  under concurrent writers (DynamoDB ``ADD``); ``DynamoStore`` sends one ``UpdateItem``, or a
  second one without ``changes`` when the first one's condition fails.

* **Fleet counters**: :meth:`Store.add_counters` adds to several
  numeric attributes in one write (one ``UpdateItem``, one write unit) that creates the item when
  missing, setting ``defaults`` only where absent. :meth:`Store.query_range` reads one partition's
  items whose sort key is ``BETWEEN`` two bounds (inclusive, ordered by sort key), eventually
  consistent by default. ``get(..., consistent=False)`` is an eventually consistent ``GetItem``
  (half a read unit); every other read stays strongly consistent.

The admin console's audit table (``admin_audit``) is in :data:`ADMIN_SCHEMAS`, not
:data:`SCHEMAS`: it belongs to the ``OwlBankAdmin`` stack, not the Data stack. The model-failover
table (``modelstats``) is in :data:`FAILOVER_SCHEMAS`, always built (in the Data stack). Both
stores accept them; a :class:`DynamoStore` needs a physical name for one only when it's used.
"""

from __future__ import annotations

import copy
import logging
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal, Protocol, runtime_checkable

from owlbank.obs import SLOW_CALL_MS, emit, timed_call

#: A failed condition is normal control flow (rate-limit slots, one-time claims): DEBUG, not a failure.
_CONDITION_CODES = ("ConditionalCheckFailedException",)

Item = dict[str, Any]
Condition = Literal["not_exists", "exists"]


class StoreError(Exception):
    """Base class for store errors."""


class ConditionFailed(StoreError):
    """A put/update condition did not hold."""


class TransactionCancelled(ConditionFailed):
    """A :meth:`Store.transact_write` was cancelled; nothing was written.

    ``failed`` holds the indexes of the puts whose condition failed (or that conflicted with
    another transaction in flight)."""

    def __init__(self, message: str, failed: Sequence[int] = ()) -> None:
        super().__init__(message)
        self.failed: tuple[int, ...] = tuple(failed)


#: DynamoDB's ``TransactWriteItems`` limit.
MAX_TRANSACTION_ITEMS = 100

#: DynamoDB's ``BatchGetItem`` limit (keys per request).
MAX_BATCH_GET_KEYS = 100
#: The upper bound of each wait before re-sending ``UnprocessedKeys`` (seconds; each wait is
#: drawn between half the bound and the bound). Four retries, at most 0.3 s in all: a reload runs
#: under the cache lock, so it gives up soon and the cache keeps its last good copy.
BATCH_GET_RETRY_DELAYS_S: tuple[float, ...] = (0.02, 0.04, 0.08, 0.16)


@dataclass(frozen=True, slots=True)
class Expect:
    """A value condition on the stored item (optimistic concurrency).

    Holds when ``attr`` equals ``value``; a missing attribute (or a missing item) counts as
    ``absent_as``, so ``Expect("rev", 0, absent_as=0)`` holds for an item without ``rev``.
    With ``must_exist`` the item must also exist."""

    attr: str
    value: str | int
    absent_as: str | int | None = None
    must_exist: bool = False

    @property
    def absent_matches(self) -> bool:
        return self.absent_as is not None and type(self.absent_as) is type(self.value) \
            and self.absent_as == self.value

    def holds(self, item: Mapping[str, Any] | None) -> bool:
        if item is None:
            return not self.must_exist and self.absent_matches
        if self.attr not in item:
            return self.absent_matches
        current = item[self.attr]
        return type(current) is type(self.value) and current == self.value


@dataclass(frozen=True, slots=True)
class TxPut:
    """One whole-item write in a :meth:`Store.transact_write`. ``condition``: ``None``,
    ``"not_exists"``, ``"exists"`` or an :class:`Expect`."""

    table: str
    item: Mapping[str, Any]
    condition: Condition | Expect | None = None


@dataclass(frozen=True, slots=True)
class IndexSchema:
    """A global secondary index: partition key, optional sort key, and what it projects (``ALL``
    attributes, or ``KEYS_ONLY``: the table's and the index's key attributes)."""

    pk: str
    sk: str | None = None
    projection: Literal["ALL", "KEYS_ONLY"] = "ALL"


@dataclass(frozen=True, slots=True)
class TableSchema:
    """Key schema of one logical table and its indexes."""

    pk: str
    sk: str | None = None
    indexes: Mapping[str, IndexSchema] = field(default_factory=dict)

    @property
    def key_attrs(self) -> tuple[str, ...]:
        return (self.pk,) if self.sk is None else (self.pk, self.sk)

    def keys_for(self, index: str | None) -> IndexSchema:
        if index is None:
            return IndexSchema(self.pk, self.sk)
        try:
            return self.indexes[index]
        except KeyError:
            raise StoreError(f"unknown index {index!r}") from None


#: Key schemas for every logical table.
SCHEMAS: Mapping[str, TableSchema] = {
    # One scalar `phone` per customer record; a shared household phone is two
    # records with the same `phone`, found together by the ANI lookup.
    "customers": TableSchema(pk="customer_id", indexes={"phone-index": IndexSchema("phone")}),
    "tokens": TableSchema(pk="token_hash", indexes={"call_sid-index": IndexSchema("call_sid")}),
    # ``sweep-due``: sparse, KEYS_ONLY; only escalated calls whose conversation is
    # not closed carry its keys (derived by ``owlbank.callstate.CallState.to_item``).
    "calls": TableSchema(pk="call_sid", indexes={
        "sweep-due": IndexSchema("sweep_open", "sweep_due_at", projection="KEYS_ONLY")}),
    "segments": TableSchema(pk="call_sid", sk="seq"),
    "policy": TableSchema(pk="pk"),
    "catalog": TableSchema(pk="key", sk="lang"),
    # Untrusted-ANI failure slots: ``untrusted#<customer_id>#<n>``,
    # TTL ``expires_at``. Read and written only by auth_service; no ANI, no factor values.
    "risk": TableSchema(pk="risk_key"),
}

#: Tables of the ``OwlBankAdmin`` stack. ``admin_audit``: pk = the changed item's
#: key (``policy#tool#get_balance``, ``catalog#mask.get_balance``, ``user#<username>``), sk = ISO
#: time + random suffix, TTL ``expires_at``.
ADMIN_SCHEMAS: Mapping[str, TableSchema] = {
    "admin_audit": TableSchema(pk="pk", sk="sk"),
}

#: The model-failover table, always built; only the bridge and the poller (with
#: ``modelFailover=on``) get its name. ``modelstats``: pk ``s`` (the fixed ``samples`` for samples, ``state``
#: for the verdict; each stage has its own table), sk ``sk`` (``<bucket_start>#<model_tag>`` or ``state``), TTL ``expires_at``.
FAILOVER_SCHEMAS: Mapping[str, TableSchema] = {
    "modelstats": TableSchema(pk="s", sk="sk"),
}

#: Every table either store accepts.
ALL_SCHEMAS: Mapping[str, TableSchema] = {**SCHEMAS, **ADMIN_SCHEMAS, **FAILOVER_SCHEMAS}


#: Key attributes stored as DynamoDB numbers (all other key attributes are strings).
NUMERIC_KEYS: frozenset[str] = frozenset({"seq", "sweep_due_at"})


@runtime_checkable
class Store(Protocol):
    """The storage interface every component depends on."""

    def get(self, table: str, key: Mapping[str, Any], *, consistent: bool = True) -> Item | None:
        """Return the item with exactly this key, or ``None``; ``consistent=False`` reads
        eventually consistent (half a read unit)."""
        ...

    def put(self, table: str, item: Mapping[str, Any], *, condition: Condition | None = None) -> None:
        """Write a whole item (replacing any existing one), subject to ``condition``."""
        ...

    def query(
        self, table: str, key_cond: Mapping[str, Any], *, index: str | None = None
    ) -> list[Item]:
        """Return items whose partition (and optional sort) key equal ``key_cond``."""
        ...

    def query_range(self, table: str, partition: Mapping[str, Any], *, low: str | int, high: str | int,
                    consistent: bool = False, index: str | None = None) -> list[Item]:
        """Return the items of one partition (``partition`` holds only the partition key) whose sort
        key is between ``low`` and ``high``, both included, ordered by sort key; eventually consistent
        unless ``consistent``. With ``index``, the partition and sort key are that index's (never
        consistent: a global secondary index can't be), and the items carry what it projects."""
        ...

    def batch_get(self, table: str, keys: Iterable[Mapping[str, Any]]) -> list[Item]:
        """Return the items with these keys (missing ones left out), in no particular order."""
        ...

    def scan(self, table: str) -> list[Item]:
        """Return every item of the table, in no particular order (small tables only)."""
        ...

    def update(
        self,
        table: str,
        key: Mapping[str, Any],
        updates: Mapping[str, Any],
        *,
        remove: Iterable[str] = (),
    ) -> Item:
        """Set ``updates`` and drop ``remove`` on an existing item; return the new item."""
        ...

    def increment(
        self,
        table: str,
        key: Mapping[str, Any],
        attr: str,
        *,
        by: int = 1,
        defaults: Mapping[str, Any] | None = None,
        changes: Mapping[str, Any] | None = None,
        unless_equal: tuple[str, int | str] | None = None,
    ) -> None:
        """Add ``by`` to ``attr`` (missing counts as 0), creating the item if missing; set
        ``defaults`` only where absent, and ``changes`` unless the stored ``unless_equal[0]``
        equals ``unless_equal[1]``."""
        ...

    def add_counters(self, table: str, key: Mapping[str, Any], counters: Mapping[str, int], *,
                     defaults: Mapping[str, Any] | None = None) -> None:
        """Add each of ``counters`` (missing counts as 0) in one write, creating the item if missing;
        set ``defaults`` only where absent."""
        ...

    def set_if_greater(self, table: str, key: Mapping[str, Any], attr: str, value: int, *,
                       defaults: Mapping[str, Any] | None = None,
                       changes: Mapping[str, Any] | None = None) -> bool:
        """Set ``attr`` to ``value`` and ``changes`` (and ``defaults`` only where absent), creating
        the item if missing, only if the stored ``attr`` is missing or less than ``value``. One
        conditional write; ``False`` (nothing written) when the stored value is ``value`` or more."""
        ...

    def delete(self, table: str, key: Mapping[str, Any]) -> None:
        """Delete the item if present (no error if absent)."""
        ...

    def transact_write(self, puts: Sequence[TxPut]) -> None:
        """Write every put, or none: :class:`TransactionCancelled` if any condition fails."""
        ...


def create_table_kwargs(table: str, physical_name: str) -> dict[str, Any]:
    """``CreateTable`` arguments for a logical table (used by tests and seeding tools).

    Key attribute types: :data:`NUMERIC_KEYS` (``segments.seq``, ``calls.sweep_due_at``) are numbers;
    every other key is a string.
    On-demand billing; the CDK app adds encryption, PITR and TTL (``expires_at``).
    """
    schema = _schema(table)
    attr_types: dict[str, str] = {}

    def key_schema(pk: str, sk: str | None) -> list[dict[str, str]]:
        out = [{"AttributeName": pk, "KeyType": "HASH"}]
        if sk:
            out.append({"AttributeName": sk, "KeyType": "RANGE"})
        for attr in (pk, sk):
            if attr:
                attr_types[attr] = "N" if attr in NUMERIC_KEYS else "S"
        return out

    kwargs: dict[str, Any] = {
        "TableName": physical_name,
        "KeySchema": key_schema(schema.pk, schema.sk),
        "BillingMode": "PAY_PER_REQUEST",
    }
    if schema.indexes:
        kwargs["GlobalSecondaryIndexes"] = [
            {"IndexName": name, "KeySchema": key_schema(idx.pk, idx.sk),
             "Projection": {"ProjectionType": idx.projection}}
            for name, idx in schema.indexes.items()
        ]
    kwargs["AttributeDefinitions"] = [
        {"AttributeName": a, "AttributeType": t} for a, t in attr_types.items()
    ]
    return kwargs


# --------------------------------------------------------------------------- helpers


def _schema(table: str) -> TableSchema:
    try:
        return ALL_SCHEMAS[table]
    except KeyError:
        raise StoreError(f"unknown table {table!r}") from None


def _check_transaction(puts: Sequence[TxPut]) -> list[tuple[TableSchema, tuple[Any, ...]]]:
    """Validate a transaction's puts; return each one's schema and key tuple."""
    if not puts:
        raise StoreError("a transaction needs at least one put")
    if len(puts) > MAX_TRANSACTION_ITEMS:
        raise StoreError(f"a transaction takes at most {MAX_TRANSACTION_ITEMS} puts")
    out: list[tuple[TableSchema, tuple[Any, ...]]] = []
    seen: set[tuple[str, tuple[Any, ...]]] = set()
    for put in puts:
        if not isinstance(put, TxPut):
            raise StoreError("transaction items must be TxPut")
        schema = _schema(put.table)
        _check_item(schema, put.item)
        cond = put.condition
        if cond is not None and not isinstance(cond, Expect) and cond not in ("not_exists", "exists"):
            raise StoreError(f"unknown condition {cond!r}")
        if isinstance(cond, Expect):
            if cond.attr in schema.key_attrs:
                raise StoreError("an Expect condition can't test a key attribute")
            _check_value(cond.value, cond.attr)
        k = tuple(put.item[a] for a in schema.key_attrs)
        if (put.table, k) in seen:
            raise StoreError("a transaction may write each item only once")
        seen.add((put.table, k))
        out.append((schema, k))
    return out


def _condition_expression(schema: TableSchema, cond: Condition | Expect | None
                          ) -> tuple[str | None, dict[str, str], dict[str, Any]]:
    """A DynamoDB ``ConditionExpression`` (with its names and values) for a transaction put."""
    if cond is None:
        return None, {}, {}
    names = {"#pk": schema.pk}
    if cond == "not_exists":
        return "attribute_not_exists(#pk)", names, {}
    if cond == "exists":
        return "attribute_exists(#pk)", names, {}
    if not isinstance(cond, Expect):
        raise StoreError(f"unknown condition {cond!r}")
    names = {"#a": cond.attr}  # DynamoDB refuses unused names
    expr = "(attribute_not_exists(#a) OR #a = :v)" if cond.absent_matches else "#a = :v"
    if cond.must_exist:
        names["#pk"] = schema.pk
        expr = f"attribute_exists(#pk) AND {expr}"
    return expr, names, {":v": cond.value}


def _check_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, (float, Decimal)):
        raise TypeError(f"{path}: floats/Decimals are not allowed; use int or a decimal string")
    if isinstance(value, list):
        for i, v in enumerate(value):
            _check_value(v, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"{path}: map keys must be str")
            _check_value(v, f"{path}.{k}")
        return
    raise TypeError(f"{path}: unsupported type {type(value).__name__}")


def _check_item(schema: TableSchema, item: Mapping[str, Any]) -> None:
    for attr in schema.key_attrs:
        if item.get(attr) in (None, ""):
            raise StoreError(f"item missing key attribute {attr!r}")
    index_attrs = [a for idx in schema.indexes.values() for a in (idx.pk, idx.sk) if a]
    for attr in (*schema.key_attrs, *index_attrs):
        if attr not in item:
            continue
        want: type = int if attr in NUMERIC_KEYS else str
        value = item[attr]
        if not isinstance(value, want) or isinstance(value, bool):
            raise StoreError(f"key attribute {attr!r} must be {want.__name__}")
    for k, v in item.items():
        _check_value(v, k)


def _check_key(schema: TableSchema, key: Mapping[str, Any]) -> tuple[Any, ...]:
    if set(key) != set(schema.key_attrs):
        raise StoreError(f"key must have exactly {schema.key_attrs}, got {sorted(key)}")
    return tuple(key[a] for a in schema.key_attrs)


def _unique_keys(schema: TableSchema, keys: Iterable[Mapping[str, Any]]
                 ) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Each distinct key once, in first-seen order: ``(key tuple, key dict)``."""
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key in keys:
        k = _check_key(schema, key)
        out.setdefault(k, {a: key[a] for a in schema.key_attrs})
    return list(out.items())


def _check_query(schema: TableSchema, key_cond: Mapping[str, Any], index: str | None) -> IndexSchema:
    keys = schema.keys_for(index)
    allowed = {keys.pk} | ({keys.sk} if keys.sk else set())
    if keys.pk not in key_cond or not set(key_cond) <= allowed:
        raise StoreError(f"query needs {keys.pk!r} (and optionally {keys.sk!r}), got {sorted(key_cond)}")
    return keys


def _check_range(schema: TableSchema, partition: Mapping[str, Any], low: Any, high: Any,
                 index: str | None = None, consistent: bool = False) -> IndexSchema:
    """Validate a range query on the table or one of its indexes; return that key schema."""
    keys = schema.keys_for(index)
    if keys.sk is None:
        raise StoreError("a range query needs a table or index with a sort key")
    if index is not None and consistent:
        raise StoreError("a global secondary index can't be read consistently")
    if set(partition) != {keys.pk}:
        raise StoreError(f"a range query needs exactly {keys.pk!r}, got {sorted(partition)}")
    want: type = int if keys.sk in NUMERIC_KEYS else str
    for bound in (low, high):
        if not isinstance(bound, want) or isinstance(bound, bool):
            raise StoreError(f"range bounds on {keys.sk!r} must be {want.__name__}")
    return keys


def _project(schema: TableSchema, keys: IndexSchema, item: Item) -> Item:
    """What an index read returns of ``item``: all of it, or only the key attributes (KEYS_ONLY)."""
    if keys.projection == "ALL":
        return copy.deepcopy(item)
    names = {*schema.key_attrs, keys.pk, *((keys.sk,) if keys.sk else ())}
    return {k: copy.deepcopy(v) for k, v in item.items() if k in names}


def _check_counters(schema: TableSchema, counters: Mapping[str, int], defaults: Mapping[str, Any]) -> None:
    if not counters:
        raise StoreError("add_counters: nothing to add")
    touched = [*counters, *defaults]
    if set(touched) & set(schema.key_attrs):
        raise StoreError("add_counters may not change key attributes")
    if len(touched) != len(set(touched)):
        raise StoreError("add_counters: an attribute appears twice")
    for name, by in counters.items():
        if isinstance(by, bool) or not isinstance(by, int):
            raise TypeError(f"add_counters: {name} must be an int")
    for k, v in defaults.items():
        _check_value(v, k)


def _check_updates(schema: TableSchema, updates: Mapping[str, Any], remove: Iterable[str]) -> list[str]:
    remove = list(remove)
    touched = set(updates) | set(remove)
    if touched & set(schema.key_attrs):
        raise StoreError("updates may not change key attributes")
    if set(updates) & set(remove):
        raise StoreError("an attribute can't be both set and removed")
    if not touched:
        raise StoreError("nothing to update")
    for k, v in updates.items():
        _check_value(v, k)
    return remove


def _check_increment(schema: TableSchema, attr: str, by: int, defaults: Mapping[str, Any],
                     changes: Mapping[str, Any], unless_equal: tuple[str, int | str] | None) -> None:
    if isinstance(by, bool) or not isinstance(by, int):
        raise TypeError("increment: by must be an int")
    touched = [attr, *defaults, *changes]
    if set(touched) & set(schema.key_attrs):
        raise StoreError("increment may not change key attributes")
    if len(touched) != len(set(touched)):
        raise StoreError("increment: an attribute appears twice")
    if unless_equal is not None:
        if unless_equal[0] in schema.key_attrs:
            raise StoreError("increment: unless_equal can't test a key attribute")
        _check_value(unless_equal[1], unless_equal[0])
    for k, v in {**defaults, **changes}.items():
        _check_value(v, k)


def _check_set_if_greater(schema: TableSchema, attr: str, value: int, defaults: Mapping[str, Any],
                          changes: Mapping[str, Any]) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("set_if_greater: value must be an int")
    _check_increment(schema, attr, 0, defaults, changes, None)


def _from_dynamo(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else value
    if isinstance(value, list):
        return [_from_dynamo(v) for v in value]
    if isinstance(value, dict):
        return {k: _from_dynamo(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------- MemoryStore


class MemoryStore:
    """In-memory :class:`Store` for tests. Not thread-safe; not for production."""

    def __init__(self) -> None:
        self._tables: dict[str, dict[tuple[Any, ...], Item]] = {name: {} for name in ALL_SCHEMAS}

    def get(self, table: str, key: Mapping[str, Any], *, consistent: bool = True) -> Item | None:
        k = _check_key(_schema(table), key)
        item = self._tables[table].get(k)
        return copy.deepcopy(item) if item is not None else None

    def put(self, table: str, item: Mapping[str, Any], *, condition: Condition | None = None) -> None:
        schema = _schema(table)
        _check_item(schema, item)
        k = tuple(item[a] for a in schema.key_attrs)
        exists = k in self._tables[table]
        if condition == "not_exists" and exists:
            raise ConditionFailed(f"{table}: item exists")
        if condition == "exists" and not exists:
            raise ConditionFailed(f"{table}: item does not exist")
        self._tables[table][k] = copy.deepcopy(dict(item))

    def query(
        self, table: str, key_cond: Mapping[str, Any], *, index: str | None = None
    ) -> list[Item]:
        schema = _schema(table)
        keys = _check_query(schema, key_cond, index)
        hits = [
            item
            for item in self._tables[table].values()
            if all(attr in item and item[attr] == val for attr, val in key_cond.items())
        ]
        sk = keys.sk
        if sk:
            hits = [it for it in hits if sk in it]  # an index holds only items with its keys
            hits.sort(key=lambda it: it[sk])
        return [_project(schema, keys, it) for it in hits]

    def batch_get(self, table: str, keys: Iterable[Mapping[str, Any]]) -> list[Item]:
        unique = _unique_keys(_schema(table), keys)
        rows = self._tables[table]
        return [copy.deepcopy(rows[k]) for k, _ in unique if k in rows]

    def scan(self, table: str) -> list[Item]:
        _schema(table)
        return copy.deepcopy(list(self._tables[table].values()))

    def update(
        self,
        table: str,
        key: Mapping[str, Any],
        updates: Mapping[str, Any],
        *,
        remove: Iterable[str] = (),
    ) -> Item:
        schema = _schema(table)
        k = _check_key(schema, key)
        remove = _check_updates(schema, updates, remove)
        current = self._tables[table].get(k)
        if current is None:
            raise ConditionFailed(f"{table}: item does not exist")
        current.update(copy.deepcopy(dict(updates)))
        for attr in remove:
            current.pop(attr, None)
        return copy.deepcopy(current)

    def increment(
        self,
        table: str,
        key: Mapping[str, Any],
        attr: str,
        *,
        by: int = 1,
        defaults: Mapping[str, Any] | None = None,
        changes: Mapping[str, Any] | None = None,
        unless_equal: tuple[str, int | str] | None = None,
    ) -> None:
        schema = _schema(table)
        k = _check_key(schema, key)
        defaults, changes = dict(defaults or {}), dict(changes or {})
        _check_increment(schema, attr, by, defaults, changes, unless_equal)
        current = self._tables[table].setdefault(k, {a: key[a] for a in schema.key_attrs})
        old = current.get(attr, 0)
        if isinstance(old, bool) or not isinstance(old, int):
            raise StoreError(f"{table}: {attr} is not a number")
        skip = unless_equal is not None and unless_equal[0] in current and \
            type(current[unless_equal[0]]) is type(unless_equal[1]) and \
            current[unless_equal[0]] == unless_equal[1]
        current[attr] = old + by
        for name, value in defaults.items():
            current.setdefault(name, copy.deepcopy(value))
        if not skip:
            current.update(copy.deepcopy(changes))

    def query_range(self, table: str, partition: Mapping[str, Any], *, low: str | int, high: str | int,
                    consistent: bool = False, index: str | None = None) -> list[Item]:
        schema = _schema(table)
        keys = _check_range(schema, partition, low, high, index, consistent)
        sk = keys.sk or ""  # _check_range: a range query has a sort key
        pk_value = partition[keys.pk]
        # An index holds only the items that carry both its keys (sparse).
        hits = [item for item in self._tables[table].values()
                if keys.pk in item and sk in item and item[keys.pk] == pk_value and low <= item[sk] <= high]
        hits.sort(key=lambda it: it[sk])
        return [_project(schema, keys, i) for i in hits]

    def add_counters(self, table: str, key: Mapping[str, Any], counters: Mapping[str, int], *,
                     defaults: Mapping[str, Any] | None = None) -> None:
        schema = _schema(table)
        k = _check_key(schema, key)
        defaults = dict(defaults or {})
        _check_counters(schema, counters, defaults)
        current = self._tables[table].get(k, {a: key[a] for a in schema.key_attrs})
        for name in counters:
            old = current.get(name, 0)
            if isinstance(old, bool) or not isinstance(old, int):
                raise StoreError(f"{table}: {name} is not a number")
        for name, by in counters.items():
            current[name] = current.get(name, 0) + by
        for name, value in defaults.items():
            current.setdefault(name, copy.deepcopy(value))
        self._tables[table][k] = current

    def set_if_greater(self, table: str, key: Mapping[str, Any], attr: str, value: int, *,
                       defaults: Mapping[str, Any] | None = None,
                       changes: Mapping[str, Any] | None = None) -> bool:
        schema = _schema(table)
        k = _check_key(schema, key)
        defaults, changes = dict(defaults or {}), dict(changes or {})
        _check_set_if_greater(schema, attr, value, defaults, changes)
        current = self._tables[table].get(k)
        if current is not None and attr in current:
            old = current[attr]
            if isinstance(old, bool) or not isinstance(old, int):
                raise StoreError(f"{table}: {attr} is not a number")
            if old >= value:
                return False
        current = self._tables[table].setdefault(k, {a: key[a] for a in schema.key_attrs})
        current[attr] = value
        for name, v in defaults.items():
            current.setdefault(name, copy.deepcopy(v))
        current.update(copy.deepcopy(changes))
        return True

    def delete(self, table: str, key: Mapping[str, Any]) -> None:
        k = _check_key(_schema(table), key)
        self._tables[table].pop(k, None)

    def transact_write(self, puts: Sequence[TxPut]) -> None:
        checked = _check_transaction(puts)
        failed: list[int] = []
        for i, (put, (_s, k)) in enumerate(zip(puts, checked, strict=True)):
            current = self._tables[put.table].get(k)
            cond = put.condition
            ok = (cond is None
                  or (cond == "not_exists" and current is None)
                  or (cond == "exists" and current is not None)
                  or (isinstance(cond, Expect) and cond.holds(current)))
            if not ok:
                failed.append(i)
        if failed:
            raise TransactionCancelled("transaction cancelled: condition failed", failed)
        for put, (_s, k) in zip(puts, checked, strict=True):
            self._tables[put.table][k] = copy.deepcopy(dict(put.item))


# --------------------------------------------------------------------------- DynamoStore


class DynamoStore:
    """DynamoDB :class:`Store`.

    ``table_names`` maps logical to physical names (``Settings.tables``). The boto3
    resource is created lazily on first use, so constructing a store makes no AWS call.
    """

    def __init__(self, table_names: Mapping[str, str], *, resource: Any | None = None,
                 region_name: str | None = None, required: Iterable[str] | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        """``required``: the logical tables that must have a physical name (default: every
        Data-stack table in :data:`SCHEMAS`). A component that uses a few tables (the admin
        API) names them; using a table with no physical name then raises :class:`StoreError`.
        ``sleep``: the wait before re-sending ``UnprocessedKeys`` (tests pass a recorder)."""
        required = set(SCHEMAS) if required is None else set(required)
        unknown = (required | set(table_names)) - set(ALL_SCHEMAS)
        if unknown:
            raise StoreError(f"unknown tables {sorted(unknown)}")
        missing = required - set(table_names)
        if missing:
            raise StoreError(f"no physical name for tables {sorted(missing)}")
        self._names = dict(table_names)
        self._resource = resource
        self._region = region_name
        self._handles: dict[str, Any] = {}
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Any, *, resource: Any | None = None) -> DynamoStore:
        """Build from :class:`owlbank.config.Settings`."""
        return cls(settings.tables, resource=resource, region_name=settings.aws_region)

    def _table(self, table: str) -> Any:
        _schema(table)
        if table not in self._names:
            raise StoreError(f"no physical name for table {table!r}")
        if table not in self._handles:
            self._handles[table] = self._dynamodb().Table(self._names[table])
        return self._handles[table]

    def _dynamodb(self) -> Any:
        if self._resource is None:
            import boto3

            self._resource = boto3.resource("dynamodb", region_name=self._region)
        return self._resource

    @staticmethod
    def _client_error_code(exc: Exception) -> str | None:
        response = getattr(exc, "response", None)
        return response.get("Error", {}).get("Code") if isinstance(response, dict) else None

    def get(self, table: str, key: Mapping[str, Any], *, consistent: bool = True) -> Item | None:
        _check_key(_schema(table), key)
        with timed_call("dynamodb", f"get_item:{table}", ok_level=logging.DEBUG):
            resp = self._table(table).get_item(Key=dict(key), ConsistentRead=consistent)
        item = resp.get("Item")
        return _from_dynamo(item) if item is not None else None

    def put(self, table: str, item: Mapping[str, Any], *, condition: Condition | None = None) -> None:
        schema = _schema(table)
        _check_item(schema, item)
        kwargs: dict[str, Any] = {"Item": dict(item)}
        if condition is not None:
            fn = "attribute_not_exists" if condition == "not_exists" else "attribute_exists"
            kwargs["ConditionExpression"] = f"{fn}(#pk)"
            kwargs["ExpressionAttributeNames"] = {"#pk": schema.pk}
        try:
            with timed_call("dynamodb", f"put_item:{table}", ok_level=logging.DEBUG,
                            expected_codes=_CONDITION_CODES):
                self._table(table).put_item(**kwargs)
        except Exception as exc:
            if self._client_error_code(exc) == "ConditionalCheckFailedException":
                raise ConditionFailed(f"{table}: condition {condition} failed") from exc
            raise

    def query(
        self, table: str, key_cond: Mapping[str, Any], *, index: str | None = None
    ) -> list[Item]:
        from boto3.dynamodb.conditions import Key

        schema = _schema(table)
        keys = _check_query(schema, key_cond, index)
        expr = Key(keys.pk).eq(key_cond[keys.pk])
        if keys.sk and keys.sk in key_cond:
            expr = expr & Key(keys.sk).eq(key_cond[keys.sk])
        kwargs: dict[str, Any] = {"KeyConditionExpression": expr}
        if index is not None:
            kwargs["IndexName"] = index
        else:
            kwargs["ConsistentRead"] = True
        items: list[Item] = []
        with timed_call("dynamodb", f"query:{table}", ok_level=logging.DEBUG):
            while True:
                resp = self._table(table).query(**kwargs)
                items.extend(_from_dynamo(i) for i in resp.get("Items", []))
                if "LastEvaluatedKey" not in resp:
                    return items
                kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    def batch_get(self, table: str, keys: Iterable[Mapping[str, Any]]) -> list[Item]:
        unique = [key for _, key in _unique_keys(_schema(table), keys)]
        if not unique:
            return []
        self._table(table)  # checks the physical name
        physical = self._names[table]
        items: list[Item] = []
        with timed_call("dynamodb", f"batch_get_item:{table}", ok_level=logging.DEBUG):
            for start in range(0, len(unique), MAX_BATCH_GET_KEYS):
                request: dict[str, Any] = {physical: {"Keys": unique[start:start + MAX_BATCH_GET_KEYS],
                                                      "ConsistentRead": True}}
                retries = iter(BATCH_GET_RETRY_DELAYS_S)
                while True:
                    # The resource's batch_get_item serializes keys and deserializes items.
                    resp = self._dynamodb().batch_get_item(RequestItems=request)
                    items.extend(_from_dynamo(i) for i in resp.get("Responses", {}).get(physical, []))
                    request = resp.get("UnprocessedKeys") or {}
                    if not request:
                        break
                    bound = next(retries, None)
                    if bound is None:
                        raise StoreError(f"{table}: keys still unprocessed after "
                                         f"{len(BATCH_GET_RETRY_DELAYS_S)} retries")
                    self._sleep(random.uniform(bound / 2, bound))  # noqa: S311 (jitter, not security)
        return items

    def scan(self, table: str) -> list[Item]:
        _schema(table)
        kwargs: dict[str, Any] = {"ConsistentRead": True}
        items: list[Item] = []
        with timed_call("dynamodb", f"scan:{table}", ok_level=logging.DEBUG):
            while True:
                resp = self._table(table).scan(**kwargs)
                items.extend(_from_dynamo(i) for i in resp.get("Items", []))
                if "LastEvaluatedKey" not in resp:
                    return items
                kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    def update(
        self,
        table: str,
        key: Mapping[str, Any],
        updates: Mapping[str, Any],
        *,
        remove: Iterable[str] = (),
    ) -> Item:
        schema = _schema(table)
        _check_key(schema, key)
        remove = _check_updates(schema, updates, remove)
        names: dict[str, str] = {"#pk": schema.pk}
        values: dict[str, Any] = {}
        clauses: list[str] = []
        sets = []
        for i, (attr, val) in enumerate(updates.items()):
            names[f"#s{i}"] = attr
            values[f":s{i}"] = val
            sets.append(f"#s{i} = :s{i}")
        if sets:
            clauses.append("SET " + ", ".join(sets))
        if remove:
            for i, attr in enumerate(remove):
                names[f"#r{i}"] = attr
            clauses.append("REMOVE " + ", ".join(f"#r{i}" for i in range(len(remove))))
        kwargs: dict[str, Any] = {
            "Key": dict(key),
            "UpdateExpression": " ".join(clauses),
            "ConditionExpression": "attribute_exists(#pk)",
            "ExpressionAttributeNames": names,
            "ReturnValues": "ALL_NEW",
        }
        if values:
            kwargs["ExpressionAttributeValues"] = values
        try:
            with timed_call("dynamodb", f"update_item:{table}", ok_level=logging.DEBUG,
                            expected_codes=_CONDITION_CODES):
                resp = self._table(table).update_item(**kwargs)
        except Exception as exc:
            if self._client_error_code(exc) == "ConditionalCheckFailedException":
                raise ConditionFailed(f"{table}: item does not exist") from exc
            raise
        return _from_dynamo(resp["Attributes"])

    def increment(
        self,
        table: str,
        key: Mapping[str, Any],
        attr: str,
        *,
        by: int = 1,
        defaults: Mapping[str, Any] | None = None,
        changes: Mapping[str, Any] | None = None,
        unless_equal: tuple[str, int | str] | None = None,
    ) -> None:
        schema = _schema(table)
        _check_key(schema, key)
        defaults, changes = dict(defaults or {}), dict(changes or {})
        _check_increment(schema, attr, by, defaults, changes, unless_equal)
        conditional = bool(changes) and unless_equal is not None
        try:
            self._increment(table, key, attr, by, defaults, changes,
                            unless_equal if conditional else None)
        except ConditionFailed:
            if not conditional:  # pragma: no cover - an unconditional write can't fail its condition
                raise
            # The stored value already equals unless_equal: only the counter and the defaults.
            self._increment(table, key, attr, by, defaults, {}, None)

    def _increment(self, table: str, key: Mapping[str, Any], attr: str, by: int,
                   defaults: Mapping[str, Any], changes: Mapping[str, Any],
                   unless_equal: tuple[str, int | str] | None) -> None:
        names: dict[str, str] = {"#c": attr}
        values: dict[str, Any] = {":by": by}
        sets: list[str] = []
        for i, (name, value) in enumerate(defaults.items()):
            names[f"#d{i}"], values[f":d{i}"] = name, value
            sets.append(f"#d{i} = if_not_exists(#d{i}, :d{i})")
        for i, (name, value) in enumerate(changes.items()):
            names[f"#s{i}"], values[f":s{i}"] = name, value
            sets.append(f"#s{i} = :s{i}")
        expr = "ADD #c :by" + (" SET " + ", ".join(sets) if sets else "")
        kwargs: dict[str, Any] = {"Key": dict(key), "UpdateExpression": expr}
        if unless_equal is not None:
            names["#w"], values[":w"] = unless_equal
            kwargs["ConditionExpression"] = "attribute_not_exists(#w) OR #w <> :w"
        kwargs["ExpressionAttributeNames"] = names
        kwargs["ExpressionAttributeValues"] = values
        try:
            with timed_call("dynamodb", f"update_item:{table}", ok_level=logging.DEBUG,
                            expected_codes=_CONDITION_CODES):
                self._table(table).update_item(**kwargs)
        except Exception as exc:
            if self._client_error_code(exc) == "ConditionalCheckFailedException":
                raise ConditionFailed(f"{table}: {unless_equal[0] if unless_equal else attr} "
                                      "unchanged") from exc
            raise

    def query_range(self, table: str, partition: Mapping[str, Any], *, low: str | int, high: str | int,
                    consistent: bool = False, index: str | None = None) -> list[Item]:
        from boto3.dynamodb.conditions import Key

        schema = _schema(table)
        keys = _check_range(schema, partition, low, high, index, consistent)
        kwargs: dict[str, Any] = {
            "KeyConditionExpression": Key(keys.pk).eq(partition[keys.pk]) & Key(keys.sk).between(low, high),
            "ConsistentRead": consistent,
        }
        if index is not None:
            kwargs["IndexName"] = index
        items: list[Item] = []
        with timed_call("dynamodb", f"query:{table}", ok_level=logging.DEBUG):
            while True:
                resp = self._table(table).query(**kwargs)
                items.extend(_from_dynamo(i) for i in resp.get("Items", []))
                if "LastEvaluatedKey" not in resp:
                    return items
                kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    def add_counters(self, table: str, key: Mapping[str, Any], counters: Mapping[str, int], *,
                     defaults: Mapping[str, Any] | None = None) -> None:
        schema = _schema(table)
        _check_key(schema, key)
        defaults = dict(defaults or {})
        _check_counters(schema, counters, defaults)
        names: dict[str, str] = {}
        values: dict[str, Any] = {}
        adds, sets = [], []
        for i, (name, by) in enumerate(counters.items()):
            names[f"#c{i}"], values[f":c{i}"] = name, by
            adds.append(f"#c{i} :c{i}")
        for i, (name, value) in enumerate(defaults.items()):
            names[f"#d{i}"], values[f":d{i}"] = name, value
            sets.append(f"#d{i} = if_not_exists(#d{i}, :d{i})")
        expr = "ADD " + ", ".join(adds) + (" SET " + ", ".join(sets) if sets else "")
        with timed_call("dynamodb", f"update_item:{table}", ok_level=logging.DEBUG):
            self._table(table).update_item(Key=dict(key), UpdateExpression=expr,
                                           ExpressionAttributeNames=names,
                                           ExpressionAttributeValues=values)

    def set_if_greater(self, table: str, key: Mapping[str, Any], attr: str, value: int, *,
                       defaults: Mapping[str, Any] | None = None,
                       changes: Mapping[str, Any] | None = None) -> bool:
        schema = _schema(table)
        _check_key(schema, key)
        defaults, changes = dict(defaults or {}), dict(changes or {})
        _check_set_if_greater(schema, attr, value, defaults, changes)
        names: dict[str, str] = {"#a": attr}
        values: dict[str, Any] = {":v": value}
        sets = ["#a = :v"]
        for i, (name, v) in enumerate(defaults.items()):
            names[f"#d{i}"], values[f":d{i}"] = name, v
            sets.append(f"#d{i} = if_not_exists(#d{i}, :d{i})")
        for i, (name, v) in enumerate(changes.items()):
            names[f"#s{i}"], values[f":s{i}"] = name, v
            sets.append(f"#s{i} = :s{i}")
        try:
            with timed_call("dynamodb", f"update_item:{table}", ok_level=logging.DEBUG,
                            expected_codes=_CONDITION_CODES):
                self._table(table).update_item(
                    Key=dict(key), UpdateExpression="SET " + ", ".join(sets),
                    ConditionExpression="attribute_not_exists(#a) OR #a < :v",
                    ExpressionAttributeNames=names, ExpressionAttributeValues=values)
        except Exception as exc:
            if self._client_error_code(exc) == "ConditionalCheckFailedException":
                return False
            raise
        return True

    def delete(self, table: str, key: Mapping[str, Any]) -> None:
        _check_key(_schema(table), key)
        with timed_call("dynamodb", f"delete_item:{table}", ok_level=logging.DEBUG):
            self._table(table).delete_item(Key=dict(key))

    def transact_write(self, puts: Sequence[TxPut]) -> None:
        checked = _check_transaction(puts)
        items: list[dict[str, Any]] = []
        for put, (schema, _k) in zip(puts, checked, strict=True):
            if put.table not in self._names:
                raise StoreError(f"no physical name for table {put.table!r}")
            # The resource's client serializes Python values itself (as Table methods do).
            spec: dict[str, Any] = {"TableName": self._names[put.table], "Item": dict(put.item)}
            expr, names, values = _condition_expression(schema, put.condition)
            if expr is not None:
                spec["ConditionExpression"] = expr
                spec["ExpressionAttributeNames"] = names
                if values:
                    spec["ExpressionAttributeValues"] = values
            items.append({"Put": spec})
        try:
            with timed_call("dynamodb", "transact_write_items", ok_level=logging.DEBUG,
                            expected_codes=("TransactionCanceledException",)):
                self._dynamodb().meta.client.transact_write_items(TransactItems=items)
        except Exception as exc:
            if self._client_error_code(exc) != "TransactionCanceledException":
                raise
            response = getattr(exc, "response", None)
            reasons = (response.get("CancellationReasons") if isinstance(response, dict) else None) or []
            codes = [r.get("Code") if isinstance(r, Mapping) else None for r in reasons]
            # Anything but a failed condition or a conflicting transaction is a real error.
            if any(c not in (None, "None", "ConditionalCheckFailed", "TransactionConflict") for c in codes):
                raise
            failed = [i for i, c in enumerate(codes) if c not in (None, "None")]
            raise TransactionCancelled("transaction cancelled", failed) from exc


# --------------------------------------------------------------------------- reload logging


class timed_reload(timed_call):  # noqa: N801  (used like timed_call)
    """One ``dependency_call`` for a whole cache reload: ``dependency`` ``dynamodb``,
    ``operation`` ``reload:<name>``, ``duration_ms`` and ``items`` (set :attr:`items` to the
    number of items read). A success logs at INFO, or WARNING when slower than
    :data:`owlbank.obs.SLOW_CALL_MS` (``outcome`` ``slow``); a failure as :class:`timed_call`."""

    def __init__(self, name: str) -> None:
        super().__init__("dynamodb", f"reload:{name}")
        self.items = 0

    def __enter__(self) -> timed_reload:
        super().__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> Literal[False]:
        if exc is not None:
            return super().__exit__(exc_type, exc, tb)
        duration = round((time.perf_counter() - self._started) * 1000, 1)
        slow = duration > SLOW_CALL_MS
        emit(_reload_log, logging.WARNING if slow else self.ok_level, "dependency_call",
             outcome="slow" if slow else "ok", dependency=self.dependency,
             operation=self.operation, duration_ms=duration, items=self.items)
        return False


_reload_log = logging.getLogger("owlbank.dependency")
