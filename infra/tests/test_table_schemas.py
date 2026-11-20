"""The CDK tables must match owlbank.store exactly (SCHEMAS / create_table_kwargs)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(scope="module")
def store():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    return importlib.import_module("owlbank.store")


def _norm_keys(ks):
    return sorted((k["AttributeName"], k["KeyType"]) for k in ks)


def _cdk_tables(synth):
    """{logical name: CloudFormation table properties}, by the construct id prefix."""
    out = {}
    for _, lid, res in synth.resources("AWS::DynamoDB::Table"):
        logical = lid.split("Table")[0].lower()
        out[logical] = res["Properties"]
    return out


def test_infra_copy_equals_store_schemas(store):
    from owlbank_infra.contract import NUMERIC_KEYS, TABLES
    assert set(TABLES) == set(store.SCHEMAS)
    assert NUMERIC_KEYS == store.NUMERIC_KEYS
    for name, schema in store.SCHEMAS.items():
        td = TABLES[name]
        assert (td.pk, td.sk) == (schema.pk, schema.sk), name
        assert {i: (pk, sk, td.projection(i)) for i, pk, sk in td.indexes} == \
            {i: (idx.pk, idx.sk, idx.projection) for i, idx in schema.indexes.items()}, name
        assert set(td.keys_only) | set(td.own_grant) <= {i for i, _, _ in td.indexes}, name


def test_sweep_due_index_names_agree(store):
    """Human-leg: one name for the index in the app, the infra copy and the store schema."""
    from owlbank_infra.contract import SWEEP_DUE_INDEX
    callstate = importlib.import_module("owlbank.callstate")
    assert SWEEP_DUE_INDEX == callstate.SWEEP_INDEX
    idx = store.SCHEMAS["calls"].indexes[SWEEP_DUE_INDEX]
    assert (idx.pk, idx.sk, idx.projection) == (callstate.SWEEP_OPEN_ATTR, callstate.SWEEP_DUE_ATTR, "KEYS_ONLY")
    assert callstate.SWEEP_DUE_ATTR in store.NUMERIC_KEYS


def test_infra_copy_equals_failover_schemas(store):
    """The model-failover table: its own entry, never among the Data-stack tables."""
    from owlbank_infra.contract import FAILOVER_TABLES, TABLES
    assert set(FAILOVER_TABLES) == set(store.FAILOVER_SCHEMAS) == {"modelstats"}
    assert not set(FAILOVER_TABLES) & set(TABLES)
    for name, schema in store.FAILOVER_SCHEMAS.items():
        td = FAILOVER_TABLES[name]
        assert (td.pk, td.sk) == (schema.pk, schema.sk), name
        assert td.indexes == () and not schema.indexes, name
        assert td.ttl, name


def test_synthesized_tables_match_create_table_kwargs(synth_default, store):
    cdk = _cdk_tables(synth_default)
    # Model failover is on with the default list: modelstats is built too.
    assert synth_default.cfg.failover_on
    assert set(cdk) == set(store.SCHEMAS) | set(store.FAILOVER_SCHEMAS)
    for name in (*store.SCHEMAS, *store.FAILOVER_SCHEMAS):
        want = store.create_table_kwargs(name, "physical")
        got = cdk[name]
        assert _norm_keys(got["KeySchema"]) == _norm_keys(want["KeySchema"]), name
        assert sorted((a["AttributeName"], a["AttributeType"]) for a in got["AttributeDefinitions"]) == \
            sorted((a["AttributeName"], a["AttributeType"]) for a in want["AttributeDefinitions"]), name
        got_gsi = {g["IndexName"]: (_norm_keys(g["KeySchema"]), g["Projection"]["ProjectionType"])
                   for g in got.get("GlobalSecondaryIndexes", [])}
        want_gsi = {g["IndexName"]: (_norm_keys(g["KeySchema"]), g["Projection"]["ProjectionType"])
                    for g in want.get("GlobalSecondaryIndexes", [])}
        assert got_gsi == want_gsi, name
        assert got["BillingMode"] == want["BillingMode"], name


def test_lambda_and_bridge_env_name_every_table(synth_default, store):
    """DynamoStore needs every logical table's physical name. The model-failover poller and the
    human-leg conversation sweep are the exceptions: they use one table each (``modelstats``,
    ``calls``; ``DynamoStore(..., required=(<that one>,))``) and are given only that name
    (test_model_failover_infra, test_human_leg_infra)."""
    want = {f"OWLBANK_TABLE_{n.upper()}" for n in store.SCHEMAS}
    for _, lid, res in synth_default.resources("AWS::Lambda::Function"):
        if res["Properties"]["Handler"].startswith(("lambdas.model_poller.", "lambdas.conversation_sweep.")):
            continue
        env = res["Properties"].get("Environment", {}).get("Variables", {})
        assert want <= set(env), lid
    td = next(r for _, _, r in synth_default.resources("AWS::ECS::TaskDefinition"))
    names = {e["Name"] for e in td["Properties"]["ContainerDefinitions"][0]["Environment"]}
    assert want <= names
