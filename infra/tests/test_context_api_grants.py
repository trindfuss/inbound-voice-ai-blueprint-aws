"""The ``context_api`` role reads the
``policy`` table (its ``redaction`` item) and may call Comprehend's ``DetectPiiEntities``. It gets
no new KMS grant: the policy table is encrypted with the same data CMK as ``calls`` and
``segments``, already covered by the one ``kms:ViaService`` = DynamoDB statement, and it never
touches the token-sealing key (prompts never carry a sensitive value, 9.3).

The role also reads the ``catalog`` table (``escalation.summary_withheld``
for a withheld prompt), read only, on the same data CMK, so still no new KMS statement.
The CDK context key ``classifierModelId`` is now ``resolverModelId``, with no alias.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import _synth
from test_synth_and_nag import _nag_report

from owlbank_infra.config import ConfigError, InfraConfig, from_context

COMPUTE, DATA = "OwlBankCompute", "OwlBankData"
BASE = Path(__file__).resolve().parents[1]


def _lid(synth, stack: str, prefix: str, rtype: str) -> str:
    found = [lid for s, lid, _ in synth.resources(rtype) if s == stack and lid.startswith(prefix)]
    assert len(found) == 1, (prefix, found)
    return found[0]


def _stmts(synth) -> list[dict]:
    role = _lid(synth, COMPUTE, "ContextApiRole", "AWS::IAM::Role")
    out: list[dict] = []
    for s, _, res in synth.resources("AWS::IAM::Policy"):
        if s == COMPUTE and {"Ref": role} in res["Properties"].get("Roles", []):
            out += res["Properties"]["PolicyDocument"]["Statement"]
    return out


def _actions(st) -> list[str]:
    return st["Action"] if isinstance(st["Action"], list) else [st["Action"]]


def _import_names(value) -> list[str]:
    """``Fn::ImportValue`` export names inside a CFN value (Data-stack references)."""
    text = json.dumps(value)
    return [part for part in text.split('"') if part.startswith(DATA)]


@pytest.fixture(params=["synth_default", "synth_prod"])
def synth(request):
    return request.getfixturevalue(request.param)


def test_reads_the_policy_calls_segments_and_catalog_tables(synth) -> None:
    read = [st for st in _stmts(synth) if st.get("Sid") == "TablesRead"]
    assert len(read) == 1
    assert set(_actions(read[0])) == {"dynamodb:GetItem", "dynamodb:BatchGetItem", "dynamodb:Query",
                                      "dynamodb:DescribeTable"}
    blob = json.dumps(read[0]["Resource"])
    for table in ("PolicyTable", "CallsTable", "SegmentsTable", "CatalogTable"):
        assert table in blob, table
    assert "CustomersTable" not in blob and "TokensTable" not in blob and "RiskTable" not in blob
    assert not [st for st in _stmts(synth) if st.get("Sid") == "TablesReadWrite"]


def test_never_writes(synth) -> None:
    actions = {a for st in _stmts(synth) for a in _actions(st)}
    assert not {a for a in actions if a.startswith("dynamodb:")
                and any(w in a for w in ("Put", "Update", "Delete", "Write"))}


def test_detects_pii(synth) -> None:
    pii = [st for st in _stmts(synth) if st.get("Sid") == "DetectPiiForRedaction"]
    assert len(pii) == 1
    assert pii[0]["Action"] == "comprehend:DetectPiiEntities" and pii[0]["Resource"] == "*"


def test_no_new_kms_grant(synth) -> None:
    """Tables: the one DynamoDB statement. The human-leg conversation adds the secret's decrypt, only through
    Secrets Manager (the shared ``grant_secret_decrypt``); still never the token key."""
    kms = [st for st in _stmts(synth) if any(a.startswith("kms:") for a in _actions(st))]
    assert sorted(st["Sid"] for st in kms) == ["DecryptSecretsViaSecretsManager",
                                               "TablesKmsViaDynamoDB"], kms
    sm = next(st for st in kms if st["Sid"] == "DecryptSecretsViaSecretsManager")
    assert _actions(sm) == ["kms:Decrypt"] and "secretsmanager." in json.dumps(sm["Condition"])
    assert "TokenKey" not in json.dumps(sm["Resource"])
    st = next(st for st in kms if st["Sid"] == "TablesKmsViaDynamoDB")
    assert sorted(_actions(st)) == ["kms:Decrypt", "kms:DescribeKey"]
    assert "kms:ViaService" in json.dumps(st["Condition"])
    assert "dynamodb." in json.dumps(st["Condition"])
    blob = json.dumps(st["Resource"])
    assert "TokenKey" not in blob
    assert all("TokenKey" not in n for n in _import_names(st["Resource"]))


def test_policy_and_catalog_tables_use_the_same_cmk_as_calls_and_segments(synth) -> None:
    tables = synth.templates[DATA]["Resources"]
    keys = {name: json.dumps(tables[_lid(synth, DATA, f"{name}Table", "AWS::DynamoDB::Table")]
                             ["Properties"]["SSESpecification"]["KMSMasterKeyId"])
            for name in ("Policy", "Calls", "Segments", "Catalog")}
    assert len(set(keys.values())) == 1, keys


def test_one_secret_and_no_other_service(synth) -> None:
    """Human-leg: Secrets Manager for the one Twilio key secret (pinned in
    tests/test_human_leg_infra.py); no other service."""
    services = {a.split(":", 1)[0] for st in _stmts(synth) for a in _actions(st)}
    assert services <= {"dynamodb", "kms", "comprehend", "logs", "xray", "secretsmanager"}, services


def test_env_names_the_policy_table(synth) -> None:
    res = synth.templates[COMPUTE]["Resources"]
    env = res[_lid(synth, COMPUTE, "ContextApiFn", "AWS::Lambda::Function")]["Properties"][
        "Environment"]["Variables"]
    assert "OWLBANK_TABLE_POLICY" in env
    assert "OWLBANK_TABLE_CATALOG" in env


def test_cdk_nag_clean(synth) -> None:
    report = _nag_report(synth)
    if report is not None:
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [(v["ruleName"], v.get("violatingResources")) for v in errors]
        assert report["conclusion"] == "success"


# ------------------------------------------------------------------- resolverModelId

OTHER_MODEL = "us.anthropic.claude-sonnet-4-5"


def _fd_env(synth) -> dict:
    res = synth.templates[COMPUTE]["Resources"]
    return res[_lid(synth, COMPUTE, "FrontdoorFn", "AWS::Lambda::Function")]["Properties"][
        "Environment"]["Variables"]


def test_config_reads_resolver_model_id_and_ignores_the_old_key() -> None:
    assert from_context({"resolverModelId": OTHER_MODEL}, base_dir=BASE).resolver_model_id == OTHER_MODEL
    old = from_context({"classifierModelId": OTHER_MODEL}, base_dir=BASE)
    assert old.resolver_model_id == InfraConfig.resolver_model_id
    assert not hasattr(old, "classifier_model_id")


def test_classifier_model_id_fails_synth(tmp_path_factory) -> None:
    """The old key is unknown context, so synth refuses it instead of
    ignoring it."""
    with pytest.raises(ConfigError, match="'classifierModelId'"):
        _synth(tmp_path_factory, "old-classifier-key", {"classifierModelId": OTHER_MODEL})


def test_resolver_model_id_reaches_the_front_door(tmp_path_factory) -> None:
    synth = _synth(tmp_path_factory, "resolver-key", {"resolverModelId": OTHER_MODEL})
    assert _fd_env(synth)["OWLBANK_RESOLVER_MODEL_ID"] == OTHER_MODEL
    assert OTHER_MODEL in json.dumps(synth.templates[COMPUTE])
