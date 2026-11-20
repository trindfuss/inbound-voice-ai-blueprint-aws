"""Model-assisted slots: the cache reloads in a few calls.

The prompt catalog reloads with one paginated ``Scan`` of the catalog table (it has no index to
Query), and the policy with one ``BatchGetItem`` of the policy table. So:

- ``dynamodb:Scan`` appears in exactly one statement per role, ``CatalogScan``, with that one
  action on exactly the catalog table ARN (no index, no other table, never in ``TablesRead``);
- the roles holding it are exactly the ones that load the catalog with
  ``PromptCatalog.from_store`` / ``catalog._doc_from_store``: the front door, ``context_api``, the
  bridge (the Fargate task role, or the AgentCore Runtime's execution role, both ``TaskRole``) and
  the admin API;
- every role that loads the policy (front door, ``auth_service``, ``bank_tools``, the bridge, the
  admin API) may ``BatchGetItem`` the policy table;
- no new KMS statement: the catalog table is encrypted with the same data CMK the role's existing
  ``kms:ViaService`` = DynamoDB statement names.

Every context of the AgentCore hosting matrix (both hostings), plus the admin console synth.
"""

from __future__ import annotations

import json

import pytest
from conftest import _synth

DATA, COMPUTE, BRIDGE, ADMIN = "OwlBankData", "OwlBankCompute", "OwlBankBridge", "OwlBankAdmin"
#: (stack, role logical-id prefix) of every role that loads the catalog.
CATALOG_LOADERS = {(COMPUTE, "FrontdoorRole"), (COMPUTE, "ContextApiRole"), (BRIDGE, "TaskRole")}
#: ... and of every role that loads the policy (``Policy.from_store`` or ``policy._doc_from_store``).
POLICY_LOADERS = {(COMPUTE, "FrontdoorRole"), (COMPUTE, "AuthServiceRole"),
                  (COMPUTE, "BankToolsRole"), (BRIDGE, "TaskRole")}
ADMIN_ROLE = (ADMIN, "ApiFnRole")
#: The recap writer loads both, when it exists (``memoryStoreId`` in twilioOutputs).
RECAP_WRITER_ROLE = (COMPUTE, "RecapWriterRole")


def _loaders(synth, base: set[tuple[str, str]]) -> set[tuple[str, str]]:
    return base | ({RECAP_WRITER_ROLE} if synth.cfg.recap_writer_on else set())


def _actions(st: dict) -> list[str]:
    return st["Action"] if isinstance(st["Action"], list) else [st["Action"]]


def _resources(st: dict) -> list:
    return st["Resource"] if isinstance(st["Resource"], list) else [st["Resource"]]


def _role_statements(synth) -> dict[tuple[str, str], list[dict]]:
    """(stack, role logical id) -> every statement attached to it (policies and inline)."""
    out: dict[tuple[str, str], list[dict]] = {}
    for stack, lid, res in synth.resources("AWS::IAM::Role"):
        out.setdefault((stack, lid), [])
        for pol in res["Properties"].get("Policies", []):
            out[(stack, lid)] += pol["PolicyDocument"]["Statement"]
    for stack, _, res in synth.resources("AWS::IAM::Policy"):
        for ref in res["Properties"].get("Roles", []):
            if isinstance(ref, dict) and "Ref" in ref:
                out.setdefault((stack, ref["Ref"]), []).extend(res["Properties"]["PolicyDocument"]["Statement"])
    return out


def _lid(synth, stack: str, prefix: str, rtype: str) -> str:
    found = [lid for s, lid, _ in synth.resources(rtype) if s == stack and lid.startswith(prefix)]
    assert len(found) == 1, (stack, prefix, found)
    return found[0]


def _export_of(synth, table_prefix: str) -> dict:
    """The ``Fn::ImportValue`` of a Data-stack table's ARN, as the other stacks name it."""
    lid = _lid(synth, DATA, table_prefix, "AWS::DynamoDB::Table")
    names = {json.dumps(r) for stmts in _role_statements(synth).values() for st in stmts
             for r in _resources(st) if isinstance(r, dict) and "Fn::ImportValue" in r
             and f"FnGetAtt{lid}Arn" in r["Fn::ImportValue"]}
    assert len(names) == 1, (table_prefix, names)
    return json.loads(names.pop())


def _roles_with(stmts: dict, pred) -> set[tuple[str, str]]:
    return {key for key, sts in stmts.items() if any(pred(st) for st in sts)}


def _prefixes(synth, keys: set[tuple[str, str]], wanted: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """Map role logical ids back to the ``wanted`` prefixes (each must match exactly one id)."""
    out = set()
    for stack, lid in keys:
        hits = [(s, p) for s, p in wanted if s == stack and lid.startswith(p)]
        out |= set(hits) if hits else {(stack, lid)}
    return out


# ------------------------------------------------------------------- call path (both hostings)


def test_scan_is_only_the_catalog_scan_statement(synth_matrix) -> None:
    catalog = _export_of(synth_matrix, "CatalogTable")
    for key, sts in _role_statements(synth_matrix).items():
        for st in sts:
            if any(a.startswith("dynamodb:") and "Scan" in a for a in _actions(st)):
                assert st.get("Sid") == "CatalogScan", (key, st)
                assert st["Effect"] == "Allow"
                assert _actions(st) == ["dynamodb:Scan"], (key, st)
                assert _resources(st) == [catalog], (key, st)
                assert "Condition" not in st
        assert len([st for st in sts if st.get("Sid") == "CatalogScan"]) <= 1, key


def test_the_roles_that_scan_are_exactly_the_catalog_loaders(synth_matrix) -> None:
    stmts = _role_statements(synth_matrix)
    scanners = _roles_with(stmts, lambda st: "dynamodb:Scan" in _actions(st))
    want = _loaders(synth_matrix, CATALOG_LOADERS)
    assert _prefixes(synth_matrix, scanners, want) == want


def test_the_bridge_scanner_is_the_hostings_role(synth_matrix) -> None:
    """Fargate: the ECS task role; AgentCore: the Runtime execution role (also ``TaskRole``)."""
    role = _lid(synth_matrix, BRIDGE, "TaskRole", "AWS::IAM::Role")
    trust = synth_matrix.templates[BRIDGE]["Resources"][role]["Properties"]["AssumeRolePolicyDocument"]
    service = trust["Statement"][0]["Principal"]["Service"]
    want = "bedrock-agentcore.amazonaws.com" if synth_matrix.cfg.is_agentcore else "ecs-tasks.amazonaws.com"
    assert service == want


def test_policy_loaders_may_batch_get_the_policy_table(synth_matrix) -> None:
    policy = _export_of(synth_matrix, "PolicyTable")
    stmts = _role_statements(synth_matrix)
    batch = _roles_with(stmts, lambda st: "dynamodb:BatchGetItem" in _actions(st)
                        and policy in _resources(st))
    want = _loaders(synth_matrix, POLICY_LOADERS)
    assert want <= _prefixes(synth_matrix, batch, want)


def test_catalog_scanners_need_no_new_kms_statement(synth_matrix) -> None:
    """The catalog table uses the data CMK, which each scanner's one DynamoDB-scoped KMS statement
    already names; nothing grants KMS for the Scan on its own."""
    sse = synth_matrix.templates[DATA]["Resources"][_lid(synth_matrix, DATA, "CatalogTable",
                                                          "AWS::DynamoDB::Table")]["Properties"]["SSESpecification"]
    key_lid = sse["KMSMasterKeyId"]["Fn::GetAtt"][0]
    stmts = _role_statements(synth_matrix)
    for key in _roles_with(stmts, lambda st: st.get("Sid") == "CatalogScan"):
        via_dynamodb = [st for st in stmts[key] if any(a.startswith("kms:") for a in _actions(st))
                        and "dynamodb." in json.dumps(st.get("Condition", {}))]
        assert len(via_dynamodb) == 1, (key, via_dynamodb)
        assert via_dynamodb[0]["Sid"] == "TablesKmsViaDynamoDB"
        assert "kms:Decrypt" in _actions(via_dynamodb[0])
        assert key_lid in json.dumps(via_dynamodb[0]["Resource"])


# ------------------------------------------------------------------------------- admin console


@pytest.fixture(scope="module")
def admin(tmp_path_factory):
    return _synth(tmp_path_factory, "f11-admin", {"adminConsole": "true"})


def test_admin_api_scans_the_catalog_and_batch_gets_the_policy(admin) -> None:
    stmts = _role_statements(admin)
    catalog, policy = _export_of(admin, "CatalogTable"), _export_of(admin, "PolicyTable")
    scanners = _roles_with(stmts, lambda st: "dynamodb:Scan" in _actions(st))
    assert {k for k in scanners if k[0] == ADMIN} == {(ADMIN, _lid(admin, ADMIN, "ApiFnRole", "AWS::IAM::Role"))}
    (role,) = [k for k in scanners if k[0] == ADMIN]
    by_sid = {st.get("Sid"): st for st in stmts[role]}
    assert _actions(by_sid["CatalogScan"]) == ["dynamodb:Scan"]
    assert _resources(by_sid["CatalogScan"]) == [catalog]
    assert _actions(by_sid["PolicyBatchRead"]) == ["dynamodb:BatchGetItem"]
    assert _resources(by_sid["PolicyBatchRead"]) == [policy]
    # Neither action is in the shared table statement.
    assert not {"dynamodb:Scan", "dynamodb:BatchGetItem"} & set(_actions(by_sid["ConfigAndAuditTables"]))
