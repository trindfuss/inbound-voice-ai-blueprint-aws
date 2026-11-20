"""The post-call recap writer's infrastructure.

Template assertions over offline synths:

* The writer exists only with ``memoryStoreId`` in ``twilioOutputs`` (its handler requires it):
  ``lambdas.recap_writer.handler.handler``, 256 MB, 60 s, one async retry, its own log group and a
  fixed-name role. Without the store: no function, no token-key statement, no
  ``OWLBANK_RECAP_FUNCTION`` anywhere.
* IAM, exactly: the four tables (``calls`` GetItem/PutItem/DeleteItem, ``customers`` GetItem,
  ``policy`` GetItem/BatchGetItem, ``catalog`` Scan), the data CMK through DynamoDB and Secrets
  Manager only, ``bedrock:InvokeModel`` on the recap model's inference profile and its foundation
  model only through that profile, ``comprehend:DetectPiiEntities`` (the only ``*`` resource besides
  the factory's X-Ray statement), ``kms:Decrypt`` on the token key with a ``call_sid`` context, and
  the ``twilio_api_key`` secret by its one ARN. The token key's policy names the role.
* The front door and (with ``humanLegConversation=on``) the sweep may invoke the writer's one
  unqualified ARN, and carry it as ``OWLBANK_RECAP_FUNCTION``; no other role may invoke it.

PBT: skipped (CDK wiring; fixed assertions over a small context matrix).
"""

from __future__ import annotations

import json

import pytest
from conftest import _synth

from owlbank_infra.compute_stack import (
    RECAP_FUNCTION_ENV,
    RECAP_MODEL_ENV,
    RECAP_TABLE_ACTIONS,
    RECAP_WRITER,
)

COMPUTE, DATA = "OwlBankCompute", "OwlBankData"
WRITER_FN, WRITER_ROLE = "RecapWriterFn", "RecapWriterRole"
STORE = "mem_store_01"
SECRET_ACTIONS = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
CALL_SID = {"ForAnyValue:StringEquals": {"kms:EncryptionContextKeys": ["call_sid"]}}


@pytest.fixture(scope="module")
def rw(tmp_path_factory) -> dict:
    path = tmp_path_factory.mktemp("rw-tw") / "twilio.context.json"
    path.write_text(json.dumps({"accountSid": "AC" + "1" * 32, "memoryStoreId": STORE,
                                "workflowSid": "WW" + "2" * 32, "workspaceSid": "WS" + "4" * 32}))
    contexts = {
        "on": {"twilioOutputs": str(path)},
        "on-hl-off": {"twilioOutputs": str(path), "humanLegConversation": "off"},
        "on-model": {"twilioOutputs": str(path), "resolverModelId": "us.anthropic.claude-sonnet-5-5"},
    }
    return {name: _synth(tmp_path_factory, f"rw-{name}", ctx) for name, ctx in contexts.items()}


# --------------------------------------------------------------------------- helpers

def _res(synth, stack: str = COMPUTE) -> dict:
    return synth.templates[stack]["Resources"]


def _find(synth, prefix: str, rtype: str) -> list[tuple[str, dict]]:
    return [(lid, r) for lid, r in _res(synth).items() if r["Type"] == rtype and lid.startswith(prefix)]


def _one(synth, prefix: str, rtype: str) -> tuple[str, dict]:
    found = _find(synth, prefix, rtype)
    assert len(found) == 1, (prefix, [lid for lid, _ in found])
    return found[0]


def _env(synth, fn_prefix: str) -> dict:
    return _one(synth, fn_prefix, "AWS::Lambda::Function")[1]["Properties"]["Environment"]["Variables"]


def _role_statements(synth, role_prefix: str) -> list[dict]:
    role, _ = _one(synth, role_prefix, "AWS::IAM::Role")
    return [st for r in _res(synth).values()
            if r["Type"] == "AWS::IAM::Policy" and {"Ref": role} in r["Properties"].get("Roles", [])
            for st in r["Properties"]["PolicyDocument"]["Statement"]]


def _actions(st: dict) -> list[str]:
    return st["Action"] if isinstance(st["Action"], list) else [st["Action"]]


def _by_sid(synth) -> dict[str, dict]:
    sts = _role_statements(synth, WRITER_ROLE)
    named = {st["Sid"]: st for st in sts if "Sid" in st}
    assert len(named) == len({st["Sid"] for st in sts if "Sid" in st})
    return named


def _import(synth, export_fragment: str) -> dict:
    """``{"Fn::ImportValue": <the one Data export whose name contains export_fragment>}``."""
    names = [o["Export"]["Name"] for o in synth.templates[DATA].get("Outputs", {}).values()
             if "Export" in o and export_fragment in o["Export"]["Name"]]
    assert len(names) == 1, (export_fragment, names)
    return {"Fn::ImportValue": names[0]}


# --------------------------------------------------------------------------- the function

@pytest.mark.parametrize("name", ["on", "on-hl-off"])
def test_writer_function_shape(rw, name) -> None:
    synth = rw[name]
    fn_lid, fn = _one(synth, WRITER_FN, "AWS::Lambda::Function")
    p = fn["Properties"]
    assert p["Handler"] == f"lambdas.{RECAP_WRITER}.handler.handler"
    assert p["Runtime"] == "python3.12" and p["Architectures"] == ["arm64"]
    assert p["MemorySize"] == 256 and p["Timeout"] == 60
    assert p["TracingConfig"] == {"Mode": "Active"}
    # Its own CMK-encrypted log group, /owlbank/lambda/recap_writer.
    log_lid, log = _one(synth, "RecapWriterLogs", "AWS::Logs::LogGroup")
    assert log["Properties"]["LogGroupName"].endswith("/owlbank/lambda/recap_writer")
    assert p["LoggingConfig"]["LogGroup"] == {"Ref": log_lid}
    # Exactly one Lambda async retry.
    (cfg,) = [r["Properties"] for _, r in _find(synth, "", "AWS::Lambda::EventInvokeConfig")
              if r["Properties"]["FunctionName"] == {"Ref": fn_lid}]
    assert cfg["MaximumRetryAttempts"] == 1
    # Fixed role name (the token-key policy names it).
    _, role = _one(synth, WRITER_ROLE, "AWS::IAM::Role")
    assert role["Properties"]["RoleName"] == "owlbank-dev-recap-writer"
    # No URL, no API, no trigger, no resource policy.
    assert not [r for r in _res(synth).values() if r["Type"] in (
        "AWS::Lambda::Url", "AWS::Lambda::EventSourceMapping", "AWS::Events::Rule", "AWS::Lambda::Permission")
        and fn_lid in json.dumps(r)]
    assert fn_lid not in json.dumps(synth.templates["OwlBankApi"])
    # Errors and Throttles alarms on it.
    alarms = [r["Properties"] for _, r in _find(synth, "", "AWS::CloudWatch::Alarm")
              if {"Ref": fn_lid} in [d.get("Value") for d in r["Properties"].get("Dimensions", [])]]
    assert {a["MetricName"] for a in alarms} == {"Errors", "Throttles"}


def test_writer_env(rw) -> None:
    synth = rw["on"]
    env = _env(synth, WRITER_FN)
    assert env["OWLBANK_MEMORY_STORE_ID"] == STORE
    assert env[RECAP_MODEL_ENV] == synth.cfg.resolver_model_id == "us.anthropic.claude-haiku-5-5"
    assert env["OWLBANK_SECRET_TWILIO_API_KEY"] == _import(synth, "TwilioApiKeySecret")
    assert env["OWLBANK_TOKEN_KMS_KEY"] == _import(synth, "TokenKey")
    assert {k for k in env if k.startswith("OWLBANK_TABLE_")} == {
        "OWLBANK_TABLE_CALLS", "OWLBANK_TABLE_CUSTOMERS", "OWLBANK_TABLE_POLICY", "OWLBANK_TABLE_CATALOG"}
    assert _env(rw["on-model"], WRITER_FN)[RECAP_MODEL_ENV] == "us.anthropic.claude-sonnet-5-5"


def test_off_without_a_memory_store(synth_default) -> None:
    assert not synth_default.cfg.recap_writer_on
    blob = json.dumps(synth_default.templates)
    assert WRITER_FN not in blob and WRITER_ROLE not in blob and "recap_writer" not in blob
    assert RECAP_FUNCTION_ENV not in blob and "recap-writer" not in blob


# --------------------------------------------------------------------------- the writer's grants

@pytest.mark.parametrize("name", ["on", "on-model"])
def test_writer_grants_are_exactly_these(rw, name) -> None:
    synth = rw[name]
    sts = _role_statements(synth, WRITER_ROLE)
    sids = _by_sid(synth)
    # The factory's X-Ray statement has no Sid, nor does CDK's secret grant; everything else does.
    unnamed = [st for st in sts if "Sid" not in st]
    assert sorted(tuple(_actions(st)) for st in unnamed) == sorted(
        [tuple(SECRET_ACTIONS), ("xray:PutTraceSegments", "xray:PutTelemetryRecords")])
    assert set(sids) == {"WriteOwnLogs", "TableCalls", "TableCustomers", "TablePolicy", "CatalogScan",
                         "TablesKmsViaDynamoDB", "RecapModelProfile", "RecapModelViaProfile",
                         "DetectPiiForRedaction", "SensitiveUnseal", "DecryptSecretsViaSecretsManager"}
    for st in sts:
        assert st["Effect"] == "Allow" and "NotAction" not in st and "NotResource" not in st
        assert not [a for a in _actions(st) if "*" in a], st


def test_writer_table_grants(rw) -> None:
    synth = rw["on"]
    sids = _by_sid(synth)
    want = {"TableCalls": ("CallsTable", ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"]),
            "TableCustomers": ("CustomersTable", ["dynamodb:GetItem"]),
            "TablePolicy": ("PolicyTable", ["dynamodb:GetItem", "dynamodb:BatchGetItem"]),
            "CatalogScan": ("CatalogTable", ["dynamodb:Scan"])}
    for sid, (table, actions) in want.items():
        assert sorted(_actions(sids[sid])) == sorted(actions), sid
        assert not isinstance(sids[sid]["Resource"], list), sid  # the one table ARN
        assert table in json.dumps(sids[sid]["Resource"]), sid
        assert "/index/" not in json.dumps(sids[sid]["Resource"]), sid
    assert {k: sorted(v) for k, v in RECAP_TABLE_ACTIONS.items()} == {
        "calls": sorted(want["TableCalls"][1]), "customers": want["TableCustomers"][1],
        "policy": sorted(want["TablePolicy"][1])}
    # No other table: tokens, segments, risk are never reached.
    dynamo = [st for st in _role_statements(synth, WRITER_ROLE)
              if any(a.startswith("dynamodb:") for a in _actions(st))]
    reached = json.dumps([st["Resource"] for st in dynamo])
    assert not [t for t in ("Tokens", "Segments", "Risk") if f"{t}Table" in reached]
    kms = sids["TablesKmsViaDynamoDB"]
    assert kms["Resource"] == _import(synth, "DataKey")
    assert kms["Condition"] == {"StringEquals": {"kms:ViaService": "dynamodb.us-east-1.amazonaws.com"}}


@pytest.mark.parametrize("name,model", [("on", "anthropic.claude-haiku-5-5"),
                                        ("on-model", "anthropic.claude-sonnet-5-5")])
def test_writer_bedrock_only_on_the_recap_model(rw, name, model) -> None:
    synth = rw[name]
    bedrock = [st for st in _role_statements(synth, WRITER_ROLE)
               if any(a.startswith("bedrock:") for a in _actions(st))]
    assert len(bedrock) == 2 and all(_actions(st) == ["bedrock:InvokeModel"] for st in bedrock)
    sids = _by_sid(synth)
    profile = sids["RecapModelProfile"]["Resource"]
    assert json.dumps(profile).endswith(f':bedrock:us-east-1:111111111111:inference-profile/us.{model}"]]}}')
    assert "Condition" not in sids["RecapModelProfile"]
    via = sids["RecapModelViaProfile"]
    # The foundation model in any Region, but only through our geo profile (bedrock_invoke_statements).
    assert json.dumps(via["Resource"]).endswith(f':bedrock:*::foundation-model/{model}"]]}}')
    assert via["Condition"] == {"StringEquals": {"bedrock:InferenceProfileArn": profile}}


def test_writer_star_resources_are_only_comprehend_and_xray(rw) -> None:
    """No wildcard resource except ``comprehend:DetectPiiEntities`` (no resource-level
    permissions exist for it; PII_DETECT_REASON) and the factory's X-Ray statement. The Bedrock
    foundation-model ARN's Region ``*`` is conditioned on our inference profile (checked above)."""
    for st in _role_statements(rw["on"], WRITER_ROLE):
        if st["Resource"] == "*":
            assert _actions(st) in (["comprehend:DetectPiiEntities"],
                                    ["xray:PutTraceSegments", "xray:PutTelemetryRecords"]), st
        elif st.get("Sid") == "RecapModelViaProfile":
            assert "bedrock:InferenceProfileArn" in st["Condition"]["StringEquals"]
        else:
            assert "*" not in json.dumps(st["Resource"]), st


def test_writer_token_key_decrypt_only(rw) -> None:
    synth = rw["on"]
    sids = _by_sid(synth)
    st = sids["SensitiveUnseal"]
    assert _actions(st) == ["kms:Decrypt"]
    assert st["Resource"] == _import(synth, "TokenKey")
    assert st["Condition"] == CALL_SID
    # No other use of the token key by this role.
    token = json.dumps(_import(synth, "TokenKey"))
    assert [s.get("Sid") for s in _role_statements(synth, WRITER_ROLE) if token in json.dumps(s["Resource"])] \
        == ["SensitiveUnseal"]
    # The key policy names the role, decrypt only, with the call_sid context, and its Deny exempts it.
    key = next(r for lid, r in _res(synth, DATA).items()
               if r["Type"] == "AWS::KMS::Key" and lid.startswith("TokenKey"))
    stmts = {s["Sid"]: s for s in key["Properties"]["KeyPolicy"]["Statement"]}
    rw_arn = json.dumps(stmts["RecapWriterUnseal"]["Condition"]["ArnEquals"]["aws:PrincipalArn"])
    assert "role/owlbank-dev-recap-writer" in rw_arn
    assert stmts["RecapWriterUnseal"]["Action"] == "kms:Decrypt"
    assert {k: v for k, v in stmts["RecapWriterUnseal"]["Condition"].items() if k != "ArnEquals"} == CALL_SID
    assert "role/owlbank-dev-recap-writer" in json.dumps(stmts["DenyCryptoExceptSealingRoles"]["Condition"])


def test_writer_reads_only_the_api_key_secret(rw) -> None:
    synth = rw["on"]
    sts = [st for st in _role_statements(synth, WRITER_ROLE)
           if any(a.startswith("secretsmanager:") for a in _actions(st))]
    assert len(sts) == 1 and _actions(sts[0]) == SECRET_ACTIONS
    assert sts[0]["Resource"] == _import(synth, "TwilioApiKeySecret")
    kms = _by_sid(synth)["DecryptSecretsViaSecretsManager"]
    assert _actions(kms) == ["kms:Decrypt"] and kms["Resource"] == _import(synth, "DataKey")
    assert kms["Condition"] == {"StringEquals": {"kms:ViaService": "secretsmanager.us-east-1.amazonaws.com"}}


# --------------------------------------------------------------------------- the callers

@pytest.mark.parametrize("name,callers", [("on", {"FrontdoorRole", "ConversationSweepRole"}),
                                          ("on-hl-off", {"FrontdoorRole"})])
def test_only_the_callers_invoke_the_writer_by_its_arn(rw, name, callers) -> None:
    synth = rw[name]
    fn_lid, _ = _one(synth, WRITER_FN, "AWS::Lambda::Function")
    arn = {"Fn::GetAtt": [fn_lid, "Arn"]}
    holders = set()
    for stack, tpl in synth.templates.items():
        for r in tpl["Resources"].values():
            if r["Type"] != "AWS::IAM::Policy":
                continue
            for st in r["Properties"]["PolicyDocument"]["Statement"]:
                if fn_lid in json.dumps(st.get("Resource")) and "lambda:InvokeFunction" in _actions(st):
                    assert st["Resource"] == arn, st  # the one unqualified ARN, never :* or a list
                    assert st["Sid"] == "InvokeRecapWriter"
                    holders |= {(stack, ref["Ref"].rstrip("0123456789ABCDEF")) for ref in r["Properties"]["Roles"]}
    assert holders == {(COMPUTE, c) for c in callers}
    for caller in callers:
        fn_prefix = caller.removesuffix("Role") + "Fn"
        assert _env(synth, fn_prefix)[RECAP_FUNCTION_ENV] == arn
    others = [lid for lid, r in _res(synth).items() if r["Type"] == "AWS::Lambda::Function"
              and RECAP_FUNCTION_ENV in r["Properties"].get("Environment", {}).get("Variables", {})]
    assert sorted(o.rstrip("0123456789ABCDEF") for o in others) == sorted(
        c.removesuffix("Role") + "Fn" for c in callers)
