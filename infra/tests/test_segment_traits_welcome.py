"""Web-activity traits: the auth service's trait settings.

The auth service reads the traits to choose the welcome offer; it makes no model call, so its role
has no Bedrock statement in any context and ``welcomeModelId`` no longer exists. It gets the trait
settings and the Memory store ID; the front door the trait groups.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
from conftest import _synth
from test_slot_mode_ceiling import _bedrock_invoke, _fn_env
from test_synth_and_nag import _nag_report

from owlbank_infra import app_builder
from owlbank_infra.config import ConfigError, InfraConfig, from_context

BASE = Path(__file__).resolve().parents[1]
README = BASE / "README.md"
WELCOME_SIDS = {"WelcomeModelProfile", "WelcomeModelViaProfile"}


@pytest.fixture(scope="module")
def synth_no_web(tmp_path_factory):
    return _synth(tmp_path_factory, "no-web-activity", {"memoryTraitGroups": "Contact"})


def _role_statements(synth, stack: str, role_prefix: str) -> list[dict]:
    res = synth.templates[stack]["Resources"]
    role_ids = [lid for lid, r in res.items() if r["Type"] == "AWS::IAM::Role" and lid.startswith(role_prefix)]
    assert len(role_ids) == 1, (role_prefix, role_ids)
    out = []
    for r in res.values():
        if r["Type"] == "AWS::IAM::Policy" and {"Ref": role_ids[0]} in r["Properties"].get("Roles", []):
            out.extend(r["Properties"]["PolicyDocument"]["Statement"])
    for p in res[role_ids[0]]["Properties"].get("Policies", []):
        out.extend(p["PolicyDocument"]["Statement"])
    return out


def _all_statements(synth, stack: str) -> list[dict]:
    out = []
    for r in synth.templates[stack]["Resources"].values():
        if r["Type"] == "AWS::IAM::Policy":
            out.extend(r["Properties"]["PolicyDocument"]["Statement"])
        elif r["Type"] == "AWS::IAM::Role":
            for p in r["Properties"].get("Policies", []):
                out.extend(p["PolicyDocument"]["Statement"])
    return out


# ---------------------------------------------------------------------------------- config

def test_welcome_model_id_is_gone() -> None:
    """Unknown context keys are ignored (build_app reads a fixed key list): nothing reads it."""
    cfg = from_context({"welcomeModelId": "us.anthropic.claude-sonnet-5-5"}, base_dir=BASE)
    assert not hasattr(cfg, "welcome_model_id") and not hasattr(cfg, "welcome_on")
    assert "welcomeModelId" not in inspect.getsource(app_builder)


# ---------------------------------------------------------------------------------- IAM

@pytest.mark.parametrize("synth_name", ["synth_default", "synth_no_web", "synth_agentcore_public_dev"])
def test_no_bedrock_statement_on_the_auth_service_role(synth_name, request) -> None:
    synth = request.getfixturevalue(synth_name)
    assert _bedrock_invoke(_role_statements(synth, "OwlBankCompute", "AuthServiceRole")) == []
    acks = [(p, r) for p, r, _ in synth.acks if p.startswith("OwlBankCompute/AuthServiceRole")
            and "bedrock" in r]
    assert acks == []
    assert not [s for s in _all_statements(synth, "OwlBankCompute") if s.get("Sid", "") in WELCOME_SIDS]


def test_welcome_model_id_context_fails_synth(tmp_path_factory) -> None:
    """The removed key is unknown context, so synth refuses it."""
    with pytest.raises(ConfigError, match="'welcomeModelId'"):
        _synth(tmp_path_factory, "welcome-ignored", {"welcomeModelId": "us.anthropic.claude-sonnet-5-5"})


@pytest.mark.parametrize("synth_name", ["synth_default", "synth_no_web", "synth_agentcore_public_dev"])
def test_front_door_and_bridge_statements_unchanged(synth_name, request) -> None:
    synth = request.getfixturevalue(synth_name)
    fd = _bedrock_invoke(_role_statements(synth, "OwlBankCompute", "FrontdoorRole"))
    assert {s["Sid"] for s in fd} == {"SlotResolverProfile", "SlotResolverViaProfile"}
    for s in _all_statements(synth, "OwlBankBridge"):
        assert not s.get("Sid", "").startswith("WelcomeModel"), s


@pytest.mark.parametrize("synth_name", ["synth_default", "synth_no_web", "synth_agentcore_public_dev"])
def test_auth_service_may_only_encrypt_with_the_token_key(synth_name, request) -> None:
    """12.4, 13.4: it seals the prepared welcome to the call; the front door unseals it. Encrypt
    only, on the token key only, and only with a ``call_sid`` encryption context."""
    synth = request.getfixturevalue(synth_name)
    kms_stmts = [s for s in _role_statements(synth, "OwlBankCompute", "AuthServiceRole")
                 if any(str(a).startswith("kms:") for a in (s["Action"] if isinstance(s["Action"], list)
                                                            else [s["Action"]]))
                 and s.get("Sid") != "KmsDecryptViaSecretsManager"]
    sealing = [s for s in kms_stmts if s.get("Sid") == "WelcomeSealing"]
    assert len(sealing) == 1
    assert sealing[0]["Action"] == "kms:Encrypt"
    assert sealing[0]["Condition"] == {"ForAnyValue:StringEquals": {"kms:EncryptionContextKeys": ["call_sid"]}}
    assert "*" not in str(sealing[0]["Resource"])
    # No other use of the token key (the tables' data key grant is a different key).
    assert [s for s in kms_stmts if s["Resource"] == sealing[0]["Resource"]] == sealing
    assert "OWLBANK_TOKEN_KMS_KEY" in _fn_env(synth, "OwlBankCompute", "AuthServiceFn")


def test_auth_service_access_is_unchanged() -> None:
    """It already reads ``customers`` (``first_name``, ``memory_email``) and the Twilio API key
    secret (the Memory reads' credentials): nothing new is granted for them."""
    from owlbank_infra.compute_stack import ACCESS
    ro, _, secrets = ACCESS["auth_service"]
    assert "customers" in ro and secrets == ("twilio_api_key",)


# ---------------------------------------------------------------------------------- env

def test_auth_service_env(synth_default) -> None:
    env = _fn_env(synth_default, "OwlBankCompute", "AuthServiceFn")
    assert env["OWLBANK_MEMORY_TRAIT_GROUPS"] == "Contact,WebActivity"
    assert env["OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS"] == "24"
    assert "OWLBANK_WELCOME_MODEL_ID" not in env
    assert "OWLBANK_SECRET_TWILIO_API_KEY" in env
    # No twilioOutputs in the default fixture: no store ID.
    assert "OWLBANK_MEMORY_STORE_ID" not in env


def test_auth_service_gets_the_memory_store_id(synth_domain) -> None:
    env = _fn_env(synth_domain, "OwlBankCompute", "AuthServiceFn")
    assert env["OWLBANK_MEMORY_STORE_ID"] == "mem_store_01"


def test_env_follows_the_settings(tmp_path_factory) -> None:
    synth = _synth(tmp_path_factory, "welcome-env", {
        "memoryTraitGroups": "Contact", "webActivityMaxAgeHours": 48})
    auth = _fn_env(synth, "OwlBankCompute", "AuthServiceFn")
    assert auth["OWLBANK_MEMORY_TRAIT_GROUPS"] == "Contact"
    assert auth["OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS"] == "48"
    assert "OWLBANK_WELCOME_MODEL_ID" not in auth
    assert _fn_env(synth, "OwlBankCompute", "FrontdoorFn")["OWLBANK_MEMORY_TRAIT_GROUPS"] == "Contact"


def test_front_door_env(synth_default) -> None:
    env = _fn_env(synth_default, "OwlBankCompute", "FrontdoorFn")
    assert env["OWLBANK_MEMORY_TRAIT_GROUPS"] == "Contact,WebActivity"
    assert "OWLBANK_WELCOME_MODEL_ID" not in env


def test_other_functions_get_no_welcome_settings(synth_default) -> None:
    for lid, r in synth_default.templates["OwlBankCompute"]["Resources"].items():
        if r["Type"] != "AWS::Lambda::Function" or lid.startswith(("AuthServiceFn", "FrontdoorFn")):
            continue
        env = r["Properties"].get("Environment", {}).get("Variables", {})
        for name in ("OWLBANK_MEMORY_TRAIT_GROUPS", "OWLBANK_WELCOME_MODEL_ID", "OWLBANK_MEMORY_STORE_ID"):
            assert name not in env, (lid, name)


# ---------------------------------------------------------------------------------- docs

def _row(key: str) -> list[str]:
    rows = [line for line in README.read_text().splitlines() if line.startswith(f"| `{key}` |")]
    assert len(rows) == 1, (key, rows)
    return [c.strip() for c in rows[0].strip("|").split("|")]


def test_settings_table_documents_the_welcome_settings() -> None:
    assert not [ln for ln in README.read_text().splitlines() if ln.startswith("| `welcomeModelId`")]
    assert "OWLBANK_WELCOME_MODEL_ID" not in README.read_text()
    _, default, meaning = _row("memoryTraitGroups")
    assert default == "`" + ",".join(InfraConfig.memory_trait_groups) + "`"
    assert re.search(r"auth service, which reads the traits to choose the welcome offer", meaning)
    assert "Standard_Administrator" not in meaning
    _, default, meaning = _row("webActivityMaxAgeHours")
    assert default == f"`{InfraConfig.web_activity_max_age_hours}`" and "auth service" in meaning


# ---------------------------------------------------------------------------------- cdk-nag

@pytest.mark.parametrize("synth_name", ["synth_default", "synth_no_web"])
def test_cdk_nag_clean(synth_name, request) -> None:
    synth = request.getfixturevalue(synth_name)
    report = _nag_report(synth)
    if report is not None:
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [(v["ruleName"], v.get("violatingResources")) for v in errors]
        assert report["conclusion"] == "success"
