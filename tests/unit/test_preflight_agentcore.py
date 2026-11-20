"""AgentCore hosting: preflight's AgentCore availability and quota checks.

Clients are botocore Stubbers (and one fake for a connection failure): no AWS account, no network.
No PBT (I/O glue; assertions instead).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import EndpointConnectionError
from botocore.stub import Stubber

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "infra" / "scripts" / "preflight.py"
_spec = importlib.util.spec_from_file_location("preflight_agentcore", SCRIPT)
assert _spec and _spec.loader
preflight = importlib.util.module_from_spec(_spec)
sys.modules["preflight_agentcore"] = preflight  # dataclasses need the module registered
_spec.loader.exec_module(preflight)

ACCOUNT = "111111111111"  # fictional
SVC = {"ServiceCode": "bedrock-agentcore"}


def _client(service, region="us-east-1"):
    return boto3.client(service, region_name=region, aws_access_key_id="testing",
                        aws_secret_access_key="testing")


def _quota(name, value, code="L-00000000"):
    return {"ServiceCode": "bedrock-agentcore", "QuotaName": name, "QuotaCode": code, "Value": value,
            "QuotaArn": f"arn:aws:servicequotas:us-east-1:{ACCOUNT}:bedrock-agentcore/{code}",
            "Adjustable": True}


D8_VALUES = {preflight.ACTIVE_SESSIONS_QUOTA: 5000.0, preflight.NEW_SESSION_RATE_QUOTA: 25.0,
             "Rate of Runtime data plane APIs": 1000.0, "Streaming maximum duration": 60.0,
             "WebSocket frame size": 32.0, "WebSocket frame rate": 250.0}


def _quotas(applied, defaults=None):
    """A Service Quotas client stubbed with ``applied`` (and, if given, the AWS defaults page)."""
    c = _client("service-quotas")
    stub = Stubber(c)
    stub.add_response("list_service_quotas",
                      {"Quotas": [_quota(n, v) for n, v in applied.items()]}, SVC)
    if defaults is not None:
        stub.add_response("list_aws_default_service_quotas",
                          {"Quotas": [_quota(n, v) for n, v in defaults.items()]}, SVC)
    return c, stub


def _check_quotas(applied, defaults=None, session_quota=None, region="us-east-1"):
    c, stub = _quotas(applied, defaults)
    with stub:
        out = preflight.check_agentcore_quotas(c, region, session_quota)
        stub.assert_no_pending_responses()
    return out


# ------------------------------------------------------------------ availability

def test_available_region_passes_after_the_probe():
    c = _client("bedrock-agentcore-control")
    with Stubber(c) as stub:
        stub.add_response("list_agent_runtimes", {"agentRuntimes": []}, {"maxResults": 1})
        r = preflight.check_agentcore_available(c, "us-east-1")
        stub.assert_no_pending_responses()
    assert r.ok and "us-east-1" in r.name


def test_unlisted_region_fails_naming_it_without_probing():
    class NoCalls:
        def __getattr__(self, name):
            raise AssertionError(f"probed {name}")

    r = preflight.check_agentcore_available(NoCalls(), "af-south-1")
    assert not r.ok and "af-south-1" in r.name and "af-south-1" in r.detail
    assert "bridgeHosting=fargate" in r.fix


@pytest.mark.parametrize("code", ["UnrecognizedClientException", "ValidationException",
                                  "ServiceUnavailableException"])
def test_failed_probe_fails_naming_the_region(code):
    c = _client("bedrock-agentcore-control", "eu-west-1")
    with Stubber(c) as stub:
        stub.add_client_error("list_agent_runtimes", service_error_code=code)
        r = preflight.check_agentcore_available(c, "eu-west-1")
    assert not r.ok and "eu-west-1" in r.detail and code in r.detail


def test_unreachable_endpoint_fails_naming_the_region():
    class Unreachable:
        def list_agent_runtimes(self, **kw):
            raise EndpointConnectionError(endpoint_url="https://bedrock-agentcore-control.example")

    r = preflight.check_agentcore_available(Unreachable(), "us-west-1")
    assert not r.ok and "us-west-1" in r.detail and "EndpointConnectionError" in r.detail


def test_denied_probe_fails_asking_for_the_permission():
    c = _client("bedrock-agentcore-control")
    with Stubber(c) as stub:
        stub.add_client_error("list_agent_runtimes", service_error_code="AccessDeniedException")
        r = preflight.check_agentcore_available(c, "us-east-1")
    assert not r.ok and "bedrock-agentcore:ListAgentRuntimes" in r.fix


# ------------------------------------------------------------------ quotas

def test_quotas_printed_next_to_the_sizing_rule(capsys):
    out = _check_quotas(D8_VALUES)
    printed = capsys.readouterr().out
    assert all(r.ok and not r.warn for r in out)
    assert "5,000" in out[0].detail and "25 per second" in out[0].detail
    assert preflight.SIZING_RULE in out[0].detail and preflight.SIZING_RULE in printed
    for name in D8_VALUES:
        assert name in printed


def test_quota_missing_from_applied_falls_back_to_the_aws_default(capsys):
    applied = {preflight.ACTIVE_SESSIONS_QUOTA: 5000.0}
    defaults = {preflight.NEW_SESSION_RATE_QUOTA: 25.0, preflight.ACTIVE_SESSIONS_QUOTA: 1.0}
    out = _check_quotas(applied, defaults)
    printed = capsys.readouterr().out
    assert out[0].ok and "active sessions 5,000" in out[0].detail  # applied wins over the default
    assert f"{preflight.NEW_SESSION_RATE_QUOTA}: 25 (AWS default" in printed


def test_sizing_quota_not_found_anywhere_fails():
    out = _check_quotas({preflight.ACTIVE_SESSIONS_QUOTA: 5000.0},
                        {preflight.ACTIVE_SESSIONS_QUOTA: 5000.0})
    assert len(out) == 1 and not out[0].ok
    assert preflight.NEW_SESSION_RATE_QUOTA in out[0].detail and "list-service-quotas" in out[0].fix


def test_service_quotas_error_fails():
    c = _client("service-quotas")
    with Stubber(c) as stub:
        stub.add_client_error("list_service_quotas", service_error_code="AccessDeniedException")
        out = preflight.check_agentcore_quotas(c, "us-east-1")
    assert len(out) == 1 and not out[0].ok and "AccessDeniedException" in out[0].detail


def test_session_quota_mismatch_warns_without_failing(capsys):
    out = _check_quotas({**D8_VALUES, preflight.ACTIVE_SESSIONS_QUOTA: 8000.0})
    mismatch = out[-1]
    assert mismatch.ok and mismatch.warn
    assert "agentcoreSessionQuota=5,000" in mismatch.detail and "8,000" in mismatch.detail
    assert mismatch.fix == "deploy with -c agentcoreSessionQuota=8000"
    capsys.readouterr()
    assert preflight.report(out) == 0
    printed = capsys.readouterr().out
    assert "[WARN] agentcoreSessionQuota" in printed and "agentcoreSessionQuota=8000" in printed


def test_explicit_session_quota_is_compared():
    assert not _check_quotas(D8_VALUES, session_quota=5000)[-1].warn
    assert _check_quotas(D8_VALUES, session_quota=4000)[-1].warn


def test_default_session_quota_follows_the_region():
    assert preflight.default_session_quota("us-west-2") == 5000
    assert preflight.default_session_quota("eu-west-1") == 2500
    # In a 2,500 Region the measured-style 5,000 differs from the default: WARN.
    assert _check_quotas(D8_VALUES, region="eu-west-1")[-1].warn


# ------------------------------------------------------------------ run() and the CLI

def _run(control, quotas, capsys, **kw):
    class Sts:
        def get_caller_identity(self):
            return {"Account": ACCOUNT}

    import unittest.mock as mock
    with mock.patch.object(preflight, "check_s3_public_access_block",
                           lambda *a: preflight.Result("bpa", True, "")), \
            mock.patch.object(preflight, "check_inference_profile", lambda *a: []):
        results = preflight.run(sts=Sts(), s3control=None, bedrock=None, analyzer=None, stage="dev",
                                agentcore_control=control, quotas=quotas, **kw)
    # Only the AgentCore results: the logging and CloudTrail checks always run.
    skip = ("Bedrock invocation logging", "CloudTrail")
    return [r for r in results[1:] if not r.name.startswith(skip)], capsys.readouterr().out


def test_run_adds_availability_then_quotas(capsys):
    control = _client("bedrock-agentcore-control")
    quotas, qstub = _quotas(D8_VALUES)
    with Stubber(control) as cstub, qstub:
        cstub.add_response("list_agent_runtimes", {"agentRuntimes": [
            {"agentRuntimeArn": f"arn:aws:bedrock-agentcore:us-east-1:{ACCOUNT}:runtime/owlbank_dev_bridge-abc",
             "agentRuntimeId": "owlbank_dev_bridge-abc", "agentRuntimeVersion": "1",
             "agentRuntimeName": "owlbank_dev_bridge", "description": "bridge","lastUpdatedAt": "2026-10-19",
             "status": "READY"}]}, {"maxResults": 1})
        results, printed = _run(control, quotas, capsys, region="us-east-1")
    names = [r.name for r in results]
    assert names == ["AgentCore Runtime offered in us-east-1", "AgentCore quotas (us-east-1)",
                     "agentcoreSessionQuota matches the active-session quota"]
    assert preflight.report(results) == 0
    printed += capsys.readouterr().out
    # Nothing account-specific is printed: no account ID, no ARN.
    assert ACCOUNT not in printed and "arn:" not in printed


def test_run_skips_quotas_when_unavailable(capsys):
    class NoCalls:
        def __getattr__(self, name):
            raise AssertionError(f"called {name}")

    results, _ = _run(NoCalls(), NoCalls(), capsys, region="af-south-1")
    assert [r.ok for r in results] == [False]
    assert preflight.report(results) == 1


def test_run_without_agentcore_clients_adds_nothing(capsys):
    results, _ = _run(None, None, capsys)
    assert results == []


@pytest.mark.parametrize("argv,quota", [([], None), (["--agentcore-session-quota", "8000"], 8000)])
def test_cli_agentcore_builds_the_clients(argv, quota, monkeypatch):
    seen = {}

    class Session:
        def __init__(self, **kw):
            pass

        def client(self, name):
            return name

    monkeypatch.setitem(sys.modules, "boto3", type(sys)("boto3"))
    sys.modules["boto3"].Session = Session
    monkeypatch.setattr(preflight, "run", lambda **kw: seen.update(kw) or [])
    assert preflight.main(["--agentcore", "--no-prefix-list", *argv]) == 0
    assert seen["agentcore_control"] == "bedrock-agentcore-control"
    assert seen["quotas"] == "service-quotas"
    assert seen["agentcore_session_quota"] == quota
    seen.clear()
    assert preflight.main(["--no-prefix-list"]) == 0
    assert seen["agentcore_control"] is None and seen["quotas"] is None
