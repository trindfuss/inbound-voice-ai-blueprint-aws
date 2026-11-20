"""Preflight's AZ ID check and cache, with a stubbed EC2 client (no AWS calls)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import boto3
import pytest
from botocore.stub import Stubber

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"
spec = importlib.util.spec_from_file_location("preflight_azs", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
sys.modules["preflight_azs"] = preflight
spec.loader.exec_module(preflight)

ACCOUNT = "111111111111"
KEY = f"owlbank:az-ids:{ACCOUNT}:us-east-1"
FILTERS = {"Filters": [{"Name": "zone-type", "Values": ["availability-zone"]}]}


def _zones(mapping):
    return {"AvailabilityZones": [{"ZoneName": n, "ZoneId": i, "RegionName": "us-east-1",
                                   "State": "available", "ZoneType": "availability-zone"}
                                  for n, i in mapping.items()]}


def _check(mapping, tmp_path, vpc_azs=("us-east-1a", "us-east-1b"), existing=None):
    ec2 = boto3.client("ec2", region_name="us-east-1", aws_access_key_id="testing",
                       aws_secret_access_key="testing")
    ctx = tmp_path / "cdk.context.json"
    if existing is not None:
        ctx.write_text(existing, encoding="utf-8")
    with Stubber(ec2) as stub:
        stub.add_response("describe_availability_zones", _zones(mapping), FILTERS)
        results = preflight.check_agentcore_azs(ec2, ACCOUNT, "us-east-1", vpc_azs, ctx)
        stub.assert_no_pending_responses()
    return results, ctx


BOTH = {"us-east-1a": "use1-az1", "us-east-1b": "use1-az2", "us-east-1c": "use1-az3"}


def test_caches_map_under_account_region_key(tmp_path):
    results, ctx = _check(BOTH, tmp_path)
    assert json.loads(ctx.read_text())[KEY] == BOTH
    assert all(r.ok for r in results)
    assert not list(tmp_path.glob(".*.tmp"))


def test_cache_keeps_other_context_keys(tmp_path):
    other = {"availability-zones:account=1:region=x": ["a"], KEY: {"stale": "id"}}
    _, ctx = _check(BOTH, tmp_path, existing=json.dumps(other))
    doc = json.loads(ctx.read_text())
    assert doc["availability-zones:account=1:region=x"] == ["a"] and doc[KEY] == BOTH


def test_non_object_context_file_is_left_alone(tmp_path):
    results, ctx = _check(BOTH, tmp_path, existing="[1]")
    assert ctx.read_text() == "[1]"
    assert not results[0].ok


def test_one_supported_passes_with_single_az_detail(tmp_path):
    results, _ = _check({"us-east-1a": "use1-az3", "us-east-1b": "use1-az4"}, tmp_path)
    az = results[-1]
    assert az.ok and "us-east-1b" in az.detail and "agentcoreSingleAzAck" in az.detail


def test_none_supported_fails_naming_ids_and_candidates(tmp_path):
    mapping = {"us-east-1a": "use1-az3", "us-east-1b": "use1-az5", "us-east-1c": "use1-az1",
               "us-east-1d": "use1-az6", "us-east-1e": "use1-az4"}
    results, ctx = _check(mapping, tmp_path)
    az = results[-1]
    assert not az.ok
    for value in ("use1-az3", "use1-az5", *preflight.AGENTCORE_SUPPORTED_AZ_IDS["us-east-1"]):
        assert value in az.detail
    assert "availabilityZones" in az.fix and "us-east-1c" in az.fix and "us-east-1e" in az.fix
    assert json.loads(ctx.read_text())[KEY] == mapping  # cached even when failing


def test_explicit_vpc_azs_checked(tmp_path):
    results, _ = _check(BOTH, tmp_path, vpc_azs=("us-east-1a", "us-east-1c"))
    assert results[-1].ok and "only us-east-1a" in results[-1].detail


def test_vpc_az_not_in_account_fails(tmp_path):
    results, _ = _check(BOTH, tmp_path, vpc_azs=("us-east-1a", "us-east-1f"))
    assert not results[-1].ok and "us-east-1f" in results[-1].detail


def test_unlisted_region_fails(tmp_path):
    ec2 = boto3.client("ec2", region_name="eu-west-1", aws_access_key_id="testing",
                       aws_secret_access_key="testing")
    with Stubber(ec2) as stub:
        stub.add_response("describe_availability_zones", _zones({"eu-west-1a": "euw1-az1"}), FILTERS)
        results = preflight.check_agentcore_azs(ec2, ACCOUNT, "eu-west-1", ("eu-west-1a", "eu-west-1b"),
                                                tmp_path / "c.json")
    assert not results[-1].ok and "AGENTCORE_SUPPORTED_AZ_IDS" in results[-1].fix


def test_empty_describe_result_does_not_write_cache(tmp_path):
    results, ctx = _check({}, tmp_path)
    assert not results[0].ok and not ctx.exists()


def test_run_adds_az_check_only_with_az_client(tmp_path, monkeypatch):
    calls = []

    class Fake:
        def get_caller_identity(self):
            return {"Account": ACCOUNT}

    monkeypatch.setattr(preflight, "check_s3_public_access_block", lambda *a: preflight.Result("bpa", True, ""))
    monkeypatch.setattr(preflight, "check_inference_profile", lambda *a: [])
    monkeypatch.setattr(preflight, "check_agentcore_azs",
                        lambda *a: calls.append(a) or [preflight.Result("az", True, "")])
    preflight.run(sts=Fake(), s3control=None, bedrock=None, analyzer=None, stage="dev")
    assert calls == []
    ctx = tmp_path / "cdk.context.json"
    preflight.run(sts=Fake(), s3control=None, bedrock=None, analyzer=None, stage="dev",
                  az_ec2="ec2", region="us-east-1", context_file=ctx)
    assert calls == [("ec2", ACCOUNT, "us-east-1", ("us-east-1a", "us-east-1b"), ctx)]


@pytest.mark.parametrize("argv,expected", [([], None), (["--availability-zones", "us-east-1c, us-east-1d"],
                                                         ("us-east-1c", "us-east-1d"))])
def test_cli_flags_parse(argv, expected, monkeypatch):
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
    assert seen["az_ec2"] == "ec2" and seen["vpc_azs"] == expected
    assert seen["context_file"] == Path("cdk.context.json")
