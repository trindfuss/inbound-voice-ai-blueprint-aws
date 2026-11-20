"""scripts/preflight.py with botocore Stubber: no AWS account, no network, read-only calls only.

The Transaction Search check has no property-based tests: it is I/O glue over two read calls,
so the Stubber cases below cover its states (active, pending, off, no policy, unreadable)."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import boto3
import pytest
from botocore.stub import Stubber

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"
spec = importlib.util.spec_from_file_location("preflight", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
sys.modules["preflight"] = preflight  # dataclasses need the module registered
spec.loader.exec_module(preflight)

ACCOUNT = "111111111111"
ANALYZER_ARN = f"arn:aws:access-analyzer:us-east-1:{ACCOUNT}:analyzer/owlbank-dev-account"
#: Every API the script may call. All are reads; a write would need a reviewed change here.
READ_ONLY_OPS = {"GetCallerIdentity", "GetPublicAccessBlock", "GetInferenceProfile",
                 "GetFoundationModelAvailability", "ListAnalyzers", "ListFindings",
                 "DescribeManagedPrefixLists", "DescribeSecret", "DescribeStacks",
                 "ListStackResources", "GetBucketOwnershipControls", "GetBucketPolicyStatus",
                 "DescribeAvailabilityZones", "ListAgentRuntimes", "ListServiceQuotas",
                 "ListAwsDefaultServiceQuotas", "GetModelInvocationLoggingConfiguration",
                 "DescribeTrails", "GetTrailStatus", "GetEventSelectors",
                 "GetTraceSegmentDestination", "DescribeResourcePolicies"}
#: The Regions the invocation-logging check reads by default: the deployment Region only (the
#: source Region's configuration applies). ``ROUTED``: every Region ``us.`` routes to, for tests that
#: pass several Regions explicitly.
LOGGING_REGIONS = ("us-east-1",)
ROUTED = ("us-east-1", "us-east-2", "us-west-2")
TRAIL_ARN = f"arn:aws:cloudtrail:us-east-1:{ACCOUNT}:trail/management"


def _client(service, region="us-east-1"):
    return boto3.client(service, region_name=region, aws_access_key_id="testing",
                        aws_secret_access_key="testing")


def _profile(regions):
    return {"inferenceProfileName": "US Haiku", "inferenceProfileArn":
            f"arn:aws:bedrock:us-east-1:{ACCOUNT}:inference-profile/us.anthropic.claude-haiku-5-5",
            "inferenceProfileId": "us.anthropic.claude-haiku-5-5", "status": "ACTIVE",
            "type": "SYSTEM_DEFINED",
            "models": [{"modelArn": f"arn:aws:bedrock:{r}::foundation-model/anthropic.claude-haiku-5-5"}
                       for r in regions]}


AVAILABLE = {"modelId": "anthropic.claude-haiku-5-5", "agreementAvailability": {"status": "AVAILABLE"},
             "authorizationStatus": "AUTHORIZED", "entitlementAvailability": "AVAILABLE",
             "regionAvailability": "AVAILABLE"}
BPA_ON = {"PublicAccessBlockConfiguration": dict.fromkeys(preflight.BPA_KEYS, True)}


class Clients:
    def __init__(self):
        self.c = {n: _client(n) for n in ("sts", "s3control", "bedrock", "accessanalyzer",
                                                 "cloudformation", "s3", "cloudtrail")}
        # The deployment Region's Bedrock client is ``bedrock``; the other routed Regions get their own.
        for r in ROUTED[1:]:
            self.c[f"bedrock:{r}"] = _client("bedrock", r)
        self.s = {n: Stubber(c) for n, c in self.c.items()}

    def bedrock_for_region(self, region):
        return self.c["bedrock" if region == "us-east-1" else f"bedrock:{region}"]

    def add(self, svc, op, resp, params=None):
        self.s[svc].add_response(op, resp, params)

    def run(self, **kw):
        for s in self.s.values():
            s.activate()
        try:
            kw.setdefault("stage", "dev")
            return preflight.run(sts=self.c["sts"], s3control=self.c["s3control"], bedrock=self.c["bedrock"],
                                 analyzer=self.c["accessanalyzer"], cloudformation=self.c["cloudformation"],
                                 s3=self.c["s3"], cloudtrail=self.c["cloudtrail"],
                                 bedrock_for_region=self.bedrock_for_region, **kw)
        finally:
            for s in self.s.values():
                s.assert_no_pending_responses()
                s.deactivate()


def _pre(c, *, bpa=BPA_ON, regions=("us-east-1", "us-east-2", "us-west-1", "us-west-2"), avail=AVAILABLE,
         logging=None, trail=True, logging_regions=LOGGING_REGIONS):
    """The pre-deploy reads, in call order. ``logging`` maps a Region to its
    ``GetModelInvocationLoggingConfiguration`` response (default: none anywhere); ``trail`` stubs one
    multi-Region trail that logs management events (``False``: no trail at all)."""
    c.add("sts", "get_caller_identity", {"Account": ACCOUNT, "Arn": f"arn:aws:sts::{ACCOUNT}:assumed-role/x/y", "UserId": "AROAEXAMPLEID"})
    if isinstance(bpa, str):
        c.s["s3control"].add_client_error("get_public_access_block", service_error_code=bpa)
    else:
        c.add("s3control", "get_public_access_block", bpa, {"AccountId": ACCOUNT})
    c.add("bedrock", "get_inference_profile", _profile(regions),
          {"inferenceProfileIdentifier": "us.anthropic.claude-haiku-5-5"})
    c.add("bedrock", "get_foundation_model_availability", avail, {"modelId": "anthropic.claude-haiku-5-5"})
    for r in logging_regions:
        c.add("bedrock" if r == "us-east-1" else f"bedrock:{r}", "get_model_invocation_logging_configuration",
              (logging or {}).get(r, {}), {})
    if trail:
        _trails(c, [_trail()])
        c.add("cloudtrail", "get_trail_status", {"IsLogging": True}, {"Name": TRAIL_ARN})
        c.add("cloudtrail", "get_event_selectors", {"TrailARN": TRAIL_ARN, "EventSelectors": [
            {"ReadWriteType": "All", "IncludeManagementEvents": True}]}, {"TrailName": TRAIL_ARN})
    else:
        _trails(c, [])


def _trail(arn=TRAIL_ARN, *, multi=True, home="us-west-2", **extra):
    return {"Name": arn.rsplit("/", 1)[1], "TrailARN": arn, "IsMultiRegionTrail": multi, "HomeRegion": home,
            "S3BucketName": "secret-trail-bucket-name",
            "CloudWatchLogsLogGroupArn": f"arn:aws:logs:us-east-1:{ACCOUNT}:log-group:secret-trail-group:*",
            **extra}


def _trails(c, trails):
    c.add("cloudtrail", "describe_trails", {"trailList": trails}, {"includeShadowTrails": True})


def test_all_good_before_deploy(capsys):
    c = Clients()
    _pre(c)
    results = c.run()
    assert all(r.ok for r in results) and preflight.report(results) == 0
    assert "us-east-1, us-east-2, us-west-1, us-west-2" in capsys.readouterr().out


@pytest.mark.parametrize("bpa", [
    {"PublicAccessBlockConfiguration": {**BPA_ON["PublicAccessBlockConfiguration"], "RestrictPublicBuckets": False}},
    "NoSuchPublicAccessBlockConfiguration",
])
def test_block_public_access_must_be_fully_on(bpa, capsys):
    c = Clients()
    _pre(c, bpa=bpa)
    results = c.run()
    bpa_result = results[0]
    assert not bpa_result.ok and "put-public-access-block" in bpa_result.fix
    assert preflight.report(results) == 1
    assert "fix (run it yourself)" in capsys.readouterr().out


def test_profile_region_outside_us_fails():
    c = Clients()
    _pre(c, regions=("us-east-1", "eu-west-1"))
    prof = c.run()[1]
    assert not prof.ok and "eu-west-1" in prof.detail


def test_model_not_authorized_fails():
    c = Clients()
    _pre(c, avail={**AVAILABLE, "authorizationStatus": "NOT_AUTHORIZED"})
    access = c.run()[2]
    assert not access.ok and "Model access" in access.fix


def _own_resources(c, buckets):
    """The Access Analyzer check's own read of the ``OwlBank*`` stacks' resources (2nd pass)."""
    ours = sorted({st for st, _ in buckets if st.startswith("OwlBank")} | {"OwlBankData"})
    c.add("cloudformation", "describe_stacks",
          {"Stacks": [{"StackName": n, "CreationTime": "2026-10-12T00:00:00Z", "StackStatus": "CREATE_COMPLETE"}
                      for n in ours]})
    for stack in ours:
        res = [{"LogicalResourceId": f"L{i}", "PhysicalResourceId": b, "ResourceType": "AWS::S3::Bucket",
                "LastUpdatedTimestamp": "2026-10-12T00:00:00Z", "ResourceStatus": "CREATE_COMPLETE"}
               for i, (st, b) in enumerate(buckets) if st == stack]
        res.append({"LogicalResourceId": "Key", "PhysicalResourceId": "key-1", "ResourceType": "AWS::KMS::Key",
                    "LastUpdatedTimestamp": "2026-10-12T00:00:00Z", "ResourceStatus": "CREATE_COMPLETE"})
        c.add("cloudformation", "list_stack_resources", {"StackResourceSummaries": res}, {"StackName": stack})


def _post(c, findings, *, buckets=None, own_error=None):
    buckets = buckets if buckets is not None else [("OwlBankBridge", "alb-logs-bucket")]
    _stacks_and_buckets(c, buckets)
    for _stack, bucket in buckets:
        _bucket_ok(c, bucket)
    if own_error:
        c.s["cloudformation"].add_client_error("describe_stacks", service_error_code=own_error)
    else:
        _own_resources(c, buckets)
    c.add("accessanalyzer", "list_analyzers",
          {"analyzers": [{"arn": ANALYZER_ARN, "name": "owlbank-dev-account", "type": "ACCOUNT",
                          "createdAt": "2026-10-12T00:00:00Z", "status": "ACTIVE"}]}, {"type": "ACCOUNT"})
    c.add("accessanalyzer", "list_findings", {"findings": findings},
          {"analyzerArn": ANALYZER_ARN, "filter": {"status": {"eq": ["ACTIVE"]}}})


def test_post_deploy_no_findings():
    c = Clients()
    _pre(c)
    _post(c, [])
    results = c.run(post_deploy=True)
    assert results[-1].ok and results[-1].name.endswith("owlbank-dev-account")


def _finding(resource, rtype="AWS::S3::Bucket"):
    return {"id": "f1", "resource": resource, "resourceType": rtype,
            "condition": {}, "createdAt": "2026-10-12T00:00:00Z", "analyzedAt": "2026-10-12T00:00:00Z",
            "updatedAt": "2026-10-12T00:00:00Z", "status": "ACTIVE", "resourceOwnerAccount": ACCOUNT}


def test_post_deploy_active_finding_on_our_resource_fails():
    c = Clients()
    _pre(c)
    _post(c, [_finding("arn:aws:s3:::alb-logs-bucket"), _finding("arn:aws:s3:::someone-elses")])
    last = c.run(post_deploy=True)[-1]
    assert not last.ok and "arn:aws:s3:::alb-logs-bucket" in last.detail and "list-findings" in last.fix
    assert "someone-elses" not in last.detail and "1 more on other resources" in last.detail


def test_post_deploy_finding_on_another_resource_is_a_note_with_the_count():
    c = Clients()
    _pre(c)
    _post(c, [_finding("arn:aws:s3:::leaky"), _finding("arn:aws:iam::111111111111:role/other", "AWS::IAM::Role")])
    results = c.run(post_deploy=True)
    last = results[-1]
    assert last.ok and last.warn and "2 on other resources" in last.detail and "--strict-account" in last.detail
    assert preflight.report(results) == 0


def test_strict_account_fails_on_any_finding():
    c = Clients()
    _pre(c)
    _post(c, [_finding("arn:aws:s3:::leaky")])
    last = c.run(post_deploy=True, strict_account=True)[-1]
    assert not last.ok and "arn:aws:s3:::leaky" in last.detail and "--strict-account" in last.detail


def test_unlistable_stack_resources_report_the_count_unfiltered_with_the_reason():
    c = Clients()
    _pre(c)
    _post(c, [_finding("arn:aws:s3:::leaky"), _finding("arn:aws:s3:::other")], own_error="AccessDenied")
    last = c.run(post_deploy=True)[-1]
    assert not last.ok
    assert "2 ACTIVE finding(s), unfiltered" in last.detail and "AccessDenied" in last.detail


@pytest.mark.parametrize("arn, pid, match", [
    ("arn:aws:s3:::b", "b", True),
    ("arn:aws:iam::1:role/r", "r", True),
    ("arn:aws:lambda:us-east-1:1:function:fn", "fn", True),
    ("arn:aws:kms:us-east-1:1:key/abc", "abc", True),
    ("arn:aws:secretsmanager:us-east-1:1:secret:s-AbCdEf", "arn:aws:secretsmanager:us-east-1:1:secret:s-AbCdEf", True),
    ("arn:aws:s3:::xb", "b", False),
    ("arn:aws:s3:::b", "", False),
])
def test_arn_match_is_equality_or_a_slash_or_colon_tail(arn, pid, match):
    assert preflight.arn_matches(arn, pid) is match


# ---------------------------------------------------------------- Bucket check

def _stacks_and_buckets(c, buckets, *, other_stacks=("Unrelated", "OwlBankData", "CDKToolkit")):
    """Stub DescribeStacks (our stacks, CDKToolkit and an unrelated one) and ListStackResources for each
    matching stack. ``buckets`` is ``[(stack, bucket name)]``."""
    names = sorted({st for st, _ in buckets} | set(other_stacks))
    c.add("cloudformation", "describe_stacks",
          {"Stacks": [{"StackName": n, "CreationTime": "2026-10-12T00:00:00Z", "StackStatus": "CREATE_COMPLETE"}
                      for n in names]})
    for stack in sorted(n for n in names if n.startswith("OwlBank") or n == "CDKToolkit"):
        res = [{"LogicalResourceId": f"L{i}", "PhysicalResourceId": b, "ResourceType": "AWS::S3::Bucket",
                "LastUpdatedTimestamp": "2026-10-12T00:00:00Z", "ResourceStatus": "CREATE_COMPLETE"}
               for i, (st, b) in enumerate(buckets) if st == stack]
        res.append({"LogicalResourceId": "Key", "PhysicalResourceId": "key-1", "ResourceType": "AWS::KMS::Key",
                    "LastUpdatedTimestamp": "2026-10-12T00:00:00Z", "ResourceStatus": "CREATE_COMPLETE"})
        c.add("cloudformation", "list_stack_resources", {"StackResourceSummaries": res}, {"StackName": stack})


OWNER_OK = {"OwnershipControls": {"Rules": [{"ObjectOwnership": "BucketOwnerEnforced"}]}}


def _bucket_ok(c, bucket, *, bpa=None, owner=OWNER_OK, public=False):
    """Stub the three bucket reads. A string for ``bpa``, ``owner`` or ``public`` is an error code."""
    p = {"Bucket": bucket}
    for op, val in (
            ("get_public_access_block", BPA_ON if bpa is None else bpa),
            ("get_bucket_ownership_controls", owner),
            ("get_bucket_policy_status", {"PolicyStatus": {"IsPublic": public}}
             if not isinstance(public, str) else public)):
        if isinstance(val, str):
            c.s["s3"].add_client_error(op, service_error_code=val, expected_params=p)
        else:
            c.add("s3", op, val, p)


def _only_buckets(c, buckets, **kw):
    _stacks_and_buckets(c, buckets)
    for _stack, bucket in buckets:
        _bucket_ok(c, bucket, **kw)
    for s in c.s.values():
        s.activate()
    try:
        return list(preflight.check_buckets(c.c["cloudformation"], c.c["s3"]))
    finally:
        for s in c.s.values():
            s.assert_no_pending_responses()
            s.deactivate()


def test_buckets_are_found_through_stack_resources_not_names():
    c = Clients()
    results = _only_buckets(c, [("CDKToolkit", "cdk-hnb659fds-assets-111111111111-us-east-1"),
                                ("OwlBankBridge", "totally-unrelated-name-1")])
    assert [r.ok for r in results] == [True, True]
    assert {r.name.split()[2] for r in results} == {
        "totally-unrelated-name-1", "cdk-hnb659fds-assets-111111111111-us-east-1"}


@pytest.mark.parametrize("kw, expect", [
    ({"bpa": {"PublicAccessBlockConfiguration": {**BPA_ON["PublicAccessBlockConfiguration"],
                                                 "BlockPublicPolicy": False}}}, "BlockPublicPolicy"),
    ({"bpa": "NoSuchPublicAccessBlockConfiguration"}, "no bucket-level Block Public Access"),
    ({"bpa": "AccessDenied"}, "GetPublicAccessBlock failed: AccessDenied"),
    ({"owner": {"OwnershipControls": {"Rules": [{"ObjectOwnership": "BucketOwnerPreferred"}]}}},
     "BucketOwnerPreferred"),
    ({"owner": {"OwnershipControls": {"Rules": [{"ObjectOwnership": "ObjectWriter"}]}}}, "ObjectWriter"),
    ({"owner": "OwnershipControlsNotFoundError"}, "ACLs enabled"),
    ({"public": True}, "PUBLIC"),
    ({"public": "AccessDenied"}, "GetBucketPolicyStatus failed: AccessDenied"),
])
def test_bucket_that_is_not_private_fails_loudly(kw, expect, capsys):
    results = _only_buckets(Clients(), [("OwlBankBridge", "b1")], **kw)
    (r,) = results
    assert not r.ok and expect in r.detail and "b1" in r.name and r.fix
    assert preflight.report(results) == 1
    assert "[FAIL]" in capsys.readouterr().out


def test_no_policy_at_all_is_not_public():
    (r,) = _only_buckets(Clients(), [("OwlBankBridge", "b1")], public="NoSuchBucketPolicy")
    assert r.ok


def test_one_bad_bucket_fails_the_run_among_good_ones():
    c = Clients()
    _stacks_and_buckets(c, [("CDKToolkit", "bad"), ("OwlBankBridge", "good")])  # stacks are read in name order
    _bucket_ok(c, "bad", owner="OwnershipControlsNotFoundError")
    _bucket_ok(c, "good")
    for s in c.s.values():
        s.activate()
    results = list(preflight.check_buckets(c.c["cloudformation"], c.c["s3"]))
    assert [r.ok for r in results] == [False, True] and preflight.report(results) == 1


def test_no_bucket_found_is_a_failure():
    c = Clients()
    c.add("cloudformation", "describe_stacks", {"Stacks": [
        {"StackName": "Other", "CreationTime": "2026-10-12T00:00:00Z", "StackStatus": "CREATE_COMPLETE"}]})
    for s in c.s.values():
        s.activate()
    (r,) = preflight.check_buckets(c.c["cloudformation"], c.c["s3"])
    assert not r.ok and "no S3 bucket found" in r.detail


def test_cloudformation_error_is_a_failure():
    c = Clients()
    c.s["cloudformation"].add_client_error("describe_stacks", service_error_code="AccessDenied")
    for s in c.s.values():
        s.activate()
    (r,) = preflight.check_buckets(c.c["cloudformation"], c.c["s3"])
    assert not r.ok and "AccessDenied" in r.detail


def test_deleted_resources_are_skipped():
    c = Clients()
    c.add("cloudformation", "describe_stacks", {"Stacks": [
        {"StackName": "OwlBankBridge", "CreationTime": "2026-10-12T00:00:00Z", "StackStatus": "UPDATE_COMPLETE"}]})
    c.add("cloudformation", "list_stack_resources", {"StackResourceSummaries": [
        {"LogicalResourceId": "Old", "PhysicalResourceId": "gone", "ResourceType": "AWS::S3::Bucket",
         "LastUpdatedTimestamp": "2026-10-12T00:00:00Z", "ResourceStatus": "DELETE_COMPLETE"}]},
        {"StackName": "OwlBankBridge"})
    for s in c.s.values():
        s.activate()
    (r,) = preflight.check_buckets(c.c["cloudformation"], c.c["s3"])
    assert not r.ok and "no S3 bucket found" in r.detail


def test_post_deploy_runs_the_bucket_check_before_the_analyzer():
    c = Clients()
    _pre(c)
    _post(c, [], buckets=[("CDKToolkit", "cdk-assets"), ("OwlBankBridge", "alb-logs")])
    results = c.run(post_deploy=True)
    names = [r.name for r in results]
    assert [n for n in names if n.startswith("S3 bucket ")] == [
        "S3 bucket cdk-assets (CDKToolkit)", "S3 bucket alb-logs (OwlBankBridge)"]
    assert names[-1].startswith("IAM Access Analyzer") and all(r.ok for r in results)


def test_post_deploy_without_the_bucket_clients_is_refused():
    c = Clients()
    _pre(c)
    for s in c.s.values():
        s.activate()
    with pytest.raises(ValueError, match="CloudFormation and S3"):
        preflight.run(sts=c.c["sts"], s3control=c.c["s3control"], bedrock=c.c["bedrock"],
                      analyzer=c.c["accessanalyzer"], stage="dev", post_deploy=True)


def test_script_calls_read_only_apis_only():
    """Every boto3 call in the script is one of the named read-only operations."""
    import re
    src = SCRIPT.read_text()
    calls = set(re.findall(r"\.(get_[a-z_]+|list_[a-z_]+|describe_[a-z_]+|put_[a-z_]+|create_[a-z_]+|delete_[a-z_]+|update_[a-z_]+)\(",
                           src))
    calls |= set(re.findall(r'get_paginator\("([a-z_]+)"\)', src))
    calls.discard("get_paginator")
    ops = {"".join(p.capitalize() for p in c.split("_")) for c in calls}
    assert ops <= READ_ONLY_OPS, ops - READ_ONLY_OPS
    assert not any(op.startswith(("Put", "Create", "Delete", "Update")) for op in ops)


# ---------------------------------------------------------------- Prefix list, secrets, Twilio output

def _ec2():
    c = _client("ec2")
    return c, Stubber(c)


def _prefix_lists(ids):
    return {"PrefixLists": [{"PrefixListId": i, "AddressFamily": "IPv4", "State": "create-complete",
                             "PrefixListArn": f"arn:aws:ec2:us-east-1:aws:prefix-list/{i}",
                             "PrefixListName": preflight.CLOUDFRONT_PREFIX_LIST_NAME, "MaxEntries": 55,
                             "Version": 1, "OwnerId": "AWS"} for i in ids]}


def test_prefix_list_matches_or_names_the_right_id():
    for ids, ok in ((["pl-3b927c52"], True), (["pl-aaaaaaaa"], False), ([], False)):
        client, stub = _ec2()
        stub.add_response("describe_managed_prefix_lists", _prefix_lists(ids),
                          {"Filters": [{"Name": "prefix-list-name",
                                        "Values": [preflight.CLOUDFRONT_PREFIX_LIST_NAME]}]})
        with stub:
            r = preflight.check_cloudfront_prefix_list(client, "pl-3b927c52")
        assert r.ok is ok
        if ids and not ok:
            assert "pl-aaaaaaaa" in r.fix and "cloudFrontOriginPrefixListId" in r.fix


def test_run_adds_the_prefix_list_check_only_when_asked():
    c = Clients()
    _pre(c)
    assert len(c.run()) == 5  # BPA, profile, model access, logging, trail: no EC2 client
    c = Clients()
    _pre(c)
    ec2, stub = _ec2()
    stub.add_response("describe_managed_prefix_lists", _prefix_lists(["pl-3b927c52"]),
                      {"Filters": [{"Name": "prefix-list-name", "Values": [preflight.CLOUDFRONT_PREFIX_LIST_NAME]}]})
    with stub:
        results = c.run(ec2=ec2, prefix_list_id="pl-3b927c52")
    assert len(results) == 6 and results[-1].ok


V1, V2 = "1" * 32, "2" * 32


def _secret_desc(name, stages):
    return {"ARN": f"arn:aws:secretsmanager:us-east-1:{ACCOUNT}:secret:{name}-AbCdEf", "Name": name,
            "VersionIdsToStages": stages}


def _describe_all(stages_by_name, extra=None):
    client = _client("secretsmanager")
    stub = Stubber(client)
    for name in preflight.TWILIO_SECRETS:
        stub.add_response("describe_secret", {**_secret_desc(name, stages_by_name[name]), **(extra or {})},
                          {"SecretId": name})
    return client, stub


API, TOKEN, BASIC = preflight.TWILIO_SECRETS
CREATED = {V1: ["AWSCURRENT"]}
WRITTEN = {V1: ["AWSPREVIOUS"], V2: ["AWSCURRENT"]}


def test_secrets_populated_is_judged_from_version_stages_only():
    """A fresh account: the two Twilio secrets written, the Basic one only generated (one version)."""
    client, stub = _describe_all({API: WRITTEN, TOKEN: CREATED, BASIC: CREATED})
    with stub:
        results = preflight.check_secrets_populated(client)
    assert [r.ok for r in results] == [True, False, True]
    assert "populate_secrets.py" in results[1].fix and "--apply" in results[1].fix
    assert "--account-sid" not in results[1].fix
    assert "SecretString" not in str(results)


def test_generated_only_basic_secret_passes():
    """CDK generates basic-auth and nothing writes it, so it has no AWSPREVIOUS."""
    assert BASIC in preflight.GENERATED_SECRETS
    assert not ({API, TOKEN} & preflight.GENERATED_SECRETS)
    client, stub = _describe_all({API: WRITTEN, TOKEN: WRITTEN, BASIC: CREATED})
    with stub:
        results = preflight.check_secrets_populated(client)
    assert all(r.ok for r in results), results


def test_secret_without_a_current_version_or_scheduled_for_deletion_fails():
    client, stub = _describe_all({API: WRITTEN, TOKEN: WRITTEN, BASIC: {V1: ["AWSPENDING"]}})
    with stub:
        assert [r.ok for r in preflight.check_secrets_populated(client)] == [True, True, False]
    client, stub = _describe_all({API: WRITTEN, TOKEN: WRITTEN, BASIC: CREATED},
                                 {"DeletedDate": datetime(2026, 11, 1, tzinfo=UTC)})
    with stub:
        results = preflight.check_secrets_populated(client)
    assert not any(r.ok for r in results) and all("deletion" in r.detail for r in results)


def test_missing_secret_says_to_deploy_data_first():
    client = _client("secretsmanager")
    stub = Stubber(client)
    stub.add_client_error("describe_secret", service_error_code="ResourceNotFoundException")
    with stub:
        (r,) = preflight.check_secrets_populated(client, ("owlbank/twilio/api-key",))
    assert not r.ok and "OwlBankData" in r.detail


CTX = "owlbank/twilio/context-api-key"


def test_context_api_key_must_be_written_when_the_secret_exists():
    for stages, ok in ((CREATED, False), (WRITTEN, True)):
        client = _client("secretsmanager")
        stub = Stubber(client)
        stub.add_response("describe_secret", _secret_desc(CTX, stages), {"SecretId": CTX})
        with stub:
            (r,) = preflight.check_context_api_key_written(client)
        assert r.ok is ok
        if not ok:
            assert "twilio_setup.py --phase pre" in r.fix and "503" in r.detail


def test_context_api_key_absent_secret_is_skipped():
    client = _client("secretsmanager")
    stub = Stubber(client)
    stub.add_client_error("describe_secret", service_error_code="ResourceNotFoundException")
    with stub:
        assert preflight.check_context_api_key_written(client) == []


def test_twilio_context_needs_both_ids(tmp_path):
    p = tmp_path / "twilio.context.json"
    assert not preflight.check_twilio_context(p).ok  # missing
    p.write_text("[]")
    assert not preflight.check_twilio_context(p).ok
    p.write_text('{"verifyServiceSid": "VA1"}')
    r = preflight.check_twilio_context(p)
    assert not r.ok and "conversationConfigurationId" in r.detail and "--phase pre" in r.fix
    assert "--cdk-outputs" in r.fix  # phase A reads the API key secret's ARN from the outputs
    p.write_text('{"verifyServiceSid": "VA1", "conversationConfigurationId": "conv_configuration_1"}')
    assert preflight.check_twilio_context(p).ok


def test_run_with_before_bridge_checks(tmp_path):
    c = Clients()
    _pre(c)
    client, stub = _describe_all({API: WRITTEN, TOKEN: WRITTEN, BASIC: CREATED})
    stub.add_response("describe_secret", _secret_desc("owlbank/twilio/context-api-key", WRITTEN),
                      {"SecretId": "owlbank/twilio/context-api-key"})
    p = tmp_path / "t.json"
    p.write_text('{"verifyServiceSid": "VA1", "conversationConfigurationId": "conv_1"}')
    with stub:
        results = c.run(secrets=client, twilio_context=p)
    assert len(results) == 10 and all(r.ok for r in results)


def test_agentcore_session_quota_defaults_match_config():
    """Preflight stays standalone (no CDK import) but compares against the same
    ``agentcoreSessionQuota`` default that synth uses."""
    from owlbank_infra import config
    ours, synth = preflight.AGENTCORE_SESSION_QUOTA_DEFAULTS, dict(config.AGENTCORE_SESSION_QUOTA_DEFAULTS)
    assert ours == synth
    other = preflight.AGENTCORE_SESSION_QUOTA_OTHER_REGIONS
    assert other == config.AGENTCORE_SESSION_QUOTA_OTHER_REGIONS
    assert config.InfraConfig.__dataclass_fields__["agentcore_session_quota"].default == \
        preflight.default_session_quota("us-east-1")


# ---------------------------------------------------------------- Invocation logging, CloudTrail

SECRET_BUCKET, SECRET_GROUP = "secret-logging-bucket", "secret-logging-group"
TEXT_ON = {"loggingConfig": {
    "s3Config": {"bucketName": SECRET_BUCKET, "keyPrefix": "secret-prefix"},
    "cloudWatchConfig": {"logGroupName": SECRET_GROUP,
                         "roleArn": f"arn:aws:iam::{ACCOUNT}:role/secret-logging-role"},
    "textDataDeliveryEnabled": True, "imageDataDeliveryEnabled": False}}
TEXT_OFF = {"loggingConfig": {**TEXT_ON["loggingConfig"], "textDataDeliveryEnabled": False}}
TEXT_UNSET = {"loggingConfig": {k: v for k, v in TEXT_ON["loggingConfig"].items()
                                if k != "textDataDeliveryEnabled"}}


def _no_secret(r):
    text = f"{r.name} {r.detail} {r.fix}"
    for secret in (SECRET_BUCKET, SECRET_GROUP, "secret-prefix", "secret-logging-role",
                   "secret-trail-bucket-name", "secret-trail-group"):
        assert secret not in text, (secret, text)


def _logging(responses, ceiling="assisted", regions=ROUTED, **kw):
    """``check_invocation_logging`` with one stubbed client per Region; ``responses`` maps a Region
    to its response (a string is an error code). Every stub must be read."""
    clients = {r: _client("bedrock", r) for r in regions}
    stubs = {r: Stubber(c) for r, c in clients.items()}
    for r in regions:
        resp = responses.get(r, {})
        if isinstance(resp, str):
            stubs[r].add_client_error("get_model_invocation_logging_configuration", service_error_code=resp)
        else:
            stubs[r].add_response("get_model_invocation_logging_configuration", resp, {})
        stubs[r].activate()
    try:
        return preflight.check_invocation_logging(clients.get, ceiling, regions, **kw)
    finally:
        for s in stubs.values():
            s.assert_no_pending_responses()
            s.deactivate()


def _status(r):
    return "FAIL" if not r.ok else "WARN" if r.warn else "PASS"


@pytest.mark.parametrize("resp, detail", [({}, "no invocation logging configuration"),
                                          (TEXT_OFF, "textDataDeliveryEnabled is false")])
def test_logging_off_or_text_off_passes(resp, detail):
    results = _logging({"us-east-2": resp})
    assert [r.name for r in results] == [f"Bedrock invocation logging ({r})" for r in ROUTED]
    assert [_status(r) for r in results] == ["PASS"] * 3
    assert detail in results[1].detail
    _no_secret(results[1])


@pytest.mark.parametrize("resp", [TEXT_ON, TEXT_UNSET])
def test_text_delivery_on_fails_naming_the_field_and_the_fix(resp, capsys):
    results = _logging({"us-west-2": resp})
    assert [_status(r) for r in results] == ["PASS", "PASS", "FAIL"]
    bad = results[2]
    assert "textDataDeliveryEnabled" in bad.detail and "us-west-2" in bad.detail
    assert "callers' digits" in bad.detail  # why it matters
    assert "s3Config" in bad.detail and "cloudWatchConfig" in bad.detail  # field names only
    assert "--accept-text-logging" in bad.fix and "--slot-mode-ceiling deterministic" in bad.fix
    assert "whoever enabled it" in bad.fix
    _no_secret(bad)
    assert preflight.report(results) == 1
    out = capsys.readouterr().out
    assert "[FAIL] Bedrock invocation logging (us-west-2)" in out
    assert SECRET_BUCKET not in out and SECRET_GROUP not in out


def test_text_delivery_on_only_warns_with_a_deterministic_ceiling(capsys):
    results = _logging({"us-east-1": TEXT_ON}, ceiling="deterministic")
    assert [_status(r) for r in results] == ["WARN", "PASS", "PASS"]
    assert "slotModeCeiling is deterministic" in results[0].detail
    assert "--accept-text-logging" in results[0].fix and "--slot-mode-ceiling deterministic" in results[0].fix
    _no_secret(results[0])
    assert preflight.report(results) == 0
    assert "[WARN] Bedrock invocation logging (us-east-1)" in capsys.readouterr().out


@pytest.mark.parametrize("ceiling, expect", [("deterministic", "WARN"), ("assisted", "FAIL"),
                                             ("generative", "FAIL"), ("bogus", "FAIL")])
def test_unreadable_configuration_fails_closed(ceiling, expect):
    (r,) = _logging({"us-east-1": "AccessDeniedException"}, ceiling=ceiling, regions=("us-east-1",))
    assert _status(r) == expect
    assert "AccessDeniedException" in r.detail and "bedrock:GetModelInvocationLoggingConfiguration" in r.fix


def test_missing_client_and_empty_regions_fail_closed():
    (r,) = preflight.check_invocation_logging(lambda region: None, "assisted", ("us-east-2",))
    assert not r.ok and "us-east-2" in r.detail
    (r,) = preflight.check_invocation_logging(lambda region: None, "assisted", ())
    assert not r.ok and "no Region" in r.detail


def test_default_logging_regions_are_the_deployment_region_only():
    assert preflight.LOGGING_REGIONS == ()
    assert preflight.logging_regions("us-east-1") == LOGGING_REGIONS
    assert preflight.logging_regions("us-west-2") == ("us-west-2",)
    assert preflight.logging_regions("us-east-1", ("us-west-2",)) == ("us-east-1", "us-west-2")
    assert preflight.logging_regions("us-east-1", ("us-east-1",)) == ("us-east-1",)


def test_accept_text_logging_turns_the_fail_into_a_warn(capsys):
    (r,) = _logging({"us-east-1": TEXT_ON}, regions=("us-east-1",), accept_text_logging=True)
    assert _status(r) == "WARN" and "--accept-text-logging" in r.detail
    assert preflight.report([r]) == 0
    (r,) = _logging({"us-east-1": "AccessDeniedException"}, regions=("us-east-1",),
                    accept_text_logging=True)
    assert _status(r) == "WARN"
    (r,) = _logging({"us-east-1": TEXT_OFF}, regions=("us-east-1",), accept_text_logging=True)
    assert _status(r) == "PASS"


def test_accept_text_logging_is_refused_at_stage_prod(capsys):
    with pytest.raises(ValueError, match="stage prod"):
        Clients().run(stage="prod", accept_text_logging=True)
    with pytest.raises(SystemExit):
        preflight.main(["--stage", "prod", "--accept-text-logging"])
    assert "refused with --stage prod" in capsys.readouterr().err


def test_slot_modes_match_config():
    from owlbank_infra import config
    assert preflight.SLOT_MODES == config.SLOT_MODES
    assert preflight.DEFAULT_SLOT_MODE_CEILING == config.DEFAULT_SLOT_MODE_CEILING


def _trail_check(trails, *, stage="dev", status=None, selectors=None, region="us-east-1"):
    """``check_management_trail`` with stubbed CloudTrail. ``trails`` is DescribeTrails' list (or an
    error code); ``status`` and ``selectors`` map a trail ARN to its response (or an error code). Only
    the trails the check is expected to read get stubs, in order."""
    client = _client("cloudtrail", region)
    stub = Stubber(client)
    if isinstance(trails, str):
        stub.add_client_error("describe_trails", service_error_code=trails)
    else:
        stub.add_response("describe_trails", {"trailList": trails}, {"includeShadowTrails": True})
    for arn, resp in (status or {}).items():
        if isinstance(resp, str):
            stub.add_client_error("get_trail_status", service_error_code=resp, expected_params={"Name": arn})
        else:
            stub.add_response("get_trail_status", resp, {"Name": arn})
        sel = (selectors or {}).get(arn)
        if sel is None:
            continue
        if isinstance(sel, str):
            stub.add_client_error("get_event_selectors", service_error_code=sel,
                                  expected_params={"TrailName": arn})
        else:
            stub.add_response("get_event_selectors", {"TrailARN": arn, **sel}, {"TrailName": arn})
    with stub:
        r = preflight.check_management_trail(client, region, stage)
        stub.assert_no_pending_responses()
    _no_secret(r)
    return r


BASIC_ALL = {"EventSelectors": [{"ReadWriteType": "All", "IncludeManagementEvents": True}]}
ADVANCED_MGMT = {"AdvancedEventSelectors": [{"Name": "mgmt", "FieldSelectors": [
    {"Field": "eventCategory", "Equals": ["Management"]}]}]}
TRAIL2 = f"arn:aws:cloudtrail:us-east-1:{ACCOUNT}:trail/second"


@pytest.mark.parametrize("trail, selectors", [
    (_trail(), BASIC_ALL),                                    # multi-Region, home elsewhere
    (_trail(multi=False, home="us-east-1"), BASIC_ALL),       # single-Region, here
    (_trail(), {"EventSelectors": [{"ReadWriteType": "WriteOnly"}]}),  # IncludeManagementEvents defaults on
    (_trail(), ADVANCED_MGMT),
    (_trail(IsOrganizationTrail=True), {"AdvancedEventSelectors": [{"FieldSelectors": [
        {"Field": "eventCategory", "Equals": ["Management"]}, {"Field": "readOnly", "Equals": ["false"]},
        {"Field": "eventSource", "NotEquals": ["kms.amazonaws.com"]}]}]}),
])
def test_logging_trail_with_management_writes_passes(trail, selectors):
    r = _trail_check([trail], status={TRAIL_ARN: {"IsLogging": True}}, selectors={TRAIL_ARN: selectors},
                     stage="prod")
    assert _status(r) == "PASS" and r.name == "CloudTrail management write events (us-east-1)"


@pytest.mark.parametrize("stage, expect", [("dev", "WARN"), ("test", "WARN"), ("prod", "FAIL")])
def test_no_trail_warns_in_dev_and_test_and_fails_in_prod(stage, expect, capsys):
    r = _trail_check([], stage=stage)
    assert _status(r) == expect
    assert r.detail.startswith("No CloudTrail trail is logging management write events in us-east-1")
    assert "EventBridge never receives PutModelInvocationLoggingConfiguration" in r.detail
    assert "alarm on Bedrock invocation logging can't fire" in r.detail
    assert "aws cloudtrail create-trail" in r.fix and "start-logging" in r.fix
    assert "organization trail" in r.fix and "rerun preflight" in r.fix
    assert preflight.report([r]) == (1 if stage == "prod" else 0)
    assert f"[{expect}] CloudTrail management write events (us-east-1)" in capsys.readouterr().out


@pytest.mark.parametrize("trail, status, selectors, reason", [
    (_trail(), {"IsLogging": False}, None, "1 not logging"),
    (_trail(), {"IsLogging": True}, {"EventSelectors": [{"ReadWriteType": "ReadOnly"}]},
     "1 not recording management write events"),
    (_trail(), {"IsLogging": True}, {"EventSelectors": [{"IncludeManagementEvents": False}]},
     "1 not recording management write events"),
    (_trail(), {"IsLogging": True}, {"EventSelectors": [
        {"ExcludeManagementEventSources": ["bedrock.amazonaws.com"]}]}, "1 not recording management write events"),
    (_trail(), {"IsLogging": True}, {"AdvancedEventSelectors": [{"FieldSelectors": [
        {"Field": "eventCategory", "Equals": ["Management"]}, {"Field": "readOnly", "Equals": ["true"]}]}]},
     "1 not recording management write events"),
    (_trail(), {"IsLogging": True}, {"AdvancedEventSelectors": [{"FieldSelectors": [
        {"Field": "eventCategory", "Equals": ["Data"]},
        {"Field": "resources.type", "Equals": ["AWS::S3::Object"]}]}]}, "1 not recording management write events"),
    (_trail(), "AccessDeniedException", None, "1 unreadable (AccessDeniedException)"),
    (_trail(), {"IsLogging": True}, "TrailNotFoundException", "1 unreadable (TrailNotFoundException)"),
])
def test_trail_that_does_not_cover_writes_is_explained(trail, status, selectors, reason):
    r = _trail_check([trail], status={TRAIL_ARN: status}, selectors={TRAIL_ARN: selectors} if selectors else None)
    assert _status(r) == "WARN" and reason in r.detail


def test_single_region_trail_elsewhere_is_not_read():
    r = _trail_check([_trail(multi=False, home="us-west-2")])
    assert _status(r) == "WARN" and "1 in another Region" in r.detail


def test_second_trail_can_cover_when_the_first_does_not():
    r = _trail_check([_trail(), _trail(TRAIL2)],
                     status={TRAIL_ARN: {"IsLogging": False}, TRAIL2: {"IsLogging": True}},
                     selectors={TRAIL2: ADVANCED_MGMT}, stage="prod")
    assert _status(r) == "PASS"


def test_describe_trails_error_is_reported_by_stage():
    for stage, expect in (("dev", "WARN"), ("prod", "FAIL")):
        r = _trail_check("AccessDeniedException", stage=stage)
        assert _status(r) == expect and "AccessDeniedException" in r.detail
        assert "cloudtrail:DescribeTrails" in r.fix


def test_run_includes_both_checks_and_post_deploy_repeats_them():
    c = Clients()
    _pre(c, logging={"us-east-1": TEXT_ON}, trail=False)
    results = c.run()
    names = [r.name for r in results]
    assert names[3:5] == ["Bedrock invocation logging (us-east-1)",
                          "CloudTrail management write events (us-east-1)"]
    assert [_status(r) for r in results[3:5]] == ["FAIL", "WARN"]
    c = Clients()
    _pre(c, logging={"us-east-1": TEXT_ON}, trail=False)
    _post(c, [])
    results = c.run(post_deploy=True, slot_mode_ceiling="deterministic")
    assert [_status(r) for r in results[3:5]] == ["WARN", "WARN"]
    assert results[-1].name.startswith("IAM Access Analyzer")
    c = Clients()
    _pre(c, logging={"us-east-1": TEXT_ON}, trail=False)
    results = c.run(accept_text_logging=True)
    assert [_status(r) for r in results[3:5]] == ["WARN", "WARN"]


def test_run_reads_the_logging_regions_it_is_given():
    c = Clients()
    _pre(c, logging_regions=("us-east-1", "us-west-2"))  # the deployment Region is always read
    results = c.run(logging_regions=("us-west-2",))
    assert [r.name for r in results if r.name.startswith("Bedrock invocation")] == [
        "Bedrock invocation logging (us-east-1)", "Bedrock invocation logging (us-west-2)"]


class _Session:
    calls: list = []

    def __init__(self, **kw):
        _Session.calls.append(("session", kw))

    def client(self, name, **kw):
        _Session.calls.append((name, kw))
        return f"{name}@{kw.get('region_name', 'default')}"


@pytest.mark.parametrize("argv, ceiling, regions", [
    ([], "assisted", None),
    (["--slot-mode-ceiling", "deterministic"], "deterministic", None),
    (["--logging-regions", "us-west-2, us-east-2"], "assisted", ("us-west-2", "us-east-2")),
])
def test_cli_passes_the_ceiling_regions_and_clients(argv, ceiling, regions, monkeypatch):
    seen = {}
    _Session.calls = []
    monkeypatch.setitem(sys.modules, "boto3", type(sys)("boto3"))
    sys.modules["boto3"].Session = _Session
    monkeypatch.setattr(preflight, "run", lambda **kw: seen.update(kw) or [])
    assert preflight.main(["--no-prefix-list", *argv]) == 0
    assert seen["slot_mode_ceiling"] == ceiling and seen["logging_regions"] == regions
    assert seen["cloudtrail"] == "cloudtrail@default"
    assert seen["bedrock_for_region"]("us-east-1") == seen["bedrock"]  # the deployment Region's client
    assert seen["bedrock_for_region"]("us-west-2") == "bedrock@us-west-2"


def test_cli_refuses_an_unknown_ceiling(monkeypatch):
    monkeypatch.setattr(preflight, "run", lambda **kw: [])
    with pytest.raises(SystemExit) as exc:
        preflight.main(["--slot-mode-ceiling", "bogus"])
    assert exc.value.code == 2


# ---------------------------------------------------------------- 6.6: X-Ray Transaction Search

XRAY_POLICY = {"policyName": "TransactionSearchXRayAccess", "policyScope": "ACCOUNT",
               "policyDocument": json.dumps(preflight.transaction_search_policy("us-east-1", ACCOUNT))}
OTHER_POLICY = {"policyName": "Route53Logs", "policyDocument": json.dumps({
    "Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "route53.amazonaws.com"},
                                            "Action": "logs:PutLogEvents", "Resource": "*"}]})}
ACTIVE = {"Destination": "CloudWatchLogs", "Status": "ACTIVE"}


def _ts(dest=ACTIVE, policies=(XRAY_POLICY,), *, xray_error=None, logs_error=None):
    xray, logs = _client("xray"), _client("logs")
    with Stubber(xray) as xs, Stubber(logs) as ls:
        if xray_error:
            xs.add_client_error("get_trace_segment_destination", service_error_code=xray_error)
        else:
            xs.add_response("get_trace_segment_destination", dest, {})
            if logs_error:
                ls.add_client_error("describe_resource_policies", service_error_code=logs_error)
            else:
                ls.add_response("describe_resource_policies", {"resourcePolicies": list(policies)}, {})
        result = preflight.check_transaction_search(xray, logs, "us-east-1", ACCOUNT)
        xs.assert_no_pending_responses()
        ls.assert_no_pending_responses()
    return result


def _assert_fix_commands(fix):
    policy_cmd, dest_cmd = fix.split(" && ")
    prefix = ("aws logs put-resource-policy --region us-east-1 --policy-name TransactionSearchXRayAccess "
              "--policy-document '")
    assert policy_cmd.startswith(prefix) and policy_cmd.endswith("'")
    doc = json.loads(policy_cmd[len(prefix):-1])
    (st,) = doc["Statement"]
    assert st["Effect"] == "Allow" and st["Principal"] == {"Service": "xray.amazonaws.com"}
    assert st["Action"] == "logs:PutLogEvents"
    assert st["Resource"] == [f"arn:aws:logs:us-east-1:{ACCOUNT}:log-group:aws/spans:*",
                              f"arn:aws:logs:us-east-1:{ACCOUNT}:log-group:/aws/application-signals/data:*"]
    assert st["Condition"] == {"ArnLike": {"aws:SourceArn": f"arn:aws:xray:us-east-1:{ACCOUNT}:*"},
                               "StringEquals": {"aws:SourceAccount": ACCOUNT}}
    assert dest_cmd == "aws xray update-trace-segment-destination --region us-east-1 --destination CloudWatchLogs"


def test_transaction_search_active_with_policy_passes():
    r = _ts(policies=(OTHER_POLICY, XRAY_POLICY))
    assert _status(r) == "PASS" and "account-wide" in r.detail


def test_transaction_search_pending_warns():
    r = _ts({"Destination": "CloudWatchLogs", "Status": "PENDING"})
    assert _status(r) == "WARN" and "within minutes" in r.detail and "account-wide" in r.detail


def test_transaction_search_off_fails_with_both_commands(capsys):
    r = _ts({"Destination": "XRay", "Status": "ACTIVE"}, policies=())
    assert _status(r) == "FAIL" and "XRay" in r.detail and "account-wide" in r.detail
    _assert_fix_commands(r.fix)
    assert preflight.report([r]) == 1
    assert "update-trace-segment-destination" in capsys.readouterr().out


@pytest.mark.parametrize("policies", [(), (OTHER_POLICY,)])
def test_transaction_search_without_the_xray_policy_fails(policies):
    r = _ts(policies=policies)
    assert _status(r) == "FAIL" and "resource policy" in r.detail
    _assert_fix_commands(r.fix)


@pytest.mark.parametrize("kw, op", [({"xray_error": "AccessDeniedException"}, "GetTraceSegmentDestination"),
                                    ({"logs_error": "AccessDeniedException"}, "DescribeResourcePolicies")])
def test_transaction_search_unreadable_fails_closed(kw, op):
    r = _ts(**kw)
    assert _status(r) == "FAIL" and op in r.detail and "AccessDeniedException" in r.detail
    assert "xray:GetTraceSegmentDestination" in r.fix


def _run_ts(**kw):
    c = Clients()
    _pre(c)
    xray, logs = _client("xray"), _client("logs")
    c.c.update(xray=xray, logs=logs)
    c.s.update(xray=Stubber(xray), logs=Stubber(logs))
    return c, c.run(xray=xray, logs=logs, **kw)


def test_run_skips_transaction_search_without_the_flag():
    _, results = _run_ts()  # no stubbed xray/logs responses: any call would raise
    assert not any(r.name.startswith("X-Ray Transaction Search") for r in results)


def test_run_checks_transaction_search_with_the_flag():
    c = Clients()
    _pre(c)
    xray, logs = _client("xray"), _client("logs")
    c.s.update(xray=Stubber(xray), logs=Stubber(logs))
    c.s["xray"].add_response("get_trace_segment_destination", ACTIVE, {})
    c.s["logs"].add_response("describe_resource_policies", {"resourcePolicies": [XRAY_POLICY]}, {})
    results = c.run(xray=xray, logs=logs, call_observability=True)
    assert [_status(r) for r in results if r.name == "X-Ray Transaction Search (us-east-1)"] == ["PASS"]


@pytest.mark.parametrize("argv, expect", [([], None), (["--call-observability"], True)])
def test_cli_creates_the_xray_and_logs_clients_only_with_the_flag(argv, expect, monkeypatch):
    seen = {}
    _Session.calls = []
    monkeypatch.setitem(sys.modules, "boto3", type(sys)("boto3"))
    sys.modules["boto3"].Session = _Session
    monkeypatch.setattr(preflight, "run", lambda **kw: seen.update(kw) or [])
    assert preflight.main(["--no-prefix-list", *argv]) == 0
    created = {name for name, _ in _Session.calls}
    if expect:
        assert seen["call_observability"] is True
        assert (seen["xray"], seen["logs"]) == ("xray@default", "logs@default")
        assert {"xray", "logs"} <= created
    else:
        assert seen["call_observability"] is False and seen["xray"] is None and seen["logs"] is None
        assert not {"xray", "logs"} & created


# ---------------------------------------------------------------- Model access rerun and logging text

NOT_AUTH = {**AVAILABLE, "authorizationStatus": "NOT_AUTHORIZED"}
NOT_AVAILABLE_REGION = {**AVAILABLE, "regionAvailability": "NOT_AVAILABLE"}


class _Res:
    def __init__(self, ids):
        self.model_ids = ids
        self.entries = ids
        self.findings = []


def _run_with_live(monkeypatch, first, used, second=None):
    c = Clients()
    _pre(c, avail=first, logging_regions=())  # the rerun comes before the logging read in the queue
    if second is not None:
        c.add("bedrock", "get_foundation_model_availability", second, {"modelId": "anthropic.claude-haiku-5-5"})
    c.add("bedrock", "get_model_invocation_logging_configuration", {}, {})
    monkeypatch.setattr(preflight, "resolve_list", lambda *a, **k: _Res(used))
    return c.run(bedrock_runtime=object())


def _access(results):
    (r,) = [r for r in results if r.name == "Bedrock model access anthropic.claude-haiku-5-5"]
    return r


def test_failed_access_is_rerun_after_the_live_invocation_used_the_model(monkeypatch):
    r = _access(_run_with_live(monkeypatch, NOT_AUTH, ["us.anthropic.claude-haiku-5-5"], AVAILABLE))
    assert r.ok


def test_a_rerun_that_still_fails_reports_the_second_result(monkeypatch):
    r = _access(_run_with_live(monkeypatch, NOT_AUTH, ["us.anthropic.claude-haiku-5-5"],
                               {**NOT_AUTH, "entitlementAvailability": "NOT_AVAILABLE"}))
    assert not r.ok and "entitlement=NOT_AVAILABLE" in r.detail and "rerun" in r.fix


def test_no_rerun_when_the_live_invocation_used_another_model(monkeypatch):
    r = _access(_run_with_live(monkeypatch, NOT_AUTH, ["us.anthropic.claude-sonnet-5-5"]))
    assert not r.ok and "may clear after the model's first invocation" in r.fix


def test_no_rerun_when_the_region_is_not_available_or_the_first_passed(monkeypatch):
    assert not _access(_run_with_live(monkeypatch, NOT_AVAILABLE_REGION,
                                      ["us.anthropic.claude-haiku-5-5"])).ok
    assert _access(_run_with_live(monkeypatch, AVAILABLE, ["us.anthropic.claude-haiku-5-5"])).ok


def test_both_logging_options_are_named_and_delete_advice_is_gone():
    fix = preflight.LOGGING_FIX
    assert "--accept-text-logging" in fix and "--slot-mode-ceiling deterministic" in fix
    assert "briefing" in fix and "whoever enabled it" in fix
    for text in (SCRIPT.read_text(), (SCRIPT.parents[2] / "scripts" / "check_resolver.py").read_text()):
        assert "delete-model-invocation-logging-configuration" not in text
        assert "put-model-invocation-logging-configuration" not in text
        assert "text delivery off" not in text
