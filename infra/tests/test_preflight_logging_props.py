"""Property tests: preflight's invocation-logging
and CloudTrail checks over generated configurations. Fake clients only; no AWS call.

The trail property generates each selector from a *meaning* (does it record a management write
from ``bedrock.amazonaws.com``?) and renders it in one of several API spellings, so the oracle is
the generated meaning, not a second copy of the check's logic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from botocore.exceptions import ClientError
from hypothesis import given, settings
from hypothesis import strategies as st

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"
spec = importlib.util.spec_from_file_location("preflight_logging_props", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
sys.modules["preflight_logging_props"] = preflight
spec.loader.exec_module(preflight)

REGIONS = ("us-east-1", "us-east-2", "us-west-2")
#: Generated names carry a marker no preflight message contains, so "not in the message" is meaningful.
MARK = "zq7secret"
secret = st.from_regex(r"[a-z0-9-]{4,24}", fullmatch=True).map(lambda s: f"{MARK}{s}")


def _error(code: str, op: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": f"{MARK} detail"}}, op)


def _status(r) -> str:
    return "FAIL" if not r.ok else "WARN" if r.warn else "PASS"


def _text(r) -> str:
    return f"{r.name}\n{r.detail}\n{r.fix}"


# ---------------------------------------------------------------- invocation logging

FLAGS = ("textDataDeliveryEnabled", "imageDataDeliveryEnabled", "embeddingDataDeliveryEnabled",
         "videoDataDeliveryEnabled", "audioDataDeliveryEnabled")


@st.composite
def logging_response(draw):
    """``GetModelInvocationLoggingConfiguration``'s response: no config, or any subset of fields."""
    if draw(st.booleans()):
        return {}
    conf = {f: draw(st.booleans()) for f in draw(st.sets(st.sampled_from(FLAGS)))}
    if draw(st.booleans()):
        conf["s3Config"] = {"bucketName": draw(secret), **({"keyPrefix": draw(secret)} if draw(st.booleans()) else {})}
    if draw(st.booleans()):
        cw = {"logGroupName": draw(secret), "roleArn": f"arn:aws:iam::111111111111:role/{draw(secret)}"}
        if draw(st.booleans()):
            cw["largeDataDeliveryS3Config"] = {"bucketName": draw(secret)}
        conf["cloudWatchConfig"] = cw
    return {"loggingConfig": conf}


class FakeBedrock:
    def __init__(self, resp):
        self.resp = resp

    def get_model_invocation_logging_configuration(self):
        if isinstance(self.resp, str):
            raise _error(self.resp, "GetModelInvocationLoggingConfiguration")
        return self.resp


ceilings = st.one_of(st.sampled_from(("deterministic", "assisted", "generative")), st.text(max_size=12))


@settings(max_examples=300, deadline=None)
@given(regions=st.lists(st.sampled_from(REGIONS), min_size=1, max_size=3, unique=True),
       responses=st.lists(st.one_of(logging_response(), st.sampled_from(("AccessDeniedException",))),
                          min_size=3, max_size=3),
       ceiling=ceilings)
def test_logging_passes_iff_no_config_or_text_off_and_warns_iff_deterministic(regions, responses, ceiling):
    by_region = dict(zip(regions, responses, strict=False))
    results = preflight.check_invocation_logging(lambda r: FakeBedrock(by_region[r]), ceiling, tuple(regions))
    assert [r.name for r in results] == [f"Bedrock invocation logging ({r})" for r in regions]
    for region, r in zip(regions, results, strict=True):
        resp = by_region[region]
        safe = isinstance(resp, dict) and (
            "loggingConfig" not in resp or resp["loggingConfig"].get("textDataDeliveryEnabled") is False)
        expected = "PASS" if safe else "WARN" if ceiling == "deterministic" else "FAIL"
        assert _status(r) == expected, (resp, ceiling, r)
        assert MARK not in _text(r)
        assert region in r.detail
        if expected != "PASS":
            assert r.fix  # every non-PASS gives the fix


# ---------------------------------------------------------------- CloudTrail

BEDROCK = "bedrock.amazonaws.com"


@st.composite
def basic_selector(draw):
    """``(selector, covers)``: a basic event selector and whether it records the write."""
    include = draw(st.sampled_from((True, False, None)))  # None: absent, CloudTrail's default True
    rw = draw(st.sampled_from(("All", "WriteOnly", "ReadOnly", None)))  # None: absent, default All
    excludes = draw(st.sets(st.sampled_from(("kms.amazonaws.com", "rdsdata.amazonaws.com", BEDROCK))))
    sel = {}
    if include is not None:
        sel["IncludeManagementEvents"] = include
    if rw is not None:
        sel["ReadWriteType"] = rw
    if excludes or draw(st.booleans()):
        sel["ExcludeManagementEventSources"] = sorted(excludes)
    if draw(st.booleans()):
        sel["DataResources"] = [{"Type": "AWS::S3::Object", "Values": [f"arn:aws:s3:::{draw(secret)}/"]}]
    covers = include is not False and rw != "ReadOnly" and BEDROCK not in excludes
    return sel, covers


#: Field-selector spellings with whether each lets the write through.
CATEGORY = [(None, False), ({"Field": "eventCategory", "Equals": ["Management"]}, True),
            ({"Field": "eventCategory", "Equals": ["Data"]}, False),
            ({"Field": "eventCategory", "Equals": ["NetworkActivity"]}, False)]
READ_ONLY = [(None, True), ({"Field": "readOnly", "Equals": ["false"]}, True),
             ({"Field": "readOnly", "Equals": ["true"]}, False)]
SOURCE = [(None, True), ({"Field": "eventSource", "NotEquals": ["kms.amazonaws.com"]}, True),
          ({"Field": "eventSource", "NotEquals": [BEDROCK]}, False),
          ({"Field": "eventSource", "NotEquals": ["kms.amazonaws.com", BEDROCK]}, False),
          ({"Field": "eventSource", "Equals": [BEDROCK]}, True),
          ({"Field": "eventSource", "Equals": ["s3.amazonaws.com"]}, False)]
#: A field the check doesn't model never counts as covering (fail closed).
OTHER = [(None, True), ({"Field": "resources.type", "Equals": ["AWS::S3::Object"]}, False)]


@st.composite
def advanced_selector(draw):
    parts = [draw(st.sampled_from(menu)) for menu in (CATEGORY, READ_ONLY, SOURCE, OTHER)]
    fields = [f for f, _ in parts if f is not None]
    fields = draw(st.permutations(fields))
    sel = {"FieldSelectors": list(fields)}
    if draw(st.booleans()):
        sel["Name"] = "selector"
    return sel, all(ok for _, ok in parts)


@st.composite
def trail_world(draw):
    """``(trails, status, selectors, covered)`` for one Region (us-east-1)."""
    n = draw(st.integers(0, 4))
    trails, status, selectors, covered = [], {}, {}, False
    for i in range(n):
        arn = f"arn:aws:cloudtrail:us-west-2:111111111111:trail/t{i}"
        where = draw(st.sampled_from(("multi", "here", "elsewhere")))
        trail = {"Name": f"t{i}", "TrailARN": arn, "IsMultiRegionTrail": where == "multi",
                 "HomeRegion": "us-east-1" if where == "here" else "us-west-2",
                 "S3BucketName": draw(secret),
                 "IsOrganizationTrail": draw(st.booleans())}
        if draw(st.booleans()):
            trail["CloudWatchLogsLogGroupArn"] = f"arn:aws:logs:us-east-1:111111111111:log-group:{draw(secret)}:*"
        trails.append(trail)
        status[arn] = draw(st.sampled_from((True, False, "AccessDeniedException")))
        kind = draw(st.sampled_from(("basic", "advanced", "none", "error")))
        if kind == "basic":
            pairs = draw(st.lists(basic_selector(), max_size=3))
            selectors[arn] = {"EventSelectors": [s for s, _ in pairs]}
            covers = any(c for _, c in pairs)
        elif kind == "advanced":
            pairs = draw(st.lists(advanced_selector(), max_size=3))
            selectors[arn] = {"AdvancedEventSelectors": [s for s, _ in pairs]}
            covers = any(c for _, c in pairs)
        elif kind == "none":
            selectors[arn], covers = {}, False
        else:
            selectors[arn], covers = "TrailNotFoundException", False
        covered = covered or (where != "elsewhere" and status[arn] is True and covers)
    return trails, status, selectors, covered


class FakeCloudTrail:
    def __init__(self, trails, status, selectors):
        self.trails, self.status, self.selectors = trails, status, selectors

    def describe_trails(self, **kw):
        assert kw == {"includeShadowTrails": True}
        if isinstance(self.trails, str):
            raise _error(self.trails, "DescribeTrails")
        return {"trailList": self.trails}

    def get_trail_status(self, *, Name):
        s = self.status[Name]
        if isinstance(s, str):
            raise _error(s, "GetTrailStatus")
        return {"IsLogging": s}

    def get_event_selectors(self, *, TrailName):
        s = self.selectors[TrailName]
        if isinstance(s, str):
            raise _error(s, "GetEventSelectors")
        return {"TrailARN": TrailName, **s}


@settings(max_examples=400, deadline=None)
@given(world=trail_world(), stage=st.sampled_from(("dev", "test", "prod")))
def test_trail_passes_iff_a_logging_trail_records_management_writes_here(world, stage):
    trails, status, selectors, covered = world
    r = preflight.check_management_trail(FakeCloudTrail(trails, status, selectors), "us-east-1", stage)
    expected = "PASS" if covered else "FAIL" if stage == "prod" else "WARN"
    assert _status(r) == expected, (world, r)
    assert MARK not in _text(r)
    if not covered:
        assert "us-east-1" in r.detail and "create-trail" in r.fix


@settings(max_examples=50, deadline=None)
@given(code=st.sampled_from(("AccessDeniedException", "UnrecognizedClientException", "ThrottlingException")),
       stage=st.sampled_from(("dev", "test", "prod")))
def test_unreadable_trails_never_pass(code, stage):
    r = preflight.check_management_trail(FakeCloudTrail(code, {}, {}), "us-east-1", stage)
    assert _status(r) == ("FAIL" if stage == "prod" else "WARN") and code in r.detail
    assert MARK not in _text(r)
