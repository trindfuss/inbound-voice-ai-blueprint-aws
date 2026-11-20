"""Property tests: preflight's Access Analyzer split and the model-access rerun.

Fake clients only; no AWS call.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"
spec = importlib.util.spec_from_file_location("preflight_fdf_props", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
sys.modules["preflight_fdf_props"] = preflight
spec.loader.exec_module(preflight)

NAME = st.from_regex(r"[a-z][a-z0-9-]{2,12}", fullmatch=True)
#: A resource ARN for an ID, in the spellings the analyzer uses (``/<id>`` and ``:<id>`` tails).
SPELLINGS = ("arn:aws:s3:::{}", "arn:aws:iam::111111111111:role/{}", "arn:aws:lambda:us-east-1:1:function:{}",
             "arn:aws:kms:us-east-1:1:key/{}")


class FakeAnalyzer:
    def __init__(self, findings):
        self.findings = findings

    def get_paginator(self, op):
        outer = self

        class P:
            def paginate(self, **kw):
                if op == "list_analyzers":
                    return [{"analyzers": [{"arn": "arn:analyzer", "name": "a"}]}]
                return [{"findings": outer.findings}]
        return P()


@st.composite
def scenario(draw):
    ids = draw(st.lists(NAME, min_size=0, max_size=6, unique=True))
    own = [{"resource": draw(st.sampled_from(SPELLINGS)).format(i)} for i in ids]
    # Other resources get IDs that don't end an own ID's spelling (a prefix marker no own ID has).
    others = [{"resource": draw(st.sampled_from(SPELLINGS)).format("zz9" + o)}
              for o in draw(st.lists(NAME, max_size=6, unique=True))]
    findings = own + others
    order = draw(st.permutations(range(len(findings))))
    return set(ids), [findings[i] for i in order], len(own), len(others)


@settings(max_examples=200, deadline=None)
@given(scenario(), st.booleans())
def test_split_has_no_overlap_and_failure_iff_own_or_strict(sc, strict):
    """Own and other partition the findings; fail iff own is non-empty, or strict and any."""
    ids, findings, n_own, n_other = sc
    own, other = preflight.split_findings(findings, ids)
    assert len(own) == n_own and len(other) == n_other
    assert not any(a is b for a in own for b in other)
    assert len(own) + len(other) == len(findings)
    r = preflight.check_access_analyzer(FakeAnalyzer(findings), "a", ids, strict=strict)
    assert (not r.ok) == (bool(own) or (strict and bool(findings)))


@settings(max_examples=100, deadline=None)
@given(scenario())
def test_p10_unscoped_fails_on_any_finding_and_never_hides_the_count(sc):
    _ids, findings, _a, _b = sc
    r = preflight.check_access_analyzer(FakeAnalyzer(findings), "a", None, unscoped_reason="AccessDenied")
    assert (not r.ok) == bool(findings)
    if findings:
        assert f"{len(findings)} ACTIVE finding(s), unfiltered" in r.detail and "AccessDenied" in r.detail


# ------------------------------------------------------------------ model-access rerun

PROFILE = "us.anthropic.claude-haiku-5-5"
AUTH = st.sampled_from(["AUTHORIZED", "NOT_AUTHORIZED"])
AVAIL = st.sampled_from(["AVAILABLE", "NOT_AVAILABLE"])
STATUS = st.fixed_dictionaries({"authorizationStatus": AUTH, "entitlementAvailability": AVAIL,
                                "regionAvailability": AVAIL,
                                "agreementAvailability": st.fixed_dictionaries({"status": AVAIL})})


class FakeBedrock:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0

    def get_foundation_model_availability(self, modelId):
        self.calls += 1
        return self.answers.pop(0)


@settings(max_examples=200, deadline=None)
@given(STATUS, STATUS, st.booleans())
def test_p14_rerun_never_turns_pass_into_fail_and_only_follows_a_clearable_failure(a1, a2, used):
    first = preflight.check_model_access(FakeBedrock(a1), PROFILE)
    bedrock = FakeBedrock(a2)
    out = preflight.recheck_model_access(first, bedrock, PROFILE, [PROFILE] if used else [])
    if first.ok:
        assert out is first and bedrock.calls == 0   # a PASS stands, with no second read
    elif used and first.clearable:
        assert bedrock.calls == 1 and out.ok == preflight.check_model_access(FakeBedrock(a2), PROFILE).ok
    else:
        assert out is first and bedrock.calls == 0   # not clearable, or not used: the first stands
    assert (not out.ok) <= (not first.ok)
