"""AgentCore hosting: ``POST /voice/orchestrator`` on the live API.

* ``agentcore``: ``POST /voice/orchestrator`` -> the front door (the same Lambda proxy integration,
  request validator and no-authorizer method settings as the voice routes), on the same stage
  (throttling, access and execution logs unchanged).
* WAF: the path is on the allow-list (POST only); the Basic-challenge rule covers it (401 +
  ``WWW-Authenticate`` from WAF, since API Gateway REST renames a Lambda's header); it is the one
  large-body path: 64 KB cap there with API Gateway's body inspection raised to 64 KB, 8 KB everywhere
  else, and the CRS's 8 KB body rule kept on every other path.
* ``fargate``: no such route, no large-body rules, no inspection change (the Fargate snapshot test covers the
  rest).
* No Function URL; cdk-nag clean.

No PBT: IaC, checked by assertions.
"""

from __future__ import annotations

import re

import pytest
from test_bridge_agentcore import _ctx, _quiet_synth
from test_synth_and_nag import _nag_report

from owlbank_infra.api_stack import MAX_BODY_BYTES, ORCHESTRATOR_MAX_BODY_BYTES, VOICE_ROUTES
from owlbank_infra.waf import BASIC_CHALLENGE_HEADER, BASIC_CHALLENGE_STATUS

API = "OwlBankApi"
PATHS = ("/v1/voice/orchestrator", "/voice/orchestrator")


@pytest.fixture(scope="module", params=["dev", "prod"])
def ac(request, tmp_path_factory):
    synth = _quiet_synth(tmp_path_factory, f"api-orch-{request.param}", _ctx("vpc", request.param))
    synth.stage = request.param  # type: ignore[attr-defined]
    return synth


@pytest.fixture
def fargate(ac, request):
    return request.getfixturevalue("synth_prod" if ac.stage == "prod" else "synth_default")


@pytest.fixture(params=["default", "prod"])
def fargate_any(request):
    return request.getfixturevalue(f"synth_{request.param}")


def _res(synth, rtype):
    return {lid: r for lid, r in synth.templates[API]["Resources"].items() if r["Type"] == rtype}


def _resource_lid(synth, part):
    (lid,) = [lid for lid, r in _res(synth, "AWS::ApiGateway::Resource").items()
              if r["Properties"]["PathPart"] == part]
    return lid


def _post(synth, part):
    lid = _resource_lid(synth, part)
    (m,) = [r["Properties"] for r in _res(synth, "AWS::ApiGateway::Method").values()
            if r["Properties"]["ResourceId"] == {"Ref": lid} and r["Properties"]["HttpMethod"] == "POST"]
    return m


def _acl(synth):
    (acl,) = _res(synth, "AWS::WAFv2::WebACL").values()
    return acl["Properties"]


def _rule(acl, name):
    (rule,) = [r for r in acl["Rules"] if r["Name"] == name]
    return rule


def _regexes(stmt):
    """Every ``uri_path`` regex under ``stmt``."""
    if isinstance(stmt, dict):
        if "RegexMatchStatement" in stmt and "UriPath" in stmt["RegexMatchStatement"]["FieldToMatch"]:
            yield stmt["RegexMatchStatement"]["RegexString"]
        for v in stmt.values():
            yield from _regexes(v)
    elif isinstance(stmt, list):
        for v in stmt:
            yield from _regexes(v)


def _allowed(acl):
    """(path regex, methods) pairs of the allow-list rule."""
    inner = _rule(acl, "AllowedPathsAndMethodsOnly")["Statement"]["NotStatement"]["Statement"]
    pairs = inner["OrStatement"]["Statements"] if "OrStatement" in inner else [inner]
    out = []
    for p in pairs:
        path, method = p["AndStatement"]["Statements"]
        ms = method["OrStatement"]["Statements"] if "OrStatement" in method else [method]
        out.append((path["RegexMatchStatement"]["RegexString"],
                    {m["ByteMatchStatement"]["SearchString"] for m in ms}))
    return out


def _allowed_methods(acl, path):
    return set().union(*[ms for rx, ms in _allowed(acl) if re.search(rx, path)])


# --- the route ----------------------------------------------------------------------------------

def test_route_under_voice_with_the_voice_routes_method_settings(ac):
    voice = _resource_lid(ac, "voice")
    orch = _res(ac, "AWS::ApiGateway::Resource")[_resource_lid(ac, "orchestrator")]["Properties"]
    assert orch["ParentId"] == {"Ref": voice}
    m, incoming = _post(ac, "orchestrator"), _post(ac, "incoming")
    assert m["AuthorizationType"] == "NONE" and "AuthorizerId" not in m
    for key in ("Integration", "RequestValidatorId", "AuthorizationType"):
        assert m[key] == incoming[key], key
    assert m["Integration"]["Type"] == "AWS_PROXY"


def test_front_door_may_be_invoked_on_the_route(ac):
    perms = [r["Properties"] for r in _res(ac, "AWS::Lambda::Permission").values()
             if "/voice/orchestrator" in str(r["Properties"].get("SourceArn"))]
    assert perms and all(p["Action"] == "lambda:InvokeFunction"
                         and p["Principal"] == "apigateway.amazonaws.com" for p in perms)
    fn = _post(ac, "incoming")["Integration"]["Uri"]
    assert all(str(p["FunctionName"]) in str(fn) for p in perms)


def test_same_stage_throttling_and_logging_as_fargate(ac, fargate):
    (stage,) = _res(ac, "AWS::ApiGateway::Stage").values()
    (f_stage,) = _res(fargate, "AWS::ApiGateway::Stage").values()
    for key in ("MethodSettings", "AccessLogSetting", "TracingEnabled", "StageName"):
        assert stage["Properties"][key] == f_stage["Properties"][key], key


def test_only_the_route_is_added_to_the_api(ac, fargate):
    def parts(s):
        return sorted(r["Properties"]["PathPart"] for r in _res(s, "AWS::ApiGateway::Resource").values())
    assert parts(ac) == sorted([*parts(fargate), "orchestrator"])
    assert len(_res(ac, "AWS::ApiGateway::Method")) == len(_res(fargate, "AWS::ApiGateway::Method")) + 1


def test_no_function_url_anywhere(ac):
    assert not list(ac.resources("AWS::Lambda::Url"))


# --- WAF ----------------------------------------------------------------------------------------

def test_waf_allows_post_only_on_the_route(ac):
    acl = _acl(ac)
    for path in PATHS:
        assert _allowed_methods(acl, path) == {"POST"}
    assert not _allowed_methods(acl, "/v1/voice/orchestrator/x")


def test_waf_basic_challenge_covers_the_route(ac):
    rule = _rule(_acl(ac), "AuthorizationHeaderRequired")
    (rx,) = list(_regexes(rule["Statement"]))
    assert all(re.search(rx, p) for p in PATHS)
    resp = rule["Action"]["Block"]["CustomResponse"]
    assert resp["ResponseCode"] == BASIC_CHALLENGE_STATUS
    assert resp["ResponseHeaders"] == [{"Name": "WWW-Authenticate", "Value": BASIC_CHALLENGE_HEADER}]


def test_waf_large_body_cap_on_the_route_only(ac):
    acl = _acl(ac)
    large = _rule(acl, "LargeBodySizeLimit")["Statement"]["AndStatement"]["Statements"]
    sizes = [s["SizeConstraintStatement"]["Size"] for s in large if "SizeConstraintStatement" in s]
    assert sizes == [ORCHESTRATOR_MAX_BODY_BYTES] == [65536]
    (rx,) = list(_regexes(large))
    assert all(re.search(rx, p) for p in PATHS)
    assert not any(re.search(rx, f"/v1/voice/{r}") for r in VOICE_ROUTES)
    assert not re.search(rx, "/v1/context/abc")
    # Everywhere else: the 8 KB cap, and the CRS's 8 KB body rule via its label.
    small = _rule(acl, "BodySizeLimit")["Statement"]["AndStatement"]["Statements"]
    assert small[0]["SizeConstraintStatement"]["Size"] == MAX_BODY_BYTES == 8192
    assert list(_regexes(small[1]["NotStatement"])) == [rx]
    assert list(_regexes(_rule(acl, "OversizeBodyOutsideLargeBodyPaths")["Statement"])) == [rx]
    assert acl["AssociationConfig"] == {"RequestBody": {"API_GATEWAY": {"DefaultSizeInspectionLimit": "KB_64"}}}


def test_waf_otherwise_as_in_fargate(ac, fargate):
    added = {"LargeBodySizeLimit", "OversizeBodyOutsideLargeBodyPaths"}
    changed = {"AllowedPathsAndMethodsOnly", "BodySizeLimit", "AWSManagedRulesCommonRuleSet"}
    on, off = ({r["Name"]: r for r in _acl(s)["Rules"]} for s in (ac, fargate))
    assert set(on) == set(off) | added
    assert {n: on[n] for n in set(off) - changed} == {n: off[n] for n in set(off) - changed}
    assert {k: v for k, v in _acl(ac).items() if k not in ("Rules", "AssociationConfig")} == \
        {k: v for k, v in _acl(fargate).items() if k != "Rules"}


# --- fargate ------------------------------------------------------------------------------------

def test_fargate_has_no_route_and_no_large_body_scope(fargate_any):
    parts = {r["Properties"]["PathPart"] for r in _res(fargate_any, "AWS::ApiGateway::Resource").values()}
    assert "orchestrator" not in parts
    acl = _acl(fargate_any)
    names = {r["Name"] for r in acl["Rules"]}
    assert not names & {"LargeBodySizeLimit", "OversizeBodyOutsideLargeBodyPaths"}
    assert "AssociationConfig" not in acl
    assert all(not _allowed_methods(acl, p) for p in PATHS)


def test_cdk_nag_clean(ac):
    report = _nag_report(ac)
    if report is not None:
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [(v["ruleName"], v.get("violatingResources")) for v in errors]
        assert report["conclusion"] == "success"
