"""The ``/voice/partial`` route, ``earlyDigits`` and the front door's Twilio API key grant."""

from __future__ import annotations

import json
import re

import pytest

from owlbank_infra.app_builder import build_app


@pytest.fixture(scope="module")
def templates(tmp_path_factory):
    app, _ = build_app({}, outdir=str(tmp_path_factory.mktemp("m16")))
    asm = app.synth()
    return {n: asm.get_stack_by_name(n).template for n in ("OwlBankApi", "OwlBankCompute")}


def _of(tpl, rtype):
    return {lid: r for lid, r in tpl["Resources"].items() if r["Type"] == rtype}


def _frontdoor(tpl):
    return next(r for lid, r in _of(tpl, "AWS::Lambda::Function").items() if lid.startswith("FrontdoorFn"))


def test_partial_route_is_wired_like_the_other_voice_routes(templates):
    api = templates["OwlBankApi"]
    resources = {r["Properties"]["PathPart"]: lid for lid, r in _of(api, "AWS::ApiGateway::Resource").items()}
    assert "partial" in resources and "factor" in resources
    methods = _of(api, "AWS::ApiGateway::Method")

    def method_for(part):
        return next(m["Properties"] for m in methods.values()
                    if m["Properties"].get("ResourceId", {}).get("Ref") == resources[part])

    partial, factor = method_for("partial"), method_for("factor")
    assert partial["HttpMethod"] == "POST"
    # Same auth model (no API Gateway authorizer: Basic challenge + X-Twilio-Signature in the
    # handler), same validator and the same Lambda proxy integration as /voice/factor.
    for k in ("AuthorizationType", "RequestValidatorId", "Integration"):
        assert partial[k] == factor[k], k


def _strings(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)
    elif isinstance(node, str):
        yield node


def test_waf_allow_list_admits_partial(templates):
    acl = next(iter(_of(templates["OwlBankApi"], "AWS::WAFv2::WebACL").values()))
    allowed = [s for s in _strings(acl) if "voice" in s and "incoming" in s]
    assert allowed
    regex = re.compile(allowed[0])
    assert regex.match("/v1/voice/partial") and regex.match("/voice/partial")
    assert regex.match("/v1/voice/factor")
    assert not regex.match("/v1/voice/partials")


@pytest.mark.parametrize("ctx,want", [({}, "true"), ({"earlyDigits": "false"}, "false"),
                                      ({"earlyDigits": "true"}, "true")])
def test_early_digits_env(tmp_path, ctx, want):
    app, _ = build_app(ctx, outdir=str(tmp_path))
    tpl = app.synth().get_stack_by_name("OwlBankCompute").template
    assert _frontdoor(tpl)["Properties"]["Environment"]["Variables"]["OWLBANK_EARLY_DIGITS"] == want


def _role_statements(tpl, role_prefix):
    out = []
    for r in _of(tpl, "AWS::IAM::Policy").values():
        roles = [x.get("Ref", "") for x in r["Properties"].get("Roles", [])]
        if any(x.startswith(role_prefix) for x in roles):
            out += r["Properties"]["PolicyDocument"]["Statement"]
    return out


def test_frontdoor_reads_the_twilio_api_key_secret_and_nothing_wider(templates):
    """GetSecretValue on named secret ARNs only (no wildcard), the Twilio API key (the one
    auth_service uses) among them, and kms:Decrypt only via Secrets Manager."""
    compute = templates["OwlBankCompute"]
    stmts = _role_statements(compute, "FrontdoorRole")
    reads = [s for s in stmts if "secretsmanager:GetSecretValue" in json.dumps(s.get("Action"))]
    assert reads
    resources = []
    for s in reads:
        resources += s["Resource"] if isinstance(s["Resource"], list) else [s["Resource"]]
    assert all(r != "*" for r in resources)
    env = _frontdoor(compute)["Properties"]["Environment"]["Variables"]
    want = json.dumps(env["OWLBANK_SECRET_TWILIO_API_KEY"], sort_keys=True)
    assert want in [json.dumps(r, sort_keys=True) for r in resources]
    assert len(resources) == 3  # twilio_auth_token, basic_auth, twilio_api_key
    auth_env = next(r for lid, r in _of(compute, "AWS::Lambda::Function").items()
                    if lid.startswith("AuthServiceFn"))["Properties"]["Environment"]["Variables"]
    assert auth_env["OWLBANK_SECRET_TWILIO_API_KEY"] == env["OWLBANK_SECRET_TWILIO_API_KEY"]
    decrypt = [s for s in stmts if s.get("Sid") == "DecryptSecretsViaSecretsManager"]
    assert decrypt and "kms:ViaService" in json.dumps(decrypt[0]["Condition"])
