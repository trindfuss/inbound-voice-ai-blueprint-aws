"""Synth works offline, cdk-nag (AwsSolutions) runs and reports no unacknowledged errors.

The synth, lookup and nag checks run over ``synth_matrix`` (conftest ``MATRIX``): the four Fargate
context sets (``s3PublicAccessBlockMode=account`` among them) and ``agentcore`` x {vpc, public}
x {dev, prod-with-ack}.
"""

from __future__ import annotations

import json

import pytest

from owlbank_infra.app_builder import build_app
from owlbank_infra.config import ConfigError
from owlbank_infra.nag import ACKS

#: The bridge image repository is in the app on both hostings.
STACKS = {"OwlBankData", "OwlBankNetwork", "OwlBankCompute", "OwlBankAgent", "OwlBankBridge", "OwlBankApi",
          "OwlBankBridgeImage"}


def test_synth_produces_all_stacks(synth_matrix):
    cfg = synth_matrix.cfg
    # agentcore public mode builds no VPC, so no Network stack.
    no_vpc = cfg.is_agentcore and cfg.agentcore_network_mode == "public"
    assert set(synth_matrix.templates) == STACKS - ({"OwlBankNetwork"} if no_vpc else set())


def test_env_is_fake_account_and_region(synth_matrix):
    for stack in synth_matrix.stacks.values():
        assert stack.account == "111111111111" and stack.region == "us-east-1"


def test_no_context_lookups(synth_matrix):
    manifest = json.loads((synth_matrix.outdir / "manifest.json").read_text())
    assert not manifest.get("missing"), manifest.get("missing")


def _nag_report(synth):
    path = synth.outdir / "validation-report.json"
    if not path.exists():
        return None
    # On success cdk-nag may leave no AwsSolutions entry at all.
    return next((p for p in json.loads(path.read_text())["pluginReports"]
                 if p["pluginName"] == "AwsSolutions"), None)


def test_cdk_nag_no_unacknowledged_errors(synth_matrix):
    """Every context, ``s3PublicAccessBlockMode=account`` and the agentcore ones included."""
    report = _nag_report(synth_matrix)
    if report is not None:
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [(v["ruleName"], v.get("violatingResources")) for v in errors]
        assert report["conclusion"] == "success"


def test_every_acknowledgement_has_a_reason(synth_matrix):
    """The acknowledgements this context's app made: each an AwsSolutions rule with a written
    reason, and one per construct and rule."""
    acks = synth_matrix.acks
    assert acks and set(acks) <= set(ACKS)
    for path, rule, reason in acks:
        assert rule.startswith("AwsSolutions-"), rule
        assert len(reason.strip()) >= 40, (path, rule)
    assert len({(path, rule) for path, rule, _ in acks}) == len(acks)


def test_cdk_nag_is_active(tmp_path):
    """An unacknowledged violation must fail synth: proves the plugin really runs."""
    from aws_cdk import aws_sqs as sqs

    app, stacks = build_app({}, outdir=str(tmp_path))
    sqs.Queue(stacks["data"], "UnencryptedNoSslQueue", encryption=sqs.QueueEncryption.UNENCRYPTED)
    with pytest.raises(Exception, match="ValidationFailed"):
        app.synth()


@pytest.mark.parametrize("ctx,msg", [
    ({"domainName": "owlbank.example.com"}, "certificateArn"),
    ({"stage": "prod", "developerPhone": "+12025550100"}, "dev and test"),
    ({"developerPhone": "2025550100"}, "E.164"),
    ({"accountId": "123"}, "12 digits"),
    ({"bridgeDesiredCount": 99}, "between"),
])
def test_bad_context_is_rejected(tmp_path, ctx, msg):
    with pytest.raises(ConfigError, match=msg):
        build_app(ctx, outdir=str(tmp_path))


def test_twilio_outputs_reject_secrets(tmp_path):
    p = tmp_path / "tw.json"
    p.write_text(json.dumps({"authToken": "x"}))
    with pytest.raises(ConfigError, match="secret"):
        build_app({"twilioOutputs": str(p)}, outdir=str(tmp_path / "out"))


def test_owned_numbers_output_only_in_dev(synth_domain, synth_default):
    out = synth_domain.templates["OwlBankData"]["Outputs"]
    assert out["SeedOwnedNumbers"]["Value"] == "+12025550100,+12025550111,+12025550112"
    assert "SeedOwnedNumbers" not in synth_default.templates["OwlBankData"]["Outputs"]


def test_script_contract_outputs(synth_domain, synth_default):
    """Output names read by scripts/seed.py, populate_secrets.py and twilio_setup.py."""
    data = synth_default.templates["OwlBankData"]["Outputs"]
    for name in ("TableCustomers", "TableTokens", "TableCalls", "TableSegments", "TablePolicy",
                 "TableCatalog", "SecretTwilioApiKey", "SecretTwilioAuthToken", "SecretBasicAuth"):
        assert name in data, name
    assert synth_domain.templates["OwlBankApi"]["Outputs"]["ApiDomain"]["Value"] == "voice.owlbank.example.com"
    assert synth_domain.templates["OwlBankBridge"]["Outputs"]["RelayDomain"]["Value"] == "relay.owlbank.example.com"


def test_bridge_desired_count_parameter(synth_domain, synth_default):
    def count(s):
        return next(r for _, _, r in s.resources("AWS::ECS::Service"))["Properties"]["DesiredCount"]
    assert count(synth_default) == 1 and count(synth_domain) == 2


def test_twilio_outputs_reach_frontdoor_and_bridge(synth_domain):
    fd = next(r for _, lid, r in synth_domain.resources("AWS::Lambda::Function") if lid.startswith("FrontdoorFn"))
    env = fd["Properties"]["Environment"]["Variables"]
    assert env["OWLBANK_CONVERSATION_CONFIGURATION_ID"] == "conv_configuration_01"
    # The relay host reaches the front door through the Bridge's SSM parameter, not the env.
    assert "OWLBANK_RELAY_WS_URL" not in env and env["OWLBANK_RELAY_PARAMETER"] == "/owlbank/dev/bridge/relay"
    (param,) = [r for _, _, r in synth_domain.resources("AWS::SSM::Parameter")]
    assert param["Properties"]["Value"] == "relay.owlbank.example.com"
    auth = next(r for _, lid, r in synth_domain.resources("AWS::Lambda::Function") if lid.startswith("AuthServiceFn"))
    assert auth["Properties"]["Environment"]["Variables"]["OWLBANK_VERIFY_SERVICE_SID"] == "VA" + "0" * 32
    td = next(r for _, _, r in synth_domain.resources("AWS::ECS::TaskDefinition"))
    benv = {e["Name"]: e["Value"] for e in td["Properties"]["ContainerDefinitions"][0]["Environment"]}
    assert benv["TWILIO_CONVERSATION_CONFIGURATION_ID"] == "conv_configuration_01"
    assert benv["TWILIO_VOICE_PUBLIC_DOMAIN"] == "relay.owlbank.example.com"
    assert benv["TWILIO_PHONE_NUMBER"] == "+12025550120"
    assert benv["TAC_ANALYTICS_DISABLED"] == "true"


def test_alb_idle_timeout(synth_default):
    alb = next(r for _, _, r in synth_default.resources("AWS::ElasticLoadBalancingV2::LoadBalancer"))
    attrs = {a["Key"]: a["Value"] for a in alb["Properties"]["LoadBalancerAttributes"]}
    assert attrs["idle_timeout.timeout_seconds"] == "900"
    assert attrs["routing.http.drop_invalid_header_fields.enabled"] == "true"
    assert attrs["access_logs.s3.enabled"] == "true"


def test_lambda_bundle_contents(synth_default):
    from owlbank_infra.compute_stack import LAMBDA_SRC_DIR
    for rel in ("owlbank/store.py", "lambdas/frontdoor/handler.py", "lambdas/auth_service/handler.py",
                "scripts/hash_tool.py", "scripts/__init__.py", "data/faq.json"):
        assert (LAMBDA_SRC_DIR / rel).is_file(), rel
    assert not (LAMBDA_SRC_DIR / "tac_bridge").exists()
    assert not list(LAMBDA_SRC_DIR.rglob("__pycache__"))
    for _, lid, r in synth_default.resources("AWS::Lambda::Function"):
        assert r["Properties"]["Handler"].startswith("lambdas.") and r["Properties"]["Runtime"] == "python3.12"
