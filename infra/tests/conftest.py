"""Synthesize the app once per configuration, offline, and hand templates to the tests.

No AWS credentials are needed or used: the account is a fake default, nothing is
looked up, and nothing is deployed.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

INFRA = Path(__file__).resolve().parents[1]
REPO = INFRA.parent
if str(INFRA) not in sys.path:
    sys.path.insert(0, str(INFRA))

import owlbank_infra.node_bootstrap  # noqa: E402,F401  (before aws_cdk)

# Make sure no ambient credentials or profile could be picked up.
for var in ("AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
    os.environ.pop(var, None)

from owlbank_infra.app_builder import build_app  # noqa: E402
from owlbank_infra.nag import ACKS  # noqa: E402

FAKE_CERT = "arn:aws:acm:us-east-1:111111111111:certificate/11111111-2222-3333-4444-555555555555"
VERIFY_SID = "VA" + "0" * 32


@dataclass
class Synth:
    templates: dict[str, dict]
    stacks: dict
    outdir: Path
    #: The cdk-nag acknowledgements made while building this app (``nag.ACKS`` is process-wide).
    acks: list = field(default_factory=list)

    @property
    def cfg(self):
        """The app's ``InfraConfig`` (hosting, network mode, stage)."""
        return self.stacks["compute"].cfg

    @property
    def domain_mode(self) -> bool:
        """True in the custom-domain mode (``domainName`` set); False in the CloudFront mode."""
        return bool(self.stacks["data"].node.try_get_context("domainName"))

    @property
    def bpa_mode(self) -> str:
        """``s3PublicAccessBlockMode`` of this synth (``explicit`` unless the context says ``account``)."""
        return self.stacks["data"].node.try_get_context("s3PublicAccessBlockMode") or "explicit"

    def resources(self, rtype: str | None = None):
        """Yield (stack, logical id, resource) over all templates."""
        for stack, tpl in self.templates.items():
            for lid, res in tpl.get("Resources", {}).items():
                if rtype is None or res["Type"] == rtype:
                    yield stack, lid, res


def _synth(tmp_path_factory, name: str, context: dict) -> Synth:
    """Synthesize ``context``. These fixtures test the call path, so the optional admin console
    (``OwlBankAdmin``, on by default in dev) is left out unless ``context`` names
    ``adminConsole``; tests/test_admin_stack.py synthesizes it on its own."""
    context = {"adminConsole": "false", **context}
    outdir = tmp_path_factory.mktemp(name)
    first_ack = len(ACKS)
    app, stacks = build_app(context, outdir=str(outdir))
    assembly = app.synth()
    templates = {s.stack_name: s.template for s in assembly.stacks}
    return Synth(templates=templates, stacks=stacks, outdir=outdir, acks=ACKS[first_ack:])


@pytest.fixture(scope="session")
def synth_default(tmp_path_factory) -> Synth:
    """Defaults only: no domain (so the CloudFront mode), fake account."""
    return _synth(tmp_path_factory, "default", {})


@pytest.fixture(scope="session")
def synth_cloudfront(synth_default) -> Synth:
    """The CloudFront mode (no ``domainName``): the default configuration."""
    return synth_default


def flex_outputs(tmp_path_factory) -> str:
    """A twilioOutputs file in Flex mode (both ``workspaceSid`` and ``workflowSid``)."""
    path = tmp_path_factory.mktemp("tw-flex") / "twilio.context.json"
    path.write_text(json.dumps({
        "accountSid": "AC" + "1" * 32, "conversationConfigurationId": "conv_configuration_01",
        "workflowSid": "WW" + "2" * 32, "workspaceSid": "WS" + "4" * 32,
        "queueSids": {"general": "WQ" + "3" * 32},
    }))
    return str(path)


@pytest.fixture(scope="session")
def synth_flex(tmp_path_factory) -> Synth:
    """The defaults plus a Flex outputs file: human leg, sweep, transcription and TaskRouter routes on."""
    return _synth(tmp_path_factory, "flex", {"twilioOutputs": flex_outputs(tmp_path_factory)})


@pytest.fixture(scope="session")
def synth_prod_flex(tmp_path_factory) -> Synth:
    """``synth_prod`` plus a Flex outputs file."""
    return _synth(tmp_path_factory, "prod-flex", {
        "stage": "prod", "domainName": "owlbank.example.com", "certificateArn": FAKE_CERT,
        "bridgeDesiredCount": 2, "alarmEmail": "oncall@example.com",
        "owner": "voice-team", "costCenter": "cc-1234", "dataClassification": "restricted",
        "twilioOutputs": flex_outputs(tmp_path_factory),
    })


@pytest.fixture(scope="session")
def synth_domain(tmp_path_factory) -> Synth:
    """A custom domain and certificate, Twilio outputs and owned test numbers (dev)."""
    outputs = tmp_path_factory.mktemp("tw") / "twilio.context.json"
    outputs.write_text(json.dumps({
        "accountSid": "AC" + "1" * 32, "conversationConfigurationId": "conv_configuration_01",
        "workflowSid": "WW" + "2" * 32, "workspaceSid": "WS" + "4" * 32,
        "memoryStoreId": "mem_store_01",
        "queueSids": {"general": "WQ" + "3" * 32},
    }))
    return _synth(tmp_path_factory, "domain", {
        "domainName": "owlbank.example.com", "certificateArn": FAKE_CERT,
        "developerPhone": "+12025550100", "testNumbers": "+12025550111,+12025550112",
        "twilioOutputs": str(outputs), "verifyServiceSid": VERIFY_SID,
        "twilioPhoneNumber": "+12025550120", "bridgeDesiredCount": 2,
    })


@pytest.fixture(scope="session")
def synth_prod(tmp_path_factory) -> Synth:
    """stage=prod (AWS-M7), with a domain, an alarm email and explicit tags."""
    return _synth(tmp_path_factory, "prod", {
        "stage": "prod", "domainName": "owlbank.example.com", "certificateArn": FAKE_CERT,
        "bridgeDesiredCount": 2, "alarmEmail": "oncall@example.com",
        "owner": "voice-team", "costCenter": "cc-1234", "dataClassification": "restricted",
    })


@pytest.fixture(scope="session")
def synth_bpa_account(tmp_path_factory) -> Synth:
    """``s3PublicAccessBlockMode=account``: no per-bucket Block Public Access property.
    Defaults otherwise (CloudFront mode)."""
    return _synth(tmp_path_factory, "bpa-account", {"s3PublicAccessBlockMode": "account"})


@pytest.fixture(scope="session")
def synth_tuned(tmp_path_factory) -> Synth:
    """Latency knobs set: 2048 MB for the critical-path three, 3 provisioned executions."""
    return _synth(tmp_path_factory, "tuned", {"lambdaMemoryMb": 2048, "lambdaProvisionedConcurrency": 3})


@pytest.fixture(scope="session")
def synth_no_pc(tmp_path_factory) -> Synth:
    """``lambdaProvisionedConcurrency=0``: no alias anywhere, every invoker unqualified."""
    return _synth(tmp_path_factory, "no-pc", {"lambdaProvisionedConcurrency": 0})


@pytest.fixture(scope="session")
def synth_thinking_off(tmp_path_factory) -> Synth:
    """``modelParams`` thinking off on the first entry: it gets ``off`` and 1000 tokens.
    ``text_release=message`` is needed: ``stream`` with reasoning off is refused."""
    params = {"us.anthropic.claude-haiku-5-5": {"thinking": "off", "text_release": "message"}}
    return _synth(tmp_path_factory, "thinking-off", {"modelParams": json.dumps(params)})


@pytest.fixture(scope="session", params=["default", "domain", "prod"])
def synth_any(request) -> Synth:
    return request.getfixturevalue(f"synth_{request.param}")


@pytest.fixture(scope="session", params=["flex", "domain", "prod_flex"])
def synth_flex_any(request) -> Synth:
    """The same three configurations, each in Flex mode (``synth_domain`` already is)."""
    return request.getfixturevalue(f"synth_{request.param}")


@pytest.fixture(scope="session", params=["domain", "prod"])
def synth_domain_any(request) -> Synth:
    """Both custom-domain configurations (dev and prod stage)."""
    return request.getfixturevalue(f"synth_{request.param}")


# --------------------------------------------------------------------------- AgentCore hosting matrix

def _synth_agentcore(tmp_path_factory, mode: str, stage: str) -> Synth:
    """``bridgeHosting=agentcore`` with ``test_bridge_agentcore``'s contexts (digest, AZ-ID map, test
    phone number; prod adds the domain and, for ``public``, the acknowledgement). Imported here, not at
    the top, because that module imports this one."""
    from test_bridge_agentcore import _ctx, _quiet_synth
    return _quiet_synth(tmp_path_factory, f"matrix-ac-{mode}-{stage}", _ctx(mode, stage))


@pytest.fixture(scope="session")
def synth_agentcore_vpc_dev(tmp_path_factory) -> Synth:
    return _synth_agentcore(tmp_path_factory, "vpc", "dev")


@pytest.fixture(scope="session")
def synth_agentcore_public_dev(tmp_path_factory) -> Synth:
    return _synth_agentcore(tmp_path_factory, "public", "dev")


@pytest.fixture(scope="session")
def synth_agentcore_vpc_prod(tmp_path_factory) -> Synth:
    return _synth_agentcore(tmp_path_factory, "vpc", "prod")


@pytest.fixture(scope="session")
def synth_agentcore_public_prod(tmp_path_factory) -> Synth:
    """``public`` at prod, with ``agentcoreNetworkModeAck=public-at-prod``."""
    return _synth_agentcore(tmp_path_factory, "public", "prod")


#: The four Fargate context sets (``test_fargate_snapshot.CONTEXT_SETS``) and ``agentcore`` x {vpc, public}
#: x {dev, prod-with-ack}: every context the AgentCore hosting security matrix covers.
MATRIX = {
    "fargate-default": "synth_default", "fargate-custom-domain": "synth_domain",
    "fargate-bpa-account": "synth_bpa_account", "fargate-prod-domain": "synth_prod",
    "agentcore-vpc-dev": "synth_agentcore_vpc_dev", "agentcore-public-dev": "synth_agentcore_public_dev",
    "agentcore-vpc-prod": "synth_agentcore_vpc_prod", "agentcore-public-prod": "synth_agentcore_public_prod",
}


@pytest.fixture(scope="session", params=sorted(MATRIX))
def synth_matrix(request) -> Synth:
    return request.getfixturevalue(MATRIX[request.param])
