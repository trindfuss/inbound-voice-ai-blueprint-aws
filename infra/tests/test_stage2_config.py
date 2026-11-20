"""AgentCore hosting: the image digest, AZ keys and shared names.

Each key's accepted and rejected values (errors name the key), the ``InfraConfig`` names later tasks
use, and the synth-level promise that absent or off keys change no template. The
Hypothesis properties are in ``tests/property/test_stage2_config_props.py``: this uv
project has no Hypothesis, and ``config.py`` is stdlib-only, so it is loaded there by path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import _synth
from owlbank_infra.config import (
    AGENTCORE_SUPPORTED_AZ_IDS,
    RUNTIME_NAME_PATTERN,
    ConfigError,
    from_context,
)

BASE = Path(__file__).resolve().parents[1]
DIGEST = "sha256:" + "ab" * 32


def cfg(**ctx):
    return from_context(ctx, base_dir=BASE)


# --- defaults -------------------------------------------------------------------------------------

def test_defaults_are_off_and_unset():
    c = cfg()
    assert c.agentcore_single_az_ack is False
    assert c.bridge_image_digest is None and c.availability_zones is None
    assert c.azs == ["us-east-1a", "us-east-1b"]


# --- boolean keys ---------------------------------------------------------------------------------

BOOL_KEYS = (("agentcoreSingleAzAck", "agentcore_single_az_ack"),)


@pytest.mark.parametrize(("key", "attr"), BOOL_KEYS)
@pytest.mark.parametrize(("value", "expected"), [
    (True, True), (False, False), ("true", True), ("false", False), (" TRUE ", True), ("False", False),
    ("", False), (None, False)])
def test_bool_keys_accept(key, attr, value, expected):
    assert getattr(cfg(**{key: value}), attr) is expected


@pytest.mark.parametrize(("key", "attr"), BOOL_KEYS)
@pytest.mark.parametrize("value", ["yes", "on", "1", 1, 0, "maybe", ["true"]])
def test_bool_keys_reject(key, attr, value):
    with pytest.raises(ConfigError, match=key):
        cfg(**{key: value})


# --- bridgeImageDigest ----------------------------------------------------------------------------

@pytest.mark.parametrize("value", [DIGEST, "sha256:" + "0" * 64, f"  {DIGEST} "])
def test_digest_accepts(value):
    assert cfg(bridgeImageDigest=value).bridge_image_digest == value.strip()


@pytest.mark.parametrize("value", [
    "ab" * 32, "sha256:" + "ab" * 31, "sha256:" + "ab" * 33, "sha256:" + "AB" * 32,
    "sha512:" + "ab" * 32, "sha256:" + "g" * 64, "SHA256:" + "ab" * 32, f"repo@{DIGEST}",
    f"{DIGEST}\nx", 123, True, [DIGEST]])
def test_digest_rejects(value):
    with pytest.raises(ConfigError, match="bridgeImageDigest"):
        cfg(bridgeImageDigest=value)


def test_require_digest_names_the_key_and_the_consumer():
    with pytest.raises(ConfigError, match="bridgeImageDigest.*the bridge stack"):
        cfg().require_bridge_image_digest("the bridge stack")
    assert cfg(bridgeImageDigest=DIGEST).require_bridge_image_digest("the bridge stack") == DIGEST


def test_no_digest_is_accepted_by_config():
    """The image repository deploys before any image exists, so the digest is checked by the stacks
    that need it (``require_bridge_image_digest``), not here."""
    assert cfg().bridge_image_digest is None


# --- availabilityZones ----------------------------------------------------------------------------

@pytest.mark.parametrize(("value", "expected"), [
    (["us-east-1a", "us-east-1c"], ("us-east-1a", "us-east-1c")),
    (("us-east-1d", "us-east-1b"), ("us-east-1d", "us-east-1b")),
    ("us-east-1a,us-east-1f", ("us-east-1a", "us-east-1f")),
    (" us-east-1a , us-east-1c ", ("us-east-1a", "us-east-1c")),
    ('["us-east-1b", "us-east-1d"]', ("us-east-1b", "us-east-1d"))])
def test_azs_accept(value, expected):
    c = cfg(availabilityZones=value)
    assert c.availability_zones == expected
    assert c.azs == list(expected)


@pytest.mark.parametrize("value", [
    ["us-east-1a"], ["us-east-1a", "us-east-1b", "us-east-1c"], ["us-east-1a", "us-east-1a"],
    ["us-west-2a", "us-west-2b"], ["us-east-1A", "us-east-1b"], ["us-east-1-bos-1a", "us-east-1b"],
    ["use1-az1", "use1-az2"], ["us-east-1ab", "us-east-1b"], "[not json", "us-east-1a", [],
    ["us-east-1a", 1], 7, {"a": 1}, "us-east-1a,,us-east-1b"])
def test_azs_reject(value):
    with pytest.raises(ConfigError, match="availabilityZones"):
        cfg(availabilityZones=value)


def test_azs_follow_the_region():
    c = cfg(region="us-west-2", domainName="owlbank.example.com",
            certificateArn="arn:aws:acm:us-west-2:111111111111:certificate/x",
            availabilityZones="us-west-2b,us-west-2c")
    assert c.azs == ["us-west-2b", "us-west-2c"]


# --- shared names ---------------------------------------------------------------------------------

@pytest.mark.parametrize("stage", ["dev", "test", "prod"])
def test_names(stage):
    extra = {"domainName": "owlbank.example.com",
             "certificateArn": "arn:aws:acm:us-east-1:111111111111:certificate/x"} if stage == "prod" else {}
    c = cfg(stage=stage, **extra)
    assert c.runtime_name == f"owlbank_{stage}_bridge"
    assert c.presign_role_name == f"owlbank-{stage}-relay-presign"
    assert c.image_repo_name == f"owlbank-{stage}-bridge"
    assert RUNTIME_NAME_PATTERN.fullmatch(c.runtime_name)
    roles = {c.presign_role_name, c.frontdoor_role_name, c.bridge_task_role_name}
    assert len(roles) == 3 and all(len(r) <= 64 for r in roles)


def test_supported_az_ids():
    assert cfg().supported_az_ids == ("use1-az1", "use1-az2", "use1-az4")
    assert AGENTCORE_SUPPORTED_AZ_IDS["us-east-1"] == cfg().supported_az_ids
    c = cfg(region="eu-west-1", domainName="owlbank.example.com",
            certificateArn="arn:aws:acm:eu-west-1:111111111111:certificate/x")
    assert c.supported_az_ids == ()


# --- synth: off keys change nothing; the AZ override is the only VPC change -----------------------

def _subnet_azs(synth):
    return sorted({res["Properties"]["AvailabilityZone"]
                   for _, _, res in synth.resources("AWS::EC2::Subnet")})


def test_off_keys_change_no_template(tmp_path_factory, synth_default):
    off = _synth(tmp_path_factory, "stage2-off", {
        "agentcoreSingleAzAck": "true",
        "bridgeImageDigest": DIGEST})
    assert off.templates == synth_default.templates


def test_build_app_rejects_bad_keys(tmp_path):
    from owlbank_infra.app_builder import build_app
    for key, value in (("bridgeImageDigest", "latest"), ("availabilityZones", "us-east-1a"),
                       ("agentcoreSingleAzAck", "y")):
        with pytest.raises(ConfigError, match=key):
            build_app({key: value}, outdir=str(tmp_path))


def test_availability_zones_override_moves_the_subnets(tmp_path_factory, synth_default):
    assert _subnet_azs(synth_default) == ["us-east-1a", "us-east-1b"]
    moved = _synth(tmp_path_factory, "stage2-azs", {"availabilityZones": "us-east-1c,us-east-1d"})
    assert _subnet_azs(moved) == ["us-east-1c", "us-east-1d"]


# --- Hosting keys --

PROD = {"stage": "prod", "domainName": "owlbank.example.com",
        "certificateArn": "arn:aws:acm:us-east-1:111111111111:certificate/x"}
AGENTCORE = {"bridgeHosting": "agentcore", "bridgeImageDigest": DIGEST}


def test_hosting_defaults():
    c = cfg()
    assert c.bridge_hosting == "fargate" and not c.is_agentcore and not c.agentcore_in_vpc
    assert (c.agentcore_network_mode, c.agentcore_idle_timeout_s, c.agentcore_session_quota) == ("vpc", 900, 5000)


@pytest.mark.parametrize(("value", "expected"), [
    ("fargate", "fargate"), ("agentcore", "agentcore"), (" AgentCore ", "agentcore"), ("", "fargate"),
    (None, "fargate")])
def test_bridge_hosting_accepts(value, expected):
    c = cfg(bridgeHosting=value, bridgeImageDigest=DIGEST)
    assert c.bridge_hosting == expected and c.is_agentcore is (expected == "agentcore")


@pytest.mark.parametrize("value", ["ecs", "agent-core", "agentcore,fargate", "lambda", 1, True, ["fargate"]])
def test_bridge_hosting_rejects(value):
    with pytest.raises(ConfigError, match="bridgeHosting") as err:
        cfg(bridgeHosting=value, bridgeImageDigest=DIGEST)
    if isinstance(value, str):
        assert "fargate, agentcore" in str(err.value)


def test_agentcore_in_vpc():
    assert cfg(**AGENTCORE).agentcore_in_vpc is True
    assert cfg(**AGENTCORE, agentcoreNetworkMode="vpc").agentcore_in_vpc is True
    with pytest.warns(UserWarning):
        assert cfg(**AGENTCORE, agentcoreNetworkMode="public").agentcore_in_vpc is False


@pytest.mark.parametrize("hosting", ["fargate", "agentcore"])
@pytest.mark.parametrize("value", ["private", "VPC-public", "none", 0, False])
def test_network_mode_rejects_in_both_hostings(hosting, value):
    with pytest.raises(ConfigError, match="agentcoreNetworkMode") as err:
        cfg(bridgeHosting=hosting, bridgeImageDigest=DIGEST, agentcoreNetworkMode=value)
    if isinstance(value, str):
        assert "vpc, public" in str(err.value)


def test_public_warns_naming_what_is_given_up():
    with pytest.warns(UserWarning) as record:
        cfg(**AGENTCORE, agentcoreNetworkMode="public")
    text = " ".join(str(w.message) for w in record)
    for given_up in ("egress control", "flow logs", "DynamoDB gateway endpoint"):
        assert given_up in text


@pytest.mark.parametrize("mode", ["vpc", "public"])
def test_network_mode_ignored_with_a_warning_on_fargate(mode):
    with pytest.warns(UserWarning, match="agentcoreNetworkMode is ignored with bridgeHosting=fargate"):
        c = cfg(agentcoreNetworkMode=mode)
    assert c.agentcore_in_vpc is False and not c.is_agentcore


def test_fargate_prod_public_needs_no_ack():
    with pytest.warns(UserWarning, match="ignored"):
        assert cfg(**PROD, agentcoreNetworkMode="public").bridge_hosting == "fargate"


def test_prod_public_needs_the_ack():
    with pytest.raises(ConfigError, match="agentcoreNetworkModeAck=public-at-prod"):
        cfg(**PROD, **AGENTCORE, agentcoreNetworkMode="public")
    with pytest.warns(UserWarning, match="egress control"):
        c = cfg(**PROD, **AGENTCORE, agentcoreNetworkMode="public", agentcoreNetworkModeAck="public-at-prod")
    assert c.is_agentcore and not c.agentcore_in_vpc


@pytest.mark.parametrize("stage", ["dev", "test"])
def test_public_outside_prod_needs_no_ack(stage):
    with pytest.warns(UserWarning):
        assert cfg(stage=stage, **AGENTCORE, agentcoreNetworkMode="public").is_agentcore


def test_prod_vpc_needs_no_ack():
    assert cfg(**PROD, **AGENTCORE).agentcore_in_vpc


@pytest.mark.parametrize("extra", [{}, AGENTCORE])
@pytest.mark.parametrize("value", ["yes", "true", "public", "public_at_prod", "public-at-dev", 1, True])
def test_ack_rejects_any_other_value(extra, value):
    with pytest.raises(ConfigError, match="agentcoreNetworkModeAck") as err:
        cfg(**extra, agentcoreNetworkModeAck=value)
    if isinstance(value, str):
        assert "public-at-prod" in str(err.value)


@pytest.mark.parametrize(("value", "expected"), [
    (60, 60), ("900", 900), (28799, 28799), (3600.0, 3600), ("", 900), (None, 900)])
def test_idle_timeout_accepts(value, expected):
    assert cfg(agentcoreIdleTimeoutSeconds=value).agentcore_idle_timeout_s == expected


@pytest.mark.parametrize("value", [59, 0, -1, 28800, 28801, "15m", 900.5, True, [900]])
def test_idle_timeout_rejects(value):
    with pytest.raises(ConfigError, match="agentcoreIdleTimeoutSeconds"):
        cfg(**AGENTCORE, agentcoreIdleTimeoutSeconds=value)


def test_idle_timeout_at_max_lifetime_names_it():
    with pytest.raises(ConfigError, match="below the Runtime's maximum lifetime"):
        cfg(agentcoreIdleTimeoutSeconds=28800)


@pytest.mark.parametrize(("value", "expected"), [(1, 1), ("2500", 2500), (1_000_000, 1_000_000)])
def test_session_quota_accepts(value, expected):
    assert cfg(agentcoreSessionQuota=value).agentcore_session_quota == expected


@pytest.mark.parametrize("value", [0, -5, 1_000_001, "lots", 2.5, False])
def test_session_quota_rejects(value):
    with pytest.raises(ConfigError, match="agentcoreSessionQuota"):
        cfg(**AGENTCORE, agentcoreSessionQuota=value)


@pytest.mark.parametrize(("region", "expected"), [("us-east-1", 5000), ("us-west-2", 5000), ("eu-west-1", 2500)])
def test_session_quota_default_by_region(region, expected):
    extra = {} if region == "us-east-1" else {
        "domainName": "owlbank.example.com",
        "certificateArn": f"arn:aws:acm:{region}:111111111111:certificate/x"}
    assert cfg(region=region, **extra).agentcore_session_quota == expected


def test_agentcore_config_needs_no_digest_the_runtime_stack_does():
    """Push-then-deploy: the image repository deploys before any digest exists, so config
    accepts agentcore without one; the stack that builds the Runtime requires it."""
    c = cfg(bridgeHosting="agentcore")
    assert c.is_agentcore and c.bridge_image_digest is None
    with pytest.raises(ConfigError, match="bridgeImageDigest is required by the Runtime"):
        c.require_bridge_image_digest("the Runtime")
    assert cfg(**AGENTCORE).bridge_image_digest == DIGEST


@pytest.mark.parametrize("reuse", [False, "false", "False"])
def test_agentcore_refuses_reuse_conversation_false(reuse):
    with pytest.raises(ConfigError) as err:
        cfg(**AGENTCORE, reuseConversation=reuse)
    assert "bridgeHosting=agentcore" in str(err.value) and "reuseConversation" in str(err.value)


@pytest.mark.parametrize("reuse", [True, False, "true", "false", None])
def test_fargate_accepts_either_reuse_conversation(reuse):
    assert cfg(reuseConversation=reuse).bridge_hosting == "fargate"


def test_agentcore_accepts_reuse_conversation_true():
    assert cfg(**AGENTCORE, reuseConversation="true").reuse_conversation is True


def test_hosting_keys_reach_build_app(tmp_path):
    from owlbank_infra.app_builder import build_app
    for key, value in (("bridgeHosting", "ecs"), ("agentcoreNetworkMode", "private"),
                       ("agentcoreNetworkModeAck", "yes"), ("agentcoreIdleTimeoutSeconds", "30"),
                       ("agentcoreSessionQuota", "0")):
        with pytest.raises(ConfigError, match=key):
            build_app({key: value}, outdir=str(tmp_path))
    with pytest.raises(ConfigError, match="reuseConversation"):
        build_app({**AGENTCORE, "reuseConversation": "false"}, outdir=str(tmp_path))


def test_fargate_hosting_keys_change_no_template(tmp_path_factory, synth_default):
    """Explicit ``fargate`` with every agentcore-only key at a valid value: the templates are the
    default's."""
    with pytest.warns(UserWarning, match="ignored"):
        same = _synth(tmp_path_factory, "stage2-hosting-fargate", {
            "bridgeHosting": "fargate", "agentcoreNetworkMode": "public",
            "agentcoreNetworkModeAck": "public-at-prod", "agentcoreIdleTimeoutSeconds": "1800",
            "agentcoreSessionQuota": "100", "reuseConversation": "true"})
    assert same.templates == synth_default.templates
