"""The AgentCore Runtime's subnets in VPC mode, chosen by AZ ID.

Offline: the AZ name -> ID maps are made up, nothing is looked up. The Fargate templates don't use
``azs.py``; ``test_fargate_snapshot.py`` shows they are unchanged.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from aws_cdk import App, Environment, Stack
from aws_cdk import aws_ec2 as ec2
from aws_cdk.assertions import Annotations, Match, Template
from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank_infra import azs
from owlbank_infra.azs import AzSelectionError, az_ids_context_key, runtime_subnets, select_runtime_subnets
from owlbank_infra.config import AGENTCORE_SUPPORTED_AZ_IDS, FAKE_ACCOUNT, InfraConfig

SUPPORTED = AGENTCORE_SUPPORTED_AZ_IDS["us-east-1"]
KEY = az_ids_context_key(FAKE_ACCOUNT, "us-east-1")


@dataclass(frozen=True)
class FakeSubnet:
    availability_zone: str
    name: str = ""


def _subnets(*names: str) -> list[FakeSubnet]:
    return [FakeSubnet(n, f"private-{i}") for i, n in enumerate(names)]


MAP_BOTH = {"us-east-1a": "use1-az1", "us-east-1b": "use1-az4", "us-east-1c": "use1-az6"}
MAP_ONE = {"us-east-1a": "use1-az6", "us-east-1b": "use1-az2", "us-east-1c": "use1-az1",
           "us-east-1d": "use1-az3"}
MAP_NONE = {"us-east-1a": "use1-az3", "us-east-1b": "use1-az5", "us-east-1c": "use1-az6",
            "us-east-1d": "use1-az1", "us-east-1e": "use1-az4"}
AB = ("us-east-1a", "us-east-1b")


# ---- unit: each branch -------------------------------------------------------------------------

def test_context_key_shape():
    assert az_ids_context_key("123456789012", "us-west-2") == "owlbank:az-ids:123456789012:us-west-2"


def test_two_supported_gives_both_without_warning():
    subs = _subnets(*AB)
    r = runtime_subnets(subs, MAP_BOTH, SUPPORTED, "prod", False)
    assert r.subnets == tuple(subs)
    assert r.az_ids == ("use1-az1", "use1-az4")
    assert r.warning is None and r.kind == "multi_az"


@pytest.mark.parametrize("stage,ack", [("dev", False), ("test", False), ("dev", True), ("prod", True)])
def test_one_supported_gives_that_subnet_with_warning(stage, ack):
    subs = _subnets(*AB)
    r = runtime_subnets(subs, MAP_ONE, SUPPORTED, stage, ack)
    assert r.subnets == (subs[1],) and r.az_ids == ("use1-az2",) and r.kind == "single_az"
    assert r.warning is not None
    for value in ("us-east-1b", "use1-az2", "us-east-1a", "use1-az6", "availabilityZones", *SUPPORTED):
        assert value in r.warning


def test_one_supported_fails_at_prod_without_ack():
    with pytest.raises(AzSelectionError) as exc:
        runtime_subnets(_subnets(*AB), MAP_ONE, SUPPORTED, "prod", False)
    msg = str(exc.value)
    for value in ("agentcoreSingleAzAck", "stage=prod", "us-east-1b", "use1-az2", "use1-az6"):
        assert value in msg


def test_none_supported_fails_naming_ids_supported_and_key():
    with pytest.raises(AzSelectionError) as exc:
        runtime_subnets(_subnets(*AB), MAP_NONE, SUPPORTED, "dev", True, region="us-east-1")
    msg = str(exc.value)
    for value in ("us-east-1a", "use1-az3", "us-east-1b", "use1-az5", "availabilityZones", *SUPPORTED):
        assert value in msg
    # The AZ names in this account whose IDs would work.
    assert "us-east-1d (use1-az1)" in msg and "us-east-1e (use1-az4)" in msg


def test_region_without_supported_list_fails():
    with pytest.raises(AzSelectionError) as exc:
        runtime_subnets(_subnets(*AB), MAP_BOTH, (), "dev", False, region="eu-west-1")
    assert "AGENTCORE_SUPPORTED_AZ_IDS" in str(exc.value)


def test_no_cached_map_fails_asking_for_preflight():
    with pytest.raises(AzSelectionError) as exc:
        runtime_subnets(_subnets(*AB), None, SUPPORTED, "dev", False, context_key=KEY)
    msg = str(exc.value)
    assert KEY in msg and "preflight.py" in msg and "--agentcore" in msg


@pytest.mark.parametrize("raw", [{}, ["us-east-1a"], {"us-east-1a": 3}, "use1-az1", {"": "use1-az1"}])
def test_malformed_map_fails_asking_for_preflight(raw):
    with pytest.raises(AzSelectionError) as exc:
        runtime_subnets(_subnets(*AB), raw, SUPPORTED, "dev", False, context_key=KEY)
    assert KEY in str(exc.value) and "preflight.py" in str(exc.value)


def test_map_without_our_az_fails_asking_for_preflight_never_guesses():
    with pytest.raises(AzSelectionError) as exc:
        runtime_subnets(_subnets("us-east-1a", "us-east-1f"), MAP_BOTH, SUPPORTED, "dev", False)
    assert "us-east-1f" in str(exc.value) and "preflight.py" in str(exc.value)


def test_no_subnets_fails():
    with pytest.raises(AzSelectionError):
        runtime_subnets([], MAP_BOTH, SUPPORTED, "dev", False)


def test_az_selection_error_is_a_config_error():
    from owlbank_infra.config import ConfigError
    assert issubclass(AzSelectionError, ConfigError)


def test_preflight_constants_match_config():
    """preflight.py stays standalone (no CDK import) but must cache under the key synth reads and
    check the same supported AZ IDs."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"
    spec = importlib.util.spec_from_file_location("preflight_for_azs", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["preflight_for_azs"] = mod
    spec.loader.exec_module(mod)
    assert mod.AZ_IDS_CONTEXT_PREFIX == azs.AZ_IDS_CONTEXT_PREFIX
    assert dict(mod.AGENTCORE_SUPPORTED_AZ_IDS) == dict(AGENTCORE_SUPPORTED_AZ_IDS)


# ---- CDK: the real VPC's private subnets, context lookup, synth warning, VPC unchanged ----------

def _vpc_stack(context: dict, cfg: InfraConfig, cid: str = "Net") -> tuple[Stack, ec2.Vpc]:
    app = App(context=context)
    stack = Stack(app, cid, env=Environment(account=cfg.account, region=cfg.region))
    vpc = ec2.Vpc(stack, "Vpc", availability_zones=cfg.azs, nat_gateways=1, subnet_configuration=[
        ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24),
        ec2.SubnetConfiguration(name="private", subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                                cidr_mask=22)])
    return stack, vpc


def test_select_reads_context_and_warns_for_single_az():
    cfg = InfraConfig()
    stack, vpc = _vpc_stack({KEY: MAP_ONE}, cfg)
    chosen = select_runtime_subnets(stack, cfg, vpc.private_subnets)
    assert [s.availability_zone for s in chosen] == ["us-east-1b"]
    Annotations.from_stack(stack).has_warning("/Net", Match.string_like_regexp("one AZ only"))


def test_select_two_supported_no_warning():
    cfg = InfraConfig()
    stack, vpc = _vpc_stack({KEY: MAP_BOTH}, cfg)
    chosen = select_runtime_subnets(stack, cfg, vpc.private_subnets)
    assert [s.node.path for s in chosen] == [s.node.path for s in vpc.private_subnets]
    Annotations.from_stack(stack).has_no_warning("*", Match.any_value())


def test_select_without_context_fails():
    cfg = InfraConfig()
    stack, vpc = _vpc_stack({}, cfg)
    with pytest.raises(AzSelectionError) as exc:
        select_runtime_subnets(stack, cfg, vpc.private_subnets)
    assert KEY in str(exc.value)


def test_select_prod_single_az_needs_ack():
    prod = InfraConfig(stage="prod")
    stack, vpc = _vpc_stack({KEY: MAP_ONE}, prod)
    with pytest.raises(AzSelectionError):
        select_runtime_subnets(stack, prod, vpc.private_subnets)
    acked = InfraConfig(stage="prod", agentcore_single_az_ack=True)
    stack, vpc = _vpc_stack({KEY: MAP_ONE}, acked, "Net2")
    assert len(select_runtime_subnets(stack, acked, vpc.private_subnets)) == 1


def test_select_leaves_the_vpc_template_unchanged():
    cfg = InfraConfig()
    plain, _ = _vpc_stack({KEY: MAP_ONE}, cfg)
    used, vpc = _vpc_stack({KEY: MAP_ONE}, cfg)
    select_runtime_subnets(used, cfg, vpc.private_subnets)
    assert Template.from_stack(used).to_json() == Template.from_stack(plain).to_json()


# ---- PBT -------------------------------------------------------------------------------

AZ_IDS = st.sampled_from([f"use1-az{i}" for i in range(1, 7)])
AZ_NAMES = st.sampled_from([f"us-east-1{c}" for c in "abcdef"])


@st.composite
def scenario(draw):
    az_map = draw(st.dictionaries(AZ_NAMES, AZ_IDS, min_size=1, max_size=6))
    names = draw(st.lists(st.sampled_from(sorted(az_map)), min_size=1, max_size=4, unique=True))
    subnets = [FakeSubnet(n, f"s{i}") for i, n in enumerate(names)]
    supported = tuple(draw(st.lists(AZ_IDS, max_size=6, unique=True)))
    stage = draw(st.sampled_from(["dev", "test", "prod"]))
    ack = draw(st.booleans())
    return az_map, subnets, supported, stage, ack


@settings(max_examples=300, deadline=None)
@given(scenario())
def test_selected_subnets_are_exactly_the_supported_ones(case):
    """For any name -> ID map and supported set, the Runtime's subnets are exactly
    the private subnets whose AZ ID is supported (order kept), the outcome kind matches their count,
    and the inputs (the VPC's subnets, the map) are never changed."""
    az_map, subnets, supported, stage, ack = case
    before = (copy.deepcopy(az_map), list(subnets), supported)
    expected = [s for s in subnets if az_map[s.availability_zone] in supported]
    try:
        r = runtime_subnets(subnets, az_map, supported, stage, ack)
    except AzSelectionError as exc:
        msg = str(exc)
        if not expected:
            for s in subnets:
                assert az_map[s.availability_zone] in msg and s.availability_zone in msg
            for i in supported:
                assert i in msg
            assert "availabilityZones" in msg
        else:
            assert len(expected) == 1 and stage == "prod" and not ack
            assert "agentcoreSingleAzAck" in msg
    else:
        assert list(r.subnets) == expected
        assert r.az_ids == tuple(az_map[s.availability_zone] for s in expected)
        assert set(r.az_ids) <= set(supported)
        if len(expected) == 1:
            assert r.kind == "single_az" and r.warning is not None
            assert not (stage == "prod" and not ack)
        else:
            assert len(expected) >= 2 and r.kind == "multi_az" and r.warning is None
    assert (az_map, subnets, supported) == before


@settings(max_examples=100, deadline=None)
@given(scenario(), AZ_NAMES)
def test_never_guesses_a_missing_az_id(case, missing):
    """A subnet whose AZ name is not in the cached map always fails, asking for preflight."""
    az_map, subnets, supported, stage, ack = case
    az_map = {k: v for k, v in az_map.items() if k != missing}
    subnets = [*(s for s in subnets if s.availability_zone != missing), FakeSubnet(missing, "x")]
    with pytest.raises(AzSelectionError) as exc:
        runtime_subnets(subnets, az_map or None, supported, stage, ack)
    assert "preflight.py" in str(exc.value)
