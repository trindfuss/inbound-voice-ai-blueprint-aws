"""``s3PublicAccessBlockMode`` context parameter: validation and wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import _synth
from owlbank_infra.config import ConfigError, from_context
from test_aws_defaults import BPA_ALL_ON

BASE = Path(__file__).resolve().parents[1]


def test_default_is_explicit():
    assert from_context({}, base_dir=BASE).s3_public_access_block_mode == "explicit"
    assert from_context({"s3PublicAccessBlockMode": ""}, base_dir=BASE).explicit_public_access_block


@pytest.mark.parametrize("value, explicit", [("explicit", True), ("account", False)])
def test_valid_values(value, explicit):
    cfg = from_context({"s3PublicAccessBlockMode": value}, base_dir=BASE)
    assert cfg.s3_public_access_block_mode == value and cfg.explicit_public_access_block is explicit


@pytest.mark.parametrize("value", ["Account", "EXPLICIT", "none", "off", "false", " account", 1, True, ["account"]])
def test_anything_else_fails_synth(value):
    with pytest.raises(ConfigError, match="s3PublicAccessBlockMode"):
        from_context({"s3PublicAccessBlockMode": value}, base_dir=BASE)


def test_build_app_rejects_a_bad_value(tmp_path):
    from owlbank_infra.app_builder import build_app
    with pytest.raises(ConfigError, match="s3PublicAccessBlockMode"):
        build_app({"s3PublicAccessBlockMode": "maybe"}, outdir=str(tmp_path))


def test_shipped_cdk_json_leaves_the_mode_unset():
    """The checked-in context keeps the default; an account whose SCP forbids the property passes
    ``-c s3PublicAccessBlockMode=account`` on the command line."""
    ctx = json.loads((BASE / "cdk.json").read_text())["context"]
    assert "s3PublicAccessBlockMode" not in ctx


def test_shipped_context_blocks_public_access_on_every_bucket(tmp_path_factory):
    """Synthesized with the checked-in context (admin console included; the Twilio outputs file
    exists only after setup), every bucket of every stack sets all four Block Public Access settings."""
    ctx = json.loads((BASE / "cdk.json").read_text())["context"]
    ctx.pop("twilioOutputs", None)
    synth = _synth(tmp_path_factory, "shipped-context", {**ctx, "adminConsole": "true"})
    buckets = list(synth.resources("AWS::S3::Bucket"))
    assert {s for s, _, _ in buckets} >= {"OwlBankBridge", "OwlBankAdmin"}
    for stack, lid, res in buckets:
        assert res["Properties"].get("PublicAccessBlockConfiguration") == BPA_ALL_ON, (stack, lid)
