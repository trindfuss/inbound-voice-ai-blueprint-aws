"""AgentCore hosting: the ``bridgeImageDigest`` and
``availabilityZones`` parsers accept exactly their domain, and the generated Runtime names match
the CFN ``AgentRuntimeName`` pattern.

The parsers live in the CDK app (``infra/owlbank_infra/config.py``), a separate uv project without
Hypothesis; that module is stdlib-only, so it is loaded by path, as in
``test_agent_settings_props.py``. The example-based checks and the synth checks are in
``infra/tests/test_stage2_config.py``.
"""

from __future__ import annotations

import json
import re
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.property.test_agent_settings_props import INFRA, config

DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
CFN_RUNTIME_NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,47}")
HEX = "0123456789abcdef"

valid_digests = st.text(alphabet=HEX, min_size=64, max_size=64).map(lambda h: "sha256:" + h)
near_digests = st.one_of(
    st.text(alphabet=HEX + "ABCDEFg", min_size=60, max_size=68).map(lambda h: "sha256:" + h),
    st.tuples(st.sampled_from(["sha256", "SHA256", "sha512", "sha256 ", ""]),
              st.sampled_from([":", "@", ""]), st.text(alphabet=HEX, min_size=64, max_size=64))
    .map("".join),
)
whitespace = st.sampled_from(["", " ", "\t", "\n", " \n "])


def _ctx(**kv):
    return config.from_context(kv, base_dir=INFRA)


@settings(max_examples=400, deadline=None)
@given(core=st.one_of(valid_digests, near_digests, st.text(max_size=80)), lead=whitespace, trail=whitespace)
def test_digest_parser_accepts_exactly_its_domain(core, lead, trail):
    value = lead + core + trail
    trimmed = value.strip()
    if trimmed == "" and value == "":
        assert _ctx(bridgeImageDigest=value).bridge_image_digest is None
    elif DIGEST_RE.fullmatch(trimmed):
        assert _ctx(bridgeImageDigest=value).bridge_image_digest == trimmed
    else:
        with pytest.raises(config.ConfigError, match="bridgeImageDigest"):
            _ctx(bridgeImageDigest=value)


@given(value=st.one_of(st.integers(), st.booleans(), st.lists(valid_digests, min_size=1, max_size=2),
                       st.dictionaries(st.text(max_size=3), st.text(max_size=3), min_size=1)))
def test_digest_parser_rejects_non_strings(value):
    with pytest.raises(config.ConfigError, match="bridgeImageDigest"):
        _ctx(bridgeImageDigest=value)


REGION = "us-east-1"
az_like = st.one_of(
    st.sampled_from(string.ascii_lowercase).map(lambda c: REGION + c),
    st.sampled_from(["us-east-1A", "us-west-2a", "us-east-1", "use1-az1", "us-east-1aa", "us-east-1-bos-1a", ""]),
    st.text(max_size=12),
)


def _in_domain(names: list[str]) -> bool:
    stripped = [n.strip() for n in names]
    return (len(stripped) == 2 and len(set(stripped)) == 2
            and all(re.fullmatch(r"us-east-1[a-z]", n) for n in stripped))


@settings(max_examples=400, deadline=None)
@given(names=st.lists(az_like, min_size=0, max_size=4), form=st.sampled_from(["list", "json", "csv"]))
def test_az_parser_accepts_exactly_its_domain(names, form):
    """Any list of strings, given as a list, a JSON list string or a comma-separated string: accepted
    iff it is exactly two distinct ``us-east-1<letter>`` names (after trimming), and then ``azs`` is
    that pair; anything else fails naming ``availabilityZones``."""
    if form == "csv":
        value: object = ",".join(names)
        if value == "" or any("," in n for n in names) or str(value).strip().startswith("["):
            return  # empty = absent; a comma inside a name or a leading "[" is a different input
    else:
        value = names if form == "list" else json.dumps(names)
    if _in_domain(names):
        cfg = _ctx(availabilityZones=value)
        assert cfg.availability_zones == tuple(n.strip() for n in names)
        assert cfg.azs == list(cfg.availability_zones)
    else:
        with pytest.raises(config.ConfigError, match="availabilityZones"):
            _ctx(availabilityZones=value)


def test_az_empty_value_is_absent():
    assert _ctx(availabilityZones="").azs == ["us-east-1a", "us-east-1b"]


@given(value=st.one_of(st.integers(), st.booleans(), st.dictionaries(st.text(max_size=3), st.text(max_size=3), min_size=1),
                       st.lists(st.one_of(st.integers(), st.none()), min_size=1, max_size=3)))
def test_az_parser_rejects_non_string_values(value):
    with pytest.raises(config.ConfigError, match="availabilityZones"):
        _ctx(availabilityZones=value)


@given(stage=st.sampled_from(["dev", "test", "prod"]))
def test_runtime_names_match_the_cfn_pattern(stage):
    """For every accepted stage, the Runtime name matches ``[a-zA-Z][a-zA-Z0-9_]{0,47}``."""
    extra = {"domainName": "owlbank.example.com",
             "certificateArn": "arn:aws:acm:us-east-1:111111111111:certificate/x"} if stage == "prod" else {}
    name = _ctx(stage=stage, **extra).runtime_name
    assert CFN_RUNTIME_NAME.fullmatch(name) and name == f"owlbank_{stage}_bridge"


@given(stage=st.text(max_size=60), kind=st.text(max_size=20))
def test_runtime_name_helper_never_returns_an_invalid_name(stage, kind):
    """``runtime_name_for`` either returns a name in the CFN pattern or raises ConfigError."""
    try:
        name = config.runtime_name_for(stage, kind)
    except config.ConfigError:
        assert not CFN_RUNTIME_NAME.fullmatch(f"owlbank_{stage}_{kind}")
    else:
        assert CFN_RUNTIME_NAME.fullmatch(name)
