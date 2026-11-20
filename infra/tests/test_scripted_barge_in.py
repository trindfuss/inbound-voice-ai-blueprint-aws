"""How a caller may interrupt a scripted read-out follows ``scriptedBargeIn``."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import _synth
from owlbank_infra.config import ConfigError, from_context

BASE = Path(__file__).resolve().parents[1]
ENV = "OWLBANK_SCRIPTED_BARGE_IN"


def _frontdoor_env(synth) -> dict:
    fd = next(r for _, lid, r in synth.resources("AWS::Lambda::Function") if lid.startswith("FrontdoorFn"))
    return fd["Properties"]["Environment"]["Variables"]


@pytest.fixture(scope="module")
def synth_speech(tmp_path_factory):
    return _synth(tmp_path_factory, "bargein", {"scriptedBargeIn": "speech"})


def test_default_is_any(synth_any):
    assert _frontdoor_env(synth_any)[ENV] == "any"


def test_context_value_reaches_the_front_door(synth_speech):
    assert _frontdoor_env(synth_speech)[ENV] == "speech"


def test_only_the_front_door_gets_the_setting(synth_default):
    for _, lid, r in synth_default.resources("AWS::Lambda::Function"):
        env = r["Properties"].get("Environment", {}).get("Variables", {})
        if not lid.startswith("FrontdoorFn"):
            assert ENV not in env, lid


@pytest.mark.parametrize("value", ["any", "speech", "dtmf", "none"])
def test_valid_values(value):
    assert from_context({"scriptedBargeIn": value}, base_dir=BASE).scripted_barge_in == value


def test_unset_or_empty_is_any():
    assert from_context({}, base_dir=BASE).scripted_barge_in == "any"
    assert from_context({"scriptedBargeIn": ""}, base_dir=BASE).scripted_barge_in == "any"


@pytest.mark.parametrize("value", ["junk", "all", "off", "true", "speech,dtmf"])
def test_junk_is_refused(value):
    with pytest.raises(ConfigError, match="scriptedBargeIn"):
        from_context({"scriptedBargeIn": value}, base_dir=BASE)


def test_build_app_rejects_junk(tmp_path):
    from owlbank_infra.app_builder import build_app
    with pytest.raises(ConfigError, match="scriptedBargeIn"):
        build_app({"scriptedBargeIn": "junk"}, outdir=str(tmp_path))
