"""The front door's spoken-factor switch follows ``acceptSpokenFactors``,
which is on by default in every stage, prod included."""

from __future__ import annotations

import pytest
from conftest import _synth


def _frontdoor_env(synth) -> dict:
    fd = next(r for _, lid, r in synth.resources("AWS::Lambda::Function") if lid.startswith("FrontdoorFn"))
    return fd["Properties"]["Environment"]["Variables"]


@pytest.fixture(scope="module")
def synth_spoken_off(tmp_path_factory):
    """Defaults with ``-c acceptSpokenFactors=false``."""
    return _synth(tmp_path_factory, "spoken-off", {"acceptSpokenFactors": "false"})


def test_dev_default_accepts_spoken_factors(synth_default):
    assert _frontdoor_env(synth_default)["OWLBANK_ACCEPT_SPOKEN_FACTORS"] == "true"


def test_prod_default_accepts_spoken_factors(synth_prod):
    assert _frontdoor_env(synth_prod)["OWLBANK_ACCEPT_SPOKEN_FACTORS"] == "true"


def test_override_false_refuses_spoken_factors(synth_spoken_off):
    assert _frontdoor_env(synth_spoken_off)["OWLBANK_ACCEPT_SPOKEN_FACTORS"] == "false"


def test_only_the_front_door_gets_the_switch(synth_default):
    for _, lid, r in synth_default.resources("AWS::Lambda::Function"):
        env = r["Properties"].get("Environment", {}).get("Variables", {})
        if not lid.startswith("FrontdoorFn"):
            assert "OWLBANK_ACCEPT_SPOKEN_FACTORS" not in env, lid
