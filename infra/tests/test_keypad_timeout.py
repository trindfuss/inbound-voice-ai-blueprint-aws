"""Keyed-phone: the front door's keypad Gather ``timeout`` (``keypadTimeout`` context,
``OWLBANK_KEYPAD_TIMEOUT``), whole seconds 1-10, default 3."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import _synth
from owlbank_infra.config import ConfigError, InfraConfig, from_context

BASE = Path(__file__).resolve().parent.parent
ENV = "OWLBANK_KEYPAD_TIMEOUT"


def _cfg(**ctx: object) -> InfraConfig:
    return from_context(dict(ctx), base_dir=BASE)


def _frontdoor_env(synth) -> dict:
    fd = next(r for _, lid, r in synth.resources("AWS::Lambda::Function") if lid.startswith("FrontdoorFn"))
    return fd["Properties"]["Environment"]["Variables"]


@pytest.mark.parametrize("raw, expected", [(None, "3"), ("", "3"), ("5", "5"), (1, "1"), ("10", "10")])
def test_keypad_timeout_values(raw: object, expected: str) -> None:
    ctx = {} if raw is None else {"keypadTimeout": raw}
    assert _cfg(**ctx).keypad_timeout == expected


@pytest.mark.parametrize("raw", ["0", "11", "1.5", "fast", "auto", True])
def test_bad_keypad_timeout_is_refused(raw: object) -> None:
    with pytest.raises(ConfigError, match="keypadTimeout"):
        _cfg(keypadTimeout=raw)


@pytest.fixture(scope="module")
def synth_keypad(tmp_path_factory):
    return _synth(tmp_path_factory, "keypadtimeout", {"keypadTimeout": "5"})


def test_default_reaches_the_front_door(synth_default):
    assert _frontdoor_env(synth_default)[ENV] == "3"


def test_context_value_reaches_the_front_door(synth_keypad):
    assert _frontdoor_env(synth_keypad)[ENV] == "5"


def test_only_the_front_door_gets_the_setting(synth_default):
    for _, lid, r in synth_default.resources("AWS::Lambda::Function"):
        env = r["Properties"].get("Environment", {}).get("Variables", {})
        if not lid.startswith("FrontdoorFn"):
            assert ENV not in env, lid


def test_build_app_rejects_junk(tmp_path):
    from owlbank_infra.app_builder import build_app
    with pytest.raises(ConfigError, match="keypadTimeout"):
        build_app({"keypadTimeout": "junk"}, outdir=str(tmp_path))
