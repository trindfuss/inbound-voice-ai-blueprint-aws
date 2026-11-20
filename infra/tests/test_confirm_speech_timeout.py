"""The front door's Gather speechTimeout for yes/no confirmations
(``confirmSpeechTimeout`` context), separate from digit factors' ``speechTimeout``."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import _synth
from owlbank_infra.config import ConfigError, InfraConfig, from_context

BASE = Path(__file__).resolve().parent.parent
ENV = "OWLBANK_CONFIRM_SPEECH_TIMEOUT"


def _cfg(**ctx: object) -> InfraConfig:
    return from_context(dict(ctx), base_dir=BASE)


def _frontdoor_env(synth) -> dict:
    fd = next(r for _, lid, r in synth.resources("AWS::Lambda::Function") if lid.startswith("FrontdoorFn"))
    return fd["Properties"]["Environment"]["Variables"]


@pytest.mark.parametrize("raw, expected", [(None, "2"), ("", "2"), ("auto", "auto"), ("AUTO", "auto"),
                                           ("3", "3"), (1, "1"), ("10", "10")])
def test_confirm_speech_timeout_values(raw: object, expected: str) -> None:
    ctx = {} if raw is None else {"confirmSpeechTimeout": raw}
    assert _cfg(**ctx).confirm_speech_timeout == expected


@pytest.mark.parametrize("raw", ["0", "11", "1.5", "fast", True])
def test_bad_confirm_speech_timeout_is_refused(raw: object) -> None:
    with pytest.raises(ConfigError, match="confirmSpeechTimeout"):
        _cfg(confirmSpeechTimeout=raw)


def test_independent_of_speech_timeout() -> None:
    cfg = _cfg(speechTimeout="auto")
    assert (cfg.speech_timeout, cfg.confirm_speech_timeout) == ("auto", "2")
    cfg = _cfg(confirmSpeechTimeout="4")
    assert (cfg.speech_timeout, cfg.confirm_speech_timeout) == ("3", "4")


@pytest.fixture(scope="module")
def synth_confirm(tmp_path_factory):
    return _synth(tmp_path_factory, "confirmtimeout", {"confirmSpeechTimeout": "3"})


def test_default_reaches_the_front_door(synth_default):
    assert _frontdoor_env(synth_default)[ENV] == "2"


def test_context_value_reaches_the_front_door(synth_confirm):
    assert _frontdoor_env(synth_confirm)[ENV] == "3"


def test_only_the_front_door_gets_the_setting(synth_default):
    for _, lid, r in synth_default.resources("AWS::Lambda::Function"):
        env = r["Properties"].get("Environment", {}).get("Variables", {})
        if not lid.startswith("FrontdoorFn"):
            assert ENV not in env, lid


def test_build_app_rejects_junk(tmp_path):
    from owlbank_infra.app_builder import build_app
    with pytest.raises(ConfigError, match="confirmSpeechTimeout"):
        build_app({"confirmSpeechTimeout": "junk"}, outdir=str(tmp_path))
