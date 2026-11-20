"""Model-assisted slots: ``speechLog`` and ``speechLogRedaction``.

``speechLog`` (``on`` | ``off``, default ``on``) and ``speechLogRedaction`` (``on`` | ``off``, default
``on`` in every stage) reach the front door as ``OWLBANK_SPEECH_LOG`` and ``OWLBANK_SPEECH_LOG_REDACTION``; any
other value fails synth. No new IAM: the front door already has ``comprehend:DetectPiiEntities``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FAKE_CERT, _synth
from hypothesis import given, settings
from hypothesis import strategies as st
from test_slot_mode_ceiling import _actions, _fn_env, _frontdoor_role_statements

from owlbank_infra.app_builder import build_app
from owlbank_infra.config import (
    DEFAULT_SPEECH_LOG,
    DEFAULT_SPEECH_LOG_REDACTION,
    SPEECH_LOG_MODES,
    ConfigError,
    from_context,
)

BASE = Path(__file__).resolve().parents[1]
KEYS = (("speechLog", "speech_log", DEFAULT_SPEECH_LOG),
        ("speechLogRedaction", "speech_log_redaction", DEFAULT_SPEECH_LOG_REDACTION))


def test_defaults():
    cfg = from_context({}, base_dir=BASE)
    assert (cfg.speech_log, cfg.speech_log_redaction) == ("on", "on")


@pytest.mark.parametrize("stage", ["dev", "test", "prod"])
def test_redaction_is_on_by_default_in_every_stage(stage):
    domain = {"domainName": "owlbank.example.com", "certificateArn": FAKE_CERT} if stage == "prod" else {}
    assert from_context({"stage": stage, **domain}, base_dir=BASE).speech_log_redaction == "on"


def test_default_synth_redacts(synth_default):
    assert _fn_env(synth_default, "OwlBankCompute", "FrontdoorFn")["OWLBANK_SPEECH_LOG_REDACTION"] == "on"


@pytest.mark.parametrize(("key", "attr", "_default"), KEYS)
@pytest.mark.parametrize("mode", SPEECH_LOG_MODES)
def test_each_value_is_accepted(key, attr, _default, mode):
    assert getattr(from_context({key: mode}, base_dir=BASE), attr) == mode


@pytest.mark.parametrize(("key", "_attr", "_default"), KEYS)
def test_synth_refuses_a_bogus_value(tmp_path, key, _attr, _default):
    with pytest.raises(ConfigError, match=key):
        build_app({key: "maybe"}, outdir=str(tmp_path))


@settings(max_examples=100, deadline=None)
@given(st.text(max_size=12), st.sampled_from(KEYS))
def test_any_other_value_is_refused(raw, key):
    name, attr, default = key
    if not raw.strip():
        assert getattr(from_context({name: raw}, base_dir=BASE), attr) == default
    elif raw.strip().lower() in SPEECH_LOG_MODES:
        assert getattr(from_context({name: raw}, base_dir=BASE), attr) == raw.strip().lower()
    else:
        with pytest.raises(ConfigError, match=name):
            from_context({name: raw}, base_dir=BASE)


@pytest.fixture(scope="module")
def speech_synth(tmp_path_factory):
    return _synth(tmp_path_factory, "speech-log-off-redaction-off",
                  {"speechLog": "off", "speechLogRedaction": "off"})


def test_front_door_env_carries_both(speech_synth):
    env = _fn_env(speech_synth, "OwlBankCompute", "FrontdoorFn")
    assert env["OWLBANK_SPEECH_LOG"] == "off"
    assert env["OWLBANK_SPEECH_LOG_REDACTION"] == "off"


def test_front_door_can_detect_pii(speech_synth):
    """18.19(b): the redaction call is the existing PiiRedactor's; the grant is already there."""
    actions = {a for s in _frontdoor_role_statements(speech_synth) for a in _actions(s)}
    assert "comprehend:DetectPiiEntities" in actions
