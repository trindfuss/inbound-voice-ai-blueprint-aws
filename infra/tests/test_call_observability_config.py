"""Call observability: infra settings.

``callObservability`` (``off`` | ``on``, default ``off``) and the keys read only when it is ``on``:
``traceSampleRate`` (0.0-1.0), ``traceContent`` (``off`` | ``redacted`` | ``full``; ``full`` refused
at ``stage=prod``), ``traceContentRetentionDays`` (1, 3, 5, 7, 14 or 30), ``tracesEnableTransactionSearch`` and the
fixed alarm ceilings. Every key is validated in both modes; ``off`` plus any of them warns that they
have no effect, and ``on`` with ``traceContent`` other than ``off`` warns that text is exported.

Config only (no synth) except the allow-list checks, which go through ``build_app``. The Hypothesis
tests cover "anything else is refused" for the two choice keys and the sample rate.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import owlbank_infra.app_builder as app_builder
from owlbank_infra.app_builder import build_app
from owlbank_infra.config import (
    CALL_OBSERVABILITY_KEYS,
    CALL_OBSERVABILITY_MODES,
    TRACE_CONTENT_MODES,
    ConfigError,
    from_context,
)

BASE = Path(__file__).resolve().parents[1]
ON = {"callObservability": "on"}
PROD = {"stage": "prod", "domainName": "voice.example.com",
        "certificateArn": "arn:aws:acm:us-east-1:111111111111:certificate/11111111-2222-3333-4444-555555555555"}

#: A valid, non-default value for every key read only when on.
NON_DEFAULTS = {
    "traceSampleRate": "0.25",
    "traceContent": "redacted",
    "traceContentRetentionDays": "14",
    "tracesEnableTransactionSearch": "true",
    "ttftCeilingSeconds": "2.5",
    "turnWaitCeilingSeconds": "6",
    "silenceCeilingSeconds": "3",
    "tokensPerCallCeiling": "80000",
}


def cfg(**ctx):
    return from_context(ctx, base_dir=BASE)


def quiet(**ctx):
    """``cfg`` with this module's warnings silenced (for tests about something else)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return cfg(**ctx)


# ---------------------------------------------------------------------------------- defaults

def test_defaults(recwarn):
    c = cfg()
    assert c.call_observability == "off" and c.call_observability_on is False
    assert c.trace_sample_rate == 1.0
    assert c.trace_content == "off"
    assert c.trace_content_retention_days == 7
    assert c.traces_enable_transaction_search is False
    assert (c.ttft_ceiling_seconds, c.turn_wait_ceiling_seconds, c.silence_ceiling_seconds) == (3, 5, 4)
    assert c.tokens_per_call_ceiling == 60000
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_on_with_defaults_does_not_warn(recwarn):
    c = cfg(**ON)
    assert c.call_observability_on is True and c.trace_content == "off"
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_on_reads_every_key():
    c = quiet(**ON, **NON_DEFAULTS)
    assert c.trace_sample_rate == 0.25
    assert c.trace_content == "redacted"
    assert c.trace_content_retention_days == 14
    assert c.traces_enable_transaction_search is True
    assert c.ttft_ceiling_seconds == 2.5
    assert c.turn_wait_ceiling_seconds == 6.0
    assert c.silence_ceiling_seconds == 3.0
    assert c.tokens_per_call_ceiling == 80000


def test_native_types_are_accepted():
    """``cdk.json`` and ``build_app(context=...)`` pass numbers and booleans, not strings."""
    c = quiet(**ON, traceSampleRate=0, traceContentRetentionDays=30, tracesEnableTransactionSearch=True,
              ttftCeilingSeconds=1, tokensPerCallCeiling=1)
    assert c.trace_sample_rate == 0.0 and c.trace_content_retention_days == 30
    assert c.traces_enable_transaction_search is True
    assert c.ttft_ceiling_seconds == 1.0 and c.tokens_per_call_ceiling == 1


@pytest.mark.parametrize("mode", CALL_OBSERVABILITY_MODES)
def test_each_mode_is_accepted(mode):
    assert cfg(callObservability=mode).call_observability == mode


@pytest.mark.parametrize("content", TRACE_CONTENT_MODES)
def test_each_content_is_accepted_outside_prod(content):
    assert quiet(**ON, traceContent=content).trace_content == content


# ---------------------------------------------------------------------------------- invalid values

INVALID = [
    ("callObservability", "yes"), ("callObservability", 1), ("callObservability", True),
    ("traceSampleRate", "1.5"), ("traceSampleRate", -0.1), ("traceSampleRate", "nan"),
    ("traceSampleRate", "inf"), ("traceSampleRate", "half"), ("traceSampleRate", True),
    ("traceSampleRate", [0.5]),
    ("traceContent", "redact"), ("traceContent", "on"), ("traceContent", 0),
    ("traceContentRetentionDays", 0), ("traceContentRetentionDays", "31"), ("traceContentRetentionDays", 2),
    ("traceContentRetentionDays", "7.5"), ("traceContentRetentionDays", True),
    ("tracesEnableTransactionSearch", "yes"), ("tracesEnableTransactionSearch", 1),
    ("ttftCeilingSeconds", 0), ("ttftCeilingSeconds", "61"), ("ttftCeilingSeconds", "fast"),
    ("turnWaitCeilingSeconds", -1), ("turnWaitCeilingSeconds", "inf"),
    ("silenceCeilingSeconds", "0.05"), ("silenceCeilingSeconds", False),
    ("tokensPerCallCeiling", 0), ("tokensPerCallCeiling", "60000.5"), ("tokensPerCallCeiling", "lots"),
]


@pytest.mark.parametrize("observability", ["on", "off"])
@pytest.mark.parametrize(("key", "value"), INVALID)
def test_invalid_value_names_its_setting(observability, key, value):
    """Refused in both modes: a typo fails now, not on the day observability is turned on."""
    ctx = {"callObservability": observability, key: value}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ConfigError, match=key):
            from_context(ctx, base_dir=BASE)


# ---------------------------------------------------------------------------------- full at prod

@pytest.mark.parametrize("observability", ["on", "off"])
def test_full_is_refused_at_prod(observability):
    with pytest.raises(ConfigError, match="traceContent"):
        quiet(**PROD, callObservability=observability, traceContent="full")


@pytest.mark.parametrize("content", ["off", "redacted"])
def test_prod_accepts_off_and_redacted(content):
    assert quiet(**PROD, **ON, traceContent=content).trace_content == content


@pytest.mark.parametrize("stage", ["dev", "test"])
def test_full_is_accepted_outside_prod(stage):
    assert quiet(stage=stage, **ON, traceContent="full").trace_content == "full"


# ---------------------------------------------------------------------------------- warnings

@pytest.mark.parametrize("content", ["redacted", "full"])
def test_content_other_than_off_warns_that_text_is_exported(content):
    with pytest.warns(UserWarning, match=f"traceContent={content}.*exported.*traces log group"):
        cfg(**ON, traceContent=content)


def test_keys_while_off_warn_once_naming_them():
    with pytest.warns(UserWarning, match="no effect with callObservability=off") as record:
        c = cfg(traceSampleRate="0.5", ttftCeilingSeconds="2")
    messages = [str(w.message) for w in record if issubclass(w.category, UserWarning)]
    assert len(messages) == 1
    assert "traceSampleRate" in messages[0] and "ttftCeilingSeconds" in messages[0]
    assert "traceContent" not in messages[0]
    assert c.call_observability_on is False


def test_every_key_while_off_is_named():
    with pytest.warns(UserWarning, match="no effect") as record:
        cfg(**NON_DEFAULTS)
    text = " ".join(str(w.message) for w in record)
    for key in CALL_OBSERVABILITY_KEYS:
        assert key in text
    assert "exported" not in text  # the content warning is for on only


def test_blank_keys_while_off_do_not_warn(recwarn):
    cfg(**dict.fromkeys(CALL_OBSERVABILITY_KEYS, ""))
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


# ---------------------------------------------------------------------------------- allow-list

def test_build_app_passes_every_key_to_from_context(monkeypatch, tmp_path):
    """Every key reaches ``from_context`` through ``build_app``'s context allow-list."""
    seen = {}

    class Stop(Exception):
        pass

    def capture(ctx, *, base_dir):
        seen.update(ctx)
        raise Stop

    monkeypatch.setattr(app_builder, "from_context", capture)
    context = {**ON, **NON_DEFAULTS}
    with pytest.raises(Stop):
        build_app(context, outdir=str(tmp_path))
    for key, value in context.items():
        assert seen.get(key) == value, key


def test_build_app_applies_the_settings(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, stacks = build_app({**ON, **NON_DEFAULTS}, outdir=str(tmp_path))
    c = stacks["compute"].cfg
    assert c.call_observability_on and c.trace_content == "redacted" and c.trace_sample_rate == 0.25
    assert c.tokens_per_call_ceiling == 80000


def test_build_app_refuses_a_bad_value(tmp_path):
    with pytest.raises(ConfigError, match="traceSampleRate"):
        build_app({**ON, "traceSampleRate": "2"}, outdir=str(tmp_path))


# ---------------------------------------------------------------------------------- properties

def _check_choice(key, allowed, default, raw, base):
    ctx = {**base, key: raw}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if not raw.strip():
            assert getattr(from_context(ctx, base_dir=BASE), _attr(key)) == default
        elif raw.strip().lower() in allowed:
            assert getattr(from_context(ctx, base_dir=BASE), _attr(key)) == raw.strip().lower()
        else:
            with pytest.raises(ConfigError, match=key):
                from_context(ctx, base_dir=BASE)


def _attr(key):
    return {"callObservability": "call_observability", "traceContent": "trace_content"}[key]


@settings(max_examples=200, deadline=None)
@given(st.text(max_size=20))
def test_any_other_call_observability_is_refused(raw):
    """6.1: anything that isn't off or on (blank means the default) fails, naming the key."""
    _check_choice("callObservability", CALL_OBSERVABILITY_MODES, "off", raw, {})


@settings(max_examples=200, deadline=None)
@given(st.text(max_size=20))
def test_any_other_trace_content_is_refused(raw):
    """5.2: anything that isn't off, redacted or full (blank means the default) fails."""
    _check_choice("traceContent", TRACE_CONTENT_MODES, "off", raw, ON)


_OUTSIDE = st.one_of(
    st.floats(max_value=0.0, exclude_max=True),
    st.floats(min_value=1.0, exclude_min=True),
    st.sampled_from([math.nan, math.inf, -math.inf]),
)


@settings(max_examples=200, deadline=None)
@given(_OUTSIDE, st.booleans())
def test_sample_rate_outside_unit_interval_is_refused(value, as_text):
    """6.1: any number outside [0, 1], or not finite, as a number or as text, fails."""
    raw = repr(value) if as_text else value
    with pytest.raises(ConfigError, match="traceSampleRate"):
        from_context({**ON, "traceSampleRate": raw}, base_dir=BASE)


@settings(max_examples=200, deadline=None)
@given(st.floats(min_value=0.0, max_value=1.0), st.booleans())
def test_sample_rate_inside_unit_interval_is_accepted(value, as_text):
    raw = repr(value) if as_text else value
    assert from_context({**ON, "traceSampleRate": raw}, base_dir=BASE).trace_sample_rate == value
