"""Loading any ``masking`` document never raises and always gives in-bounds values."""

from __future__ import annotations

import math
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from owlbank.masking_config import (
    CHARS_PER_S_RANGE,
    END_MODES,
    HANDOFF_WAIT_MS_RANGE,
    MASK_AFTER_MS_RANGE,
)
from owlbank.policy import load_masking, parse_masking
from owlbank.policy import PolicySectionError

FIELDS = ("after_ms", "handoff_wait_ms", "end_mode", "chars_per_s")

scalars = st.one_of(
    st.none(), st.booleans(), st.integers(-10**6, 10**6), st.integers(),
    st.floats(allow_nan=True, allow_infinity=True), st.text(max_size=12),
    st.sampled_from(["wait", "immediate", "15", "8.0", "nan", "inf", "1e999", "-0"]),
    st.decimals(allow_nan=True, allow_infinity=True),
)
json_like = st.recursive(scalars, lambda c: st.one_of(
    st.lists(c, max_size=3), st.dictionaries(st.text(max_size=8), c, max_size=3)), max_leaves=6)
keys = st.one_of(st.sampled_from(FIELDS), st.text(max_size=10))
docs = st.one_of(json_like, st.dictionaries(keys, json_like, max_size=6))


@given(raw=docs)
def test_loading_never_raises_and_stays_in_bounds(raw: Any) -> None:
    cfg = load_masking(raw)
    assert isinstance(cfg.after_ms, int) and not isinstance(cfg.after_ms, bool)
    assert MASK_AFTER_MS_RANGE[0] <= cfg.after_ms <= MASK_AFTER_MS_RANGE[1]
    assert isinstance(cfg.handoff_wait_ms, int) and not isinstance(cfg.handoff_wait_ms, bool)
    assert HANDOFF_WAIT_MS_RANGE[0] <= cfg.handoff_wait_ms <= HANDOFF_WAIT_MS_RANGE[1]
    assert cfg.end_mode in END_MODES
    assert isinstance(cfg.chars_per_s, float) and math.isfinite(cfg.chars_per_s)
    assert CHARS_PER_S_RANGE[0] <= cfg.chars_per_s <= CHARS_PER_S_RANGE[1]
    # The seed shape always re-parses strictly to the same config (store round trip).
    assert parse_masking(cfg.to_doc()) == cfg


@given(raw=docs)
def test_strict_parse_raises_only_section_errors(raw: Any) -> None:
    try:
        cfg = parse_masking(raw)
    except PolicySectionError as exc:
        assert exc.field.startswith("masking")
    else:
        assert cfg == load_masking(raw)
