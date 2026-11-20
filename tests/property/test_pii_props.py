"""App-H1 properties: card numbers never survive masking, masking is idempotent, and the
configured entity types (and only those) are masked.

Redaction config: ``parse_redaction_config(cfg.to_doc()) == cfg``;
``enabled: false`` makes zero redactor calls for any text and any surface; ``enabled: true`` with an
empty ``entity_types`` is refused; any invalid section (any JSON) loads as the defaults with exactly
one ERROR naming a field and carrying no text from the section."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from owlbank import pii
from owlbank.pii import (
    COMPREHEND_PII_ENTITY_TYPES,
    DEFAULT_CONFIG,
    Entity,
    FakePiiRedactor,
    PiiGuard,
    RedactionConfig,
    RedactionConfigError,
    load_redaction_config,
    luhn_ok,
    mask_entities,
    parse_redaction_config,
)

NO_DIGITS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'\n:"
ANY_TEXT = NO_DIGITS + "0123456789-  "
TYPES = sorted(COMPREHEND_PII_ENTITY_TYPES)


def _check_digit(body: str) -> str:
    for d in "0123456789":
        if luhn_ok(body + d):
            return d
    raise AssertionError("unreachable")


@st.composite
def cards(draw) -> str:
    body = draw(st.text("0123456789", min_size=12, max_size=18))
    return body + _check_digit(body)


@st.composite
def spaced(draw, digits: str) -> str:
    seps = draw(st.lists(st.sampled_from(["", "", " ", "-"]), min_size=len(digits) - 1,
                         max_size=len(digits) - 1))
    return digits[0] + "".join(s + d for s, d in zip(seps, digits[1:], strict=True))


@settings(max_examples=300, deadline=None)
@given(data=st.data(), card=cards(), text=st.text(NO_DIGITS, max_size=80))
def test_property_a_luhn_card_never_survives(data, card: str, text: str) -> None:
    written = data.draw(spaced(card))
    at = data.draw(st.integers(0, len(text)))
    sample = text[:at] + written + text[at:]
    out = FakePiiRedactor().redact(sample, language="en-US").text
    assert card not in re.sub(r"[\s\-]", "", out)
    assert "[CREDIT_DEBIT_NUMBER]" in out


@settings(max_examples=300, deadline=None)
@given(data=st.data(), text=st.text(ANY_TEXT, max_size=120), card=st.none() | cards())
def test_property_masking_is_idempotent(data, text: str, card: str | None) -> None:
    if card is not None:
        at = data.draw(st.integers(0, len(text)))
        text = text[:at] + " " + data.draw(spaced(card)) + " " + text[at:]
    f = FakePiiRedactor()
    once = f.redact(text, language="en-US").text
    assert f.redact(once, language="en-US").text == once


@st.composite
def valid_docs(draw, *, enabled: bool | None = None) -> dict[str, Any]:
    on = draw(st.booleans()) if enabled is None else enabled
    raw: dict[str, Any] = {
        "enabled": on,
        "entity_types": draw(st.lists(st.sampled_from(TYPES), unique=True, min_size=1 if on else 0,
                                      max_size=len(TYPES))),
        "min_confidence": draw(st.floats(0, 1, allow_nan=False) | st.floats(0, 1).map(repr)),
        "mask": draw(st.text("abcXYZ*#[]:<>_ ", min_size=0, max_size=6))
        + draw(st.sampled_from(["{type}", ""]))
        + draw(st.text("abc*]>", max_size=3)),
        "on_failure": draw(st.sampled_from(["withhold", "allow"])),
    }
    if not raw["mask"]:
        raw["mask"] = "[{type}]"
    optional = sorted(raw) if enabled is None else sorted(set(raw) - {"enabled"})
    for name in draw(st.sets(st.sampled_from(optional), max_size=3)):  # missing fields: defaults
        del raw[name]
    if raw.get("enabled", True) and raw.get("entity_types") == []:
        raw["entity_types"] = ["SSN"]
    return raw


@st.composite
def configs(draw, *, enabled: bool | None = None) -> RedactionConfig:
    return parse_redaction_config(draw(valid_docs(enabled=enabled)))  # only valid configs


@settings(max_examples=300, deadline=None)
@given(cfg=configs(), parts=st.lists(
    st.tuples(st.text("abcdefgh", min_size=1, max_size=6),        # a gap (never mergeable)
              st.text("PQRSTUVW0123", min_size=1, max_size=8),     # the span's text
              st.sampled_from(TYPES), st.floats(0, 1, allow_nan=False)),
    max_size=6))
def test_property_configured_types_masked_and_only_those(cfg: RedactionConfig, parts) -> None:
    text, entities, expected = "", [], ""
    for gap, span, etype, score in parts:
        text += gap
        expected += gap
        entities.append(Entity(len(text), len(text) + len(span), etype, score))
        text += span
        masked = etype in cfg.entity_types and score >= cfg.min_confidence
        expected += cfg.render_mask(etype) if masked else span
    out = mask_entities(text, entities, cfg)
    assert out.text == expected
    assert set(out.entity_types) == {e.type for e in entities if e.type in cfg.entity_types
                                     and e.score >= cfg.min_confidence}




@settings(max_examples=300, deadline=None)
@given(cfg=configs())
def test_config_round_trips_through_its_doc(cfg: RedactionConfig) -> None:
    assert parse_redaction_config(cfg.to_doc()) == cfg


@pytest.fixture(autouse=True)
def _fresh_off_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pii, "_warned_off", set())


@settings(max_examples=300, deadline=None)
@given(cfg=configs(enabled=False), fail=st.booleans(),
       calls=st.lists(st.tuples(st.sampled_from([pii.SURFACE_AI_TURN, pii.SURFACE_ESCALATION_SUMMARY])
                                | st.text(max_size=12), st.text(ANY_TEXT, max_size=80)
                                | st.text(max_size=80)), max_size=8))
def test_disabled_makes_zero_redactor_calls(cfg: RedactionConfig, fail: bool,
                                                  calls: list[tuple[str, str]]) -> None:
    fake = FakePiiRedactor(fail=fail)
    guard = PiiGuard(fake, cfg)
    for surface, text in calls:
        assert guard.text(surface, text, language="en-US", withheld="[WITHHELD]") == text
    assert fake.calls == 0


@settings(max_examples=200, deadline=None)
@given(raw=valid_docs(enabled=True))
def test_enabled_with_no_types_is_refused(raw: dict[str, Any]) -> None:
    raw = {**raw, "entity_types": []}
    with pytest.raises(RedactionConfigError) as ei:
        parse_redaction_config(raw)
    assert ei.value.field == "entity_types"


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def _pii_logs() -> Iterator[list[logging.LogRecord]]:
    lg = logging.getLogger("owlbank.pii")
    h, level = _Capture(), lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        yield h.records
    finally:
        lg.removeHandler(h)
        lg.setLevel(level)


json_scalars = st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=20)
any_json = st.recursive(json_scalars, lambda inner: st.lists(inner, max_size=4)
                        | st.dictionaries(st.text(max_size=12), inner, max_size=4), max_leaves=12)
FIELDS = ["enabled", "entity_types", "min_confidence", "mask", "on_failure"]


@st.composite
def sections(draw) -> Any:
    """Any JSON: arbitrary values, and valid docs with one or more fields replaced by any JSON or an
    extra key added (so every field's check is reached)."""
    if draw(st.booleans()):
        return draw(any_json)
    raw = draw(valid_docs())
    for name in draw(st.sets(st.sampled_from(FIELDS), min_size=1, max_size=2)):
        raw[name] = draw(any_json)
    if draw(st.booleans()):
        raw[draw(st.text(max_size=12))] = draw(any_json)
    return raw


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _strings(v)


@settings(max_examples=500, deadline=None)
@given(raw=sections())
@example(raw={"A\n": None})  # a trailing newline once passed the field-name check
def test_any_invalid_section_loads_the_defaults_with_one_error(raw: Any) -> None:
    try:
        expected: RedactionConfig | None = parse_redaction_config(raw)
    except RedactionConfigError:
        expected = None
    with _pii_logs() as records:
        cfg = load_redaction_config(raw)
    errors = [r for r in records if r.levelno >= logging.ERROR]
    if expected is not None:
        assert cfg == expected and errors == []
        return
    assert cfg == DEFAULT_CONFIG
    assert len(errors) == 1
    (error,) = errors
    known = {*FIELDS, "(section)", "(unknown field)"}
    extra_keys = set(raw) - set(FIELDS) if isinstance(raw, dict) else set()
    assert error.field in known | extra_keys  # type: ignore[attr-defined]
    assert error.field in known or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,39}", error.field)  # type: ignore[attr-defined]
    # No text from the section: the message is the fixed template with the field name only.
    assert error.args in ((), (error.field,))  # type: ignore[attr-defined]
    message = error.getMessage()
    for text in _strings(raw):
        if len(text) >= 3 and text != error.field and text not in known:  # type: ignore[attr-defined]
            # Text that is part of the code's fixed template ("fie" in "field") isn't from the section.
            assert text not in message.replace(error.field, "") or text in str(error.msg)  # type: ignore[attr-defined]
