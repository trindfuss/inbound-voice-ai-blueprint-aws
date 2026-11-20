"""owlbank.pii: Comprehend via a botocore Stubber, masking, config validation, the guard."""

from __future__ import annotations

import logging
from typing import Any

import boto3
import pytest
from botocore.exceptions import ReadTimeoutError
from botocore.stub import Stubber

from owlbank import pii
from owlbank.pii import (
    DEFAULT_CONFIG,
    ComprehendPiiRedactor,
    Entity,
    FakePiiRedactor,
    PiiCheckUnavailable,
    PiiGuard,
    RedactionConfig,
    RedactionConfigError,
    mask_entities,
    parse_redaction_config,
)


def client() -> Any:
    return boto3.client("comprehend", region_name="us-east-1", aws_access_key_id="x",
                        aws_secret_access_key="x")


def ent(begin: int, end: int, etype: str, score: float = 0.99) -> dict[str, Any]:
    return {"BeginOffset": begin, "EndOffset": end, "Type": etype, "Score": score}


def stubbed(*responses: tuple[str, list[dict[str, Any]], str]) -> tuple[ComprehendPiiRedactor, Stubber]:
    c = client()
    stub = Stubber(c)
    for text, entities, lang in responses:
        stub.add_response("detect_pii_entities", {"Entities": entities},
                          {"Text": text, "LanguageCode": lang})
    stub.activate()
    return ComprehendPiiRedactor(c), stub


# --------------------------------------------------------------------------- Comprehend


@pytest.mark.parametrize("lang,code", [("en-US", "en"), ("es-US", "es"), ("es", "es"),
                                       ("fr-FR", "en"), ("", "en")])
def test_language_mapping(lang: str, code: str) -> None:
    text = "my ssn is 123-45-6789"
    r, stub = stubbed((text, [ent(10, 21, "SSN")], code))
    out = r.redact(text, language=lang)
    assert out.text == "my ssn is [SSN]" and out.entity_types == ("SSN",)
    stub.assert_no_pending_responses()


def test_only_configured_types_above_threshold_are_masked() -> None:
    text = "Thomas card 4111111111111111 pin 1234 call me"
    entities = [ent(0, 6, "NAME"), ent(12, 28, "CREDIT_DEBIT_NUMBER"),
                ent(33, 37, "PIN", score=0.49)]
    r, _ = stubbed((text, entities, "en"))
    out = r.redact(text, language="en-US")
    assert out.text == "Thomas card [CREDIT_DEBIT_NUMBER] pin 1234 call me"
    assert out.entity_types == ("CREDIT_DEBIT_NUMBER",)
    assert pii.DEFAULT_MIN_CONFIDENCE == 0.5  # PIN at 0.49 is below it


def test_threshold_is_inclusive() -> None:
    text = "pin 1234"
    r, _ = stubbed((text, [ent(4, 8, "PIN", score=0.5)], "en"))
    assert r.redact(text, language="en-US").text == "pin [PIN]"


def test_overlapping_and_adjacent_spans_merge() -> None:
    text = "acct 1234 5678 9012 done"
    # overlapping (5-14, 10-19) and adjacent-through-a-space spans become one mask
    entities = [ent(5, 14, "BANK_ACCOUNT_NUMBER"), ent(10, 19, "CREDIT_DEBIT_NUMBER"),
                ent(15, 19, "PIN")]
    r, _ = stubbed((text, entities, "en"))
    out = r.redact(text, language="en-US")
    assert out.text == "acct [BANK_ACCOUNT_NUMBER] done"
    assert out.entity_types == ("BANK_ACCOUNT_NUMBER", "CREDIT_DEBIT_NUMBER", "PIN")


def test_separate_spans_stay_separate() -> None:
    cfg = DEFAULT_CONFIG
    out = mask_entities("a 123-45-6789 and 4111", [Entity(2, 13, "SSN"), Entity(18, 22, "PIN")], cfg)
    assert out.text == "a [SSN] and [PIN]"


def test_blank_text_makes_no_call() -> None:
    r, stub = stubbed()  # any call would fail: no responses queued
    for text in ("", "   ", "\n\t"):
        assert r.redact(text, language="en-US").text == text
    stub.assert_no_pending_responses()


def test_empty_entity_types_makes_no_call() -> None:
    r, _ = stubbed()
    cfg = RedactionConfig(entity_types=())
    assert r.redact("ssn 123-45-6789", language="en-US", config=cfg).text == "ssn 123-45-6789"


def test_long_text_is_chunked_on_whitespace_with_offsets() -> None:
    words = ["alpha", "beta", "123-45-6789", "gamma", "delta"]
    text = " ".join(words)
    pieces = pii.chunks(text, max_bytes=12)
    assert "".join(p for _, p in pieces) == text
    assert all(len(p.encode()) <= 12 for _, p in pieces)
    assert all(text[o:o + len(p)] == p for o, p in pieces)
    c = client()
    stub = Stubber(c)
    for offset, piece in pieces:
        found = [ent(piece.index("123"), piece.index("123") + 11, "SSN")] if "123" in piece else []
        stub.add_response("detect_pii_entities", {"Entities": found}, {"Text": piece, "LanguageCode": "en"})
    stub.activate()
    out = ComprehendPiiRedactor(c, max_bytes=12).redact(text, language="en-US")
    assert out.text == "alpha beta [SSN] gamma delta"
    stub.assert_no_pending_responses()


def test_chunks_respect_utf8_bytes_and_hard_cut() -> None:
    text = "é" * 10  # 2 bytes each, no whitespace
    pieces = pii.chunks(text, max_bytes=5)
    assert [p for _, p in pieces] == ["éé", "éé", "éé", "éé", "éé"]
    assert pii.COMPREHEND_MAX_BYTES < 100_000


@pytest.mark.parametrize("error", ["ThrottlingException", "InternalServerException",
                                   "TextSizeLimitExceededException", "AccessDeniedException"])
def test_service_errors_raise_pii_check_unavailable(error: str) -> None:
    c = client()
    stub = Stubber(c)
    stub.add_client_error("detect_pii_entities", service_error_code=error)
    stub.activate()
    with pytest.raises(PiiCheckUnavailable) as ei:
        ComprehendPiiRedactor(c).redact("card 4111111111111111", language="en-US")
    assert "4111" not in str(ei.value)


def test_timeout_raises_pii_check_unavailable() -> None:
    class TimingOut:
        def detect_pii_entities(self, **_: Any) -> Any:
            raise ReadTimeoutError(endpoint_url="https://comprehend.us-east-1.amazonaws.com")

    with pytest.raises(PiiCheckUnavailable):
        ComprehendPiiRedactor(TimingOut()).redact("ssn 123-45-6789", language="en-US")


def test_malformed_response_raises_pii_check_unavailable() -> None:
    class Weird:
        def detect_pii_entities(self, **_: Any) -> Any:
            return {"Entities": [{"Type": "SSN"}]}

    with pytest.raises(PiiCheckUnavailable):
        ComprehendPiiRedactor(Weird()).redact("ssn 123-45-6789", language="en-US")


def test_client_timeouts_are_short() -> None:
    assert pii.COMPREHEND_CONNECT_TIMEOUT_S <= 2 and pii.COMPREHEND_READ_TIMEOUT_S <= 3


def test_entity_type_list_matches_botocore() -> None:
    import botocore.session

    model = botocore.session.get_session().get_service_model("comprehend")
    assert set(model.shape_for("PiiEntityType").enum) - {"ALL"} == pii.COMPREHEND_PII_ENTITY_TYPES


# --------------------------------------------------------------------------- fake


def test_fake_detects_cards_ssns_and_labelled_numbers() -> None:
    f = FakePiiRedactor()
    text = ("card 4111 1111 1111 1111, ssn 123-45-6789, routing number 021000021, "
            "account number is 12345678, order 411111111112")
    out = f.redact(text, language="en-US").text
    assert "4111 1111" not in out and "123-45-6789" not in out
    assert "021000021" not in out and "12345678" not in out
    assert "order 411111111112" in out  # 12 digits: too short for a card, not an SSN shape
    assert f.calls == 1


def test_fake_can_fail() -> None:
    with pytest.raises(PiiCheckUnavailable):
        FakePiiRedactor(fail=True).redact("x 123-45-6789", language="en-US")


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize("raw,field", [
    ({"entity_types": ["SSN", "NOT_A_TYPE"]}, "entity_types"),
    ({"entity_types": "SSN"}, "entity_types"),
    ({"entity_types": ["ALL"]}, "entity_types"),
    ({"min_confidence": 1.5}, "min_confidence"),
    ({"min_confidence": -0.1}, "min_confidence"),
    ({"min_confidence": "abc"}, "min_confidence"),
    ({"min_confidence": True}, "min_confidence"),
    ({"min_confidence": float("nan")}, "min_confidence"),
    ({"mask": "[{type}:{text}]"}, "mask"),
    ({"mask": "{0}"}, "mask"),
    ({"mask": "{type"}, "mask"),
    ({"mask": ""}, "mask"),
    ({"on_failure": "ignore"}, "on_failure"),
    ({"enabled": "yes"}, "enabled"),
    ({"enabled": 1}, "enabled"),
    ({"enabled": None}, "enabled"),
    ({"entity_types": []}, "entity_types"),  # An empty list no longer means "off"
    ({"enabled": True, "entity_types": []}, "entity_types"),
    ({"apply_to": ["ai_turn"]}, "apply_to"),  # removed: now an unknown field
    ({"surprise": 1}, "surprise"),
])
def test_invalid_config_names_the_field(raw: dict[str, Any], field: str) -> None:
    with pytest.raises(RedactionConfigError) as ei:
        parse_redaction_config(raw)
    assert ei.value.field == field


def test_valid_config_and_defaults() -> None:
    cfg = parse_redaction_config({"enabled": True, "entity_types": ["SSN", "ADDRESS"],
                                  "min_confidence": "0.8", "mask": "<{type}>", "on_failure": "allow"})
    assert cfg == RedactionConfig(enabled=True, entity_types=("SSN", "ADDRESS"), min_confidence=0.8,
                                  mask="<{type}>", on_failure="allow")
    assert DEFAULT_CONFIG.enabled is True
    assert "apply_to" not in DEFAULT_CONFIG.to_doc() and DEFAULT_CONFIG.to_doc()["enabled"] is True
    assert parse_redaction_config({}) == DEFAULT_CONFIG
    assert parse_redaction_config({"mask": "***"}).render_mask("SSN") == "***"
    assert parse_redaction_config(DEFAULT_CONFIG.to_doc()) == DEFAULT_CONFIG  # store round trip


def test_load_falls_back_and_logs_the_field_not_the_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="owlbank.pii")
    assert pii.load_redaction_config({"mask": "{secret-4111}"}) == DEFAULT_CONFIG
    assert pii.load_redaction_config(None) == DEFAULT_CONFIG
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 2 and "mask" in errors[0].getMessage()
    assert all("4111" not in r.getMessage() for r in caplog.records)


def test_disabled_config_may_have_any_valid_type_list() -> None:
    off = parse_redaction_config({"enabled": False, "entity_types": []})
    assert off.enabled is False and off.entity_types == ()
    kept = parse_redaction_config({"enabled": False, "entity_types": ["NAME"]})
    assert kept.enabled is False and kept.entity_types == ("NAME",)
    assert parse_redaction_config({"enabled": False}).entity_types == DEFAULT_CONFIG.entity_types
    with pytest.raises(RedactionConfigError) as ei:  # still validated while off
        parse_redaction_config({"enabled": False, "entity_types": ["NOT_A_TYPE"]})
    assert ei.value.field == "entity_types"


def test_empty_types_while_enabled_loads_the_defaults_with_an_error(
        caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="owlbank.pii")
    assert pii.load_redaction_config({"enabled": True, "entity_types": []}) == DEFAULT_CONFIG
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and errors[0].field == "entity_types"


def test_an_unknown_field_name_that_is_not_an_identifier_is_not_logged(
        caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger="owlbank.pii")
    assert pii.load_redaction_config({"my card 4111 1111": 1}) == DEFAULT_CONFIG
    (error,) = caplog.records
    assert "4111" not in error.getMessage() and error.field == "(unknown field)"


# --------------------------------------------------------------------------- guard


SECRET = "my card is 4111 1111 1111 1111"


def test_guard_masks_every_surface_when_enabled() -> None:
    f = FakePiiRedactor()
    g = PiiGuard(f, RedactionConfig())
    for surface in (pii.SURFACE_AI_TURN, pii.SURFACE_ESCALATION_SUMMARY):
        assert g.text(surface, SECRET, language="es-US", withheld="W") == \
            "my card is [CREDIT_DEBIT_NUMBER]"
    assert f.languages == ["es", "es"]


@pytest.fixture
def fresh_off_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pii, "_warned_off", set())


def test_guard_disabled_makes_no_call_and_warns_once_per_component(
        caplog: pytest.LogCaptureFixture, fresh_off_warning: None) -> None:
    from owlbank import obs

    caplog.set_level(logging.WARNING, logger="owlbank.pii")
    f = FakePiiRedactor(fail=True)  # would withhold if it were called
    g = PiiGuard(f, RedactionConfig(enabled=False))
    for component in ("bridge", "bridge", "frontdoor"):
        with obs.call_context(component=component):
            for surface in (pii.SURFACE_AI_TURN, pii.SURFACE_ESCALATION_SUMMARY):
                assert g.text(surface, SECRET, language="en-US", withheld="W") == SECRET
    assert f.calls == 0
    off = [r for r in caplog.records if r.getMessage() == "transcript_redaction_off"]
    assert [(r.levelno, r.component) for r in off] == [(logging.WARNING, "bridge"),
                                                        (logging.WARNING, "frontdoor")]
    assert all("4111" not in r.getMessage() for r in caplog.records)


def test_guard_off_warning_is_not_repeated_when_switched_off_again(
        caplog: pytest.LogCaptureFixture, fresh_off_warning: None) -> None:
    """Once per process: a later off period in the same process doesn't warn again."""
    caplog.set_level(logging.WARNING, logger="owlbank.pii")
    state = {"cfg": RedactionConfig(enabled=False)}
    g = PiiGuard(FakePiiRedactor(), lambda: state["cfg"])
    g.text("ai_turn", SECRET, language="en-US", withheld="W")
    state["cfg"] = RedactionConfig()
    assert "4111" not in g.text("ai_turn", SECRET, language="en-US", withheld="W")
    state["cfg"] = RedactionConfig(enabled=False)
    g.text("ai_turn", SECRET, language="en-US", withheld="W")
    assert sum(r.getMessage() == "transcript_redaction_off" for r in caplog.records) == 1


def test_guard_withholds_on_failure(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    g = PiiGuard(FakePiiRedactor(fail=True))
    assert g.text("ai_turn", SECRET, language="en-US", withheld="[WITHHELD]") == "[WITHHELD]"
    assert all("4111" not in r.getMessage() for r in caplog.records)


def test_guard_allow_sends_text_and_warns_each_time(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="owlbank.pii")
    g = PiiGuard(FakePiiRedactor(fail=True), RedactionConfig(on_failure="allow"))
    for _ in range(2):
        assert g.text("ai_turn", SECRET, language="en-US", withheld="W") == SECRET
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "unredacted" in r.getMessage()]
    assert len(warnings) == 2 and all("4111" not in r.getMessage() for r in warnings)


def test_guard_treats_a_redactor_bug_as_an_outage() -> None:
    class Broken:
        def redact(self, *a: Any, **k: Any) -> Any:
            raise KeyError("bug")

    assert PiiGuard(Broken()).text("ai_turn", SECRET, language="en-US", withheld="W") == "W"


def test_guard_follows_the_policy_config() -> None:
    class P:
        redaction = DEFAULT_CONFIG

    p = P()
    g = PiiGuard.for_policy(FakePiiRedactor(), p)
    assert "4111" not in g.text("ai_turn", SECRET, language="en-US", withheld="W")
    p.redaction = RedactionConfig(entity_types=("SSN",))
    assert g.text("ai_turn", SECRET, language="en-US", withheld="W") == SECRET
    p.redaction = RedactionConfig(enabled=False)
    assert g.text("ai_turn", SECRET, language="en-US", withheld="W") == SECRET
