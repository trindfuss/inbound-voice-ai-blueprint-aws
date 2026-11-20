"""Front-door building blocks: spoken money/digits (with round-trip PBT against the slot
engine's amount parser), back-end response parsing, and URL reconstruction."""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lambdas.frontdoor import http
from lambdas.frontdoor.backend import (
    LambdaBackendClient, parse_lookup, parse_send_otp, parse_tool, parse_verify,
)
from lambdas.frontdoor.fakes import FakeLambdaClient
from lambdas.frontdoor.speech import digits_words, money_words
from owlbank.slots.normalize import normalize, parse_amount

from tests.component.test_frontdoor_support import HOST, STAGE, make_event


# --------------------------------------------------------------------------- speech


@pytest.mark.parametrize("amount,lang,text", [
    ("1234.56", "en-US", "one thousand two hundred thirty-four dollars and fifty-six cents"),
    ("1310.06", "en-US", "one thousand three hundred ten dollars and six cents"),
    ("0", "en-US", "zero dollars"),
    ("0.50", "en-US", "fifty cents"),
    ("1.01", "en-US", "one dollar and one cent"),
    ("8420.00", "en-US", "eight thousand four hundred twenty dollars"),
    ("1234.56", "es-US", "mil doscientos treinta y cuatro dólares con cincuenta y seis centavos"),
    ("21.00", "es-US", "veintiún dólares"),
    ("1.00", "es-US", "un dólar"),
    ("100", "es-US", "cien dólares"),
    ("101", "es-US", "ciento un dólares"),
    ("0.01", "es-US", "un centavo"),
    ("2000000", "es-US", "dos millones de dólares"),
    ("1000000", "en-US", "one million dollars"),
])
def test_money_words(amount, lang, text):
    assert money_words(amount, lang) == text


@pytest.mark.parametrize("bad", ["-1", "abc", "1e12", "NaN"])
def test_money_words_rejects_negative_and_junk(bad):
    with pytest.raises(ValueError):
        money_words(bad, "en-US")


def test_digits_words():
    assert digits_words("4821", "en-US") == "four, eight, two, one"
    assert digits_words("7730", "es-US") == "siete, siete, tres, cero"


@given(cents=st.integers(min_value=0, max_value=99_999_999), lang=st.sampled_from(["en-US", "es-US"]))
def test_money_words_round_trip_through_the_slot_parser(cents, lang):
    """What we speak is what a caller saying it back would be understood as (amounts < 1M)."""
    amount = Decimal(cents) / 100
    assert parse_amount(normalize(money_words(amount, lang), lang), lang) == amount


@given(d=st.text(alphabet="0123456789", min_size=1, max_size=8))
def test_digits_words_has_one_word_per_digit(d):
    assert len(digits_words(d, "en-US").split(", ")) == len(d)


# --------------------------------------------------------------------------- backend parsing


def test_parse_verify_shapes():
    ok = parse_verify({"result": "verified", "level": 2, "customer_id": "C-1", "token": "t",
                       "factors_verified": ["ani_match", "zip", "ssn4"], "step_up": True})
    assert ok.status == "verified" and ok.level == 2 and ok.factors_verified[-1] == "ssn4"
    assert "t" not in repr(ok).split("token")[-1][:6] and "token" not in repr(ok)
    assert parse_verify({"result": "verified", "level": 1}).status == "error"  # no token
    assert parse_verify({"result": "not_verified", "reason": "ambiguous"}).status == "ambiguous"
    assert parse_verify({"result": "not_verified", "reason": "mismatch",
                         "attempts_remaining": 2}).status == "not_verified"
    assert parse_verify({"result": "not_verified", "reason": "mismatch",
                         "attempts_remaining": 0}).status == "locked"
    assert parse_verify({"result": "not_verified", "reason": "no_candidates"}).status == "locked"
    for r in ("rate_limited",):
        assert parse_verify({"result": r}).status == "locked"
    for r in ("invalid_level", "unavailable", "bad_request", "???"):
        assert parse_verify({"result": r}).status == "error"
    assert parse_verify(None).status == "error"


def test_parse_lookup_and_tool_shapes():
    assert parse_lookup({"tool": "ani_lookup", "data": {"candidates": ["C-1", "C-2"],
                                                        "withheld": False}}).candidate_ids == ("C-1", "C-2")
    assert parse_lookup({"error": "UNAVAILABLE", "tool": "ani_lookup"}).status == "error"
    assert parse_lookup(None).status == "error"
    assert parse_tool({"tool": "get_balance", "data": {"last4": "1"}}).data == {"last4": "1"}
    assert parse_tool({"error": "NOT_FOUND", "tool": "get_balance"}).status == "not_found"
    assert parse_tool({"error": "AUTH_LEVEL_INSUFFICIENT", "tool": "x"}).status == "auth_insufficient"
    assert parse_tool({"error": "UNAVAILABLE"}).status == "error"


@pytest.mark.parametrize("payload, status", [
    ({"result": "sent"}, "sent"),
    ({"result": "escalate", "reason": "otp_unavailable"}, "escalate"),
    ({"result": "escalate", "reason": "verification_risk"}, "error"),  # only otp_unavailable
    ({"result": "escalate"}, "error"),
    ({"result": "unavailable"}, "error"),
    ({"result": "rate_limited"}, "error"),
    ({"result": "not_verified"}, "error"),
    (None, "error"), ("sent", "error"), ([], "error"),
])
def test_parse_send_otp(payload, status):
    assert parse_send_otp(payload) == status


def test_lambda_client_sends_the_agreed_events_and_survives_failures():
    lam = FakeLambdaClient({"a": lambda e: {"result": "sent"}, "b": lambda e: {"data": {"candidates": []}}})
    c = LambdaBackendClient(lam, auth_function="a", bank_function="b")
    assert c.send_otp(call_sid="CAx", ani="+1", language="es-US", stir_verstat="V") == "sent"
    # No identity_source for an ANI call (absent = ani), so an auth service
    # from before it was added still accepts it during the update.
    assert lam.payloads[-1] == {"op": "send_otp", "call_sid": "CAx", "ani": "+1",
                                "language": "es-US", "stir_verstat": "V"}
    c.send_otp(call_sid="CAx", ani="+1", identity_source="keyed_phone")
    assert lam.payloads[-1]["identity_source"] == "keyed_phone"
    assert c.ani_lookup(call_sid="CAx", ani="+1").status == "ok"
    assert lam.payloads[-1] == {"op": "ani_lookup", "token": None, "call_sid": "CAx",
                                "args": {"ani": "+1"}}
    c.verify(call_sid="CAx", ani="+1", factors={"zip": "x"}, target_level=1)
    assert set(lam.payloads[-1]) == {"op", "call_sid", "ani", "factors", "target_level",
                                     "stir_verstat"}  # ani is the default, left out
    c.verify(call_sid="CAx", ani="+1", factors={"zip": "x"}, target_level=1,
             identity_source="keyed_phone")
    assert lam.payloads[-1]["identity_source"] == "keyed_phone"
    c.precheck(call_sid="CAx", ani="+1", target_level=2)
    assert "identity_source" not in lam.payloads[-1]
    c.precheck(call_sid="CAx", ani="+1", target_level=2, identity_source="keyed_phone")
    assert lam.payloads[-1] == {"op": "precheck", "call_sid": "CAx", "ani": "+1", "target_level": 2,
                                "stir_verstat": None, "identity_source": "keyed_phone"}
    c.verify(call_sid="CAx", ani="+1", factors={"zip": "x"}, target_level=1)
    assert lam.payloads[-1]["stir_verstat"] is None  # absent: the auth service treats it as untrusted
    lam.fail_ops.add("verify")
    assert c.verify(call_sid="CAx", ani="+1", factors={}, target_level=1).status == "error"
    lam.error_ops.add("get_balance")
    assert c.get_balance(token="t", call_sid="CAx", account="checking").status == "error"


# --------------------------------------------------------------------------- URL


def test_public_url_keeps_stage_and_drops_default_port():
    assert http.public_url(make_event("factor", {})) == f"https://{HOST}{STAGE}/voice/factor"


def test_public_url_without_request_context_path_uses_stage():
    event = make_event("factor", {})
    del event["requestContext"]["path"]
    assert http.public_url(event) == f"https://{HOST}/prod/voice/factor"


def test_route_of():
    assert http.route_of("/prod/voice/connect-action") == ("/prod", "connect-action")
    assert http.route_of("/voice/status") == ("", "status")
    assert http.route_of("/prod/voice/twiml") is None
