"""Endpoint authentication at the handler, handler errors, and
the no-leak check at the handler boundary as a property test."""

from __future__ import annotations

from tests.logutil import lines as lines_of

import json
import logging
import xml.etree.ElementTree as ET

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from owlbank import callstate
from owlbank.policy import Policy
from owlbank.segments import clear_sensitive
from owlbank.slots import FakeResolver, ResolverOutput

from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    AUTH_TOKEN, BASIC, DATA, HOST, STAGE, THOMAS, all_fixture_factor_values, assert_no_leak,
    assert_scripted,
    NO_AUTH, URL_FORMS, basic_header, build, fd, make_event, says, server_noise, sid, sign, signed_url,
    twilio_post,
)

PARAMS = {"CallSid": sid(1), "From": THOMAS, "To": "+12025550100", "AccountSid": "AC" + "b" * 32}


def _forbidden(resp):
    assert resp == {"statusCode": 403, "headers": {}, "body": ""}


CHALLENGE = {"statusCode": 401, "headers": {"WWW-Authenticate": 'Basic realm="owlbank"'}, "body": ""}


def test_valid_request_passes(fd):
    resp = fd.app.handle(make_event("incoming", PARAMS))
    assert resp["statusCode"] == 200 and resp["headers"]["Content-Type"].startswith("text/xml")


def test_bad_signature_is_403(fd):
    _forbidden(fd.app.handle(make_event("incoming", PARAMS, signature="AAAA" + "=" * 24)))


def test_signature_with_other_auth_token_is_403(fd):
    url = f"https://{HOST}{STAGE}/voice/incoming"
    _forbidden(fd.app.handle(make_event("incoming", PARAMS,
                                        signature=sign(url, PARAMS, "another-token"))))


def test_tampered_parameter_is_403(fd):
    event = make_event("incoming", PARAMS)
    event["body"] = event["body"].replace("12025550101", "12025550102")
    _forbidden(fd.app.handle(event))


def test_signature_over_other_route_is_403(fd):
    event = make_event("incoming", PARAMS)
    event["requestContext"]["path"] = f"{STAGE}/voice/language"
    _forbidden(fd.app.handle(event))


def test_wrong_basic_credentials_are_403(fd):
    _forbidden(fd.app.handle(make_event("incoming", PARAMS, auth=basic_header(BASIC[0], "nope"))))
    _forbidden(fd.app.handle(make_event("incoming", PARAMS, auth="Bearer " + AUTH_TOKEN)))


def test_request_without_authorization_gets_the_basic_challenge(fd):
    """Twilio's first request carries no credentials; the 401 + WWW-Authenticate makes it repeat."""
    assert fd.app.handle(make_event("incoming", PARAMS, auth=NO_AUTH)) == CHALLENGE


def test_the_challenge_is_answered_before_anything_else_is_looked_at(fd):
    """No secret is read, no body parsed, no route checked, nothing stored."""
    def boom() -> str:
        raise AssertionError("secret read")
    fd.app.d.auth_token = boom
    fd.app.d.basic_credentials = boom  # type: ignore[assignment]
    fd.app.d.basic_secret = boom  # type: ignore[assignment]
    event = make_event("incoming", PARAMS, auth=NO_AUTH, signature="")
    event["body"] = "%%%not a body%%%"
    event["requestContext"]["path"] = f"{STAGE}/twiml"  # not even a known route
    assert fd.app.handle(event) == CHALLENGE
    event["headers"]["Authorization"] = ""  # an empty header is no header (the WAF rule: size > 0)
    assert fd.app.handle(event) == CHALLENGE
    assert callstate.load(fd.store, PARAMS["CallSid"]) is None


def test_missing_signature_with_valid_basic_is_403(fd):
    event = make_event("incoming", PARAMS)
    del event["headers"]["X-Twilio-Signature"]
    _forbidden(fd.app.handle(event))
    _forbidden(fd.app.handle(make_event("incoming", PARAMS, signature="")))


@pytest.mark.parametrize("form", URL_FORMS)
def test_twilio_dance_with_either_signed_url_form(fd, form):
    """The first request is challenged; the repeat with Authorization and a signature over the
    URL without or with ``user:password@`` is accepted."""
    resp = twilio_post(fd.app.handle, "incoming", PARAMS, form=form)
    assert resp["statusCode"] == 200 and resp["headers"]["Content-Type"].startswith("text/xml")


def test_signature_over_credentials_for_other_credentials_is_403(fd):
    """Only the credentials from the verified header are tried: not any user:pass@ form."""
    url = f"https://other:creds@{HOST}{STAGE}/voice/incoming"
    _forbidden(fd.app.handle(make_event("incoming", PARAMS, signature=sign(url, PARAMS))))


def test_signature_over_other_credentials_form_with_wrong_basic_is_403(fd):
    event = make_event("incoming", PARAMS, form="credentials",
                       auth=basic_header(BASIC[0], "nope"))
    _forbidden(fd.app.handle(event))


def test_credentials_in_the_signed_url_do_not_help_a_wrong_signature(fd):
    url = signed_url(f"{STAGE}/voice/incoming", "credentials")
    _forbidden(fd.app.handle(make_event("incoming", PARAMS, signature=sign(url, PARAMS, "another-token"))))


def test_unknown_route_or_method_is_403(fd):
    event = make_event("incoming", PARAMS)
    event["requestContext"]["path"] = f"{STAGE}/twiml"
    _forbidden(fd.app.handle(event))
    event = make_event("incoming", PARAMS)
    event["httpMethod"] = "GET"
    _forbidden(fd.app.handle(event))


def test_unreadable_secret_fails_closed(fd):
    def broken() -> str:
        raise RuntimeError("secrets manager down")
    fd.app.d.auth_token = broken
    _forbidden(fd.app.handle(make_event("incoming", PARAMS)))


def test_rejections_are_logged_as_security_events_without_header_values(fd, caplog):
    caplog.set_level(logging.DEBUG)
    fd.app.handle(make_event("incoming", PARAMS, auth=basic_header(BASIC[0], "nope")))
    rec = [r for r in caplog.records if r.getMessage() == "security: rejected request"]
    assert rec and rec[0].reason == "basic_auth"
    text = " ".join(str(v) for r in caplog.records for v in r.__dict__.values())
    assert "nope" not in text and BASIC[1] not in text


def test_base64_body_is_accepted(fd):
    import base64
    event = make_event("incoming", PARAMS)
    event["body"] = base64.b64encode(event["body"].encode()).decode()
    event["isBase64Encoded"] = True
    assert fd.app.handle(event)["statusCode"] == 200


# --------------------------------------------------------------------------- handler errors


def test_backend_crash_gives_apology_and_escalation(fd):
    fd.lam.error_ops.add("verify")  # FunctionError: parsed as an error, not raised
    fd.start(sid(1), THOMAS)
    root = fd.factor(sid(1), "94105")
    assert_scripted(root, "I'm sorry, something went wrong on our end.",
                    "I am the OwlBank virtual agent. How can I help you today?")


def test_unexpected_exception_gives_apology_twiml_not_an_error(fd):
    fd.start(sid(1), THOMAS)

    def boom(**_):
        raise RuntimeError("unexpected 94105")  # even an exception carrying a value
    fd.app.d.backend.verify = boom
    fd.sent_factors.add("94105")
    resp = fd.post("factor", sid(1), Digits="94105")
    assert resp["statusCode"] == 200
    root = ET.fromstring(resp["body"].encode())
    assert says(root)[0] == "I'm sorry, something went wrong on our end."
    assert root.find("Enqueue") is not None  # fallback: escalate to a human


def test_a_handler_error_logs_its_stack_and_decision_but_not_its_message(fd, caplog):
    fd.start(sid(1), THOMAS)
    secret = "unexpected 94105 for +12025550123"

    def boom(**_):
        raise RuntimeError(secret)
    fd.app.d.backend.verify = boom
    fd.sent_factors.add("94105")
    fd.post("factor", sid(1), Digits="94105")
    (rec,) = [r for r in caplog.records if r.getMessage() == "handler error"]
    f = lines_of([rec], rec.name)[0]
    assert rec.levelno == logging.ERROR and f["decision"] == "apology_and_escalate" and f["route"] == "factor"
    assert f["error_class"] == "RuntimeError" and "error_message" not in f and rec.exc_info is None
    assert f["stack"].startswith("Traceback") and "boom" in f["stack"] and "RuntimeError" in f["stack"]
    assert "94105" not in f["stack"] and "12025550123" not in f["stack"] and "unexpected" not in f["stack"]


def test_error_without_call_state_apologises_and_hangs_up(fd):
    root = fd.xml("factor", sid(9), Digits="1")
    assert says(root) == ["I'm sorry, something went wrong on our end.",
                          "Thank you for calling OwlBank. Goodbye."]
    assert root.find("Hangup") is not None


def test_status_errors_still_return_empty_twiml(fd):
    def boom(*a, **k):
        raise RuntimeError("store down")
    fd.store.query = boom
    resp = fd.post("status", sid(1), CallStatus="completed")
    assert resp["statusCode"] == 200 and "<Response />" in resp["body"]


# --------------------------------------------------------------------------- (PBT)

_SEED = json.loads((DATA / "policy.json").read_text())
_ZIP_SENSITIVE = Policy.from_dict({**_SEED, "slots": {**_SEED["slots"],
                                                      "zip": {**_SEED["slots"]["zip"], "sensitive": True}}})

_digits = st.text(alphabet="0123456789", min_size=5, max_size=5)
_words = st.lists(st.sampled_from(["actually", "my", "balance", "please", "what", "is",
                                   "agent", "the", "hours", "zip"]), max_size=6)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(zip_value=_digits, words=_words, kind=st.sampled_from(["intent_switch", "none",
                                                               "escalate", "exact", "inferred"]),
       spoken=st.booleans())
def test_no_factor_value_in_any_response_log_or_store(caplog, zip_value, words, kind, spoken):
    """For any ZIP the caller enters or says (embedded in arbitrary speech), and any resolver
    outcome, the value never appears in a TwiML body, a log record or a stored item.

    With ZIP flagged ``sensitive``. The seed leaves ZIP unflagged, so its words reach
    the AI unchanged (``test_frontdoor_login``); this property is about a sensitive factor."""
    assume(zip_value not in server_noise().replace(" ", "|"))  # e.g. "86400" in a TTL
    caplog.clear()
    caplog.set_level(logging.DEBUG)
    h = build(resolver=FakeResolver([ResolverOutput(kind=kind, value=zip_value)]))
    h.app.d.policy = _ZIP_SENSITIVE
    call = sid(3)
    try:
        h.start(call, THOMAS)
        if spoken:
            h.sent_factors.add(zip_value)
            text = " ".join([*words[:3], " ".join(zip_value), *words[3:]])
            h.post("factor", call, SpeechResult=text, Confidence="0.8")
        else:
            h.factor(call, zip_value)
        h.post("factor", call, Digits=zip_value)  # whatever state the flow is in now
        h.post("status", call, CallStatus="completed")
        values = all_fixture_factor_values() | ({zip_value} if len(set(zip_value)) > 1 else set())
        assert_no_leak(h, caplog.records, values)
    finally:
        clear_sensitive(call)


def test_rejections_emit_the_stable_auth_failed_line(fd, caplog) -> None:
    """The metric filter matches ``twilio_auth_failed`` (owlbank.security_log)."""
    import logging

    from tests.component.test_frontdoor_support import make_event

    caplog.set_level(logging.WARNING)
    ev = make_event("incoming", {"CallSid": "CA" + "e" * 32})
    ev["headers"] = {**ev["headers"], "X-Twilio-Signature": "bad"}
    assert fd.app.handle(ev)["statusCode"] == 403
    messages = [r.getMessage() for r in caplog.records]
    assert {"event": "twilio_auth_failed", "component": "frontdoor", "reason": "signature"} in lines_of(
        caplog.records, "owlbank.security")
    assert "security: rejected request" in messages  # the existing line stays


def test_challenge_is_logged_as_its_own_info_event_never_as_an_auth_failure(fd, caplog) -> None:
    """The auth-failure alarm filters on the term ``twilio_auth_failed``: the challenge, which
    every Twilio webhook starts with, must not contain it."""
    caplog.set_level(logging.DEBUG)
    assert fd.app.handle(make_event("incoming", PARAMS, auth=NO_AUTH)) == CHALLENGE
    sec = [r for r in caplog.records if r.name == "owlbank.security"]
    assert lines_of(sec, "owlbank.security") == [
        {"event": "twilio_basic_challenge", "component": "frontdoor", "reason": "basic", "challenge": True}]
    assert [r.levelno for r in sec] == [logging.INFO]
    text = " ".join(str(v) for r in caplog.records for v in r.__dict__.values()) + " ".join(
        r.getMessage() for r in caplog.records)
    assert "twilio_auth_failed" not in text and "rejected request" not in text


def test_wrong_credentials_log_the_auth_failed_line_with_reason_basic(fd, caplog) -> None:
    caplog.set_level(logging.WARNING)
    _forbidden(fd.app.handle(make_event("incoming", PARAMS, auth=basic_header(BASIC[0], "nope"))))
    assert {"event": "twilio_auth_failed", "component": "frontdoor", "reason": "basic"} in lines_of(
        caplog.records, "owlbank.security")


@pytest.mark.parametrize("form", URL_FORMS)
def test_the_validated_url_form_is_logged_by_name_only(fd, caplog, form) -> None:
    """First real call, see which form Twilio signs; no URL or credential on the line."""
    caplog.set_level(logging.DEBUG)
    assert fd.app.handle(make_event("incoming", PARAMS, form=form))["statusCode"] == 200
    lines = [ln for ln in lines_of(caplog.records, "owlbank.security") if ln["event"] == "twilio_signature_form"]
    assert lines == [{"event": "twilio_signature_form", "component": "frontdoor", "form": form}]
    text = " ".join(str(v) for r in caplog.records for v in r.__dict__.values())
    assert BASIC[1] not in text and BASIC[0] not in text and "https://" not in text
