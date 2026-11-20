"""Basic password rotation on the request path: the previous password is accepted during the
overlap, the Twilio signature is checked with the password that matched, a miss triggers one
rate-limited forced re-read, and no log line carries a password."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from twilio.request_validator import RequestValidator

from lambdas.frontdoor import handler, http
from owlbank import secrets as sm
from owlbank.basic_auth import BasicSecret
from tests.component.test_frontdoor_orchestrator_sink import BODY, PATH, sink_event
from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    AUTH_TOKEN, BASIC, CLOCK, HOST, STAGE, THOMAS, basic_header, fd, make_event, sid, sign,
)

USER = BASIC[0]
CURRENT = "c" * 40
PREVIOUS = "a" * 40
OTHER = "b" * 40
PARAMS = {"CallSid": sid(1), "From": THOMAS, "To": "+12025550100", "AccountSid": "AC" + "b" * 32}
FORM_PATH = f"{STAGE}/voice/incoming"


def secret(until: float | None = CLOCK + 3600, previous: str | None = PREVIOUS) -> BasicSecret:
    return BasicSecret(USER, CURRENT, previous, until if previous else None)


def reader(value: BasicSecret, forced: BasicSecret | None = None, calls: list[bool] | None = None):
    def read(force: bool) -> BasicSecret:
        if calls is not None:
            calls.append(force)
        return forced if (force and forced is not None) else value
    return read


def form_event(auth_password: str, sig_password: str) -> dict[str, Any]:
    url = f"https://{USER}:{sig_password}@{HOST}{FORM_PATH}"
    return make_event("incoming", PARAMS, auth=basic_header(USER, auth_password),
                      signature=sign(url, PARAMS))


def callback_event(auth_password: str, sig_password: str) -> dict[str, Any]:
    validator = RequestValidator(AUTH_TOKEN)
    query = urlencode({"bodySHA256": validator.compute_hash(BODY)})
    url = f"https://{USER}:{sig_password}@{HOST}{PATH}?{query}"
    return sink_event(auth=basic_header(USER, auth_password),
                      signature=validator.compute_signature(url, {}))


def run_form(event, read, now=CLOCK):
    return http.authenticate(event, auth_token=lambda: AUTH_TOKEN, basic_secret=read,
                             clock=lambda: now)


def run_callback(event, read, now=CLOCK):
    return http.authenticate_callback(event, auth_token=lambda: AUTH_TOKEN, basic_secret=read,
                                      clock=lambda: now)


ROUTES = [(form_event, run_form), (callback_event, run_callback)]
IDS = ["form", "callback"]


@pytest.mark.parametrize(("make", "run"), ROUTES, ids=IDS)
def test_previous_password_signed_url_is_accepted_during_the_overlap(make, run):
    run(make(PREVIOUS, PREVIOUS), reader(secret()))


@pytest.mark.parametrize(("make", "run"), ROUTES, ids=IDS)
def test_previous_password_is_rejected_after_the_overlap(make, run):
    s = secret(until=CLOCK + 10)
    with pytest.raises((http.Rejected, http.Unauthorized)):
        run(make(PREVIOUS, PREVIOUS), reader(s), now=CLOCK + 10)
    run(make(PREVIOUS, PREVIOUS), reader(s), now=CLOCK + 9)


@pytest.mark.parametrize(("make", "run"), ROUTES, ids=IDS)
def test_current_password_still_accepted_with_a_previous_one(make, run):
    run(make(CURRENT, CURRENT), reader(secret()))


@pytest.mark.parametrize(("make", "run"), ROUTES, ids=IDS)
@pytest.mark.parametrize(("auth_pw", "sig_pw"), [(PREVIOUS, CURRENT), (CURRENT, PREVIOUS)])
def test_signature_uses_the_password_that_matched(make, run, auth_pw, sig_pw):
    """A URL signed with the other password does not pass, whichever way round."""
    with pytest.raises((http.Rejected, http.Unauthorized)):
        run(make(auth_pw, sig_pw), reader(secret()))


def test_whole_form_path_accepts_previous_then_rejects_after_expiry(fd):
    fd.app.d.basic_secret = reader(secret(until=CLOCK + 5))
    ok = fd.app.handle(form_event(PREVIOUS, PREVIOUS))
    assert ok["statusCode"] == 200
    fd.app.d.clock = lambda: CLOCK + 5
    assert fd.app.handle(form_event(PREVIOUS, PREVIOUS)) == {"statusCode": 403, "headers": {}, "body": ""}


def test_previous_accepted_is_logged_with_the_call_and_no_password(fd, caplog):
    fd.app.d.basic_secret = reader(secret())
    fd.app.handle(form_event(PREVIOUS, PREVIOUS))
    fd.app.handle(form_event(CURRENT, CURRENT))
    events = [r for r in caplog.records if r.getMessage() == "twilio_basic_previous_accepted"]
    assert len(events) == 1  # only the previous-password request
    assert events[0].call_sid == PARAMS["CallSid"]


def test_no_log_line_holds_a_password(fd, caplog):
    fd.app.d.basic_secret = reader(secret(), forced=secret())
    fd.app.handle(form_event(PREVIOUS, PREVIOUS))
    fd.app.handle(form_event(CURRENT, CURRENT))
    fd.app.handle(form_event(OTHER, OTHER))  # wrong password: forced re-read, then refused

    def boom(force: bool) -> BasicSecret:
        if force:
            raise RuntimeError(f"cannot read {CURRENT} {PREVIOUS}")
        return secret()
    fd.app.d.basic_secret = boom
    fd.app.handle(form_event(OTHER, OTHER))
    for rec in caplog.records:
        text = json.dumps(rec.__dict__, default=repr) + rec.getMessage()
        assert CURRENT not in text and PREVIOUS not in text and OTHER not in text, rec.getMessage()
    assert repr(secret()).find(CURRENT) < 0 and repr(secret()).find(PREVIOUS) < 0


@pytest.mark.parametrize(("make", "run"), ROUTES, ids=IDS)
def test_a_miss_re_reads_once_and_accepts_a_newer_secret(make, run):
    calls: list[bool] = []
    stale = BasicSecret(USER, PREVIOUS)  # this instance's cache predates the rotation
    fresh = BasicSecret(USER, CURRENT)
    run(make(CURRENT, CURRENT), reader(stale, forced=fresh, calls=calls))
    assert calls == [False, True]


@pytest.mark.parametrize(("make", "run"), ROUTES, ids=IDS)
def test_a_match_does_not_force_a_read(make, run):
    calls: list[bool] = []
    run(make(CURRENT, CURRENT), reader(secret(), calls=calls))
    assert calls == [False]


@pytest.mark.parametrize(("make", "run"), ROUTES, ids=IDS)
def test_a_failed_forced_read_is_an_auth_rejection_not_unverifiable(make, run, caplog):
    def read(force: bool) -> BasicSecret:
        if force:
            raise RuntimeError("secrets manager down")
        return secret()
    with pytest.raises((http.Rejected, http.Unauthorized)) as info:
        run(make(OTHER, OTHER), read)
    assert str(info.value) in ("auth", "basic")
    assert "unverifiable" not in {getattr(r, "reason", None) for r in caplog.records}


@pytest.mark.parametrize(("make", "run"), ROUTES, ids=IDS)
def test_an_unreadable_secret_is_still_unverifiable(make, run):
    def read(force: bool) -> BasicSecret:
        raise RuntimeError("down")
    with pytest.raises(http.Rejected) as info:
        run(make(CURRENT, CURRENT), read)
    assert str(info.value) == "unverifiable"


# --- the real secret reader: at most one forced read per ARN per 10 s ---

class Scripted:
    def __init__(self, *values: dict[str, Any]) -> None:
        self.values, self.calls, self.fail = list(values), 0, False

    def get_secret_value(self, SecretId: str) -> dict[str, Any]:  # noqa: N803
        self.calls += 1
        if self.fail:
            raise RuntimeError("down")
        return {"SecretString": json.dumps(self.values[min(self.calls - 1, len(self.values) - 1)])}


@pytest.fixture
def clock(monkeypatch):
    sm.clear_cache()
    t = [1000.0]
    monkeypatch.setattr(sm, "clock", lambda: t[0])
    yield t
    sm.clear_cache()


def real_reader(client):
    def read(force: bool) -> BasicSecret:
        return handler.basic_secret_of(sm.get_secret("arn:basic", client=client, logical="basic_auth",
                                                     force=force))
    return read


BEFORE_ROTATION = {"user": USER, "password": PREVIOUS}
AFTER_ROTATION = {"user": USER, "password": CURRENT, "previous_password": PREVIOUS, "previous_until": int(CLOCK) + 3600}


def test_forced_re_read_happens_at_most_once_per_arn_per_10_seconds(clock):
    client = Scripted(BEFORE_ROTATION, BEFORE_ROTATION)
    read = real_reader(client)
    for _ in range(5):  # junk requests: each misses
        with pytest.raises(http.Rejected):
            run_form(form_event(OTHER, OTHER), read)
        clock[0] += 1.0
    assert client.calls == 2  # one read, one forced read
    clock[0] += 10.0
    with pytest.raises(http.Rejected):
        run_form(form_event(OTHER, OTHER), read)
    assert client.calls == 3


def test_real_reader_picks_up_a_rotation_and_survives_a_failed_forced_read(clock):
    client = Scripted(BEFORE_ROTATION, AFTER_ROTATION)
    read = real_reader(client)
    run_form(form_event(PREVIOUS, PREVIOUS), read)  # cached BEFORE_ROTATION
    run_form(form_event(CURRENT, CURRENT), read)  # miss, forced read gets AFTER_ROTATION
    clock[0] += 11.0
    client.fail = True
    with pytest.raises(http.Rejected) as info:
        run_form(form_event(OTHER, OTHER), read)  # forced read fails: cached AFTER_ROTATION, plain refusal
    assert str(info.value) == "auth"


def test_a_malformed_previous_pair_means_no_previous_password():
    s = handler.basic_secret_of({"user": USER, "password": CURRENT, "previous_password": PREVIOUS})
    assert s.previous_password is None and s.previous_until is None


# --- property: authenticated iff the match says so; the signature uses the matched password ---

HEX = st.text(alphabet="0123456789abcdef", min_size=8, max_size=12)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(current=HEX, previous=st.none() | HEX, until=st.integers(0, 2000),
       offset=st.integers(-1000, 1000), header_pw=st.sampled_from(["current", "previous", "other"]),
       sig_pw=st.sampled_from(["current", "previous", "other"]))
def test_property_authenticated_iff_matched_and_signed_with_the_matched_password(
        current, previous, until, offset, header_pw, sig_pw):
    now = CLOCK + offset
    s = BasicSecret(USER, current, previous, CLOCK + until if previous is not None else None)
    pick = {"current": current, "previous": previous if previous is not None else "0" * 7,
            "other": "f" * 14}
    h, g = pick[header_pw], pick[sig_pw]
    basic_ok = h == current or (previous is not None and h == previous and now < CLOCK + until)
    expected = basic_ok and g == h
    for make, run in ROUTES:
        try:
            run(make(h, g), reader(s), now=now)
            got = True
        except (http.Rejected, http.Unauthorized):
            got = False
        assert got == expected
