"""Spoken digits finish early via partial results.

One verification per Gather, whatever the order of partials, the final webhook, the early
redirect and Twilio's replays, and identical TwiML for every answer to that Gather. Early
only on a unique full-length deterministic parse; partial text and digits never in a log, the
journal, the store or a response. ``earlyDigits`` off gives today's TwiML.
The no-leak scan runs at teardown by :func:`assert_no_leak`.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from urllib.parse import parse_qs, urlsplit

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.frontdoor import gathers
from lambdas.frontdoor.app import FrontDoorConfig
from lambdas.frontdoor.calls import FakeCallUpdater
from owlbank.segments import clear_sensitive
from owlbank.slots import SlotDecl, early_digits
from tests.component.test_frontdoor_support import (
    BASIC, CLOCK, HOST, NO_AUTH, STAGE, TEST_FACTORS, THOMAS, Harness, all_fixture_factor_values,
    assert_no_leak, build, make_event, relay, sid,
)

ZIP = TEST_FACTORS["C-1001"]["zip"]
EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ZIP_WORDS = " ".join(EN[int(c)] for c in ZIP)
PARTIAL_KEYS = {"slot", "chars", "ms_since_gather", "outcome"}


def _cleanup(h: Harness) -> None:
    for call in {k[0].split("#")[0] for k in h.store._tables["calls"]}:  # noqa: SLF001
        clear_sensitive(call)


def make(**config_kw):
    calls = FakeCallUpdater(fail=config_kw.pop("fail", False))
    h = build(config_kw={"accept_spoken_factors": True, **config_kw}, calls=calls)
    h.sent_factors.add(ZIP)
    return h, calls


@pytest.fixture
def early(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    h, calls = make()
    yield h, calls
    assert_no_leak(h, caplog.records, all_fixture_factor_values() | h.sent_factors)
    _no_partial_text(caplog.records)
    _cleanup(h)


def _no_partial_text(records) -> None:
    for r in records:
        blob = r.getMessage() + " " + " ".join(str(v) for v in r.__dict__.values())
        assert ZIP_WORDS not in blob and "nine four one" not in blob


def gather_of(root: ET.Element) -> ET.Element:
    g = root.find("Gather")
    assert g is not None, ET.tostring(root)
    return g


def gid_of(g: ET.Element) -> str:
    return parse_qs(urlsplit(g.get("action") or "").query)["g"][0]


def route_of(url: str) -> str:
    return url.rsplit("/voice/", 1)[1]


def partial(h: Harness, call: str, g: ET.Element, text: str, **extra: str) -> dict:
    resp = h.post(route_of(g.get("partialResultCallback") or ""), call,
                  UnstableSpeechResult=text, SequenceNumber="1", **extra)
    assert resp == {"statusCode": 204, "headers": {}, "body": ""}
    return resp


def redirect_route(twiml: str) -> str:
    url = ET.fromstring(twiml.encode()).find("Redirect").text or ""
    parts = urlsplit(url)
    assert parts.scheme == "https" and parts.hostname == HOST
    assert (parts.username, parts.password) == BASIC  # the URL carries Basic, like the webhooks
    assert parts.path == f"{STAGE}/voice/factor"
    return route_of(url)


def verifies(h: Harness) -> int:
    return h.lam.ops().count("verify")


def events(records, name: str) -> list[logging.LogRecord]:
    return [r for r in records if getattr(r, "event", None) == name]


# --------------------------------------------------------------------------- the Gather


def test_eligible_gather_reports_partials(early):
    h, _ = early
    g = gather_of(h.start(sid(1), THOMAS))
    gid = gid_of(g)
    assert g.get("input") == "dtmf speech"
    assert g.get("partialResultCallback") == f"{STAGE}/voice/partial?g={gid}"
    assert g.get("partialResultCallbackMethod") == "POST"
    assert h.session(sid(1)).flow["gather"]["id"] == gid
    assert gathers.load(h.store, sid(1), gid) is None  # nothing claimed yet


def _strip_ids(body: str) -> str:
    root = ET.fromstring(body.encode())
    for g in root.iter("Gather"):
        g.attrib.pop("partialResultCallback", None)
        g.attrib.pop("partialResultCallbackMethod", None)
        g.set("action", (g.get("action") or "").split("?")[0])
    return ET.tostring(root, encoding="unicode")


@pytest.mark.parametrize("why", ["setting_off", "no_updater"])
def test_off_gives_todays_twiml(caplog, why):
    """With earlyDigits off (or no Twilio updater) the TwiML is byte-for-byte today's."""
    caplog.set_level(logging.DEBUG)
    today = build(config_kw={"accept_spoken_factors": True})
    if why == "setting_off":
        off = build(config_kw={"accept_spoken_factors": True, "early_digits": False},
                    calls=FakeCallUpdater())
    else:
        off = build(config_kw={"accept_spoken_factors": True, "early_digits": True})
    on, _ = make()
    for h in (today, off, on):
        h.start(sid(2), THOMAS)
        h.xml("factor", sid(2), SpeechResult="what", Confidence="0.9")  # an unclear answer: a reprompt
    assert today.bodies == off.bodies
    assert "partialResultCallback" not in "".join(off.bodies)
    # The only difference with it on: the partial callback and the Gather id.
    assert [_strip_ids(b) for b in on.bodies if b.startswith("<?xml") or "<Gather" in b] == \
        [_strip_ids(b) for b in today.bodies if b.startswith("<?xml") or "<Gather" in b]
    for h in (today, off, on):
        _cleanup(h)


def test_from_env_reads_early_digits():
    base = {"OWLBANK_RELAY_WS_URL": "wss://x/ws"}
    assert FrontDoorConfig.from_env(base).early_digits is True
    assert FrontDoorConfig.from_env({**base, "OWLBANK_EARLY_DIGITS": "false"}).early_digits is False


def test_keypad_only_factors_untouched(caplog):
    caplog.set_level(logging.DEBUG)
    h = build(config_kw={"accept_spoken_factors": False}, calls=FakeCallUpdater())  # keypad only
    g = gather_of(h.start(sid(3), THOMAS))
    assert g.get("input") == "dtmf" and g.get("partialResultCallback") is None
    assert "?" not in (g.get("action") or "")
    _cleanup(h)


def test_step_up_digits_get_partials_but_yes_no_does_not(early):
    from tests.component.test_frontdoor_caller_id_trust import confirm_handoff, to_level3

    h, _ = early
    call = sid(4)
    h.login_thomas(call)
    h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    root = h.handoff(call, {"reason": "step-up", "tool": "make_payment", "journal": "jr-1"})
    assert gather_of(root).get("partialResultCallback")  # SSN last four: spoken digits, 4 long
    h.sent_factors |= {TEST_FACTORS["C-1001"]["ssn4"]}
    call2 = sid(14)
    to_level3(h, call2)
    g = gather_of(confirm_handoff(h, call2))  # the yes/no read-back
    assert "speech" in (g.get("input") or "")
    assert g.get("partialResultCallback") is None and "?" not in (g.get("action") or "")


# --------------------------------------------------------------------------- the early path


def test_complete_partial_finishes_early_once(early, caplog):
    h, calls = early
    call = sid(5)
    g = gather_of(h.start(call, THOMAS))
    partial(h, call, g, "nine four one")
    assert calls.updates == []
    partial(h, call, g, ZIP_WORDS)
    assert len(calls.updates) == 1 and calls.updates[0][0] == call
    partial(h, call, g, ZIP_WORDS + " yes")  # later partials: already claimed, no second update
    assert len(calls.updates) == 1
    assert verifies(h) == 0  # nothing verified on a partial
    route = redirect_route(calls.updates[0][1])
    first = h.post(route, call)
    assert first["statusCode"] == 200 and verifies(h) == 1
    root = ET.fromstring(first["body"].encode())
    assert relay(root) is not None and h.state(call).level == 1
    # Twilio's final webhook for the same Gather (or a retry of the redirect): same TwiML, no verify.
    final = h.post(route_of(g.get("action") or ""), call, SpeechResult=ZIP_WORDS, Confidence="0.9")
    again = h.post(route, call)
    assert final["body"] == first["body"] == again["body"]
    assert verifies(h) == 1
    outcomes = [r.outcome for r in events(caplog.records, "factor_partial")]
    assert outcomes == ["incomplete", "complete", "already_claimed"]
    for r in events(caplog.records, "factor_partial"):
        assert r.slot == "zip" and isinstance(r.chars, int)
    (fe,) = events(caplog.records, "factor_early")
    assert fe.slot == "zip" and fe.digits_present is True and fe.speech_timeout_ms == 3000
    ans = events(caplog.records, "factor_answer")   # one answer, by speech (a log line)
    assert len(ans) == 1 and ans[0].input_source == "speech"
    res = [e for e in h.journal(call) if e["type"] == "slot_resolution"]
    assert len(res) == 1 and res[0]["source"] == "speech"
    assert not [e for e in h.journal(call) if "partial" in str(e.get("type"))]


def test_ambiguous_partial_waits(early, caplog):
    h, calls = early
    call = sid(6)
    g = gather_of(h.start(call, THOMAS))
    partial(h, call, g, "one two three four five six seven eight nine one")
    partial(h, call, g, "nine four one oh five or one two three four five")
    assert calls.updates == []
    assert {r.outcome for r in events(caplog.records, "factor_partial")} <= {"ambiguous", "incomplete"}
    assert "ambiguous" in {r.outcome for r in events(caplog.records, "factor_partial")}


def test_final_before_early_claims_the_gather(early):
    h, calls = early
    call = sid(7)
    g = gather_of(h.start(call, THOMAS))
    final = h.post(route_of(g.get("action") or ""), call, SpeechResult=ZIP_WORDS, Confidence="0.9")
    assert verifies(h) == 1
    partial(h, call, g, ZIP_WORDS)  # a late partial for an answered Gather: nothing happens
    assert calls.updates == [] and verifies(h) == 1
    # A replayed final webhook: the same TwiML.
    again = h.post(route_of(g.get("action") or ""), call, SpeechResult=ZIP_WORDS, Confidence="0.9")
    assert again["body"] == final["body"] and verifies(h) == 1


def test_late_early_redirect_after_final_replays(early):
    """The update reached Twilio while the final webhook was already being answered."""
    h, calls = early
    call = sid(8)
    g = gather_of(h.start(call, THOMAS))
    partial(h, call, g, ZIP_WORDS)
    final = h.post(route_of(g.get("action") or ""), call, SpeechResult=ZIP_WORDS, Confidence="0.9")
    late = h.post(redirect_route(calls.updates[0][1]), call)
    assert late["body"] == final["body"] and verifies(h) == 1


def test_update_failure_leaves_the_normal_path(caplog):
    caplog.set_level(logging.DEBUG)
    h, calls = make(fail=True)
    call = sid(9)
    g = gather_of(h.start(call, THOMAS))
    partial(h, call, g, ZIP_WORDS)
    assert calls.updates == []
    assert gathers.state_of(gathers.load(h.store, call, gid_of(g))) == gathers.RELEASED
    assert events(caplog.records, "factor_early_update_failed")
    final = h.post(route_of(g.get("action") or ""), call, SpeechResult=ZIP_WORDS, Confidence="0.9")
    assert final["statusCode"] == 200 and verifies(h) == 1
    assert relay(ET.fromstring(final["body"].encode())) is not None
    partial(h, call, g, ZIP_WORDS)  # nothing more
    assert verifies(h) == 1
    assert_no_leak(h, caplog.records, all_fixture_factor_values() | h.sent_factors)
    _cleanup(h)


def test_expired_early_digits_are_asked_again(early):
    h, calls = early
    call = sid(10)
    g = gather_of(h.start(call, THOMAS))
    partial(h, call, g, ZIP_WORDS)
    item = gathers.load(h.store, call, gid_of(g))
    item["digits_expires_at"] = 0
    h.store.put("calls", item)
    root = ET.fromstring(h.post(redirect_route(calls.updates[0][1]), call)["body"].encode())
    assert root.find("Gather") is not None and verifies(h) == 0  # an unclear answer: asked again


def test_digits_sealed_and_dropped_once_taken(early):
    h, calls = early
    call = sid(11)
    g = gather_of(h.start(call, THOMAS))
    partial(h, call, g, ZIP_WORDS)
    item = gathers.load(h.store, call, gid_of(g))
    assert item["state"] == gathers.EARLY and ZIP not in str(item)
    assert h.sealer.unseal(item["sealed_value"], call_sid=call) == ZIP
    assert item["digits_expires_at"] == int(CLOCK) + gathers.DIGITS_TTL_S
    h.post(redirect_route(calls.updates[0][1]), call)
    item = gathers.load(h.store, call, gid_of(g))
    assert item["state"] == gathers.DONE and "sealed_value" not in item
    assert "<Response" not in str(item)  # the stored TwiML is sealed too


# --------------------------------------------------------------------------- the route's auth


def test_partial_requires_signature_and_basic(early):
    h, calls = early
    call = sid(12)
    g = gather_of(h.start(call, THOMAS))
    route = route_of(g.get("partialResultCallback") or "")
    params = {"CallSid": call, "UnstableSpeechResult": ZIP_WORDS}
    assert h.app.handle(make_event(route, params, signature="bad"))["statusCode"] == 403
    assert h.app.handle(make_event(route, params, auth=NO_AUTH))["statusCode"] == 401
    assert calls.updates == []
    assert gathers.load(h.store, call, gid_of(g)) is None


def test_partial_for_unknown_gather_is_ignored(early, caplog):
    h, calls = early
    call = sid(13)
    h.start(call, THOMAS)
    resp = h.post("partial?g=" + "Q" * 16, call, UnstableSpeechResult=ZIP_WORDS)
    assert resp["statusCode"] == 204 and calls.updates == []
    assert [r.outcome for r in events(caplog.records, "factor_partial")] == ["stale"]


# --------------------------------------------------------------------------- (PBT)

_ACTIONS = st.lists(st.sampled_from(["incomplete", "complete", "ambiguous", "final", "early",
                                     "final_again", "early_again"]), min_size=1, max_size=10)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(actions=_ACTIONS, fail=st.booleans())
def test_one_verification_in_any_order(actions, fail):
    """Any interleaving of partials (incomplete, complete, ambiguous), the final webhook,
    the early redirect and replays of either: at most one REST update, exactly one verification
    once an answer request arrived, and the same TwiML for every answer to the Gather."""
    h, calls = make(fail=fail)
    call = sid(20)
    try:
        g = gather_of(h.start(call, THOMAS))
        texts = {"incomplete": "nine four", "complete": ZIP_WORDS,
                 "ambiguous": "nine four one oh five one two three four five"}
        bodies: list[str] = []
        answered = False
        for a in actions:
            if a in texts:
                partial(h, call, g, texts[a])
            elif a.startswith("final"):
                r = h.post(route_of(g.get("action") or ""), call, SpeechResult=ZIP_WORDS,
                           Confidence="0.9")
                bodies.append(r["body"])
                answered = True
            elif calls.updates:  # Twilio only redirects after an update it received
                bodies.append(h.post(redirect_route(calls.updates[0][1]), call)["body"])
                answered = True
            assert len(calls.updates) <= 1
            assert verifies(h) <= 1
        assert verifies(h) == (1 if answered else 0)
        assert len(set(bodies)) <= 1
        if bodies:
            assert relay(ET.fromstring(bodies[0].encode())) is not None
    finally:
        _cleanup(h)


# --------------------------------------------------------------------------- (PBT, pure)

ZIP_SLOT = SlotDecl.from_json({"slotType": "digits", "constraints": {"length": 5}, "sensitive": True})
_WORDS = EN + ["twenty", "ninety", "double", "is", "my", "zip", "um", "the", "and", "hundred"]


@settings(max_examples=400, deadline=None)
@given(toks=st.lists(st.sampled_from(_WORDS), min_size=0, max_size=12))
def test_complete_only_on_a_unique_full_length_parse(toks):
    from owlbank.slots import normalize
    from owlbank.slots.validators import _run_candidates

    text = " ".join(toks)
    outcome, value = early_digits(ZIP_SLOT, text, "en-US")
    norm = normalize(text, "en-US").split()  # fillers ("um") are dropped first, as for finals
    sized = {c for c, _ in _run_candidates(norm, "en-US") if len(c) == 5}
    if outcome == "complete":
        assert value is not None and len(value) == 5 and value.isdigit()
        assert sized == {value}  # exactly one candidate of the expected length
        assert norm[-1] not in ("twenty", "ninety", "double", "hundred")  # can't still grow
    else:
        assert value is None
    if outcome == "ambiguous":
        assert len(sized) > 1


@pytest.mark.parametrize("text,want", [
    ("nine four one oh five", ("complete", "94105")),
    ("my zip is 94105", ("complete", "94105")),
    ("nine four one oh", ("incomplete", None)),
    ("nine four one oh twenty", ("incomplete", None)),  # "twenty" may become "twenty five"
    ("1 2 3 20", ("incomplete", None)),
    ("double", ("incomplete", None)),
    ("12345 or 54321", ("ambiguous", None)),
    ("", ("incomplete", None)),
])
def test_early_digits_examples(text, want):
    assert early_digits(ZIP_SLOT, text, "en-US") == want


def test_early_digits_non_fixed_slots():
    assert early_digits(SlotDecl("yes_no"), "yes", "en-US") == ("incomplete", None)
    ranged = SlotDecl("digits", {"minLength": 4, "maxLength": 6})
    assert early_digits(ranged, "1234", "en-US") == ("incomplete", None)


# --------------------------------------------------------------------------- units


def test_twilio_call_updater_is_lazy_and_reused():
    from lambdas.frontdoor.calls import TwilioCallUpdater

    built: list[int] = []
    sent: list[tuple[str, str]] = []

    class _Call:
        def __init__(self, call_sid):
            self.call_sid = call_sid

        def update(self, *, twiml):
            sent.append((self.call_sid, twiml))

    class _Client:
        def calls(self, call_sid):
            return _Call(call_sid)

    upd = TwilioCallUpdater(lambda: built.append(1) or _Client())
    assert built == []  # no secret read, no client, until the first update
    upd.update_twiml("CA1", "<Response/>")
    upd.update_twiml("CA2", "<Response/>")
    assert built == [1] and [s for s, _ in sent] == ["CA1", "CA2"]


def test_absolute_url_keeps_stage_and_quotes_credentials():
    from lambdas.frontdoor.http import TwilioRequest

    req = TwilioRequest(route="partial", params={}, base_path="/v1", origin="https://h.example")
    assert req.absolute_url("factor", early="abc") == "https://h.example/v1/voice/factor?early=abc"
    assert req.absolute_url("factor", credentials=("u", "p@ss:")) == \
        "https://u:p%40ss%3A@h.example/v1/voice/factor"
    with pytest.raises(ValueError):
        TwilioRequest(route="x", params={}, base_path="").absolute_url("factor")


def test_sealed_twiml_round_trips_in_chunks():
    from owlbank.token_seal import FakeSealer

    sealer = FakeSealer()
    xml = "<Response>" + "<Say>hello there</Say>" * 2000 + "</Response>"
    chunks = gathers.seal_twiml(sealer, xml, call_sid="CA1")
    assert gathers.unseal_twiml(sealer, chunks, call_sid="CA1") == xml
