"""Front door: incoming -> language -> factor, every caller-lookup outcome, language
selection, off-script input at the ZIP prompt. The no-leak scan runs at teardown."""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from urllib.parse import parse_qs, urlsplit

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as hst

from owlbank import recap as rc
from owlbank import sensitive, tokens, ws_nonce
from owlbank.segments import clear_sensitive
from owlbank.slots import FakeResolver, ResolverOutput
from owlbank.token_seal import UnsealError

from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    CLOCK, FAILED_AI, LEES, MARIA, NOBODY, OTP, TEST_FACTORS, THOMAS, WORKFLOW, all_fixture_factor_values,
    assert_no_leak, assert_scripted, build, fd, relay, says, sid,
)

THOMAS_ZIP = TEST_FACTORS["C-1001"]["zip"]


@pytest.fixture
def keypad(caplog: pytest.LogCaptureFixture):
    """Spoken factors off (``accept_spoken_factors=False``; the default is on):
    every factor is keypad-only. The no-leak scan runs at teardown, as for ``fd``."""
    caplog.set_level(logging.DEBUG)
    h = build(config_kw={"accept_spoken_factors": False})
    yield h
    assert_no_leak(h, caplog.records, all_fixture_factor_values() | h.sent_factors)
    for call in {k[0].split("#")[0] for k in h.store._tables["calls"]}:  # noqa: SLF001
        clear_sensitive(call)


def test_incoming_plays_language_select_in_a_one_digit_gather(fd):
    root = fd.xml("incoming", sid(1), From=THOMAS)
    g = root.find("Gather")
    assert g is not None and g.get("numDigits") == "1" and g.get("input") == "dtmf"
    assert int(g.get("timeout")) <= 5 and g.get("action") == "/prod/voice/language"
    assert says(root) == ["Welcome to OwlBank. This call may be recorded and transcribed for quality and training.", "Para español, oprima dos."]
    st = fd.state(sid(1))
    assert st.candidate_ids == ("C-1001",) and st.level == 0
    assert fd.lam.ops() == ["ani_lookup"]


def test_incoming_is_idempotent_on_twilio_retry(fd):
    fd.xml("incoming", sid(1), From=THOMAS)
    fd.xml("incoming", sid(1), From=THOMAS)
    assert fd.lam.ops() == ["ani_lookup"]
    assert [e["type"] for e in fd.journal(sid(1))].count("call_start") == 1


def test_no_input_defaults_to_english_and_asks_level1_factor(keypad):
    fd = keypad
    root = fd.start(sid(1), THOMAS)
    assert fd.state(sid(1)).language == "en-US"
    g = root.find("Gather")
    # Spoken factors off: the factor prompt listens for the keypad only.
    assert g.get("input") == "dtmf" and g.get("numDigits") == "5"
    assert g.get("action") == "/prod/voice/factor" and g.get("actionOnEmptyResult") == "true"
    # the welcome was heard at incoming; English goes straight to the factor.
    assert says(root) == ["Please enter your five-digit ZIP code. If you can't, press pound."]


def test_press_two_selects_spanish(keypad):
    fd = keypad
    root = fd.start(sid(1), MARIA, lang_digit="2")
    assert fd.state(sid(1)).language == "es-US"
    assert says(root)[1].startswith("Por favor, ingrese su código postal")
    assert root.find("Gather").get("input") == "dtmf"  # keypad only: no speech language
    root = fd.factor(sid(1), TEST_FACTORS["C-1004"]["zip"])
    cr = relay(root)
    assert cr.get("language") == "es-US" and cr.get("transcriptionLanguage") == "es"
    assert "María" in cr.get("welcomeGreeting")


# --------------------------------------------------------------------------- outcomes


def test_one_match_verified_connects_relay_with_personal_greeting(fd):
    root = fd.login_thomas(sid(1))
    cr = relay(root)
    assert root.find("Connect").get("action") == "/prod/voice/connect-action"
    assert cr.get("welcomeGreeting") == (
        "Hi Thomas. I am the OwlBank virtual agent. How can I help you today?")
    url = urlsplit(cr.get("url"))
    assert url._replace(query="").geturl() == "wss://bridge.owlbank.example/ws"
    (nonce,) = parse_qs(url.query)["n"]  # one-time nonce in the signed URL, only its hash stored
    assert len(nonce) >= 32 and fd.state(sid(1)).ws_nonce_hash == ws_nonce.nonce_hash(nonce)
    assert nonce not in json.dumps(fd.store.get("calls", {"call_sid": sid(1)}))
    for attr, value in {"language": "en-US", "ttsProvider": "ElevenLabs",
                        "voice": "UgBBYS2sOqTuMpoF3BR0-v4_turbo",
                        "transcriptionProvider": "Deepgram", "transcriptionLanguage": "en-US",
                        "speechModel": "nova-3-general", "dtmfDetection": "false"}.items():
        assert cr.get(attr) == value, attr
    assert cr.get("events", "").split() == ["speaker-events"]
    assert cr.get("conversationConfiguration").startswith("conv_configuration_")
    st = fd.state(sid(1))
    assert st.level == 1 and st.verified_customer_id == "C-1001"
    assert set(st.factors_verified) == {"ani_match", "zip"}
    assert st.pending_result is None
    # the token rests only sealed, and unseals only for this call.
    token = fd.sealer.unseal(st.sealed_token, call_sid=sid(1))
    claims = tokens.validate(fd.store, token, call_sid=sid(1), min_level=1, now=int(CLOCK))
    assert not isinstance(claims, tokens.AuthError) and claims.customer_id == "C-1001"
    try:
        fd.sealer.unseal(st.sealed_token, call_sid=sid(2))
        raise AssertionError("unsealed under another call")
    except UnsealError:
        pass
    assert token not in str(fd.store._tables["calls"])  # noqa: SLF001


def test_one_match_failed_retries_then_goes_to_the_ai(fd):
    fd.start(sid(1), THOMAS)
    root = fd.factor(sid(1), "94106")
    assert says(root) == ["Sorry, that didn't match our records. Let's try again.",
                          "Please say or enter your five-digit ZIP code. If you can't, press pound."]
    fd.factor(sid(1), "94107")
    root = fd.factor(sid(1), "94108")  # third failure: verify_attempts_per_call = 3
    # the AI by default, with the one landing phrase as the greeting.
    assert relay(root).get("welcomeGreeting") == FAILED_AI
    st = fd.state(sid(1))
    assert st.level == 0 and st.sealed_token is None
    assert st.login_closed is True  # no second try in this call


def test_one_match_failed_escalates_when_the_bank_configured_it():
    # OWLBANK_FALLBACKS {"verify_failed": "escalate"} keeps today's path.
    h = build(config_kw={"fallbacks": {"verify_failed": "escalate"}})
    h.start(sid(1), THOMAS)
    for z in ("94106", "94107"):
        h.factor(sid(1), z)
    root = h.factor(sid(1), "94108")
    assert "I'm sorry, I wasn't able to verify your identity." in says(root)
    assert FAILED_AI not in says(root)
    enq = root.find("Enqueue")
    assert enq is not None and enq.get("workflowSid") == WORKFLOW
    attrs = json.loads(enq.find("Task").text)
    assert attrs["escalation_reason"] == "step_up_failed" and attrs["auth_level"] == 0
    assert h.state(sid(1)).level == 0 and h.state(sid(1)).login_closed is True


def test_several_records_distinct_zip_verifies_the_right_customer(fd):
    fd.start(sid(1), LEES)
    assert fd.state(sid(1)).candidate_ids == ("C-1002", "C-1003")
    root = fd.factor(sid(1), TEST_FACTORS["C-1003"]["zip"])
    assert relay(root) is not None
    assert fd.state(sid(1)).verified_customer_id == "C-1003"


def test_several_records_ambiguous_zip_asks_a_disambiguating_factor(fd):
    # Make the household share one ZIP, so the ZIP alone can't tell them apart.
    jordan = fd.store.get("customers", {"customer_id": "C-1003"})
    sam = fd.store.get("customers", {"customer_id": "C-1002"})
    jordan["factors"]["zip"] = sam["factors"]["zip"]
    fd.store.put("customers", jordan)
    fd.start(sid(1), LEES)
    root = fd.factor(sid(1), TEST_FACTORS["C-1002"]["zip"])
    assert says(root) == ["Please say or enter the last four digits of your Social Security number. If you can't, press pound."]
    assert root.find("Gather").get("numDigits") == "4"
    held = fd.session(sid(1)).flow["held"]
    assert set(held) == {"zip"}  # sealed, not plaintext (the teardown scan checks the store)
    root = fd.factor(sid(1), TEST_FACTORS["C-1003"]["ssn4"])
    assert relay(root) is not None
    st = fd.state(sid(1))
    assert st.verified_customer_id == "C-1003" and st.level == 2
    assert "held" not in (fd.session(sid(1)).flow or {})
    verify_events = [e for e in fd.lam.payloads if e.get("op") == "verify"]
    assert sorted(verify_events[-1]["factors"]) == ["ssn4", "zip"]


def test_no_match_goes_to_limited_ai_with_the_keyed_phone_switch_off(caplog):
    h = keyed_harness(caplog, switch=False)
    root = h.start(sid(1), NOBODY)
    assert root.find("Gather") is None
    assert relay(root).get("welcomeGreeting") == "I am the OwlBank virtual agent. How can I help you today?"
    st = h.state(sid(1))
    assert st.level == 0 and st.candidate_ids == () and st.sealed_token is None
    keyed_teardown(h, caplog)


def test_ani_withheld_skips_lookup_and_goes_to_limited_ai_with_the_switch_off(caplog):
    h = keyed_harness(caplog, switch=False)
    root = h.start(sid(1), "anonymous")
    assert h.lam.ops() == []
    assert relay(root) is not None and h.state(sid(1)).ani is None
    lookup = [e for e in h.journal(sid(1)) if e["type"] == "lookup"][0]
    assert lookup["outcome"] == "withheld"
    assert not [e for e in h.journal(sid(1)) if e["type"] == "keyed_lookup"]
    keyed_teardown(h, caplog)


def test_lookup_error_apologises_and_falls_back(fd):
    fd.lam.fail_ops.add("ani_lookup")
    root = fd.start(sid(1), THOMAS)
    assert fd.lam.ops() == ["ani_lookup", "ani_lookup"]  # one retry after the language prompt
    assert_scripted(root, "I'm sorry, I can't look up your account right now.",
                    "I am the OwlBank virtual agent. How can I help you today?")
    assert fd.state(sid(1)).level == 0


def test_verify_unavailable_apologises_and_falls_back(fd):
    fd.lam.overrides["verify"] = {"result": "unavailable", "level": 0}
    fd.start(sid(1), THOMAS)
    root = fd.factor(sid(1), THOMAS_ZIP)
    assert_scripted(root, "I'm sorry, something went wrong on our end.",
                    "I am the OwlBank virtual agent. How can I help you today?")


# --------------------------------------------------------------------------- slot engine


def test_unclear_reprompts_with_clarification_then_falls_back(fd):
    fd.start(sid(1), THOMAS)
    root = fd.factor(sid(1), "12")  # too short: unclear
    assert says(root) == ["Please enter the 5 digits of your ZIP code on your keypad."]
    root = fd.factor(sid(1), "")  # silence: unclear again, second retry (maxRetries 2)
    assert says(root) == ["Please enter the 5 digits of your ZIP code on your keypad."]
    root = fd.factor(sid(1), "")  # retries exhausted: verify_failed, the AI by default
    assert relay(root).get("welcomeGreeting") == FAILED_AI


def test_spoken_factor_is_not_used_and_keypad_is_requested(keypad):
    fd = keypad
    fd.start(sid(1), THOMAS)
    fd.sent_factors.add(THOMAS_ZIP)
    root = fd.xml("factor", sid(1), SpeechResult="nine four one oh five", Confidence="0.93")
    assert says(root) == ["For your security, please use your keypad.",
                          "Please enter your five-digit ZIP code. If you can't, press pound."]
    assert "verify" not in fd.lam.ops()


def test_repeat_command_replays_the_question(fd):
    fd.start(sid(1), THOMAS)
    root = fd.xml("factor", sid(1), SpeechResult="repeat", Confidence="0.9")
    assert says(root) == ["Please say or enter your five-digit ZIP code. If you can't, press pound."]


def test_agent_command_at_login_escalates(fd):
    fd.start(sid(1), THOMAS)
    root = fd.xml("factor", sid(1), SpeechResult="agent", Confidence="0.9")
    attrs = json.loads(root.find("Enqueue/Task").text)
    assert attrs["escalation_reason"] == "caller_requested_human"


def test_intent_switch_at_zip_prompt_returns_to_ai_as_interrupted(caplog):
    import logging

    from tests.component.test_frontdoor_support import TEST_FACTORS, all_fixture_factor_values, assert_no_leak
    caplog.set_level(logging.DEBUG)
    h = build(resolver=FakeResolver([ResolverOutput(kind="intent_switch")]))
    h.start(sid(1), THOMAS)
    utterance = "Actually my zip is 94105 but what are your branch hours?"
    root = h.xml("factor", sid(1), SpeechResult=utterance, Confidence="0.9")
    cr = relay(root)
    assert cr.get("welcomeGreeting") == "One moment while I get that for you."
    env = h.pending_result(sid(1))
    assert env.status == "interrupted" and env.next == "handle_utterance"
    assert env.tool == "authenticate" and env.completed is False
    # ZIP isn't sensitive in the seed, so the AI gets the caller's words.
    assert env.callerUtterance == utterance
    assert h.state(sid(1)).interrupted_utterance == env.callerUtterance
    assert [s.template for s in env.spoken] == ["factor.zip", "common.one_moment"]
    # the resolver gets the caller's words unredacted, digits included.
    assert h.resolver.calls[0]["req"].utterance == utterance
    assert h.state(sid(1)).level == 0
    zips = {f["zip"] for f in TEST_FACTORS.values() if "zip" in f}
    assert_no_leak(h, caplog.records, all_fixture_factor_values() - zips)


# --------------------------------------------------------------------------- keyed-in phone login
# Fixtures for the keyed-phone flow. The harness runs the real auth service
# and bank tools; ``start`` sends a carrier-attested ``StirVerstat`` (TN-Validation-Passed-A), so
# every keyed call below also shows that a keyed number is untrusted whatever the call's verstat.

PHONE_Q = "What is the phone number associated with your account? You can say or enter it. If you don't have an account with us, press pound."
PHONE_RETRY = "Please say or enter the 10-digit phone number on your account. If you don't have an account with us, press pound."
ZIP_Q = "Please say or enter your five-digit ZIP code. If you can't, press pound."
SSN_Q = "Please say or enter the last four digits of your Social Security number. If you can't, press pound."
OTP_Q = "We just sent a six-digit code to your phone by text message. Please say or enter it now, or press pound if you didn't get it."
THOMAS_KEYED = THOMAS[2:]           # 2025550101
LEES_KEYED = LEES[2:]               # 2025550102 (Sam and Jordan Doe)
UNKNOWN_KEYED = ("2025550177", "2025550188")
THOMAS_SSN4 = TEST_FACTORS["C-1001"]["ssn4"]


def keyed_harness(caplog: pytest.LogCaptureFixture, *, switch: bool = True,
                  on_untrusted: str | None = None, **kw):
    caplog.set_level(logging.DEBUG)

    def edit(doc: dict) -> None:
        doc["ani_trust"]["ask_phone_on_no_match"] = switch
        if on_untrusted is not None:
            doc["ani_trust"]["on_untrusted"] = on_untrusted

    return build(policy_edit=edit, **kw)


def keyed_teardown(h, caplog: pytest.LogCaptureFixture) -> None:
    """No factor value and no keyed number (10 digits or E.164) anywhere."""
    keyed = {v for v in h.sent_factors if len(v) == 10 and v.isdigit()}
    try:
        assert_no_leak(h, caplog.records,
                       all_fixture_factor_values() | h.sent_factors | {"+1" + v for v in keyed})
    finally:
        for call in {k[0].split("#")[0] for k in h.store._tables["calls"]}:  # noqa: SLF001
            clear_sensitive(call)
            sensitive.forget(call)


@pytest.fixture
def kfd(caplog: pytest.LogCaptureFixture):
    h = keyed_harness(caplog)
    yield h
    keyed_teardown(h, caplog)


def keyed_events(h, call: str) -> list[dict]:
    return [{k: e[k] for k in ("outcome", "count")}
            for e in h.journal(call) if e["type"] == "keyed_lookup"]


def auth_payloads(h, op: str) -> list[dict]:
    return [p for p in h.lam.payloads if p.get("op") == op]


def keyed_login(h, call: str, ani: str = "anonymous", number: str = THOMAS_KEYED):
    root = h.start(call, ani)
    assert says(root) == [PHONE_Q]
    return h.factor(call, number)


def test_withheld_then_matching_number_logs_in_with_zip_and_ssn4(kfd):
    root = kfd.start(sid(40), "anonymous")
    assert kfd.lam.ops() == []  # withheld: no ANI lookup
    g = root.find("Gather")
    assert says(root) == [PHONE_Q]
    assert g.get("numDigits") == "10" and g.get("input") == "dtmf speech"  # speech always
    root = kfd.factor(sid(40), THOMAS_KEYED)
    assert says(root) == [ZIP_Q]  # the untrusted set: ZIP first ...
    assert kfd.lam.ops() == ["ani_lookup", "precheck"]
    (lookup,) = auth_payloads(kfd, "ani_lookup")
    assert lookup["args"]["ani"] == THOMAS  # "+1" + the 10 digits
    (pre,) = auth_payloads(kfd, "precheck")
    assert pre["ani"] == THOMAS and pre["stir_verstat"] is None
    st = kfd.state(sid(40))
    assert st.identity_source == "keyed_phone" and st.ani is None and st.ani_trusted is False
    assert st.candidate_ids == ("C-1001",)
    root = kfd.factor(sid(40), THOMAS_ZIP)
    assert SSN_Q in says(root)  # ... then SSN-4: ZIP alone is not level 1 on a keyed call
    root = kfd.factor(sid(40), THOMAS_SSN4)
    assert relay(root).get("welcomeGreeting").startswith("Hi Thomas.")
    st = kfd.state(sid(40))
    assert st.level == 1 and st.verified_customer_id == "C-1001"
    assert "ani_match" not in st.factors_verified
    (ver,) = auth_payloads(kfd, "verify")
    assert ver["ani"] == THOMAS and ver["stir_verstat"] is None
    assert sorted(ver["factors"]) == ["ssn4", "zip"]
    assert keyed_events(kfd, sid(40)) == [{"outcome": "match", "count": 1}]
    (ev,) = [e for e in kfd.journal(sid(40)) if e["type"] == "keyed_lookup"]
    assert set(ev) - {"call_sid", "seq", "ts", "expires_at"} == {"type", "outcome", "count"}


def test_unmatched_ani_then_matching_number(kfd):
    root = kfd.start(sid(41), NOBODY)
    assert says(root) == [PHONE_Q]
    root = kfd.factor(sid(41), THOMAS_KEYED)
    assert says(root) == [ZIP_Q]
    st = kfd.state(sid(41))
    assert st.ani == NOBODY and st.identity_source == "keyed_phone"  # the call's own ANI is kept
    kfd.factor(sid(41), THOMAS_ZIP)
    kfd.factor(sid(41), THOMAS_SSN4)
    assert kfd.state(sid(41)).level == 1
    for op in ("precheck", "verify"):
        assert {(p["ani"], p["stir_verstat"]) for p in auth_payloads(kfd, op)} == {(THOMAS, None)}
    assert [e["outcome"] for e in kfd.journal(sid(41)) if e["type"] == "lookup"] == ["no_match"]


def test_two_misses_go_to_the_no_match_fallback(kfd, caplog):
    kfd.start(sid(42), NOBODY)
    root = kfd.factor(sid(42), UNKNOWN_KEYED[0])
    assert says(root) == [PHONE_RETRY]  # says nothing about the number
    assert root.find("Gather").get("numDigits") == "10"
    root = kfd.factor(sid(42), UNKNOWN_KEYED[1])
    assert root.find("Gather") is None
    assert relay(root).get("welcomeGreeting") == "I am the OwlBank virtual agent. How can I help you today?"  # today's no_match
    assert keyed_events(kfd, sid(42)) == [{"outcome": "no_match", "count": 0}] * 2
    assert kfd.lam.ops().count("ani_lookup") == 3  # the ANI, then two keyed numbers, no more
    outcomes = [e.get("outcome") for e in kfd.journal(sid(42)) if e["type"] == "verification"]
    assert outcomes == ["fallback_no_match"]
    st = kfd.state(sid(42))
    assert st.level == 0 and st.identity_source == "ani" and st.candidate_ids == ()
    assert "verify" not in kfd.lam.ops() and "precheck" not in kfd.lam.ops()


def test_malformed_answer_is_asked_again_then_goes_to_no_match(kfd):
    kfd.start(sid(43), "anonymous")
    root = kfd.factor(sid(43), "12345")  # too short: the slot engine's clarify, no lookup
    assert says(root) == ["Please enter the 10 digits of your phone number on your keypad."]
    kfd.factor(sid(43), "")
    root = kfd.factor(sid(43), "")  # retries exhausted (maxRetries 2)
    assert root.find("Enqueue") is None  # not verify_failed's escalation ...
    assert relay(root).get("welcomeGreeting") == "I am the OwlBank virtual agent. How can I help you today?"  # ... but no_match
    assert kfd.lam.ops() == []
    assert keyed_events(kfd, sid(43)) == [{"outcome": "unclear", "count": 0}]
    outcomes = [e.get("outcome") for e in kfd.journal(sid(43)) if e["type"] == "verification"]
    assert outcomes == ["fallback_no_match"]


def test_multiple_match_disambiguates_by_the_untrusted_factors(kfd):
    root = keyed_login(kfd, sid(44), number=LEES_KEYED)
    assert says(root) == [ZIP_Q]
    assert kfd.state(sid(44)).candidate_ids == ("C-1002", "C-1003")
    assert keyed_events(kfd, sid(44)) == [{"outcome": "multiple", "count": 2}]
    kfd.factor(sid(44), TEST_FACTORS["C-1003"]["zip"])
    root = kfd.factor(sid(44), TEST_FACTORS["C-1003"]["ssn4"])
    assert relay(root) is not None
    st = kfd.state(sid(44))
    assert st.verified_customer_id == "C-1003" and st.level == 1


def test_lookup_error_retries_once_then_takes_the_lookup_error_fallback(kfd):
    kfd.start(sid(45), "anonymous")
    kfd.lam.fail_ops.add("ani_lookup")
    root = kfd.factor(sid(45), THOMAS_KEYED)
    assert_scripted(root, "I'm sorry, I can't look up your account right now.",
                    "I am the OwlBank virtual agent. How can I help you today?")
    assert kfd.lam.ops() == ["ani_lookup", "ani_lookup"]
    assert keyed_events(kfd, sid(45)) == [{"outcome": "error", "count": 0}]
    assert kfd.state(sid(45)).identity_source == "ani"


# Memory-writes (option A): a caller who keyed in a number is untrusted whatever STIR/SHAKEN
# said, so an unverified keyed-in caller gets no recap on the caller-ID profile, also when the keyed
# number matched no one or the lookup failed (the call stays identity_source "ani").

def test_keyed_no_match_after_the_last_try_rules_out_the_caller_id_recap(kfd):
    kfd.start(sid(55), NOBODY)
    assert kfd.state(sid(55)).ani_trusted is True  # carrier-attested caller ID, no candidate
    kfd.factor(sid(55), UNKNOWN_KEYED[0])
    root = kfd.factor(sid(55), UNKNOWN_KEYED[1])
    assert relay(root) is not None  # the no_match fallback
    st = kfd.state(sid(55))
    assert st.ani == NOBODY and st.identity_source == "ani" and st.level == 0
    assert st.ani_trusted is False
    assert rc.recap_target(st, None) == rc.Skip(rc.ANI_UNTRUSTED)


def test_keyed_lookup_error_rules_out_the_caller_id_recap(kfd):
    kfd.start(sid(56), NOBODY)
    kfd.lam.fail_ops.add("ani_lookup")
    root = kfd.factor(sid(56), THOMAS_KEYED)
    assert_scripted(root, "I'm sorry, I can't look up your account right now.",
                    "I am the OwlBank virtual agent. How can I help you today?")
    st = kfd.state(sid(56))
    assert st.ani == NOBODY and st.identity_source == "ani" and st.ani_trusted is False
    assert rc.recap_target(st, None) == rc.Skip(rc.ANI_UNTRUSTED)


def test_an_unmatched_trusted_ani_never_asked_for_a_number_keeps_its_recap(caplog):
    h = keyed_harness(caplog, switch=False)
    h.start(sid(57), NOBODY)
    st = h.state(sid(57))
    assert st.ani_trusted is True
    assert rc.recap_target(st, None) == rc.Target("phone", NOBODY, create=False)
    keyed_teardown(h, caplog)


def test_owner_phrase_in_groups_is_resolved_by_the_resolver(caplog):
    h = keyed_harness(caplog, resolver=FakeResolver([
        ResolverOutput(kind="inferred", value=THOMAS_KEYED)]))
    h.start(sid(46), "anonymous")
    h.sent_factors.add(THOMAS_KEYED)
    said = "Yeah the number is 202, eh 555, what was it 0101"
    root = h.xml("factor", sid(46), SpeechResult=said, Confidence="0.9")
    assert says(root) == [ZIP_Q]  # confirm: never: no read-back, straight to the lookup
    assert h.resolver.calls and auth_payloads(h, "ani_lookup")[0]["args"]["ani"] == THOMAS
    assert h.state(sid(46)).identity_source == "keyed_phone"
    keyed_teardown(h, caplog)


def test_early_finish_at_ten_spoken_digits(caplog):
    from urllib.parse import parse_qs as _qs

    from lambdas.frontdoor.calls import FakeCallUpdater

    calls = FakeCallUpdater()
    h = keyed_harness(caplog, calls=calls)
    h.sent_factors.add(THOMAS_KEYED)
    g = h.start(sid(47), "anonymous").find("Gather")
    gid = _qs(urlsplit(g.get("action")).query)["g"][0]
    assert g.get("partialResultCallback", "").endswith(f"/voice/partial?g={gid}")
    words = "two oh two five five five oh one oh one"
    resp = h.post(g.get("partialResultCallback").rsplit("/voice/", 1)[1], sid(47),
                  UnstableSpeechResult=words, SequenceNumber="1")
    assert resp["statusCode"] == 204 and len(calls.updates) == 1
    url = ET.fromstring(calls.updates[0][1].encode()).find("Redirect").text
    root = ET.fromstring(h.post(url.rsplit("/voice/", 1)[1], sid(47))["body"].encode())
    assert says(root) == [ZIP_Q] and h.state(sid(47)).identity_source == "keyed_phone"
    keyed_teardown(h, caplog)


def test_switch_off_with_an_unmatched_ani_never_asks(caplog):
    h = keyed_harness(caplog, switch=False)
    root = h.start(sid(48), NOBODY)
    assert PHONE_Q not in says(root) and relay(root) is not None
    assert h.lam.ops() == ["ani_lookup"]
    keyed_teardown(h, caplog)


def _old_policy(doc: dict) -> None:
    """The policy table before the reseed: no ``phone`` slot, no switch (which defaults on)."""
    del doc["slots"]["phone"]
    doc["ani_trust"].pop("ask_phone_on_no_match", None)


def _old_catalog(doc: dict) -> None:
    """The prompt table before the reseed: no ``login.phone_number*`` questions."""
    doc["entries"] = [e for e in doc["entries"] if not e["key"].startswith("login.phone_number")]


@pytest.mark.parametrize("edits", [{"policy_edit": _old_policy}, {"catalog_edit": _old_catalog}],
                         ids=["old_policy", "old_catalog"])
def test_new_code_with_old_tables_takes_the_no_match_path(caplog, edits):
    """New code before the reseed never asks for a number (the default-on
    switch would otherwise ask an unknown slot and send the caller to a human)."""
    caplog.set_level(logging.DEBUG)
    h = build(**edits)
    for n in (50, 51):
        root = h.start(sid(n), NOBODY)
        assert PHONE_Q not in says(root)
        assert relay(root).get("welcomeGreeting") == "I am the OwlBank virtual agent. How can I help you today?"
        assert h.state(sid(n)).level == 0
    warned = [r for r in caplog.records
              if r.levelno == logging.WARNING and "keyed_phone_unavailable" in r.getMessage()]
    assert len(warned) == 1  # once per instance
    keyed_teardown(h, caplog)


def test_matched_ani_is_never_asked_for_a_number(kfd):
    root = kfd.start(sid(49), THOMAS)
    assert says(root) == [ZIP_Q]
    kfd.factor(sid(49), THOMAS_ZIP)
    st = kfd.state(sid(49))
    assert st.level == 1 and st.identity_source == "ani"
    assert keyed_events(kfd, sid(49)) == []


def test_level2_code_goes_to_the_keyed_number_only_after_zip_and_ssn4(kfd):
    call = sid(50)
    keyed_login(kfd, call)
    kfd.factor(call, THOMAS_ZIP)
    assert "send_otp" not in kfd.lam.ops() and kfd.verify.sent == []  # nothing before ZIP + SSN4
    kfd.factor(call, THOMAS_SSN4)
    assert kfd.state(call).level == 1 and kfd.verify.sent == []
    kfd.set_pending_tool(call, "authenticate", {"tool": "change_address"})
    root = kfd.handoff(call, {"reason": "step-up", "tool": "change_address", "journal": "jr-k"})
    assert says(root)[-1] == OTP_Q
    ops = kfd.lam.ops()
    assert ops.index("send_otp") > ops.index("verify")  # the code only after the level-1 verify
    assert kfd.verify.sent == [(THOMAS, "sms")]  # the number on file, which the caller keyed
    (send,) = auth_payloads(kfd, "send_otp")
    assert send["ani"] == THOMAS and send["stir_verstat"] is None
    kfd.sent_factors.add(OTP)
    kfd.factor(call, OTP)
    st = kfd.state(call)
    # The untrusted level 2 holds the factors of level 3 (effective_factors), as for any untrusted ANI.
    assert st.level == 3 and "ani_match" not in st.factors_verified
    otp_verify = [p for p in auth_payloads(kfd, "verify") if "otp" in p["factors"]]
    assert [(p["ani"], p["stir_verstat"]) for p in otp_verify] == [(THOMAS, None)]


def test_keyed_call_escalation_tells_the_agent_without_the_number(kfd):
    keyed_login(kfd, sid(51))
    root = kfd.xml("factor", sid(51), SpeechResult="agent", Confidence="0.9")
    attrs = json.loads(root.find("Enqueue/Task").text)
    assert attrs["identity_source"] == "keyed_phone"  # the teardown scans for the digits


def test_keyed_number_lost_fails_closed_to_a_human(kfd):
    """The number lives sealed in call state; a later webhook that can't read it back never
    calls the auth service with another number: the apology and a human."""
    keyed_login(kfd, sid(52))
    from owlbank import callstate as _cs
    _cs.mutate(kfd.store, sid(52), lambda s: {"sealed_sensitive": ()})
    sensitive.forget(sid(52))
    root = kfd.factor(sid(52), THOMAS_ZIP)
    root = kfd.factor(sid(52), THOMAS_SSN4)
    assert root.find("Enqueue") is not None
    assert auth_payloads(kfd, "verify") == []


def test_keyed_level2_step_up_under_on_untrusted_allow(caplog):
    """Under ``on_untrusted: allow`` (test stages) the auth service is told the
    call is keyed, so it applies the untrusted factor rule too: ZIP + SSN-4 at login, then only
    the code for level 2 (before the auth service knew the call was keyed it asked for SSN-4 again and the step-up failed closed)."""
    h = keyed_harness(caplog, on_untrusted="allow")
    call = sid(53)
    try:
        keyed_login(h, call)
        root = h.factor(call, THOMAS_ZIP)
        assert SSN_Q in says(root)  # ZIP alone is not level 1 on a keyed call, also under allow
        h.factor(call, THOMAS_SSN4)
        assert h.state(call).level == 1
        h.set_pending_tool(call, "authenticate", {"tool": "change_address"})
        root = h.handoff(call, {"reason": "step-up", "tool": "change_address", "journal": "jr-ka"})
        assert says(root)[-1] == OTP_Q
        assert h.verify.sent == [(THOMAS, "sms")]
        h.sent_factors.add(OTP)
        h.factor(call, OTP)
        st = h.state(call)
        assert st.level >= 2 and "ani_match" not in st.factors_verified
        assert not [p for p in auth_payloads(h, "verify") if "ssn4" in p["factors"]
                    and "otp" in p["factors"]]  # the step-up sent only the code
        auth = [p for p in h.lam.payloads if p.get("op") in ("precheck", "verify", "send_otp")]
        assert {"precheck", "verify", "send_otp"} <= {p["op"] for p in auth}
        assert {(p["ani"], p["stir_verstat"], p["identity_source"]) for p in auth} == \
            {(THOMAS, None, "keyed_phone")}
    finally:
        keyed_teardown(h, caplog)


def test_ani_calls_leave_identity_source_out(kfd):
    """A call identified by its own ANI sends no ``identity_source``
    (absent means ``ani``), so an auth service from before ``identity_source`` existed accepts it during the update."""
    call = sid(54)
    kfd.start(call, THOMAS)
    kfd.factor(call, THOMAS_ZIP)
    kfd.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    kfd.handoff(call, {"reason": "step-up", "tool": "make_payment", "journal": "jr-a"})
    auth = [p for p in kfd.lam.payloads if p.get("op") in ("precheck", "verify", "send_otp")]
    assert auth and all("identity_source" not in p for p in auth)


# --------------------------------------------------------------------------- and H
# A decline (the
# resolver's ``decline`` or ``#`` alone) at a login or step-up question, a cancel at a login
# security question, and where a failed or declined login lands; a timeout with no input and a
# lone ``*`` are asked again. All at the front door.

UNVERIFIED = "I am the OwlBank virtual agent. How can I help you today?"
DECLINE = ResolverOutput(kind="decline")


def outcomes(h, call: str) -> list[str]:
    return [e.get("outcome") for e in h.journal(call) if e["type"] == "verification"]


def pound(h, call: str) -> ET.Element:
    """``#`` alone on the keypad, as Twilio sends it: empty ``Digits``, ``FinishedOnKey=#``."""
    return h.xml("factor", call, Digits="", FinishedOnKey="#")


def timeout(h, call: str) -> ET.Element:
    """A Gather that timed out with no input: empty ``Digits`` and ``FinishedOnKey``."""
    return h.xml("factor", call, Digits="", FinishedOnKey="")


def star(h, call: str) -> ET.Element:
    """A lone ``*``, as Twilio sends it once the Gather times out: no decline."""
    return h.xml("factor", call, Digits="*", FinishedOnKey="")


def to_question(h, call: str, factor: str) -> None:
    """Drive a call to ``factor``'s question: the keyed phone question, the ZIP at login (trusted
    ANI), SSN-4 at login (keyed call, ZIP held) or the one-time code at a step-up (keyed call)."""
    if factor == "phone":
        assert says(h.start(call, "anonymous")) == [PHONE_Q]
    elif factor == "zip":
        assert says(h.start(call, THOMAS)) == [ZIP_Q]
    elif factor == "ssn4":
        keyed_login(h, call)
        assert SSN_Q in says(h.factor(call, THOMAS_ZIP))
    else:
        keyed_login(h, call)
        h.factor(call, THOMAS_ZIP)
        h.factor(call, THOMAS_SSN4)
        assert h.state(call).level == 1
        h.set_pending_tool(call, "authenticate", {"tool": "change_address"})
        root = h.handoff(call, {"reason": "step-up", "tool": "change_address", "journal": "jr-d"})
        assert says(root)[-1] == OTP_Q


def decline_by(h, call: str, source: str) -> ET.Element:
    if source == "pound":
        return pound(h, call)
    if source == "model":
        h.resolver.default = DECLINE
        try:
            return h.xml("factor", call, SpeechResult="I'd rather not say", Confidence="0.9")
        finally:
            h.resolver.default = ResolverOutput(kind="none")
    return h.xml("factor", call, SpeechResult="cancel", Confidence="0.9")


@pytest.mark.parametrize("source", ["model", "pound"])
def test_phone_decline_takes_the_no_match_path_at_once(kfd, source):
    call = sid(60)
    to_question(kfd, call, "phone")
    root = decline_by(kfd, call, source)
    assert relay(root).get("welcomeGreeting") == UNVERIFIED  # today's no_match ending
    assert keyed_events(kfd, call) == [{"outcome": "declined", "count": 0}]
    assert outcomes(kfd, call) == ["off_script_decline", "fallback_no_match"]
    st = kfd.state(call)
    assert st.login_closed is False and st.attempts == {}  # no account: nothing to close
    assert kfd.lam.ops() == []  # no lookup, no verify


def test_phone_cancel_stays_cancelled(kfd):
    call = sid(61)
    to_question(kfd, call, "phone")
    root = decline_by(kfd, call, "cancel")
    assert relay(root).get("welcomeGreeting") == UNVERIFIED
    assert outcomes(kfd, call) == ["off_script_cancel", "fallback_cancelled"]
    assert kfd.state(call).login_closed is False


@pytest.mark.parametrize("source", ["model", "pound", "cancel"])
def test_zip_decline_or_cancel_goes_to_the_ai_and_closes_the_login(fd, source):
    call = sid(62)
    to_question(fd, call, "zip")
    root = decline_by(fd, call, source)
    assert relay(root).get("welcomeGreeting") == FAILED_AI
    kind = "cancel" if source == "cancel" else "decline"
    assert outcomes(fd, call) == [f"off_script_{kind}", "declined", "fallback_declined"]
    (ev,) = [e for e in fd.journal(call) if e.get("outcome") == "declined"]
    assert ev["factors"] == ["zip"]
    st = fd.state(call)
    assert st.login_closed is True and st.attempts == {} and st.level == 0
    assert "verify" not in fd.lam.ops()


@pytest.mark.parametrize("source", ["model", "pound", "cancel"])
def test_ssn4_decline_on_a_keyed_call_goes_to_the_ai(kfd, source):
    call = sid(63)
    to_question(kfd, call, "ssn4")
    root = decline_by(kfd, call, source)
    assert relay(root).get("welcomeGreeting") == FAILED_AI
    (ev,) = [e for e in kfd.journal(call) if e.get("outcome") == "declined"]
    assert ev["factors"] == ["ssn4"]
    st = kfd.state(call)
    assert st.login_closed is True and st.attempts == {}
    assert auth_payloads(kfd, "verify") == []  # the held ZIP was never sent


@pytest.mark.parametrize("source", ["model", "pound"])
def test_otp_decline_at_a_step_up_returns_declined(kfd, source):
    call = sid(64)
    to_question(kfd, call, "otp")
    verifies = len(auth_payloads(kfd, "verify"))
    decline_by(kfd, call, source)
    env = kfd.pending_result(call)
    assert env.status == "declined" and env.next == "offer_alternative"
    assert len(auth_payloads(kfd, "verify")) == verifies  # the code was never checked
    st = kfd.state(call)
    assert st.level == 1 and st.attempts == {}
    assert st.login_closed is False  # a step-up decline is not a login ending
    (ev,) = [e for e in kfd.journal(call)
             if e["type"] == "verification" and e.get("outcome") == "declined"]
    assert ev["factors"] == ["otp"]


#: The re-ask after an entry that isn't an answer (a timeout, or a lone ``*``), per question.
REASK = {
    "phone": "Please enter the 10 digits of your phone number on your keypad.",
    "zip": "Please enter the 5 digits of your ZIP code on your keypad.",
    "ssn4": "Please enter the 4 digits of your Social Security number on your keypad.",
    "otp": "Please enter the 6 digits of your one-time code on your keypad.",
}


@pytest.mark.parametrize("factor", ["phone", "zip", "ssn4", "otp"])
@pytest.mark.parametrize("entry", ["timeout", "star"])
def test_a_timeout_or_a_lone_star_is_asked_again(kfd, factor, entry):
    """Silence and a lone ``*`` are no decline: the question is asked
    again, with no try used, nothing verified and the login still open."""
    call = sid(73)
    to_question(kfd, call, factor)
    before = (outcomes(kfd, call), keyed_events(kfd, call), len(auth_payloads(kfd, "verify")))
    root = timeout(kfd, call) if entry == "timeout" else star(kfd, call)
    assert root.find("Gather") is not None and says(root) == [REASK[factor]]
    assert (outcomes(kfd, call), keyed_events(kfd, call),
            len(auth_payloads(kfd, "verify"))) == before          # no decline, no lookup, no verify
    st = kfd.state(call)
    assert st.attempts == {} and st.login_closed is False


def test_pound_after_a_lone_star_still_declines(kfd):
    """A ``*`` asked again leaves the question declinable: ``#`` alone then declines."""
    call = sid(74)
    to_question(kfd, call, "phone")
    star(kfd, call)
    root = pound(kfd, call)
    assert relay(root).get("welcomeGreeting") == UNVERIFIED
    assert outcomes(kfd, call) == ["off_script_decline", "fallback_no_match"]


def test_locked_account_goes_to_the_ai_like_a_failed_login(fd):
    fd.lam.overrides["verify"] = {"result": "rate_limited", "level": 0}  # parsed as locked
    fd.start(sid(65), THOMAS)
    root = fd.factor(sid(65), THOMAS_ZIP)
    assert relay(root).get("welcomeGreeting") == FAILED_AI
    assert "locked" in outcomes(fd, sid(65))
    assert fd.state(sid(65)).login_closed is True


def test_verification_risk_still_escalates(fd):
    fd.lam.overrides["verify"] = {"result": "escalate", "reason": "verification_risk"}
    fd.start(sid(66), THOMAS)
    root = fd.factor(sid(66), THOMAS_ZIP)
    attrs = json.loads(root.find("Enqueue/Task").text)
    assert attrs["escalation_reason"] == "verification_risk"  # unchanged
    assert FAILED_AI not in says(root)


def test_declined_login_escalates_when_the_bank_configured_it():
    h = build(config_kw={"fallbacks": {"declined": "escalate"}})
    h.start(sid(67), THOMAS)
    root = pound(h, sid(67))
    assert "I'm sorry, I wasn't able to verify your identity." in says(root)
    assert root.find("Enqueue") is not None and h.state(sid(67)).login_closed is True


def test_step_up_after_a_closed_login_asks_nothing(fd):
    call = sid(68)
    fd.start(call, THOMAS)
    pound(fd, call)
    assert fd.state(call).login_closed is True
    fd.set_pending_tool(call, "authenticate", {"tool": "get_balance"})
    root = fd.handoff(call, {"reason": "step-up", "tool": "get_balance", "journal": "jr-c"})
    assert root.find("Gather") is None and relay(root) is not None  # nothing asked
    env = fd.pending_result(call)
    assert env.status == "not_verified" and env.next == "offer_alternative"
    assert "verify" not in fd.lam.ops() and "precheck" not in fd.lam.ops()[1:]
    assert outcomes(fd, call)[-1] == "login_closed"


def test_a_failed_login_closed_write_still_ends_on_the_fallback(fd, monkeypatch, caplog):
    call = sid(69)
    fd.start(call, THOMAS)
    real = fd.app._update  # noqa: SLF001

    def update(st, **changes):
        if "login_closed" in changes:
            raise RuntimeError("store down")
        return real(st, **changes)

    monkeypatch.setattr(fd.app, "_update", update)
    root = pound(fd, call)
    assert relay(root).get("welcomeGreeting") == FAILED_AI  # the login still ends
    assert fd.state(call).login_closed is False
    assert [r for r in caplog.records if r.getMessage() == "login_closed not saved"]


# --------------------------------------------------------------------------- properties

_PROPS = settings(max_examples=30, deadline=None,
                  suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])


def _harness(**kw):
    """A fresh harness with the keyed-phone switch on (one per example)."""
    def edit(doc: dict) -> None:
        doc["ani_trust"]["ask_phone_on_no_match"] = True
    return build(policy_edit=edit, **kw)


def _forget(h) -> None:
    for call in {k[0].split("#")[0] for k in h.store._tables["calls"]}:  # noqa: SLF001
        clear_sensitive(call)
        sensitive.forget(call)


def _counters(h) -> str:
    """Every lockout and risk counter the auth service keeps (customers, risk), as one value."""
    tables = h.store._tables  # noqa: SLF001
    return json.dumps({t: sorted(map(str, (tables.get(t) or {}).items()))
                       for t in ("customers", "risk")}, sort_keys=True)


@_PROPS
@given(factor=hst.sampled_from(["phone", "zip", "ssn4", "otp"]),
       source=hst.sampled_from(["model", "pound", "cancel"]))
def test_a_decline_never_verifies_or_counts(factor, source):
    """For every declinable question and decline source (and a cancel), the
    flow ends with no ``verify`` call and no change to ``attempts`` or any lockout counter."""
    h = _harness(resolver=FakeResolver())
    call = sid(70)
    try:
        to_question(h, call, factor)
        verifies, attempts, counters = (len(auth_payloads(h, "verify")),
                                        dict(h.state(call).attempts), _counters(h))
        root = decline_by(h, call, source)
        assert root.find("Gather") is None  # the question is not asked again
        assert len(auth_payloads(h, "verify")) == verifies
        assert dict(h.state(call).attempts) == attempts and _counters(h) == counters
    finally:
        _forget(h)


_ENDINGS = ["verify_failed", "locked", "declined", "cancel_factor", "verification_risk",
            "phone_decline"]


def _end_login(h, call: str, ending: str) -> ET.Element:
    if ending == "phone_decline":
        to_question(h, call, "phone")
        return pound(h, call)
    h.start(call, THOMAS)
    if ending == "declined":
        return pound(h, call)
    if ending == "cancel_factor":
        return h.xml("factor", call, SpeechResult="cancel", Confidence="0.9")
    if ending == "locked":
        h.lam.overrides["verify"] = {"result": "rate_limited", "level": 0}
    elif ending == "verification_risk":
        h.lam.overrides["verify"] = {"result": "escalate", "reason": "verification_risk"}
    elif ending == "verify_failed":
        for z in ("94106", "94107"):
            h.factor(call, z)
        return h.factor(call, "94108")
    return h.factor(call, THOMAS_ZIP)


@_PROPS
@given(ending=hst.sampled_from(_ENDINGS),
       vf=hst.sampled_from([None, "limited_ai", "escalate"]),
       dec=hst.sampled_from([None, "limited_ai", "escalate"]))
def test_login_end_routing(ending, vf, dec):
    """``verify_failed``, ``locked`` and ``declined`` (a cancel at a factor too) land in the
    AI with ``verify.failed_ai`` by default and escalate only when configured; ``verification_risk``
    always escalates; a phone decline is always the ``no_match`` path."""
    fallbacks = {k: v for k, v in (("verify_failed", vf), ("declined", dec)) if v is not None}
    h = _harness(config_kw={"fallbacks": fallbacks})
    call = sid(71)
    try:
        root = _end_login(h, call, ending)
        enq = root.find("Enqueue")
        reason = json.loads(enq.find("Task").text)["escalation_reason"] if enq is not None else None
        configured = vf if ending in ("verify_failed", "locked") else dec
        if ending == "verification_risk":
            assert reason == "verification_risk"
        elif ending == "phone_decline":
            assert enq is None and relay(root).get("welcomeGreeting") == UNVERIFIED
            assert h.state(call).login_closed is False
        elif configured == "escalate":
            assert reason == ("step_up_failed" if ending in ("verify_failed", "locked")
                              else "unable_to_resolve")
            assert "I'm sorry, I wasn't able to verify your identity." in says(root)
            assert h.state(call).login_closed is True
        else:
            assert enq is None and relay(root).get("welcomeGreeting") == FAILED_AI
            assert says(root) == []  # one landing line, no verify.failed before it
            assert h.state(call).login_closed is True
    finally:
        _forget(h)


_WORDS = ["purple", "giraffe", "lantern", "marble", "pickle", "velvet", "orchid", "tundra",
          "quartz", "nimbus"]


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage() + " " + str(record.__dict__))


@_PROPS
@given(factor=hst.sampled_from(["phone", "zip", "ssn4"]),
       words=hst.lists(hst.sampled_from(_WORDS), min_size=2, max_size=6))
def test_no_caller_words_in_decline_logs_or_journal(factor, words):
    """A decline is logged and journaled with the factor name and outcome only. With the
    speech log off (its ``factor_answer.speech`` is the one line that carries words on purpose), none of the caller's words reach any log line, journal event or stored item."""
    h = _harness(resolver=FakeResolver(), config_kw={"speech_log": False})
    call = sid(72)
    said = " ".join(words)
    rec = _Records()
    root_log = logging.getLogger()
    old_level = root_log.level
    root_log.addHandler(rec)
    root_log.setLevel(logging.DEBUG)
    try:
        to_question(h, call, factor)
        rec.lines.clear()
        h.resolver.default = DECLINE
        h.xml("factor", call, SpeechResult=said, Confidence="0.9")
        journal = [json.dumps(e) for e in h.journal(call)]
        for w in set(words):
            for where, texts in (("log", rec.lines), ("journal", journal),
                                 ("store", h.stored_strings())):
                assert not [t for t in texts if w in t], f"caller words in the {where}"
        allowed = {"type", "outcome", "factors", "level", "count", "action",
                   "call_sid", "seq", "ts", "expires_at"}
        for e in h.journal(call):
            if e["type"] in ("verification", "keyed_lookup") and "declined" in str(e.get("outcome")):
                assert set(e) <= allowed
    finally:
        root_log.removeHandler(rec)
        root_log.setLevel(old_level)
        _forget(h)
