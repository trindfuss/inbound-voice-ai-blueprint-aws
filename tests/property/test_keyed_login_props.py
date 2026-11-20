"""Keyed-in phone login properties, over signed webhooks into
the real front door, auth service and bank tools (``tests.component.test_frontdoor_support.build``).

* Never trusted: for any ``StirVerstat``, ``on_untrusted`` and level, a keyed call is asked
  exactly ``effective_factors(level, ani_counts=False)``, reaches the level, never holds
  ``ani_match``, and every auth-service call carries the keyed number, ``stir_verstat=None`` and
  ``identity_source="keyed_phone"``.
* A matched ANI is unchanged: the ``phone`` slot is never asked, and the prompts are the same
  with the switch on or off.
* Bounded lookups: for any answer sequence, at most two keyed lookups; two misses reach the
  ``no_match`` fallback.
* Format: the lookup and the auth calls receive exactly ``"+1" + digits``.
* No leak: the keyed number (10 digits or E.164) is in no log record, journal event, stored
  item, response or task attribute.
* OTP destination: a code goes only to a number on file for the customer; send, async send and
  verify get the same ``ani``; on a keyed call never before ZIP + SSN-4 (level 0 escalates). All
  four auth calls (precheck, send, async send, verify) carry ``identity_source="keyed_phone"`` on
  a keyed call and ``"ani"`` otherwise.
* Miss wording: the prompts after a miss don't depend on which number missed or why.
* The auth service and a keyed call: for any ``StirVerstat``, ``on_untrusted`` and target
  level, a ``keyed_phone`` request answers exactly as the untrusted rule (an ANI request with no
  verstat under ``no_ani_match``) does, never counts ``ani_match`` and never logs
  ``ani_trusted: true``.
"""

from __future__ import annotations

import copy
import json
import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from lambdas.frontdoor import http
from owlbank import sensitive
from owlbank.policy import Policy
from owlbank.segments import clear_sensitive
from tests.component.test_frontdoor_support import (
    DATA, LEES, NOBODY, OTP, STAGE, TEST_FACTORS, THOMAS, Harness, all_fixture_factor_values,
    assert_no_leak, build, sid,
)

PROPS = settings(max_examples=40, deadline=None,
                 suppress_health_check=[HealthCheck.too_slow,
                                        HealthCheck.function_scoped_fixture])
POLICY = Policy.from_dict(json.loads((DATA / "policy.json").read_text()))
CALL = sid(60)
TRUSTED = "TN-Validation-Passed-A"
VERSTATS = st.sampled_from([None, TRUSTED, "TN-Validation-Passed-B", "TN-Validation-Passed-C",
                            "No-TN-Validation", "TN-Validation-Failed-A"])
NO_MATCH_ANIS = st.sampled_from(["anonymous", "", NOBODY, "restricted"])
Z, S = TEST_FACTORS["C-1001"]["zip"], TEST_FACTORS["C-1001"]["ssn4"]
#: NANP-shaped (area code and exchange start 2-9), so no number looks like a timestamp.
NUMBERS = st.builds(lambda a, b, c: a + b + c,
                    st.from_regex(r"[2-9][0-9]{2}", fullmatch=True),
                    st.from_regex(r"[2-9][0-9]{2}", fullmatch=True),
                    st.from_regex(r"[0-9]{4}", fullmatch=True))
FIXTURE_PHONES = {THOMAS[2:], LEES[2:], "2025550103"}
UNKNOWN = NUMBERS.filter(lambda n: n not in FIXTURE_PHONES)


def _edit(switch: bool = True, on_untrusted: str | None = None):
    def edit(doc: dict[str, Any]) -> None:
        doc["ani_trust"]["ask_phone_on_no_match"] = switch
        if on_untrusted is not None:
            doc["ani_trust"]["on_untrusted"] = on_untrusted
    return edit


@contextmanager
def harness(**kw: Any) -> Iterator[Harness]:
    h = build(policy_edit=_edit(kw.pop("switch", True), kw.pop("on_untrusted", None)), **kw)
    try:
        yield h
    finally:
        for call in {k[0].split("#")[0] for k in h.store._tables["calls"]}:  # noqa: SLF001
            clear_sensitive(call)
            sensitive.forget(call)


def incoming(h: Harness, call: str, ani: str, verstat: str | None) -> None:
    extra = {"StirVerstat": verstat} if verstat is not None else {}
    h.xml("incoming", call, From=ani, To="+12025550100", CallStatus="ringing", **extra)
    h.xml("language", call, Digits="")


def asked(h: Harness, call: str) -> list[str]:
    return [e["factor"] for e in h.journal(call) if e["type"] == "factor_asked"]


def prompt_keys(h: Harness, call: str) -> list[str]:
    return [e["prompt_key"] for e in h.journal(call) if e["type"] == "prompt"]


def payloads(h: Harness, *ops: str) -> list[dict[str, Any]]:
    return [p for p in h.lam.payloads if p.get("op") in ops]


def keyed_lookups(h: Harness) -> list[str]:
    """The ``ani_lookup`` arguments after the call's own ANI lookup (that one is first, if any)."""
    return [p["args"]["ani"] for p in payloads(h, "ani_lookup")]


def add_customer(h: Harness, number: str) -> None:
    """A copy of Thomas (same factors) whose phone on file is ``+1`` + ``number``."""
    thomas = copy.deepcopy(h.store.get("customers", {"customer_id": "C-1001"}))
    thomas.update(customer_id="C-9001", phone="+1" + number)
    h.store.put("customers", thomas)


def phone_open(h: Harness, call: str) -> bool:
    flow = h.session(call).flow or {}
    return flow.get("kind") == "login" and flow.get("factor") == "phone"




@PROPS
@given(verstat=VERSTATS, mode=st.sampled_from(["no_ani_match", "allow"]),
       level=st.sampled_from([1, 2, 3]), ani=NO_MATCH_ANIS)
def test_a_keyed_call_is_never_trusted(verstat, mode, level, ani) -> None:
    with harness(on_untrusted=mode) as h:
        incoming(h, CALL, ani, verstat)
        h.factor(CALL, THOMAS[2:])
        h.factor(CALL, Z)
        h.factor(CALL, S)
        login = asked(h, CALL)
        assert login == ["phone", *POLICY.effective_factors(1, ani_counts=False)]
        assert h.state(CALL).level == 1
        if level > 1:
            tool = {2: "change_address", 3: "make_payment"}[level]
            h.set_pending_tool(CALL, "authenticate", {"tool": tool})
            h.handoff(CALL, {"reason": "step-up", "tool": tool, "journal": "jr-kp1"})
            have = set(POLICY.effective_factors(1, ani_counts=False))
            want = [f for f in POLICY.effective_factors(level, ani_counts=False) if f not in have]
            assert asked(h, CALL)[len(login):] == want
            h.factor(CALL, OTP)
            assert h.state(CALL).level >= level  # under ``allow`` too
        st_ = h.state(CALL)
        assert st_.identity_source == "keyed_phone" and "ani_match" not in st_.factors_verified
        auth = payloads(h, "precheck", "verify", "send_otp")
        assert auth and {(p["ani"], p["stir_verstat"], p["identity_source"]) for p in auth} == \
            {(THOMAS, None, "keyed_phone")}
        # The untrusted-caller gate ran before the first factor, whatever the verstat.
        assert h.lam.ops().index("precheck") < h.lam.ops().index("verify")




ANSWERS = st.lists(st.sampled_from([Z, S, "94106", "0000", "", "12"]), min_size=1, max_size=4)


@PROPS
@given(ani=st.sampled_from([THOMAS, LEES, "+12025550103"]), verstat=VERSTATS, answers=ANSWERS)
def test_a_matched_ani_is_unchanged(ani, verstat, answers) -> None:
    runs = []
    for switch in (True, False):
        with harness(switch=switch) as h:
            incoming(h, CALL, ani, verstat)
            for a in answers:
                if h.session(CALL).flow is None or not (h.session(CALL).flow or {}).get("factor"):
                    break
                h.factor(CALL, a)
            assert "phone" not in asked(h, CALL)
            assert not [e for e in h.journal(CALL) if e["type"] == "keyed_lookup"]
            runs.append((asked(h, CALL), prompt_keys(h, CALL), h.lam.ops(), h.state(CALL).level))
    assert runs[0] == runs[1]




ANSWER = st.one_of(UNKNOWN, st.sampled_from(["", "12", "12345", "123456789012"]),
                   st.just(THOMAS[2:]))


@PROPS
@given(ani=NO_MATCH_ANIS, answers=st.lists(ANSWER, min_size=1, max_size=10))
def test_at_most_two_keyed_lookups_and_two_misses_reach_no_match(ani, answers) -> None:
    with harness() as h:
        incoming(h, CALL, ani, TRUSTED)
        own = len(payloads(h, "ani_lookup"))
        for a in answers:
            if not phone_open(h, CALL):
                break
            h.factor(CALL, a)
        keyed = keyed_lookups(h)[own:]
        assert len(keyed) <= 2
        events = [e["outcome"] for e in h.journal(CALL) if e["type"] == "keyed_lookup"]
        assert len([o for o in events if o != "unclear"]) == len(keyed)
        if events.count("no_match") == 2:
            fallbacks = [e["outcome"] for e in h.journal(CALL) if e["type"] == "verification"]
            assert fallbacks == ["fallback_no_match"] and not phone_open(h, CALL)
            assert h.state(CALL).identity_source == "ani"


@PROPS
@given(ani=NO_MATCH_ANIS, misses=st.lists(st.one_of(UNKNOWN, st.sampled_from(["", "12"])),
                                          min_size=6, max_size=6))
def test_the_words_after_a_miss_never_depend_on_the_number(ani, misses) -> None:
    """Two keyed misses (any numbers, any unclear answers between them) speak the same keys as a
    call with two fixed unknown numbers, and end in today's ``no_match`` welcome."""
    def run(answers: list[str]) -> list[str]:
        with harness() as h:
            incoming(h, CALL, ani, TRUSTED)
            for a in answers:
                if not phone_open(h, CALL):
                    break
                h.factor(CALL, a)
            return [k for k in prompt_keys(h, CALL) if not k.startswith("clarify.")]
    numbers = [m for m in misses if len(m) == 10][:2]
    assume(len(numbers) == 2)
    reference = run(["2025550177", "2025550188"])
    assert run(numbers) == reference
    assert reference[-2:] == ["login.phone_number_retry", "greeting.unverified"]




@PROPS
@given(number=NUMBERS.filter(lambda n: n not in FIXTURE_PHONES), on_file=st.booleans(),
       ani=NO_MATCH_ANIS)
def test_format_and_no_leak(number, on_file, ani, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    caplog.clear()
    with harness() as h:
        if on_file:
            add_customer(h, number)
        incoming(h, CALL, ani, TRUSTED)
        own = len(payloads(h, "ani_lookup"))
        h.factor(CALL, number)
        assert keyed_lookups(h)[own:] == ["+1" + number]
        if on_file:
            h.factor(CALL, Z)
            h.factor(CALL, S)
            assert h.state(CALL).level == 1
            h.set_pending_tool(CALL, "authenticate", {"tool": "change_address"})
            h.handoff(CALL, {"reason": "step-up", "tool": "change_address", "journal": "jr-kp5"})
            h.factor(CALL, OTP)
            auth = payloads(h, "precheck", "verify", "send_otp")
            assert {p["ani"] for p in auth} == {"+1" + number}
            assert h.verify.sent == [("+1" + number, "sms")]
            h.xml("connect-action", CALL, CallStatus="in-progress", SessionStatus="ended",
                  SessionId="VX" + "c" * 32, SessionDuration="12",
                  HandoffData=json.dumps({"reason": "escalate", "journal": "jr-kp5b"}))
        else:
            h.factor(CALL, "2025550177")
        values = all_fixture_factor_values() | h.sent_factors | {number, "+1" + number}
        assert_no_leak(h, caplog.records, values)




def _req(call: str) -> http.TwilioRequest:
    return http.TwilioRequest("factor", {"CallSid": call}, STAGE, origin="https://h.example")


def _ack(h: Harness):
    return [h.app._say("factor.ack", "en-US")]  # noqa: SLF001


@PROPS
@given(entry=st.sampled_from(["keyed_withheld", "keyed_unmatched", "ani_untrusted"]),
       level=st.sampled_from([0, 1]), verstat=VERSTATS, ack_first=st.booleans())
def test_the_code_goes_only_to_a_number_on_file(entry, level, verstat, ack_first) -> None:
    keyed = entry.startswith("keyed")
    if not keyed and verstat == TRUSTED:
        verstat = None  # an untrusted ANI: the same factor set as a keyed call
    with harness() as h:
        ani = {"keyed_withheld": "anonymous", "keyed_unmatched": NOBODY,
               "ani_untrusted": THOMAS}[entry]
        incoming(h, CALL, ani, verstat)
        if keyed:
            h.factor(CALL, THOMAS[2:])
        if level == 1:
            h.factor(CALL, Z)
            h.factor(CALL, S)
            assert h.state(CALL).level == 1
            h.set_pending_tool(CALL, "authenticate", {"tool": "change_address"})
            root = h.handoff(CALL, {"reason": "step-up", "tool": "change_address",
                                    "journal": "jr-kp6"})
            assert root.find("Gather") is not None
            if ack_first:  # the asynchronous send of the same code, for the same flow
                st_, sess = h.app._load(CALL)  # noqa: SLF001
                out = h.app._ack_while_sending(_req(CALL), st_, sess, "otp", _ack(h))  # noqa: SLF001
                assert out is not None and "Redirect" in out
                h.xml("factor", CALL)  # the redirect: waits for the send, asks the code
            h.factor(CALL, OTP)
            sends = payloads(h, "send_otp")
            verifies = [p for p in payloads(h, "verify") if "otp" in p["factors"]]
            assert len(sends) == (2 if ack_first else 1) and len(verifies) == 1
            assert {p["ani"] for p in sends + verifies} == {THOMAS}
            assert {p["stir_verstat"] for p in sends + verifies} == {None if keyed else verstat}
            # All four calls say how the caller was identified (the async send too).
            auth = payloads(h, "precheck", "send_otp", "verify")
            assert {"precheck", "send_otp", "verify"} <= {p["op"] for p in auth}
            if ack_first:
                assert [p for p in sends if p.get("record") is True]
            # An ANI call leaves the field out (absent = ani).
            assert {p.get("identity_source") for p in auth} == {"keyed_phone" if keyed else None}
            customer = h.state(CALL).verified_customer_id
            for to, _channel in h.verify.sent:
                on_file = h.store.query("customers", {"phone": to}, index="phone-index")
                assert customer in {c.get("alias_of", c["customer_id"]) for c in on_file}
        else:  # level 0: the login flow is open; a code is never sent before ZIP + SSN-4
            st_, sess = h.app._load(CALL)  # noqa: SLF001
            assert st_.level == 0
            out = h.app._ask(_req(CALL), st_, sess, "otp", pre=_ack(h),  # noqa: SLF001
                             ack_first=ack_first)
            attrs = json.loads(ET.fromstring(out.encode()).find("Enqueue/Task").text)
            assert attrs["escalation_reason"] == "verification_risk"
            assert payloads(h, "send_otp") == [] and h.verify.sent == []
            assert {p.get("identity_source") for p in payloads(h, "precheck", "verify")} <= \
                {"keyed_phone" if keyed else None}


@pytest.fixture(autouse=True)
def _sensitive_reset() -> Iterator[None]:
    yield
    clear_sensitive(CALL)
    sensitive.forget(CALL)




class _Audits(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.rows: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.rows.append(dict(record.__dict__))


@contextmanager
def _audits() -> Iterator[_Audits]:
    logger = logging.getLogger("owlbank.audit")
    handler, old = _Audits(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old)


class _Verify:
    """Twilio Verify fake: ``OTP`` is the one valid code."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.checked: list[str] = []

    def send(self, to: str, channel: str, locale: str | None = None) -> None:
        self.sent.append((to, channel))

    def check(self, to: str, code: str) -> bool:
        self.checked.append(to)
        return code == OTP


_FIXTURES = json.loads((DATA / "fixtures.json").read_text())["customers"]


def _auth_deps(mode: str) -> Any:
    from lambdas.auth_service import service
    from owlbank.store import MemoryStore

    doc = json.loads((DATA / "policy.json").read_text())
    doc["ani_trust"]["on_untrusted"] = mode
    store = MemoryStore()
    for c in _FIXTURES:
        store.put("customers", copy.deepcopy(c))
    return service.AuthDeps(store=store, policy=Policy.from_dict(doc), verify_client=_Verify(),
                            clock=lambda: 1_760_000_000, token_max_ttl_s=1800,
                            record_ttl_s=86400)


def _run(mode: str, verstat: Any, source: str, *, step_up: bool, level: int,
         factors: dict[str, str], send: bool) -> tuple[Any, ...]:
    """One call's auth-service conversation; everything observable but the token values."""
    from lambdas.auth_service import service

    d = _auth_deps(mode)
    kw = {"call_sid": CALL, "ani": THOMAS, "stir_verstat": verstat, "identity_source": source}
    out: list[Any] = []
    if step_up:
        out.append(service.verify(d, factors={"zip": Z, "ssn4": S}, target_level=1, **kw))
    out.append(service.precheck(d, target_level=level, **kw))
    if send:
        out.append(service.send_otp(d, **kw))
    out.append(service.verify(d, factors=factors, target_level=level, **kw))
    out = [{k: v for k, v in o.items() if k != "token"} for o in out]
    tables = d.store._tables  # noqa: SLF001
    return (out, sorted(map(str, tables["calls"])), sorted(map(str, tables["risk"])),
            d.verify_client.sent, d.verify_client.checked)


FACTOR_VALUES = st.fixed_dictionaries({}, optional={
    "zip": st.sampled_from([Z, "00000"]), "ssn4": st.sampled_from([S, "0000"]),
    "otp": st.sampled_from([OTP, "000000"])})


@settings(max_examples=300, deadline=None)
@given(verstat=st.one_of(VERSTATS, st.sampled_from(["", "junk value!", 7])),
       mode=st.sampled_from(["no_ani_match", "allow"]), level=st.sampled_from([1, 2, 3]),
       step_up=st.booleans(), send=st.booleans(), factors=FACTOR_VALUES)
def test_a_keyed_request_never_takes_the_trusted_rule(verstat, mode, level, step_up, send,
                                                          factors) -> None:
    assume(factors)
    if step_up:
        level = max(level, 2)  # a step-up goes above the level-1 login
    with _audits() as log:
        keyed = _run(mode, verstat, "keyed_phone", step_up=step_up, level=level,
                     factors=factors, send=send)
    # The untrusted rule: an ANI request with no verstat under ``no_ani_match``.
    reference = _run("no_ani_match", None, "ani", step_up=step_up, level=level,
                     factors=factors, send=send)
    assert keyed == reference
    for result in keyed[0]:
        assert "ani_match" not in result.get("factors_verified", [])
    assert not [r for r in log.rows if r.get("ani_trusted") is True]
    assert not [r for r in log.rows if r.get("ani_match_counted") is True]
    assert not [r for r in log.rows if r.get("customer_lockout_counted") is True]
