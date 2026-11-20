"""Segment-traits properties over generated logins.

Generated: the caller's language, whether the caller ID is trusted (one factor, or ZIP then SSN-4),
what the stand-in ``prepare_welcome`` writes (a matched offer with or without a phrase, another
reason, another customer, another language, a record sealed for another call, nothing), when it
lands (at the invoke, after it but before the answer, or only after the call connected), and the
caller's answers (right or wrong, any number). Properties:

- The offer is spoken only after verification succeeds, only to its customer and in its
  language: no response before verification holds an offer phrase, and ``welcome_offer`` is unset
  until the login connect, which sets it with the offer; a later connect (a relay-drop
  reconnect) keeps it as it was, and never sets it by itself.
- Every login that doesn't get the offer gets ``greeting.personal`` exactly and no
  ``welcome_offer``; a failed login never gets either.
- At most one ``prepare_welcome`` per call; none without an identified customer.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from hypothesis import HealthCheck, event, example, given, settings
from hypothesis import strategies as st

from owlbank import callstate, last_spoken, welcome
from owlbank.catalog import PromptCatalog
from tests.component.test_frontdoor_support import (
    CLOCK,
    DATA,
    TEST_FACTORS,
    THOMAS,
    all_fixture_factor_values,
    assert_no_leak,
    build,
    sid,
)
from tests.component.test_frontdoor_welcome import Prep, install

PROPS = settings(max_examples=200, deadline=None,
                 suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
CATALOG = PromptCatalog.from_dict(json.loads((DATA / "prompt_catalog.json").read_text()))
PHRASES = tuple(CATALOG.template(k, lang) for k in CATALOG.keys if k.startswith(welcome.OFFER_KEY_PREFIX)
                for lang in CATALOG.languages if CATALOG.has(k, lang))
RIGHT = TEST_FACTORS["C-1001"]
WRONG = {"zip": "98101", "ssn4": "8642"}
CALL = sid(40)

#: What the stand-in writes. The offer path is listed four times: it is the one the offer property is about.
_PREP_KINDS = ["offer"] * 4 + ["no_phrase", "reason", "none", "other_call", "other_customer",
                                "other_language"]
preps = st.sampled_from(_PREP_KINDS).flatmap(lambda kind: {
    "offer": st.builds(Prep, offer_id=st.sampled_from(["heloc", "auto_loan"])),
    "no_phrase": st.just(Prep(offer_id="mortgage")),
    "reason": st.builds(Prep, kind=st.sampled_from(sorted(welcome.REASONS - {welcome.MATCHED}))),
    "none": st.just(Prep(kind="none")),
    "other_call": st.just(Prep(kind="other_call")),
    "other_customer": st.builds(Prep, customer_id=st.sampled_from(["C-1002", "C-1004"])),
    "other_language": st.just(Prep(language="other")),
}[kind])


#: The caller's answers in order (``True`` right, ``False`` wrong): some wrong ones (three use up
#: the login's attempts), then, or not, right ones; or any mix.
_answers = st.one_of(
    st.builds(lambda wrong, finish: [False] * wrong + ([True] * 4 if finish else []),
              st.integers(0, 4), st.booleans()),
    st.lists(st.booleans(), max_size=7),
)


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _offer_in(body: str) -> bool:
    return any(p in body for p in PHRASES)


@PROPS
@given(spanish=st.booleans(), trusted=st.booleans(), prep=preps,
       timing=st.sampled_from(["now", "before", "late"]),
       answers=_answers)
@example(spanish=False, trusted=True, prep=Prep(), timing="now", answers=[True])
@example(spanish=True, trusted=False, prep=Prep(offer_id="auto_loan"), timing="before",
         answers=[True, False, True, True])
@example(spanish=False, trusted=False, prep=Prep(), timing="now", answers=[False, False, False, False, False, False])
def test_offer_only_after_verification_to_its_customer_and_language(
        spanish: bool, trusted: bool, prep: Prep, timing: str, answers: list[bool]) -> None:
    lang = "es-US" if spanish else "en-US"
    if prep.language == "other":
        prep = Prep(kind=prep.kind, offer_id=prep.offer_id, language="en-US" if spanish else "es-US")
    h = build(config_kw={"memory_trait_groups": ("Contact", "WebActivity")})
    install(h, prep)
    h.lam.defer_events = timing != "now"
    cap = _Records()
    root_logger = logging.getLogger()
    old = root_logger.level
    root_logger.addHandler(cap)
    root_logger.setLevel(logging.DEBUG)
    try:
        _run(h, lang, trusted, timing, answers, prep)
        assert_no_leak(h, cap.records, all_fixture_factor_values() | h.sent_factors)
    finally:
        root_logger.removeHandler(cap)
        root_logger.setLevel(old)
        from owlbank import segments, sensitive
        segments.clear_sensitive(CALL)
        sensitive.forget(CALL)


def _check_before_verification(h: Any) -> None:
    s = h.state(CALL)
    if s.level == 0:
        assert not _offer_in(h.bodies[-1]), "an offer phrase before verification"
        assert s.welcome_offer is None


def _run(h: Any, lang: str, trusted: bool, timing: str, answers: list[bool], prep: Prep) -> None:
    incoming: dict[str, str] = {"From": THOMAS, "To": "+12025550100", "CallStatus": "ringing"}
    if trusted:
        incoming["StirVerstat"] = "TN-Validation-Passed-A"
    h.xml("incoming", CALL, **incoming)
    h.xml("language", CALL, Digits="2" if lang == "es-US" else "")
    _check_before_verification(h)
    if timing == "before":
        h.lam.run_deferred()
    for right in answers:
        flow = h.session(CALL).flow or {}
        factor = flow.get("factor")
        if flow.get("kind") != "login" or factor not in WRONG:
            break
        value = RIGHT[factor] if right else WRONG[factor]
        h.factor(CALL, value)
        _check_before_verification(h)
        if timing == "before":
            h.lam.run_deferred()
    assert len(h.lam.welcome_requests()) <= 1
    st_ = h.state(CALL)
    flow = h.session(CALL).flow or {}
    in_login = flow.get("kind") == "login"
    text = last_spoken.unseal(h.sealer, st_.last_spoken, call_sid=CALL, now=CLOCK)
    if st_.level >= 1:  # the login connected (only now may the offer be spoken)
        assert h.lam.welcome_requests() == [{"call_sid": CALL, "customer_id": "C-1001", "language": lang}]
        offer = None
        if (prep.kind == "matched" and timing != "late" and prep.customer_id in (None, "C-1001")
                and prep.language in (None, lang)):
            offer = welcome.render_offer(CATALOG, lang, "Thomas", prep.offer_id)
        personal = CATALOG.text("greeting.personal", lang, firstName="Thomas")
        event("login: offer" if offer is not None else "login: greeting.personal")
        if offer is not None:
            assert text == offer
            assert st_.welcome_offer == callstate.WELCOME_OFFER_MADE
        else:
            assert text == personal
            assert st_.welcome_offer is None
        assert st_.welcome_prepared is None
    else:
        event("not verified")
        assert st_.welcome_offer is None
        assert not any(_offer_in(b) for b in h.bodies)
        if not in_login:  # the login ended without verification: the record is discarded
            assert st_.welcome_prepared is None
    if timing == "late":  # a record that lands after the connect changes nothing spoken
        before = (st_.welcome_offer, text)
        h.lam.run_deferred()
        after = h.state(CALL)
        assert (after.welcome_offer,
                last_spoken.unseal(h.sealer, after.last_spoken, call_sid=CALL, now=CLOCK)) == before
    if st_.level >= 1:  # a later connect keeps welcome_offer as it was
        kept = h.state(CALL).welcome_offer
        h.xml("connect-action", CALL, CallStatus="in-progress", SessionStatus="failed",
              SessionId="VX" + "c" * 32, ErrorCode="64107")
        assert h.state(CALL).welcome_offer == kept
