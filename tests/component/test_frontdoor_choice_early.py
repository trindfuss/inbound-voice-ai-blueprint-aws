"""Early finish for choice answers.

The card question and its follow-ups that list options report partial results like a digit
factor's Gather. A partial that is already an exact deterministic match (a spoken name or alias,
or an ordinal over the options just listed) of exactly one of the caller's cards ends the Gather
at once; the redirect answers the card as ``match: exact, source: early`` and the read-back
follows. Anything else ("red no", "the OwlBank card", "yes") waits for the final result. One
answer per Gather whatever the order of partials, the redirect and the final webhook.
Yes/no Gathers (the read-back, a not-available question with one card) never finish early.

``factor_answer`` carries ``timings_ms``: span names and whole milliseconds only.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from lambdas.frontdoor import gathers
from lambdas.frontdoor.calls import FakeCallUpdater
from owlbank import sensitive
from owlbank.segments import clear_sensitive
from owlbank.slots import FakeResolver, ResolverOutput
from tests.component.test_frontdoor_card import (
    PROMPT,
    PURPLE,
    RED,
    REWARDS,
    payments,
    pending_to,
    propose,
    readback,
    resolutions,
    says,
    to_level3,
)
from tests.component.test_frontdoor_support import (
    BASIC,
    HOST,
    STAGE,
    TEST_FACTORS,
    OTP,
    Harness,
    all_fixture_factor_values,
    assert_no_leak,
    build,
    sid,
)

PARTIAL_KEYS = {"slot", "slot_type", "chars", "ms_since_gather", "outcome"}
SPAN = re.compile(r"[a-z][a-z0-9_]{0,39}")


@pytest.fixture(autouse=True)
def _forget_calls() -> Any:
    yield
    for n in range(100):
        clear_sensitive(sid(n))
        sensitive.forget(sid(n))


def make(*outs: ResolverOutput, fail: bool = False, card: dict[str, Any] | None = None,
         **config_kw: Any) -> tuple[Harness, FakeCallUpdater]:
    calls = FakeCallUpdater(fail=fail)
    h = build(resolver=FakeResolver(list(outs)), config_kw=config_kw, calls=calls)
    if card:
        import json

        from owlbank.policy import Policy
        from tests.component.test_frontdoor_support import DATA

        raw = json.loads((DATA / "policy.json").read_text())
        raw["slots"]["card"].update(card)
        h.app.d.policy = Policy.from_dict(raw)
    h.sent_factors |= {TEST_FACTORS["C-1001"]["zip"], TEST_FACTORS["C-1001"]["ssn4"], OTP,
                       TEST_FACTORS["C-1002"]["zip"], TEST_FACTORS["C-1002"]["ssn4"]}
    return h, calls


def gather_of(root: ET.Element) -> ET.Element:
    g = root.find("Gather")
    assert g is not None, ET.tostring(root)
    return g


def gid_of(g: ET.Element) -> str:
    return parse_qs(urlsplit(g.get("action") or "").query)["g"][0]


def route_of(url: str) -> str:
    return url.rsplit("/voice/", 1)[1]


def partial(h: Harness, call: str, g: ET.Element, text: str) -> None:
    resp = h.post(route_of(g.get("partialResultCallback") or ""), call,
                  UnstableSpeechResult=text, SequenceNumber="1")
    assert resp == {"statusCode": 204, "headers": {}, "body": ""}


def redirect_route(twiml: str) -> str:
    url = ET.fromstring(twiml.encode()).find("Redirect").text or ""
    parts = urlsplit(url)
    assert parts.scheme == "https" and parts.hostname == HOST
    assert (parts.username, parts.password) == BASIC
    assert parts.path == f"{STAGE}/voice/factor"
    return route_of(url)


def final(h: Harness, call: str, g: ET.Element, text: str) -> dict[str, Any]:
    return h.post(route_of(g.get("action") or ""), call, SpeechResult=text, Confidence="0.9")


def events(records: list[logging.LogRecord], name: str) -> list[logging.LogRecord]:
    return [r for r in records if getattr(r, "event", None) == name]


def card_answers(h: Harness, call: str) -> list[dict[str, Any]]:
    return [e for e in resolutions(h, call) if e["slot"] == "card"]


def at_card_question(h: Harness, call: str, who: str = "C-1001", **kw: Any) -> ET.Element:
    to_level3(h, call, who)
    return gather_of(propose(h, call, **kw))


# --------------------------------------------------------------------------- the Gathers


def test_the_card_question_reports_partials(caplog: pytest.LogCaptureFixture) -> None:
    h, _ = make()
    call = sid(1)
    g = at_card_question(h, call, absent=True)
    gid = gid_of(g)
    assert g.get("input") == "dtmf speech"
    assert g.get("partialResultCallback") == f"{STAGE}/voice/partial?g={gid}"
    assert g.get("partialResultCallbackMethod") == "POST"
    assert h.session(call).flow["gather"]["id"] == gid


def test_listing_follow_ups_report_partials_and_yes_no_ones_do_not() -> None:
    h, _ = make(ResolverOutput(kind="ambiguous", candidates=["red", "purple"]),
                ResolverOutput(kind="inferred", value="purple"))
    call = sid(2)
    g = at_card_question(h, call, absent=True)
    root = h.xml("factor", call, SpeechResult="the OwlBank card", Confidence="0.9")
    g = gather_of(root)                                           # the disambiguation
    assert g.get("partialResultCallback") and "g=" in (g.get("action") or "")
    root = final(h, call, g, "the violet one")                    # "Did you mean ...?" (yes/no)
    g = gather_of(ET.fromstring(root["body"].encode()))
    assert g.get("partialResultCallback") is None and "?" not in (g.get("action") or "")


def test_not_available_with_several_cards_reports_partials_with_one_card_it_does_not() -> None:
    h, _ = make()
    call = sid(3)
    g = at_card_question(h, call, to="rewards")                  # two cards
    assert g.get("partialResultCallback")
    call2 = sid(4)
    g = at_card_question(h, call2, "C-1002", to="purple")        # one card: a yes/no question
    assert g.get("partialResultCallback") is None


def test_the_read_back_never_reports_partials() -> None:
    h, _ = make()
    call = sid(5)
    g = at_card_question(h, call, to="red")
    assert g.get("partialResultCallback") is None and "?" not in (g.get("action") or "")


def test_a_keypad_only_card_never_reports_partials() -> None:
    """Keypad-only, no Gather id, so a partial can't finish it."""
    h, _ = make(card={"keypad_only": True})
    call = sid(6)
    g = at_card_question(h, call, absent=True)
    assert g.get("input") == "dtmf" and g.get("partialResultCallback") is None
    assert "?" not in (g.get("action") or "")


@pytest.mark.parametrize("why", ["setting_off", "no_updater"])
def test_off_gives_todays_card_gather(why: str) -> None:
    if why == "setting_off":
        h, _ = make(early_digits=False)
    else:
        h = build(resolver=FakeResolver())
    call = sid(7)
    g = at_card_question(h, call, absent=True)
    assert g.get("partialResultCallback") is None and "?" not in (g.get("action") or "")


# --------------------------------------------------------------------------- early finish


def test_red_finishes_early_and_the_read_back_follows(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    h, calls = make()
    call = sid(10)
    g = at_card_question(h, call, absent=True)
    partial(h, call, g, "red")
    assert len(calls.updates) == 1 and calls.updates[0][0] == call
    assert pending_to(h, call) is None                    # nothing selected on a partial
    partial(h, call, g, "red card")                       # later partials: already claimed
    assert len(calls.updates) == 1
    first = h.post(redirect_route(calls.updates[0][1]), call)
    assert says(ET.fromstring(first["body"].encode())) == [readback(RED)]
    assert pending_to(h, call) == "red" and h.resolver.calls == []
    # Twilio's final webhook for the same Gather, and a replayed redirect: the same TwiML.
    assert final(h, call, g, "red card")["body"] == first["body"]
    assert h.post(redirect_route(calls.updates[0][1]), call)["body"] == first["body"]
    (ans,) = card_answers(h, call)
    assert ans["match"] == "exact" and ans["source"] == "early" and ans["value"] == "red"
    (line,) = [r for r in events(caplog.records, "factor_answer") if r.factor == "card"]
    assert line.match == "exact" and line.source == "early"
    outcomes = [r.outcome for r in events(caplog.records, "factor_partial")]
    assert outcomes == ["complete", "already_claimed"]
    for r in events(caplog.records, "factor_partial"):
        assert r.slot == "card" and r.slot_type == "choice"
        assert set(PARTIAL_KEYS) <= set(r.__dict__)
        assert "red" not in r.getMessage().lower()
    item = gathers.load(h.store, call, gid_of(g))
    assert item["state"] == gathers.DONE and "sealed_value" not in item
    assert_no_leak(h, caplog.records, all_fixture_factor_values() | h.sent_factors)
    assert payments(h) == []


def test_the_early_value_is_sealed_until_taken() -> None:
    h, calls = make()
    call = sid(11)
    g = at_card_question(h, call, absent=True)
    partial(h, call, g, "the purple one")
    item = gathers.load(h.store, call, gid_of(g))
    assert item["state"] == gathers.EARLY and "purple" not in str(item)
    assert h.sealer.unseal(item["sealed_value"], call_sid=call) == "purple"
    assert says(ET.fromstring(h.post(redirect_route(calls.updates[0][1]), call)["body"]
                              .encode())) == [readback(PURPLE)]


@pytest.mark.parametrize("text", ["red no", "the OwlBank card", "yes", "red or purple",
                                  "no red", "rewards", "the third one", ""])
def test_partials_that_arent_one_exact_card_wait_for_the_final_result(
        text: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    h, calls = make()
    call = sid(12)
    g = at_card_question(h, call, absent=True)
    partial(h, call, g, text)
    assert calls.updates == []
    assert gathers.load(h.store, call, gid_of(g)) is None
    assert [r.outcome for r in events(caplog.records, "factor_partial")] == ["incomplete"]


def test_the_second_one_finishes_early_over_the_listed_cards() -> None:
    h, calls = make()
    call = sid(13)
    g = at_card_question(h, call, absent=True)
    partial(h, call, g, "the second one")
    root = ET.fromstring(h.post(redirect_route(calls.updates[0][1]), call)["body"].encode())
    assert says(root) == [readback(PURPLE)]


def test_a_disambiguation_finishes_early_on_a_name_or_an_ordinal() -> None:
    h, calls = make(ResolverOutput(kind="ambiguous", candidates=["red", "purple"]))
    call = sid(14)
    at_card_question(h, call, absent=True)
    g = gather_of(h.xml("factor", call, SpeechResult="the OwlBank card", Confidence="0.9"))
    partial(h, call, g, "the second one")
    root = ET.fromstring(h.post(redirect_route(calls.updates[0][1]), call)["body"].encode())
    assert says(root) == [readback(PURPLE)] and pending_to(h, call) == "purple"


def test_not_available_with_several_cards_finishes_early() -> None:
    h, calls = make()
    call = sid(15)
    g = at_card_question(h, call, to="rewards")
    partial(h, call, g, "red")
    root = ET.fromstring(h.post(redirect_route(calls.updates[0][1]), call)["body"].encode())
    assert says(root) == [readback(RED)] and pending_to(h, call) == "red"


def test_a_partial_naming_a_card_the_caller_doesnt_hold_waits() -> None:
    """"rewards" is a known card Thomas doesn't hold: the final answer is a not-available
    question, never an early finish."""
    h, calls = make()
    call = sid(16)
    g = at_card_question(h, call, absent=True)
    partial(h, call, g, "the rewards card")
    assert calls.updates == []
    root = ET.fromstring(final(h, call, g, "the rewards card")["body"].encode())
    assert says(root)[0].startswith(f"I don't see an {REWARDS}")


# --------------------------------------------------------------------------- races


def test_final_before_the_redirect_answers_once() -> None:
    h, calls = make()
    call = sid(20)
    g = at_card_question(h, call, absent=True)
    partial(h, call, g, "red")
    fin = final(h, call, g, "purple")          # the final result wins the claim
    late = h.post(redirect_route(calls.updates[0][1]), call)
    assert late["body"] == fin["body"]
    assert says(ET.fromstring(fin["body"].encode())) == [readback(PURPLE)]
    assert len(card_answers(h, call)) == 1 and pending_to(h, call) == "purple"


def test_final_first_then_a_late_partial_does_nothing() -> None:
    h, calls = make()
    call = sid(21)
    g = at_card_question(h, call, absent=True)
    fin = final(h, call, g, "red")
    partial(h, call, g, "red")
    assert calls.updates == []
    assert final(h, call, g, "red")["body"] == fin["body"]
    assert len(card_answers(h, call)) == 1


def test_a_failed_update_leaves_the_normal_path() -> None:
    h, calls = make(fail=True)
    call = sid(22)
    g = at_card_question(h, call, absent=True)
    partial(h, call, g, "red")
    assert calls.updates == []
    assert gathers.state_of(gathers.load(h.store, call, gid_of(g))) == gathers.RELEASED
    root = ET.fromstring(final(h, call, g, "red")["body"].encode())
    assert says(root) == [readback(RED)] and len(card_answers(h, call)) == 1


def test_an_expired_early_value_is_asked_again() -> None:
    h, calls = make()
    call = sid(23)
    g = at_card_question(h, call, absent=True)
    partial(h, call, g, "red")
    item = gathers.load(h.store, call, gid_of(g))
    item["digits_expires_at"] = 0
    h.store.put("calls", item)
    root = ET.fromstring(h.post(redirect_route(calls.updates[0][1]), call)["body"].encode())
    assert says(root)[-1] == PROMPT and pending_to(h, call) is None


def test_a_forged_early_value_on_a_webhook_is_ignored() -> None:
    """Only the sealed claim carries an early value; a webhook parameter of that name is dropped."""
    from lambdas.frontdoor.slotflow import EARLY_VALUE

    h, _ = make()
    call = sid(24)
    at_card_question(h, call, absent=True)
    root = h.xml("factor", call, **{EARLY_VALUE: "red"})
    assert pending_to(h, call) is None and says(root)[-1] == PROMPT


# --------------------------------------------------------------------------- timings


def test_factor_answer_carries_timings_with_names_and_integers_only(
        caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    h, calls = make()
    call = sid(30)
    g = at_card_question(h, call, absent=True)
    partial(h, call, g, "red")
    h.post(redirect_route(calls.updates[0][1]), call)
    lines = events(caplog.records, "factor_answer")
    assert lines
    for r in lines:
        t = r.timings_ms
        assert isinstance(t, dict) and t
        for k, v in t.items():
            assert isinstance(k, str) and SPAN.fullmatch(k)
            assert isinstance(v, int) and not isinstance(v, bool) and v >= 0
    card = [r for r in lines if r.factor == "card"][-1]
    assert {"state_load", "policy", "catalog", "resolve", "render", "state_save"} <= set(
        card.timings_ms)
