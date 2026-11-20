"""Front-door properties for corrections.

* For any sequence of corrections and wrong answers, each slot is asked at
  most ``maxRetries + 1`` times, no try count ever goes down, the flow ends once the counts
  allow no more questions, and a caller who corrects on every try gives up no later than one who
  answers wrong every try.
* Whatever the resolver proposes, no slot that hasn't been asked is ever
  written to the session (``collected``, ``held``, ``pending``, ``tries``) or the call state
  (``factors_verified``, ``sealed_sensitive``).
* For any sequence of answers, corrections and API calls in a flow, a correction
  is applied only to a slot collected after the flow's last API call: over the bookkeeping
  (``collect``, ``SlotFlow.api_call``, ``fresh``) and through the front door.
* A corrected sensitive value (and the one it replaced) appears in no
  captured log record, journal event, prompt, TwiML or stored item.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from lambdas.auth_service import service as auth_service
from lambdas.frontdoor import slotflow
from lambdas.frontdoor import state as sessions
from owlbank import segments, sensitive
from owlbank.policy import Policy
from owlbank.slots import FakeResolver, ResolverOutput
from tests.component.test_frontdoor_caller_id_trust import S, Z
from tests.component.test_frontdoor_support import (
    CLOCK,
    DATA,
    THOMAS,
    Harness,
    assert_no_leak,
    build,
    sid,
)

PROPS = settings(max_examples=60, deadline=None,
                 suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
SEED = json.loads((DATA / "policy.json").read_text())
WRONG_ZIP = "98101"
EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def words(digits: str) -> str:
    return " ".join(EN[int(c)] for c in digits)


def _harness(*, max_retries: dict[str, int] | None = None, ssn_first: bool = False) -> Harness:
    raw = json.loads(json.dumps(SEED))
    for name, n in (max_retries or {}).items():
        raw["factors"][name]["maxRetries"] = n
    if ssn_first:
        raw["levels"] = {"1": ["ani_match", "ssn4"], "2": ["ani_match", "ssn4", "zip"],
                         "3": ["ani_match", "ssn4", "zip", "otp"]}
    h = build(resolver=FakeResolver())
    policy = Policy.from_dict(raw)
    h.app.d.policy = policy
    deps = auth_service.AuthDeps(store=h.store, policy=policy, verify_client=h.verify,
                                 clock=lambda: int(CLOCK))
    h.lam.functions["owlbank-auth"] = lambda e: auth_service.handle_event(deps, e)  # type: ignore[index]
    return h


@contextmanager
def _call(n: int) -> Iterator[str]:
    call = sid(n)
    try:
        yield call
    finally:
        segments.clear_sensitive(call)
        sensitive.forget(call)


def _untrusted(h: Harness, call: str) -> ET.Element:
    h.xml("incoming", call, From=THOMAS, To="+12025550100", CallStatus="ringing")
    return h.xml("language", call, Digits="")


def _turn(h: Harness, call: str, params: dict[str, str], out: ResolverOutput | None) -> ET.Element:
    """One answer, with ``out`` as the resolver's output if it is called."""
    h.resolver._outputs = [out] if out is not None else []          # noqa: SLF001
    h.resolver.default = ResolverOutput(kind="none")
    resp = h.post("factor", call, **params)
    assert resp["statusCode"] == 200, resp
    return ET.fromstring(resp["body"].encode())


def _asked_slot(h: Harness, call: str) -> str | None:
    """The slot the open question is about, or ``None`` once the flow is over."""
    flow = h.session(call).flow
    if not flow or not flow.get("factor") or flow.get("kind") not in ("login", "stepup"):
        return None
    p = flow.get("pending")
    if isinstance(p, dict) and p.get("kind") == "correction_reask":
        return str(p["slot"])
    return str(flow["factor"])



#: A caller's turn at the SSN prompt (the ZIP held, untrusted caller ID), never the right SSN.
TURNS: dict[str, tuple[dict[str, str], ResolverOutput | None]] = {
    "wrong": ({"SpeechResult": "hmm I am not sure", "Confidence": "0.9"},
              ResolverOutput(kind="none")),
    "short_keys": ({"Digits": "12"}, None),
    "correct": ({"SpeechResult": f"actually my zip is {words(Z)}", "Confidence": "0.9"},
                ResolverOutput(kind="correction", slot="zip", value=Z)),
    "bad_correct": ({"SpeechResult": "actually my zip is different", "Confidence": "0.9"},
                    ResolverOutput(kind="correction", slot="zip", value="12345")),
    "zip_keys": ({"Digits": Z}, None),   # a ZIP on the keypad: the answer to a ZIP re-ask
}


def _questions(h: Harness, call: str, kinds: list[str]) -> tuple[dict[str, int], int, bool]:
    """Drive ``kinds`` at the SSN prompt; the questions per slot, the turns used, whether the flow
    ended. Checks on the way that no try count goes down."""
    asked = {"zip": 1}
    _untrusted(h, call)
    root = _turn(h, call, {"Digits": WRONG_ZIP}, None)
    assert root.find(".//Gather") is not None
    asked["ssn4"] = 1
    prev: dict[str, int] = {}
    used = 0
    for kind in kinds:
        params, out = TURNS[kind]
        root = _turn(h, call, params, out)
        used += 1
        slot = _asked_slot(h, call)
        if slot is None:
            return asked, used, True
        assert root.find(".//Gather") is not None
        asked[slot] = asked.get(slot, 0) + 1
        tries = dict((h.session(call).flow or {}).get("tries") or {})
        assert all(tries.get(k, 0) >= v for k, v in prev.items()), (prev, tries)   # 7.2
        prev = tries
    return asked, used, False


@PROPS
@given(zip_max=st.integers(1, 3), ssn_max=st.integers(1, 3),
       kinds=st.lists(st.sampled_from(sorted(TURNS)), min_size=1, max_size=14))
def test_questions_per_slot_are_bounded_and_counts_never_decrease(zip_max, ssn_max, kinds):
    limits = {"zip": zip_max, "ssn4": ssn_max}
    h = _harness(max_retries=limits)
    with _call(1) as call:
        asked, used, ended = _questions(h, call, kinds)
    for name, n in asked.items():
        assert n <= limits[name] + 1, (name, n, asked)
    # Every turn uses a try of the SSN last four, except the answer to a ZIP re-ask (whose try was
    # used by the correction before it), so the flow can't outlast 2 * (maxRetries + 1) turns.
    assert ended or used < 2 * (ssn_max + 1)
    assert not [p for p in h.lam.payloads if p.get("op") == "verify"]   # never the right SSN


def _give_up(h: Harness, call: str, kind: str) -> tuple[dict[str, int], int]:
    asked, used, ended = _questions(h, call, [kind] * 20)
    assert ended
    return asked, used


@PROPS
@given(zip_max=st.integers(1, 3), ssn_max=st.integers(1, 3),
       wrong=st.sampled_from(["wrong", "short_keys", "zip_keys"]))
def test_a_corrector_gives_up_no_later_than_a_wrong_answerer(zip_max, ssn_max, wrong):
    """7.2: a caller who corrects the held ZIP on every try reaches the give-up point no later
    (in turns) than one who answers the SSN prompt wrong on every try."""
    limits = {"zip": zip_max, "ssn4": ssn_max}
    with _call(2) as c1:
        _, corrector = _give_up(_harness(max_retries=limits), c1, "correct")
    with _call(3) as c2:
        _, wrong_answerer = _give_up(_harness(max_retries=limits), c2, wrong)
    assert corrector <= wrong_answerer


@PROPS
@given(zip_max=st.integers(1, 3), ssn_max=st.integers(1, 3))
def test_a_caller_whose_corrections_all_fail_uses_no_extra_tries(zip_max, ssn_max):
    """A failed correction is re-asked once for the corrected slot, a question that the
    correction's try of that slot paid for. So such a caller is asked the interrupted slot no more
    often than a wrong-answerer, and the extra turns are within the corrected slot's own
    ``maxRetries + 1``."""
    limits = {"zip": zip_max, "ssn4": ssn_max}
    with _call(2) as c1:
        asked, corrector = _give_up(_harness(max_retries=limits), c1, "bad_correct")
    with _call(3) as c2:
        asked_w, wrong_answerer = _give_up(_harness(max_retries=limits), c2, "wrong")
    assert asked["ssn4"] <= asked_w["ssn4"]
    assert corrector <= wrong_answerer + zip_max + 1



SLOTS = ["zip", "ssn4", "otp", "yes_no", "card", "nope"]
VALUES = st.from_regex(r"[0-9]{3,7}", fullmatch=True)
proposal = st.builds(
    ResolverOutput,
    kind=st.sampled_from(["correction", "correction", "exact", "inferred", "intent_switch", "none"]),
    value=st.one_of(st.none(), VALUES, st.just(Z), st.just(S), st.sampled_from(["yes", "red"])),
    slot=st.one_of(st.none(), st.sampled_from(SLOTS)),
)
answer = st.one_of(
    st.builds(lambda o: ({"SpeechResult": f"actually it is {o.value or 'nothing'}",
                          "Confidence": "0.9"}, o), proposal),
    st.builds(lambda d: ({"Digits": d}, None), st.sampled_from(["12", WRONG_ZIP, "9999", Z])),
)


def _written_slots(h: Harness, call: str) -> set[str]:
    sess = h.session(call)
    flow = sess.flow or {}
    out = {str(r.get("name")) for r in sess.collected}
    out |= set(flow.get("held") or {})
    out |= set(flow.get("tries") or {})
    p = flow.get("pending")
    if isinstance(p, dict):
        out.add(str(p.get("slot")))
    stt = h.state(call)
    out |= set(stt.factors_verified) - {"ani_match"}
    for blob in stt.sealed_sensitive:
        out.add(json.loads(h.sealer.unseal(blob, call_sid=call))["slot"])
    return out


def _asked(h: Harness, call: str) -> set[str]:
    return {e["factor"] for e in h.journal(call) if e["type"] == "factor_asked"}


@PROPS
@given(trusted=st.booleans(), answers=st.lists(answer, min_size=1, max_size=8))
def test_no_unasked_slot_is_ever_written(trusted, answers):
    h = _harness()
    with _call(4) as call:
        if trusted:
            h.start(call, THOMAS)
        else:
            _untrusted(h, call)
        for params, out in answers:
            if _asked_slot(h, call) is None:
                break
            _turn(h, call, params, out)
            assert _written_slots(h, call) <= _asked(h, call)



ssn_value = st.from_regex(r"[0-9]{4}", fullmatch=True).filter(
    lambda v: v not in (S, Z[:4], Z[1:]) and len(set(v)) > 1)


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@PROPS
@given(first=ssn_value, fixes=st.lists(st.tuples(ssn_value, st.booleans(), st.booleans()),
                                       min_size=1, max_size=4))
# The harness clock's escalated_at (2025-10-09T...) once read as the value 1009 (leakscan).
@example(first="0001", fixes=[("0001", False, False), ("0001", False, False), ("1009", False, False)])
def test_a_corrected_sensitive_value_reaches_no_log_or_journal(first, fixes):
    """The SSN last four is asked first and held; each correction of it (spoken as digits or
    words, grounded or not) and the value it replaced appear nowhere."""
    h = _harness(ssn_first=True)
    cap = _Records()
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(cap)
    root.setLevel(logging.DEBUG)
    values = {first}
    try:
        with _call(5) as call:
            _untrusted(h, call)
            _turn(h, call, {"Digits": first}, None)
            for value, grounded, as_words in fixes:
                if _asked_slot(h, call) is None:
                    break
                spoken = words(value) if as_words else value
                said = f"sorry my last four are {spoken}" if grounded else "sorry my last four changed"
                _turn(h, call, {"SpeechResult": said, "Confidence": "0.9"},
                      ResolverOutput(kind="correction", slot="ssn4", value=value))
                values.add(value)
                if _asked_slot(h, call) == "ssn4":       # the failed correction's re-ask
                    _turn(h, call, {"Digits": value}, None)
            assert_no_leak(h, cap.records, values)
    finally:
        root.removeHandler(cap)
        root.setLevel(old_level)



NAMES = ["zip", "ssn4", "otp", "card"]
op_ = st.one_of(
    st.tuples(st.just("collect"), st.sampled_from(NAMES), st.booleans()),
    st.tuples(st.just("api"), st.none(), st.none()),
    st.tuples(st.just("flow"), st.none(), st.none()),
)


@PROPS
@given(ops=st.lists(op_, max_size=30))
def test_only_a_slot_collected_after_the_flows_last_api_call_is_fresh(ops):
    """The bookkeeping: ``fresh(name)`` iff ``name`` was first collected in the current flow with
    no API call since; a slot collected again keeps its first ``at_api_call`` (fail safe)."""
    h = build()
    sess = sessions.Session(call_sid=sid(6), expires_at=0)
    sess.flow_seq = 1
    sess.flow = {"seq": 1, "api_calls": 0}
    first: dict[str, int] = {}       # this flow: the API calls made before each slot's first collection
    calls = 0
    for kind, name, verified in ops:
        if kind == "collect":
            assert name is not None
            slotflow.collect(sess, name, "digits", verified=bool(verified))
            first.setdefault(name, calls)
        elif kind == "api":
            assert h.app.slots.api_call(sess, lambda: "ok") == "ok"
            calls += 1
        else:                                       # a new flow: nothing in it collected yet
            sess.flow_seq += 1
            sess.flow = {"seq": sess.flow_seq, "api_calls": 0}
            first, calls = {}, 0
        assert (sess.flow or {})["api_calls"] == calls
        for n in NAMES:
            assert slotflow.fresh(sess, n) == (n in first and first[n] == calls), (n, ops)


STEPS = ["api", "correct", "wrong"]


@PROPS
@given(steps=st.lists(st.sampled_from(STEPS), min_size=1, max_size=6))
def test_a_correction_is_applied_only_with_no_api_call_since_the_slot(steps):
    """Through the front door: the untrusted login holds the ZIP at the SSN prompt; API calls in
    the flow (through the helper) and corrections of the ZIP in any order. A correction is applied
    (the held ZIP replaced, journaled) only while no API call has happened since the ZIP."""
    h = _harness()
    with _call(7) as call:
        _untrusted(h, call)
        _turn(h, call, {"Digits": WRONG_ZIP}, None)
        locked = False
        for step in steps:
            if _asked_slot(h, call) is None:
                break
            if step == "api":
                sess = h.session(call)
                h.app.slots.api_call(sess, lambda: None)
                sessions.save(h.store, sess)
                locked = True
                continue
            before = [e for e in h.journal(call)
                      if e["type"] == "slot_resolution" and e.get("correction")]
            held_before = (h.session(call).flow or {}).get("held")
            if step == "correct":
                _turn(h, call, {"SpeechResult": f"actually my zip is {words(Z)}",
                                "Confidence": "0.9"},
                      ResolverOutput(kind="correction", slot="zip", value=Z))
            else:
                _turn(h, call, {"SpeechResult": "hmm I am not sure", "Confidence": "0.9"},
                      ResolverOutput(kind="none"))
            after = [e for e in h.journal(call)
                     if e["type"] == "slot_resolution" and e.get("correction")]
            applied = len(after) - len(before)
            if locked or step != "correct":
                assert applied == 0, (steps, step)          # the property
            elif _asked_slot(h, call) is not None:          # not the try that gives up
                assert applied == 1, (steps, step)          # and a fresh slot is still corrected
            if locked:
                flow = h.session(call).flow or {}
                assert flow.get("held") in (held_before, None)   # never replaced
