"""Front-door properties for the card in the payment confirmation.

* For any customer (two cards or one), any card the AI proposes (held, not
  held, unknown or none), any mode and any sequence of caller turns and resolver outputs
  (adversarial values included), ``make_payment`` never runs with a ``to`` outside the caller's
  cards, and the card sent is the one the last read-back named, answered with a deterministic yes.
* No card is selected (written to ``pending_tool.args.to``) that the caller
  didn't name in that turn's words, press on the keypad, or confirm with a deterministic yes to a
  question naming it.
* Every spoken text is a catalog template rendered with values: each ``<Say>`` is
  a journaled prompt whose key is in the catalog and whose text matches that template with its
  placeholders filled; nothing the resolver says (``say``) reaches TwiML.
* The card Gather and its follow-up Gathers follow ``card``'s settings:
  keypad-only gives ``input="dtmf"`` with no speech attributes; ``barge_in: false`` plays the
  prompts before an empty ``<Gather>`` (Path B); ``barge_in: true`` nests them.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from owlbank import callstate, segments, sensitive
from owlbank.callstate import PendingTool
from owlbank.catalog import PLACEHOLDER
from owlbank.policy import Policy
from owlbank.slots import FakeResolver, ResolverOutput
from owlbank.slots.resolver import RESOLVER_KINDS
from tests.component.test_frontdoor_support import (
    DATA,
    LEES,
    OTP,
    TEST_FACTORS,
    THOMAS,
    Harness,
    build,
    sid,
)

PROPS = settings(max_examples=60, deadline=None,
                 suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
SEED = json.loads((DATA / "policy.json").read_text())
CARDS = {"C-1001": ("red", "purple"), "C-1002": ("rewards",)}
NAMES = {"red": "OwlBank Red Card", "purple": "OwlBank Purple Card", "rewards": "OwlBank Rewards Card"}
SENTINEL = "Zebra quartz sentinel"
SPEECH_ATTRS = ("speechTimeout", "speechModel", "hints", "partialResultCallback", "language")

#: A caller's turn: the Gather parameters, the cards its words name (by construction), and whether
#: it is a deterministic yes (keypad 1 or a deterministic yes phrase).
TURNS: dict[str, tuple[dict[str, str], frozenset[str], bool]] = {
    "yes": ({"SpeechResult": "yes"}, frozenset(), True),
    "yeah": ({"SpeechResult": "yeah that's right"}, frozenset(), True),
    "no": ({"SpeechResult": "no"}, frozenset(), False),
    "key1": ({"Digits": "1"}, frozenset(), True),
    "key2": ({"Digits": "2"}, frozenset(), False),
    "key3": ({"Digits": "3"}, frozenset(), False),
    "red": ({"SpeechResult": "the red one"}, frozenset({"red"}), False),
    "purple": ({"SpeechResult": "the purple card"}, frozenset({"purple"}), False),
    "rewards": ({"SpeechResult": "OwlBank Rewards Card"}, frozenset({"rewards"}), False),
    "that_purple": ({"SpeechResult": "the one that's purple"}, frozenset({"purple"}), False),
    "no_red": ({"SpeechResult": "no, the red one"}, frozenset({"red"}), False),
    "meant_purple": ({"SpeechResult": "no, I meant the purple one"}, frozenset({"purple"}), False),
    "other": ({"SpeechResult": "the other one"}, frozenset(), False),
    "owlbank": ({"SpeechResult": "the OwlBank card"}, frozenset(), False),
    "balance": ({"SpeechResult": "what's my balance?"}, frozenset(), False),
    "go": ({"SpeechResult": "go with that"}, frozenset(), False),
    "hmm": ({"SpeechResult": "hmm"}, frozenset(), False),
    "again": ({"SpeechResult": "say that again"}, frozenset(), False),
    "silence": ({}, frozenset(), False),
    "second": ({"SpeechResult": "the second one"}, frozenset(), False),
    "last": ({"SpeechResult": "the last one"}, frozenset(), False),
}
#: Ordinal turns: the position they pick from the cards the question just listed.
ORDINALS = {"second": 1, "last": -1}
#: Questions that list the caller's cards.
LISTING = ("prompt.card", "slot.disambiguate", "slot.not_available_many", "clarify.choice")

values = st.sampled_from([None, "red", "purple", "rewards", "yes", "no", "credit_card", "25.00",
                          "1", SENTINEL])
outputs = st.builds(
    ResolverOutput,
    kind=st.sampled_from(RESOLVER_KINDS),
    value=values,
    candidates=st.lists(st.sampled_from(["red", "purple", "rewards", "gold"]), max_size=3
                        ).map(tuple),
    slot=st.sampled_from([None, "card", "amount", "from_account", "zip", "yes_no"]),
    reply=st.sampled_from([None, "yes", "no"]),
    say=st.sampled_from([None, SENTINEL]),
    say_lang=st.sampled_from([None, "en-US"]),
)
#: Deterministic yeses weigh more, so read-backs are often answered and payments run.
_POOL = sorted(TURNS) + ["yes", "key1", "yes", "key1"]
turns = st.lists(st.tuples(st.sampled_from(_POOL), outputs), min_size=1, max_size=8)
proposed = st.sampled_from([None, "red", "purple", "rewards", "credit_card"])


@contextmanager
def _call(n: int) -> Iterator[str]:
    call = sid(n)
    try:
        yield call
    finally:
        segments.clear_sensitive(call)
        sensitive.forget(call)


def _harness(mode: str, card: dict[str, Any] | None = None) -> Harness:
    kw = {"slot_mode_ceiling": "deterministic"} if mode == "deterministic" else {}
    h = build(resolver=FakeResolver(), config_kw=kw)
    if card:
        raw = json.loads(json.dumps(SEED))
        raw["slots"]["card"].update(card)
        h.app.d.policy = Policy.from_dict(raw)
    return h


def _level3(h: Harness, call: str, who: str) -> None:
    h.start(call, THOMAS if who == "C-1001" else LEES)
    h.factor(call, TEST_FACTORS[who]["zip"])
    h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    h.handoff(call, {"reason": "step-up", "tool": "make_payment"})
    h.factor(call, TEST_FACTORS[who]["ssn4"])
    h.factor(call, OTP)
    assert h.state(call).level == 3
    callstate.mutate(h.store, call, lambda s: {"pending_result": None, "pending_tool": None})


def _propose(h: Harness, call: str, to: str | None) -> ET.Element:
    args: dict[str, Any] = {"from_account": "savings", "amount": "1.00"}
    if to is not None:
        args["to"] = to
    callstate.mutate(h.store, call, lambda s: {
        "pending_tool": PendingTool(tool_use_id="tu_pay", name="make_payment", args=args)})
    resp = h.post("connect-action", call, CallStatus="in-progress", SessionStatus="ended",
                  SessionId="VX" + "c" * 32, SessionDuration="12",
                  HandoffData=json.dumps({"reason": "confirm", "tool": "make_payment"}))
    assert resp["statusCode"] == 200, resp
    return ET.fromstring(resp["body"].encode())


def _answer(h: Harness, call: str, params: dict[str, str], out: ResolverOutput) -> ET.Element:
    h.resolver._outputs = [out]                        # noqa: SLF001
    h.resolver.default = ResolverOutput(kind="none")
    resp = h.post("factor", call, **params)
    assert resp["statusCode"] == 200, resp
    return ET.fromstring(resp["body"].encode())


def _payments(h: Harness) -> list[dict[str, Any]]:
    return [p for p in h.lam.payloads if p.get("op") == "make_payment"]


def _to(h: Harness, call: str) -> Any:
    pt = h.state(call).pending_tool
    return None if pt is None else pt.args.get("to")


def _confirming(h: Harness, call: str) -> bool:
    flow = h.session(call).flow
    return bool(flow) and flow.get("kind") == "confirm"


def _prompts(h: Harness, call: str, since: int) -> list[dict[str, Any]]:
    return [e for e in h.journal(call)[since:] if e["type"] == "prompt"]


def _names(text: str, card: str) -> bool:
    return NAMES[card] in text


def _ordinal_pick(asked: list[dict[str, Any]], cards: tuple[str, ...], pos: int) -> str | None:
    """The card at ``pos`` in the order the last listing question read the caller's cards."""
    listing = [e for e in asked if e["prompt_key"] in LISTING]
    if not listing:
        return None
    text = listing[-1]["rendered_text"]
    read = sorted((text.index(NAMES[c]), c) for c in cards if NAMES[c] in text)
    order = [c for _, c in read]
    return order[pos] if -len(order) <= pos < len(order) else None




def _template_re(template: str) -> re.Pattern[str]:
    parts, pos = [], 0
    for m in PLACEHOLDER.finditer(template):
        parts.append(re.escape(template[pos:m.start()]))
        parts.append(r".*?")
        pos = m.end()
    parts.append(re.escape(template[pos:]))
    return re.compile("".join(parts), re.DOTALL)


def _check_spoken(h: Harness, call: str, root: ET.Element, since: int) -> None:
    """Every ``<Say>`` of this response is a journaled prompt of a catalog template."""
    cat = h.app.d.catalog
    lang = h.state(call).language
    events = _prompts(h, call, since)
    rendered = [e["rendered_text"] for e in events]
    for e in events:
        assert e["prompt_key"] in cat.keys, e["prompt_key"]
        template = cat.entry(e["prompt_key"], lang).text
        assert _template_re(template).fullmatch(e["rendered_text"]), (e["prompt_key"], e)
    body = ET.tostring(root, encoding="unicode")
    assert SENTINEL not in body
    for say in root.iter("Say"):
        assert (say.text or "") in rendered, say.text


@settings(PROPS, max_examples=300)
@given(who=st.sampled_from(sorted(CARDS)), mode=st.sampled_from(["assisted", "deterministic"]),
       to=proposed, script=turns)
def test_the_card_paid_is_the_callers_named_or_confirmed_card(
        who: str, mode: str, to: str | None, script: list[tuple[str, ResolverOutput]]) -> None:
    h = _harness(mode)
    cards = CARDS[who]
    with _call(40) as call:
        _level3(h, call, who)
        mark = len(h.journal(call))
        root = _propose(h, call, to)
        _check_spoken(h, call, root, mark)
        last = _prompts(h, call, mark)
        prev = _to(h, call)
        if to in cards or (to not in NAMES and len(cards) == 1):
            # Held, or no/unknown card with one card on file: read back with no question.
            assert [e["prompt_key"] for e in last] == ["confirm.make_payment"]
            assert _names(last[0]["rendered_text"], to if to in cards else cards[0])
        elif to in NAMES:
            # A known card the caller doesn't hold: the not-available question, never mapped.
            assert [e["prompt_key"] for e in last] == [
                "slot.not_available_one" if len(cards) == 1 else "slot.not_available_many"]
        elif _confirming(h, call):
            assert [e["prompt_key"] for e in last] == ["prompt.card"]     # asked, never mapped
            text = last[0]["rendered_text"]
            assert all(_names(text, c) for c in cards)                    # 18.13: lists them
            assert [text.index(NAMES[c]) for c in cards] == sorted(
                text.index(NAMES[c]) for c in cards)                      # in policy order
        for name, out in script:
            if not _confirming(h, call):
                break
            params, named, det_yes = TURNS[name]
            asked = last
            mark = len(h.journal(call))
            paid_before = len(_payments(h))
            root = _answer(h, call, params, out)
            _check_spoken(h, call, root, mark)
            last = _prompts(h, call, mark)
            pays = _payments(h)[paid_before:]
            if pays:
                (pay,) = pays
                sent = pay["args"].get("to")
                assert sent in cards
                readbacks = [e for e in asked if e["prompt_key"] == "confirm.make_payment"]
                assert readbacks and _names(readbacks[-1]["rendered_text"], sent)
                assert det_yes
                event(f"paid {sent}")
                break
            now = _to(h, call)
            if now != prev and now is not None:
                # A selection is named, keyed, or the yes to a question naming it.
                assert now in cards
                keyed = params.get("Digits") in {str(i + 1) for i, c in enumerate(cards)
                                                 if c == now}
                question = any(e["prompt_key"] in ("slot.confirm_option", "slot.not_available_one")
                               and _names(e["rendered_text"], now) for e in asked)
                ordinal = name in ORDINALS and now == _ordinal_pick(asked, cards, ORDINALS[name])
                assert now in named or keyed or (det_yes and question) or ordinal, (name, out, now)
                event("selected " + ("named" if now in named else "keyed" if keyed
                                     else "ordinal" if ordinal else "yes"))
            prev = now
        assert all(p["args"].get("to") in cards for p in _payments(h))




def _assert_shape(root: ET.Element, *, keypad_only: bool, barge_in: bool) -> None:
    tags = [c.tag for c in root]
    assert tags and tags[-1] == "Gather", tags
    g = root[-1]
    if keypad_only:
        assert g.get("input") == "dtmf"
        for attr in SPEECH_ATTRS:
            assert attr not in g.attrib, attr
    else:
        assert g.get("input") == "dtmf speech"
    if barge_in:
        assert tags == ["Gather"] and len(g) >= 1
    else:
        assert len(g) == 0 and len(tags) >= 2 and set(tags[:-1]) <= {"Say", "Play"}
    assert "bargeIn" not in g.attrib


@PROPS
@given(keypad_only=st.booleans(), barge_in=st.booleans(), asf=st.booleans(),
       who=st.sampled_from(sorted(CARDS)))
def test_the_card_gathers_follow_the_card_settings(
        keypad_only: bool, barge_in: bool, asf: bool, who: str) -> None:
    h = build(resolver=FakeResolver(), config_kw={"accept_spoken_factors": asf})
    raw = json.loads(json.dumps(SEED))
    raw["slots"]["card"].update(keypad_only=keypad_only, barge_in=barge_in)
    h.app.d.policy = Policy.from_dict(raw)
    shape = {"keypad_only": keypad_only, "barge_in": barge_in}
    with _call(41) as call:
        _level3(h, call, who)
        # The card question (C-1001); C-1002's one card needs none, so the not-available
        # question about a card he doesn't hold is the card Gather asked.
        _assert_shape(_propose(h, call, None if who == "C-1001" else "purple"), **shape)
        if keypad_only or who == "C-1002":    # an unclear answer: the card's re-ask
            unclear = {"Digits": "9"} if keypad_only else {"SpeechResult": "hmm"}
            _assert_shape(_answer(h, call, unclear, ResolverOutput(kind="none")), **shape)
        if keypad_only:
            return
        if who == "C-1001":       # a disambiguation, then an option confirmation (yes/no)
            amb = ResolverOutput(kind="ambiguous", candidates=("red", "purple"))
            _assert_shape(_answer(h, call, {"SpeechResult": "the OwlBank card"}, amb), **shape)
            inf = ResolverOutput(kind="inferred", value="purple")
            _assert_shape(_answer(h, call, {"SpeechResult": "the violet one"}, inf), **shape)



#: Partial texts over the card question: names, aliases, ordinals, yes/no words, fillers, cards the
#: caller doesn't hold, and phrases only the model could resolve.
_PARTIAL_WORDS = ["red", "purple", "rewards", "the", "one", "card", "OwlBank", "my", "no", "yes",
                  "second", "first", "last", "third", "or", "violet", "um", "pay", "please",
                  "Red", "Card", "that's", "right"]
_EXACT = ["red", "the red one", "purple", "the purple card", "OwlBank Red Card", "rewards",
          "the second one", "the last one", "the first one", "my purple card"]
_random = st.lists(st.sampled_from(_PARTIAL_WORDS), min_size=0, max_size=5).map(" ".join)
partial_texts = st.one_of(
    st.sampled_from(_EXACT),
    st.tuples(st.sampled_from(_EXACT), st.sampled_from(_PARTIAL_WORDS), st.booleans()).map(
        lambda t: f"{t[1]} {t[0]}" if t[2] else f"{t[0]} {t[1]}"),
    _random)


def _early_harness(*, keypad_only: bool = False) -> tuple[Harness, Any]:
    from lambdas.frontdoor.calls import FakeCallUpdater

    calls = FakeCallUpdater()
    h = build(resolver=FakeResolver(), config_kw={"slot_mode_ceiling": "deterministic"},
              calls=calls)
    if keypad_only:
        raw = json.loads(json.dumps(SEED))
        raw["slots"]["card"].update(keypad_only=True)
        h.app.d.policy = Policy.from_dict(raw)
    return h, calls


def _route(url: str) -> str:
    return url.rsplit("/voice/", 1)[1]


def _redirect(twiml: str) -> str:
    root = ET.fromstring(twiml.encode())
    return _route(root.find("Redirect").text or "")


@settings(PROPS, max_examples=120)
@given(text=partial_texts, who=st.sampled_from(["C-1001", "C-1001", "C-1002"]),
       to=st.sampled_from([None, None, "rewards", "purple"]))
def test_an_early_finish_is_the_card_the_final_parse_would_take(
        text: str, who: str, to: str | None) -> None:
    """A partial finishes the card Gather only when the final answer of the
    same words would select that same card deterministically. A partial with a yes/no word, two
    cards or a card the caller doesn't hold never finishes early."""
    h, calls = _early_harness()
    ref, _ = _early_harness()
    with _call(42) as call:
        _level3(h, call, who)
        _level3(ref, call, who)
        root = _propose(h, call, to)
        _propose(ref, call, to)
        g = root[-1]
        if g.tag != "Gather" or not g.get("partialResultCallback"):
            event("not eligible")
            return       # a read-back or a yes/no question (one card): never early
        resp = h.post(_route(g.get("partialResultCallback") or ""), call,
                      UnstableSpeechResult=text, SequenceNumber="1")
        assert resp["statusCode"] == 204
        toks = text.lower().split()
        if not calls.updates:
            event("waited")
            return
        assert "yes" not in toks and "no" not in toks     # Yes/no words never finish early
        early = h.post(_redirect(calls.updates[0][1]), call)
        took = _to(h, call)
        assert took in CARDS[who]
        # The same words as the final result, on the same question, in a twin call.
        ref.post(_route(g.get("action") or "").split("?")[0], call, SpeechResult=text,
                 Confidence="0.9")
        assert _to(ref, call) == took, (text, took)
        assert NAMES[took] in early["body"]               # the read-back names it
        event(f"early {took}")


_RACE = st.lists(st.sampled_from(["incomplete", "complete", "final", "early", "final_again",
                                  "early_again"]), min_size=1, max_size=8)


@settings(PROPS, max_examples=60)
@given(actions=_RACE, fail=st.booleans())
def test_one_answer_per_card_gather_in_any_order(actions: list[str], fail: bool) -> None:
    """One answer per card Gather: any interleaving of partials, the final webhook,
    the early redirect and their replays gives at most one REST update, exactly one card answer
    once an answer request arrived, and the same TwiML for every answer to the Gather."""
    from lambdas.frontdoor.calls import FakeCallUpdater

    calls = FakeCallUpdater(fail=fail)
    h = build(resolver=FakeResolver(), calls=calls)
    with _call(43) as call:
        _level3(h, call, "C-1001")
        g = _propose(h, call, None)[-1]
        assert g.get("partialResultCallback")
        texts = {"incomplete": "red no", "complete": "the purple one"}
        bodies: list[str] = []
        answered = False
        for a in actions:
            if a in texts:
                r = h.post(_route(g.get("partialResultCallback") or ""), call,
                           UnstableSpeechResult=texts[a])
                assert r["statusCode"] == 204
            elif a.startswith("final"):
                r = h.post(_route(g.get("action") or ""), call, SpeechResult="the purple one",
                           Confidence="0.9")
                bodies.append(r["body"])
                answered = True
            elif calls.updates:
                bodies.append(h.post(_redirect(calls.updates[0][1]), call)["body"])
                answered = True
            assert len(calls.updates) <= 1
        card = [e for e in h.journal(call) if e["type"] == "slot_resolution" and e["slot"] == "card"]
        assert len(card) == (1 if answered else 0)
        assert len(set(bodies)) <= 1
        if answered:
            assert _to(h, call) == "purple" and _payments(h) == []


@PROPS
@given(keypad_only=st.booleans(), who=st.sampled_from(sorted(CARDS)),
       to=st.sampled_from([None, "rewards", "purple", "red"]))
def test_no_early_finish_on_a_keypad_only_card(
        keypad_only: bool, who: str, to: str | None) -> None:
    """With early finish on: a keypad-only card Gather never has a Gather id or a
    partial callback; a spoken one has them exactly when it lists options (not a yes/no)."""
    h, _ = _early_harness(keypad_only=keypad_only)
    with _call(44) as call:
        _level3(h, call, who)
        root = _propose(h, call, to)
        g = root[-1]
        assert g.tag == "Gather"
        early = bool(g.get("partialResultCallback"))
        flow = h.session(call).flow
        if keypad_only:
            assert not early and "?" not in (g.get("action") or "")
            return
        listing = flow["factor"] == "card" and (
            flow["pending"] is None or flow["pending"]["kind"] in ("disambiguate",
                                                                  "not_available_many"))
        assert early == listing, (who, to, flow["factor"], flow["pending"])
