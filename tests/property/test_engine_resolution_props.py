"""Engine properties for model-assisted slots.

* A valid keypad answer never reaches the resolver, in any
  mode, whatever speech comes with it. (The differential half ran against the pre-feature engine
  while the deterministic stages changed, and was retired with the reference copy.)
* For any utterance and any resolver output
  (adversarial values included), a digit value the dialog accepts has exactly
  ``constraints.length`` digits and is an ordered subsequence of the utterance's digit stream.
* For any sensitive slot, any utterance and any resolver
  output, ``utterance_for_ai`` has no exact-value match, and outside the ``[<slot name>]`` tokens
  no number run of 2 or more digits (or tokens, or that can't be read) and no run that supplied a
  value digit; for a non-sensitive slot it equals the speech.
* Stage 2b never accepts an utterance that contains another collected slot's label.
* On a declinable slot, in any mode, ``#`` alone
  (``FinishedOnKey="#"``, no other key, no speech) and, in ``assisted`` and ``generative``, a
  resolver ``decline`` the engine reaches end the dialog with ``Command("decline")`` and use no
  try. (A spoken "cancel" is checked at the front door, ``test_frontdoor_login.py``.)
* On a slot that isn't declinable ``#`` alone resolves exactly as an empty result did
  (``none``), and a resolver ``decline`` gives the same action and state as ``none``;
  ``declinable`` changes nothing but ``#`` alone.
* In ``deterministic`` mode a slot resolves to ``decline`` only from ``#`` alone.
* For any slot and mode, a result with an empty or absent ``FinishedOnKey``, no digits
  and no speech never resolves to ``decline`` (silence is asked again).
* On a declinable slot, in any mode, ``Digits`` of only ``*`` (with or without ``#``)
  never declines and resolves exactly as another invalid entry (``**``) does; an amount keeps
  ``*`` as its decimal point.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from dataclasses import replace

from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from owlbank.policy import Policy
from owlbank.segments import contains_value
from owlbank.slot_policy import SlotSettings
from owlbank.slots import (
    CollectedSlot,
    SlotDecl,
    FakeResolver,
    OptionView,
    ResolverOutput,
    commands_from_catalog,
    decide,
    make_context,
    normalize,
    number_runs,
    resolve,
    slot_options,
)
from owlbank.slots.dialog import Accept, Command, Turn
from owlbank.slots.grounding import digit_stream, is_ordered_subsequence
from tests.property.test_ai_text_props import utterance

DATA = Path(__file__).resolve().parents[2] / "data"
CATALOG = json.loads((DATA / "prompt_catalog.json").read_text())
POLICY = Policy.from_dict(json.loads((DATA / "policy.json").read_text()))
LANGS = ("en-US", "es-US")
TODAY = date(2026, 10, 20)
TEXTS = {lang: {e["key"]: e["text"] for e in CATALOG["entries"] if e["lang"] == lang} for lang in LANGS}
COMMANDS = {lang: commands_from_catalog(CATALOG["entries"], lang) for lang in LANGS}
PROPS = settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
MODES = ("deterministic", "assisted", "generative")


def _options(lang: str) -> tuple[OptionView, ...]:
    card = POLICY.slot("card")
    return tuple(OptionView(o.value, TEXTS[lang][f"option.card.{o.value}"], tuple(o.aliases.get(lang, ())))
                 for o in card.options)


def _ctx(slot: SlotSettings, lang: str, collected: tuple[CollectedSlot, ...] = ()):
    known = _options(lang) if slot.name == "card" else ()
    return make_context(question_text="question", labels=TEXTS[lang], commands=COMMANDS[lang],
                        known_options=known, collected=collected)


def _sensitive(s: SlotSettings) -> SlotSettings:
    return SlotSettings(name=s.name, decl=s.decl, is_factor=s.is_factor, sensitive=True,
                        keypad_only=s.keypad_only, barge_in=s.barge_in, flow=s.flow,
                        confirm=s.confirm, options=s.options)


class Refuses:
    def resolve(self, *a, **k):  # pragma: no cover - must never run
        raise AssertionError("the resolver was called for a valid keypad answer")


#: Adversarial resolver outputs: any kind, values of every shape (digits of every length, option
#: values, yes/no, junk), any target slot, any reply and outcome.
resolver_outputs = st.builds(
    ResolverOutput,
    kind=st.sampled_from(["exact", "inferred", "ambiguous", "none", "correction", "intent_switch",
                          "repeat", "escalate", "decline"]),
    value=st.one_of(st.none(), st.from_regex(r"[0-9]{1,8}", fullmatch=True),
                    st.sampled_from(["red", "purple", "rewards", "yes", "no", "1111", "94105"]),
                    st.text(max_size=8)),
    candidates=st.lists(st.sampled_from(["red", "purple", "rewards", "zip"]), max_size=4).map(tuple),
    slot=st.one_of(st.none(), st.sampled_from(["zip", "ssn4", "otp", "card", "yes_no"])),
    reply=st.sampled_from([None, "yes", "no"]),
    outcome=st.sampled_from(["ok", "ok", "ok", "timeout", "error", "malformed"]),
)


# --------------------------------------------------------------------------- (keypad half)


@st.composite
def keypad_answers(draw):
    name = draw(st.sampled_from(POLICY.slots()))
    slot = POLICY.slot(name)
    t, c = slot.decl.slot_type, slot.decl.constraints
    if t == "digits":
        n = c.get("length") or draw(st.integers(c.get("minLength", 1), c.get("maxLength", 8)))
        key = draw(st.from_regex(rf"[0-9]{{{n}}}", fullmatch=True))
    elif t == "yes_no":
        key = draw(st.sampled_from(["1", "2"]))
    else:
        key = str(draw(st.integers(1, len(slot.options))))
    return slot, key + draw(st.sampled_from(["", "#"]))


@PROPS
@given(case=keypad_answers(), lang=st.sampled_from(LANGS), mode=st.sampled_from(MODES),
       speech=st.one_of(st.none(), st.text(max_size=30)))
def test_a_valid_keypad_answer_never_reaches_the_resolver(case, lang, mode, speech):
    slot, key = case
    r = resolve(slot, digits=key, speech=speech, confidence=0.9, lang=lang, mode=mode,
                ctx=_ctx(slot, lang), resolver=Refuses(), today=TODAY)
    assert (r.match, r.source, r.model) == ("exact", "dtmf", "not_called")




DIGIT_SLOTS = [POLICY.slot(n) for n in ("zip", "ssn4", "otp")]


@settings(PROPS, max_examples=1000)
@given(data=st.data(), slot=st.sampled_from(DIGIT_SLOTS), lang=st.sampled_from(LANGS),
       out=resolver_outputs, confirm=st.sampled_from(["when_inferred", "always", "never"]))
def test_an_accepted_digit_value_is_grounded(data, slot, lang, out, confirm):
    s = SlotSettings(name=slot.name, decl=slot.decl, is_factor=True, sensitive=slot.sensitive,
                     confirm=confirm)  # type: ignore[arg-type]
    speech, generated, _ = data.draw(utterance(s, data.draw(st.booleans())))
    stream = digit_stream(normalize(speech, lang), lang)
    n = int(s.decl.constraints["length"])
    # Half the time the resolver proposes a value built from the words: the caller's own value, or
    # any n digits picked from the digit stream in order (grounded), or out of order (not).
    pick = data.draw(st.sampled_from(["as_drawn", "generated", "subsequence", "shuffled"]))
    if pick == "generated" and generated:
        out = replace(out, value=generated, outcome="ok")
    elif pick in ("subsequence", "shuffled") and len(stream) >= n:
        idx = sorted(data.draw(st.sets(st.integers(0, len(stream) - 1), min_size=n, max_size=n)))
        chosen = [stream[i] for i in idx]
        if pick == "shuffled":
            chosen = data.draw(st.permutations(chosen))
        out = replace(out, value="".join(chosen), outcome="ok")
    ctx = _ctx(s, lang)
    r = resolve(s, digits=None, speech=speech, confidence=0.9, lang=lang, mode="assisted", ctx=ctx,
                resolver=FakeResolver([out]), today=TODAY)
    event(f"match={r.match} source={r.source}")
    turn = Turn(slot=s, mode="assisted", tries={}, pending=None, collected=(), earlier_strikes=0,
                owned=frozenset({s.name}))
    known, caller = slot_options(s, ctx)
    action, _ = decide(turn, r, known=known, caller=caller)
    for v in [r.value] if r.value is not None and r.match != "correction" else []:
        # every value the engine hands on is already grounded (the dialog doesn't re-check it)
        assert len(v) == int(s.decl.constraints["length"]), (speech, v)
        assert is_ordered_subsequence(v, digit_stream(normalize(speech, lang), lang)), (speech, v)
    if isinstance(action, Accept):
        event("accepted" + (" from the model" if r.source == "model" else ""))
        v = action.value
        assert len(v) == int(s.decl.constraints["length"]), (speech, v)
        assert is_ordered_subsequence(v, digit_stream(normalize(speech, lang), lang)), (speech, v)


# --------------------------------------------------------------------------- (phase 2)


SENSITIVE = [_sensitive(POLICY.slot(n)) for n in POLICY.slots()]


def _outside(text: str, slot: SlotSettings) -> str:
    return text.replace(f"[{slot.name}]", " | ")


@PROPS
@given(data=st.data(), slot=st.sampled_from(SENSITIVE), lang=st.sampled_from(LANGS),
       disjoint=st.booleans(), out=resolver_outputs, keyed=st.booleans())
def test_ai_text_of_a_sensitive_slot_for_any_resolver_output(data, slot, lang, disjoint, out, keyed):
    speech, generated, lone = data.draw(utterance(slot, disjoint))
    digits = generated if keyed and generated else None
    r = resolve(slot, digits=digits, speech=speech, confidence=0.9, lang=lang, mode="assisted",
                ctx=_ctx(slot, lang), resolver=FakeResolver([out]), today=TODAY)
    text = r.utterance_for_ai
    assert text is not None
    rest = _outside(text, slot)
    for run in number_runs(rest, ("en", "es")):   # every surviving run: one token, one digit
        assert run.tokens == 1 and run.digits is not None and len(run.digits) == 1, (speech, text)
    value = r.value if r.match != "correction" else None
    if value is not None:
        assert not contains_value(rest, value), (speech, text)
        if not set(lone) & set(value):
            survivors = {run.digits for run in number_runs(rest, ("en", "es"))}
            assert not survivors & set(value), (speech, text)


@PROPS
@given(data=st.data(), name=st.sampled_from(POLICY.slots()), lang=st.sampled_from(LANGS),
       out=resolver_outputs)
def test_non_sensitive_slot_gets_the_words_unchanged(data, name, lang, out):
    base = POLICY.slot(name)
    slot = SlotSettings(name=name, decl=base.decl, is_factor=base.is_factor, sensitive=False,
                        options=base.options)
    speech, _, _ = data.draw(utterance(slot, False))
    r = resolve(slot, digits=None, speech=speech, confidence=0.9, lang=lang, mode="assisted",
                ctx=_ctx(slot, lang), resolver=FakeResolver([out]), today=TODAY)
    if r.match == "correction" and r.target is not None:
        return  # a correction of another slot is that slot's text rule (covered in the unit tests)
    assert r.utterance_for_ai == speech




FILL = ["my", "the", "is", "it's", "um", "uh", "okay", "please", "actually", "and", "wait"]


@st.composite
def labelled_answers(draw):
    """An utterance for a digit slot that names another collected slot's label, with a value of the
    slot's length inside it (2b's case), in either language."""
    lang = draw(st.sampled_from(LANGS))
    slot = draw(st.sampled_from(DIGIT_SLOTS))
    others = [s for s in DIGIT_SLOTS if s.name != slot.name]
    other = draw(st.sampled_from(others))
    label = TEXTS[lang][other.decl.label_key or ""]
    n = int(slot.decl.constraints["length"])
    value = draw(st.from_regex(rf"[0-9]{{{n}}}", fullmatch=True))
    before = draw(st.lists(st.sampled_from(FILL), max_size=3))
    middle = draw(st.lists(st.sampled_from(FILL), max_size=2))
    after = draw(st.lists(st.sampled_from(FILL), max_size=2))
    form = draw(st.sampled_from(["digits", "spaced"]))
    said_value = value if form == "digits" else " ".join(value)
    words = [*before, label, *middle, said_value, *after]
    if draw(st.booleans()):
        words = [*before, said_value, *middle, label, *after][: len(words)]
    coll = tuple(CollectedSlot(name=s.name, slot_type="digits", label=TEXTS[lang][s.decl.label_key or ""],
                               scope=draw(st.sampled_from(["current", "earlier"])),
                               verified=draw(st.booleans()))
                 for s in (other, *draw(st.lists(st.sampled_from(others), max_size=1))))
    return slot, lang, " ".join(words), coll


@PROPS
@given(case=labelled_answers())
def test_2b_never_accepts_an_utterance_naming_another_collected_label(case):
    slot, lang, speech, coll = case
    r = resolve(slot, digits=None, speech=speech, confidence=0.9, lang=lang, mode="deterministic",
                ctx=_ctx(slot, lang, coll), resolver=None, today=TODAY)
    assert not (r.match == "exact" and r.source == "speech"), (speech, r)


# --------------------------------------------------------------------------- (decline)


AMOUNT = SlotSettings(name="amount", decl=SlotDecl("amount", {"min": 1, "max": 5000}), is_factor=False,
                      sensitive=False, flow="confirm")
DECLINE_SLOTS = [*(POLICY.slot(n) for n in POLICY.slots()), AMOUNT]
_keys = st.text("0123456789*#", max_size=8)
_words = st.text(st.characters(whitelist_categories=("Ll", "Lu", "Zs")), max_size=30)


def _declinable(s: SlotSettings, on: bool) -> SlotSettings:
    return replace(s, declinable=on)


def _decide(slot: SlotSettings, r, mode: str, ctx):
    turn = Turn(slot=slot, mode=mode, tries={}, pending=None, collected=(), earlier_strikes=0,  # type: ignore[arg-type]
                owned=frozenset({slot.name}))
    known, caller = slot_options(slot, ctx)
    return decide(turn, r, known=known, caller=caller)


#: ``Digits`` as Twilio may send it with ``#`` alone (empty) or that strip to nothing.
_no_keys = st.sampled_from([None, "", "#", " ", "#\n"])
#: Speech that is no speech (absent, empty or blank).
_no_speech = st.sampled_from([None, "", " ", "\t"])
_finish = st.sampled_from([None, "", "#"])


def _pound_alone(digits: str | None, speech: str | None, finished_on_key: str | None) -> bool:
    return (finished_on_key == "#" and not (digits or "").strip().rstrip("#").strip()
            and not (speech or "").strip())


@PROPS
@given(slot=st.sampled_from(DECLINE_SLOTS), lang=st.sampled_from(LANGS), mode=st.sampled_from(MODES),
       digits=_no_keys, speech=_no_speech)
def test_pound_alone_declines_a_declinable_slot_in_every_mode(slot, lang, mode, digits, speech):
    s = _declinable(slot, True)
    ctx = _ctx(s, lang)
    r = resolve(s, digits=digits, speech=speech, confidence=0.9, lang=lang, mode=mode, ctx=ctx,
                resolver=Refuses(), today=TODAY, finished_on_key="#")
    assert (r.match, r.source, r.model, r.value) == ("decline", "dtmf", "not_called", None)
    action, nxt = _decide(s, r, mode, ctx)
    assert action == Command("decline")
    assert dict(nxt.tries) == {}


@PROPS
@given(slot=st.sampled_from(DECLINE_SLOTS), lang=st.sampled_from(LANGS),
       mode=st.sampled_from(["assisted", "generative"]), speech=_words, out=resolver_outputs)
def test_a_resolver_decline_ends_a_declinable_slot(slot, lang, mode, speech, out):
    s = _declinable(slot, True)
    ctx = _ctx(s, lang)
    fake = FakeResolver([replace(out, kind="decline", value=None, outcome="ok")])
    r = resolve(s, digits=None, speech=speech, confidence=0.9, lang=lang, mode=mode, ctx=ctx,
                resolver=fake, today=TODAY)
    event(f"resolver called={bool(fake.calls)}")
    if not fake.calls:
        return  # the words resolved first (a value, a command, a short answer): no decline asked
    action, nxt = _decide(s, r, mode, ctx)
    assert action == Command("decline")
    assert dict(nxt.tries) == {}


@PROPS
@given(slot=st.sampled_from(DECLINE_SLOTS), lang=st.sampled_from(LANGS),
       mode=st.sampled_from(["assisted", "generative"]), speech=st.text(max_size=30),
       out=resolver_outputs)
def test_a_resolver_decline_is_none_on_a_slot_that_isnt_declinable(slot, lang, mode, speech, out):
    s = _declinable(slot, False)
    ctx = _ctx(s, lang)
    got = []
    for kind in ("decline", "none"):
        r = resolve(s, digits=None, speech=speech, confidence=0.9, lang=lang, mode=mode, ctx=ctx,
                    resolver=FakeResolver([replace(out, kind=kind)]), today=TODAY)
        got.append(_decide(s, r, mode, ctx))
    assert got[0] == got[1]
    assert not isinstance(got[0][0], Command) or got[0][0].kind != "decline"


@PROPS
@given(slot=st.sampled_from(DECLINE_SLOTS), lang=st.sampled_from(LANGS), mode=st.sampled_from(MODES),
       digits=_no_keys, speech=_no_speech)
def test_pound_alone_on_a_slot_that_isnt_declinable_is_an_empty_result(slot, lang, mode, digits,
                                                                           speech):
    s = _declinable(slot, False)
    ctx = _ctx(s, lang)
    got = [resolve(s, digits=digits, speech=speech, confidence=0.9, lang=lang, mode=mode, ctx=ctx,
                   resolver=Refuses(), today=TODAY, finished_on_key=k) for k in ("#", None)]
    assert got[0] == got[1]
    assert got[0].match == "none"
    assert _decide(s, got[0], mode, ctx) == _decide(s, got[1], mode, ctx)


@PROPS
@given(slot=st.sampled_from(DECLINE_SLOTS), lang=st.sampled_from(LANGS), mode=st.sampled_from(MODES),
       digits=st.one_of(_no_keys, _keys), speech=st.one_of(_no_speech, st.text(max_size=30)),
       finished_on_key=_finish, out=resolver_outputs)
def test_declinable_changes_nothing_but_pound_alone(slot, lang, mode, digits, speech,
                                                        finished_on_key, out):
    off = _declinable(slot, False)
    r_off = resolve(off, digits=digits, speech=speech, confidence=0.9, lang=lang, mode=mode,
                    ctx=_ctx(off, lang), resolver=FakeResolver([out]), today=TODAY,
                    finished_on_key=finished_on_key)
    assert r_off.match != "decline" or r_off.source == "model"
    assert _decide(off, r_off, mode, _ctx(off, lang))[0] != Command("decline")
    if _pound_alone(digits, speech, finished_on_key):
        return
    on = _declinable(slot, True)
    r_on = resolve(on, digits=digits, speech=speech, confidence=0.9, lang=lang, mode=mode,
                   ctx=_ctx(on, lang), resolver=FakeResolver([out]), today=TODAY,
                   finished_on_key=finished_on_key)
    assert r_on == r_off


@PROPS
@given(slot=st.sampled_from(DECLINE_SLOTS), lang=st.sampled_from(LANGS),
       digits=st.one_of(_no_keys, _keys), speech=st.one_of(_no_speech, st.text(max_size=30)),
       finished_on_key=_finish)
def test_deterministic_mode_declines_only_from_pound_alone(slot, lang, digits, speech,
                                                                 finished_on_key):
    s = _declinable(slot, True)
    ctx = _ctx(s, lang)
    r = resolve(s, digits=digits, speech=speech, confidence=0.9, lang=lang, mode="deterministic",
                ctx=ctx, resolver=FakeResolver(default=ResolverOutput(kind="decline")), today=TODAY,
                finished_on_key=finished_on_key)
    pound = _pound_alone(digits, speech, finished_on_key)
    event(f"pound alone={pound}")
    assert (r.match == "decline") == pound
    assert (_decide(s, r, "deterministic", ctx)[0] == Command("decline")) == pound


@PROPS
@given(slot=st.sampled_from(DECLINE_SLOTS), declinable=st.booleans(), lang=st.sampled_from(LANGS),
       mode=st.sampled_from(MODES), digits=st.sampled_from([None, ""]), speech=_no_speech,
       finished_on_key=st.sampled_from([None, ""]))
def test_silence_never_declines(slot, declinable, lang, mode, digits, speech, finished_on_key):
    s = _declinable(slot, declinable)
    ctx = _ctx(s, lang)
    r = resolve(s, digits=digits, speech=speech, confidence=0.9, lang=lang, mode=mode, ctx=ctx,
                resolver=FakeResolver(default=ResolverOutput(kind="decline")), today=TODAY,
                finished_on_key=finished_on_key)
    assert r.match != "decline"
    assert _decide(s, r, mode, ctx)[0] != Command("decline")


@PROPS
@given(slot=st.sampled_from(DECLINE_SLOTS), lang=st.sampled_from(LANGS), mode=st.sampled_from(MODES),
       star=st.sampled_from(["*", "*#", " * ", "*#\n"]), finished_on_key=_finish,
       speech=st.one_of(_no_speech, st.text(max_size=30)), out=resolver_outputs)
def test_star_never_declines_and_is_an_invalid_entry(slot, lang, mode, star, finished_on_key,
                                                          speech, out):
    s = _declinable(slot, True)
    ctx = _ctx(s, lang)

    def run(digits: str):
        return resolve(s, digits=digits, speech=speech, confidence=0.9, lang=lang, mode=mode,
                       ctx=ctx, resolver=FakeResolver([out]), today=TODAY,
                       finished_on_key=finished_on_key)

    r = run(star)
    assert r.match != "decline" or r.source == "model"   # only the words can decline (a resolver)
    assert r == run("**")                                 # any other invalid entry
    if not (speech or "").strip():
        assert (r.match, r.source) == ("none", "dtmf")    # asked again
        assert _decide(s, r, mode, ctx)[0] != Command("decline")


@given(whole=st.integers(1, 4999), cents=st.integers(0, 99), declinable=st.booleans(),
       finished_on_key=_finish)
def test_star_stays_the_decimal_point_of_an_amount(whole, cents, declinable, finished_on_key):
    s = _declinable(AMOUNT, declinable)
    r = resolve(s, digits=f"{whole}*{cents:02d}#", speech=None, confidence=None, lang="en-US",
                mode="assisted", ctx=_ctx(s, "en-US"), resolver=Refuses(), today=TODAY,
                finished_on_key=finished_on_key)
    assert (r.match, r.source, r.value) == ("exact", "dtmf", f"{whole}.{cents:02d}")
