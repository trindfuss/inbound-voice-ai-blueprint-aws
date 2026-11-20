"""Property tests for the slot engine over any
``FakeResolver`` output (the resolver gets the words unredacted)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank.segments import contains_value
from owlbank.slot_policy import SlotSettings
from owlbank.slots import (
    FakeResolver,
    ResolverOutput,
    SlotDecl,
    commands_from_catalog,
    make_context,
    normalize,
    resolve,
    spoken_digits,
    validate,
)
from owlbank.slots.normalize import NUMBER_WORDS, number_runs

CATALOG = json.loads(
    (Path(__file__).resolve().parents[2] / "data" / "prompt_catalog.json").read_text()
)
LANGS = ("en-US", "es-US")
TODAY = date(2026, 10, 12)


def texts(lang):
    return {e["key"]: e["text"] for e in CATALOG["entries"] if e["lang"] == lang}


LABELS = {lang: texts(lang) for lang in LANGS}
COMMANDS = {lang: commands_from_catalog(CATALOG["entries"], lang) for lang in LANGS}

OPTIONS = [
    {"value": "checking", "labelKey": "account.checking"},
    {"value": "savings", "labelKey": "account.savings"},
    {"value": "credit_card", "labelKey": "account.credit_card"},
]

def _s(name, decl, *, sensitive=False, is_factor=False):
    return SlotSettings(name=name, decl=decl, is_factor=is_factor, sensitive=sensitive)


SLOTS = [
    _s("zip", SlotDecl("digits", {"length": 5}, "label.zip", "factor.zip"), sensitive=True, is_factor=True),
    _s("ssn4", SlotDecl("digits", {"length": 4}, "label.ssn4", "factor.ssn4"), sensitive=True,
       is_factor=True),
    _s("otp", SlotDecl("digits", {"minLength": 4, "maxLength": 8}, "label.otp", "factor.otp")),
    _s("yes_no", SlotDecl("yes_no")),
    _s("yes_no", SlotDecl("yes_no"), sensitive=True),
    _s("account", SlotDecl("choice", {"options": OPTIONS[:2]})),
    _s("account", SlotDecl("choice", {"options": OPTIONS}), sensitive=True),
    _s("amount", SlotDecl("amount", {"min": "1", "max": "5000", "currency": "USD"})),
    _s("amount", SlotDecl("amount", {"min": 0, "max": 100}), sensitive=True),
    _s("date", SlotDecl("date", {"min": "2026-10-12", "max": "2027-01-31"})),
    _s("date", SlotDecl("date", {"min": "1900-01-01", "max": "2026-10-12"}), sensitive=True),
]

VOCAB = {lang: sorted(NUMBER_WORDS[lang[:2]]) for lang in LANGS}
PLAIN_WORDS = [
    "checking", "savings", "cheques", "ahorros", "agent", "agente", "yes", "no", "si",
    "my", "the", "account", "balance", "saldo", "dollars", "dolares", "october", "octubre",
    "quince", "fifteenth", "it's", "please", "actually", "credit", "card", "de", "tarjeta",
    "crédito", "never", "mind", "repeat", "what", "um", "uh", "zip", "code", "is",
    "$250", "250.50", "1,234", "15th", "o'clock", "someone", "often", "tonight",
]


def word_strategy(lang):
    return st.one_of(
        st.sampled_from(VOCAB[lang]),
        st.sampled_from(PLAIN_WORDS),
        st.from_regex(r"[0-9]{1,6}", fullmatch=True),
        st.text(alphabet="abcdefghijklmnopqrstuvwxyzáéíóúñ", min_size=1, max_size=8),
    )


@st.composite
def utterances(draw, lang):
    words = draw(st.lists(word_strategy(lang), min_size=1, max_size=10))
    seps = draw(st.lists(st.sampled_from([" ", ", ", " - ", ". ", "? ", "  "]),
                         min_size=len(words), max_size=len(words)))
    text = "".join(w + s for w, s in zip(words, seps, strict=True)).strip()
    if draw(st.booleans()):
        text = text.capitalize()
    return text


resolver_outputs = st.builds(
    ResolverOutput,
    kind=st.sampled_from(["exact", "inferred", "ambiguous", "none", "correction", "intent_switch",
                          "repeat", "escalate"]),
    value=st.one_of(
        st.none(), st.sampled_from(["checking", "savings", "credit_card", "yes", "no",
                                    "94105", "1234", "250.00", "2026-10-15", "1e3", "NaN"]),
        st.text(max_size=10),
    ),
    candidates=st.lists(st.sampled_from(["checking", "savings", "credit_card", "x"]), max_size=3).map(tuple),
    slot=st.one_of(st.none(), st.sampled_from(["zip", "ssn4", "account"])),
    reply=st.sampled_from([None, "yes", "no"]),
    outcome=st.sampled_from(["ok", "ok", "ok", "timeout", "error", "malformed", "refusal"]),
)


class RaisingResolver:
    def resolve(self, *args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("resolver called for valid DTMF")


def _resolve(slot, lang, *, speech=None, digits=None, resolver=None):
    ctx = make_context(question_text="question", labels=LABELS[lang], commands=COMMANDS[lang])
    return resolve(slot, digits=digits, speech=speech, confidence=0.8, lang=lang, mode="assisted",
                   ctx=ctx, resolver=resolver, today=TODAY)



PROPS = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@PROPS
@given(data=st.data(), slot=st.sampled_from(SLOTS), lang=st.sampled_from(LANGS),
       out=resolver_outputs)
def test_returned_values_pass_the_validator(data, slot, lang, out):
    speech = data.draw(st.one_of(st.none(), utterances(lang)))
    digits = data.draw(st.one_of(st.none(), st.from_regex(r"[0-9*#]{0,10}", fullmatch=True)))
    r = _resolve(slot, lang, speech=speech, digits=digits, resolver=FakeResolver([out]))
    if r.match in ("exact", "inferred") and r.value is not None:
        assert validate(slot.decl, r.value, today=TODAY) == r.value
    elif r.match != "correction":
        assert r.value is None


@st.composite
def valid_dtmf(draw):
    slot = draw(st.sampled_from(SLOTS))
    t = slot.decl.slot_type
    c = slot.decl.constraints
    if t == "digits":
        n = c.get("length") or draw(st.integers(c["minLength"], c["maxLength"]))
        return slot, draw(st.from_regex(rf"[0-9]{{{n}}}", fullmatch=True))
    if t == "yes_no":
        return slot, draw(st.sampled_from(["1", "2"]))
    if t == "choice":
        return slot, str(draw(st.integers(1, len(c["options"]))))
    if t == "amount":
        whole = draw(st.integers(int(c["min"]), int(c["max"]) - 1))
        cents = draw(st.one_of(st.just(""), st.from_regex(r"\*[0-9]{2}", fullmatch=True)))
        return slot, f"{whole}{cents}"
    lo, hi = date.fromisoformat(c["min"]), date.fromisoformat(c["max"])
    d = date.fromordinal(draw(st.integers(lo.toordinal(), hi.toordinal())))
    return slot, d.strftime("%m%d%Y")


@PROPS
@given(case=valid_dtmf(), lang=st.sampled_from(LANGS), speech=st.one_of(st.none(), st.text()))
def test_valid_dtmf_resolves_without_the_resolver(case, lang, speech):
    slot, digits = case
    r = _resolve(slot, lang, digits=digits, speech=speech, resolver=RaisingResolver())
    assert (r.match, r.source) == ("exact", "dtmf")
    assert validate(slot.decl, r.value, today=TODAY) == r.value


def _assert_ai_text_clean(text, slot, value):
    """The AI-text form: outside the ``[<slot name>]`` tokens, no number run of 2 or more
    tokens or digits, none that can't be read, and no exact-value match."""
    rest = text.replace(f"[{slot.name}]", " | ")
    for run in number_runs(rest, ("en", "es")):
        assert run.tokens == 1 and run.digits is not None and len(run.digits) == 1, (text, run)
    if value is not None:
        assert not contains_value(rest, value), text


@PROPS
@given(data=st.data(), slot=st.sampled_from([s for s in SLOTS if s.sensitive]),
       lang=st.sampled_from(LANGS), out=resolver_outputs)
def test_sensitive_text_has_no_digit_sequences(data, slot, lang, out):
    speech = data.draw(utterances(lang))
    fake = FakeResolver([out])
    r = _resolve(slot, lang, speech=speech, resolver=fake)
    for req in fake.requests:  # 4.1: the resolver gets the words as said
        assert req.utterance == speech
    if r.utterance_for_ai is not None:  # the AI gets the placeholder text
        _assert_ai_text_clean(r.utterance_for_ai, slot, r.value)



ZERO = {"en-US": ["zero", "oh", "o", "Zero", "OH"], "es-US": ["cero", "Cero"]}
DIGIT_WORDS = {
    "en-US": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
    "es-US": ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"],
}
REPEAT = {"en-US": {2: "double", 3: "triple"}, "es-US": {2: "doble", 3: "triple"}}


@st.composite
def spelled(draw, lang):
    d = draw(st.from_regex(r"[0-9]{1,12}", fullmatch=True))
    parts: list[str] = []
    i = 0
    while i < len(d):
        run = 1
        while i + run < len(d) and d[i + run] == d[i] and run < 3:
            run += 1
        word = draw(st.sampled_from(ZERO[lang])) if d[i] == "0" else DIGIT_WORDS[lang][int(d[i])]
        if run >= 2 and draw(st.booleans()):
            parts.append(f"{REPEAT[lang][run]} {word}")
            i += run
        else:
            parts.append(word)
            i += 1
    sep = draw(st.sampled_from([" ", ", ", "-", " . ", "  "]))
    text = sep.join(parts)
    if draw(st.booleans()):
        text = text.upper()
    return d, text


@settings(max_examples=500, deadline=None)
@given(data=st.data(), lang=st.sampled_from(LANGS))
def test_spelled_digits_round_trip(data, lang):
    d, text = data.draw(spelled(lang))
    assert spoken_digits(normalize(text, lang), lang) == d, text
