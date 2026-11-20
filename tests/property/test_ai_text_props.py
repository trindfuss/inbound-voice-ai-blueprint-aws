"""Sensitive slots at the engine.

For any sensitive slot of any type, any utterance and any ``FakeResolver`` output (over arbitrary resolver outputs in place of the classifier's),
``utterance_for_ai`` has no exact-value match, and outside the ``[<slot name>]`` tokens no number
run of 2 or more digits (or tokens, or that can't be read) and no run that supplied a value digit.
For a non-sensitive slot it equals the speech. Also: the located parse reports the value
``parse_speech`` gives, at offsets that hold the words it read.
"""

from __future__ import annotations

import json
import unicodedata
from datetime import date
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank.policy import Policy
from owlbank.segments import contains_value
from owlbank.slot_policy import SlotSettings
from owlbank.slots import (
    FakeResolver,
    ResolverOutput,
    SlotDecl,
    commands_from_catalog,
    normalize,
    normalize_located,
    number_runs,
    parse_speech,
    parse_speech_located,
    make_context,
    resolve,
)

DATA = Path(__file__).resolve().parents[2] / "data"
SEED = Policy.from_dict(json.loads((DATA / "policy.json").read_text()))
CATALOG = json.loads((DATA / "prompt_catalog.json").read_text())
LANGS = ("en-US", "es-US")
TODAY = date(2026, 10, 19)
TEXTS = {lang: {e["key"]: e["text"] for e in CATALOG["entries"] if e["lang"] == lang} for lang in LANGS}
COMMANDS = {lang: commands_from_catalog(CATALOG["entries"], lang) for lang in LANGS}
PROPS = settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])

OPTIONS = [{"value": "checking", "labelKey": "account.checking"},
           {"value": "savings", "labelKey": "account.savings"},
           {"value": "credit_card", "labelKey": "account.credit_card"}]


def _sensitive(s: SlotSettings) -> SlotSettings:
    return SlotSettings(name=s.name, decl=s.decl, is_factor=s.is_factor, sensitive=True,
                        keypad_only=s.keypad_only, barge_in=s.barge_in, flow=s.flow)


#: Every seed slot made sensitive, and one sensitive slot of each other type.
SENSITIVE = [_sensitive(SEED.slot(n)) for n in SEED.slots()] + [
    SlotSettings(name="pin", decl=SlotDecl("digits", {"minLength": 4, "maxLength": 6}), is_factor=False,
                 sensitive=True),
    SlotSettings(name="account", decl=SlotDecl("choice", {"options": OPTIONS}), is_factor=False,
                 sensitive=True),
    SlotSettings(name="amount", decl=SlotDecl("amount", {"min": "1", "max": "5000"}), is_factor=False,
                 sensitive=True),
    SlotSettings(name="when", decl=SlotDecl("date", {"min": "2026-10-01", "max": "2027-01-31"}),
                 is_factor=False, sensitive=True),
]
DIGIT_WORDS = {"en": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
               "es": ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]}
FILLERS = ["my", "last", "four", "are", "it's", "is", "the", "code", "um", "uh", "pues", "no", "so",
           "balance", "saldo", "what's", "hold", "on", "agent", "repeat", "yes", "yeah", "sí", "claro",
           "checking", "savings", "credit card", "cuenta de ahorros", "correct", "nope", "o", "una"]
#: The fillers that are lone number words ("my last *four*", a Spanish "o" or "una").
NUMBER_FILLERS = {"four": "4", "o": "0", "una": "1"}
SEPS = [" ", ", ", ". ", "? ", " - ", ", uh, ", " no "]


def _length(slot: SlotSettings) -> int:
    c = slot.decl.constraints
    return int(c.get("length") or c.get("minLength") or 4)


@st.composite
def rendered_digits(draw: st.DrawFn, value: str) -> str:
    """``value`` as one run or split across runs, in digits or English or Spanish words."""
    fam = draw(st.sampled_from(["en", "es"]))
    style = draw(st.sampled_from(["digits", "spaced", "words", "mixed"]))

    def piece(chunk: str) -> str:
        if style == "digits":
            return chunk
        if style == "spaced":
            return " ".join(chunk)
        if style == "words":
            return " ".join(DIGIT_WORDS[fam][int(c)] for c in chunk)
        return " ".join(DIGIT_WORDS[fam][int(c)] if draw(st.booleans()) else c for c in chunk)

    cuts = sorted(draw(st.sets(st.integers(1, len(value) - 1), max_size=2))) if len(value) > 1 else []
    chunks = [value[a:b] for a, b in zip([0, *cuts], [*cuts, len(value)], strict=True)]
    return draw(st.sampled_from([", ", ", uh, ", " ", ". "])).join(piece(c) for c in chunks)


@st.composite
def utterance(draw: st.DrawFn, slot: SlotSettings, disjoint: bool) -> tuple[str, str | None, str]:
    """``(speech, generated value or None, digits of the lone number words)``."""
    t = slot.decl.slot_type
    value: str | None = None
    core = ""
    if t == "digits" and draw(st.booleans()):
        value = draw(st.text("0123456789", min_size=_length(slot), max_size=_length(slot)))
        core = draw(rendered_digits(value))
    elif t == "yes_no":
        core = draw(st.sampled_from(["yes", "yeah", "no", "nope", "sí", "claro", "correct", "well, no"]))
    elif t == "choice":
        core = draw(st.sampled_from(["checking", "savings", "my savings account", "credit card"]))
    elif t == "amount":
        core = draw(st.sampled_from(["two hundred fifty dollars", "$250", "250.50", "fifty"]))
    elif t == "date":
        core = draw(st.sampled_from(["October fifteenth", "the 15th of november", "el quince de octubre"]))
    allowed = [d for d in "0123456789" if not (disjoint and value and d in value)] or ["0"]
    lone: list[str] = []
    words: list[str] = [core] if core else []
    for _ in range(draw(st.integers(0, 4))):
        if draw(st.booleans()):
            d = draw(st.sampled_from(allowed))
            lone.append(d)
            words.insert(draw(st.integers(0, len(words))),
                         f"the {DIGIT_WORDS[draw(st.sampled_from(['en', 'es']))][int(d)]} thing"
                         if draw(st.booleans()) else f"my {d} thing")
        else:
            word = draw(st.sampled_from(FILLERS))
            if word in NUMBER_FILLERS:
                if NUMBER_FILLERS[word] not in allowed:
                    continue
                lone.append(NUMBER_FILLERS[word])
            words.insert(draw(st.integers(0, len(words))), word)
    seps = draw(st.lists(st.sampled_from(SEPS), min_size=len(words), max_size=len(words)))
    text = "".join(w + s for w, s in zip(words, seps, strict=True)).strip() or "um"
    return text, value, "".join(lone)


resolver_outputs = st.builds(
    ResolverOutput,
    kind=st.sampled_from(["exact", "inferred", "ambiguous", "none", "correction", "intent_switch",
                          "repeat", "escalate"]),
    value=st.one_of(st.none(), st.sampled_from(["94105", "1234", "123456", "1", "yes", "no",
                                                "savings", "250.00", "2026-10-15"]),
                    st.from_regex(r"[0-9]{4,6}", fullmatch=True), st.text(max_size=6)),
    slot=st.one_of(st.none(), st.sampled_from(["zip", "ssn4", "otp"])),
    reply=st.sampled_from([None, "yes", "no"]),
)


def _run(slot: SlotSettings, lang: str, speech: str | None, *, digits: str | None = None,
         out: ResolverOutput | None = None):
    labels = {k: v for k, v in TEXTS[lang].items() if k.startswith(("label.", "account."))}
    ctx = make_context(question_text="question", labels=labels, commands=COMMANDS[lang])
    return resolve(slot, digits=digits, speech=speech, confidence=0.9, lang=lang, mode="assisted",
                   ctx=ctx, resolver=None if out is None else FakeResolver([out]), today=TODAY)


def _outside_tokens(text: str, slot: SlotSettings) -> str:
    return text.replace(f"[{slot.name}]", " | ")  # "|" ends a run, so tokens don't join runs


@PROPS
@given(data=st.data(), slot=st.sampled_from(SENSITIVE), lang=st.sampled_from(LANGS),
       disjoint=st.booleans(), out=st.one_of(st.none(), resolver_outputs),
       keyed=st.booleans())
def test_ai_text_of_a_sensitive_slot(data, slot, lang, disjoint, out, keyed):
    speech, generated, lone = data.draw(utterance(slot, disjoint))
    digits = generated if keyed and generated else None
    r = _run(slot, lang, speech, digits=digits, out=out)
    text = r.utterance_for_ai
    assert text is not None
    rest = _outside_tokens(text, slot)
    for run in number_runs(rest, ("en", "es")):  # every surviving run: one token, one digit
        assert run.tokens == 1 and run.digits is not None and len(run.digits) == 1, (speech, text)
    if r.value is not None:  # with a value, it never survives (exact-value match)
        assert not contains_value(rest, r.value), (speech, text)
        if not set(lone) & set(r.value):
            # The lone words supplied no value digit, so a surviving run that holds one would be a
            # run that supplied the value.
            survivors = {run.digits for run in number_runs(rest, ("en", "es"))}
            assert not survivors & set(r.value), (speech, text)


@PROPS
@given(data=st.data(), name=st.sampled_from(SEED.slots()), lang=st.sampled_from(LANGS),
       out=st.one_of(st.none(), resolver_outputs))
def test_non_sensitive_slot_gets_the_words_unchanged(data, name, lang, out):
    slot = SlotSettings(name=name, decl=SEED.slot(name).decl, is_factor=SEED.slot(name).is_factor,
                        sensitive=False)
    speech, _, _ = data.draw(utterance(slot, False))
    r = _run(slot, lang, speech, out=out)
    assert r.utterance_for_ai == speech


@PROPS
@given(text=st.one_of(st.text(max_size=40),
                      st.lists(st.sampled_from([*FILLERS, "$250", "1,234", "15th", "Ángel", "ﬁx", "ΑΣ",
                                                "it’s", "dólares", "o'clock", "250.50", "_x_"]),
                               max_size=8).map(" ".join)),
       lang=st.sampled_from(LANGS))
def test_located_tokens_are_normalize_tokens(text, lang):
    located = normalize_located(text, lang)
    if located is None:  # can't locate: the caller falls back (fail closed)
        return
    assert " ".join(t.text for t in located) == normalize(text, lang)
    for t in located:
        assert 0 <= t.start < t.end <= len(text)


@PROPS
@given(data=st.data(), slot=st.sampled_from(SENSITIVE), lang=st.sampled_from(LANGS))
def test_located_parse_reports_parse_speech_value_and_real_offsets(data, slot, lang):
    speech, _, _ = data.draw(utterance(slot, False))
    labels = {k: v for k, v in TEXTS[lang].items() if k.startswith(("label.", "account."))}
    got = parse_speech_located(slot.decl, speech, lang, labels=labels, today=TODAY)
    assert got.value == parse_speech(slot.decl, normalize(speech, lang), lang, labels=labels, today=TODAY)
    nfc = unicodedata.normalize("NFC", speech)
    if got.value_runs:
        runs = number_runs(nfc, ("en", "es"))
        assert all(0 <= i < len(runs) for i in got.value_runs)
        assert slot.decl.slot_type == "digits"
        # A run mixing languages ("four 0 0 0 cero") can read as no digits over en+es together
        # even though the parse read the value from it; the placeholder cuts such a run anyway
        # (rule 1, fail closed). Every readable picked run must account for the value's digits.
        if all(runs[i].digits is not None for i in got.value_runs):
            digits = "".join(runs[i].digits or "" for i in got.value_runs)
            assert digits and set(got.value) <= set(digits)
    if got.span is not None:
        start, end = got.span
        assert slot.decl.slot_type in ("yes_no", "choice") and 0 <= start < end <= len(nfc)
        assert nfc[start:end].strip()
