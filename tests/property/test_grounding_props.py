"""Property tests for grounding.

* For any utterance (digits, en/es number words, fillers, "oh",
  "double") and any proposed value (any JSON), ``ground_digits`` returns a value only if it has
  exactly ``constraints.length`` digits and is an ordered subsequence of
  ``digit_stream(normalize(utterance))``, and returns one whenever both hold.
  Checked twice: against ``digit_stream`` on arbitrary text, and against an independent oracle
  (the digits the generator rendered) on generated speech. The generated speech includes runs
  joined by the language's joiner ("tres y cuatro", "three and four") and Spanish tens with "y"
  ("treinta y uno" is 31): a run that doesn't read is split at its joiners.
* ``ground_choice`` says ``caller`` only for a caller option and
  ``known_other`` only for a known option that isn't one; ``ground_candidates`` returns at least 2
  distinct caller options or ``None``.
* **Located runs:** every run in ``grounded_runs`` holds a digit of the value (read as a whole,
  or split at its joiners: ``run_digits``), and together they hold the value as an ordered
  subsequence; a value the runs hold is always located. The generated speech joins pieces with
  "y"/"and" too.
"""

from __future__ import annotations

import itertools
import unicodedata

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from owlbank.slot_policy import SlotSettings
from owlbank.slots import number_runs
from owlbank.slots.grounding import (
    digit_stream,
    ground_candidates,
    ground_choice,
    ground_digits,
    grounded_runs,
    is_ordered_subsequence,
    named_in,
    run_digits,
)
from owlbank.slots.normalize import FILLERS, normalize
from owlbank.slots.types import OptionView, SlotDecl

LANGS = ("en-US", "es-US")

_UNIT_WORDS = {
    "en": {0: ["zero", "oh", "o"], 1: ["one"], 2: ["two"], 3: ["three"], 4: ["four"], 5: ["five"],
           6: ["six"], 7: ["seven"], 8: ["eight"], 9: ["nine"]},
    "es": {0: ["cero"], 1: ["uno", "un", "una"], 2: ["dos"], 3: ["tres"], 4: ["cuatro"],
           5: ["cinco"], 6: ["seis"], 7: ["siete"], 8: ["ocho"], 9: ["nueve"]},
}
_REPEAT = {"en": ("double", "triple"), "es": ("doble", "triple")}
_JOINER = {"en": "and", "es": "y"}
_ES_TENS = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta",
            8: "ochenta", 9: "noventa"}
#: Words that are never number tokens in either language: they split runs but add no digits.
_ORDINARY = ["no", "actually", "it", "is", "my", "wait", "sorry", "la", "es", "mi", "code", "yes",
             "hold", "on", "codigo", "número", "the"]
_SEPS = [" ", ", ", " , ", "... ", " - ", "  ", ",\n"]


def _fam(lang: str) -> str:
    return lang[:2]


def _json_values() -> st.SearchStrategy[object]:
    scalars = st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False),
                        st.text(max_size=12))
    return st.recursive(scalars, lambda inner: st.one_of(
        st.lists(inner, max_size=3), st.dictionaries(st.text(max_size=4), inner, max_size=3)),
        max_leaves=6)


def _slot(length: int) -> SlotSettings:
    return SlotSettings(name="f", decl=SlotDecl("digits", {"length": length}), is_factor=True,
                        sensitive=False)


def _oracle_subsequence(value: str, stream: str) -> bool:
    """Brute force: some choice of len(value) positions of ``stream`` spells ``value``."""
    return any("".join(stream[i] for i in idx) == value
               for idx in itertools.combinations(range(len(stream)), len(value)))


@st.composite
def spoken(draw: st.DrawFn, lang: str) -> tuple[str, str]:
    """``(speech, digits)``: a digit string said with digits, number words, "oh", repeaters,
    fillers and ordinary words in ``lang``. ``digits`` is what the speech holds, in order."""
    fam = _fam(lang)
    pieces: list[str] = []
    digits: list[str] = []
    for _ in range(draw(st.integers(0, 8))):
        kind = draw(st.sampled_from(["digit", "digits", "word", "repeat", "filler", "ordinary",
                                     "joined", "tens"]))
        if kind == "digit":
            d = draw(st.integers(0, 9))
            pieces.append(draw(st.sampled_from(_UNIT_WORDS[fam][d])))
            digits.append(str(d))
        elif kind == "digits":
            s = draw(st.text("0123456789", min_size=1, max_size=5))
            pieces.append(s)
            digits.append(s)
        elif kind == "word":
            d = draw(st.integers(0, 9))
            word = draw(st.sampled_from(_UNIT_WORDS[fam][d]))
            pieces.append(draw(st.sampled_from([word, word.upper(), word.capitalize()])))
            digits.append(str(d))
        elif kind == "repeat":
            d = draw(st.integers(0, 9))
            rep = draw(st.sampled_from(_REPEAT[fam]))
            # "oh"/"o" after a repeater: spoken_digits reads any unit word.
            pieces.append(f"{rep} {draw(st.sampled_from(_UNIT_WORDS[fam][d]))}")
            digits.append(str(d) * (2 if rep in ("double", "doble") else 3))
        elif kind == "joined":
            # Units or digit tokens joined by the language's joiner: "tres y cuatro".
            ds = draw(st.lists(st.integers(0, 9), min_size=2, max_size=4))
            words = [draw(st.sampled_from([*_UNIT_WORDS[fam][d], str(d)])) for d in ds]
            sep = draw(st.sampled_from([f" {_JOINER[fam]} ", f", {_JOINER[fam]} "]))
            pieces.append(sep.join(words))
            digits.append("".join(map(str, ds)))
        elif kind == "tens" and fam == "es":
            # Spanish tens, alone or with "y" and a unit: one number, never split.
            t = draw(st.sampled_from(sorted(_ES_TENS)))
            u = draw(st.integers(0, 9))
            if u:
                pieces.append(f"{_ES_TENS[t]} y {draw(st.sampled_from(_UNIT_WORDS['es'][u]))}")
                digits.append(str(t * 10 + u))
            else:
                pieces.append(_ES_TENS[t])
                digits.append(str(t * 10))
        elif kind == "filler":
            pieces.append(draw(st.sampled_from(sorted(FILLERS[fam]))))
        else:
            pieces.append(draw(st.sampled_from(_ORDINARY)))
    speech = ""
    for p in pieces:
        speech += (draw(st.sampled_from(_SEPS)) if speech else "") + p
    return speech, "".join(digits)


@st.composite
def proposals(draw: st.DrawFn, stream: str, length: int) -> object:
    """A model's value: a subsequence of the stream, any digit string, a padded or mangled one,
    or any JSON."""
    choice = draw(st.integers(0, 4))
    if choice == 0 and len(stream) >= length:
        idx = sorted(draw(st.lists(st.integers(0, len(stream) - 1), min_size=length,
                                   max_size=length, unique=True)))
        return "".join(stream[i] for i in idx)
    if choice == 1:
        return draw(st.text("0123456789", min_size=max(length - 1, 0), max_size=length + 1))
    if choice == 2:
        core = draw(st.text("0123456789", min_size=length, max_size=length))
        return draw(st.sampled_from([f" {core}", f"{core}\n", f"{core}a", int(core) if core else 0,
                                     f"{core[:1]} {core[1:]}"]))
    if choice == 3:
        return draw(st.text(max_size=length + 2))
    return draw(_json_values())


def _expected_digits(value: object, length: int, stream: str) -> str | None:
    """The digit-grounding property restated independently: a string whose stripped form is exactly ``length`` ASCII digits
    and an ordered subsequence of ``stream``."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    if len(s) != length or not (s.isascii() and s.isdigit()):
        return None
    return s if _oracle_subsequence(s, stream) else None




@settings(max_examples=300)
@given(data=st.data(), lang=st.sampled_from(LANGS), length=st.integers(1, 6))
def test_generated_speech_against_an_independent_oracle(data: st.DataObject, lang: str,
                                                              length: int) -> None:
    speech, digits = data.draw(spoken(lang))
    assert digit_stream(normalize(speech, lang), lang) == digits
    value = data.draw(proposals(digits, length))
    got = ground_digits(_slot(length), value, speech, lang)
    assert got == _expected_digits(value, length, digits)


@settings(max_examples=300)
@given(speech=st.text(max_size=40), value=_json_values() | st.text("0123456789 ", max_size=7),
       lang=st.sampled_from(LANGS), length=st.integers(1, 6))
def test_any_text_against_digit_stream(speech: str, value: object, lang: str,
                                            length: int) -> None:
    stream = digit_stream(normalize(speech, lang), lang)
    got = ground_digits(_slot(length), value, speech, lang)
    if got is not None:
        assert len(got) == length and got.isascii() and got.isdigit()
        assert is_ordered_subsequence(got, stream)
        assert isinstance(value, str) and value.strip() == got
    expected = None
    if isinstance(value, str):
        s = value.strip()
        if len(s) == length and s.isascii() and s.isdigit() and is_ordered_subsequence(s, stream):
            expected = s
    assert got == expected


@given(speech=st.text(max_size=40), lang=st.sampled_from(LANGS))
def test_stream_is_digits_only(speech: str, lang: str) -> None:
    stream = digit_stream(normalize(speech, lang), lang)
    assert stream == "" or (stream.isascii() and stream.isdigit())


@given(value=st.text("012", max_size=5), stream=st.text("012", max_size=8))
def test_is_ordered_subsequence_matches_brute_force(value: str, stream: str) -> None:
    assert is_ordered_subsequence(value, stream) is _oracle_subsequence(value, stream)



_OPTION_VALUES = st.sampled_from(["red", "purple", "rewards", "gold", "silver", "platinum"])


@st.composite
def option_sets(draw: st.DrawFn) -> tuple[list[str], list[str]]:
    known = draw(st.lists(_OPTION_VALUES, unique=True, max_size=5))
    if draw(st.booleans()):
        caller = [v for v in known if draw(st.booleans())]   # 5.1: a subset
    else:
        caller = draw(st.lists(_OPTION_VALUES, unique=True, max_size=5))   # anything at all
    return known, caller


@given(sets=option_sets(), value=_OPTION_VALUES | _json_values() | st.sampled_from(
    ["Red", " red", "red ", "RED", "purple\n", ""]))
def test_ground_choice_verdicts(sets: tuple[list[str], list[str]], value: object) -> None:
    known, caller = sets
    got = ground_choice(value, known=known, caller=caller)
    assert got in ("caller", "known_other", "none")
    if got == "caller":
        assert isinstance(value, str) and value in caller and value in known
    if got == "known_other":
        assert isinstance(value, str) and value in known and value not in caller
    # And the converse: no known option is dropped.
    if isinstance(value, str) and value in known:
        assert got == ("caller" if value in caller else "known_other")
    else:
        assert got == "none"


@given(caller=st.lists(_OPTION_VALUES, unique=True, max_size=5),
       values=st.lists(_OPTION_VALUES | st.sampled_from(["Red", "", " gold"]) | _json_values(),
                       max_size=5).map(lambda v: v if len(v) % 2 else tuple(v))
       | _json_values())
def test_ground_candidates_result(caller: list[str], values: object) -> None:
    got = ground_candidates(values, caller=caller)
    if got is not None:
        assert isinstance(got, tuple)
        assert len(got) >= 2 and len(set(got)) == len(got)
        assert all(v in caller for v in got)
    ok = (isinstance(values, (list, tuple)) and all(isinstance(v, str) and v in caller
                                                    for v in values)
          and len(set(values)) >= 2)
    assert (got is not None) is ok
    if ok:
        assert isinstance(values, (list, tuple))
        assert got == tuple(dict.fromkeys(values))


# ------------------------------------------------------------------- named_in

_WORD = st.text("abcdefghijklmnopqrstuvwxyzáéíóúñ", min_size=1, max_size=7)


@given(before=st.lists(_WORD, max_size=4), after=st.lists(_WORD, max_size=4),
       alias=st.lists(_WORD, min_size=1, max_size=3), lang=st.sampled_from(LANGS),
       name=st.text(max_size=20))
def test_named_in_finds_an_alias_said_inside_any_sentence(before: list[str], after: list[str],
                                                          alias: list[str], lang: str,
                                                          name: str) -> None:
    phrase = " ".join(alias)
    assume(normalize(phrase, lang))   # an alias of only filler words names nothing
    option = OptionView("x", name, (phrase,))
    speech = ", ".join([*before, phrase.upper(), *after])
    assert named_in(normalize(speech, lang), option, lang)


@given(speech=st.lists(_WORD, max_size=6), alias=st.lists(_WORD, min_size=1, max_size=3),
       lang=st.sampled_from(LANGS))
def test_named_in_is_a_contiguous_token_match(speech: list[str], alias: list[str],
                                              lang: str) -> None:
    norm = normalize(" ".join(speech), lang)
    phrase = normalize(" ".join(alias), lang)
    toks, want = norm.split(), phrase.split()
    expected = bool(want) and any(toks[i:i + len(want)] == want
                                  for i in range(len(toks) - len(want) + 1))
    assert named_in(norm, OptionView("x", " ".join(alias), ()), lang) is expected
    assert named_in(norm, OptionView("x", "", (" ".join(alias),)), lang) is expected


# --------------------------------------------------------------------- located runs


@st.composite
def located_speech(draw: st.DrawFn) -> tuple[str, str]:
    """``(speech, value)``: a value said in pieces among other numbers and words, en or es."""
    lang = draw(st.sampled_from(LANGS))
    fam = _fam(lang)
    value = draw(st.text("0123456789", min_size=1, max_size=6))
    pieces: list[str] = []
    for d in value:
        for _ in range(draw(st.integers(0, 2))):
            noise = draw(st.sampled_from(["digit", "word", "filler", "ordinary"]))
            if noise == "digit":
                pieces.append(draw(st.text("0123456789", min_size=1, max_size=3)))
            elif noise == "word":
                pieces.append(draw(st.sampled_from(_UNIT_WORDS[fam][draw(st.integers(0, 9))])))
            elif noise == "filler":
                pieces.append(draw(st.sampled_from(sorted(FILLERS[fam]))))
            else:
                pieces.append(draw(st.sampled_from(_ORDINARY)))
        pieces.append(draw(st.sampled_from([d, *_UNIT_WORDS[fam][int(d)]])))
    joiners = [f" {_JOINER[fam]} ", f", {_JOINER[fam]} "]
    speech = ""
    for p in pieces:
        speech += (draw(st.sampled_from(_SEPS + joiners)) if speech else "") + p
    if draw(st.booleans()):
        speech = unicodedata.normalize("NFD", speech)
    return speech, value


def _check_located(value: str, speech: str, got: tuple[int, ...] | None) -> None:
    runs = number_runs(unicodedata.normalize("NFC", speech), ("en", "es"))
    text = unicodedata.normalize("NFC", speech)
    if got is None:
        held = "".join(run_digits(text, r) or "" for r in runs)
        assert not (value.isascii() and value.isdigit() and value
                    and is_ordered_subsequence(value, held))
        return
    assert list(got) == sorted(set(got))
    assert all(0 <= i < len(runs) for i in got)
    for i in got:
        digits = run_digits(text, runs[i])
        assert digits is not None and any(c in value for c in digits)
        # A run that reads as a whole is never split.
        if runs[i].digits is not None:
            assert digits == runs[i].digits
    assert is_ordered_subsequence(value, "".join(run_digits(text, runs[i]) or "" for i in got))


@settings(max_examples=300)
@given(case=located_speech())
def test_located_runs_hold_the_value(case: tuple[str, str]) -> None:
    speech, value = case
    got = grounded_runs(value, speech)
    assert got is not None   # the value was said, digit by digit, in order
    _check_located(value, speech, got)


@given(speech=st.text(max_size=40), value=st.text("0123456789", max_size=5) | st.text(max_size=5))
def test_located_runs_any_text(speech: str, value: str) -> None:
    _check_located(value, speech, grounded_runs(value, speech))



@given(speech=st.text(max_size=40) | located_speech().map(lambda c: c[0]))
def test_run_digits_only_reads_the_runs_own_words(speech: str) -> None:
    """``run_digits`` keeps a whole reading, and otherwise is the pieces' readings in order:
    its digits come from the run's own text only (18.10: the split adds only digits said)."""
    text = unicodedata.normalize("NFC", speech)
    for run in number_runs(text, ("en", "es")):
        got = run_digits(text, run)
        if run.digits is not None:
            assert got == run.digits
        if got is not None:
            assert got.isascii() and got.isdigit()
            assert run_digits(text[run.start:run.end], number_runs(
                text[run.start:run.end], ("en", "es"))[0]) == got
