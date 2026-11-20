"""A yes/no answer made only of affirmations (or only of denials) resolves without the
resolver; a mix, or any other word, goes to the resolver (the classifier)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.slots.normalize import normalize
from owlbank.slots.validators import NO_WORDS, YES_NO_NEUTRAL, YES_WORDS, yes_no_speech
from tests.unit.test_slots import YES_NO, FakeResolver, run

LANGS = {"en": "en-US", "es": "es-US"}


@pytest.mark.parametrize("speech,lang,value", [
    ("Yeah, that's correct.", "en-US", "yes"),
    ("Yes, that's right.", "en-US", "yes"),
    ("Yes correct", "en-US", "yes"),
    ("Sure, that's fine.", "en-US", "yes"),
    ("Um, yes.", "en-US", "yes"),
    ("Oh yes, absolutely", "en-US", "yes"),
    ("Mhm.", "en-US", "yes"),
    ("No, that's wrong.", "en-US", "no"),
    ("Nope, not right.", "en-US", "no"),
    ("No, that's not correct.", "en-US", "no"),
    ("Of course not.", "en-US", "no"),
    ("Sí, claro.", "es-US", "yes"),
    ("Sí, es correcto.", "es-US", "yes"),
    ("Eso es.", "es-US", "yes"),
    ("¡Exacto!", "es-US", "yes"),
    ("Bueno, sí.", "es-US", "yes"),
    ("No, no es correcto.", "es-US", "no"),
    ("Para nada.", "es-US", "no"),
    ("No está bien.", "es-US", "no"),
])
def test_affirmations_and_denials_skip_the_resolver(speech: str, lang: str, value: str) -> None:
    fake = FakeResolver()
    r = run(YES_NO, speech=speech, lang=lang, resolver=fake)
    assert (r.match, r.value, r.source) == ("exact", value, "speech")
    assert fake.calls == []


@pytest.mark.parametrize("speech,lang", [
    ("yes no", "en-US"), ("yeah, not right", "en-US"), ("no problem", "en-US"),
    ("yes I think so", "en-US"), ("uh huh", "en-US"), ("oh well", "en-US"),
    ("sí, no es correcto", "es-US"), ("claro, para nada", "es-US"), ("sí señor mío", "es-US"),
])
def test_mixed_or_unknown_answers_are_not_deterministic(speech: str, lang: str) -> None:
    fam = lang[:2]
    assert yes_no_speech(normalize(speech, lang), fam) is None


# ---------------------------------------------------------------- properties

def _reference_fewest(tokens: list[str], fam: str, polarity: str) -> int | None:
    """Fewest phrases splitting ``tokens`` using only ``polarity`` phrases and neutral words."""
    phrases = {tuple(p.split()) for p in (YES_WORDS if polarity == "yes" else NO_WORDS)[fam]}
    phrases |= {(w,) for w in YES_NO_NEUTRAL[fam]}
    best: list[int | None] = [0] + [None] * len(tokens)
    for i in range(1, len(tokens) + 1):
        for j in range(i):
            b = best[j]
            if b is not None and tuple(tokens[j:i]) in phrases and (best[i] is None or b + 1 < best[i]):
                best[i] = b + 1
    return best[len(tokens)]


def _phrases(fam: str, words: dict[str, frozenset[str]]) -> st.SearchStrategy[str]:
    return st.sampled_from(sorted(words[fam]))


FAMS = st.sampled_from(["en", "es"])


@settings(max_examples=300)
@given(st.data())
def test_any_run_of_affirmations_is_yes_and_of_denials_is_no(data: st.DataObject) -> None:
    fam = data.draw(FAMS)
    for words, want in ((YES_WORDS, "yes"), (NO_WORDS, "no")):
        seq = data.draw(st.lists(_phrases(fam, words), min_size=1, max_size=3))
        neutral = data.draw(st.lists(st.sampled_from(sorted(YES_NO_NEUTRAL[fam])), max_size=1))
        text = " ".join(neutral + seq)
        assert yes_no_speech(text, fam) == want, text
        # The same answer spoken: normalization keeps the phrases.
        assert yes_no_speech(normalize(text.capitalize() + ".", LANGS[fam]), fam) == want


@settings(max_examples=300)
@given(st.data())
def test_a_mix_is_only_resolved_through_a_shorter_single_polarity_reading(
        data: st.DataObject) -> None:
    """A sequence holding both a yes and a no phrase is not resolved, unless its words also read
    as fewer phrases of one polarity ("no" + "es correcto" is the one denial "no es correcto")."""
    fam = data.draw(FAMS)
    yes = data.draw(st.lists(_phrases(fam, YES_WORDS), min_size=1, max_size=2))
    no = data.draw(st.lists(_phrases(fam, NO_WORDS), min_size=1, max_size=2))
    seq = data.draw(st.permutations(yes + no))
    text = " ".join(seq)
    got = yes_no_speech(text, fam)
    if got is not None:
        fewest = _reference_fewest(text.split(), fam, got)
        assert fewest is not None and fewest < len(seq), text


@settings(max_examples=300)
@given(st.data())
def test_an_unknown_word_is_never_resolved(data: st.DataObject) -> None:
    fam = data.draw(FAMS)
    vocab = {t for p in YES_WORDS[fam] | NO_WORDS[fam] | YES_NO_NEUTRAL[fam] for t in p.split()}
    words = YES_WORDS if data.draw(st.booleans()) else NO_WORDS
    seq = data.draw(st.lists(_phrases(fam, words), min_size=0, max_size=3))
    unknown = data.draw(st.text("abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)
                        .filter(lambda w: w not in vocab))
    seq.insert(data.draw(st.integers(0, len(seq))), unknown)
    assert yes_no_speech(" ".join(seq), fam) is None
