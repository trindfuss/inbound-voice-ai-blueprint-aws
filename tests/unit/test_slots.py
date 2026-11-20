"""Slot engine: handover worked examples, commands, Spanish, redaction.

``resolve`` returns a :class:`Resolution` and step 4 is the resolver (``FakeResolver``) in
place of the classifier. An answer is ``match="exact"`` (deterministic) or the resolver's grounded
value; what the classifier called unclear is ``none``. The two tests are rewritten:
the resolver sees the caller's words unredacted, and its value counts only when grounded."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from owlbank.slot_policy import SlotSettings
from owlbank.slots import (
    FakeResolver,
    Resolution,
    ResolverOutput,
    SlotDecl,
    commands_from_catalog,
    hints,
    make_context,
    match_command,
    normalize,
    resolve,
    spoken_digits,
    validate,
)
from owlbank.slots.normalize import parse_amount

CATALOG = json.loads(
    (Path(__file__).resolve().parents[2] / "data" / "prompt_catalog.json").read_text()
)
TODAY = date(2026, 10, 12)


def texts(lang: str) -> dict[str, str]:
    return {e["key"]: e["text"] for e in CATALOG["entries"] if e["lang"] == lang}


def commands(lang: str) -> dict[str, str]:
    return commands_from_catalog(CATALOG["entries"], lang)


ACCOUNT = SlotDecl(
    slot_type="choice",
    constraints={"options": [
        {"value": "checking", "labelKey": "account.checking"},
        {"value": "savings", "labelKey": "account.savings"},
    ]},
    prompt_key="balance.which_account",
)
ACCOUNT3 = SlotDecl(
    slot_type="choice",
    constraints={"options": [
        {"value": "checking", "labelKey": "account.checking"},
        {"value": "savings", "labelKey": "account.savings"},
        {"value": "credit_card", "labelKey": "account.credit_card"},
    ]},
)
ZIP = SlotSettings(name="zip", decl=SlotDecl.from_json({
    "slotType": "digits", "constraints": {"length": 5}, "labelKey": "label.zip",
    "promptKey": "factor.zip", "maxRetries": 1,
}), is_factor=True, sensitive=True)
SSN4 = SlotSettings(name="ssn4", decl=SlotDecl("digits", {"length": 4}, "label.ssn4", "factor.ssn4"),
                    is_factor=True, sensitive=True)
PIN = SlotDecl("digits", {"minLength": 4, "maxLength": 6}, None, None)
YES_NO = SlotDecl("yes_no")
AMOUNT = SlotDecl("amount", {"min": "1", "max": "5000", "currency": "USD"})
DATE = SlotDecl("date", {"min": "2026-10-12", "max": "2027-01-31"})

QUESTION = "Which account, checking or savings?"


def settings(slot):
    """``resolve`` takes slot settings: a bare declaration is a non-factor, non-sensitive slot."""
    if isinstance(slot, SlotDecl):
        return SlotSettings(name="slot", decl=slot, is_factor=False, sensitive=False)
    return slot


def run(slot, *, speech=None, digits=None, lang="en-US", resolver=None,
        confidence=0.9, question=QUESTION, mode="assisted"):
    slot = settings(slot)
    ctx = make_context(question_text=question, labels=texts(lang), commands=commands(lang))
    return resolve(slot, digits=digits, speech=speech, confidence=confidence, lang=lang,
                   mode=mode, ctx=ctx, resolver=resolver, today=TODAY)


def out(kind="none", value=None, **kw):
    return ResolverOutput(kind=kind, value=value, **kw)


# ---------------------------------------------------------- worked examples


def test_example_checking_resolves_deterministically():
    fake = FakeResolver()
    r = run(ACCOUNT, speech="Checking", resolver=fake)
    assert r == Resolution("exact", "speech", "checking", utterance_for_ai="Checking")
    assert fake.calls == []


def test_example_dtmf_two_is_savings():
    fake = FakeResolver()
    r = run(ACCOUNT, digits="2", resolver=fake)
    assert (r.match, r.value, r.source) == ("exact", "savings", "dtmf")
    assert fake.calls == []


def test_example_one_for_my_savings_goes_to_the_resolver():
    fake = FakeResolver([out("inferred", "savings")])
    r = run(ACCOUNT, speech="Yeah, I want the one for my savings", resolver=fake)
    assert (r.match, r.value, r.source, r.model) == ("inferred", "savings", "model", "ok")
    req = fake.requests[0]
    assert [(o.value, o.name) for o in req.known_options] == [("checking", "checking"),
                                                              ("savings", "savings")]
    assert req.question == QUESTION
    assert req.utterance == "Yeah, I want the one for my savings"


def test_example_the_other_one_is_none():
    fake = FakeResolver([out("none")])
    r = run(ACCOUNT, speech="Um, the other one", resolver=fake)
    assert (r.match, r.value) == ("none", None)
    assert r.utterance_for_ai == "Um, the other one"


def test_example_agent_is_a_command():
    fake = FakeResolver()
    r = run(ACCOUNT, speech="Agent", resolver=fake)
    assert (r.match, r.source) == ("escalate", "command")
    assert fake.calls == []


def test_example_speak_to_an_agent_goes_to_the_resolver():
    fake = FakeResolver([out("escalate")])
    r = run(ACCOUNT, speech="I'd like to speak to an agent", resolver=fake)
    assert (r.match, r.source) == ("escalate", "model")
    assert r.utterance_for_ai == "I'd like to speak to an agent"
    assert len(fake.calls) == 1


def test_example_credit_card_due_date_is_intent_switch():
    fake = FakeResolver([out("intent_switch")])
    r = run(ACCOUNT, speech="Actually, what's my credit card due date?", resolver=fake)
    assert (r.match, r.source) == ("intent_switch", "model")
    assert r.utterance_for_ai == "Actually, what's my credit card due date?"


@pytest.mark.parametrize("o", [
    out("exact", "credit_card"),   # not an option here
    out("exact", None),
    out("inferred", "Savings"),    # not canonical
])
def test_resolver_value_must_be_a_known_option(o):
    r = run(ACCOUNT, speech="hmm maybe the savings thing", resolver=FakeResolver([o]))
    assert (r.match, r.value) == ("none", None)


def test_resolver_exact_choice_not_named_is_demoted_to_inferred():
    """A model ``exact`` counts as exact only if the option's name or an alias was said."""
    r = run(ACCOUNT, speech="hmm maybe the money I keep aside", resolver=FakeResolver([out("exact", "savings")]))
    assert (r.match, r.value) == ("inferred", "savings")
    r = run(ACCOUNT, speech="savings I think, the savings one", resolver=FakeResolver([out("exact", "savings")]))
    assert (r.match, r.value) == ("exact", "savings")


def test_resolver_repeat_passes_through():
    r = run(ACCOUNT, speech="sorry, can you say it one more time", resolver=FakeResolver([out("repeat")]))
    assert (r.match, r.source) == ("repeat", "model")


def test_resolver_exception_is_none():
    class Boom:
        def resolve(self, *a, **k):
            raise RuntimeError("down")
    r = run(ACCOUNT, speech="the other one", resolver=Boom())
    assert (r.match, r.model) == ("none", "error")


@pytest.mark.parametrize("outcome", ["timeout", "error", "refusal", "malformed"])
def test_resolver_failure_is_none(outcome):
    r = run(ACCOUNT, speech="the other one", resolver=FakeResolver([ResolverOutput(kind="none", outcome=outcome)]))
    assert (r.match, r.model) == ("none", outcome)


def test_no_resolver_is_none():
    assert run(ACCOUNT, speech="the other one", resolver=None).match == "none"


def test_deterministic_mode_never_touches_the_resolver():
    fake = FakeResolver([out("exact", "savings")])
    r = run(ACCOUNT, speech="the other one", resolver=fake, mode="deterministic")
    assert (r.match, r.model) == ("none", "not_called") and fake.calls == []


def test_no_input_is_none():
    assert run(ACCOUNT, speech=None, digits=None).match == "none"
    assert run(ACCOUNT, speech="   ").match == "none"


def test_hints_are_catalog_labels():
    assert hints(ACCOUNT3, texts("es-US")) == ["cheques", "ahorros", "tarjeta de crédito"]


@pytest.mark.parametrize("speech", ["my savings account", "Savings, please.", "the savings one"])
def test_choice_carrier_words(speech):
    assert run(ACCOUNT, speech=speech).value == "savings"


# --------------------------------------------------------------- commands


def test_command_must_be_whole_utterance():
    assert match_command("my agent at the branch told me to call", "en-US",
                         commands("en-US")) is None
    fake = FakeResolver([out("intent_switch")])
    r = run(ACCOUNT, speech="My agent at the branch told me to call", resolver=fake)
    assert r.source == "model" and r.match == "intent_switch"
    assert len(fake.calls) == 1


@pytest.mark.parametrize("speech,kind", [
    ("Agent.", "escalate"), ("representative", "escalate"), ("A person!", "escalate"),
    ("Repeat", "repeat"), ("Say that again.", "repeat"), ("What?", "repeat"),
    ("Cancel", "cancel"), ("never mind", "cancel"), ("Stop.", "cancel"),
    ("um, agent", "escalate"),  # filler words don't count
])
def test_english_commands(speech, kind):
    fake = FakeResolver()
    r = run(ACCOUNT, speech=speech, resolver=fake)
    assert (r.match, r.source) == (kind, "command")
    assert fake.calls == []


@pytest.mark.parametrize("speech", ["cancel my card", "what is my balance", "stop payment"])
def test_command_words_inside_sentences_are_not_commands(speech):
    assert match_command(speech, "en-US", commands("en-US")) is None


def test_command_short_keys_accepted():
    assert match_command("agent", "en-US", {"agent": ["agent"]}) == "escalate"


# ---------------------------------------------------------------- Spanish


@pytest.mark.parametrize("speech,value", [
    ("Ahorros", "savings"), ("cheques", "checking"), ("La cuenta de ahorros", "savings"),
    ("Mi tarjeta de crédito", "credit_card"), ("tarjeta de credito", "credit_card"),
])
def test_spanish_choice(speech, value):
    r = run(ACCOUNT3, speech=speech, lang="es-US")
    assert (r.match, r.value, r.source) == ("exact", value, "speech")


@pytest.mark.parametrize("speech,kind", [
    ("Agente", "escalate"), ("una persona", "escalate"), ("Olvídelo", "cancel"),
    ("olvidelo", "cancel"), ("Repita, por favor", None), ("repita", "repeat"),
    ("¿Cómo?", "repeat"), ("Otra vez.", "repeat"), ("cancelar", "cancel"),
])
def test_spanish_commands(speech, kind):
    fake = FakeResolver([out("repeat")])
    r = run(ACCOUNT, speech=speech, lang="es-US", resolver=fake)
    if kind is None:  # not an exact phrase: the resolver decides
        assert r.source == "model"
    else:
        assert (r.match, r.source) == (kind, "command")


def test_spanish_digits_amount_date_yes():
    assert run(ZIP, speech="nueve cuatro uno cero cinco", lang="es-US").value == "94105"
    assert run(AMOUNT, speech="doscientos cincuenta", lang="es-US").value == "250.00"
    assert run(AMOUNT, speech="cien dólares con cincuenta centavos",
               lang="es-US").value == "100.50"
    assert run(DATE, speech="el quince de octubre", lang="es-US").value == "2026-10-15"
    assert run(YES_NO, speech="Sí", lang="es-US").value == "yes"
    assert run(YES_NO, speech="no", lang="es-US").value == "no"


# ------------------------------------------------------------- slot types


@pytest.mark.parametrize("speech", [
    "nine four one oh five", "Nine, four, one, zero, five.", "94105", "9 4 1 0 5",
    "It's 94105", "my zip code is nine four one oh five", "ninety four one oh five",
    "nine four one o five",
])
def test_zip_speech(speech):
    r = run(ZIP, speech=speech)
    assert (r.match, r.value, r.source) == ("exact", "94105", "speech")


def test_double_five():
    assert spoken_digits(normalize("double five one two", "en-US"), "en-US") == "5512"
    assert run(SSN4, speech="double five one two").value == "5512"


def test_digits_dtmf():
    fake = FakeResolver()
    assert run(ZIP, digits="94105", resolver=fake).value == "94105"
    assert run(ZIP, digits="94105#", resolver=fake).value == "94105"
    bad = run(ZIP, digits="9410", resolver=fake)
    assert (bad.match, bad.source) == ("none", "dtmf")
    assert fake.calls == []
    assert run(PIN, digits="12345").value == "12345"
    assert run(PIN, digits="123").match == "none"


def test_yes_no():
    assert run(YES_NO, speech="Yeah.").value == "yes"
    assert run(YES_NO, speech="nope").value == "no"
    assert run(YES_NO, digits="1").value == "yes"
    assert run(YES_NO, digits="2").value == "no"
    assert run(YES_NO, speech="no, I want my balance").match == "none"


@pytest.mark.parametrize("speech,value", [
    ("two hundred fifty dollars", "250.00"), ("$250", "250.00"), ("250", "250.00"),
    ("two hundred and fifty", "250.00"), ("$1,234.56", "1234.56"),
    ("fifty dollars and twenty five cents", "50.25"),
    ("one thousand two hundred thirty four dollars and fifty six cents", "1234.56"),
])
def test_amount_speech(speech, value):
    assert run(AMOUNT, speech=speech).value == value


def test_amount_bounds_and_dtmf():
    fake = FakeResolver([out("exact", "9999")])
    r = run(AMOUNT, speech="nine thousand nine hundred ninety nine dollars", resolver=fake)
    assert r.match == "none"  # over max; an amount is never taken from the model (fail closed)
    assert run(AMOUNT, digits="25*50").value == "25.50"
    assert validate(AMOUNT, "12.345") is None
    assert parse_amount(normalize("two five", "en-US"), "en-US") is None


@pytest.mark.parametrize("speech,value", [
    ("October fifteenth", "2026-10-15"), ("October 15th", "2026-10-15"),
    ("the fifteenth of October", "2026-10-15"), ("January fifth", "2027-01-05"),
    ("tomorrow", "2026-10-13"),
])
def test_date_speech(speech, value):
    assert run(DATE, speech=speech).value == value


def test_date_outside_range_and_dtmf():
    assert run(DATE, speech="March third").match == "none"  # 2026-03-03 / 2027-03-03 out
    assert run(DATE, digits="10152026").value == "2026-10-15"
    assert run(DATE, digits="02302026").match == "none"


# --------------------------------------------------------------- redaction

DIGITISH = re.compile(r"\d")


def test_sensitive_value_utterance_is_redacted():
    # The AI gets the slot's placeholder for the answer, not ``[digits]``.
    r = run(SSN4, speech="it's one two three four")
    assert (r.match, r.value) == ("exact", "1234")
    assert r.utterance_for_ai == "it's [ssn4]"


@pytest.mark.parametrize("speech,lang", [
    ("Actually my social is one two three four five, can I just check my balance?", "en-US"),
    ("It's 12345 no wait, I want an agent", "en-US"),
    ("I think it's nine four one oh, hold on, what's my balance", "en-US"),
    ("Es uno dos tres cuatro cinco, quiero mi saldo", "es-US"),
    ("Creo que es dieciséis veintidós, ¿cuál es mi saldo?", "es-US"),
])
def test_sensitive_off_script_reaches_the_resolver_unredacted(speech, lang):
    # Rewritten: the resolver gets the
    # caller's words as said, digits included; the AI text still gets the placeholder.
    fake = FakeResolver([out("intent_switch")])
    r = run(SSN4, speech=speech, lang=lang, resolver=fake)
    assert r.match == "intent_switch"
    assert fake.requests[0].utterance == speech
    ai = r.utterance_for_ai
    assert "[ssn4]" in ai and "[digits]" not in ai and ai != speech
    assert not DIGITISH.search(ai.replace("[ssn4]", ""))
    for word in ("one", "two", "nine", "uno", "dos", "dieciséis", "veintidós"):
        assert not re.search(rf"\b{word}\b", ai, re.IGNORECASE)


def test_sensitive_resolver_numeric_value_needs_grounding():
    # Rewritten: the resolver saw the
    # digits, so its value counts when the caller said it (ordered subsequence, exact length), and
    # never otherwise. For a factor and a sensitive slot alike. ("and" splits the run: without
    # it the parser reads "one, one two three four" itself, dropping the leading one.)
    speech = "it is, uh, the usual one, one two and three four I think"
    for slot in (SSN4, SlotSettings(name="ssn4", decl=SSN4.decl, is_factor=True, sensitive=False),
                 SlotSettings(name="pin", decl=SSN4.decl, is_factor=False, sensitive=True)):
        fake = FakeResolver([out("exact", "1234")])
        r = run(slot, speech=speech, resolver=fake)
        assert (r.match, r.value, r.source) == ("exact", "1234", "model")
        assert fake.requests[0].utterance == speech
        r = run(slot, speech=speech, resolver=FakeResolver([out("exact", "4321")]))
        assert (r.match, r.value, r.model) == ("none", None, "ungrounded")


def test_non_factor_non_sensitive_resolver_numeric_value_accepted():
    fake = FakeResolver([out("exact", "1234")])
    pin = SlotSettings(name="pin", decl=SSN4.decl, is_factor=False, sensitive=False)
    speech = "it is, uh, the usual one, one two and three four I think"
    r = run(pin, speech=speech, resolver=fake)
    assert (r.match, r.value, r.source) == ("exact", "1234", "model")
    assert fake.requests[0].utterance == speech and r.utterance_for_ai == speech


def test_sensitive_dtmf_has_no_utterance():
    r = run(ZIP, digits="94105")
    assert r.utterance_for_ai is None


def test_non_sensitive_utterance_kept_verbatim():
    r = run(ACCOUNT, speech="Call me at 555 1234 instead", resolver=FakeResolver([out("intent_switch")]))
    assert r.utterance_for_ai == "Call me at 555 1234 instead"


def test_resolution_invariants():
    with pytest.raises(ValueError):
        Resolution("value", "speech")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Resolution("exact", "classifier")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SlotDecl("address")


# ------------------------------------------------- AI-bound text

SSN4_PUBLIC = SlotSettings(name="ssn4", decl=SSN4.decl, is_factor=True, sensitive=False)
YES_NO_SENSITIVE = SlotSettings(name="yes_no", decl=YES_NO, is_factor=False, sensitive=True,
                                flow="confirm")
ACCOUNT_SENSITIVE = SlotSettings(name="account", decl=ACCOUNT, is_factor=False, sensitive=True)


@pytest.mark.parametrize("speech,want", [
    ("yes my last four are 1234", "yes my last four are [ssn4]"),
    ("one, uh, two three four", "[ssn4], uh, [ssn4]"),
    ("1 2, no, 1234", "[ssn4], no, [ssn4]"),
    ("the last four 1111", "the last [ssn4]"),
])
def test_sensitive_speech_value_placeholder(speech, want):
    r = run(SSN4, speech=speech)
    assert r.match == "exact" and r.source == "speech"
    assert r.utterance_for_ai == want


def test_sensitive_spanish_value_placeholder():
    r = run(SSN4, speech="uno dos tres cuatro", lang="es-US")
    assert (r.value, r.utterance_for_ai) == ("1234", "[ssn4]")


def test_sensitive_no_value_keeps_lone_words():
    fake = FakeResolver([out("intent_switch")])
    speech = "what's my balance? I can't find my four digits"
    r = run(SSN4, speech=speech, resolver=fake)
    assert r.match == "intent_switch" and r.utterance_for_ai == speech
    assert fake.requests[0].utterance == speech  # Unredacted


def test_sensitive_resolver_value_placeholder_from_grounding():
    """With a value, the AI text is the placeholder text for it (rule 1, ``grounded_runs``)."""
    speech = "yes my last four are 1234, and what's my balance?"
    r = run(SSN4, speech=speech, resolver=FakeResolver([out("intent_switch", "1234")]))
    assert (r.match, r.value, r.source) == ("inferred", "1234", "model")
    assert r.utterance_for_ai == "yes my last four are [ssn4], and what's my balance?"


def test_keypad_value_with_speech_uses_the_fallback():
    r = run(SSN4, digits="1234", speech="four")
    assert (r.source, r.utterance_for_ai) == ("dtmf", "[ssn4]")


def test_non_sensitive_factor_reaches_the_ai_and_the_resolver_unchanged():
    # ZIP isn't sensitive in the seed; the AI gets the caller's words, and so does the
    # resolver.
    fake = FakeResolver([out("intent_switch")])
    speech = "Actually my zip is 94105 but what are your branch hours?"
    r = run(SSN4_PUBLIC, speech=speech, resolver=fake)
    assert r.utterance_for_ai == speech
    assert fake.requests[0].utterance == speech
    assert run(SSN4_PUBLIC, speech="it's one two three four").utterance_for_ai == "it's one two three four"


def test_sensitive_yes_no_span_and_resolver_value():
    assert run(YES_NO_SENSITIVE, speech="Oh yes please").utterance_for_ai == "Oh [yes_no]"
    fake = FakeResolver([out("inferred", "yes")])
    r = run(YES_NO_SENSITIVE, speech="go ahead and do it", resolver=fake)
    assert (r.value, r.source, r.utterance_for_ai) == ("yes", "model", "[yes_no]")


def test_sensitive_choice_span():
    r = run(ACCOUNT_SENSITIVE, speech="my savings account")
    assert (r.value, r.utterance_for_ai) == ("savings", "my [account] account")
    r = run(ACCOUNT_SENSITIVE, speech="Checking")
    assert (r.value, r.utterance_for_ai) == ("checking", "[account]")


@pytest.mark.parametrize("speech,lang,runs", [
    ("yes my last four are 1234", "en-US", (1,)),
    ("one, uh, two three four", "en-US", (0, 1)),
    ("1 2, no, 1234", "en-US", (1,)),
    ("Es uno dos tres cuatro", "es-US", (0,)),
])
def test_digits_parse_reports_value_runs(speech, lang, runs):
    from owlbank.slots import parse_speech, parse_speech_located
    got = parse_speech_located(SSN4.decl, speech, lang)
    assert got.value == parse_speech(SSN4.decl, normalize(speech, lang), lang) == "1234"
    assert (got.value_runs, got.span) == (runs, None)


def test_yes_no_and_choice_parse_report_span():
    from owlbank.slots import parse_speech_located
    got = parse_speech_located(YES_NO, "Well, yeah.", "en-US")
    assert got.value == "yes" and "Well, yeah."[slice(*got.span)] == "yeah"
    got = parse_speech_located(ACCOUNT, "the savings one", "en-US", labels=texts("en-US"))
    assert got.value == "savings" and "the savings one"[slice(*got.span)] == "savings"


def test_amount_parse_is_not_located():
    from owlbank.slots import parse_speech_located
    got = parse_speech_located(AMOUNT, "two hundred fifty dollars", "en-US")
    assert (got.value, got.value_runs, got.span) == ("250", None, None)
