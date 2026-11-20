"""Generative wording in the front door.

Signed fake webhooks into the real front door (``test_frontdoor_support.build``) with
``FakeResolver``. The ``card`` slot (a non-sensitive choice slot) is configured ``generative``,
with ``generative_enabled`` and the ``generative`` ceiling unless a test says otherwise.

* Only ``ConfirmOption``, ``Disambiguate`` and ``NotAvailable`` speak the model's ``say``, rendered,
  and only when ``generative.check`` accepts it; the ``prompt`` event has ``generated: true``.
* Otherwise the catalog template, no second resolver call, and one ``generative_fallback`` with
  the reason (never the text). A resolver timeout logs ``budget``.
* A digit slot, a sensitive choice slot, the ``assisted`` ceiling and ``generative_enabled: false``
  keep the catalog, with the cap shown in ``slot_resolution``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from lambdas.frontdoor import state as sessions
from owlbank import callstate, sensitive
from owlbank.policy import Policy
from owlbank.segments import clear_sensitive
from owlbank.slots import FakeResolver, ResolverOutput
from owlbank.slots.resolver import failed
from tests.component.test_frontdoor_card import (
    PURPLE,
    RED,
    REWARDS,
    paid_with,
    pending_to,
    propose,
    readback,
    resolutions,
    speak,
    to_level3,
)
from tests.component.test_frontdoor_support import (
    DATA,
    OTP,
    TEST_FACTORS,
    THOMAS,
    Harness,
    build,
    says,
    sid,
)

SEED = json.loads((DATA / "policy.json").read_text())
EN = "en-US"
DISAMBIG = f"Which one did you mean: {RED} or {PURPLE}?"
CONFIRM_PURPLE = f"Did you mean the {PURPLE}?"
PROMPT = f"Which card would you like to pay: your {RED} or your {PURPLE}?"
PROMPT_SAM = f"Which card would you like to pay: your {REWARDS}?"


@pytest.fixture(autouse=True)
def _forget_calls() -> Any:
    yield
    for n in range(64):
        clear_sensitive(sid(200 + n))
        sensitive.forget(sid(200 + n))


def build_g(*outs: ResolverOutput, ceiling: str = "generative", enabled: bool = True,
            slots: dict[str, dict[str, Any]] | None = None, default: bool = False) -> Harness:
    """``card`` configured ``generative`` (``default``: through ``default_mode``, so slots that
    can't be generative inherit it and are capped), any ``slots`` changes, ``generative_enabled``
    and the running ceiling ``ceiling``."""
    h = build(resolver=FakeResolver(list(outs)), config_kw={"slot_mode_ceiling": ceiling})
    raw = json.loads(json.dumps(SEED))
    raw["slot_resolution"]["generative_enabled"] = enabled
    if default:
        raw["slot_resolution"]["default_mode"] = "generative"
    else:
        raw["slots"]["card"]["mode"] = "generative"
    for name, change in (slots or {}).items():
        raw["slots"][name].update(change)
    h.app.d.policy = Policy.from_dict(raw)
    return h


def fallbacks(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "event", None) == "generative_fallback"]


def prompts(h: Harness, call: str) -> list[dict[str, Any]]:
    return [e for e in h.journal(call) if e["type"] == "prompt"]


def card_call(h: Harness, n: int, who: str = "C-1001") -> str:
    """At the card question (``to`` missing)."""
    call = sid(200 + n)
    to_level3(h, call, who)
    if who == "C-1001":
        assert says(propose(h, call, absent=True)) == [PROMPT]
    else:
        # one card on file needs no card question; it comes after a "no" to the
        # not-available question about a card he doesn't hold.
        propose(h, call, "purple")
        assert says(speak(h, call, "no")) == [PROMPT_SAM]
    return call


def no_text_in_logs(caplog: pytest.LogCaptureFixture, text: str) -> None:
    for r in caplog.records:
        assert text not in r.getMessage()
        assert all(text not in str(v) for v in vars(r).values())


# --------------------------------------------------------------------------- spoken


def test_a_valid_generated_disambiguation_is_spoken_and_journaled_as_generated(caplog) -> None:
    caplog.set_level(logging.INFO)
    say = "Sure, the {option:red} or the {option:purple}?"
    h = build_g(ResolverOutput(kind="ambiguous", candidates=("red", "purple"), say=say,
                               say_lang=EN))
    call = card_call(h, 1)
    root = speak(h, call, "the OwlBank card")
    spoken = f"Sure, the {RED} or the {PURPLE}?"
    assert says(root) == [spoken]
    (call1,) = h.resolver.calls
    assert call1["generative"] is True
    last = prompts(h, call)[-1]
    assert last == {**last, "prompt_key": "slot.disambiguate", "rendered_text": spoken,
                    "generated": True}
    assert all("generated" not in p for p in prompts(h, call)[:-1])
    assert fallbacks(caplog) == []
    # The follow-up still works over the candidates: the caller names one.
    assert says(speak(h, call, "the red one")) == [readback(RED)]
    paid_with(h, call, "red", RED)


def test_a_valid_generated_confirmation_is_spoken_and_only_a_deterministic_yes_selects() -> None:
    h = build_g(ResolverOutput(kind="inferred", value="purple",
                               say="You'd like the {option:purple}, right?", say_lang=EN))
    call = card_call(h, 2)
    root = speak(h, call, "the violet one")
    assert says(root) == [f"You'd like the {PURPLE}, right?"] and pending_to(h, call) is None
    assert says(speak(h, call, "yes")) == [readback(PURPLE)]
    paid_with(h, call, "purple", PURPLE)


def test_a_valid_generated_not_available_question_may_name_the_callers_card() -> None:
    """Sam holds Rewards only; the model infers Purple: allowed = Purple and Rewards."""
    say = "I can't find the {option:purple} for you. Would the {option:rewards} work?"
    h = build_g(ResolverOutput(kind="inferred", value="purple", say=say, say_lang=EN))
    call = card_call(h, 3, "C-1002")
    root = speak(h, call, "the violet one")
    assert says(root) == [f"I can't find the {PURPLE} for you. Would the {REWARDS} work?"]
    assert prompts(h, call)[-1]["prompt_key"] == "slot.not_available_one"
    assert says(speak(h, call, "yes")) == [readback(REWARDS)]


def test_valid_spanish_wording_is_spoken() -> None:
    h = build_g(ResolverOutput(kind="inferred", value="purple",
                               say="¿Prefiere la {option:purple}?", say_lang="es-US"))
    call = sid(204)
    h.start(call, THOMAS, lang_digit="2")
    assert h.state(call).language == "es-US"
    h.factor(call, TEST_FACTORS["C-1001"]["zip"])
    h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    h.handoff(call, {"reason": "step-up", "tool": "make_payment", "journal": "jr-1"})
    h.factor(call, TEST_FACTORS["C-1001"]["ssn4"])
    h.factor(call, OTP)
    callstate.mutate(h.store, call, lambda s: {"pending_result": None, "pending_tool": None})
    propose(h, call, absent=True)
    root = speak(h, call, "la violeta")
    assert says(root) == ["¿Prefiere la Tarjeta OwlBank Morada?"]


# --------------------------------------------------------------------------- fallbacks


@pytest.mark.parametrize(("say", "say_lang", "reason"), [
    (None, None, "missing"),
    ("   ", EN, "missing"),
    ("¿La {option:red} o la {option:purple}?", "es-US", "lang"),
    ("The {option:red} or the {option:rewards}?", EN, "placeholder"),
    ("The {option:red} or {the purple}?", EN, "placeholder"),
    ("The red one or the purple one?", EN, "number"),
    ("Card 2 or card 3?", EN, "number"),
    ("The {option:red} or the {option:purple}? " + "Please tell me which " * 10, EN, "length"),
])
def test_each_fallback_reason_speaks_the_template(caplog, say, say_lang, reason) -> None:
    caplog.set_level(logging.INFO)
    h = build_g(ResolverOutput(kind="ambiguous", candidates=("red", "purple"), say=say,
                               say_lang=say_lang))
    call = card_call(h, 5)
    root = speak(h, call, "the OwlBank card")
    assert says(root) == [DISAMBIG]
    assert len(h.resolver.calls) == 1                      # no second call
    (fb,) = fallbacks(caplog)
    assert (getattr(fb, "reason", None), getattr(fb, "slot", None), getattr(fb, "asked", None)) \
        == (reason, "card", "disambiguate")
    assert all("generated" not in p for p in prompts(h, call))
    if say and say.strip():
        no_text_in_logs(caplog, say)


@pytest.mark.parametrize("say", [
    "The {option:red} or the {option:purple}?\x1b",           # ESC
    "The {option:red}\u200b or the {option:purple}?",          # zero-width space (Cf)
    "The {option:red}\nor the\t{option:purple}?",              # line feed and tab: one space
    "The {option:red} or the {option:purple}?\ufffe",          # a non-character
])
def test_generated_text_with_characters_twiml_cant_carry_is_healed_and_spoken(caplog, say) -> None:
    """The front door's own ``charset`` guard is gone; ``generative.heal`` removes what XML
    can't carry, and the healed text is spoken."""
    caplog.set_level(logging.INFO)
    h = build_g(ResolverOutput(kind="ambiguous", candidates=("red", "purple"), say=say,
                               say_lang=EN))
    call = card_call(h, 15)
    root = speak(h, call, "the OwlBank card")
    assert says(root) == [f"The {RED} or the {PURPLE}?"]
    assert fallbacks(caplog) == [] and prompts(h, call)[-1].get("generated") is True


def test_a_render_defect_speaks_the_template_and_logs_twiml_invalid_at_error(
        caplog, monkeypatch) -> None:
    """The response holding generated text is parsed once; when it doesn't parse (a
    rendering defect, forced here), the catalog template is spoken instead, with one
    ``generative_fallback`` (``twiml_invalid``) at ERROR and never the text."""
    from owlbank import twiml

    caplog.set_level(logging.INFO)
    say = "Sure, the {option:red} or the {option:purple}?"
    spoken = f"Sure, the {RED} or the {PURPLE}?"
    real = twiml.render
    renders: list[str] = []

    def broken(*verbs: Any) -> str:
        out = real(*verbs)
        renders.append(out)
        return out.replace("</Say>", "</Sya>", 1) if "Sure, the" in out else out

    monkeypatch.setattr(twiml, "render", broken)
    h = build_g(ResolverOutput(kind="ambiguous", candidates=("red", "purple"), say=say,
                               say_lang=EN))
    call = card_call(h, 16)
    n = len(renders)
    root = speak(h, call, "the OwlBank card")
    assert says(root) == [DISAMBIG]
    assert len(renders) - n == 2                     # the defective response, then the template's
    (fb,) = fallbacks(caplog)
    assert fb.levelno == logging.ERROR
    assert (getattr(fb, "reason", None), getattr(fb, "slot", None)) == ("twiml_invalid", "card")
    assert getattr(fb, "decision", None) == "template"
    last = prompts(h, call)[-1]
    assert last["prompt_key"] == "slot.disambiguate" and last["rendered_text"] == DISAMBIG
    assert all("generated" not in p for p in prompts(h, call))
    assert all(spoken not in t["text"] for t in (h.session(call).flow or {})["spoken"])
    no_text_in_logs(caplog, spoken)
    no_text_in_logs(caplog, say)
    # The follow-up still works over the candidates.
    assert says(speak(h, call, "the red one")) == [readback(RED)]


def test_a_response_without_generated_text_is_not_parsed_again(monkeypatch) -> None:
    """Only a response that holds generated text is parsed; a template is rendered once."""
    from lambdas.frontdoor import app as app_mod

    parsed: list[str] = []
    real = app_mod.twiml_parses

    def spy(text: str) -> bool:
        parsed.append(text)
        return real(text)

    monkeypatch.setattr(app_mod, "twiml_parses", spy)
    h = build_g(ResolverOutput(kind="ambiguous", candidates=("red", "purple"), say=None))
    call = card_call(h, 17)
    n = len(parsed)
    assert says(speak(h, call, "the OwlBank card")) == [DISAMBIG]
    assert len(parsed) == n
    h2 = build_g(ResolverOutput(kind="ambiguous", candidates=("red", "purple"),
                                say="Sure, the {option:red} or the {option:purple}?", say_lang=EN))
    call2 = card_call(h2, 18)
    n = len(parsed)
    speak(h2, call2, "the OwlBank card")
    assert len(parsed) == n + 1


def test_a_resolver_timeout_logs_budget_and_speaks_the_catalog(caplog) -> None:
    caplog.set_level(logging.INFO)
    h = build_g(failed("timeout"))
    call = card_call(h, 6)
    root = speak(h, call, "the OwlBank card")
    assert says(root) == [PROMPT]  # the question lists the cards; no "Please say" first
    assert len(h.resolver.calls) == 1
    (fb,) = fallbacks(caplog)
    assert getattr(fb, "reason", None) == "budget"


def test_wording_is_not_needed_for_an_accepted_card_and_nothing_is_logged(caplog) -> None:
    """An exact model match needs no question: ``say`` is never spoken, no fallback is logged."""
    caplog.set_level(logging.INFO)
    h = build_g(ResolverOutput(kind="exact", value="red", say="Zebra quartz", say_lang=EN))
    call = card_call(h, 7)
    root = speak(h, call, "the OwlBank Red Card please")
    assert says(root) == [readback(RED)]
    assert fallbacks(caplog) == []


def test_a_repeat_of_a_generated_question_is_the_catalog_template() -> None:
    """Design silent: the repeat is the catalog's (fail safe, 8.1 "every other sentence")."""
    h = build_g(ResolverOutput(kind="inferred", value="purple",
                               say="You'd like the {option:purple}, right?", say_lang=EN))
    call = card_call(h, 8)
    speak(h, call, "the violet one")
    root = speak(h, call, "say that again")
    assert says(root) == [CONFIRM_PURPLE]


# --------------------------------------------------------------------------- where it applies

SENTINEL = "Zebra quartz sentinel"


def test_a_say_on_a_digit_slot_is_ignored() -> None:
    h = build_g(ResolverOutput(kind="none", say=SENTINEL, say_lang=EN),
                default=True)
    call = sid(209)
    h.start(call, THOMAS)
    root = speak(h, call, "hmm let me think")
    assert all(SENTINEL not in t for t in says(root))
    (c,) = h.resolver.calls
    assert c["generative"] is False
    (ev,) = resolutions(h, call)
    assert ev["configured_mode"] == "generative" and ev["effective_mode"] == "assisted"
    assert ev["capped"] == "type"


def test_a_say_on_a_sensitive_choice_slot_is_ignored() -> None:
    """A sensitive card inheriting ``generative`` runs ``assisted`` (capped ``sensitive``); the
    say is never spoken."""
    h = build_g(ResolverOutput(kind="inferred", value="purple", say="The {option:purple}?",
                               say_lang=EN), slots={"card": {"sensitive": True}}, default=True)
    call = card_call(h, 10)
    root = speak(h, call, "the violet one")
    assert says(root) == [CONFIRM_PURPLE]
    assert h.resolver.calls[-1]["generative"] is False
    assert all("generated" not in p for p in prompts(h, call))
    assert resolutions(h, call)[-1]["capped"] == "sensitive"


def test_a_frozen_generative_mode_on_a_now_sensitive_slot_fails_closed() -> None:
    """The running policy is checked again: a session whose frozen card mode says ``generative``
    while the card is sensitive gets the catalog, and the resolver ``generative=False``."""
    h = build_g(ResolverOutput(kind="inferred", value="purple", say="The {option:purple}?",
                               say_lang=EN), slots={"card": {"sensitive": True}}, default=True)
    call = card_call(h, 11)
    sess = h.session(call)
    sess.flow["modes"]["card"]["effective"] = "generative"
    sessions.save(h.store, sess)
    root = speak(h, call, "the violet one")
    assert says(root) == [CONFIRM_PURPLE]
    assert h.resolver.calls[-1]["generative"] is False


def test_the_assisted_ceiling_caps_a_generative_slot_and_shows_it() -> None:
    h = build_g(ResolverOutput(kind="inferred", value="purple", say="The {option:purple}?",
                               say_lang=EN), ceiling="assisted")
    call = card_call(h, 12)
    root = speak(h, call, "the violet one")
    assert says(root) == [CONFIRM_PURPLE]
    assert h.resolver.calls[-1]["generative"] is False
    ev = resolutions(h, call)[-1]
    assert ev["slot"] == "card" and ev["configured_mode"] == "generative"
    assert ev["effective_mode"] == "assisted" and ev["capped"] == "ceiling"
    assert h.session(call).flow["modes"]["card"]["capped"] == "ceiling"


def test_generative_disabled_caps_to_assisted() -> None:
    h = build_g(ResolverOutput(kind="inferred", value="purple", say="The {option:purple}?",
                               say_lang=EN), enabled=False)
    call = card_call(h, 13)
    assert says(speak(h, call, "the violet one")) == [CONFIRM_PURPLE]
    assert h.resolver.calls[-1]["generative"] is False
    ev = resolutions(h, call)[-1]
    assert ev["effective_mode"] == "assisted" and ev["capped"] == "generative_disabled"


def test_the_read_back_and_other_sentences_never_speak_the_say() -> None:
    """The yes/no slot (never generative) and every re-ask stay on the catalog."""
    out = ResolverOutput(kind="none", say=SENTINEL, say_lang=EN)
    h = build_g(out, out, out, out)
    call = card_call(h, 14)
    root = speak(h, call, "hmm")                   # card: unclear -> clarify.choice
    assert says(root) == [PROMPT]  # the question lists the cards; no "Please say" first
    assert h.resolver.calls[-1]["generative"] is True
    root = speak(h, call, "the OwlBank Purple Card")
    assert says(root) == [readback(PURPLE)]
    root = speak(h, call, "hmm")                   # the read-back: yes_no
    assert all(SENTINEL not in t for t in says(root))
    assert h.resolver.calls[-1]["generative"] is False
