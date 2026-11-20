"""Keyed-in phone login: the ``phone`` slot, the ``ani_trust.ask_phone_on_no_match`` switch and
the catalog entries, all as data."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from owlbank.policy import DEFAULT_ANI_TRUST, Policy, parse_ani_trust
from owlbank.slot_policy import gather_input
from owlbank.store import MemoryStore

DATA = Path(__file__).resolve().parents[2] / "data"
SEED: dict[str, Any] = json.loads((DATA / "policy.json").read_text(encoding="utf-8"))
CATALOG: dict[str, Any] = json.loads((DATA / "prompt_catalog.json").read_text(encoding="utf-8"))


def _doc(ani_trust: Any) -> dict[str, Any]:
    d = copy.deepcopy(SEED)
    d["ani_trust"] = ani_trust
    return d


# --------------------------------------------------------------------------- the switch


def test_seed_switch_is_on_and_default_is_on() -> None:
    assert SEED["ani_trust"]["ask_phone_on_no_match"] is True
    assert Policy.from_dict(SEED).ani_trust.ask_phone_on_no_match is True
    assert DEFAULT_ANI_TRUST.ask_phone_on_no_match is True


@pytest.mark.parametrize("value", [True, False])
def test_switch_valid_values_parse(value: bool) -> None:
    assert parse_ani_trust({"ask_phone_on_no_match": value}).ask_phone_on_no_match is value
    assert Policy.from_dict(_doc({"ask_phone_on_no_match": value})).ani_trust.ask_phone_on_no_match is value


def test_switch_missing_gives_the_default_without_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="owlbank.policy"):
        cfg = parse_ani_trust({"on_untrusted": "allow"})
    assert cfg.ask_phone_on_no_match is True and cfg.on_untrusted == "allow"
    assert caplog.records == []


@pytest.mark.parametrize("value", ["false", "yes", 0, 1, None, [], {}])
def test_switch_invalid_gives_the_default_with_one_warning(
        value: Any, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="owlbank.policy"):
        cfg = parse_ani_trust({"on_untrusted": "allow", "ask_phone_on_no_match": value})
    assert cfg.ask_phone_on_no_match is True
    assert cfg.on_untrusted == "allow"  # the rest of the section is kept
    (rec,) = caplog.records
    assert rec.levelno == logging.WARNING and "ask_phone_on_no_match" in rec.getMessage()


def test_switch_round_trips_through_the_policy_table() -> None:
    store = MemoryStore()
    for item in Policy.to_items(_doc({"ask_phone_on_no_match": False})):
        store.put("policy", item)
    assert Policy.from_store(store).ani_trust.ask_phone_on_no_match is False


# --------------------------------------------------------------------------- the slot


def test_phone_slot_is_a_fixed_length_sensitive_speech_allowed_digit_slot() -> None:
    s = Policy.from_dict(SEED).slot("phone")
    assert not s.is_factor and s.flow == "login"
    assert s.decl.slot_type == "digits" and dict(s.decl.constraints) == {"length": 10}
    assert s.decl.prompt_key == "login.phone_number" and s.decl.label_key == "label.phone"
    assert s.decl.max_retries == 2
    assert s.sensitive is True and s.confirm == "never"
    assert s.keypad_only is False and s.barge_in is True


@pytest.mark.parametrize("accept_spoken_factors", [True, False])
def test_phone_takes_speech_even_when_spoken_factors_are_off(accept_spoken_factors: bool) -> None:
    """The env switch ``OWLBANK_ACCEPT_SPOKEN_FACTORS`` applies to factor slots only."""
    s = Policy.from_dict(SEED).slot("phone")
    assert gather_input(s, accept_spoken_factors).keypad_only is False


def test_spoken_factors_off_still_makes_a_factor_keypad_only() -> None:
    assert gather_input(Policy.from_dict(SEED).slot("zip"), False).keypad_only is True


def test_seeded_meta_slots_holds_phone() -> None:
    items = {i["pk"]: i for i in Policy.to_items(SEED)}
    assert "phone" in items["meta"]["slots"]
    assert items["slot#phone"]["slotType"] == "digits" and items["slot#phone"]["sensitive"] is True


# --------------------------------------------------------------------------- the catalog


def _text(key: str, lang: str) -> str | None:
    for e in CATALOG["entries"]:
        if e["key"] == key and e["lang"] == lang:
            assert e["status"] == "published"
            return e["text"]
    return None


@pytest.mark.parametrize("lang", ["en-US", "es-US"])
@pytest.mark.parametrize("key", ["login.phone_number", "login.phone_number_retry", "label.phone"])
def test_catalog_has_the_phone_entries_in_both_languages(key: str, lang: str) -> None:
    text = _text(key, lang)
    assert isinstance(text, str) and text.strip()


# Today's wording, verbatim: the ``.keypad`` twins serve a slot that is keypad-only.
POUND_PROMPTS = {
    "factor.zip": ("Please enter your five-digit ZIP code. If you can't, press pound.",
                   "Por favor, ingrese su código postal de cinco dígitos. Si no puede, presione numeral."),
    "factor.ssn4": ("Please enter the last four digits of your Social Security number. If you can't, press pound.",
                    "Por favor, ingrese los últimos cuatro dígitos de su número de Seguro Social. "
                    "Si no puede, presione numeral."),
    "factor.otp": ("We just sent a six-digit code to your phone by text message. "
                   "Please enter it now, or press pound if you didn't get it.",
                   "Le acabamos de enviar un código de seis dígitos por mensaje de texto. "
                   "Por favor, ingréselo ahora, o presione numeral si no lo recibió."),
    "login.phone_number": ("What is the phone number associated with your account? "
                           "If you don't have an account with us, press pound.",
                           "¿Cuál es el número de teléfono asociado a su cuenta? "
                           "Si no tiene una cuenta con nosotros, presione numeral."),
    "login.phone_number_retry": ("Please enter the 10-digit phone number on your account. "
                                 "If you don't have an account with us, press pound.",
                                 "Por favor, ingrese el número de teléfono de 10 dígitos de su cuenta. "
                                 "Si no tiene una cuenta con nosotros, presione numeral."),
}

# The base keys: "say or enter".
SAY_OR_ENTER_PROMPTS = {
    "factor.zip": ("Please say or enter your five-digit ZIP code. If you can't, press pound.",
                   "Por favor, diga o ingrese su código postal de cinco dígitos. Si no puede, presione numeral."),
    "factor.ssn4": ("Please say or enter the last four digits of your Social Security number. "
                    "If you can't, press pound.",
                    "Por favor, diga o ingrese los últimos cuatro dígitos de su número de Seguro Social. "
                    "Si no puede, presione numeral."),
    "factor.otp": ("We just sent a six-digit code to your phone by text message. "
                   "Please say or enter it now, or press pound if you didn't get it.",
                   "Le acabamos de enviar un código de seis dígitos por mensaje de texto. "
                   "Por favor, dígalo o ingréselo ahora, o presione numeral si no lo recibió."),
    "login.phone_number": ("What is the phone number associated with your account? You can say or enter it. "
                           "If you don't have an account with us, press pound.",
                           "¿Cuál es el número de teléfono asociado a su cuenta? Puede decirlo o ingresarlo. "
                           "Si no tiene una cuenta con nosotros, presione numeral."),
    "login.phone_number_retry": ("Please say or enter the 10-digit phone number on your account. "
                                 "If you don't have an account with us, press pound.",
                                 "Por favor, diga o ingrese el número de teléfono de 10 dígitos de su cuenta. "
                                 "Si no tiene una cuenta con nosotros, presione numeral."),
}


@pytest.mark.parametrize("key", sorted(POUND_PROMPTS))
def test_catalog_texts_for_the_decline_prompts(key: str) -> None:
    """Both variants, both languages: the decline key is pound, never star or asterisco; the
    base key says "say or enter", its ``.keypad`` twin is the old wording."""
    for texts, suffix in ((SAY_OR_ENTER_PROMPTS, ""), (POUND_PROMPTS, ".keypad")):
        en, es = texts[key]
        assert _text(key + suffix, "en-US") == en and _text(key + suffix, "es-US") == es
        assert "star" not in en.lower() and "asterisco" not in es.lower()
