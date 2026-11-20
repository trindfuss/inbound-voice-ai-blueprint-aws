"""The factor question is the ``.keypad`` twin
exactly when the slot is effectively keypad-only and the twin exists in the language."""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from lambdas.frontdoor.slotflow import _prompt_key
from owlbank.catalog import PromptCatalog
from owlbank.slot_policy import SlotDecl, SlotSettings, gather_input

DOC = json.loads((Path(__file__).resolve().parents[2] / "data" / "prompt_catalog.json")
                 .read_text(encoding="utf-8"))
CATALOG = PromptCatalog.from_dict(DOC)
BASES = ("factor.zip", "factor.ssn4", "factor.otp", "login.phone_number",
         "login.phone_number_retry", "factor.ack", "label.zip")   # the last two have no twin


@given(base=st.sampled_from(BASES), lang=st.sampled_from(("en-US", "es-US")),
       keypad_only=st.booleans(), is_factor=st.booleans(), accept_spoken=st.booleans(),
       via_base=st.booleans())
def test_p12_keypad_twin_iff_keypad_only_and_it_exists(base: str, lang: str, keypad_only: bool,
                                                       is_factor: bool, accept_spoken: bool,
                                                       via_base: bool) -> None:
    slot = SlotSettings(name="zip", decl=SlotDecl("digits", {"length": 5}, "label.zip",
                                                  None if via_base else base),
                        is_factor=is_factor, sensitive=True, keypad_only=keypad_only)
    effective = gather_input(slot, accept_spoken).keypad_only
    assert effective == (keypad_only or (is_factor and not accept_spoken))
    key = _prompt_key(slot, effective, CATALOG, lang, base if via_base else None)
    twin = f"{base}.keypad"
    assert key == (twin if effective and CATALOG.has(twin, lang) else base)
    assert CATALOG.has(key, lang)
