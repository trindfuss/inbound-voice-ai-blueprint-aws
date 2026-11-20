"""Gather-shape properties over the rendered TwiML.

For any slot settings (``sensitive``, ``keypad_only``, ``barge_in`` on each of the four seed
slots), ``acceptSpokenFactors``, early finish on or off and either language, one call is driven
through every factor and the confirmation (``test_frontdoor_gather_input.scenario``) and each
``<Gather>`` response is checked:

* An effective keypad-only slot's Gather (parsed XML) has ``input="dtmf"`` and no
  ``speechTimeout``, ``speechModel``, ``hints`` or ``partialResultCallback``, and
  ``_early_eligible`` is false.
* ``acceptSpokenFactors=false`` makes every factor's Gather keypad-only, whatever its
  ``keypad_only``; the non-factor ``yes_no`` renders exactly as with ``acceptSpokenFactors=true``.
* ``barge_in: false`` plays the prompt verbs before a ``<Gather>`` that
  nests none, with the same attributes (timeouts, input, action) and the same reprompt as the
  ``barge_in: true`` rendering; ``barge_in: true`` (with ``keypad_only`` off) equals the golden
  rendering for the same slot.
"""

from __future__ import annotations

from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank.slot_policy import gather_input
from tests.component.test_frontdoor_gather_input import (
    FACTORS,
    LANGS,
    STEP_SLOT,
    Harness,
    assert_keypad_only,
    assert_path_b,
    golden,
    scenario,
)

SLOT_NAMES = (*FACTORS, "yes_no")

slot_fields = st.fixed_dictionaries({"sensitive": st.booleans(), "keypad_only": st.booleans(),
                                     "barge_in": st.booleans()})
all_slots = st.fixed_dictionaries(dict.fromkeys(SLOT_NAMES, slot_fields))
SETTINGS = settings(max_examples=100, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


def _with(slots: dict[str, dict[str, Any]], **fields: Any) -> dict[str, dict[str, Any]]:
    return {name: {**f, **fields} for name, f in slots.items()}


@SETTINGS
@given(slots=all_slots, asf=st.booleans(), early=st.booleans(), lang=st.sampled_from(list(LANGS)))
def test_keypad_only_gathers_take_no_speech(slots, asf, early, lang) -> None:
    holder: list[Harness] = []
    out = scenario(lang, asf, early, slots, harness=holder)
    app = holder[0].app
    for step, body in out.items():
        name = STEP_SLOT[step]
        setting = app.d.policy.slot(name)
        if gather_input(setting, asf).keypad_only:
            assert_keypad_only(body)
            assert not app._early_eligible(app._slot(name))  # noqa: SLF001
        else:
            assert 'input="dtmf speech"' in body


@SETTINGS
@given(slots=all_slots, early=st.booleans(), lang=st.sampled_from(list(LANGS)))
def test_spoken_factors_off_forces_every_factor_keypad_only(slots, early, lang) -> None:
    off = scenario(lang, False, early, slots)
    on = scenario(lang, True, early, slots)
    for step, body in off.items():
        if STEP_SLOT[step] in FACTORS:
            assert_keypad_only(body)
        else:
            assert body == on[step]  # the non-factor slot ignores the ceiling


@SETTINGS
@given(slots=all_slots, asf=st.booleans(), early=st.booleans(), lang=st.sampled_from(list(LANGS)))
def test_barge_in_false_moves_the_prompts_before_the_gather(slots, asf, early, lang) -> None:
    out = scenario(lang, asf, early, slots)
    reference = scenario(lang, asf, early, _with(slots, barge_in=True))
    assert out.keys() == reference.keys()
    for step, body in out.items():
        if slots[STEP_SLOT[step]]["barge_in"]:
            assert body == reference[step]
        else:
            assert_path_b(body, reference[step])


@SETTINGS
@given(slots=all_slots, asf=st.booleans(), early=st.booleans(), lang=st.sampled_from(list(LANGS)))
def test_barge_in_true_equals_the_golden_twiml(slots, asf, early, lang) -> None:
    """``sensitive`` doesn't touch the Gather; with ``keypad_only`` off and ``barge_in`` on the
    TwiML is today's (the golden files, captured before the slot settings existed)."""
    out = scenario(lang, asf, early, _with(slots, keypad_only=False, barge_in=True))
    for step, body in out.items():
        assert body == golden(lang, asf, early, step), step
