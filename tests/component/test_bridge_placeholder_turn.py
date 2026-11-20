"""The system-prompt sentence about placeholders.

The front door replaces a sensitive slot's words on the answering turn with ``[<slot name>]``
(``[ssn4]``). The bridge's system prompt carries one sentence, per language, telling the agent such
a token is an answer the caller gave that is withheld, which it must not ask for again or repeat.
The sentence is a constant in ``owlbank.agent_prompt`` appended to the catalog's ``agent.system`` text (not a
catalog key, so the bridge deploy doesn't depend on a reseed), on both the normal and the fallback
system-prompt paths. The caller's placeholder turn itself reaches the agent unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

from owlbank.agent_prompt import PLACEHOLDER_SENTENCE
from tests.component.test_bridge_fakes import make_rig, run, say, spoken, write_call_state

LANGS = ("en-US", "es-US")
UTTERANCE = "what's my balance? my last four are [ssn4]"


def test_both_languages_have_the_sentence() -> None:
    assert set(PLACEHOLDER_SENTENCE) == set(LANGS)
    for lang in LANGS:
        sentence = PLACEHOLDER_SENTENCE[lang]
        assert sentence.strip() == sentence and "\n" not in sentence
        assert "[ssn4]" in sentence  # the token form the agent will see
    assert PLACEHOLDER_SENTENCE["en-US"] != PLACEHOLDER_SENTENCE["es-US"]


def _two_turns(lang: str, *, fallback: bool, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, str, str]:
    rig = make_rig(script=[say("Your checking balance is twelve dollars."), say("Anything else?")])
    write_call_state(rig.store, language=lang)
    if fallback:
        async def broken(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("profile read failed")
        monkeypatch.setattr(rig.bridge, "_system_prompt", broken)

    async def go() -> tuple[str, str]:
        sim = rig.sim()
        await sim.connect()
        first = spoken(await sim.prompt(UTTERANCE))
        second = spoken(await sim.prompt("thanks"))
        await sim.hangup()
        return first, second

    first, second = run(go())
    return rig, first, second


@pytest.mark.parametrize("fallback", [False, True], ids=["normal", "fallback"])
@pytest.mark.parametrize("lang", LANGS)
def test_placeholder_turn_reaches_the_agent_unchanged_with_the_sentence(
        lang: str, fallback: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    rig, first, second = _two_turns(lang, fallback=fallback, monkeypatch=monkeypatch)
    assert first == "Your checking balance is twelve dollars."  # the agent's reply is spoken
    assert second == "Anything else?"

    one, two = rig.agent.calls
    assert one["user_text"] == UTTERANCE  # the placeholder passes through unchanged
    sentence = PLACEHOLDER_SENTENCE[lang]
    other = PLACEHOLDER_SENTENCE["es-US" if lang == "en-US" else "en-US"]
    for call in (one, two):
        assert call["system_prompt"].count(sentence) == 1
        assert other not in call["system_prompt"]
    # Appended, not substituted: the catalog's own text still opens the prompt.
    opening = "OwlBank's voice assistant" if lang == "en-US" else "asistente de voz de OwlBank"
    assert opening in one["system_prompt"].split(sentence)[0]
    # A stable base (the per-(customer, language) cache, the prompt cache, reasoning signatures on
    # tool-result hops): the clock line (set once per call, rebuilt on a new bank-local day or
    # language) is sent apart as the suffix.
    base = [c["system_prompt"].removesuffix("\n" + c["system_suffix"]) for c in (one, two)]
    assert base[0] == base[1]
    assert sentence in base[0]

