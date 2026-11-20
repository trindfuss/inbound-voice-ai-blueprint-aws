"""Splitting the system text into ``(base, clock_line)`` changes nothing
the model sees. ``join_system`` joins the parts with ``"\\n"``, and the result is the string the
bridge sent before the split (the prompt, a newline, the clock line). ``join_system`` and
:meth:`RenderedPrompt.joined` are checked against it.

The "before" string is written out here from the old formula, not taken from the bridge, so a
change to the bridge's join can't move both sides at once."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

from hypothesis import given
from hypothesis import strategies as st

from owlbank.agent_prompt import RenderedPrompt
from tac_bridge.agent_client import join_system
from tac_bridge.bridge import META_KEY, Bridge
from tac_bridge.clock import local_now_line, zone


# Printable text with newlines (and empty), as a catalog prompt or memory section could be.
base_text = st.text(st.characters(categories=("L", "M", "N", "P", "S", "Zs"), include_characters="\n\t"),
                    max_size=400)
langs = st.sampled_from(["en-US", "es-US"])
times = st.floats(min_value=0, max_value=4_102_444_800, allow_nan=False, allow_infinity=False)  # to 2100
zones = st.sampled_from(["America/New_York", "America/Los_Angeles", "UTC", "Asia/Kolkata", "", "Not/AZone"])


def old_with_clock(prompt: str, lang: str, now: float, tz: str) -> str:
    """The bridge's system text before the split."""
    return f"{prompt}\n{local_now_line(datetime.fromtimestamp(now, UTC), zone(tz), lang)}"


def bridge_stub(now: float, tz: str) -> Bridge:
    bridge = Bridge.__new__(Bridge)  # only the pieces the system prompt's cached path touches
    bridge._now = lambda: now
    bridge._cfg = SimpleNamespace(timezone=tz)
    return bridge


@given(base=base_text, lang=langs, now=times, tz=zones)
def test_joined_system_text_equals_the_old_joined_prompt(base: str, lang: str, now: float, tz: str) -> None:
    clock = local_now_line(datetime.fromtimestamp(now, UTC), zone(tz), lang)
    assert join_system(base, clock) == old_with_clock(base, lang, now, tz)


@given(base=base_text)
def test_no_suffix_leaves_the_prompt_unchanged(base: str) -> None:
    assert join_system(base, None) == base


per_call_text = st.text(st.characters(categories=("L", "M", "N", "P", "S", "Zs"), include_characters="\n<>"),
                       max_size=200)


@given(base=base_text.filter(bool), per_call=per_call_text, lang=langs, now=times, tz=zones)
def test_bridge_parts_joined_equal_what_the_bridge_sent(base: str, per_call: str, lang: str, now: float,
                                                           tz: str) -> None:
    """The cached path returns the static part per model key and the
    per-call part; the turn's suffix is ``per_call + "\n" + clock``, and joined as the client gives
    it to the model it is the old single prompt (static, then per-call) with the clock line."""
    bridge = bridge_stub(now, tz)
    view = SimpleNamespace(call_sid="CA1", verified=False, verified_customer_id=None, language=lang,
                           welcome_offer=None)
    cached = {"haiku-5-5": base}  # The static part per model key, cached per call with the per-call part
    context = SimpleNamespace(metadata={META_KEY: {"prompt_key": ["CA1", None, lang, None], "system_prompt": cached,
                                                   "per_call": per_call}})
    got, got_per_call, clock = asyncio.run(bridge._system_prompt(context, view))
    assert got == cached and got_per_call == per_call
    assert "\n" not in clock  # one line, after the cache point
    suffix = f"{got_per_call}\n{clock}"
    assert join_system(got["haiku-5-5"], suffix) == old_with_clock(f"{base}\n{per_call}", lang, now, tz)
    assert RenderedPrompt(static=base, per_call=per_call).joined(clock) == old_with_clock(f"{base}\n{per_call}",
                                                                                          lang, now, tz)
