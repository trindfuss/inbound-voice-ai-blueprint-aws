"""Briefing properties: check_output, generate's wall-clock bound, and the facts."""

from __future__ import annotations

import time
import unicodedata
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank import briefing as b
from owlbank.callstate import CallState

_ALLOWED_TOOLS = b.known_tools()
_FACTORS = ("ani_match", "zip", "last4_ssn", "dob")


@given(st.text())
def test_p3a_check_output_never_unsafe(text: str) -> None:
    out = b.check_output(text)
    if out is not None:
        assert out
        assert not any(unicodedata.category(c).startswith("N") for c in out)
        assert not any(unicodedata.category(c).startswith("C") and c != "\t" for c in out)
        assert len(out.split()) <= b.MAX_WORDS
        assert not any(c in "<>&{}`\n\r" for c in out)


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs", "Nd", "Nl", "No", "Cc", "Cf", "Co", "Cn", "Zl", "Zp"),
                                      blacklist_characters="<>&{}`"), min_size=1),
       st.integers(1, 40))
def test_p3a_clean_text_passes_iff_short_enough(text: str, limit: int) -> None:
    out = b.check_output(text, max_words=limit)
    n = len(text.split())
    assert (out is not None) == (0 < n <= limit)


class _Model:
    def __init__(self, mode: str, payload: Any) -> None:
        self.mode, self.payload = mode, payload

    def summarise(self, facts: b.BriefingFacts, *, deadline: float) -> Any:
        if self.mode == "raise":
            raise ValueError("x")
        if self.mode == "sleep":
            time.sleep(0.6)
        return self.payload


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(mode=st.sampled_from(["ok", "raise", "none", "sleep", "garbage"]),
       payload=st.one_of(st.none(), st.text(max_size=200), st.integers()),
       budget=st.integers(0, 150))
def test_p4_generate_bounded_for_any_model(mode: str, payload: Any, budget: int) -> None:
    t0 = time.monotonic()
    out = b.generate(_Model(mode, payload), _facts(), budget_ms=budget)
    assert time.monotonic() - t0 < budget / 1000 + 0.4
    assert out.source in ("model", "fallback")
    if out.source == "fallback":
        assert out.summary is None and out.words == 0
    else:
        assert out.summary is not None and b.check_output(out.summary) == out.summary


def _facts() -> b.BriefingFacts:
    return b.BriefingFacts("en-US", 0, "unable_to_resolve", (), ())


@given(
    factors=st.lists(st.one_of(st.sampled_from(_FACTORS), st.text(max_size=20)), max_size=8),
    tools=st.lists(st.one_of(st.sampled_from(sorted(_ALLOWED_TOOLS)), st.text(max_size=20)), max_size=8),
    level=st.integers(0, 5),
    reason=st.text(max_size=20),
    language=st.sampled_from(["en-US", "es-US"]),
)
def test_p5_facts_hold_only_registry_identifiers(factors: list[str], tools: list[str], level: int,
                                                 reason: str, language: str) -> None:
    state = CallState(call_sid="CA1", expires_at=1, language=language, level=level,
                      factors_verified=tuple(factors))
    f = b.build_facts(state, {"toolsCalled": tools, "levelReached": level + 3}, reason=reason,
                      known_factors=_FACTORS)
    assert f.level == level  # the state's only; the journal's highest level is not used
    plain_f = {b.plain_words(n) for n in (*_FACTORS, "ani_match")}
    plain_t = {b.plain_words(n) for n in _ALLOWED_TOOLS}
    assert set(f.factors) <= plain_f
    assert set(f.tools) <= plain_t
    payload = str(b.facts_payload(f))
    stripped = payload.replace(language, "")
    assert not any(c.isdecimal() for c in stripped)
